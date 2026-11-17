// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! SSE2, SSSE3, and AVX2 implementations of Quantization, Dequantization,
//! Hadamard Transforms, Non-Zero Coefficient Counting, and CAVLC Parameter Calculation.
#![deny(unsafe_code)]

use core::arch::x86_64::*;

use super::lanes::{ld4_i16, ld8, ld8_i16, ld8_u16, ld16_i16, st4_i16, st8_i16, st16_i16};
use super::{sse2_token, ssse3_token};
use crate::simd::{
    simd_entry,
    tokens::{Avx2Token, Sse2Token, Ssse3Token},
};

// ============================================================================
// CAVLC Lookup Tables (`codec/encoder/core/x86/coeff.asm`)
// ============================================================================

const Z: u8 = 0x80;

#[repr(C, align(16))]
struct CavlcLut {
    shufb: [[u8; 8]; 16],
    run: [u32; 16],
}

/// Combined SSSE3 `pshufb` compaction + 4-byte zero-run lookup table (`coeff.asm:60-97`,
/// permuted for unreversed 4-bit nibble index `bit 0..3` <-> `q[0..4] != 0`).
#[rustfmt::skip]
static CAVLC_LUT: CavlcLut = CavlcLut {
    shufb: [
        [Z, Z, Z, Z, Z, Z, Z, Z], // 0b0000: -
        [0, 1, Z, Z, Z, Z, Z, Z], // 0b0001: q[0]
        [2, 3, Z, Z, Z, Z, Z, Z], // 0b0010: q[1]
        [2, 3, 0, 1, Z, Z, Z, Z], // 0b0011: q[1], q[0]
        [4, 5, Z, Z, Z, Z, Z, Z], // 0b0100: q[2]
        [4, 5, 0, 1, Z, Z, Z, Z], // 0b0101: q[2], q[0]
        [4, 5, 2, 3, Z, Z, Z, Z], // 0b0110: q[2], q[1]
        [4, 5, 2, 3, 0, 1, Z, Z], // 0b0111: q[2], q[1], q[0]
        [6, 7, Z, Z, Z, Z, Z, Z], // 0b1000: q[3]
        [6, 7, 0, 1, Z, Z, Z, Z], // 0b1001: q[3], q[0]
        [6, 7, 2, 3, Z, Z, Z, Z], // 0b1010: q[3], q[1]
        [6, 7, 2, 3, 0, 1, Z, Z], // 0b1011: q[3], q[1], q[0]
        [6, 7, 4, 5, Z, Z, Z, Z], // 0b1100: q[3], q[2]
        [6, 7, 4, 5, 0, 1, Z, Z], // 0b1101: q[3], q[2], q[0]
        [6, 7, 4, 5, 2, 3, Z, Z], // 0b1110: q[3], q[2], q[1]
        [6, 7, 4, 5, 2, 3, 0, 1], // 0b1111: q[3], q[2], q[1], q[0]
    ],
    run: [
        u32::from_le_bytes([4, 0, 0, 0]), // 0b0000
        u32::from_le_bytes([3, 0, 0, 0]), // 0b0001
        u32::from_le_bytes([2, 1, 0, 0]), // 0b0010
        u32::from_le_bytes([2, 0, 0, 0]), // 0b0011
        u32::from_le_bytes([1, 2, 0, 0]), // 0b0100
        u32::from_le_bytes([1, 1, 0, 0]), // 0b0101
        u32::from_le_bytes([1, 0, 1, 0]), // 0b0110
        u32::from_le_bytes([1, 0, 0, 0]), // 0b0111
        u32::from_le_bytes([0, 3, 0, 0]), // 0b1000
        u32::from_le_bytes([0, 2, 0, 0]), // 0b1001
        u32::from_le_bytes([0, 1, 1, 0]), // 0b1010
        u32::from_le_bytes([0, 1, 0, 0]), // 0b1011
        u32::from_le_bytes([0, 0, 2, 0]), // 0b1100
        u32::from_le_bytes([0, 0, 1, 0]), // 0b1101
        u32::from_le_bytes([0, 0, 0, 1]), // 0b1110
        u32::from_le_bytes([0, 0, 0, 0]), // 0b1111
    ],
};

#[inline(always)]
fn nib_pop(nib: usize) -> usize {
    ((0x4332_3221_3221_2110u64 >> (nib * 4)) & 0xF) as usize
}

// ============================================================================
// Forward Quantization
// ============================================================================

/// Dead-zone quantization of 16 consecutive 16-bit coefficients, returning both
/// the signed quantized values and the unsigned magnitudes.
///
/// C++: `AVX2_Quant`, `codec/encoder/core/x86/quant.asm`.
#[target_feature(enable = "avx2")]
#[inline]
fn quant_16_avx2(v: __m256i, ff: __m256i, mf: __m256i) -> (__m256i, __m256i) {
    let abs = _mm256_abs_epi16(v);
    let sign_mask = _mm256_or_si256(v, _mm256_set1_epi16(0x7FFF));
    let q_mag = _mm256_mulhi_epu16(_mm256_adds_epu16(abs, ff), mf);
    let q_signed = _mm256_sign_epi16(q_mag, sign_mask);
    (q_signed, q_mag)
}

/// In-place dead-zone quantization of 8 consecutive 16-bit coefficients.
///
/// C++: `SSE2_Quant8`, `codec/encoder/core/x86/quant.asm`.
#[target_feature(enable = "sse2")]
#[inline]
fn quant_8(v: __m128i, ff: __m128i, mf: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let sign = _mm_cmpgt_epi16(zero, v); // 0xFFFF where v < 0, 0 where v >= 0
    let abs = _mm_sub_epi16(_mm_xor_si128(v, sign), sign);
    let abs_ff = _mm_adds_epu16(abs, ff);
    let q = _mm_mulhi_epu16(abs_ff, mf);
    _mm_sub_epi16(_mm_xor_si128(q, sign), sign)
}

/// In-place dead-zone quantization of 8 consecutive 16-bit coefficients,
/// returning both the signed quantized values and the un-signed magnitudes.
#[target_feature(enable = "sse2")]
#[inline]
fn quant_8_with_mag(v: __m128i, ff: __m128i, mf: __m128i) -> (__m128i, __m128i) {
    let zero = _mm_setzero_si128();
    let sign = _mm_cmpgt_epi16(zero, v);
    let abs = _mm_sub_epi16(_mm_xor_si128(v, sign), sign);
    let abs_ff = _mm_adds_epu16(abs, ff);
    let q_mag = _mm_mulhi_epu16(abs_ff, mf);
    let q_signed = _mm_sub_epi16(_mm_xor_si128(q_mag, sign), sign);
    (q_signed, q_mag)
}

/// Horizontal maximum of 8 unsigned 16-bit values in an XMM register.
#[target_feature(enable = "sse2")]
#[inline]
fn hmax_u16(m: __m128i) -> i16 {
    let m1 = _mm_shuffle_epi32(m, 0b01_00_11_10);
    let m2 = _mm_max_epi16(m, m1);
    let m3 = _mm_shufflelo_epi16(m2, 0b01_00_11_10);
    let m4 = _mm_max_epi16(m2, m3);
    let m5 = _mm_srli_epi32(m4, 16);
    let m6 = _mm_max_epi16(m4, m5);
    _mm_cvtsi128_si32(m6) as i16
}

/// Transposes four vectors of four `i32` lanes: lane `j` of result `k` becomes lane
/// `k` of input `j`.
#[target_feature(enable = "sse2")]
#[inline]
fn transpose4_epi32(
    v0: __m128i,
    v1: __m128i,
    v2: __m128i,
    v3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let a = _mm_unpacklo_epi32(v0, v1);
    let b = _mm_unpackhi_epi32(v0, v1);
    let c = _mm_unpacklo_epi32(v2, v3);
    let d = _mm_unpackhi_epi32(v2, v3);
    (
        _mm_unpacklo_epi64(a, c),
        _mm_unpackhi_epi64(a, c),
        _mm_unpacklo_epi64(b, d),
        _mm_unpackhi_epi64(b, d),
    )
}

/// Transposes four vectors whose **low four `i16` lanes** hold a row. The upper four
/// lanes of each result are the next row's data and are never read: the butterflies
/// are lane-wise and the stores are `_mm_storel_epi64`.
#[target_feature(enable = "sse2")]
#[inline]
fn transpose4_epi16_lo(
    v0: __m128i,
    v1: __m128i,
    v2: __m128i,
    v3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let a = _mm_unpacklo_epi16(v0, v1);
    let b = _mm_unpacklo_epi16(v2, v3);
    let lo = _mm_unpacklo_epi32(a, b);
    let hi = _mm_unpackhi_epi32(a, b);
    (lo, _mm_srli_si128(lo, 8), hi, _mm_srli_si128(hi, 8))
}

/// The inverse-Hadamard butterfly, which is the same in both passes.
///
/// `(a0, a1, a2, a3)` are the four taps of one line — a row in the first pass, a column
/// in the second — with one line per lane.
#[target_feature(enable = "sse2")]
#[inline]
fn ihadamard_butterfly(
    a0: __m128i,
    a1: __m128i,
    a2: __m128i,
    a3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let t0 = _mm_add_epi16(a0, a2);
    let t1 = _mm_sub_epi16(a0, a2);
    let t2 = _mm_sub_epi16(a1, a3);
    let t3 = _mm_add_epi16(a1, a3);
    (
        _mm_add_epi16(t0, t3),
        _mm_add_epi16(t1, t2),
        _mm_sub_epi16(t1, t2),
        _mm_sub_epi16(t0, t3),
    )
}

simd_entry! {
    #[target_feature(enable = "avx2")]
    fn quant_4x4_avx2(dct: &mut [i16; 16], ff: &[i16; 8], mf: &[i16; 8], _token: Avx2Token) {
        let vff = _mm256_broadcastsi128_si256(ld8_i16(ff));
        let vmf = _mm256_broadcastsi128_si256(ld8_i16(mf));
        let (q, _) = quant_16_avx2(ld16_i16(dct), vff, vmf);
        st16_i16(dct, q);
    }

    #[target_feature(enable = "avx2")]
    fn quant_4x4_dc_avx2(dct: &mut [i16; 16], ff: i16, mf: i16, _token: Avx2Token) {
        let vff = _mm256_set1_epi16(ff);
        let vmf = _mm256_set1_epi16(mf);
        let (q, _) = quant_16_avx2(ld16_i16(dct), vff, vmf);
        st16_i16(dct, q);
    }

    #[target_feature(enable = "avx2")]
    fn quant_four_4x4_avx2(dct: &mut [i16; 64], ff: &[i16; 8], mf: &[i16; 8], _token: Avx2Token) {
        let vff = _mm256_broadcastsi128_si256(ld8_i16(ff));
        let vmf = _mm256_broadcastsi128_si256(ld8_i16(mf));
        for block in dct.as_chunks_mut::<16>().0 {
            let (q, _) = quant_16_avx2(ld16_i16(block), vff, vmf);
            st16_i16(block, q);
        }
    }

    #[target_feature(enable = "avx2")]
    fn quant_four_4x4_max_avx2(
        dct: &mut [i16; 64],
        ff: &[i16; 8],
        mf: &[i16; 8],
        max: &mut [i16; 4],
        _token: Avx2Token,
    ) {
        let vff = _mm256_broadcastsi128_si256(ld8_i16(ff));
        let vmf = _mm256_broadcastsi128_si256(ld8_i16(mf));
        let mask_7fff = _mm256_set1_epi16(0x7FFF);

        let mut mags = [_mm256_setzero_si256(); 4];
        for (k, block) in dct.as_chunks_mut::<16>().0.iter_mut().enumerate() {
            let (q, mag) = quant_16_avx2(ld16_i16(block), vff, vmf);
            st16_i16(block, q);
            mags[k] = mag;
        }

        let lo01 = _mm256_permute2x128_si256::<0x20>(mags[0], mags[1]);
        let hi01 = _mm256_permute2x128_si256::<0x31>(mags[0], mags[1]);
        let max01 = _mm256_xor_si256(_mm256_max_epi16(lo01, hi01), mask_7fff);

        let lo23 = _mm256_permute2x128_si256::<0x20>(mags[2], mags[3]);
        let hi23 = _mm256_permute2x128_si256::<0x31>(mags[2], mags[3]);
        let max23 = _mm256_xor_si256(_mm256_max_epi16(lo23, hi23), mask_7fff);

        let min0 = _mm_minpos_epu16(_mm256_castsi256_si128(max01));
        let min1 = _mm_minpos_epu16(_mm256_extracti128_si256::<1>(max01));
        let min2 = _mm_minpos_epu16(_mm256_castsi256_si128(max23));
        let min3 = _mm_minpos_epu16(_mm256_extracti128_si256::<1>(max23));

        let p01 = _mm_unpacklo_epi16(min0, min1);
        let p23 = _mm_unpacklo_epi16(min2, min3);
        let p0123 = _mm_unpacklo_epi32(p01, p23);
        let res = _mm_xor_si128(p0123, _mm256_castsi256_si128(mask_7fff));
        st4_i16(max, res);
    }

    #[target_feature(enable = "sse2")]
    fn quant_4x4_sse2(dct: &mut [i16; 16], ff: &[i16; 8], mf: &[i16; 8], _token: Sse2Token) {
        let vff = ld8_i16(ff);
        let vmf = ld8_i16(mf);
        let v0 = ld8_i16(&dct[..8]);
        let v1 = ld8_i16(&dct[8..]);
        st8_i16(&mut dct[..8], quant_8(v0, vff, vmf));
        st8_i16(&mut dct[8..], quant_8(v1, vff, vmf));
    }

    #[target_feature(enable = "sse2")]
    fn quant_4x4_dc_sse2(dct: &mut [i16; 16], ff: i16, mf: i16, _token: Sse2Token) {
        let vff = _mm_set1_epi16(ff);
        let vmf = _mm_set1_epi16(mf);
        let v0 = ld8_i16(&dct[..8]);
        let v1 = ld8_i16(&dct[8..]);
        st8_i16(&mut dct[..8], quant_8(v0, vff, vmf));
        st8_i16(&mut dct[8..], quant_8(v1, vff, vmf));
    }

    #[target_feature(enable = "sse2")]
    fn quant_four_4x4_sse2(dct: &mut [i16; 64], ff: &[i16; 8], mf: &[i16; 8], _token: Sse2Token) {
        let vff = ld8_i16(ff);
        let vmf = ld8_i16(mf);
        for chunk in dct.as_chunks_mut::<8>().0 {
            let v = ld8_i16(chunk);
            st8_i16(chunk, quant_8(v, vff, vmf));
        }
    }

    #[target_feature(enable = "sse2")]
    fn quant_four_4x4_max_sse2(
        dct: &mut [i16; 64],
        ff: &[i16; 8],
        mf: &[i16; 8],
        max: &mut [i16; 4],
        _token: Sse2Token,
    ) {
        let vff = ld8_i16(ff);
        let vmf = ld8_i16(mf);
        for (k, block) in dct.as_chunks_mut::<16>().0.iter_mut().enumerate() {
            let v0 = ld8_i16(&block[..8]);
            let v1 = ld8_i16(&block[8..]);
            let (q0, mag0) = quant_8_with_mag(v0, vff, vmf);
            let (q1, mag1) = quant_8_with_mag(v1, vff, vmf);
            st8_i16(&mut block[..8], q0);
            st8_i16(&mut block[8..], q1);
            max[k] = hmax_u16(_mm_max_epi16(mag0, mag1));
        }
    }

    #[target_feature(enable = "sse2")]
    fn dequant_4x4_sse2(res: &mut [i16; 16], mf: &[u16; 8], _token: Sse2Token) {
        let vmf = ld8_u16(mf);
        let v0 = ld8_i16(&res[..8]);
        let v1 = ld8_i16(&res[8..]);
        st8_i16(&mut res[..8], _mm_mullo_epi16(v0, vmf));
        st8_i16(&mut res[8..], _mm_mullo_epi16(v1, vmf));
    }

    #[target_feature(enable = "sse2")]
    fn dequant_four_4x4_sse2(res: &mut [i16; 64], mf: &[u16; 8], _token: Sse2Token) {
        let vmf = ld8_u16(mf);
        for chunk in res.as_chunks_mut::<8>().0 {
            let v = ld8_i16(chunk);
            st8_i16(chunk, _mm_mullo_epi16(v, vmf));
        }
    }

    #[target_feature(enable = "sse2")]
    fn get_none_zero_count_sse2(level: &[i16; 16], _token: Sse2Token) -> i32 {
        let zero = _mm_setzero_si128();
        let v0 = ld8_i16(&level[..8]);
        let v1 = ld8_i16(&level[8..]);

        let eq0 = _mm_cmpeq_epi16(v0, zero);
        let eq1 = _mm_cmpeq_epi16(v1, zero);

        // packs saturates signed 16-bit to signed 8-bit.
        // 0xFFFF (-1) saturates to 0xFF (-1), 0x0000 (0) saturates to 0x00 (0).
        // Each zero coefficient produces one 0xFF byte; each non-zero produces 0x00.
        let eq = _mm_packs_epi16(eq0, eq1);
        let mask = _mm_movemask_epi8(eq) as u32;
        16 - mask.count_ones() as i32
    }

    #[target_feature(enable = "sse2")]
    fn hadamard_t4_dc_sse2(luma_dc: &mut [i16; 16], dct: &[i16; 241], _token: Sse2Token) {
        // Lane k = scalar row k. Within a row: A = dct[idx], B = dct[idx + 16],
        // C = dct[idx + 64], D = dct[idx + 80] — the scalar's d0, d16, d64, d80.
        let va = _mm_set_epi32(
            dct[160] as i32,
            dct[128] as i32,
            dct[32] as i32,
            dct[0] as i32,
        );
        let vb = _mm_set_epi32(
            dct[176] as i32,
            dct[144] as i32,
            dct[48] as i32,
            dct[16] as i32,
        );
        let vc = _mm_set_epi32(
            dct[224] as i32,
            dct[192] as i32,
            dct[96] as i32,
            dct[64] as i32,
        );
        let vd = _mm_set_epi32(
            dct[240] as i32,
            dct[208] as i32,
            dct[112] as i32,
            dct[80] as i32,
        );

        // Row pass. `pj` holds `p[4k + j]` in lane `k`.
        let s0 = _mm_add_epi32(va, vd);
        let s3 = _mm_sub_epi32(va, vd);
        let s1 = _mm_add_epi32(vb, vc);
        let s2 = _mm_sub_epi32(vb, vc);
        let p0 = _mm_add_epi32(s0, s1);
        let p1 = _mm_add_epi32(s3, s2);
        let p2 = _mm_sub_epi32(s0, s1);
        let p3 = _mm_sub_epi32(s3, s2);

        // `rk` now holds `p[4k + j]` in lane `j`, so the column pass is lane-wise.
        let (r0, r1, r2, r3) = transpose4_epi32(p0, p1, p2, p3);

        let one = _mm_set1_epi32(1);
        let t0 = _mm_add_epi32(r0, r3);
        let t3 = _mm_sub_epi32(r0, r3);
        let t1 = _mm_add_epi32(r1, r2);
        let t2 = _mm_sub_epi32(r1, r2);
        let o0 = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(t0, t1), one), 1);
        let o1 = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(t3, t2), one), 1);
        let o2 = _mm_srai_epi32(_mm_add_epi32(_mm_sub_epi32(t0, t1), one), 1);
        let o3 = _mm_srai_epi32(_mm_add_epi32(_mm_sub_epi32(t3, t2), one), 1);

        st8_i16(&mut luma_dc[..8], _mm_packs_epi32(o0, o1));
        st8_i16(&mut luma_dc[8..], _mm_packs_epi32(o2, o3));
    }

    #[target_feature(enable = "sse2")]
    fn dequant_ihadamard_4x4_sse2(res: &mut [i16; 16], mf: u16, _token: Sse2Token) {
        let r01 = ld8_i16(&res[..8]);
        let r23 = ld8_i16(&res[8..]);

        let r0 = r01;
        let r1 = _mm_srli_si128(r01, 8);
        let r2 = r23;
        let r3 = _mm_srli_si128(r23, 8);

        // `cm` holds `res[4k + m]` in lane `k`, so the row pass is lane-wise.
        let (c0, c1, c2, c3) = transpose4_epi16_lo(r0, r1, r2, r3);
        let (w0, w1, w2, w3) = ihadamard_butterfly(c0, c1, c2, c3);
        // Back to one row per vector, which is what the column pass wants lane-wise.
        let (x0, x1, x2, x3) = transpose4_epi16_lo(w0, w1, w2, w3);
        let (y0, y1, y2, y3) = ihadamard_butterfly(x0, x1, x2, x3);

        let out01 = _mm_unpacklo_epi64(y0, y1);
        let out23 = _mm_unpacklo_epi64(y2, y3);

        let mfv = _mm_set1_epi16(mf as i16);
        st8_i16(&mut res[..8], _mm_mullo_epi16(out01, mfv));
        st8_i16(&mut res[8..], _mm_mullo_epi16(out23, mfv));
    }

    #[target_feature(enable = "ssse3")]
    fn cavlc_param_cal_ssse3(
        pCoffLevel: &[i16],
        pRun: &mut [u8; 16],
        pLevel: &mut [i16; 16],
        pTotalCoeff: &mut i32,
        iEndIdx: i32,
        _token: Ssse3Token,
    ) -> i32 {
        if iEndIdx < 0 || pCoffLevel.is_empty() {
            *pTotalCoeff = 0;
            return 0;
        }
        let last_bound = (iEndIdx as usize).min(pCoffLevel.len() - 1).min(15);
        let valid_bits = (1u32 << (last_bound + 1)) - 1;
        let zero = _mm_setzero_si128();
        let lut = &CAVLC_LUT;

        if last_bound <= 3 && pCoffLevel.len() >= 4 {
            let v0 = ld4_i16(&pCoffLevel[..4]);
            let eq = _mm_cmpeq_epi8(_mm_packs_epi16(v0, zero), zero);
            let m = (!(_mm_movemask_epi8(eq) as u32)) & valid_bits;
            if m == 0 {
                *pTotalCoeff = 0;
                return 0;
            }
            let first_idx = 31 - m.leading_zeros();
            let low_idx = m.trailing_zeros();
            let nib = (m & 0xF) as usize;
            st4_i16(
                &mut pLevel[..4],
                _mm_shuffle_epi8(v0, ld8(&lut.shufb[nib])),
            );
            let out = nib_pop(nib);
            *pRun.first_chunk_mut::<4>().expect("4 bytes") = (lut.run[nib] >> 8).to_le_bytes();
            pRun[(out - 1) & 3] = low_idx as u8;
            *pTotalCoeff = out as i32;
            return (first_idx as i32 + 1) - (out as i32);
        }

        let Some(c16) = pCoffLevel.first_chunk::<16>() else {
            return crate::encoder::svc_set_mb_syn_cavlc::CavlcParamCal_c(
                pCoffLevel,
                pRun,
                pLevel,
                pTotalCoeff,
                iEndIdx,
            );
        };

        let v0 = ld8_i16(&c16[..8]);
        let v1 = ld8_i16(&c16[8..16]);
        let eq = _mm_cmpeq_epi8(_mm_packs_epi16(v0, v1), zero);
        let m = (!(_mm_movemask_epi8(eq) as u32)) & valid_bits;
        if m == 0 {
            *pTotalCoeff = 0;
            return 0;
        }

        // Zero-initialize `pRun` so implied 5th zero in the all-4-nonzero quadruple is 0.
        *pRun = [0u8; 16];

        let first_idx = 31 - m.leading_zeros();
        let low_idx = m.trailing_zeros();
        let first_q = (first_idx >> 2) as usize;
        let low_q = (low_idx >> 2) as usize;

        let quads = c16.as_chunks::<4>().0.first_chunk::<4>().expect("4 quads");

        // First quadruple containing a non-zero coefficient (`first_q`).
        let nib0 = ((m >> (first_q * 4)) & 0xF) as usize;
        let q_xmm0 = ld4_i16(&quads[first_q & 3]);
        st4_i16(
            &mut pLevel[..4],
            _mm_shuffle_epi8(q_xmm0, ld8(&lut.shufb[nib0])),
        );
        *pRun.first_chunk_mut::<4>().expect("4 bytes") = (lut.run[nib0] >> 8).to_le_bytes();
        let mut out = nib_pop(nib0);

        // Remaining quadruples down to `low_q`. Before each step `1 <= out <= 12`, so
        // `out.min(12) == out` and `(out - 1).min(12) == out - 1`, writing directly into
        // `pLevel` and `pRun` with zero stack buffers and zero bounds checks.
        let mut q = first_q;
        while q > low_q {
            q -= 1;
            let nib = ((m >> (q * 4)) & 0xF) as usize;
            let q_xmm = ld4_i16(&quads[q & 3]);
            let l_idx = out.min(12);
            st4_i16(
                &mut pLevel[l_idx..l_idx + 4],
                _mm_shuffle_epi8(q_xmm, ld8(&lut.shufb[nib])),
            );
            let r_idx = (out - 1).min(12);
            let r_val = (pRun[r_idx] as u32) + lut.run[nib];
            *pRun[r_idx..r_idx + 4]
                .first_chunk_mut::<4>()
                .expect("4 bytes") = r_val.to_le_bytes();
            out += nib_pop(nib);
        }

        pRun[(out - 1) & 15] = low_idx as u8;
        *pTotalCoeff = out as i32;
        (first_idx as i32 + 1) - (out as i32)
    }
}

#[cold]
#[inline(never)]
fn quant_4x4_fallback(dct: &mut [i16; 16], ff: &[i16; 8], mf: &[i16; 8]) {
    if let Some(t) = Avx2Token::try_new_enabled() {
        return quant_4x4_avx2(dct, ff, mf, t);
    }
    quant_4x4_sse2(dct, ff, mf, sse2_token());
}

#[cold]
#[inline(never)]
fn quant_4x4_dc_fallback(dct: &mut [i16; 16], ff: i16, mf: i16) {
    if let Some(t) = Avx2Token::try_new_enabled() {
        return quant_4x4_dc_avx2(dct, ff, mf, t);
    }
    quant_4x4_dc_sse2(dct, ff, mf, sse2_token());
}

#[cold]
#[inline(never)]
fn quant_four_4x4_fallback(dct: &mut [i16; 64], ff: &[i16; 8], mf: &[i16; 8]) {
    if let Some(t) = Avx2Token::try_new_enabled() {
        return quant_four_4x4_avx2(dct, ff, mf, t);
    }
    quant_four_4x4_sse2(dct, ff, mf, sse2_token());
}

#[cold]
#[inline(never)]
fn quant_four_4x4_max_fallback(
    dct: &mut [i16; 64],
    ff: &[i16; 8],
    mf: &[i16; 8],
    max: &mut [i16; 4],
) {
    if let Some(t) = Avx2Token::try_new_enabled() {
        return quant_four_4x4_max_avx2(dct, ff, mf, max, t);
    }
    quant_four_4x4_max_sse2(dct, ff, mf, max, sse2_token());
}

/// In-place dead-zone forward quantization of a 4x4 block using AVX2 or SSE2.
///
/// C++: `WelsQuant4x4_avx2` / `WelsQuant4x4_sse2`, `codec/encoder/core/x86/quant.asm`.
#[inline(always)]
pub fn quant_4x4(dct: &mut [i16; 16], ff: &[i16; 8], mf: &[i16; 8]) {
    if let Some(t) = Avx2Token::try_fast_enabled() {
        return quant_4x4_avx2(dct, ff, mf, t);
    }
    quant_4x4_fallback(dct, ff, mf);
}

/// In-place quantization of 16 Hadamard-transformed luma DC coefficients using AVX2 or SSE2.
///
/// C++: `WelsQuant4x4Dc_avx2` / `WelsQuant4x4Dc_sse2`, `codec/encoder/core/x86/quant.asm`.
#[inline(always)]
pub fn quant_4x4_dc(dct: &mut [i16; 16], ff: i16, mf: i16) {
    if let Some(t) = Avx2Token::try_fast_enabled() {
        return quant_4x4_dc_avx2(dct, ff, mf, t);
    }
    quant_4x4_dc_fallback(dct, ff, mf);
}

/// In-place dead-zone quantization of four consecutive 4x4 blocks using AVX2 or SSE2.
///
/// C++: `WelsQuantFour4x4_avx2` / `WelsQuantFour4x4_sse2`, `codec/encoder/core/x86/quant.asm`.
#[inline(always)]
pub fn quant_four_4x4(dct: &mut [i16; 64], ff: &[i16; 8], mf: &[i16; 8]) {
    if let Some(t) = Avx2Token::try_fast_enabled() {
        return quant_four_4x4_avx2(dct, ff, mf, t);
    }
    quant_four_4x4_fallback(dct, ff, mf);
}

/// In-place dead-zone quantization of four 4x4 blocks with early-termination max levels
/// using AVX2 (`vphminposuw`) or SSE2.
///
/// C++: `WelsQuantFour4x4Max_avx2` / `WelsQuantFour4x4Max_sse2`, `codec/encoder/core/x86/quant.asm`.
#[inline(always)]
pub fn quant_four_4x4_max(dct: &mut [i16; 64], ff: &[i16; 8], mf: &[i16; 8], max: &mut [i16; 4]) {
    if let Some(t) = Avx2Token::try_fast_enabled() {
        return quant_four_4x4_max_avx2(dct, ff, mf, max, t);
    }
    quant_four_4x4_max_fallback(dct, ff, mf, max);
}

// ============================================================================
// Dequantization
// ============================================================================

/// In-place dequantization of one 4x4 coefficient block using SSE2.
///
/// C++: `WelsDequant4x4_sse2`, `codec/encoder/core/x86/quant.asm`.
#[inline(always)]
pub fn dequant_4x4(res: &mut [i16; 16], mf: &[u16; 8]) {
    dequant_4x4_sse2(res, mf, sse2_token());
}

/// In-place dequantization of four consecutive 4x4 coefficient blocks using SSE2.
///
/// C++: `WelsDequantFour4x4_sse2`, `codec/encoder/core/x86/quant.asm`.
#[inline(always)]
pub fn dequant_four_4x4(res: &mut [i16; 64], mf: &[u16; 8]) {
    dequant_four_4x4_sse2(res, mf, sse2_token());
}

// ============================================================================
// Non-Zero Count
// ============================================================================

/// Count of non-zero coefficients in a 16-element level array using SSE2.
///
/// C++: `WelsGetNoneZeroCount_sse2`, `codec/encoder/core/x86/score.asm`.
#[inline(always)]
pub fn get_none_zero_count(level: &[i16; 16]) -> i32 {
    get_none_zero_count_sse2(level, sse2_token())
}

// ============================================================================
// Hadamard Transforms of the Luma DC Block (SSE2)
// ============================================================================

/// 4x4 forward Hadamard transform of 16 luma DC coefficients using SSE2.
///
/// C++: `WelsHadamardT4Dc_sse2`, `codec/encoder/core/x86/dct.asm`.
#[inline(always)]
pub fn hadamard_t4_dc(luma_dc: &mut [i16; 16], dct: &[i16; 241]) {
    hadamard_t4_dc_sse2(luma_dc, dct, sse2_token());
}

/// In-place dequantization and inverse 4x4 Hadamard transform of the luma DC block.
///
/// Upstream asm: `WelsDequantIHadamard4x4_sse2` in `codec/encoder/core/x86/quant.asm`.
#[inline(always)]
pub fn dequant_ihadamard_4x4(res: &mut [i16; 16], mf: u16) {
    dequant_ihadamard_4x4_sse2(res, mf, sse2_token());
}

/// Calculates non-zero count, level, run, and total zero statistics for CAVLC transform blocks
/// using SSSE3 4-bit nibble `pshufb` compaction and 4-byte packed run-LUT accumulation.
///
/// Only the first `*pTotalCoeff` entries of `pLevel` and `pRun` are populated: when
/// non-zero coefficients are present, `pRun` is zero-initialized (or overwritten in
/// 4-byte chunks on the `iEndIdx <= 3` path) and `pLevel` is written in 4-element
/// quads, clobbering slots past `*pTotalCoeff` up to the quad boundary.
///
/// C++: `CavlcParamCal_sse42`, `codec/encoder/core/x86/coeff.asm`.
#[inline(always)]
pub fn cavlc_param_cal(
    pCoffLevel: &[i16],
    pRun: &mut [u8; 16],
    pLevel: &mut [i16; 16],
    pTotalCoeff: &mut i32,
    iEndIdx: i32,
) -> i32 {
    cavlc_param_cal_ssse3(
        pCoffLevel,
        pRun,
        pLevel,
        pTotalCoeff,
        iEndIdx,
        ssse3_token(),
    )
}

// ============================================================================
// Unit Tests & Parity
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::decode_mb_aux::{
        dequant_4x4 as scalar_dequant_4x4, dequant_four_4x4 as scalar_dequant_four_4x4,
        dequant_ihadamard_4x4 as scalar_dequant_ihadamard_4x4,
    };
    use crate::encoder::encode_mb_aux::{
        G_KI_QUANT_INTER_FF, g_kiQuantMF, get_none_zero_count as scalar_get_none_zero_count,
        hadamard_t4_dc as scalar_hadamard_t4_dc, quant_4x4 as scalar_quant_4x4,
        quant_4x4_dc as scalar_quant_4x4_dc, quant_four_4x4 as scalar_quant_four_4x4,
        quant_four_4x4_max as scalar_quant_four_4x4_max,
    };

    fn lcg(seed: &mut u64) -> i16 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 32) as i32 % 4000 - 2000) as i16
    }

    #[test]
    fn test_quant_4x4_parity() {
        let mut seed = 42u64;
        let ff = G_KI_QUANT_INTER_FF.0[16];
        let mf = g_kiQuantMF[16];

        for _ in 0..100 {
            let mut block_c = [0i16; 16];
            for v in block_c.iter_mut() {
                *v = lcg(&mut seed);
            }
            let mut block_simd = block_c;

            scalar_quant_4x4(&mut block_c, &ff, &mf);
            quant_4x4(&mut block_simd, &ff, &mf);

            assert_eq!(block_simd, block_c);
        }
    }

    #[test]
    fn test_quant_4x4_dc_parity() {
        let mut seed = 123u64;
        let ff = 17i16;
        let mf = 26i16;

        for _ in 0..100 {
            let mut block_c = [0i16; 16];
            for v in block_c.iter_mut() {
                *v = lcg(&mut seed);
            }
            let mut block_simd = block_c;

            scalar_quant_4x4_dc(&mut block_c, ff, mf);
            quant_4x4_dc(&mut block_simd, ff, mf);

            assert_eq!(block_simd, block_c);
        }
    }

    #[test]
    fn test_quant_four_4x4_parity() {
        let mut seed = 999u64;
        let ff = G_KI_QUANT_INTER_FF.0[22];
        let mf = g_kiQuantMF[22];

        for _ in 0..50 {
            let mut block_c = [0i16; 64];
            for v in block_c.iter_mut() {
                *v = lcg(&mut seed);
            }
            let mut block_simd = block_c;

            scalar_quant_four_4x4(&mut block_c, &ff, &mf);
            quant_four_4x4(&mut block_simd, &ff, &mf);

            assert_eq!(block_simd, block_c);
        }
    }

    #[test]
    fn test_quant_four_4x4_max_parity() {
        let mut seed = 7777u64;
        let ff = G_KI_QUANT_INTER_FF.0[28];
        let mf = g_kiQuantMF[28];

        for _ in 0..50 {
            let mut block_c = [0i16; 64];
            for v in block_c.iter_mut() {
                *v = lcg(&mut seed);
            }
            let mut block_simd = block_c;

            let mut max_c = [0i16; 4];
            let mut max_simd = [0i16; 4];

            scalar_quant_four_4x4_max(&mut block_c, &ff, &mf, &mut max_c);
            quant_four_4x4_max(&mut block_simd, &ff, &mf, &mut max_simd);

            assert_eq!(block_simd, block_c);
            assert_eq!(max_simd, max_c);
        }
    }

    #[test]
    fn test_dequant_4x4_parity() {
        let mut seed = 8888u64;
        let mf: [u16; 8] = [10, 13, 16, 13, 10, 13, 16, 13];

        for _ in 0..100 {
            let mut block_c = [0i16; 16];
            for v in block_c.iter_mut() {
                *v = lcg(&mut seed);
            }
            let mut block_simd = block_c;

            scalar_dequant_4x4(&mut block_c, &mf);
            dequant_4x4(&mut block_simd, &mf);

            assert_eq!(block_simd, block_c);
        }
    }

    #[test]
    fn test_dequant_four_4x4_parity() {
        let mut seed = 5555u64;
        let mf: [u16; 8] = [10, 13, 16, 13, 10, 13, 16, 13];

        for _ in 0..50 {
            let mut block_c = [0i16; 64];
            for v in block_c.iter_mut() {
                *v = lcg(&mut seed);
            }
            let mut block_simd = block_c;

            scalar_dequant_four_4x4(&mut block_c, &mf);
            dequant_four_4x4(&mut block_simd, &mf);

            assert_eq!(block_simd, block_c);
        }
    }

    #[test]
    fn test_get_none_zero_count_parity() {
        let mut seed = 3333u64;
        for _ in 0..100 {
            let mut block = [0i16; 16];
            for v in block.iter_mut() {
                let r = lcg(&mut seed);
                *v = if r % 3 == 0 { 0 } else { r };
            }
            let c_count = scalar_get_none_zero_count(&block);
            let simd_count = get_none_zero_count(&block);
            assert_eq!(simd_count, c_count);
        }
    }

    // ========================================================================
    // The two luma-DC Hadamard kernels.
    //
    // Both sweep the full `i16` input range: the ihadamard's overflow is observable
    // output, and `hadamard_t4_dc`'s clamp is only reachable from large inputs.
    // ========================================================================

    fn lcg_full_i16(seed: &mut u64) -> i16 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 32) as u16 as i16
    }

    #[test]
    fn hadamard_t4_dc_parity() {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..2000 {
            let mut dct = [0i16; 241];
            for v in dct.iter_mut() {
                *v = lcg_full_i16(&mut seed);
            }
            let mut want = [0i16; 16];
            let mut got = [0i16; 16];
            scalar_hadamard_t4_dc(&mut want, &dct);
            hadamard_t4_dc(&mut got, &dct);
            assert_eq!(got, want);
        }
    }

    /// Drives the saturating store: all sixteen DC coefficients at `i16::MIN`/`MAX`
    /// puts every output past the clamp in both directions.
    #[test]
    fn hadamard_t4_dc_saturates_like_the_scalar() {
        const DC_IDX: [usize; 16] = [
            0, 16, 64, 80, 32, 48, 96, 112, 128, 144, 192, 208, 160, 176, 224, 240,
        ];
        // Every assignment of the two extremes across the sixteen DC positions:
        // 65536 sign patterns, driving the clamp both ways.
        for pattern in 0u32..(1 << 16) {
            let mut dct = [0i16; 241];
            for (n, &idx) in DC_IDX.iter().enumerate() {
                dct[idx] = if pattern & (1 << n) != 0 {
                    i16::MAX
                } else {
                    i16::MIN
                };
            }
            let mut want = [0i16; 16];
            let mut got = [0i16; 16];
            scalar_hadamard_t4_dc(&mut want, &dct);
            hadamard_t4_dc(&mut got, &dct);
            assert_eq!(got, want, "pattern {pattern:#018b}");
        }

        // A check that the clamp really is reached.
        let mut dct = [0i16; 241];
        for &idx in &DC_IDX {
            dct[idx] = i16::MAX;
        }
        let mut want = [0i16; 16];
        scalar_hadamard_t4_dc(&mut want, &dct);
        assert_eq!(
            want[0],
            i16::MAX,
            "the all-MAX case should saturate the DC output"
        );
    }

    #[test]
    fn dequant_ihadamard_4x4_parity() {
        let mut seed = 0x8A5C_D789_635D_2DFFu64;
        // Every `mf` the dequant tables can produce, plus the ends of the range: the
        // multiply wraps, so a value near `u16::MAX` is a different test from a small one.
        let mfs: Vec<u16> = (0..6)
            .map(|r| crate::encoder::svc_encode_mb::g_kuiDequantCoeff[r][0])
            .chain([0u16, 1, 0x7FFF, 0x8000, u16::MAX])
            .collect();

        for &mf in &mfs {
            for _ in 0..40 {
                let mut want = [0i16; 16];
                for v in want.iter_mut() {
                    *v = lcg_full_i16(&mut seed);
                }
                let mut got = want;
                scalar_dequant_ihadamard_4x4(&mut want, mf);
                dequant_ihadamard_4x4(&mut got, mf);
                assert_eq!(got, want, "mf = {mf}");
            }
        }
    }

    /// Pins that the SSE2 kernel wraps rather than saturates, with inputs chosen to
    /// overflow every intermediate.
    #[test]
    fn dequant_ihadamard_4x4_wraps_like_the_scalar() {
        for &v in &[i16::MIN, i16::MAX, -1, 1] {
            for &mf in &[1u16, 2, 0x8000, u16::MAX] {
                let mut want = [v; 16];
                let mut got = [v; 16];
                scalar_dequant_ihadamard_4x4(&mut want, mf);
                dequant_ihadamard_4x4(&mut got, mf);
                assert_eq!(got, want, "v = {v}, mf = {mf}");
            }
        }
    }

    #[test]
    fn cavlc_param_cal_all_65536_masks_and_end_indices() {
        use crate::encoder::svc_set_mb_syn_cavlc::CavlcParamCal_c;

        // Distinct non-zero values at each index (including multiples of 256 and extremes)
        // so any shuffle/order or narrow-truncation bug is caught on every mask.
        const VALS: [i16; 16] = [
            1,
            -1,
            256,
            -256,
            i16::MAX,
            i16::MIN,
            512,
            -512,
            3,
            -7,
            1024,
            -2048,
            0x7F00,
            -32767,
            42,
            -999,
        ];

        // 1. All 2^16 non-zero bitmask patterns for iEndIdx = 15.
        for mask in 0u32..=0xFFFF {
            let block: [i16; 16] =
                core::array::from_fn(|i| if (mask >> i) & 1 != 0 { VALS[i] } else { 0 });

            let (mut run_c, mut run_s) = ([0xAAu8; 16], [0x55u8; 16]);
            let (mut lvl_c, mut lvl_s) = ([0x1234i16; 16], [-0x4321i16; 16]);
            let (mut tc_c, mut tc_s) = (-1i32, -2i32);

            let tz_c = CavlcParamCal_c(&block, &mut run_c, &mut lvl_c, &mut tc_c, 15);
            let tz_s = cavlc_param_cal(&block, &mut run_s, &mut lvl_s, &mut tc_s, 15);

            assert_eq!(tz_s, tz_c, "tz mismatch mask={mask:#06x}");
            assert_eq!(tc_s, tc_c, "tc mismatch mask={mask:#06x}");
            let n = tc_c as usize;
            assert_eq!(&lvl_s[..n], &lvl_c[..n], "lvl mismatch mask={mask:#06x}");
            assert_eq!(&run_s[..n], &run_c[..n], "run mismatch mask={mask:#06x}");
        }

        // 2. Intra16x16 AC (`iEndIdx = 14`) across all 2^15 masks with non-zero garbage at [15].
        for mask in 0u32..(1 << 15) {
            let mut block: [i16; 16] =
                core::array::from_fn(|i| if (mask >> i) & 1 != 0 { VALS[i] } else { 0 });
            block[15] = if mask & 1 == 0 { i16::MIN } else { 7777 };

            let (mut run_c, mut run_s) = ([0u8; 16], [0u8; 16]);
            let (mut lvl_c, mut lvl_s) = ([0i16; 16], [0i16; 16]);
            let (mut tc_c, mut tc_s) = (0i32, 0i32);

            let tz_c = CavlcParamCal_c(&block, &mut run_c, &mut lvl_c, &mut tc_c, 14);
            let tz_s = cavlc_param_cal(&block, &mut run_s, &mut lvl_s, &mut tc_s, 14);

            assert_eq!(tz_s, tz_c, "iEndIdx=14 tz mask={mask:#06x}");
            assert_eq!(tc_s, tc_c, "iEndIdx=14 tc mask={mask:#06x}");
            let n = tc_c as usize;
            assert_eq!(&lvl_s[..n], &lvl_c[..n], "iEndIdx=14 lvl mask={mask:#06x}");
            assert_eq!(&run_s[..n], &run_c[..n], "iEndIdx=14 run mask={mask:#06x}");
        }

        // 3. Edge `iEndIdx` values (-1, 0, 1, 2, 3, 7, 11) with non-zero garbage past `iEndIdx`,
        //    plus short slices exercising the 4-element Chroma DC path and scalar fallback.
        for &end_idx in &[-1i32, 0, 1, 2, 3, 7, 11] {
            for mask in 0u32..256 {
                let mut block = [i16::MIN; 16];
                for i in 0..16 {
                    if (i as i32) <= end_idx {
                        block[i] = if (mask >> (i & 7)) & 1 != 0 {
                            VALS[i]
                        } else {
                            0
                        };
                    }
                }
                for &slice_len in &[0usize, 1, 3, 4, 5, 12, 15, 16] {
                    let sub = &block[..slice_len];
                    let (mut run_c, mut run_s) = ([0u8; 16], [0u8; 16]);
                    let (mut lvl_c, mut lvl_s) = ([0i16; 16], [0i16; 16]);
                    let (mut tc_c, mut tc_s) = (99i32, 88i32);

                    let tz_c = CavlcParamCal_c(sub, &mut run_c, &mut lvl_c, &mut tc_c, end_idx);
                    let tz_s = cavlc_param_cal(sub, &mut run_s, &mut lvl_s, &mut tc_s, end_idx);

                    assert_eq!(tz_s, tz_c, "end={end_idx} len={slice_len} mask={mask}");
                    assert_eq!(tc_s, tc_c, "end={end_idx} len={slice_len} mask={mask}");
                    let n = tc_c as usize;
                    assert_eq!(&lvl_s[..n], &lvl_c[..n]);
                    assert_eq!(&run_s[..n], &run_c[..n]);
                }
            }
        }
    }

    #[test]
    fn avx2_and_sse2_quant_full_i16_range_and_all_qps() {
        let avx2 = Avx2Token::try_new();
        if avx2.is_none() {
            eprintln!("NOTE: host CPU lacks AVX2; testing SSE2 and public quant entry points only");
        }
        let sse2 = sse2_token();

        // Sweep every i16 value (-32768..=32767) in 1024 blocks of 64 across all 52 QPs
        // (cycling inter and intra FF rows so every QP table entry is exercised).
        for qp in 0..52usize {
            let mf = g_kiQuantMF[qp];
            for &ff in &[G_KI_QUANT_INTER_FF.0[qp], G_KI_QUANT_INTER_FF.0[qp + 6]] {
                // Edge-case 16-lane and 64-lane blocks at every (qp, intra/inter) pair.
                let edge16: [i16; 16] = [
                    i16::MIN,
                    i16::MIN + 1,
                    -16384,
                    -1000,
                    -256,
                    -2,
                    -1,
                    0,
                    1,
                    2,
                    255,
                    256,
                    1000,
                    16384,
                    i16::MAX - 1,
                    i16::MAX,
                ];

                let mut want16 = edge16;
                scalar_quant_4x4(&mut want16, &ff, &mf);
                let mut got_pub = edge16;
                quant_4x4(&mut got_pub, &ff, &mf);
                assert_eq!(got_pub, want16, "quant_4x4 pub qp={qp}");
                let mut got_sse2 = edge16;
                quant_4x4_sse2(&mut got_sse2, &ff, &mf, sse2);
                assert_eq!(got_sse2, want16, "quant_4x4 sse2 qp={qp}");
                if let Some(t) = avx2 {
                    let mut got_avx2 = edge16;
                    quant_4x4_avx2(&mut got_avx2, &ff, &mf, t);
                    assert_eq!(got_avx2, want16, "quant_4x4 avx2 qp={qp}");
                }

                let (dc_ff, dc_mf) = (ff[0] << 1, mf[0] >> 1);
                let mut want_dc = edge16;
                scalar_quant_4x4_dc(&mut want_dc, dc_ff, dc_mf);
                let mut got_dc_pub = edge16;
                quant_4x4_dc(&mut got_dc_pub, dc_ff, dc_mf);
                assert_eq!(got_dc_pub, want_dc, "quant_4x4_dc pub qp={qp}");
                if let Some(t) = avx2 {
                    let mut got_dc_avx2 = edge16;
                    quant_4x4_dc_avx2(&mut got_dc_avx2, dc_ff, dc_mf, t);
                    assert_eq!(got_dc_avx2, want_dc, "quant_4x4_dc avx2 qp={qp}");
                }
            }

            // Full -32768..=32767 sweep across all 52 QPs.
            let ff = G_KI_QUANT_INTER_FF.0[qp + (qp & 1) * 6];
            let (dc_ff, dc_mf) = (ff[0] << 1, mf[0] >> 1);
            for chunk_idx in 0..1024usize {
                let base = (chunk_idx * 64) as i32 + (i16::MIN as i32);
                let blk64: [i16; 64] = core::array::from_fn(|k| (base + k as i32) as i16);

                let mut want64 = blk64;
                let mut want_max = [0i16; 4];
                scalar_quant_four_4x4_max(&mut want64, &ff, &mf, &mut want_max);

                if let Some(t) = avx2 {
                    let mut got_max_blk = blk64;
                    let mut got_max = [0i16; 4];
                    quant_four_4x4_max_avx2(&mut got_max_blk, &ff, &mf, &mut got_max, t);
                    assert_eq!(got_max_blk, want64, "four_4x4_max avx2 dct qp={qp}");
                    assert_eq!(got_max, want_max, "four_4x4_max avx2 max qp={qp}");

                    let mut got_four = blk64;
                    quant_four_4x4_avx2(&mut got_four, &ff, &mf, t);
                    assert_eq!(got_four, want64, "four_4x4 avx2 qp={qp}");

                    // Also exercise 16-lane AVX2 quant_4x4 and quant_4x4_dc over every lane
                    // position mod 8 across the full i16 range on boundary/stride QPs.
                    if qp == 0 || qp == 22 || qp == 51 {
                        for sub in 0..4 {
                            let b16: [i16; 16] =
                                *blk64[sub * 16..sub * 16 + 16].first_chunk::<16>().unwrap();
                            let mut g4 = b16;
                            quant_4x4_avx2(&mut g4, &ff, &mf, t);
                            assert_eq!(
                                g4,
                                want64[sub * 16..sub * 16 + 16],
                                "4x4 avx2 full range qp={qp}"
                            );

                            let mut want_dc = b16;
                            let mut got_dc = b16;
                            scalar_quant_4x4_dc(&mut want_dc, dc_ff, dc_mf);
                            quant_4x4_dc_avx2(&mut got_dc, dc_ff, dc_mf, t);
                            assert_eq!(got_dc, want_dc, "4x4_dc avx2 full range qp={qp}");
                        }
                    }
                } else {
                    let mut got_pub = blk64;
                    let mut got_max = [0i16; 4];
                    quant_four_4x4_max(&mut got_pub, &ff, &mf, &mut got_max);
                    assert_eq!(got_pub, want64);
                    assert_eq!(got_max, want_max);
                }
            }
        }
    }
}
