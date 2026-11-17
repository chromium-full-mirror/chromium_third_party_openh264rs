// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! x86_64 SIMD implementations (SSE2, SSSE3, AVX2).

#![deny(unsafe_code)]

pub(super) use crate::simd::tokens::{sse2_token, ssse3_token};

/// Safe array/slice loads and stores shared by the x86_64 kernels, built entirely from
/// value-based SSE2/AVX2 intrinsics (`_mm_set_epi64x`, `_mm_cvtsi128_si64`, etc.) which
/// LLVM lowers to single `movdqu`/`movq`/`movd`/`vmovdqu` instructions without raw
/// pointers or `unsafe` blocks.
#[forbid(unsafe_code)]
#[allow(dead_code)]
pub(super) mod lanes {
    use core::arch::x86_64::*;

    #[target_feature(enable = "sse2")]
    #[inline]
    pub(super) fn ld16(r: &[u8]) -> __m128i {
        let &[
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
        ] = r.first_chunk::<16>().expect("16 bytes");
        _mm_set_epi64x(
            i64::from_ne_bytes([c0, c1, c2, c3, c4, c5, c6, c7]),
            i64::from_ne_bytes([b0, b1, b2, b3, b4, b5, b6, b7]),
        )
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub(super) fn ld8(r: &[u8]) -> __m128i {
        let a = r.first_chunk::<8>().expect("8 bytes");
        _mm_cvtsi64_si128(i64::from_ne_bytes(*a))
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub(super) fn ld4(r: &[u8]) -> __m128i {
        let a = r.first_chunk::<4>().expect("4 bytes");
        _mm_cvtsi32_si128(i32::from_ne_bytes(*a))
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub(super) fn to16(v: __m128i) -> [u8; 16] {
        let [b0, b1, b2, b3, b4, b5, b6, b7] = _mm_cvtsi128_si64(v).to_ne_bytes();
        let [c0, c1, c2, c3, c4, c5, c6, c7] =
            _mm_cvtsi128_si64(_mm_srli_si128::<8>(v)).to_ne_bytes();
        [
            b0, b1, b2, b3, b4, b5, b6, b7, c0, c1, c2, c3, c4, c5, c6, c7,
        ]
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub(super) fn st16(out: &mut [u8], v: __m128i) {
        *out.first_chunk_mut::<16>().expect("16 bytes") = to16(v);
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub(super) fn st8(out: &mut [u8], v: __m128i) {
        *out.first_chunk_mut::<8>().expect("8 bytes") = _mm_cvtsi128_si64(v).to_ne_bytes();
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub(super) fn st4(out: &mut [u8], v: __m128i) {
        *out.first_chunk_mut::<4>().expect("4 bytes") = _mm_cvtsi128_si32(v).to_ne_bytes();
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub(super) fn to8(v: __m128i) -> [u8; 8] {
        _mm_cvtsi128_si64(v).to_ne_bytes()
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub(super) fn to4(v: __m128i) -> [u8; 4] {
        _mm_cvtsi128_si32(v).to_ne_bytes()
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub(super) fn ld8_i16(r: &[i16]) -> __m128i {
        let a = r.first_chunk::<8>().expect("8 i16s");
        _mm_setr_epi16(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub(super) fn ld4_i16(r: &[i16]) -> __m128i {
        let a = r.first_chunk::<4>().expect("4 i16s");
        _mm_setr_epi16(a[0], a[1], a[2], a[3], 0, 0, 0, 0)
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub(super) fn st4_i16(out: &mut [i16], v: __m128i) {
        let a = out.first_chunk_mut::<4>().expect("4 i16s");
        let lo = (_mm_cvtsi128_si64(v) as u64).to_ne_bytes();
        a[0] = i16::from_ne_bytes([lo[0], lo[1]]);
        a[1] = i16::from_ne_bytes([lo[2], lo[3]]);
        a[2] = i16::from_ne_bytes([lo[4], lo[5]]);
        a[3] = i16::from_ne_bytes([lo[6], lo[7]]);
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub(super) fn st8_i16(out: &mut [i16], v: __m128i) {
        let a = out.first_chunk_mut::<8>().expect("8 i16s");
        let lo = (_mm_cvtsi128_si64(v) as u64).to_ne_bytes();
        let hi = (_mm_cvtsi128_si64(_mm_srli_si128::<8>(v)) as u64).to_ne_bytes();
        a[0] = i16::from_ne_bytes([lo[0], lo[1]]);
        a[1] = i16::from_ne_bytes([lo[2], lo[3]]);
        a[2] = i16::from_ne_bytes([lo[4], lo[5]]);
        a[3] = i16::from_ne_bytes([lo[6], lo[7]]);
        a[4] = i16::from_ne_bytes([hi[0], hi[1]]);
        a[5] = i16::from_ne_bytes([hi[2], hi[3]]);
        a[6] = i16::from_ne_bytes([hi[4], hi[5]]);
        a[7] = i16::from_ne_bytes([hi[6], hi[7]]);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) fn ld16_i16(r: &[i16]) -> __m256i {
        let a = r.first_chunk::<16>().expect("16 i16s");
        _mm256_set_m128i(ld8_i16(&a[8..]), ld8_i16(&a[..8]))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) fn st16_i16(out: &mut [i16], v: __m256i) {
        let a = out.first_chunk_mut::<16>().expect("16 i16s");
        st8_i16(&mut a[..8], _mm256_castsi256_si128(v));
        st8_i16(&mut a[8..], _mm256_extracti128_si256::<1>(v));
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub(super) fn ld8_u16(r: &[u16]) -> __m128i {
        let a = r.first_chunk::<8>().expect("8 u16s");
        _mm_setr_epi16(
            a[0] as i16,
            a[1] as i16,
            a[2] as i16,
            a[3] as i16,
            a[4] as i16,
            a[5] as i16,
            a[6] as i16,
            a[7] as i16,
        )
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub(super) fn st8_u16(out: &mut [u16], v: __m128i) {
        let a = out.first_chunk_mut::<8>().expect("8 u16s");
        let lo = (_mm_cvtsi128_si64(v) as u64).to_ne_bytes();
        let hi = (_mm_cvtsi128_si64(_mm_srli_si128::<8>(v)) as u64).to_ne_bytes();
        a[0] = u16::from_ne_bytes([lo[0], lo[1]]);
        a[1] = u16::from_ne_bytes([lo[2], lo[3]]);
        a[2] = u16::from_ne_bytes([lo[4], lo[5]]);
        a[3] = u16::from_ne_bytes([lo[6], lo[7]]);
        a[4] = u16::from_ne_bytes([hi[0], hi[1]]);
        a[5] = u16::from_ne_bytes([hi[2], hi[3]]);
        a[6] = u16::from_ne_bytes([hi[4], hi[5]]);
        a[7] = u16::from_ne_bytes([hi[6], hi[7]]);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) fn ld32_i8(r: &[i8; 32]) -> __m256i {
        let &[
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            d0,
            d1,
            d2,
            d3,
            d4,
            d5,
            d6,
            d7,
        ] = r;
        let lo = _mm_set_epi64x(
            i64::from_ne_bytes([
                b0 as u8, b1 as u8, b2 as u8, b3 as u8, b4 as u8, b5 as u8, b6 as u8, b7 as u8,
            ]),
            i64::from_ne_bytes([
                a0 as u8, a1 as u8, a2 as u8, a3 as u8, a4 as u8, a5 as u8, a6 as u8, a7 as u8,
            ]),
        );
        let hi = _mm_set_epi64x(
            i64::from_ne_bytes([
                d0 as u8, d1 as u8, d2 as u8, d3 as u8, d4 as u8, d5 as u8, d6 as u8, d7 as u8,
            ]),
            i64::from_ne_bytes([
                c0 as u8, c1 as u8, c2 as u8, c3 as u8, c4 as u8, c5 as u8, c6 as u8, c7 as u8,
            ]),
        );
        _mm256_set_m128i(hi, lo)
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) fn st32(out: &mut [u8], v: __m256i) {
        let [
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
        ] = to16(_mm256_castsi256_si128(v));
        let [
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            d0,
            d1,
            d2,
            d3,
            d4,
            d5,
            d6,
            d7,
        ] = to16(_mm256_extracti128_si256::<1>(v));
        *out.first_chunk_mut::<32>().expect("32 bytes") = [
            a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7, c0, c1, c2, c3, c4, c5,
            c6, c7, d0, d1, d2, d3, d4, d5, d6, d7,
        ];
    }
}

pub mod copy;
pub mod dct;
pub mod deblock;
pub mod intra_pred;
pub mod mc;
pub mod me;
pub mod quant;
pub mod sad;
pub mod satd;
pub mod score;
pub mod vaa;
