// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! SSE2 implementations of Forward 4x4 DCT and Inverse DCT (IDCT) with Prediction Addition.
#![deny(unsafe_code)]

use super::lanes::{ld8_i16, ld16, st8_i16, st16, to16};
use super::sse2_token;
use crate::encoder::rec_view::RecCursor;
use crate::safe::plane::{BlockRows, PlaneCursor, PlaneCursorMut, RefSamples, SampleCursor};
use crate::simd::{simd_entry, tokens::Sse2Token};
use core::arch::x86_64::*;

// ============================================================================
// Forward 4x4 Integer DCT
// ============================================================================

/// Butterfly stage for 4-point 1D integer DCT on 16-bit lanes.
///
/// Computes:
///   y0 = s0 + s1
///   y1 = (s3 << 1) + s2
///   y2 = s0 - s1
///   y3 = s3 - (s2 << 1)
/// where:
///   s0 = d0 + d3, s3 = d0 - d3
///   s1 = d1 + d2, s2 = d1 - d2
///
/// Operates in parallel across all lanes of `__m128i`.
#[target_feature(enable = "sse2")]
#[inline]
fn dct_pass(
    d0: __m128i,
    d1: __m128i,
    d2: __m128i,
    d3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let s0 = _mm_add_epi16(d0, d3);
    let s3 = _mm_sub_epi16(d0, d3);
    let s1 = _mm_add_epi16(d1, d2);
    let s2 = _mm_sub_epi16(d1, d2);

    let y0 = _mm_add_epi16(s0, s1);
    let y1 = _mm_add_epi16(_mm_slli_epi16(s3, 1), s2);
    let y2 = _mm_sub_epi16(s0, s1);
    let y3 = _mm_sub_epi16(s3, _mm_slli_epi16(s2, 1));

    (y0, y1, y2, y3)
}

/// In-register transpose of a 4x4 matrix of 16-bit integers stored in the lower 64 bits of 4 registers.
#[target_feature(enable = "sse2")]
#[inline]
fn transpose4(
    v0: __m128i,
    v1: __m128i,
    v2: __m128i,
    v3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let t0 = _mm_unpacklo_epi16(v0, v1);
    let t1 = _mm_unpacklo_epi16(v2, v3);
    let u0 = _mm_unpacklo_epi32(t0, t1);
    let u1 = _mm_unpackhi_epi32(t0, t1);

    let c0 = u0;
    let c1 = _mm_srli_si128(u0, 8);
    let c2 = u1;
    let c3 = _mm_srli_si128(u1, 8);

    (c0, c1, c2, c3)
}

/// In-register transpose of two 4x4 blocks stored side-by-side in four 8-word registers.
#[target_feature(enable = "sse2")]
#[inline]
fn transpose8(
    v0: __m128i,
    v1: __m128i,
    v2: __m128i,
    v3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let t0 = _mm_unpacklo_epi16(v0, v1);
    let t1 = _mm_unpackhi_epi16(v0, v1);
    let t2 = _mm_unpacklo_epi16(v2, v3);
    let t3 = _mm_unpackhi_epi16(v2, v3);

    let u0 = _mm_unpacklo_epi32(t0, t2);
    let u1 = _mm_unpackhi_epi32(t0, t2);
    let u2 = _mm_unpacklo_epi32(t1, t3);
    let u3 = _mm_unpackhi_epi32(t1, t3);

    let c0 = _mm_unpacklo_epi64(u0, u2);
    let c1 = _mm_unpackhi_epi64(u0, u2);
    let c2 = _mm_unpacklo_epi64(u1, u3);
    let c3 = _mm_unpackhi_epi64(u1, u3);

    (c0, c1, c2, c3)
}

/// Transforms two horizontally adjacent 4x4 blocks side-by-side using 8-wide SSE2 SIMD
/// over an already bounds-checked `BlockRows` span.
macro_rules! dct_two_4x4_from_span {
    ($out:expr, $s1:expr, $s2:expr, $y0:expr) => {{
        let (out, s1, s2, y0) = ($out, $s1, $s2, $y0);
        let zero = _mm_setzero_si128();
        let r1_0 = s1.row::<8>(y0, 0);
        let r2_0 = s2.row::<8>(y0, 0);
        let diff0 = _mm_sub_epi16(
            _mm_unpacklo_epi8(_mm_cvtsi64_si128(i64::from_ne_bytes(r1_0)), zero),
            _mm_unpacklo_epi8(_mm_cvtsi64_si128(i64::from_ne_bytes(r2_0)), zero),
        );

        let r1_1 = s1.row::<8>(y0 + 1, 0);
        let r2_1 = s2.row::<8>(y0 + 1, 0);
        let diff1 = _mm_sub_epi16(
            _mm_unpacklo_epi8(_mm_cvtsi64_si128(i64::from_ne_bytes(r1_1)), zero),
            _mm_unpacklo_epi8(_mm_cvtsi64_si128(i64::from_ne_bytes(r2_1)), zero),
        );

        let r1_2 = s1.row::<8>(y0 + 2, 0);
        let r2_2 = s2.row::<8>(y0 + 2, 0);
        let diff2 = _mm_sub_epi16(
            _mm_unpacklo_epi8(_mm_cvtsi64_si128(i64::from_ne_bytes(r1_2)), zero),
            _mm_unpacklo_epi8(_mm_cvtsi64_si128(i64::from_ne_bytes(r2_2)), zero),
        );

        let r1_3 = s1.row::<8>(y0 + 3, 0);
        let r2_3 = s2.row::<8>(y0 + 3, 0);
        let diff3 = _mm_sub_epi16(
            _mm_unpacklo_epi8(_mm_cvtsi64_si128(i64::from_ne_bytes(r1_3)), zero),
            _mm_unpacklo_epi8(_mm_cvtsi64_si128(i64::from_ne_bytes(r2_3)), zero),
        );

        let (v0, v1, v2, v3) = dct_pass(diff0, diff1, diff2, diff3);
        let (c0, c1, c2, c3) = transpose8(v0, v1, v2, v3);
        let (w0, w1, w2, w3) = dct_pass(c0, c1, c2, c3);

        let t0 = _mm_unpacklo_epi16(w0, w1);
        let t1 = _mm_unpacklo_epi16(w2, w3);
        let left_r01 = _mm_unpacklo_epi32(t0, t1);
        let left_r23 = _mm_unpackhi_epi32(t0, t1);

        let t2 = _mm_unpackhi_epi16(w0, w1);
        let t3 = _mm_unpackhi_epi16(w2, w3);
        let right_r01 = _mm_unpacklo_epi32(t2, t3);
        let right_r23 = _mm_unpackhi_epi32(t2, t3);

        st8_i16(&mut out[..8], left_r01);
        st8_i16(&mut out[8..16], left_r23);
        st8_i16(&mut out[16..24], right_r01);
        st8_i16(&mut out[24..32], right_r23);
    }};
}

#[target_feature(enable = "sse2")]
#[inline]
fn add_res_and_clip(pred_4bytes: [u8; 4], res: __m128i) -> [u8; 4] {
    let p_vec = _mm_cvtsi32_si128(i32::from_ne_bytes(pred_4bytes));
    let p_unp = _mm_unpacklo_epi8(p_vec, _mm_setzero_si128());
    let sum = _mm_add_epi16(p_unp, res);
    let packed = _mm_packus_epi16(sum, sum);
    _mm_cvtsi128_si32(packed).to_ne_bytes()
}

#[target_feature(enable = "sse2")]
#[inline]
fn add_res_and_clip8(pred_8bytes: [u8; 8], res: __m128i) -> [u8; 8] {
    let p_vec = _mm_cvtsi64_si128(i64::from_ne_bytes(pred_8bytes));
    let p_unp = _mm_unpacklo_epi8(p_vec, _mm_setzero_si128());
    let sum = _mm_add_epi16(p_unp, res);
    let packed = _mm_packus_epi16(sum, sum);
    _mm_cvtsi128_si64(packed).to_ne_bytes()
}

#[target_feature(enable = "sse2")]
#[inline]
fn idct_col_butterfly_i32(
    s0_32: __m128i,
    s4_32: __m128i,
    s8_32: __m128i,
    s12_32: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let c32 = _mm_set1_epi32(32);
    let t1_a = _mm_add_epi32(s0_32, s8_32);
    let t2_a = _mm_add_epi32(s4_32, _mm_srai_epi32(s12_32, 1));
    let res0 = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(t1_a, t2_a), c32), 6);
    let res3 = _mm_srai_epi32(_mm_add_epi32(_mm_sub_epi32(t1_a, t2_a), c32), 6);

    let t1_b = _mm_sub_epi32(s0_32, s8_32);
    let t2_b = _mm_sub_epi32(_mm_srai_epi32(s4_32, 1), s12_32);
    let res1 = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(t1_b, t2_b), c32), 6);
    let res2 = _mm_srai_epi32(_mm_add_epi32(_mm_sub_epi32(t1_b, t2_b), c32), 6);
    (res0, res1, res2, res3)
}

/// Computes the 4x4 IDCT residual vectors for 4 rows using baseline SSE2.
macro_rules! compute_idct_residuals {
    ($dct:expr) => {{
        let dct: &[i16; 16] = $dct;
        let r01 = ld8_i16(&dct[..8]);
        let r23 = ld8_i16(&dct[8..]);

        let row0 = r01;
        let row1 = _mm_srli_si128(r01, 8);
        let row2 = r23;
        let row3 = _mm_srli_si128(r23, 8);

        let (c0, c1, c2, c3) = transpose4(row0, row1, row2, row3);

        let t0 = _mm_add_epi16(c0, c2);
        let t1 = _mm_sub_epi16(c0, c2);
        let t2 = _mm_sub_epi16(_mm_srai_epi16(c1, 1), c3);
        let t3 = _mm_add_epi16(c1, _mm_srai_epi16(c3, 1));

        let s0 = _mm_add_epi16(t0, t3);
        let s1 = _mm_add_epi16(t1, t2);
        let s2 = _mm_sub_epi16(t1, t2);
        let s3 = _mm_sub_epi16(t0, t3);

        let (r0, r1, r2, r3) = transpose4(s0, s1, s2, s3);

        let s0_32 = _mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16);
        let s4_32 = _mm_srai_epi32(_mm_unpacklo_epi16(r1, r1), 16);
        let s8_32 = _mm_srai_epi32(_mm_unpacklo_epi16(r2, r2), 16);
        let s12_32 = _mm_srai_epi32(_mm_unpacklo_epi16(r3, r3), 16);

        let (res0, res1, res2, res3) = idct_col_butterfly_i32(s0_32, s4_32, s8_32, s12_32);

        let zero = _mm_setzero_si128();
        (
            _mm_packs_epi32(res0, zero),
            _mm_packs_epi32(res1, zero),
            _mm_packs_epi32(res2, zero),
            _mm_packs_epi32(res3, zero),
        )
    }};
}

/// Computes the IDCT residuals of two horizontally adjacent 4x4 blocks (`&[i16; 32]`)
/// in parallel across 8 lanes (`[left 4 | right 4]` per row).
macro_rules! compute_idct_residuals_two {
    ($dct:expr) => {{
        let dct: &[i16; 32] = $dct;
        let l01 = ld8_i16(&dct[..8]);
        let l23 = ld8_i16(&dct[8..16]);
        let r01 = ld8_i16(&dct[16..24]);
        let r23 = ld8_i16(&dct[24..32]);

        let row0 = _mm_unpacklo_epi64(l01, r01);
        let row1 = _mm_unpackhi_epi64(l01, r01);
        let row2 = _mm_unpacklo_epi64(l23, r23);
        let row3 = _mm_unpackhi_epi64(l23, r23);

        let (c0, c1, c2, c3) = transpose8(row0, row1, row2, row3);

        let t0 = _mm_add_epi16(c0, c2);
        let t1 = _mm_sub_epi16(c0, c2);
        let t2 = _mm_sub_epi16(_mm_srai_epi16(c1, 1), c3);
        let t3 = _mm_add_epi16(c1, _mm_srai_epi16(c3, 1));

        let s0 = _mm_add_epi16(t0, t3);
        let s1 = _mm_add_epi16(t1, t2);
        let s2 = _mm_sub_epi16(t1, t2);
        let s3 = _mm_sub_epi16(t0, t3);

        let (g0, g1, g2, g3) = transpose8(s0, s1, s2, s3);

        let (l_res0, l_res1, l_res2, l_res3) = idct_col_butterfly_i32(
            _mm_srai_epi32(_mm_unpacklo_epi16(g0, g0), 16),
            _mm_srai_epi32(_mm_unpacklo_epi16(g1, g1), 16),
            _mm_srai_epi32(_mm_unpacklo_epi16(g2, g2), 16),
            _mm_srai_epi32(_mm_unpacklo_epi16(g3, g3), 16),
        );
        let (r_res0, r_res1, r_res2, r_res3) = idct_col_butterfly_i32(
            _mm_srai_epi32(_mm_unpackhi_epi16(g0, g0), 16),
            _mm_srai_epi32(_mm_unpackhi_epi16(g1, g1), 16),
            _mm_srai_epi32(_mm_unpackhi_epi16(g2, g2), 16),
            _mm_srai_epi32(_mm_unpackhi_epi16(g3, g3), 16),
        );

        [
            _mm_packs_epi32(l_res0, r_res0),
            _mm_packs_epi32(l_res1, r_res1),
            _mm_packs_epi32(l_res2, r_res2),
            _mm_packs_epi32(l_res3, r_res3),
        ]
    }};
}

simd_entry! {
    #[target_feature(enable = "sse2")]
    fn dct_4x4_sse2<A: SampleCursor, B: SampleCursor>(
        dct: &mut [i16; 16],
        pix1: &A,
        pix2: &B,
        _token: Sse2Token,
    ) {
        let (s1, s2) = (pix1.span::<4, 4>(0, 0), pix2.span::<4, 4>(0, 0));
        let zero = _mm_setzero_si128();

        // 1. Compute pixel differences for each 4-sample row
        let r1_0 = s1.row::<4>(0, 0);
        let r2_0 = s2.row::<4>(0, 0);
        let diff0 = _mm_sub_epi16(
            _mm_unpacklo_epi8(_mm_cvtsi32_si128(i32::from_ne_bytes(r1_0)), zero),
            _mm_unpacklo_epi8(_mm_cvtsi32_si128(i32::from_ne_bytes(r2_0)), zero),
        );

        let r1_1 = s1.row::<4>(1, 0);
        let r2_1 = s2.row::<4>(1, 0);
        let diff1 = _mm_sub_epi16(
            _mm_unpacklo_epi8(_mm_cvtsi32_si128(i32::from_ne_bytes(r1_1)), zero),
            _mm_unpacklo_epi8(_mm_cvtsi32_si128(i32::from_ne_bytes(r2_1)), zero),
        );

        let r1_2 = s1.row::<4>(2, 0);
        let r2_2 = s2.row::<4>(2, 0);
        let diff2 = _mm_sub_epi16(
            _mm_unpacklo_epi8(_mm_cvtsi32_si128(i32::from_ne_bytes(r1_2)), zero),
            _mm_unpacklo_epi8(_mm_cvtsi32_si128(i32::from_ne_bytes(r2_2)), zero),
        );

        let r1_3 = s1.row::<4>(3, 0);
        let r2_3 = s2.row::<4>(3, 0);
        let diff3 = _mm_sub_epi16(
            _mm_unpacklo_epi8(_mm_cvtsi32_si128(i32::from_ne_bytes(r1_3)), zero),
            _mm_unpacklo_epi8(_mm_cvtsi32_si128(i32::from_ne_bytes(r2_3)), zero),
        );

        // 2. Vertical 1D DCT across all 4 columns simultaneously
        let (v0, v1, v2, v3) = dct_pass(diff0, diff1, diff2, diff3);

        // 3. In-register transpose to columns
        let (c0, c1, c2, c3) = transpose4(v0, v1, v2, v3);

        // 4. Horizontal 1D DCT across all 4 rows simultaneously
        let (w0, w1, w2, w3) = dct_pass(c0, c1, c2, c3);

        // 5. Pack transposed columns back into row order and store
        let r01 = _mm_unpacklo_epi16(w0, w1);
        let r23 = _mm_unpacklo_epi16(w2, w3);
        let out01 = _mm_unpacklo_epi32(r01, r23);
        let out23 = _mm_unpackhi_epi32(r01, r23);

        st8_i16(&mut dct[..8], out01);
        st8_i16(&mut dct[8..], out23);
    }

    #[target_feature(enable = "sse2")]
    fn dct_four_4x4_sse2<A: SampleCursor, B: SampleCursor>(
        dct: &mut [i16; 64],
        pix1: &A,
        pix2: &B,
        _token: Sse2Token,
    ) {
        let (s1, s2) = (pix1.span::<8, 8>(0, 0), pix2.span::<8, 8>(0, 0));
        let (lo, hi) = dct.split_at_mut(32);
        let lo: &mut [i16; 32] = lo.try_into().unwrap();
        let hi: &mut [i16; 32] = hi.try_into().unwrap();
        dct_two_4x4_from_span!(lo, &s1, &s2, 0);
        dct_two_4x4_from_span!(hi, &s1, &s2, 4);
    }

    #[target_feature(enable = "sse2")]
    fn idct_res_add_pred_sse2(
        pred: &mut PlaneCursorMut<'_>,
        rs: &[i16; 16],
        _token: Sse2Token,
    ) {
        let (res0, res1, res2, res3) = compute_idct_residuals!(rs);
        let mut s = pred.span_mut::<4, 4>(0, 0);
        for (dy, res) in [res0, res1, res2, res3].into_iter().enumerate() {
            let row = s.row_mut::<4>(dy, 0);
            *row = add_res_and_clip(*row, res);
        }
    }

    #[target_feature(enable = "sse2")]
    fn idct_t4_rec_sse2(
        rec: &mut PlaneCursorMut<'_>,
        pred: &PlaneCursor<'_>,
        dct: &[i16; 16],
        _token: Sse2Token,
    ) {
        let (res0, res1, res2, res3) = compute_idct_residuals!(dct);
        let sp = pred.span::<4, 4>(0, 0);
        let mut sr = rec.span_mut::<4, 4>(0, 0);
        for (dy, res) in [res0, res1, res2, res3].into_iter().enumerate() {
            let p = sp.row::<4>(dy, 0);
            *sr.row_mut::<4>(dy, 0) = add_res_and_clip(p, res);
        }
    }

    #[target_feature(enable = "sse2")]
    fn idct_four_t4_rec_sse2(
        rec: &mut PlaneCursorMut<'_>,
        pred: &PlaneCursor<'_>,
        dct: &[i16; 64],
        _token: Sse2Token,
    ) {
        let sp = pred.span::<8, 8>(0, 0);
        let mut sr = rec.span_mut::<8, 8>(0, 0);
        for k in 0..2usize {
            let sub: &[i16; 32] = (&dct[k * 32..][..32]).try_into().unwrap();
            let rows = compute_idct_residuals_two!(sub);
            for (j, res) in rows.into_iter().enumerate() {
                let y = 4 * k + j;
                let p = sp.row::<8>(y, 0);
                *sr.row_mut::<8>(y, 0) = add_res_and_clip8(p, res);
            }
        }
    }

    #[target_feature(enable = "sse2")]
    fn idct_four_t4_rec_in_place_sse2(
        rec: &mut PlaneCursorMut<'_>,
        dct: &[i16; 64],
        _token: Sse2Token,
    ) {
        let mut sr = rec.span_mut::<8, 8>(0, 0);
        for k in 0..2usize {
            let sub: &[i16; 32] = (&dct[k * 32..][..32]).try_into().unwrap();
            let rows = compute_idct_residuals_two!(sub);
            for (j, res) in rows.into_iter().enumerate() {
                let y = 4 * k + j;
                let row = sr.row_mut::<8>(y, 0);
                *row = add_res_and_clip8(*row, res);
            }
        }
    }

    #[target_feature(enable = "sse2")]
    fn idct_t4_rec_to_view_sse2(
        rec: &RecCursor<'_>,
        pred: &[u8],
        pred_stride: usize,
        dct: &[i16; 16],
        _token: Sse2Token,
    ) {
        let (res0, res1, res2, res3) = compute_idct_residuals!(dct);
        let d = rec.span::<4, 4>(0, 0);
        for (dy, res) in [res0, res1, res2, res3].into_iter().enumerate() {
            let p: [u8; 4] = pred[dy * pred_stride..][..4].try_into().unwrap();
            let out = add_res_and_clip(p, res);
            d.set_row::<4>(dy, &out);
        }
    }

    #[target_feature(enable = "sse2")]
    fn idct_four_t4_rec_to_view_sse2(
        rec: &RecCursor<'_>,
        pred: &[u8],
        pred_stride: usize,
        dct: &[i16; 64],
        _token: Sse2Token,
    ) {
        let d = rec.span::<8, 8>(0, 0);
        for k in 0..2usize {
            let sub: &[i16; 32] = (&dct[k * 32..][..32]).try_into().unwrap();
            let rows = compute_idct_residuals_two!(sub);
            for (j, res) in rows.into_iter().enumerate() {
                let y = 4 * k + j;
                let p: [u8; 8] = pred[y * pred_stride..][..8].try_into().unwrap();
                let out = add_res_and_clip8(p, res);
                d.set_row::<8>(y, &out);
            }
        }
    }

    #[target_feature(enable = "sse2")]
    fn idct_t4_rec_in_place_view_sse2(
        rec: &RecCursor<'_>,
        dct: &[i16; 16],
        _token: Sse2Token,
    ) {
        let (res0, res1, res2, res3) = compute_idct_residuals!(dct);
        let s = rec.span::<4, 4>(0, 0);
        for (dy, res) in [res0, res1, res2, res3].into_iter().enumerate() {
            let cur = s.row::<4>(dy, 0);
            let out = add_res_and_clip(cur, res);
            s.set_row::<4>(dy, &out);
        }
    }

    #[target_feature(enable = "sse2")]
    fn idct_four_t4_rec_in_place_view_sse2(
        rec: &RecCursor<'_>,
        dct: &[i16; 64],
        _token: Sse2Token,
    ) {
        let s = rec.span::<8, 8>(0, 0);
        for k in 0..2usize {
            let sub: &[i16; 32] = (&dct[k * 32..][..32]).try_into().unwrap();
            let rows = compute_idct_residuals_two!(sub);
            for (j, res) in rows.into_iter().enumerate() {
                let y = 4 * k + j;
                let cur = s.row::<8>(y, 0);
                let out = add_res_and_clip8(cur, res);
                s.set_row::<8>(y, &out);
            }
        }
    }

    #[target_feature(enable = "sse2")]
    fn idct_rec_i16x16_dc_sse2_impl(
        rec: &mut PlaneCursorMut<'_>,
        pred: &PlaneCursor<'_>,
        dc: &[i16; 16],
        _token: Sse2Token,
    ) {
        let zero = _mm_setzero_si128();
        let sp = pred.span::<16, 16>(0, 0);
        let mut sr = rec.span_mut::<16, 16>(0, 0);

        for k in 0..4usize {
            let ws = sp.window::<16>(4 * k, 4);
            let mut wd = sr.window_mut::<16>(4 * k, 4);
            let dc_row = 4 * k;
            let d0 = ((dc[dc_row] as i32 + 32) >> 6) as i16;
            let d1 = ((dc[dc_row + 1] as i32 + 32) >> 6) as i16;
            let d2 = ((dc[dc_row + 2] as i32 + 32) >> 6) as i16;
            let d3 = ((dc[dc_row + 3] as i32 + 32) >> 6) as i16;
            let dc_lo = _mm_set_epi16(d1, d1, d1, d1, d0, d0, d0, d0);
            let dc_hi = _mm_set_epi16(d3, d3, d3, d3, d2, d2, d2, d2);
            for j in 0..4usize {
                let p_vec = ld16(&ws.row::<16>(j, 0));
                let sum_lo = _mm_add_epi16(_mm_unpacklo_epi8(p_vec, zero), dc_lo);
                let sum_hi = _mm_add_epi16(_mm_unpackhi_epi8(p_vec, zero), dc_hi);
                st16(wd.row_mut::<16>(j, 0), _mm_packus_epi16(sum_lo, sum_hi));
            }
        }
    }

    #[target_feature(enable = "sse2")]
    fn idct_rec_i16x16_dc_to_view_sse2_impl(
        rec: &RecCursor<'_>,
        pred: &[u8],
        pred_stride: usize,
        dc: &[i16; 16],
        _token: Sse2Token,
    ) {
        let zero = _mm_setzero_si128();
        let d = rec.span::<16, 16>(0, 0);

        for k in 0..4usize {
            let wd = d.window::<16>(4 * k, 4);
            let dc_row = 4 * k;
            let d0 = ((dc[dc_row] as i32 + 32) >> 6) as i16;
            let d1 = ((dc[dc_row + 1] as i32 + 32) >> 6) as i16;
            let d2 = ((dc[dc_row + 2] as i32 + 32) >> 6) as i16;
            let d3 = ((dc[dc_row + 3] as i32 + 32) >> 6) as i16;
            let dc_lo = _mm_set_epi16(d1, d1, d1, d1, d0, d0, d0, d0);
            let dc_hi = _mm_set_epi16(d3, d3, d3, d3, d2, d2, d2, d2);
            for j in 0..4usize {
                let i = 4 * k + j;
                let p_vec = ld16(&pred[i * pred_stride..][..16]);
                let sum_lo = _mm_add_epi16(_mm_unpacklo_epi8(p_vec, zero), dc_lo);
                let sum_hi = _mm_add_epi16(_mm_unpackhi_epi8(p_vec, zero), dc_hi);
                wd.set_row::<16>(j, &to16(_mm_packus_epi16(sum_lo, sum_hi)));
            }
        }
    }
}

/// 4x4 Forward Integer DCT of the pixel difference `(pix1 - pix2)` using SSE2.
///
/// C++: `WelsDctT4_sse2`, `codec/common/x86/dct.asm`.
#[inline(always)]
pub fn dct_4x4<A: SampleCursor, B: SampleCursor>(dct: &mut [i16; 16], pix1: &A, pix2: &B) {
    dct_4x4_sse2(dct, pix1, pix2, sse2_token());
}

/// Performs 4x4 FDCT on four adjacent 4x4 blocks forming an 8x8 quadrant using SSE2.
///
/// C++: `WelsDctFourT4_sse2`, `codec/common/x86/dct.asm`.
#[inline(always)]
pub fn dct_four_4x4<A: SampleCursor, B: SampleCursor>(dct: &mut [i16; 64], pix1: &A, pix2: &B) {
    dct_four_4x4_sse2(dct, pix1, pix2, sse2_token());
}

// ============================================================================
// Inverse 4x4 Integer DCT & Prediction Addition
// ============================================================================

/// 4x4 inverse integer DCT of `rs`, added to `pred` and saturated to `[0, 255]` in place.
///
/// C++: `IdctResAddPred_sse2`, `codec/common/x86/dct.asm`.
#[inline(always)]
pub fn idct_res_add_pred(pred: &mut PlaneCursorMut<'_>, rs: &[i16; 16]) {
    idct_res_add_pred_sse2(pred, rs, sse2_token());
}

/// 4x4 IDCT with separate source prediction cursor.
///
/// C++: `WelsIDctT4Rec_sse2`, `codec/common/x86/dct.asm`.
#[inline(always)]
pub fn idct_t4_rec(rec: &mut PlaneCursorMut<'_>, pred: &PlaneCursor<'_>, dct: &[i16; 16]) {
    idct_t4_rec_sse2(rec, pred, dct, sse2_token());
}

/// [`idct_t4_rec`] in place on `rec`.
#[inline(always)]
pub fn idct_t4_rec_in_place(rec: &mut PlaneCursorMut<'_>, dct: &[i16; 16]) {
    idct_res_add_pred(rec, dct)
}

/// IDCT over four 4x4 blocks forming an 8x8 quadrant.
///
/// C++: `WelsIDctFourT4Rec_sse2`, `codec/common/x86/dct.asm`.
#[inline(always)]
pub fn idct_four_t4_rec(rec: &mut PlaneCursorMut<'_>, pred: &PlaneCursor<'_>, dct: &[i16; 64]) {
    idct_four_t4_rec_sse2(rec, pred, dct, sse2_token());
}

/// [`idct_t4_rec_in_place`] over four 4x4 blocks forming an 8x8 quadrant.
#[inline(always)]
pub fn idct_four_t4_rec_in_place(rec: &mut PlaneCursorMut<'_>, dct: &[i16; 64]) {
    idct_four_t4_rec_in_place_sse2(rec, dct, sse2_token());
}

/// [`idct_t4_rec_to_view`] using SSE2.
#[inline(always)]
pub fn idct_t4_rec_to_view(rec: &RecCursor<'_>, pred: &[u8], pred_stride: usize, dct: &[i16; 16]) {
    idct_t4_rec_to_view_sse2(rec, pred, pred_stride, dct, sse2_token());
}

/// [`idct_four_t4_rec_to_view`] using SSE2.
#[inline(always)]
pub fn idct_four_t4_rec_to_view(
    rec: &RecCursor<'_>,
    pred: &[u8],
    pred_stride: usize,
    dct: &[i16; 64],
) {
    idct_four_t4_rec_to_view_sse2(rec, pred, pred_stride, dct, sse2_token());
}

/// [`idct_t4_rec_in_place_view`] using SSE2.
#[inline(always)]
pub fn idct_t4_rec_in_place_view(rec: &RecCursor<'_>, dct: &[i16; 16]) {
    idct_t4_rec_in_place_view_sse2(rec, dct, sse2_token());
}

/// [`idct_four_t4_rec_in_place_view`] using SSE2.
#[inline(always)]
pub fn idct_four_t4_rec_in_place_view(rec: &RecCursor<'_>, dct: &[i16; 64]) {
    idct_four_t4_rec_in_place_view_sse2(rec, dct, sse2_token());
}

/// [`idct_t4_rec_on_mb_in_place_view`] using SSE2.
#[inline(always)]
pub fn idct_t4_rec_on_mb_in_place_view(rec: &RecCursor<'_>, dct: &[i16; 256]) {
    const QUADS: [(isize, isize); 4] = [(0, 0), (8, 0), (0, 8), (8, 8)];
    for (k, &(dx, dy)) in QUADS.iter().enumerate() {
        let sub: &[i16; 64] = (&dct[k << 6..][..64]).try_into().unwrap();
        idct_four_t4_rec_in_place_view(&rec.advance(dx, dy), sub);
    }
}

/// 16x16 macroblock DC luma reconstruction using SSE2.
///
/// C++: `WelsIDctRecI16x16Dc_sse2`, `codec/common/x86/dct.asm`.
#[inline]
pub fn idct_rec_i16x16_dc(rec: &mut PlaneCursorMut<'_>, pred: &PlaneCursor<'_>, dc: &[i16; 16]) {
    idct_rec_i16x16_dc_sse2_impl(rec, pred, dc, sse2_token());
}

/// 16x16 macroblock DC luma reconstruction to view using SSE2.
#[inline]
pub fn idct_rec_i16x16_dc_to_view(
    rec: &RecCursor<'_>,
    pred: &[u8],
    pred_stride: usize,
    dc: &[i16; 16],
) {
    idct_rec_i16x16_dc_to_view_sse2_impl(rec, pred, pred_stride, dc, sse2_token());
}

// ============================================================================
// Unit Tests & Parity
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::encode_mb_aux::{dct_4x4 as dct_4x4_c, dct_four_4x4 as dct_four_4x4_c};
    // The `_c` scalar kernels, not the same-named dispatchers: a dispatcher routes to
    // the kernel under test, which would make every assertion below a tautology.
    use crate::decoder::decode_mb_aux::idct_res_add_pred_c;
    use crate::encoder::decode_mb_aux::{
        idct_rec_i16x16_dc_c, idct_t4_rec_c, idct_t4_rec_in_place_c,
    };
    use crate::encoder::rec_view::shared_plane_for_test;
    use crate::safe::plane::PaddedPlane;

    fn lcg(seed: &mut u64) -> u8 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 32) & 0xFF) as u8
    }

    /// Coefficients over the full `i16` range, which is what the decoder hands the
    /// IDCT after dequantisation. A narrower cap keeps the vertical pass inside the
    /// range where 16- and 32-bit lanes agree, and would pass on a wrong kernel.
    fn lcg_i16(seed: &mut u64) -> i16 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 32) as u16 as i16
    }

    #[test]
    fn test_dct_4x4_parity() {
        let mut seed = 12345u64;
        let (w, h, pad, stride) = (16usize, 16usize, 16usize, 64usize);
        let mut p1 = PaddedPlane::new(w, h, pad, stride);
        let mut p2 = PaddedPlane::new(w, h, pad, stride);

        for _ in 0..100 {
            for y in 0..4isize {
                for x in 0..4isize {
                    p1.set(x, y, lcg(&mut seed));
                    p2.set(x, y, lcg(&mut seed));
                }
            }

            let mut dct_c = [0i16; 16];
            let mut dct_simd = [0i16; 16];

            dct_4x4_c(&mut dct_c, &p1.cursor(0, 0), &p2.cursor(0, 0));
            dct_4x4(&mut dct_simd, &p1.cursor(0, 0), &p2.cursor(0, 0));

            assert_eq!(dct_simd, dct_c);
        }
    }

    #[test]
    fn test_dct_four_4x4_parity() {
        let mut seed = 54321u64;
        let (w, h, pad, stride) = (16usize, 16usize, 16usize, 64usize);
        let mut p1 = PaddedPlane::new(w, h, pad, stride);
        let mut p2 = PaddedPlane::new(w, h, pad, stride);

        for _ in 0..50 {
            for y in 0..8isize {
                for x in 0..8isize {
                    p1.set(x, y, lcg(&mut seed));
                    p2.set(x, y, lcg(&mut seed));
                }
            }

            let mut dct_c = [0i16; 64];
            let mut dct_simd = [0i16; 64];

            dct_four_4x4_c(&mut dct_c, &p1.cursor(0, 0), &p2.cursor(0, 0));
            dct_four_4x4(&mut dct_simd, &p1.cursor(0, 0), &p2.cursor(0, 0));

            assert_eq!(dct_simd, dct_c);
        }
    }

    #[test]
    fn test_idct_res_add_pred_parity() {
        let mut seed = 98765u64;
        let (w, h, pad, stride) = (16usize, 16usize, 16usize, 64usize);
        let mut p_c = PaddedPlane::new(w, h, pad, stride);
        let mut p_simd = PaddedPlane::new(w, h, pad, stride);

        for _ in 0..100 {
            for y in 0..4isize {
                for x in 0..4isize {
                    let v = lcg(&mut seed);
                    p_c.set(x, y, v);
                    p_simd.set(x, y, v);
                }
            }

            let mut rs = [0i16; 16];
            for v in rs.iter_mut() {
                *v = lcg_i16(&mut seed);
            }

            idct_res_add_pred_c(&mut p_c.cursor_mut(0, 0), &rs);
            idct_res_add_pred(&mut p_simd.cursor_mut(0, 0), &rs);

            for y in 0..4isize {
                for x in 0..4isize {
                    assert_eq!(p_simd.at(x, y), p_c.at(x, y), "mismatch at ({x}, {y})");
                }
            }
        }
    }

    /// `rs[0] = rs[8] = 20000` with a zero prediction puts `t1 = s0 + s8 = 40000` into
    /// the vertical butterfly. In `i32` that is `(32 + 40000) >> 6 = 625`, clipped to
    /// 255; in 16-bit lanes it wraps to `-25536`, `>> 6 = -399`, which `packus`
    /// saturates to 0.
    #[test]
    fn idct_vertical_pass_does_not_wrap_at_16_bits() {
        let (w, h, pad, stride) = (16usize, 16usize, 16usize, 64usize);
        let mut p_c = PaddedPlane::new(w, h, pad, stride);
        let mut p_simd = PaddedPlane::new(w, h, pad, stride);
        for y in 0..4isize {
            for x in 0..4isize {
                p_c.set(x, y, 0);
                p_simd.set(x, y, 0);
            }
        }

        let mut rs = [0i16; 16];
        rs[0] = 20000;
        rs[8] = 20000;

        idct_res_add_pred_c(&mut p_c.cursor_mut(0, 0), &rs);
        idct_res_add_pred(&mut p_simd.cursor_mut(0, 0), &rs);

        assert_eq!(p_c.at(0, 0), 255, "the scalar reference itself moved");
        for y in 0..4isize {
            for x in 0..4isize {
                assert_eq!(p_simd.at(x, y), p_c.at(x, y), "mismatch at ({x}, {y})");
            }
        }
    }

    #[test]
    fn test_idct_t4_rec_parity() {
        let mut seed = 112233u64;
        let (w, h, pad, stride) = (16usize, 16usize, 16usize, 64usize);
        let pred = PaddedPlane::new(w, h, pad, stride);
        let mut rec_c = PaddedPlane::new(w, h, pad, stride);
        let mut rec_simd = PaddedPlane::new(w, h, pad, stride);

        for _ in 0..100 {
            let mut rs = [0i16; 16];
            for v in rs.iter_mut() {
                *v = lcg_i16(&mut seed);
            }

            idct_t4_rec_c(&mut rec_c.cursor_mut(0, 0), &pred.cursor(0, 0), &rs);
            idct_t4_rec(&mut rec_simd.cursor_mut(0, 0), &pred.cursor(0, 0), &rs);

            for y in 0..4isize {
                for x in 0..4isize {
                    assert_eq!(rec_simd.at(x, y), rec_c.at(x, y), "mismatch at ({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn test_idct_rec_i16x16_dc_parity() {
        let mut seed = 445566u64;
        let (w, h, pad, stride) = (32usize, 32usize, 16usize, 64usize);
        let mut pred = PaddedPlane::new(w, h, pad, stride);
        let mut rec_c = PaddedPlane::new(w, h, pad, stride);
        let mut rec_simd = PaddedPlane::new(w, h, pad, stride);

        for y in 0..16isize {
            for x in 0..16isize {
                pred.set(x, y, lcg(&mut seed));
            }
        }

        for _ in 0..20 {
            let mut dc = [0i16; 16];
            for v in dc.iter_mut() {
                *v = lcg_i16(&mut seed);
            }

            idct_rec_i16x16_dc_c(&mut rec_c.cursor_mut(0, 0), &pred.cursor(0, 0), &dc);
            idct_rec_i16x16_dc(&mut rec_simd.cursor_mut(0, 0), &pred.cursor(0, 0), &dc);

            for y in 0..16isize {
                for x in 0..16isize {
                    assert_eq!(
                        rec_simd.at(x, y),
                        rec_c.at(x, y),
                        "dc mismatch at ({x}, {y})"
                    );
                }
            }
        }
    }

    // ========================================================================
    // The reconstruction-seam entry points.
    //
    // Each runs an `_sse2` kernel against `idct_t4_rec_c` / `idct_t4_rec_in_place_c` /
    // `idct_rec_i16x16_dc_c`, which cannot route back here.
    //
    // The multi-block forms are referenced against the scalar applied per block at the
    // sub-offsets, not against this file's own four-block loop, so a transposed
    // `(dx, dy)` in the `off`/`advance` arithmetic fails rather than being a shared
    // assumption. Whole allocations are compared, never just the block.
    // ========================================================================

    /// Two planes of identical geometry filled with the same noise, padding included,
    /// so an out-of-block write shows up as an allocation difference.
    fn twin_planes(seed: &mut u64) -> (PaddedPlane, PaddedPlane) {
        let (w, h, pad, stride) = (32usize, 32usize, 16usize, 64usize);
        let mut a = PaddedPlane::new(w, h, pad, stride);
        let mut b = PaddedPlane::new(w, h, pad, stride);
        for y in -(pad as isize)..(h + pad) as isize {
            for x in -(pad as isize)..(w + pad) as isize {
                let v = lcg(seed);
                a.set(x, y, v);
                b.set(x, y, v);
            }
        }
        (a, b)
    }

    /// A flat prediction arena at `stride` — the shape `sMemPredMb` has.
    fn noisy_pred(seed: &mut u64, stride: usize, rows: usize) -> Vec<u8> {
        (0..stride * rows).map(|_| lcg(seed)).collect()
    }

    /// The same bytes as a plane, for the `PlaneCursorMut` scalar reference.
    fn pred_as_plane(pred: &[u8], stride: usize, rows: usize) -> PaddedPlane {
        let mut pp = PaddedPlane::new(stride, rows, 0, stride);
        for y in 0..rows as isize {
            for x in 0..stride as isize {
                pp.set(x, y, pred[y as usize * stride + x as usize]);
            }
        }
        pp
    }

    fn coeffs<const N: usize>(seed: &mut u64) -> [i16; N] {
        core::array::from_fn(|_| lcg_i16(seed))
    }

    const SUBS: [(isize, isize); 4] = [(0, 0), (4, 0), (0, 4), (4, 4)];
    const QUADS: [(isize, isize); 4] = [(0, 0), (8, 0), (0, 8), (8, 8)];

    #[test]
    fn idct_t4_rec_in_place_matches_the_scalar() {
        let mut seed = 0x0DDB_1A5E_5BAD_5EEDu64;
        let (mut pa, mut pb) = twin_planes(&mut seed);
        let dct: [i16; 16] = coeffs(&mut seed);

        idct_t4_rec_in_place_c(&mut pa.cursor_mut(5, 7), &dct);
        idct_t4_rec_in_place(&mut pb.cursor_mut(5, 7), &dct);

        assert_eq!(pa.as_slice(), pb.as_slice());
    }

    #[test]
    fn idct_four_t4_rec_matches_four_scalar_blocks() {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let (mut pa, mut pb) = twin_planes(&mut seed);
        let dct: [i16; 64] = coeffs(&mut seed);
        let pred = noisy_pred(&mut seed, 16, 8);
        let pp = pred_as_plane(&pred, 16, 8);

        for (k, &(dx, dy)) in SUBS.iter().enumerate() {
            let sub: &[i16; 16] = (&dct[k << 4..][..16]).try_into().unwrap();
            idct_t4_rec_c(&mut pa.cursor_mut(6 + dx, 9 + dy), &pp.cursor(dx, dy), sub);
        }
        idct_four_t4_rec(&mut pb.cursor_mut(6, 9), &pp.cursor(0, 0), &dct);

        assert_eq!(pa.as_slice(), pb.as_slice());
    }

    #[test]
    fn idct_four_t4_rec_in_place_matches_four_scalar_blocks() {
        let mut seed = 0x8A5C_D789_635D_2DFFu64;
        let (mut pa, mut pb) = twin_planes(&mut seed);
        let dct: [i16; 64] = coeffs(&mut seed);

        for (k, &(dx, dy)) in SUBS.iter().enumerate() {
            let sub: &[i16; 16] = (&dct[k << 4..][..16]).try_into().unwrap();
            idct_t4_rec_in_place_c(&mut pa.cursor_mut(6 + dx, 9 + dy), sub);
        }
        idct_four_t4_rec_in_place(&mut pb.cursor_mut(6, 9), &dct);

        assert_eq!(pa.as_slice(), pb.as_slice());
    }

    #[test]
    fn idct_t4_rec_to_view_matches_the_scalar() {
        let mut seed = 0x1D87_2E7F_0000_0001u64;
        let (mut pa, mut pb) = twin_planes(&mut seed);
        let dct: [i16; 16] = coeffs(&mut seed);
        let pred = noisy_pred(&mut seed, 16, 4);
        let pp = pred_as_plane(&pred, 16, 4);

        idct_t4_rec_c(&mut pa.cursor_mut(5, 7), &pp.cursor(0, 0), &dct);

        let view = shared_plane_for_test(&mut pb);
        idct_t4_rec_to_view(&view.cursor(5, 7), &pred, 16, &dct);

        assert_eq!(pa.as_slice(), pb.as_slice());
    }

    #[test]
    fn idct_four_t4_rec_to_view_matches_four_scalar_blocks() {
        let mut seed = 0x6C07_8965_0000_0001u64;
        let (mut pa, mut pb) = twin_planes(&mut seed);
        let dct: [i16; 64] = coeffs(&mut seed);
        let pred = noisy_pred(&mut seed, 16, 8);
        let pp = pred_as_plane(&pred, 16, 8);

        for (k, &(dx, dy)) in SUBS.iter().enumerate() {
            let sub: &[i16; 16] = (&dct[k << 4..][..16]).try_into().unwrap();
            idct_t4_rec_c(&mut pa.cursor_mut(6 + dx, 9 + dy), &pp.cursor(dx, dy), sub);
        }

        let view = shared_plane_for_test(&mut pb);
        idct_four_t4_rec_to_view(&view.cursor(6, 9), &pred, 16, &dct);

        assert_eq!(pa.as_slice(), pb.as_slice());
    }

    #[test]
    fn idct_t4_rec_in_place_view_matches_the_scalar() {
        let mut seed = 0x41C6_4E6D_0000_0001u64;
        let (mut pa, mut pb) = twin_planes(&mut seed);
        let dct: [i16; 16] = coeffs(&mut seed);

        idct_t4_rec_in_place_c(&mut pa.cursor_mut(5, 7), &dct);

        let view = shared_plane_for_test(&mut pb);
        idct_t4_rec_in_place_view(&view.cursor(5, 7), &dct);

        assert_eq!(pa.as_slice(), pb.as_slice());
    }

    #[test]
    fn idct_four_t4_rec_in_place_view_matches_four_scalar_blocks() {
        let mut seed = 0x3C6E_F35F_0000_0001u64;
        let (mut pa, mut pb) = twin_planes(&mut seed);
        let dct: [i16; 64] = coeffs(&mut seed);

        for (k, &(dx, dy)) in SUBS.iter().enumerate() {
            let sub: &[i16; 16] = (&dct[k << 4..][..16]).try_into().unwrap();
            idct_t4_rec_in_place_c(&mut pa.cursor_mut(6 + dx, 9 + dy), sub);
        }

        let view = shared_plane_for_test(&mut pb);
        idct_four_t4_rec_in_place_view(&view.cursor(6, 9), &dct);

        assert_eq!(pa.as_slice(), pb.as_slice());
    }

    /// The 16x16 form: four quadrants of four blocks, so a quadrant-level `(dx, dy)`
    /// swap and a block-level one are both visible.
    #[test]
    fn idct_t4_rec_on_mb_in_place_view_matches_sixteen_scalar_blocks() {
        let mut seed = 0x9E37_79B9_0000_0001u64;
        let (mut pa, mut pb) = twin_planes(&mut seed);
        let dct: [i16; 256] = coeffs(&mut seed);

        for (q, &(qx, qy)) in QUADS.iter().enumerate() {
            for (k, &(dx, dy)) in SUBS.iter().enumerate() {
                let off = (q << 6) + (k << 4);
                let sub: &[i16; 16] = (&dct[off..][..16]).try_into().unwrap();
                idct_t4_rec_in_place_c(&mut pa.cursor_mut(4 + qx + dx, 6 + qy + dy), sub);
            }
        }

        let view = shared_plane_for_test(&mut pb);
        idct_t4_rec_on_mb_in_place_view(&view.cursor(4, 6), &dct);

        assert_eq!(pa.as_slice(), pb.as_slice());
    }

    #[test]
    fn idct_rec_i16x16_dc_to_view_matches_the_scalar() {
        let mut seed = 0xB502_6F5A_0000_0001u64;
        let (mut pa, mut pb) = twin_planes(&mut seed);
        let dc: [i16; 16] = coeffs(&mut seed);
        let pred = noisy_pred(&mut seed, 16, 16);
        let pp = pred_as_plane(&pred, 16, 16);

        idct_rec_i16x16_dc_c(&mut pa.cursor_mut(5, 7), &pp.cursor(0, 0), &dc);

        let view = shared_plane_for_test(&mut pb);
        idct_rec_i16x16_dc_to_view(&view.cursor(5, 7), &pred, 16, &dc);

        assert_eq!(pa.as_slice(), pb.as_slice());
    }
}
