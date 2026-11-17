// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! x86_64 SSE2 Intra Prediction Kernels.
//!
//! 16x16 luma, 8x8 chroma, and 4x4 luma intra predictors, serving both the encoder
//! candidate generator and the decoder in-place reconstructor.
//!
//! A kernel here is named for the operation, not for an instruction set — the same
//! name its twin has in `simd::aarch64`, since the module path says which implementation
//! is in use. Fourteen of these predictors are not vectorized: the pure V, H, DC and
//! DC-NA fills are word-wide rewrites of their scalar twins, which the init tables
//! install deliberately. `every_kernel_here_reaches_an_intrinsic` lists those fourteen
//! and requires intrinsics in every other public kernel.

#![deny(unsafe_code)]

use core::arch::x86_64::*;

use super::lanes::{ld8, ld16, st16, st32, to8, to16};
use super::{sse2_token, ssse3_token};
use crate::encoder::rec_view::RecCursor;
use crate::safe::plane::{BlockRows, PlaneCursorMut, RefSamples};
use crate::simd::{
    simd_entry,
    tokens::{Avx2Token, Sse2Token, Ssse3Token},
};

// ============================================================================
// 16x16 Luma Intra Prediction (SSE2)
// ============================================================================

// ============================================================================
// Shared enc/dec bodies
// ============================================================================

/// Where a predictor's rows go. The `enc_`/`dec_` pairs share their neighbour reads
/// (`RefSamples::at` and `row_n`, which `RecCursor` and `PlaneCursorMut` both
/// implement) and every line of arithmetic; they differ only at the store — the
/// encoder fills a packed candidate buffer at a fixed pitch, the decoder writes back
/// through the same cursor it read from.
///
/// The decoder reads neighbours and writes rows through one cursor, so a body cannot
/// hold `&S` and `&mut O` to it at once: every body reads and computes from `&S`
/// first, then writes.
trait PredOut {
    /// Writes `row` as row `dy` of the destination.
    fn put<const N: usize>(&mut self, dy: usize, row: &[u8; N]);
}

/// The encoder's packed candidate buffer — `W` bytes per row, `W * W` long, the
/// `sMemPredMb` arena shape.
struct Packed<'a, const W: usize>(&'a mut [u8]);

impl<const W: usize> PredOut for Packed<'_, W> {
    #[inline(always)]
    fn put<const N: usize>(&mut self, dy: usize, row: &[u8; N]) {
        self.0[dy * W..][..N].copy_from_slice(row);
    }
}

impl PredOut for PlaneCursorMut<'_> {
    #[inline(always)]
    fn put<const N: usize>(&mut self, dy: usize, row: &[u8; N]) {
        self.row_mut(dy as isize, 0, N).copy_from_slice(row);
    }
}

/// Fills `rows` rows of `N` bytes with one value — V, H, DC and the DC variants all
/// reduce to this once their mean or their source row is known.
#[inline(always)]
fn fill_rows<const N: usize, O: PredOut>(out: &mut O, rows: usize, row: &[u8; N]) {
    for dy in 0..rows {
        out.put(dy, row);
    }
}

/// Reads `H` single-sample rows at `(dx0, dy0)` without per-row bounds checks or stack spills.
#[inline(always)]
fn read_col1<const H: usize, S: RefSamples>(src: &S, dy0: isize, dx0: isize) -> [u8; H] {
    let span = src.span::<1, H>(dy0, dx0);
    let mut out = [0u8; H];
    if H == 16 {
        let (qs, st) = span.all_4_quads::<1>();
        for (i, q) in qs.into_iter().enumerate() {
            let (a0, a1, a2, a3) = S::Span::read_quad::<1>(q, st);
            out[i * 4] = a0[0];
            out[i * 4 + 1] = a1[0];
            out[i * 4 + 2] = a2[0];
            out[i * 4 + 3] = a3[0];
        }
    } else if H == 8 {
        let (qs, st) = span.all_2_quads::<1>();
        for (i, q) in qs.into_iter().enumerate() {
            let (a0, a1, a2, a3) = S::Span::read_quad::<1>(q, st);
            out[i * 4] = a0[0];
            out[i * 4 + 1] = a1[0];
            out[i * 4 + 2] = a2[0];
            out[i * 4 + 3] = a3[0];
        }
    } else if H == 4 {
        let (q, st) = span.all_1_quad::<1>();
        let (a0, a1, a2, a3) = S::Span::read_quad::<1>(q, st);
        out[0] = a0[0];
        out[1] = a1[0];
        out[2] = a2[0];
        out[3] = a3[0];
    } else {
        for (y, dst) in out.iter_mut().enumerate() {
            *dst = span.row::<1>(y, 0)[0];
        }
    }
    out
}

/// `(lt_shift, top_shift, left_shift)` for the 16x16 plane predictor.
///
/// C++: `WelsI16x16LumaPredPlane_c`, `codec/common/src/intra_pred_common.cpp`.
#[inline(always)]
fn i16x16_plane_coeffs<S: RefSamples>(src: &S) -> (i32, i32, i32) {
    let top = src.row_n::<17>(-1, -1);
    let col = read_col1::<16, _>(src, 0, -1);
    let mut left = [0u8; 17];
    left[0] = top[0];
    left[1..].copy_from_slice(&col);
    let mut top_sum: i32 = 0;
    let mut left_sum: i32 = 0;
    for i in 0..8usize {
        top_sum += (i as i32 + 1) * (top[9 + i] as i32 - top[7 - i] as i32);
        left_sum += (i as i32 + 1) * (left[9 + i] as i32 - left[7 - i] as i32);
    }
    let lt_shift = (left[16] as i32 + top[16] as i32) << 4;
    ((5 * top_sum + 32) >> 6, (5 * left_sum + 32) >> 6, lt_shift)
}

simd_entry! {
    /// The 16x16 plane fill, from the three coefficients.
    #[target_feature(enable = "sse2")]
    fn i16x16_plane_fill<O: PredOut>(
        out: &mut O,
        top_shift: i32,
        left_shift: i32,
        lt_shift: i32,
        _token: Sse2Token,
    ) {
        let inc_minus = _mm_setr_epi16(-7, -6, -5, -4, -3, -2, -1, 0);
        let inc = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b_vec = _mm_set1_epi16(top_shift as i16);
        let c_vec = _mm_set1_epi16(left_shift as i16);
        let mut s_vec = _mm_set1_epi16((lt_shift + 16 - 7 * left_shift) as i16);

        let term_lo = _mm_mullo_epi16(b_vec, inc_minus);
        let term_hi = _mm_mullo_epi16(b_vec, inc);

        for dy in 0..16 {
            let row_lo = _mm_srai_epi16(_mm_add_epi16(term_lo, s_vec), 5);
            let row_hi = _mm_srai_epi16(_mm_add_epi16(term_hi, s_vec), 5);
            let row = to16(_mm_packus_epi16(row_lo, row_hi));
            out.put(dy, &row);
            s_vec = _mm_add_epi16(s_vec, c_vec);
        }
    }

    /// The 16x16 DC mean over whichever of the two neighbour edges the variant uses.
    ///
    /// C++: `WelsI16x16LumaPredDc_c` and its `_T`/`_NA` siblings, which differ only in
    /// which sums are in scope and the rounding.
    #[target_feature(enable = "sse2")]
    fn i16x16_dc_mean<S: RefSamples, const USE_TOP: bool, const USE_LEFT: bool>(
        src: &S,
        _token: Sse2Token,
    ) -> u8 {
        let sum_top = if USE_TOP {
            let top = src.row_n::<16>(-1, 0);
            let sad = _mm_sad_epu8(ld16(&top), _mm_setzero_si128());
            _mm_cvtsi128_si32(sad) + _mm_extract_epi16(sad, 4)
        } else {
            0
        };
        let sum_left: i32 = if USE_LEFT {
            read_col1::<16, _>(src, 0, -1).into_iter().map(i32::from).sum()
        } else {
            0
        };
        match (USE_TOP, USE_LEFT) {
            (true, true) => ((16 + sum_top + sum_left) >> 5) as u8,
            (true, false) => ((8 + sum_top) >> 4) as u8,
            (false, true) => ((8 + sum_left) >> 4) as u8,
            (false, false) => 0x80,
        }
    }
}

/// The four 4x4-quadrant means of the 8x8 chroma DC predictor, as the two row values
/// the top and bottom halves are filled with.
///
/// C++: `WelsIChromaPredDc_c`.
#[inline(always)]
fn chroma_dc_rows<S: RefSamples>(src: &S) -> ([u8; 8], [u8; 8]) {
    let top = src.row_n::<8>(-1, 0);
    let left = read_col1::<8, _>(src, 0, -1);
    let sum_top0: i32 = (0..4).map(|i| top[i] as i32).sum();
    let sum_top1: i32 = (0..4).map(|i| top[i + 4] as i32).sum();
    let sum_left0: i32 = (0..4).map(|y| left[y] as i32).sum();
    let sum_left1: i32 = (0..4).map(|y| left[y + 4] as i32).sum();

    let mean1 = ((sum_top0 + sum_left0 + 4) >> 3) as u8;
    let mean2 = ((sum_top1 + 2) >> 2) as u8;
    let mean3 = ((sum_left1 + 2) >> 2) as u8;
    let mean4 = ((sum_top1 + sum_left1 + 4) >> 3) as u8;

    (
        [mean1, mean1, mean1, mean1, mean2, mean2, mean2, mean2],
        [mean3, mean3, mean3, mean3, mean4, mean4, mean4, mean4],
    )
}

/// `(top_shift, left_shift, lt_shift)` for the 8x8 chroma plane predictor.
///
/// C++: `WelsIChromaPredPlane_c`.
#[inline(always)]
fn chroma_plane_coeffs<S: RefSamples>(src: &S) -> (i32, i32, i32) {
    let top = src.row_n::<9>(-1, -1);
    let col = read_col1::<8, _>(src, 0, -1);
    let mut left = [0u8; 9];
    left[0] = top[0];
    left[1..].copy_from_slice(&col);
    let mut top_sum: i32 = 0;
    let mut left_sum: i32 = 0;
    for i in 0..4usize {
        top_sum += (i as i32 + 1) * (top[5 + i] as i32 - top[3 - i] as i32);
        left_sum += (i as i32 + 1) * (left[5 + i] as i32 - left[3 - i] as i32);
    }
    let lt_shift = (left[8] as i32 + top[8] as i32) << 4;
    (
        (17 * top_sum + 16) >> 5,
        (17 * left_sum + 16) >> 5,
        lt_shift,
    )
}

simd_entry! {
    /// The 8x8 chroma plane fill, from the three coefficients.
    #[target_feature(enable = "sse2")]
    fn chroma_plane_fill<O: PredOut>(
        out: &mut O,
        top_shift: i32,
        left_shift: i32,
        lt_shift: i32,
        _token: Sse2Token,
    ) {
        let mul_b = _mm_setr_epi16(-3, -2, -1, 0, 1, 2, 3, 4);
        let b_vec = _mm_set1_epi16(top_shift as i16);
        let c_vec = _mm_set1_epi16(left_shift as i16);
        let mut s_vec = _mm_set1_epi16((lt_shift + 16 - 3 * left_shift) as i16);
        let term = _mm_mullo_epi16(b_vec, mul_b);

        for dy in 0..8 {
            let row_w = _mm_srai_epi16(_mm_add_epi16(term, s_vec), 5);
            let row_b = _mm_packus_epi16(row_w, row_w);
            out.put(dy, &to8(row_b));
            s_vec = _mm_add_epi16(s_vec, c_vec);
        }
    }
}

/// Fills every row with the neighbour row above — the V predictors, at either width.
#[inline(always)]
fn pred_v<const N: usize, S: RefSamples, O: PredOut>(src: &S, out: &mut O, rows: usize) {
    let top = src.row_n::<N>(-1, 0);
    fill_rows(out, rows, &top);
}

/// Fills each row with its own left neighbour — the H predictors, at either width.
#[inline(always)]
fn pred_h<const N: usize, S: RefSamples, O: PredOut>(src: &S, out: &mut O, rows: usize) {
    for dy in 0..rows {
        out.put(dy, &[src.at(-1, dy as isize); N]);
    }
}

/// Vertical 16x16 predictor for packed candidate buffer (encoder).
#[inline]
pub fn enc_i16x16_luma_pred_v(pred: &mut [u8; 256], rec: &RecCursor<'_>) {
    pred_v::<16, _, _>(rec, &mut Packed::<16>(pred), 16)
}

/// Vertical 16x16 predictor in place (decoder).
#[inline]
pub fn dec_i16x16_luma_pred_v(pred: &mut PlaneCursorMut<'_>) {
    let top = pred.row_n::<16>(-1, 0);
    fill_rows(pred, 16, &top)
}

/// Horizontal 16x16 predictor for packed candidate buffer (encoder).
#[inline]
pub fn enc_i16x16_luma_pred_h(pred: &mut [u8; 256], rec: &RecCursor<'_>) {
    pred_h::<16, _, _>(rec, &mut Packed::<16>(pred), 16)
}

/// Horizontal 16x16 predictor in place (decoder).
#[inline]
pub fn dec_i16x16_luma_pred_h(pred: &mut PlaneCursorMut<'_>) {
    for dy in 0..16 {
        let v = pred.at(-1, dy as isize);
        pred.put(dy, &[v; 16]);
    }
}

/// DC 16x16 predictor for packed candidate buffer (encoder).
#[inline]
pub fn enc_i16x16_luma_pred_dc(pred: &mut [u8; 256], rec: &RecCursor<'_>) {
    let mean = i16x16_dc_mean::<_, true, true>(rec, sse2_token());
    fill_rows(&mut Packed::<16>(pred), 16, &[mean; 16])
}

/// DC 16x16 predictor in place (decoder).
#[inline]
pub fn dec_i16x16_luma_pred_dc(pred: &mut PlaneCursorMut<'_>) {
    let mean = i16x16_dc_mean::<_, true, true>(pred, sse2_token());
    fill_rows(pred, 16, &[mean; 16])
}

/// DC Top 16x16 predictor (decoder).
#[inline]
pub fn dec_i16x16_luma_pred_dc_top(pred: &mut PlaneCursorMut<'_>) {
    let mean = i16x16_dc_mean::<_, true, false>(pred, sse2_token());
    fill_rows(pred, 16, &[mean; 16])
}

/// DC Left 16x16 predictor (decoder).
#[inline]
pub fn dec_i16x16_luma_pred_dc_left(pred: &mut PlaneCursorMut<'_>) {
    let mean = i16x16_dc_mean::<_, false, true>(pred, sse2_token());
    fill_rows(pred, 16, &[mean; 16])
}

/// DC NA 16x16 predictor (decoder).
#[inline]
pub fn dec_i16x16_luma_pred_dc_na(pred: &mut PlaneCursorMut<'_>) {
    fill_rows(pred, 16, &[0x80u8; 16])
}

/// Plane 16x16 predictor for packed candidate buffer (encoder).
#[inline]
pub fn enc_i16x16_luma_pred_plane(pred: &mut [u8; 256], rec: &RecCursor<'_>) {
    let (top_shift, left_shift, lt_shift) = i16x16_plane_coeffs(rec);
    i16x16_plane_fill(
        &mut Packed::<16>(pred),
        top_shift,
        left_shift,
        lt_shift,
        sse2_token(),
    )
}

/// Plane 16x16 predictor in place (decoder).
#[inline]
pub fn dec_i16x16_luma_pred_plane(pred: &mut PlaneCursorMut<'_>) {
    let (top_shift, left_shift, lt_shift) = i16x16_plane_coeffs(pred);
    i16x16_plane_fill(pred, top_shift, left_shift, lt_shift, sse2_token())
}

// ============================================================================
// 8x8 Chroma Intra Prediction (SSE2)
// ============================================================================

/// Vertical Chroma 8x8 predictor for packed candidate buffer (encoder).
#[inline]
pub fn enc_chroma_pred_v(pred: &mut [u8; 64], rec: &RecCursor<'_>) {
    pred_v::<8, _, _>(rec, &mut Packed::<8>(pred), 8)
}

/// Vertical Chroma 8x8 predictor in place (decoder).
#[inline]
pub fn dec_chroma_pred_v(pred: &mut PlaneCursorMut<'_>) {
    let top = pred.row_n::<8>(-1, 0);
    fill_rows(pred, 8, &top)
}

/// Horizontal Chroma 8x8 predictor for packed candidate buffer (encoder).
#[inline]
pub fn enc_chroma_pred_h(pred: &mut [u8; 64], rec: &RecCursor<'_>) {
    pred_h::<8, _, _>(rec, &mut Packed::<8>(pred), 8)
}

/// Horizontal Chroma 8x8 predictor in place (decoder).
#[inline]
pub fn dec_chroma_pred_h(pred: &mut PlaneCursorMut<'_>) {
    for dy in 0..8 {
        let v = pred.at(-1, dy as isize);
        pred.put(dy, &[v; 8]);
    }
}

/// DC Chroma 8x8 predictor for packed candidate buffer (encoder).
#[inline]
pub fn enc_chroma_pred_dc(pred: &mut [u8; 64], rec: &RecCursor<'_>) {
    let (top, bot) = chroma_dc_rows(rec);
    let out = &mut Packed::<8>(pred);
    for dy in 0..4 {
        out.put(dy, &top);
    }
    for dy in 4..8 {
        out.put(dy, &bot);
    }
}

/// DC Chroma 8x8 predictor in place (decoder).
#[inline]
pub fn dec_chroma_pred_dc(pred: &mut PlaneCursorMut<'_>) {
    let (top, bot) = chroma_dc_rows(pred);
    for dy in 0..4 {
        pred.put(dy, &top);
    }
    for dy in 4..8 {
        pred.put(dy, &bot);
    }
}

simd_entry! {
    /// The row every line of the DC-top chroma predictor is filled with: the two
    /// 4-sample means of the neighbour row above, one `_mm_sad_epu8` over the pair.
    ///
    /// C++: `WelsIChromaPredDcTop_c` / `WelsDecoderIChromaPredDcTop_sse2`.
    #[target_feature(enable = "sse2")]
    fn chroma_dc_top_row<S: RefSamples>(src: &S, _token: Sse2Token) -> [u8; 8] {
        let top = src.row_n::<8>(-1, 0);
        // The halves in separate quadwords, so one `psadbw` sums each on its own.
        let halves = _mm_setr_epi32(
            i32::from_ne_bytes(top[..4].try_into().expect("top left half")),
            0,
            i32::from_ne_bytes(top[4..].try_into().expect("top right half")),
            0,
        );
        let sad = _mm_sad_epu8(halves, _mm_setzero_si128());
        let m1 = ((_mm_cvtsi128_si32(sad) + 2) >> 2) as u8;
        let m2 = ((_mm_extract_epi16(sad, 4) + 2) >> 2) as u8;
        [m1, m1, m1, m1, m2, m2, m2, m2]
    }
}

/// DC Top Chroma 8x8 predictor in place (decoder).
#[inline]
pub fn dec_chroma_pred_dc_top(pred: &mut PlaneCursorMut<'_>) {
    let row = chroma_dc_top_row(pred, sse2_token());
    fill_rows(pred, 8, &row)
}

/// Plane Chroma 8x8 predictor for packed candidate buffer (encoder).
#[inline]
pub fn enc_chroma_pred_plane(pred: &mut [u8; 64], rec: &RecCursor<'_>) {
    let (top_shift, left_shift, lt_shift) = chroma_plane_coeffs(rec);
    chroma_plane_fill(
        &mut Packed::<8>(pred),
        top_shift,
        left_shift,
        lt_shift,
        sse2_token(),
    )
}

/// Plane Chroma 8x8 predictor in place (decoder).
#[inline]
pub fn dec_chroma_pred_plane(pred: &mut PlaneCursorMut<'_>) {
    let (top_shift, left_shift, lt_shift) = chroma_plane_coeffs(pred);
    chroma_plane_fill(pred, top_shift, left_shift, lt_shift, sse2_token())
}

// ============================================================================
// 4x4 Luma Intra Prediction (SSE2)
// ============================================================================

simd_entry! {
    #[target_feature(enable = "sse2")]
    fn i4x4_v(pred: &mut [u8; 16], top: [u8; 4], _token: Sse2Token) {
        let t_u32 = i32::from_ne_bytes(top);
        let v = _mm_set1_epi32(t_u32);
        st16(pred, v);
    }

    #[target_feature(enable = "sse2")]
    fn i4x4_h(pred: &mut [u8; 16], l0: i8, l1: i8, l2: i8, l3: i8, _token: Sse2Token) {
        let v = _mm_setr_epi8(
            l0, l0, l0, l0, l1, l1, l1, l1, l2, l2, l2, l2, l3, l3, l3, l3,
        );
        st16(pred, v);
    }

    #[target_feature(enable = "sse2")]
    fn i4x4_dc(pred: &mut [u8; 16], mean: u8, _token: Sse2Token) {
        let v = _mm_set1_epi8(mean as i8);
        st16(pred, v);
    }
}

/// Vertical 4x4 predictor for packed candidate buffer (encoder).
#[inline]
pub fn enc_i4x4_luma_pred_v(pred: &mut [u8; 16], rec: &RecCursor<'_>) {
    let top = rec.row_n::<4>(-1, 0);
    i4x4_v(pred, top, sse2_token());
}

/// Vertical 4x4 predictor in place (decoder).
#[inline]
pub fn dec_i4x4_luma_pred_v(pred: &mut PlaneCursorMut<'_>) {
    let top: [u8; 4] = pred.row(-1, 0, 4).try_into().unwrap();
    let t_u32 = u32::from_ne_bytes(top);
    for dy in 0..4 {
        let row: &mut [u8; 4] = pred.row_mut(dy, 0, 4).try_into().unwrap();
        *row = t_u32.to_ne_bytes();
    }
}

/// Horizontal 4x4 predictor for packed candidate buffer (encoder).
#[inline]
pub fn enc_i4x4_luma_pred_h(pred: &mut [u8; 16], rec: &RecCursor<'_>) {
    let l0 = rec.at(-1, 0) as i8;
    let l1 = rec.at(-1, 1) as i8;
    let l2 = rec.at(-1, 2) as i8;
    let l3 = rec.at(-1, 3) as i8;
    i4x4_h(pred, l0, l1, l2, l3, sse2_token());
}

/// Horizontal 4x4 predictor in place (decoder).
#[inline]
pub fn dec_i4x4_luma_pred_h(pred: &mut PlaneCursorMut<'_>) {
    for dy in 0..4 {
        let val = pred.at(-1, dy);
        let row: &mut [u8; 4] = pred.row_mut(dy, 0, 4).try_into().unwrap();
        *row = [val; 4];
    }
}

/// DC 4x4 predictor for packed candidate buffer (encoder).
#[inline]
pub fn enc_i4x4_luma_pred_dc(pred: &mut [u8; 16], rec: &RecCursor<'_>) {
    let top = rec.row_n::<4>(-1, 0);
    let mut sum = 4i32;
    for y in 0..4 {
        sum += rec.at(-1, y as isize) as i32;
        sum += top[y] as i32;
    }
    let mean = (sum >> 3) as u8;
    i4x4_dc(pred, mean, sse2_token());
}

/// DC 4x4 predictor in place (decoder).
#[inline]
pub fn dec_i4x4_luma_pred_dc(pred: &mut PlaneCursorMut<'_>) {
    let top: [u8; 4] = pred.row(-1, 0, 4).try_into().unwrap();
    let mut sum = 4i32;
    for y in 0..4 {
        sum += pred.at(-1, y as isize) as i32;
        sum += top[y] as i32;
    }
    let mean = (sum >> 3) as u8;
    for dy in 0..4 {
        let row: &mut [u8; 4] = pred.row_mut(dy, 0, 4).try_into().unwrap();
        *row = [mean; 4];
    }
}

/// Four packed rows of four samples into a destination — the 4x4 predictors compute
/// their whole block as one 16-byte vector, which the encoder stores in one go and the
/// decoder writes back a row at a time.
#[inline(always)]
fn put4<O: PredOut>(out: &mut O, rows: &[u8; 16]) {
    for dy in 0..4 {
        let row: &[u8; 4] = rows[dy * 4..][..4].try_into().expect("row");
        out.put(dy, row);
    }
}

simd_entry! {
    /// Diagonal Down-Left (DDL) 4x4 block, row-major.
    ///
    /// C++: `WelsI4x4LumaPredDDL_c` / `WelsDecoderI4x4LumaPredDDL_mmx`.
    #[target_feature(enable = "sse2")]
    fn i4x4_ddl<S: RefSamples>(src: &S, _token: Sse2Token) -> [u8; 16] {
        let top = src.row_n::<8>(-1, 0);
        let t = |i: usize| top[i] as i32;
        let ddl0 = ((2 + t(0) + t(2) + (t(1) << 1)) >> 2) as u8;
        let ddl1 = ((2 + t(1) + t(3) + (t(2) << 1)) >> 2) as u8;
        let ddl2 = ((2 + t(2) + t(4) + (t(3) << 1)) >> 2) as u8;
        let ddl3 = ((2 + t(3) + t(5) + (t(4) << 1)) >> 2) as u8;
        let ddl4 = ((2 + t(4) + t(6) + (t(5) << 1)) >> 2) as u8;
        let ddl5 = ((2 + t(5) + t(7) + (t(6) << 1)) >> 2) as u8;
        let ddl6 = ((2 + t(6) + t(7) + (t(7) << 1)) >> 2) as u8;

        let v = _mm_setr_epi8(
            ddl0 as i8, ddl1 as i8, ddl2 as i8, ddl3 as i8, ddl1 as i8, ddl2 as i8, ddl3 as i8,
            ddl4 as i8, ddl2 as i8, ddl3 as i8, ddl4 as i8, ddl5 as i8, ddl3 as i8, ddl4 as i8,
            ddl5 as i8, ddl6 as i8,
        );
        to16(v)
    }

    /// Diagonal Down-Right (DDR) 4x4 block, row-major.
    ///
    /// C++: `WelsI4x4LumaPredDDR_c` / `WelsDecoderI4x4LumaPredDDR_mmx`.
    #[target_feature(enable = "sse2")]
    fn i4x4_ddr<S: RefSamples>(src: &S, _token: Sse2Token) -> [u8; 16] {
        let lt = src.at(-1, -1) as i32;
        let l0 = src.at(-1, 0) as i32;
        let l1 = src.at(-1, 1) as i32;
        let l2 = src.at(-1, 2) as i32;
        let l3 = src.at(-1, 3) as i32;
        let top = src.row_n::<4>(-1, 0);
        let (t0, t1, t2, t3) = (top[0] as i32, top[1] as i32, top[2] as i32, top[3] as i32);
        let tl0 = 1 + lt + l0;
        let lt0 = 1 + lt + t0;
        let t01 = 1 + t0 + t1;
        let t12 = 1 + t1 + t2;
        let t23 = 1 + t2 + t3;
        let l01 = 1 + l0 + l1;
        let l12 = 1 + l1 + l2;
        let l23 = 1 + l2 + l3;
        let ddr0 = ((tl0 + lt0) >> 2) as u8;
        let ddr1 = ((lt0 + t01) >> 2) as u8;
        let ddr2 = ((t01 + t12) >> 2) as u8;
        let ddr3 = ((t12 + t23) >> 2) as u8;
        let ddr4 = ((tl0 + l01) >> 2) as u8;
        let ddr5 = ((l01 + l12) >> 2) as u8;
        let ddr6 = ((l12 + l23) >> 2) as u8;

        let v = _mm_setr_epi8(
            ddr0 as i8, ddr1 as i8, ddr2 as i8, ddr3 as i8, ddr4 as i8, ddr0 as i8, ddr1 as i8,
            ddr2 as i8, ddr5 as i8, ddr4 as i8, ddr0 as i8, ddr1 as i8, ddr6 as i8, ddr5 as i8,
            ddr4 as i8, ddr0 as i8,
        );
        to16(v)
    }

    /// Vertical Right (VR) 4x4 block, row-major.
    ///
    /// C++: `WelsI4x4LumaPredVR_c` / `WelsDecoderI4x4LumaPredVR_mmx`.
    #[target_feature(enable = "sse2")]
    fn i4x4_vr<S: RefSamples>(src: &S, _token: Sse2Token) -> [u8; 16] {
        let lt = src.at(-1, -1) as i32;
        let l0 = src.at(-1, 0) as i32;
        let l1 = src.at(-1, 1) as i32;
        let l2 = src.at(-1, 2) as i32;
        let top = src.row_n::<4>(-1, 0);
        let (t0, t1, t2, t3) = (top[0] as i32, top[1] as i32, top[2] as i32, top[3] as i32);
        let vr0 = ((1 + lt + t0) >> 1) as u8;
        let vr1 = ((1 + t0 + t1) >> 1) as u8;
        let vr2 = ((1 + t1 + t2) >> 1) as u8;
        let vr3 = ((1 + t2 + t3) >> 1) as u8;
        let vr4 = ((2 + l0 + (lt << 1) + t0) >> 2) as u8;
        let vr5 = ((2 + lt + (t0 << 1) + t1) >> 2) as u8;
        let vr6 = ((2 + t0 + (t1 << 1) + t2) >> 2) as u8;
        let vr7 = ((2 + t1 + (t2 << 1) + t3) >> 2) as u8;
        let vr8 = ((2 + lt + (l0 << 1) + l1) >> 2) as u8;
        let vr9 = ((2 + l0 + (l1 << 1) + l2) >> 2) as u8;

        let v = _mm_setr_epi8(
            vr0 as i8, vr1 as i8, vr2 as i8, vr3 as i8, vr4 as i8, vr5 as i8, vr6 as i8, vr7 as i8,
            vr8 as i8, vr0 as i8, vr1 as i8, vr2 as i8, vr9 as i8, vr4 as i8, vr5 as i8, vr6 as i8,
        );
        to16(v)
    }

    /// Horizontal Down (HD) 4x4 block, row-major.
    ///
    /// C++: `WelsI4x4LumaPredHD_c` / `WelsDecoderI4x4LumaPredHD_mmx`.
    #[target_feature(enable = "sse2")]
    fn i4x4_hd<S: RefSamples>(src: &S, _token: Sse2Token) -> [u8; 16] {
        let lt = src.at(-1, -1) as i32;
        let l0 = src.at(-1, 0) as i32;
        let l1 = src.at(-1, 1) as i32;
        let l2 = src.at(-1, 2) as i32;
        let l3 = src.at(-1, 3) as i32;
        let top = src.row_n::<4>(-1, 0);
        let (t0, t1, t2) = (top[0] as i32, top[1] as i32, top[2] as i32);
        let hd0 = ((1 + lt + l0) >> 1) as u8;
        let hd1 = ((2 + lt + (l0 << 1) + l1) >> 2) as u8;
        let hd2 = ((1 + l0 + l1) >> 1) as u8;
        let hd3 = ((2 + l0 + (l1 << 1) + l2) >> 2) as u8;
        let hd4 = ((1 + l1 + l2) >> 1) as u8;
        let hd5 = ((2 + l1 + (l2 << 1) + l3) >> 2) as u8;
        let hd6 = ((1 + l2 + l3) >> 1) as u8;
        let hd7 = ((2 + l0 + (lt << 1) + t0) >> 2) as u8;
        let hd8 = ((2 + lt + (t0 << 1) + t1) >> 2) as u8;
        let hd9 = ((2 + t0 + (t1 << 1) + t2) >> 2) as u8;

        let v = _mm_setr_epi8(
            hd0 as i8, hd7 as i8, hd8 as i8, hd9 as i8, hd2 as i8, hd1 as i8, hd0 as i8, hd7 as i8,
            hd4 as i8, hd3 as i8, hd2 as i8, hd1 as i8, hd6 as i8, hd5 as i8, hd4 as i8, hd3 as i8,
        );
        to16(v)
    }

    /// Vertical Left (VL) 4x4 block, row-major.
    ///
    /// C++: `WelsI4x4LumaPredVL_c` / `WelsDecoderI4x4LumaPredVL_mmx`.
    #[target_feature(enable = "sse2")]
    fn i4x4_vl<S: RefSamples>(src: &S, _token: Sse2Token) -> [u8; 16] {
        let top = src.row_n::<7>(-1, 0);
        let t = |i: usize| top[i] as i32;
        let vl0 = ((1 + t(0) + t(1)) >> 1) as u8;
        let vl1 = ((1 + t(1) + t(2)) >> 1) as u8;
        let vl2 = ((1 + t(2) + t(3)) >> 1) as u8;
        let vl3 = ((1 + t(3) + t(4)) >> 1) as u8;
        let vl4 = ((1 + t(4) + t(5)) >> 1) as u8;
        let vl5 = ((2 + t(0) + (t(1) << 1) + t(2)) >> 2) as u8;
        let vl6 = ((2 + t(1) + (t(2) << 1) + t(3)) >> 2) as u8;
        let vl7 = ((2 + t(2) + (t(3) << 1) + t(4)) >> 2) as u8;
        let vl8 = ((2 + t(3) + (t(4) << 1) + t(5)) >> 2) as u8;
        let vl9 = ((2 + t(4) + (t(5) << 1) + t(6)) >> 2) as u8;

        let v = _mm_setr_epi8(
            vl0 as i8, vl1 as i8, vl2 as i8, vl3 as i8, vl5 as i8, vl6 as i8, vl7 as i8, vl8 as i8,
            vl1 as i8, vl2 as i8, vl3 as i8, vl4 as i8, vl6 as i8, vl7 as i8, vl8 as i8, vl9 as i8,
        );
        to16(v)
    }

    /// Horizontal Up (HU) 4x4 block, row-major.
    ///
    /// C++: `WelsI4x4LumaPredHU_c` / `WelsDecoderI4x4LumaPredHU_mmx`.
    #[target_feature(enable = "sse2")]
    fn i4x4_hu<S: RefSamples>(src: &S, _token: Sse2Token) -> [u8; 16] {
        let l0 = src.at(-1, 0) as i32;
        let l1 = src.at(-1, 1) as i32;
        let l2 = src.at(-1, 2) as i32;
        let l3 = src.at(-1, 3) as i32;
        let l01 = 1 + l0 + l1;
        let l12 = 1 + l1 + l2;
        let l23 = 1 + l2 + l3;
        let hu0 = (l01 >> 1) as u8;
        let hu1 = ((l01 + l12) >> 2) as u8;
        let hu2 = (l12 >> 1) as u8;
        let hu3 = ((l12 + l23) >> 2) as u8;
        let hu4 = (l23 >> 1) as u8;
        let hu5 = ((1 + l23 + (l3 << 1)) >> 2) as u8;

        let v = _mm_setr_epi8(
            hu0 as i8, hu1 as i8, hu2 as i8, hu3 as i8, hu2 as i8, hu3 as i8, hu4 as i8, hu5 as i8,
            hu4 as i8, hu5 as i8, l3 as i8, l3 as i8, l3 as i8, l3 as i8, l3 as i8, l3 as i8,
        );
        to16(v)
    }
}

/// Diagonal Down-Left (DDL) 4x4 predictor for packed candidate buffer (encoder).
#[inline]
pub fn enc_i4x4_luma_pred_ddl(pred: &mut [u8; 16], rec: &RecCursor<'_>) {
    *pred = i4x4_ddl(rec, sse2_token());
}

/// Diagonal Down-Left (DDL) 4x4 predictor in place (decoder).
#[inline]
pub fn dec_i4x4_luma_pred_ddl(pred: &mut PlaneCursorMut<'_>) {
    let rows = i4x4_ddl(&*pred, sse2_token());
    put4(pred, &rows)
}

/// Diagonal Down-Right (DDR) 4x4 predictor for packed candidate buffer (encoder).
#[inline]
pub fn enc_i4x4_luma_pred_ddr(pred: &mut [u8; 16], rec: &RecCursor<'_>) {
    *pred = i4x4_ddr(rec, sse2_token());
}

/// Diagonal Down-Right (DDR) 4x4 predictor in place (decoder).
#[inline]
pub fn dec_i4x4_luma_pred_ddr(pred: &mut PlaneCursorMut<'_>) {
    let rows = i4x4_ddr(&*pred, sse2_token());
    put4(pred, &rows)
}

/// Vertical Right (VR) 4x4 predictor for packed candidate buffer (encoder).
#[inline]
pub fn enc_i4x4_luma_pred_vr(pred: &mut [u8; 16], rec: &RecCursor<'_>) {
    *pred = i4x4_vr(rec, sse2_token());
}

/// Vertical Right (VR) 4x4 predictor in place (decoder).
#[inline]
pub fn dec_i4x4_luma_pred_vr(pred: &mut PlaneCursorMut<'_>) {
    let rows = i4x4_vr(&*pred, sse2_token());
    put4(pred, &rows)
}

/// Horizontal Down (HD) 4x4 predictor for packed candidate buffer (encoder).
#[inline]
pub fn enc_i4x4_luma_pred_hd(pred: &mut [u8; 16], rec: &RecCursor<'_>) {
    *pred = i4x4_hd(rec, sse2_token());
}

/// Horizontal Down (HD) 4x4 predictor in place (decoder).
#[inline]
pub fn dec_i4x4_luma_pred_hd(pred: &mut PlaneCursorMut<'_>) {
    let rows = i4x4_hd(&*pred, sse2_token());
    put4(pred, &rows)
}

/// Vertical Left (VL) 4x4 predictor for packed candidate buffer (encoder).
#[inline]
pub fn enc_i4x4_luma_pred_vl(pred: &mut [u8; 16], rec: &RecCursor<'_>) {
    *pred = i4x4_vl(rec, sse2_token());
}

/// Vertical Left (VL) 4x4 predictor in place (decoder).
#[inline]
pub fn dec_i4x4_luma_pred_vl(pred: &mut PlaneCursorMut<'_>) {
    let rows = i4x4_vl(&*pred, sse2_token());
    put4(pred, &rows)
}

/// Horizontal Up (HU) 4x4 predictor for packed candidate buffer (encoder).
#[inline]
pub fn enc_i4x4_luma_pred_hu(pred: &mut [u8; 16], rec: &RecCursor<'_>) {
    *pred = i4x4_hu(rec, sse2_token());
}

/// Horizontal Up (HU) 4x4 predictor in place (decoder).
#[inline]
pub fn dec_i4x4_luma_pred_hu(pred: &mut PlaneCursorMut<'_>) {
    let rows = i4x4_hu(&*pred, sse2_token());
    put4(pred, &rows)
}

#[target_feature(enable = "avx2")]
#[inline]
fn reduce256(acc: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(acc);
    let hi = _mm256_extracti128_si256(acc, 1);
    let sum128 = _mm_add_epi64(lo, hi);
    let sum_hi = _mm_srli_si128(sum128, 8);
    _mm_cvtsi128_si32(_mm_add_epi32(sum128, sum_hi))
}

#[cold]
#[inline(never)]
fn combined3_sad_narrow(
    pred: &mut [u8; 256],
    rec: &RecCursor<'_>,
    enc: &RecCursor<'_>,
    lambda: i32,
) -> (u8, i32) {
    let mut pred_v = [0u8; 256];
    let mut pred_h = [0u8; 256];
    let mut pred_dc = [0u8; 256];
    crate::encoder::get_intra_predictor::WelsI16x16LumaPredV_c(&mut pred_v, rec);
    crate::encoder::get_intra_predictor::WelsI16x16LumaPredH_c(&mut pred_h, rec);
    crate::encoder::get_intra_predictor::WelsI16x16LumaPredDc_c(&mut pred_dc, rec);
    let sad_v = crate::common::sad_common::sample_sad::<16, 16, _>(
        &RecCursor::over_owned(&mut pred_v, 0, 16),
        enc,
    );
    let sad_h = crate::common::sad_common::sample_sad::<16, 16, _>(
        &RecCursor::over_owned(&mut pred_h, 0, 16),
        enc,
    );
    let sad_dc = crate::common::sad_common::sample_sad::<16, 16, _>(
        &RecCursor::over_owned(&mut pred_dc, 0, 16),
        enc,
    );
    let cost_v = sad_v + lambda;
    let cost_h = sad_h + lambda * 3;
    let cost_dc = sad_dc + lambda * 3;
    if cost_dc < cost_h && cost_dc < cost_v {
        *pred = pred_dc;
        (2u8, cost_dc)
    } else if cost_h < cost_v {
        *pred = pred_h;
        (1u8, cost_h)
    } else {
        *pred = pred_v;
        (0u8, cost_v)
    }
}

simd_entry! {
    #[target_feature(enable = "avx2")]
    fn intra_16x16_combined3_sad_avx2(
        pred: &mut [u8; 256],
        rec: &RecCursor<'_>,
        enc: &RecCursor<'_>,
        lambda: i32,
        _token: Avx2Token,
    ) -> (u8, i32) {
        let top = rec.row_n::<16>(-1, 0);
        let left_span = rec.span::<1, 16>(0, -1);
        let enc_span = enc.span::<16, 16>(0, 0);
        let (Some((mut lh, lt)), Some((mut eh, et))) = (
            left_span.split_rows::<1, 16>(),
            enc_span.split_rows::<16, 16>(),
        ) else {
            return combined3_sad_narrow(pred, rec, enc, lambda);
        };

        let v_vec128 = ld16(&top);
        let sad_top = _mm_sad_epu8(v_vec128, _mm_setzero_si128());
        let sum_top = _mm_cvtsi128_si32(sad_top) + _mm_extract_epi16(sad_top, 4);

        let mut left = [0u8; 16];
        for dst in &mut left[..15] {
            *dst = lh.next().expect("left row")[0];
        }
        left[15] = lt()[0];
        let sad_left = _mm_sad_epu8(ld16(&left), _mm_setzero_si128());
        let sum_left = _mm_cvtsi128_si32(sad_left) + _mm_extract_epi16(sad_left, 4);

        let dc_val = ((16 + sum_top + sum_left) >> 5) as u8;
        let dc_vec256 = _mm256_set1_epi8(dc_val as i8);
        let v_vec256 = _mm256_broadcastsi128_si256(v_vec128);

        let mut acc_v = _mm256_setzero_si256();
        let mut acc_h = _mm256_setzero_si256();
        let mut acc_dc = _mm256_setzero_si256();

        for step in 0..7 {
            let y = step * 2;
            let enc0 = ld16(&eh.next().expect("enc row 0"));
            let enc1 = ld16(&eh.next().expect("enc row 1"));
            let enc256 = _mm256_set_m128i(enc1, enc0);

            let h0 = _mm_set1_epi8(left[y] as i8);
            let h1 = _mm_set1_epi8(left[y + 1] as i8);
            let h256 = _mm256_set_m128i(h1, h0);

            acc_v = _mm256_add_epi64(acc_v, _mm256_sad_epu8(enc256, v_vec256));
            acc_h = _mm256_add_epi64(acc_h, _mm256_sad_epu8(enc256, h256));
            acc_dc = _mm256_add_epi64(acc_dc, _mm256_sad_epu8(enc256, dc_vec256));
        }
        {
            let enc0 = ld16(&eh.next().expect("enc row 14"));
            let enc1 = ld16(&et());
            let enc256 = _mm256_set_m128i(enc1, enc0);
            let h0 = _mm_set1_epi8(left[14] as i8);
            let h1 = _mm_set1_epi8(left[15] as i8);
            let h256 = _mm256_set_m128i(h1, h0);
            acc_v = _mm256_add_epi64(acc_v, _mm256_sad_epu8(enc256, v_vec256));
            acc_h = _mm256_add_epi64(acc_h, _mm256_sad_epu8(enc256, h256));
            acc_dc = _mm256_add_epi64(acc_dc, _mm256_sad_epu8(enc256, dc_vec256));
        }

        let sad_v = reduce256(acc_v);
        let sad_h = reduce256(acc_h);
        let sad_dc = reduce256(acc_dc);

        let cost_v = sad_v + lambda;
        let cost_h = sad_h + lambda * 3;
        let cost_dc = sad_dc + lambda * 3;

        let (best_mode, best_cost) = if cost_dc < cost_h && cost_dc < cost_v {
            (2u8, cost_dc)
        } else if cost_h < cost_v {
            (1u8, cost_h)
        } else {
            (0u8, cost_v)
        };

        match best_mode {
            0 => {
                for chunk in pred.as_chunks_mut::<32>().0 {
                    st32(chunk, v_vec256);
                }
            }
            1 => {
                for (y, chunk) in pred.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                    st16(chunk, _mm_set1_epi8(left[y] as i8));
                }
            }
            _ => {
                for chunk in pred.as_chunks_mut::<32>().0 {
                    st32(chunk, dc_vec256);
                }
            }
        }

        (best_mode, best_cost)
    }

    #[target_feature(enable = "sse2")]
    fn intra_16x16_combined3_sad_sse2(
        pred: &mut [u8; 256],
        rec: &RecCursor<'_>,
        enc: &RecCursor<'_>,
        lambda: i32,
        _token: Sse2Token,
    ) -> (u8, i32) {
        let top = rec.row_n::<16>(-1, 0);
        let left_span = rec.span::<1, 16>(0, -1);
        let enc_span = enc.span::<16, 16>(0, 0);
        let (Some((mut lh, lt)), Some((eh, et))) = (
            left_span.split_rows::<1, 16>(),
            enc_span.split_rows::<16, 16>(),
        ) else {
            return combined3_sad_narrow(pred, rec, enc, lambda);
        };

        let v_vec = ld16(&top);
        let sad_top = _mm_sad_epu8(v_vec, _mm_setzero_si128());
        let sum_top = _mm_cvtsi128_si32(sad_top) + _mm_extract_epi16(sad_top, 4);

        let mut left = [0u8; 16];
        for dst in &mut left[..15] {
            *dst = lh.next().expect("left row")[0];
        }
        left[15] = lt()[0];
        let sad_left = _mm_sad_epu8(ld16(&left), _mm_setzero_si128());
        let sum_left = _mm_cvtsi128_si32(sad_left) + _mm_extract_epi16(sad_left, 4);

        let dc_val = ((16 + sum_top + sum_left) >> 5) as u8;
        let dc_vec = _mm_set1_epi8(dc_val as i8);

        let mut acc_v = _mm_setzero_si128();
        let mut acc_h = _mm_setzero_si128();
        let mut acc_dc = _mm_setzero_si128();

        for (y, er) in eh.enumerate() {
            let enc_vec = ld16(&er);
            let h_vec = _mm_set1_epi8(left[y] as i8);
            acc_v = _mm_add_epi64(acc_v, _mm_sad_epu8(enc_vec, v_vec));
            acc_h = _mm_add_epi64(acc_h, _mm_sad_epu8(enc_vec, h_vec));
            acc_dc = _mm_add_epi64(acc_dc, _mm_sad_epu8(enc_vec, dc_vec));
        }
        {
            let enc_vec = ld16(&et());
            let h_vec = _mm_set1_epi8(left[15] as i8);
            acc_v = _mm_add_epi64(acc_v, _mm_sad_epu8(enc_vec, v_vec));
            acc_h = _mm_add_epi64(acc_h, _mm_sad_epu8(enc_vec, h_vec));
            acc_dc = _mm_add_epi64(acc_dc, _mm_sad_epu8(enc_vec, dc_vec));
        }

        let hi_v = _mm_srli_si128(acc_v, 8);
        let sum_v = _mm_add_epi32(acc_v, hi_v);
        let sad_v = _mm_cvtsi128_si32(sum_v);

        let hi_h = _mm_srli_si128(acc_h, 8);
        let sum_h = _mm_add_epi32(acc_h, hi_h);
        let sad_h = _mm_cvtsi128_si32(sum_h);

        let hi_dc = _mm_srli_si128(acc_dc, 8);
        let sum_dc = _mm_add_epi32(acc_dc, hi_dc);
        let sad_dc = _mm_cvtsi128_si32(sum_dc);

        let cost_v = sad_v + lambda;
        let cost_h = sad_h + lambda * 3;
        let cost_dc = sad_dc + lambda * 3;

        let (best_mode, best_cost) = if cost_dc < cost_h && cost_dc < cost_v {
            (2u8, cost_dc)
        } else if cost_h < cost_v {
            (1u8, cost_h)
        } else {
            (0u8, cost_v)
        };

        match best_mode {
            0 => {
                for chunk in pred.as_chunks_mut::<16>().0 {
                    st16(chunk, v_vec);
                }
            }
            1 => {
                for (y, chunk) in pred.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                    st16(chunk, _mm_set1_epi8(left[y] as i8));
                }
            }
            _ => {
                for chunk in pred.as_chunks_mut::<16>().0 {
                    st16(chunk, dc_vec);
                }
            }
        }

        (best_mode, best_cost)
    }
}

#[cold]
#[inline(never)]
fn intra_16x16_combined3_sad_fallback(
    pred: &mut [u8; 256],
    rec: &RecCursor<'_>,
    enc: &RecCursor<'_>,
    lambda: i32,
) -> (u8, i32) {
    if let Some(token) = Avx2Token::try_new_enabled() {
        return intra_16x16_combined3_sad_avx2(pred, rec, enc, lambda, token);
    }
    intra_16x16_combined3_sad_sse2(pred, rec, enc, lambda, sse2_token())
}

/// Combined 3-mode (Vertical, Horizontal, DC) 16x16 Intra Prediction and SAD evaluation.
///
/// Evaluates modes 0 (V), 1 (H), and 2 (DC) simultaneously in a single streaming pass
/// over the source and reference macroblock lines, avoiding intermediate prediction buffers
/// and multiple passes over the input samples.
///
/// Upstream asm: `WelsIntra16x16Combined3Sad_ssse3` in `codec/common/x86/satd_sad.asm`.
#[inline(always)]
pub fn intra_16x16_combined3_sad(
    pred: &mut [u8; 256],
    rec: &RecCursor<'_>,
    enc: &RecCursor<'_>,
    lambda: i32,
) -> (u8, i32) {
    if let Some(token) = Avx2Token::try_fast_enabled() {
        return intra_16x16_combined3_sad_avx2(pred, rec, enc, lambda, token);
    }
    intra_16x16_combined3_sad_fallback(pred, rec, enc, lambda)
}

type EncSpan<'a> = <RecCursor<'a> as RefSamples>::Span<'a>;

/// 1D 4-point Hadamard transform scaled by 4 (`<< 2`), returning `(sum, [4*g0, 4*g1, 4*g2, 4*g3])`.
#[inline(always)]
fn hdm4_u8x4_scaled4(x: [u8; 4]) -> (i32, [i16; 4]) {
    let (x0, x1, x2, x3) = (x[0] as i32, x[1] as i32, x[2] as i32, x[3] as i32);
    let p0 = x0 + x1;
    let p1 = x2 + x3;
    let m0 = x0 - x1;
    let m1 = x2 - x3;
    let sum = p0 + p1;
    (
        sum,
        [
            (sum << 2) as i16,
            ((p0 - p1) << 2) as i16,
            ((m0 + m1) << 2) as i16,
            ((m0 - m1) << 2) as i16,
        ],
    )
}

#[inline(always)]
fn pack_i16x4(w: [i16; 4]) -> i64 {
    let bytes: [[u8; 2]; 4] = w.map(i16::to_ne_bytes);
    i64::from_ne_bytes(*bytes.as_flattened().first_chunk::<8>().expect("8 bytes"))
}

#[inline(always)]
fn splat8_i64(v: u8) -> i64 {
    i64::from_ne_bytes([v; 8])
}

#[inline(always)]
fn chroma_dc_quads_from_sums(t0: i32, t1: i32, l0: i32, l1: i32) -> ([u8; 4], [u8; 8], [u8; 8]) {
    let m0 = ((t0 + l0 + 4) >> 3) as u8;
    let m1 = ((t1 + 2) >> 2) as u8;
    let m2 = ((l1 + 2) >> 2) as u8;
    let m3 = ((t1 + l1 + 4) >> 3) as u8;
    (
        [m0, m1, m2, m3],
        [m0, m0, m0, m0, m1, m1, m1, m1],
        [m2, m2, m2, m2, m3, m3, m3, m3],
    )
}

#[target_feature(enable = "sse2")]
#[inline]
fn sum_pos_i16_8(v: __m128i) -> i32 {
    let d = _mm_madd_epi16(v, _mm_set1_epi16(1));
    let s64 = _mm_add_epi32(d, _mm_unpackhi_epi64(d, d));
    _mm_cvtsi128_si32(_mm_add_epi32(s64, _mm_srli_si128::<4>(s64)))
}

#[target_feature(enable = "sse2")]
#[inline]
fn sum_u16_8_wide(v: __m128i) -> i32 {
    let zero = _mm_setzero_si128();
    let sum32 = _mm_add_epi32(_mm_unpacklo_epi16(v, zero), _mm_unpackhi_epi16(v, zero));
    let s64 = _mm_add_epi32(sum32, _mm_unpackhi_epi64(sum32, sum32));
    _mm_cvtsi128_si32(_mm_add_epi32(s64, _mm_srli_si128::<4>(s64)))
}

#[target_feature(enable = "avx2")]
#[inline]
fn sum_pos_i16_16_avx2(v: __m256i) -> i32 {
    let d = _mm256_madd_epi16(v, _mm256_set1_epi16(1));
    let sum128 = _mm_add_epi32(_mm256_castsi256_si128(d), _mm256_extracti128_si256::<1>(d));
    let s64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
    _mm_cvtsi128_si32(_mm_add_epi32(s64, _mm_srli_si128::<4>(s64)))
}

#[target_feature(enable = "sse2")]
#[inline]
fn hdm4_sse2(
    a0: __m128i,
    a1: __m128i,
    a2: __m128i,
    a3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let p0 = _mm_add_epi16(a0, a1);
    let p1 = _mm_add_epi16(a2, a3);
    let m0 = _mm_sub_epi16(a0, a1);
    let m1 = _mm_sub_epi16(a2, a3);
    (
        _mm_add_epi16(p0, p1),
        _mm_sub_epi16(p0, p1),
        _mm_add_epi16(m0, m1),
        _mm_sub_epi16(m0, m1),
    )
}

#[target_feature(enable = "sse2")]
#[inline]
fn transpose_4x4x2_sse2(
    v0: __m128i,
    v1: __m128i,
    v2: __m128i,
    v3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let t01_lo = _mm_unpacklo_epi16(v0, v1);
    let t23_lo = _mm_unpacklo_epi16(v2, v3);
    let t01_hi = _mm_unpackhi_epi16(v0, v1);
    let t23_hi = _mm_unpackhi_epi16(v2, v3);

    let c01_b0 = _mm_unpacklo_epi32(t01_lo, t23_lo);
    let c23_b0 = _mm_unpackhi_epi32(t01_lo, t23_lo);
    let c01_b1 = _mm_unpacklo_epi32(t01_hi, t23_hi);
    let c23_b1 = _mm_unpackhi_epi32(t01_hi, t23_hi);

    (
        _mm_unpacklo_epi64(c01_b0, c01_b1),
        _mm_unpackhi_epi64(c01_b0, c01_b1),
        _mm_unpacklo_epi64(c23_b0, c23_b1),
        _mm_unpackhi_epi64(c23_b0, c23_b1),
    )
}

#[target_feature(enable = "sse2")]
#[inline]
fn top_v_subs_sse2(top8: __m128i) -> (__m128i, __m128i, __m128i, __m128i) {
    let zero = _mm_setzero_si128();
    let top_w4 = _mm_slli_epi16::<2>(_mm_unpacklo_epi8(top8, zero));
    let (c0, c1, c2, c3) = transpose_4x4x2_sse2(top_w4, zero, zero, zero);
    hdm4_sse2(c0, c1, c2, c3)
}

#[target_feature(enable = "ssse3")]
#[inline]
fn satd_strip_8x4_ssse3(
    e0: &[u8],
    e1: &[u8],
    e2: &[u8],
    e3: &[u8],
    v_subs: (__m128i, __m128i, __m128i, __m128i),
    h_vec: __m128i,
    dc_vec: __m128i,
) -> (__m128i, __m128i, __m128i) {
    let zero = _mm_setzero_si128();
    let r0 = _mm_unpacklo_epi8(ld8(e0), zero);
    let r1 = _mm_unpacklo_epi8(ld8(e1), zero);
    let r2 = _mm_unpacklo_epi8(ld8(e2), zero);
    let r3 = _mm_unpacklo_epi8(ld8(e3), zero);

    let (v0, v1, v2, v3) = hdm4_sse2(r0, r1, r2, r3);
    let (c0, c1, c2, c3) = transpose_4x4x2_sse2(v0, v1, v2, v3);
    let (f0, f1, f2, f3) = hdm4_sse2(c0, c1, c2, c3);

    let dv0 = _mm_abs_epi16(_mm_sub_epi16(f0, v_subs.0));
    let dv1 = _mm_abs_epi16(_mm_sub_epi16(f1, v_subs.1));
    let dv2 = _mm_abs_epi16(_mm_sub_epi16(f2, v_subs.2));
    let dv3 = _mm_abs_epi16(_mm_sub_epi16(f3, v_subs.3));
    let strip_v = _mm_add_epi16(_mm_add_epi16(dv0, dv1), _mm_add_epi16(dv2, dv3));

    let abs123 = _mm_add_epi16(
        _mm_abs_epi16(f1),
        _mm_add_epi16(_mm_abs_epi16(f2), _mm_abs_epi16(f3)),
    );
    let strip_h = _mm_add_epi16(_mm_abs_epi16(_mm_sub_epi16(f0, h_vec)), abs123);
    let strip_dc = _mm_add_epi16(_mm_abs_epi16(_mm_sub_epi16(f0, dc_vec)), abs123);

    (strip_v, strip_h, strip_dc)
}

#[target_feature(enable = "avx2")]
#[inline]
fn hdm4_avx2(
    a0: __m256i,
    a1: __m256i,
    a2: __m256i,
    a3: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let p0 = _mm256_add_epi16(a0, a1);
    let p1 = _mm256_add_epi16(a2, a3);
    let m0 = _mm256_sub_epi16(a0, a1);
    let m1 = _mm256_sub_epi16(a2, a3);
    (
        _mm256_add_epi16(p0, p1),
        _mm256_sub_epi16(p0, p1),
        _mm256_add_epi16(m0, m1),
        _mm256_sub_epi16(m0, m1),
    )
}

#[target_feature(enable = "avx2")]
#[inline]
fn transpose_4x4x4_avx2(
    v0: __m256i,
    v1: __m256i,
    v2: __m256i,
    v3: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let t01_lo = _mm256_unpacklo_epi16(v0, v1);
    let t23_lo = _mm256_unpacklo_epi16(v2, v3);
    let t01_hi = _mm256_unpackhi_epi16(v0, v1);
    let t23_hi = _mm256_unpackhi_epi16(v2, v3);

    let c01_b02 = _mm256_unpacklo_epi32(t01_lo, t23_lo);
    let c23_b02 = _mm256_unpackhi_epi32(t01_lo, t23_lo);
    let c01_b13 = _mm256_unpacklo_epi32(t01_hi, t23_hi);
    let c23_b13 = _mm256_unpackhi_epi32(t01_hi, t23_hi);

    (
        _mm256_unpacklo_epi64(c01_b02, c01_b13),
        _mm256_unpackhi_epi64(c01_b02, c01_b13),
        _mm256_unpacklo_epi64(c23_b02, c23_b13),
        _mm256_unpackhi_epi64(c23_b02, c23_b13),
    )
}

#[target_feature(enable = "avx2")]
#[inline]
fn top_v_subs_avx2(top16: __m128i) -> (__m256i, __m256i, __m256i, __m256i) {
    let top_w4 = _mm256_slli_epi16::<2>(_mm256_cvtepu8_epi16(top16));
    let zero = _mm256_setzero_si256();
    let (c0, c1, c2, c3) = transpose_4x4x4_avx2(top_w4, zero, zero, zero);
    hdm4_avx2(c0, c1, c2, c3)
}

#[target_feature(enable = "avx2")]
#[inline]
fn satd_strip_16x4_avx2(
    e0: __m128i,
    e1: __m128i,
    e2: __m128i,
    e3: __m128i,
    v_subs: (__m256i, __m256i, __m256i, __m256i),
    h_vec: __m256i,
    dc_vec: __m256i,
) -> (__m256i, __m256i, __m256i) {
    let r0 = _mm256_cvtepu8_epi16(e0);
    let r1 = _mm256_cvtepu8_epi16(e1);
    let r2 = _mm256_cvtepu8_epi16(e2);
    let r3 = _mm256_cvtepu8_epi16(e3);

    let (v0, v1, v2, v3) = hdm4_avx2(r0, r1, r2, r3);
    let (c0, c1, c2, c3) = transpose_4x4x4_avx2(v0, v1, v2, v3);
    let (f0, f1, f2, f3) = hdm4_avx2(c0, c1, c2, c3);

    let dv0 = _mm256_abs_epi16(_mm256_sub_epi16(f0, v_subs.0));
    let dv1 = _mm256_abs_epi16(_mm256_sub_epi16(f1, v_subs.1));
    let dv2 = _mm256_abs_epi16(_mm256_sub_epi16(f2, v_subs.2));
    let dv3 = _mm256_abs_epi16(_mm256_sub_epi16(f3, v_subs.3));
    let strip_v = _mm256_add_epi16(_mm256_add_epi16(dv0, dv1), _mm256_add_epi16(dv2, dv3));

    let abs123 = _mm256_add_epi16(
        _mm256_abs_epi16(f1),
        _mm256_add_epi16(_mm256_abs_epi16(f2), _mm256_abs_epi16(f3)),
    );
    let strip_h = _mm256_add_epi16(_mm256_abs_epi16(_mm256_sub_epi16(f0, h_vec)), abs123);
    let strip_dc = _mm256_add_epi16(_mm256_abs_epi16(_mm256_sub_epi16(f0, dc_vec)), abs123);

    (strip_v, strip_h, strip_dc)
}

#[target_feature(enable = "sse2")]
#[inline]
fn write_chroma_pred_sse2(
    dst: &mut [u8; 64],
    mode: u8,
    top: &[u8; 8],
    left: &[u8; 8],
    dc_top: &[u8; 8],
    dc_bot: &[u8; 8],
) {
    match mode {
        2 => {
            let v = _mm_set1_epi64x(i64::from_ne_bytes(*top));
            for chunk in dst.as_chunks_mut::<16>().0 {
                st16(chunk, v);
            }
        }
        1 => {
            for (step, chunk) in dst.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                let y = step * 2;
                st16(
                    chunk,
                    _mm_set_epi64x(splat8_i64(left[y + 1]), splat8_i64(left[y])),
                );
            }
        }
        _ => {
            let vt = _mm_set1_epi64x(i64::from_ne_bytes(*dc_top));
            let vb = _mm_set1_epi64x(i64::from_ne_bytes(*dc_bot));
            st16(&mut dst[..16], vt);
            st16(&mut dst[16..32], vt);
            st16(&mut dst[32..48], vb);
            st16(&mut dst[48..64], vb);
        }
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn write_chroma_pred_avx2(
    dst: &mut [u8; 64],
    mode: u8,
    top: &[u8; 8],
    left: &[u8; 8],
    dc_top: &[u8; 8],
    dc_bot: &[u8; 8],
) {
    match mode {
        2 => {
            let v = _mm256_set1_epi64x(i64::from_ne_bytes(*top));
            st32(&mut dst[..32], v);
            st32(&mut dst[32..64], v);
        }
        1 => {
            st32(
                &mut dst[..32],
                _mm256_set_epi64x(
                    splat8_i64(left[3]),
                    splat8_i64(left[2]),
                    splat8_i64(left[1]),
                    splat8_i64(left[0]),
                ),
            );
            st32(
                &mut dst[32..64],
                _mm256_set_epi64x(
                    splat8_i64(left[7]),
                    splat8_i64(left[6]),
                    splat8_i64(left[5]),
                    splat8_i64(left[4]),
                ),
            );
        }
        _ => {
            st32(
                &mut dst[..32],
                _mm256_set1_epi64x(i64::from_ne_bytes(*dc_top)),
            );
            st32(
                &mut dst[32..64],
                _mm256_set1_epi64x(i64::from_ne_bytes(*dc_bot)),
            );
        }
    }
}

simd_entry! {
    #[target_feature(enable = "ssse3")]
    fn intra_4x4_combined3_satd_ssse3(
        pred: &mut [u8; 16],
        rec: &RecCursor<'_>,
        enc: &RecCursor<'_>,
        lambda_dc: i32,
        lambda_h: i32,
        lambda_v: i32,
        _token: Ssse3Token,
    ) -> (i8, i32) {
        let top = rec.row_n::<4>(-1, 0);
        let left = read_col1::<4, _>(rec, 0, -1);
        let enc_span = enc.span::<4, 4>(0, 0);
        let (q, st) = enc_span.all_1_quad::<4>();
        let (e0, e1, e2, e3) = EncSpan::read_quad::<4>(q, st);

        let zero = _mm_setzero_si128();
        let p02 = _mm_set_epi32(0, 0, i32::from_ne_bytes(e2), i32::from_ne_bytes(e0));
        let p13 = _mm_set_epi32(0, 0, i32::from_ne_bytes(e3), i32::from_ne_bytes(e1));
        let r02 = _mm_unpacklo_epi8(p02, zero);
        let r13 = _mm_unpacklo_epi8(p13, zero);

        let s01 = _mm_add_epi16(r02, r13);
        let d01 = _mm_sub_epi16(r02, r13);
        let vlo = _mm_unpacklo_epi64(s01, d01);
        let vhi = _mm_unpackhi_epi64(s01, d01);
        let v02 = _mm_add_epi16(vlo, vhi);
        let v13 = _mm_sub_epi16(vlo, vhi);

        let t0 = _mm_unpacklo_epi16(v02, v13);
        let t1 = _mm_unpackhi_epi16(v02, v13);
        let col01 = _mm_unpacklo_epi32(t0, t1);
        let col23 = _mm_unpackhi_epi32(t0, t1);

        let cs = _mm_add_epi16(col01, col23);
        let cd = _mm_sub_epi16(col01, col23);
        let clo = _mm_unpacklo_epi64(cs, cd);
        let chi = _mm_unpackhi_epi64(cs, cd);
        let f02 = _mm_add_epi16(clo, chi);
        let f13 = _mm_sub_epi16(clo, chi);

        let (sum_t, ht) = hdm4_u8x4_scaled4(top);
        let (sum_l, hl) = hdm4_u8x4_scaled4(left);
        let dc_val = ((sum_t + sum_l + 4) >> 3) as u8;

        let v_sub_02 = _mm_setr_epi16(ht[0], 0, 0, 0, ht[1], 0, 0, 0);
        let v_sub_13 = _mm_setr_epi16(ht[2], 0, 0, 0, ht[3], 0, 0, 0);
        let h_sub_02 = _mm_setr_epi16(hl[0], hl[1], hl[2], hl[3], 0, 0, 0, 0);
        let dc_sub_02 = _mm_cvtsi32_si128((dc_val as i32) << 4);

        let abs_f13 = _mm_abs_epi16(f13);
        let vec_v = _mm_add_epi16(
            _mm_abs_epi16(_mm_sub_epi16(f02, v_sub_02)),
            _mm_abs_epi16(_mm_sub_epi16(f13, v_sub_13)),
        );
        let vec_h = _mm_add_epi16(_mm_abs_epi16(_mm_sub_epi16(f02, h_sub_02)), abs_f13);
        let vec_dc = _mm_add_epi16(_mm_abs_epi16(_mm_sub_epi16(f02, dc_sub_02)), abs_f13);

        let satd_v = sum_pos_i16_8(vec_v) >> 1;
        let satd_h = sum_pos_i16_8(vec_h) >> 1;
        let satd_dc = sum_pos_i16_8(vec_dc) >> 1;

        let cost_dc = satd_dc + lambda_dc;
        let cost_h = satd_h + lambda_h;
        let cost_v = satd_v + lambda_v;

        if cost_dc <= cost_h && cost_dc <= cost_v {
            st16(pred, _mm_set1_epi8(dc_val as i8));
            (2i8, cost_dc)
        } else if cost_h <= cost_v {
            let l_pack = _mm_cvtsi32_si128(i32::from_ne_bytes(left));
            let l_16 = _mm_unpacklo_epi8(l_pack, l_pack);
            st16(pred, _mm_unpacklo_epi16(l_16, l_16));
            (1i8, cost_h)
        } else {
            st16(pred, _mm_set1_epi32(i32::from_ne_bytes(top)));
            (0i8, cost_v)
        }
    }

    #[target_feature(enable = "avx2")]
    fn intra_16x16_combined3_satd_avx2(
        pred: &mut [u8; 256],
        rec: &RecCursor<'_>,
        enc: &RecCursor<'_>,
        lambda: i32,
        _token: Avx2Token,
    ) -> (u8, i32) {
        let top = rec.row_n::<16>(-1, 0);
        let left = read_col1::<16, _>(rec, 0, -1);
        let enc_span = enc.span::<16, 16>(0, 0);
        let (qs, st) = enc_span.all_4_quads::<16>();

        let v_vec128 = ld16(&top);
        let sad_top = _mm_sad_epu8(v_vec128, _mm_setzero_si128());
        let sum_top = _mm_cvtsi128_si32(sad_top) + _mm_extract_epi16::<4>(sad_top);

        let mut sum_left = 0i32;
        let mut h_quads = [0i64; 4];
        for s in 0..4 {
            let l4 = [
                left[s * 4],
                left[s * 4 + 1],
                left[s * 4 + 2],
                left[s * 4 + 3],
            ];
            let (sl, hl) = hdm4_u8x4_scaled4(l4);
            sum_left += sl;
            h_quads[s] = pack_i16x4(hl);
        }

        let dc_val = ((16 + sum_top + sum_left) >> 5) as u8;
        let dc_vec = _mm256_set1_epi64x(pack_i16x4([(dc_val as i16) << 4, 0, 0, 0]));
        let v_subs = top_v_subs_avx2(v_vec128);

        let mut acc_v = _mm256_setzero_si256();
        let mut acc_h = _mm256_setzero_si256();
        let mut acc_dc = _mm256_setzero_si256();

        for (s, q) in qs.into_iter().enumerate() {
            let (e0, e1, e2, e3) = EncSpan::read_quad::<16>(q, st);
            let (sv, sh, sdc) = satd_strip_16x4_avx2(
                ld16(&e0),
                ld16(&e1),
                ld16(&e2),
                ld16(&e3),
                v_subs,
                _mm256_set1_epi64x(h_quads[s]),
                dc_vec,
            );
            acc_v = _mm256_add_epi16(acc_v, sv);
            acc_h = _mm256_add_epi16(acc_h, sh);
            acc_dc = _mm256_add_epi16(acc_dc, sdc);
        }

        let satd_v = sum_pos_i16_16_avx2(acc_v) >> 1;
        let satd_h = sum_pos_i16_16_avx2(acc_h) >> 1;
        let satd_dc = sum_pos_i16_16_avx2(acc_dc) >> 1;

        let cost_v = satd_v + lambda;
        let cost_h = satd_h + lambda * 3;
        let cost_dc = satd_dc + lambda * 3;

        let (best_mode, best_cost) = if cost_dc < cost_h && cost_dc < cost_v {
            (2u8, cost_dc)
        } else if cost_h < cost_v {
            (1u8, cost_h)
        } else {
            (0u8, cost_v)
        };

        match best_mode {
            0 => {
                let v_vec256 = _mm256_broadcastsi128_si256(v_vec128);
                for chunk in pred.as_chunks_mut::<32>().0 {
                    st32(chunk, v_vec256);
                }
            }
            1 => {
                for (y, chunk) in pred.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                    st16(chunk, _mm_set1_epi8(left[y] as i8));
                }
            }
            _ => {
                let dc_vec256 = _mm256_set1_epi8(dc_val as i8);
                for chunk in pred.as_chunks_mut::<32>().0 {
                    st32(chunk, dc_vec256);
                }
            }
        }

        (best_mode, best_cost)
    }

    #[target_feature(enable = "ssse3")]
    fn intra_16x16_combined3_satd_ssse3(
        pred: &mut [u8; 256],
        rec: &RecCursor<'_>,
        enc: &RecCursor<'_>,
        lambda: i32,
        _token: Ssse3Token,
    ) -> (u8, i32) {
        let top = rec.row_n::<16>(-1, 0);
        let left = read_col1::<16, _>(rec, 0, -1);
        let enc_span = enc.span::<16, 16>(0, 0);
        let (qs, st) = enc_span.all_4_quads::<16>();

        let v_vec = ld16(&top);
        let sad_top = _mm_sad_epu8(v_vec, _mm_setzero_si128());
        let sum_top = _mm_cvtsi128_si32(sad_top) + _mm_extract_epi16::<4>(sad_top);

        let mut sum_left = 0i32;
        let mut h_quads = [0i64; 4];
        for s in 0..4 {
            let l4 = [
                left[s * 4],
                left[s * 4 + 1],
                left[s * 4 + 2],
                left[s * 4 + 3],
            ];
            let (sl, hl) = hdm4_u8x4_scaled4(l4);
            sum_left += sl;
            h_quads[s] = pack_i16x4(hl);
        }

        let dc_val = ((16 + sum_top + sum_left) >> 5) as u8;
        let dc_vec = _mm_set1_epi64x(pack_i16x4([(dc_val as i16) << 4, 0, 0, 0]));
        let v_subs_lo = top_v_subs_sse2(ld8(&top[..8]));
        let v_subs_hi = top_v_subs_sse2(ld8(&top[8..]));

        let mut acc_v = _mm_setzero_si128();
        let mut acc_h = _mm_setzero_si128();
        let mut acc_dc = _mm_setzero_si128();

        for (s, q) in qs.into_iter().enumerate() {
            let (e0, e1, e2, e3) = EncSpan::read_quad::<16>(q, st);
            let h_vec = _mm_set1_epi64x(h_quads[s]);
            let (sv0, sh0, sdc0) =
                satd_strip_8x4_ssse3(&e0[..8], &e1[..8], &e2[..8], &e3[..8], v_subs_lo, h_vec, dc_vec);
            let (sv1, sh1, sdc1) =
                satd_strip_8x4_ssse3(&e0[8..], &e1[8..], &e2[8..], &e3[8..], v_subs_hi, h_vec, dc_vec);
            acc_v = _mm_add_epi16(acc_v, _mm_add_epi16(sv0, sv1));
            acc_h = _mm_add_epi16(acc_h, _mm_add_epi16(sh0, sh1));
            acc_dc = _mm_add_epi16(acc_dc, _mm_add_epi16(sdc0, sdc1));
        }

        let satd_v = sum_u16_8_wide(acc_v) >> 1;
        let satd_h = sum_u16_8_wide(acc_h) >> 1;
        let satd_dc = sum_u16_8_wide(acc_dc) >> 1;

        let cost_v = satd_v + lambda;
        let cost_h = satd_h + lambda * 3;
        let cost_dc = satd_dc + lambda * 3;

        let (best_mode, best_cost) = if cost_dc < cost_h && cost_dc < cost_v {
            (2u8, cost_dc)
        } else if cost_h < cost_v {
            (1u8, cost_h)
        } else {
            (0u8, cost_v)
        };

        match best_mode {
            0 => {
                for chunk in pred.as_chunks_mut::<16>().0 {
                    st16(chunk, v_vec);
                }
            }
            1 => {
                for (y, chunk) in pred.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                    st16(chunk, _mm_set1_epi8(left[y] as i8));
                }
            }
            _ => {
                let dc_vec_b = _mm_set1_epi8(dc_val as i8);
                for chunk in pred.as_chunks_mut::<16>().0 {
                    st16(chunk, dc_vec_b);
                }
            }
        }

        (best_mode, best_cost)
    }

    #[target_feature(enable = "avx2")]
    fn intra_chroma_8x8_combined3_sad_avx2(
        pred_cb_cr: &mut [u8; 128],
        rec_cb: &RecCursor<'_>,
        enc_cb: &RecCursor<'_>,
        rec_cr: &RecCursor<'_>,
        enc_cr: &RecCursor<'_>,
        lambda: i32,
        _token: Avx2Token,
    ) -> (u8, i32) {
        let top_cb = rec_cb.row_n::<8>(-1, 0);
        let top_cr = rec_cr.row_n::<8>(-1, 0);
        let left_cb = read_col1::<8, _>(rec_cb, 0, -1);
        let left_cr = read_col1::<8, _>(rec_cr, 0, -1);
        let enc_cb_span = enc_cb.span::<8, 8>(0, 0);
        let enc_cr_span = enc_cr.span::<8, 8>(0, 0);
        let ([qb0, qb1], stb) = enc_cb_span.all_2_quads::<8>();
        let ([qr0, qr1], str_) = enc_cr_span.all_2_quads::<8>();

        let v_vec128 = _mm_set_epi64x(i64::from_ne_bytes(top_cr), i64::from_ne_bytes(top_cb));
        let l_vec128 = _mm_set_epi64x(i64::from_ne_bytes(left_cr), i64::from_ne_bytes(left_cb));
        let edges256 = _mm256_set_m128i(l_vec128, v_vec128);
        let sums4 = _mm256_madd_epi16(
            _mm256_maddubs_epi16(edges256, _mm256_set1_epi8(1)),
            _mm256_set1_epi16(1),
        );
        let sums_t = _mm256_castsi256_si128(sums4);
        let sums_l = _mm256_extracti128_si256::<1>(sums4);
        let (_, dc_top_cb, dc_bot_cb) = chroma_dc_quads_from_sums(
            _mm_cvtsi128_si32(sums_t),
            _mm_cvtsi128_si32(_mm_srli_si128::<4>(sums_t)),
            _mm_cvtsi128_si32(sums_l),
            _mm_cvtsi128_si32(_mm_srli_si128::<4>(sums_l)),
        );
        let (_, dc_top_cr, dc_bot_cr) = chroma_dc_quads_from_sums(
            _mm_cvtsi128_si32(_mm_srli_si128::<8>(sums_t)),
            _mm_cvtsi128_si32(_mm_srli_si128::<12>(sums_t)),
            _mm_cvtsi128_si32(_mm_srli_si128::<8>(sums_l)),
            _mm_cvtsi128_si32(_mm_srli_si128::<12>(sums_l)),
        );

        let v_vec256 = _mm256_broadcastsi128_si256(v_vec128);
        let dc_top256 = _mm256_broadcastsi128_si256(_mm_set_epi64x(
            i64::from_ne_bytes(dc_top_cr),
            i64::from_ne_bytes(dc_top_cb),
        ));
        let dc_bot256 = _mm256_broadcastsi128_si256(_mm_set_epi64x(
            i64::from_ne_bytes(dc_bot_cr),
            i64::from_ne_bytes(dc_bot_cb),
        ));

        let (cb0, cb1, cb2, cb3) = EncSpan::read_quad::<8>(qb0, stb);
        let (cr0, cr1, cr2, cr3) = EncSpan::read_quad::<8>(qr0, str_);
        let enc01 = _mm256_set_epi64x(
            i64::from_ne_bytes(cr1),
            i64::from_ne_bytes(cb1),
            i64::from_ne_bytes(cr0),
            i64::from_ne_bytes(cb0),
        );
        let enc23 = _mm256_set_epi64x(
            i64::from_ne_bytes(cr3),
            i64::from_ne_bytes(cb3),
            i64::from_ne_bytes(cr2),
            i64::from_ne_bytes(cb2),
        );
        let h01 = _mm256_set_epi64x(
            splat8_i64(left_cr[1]),
            splat8_i64(left_cb[1]),
            splat8_i64(left_cr[0]),
            splat8_i64(left_cb[0]),
        );
        let h23 = _mm256_set_epi64x(
            splat8_i64(left_cr[3]),
            splat8_i64(left_cb[3]),
            splat8_i64(left_cr[2]),
            splat8_i64(left_cb[2]),
        );

        let mut acc_v = _mm256_add_epi64(
            _mm256_sad_epu8(enc01, v_vec256),
            _mm256_sad_epu8(enc23, v_vec256),
        );
        let mut acc_h = _mm256_add_epi64(
            _mm256_sad_epu8(enc01, h01),
            _mm256_sad_epu8(enc23, h23),
        );
        let mut acc_dc = _mm256_add_epi64(
            _mm256_sad_epu8(enc01, dc_top256),
            _mm256_sad_epu8(enc23, dc_top256),
        );

        let (cb4, cb5, cb6, cb7) = EncSpan::read_quad::<8>(qb1, stb);
        let (cr4, cr5, cr6, cr7) = EncSpan::read_quad::<8>(qr1, str_);
        let enc45 = _mm256_set_epi64x(
            i64::from_ne_bytes(cr5),
            i64::from_ne_bytes(cb5),
            i64::from_ne_bytes(cr4),
            i64::from_ne_bytes(cb4),
        );
        let enc67 = _mm256_set_epi64x(
            i64::from_ne_bytes(cr7),
            i64::from_ne_bytes(cb7),
            i64::from_ne_bytes(cr6),
            i64::from_ne_bytes(cb6),
        );
        let h45 = _mm256_set_epi64x(
            splat8_i64(left_cr[5]),
            splat8_i64(left_cb[5]),
            splat8_i64(left_cr[4]),
            splat8_i64(left_cb[4]),
        );
        let h67 = _mm256_set_epi64x(
            splat8_i64(left_cr[7]),
            splat8_i64(left_cb[7]),
            splat8_i64(left_cr[6]),
            splat8_i64(left_cb[6]),
        );

        acc_v = _mm256_add_epi64(
            acc_v,
            _mm256_add_epi64(
                _mm256_sad_epu8(enc45, v_vec256),
                _mm256_sad_epu8(enc67, v_vec256),
            ),
        );
        acc_h = _mm256_add_epi64(
            acc_h,
            _mm256_add_epi64(
                _mm256_sad_epu8(enc45, h45),
                _mm256_sad_epu8(enc67, h67),
            ),
        );
        acc_dc = _mm256_add_epi64(
            acc_dc,
            _mm256_add_epi64(
                _mm256_sad_epu8(enc45, dc_bot256),
                _mm256_sad_epu8(enc67, dc_bot256),
            ),
        );

        let sad_v = reduce256(acc_v);
        let sad_h = reduce256(acc_h);
        let sad_dc = reduce256(acc_dc);

        let cost_dc = sad_dc + lambda;
        let cost_h = sad_h + lambda * 3;
        let cost_v = sad_v + lambda * 3;

        let (best_mode, best_cost) = if cost_dc < cost_h && cost_dc < cost_v {
            (0u8, cost_dc)
        } else if cost_h < cost_v {
            (1u8, cost_h)
        } else {
            (2u8, cost_v)
        };

        let (cb_dst, cr_dst) = pred_cb_cr.split_at_mut(64);
        write_chroma_pred_avx2(
            cb_dst.first_chunk_mut::<64>().expect("cb 64"),
            best_mode,
            &top_cb,
            &left_cb,
            &dc_top_cb,
            &dc_bot_cb,
        );
        write_chroma_pred_avx2(
            cr_dst.first_chunk_mut::<64>().expect("cr 64"),
            best_mode,
            &top_cr,
            &left_cr,
            &dc_top_cr,
            &dc_bot_cr,
        );

        (best_mode, best_cost)
    }

    #[target_feature(enable = "sse2")]
    fn intra_chroma_8x8_combined3_sad_sse2(
        pred_cb_cr: &mut [u8; 128],
        rec_cb: &RecCursor<'_>,
        enc_cb: &RecCursor<'_>,
        rec_cr: &RecCursor<'_>,
        enc_cr: &RecCursor<'_>,
        lambda: i32,
        _token: Sse2Token,
    ) -> (u8, i32) {
        let top_cb = rec_cb.row_n::<8>(-1, 0);
        let top_cr = rec_cr.row_n::<8>(-1, 0);
        let left_cb = read_col1::<8, _>(rec_cb, 0, -1);
        let left_cr = read_col1::<8, _>(rec_cr, 0, -1);
        let enc_cb_span = enc_cb.span::<8, 8>(0, 0);
        let enc_cr_span = enc_cr.span::<8, 8>(0, 0);
        let ([qb0, qb1], stb) = enc_cb_span.all_2_quads::<8>();
        let ([qr0, qr1], str_) = enc_cr_span.all_2_quads::<8>();

        let (t0_cb, _) = hdm4_u8x4_scaled4([top_cb[0], top_cb[1], top_cb[2], top_cb[3]]);
        let (t1_cb, _) = hdm4_u8x4_scaled4([top_cb[4], top_cb[5], top_cb[6], top_cb[7]]);
        let (l0_cb, _) = hdm4_u8x4_scaled4([left_cb[0], left_cb[1], left_cb[2], left_cb[3]]);
        let (l1_cb, _) = hdm4_u8x4_scaled4([left_cb[4], left_cb[5], left_cb[6], left_cb[7]]);
        let (_, dc_top_cb, dc_bot_cb) = chroma_dc_quads_from_sums(t0_cb, t1_cb, l0_cb, l1_cb);

        let (t0_cr, _) = hdm4_u8x4_scaled4([top_cr[0], top_cr[1], top_cr[2], top_cr[3]]);
        let (t1_cr, _) = hdm4_u8x4_scaled4([top_cr[4], top_cr[5], top_cr[6], top_cr[7]]);
        let (l0_cr, _) = hdm4_u8x4_scaled4([left_cr[0], left_cr[1], left_cr[2], left_cr[3]]);
        let (l1_cr, _) = hdm4_u8x4_scaled4([left_cr[4], left_cr[5], left_cr[6], left_cr[7]]);
        let (_, dc_top_cr, dc_bot_cr) = chroma_dc_quads_from_sums(t0_cr, t1_cr, l0_cr, l1_cr);

        let v_vec = _mm_set_epi64x(i64::from_ne_bytes(top_cr), i64::from_ne_bytes(top_cb));
        let dc_top_vec = _mm_set_epi64x(
            i64::from_ne_bytes(dc_top_cr),
            i64::from_ne_bytes(dc_top_cb),
        );
        let dc_bot_vec = _mm_set_epi64x(
            i64::from_ne_bytes(dc_bot_cr),
            i64::from_ne_bytes(dc_bot_cb),
        );

        let mut acc_v = _mm_setzero_si128();
        let mut acc_h = _mm_setzero_si128();
        let mut acc_dc = _mm_setzero_si128();

        let (cb0, cb1, cb2, cb3) = EncSpan::read_quad::<8>(qb0, stb);
        let (cr0, cr1, cr2, cr3) = EncSpan::read_quad::<8>(qr0, str_);
        for (y, (cb_r, cr_r)) in [(cb0, cr0), (cb1, cr1), (cb2, cr2), (cb3, cr3)]
            .into_iter()
            .enumerate()
        {
            let enc_vec = _mm_set_epi64x(i64::from_ne_bytes(cr_r), i64::from_ne_bytes(cb_r));
            let h_vec = _mm_set_epi64x(splat8_i64(left_cr[y]), splat8_i64(left_cb[y]));
            acc_v = _mm_add_epi64(acc_v, _mm_sad_epu8(enc_vec, v_vec));
            acc_h = _mm_add_epi64(acc_h, _mm_sad_epu8(enc_vec, h_vec));
            acc_dc = _mm_add_epi64(acc_dc, _mm_sad_epu8(enc_vec, dc_top_vec));
        }

        let (cb4, cb5, cb6, cb7) = EncSpan::read_quad::<8>(qb1, stb);
        let (cr4, cr5, cr6, cr7) = EncSpan::read_quad::<8>(qr1, str_);
        for (k, (cb_r, cr_r)) in [(cb4, cr4), (cb5, cr5), (cb6, cr6), (cb7, cr7)]
            .into_iter()
            .enumerate()
        {
            let y = k + 4;
            let enc_vec = _mm_set_epi64x(i64::from_ne_bytes(cr_r), i64::from_ne_bytes(cb_r));
            let h_vec = _mm_set_epi64x(splat8_i64(left_cr[y]), splat8_i64(left_cb[y]));
            acc_v = _mm_add_epi64(acc_v, _mm_sad_epu8(enc_vec, v_vec));
            acc_h = _mm_add_epi64(acc_h, _mm_sad_epu8(enc_vec, h_vec));
            acc_dc = _mm_add_epi64(acc_dc, _mm_sad_epu8(enc_vec, dc_bot_vec));
        }

        let sad_v = _mm_cvtsi128_si32(_mm_add_epi32(acc_v, _mm_srli_si128::<8>(acc_v)));
        let sad_h = _mm_cvtsi128_si32(_mm_add_epi32(acc_h, _mm_srli_si128::<8>(acc_h)));
        let sad_dc = _mm_cvtsi128_si32(_mm_add_epi32(acc_dc, _mm_srli_si128::<8>(acc_dc)));

        let cost_dc = sad_dc + lambda;
        let cost_h = sad_h + lambda * 3;
        let cost_v = sad_v + lambda * 3;

        let (best_mode, best_cost) = if cost_dc < cost_h && cost_dc < cost_v {
            (0u8, cost_dc)
        } else if cost_h < cost_v {
            (1u8, cost_h)
        } else {
            (2u8, cost_v)
        };

        let (cb_dst, cr_dst) = pred_cb_cr.split_at_mut(64);
        write_chroma_pred_sse2(
            cb_dst.first_chunk_mut::<64>().expect("cb 64"),
            best_mode,
            &top_cb,
            &left_cb,
            &dc_top_cb,
            &dc_bot_cb,
        );
        write_chroma_pred_sse2(
            cr_dst.first_chunk_mut::<64>().expect("cr 64"),
            best_mode,
            &top_cr,
            &left_cr,
            &dc_top_cr,
            &dc_bot_cr,
        );

        (best_mode, best_cost)
    }

    #[target_feature(enable = "avx2")]
    fn intra_chroma_8x8_combined3_satd_avx2(
        pred_cb_cr: &mut [u8; 128],
        rec_cb: &RecCursor<'_>,
        enc_cb: &RecCursor<'_>,
        rec_cr: &RecCursor<'_>,
        enc_cr: &RecCursor<'_>,
        lambda: i32,
        _token: Avx2Token,
    ) -> (u8, i32) {
        let top_cb = rec_cb.row_n::<8>(-1, 0);
        let top_cr = rec_cr.row_n::<8>(-1, 0);
        let left_cb = read_col1::<8, _>(rec_cb, 0, -1);
        let left_cr = read_col1::<8, _>(rec_cr, 0, -1);
        let enc_cb_span = enc_cb.span::<8, 8>(0, 0);
        let enc_cr_span = enc_cr.span::<8, 8>(0, 0);
        let ([qb0, qb1], stb) = enc_cb_span.all_2_quads::<8>();
        let ([qr0, qr1], str_) = enc_cr_span.all_2_quads::<8>();

        let (t0_cb, _) = hdm4_u8x4_scaled4([top_cb[0], top_cb[1], top_cb[2], top_cb[3]]);
        let (t1_cb, _) = hdm4_u8x4_scaled4([top_cb[4], top_cb[5], top_cb[6], top_cb[7]]);
        let (l0_cb, hl0_cb) = hdm4_u8x4_scaled4([left_cb[0], left_cb[1], left_cb[2], left_cb[3]]);
        let (l1_cb, hl1_cb) = hdm4_u8x4_scaled4([left_cb[4], left_cb[5], left_cb[6], left_cb[7]]);
        let (dc_cb, dc_top_cb, dc_bot_cb) = chroma_dc_quads_from_sums(t0_cb, t1_cb, l0_cb, l1_cb);

        let (t0_cr, _) = hdm4_u8x4_scaled4([top_cr[0], top_cr[1], top_cr[2], top_cr[3]]);
        let (t1_cr, _) = hdm4_u8x4_scaled4([top_cr[4], top_cr[5], top_cr[6], top_cr[7]]);
        let (l0_cr, hl0_cr) = hdm4_u8x4_scaled4([left_cr[0], left_cr[1], left_cr[2], left_cr[3]]);
        let (l1_cr, hl1_cr) = hdm4_u8x4_scaled4([left_cr[4], left_cr[5], left_cr[6], left_cr[7]]);
        let (dc_cr, dc_top_cr, dc_bot_cr) = chroma_dc_quads_from_sums(t0_cr, t1_cr, l0_cr, l1_cr);

        let top_cb_cr = _mm_set_epi64x(i64::from_ne_bytes(top_cr), i64::from_ne_bytes(top_cb));
        let v_subs = top_v_subs_avx2(top_cb_cr);

        let h0_cb_i64 = pack_i16x4(hl0_cb);
        let h0_cr_i64 = pack_i16x4(hl0_cr);
        let h_top = _mm256_set_epi64x(h0_cr_i64, h0_cr_i64, h0_cb_i64, h0_cb_i64);
        let dc_top = _mm256_set_epi64x(
            pack_i16x4([(dc_cr[1] as i16) << 4, 0, 0, 0]),
            pack_i16x4([(dc_cr[0] as i16) << 4, 0, 0, 0]),
            pack_i16x4([(dc_cb[1] as i16) << 4, 0, 0, 0]),
            pack_i16x4([(dc_cb[0] as i16) << 4, 0, 0, 0]),
        );

        let (cb0, cb1, cb2, cb3) = EncSpan::read_quad::<8>(qb0, stb);
        let (cr0, cr1, cr2, cr3) = EncSpan::read_quad::<8>(qr0, str_);
        let (acc_v0, acc_h0, acc_dc0) = satd_strip_16x4_avx2(
            _mm_set_epi64x(i64::from_ne_bytes(cr0), i64::from_ne_bytes(cb0)),
            _mm_set_epi64x(i64::from_ne_bytes(cr1), i64::from_ne_bytes(cb1)),
            _mm_set_epi64x(i64::from_ne_bytes(cr2), i64::from_ne_bytes(cb2)),
            _mm_set_epi64x(i64::from_ne_bytes(cr3), i64::from_ne_bytes(cb3)),
            v_subs,
            h_top,
            dc_top,
        );

        let h1_cb_i64 = pack_i16x4(hl1_cb);
        let h1_cr_i64 = pack_i16x4(hl1_cr);
        let h_bot = _mm256_set_epi64x(h1_cr_i64, h1_cr_i64, h1_cb_i64, h1_cb_i64);
        let dc_bot = _mm256_set_epi64x(
            pack_i16x4([(dc_cr[3] as i16) << 4, 0, 0, 0]),
            pack_i16x4([(dc_cr[2] as i16) << 4, 0, 0, 0]),
            pack_i16x4([(dc_cb[3] as i16) << 4, 0, 0, 0]),
            pack_i16x4([(dc_cb[2] as i16) << 4, 0, 0, 0]),
        );

        let (cb4, cb5, cb6, cb7) = EncSpan::read_quad::<8>(qb1, stb);
        let (cr4, cr5, cr6, cr7) = EncSpan::read_quad::<8>(qr1, str_);
        let (acc_v1, acc_h1, acc_dc1) = satd_strip_16x4_avx2(
            _mm_set_epi64x(i64::from_ne_bytes(cr4), i64::from_ne_bytes(cb4)),
            _mm_set_epi64x(i64::from_ne_bytes(cr5), i64::from_ne_bytes(cb5)),
            _mm_set_epi64x(i64::from_ne_bytes(cr6), i64::from_ne_bytes(cb6)),
            _mm_set_epi64x(i64::from_ne_bytes(cr7), i64::from_ne_bytes(cb7)),
            v_subs,
            h_bot,
            dc_bot,
        );

        let satd_v = sum_pos_i16_16_avx2(_mm256_add_epi16(acc_v0, acc_v1)) >> 1;
        let satd_h = sum_pos_i16_16_avx2(_mm256_add_epi16(acc_h0, acc_h1)) >> 1;
        let satd_dc = sum_pos_i16_16_avx2(_mm256_add_epi16(acc_dc0, acc_dc1)) >> 1;

        let cost_dc = satd_dc + lambda;
        let cost_h = satd_h + lambda * 3;
        let cost_v = satd_v + lambda * 3;

        let (best_mode, best_cost) = if cost_dc < cost_h && cost_dc < cost_v {
            (0u8, cost_dc)
        } else if cost_h < cost_v {
            (1u8, cost_h)
        } else {
            (2u8, cost_v)
        };

        let (cb_dst, cr_dst) = pred_cb_cr.split_at_mut(64);
        write_chroma_pred_avx2(
            cb_dst.first_chunk_mut::<64>().expect("cb 64"),
            best_mode,
            &top_cb,
            &left_cb,
            &dc_top_cb,
            &dc_bot_cb,
        );
        write_chroma_pred_avx2(
            cr_dst.first_chunk_mut::<64>().expect("cr 64"),
            best_mode,
            &top_cr,
            &left_cr,
            &dc_top_cr,
            &dc_bot_cr,
        );

        (best_mode, best_cost)
    }

    #[target_feature(enable = "ssse3")]
    fn intra_chroma_8x8_combined3_satd_ssse3(
        pred_cb_cr: &mut [u8; 128],
        rec_cb: &RecCursor<'_>,
        enc_cb: &RecCursor<'_>,
        rec_cr: &RecCursor<'_>,
        enc_cr: &RecCursor<'_>,
        lambda: i32,
        _token: Ssse3Token,
    ) -> (u8, i32) {
        let top_cb = rec_cb.row_n::<8>(-1, 0);
        let top_cr = rec_cr.row_n::<8>(-1, 0);
        let left_cb = read_col1::<8, _>(rec_cb, 0, -1);
        let left_cr = read_col1::<8, _>(rec_cr, 0, -1);
        let enc_cb_span = enc_cb.span::<8, 8>(0, 0);
        let enc_cr_span = enc_cr.span::<8, 8>(0, 0);
        let ([qb0, qb1], stb) = enc_cb_span.all_2_quads::<8>();
        let ([qr0, qr1], str_) = enc_cr_span.all_2_quads::<8>();

        let (t0_cb, _) = hdm4_u8x4_scaled4([top_cb[0], top_cb[1], top_cb[2], top_cb[3]]);
        let (t1_cb, _) = hdm4_u8x4_scaled4([top_cb[4], top_cb[5], top_cb[6], top_cb[7]]);
        let (l0_cb, hl0_cb) = hdm4_u8x4_scaled4([left_cb[0], left_cb[1], left_cb[2], left_cb[3]]);
        let (l1_cb, hl1_cb) = hdm4_u8x4_scaled4([left_cb[4], left_cb[5], left_cb[6], left_cb[7]]);
        let (dc_cb, dc_top_cb, dc_bot_cb) = chroma_dc_quads_from_sums(t0_cb, t1_cb, l0_cb, l1_cb);

        let (t0_cr, _) = hdm4_u8x4_scaled4([top_cr[0], top_cr[1], top_cr[2], top_cr[3]]);
        let (t1_cr, _) = hdm4_u8x4_scaled4([top_cr[4], top_cr[5], top_cr[6], top_cr[7]]);
        let (l0_cr, hl0_cr) = hdm4_u8x4_scaled4([left_cr[0], left_cr[1], left_cr[2], left_cr[3]]);
        let (l1_cr, hl1_cr) = hdm4_u8x4_scaled4([left_cr[4], left_cr[5], left_cr[6], left_cr[7]]);
        let (dc_cr, dc_top_cr, dc_bot_cr) = chroma_dc_quads_from_sums(t0_cr, t1_cr, l0_cr, l1_cr);

        let v_subs_cb = top_v_subs_sse2(ld8(&top_cb));
        let v_subs_cr = top_v_subs_sse2(ld8(&top_cr));

        let (cb0, cb1, cb2, cb3) = EncSpan::read_quad::<8>(qb0, stb);
        let (cb4, cb5, cb6, cb7) = EncSpan::read_quad::<8>(qb1, stb);
        let (sv_b0, sh_b0, sdc_b0) = satd_strip_8x4_ssse3(
            &cb0,
            &cb1,
            &cb2,
            &cb3,
            v_subs_cb,
            _mm_set1_epi64x(pack_i16x4(hl0_cb)),
            _mm_set_epi64x(
                pack_i16x4([(dc_cb[1] as i16) << 4, 0, 0, 0]),
                pack_i16x4([(dc_cb[0] as i16) << 4, 0, 0, 0]),
            ),
        );
        let (sv_b1, sh_b1, sdc_b1) = satd_strip_8x4_ssse3(
            &cb4,
            &cb5,
            &cb6,
            &cb7,
            v_subs_cb,
            _mm_set1_epi64x(pack_i16x4(hl1_cb)),
            _mm_set_epi64x(
                pack_i16x4([(dc_cb[3] as i16) << 4, 0, 0, 0]),
                pack_i16x4([(dc_cb[2] as i16) << 4, 0, 0, 0]),
            ),
        );

        let (cr0, cr1, cr2, cr3) = EncSpan::read_quad::<8>(qr0, str_);
        let (cr4, cr5, cr6, cr7) = EncSpan::read_quad::<8>(qr1, str_);
        let (sv_r0, sh_r0, sdc_r0) = satd_strip_8x4_ssse3(
            &cr0,
            &cr1,
            &cr2,
            &cr3,
            v_subs_cr,
            _mm_set1_epi64x(pack_i16x4(hl0_cr)),
            _mm_set_epi64x(
                pack_i16x4([(dc_cr[1] as i16) << 4, 0, 0, 0]),
                pack_i16x4([(dc_cr[0] as i16) << 4, 0, 0, 0]),
            ),
        );
        let (sv_r1, sh_r1, sdc_r1) = satd_strip_8x4_ssse3(
            &cr4,
            &cr5,
            &cr6,
            &cr7,
            v_subs_cr,
            _mm_set1_epi64x(pack_i16x4(hl1_cr)),
            _mm_set_epi64x(
                pack_i16x4([(dc_cr[3] as i16) << 4, 0, 0, 0]),
                pack_i16x4([(dc_cr[2] as i16) << 4, 0, 0, 0]),
            ),
        );

        let acc_v = _mm_add_epi16(_mm_add_epi16(sv_b0, sv_b1), _mm_add_epi16(sv_r0, sv_r1));
        let acc_h = _mm_add_epi16(_mm_add_epi16(sh_b0, sh_b1), _mm_add_epi16(sh_r0, sh_r1));
        let acc_dc = _mm_add_epi16(
            _mm_add_epi16(sdc_b0, sdc_b1),
            _mm_add_epi16(sdc_r0, sdc_r1),
        );

        let satd_v = sum_pos_i16_8(acc_v) >> 1;
        let satd_h = sum_pos_i16_8(acc_h) >> 1;
        let satd_dc = sum_pos_i16_8(acc_dc) >> 1;

        let cost_dc = satd_dc + lambda;
        let cost_h = satd_h + lambda * 3;
        let cost_v = satd_v + lambda * 3;

        let (best_mode, best_cost) = if cost_dc < cost_h && cost_dc < cost_v {
            (0u8, cost_dc)
        } else if cost_h < cost_v {
            (1u8, cost_h)
        } else {
            (2u8, cost_v)
        };

        let (cb_dst, cr_dst) = pred_cb_cr.split_at_mut(64);
        write_chroma_pred_sse2(
            cb_dst.first_chunk_mut::<64>().expect("cb 64"),
            best_mode,
            &top_cb,
            &left_cb,
            &dc_top_cb,
            &dc_bot_cb,
        );
        write_chroma_pred_sse2(
            cr_dst.first_chunk_mut::<64>().expect("cr 64"),
            best_mode,
            &top_cr,
            &left_cr,
            &dc_top_cr,
            &dc_bot_cr,
        );

        (best_mode, best_cost)
    }
}

#[cold]
#[inline(never)]
fn intra_16x16_combined3_satd_fallback(
    pred: &mut [u8; 256],
    rec: &RecCursor<'_>,
    enc: &RecCursor<'_>,
    lambda: i32,
) -> (u8, i32) {
    if let Some(token) = Avx2Token::try_new_enabled() {
        return intra_16x16_combined3_satd_avx2(pred, rec, enc, lambda, token);
    }
    intra_16x16_combined3_satd_ssse3(pred, rec, enc, lambda, ssse3_token())
}

#[cold]
#[inline(never)]
fn intra_chroma_8x8_combined3_sad_fallback(
    pred_cb_cr: &mut [u8; 128],
    rec_cb: &RecCursor<'_>,
    enc_cb: &RecCursor<'_>,
    rec_cr: &RecCursor<'_>,
    enc_cr: &RecCursor<'_>,
    lambda: i32,
) -> (u8, i32) {
    if let Some(token) = Avx2Token::try_new_enabled() {
        return intra_chroma_8x8_combined3_sad_avx2(
            pred_cb_cr, rec_cb, enc_cb, rec_cr, enc_cr, lambda, token,
        );
    }
    intra_chroma_8x8_combined3_sad_sse2(
        pred_cb_cr,
        rec_cb,
        enc_cb,
        rec_cr,
        enc_cr,
        lambda,
        sse2_token(),
    )
}

#[cold]
#[inline(never)]
fn intra_chroma_8x8_combined3_satd_fallback(
    pred_cb_cr: &mut [u8; 128],
    rec_cb: &RecCursor<'_>,
    enc_cb: &RecCursor<'_>,
    rec_cr: &RecCursor<'_>,
    enc_cr: &RecCursor<'_>,
    lambda: i32,
) -> (u8, i32) {
    if let Some(token) = Avx2Token::try_new_enabled() {
        return intra_chroma_8x8_combined3_satd_avx2(
            pred_cb_cr, rec_cb, enc_cb, rec_cr, enc_cr, lambda, token,
        );
    }
    intra_chroma_8x8_combined3_satd_ssse3(
        pred_cb_cr,
        rec_cb,
        enc_cb,
        rec_cr,
        enc_cr,
        lambda,
        ssse3_token(),
    )
}

/// Combined 3-mode (DC, Horizontal, Vertical) 4x4 Intra Prediction and SATD evaluation
/// using SSSE3 (`_mm_abs_epi16`).
///
/// Replaces upstream's `WelsSampleSatdThree4x4_sse2` (`codec/common/x86/satd_sad.asm`).
#[inline(always)]
pub fn intra_4x4_combined3_satd(
    pred: &mut [u8; 16],
    rec: &RecCursor<'_>,
    enc: &RecCursor<'_>,
    lambda_dc: i32,
    lambda_h: i32,
    lambda_v: i32,
) -> (i8, i32) {
    intra_4x4_combined3_satd_ssse3(pred, rec, enc, lambda_dc, lambda_h, lambda_v, ssse3_token())
}

/// Combined 3-mode (Vertical, Horizontal, DC) 16x16 Intra Prediction and SATD evaluation.
///
/// Upstream asm: `WelsIntra16x16Combined3Satd_sse41` in `codec/common/x86/satd_sad.asm`.
#[inline(always)]
pub fn intra_16x16_combined3_satd(
    pred: &mut [u8; 256],
    rec: &RecCursor<'_>,
    enc: &RecCursor<'_>,
    lambda: i32,
) -> (u8, i32) {
    if let Some(token) = Avx2Token::try_fast_enabled() {
        return intra_16x16_combined3_satd_avx2(pred, rec, enc, lambda, token);
    }
    intra_16x16_combined3_satd_fallback(pred, rec, enc, lambda)
}

/// Combined 3-mode (Vertical, Horizontal, DC) 8x8 Chroma Cb+Cr Intra Prediction and SAD evaluation.
#[inline(always)]
pub fn intra_chroma_8x8_combined3_sad(
    pred_cb_cr: &mut [u8; 128],
    rec_cb: &RecCursor<'_>,
    enc_cb: &RecCursor<'_>,
    rec_cr: &RecCursor<'_>,
    enc_cr: &RecCursor<'_>,
    lambda: i32,
) -> (u8, i32) {
    if let Some(token) = Avx2Token::try_fast_enabled() {
        return intra_chroma_8x8_combined3_sad_avx2(
            pred_cb_cr, rec_cb, enc_cb, rec_cr, enc_cr, lambda, token,
        );
    }
    intra_chroma_8x8_combined3_sad_fallback(pred_cb_cr, rec_cb, enc_cb, rec_cr, enc_cr, lambda)
}

/// Combined 3-mode (Vertical, Horizontal, DC) 8x8 Chroma Cb+Cr Intra Prediction and SATD evaluation.
///
/// Upstream asm: `WelsIntraChroma8x8Combined3Satd_sse41` in `codec/common/x86/satd_sad.asm`.
#[inline(always)]
pub fn intra_chroma_8x8_combined3_satd(
    pred_cb_cr: &mut [u8; 128],
    rec_cb: &RecCursor<'_>,
    enc_cb: &RecCursor<'_>,
    rec_cr: &RecCursor<'_>,
    enc_cr: &RecCursor<'_>,
    lambda: i32,
) -> (u8, i32) {
    if let Some(token) = Avx2Token::try_fast_enabled() {
        return intra_chroma_8x8_combined3_satd_avx2(
            pred_cb_cr, rec_cb, enc_cb, rec_cr, enc_cr, lambda, token,
        );
    }
    intra_chroma_8x8_combined3_satd_fallback(pred_cb_cr, rec_cb, enc_cb, rec_cr, enc_cr, lambda)
}

// ============================================================================
// Unit Tests (Parity against scalar implementations)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::get_intra_predictor::*;
    use crate::encoder::rec_view::shared_plane_for_test;
    use crate::safe::plane::PaddedPlane;
    fn reference_combined3_sad(
        pred: &mut [u8; 256],
        rec: &RecCursor<'_>,
        enc: &RecCursor<'_>,
        lambda: i32,
    ) -> (u8, i32) {
        let mut pred_v = [0u8; 256];
        let mut pred_h = [0u8; 256];
        let mut pred_dc = [0u8; 256];
        WelsI16x16LumaPredV_c(&mut pred_v, rec);
        WelsI16x16LumaPredH_c(&mut pred_h, rec);
        WelsI16x16LumaPredDc_c(&mut pred_dc, rec);

        let sad_v = crate::common::sad_common::sample_sad::<16, 16, _>(
            &RecCursor::over_owned(&mut pred_v, 0, 16),
            enc,
        );
        let sad_h = crate::common::sad_common::sample_sad::<16, 16, _>(
            &RecCursor::over_owned(&mut pred_h, 0, 16),
            enc,
        );
        let sad_dc = crate::common::sad_common::sample_sad::<16, 16, _>(
            &RecCursor::over_owned(&mut pred_dc, 0, 16),
            enc,
        );

        let cost_v = sad_v + lambda * 1;
        let cost_h = sad_h + lambda * 3;
        let cost_dc = sad_dc + lambda * 3;

        let (best_mode, best_cost) = if cost_dc < cost_h && cost_dc < cost_v {
            (2u8, cost_dc)
        } else if cost_h < cost_v {
            (1u8, cost_h)
        } else {
            (0u8, cost_v)
        };

        match best_mode {
            0 => *pred = pred_v,
            1 => *pred = pred_h,
            _ => *pred = pred_dc,
        }

        (best_mode, best_cost)
    }

    #[test]
    fn test_intra_16x16_combined3_sad_parity() {
        for seed in [13u8, 42, 107, 233] {
            let mut p_rec = test_plane(32, 32, 16, 64);
            let mut p_enc = test_plane(32, 32, 16, 64);
            // Perturb top-left sample with test seed.
            p_enc.set(0, 0, seed);

            let v_rec = shared_plane_for_test(&mut p_rec);
            let v_enc = shared_plane_for_test(&mut p_enc);
            let rec = v_rec.cursor(0, 0);
            let enc = v_enc.cursor(0, 0);

            for lambda in [0, 5, 10, 50, 100] {
                let mut pred_simd = [0u8; 256];
                let mut pred_scalar = [0u8; 256];

                let (mode_simd, cost_simd) =
                    intra_16x16_combined3_sad(&mut pred_simd, &rec, &enc, lambda);
                let (mode_scalar, cost_scalar) =
                    reference_combined3_sad(&mut pred_scalar, &rec, &enc, lambda);

                assert_eq!(
                    mode_simd, mode_scalar,
                    "mode mismatch for seed={seed}, lambda={lambda}"
                );
                assert_eq!(
                    cost_simd, cost_scalar,
                    "cost mismatch for seed={seed}, lambda={lambda}"
                );
                assert_eq!(
                    pred_simd, pred_scalar,
                    "pred buffer mismatch for seed={seed}, lambda={lambda}"
                );
            }
        }
    }

    fn test_plane(w: usize, h: usize, pad: usize, stride: usize) -> PaddedPlane {
        let mut p = PaddedPlane::new(w, h, pad, stride);
        for y in -(pad as isize)..(h + pad) as isize {
            for x in -(pad as isize)..(w + pad) as isize {
                p.set(x, y, (((y + 37) * stride as isize + (x + 43)) % 251) as u8);
            }
        }
        p
    }

    fn lcg_u8(seed: &mut u64) -> u8 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 32) as u8
    }

    fn random_plane(seed: &mut u64) -> PaddedPlane {
        let (w, h, pad, stride) = (32usize, 32usize, 16usize, 64usize);
        let mut p = PaddedPlane::new(w, h, pad, stride);
        for y in -(pad as isize)..(h + pad) as isize {
            for x in -(pad as isize)..(w + pad) as isize {
                p.set(x, y, lcg_u8(seed));
            }
        }
        p
    }

    fn constant_plane(val: u8) -> PaddedPlane {
        let (w, h, pad, stride) = (32usize, 32usize, 16usize, 64usize);
        let mut p = PaddedPlane::new(w, h, pad, stride);
        for y in -(pad as isize)..(h + pad) as isize {
            for x in -(pad as isize)..(w + pad) as isize {
                p.set(x, y, val);
            }
        }
        p
    }

    /// Extreme `[255, 0, 0, 0]` column pattern; paired with [`max_lane_rec_plane`]
    /// (`[0, 255, 255, 255]`), each strip reaches the `8160` per-lane Hadamard peak
    /// (`32,640` across four strips).
    fn max_lane_enc_plane() -> PaddedPlane {
        let (w, h, pad, stride) = (32usize, 32usize, 16usize, 64usize);
        let mut p = PaddedPlane::new(w, h, pad, stride);
        for y in -(pad as isize)..(h + pad) as isize {
            for x in -(pad as isize)..(w + pad) as isize {
                p.set(x, y, if x.rem_euclid(4) == 0 { 255 } else { 0 });
            }
        }
        p
    }

    /// Complementary `[0, 255, 255, 255]` column pattern whose top row pairs with
    /// [`max_lane_enc_plane`] to hit the `8160` per-strip / `32,640` four-strip `i16`
    /// Hadamard lane maximum.
    fn max_lane_rec_plane() -> PaddedPlane {
        let (w, h, pad, stride) = (32usize, 32usize, 16usize, 64usize);
        let mut p = PaddedPlane::new(w, h, pad, stride);
        for y in -(pad as isize)..(h + pad) as isize {
            for x in -(pad as isize)..(w + pad) as isize {
                p.set(x, y, if x.rem_euclid(4) == 0 { 0 } else { 255 });
            }
        }
        p
    }

    fn reference_4x4_combined3_satd(
        pred: &mut [u8; 16],
        rec: &RecCursor<'_>,
        enc: &RecCursor<'_>,
        lambda_dc: i32,
        lambda_h: i32,
        lambda_v: i32,
    ) -> (i8, i32) {
        let mut pred_dc = [0u8; 16];
        let mut pred_h = [0u8; 16];
        let mut pred_v = [0u8; 16];
        WelsI4x4LumaPredDc_c(&mut pred_dc, rec);
        WelsI4x4LumaPredH_c(&mut pred_h, rec);
        WelsI4x4LumaPredV_c(&mut pred_v, rec);

        let cost_dc =
            crate::encoder::sample::satd_4x4(&RecCursor::over_owned(&mut pred_dc, 0, 4), enc)
                + lambda_dc;
        let cost_h =
            crate::encoder::sample::satd_4x4(&RecCursor::over_owned(&mut pred_h, 0, 4), enc)
                + lambda_h;
        let cost_v =
            crate::encoder::sample::satd_4x4(&RecCursor::over_owned(&mut pred_v, 0, 4), enc)
                + lambda_v;

        let mut best_mode = 2i8;
        let mut best_cost = cost_dc;
        *pred = pred_dc;
        if cost_h < best_cost {
            best_mode = 1;
            best_cost = cost_h;
            *pred = pred_h;
        }
        if cost_v < best_cost {
            best_mode = 0;
            best_cost = cost_v;
            *pred = pred_v;
        }
        (best_mode, best_cost)
    }

    fn reference_16x16_combined3_satd(
        pred: &mut [u8; 256],
        rec: &RecCursor<'_>,
        enc: &RecCursor<'_>,
        lambda: i32,
    ) -> (u8, i32) {
        let mut pred_v = [0u8; 256];
        let mut pred_h = [0u8; 256];
        let mut pred_dc = [0u8; 256];
        WelsI16x16LumaPredV_c(&mut pred_v, rec);
        WelsI16x16LumaPredH_c(&mut pred_h, rec);
        WelsI16x16LumaPredDc_c(&mut pred_dc, rec);

        let cost_v =
            crate::encoder::sample::satd_16x16(&RecCursor::over_owned(&mut pred_v, 0, 16), enc)
                + lambda;
        let cost_h =
            crate::encoder::sample::satd_16x16(&RecCursor::over_owned(&mut pred_h, 0, 16), enc)
                + lambda * 3;
        let cost_dc =
            crate::encoder::sample::satd_16x16(&RecCursor::over_owned(&mut pred_dc, 0, 16), enc)
                + lambda * 3;

        let mut best_mode = 0u8;
        let mut best_cost = cost_v;
        *pred = pred_v;
        if cost_h < best_cost {
            best_mode = 1;
            best_cost = cost_h;
            *pred = pred_h;
        }
        if cost_dc < best_cost {
            best_mode = 2;
            best_cost = cost_dc;
            *pred = pred_dc;
        }
        (best_mode, best_cost)
    }

    fn reference_chroma_combined3(
        pred_cb_cr: &mut [u8; 128],
        rec_cb: &RecCursor<'_>,
        enc_cb: &RecCursor<'_>,
        rec_cr: &RecCursor<'_>,
        enc_cr: &RecCursor<'_>,
        lambda: i32,
        use_satd: bool,
    ) -> (u8, i32) {
        let cost8x8 = |p: &mut [u8; 64], e: &RecCursor<'_>| -> i32 {
            let c = RecCursor::over_owned(p, 0, 8);
            if use_satd {
                crate::encoder::sample::satd_8x8(&c, e)
            } else {
                crate::common::sad_common::sample_sad::<8, 8, _>(&c, e)
            }
        };

        let mut best_mode = 2u8;
        let mut best_cost = i32::MAX;
        for (mode, ue_bits) in [(2u8, 3), (1u8, 3), (0u8, 1)] {
            let mut cb = [0u8; 64];
            let mut cr = [0u8; 64];
            match mode {
                2 => {
                    WelsIChromaPredV_c(&mut cb, rec_cb);
                    WelsIChromaPredV_c(&mut cr, rec_cr);
                }
                1 => {
                    WelsIChromaPredH_c(&mut cb, rec_cb);
                    WelsIChromaPredH_c(&mut cr, rec_cr);
                }
                _ => {
                    WelsIChromaPredDc_c(&mut cb, rec_cb);
                    WelsIChromaPredDc_c(&mut cr, rec_cr);
                }
            }
            let cur = cost8x8(&mut cb, enc_cb) + cost8x8(&mut cr, enc_cr) + lambda * ue_bits;
            if cur < best_cost {
                best_mode = mode;
                best_cost = cur;
                pred_cb_cr[..64].copy_from_slice(&cb);
                pred_cb_cr[64..].copy_from_slice(&cr);
            }
        }
        (best_mode, best_cost)
    }

    #[test]
    fn test_intra_4x4_combined3_satd_parity() {
        let mut seed = 0x1234_5678_9ABC_DEF0u64;
        let mut pairs = vec![
            (constant_plane(128), constant_plane(128)),
            (constant_plane(255), max_lane_enc_plane()),
            (constant_plane(0), constant_plane(255)),
            (test_plane(32, 32, 16, 64), test_plane(32, 32, 16, 64)),
        ];
        for _ in 0..16 {
            pairs.push((random_plane(&mut seed), random_plane(&mut seed)));
        }

        for (case_idx, (mut p_rec, mut p_enc)) in pairs.into_iter().enumerate() {
            let v_rec = shared_plane_for_test(&mut p_rec);
            let v_enc = shared_plane_for_test(&mut p_enc);
            for (ax, ay) in [(0isize, 0isize), (4, 0), (8, 12), (12, 12)] {
                let rec = v_rec.cursor(ax, ay);
                let enc = v_enc.cursor(ax, ay);
                for (ldc, lh, lv) in [
                    (0, 0, 0),
                    (4, 16, 16),
                    (16, 4, 16),
                    (16, 16, 4),
                    (40, 10, 40),
                ] {
                    let mut pred_simd = [0u8; 16];
                    let mut pred_ref = [0u8; 16];
                    let got = intra_4x4_combined3_satd(&mut pred_simd, &rec, &enc, ldc, lh, lv);
                    let want = reference_4x4_combined3_satd(&mut pred_ref, &rec, &enc, ldc, lh, lv);
                    assert_eq!(
                        got, want,
                        "4x4 case={case_idx} anchor=({ax},{ay}) l=({ldc},{lh},{lv})"
                    );
                    assert_eq!(pred_simd, pred_ref, "4x4 pred case={case_idx}");
                }
            }
        }
    }

    #[test]
    fn test_intra_16x16_combined3_satd_parity() {
        let mut seed = 0xCAFE_BABE_0123_4567u64;
        let mut pairs = vec![
            (constant_plane(128), constant_plane(138)),
            (constant_plane(255), max_lane_enc_plane()),
            (max_lane_rec_plane(), max_lane_enc_plane()),
            (constant_plane(0), constant_plane(255)),
            (constant_plane(255), constant_plane(0)),
            (test_plane(32, 32, 16, 64), test_plane(32, 32, 16, 64)),
        ];
        for _ in 0..16 {
            pairs.push((random_plane(&mut seed), random_plane(&mut seed)));
        }

        for (case_idx, (mut p_rec, mut p_enc)) in pairs.into_iter().enumerate() {
            let v_rec = shared_plane_for_test(&mut p_rec);
            let v_enc = shared_plane_for_test(&mut p_enc);
            for (ax, ay) in [(0isize, 0isize), (16, 0), (0, 16), (8, 8)] {
                let rec = v_rec.cursor(ax, ay);
                let enc = v_enc.cursor(ax, ay);
                for lambda in [0, 1, 7, 25, 100] {
                    let mut pred_ref = [0u8; 256];
                    let want = reference_16x16_combined3_satd(&mut pred_ref, &rec, &enc, lambda);

                    let mut pred_ssse3 = [0u8; 256];
                    let got_ssse3 = intra_16x16_combined3_satd_ssse3(
                        &mut pred_ssse3,
                        &rec,
                        &enc,
                        lambda,
                        ssse3_token(),
                    );
                    assert_eq!(
                        got_ssse3, want,
                        "16x16 ssse3 case={case_idx} lambda={lambda}"
                    );
                    assert_eq!(pred_ssse3, pred_ref, "16x16 ssse3 pred case={case_idx}");

                    if let Some(tok) = Avx2Token::try_new() {
                        let mut pred_avx2 = [0u8; 256];
                        let got_avx2 = intra_16x16_combined3_satd_avx2(
                            &mut pred_avx2,
                            &rec,
                            &enc,
                            lambda,
                            tok,
                        );
                        assert_eq!(got_avx2, want, "16x16 avx2 case={case_idx} lambda={lambda}");
                        assert_eq!(pred_avx2, pred_ref, "16x16 avx2 pred case={case_idx}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_intra_chroma_8x8_combined3_sad_and_satd_parity() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut quads = vec![
            (
                constant_plane(128),
                constant_plane(138),
                constant_plane(128),
                constant_plane(138),
            ),
            (
                constant_plane(255),
                max_lane_enc_plane(),
                constant_plane(255),
                max_lane_enc_plane(),
            ),
            (
                max_lane_rec_plane(),
                max_lane_enc_plane(),
                max_lane_rec_plane(),
                max_lane_enc_plane(),
            ),
            (
                constant_plane(0),
                constant_plane(255),
                constant_plane(255),
                constant_plane(0),
            ),
        ];
        for _ in 0..16 {
            quads.push((
                random_plane(&mut seed),
                random_plane(&mut seed),
                random_plane(&mut seed),
                random_plane(&mut seed),
            ));
        }

        for (case_idx, (mut r_cb, mut e_cb, mut r_cr, mut e_cr)) in quads.into_iter().enumerate() {
            let vr_cb = shared_plane_for_test(&mut r_cb);
            let ve_cb = shared_plane_for_test(&mut e_cb);
            let vr_cr = shared_plane_for_test(&mut r_cr);
            let ve_cr = shared_plane_for_test(&mut e_cr);

            for (ax, ay) in [(0isize, 0isize), (8, 8), (16, 4)] {
                let c_rcb = vr_cb.cursor(ax, ay);
                let c_ecb = ve_cb.cursor(ax, ay);
                let c_rcr = vr_cr.cursor(ax, ay);
                let c_ecr = ve_cr.cursor(ax, ay);

                for lambda in [0, 1, 5, 20, 80] {
                    // SAD
                    let mut ref_sad = [0u8; 128];
                    let want_sad = reference_chroma_combined3(
                        &mut ref_sad,
                        &c_rcb,
                        &c_ecb,
                        &c_rcr,
                        &c_ecr,
                        lambda,
                        false,
                    );

                    let mut sse2_sad = [0u8; 128];
                    let got_sse2_sad = intra_chroma_8x8_combined3_sad_sse2(
                        &mut sse2_sad,
                        &c_rcb,
                        &c_ecb,
                        &c_rcr,
                        &c_ecr,
                        lambda,
                        sse2_token(),
                    );
                    assert_eq!(
                        got_sse2_sad, want_sad,
                        "chroma sad sse2 case={case_idx} l={lambda}"
                    );
                    assert_eq!(sse2_sad, ref_sad, "chroma sad sse2 pred case={case_idx}");

                    if let Some(tok) = Avx2Token::try_new() {
                        let mut avx2_sad = [0u8; 128];
                        let got_avx2_sad = intra_chroma_8x8_combined3_sad_avx2(
                            &mut avx2_sad,
                            &c_rcb,
                            &c_ecb,
                            &c_rcr,
                            &c_ecr,
                            lambda,
                            tok,
                        );
                        assert_eq!(
                            got_avx2_sad, want_sad,
                            "chroma sad avx2 case={case_idx} l={lambda}"
                        );
                        assert_eq!(avx2_sad, ref_sad, "chroma sad avx2 pred case={case_idx}");
                    }

                    // SATD
                    let mut ref_satd = [0u8; 128];
                    let want_satd = reference_chroma_combined3(
                        &mut ref_satd,
                        &c_rcb,
                        &c_ecb,
                        &c_rcr,
                        &c_ecr,
                        lambda,
                        true,
                    );

                    let mut ssse3_satd = [0u8; 128];
                    let got_ssse3_satd = intra_chroma_8x8_combined3_satd_ssse3(
                        &mut ssse3_satd,
                        &c_rcb,
                        &c_ecb,
                        &c_rcr,
                        &c_ecr,
                        lambda,
                        ssse3_token(),
                    );
                    assert_eq!(
                        got_ssse3_satd, want_satd,
                        "chroma satd ssse3 case={case_idx} l={lambda}"
                    );
                    assert_eq!(
                        ssse3_satd, ref_satd,
                        "chroma satd ssse3 pred case={case_idx}"
                    );

                    if let Some(tok) = Avx2Token::try_new() {
                        let mut avx2_satd = [0u8; 128];
                        let got_avx2_satd = intra_chroma_8x8_combined3_satd_avx2(
                            &mut avx2_satd,
                            &c_rcb,
                            &c_ecb,
                            &c_rcr,
                            &c_ecr,
                            lambda,
                            tok,
                        );
                        assert_eq!(
                            got_avx2_satd, want_satd,
                            "chroma satd avx2 case={case_idx} l={lambda}"
                        );
                        assert_eq!(avx2_satd, ref_satd, "chroma satd avx2 pred case={case_idx}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_i16x16_luma_pred_parity() {
        let mut p = test_plane(32, 32, 16, 64);
        let view = shared_plane_for_test(&mut p);
        let rec = view.cursor(0, 0);

        // V
        let mut pred_c = [0u8; 256];
        let mut pred_simd = [0u8; 256];
        WelsI16x16LumaPredV_c(&mut pred_c, &rec);
        enc_i16x16_luma_pred_v(&mut pred_simd, &rec);
        assert_eq!(pred_c, pred_simd, "16x16 V mismatch");

        // H
        let mut pred_c = [0u8; 256];
        let mut pred_simd = [0u8; 256];
        WelsI16x16LumaPredH_c(&mut pred_c, &rec);
        enc_i16x16_luma_pred_h(&mut pred_simd, &rec);
        assert_eq!(pred_c, pred_simd, "16x16 H mismatch");

        // DC
        let mut pred_c = [0u8; 256];
        let mut pred_simd = [0u8; 256];
        WelsI16x16LumaPredDc_c(&mut pred_c, &rec);
        enc_i16x16_luma_pred_dc(&mut pred_simd, &rec);
        assert_eq!(pred_c, pred_simd, "16x16 DC mismatch");

        // Plane
        let mut pred_c = [0u8; 256];
        let mut pred_simd = [0u8; 256];
        WelsI16x16LumaPredPlane_c(&mut pred_c, &rec);
        enc_i16x16_luma_pred_plane(&mut pred_simd, &rec);
        assert_eq!(pred_c, pred_simd, "16x16 Plane mismatch");
    }

    #[test]
    fn test_chroma_pred_parity() {
        let mut p = test_plane(32, 32, 16, 64);
        let view = shared_plane_for_test(&mut p);
        let rec = view.cursor(0, 0);

        // V
        let mut pred_c = [0u8; 64];
        let mut pred_simd = [0u8; 64];
        WelsIChromaPredV_c(&mut pred_c, &rec);
        enc_chroma_pred_v(&mut pred_simd, &rec);
        assert_eq!(pred_c, pred_simd, "Chroma V mismatch");

        // H
        let mut pred_c = [0u8; 64];
        let mut pred_simd = [0u8; 64];
        WelsIChromaPredH_c(&mut pred_c, &rec);
        enc_chroma_pred_h(&mut pred_simd, &rec);
        assert_eq!(pred_c, pred_simd, "Chroma H mismatch");

        // DC
        let mut pred_c = [0u8; 64];
        let mut pred_simd = [0u8; 64];
        WelsIChromaPredDc_c(&mut pred_c, &rec);
        enc_chroma_pred_dc(&mut pred_simd, &rec);
        assert_eq!(pred_c, pred_simd, "Chroma DC mismatch");

        // Plane
        let mut pred_c = [0u8; 64];
        let mut pred_simd = [0u8; 64];
        WelsIChromaPredPlane_c(&mut pred_c, &rec);
        enc_chroma_pred_plane(&mut pred_simd, &rec);
        assert_eq!(pred_c, pred_simd, "Chroma Plane mismatch");
    }

    #[test]
    fn test_i4x4_luma_pred_parity() {
        let mut p = test_plane(32, 32, 16, 64);
        let view = shared_plane_for_test(&mut p);
        let rec = view.cursor(0, 0);

        // V
        let mut pred_c = [0u8; 16];
        let mut pred_simd = [0u8; 16];
        WelsI4x4LumaPredV_c(&mut pred_c, &rec);
        enc_i4x4_luma_pred_v(&mut pred_simd, &rec);
        assert_eq!(pred_c, pred_simd, "4x4 V mismatch");

        // H
        let mut pred_c = [0u8; 16];
        let mut pred_simd = [0u8; 16];
        WelsI4x4LumaPredH_c(&mut pred_c, &rec);
        enc_i4x4_luma_pred_h(&mut pred_simd, &rec);
        assert_eq!(pred_c, pred_simd, "4x4 H mismatch");

        // DC
        let mut pred_c = [0u8; 16];
        let mut pred_simd = [0u8; 16];
        WelsI4x4LumaPredDc_c(&mut pred_c, &rec);
        enc_i4x4_luma_pred_dc(&mut pred_simd, &rec);
        assert_eq!(pred_c, pred_simd, "4x4 DC mismatch");

        // DDL
        let mut pred_c = [0u8; 16];
        let mut pred_simd = [0u8; 16];
        WelsI4x4LumaPredDDL_c(&mut pred_c, &rec);
        enc_i4x4_luma_pred_ddl(&mut pred_simd, &rec);
        assert_eq!(pred_c, pred_simd, "4x4 DDL mismatch");

        // DDR
        let mut pred_c = [0u8; 16];
        let mut pred_simd = [0u8; 16];
        WelsI4x4LumaPredDDR_c(&mut pred_c, &rec);
        enc_i4x4_luma_pred_ddr(&mut pred_simd, &rec);
        assert_eq!(pred_c, pred_simd, "4x4 DDR mismatch");

        // VR
        let mut pred_c = [0u8; 16];
        let mut pred_simd = [0u8; 16];
        WelsI4x4LumaPredVR_c(&mut pred_c, &rec);
        enc_i4x4_luma_pred_vr(&mut pred_simd, &rec);
        assert_eq!(pred_c, pred_simd, "4x4 VR mismatch");

        // HD
        let mut pred_c = [0u8; 16];
        let mut pred_simd = [0u8; 16];
        WelsI4x4LumaPredHD_c(&mut pred_c, &rec);
        enc_i4x4_luma_pred_hd(&mut pred_simd, &rec);
        assert_eq!(pred_c, pred_simd, "4x4 HD mismatch");

        // VL
        let mut pred_c = [0u8; 16];
        let mut pred_simd = [0u8; 16];
        WelsI4x4LumaPredVL_c(&mut pred_c, &rec);
        enc_i4x4_luma_pred_vl(&mut pred_simd, &rec);
        assert_eq!(pred_c, pred_simd, "4x4 VL mismatch");

        // HU
        let mut pred_c = [0u8; 16];
        let mut pred_simd = [0u8; 16];
        WelsI4x4LumaPredHU_c(&mut pred_c, &rec);
        enc_i4x4_luma_pred_hu(&mut pred_simd, &rec);
        assert_eq!(pred_c, pred_simd, "4x4 HU mismatch");
    }

    // ========================================================================
    // The decoder-side predictors.
    //
    // The `dec_*` kernels are the in-place reconstructors the decoder installs:
    // they write back into the picture through a
    // `PlaneCursorMut` whose neighbours are the samples just read, which is where an
    // off-by-one row or column hides, so these compare the whole allocation of two
    // identically built planes rather than the block.
    //
    // The reference, `decoder::get_intra_predictor`, has no SIMD dispatch of its own —
    // the SSE2 kernels are installed over it in the table, never called from it — so
    // no assertion here routes back into the kernel under test.
    // ========================================================================

    /// Two planes with identical content, both padded, for an in-place kernel pair.
    fn twin_pred_planes() -> (PaddedPlane, PaddedPlane) {
        (test_plane(32, 32, 16, 64), test_plane(32, 32, 16, 64))
    }

    /// Runs `scalar` and `simd` on twin planes anchored at `(8, 8)` and compares every
    /// byte of both allocations.
    fn assert_dec_parity(
        name: &str,
        scalar: fn(&mut PlaneCursorMut<'_>),
        simd: fn(&mut PlaneCursorMut<'_>),
    ) {
        let (mut pa, mut pb) = twin_pred_planes();
        assert_eq!(
            pa.as_slice(),
            pb.as_slice(),
            "{name}: twins started out different"
        );
        scalar(&mut pa.cursor_mut(8, 8));
        simd(&mut pb.cursor_mut(8, 8));
        assert_eq!(
            pa.as_slice(),
            pb.as_slice(),
            "{name}: SSE2 and scalar disagree somewhere in the allocation"
        );
    }

    #[test]
    fn dec_i16x16_luma_pred_parity() {
        use crate::decoder::get_intra_predictor as dec;
        assert_dec_parity("16x16 V", dec::i16x16_luma_pred_v, dec_i16x16_luma_pred_v);
        assert_dec_parity("16x16 H", dec::i16x16_luma_pred_h, dec_i16x16_luma_pred_h);
        assert_dec_parity(
            "16x16 DC",
            dec::i16x16_luma_pred_dc,
            dec_i16x16_luma_pred_dc,
        );
        assert_dec_parity(
            "16x16 DC top",
            dec::i16x16_luma_pred_dc_top,
            dec_i16x16_luma_pred_dc_top,
        );
        assert_dec_parity(
            "16x16 DC left",
            dec::i16x16_luma_pred_dc_left,
            dec_i16x16_luma_pred_dc_left,
        );
        assert_dec_parity(
            "16x16 DC n/a",
            dec::i16x16_luma_pred_dc_na,
            dec_i16x16_luma_pred_dc_na,
        );
        assert_dec_parity(
            "16x16 Plane",
            dec::i16x16_luma_pred_plane,
            dec_i16x16_luma_pred_plane,
        );
    }

    #[test]
    fn dec_chroma_pred_parity() {
        use crate::decoder::get_intra_predictor as dec;
        assert_dec_parity("Chroma V", dec::chroma_pred_v, dec_chroma_pred_v);
        assert_dec_parity("Chroma H", dec::chroma_pred_h, dec_chroma_pred_h);
        assert_dec_parity("Chroma DC", dec::chroma_pred_dc, dec_chroma_pred_dc);
        assert_dec_parity(
            "Chroma DC top",
            dec::chroma_pred_dc_top,
            dec_chroma_pred_dc_top,
        );
        assert_dec_parity(
            "Chroma Plane",
            dec::chroma_pred_plane,
            dec_chroma_pred_plane,
        );
    }

    #[test]
    fn dec_i4x4_luma_pred_parity() {
        use crate::decoder::get_intra_predictor as dec;
        assert_dec_parity("4x4 V", dec::i4x4_luma_pred_v, dec_i4x4_luma_pred_v);
        assert_dec_parity("4x4 H", dec::i4x4_luma_pred_h, dec_i4x4_luma_pred_h);
        assert_dec_parity("4x4 DC", dec::i4x4_luma_pred_dc, dec_i4x4_luma_pred_dc);
        assert_dec_parity("4x4 DDL", dec::i4x4_luma_pred_ddl, dec_i4x4_luma_pred_ddl);
        assert_dec_parity("4x4 DDR", dec::i4x4_luma_pred_ddr, dec_i4x4_luma_pred_ddr);
        assert_dec_parity("4x4 VR", dec::i4x4_luma_pred_vr, dec_i4x4_luma_pred_vr);
        assert_dec_parity("4x4 HD", dec::i4x4_luma_pred_hd, dec_i4x4_luma_pred_hd);
        assert_dec_parity("4x4 VL", dec::i4x4_luma_pred_vl, dec_i4x4_luma_pred_vl);
        assert_dec_parity("4x4 HU", dec::i4x4_luma_pred_hu, dec_i4x4_luma_pred_hu);
    }

    /// Every public kernel must reach at least one `_mm_*` intrinsic — in its own body,
    /// or in a function it calls (one hop, which is how the wrappers here delegate to
    /// the shared fills) — unless it is one of the fourteen listed below as scalar by
    /// design. Decided by reading this file's own source.
    #[test]
    fn every_kernel_here_reaches_an_intrinsic() {
        /// The predictors that vectorize nothing, on purpose: a V fill is a row
        /// broadcast, an H fill a byte splat, a DC fill a mean and a splat — each a
        /// word-wide rewrite of its scalar twin.
        const SCALAR_BY_DESIGN: [&str; 14] = [
            "enc_i16x16_luma_pred_v",
            "dec_i16x16_luma_pred_v",
            "enc_i16x16_luma_pred_h",
            "dec_i16x16_luma_pred_h",
            "dec_i16x16_luma_pred_dc_na",
            "enc_chroma_pred_v",
            "dec_chroma_pred_v",
            "enc_chroma_pred_h",
            "dec_chroma_pred_h",
            "enc_chroma_pred_dc",
            "dec_chroma_pred_dc",
            "dec_i4x4_luma_pred_v",
            "dec_i4x4_luma_pred_h",
            "dec_i4x4_luma_pred_dc",
        ];

        // The test module's own `fn`s are not `pub`, but cut it off anyway so a helper
        // named like a kernel cannot be mistaken for one.
        let src = include_str!("intra_pred.rs");
        let src = src
            .split("#[cfg(test)]")
            .next()
            .expect("source before the tests");

        // Body of the item starting at byte `i`, by brace matching.
        fn body_at(s: &str, i: usize) -> &str {
            let b = s.as_bytes();
            let (mut depth, mut j, mut started) = (0usize, i, false);
            while j < b.len() {
                match b[j] {
                    b'{' => {
                        depth += 1;
                        started = true;
                    }
                    // Only once an opening brace has been seen: a `fn ` match inside a
                    // comment can otherwise reach a closing brace first and underflow.
                    b'}' if started => {
                        depth -= 1;
                        if depth == 0 {
                            return &s[i..=j];
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            &s[i..]
        }

        let ident = |s: &str| -> String {
            s.chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect()
        };

        // Every `fn` in the file, by name, so a one-hop call can be resolved.
        let mut bodies: Vec<(String, &str)> = Vec::new();
        for (off, _) in src.match_indices("fn ") {
            let name = ident(&src[off + 3..]);
            if !name.is_empty() {
                bodies.push((name, body_at(src, off)));
            }
        }

        // The public surface: what the init tables can install.
        let mut public: Vec<String> = Vec::new();
        for (off, _) in src.match_indices("pub ") {
            let rest = &src[off + 4..];
            let rest = rest.strip_prefix("(crate) ").unwrap_or(rest);
            let rest = rest.strip_prefix("unsafe ").unwrap_or(rest);
            if let Some(rest) = rest.strip_prefix("fn ") {
                let name = ident(rest);
                if !name.is_empty() {
                    public.push(name);
                }
            }
        }
        assert!(
            public.len() >= 30,
            "found only {} public kernels — the scan broke",
            public.len()
        );

        let intrinsics = |b: &str| b.contains("_mm_");
        let mut offenders = Vec::new();
        for name in &public {
            if SCALAR_BY_DESIGN.contains(&name.as_str()) {
                continue;
            }
            let body = bodies
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, b)| *b)
                .unwrap_or("");
            if intrinsics(body) {
                continue;
            }
            // One hop: any function this body names as a whole identifier that reaches
            // an intrinsic. Substring matching is not enough — `..._dc_na` contains
            // `..._dc`, so a plain `contains` would let an empty kernel borrow a
            // sibling's intrinsics.
            let called: std::collections::HashSet<&str> = body
                .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .filter(|t| !t.is_empty())
                .collect();
            let reaches = bodies.iter().any(|(callee, cbody)| {
                callee != name && called.contains(callee.as_str()) && intrinsics(cbody)
            });
            if !reaches {
                offenders.push(name.clone());
            }
        }

        assert!(
            offenders.is_empty(),
            "these are installed as x86_64 kernels but reach no `_mm_*` intrinsic — \
             implement them, or add them to `SCALAR_BY_DESIGN` and say why \
             (see the module header): {offenders:?}"
        );

        // A name on the list that no longer exists, or that grew intrinsics, is a
        // stale exemption hiding a kernel nobody checks.
        for name in SCALAR_BY_DESIGN {
            let body = bodies.iter().find(|(n, _)| n == name).map(|(_, b)| *b);
            let body = body.unwrap_or_else(|| panic!("`{name}` is exempt but no longer exists"));
            assert!(
                !intrinsics(body),
                "`{name}` is exempt but now has intrinsics — drop it from the list"
            );
        }
    }
}
