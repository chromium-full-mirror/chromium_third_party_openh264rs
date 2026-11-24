// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Intra prediction — `WelsI16x16LumaPred*_AArch64_neon`, `WelsIChromaPred*_AArch64_neon`
//! and `WelsI4x4LumaPred*_AArch64_neon` in `codec/common/arm64/intra_pred_common_aarch64_neon.S`
//! and `codec/encoder/core/arm64/intra_pred_aarch64_neon.S`. The decoder twins in
//! `codec/decoder/core/arm64/intra_pred_aarch64_neon.S` are the same bodies writing
//! back through the stride they read from — the `PredOut` seam below — so each
//! predictor is written once.
//!
//! # What is vectorised
//!
//! The two plane predictors compute their `H`/`V` sums the asm's way — the neighbour
//! differences `mul`led by `[5 .. 40]` or `[17 .. 68]` and reduced — and fill with
//! `mla` and `sqrshrun #5`. The DC means are `uaddlv` reductions (`uaddlp` pairs for
//! the four chroma quadrants). The 4x4 directional predictors are the asm's `ext`,
//! `uaddl`, `uqrshrn` sequences on one eight-byte neighbour line, with the rows read
//! back out of the result at the offsets the asm stores from.
//!
//! Upstream has no arm64 `DDR`, so `enc_i4x4_luma_pred_ddr` follows the idiom of its
//! `HD` and `VR` neighbours: the nine-sample line `l3 .. lt .. t3`, one three-tap pass,
//! four rows at `ext` offsets. Nor has it `V` predictors at any size, a `DC_128` fill,
//! or a 4x4 `H`: those are broadcasts and fills, and
//! `every_kernel_here_reaches_an_intrinsic` lists them as scalar by design.
#![deny(unsafe_code)]

use core::arch::aarch64::*;

use super::lanes::{ld8, ld8_i16, ld16, low4, st16, to8, to16};
use super::neon_token;
use crate::safe::plane::{BlockRows, PlaneCursor, PlaneCursorMut, RefSamples};
use crate::simd::{simd_entry, tokens::NeonToken};

// ============================================================================
// The enc/dec seam
// ============================================================================

/// Where a predictor's rows go: the encoder's packed candidate buffer, or the
/// decoder's picture through the cursor it read its neighbours from.
trait PredOut {
    fn put<const N: usize>(&mut self, dy: usize, row: &[u8; N]);
}

/// The encoder's packed candidate buffer, `W` bytes per row.
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

#[inline(always)]
fn fill_rows<const N: usize, O: PredOut>(out: &mut O, rows: usize, row: &[u8; N]) {
    for dy in 0..rows {
        out.put(dy, row);
    }
}

/// The asm's constant pools.
static INTRA_1_TO_8: [i16; 8] = [5, 10, 15, 20, 25, 30, 35, 40];
static INTRA_M7_TO_0: [i16; 8] = [-7, -6, -5, -4, -3, -2, -1, 0];
static INTRA_P1_TO_8: [i16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
static INTRA_1_TO_4: [i16; 8] = [17, 34, 51, 68, 17, 34, 51, 68];
static INTRA_M3_TO_P4: [i16; 8] = [-3, -2, -1, 0, 1, 2, 3, 4];

/// Eight left neighbours from row `y0` down, as one vector.
#[inline]
#[target_feature(enable = "neon")]
fn left8<S: RefSamples>(src: &S, y0: isize) -> uint8x8_t {
    let mut l = [0u8; 8];
    for (y, v) in l.iter_mut().enumerate() {
        *v = src.at(-1, y0 + y as isize);
    }
    ld8(&l)
}

/// Sixteen left neighbours.
#[inline]
#[target_feature(enable = "neon")]
fn left16<S: RefSamples>(src: &S) -> uint8x16_t {
    let mut l = [0u8; 16];
    for (y, v) in l.iter_mut().enumerate() {
        *v = src.at(-1, y as isize);
    }
    ld16(&l)
}

/// Four rows of four, packed.
#[inline(always)]
fn pack4(r0: [u8; 4], r1: [u8; 4], r2: [u8; 4], r3: [u8; 4]) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[..4].copy_from_slice(&r0);
    out[4..8].copy_from_slice(&r1);
    out[8..12].copy_from_slice(&r2);
    out[12..].copy_from_slice(&r3);
    out
}

/// Lanes `4..8` as an array — the asm's `st1 {v.s}[1]`.
#[inline]
#[target_feature(enable = "neon")]
fn high4(v: uint8x8_t) -> [u8; 4] {
    vget_lane_u32::<1>(vreinterpret_u32_u8(v)).to_ne_bytes()
}

// ============================================================================
// 16x16 luma
// ============================================================================

simd_entry! {
/// `WelsI16x16LumaPredDc_AArch64_neon` and its `DcTop`/`DcLeft` siblings, and the
/// `DC_128` case: `uaddlv` over whichever edges are in scope, then the rounding the
/// asm does with `uqrshrn`.
#[target_feature(enable = "neon")]
fn i16x16_dc_mean<S: RefSamples>(src: &S, use_top: bool, use_left: bool, _t: NeonToken) -> u8 {
    let sum_top = if use_top {
        vaddlvq_u8(ld16(&src.row_n::<16>(-1, 0))) as i32
    } else {
        0
    };
    let sum_left = if use_left {
        vaddlvq_u8(left16(src)) as i32
    } else {
        0
    };
    match (use_top, use_left) {
        (true, true) => ((16 + sum_top + sum_left) >> 5) as u8,
        (true, false) => ((8 + sum_top) >> 4) as u8,
        (false, true) => ((8 + sum_left) >> 4) as u8,
        (false, false) => 0x80,
    }
}

/// `WelsI16x16LumaPredPlane_AArch64_neon`, the coefficient half: `(a, b, c)`.
///
/// The asm loads `top[-1 .. 7]` reversed and `top[8 .. 16]`, subtracts, multiplies
/// by `[5, 10 .. 40]` and `saddlv`s — that is `5 * Σ (i + 1) (top[8 + i] - top[6 - i])`
/// — and `sqrshrn #6`s it to `b`; the same down the left edge gives `c`; `a` is
/// `(top[15] + left[15]) << 4`.
#[target_feature(enable = "neon")]
fn i16x16_plane_coeffs<S: RefSamples>(src: &S, _t: NeonToken) -> (i16, i16, i16) {
    let top_lo = ld8(&src.row_n::<8>(-1, -1));
    let top_hi = ld8(&src.row_n::<8>(-1, 8));
    let left_lo = left8(src, -1);
    let left_hi = left8(src, 8);
    let dt = vreinterpretq_s16_u16(vsubl_u8(top_hi, vrev64_u8(top_lo)));
    let dl = vreinterpretq_s16_u16(vsubl_u8(left_hi, vrev64_u8(left_lo)));
    let weights = ld8_i16(&INTRA_1_TO_8);
    let h = vaddlvq_s16(vmulq_s16(dt, weights));
    let v = vaddlvq_s16(vmulq_s16(dl, weights));
    let b = ((h + 32) >> 6) as i16;
    let c = ((v + 32) >> 6) as i16;
    let a = (vget_lane_u8::<7>(top_hi) as i16 + vget_lane_u8::<7>(left_hi) as i16) << 4;
    (a, b, c)
}

/// The fill half: `a + b (x - 7) + c (y - 7)`, `sqrshrun #5` per row, `c` added
/// between rows.
#[target_feature(enable = "neon")]
fn i16x16_plane_fill<O: PredOut>(out: &mut O, a: i16, b: i16, c: i16, _t: NeonToken) {
    let base = vdupq_n_s16(a.wrapping_sub(c.wrapping_mul(7)));
    let mut lo = vmlaq_n_s16(base, ld8_i16(&INTRA_M7_TO_0), b);
    let mut hi = vmlaq_n_s16(base, ld8_i16(&INTRA_P1_TO_8), b);
    let cv = vdupq_n_s16(c);
    for dy in 0..16 {
        let row = vcombine_u8(vqrshrun_n_s16::<5>(lo), vqrshrun_n_s16::<5>(hi));
        out.put(dy, &to16(row));
        lo = vaddq_s16(lo, cv);
        hi = vaddq_s16(hi, cv);
    }
}

// ============================================================================
// 8x8 chroma
// ============================================================================

/// `WelsIChromaPredDc_AArch64_neon`: `[top | left]` in one register, `uaddlp` twice
/// to the four quadrant sums, `urshr` for the four means, as the two row values the
/// top and bottom halves are filled with.
#[target_feature(enable = "neon")]
fn chroma_dc_rows<S: RefSamples>(src: &S, _t: NeonToken) -> ([u8; 8], [u8; 8]) {
    let both = vcombine_u8(ld8(&src.row_n::<8>(-1, 0)), left8(src, 0));
    let sums = vpaddlq_u16(vpaddlq_u8(both)); // [T0, T1, L0, L1]
    let tl = vadd_u32(vget_low_u32(sums), vget_high_u32(sums)); // [T0 + L0, T1 + L1]
    let quads = vrshrq_n_u32::<2>(sums);
    let pairs = vrshr_n_u32::<3>(tl);
    let mean1 = vget_lane_u32::<0>(pairs) as u8;
    let mean2 = vgetq_lane_u32::<1>(quads) as u8;
    let mean3 = vgetq_lane_u32::<3>(quads) as u8;
    let mean4 = vget_lane_u32::<1>(pairs) as u8;
    (
        [mean1, mean1, mean1, mean1, mean2, mean2, mean2, mean2],
        [mean3, mean3, mean3, mean3, mean4, mean4, mean4, mean4],
    )
}

/// `WelsDecoderIChromaPredDcTop_AArch64_neon`: the neighbour row above folded to its
/// two 4-sample means by `uaddlp` twice and `urshr #2` — the one row every line of the
/// block is filled with.
#[target_feature(enable = "neon")]
fn chroma_dc_top_row<S: RefSamples>(src: &S, _t: NeonToken) -> [u8; 8] {
    let top = ld8(&src.row_n::<8>(-1, 0));
    let sums = vpaddl_u16(vpaddl_u8(top)); // [T0, T1]
    let means = vrshr_n_u32::<2>(sums);
    let m1 = vget_lane_u32::<0>(means) as u8;
    let m2 = vget_lane_u32::<1>(means) as u8;
    [m1, m1, m1, m1, m2, m2, m2, m2]
}

/// `WelsIChromaPredPlane_AArch64_neon`, the coefficient half.
///
/// `[t2 t1 t0 t-1 | l2 l1 l0 l-1]` subtracted from `[t4 .. t7 | l4 .. l7]`,
/// `mul` by `[17, 34, 51, 68]` twice over, `saddlp`/`addp` to `17H` and `17V`,
/// `sqrshrn #5`.
#[target_feature(enable = "neon")]
fn chroma_plane_coeffs<S: RefSamples>(src: &S, _t: NeonToken) -> (i16, i16, i16) {
    let t = src.row_n::<4>(-1, -1);
    let t4 = src.row_n::<4>(-1, 4);
    let inner = ld8(&[
        t[3],
        t[2],
        t[1],
        t[0],
        src.at(-1, 2),
        src.at(-1, 1),
        src.at(-1, 0),
        src.at(-1, -1),
    ]);
    let outer = ld8(&[
        t4[0],
        t4[1],
        t4[2],
        t4[3],
        src.at(-1, 4),
        src.at(-1, 5),
        src.at(-1, 6),
        src.at(-1, 7),
    ]);
    let d = vreinterpretq_s16_u16(vsubl_u8(outer, inner));
    let m = vmulq_s16(d, ld8_i16(&INTRA_1_TO_4));
    let p = vpaddlq_s16(m);
    let hv = vpaddq_s32(p, p);
    let b = ((vgetq_lane_s32::<0>(hv) + 16) >> 5) as i16;
    let c = ((vgetq_lane_s32::<1>(hv) + 16) >> 5) as i16;
    let a = (t4[3] as i16 + src.at(-1, 7) as i16) << 4;
    (a, b, c)
}

/// The fill half: `a + b (x - 3) + c (y - 3)`, `sqrshrun #5`.
#[target_feature(enable = "neon")]
fn chroma_plane_fill<O: PredOut>(out: &mut O, a: i16, b: i16, c: i16, _t: NeonToken) {
    let base = vdupq_n_s16(a.wrapping_sub(c.wrapping_mul(3)));
    let mut row = vmlaq_n_s16(base, ld8_i16(&INTRA_M3_TO_P4), b);
    let cv = vdupq_n_s16(c);
    for dy in 0..8 {
        out.put(dy, &to8(vqrshrun_n_s16::<5>(row)));
        row = vaddq_s16(row, cv);
    }
}

// ============================================================================
// 4x4 luma
// ============================================================================

/// `WelsI4x4LumaPredDc_AArch64_neon`: `[top | left]`, `uaddlv`, `uqrshrn #3`.
#[target_feature(enable = "neon")]
fn i4x4_dc<S: RefSamples>(src: &S, _t: NeonToken) -> u8 {
    let t = src.row_n::<4>(-1, 0);
    let both = ld8(&[
        t[0],
        t[1],
        t[2],
        t[3],
        src.at(-1, 0),
        src.at(-1, 1),
        src.at(-1, 2),
        src.at(-1, 3),
    ]);
    ((vaddlv_u8(both) as u32 + 4) >> 3) as u8
}

/// `WelsI4x4LumaPredDDL_AArch64_neon`: `t0 + 2 t1 + t2` and on, with `t7` repeated
/// past the end, `uqrshrn #2`, rows at `ext` offsets 0 to 3.
#[target_feature(enable = "neon")]
fn i4x4_ddl<S: RefSamples>(src: &S, _t: NeonToken) -> [u8; 16] {
    let top = ld8(&src.row_n::<8>(-1, 0));
    let last = vdup_lane_u8::<7>(top);
    let t1 = vext_u8::<1>(top, last);
    let t2 = vext_u8::<2>(top, last);
    let sum = vaddq_u16(vaddl_u8(t2, top), vshll_n_u8::<1>(t1));
    let r = vqrshrn_n_u16::<2>(sum);
    pack4(
        low4(r),
        low4(vext_u8::<1>(r, r)),
        low4(vext_u8::<2>(r, r)),
        low4(vext_u8::<3>(r, r)),
    )
}

/// `WelsI4x4LumaPredVL_AArch64_neon`: the two-tap and three-tap lines of the top
/// row, rows 0 and 1 from their heads and rows 2 and 3 one lane along.
#[target_feature(enable = "neon")]
fn i4x4_vl<S: RefSamples>(src: &S, _t: NeonToken) -> [u8; 16] {
    let top = ld8(&src.row_n::<8>(-1, 0));
    let pairs = vaddl_u8(vext_u8::<1>(top, top), top);
    let two = vqrshrn_n_u16::<1>(pairs);
    let triples = vaddq_u16(vextq_u16::<1>(pairs, pairs), pairs);
    let three = vqrshrn_n_u16::<2>(triples);
    pack4(
        low4(two),
        low4(three),
        low4(vext_u8::<1>(two, two)),
        low4(vext_u8::<1>(three, three)),
    )
}

/// `WelsI4x4LumaPredVR_AArch64_neon` on the line `l2 l1 l0 lt t0 t1 t2 t3`.
#[target_feature(enable = "neon")]
fn i4x4_vr<S: RefSamples>(src: &S, _t: NeonToken) -> [u8; 16] {
    let t = src.row_n::<4>(-1, 0);
    let line = ld8(&[
        src.at(-1, 2),
        src.at(-1, 1),
        src.at(-1, 0),
        src.at(-1, -1),
        t[0],
        t[1],
        t[2],
        t[3],
    ]);
    let pairs = vaddl_u8(vext_u8::<7>(line, line), line);
    let triples = vaddq_u16(pairs, vextq_u16::<7>(pairs, pairs));
    let three = vqrshrn_n_u16::<2>(triples);
    let two = vqrshrn_n_u16::<1>(pairs);
    let row2 = vset_lane_u8::<4>(vget_lane_u8::<3>(three), vext_u8::<7>(two, two));
    let shifted = vext_u8::<7>(three, three);
    let row3 = vset_lane_u8::<4>(vget_lane_u8::<3>(shifted), shifted);
    pack4(high4(two), high4(three), high4(row2), high4(row3))
}

/// `WelsI4x4LumaPredHU_AArch64_neon` on the line `l3 l3 l3 l3 l0 l1 l2 l3`.
#[target_feature(enable = "neon")]
fn i4x4_hu<S: RefSamples>(src: &S, _t: NeonToken) -> [u8; 16] {
    let l3 = src.at(-1, 3);
    let line = ld8(&[
        l3,
        l3,
        l3,
        l3,
        src.at(-1, 0),
        src.at(-1, 1),
        src.at(-1, 2),
        l3,
    ]);
    let pairs = vaddl_u8(line, vext_u8::<1>(line, line));
    let triples = vaddq_u16(vextq_u16::<1>(pairs, pairs), pairs);
    let two = vqrshrn_n_u16::<1>(pairs);
    let three = vqrshrn_n_u16::<2>(triples);
    let z = vzip2_u8(two, three); // hu0 hu1 hu2 hu3 hu4 hu5 l3 l3
    let z = vset_lane_u8::<7>(l3, vset_lane_u8::<6>(l3, z));
    pack4(low4(z), low4(vext_u8::<2>(z, line)), high4(z), [l3; 4])
}

/// `WelsI4x4LumaPredHD_AArch64_neon` on the line `l3 l2 l1 l0 lt t0 t1 t2`.
#[target_feature(enable = "neon")]
fn i4x4_hd<S: RefSamples>(src: &S, _t: NeonToken) -> [u8; 16] {
    let t = src.row_n::<4>(-1, 0);
    let line = ld8(&[
        src.at(-1, 3),
        src.at(-1, 2),
        src.at(-1, 1),
        src.at(-1, 0),
        src.at(-1, -1),
        t[0],
        t[1],
        t[2],
    ]);
    let pairs = vaddl_u8(line, vext_u8::<1>(line, line));
    let triples = vaddq_u16(vextq_u16::<1>(pairs, pairs), pairs);
    let two = vqrshrn_n_u16::<1>(pairs); // hd6 hd4 hd2 hd0 ..
    let three = vqrshrn_n_u16::<2>(triples); // hd5 hd3 hd1 hd7 hd8 hd9 ..
    let z = vzip1_u8(two, three); // hd6 hd5 hd4 hd3 hd2 hd1 hd0 hd7
    let tail = vreinterpret_u8_u16(vset_lane_u16::<0>(
        vget_lane_u16::<2>(vreinterpret_u16_u8(three)),
        vreinterpret_u16_u8(line),
    ));
    pack4(
        low4(vext_u8::<6>(z, tail)),
        high4(z),
        low4(vext_u8::<2>(z, tail)),
        low4(z),
    )
}

/// Diagonal down-right, in the idiom of `HD` and `VR` — see the header. On the line
/// `l3 l2 l1 l0 lt t0 t1 t2 t3` the three-tap outputs `f[0..7]` are `ddr6 .. ddr4,
/// ddr0 .. ddr3`, and row `r` is `f[3 - r ..]`.
///
/// The line is nine samples, one more than a `d` register holds, so it is read into a
/// `q` register and the three taps are `ext`s of it. Reading the three taps as three
/// overlapping eight-byte loads instead costs a store-forwarding stall apiece, enough
/// to put the kernel level with the scalar loop.
#[target_feature(enable = "neon")]
fn i4x4_ddr<S: RefSamples>(src: &S, _t: NeonToken) -> [u8; 16] {
    let t = src.row_n::<4>(-1, 0);
    let line = ld16(&[
        src.at(-1, 3),
        src.at(-1, 2),
        src.at(-1, 1),
        src.at(-1, 0),
        src.at(-1, -1),
        t[0],
        t[1],
        t[2],
        t[3],
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]);
    let (l0, l1, l2) = (
        vget_low_u8(line),
        vget_low_u8(vextq_u8::<1>(line, line)),
        vget_low_u8(vextq_u8::<2>(line, line)),
    );
    let sum = vaddq_u16(vaddl_u8(l0, l2), vshll_n_u8::<1>(l1));
    let f = vqrshrn_n_u16::<2>(sum);
    pack4(
        low4(vext_u8::<3>(f, f)),
        low4(vext_u8::<2>(f, f)),
        low4(vext_u8::<1>(f, f)),
        low4(f),
    )
}
}

/// Four packed rows into a `PredOut`.
#[inline(always)]
fn put4<O: PredOut>(out: &mut O, rows: &[u8; 16]) {
    for dy in 0..4 {
        let row: &[u8; 4] = rows[dy * 4..][..4].try_into().expect("row");
        out.put(dy, row);
    }
}

// ============================================================================
// The entry points, named as the slots they fill
// ============================================================================

/// `WelsI16x16LumaPredV_AArch64_neon`.
#[inline]
pub fn enc_i16x16_luma_pred_v(pred: &mut [u8; 256], rec: &PlaneCursor<'_>) {
    let top = rec.row_n::<16>(-1, 0);
    fill_rows(&mut Packed::<16>(pred), 16, &top)
}

/// `WelsDecoderI16x16LumaPredV_AArch64_neon`.
#[inline]
pub fn dec_i16x16_luma_pred_v(pred: &mut PlaneCursorMut<'_>) {
    let top = pred.row_n::<16>(-1, 0);
    fill_rows(pred, 16, &top)
}

/// `WelsI16x16LumaPredH_AArch64_neon`.
#[inline]
pub fn enc_i16x16_luma_pred_h(pred: &mut [u8; 256], rec: &PlaneCursor<'_>) {
    let out = &mut Packed::<16>(pred);
    for dy in 0..16 {
        out.put(dy, &[rec.at(-1, dy as isize); 16]);
    }
}

/// `WelsDecoderI16x16LumaPredH_AArch64_neon`.
#[inline]
pub fn dec_i16x16_luma_pred_h(pred: &mut PlaneCursorMut<'_>) {
    for dy in 0..16 {
        let l = pred.at(-1, dy as isize);
        pred.put(dy, &[l; 16]);
    }
}

/// `WelsI16x16LumaPredDc_AArch64_neon`.
#[inline]
pub fn enc_i16x16_luma_pred_dc(pred: &mut [u8; 256], rec: &PlaneCursor<'_>) {
    let mean = i16x16_dc_mean(rec, true, true, neon_token());
    fill_rows(&mut Packed::<16>(pred), 16, &[mean; 16])
}

/// `WelsDecoderI16x16LumaPredDc_AArch64_neon`.
#[inline]
pub fn dec_i16x16_luma_pred_dc(pred: &mut PlaneCursorMut<'_>) {
    let mean = i16x16_dc_mean(pred, true, true, neon_token());
    fill_rows(pred, 16, &[mean; 16])
}

/// `WelsDecoderI16x16LumaPredDcTop_AArch64_neon`.
#[inline]
pub fn dec_i16x16_luma_pred_dc_top(pred: &mut PlaneCursorMut<'_>) {
    let mean = i16x16_dc_mean(pred, true, false, neon_token());
    fill_rows(pred, 16, &[mean; 16])
}

/// `WelsDecoderI16x16LumaPredDcLeft_AArch64_neon`.
#[inline]
pub fn dec_i16x16_luma_pred_dc_left(pred: &mut PlaneCursorMut<'_>) {
    let mean = i16x16_dc_mean(pred, false, true, neon_token());
    fill_rows(pred, 16, &[mean; 16])
}

/// The `DC_128` fill, which upstream keeps in C.
#[inline]
pub fn dec_i16x16_luma_pred_dc_na(pred: &mut PlaneCursorMut<'_>) {
    fill_rows(pred, 16, &[0x80u8; 16])
}

/// `WelsI16x16LumaPredPlane_AArch64_neon`.
#[inline]
pub fn enc_i16x16_luma_pred_plane(pred: &mut [u8; 256], rec: &PlaneCursor<'_>) {
    let tok = neon_token();
    let (a, b, c) = i16x16_plane_coeffs(rec, tok);
    i16x16_plane_fill(&mut Packed::<16>(pred), a, b, c, tok)
}

/// `WelsDecoderI16x16LumaPredPlane_AArch64_neon`.
#[inline]
pub fn dec_i16x16_luma_pred_plane(pred: &mut PlaneCursorMut<'_>) {
    let tok = neon_token();
    let (a, b, c) = i16x16_plane_coeffs(pred, tok);
    i16x16_plane_fill(pred, a, b, c, tok)
}

/// `WelsIChromaPredV_AArch64_neon`.
#[inline]
pub fn enc_chroma_pred_v(pred: &mut [u8; 64], rec: &PlaneCursor<'_>) {
    let top = rec.row_n::<8>(-1, 0);
    fill_rows(&mut Packed::<8>(pred), 8, &top)
}

/// `WelsDecoderIChromaPredV_AArch64_neon`.
#[inline]
pub fn dec_chroma_pred_v(pred: &mut PlaneCursorMut<'_>) {
    let top = pred.row_n::<8>(-1, 0);
    fill_rows(pred, 8, &top)
}

/// `WelsIChromaPredH_AArch64_neon`.
#[inline]
pub fn enc_chroma_pred_h(pred: &mut [u8; 64], rec: &PlaneCursor<'_>) {
    let out = &mut Packed::<8>(pred);
    for dy in 0..8 {
        out.put(dy, &[rec.at(-1, dy as isize); 8]);
    }
}

/// `WelsDecoderIChromaPredH_AArch64_neon`.
#[inline]
pub fn dec_chroma_pred_h(pred: &mut PlaneCursorMut<'_>) {
    for dy in 0..8 {
        let l = pred.at(-1, dy as isize);
        pred.put(dy, &[l; 8]);
    }
}

/// `WelsIChromaPredDc_AArch64_neon`.
#[inline]
pub fn enc_chroma_pred_dc(pred: &mut [u8; 64], rec: &PlaneCursor<'_>) {
    let (top, bot) = chroma_dc_rows(rec, neon_token());
    let out = &mut Packed::<8>(pred);
    for dy in 0..4 {
        out.put(dy, &top);
    }
    for dy in 4..8 {
        out.put(dy, &bot);
    }
}

/// `WelsDecoderIChromaPredDc_AArch64_neon`.
#[inline]
pub fn dec_chroma_pred_dc(pred: &mut PlaneCursorMut<'_>) {
    let (top, bot) = chroma_dc_rows(pred, neon_token());
    for dy in 0..4 {
        pred.put(dy, &top);
    }
    for dy in 4..8 {
        pred.put(dy, &bot);
    }
}

/// `WelsDecoderIChromaPredDcTop_AArch64_neon`.
#[inline]
pub fn dec_chroma_pred_dc_top(pred: &mut PlaneCursorMut<'_>) {
    let row = chroma_dc_top_row(pred, neon_token());
    fill_rows(pred, 8, &row)
}

/// `WelsIChromaPredPlane_AArch64_neon`.
#[inline]
pub fn enc_chroma_pred_plane(pred: &mut [u8; 64], rec: &PlaneCursor<'_>) {
    let tok = neon_token();
    let (a, b, c) = chroma_plane_coeffs(rec, tok);
    chroma_plane_fill(&mut Packed::<8>(pred), a, b, c, tok)
}

/// `WelsDecoderIChromaPredPlane_AArch64_neon`.
#[inline]
pub fn dec_chroma_pred_plane(pred: &mut PlaneCursorMut<'_>) {
    let tok = neon_token();
    let (a, b, c) = chroma_plane_coeffs(pred, tok);
    chroma_plane_fill(pred, a, b, c, tok)
}

/// The 4x4 vertical fill, which upstream keeps in C.
#[inline]
pub fn enc_i4x4_luma_pred_v(pred: &mut [u8; 16], rec: &PlaneCursor<'_>) {
    let top = rec.row_n::<4>(-1, 0);
    put4(&mut Packed::<4>(pred), &pack4(top, top, top, top))
}

/// The 4x4 vertical fill, in place.
#[inline]
pub fn dec_i4x4_luma_pred_v(pred: &mut PlaneCursorMut<'_>) {
    let top = pred.row_n::<4>(-1, 0);
    fill_rows(pred, 4, &top)
}

/// `WelsI4x4LumaPredH_AArch64_neon`.
#[inline]
pub fn enc_i4x4_luma_pred_h(pred: &mut [u8; 16], rec: &PlaneCursor<'_>) {
    let out = &mut Packed::<4>(pred);
    for dy in 0..4 {
        out.put(dy, &[rec.at(-1, dy as isize); 4]);
    }
}

/// `WelsDecoderI4x4LumaPredH_AArch64_neon`.
#[inline]
pub fn dec_i4x4_luma_pred_h(pred: &mut PlaneCursorMut<'_>) {
    for dy in 0..4 {
        let l = pred.at(-1, dy as isize);
        pred.put(dy, &[l; 4]);
    }
}

/// `WelsI4x4LumaPredDc_AArch64_neon`.
#[inline]
pub fn enc_i4x4_luma_pred_dc(pred: &mut [u8; 16], rec: &PlaneCursor<'_>) {
    let mean = i4x4_dc(rec, neon_token());
    *pred = [mean; 16];
}

/// `WelsDecoderI4x4LumaPredDc_AArch64_neon`.
#[inline]
pub fn dec_i4x4_luma_pred_dc(pred: &mut PlaneCursorMut<'_>) {
    let mean = i4x4_dc(pred, neon_token());
    fill_rows(pred, 4, &[mean; 4])
}

/// `WelsI4x4LumaPredDDL_AArch64_neon`.
#[inline]
pub fn enc_i4x4_luma_pred_ddl(pred: &mut [u8; 16], rec: &PlaneCursor<'_>) {
    *pred = i4x4_ddl(rec, neon_token());
}

/// `WelsDecoderI4x4LumaPredDDL_AArch64_neon`.
#[inline]
pub fn dec_i4x4_luma_pred_ddl(pred: &mut PlaneCursorMut<'_>) {
    let rows = i4x4_ddl(&*pred, neon_token());
    put4(pred, &rows)
}

/// Diagonal down-right; see the header.
#[inline]
pub fn enc_i4x4_luma_pred_ddr(pred: &mut [u8; 16], rec: &PlaneCursor<'_>) {
    *pred = i4x4_ddr(rec, neon_token());
}

/// `WelsDecoderI4x4LumaPredDDR_AArch64_neon`.
#[inline]
pub fn dec_i4x4_luma_pred_ddr(pred: &mut PlaneCursorMut<'_>) {
    let rows = i4x4_ddr(&*pred, neon_token());
    put4(pred, &rows)
}

/// `WelsI4x4LumaPredVR_AArch64_neon`.
#[inline]
pub fn enc_i4x4_luma_pred_vr(pred: &mut [u8; 16], rec: &PlaneCursor<'_>) {
    *pred = i4x4_vr(rec, neon_token());
}

/// `WelsDecoderI4x4LumaPredVR_AArch64_neon`.
#[inline]
pub fn dec_i4x4_luma_pred_vr(pred: &mut PlaneCursorMut<'_>) {
    let rows = i4x4_vr(&*pred, neon_token());
    put4(pred, &rows)
}

/// `WelsI4x4LumaPredHD_AArch64_neon`.
#[inline]
pub fn enc_i4x4_luma_pred_hd(pred: &mut [u8; 16], rec: &PlaneCursor<'_>) {
    *pred = i4x4_hd(rec, neon_token());
}

/// `WelsDecoderI4x4LumaPredHD_AArch64_neon`.
#[inline]
pub fn dec_i4x4_luma_pred_hd(pred: &mut PlaneCursorMut<'_>) {
    let rows = i4x4_hd(&*pred, neon_token());
    put4(pred, &rows)
}

/// `WelsI4x4LumaPredVL_AArch64_neon`.
#[inline]
pub fn enc_i4x4_luma_pred_vl(pred: &mut [u8; 16], rec: &PlaneCursor<'_>) {
    *pred = i4x4_vl(rec, neon_token());
}

/// `WelsDecoderI4x4LumaPredVL_AArch64_neon`.
#[inline]
pub fn dec_i4x4_luma_pred_vl(pred: &mut PlaneCursorMut<'_>) {
    let rows = i4x4_vl(&*pred, neon_token());
    put4(pred, &rows)
}

/// `WelsI4x4LumaPredHU_AArch64_neon`.
#[inline]
pub fn enc_i4x4_luma_pred_hu(pred: &mut [u8; 16], rec: &PlaneCursor<'_>) {
    *pred = i4x4_hu(rec, neon_token());
}

/// `WelsDecoderI4x4LumaPredHU_AArch64_neon`.
#[inline]
pub fn dec_i4x4_luma_pred_hu(pred: &mut PlaneCursorMut<'_>) {
    let rows = i4x4_hu(&*pred, neon_token());
    put4(pred, &rows)
}

#[inline(always)]
pub fn intra_16x16_combined3_sad(
    pred: &mut [u8; 256],
    rec: &PlaneCursor<'_>,
    enc: &PlaneCursor<'_>,
    lambda: i32,
) -> (u8, i32) {
    intra_16x16_combined3_sad_neon(pred, rec, enc, lambda, neon_token())
}

simd_entry! {
#[target_feature(enable = "neon")]
fn intra_16x16_combined3_sad_neon(
    pred: &mut [u8; 256],
    rec: &PlaneCursor<'_>,
    enc: &PlaneCursor<'_>,
    lambda: i32,
    _t: NeonToken,
) -> (u8, i32) {
    let top = rec.row_n::<16>(-1, 0);
    let v_vec = ld16(&top);
    let sum_top = vaddlvq_u8(v_vec) as i32;

    let mut left = [0u8; 16];
    let mut sum_left: i32 = 0;
    for y in 0..16 {
        let val = rec.at(-1, y as isize);
        left[y] = val;
        sum_left += val as i32;
    }

    let dc_val = ((16 + sum_top + sum_left) >> 5) as u8;
    let dc_vec = vdupq_n_u8(dc_val);

    let mut acc_v = vdupq_n_u32(0);
    let mut acc_h = vdupq_n_u32(0);
    let mut acc_dc = vdupq_n_u32(0);

    for y in 0..16 {
        let enc_row = enc.row_n::<16>(y as isize, 0);
        let enc_vec = ld16(&enc_row);
        let h_vec = vdupq_n_u8(left[y]);

        let diff_v = vabdq_u8(enc_vec, v_vec);
        let diff_h = vabdq_u8(enc_vec, h_vec);
        let diff_dc = vabdq_u8(enc_vec, dc_vec);

        acc_v = vpadalq_u16(acc_v, vpaddlq_u8(diff_v));
        acc_h = vpadalq_u16(acc_h, vpaddlq_u8(diff_h));
        acc_dc = vpadalq_u16(acc_dc, vpaddlq_u8(diff_dc));
    }

    let sad_v = vaddvq_u32(acc_v) as i32;
    let sad_h = vaddvq_u32(acc_h) as i32;
    let sad_dc = vaddvq_u32(acc_dc) as i32;

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
        0 => {
            for y in 0..16 {
                pred[y * 16..(y + 1) * 16].copy_from_slice(&top);
            }
        }
        1 => {
            for y in 0..16 {
                pred[y * 16..(y + 1) * 16].fill(left[y]);
            }
        }
        _ => {
            pred.fill(dc_val);
        }
    }

    (best_mode, best_cost)
}
}

type EncSpan<'a> = <PlaneCursor<'a> as RefSamples>::Span<'a>;

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

#[target_feature(enable = "neon")]
#[inline]
fn hdm4_neon(
    a0: int16x8_t,
    a1: int16x8_t,
    a2: int16x8_t,
    a3: int16x8_t,
) -> (int16x8_t, int16x8_t, int16x8_t, int16x8_t) {
    let p0 = vaddq_s16(a0, a1);
    let p1 = vaddq_s16(a2, a3);
    let m0 = vsubq_s16(a0, a1);
    let m1 = vsubq_s16(a2, a3);
    (
        vaddq_s16(p0, p1),
        vsubq_s16(p0, p1),
        vaddq_s16(m0, m1),
        vsubq_s16(m0, m1),
    )
}

#[target_feature(enable = "neon")]
#[inline]
fn transpose_4x4x2_neon(
    v0: int16x8_t,
    v1: int16x8_t,
    v2: int16x8_t,
    v3: int16x8_t,
) -> (int16x8_t, int16x8_t, int16x8_t, int16x8_t) {
    let t01_lo = vzip1q_s16(v0, v1);
    let t23_lo = vzip1q_s16(v2, v3);
    let t01_hi = vzip2q_s16(v0, v1);
    let t23_hi = vzip2q_s16(v2, v3);

    let c01_b0 = vreinterpretq_s16_s32(vzip1q_s32(
        vreinterpretq_s32_s16(t01_lo),
        vreinterpretq_s32_s16(t23_lo),
    ));
    let c23_b0 = vreinterpretq_s16_s32(vzip2q_s32(
        vreinterpretq_s32_s16(t01_lo),
        vreinterpretq_s32_s16(t23_lo),
    ));
    let c01_b1 = vreinterpretq_s16_s32(vzip1q_s32(
        vreinterpretq_s32_s16(t01_hi),
        vreinterpretq_s32_s16(t23_hi),
    ));
    let c23_b1 = vreinterpretq_s16_s32(vzip2q_s32(
        vreinterpretq_s32_s16(t01_hi),
        vreinterpretq_s32_s16(t23_hi),
    ));

    (
        vreinterpretq_s16_s64(vzip1q_s64(
            vreinterpretq_s64_s16(c01_b0),
            vreinterpretq_s64_s16(c01_b1),
        )),
        vreinterpretq_s16_s64(vzip2q_s64(
            vreinterpretq_s64_s16(c01_b0),
            vreinterpretq_s64_s16(c01_b1),
        )),
        vreinterpretq_s16_s64(vzip1q_s64(
            vreinterpretq_s64_s16(c23_b0),
            vreinterpretq_s64_s16(c23_b1),
        )),
        vreinterpretq_s16_s64(vzip2q_s64(
            vreinterpretq_s64_s16(c23_b0),
            vreinterpretq_s64_s16(c23_b1),
        )),
    )
}

#[target_feature(enable = "neon")]
#[inline]
fn top_v_subs_neon(top8: uint8x8_t) -> (int16x8_t, int16x8_t, int16x8_t, int16x8_t) {
    let top_w4 = vreinterpretq_s16_u16(vshll_n_u8::<2>(top8));
    let zero = vdupq_n_s16(0);
    let (c0, c1, c2, c3) = transpose_4x4x2_neon(top_w4, zero, zero, zero);
    hdm4_neon(c0, c1, c2, c3)
}

#[target_feature(enable = "neon")]
#[inline]
fn satd_strip_8x4_neon(
    e0: &[u8],
    e1: &[u8],
    e2: &[u8],
    e3: &[u8],
    v_subs: (int16x8_t, int16x8_t, int16x8_t, int16x8_t),
    h_vec: int16x8_t,
    dc_vec: int16x8_t,
) -> (uint16x8_t, uint16x8_t, uint16x8_t) {
    let r0 = vreinterpretq_s16_u16(vmovl_u8(ld8(e0)));
    let r1 = vreinterpretq_s16_u16(vmovl_u8(ld8(e1)));
    let r2 = vreinterpretq_s16_u16(vmovl_u8(ld8(e2)));
    let r3 = vreinterpretq_s16_u16(vmovl_u8(ld8(e3)));

    let (v0, v1, v2, v3) = hdm4_neon(r0, r1, r2, r3);
    let (c0, c1, c2, c3) = transpose_4x4x2_neon(v0, v1, v2, v3);
    let (f0, f1, f2, f3) = hdm4_neon(c0, c1, c2, c3);

    let dv0 = vreinterpretq_u16_s16(vabsq_s16(vsubq_s16(f0, v_subs.0)));
    let dv1 = vreinterpretq_u16_s16(vabsq_s16(vsubq_s16(f1, v_subs.1)));
    let dv2 = vreinterpretq_u16_s16(vabsq_s16(vsubq_s16(f2, v_subs.2)));
    let dv3 = vreinterpretq_u16_s16(vabsq_s16(vsubq_s16(f3, v_subs.3)));
    let strip_v = vaddq_u16(vaddq_u16(dv0, dv1), vaddq_u16(dv2, dv3));

    let abs123 = vaddq_u16(
        vreinterpretq_u16_s16(vabsq_s16(f1)),
        vaddq_u16(
            vreinterpretq_u16_s16(vabsq_s16(f2)),
            vreinterpretq_u16_s16(vabsq_s16(f3)),
        ),
    );
    let strip_h = vaddq_u16(
        vreinterpretq_u16_s16(vabsq_s16(vsubq_s16(f0, h_vec))),
        abs123,
    );
    let strip_dc = vaddq_u16(
        vreinterpretq_u16_s16(vabsq_s16(vsubq_s16(f0, dc_vec))),
        abs123,
    );

    (strip_v, strip_h, strip_dc)
}

#[target_feature(enable = "neon")]
#[inline]
fn write_chroma_pred_neon(
    dst: &mut [u8; 64],
    mode: u8,
    top: &[u8; 8],
    left: &[u8; 8],
    dc_top: &[u8; 8],
    dc_bot: &[u8; 8],
) {
    match mode {
        2 => {
            let v = vcombine_u8(ld8(top), ld8(top));
            for chunk in dst.as_chunks_mut::<16>().0 {
                st16(chunk, v);
            }
        }
        1 => {
            for (step, chunk) in dst.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                let y = step * 2;
                st16(
                    chunk,
                    vcombine_u8(vdup_n_u8(left[y]), vdup_n_u8(left[y + 1])),
                );
            }
        }
        _ => {
            let vt = vcombine_u8(ld8(dc_top), ld8(dc_top));
            let vb = vcombine_u8(ld8(dc_bot), ld8(dc_bot));
            st16(&mut dst[..16], vt);
            st16(&mut dst[16..32], vt);
            st16(&mut dst[32..48], vb);
            st16(&mut dst[48..64], vb);
        }
    }
}

#[inline(always)]
pub fn intra_4x4_combined3_satd(
    pred: &mut [u8; 16],
    rec: &PlaneCursor<'_>,
    enc: &PlaneCursor<'_>,
    lambda_dc: i32,
    lambda_h: i32,
    lambda_v: i32,
) -> (i8, i32) {
    intra_4x4_combined3_satd_neon(pred, rec, enc, lambda_dc, lambda_h, lambda_v, neon_token())
}

#[inline(always)]
pub fn intra_16x16_combined3_satd(
    pred: &mut [u8; 256],
    rec: &PlaneCursor<'_>,
    enc: &PlaneCursor<'_>,
    lambda: i32,
) -> (u8, i32) {
    intra_16x16_combined3_satd_neon(pred, rec, enc, lambda, neon_token())
}

#[inline(always)]
pub fn intra_chroma_8x8_combined3_sad(
    pred_cb_cr: &mut [u8; 128],
    rec_cb: &PlaneCursor<'_>,
    enc_cb: &PlaneCursor<'_>,
    rec_cr: &PlaneCursor<'_>,
    enc_cr: &PlaneCursor<'_>,
    lambda: i32,
) -> (u8, i32) {
    intra_chroma_8x8_combined3_sad_neon(
        pred_cb_cr,
        rec_cb,
        enc_cb,
        rec_cr,
        enc_cr,
        lambda,
        neon_token(),
    )
}

#[inline(always)]
pub fn intra_chroma_8x8_combined3_satd(
    pred_cb_cr: &mut [u8; 128],
    rec_cb: &PlaneCursor<'_>,
    enc_cb: &PlaneCursor<'_>,
    rec_cr: &PlaneCursor<'_>,
    enc_cr: &PlaneCursor<'_>,
    lambda: i32,
) -> (u8, i32) {
    intra_chroma_8x8_combined3_satd_neon(
        pred_cb_cr,
        rec_cb,
        enc_cb,
        rec_cr,
        enc_cr,
        lambda,
        neon_token(),
    )
}

simd_entry! {
#[target_feature(enable = "neon")]
fn intra_4x4_combined3_satd_neon(
    pred: &mut [u8; 16],
    rec: &PlaneCursor<'_>,
    enc: &PlaneCursor<'_>,
    lambda_dc: i32,
    lambda_h: i32,
    lambda_v: i32,
    _t: NeonToken,
) -> (i8, i32) {
    let top = rec.row_n::<4>(-1, 0);
    let left = [
        rec.at(-1, 0),
        rec.at(-1, 1),
        rec.at(-1, 2),
        rec.at(-1, 3),
    ];
    let enc_span = enc.span::<4, 4>(0, 0);
    let (q, st) = enc_span.all_1_quad::<4>();
    let (e0, e1, e2, e3) = EncSpan::read_quad::<4>(q, st);

    let r02 = vreinterpretq_s16_u16(vmovl_u8(ld8(&[
        e0[0], e0[1], e0[2], e0[3], e2[0], e2[1], e2[2], e2[3],
    ])));
    let r13 = vreinterpretq_s16_u16(vmovl_u8(ld8(&[
        e1[0], e1[1], e1[2], e1[3], e3[0], e3[1], e3[2], e3[3],
    ])));

    let s01 = vaddq_s16(r02, r13);
    let d01 = vsubq_s16(r02, r13);
    let vlo = vreinterpretq_s16_s64(vzip1q_s64(vreinterpretq_s64_s16(s01), vreinterpretq_s64_s16(d01)));
    let vhi = vreinterpretq_s16_s64(vzip2q_s64(vreinterpretq_s64_s16(s01), vreinterpretq_s64_s16(d01)));
    let v02 = vaddq_s16(vlo, vhi);
    let v13 = vsubq_s16(vlo, vhi);

    let t0 = vzip1q_s16(v02, v13);
    let t1 = vzip2q_s16(v02, v13);
    let col01 = vreinterpretq_s16_s32(vzip1q_s32(vreinterpretq_s32_s16(t0), vreinterpretq_s32_s16(t1)));
    let col23 = vreinterpretq_s16_s32(vzip2q_s32(vreinterpretq_s32_s16(t0), vreinterpretq_s32_s16(t1)));

    let cs = vaddq_s16(col01, col23);
    let cd = vsubq_s16(col01, col23);
    let clo = vreinterpretq_s16_s64(vzip1q_s64(vreinterpretq_s64_s16(cs), vreinterpretq_s64_s16(cd)));
    let chi = vreinterpretq_s16_s64(vzip2q_s64(vreinterpretq_s64_s16(cs), vreinterpretq_s64_s16(cd)));
    let f02 = vaddq_s16(clo, chi);
    let f13 = vsubq_s16(clo, chi);

    let (sum_t, ht) = hdm4_u8x4_scaled4(top);
    let (sum_l, hl) = hdm4_u8x4_scaled4(left);
    let dc_val = ((sum_t + sum_l + 4) >> 3) as u8;

    let v_sub_02 = ld8_i16(&[ht[0], 0, 0, 0, ht[1], 0, 0, 0]);
    let v_sub_13 = ld8_i16(&[ht[2], 0, 0, 0, ht[3], 0, 0, 0]);
    let h_sub_02 = ld8_i16(&[hl[0], hl[1], hl[2], hl[3], 0, 0, 0, 0]);
    let dc_sub_02 = ld8_i16(&[(dc_val as i16) << 4, 0, 0, 0, 0, 0, 0, 0]);

    let abs_f13 = vreinterpretq_u16_s16(vabsq_s16(f13));
    let vec_v = vaddq_u16(
        vreinterpretq_u16_s16(vabsq_s16(vsubq_s16(f02, v_sub_02))),
        vreinterpretq_u16_s16(vabsq_s16(vsubq_s16(f13, v_sub_13))),
    );
    let vec_h = vaddq_u16(vreinterpretq_u16_s16(vabsq_s16(vsubq_s16(f02, h_sub_02))), abs_f13);
    let vec_dc = vaddq_u16(vreinterpretq_u16_s16(vabsq_s16(vsubq_s16(f02, dc_sub_02))), abs_f13);

    let satd_v = (vaddlvq_u16(vec_v) >> 1) as i32;
    let satd_h = (vaddlvq_u16(vec_h) >> 1) as i32;
    let satd_dc = (vaddlvq_u16(vec_dc) >> 1) as i32;

    let cost_dc = satd_dc + lambda_dc;
    let cost_h = satd_h + lambda_h;
    let cost_v = satd_v + lambda_v;

    if cost_dc <= cost_h && cost_dc <= cost_v {
        st16(pred, vdupq_n_u8(dc_val));
        (2i8, cost_dc)
    } else if cost_h <= cost_v {
        *pred = pack4([left[0]; 4], [left[1]; 4], [left[2]; 4], [left[3]; 4]);
        (1i8, cost_h)
    } else {
        st16(pred, vreinterpretq_u8_u32(vdupq_n_u32(u32::from_ne_bytes(top))));
        (0i8, cost_v)
    }
}

#[target_feature(enable = "neon")]
fn intra_16x16_combined3_satd_neon(
    pred: &mut [u8; 256],
    rec: &PlaneCursor<'_>,
    enc: &PlaneCursor<'_>,
    lambda: i32,
    _t: NeonToken,
) -> (u8, i32) {
    let top = rec.row_n::<16>(-1, 0);
    let v_vec = ld16(&top);
    let sum_top = vaddlvq_u8(v_vec) as i32;

    let mut left = [0u8; 16];
    for (y, dst) in left.iter_mut().enumerate() {
        *dst = rec.at(-1, y as isize);
    }
    let enc_span = enc.span::<16, 16>(0, 0);
    let (qs, st) = enc_span.all_4_quads::<16>();

    let mut sum_left = 0i32;
    let mut h_vecs = [vdupq_n_s16(0); 4];
    for s in 0..4 {
        let (sl, hl) = hdm4_u8x4_scaled4([
            left[s * 4],
            left[s * 4 + 1],
            left[s * 4 + 2],
            left[s * 4 + 3],
        ]);
        sum_left += sl;
        h_vecs[s] = ld8_i16(&[hl[0], hl[1], hl[2], hl[3], hl[0], hl[1], hl[2], hl[3]]);
    }

    let dc_val = ((16 + sum_top + sum_left) >> 5) as u8;
    let dc16 = (dc_val as i16) << 4;
    let dc_vec = ld8_i16(&[dc16, 0, 0, 0, dc16, 0, 0, 0]);
    let v_subs_lo = top_v_subs_neon(vget_low_u8(v_vec));
    let v_subs_hi = top_v_subs_neon(vget_high_u8(v_vec));

    let mut acc_v = vdupq_n_u16(0);
    let mut acc_h = vdupq_n_u16(0);
    let mut acc_dc = vdupq_n_u16(0);

    for (s, q) in qs.into_iter().enumerate() {
        let (e0, e1, e2, e3) = EncSpan::read_quad::<16>(q, st);
        let (sv0, sh0, sdc0) =
            satd_strip_8x4_neon(&e0[..8], &e1[..8], &e2[..8], &e3[..8], v_subs_lo, h_vecs[s], dc_vec);
        let (sv1, sh1, sdc1) =
            satd_strip_8x4_neon(&e0[8..], &e1[8..], &e2[8..], &e3[8..], v_subs_hi, h_vecs[s], dc_vec);
        acc_v = vaddq_u16(acc_v, vaddq_u16(sv0, sv1));
        acc_h = vaddq_u16(acc_h, vaddq_u16(sh0, sh1));
        acc_dc = vaddq_u16(acc_dc, vaddq_u16(sdc0, sdc1));
    }

    let satd_v = (vaddlvq_u16(acc_v) >> 1) as i32;
    let satd_h = (vaddlvq_u16(acc_h) >> 1) as i32;
    let satd_dc = (vaddlvq_u16(acc_dc) >> 1) as i32;

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
                st16(chunk, vdupq_n_u8(left[y]));
            }
        }
        _ => {
            let dv = vdupq_n_u8(dc_val);
            for chunk in pred.as_chunks_mut::<16>().0 {
                st16(chunk, dv);
            }
        }
    }

    (best_mode, best_cost)
}

#[target_feature(enable = "neon")]
fn intra_chroma_8x8_combined3_sad_neon(
    pred_cb_cr: &mut [u8; 128],
    rec_cb: &PlaneCursor<'_>,
    enc_cb: &PlaneCursor<'_>,
    rec_cr: &PlaneCursor<'_>,
    enc_cr: &PlaneCursor<'_>,
    lambda: i32,
    _t: NeonToken,
) -> (u8, i32) {
    let top_cb = rec_cb.row_n::<8>(-1, 0);
    let top_cr = rec_cr.row_n::<8>(-1, 0);
    let mut left_cb = [0u8; 8];
    let mut left_cr = [0u8; 8];
    for y in 0..8 {
        left_cb[y] = rec_cb.at(-1, y as isize);
        left_cr[y] = rec_cr.at(-1, y as isize);
    }
    let enc_cb_span = enc_cb.span::<8, 8>(0, 0);
    let enc_cr_span = enc_cr.span::<8, 8>(0, 0);
    let ([qb0, qb1], stb) = enc_cb_span.all_2_quads::<8>();
    let ([qr0, qr1], str_) = enc_cr_span.all_2_quads::<8>();

    let (dc_top_cb, dc_bot_cb) = chroma_dc_rows(rec_cb, _t);
    let (dc_top_cr, dc_bot_cr) = chroma_dc_rows(rec_cr, _t);

    let v_vec = vcombine_u8(ld8(&top_cb), ld8(&top_cr));
    let dc_top_vec = vcombine_u8(ld8(&dc_top_cb), ld8(&dc_top_cr));
    let dc_bot_vec = vcombine_u8(ld8(&dc_bot_cb), ld8(&dc_bot_cr));

    let mut acc_v = vdupq_n_u32(0);
    let mut acc_h = vdupq_n_u32(0);
    let mut acc_dc = vdupq_n_u32(0);

    let (cb0, cb1, cb2, cb3) = EncSpan::read_quad::<8>(qb0, stb);
    let (cr0, cr1, cr2, cr3) = EncSpan::read_quad::<8>(qr0, str_);
    for (y, (cb_r, cr_r)) in [(cb0, cr0), (cb1, cr1), (cb2, cr2), (cb3, cr3)]
        .into_iter()
        .enumerate()
    {
        let enc_vec = vcombine_u8(ld8(&cb_r), ld8(&cr_r));
        let h_vec = vcombine_u8(vdup_n_u8(left_cb[y]), vdup_n_u8(left_cr[y]));
        acc_v = vpadalq_u16(acc_v, vpaddlq_u8(vabdq_u8(enc_vec, v_vec)));
        acc_h = vpadalq_u16(acc_h, vpaddlq_u8(vabdq_u8(enc_vec, h_vec)));
        acc_dc = vpadalq_u16(acc_dc, vpaddlq_u8(vabdq_u8(enc_vec, dc_top_vec)));
    }

    let (cb4, cb5, cb6, cb7) = EncSpan::read_quad::<8>(qb1, stb);
    let (cr4, cr5, cr6, cr7) = EncSpan::read_quad::<8>(qr1, str_);
    for (k, (cb_r, cr_r)) in [(cb4, cr4), (cb5, cr5), (cb6, cr6), (cb7, cr7)]
        .into_iter()
        .enumerate()
    {
        let y = k + 4;
        let enc_vec = vcombine_u8(ld8(&cb_r), ld8(&cr_r));
        let h_vec = vcombine_u8(vdup_n_u8(left_cb[y]), vdup_n_u8(left_cr[y]));
        acc_v = vpadalq_u16(acc_v, vpaddlq_u8(vabdq_u8(enc_vec, v_vec)));
        acc_h = vpadalq_u16(acc_h, vpaddlq_u8(vabdq_u8(enc_vec, h_vec)));
        acc_dc = vpadalq_u16(acc_dc, vpaddlq_u8(vabdq_u8(enc_vec, dc_bot_vec)));
    }

    let sad_v = vaddvq_u32(acc_v) as i32;
    let sad_h = vaddvq_u32(acc_h) as i32;
    let sad_dc = vaddvq_u32(acc_dc) as i32;

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
    write_chroma_pred_neon(
        cb_dst.first_chunk_mut::<64>().expect("cb 64"),
        best_mode,
        &top_cb,
        &left_cb,
        &dc_top_cb,
        &dc_bot_cb,
    );
    write_chroma_pred_neon(
        cr_dst.first_chunk_mut::<64>().expect("cr 64"),
        best_mode,
        &top_cr,
        &left_cr,
        &dc_top_cr,
        &dc_bot_cr,
    );

    (best_mode, best_cost)
}

#[target_feature(enable = "neon")]
fn intra_chroma_8x8_combined3_satd_neon(
    pred_cb_cr: &mut [u8; 128],
    rec_cb: &PlaneCursor<'_>,
    enc_cb: &PlaneCursor<'_>,
    rec_cr: &PlaneCursor<'_>,
    enc_cr: &PlaneCursor<'_>,
    lambda: i32,
    _t: NeonToken,
) -> (u8, i32) {
    let top_cb = rec_cb.row_n::<8>(-1, 0);
    let top_cr = rec_cr.row_n::<8>(-1, 0);
    let mut left_cb = [0u8; 8];
    let mut left_cr = [0u8; 8];
    for y in 0..8 {
        left_cb[y] = rec_cb.at(-1, y as isize);
        left_cr[y] = rec_cr.at(-1, y as isize);
    }
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

    let v_subs_cb = top_v_subs_neon(ld8(&top_cb));
    let v_subs_cr = top_v_subs_neon(ld8(&top_cr));

    let (cb0, cb1, cb2, cb3) = EncSpan::read_quad::<8>(qb0, stb);
    let (cb4, cb5, cb6, cb7) = EncSpan::read_quad::<8>(qb1, stb);
    let (sv_b0, sh_b0, sdc_b0) = satd_strip_8x4_neon(
        &cb0,
        &cb1,
        &cb2,
        &cb3,
        v_subs_cb,
        ld8_i16(&[
            hl0_cb[0], hl0_cb[1], hl0_cb[2], hl0_cb[3], hl0_cb[0], hl0_cb[1], hl0_cb[2], hl0_cb[3],
        ]),
        ld8_i16(&[(dc_cb[0] as i16) << 4, 0, 0, 0, (dc_cb[1] as i16) << 4, 0, 0, 0]),
    );
    let (sv_b1, sh_b1, sdc_b1) = satd_strip_8x4_neon(
        &cb4,
        &cb5,
        &cb6,
        &cb7,
        v_subs_cb,
        ld8_i16(&[
            hl1_cb[0], hl1_cb[1], hl1_cb[2], hl1_cb[3], hl1_cb[0], hl1_cb[1], hl1_cb[2], hl1_cb[3],
        ]),
        ld8_i16(&[(dc_cb[2] as i16) << 4, 0, 0, 0, (dc_cb[3] as i16) << 4, 0, 0, 0]),
    );

    let (cr0, cr1, cr2, cr3) = EncSpan::read_quad::<8>(qr0, str_);
    let (cr4, cr5, cr6, cr7) = EncSpan::read_quad::<8>(qr1, str_);
    let (sv_r0, sh_r0, sdc_r0) = satd_strip_8x4_neon(
        &cr0,
        &cr1,
        &cr2,
        &cr3,
        v_subs_cr,
        ld8_i16(&[
            hl0_cr[0], hl0_cr[1], hl0_cr[2], hl0_cr[3], hl0_cr[0], hl0_cr[1], hl0_cr[2], hl0_cr[3],
        ]),
        ld8_i16(&[(dc_cr[0] as i16) << 4, 0, 0, 0, (dc_cr[1] as i16) << 4, 0, 0, 0]),
    );
    let (sv_r1, sh_r1, sdc_r1) = satd_strip_8x4_neon(
        &cr4,
        &cr5,
        &cr6,
        &cr7,
        v_subs_cr,
        ld8_i16(&[
            hl1_cr[0], hl1_cr[1], hl1_cr[2], hl1_cr[3], hl1_cr[0], hl1_cr[1], hl1_cr[2], hl1_cr[3],
        ]),
        ld8_i16(&[(dc_cr[2] as i16) << 4, 0, 0, 0, (dc_cr[3] as i16) << 4, 0, 0, 0]),
    );

    let acc_v = vaddq_u16(vaddq_u16(sv_b0, sv_b1), vaddq_u16(sv_r0, sv_r1));
    let acc_h = vaddq_u16(vaddq_u16(sh_b0, sh_b1), vaddq_u16(sh_r0, sh_r1));
    let acc_dc = vaddq_u16(vaddq_u16(sdc_b0, sdc_b1), vaddq_u16(sdc_r0, sdc_r1));

    let satd_v = (vaddlvq_u16(acc_v) >> 1) as i32;
    let satd_h = (vaddlvq_u16(acc_h) >> 1) as i32;
    let satd_dc = (vaddlvq_u16(acc_dc) >> 1) as i32;

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
    write_chroma_pred_neon(
        cb_dst.first_chunk_mut::<64>().expect("cb 64"),
        best_mode,
        &top_cb,
        &left_cb,
        &dc_top_cb,
        &dc_bot_cb,
    );
    write_chroma_pred_neon(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::get_intra_predictor::*;
    use crate::safe::plane::PaddedPlane;

    fn test_plane(w: usize, h: usize, pad: usize, stride: usize) -> PaddedPlane {
        let mut p = PaddedPlane::new(w, h, pad, stride);
        for y in -(pad as isize)..(h + pad) as isize {
            for x in -(pad as isize)..(w + pad) as isize {
                p.set(x, y, (((y + 37) * stride as isize + (x + 43)) % 251) as u8);
            }
        }
        p
    }

    /// A 64-bit LCG, so a failing seed is replayable.
    fn lcg(seed: &mut u64) -> u8 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 32) as u8
    }

    fn noise_plane(seed: &mut u64) -> PaddedPlane {
        let (w, h, pad, stride) = (32usize, 32usize, 16usize, 64usize);
        let mut p = PaddedPlane::new(w, h, pad, stride);
        for y in -(pad as isize)..(h + pad) as isize {
            for x in -(pad as isize)..(w + pad) as isize {
                p.set(x, y, lcg(seed));
            }
        }
        p
    }

    /// A plane whose neighbours are all one value — the ends of every mean and of
    /// the plane predictors' `a`.
    fn flat_plane(v: u8) -> PaddedPlane {
        let (w, h, pad, stride) = (32usize, 32usize, 16usize, 64usize);
        let mut p = PaddedPlane::new(w, h, pad, stride);
        for y in -(pad as isize)..(h + pad) as isize {
            for x in -(pad as isize)..(w + pad) as isize {
                p.set(x, y, v);
            }
        }
        p
    }

    /// The steepest plane gradients: 0 on one side of the edge, 255 on the other.
    fn step_plane(rising: bool) -> PaddedPlane {
        let (w, h, pad, stride) = (32usize, 32usize, 16usize, 64usize);
        let mut p = PaddedPlane::new(w, h, pad, stride);
        for y in -(pad as isize)..(h + pad) as isize {
            for x in -(pad as isize)..(w + pad) as isize {
                let hi = (x + y >= 8) == rising;
                p.set(x, y, if hi { 255 } else { 0 });
            }
        }
        p
    }

    fn enc_planes() -> Vec<PaddedPlane> {
        let mut seed = 0x5DEECE66Du64;
        let mut v = vec![
            test_plane(32, 32, 16, 64),
            flat_plane(0),
            flat_plane(255),
            step_plane(true),
            step_plane(false),
        ];
        for _ in 0..6 {
            v.push(noise_plane(&mut seed));
        }
        v
    }

    #[test]
    fn test_i16x16_luma_pred_parity() {
        for p in enc_planes() {
            for anchor in [(0isize, 0isize), (16, 16), (8, 4)] {
                let rec = p.cursor(anchor.0, anchor.1);
                let pairs: [(
                    &str,
                    fn(&mut [u8; 256], &PlaneCursor),
                    fn(&mut [u8; 256], &PlaneCursor),
                ); 4] = [
                    ("V", WelsI16x16LumaPredV_c, enc_i16x16_luma_pred_v),
                    ("H", WelsI16x16LumaPredH_c, enc_i16x16_luma_pred_h),
                    ("DC", WelsI16x16LumaPredDc_c, enc_i16x16_luma_pred_dc),
                    (
                        "Plane",
                        WelsI16x16LumaPredPlane_c,
                        enc_i16x16_luma_pred_plane,
                    ),
                ];
                for (name, scalar, simd) in pairs {
                    let mut want = [0u8; 256];
                    let mut got = [0u8; 256];
                    scalar(&mut want, &rec);
                    simd(&mut got, &rec);
                    assert_eq!(got, want, "16x16 {name} mismatch at {anchor:?}");
                }
            }
        }
    }

    #[test]
    fn test_chroma_pred_parity() {
        for p in enc_planes() {
            for anchor in [(0isize, 0isize), (8, 8), (16, 4)] {
                let rec = p.cursor(anchor.0, anchor.1);
                let pairs: [(
                    &str,
                    fn(&mut [u8; 64], &PlaneCursor),
                    fn(&mut [u8; 64], &PlaneCursor),
                ); 4] = [
                    ("V", WelsIChromaPredV_c, enc_chroma_pred_v),
                    ("H", WelsIChromaPredH_c, enc_chroma_pred_h),
                    ("DC", WelsIChromaPredDc_c, enc_chroma_pred_dc),
                    ("Plane", WelsIChromaPredPlane_c, enc_chroma_pred_plane),
                ];
                for (name, scalar, simd) in pairs {
                    let mut want = [0u8; 64];
                    let mut got = [0u8; 64];
                    scalar(&mut want, &rec);
                    simd(&mut got, &rec);
                    assert_eq!(got, want, "Chroma {name} mismatch at {anchor:?}");
                }
            }
        }
    }

    #[test]
    fn test_i4x4_luma_pred_parity() {
        for p in enc_planes() {
            for anchor in [(0isize, 0isize), (4, 4), (12, 8), (8, 20)] {
                let rec = p.cursor(anchor.0, anchor.1);
                let pairs: [(
                    &str,
                    fn(&mut [u8; 16], &PlaneCursor),
                    fn(&mut [u8; 16], &PlaneCursor),
                ); 9] = [
                    ("V", WelsI4x4LumaPredV_c, enc_i4x4_luma_pred_v),
                    ("H", WelsI4x4LumaPredH_c, enc_i4x4_luma_pred_h),
                    ("DC", WelsI4x4LumaPredDc_c, enc_i4x4_luma_pred_dc),
                    ("DDL", WelsI4x4LumaPredDDL_c, enc_i4x4_luma_pred_ddl),
                    ("DDR", WelsI4x4LumaPredDDR_c, enc_i4x4_luma_pred_ddr),
                    ("VR", WelsI4x4LumaPredVR_c, enc_i4x4_luma_pred_vr),
                    ("HD", WelsI4x4LumaPredHD_c, enc_i4x4_luma_pred_hd),
                    ("VL", WelsI4x4LumaPredVL_c, enc_i4x4_luma_pred_vl),
                    ("HU", WelsI4x4LumaPredHU_c, enc_i4x4_luma_pred_hu),
                ];
                for (name, scalar, simd) in pairs {
                    let mut want = [0u8; 16];
                    let mut got = [0u8; 16];
                    scalar(&mut want, &rec);
                    simd(&mut got, &rec);
                    assert_eq!(got, want, "4x4 {name} mismatch at {anchor:?}");
                }
            }
        }
    }

    #[test]
    fn test_combined3_parity() {
        let mut seed = 0x1357_9BDF_2468_ACE0u64;
        let mut pairs = vec![
            (flat_plane(128), flat_plane(138)),
            (flat_plane(0), flat_plane(255)),
            (step_plane(true), step_plane(false)),
        ];
        for _ in 0..8 {
            pairs.push((noise_plane(&mut seed), noise_plane(&mut seed)));
        }

        for (case_idx, (p_rec, p_enc)) in pairs.into_iter().enumerate() {
            let rec = p_rec.cursor(8, 8);
            let enc = p_enc.cursor(8, 8);

            for (ldc, lh, lv) in [(0, 0, 0), (4, 16, 16), (16, 4, 16), (16, 16, 4)] {
                let mut pred_dc = [0u8; 16];
                let mut pred_h = [0u8; 16];
                let mut pred_v = [0u8; 16];
                WelsI4x4LumaPredDc_c(&mut pred_dc, &rec);
                WelsI4x4LumaPredH_c(&mut pred_h, &rec);
                WelsI4x4LumaPredV_c(&mut pred_v, &rec);
                let cdc = crate::encoder::sample::satd_4x4(
                    &PlaneCursor::new(&pred_dc, 0, 4),
                    &enc,
                ) + ldc;
                let ch = crate::encoder::sample::satd_4x4(
                    &PlaneCursor::new(&pred_h, 0, 4),
                    &enc,
                ) + lh;
                let cv = crate::encoder::sample::satd_4x4(
                    &PlaneCursor::new(&pred_v, 0, 4),
                    &enc,
                ) + lv;
                let (want_m, want_c, want_p) = if cdc <= ch && cdc <= cv {
                    (2i8, cdc, pred_dc)
                } else if ch <= cv {
                    (1i8, ch, pred_h)
                } else {
                    (0i8, cv, pred_v)
                };

                let mut got_p = [0u8; 16];
                let (got_m, got_c) = intra_4x4_combined3_satd(&mut got_p, &rec, &enc, ldc, lh, lv);
                assert_eq!((got_m, got_c), (want_m, want_c), "4x4 case={case_idx}");
                assert_eq!(got_p, want_p, "4x4 pred case={case_idx}");
            }

            for lambda in [0, 5, 25] {
                let mut pv = [0u8; 256];
                let mut ph = [0u8; 256];
                let mut pdc = [0u8; 256];
                WelsI16x16LumaPredV_c(&mut pv, &rec);
                WelsI16x16LumaPredH_c(&mut ph, &rec);
                WelsI16x16LumaPredDc_c(&mut pdc, &rec);

                let cv = crate::encoder::sample::satd_16x16(
                    &PlaneCursor::new(&pv, 0, 16),
                    &enc,
                ) + lambda;
                let ch = crate::encoder::sample::satd_16x16(
                    &PlaneCursor::new(&ph, 0, 16),
                    &enc,
                ) + lambda * 3;
                let cdc = crate::encoder::sample::satd_16x16(
                    &PlaneCursor::new(&pdc, 0, 16),
                    &enc,
                ) + lambda * 3;
                let (wm, wc, wp) = if cdc < ch && cdc < cv {
                    (2u8, cdc, pdc)
                } else if ch < cv {
                    (1u8, ch, ph)
                } else {
                    (0u8, cv, pv)
                };

                let mut gp = [0u8; 256];
                let (gm, gc) = intra_16x16_combined3_satd(&mut gp, &rec, &enc, lambda);
                assert_eq!((gm, gc), (wm, wc), "16x16 satd case={case_idx} l={lambda}");
                assert_eq!(gp, wp, "16x16 satd pred case={case_idx}");

                let rec_cr = p_rec.cursor(16, 8);
                let enc_cr = p_enc.cursor(16, 8);
                for use_satd in [false, true] {
                    let mut best_m = 2u8;
                    let mut best_c = i32::MAX;
                    let mut want_cb_cr = [0u8; 128];
                    for (m, bits) in [(2u8, 3), (1u8, 3), (0u8, 1)] {
                        let mut cb = [0u8; 64];
                        let mut cr = [0u8; 64];
                        match m {
                            2 => {
                                WelsIChromaPredV_c(&mut cb, &rec);
                                WelsIChromaPredV_c(&mut cr, &rec_cr);
                            }
                            1 => {
                                WelsIChromaPredH_c(&mut cb, &rec);
                                WelsIChromaPredH_c(&mut cr, &rec_cr);
                            }
                            _ => {
                                WelsIChromaPredDc_c(&mut cb, &rec);
                                WelsIChromaPredDc_c(&mut cr, &rec_cr);
                            }
                        }
                        let c1 = PlaneCursor::new(&cb, 0, 8);
                        let c2 = PlaneCursor::new(&cr, 0, 8);
                        let dist = if use_satd {
                            crate::encoder::sample::satd_8x8(&c1, &enc)
                                + crate::encoder::sample::satd_8x8(&c2, &enc_cr)
                        } else {
                            crate::common::sad_common::sample_sad::<8, 8, _>(&c1, &enc)
                                + crate::common::sad_common::sample_sad::<8, 8, _>(&c2, &enc_cr)
                        };
                        let cur = dist + lambda * bits;
                        if cur < best_c {
                            best_m = m;
                            best_c = cur;
                            want_cb_cr[..64].copy_from_slice(&cb);
                            want_cb_cr[64..].copy_from_slice(&cr);
                        }
                    }

                    let mut got_cb_cr = [0u8; 128];
                    let got = if use_satd {
                        intra_chroma_8x8_combined3_satd(
                            &mut got_cb_cr,
                            &rec,
                            &enc,
                            &rec_cr,
                            &enc_cr,
                            lambda,
                        )
                    } else {
                        intra_chroma_8x8_combined3_sad(
                            &mut got_cb_cr,
                            &rec,
                            &enc,
                            &rec_cr,
                            &enc_cr,
                            lambda,
                        )
                    };
                    assert_eq!(
                        got,
                        (best_m, best_c),
                        "chroma satd={use_satd} case={case_idx} l={lambda}"
                    );
                    assert_eq!(
                        got_cb_cr, want_cb_cr,
                        "chroma pred satd={use_satd} case={case_idx}"
                    );
                }
            }
        }
    }

    // ========================================================================
    // The decoder-side predictors: the in-place reconstructors, compared over the
    // whole allocation of two identically built planes. The reference is
    // `decoder::get_intra_predictor`, which has no SIMD dispatch of its own.
    // ========================================================================

    fn assert_dec_parity(
        name: &str,
        scalar: fn(&mut PlaneCursorMut<'_>),
        simd: fn(&mut PlaneCursorMut<'_>),
    ) {
        let mut seed = 0xB502_6F5Au64;
        let mut planes = vec![
            test_plane(32, 32, 16, 64),
            flat_plane(0),
            flat_plane(255),
            step_plane(true),
            step_plane(false),
        ];
        planes.push(noise_plane(&mut seed));
        planes.push(noise_plane(&mut seed));
        for pa in planes {
            for anchor in [(8isize, 8isize), (4, 12), (16, 16)] {
                let mut a = pa.clone();
                let mut b = pa.clone();
                scalar(&mut a.cursor_mut(anchor.0, anchor.1));
                simd(&mut b.cursor_mut(anchor.0, anchor.1));
                assert_eq!(
                    a.as_slice(),
                    b.as_slice(),
                    "{name}: NEON and scalar disagree somewhere in the allocation"
                );
            }
        }
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

    /// Every public kernel here reaches at least one NEON intrinsic — in its own body
    /// or in a function it calls — unless it is one of the fills listed below as scalar
    /// by design.
    #[test]
    fn every_kernel_here_reaches_an_intrinsic() {
        /// The broadcasts and fills: a V fill is a row copy, an H fill a byte splat,
        /// `DC_128` a constant.
        const SCALAR_BY_DESIGN: [&str; 13] = [
            "enc_i16x16_luma_pred_v",
            "dec_i16x16_luma_pred_v",
            "enc_i16x16_luma_pred_h",
            "dec_i16x16_luma_pred_h",
            "dec_i16x16_luma_pred_dc_na",
            "enc_chroma_pred_v",
            "dec_chroma_pred_v",
            "enc_chroma_pred_h",
            "dec_chroma_pred_h",
            "enc_i4x4_luma_pred_v",
            "dec_i4x4_luma_pred_v",
            "enc_i4x4_luma_pred_h",
            "dec_i4x4_luma_pred_h",
        ];

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

        let mut bodies: Vec<(String, &str)> = Vec::new();
        for (off, _) in src.match_indices("fn ") {
            let name = ident(&src[off + 3..]);
            if !name.is_empty() {
                bodies.push((name, body_at(src, off)));
            }
        }

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

        // A NEON intrinsic: `v…` with a lane-type suffix, which no other identifier
        // in this file has.
        fn is_intrinsic(tok: &str) -> bool {
            tok.starts_with('v')
                && ["_u8", "_s8", "_u16", "_s16", "_u32", "_s32", "_u64", "_s64"]
                    .iter()
                    .any(|s| tok.ends_with(s))
        }
        let intrinsics = |b: &str| {
            b.split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .any(is_intrinsic)
        };

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
            "these are installed as aarch64 kernels but reach no NEON intrinsic — implement them, or add them to \
             `SCALAR_BY_DESIGN` and say why: {offenders:?}"
        );

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
