// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! SAD and four-point SAD — `WelsSampleSad*_AArch64_neon` and
//! `WelsSampleSadFour*_AArch64_neon`, `codec/encoder/core/arm64/pixel_aarch64_neon.S`.
//!
//! The 16-wide kernels (`WelsSampleSad16x{8,16}_AArch64_neon` and
//! `WelsSampleSadFour16x{8,16}_AArch64_neon`) compute byte absolute differences
//! (`uabd`) and pairwise-accumulate even and odd rows into two `.8h`
//! accumulators per candidate (`uaddlp`/`uadalp`), four rows per step.
//! Single-candidate kernels reduce each accumulator separately with `uaddlv`
//! and add the two scalar totals so LLVM does not reassociate a vector
//! `vaddq_u16` back into a single serial `uadalp` chain. The 16-wide four-point
//! kernels pairwise-reduce the four even and four odd accumulators in parallel
//! (`addp.8h` -> `uaddlp.4s`/`uadalp.4s` -> one vector store), which keeps the
//! chains independent and avoids four cross-domain `uaddlv`+`fmov` transfers.
//! Each `.8h` lane gains at most `2 * 255` per row over at most 8 rows per
//! accumulator (`4080`, well within `u16::MAX`).
//!
//! The narrower kernels (`8x*`, `4x*`) use `uabdl`/`uabal` into `.8h`
//! accumulators with the same `8160` lane bound (one instruction per row at
//! 8-byte width, where `uabd`+`uadalp` would take two).
//! The four-point kernels cut `sample2` `W + 2` wide and `H + 2` tall — the
//! reach of all four probes — and read the up/down/left/right probes at span
//! columns 0, 1 and 2.
//!
//! There is no upstream 8x4 or 4x8 kernel; those slots run the same row loop at those
//! heights. Nor is there a second, wider tier on aarch64: the `_avx2` constructors
//! return `None`, leaving the baseline NEON slots installed.
//!
//! # Accumulators
//!
//! One accumulator per block makes a sixteen-row block a chain of thirty-two dependent
//! `uabal`s on an out-of-order core, half the speed of the scalar loop LLVM vectorises
//! for itself. The loops here keep two rows in flight — two pairwise `uabd`+`uadalp`
//! accumulators per candidate for the sixteen-wide shapes (halving widening-accumulate
//! ops versus `uabal`+`uabal2` and capping each `uadalp` dependency chain at `H / 2`
//! rows), and two `uabal` accumulators for the narrower shapes.
//!
//! # How the rows are addressed
//!
//! Each operand is cut once into a `RefSamples::span` — a slice the compiler knows to
//! be `(H - 1) * stride + W` long — and the rows are indexed inside it. That leaves one
//! cut per operand per call, two compare-and-branch pairs and both before the loop
//! rather than two per row. The rows are in range by construction and LLVM proves it
//! from the span's length.
//!
//! The 16-wide kernels walk 4-row quads inside a single `BlockRows::slice_prefix`
//! via `split_suffix` + `read_quad`/`read_quad3`, avoiding per-quad bounds checks
//! and stride multiplications. The narrower four-point kernels (`sad_four_8x`,
//! `sad_four_4x`) take that block a `G`-row window at a time so each window has
//! constant row offsets inside its span.
#![deny(unsafe_code)]

use core::arch::aarch64::*;

use super::lanes::{ld4, ld8, ld16, st4_i32};
use super::neon_token;
use crate::safe::plane::{BlockRows, RefSamples};
use crate::simd::{simd_entry, tokens::NeonToken};

// ============================================================================
// Row loops
// ============================================================================

/// Accumulate four 16-wide rows of single-candidate SAD into two `.8h`
/// accumulators (`$acc0` for even rows `0, 2` and `$acc1` for odd rows `1, 3`).
///
/// The `init` arm starts both chains with `vpaddlq_u8` (`uaddlp.8h`) on the
/// first quad (avoiding zero-initialised registers); subsequent quads use the
/// `acc` arm (`vpadalq_u8` / `uadalp.8h`). Reading four rows at a time via
/// `BlockRows::read_quad` inside a single `slice_prefix` eliminates per-row
/// bounds checks and `umull` stride multiplies.
macro_rules! sad_16x_neon_quad {
    (init $S:ident, $q1:expr, $st1:expr, $q2:expr, $st2:expr, $acc0:ident, $acc1:ident) => {{
        let (a0, a1, a2, a3) = $S::Span::read_quad::<16>($q1, $st1);
        let (b0, b1, b2, b3) = $S::Span::read_quad::<16>($q2, $st2);
        $acc0 = vpadalq_u8(
            vpaddlq_u8(vabdq_u8(ld16(&a0), ld16(&b0))),
            vabdq_u8(ld16(&a2), ld16(&b2)),
        );
        $acc1 = vpadalq_u8(
            vpaddlq_u8(vabdq_u8(ld16(&a1), ld16(&b1))),
            vabdq_u8(ld16(&a3), ld16(&b3)),
        );
    }};
    (acc $S:ident, $q1:expr, $st1:expr, $q2:expr, $st2:expr, $acc0:ident, $acc1:ident) => {{
        let (a0, a1, a2, a3) = $S::Span::read_quad::<16>($q1, $st1);
        let (b0, b1, b2, b3) = $S::Span::read_quad::<16>($q2, $st2);
        $acc0 = vpadalq_u8(
            vpadalq_u8($acc0, vabdq_u8(ld16(&a0), ld16(&b0))),
            vabdq_u8(ld16(&a2), ld16(&b2)),
        );
        $acc1 = vpadalq_u8(
            vpadalq_u8($acc1, vabdq_u8(ld16(&a1), ld16(&b1))),
            vabdq_u8(ld16(&a3), ld16(&b3)),
        );
    }};
}

/// Accumulate four 16-wide rows of four-point SAD (`[up, down, left, right]`)
/// into `acc[probe][row & 1]`.
///
/// Reads four rows (`a0..a3`) from `$q1` and the overlapping six-row × 18-byte
/// probe window (`r0..r5`) from `$q2`, pairing each `a_j` with
/// `[r_{j,m}, r_{j+2,m}, r_{j+1,l}, r_{j+1,r}]`. Processing the four rows in
/// order (rather than carrying probe registers across quads or grouping even
/// and odd rows separately) keeps live ranges short so LLVM folds the initial
/// zero-accumulations into `uaddlp.8h` and avoids GPR spills on 16×16 (and all
/// spills on 16×8), while `BlockRows` suffix splits eliminate per-quad bounds
/// checks and `umull` row-offset multiplies.
macro_rules! sad_four_16x_neon_quad {
    ($S:ident, $q1:expr, $st1:expr, $q2:expr, $st2:expr, $acc:ident) => {{
        let (a0, a1, a2, a3) = $S::Span::read_quad::<16>($q1, $st1);
        let q2_4 = $S::Span::split_suffix($q2, 4 * $st2);
        let q2_5 = $S::Span::split_suffix(q2_4, $st2);
        let ((_, r0_m, _), (r1_l, r1_m, r1_r), (r2_l, r2_m, r2_r), (r3_l, r3_m, r3_r)) =
            $S::Span::read_quad3::<16, 18>($q2, $st2);
        let (r4_l, r4_m, r4_r) = $S::Span::read_head3::<16, 18>(q2_4);
        let (_, r5_m, _) = $S::Span::read_head3::<16, 18>(q2_5);
        let rows = [
            (a0, [r0_m, r2_m, r1_l, r1_r]),
            (a1, [r1_m, r3_m, r2_l, r2_r]),
            (a2, [r2_m, r4_m, r3_l, r3_r]),
            (a3, [r3_m, r5_m, r4_l, r4_r]),
        ];
        for (j, (a, probes)) in rows.into_iter().enumerate() {
            let va = ld16(&a);
            let parity = j & 1;
            for (k, p) in probes.into_iter().enumerate() {
                $acc[k][parity] = vpadalq_u8($acc[k][parity], vabdq_u8(va, ld16(&p)));
            }
        }
    }};
}

simd_entry! {
    /// Two independent pairwise `uabd` + `uaddlp`/`uadalp` accumulation chains;
    /// see the header.
    #[target_feature(enable = "neon")]
    fn sad_16x<S: RefSamples, const H: usize>(
        sample1: &S,
        sample2: &S,
        dx: isize,
        dy: isize,
        _token: NeonToken,
    ) -> i32 {
        const { assert!(H == 8 || H == 16, "sad_16x supports H = 8 or 16") };
        let (s1, s2) = (sample1.span::<16, H>(0, 0), sample2.span::<16, H>(dy, dx));
        let (raw1, st1) = s1.raw_parts();
        let (raw2, st2) = s2.raw_parts();
        let (st1, st2) = (st1 as usize, st2 as usize);
        let q1_0 = S::Span::slice_prefix(raw1, (H - 1) * st1 + 16);
        let q2_0 = S::Span::slice_prefix(raw2, (H - 1) * st2 + 16);
        let mut acc0;
        let mut acc1;
        sad_16x_neon_quad!(init S, q1_0, st1, q2_0, st2, acc0, acc1);
        sad_16x_neon_quad!(
            acc S,
            S::Span::split_suffix(q1_0, 4 * st1),
            st1,
            S::Span::split_suffix(q2_0, 4 * st2),
            st2,
            acc0,
            acc1
        );
        if H == 16 {
            let q1_2 = S::Span::split_suffix(q1_0, 8 * st1);
            let q2_2 = S::Span::split_suffix(q2_0, 8 * st2);
            sad_16x_neon_quad!(acc S, q1_2, st1, q2_2, st2, acc0, acc1);
            sad_16x_neon_quad!(
                acc S,
                S::Span::split_suffix(q1_2, 4 * st1),
                st1,
                S::Span::split_suffix(q2_2, 4 * st2),
                st2,
                acc0,
                acc1
            );
        }
        // Reduce each accumulator separately so LLVM keeps the two `uadalp`
        // chains independent instead of folding `vaddq_u16(acc0, acc1)` into a
        // single H-deep serial chain.
        (vaddlvq_u16(acc0) + vaddlvq_u16(acc1)) as i32
    }

    /// Two rows per step into two accumulators.
    #[target_feature(enable = "neon")]
    fn sad_8x<S: RefSamples, const H: usize>(
        sample1: &S,
        sample2: &S,
        dx: isize,
        dy: isize,
        _token: NeonToken,
    ) -> i32 {
        const { assert!(H % 2 == 0, "sad_8x steps two rows; H must be even") };
        let (s1, s2) = (sample1.span::<8, H>(0, 0), sample2.span::<8, H>(dy, dx));
        let mut acc = [vdupq_n_u16(0); 2];
        let mut y = 0usize;
        while y < H {
            acc[0] = vabal_u8(acc[0], ld8(&s1.row::<8>(y, 0)), ld8(&s2.row::<8>(y, 0)));
            acc[1] = vabal_u8(
                acc[1],
                ld8(&s1.row::<8>(y + 1, 0)),
                ld8(&s2.row::<8>(y + 1, 0)),
            );
            y += 2;
        }
        vaddlvq_u16(vaddq_u16(acc[0], acc[1])) as i32
    }

    /// The asm loads each row into `.s[0]` and reduces over `.4h`; `ld4` zeroes the
    /// upper lanes instead, so the full-width reduce is exact.
    #[target_feature(enable = "neon")]
    fn sad_4x<S: RefSamples, const H: usize>(
        sample1: &S,
        sample2: &S,
        dx: isize,
        dy: isize,
        _token: NeonToken,
    ) -> i32 {
        const { assert!(H % 2 == 0, "sad_4x steps two rows; H must be even") };
        let (s1, s2) = (sample1.span::<4, H>(0, 0), sample2.span::<4, H>(dy, dx));
        let mut acc = [vdupq_n_u16(0); 2];
        let mut y = 0usize;
        while y < H {
            acc[0] = vabal_u8(acc[0], ld4(&s1.row::<4>(y, 0)), ld4(&s2.row::<4>(y, 0)));
            acc[1] = vabal_u8(
                acc[1],
                ld4(&s1.row::<4>(y + 1, 0)),
                ld4(&s2.row::<4>(y + 1, 0)),
            );
            y += 2;
        }
        vaddlvq_u16(vaddq_u16(acc[0], acc[1])) as i32
    }

    /// The four whole-sample neighbours — up, down, left, right — two pairwise
    /// `uabd` + `uaddlp`/`uadalp` accumulators each (even and odd rows), four rows
    /// per step, capping every `uadalp` dependency chain at `H / 2` rows.
    #[target_feature(enable = "neon")]
    fn sad_four_16x<S: RefSamples, const H: usize, const HW: usize>(
        sample1: &S,
        sample2: &S,
        sad: &mut [i32; 4],
        _token: NeonToken,
    ) {
        const { assert!(H == 8 || H == 16, "sad_four_16x supports H = 8 or 16") };
        const {
            assert!(
                HW == H + 2,
                "the probe span is two rows taller than the block"
            )
        };
        let s1 = sample1.span::<16, H>(0, 0);
        let s2 = sample2.span::<18, HW>(-1, -1);
        let (raw1, st1) = s1.raw_parts();
        let (raw2, st2) = s2.raw_parts();
        let (st1, st2) = (st1 as usize, st2 as usize);
        let q1_0 = S::Span::slice_prefix(raw1, (H - 1) * st1 + 16);
        let q2_0 = S::Span::slice_prefix(raw2, (H + 1) * st2 + 18);
        let mut acc = [[vdupq_n_u16(0); 2]; 4];
        sad_four_16x_neon_quad!(S, q1_0, st1, q2_0, st2, acc);
        sad_four_16x_neon_quad!(
            S,
            S::Span::split_suffix(q1_0, 4 * st1),
            st1,
            S::Span::split_suffix(q2_0, 4 * st2),
            st2,
            acc
        );
        if H == 16 {
            let q1_2 = S::Span::split_suffix(q1_0, 8 * st1);
            let q2_2 = S::Span::split_suffix(q2_0, 8 * st2);
            sad_four_16x_neon_quad!(S, q1_2, st1, q2_2, st2, acc);
            sad_four_16x_neon_quad!(
                S,
                S::Span::split_suffix(q1_2, 4 * st1),
                st1,
                S::Span::split_suffix(q2_2, 4 * st2),
                st2,
                acc
            );
        }
        // Pairwise-reduce even and odd accumulators in parallel to `.4s` — keeps
        // the two `uadalp` chains per probe independent and replaces four
        // `uaddlv`+`fmov` cross-domain transfers with one vector store. Each lane
        // peaks at `8 * 2 * 255 = 4080`, so two `vpaddq_u16` steps stay <= 16320.
        let even = vpaddq_u16(
            vpaddq_u16(acc[0][0], acc[1][0]),
            vpaddq_u16(acc[2][0], acc[3][0]),
        );
        let odd = vpaddq_u16(
            vpaddq_u16(acc[0][1], acc[1][1]),
            vpaddq_u16(acc[2][1], acc[3][1]),
        );
        st4_i32(
            sad,
            vreinterpretq_s32_u32(vpadalq_u16(vpaddlq_u16(even), odd)),
        );
    }

    /// Four-point 8-wide SAD walking `G`-row windows (`HW == H + 2` for the
    /// up/down probe span).
    #[target_feature(enable = "neon")]
    fn sad_four_8x<S: RefSamples, const H: usize, const HW: usize, const G: usize>(
        sample1: &S,
        sample2: &S,
        sad: &mut [i32; 4],
        _token: NeonToken,
    ) {
        const {
            assert!(
                G % 2 == 0 && H % G == 0,
                "the block is a whole number of G-row cuts"
            )
        };
        const {
            assert!(
                HW == H + 2,
                "the probe span is two rows taller than the block"
            )
        };
        let mut acc = [[vdupq_n_u16(0); 2]; 4];
        let s1 = sample1.span::<8, H>(0, 0);
        let s2 = sample2.span::<10, HW>(-1, -1);
        let mut y = 0usize;
        while y < H {
            let s1 = s1.window::<8>(y, G);
            let s2 = s2.window::<10>(y, G + 2);
            let mut i = 0usize;
            while i < G {
                for j in 0..2usize {
                    let yy = i + j;
                    let a = ld8(&s1.row::<8>(yy, 0));
                    let probes = [
                        ld8(&s2.row::<8>(yy, 1)),
                        ld8(&s2.row::<8>(yy + 2, 1)),
                        ld8(&s2.row::<8>(yy + 1, 0)),
                        ld8(&s2.row::<8>(yy + 1, 2)),
                    ];
                    for (k, p) in probes.into_iter().enumerate() {
                        acc[k][j] = vabal_u8(acc[k][j], a, p);
                    }
                }
                i += 2;
            }
            y += G;
        }
        for k in 0..4 {
            sad[k] = vaddlvq_u16(vaddq_u16(acc[k][0], acc[k][1])) as i32;
        }
    }

    /// See [`sad_four_8x`] for the probe span and for what `G` and `HW` select.
    #[target_feature(enable = "neon")]
    fn sad_four_4x<S: RefSamples, const H: usize, const HW: usize, const G: usize>(
        sample1: &S,
        sample2: &S,
        sad: &mut [i32; 4],
        _token: NeonToken,
    ) {
        const {
            assert!(
                G % 2 == 0 && H % G == 0,
                "the block is a whole number of G-row cuts"
            )
        };
        const {
            assert!(
                HW == H + 2,
                "the probe span is two rows taller than the block"
            )
        };
        let mut acc = [[vdupq_n_u16(0); 2]; 4];
        let s1 = sample1.span::<4, H>(0, 0);
        let s2 = sample2.span::<6, HW>(-1, -1);
        let mut y = 0usize;
        while y < H {
            let s1 = s1.window::<4>(y, G);
            let s2 = s2.window::<6>(y, G + 2);
            let mut i = 0usize;
            while i < G {
                for j in 0..2usize {
                    let yy = i + j;
                    let a = ld4(&s1.row::<4>(yy, 0));
                    let probes = [
                        ld4(&s2.row::<4>(yy, 1)),
                        ld4(&s2.row::<4>(yy + 2, 1)),
                        ld4(&s2.row::<4>(yy + 1, 0)),
                        ld4(&s2.row::<4>(yy + 1, 2)),
                    ];
                    for (k, p) in probes.into_iter().enumerate() {
                        acc[k][j] = vabal_u8(acc[k][j], a, p);
                    }
                }
                i += 2;
            }
            y += G;
        }
        for k in 0..4 {
            sad[k] = vaddlvq_u16(vaddq_u16(acc[k][0], acc[k][1])) as i32;
        }
    }
}

// ============================================================================
// The entry points, named as the slots they fill
// ============================================================================

/// `WelsSampleSad16x16_AArch64_neon`.
#[inline]
pub fn sample_sad_16x16<S: RefSamples>(sample1: &S, sample2: &S) -> i32 {
    sad_16x::<S, 16>(sample1, sample2, 0, 0, neon_token())
}

/// The AVX2 slot's kernel — never installed on aarch64; see the module header.
// Every kernel set carries the pair to satisfy `encoder/sample.rs`'s AVX2 slot; a set
// that is compiled but not aliased goes dead.
#[allow(dead_code)]
#[inline]
pub(crate) fn sample_sad_16x16_avx2() -> Option<
    fn(&crate::safe::plane::PlaneCursor<'_>, &crate::safe::plane::PlaneCursor<'_>) -> i32,
> {
    None
}

/// `WelsSampleSad16x8_AArch64_neon`.
#[inline]
pub fn sample_sad_16x8<S: RefSamples>(sample1: &S, sample2: &S) -> i32 {
    sad_16x::<S, 8>(sample1, sample2, 0, 0, neon_token())
}

/// See [`sample_sad_16x16_avx2`].
// Dead in a set that is compiled but not aliased, as above.
#[allow(dead_code)]
#[inline]
pub(crate) fn sample_sad_16x8_avx2() -> Option<
    fn(&crate::safe::plane::PlaneCursor<'_>, &crate::safe::plane::PlaneCursor<'_>) -> i32,
> {
    None
}

/// `WelsSampleSad8x16_AArch64_neon`.
#[inline]
pub fn sample_sad_8x16<S: RefSamples>(sample1: &S, sample2: &S) -> i32 {
    sad_8x::<S, 16>(sample1, sample2, 0, 0, neon_token())
}

/// `WelsSampleSad8x8_AArch64_neon`.
#[inline]
pub fn sample_sad_8x8<S: RefSamples>(sample1: &S, sample2: &S) -> i32 {
    sad_8x::<S, 8>(sample1, sample2, 0, 0, neon_token())
}

/// `WelsSampleSad4x4_AArch64_neon`.
#[inline]
pub fn sample_sad_4x4<S: RefSamples>(sample1: &S, sample2: &S) -> i32 {
    sad_4x::<S, 4>(sample1, sample2, 0, 0, neon_token())
}

/// No upstream kernel; the 8-wide loop at height 4.
#[inline]
pub fn sample_sad_8x4<S: RefSamples>(sample1: &S, sample2: &S) -> i32 {
    sad_8x::<S, 4>(sample1, sample2, 0, 0, neon_token())
}

/// No upstream kernel; the 4-wide loop at height 8.
#[inline]
pub fn sample_sad_4x8<S: RefSamples>(sample1: &S, sample2: &S) -> i32 {
    sad_4x::<S, 8>(sample1, sample2, 0, 0, neon_token())
}

/// `WelsSampleSadFour16x16_AArch64_neon`.
#[inline]
pub fn sample_sad_four_16x16<S: RefSamples>(sample1: &S, sample2: &S, sad: &mut [i32; 4]) {
    sad_four_16x::<S, 16, 18>(sample1, sample2, sad, neon_token())
}

/// `WelsSampleSadFour16x8_AArch64_neon`.
#[inline]
pub fn sample_sad_four_16x8<S: RefSamples>(sample1: &S, sample2: &S, sad: &mut [i32; 4]) {
    sad_four_16x::<S, 8, 10>(sample1, sample2, sad, neon_token())
}

/// `WelsSampleSadFour8x16_AArch64_neon`.
#[inline]
pub fn sample_sad_four_8x16<S: RefSamples>(sample1: &S, sample2: &S, sad: &mut [i32; 4]) {
    sad_four_8x::<S, 16, 18, 8>(sample1, sample2, sad, neon_token())
}

/// `WelsSampleSadFour8x8_AArch64_neon`.
#[inline]
pub fn sample_sad_four_8x8<S: RefSamples>(sample1: &S, sample2: &S, sad: &mut [i32; 4]) {
    sad_four_8x::<S, 8, 10, 8>(sample1, sample2, sad, neon_token())
}

/// `WelsSampleSadFour4x4_AArch64_neon`.
#[inline]
pub fn sample_sad_four_4x4<S: RefSamples>(sample1: &S, sample2: &S, sad: &mut [i32; 4]) {
    sad_four_4x::<S, 4, 6, 4>(sample1, sample2, sad, neon_token())
}

/// No upstream kernel; the 8-wide loop at height 4.
#[inline]
pub fn sample_sad_four_8x4<S: RefSamples>(sample1: &S, sample2: &S, sad: &mut [i32; 4]) {
    sad_four_8x::<S, 4, 6, 4>(sample1, sample2, sad, neon_token())
}

/// No upstream kernel; the 4-wide loop at height 8.
#[inline]
pub fn sample_sad_four_4x8<S: RefSamples>(sample1: &S, sample2: &S, sad: &mut [i32; 4]) {
    sad_four_4x::<S, 8, 10, 8>(sample1, sample2, sad, neon_token())
}

// ============================================================================
// Unit Tests: Differential Parity Against Scalar Kernels
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::sad_common::{sample_sad, sample_sad_four};
    use crate::safe::plane::PlaneCursor;

    fn make_test_planes(stride: usize, height: usize) -> (Vec<u8>, Vec<u8>) {
        let mut p1 = vec![0u8; stride * height];
        let mut p2 = vec![0u8; stride * height];
        for (i, b) in p1.iter_mut().enumerate() {
            *b = ((i * 17 + 5) & 0xFF) as u8;
        }
        for (i, b) in p2.iter_mut().enumerate() {
            *b = ((i * 31 + 13) & 0xFF) as u8;
        }
        (p1, p2)
    }

    #[test]
    fn test_sad_parity_all_shapes() {
        let (p1, p2) = make_test_planes(64, 64);
        let c1 = PlaneCursor::new(&p1, 64 * 8 + 8, 64);
        let c2 = PlaneCursor::new(&p2, 64 * 8 + 8, 64);

        assert_eq!(
            sample_sad_16x16(&c1, &c2),
            sample_sad::<16, 16, _>(&c1, &c2)
        );
        assert_eq!(sample_sad_16x8(&c1, &c2), sample_sad::<16, 8, _>(&c1, &c2));
        assert_eq!(sample_sad_8x16(&c1, &c2), sample_sad::<8, 16, _>(&c1, &c2));
        assert_eq!(sample_sad_8x8(&c1, &c2), sample_sad::<8, 8, _>(&c1, &c2));
        assert_eq!(sample_sad_4x4(&c1, &c2), sample_sad::<4, 4, _>(&c1, &c2));
        assert_eq!(sample_sad_8x4(&c1, &c2), sample_sad::<8, 4, _>(&c1, &c2));
        assert_eq!(sample_sad_4x8(&c1, &c2), sample_sad::<4, 8, _>(&c1, &c2));
    }

    /// On aarch64 there is no wider SAD tier, so the `_avx2` constructors return `None`.
    #[test]
    fn test_avx2_sad_parity() {
        assert!(sample_sad_16x16_avx2().is_none());
        assert!(sample_sad_16x8_avx2().is_none());
    }

    #[test]
    fn test_sample_sad_four_parity() {
        let (p1, p2) = make_test_planes(64, 64);
        let c1 = PlaneCursor::new(&p1, 64 * 16 + 16, 64);
        let c2 = PlaneCursor::new(&p2, 64 * 16 + 16, 64);

        let mut expected = [0i32; 4];
        let mut actual = [0i32; 4];

        sample_sad_four::<16, 16, _>(&c1, &c2, &mut expected);
        sample_sad_four_16x16(&c1, &c2, &mut actual);
        assert_eq!(actual, expected, "16x16 four-point SAD mismatch");

        sample_sad_four::<16, 8, _>(&c1, &c2, &mut expected);
        sample_sad_four_16x8(&c1, &c2, &mut actual);
        assert_eq!(actual, expected, "16x8 four-point SAD mismatch");

        sample_sad_four::<8, 16, _>(&c1, &c2, &mut expected);
        sample_sad_four_8x16(&c1, &c2, &mut actual);
        assert_eq!(actual, expected, "8x16 four-point SAD mismatch");

        sample_sad_four::<8, 8, _>(&c1, &c2, &mut expected);
        sample_sad_four_8x8(&c1, &c2, &mut actual);
        assert_eq!(actual, expected, "8x8 four-point SAD mismatch");

        sample_sad_four::<4, 4, _>(&c1, &c2, &mut expected);
        sample_sad_four_4x4(&c1, &c2, &mut actual);
        assert_eq!(actual, expected, "4x4 four-point SAD mismatch");

        sample_sad_four::<8, 4, _>(&c1, &c2, &mut expected);
        sample_sad_four_8x4(&c1, &c2, &mut actual);
        assert_eq!(actual, expected, "8x4 four-point SAD mismatch");

        sample_sad_four::<4, 8, _>(&c1, &c2, &mut expected);
        sample_sad_four_4x8(&c1, &c2, &mut actual);
        assert_eq!(actual, expected, "4x8 four-point SAD mismatch");
    }

    // ========================================================================
    // Input and anchor coverage.
    //
    // The sweep below runs every kernel over four anchors, one per residue class mod 8,
    // and five distributions. The all-`0xFF`/all-`0x00` pair and the near-identical
    // pair are the ends of the accumulator's range, where a lane that widened or
    // saturated wrongly would show.
    // ========================================================================

    /// A 64-bit LCG, so a failing seed is replayable.
    fn lcg(seed: &mut u64) -> u8 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 32) as u8
    }

    /// The five input distributions, each a pair of planes of `stride * height` bytes.
    fn input_pairs(stride: usize, height: usize) -> Vec<(&'static str, Vec<u8>, Vec<u8>)> {
        let n = stride * height;
        let mut seed = 0x5DEECE66Du64;
        let noise1: Vec<u8> = (0..n).map(|_| lcg(&mut seed)).collect();
        let noise2: Vec<u8> = (0..n).map(|_| lcg(&mut seed)).collect();

        let mut near = noise1.clone();
        // A handful of differing bytes: the accumulator spends most of its range at 0.
        for (i, b) in near.iter_mut().enumerate() {
            if i % 97 == 0 {
                *b = b.wrapping_add(1);
            }
        }

        let (ramp1, ramp2) = make_test_planes(stride, height);
        vec![
            ("ramps", ramp1, ramp2),
            ("noise", noise1.clone(), noise2),
            ("max-diff", vec![0xFFu8; n], vec![0x00u8; n]),
            ("near-identical", noise1, near),
            ("identical", vec![0x5Au8; n], vec![0x5Au8; n]),
        ]
    }

    /// Four anchors covering every residue mod 4 (with residues [0, 3, 6, 5] mod 8), so the
    /// aligned case is not the only one tested. Each leaves at least 16 rows and 16 columns of margin on all sides.
    const ANCHORS: [usize; 4] = [64 * 16 + 16, 64 * 17 + 19, 64 * 18 + 22, 64 * 19 + 21];

    #[test]
    fn sad_parity_over_anchors_and_distributions() {
        for (name, p1, p2) in input_pairs(64, 64) {
            for anchor in ANCHORS {
                let c1 = PlaneCursor::new(&p1, anchor, 64);
                let c2 = PlaneCursor::new(&p2, anchor, 64);
                let at = format!("{name} @ {anchor}");

                assert_eq!(
                    sample_sad_16x16(&c1, &c2),
                    sample_sad::<16, 16, _>(&c1, &c2),
                    "16x16 {at}"
                );
                assert_eq!(
                    sample_sad_16x8(&c1, &c2),
                    sample_sad::<16, 8, _>(&c1, &c2),
                    "16x8 {at}"
                );
                assert_eq!(
                    sample_sad_8x16(&c1, &c2),
                    sample_sad::<8, 16, _>(&c1, &c2),
                    "8x16 {at}"
                );
                assert_eq!(
                    sample_sad_8x8(&c1, &c2),
                    sample_sad::<8, 8, _>(&c1, &c2),
                    "8x8 {at}"
                );
                assert_eq!(
                    sample_sad_4x4(&c1, &c2),
                    sample_sad::<4, 4, _>(&c1, &c2),
                    "4x4 {at}"
                );
                assert_eq!(
                    sample_sad_8x4(&c1, &c2),
                    sample_sad::<8, 4, _>(&c1, &c2),
                    "8x4 {at}"
                );
                assert_eq!(
                    sample_sad_4x8(&c1, &c2),
                    sample_sad::<4, 8, _>(&c1, &c2),
                    "4x8 {at}"
                );
            }
        }
    }

    #[test]
    fn sample_sad_four_parity_over_anchors_and_distributions() {
        for (name, p1, p2) in input_pairs(64, 64) {
            for anchor in ANCHORS {
                let c1 = PlaneCursor::new(&p1, anchor, 64);
                let c2 = PlaneCursor::new(&p2, anchor, 64);
                let at = format!("{name} @ {anchor}");
                let (mut want, mut got) = ([0i32; 4], [0i32; 4]);

                sample_sad_four::<16, 16, _>(&c1, &c2, &mut want);
                sample_sad_four_16x16(&c1, &c2, &mut got);
                assert_eq!(got, want, "16x16 four-point {at}");

                sample_sad_four::<16, 8, _>(&c1, &c2, &mut want);
                sample_sad_four_16x8(&c1, &c2, &mut got);
                assert_eq!(got, want, "16x8 four-point {at}");

                sample_sad_four::<8, 16, _>(&c1, &c2, &mut want);
                sample_sad_four_8x16(&c1, &c2, &mut got);
                assert_eq!(got, want, "8x16 four-point {at}");

                sample_sad_four::<8, 8, _>(&c1, &c2, &mut want);
                sample_sad_four_8x8(&c1, &c2, &mut got);
                assert_eq!(got, want, "8x8 four-point {at}");

                sample_sad_four::<4, 4, _>(&c1, &c2, &mut want);
                sample_sad_four_4x4(&c1, &c2, &mut got);
                assert_eq!(got, want, "4x4 four-point {at}");

                sample_sad_four::<8, 4, _>(&c1, &c2, &mut want);
                sample_sad_four_8x4(&c1, &c2, &mut got);
                assert_eq!(got, want, "8x4 four-point {at}");

                sample_sad_four::<4, 8, _>(&c1, &c2, &mut want);
                sample_sad_four_4x8(&c1, &c2, &mut got);
                assert_eq!(got, want, "4x8 four-point {at}");
            }
        }
    }

    /// Port of `SadSatdAssemblyFuncTest.Sad16_AArch64_neon_StridesAndExtremes`
    /// (`test/encoder/EncUT_Sample.cpp`): sweeps aligned and unaligned strides
    /// (including non-multiple-of-16 stride `23` and asymmetric pairs), all 16
    /// byte-alignment offsets, and extreme pixel patterns (`0x00`/`0xFF`, identical,
    /// random). `PlaneCursor` strides are unsigned, so negative strides are not
    /// representable here.
    #[test]
    fn sad_16x_strides_offsets_and_extremes() {
        const STRIDES: [usize; 3] = [16, 23, 32];
        const BUF_LEN: usize = 32 * 32;
        let mut seed = 0x1234_5678_9ABC_DEF0u64;
        let mut a = vec![0u8; BUF_LEN];
        let mut b = vec![0u8; BUF_LEN];

        for pattern in 0..4usize {
            match pattern {
                0 => {
                    a.fill(0x00);
                    b.fill(0xFF);
                }
                1 => {
                    a.fill(0xFF);
                    b.fill(0x00);
                }
                2 => {
                    // Uniform constant so `c1` and `c2` see identical blocks (SAD == 0)
                    // at every stride pair and alignment offset.
                    a.fill(0x5A);
                    b.fill(0x5A);
                }
                _ => {
                    for i in 0..BUF_LEN {
                        a[i] = lcg(&mut seed);
                        b[i] = lcg(&mut seed);
                    }
                }
            }
            for &stride_a in &STRIDES {
                for &stride_b in &STRIDES {
                    for offset in 0..16usize {
                        let c1 = PlaneCursor::new(&a, offset, stride_a);
                        let c2 = PlaneCursor::new(&b, 15 - offset, stride_b);
                        let got_8 = sample_sad_16x8(&c1, &c2);
                        let got_16 = sample_sad_16x16(&c1, &c2);
                        assert_eq!(
                            got_8,
                            sample_sad::<16, 8, _>(&c1, &c2),
                            "16x8 pattern={pattern} sa={stride_a} sb={stride_b} off={offset}"
                        );
                        assert_eq!(
                            got_16,
                            sample_sad::<16, 16, _>(&c1, &c2),
                            "16x16 pattern={pattern} sa={stride_a} sb={stride_b} off={offset}"
                        );
                        if pattern == 2 {
                            assert_eq!((got_8, got_16), (0, 0));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sad_four_16x_strides_offsets_and_extremes() {
        const STRIDES: [usize; 3] = [18, 23, 32];
        const BUF_LEN: usize = 32 * 32;
        let mut seed = 0xCAFE_BABE_1234_5678u64;
        let mut a = vec![0u8; BUF_LEN];
        let mut b = vec![0u8; BUF_LEN];

        for pattern in 0..4usize {
            match pattern {
                0 => {
                    a.fill(0x00);
                    b.fill(0xFF);
                }
                1 => {
                    a.fill(0xFF);
                    b.fill(0x00);
                }
                2 => {
                    // Uniform constant so all four probes see identical blocks (`[0; 4]`)
                    // at every stride pair and alignment offset.
                    a.fill(0x5A);
                    b.fill(0x5A);
                }
                _ => {
                    for i in 0..BUF_LEN {
                        a[i] = lcg(&mut seed);
                        b[i] = lcg(&mut seed);
                    }
                }
            }
            for &stride_a in &STRIDES {
                for &stride_b in &STRIDES {
                    for offset in 0..16usize {
                        let c1 = PlaneCursor::new(&a, offset, stride_a);
                        // Leave one full row + 1 byte before `c2`'s origin for the (-1, -1) probe span.
                        let c2 = PlaneCursor::new(&b, stride_b + 1 + (15 - offset), stride_b);
                        let (mut want, mut got) = ([0i32; 4], [0i32; 4]);
                        sample_sad_four::<16, 8, _>(&c1, &c2, &mut want);
                        sample_sad_four_16x8(&c1, &c2, &mut got);
                        assert_eq!(
                            got, want,
                            "four 16x8 pattern={pattern} sa={stride_a} sb={stride_b} off={offset}"
                        );
                        if pattern == 2 {
                            assert_eq!(got, [0; 4]);
                        }
                        sample_sad_four::<16, 16, _>(&c1, &c2, &mut want);
                        sample_sad_four_16x16(&c1, &c2, &mut got);
                        assert_eq!(
                            got, want,
                            "four 16x16 pattern={pattern} sa={stride_a} sb={stride_b} off={offset}"
                        );
                        if pattern == 2 {
                            assert_eq!(got, [0; 4]);
                        }
                    }
                }
            }
        }
    }

    /// The AVX2 slot never changes on this target: `has_avx2()` is false on every
    /// aarch64 build, so asking for AVX2 must leave the baseline kernel installed.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn init_sample_sad_installs_avx2_only_where_the_cpu_has_it() {
        use crate::common::cpu_core::{WELS_CPU_AVX2, WELS_CPU_SSE2};
        use crate::encoder::sample::WelsInitSampleSadFunc;
        use crate::encoder::svc_mode_decision::BLOCK_16x16;
        use crate::encoder::wels_func_ptr_def::SWelsFuncPtrList;

        let slot = |flags: u32| {
            let mut fl = SWelsFuncPtrList::default();
            WelsInitSampleSadFunc(&mut fl, flags);
            fl.sSampleDealingFuncs.pfSampleSad[BLOCK_16x16].map(|f| f as usize)
        };

        let baseline = slot(WELS_CPU_SSE2);
        let asked_for_avx2 = slot(WELS_CPU_SSE2 | WELS_CPU_AVX2);
        assert!(baseline.is_some() && asked_for_avx2.is_some());
        assert!(
            !crate::simd::has_avx2(),
            "no aarch64 build can answer true here"
        );
        assert_eq!(
            asked_for_avx2, baseline,
            "the flag alone must not install an AVX2 kernel"
        );
    }
}
