// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! x86_64 SSE2 & AVX2 implementations of SAD and 4-point SAD kernels.
//!
//! Each operand is cut into a `RefSamples::span` — a slice known to be
//! `(H - 1) * stride + W` long — and the rows are indexed inside it, which costs one
//! bounds check per operand instead of two per row.
//!
//! The four-point kernels cut a `W + 2` by `G + 2` window of `sample2` per group of
//! `G` rows — the reach of all four probes of those rows — and read the up and down
//! probes at span rows `j` and `j + 2` of column 1, the left and right at row `j + 1`
//! of columns 0 and 2. Per group rather than once per block because only a constant
//! row offset inside a span folds; `G` is sized so the group's walk unrolls.
#![deny(unsafe_code)]

use super::lanes::{ld4, ld8, ld16};
use super::sse2_token;
use crate::safe::plane::{BlockRows, PlaneCursor, RefSamples};
use crate::simd::{
    Avx2PlaneSadKernel, simd_entry,
    tokens::{Avx2Token, Sse2Token},
};
use core::arch::x86_64::*;

// ============================================================================
// Internal SSE2 & AVX2 Kernels
// ============================================================================

#[target_feature(enable = "sse2")]
#[inline]
fn reduce16(acc: __m128i) -> i32 {
    let hi = _mm_srli_si128(acc, 8);
    let sum = _mm_add_epi32(acc, hi);
    _mm_cvtsi128_si32(sum)
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

#[inline(always)]
fn split_probe_header<'b, B: BlockRows + 'b, const W: usize, const SW: usize, const H: usize>(
    s2: &'b B,
) -> ([u8; W], ([u8; W], [u8; W], [u8; W]), B::RawSlice<'b>, usize) {
    let (raw2, st2) = s2.raw_parts();
    let st2 = st2 as usize;
    let p0 = B::slice_prefix(raw2, (H + 1) * st2 + SW);
    let (_, r0_m, _) = B::read_head3::<W, SW>(p0);
    let p1 = B::split_suffix(p0, st2);
    let r1 = B::read_head3::<W, SW>(p1);
    let p2 = B::split_suffix(p1, st2);
    (r0_m, r1, p2, st2)
}

macro_rules! sad_four_sse2_step {
    ($ld:ident, $a:expr, $p:expr, $acc0:ident, $acc1:ident, $acc2:ident, $acc3:ident, $v2_up:ident, $v2_mid:ident, $v2_lt:ident, $v2_rt:ident) => {{
        let (dn_l, dn_m, dn_r) = $p;
        let v1 = $ld(&$a);
        let v2_dn = $ld(&dn_m);
        $acc0 = _mm_add_epi64($acc0, _mm_sad_epu8(v1, $v2_up));
        $acc1 = _mm_add_epi64($acc1, _mm_sad_epu8(v1, v2_dn));
        $acc2 = _mm_add_epi64($acc2, _mm_sad_epu8(v1, $v2_lt));
        $acc3 = _mm_add_epi64($acc3, _mm_sad_epu8(v1, $v2_rt));
        $v2_up = $v2_mid;
        $v2_mid = v2_dn;
        $v2_lt = $ld(&dn_l);
        $v2_rt = $ld(&dn_r);
    }};
}

macro_rules! sad_four_avx2_pair {
    ($a0:expr, $a1:expr, $p2:expr, $p3:expr, $acc0:ident, $acc1:ident, $acc2:ident, $acc3:ident, $v2_r0:ident, $v2_r1:ident, $v2_lt_0:ident, $v2_rt_0:ident) => {{
        let (r2_l, r2_m, r2_r) = $p2;
        let (r3_l, r3_m, r3_r) = $p3;
        let v1 = _mm256_set_m128i(ld16(&$a1), ld16(&$a0));
        let v2_r2 = ld16(&r2_m);
        let v2_r3 = ld16(&r3_m);
        let v2_lt_1 = ld16(&r2_l);
        let v2_rt_1 = ld16(&r2_r);
        $acc0 = _mm256_add_epi64($acc0, _mm256_sad_epu8(v1, _mm256_set_m128i($v2_r1, $v2_r0)));
        $acc1 = _mm256_add_epi64($acc1, _mm256_sad_epu8(v1, _mm256_set_m128i(v2_r3, v2_r2)));
        $acc2 = _mm256_add_epi64(
            $acc2,
            _mm256_sad_epu8(v1, _mm256_set_m128i(v2_lt_1, $v2_lt_0)),
        );
        $acc3 = _mm256_add_epi64(
            $acc3,
            _mm256_sad_epu8(v1, _mm256_set_m128i(v2_rt_1, $v2_rt_0)),
        );
        $v2_r0 = v2_r2;
        $v2_r1 = v2_r3;
        $v2_lt_0 = ld16(&r3_l);
        $v2_rt_0 = ld16(&r3_r);
    }};
    (last $a0:expr, $a1:expr, $p2:expr, $p3:expr, $acc0:ident, $acc1:ident, $acc2:ident, $acc3:ident, $v2_r0:ident, $v2_r1:ident, $v2_lt_0:ident, $v2_rt_0:ident) => {{
        let (r2_l, r2_m, r2_r) = $p2;
        let (_, r3_m, _) = $p3;
        let v1 = _mm256_set_m128i(ld16(&$a1), ld16(&$a0));
        let v2_r2 = ld16(&r2_m);
        let v2_r3 = ld16(&r3_m);
        let v2_lt_1 = ld16(&r2_l);
        let v2_rt_1 = ld16(&r2_r);
        $acc0 = _mm256_add_epi64($acc0, _mm256_sad_epu8(v1, _mm256_set_m128i($v2_r1, $v2_r0)));
        $acc1 = _mm256_add_epi64($acc1, _mm256_sad_epu8(v1, _mm256_set_m128i(v2_r3, v2_r2)));
        $acc2 = _mm256_add_epi64(
            $acc2,
            _mm256_sad_epu8(v1, _mm256_set_m128i(v2_lt_1, $v2_lt_0)),
        );
        $acc3 = _mm256_add_epi64(
            $acc3,
            _mm256_sad_epu8(v1, _mm256_set_m128i(v2_rt_1, $v2_rt_0)),
        );
    }};
}

macro_rules! sad_16x_sse2_quad {
    ($S:ident, $q1:expr, $st1:expr, $q2:expr, $st2:expr, $acc:ident) => {{
        let (a0, a1, a2, a3) = $S::Span::read_quad::<16>($q1, $st1);
        let (c0, c1, c2, c3) = $S::Span::read_quad::<16>($q2, $st2);
        $acc = _mm_add_epi64($acc, _mm_sad_epu8(ld16(&a0), ld16(&c0)));
        $acc = _mm_add_epi64($acc, _mm_sad_epu8(ld16(&a1), ld16(&c1)));
        $acc = _mm_add_epi64($acc, _mm_sad_epu8(ld16(&a2), ld16(&c2)));
        $acc = _mm_add_epi64($acc, _mm_sad_epu8(ld16(&a3), ld16(&c3)));
    }};
}

macro_rules! sad_16x_avx2_quad {
    ($S:ident, $q1:expr, $st1:expr, $q2:expr, $st2:expr, $acc:ident) => {{
        let (a0, a1, a2, a3) = $S::Span::read_quad::<16>($q1, $st1);
        let (c0, c1, c2, c3) = $S::Span::read_quad::<16>($q2, $st2);
        let va0 = _mm256_set_m128i(ld16(&a1), ld16(&a0));
        let vc0 = _mm256_set_m128i(ld16(&c1), ld16(&c0));
        let va1 = _mm256_set_m128i(ld16(&a3), ld16(&a2));
        let vc1 = _mm256_set_m128i(ld16(&c3), ld16(&c2));
        $acc = _mm256_add_epi64($acc, _mm256_sad_epu8(va0, vc0));
        $acc = _mm256_add_epi64($acc, _mm256_sad_epu8(va1, vc1));
    }};
}

simd_entry! {
    #[target_feature(enable = "sse2")]
    fn sad_16x<S: RefSamples, const H: usize>(
        sample1: &S,
        sample2: &S,
        dx: isize,
        dy: isize,
        _token: Sse2Token,
    ) -> i32 {
        const { assert!(H == 8 || H == 16, "sad_16x supports H = 8 or 16") };
        let (s1, s2) = (sample1.span::<16, H>(0, 0), sample2.span::<16, H>(dy, dx));
        let (raw1, st1) = s1.raw_parts();
        let (raw2, st2) = s2.raw_parts();
        let (st1, st2) = (st1 as usize, st2 as usize);
        let mut acc = _mm_setzero_si128();
        if H == 16 {
            let q1_0 = S::Span::slice_prefix(raw1, 15 * st1 + 16);
            let q2_0 = S::Span::slice_prefix(raw2, 15 * st2 + 16);
            sad_16x_sse2_quad!(S, q1_0, st1, q2_0, st2, acc);
            sad_16x_sse2_quad!(S, S::Span::split_suffix(q1_0, 4 * st1), st1, S::Span::split_suffix(q2_0, 4 * st2), st2, acc);
            let q1_2 = S::Span::split_suffix(q1_0, 8 * st1);
            let q2_2 = S::Span::split_suffix(q2_0, 8 * st2);
            sad_16x_sse2_quad!(S, q1_2, st1, q2_2, st2, acc);
            sad_16x_sse2_quad!(S, S::Span::split_suffix(q1_2, 4 * st1), st1, S::Span::split_suffix(q2_2, 4 * st2), st2, acc);
        } else {
            let q1_0 = S::Span::slice_prefix(raw1, 7 * st1 + 16);
            let q2_0 = S::Span::slice_prefix(raw2, 7 * st2 + 16);
            sad_16x_sse2_quad!(S, q1_0, st1, q2_0, st2, acc);
            sad_16x_sse2_quad!(S, S::Span::split_suffix(q1_0, 4 * st1), st1, S::Span::split_suffix(q2_0, 4 * st2), st2, acc);
        }
        reduce16(acc)
    }

    #[target_feature(enable = "sse2")]
    fn sad_8x<S: RefSamples, const H: usize>(
        sample1: &S,
        sample2: &S,
        dx: isize,
        dy: isize,
        _token: Sse2Token,
    ) -> i32 {
        const { assert!(H == 4 || H == 8 || H == 16, "sad_8x supports H = 4, 8, 16") };
        let (s1, s2) = (sample1.span::<8, H>(0, 0), sample2.span::<8, H>(dy, dx));
        let mut acc = _mm_setzero_si128();
        if H == 16 {
            let (qs1, st1) = s1.all_4_quads::<8>();
            let (qs2, st2) = s2.all_4_quads::<8>();
            for (q1, q2) in qs1.into_iter().zip(qs2) {
                let (a0, a1, a2, a3) = S::Span::read_quad::<8>(q1, st1);
                let (c0, c1, c2, c3) = S::Span::read_quad::<8>(q2, st2);
                acc = _mm_add_epi64(acc, _mm_sad_epu8(ld8(&a0), ld8(&c0)));
                acc = _mm_add_epi64(acc, _mm_sad_epu8(ld8(&a1), ld8(&c1)));
                acc = _mm_add_epi64(acc, _mm_sad_epu8(ld8(&a2), ld8(&c2)));
                acc = _mm_add_epi64(acc, _mm_sad_epu8(ld8(&a3), ld8(&c3)));
            }
        } else if H == 8 {
            let (qs1, st1) = s1.all_2_quads::<8>();
            let (qs2, st2) = s2.all_2_quads::<8>();
            for (q1, q2) in qs1.into_iter().zip(qs2) {
                let (a0, a1, a2, a3) = S::Span::read_quad::<8>(q1, st1);
                let (c0, c1, c2, c3) = S::Span::read_quad::<8>(q2, st2);
                acc = _mm_add_epi64(acc, _mm_sad_epu8(ld8(&a0), ld8(&c0)));
                acc = _mm_add_epi64(acc, _mm_sad_epu8(ld8(&a1), ld8(&c1)));
                acc = _mm_add_epi64(acc, _mm_sad_epu8(ld8(&a2), ld8(&c2)));
                acc = _mm_add_epi64(acc, _mm_sad_epu8(ld8(&a3), ld8(&c3)));
            }
        } else {
            let (q1, st1) = s1.all_1_quad::<8>();
            let (q2, st2) = s2.all_1_quad::<8>();
            let (a0, a1, a2, a3) = S::Span::read_quad::<8>(q1, st1);
            let (c0, c1, c2, c3) = S::Span::read_quad::<8>(q2, st2);
            acc = _mm_add_epi64(acc, _mm_sad_epu8(ld8(&a0), ld8(&c0)));
            acc = _mm_add_epi64(acc, _mm_sad_epu8(ld8(&a1), ld8(&c1)));
            acc = _mm_add_epi64(acc, _mm_sad_epu8(ld8(&a2), ld8(&c2)));
            acc = _mm_add_epi64(acc, _mm_sad_epu8(ld8(&a3), ld8(&c3)));
        }
        _mm_cvtsi128_si32(acc)
    }

    #[target_feature(enable = "sse2")]
    fn sad_4x<S: RefSamples, const H: usize>(
        sample1: &S,
        sample2: &S,
        dx: isize,
        dy: isize,
        _token: Sse2Token,
    ) -> i32 {
        const { assert!(H == 4 || H == 8, "sad_4x supports H = 4 or 8") };
        let (s1, s2) = (sample1.span::<4, H>(0, 0), sample2.span::<4, H>(dy, dx));
        let mut acc = _mm_setzero_si128();
        if H == 8 {
            let (qs1, st1) = s1.all_2_quads::<4>();
            let (qs2, st2) = s2.all_2_quads::<4>();
            for (q1, q2) in qs1.into_iter().zip(qs2) {
                let (a0, a1, a2, a3) = S::Span::read_quad::<4>(q1, st1);
                let (c0, c1, c2, c3) = S::Span::read_quad::<4>(q2, st2);
                acc = _mm_add_epi64(acc, _mm_sad_epu8(ld4(&a0), ld4(&c0)));
                acc = _mm_add_epi64(acc, _mm_sad_epu8(ld4(&a1), ld4(&c1)));
                acc = _mm_add_epi64(acc, _mm_sad_epu8(ld4(&a2), ld4(&c2)));
                acc = _mm_add_epi64(acc, _mm_sad_epu8(ld4(&a3), ld4(&c3)));
            }
        } else {
            let (q1, st1) = s1.all_1_quad::<4>();
            let (q2, st2) = s2.all_1_quad::<4>();
            let (a0, a1, a2, a3) = S::Span::read_quad::<4>(q1, st1);
            let (c0, c1, c2, c3) = S::Span::read_quad::<4>(q2, st2);
            acc = _mm_add_epi64(acc, _mm_sad_epu8(ld4(&a0), ld4(&c0)));
            acc = _mm_add_epi64(acc, _mm_sad_epu8(ld4(&a1), ld4(&c1)));
            acc = _mm_add_epi64(acc, _mm_sad_epu8(ld4(&a2), ld4(&c2)));
            acc = _mm_add_epi64(acc, _mm_sad_epu8(ld4(&a3), ld4(&c3)));
        }
        _mm_cvtsi128_si32(acc)
    }

    #[target_feature(enable = "sse2")]
    fn sample_sad_four_16x_sse2<
        S: RefSamples,
        const H: usize,
        const HW: usize,
        const G: usize,
    >(
        sample1: &S,
        sample2: &S,
        sad: &mut [i32; 4],
        _token: Sse2Token,
    ) {
        const { assert!(H == 8 || H == 16, "sample_sad_four_16x_sse2 supports H = 8 or 16") };
        const { assert!(H % G == 0, "the block is a whole number of G-row cuts") };
        const { assert!(HW == H + 2, "the probe span is two rows taller than the block") };
        let s1 = sample1.span::<16, H>(0, 0);
        let s2 = sample2.span::<18, HW>(-1, -1);

        let mut acc0 = _mm_setzero_si128();
        let mut acc1 = _mm_setzero_si128();
        let mut acc2 = _mm_setzero_si128();
        let mut acc3 = _mm_setzero_si128();

        let (r0_m, (r1_l, r1_m, r1_r), p2, st2) = split_probe_header::<_, 16, 18, H>(&s2);
        let mut v2_up = ld16(&r0_m);
        let mut v2_mid = ld16(&r1_m);
        let mut v2_lt = ld16(&r1_l);
        let mut v2_rt = ld16(&r1_r);

        if H == 16 {
            let (qs1, st1) = s1.all_4_quads::<16>();
            let q2_2 = S::Span::split_suffix(p2, 8 * st2);
            let q2_1 = S::Span::split_suffix(p2, 4 * st2);
            let q2_3 = S::Span::split_suffix(q2_2, 4 * st2);
            for (q1, q2) in qs1.into_iter().zip([p2, q2_1, q2_2, q2_3]) {
                let (a0, a1, a2, a3) = S::Span::read_quad::<16>(q1, st1);
                let (p2_0, p2_1, p2_2, p2_3) = S::Span::read_quad3::<16, 18>(q2, st2);
                sad_four_sse2_step!(ld16, a0, p2_0, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld16, a1, p2_1, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld16, a2, p2_2, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld16, a3, p2_3, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
            }
        } else {
            let (qs1, st1) = s1.all_2_quads::<16>();
            let q2_1 = S::Span::split_suffix(p2, 4 * st2);
            for (q1, q2) in qs1.into_iter().zip([p2, q2_1]) {
                let (a0, a1, a2, a3) = S::Span::read_quad::<16>(q1, st1);
                let (p2_0, p2_1, p2_2, p2_3) = S::Span::read_quad3::<16, 18>(q2, st2);
                sad_four_sse2_step!(ld16, a0, p2_0, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld16, a1, p2_1, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld16, a2, p2_2, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld16, a3, p2_3, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
            }
        }

        sad[0] = reduce16(acc0);
        sad[1] = reduce16(acc1);
        sad[2] = reduce16(acc2);
        sad[3] = reduce16(acc3);
    }

    #[target_feature(enable = "sse2")]
    fn sample_sad_four_8x<S: RefSamples, const H: usize, const HW: usize, const G: usize>(
        sample1: &S,
        sample2: &S,
        sad: &mut [i32; 4],
        _token: Sse2Token,
    ) {
        const { assert!(H == 4 || H == 8 || H == 16, "sample_sad_four_8x supports H = 4, 8, 16") };
        const { assert!(H % G == 0, "the block is a whole number of G-row cuts") };
        const { assert!(HW == H + 2, "the probe span is two rows taller than the block") };
        let s1 = sample1.span::<8, H>(0, 0);
        let s2 = sample2.span::<10, HW>(-1, -1);

        let mut acc0 = _mm_setzero_si128();
        let mut acc1 = _mm_setzero_si128();
        let mut acc2 = _mm_setzero_si128();
        let mut acc3 = _mm_setzero_si128();

        let (r0_m, (r1_l, r1_m, r1_r), p2, st2) = split_probe_header::<_, 8, 10, H>(&s2);
        let mut v2_up = ld8(&r0_m);
        let mut v2_mid = ld8(&r1_m);
        let mut v2_lt = ld8(&r1_l);
        let mut v2_rt = ld8(&r1_r);

        if H == 16 {
            let (qs1, st1) = s1.all_4_quads::<8>();
            let q2_2 = S::Span::split_suffix(p2, 8 * st2);
            let q2_1 = S::Span::split_suffix(p2, 4 * st2);
            let q2_3 = S::Span::split_suffix(q2_2, 4 * st2);
            for (q1, q2) in qs1.into_iter().zip([p2, q2_1, q2_2, q2_3]) {
                let (a0, a1, a2, a3) = S::Span::read_quad::<8>(q1, st1);
                let (p2_0, p2_1, p2_2, p2_3) = S::Span::read_quad3::<8, 10>(q2, st2);
                sad_four_sse2_step!(ld8, a0, p2_0, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld8, a1, p2_1, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld8, a2, p2_2, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld8, a3, p2_3, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
            }
        } else if H == 8 {
            let (qs1, st1) = s1.all_2_quads::<8>();
            let q2_1 = S::Span::split_suffix(p2, 4 * st2);
            for (q1, q2) in qs1.into_iter().zip([p2, q2_1]) {
                let (a0, a1, a2, a3) = S::Span::read_quad::<8>(q1, st1);
                let (p2_0, p2_1, p2_2, p2_3) = S::Span::read_quad3::<8, 10>(q2, st2);
                sad_four_sse2_step!(ld8, a0, p2_0, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld8, a1, p2_1, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld8, a2, p2_2, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld8, a3, p2_3, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
            }
        } else {
            let (q1, st1) = s1.all_1_quad::<8>();
            for (q1, q2) in [q1].into_iter().zip([p2]) {
                let (a0, a1, a2, a3) = S::Span::read_quad::<8>(q1, st1);
                let (p2_0, p2_1, p2_2, p2_3) = S::Span::read_quad3::<8, 10>(q2, st2);
                sad_four_sse2_step!(ld8, a0, p2_0, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld8, a1, p2_1, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld8, a2, p2_2, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld8, a3, p2_3, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
            }
        }

        sad[0] = _mm_cvtsi128_si32(acc0);
        sad[1] = _mm_cvtsi128_si32(acc1);
        sad[2] = _mm_cvtsi128_si32(acc2);
        sad[3] = _mm_cvtsi128_si32(acc3);
    }

    #[target_feature(enable = "sse2")]
    fn sample_sad_four_4x<S: RefSamples, const H: usize, const HW: usize, const G: usize>(
        sample1: &S,
        sample2: &S,
        sad: &mut [i32; 4],
        _token: Sse2Token,
    ) {
        const { assert!(H == 4 || H == 8, "sample_sad_four_4x supports H = 4 or 8") };
        const { assert!(H % G == 0, "the block is a whole number of G-row cuts") };
        const { assert!(HW == H + 2, "the probe span is two rows taller than the block") };
        let s1 = sample1.span::<4, H>(0, 0);
        let s2 = sample2.span::<6, HW>(-1, -1);

        let mut acc0 = _mm_setzero_si128();
        let mut acc1 = _mm_setzero_si128();
        let mut acc2 = _mm_setzero_si128();
        let mut acc3 = _mm_setzero_si128();

        let (r0_m, (r1_l, r1_m, r1_r), p2, st2) = split_probe_header::<_, 4, 6, H>(&s2);
        let mut v2_up = ld4(&r0_m);
        let mut v2_mid = ld4(&r1_m);
        let mut v2_lt = ld4(&r1_l);
        let mut v2_rt = ld4(&r1_r);

        if H == 8 {
            let (qs1, st1) = s1.all_2_quads::<4>();
            let q2_1 = S::Span::split_suffix(p2, 4 * st2);
            for (q1, q2) in qs1.into_iter().zip([p2, q2_1]) {
                let (a0, a1, a2, a3) = S::Span::read_quad::<4>(q1, st1);
                let (p2_0, p2_1, p2_2, p2_3) = S::Span::read_quad3::<4, 6>(q2, st2);
                sad_four_sse2_step!(ld4, a0, p2_0, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld4, a1, p2_1, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld4, a2, p2_2, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld4, a3, p2_3, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
            }
        } else {
            let (q1, st1) = s1.all_1_quad::<4>();
            for (q1, q2) in [q1].into_iter().zip([p2]) {
                let (a0, a1, a2, a3) = S::Span::read_quad::<4>(q1, st1);
                let (p2_0, p2_1, p2_2, p2_3) = S::Span::read_quad3::<4, 6>(q2, st2);
                sad_four_sse2_step!(ld4, a0, p2_0, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld4, a1, p2_1, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld4, a2, p2_2, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
                sad_four_sse2_step!(ld4, a3, p2_3, acc0, acc1, acc2, acc3, v2_up, v2_mid, v2_lt, v2_rt);
            }
        }

        sad[0] = _mm_cvtsi128_si32(acc0);
        sad[1] = _mm_cvtsi128_si32(acc1);
        sad[2] = _mm_cvtsi128_si32(acc2);
        sad[3] = _mm_cvtsi128_si32(acc3);
    }

    #[target_feature(enable = "avx2")]
    fn sad_16x_avx2<S: RefSamples, const H: usize>(
        sample1: &S,
        sample2: &S,
        _token: Avx2Token,
    ) -> i32 {
        const {
            assert!(
                H == 8 || H == 16,
                "sad_16x_avx2 supports H = 8 or 16"
            )
        };
        let (s1, s2) = (sample1.span::<16, H>(0, 0), sample2.span::<16, H>(0, 0));
        let (raw1, st1) = s1.raw_parts();
        let (raw2, st2) = s2.raw_parts();
        let (st1, st2) = (st1 as usize, st2 as usize);
        let mut acc = _mm256_setzero_si256();
        if H == 16 {
            let q1_0 = S::Span::slice_prefix(raw1, 15 * st1 + 16);
            let q2_0 = S::Span::slice_prefix(raw2, 15 * st2 + 16);
            sad_16x_avx2_quad!(S, q1_0, st1, q2_0, st2, acc);
            sad_16x_avx2_quad!(S, S::Span::split_suffix(q1_0, 4 * st1), st1, S::Span::split_suffix(q2_0, 4 * st2), st2, acc);
            let q1_2 = S::Span::split_suffix(q1_0, 8 * st1);
            let q2_2 = S::Span::split_suffix(q2_0, 8 * st2);
            sad_16x_avx2_quad!(S, q1_2, st1, q2_2, st2, acc);
            sad_16x_avx2_quad!(S, S::Span::split_suffix(q1_2, 4 * st1), st1, S::Span::split_suffix(q2_2, 4 * st2), st2, acc);
        } else {
            let q1_0 = S::Span::slice_prefix(raw1, 7 * st1 + 16);
            let q2_0 = S::Span::slice_prefix(raw2, 7 * st2 + 16);
            sad_16x_avx2_quad!(S, q1_0, st1, q2_0, st2, acc);
            sad_16x_avx2_quad!(S, S::Span::split_suffix(q1_0, 4 * st1), st1, S::Span::split_suffix(q2_0, 4 * st2), st2, acc);
        }
        reduce256(acc)
    }

    #[target_feature(enable = "avx2")]
    fn sample_sad_four_16x_avx2<
        S: RefSamples,
        const H: usize,
        const HW: usize,
        const G: usize,
    >(
        sample1: &S,
        sample2: &S,
        sad: &mut [i32; 4],
        _token: Avx2Token,
    ) {
        const { assert!(H == 8 || H == 16, "sample_sad_four_16x_avx2 supports H = 8 or 16") };
        const { assert!(G % 2 == 0 && H % G == 0, "the block is a whole number of even G-row cuts") };
        const { assert!(HW == H + 2, "HW must equal H + 2") };
        let s1 = sample1.span::<16, H>(0, 0);
        let s2 = sample2.span::<18, HW>(-1, -1);

        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();

        let (r0_m, (r1_l, r1_m, r1_r), p2, st2) = split_probe_header::<_, 16, 18, H>(&s2);
        let mut v2_r0 = ld16(&r0_m);
        let mut v2_r1 = ld16(&r1_m);
        let mut v2_lt_0 = ld16(&r1_l);
        let mut v2_rt_0 = ld16(&r1_r);

        let (raw1, st1) = s1.raw_parts();
        let st1 = st1 as usize;
        if H == 16 {
            let q1_0 = S::Span::slice_prefix(raw1, 15 * st1 + 16);
            let (a0, a1, a2, a3) = S::Span::read_quad::<16>(q1_0, st1);
            let (p2_0, p2_1, p2_2, p2_3) = S::Span::read_quad3::<16, 18>(p2, st2);
            sad_four_avx2_pair!(a0, a1, p2_0, p2_1, acc0, acc1, acc2, acc3, v2_r0, v2_r1, v2_lt_0, v2_rt_0);
            sad_four_avx2_pair!(a2, a3, p2_2, p2_3, acc0, acc1, acc2, acc3, v2_r0, v2_r1, v2_lt_0, v2_rt_0);

            let (a0, a1, a2, a3) = S::Span::read_quad::<16>(S::Span::split_suffix(q1_0, 4 * st1), st1);
            let (p2_0, p2_1, p2_2, p2_3) = S::Span::read_quad3::<16, 18>(S::Span::split_suffix(p2, 4 * st2), st2);
            sad_four_avx2_pair!(a0, a1, p2_0, p2_1, acc0, acc1, acc2, acc3, v2_r0, v2_r1, v2_lt_0, v2_rt_0);
            sad_four_avx2_pair!(a2, a3, p2_2, p2_3, acc0, acc1, acc2, acc3, v2_r0, v2_r1, v2_lt_0, v2_rt_0);

            let q1_2 = S::Span::split_suffix(q1_0, 8 * st1);
            let q2_2 = S::Span::split_suffix(p2, 8 * st2);
            let (a0, a1, a2, a3) = S::Span::read_quad::<16>(q1_2, st1);
            let (p2_0, p2_1, p2_2, p2_3) = S::Span::read_quad3::<16, 18>(q2_2, st2);
            sad_four_avx2_pair!(a0, a1, p2_0, p2_1, acc0, acc1, acc2, acc3, v2_r0, v2_r1, v2_lt_0, v2_rt_0);
            sad_four_avx2_pair!(a2, a3, p2_2, p2_3, acc0, acc1, acc2, acc3, v2_r0, v2_r1, v2_lt_0, v2_rt_0);

            let (a0, a1, a2, a3) = S::Span::read_quad::<16>(S::Span::split_suffix(q1_2, 4 * st1), st1);
            let (p2_0, p2_1, p2_2, p2_3) = S::Span::read_quad3::<16, 18>(S::Span::split_suffix(q2_2, 4 * st2), st2);
            sad_four_avx2_pair!(a0, a1, p2_0, p2_1, acc0, acc1, acc2, acc3, v2_r0, v2_r1, v2_lt_0, v2_rt_0);
            sad_four_avx2_pair!(last a2, a3, p2_2, p2_3, acc0, acc1, acc2, acc3, v2_r0, v2_r1, v2_lt_0, v2_rt_0);
        } else {
            let q1_0 = S::Span::slice_prefix(raw1, 7 * st1 + 16);
            let (a0, a1, a2, a3) = S::Span::read_quad::<16>(q1_0, st1);
            let (p2_0, p2_1, p2_2, p2_3) = S::Span::read_quad3::<16, 18>(p2, st2);
            sad_four_avx2_pair!(a0, a1, p2_0, p2_1, acc0, acc1, acc2, acc3, v2_r0, v2_r1, v2_lt_0, v2_rt_0);
            sad_four_avx2_pair!(a2, a3, p2_2, p2_3, acc0, acc1, acc2, acc3, v2_r0, v2_r1, v2_lt_0, v2_rt_0);

            let (a0, a1, a2, a3) = S::Span::read_quad::<16>(S::Span::split_suffix(q1_0, 4 * st1), st1);
            let (p2_0, p2_1, p2_2, p2_3) = S::Span::read_quad3::<16, 18>(S::Span::split_suffix(p2, 4 * st2), st2);
            sad_four_avx2_pair!(a0, a1, p2_0, p2_1, acc0, acc1, acc2, acc3, v2_r0, v2_r1, v2_lt_0, v2_rt_0);
            sad_four_avx2_pair!(last a2, a3, p2_2, p2_3, acc0, acc1, acc2, acc3, v2_r0, v2_r1, v2_lt_0, v2_rt_0);
        }

        sad[0] = reduce256(acc0);
        sad[1] = reduce256(acc1);
        sad[2] = reduce256(acc2);
        sad[3] = reduce256(acc3);
    }
}

#[inline(always)]
fn sample_sad_four_16x<S: RefSamples, const H: usize, const HW: usize, const G: usize>(
    sample1: &S,
    sample2: &S,
    sad: &mut [i32; 4],
) {
    #[cold]
    #[inline(never)]
    fn fallback<S: RefSamples, const H: usize, const HW: usize, const G: usize>(
        sample1: &S,
        sample2: &S,
        sad: &mut [i32; 4],
    ) {
        if let Some(token) = Avx2Token::try_new_enabled() {
            sample_sad_four_16x_avx2::<S, H, HW, G>(sample1, sample2, sad, token);
        } else {
            sample_sad_four_16x_sse2::<S, H, HW, G>(sample1, sample2, sad, sse2_token());
        }
    }
    if let Some(token) = Avx2Token::try_fast_enabled() {
        sample_sad_four_16x_avx2::<S, H, HW, G>(sample1, sample2, sad, token);
        return;
    }
    fallback::<S, H, HW, G>(sample1, sample2, sad);
}

// ============================================================================
// Safe Public Wrappers
// ============================================================================

#[inline(always)]
pub fn sample_sad_16x16<S: RefSamples>(sample1: &S, sample2: &S) -> i32 {
    sad_16x::<S, 16>(sample1, sample2, 0, 0, sse2_token())
}

struct Sad16xKernel<const H: usize>;
impl<const H: usize> Avx2PlaneSadKernel for Sad16xKernel<H> {
    #[inline(always)]
    fn run(a: &PlaneCursor<'_>, b: &PlaneCursor<'_>, token: Avx2Token) -> i32 {
        sad_16x_avx2::<PlaneCursor<'_>, H>(a, b, token)
    }
}

/// Returns a direct `sad_16x_avx2` function pointer when AVX2 is available on the host,
/// amortizing the capability check at table-init time (`WelsInitSampleSadFunc`).
// Dead in a kernel set no dispatch site aliases.
#[allow(dead_code)]
#[inline]
pub(crate) fn sample_sad_16x16_avx2() -> Option<fn(&PlaneCursor<'_>, &PlaneCursor<'_>) -> i32> {
    Avx2Token::try_new().map(Avx2Token::into_plane_sad_fn::<Sad16xKernel<16>>)
}

#[inline(always)]
pub fn sample_sad_16x8<S: RefSamples>(sample1: &S, sample2: &S) -> i32 {
    sad_16x::<S, 8>(sample1, sample2, 0, 0, sse2_token())
}

/// Returns a direct `sad_16x_avx2` function pointer when AVX2 is available on the host,
/// amortizing the capability check at table-init time (`WelsInitSampleSadFunc`).
// Dead in a kernel set no dispatch site aliases.
#[allow(dead_code)]
#[inline]
pub(crate) fn sample_sad_16x8_avx2() -> Option<fn(&PlaneCursor<'_>, &PlaneCursor<'_>) -> i32> {
    Avx2Token::try_new().map(Avx2Token::into_plane_sad_fn::<Sad16xKernel<8>>)
}

#[inline(always)]
pub fn sample_sad_8x16<S: RefSamples>(sample1: &S, sample2: &S) -> i32 {
    sad_8x::<S, 16>(sample1, sample2, 0, 0, sse2_token())
}

#[inline(always)]
pub fn sample_sad_8x8<S: RefSamples>(sample1: &S, sample2: &S) -> i32 {
    sad_8x::<S, 8>(sample1, sample2, 0, 0, sse2_token())
}

#[inline(always)]
pub fn sample_sad_4x4<S: RefSamples>(sample1: &S, sample2: &S) -> i32 {
    sad_4x::<S, 4>(sample1, sample2, 0, 0, sse2_token())
}

#[inline(always)]
pub fn sample_sad_8x4<S: RefSamples>(sample1: &S, sample2: &S) -> i32 {
    sad_8x::<S, 4>(sample1, sample2, 0, 0, sse2_token())
}

#[inline(always)]
pub fn sample_sad_4x8<S: RefSamples>(sample1: &S, sample2: &S) -> i32 {
    sad_4x::<S, 8>(sample1, sample2, 0, 0, sse2_token())
}

#[inline(always)]
pub fn sample_sad_four_16x16<S: RefSamples>(sample1: &S, sample2: &S, sad: &mut [i32; 4]) {
    sample_sad_four_16x::<S, 16, 18, 4>(sample1, sample2, sad)
}

#[inline(always)]
pub fn sample_sad_four_16x8<S: RefSamples>(sample1: &S, sample2: &S, sad: &mut [i32; 4]) {
    sample_sad_four_16x::<S, 8, 10, 4>(sample1, sample2, sad)
}

#[inline(always)]
pub fn sample_sad_four_8x16<S: RefSamples>(sample1: &S, sample2: &S, sad: &mut [i32; 4]) {
    sample_sad_four_8x::<S, 16, 18, 8>(sample1, sample2, sad, sse2_token())
}

#[inline(always)]
pub fn sample_sad_four_8x8<S: RefSamples>(sample1: &S, sample2: &S, sad: &mut [i32; 4]) {
    sample_sad_four_8x::<S, 8, 10, 8>(sample1, sample2, sad, sse2_token())
}

#[inline(always)]
pub fn sample_sad_four_4x4<S: RefSamples>(sample1: &S, sample2: &S, sad: &mut [i32; 4]) {
    sample_sad_four_4x::<S, 4, 6, 4>(sample1, sample2, sad, sse2_token())
}

#[inline(always)]
pub fn sample_sad_four_8x4<S: RefSamples>(sample1: &S, sample2: &S, sad: &mut [i32; 4]) {
    sample_sad_four_8x::<S, 4, 6, 4>(sample1, sample2, sad, sse2_token())
}

#[inline(always)]
pub fn sample_sad_four_4x8<S: RefSamples>(sample1: &S, sample2: &S, sad: &mut [i32; 4]) {
    sample_sad_four_4x::<S, 8, 10, 8>(sample1, sample2, sad, sse2_token())
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

    #[test]
    fn test_avx2_sad_parity() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let (p1, p2) = make_test_planes(64, 64);
        let c1 = PlaneCursor::new(&p1, 64 * 8 + 8, 64);
        let c2 = PlaneCursor::new(&p2, 64 * 8 + 8, 64);

        let token = Avx2Token::try_new().expect("avx2 available");
        assert_eq!(
            sad_16x_avx2::<_, 16>(&c1, &c2, token),
            sample_sad::<16, 16, _>(&c1, &c2)
        );
        assert_eq!(
            sad_16x_avx2::<_, 8>(&c1, &c2, token),
            sample_sad::<16, 8, _>(&c1, &c2)
        );
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
    // Input and anchor coverage: every kernel over four anchors, one per residue
    // class mod 8, and five distributions. The all-`0xFF`/all-`0x00` and
    // near-identical pairs sit at the ends of the accumulator's range, where a
    // `psadbw` accumulation that widened or saturated wrongly would show.
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

    /// Four anchors covering every residue mod 4 (with residues [0, 3, 6, 5] mod 8). Each
    /// leaves at least 16 rows and 16 columns of margin on all sides.
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

    /// The AVX2 pair over the same sweep; on a host without AVX2 it prints that it
    /// executed nothing rather than passing silently.
    #[test]
    fn avx2_sad_parity_over_anchors_and_distributions() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!(
                "SKIPPED avx2_sad_parity_over_anchors_and_distributions: \
                 this host has no AVX2, so the two AVX2 kernels were not executed"
            );
            return;
        }
        let token = Avx2Token::try_new().expect("avx2 available");
        for (name, p1, p2) in input_pairs(64, 64) {
            for anchor in ANCHORS {
                let c1 = PlaneCursor::new(&p1, anchor, 64);
                let c2 = PlaneCursor::new(&p2, anchor, 64);
                let at = format!("{name} @ {anchor}");
                assert_eq!(
                    sad_16x_avx2::<_, 16>(&c1, &c2, token),
                    sample_sad::<16, 16, _>(&c1, &c2),
                    "16x16 avx2 {at}"
                );
                assert_eq!(
                    sad_16x_avx2::<_, 8>(&c1, &c2, token),
                    sample_sad::<16, 8, _>(&c1, &c2),
                    "16x8 avx2 {at}"
                );
            }
        }
    }

    /// `WelsInitSampleSadFunc` fills `pfSampleSad[BLOCK_16x16]` from the AVX2 kernel
    /// exactly when `uiCpuFlag` asks for AVX2 *and* this CPU has it. On a host with
    /// AVX2 both spellings install the same kernel, so only the flag half is pinned
    /// there.
    ///
    /// The comparison is function-pointer identity: both addresses come from the same
    /// `WelsInitSampleSadFunc` instantiation, but Miri mints a fresh synthetic address
    /// per reification and is excluded.
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

        // The oracle is `has_avx2()`, not `is_x86_feature_detected!`: the table arm
        // consults the crate's probe, which answers from the build as well as the CPU —
        // under `--features scalar` the feature word is `0`, so a host that *has* AVX2
        // still gets the baseline entry.
        if crate::simd::has_avx2() {
            assert_ne!(
                asked_for_avx2, baseline,
                "has_avx2() is true, so asking for AVX2 must change the installed kernel"
            );
        } else {
            assert_eq!(
                asked_for_avx2, baseline,
                "has_avx2() is false, so the flag alone must not install an AVX2 kernel"
            );
        }
    }
}
