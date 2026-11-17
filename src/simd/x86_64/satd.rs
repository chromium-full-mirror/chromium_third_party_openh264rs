// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! x86_64 SSE2 & AVX2 implementations of SATD (Hadamard transformed SAD).
//!
//! The 4x4 block every shape is built from cuts each operand once into a
//! `RefSamples::span` and indexes its four rows inside it: one bounds cut per operand
//! rather than two checks per row. See `RefSamples::span`.
#![deny(unsafe_code)]

use super::lanes::{ld4, ld16, ld32_i8};
use super::sse2_token;
use crate::safe::plane::{BlockRows, RefSamples};
use crate::simd::{
    simd_entry,
    tokens::{Avx2Token, Sse2Token},
};
use core::arch::x86_64::*;

#[target_feature(enable = "sse2")]
#[inline]
fn abs_epi16(v: __m128i) -> __m128i {
    let sign = _mm_srai_epi16(v, 15);
    _mm_sub_epi16(_mm_xor_si128(v, sign), sign)
}

#[target_feature(enable = "sse2")]
#[inline]
fn sum_sub(a: &mut __m128i, b: &mut __m128i) {
    let tmp = *b;
    *b = _mm_add_epi16(*a, *b);
    *a = _mm_sub_epi16(*a, tmp);
}

/// 1D 4-point Hadamard transform over SIMD vectors:
/// in: r0, r1, r2, r3 -> out: r0, r2, r1, r3 (butterfly permutation).
#[target_feature(enable = "sse2")]
#[inline]
fn hdm4(r0: &mut __m128i, r1: &mut __m128i, r2: &mut __m128i, r3: &mut __m128i) {
    sum_sub(r0, r1);
    sum_sub(r2, r3);
    sum_sub(r1, r3);
    sum_sub(r0, r2);
}

/// Transposes 4x4 matrix of 16-bit words in lower 64 bits of 4 registers.
/// Returns (col01, col23) where col01 = [col0 (64b), col1 (64b)] and col23 = [col2 (64b), col3 (64b)].
#[target_feature(enable = "sse2")]
#[inline]
fn transpose_4x4_w(r0: __m128i, r1: __m128i, r2: __m128i, r3: __m128i) -> (__m128i, __m128i) {
    let t01 = _mm_unpacklo_epi16(r0, r1);
    let t23 = _mm_unpacklo_epi16(r2, r3);
    let c01 = _mm_unpacklo_epi32(t01, t23);
    let c23 = _mm_unpackhi_epi32(t01, t23);
    (c01, c23)
}

/// Horizontal sum of all 8 unsigned 16-bit integers in a 128-bit register.
#[target_feature(enable = "sse2")]
#[inline]
fn sum_u16_8(v: __m128i) -> i32 {
    let hi64 = _mm_srli_si128(v, 8);
    let sum64 = _mm_add_epi16(v, hi64);
    let hi32 = _mm_srli_si128(sum64, 4);
    let sum32 = _mm_add_epi16(sum64, hi32);
    let hi16 = _mm_srli_si128(sum32, 2);
    let sum16 = _mm_add_epi16(sum32, hi16);
    _mm_cvtsi128_si32(sum16) & 0xFFFF
}

#[rustfmt::skip]
const HSUM_SUB_DB1_256: [i8; 32] = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
];

#[rustfmt::skip]
const HSUM_SUB_DB1_128X2: [i8; 32] = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1,
];

#[target_feature(enable = "avx2")]
#[inline]
fn sum_w_horizon_avx2(acc: __m256i) -> i32 {
    let ones = _mm256_set1_epi16(1);
    let dwords = _mm256_madd_epi16(acc, ones);
    let lo128 = _mm256_castsi256_si128(dwords);
    let hi128 = _mm256_extracti128_si256(dwords, 1);
    let sum128 = _mm_add_epi32(lo128, hi128);
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_srli_si128(sum64, 4);
    let total = _mm_add_epi32(sum64, hi32);
    _mm_cvtsi128_si32(total)
}

macro_rules! satd_16x4_step {
    (
        $r1_0:expr, $r2_0:expr,
        $r1_1:expr, $r2_1:expr,
        $r1_2:expr, $r2_2:expr,
        $r1_3:expr, $r2_3:expr,
        $hsum_const:expr
    ) => {{
        let a0: [u8; 16] = $r1_0;
        let b0: [u8; 16] = $r2_0;
        let a1: [u8; 16] = $r1_1;
        let b1: [u8; 16] = $r2_1;
        let a2: [u8; 16] = $r1_2;
        let b2: [u8; 16] = $r2_2;
        let a3: [u8; 16] = $r1_3;
        let b3: [u8; 16] = $r2_3;
        let hsum_const = $hsum_const;

        let y1_0 = _mm256_broadcastsi128_si256(ld16(&a0));
        let y2_0 = _mm256_broadcastsi128_si256(ld16(&b0));
        let d0 = _mm256_sub_epi16(
            _mm256_maddubs_epi16(y1_0, hsum_const),
            _mm256_maddubs_epi16(y2_0, hsum_const),
        );

        let y1_1 = _mm256_broadcastsi128_si256(ld16(&a1));
        let y2_1 = _mm256_broadcastsi128_si256(ld16(&b1));
        let d1 = _mm256_sub_epi16(
            _mm256_maddubs_epi16(y1_1, hsum_const),
            _mm256_maddubs_epi16(y2_1, hsum_const),
        );

        let y1_2 = _mm256_broadcastsi128_si256(ld16(&a2));
        let y2_2 = _mm256_broadcastsi128_si256(ld16(&b2));
        let d2 = _mm256_sub_epi16(
            _mm256_maddubs_epi16(y1_2, hsum_const),
            _mm256_maddubs_epi16(y2_2, hsum_const),
        );

        let y1_3 = _mm256_broadcastsi128_si256(ld16(&a3));
        let y2_3 = _mm256_broadcastsi128_si256(ld16(&b3));
        let d3 = _mm256_sub_epi16(
            _mm256_maddubs_epi16(y1_3, hsum_const),
            _mm256_maddubs_epi16(y2_3, hsum_const),
        );

        let s3 = _mm256_sub_epi16(d0, d3);
        let s0 = _mm256_add_epi16(d0, d3);
        let s2 = _mm256_sub_epi16(d1, d2);
        let s1 = _mm256_add_epi16(d1, d2);

        let y0 = _mm256_abs_epi16(_mm256_add_epi16(s0, s1));
        let y2 = _mm256_abs_epi16(_mm256_sub_epi16(s0, s1));
        let y1 = _mm256_abs_epi16(_mm256_add_epi16(s3, s2));
        let y3 = _mm256_abs_epi16(_mm256_sub_epi16(s3, s2));

        let t1 = _mm256_blend_epi16(y0, y1, 0xAA);
        let shifted_y1 = _mm256_slli_epi32(y1, 16);
        let shifted_y0 = _mm256_srli_epi32(y0, 16);
        let ored1 = _mm256_or_si256(shifted_y1, shifted_y0);
        let max1 = _mm256_max_epu16(ored1, t1);

        let t2 = _mm256_blend_epi16(y2, y3, 0xAA);
        let shifted_y3 = _mm256_slli_epi32(y3, 16);
        let shifted_y2 = _mm256_srli_epi32(y2, 16);
        let ored2 = _mm256_or_si256(shifted_y3, shifted_y2);
        let max2 = _mm256_max_epu16(t2, ored2);

        _mm256_add_epi16(max1, max2)
    }};
}

macro_rules! satd_8x8_step {
    (
        $r1_0:expr, $r1_1:expr, $r1_2:expr, $r1_3:expr,
        $r1_4:expr, $r1_5:expr, $r1_6:expr, $r1_7:expr,
        $r2_0:expr, $r2_1:expr, $r2_2:expr, $r2_3:expr,
        $r2_4:expr, $r2_5:expr, $r2_6:expr, $r2_7:expr,
        $hsum_const:expr
    ) => {{
        let r1_0 = i64::from_ne_bytes($r1_0);
        let r1_1 = i64::from_ne_bytes($r1_1);
        let r1_2 = i64::from_ne_bytes($r1_2);
        let r1_3 = i64::from_ne_bytes($r1_3);
        let r1_4 = i64::from_ne_bytes($r1_4);
        let r1_5 = i64::from_ne_bytes($r1_5);
        let r1_6 = i64::from_ne_bytes($r1_6);
        let r1_7 = i64::from_ne_bytes($r1_7);

        let r2_0 = i64::from_ne_bytes($r2_0);
        let r2_1 = i64::from_ne_bytes($r2_1);
        let r2_2 = i64::from_ne_bytes($r2_2);
        let r2_3 = i64::from_ne_bytes($r2_3);
        let r2_4 = i64::from_ne_bytes($r2_4);
        let r2_5 = i64::from_ne_bytes($r2_5);
        let r2_6 = i64::from_ne_bytes($r2_6);
        let r2_7 = i64::from_ne_bytes($r2_7);
        let hsum_const = $hsum_const;

        let v1_04 = _mm256_set_epi64x(r1_4, r1_4, r1_0, r1_0);
        let v2_04 = _mm256_set_epi64x(r2_4, r2_4, r2_0, r2_0);
        let d04 = _mm256_sub_epi16(
            _mm256_maddubs_epi16(v1_04, hsum_const),
            _mm256_maddubs_epi16(v2_04, hsum_const),
        );

        let v1_15 = _mm256_set_epi64x(r1_5, r1_5, r1_1, r1_1);
        let v2_15 = _mm256_set_epi64x(r2_5, r2_5, r2_1, r2_1);
        let d15 = _mm256_sub_epi16(
            _mm256_maddubs_epi16(v1_15, hsum_const),
            _mm256_maddubs_epi16(v2_15, hsum_const),
        );

        let v1_26 = _mm256_set_epi64x(r1_6, r1_6, r1_2, r1_2);
        let v2_26 = _mm256_set_epi64x(r2_6, r2_6, r2_2, r2_2);
        let d26 = _mm256_sub_epi16(
            _mm256_maddubs_epi16(v1_26, hsum_const),
            _mm256_maddubs_epi16(v2_26, hsum_const),
        );

        let v1_37 = _mm256_set_epi64x(r1_7, r1_7, r1_3, r1_3);
        let v2_37 = _mm256_set_epi64x(r2_7, r2_7, r2_3, r2_3);
        let d37 = _mm256_sub_epi16(
            _mm256_maddubs_epi16(v1_37, hsum_const),
            _mm256_maddubs_epi16(v2_37, hsum_const),
        );

        let s3 = _mm256_sub_epi16(d04, d37);
        let s0 = _mm256_add_epi16(d04, d37);
        let s2 = _mm256_sub_epi16(d15, d26);
        let s1 = _mm256_add_epi16(d15, d26);

        let y0 = _mm256_abs_epi16(_mm256_add_epi16(s0, s1));
        let y2 = _mm256_abs_epi16(_mm256_sub_epi16(s0, s1));
        let y1 = _mm256_abs_epi16(_mm256_add_epi16(s3, s2));
        let y3 = _mm256_abs_epi16(_mm256_sub_epi16(s3, s2));

        let t1 = _mm256_blend_epi16(y0, y1, 0xAA);
        let shifted_y1 = _mm256_slli_epi32(y1, 16);
        let shifted_y0 = _mm256_srli_epi32(y0, 16);
        let ored1 = _mm256_or_si256(shifted_y1, shifted_y0);
        let max1 = _mm256_max_epu16(ored1, t1);

        let t2 = _mm256_blend_epi16(y2, y3, 0xAA);
        let shifted_y3 = _mm256_slli_epi32(y3, 16);
        let shifted_y2 = _mm256_srli_epi32(y2, 16);
        let ored2 = _mm256_or_si256(shifted_y3, shifted_y2);
        let max2 = _mm256_max_epu16(t2, ored2);

        _mm256_add_epi16(max1, max2)
    }};
}

#[target_feature(enable = "sse2")]
#[inline]
fn satd_4x4_sse2<A: RefSamples + Copy, B: RefSamples + Copy>(c1: &A, c2: &B) -> i32 {
    let (s1, s2) = (c1.span::<4, 4>(0, 0), c2.span::<4, 4>(0, 0));
    let (b1, st1) = s1.all_1_quad::<4>();
    let (b2, st2) = s2.all_1_quad::<4>();

    let (r1_0, r1_1, r1_2, r1_3) = A::Span::read_quad::<4>(b1, st1);
    let (r2_0, r2_1, r2_2, r2_3) = B::Span::read_quad::<4>(b2, st2);
    let v1_0 = ld4(&r1_0);
    let v2_0 = ld4(&r2_0);
    let v1_1 = ld4(&r1_1);
    let v2_1 = ld4(&r2_1);
    let v1_2 = ld4(&r1_2);
    let v2_2 = ld4(&r2_2);
    let v1_3 = ld4(&r1_3);
    let v2_3 = ld4(&r2_3);

    let zero = _mm_setzero_si128();
    let mut d0 = _mm_sub_epi16(_mm_unpacklo_epi8(v1_0, zero), _mm_unpacklo_epi8(v2_0, zero));
    let mut d1 = _mm_sub_epi16(_mm_unpacklo_epi8(v1_1, zero), _mm_unpacklo_epi8(v2_1, zero));
    let mut d2 = _mm_sub_epi16(_mm_unpacklo_epi8(v1_2, zero), _mm_unpacklo_epi8(v2_2, zero));
    let mut d3 = _mm_sub_epi16(_mm_unpacklo_epi8(v1_3, zero), _mm_unpacklo_epi8(v2_3, zero));

    hdm4(&mut d0, &mut d1, &mut d2, &mut d3);

    let (c01, c23) = transpose_4x4_w(d0, d1, d2, d3);
    let mut col0 = c01;
    let mut col1 = _mm_srli_si128(c01, 8);
    let mut col2 = c23;
    let mut col3 = _mm_srli_si128(c23, 8);

    hdm4(&mut col0, &mut col1, &mut col2, &mut col3);

    let abs0 = abs_epi16(col0);
    let abs1 = abs_epi16(col1);
    let abs2 = abs_epi16(col2);
    let abs3 = abs_epi16(col3);

    let abs01 = _mm_unpacklo_epi64(abs0, abs1);
    let abs23 = _mm_unpacklo_epi64(abs2, abs3);
    let total_v = _mm_add_epi16(abs01, abs23);

    (sum_u16_8(total_v) + 1) >> 1
}

#[target_feature(enable = "sse2")]
#[inline]
fn satd_8x8_sse2<A: RefSamples + Copy, B: RefSamples + Copy>(c1: &A, c2: &B) -> i32 {
    satd_4x4_sse2(c1, c2)
        + satd_4x4_sse2(&c1.advance(4, 0), &c2.advance(4, 0))
        + satd_4x4_sse2(&c1.advance(0, 4), &c2.advance(0, 4))
        + satd_4x4_sse2(&c1.advance(4, 4), &c2.advance(4, 4))
}

simd_entry! {
    #[target_feature(enable = "sse2")]
    fn satd_4x4_sse2_impl<A: RefSamples + Copy, B: RefSamples + Copy>(
        c1: &A,
        c2: &B,
        _token: Sse2Token,
    ) -> i32 {
        satd_4x4_sse2(c1, c2)
    }

    #[target_feature(enable = "sse2")]
    fn satd_8x4_sse2_impl<A: RefSamples + Copy, B: RefSamples + Copy>(
        c1: &A,
        c2: &B,
        _token: Sse2Token,
    ) -> i32 {
        satd_4x4_sse2(c1, c2) + satd_4x4_sse2(&c1.advance(4, 0), &c2.advance(4, 0))
    }

    #[target_feature(enable = "sse2")]
    fn satd_4x8_sse2_impl<A: RefSamples + Copy, B: RefSamples + Copy>(
        c1: &A,
        c2: &B,
        _token: Sse2Token,
    ) -> i32 {
        satd_4x4_sse2(c1, c2) + satd_4x4_sse2(&c1.advance(0, 4), &c2.advance(0, 4))
    }

    #[target_feature(enable = "sse2")]
    fn satd_8x8_sse2_impl<A: RefSamples + Copy, B: RefSamples + Copy>(
        c1: &A,
        c2: &B,
        _token: Sse2Token,
    ) -> i32 {
        satd_8x8_sse2(c1, c2)
    }

    #[target_feature(enable = "sse2")]
    fn satd_16x8_sse2_impl<A: RefSamples + Copy, B: RefSamples + Copy>(
        c1: &A,
        c2: &B,
        _token: Sse2Token,
    ) -> i32 {
        satd_8x8_sse2(c1, c2) + satd_8x8_sse2(&c1.advance(8, 0), &c2.advance(8, 0))
    }

    #[target_feature(enable = "sse2")]
    fn satd_8x16_sse2_impl<A: RefSamples + Copy, B: RefSamples + Copy>(
        c1: &A,
        c2: &B,
        _token: Sse2Token,
    ) -> i32 {
        satd_8x8_sse2(c1, c2) + satd_8x8_sse2(&c1.advance(0, 8), &c2.advance(0, 8))
    }

    #[target_feature(enable = "sse2")]
    fn satd_16x16_sse2_impl<A: RefSamples + Copy, B: RefSamples + Copy>(
        c1: &A,
        c2: &B,
        _token: Sse2Token,
    ) -> i32 {
        satd_8x8_sse2(c1, c2)
            + satd_8x8_sse2(&c1.advance(8, 0), &c2.advance(8, 0))
            + satd_8x8_sse2(&c1.advance(0, 8), &c2.advance(0, 8))
            + satd_8x8_sse2(&c1.advance(8, 8), &c2.advance(8, 8))
    }

    #[target_feature(enable = "avx2")]
    fn satd_16x16_avx2_impl<A: RefSamples + Copy, B: RefSamples + Copy>(
        c1: &A,
        c2: &B,
        _token: Avx2Token,
    ) -> i32 {
        let (s1, s2) = (c1.span::<16, 16>(0, 0), c2.span::<16, 16>(0, 0));
        let (qs1, st1) = s1.all_4_quads::<16>();
        let (qs2, st2) = s2.all_4_quads::<16>();
        let hsum_const = ld32_i8(&HSUM_SUB_DB1_256);
        let (a0, a1, a2, a3) = A::Span::read_quad::<16>(qs1[0], st1);
        let (c0, c1, c2, c3) = B::Span::read_quad::<16>(qs2[0], st2);
        let mut acc = satd_16x4_step!(a0, c0, a1, c1, a2, c2, a3, c3, hsum_const);
        for (q1, q2) in qs1[1..].iter().copied().zip(qs2[1..].iter().copied()) {
            let (a0, a1, a2, a3) = A::Span::read_quad::<16>(q1, st1);
            let (c0, c1, c2, c3) = B::Span::read_quad::<16>(q2, st2);
            acc = _mm256_add_epi16(
                acc,
                satd_16x4_step!(a0, c0, a1, c1, a2, c2, a3, c3, hsum_const),
            );
        }
        sum_w_horizon_avx2(acc)
    }

    #[target_feature(enable = "avx2")]
    fn satd_16x8_avx2_impl<A: RefSamples + Copy, B: RefSamples + Copy>(
        c1: &A,
        c2: &B,
        _token: Avx2Token,
    ) -> i32 {
        let (s1, s2) = (c1.span::<16, 8>(0, 0), c2.span::<16, 8>(0, 0));
        let ([q1_0, q1_1], st1) = s1.all_2_quads::<16>();
        let ([q2_0, q2_1], st2) = s2.all_2_quads::<16>();
        let hsum_const = ld32_i8(&HSUM_SUB_DB1_256);
        let (a0, a1, a2, a3) = A::Span::read_quad::<16>(q1_0, st1);
        let (c0, c1, c2, c3) = B::Span::read_quad::<16>(q2_0, st2);
        let mut acc = satd_16x4_step!(a0, c0, a1, c1, a2, c2, a3, c3, hsum_const);
        let (a4, a5, a6, a7) = A::Span::read_quad::<16>(q1_1, st1);
        let (c4, c5, c6, c7) = B::Span::read_quad::<16>(q2_1, st2);
        acc = _mm256_add_epi16(
            acc,
            satd_16x4_step!(a4, c4, a5, c5, a6, c6, a7, c7, hsum_const),
        );
        sum_w_horizon_avx2(acc)
    }

    #[target_feature(enable = "avx2")]
    fn satd_8x8_avx2_impl<A: RefSamples + Copy, B: RefSamples + Copy>(
        c1: &A,
        c2: &B,
        _token: Avx2Token,
    ) -> i32 {
        let (s1, s2) = (c1.span::<8, 8>(0, 0), c2.span::<8, 8>(0, 0));
        let ([q1_0, q1_1], st1) = s1.all_2_quads::<8>();
        let ([q2_0, q2_1], st2) = s2.all_2_quads::<8>();
        let (r1_0, r1_1, r1_2, r1_3) = A::Span::read_quad::<8>(q1_0, st1);
        let (r1_4, r1_5, r1_6, r1_7) = A::Span::read_quad::<8>(q1_1, st1);
        let (r2_0, r2_1, r2_2, r2_3) = B::Span::read_quad::<8>(q2_0, st2);
        let (r2_4, r2_5, r2_6, r2_7) = B::Span::read_quad::<8>(q2_1, st2);
        let hsum_const = ld32_i8(&HSUM_SUB_DB1_128X2);
        let sum = satd_8x8_step!(
            r1_0, r1_1, r1_2, r1_3, r1_4, r1_5, r1_6, r1_7,
            r2_0, r2_1, r2_2, r2_3, r2_4, r2_5, r2_6, r2_7,
            hsum_const
        );
        sum_w_horizon_avx2(sum)
    }

    #[target_feature(enable = "avx2")]
    fn satd_8x16_avx2_impl<A: RefSamples + Copy, B: RefSamples + Copy>(
        c1: &A,
        c2: &B,
        _token: Avx2Token,
    ) -> i32 {
        let (s1, s2) = (c1.span::<8, 16>(0, 0), c2.span::<8, 16>(0, 0));
        let ([q1_0, q1_1, q1_2, q1_3], st1) = s1.all_4_quads::<8>();
        let ([q2_0, q2_1, q2_2, q2_3], st2) = s2.all_4_quads::<8>();
        let hsum_const = ld32_i8(&HSUM_SUB_DB1_128X2);
        let (a0, a1, a2, a3) = A::Span::read_quad::<8>(q1_0, st1);
        let (a4, a5, a6, a7) = A::Span::read_quad::<8>(q1_1, st1);
        let (c0, c1, c2, c3) = B::Span::read_quad::<8>(q2_0, st2);
        let (c4, c5, c6, c7) = B::Span::read_quad::<8>(q2_1, st2);
        let top = satd_8x8_step!(
            a0, a1, a2, a3, a4, a5, a6, a7,
            c0, c1, c2, c3, c4, c5, c6, c7,
            hsum_const
        );
        let (a8, a9, a10, a11) = A::Span::read_quad::<8>(q1_2, st1);
        let (a12, a13, a14, a15) = A::Span::read_quad::<8>(q1_3, st1);
        let (c8, c9, c10, c11) = B::Span::read_quad::<8>(q2_2, st2);
        let (c12, c13, c14, c15) = B::Span::read_quad::<8>(q2_3, st2);
        let bot = satd_8x8_step!(
            a8, a9, a10, a11, a12, a13, a14, a15,
            c8, c9, c10, c11, c12, c13, c14, c15,
            hsum_const
        );
        sum_w_horizon_avx2(_mm256_add_epi16(top, bot))
    }
}

// ============================================================================
// Safe Public Wrappers
// ============================================================================

#[inline(always)]
pub fn satd_4x4<A: RefSamples + Copy, B: RefSamples + Copy>(c1: &A, c2: &B) -> i32 {
    satd_4x4_sse2_impl(c1, c2, sse2_token())
}

#[inline(always)]
pub fn satd_8x4<A: RefSamples + Copy, B: RefSamples + Copy>(c1: &A, c2: &B) -> i32 {
    satd_8x4_sse2_impl(c1, c2, sse2_token())
}

#[inline(always)]
pub fn satd_4x8<A: RefSamples + Copy, B: RefSamples + Copy>(c1: &A, c2: &B) -> i32 {
    satd_4x8_sse2_impl(c1, c2, sse2_token())
}

#[inline(always)]
pub fn satd_8x8<A: RefSamples + Copy, B: RefSamples + Copy>(c1: &A, c2: &B) -> i32 {
    #[cold]
    #[inline(never)]
    fn fallback<A: RefSamples + Copy, B: RefSamples + Copy>(c1: &A, c2: &B) -> i32 {
        if let Some(token) = Avx2Token::try_new_enabled() {
            satd_8x8_avx2_impl(c1, c2, token)
        } else {
            satd_8x8_sse2_impl(c1, c2, sse2_token())
        }
    }
    if let Some(token) = Avx2Token::try_fast_enabled() {
        return satd_8x8_avx2_impl(c1, c2, token);
    }
    fallback(c1, c2)
}

#[inline(always)]
pub fn satd_16x8<A: RefSamples + Copy, B: RefSamples + Copy>(c1: &A, c2: &B) -> i32 {
    #[cold]
    #[inline(never)]
    fn fallback<A: RefSamples + Copy, B: RefSamples + Copy>(c1: &A, c2: &B) -> i32 {
        if let Some(token) = Avx2Token::try_new_enabled() {
            satd_16x8_avx2_impl(c1, c2, token)
        } else {
            satd_16x8_sse2_impl(c1, c2, sse2_token())
        }
    }
    if let Some(token) = Avx2Token::try_fast_enabled() {
        return satd_16x8_avx2_impl(c1, c2, token);
    }
    fallback(c1, c2)
}

#[inline(always)]
pub fn satd_8x16<A: RefSamples + Copy, B: RefSamples + Copy>(c1: &A, c2: &B) -> i32 {
    #[cold]
    #[inline(never)]
    fn fallback<A: RefSamples + Copy, B: RefSamples + Copy>(c1: &A, c2: &B) -> i32 {
        if let Some(token) = Avx2Token::try_new_enabled() {
            satd_8x16_avx2_impl(c1, c2, token)
        } else {
            satd_8x16_sse2_impl(c1, c2, sse2_token())
        }
    }
    if let Some(token) = Avx2Token::try_fast_enabled() {
        return satd_8x16_avx2_impl(c1, c2, token);
    }
    fallback(c1, c2)
}

#[inline(always)]
pub fn satd_16x16<A: RefSamples + Copy, B: RefSamples + Copy>(c1: &A, c2: &B) -> i32 {
    #[cold]
    #[inline(never)]
    fn fallback<A: RefSamples + Copy, B: RefSamples + Copy>(c1: &A, c2: &B) -> i32 {
        if let Some(token) = Avx2Token::try_new_enabled() {
            satd_16x16_avx2_impl(c1, c2, token)
        } else {
            satd_16x16_sse2_impl(c1, c2, sse2_token())
        }
    }
    if let Some(token) = Avx2Token::try_fast_enabled() {
        return satd_16x16_avx2_impl(c1, c2, token);
    }
    fallback(c1, c2)
}

// ============================================================================
// Unit Tests: Differential Parity Against Scalar Kernels
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::sample as scalar_satd;
    use crate::safe::plane::PlaneCursor;

    fn make_test_planes(stride: usize, height: usize) -> (Vec<u8>, Vec<u8>) {
        let mut p1 = vec![0u8; stride * height];
        let mut p2 = vec![0u8; stride * height];
        for (i, b) in p1.iter_mut().enumerate() {
            *b = ((i * 19 + 7) & 0xFF) as u8;
        }
        for (i, b) in p2.iter_mut().enumerate() {
            *b = ((i * 29 + 23) & 0xFF) as u8;
        }
        (p1, p2)
    }

    #[test]
    fn test_satd_parity_identical() {
        let buf = [42u8; 64];
        let c = PlaneCursor::new(&buf, 0, 8);
        assert_eq!(satd_4x4(&c, &c), 0);
        assert_eq!(satd_8x8(&c, &c), 0);
        assert_eq!(satd_4x4(&c, &c), scalar_satd::satd_4x4(&c, &c));
    }

    #[test]
    fn test_satd_parity_all_shapes() {
        let (p1, p2) = make_test_planes(64, 64);
        let c1 = PlaneCursor::new(&p1, 64 * 8 + 8, 64);
        let c2 = PlaneCursor::new(&p2, 64 * 8 + 8, 64);

        assert_eq!(
            satd_4x4(&c1, &c2),
            scalar_satd::satd_4x4(&c1, &c2),
            "satd_4x4 mismatch"
        );
        assert_eq!(
            satd_8x4(&c1, &c2),
            scalar_satd::satd_8x4(&c1, &c2),
            "satd_8x4 mismatch"
        );
        assert_eq!(
            satd_4x8(&c1, &c2),
            scalar_satd::satd_4x8(&c1, &c2),
            "satd_4x8 mismatch"
        );
        assert_eq!(
            satd_8x8(&c1, &c2),
            scalar_satd::satd_8x8(&c1, &c2),
            "satd_8x8 mismatch"
        );
        assert_eq!(
            satd_16x8(&c1, &c2),
            scalar_satd::satd_16x8(&c1, &c2),
            "satd_16x8 mismatch"
        );
        assert_eq!(
            satd_8x16(&c1, &c2),
            scalar_satd::satd_8x16(&c1, &c2),
            "satd_8x16 mismatch"
        );
        assert_eq!(
            satd_16x16(&c1, &c2),
            scalar_satd::satd_16x16(&c1, &c2),
            "satd_16x16 mismatch"
        );

        if let Some(token) = Avx2Token::try_new() {
            assert_eq!(
                satd_8x8_avx2_impl(&c1, &c2, token),
                scalar_satd::satd_8x8(&c1, &c2),
                "satd_8x8_avx2 mismatch"
            );
            assert_eq!(
                satd_16x8_avx2_impl(&c1, &c2, token),
                scalar_satd::satd_16x8(&c1, &c2),
                "satd_16x8_avx2 mismatch"
            );
            assert_eq!(
                satd_8x8_avx2_impl(&c1, &c2, token)
                    + satd_8x8_avx2_impl(&c1.advance(0, 8), &c2.advance(0, 8), token),
                scalar_satd::satd_8x16(&c1, &c2),
                "satd_8x16_avx2 mismatch"
            );
            assert_eq!(
                satd_16x16_avx2_impl(&c1, &c2, token),
                scalar_satd::satd_16x16(&c1, &c2),
                "satd_16x16_avx2 mismatch"
            );
        }
    }

    #[test]
    fn test_satd_random_blocks() {
        for seed in 0..100 {
            let mut p1 = [0u8; 16];
            let mut p2 = [0u8; 16];
            for i in 0..16 {
                p1[i] = ((seed * 37 + i * 11) & 0xFF) as u8;
                p2[i] = ((seed * 43 + i * 17) & 0xFF) as u8;
            }
            let c1 = PlaneCursor::new(&p1, 0, 4);
            let c2 = PlaneCursor::new(&p2, 0, 4);
            assert_eq!(
                satd_4x4(&c1, &c2),
                scalar_satd::satd_4x4(&c1, &c2),
                "mismatch at seed {seed}"
            );
        }
    }
}
