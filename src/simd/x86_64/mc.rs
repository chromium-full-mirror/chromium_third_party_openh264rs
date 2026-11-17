// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! SSE2 / SSSE3 implementations of Motion Compensation (MC) kernels:
//! - Pixel averaging (`pixel_avg`)
//! - Chroma motion compensation (`mc_chroma`)
//! - Horizontal 6-tap Wiener filter (`mc_hor_ver20`)
//! - Vertical 6-tap Wiener filter (`mc_hor_ver02`)
//! - 2D center 6x6-tap Wiener filter (`mc_hor_ver22`)
//! - Luma quarter-pel motion compensation (`mc_luma`)
#![deny(unsafe_code)]

use super::lanes::{ld4, ld8, ld16, st4, st8, st8_i16, st16};
use super::{sse2_token, ssse3_token};
use crate::common::mc::mc_luma_with;
use crate::common::mc::{
    McLeaves, WelsClip1, avg_shaped, cen_shaped, chroma_shaped, filter_input_8bit, g_kuiABCD,
    hor_filter_input_16bit, hor_shaped, mc_copy, ver_shaped,
};
use crate::safe::plane::{BlockRows, PlaneCursorMut, RefSamples, RowSampleView};
use crate::simd::{
    simd_entry,
    tokens::{Avx2Token, Sse2Token, Ssse3Token},
};
use core::arch::x86_64::*;

// ============================================================================
// Block shapes and lane moves
// ============================================================================

/// Eight bytes of a window row widened to eight words.
#[target_feature(enable = "sse2")]
#[inline]
fn w8<R: BlockRows>(r: &R, y: usize, x: usize) -> __m128i {
    _mm_unpacklo_epi8(ld8(&r.row::<8>(y, x)), _mm_setzero_si128())
}

/// Four bytes of a window row widened to four words in the low half.
#[target_feature(enable = "sse2")]
#[inline]
fn w4<R: BlockRows>(r: &R, y: usize, x: usize) -> __m128i {
    _mm_unpacklo_epi8(ld4(&r.row::<4>(y, x)), _mm_setzero_si128())
}

// ============================================================================
// Pixel Averaging (SSE2)
// ============================================================================

/// Rounded pixel average of two rows: `((a + b + 1) >> 1) as u8`, `pavgb`.
///
/// `W` is a const parameter so the width dispatch below resolves at compile time.
#[target_feature(enable = "sse2")]
#[inline]
fn avg_row<const W: usize>(out: &mut [u8; W], a: &[u8; W], b: &[u8; W]) {
    if W == 16 {
        st16(out, _mm_avg_epu8(ld16(a), ld16(b)));
    } else if W == 8 {
        st8(out, _mm_avg_epu8(ld8(a), ld8(b)));
    } else if W == 4 {
        st4(out, _mm_avg_epu8(ld4(a), ld4(b)));
    } else {
        for j in 0..W {
            out[j] = (((a[j] as u32) + (b[j] as u32) + 1) >> 1) as u8;
        }
    }
}

#[cold]
#[inline(never)]
fn avg_block_narrow_stride<A: RefSamples, B: RefSamples, const W: usize, const H: usize>(
    dst: &mut PlaneCursorMut<'_>,
    a: &A,
    b: &B,
) {
    let sa = a.span::<W, H>(0, 0);
    let sb = b.span::<W, H>(0, 0);
    let mut d = dst.span_mut::<W, H>(0, 0);
    for y in 0..H {
        let (ra, rb) = (sa.row::<W>(y, 0), sb.row::<W>(y, 0));
        let out = d.row_mut::<W>(y, 0);
        for j in 0..W {
            out[j] = (((ra[j] as u32) + (rb[j] as u32) + 1) >> 1) as u8;
        }
    }
}

simd_entry! {
    #[target_feature(enable = "sse2")]
    fn avg_block_sse2<A: RefSamples, B: RefSamples, const W: usize, const H: usize>(
        dst: &mut PlaneCursorMut<'_>,
        a: &A,
        b: &B,
        _token: Sse2Token,
    ) {
        let sa = a.span::<W, H>(0, 0);
        let sb = b.span::<W, H>(0, 0);
        let mut d = dst.span_mut::<W, H>(0, 0);
        if let (Some((mut ah, at)), Some((mut bh, bt)), Some((dh, dt))) = (
            sa.split_rows::<W, H>(),
            sb.split_rows::<W, H>(),
            d.split_rows_mut::<W, H>(),
        ) {
            for out in dh {
                let (Some(ra), Some(rb)) = (ah.next(), bh.next()) else {
                    break;
                };
                avg_row::<W>(out, &ra, &rb);
            }
            avg_row::<W>(dt, &at(), &bt());
        } else {
            avg_block_narrow_stride::<A, B, W, H>(dst, a, b);
        }
    }
}

#[inline(always)]
fn avg_block<A: RefSamples, B: RefSamples, const W: usize, const H: usize>(
    dst: &mut PlaneCursorMut<'_>,
    a: &A,
    b: &B,
) {
    avg_block_sse2::<A, B, W, H>(dst, a, b, sse2_token());
}

/// The run-time-shape twin — cold; see [`McLeaves`].
fn avg_any<A: RefSamples, B: RefSamples>(
    dst: &mut PlaneCursorMut<'_>,
    a: &A,
    b: &B,
    width: usize,
    height: usize,
) {
    for dy in 0..height as isize {
        let out = dst.row_mut(dy, 0, width);
        for (j, o) in out.iter_mut().enumerate() {
            *o = (((a.at(j as isize, dy) as u32) + (b.at(j as isize, dy) as u32) + 1) >> 1) as u8;
        }
    }
}

/// Public safe entry point for SSE2 pixel averaging.
#[inline(always)]
pub fn pixel_avg<A: RefSamples, B: RefSamples>(
    dst: &mut PlaneCursorMut<'_>,
    a: &A,
    b: &B,
    width: usize,
    height: usize,
) {
    avg_shaped::<Sse2Leaves, A, B>(dst, a, b, width, height)
}

// ============================================================================
// Chroma MC (SSE2 / SSSE3)
// ============================================================================

/// One output row of the bilinear filter at width 8 or 4, over the two one-row
/// windows `r0` and `r1`.
#[allow(dead_code)]
#[target_feature(enable = "sse2")]
fn chroma_row<R: BlockRows, const W: usize>(
    out: &mut [u8; W],
    r0: &R,
    r1: &R,
    vA: __m128i,
    vB: __m128i,
    vC: __m128i,
    vD: __m128i,
) {
    let (p00, p01, p10, p11) = if W == 8 {
        (w8(r0, 0, 0), w8(r0, 0, 1), w8(r1, 0, 0), w8(r1, 0, 1))
    } else {
        (w4(r0, 0, 0), w4(r0, 0, 1), w4(r1, 0, 0), w4(r1, 0, 1))
    };
    let sum = _mm_add_epi16(
        _mm_add_epi16(_mm_mullo_epi16(p00, vA), _mm_mullo_epi16(p01, vB)),
        _mm_add_epi16(_mm_mullo_epi16(p10, vC), _mm_mullo_epi16(p11, vD)),
    );
    let shifted = _mm_srli_epi16(_mm_add_epi16(sum, _mm_set1_epi16(32)), 6);
    let packed = _mm_packus_epi16(shifted, _mm_setzero_si128());
    if W == 8 {
        st8(out, packed);
    } else {
        st4(out, packed);
    }
}

#[cold]
#[inline(never)]
fn chroma_narrow<
    S: RefSamples + Copy,
    const W: usize,
    const SW: usize,
    const H: usize,
    const SH: usize,
>(
    src: &S,
    dst: &mut PlaneCursorMut<'_>,
    w: &[u8; 4],
) {
    let s = src.span::<SW, SH>(0, 0);
    let mut d = dst.span_mut::<W, H>(0, 0);
    let (iA, iB, iC, iD) = (w[0] as i32, w[1] as i32, w[2] as i32, w[3] as i32);
    for y in 0..H {
        let (r0, r1) = (s.row::<SW>(y, 0), s.row::<SW>(y + 1, 0));
        let out = d.row_mut::<W>(y, 0);
        for j in 0..W {
            out[j] = ((iA * (r0[j] as i32)
                + iB * (r0[j + 1] as i32)
                + iC * (r1[j] as i32)
                + iD * (r1[j + 1] as i32)
                + 32)
                >> 6) as u8;
        }
    }
}

simd_entry! {
    /// The bilinear chroma filter over one const-shape block. Widths 8 and 4 take the
    /// lane path; width 2 is the scalar.
    #[target_feature(enable = "ssse3")]
    fn chroma_block<
        S: RefSamples + Copy,
        const W: usize,
        const SW: usize,
        const H: usize,
        const SH: usize,
    >(
        src: &S,
        dst: &mut PlaneCursorMut<'_>,
        w: &[u8; 4],
        _token: Ssse3Token,
    ) {
        let s = src.span::<SW, SH>(0, 0);
        let mut d = dst.span_mut::<W, H>(0, 0);
        let (Some((mut sh, st)), Some((dh, dt))) =
            (s.split_views::<SW, SH>(), d.split_rows_mut::<W, H>())
        else {
            chroma_narrow::<S, W, SW, H, SH>(src, dst, w);
            return;
        };
        if W == 8 {
            let coeff_ab = _mm_set1_epi16(((w[1] as i16) << 8) | (w[0] as i16));
            let coeff_cd = _mm_set1_epi16(((w[3] as i16) << 8) | (w[2] as i16));
            let round_32 = _mm_set1_epi16(32);

            let interleave8 = |row: <S::Span<'_> as BlockRows>::View<'_, SW>| -> __m128i {
                let r0 = ld8(&row.sub::<8>(0));
                let r1 = ld8(&row.sub::<8>(1));
                _mm_unpacklo_epi8(r0, r1)
            };

            let mut curr = interleave8(sh.next().expect("chroma row 0"));
            for out in dh {
                let next = interleave8(sh.next().expect("chroma head row"));
                let sum = _mm_add_epi16(
                    _mm_maddubs_epi16(curr, coeff_ab),
                    _mm_maddubs_epi16(next, coeff_cd),
                );
                let shifted = _mm_srli_epi16(_mm_add_epi16(sum, round_32), 6);
                let packed = _mm_packus_epi16(shifted, shifted);
                st8(out, packed);
                curr = next;
            }
            let next = interleave8(st);
            let sum = _mm_add_epi16(
                _mm_maddubs_epi16(curr, coeff_ab),
                _mm_maddubs_epi16(next, coeff_cd),
            );
            let shifted = _mm_srli_epi16(_mm_add_epi16(sum, round_32), 6);
            let packed = _mm_packus_epi16(shifted, shifted);
            st8(dt, packed);
        } else if W == 4 {
            let coeff_ab = _mm_set1_epi16(((w[1] as i16) << 8) | (w[0] as i16));
            let coeff_cd = _mm_set1_epi16(((w[3] as i16) << 8) | (w[2] as i16));
            let round_32 = _mm_set1_epi16(32);

            let interleave4 = |row: <S::Span<'_> as BlockRows>::View<'_, SW>| -> __m128i {
                let r0 = ld4(&row.sub::<4>(0));
                let r1 = ld4(&row.sub::<4>(1));
                _mm_unpacklo_epi8(r0, r1)
            };

            let mut curr = interleave4(sh.next().expect("chroma row 0"));
            for out in dh {
                let next = interleave4(sh.next().expect("chroma head row"));
                let sum = _mm_add_epi16(
                    _mm_maddubs_epi16(curr, coeff_ab),
                    _mm_maddubs_epi16(next, coeff_cd),
                );
                let shifted = _mm_srli_epi16(_mm_add_epi16(sum, round_32), 6);
                let packed = _mm_packus_epi16(shifted, shifted);
                st4(out, packed);
                curr = next;
            }
            let next = interleave4(st);
            let sum = _mm_add_epi16(
                _mm_maddubs_epi16(curr, coeff_ab),
                _mm_maddubs_epi16(next, coeff_cd),
            );
            let shifted = _mm_srli_epi16(_mm_add_epi16(sum, round_32), 6);
            let packed = _mm_packus_epi16(shifted, shifted);
            st4(dt, packed);
        } else {
            let (iA, iB, iC, iD) = (w[0] as i32, w[1] as i32, w[2] as i32, w[3] as i32);
            let mut r0 = sh.next().expect("chroma row 0");
            for out in dh {
                let r1 = sh.next().expect("chroma head row");
                for (j, o) in out.iter_mut().enumerate() {
                    *o = ((iA * (r0.sample(j) as i32)
                        + iB * (r0.sample(j + 1) as i32)
                        + iC * (r1.sample(j) as i32)
                        + iD * (r1.sample(j + 1) as i32)
                        + 32)
                        >> 6) as u8;
                }
                r0 = r1;
            }
            let r1 = st;
            for (j, o) in dt.iter_mut().enumerate() {
                *o = ((iA * (r0.sample(j) as i32)
                    + iB * (r0.sample(j + 1) as i32)
                    + iC * (r1.sample(j) as i32)
                    + iD * (r1.sample(j + 1) as i32)
                    + 32)
                    >> 6) as u8;
            }
        }
    }
}

/// The run-time-shape twin — cold; see [`McLeaves`].
fn chroma_any<S: RefSamples + Copy>(
    src: &S,
    dst: &mut PlaneCursorMut<'_>,
    w: &[u8; 4],
    width: usize,
    height: usize,
) {
    let (iA, iB, iC, iD) = (w[0] as i32, w[1] as i32, w[2] as i32, w[3] as i32);
    for dy in 0..height as isize {
        let out = dst.row_mut(dy, 0, width);
        for (j, o) in out.iter_mut().enumerate() {
            let x = j as isize;
            *o = ((iA * (src.at(x, dy) as i32)
                + iB * (src.at(x + 1, dy) as i32)
                + iC * (src.at(x, dy + 1) as i32)
                + iD * (src.at(x + 1, dy + 1) as i32)
                + 32)
                >> 6) as u8;
        }
    }
}

/// SIMD block copy for whole-pixel motion vectors.
#[inline(always)]
fn copy_block_simd<const W: usize, const H: usize, S: RefSamples + Copy>(
    src: &S,
    dst: &mut PlaneCursorMut<'_>,
) {
    #[cold]
    #[inline(never)]
    fn narrow<const W: usize, const H: usize, S: RefSamples + Copy>(
        src: &S,
        dst: &mut PlaneCursorMut<'_>,
    ) {
        let s = src.span::<W, H>(0, 0);
        let mut d = dst.span_mut::<W, H>(0, 0);
        for y in 0..H {
            *d.row_mut::<W>(y, 0) = s.row::<W>(y, 0);
        }
    }
    let s = src.span::<W, H>(0, 0);
    let mut d = dst.span_mut::<W, H>(0, 0);
    if let (Some((mut sh, st)), Some((mut dh, dt))) =
        (s.split_rows::<W, H>(), d.split_rows_mut::<W, H>())
    {
        for _ in 0..(H - 1) {
            *dh.next().expect("mc copy dst row") = sh.next().expect("mc copy src row");
        }
        *dt = st();
    } else {
        narrow::<W, H, S>(src, dst);
    }
}

/// Public safe entry point for SSE2 chroma MC.
#[inline(always)]
pub fn mc_chroma<S: RefSamples + Copy>(
    src: &S,
    dst: &mut PlaneCursorMut<'_>,
    mv_x: i16,
    mv_y: i16,
    width: usize,
    height: usize,
) {
    if (mv_x & 0x07) == 0 && (mv_y & 0x07) == 0 {
        match (width, height) {
            (8, 8) => copy_block_simd::<8, 8, S>(src, dst),
            (8, 4) => copy_block_simd::<8, 4, S>(src, dst),
            (4, 8) => copy_block_simd::<4, 8, S>(src, dst),
            (4, 4) => copy_block_simd::<4, 4, S>(src, dst),
            (2, 2) => copy_block_simd::<2, 2, S>(src, dst),
            _ => mc_copy(src, dst, width, height),
        }
        return;
    }
    mc_chroma_frac(src, dst, mv_x, mv_y, width, height)
}

/// The fractional half of [`mc_chroma`], out of line so the entry point stays small
/// enough to inline. The whole-sample vector is the common chroma case and a block copy;
/// with the bilinear dispatch in the same body, `mc_copy` lost its constant width and
/// height at that call site.
#[inline(always)]
fn mc_chroma_frac<S: RefSamples + Copy>(
    src: &S,
    dst: &mut PlaneCursorMut<'_>,
    mv_x: i16,
    mv_y: i16,
    width: usize,
    height: usize,
) {
    if width == 0 {
        return;
    }
    let w = &g_kuiABCD[(mv_y & 0x07) as usize][(mv_x & 0x07) as usize];
    chroma_shaped::<Sse2Leaves, S>(src, dst, w, width, height)
}

// ============================================================================
// 6-Tap Filter Helpers (SSE2)
// ============================================================================

/// Vectorized 6-tap Wiener filter on 8 samples:
/// `val = (p0 + p5) - 5 * (p1 + p4) + 20 * (p2 + p3)`
/// `res = WelsClip1((val + 16) >> 5)`
///
/// Implemented using the identity:
/// `x = 4 * (p2 + p3) - (p1 + p4)`
/// `val = (p0 + p5) + x + (x << 2)`
#[allow(dead_code)]
#[target_feature(enable = "sse2")]
fn filter_6tap_8_samples(
    p0: __m128i,
    p1: __m128i,
    p2: __m128i,
    p3: __m128i,
    p4: __m128i,
    p5: __m128i,
) -> __m128i {
    let p14 = _mm_add_epi16(p1, p4);
    let p23 = _mm_add_epi16(p2, p3);
    let x = _mm_sub_epi16(_mm_slli_epi16(p23, 2), p14);
    let p05 = _mm_add_epi16(p0, p5);
    let sum = _mm_add_epi16(p05, _mm_add_epi16(x, _mm_slli_epi16(x, 2)));
    let rounded = _mm_add_epi16(sum, _mm_set1_epi16(16));
    let shifted = _mm_srai_epi16(rounded, 5);
    _mm_packus_epi16(shifted, _mm_setzero_si128())
}

/// Computes the unclipped 16-bit intermediate for 2D filter:
/// `val = (p0 + p5) - 5 * (p1 + p4) + 20 * (p2 + p3)`
#[target_feature(enable = "sse2")]
#[inline]
fn filter_6tap_intermediate_8_samples(
    p0: __m128i,
    p1: __m128i,
    p2: __m128i,
    p3: __m128i,
    p4: __m128i,
    p5: __m128i,
) -> __m128i {
    let p14 = _mm_add_epi16(p1, p4);
    let p23 = _mm_add_epi16(p2, p3);
    let x = _mm_sub_epi16(_mm_slli_epi16(p23, 2), p14);
    let p05 = _mm_add_epi16(p0, p5);
    _mm_add_epi16(p05, _mm_add_epi16(x, _mm_slli_epi16(x, 2)))
}

// ============================================================================
// Horizontal 6-Tap Filter: McHorVer20 (SSSE3)
// ============================================================================

/// Vectorized 6-tap Wiener filter on 8 samples using SSSE3 pmaddubsw and pshufb.
#[target_feature(enable = "ssse3")]
#[inline]
fn filter_6tap_8px(raw: __m128i) -> __m128i {
    let mask_01 = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8);
    let mask_23 = _mm_setr_epi8(2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10);
    let mask_45 = _mm_setr_epi8(4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12);

    let coeff_01 = _mm_setr_epi8(1, -5, 1, -5, 1, -5, 1, -5, 1, -5, 1, -5, 1, -5, 1, -5);
    let coeff_23 = _mm_set1_epi8(20);
    let coeff_45 = _mm_setr_epi8(-5, 1, -5, 1, -5, 1, -5, 1, -5, 1, -5, 1, -5, 1, -5, 1);

    let p01 = _mm_shuffle_epi8(raw, mask_01);
    let p23 = _mm_shuffle_epi8(raw, mask_23);
    let p45 = _mm_shuffle_epi8(raw, mask_45);

    let m01 = _mm_maddubs_epi16(p01, coeff_01);
    let m23 = _mm_maddubs_epi16(p23, coeff_23);
    let m45 = _mm_maddubs_epi16(p45, coeff_45);

    _mm_add_epi16(_mm_add_epi16(m01, m45), m23)
}

macro_rules! hor_row_view {
    ($out:expr, $sr:expr, $W:expr, $SW:expr, $AVG:expr) => {{
        let out: &mut [u8; $W] = $out;
        let sr = $sr;
        if $W >= 16 {
            let r0 = ld16(&sr.sub::<16>(0));
            let rt = ld8(&sr.sub::<8>($SW - 8));
            let r1 = if $SW == 22 {
                _mm_srli_si128::<2>(rt)
            } else {
                _mm_srli_si128::<3>(rt)
            };
            let r_hi = _mm_alignr_epi8(r1, r0, 8);
            let sum_lo = filter_6tap_8px(r0);
            let sum_hi = filter_6tap_8px(r_hi);
            let shifted_lo = _mm_srai_epi16(_mm_add_epi16(sum_lo, _mm_set1_epi16(16)), 5);
            let shifted_hi = _mm_srai_epi16(_mm_add_epi16(sum_hi, _mm_set1_epi16(16)), 5);
            let mut res16 = _mm_packus_epi16(shifted_lo, shifted_hi);
            if $AVG != 0 {
                let tap16 = ld16(&sr.sub::<16>($AVG));
                res16 = _mm_avg_epu8(res16, tap16);
            }
            st16(out, res16);
            if $W > 16 {
                let c = $W - 8;
                let raw_t = _mm_alignr_epi8(r1, r0, 9);
                let sum_t = filter_6tap_8px(raw_t);
                let shifted_t = _mm_srai_epi16(_mm_add_epi16(sum_t, _mm_set1_epi16(16)), 5);
                let mut res_t = _mm_packus_epi16(shifted_t, shifted_t);
                if $AVG != 0 {
                    let tap8 = ld8(&sr.sub::<8>(c + $AVG));
                    res_t = _mm_avg_epu8(res_t, tap8);
                }
                st8(&mut out[c..], res_t);
            }
        } else if $W >= 8 {
            let r0 = ld8(&sr.sub::<8>(0));
            let rt = ld8(&sr.sub::<8>($SW - 8));
            let r1 = if $SW == 14 {
                _mm_srli_si128::<2>(rt)
            } else {
                _mm_srli_si128::<3>(rt)
            };
            let raw0 = _mm_unpacklo_epi64(r0, r1);
            let sum0 = filter_6tap_8px(raw0);
            let shifted0 = _mm_srai_epi16(_mm_add_epi16(sum0, _mm_set1_epi16(16)), 5);
            let mut res8 = _mm_packus_epi16(shifted0, shifted0);
            if $AVG != 0 {
                let tap8 = ld8(&sr.sub::<8>($AVG));
                res8 = _mm_avg_epu8(res8, tap8);
            }
            st8(out, res8);
            if $W > 8 {
                let c = $W - 8;
                let raw_t = _mm_srli_si128::<1>(raw0);
                let sum_t = filter_6tap_8px(raw_t);
                let shifted_t = _mm_srai_epi16(_mm_add_epi16(sum_t, _mm_set1_epi16(16)), 5);
                let mut res_t = _mm_packus_epi16(shifted_t, shifted_t);
                if $AVG != 0 {
                    let tap8 = ld8(&sr.sub::<8>(c + $AVG));
                    res_t = _mm_avg_epu8(res_t, tap8);
                }
                st8(&mut out[c..], res_t);
            }
        } else if $W >= 4 {
            let r0 = ld8(&sr.sub::<8>(0));
            let rt = ld4(&sr.sub::<4>($SW - 4));
            let r1 = if $SW == 10 {
                _mm_srli_si128::<2>(rt)
            } else {
                _mm_srli_si128::<3>(rt)
            };
            let raw0 = _mm_unpacklo_epi64(r0, r1);
            let sum0 = filter_6tap_8px(raw0);
            let shifted0 = _mm_srai_epi16(_mm_add_epi16(sum0, _mm_set1_epi16(16)), 5);
            let mut res4 = _mm_packus_epi16(shifted0, shifted0);
            if $AVG != 0 {
                let tap4 = ld4(&sr.sub::<4>($AVG));
                res4 = _mm_avg_epu8(res4, tap4);
            }
            st4(out, res4);
            if $W > 4 {
                let c = $W - 4;
                let raw_t = _mm_srli_si128::<1>(raw0);
                let sum_t = filter_6tap_8px(raw_t);
                let shifted_t = _mm_srai_epi16(_mm_add_epi16(sum_t, _mm_set1_epi16(16)), 5);
                let mut res_t = _mm_packus_epi16(shifted_t, shifted_t);
                if $AVG != 0 {
                    let tap4 = ld4(&sr.sub::<4>(c + $AVG));
                    res_t = _mm_avg_epu8(res_t, tap4);
                }
                st4(&mut out[c..], res_t);
            }
        } else {
            for col in 0..$W {
                let t = sr.sub::<6>(col);
                let mut v = WelsClip1((filter_input_8bit(&t) + 16) >> 5);
                if $AVG != 0 {
                    v = ((v as u32 + sr.sample(col + $AVG) as u32 + 1) >> 1) as u8;
                }
                out[col] = v;
            }
        }
    }};
}

#[cold]
#[inline(never)]
fn hor_narrow<
    S: RefSamples + Copy,
    const W: usize,
    const SW: usize,
    const H: usize,
    const AVG: usize,
>(
    src: &S,
    dst: &mut PlaneCursorMut<'_>,
) {
    let s = src.span::<SW, H>(0, -2);
    let mut d = dst.span_mut::<W, H>(0, 0);
    for y in 0..H {
        let sr = s.row::<SW>(y, 0);
        let out = d.row_mut::<W>(y, 0);
        for col in 0..W {
            let t: [u8; 6] = sr[col..col + 6].try_into().unwrap();
            let mut v = WelsClip1((filter_input_8bit(&t) + 16) >> 5);
            if AVG != 0 {
                v = ((v as u32 + sr[col + AVG] as u32 + 1) >> 1) as u8;
            }
            out[col] = v;
        }
    }
}

simd_entry! {
    /// `McHorVer20` over one const-shape block: one span for the source, one for the
    /// destination, walked via `split_views` / `split_rows_mut` with zero per-row checks.
    #[target_feature(enable = "ssse3")]
    fn hor_block<
        S: RefSamples + Copy,
        const W: usize,
        const SW: usize,
        const H: usize,
        const AVG: usize,
    >(
        src: &S,
        dst: &mut PlaneCursorMut<'_>,
        _token: Ssse3Token,
    ) {
        let s = src.span::<SW, H>(0, -2);
        let mut d = dst.span_mut::<W, H>(0, 0);
        if let (Some((mut sh, st)), Some((dh, dt))) =
            (s.split_views::<SW, H>(), d.split_rows_mut::<W, H>())
        {
            for out in dh {
                let sr = sh.next().expect("hor head row");
                hor_row_view!(out, sr, W, SW, AVG);
            }
            hor_row_view!(dt, st, W, SW, AVG);
        } else {
            hor_narrow::<S, W, SW, H, AVG>(src, dst);
        }
    }
}

/// The run-time-shape twin — cold, and scalar; see [`McLeaves`].
fn hor_any<S: RefSamples + Copy, const AVG: usize>(
    src: &S,
    dst: &mut PlaneCursorMut<'_>,
    width: usize,
    height: usize,
) {
    for dy in 0..height as isize {
        let out = dst.row_mut(dy, 0, width);
        for (x, o) in out.iter_mut().enumerate() {
            let t: [u8; 6] = std::array::from_fn(|k| src.at(x as isize + k as isize - 2, dy));
            let mut v = WelsClip1((filter_input_8bit(&t) + 16) >> 5);
            if AVG != 0 {
                v = ((v as u32 + t[AVG] as u32 + 1) >> 1) as u8;
            }
            *o = v;
        }
    }
}

/// Public safe entry point for SSE2 horizontal half-pel filter.
pub fn mc_hor_ver20<S: RefSamples + Copy>(
    src: &S,
    dst: &mut PlaneCursorMut<'_>,
    width: usize,
    height: usize,
) {
    hor_shaped::<Sse2Leaves, S, 0>(src, dst, width, height)
}

// ============================================================================
// Vertical 6-Tap Filter: McHorVer02 (SSE2 / AVX2)
// ============================================================================

#[target_feature(enable = "sse2")]
#[inline]
fn filter_6tap_vertical_words(
    p0: __m128i,
    p1: __m128i,
    p2: __m128i,
    p3: __m128i,
    p4: __m128i,
    p5: __m128i,
) -> __m128i {
    let p14 = _mm_add_epi16(p1, p4);
    let p23 = _mm_add_epi16(p2, p3);
    let x = _mm_sub_epi16(_mm_slli_epi16(p23, 2), p14);
    let p05 = _mm_add_epi16(p0, p5);
    let sum = _mm_add_epi16(p05, _mm_add_epi16(x, _mm_slli_epi16(x, 2)));
    let rounded = _mm_add_epi16(sum, _mm_set1_epi16(16));
    _mm_srai_epi16(rounded, 5)
}

#[target_feature(enable = "avx2")]
#[inline]
fn filter_6tap_vertical_avx2(
    p0: __m256i,
    p1: __m256i,
    p2: __m256i,
    p3: __m256i,
    p4: __m256i,
    p5: __m256i,
) -> __m256i {
    let p14 = _mm256_add_epi16(p1, p4);
    let p23 = _mm256_add_epi16(p2, p3);
    let x = _mm256_sub_epi16(_mm256_slli_epi16(p23, 2), p14);
    let p05 = _mm256_add_epi16(p0, p5);
    let sum = _mm256_add_epi16(p05, _mm256_add_epi16(x, _mm256_slli_epi16(x, 2)));
    let rounded = _mm256_add_epi16(sum, _mm256_set1_epi16(16));
    _mm256_srai_epi16(rounded, 5)
}

#[cold]
#[inline(never)]
fn ver_narrow<
    S: RefSamples + Copy,
    const W: usize,
    const H: usize,
    const SH: usize,
    const AVG: usize,
>(
    src: &S,
    dst: &mut PlaneCursorMut<'_>,
) {
    let s = src.span::<W, SH>(-2, 0);
    let mut d = dst.span_mut::<W, H>(0, 0);
    for y in 0..H {
        let out = d.row_mut::<W>(y, 0);
        for (x, o) in out.iter_mut().enumerate() {
            let t: [u8; 6] = std::array::from_fn(|k| s.row::<1>(y + k, x)[0]);
            let mut v = WelsClip1((filter_input_8bit(&t) + 16) >> 5);
            if AVG != 0 {
                v = ((v as u32 + t[AVG] as u32 + 1) >> 1) as u8;
            }
            *o = v;
        }
    }
}

simd_entry! {
    #[target_feature(enable = "avx2")]
    fn ver_lanes_avx2_16x<
        S: RefSamples + Copy,
        const H: usize,
        const SH: usize,
        const AVG: usize,
    >(
        src: &S,
        dst: &mut PlaneCursorMut<'_>,
        _token: Avx2Token,
    ) {
        const { assert!(AVG == 0 || AVG == 2 || AVG == 3, "AVG must be 0, 2, or 3") };
        let s = src.span::<16, SH>(-2, 0);
        let mut d = dst.span_mut::<16, H>(0, 0);
        let (Some((mut sh, st)), Some((dh, dt))) =
            (s.split_rows::<16, SH>(), d.split_rows_mut::<16, H>())
        else {
            ver_narrow::<S, 16, H, SH, AVG>(src, dst);
            return;
        };

        let mut next_raw = || ld16(&sh.next().expect("ver_avx2 src row"));
        let (raw0, raw1, mut raw2, mut raw3, mut raw4) =
            (next_raw(), next_raw(), next_raw(), next_raw(), next_raw());
        let (mut r0, mut r1, mut r2, mut r3, mut r4) = (
            _mm256_cvtepu8_epi16(raw0),
            _mm256_cvtepu8_epi16(raw1),
            _mm256_cvtepu8_epi16(raw2),
            _mm256_cvtepu8_epi16(raw3),
            _mm256_cvtepu8_epi16(raw4),
        );

        for out in dh {
            let raw5 = next_raw();
            let r5 = _mm256_cvtepu8_epi16(raw5);
            let w = filter_6tap_vertical_avx2(r0, r1, r2, r3, r4, r5);
            let lo = _mm256_castsi256_si128(w);
            let hi = _mm256_extracti128_si256(w, 1);
            let mut both = _mm_packus_epi16(lo, hi);
            if AVG == 2 {
                both = _mm_avg_epu8(both, raw2);
            } else if AVG == 3 {
                both = _mm_avg_epu8(both, raw3);
            }
            st16(out, both);
            (r0, r1, r2, r3, r4) = (r1, r2, r3, r4, r5);
            if AVG != 0 {
                (raw2, raw3, raw4) = (raw3, raw4, raw5);
            }
        }
        {
            let raw5 = ld16(&st());
            let r5 = _mm256_cvtepu8_epi16(raw5);
            let w = filter_6tap_vertical_avx2(r0, r1, r2, r3, r4, r5);
            let lo = _mm256_castsi256_si128(w);
            let hi = _mm256_extracti128_si256(w, 1);
            let mut both = _mm_packus_epi16(lo, hi);
            if AVG == 2 {
                both = _mm_avg_epu8(both, raw2);
            } else if AVG == 3 {
                both = _mm_avg_epu8(both, raw3);
            }
            st16(dt, both);
        }
    }
}

#[target_feature(enable = "sse2")]
#[inline]
fn ver_load_row_view<const W: usize, V: RowSampleView<W>>(sr: V) -> [__m128i; 3] {
    let zero = _mm_setzero_si128();
    if W >= 16 {
        let raw0 = ld16(&sr.sub::<16>(0));
        let tail = if W > 16 {
            _mm_unpacklo_epi8(ld8(&sr.sub::<8>(W - 8)), zero)
        } else {
            zero
        };
        [
            _mm_unpacklo_epi8(raw0, zero),
            _mm_unpackhi_epi8(raw0, zero),
            tail,
        ]
    } else if W >= 8 {
        let raw0 = ld8(&sr.sub::<8>(0));
        let tail = if W > 8 {
            _mm_unpacklo_epi8(ld8(&sr.sub::<8>(W - 8)), zero)
        } else {
            zero
        };
        [_mm_unpacklo_epi8(raw0, zero), tail, zero]
    } else {
        let raw0 = ld4(&sr.sub::<4>(0));
        let tail = if W > 4 {
            _mm_unpacklo_epi8(ld4(&sr.sub::<4>(W - 4)), zero)
        } else {
            zero
        };
        [_mm_unpacklo_epi8(raw0, zero), tail, zero]
    }
}

#[target_feature(enable = "sse2")]
#[inline]
fn ver_store_row<const W: usize, const AVG: usize>(
    out: &mut [u8; W],
    r0: &[__m128i; 3],
    r1: &[__m128i; 3],
    r2: &[__m128i; 3],
    r3: &[__m128i; 3],
    r4: &[__m128i; 3],
    r5: &[__m128i; 3],
) {
    const { assert!(AVG == 0 || AVG == 2 || AVG == 3, "AVG must be 0, 2, or 3") };
    let r_avg = if AVG == 2 { r2 } else { r3 };
    if W >= 16 {
        let w_lo = filter_6tap_vertical_words(r0[0], r1[0], r2[0], r3[0], r4[0], r5[0]);
        let w_hi = filter_6tap_vertical_words(r0[1], r1[1], r2[1], r3[1], r4[1], r5[1]);
        let mut both = _mm_packus_epi16(w_lo, w_hi);
        if AVG != 0 {
            both = _mm_avg_epu8(both, _mm_packus_epi16(r_avg[0], r_avg[1]));
        }
        st16(out, both);
        if W > 16 {
            let w_t = filter_6tap_vertical_words(r0[2], r1[2], r2[2], r3[2], r4[2], r5[2]);
            let mut res_t = _mm_packus_epi16(w_t, w_t);
            if AVG != 0 {
                res_t = _mm_avg_epu8(res_t, _mm_packus_epi16(r_avg[2], r_avg[2]));
            }
            st8(&mut out[W - 8..], res_t);
        }
    } else if W >= 8 {
        let w_lo = filter_6tap_vertical_words(r0[0], r1[0], r2[0], r3[0], r4[0], r5[0]);
        let mut res = _mm_packus_epi16(w_lo, w_lo);
        if AVG != 0 {
            res = _mm_avg_epu8(res, _mm_packus_epi16(r_avg[0], r_avg[0]));
        }
        st8(out, res);
        if W > 8 {
            let w_t = filter_6tap_vertical_words(r0[1], r1[1], r2[1], r3[1], r4[1], r5[1]);
            let mut res_t = _mm_packus_epi16(w_t, w_t);
            if AVG != 0 {
                res_t = _mm_avg_epu8(res_t, _mm_packus_epi16(r_avg[1], r_avg[1]));
            }
            st8(&mut out[W - 8..], res_t);
        }
    } else {
        let w_lo = filter_6tap_vertical_words(r0[0], r1[0], r2[0], r3[0], r4[0], r5[0]);
        let mut res = _mm_packus_epi16(w_lo, w_lo);
        if AVG != 0 {
            res = _mm_avg_epu8(res, _mm_packus_epi16(r_avg[0], r_avg[0]));
        }
        st4(out, res);
        if W > 4 {
            let w_t = filter_6tap_vertical_words(r0[1], r1[1], r2[1], r3[1], r4[1], r5[1]);
            let mut res_t = _mm_packus_epi16(w_t, w_t);
            if AVG != 0 {
                res_t = _mm_avg_epu8(res_t, _mm_packus_epi16(r_avg[1], r_avg[1]));
            }
            st4(&mut out[W - 4..], res_t);
        }
    }
}

/// The vertical filter at width 16, 17, 8, 9, 4 or 5: the five-row window carried in
/// widened registers and one new row read per output row via `split_views`.
#[target_feature(enable = "sse2")]
#[inline]
fn ver_lanes<
    S: RefSamples + Copy,
    const W: usize,
    const H: usize,
    const SH: usize,
    const AVG: usize,
>(
    src: &S,
    dst: &mut PlaneCursorMut<'_>,
) {
    let s = src.span::<W, SH>(-2, 0);
    let mut d = dst.span_mut::<W, H>(0, 0);
    let (Some((mut sh, st)), Some((dh, dt))) =
        (s.split_views::<W, SH>(), d.split_rows_mut::<W, H>())
    else {
        ver_narrow::<S, W, H, SH, AVG>(src, dst);
        return;
    };

    let mut next_r = || ver_load_row_view::<W, _>(sh.next().expect("ver_sse2 src row"));
    let (mut r0, mut r1, mut r2, mut r3, mut r4) =
        (next_r(), next_r(), next_r(), next_r(), next_r());

    for out in dh {
        let r5 = next_r();
        ver_store_row::<W, AVG>(out, &r0, &r1, &r2, &r3, &r4, &r5);
        (r0, r1, r2, r3, r4) = (r1, r2, r3, r4, r5);
    }
    let r5 = ver_load_row_view::<W, _>(st);
    ver_store_row::<W, AVG>(dt, &r0, &r1, &r2, &r3, &r4, &r5);
}

/// The widths the lane path has no form for: the scalar over the same span.
#[target_feature(enable = "sse2")]
#[inline]
fn ver_odd<
    S: RefSamples + Copy,
    const W: usize,
    const H: usize,
    const SH: usize,
    const AVG: usize,
>(
    src: &S,
    dst: &mut PlaneCursorMut<'_>,
) {
    ver_narrow::<S, W, H, SH, AVG>(src, dst);
}

simd_entry! {
    #[target_feature(enable = "sse2")]
    fn ver_block_sse2<
        S: RefSamples + Copy,
        const W: usize,
        const H: usize,
        const SH: usize,
        const AVG: usize,
    >(
        src: &S,
        dst: &mut PlaneCursorMut<'_>,
        _token: Sse2Token,
    ) {
        match W {
            17 | 16 | 9 | 8 | 5 | 4 => ver_lanes::<S, W, H, SH, AVG>(src, dst),
            _ => ver_odd::<S, W, H, SH, AVG>(src, dst),
        }
    }
}

#[cold]
#[inline(never)]
fn ver_block_fallback<
    S: RefSamples + Copy,
    const W: usize,
    const H: usize,
    const SH: usize,
    const AVG: usize,
>(
    src: &S,
    dst: &mut PlaneCursorMut<'_>,
) {
    if W == 16
        && let Some(token) = Avx2Token::try_new_enabled()
    {
        ver_lanes_avx2_16x::<S, H, SH, AVG>(src, dst, token)
    } else {
        ver_block_sse2::<S, W, H, SH, AVG>(src, dst, sse2_token())
    }
}

/// `McHorVer02` over one const-shape block: the width picks the path, and the
/// `match` folds because `W` is a constant.
#[inline(always)]
fn ver_block<
    S: RefSamples + Copy,
    const W: usize,
    const H: usize,
    const SH: usize,
    const AVG: usize,
>(
    src: &S,
    dst: &mut PlaneCursorMut<'_>,
) {
    if W == 16 {
        if let Some(token) = Avx2Token::try_fast_enabled() {
            ver_lanes_avx2_16x::<S, H, SH, AVG>(src, dst, token)
        } else {
            ver_block_fallback::<S, W, H, SH, AVG>(src, dst)
        }
    } else {
        ver_block_sse2::<S, W, H, SH, AVG>(src, dst, sse2_token())
    }
}

/// The run-time-shape twin — cold, and scalar; see [`McLeaves`].
fn ver_any<S: RefSamples + Copy, const AVG: usize>(
    src: &S,
    dst: &mut PlaneCursorMut<'_>,
    width: usize,
    height: usize,
) {
    for dy in 0..height as isize {
        let out = dst.row_mut(dy, 0, width);
        for (x, o) in out.iter_mut().enumerate() {
            let t: [u8; 6] = std::array::from_fn(|k| src.at(x as isize, dy + k as isize - 2));
            let mut v = WelsClip1((filter_input_8bit(&t) + 16) >> 5);
            if AVG != 0 {
                v = ((v as u32 + t[AVG] as u32 + 1) >> 1) as u8;
            }
            *o = v;
        }
    }
}

/// Public safe entry point for SSE2 vertical half-pel filter.
pub fn mc_hor_ver02<S: RefSamples + Copy>(
    src: &S,
    dst: &mut PlaneCursorMut<'_>,
    width: usize,
    height: usize,
) {
    ver_shaped::<Sse2Leaves, S, 0>(src, dst, width, height)
}

// ============================================================================
// 2D Center 6x6-Tap Filter: McHorVer22 (SSSE3)
// ============================================================================

#[target_feature(enable = "ssse3")]
#[inline]
fn hor_filter_4px_16bit(v0: __m128i, v1: __m128i) -> __m128i {
    let mask_01 = _mm_setr_epi8(0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9);
    let mask_23 = _mm_setr_epi8(4, 5, 6, 7, 6, 7, 8, 9, 8, 9, 10, 11, 10, 11, 12, 13);

    let coeff_01 = _mm_setr_epi16(1, -5, 1, -5, 1, -5, 1, -5);
    let coeff_23 = _mm_set1_epi16(20);
    let coeff_45 = _mm_setr_epi16(-5, 1, -5, 1, -5, 1, -5, 1);

    let p01 = _mm_shuffle_epi8(v0, mask_01);
    let p23 = _mm_shuffle_epi8(v0, mask_23);
    let v_align4 = _mm_alignr_epi8(v1, v0, 8);
    let p45 = _mm_shuffle_epi8(v_align4, mask_01);

    let m01 = _mm_madd_epi16(p01, coeff_01);
    let m23 = _mm_madd_epi16(p23, coeff_23);
    let m45 = _mm_madd_epi16(p45, coeff_45);

    let sum = _mm_add_epi32(_mm_add_epi32(m01, m45), m23);
    let rounded = _mm_add_epi32(sum, _mm_set1_epi32(512));
    _mm_srai_epi32(rounded, 10)
}

#[target_feature(enable = "ssse3")]
#[inline]
fn hor_filter_8px_from_regs(v0: __m128i, v1: __m128i, v2: __m128i) -> __m128i {
    let s0 = hor_filter_4px_16bit(v0, v1);
    let v0_hi = _mm_alignr_epi8(v1, v0, 8);
    let v1_hi = _mm_alignr_epi8(v2, v1, 8);
    let s1 = hor_filter_4px_16bit(v0_hi, v1_hi);
    _mm_packs_epi32(s0, s1)
}

/// Loads one row of `SW` bytes (`SW <= 22`) into three 8-word `i16` vectors (`[__m128i; 3]`)
/// using only 16-byte and 8-byte register loads directly from the block span.
#[target_feature(enable = "ssse3")]
#[inline]
fn cen_load_row<R: BlockRows, const SW: usize>(r: &R, y: usize) -> [__m128i; 3] {
    let zero = _mm_setzero_si128();
    if SW >= 16 {
        let v01 = ld16(&r.row::<16>(y, 0));
        let v2 = ld8(&r.row::<8>(y, SW - 8));
        [
            _mm_unpacklo_epi8(v01, zero),
            _mm_unpackhi_epi8(v01, zero),
            _mm_unpacklo_epi8(v2, zero),
        ]
    } else if SW >= 8 {
        let v0 = ld8(&r.row::<8>(y, 0));
        let v1 = ld8(&r.row::<8>(y, SW - 8));
        [
            _mm_unpacklo_epi8(v0, zero),
            _mm_unpacklo_epi8(v1, zero),
            zero,
        ]
    } else {
        let row = r.row::<SW>(y, 0);
        let mut buf = [0u8; 8];
        buf[..SW].copy_from_slice(&row);
        let v0 = ld8(&buf);
        [_mm_unpacklo_epi8(v0, zero), zero, zero]
    }
}

macro_rules! cen_filter_row {
    ($out:expr, $W:expr, $SW:expr, $w0:expr, $w1:expr, $w2:expr, $w3:expr, $w4:expr, $w5:expr) => {{
        let out = $out;
        let t0 = filter_6tap_intermediate_8_samples($w0[0], $w1[0], $w2[0], $w3[0], $w4[0], $w5[0]);
        if $W >= 16 {
            let t1 =
                filter_6tap_intermediate_8_samples($w0[1], $w1[1], $w2[1], $w3[1], $w4[1], $w5[1]);
            let t2_overlap =
                filter_6tap_intermediate_8_samples($w0[2], $w1[2], $w2[2], $w3[2], $w4[2], $w5[2]);
            let t2 = if $SW == 22 {
                _mm_srli_si128::<4>(t2_overlap)
            } else {
                _mm_srli_si128::<6>(t2_overlap)
            };
            let w_lo = hor_filter_8px_from_regs(t0, t1, t2);
            let w_hi = hor_filter_8px_from_regs(t1, t2, _mm_setzero_si128());
            st16(out, _mm_packus_epi16(w_lo, w_hi));
            if $W > 16 {
                let v0 = _mm_alignr_epi8(t2, t1, 10);
                let v1 = _mm_srli_si128::<10>(t2);
                let s0 = hor_filter_4px_16bit(v0, v1);
                let w4 = _mm_packs_epi32(s0, s0);
                st4(&mut out[$W - 4..], _mm_packus_epi16(w4, w4));
            }
        } else if $W >= 8 {
            let t1_overlap =
                filter_6tap_intermediate_8_samples($w0[1], $w1[1], $w2[1], $w3[1], $w4[1], $w5[1]);
            let t1 = if $SW == 14 {
                _mm_srli_si128::<4>(t1_overlap)
            } else {
                _mm_srli_si128::<6>(t1_overlap)
            };
            let w_lo = hor_filter_8px_from_regs(t0, t1, _mm_setzero_si128());
            st8(out, _mm_packus_epi16(w_lo, w_lo));
            if $W > 8 {
                let v0 = _mm_alignr_epi8(t1, t0, 10);
                let v1 = _mm_srli_si128::<10>(t1);
                let s0 = hor_filter_4px_16bit(v0, v1);
                let w4 = _mm_packs_epi32(s0, s0);
                st4(&mut out[$W - 4..], _mm_packus_epi16(w4, w4));
            }
        } else if $W >= 4 {
            let t1_overlap =
                filter_6tap_intermediate_8_samples($w0[1], $w1[1], $w2[1], $w3[1], $w4[1], $w5[1]);
            let t1 = if $SW == 10 {
                _mm_srli_si128::<12>(t1_overlap)
            } else {
                _mm_srli_si128::<14>(t1_overlap)
            };
            let s0 = hor_filter_4px_16bit(t0, t1);
            let w4 = _mm_packs_epi32(s0, s0);
            st4(out, _mm_packus_epi16(w4, w4));
            if $W > 4 {
                let v0 = _mm_alignr_epi8(t1, t0, 2);
                let v1 = _mm_srli_si128::<2>(t1);
                let s1 = hor_filter_4px_16bit(v0, v1);
                let w4_t = _mm_packs_epi32(s1, s1);
                st4(&mut out[$W - 4..], _mm_packus_epi16(w4_t, w4_t));
            }
        } else {
            let mut tmp = [0i16; 8];
            st8_i16(&mut tmp, t0);
            for col in 0..$W {
                let t: &[i16; 6] = tmp[col..col + 6].try_into().unwrap();
                out[col] = WelsClip1((hor_filter_input_16bit(t) + 512) >> 10);
            }
        }
    }};
}

simd_entry! {
    /// `McHorVer22` over one const-shape block: the vertical 6-tap and horizontal 6-tap
    /// executed entirely in XMM registers over a sliding 6-row window.
    #[target_feature(enable = "ssse3")]
    fn cen_block<
        S: RefSamples + Copy,
        const W: usize,
        const SW: usize,
        const H: usize,
        const SH: usize,
    >(
        src: &S,
        dst: &mut PlaneCursorMut<'_>,
        _token: Ssse3Token,
    ) {
        const {
            assert!(
                SW <= 17 + 5,
                "mc_hor_ver22 width exceeds the 17 iTmp is sized for"
            )
        };
        let s = src.span::<SW, SH>(-2, -2);
        let mut d = dst.span_mut::<W, H>(0, 0);

        let (mut w0, mut w1, mut w2, mut w3, mut w4) = (
            cen_load_row::<_, SW>(&s, 0),
            cen_load_row::<_, SW>(&s, 1),
            cen_load_row::<_, SW>(&s, 2),
            cen_load_row::<_, SW>(&s, 3),
            cen_load_row::<_, SW>(&s, 4),
        );

        for y in 0..H {
            let w5 = cen_load_row::<_, SW>(&s, y + 5);
            let out = d.row_mut::<W>(y, 0);
            cen_filter_row!(out, W, SW, w0, w1, w2, w3, w4, w5);
            (w0, w1, w2, w3, w4) = (w1, w2, w3, w4, w5);
        }
    }
}

/// The run-time-shape twin — cold; see [`McLeaves`]. The `width <= 17` contract is what
/// sizes `iTmp`.
fn cen_any<S: RefSamples + Copy>(
    src: &S,
    dst: &mut PlaneCursorMut<'_>,
    width: usize,
    height: usize,
) {
    assert!(
        width <= 17,
        "mc_hor_ver22 width {width} exceeds the 17 iTmp is sized for"
    );
    let n = width + 5;
    let mut iTmp = [0i16; 17 + 5];
    for dy in 0..height as isize {
        for (j, t) in iTmp[..n].iter_mut().enumerate() {
            let x = j as isize - 2;
            let p: [u8; 6] = std::array::from_fn(|k| src.at(x, dy + k as isize - 2));
            *t = filter_input_8bit(&p) as i16;
        }
        let out = dst.row_mut(dy, 0, width);
        for (o, t) in out.iter_mut().zip(iTmp[..n].windows(6)) {
            *o = WelsClip1((hor_filter_input_16bit(t.try_into().expect("six taps")) + 512) >> 10);
        }
    }
}

/// Public safe entry point for SSE2 center half-pel filter.
pub fn mc_hor_ver22<S: RefSamples + Copy>(
    src: &S,
    dst: &mut PlaneCursorMut<'_>,
    width: usize,
    height: usize,
) {
    cen_shaped::<Sse2Leaves, S>(src, dst, width, height)
}

// ============================================================================
// Luma Quarter-Pel MC (SSE2 / SSSE3)
// ============================================================================

/// The SSE2 leaf set — `McLeaves` with the four kernels in this file. The twelve
/// quarter-pel composites live once, in [`crate::common::mc::McLeaves`].
pub struct Sse2Leaves;

impl McLeaves for Sse2Leaves {
    const FUSED_QPEL: bool = true;
    #[inline(always)]
    fn hor<
        S: RefSamples + Copy,
        const W: usize,
        const SW: usize,
        const H: usize,
        const AVG: usize,
    >(
        src: &S,
        dst: &mut PlaneCursorMut<'_>,
    ) {
        hor_block::<S, W, SW, H, AVG>(src, dst, ssse3_token())
    }
    #[inline(always)]
    fn hor_any<S: RefSamples + Copy, const AVG: usize>(
        src: &S,
        dst: &mut PlaneCursorMut<'_>,
        width: usize,
        height: usize,
    ) {
        hor_any::<S, AVG>(src, dst, width, height)
    }
    #[inline(always)]
    fn ver<
        S: RefSamples + Copy,
        const W: usize,
        const H: usize,
        const SH: usize,
        const AVG: usize,
    >(
        src: &S,
        dst: &mut PlaneCursorMut<'_>,
    ) {
        ver_block::<S, W, H, SH, AVG>(src, dst)
    }
    #[inline(always)]
    fn ver_any<S: RefSamples + Copy, const AVG: usize>(
        src: &S,
        dst: &mut PlaneCursorMut<'_>,
        width: usize,
        height: usize,
    ) {
        ver_any::<S, AVG>(src, dst, width, height)
    }
    #[inline(always)]
    fn cen<
        S: RefSamples + Copy,
        const W: usize,
        const SW: usize,
        const H: usize,
        const SH: usize,
    >(
        src: &S,
        dst: &mut PlaneCursorMut<'_>,
    ) {
        cen_block::<S, W, SW, H, SH>(src, dst, ssse3_token())
    }
    #[inline(always)]
    fn cen_any<S: RefSamples + Copy>(
        src: &S,
        dst: &mut PlaneCursorMut<'_>,
        width: usize,
        height: usize,
    ) {
        cen_any::<S>(src, dst, width, height)
    }
    #[inline(always)]
    fn avg<A: RefSamples, B: RefSamples, const W: usize, const H: usize>(
        dst: &mut PlaneCursorMut<'_>,
        a: &A,
        b: &B,
    ) {
        avg_block::<A, B, W, H>(dst, a, b);
    }
    #[inline(always)]
    fn avg_any<A: RefSamples, B: RefSamples>(
        dst: &mut PlaneCursorMut<'_>,
        a: &A,
        b: &B,
        width: usize,
        height: usize,
    ) {
        avg_any::<A, B>(dst, a, b, width, height)
    }
    #[inline(always)]
    fn chroma<
        S: RefSamples + Copy,
        const W: usize,
        const SW: usize,
        const H: usize,
        const SH: usize,
    >(
        src: &S,
        dst: &mut PlaneCursorMut<'_>,
        w: &[u8; 4],
    ) {
        chroma_block::<S, W, SW, H, SH>(src, dst, w, ssse3_token())
    }
    #[inline(always)]
    fn chroma_any<S: RefSamples + Copy>(
        src: &S,
        dst: &mut PlaneCursorMut<'_>,
        w: &[u8; 4],
        width: usize,
        height: usize,
    ) {
        chroma_any::<S>(src, dst, w, width, height)
    }
}

/// Public safe entry point for SSE2 luma quarter-pel MC.
#[inline(always)]
pub fn mc_luma<S: RefSamples + Copy>(
    src: &S,
    dst: &mut PlaneCursorMut<'_>,
    mv_x: i16,
    mv_y: i16,
    width: usize,
    height: usize,
) {
    if (mv_x & 0x03) == 0 && (mv_y & 0x03) == 0 {
        match (width, height) {
            (16, 16) => copy_block_simd::<16, 16, S>(src, dst),
            (16, 8) => copy_block_simd::<16, 8, S>(src, dst),
            (8, 16) => copy_block_simd::<8, 16, S>(src, dst),
            (8, 8) => copy_block_simd::<8, 8, S>(src, dst),
            (8, 4) => copy_block_simd::<8, 4, S>(src, dst),
            (4, 8) => copy_block_simd::<4, 8, S>(src, dst),
            (4, 4) => copy_block_simd::<4, 4, S>(src, dst),
            _ => mc_copy(src, dst, width, height),
        }
        return;
    }
    mc_luma_frac(src, dst, mv_x, mv_y, width, height)
}

#[inline(always)]
fn mc_luma_frac<S: RefSamples + Copy>(
    src: &S,
    dst: &mut PlaneCursorMut<'_>,
    mv_x: i16,
    mv_y: i16,
    width: usize,
    height: usize,
) {
    mc_luma_with::<Sse2Leaves, S>(src, dst, mv_x, mv_y, width, height)
}

// ============================================================================
// Unit Tests: Differential Parity Against Scalar Kernels
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safe::plane::PlaneCursor;
    // These MUST be the `_c` scalar kernels, not the same-named dispatchers, which
    // route to the very SSE2 kernels under test.
    use crate::common::mc::McLeaves;
    use crate::common::mc::{
        mc_chroma_with_frag_mv, mc_hor_ver02_c as scalar_hor_ver02,
        mc_hor_ver20_c as scalar_hor_ver20, mc_hor_ver22_c as scalar_hor_ver22,
        mc_luma_c as scalar_luma, pixel_avg_c as scalar_pixel_avg,
    };
    use crate::encoder::rec_view::RecCursor;

    const STRIDE: usize = 64;
    const ROWS: usize = 64;

    fn filled_plane() -> Vec<u8> {
        let mut v = vec![0u8; STRIDE * ROWS];
        let mut s: u32 = 0xdead_beef;
        for b in v.iter_mut() {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *b = (s >> 16) as u8;
        }
        v
    }

    #[test]
    fn test_pixel_avg_parity() {
        let a = filled_plane();
        let mut b = a.clone();
        for x in b.iter_mut() {
            *x = x.wrapping_add(42);
        }

        let ca = PlaneCursor::new(&a, 10 * STRIDE + 8, STRIDE);
        let cb = PlaneCursor::new(&b, 12 * STRIDE + 8, STRIDE);

        for (w, h) in [
            (16, 16),
            (16, 8),
            (8, 16),
            (8, 8),
            (8, 4),
            (4, 8),
            (4, 4),
            (17, 16),
            (9, 8),
            (5, 4),
            (2, 2),
        ] {
            let mut dst_scalar = vec![0u8; STRIDE * ROWS];
            let mut dst_simd = vec![0u8; STRIDE * ROWS];

            let mut cur_scalar = PlaneCursorMut::new(&mut dst_scalar, 10 * STRIDE + 8, STRIDE);
            scalar_pixel_avg(&mut cur_scalar, &ca, &cb, w, h);

            let mut cur_simd = PlaneCursorMut::new(&mut dst_simd, 10 * STRIDE + 8, STRIDE);
            pixel_avg(&mut cur_simd, &ca, &cb, w, h);

            assert_eq!(dst_scalar, dst_simd, "pixel_avg mismatch at {w}x{h}");
        }
    }

    #[test]
    fn test_mc_chroma_parity() {
        let base = filled_plane();
        let src_c = 10 * STRIDE + 10;
        let dst_c = 20 * STRIDE + 10;
        let src = PlaneCursor::new(&base, src_c, STRIDE);

        let shapes = [(8, 8), (8, 4), (4, 8), (4, 4), (4, 2), (2, 4), (2, 2)];

        for &(w, h) in &shapes {
            for dy in 0..8i16 {
                for dx in 0..8i16 {
                    let mut dst_scalar = vec![0u8; STRIDE * ROWS];
                    let mut dst_simd = vec![0u8; STRIDE * ROWS];

                    let mut cur_scalar = PlaneCursorMut::new(&mut dst_scalar, dst_c, STRIDE);
                    if (dx & 7) == 0 && (dy & 7) == 0 {
                        mc_copy(&src, &mut cur_scalar, w, h);
                    } else {
                        mc_chroma_with_frag_mv(&src, &mut cur_scalar, dx, dy, w, h);
                    }

                    let mut cur_simd = PlaneCursorMut::new(&mut dst_simd, dst_c, STRIDE);
                    mc_chroma(&src, &mut cur_simd, dx, dy, w, h);

                    assert_eq!(
                        dst_scalar, dst_simd,
                        "mc_chroma mismatch at {w}x{h} with mv=({dx}, {dy})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_mc_hor_ver20_parity() {
        let base = filled_plane();
        let src_c = 10 * STRIDE + 10;
        let dst_c = 20 * STRIDE + 10;
        let src = PlaneCursor::new(&base, src_c, STRIDE);

        let shapes = [
            (16, 16),
            (16, 8),
            (8, 16),
            (8, 8),
            (8, 4),
            (4, 8),
            (4, 4),
            (17, 16),
            (17, 8),
            (9, 16),
            (9, 8),
            // Outside the const tables, so these drive the run-time fallback.
            (5, 8),
            (5, 4),
            (17, 17),
        ];

        for &(w, h) in &shapes {
            let mut dst_scalar = vec![0u8; STRIDE * ROWS];
            let mut dst_simd = vec![0u8; STRIDE * ROWS];

            let mut cur_scalar = PlaneCursorMut::new(&mut dst_scalar, dst_c, STRIDE);
            scalar_hor_ver20(&src, &mut cur_scalar, w, h);

            let mut cur_simd = PlaneCursorMut::new(&mut dst_simd, dst_c, STRIDE);
            mc_hor_ver20(&src, &mut cur_simd, w, h);

            assert_eq!(dst_scalar, dst_simd, "mc_hor_ver20 mismatch at {w}x{h}");
        }
    }

    #[test]
    fn test_mc_hor_ver02_parity() {
        let base = filled_plane();
        let src_c = 10 * STRIDE + 10;
        let dst_c = 20 * STRIDE + 10;
        let src = PlaneCursor::new(&base, src_c, STRIDE);

        let shapes = [
            (16, 16),
            (16, 8),
            (8, 16),
            (8, 8),
            (8, 4),
            (4, 8),
            (4, 4),
            (16, 17),
            (16, 9),
            (8, 17),
            (8, 9),
            // Outside the const tables, so these drive the run-time fallback.
            (8, 5),
            (4, 5),
            (17, 17),
        ];

        for &(w, h) in &shapes {
            let mut dst_scalar = vec![0u8; STRIDE * ROWS];
            let mut dst_simd = vec![0u8; STRIDE * ROWS];

            let mut cur_scalar = PlaneCursorMut::new(&mut dst_scalar, dst_c, STRIDE);
            scalar_hor_ver02(&src, &mut cur_scalar, w, h);

            let mut cur_simd = PlaneCursorMut::new(&mut dst_simd, dst_c, STRIDE);
            mc_hor_ver02(&src, &mut cur_simd, w, h);

            assert_eq!(dst_scalar, dst_simd, "mc_hor_ver02 mismatch at {w}x{h}");
        }
    }

    #[test]
    fn test_mc_hor_ver22_parity() {
        let base = filled_plane();
        let src_c = 10 * STRIDE + 10;
        let dst_c = 20 * STRIDE + 10;
        let src = PlaneCursor::new(&base, src_c, STRIDE);

        let shapes = [
            (16, 16),
            (16, 8),
            (8, 16),
            (8, 8),
            (8, 4),
            (4, 8),
            (4, 4),
            (17, 17),
            (17, 9),
            (9, 17),
            (9, 9),
            // Outside the const tables, so these drive the run-time fallback.
            (9, 5),
            (5, 5),
            (17, 16),
        ];

        for &(w, h) in &shapes {
            let mut dst_scalar = vec![0u8; STRIDE * ROWS];
            let mut dst_simd = vec![0u8; STRIDE * ROWS];

            let mut cur_scalar = PlaneCursorMut::new(&mut dst_scalar, dst_c, STRIDE);
            scalar_hor_ver22(&src, &mut cur_scalar, w, h);

            let mut cur_simd = PlaneCursorMut::new(&mut dst_simd, dst_c, STRIDE);
            mc_hor_ver22(&src, &mut cur_simd, w, h);

            assert_eq!(dst_scalar, dst_simd, "mc_hor_ver22 mismatch at {w}x{h}");
        }
    }

    #[test]
    fn test_mc_luma_parity() {
        let base = filled_plane();
        let src_c = 10 * STRIDE + 10;
        let dst_c = 20 * STRIDE + 10;
        let src = PlaneCursor::new(&base, src_c, STRIDE);

        let shapes = [(16, 16), (16, 8), (8, 16), (8, 8), (8, 4), (4, 8), (4, 4)];

        for &(w, h) in &shapes {
            for qy in 0..4i16 {
                for qx in 0..4i16 {
                    let mut dst_scalar = vec![0u8; STRIDE * ROWS];
                    let mut dst_simd = vec![0u8; STRIDE * ROWS];

                    let mut cur_scalar = PlaneCursorMut::new(&mut dst_scalar, dst_c, STRIDE);
                    scalar_luma(&src, &mut cur_scalar, qx, qy, w, h);

                    let mut cur_simd = PlaneCursorMut::new(&mut dst_simd, dst_c, STRIDE);
                    mc_luma(&src, &mut cur_simd, qx, qy, w, h);

                    assert_eq!(
                        dst_scalar, dst_simd,
                        "mc_luma mismatch at {w}x{h} with qpos=({qx}, {qy})"
                    );
                }
            }
        }
    }
    /// The same kernels reached through the shared cursor, whose rows arrive by
    /// value: the encoder's reference picture is one of these, and it is what the
    /// half-pel refinement filters read at `kiW + 1` by `kiH + 1`.
    #[test]
    fn mc_parity_through_the_shared_cursor() {
        let mut base = filled_plane();
        let src_c = 10 * STRIDE + 10;
        let dst_c = 20 * STRIDE + 10;
        let mut want = vec![0u8; STRIDE * ROWS];
        let mut got = vec![0u8; STRIDE * ROWS];
        macro_rules! pair {
            ($scalar:expr, $simd:expr) => {{
                want.iter_mut().for_each(|b| *b = 0);
                got.iter_mut().for_each(|b| *b = 0);
                {
                    let src = PlaneCursor::new(&base, src_c, STRIDE);
                    let mut d = PlaneCursorMut::new(&mut want, dst_c, STRIDE);
                    #[allow(clippy::redundant_closure_call)]
                    ($scalar)(&src, &mut d);
                }
                {
                    let src = RecCursor::over_owned(&mut base, src_c, STRIDE);
                    let mut d = PlaneCursorMut::new(&mut got, dst_c, STRIDE);
                    #[allow(clippy::redundant_closure_call)]
                    ($simd)(&src, &mut d);
                }
                assert_eq!(want, got);
            }};
        }
        for (qx, qy) in [
            (0i16, 0i16),
            (1, 0),
            (2, 0),
            (3, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 1),
            (2, 2),
            (3, 3),
        ] {
            for (w, h) in [(16, 16), (16, 8), (8, 16), (8, 8), (8, 4), (4, 8), (4, 4)] {
                pair!(
                    |s: &PlaneCursor<'_>, d: &mut PlaneCursorMut<'_>| scalar_luma(
                        s, d, qx, qy, w, h
                    ),
                    |s: &RecCursor<'_>, d: &mut PlaneCursorMut<'_>| mc_luma(s, d, qx, qy, w, h)
                );
            }
        }
        // The half-pel refinement shapes, which only ever run over this cursor.
        for (w, h) in [(17, 16), (17, 8), (9, 16), (9, 8)] {
            pair!(
                |s: &PlaneCursor<'_>, d: &mut PlaneCursorMut<'_>| scalar_hor_ver20(s, d, w, h),
                |s: &RecCursor<'_>, d: &mut PlaneCursorMut<'_>| mc_hor_ver20(s, d, w, h)
            );
        }
        for (w, h) in [(16, 17), (16, 9), (8, 17), (8, 9)] {
            pair!(
                |s: &PlaneCursor<'_>, d: &mut PlaneCursorMut<'_>| scalar_hor_ver02(s, d, w, h),
                |s: &RecCursor<'_>, d: &mut PlaneCursorMut<'_>| mc_hor_ver02(s, d, w, h)
            );
        }
        for (w, h) in [(17, 17), (17, 9), (9, 17), (9, 9)] {
            pair!(
                |s: &PlaneCursor<'_>, d: &mut PlaneCursorMut<'_>| scalar_hor_ver22(s, d, w, h),
                |s: &RecCursor<'_>, d: &mut PlaneCursorMut<'_>| mc_hor_ver22(s, d, w, h)
            );
        }
        for (w, h) in [(8, 8), (8, 4), (4, 8), (4, 4), (4, 2), (2, 4), (2, 2)] {
            for (mx, my) in [(0i16, 0i16), (3, 5), (1, 0), (0, 7)] {
                pair!(
                    |s: &PlaneCursor<'_>, d: &mut PlaneCursorMut<'_>| {
                        if (mx & 7) == 0 && (my & 7) == 0 {
                            mc_copy(s, d, w, h)
                        } else {
                            mc_chroma_with_frag_mv(s, d, mx, my, w, h)
                        }
                    },
                    |s: &RecCursor<'_>, d: &mut PlaneCursorMut<'_>| mc_chroma(s, d, mx, my, w, h)
                );
            }
        }
        // `pixel_avg`'s second operand is the reference block itself in four of the
        // quarter-pixel candidates, so it too sees this cursor.
        for (w, h) in [(16, 16), (16, 8), (8, 16), (8, 8)] {
            let a = filled_plane();
            want.iter_mut().for_each(|b| *b = 0);
            got.iter_mut().for_each(|b| *b = 0);
            {
                let ca = PlaneCursor::new(&a, src_c, STRIDE);
                let cb = PlaneCursor::new(&base, src_c, STRIDE);
                scalar_pixel_avg(
                    &mut PlaneCursorMut::new(&mut want, dst_c, STRIDE),
                    &ca,
                    &cb,
                    w,
                    h,
                );
            }
            {
                let ca = PlaneCursor::new(&a, src_c, STRIDE);
                let cb = RecCursor::over_owned(&mut base, src_c, STRIDE);
                pixel_avg(
                    &mut PlaneCursorMut::new(&mut got, dst_c, STRIDE),
                    &ca,
                    &cb,
                    w,
                    h,
                );
            }
            assert_eq!(want, got, "pixel_avg via RecCursor at {w}x{h}");
        }
    }

    /// The `_AVERAGE_WITH_` forms of the two direct filters agree with the composites
    /// they would replace at quarter-pel `(1, 0)`, `(3, 0)`, `(0, 1)` and `(0, 3)`: the
    /// fused form is a rounded average against one of the filter's own taps, which is
    /// what the composite computes with an averaging pass.
    #[test]
    fn the_fused_quarter_pel_arms_agree_with_the_composites() {
        let base = filled_plane();
        let src = PlaneCursor::new(&base, 10 * STRIDE + 10, STRIDE);
        let dst_c = 20 * STRIDE + 10;
        for (qx, qy, avg) in [(1i16, 0i16, 2usize), (3, 0, 3), (0, 1, 2), (0, 3, 3)] {
            let mut want = vec![0u8; STRIDE * ROWS];
            let mut got = vec![0u8; STRIDE * ROWS];
            mc_luma(
                &src,
                &mut PlaneCursorMut::new(&mut want, dst_c, STRIDE),
                qx,
                qy,
                16,
                16,
            );
            {
                let mut d = PlaneCursorMut::new(&mut got, dst_c, STRIDE);
                match avg {
                    2 if qy == 0 => Sse2Leaves::hor::<_, 16, 21, 16, 2>(&src, &mut d),
                    3 if qy == 0 => Sse2Leaves::hor::<_, 16, 21, 16, 3>(&src, &mut d),
                    2 => Sse2Leaves::ver::<_, 16, 16, 21, 2>(&src, &mut d),
                    _ => Sse2Leaves::ver::<_, 16, 16, 21, 3>(&src, &mut d),
                }
            }
            assert_eq!(
                want, got,
                "fused quarter-pel ({qx}, {qy}) differs from the composite"
            );
        }
    }
}
