// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Fixed-shape macroblock copies for the aarch64 kernel table
//! (`WelsCopy*_AArch64_neon`, `codec/common/arm64/copy_mb_aarch64_neon.S`).
//!
//! Each kernel cuts the block's span once per operand — one bounds check per operand
//! rather than per row — and walks rows via [`BlockRows::split_rows`] and
//! [`PlaneSpanMut::split_rows_mut`](crate::safe::plane::PlaneSpanMut::split_rows_mut) so
//! LLVM lowers the constant-width loops directly to unaligned NEON loads and stores with
//! zero `unsafe` blocks.
#![forbid(unsafe_code)]

use crate::safe::plane::{BlockRows, PlaneCursor, PlaneCursorMut, RefSamples};

#[cold]
#[inline(never)]
fn copy_narrow_stride<const W: usize, const H: usize>(
    dst: &mut PlaneCursorMut<'_>,
    src: &PlaneCursor<'_>,
) {
    let s = RefSamples::span::<W, H>(src, 0, 0);
    let mut d = dst.span_mut::<W, H>(0, 0);
    for y in 0..H {
        *d.row_mut::<W>(y, 0) = s.row::<W>(y, 0);
    }
}

#[cold]
#[inline(never)]
fn copy_slice_narrow_stride<const W: usize, const H: usize>(
    dst: &mut PlaneCursorMut<'_>,
    src: &[u8],
) {
    let mut d = dst.span_mut::<W, H>(0, 0);
    let (rows, _) = src[..W * H].as_chunks::<W>();
    for (y, r) in rows.iter().enumerate() {
        *d.row_mut::<W>(y, 0) = *r;
    }
}

/// Copies `H` rows of `W` bytes from `src` to `dst`, each walking its own stride.
///
/// Two strides rather than one because `WelsMdBackgroundMbEnc` copies the
/// mode-decision scratch — a stride-16 array — into a picture plane.
#[inline(always)]
fn copy_block<const W: usize, const H: usize>(
    dst: &mut PlaneCursorMut<'_>,
    src: &PlaneCursor<'_>,
) {
    {
        let s = RefSamples::span::<W, H>(src, 0, 0);
        let mut d = dst.span_mut::<W, H>(0, 0);
        if let (Some((sh, st)), Some((dh, dt))) =
            (s.split_rows::<W, H>(), d.split_rows_mut::<W, H>())
        {
            for (sr, dr) in sh.zip(dh) {
                *dr = sr;
            }
            *dt = st();
            return;
        }
    }
    copy_narrow_stride::<W, H>(dst, src);
}

/// Copies a packed `W * H` slice from `src` into `dst`.
#[inline(always)]
fn copy_slice<const W: usize, const H: usize>(dst: &mut PlaneCursorMut<'_>, src: &[u8]) {
    let (rows, _) = src[..W * H].as_chunks::<W>();
    let Some((last_row, head_rows)) = rows.split_last() else {
        return;
    };
    {
        let mut d = dst.span_mut::<W, H>(0, 0);
        if let Some((dh, dt)) = d.split_rows_mut::<W, H>() {
            for (sr, dr) in head_rows.iter().zip(dh) {
                *dr = *sr;
            }
            *dt = *last_row;
            return;
        }
    }
    copy_slice_narrow_stride::<W, H>(dst, src);
}

/// `WelsCopy16x16_AArch64_neon` and `WelsCopy16x16NotAligned_AArch64_neon`.
#[inline(always)]
pub fn copy_16x16(dst: &mut PlaneCursorMut<'_>, src: &PlaneCursor<'_>) {
    copy_block::<16, 16>(dst, src);
}

/// `WelsCopy16x8NotAligned_AArch64_neon`.
#[inline(always)]
pub fn copy_16x8(dst: &mut PlaneCursorMut<'_>, src: &PlaneCursor<'_>) {
    copy_block::<16, 8>(dst, src);
}

/// `WelsCopy8x16_AArch64_neon`.
#[inline(always)]
pub fn copy_8x16(dst: &mut PlaneCursorMut<'_>, src: &PlaneCursor<'_>) {
    copy_block::<8, 16>(dst, src);
}

/// `WelsCopy8x8_AArch64_neon`.
#[inline(always)]
pub fn copy_8x8(dst: &mut PlaneCursorMut<'_>, src: &PlaneCursor<'_>) {
    copy_block::<8, 8>(dst, src);
}

/// Copies a packed 16x16 block from `src` into `dst`.
#[inline(always)]
pub fn copy_16x16_slice(dst: &mut PlaneCursorMut<'_>, src: &[u8]) {
    copy_slice::<16, 16>(dst, src);
}

/// Copies a packed 8x8 block from `src` into `dst`.
#[inline(always)]
pub fn copy_8x8_slice(dst: &mut PlaneCursorMut<'_>, src: &[u8]) {
    copy_slice::<8, 8>(dst, src);
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::encode_mb_aux::{
        WelsCopy8x8_c, WelsCopy8x16_c, WelsCopy16x8_c, WelsCopy16x16_c,
    };

    /// A plane of `stride * rows` distinct-ish bytes.
    fn plane(stride: usize, rows: usize, seed: u8) -> Vec<u8> {
        (0..stride * rows)
            .map(|i| (i as u8).wrapping_mul(37).wrapping_add(seed))
            .collect()
    }

    /// Runs one shape through both the scalar slot body and the kernel here over
    /// identical planes, and requires the whole plane to match afterwards, not just
    /// the block: a kernel that ran a row long or walked the wrong stride lands
    /// outside the block.
    fn check(
        w: usize,
        h: usize,
        stride: usize,
        scalar: fn(&mut PlaneCursorMut<'_>, &PlaneCursor<'_>),
        simd: fn(&mut PlaneCursorMut<'_>, &PlaneCursor<'_>),
    ) {
        // Two spare rows below the block and an anchor off (0, 0), so a kernel that
        // ignored the anchor or ran a row long has somewhere to land.
        let (ax, ay) = (3isize, 2isize);
        let rows = h + ay as usize + 2;
        let mut want = plane(stride, rows, 91);
        let mut got = want.clone();

        for (dst, f) in [(&mut want, scalar), (&mut got, simd)] {
            let s = plane(stride, rows, 0);
            let sc = PlaneCursor::new(&s[..], 0, stride).advance(ax, ay);
            let mut dc = PlaneCursorMut::new(&mut dst[..], 0, stride).advance(ax, ay);
            f(&mut dc, &sc);
        }

        assert_eq!(got, want, "{w}x{h} over stride {stride}");
    }

    #[test]
    fn copy_matches_the_scalar_slots() {
        for &stride in &[16usize, 24, 33, 64] {
            check(16, 16, stride.max(20), WelsCopy16x16_c, copy_16x16);
            check(16, 8, stride.max(20), WelsCopy16x8_c, copy_16x8);
            check(8, 16, stride.max(12), WelsCopy8x16_c, copy_8x16);
            check(8, 8, stride.max(12), WelsCopy8x8_c, copy_8x8);
        }
    }

    /// A stride equal to the block width leaves no gap between rows, so a kernel
    /// that over-wrote its row would corrupt the next one instead of landing in
    /// padding — the one geometry where an over-long store stays in bounds and
    /// still changes the answer.
    #[test]
    fn copy_is_exact_when_rows_are_contiguous() {
        check(16, 16, 16, WelsCopy16x16_c, copy_16x16);
        check(8, 8, 8, WelsCopy8x8_c, copy_8x8);
    }

    /// The two operands need not share a stride: `WelsMdBackgroundMbEnc` hands
    /// `pfCopy16x16Aligned` a picture plane as the destination and a stride-16
    /// scratch array as the source.
    #[test]
    fn copy_walks_each_operand_on_its_own_stride() {
        for &(dw, sw) in &[(64usize, 16usize), (16, 64), (33, 16), (16, 16)] {
            for (w, h, scalar, simd) in [
                (
                    16usize,
                    16usize,
                    WelsCopy16x16_c as fn(&mut PlaneCursorMut<'_>, &PlaneCursor<'_>),
                    copy_16x16 as fn(&mut PlaneCursorMut<'_>, &PlaneCursor<'_>),
                ),
                (16, 8, WelsCopy16x8_c, copy_16x8),
                (8, 16, WelsCopy8x16_c, copy_8x16),
                (8, 8, WelsCopy8x8_c, copy_8x8),
            ] {
                if dw < w || sw < w {
                    continue;
                }
                let mut want = plane(dw, h + 2, 91);
                let mut got = want.clone();
                for (dst, f) in [(&mut want, scalar), (&mut got, simd)] {
                    let s = plane(sw, h + 2, 0);
                    let sc = PlaneCursor::new(&s[..], 0, sw);
                    let mut dc = PlaneCursorMut::new(&mut dst[..], 0, dw);
                    f(&mut dc, &sc);
                }
                assert_eq!(got, want, "{w}x{h}, dst stride {dw}, src stride {sw}");
            }
        }
    }

    /// `span_mut` validates `(H - 1) * stride + W` bytes up front. This
    /// block overruns its buffer by less than one row's width, which a span missing
    /// the last row's width would accept.
    #[test]
    #[should_panic(expected = "out of range")]
    fn copy_rejects_a_block_whose_last_row_overruns() {
        let src = vec![0u8; 16 * 16];
        // 250 < the block's 256-byte span, but >= the 240 bytes a span missing the
        // last row's width would ask for.
        let mut dst = vec![0u8; 250];
        let s = PlaneCursor::new(&src[..], 0, 16);
        let mut d = PlaneCursorMut::new(&mut dst[..], 0, 16);
        copy_16x16(&mut d, &s);
    }

    /// `span_mut` turns an out-of-range block into a panic rather than
    /// reading or writing past the plane.
    #[test]
    #[should_panic(expected = "out of range")]
    fn copy_panics_rather_than_running_off_the_plane() {
        let mut buf = vec![0u8; 16 * 16];
        let src = vec![0u8; 16 * 16];
        let s = PlaneCursor::new(&src, 0, 16);
        let mut d = PlaneCursorMut::new(&mut buf, 0, 16).advance(0, 8);
        copy_16x16(&mut d, &s);
    }

    #[test]
    fn copy_accepts_tight_span_allocation() {
        let stride = 64usize;
        let len16 = 15 * stride + 16;
        let src16: Vec<u8> = (0..len16).map(|i| (i as u8).wrapping_mul(13)).collect();
        let mut dst16 = vec![0u8; len16];
        copy_16x16(
            &mut PlaneCursorMut::new(&mut dst16, 0, stride),
            &PlaneCursor::new(&src16, 0, stride),
        );
        for y in 0..16 {
            assert_eq!(
                &dst16[y * stride..y * stride + 16],
                &src16[y * stride..y * stride + 16]
            );
        }

        let len8 = 7 * stride + 8;
        let src8: Vec<u8> = (0..len8).map(|i| (i as u8).wrapping_mul(19)).collect();
        let mut dst8 = vec![0u8; len8];
        copy_8x8(
            &mut PlaneCursorMut::new(&mut dst8, 0, stride),
            &PlaneCursor::new(&src8, 0, stride),
        );
        for y in 0..8 {
            assert_eq!(
                &dst8[y * stride..y * stride + 8],
                &src8[y * stride..y * stride + 8]
            );
        }
    }
}
