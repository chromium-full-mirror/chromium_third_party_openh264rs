// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Fixed-shape macroblock copies for the aarch64 kernel table
//! (`WelsCopy*_AArch64_neon`, `codec/common/arm64/copy_mb_aarch64_neon.S`).
//!
//! Each kernel cuts the block's span once per operand through [`RefSamples::span`] —
//! one bounds check per operand rather than per row — and walks rows via
//! [`CellSpan::split_cell_rows`](crate::encoder::rec_view::CellSpan::split_cell_rows)
//! so LLVM lowers the constant-width cell loops directly to unaligned NEON loads and
//! stores with zero `unsafe` blocks.
#![forbid(unsafe_code)]

use crate::encoder::rec_view::RecCursor;
use crate::safe::plane::{BlockRows, RefSamples};

#[cold]
#[inline(never)]
fn copy_narrow_stride<const W: usize, const H: usize>(dst: &RecCursor<'_>, src: &RecCursor<'_>) {
    let s = RefSamples::span::<W, H>(src, 0, 0);
    let d = RefSamples::span::<W, H>(dst, 0, 0);
    for y in 0..H {
        d.set_row::<W>(y, &s.row::<W>(y, 0));
    }
}

#[cold]
#[inline(never)]
fn copy_slice_narrow_stride<const W: usize, const H: usize>(dst: &RecCursor<'_>, src: &[u8]) {
    let d = RefSamples::span::<W, H>(dst, 0, 0);
    let (rows, _) = src[..W * H].as_chunks::<W>();
    for (y, r) in rows.iter().enumerate() {
        d.set_row::<W>(y, r);
    }
}

/// Copies `H` rows of `W` bytes from `src` to `dst`, each walking its own stride.
///
/// Two strides rather than one because `WelsMdBackgroundMbEnc` copies the
/// mode-decision scratch — a stride-16 array — into a picture plane.
///
/// Overlapping operands in the same `SharedPlane` behave identically to
/// `copy_rows_shared`: each row `y` (`0..H`) is read into `[u8; W]` immediately before
/// row `y` of `dst` is written.
#[inline(always)]
fn copy_block<const W: usize, const H: usize>(dst: &RecCursor<'_>, src: &RecCursor<'_>) {
    let s = RefSamples::span::<W, H>(src, 0, 0);
    let d = RefSamples::span::<W, H>(dst, 0, 0);
    if let (Some((sh, st)), Some((dh, dt))) =
        (s.split_cell_rows::<W, H>(), d.split_cell_rows::<W, H>())
    {
        for (sc, dc) in sh.zip(dh) {
            let row: [u8; W] = std::array::from_fn(|i| sc[i].get());
            for (c, b) in dc.iter().zip(row) {
                c.set(b);
            }
        }
        let row: [u8; W] = std::array::from_fn(|i| st[i].get());
        for (c, b) in dt.iter().zip(row) {
            c.set(b);
        }
    } else {
        copy_narrow_stride::<W, H>(dst, src);
    }
}

/// Copies a packed `W * H` slice from `src` into `dst`.
#[inline(always)]
fn copy_slice<const W: usize, const H: usize>(dst: &RecCursor<'_>, src: &[u8]) {
    let (rows, _) = src[..W * H].as_chunks::<W>();
    let Some((last_row, head_rows)) = rows.split_last() else {
        return;
    };
    let d = RefSamples::span::<W, H>(dst, 0, 0);
    if let Some((dh, dt)) = d.split_cell_rows::<W, H>() {
        for (sr, dc) in head_rows.iter().zip(dh) {
            let row = *sr;
            for (c, b) in dc.iter().zip(row) {
                c.set(b);
            }
        }
        let row = *last_row;
        for (c, b) in dt.iter().zip(row) {
            c.set(b);
        }
    } else {
        copy_slice_narrow_stride::<W, H>(dst, src);
    }
}

/// `WelsCopy16x16_AArch64_neon` and `WelsCopy16x16NotAligned_AArch64_neon`.
#[inline(always)]
pub fn copy_16x16(dst: &RecCursor<'_>, src: &RecCursor<'_>) {
    copy_block::<16, 16>(dst, src);
}

/// `WelsCopy16x8NotAligned_AArch64_neon`.
#[inline(always)]
pub fn copy_16x8(dst: &RecCursor<'_>, src: &RecCursor<'_>) {
    copy_block::<16, 8>(dst, src);
}

/// `WelsCopy8x16_AArch64_neon`.
#[inline(always)]
pub fn copy_8x16(dst: &RecCursor<'_>, src: &RecCursor<'_>) {
    copy_block::<8, 16>(dst, src);
}

/// `WelsCopy8x8_AArch64_neon`.
#[inline(always)]
pub fn copy_8x8(dst: &RecCursor<'_>, src: &RecCursor<'_>) {
    copy_block::<8, 8>(dst, src);
}

/// Copies a packed 16x16 block from `src` into `dst`.
#[inline(always)]
pub fn copy_16x16_slice(dst: &RecCursor<'_>, src: &[u8]) {
    copy_slice::<16, 16>(dst, src);
}

/// Copies a packed 8x8 block from `src` into `dst`.
#[inline(always)]
pub fn copy_8x8_slice(dst: &RecCursor<'_>, src: &[u8]) {
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

    /// A plane of `stride * rows` distinct-ish bytes, as cells.
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
        scalar: fn(&RecCursor, &RecCursor),
        simd: fn(&RecCursor, &RecCursor),
    ) {
        // Two spare rows below the block and an anchor off (0, 0), so a kernel that
        // ignored the anchor or ran a row long has somewhere to land.
        let (ax, ay) = (3isize, 2isize);
        let rows = h + ay as usize + 2;
        let mut want = plane(stride, rows, 91);
        let mut got = want.clone();

        for (dst, f) in [(&mut want, scalar), (&mut got, simd)] {
            let mut s = plane(stride, rows, 0);
            let sc = RecCursor::over_owned(&mut s[..], 0, stride).advance(ax, ay);
            let dc = RecCursor::over_owned(&mut dst[..], 0, stride).advance(ax, ay);
            f(&dc, &sc);
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
                    WelsCopy16x16_c as fn(&RecCursor, &RecCursor),
                    copy_16x16 as fn(&RecCursor, &RecCursor),
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
                    let mut s = plane(sw, h + 2, 0);
                    let sc = RecCursor::over_owned(&mut s[..], 0, sw);
                    let dc = RecCursor::over_owned(&mut dst[..], 0, dw);
                    f(&dc, &sc);
                }
                assert_eq!(got, want, "{w}x{h}, dst stride {dw}, src stride {sw}");
            }
        }
    }

    /// `RefSamples::span` validates `(H - 1) * stride + W` bytes up front. This
    /// block overruns its buffer by less than one row's width, which a span missing
    /// the last row's width would accept.
    #[test]
    #[should_panic(expected = "out of range")]
    fn copy_rejects_a_block_whose_last_row_overruns() {
        let mut src = vec![0u8; 16 * 16];
        // 250 < the block's 256-byte span, but >= the 240 bytes a span missing the
        // last row's width would ask for.
        let mut dst = vec![0u8; 250];
        let s = RecCursor::over_owned(&mut src[..], 0, 16);
        let d = RecCursor::over_owned(&mut dst[..], 0, 16);
        copy_16x16(&d, &s);
    }

    /// `RefSamples::span` turns an out-of-range block into a panic rather than
    /// reading or writing past the plane.
    #[test]
    #[should_panic(expected = "out of range")]
    fn copy_panics_rather_than_running_off_the_plane() {
        let mut buf = vec![0u8; 16 * 16];
        let mut src = vec![0u8; 16 * 16];
        let s = RecCursor::over_owned(&mut src, 0, 16);
        let d = RecCursor::over_owned(&mut buf, 0, 16).advance(0, 8);
        copy_16x16(&d, &s);
    }

    #[test]
    fn copy_matches_scalar_on_same_plane_overlap() {
        let stride = 32usize;
        for &(dx, dy) in &[(2isize, 0isize), (-2, 0), (0, 1), (0, -1), (2, 1), (-1, 2)] {
            for (name, scalar, simd) in [
                (
                    "16x16",
                    WelsCopy16x16_c as fn(&RecCursor, &RecCursor),
                    copy_16x16 as fn(&RecCursor, &RecCursor),
                ),
                ("16x8", WelsCopy16x8_c, copy_16x8),
                ("8x16", WelsCopy8x16_c, copy_8x16),
                ("8x8", WelsCopy8x8_c, copy_8x8),
            ] {
                let mut want = plane(stride, 24, 17);
                let mut got = want.clone();
                for (buf, f) in [(&mut want, scalar), (&mut got, simd)] {
                    let base = RecCursor::over_owned(&mut buf[..], 0, stride).advance(4, 4);
                    let dc = base.advance(dx, dy);
                    f(&dc, &base);
                }
                assert_eq!(got, want, "{name} overlap ({dx}, {dy})");
            }
        }
    }

    #[test]
    fn copy_accepts_tight_span_allocation() {
        let stride = 64usize;
        let len16 = 15 * stride + 16;
        let mut src16: Vec<u8> = (0..len16).map(|i| (i as u8).wrapping_mul(13)).collect();
        let mut dst16 = vec![0u8; len16];
        copy_16x16(
            &RecCursor::over_owned(&mut dst16, 0, stride),
            &RecCursor::over_owned(&mut src16, 0, stride),
        );
        for y in 0..16 {
            assert_eq!(
                &dst16[y * stride..y * stride + 16],
                &src16[y * stride..y * stride + 16]
            );
        }

        let len8 = 7 * stride + 8;
        let mut src8: Vec<u8> = (0..len8).map(|i| (i as u8).wrapping_mul(19)).collect();
        let mut dst8 = vec![0u8; len8];
        copy_8x8(
            &RecCursor::over_owned(&mut dst8, 0, stride),
            &RecCursor::over_owned(&mut src8, 0, stride),
        );
        for y in 0..8 {
            assert_eq!(
                &dst8[y * stride..y * stride + 8],
                &src8[y * stride..y * stride + 8]
            );
        }
    }
}
