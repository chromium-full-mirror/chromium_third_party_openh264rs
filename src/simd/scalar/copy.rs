// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

#![forbid(unsafe_code)]
//! Scalar forwards for the `copy` kernels — see the module header.

use crate::encoder::encode_mb_aux::{
    WelsCopy8x8_c, WelsCopy8x16_c, WelsCopy16x8_c, WelsCopy16x16_c,
};
use crate::encoder::rec_view::RecCursor;

#[inline(always)]
pub fn copy_16x16(dst: &RecCursor<'_>, src: &RecCursor<'_>) {
    WelsCopy16x16_c(dst, src)
}

#[inline(always)]
pub fn copy_16x8(dst: &RecCursor<'_>, src: &RecCursor<'_>) {
    WelsCopy16x8_c(dst, src)
}

#[inline(always)]
pub fn copy_8x16(dst: &RecCursor<'_>, src: &RecCursor<'_>) {
    WelsCopy8x16_c(dst, src)
}

#[inline(always)]
pub fn copy_8x8(dst: &RecCursor<'_>, src: &RecCursor<'_>) {
    WelsCopy8x8_c(dst, src)
}

#[inline(always)]
pub fn copy_16x16_slice(dst: &RecCursor<'_>, src: &[u8]) {
    let s = &src[..16 * 16];
    for y in 0..16 {
        let row: &[u8; 16] = s[y * 16..][..16].try_into().expect("16 bytes");
        dst.write_row::<16>(y as isize, 0, row);
    }
}

#[inline(always)]
pub fn copy_8x8_slice(dst: &RecCursor<'_>, src: &[u8]) {
    let s = &src[..8 * 8];
    for y in 0..8 {
        let row: &[u8; 8] = s[y * 8..][..8].try_into().expect("8 bytes");
        dst.write_row::<8>(y as isize, 0, row);
    }
}
