// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

#![forbid(unsafe_code)]
//! Scalar forwards for the `copy` kernels — see the module header.

use crate::encoder::encode_mb_aux::{
    WelsCopy8x8_c, WelsCopy8x16_c, WelsCopy16x8_c, WelsCopy16x16_c,
};
use crate::safe::plane::{PlaneCursor, PlaneCursorMut};

#[inline(always)]
pub fn copy_16x16(dst: &mut PlaneCursorMut<'_>, src: &PlaneCursor<'_>) {
    WelsCopy16x16_c(dst, src)
}

#[inline(always)]
pub fn copy_16x8(dst: &mut PlaneCursorMut<'_>, src: &PlaneCursor<'_>) {
    WelsCopy16x8_c(dst, src)
}

#[inline(always)]
pub fn copy_8x16(dst: &mut PlaneCursorMut<'_>, src: &PlaneCursor<'_>) {
    WelsCopy8x16_c(dst, src)
}

#[inline(always)]
pub fn copy_8x8(dst: &mut PlaneCursorMut<'_>, src: &PlaneCursor<'_>) {
    WelsCopy8x8_c(dst, src)
}

#[inline(always)]
pub fn copy_16x16_slice(dst: &mut PlaneCursorMut<'_>, src: &[u8]) {
    let s = &src[..16 * 16];
    for y in 0..16 {
        dst.row_mut(y as isize, 0, 16)
            .copy_from_slice(&s[y * 16..][..16]);
    }
}

#[inline(always)]
pub fn copy_8x8_slice(dst: &mut PlaneCursorMut<'_>, src: &[u8]) {
    let s = &src[..8 * 8];
    for y in 0..8 {
        dst.row_mut(y as isize, 0, 8)
            .copy_from_slice(&s[y * 8..][..8]);
    }
}
