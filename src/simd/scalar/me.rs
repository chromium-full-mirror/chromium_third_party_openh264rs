// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

#![forbid(unsafe_code)]
//! Scalar forwards for the motion-estimation feature kernels.

use crate::encoder::svc_motion_estimate;
use crate::safe::plane::PlaneCursor;

#[inline(always)]
pub fn sum_of_8x8_single_block(cRef: &PlaneCursor<'_>) -> i32 {
    svc_motion_estimate::sum_of_8x8_single_block(cRef)
}

#[inline(always)]
pub fn sum_of_16x16_single_block(cRef: &PlaneCursor<'_>) -> i32 {
    svc_motion_estimate::sum_of_16x16_single_block(cRef)
}

#[inline(always)]
pub fn sum_of_8x8_block_of_frame(
    kpRefPicture: &[u8],
    kiWidth: i32,
    kiHeight: i32,
    kiRefStride: i32,
    pFeatureOfBlock: &mut [u16],
    pTimesOfFeatureValue: &mut [u32],
) {
    svc_motion_estimate::SumOf8x8BlockOfFrame_c(
        kpRefPicture,
        kiWidth,
        kiHeight,
        kiRefStride,
        pFeatureOfBlock,
        pTimesOfFeatureValue,
    );
}

#[inline(always)]
pub fn sum_of_16x16_block_of_frame(
    kpRefPicture: &[u8],
    kiWidth: i32,
    kiHeight: i32,
    kiRefStride: i32,
    pFeatureOfBlock: &mut [u16],
    pTimesOfFeatureValue: &mut [u32],
) {
    svc_motion_estimate::SumOf16x16BlockOfFrame_c(
        kpRefPicture,
        kiWidth,
        kiHeight,
        kiRefStride,
        pFeatureOfBlock,
        pTimesOfFeatureValue,
    );
}

#[inline(always)]
pub fn sea_row_half_sums<const N: usize>(c: &PlaneCursor<'_>, a0: &mut [u16], a1: &mut [u16]) {
    svc_motion_estimate::sea_row_half_sums_c::<N>(c, a0, a1);
}

#[inline(always)]
pub fn sea_col_half_sums<const N: usize>(c: &PlaneCursor<'_>, a0: &mut [u16], a1: &mut [u16]) {
    svc_motion_estimate::sea_col_half_sums_c::<N>(c, a0, a1);
}

#[inline(always)]
pub fn sea_bounds(
    w0: &[u16],
    w1: &[u16],
    half: usize,
    e: [u16; 4],
    iFixedMvd: u16,
    mvd: &[u16],
    bound: &mut [u16],
) {
    svc_motion_estimate::sea_bounds_c(w0, w1, half, e, iFixedMvd, mvd, bound);
}
