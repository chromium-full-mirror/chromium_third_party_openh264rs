// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Integration test for encoder initialization and parameter set generation.

use openh264rs::api::codec_api::*;

#[test]
pub(crate) fn test_simple_encode_init_and_encode_param_sets() {
    let mut enc = Encoder::new();

    let mut param = SEncParamBase::default();
    param.iPicWidth = 160;
    param.iPicHeight = 120;
    param.fMaxFrameRate = 30.0;
    param.iTargetBitrate = 250000;
    param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;

    let init_ret = enc.initialize(&param);
    assert_eq!(init_ret, CM_RESULT_SUCCESS);

    let mut bs_info = SFrameBSInfo::default();
    let enc_param_ret = enc.encode_parameter_sets(&mut bs_info);
    assert_eq!(enc_param_ret, CM_RESULT_SUCCESS);

    let uninit_ret = enc.uninitialize();
    assert_eq!(uninit_ret, CM_RESULT_SUCCESS);
}
