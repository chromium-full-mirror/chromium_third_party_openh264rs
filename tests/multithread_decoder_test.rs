// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Integration test verifying that decoder initialization accepts multi-threaded
//! configuration parameters (`uiCpuLoad > 1`) on this single-threaded port.

use openh264rs::api::codec_api::*;

#[test]
pub(crate) fn test_multithread_decoder_initialization() {
    let mut decoder = Decoder::new();

    let mut param = SDecodingParam::default();
    param.uiTargetDqLayer = u8::MAX;
    param.uiCpuLoad = 2; // Request 2 worker threads

    let init_ret = decoder.initialize(&param);
    assert_eq!(init_ret, CM_RESULT_SUCCESS as std::ffi::c_long);

    let uninit_ret = decoder.uninitialize();
    assert_eq!(uninit_ret, CM_RESULT_SUCCESS as std::ffi::c_long);
}
