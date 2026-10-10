// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Integration test verifying that decoder initialization accepts multi-threaded
//! configuration parameters (`uiCpuLoad > 1`) on this single-threaded port.

use openh264rs::api::codec_api::*;

#[test]
fn test_multithread_decoder_initialization() {
    unsafe {
        let mut p_decoder: *mut ISVCDecoder = std::ptr::null_mut();
        let ret = WelsCreateDecoder(&mut p_decoder);
        assert_eq!(ret, CM_RESULT_SUCCESS as i64);
        assert!(!p_decoder.is_null());

        let mut param = SDecodingParam::default();
        param.uiTargetDqLayer = u8::MAX;
        param.uiCpuLoad = 2; // Request 2 worker threads

        let init_ret = ISVCDecoder::Initialize(p_decoder, &param as *const SDecodingParam);
        assert_eq!(init_ret, CM_RESULT_SUCCESS as i64);

        let uninit_ret = ISVCDecoder::Uninitialize(p_decoder);
        assert_eq!(uninit_ret, CM_RESULT_SUCCESS as i64);

        WelsDestroyDecoder(p_decoder);
    }
}
