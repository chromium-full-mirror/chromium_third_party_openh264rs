// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Decoder error-concealment modes — `test/api/decoder_ec_test.cpp`.

#[path = "common/mod.rs"]
mod common;

use openh264rs::api::codec_api::*;

#[test]
pub(crate) fn test_decoder_error_concealment_modes() {
    let ec_modes = [
        ERROR_CON_IDC::ERROR_CON_DISABLE,
        ERROR_CON_IDC::ERROR_CON_FRAME_COPY,
        ERROR_CON_IDC::ERROR_CON_SLICE_COPY,
        ERROR_CON_IDC::ERROR_CON_FRAME_COPY_CROSS_IDR,
        ERROR_CON_IDC::ERROR_CON_SLICE_COPY_CROSS_IDR,
        ERROR_CON_IDC::ERROR_CON_SLICE_COPY_CROSS_IDR_FREEZE_RES_CHANGE,
        ERROR_CON_IDC::ERROR_CON_SLICE_MV_COPY_CROSS_IDR,
        ERROR_CON_IDC::ERROR_CON_SLICE_MV_COPY_CROSS_IDR_FREEZE_RES_CHANGE,
    ];

    for &ec_mode in &ec_modes {
        let mut decoder = Decoder::new();

        let mut param = SDecodingParam::default();
        param.uiTargetDqLayer = u8::MAX;
        param.eEcActiveIdc = ec_mode;

        let init_ret = decoder.initialize(&param);
        assert_eq!(init_ret, CM_RESULT_SUCCESS as std::ffi::c_long);

        assert_eq!(decoder.error_concealment(), Some(ec_mode));

        let uninit_ret = decoder.uninitialize();
        assert_eq!(uninit_ret, CM_RESULT_SUCCESS as std::ffi::c_long);
    }
}

/// Decodes `data` and returns the OR of every `DecodeFrame2` state, optionally
/// switching the concealment mode through `set_error_concealment` after `Initialize` — the one
/// write the public API makes into the parameter block after configuration.
fn decode_states(data: &[u8], init_ec: ERROR_CON_IDC, switch_to: Option<ERROR_CON_IDC>) -> i32 {
    let mut decoder = Decoder::new();

    let mut param = SDecodingParam::default();
    param.uiTargetDqLayer = u8::MAX;
    param.eEcActiveIdc = init_ec;
    param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
    assert_eq!(
        decoder.initialize(&param),
        CM_RESULT_SUCCESS as std::ffi::c_long
    );

    if let Some(mode) = switch_to {
        decoder.set_error_concealment(mode as i32);
    }

    let mut states = 0i32;
    for unit in openh264rs::split_annexb_units(data) {
        let mut buf_info = SBufferInfo::default();
        let ret = decoder.decode(Some(unit), &mut buf_info);
        states |= ret.0;
    }
    decoder.uninitialize();
    states
}

/// The block the api writes and the block the decoder reads are one block: the
/// context's `SDecodingParam`, which `SetOption(DECODER_OPTION_ERROR_CON_IDC)` writes
/// at `welsDecoderExt.cpp:535`.
///
/// Concealment is switched off after `Initialize` on a stream that conceals:
///
/// * initialised with `ERROR_CON_SLICE_COPY`, `BA_MW_D_IDR_LOST.264` comes back with
///   `dsDataErrorConcealed` set;
/// * switched to `ERROR_CON_DISABLE` after the same init, it does not, and reports
///   `dsBitstreamError` instead.
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_error_con_idc_set_after_initialize_reaches_the_decoder() {
    let data = common::read_stream("BA_MW_D_IDR_LOST.264");

    const CONCEALED: i32 = 0x20; // dsDataErrorConcealed

    let concealing = decode_states(&data, ERROR_CON_IDC::ERROR_CON_SLICE_COPY, None);
    assert_ne!(
        concealing & CONCEALED,
        0,
        "the asset did not conceal at all — this test no longer covers the behaviour it checks"
    );

    let switched = decode_states(
        &data,
        ERROR_CON_IDC::ERROR_CON_SLICE_COPY,
        Some(ERROR_CON_IDC::ERROR_CON_DISABLE),
    );
    assert_eq!(
        switched & CONCEALED,
        0,
        "concealment still ran after SetOption(ERROR_CON_IDC = DISABLE): the api wrote one \
         parameter block and the decoder read another"
    );
}
