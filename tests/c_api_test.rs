// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Integration tests specifically exercising the raw C ABI (`feature = "c-api"`):
//! `WelsCreateSVCEncoder`, `WelsDestroySVCEncoder`, `WelsCreateDecoder`,
//! `WelsDestroyDecoder`, `WelsGetDecoderCapability`, `WelsGetCodecVersion`,
//! `WelsGetCodecVersionEx`, vtable dispatch, and null-pointer handling.

use openh264rs::api::c_api::*;
use openh264rs::api::codec_api::*;
use openh264rs::api::version::{WelsGetCodecVersion, WelsGetCodecVersionEx};
use std::ffi::c_long;

#[test]
fn test_c_api_decoder_vtable_lifecycle_and_null_handling() {
    unsafe {
        assert_ne!(
            WelsCreateDecoder(std::ptr::null_mut()),
            CM_RESULT_SUCCESS as c_long
        );

        let mut p_decoder: *mut ISVCDecoder = std::ptr::null_mut();
        let ret = WelsCreateDecoder(&mut p_decoder);
        assert_eq!(ret, CM_RESULT_SUCCESS as c_long);
        assert!(!p_decoder.is_null());

        assert_ne!(
            ISVCDecoder::Initialize(p_decoder, std::ptr::null()),
            CM_RESULT_SUCCESS as c_long
        );

        let mut param = SDecodingParam::default();
        param.uiTargetDqLayer = u8::MAX;
        assert_eq!(
            ISVCDecoder::Initialize(p_decoder, &param as *const SDecodingParam),
            CM_RESULT_SUCCESS as c_long
        );

        let mut p_dst: [*mut u8; 3] = [std::ptr::null_mut(); 3];
        let mut stride: [i32; 2] = [0; 2];
        let mut width: i32 = 0;
        let mut height: i32 = 0;
        let dec_state = ISVCDecoder::DecodeFrame(
            p_decoder,
            std::ptr::null(),
            0,
            p_dst.as_mut_ptr(),
            stride.as_mut_ptr(),
            &mut width,
            &mut height,
        );
        assert_eq!(dec_state, DECODING_STATE::dsErrorFree);

        let mut buf_info = SBufferInfo::default();
        let dec_nodelay_state = ISVCDecoder::DecodeFrameNoDelay(
            p_decoder,
            std::ptr::null(),
            0,
            p_dst.as_mut_ptr(),
            &mut buf_info,
        );
        assert_eq!(dec_nodelay_state, DECODING_STATE::dsErrorFree);

        let dec2_state = ISVCDecoder::DecodeFrame2(
            p_decoder,
            std::ptr::null(),
            0,
            p_dst.as_mut_ptr(),
            &mut buf_info,
        );
        assert_eq!(dec2_state, DECODING_STATE::dsErrorFree);

        let flush_state = ISVCDecoder::FlushFrame(p_decoder, p_dst.as_mut_ptr(), &mut buf_info);
        assert_eq!(flush_state, DECODING_STATE::dsErrorFree);

        let mut parser_info = SParserBsInfo::default();
        let parse_state =
            ISVCDecoder::DecodeParser(p_decoder, std::ptr::null(), 0, &mut parser_info);
        assert_eq!(parse_state, DECODING_STATE::dsInvalidArgument);

        let mut dst_len: i32 = 0;
        let mut color_fmt: i32 = 0;
        let dec_ex_state = ISVCDecoder::DecodeFrameEx(
            p_decoder,
            std::ptr::null(),
            0,
            std::ptr::null_mut(),
            0,
            &mut dst_len,
            &mut width,
            &mut height,
            &mut color_fmt,
        );
        assert_eq!(dec_ex_state, DECODING_STATE::dsErrorFree);

        let mut trace_level = 0i32;
        assert_eq!(
            ISVCDecoder::SetOption(
                p_decoder,
                DECODER_OPTION::DECODER_OPTION_TRACE_LEVEL,
                &mut trace_level as *mut i32 as *mut std::ffi::c_void,
            ),
            CM_RESULT_SUCCESS as c_long
        );
        assert_eq!(
            ISVCDecoder::GetOption(
                p_decoder,
                DECODER_OPTION::DECODER_OPTION_TRACE_LEVEL,
                &mut trace_level as *mut i32 as *mut std::ffi::c_void,
            ),
            CM_INIT_PARA_ERROR as c_long
        );

        assert_eq!(
            ISVCDecoder::Uninitialize(p_decoder),
            CM_RESULT_SUCCESS as c_long
        );
        WelsDestroyDecoder(p_decoder);
    }
}

#[test]
fn test_c_api_encoder_vtable_lifecycle_and_null_handling() {
    unsafe {
        assert_ne!(
            WelsCreateSVCEncoder(std::ptr::null_mut()),
            CM_RESULT_SUCCESS
        );

        let mut p_encoder: *mut ISVCEncoder = std::ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut p_encoder), CM_RESULT_SUCCESS);
        assert!(!p_encoder.is_null());

        assert_ne!(
            ISVCEncoder::Initialize(p_encoder, std::ptr::null()),
            CM_RESULT_SUCCESS
        );

        let mut param = SEncParamBase::default();
        param.iPicWidth = 320;
        param.iPicHeight = 240;
        param.fMaxFrameRate = 30.0;
        param.iTargetBitrate = 500000;
        param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
        assert_eq!(
            ISVCEncoder::Initialize(p_encoder, &param as *const SEncParamBase),
            CM_RESULT_SUCCESS
        );

        let mut default_param = SEncParamExt::default();
        assert_eq!(
            ISVCEncoder::GetDefaultParams(p_encoder, &mut default_param as *mut SEncParamExt),
            CM_RESULT_SUCCESS
        );

        let mut invalid_fmt: i32 = 0;
        assert_eq!(
            ISVCEncoder::SetOption(
                p_encoder,
                ENCODER_OPTION::ENCODER_OPTION_DATAFORMAT,
                (&mut invalid_fmt as *mut i32).cast()
            ),
            CM_INIT_PARA_ERROR
        );

        let mut out_of_range_strategy: i32 = 99;
        assert_eq!(
            ISVCEncoder::SetOption(
                p_encoder,
                ENCODER_OPTION::ENCODER_OPTION_SPS_PPS_ID_STRATEGY,
                (&mut out_of_range_strategy as *mut i32).cast()
            ),
            CM_RESULT_SUCCESS
        );

        assert_eq!(ISVCEncoder::Uninitialize(p_encoder), CM_RESULT_SUCCESS);
        WelsDestroySVCEncoder(p_encoder);
    }
}

#[test]
fn test_c_api_capability_and_version_exports() {
    unsafe {
        assert_eq!(
            WelsGetDecoderCapability(std::ptr::null_mut()),
            CM_INIT_PARA_ERROR
        );
        let mut cap = SDecoderCapability::default();
        assert_eq!(WelsGetDecoderCapability(&mut cap), CM_RESULT_SUCCESS);
        assert_eq!(cap, Decoder::capability());

        let v = WelsGetCodecVersion();
        let mut v_ex = OpenH264Version {
            uMajor: 0,
            uMinor: 0,
            uRevision: 0,
            uReserved: 0,
        };
        WelsGetCodecVersionEx(&mut v_ex);
        assert_eq!(v, v_ex);
    }
}

/// Tests that out-of-range error concealment mode (`eEcActiveIdc`) values passed by a C caller
/// in `SDecodingParam` are clamped at `ISVCDecoder::Initialize`.
#[test]
fn test_error_con_idc_out_of_range_is_clamped_at_initialize() {
    let mut param = SDecodingParam::default();
    param.uiTargetDqLayer = u8::MAX;
    param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
    unsafe {
        for (raw, want) in [(99i32, 7i32), (-3, 0), (7, 7), (0, 0), (2, 2)] {
            let mut p_decoder: *mut ISVCDecoder = std::ptr::null_mut();
            assert_eq!(
                WelsCreateDecoder(&mut p_decoder),
                CM_RESULT_SUCCESS as c_long
            );

            let mut buf = std::mem::MaybeUninit::<SDecodingParam>::uninit();
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref(&param).cast::<u8>(),
                buf.as_mut_ptr().cast::<u8>(),
                std::mem::size_of::<SDecodingParam>(),
            );
            std::ptr::addr_of_mut!((*buf.as_mut_ptr()).eEcActiveIdc)
                .cast::<i32>()
                .write(raw);
            assert_eq!(
                ISVCDecoder::Initialize(p_decoder, buf.as_ptr()),
                CM_RESULT_SUCCESS as c_long
            );

            let mut v = -1i32;
            assert_eq!(
                ISVCDecoder::GetOption(
                    p_decoder,
                    DECODER_OPTION::DECODER_OPTION_ERROR_CON_IDC,
                    std::ptr::from_mut(&mut v).cast(),
                ),
                CM_RESULT_SUCCESS as c_long
            );
            assert_eq!(v, want, "eEcActiveIdc = {raw} was not clamped to {want}");

            assert_eq!(
                ISVCDecoder::Uninitialize(p_decoder),
                CM_RESULT_SUCCESS as c_long
            );
            WelsDestroyDecoder(p_decoder);
        }
    }
}

#[test]
fn test_c_api_encoder_options_null_and_invalid_raw_values() {
    unsafe {
        let mut p_encoder: *mut ISVCEncoder = std::ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut p_encoder), CM_RESULT_SUCCESS);
        let set_opt = (*(*p_encoder).lpVtbl).SetOption;
        let get_opt = (*(*p_encoder).lpVtbl).GetOption;

        let mut trace_level = 0i32;
        set_opt(
            p_encoder,
            ENCODER_OPTION::ENCODER_OPTION_TRACE_LEVEL as i32,
            std::ptr::from_mut(&mut trace_level).cast(),
        );

        let param = SEncParamBase {
            iUsageType: EUsageType::CAMERA_VIDEO_REAL_TIME,
            iPicWidth: 320,
            iPicHeight: 240,
            iTargetBitrate: 500_000,
            iRCMode: RC_MODES::RC_BITRATE_MODE,
            fMaxFrameRate: 30.0,
        };
        assert_eq!(
            ISVCEncoder::Initialize(p_encoder, &param),
            CM_RESULT_SUCCESS
        );

        // Null pointer for every option ID (0..32) and out-of-range option IDs (-1, 32).
        let mut dummy = 0i32;
        let dummy_ptr = std::ptr::from_mut(&mut dummy).cast();
        for id in -1..=32 {
            assert_eq!(
                set_opt(p_encoder, id, std::ptr::null_mut()),
                CM_INIT_PARA_ERROR
            );
            assert_eq!(
                get_opt(p_encoder, id, std::ptr::null_mut()),
                CM_INIT_PARA_ERROR
            );
        }
        assert_eq!(set_opt(p_encoder, -1, dummy_ptr), CM_INIT_PARA_ERROR);
        assert_eq!(set_opt(p_encoder, 32, dummy_ptr), CM_INIT_PARA_ERROR);
        assert_eq!(get_opt(p_encoder, -1, dummy_ptr), CM_INIT_PARA_ERROR);
        assert_eq!(get_opt(p_encoder, 32, dummy_ptr), CM_INIT_PARA_ERROR);

        // Non-zero raw bool bytes are treated as true without UB.
        for opt in [
            ENCODER_OPTION::ENCODER_OPTION_RC_FRAME_SKIP,
            ENCODER_OPTION::ENCODER_OPTION_IS_LOSSLESS_LINK,
            ENCODER_OPTION::ENCODER_OPTION_ENABLE_SSEI,
            ENCODER_OPTION::ENCODER_OPTION_ENABLE_PREFIX_NAL_ADDING,
        ] {
            let mut raw_bool: u8 = 2;
            assert_eq!(
                ISVCEncoder::SetOption(p_encoder, opt, std::ptr::from_mut(&mut raw_bool).cast()),
                CM_RESULT_SUCCESS
            );
        }

        // Out-of-range raw SBitrateInfo.iLayer (-1, 99) is rejected.
        for bad_layer in [-1i32, 99] {
            let mut raw = [0i32; 2];
            raw[0] = bad_layer;
            raw[1] = 200_000;
            assert_eq!(
                ISVCEncoder::SetOption(
                    p_encoder,
                    ENCODER_OPTION::ENCODER_OPTION_BITRATE,
                    raw.as_mut_ptr().cast(),
                ),
                CM_INIT_PARA_ERROR
            );
            assert_eq!(
                ISVCEncoder::GetOption(
                    p_encoder,
                    ENCODER_OPTION::ENCODER_OPTION_BITRATE,
                    raw.as_mut_ptr().cast(),
                ),
                CM_INIT_PARA_ERROR
            );
        }

        // Set-only options reject GetOption.
        let mut rc_mode = 0i32;
        assert_eq!(
            ISVCEncoder::GetOption(
                p_encoder,
                ENCODER_OPTION::ENCODER_OPTION_RC_MODE,
                std::ptr::from_mut(&mut rc_mode).cast(),
            ),
            CM_INIT_PARA_ERROR
        );

        assert_eq!(ISVCEncoder::Uninitialize(p_encoder), CM_RESULT_SUCCESS);
        WelsDestroySVCEncoder(p_encoder);
    }
}

#[test]
fn test_c_api_decoder_options_null_and_invalid_raw_values() {
    unsafe {
        let mut p_decoder: *mut ISVCDecoder = std::ptr::null_mut();
        assert_eq!(
            WelsCreateDecoder(&mut p_decoder),
            CM_RESULT_SUCCESS as c_long
        );
        let set_opt = (*(*p_decoder).lpVtbl).SetOption;
        let get_opt = (*(*p_decoder).lpVtbl).GetOption;

        let mut trace_level = 0i32;
        set_opt(
            p_decoder,
            DECODER_OPTION::DECODER_OPTION_TRACE_LEVEL as i32,
            std::ptr::from_mut(&mut trace_level).cast(),
        );

        let mut param = SDecodingParam::default();
        param.uiTargetDqLayer = u8::MAX;
        assert_eq!(
            ISVCDecoder::Initialize(p_decoder, &param),
            CM_RESULT_SUCCESS as c_long
        );

        let mut dummy = 0i32;
        let dummy_ptr = std::ptr::from_mut(&mut dummy).cast();
        for id in 0..=20 {
            if id != DECODER_OPTION::DECODER_OPTION_NUM_OF_THREADS as i32 {
                assert_ne!(
                    set_opt(p_decoder, id, std::ptr::null_mut()),
                    CM_RESULT_SUCCESS as c_long
                );
            }
            assert_ne!(
                get_opt(p_decoder, id, std::ptr::null_mut()),
                CM_RESULT_SUCCESS as c_long
            );
        }
        assert_eq!(
            set_opt(p_decoder, 0, dummy_ptr),
            CM_INIT_PARA_ERROR as c_long
        );
        assert_eq!(
            set_opt(p_decoder, 20, dummy_ptr),
            CM_INIT_PARA_ERROR as c_long
        );
        assert_eq!(
            get_opt(p_decoder, 0, dummy_ptr),
            CM_INIT_PARA_ERROR as c_long
        );
        assert_eq!(
            get_opt(p_decoder, 20, dummy_ptr),
            CM_INIT_PARA_ERROR as c_long
        );

        assert_eq!(
            ISVCDecoder::Uninitialize(p_decoder),
            CM_RESULT_SUCCESS as c_long
        );
        WelsDestroyDecoder(p_decoder);
    }
}
