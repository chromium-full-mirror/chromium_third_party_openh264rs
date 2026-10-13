// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Integration tests for `ISVCEncoder` extended initialization, dynamic options,
//! statistics, parameter sets, spatial/temporal layer control, and extreme geometries.
//!
//! Ported from upstream `test/encoder/EncUT_EncoderExt.cpp` (`EncoderInterfaceTest`)
//! and `test/api/encoder_test.cpp` (`EncoderInitTest`).

mod common;

use common::frame_gen::{fill_camera_frame, fill_seeded_noise};
use openh264rs::api::codec_api::*;
use std::ptr;

fn valid_size(size: i32) -> i32 {
    if size > 16 { size } else { 16 }
}

fn prepare_src_frame(
    buf: &mut Vec<u8>,
    width: i32,
    height: i32,
    frame_idx: usize,
) -> SSourcePicture {
    let w = valid_size(width) as usize;
    let h = valid_size(height) as usize;
    let frame_size = w * h * 3 / 2;
    buf.resize(frame_size, 0);
    fill_camera_frame(buf, w, h, frame_idx);

    let mut pic = SSourcePicture::default();
    pic.iColorFormat = EVideoFormatType::videoFormatI420 as i32;
    pic.uiTimeStamp = (frame_idx as i64) * 30;
    pic.iPicWidth = w as i32;
    pic.iPicHeight = h as i32;
    pic.iStride[0] = w as i32;
    pic.iStride[1] = (w >> 1) as i32;
    pic.iStride[2] = (w >> 1) as i32;
    unsafe {
        pic.pData[0] = buf.as_mut_ptr();
        pic.pData[1] = buf.as_mut_ptr().add(w * h);
        pic.pData[2] = buf.as_mut_ptr().add(w * h + (w * h >> 2));
    }
    pic
}

fn init_param_ext_1280x720(param: &mut SEncParamExt) {
    param.iPicWidth = 1280;
    param.iPicHeight = 720;
    param.iTargetBitrate = 50000;
    param.iTemporalLayerNum = 3;
    param.iSpatialLayerNum = 1;
    param.iNumRefFrame = AUTO_REF_PIC_COUNT;
    param.sSpatialLayers[0].iVideoHeight = param.iPicHeight;
    param.sSpatialLayers[0].iVideoWidth = param.iPicWidth;
    param.sSpatialLayers[0].iSpatialBitrate = 50000;
}

fn make_valid_enc_param_base(width: i32, height: i32, fps: f32) -> SEncParamBase {
    let w = valid_size(width & !1);
    let h = valid_size(height & !1);
    let min_br = (w * h / 50).max(100_000);
    SEncParamBase {
        iUsageType: EUsageType::CAMERA_VIDEO_REAL_TIME,
        iPicWidth: w,
        iPicHeight: h,
        iTargetBitrate: min_br.min(60_000_000),
        iRCMode: RC_MODES::RC_BITRATE_MODE,
        fMaxFrameRate: fps,
    }
}

/// Port of `EncoderInitTest.VeryLargeSlices` (`test/api/encoder_test.cpp:247-296`).
///
/// Uses random noise input and a low QP (`12`) in `RC_OFF_MODE` with 4 threads and 4 fixed
/// slices to intentionally produce <1:1 compression and verify that inflated slice buffers
/// encode and decode cleanly without out-of-memory errors.
#[test]
fn encoder_init_test_very_large_slices() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let mut param = SEncParamExt::default();
        assert_eq!(
            ISVCEncoder::GetDefaultParams(encoder, &mut param),
            CM_RESULT_SUCCESS
        );

        param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
        param.iPicWidth = 1280;
        param.iPicHeight = 720;
        param.fMaxFrameRate = 30.0;
        param.iSpatialLayerNum = 1;
        param.iMultipleThreadIdc = 4;
        param.iRCMode = RC_MODES::RC_OFF_MODE;

        param.sSpatialLayers[0].iVideoWidth = param.iPicWidth;
        param.sSpatialLayers[0].iVideoHeight = param.iPicHeight;
        param.sSpatialLayers[0].fFrameRate = param.fMaxFrameRate;
        param.sSpatialLayers[0].sSliceArgument.uiSliceMode = SliceModeEnum::SM_FIXEDSLCNUM_SLICE;
        param.sSpatialLayers[0].sSliceArgument.uiSliceNum = 4;
        param.sSpatialLayers[0].iDLayerQp = 12;
        param.iMinQp = 0;
        param.iMaxQp = 51;

        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param),
            CM_RESULT_SUCCESS
        );

        let w = param.iPicWidth as usize;
        let h = param.iPicHeight as usize;
        let frame_size = w * h * 3 / 2;
        let mut buf = vec![0u8; frame_size];
        let mut rng = 0xDEAD_BEEFu32;
        fill_seeded_noise(&mut buf, &mut rng);

        let mut pic = SSourcePicture::default();
        pic.iPicWidth = param.iPicWidth;
        pic.iPicHeight = param.iPicHeight;
        pic.iColorFormat = EVideoFormatType::videoFormatI420 as i32;
        pic.iStride[0] = param.iPicWidth;
        pic.iStride[1] = param.iPicWidth >> 1;
        pic.iStride[2] = param.iPicWidth >> 1;
        pic.pData[0] = buf.as_mut_ptr();
        pic.pData[1] = buf.as_mut_ptr().add(w * h);
        pic.pData[2] = buf.as_mut_ptr().add(w * h + (w * h >> 2));

        let mut info = SFrameBSInfo::default();
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic, &mut info),
            CM_RESULT_SUCCESS
        );
        assert!(
            info.iFrameSizeInBytes > (w * h) as i32,
            "expected inflated bitstream for QP=12 noise, got {} bytes",
            info.iFrameSizeInBytes
        );

        ISVCEncoder::Uninitialize(encoder);
        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncoderInitTest.DynamicAdjustSlicingExtremeAspectRatio`
/// (`test/api/encoder_test.cpp:395-463`).
#[test]
fn encoder_init_test_dynamic_adjust_slicing_extreme_aspect_ratio() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let mut param = SEncParamExt::default();
        assert_eq!(
            ISVCEncoder::GetDefaultParams(encoder, &mut param),
            CM_RESULT_SUCCESS
        );

        param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
        param.iPicWidth = 32752;
        param.iPicHeight = 32;
        param.fMaxFrameRate = 30.0;
        param.iSpatialLayerNum = 1;
        param.iMultipleThreadIdc = 4;
        param.iRCMode = RC_MODES::RC_OFF_MODE;
        param.bUseLoadBalancing = true;
        param.iComplexityMode = ECOMPLEXITY_MODE::HIGH_COMPLEXITY;
        param.iMinQp = 0;
        param.iMaxQp = 51;

        param.sSpatialLayers[0].iVideoWidth = param.iPicWidth;
        param.sSpatialLayers[0].iVideoHeight = param.iPicHeight;
        param.sSpatialLayers[0].fFrameRate = param.fMaxFrameRate;
        param.sSpatialLayers[0].sSliceArgument.uiSliceMode = SliceModeEnum::SM_FIXEDSLCNUM_SLICE;
        param.sSpatialLayers[0].sSliceArgument.uiSliceNum = 4;
        param.sSpatialLayers[0].iDLayerQp = 24;

        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param),
            CM_RESULT_SUCCESS
        );

        let w = param.iPicWidth as usize;
        let h = param.iPicHeight as usize;
        let frame_size = w * h * 3 / 2;
        let mut buf = vec![0u8; frame_size];
        let mut rng = 0x1357_9BDFu32;

        let mut pic = SSourcePicture::default();
        pic.iPicWidth = param.iPicWidth;
        pic.iPicHeight = param.iPicHeight;
        pic.iColorFormat = EVideoFormatType::videoFormatI420 as i32;
        pic.iStride[0] = param.iPicWidth;
        pic.iStride[1] = param.iPicWidth >> 1;
        pic.iStride[2] = param.iPicWidth >> 1;
        pic.pData[0] = buf.as_mut_ptr();
        pic.pData[1] = buf.as_mut_ptr().add(w * h);
        pic.pData[2] = buf.as_mut_ptr().add(w * h + (w * h >> 2));

        for _ in 0..10 {
            fill_seeded_noise(&mut buf, &mut rng);
            // Keep Slice 0 completely flat (zeros) while Slices 1..3 have random noise.
            for y in 0..16 {
                ptr::write_bytes(pic.pData[0].add(y * (pic.iStride[0] as usize)), 0, 16368);
            }
            for y in 0..8 {
                ptr::write_bytes(pic.pData[1].add(y * (pic.iStride[1] as usize)), 0, 8184);
                ptr::write_bytes(pic.pData[2].add(y * (pic.iStride[2] as usize)), 0, 8184);
            }
            let mut info = SFrameBSInfo::default();
            assert_eq!(
                ISVCEncoder::EncodeFrame(encoder, &pic, &mut info),
                CM_RESULT_SUCCESS
            );
        }

        ISVCEncoder::Uninitialize(encoder);
        WelsDestroySVCEncoder(encoder);
    }
}

/// Tests per-frame PSNR reporting (`SSourcePicture.bPsnrY/bPsnrU/bPsnrV` -> `SLayerBSInfo.rPsnr`).
#[test]
fn encoder_init_test_per_frame_psnr() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let mut param = SEncParamExt::default();
        assert_eq!(
            ISVCEncoder::GetDefaultParams(encoder, &mut param),
            CM_RESULT_SUCCESS
        );
        param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
        param.iPicWidth = 176;
        param.iPicHeight = 144;
        param.fMaxFrameRate = 30.0;
        param.iSpatialLayerNum = 1;
        param.iRCMode = RC_MODES::RC_OFF_MODE;
        param.sSpatialLayers[0].iVideoWidth = 176;
        param.sSpatialLayers[0].iVideoHeight = 144;
        param.sSpatialLayers[0].fFrameRate = 30.0;
        param.sSpatialLayers[0].iDLayerQp = 24;

        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param),
            CM_RESULT_SUCCESS
        );

        let mut buf = Vec::new();
        for mask in 0..8u8 {
            let mut pic = prepare_src_frame(&mut buf, 176, 144, mask as usize);
            pic.bPsnrY = (mask & 1) != 0;
            pic.bPsnrU = (mask & 2) != 0;
            pic.bPsnrV = (mask & 4) != 0;

            let mut info = SFrameBSInfo::default();
            assert_eq!(
                ISVCEncoder::EncodeFrame(encoder, &pic, &mut info),
                CM_RESULT_SUCCESS
            );
            // Find the VCL layer.
            let vcl_layer = (0..info.iLayerNum as usize)
                .map(|i| &info.sLayerInfo[i])
                .find(|l| l.uiLayerType == LAYER_TYPE::VIDEO_CODING_LAYER as u8)
                .expect("expected a VCL layer");

            for (plane, enabled) in [(0, pic.bPsnrY), (1, pic.bPsnrU), (2, pic.bPsnrV)] {
                let psnr = vcl_layer.rPsnr[plane];
                if enabled {
                    assert!(
                        (20.0..=100.0).contains(&psnr),
                        "mask={mask}: expected valid PSNR for plane {plane}, got {psnr}"
                    );
                } else {
                    assert_eq!(
                        psnr, 0.0,
                        "mask={mask}: expected 0.0 PSNR for disabled plane {plane}, got {psnr}"
                    );
                }
            }
        }

        ISVCEncoder::Uninitialize(encoder);
        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncoderInterfaceTest.EncoderOptionSetTest` (`test/encoder/EncUT_EncoderExt.cpp:153-275`).
#[test]
fn encoder_interface_test_encoder_option_set() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let mut trace_level: i32 = 0; // WELS_LOG_QUIET
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_OPTION_TRACE_LEVEL,
            ptr::addr_of_mut!(trace_level).cast(),
        );

        let mut param_ext = SEncParamExt::default();
        ISVCEncoder::GetDefaultParams(encoder, &mut param_ext);
        init_param_ext_1280x720(&mut param_ext);
        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param_ext),
            CM_RESULT_SUCCESS
        );

        let mut buf = Vec::new();

        // 1. ENCODER_OPTION_DATAFORMAT
        for &val in &[0i32, EVideoFormatType::videoFormatI420 as i32] {
            let mut v = val;
            let res = ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_DATAFORMAT,
                ptr::addr_of_mut!(v).cast(),
            );
            if val == 0 {
                assert_eq!(res, CM_INIT_PARA_ERROR);
            } else {
                assert_eq!(res, CM_RESULT_SUCCESS);
                let mut ret_val = -1i32;
                assert_eq!(
                    ISVCEncoder::GetOption(
                        encoder,
                        ENCODER_OPTION::ENCODER_OPTION_DATAFORMAT,
                        ptr::addr_of_mut!(ret_val).cast(),
                    ),
                    CM_RESULT_SUCCESS
                );
                assert_eq!(ret_val, val);
            }
        }

        let mut pic = prepare_src_frame(&mut buf, 1280, 720, 0);
        let mut fbi = SFrameBSInfo::default();
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
            CM_RESULT_SUCCESS
        );

        // 2. ENCODER_OPTION_IDR_INTERVAL
        for &val in &[-3i32, 0, 30] {
            let mut v = val;
            assert_eq!(
                ISVCEncoder::SetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_OPTION_IDR_INTERVAL,
                    ptr::addr_of_mut!(v).cast(),
                ),
                CM_RESULT_SUCCESS
            );
            let mut ret_val = -99i32;
            assert_eq!(
                ISVCEncoder::GetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_OPTION_IDR_INTERVAL,
                    ptr::addr_of_mut!(ret_val).cast(),
                ),
                CM_RESULT_SUCCESS
            );
            let expected = if val <= -1 { 0 } else { val };
            assert_eq!(ret_val, expected);
        }

        pic = prepare_src_frame(&mut buf, 1280, 720, 1);
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
            CM_RESULT_SUCCESS
        );

        // 3. ENCODER_OPTION_FRAME_RATE
        for &fps in &[-2.0f32, 0.0, 25.0, 90.0] {
            let mut f = fps;
            let res = ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_FRAME_RATE,
                ptr::addr_of_mut!(f).cast(),
            );
            if fps <= 0.0 {
                assert_eq!(res, CM_INIT_PARA_ERROR);
            } else {
                assert_eq!(res, CM_RESULT_SUCCESS);
                let mut ret_fps = 0.0f32;
                assert_eq!(
                    ISVCEncoder::GetOption(
                        encoder,
                        ENCODER_OPTION::ENCODER_OPTION_FRAME_RATE,
                        ptr::addr_of_mut!(ret_fps).cast(),
                    ),
                    CM_RESULT_SUCCESS
                );
                assert_eq!(ret_fps, fps.clamp(1.0, 60.0));
            }
        }

        pic = prepare_src_frame(&mut buf, 1280, 720, 2);
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
            CM_RESULT_SUCCESS
        );

        // 4. ENCODER_OPTION_BITRATE & ENCODER_OPTION_MAX_BITRATE
        let mut max_br = SBitrateInfo {
            iLayer: LAYER_NUM::SPATIAL_LAYER_0,
            iBitrate: 500_000,
        };
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_MAX_BITRATE,
                ptr::addr_of_mut!(max_br).cast(),
            ),
            CM_RESULT_SUCCESS
        );
        let mut target_br = SBitrateInfo {
            iLayer: LAYER_NUM::SPATIAL_LAYER_0,
            iBitrate: 250_000,
        };
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_BITRATE,
                ptr::addr_of_mut!(target_br).cast(),
            ),
            CM_RESULT_SUCCESS
        );
        let mut got_br = SBitrateInfo {
            iLayer: LAYER_NUM::SPATIAL_LAYER_0,
            iBitrate: 0,
        };
        assert_eq!(
            ISVCEncoder::GetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_BITRATE,
                ptr::addr_of_mut!(got_br).cast(),
            ),
            CM_RESULT_SUCCESS
        );
        assert_eq!(got_br.iBitrate, 250_000);

        // 5. ENCODER_OPTION_RC_MODE
        for &rc in &[-1i32, 0, 1, 2] {
            let mut v = rc;
            assert_eq!(
                ISVCEncoder::SetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_OPTION_RC_MODE,
                    ptr::addr_of_mut!(v).cast(),
                ),
                CM_RESULT_SUCCESS
            );
        }

        pic = prepare_src_frame(&mut buf, 1280, 720, 3);
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
            CM_RESULT_SUCCESS
        );

        ISVCEncoder::Uninitialize(encoder);
        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncoderInterfaceTest.MemoryCheckTest` (`test/encoder/EncUT_EncoderExt.cpp:400-473`).
#[test]
fn encoder_interface_test_memory_check() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let mut param_ext = SEncParamExt::default();
        ISVCEncoder::GetDefaultParams(encoder, &mut param_ext);
        init_param_ext_1280x720(&mut param_ext);
        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param_ext),
            CM_RESULT_SUCCESS
        );

        let mut buf = Vec::new();
        let mut fbi = SFrameBSInfo::default();
        for i in 0..10 {
            let pic = prepare_src_frame(&mut buf, param_ext.iPicWidth, param_ext.iPicHeight, i);
            assert_eq!(
                ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
                CM_RESULT_SUCCESS
            );
        }

        param_ext.iPicWidth += 128;
        param_ext.iPicHeight += 64;
        param_ext.sSpatialLayers[0].iVideoWidth = param_ext.iPicWidth;
        param_ext.sSpatialLayers[0].iVideoHeight = param_ext.iPicHeight;
        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param_ext),
            CM_RESULT_SUCCESS
        );

        let mut option = param_ext;
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                ptr::addr_of_mut!(option).cast(),
            ),
            CM_RESULT_SUCCESS
        );

        for i in 0..10 {
            let pic = prepare_src_frame(&mut buf, option.iPicWidth, option.iPicHeight, i);
            assert_eq!(
                ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
                CM_RESULT_SUCCESS
            );
        }

        option.iLTRRefNum += 2;
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                ptr::addr_of_mut!(option).cast(),
            ),
            CM_RESULT_SUCCESS
        );

        for i in 0..10 {
            let pic = prepare_src_frame(&mut buf, option.iPicWidth, option.iPicHeight, i);
            assert_eq!(
                ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
                CM_RESULT_SUCCESS
            );
        }

        assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncoderInterfaceTest.BasicInitializeTest` and `BaseParamSettingTest`
/// (`test/encoder/EncUT_EncoderExt.cpp:492-536`).
#[test]
fn encoder_interface_test_basic_initialize_and_base_param_setting() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let mut param_base = make_valid_enc_param_base(640, 360, 30.0);
        assert_eq!(
            ISVCEncoder::Initialize(encoder, &param_base),
            CM_RESULT_SUCCESS
        );

        let mut buf = Vec::new();
        let mut fbi = SFrameBSInfo::default();
        let pic = prepare_src_frame(&mut buf, param_base.iPicWidth, param_base.iPicHeight, 0);
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
            CM_RESULT_SUCCESS
        );
        assert_eq!(fbi.eFrameType, EVideoFrameType::videoFrameTypeIDR);

        param_base = make_valid_enc_param_base(320, 192, 15.0);
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_BASE,
                ptr::addr_of_mut!(param_base).cast(),
            ),
            CM_RESULT_SUCCESS
        );

        let pic2 = prepare_src_frame(&mut buf, param_base.iPicWidth, param_base.iPicHeight, 1);
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic2, &mut fbi),
            CM_RESULT_SUCCESS
        );

        assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncoderInterfaceTest.BasicInitializeTestFalse`
/// (`test/encoder/EncUT_EncoderExt.cpp:538-613`).
#[test]
fn encoder_interface_test_basic_initialize_false() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let mut trace_level: i32 = 0; // WELS_LOG_QUIET
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_OPTION_TRACE_LEVEL,
            ptr::addr_of_mut!(trace_level).cast(),
        );

        // 1. Invalid iUsageType (CAMERA_VIDEO_NON_REAL_TIME = 2)
        let mut param_base = make_valid_enc_param_base(640, 360, 30.0);
        param_base.iUsageType = EUsageType::CAMERA_VIDEO_NON_REAL_TIME;
        assert_eq!(
            ISVCEncoder::Initialize(encoder, &param_base),
            CM_INIT_PARA_ERROR
        );

        // 2. Invalid iPicWidth (0, -1)
        for &w in &[0, -1] {
            param_base = make_valid_enc_param_base(640, 360, 30.0);
            param_base.iPicWidth = w;
            assert_eq!(
                ISVCEncoder::Initialize(encoder, &param_base),
                CM_INIT_PARA_ERROR
            );
        }

        // 3. Invalid iPicHeight (0, -1)
        for &h in &[0, -1] {
            param_base = make_valid_enc_param_base(640, 360, 30.0);
            param_base.iPicHeight = h;
            assert_eq!(
                ISVCEncoder::Initialize(encoder, &param_base),
                CM_INIT_PARA_ERROR
            );
        }

        // 4. Level 5.2 macroblock limit: iPicWidth * iPicHeight <= 36864 * 16 * 16
        param_base = make_valid_enc_param_base(640, 360, 30.0);
        param_base.iPicWidth = 5000;
        param_base.iPicHeight = 5000;
        assert_eq!(
            ISVCEncoder::Initialize(encoder, &param_base),
            CM_INIT_PARA_ERROR
        );

        param_base = make_valid_enc_param_base(640, 360, 30.0);
        param_base.iPicWidth = 36864;
        param_base.iPicHeight = 256;
        assert_eq!(
            ISVCEncoder::Initialize(encoder, &param_base),
            CM_RESULT_SUCCESS
        );

        param_base.iPicWidth = 256;
        param_base.iPicHeight = 36864;
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_BASE,
                ptr::addr_of_mut!(param_base).cast(),
            ),
            CM_RESULT_SUCCESS
        );

        param_base.iPicWidth = 5000;
        param_base.iPicHeight = 5000;
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_BASE,
                ptr::addr_of_mut!(param_base).cast(),
            ),
            CM_INIT_PARA_ERROR
        );
        ISVCEncoder::Uninitialize(encoder);

        // 5. Invalid iTargetBitrate (0, -1)
        for &br in &[0, -1] {
            param_base = make_valid_enc_param_base(640, 360, 30.0);
            param_base.iTargetBitrate = br;
            assert_eq!(
                ISVCEncoder::Initialize(encoder, &param_base),
                CM_INIT_PARA_ERROR
            );
        }

        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncoderInterfaceTest.BasicInitializeTestAutoAdjustment`
/// (`test/encoder/EncUT_EncoderExt.cpp:615-638`).
#[test]
fn encoder_interface_test_basic_initialize_auto_adjustment() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let mut buf = Vec::new();
        for &fps in &[50000.0f32, 0.0, -1.0] {
            let mut param_base = make_valid_enc_param_base(320, 192, fps);
            assert_eq!(
                ISVCEncoder::Initialize(encoder, &param_base),
                CM_RESULT_SUCCESS
            );
            let pic = prepare_src_frame(&mut buf, param_base.iPicWidth, param_base.iPicHeight, 0);
            let mut fbi = SFrameBSInfo::default();
            assert_eq!(
                ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
                CM_RESULT_SUCCESS
            );
            assert_eq!(fbi.eFrameType, EVideoFrameType::videoFrameTypeIDR);
            assert_eq!(
                ISVCEncoder::GetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_BASE,
                    ptr::addr_of_mut!(param_base).cast(),
                ),
                CM_RESULT_SUCCESS
            );
            assert!(
                (1.0..=60.0).contains(&param_base.fMaxFrameRate),
                "fMaxFrameRate was not clamped: {}",
                param_base.fMaxFrameRate
            );
            assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
        }

        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncoderInterfaceTest.ForceIntraFrameSimulCastAVC`
/// (`test/encoder/EncUT_EncoderExt.cpp:640-696`).
#[test]
fn encoder_interface_test_force_intra_frame_simulcast_avc() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let mut param = SEncParamExt::default();
        ISVCEncoder::GetDefaultParams(encoder, &mut param);
        param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
        param.iSpatialLayerNum = 3;
        param.iPicWidth = 640;
        param.iPicHeight = 360;
        param.iTargetBitrate = 0;
        param.iRCMode = RC_MODES::RC_BITRATE_MODE;
        param.bEnableFrameSkip = false;
        param.fMaxFrameRate = 30.0;
        param.bSimulcastAVC = true;

        for i_num in (0..param.iSpatialLayerNum as usize).rev() {
            let scale = 1i32 << ((param.iSpatialLayerNum as usize - 1) - i_num);
            param.sSpatialLayers[i_num].iVideoWidth = param.iPicWidth / scale;
            param.sSpatialLayers[i_num].iVideoHeight = param.iPicHeight / scale;
            param.sSpatialLayers[i_num].fFrameRate = 30.0;
            param.sSpatialLayers[i_num].iSpatialBitrate = (param.sSpatialLayers[i_num].iVideoWidth
                * param.sSpatialLayers[i_num].iVideoHeight)
                / 3;
            param.iTargetBitrate += param.sSpatialLayers[i_num].iSpatialBitrate;
        }

        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param),
            CM_RESULT_SUCCESS
        );

        let mut buf = Vec::new();
        let mut fbi = SFrameBSInfo::default();
        let vtbl = (*encoder).lpVtbl;

        for frame_idx in 0..24usize {
            let pic = prepare_src_frame(&mut buf, param.iPicWidth, param.iPicHeight, frame_idx);
            let forced_layer: Option<u8> = match frame_idx {
                6 => Some(0),
                12 => Some(1),
                18 => Some(2),
                _ => None,
            };
            if let Some(layer_id) = forced_layer {
                assert_eq!(
                    ((*vtbl).ForceIntraFrame)(encoder, 1, layer_id as i32),
                    CM_RESULT_SUCCESS
                );
            }

            assert_eq!(
                ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
                CM_RESULT_SUCCESS
            );

            if let Some(layer_id) = forced_layer {
                let mut found = false;
                for i in 0..fbi.iLayerNum as usize {
                    if fbi.sLayerInfo[i].uiSpatialId == layer_id {
                        assert_eq!(
                            fbi.sLayerInfo[i].eFrameType,
                            EVideoFrameType::videoFrameTypeIDR,
                            "frame {frame_idx}: spatial layer {layer_id} did not emit IDR"
                        );
                        found = true;
                    }
                }
                assert!(found, "frame {frame_idx}: spatial layer {layer_id} missing");
            }
        }

        assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncoderInterfaceTest.ForceIntraFrame` and `ForceIntraFrameWithTemporal`
/// (`test/encoder/EncUT_EncoderExt.cpp:699-771`).
#[test]
fn encoder_interface_test_force_intra_frame_and_temporal() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        // Part 1: ForceIntraFrame with SEncParamBase
        let param_base = make_valid_enc_param_base(320, 192, 30.0);
        assert_eq!(
            ISVCEncoder::Initialize(encoder, &param_base),
            CM_RESULT_SUCCESS
        );

        let mut buf = Vec::new();
        let mut fbi = SFrameBSInfo::default();

        ISVCEncoder::ForceIntraFrame(encoder, true);
        let mut pic = prepare_src_frame(&mut buf, 320, 192, 0);
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
            CM_RESULT_SUCCESS
        );
        assert_eq!(fbi.eFrameType, EVideoFrameType::videoFrameTypeIDR);

        pic = prepare_src_frame(&mut buf, 320, 192, 1);
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
            CM_RESULT_SUCCESS
        );

        ISVCEncoder::ForceIntraFrame(encoder, true);
        let mut count = 0;
        loop {
            pic = prepare_src_frame(&mut buf, 320, 192, 2 + count);
            assert_eq!(
                ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
                CM_RESULT_SUCCESS
            );
            count += 1;
            if fbi.eFrameType != EVideoFrameType::videoFrameTypeSkip || count >= 100 {
                break;
            }
        }
        assert_eq!(fbi.eFrameType, EVideoFrameType::videoFrameTypeIDR);
        assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);

        // Part 2: ForceIntraFrameWithTemporal (iTemporalLayerNum = 3)
        let mut param_ext = SEncParamExt::default();
        ISVCEncoder::GetDefaultParams(encoder, &mut param_ext);
        param_ext.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
        param_ext.iPicWidth = 320;
        param_ext.iPicHeight = 192;
        param_ext.iTargetBitrate = 500_000;
        param_ext.iRCMode = RC_MODES::RC_BITRATE_MODE;
        param_ext.fMaxFrameRate = 30.0;
        param_ext.iTemporalLayerNum = 3;
        param_ext.sSpatialLayers[0].iVideoWidth = 320;
        param_ext.sSpatialLayers[0].iVideoHeight = 192;
        param_ext.sSpatialLayers[0].fFrameRate = 30.0;
        param_ext.sSpatialLayers[0].iSpatialBitrate = 500_000;

        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param_ext),
            CM_RESULT_SUCCESS
        );

        pic = prepare_src_frame(&mut buf, 320, 192, 0);
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
            CM_RESULT_SUCCESS
        );
        assert_eq!(fbi.eFrameType, EVideoFrameType::videoFrameTypeIDR);

        pic = prepare_src_frame(&mut buf, 320, 192, 1);
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
            CM_RESULT_SUCCESS
        );

        ISVCEncoder::ForceIntraFrame(encoder, true);
        count = 0;
        loop {
            pic = prepare_src_frame(&mut buf, 320, 192, 2 + count);
            assert_eq!(
                ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
                CM_RESULT_SUCCESS
            );
            count += 1;
            if fbi.eFrameType != EVideoFrameType::videoFrameTypeSkip || count >= 100 {
                break;
            }
        }
        assert_eq!(fbi.eFrameType, EVideoFrameType::videoFrameTypeIDR);
        assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);

        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncoderInterfaceTest.EncodeParameterSets`
/// (`test/encoder/EncUT_EncoderExt.cpp:773-811`).
#[test]
fn encoder_interface_test_encode_parameter_sets() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let param_base = make_valid_enc_param_base(320, 192, 30.0);
        assert_eq!(
            ISVCEncoder::Initialize(encoder, &param_base),
            CM_RESULT_SUCCESS
        );

        let mut buf = Vec::new();
        let mut fbi = SFrameBSInfo::default();

        let pic0 = prepare_src_frame(&mut buf, 320, 192, 0);
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic0, &mut fbi),
            CM_RESULT_SUCCESS
        );
        assert_eq!(fbi.eFrameType, EVideoFrameType::videoFrameTypeIDR);

        let pic1 = prepare_src_frame(&mut buf, 320, 192, 1);
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic1, &mut fbi),
            CM_RESULT_SUCCESS
        );

        // Call EncodeParameterSets
        fbi = SFrameBSInfo::default();
        assert_eq!(
            ISVCEncoder::EncodeParameterSets(encoder, &mut fbi),
            CM_RESULT_SUCCESS
        );
        assert_eq!(fbi.eFrameType, EVideoFrameType::videoFrameTypeInvalid);
        assert!(fbi.iLayerNum > 0);

        for i in 0..fbi.iLayerNum as usize {
            let layer = &fbi.sLayerInfo[i];
            assert_eq!(layer.uiLayerType, LAYER_TYPE::NON_VIDEO_CODING_LAYER as u8);
            assert!(layer.iNalCount > 0);
            let nal_lens =
                std::slice::from_raw_parts(layer.pNalLengthInByte, layer.iNalCount as usize);
            let mut offset = 0usize;
            for &len in nal_lens {
                let nal_type = (*layer.pBsBuf.add(offset + 4)) & 0x0f;
                assert!(
                    nal_type == 7 || nal_type == 8 || nal_type == 15,
                    "expected SPS/PPS/SubsetSPS NAL type, got {nal_type}"
                );
                offset += len as usize;
            }
        }

        // Subsequent P-frame should not be forced to IDR
        let pic2 = prepare_src_frame(&mut buf, 320, 192, 2);
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic2, &mut fbi),
            CM_RESULT_SUCCESS
        );
        assert_ne!(fbi.eFrameType, EVideoFrameType::videoFrameTypeIDR);

        assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncoderInterfaceTest.GetStatistics` (`test/encoder/EncUT_EncoderExt.cpp:817-919`).
#[test]
fn encoder_interface_test_get_statistics() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let param_base = make_valid_enc_param_base(320, 192, 30.0);
        assert_eq!(
            ISVCEncoder::Initialize(encoder, &param_base),
            CM_RESULT_SUCCESS
        );

        let mut buf = Vec::new();
        let mut fbi = SFrameBSInfo::default();

        let mut pic = prepare_src_frame(&mut buf, 320, 192, 0);
        pic.uiTimeStamp = 0;
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
            CM_RESULT_SUCCESS
        );
        assert_eq!(fbi.eFrameType, EVideoFrameType::videoFrameTypeIDR);

        pic = prepare_src_frame(&mut buf, 320, 192, 1);
        pic.uiTimeStamp = 30;
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
            CM_RESULT_SUCCESS
        );

        let mut stats = SEncoderStatistics::default();
        assert_eq!(
            ISVCEncoder::GetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_GET_STATISTICS,
                ptr::addr_of_mut!(stats).cast(),
            ),
            CM_RESULT_SUCCESS
        );
        assert_eq!(stats.uiInputFrameCount, 2);
        assert_eq!(stats.uiIDRSentNum, 1);
        assert_eq!(stats.uiResolutionChangeTimes, 0);
        assert_eq!(stats.uiWidth, 320);
        assert_eq!(stats.uiHeight, 192);

        // Change resolution via ENCODER_OPTION_SVC_ENCODE_PARAM_EXT
        let mut param_ext = SEncParamExt::default();
        assert_eq!(
            ISVCEncoder::GetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                ptr::addr_of_mut!(param_ext).cast(),
            ),
            CM_RESULT_SUCCESS
        );
        param_ext.iPicWidth = 640;
        param_ext.iPicHeight = 360;
        param_ext.sSpatialLayers[0].iVideoWidth = 640;
        param_ext.sSpatialLayers[0].iVideoHeight = 360;
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                ptr::addr_of_mut!(param_ext).cast(),
            ),
            CM_RESULT_SUCCESS
        );

        pic = prepare_src_frame(&mut buf, 640, 360, 2);
        pic.uiTimeStamp = 60;
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
            CM_RESULT_SUCCESS
        );

        assert_eq!(
            ISVCEncoder::GetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_GET_STATISTICS,
                ptr::addr_of_mut!(stats).cast(),
            ),
            CM_RESULT_SUCCESS
        );
        assert_eq!(stats.uiInputFrameCount, 3);
        assert_eq!(stats.uiResolutionChangeTimes, 1);
        assert_eq!(stats.uiIDRSentNum, 2);
        assert_eq!(stats.uiWidth, 640);
        assert_eq!(stats.uiHeight, 360);

        // Encode at timestamp 1000ms and check fAverageFrameRate
        pic = prepare_src_frame(&mut buf, 640, 360, 3);
        pic.uiTimeStamp = 1000;
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
            CM_RESULT_SUCCESS
        );
        assert_eq!(
            ISVCEncoder::GetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_GET_STATISTICS,
                ptr::addr_of_mut!(stats).cast(),
            ),
            CM_RESULT_SUCCESS
        );
        assert_eq!(stats.fAverageFrameRate as u32, stats.uiInputFrameCount);

        // Check statistics log interval get/set
        let mut interval = 0i32;
        assert_eq!(
            ISVCEncoder::GetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_STATISTICS_LOG_INTERVAL,
                ptr::addr_of_mut!(interval).cast(),
            ),
            CM_RESULT_SUCCESS
        );
        assert_eq!(interval, 5000);

        for &new_interval in &[2000i32, 0] {
            let mut v = new_interval;
            assert_eq!(
                ISVCEncoder::SetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_OPTION_STATISTICS_LOG_INTERVAL,
                    ptr::addr_of_mut!(v).cast(),
                ),
                CM_RESULT_SUCCESS
            );
            assert_eq!(
                ISVCEncoder::GetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_OPTION_STATISTICS_LOG_INTERVAL,
                    ptr::addr_of_mut!(interval).cast(),
                ),
                CM_RESULT_SUCCESS
            );
            assert_eq!(interval, new_interval);
        }

        assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncoderInterfaceTest.FrameSizeCheck` (`test/encoder/EncUT_EncoderExt.cpp:921-943`).
#[test]
fn encoder_interface_test_frame_size_check() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let param_base = make_valid_enc_param_base(640, 360, 30.0);
        assert_eq!(
            ISVCEncoder::Initialize(encoder, &param_base),
            CM_RESULT_SUCCESS
        );

        let mut buf = Vec::new();
        let pic = prepare_src_frame(&mut buf, 640, 360, 0);
        let mut fbi = SFrameBSInfo::default();
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
            CM_RESULT_SUCCESS
        );

        let mut length = 0i32;
        for i in 0..fbi.iLayerNum as usize {
            for j in 0..fbi.sLayerInfo[i].iNalCount as usize {
                length += *fbi.sLayerInfo[i].pNalLengthInByte.add(j);
            }
        }
        assert_eq!(length, fbi.iFrameSizeInBytes);

        assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncoderInterfaceTest.SkipFrameCheck` (`test/encoder/EncUT_EncoderExt.cpp:945-1008`).
#[test]
fn encoder_interface_test_skip_frame_check() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let mut param = SEncParamExt::default();
        assert_eq!(
            ISVCEncoder::GetDefaultParams(encoder, &mut param),
            CM_RESULT_SUCCESS
        );

        param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
        param.iPicWidth = 360;
        param.iPicHeight = 640;
        param.iTargetBitrate = 573000;
        param.iRCMode = RC_MODES::RC_BITRATE_MODE;
        param.fMaxFrameRate = 28.248587;
        param.iTemporalLayerNum = 3;
        param.iSpatialLayerNum = 1;
        param.bEnableLongTermReference = false;
        param.bEnableSceneChangeDetect = false;
        param.bEnableFrameSkip = true;
        param.iMaxBitrate = 895839855;
        param.uiMaxNalSize = 0;

        param.sSpatialLayers[0].uiLevelIdc = ELevelIdc::LEVEL_5_0;
        param.sSpatialLayers[0].iVideoWidth = 360;
        param.sSpatialLayers[0].iVideoHeight = 640;
        param.sSpatialLayers[0].fFrameRate = 28.248587;
        param.sSpatialLayers[0].iSpatialBitrate = 573000;
        param.sSpatialLayers[0].iMaxSpatialBitrate = 895839855;
        param.sSpatialLayers[0].sSliceArgument.uiSliceMode = SliceModeEnum::SM_FIXEDSLCNUM_SLICE;

        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param),
            CM_RESULT_SUCCESS
        );

        let mut interval = 300i32;
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_STATISTICS_LOG_INTERVAL,
                ptr::addr_of_mut!(interval).cast(),
            ),
            CM_RESULT_SUCCESS
        );

        let mut buf = Vec::new();
        let mut fbi = SFrameBSInfo::default();
        for i in 0..50usize {
            let mut pic = prepare_src_frame(&mut buf, 360, 640, i);
            pic.uiTimeStamp = (i as i64) * 33;
            assert_eq!(
                ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
                CM_RESULT_SUCCESS
            );
            if i == 25 {
                param.fMaxFrameRate = 28.0;
                param.sSpatialLayers[0].fFrameRate = 28.0;
                param.sSpatialLayers[0].iSpatialBitrate = 500000;
                assert_eq!(
                    ISVCEncoder::SetOption(
                        encoder,
                        ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                        ptr::addr_of_mut!(param).cast(),
                    ),
                    CM_RESULT_SUCCESS
                );
            }
        }

        let mut stats = SEncoderStatistics::default();
        assert_eq!(
            ISVCEncoder::GetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_GET_STATISTICS,
                ptr::addr_of_mut!(stats).cast(),
            ),
            CM_RESULT_SUCCESS
        );
        assert!(
            stats.uiInputFrameCount - stats.uiSkippedFrameCount > 2,
            "uiInputFrameCount = {}, uiSkippedFrameCount = {}",
            stats.uiInputFrameCount,
            stats.uiSkippedFrameCount
        );

        assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncoderInterfaceTest.DiffResolutionCheck`
/// (`test/encoder/EncUT_EncoderExt.cpp:1010-1117`).
#[test]
fn encoder_interface_test_diff_resolution_check() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let mut param = SEncParamExt::default();
        assert_eq!(
            ISVCEncoder::GetDefaultParams(encoder, &mut param),
            CM_RESULT_SUCCESS
        );

        param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
        param.iPicWidth = 360;
        param.iPicHeight = 640;
        param.iTargetBitrate = 1000000;
        param.iRCMode = RC_MODES::RC_BITRATE_MODE;
        param.fMaxFrameRate = 24.0;
        param.iTemporalLayerNum = 3;
        param.iSpatialLayerNum = 3;
        param.bEnableLongTermReference = false;
        param.bEnableSceneChangeDetect = false;
        param.bEnableFrameSkip = true;
        param.uiMaxNalSize = 0;

        param.sSpatialLayers[0].uiLevelIdc = ELevelIdc::LEVEL_UNKNOWN;
        param.sSpatialLayers[0].iVideoWidth = 90;
        param.sSpatialLayers[0].iVideoHeight = 160;
        param.sSpatialLayers[0].fFrameRate = 6.0;
        param.sSpatialLayers[0].iSpatialBitrate = 80000;
        param.sSpatialLayers[0].sSliceArgument.uiSliceMode = SliceModeEnum::SM_FIXEDSLCNUM_SLICE;

        param.sSpatialLayers[1].uiLevelIdc = ELevelIdc::LEVEL_UNKNOWN;
        param.sSpatialLayers[1].iVideoWidth = 180;
        param.sSpatialLayers[1].iVideoHeight = 320;
        param.sSpatialLayers[1].fFrameRate = 12.0;
        param.sSpatialLayers[1].iSpatialBitrate = 200000;
        param.sSpatialLayers[1].sSliceArgument.uiSliceMode = SliceModeEnum::SM_FIXEDSLCNUM_SLICE;

        param.sSpatialLayers[2].uiLevelIdc = ELevelIdc::LEVEL_UNKNOWN;
        param.sSpatialLayers[2].iVideoWidth = 360;
        param.sSpatialLayers[2].iVideoHeight = 640;
        param.sSpatialLayers[2].fFrameRate = 24.0;
        param.sSpatialLayers[2].iSpatialBitrate = 600000;
        param.sSpatialLayers[2].sSliceArgument.uiSliceMode = SliceModeEnum::SM_FIXEDSLCNUM_SLICE;

        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param),
            CM_RESULT_SUCCESS
        );

        let mut buf = Vec::new();
        let mut fbi = SFrameBSInfo::default();
        let pic0 = prepare_src_frame(&mut buf, 360, 640, 0);
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic0, &mut fbi),
            CM_RESULT_SUCCESS
        );

        // Reconfigure to 2 valid layers (90x160 and 360x640)
        param.iSpatialLayerNum = 2;
        param.sSpatialLayers[1].uiLevelIdc = ELevelIdc::LEVEL_UNKNOWN;
        param.sSpatialLayers[1].iVideoWidth = 360;
        param.sSpatialLayers[1].iVideoHeight = 640;
        param.sSpatialLayers[1].fFrameRate = 24.0;
        param.sSpatialLayers[1].iSpatialBitrate = 600000;
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                ptr::addr_of_mut!(param).cast(),
            ),
            CM_RESULT_SUCCESS
        );

        let pic1 = prepare_src_frame(&mut buf, 360, 640, 1);
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic1, &mut fbi),
            CM_RESULT_SUCCESS
        );

        // Invalid setting 1: layer 0 larger than layer 1
        let mut trace_level: i32 = 0; // WELS_LOG_QUIET
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_OPTION_TRACE_LEVEL,
            ptr::addr_of_mut!(trace_level).cast(),
        );

        param.sSpatialLayers[1].iVideoWidth = 90;
        param.sSpatialLayers[1].iVideoHeight = 160;
        param.sSpatialLayers[1].fFrameRate = 6.0;
        param.sSpatialLayers[1].iSpatialBitrate = 80000;

        param.sSpatialLayers[0].iVideoWidth = 360;
        param.sSpatialLayers[0].iVideoHeight = 640;
        param.sSpatialLayers[0].fFrameRate = 24.0;
        param.sSpatialLayers[0].iSpatialBitrate = 600000;
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                ptr::addr_of_mut!(param).cast(),
            ),
            CM_INIT_PARA_ERROR
        );

        // Invalid setting 2: 3 layers with non-monotonic resolutions (180x320, 90x160, 360x640)
        param.iSpatialLayerNum = 3;
        param.sSpatialLayers[0].iVideoWidth = 180;
        param.sSpatialLayers[0].iVideoHeight = 320;
        param.sSpatialLayers[0].fFrameRate = 12.0;
        param.sSpatialLayers[0].iSpatialBitrate = 200000;

        param.sSpatialLayers[1].iVideoWidth = 90;
        param.sSpatialLayers[1].iVideoHeight = 160;
        param.sSpatialLayers[1].fFrameRate = 6.0;
        param.sSpatialLayers[1].iSpatialBitrate = 80000;

        param.sSpatialLayers[2].iVideoWidth = 360;
        param.sSpatialLayers[2].iVideoHeight = 640;
        param.sSpatialLayers[2].fFrameRate = 24.0;
        param.sSpatialLayers[2].iSpatialBitrate = 600000;
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                ptr::addr_of_mut!(param).cast(),
            ),
            CM_INIT_PARA_ERROR
        );

        assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncoderInterfaceTest.NalSizeChecking` (`test/encoder/EncUT_EncoderExt.cpp:1119-1212`).
#[test]
fn encoder_interface_test_nal_size_checking() {
    for entropy_mode in [0i32, 1i32] {
        unsafe {
            let mut encoder: *mut ISVCEncoder = ptr::null_mut();
            assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

            let max_nal_size = 1500i32;
            let mut param = SEncParamExt::default();
            ISVCEncoder::GetDefaultParams(encoder, &mut param);
            param.iPicWidth = 640;
            param.iPicHeight = 360;
            param.fMaxFrameRate = 30.0;
            param.iTemporalLayerNum = 2;
            param.iSpatialLayerNum = 2;
            param.iNumRefFrame = AUTO_REF_PIC_COUNT;
            param.iMultipleThreadIdc = 1;
            param.iEntropyCodingModeFlag = entropy_mode;
            param.uiMaxNalSize = max_nal_size as u32;
            param.iTargetBitrate = 0;

            for i in 0..param.iSpatialLayerNum as usize {
                let scale = 1i32 << (param.iSpatialLayerNum as usize - 1 - i);
                param.sSpatialLayers[i].sSliceArgument.uiSliceMode =
                    SliceModeEnum::SM_SIZELIMITED_SLICE;
                param.sSpatialLayers[i].sSliceArgument.uiSliceSizeConstraint = max_nal_size as u32;
                param.sSpatialLayers[i].iVideoWidth = param.iPicWidth / scale;
                param.sSpatialLayers[i].iVideoHeight = param.iPicHeight / scale;
                param.sSpatialLayers[i].iSpatialBitrate = 800_000;
                param.iTargetBitrate += param.sSpatialLayers[i].iSpatialBitrate;
                param.sSpatialLayers[i].fFrameRate = 30.0;
            }

            assert_eq!(
                ISVCEncoder::InitializeExt(encoder, &param),
                CM_RESULT_SUCCESS
            );

            let mut buf = Vec::new();
            let mut fbi = SFrameBSInfo::default();
            for frame_idx in 0..15usize {
                let pic = prepare_src_frame(&mut buf, param.iPicWidth, param.iPicHeight, frame_idx);
                assert_eq!(
                    ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
                    CM_RESULT_SUCCESS
                );
                for i in 0..fbi.iLayerNum as usize {
                    for j in 0..fbi.sLayerInfo[i].iNalCount as usize {
                        let length = *fbi.sLayerInfo[i].pNalLengthInByte.add(j);
                        assert!(
                            length <= max_nal_size,
                            "entropy={entropy_mode}, frame={frame_idx}: NAL length {length} > {max_nal_size}"
                        );
                    }
                }
            }

            // Reconfigure resolution and verify NAL size constraint continues to hold
            param.iPicWidth = 800;
            param.iPicHeight = 448;
            for i in 0..param.iSpatialLayerNum as usize {
                let scale = 1i32 << (param.iSpatialLayerNum as usize - 1 - i);
                param.sSpatialLayers[i].iVideoWidth = param.iPicWidth / scale;
                param.sSpatialLayers[i].iVideoHeight = param.iPicHeight / scale;
            }
            assert_eq!(
                ISVCEncoder::SetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                    ptr::addr_of_mut!(param).cast(),
                ),
                CM_RESULT_SUCCESS
            );

            for frame_idx in 0..10usize {
                let pic = prepare_src_frame(&mut buf, param.iPicWidth, param.iPicHeight, frame_idx);
                assert_eq!(
                    ISVCEncoder::EncodeFrame(encoder, &pic, &mut fbi),
                    CM_RESULT_SUCCESS
                );
                for i in 0..fbi.iLayerNum as usize {
                    for j in 0..fbi.sLayerInfo[i].iNalCount as usize {
                        let length = *fbi.sLayerInfo[i].pNalLengthInByte.add(j);
                        assert!(
                            length <= max_nal_size,
                            "entropy={entropy_mode} after resize, frame={frame_idx}: NAL length {length} > {max_nal_size}"
                        );
                    }
                }
            }

            assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
            WelsDestroySVCEncoder(encoder);
        }
    }
}
