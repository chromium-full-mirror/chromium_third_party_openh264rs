// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Integration tests for OpenH264 parameter validation and boundary error checks.
//! C++: `test/api/encoder_test.cpp`, `test/api/decode_api_test.cpp`.

use openh264rs::api::codec_api::*;
use std::ffi::c_long;

#[test]
pub(crate) fn test_encoder_invalid_resolution_rejected() {
    let mut enc = Encoder::new();

    let mut param = SEncParamBase::default();
    param.iPicWidth = 0;
    param.iPicHeight = 240;
    param.fMaxFrameRate = 30.0;

    let init_ret = enc.initialize(&param);
    assert_ne!(init_ret, CM_RESULT_SUCCESS);
}

// This configuration is rejected with `cmInitParaError`: `iTargetBitrate` is left at 0
// while `iRCMode` defaults to `RC_QUALITY_MODE`, and `ParamValidation()` rejects
// `iTargetBitrate <= 0` for every RC mode but `RC_OFF`.
#[test]
pub(crate) fn test_encoder_very_large_slices() {
    let mut enc = Encoder::new();

    let mut param = SEncParamExt::default();
    let get_ret = enc.default_params(&mut param);
    assert_eq!(get_ret, CM_RESULT_SUCCESS);

    param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
    param.iPicWidth = 1280;
    param.iPicHeight = 720;
    param.fMaxFrameRate = 30.0;
    param.iSpatialLayerNum = 1;
    param.iMultipleThreadIdc = 4;

    param.sSpatialLayers[0].iVideoWidth = param.iPicWidth;
    param.sSpatialLayers[0].iVideoHeight = param.iPicHeight;
    param.sSpatialLayers[0].fFrameRate = param.fMaxFrameRate;
    param.sSpatialLayers[0].sSliceArgument.uiSliceMode = SliceModeEnum::SM_FIXEDSLCNUM_SLICE;
    param.sSpatialLayers[0].sSliceArgument.uiSliceNum = 4;
    param.sSpatialLayers[0].iDLayerQp = 12;

    let init_ret = enc.initialize_ext(&param);
    assert_eq!(init_ret, CM_INIT_PARA_ERROR);

    enc.uninitialize();
}

/// C++: `EncoderInitTest.ScreenContentScrollMotionVectorBounds`,
/// `test/api/encoder_test.cpp`.
///
/// A 640x1800 screen encode initializes and encodes two frames, the second scrolled
/// down by 512 rows with one macroblock corrupted, so scroll detection predicts a large
/// vertical displacement and MVD-table indexing has to stay in bounds. The layer size
/// is aligned to 16 and cropped, so `ParamValidationExt`'s `& 0x0F` check sees 1808 and
/// the 1800-row height is accepted.
#[test]
pub(crate) fn test_encoder_screen_content_scroll_motion_vector_bounds() {
    let mut enc = Encoder::new();

    let mut param = SEncParamExt::default();
    let get_ret = enc.default_params(&mut param);
    assert_eq!(get_ret, CM_RESULT_SUCCESS);

    param.iUsageType = EUsageType::SCREEN_CONTENT_REAL_TIME;
    param.iPicWidth = 640;
    param.iPicHeight = 1800;
    param.fMaxFrameRate = 30.0;
    param.iSpatialLayerNum = 1;
    param.iRCMode = RC_MODES::RC_OFF_MODE;

    param.sSpatialLayers[0].iVideoWidth = param.iPicWidth;
    param.sSpatialLayers[0].iVideoHeight = param.iPicHeight;
    param.sSpatialLayers[0].fFrameRate = param.fMaxFrameRate;
    param.sSpatialLayers[0].sSliceArgument.uiSliceMode = SliceModeEnum::SM_SINGLE_SLICE;
    param.sSpatialLayers[0].iDLayerQp = 51;
    param.iMinQp = 51;
    param.iMaxQp = 51;

    let rv = enc.initialize_ext(&param);
    assert_eq!(rv, 0);

    let width = param.iPicWidth as usize;
    let height = param.iPicHeight as usize;
    let frame_size = width * height * 3 / 2;
    let mut frame0 = vec![128u8; frame_size];
    let mut frame1 = vec![128u8; frame_size];

    // Fill frame0 luma with pseudo-random patterns to ensure CheckLine qualifies
    for y in 0..height {
        for x in 0..width {
            frame0[y * width + x] = ((x * 29 + y * 43 + 17) % 251) as u8;
        }
    }

    // Frame 1: shift vertically by -512 pixels (content moving downward by 512)
    let scroll_mv: isize = -512;
    for y in 0..height {
        let src = y as isize + scroll_mv;
        if src >= 0 && src < height as isize {
            let src = src as usize;
            frame1[y * width..(y + 1) * width]
                .copy_from_slice(&frame0[src * width..(src + 1) * width]);
        } else {
            for x in 0..width {
                frame1[y * width + x] = ((x * 53 + y * 71 + 101) & 0xFF) as u8;
            }
        }
    }

    // Modify macroblock (1, 32) at pixel rows 512..527 and cols 16..31 in frame1.
    // MB(0, 32) is skipped by scroll detection with MV -2048; MB(1, 32) cannot be,
    // so it enters motion estimation with MB(0, 32)'s scrolled MV as a predictor.
    for y in 512..528 {
        for x in 16..32 {
            frame1[y * width + x] ^= 0xFF;
        }
    }

    let mut info = SFrameBSInfo::default();

    let pic0 = SourcePicture::from_i420(&frame0, width as i32, height as i32);
    let rv = enc.encode_frame(&pic0, &mut info);
    assert_eq!(rv, 0);

    let pic1 = SourcePicture::from_i420(&frame1, width as i32, height as i32).with_timestamp(33);
    let rv = enc.encode_frame(&pic1, &mut info);
    assert_eq!(rv, 0);

    enc.uninitialize();
}

#[test]
pub(crate) fn test_decoder_get_set_options_vcl_nal_framenum_idr_isref() {
    let mut dec = Decoder::new();

    let mut dec_param = SDecodingParam::default();
    dec_param.uiTargetDqLayer = u8::MAX;
    let init_ret = dec.initialize(&dec_param);
    assert_eq!(init_ret, CM_RESULT_SUCCESS as c_long);

    assert!(dec.feedback_vcl_nal().is_some());
    assert!(dec.frame_num().is_some());
    assert!(dec.cur_idr_pic_id().is_some());
    assert!(dec.feedback_is_ref_pic().is_some());

    dec.uninitialize();
}

/// C++: `EncoderInitTest.CustomChromaPlaneStrides`, `test/api/encoder_test.cpp`.
#[test]
pub(crate) fn test_encoder_custom_chroma_plane_strides() {
    fn generate_pattern(
        y_buf: &mut [u8],
        y_stride: usize,
        u_buf: &mut [u8],
        u_stride: usize,
        v_buf: &mut [u8],
        v_stride: usize,
        width: usize,
        height: usize,
    ) {
        for y in 0..height {
            for x in 0..width.min(y_stride) {
                y_buf[y * y_stride + x] = ((x * 4 + y * 4) % 256) as u8;
            }
        }
        for y in 0..(height >> 1) {
            for x in 0..(width >> 1) {
                if x < u_stride {
                    u_buf[y * u_stride + x] = ((x * 8) % 256) as u8;
                }
                if x < v_stride {
                    v_buf[y * v_stride + x] = ((y * 8) % 256) as u8;
                }
            }
        }
    }

    fn calculate_plane_psnr(
        ref_buf: &[u8],
        ref_stride: usize,
        test_buf: &[u8],
        test_stride: usize,
        width: usize,
        height: usize,
    ) -> f64 {
        let mut mse = 0.0f64;
        let compare_width = width.min(ref_stride).min(test_stride);
        for y in 0..height {
            for x in 0..compare_width {
                let r = ref_buf[y * ref_stride + x] as f64;
                let t = test_buf[y * test_stride + x] as f64;
                let diff = r - t;
                mse += diff * diff;
            }
        }
        mse /= (compare_width * height) as f64;
        if mse == 0.0 {
            return 99.0;
        }
        10.0 * ((255.0 * 255.0) / mse).log10()
    }

    let mut enc = Encoder::new();

    let mut param = SEncParamExt::default();
    assert_eq!(enc.default_params(&mut param), CM_RESULT_SUCCESS);

    param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
    param.iPicWidth = 64;
    param.iPicHeight = 64;
    param.fMaxFrameRate = 30.0;
    param.iSpatialLayerNum = 1;
    param.iRCMode = RC_MODES::RC_OFF_MODE;

    param.sSpatialLayers[0].iVideoWidth = param.iPicWidth;
    param.sSpatialLayers[0].iVideoHeight = param.iPicHeight;
    param.sSpatialLayers[0].fFrameRate = param.fMaxFrameRate;
    param.sSpatialLayers[0].sSliceArgument.uiSliceMode = SliceModeEnum::SM_SINGLE_SLICE;
    param.sSpatialLayers[0].iDLayerQp = 0;

    assert_eq!(enc.initialize_ext(&param), CM_RESULT_SUCCESS);

    let mut dec = Decoder::new();

    let mut dec_param = SDecodingParam::default();
    dec_param.uiTargetDqLayer = u8::MAX;
    dec_param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
    dec_param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
    assert_eq!(dec.initialize(&dec_param), CM_RESULT_SUCCESS as _);

    let stride_y: usize = 64;
    let stride_u: usize = 4096;
    let stride_v: usize = 32;
    let width = param.iPicWidth as usize;
    let height = param.iPicHeight as usize;

    let mut buf_y = vec![0u8; stride_y * height];
    let mut buf_u = vec![0u8; stride_u * (height >> 1)];
    let mut buf_v = vec![0u8; stride_v * (height >> 1)];
    generate_pattern(
        &mut buf_y, stride_y, &mut buf_u, stride_u, &mut buf_v, stride_v, width, height,
    );

    let pic = SourcePicture::from_i420_strided(
        &buf_y,
        &buf_u,
        &buf_v,
        param.iPicWidth,
        param.iPicHeight,
        [stride_y as i32, stride_u as i32, stride_v as i32],
    );

    let mut info = SFrameBSInfo::default();
    assert_eq!(enc.encode_frame(&pic, &mut info), CM_RESULT_SUCCESS);

    let bs: Vec<u8> = info
        .layers()
        .iter()
        .flat_map(|l| l.bs_buf(&enc))
        .copied()
        .collect();
    assert!(!bs.is_empty());

    let mut dst_buf_info = SBufferInfo::default();
    let rv = dec.decode(Some(&bs), &mut dst_buf_info);
    assert_eq!(rv, DECODING_STATE::dsErrorFree);

    if dst_buf_info.iBufferStatus == 0 {
        let rv = dec.decode(None, &mut dst_buf_info);
        assert_eq!(rv, DECODING_STATE::dsErrorFree);
    }
    assert_eq!(dst_buf_info.iBufferStatus, 1);

    let decoded_width_u = (dst_buf_info.UsrData.sSystemBuffer.iWidth >> 1) as usize;
    let decoded_height_u = (dst_buf_info.UsrData.sSystemBuffer.iHeight >> 1) as usize;
    let stride_dec_y = dst_buf_info.UsrData.sSystemBuffer.iStride[0] as usize;
    let stride_dec_chroma = dst_buf_info.UsrData.sSystemBuffer.iStride[1] as usize;

    let (dec_y, dec_u, dec_v) = dst_buf_info.yuv_planes(&dec).expect("decoded YUV planes");

    let psnr_y = calculate_plane_psnr(&buf_y, stride_y, dec_y, stride_dec_y, width, height);
    let psnr_u = calculate_plane_psnr(
        &buf_u,
        stride_u,
        dec_u,
        stride_dec_chroma,
        decoded_width_u,
        decoded_height_u,
    );
    let psnr_v = calculate_plane_psnr(
        &buf_v,
        stride_v,
        dec_v,
        stride_dec_chroma,
        decoded_width_u,
        decoded_height_u,
    );

    assert!(psnr_y > 40.0, "psnr_y = {psnr_y}");
    assert!(psnr_u > 40.0, "psnr_u = {psnr_u}");
    assert!(psnr_v > 40.0, "psnr_v = {psnr_v}");

    dec.uninitialize();
    enc.uninitialize();
}

/// C++: `EncoderInitTest.CustomChromaPlaneStridesInvalidSrc`, `test/api/encoder_test.cpp`.
#[test]
pub(crate) fn test_encoder_custom_chroma_plane_strides_invalid_src() {
    let mut enc = Encoder::new();

    let mut param = SEncParamExt::default();
    assert_eq!(enc.default_params(&mut param), CM_RESULT_SUCCESS);

    param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
    param.iPicWidth = 64;
    param.iPicHeight = 64;
    param.fMaxFrameRate = 30.0;
    param.iSpatialLayerNum = 1;
    param.iRCMode = RC_MODES::RC_OFF_MODE;

    param.sSpatialLayers[0].iVideoWidth = param.iPicWidth;
    param.sSpatialLayers[0].iVideoHeight = param.iPicHeight;
    param.sSpatialLayers[0].fFrameRate = param.fMaxFrameRate;
    param.sSpatialLayers[0].sSliceArgument.uiSliceMode = SliceModeEnum::SM_SINGLE_SLICE;
    param.sSpatialLayers[0].iDLayerQp = 0;

    assert_eq!(enc.initialize_ext(&param), CM_RESULT_SUCCESS);

    let stride_y: usize = 64;
    let stride_u: usize = 16; // Invalid: smaller than width / 2 (32)
    let stride_v: usize = 32;
    let height = param.iPicHeight as usize;

    let buf_y = vec![128u8; stride_y * height];
    let buf_u = vec![100u8; stride_u * (height >> 1)];
    let buf_v = vec![200u8; stride_v * (height >> 1)];

    let pic = SourcePicture::from_i420_strided(
        &buf_y,
        &buf_u,
        &buf_v,
        param.iPicWidth,
        param.iPicHeight,
        [stride_y as i32, stride_u as i32, stride_v as i32],
    );

    let mut info = SFrameBSInfo::default();
    assert_eq!(enc.encode_frame(&pic, &mut info), CM_UNSUPPORTED_DATA);

    enc.uninitialize();
}

#[test]
pub(crate) fn test_encoder_initialize_ext_rejects_geometry_overflow() {
    // C++ test: EncoderInterfaceTest.InitializeExtRejectsGeometryOverflow
    let mut enc = Encoder::new();

    let mut param = SEncParamExt::default();
    enc.default_params(&mut param);

    // 46368 * 46368 overflows signed int32_t (46368 * 46368 = 2,149,991,424 > 2,147,483,647)
    param.iPicWidth = 46368;
    param.iPicHeight = 46368;
    param.iSpatialLayerNum = 1;
    param.iTemporalLayerNum = 1;
    param.iTargetBitrate = 500000;
    param.fMaxFrameRate = 30.0;
    param.sSpatialLayers[0].iVideoWidth = param.iPicWidth;
    param.sSpatialLayers[0].iVideoHeight = param.iPicHeight;
    param.sSpatialLayers[0].iSpatialBitrate = param.iTargetBitrate;
    param.sSpatialLayers[0].fFrameRate = param.fMaxFrameRate;

    let init_ret = enc.initialize_ext(&param);
    assert_ne!(init_ret, CM_RESULT_SUCCESS);
}
