// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Integration tests for OpenH264 parameter validation and boundary error checks.
//! C++: `test/api/encoder_test.cpp`, `test/api/decode_api_test.cpp`.

use openh264rs::api::codec_api::*;

#[test]
fn test_encoder_null_param_rejected() {
    unsafe {
        let mut p_encoder: *mut ISVCEncoder = std::ptr::null_mut();
        let ret = WelsCreateSVCEncoder(&mut p_encoder);
        assert_eq!(ret, CM_RESULT_SUCCESS);
        assert!(!p_encoder.is_null());

        let init_ret = ISVCEncoder::Initialize(p_encoder, std::ptr::null());
        assert_ne!(init_ret, CM_RESULT_SUCCESS);

        WelsDestroySVCEncoder(p_encoder);
    }
}

#[test]
fn test_encoder_invalid_resolution_rejected() {
    unsafe {
        let mut p_encoder: *mut ISVCEncoder = std::ptr::null_mut();
        let ret = WelsCreateSVCEncoder(&mut p_encoder);
        assert_eq!(ret, CM_RESULT_SUCCESS);

        let mut param = SEncParamBase::default();
        param.iPicWidth = 0;
        param.iPicHeight = 240;
        param.fMaxFrameRate = 30.0;

        let init_ret = ISVCEncoder::Initialize(p_encoder, &param as *const SEncParamBase);
        assert_ne!(init_ret, CM_RESULT_SUCCESS);

        WelsDestroySVCEncoder(p_encoder);
    }
}

#[test]
fn test_decoder_null_param_rejected() {
    unsafe {
        let mut p_decoder: *mut ISVCDecoder = std::ptr::null_mut();
        let ret = WelsCreateDecoder(&mut p_decoder);
        assert_eq!(ret, CM_RESULT_SUCCESS as i64);
        assert!(!p_decoder.is_null());

        let init_ret = ISVCDecoder::Initialize(p_decoder, std::ptr::null());
        assert_ne!(init_ret, CM_RESULT_SUCCESS as i64);

        WelsDestroyDecoder(p_decoder);
    }
}

// This configuration is rejected with `cmInitParaError`: `iTargetBitrate` is left at 0
// while `iRCMode` defaults to `RC_QUALITY_MODE`, and `ParamValidation()` rejects
// `iTargetBitrate <= 0` for every RC mode but `RC_OFF`.
#[test]
fn test_encoder_very_large_slices() {
    unsafe {
        let mut p_encoder: *mut ISVCEncoder = std::ptr::null_mut();
        let ret = WelsCreateSVCEncoder(&mut p_encoder);
        assert_eq!(ret, CM_RESULT_SUCCESS);

        let mut param = SEncParamExt::default();
        let get_ret = ISVCEncoder::GetDefaultParams(p_encoder, &mut param as *mut SEncParamExt);
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

        let init_ret = ISVCEncoder::InitializeExt(p_encoder, &param as *const SEncParamExt);
        assert_eq!(init_ret, CM_INIT_PARA_ERROR);

        ISVCEncoder::Uninitialize(p_encoder);
        WelsDestroySVCEncoder(p_encoder);
    }
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
fn test_encoder_screen_content_scroll_motion_vector_bounds() {
    unsafe {
        let mut p_encoder: *mut ISVCEncoder = std::ptr::null_mut();
        let ret = WelsCreateSVCEncoder(&mut p_encoder);
        assert_eq!(ret, CM_RESULT_SUCCESS);

        let mut param = SEncParamExt::default();
        let get_ret = ISVCEncoder::GetDefaultParams(p_encoder, &mut param as *mut SEncParamExt);
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

        let rv = ISVCEncoder::InitializeExt(p_encoder, &param as *const SEncParamExt);
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

        let luma = width * height;
        let mut pic = SSourcePicture::default();
        pic.iPicWidth = width as i32;
        pic.iPicHeight = height as i32;
        pic.iColorFormat = EVideoFormatType::videoFormatI420 as i32;
        pic.iStride[0] = width as i32;
        pic.iStride[1] = (width >> 1) as i32;
        pic.iStride[2] = (width >> 1) as i32;
        pic.pData[0] = frame0.as_mut_ptr();
        pic.pData[1] = frame0.as_mut_ptr().add(luma);
        pic.pData[2] = frame0.as_mut_ptr().add(luma + (luma >> 2));

        let mut info = SFrameBSInfo::default();

        let rv = ISVCEncoder::EncodeFrame(p_encoder, &pic as *const SSourcePicture, &mut info);
        assert_eq!(rv, 0);
        pic.uiTimeStamp += 33;

        pic.pData[0] = frame1.as_mut_ptr();
        pic.pData[1] = frame1.as_mut_ptr().add(luma);
        pic.pData[2] = frame1.as_mut_ptr().add(luma + (luma >> 2));
        let rv = ISVCEncoder::EncodeFrame(p_encoder, &pic as *const SSourcePicture, &mut info);
        assert_eq!(rv, 0);

        ISVCEncoder::Uninitialize(p_encoder);
        WelsDestroySVCEncoder(p_encoder);
    }
}

#[test]
fn test_decoder_get_set_options_vcl_nal_framenum_idr_isref() {
    unsafe {
        let mut p_decoder: *mut ISVCDecoder = std::ptr::null_mut();
        let ret = WelsCreateDecoder(&mut p_decoder);
        assert_eq!(ret, CM_RESULT_SUCCESS as i64);

        let mut dec_param = SDecodingParam::default();
        dec_param.uiTargetDqLayer = u8::MAX;
        let init_ret = ISVCDecoder::Initialize(p_decoder, &dec_param as *const SDecodingParam);
        assert_eq!(init_ret, CM_RESULT_SUCCESS as i64);

        let mut vcl_nal = 0i32;
        let opt_vcl = ISVCDecoder::GetOption(
            p_decoder,
            DECODER_OPTION::DECODER_OPTION_VCL_NAL,
            &mut vcl_nal as *mut i32 as *mut std::ffi::c_void,
        );
        assert_eq!(opt_vcl, CM_RESULT_SUCCESS as i64);

        let mut frame_num = 0i32;
        let opt_fn = ISVCDecoder::GetOption(
            p_decoder,
            DECODER_OPTION::DECODER_OPTION_FRAME_NUM,
            &mut frame_num as *mut i32 as *mut std::ffi::c_void,
        );
        assert_eq!(opt_fn, CM_RESULT_SUCCESS as i64);

        let mut idr_id = 0i32;
        let opt_idr = ISVCDecoder::GetOption(
            p_decoder,
            DECODER_OPTION::DECODER_OPTION_IDR_PIC_ID,
            &mut idr_id as *mut i32 as *mut std::ffi::c_void,
        );
        assert_eq!(opt_idr, CM_RESULT_SUCCESS as i64);

        let mut is_ref = 0i32;
        let opt_ref = ISVCDecoder::GetOption(
            p_decoder,
            DECODER_OPTION::DECODER_OPTION_IS_REF_PIC,
            &mut is_ref as *mut i32 as *mut std::ffi::c_void,
        );
        assert_eq!(opt_ref, CM_RESULT_SUCCESS as i64);

        ISVCDecoder::Uninitialize(p_decoder);
        WelsDestroyDecoder(p_decoder);
    }
}

/// C++: `EncoderInitTest.CustomChromaPlaneStrides`, `test/api/encoder_test.cpp`.
#[test]
fn test_encoder_custom_chroma_plane_strides() {
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

    unsafe fn calculate_plane_psnr(
        ref_buf: &[u8],
        ref_stride: usize,
        test_ptr: *const u8,
        test_stride: usize,
        width: usize,
        height: usize,
    ) -> f64 {
        let mut mse = 0.0f64;
        let compare_width = width.min(ref_stride).min(test_stride);
        for y in 0..height {
            for x in 0..compare_width {
                let r = ref_buf[y * ref_stride + x] as f64;
                let t = unsafe { *test_ptr.add(y * test_stride + x) } as f64;
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

    unsafe {
        let mut p_encoder: *mut ISVCEncoder = std::ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut p_encoder), CM_RESULT_SUCCESS);

        let mut param = SEncParamExt::default();
        assert_eq!(
            ISVCEncoder::GetDefaultParams(p_encoder, &mut param),
            CM_RESULT_SUCCESS
        );

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

        assert_eq!(
            ISVCEncoder::InitializeExt(p_encoder, &param),
            CM_RESULT_SUCCESS
        );

        let mut p_decoder: *mut ISVCDecoder = std::ptr::null_mut();
        assert_eq!(WelsCreateDecoder(&mut p_decoder), CM_RESULT_SUCCESS as _);
        assert!(!p_decoder.is_null());

        let mut dec_param = SDecodingParam::default();
        dec_param.uiTargetDqLayer = u8::MAX;
        dec_param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
        dec_param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
        assert_eq!(
            ISVCDecoder::Initialize(p_decoder, &dec_param),
            CM_RESULT_SUCCESS as _
        );

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

        let pic = SSourcePicture {
            iColorFormat: EVideoFormatType::videoFormatI420 as i32,
            iStride: [stride_y as i32, stride_u as i32, stride_v as i32, 0],
            pData: [
                buf_y.as_mut_ptr(),
                buf_u.as_mut_ptr(),
                buf_v.as_mut_ptr(),
                std::ptr::null_mut(),
            ],
            iPicWidth: param.iPicWidth,
            iPicHeight: param.iPicHeight,
            uiTimeStamp: 0,
            ..SSourcePicture::default()
        };

        let mut info = SFrameBSInfo::default();
        assert_eq!(
            ISVCEncoder::EncodeFrame(p_encoder, &pic, &mut info),
            CM_RESULT_SUCCESS
        );

        let mut len = 0usize;
        for i in 0..info.iLayerNum as usize {
            let layer_info = &info.sLayerInfo[i];
            for j in 0..layer_info.iNalCount as usize {
                len += *layer_info.pNalLengthInByte.add(j) as usize;
            }
        }
        assert!(len > 0);

        let mut p_data: [*mut u8; 3] = [std::ptr::null_mut(); 3];
        let mut dst_buf_info = SBufferInfo::default();
        let rv = ISVCDecoder::DecodeFrame2(
            p_decoder,
            info.sLayerInfo[0].pBsBuf,
            len as i32,
            p_data.as_mut_ptr(),
            &mut dst_buf_info,
        );
        assert_eq!(rv, DECODING_STATE::dsErrorFree);

        if dst_buf_info.iBufferStatus == 0 {
            let rv = ISVCDecoder::DecodeFrame2(
                p_decoder,
                std::ptr::null(),
                0,
                p_data.as_mut_ptr(),
                &mut dst_buf_info,
            );
            assert_eq!(rv, DECODING_STATE::dsErrorFree);
        }
        assert_eq!(dst_buf_info.iBufferStatus, 1);

        let decoded_width_u = (dst_buf_info.UsrData.sSystemBuffer.iWidth >> 1) as usize;
        let decoded_height_u = (dst_buf_info.UsrData.sSystemBuffer.iHeight >> 1) as usize;
        let stride_dec_y = dst_buf_info.UsrData.sSystemBuffer.iStride[0] as usize;
        let stride_dec_chroma = dst_buf_info.UsrData.sSystemBuffer.iStride[1] as usize;

        assert!(!p_data[0].is_null());
        assert!(!p_data[1].is_null());
        assert!(!p_data[2].is_null());

        let psnr_y = calculate_plane_psnr(&buf_y, stride_y, p_data[0], stride_dec_y, width, height);
        let psnr_u = calculate_plane_psnr(
            &buf_u,
            stride_u,
            p_data[1],
            stride_dec_chroma,
            decoded_width_u,
            decoded_height_u,
        );
        let psnr_v = calculate_plane_psnr(
            &buf_v,
            stride_v,
            p_data[2],
            stride_dec_chroma,
            decoded_width_u,
            decoded_height_u,
        );

        assert!(psnr_y > 40.0, "psnr_y = {psnr_y}");
        assert!(psnr_u > 40.0, "psnr_u = {psnr_u}");
        assert!(psnr_v > 40.0, "psnr_v = {psnr_v}");

        ISVCDecoder::Uninitialize(p_decoder);
        WelsDestroyDecoder(p_decoder);
        ISVCEncoder::Uninitialize(p_encoder);
        WelsDestroySVCEncoder(p_encoder);
    }
}

/// C++: `EncoderInitTest.CustomChromaPlaneStridesInvalidSrc`, `test/api/encoder_test.cpp`.
#[test]
fn test_encoder_custom_chroma_plane_strides_invalid_src() {
    unsafe {
        let mut p_encoder: *mut ISVCEncoder = std::ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut p_encoder), CM_RESULT_SUCCESS);

        let mut param = SEncParamExt::default();
        assert_eq!(
            ISVCEncoder::GetDefaultParams(p_encoder, &mut param),
            CM_RESULT_SUCCESS
        );

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

        assert_eq!(
            ISVCEncoder::InitializeExt(p_encoder, &param),
            CM_RESULT_SUCCESS
        );

        let stride_y: usize = 64;
        let stride_u: usize = 16; // Invalid: smaller than width / 2 (32)
        let stride_v: usize = 32;
        let height = param.iPicHeight as usize;

        let mut buf_y = vec![128u8; stride_y * height];
        let mut buf_u = vec![100u8; stride_u * (height >> 1)];
        let mut buf_v = vec![200u8; stride_v * (height >> 1)];

        let pic = SSourcePicture {
            iColorFormat: EVideoFormatType::videoFormatI420 as i32,
            iStride: [stride_y as i32, stride_u as i32, stride_v as i32, 0],
            pData: [
                buf_y.as_mut_ptr(),
                buf_u.as_mut_ptr(),
                buf_v.as_mut_ptr(),
                std::ptr::null_mut(),
            ],
            iPicWidth: param.iPicWidth,
            iPicHeight: param.iPicHeight,
            uiTimeStamp: 0,
            ..SSourcePicture::default()
        };

        let mut info = SFrameBSInfo::default();
        assert_eq!(
            ISVCEncoder::EncodeFrame(p_encoder, &pic, &mut info),
            CM_UNSUPPORTED_DATA
        );

        ISVCEncoder::Uninitialize(p_encoder);
        WelsDestroySVCEncoder(p_encoder);
    }
}

#[test]
fn test_encoder_initialize_ext_rejects_geometry_overflow() {
    // C++ test: EncoderInterfaceTest.InitializeExtRejectsGeometryOverflow
    unsafe {
        let mut p_encoder: *mut ISVCEncoder = std::ptr::null_mut();
        let ret = WelsCreateSVCEncoder(&mut p_encoder);
        assert_eq!(ret, CM_RESULT_SUCCESS);
        assert!(!p_encoder.is_null());

        let mut param = SEncParamExt::default();
        let vtbl = (*p_encoder).lpVtbl;
        ((*vtbl).GetDefaultParams)(p_encoder, &mut param);

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

        let init_ret = ((*vtbl).InitializeExt)(p_encoder, &param);
        assert_ne!(init_ret, CM_RESULT_SUCCESS);

        WelsDestroySVCEncoder(p_encoder);
    }
}
