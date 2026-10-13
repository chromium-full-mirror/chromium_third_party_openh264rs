// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Integration test for dynamic encoder options and runtime reconfiguration.
//! Ported from upstream `test/api/encode_options_test.cpp`.

mod common;

use common::frame_gen::{fill_camera_frame, fill_screen_frame, fill_seeded_noise};
use openh264rs::api::codec_api::*;
use openh264rs::encoder::wels_encoder_ext::{SLevelInfo, SProfileInfo};
use std::ffi::c_void;
use std::ptr;

#[test]
fn test_encoder_set_and_get_options() {
    unsafe {
        let mut p_encoder: *mut ISVCEncoder = std::ptr::null_mut();
        let ret = WelsCreateSVCEncoder(&mut p_encoder);
        assert_eq!(ret, CM_RESULT_SUCCESS);
        assert!(!p_encoder.is_null());

        let mut param = SEncParamBase::default();
        param.iPicWidth = 320;
        param.iPicHeight = 240;
        param.fMaxFrameRate = 30.0;
        param.iTargetBitrate = 500000;
        param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;

        let init_ret = ISVCEncoder::Initialize(p_encoder, &param as *const SEncParamBase);
        assert_eq!(init_ret, CM_RESULT_SUCCESS);

        // 1. Test frame rate option modification
        let mut f_fps: f32 = 15.0;
        let opt_ret = ISVCEncoder::SetOption(
            p_encoder,
            ENCODER_OPTION::ENCODER_OPTION_FRAME_RATE,
            &mut f_fps as *mut f32 as *mut std::ffi::c_void,
        );
        assert_eq!(opt_ret, CM_RESULT_SUCCESS);

        // 2. Test bitrate option modification
        let mut bitrate_info = SBitrateInfo::default();
        bitrate_info.iBitrate = 300000;
        let opt_br_ret = ISVCEncoder::SetOption(
            p_encoder,
            ENCODER_OPTION::ENCODER_OPTION_BITRATE,
            &mut bitrate_info as *mut SBitrateInfo as *mut std::ffi::c_void,
        );
        assert_eq!(opt_br_ret, CM_RESULT_SUCCESS);

        // 3. Test IDR interval option modification
        let mut idr_interval: i32 = 60;
        let opt_idr_ret = ISVCEncoder::SetOption(
            p_encoder,
            ENCODER_OPTION::ENCODER_OPTION_IDR_INTERVAL,
            &mut idr_interval as *mut i32 as *mut std::ffi::c_void,
        );
        assert_eq!(opt_idr_ret, CM_RESULT_SUCCESS);

        // 4. Test SEncParamBase query and set
        let mut base_param = SEncParamBase::default();
        let get_base_ret = ISVCEncoder::GetOption(
            p_encoder,
            ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_BASE,
            &mut base_param as *mut SEncParamBase as *mut std::ffi::c_void,
        );
        assert_eq!(get_base_ret, CM_RESULT_SUCCESS);

        // 5. Test SEncParamExt query and set
        let mut ext_param = SEncParamExt::default();
        let get_ext_ret = ISVCEncoder::GetOption(
            p_encoder,
            ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
            &mut ext_param as *mut SEncParamExt as *mut std::ffi::c_void,
        );
        assert_eq!(get_ext_ret, CM_RESULT_SUCCESS);

        // 6. Test statistics option query
        let mut stats = SEncoderStatistics::default();
        let get_stats_ret = ISVCEncoder::GetOption(
            p_encoder,
            ENCODER_OPTION::ENCODER_OPTION_GET_STATISTICS,
            &mut stats as *mut SEncoderStatistics as *mut std::ffi::c_void,
        );
        assert_eq!(get_stats_ret, CM_RESULT_SUCCESS);

        let uninit_ret = ISVCEncoder::Uninitialize(p_encoder);
        assert_eq!(uninit_ret, CM_RESULT_SUCCESS);

        WelsDestroySVCEncoder(p_encoder);
    }
}

/// Every `ENCODER_OPTION_*` value `CWelsH264SVCEncoder` handles, driven through
/// `SetOption`/`GetOption` on a 160x96 configuration.
#[test]
fn test_set_get_option_matches_cxx_for_every_option() {
    use openh264rs::encoder::ref_list_mgr_svc::{SLTRMarkingFeedback, SLTRRecoverRequest};
    use openh264rs::encoder::wels_encoder_ext::{
        SDeliveryStatus, SLTRConfig, SLevelInfo, SProfileInfo,
    };

    unsafe {
        let mut p_encoder: *mut ISVCEncoder = std::ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut p_encoder), CM_RESULT_SUCCESS);

        let mut param = SEncParamBase::default();
        param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
        param.iPicWidth = 160;
        param.iPicHeight = 96;
        param.iTargetBitrate = 500000;
        param.iRCMode = RC_MODES::RC_QUALITY_MODE;
        param.fMaxFrameRate = 6.0;
        assert_eq!(
            ISVCEncoder::Initialize(p_encoder, &param as *const SEncParamBase),
            CM_RESULT_SUCCESS
        );

        // ---- GetOption: which ids are answered at all -------------------------
        // 11 cases and a `default: return cmInitParaError`; the return code for every
        // id 0..=31.
        const GET_EXPECTED: [i32; 32] = [
            0, 0, 0, 0, 0, 0, 0, 0, // DATAFORMAT..INTER_SPATIAL_PRED
            1, 1, 1, 1, 1, 1, 1, 1, // RC_MODE..LTR_RECOVERY_REQUEST
            1, 1, 1, 0, 1, 1, 1, 1, // LTR_MARKING_FEEDBACK..CURRENT_PATH
            1, 1, 1, 1, 0, 0, 1, 1, // DUMP_FILE..BITS_VARY_PERCENTAGE
        ];
        let mut buf = [0u8; 8192];
        for id in 0..32i32 {
            buf.fill(0);
            if id == ENCODER_OPTION::ENCODER_OPTION_BITRATE as i32
                || id == ENCODER_OPTION::ENCODER_OPTION_MAX_BITRATE as i32
            {
                // A layer id outside `SPATIAL_LAYER_*` is a legitimate error.
                let bi = buf.as_mut_ptr() as *mut SBitrateInfo;
                (*bi).iLayer = LAYER_NUM::SPATIAL_LAYER_ALL;
            }
            let e: ENCODER_OPTION = std::mem::transmute(id);
            let got =
                ISVCEncoder::GetOption(p_encoder, e, buf.as_mut_ptr() as *mut std::ffi::c_void);
            assert_eq!(
                got, GET_EXPECTED[id as usize],
                "GetOption(id={id}) returned {got}, C++ returns {}",
                GET_EXPECTED[id as usize]
            );
        }

        // ---- SetOption --------------------------------------------------------
        let set =
            |e: ENCODER_OPTION, p: *mut std::ffi::c_void| ISVCEncoder::SetOption(p_encoder, e, p);
        macro_rules! setopt {
            ($id:ident, $val:expr) => {{
                let mut v = $val;
                set(
                    ENCODER_OPTION::$id,
                    &mut v as *mut _ as *mut std::ffi::c_void,
                )
            }};
        }

        assert_eq!(setopt!(ENCODER_OPTION_INTER_SPATIAL_PRED, 1i32), 0);
        assert_eq!(setopt!(ENCODER_OPTION_RC_MODE, 1i32), 0);
        assert_eq!(setopt!(ENCODER_OPTION_RC_FRAME_SKIP, true), 0);
        assert_eq!(setopt!(ENCODER_PADDING_PADDING, 1i32), 0);
        assert_eq!(setopt!(ENCODER_LTR_MARKING_PERIOD, 30u32), 0);
        assert_eq!(
            setopt!(
                ENCODER_OPTION_LTR,
                SLTRConfig {
                    bEnableLongTermReference: true,
                    iLTRRefNum: 2,
                }
            ),
            0
        );
        assert_eq!(setopt!(ENCODER_OPTION_ENABLE_SSEI, true), 0);
        assert_eq!(setopt!(ENCODER_OPTION_ENABLE_PREFIX_NAL_ADDING, true), 0);
        assert_eq!(setopt!(ENCODER_OPTION_SPS_PPS_ID_STRATEGY, 1i32), 0);
        // Out of range: logged, `eNewStrategy` stays at `CONSTANT_ID`, and it is
        // still applied — not an error.
        assert_eq!(setopt!(ENCODER_OPTION_SPS_PPS_ID_STRATEGY, 99i32), 0);
        assert_eq!(
            setopt!(
                ENCODER_OPTION_PROFILE,
                SProfileInfo {
                    iLayer: LAYER_NUM::SPATIAL_LAYER_0 as i32,
                    uiProfileIdc: EProfileIdc::PRO_HIGH,
                }
            ),
            0
        );
        // iLayer outside SPATIAL_LAYER_0..SPATIAL_LAYER_3 is rejected.
        assert_eq!(
            setopt!(
                ENCODER_OPTION_PROFILE,
                SProfileInfo {
                    iLayer: 7,
                    uiProfileIdc: EProfileIdc::PRO_HIGH,
                }
            ),
            CM_INIT_PARA_ERROR
        );
        assert_eq!(
            setopt!(
                ENCODER_OPTION_LEVEL,
                SLevelInfo {
                    iLayer: LAYER_NUM::SPATIAL_LAYER_0 as i32,
                    uiLevelIdc: ELevelIdc::LEVEL_3_0,
                }
            ),
            0
        );
        assert_eq!(setopt!(ENCODER_OPTION_NUMBER_REF, 3i32), 0);
        assert_eq!(
            setopt!(
                ENCODER_OPTION_DELIVERY_STATUS,
                SDeliveryStatus {
                    bDeliveryFlag: true,
                    // Reserved fields, read by nothing; the struct a caller passes
                    // is 12 bytes, not 1.
                    iDropFrameType: 0,
                    iDropFrameSize: 0,
                }
            ),
            0
        );
        assert_eq!(setopt!(ENCODER_OPTION_STATISTICS_LOG_INTERVAL, 500i32), 0);
        assert_eq!(setopt!(ENCODER_OPTION_IS_LOSSLESS_LINK, true), 0);
        assert_eq!(setopt!(ENCODER_OPTION_BITS_VARY_PERCENTAGE, 20i32), 0);
        assert_eq!(setopt!(ENCODER_OPTION_GET_STATISTICS, 0i32), 0);
        buf.fill(0);
        assert_eq!(
            set(
                ENCODER_OPTION::ENCODER_OPTION_DUMP_FILE,
                buf.as_mut_ptr() as *mut std::ffi::c_void
            ),
            0
        );
        let mut path = *b"/tmp\0";
        assert_eq!(
            set(
                ENCODER_OPTION::ENCODER_OPTION_CURRENT_PATH,
                path.as_mut_ptr() as *mut std::ffi::c_void
            ),
            0
        );
        // The two LTR feedback filters take a struct, not a scalar.
        let mut rec = SLTRRecoverRequest::default();
        assert_eq!(
            set(
                ENCODER_OPTION::ENCODER_LTR_RECOVERY_REQUEST,
                &mut rec as *mut _ as *mut std::ffi::c_void
            ),
            0
        );
        let mut fb = SLTRMarkingFeedback::default();
        assert_eq!(
            set(
                ENCODER_OPTION::ENCODER_LTR_MARKING_FEEDBACK,
                &mut fb as *mut _ as *mut std::ffi::c_void
            ),
            0
        );

        // Out-of-range option IDs via the C vtable return CM_INIT_PARA_ERROR without UB.
        let vtbl = (*p_encoder).lpVtbl;
        let mut dummy = 0i32;
        assert_eq!(
            ((*vtbl).SetOption)(
                p_encoder,
                -1,
                &mut dummy as *mut i32 as *mut std::ffi::c_void
            ),
            CM_INIT_PARA_ERROR
        );
        assert_eq!(
            ((*vtbl).SetOption)(
                p_encoder,
                32,
                &mut dummy as *mut i32 as *mut std::ffi::c_void
            ),
            CM_INIT_PARA_ERROR
        );
        assert_eq!(
            ((*vtbl).GetOption)(
                p_encoder,
                -1,
                &mut dummy as *mut i32 as *mut std::ffi::c_void
            ),
            CM_INIT_PARA_ERROR
        );
        assert_eq!(
            ((*vtbl).GetOption)(
                p_encoder,
                32,
                &mut dummy as *mut i32 as *mut std::ffi::c_void
            ),
            CM_INIT_PARA_ERROR
        );

        // Out-of-range SBitrateInfo.iLayer via raw bytes returns CM_INIT_PARA_ERROR without UB.
        #[repr(C)]
        struct RawBitrateInfo {
            i_layer: i32,
            i_bitrate: i32,
        }
        let mut raw_bi = RawBitrateInfo {
            i_layer: 99,
            i_bitrate: 250_000,
        };
        assert_eq!(
            ((*vtbl).SetOption)(
                p_encoder,
                ENCODER_OPTION::ENCODER_OPTION_BITRATE as i32,
                &mut raw_bi as *mut RawBitrateInfo as *mut std::ffi::c_void,
            ),
            CM_INIT_PARA_ERROR
        );
        assert_eq!(
            ((*vtbl).GetOption)(
                p_encoder,
                ENCODER_OPTION::ENCODER_OPTION_BITRATE as i32,
                &mut raw_bi as *mut RawBitrateInfo as *mut std::ffi::c_void,
            ),
            CM_INIT_PARA_ERROR
        );

        // Non-0/1 bool option value (e.g. 2u8 from C) is treated as truthy without UB.
        let mut raw_bool: u8 = 2;
        assert_eq!(
            ((*vtbl).SetOption)(
                p_encoder,
                ENCODER_OPTION::ENCODER_OPTION_RC_FRAME_SKIP as i32,
                &mut raw_bool as *mut u8 as *mut std::ffi::c_void,
            ),
            CM_RESULT_SUCCESS
        );

        // ---- read back what those options wrote ------------------------------
        let mut ext = SEncParamExt::default();
        assert_eq!(
            ISVCEncoder::GetOption(
                p_encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                &mut ext as *mut SEncParamExt as *mut std::ffi::c_void,
            ),
            CM_RESULT_SUCCESS
        );
        assert_eq!(ext.iRCMode, RC_MODES::RC_BITRATE_MODE, "iRCMode");
        assert!(ext.bEnableFrameSkip, "bEnableFrameSkip");
        assert_eq!(ext.iPaddingFlag, 1, "iPaddingFlag");
        assert_eq!(ext.iLtrMarkPeriod, 30, "iLtrMarkPeriod");
        assert!(ext.bEnableLongTermReference, "bEnableLongTermReference");
        assert_eq!(ext.iLTRRefNum, 2, "iLTRRefNum");
        assert!(ext.bEnableSSEI, "bEnableSSEI");
        assert!(ext.bPrefixNalAddingCtrl, "bPrefixNalAddingCtrl");
        assert_eq!(
            ext.eSpsPpsIdStrategy,
            EParameterSetStrategy::CONSTANT_ID,
            "eSpsPpsIdStrategy after the out-of-range set"
        );
        assert!(ext.bIsLosslessLink, "bIsLosslessLink");
        assert_eq!(ext.iNumRefFrame, 3, "iNumRefFrame");
        assert_eq!(
            ext.sSpatialLayers[0].uiProfileIdc,
            EProfileIdc::PRO_HIGH,
            "uiProfileIdc"
        );
        assert_eq!(
            ext.sSpatialLayers[0].uiLevelIdc,
            ELevelIdc::LEVEL_3_0,
            "uiLevelIdc"
        );

        let mut interval = -1i32;
        assert_eq!(
            ISVCEncoder::GetOption(
                p_encoder,
                ENCODER_OPTION::ENCODER_OPTION_STATISTICS_LOG_INTERVAL,
                &mut interval as *mut i32 as *mut std::ffi::c_void,
            ),
            CM_RESULT_SUCCESS
        );
        assert_eq!(interval, 500, "iStatisticsLogInterval");

        assert_eq!(ISVCEncoder::Uninitialize(p_encoder), CM_RESULT_SUCCESS);
        WelsDestroySVCEncoder(p_encoder);
    }
}

// ============================================================================
// Shared helpers for `EncodeDecodeTestAPI` ports (`test/api/encode_options_test.cpp`)
// ============================================================================

fn valid_size(size: i32) -> i32 {
    if size > 16 { size } else { 16 }
}

unsafe fn prepare_param_default(
    encoder: *mut ISVCEncoder,
    layers: i32,
    slices: u32,
    width: i32,
    height: i32,
    framerate: f32,
    param: &mut SEncParamExt,
) {
    unsafe {
        *param = SEncParamExt::default();
        assert_eq!(
            ISVCEncoder::GetDefaultParams(encoder, param as *mut SEncParamExt),
            CM_RESULT_SUCCESS
        );
        let w = valid_size(width);
        let h = valid_size(height);
        param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
        param.iPicWidth = w;
        param.iPicHeight = h;
        param.fMaxFrameRate = framerate;
        param.iRCMode = RC_MODES::RC_OFF_MODE;
        param.iMultipleThreadIdc = 1;
        param.iSpatialLayerNum = layers;
        param.iNumRefFrame = AUTO_REF_PIC_COUNT;
        for i in 0..(layers as usize) {
            let shift = (layers as usize) - i - 1;
            param.sSpatialLayers[i].iVideoWidth = w >> shift;
            param.sSpatialLayers[i].iVideoHeight = h >> shift;
            param.sSpatialLayers[i].fFrameRate = framerate;
            param.sSpatialLayers[i].sSliceArgument.uiSliceMode =
                SliceModeEnum::SM_FIXEDSLCNUM_SLICE;
            param.sSpatialLayers[i].sSliceArgument.uiSliceNum = slices;
        }
    }
}

unsafe fn collect_frame_bitstream(info: &SFrameBSInfo) -> Vec<u8> {
    let mut out = Vec::new();
    for i in 0..(info.iLayerNum as usize) {
        let layer = &info.sLayerInfo[i];
        if layer.iNalCount <= 0 || layer.pNalLengthInByte.is_null() || layer.pBsBuf.is_null() {
            continue;
        }
        let nal_lens =
            unsafe { std::slice::from_raw_parts(layer.pNalLengthInByte, layer.iNalCount as usize) };
        let total: usize = nal_lens.iter().map(|&n| n.max(0) as usize).sum();
        let bytes = unsafe { std::slice::from_raw_parts(layer.pBsBuf, total) };
        out.extend_from_slice(bytes);
    }
    out
}

unsafe fn create_and_init_decoder() -> *mut ISVCDecoder {
    unsafe {
        let mut decoder: *mut ISVCDecoder = ptr::null_mut();
        assert_eq!(WelsCreateDecoder(&mut decoder), 0);
        let dec_param = SDecodingParam {
            uiTargetDqLayer: u8::MAX,
            eEcActiveIdc: ERROR_CON_IDC::ERROR_CON_SLICE_COPY,
            sVideoProperty: SVideoProperty {
                size: size_of::<SVideoProperty>() as u32,
                eVideoBsType: VIDEO_BITSTREAM_DEFAULT,
            },
            ..SDecodingParam::default()
        };
        assert_eq!(ISVCDecoder::Initialize(decoder, &dec_param), 0);
        decoder
    }
}

unsafe fn enc_dec_one_frame(
    encoder: *mut ISVCEncoder,
    decoder: *mut ISVCDecoder,
    width: i32,
    height: i32,
    frame_idx: usize,
) -> SFrameBSInfo {
    unsafe {
        let w = width as usize;
        let h = height as usize;
        let mut buf = vec![0u8; w * h * 3 / 2];
        fill_camera_frame(&mut buf, w, h, frame_idx);

        let mut pic = SSourcePicture::default();
        pic.iPicWidth = width;
        pic.iPicHeight = height;
        pic.iColorFormat = EVideoFormatType::videoFormatI420 as i32;
        pic.iStride[0] = width;
        pic.iStride[1] = width >> 1;
        pic.iStride[2] = width >> 1;
        pic.pData[0] = buf.as_mut_ptr();
        pic.pData[1] = buf.as_mut_ptr().add(w * h);
        pic.pData[2] = buf.as_mut_ptr().add(w * h + (w * h >> 2));

        let mut info = SFrameBSInfo::default();
        let rv = ISVCEncoder::EncodeFrame(encoder, &pic, &mut info);
        assert_eq!(
            rv, CM_RESULT_SUCCESS,
            "EncodeFrame failed at frame {frame_idx}"
        );

        let bs = collect_frame_bitstream(&info);
        let mut p_data: [*mut u8; 3] = [ptr::null_mut(); 3];
        let mut dst_buf_info = SBufferInfo::default();
        let dec_rv = ISVCDecoder::DecodeFrameNoDelay(
            decoder,
            bs.as_ptr(),
            bs.len() as i32,
            p_data.as_mut_ptr(),
            &mut dst_buf_info,
        );
        assert_eq!(
            dec_rv,
            DECODING_STATE::dsErrorFree,
            "DecodeFrameNoDelay failed at frame {frame_idx}: {dec_rv:?}"
        );
        info
    }
}

/// Port of `EncodeDecodeTestAPIBase::TestOneSimulcastAVC`
/// (`test/api/encode_decode_api_test.cpp:124-201`).
unsafe fn test_one_simulcast_avc(
    encoder: *mut ISVCEncoder,
    param: &mut SEncParamExt,
    decoders: &[*mut ISVCDecoder],
    spatial_layer_num: usize,
    enc_frame_num: usize,
) {
    unsafe {
        let rv = ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
            (param as *mut SEncParamExt).cast::<c_void>(),
        );
        assert_eq!(rv, CM_RESULT_SUCCESS, "SetOption failed: rv={rv}");

        let w = param.iPicWidth as usize;
        let h = param.iPicHeight as usize;
        let mut buf = vec![0u8; w * h * 3 / 2];

        for frame_idx in 0..enc_frame_num {
            fill_camera_frame(&mut buf, w, h, frame_idx);
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

            let mut layer_bufs = vec![Vec::<u8>::new(); spatial_layer_num];
            for i_layer in 0..(info.iLayerNum as usize) {
                let layer_info = &info.sLayerInfo[i_layer];
                let nal_lens = std::slice::from_raw_parts(
                    layer_info.pNalLengthInByte,
                    layer_info.iNalCount as usize,
                );
                let layer_len: usize = nal_lens.iter().map(|&n| n.max(0) as usize).sum();
                let idx = layer_info.uiSpatialId as usize;
                assert!(idx < spatial_layer_num);
                let bytes = std::slice::from_raw_parts(layer_info.pBsBuf, layer_len);
                layer_bufs[idx].extend_from_slice(bytes);
            }

            for (idx, bs) in layer_bufs.iter().enumerate() {
                let mut p_data: [*mut u8; 3] = [ptr::null_mut(); 3];
                let mut dst_buf_info = SBufferInfo::default();
                let hr1 = ISVCDecoder::DecodeFrame2(
                    decoders[idx],
                    bs.as_ptr(),
                    bs.len() as i32,
                    p_data.as_mut_ptr(),
                    &mut dst_buf_info,
                );
                assert_eq!(
                    hr1,
                    DECODING_STATE::dsErrorFree,
                    "layer={idx} frame={frame_idx}"
                );
                let hr2 = ISVCDecoder::DecodeFrame2(
                    decoders[idx],
                    ptr::null(),
                    0,
                    p_data.as_mut_ptr(),
                    &mut dst_buf_info,
                );
                assert_eq!(
                    hr2,
                    DECODING_STATE::dsErrorFree,
                    "layer={idx} frame={frame_idx}"
                );
                assert_eq!(
                    dst_buf_info.iBufferStatus, 1,
                    "layer={idx} frame={frame_idx}"
                );
            }
        }
    }
}

/// Port of `EncodeDecodeTestAPI.SetOptionEncParamBase` (`test/api/encode_options_test.cpp:349-391`).
#[test]
fn set_option_enc_param_base_roundtrip() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);
        let decoder = create_and_init_decoder();

        let mut param = SEncParamExt::default();
        prepare_param_default(encoder, 4, 1, 320, 192, 15.0, &mut param);
        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param),
            CM_RESULT_SUCCESS
        );

        let mut trace_level: i32 = 0;
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_OPTION_TRACE_LEVEL,
            ptr::addr_of_mut!(trace_level).cast(),
        );

        let w = param.iPicWidth as usize;
        let h = param.iPicHeight as usize;
        let mut buf = vec![0u8; w * h * 3 / 2];

        for i in 0..8usize {
            let mut base = SEncParamBase {
                iUsageType: param.iUsageType,
                iPicWidth: param.iPicWidth,
                iPicHeight: param.iPicHeight,
                iTargetBitrate: ((i as i32 + 1) * 150_000),
                iRCMode: param.iRCMode,
                fMaxFrameRate: param.fMaxFrameRate,
            };
            let res = ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_BASE,
                ptr::addr_of_mut!(base).cast(),
            );
            if res == CM_RESULT_SUCCESS {
                fill_camera_frame(&mut buf, w, h, i);
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
                let bs = collect_frame_bitstream(&info);
                let mut p_data: [*mut u8; 3] = [ptr::null_mut(); 3];
                let mut dst_buf_info = SBufferInfo::default();
                assert_eq!(
                    ISVCDecoder::DecodeFrame2(
                        decoder,
                        bs.as_ptr(),
                        bs.len() as i32,
                        p_data.as_mut_ptr(),
                        &mut dst_buf_info,
                    ),
                    DECODING_STATE::dsErrorFree
                );
                assert_eq!(
                    ISVCDecoder::DecodeFrame2(
                        decoder,
                        ptr::null(),
                        0,
                        p_data.as_mut_ptr(),
                        &mut dst_buf_info,
                    ),
                    DECODING_STATE::dsErrorFree
                );
                assert_eq!(dst_buf_info.iBufferStatus, 1);
            }
        }

        ISVCDecoder::Uninitialize(decoder);
        WelsDestroyDecoder(decoder);
        ISVCEncoder::Uninitialize(encoder);
        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncodeDecodeTestAPI.ParameterSetStrategy_INCREASING_ID`
/// (`test/api/encode_options_test.cpp:401-511`) and
/// `ParameterSetStrategy_SPS_LISTING_AND_PPS_INCREASING1` (`lines 514-621`).
fn run_parameter_set_strategy_three_params(strategy: EParameterSetStrategy, simulcast_avc: bool) {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);
        let decoder = create_and_init_decoder();

        let mut param1 = SEncParamExt::default();
        prepare_param_default(encoder, 1, 1, 320, 192, 15.0, &mut param1);
        param1.bSimulcastAVC = simulcast_avc;
        param1.eSpsPpsIdStrategy = strategy;

        let mut param2 = param1;
        prepare_param_default(encoder, 1, 1, 160, 192, 15.0, &mut param2);
        param2.bSimulcastAVC = simulcast_avc;
        param2.eSpsPpsIdStrategy = strategy;

        let mut param3 = param1;
        prepare_param_default(encoder, 1, 1, 320, 96, 15.0, &mut param3);
        param3.bSimulcastAVC = simulcast_avc;
        param3.eSpsPpsIdStrategy = strategy;

        // Part 1
        let mut frame_num = 0usize;
        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param1),
            CM_RESULT_SUCCESS
        );
        enc_dec_one_frame(
            encoder,
            decoder,
            param1.iPicWidth,
            param1.iPicHeight,
            frame_num,
        );
        frame_num += 1;

        assert_eq!(
            ISVCEncoder::ForceIntraFrame(encoder, true),
            CM_RESULT_SUCCESS
        );
        enc_dec_one_frame(
            encoder,
            decoder,
            param1.iPicWidth,
            param1.iPicHeight,
            frame_num,
        );
        frame_num += 1;

        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                ptr::addr_of_mut!(param2).cast(),
            ),
            CM_RESULT_SUCCESS
        );
        enc_dec_one_frame(
            encoder,
            decoder,
            param2.iPicWidth,
            param2.iPicHeight,
            frame_num,
        );
        frame_num += 1;

        assert_eq!(
            ISVCEncoder::ForceIntraFrame(encoder, true),
            CM_RESULT_SUCCESS
        );
        enc_dec_one_frame(
            encoder,
            decoder,
            param2.iPicWidth,
            param2.iPicHeight,
            frame_num,
        );
        frame_num += 1;

        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                ptr::addr_of_mut!(param1).cast(),
            ),
            CM_RESULT_SUCCESS
        );
        enc_dec_one_frame(
            encoder,
            decoder,
            param1.iPicWidth,
            param1.iPicHeight,
            frame_num,
        );
        frame_num += 1;

        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                ptr::addr_of_mut!(param2).cast(),
            ),
            CM_RESULT_SUCCESS
        );
        enc_dec_one_frame(
            encoder,
            decoder,
            param2.iPicWidth,
            param2.iPicHeight,
            frame_num,
        );

        assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);

        // Part 2
        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param1),
            CM_RESULT_SUCCESS
        );
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                ptr::addr_of_mut!(param2).cast(),
            ),
            CM_RESULT_SUCCESS
        );
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                ptr::addr_of_mut!(param3).cast(),
            ),
            CM_RESULT_SUCCESS
        );

        enc_dec_one_frame(encoder, decoder, param3.iPicWidth, param3.iPicHeight, 0);
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                ptr::addr_of_mut!(param2).cast(),
            ),
            CM_RESULT_SUCCESS
        );
        enc_dec_one_frame(encoder, decoder, param2.iPicWidth, param2.iPicHeight, 1);
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                ptr::addr_of_mut!(param1).cast(),
            ),
            CM_RESULT_SUCCESS
        );
        enc_dec_one_frame(encoder, decoder, param1.iPicWidth, param1.iPicHeight, 2);

        assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
        ISVCDecoder::Uninitialize(decoder);
        WelsDestroyDecoder(decoder);
        WelsDestroySVCEncoder(encoder);
    }
}

#[test]
fn parameter_set_strategy_increasing_id() {
    run_parameter_set_strategy_three_params(EParameterSetStrategy::INCREASING_ID, true);
}

#[test]
fn parameter_set_strategy_sps_listing_and_pps_increasing_1() {
    run_parameter_set_strategy_three_params(
        EParameterSetStrategy::SPS_LISTING_AND_PPS_INCREASING,
        false,
    );
}

/// Port of `EncodeDecodeTestAPI.ParameterSetStrategy_SPS_LISTING_AND_PPS_INCREASING2`
/// (`test/api/encode_options_test.cpp:624-679`).
#[test]
fn parameter_set_strategy_sps_listing_and_pps_increasing_2() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);
        let decoder = create_and_init_decoder();

        let mut param1 = SEncParamExt::default();
        prepare_param_default(encoder, 1, 1, 320, 192, 15.0, &mut param1);
        param1.eSpsPpsIdStrategy = EParameterSetStrategy::SPS_LISTING_AND_PPS_INCREASING;
        param1.iTemporalLayerNum = 1;

        let mut param2 = param1;
        param2.iTemporalLayerNum = 3;

        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param1),
            CM_RESULT_SUCCESS
        );
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                ptr::addr_of_mut!(param2).cast(),
            ),
            CM_RESULT_SUCCESS
        );
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                ptr::addr_of_mut!(param1).cast(),
            ),
            CM_RESULT_SUCCESS
        );
        enc_dec_one_frame(encoder, decoder, param1.iPicWidth, param1.iPicHeight, 0);

        assert_eq!(
            ISVCEncoder::ForceIntraFrame(encoder, true),
            CM_RESULT_SUCCESS
        );
        enc_dec_one_frame(encoder, decoder, param1.iPicWidth, param1.iPicHeight, 1);

        assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
        ISVCDecoder::Uninitialize(decoder);
        WelsDestroyDecoder(decoder);
        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncodeDecodeTestAPI.ParameterSetStrategy_SPS_LISTING_AND_PPS_INCREASING3`
/// (`test/api/encode_options_test.cpp:681-735`).
#[test]
fn parameter_set_strategy_sps_listing_and_pps_increasing_3() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);
        let decoder = create_and_init_decoder();

        let mut param1 = SEncParamExt::default();
        prepare_param_default(encoder, 1, 1, 16, 128, 15.0, &mut param1);
        param1.eSpsPpsIdStrategy = EParameterSetStrategy::SPS_LISTING_AND_PPS_INCREASING;
        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param1),
            CM_RESULT_SUCCESS
        );

        let mut param2 = param1;
        // 65 distinct macroblock widths (> 2 * MAX_SPS_COUNT)
        for times in 0..65i32 {
            let width = (times + 2) * 16;
            prepare_param_default(encoder, 1, 1, width, 128, 15.0, &mut param2);
            param2.eSpsPpsIdStrategy = EParameterSetStrategy::SPS_LISTING_AND_PPS_INCREASING;
            assert_eq!(
                ISVCEncoder::SetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                    ptr::addr_of_mut!(param2).cast(),
                ),
                CM_RESULT_SUCCESS
            );
        }

        enc_dec_one_frame(encoder, decoder, param2.iPicWidth, param2.iPicHeight, 0);

        assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
        ISVCDecoder::Uninitialize(decoder);
        WelsDestroyDecoder(decoder);
        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncodeDecodeTestAPI.ParameterSetStrategy_SPS_PPS_LISTING1/2/3`
/// (`test/api/encode_options_test.cpp:738-943`).
#[test]
fn parameter_set_strategy_sps_pps_listing_1_2_3() {
    unsafe {
        // Listing 1: 1 resolution, encode IDR0, P1, ForceIntraFrame IDR2
        {
            let mut encoder: *mut ISVCEncoder = ptr::null_mut();
            assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);
            let decoder = create_and_init_decoder();

            let mut param1 = SEncParamExt::default();
            prepare_param_default(encoder, 1, 1, 320, 192, 15.0, &mut param1);
            param1.eSpsPpsIdStrategy = EParameterSetStrategy::SPS_PPS_LISTING;
            assert_eq!(
                ISVCEncoder::InitializeExt(encoder, &param1),
                CM_RESULT_SUCCESS
            );
            enc_dec_one_frame(encoder, decoder, param1.iPicWidth, param1.iPicHeight, 0);
            enc_dec_one_frame(encoder, decoder, param1.iPicWidth, param1.iPicHeight, 1);
            assert_eq!(
                ISVCEncoder::ForceIntraFrame(encoder, true),
                CM_RESULT_SUCCESS
            );
            enc_dec_one_frame(encoder, decoder, param1.iPicWidth, param1.iPicHeight, 2);

            assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
            ISVCDecoder::Uninitialize(decoder);
            WelsDestroyDecoder(decoder);
            WelsDestroySVCEncoder(encoder);
        }

        // Listing 2: 2 resolutions registered before encoding, switch between them
        {
            let mut encoder: *mut ISVCEncoder = ptr::null_mut();
            assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);
            let decoder = create_and_init_decoder();

            let mut param1 = SEncParamExt::default();
            prepare_param_default(encoder, 1, 1, 320, 192, 15.0, &mut param1);
            param1.eSpsPpsIdStrategy = EParameterSetStrategy::SPS_PPS_LISTING;

            let mut param2 = param1;
            prepare_param_default(encoder, 1, 1, 160, 192, 15.0, &mut param2);
            param2.eSpsPpsIdStrategy = EParameterSetStrategy::SPS_PPS_LISTING;

            assert_eq!(
                ISVCEncoder::InitializeExt(encoder, &param1),
                CM_RESULT_SUCCESS
            );
            assert_eq!(
                ISVCEncoder::SetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                    ptr::addr_of_mut!(param2).cast(),
                ),
                CM_RESULT_SUCCESS
            );
            enc_dec_one_frame(encoder, decoder, param2.iPicWidth, param2.iPicHeight, 0);

            assert_eq!(
                ISVCEncoder::SetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                    ptr::addr_of_mut!(param1).cast(),
                ),
                CM_RESULT_SUCCESS
            );
            enc_dec_one_frame(encoder, decoder, param1.iPicWidth, param1.iPicHeight, 1);

            assert_eq!(
                ISVCEncoder::ForceIntraFrame(encoder, true),
                CM_RESULT_SUCCESS
            );
            enc_dec_one_frame(encoder, decoder, param1.iPicWidth, param1.iPicHeight, 2);

            assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
            ISVCDecoder::Uninitialize(decoder);
            WelsDestroyDecoder(decoder);
            WelsDestroySVCEncoder(encoder);
        }

        // Listing 3: incompatible strategy change and unseen resolution after encoding fail
        {
            let mut encoder: *mut ISVCEncoder = ptr::null_mut();
            assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);
            let decoder = create_and_init_decoder();

            let mut param1 = SEncParamExt::default();
            prepare_param_default(encoder, 1, 1, 320, 192, 15.0, &mut param1);
            param1.eSpsPpsIdStrategy = EParameterSetStrategy::SPS_PPS_LISTING;

            let mut param2 = param1;
            prepare_param_default(encoder, 1, 1, 160, 192, 15.0, &mut param2);
            param2.eSpsPpsIdStrategy = EParameterSetStrategy::SPS_PPS_LISTING;

            let mut param3 = param1;
            prepare_param_default(encoder, 1, 1, 480, 192, 15.0, &mut param3);
            param3.eSpsPpsIdStrategy = EParameterSetStrategy::SPS_PPS_LISTING;

            assert_eq!(
                ISVCEncoder::InitializeExt(encoder, &param1),
                CM_RESULT_SUCCESS
            );
            assert_eq!(
                ISVCEncoder::SetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                    ptr::addr_of_mut!(param2).cast(),
                ),
                CM_RESULT_SUCCESS
            );
            enc_dec_one_frame(encoder, decoder, param2.iPicWidth, param2.iPicHeight, 0);

            let mut new_stra = EParameterSetStrategy::SPS_PPS_LISTING as i32;
            assert_eq!(
                ISVCEncoder::SetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_OPTION_SPS_PPS_ID_STRATEGY,
                    ptr::addr_of_mut!(new_stra).cast(),
                ),
                CM_RESULT_SUCCESS
            );

            let mut trace_level: i32 = 0;
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_TRACE_LEVEL,
                ptr::addr_of_mut!(trace_level).cast(),
            );
            new_stra = EParameterSetStrategy::CONSTANT_ID as i32;
            assert_ne!(
                ISVCEncoder::SetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_OPTION_SPS_PPS_ID_STRATEGY,
                    ptr::addr_of_mut!(new_stra).cast(),
                ),
                CM_RESULT_SUCCESS
            );

            enc_dec_one_frame(encoder, decoder, param2.iPicWidth, param2.iPicHeight, 1);

            // Switching to unseen resolution param3 after encoding started must fail
            assert_ne!(
                ISVCEncoder::SetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                    ptr::addr_of_mut!(param3).cast(),
                ),
                CM_RESULT_SUCCESS
            );

            assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
            ISVCDecoder::Uninitialize(decoder);
            WelsDestroyDecoder(decoder);
            WelsDestroySVCEncoder(encoder);
        }
    }
}

/// Port of `EncodeDecodeTestAPI.SimulcastSVC` (`test/api/encode_options_test.cpp:946-1063`).
#[test]
fn simulcast_svc_roundtrip() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let spatial_layer_num = 3usize;
        let width = 320i32;
        let height = 192i32;
        let mut param = SEncParamExt::default();
        prepare_param_default(
            encoder,
            spatial_layer_num as i32,
            1,
            width,
            height,
            15.0,
            &mut param,
        );
        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param),
            CM_RESULT_SUCCESS
        );

        let decoders: Vec<*mut ISVCDecoder> = (0..spatial_layer_num)
            .map(|_| create_and_init_decoder())
            .collect();

        let w = width as usize;
        let h = height as usize;
        let mut buf = vec![0u8; w * h * 3 / 2];

        for frame_idx in 0..10usize {
            fill_camera_frame(&mut buf, w, h, frame_idx);
            let mut pic = SSourcePicture::default();
            pic.iPicWidth = width;
            pic.iPicHeight = height;
            pic.iColorFormat = EVideoFormatType::videoFormatI420 as i32;
            pic.iStride[0] = width;
            pic.iStride[1] = width >> 1;
            pic.iStride[2] = width >> 1;
            pic.pData[0] = buf.as_mut_ptr();
            pic.pData[1] = buf.as_mut_ptr().add(w * h);
            pic.pData[2] = buf.as_mut_ptr().add(w * h + (w * h >> 2));

            let mut info = SFrameBSInfo::default();
            assert_eq!(
                ISVCEncoder::EncodeFrame(encoder, &pic, &mut info),
                CM_RESULT_SUCCESS
            );

            let mut layer_bufs = vec![Vec::<u8>::new(); spatial_layer_num];
            for i_layer in 0..(info.iLayerNum as usize) {
                let layer_info = &info.sLayerInfo[i_layer];
                let nal_lens = std::slice::from_raw_parts(
                    layer_info.pNalLengthInByte,
                    layer_info.iNalCount as usize,
                );
                let layer_len: usize = nal_lens.iter().map(|&n| n.max(0) as usize).sum();
                let bytes = std::slice::from_raw_parts(layer_info.pBsBuf, layer_len);

                if layer_info.uiLayerType == LAYER_TYPE::NON_VIDEO_CODING_LAYER as u8 {
                    for lb in &mut layer_bufs {
                        lb.extend_from_slice(bytes);
                    }
                } else {
                    let idx = layer_info.uiSpatialId as usize;
                    assert!(idx < spatial_layer_num);
                    layer_bufs[idx].extend_from_slice(bytes);
                }
            }

            for (idx, bs) in layer_bufs.iter().enumerate() {
                let mut p_data: [*mut u8; 3] = [ptr::null_mut(); 3];
                let mut dst_buf_info = SBufferInfo::default();
                let hr1 = ISVCDecoder::DecodeFrame2(
                    decoders[idx],
                    bs.as_ptr(),
                    bs.len() as i32,
                    p_data.as_mut_ptr(),
                    &mut dst_buf_info,
                );
                assert_eq!(hr1, DECODING_STATE::dsErrorFree);
                let hr2 = ISVCDecoder::DecodeFrame2(
                    decoders[idx],
                    ptr::null(),
                    0,
                    p_data.as_mut_ptr(),
                    &mut dst_buf_info,
                );
                assert_eq!(hr2, DECODING_STATE::dsErrorFree);
                if dst_buf_info.iBufferStatus == 0 {
                    let mut remaining = 0i32;
                    ISVCDecoder::GetOption(
                        decoders[idx],
                        DECODER_OPTION::DECODER_OPTION_NUM_OF_FRAMES_REMAINING_IN_BUFFER,
                        ptr::addr_of_mut!(remaining).cast(),
                    );
                    for _ in 0..remaining {
                        ISVCDecoder::FlushFrame(
                            decoders[idx],
                            p_data.as_mut_ptr(),
                            &mut dst_buf_info,
                        );
                    }
                }
                assert_eq!(dst_buf_info.iBufferStatus, 1, "layer={idx}");
            }
        }

        for dec in decoders {
            ISVCDecoder::Uninitialize(dec);
            WelsDestroyDecoder(dec);
        }
        ISVCEncoder::Uninitialize(encoder);
        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncodeDecodeTestAPI.SimulcastAVC` and `SimulcastAVC_SPS_PPS_LISTING`
/// (`test/api/encode_options_test.cpp:1065-1241`).
#[test]
fn simulcast_avc_and_sps_pps_listing_roundtrip() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let spatial_layer_num = 3usize;
        let width = 320i32;
        let height = 192i32;

        let mut param1 = SEncParamExt::default();
        prepare_param_default(
            encoder,
            spatial_layer_num as i32,
            1,
            width,
            height,
            15.0,
            &mut param1,
        );
        param1.eSpsPpsIdStrategy = EParameterSetStrategy::SPS_PPS_LISTING;
        param1.bSimulcastAVC = true;

        let mut param2 = param1;
        param2.sSpatialLayers[0].iVideoWidth = param1.sSpatialLayers[0].iVideoWidth / 2;
        param2.sSpatialLayers[0].iVideoHeight = param1.sSpatialLayers[0].iVideoHeight / 2;

        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param1),
            CM_RESULT_SUCCESS
        );
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                ptr::addr_of_mut!(param2).cast(),
            ),
            CM_RESULT_SUCCESS
        );

        let decoders: Vec<*mut ISVCDecoder> = (0..spatial_layer_num)
            .map(|_| create_and_init_decoder())
            .collect();

        test_one_simulcast_avc(encoder, &mut param1, &decoders, spatial_layer_num, 6);
        test_one_simulcast_avc(encoder, &mut param2, &decoders, spatial_layer_num, 6);

        for dec in decoders {
            ISVCDecoder::Uninitialize(dec);
            WelsDestroyDecoder(dec);
        }
        ISVCEncoder::Uninitialize(encoder);
        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncodeTestAPI.SetEncOptionSize` (`test/api/encode_options_test.cpp:1243-1433`).
fn run_set_enc_option_size(
    test_nal_size: bool,
    all_random: bool,
    test_decoder: bool,
    num_frames: usize,
    width: i32,
    height: i32,
    slice_mode: SliceModeEnum,
    max_nal_len: u32,
    framerate: f32,
    threads: u16,
) {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);
        let decoder = create_and_init_decoder();

        let mut param = SEncParamExt::default();
        ISVCEncoder::GetDefaultParams(encoder, &mut param);
        param.uiMaxNalSize = max_nal_len;
        param.iTemporalLayerNum = 2;
        param.iSpatialLayerNum = 1;
        param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
        param.iPicWidth = width;
        param.iPicHeight = height;
        param.fMaxFrameRate = framerate;
        param.iRCMode = RC_MODES::RC_OFF_MODE;
        param.iMultipleThreadIdc = threads;
        param.iNumRefFrame = AUTO_REF_PIC_COUNT;
        param.sSpatialLayers[0].iVideoWidth = width;
        param.sSpatialLayers[0].iVideoHeight = height;
        param.sSpatialLayers[0].fFrameRate = framerate;

        let slice_modes: &[SliceModeEnum] = if slice_mode == SliceModeEnum::SM_RESERVED {
            &[
                SliceModeEnum::SM_SINGLE_SLICE,
                SliceModeEnum::SM_FIXEDSLCNUM_SLICE,
                SliceModeEnum::SM_RASTER_SLICE,
                SliceModeEnum::SM_SIZELIMITED_SLICE,
            ]
        } else {
            std::slice::from_ref(&slice_mode)
        };

        for &sm in slice_modes {
            param.sSpatialLayers[0].sSliceArgument.uiSliceMode = sm;
            if sm == SliceModeEnum::SM_FIXEDSLCNUM_SLICE {
                param.sSpatialLayers[0].sSliceArgument.uiSliceNum = 8;
            } else if sm == SliceModeEnum::SM_RASTER_SLICE {
                let mb_slice = ((width * height) >> 10) as u32;
                for slot in &mut param.sSpatialLayers[0].sSliceArgument.uiSliceMbNum[..4] {
                    *slot = mb_slice;
                }
            } else if sm == SliceModeEnum::SM_SIZELIMITED_SLICE && max_nal_len < 450 {
                param.uiMaxNalSize = 450;
                param.sSpatialLayers[0].sSliceArgument.uiSliceSizeConstraint = 450;
            } else if sm == SliceModeEnum::SM_SIZELIMITED_SLICE {
                param.sSpatialLayers[0].sSliceArgument.uiSliceSizeConstraint = max_nal_len;
            }

            let mut trace_level: i32 = 0;
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_TRACE_LEVEL,
                ptr::addr_of_mut!(trace_level).cast(),
            );
            ISVCDecoder::SetOption(
                decoder,
                DECODER_OPTION::DECODER_OPTION_TRACE_LEVEL,
                ptr::addr_of_mut!(trace_level).cast(),
            );

            ISVCEncoder::Uninitialize(encoder);
            assert_eq!(
                ISVCEncoder::InitializeExt(encoder, &param),
                CM_RESULT_SUCCESS
            );

            let mut sps_pps_strategy: i32 = 1;
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_SPS_PPS_ID_STRATEGY,
                ptr::addr_of_mut!(sps_pps_strategy).cast(),
            );
            let mut idr_period: i32 = 6;
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_IDR_INTERVAL,
                ptr::addr_of_mut!(idr_period).cast(),
            );

            let effective_random = if threads > 1 && sm == SliceModeEnum::SM_SIZELIMITED_SLICE {
                false
            } else {
                all_random
            };

            let w = width as usize;
            let h = height as usize;
            let mut buf = vec![0u8; w * h * 3 / 2];
            let mut rng = 0x9988_7766u32;

            for idx in 0..=num_frames {
                if effective_random {
                    fill_seeded_noise(&mut buf, &mut rng);
                } else {
                    fill_camera_frame(&mut buf, w, h, idx);
                }

                let mut pic = SSourcePicture::default();
                pic.iPicWidth = width;
                pic.iPicHeight = height;
                pic.iColorFormat = EVideoFormatType::videoFormatI420 as i32;
                pic.iStride[0] = width;
                pic.iStride[1] = width >> 1;
                pic.iStride[2] = width >> 1;
                pic.pData[0] = buf.as_mut_ptr();
                pic.pData[1] = buf.as_mut_ptr().add(w * h);
                pic.pData[2] = buf.as_mut_ptr().add(w * h + (w * h >> 2));

                let mut info = SFrameBSInfo::default();
                assert_eq!(
                    ISVCEncoder::EncodeFrame(encoder, &pic, &mut info),
                    CM_RESULT_SUCCESS
                );

                let bs = collect_frame_bitstream(&info);
                if !bs.is_empty() && test_decoder {
                    let mut p_data: [*mut u8; 3] = [ptr::null_mut(); 3];
                    let mut dst_buf_info = SBufferInfo::default();
                    let dec_rv = ISVCDecoder::DecodeFrameNoDelay(
                        decoder,
                        bs.as_ptr(),
                        bs.len() as i32,
                        p_data.as_mut_ptr(),
                        &mut dst_buf_info,
                    );
                    assert_eq!(dec_rv, DECODING_STATE::dsErrorFree);
                    assert_eq!(dst_buf_info.iBufferStatus, 1);
                }

                if sm == SliceModeEnum::SM_SIZELIMITED_SLICE && test_nal_size {
                    for i_layer in 0..(info.iLayerNum as usize) {
                        let layer = &info.sLayerInfo[i_layer];
                        let max_idx = (layer.iNalCount - 2).max(0) as usize;
                        for nal_idx in 0..=max_idx {
                            if nal_idx < layer.iNalCount as usize {
                                let nal_len = *layer.pNalLengthInByte.add(nal_idx);
                                assert!(
                                    nal_len <= param.uiMaxNalSize as i32,
                                    "nal_len={nal_len} > uiMaxNalSize={}",
                                    param.uiMaxNalSize
                                );
                            }
                        }
                    }
                }
            }
        }

        ISVCDecoder::Uninitialize(decoder);
        WelsDestroyDecoder(decoder);
        ISVCEncoder::Uninitialize(encoder);
        WelsDestroySVCEncoder(encoder);
    }
}

#[test]
fn set_enc_option_size_sizelimited_decode() {
    run_set_enc_option_size(
        true,
        false,
        true,
        10,
        600,
        460,
        SliceModeEnum::SM_SIZELIMITED_SLICE,
        450,
        15.0,
        1,
    );
    run_set_enc_option_size(
        true,
        true,
        true,
        10,
        340,
        96,
        SliceModeEnum::SM_SIZELIMITED_SLICE,
        1000,
        30.0,
        1,
    );
    run_set_enc_option_size(
        true,
        true,
        true,
        10,
        16,
        16,
        SliceModeEnum::SM_SIZELIMITED_SLICE,
        500,
        7.5,
        1,
    );
}

#[test]
fn set_enc_option_size_multithread_fixed_slice() {
    run_set_enc_option_size(
        false,
        false,
        true,
        10,
        600,
        460,
        SliceModeEnum::SM_FIXEDSLCNUM_SLICE,
        0,
        15.0,
        4,
    );
    run_set_enc_option_size(
        false,
        false,
        true,
        10,
        600,
        460,
        SliceModeEnum::SM_FIXEDSLCNUM_SLICE,
        0,
        15.0,
        8,
    );
}

#[test]
fn set_enc_option_size_annex_a_extreme_dimensions() {
    // Annex A: PicWidthInMbs <= sqrt(36864*8) = 543; 543*16=8688; 36864/543=67; 67*16=1072
    run_set_enc_option_size(
        false,
        false,
        true,
        2,
        8688,
        1072,
        SliceModeEnum::SM_SINGLE_SLICE,
        0,
        15.0,
        1,
    );
    run_set_enc_option_size(
        false,
        false,
        true,
        2,
        1072,
        8688,
        SliceModeEnum::SM_SINGLE_SLICE,
        0,
        15.0,
        1,
    );
}

/// Port of `EncodeDecodeTestAPI.SimulcastAVCDiffFps`
/// (`test/api/encode_options_test.cpp:1437-1569`).
#[test]
fn simulcast_avc_diff_fps() {
    const TEMPORAL_PATTERNS: [[i32; 4]; 6] = [
        [2, 1, 1, 1],
        [4, 2, 1, 1],
        [1, 4, 2, 1],
        [1, 2, 4, 1],
        [1, 2, 2, 4],
        [1, 4, 4, 4],
    ];

    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let spatial_layer_num = 3usize;
        let width = 320i32;
        let height = 192i32;
        let frame_rate = 30.0f32;

        let mut param = SEncParamExt::default();
        prepare_param_default(
            encoder,
            spatial_layer_num as i32,
            1,
            width,
            height,
            frame_rate,
            &mut param,
        );
        param.bSimulcastAVC = true;
        param.iTemporalLayerNum = 3;

        let decoders: Vec<*mut ISVCDecoder> = (0..spatial_layer_num)
            .map(|_| create_and_init_decoder())
            .collect();

        let w = width as usize;
        let h = height as usize;
        let mut buf = vec![0u8; w * h * 3 / 2];

        for (pattern_idx, pattern) in TEMPORAL_PATTERNS.iter().enumerate() {
            for i in 0..spatial_layer_num {
                param.sSpatialLayers[i].fFrameRate = frame_rate / (pattern[i] as f32);
            }
            assert_eq!(
                ISVCEncoder::InitializeExt(encoder, &param),
                CM_RESULT_SUCCESS
            );

            for frame_idx in 0..8usize {
                fill_camera_frame(&mut buf, w, h, frame_idx);
                let mut pic = SSourcePicture::default();
                pic.iPicWidth = width;
                pic.iPicHeight = height;
                pic.iColorFormat = EVideoFormatType::videoFormatI420 as i32;
                pic.iStride[0] = width;
                pic.iStride[1] = width >> 1;
                pic.iStride[2] = width >> 1;
                pic.pData[0] = buf.as_mut_ptr();
                pic.pData[1] = buf.as_mut_ptr().add(w * h);
                pic.pData[2] = buf.as_mut_ptr().add(w * h + (w * h >> 2));

                let mut info = SFrameBSInfo::default();
                assert_eq!(
                    ISVCEncoder::EncodeFrame(encoder, &pic, &mut info),
                    CM_RESULT_SUCCESS
                );

                if frame_idx == 4 {
                    ISVCEncoder::ForceIntraFrame(encoder, true);
                }

                let mut layer_bufs = vec![Vec::<u8>::new(); spatial_layer_num];
                for i_layer in 0..(info.iLayerNum as usize) {
                    let layer_info = &info.sLayerInfo[i_layer];
                    let first_nal_type = (*layer_info.pBsBuf.add(4)) & 0x1f;
                    assert!(
                        matches!(first_nal_type, 1 | 5 | 6 | 7 | 8),
                        "unexpected first_nal_type={first_nal_type}"
                    );
                    let nal_lens = std::slice::from_raw_parts(
                        layer_info.pNalLengthInByte,
                        layer_info.iNalCount as usize,
                    );
                    let layer_len: usize = nal_lens.iter().map(|&n| n.max(0) as usize).sum();
                    let idx = layer_info.uiSpatialId as usize;
                    assert!(idx < spatial_layer_num);
                    let bytes = std::slice::from_raw_parts(layer_info.pBsBuf, layer_len);
                    layer_bufs[idx].extend_from_slice(bytes);
                }

                for (idx, bs) in layer_bufs.iter().enumerate() {
                    if !bs.is_empty() {
                        let mut p_data: [*mut u8; 3] = [ptr::null_mut(); 3];
                        let mut dst_buf_info = SBufferInfo::default();
                        let hr1 = ISVCDecoder::DecodeFrame2(
                            decoders[idx],
                            bs.as_ptr(),
                            bs.len() as i32,
                            p_data.as_mut_ptr(),
                            &mut dst_buf_info,
                        );
                        assert_eq!(
                            hr1,
                            DECODING_STATE::dsErrorFree,
                            "pattern={pattern_idx} layer={idx}"
                        );
                        let hr2 = ISVCDecoder::DecodeFrame2(
                            decoders[idx],
                            ptr::null(),
                            0,
                            p_data.as_mut_ptr(),
                            &mut dst_buf_info,
                        );
                        assert_eq!(
                            hr2,
                            DECODING_STATE::dsErrorFree,
                            "pattern={pattern_idx} layer={idx}"
                        );
                        assert_eq!(
                            dst_buf_info.iBufferStatus, 1,
                            "pattern={pattern_idx} layer={idx}"
                        );
                    }
                }
            }
        }

        for dec in decoders {
            ISVCDecoder::Uninitialize(dec);
            WelsDestroyDecoder(dec);
        }
        ISVCEncoder::Uninitialize(encoder);
        WelsDestroySVCEncoder(encoder);
    }
}

/// Port of `EncodeDecodeTestAPI.DiffSlicingInDlayer` and `DiffSlicingInDlayerMixed`
/// (`test/api/encode_options_test.cpp:1571-1712`).
#[test]
fn diff_slicing_in_dlayer_and_mixed() {
    unsafe {
        // Part 1: DiffSlicingInDlayer (3 layers: SM_RASTER_SLICE [0], SM_RASTER_SLICE [30, 32], SM_FIXEDSLCNUM_SLICE)
        {
            let mut encoder: *mut ISVCEncoder = ptr::null_mut();
            assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

            let spatial_layer_num = 3usize;
            let width = 512i32;
            let height = 288i32;
            let mut param = SEncParamExt::default();
            prepare_param_default(
                encoder,
                spatial_layer_num as i32,
                1,
                width,
                height,
                15.0,
                &mut param,
            );
            param.iMultipleThreadIdc = 2;
            param.bSimulcastAVC = true;

            param.sSpatialLayers[0].iVideoWidth = width >> 2;
            param.sSpatialLayers[0].iVideoHeight = height >> 2;
            param.sSpatialLayers[0].sSliceArgument.uiSliceMode = SliceModeEnum::SM_RASTER_SLICE;

            param.sSpatialLayers[1].iVideoWidth = width >> 1;
            param.sSpatialLayers[1].iVideoHeight = height >> 1;
            param.sSpatialLayers[1].sSliceArgument.uiSliceMode = SliceModeEnum::SM_RASTER_SLICE;
            param.sSpatialLayers[1].sSliceArgument.uiSliceMbNum[0] = 30;
            param.sSpatialLayers[1].sSliceArgument.uiSliceMbNum[1] = 32;

            param.sSpatialLayers[2].iVideoWidth = width;
            param.sSpatialLayers[2].iVideoHeight = height;
            param.sSpatialLayers[2].sSliceArgument.uiSliceMode =
                SliceModeEnum::SM_FIXEDSLCNUM_SLICE;
            param.sSpatialLayers[2].sSliceArgument.uiSliceNum = 4;

            assert_eq!(
                ISVCEncoder::InitializeExt(encoder, &param),
                CM_RESULT_SUCCESS
            );

            let decoders: Vec<*mut ISVCDecoder> = (0..spatial_layer_num)
                .map(|_| create_and_init_decoder())
                .collect();

            test_one_simulcast_avc(encoder, &mut param, &decoders, spatial_layer_num, 6);

            for dec in decoders {
                ISVCDecoder::Uninitialize(dec);
                WelsDestroyDecoder(dec);
            }
            ISVCEncoder::Uninitialize(encoder);
            WelsDestroySVCEncoder(encoder);
        }

        // Part 2: DiffSlicingInDlayerMixed (2 layers mixing SM_SIZELIMITED_SLICE and SM_FIXEDSLCNUM_SLICE)
        {
            let mut encoder: *mut ISVCEncoder = ptr::null_mut();
            assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

            let spatial_layer_num = 2usize;
            let width = 512i32;
            let height = 288i32;
            let mut param = SEncParamExt::default();
            prepare_param_default(
                encoder,
                spatial_layer_num as i32,
                1,
                width,
                height,
                15.0,
                &mut param,
            );
            param.iMultipleThreadIdc = 4;
            param.bSimulcastAVC = true;

            param.sSpatialLayers[0].iVideoWidth = width >> 2;
            param.sSpatialLayers[0].iVideoHeight = height >> 2;
            param.sSpatialLayers[0].sSliceArgument.uiSliceMode =
                SliceModeEnum::SM_SIZELIMITED_SLICE;
            param.sSpatialLayers[0].sSliceArgument.uiSliceSizeConstraint = 1500;

            param.sSpatialLayers[1].iVideoWidth = width;
            param.sSpatialLayers[1].iVideoHeight = height;
            param.sSpatialLayers[1].sSliceArgument.uiSliceMode =
                SliceModeEnum::SM_FIXEDSLCNUM_SLICE;
            param.sSpatialLayers[1].sSliceArgument.uiSliceNum = 2;
            param.sSpatialLayers[1].sSliceArgument.uiSliceSizeConstraint = 1500;

            assert_eq!(
                ISVCEncoder::InitializeExt(encoder, &param),
                CM_RESULT_SUCCESS
            );

            let decoders: Vec<*mut ISVCDecoder> = (0..spatial_layer_num)
                .map(|_| create_and_init_decoder())
                .collect();

            test_one_simulcast_avc(encoder, &mut param, &decoders, spatial_layer_num, 6);

            for dec in decoders {
                ISVCDecoder::Uninitialize(dec);
                WelsDestroyDecoder(dec);
            }
            ISVCEncoder::Uninitialize(encoder);
            WelsDestroySVCEncoder(encoder);
        }
    }
}

/// Port of `EncodeDecodeTestAPI.ThreadNumAndSliceNum` and `TriggerLoadBalancing`
/// (`test/api/encode_options_test.cpp:1714-1894`).
#[test]
fn thread_num_and_slice_num_and_load_balancing() {
    unsafe {
        // 1. ThreadNumAndSliceNum (threads=3, slices=4 and slices=2)
        for slice_num in [2u32, 4u32] {
            let mut encoder: *mut ISVCEncoder = ptr::null_mut();
            assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);
            let decoder = create_and_init_decoder();

            let mut param = SEncParamExt::default();
            prepare_param_default(encoder, 1, 1, 320, 192, 15.0, &mut param);
            param.iMultipleThreadIdc = 3;
            param.bSimulcastAVC = true;
            param.sSpatialLayers[0].sSliceArgument.uiSliceMode =
                SliceModeEnum::SM_FIXEDSLCNUM_SLICE;
            param.sSpatialLayers[0].sSliceArgument.uiSliceNum = slice_num;

            assert_eq!(
                ISVCEncoder::InitializeExt(encoder, &param),
                CM_RESULT_SUCCESS
            );
            test_one_simulcast_avc(encoder, &mut param, &[decoder], 1, 6);

            ISVCDecoder::Uninitialize(decoder);
            WelsDestroyDecoder(decoder);
            ISVCEncoder::Uninitialize(encoder);
            WelsDestroySVCEncoder(encoder);
        }

        // 2. TriggerLoadBalancing (4 threads, 4 slices, complex strip shifts from top to bottom)
        {
            let mut encoder: *mut ISVCEncoder = ptr::null_mut();
            assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);
            let decoder = create_and_init_decoder();

            let width = 320i32;
            let height = 192i32;
            let mut param = SEncParamExt::default();
            prepare_param_default(encoder, 1, 1, width, height, 15.0, &mut param);
            param.iMultipleThreadIdc = 4;
            param.bSimulcastAVC = true;
            param.sSpatialLayers[0].sSliceArgument.uiSliceMode =
                SliceModeEnum::SM_FIXEDSLCNUM_SLICE;
            param.sSpatialLayers[0].sSliceArgument.uiSliceNum = 4;
            param.sSpatialLayers[0].sSliceArgument.uiSliceSizeConstraint = 1000;

            assert_eq!(
                ISVCEncoder::InitializeExt(encoder, &param),
                CM_RESULT_SUCCESS
            );
            assert_eq!(
                ISVCEncoder::SetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                    ptr::addr_of_mut!(param).cast(),
                ),
                CM_RESULT_SUCCESS
            );

            let w = width as usize;
            let h = height as usize;
            let mut buf = vec![0u8; w * h * 3 / 2];
            let mut rng = 0x4242_4242u32;

            for frame_idx in 0..10usize {
                buf.fill(64);
                let start_row = if frame_idx > 5 { h / 2 } else { 0 };
                for row in start_row..(start_row + h / 2) {
                    fill_seeded_noise(&mut buf[row * w..(row + 1) * w], &mut rng);
                }

                let mut pic = SSourcePicture::default();
                pic.iPicWidth = width;
                pic.iPicHeight = height;
                pic.iColorFormat = EVideoFormatType::videoFormatI420 as i32;
                pic.iStride[0] = width;
                pic.iStride[1] = width >> 1;
                pic.iStride[2] = width >> 1;
                pic.pData[0] = buf.as_mut_ptr();
                pic.pData[1] = buf.as_mut_ptr().add(w * h);
                pic.pData[2] = buf.as_mut_ptr().add(w * h + (w * h >> 2));

                let mut info = SFrameBSInfo::default();
                assert_eq!(
                    ISVCEncoder::EncodeFrame(encoder, &pic, &mut info),
                    CM_RESULT_SUCCESS
                );

                let bs = collect_frame_bitstream(&info);
                let mut p_data: [*mut u8; 3] = [ptr::null_mut(); 3];
                let mut dst_buf_info = SBufferInfo::default();
                assert_eq!(
                    ISVCDecoder::DecodeFrame2(
                        decoder,
                        bs.as_ptr(),
                        bs.len() as i32,
                        p_data.as_mut_ptr(),
                        &mut dst_buf_info,
                    ),
                    DECODING_STATE::dsErrorFree
                );
                assert_eq!(
                    ISVCDecoder::DecodeFrame2(
                        decoder,
                        ptr::null(),
                        0,
                        p_data.as_mut_ptr(),
                        &mut dst_buf_info,
                    ),
                    DECODING_STATE::dsErrorFree
                );
                assert_eq!(dst_buf_info.iBufferStatus, 1);
            }

            ISVCDecoder::Uninitialize(decoder);
            WelsDestroyDecoder(decoder);
            ISVCEncoder::Uninitialize(encoder);
            WelsDestroySVCEncoder(encoder);
        }
    }
}

/// Port of `EncodeDecodeTestAPI.ProfileLevelSetting` (`test/api/encode_options_test.cpp:1931-2077`).
#[test]
fn profile_level_setting_roundtrip() {
    let cases = [
        (EProfileIdc::PRO_BASELINE, ELevelIdc::LEVEL_3_1, 0),
        (EProfileIdc::PRO_MAIN, ELevelIdc::LEVEL_3_2, 1),
        (EProfileIdc::PRO_HIGH, ELevelIdc::LEVEL_4_0, 1),
        // Unsupported profile (PRO_EXTENDED) falls back to PRO_BASELINE (CAVLC) or PRO_HIGH (CABAC)
        (EProfileIdc::PRO_EXTENDED, ELevelIdc::LEVEL_3_0, 0),
        (EProfileIdc::PRO_EXTENDED, ELevelIdc::LEVEL_3_0, 1),
    ];

    for (enc_prof, enc_lvl, cabac) in cases {
        unsafe {
            let mut encoder: *mut ISVCEncoder = ptr::null_mut();
            assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);
            let decoder = create_and_init_decoder();

            let mut param = SEncParamExt::default();
            prepare_param_default(encoder, 1, 1, 320, 192, 15.0, &mut param);
            param.bSimulcastAVC = true;
            param.iRCMode = RC_MODES::RC_TIMESTAMP_MODE;
            param.iNumRefFrame = 1;
            param.iEntropyCodingModeFlag = cabac;
            param.iTargetBitrate = 3000;
            param.sSpatialLayers[0].iSpatialBitrate = 3000;
            param.sSpatialLayers[0].uiProfileIdc = enc_prof;
            param.sSpatialLayers[0].uiLevelIdc = enc_lvl;

            assert_eq!(
                ISVCEncoder::InitializeExt(encoder, &param),
                CM_RESULT_SUCCESS
            );
            enc_dec_one_frame(encoder, decoder, 320, 192, 0);

            let mut dec_prof = 0i32;
            ISVCDecoder::GetOption(
                decoder,
                DECODER_OPTION::DECODER_OPTION_PROFILE,
                ptr::addr_of_mut!(dec_prof).cast(),
            );
            let expected_prof = match enc_prof {
                EProfileIdc::PRO_BASELINE | EProfileIdc::PRO_MAIN | EProfileIdc::PRO_HIGH => {
                    enc_prof as i32
                }
                _ => {
                    if cabac != 0 {
                        EProfileIdc::PRO_HIGH as i32
                    } else {
                        EProfileIdc::PRO_BASELINE as i32
                    }
                }
            };
            assert_eq!(dec_prof, expected_prof);

            let mut dec_lvl = 0i32;
            ISVCDecoder::GetOption(
                decoder,
                DECODER_OPTION::DECODER_OPTION_LEVEL,
                ptr::addr_of_mut!(dec_lvl).cast(),
            );
            assert_eq!(dec_lvl, enc_lvl as i32);

            // Reinitialize with new profile/level and exercise SetOption/GetOption + ForceIntraFrame
            // (matching test/api/encode_options_test.cpp:2012-2076)
            assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS as i32);

            param.sSpatialLayers[0].uiProfileIdc = EProfileIdc::PRO_MAIN;
            param.sSpatialLayers[0].uiLevelIdc = ELevelIdc::LEVEL_4_1;
            assert_eq!(
                ISVCEncoder::InitializeExt(encoder, &param),
                CM_RESULT_SUCCESS
            );

            let mut prof_info = SProfileInfo {
                iLayer: 0,
                uiProfileIdc: EProfileIdc::PRO_MAIN,
            };
            assert_eq!(
                ISVCEncoder::SetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_OPTION_PROFILE,
                    ptr::addr_of_mut!(prof_info).cast(),
                ),
                CM_RESULT_SUCCESS
            );
            let mut lvl_info = SLevelInfo {
                iLayer: 0,
                uiLevelIdc: ELevelIdc::LEVEL_4_1,
            };
            assert_eq!(
                ISVCEncoder::SetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_OPTION_LEVEL,
                    ptr::addr_of_mut!(lvl_info).cast(),
                ),
                CM_RESULT_SUCCESS
            );
            let mut ext = SEncParamExt::default();
            assert_eq!(
                ISVCEncoder::GetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                    ptr::addr_of_mut!(ext).cast(),
                ),
                CM_RESULT_SUCCESS
            );
            assert_eq!(ext.sSpatialLayers[0].uiProfileIdc, EProfileIdc::PRO_MAIN);
            assert_eq!(ext.sSpatialLayers[0].uiLevelIdc, ELevelIdc::LEVEL_4_1);

            assert_eq!(
                ISVCEncoder::ForceIntraFrame(encoder, true),
                CM_RESULT_SUCCESS
            );
            enc_dec_one_frame(encoder, decoder, 320, 192, 1);

            ISVCDecoder::GetOption(
                decoder,
                DECODER_OPTION::DECODER_OPTION_PROFILE,
                ptr::addr_of_mut!(dec_prof).cast(),
            );
            assert_eq!(dec_prof, EProfileIdc::PRO_MAIN as i32);
            ISVCDecoder::GetOption(
                decoder,
                DECODER_OPTION::DECODER_OPTION_LEVEL,
                ptr::addr_of_mut!(dec_lvl).cast(),
            );
            assert_eq!(dec_lvl, ELevelIdc::LEVEL_4_1 as i32);

            ISVCDecoder::Uninitialize(decoder);
            WelsDestroyDecoder(decoder);
            ISVCEncoder::Uninitialize(encoder);
            WelsDestroySVCEncoder(encoder);
        }
    }
}

/// Port of `EncodeDecodeTestAPI.AVCSVCExtensionCheck`
/// (`test/api/encode_options_test.cpp:2079-2135`).
#[test]
fn avc_svc_extension_check() {
    for &(spatial_layers, simulcast_avc) in &[(1, false), (1, true), (3, false), (3, true)] {
        unsafe {
            let mut encoder: *mut ISVCEncoder = ptr::null_mut();
            assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

            let width = 640i32;
            let height = 360i32;
            let mut param = SEncParamExt::default();
            prepare_param_default(encoder, spatial_layers, 1, width, height, 15.0, &mut param);
            param.bSimulcastAVC = simulcast_avc;
            param.iRCMode = RC_MODES::RC_TIMESTAMP_MODE;
            param.iNumRefFrame = 1;
            param.sSpatialLayers[0].iSpatialBitrate = 500;
            param.sSpatialLayers[1].iSpatialBitrate = 1000;
            param.sSpatialLayers[2].iSpatialBitrate = 2200;
            param.iTargetBitrate = 4000;

            assert_eq!(
                ISVCEncoder::InitializeExt(encoder, &param),
                CM_RESULT_SUCCESS
            );

            let w = width as usize;
            let h = height as usize;
            let mut buf = vec![0u8; w * h * 3 / 2];

            for frame_idx in 0..6usize {
                fill_camera_frame(&mut buf, w, h, frame_idx);
                let mut pic = SSourcePicture::default();
                pic.iPicWidth = width;
                pic.iPicHeight = height;
                pic.iColorFormat = EVideoFormatType::videoFormatI420 as i32;
                pic.iStride[0] = width;
                pic.iStride[1] = width >> 1;
                pic.iStride[2] = width >> 1;
                pic.pData[0] = buf.as_mut_ptr();
                pic.pData[1] = buf.as_mut_ptr().add(w * h);
                pic.pData[2] = buf.as_mut_ptr().add(w * h + (w * h >> 2));

                let mut info = SFrameBSInfo::default();
                assert_eq!(
                    ISVCEncoder::EncodeFrame(encoder, &pic, &mut info),
                    CM_RESULT_SUCCESS
                );

                for i_layer in 0..(info.iLayerNum as usize) {
                    let layer = &info.sLayerInfo[i_layer];
                    let mut offset = 0usize;
                    for nal_idx in 0..(layer.iNalCount as usize) {
                        let nal_len = *layer.pNalLengthInByte.add(nal_idx) as usize;
                        let type_byte = (*layer.pBsBuf.add(offset + 4)) & 0x1f;
                        offset += nal_len;
                        if simulcast_avc {
                            assert!(
                                type_byte <= 8,
                                "simulcastAVC: spatial_id={}, type_byte={type_byte}",
                                layer.uiSpatialId
                            );
                        } else if layer.uiSpatialId == 0 {
                            assert!(
                                type_byte <= 8
                                    || type_byte == 14
                                    || (layer.uiLayerType
                                        == LAYER_TYPE::NON_VIDEO_CODING_LAYER as u8
                                        && type_byte == 15),
                                "simulcastSVC: spatial_id=0, type_byte={type_byte}"
                            );
                        } else {
                            assert!(
                                type_byte >= 14,
                                "simulcastSVC: spatial_id={}, type_byte={type_byte}",
                                layer.uiSpatialId
                            );
                        }
                    }
                }
            }

            assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
            WelsDestroySVCEncoder(encoder);
        }
    }
}

/// Port of `EncodeDecodeTestAPI.UnsupportedVideoSizeInput` and
/// `ScreenContent_LosslessLink0_EnableLongTermReference`
/// (`test/api/encode_options_test.cpp:2137-2208`).
#[test]
fn unsupported_video_size_input_and_screen_ltr_non_lossless() {
    unsafe {
        // 1. UnsupportedVideoSizeInput
        {
            let mut encoder: *mut ISVCEncoder = ptr::null_mut();
            assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

            let mut param = SEncParamExt::default();
            prepare_param_default(encoder, 1, 1, 640, 360, 15.0, &mut param);
            param.iRCMode = RC_MODES::RC_OFF_MODE;
            param.iNumRefFrame = 1;
            assert_eq!(
                ISVCEncoder::InitializeExt(encoder, &param),
                CM_RESULT_SUCCESS
            );

            let mut small_buf = vec![128u8; 8 * 8 * 3 / 2];
            let mut small_pic = SSourcePicture::default();
            small_pic.iPicWidth = 8;
            small_pic.iPicHeight = 8;
            small_pic.iColorFormat = EVideoFormatType::videoFormatI420 as i32;
            small_pic.iStride[0] = 8;
            small_pic.iStride[1] = 4;
            small_pic.iStride[2] = 4;
            small_pic.pData[0] = small_buf.as_mut_ptr();
            small_pic.pData[1] = small_buf.as_mut_ptr().add(64);
            small_pic.pData[2] = small_buf.as_mut_ptr().add(80);

            let mut info = SFrameBSInfo::default();
            assert_eq!(
                ISVCEncoder::EncodeFrame(encoder, &small_pic, &mut info),
                CM_UNSUPPORTED_DATA
            );

            let mut valid_buf = vec![128u8; 640 * 360 * 3 / 2];
            let mut valid_pic = SSourcePicture::default();
            valid_pic.iPicWidth = 640;
            valid_pic.iPicHeight = 360;
            valid_pic.iColorFormat = EVideoFormatType::videoFormatI420 as i32;
            valid_pic.iStride[0] = 640;
            valid_pic.iStride[1] = 320;
            valid_pic.iStride[2] = 320;
            valid_pic.pData[0] = valid_buf.as_mut_ptr();
            valid_pic.pData[1] = valid_buf.as_mut_ptr().add(640 * 360);
            valid_pic.pData[2] = valid_buf.as_mut_ptr().add(640 * 360 + (640 * 360 >> 2));

            assert_eq!(
                ISVCEncoder::EncodeFrame(encoder, &valid_pic, &mut info),
                CM_RESULT_SUCCESS
            );
            assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
            WelsDestroySVCEncoder(encoder);
        }

        // 2. ScreenContent_LosslessLink0_EnableLongTermReference (2882x1808)
        {
            let mut encoder: *mut ISVCEncoder = ptr::null_mut();
            assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

            let width = 2882i32;
            let height = 1808i32;
            let mut param = SEncParamExt::default();
            prepare_param_default(encoder, 1, 1, width, height, 15.0, &mut param);
            param.iUsageType = EUsageType::SCREEN_CONTENT_REAL_TIME;
            param.bEnableLongTermReference = true;
            param.bIsLosslessLink = false;

            assert_eq!(
                ISVCEncoder::InitializeExt(encoder, &param),
                CM_RESULT_SUCCESS
            );

            let w = width as usize;
            let h = height as usize;
            let mut buf = vec![0u8; w * h * 3 / 2];
            fill_screen_frame(&mut buf, w, h, 0);

            let mut pic = SSourcePicture::default();
            pic.iPicWidth = width;
            pic.iPicHeight = height;
            pic.iColorFormat = EVideoFormatType::videoFormatI420 as i32;
            pic.iStride[0] = width;
            pic.iStride[1] = width >> 1;
            pic.iStride[2] = width >> 1;
            pic.pData[0] = buf.as_mut_ptr();
            pic.pData[1] = buf.as_mut_ptr().add(w * h);
            pic.pData[2] = buf.as_mut_ptr().add(w * h + (w * h >> 2));

            let mut info = SFrameBSInfo::default();
            assert_eq!(
                ISVCEncoder::EncodeFrame(encoder, &pic, &mut info),
                CM_RESULT_SUCCESS
            );
            assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
            WelsDestroySVCEncoder(encoder);
        }
    }
}

/// Port of `EncodeDecodeTestAPI.TemporalLayerChangeDuringEncoding` and
/// `TemporalLayerChangeDuringEncoding_Specific` (`test/api/encode_options_test.cpp:2211-2344`).
#[test]
fn temporal_layer_change_during_encoding() {
    unsafe {
        // 1. TemporalLayerChangeDuringEncoding (2 -> 3 -> 1 -> 2)
        {
            let mut encoder: *mut ISVCEncoder = ptr::null_mut();
            assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

            let width = 320i32;
            let height = 192i32;
            let mut param = SEncParamExt::default();
            prepare_param_default(encoder, 1, 1, width, height, 15.0, &mut param);
            param.iTemporalLayerNum = 2;

            assert_eq!(
                ISVCEncoder::InitializeExt(encoder, &param),
                CM_RESULT_SUCCESS
            );

            let w = width as usize;
            let h = height as usize;
            let mut buf = vec![0u8; w * h * 3 / 2];

            for frame_num in 0..20usize {
                fill_camera_frame(&mut buf, w, h, frame_num);
                let mut pic = SSourcePicture::default();
                pic.iPicWidth = width;
                pic.iPicHeight = height;
                pic.iColorFormat = EVideoFormatType::videoFormatI420 as i32;
                pic.iStride[0] = width;
                pic.iStride[1] = width >> 1;
                pic.iStride[2] = width >> 1;
                pic.pData[0] = buf.as_mut_ptr();
                pic.pData[1] = buf.as_mut_ptr().add(w * h);
                pic.pData[2] = buf.as_mut_ptr().add(w * h + (w * h >> 2));

                let mut info = SFrameBSInfo::default();
                assert_eq!(
                    ISVCEncoder::EncodeFrame(encoder, &pic, &mut info),
                    CM_RESULT_SUCCESS
                );

                if frame_num == 5 {
                    param.iTemporalLayerNum = 3;
                    param.iTargetBitrate = 1_500_000;
                    param.sSpatialLayers[0].iSpatialBitrate = 1_500_000;
                    param.fMaxFrameRate = 30.0;
                    assert_eq!(
                        ISVCEncoder::SetOption(
                            encoder,
                            ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                            ptr::addr_of_mut!(param).cast(),
                        ),
                        CM_RESULT_SUCCESS
                    );
                } else if frame_num == 10 {
                    param.iTemporalLayerNum = 1;
                    param.iTargetBitrate = 500_000;
                    param.sSpatialLayers[0].iSpatialBitrate = 500_000;
                    param.fMaxFrameRate = 7.5;
                    assert_eq!(
                        ISVCEncoder::SetOption(
                            encoder,
                            ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                            ptr::addr_of_mut!(param).cast(),
                        ),
                        CM_RESULT_SUCCESS
                    );
                } else if frame_num == 15 {
                    param.iTemporalLayerNum = 2;
                    param.iTargetBitrate = 1_000_000;
                    param.sSpatialLayers[0].iSpatialBitrate = 1_000_000;
                    param.fMaxFrameRate = 15.0;
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

            assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
            WelsDestroySVCEncoder(encoder);
        }

        // 2. TemporalLayerChangeDuringEncoding_Specific (steps [2, 1, 3] with iNumRefFrame = 1)
        {
            let mut encoder: *mut ISVCEncoder = ptr::null_mut();
            assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

            let width = 320i32;
            let height = 192i32;
            let mut param = SEncParamExt::default();
            prepare_param_default(encoder, 1, 1, width, height, 15.0, &mut param);
            param.iTemporalLayerNum = 1;
            param.iNumRefFrame = 1;

            assert_eq!(
                ISVCEncoder::InitializeExt(encoder, &param),
                CM_RESULT_SUCCESS
            );

            let w = width as usize;
            let h = height as usize;
            let mut buf = vec![0u8; w * h * 3 / 2];

            let steps = [2i32, 1, 3];
            let mut step_idx = 0usize;
            let total_frames = 12usize;

            for frame_num in 0..total_frames {
                let mut set_opt = false;
                if step_idx < 3 && frame_num == (total_frames / 3) * (step_idx + 1) {
                    let s = steps[step_idx];
                    param.iTemporalLayerNum = s;
                    param.iTargetBitrate = 500_000 * s;
                    param.sSpatialLayers[0].iSpatialBitrate = 500_000 * s;
                    let fr = 7.5f32 * (2.0f32).powi(s);
                    param.fMaxFrameRate = fr;
                    param.sSpatialLayers[0].fFrameRate = fr;
                    assert_eq!(
                        ISVCEncoder::SetOption(
                            encoder,
                            ENCODER_OPTION::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                            ptr::addr_of_mut!(param).cast(),
                        ),
                        CM_RESULT_SUCCESS
                    );
                    set_opt = true;
                    step_idx += 1;
                }

                fill_camera_frame(&mut buf, w, h, frame_num);
                let mut pic = SSourcePicture::default();
                pic.iPicWidth = width;
                pic.iPicHeight = height;
                pic.iColorFormat = EVideoFormatType::videoFormatI420 as i32;
                pic.iStride[0] = width;
                pic.iStride[1] = width >> 1;
                pic.iStride[2] = width >> 1;
                pic.pData[0] = buf.as_mut_ptr();
                pic.pData[1] = buf.as_mut_ptr().add(w * h);
                pic.pData[2] = buf.as_mut_ptr().add(w * h + (w * h >> 2));

                let mut info = SFrameBSInfo::default();
                assert_eq!(
                    ISVCEncoder::EncodeFrame(encoder, &pic, &mut info),
                    CM_RESULT_SUCCESS
                );

                if set_opt {
                    if step_idx == 1 || step_idx == 3 {
                        assert_eq!(
                            info.eFrameType,
                            EVideoFrameType::videoFrameTypeIDR,
                            "step_idx={step_idx} frame_num={frame_num}"
                        );
                    } else {
                        assert_ne!(
                            info.eFrameType,
                            EVideoFrameType::videoFrameTypeIDR,
                            "step_idx={step_idx} frame_num={frame_num}"
                        );
                    }
                }
            }

            assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
            WelsDestroySVCEncoder(encoder);
        }
    }
}

/// Port of `EncodeDecodeTestAPI.ENCODER_OPTION_IDR_INTERVAL`
/// (`test/api/encode_options_test.cpp:2346-2390`).
#[test]
fn encoder_option_idr_interval_schedule() {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let width = 320i32;
        let height = 192i32;
        let mut param = SEncParamExt::default();
        prepare_param_default(encoder, 1, 1, width, height, 15.0, &mut param);
        param.iTemporalLayerNum = 1;

        let mut trace_level: i32 = 0;
        assert_eq!(
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_OPTION_TRACE_LEVEL,
                ptr::addr_of_mut!(trace_level).cast(),
            ),
            CM_RESULT_SUCCESS
        );
        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param),
            CM_RESULT_SUCCESS
        );

        let w = width as usize;
        let h = height as usize;
        let mut buf = vec![0u8; w * h * 3 / 2];
        fill_camera_frame(&mut buf, w, h, 0);

        let mut pic = SSourcePicture::default();
        pic.iPicWidth = width;
        pic.iPicHeight = height;
        pic.iColorFormat = EVideoFormatType::videoFormatI420 as i32;
        pic.iStride[0] = width;
        pic.iStride[1] = width >> 1;
        pic.iStride[2] = width >> 1;
        pic.pData[0] = buf.as_mut_ptr();
        pic.pData[1] = buf.as_mut_ptr().add(w * h);
        pic.pData[2] = buf.as_mut_ptr().add(w * h + (w * h >> 2));

        let mut info = SFrameBSInfo::default();
        assert_eq!(
            ISVCEncoder::EncodeFrame(encoder, &pic, &mut info),
            CM_RESULT_SUCCESS
        );
        assert_eq!(info.eFrameType, EVideoFrameType::videoFrameTypeIDR);

        let mut last_idr_idx = 0i32;
        let mut frame = 1i32;

        for &target_intra_period in &[4i32, 1, 6, 0, -1, 5] {
            let mut period = target_intra_period;
            assert_eq!(
                ISVCEncoder::SetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_OPTION_IDR_INTERVAL,
                    ptr::addr_of_mut!(period).cast(),
                ),
                CM_RESULT_SUCCESS
            );

            let enc_frame_num = (target_intra_period * 3).max(0) as usize;
            for i in 0..enc_frame_num {
                fill_camera_frame(&mut buf, w, h, frame as usize);
                pic.pData[0] = buf.as_mut_ptr();
                pic.pData[1] = buf.as_mut_ptr().add(w * h);
                pic.pData[2] = buf.as_mut_ptr().add(w * h + (w * h >> 2));
                assert_eq!(
                    ISVCEncoder::EncodeFrame(encoder, &pic, &mut info),
                    CM_RESULT_SUCCESS
                );

                if target_intra_period <= 0 || ((frame - last_idr_idx) % target_intra_period) == 0 {
                    assert_eq!(
                        info.eFrameType,
                        EVideoFrameType::videoFrameTypeIDR,
                        "target_intra_period={target_intra_period} i={i} frame={frame}"
                    );
                    last_idr_idx = frame;
                } else {
                    assert_ne!(
                        info.eFrameType,
                        EVideoFrameType::videoFrameTypeIDR,
                        "target_intra_period={target_intra_period} i={i} frame={frame}"
                    );
                }
                frame += 1;
            }
        }

        assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
        WelsDestroySVCEncoder(encoder);
    }
}
