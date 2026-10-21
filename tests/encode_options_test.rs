// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Integration test for dynamic encoder options and runtime reconfiguration.
//! Ported from upstream `test/api/encode_options_test.cpp`.

#[path = "common/mod.rs"]
mod common;

use common::frame_gen::{fill_camera_frame, fill_screen_frame, fill_seeded_noise};
use openh264rs::api::codec_api::*;
use openh264rs::api::{Decoder, Encoder};

#[test]
pub(crate) fn test_encoder_set_and_get_options() {
    let mut encoder = Encoder::new();

    let mut param = SEncParamBase::default();
    param.iPicWidth = 320;
    param.iPicHeight = 240;
    param.fMaxFrameRate = 30.0;
    param.iTargetBitrate = 500000;
    param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;

    let init_ret = encoder.initialize(&param);
    assert_eq!(init_ret, CM_RESULT_SUCCESS);

    // 1. Test frame rate option modification
    assert_eq!(encoder.set_frame_rate(15.0), CM_RESULT_SUCCESS);

    // 2. Test bitrate option modification
    let mut bitrate_info = SBitrateInfo::default();
    bitrate_info.iBitrate = 300000;
    assert_eq!(encoder.set_bitrate(&bitrate_info), CM_RESULT_SUCCESS);

    // 3. Test IDR interval option modification
    assert_eq!(encoder.set_idr_interval(60), CM_RESULT_SUCCESS);

    // 4. Test SEncParamBase query
    assert!(encoder.encode_param_base().is_some());

    // 5. Test SEncParamExt query
    assert!(encoder.encode_param_ext().is_some());

    // 6. Test statistics option query
    assert!(encoder.statistics().is_some());

    let uninit_ret = encoder.uninitialize();
    assert_eq!(uninit_ret, CM_RESULT_SUCCESS);
}

/// Every typed option `Encoder` handles, driven on a 160x96 configuration.
#[test]
pub(crate) fn test_set_get_option_matches_cxx_for_every_option() {
    let mut encoder = Encoder::new();

    let mut param = SEncParamBase::default();
    param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
    param.iPicWidth = 160;
    param.iPicHeight = 96;
    param.iTargetBitrate = 500000;
    param.iRCMode = RC_MODES::RC_QUALITY_MODE;
    param.fMaxFrameRate = 6.0;
    assert_eq!(encoder.initialize(&param), CM_RESULT_SUCCESS);

    assert!(encoder.data_format().is_some());
    assert!(encoder.idr_interval().is_some());
    assert!(encoder.encode_param_ext().is_some());
    assert!(encoder.encode_param_base().is_some());
    assert!(encoder.frame_rate().is_some());
    assert!(encoder.bitrate(LAYER_NUM::SPATIAL_LAYER_ALL).is_some());
    assert!(encoder.max_bitrate(LAYER_NUM::SPATIAL_LAYER_ALL).is_some());
    assert!(encoder.statistics().is_some());
    assert!(encoder.statistics_log_interval().is_some());
    assert!(encoder.complexity().is_some());

    assert_eq!(encoder.set_rc_mode(RC_MODES::RC_BITRATE_MODE), 0);
    assert_eq!(encoder.set_rc_frame_skip(true), 0);
    assert_eq!(encoder.set_padding(1), 0);
    assert_eq!(encoder.set_ltr_marking_period(30), 0);
    assert_eq!(
        encoder.set_ltr_config(&SLTRConfig {
            bEnableLongTermReference: true,
            iLTRRefNum: 2,
        }),
        0
    );
    assert_eq!(encoder.set_enable_ssei(true), 0);
    assert_eq!(encoder.set_enable_prefix_nal_adding(true), 0);
    assert_eq!(
        encoder.set_sps_pps_id_strategy(EParameterSetStrategy::INCREASING_ID),
        0
    );
    assert_eq!(
        encoder.set_sps_pps_id_strategy(EParameterSetStrategy::CONSTANT_ID),
        0
    );
    assert_eq!(
        encoder.set_profile(&SProfileInfo {
            iLayer: LAYER_NUM::SPATIAL_LAYER_0 as i32,
            uiProfileIdc: EProfileIdc::PRO_HIGH,
        }),
        0
    );
    // iLayer outside SPATIAL_LAYER_0..SPATIAL_LAYER_3 is rejected.
    assert_eq!(
        encoder.set_profile(&SProfileInfo {
            iLayer: 7,
            uiProfileIdc: EProfileIdc::PRO_HIGH,
        }),
        CM_INIT_PARA_ERROR
    );
    assert_eq!(
        encoder.set_level(&SLevelInfo {
            iLayer: LAYER_NUM::SPATIAL_LAYER_0 as i32,
            uiLevelIdc: ELevelIdc::LEVEL_3_0,
        }),
        0
    );
    assert_eq!(encoder.set_number_ref(3), 0);
    assert_eq!(
        encoder.set_delivery_status(&SDeliveryStatus {
            bDeliveryFlag: true,
            iDropFrameType: 0,
            iDropFrameSize: 0,
        }),
        0
    );
    assert_eq!(encoder.set_statistics_log_interval(500), 0);
    assert_eq!(encoder.set_is_lossless_link(true), 0);
    assert_eq!(encoder.set_bits_vary_percentage(20), 0);

    let mut rec = SLTRRecoverRequest::default();
    assert_eq!(encoder.set_ltr_recovery_request(&mut rec), 0);
    let mut fb = SLTRMarkingFeedback::default();
    assert_eq!(encoder.set_ltr_marking_feedback(&mut fb), 0);

    // ---- read back what those options wrote ------------------------------
    let ext = encoder.encode_param_ext().unwrap();
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

    assert_eq!(
        encoder.statistics_log_interval(),
        Some(500),
        "iStatisticsLogInterval"
    );

    assert_eq!(encoder.uninitialize(), CM_RESULT_SUCCESS);
}

// ============================================================================
// Shared helpers for `EncodeDecodeTestAPI` ports (`test/api/encode_options_test.cpp`)
// ============================================================================

fn valid_size(size: i32) -> i32 {
    if size > 16 { size } else { 16 }
}

fn prepare_param_default(
    encoder: &Encoder,
    layers: i32,
    slices: u32,
    width: i32,
    height: i32,
    framerate: f32,
    param: &mut SEncParamExt,
) {
    *param = SEncParamExt::default();
    assert_eq!(encoder.default_params(param), CM_RESULT_SUCCESS);
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
        param.sSpatialLayers[i].sSliceArgument.uiSliceMode = SliceModeEnum::SM_FIXEDSLCNUM_SLICE;
        param.sSpatialLayers[i].sSliceArgument.uiSliceNum = slices;
    }
}

fn collect_frame_bitstream(enc: &Encoder, info: &SFrameBSInfo) -> Vec<u8> {
    let mut out = Vec::new();
    for layer in info.layers() {
        out.extend_from_slice(layer.bs_buf(enc));
    }
    out
}

fn create_and_init_decoder() -> Decoder {
    let mut decoder = Decoder::new();
    let dec_param = SDecodingParam {
        uiTargetDqLayer: u8::MAX,
        eEcActiveIdc: ERROR_CON_IDC::ERROR_CON_SLICE_COPY,
        sVideoProperty: SVideoProperty {
            size: size_of::<SVideoProperty>() as u32,
            eVideoBsType: VIDEO_BITSTREAM_DEFAULT,
        },
        ..SDecodingParam::default()
    };
    assert_eq!(decoder.initialize(&dec_param), 0);
    decoder
}

fn enc_dec_one_frame(
    encoder: &mut Encoder,
    decoder: &mut Decoder,
    width: i32,
    height: i32,
    frame_idx: usize,
) -> SFrameBSInfo {
    let w = width as usize;
    let h = height as usize;
    let mut buf = vec![0u8; w * h * 3 / 2];
    fill_camera_frame(&mut buf, w, h, frame_idx);

    let pic = SourcePicture::from_i420(&buf, width, height);

    let mut info = SFrameBSInfo::default();
    let rv = encoder.encode_frame(&pic, &mut info);
    assert_eq!(
        rv, CM_RESULT_SUCCESS,
        "EncodeFrame failed at frame {frame_idx}"
    );

    let bs = collect_frame_bitstream(encoder, &info);
    let mut dst_buf_info = SBufferInfo::default();
    let dec_rv = decoder.decode_no_delay(Some(&bs), &mut dst_buf_info);
    assert_eq!(
        dec_rv,
        DECODING_STATE::dsErrorFree,
        "DecodeFrameNoDelay failed at frame {frame_idx}: {dec_rv:?}"
    );
    info
}

/// Port of `EncodeDecodeTestAPIBase::TestOneSimulcastAVC`
/// (`test/api/encode_decode_api_test.cpp:124-201`).
fn test_one_simulcast_avc(
    encoder: &mut Encoder,
    param: &SEncParamExt,
    decoders: &mut [Decoder],
    spatial_layer_num: usize,
    enc_frame_num: usize,
) {
    let rv = encoder.set_encode_param_ext(param);
    assert_eq!(rv, CM_RESULT_SUCCESS, "SetOption failed: rv={rv}");

    let w = param.iPicWidth as usize;
    let h = param.iPicHeight as usize;
    let mut buf = vec![0u8; w * h * 3 / 2];

    for frame_idx in 0..enc_frame_num {
        fill_camera_frame(&mut buf, w, h, frame_idx);
        let pic = SourcePicture::from_i420(&buf, param.iPicWidth, param.iPicHeight);

        let mut info = SFrameBSInfo::default();
        assert_eq!(encoder.encode_frame(&pic, &mut info), CM_RESULT_SUCCESS);

        let mut layer_bufs = vec![Vec::<u8>::new(); spatial_layer_num];
        for layer_info in info.layers() {
            let idx = layer_info.uiSpatialId as usize;
            assert!(idx < spatial_layer_num);
            layer_bufs[idx].extend_from_slice(layer_info.bs_buf(encoder));
        }

        for (idx, bs) in layer_bufs.iter().enumerate() {
            let mut dst_buf_info = SBufferInfo::default();
            let hr1 = decoders[idx].decode(Some(bs), &mut dst_buf_info);
            assert_eq!(
                hr1,
                DECODING_STATE::dsErrorFree,
                "layer={idx} frame={frame_idx}"
            );
            let hr2 = decoders[idx].decode(None, &mut dst_buf_info);
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

/// Port of `EncodeDecodeTestAPI.SetOptionEncParamBase` (`test/api/encode_options_test.cpp:349-391`).
#[test]
pub(crate) fn set_option_enc_param_base_roundtrip() {
    let mut encoder = Encoder::new();
    let mut decoder = create_and_init_decoder();

    let mut param = SEncParamExt::default();
    prepare_param_default(&encoder, 4, 1, 320, 192, 15.0, &mut param);
    assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);

    encoder.set_trace_level(0);

    let w = param.iPicWidth as usize;
    let h = param.iPicHeight as usize;
    let mut buf = vec![0u8; w * h * 3 / 2];

    for i in 0..8usize {
        let base = SEncParamBase {
            iUsageType: param.iUsageType,
            iPicWidth: param.iPicWidth,
            iPicHeight: param.iPicHeight,
            iTargetBitrate: ((i as i32 + 1) * 150_000),
            iRCMode: param.iRCMode,
            fMaxFrameRate: param.fMaxFrameRate,
        };
        let res = encoder.set_encode_param_base(&base);
        if res == CM_RESULT_SUCCESS {
            fill_camera_frame(&mut buf, w, h, i);
            let pic = SourcePicture::from_i420(&buf, param.iPicWidth, param.iPicHeight);

            let mut info = SFrameBSInfo::default();
            assert_eq!(encoder.encode_frame(&pic, &mut info), CM_RESULT_SUCCESS);
            let bs = collect_frame_bitstream(&encoder, &info);
            let mut dst_buf_info = SBufferInfo::default();
            assert_eq!(
                decoder.decode(Some(&bs), &mut dst_buf_info),
                DECODING_STATE::dsErrorFree
            );
            assert_eq!(
                decoder.decode(None, &mut dst_buf_info),
                DECODING_STATE::dsErrorFree
            );
            assert_eq!(dst_buf_info.iBufferStatus, 1);
        }
    }

    decoder.uninitialize();
    encoder.uninitialize();
}

/// Port of `EncodeDecodeTestAPI.ParameterSetStrategy_INCREASING_ID`
/// (`test/api/encode_options_test.cpp:401-511`) and
/// `ParameterSetStrategy_SPS_LISTING_AND_PPS_INCREASING1` (`lines 514-621`).
fn run_parameter_set_strategy_three_params(strategy: EParameterSetStrategy, simulcast_avc: bool) {
    let mut encoder = Encoder::new();
    let mut decoder = create_and_init_decoder();

    let mut param1 = SEncParamExt::default();
    prepare_param_default(&encoder, 1, 1, 320, 192, 15.0, &mut param1);
    param1.bSimulcastAVC = simulcast_avc;
    param1.eSpsPpsIdStrategy = strategy;

    let mut param2 = param1;
    prepare_param_default(&encoder, 1, 1, 160, 192, 15.0, &mut param2);
    param2.bSimulcastAVC = simulcast_avc;
    param2.eSpsPpsIdStrategy = strategy;

    let mut param3 = param1;
    prepare_param_default(&encoder, 1, 1, 320, 96, 15.0, &mut param3);
    param3.bSimulcastAVC = simulcast_avc;
    param3.eSpsPpsIdStrategy = strategy;

    // Part 1
    let mut frame_num = 0usize;
    assert_eq!(encoder.initialize_ext(&param1), CM_RESULT_SUCCESS);
    enc_dec_one_frame(
        &mut encoder,
        &mut decoder,
        param1.iPicWidth,
        param1.iPicHeight,
        frame_num,
    );
    frame_num += 1;

    assert_eq!(encoder.force_intra_frame(true, -1), CM_RESULT_SUCCESS);
    enc_dec_one_frame(
        &mut encoder,
        &mut decoder,
        param1.iPicWidth,
        param1.iPicHeight,
        frame_num,
    );
    frame_num += 1;

    assert_eq!(encoder.set_encode_param_ext(&param2), CM_RESULT_SUCCESS);
    enc_dec_one_frame(
        &mut encoder,
        &mut decoder,
        param2.iPicWidth,
        param2.iPicHeight,
        frame_num,
    );
    frame_num += 1;

    assert_eq!(encoder.force_intra_frame(true, -1), CM_RESULT_SUCCESS);
    enc_dec_one_frame(
        &mut encoder,
        &mut decoder,
        param2.iPicWidth,
        param2.iPicHeight,
        frame_num,
    );
    frame_num += 1;

    assert_eq!(encoder.set_encode_param_ext(&param1), CM_RESULT_SUCCESS);
    enc_dec_one_frame(
        &mut encoder,
        &mut decoder,
        param1.iPicWidth,
        param1.iPicHeight,
        frame_num,
    );
    frame_num += 1;

    assert_eq!(encoder.set_encode_param_ext(&param2), CM_RESULT_SUCCESS);
    enc_dec_one_frame(
        &mut encoder,
        &mut decoder,
        param2.iPicWidth,
        param2.iPicHeight,
        frame_num,
    );

    assert_eq!(encoder.uninitialize(), CM_RESULT_SUCCESS);

    // Part 2
    assert_eq!(encoder.initialize_ext(&param1), CM_RESULT_SUCCESS);
    assert_eq!(encoder.set_encode_param_ext(&param2), CM_RESULT_SUCCESS);
    assert_eq!(encoder.set_encode_param_ext(&param3), CM_RESULT_SUCCESS);

    enc_dec_one_frame(
        &mut encoder,
        &mut decoder,
        param3.iPicWidth,
        param3.iPicHeight,
        0,
    );
    assert_eq!(encoder.set_encode_param_ext(&param2), CM_RESULT_SUCCESS);
    enc_dec_one_frame(
        &mut encoder,
        &mut decoder,
        param2.iPicWidth,
        param2.iPicHeight,
        1,
    );
    assert_eq!(encoder.set_encode_param_ext(&param1), CM_RESULT_SUCCESS);
    enc_dec_one_frame(
        &mut encoder,
        &mut decoder,
        param1.iPicWidth,
        param1.iPicHeight,
        2,
    );

    assert_eq!(encoder.uninitialize(), CM_RESULT_SUCCESS);
    decoder.uninitialize();
}

#[test]
pub(crate) fn parameter_set_strategy_increasing_id() {
    run_parameter_set_strategy_three_params(EParameterSetStrategy::INCREASING_ID, true);
}

#[test]
pub(crate) fn parameter_set_strategy_sps_listing_and_pps_increasing_1() {
    run_parameter_set_strategy_three_params(
        EParameterSetStrategy::SPS_LISTING_AND_PPS_INCREASING,
        false,
    );
}

/// Port of `EncodeDecodeTestAPI.ParameterSetStrategy_SPS_LISTING_AND_PPS_INCREASING2`
/// (`test/api/encode_options_test.cpp:624-679`).
#[test]
pub(crate) fn parameter_set_strategy_sps_listing_and_pps_increasing_2() {
    let mut encoder = Encoder::new();
    let mut decoder = create_and_init_decoder();

    let mut param1 = SEncParamExt::default();
    prepare_param_default(&encoder, 1, 1, 320, 192, 15.0, &mut param1);
    param1.eSpsPpsIdStrategy = EParameterSetStrategy::SPS_LISTING_AND_PPS_INCREASING;
    param1.iTemporalLayerNum = 1;

    let mut param2 = param1;
    param2.iTemporalLayerNum = 3;

    assert_eq!(encoder.initialize_ext(&param1), CM_RESULT_SUCCESS);
    assert_eq!(encoder.set_encode_param_ext(&param2), CM_RESULT_SUCCESS);
    assert_eq!(encoder.set_encode_param_ext(&param1), CM_RESULT_SUCCESS);
    enc_dec_one_frame(
        &mut encoder,
        &mut decoder,
        param1.iPicWidth,
        param1.iPicHeight,
        0,
    );

    assert_eq!(encoder.force_intra_frame(true, -1), CM_RESULT_SUCCESS);
    enc_dec_one_frame(
        &mut encoder,
        &mut decoder,
        param1.iPicWidth,
        param1.iPicHeight,
        1,
    );

    assert_eq!(encoder.uninitialize(), CM_RESULT_SUCCESS);
    decoder.uninitialize();
}

/// Port of `EncodeDecodeTestAPI.ParameterSetStrategy_SPS_LISTING_AND_PPS_INCREASING3`
/// (`test/api/encode_options_test.cpp:681-735`).
#[test]
pub(crate) fn parameter_set_strategy_sps_listing_and_pps_increasing_3() {
    let mut encoder = Encoder::new();
    let mut decoder = create_and_init_decoder();

    let mut param1 = SEncParamExt::default();
    prepare_param_default(&encoder, 1, 1, 16, 128, 15.0, &mut param1);
    param1.eSpsPpsIdStrategy = EParameterSetStrategy::SPS_LISTING_AND_PPS_INCREASING;
    assert_eq!(encoder.initialize_ext(&param1), CM_RESULT_SUCCESS);

    let mut param2 = param1;
    // 65 distinct macroblock widths (> 2 * MAX_SPS_COUNT)
    for times in 0..65i32 {
        let width = (times + 2) * 16;
        prepare_param_default(&encoder, 1, 1, width, 128, 15.0, &mut param2);
        param2.eSpsPpsIdStrategy = EParameterSetStrategy::SPS_LISTING_AND_PPS_INCREASING;
        assert_eq!(encoder.set_encode_param_ext(&param2), CM_RESULT_SUCCESS);
    }

    enc_dec_one_frame(
        &mut encoder,
        &mut decoder,
        param2.iPicWidth,
        param2.iPicHeight,
        0,
    );

    assert_eq!(encoder.uninitialize(), CM_RESULT_SUCCESS);
    decoder.uninitialize();
}

/// Port of `EncodeDecodeTestAPI.ParameterSetStrategy_SPS_PPS_LISTING1/2/3`
/// (`test/api/encode_options_test.cpp:738-943`).
#[test]
pub(crate) fn parameter_set_strategy_sps_pps_listing_1_2_3() {
    // Listing 1: 1 resolution, encode IDR0, P1, ForceIntraFrame IDR2
    {
        let mut encoder = Encoder::new();
        let mut decoder = create_and_init_decoder();

        let mut param1 = SEncParamExt::default();
        prepare_param_default(&encoder, 1, 1, 320, 192, 15.0, &mut param1);
        param1.eSpsPpsIdStrategy = EParameterSetStrategy::SPS_PPS_LISTING;
        assert_eq!(encoder.initialize_ext(&param1), CM_RESULT_SUCCESS);
        enc_dec_one_frame(
            &mut encoder,
            &mut decoder,
            param1.iPicWidth,
            param1.iPicHeight,
            0,
        );
        enc_dec_one_frame(
            &mut encoder,
            &mut decoder,
            param1.iPicWidth,
            param1.iPicHeight,
            1,
        );
        assert_eq!(encoder.force_intra_frame(true, -1), CM_RESULT_SUCCESS);
        enc_dec_one_frame(
            &mut encoder,
            &mut decoder,
            param1.iPicWidth,
            param1.iPicHeight,
            2,
        );

        assert_eq!(encoder.uninitialize(), CM_RESULT_SUCCESS);
        decoder.uninitialize();
    }

    // Listing 2: 2 resolutions registered before encoding, switch between them
    {
        let mut encoder = Encoder::new();
        let mut decoder = create_and_init_decoder();

        let mut param1 = SEncParamExt::default();
        prepare_param_default(&encoder, 1, 1, 320, 192, 15.0, &mut param1);
        param1.eSpsPpsIdStrategy = EParameterSetStrategy::SPS_PPS_LISTING;

        let mut param2 = param1;
        prepare_param_default(&encoder, 1, 1, 160, 192, 15.0, &mut param2);
        param2.eSpsPpsIdStrategy = EParameterSetStrategy::SPS_PPS_LISTING;

        assert_eq!(encoder.initialize_ext(&param1), CM_RESULT_SUCCESS);
        assert_eq!(encoder.set_encode_param_ext(&param2), CM_RESULT_SUCCESS);
        enc_dec_one_frame(
            &mut encoder,
            &mut decoder,
            param2.iPicWidth,
            param2.iPicHeight,
            0,
        );

        assert_eq!(encoder.set_encode_param_ext(&param1), CM_RESULT_SUCCESS);
        enc_dec_one_frame(
            &mut encoder,
            &mut decoder,
            param1.iPicWidth,
            param1.iPicHeight,
            1,
        );

        assert_eq!(encoder.force_intra_frame(true, -1), CM_RESULT_SUCCESS);
        enc_dec_one_frame(
            &mut encoder,
            &mut decoder,
            param1.iPicWidth,
            param1.iPicHeight,
            2,
        );

        assert_eq!(encoder.uninitialize(), CM_RESULT_SUCCESS);
        decoder.uninitialize();
    }

    // Listing 3: incompatible strategy change and unseen resolution after encoding fail
    {
        let mut encoder = Encoder::new();
        let mut decoder = create_and_init_decoder();

        let mut param1 = SEncParamExt::default();
        prepare_param_default(&encoder, 1, 1, 320, 192, 15.0, &mut param1);
        param1.eSpsPpsIdStrategy = EParameterSetStrategy::SPS_PPS_LISTING;

        let mut param2 = param1;
        prepare_param_default(&encoder, 1, 1, 160, 192, 15.0, &mut param2);
        param2.eSpsPpsIdStrategy = EParameterSetStrategy::SPS_PPS_LISTING;

        let mut param3 = param1;
        prepare_param_default(&encoder, 1, 1, 480, 192, 15.0, &mut param3);
        param3.eSpsPpsIdStrategy = EParameterSetStrategy::SPS_PPS_LISTING;

        assert_eq!(encoder.initialize_ext(&param1), CM_RESULT_SUCCESS);
        assert_eq!(encoder.set_encode_param_ext(&param2), CM_RESULT_SUCCESS);
        enc_dec_one_frame(
            &mut encoder,
            &mut decoder,
            param2.iPicWidth,
            param2.iPicHeight,
            0,
        );

        assert_eq!(
            encoder.set_sps_pps_id_strategy(EParameterSetStrategy::SPS_PPS_LISTING),
            CM_RESULT_SUCCESS
        );

        encoder.set_trace_level(0);
        assert_ne!(
            encoder.set_sps_pps_id_strategy(EParameterSetStrategy::CONSTANT_ID),
            CM_RESULT_SUCCESS
        );

        enc_dec_one_frame(
            &mut encoder,
            &mut decoder,
            param2.iPicWidth,
            param2.iPicHeight,
            1,
        );

        // Switching to unseen resolution param3 after encoding started must fail
        assert_ne!(encoder.set_encode_param_ext(&param3), CM_RESULT_SUCCESS);

        assert_eq!(encoder.uninitialize(), CM_RESULT_SUCCESS);
        decoder.uninitialize();
    }
}

/// Port of `EncodeDecodeTestAPI.SimulcastSVC` (`test/api/encode_options_test.cpp:946-1063`).
#[test]
pub(crate) fn simulcast_svc_roundtrip() {
    let mut encoder = Encoder::new();

    let spatial_layer_num = 3usize;
    let width = 320i32;
    let height = 192i32;
    let mut param = SEncParamExt::default();
    prepare_param_default(
        &encoder,
        spatial_layer_num as i32,
        1,
        width,
        height,
        15.0,
        &mut param,
    );
    assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);

    let mut decoders: Vec<Decoder> = (0..spatial_layer_num)
        .map(|_| create_and_init_decoder())
        .collect();

    let w = width as usize;
    let h = height as usize;
    let mut buf = vec![0u8; w * h * 3 / 2];

    for frame_idx in 0..10usize {
        fill_camera_frame(&mut buf, w, h, frame_idx);
        let pic = SourcePicture::from_i420(&buf, width, height);

        let mut info = SFrameBSInfo::default();
        assert_eq!(encoder.encode_frame(&pic, &mut info), CM_RESULT_SUCCESS);

        let mut layer_bufs = vec![Vec::<u8>::new(); spatial_layer_num];
        for layer_info in info.layers() {
            let bytes = layer_info.bs_buf(&encoder);

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
            let mut dst_buf_info = SBufferInfo::default();
            let hr1 = decoders[idx].decode(Some(bs), &mut dst_buf_info);
            assert_eq!(hr1, DECODING_STATE::dsErrorFree);
            let hr2 = decoders[idx].decode(None, &mut dst_buf_info);
            assert_eq!(hr2, DECODING_STATE::dsErrorFree);
            if dst_buf_info.iBufferStatus == 0 {
                let remaining = decoders[idx].frames_remaining();
                for _ in 0..remaining {
                    decoders[idx].flush(&mut dst_buf_info);
                }
            }
            assert_eq!(dst_buf_info.iBufferStatus, 1, "layer={idx}");
        }
    }

    for mut dec in decoders {
        dec.uninitialize();
    }
    encoder.uninitialize();
}

/// Port of `EncodeDecodeTestAPI.SimulcastAVC` and `SimulcastAVC_SPS_PPS_LISTING`
/// (`test/api/encode_options_test.cpp:1065-1241`).
#[test]
pub(crate) fn simulcast_avc_and_sps_pps_listing_roundtrip() {
    let mut encoder = Encoder::new();

    let spatial_layer_num = 3usize;
    let width = 320i32;
    let height = 192i32;

    let mut param1 = SEncParamExt::default();
    prepare_param_default(
        &encoder,
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

    assert_eq!(encoder.initialize_ext(&param1), CM_RESULT_SUCCESS);
    assert_eq!(encoder.set_encode_param_ext(&param2), CM_RESULT_SUCCESS);

    let mut decoders: Vec<Decoder> = (0..spatial_layer_num)
        .map(|_| create_and_init_decoder())
        .collect();

    test_one_simulcast_avc(&mut encoder, &param1, &mut decoders, spatial_layer_num, 6);
    test_one_simulcast_avc(&mut encoder, &param2, &mut decoders, spatial_layer_num, 6);

    for mut dec in decoders {
        dec.uninitialize();
    }
    encoder.uninitialize();
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
    let mut encoder = Encoder::new();
    let mut decoder = create_and_init_decoder();

    let mut param = SEncParamExt::default();
    encoder.default_params(&mut param);
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

        encoder.set_trace_level(0);
        decoder.set_trace_level(0);

        encoder.uninitialize();
        assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);

        encoder.set_sps_pps_id_strategy(EParameterSetStrategy::INCREASING_ID);
        encoder.set_idr_interval(6);

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

            let pic = SourcePicture::from_i420(&buf, width, height);

            let mut info = SFrameBSInfo::default();
            assert_eq!(encoder.encode_frame(&pic, &mut info), CM_RESULT_SUCCESS);

            let bs = collect_frame_bitstream(&encoder, &info);
            if !bs.is_empty() && test_decoder {
                let mut dst_buf_info = SBufferInfo::default();
                let dec_rv = decoder.decode_no_delay(Some(&bs), &mut dst_buf_info);
                assert_eq!(dec_rv, DECODING_STATE::dsErrorFree);
                assert_eq!(dst_buf_info.iBufferStatus, 1);
            }

            if sm == SliceModeEnum::SM_SIZELIMITED_SLICE && test_nal_size {
                for layer in info.layers() {
                    let nal_lens = layer.nal_lengths(&encoder);
                    let max_idx = (layer.iNalCount - 2).max(0) as usize;
                    for &nal_len in nal_lens.iter().take(max_idx + 1) {
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

    decoder.uninitialize();
    encoder.uninitialize();
}

#[test]
pub(crate) fn set_enc_option_size_sizelimited_decode() {
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
pub(crate) fn set_enc_option_size_multithread_fixed_slice() {
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
pub(crate) fn set_enc_option_size_annex_a_extreme_dimensions() {
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
pub(crate) fn simulcast_avc_diff_fps() {
    const TEMPORAL_PATTERNS: [[i32; 4]; 6] = [
        [2, 1, 1, 1],
        [4, 2, 1, 1],
        [1, 4, 2, 1],
        [1, 2, 4, 1],
        [1, 2, 2, 4],
        [1, 4, 4, 4],
    ];

    let mut encoder = Encoder::new();

    let spatial_layer_num = 3usize;
    let width = 320i32;
    let height = 192i32;
    let frame_rate = 30.0f32;

    let mut param = SEncParamExt::default();
    prepare_param_default(
        &encoder,
        spatial_layer_num as i32,
        1,
        width,
        height,
        frame_rate,
        &mut param,
    );
    param.bSimulcastAVC = true;
    param.iTemporalLayerNum = 3;

    let mut decoders: Vec<Decoder> = (0..spatial_layer_num)
        .map(|_| create_and_init_decoder())
        .collect();

    let w = width as usize;
    let h = height as usize;
    let mut buf = vec![0u8; w * h * 3 / 2];

    for (pattern_idx, pattern) in TEMPORAL_PATTERNS.iter().enumerate() {
        for i in 0..spatial_layer_num {
            param.sSpatialLayers[i].fFrameRate = frame_rate / (pattern[i] as f32);
        }
        assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);

        for frame_idx in 0..8usize {
            fill_camera_frame(&mut buf, w, h, frame_idx);
            let pic = SourcePicture::from_i420(&buf, width, height);

            let mut info = SFrameBSInfo::default();
            assert_eq!(encoder.encode_frame(&pic, &mut info), CM_RESULT_SUCCESS);

            if frame_idx == 4 {
                encoder.force_intra_frame(true, -1);
            }

            let mut layer_bufs = vec![Vec::<u8>::new(); spatial_layer_num];
            for layer_info in info.layers() {
                let bytes = layer_info.bs_buf(&encoder);
                let first_nal_type = bytes[4] & 0x1f;
                assert!(
                    matches!(first_nal_type, 1 | 5 | 6 | 7 | 8),
                    "unexpected first_nal_type={first_nal_type}"
                );
                let idx = layer_info.uiSpatialId as usize;
                assert!(idx < spatial_layer_num);
                layer_bufs[idx].extend_from_slice(bytes);
            }

            for (idx, bs) in layer_bufs.iter().enumerate() {
                if !bs.is_empty() {
                    let mut dst_buf_info = SBufferInfo::default();
                    let hr1 = decoders[idx].decode(Some(bs), &mut dst_buf_info);
                    assert_eq!(
                        hr1,
                        DECODING_STATE::dsErrorFree,
                        "pattern={pattern_idx} layer={idx}"
                    );
                    let hr2 = decoders[idx].decode(None, &mut dst_buf_info);
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

    for mut dec in decoders {
        dec.uninitialize();
    }
    encoder.uninitialize();
}

/// Port of `EncodeDecodeTestAPI.DiffSlicingInDlayer` and `DiffSlicingInDlayerMixed`
/// (`test/api/encode_options_test.cpp:1571-1712`).
#[test]
pub(crate) fn diff_slicing_in_dlayer_and_mixed() {
    // Part 1: DiffSlicingInDlayer (3 layers: SM_RASTER_SLICE [0], SM_RASTER_SLICE [30, 32], SM_FIXEDSLCNUM_SLICE)
    {
        let mut encoder = Encoder::new();

        let spatial_layer_num = 3usize;
        let width = 512i32;
        let height = 288i32;
        let mut param = SEncParamExt::default();
        prepare_param_default(
            &encoder,
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
        param.sSpatialLayers[2].sSliceArgument.uiSliceMode = SliceModeEnum::SM_FIXEDSLCNUM_SLICE;
        param.sSpatialLayers[2].sSliceArgument.uiSliceNum = 4;

        assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);

        let mut decoders: Vec<Decoder> = (0..spatial_layer_num)
            .map(|_| create_and_init_decoder())
            .collect();

        test_one_simulcast_avc(&mut encoder, &param, &mut decoders, spatial_layer_num, 6);

        for mut dec in decoders {
            dec.uninitialize();
        }
        encoder.uninitialize();
    }

    // Part 2: DiffSlicingInDlayerMixed (2 layers mixing SM_SIZELIMITED_SLICE and SM_FIXEDSLCNUM_SLICE)
    {
        let mut encoder = Encoder::new();

        let spatial_layer_num = 2usize;
        let width = 512i32;
        let height = 288i32;
        let mut param = SEncParamExt::default();
        prepare_param_default(
            &encoder,
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
        param.sSpatialLayers[0].sSliceArgument.uiSliceMode = SliceModeEnum::SM_SIZELIMITED_SLICE;
        param.sSpatialLayers[0].sSliceArgument.uiSliceSizeConstraint = 1500;

        param.sSpatialLayers[1].iVideoWidth = width;
        param.sSpatialLayers[1].iVideoHeight = height;
        param.sSpatialLayers[1].sSliceArgument.uiSliceMode = SliceModeEnum::SM_FIXEDSLCNUM_SLICE;
        param.sSpatialLayers[1].sSliceArgument.uiSliceNum = 2;
        param.sSpatialLayers[1].sSliceArgument.uiSliceSizeConstraint = 1500;

        assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);

        let mut decoders: Vec<Decoder> = (0..spatial_layer_num)
            .map(|_| create_and_init_decoder())
            .collect();

        test_one_simulcast_avc(&mut encoder, &param, &mut decoders, spatial_layer_num, 6);

        for mut dec in decoders {
            dec.uninitialize();
        }
        encoder.uninitialize();
    }
}

/// Port of `EncodeDecodeTestAPI.ThreadNumAndSliceNum` and `TriggerLoadBalancing`
/// (`test/api/encode_options_test.cpp:1714-1894`).
#[test]
pub(crate) fn thread_num_and_slice_num_and_load_balancing() {
    // 1. ThreadNumAndSliceNum (threads=3, slices=4 and slices=2)
    for slice_num in [2u32, 4u32] {
        let mut encoder = Encoder::new();
        let mut decoder = create_and_init_decoder();

        let mut param = SEncParamExt::default();
        prepare_param_default(&encoder, 1, 1, 320, 192, 15.0, &mut param);
        param.iMultipleThreadIdc = 3;
        param.bSimulcastAVC = true;
        param.sSpatialLayers[0].sSliceArgument.uiSliceMode = SliceModeEnum::SM_FIXEDSLCNUM_SLICE;
        param.sSpatialLayers[0].sSliceArgument.uiSliceNum = slice_num;

        assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);
        test_one_simulcast_avc(
            &mut encoder,
            &param,
            std::slice::from_mut(&mut decoder),
            1,
            6,
        );

        decoder.uninitialize();
        encoder.uninitialize();
    }

    // 2. TriggerLoadBalancing (4 threads, 4 slices, complex strip shifts from top to bottom)
    {
        let mut encoder = Encoder::new();
        let mut decoder = create_and_init_decoder();

        let width = 320i32;
        let height = 192i32;
        let mut param = SEncParamExt::default();
        prepare_param_default(&encoder, 1, 1, width, height, 15.0, &mut param);
        param.iMultipleThreadIdc = 4;
        param.bSimulcastAVC = true;
        param.sSpatialLayers[0].sSliceArgument.uiSliceMode = SliceModeEnum::SM_FIXEDSLCNUM_SLICE;
        param.sSpatialLayers[0].sSliceArgument.uiSliceNum = 4;
        param.sSpatialLayers[0].sSliceArgument.uiSliceSizeConstraint = 1000;

        assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);
        assert_eq!(encoder.set_encode_param_ext(&param), CM_RESULT_SUCCESS);

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

            let pic = SourcePicture::from_i420(&buf, width, height);

            let mut info = SFrameBSInfo::default();
            assert_eq!(encoder.encode_frame(&pic, &mut info), CM_RESULT_SUCCESS);

            let bs = collect_frame_bitstream(&encoder, &info);
            let mut dst_buf_info = SBufferInfo::default();
            assert_eq!(
                decoder.decode(Some(&bs), &mut dst_buf_info),
                DECODING_STATE::dsErrorFree
            );
            assert_eq!(
                decoder.decode(None, &mut dst_buf_info),
                DECODING_STATE::dsErrorFree
            );
            assert_eq!(dst_buf_info.iBufferStatus, 1);
        }

        decoder.uninitialize();
        encoder.uninitialize();
    }
}

/// Port of `EncodeDecodeTestAPI.ProfileLevelSetting` (`test/api/encode_options_test.cpp:1931-2077`).
#[test]
pub(crate) fn profile_level_setting_roundtrip() {
    let cases = [
        (EProfileIdc::PRO_BASELINE, ELevelIdc::LEVEL_3_1, 0),
        (EProfileIdc::PRO_MAIN, ELevelIdc::LEVEL_3_2, 1),
        (EProfileIdc::PRO_HIGH, ELevelIdc::LEVEL_4_0, 1),
        // Unsupported profile (PRO_EXTENDED) falls back to PRO_BASELINE (CAVLC) or PRO_HIGH (CABAC)
        (EProfileIdc::PRO_EXTENDED, ELevelIdc::LEVEL_3_0, 0),
        (EProfileIdc::PRO_EXTENDED, ELevelIdc::LEVEL_3_0, 1),
    ];

    for (enc_prof, enc_lvl, cabac) in cases {
        let mut encoder = Encoder::new();
        let mut decoder = create_and_init_decoder();

        let mut param = SEncParamExt::default();
        prepare_param_default(&encoder, 1, 1, 320, 192, 15.0, &mut param);
        param.bSimulcastAVC = true;
        param.iRCMode = RC_MODES::RC_TIMESTAMP_MODE;
        param.iNumRefFrame = 1;
        param.iEntropyCodingModeFlag = cabac;
        param.iTargetBitrate = 3000;
        param.sSpatialLayers[0].iSpatialBitrate = 3000;
        param.sSpatialLayers[0].uiProfileIdc = enc_prof;
        param.sSpatialLayers[0].uiLevelIdc = enc_lvl;

        assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);
        enc_dec_one_frame(&mut encoder, &mut decoder, 320, 192, 0);

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
        assert_eq!(decoder.active_sps_profile(), Some(Some(expected_prof)));
        assert_eq!(decoder.active_sps_level(), Some(Some(enc_lvl as i32)));

        // Reinitialize with new profile/level and exercise SetOption/GetOption + ForceIntraFrame
        // (matching test/api/encode_options_test.cpp:2012-2076)
        assert_eq!(encoder.uninitialize(), CM_RESULT_SUCCESS);

        param.sSpatialLayers[0].uiProfileIdc = EProfileIdc::PRO_MAIN;
        param.sSpatialLayers[0].uiLevelIdc = ELevelIdc::LEVEL_4_1;
        assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);

        assert_eq!(
            encoder.set_profile(&SProfileInfo {
                iLayer: 0,
                uiProfileIdc: EProfileIdc::PRO_MAIN,
            }),
            CM_RESULT_SUCCESS
        );
        assert_eq!(
            encoder.set_level(&SLevelInfo {
                iLayer: 0,
                uiLevelIdc: ELevelIdc::LEVEL_4_1,
            }),
            CM_RESULT_SUCCESS
        );
        let ext = encoder.encode_param_ext().unwrap();
        assert_eq!(ext.sSpatialLayers[0].uiProfileIdc, EProfileIdc::PRO_MAIN);
        assert_eq!(ext.sSpatialLayers[0].uiLevelIdc, ELevelIdc::LEVEL_4_1);

        assert_eq!(encoder.force_intra_frame(true, -1), CM_RESULT_SUCCESS);
        enc_dec_one_frame(&mut encoder, &mut decoder, 320, 192, 1);

        assert_eq!(
            decoder.active_sps_profile(),
            Some(Some(EProfileIdc::PRO_MAIN as i32))
        );
        assert_eq!(
            decoder.active_sps_level(),
            Some(Some(ELevelIdc::LEVEL_4_1 as i32))
        );

        decoder.uninitialize();
        encoder.uninitialize();
    }
}

/// Port of `EncodeDecodeTestAPI.AVCSVCExtensionCheck`
/// (`test/api/encode_options_test.cpp:2079-2135`).
#[test]
pub(crate) fn avc_svc_extension_check() {
    for &(spatial_layers, simulcast_avc) in &[(1, false), (1, true), (3, false), (3, true)] {
        let mut encoder = Encoder::new();

        let width = 640i32;
        let height = 360i32;
        let mut param = SEncParamExt::default();
        prepare_param_default(&encoder, spatial_layers, 1, width, height, 15.0, &mut param);
        param.bSimulcastAVC = simulcast_avc;
        param.iRCMode = RC_MODES::RC_TIMESTAMP_MODE;
        param.iNumRefFrame = 1;
        param.sSpatialLayers[0].iSpatialBitrate = 500;
        param.sSpatialLayers[1].iSpatialBitrate = 1000;
        param.sSpatialLayers[2].iSpatialBitrate = 2200;
        param.iTargetBitrate = 4000;

        assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);

        let w = width as usize;
        let h = height as usize;
        let mut buf = vec![0u8; w * h * 3 / 2];

        for frame_idx in 0..6usize {
            fill_camera_frame(&mut buf, w, h, frame_idx);
            let pic = SourcePicture::from_i420(&buf, width, height);

            let mut info = SFrameBSInfo::default();
            assert_eq!(encoder.encode_frame(&pic, &mut info), CM_RESULT_SUCCESS);

            for layer in info.layers() {
                for nal in layer.nals(&encoder) {
                    let type_byte = nal[4] & 0x1f;
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
                                || (layer.uiLayerType == LAYER_TYPE::NON_VIDEO_CODING_LAYER as u8
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

        assert_eq!(encoder.uninitialize(), CM_RESULT_SUCCESS);
    }
}

/// Port of `EncodeDecodeTestAPI.UnsupportedVideoSizeInput` and
/// `ScreenContent_LosslessLink0_EnableLongTermReference`
/// (`test/api/encode_options_test.cpp:2137-2208`).
#[test]
pub(crate) fn unsupported_video_size_input_and_screen_ltr_non_lossless() {
    // 1. UnsupportedVideoSizeInput
    {
        let mut encoder = Encoder::new();

        let mut param = SEncParamExt::default();
        prepare_param_default(&encoder, 1, 1, 640, 360, 15.0, &mut param);
        param.iRCMode = RC_MODES::RC_OFF_MODE;
        param.iNumRefFrame = 1;
        assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);

        let small_buf = vec![128u8; 8 * 8 * 3 / 2];
        let small_pic = SourcePicture::from_i420(&small_buf, 8, 8);

        let mut info = SFrameBSInfo::default();
        assert_eq!(
            encoder.encode_frame(&small_pic, &mut info),
            CM_UNSUPPORTED_DATA
        );

        let valid_buf = vec![128u8; 640 * 360 * 3 / 2];
        let valid_pic = SourcePicture::from_i420(&valid_buf, 640, 360);

        assert_eq!(
            encoder.encode_frame(&valid_pic, &mut info),
            CM_RESULT_SUCCESS
        );
        assert_eq!(encoder.uninitialize(), CM_RESULT_SUCCESS);
    }

    // 2. ScreenContent_LosslessLink0_EnableLongTermReference (2882x1808)
    {
        let mut encoder = Encoder::new();

        let width = 2882i32;
        let height = 1808i32;
        let mut param = SEncParamExt::default();
        prepare_param_default(&encoder, 1, 1, width, height, 15.0, &mut param);
        param.iUsageType = EUsageType::SCREEN_CONTENT_REAL_TIME;
        param.bEnableLongTermReference = true;
        param.bIsLosslessLink = false;

        assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);

        let w = width as usize;
        let h = height as usize;
        let mut buf = vec![0u8; w * h * 3 / 2];
        fill_screen_frame(&mut buf, w, h, 0);

        let pic = SourcePicture::from_i420(&buf, width, height);

        let mut info = SFrameBSInfo::default();
        assert_eq!(encoder.encode_frame(&pic, &mut info), CM_RESULT_SUCCESS);
        assert_eq!(encoder.uninitialize(), CM_RESULT_SUCCESS);
    }
}

/// Port of `EncodeDecodeTestAPI.TemporalLayerChangeDuringEncoding` and
/// `TemporalLayerChangeDuringEncoding_Specific` (`test/api/encode_options_test.cpp:2211-2344`).
#[test]
pub(crate) fn temporal_layer_change_during_encoding() {
    // 1. TemporalLayerChangeDuringEncoding (2 -> 3 -> 1 -> 2)
    {
        let mut encoder = Encoder::new();

        let width = 320i32;
        let height = 192i32;
        let mut param = SEncParamExt::default();
        prepare_param_default(&encoder, 1, 1, width, height, 15.0, &mut param);
        param.iTemporalLayerNum = 2;

        assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);

        let w = width as usize;
        let h = height as usize;
        let mut buf = vec![0u8; w * h * 3 / 2];

        for frame_num in 0..20usize {
            fill_camera_frame(&mut buf, w, h, frame_num);
            let pic = SourcePicture::from_i420(&buf, width, height);

            let mut info = SFrameBSInfo::default();
            assert_eq!(encoder.encode_frame(&pic, &mut info), CM_RESULT_SUCCESS);

            if frame_num == 5 {
                param.iTemporalLayerNum = 3;
                param.iTargetBitrate = 1_500_000;
                param.sSpatialLayers[0].iSpatialBitrate = 1_500_000;
                param.fMaxFrameRate = 30.0;
                assert_eq!(encoder.set_encode_param_ext(&param), CM_RESULT_SUCCESS);
            } else if frame_num == 10 {
                param.iTemporalLayerNum = 1;
                param.iTargetBitrate = 500_000;
                param.sSpatialLayers[0].iSpatialBitrate = 500_000;
                param.fMaxFrameRate = 7.5;
                assert_eq!(encoder.set_encode_param_ext(&param), CM_RESULT_SUCCESS);
            } else if frame_num == 15 {
                param.iTemporalLayerNum = 2;
                param.iTargetBitrate = 1_000_000;
                param.sSpatialLayers[0].iSpatialBitrate = 1_000_000;
                param.fMaxFrameRate = 15.0;
                assert_eq!(encoder.set_encode_param_ext(&param), CM_RESULT_SUCCESS);
            }
        }

        assert_eq!(encoder.uninitialize(), CM_RESULT_SUCCESS);
    }

    // 2. TemporalLayerChangeDuringEncoding_Specific (steps [2, 1, 3] with iNumRefFrame = 1)
    {
        let mut encoder = Encoder::new();

        let width = 320i32;
        let height = 192i32;
        let mut param = SEncParamExt::default();
        prepare_param_default(&encoder, 1, 1, width, height, 15.0, &mut param);
        param.iTemporalLayerNum = 1;
        param.iNumRefFrame = 1;

        assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);

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
                assert_eq!(encoder.set_encode_param_ext(&param), CM_RESULT_SUCCESS);
                set_opt = true;
                step_idx += 1;
            }

            fill_camera_frame(&mut buf, w, h, frame_num);
            let pic = SourcePicture::from_i420(&buf, width, height);

            let mut info = SFrameBSInfo::default();
            assert_eq!(encoder.encode_frame(&pic, &mut info), CM_RESULT_SUCCESS);

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

        assert_eq!(encoder.uninitialize(), CM_RESULT_SUCCESS);
    }
}

/// Port of `EncodeDecodeTestAPI.ENCODER_OPTION_IDR_INTERVAL`
/// (`test/api/encode_options_test.cpp:2346-2390`).
#[test]
pub(crate) fn encoder_option_idr_interval_schedule() {
    let mut encoder = Encoder::new();

    let width = 320i32;
    let height = 192i32;
    let mut param = SEncParamExt::default();
    prepare_param_default(&encoder, 1, 1, width, height, 15.0, &mut param);
    param.iTemporalLayerNum = 1;

    encoder.set_trace_level(0);
    assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);

    let w = width as usize;
    let h = height as usize;
    let mut buf = vec![0u8; w * h * 3 / 2];
    fill_camera_frame(&mut buf, w, h, 0);

    let pic = SourcePicture::from_i420(&buf, width, height);

    let mut info = SFrameBSInfo::default();
    assert_eq!(encoder.encode_frame(&pic, &mut info), CM_RESULT_SUCCESS);
    assert_eq!(info.eFrameType, EVideoFrameType::videoFrameTypeIDR);

    let mut last_idr_idx = 0i32;
    let mut frame = 1i32;

    for &target_intra_period in &[4i32, 1, 6, 0, -1, 5] {
        assert_eq!(
            encoder.set_idr_interval(target_intra_period),
            CM_RESULT_SUCCESS
        );

        let enc_frame_num = (target_intra_period * 3).max(0) as usize;
        for i in 0..enc_frame_num {
            fill_camera_frame(&mut buf, w, h, frame as usize);
            let pic = SourcePicture::from_i420(&buf, width, height);
            assert_eq!(encoder.encode_frame(&pic, &mut info), CM_RESULT_SUCCESS);

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

    assert_eq!(encoder.uninitialize(), CM_RESULT_SUCCESS);
}
