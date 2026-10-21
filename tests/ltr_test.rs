// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Integration tests for Long-Term Reference (LTR) marking, recovery requests,
//! feedback, and closed-loop encoder-decoder NAL loss recovery.
//!
//! Ported from upstream `test/api/ltr_test.cpp`, `test/api/encode_decode_api_test.cpp`,
//! and the LTR/SVC switch tests in `test/api/decoder_ec_test.cpp`.

#[path = "common/mod.rs"]
mod common;

use common::frame_gen::{fill_camera_frame, xorshift32};
use openh264rs::api::codec_api::*;
use openh264rs::encoder::ref_list_mgr_svc::{
    IDR_RECOVERY_REQUEST, LTR_MARKING_FAILED, LTR_MARKING_SUCCESS, LTR_RECOVERY_REQUEST,
    NO_RECOVERY_REQUSET,
};

#[derive(Clone, Copy)]
struct LtrTestParam {
    numframes: usize,
    width: i32,
    height: i32,
    frame_rate: f32,
    slicenum: u32,
    lost_para: bool,
    loss_sequence: &'static [u8],
}

/// Upstream `kFileParamArray` from `test/api/decoder_ec_test.cpp:137-141` (100 frames per case
/// to exercise full 60-frame IDR intervals and multiple LTR marking/recovery cycles).
const K_FILE_PARAM_ARRAY: [LtrTestParam; 3] = [
    LtrTestParam {
        numframes: 100,
        width: 160,
        height: 96,
        frame_rate: 6.0,
        slicenum: 2,
        lost_para: true,
        loss_sequence: b"000000000000001010101010101010101010101001101010100000010101000011",
    },
    LtrTestParam {
        numframes: 100,
        width: 140,
        height: 96,
        frame_rate: 6.0,
        slicenum: 4,
        lost_para: true,
        loss_sequence: b"000000000000001010101010101010101010101001101010100000010101000011",
    },
    LtrTestParam {
        numframes: 100,
        width: 140,
        height: 96,
        frame_rate: 6.0,
        slicenum: 4,
        lost_para: true,
        loss_sequence: b"000000000000001010111010101011101010101001101010100000010101110011010101",
    },
];

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

fn is_param_sets_nal(nal_type: u8) -> bool {
    // NAL_UNIT_SPS (7), NAL_UNIT_PPS (8), NAL_UNIT_SUBSET_SPS (15)
    nal_type == 7 || nal_type == 8 || nal_type == 15
}

fn is_vcl_nal(nal_type: u8) -> bool {
    // NAL_UNIT_CODED_SLICE (1) ..= NAL_UNIT_CODED_SLICE_IDR (5), NAL_UNIT_PREFIX (14), NAL_UNIT_CODED_SLICE_EXT (20)
    (1..=5).contains(&nal_type) || nal_type == 14 || nal_type == 20
}

fn is_new_introduced_svc_nal(nal_type: u8) -> bool {
    // NAL_UNIT_PREFIX (14), NAL_UNIT_CODED_SLICE_EXT (20)
    nal_type == 14 || nal_type == 20
}

fn is_vcl_nal_avc_base(nal_type: u8) -> bool {
    // NAL_UNIT_CODED_SLICE (1) ..= NAL_UNIT_CODED_SLICE_IDR (5)
    (1..=5).contains(&nal_type)
}

/// Port of `SimulateNALLoss` from `test/api/encode_decode_api_test.cpp:316-392`.
fn simulate_nal_loss(
    src: &[u8],
    loss_chars: &[u8],
    loss_para: bool,
    loss_idx: &mut usize,
    vcl_loss: &mut bool,
    rng: &mut u32,
) -> (Vec<u8>, usize) {
    let mut dst = Vec::with_capacity(src.len());
    let mut skipped_bytes = 0usize;
    let mut buf_pos = 0usize;
    let mut last_prefix_len = 0usize;
    let mut i = 0usize;
    *vcl_loss = false;

    let mut process_nal = |nal_start: usize, nal_end: usize, prefix_len: usize| {
        if nal_end > nal_start {
            let nal_type = src[nal_start + prefix_len] & 0x1f;
            let mut lost = if *loss_idx < loss_chars.len() {
                loss_chars[*loss_idx] == b'1'
            } else {
                (xorshift32(rng) & 1) == 1
            };
            if !loss_para && is_param_sets_nal(nal_type) {
                lost = false;
            }
            *loss_idx += 1;
            if !lost {
                dst.extend_from_slice(&src[nal_start..nal_end]);
            } else {
                if is_vcl_nal(nal_type) {
                    *vcl_loss = true;
                }
                skipped_bytes += nal_end - nal_start;
            }
        }
    };

    while i < src.len() {
        if i + 4 <= src.len()
            && src[i] == 0
            && src[i + 1] == 0
            && src[i + 2] == 0
            && src[i + 3] == 1
        {
            process_nal(buf_pos, i, last_prefix_len);
            last_prefix_len = 4;
            buf_pos = i;
            i += 4;
        } else if i + 3 <= src.len() && src[i] == 0 && src[i + 1] == 0 && src[i + 2] == 1 {
            process_nal(buf_pos, i, last_prefix_len);
            last_prefix_len = 3;
            buf_pos = i;
            i += 3;
        } else {
            i += 1;
        }
    }
    process_nal(buf_pos, src.len(), last_prefix_len);
    (dst, skipped_bytes)
}

/// Port of `ToRemainDidNal` + `ExtractDidNal` from `test/api/encode_decode_api_test.cpp:269-314`.
fn extract_did_nal(enc: &Encoder, info: &SFrameBSInfo, tar_did: i32) -> Vec<u8> {
    let mut dst = Vec::new();
    for layer in info.layers() {
        for nal in layer.nals(enc) {
            let prefix = if nal.len() >= 4 && nal[0..4] == [0, 0, 0, 1] {
                4
            } else if nal.len() >= 3 && nal[0..3] == [0, 0, 1] {
                3
            } else {
                0
            };
            if nal.len() > prefix + 2 {
                let nal_type = nal[prefix] & 0x1f;
                let cur_byte = nal[prefix + 2];
                let remain = if is_new_introduced_svc_nal(nal_type) {
                    let did = ((cur_byte & 0x70) >> 4) as i32;
                    did == tar_did
                } else {
                    !(is_vcl_nal_avc_base(nal_type) && tar_did != 0)
                };
                if remain {
                    dst.extend_from_slice(nal);
                }
            }
        }
    }
    dst
}

/// Port of `IsKeyFrameLost` from `test/api/encode_decode_api_test.cpp:203-231`.
fn is_key_frame_lost(
    decoder: &Decoder,
    ltr_recover_request: &mut SLTRRecoverRequest,
    hr: DECODING_STATE,
) -> u32 {
    let temporal_id = decoder.feedback_temporal_id().unwrap_or(-1);
    let p2p_mode_flag = true;
    if hr == DECODING_STATE::dsErrorFree {
        if p2p_mode_flag && temporal_id == 0 {
            let idr_pic_id = decoder.cur_idr_pic_id().unwrap_or(-1);
            if ltr_recover_request.uiIDRPicId != idr_pic_id as u32 {
                ltr_recover_request.uiIDRPicId = idr_pic_id as u32;
                ltr_recover_request.iLastCorrectFrameNum = -1;
            }
            let frame_num = decoder.frame_num().unwrap_or(-1);
            if frame_num >= 0 {
                ltr_recover_request.iLastCorrectFrameNum = frame_num;
            }
        }
        NO_RECOVERY_REQUSET
    } else if (hr.0 & DECODING_STATE::dsNoParamSets.0) != 0 {
        IDR_RECOVERY_REQUEST
    } else if (((hr.0 & DECODING_STATE::dsRefLost.0) != 0) && temporal_id == 1)
        || (hr != DECODING_STATE::dsErrorFree && temporal_id == 0)
    {
        LTR_RECOVERY_REQUEST
    } else {
        NO_RECOVERY_REQUSET
    }
}

/// Port of `IsLTRMarking` from `test/api/encode_decode_api_test.cpp:233-237`.
fn is_ltr_marking(decoder: &Decoder) -> bool {
    decoder.ltr_marking_flag().unwrap_or(0) != 0
}

/// Port of `LTRRecoveryRequest` from `test/api/encode_decode_api_test.cpp:239-257`.
fn ltr_recovery_request(
    decoder: &Decoder,
    encoder: &mut Encoder,
    req: &mut SLTRRecoverRequest,
    hr: DECODING_STATE,
    p2p_mode_flag: bool,
) {
    let klost = is_key_frame_lost(decoder, req, hr);
    if p2p_mode_flag {
        if klost == IDR_RECOVERY_REQUEST {
            req.uiFeedbackType = IDR_RECOVERY_REQUEST;
            encoder.force_intra_frame(true, -1);
        } else if klost == LTR_RECOVERY_REQUEST {
            req.uiFeedbackType = LTR_RECOVERY_REQUEST;
            if let Some(frame_num) = decoder.frame_num() {
                req.iCurrentFrameNum = frame_num;
            }
            if let Some(idr_pic_id) = decoder.cur_idr_pic_id() {
                req.uiIDRPicId = idr_pic_id as u32;
            }
            encoder.set_ltr_recovery_request(req);
        }
    } else if klost == IDR_RECOVERY_REQUEST || klost == LTR_RECOVERY_REQUEST {
        req.uiFeedbackType = IDR_RECOVERY_REQUEST;
        encoder.force_intra_frame(true, -1);
    }
}

/// Port of `LTRMarkFeedback` from `test/api/encode_decode_api_test.cpp:259-267`.
fn ltr_mark_feedback(
    decoder: &Decoder,
    encoder: &mut Encoder,
    fb: &mut SLTRMarkingFeedback,
    hr: DECODING_STATE,
) {
    if is_ltr_marking(decoder) {
        fb.uiFeedbackType = if hr == DECODING_STATE::dsErrorFree {
            LTR_MARKING_SUCCESS
        } else {
            LTR_MARKING_FAILED
        };
        if let Some(idr_pic_id) = decoder.cur_idr_pic_id() {
            fb.uiIDRPicId = idr_pic_id as u32;
        }
        if let Some(ltr_frame_num) = decoder.ltr_marked_frame_num() {
            fb.iLTRFrameNum = ltr_frame_num;
        }
        encoder.set_ltr_marking_feedback(fb);
    }
}

#[test]
pub(crate) fn test_ltr_encoder_configuration() {
    let mut enc = Encoder::new();

    let mut param = SEncParamExt::default();
    param.iPicWidth = 320;
    param.iPicHeight = 240;
    param.fMaxFrameRate = 30.0;
    param.iTargetBitrate = 500000;
    param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
    param.bEnableLongTermReference = true;
    param.iLtrMarkPeriod = 30;

    let init_ret = enc.initialize_ext(&param);
    assert_eq!(init_ret, CM_RESULT_SUCCESS);

    let uninit_ret = enc.uninitialize();
    assert_eq!(uninit_ret, CM_RESULT_SUCCESS);
}

/// Port of `EncodeDecodeTestAPI.GetOptionLTR_ALLLTR` (`test/api/ltr_test.cpp:45-104`).
fn run_get_option_ltr_all_ltr(p: LtrTestParam, seed: u32) {
    let mut encoder = Encoder::new();

    let mut param = SEncParamExt::default();
    prepare_param_default(
        &encoder,
        1,
        p.slicenum,
        p.width,
        p.height,
        p.frame_rate,
        &mut param,
    );
    assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);

    encoder.set_trace_level(0); // WELS_LOG_QUIET
    encoder.set_sps_pps_id_strategy(EParameterSetStrategy::INCREASING_ID);
    encoder.set_idr_interval(60);
    encoder.set_ltr_config(&SLTRConfig {
        bEnableLongTermReference: true,
        iLTRRefNum: 1,
    });
    encoder.set_ltr_marking_period(2);

    let w = p.width as usize;
    let h = p.height as usize;
    let frame_size = w * h * 3 / 2;
    let mut buf = vec![0u8; frame_size];

    let mut ltr_recover_request = SLTRRecoverRequest {
        uiFeedbackType: NO_RECOVERY_REQUSET,
        uiIDRPicId: 0,
        iLastCorrectFrameNum: 0,
        iCurrentFrameNum: 0,
        iLayerId: 0,
    };
    let mut ltr_marking_feedback = SLTRMarkingFeedback {
        uiFeedbackType: 0,
        uiIDRPicId: 0,
        iLTRFrameNum: 0,
        iLayerId: 0,
    };

    let mut rng = seed;
    let signed_rand_10000 = |rng: &mut u32| -> i32 {
        let neg = (xorshift32(rng) & 1) == 1;
        let mag = (xorshift32(rng) % 10000) as i32;
        if neg { -mag } else { mag }
    };

    for idx in 0..=p.numframes {
        fill_camera_frame(&mut buf, w, h, idx);
        let pic = SourcePicture::from_i420(&buf, p.width, p.height);
        let mut info = SFrameBSInfo::default();
        let rv = encoder.encode_frame(&pic, &mut info);
        if ltr_recover_request.uiFeedbackType == IDR_RECOVERY_REQUEST {
            assert_eq!(info.eFrameType, EVideoFrameType::videoFrameTypeIDR);
        }
        assert!(
            rv == CM_RESULT_SUCCESS || rv == CM_UNKNOWN_REASON,
            "unexpected EncodeFrame rv={rv} at frame {idx}"
        );

        ltr_recover_request.uiFeedbackType = LTR_RECOVERY_REQUEST;
        ltr_recover_request.iCurrentFrameNum = signed_rand_10000(&mut rng);
        ltr_recover_request.uiIDRPicId = signed_rand_10000(&mut rng) as u32;
        ltr_recover_request.iLastCorrectFrameNum = signed_rand_10000(&mut rng);
        encoder.set_ltr_recovery_request(&mut ltr_recover_request);

        ltr_marking_feedback.uiFeedbackType = if (xorshift32(&mut rng) & 1) == 1 {
            LTR_MARKING_SUCCESS
        } else {
            LTR_MARKING_FAILED
        };
        ltr_marking_feedback.uiIDRPicId = signed_rand_10000(&mut rng) as u32;
        ltr_marking_feedback.iLTRFrameNum = signed_rand_10000(&mut rng);
        encoder.set_ltr_marking_feedback(&mut ltr_marking_feedback);
    }

    assert_eq!(encoder.uninitialize(), CM_RESULT_SUCCESS);
}

#[test]
pub(crate) fn get_option_ltr_all_ltr_case_0() {
    run_get_option_ltr_all_ltr(K_FILE_PARAM_ARRAY[0], 0x1234_5678);
}

#[test]
pub(crate) fn get_option_ltr_all_ltr_case_1() {
    run_get_option_ltr_all_ltr(K_FILE_PARAM_ARRAY[1], 0x2345_6789);
}

#[test]
pub(crate) fn get_option_ltr_all_ltr_case_2() {
    run_get_option_ltr_all_ltr(K_FILE_PARAM_ARRAY[2], 0x3456_789A);
}

/// Port of `EncodeDecodeTestAPI.GetOptionLTR_Engine` (`test/api/ltr_test.cpp:106-156`).
fn run_get_option_ltr_engine(p: LtrTestParam, seed: u32) {
    let mut encoder = Encoder::new();
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

    let mut param = SEncParamExt::default();
    prepare_param_default(
        &encoder,
        1,
        p.slicenum,
        p.width,
        p.height,
        p.frame_rate,
        &mut param,
    );
    assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);

    encoder.set_trace_level(0); // WELS_LOG_QUIET
    decoder.set_trace_level(0);

    encoder.set_sps_pps_id_strategy(EParameterSetStrategy::INCREASING_ID);
    encoder.set_idr_interval(60);
    encoder.set_ltr_config(&SLTRConfig {
        bEnableLongTermReference: true,
        iLTRRefNum: 1,
    });
    encoder.set_ltr_marking_period(2);

    let w = p.width as usize;
    let h = p.height as usize;
    let frame_size = w * h * 3 / 2;
    let mut buf = vec![0u8; frame_size];

    let mut ltr_recover_request = SLTRRecoverRequest {
        uiFeedbackType: NO_RECOVERY_REQUSET,
        uiIDRPicId: 0,
        iLastCorrectFrameNum: 0,
        iCurrentFrameNum: 0,
        iLayerId: 0,
    };
    let mut ltr_marking_feedback = SLTRMarkingFeedback {
        uiFeedbackType: 0,
        uiIDRPicId: 0,
        iLTRFrameNum: 0,
        iLayerId: 0,
    };

    let mut loss_idx = 0usize;
    let mut vcl_loss = false;
    let mut rng = seed;

    for idx in 0..=p.numframes {
        fill_camera_frame(&mut buf, w, h, idx);
        let pic = SourcePicture::from_i420(&buf, p.width, p.height);
        let mut info = SFrameBSInfo::default();
        let rv = encoder.encode_frame(&pic, &mut info);
        assert!(rv == CM_RESULT_SUCCESS || rv == CM_UNKNOWN_REASON);
        if ltr_recover_request.uiFeedbackType == IDR_RECOVERY_REQUEST {
            assert_eq!(info.eFrameType, EVideoFrameType::videoFrameTypeIDR);
        }

        let raw_bs = collect_frame_bitstream(&encoder, &info);
        let (lossy_bs, _skipped) = simulate_nal_loss(
            &raw_bs,
            p.loss_sequence,
            p.lost_para,
            &mut loss_idx,
            &mut vcl_loss,
            &mut rng,
        );

        let mut dst_buf_info = SBufferInfo::default();
        let hr1 = decoder.decode(Some(&lossy_bs), &mut dst_buf_info);
        ltr_recover_request.uiFeedbackType = NO_RECOVERY_REQUSET;
        ltr_recovery_request(&decoder, &mut encoder, &mut ltr_recover_request, hr1, true);

        let hr2 = decoder.decode(None, &mut dst_buf_info);
        ltr_recovery_request(&decoder, &mut encoder, &mut ltr_recover_request, hr2, true);
        ltr_mark_feedback(&decoder, &mut encoder, &mut ltr_marking_feedback, hr2);
    }

    decoder.uninitialize();
    encoder.uninitialize();
}

#[test]
pub(crate) fn get_option_ltr_engine_case_0() {
    run_get_option_ltr_engine(K_FILE_PARAM_ARRAY[0], 0xCAFE_0001);
}

#[test]
pub(crate) fn get_option_ltr_engine_case_1() {
    run_get_option_ltr_engine(K_FILE_PARAM_ARRAY[1], 0xCAFE_0002);
}

#[test]
pub(crate) fn get_option_ltr_engine_case_2() {
    run_get_option_ltr_engine(K_FILE_PARAM_ARRAY[2], 0xCAFE_0003);
}

/// Port of `EncodeDecodeTestAPI.Engine_SVC_Switch_I` and `Engine_SVC_Switch_P`
/// (`test/api/decoder_ec_test.cpp:643-795`).
fn run_engine_svc_switch(switch_on_p: bool) {
    let mut encoder = Encoder::new();
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

    let width = 160 << 2;
    let height = 96 << 2;
    let slices = 2u32;
    let frame_rate = 6.0f32;
    let numframes = 40usize;
    let loss_seq = b"120012130101012311201221323";

    let mut param = SEncParamExt::default();
    prepare_param_default(&encoder, 4, slices, width, height, frame_rate, &mut param);
    param.iTemporalLayerNum = 2;
    param.iSpatialLayerNum = 4;
    assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);

    encoder.set_trace_level(0);
    decoder.set_trace_level(0);
    encoder.set_sps_pps_id_strategy(EParameterSetStrategy::INCREASING_ID);
    encoder.set_idr_interval(if switch_on_p { 60 } else { 8 });
    encoder.set_ltr_config(&SLTRConfig {
        bEnableLongTermReference: true,
        iLTRRefNum: 1,
    });
    encoder.set_ltr_marking_period(2);

    let w = width as usize;
    let h = height as usize;
    let frame_size = w * h * 3 / 2;
    let mut buf = vec![0u8; frame_size];

    let mut ltr_recover_request = SLTRRecoverRequest {
        uiFeedbackType: NO_RECOVERY_REQUSET,
        uiIDRPicId: 0,
        iLastCorrectFrameNum: 0,
        iCurrentFrameNum: 0,
        iLayerId: 0,
    };
    let mut ltr_marking_feedback = SLTRMarkingFeedback {
        uiFeedbackType: 0,
        uiIDRPicId: 0,
        iLTRFrameNum: 0,
        iLayerId: 0,
    };

    let mut tar_did = 0i32;
    let mut last_did = 0i32;
    let mut rng = 0x55AA_1234u32;

    for idx in 0..=numframes {
        fill_camera_frame(&mut buf, w, h, idx);
        let pic = SourcePicture::from_i420(&buf, width, height);
        let mut info = SFrameBSInfo::default();
        let rv = encoder.encode_frame(&pic, &mut info);
        assert!(rv == CM_RESULT_SUCCESS || rv == CM_UNKNOWN_REASON);
        if ltr_recover_request.uiFeedbackType == IDR_RECOVERY_REQUEST {
            assert_eq!(info.eFrameType, EVideoFrameType::videoFrameTypeIDR);
        }

        if switch_on_p {
            tar_did = if idx < loss_seq.len() {
                match loss_seq[idx] {
                    b'0' => 0,
                    b'1' => 1,
                    b'2' => 2,
                    b'3' => 3,
                    _ => (xorshift32(&mut rng) % 4) as i32,
                }
            } else {
                (xorshift32(&mut rng) % 4) as i32
            };
        } else if info.eFrameType == EVideoFrameType::videoFrameTypeIDR {
            tar_did = (xorshift32(&mut rng) % 4) as i32;
        }

        let filtered_bs = extract_did_nal(&encoder, &info, tar_did);
        let mut dst_buf_info = SBufferInfo::default();

        let hr1 = decoder.decode(Some(&filtered_bs), &mut dst_buf_info);
        if !switch_on_p {
            assert_eq!(hr1, DECODING_STATE::dsErrorFree);
        }
        assert_eq!(decoder.feedback_temporal_id(), Some(-1));

        ltr_recover_request.uiFeedbackType = NO_RECOVERY_REQUSET;
        ltr_recovery_request(
            &decoder,
            &mut encoder,
            &mut ltr_recover_request,
            hr1,
            !switch_on_p,
        );

        let hr2 = decoder.decode(None, &mut dst_buf_info);
        if switch_on_p {
            if info.eFrameType == EVideoFrameType::videoFrameTypeP && idx > 0 && last_did != tar_did
            {
                assert_ne!(hr2, DECODING_STATE::dsErrorFree);
            } else if info.eFrameType == EVideoFrameType::videoFrameTypeIDR {
                assert_eq!(hr2, DECODING_STATE::dsErrorFree);
            }
        } else {
            assert_eq!(hr2, DECODING_STATE::dsErrorFree);
        }

        assert_eq!(
            decoder.feedback_temporal_id(),
            Some(info.sLayerInfo[0].uiTemporalId as i32)
        );

        ltr_recovery_request(
            &decoder,
            &mut encoder,
            &mut ltr_recover_request,
            hr2,
            !switch_on_p,
        );
        ltr_mark_feedback(&decoder, &mut encoder, &mut ltr_marking_feedback, hr2);
        last_did = tar_did;
    }

    decoder.uninitialize();
    encoder.uninitialize();
}

#[test]
pub(crate) fn engine_svc_switch_i() {
    run_engine_svc_switch(false);
}

#[test]
pub(crate) fn engine_svc_switch_p() {
    run_engine_svc_switch(true);
}

/// Port of `EncodeDecodeTestAPI.SetOptionECFlag_ERROR_CON_DISABLE` (`test/api/decoder_ec_test.cpp:13-79`).
fn run_set_option_ec_flag_disable(p: LtrTestParam, seed: u32) {
    let mut encoder = Encoder::new();
    let mut decoder = Decoder::new();

    let dec_param = SDecodingParam {
        uiTargetDqLayer: u8::MAX,
        eEcActiveIdc: ERROR_CON_IDC::ERROR_CON_DISABLE,
        sVideoProperty: SVideoProperty {
            size: size_of::<SVideoProperty>() as u32,
            eVideoBsType: VIDEO_BITSTREAM_DEFAULT,
        },
        ..SDecodingParam::default()
    };
    assert_eq!(decoder.initialize(&dec_param), 0);

    let mut param = SEncParamExt::default();
    prepare_param_default(
        &encoder,
        1,
        p.slicenum,
        p.width,
        p.height,
        p.frame_rate,
        &mut param,
    );
    param.bEnableLongTermReference = true;
    param.iLTRRefNum = 1;
    assert_eq!(encoder.initialize_ext(&param), CM_RESULT_SUCCESS);

    encoder.set_trace_level(0);
    decoder.set_trace_level(0);
    encoder.set_sps_pps_id_strategy(EParameterSetStrategy::INCREASING_ID);
    encoder.set_idr_interval(60);
    encoder.set_ltr_config(&SLTRConfig {
        bEnableLongTermReference: true,
        iLTRRefNum: 1,
    });
    encoder.set_ltr_marking_period(2);

    let w = p.width as usize;
    let h = p.height as usize;
    let frame_size = w * h * 3 / 2;
    let mut buf = vec![0u8; frame_size];

    let mut ltr_recover_request = SLTRRecoverRequest {
        uiFeedbackType: NO_RECOVERY_REQUSET,
        uiIDRPicId: 0,
        iLastCorrectFrameNum: 0,
        iCurrentFrameNum: 0,
        iLayerId: 0,
    };
    let mut ltr_marking_feedback = SLTRMarkingFeedback {
        uiFeedbackType: 0,
        uiIDRPicId: 0,
        iLTRFrameNum: 0,
        iLayerId: 0,
    };

    let mut loss_idx = 0usize;
    let mut vcl_loss = false;
    let mut rng = seed;

    for idx in 0..=p.numframes {
        fill_camera_frame(&mut buf, w, h, idx);
        let pic = SourcePicture::from_i420(&buf, p.width, p.height);
        let mut info = SFrameBSInfo::default();
        let rv = encoder.encode_frame(&pic, &mut info);
        assert!(rv == CM_RESULT_SUCCESS || rv == CM_UNKNOWN_REASON);
        if ltr_recover_request.uiFeedbackType == IDR_RECOVERY_REQUEST {
            assert_eq!(info.eFrameType, EVideoFrameType::videoFrameTypeIDR);
        }

        let raw_bs = collect_frame_bitstream(&encoder, &info);
        let (lossy_bs, skipped_bytes) = simulate_nal_loss(
            &raw_bs,
            p.loss_sequence,
            p.lost_para,
            &mut loss_idx,
            &mut vcl_loss,
            &mut rng,
        );

        let mut dst_buf_info = SBufferInfo::default();
        let hr1 = decoder.decode(Some(&lossy_bs), &mut dst_buf_info);
        ltr_recover_request.uiFeedbackType = NO_RECOVERY_REQUSET;
        ltr_recovery_request(&decoder, &mut encoder, &mut ltr_recover_request, hr1, true);

        let hr2 = decoder.decode(None, &mut dst_buf_info);
        ltr_recovery_request(&decoder, &mut encoder, &mut ltr_recover_request, hr2, true);
        ltr_mark_feedback(&decoder, &mut encoder, &mut ltr_marking_feedback, hr2);
        if skipped_bytes > 0 && vcl_loss {
            assert_eq!(dst_buf_info.iBufferStatus, 0);
        }
    }

    decoder.uninitialize();
    encoder.uninitialize();
}

#[test]
pub(crate) fn set_option_ec_flag_error_con_disable_case_0() {
    run_set_option_ec_flag_disable(K_FILE_PARAM_ARRAY[0], 0x7777_0001);
}

#[test]
pub(crate) fn set_option_ec_flag_error_con_disable_case_1() {
    run_set_option_ec_flag_disable(K_FILE_PARAM_ARRAY[1], 0x7777_0002);
}

#[test]
pub(crate) fn set_option_ec_flag_error_con_disable_case_2() {
    run_set_option_ec_flag_disable(K_FILE_PARAM_ARRAY[2], 0x7777_0003);
}
