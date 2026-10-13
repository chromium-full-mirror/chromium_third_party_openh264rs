// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Integration tests for Long-Term Reference (LTR) marking, recovery requests,
//! feedback, and closed-loop encoder-decoder NAL loss recovery.
//!
//! Ported from upstream `test/api/ltr_test.cpp`, `test/api/encode_decode_api_test.cpp`,
//! and the LTR/SVC switch tests in `test/api/decoder_ec_test.cpp`.

mod common;

use common::frame_gen::{fill_camera_frame, xorshift32};
use openh264rs::api::codec_api::*;
use openh264rs::encoder::ref_list_mgr_svc::{
    IDR_RECOVERY_REQUEST, LTR_MARKING_FAILED, LTR_MARKING_SUCCESS, LTR_RECOVERY_REQUEST,
    NO_RECOVERY_REQUSET, SLTRMarkingFeedback, SLTRRecoverRequest,
};
use openh264rs::encoder::wels_encoder_ext::SLTRConfig;
use std::ffi::c_void;
use std::ptr;

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
unsafe fn extract_did_nal(info: &SFrameBSInfo, tar_did: i32) -> Vec<u8> {
    let mut dst = Vec::new();
    for j in 0..(info.iLayerNum as usize) {
        let layer = &info.sLayerInfo[j];
        if layer.iNalCount <= 0 || layer.pNalLengthInByte.is_null() || layer.pBsBuf.is_null() {
            continue;
        }
        let nal_lens =
            unsafe { std::slice::from_raw_parts(layer.pNalLengthInByte, layer.iNalCount as usize) };
        let mut offset = 0usize;
        for &nal_len_i32 in nal_lens {
            let nal_len = nal_len_i32.max(0) as usize;
            let nal = unsafe { std::slice::from_raw_parts(layer.pBsBuf.add(offset), nal_len) };
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
                } else if is_vcl_nal_avc_base(nal_type) && tar_did != 0 {
                    false
                } else {
                    true
                };
                if remain {
                    dst.extend_from_slice(nal);
                }
            }
            offset += nal_len;
        }
    }
    dst
}

/// Port of `IsKeyFrameLost` from `test/api/encode_decode_api_test.cpp:203-231`.
unsafe fn is_key_frame_lost(
    decoder: *mut ISVCDecoder,
    ltr_recover_request: &mut SLTRRecoverRequest,
    hr: DECODING_STATE,
) -> u32 {
    unsafe {
        let mut temp_int: i32 = -1;
        let mut temporal_id: i32 = -1;
        let p2p_mode_flag = true;
        ISVCDecoder::GetOption(
            decoder,
            DECODER_OPTION::DECODER_OPTION_TEMPORAL_ID,
            ptr::addr_of_mut!(temporal_id).cast::<c_void>(),
        );
        if hr == DECODING_STATE::dsErrorFree {
            if p2p_mode_flag && temporal_id == 0 {
                ISVCDecoder::GetOption(
                    decoder,
                    DECODER_OPTION::DECODER_OPTION_IDR_PIC_ID,
                    ptr::addr_of_mut!(temp_int).cast::<c_void>(),
                );
                if ltr_recover_request.uiIDRPicId != temp_int as u32 {
                    ltr_recover_request.uiIDRPicId = temp_int as u32;
                    ltr_recover_request.iLastCorrectFrameNum = -1;
                }
                ISVCDecoder::GetOption(
                    decoder,
                    DECODER_OPTION::DECODER_OPTION_FRAME_NUM,
                    ptr::addr_of_mut!(temp_int).cast::<c_void>(),
                );
                if temp_int >= 0 {
                    ltr_recover_request.iLastCorrectFrameNum = temp_int;
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
}

/// Port of `IsLTRMarking` from `test/api/encode_decode_api_test.cpp:233-237`.
unsafe fn is_ltr_marking(decoder: *mut ISVCDecoder) -> bool {
    unsafe {
        let mut ltr_marking_flag: i32 = 0;
        ISVCDecoder::GetOption(
            decoder,
            DECODER_OPTION::DECODER_OPTION_LTR_MARKING_FLAG,
            ptr::addr_of_mut!(ltr_marking_flag).cast::<c_void>(),
        );
        ltr_marking_flag != 0
    }
}

/// Port of `LTRRecoveryRequest` from `test/api/encode_decode_api_test.cpp:239-257`.
unsafe fn ltr_recovery_request(
    decoder: *mut ISVCDecoder,
    encoder: *mut ISVCEncoder,
    req: &mut SLTRRecoverRequest,
    hr: DECODING_STATE,
    p2p_mode_flag: bool,
) {
    unsafe {
        let klost = is_key_frame_lost(decoder, req, hr);
        if p2p_mode_flag {
            if klost == IDR_RECOVERY_REQUEST {
                req.uiFeedbackType = IDR_RECOVERY_REQUEST;
                ISVCEncoder::ForceIntraFrame(encoder, true);
            } else if klost == LTR_RECOVERY_REQUEST {
                req.uiFeedbackType = LTR_RECOVERY_REQUEST;
                ISVCDecoder::GetOption(
                    decoder,
                    DECODER_OPTION::DECODER_OPTION_FRAME_NUM,
                    ptr::addr_of_mut!(req.iCurrentFrameNum).cast::<c_void>(),
                );
                ISVCDecoder::GetOption(
                    decoder,
                    DECODER_OPTION::DECODER_OPTION_IDR_PIC_ID,
                    ptr::addr_of_mut!(req.uiIDRPicId).cast::<c_void>(),
                );
                ISVCEncoder::SetOption(
                    encoder,
                    ENCODER_OPTION::ENCODER_LTR_RECOVERY_REQUEST,
                    (req as *mut SLTRRecoverRequest).cast::<c_void>(),
                );
            }
        } else if klost == IDR_RECOVERY_REQUEST || klost == LTR_RECOVERY_REQUEST {
            req.uiFeedbackType = IDR_RECOVERY_REQUEST;
            ISVCEncoder::ForceIntraFrame(encoder, true);
        }
    }
}

/// Port of `LTRMarkFeedback` from `test/api/encode_decode_api_test.cpp:259-267`.
unsafe fn ltr_mark_feedback(
    decoder: *mut ISVCDecoder,
    encoder: *mut ISVCEncoder,
    fb: &mut SLTRMarkingFeedback,
    hr: DECODING_STATE,
) {
    unsafe {
        if is_ltr_marking(decoder) {
            fb.uiFeedbackType = if hr == DECODING_STATE::dsErrorFree {
                LTR_MARKING_SUCCESS
            } else {
                LTR_MARKING_FAILED
            };
            ISVCDecoder::GetOption(
                decoder,
                DECODER_OPTION::DECODER_OPTION_IDR_PIC_ID,
                ptr::addr_of_mut!(fb.uiIDRPicId).cast::<c_void>(),
            );
            ISVCDecoder::GetOption(
                decoder,
                DECODER_OPTION::DECODER_OPTION_LTR_MARKED_FRAME_NUM,
                ptr::addr_of_mut!(fb.iLTRFrameNum).cast::<c_void>(),
            );
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_LTR_MARKING_FEEDBACK,
                (fb as *mut SLTRMarkingFeedback).cast::<c_void>(),
            );
        }
    }
}

#[test]
fn test_ltr_encoder_configuration() {
    unsafe {
        let mut p_encoder: *mut ISVCEncoder = ptr::null_mut();
        let ret = WelsCreateSVCEncoder(&mut p_encoder);
        assert_eq!(ret, CM_RESULT_SUCCESS);
        assert!(!p_encoder.is_null());

        let mut param = SEncParamExt::default();
        param.iPicWidth = 320;
        param.iPicHeight = 240;
        param.fMaxFrameRate = 30.0;
        param.iTargetBitrate = 500000;
        param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
        param.bEnableLongTermReference = true;
        param.iLtrMarkPeriod = 30;

        let init_ret = ISVCEncoder::InitializeExt(p_encoder, &param as *const SEncParamExt);
        assert_eq!(init_ret, CM_RESULT_SUCCESS);

        let uninit_ret = ISVCEncoder::Uninitialize(p_encoder);
        assert_eq!(uninit_ret, CM_RESULT_SUCCESS);

        WelsDestroySVCEncoder(p_encoder);
    }
}

/// Port of `EncodeDecodeTestAPI.GetOptionLTR_ALLLTR` (`test/api/ltr_test.cpp:45-104`).
fn run_get_option_ltr_all_ltr(p: LtrTestParam, seed: u32) {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let mut param = SEncParamExt::default();
        prepare_param_default(
            encoder,
            1,
            p.slicenum,
            p.width,
            p.height,
            p.frame_rate,
            &mut param,
        );
        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param as *const SEncParamExt),
            CM_RESULT_SUCCESS
        );

        let mut trace_level: i32 = 0; // WELS_LOG_QUIET
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_OPTION_TRACE_LEVEL,
            ptr::addr_of_mut!(trace_level).cast(),
        );
        let mut sps_pps_strategy: i32 = 1;
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_OPTION_SPS_PPS_ID_STRATEGY,
            ptr::addr_of_mut!(sps_pps_strategy).cast(),
        );
        let mut idr_period: i32 = 60;
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_OPTION_IDR_INTERVAL,
            ptr::addr_of_mut!(idr_period).cast(),
        );
        let mut ltr_cfg = SLTRConfig {
            bEnableLongTermReference: true,
            iLTRRefNum: 1,
        };
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_OPTION_LTR,
            ptr::addr_of_mut!(ltr_cfg).cast(),
        );
        let mut ltr_period: i32 = 2;
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_LTR_MARKING_PERIOD,
            ptr::addr_of_mut!(ltr_period).cast(),
        );

        let w = p.width as usize;
        let h = p.height as usize;
        let frame_size = w * h * 3 / 2;
        let mut buf = vec![0u8; frame_size];

        let mut pic = SSourcePicture::default();
        pic.iPicWidth = p.width;
        pic.iPicHeight = p.height;
        pic.iColorFormat = EVideoFormatType::videoFormatI420 as i32;
        pic.iStride[0] = p.width;
        pic.iStride[1] = p.width >> 1;
        pic.iStride[2] = p.width >> 1;
        pic.pData[0] = buf.as_mut_ptr();
        pic.pData[1] = buf.as_mut_ptr().add(w * h);
        pic.pData[2] = buf.as_mut_ptr().add(w * h + (w * h >> 2));

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
            let mut info = SFrameBSInfo::default();
            let rv = ISVCEncoder::EncodeFrame(encoder, &pic as *const SSourcePicture, &mut info);
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
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_LTR_RECOVERY_REQUEST,
                ptr::addr_of_mut!(ltr_recover_request).cast(),
            );

            ltr_marking_feedback.uiFeedbackType = if (xorshift32(&mut rng) & 1) == 1 {
                LTR_MARKING_SUCCESS
            } else {
                LTR_MARKING_FAILED
            };
            ltr_marking_feedback.uiIDRPicId = signed_rand_10000(&mut rng) as u32;
            ltr_marking_feedback.iLTRFrameNum = signed_rand_10000(&mut rng);
            ISVCEncoder::SetOption(
                encoder,
                ENCODER_OPTION::ENCODER_LTR_MARKING_FEEDBACK,
                ptr::addr_of_mut!(ltr_marking_feedback).cast(),
            );
        }

        assert_eq!(ISVCEncoder::Uninitialize(encoder), CM_RESULT_SUCCESS);
        WelsDestroySVCEncoder(encoder);
    }
}

#[test]
fn get_option_ltr_all_ltr_case_0() {
    run_get_option_ltr_all_ltr(K_FILE_PARAM_ARRAY[0], 0x1234_5678);
}

#[test]
fn get_option_ltr_all_ltr_case_1() {
    run_get_option_ltr_all_ltr(K_FILE_PARAM_ARRAY[1], 0x2345_6789);
}

#[test]
fn get_option_ltr_all_ltr_case_2() {
    run_get_option_ltr_all_ltr(K_FILE_PARAM_ARRAY[2], 0x3456_789A);
}

/// Port of `EncodeDecodeTestAPI.GetOptionLTR_Engine` (`test/api/ltr_test.cpp:106-156`).
fn run_get_option_ltr_engine(p: LtrTestParam, seed: u32) {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

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

        let mut param = SEncParamExt::default();
        prepare_param_default(
            encoder,
            1,
            p.slicenum,
            p.width,
            p.height,
            p.frame_rate,
            &mut param,
        );
        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param as *const SEncParamExt),
            CM_RESULT_SUCCESS
        );

        let mut trace_level: i32 = 0; // WELS_LOG_QUIET
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

        let mut sps_pps_strategy: i32 = 1;
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_OPTION_SPS_PPS_ID_STRATEGY,
            ptr::addr_of_mut!(sps_pps_strategy).cast(),
        );
        let mut idr_period: i32 = 60;
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_OPTION_IDR_INTERVAL,
            ptr::addr_of_mut!(idr_period).cast(),
        );
        let mut ltr_cfg = SLTRConfig {
            bEnableLongTermReference: true,
            iLTRRefNum: 1,
        };
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_OPTION_LTR,
            ptr::addr_of_mut!(ltr_cfg).cast(),
        );
        let mut ltr_period: i32 = 2;
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_LTR_MARKING_PERIOD,
            ptr::addr_of_mut!(ltr_period).cast(),
        );

        let w = p.width as usize;
        let h = p.height as usize;
        let frame_size = w * h * 3 / 2;
        let mut buf = vec![0u8; frame_size];

        let mut pic = SSourcePicture::default();
        pic.iPicWidth = p.width;
        pic.iPicHeight = p.height;
        pic.iColorFormat = EVideoFormatType::videoFormatI420 as i32;
        pic.iStride[0] = p.width;
        pic.iStride[1] = p.width >> 1;
        pic.iStride[2] = p.width >> 1;
        pic.pData[0] = buf.as_mut_ptr();
        pic.pData[1] = buf.as_mut_ptr().add(w * h);
        pic.pData[2] = buf.as_mut_ptr().add(w * h + (w * h >> 2));

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
            let mut info = SFrameBSInfo::default();
            let rv = ISVCEncoder::EncodeFrame(encoder, &pic as *const SSourcePicture, &mut info);
            assert!(rv == CM_RESULT_SUCCESS || rv == CM_UNKNOWN_REASON);
            if ltr_recover_request.uiFeedbackType == IDR_RECOVERY_REQUEST {
                assert_eq!(info.eFrameType, EVideoFrameType::videoFrameTypeIDR);
            }

            let raw_bs = collect_frame_bitstream(&info);
            let (lossy_bs, _skipped) = simulate_nal_loss(
                &raw_bs,
                p.loss_sequence,
                p.lost_para,
                &mut loss_idx,
                &mut vcl_loss,
                &mut rng,
            );

            let mut p_data: [*mut u8; 3] = [ptr::null_mut(); 3];
            let mut dst_buf_info = SBufferInfo::default();
            let hr1 = ISVCDecoder::DecodeFrame2(
                decoder,
                lossy_bs.as_ptr(),
                lossy_bs.len() as i32,
                p_data.as_mut_ptr(),
                &mut dst_buf_info,
            );
            ltr_recover_request.uiFeedbackType = NO_RECOVERY_REQUSET;
            ltr_recovery_request(decoder, encoder, &mut ltr_recover_request, hr1, true);

            let hr2 = ISVCDecoder::DecodeFrame2(
                decoder,
                ptr::null(),
                0,
                p_data.as_mut_ptr(),
                &mut dst_buf_info,
            );
            ltr_recovery_request(decoder, encoder, &mut ltr_recover_request, hr2, true);
            ltr_mark_feedback(decoder, encoder, &mut ltr_marking_feedback, hr2);
        }

        ISVCDecoder::Uninitialize(decoder);
        WelsDestroyDecoder(decoder);
        ISVCEncoder::Uninitialize(encoder);
        WelsDestroySVCEncoder(encoder);
    }
}

#[test]
fn get_option_ltr_engine_case_0() {
    run_get_option_ltr_engine(K_FILE_PARAM_ARRAY[0], 0xCAFE_0001);
}

#[test]
fn get_option_ltr_engine_case_1() {
    run_get_option_ltr_engine(K_FILE_PARAM_ARRAY[1], 0xCAFE_0002);
}

#[test]
fn get_option_ltr_engine_case_2() {
    run_get_option_ltr_engine(K_FILE_PARAM_ARRAY[2], 0xCAFE_0003);
}

/// Port of `EncodeDecodeTestAPI.Engine_SVC_Switch_I` and `Engine_SVC_Switch_P`
/// (`test/api/decoder_ec_test.cpp:643-795`).
fn run_engine_svc_switch(switch_on_p: bool) {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

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

        let width = 160 << 2;
        let height = 96 << 2;
        let slices = 2u32;
        let frame_rate = 6.0f32;
        let numframes = 40usize;
        let loss_seq = b"120012130101012311201221323";

        let mut param = SEncParamExt::default();
        prepare_param_default(encoder, 4, slices, width, height, frame_rate, &mut param);
        param.iTemporalLayerNum = 2;
        param.iSpatialLayerNum = 4;
        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param as *const SEncParamExt),
            CM_RESULT_SUCCESS
        );

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
        let mut sps_pps_strategy: i32 = 1;
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_OPTION_SPS_PPS_ID_STRATEGY,
            ptr::addr_of_mut!(sps_pps_strategy).cast(),
        );
        let mut idr_period: i32 = if switch_on_p { 60 } else { 8 };
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_OPTION_IDR_INTERVAL,
            ptr::addr_of_mut!(idr_period).cast(),
        );
        let mut ltr_cfg = SLTRConfig {
            bEnableLongTermReference: true,
            iLTRRefNum: 1,
        };
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_OPTION_LTR,
            ptr::addr_of_mut!(ltr_cfg).cast(),
        );
        let mut ltr_period: i32 = 2;
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_LTR_MARKING_PERIOD,
            ptr::addr_of_mut!(ltr_period).cast(),
        );

        let w = width as usize;
        let h = height as usize;
        let frame_size = w * h * 3 / 2;
        let mut buf = vec![0u8; frame_size];

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
            let mut info = SFrameBSInfo::default();
            let rv = ISVCEncoder::EncodeFrame(encoder, &pic as *const SSourcePicture, &mut info);
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

            let filtered_bs = extract_did_nal(&info, tar_did);
            let mut p_data: [*mut u8; 3] = [ptr::null_mut(); 3];
            let mut dst_buf_info = SBufferInfo::default();

            let hr1 = ISVCDecoder::DecodeFrame2(
                decoder,
                filtered_bs.as_ptr(),
                filtered_bs.len() as i32,
                p_data.as_mut_ptr(),
                &mut dst_buf_info,
            );
            if !switch_on_p {
                assert_eq!(hr1, DECODING_STATE::dsErrorFree);
            }
            let mut tid: i32 = -1;
            ISVCDecoder::GetOption(
                decoder,
                DECODER_OPTION::DECODER_OPTION_TEMPORAL_ID,
                ptr::addr_of_mut!(tid).cast(),
            );
            assert_eq!(tid, -1);

            ltr_recover_request.uiFeedbackType = NO_RECOVERY_REQUSET;
            ltr_recovery_request(
                decoder,
                encoder,
                &mut ltr_recover_request,
                hr1,
                !switch_on_p,
            );

            let hr2 = ISVCDecoder::DecodeFrame2(
                decoder,
                ptr::null(),
                0,
                p_data.as_mut_ptr(),
                &mut dst_buf_info,
            );
            if switch_on_p {
                if info.eFrameType == EVideoFrameType::videoFrameTypeP
                    && idx > 0
                    && last_did != tar_did
                {
                    assert_ne!(hr2, DECODING_STATE::dsErrorFree);
                } else if info.eFrameType == EVideoFrameType::videoFrameTypeIDR {
                    assert_eq!(hr2, DECODING_STATE::dsErrorFree);
                }
            } else {
                assert_eq!(hr2, DECODING_STATE::dsErrorFree);
            }

            ISVCDecoder::GetOption(
                decoder,
                DECODER_OPTION::DECODER_OPTION_TEMPORAL_ID,
                ptr::addr_of_mut!(tid).cast(),
            );
            assert_eq!(tid, info.sLayerInfo[0].uiTemporalId as i32);

            ltr_recovery_request(
                decoder,
                encoder,
                &mut ltr_recover_request,
                hr2,
                !switch_on_p,
            );
            ltr_mark_feedback(decoder, encoder, &mut ltr_marking_feedback, hr2);
            last_did = tar_did;
        }

        ISVCDecoder::Uninitialize(decoder);
        WelsDestroyDecoder(decoder);
        ISVCEncoder::Uninitialize(encoder);
        WelsDestroySVCEncoder(encoder);
    }
}

#[test]
fn engine_svc_switch_i() {
    run_engine_svc_switch(false);
}

#[test]
fn engine_svc_switch_p() {
    run_engine_svc_switch(true);
}

/// Port of `EncodeDecodeTestAPI.SetOptionECFlag_ERROR_CON_DISABLE` (`test/api/decoder_ec_test.cpp:13-79`).
fn run_set_option_ec_flag_disable(p: LtrTestParam, seed: u32) {
    unsafe {
        let mut encoder: *mut ISVCEncoder = ptr::null_mut();
        assert_eq!(WelsCreateSVCEncoder(&mut encoder), CM_RESULT_SUCCESS);

        let mut decoder: *mut ISVCDecoder = ptr::null_mut();
        assert_eq!(WelsCreateDecoder(&mut decoder), 0);

        let dec_param = SDecodingParam {
            uiTargetDqLayer: u8::MAX,
            eEcActiveIdc: ERROR_CON_IDC::ERROR_CON_DISABLE,
            sVideoProperty: SVideoProperty {
                size: size_of::<SVideoProperty>() as u32,
                eVideoBsType: VIDEO_BITSTREAM_DEFAULT,
            },
            ..SDecodingParam::default()
        };
        assert_eq!(ISVCDecoder::Initialize(decoder, &dec_param), 0);

        let mut param = SEncParamExt::default();
        prepare_param_default(
            encoder,
            1,
            p.slicenum,
            p.width,
            p.height,
            p.frame_rate,
            &mut param,
        );
        param.bEnableLongTermReference = true;
        param.iLTRRefNum = 1;
        assert_eq!(
            ISVCEncoder::InitializeExt(encoder, &param as *const SEncParamExt),
            CM_RESULT_SUCCESS
        );

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
        let mut sps_pps_strategy: i32 = 1;
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_OPTION_SPS_PPS_ID_STRATEGY,
            ptr::addr_of_mut!(sps_pps_strategy).cast(),
        );
        let mut idr_period: i32 = 60;
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_OPTION_IDR_INTERVAL,
            ptr::addr_of_mut!(idr_period).cast(),
        );
        let mut ltr_cfg = SLTRConfig {
            bEnableLongTermReference: true,
            iLTRRefNum: 1,
        };
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_OPTION_LTR,
            ptr::addr_of_mut!(ltr_cfg).cast(),
        );
        let mut ltr_period: i32 = 2;
        ISVCEncoder::SetOption(
            encoder,
            ENCODER_OPTION::ENCODER_LTR_MARKING_PERIOD,
            ptr::addr_of_mut!(ltr_period).cast(),
        );

        let w = p.width as usize;
        let h = p.height as usize;
        let frame_size = w * h * 3 / 2;
        let mut buf = vec![0u8; frame_size];

        let mut pic = SSourcePicture::default();
        pic.iPicWidth = p.width;
        pic.iPicHeight = p.height;
        pic.iColorFormat = EVideoFormatType::videoFormatI420 as i32;
        pic.iStride[0] = p.width;
        pic.iStride[1] = p.width >> 1;
        pic.iStride[2] = p.width >> 1;
        pic.pData[0] = buf.as_mut_ptr();
        pic.pData[1] = buf.as_mut_ptr().add(w * h);
        pic.pData[2] = buf.as_mut_ptr().add(w * h + (w * h >> 2));

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
            let mut info = SFrameBSInfo::default();
            let rv = ISVCEncoder::EncodeFrame(encoder, &pic as *const SSourcePicture, &mut info);
            assert!(rv == CM_RESULT_SUCCESS || rv == CM_UNKNOWN_REASON);
            if ltr_recover_request.uiFeedbackType == IDR_RECOVERY_REQUEST {
                assert_eq!(info.eFrameType, EVideoFrameType::videoFrameTypeIDR);
            }

            let raw_bs = collect_frame_bitstream(&info);
            let (lossy_bs, skipped_bytes) = simulate_nal_loss(
                &raw_bs,
                p.loss_sequence,
                p.lost_para,
                &mut loss_idx,
                &mut vcl_loss,
                &mut rng,
            );

            let mut p_data: [*mut u8; 3] = [ptr::null_mut(); 3];
            let mut dst_buf_info = SBufferInfo::default();
            let hr1 = ISVCDecoder::DecodeFrame2(
                decoder,
                lossy_bs.as_ptr(),
                lossy_bs.len() as i32,
                p_data.as_mut_ptr(),
                &mut dst_buf_info,
            );
            ltr_recover_request.uiFeedbackType = NO_RECOVERY_REQUSET;
            ltr_recovery_request(decoder, encoder, &mut ltr_recover_request, hr1, true);

            let hr2 = ISVCDecoder::DecodeFrame2(
                decoder,
                ptr::null(),
                0,
                p_data.as_mut_ptr(),
                &mut dst_buf_info,
            );
            ltr_recovery_request(decoder, encoder, &mut ltr_recover_request, hr2, true);
            ltr_mark_feedback(decoder, encoder, &mut ltr_marking_feedback, hr2);
            if skipped_bytes > 0 && vcl_loss {
                assert_eq!(dst_buf_info.iBufferStatus, 0);
            }
        }

        ISVCDecoder::Uninitialize(decoder);
        WelsDestroyDecoder(decoder);
        ISVCEncoder::Uninitialize(encoder);
        WelsDestroySVCEncoder(encoder);
    }
}

#[test]
fn set_option_ec_flag_error_con_disable_case_0() {
    run_set_option_ec_flag_disable(K_FILE_PARAM_ARRAY[0], 0x7777_0001);
}

#[test]
fn set_option_ec_flag_error_con_disable_case_1() {
    run_set_option_ec_flag_disable(K_FILE_PARAM_ARRAY[1], 0x7777_0002);
}

#[test]
fn set_option_ec_flag_error_con_disable_case_2() {
    run_set_option_ec_flag_disable(K_FILE_PARAM_ARRAY[2], 0x7777_0003);
}
