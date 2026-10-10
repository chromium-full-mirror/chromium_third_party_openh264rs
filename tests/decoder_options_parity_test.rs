// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Tests for `ISVCDecoder::GetOption` and `ISVCDecoder::SetOption` across decode and flush calls.

mod common;

use openh264rs::api::codec_api::*;
use openh264rs::split_annexb_units;

const SENTINEL: i32 = 0x5EED_5EED;

#[derive(Debug, Clone, PartialEq, Eq)]
struct CallOptions {
    eos: (i64, i32),
    vcl: (i64, i32),
    tid: (i64, i32),
    frame_num: (i64, i32),
    idr_pic_id: (i64, i32),
    ltr_flag: (i64, i32),
    ltr_frame_num: (i64, i32),
    ec: (i64, i32),
    profile: (i64, i32),
    level: (i64, i32),
    is_ref_pic: (i64, i32),
    remaining: (i64, i32),
}

unsafe fn query_opt(dec: *mut ISVCDecoder, id: DECODER_OPTION) -> (i64, i32) {
    let mut v: i32 = SENTINEL;
    let rc = unsafe { ISVCDecoder::GetOption(dec, id, std::ptr::addr_of_mut!(v).cast()) };
    (rc, v)
}

unsafe fn snapshot_options(dec: *mut ISVCDecoder) -> CallOptions {
    unsafe {
        CallOptions {
            eos: query_opt(dec, DECODER_OPTION::DECODER_OPTION_END_OF_STREAM),
            vcl: query_opt(dec, DECODER_OPTION::DECODER_OPTION_VCL_NAL),
            tid: query_opt(dec, DECODER_OPTION::DECODER_OPTION_TEMPORAL_ID),
            frame_num: query_opt(dec, DECODER_OPTION::DECODER_OPTION_FRAME_NUM),
            idr_pic_id: query_opt(dec, DECODER_OPTION::DECODER_OPTION_IDR_PIC_ID),
            ltr_flag: query_opt(dec, DECODER_OPTION::DECODER_OPTION_LTR_MARKING_FLAG),
            ltr_frame_num: query_opt(dec, DECODER_OPTION::DECODER_OPTION_LTR_MARKED_FRAME_NUM),
            ec: query_opt(dec, DECODER_OPTION::DECODER_OPTION_ERROR_CON_IDC),
            profile: query_opt(dec, DECODER_OPTION::DECODER_OPTION_PROFILE),
            level: query_opt(dec, DECODER_OPTION::DECODER_OPTION_LEVEL),
            is_ref_pic: query_opt(dec, DECODER_OPTION::DECODER_OPTION_IS_REF_PIC),
            remaining: query_opt(
                dec,
                DECODER_OPTION::DECODER_OPTION_NUM_OF_FRAMES_REMAINING_IN_BUFFER,
            ),
        }
    }
}

unsafe fn run_stream_options(data: &[u8]) -> Vec<CallOptions> {
    unsafe {
        let mut dec: *mut ISVCDecoder = std::ptr::null_mut();
        assert_eq!(WelsCreateDecoder(&mut dec), CM_RESULT_SUCCESS as i64);
        let mut param = SDecodingParam::default();
        param.uiTargetDqLayer = u8::MAX;
        param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
        param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
        assert_eq!(
            ISVCDecoder::Initialize(dec, &param as *const SDecodingParam),
            CM_RESULT_SUCCESS as i64
        );

        let mut snapshots = Vec::new();
        let feed =
            |dec: *mut ISVCDecoder, src: *const u8, len: i32, snapshots: &mut Vec<CallOptions>| {
                let mut dst: [*mut u8; 3] = [std::ptr::null_mut(); 3];
                let mut info = SBufferInfo::default();
                ISVCDecoder::DecodeFrame2(dec, src, len, dst.as_mut_ptr(), &mut info);
                snapshots.push(snapshot_options(dec));
            };

        for unit in split_annexb_units(data) {
            feed(dec, unit.as_ptr(), unit.len() as i32, &mut snapshots);
        }

        let mut eos = 1i32;
        ISVCDecoder::SetOption(
            dec,
            DECODER_OPTION::DECODER_OPTION_END_OF_STREAM,
            std::ptr::addr_of_mut!(eos).cast(),
        );
        feed(dec, std::ptr::null(), 0, &mut snapshots);

        let mut remaining = 0i32;
        ISVCDecoder::GetOption(
            dec,
            DECODER_OPTION::DECODER_OPTION_NUM_OF_FRAMES_REMAINING_IN_BUFFER,
            std::ptr::addr_of_mut!(remaining).cast(),
        );
        for _ in 0..remaining.clamp(0, 24) {
            let mut dst: [*mut u8; 3] = [std::ptr::null_mut(); 3];
            let mut info = SBufferInfo::default();
            ISVCDecoder::FlushFrame(dec, dst.as_mut_ptr(), &mut info);
            snapshots.push(snapshot_options(dec));
        }

        ISVCDecoder::Uninitialize(dec);
        WelsDestroyDecoder(dec);
        snapshots
    }
}

fn check_common_invariants(asset: &str, calls: &[CallOptions]) {
    let ok = CM_RESULT_SUCCESS as i64;
    let init_expected = CM_INIT_EXPECTED as i64;

    // First call is always a parameter set (non-VCL): profile/level return cmInitExpected
    // without touching the caller's sentinel value.
    let first = &calls[0];
    assert_eq!(first.eos, (ok, 0), "{asset}: initial EOS");
    assert_eq!(
        first.vcl,
        (ok, FEEDBACK_VCL_NAL_IN_AU::FEEDBACK_UNKNOWN_NAL as i32),
        "{asset}: first NAL is non-VCL"
    );
    assert_eq!(first.frame_num, (ok, -1), "{asset}: initial frame_num");
    assert_eq!(first.is_ref_pic, (ok, -1), "{asset}: initial is_ref_pic");
    assert_eq!(
        first.profile,
        (init_expected, SENTINEL),
        "{asset}: profile before first slice"
    );
    assert_eq!(
        first.level,
        (init_expected, SENTINEL),
        "{asset}: level before first slice"
    );
    assert_eq!(
        first.ec,
        (ok, ERROR_CON_IDC::ERROR_CON_SLICE_COPY as i32),
        "{asset}: error concealment mode"
    );

    // Every scalar option except PROFILE/LEVEL before activation must succeed (`cmResultSuccess`).
    for (i, c) in calls.iter().enumerate() {
        assert_eq!(c.eos.0, ok, "{asset} call {i}: EOS rc");
        assert_eq!(c.vcl.0, ok, "{asset} call {i}: VCL rc");
        assert_eq!(c.tid.0, ok, "{asset} call {i}: TID rc");
        assert_eq!(c.frame_num.0, ok, "{asset} call {i}: FN rc");
        assert_eq!(c.idr_pic_id.0, ok, "{asset} call {i}: IDR rc");
        assert_eq!(c.ltr_flag.0, ok, "{asset} call {i}: LTRF rc");
        assert_eq!(c.ltr_frame_num.0, ok, "{asset} call {i}: LTRN rc");
        assert_eq!(c.ec.0, ok, "{asset} call {i}: EC rc");
        assert_eq!(c.is_ref_pic.0, ok, "{asset} call {i}: REF rc");
        assert_eq!(c.remaining.0, ok, "{asset} call {i}: REM rc");
    }

    // Final call (after EOS and flushing) must have EOS=1 and 0 frames remaining in buffer.
    let last = calls.last().unwrap();
    assert_eq!(last.eos, (ok, 1), "{asset}: final EOS must be 1");
    assert_eq!(
        last.remaining,
        (ok, 0),
        "{asset}: buffer must be drained at end of stream"
    );
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn get_option_per_call_across_streams() {
    let read = |name: &str| -> Vec<u8> { common::read_stream(&format!("{name}.264")) };
    let ok = CM_RESULT_SUCCESS as i64;

    // 1. BA_MW_D: 100 frames of CAVLC baseline (profile 66, level 10), IDR_PIC_ID in {0, 10, 12, 14}.
    let bytes = read("BA_MW_D");
    let calls = unsafe { run_stream_options(&bytes) };
    assert_eq!(calls.len(), 103);
    check_common_invariants("BA_MW_D", &calls);
    assert!(
        calls
            .iter()
            .any(|c| c.profile == (ok, 66) && c.level == (ok, 10))
    );
    assert_eq!(calls.iter().map(|c| c.frame_num.1).max(), Some(29));
    for idr in [0, 10, 12, 14] {
        assert!(
            calls.iter().any(|c| c.idr_pic_id == (ok, idr)),
            "BA_MW_D: expected IDR_PIC_ID={idr}"
        );
    }

    // 2. Error_I_P: resolution/SPS switch mid-stream changes LEVEL from 13 to 30.
    let bytes = read("Error_I_P");
    let calls = unsafe { run_stream_options(&bytes) };
    assert_eq!(calls.len(), 17);
    check_common_invariants("Error_I_P", &calls);
    assert!(
        calls
            .iter()
            .any(|c| c.profile == (ok, 66) && c.level == (ok, 13))
    );
    assert!(
        calls
            .iter()
            .any(|c| c.profile == (ok, 66) && c.level == (ok, 30))
    );

    // 3. MR2_TANDBERG_E: exercises long-term reference marking (LTR_MARKING_FLAG = 1).
    let bytes = read("MR2_TANDBERG_E");
    let calls = unsafe { run_stream_options(&bytes) };
    assert_eq!(calls.len(), 303);
    check_common_invariants("MR2_TANDBERG_E", &calls);
    assert!(
        calls
            .iter()
            .any(|c| c.profile == (ok, 66) && c.level == (ok, 31))
    );
    assert!(calls.iter().any(|c| c.ltr_flag == (ok, 1)));
    assert!(calls.iter().any(|c| c.ltr_frame_num.1 > 0));

    // 4. Cisco_Men_whisper_640x320_CABAC_Bframe_9: CABAC with B-frames (IS_REF_PIC = 0 on B-frames).
    let bytes = read("Cisco_Men_whisper_640x320_CABAC_Bframe_9");
    let calls = unsafe { run_stream_options(&bytes) };
    assert_eq!(calls.len(), 20);
    check_common_invariants("Cisco_Men_whisper_640x320_CABAC_Bframe_9", &calls);
    assert!(
        calls
            .iter()
            .any(|c| c.profile == (ok, 77) && c.level == (ok, 52))
    );
    assert!(calls.iter().any(|c| c.is_ref_pic == (ok, 1)));
    assert!(calls.iter().any(|c| c.is_ref_pic == (ok, 0)));

    // 5. QCIF_2P_I_allIPCM: High profile (100), level 40, I_PCM macroblocks.
    let bytes = read("QCIF_2P_I_allIPCM");
    let calls = unsafe { run_stream_options(&bytes) };
    assert_eq!(calls.len(), 7);
    check_common_invariants("QCIF_2P_I_allIPCM", &calls);
    assert!(
        calls
            .iter()
            .any(|c| c.profile == (ok, 100) && c.level == (ok, 40))
    );

    // 6. narrow_16x16: 1-macroblock frames across 4 IDR periods (IDR_PIC_ID 0..=3).
    let bytes = read("narrow_16x16");
    let calls = unsafe { run_stream_options(&bytes) };
    assert_eq!(calls.len(), 31);
    check_common_invariants("narrow_16x16", &calls);
    assert!(
        calls
            .iter()
            .any(|c| c.profile == (ok, 100) && c.level == (ok, 13))
    );
    for idr in 0..=3 {
        assert!(
            calls.iter().any(|c| c.idr_pic_id == (ok, idr)),
            "narrow_16x16: expected IDR_PIC_ID={idr}"
        );
    }
}

/// Tests struct-valued options and error codes around `GetOption` / `SetOption`.
#[test]
fn option_error_codes_match_the_reference() {
    unsafe {
        let mut dec: *mut ISVCDecoder = std::ptr::null_mut();
        assert_eq!(WelsCreateDecoder(&mut dec), CM_RESULT_SUCCESS as i64);

        // ---- before Initialize -------------------------------------------
        let mut v = 0i32;
        let p: *mut std::ffi::c_void = std::ptr::addr_of_mut!(v).cast();
        assert_eq!(
            ISVCDecoder::GetOption(dec, DECODER_OPTION::DECODER_OPTION_NUM_OF_THREADS, p),
            CM_RESULT_SUCCESS as i64,
            "NUM_OF_THREADS is the object's field and works before Initialize"
        );
        assert_eq!(v, 0, "this port is single-threaded");
        assert_eq!(
            ISVCDecoder::GetOption(dec, DECODER_OPTION::DECODER_OPTION_VCL_NAL, p),
            CM_INIT_EXPECTED as i64,
            "no context yet"
        );
        assert_eq!(
            ISVCDecoder::GetOption(
                dec,
                DECODER_OPTION::DECODER_OPTION_VCL_NAL,
                std::ptr::null_mut()
            ),
            CM_INIT_EXPECTED as i64,
            "the context is tested before pOption"
        );
        assert_eq!(
            ISVCDecoder::SetOption(dec, DECODER_OPTION::DECODER_OPTION_END_OF_STREAM, p),
            i64::from(DECODING_STATE::dsInitialOptExpected.0),
        );

        // ---- after Initialize --------------------------------------------
        let mut param = SDecodingParam::default();
        param.uiTargetDqLayer = u8::MAX;
        param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
        param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
        assert_eq!(
            ISVCDecoder::Initialize(dec, &param as *const SDecodingParam),
            CM_RESULT_SUCCESS as i64
        );

        assert_eq!(
            ISVCDecoder::GetOption(
                dec,
                DECODER_OPTION::DECODER_OPTION_VCL_NAL,
                std::ptr::null_mut()
            ),
            CM_INIT_PARA_ERROR as i64,
        );

        assert_eq!(
            ISVCDecoder::SetOption(dec, DECODER_OPTION::DECODER_OPTION_GET_STATISTICS, p),
            CM_INIT_PARA_ERROR as i64,
        );
        assert_eq!(
            ISVCDecoder::SetOption(dec, DECODER_OPTION::DECODER_OPTION_GET_SAR_INFO, p),
            CM_INIT_PARA_ERROR as i64,
        );

        let mut interval = 0u32;
        let ip: *mut std::ffi::c_void = std::ptr::addr_of_mut!(interval).cast();
        assert_eq!(
            ISVCDecoder::GetOption(
                dec,
                DECODER_OPTION::DECODER_OPTION_STATISTICS_LOG_INTERVAL,
                ip
            ),
            CM_RESULT_SUCCESS as i64
        );
        assert_eq!(interval, 1000);
        let mut set_to = 77u32;
        assert_eq!(
            ISVCDecoder::SetOption(
                dec,
                DECODER_OPTION::DECODER_OPTION_STATISTICS_LOG_INTERVAL,
                std::ptr::addr_of_mut!(set_to).cast()
            ),
            CM_RESULT_SUCCESS as i64
        );
        interval = 0;
        ISVCDecoder::GetOption(
            dec,
            DECODER_OPTION::DECODER_OPTION_STATISTICS_LOG_INTERVAL,
            ip,
        );
        assert_eq!(interval, 77, "the set arm writes what the get arm reads");

        let mut sar = SVuiSarInfo {
            uiSarWidth: 4242,
            uiSarHeight: 4242,
            bOverscanAppropriateFlag: true,
        };
        assert_eq!(
            ISVCDecoder::GetOption(
                dec,
                DECODER_OPTION::DECODER_OPTION_GET_SAR_INFO,
                std::ptr::addr_of_mut!(sar).cast()
            ),
            CM_INIT_EXPECTED as i64
        );
        assert_eq!((sar.uiSarWidth, sar.uiSarHeight), (0, 0));

        for id in [
            DECODER_OPTION::DECODER_OPTION_TRACE_LEVEL,
            DECODER_OPTION::DECODER_OPTION_TRACE_CALLBACK,
            DECODER_OPTION::DECODER_OPTION_TRACE_CALLBACK_CONTEXT,
        ] {
            assert_eq!(
                ISVCDecoder::GetOption(dec, id, p),
                CM_INIT_PARA_ERROR as i64,
                "{id:?} has no GetOption arm"
            );
        }
        for id in [
            DECODER_OPTION::DECODER_OPTION_VCL_NAL,
            DECODER_OPTION::DECODER_OPTION_TEMPORAL_ID,
            DECODER_OPTION::DECODER_OPTION_FRAME_NUM,
            DECODER_OPTION::DECODER_OPTION_IDR_PIC_ID,
            DECODER_OPTION::DECODER_OPTION_LTR_MARKING_FLAG,
            DECODER_OPTION::DECODER_OPTION_LTR_MARKED_FRAME_NUM,
            DECODER_OPTION::DECODER_OPTION_PROFILE,
            DECODER_OPTION::DECODER_OPTION_LEVEL,
            DECODER_OPTION::DECODER_OPTION_IS_REF_PIC,
            DECODER_OPTION::DECODER_OPTION_NUM_OF_FRAMES_REMAINING_IN_BUFFER,
        ] {
            assert_eq!(
                ISVCDecoder::SetOption(dec, id, p),
                CM_INIT_PARA_ERROR as i64,
                "{id:?} has no SetOption arm"
            );
        }

        ISVCDecoder::Uninitialize(dec);
        WelsDestroyDecoder(dec);
    }
}
