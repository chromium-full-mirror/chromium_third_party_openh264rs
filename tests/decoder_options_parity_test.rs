// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Tests for `ISVCDecoder::GetOption` and `ISVCDecoder::SetOption` across decode and flush calls.

#[path = "common/mod.rs"]
mod common;

use openh264rs::api::codec_api::*;
use openh264rs::split_annexb_units;
use std::ffi::c_long;

const SENTINEL: i32 = 0x5EED_5EED;

#[derive(Debug, Clone, PartialEq, Eq)]
struct CallOptions {
    eos: (c_long, i32),
    vcl: (c_long, i32),
    tid: (c_long, i32),
    frame_num: (c_long, i32),
    idr_pic_id: (c_long, i32),
    ltr_flag: (c_long, i32),
    ltr_frame_num: (c_long, i32),
    ec: (c_long, i32),
    profile: (c_long, i32),
    level: (c_long, i32),
    is_ref_pic: (c_long, i32),
    remaining: (c_long, i32),
}

fn opt_to_rc(val: Option<i32>) -> (c_long, i32) {
    match val {
        Some(v) => (CM_RESULT_SUCCESS as c_long, v),
        None => (CM_INIT_EXPECTED as c_long, SENTINEL),
    }
}

fn snapshot_options(dec: &Decoder) -> CallOptions {
    let has_ctx = dec.has_ctx();
    CallOptions {
        eos: opt_to_rc(has_ctx.then(|| i32::from(dec.end_of_stream()))),
        vcl: opt_to_rc(dec.feedback_vcl_nal()),
        tid: opt_to_rc(dec.feedback_temporal_id()),
        frame_num: opt_to_rc(dec.frame_num()),
        idr_pic_id: opt_to_rc(dec.cur_idr_pic_id()),
        ltr_flag: opt_to_rc(dec.ltr_marking_flag()),
        ltr_frame_num: opt_to_rc(dec.ltr_marked_frame_num()),
        ec: opt_to_rc(dec.error_concealment().map(|idc| idc as i32)),
        profile: opt_to_rc(dec.active_sps_profile().flatten()),
        level: opt_to_rc(dec.active_sps_level().flatten()),
        is_ref_pic: opt_to_rc(dec.feedback_is_ref_pic()),
        remaining: opt_to_rc(has_ctx.then(|| dec.frames_remaining())),
    }
}

fn run_stream_options(data: &[u8]) -> Vec<CallOptions> {
    let mut dec = Decoder::new();
    let mut param = SDecodingParam::default();
    param.uiTargetDqLayer = u8::MAX;
    param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
    param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
    assert_eq!(dec.initialize(&param), CM_RESULT_SUCCESS as c_long);

    let mut snapshots = Vec::new();
    let feed = |dec: &mut Decoder, src: Option<&[u8]>, snapshots: &mut Vec<CallOptions>| {
        let mut info = SBufferInfo::default();
        dec.decode(src, &mut info);
        snapshots.push(snapshot_options(dec));
    };

    for unit in split_annexb_units(data) {
        feed(&mut dec, Some(unit), &mut snapshots);
    }

    dec.set_end_of_stream(true);
    feed(&mut dec, None, &mut snapshots);

    let remaining = dec.frames_remaining();
    for _ in 0..remaining.clamp(0, 24) {
        let mut info = SBufferInfo::default();
        dec.flush(&mut info);
        snapshots.push(snapshot_options(&dec));
    }

    dec.uninitialize();
    snapshots
}

fn check_common_invariants(asset: &str, calls: &[CallOptions]) {
    let ok = CM_RESULT_SUCCESS as c_long;
    let init_expected = CM_INIT_EXPECTED as c_long;

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
pub(crate) fn get_option_per_call_across_streams() {
    let read = |name: &str| -> Vec<u8> { common::read_stream(&format!("{name}.264")) };
    let ok = CM_RESULT_SUCCESS as c_long;

    // 1. BA_MW_D: 100 frames of CAVLC baseline (profile 66, level 10), IDR_PIC_ID in {0, 10, 12, 14}.
    let bytes = read("BA_MW_D");
    let calls = run_stream_options(&bytes);
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
    let calls = run_stream_options(&bytes);
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
    let calls = run_stream_options(&bytes);
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
    let calls = run_stream_options(&bytes);
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
    let calls = run_stream_options(&bytes);
    assert_eq!(calls.len(), 7);
    check_common_invariants("QCIF_2P_I_allIPCM", &calls);
    assert!(
        calls
            .iter()
            .any(|c| c.profile == (ok, 100) && c.level == (ok, 40))
    );

    // 6. narrow_16x16: 1-macroblock frames across 4 IDR periods (IDR_PIC_ID 0..=3).
    let bytes = read("narrow_16x16");
    let calls = run_stream_options(&bytes);
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

/// Tests struct-valued options and error codes around `Decoder` option accessors.
#[test]
pub(crate) fn option_error_codes_match_the_reference() {
    let mut dec = Decoder::new();

    // ---- before Initialize -------------------------------------------
    assert_eq!(
        dec.num_of_threads(),
        0,
        "NUM_OF_THREADS is the object's field and works before Initialize"
    );
    dec.set_num_of_threads(4);
    assert_eq!(dec.num_of_threads(), 0);
    assert_eq!(dec.feedback_vcl_nal(), None, "no context yet");
    assert!(!dec.set_statistics_log_interval(77));
    assert_eq!(
        dec.set_error_concealment(ERROR_CON_IDC::ERROR_CON_SLICE_COPY as i32),
        DECODING_STATE::dsInitialOptExpected.0 as c_long
    );

    // ---- after Initialize --------------------------------------------
    let mut param = SDecodingParam::default();
    param.uiTargetDqLayer = u8::MAX;
    param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
    param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
    assert_eq!(dec.initialize(&param), CM_RESULT_SUCCESS as c_long);

    assert_eq!(dec.statistics_log_interval(), Some(1000));
    assert!(dec.set_statistics_log_interval(77));
    assert_eq!(
        dec.statistics_log_interval(),
        Some(77),
        "the set arm writes what the get arm reads"
    );

    assert!(matches!(dec.sar_info(), Some(None)));
    assert_eq!(dec.active_sps_profile(), Some(None));
    assert_eq!(dec.active_sps_level(), Some(None));

    dec.uninitialize();
}
