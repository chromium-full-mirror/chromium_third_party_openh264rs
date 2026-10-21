// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! `CWelsDecoder::DecodeFrame2`'s error-reporting block, the three
//! `DecoderConfigParam` statements it depends on, and the live re-initialisation
//! rebuild.
//!
//! Upstream C++: `codec/decoder/plus/src/welsDecoderExt.cpp` and
//! `codec/decoder/core/src/decoder.cpp`.

#[path = "common/mod.rs"]
mod common;

use openh264rs::api::codec_api::*;
use openh264rs::split_annexb_units;
use std::ffi::c_long;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn asset(name: &str) -> Vec<u8> {
    common::read_stream(name)
}

/// A decoder created and initialised with `param`.
struct Dec(Decoder);

impl Dec {
    fn new(param: &SDecodingParam) -> Self {
        let mut dec = Decoder::new();
        assert_eq!(dec.initialize(param), CM_RESULT_SUCCESS as c_long);
        Self(dec)
    }

    fn get_ec_idc(&self) -> i32 {
        self.0.error_concealment().unwrap() as i32
    }

    fn set_ec_idc(&mut self, v: i32) -> c_long {
        self.0.set_error_concealment(v)
    }
}

impl Drop for Dec {
    fn drop(&mut self) {
        self.0.uninitialize();
    }
}

/// Feeds every Annex-B unit of `data` through `DecodeFrame2`, returning the OR of
/// every state and the number of frames emitted.
fn decode_all(dec: &mut Dec, data: &[u8]) -> (i32, u32) {
    let mut states = 0i32;
    let mut frames = 0u32;
    for unit in split_annexb_units(data) {
        let mut buf_info = SBufferInfo::default();
        let ret = dec.0.decode(Some(unit), &mut buf_info);
        states |= ret.0;
        if buf_info.iBufferStatus == 1 {
            frames += 1;
        }
    }
    (states, frames)
}

// ---------------------------------------------------------------------------
// The `DecoderConfigParam` statements
// ---------------------------------------------------------------------------

/// Tests that out-of-range error concealment mode values are clamped when updated via SetOption.
///
/// `WELS_CLIP3 (iVal, ERROR_CON_DISABLE, …FREEZE_RES_CHANGE)` before the store. Upstream: `welsDecoderExt.cpp`.
#[test]
pub(crate) fn test_error_con_idc_out_of_range_is_clamped_at_set_option() {
    let mut param = SDecodingParam::default();
    param.uiTargetDqLayer = u8::MAX;
    param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
    param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
    let mut dec = Dec::new(&param);
    assert_eq!(dec.get_ec_idc(), 2, "the initial mode did not survive");
    assert_eq!(dec.set_ec_idc(99), CM_RESULT_SUCCESS as c_long);
    assert_eq!(dec.get_ec_idc(), 7, "SetOption(99) was not clamped to 7");
    assert_eq!(dec.set_ec_idc(-5), CM_RESULT_SUCCESS as c_long);
    assert_eq!(dec.get_ec_idc(), 0, "SetOption(-5) was not clamped to 0");
}

/// Tests that parse-only decoding disables error concealment and prevents re-enabling it.
/// Upstream: `decoder.cpp` and `welsDecoderExt.cpp`.
#[test]
pub(crate) fn test_parse_only_disables_error_concealment() {
    let mut param = SDecodingParam::default();
    param.uiTargetDqLayer = u8::MAX;
    param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
    param.bParseOnly = true;
    param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
    let mut dec = Dec::new(&param);
    assert_eq!(
        dec.get_ec_idc(),
        0,
        "bParseOnly did not disable concealment"
    );
    // Concealment may not be re-enabled under parse-only mode.
    assert_eq!(dec.set_ec_idc(2), CM_INIT_PARA_ERROR as c_long);
    assert_eq!(dec.get_ec_idc(), 0, "a rejected SetOption still stored");
}

// ---------------------------------------------------------------------------
// A second `Initialize` on a live decoder rebuilds the context
// ---------------------------------------------------------------------------

/// Tests that calling `Initialize` on an already initialized decoder rebuilds the context and flushes buffered frames.
///
/// `CWelsDecoder::InitDecoder` calls `InitDecoderCtx` for every context, which opens
/// with `UninitDecoderCtx (pCtx)` and allocates a fresh one, so a second `Initialize`
/// on a live decoder is a rebuild. Upstream: `welsDecoderExt.cpp`.
///
/// The observable is the reordering buffer: a B-frame stream stopped mid-GOP leaves
/// pictures buffered, `DECODER_OPTION_NUM_OF_FRAMES_REMAINING_IN_BUFFER` counts
/// them, and after a rebuild there is nothing to count.
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_second_initialize_rebuilds_the_context() {
    let data = asset("Cisco_Men_whisper_640x320_CABAC_Bframe_9.264");
    let mut param = SDecodingParam::default();
    param.uiTargetDqLayer = u8::MAX;
    param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
    param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;

    let mut dec = Dec::new(&param);

    // Feed units until the reordering buffer is holding something.
    let mut buffered = 0i32;
    for unit in split_annexb_units(&data) {
        let mut buf_info = SBufferInfo::default();
        dec.0.decode(Some(unit), &mut buf_info);
        buffered = remaining_in_buffer(&dec);
        if buffered > 0 {
            break;
        }
    }
    assert!(
        buffered > 0,
        "the asset never buffered a picture — this test no longer covers the rebuild"
    );

    assert_eq!(dec.0.initialize(&param), CM_RESULT_SUCCESS as c_long);

    assert_eq!(
        remaining_in_buffer(&dec),
        0,
        "a second Initialize kept the previous session's reordering buffer"
    );
}

fn remaining_in_buffer(dec: &Dec) -> i32 {
    dec.0.frames_remaining()
}
/// Tests that AVC bitstreams notify upper layers of key frame loss on decode errors when concealment is disabled.
/// Upstream: `decoder.cpp` (`eVideoType`) and `welsDecoderExt.cpp` (`DecodeFrame2` error block).
///
/// The field's reader is the key-frame-loss notification in the `DecodeFrame2` error
/// block: for an AVC bitstream any error raises `bParamSetsLostFlag` when concealment
/// is off, and `UpdateAccessUnit`'s mosaic-avoidance block then counts one
/// `uiIDRLostNum` for the next access unit that arrives without an IDR.
///
/// The observable is that counter, not the frame count: `bParamSetsLostFlag` is
/// already true whenever a frame failed to construct, so the arm only adds something
/// when an error arrives on a call that did construct a frame — hence one truncated
/// slice in an otherwise clean stream.
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_avc_bitstream_type_notifies_key_frame_loss_when_ec_is_off() {
    let data = asset("BA_MW_D.264");
    // One slice NAL cut in half, deep enough into the stream that the frames around
    // it construct cleanly and so clear `bParamSetsLostFlag`.
    let mut units: Vec<Vec<u8>> = split_annexb_units(&data).map(|u| u.to_vec()).collect();
    assert!(
        units.len() > 8,
        "asset is too short to corrupt a settled slice"
    );
    let half = units[3].len() / 2;
    assert!(half > 6, "unit 3 is too short to corrupt");
    units[3].truncate(half);

    let mut param = SDecodingParam::default();
    param.uiTargetDqLayer = u8::MAX;
    param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_DISABLE;

    param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_TYPE::VIDEO_BITSTREAM_AVC;
    let mut avc = Dec::new(&param);
    let (avc_states, avc_frames) = decode_units(&mut avc, &units);
    let avc_stats = statistics(&avc);

    param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_TYPE::VIDEO_BITSTREAM_SVC;
    let mut svc = Dec::new(&param);
    let (svc_states, svc_frames) = decode_units(&mut svc, &units);
    let svc_stats = statistics(&svc);

    assert_ne!(
        avc_states & DECODING_STATE::dsBitstreamError.0,
        0,
        "the corrupted unit did not produce an error — this test covers nothing"
    );
    assert_eq!(
        avc_stats.uiIDRLostNum,
        svc_stats.uiIDRLostNum + 1,
        "declaring the stream AVC changed nothing (avc states {avc_states:#x}, \
         svc states {svc_states:#x})"
    );
    assert_eq!(
        avc_frames, svc_frames,
        "the notification changed the emitted frame count, which it must not"
    );
}

/// The same feed as [`decode_all`], over units the caller has already cut up.
fn decode_units(dec: &mut Dec, units: &[Vec<u8>]) -> (i32, u32) {
    let mut states = 0i32;
    let mut frames = 0u32;
    for unit in units {
        let mut buf_info = SBufferInfo::default();
        let ret = dec.0.decode(Some(unit), &mut buf_info);
        states |= ret.0;
        if buf_info.iBufferStatus == 1 {
            frames += 1;
        }
    }
    (states, frames)
}

/// Tests that error concealment statistics are accurately tracked and reported via DECODER_OPTION_GET_STATISTICS.
/// Upstream: `welsDecoderExt.cpp`.
///
/// `uiDecodedFrameCount`, `uiAvgEcRatio`, `uiAvgEcPropRatio` and `uiEcFrameNum`,
/// read back through the public `DECODER_OPTION_GET_STATISTICS` option.
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_concealment_statistics_reach_get_statistics() {
    let data = asset("BA_MW_D_IDR_LOST.264");
    let mut param = SDecodingParam::default();
    param.uiTargetDqLayer = u8::MAX;
    param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
    param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;

    let mut dec = Dec::new(&param);
    let before = statistics(&dec);
    assert_eq!(
        before.uiDecodedFrameCount, 0,
        "a fresh decoder had a frame count"
    );
    assert_eq!(
        before.uiEcFrameNum, 0,
        "a fresh decoder had concealed frames"
    );

    let (states, frames) = decode_all(&mut dec, &data);
    assert!(frames > 0, "the asset decoded nothing");
    assert_ne!(
        states & DECODING_STATE::dsDataErrorConcealed.0,
        0,
        "the asset did not conceal — this test no longer covers the statistics"
    );

    let after = statistics(&dec);
    assert!(
        after.uiDecodedFrameCount > 0,
        "uiDecodedFrameCount never moved: {after:?}"
    );
    assert_eq!(
        after.uiDecodedFrameCount, frames,
        "uiDecodedFrameCount disagrees with the frames the caller saw"
    );
    assert!(
        after.uiEcFrameNum > 0,
        "uiEcFrameNum never moved on a stream that concealed: {after:?}"
    );
    // `:645–649` — both speeds are derived from `dDecTime` at read time, so a
    // decoder that has decoded anything reports a positive one.
    assert!(
        after.fAverageFrameSpeedInMs > 0.0,
        "fAverageFrameSpeedInMs is not positive: {after:?}"
    );
}

fn statistics(dec: &Dec) -> SDecoderStatistics {
    dec.0.statistics().unwrap()
}
