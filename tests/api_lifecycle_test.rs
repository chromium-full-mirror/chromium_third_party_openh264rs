// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Integration tests for OpenH264 API lifecycle and memory management.
//! C++: `test/api/c_interface_test.c`, `test/api/cpp_interface_test.cpp`.

#[path = "common/mod.rs"]
mod common;

use openh264rs::api::codec_api::*;
use std::ffi::c_long;

#[repr(C)]
struct BoolTestStruct {
    c: std::ffi::c_char,
    b: bool,
}

#[test]
pub(crate) fn test_c_abi_bool_and_struct_alignment() {
    assert_eq!(std::mem::size_of::<bool>(), 1);
    assert_eq!(std::mem::offset_of!(BoolTestStruct, b), 1);
    assert_eq!(std::mem::size_of::<BoolTestStruct>(), 2);
}

#[test]
pub(crate) fn test_decoder_create_and_destroy_lifecycle() {
    let mut dec = Decoder::new();

    let mut param = SDecodingParam::default();
    param.uiTargetDqLayer = u8::MAX;

    let init_ret = dec.initialize(&param);
    assert_eq!(init_ret, CM_RESULT_SUCCESS as c_long);

    let mut buf_info = SBufferInfo::default();
    let dec_nodelay_state = dec.decode_no_delay(None, &mut buf_info);
    assert_eq!(dec_nodelay_state, DECODING_STATE::dsErrorFree);

    let dec2_state = dec.decode(None, &mut buf_info);
    assert_eq!(dec2_state, DECODING_STATE::dsErrorFree);

    let flush_state = dec.flush(&mut buf_info);
    assert_eq!(flush_state, DECODING_STATE::dsErrorFree);

    // This decoder was initialised without `bParseOnly`, so `decode_parser` ors
    // `dsInvalidArgument` into the context's error code and returns it
    // (`welsDecoderExt.cpp:1189-1193`).
    let mut parser_info = SParserBsInfo::default();
    let parse_state = dec.decode_parser(None, &mut parser_info);
    assert_eq!(parse_state, DECODING_STATE::dsInvalidArgument);

    dec.set_trace_level(0);
    assert!(dec.statistics_log_interval().is_some());

    let uninit_ret = dec.uninitialize();
    assert_eq!(uninit_ret, CM_RESULT_SUCCESS as c_long);
}

#[test]
pub(crate) fn test_encoder_create_and_destroy_lifecycle() {
    let mut enc = Encoder::new();

    let mut param = SEncParamBase::default();
    param.iPicWidth = 320;
    param.iPicHeight = 240;
    param.fMaxFrameRate = 30.0;
    param.iTargetBitrate = 500000;
    param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;

    let init_ret = enc.initialize(&param);
    assert_eq!(init_ret, CM_RESULT_SUCCESS);

    let mut param_ext = SEncParamExt::default();
    param_ext.iPicWidth = 320;
    param_ext.iPicHeight = 240;
    param_ext.fMaxFrameRate = 30.0;
    param_ext.iTargetBitrate = 500000;
    param_ext.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
    let init_ext_ret = enc.initialize_ext(&param_ext);
    assert_eq!(init_ext_ret, CM_RESULT_SUCCESS);

    let mut default_param = SEncParamExt::default();
    let get_def_ret = enc.default_params(&mut default_param);
    assert_eq!(get_def_ret, CM_RESULT_SUCCESS);

    let mut src_pic = SourcePicture::default();
    src_pic.iPicWidth = 160;
    src_pic.iPicHeight = 120;
    src_pic.iColorFormat = 23;
    let mut bs_info = SFrameBSInfo::default();
    // A 160x120 source picture against a 320x240 encoder, with `pData` null, is
    // rejected.
    let enc_frame_ret = enc.encode_frame(&src_pic, &mut bs_info);
    assert_eq!(enc_frame_ret, CM_UNSUPPORTED_DATA);

    let enc_ps_ret = enc.encode_parameter_sets(&mut bs_info);
    assert_eq!(enc_ps_ret, CM_RESULT_SUCCESS);

    let force_idr_ret = enc.force_intra_frame(true, -1);
    assert_eq!(force_idr_ret, CM_RESULT_SUCCESS);

    enc.set_trace_level(0);
    assert!(enc.idr_interval().is_some());

    let uninit_ret = enc.uninitialize();
    assert_eq!(uninit_ret, CM_RESULT_SUCCESS);
}

// ============================================================================
// Re-init probe
// ============================================================================

/// One decode pass over `units`, stopping after `limit` access units.
///
/// Returns `(frames emitted, frames the decoder says are still buffered)`. The
/// second number is `sReoderingStatus.iNumOfPicts`, read back through
/// `DECODER_OPTION_NUM_OF_FRAMES_REMAINING_IN_BUFFER`.
///
/// `drain` runs the end-of-stream sequence — signal `DECODER_OPTION_END_OF_STREAM`,
/// one null `decode`, then `flush` per buffered picture — so the frame
/// count covers the whole stream: a stream that reorders holds up to its DPB size of
/// pictures back (Annex C.4.5.3).
fn decode_pass(dec: &mut Decoder, units: &[&[u8]], limit: usize, drain: bool) -> (usize, i32) {
    let mut frames = 0usize;
    for unit in units.iter().take(limit) {
        let mut buf_info = SBufferInfo::default();
        dec.decode(Some(unit), &mut buf_info);
        if buf_info.iBufferStatus == 1 {
            frames += 1;
        }
    }
    if drain {
        dec.set_end_of_stream(true);
        let mut buf_info = SBufferInfo::default();
        dec.decode(None, &mut buf_info);
        if buf_info.iBufferStatus == 1 {
            frames += 1;
        }
        loop {
            let left = dec.frames_remaining();
            if left <= 0 {
                break;
            }
            let mut buf_info = SBufferInfo::default();
            dec.flush(&mut buf_info);
            if buf_info.iBufferStatus == 1 {
                frames += 1;
            } else {
                break;
            }
        }
    }
    let remaining = dec.frames_remaining();
    (frames, remaining)
}

/// A re-initialised decoder must not inherit the previous session's reordering
/// slots.
///
/// The reordering pair (`sPictInfoList`, `sReoderingStatus`) is api-owned state that
/// outlives the decoder context: `Uninitialize` frees the context and its picture
/// pool, and a second `Initialize` builds new ones, so the slots are cleared at the
/// head of `DestroyPicBuff` (`decoder.cpp:260`). A surviving non-sentinel `iPOC` makes
/// a stale slot look occupied, and its `iPicBuffIdx` then indexes the new pool with
/// the old pool's index.
///
/// The stream is interrupted while pictures are still buffered, then re-initialised
/// and decoded whole: `remaining` is 0 after the second `Initialize`, and the second
/// pass emits what a decoder that never saw the first emits.
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_decoder_reinit_does_not_inherit_reordering_slots() {
    let data = common::read_stream("CABA2_SVA_B.264");
    let units: Vec<_> = openh264rs::split_annexb_units(&data).collect();
    assert!(units.len() > 12, "asset too short to interrupt mid-stream");

    let mut param = SDecodingParam::default();
    param.uiTargetDqLayer = u8::MAX;
    param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
    param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;

    // --- the reference: a decoder that has only ever seen one session ------
    let mut fresh = Decoder::new();
    assert_eq!(fresh.initialize(&param), CM_RESULT_SUCCESS as c_long);
    let (fresh_frames, _) = decode_pass(&mut fresh, &units, units.len(), true);
    fresh.uninitialize();
    assert!(fresh_frames > 0, "the asset decoded nothing at all");

    // --- the subject: interrupted mid-stream, then re-initialised ---------
    let mut dec = Decoder::new();
    assert_eq!(dec.initialize(&param), CM_RESULT_SUCCESS as c_long);
    // Stop while the reordering buffer still holds pictures; a B-slice stream
    // buffers by construction.
    let (_, buffered) = decode_pass(&mut dec, &units, 12, false);
    assert!(
        buffered > 0,
        "nothing was buffered at the interruption — this probe no longer reaches the state it checks"
    );

    assert_eq!(dec.uninitialize(), CM_RESULT_SUCCESS as c_long);
    assert_eq!(dec.initialize(&param), CM_RESULT_SUCCESS as c_long);

    // The new session starts empty.
    let remaining = dec.frames_remaining();
    assert_eq!(
        remaining, 0,
        "the re-initialised decoder inherited {remaining} buffered picture(s) from the \
         previous session — DestroyPicBuff did not reset the reordering buffers"
    );

    // …and decodes the stream as a decoder that never saw the first pass.
    let (reinit_frames, _) = decode_pass(&mut dec, &units, units.len(), true);
    assert_eq!(
        reinit_frames, fresh_frames,
        "a re-initialised decoder emitted {reinit_frames} frames where a fresh one emits \
         {fresh_frames}"
    );

    dec.uninitialize();
}

/// The safe cores are part of the crate's public surface, not an internal detail
/// of the shells.
#[test]
pub(crate) fn test_safe_core_types_are_exported() {
    let _d: openh264rs::api::Decoder = openh264rs::api::Decoder::new();
    let _e: openh264rs::api::Encoder = openh264rs::api::Encoder::new();
    // And through the flat re-export the rest of this file uses.
    let _d2: Decoder = Decoder::default();
    let _e2: Encoder = Encoder::default();
}

#[test]
pub(crate) fn test_output_descriptors_validate_codec_and_bounds() {
    let mut enc = Encoder::new();
    let mut other_enc = Encoder::new();
    let mut enc_param = SEncParamBase::default();
    enc_param.iPicWidth = 64;
    enc_param.iPicHeight = 64;
    enc_param.fMaxFrameRate = 30.0;
    enc_param.iTargetBitrate = 100_000;
    enc_param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
    assert_eq!(enc.initialize(&enc_param), CM_RESULT_SUCCESS);
    assert_eq!(other_enc.initialize(&enc_param), CM_RESULT_SUCCESS);

    let yuv = vec![128u8; 64 * 64 * 3 / 2];
    let pic = SourcePicture::from_i420(&yuv, 64, 64);
    let mut bs_info = SFrameBSInfo::default();
    assert_eq!(enc.encode_frame(&pic, &mut bs_info), CM_RESULT_SUCCESS);
    assert!(!bs_info.layers()[0].bs_buf(&enc).is_empty());

    // Passing a different encoder instance rejects the foreign pointers.
    assert!(bs_info.layers()[0].bs_buf(&other_enc).is_empty());
    assert!(bs_info.layers()[0].nal_lengths(&other_enc).is_empty());

    // Mutating iNalCount (even by +1 within pre-allocated capacity) safely returns empty slices.
    let mut corrupted_bs = bs_info;
    corrupted_bs.sLayerInfo[0].iNalCount += 1;
    assert!(corrupted_bs.layers()[0].nal_lengths(&enc).is_empty());
    assert!(corrupted_bs.layers()[0].bs_buf(&enc).is_empty());

    let encoded_frame1: Vec<u8> = bs_info
        .layers()
        .iter()
        .flat_map(|l| l.bs_buf(&enc))
        .copied()
        .collect();

    // A second encode_frame (producing a P-frame) invalidates the earlier IDR SFrameBSInfo.
    let yuv2 = vec![160u8; 64 * 64 * 3 / 2];
    let pic2 = SourcePicture::from_i420(&yuv2, 64, 64).with_timestamp(33);
    let mut bs_info2 = SFrameBSInfo::default();
    assert_eq!(enc.encode_frame(&pic2, &mut bs_info2), CM_RESULT_SUCCESS);
    assert!(!bs_info2.layers()[0].bs_buf(&enc).is_empty());
    assert!(bs_info.layers()[0].nal_lengths(&enc).is_empty());
    assert!(bs_info.layers()[0].bs_buf(&enc).is_empty());

    let encoded_frame2: Vec<u8> = bs_info2
        .layers()
        .iter()
        .flat_map(|l| l.bs_buf(&enc))
        .copied()
        .collect();

    // After uninitialize(), stale descriptors return empty slices.
    enc.uninitialize();
    assert!(bs_info2.layers()[0].bs_buf(&enc).is_empty());
    assert!(bs_info2.layers()[0].nal_lengths(&enc).is_empty());

    // Decoder YUV plane validation:
    let mut dec = Decoder::new();
    let mut other_dec = Decoder::new();
    let mut dec_param = SDecodingParam::default();
    dec_param.uiTargetDqLayer = u8::MAX;
    assert_eq!(dec.initialize(&dec_param), CM_RESULT_SUCCESS as c_long);
    assert_eq!(
        other_dec.initialize(&dec_param),
        CM_RESULT_SUCCESS as c_long
    );

    let mut dst_info = SBufferInfo::default();
    assert_eq!(
        dec.decode_no_delay(Some(&encoded_frame1), &mut dst_info),
        DECODING_STATE::dsErrorFree
    );
    assert!(dst_info.yuv_planes(&dec).is_some());
    assert!(dst_info.yuv_planes(&other_dec).is_none());

    // Mutating dimensions out of bounds safely returns None.
    let mut corrupted_dst = dst_info;
    corrupted_dst.UsrData.sSystemBuffer.iHeight = 100_000;
    assert!(corrupted_dst.yuv_planes(&dec).is_none());

    // A second decode_no_delay advances to a new DPB slot and invalidates the first SBufferInfo.
    let mut dst_info2 = SBufferInfo::default();
    assert_eq!(
        dec.decode_no_delay(Some(&encoded_frame2), &mut dst_info2),
        DECODING_STATE::dsErrorFree
    );
    assert!(dst_info2.yuv_planes(&dec).is_some());
    assert!(dst_info.yuv_planes(&dec).is_none());

    // Flush when not at EOS resets dst_info2 and clears active output pointers.
    assert_eq!(dec.flush(&mut dst_info2), DECODING_STATE::dsErrorFree);
    assert_eq!(dst_info2.iBufferStatus, 0);
    assert!(dst_info2.yuv_planes(&dec).is_none());
}
