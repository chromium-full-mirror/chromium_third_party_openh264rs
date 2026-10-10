// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Malformed-stream error-handling and concealment tests.
//!
//! Exercises the decoder on corrupted inputs that standard conformance tests do not
//! reach: truncated NALs, emulation-prevention edges, degenerate NALs, and corrupt
//! NAL headers, verifying that the decoder does not panic/abort and returns the
//! expected `DECODING_STATE` error codes.

mod common;

use openh264rs::api::codec_api::*;
use openh264rs::split_annexb_units;

// ---------------------------------------------------------------------------
// Corpus knobs — the per-stream budget
// ---------------------------------------------------------------------------

/// Boundaries at the head of the stream that get the full ±[`FINE_DELTA`] sweep:
/// parameter sets and the first slices, which is where header parsing lives.
const FINE_HEAD: usize = 6;

/// Further boundaries, evenly spread over the rest of the stream and always including
/// the **last** one, that get the same sweep. A truncation at boundary *k* costs *k* NAL
/// decodes, so sweeping every boundary would be quadratic in the NAL count.
const FINE_SPREAD: usize = 6;

/// The sweep itself — every truncation length within ±8 bytes of the boundary.
const FINE_DELTA: i64 = 8;

/// Truncations evenly spaced across the whole stream body, on top of the boundary
/// sweeps: cheap coverage of mid-NAL truncation at arbitrary bit positions.
const COARSE_POINTS: usize = 16;

/// Emulation-prevention (`00 00 03`) sites probed, evenly spread through the stream.
const EPB_SITES: usize = 4;

/// Header-corruption and synthetic-tail variants run against a prefix this many
/// boundaries long, so their cost stays proportional to the head of the stream.
const PREFIX_BOUNDARIES: usize = 8;

/// Replacement bytes for the NAL header byte (`forbidden_zero_bit`, `nal_ref_idc`,
/// `nal_unit_type`) — chosen to hit the early-exit paths: reserved type 0, an
/// unreferenced IDR, SPS/PPS types on a slice, the unspecified type 31, a set
/// forbidden bit, and a referenced-IDR header on a non-IDR NAL.
const HEADER_BYTES: &[u8] = &[0x00, 0x05, 0x07, 0x08, 0x1F, 0x65, 0x80];

/// Recoverable bitstream/concealment `DECODING_STATE` bits expected on malformed streams:
/// `dsRefLost (0x02) | dsBitstreamError (0x04) | dsNoParamSets (0x10) | dsDataErrorConcealed (0x20)`.
const ALLOWED_ERROR_BITS: i32 = DECODING_STATE::dsRefLost.0
    | DECODING_STATE::dsBitstreamError.0
    | DECODING_STATE::dsNoParamSets.0
    | DECODING_STATE::dsDataErrorConcealed.0;

/// Error bits that every stream's mutation corpus must exercise:
/// `dsBitstreamError (0x04) | dsNoParamSets (0x10) | dsDataErrorConcealed (0x20)`.
const REQUIRED_STREAM_ERROR_BITS: i32 = DECODING_STATE::dsBitstreamError.0
    | DECODING_STATE::dsNoParamSets.0
    | DECODING_STATE::dsDataErrorConcealed.0;

/// The base streams: CAVLC and CABAC, PCM, B-frames, VUI and subset-SPS parsing, an
/// already-damaged pair, and one stream that switches resolution mid-stream.
const BASE_STREAMS: &[&str] = &[
    "SarVui.264",
    "Static.264",
    "sps_subsetsps_bothVUI.264",
    "CABA2_SVA_B.264",
    "CABA3_SVA_B.264",
    "Cisco_Men_whisper_640x320_CABAC_Bframe_9.264",
    "SVA_NL1_B.264",
    "QCIF_2P_I_allIPCM.264",
    "BA_MW_D.264",
    "BA_MW_D_IDR_LOST.264",
    "BA_MW_D_P_LOST.264",
    // Changes resolution mid-stream (352x288 -> 640x480 -> 352x288).
    "Error_I_P.264",
];

// ---------------------------------------------------------------------------
// Corpus construction
// ---------------------------------------------------------------------------

/// How a corpus entry reaches the decoder.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Feed {
    /// Split into Annex-B units and fed one NAL per `DecodeFrame2` call — the flow
    /// `decoder_conformance_test.rs` uses.
    AnnexB,
    /// One `DecodeFrame2` call with the exact bytes, start-code detection included.
    Raw,
}

struct Case {
    name: String,
    feed: Feed,
    data: Vec<u8>,
}

/// Byte offsets of the Annex-B start codes, by the same rules
/// [`split_annexb_units`] uses (3- and 4-byte prefixes, `i + 2 < len` scan bound).
/// `start_code_scan_agrees_with_split_annexb_units` pins the agreement.
fn scan_start_codes(data: &[u8]) -> Vec<usize> {
    let mut offsets = Vec::new();
    let len = data.len();
    let mut i = 0;
    while i + 2 < len {
        if data[i] == 0 && data[i + 1] == 0 {
            if data[i + 2] == 1 {
                offsets.push(i);
                i += 3;
                continue;
            } else if i + 3 < len && data[i + 2] == 0 && data[i + 3] == 1 {
                offsets.push(i);
                i += 4;
                continue;
            }
        }
        match data[i + 1..].iter().position(|&b| b == 0) {
            Some(pos) => i += 1 + pos,
            None => break,
        }
    }
    offsets
}

/// The boundary indices that get the fine sweep: the first [`FINE_HEAD`], plus
/// [`FINE_SPREAD`] evenly spread over the remainder with the last always included.
fn fine_boundary_indices(count: usize) -> Vec<usize> {
    let mut picked: Vec<usize> = (0..count.min(FINE_HEAD)).collect();
    if count > FINE_HEAD && FINE_SPREAD > 0 {
        let span = count - 1 - FINE_HEAD; // last index reachable by the spread
        let steps = FINE_SPREAD.max(2) - 1; // k = steps lands on the last boundary
        for k in 0..=steps.min(FINE_SPREAD - 1) {
            picked.push(FINE_HEAD + span * k / steps);
        }
    }
    picked.sort_unstable();
    picked.dedup();
    picked
}

/// Offsets of `00 00 03` sequences, evenly thinned to at most [`EPB_SITES`].
fn epb_sites(data: &[u8]) -> Vec<usize> {
    let all: Vec<usize> = data
        .windows(3)
        .enumerate()
        .filter(|(_, w)| w == &[0u8, 0, 3])
        .map(|(i, _)| i)
        .collect();
    if all.len() <= EPB_SITES {
        return all;
    }
    (0..EPB_SITES)
        .map(|k| all[(all.len() - 1) * k / (EPB_SITES - 1).max(1)])
        .collect()
}

fn build_corpus(data: &[u8], offsets: &[usize]) -> Vec<Case> {
    let mut cases: Vec<Case> = Vec::new();
    let mut seen_len: Vec<usize> = Vec::new();
    let total = data.len() as i64;

    let push_trunc = |cases: &mut Vec<Case>, seen: &mut Vec<usize>, name: String, len: i64| {
        if len < 0 || len > total {
            return;
        }
        let len = len as usize;
        if seen.contains(&len) {
            return;
        }
        seen.push(len);
        cases.push(Case {
            name,
            feed: Feed::AnnexB,
            data: data[..len].to_vec(),
        });
    };

    // (1) The fine sweep around selected NAL boundaries.
    for &b in &fine_boundary_indices(offsets.len()) {
        for d in -FINE_DELTA..=FINE_DELTA {
            push_trunc(
                &mut cases,
                &mut seen_len,
                format!("trunc.b{b:03}{d:+03}"),
                offsets[b] as i64 + d,
            );
        }
    }

    // (2) The coarse sweep across the body.
    for k in 1..=COARSE_POINTS {
        push_trunc(
            &mut cases,
            &mut seen_len,
            format!("trunc.coarse{k:02}"),
            total * k as i64 / (COARSE_POINTS as i64 + 1),
        );
    }

    // (3) Emulation-prevention edges: cut between `00 00` and the `03`, right after
    //     the `03`, and one byte past it.
    for (n, &s) in epb_sites(data).iter().enumerate() {
        push_trunc(
            &mut cases,
            &mut seen_len,
            format!("epb{n}.zz"),
            s as i64 + 2,
        );
        push_trunc(
            &mut cases,
            &mut seen_len,
            format!("epb{n}.at03"),
            s as i64 + 3,
        );
        push_trunc(
            &mut cases,
            &mut seen_len,
            format!("epb{n}.after"),
            s as i64 + 4,
        );
    }

    // (4) Synthetic tails on a bounded prefix: a stream ending in `00 00`, in
    //     `00 00 03`, in a bare start code, and in a start code plus one byte.
    let prefix_end = offsets
        .get(PREFIX_BOUNDARIES.min(offsets.len().saturating_sub(1)))
        .copied()
        .unwrap_or(data.len());
    let prefix = &data[..prefix_end];
    for (suffix_name, suffix) in [
        ("zz", &[0u8, 0][..]),
        ("zzz", &[0, 0, 0][..]),
        ("epb", &[0, 0, 3][..]),
        ("sc3", &[0, 0, 1][..]),
        ("sc4", &[0, 0, 0, 1][..]),
        ("sc3_byte", &[0, 0, 1, 0x65][..]),
    ] {
        let mut buf = prefix.to_vec();
        buf.extend_from_slice(suffix);
        cases.push(Case {
            name: format!("tail.{suffix_name}"),
            feed: Feed::AnnexB,
            data: buf,
        });
    }

    // (5) Header corruption on the same bounded prefix: the byte after the start
    //     code carries forbidden_zero_bit / nal_ref_idc / nal_unit_type, so these
    //     variants drive the early-exit paths.
    let sites: Vec<usize> = {
        let mut v: Vec<usize> = offsets.iter().take(2).copied().collect();
        if let Some(&vcl) = offsets.iter().find(|&&o| {
            let hdr = header_byte(data, o);
            matches!(hdr.map(|h| h & 0x1F), Some(1) | Some(5))
        }) {
            v.push(vcl);
        }
        v.retain(|&o| o < prefix_end);
        v.sort_unstable();
        v.dedup();
        v
    };
    for (n, &site) in sites.iter().enumerate() {
        let hdr_at = site + start_code_len(data, site);
        if hdr_at >= prefix.len() {
            continue;
        }
        for &byte in HEADER_BYTES {
            if prefix[hdr_at] == byte {
                continue;
            }
            let mut buf = prefix.to_vec();
            buf[hdr_at] = byte;
            cases.push(Case {
                name: format!("hdr{n}.{byte:02x}"),
                feed: Feed::AnnexB,
                data: buf,
            });
        }
    }

    cases
}

fn start_code_len(data: &[u8], offset: usize) -> usize {
    if data.get(offset + 2) == Some(&1) {
        3
    } else {
        4
    }
}

fn header_byte(data: &[u8], offset: usize) -> Option<u8> {
    data.get(offset + start_code_len(data, offset)).copied()
}

// ---------------------------------------------------------------------------
// Running one corpus entry
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Run {
    /// `(DECODING_STATE as i32, iBufferStatus)` per `DecodeFrame2`/`FlushFrame` call.
    calls: Vec<(i32, i32)>,
    frames: usize,
}

/// Hard cap on the flush loop.
const MAX_DRAIN: i32 = 24;

unsafe fn feed(decoder: *mut ISVCDecoder, unit: &[u8], run: &mut Run) {
    unsafe {
        let mut p_dst: [*mut u8; 3] = [std::ptr::null_mut(); 3];
        let mut buf_info = SBufferInfo::default();
        let src = if unit.is_empty() {
            std::ptr::null()
        } else {
            unit.as_ptr()
        };
        let ret = ISVCDecoder::DecodeFrame2(
            decoder,
            src,
            unit.len() as i32,
            p_dst.as_mut_ptr(),
            &mut buf_info,
        );
        record(ret.0, &buf_info, p_dst, run);
    }
}

unsafe fn record(ret: i32, buf_info: &SBufferInfo, dst: [*mut u8; 3], run: &mut Run) {
    assert_eq!(
        ret & !ALLOWED_ERROR_BITS,
        0,
        "unexpected DECODING_STATE bits in {:#x} ({:?})",
        ret,
        DECODING_STATE(ret)
    );
    assert!(
        buf_info.iBufferStatus == 0 || buf_info.iBufferStatus == 1,
        "unexpected iBufferStatus {}",
        buf_info.iBufferStatus
    );
    run.calls.push((ret, buf_info.iBufferStatus));
    if buf_info.iBufferStatus != 1 {
        return;
    }
    unsafe {
        let width = buf_info.UsrData.sSystemBuffer.iWidth;
        let height = buf_info.UsrData.sSystemBuffer.iHeight;
        let stride_y = buf_info.UsrData.sSystemBuffer.iStride[0] as usize;
        let stride_uv = buf_info.UsrData.sSystemBuffer.iStride[1] as usize;
        assert!(
            width > 0 && height > 0,
            "invalid dimensions {width}x{height}"
        );
        let (w, h) = (width as usize, height as usize);
        assert!(stride_y >= w, "stride_y {stride_y} < width {w}");
        assert!(
            stride_uv >= w / 2,
            "stride_uv {stride_uv} < chroma width {}",
            w / 2
        );
        verify_plane(dst[0], w, h, stride_y);
        verify_plane(dst[1], w / 2, h / 2, stride_uv);
        verify_plane(dst[2], w / 2, h / 2, stride_uv);
        run.frames += 1;
    }
}

unsafe fn verify_plane(plane: *const u8, width: usize, height: usize, stride: usize) {
    assert!(!plane.is_null(), "emitted frame has null plane pointer");
    unsafe {
        for y in 0..height {
            let row = std::slice::from_raw_parts(plane.add(y * stride), width);
            std::hint::black_box(row);
        }
    }
}

/// Drives one corpus entry through the C-API flow:
/// create → `Initialize` with `ERROR_CON_SLICE_COPY` → per-NAL `DecodeFrame2` →
/// EOS → drain → destroy.
fn decode_case(case: &Case) -> Run {
    unsafe {
        let mut decoder: *mut ISVCDecoder = std::ptr::null_mut();
        let ret = WelsCreateDecoder(&mut decoder);
        assert_eq!(ret, CM_RESULT_SUCCESS as i64, "WelsCreateDecoder");
        assert!(!decoder.is_null());

        let mut dec_param = SDecodingParam::default();
        dec_param.uiTargetDqLayer = u8::MAX;
        dec_param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
        dec_param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
        let init_ret = ISVCDecoder::Initialize(decoder, &dec_param as *const SDecodingParam);
        assert_eq!(init_ret, CM_RESULT_SUCCESS as i64, "Initialize");

        let mut run = Run::default();

        match case.feed {
            Feed::AnnexB => {
                for unit in split_annexb_units(&case.data) {
                    feed(decoder, unit, &mut run);
                }
            }
            Feed::Raw => feed(decoder, &case.data, &mut run),
        }

        // End of stream, then the null/zero-length call that flushes it.
        let mut eos_flag = 1i32;
        ISVCDecoder::SetOption(
            decoder,
            DECODER_OPTION::DECODER_OPTION_END_OF_STREAM,
            &mut eos_flag as *mut i32 as *mut std::ffi::c_void,
        );
        feed(decoder, &[], &mut run);

        let mut remaining = 0i32;
        ISVCDecoder::GetOption(
            decoder,
            DECODER_OPTION::DECODER_OPTION_NUM_OF_FRAMES_REMAINING_IN_BUFFER,
            &mut remaining as *mut i32 as *mut std::ffi::c_void,
        );
        for _ in 0..remaining.clamp(0, MAX_DRAIN) {
            let mut p_dst: [*mut u8; 3] = [std::ptr::null_mut(); 3];
            let mut buf_info = SBufferInfo::default();
            let flush_ret = ISVCDecoder::FlushFrame(decoder, p_dst.as_mut_ptr(), &mut buf_info);
            record(flush_ret.0, &buf_info, p_dst, &mut run);
        }

        ISVCDecoder::Uninitialize(decoder);
        WelsDestroyDecoder(decoder);

        run
    }
}

fn check_stream(name: &str) {
    let data = common::read_stream(name);
    let offsets = scan_start_codes(&data);
    assert!(!offsets.is_empty(), "{name} has no start codes");
    let cases = build_corpus(&data, &offsets);

    let mut seen_error_bits = 0i32;
    for case in &cases {
        let run = decode_case(case);
        let case_bits: i32 = run.calls.iter().fold(0, |acc, &(ret, _)| acc | ret);
        seen_error_bits |= case_bits;

        if case.name == "tail.sc3_byte" {
            assert_ne!(
                case_bits & DECODING_STATE::dsBitstreamError.0,
                0,
                "{name}: tail.sc3_byte (trailing 1-byte truncated IDR) must report dsBitstreamError, got {:?}",
                DECODING_STATE(case_bits)
            );
        } else if case.name == "hdr0.80" {
            assert_eq!(
                run.calls[0].0,
                DECODING_STATE::dsBitstreamError.0,
                "{name}: hdr0.80 (forbidden_zero_bit set on first NAL) must report dsBitstreamError on first call"
            );
            assert_ne!(
                case_bits & DECODING_STATE::dsNoParamSets.0,
                0,
                "{name}: hdr0.80 must report dsNoParamSets on subsequent NALs"
            );
        }
    }

    assert_eq!(
        seen_error_bits & REQUIRED_STREAM_ERROR_BITS,
        REQUIRED_STREAM_ERROR_BITS,
        "{name}: corpus did not exercise all expected error states (dsBitstreamError | dsNoParamSets | dsDataErrorConcealed); got {:?}",
        DECODING_STATE(seen_error_bits)
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The corpus builder walks start codes itself to get *offsets*; this pins that walk
/// against [`split_annexb_units`].
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn start_code_scan_agrees_with_split_annexb_units() {
    for &name in BASE_STREAMS {
        let data = common::read_stream(name);
        let offsets = scan_start_codes(&data);
        let units: Vec<_> = split_annexb_units(&data).collect();
        assert_eq!(offsets.len(), units.len(), "{name}: unit count");
        for (i, unit) in units.iter().enumerate() {
            let end = offsets.get(i + 1).copied().unwrap_or(data.len());
            assert_eq!(&data[offsets[i]..end], *unit, "{name}: unit {i}");
        }
    }
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn malformed_sarvui() {
    check_stream("SarVui.264");
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn malformed_static() {
    check_stream("Static.264");
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn malformed_sps_subsetsps_bothvui() {
    check_stream("sps_subsetsps_bothVUI.264");
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn malformed_caba2_sva_b() {
    check_stream("CABA2_SVA_B.264");
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn malformed_caba3_sva_b() {
    check_stream("CABA3_SVA_B.264");
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn malformed_cisco_men_whisper_cabac_bframe() {
    check_stream("Cisco_Men_whisper_640x320_CABAC_Bframe_9.264");
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn malformed_sva_nl1_b() {
    check_stream("SVA_NL1_B.264");
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn malformed_qcif_2p_i_allipcm() {
    check_stream("QCIF_2P_I_allIPCM.264");
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn malformed_ba_mw_d() {
    check_stream("BA_MW_D.264");
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn malformed_ba_mw_d_idr_lost() {
    check_stream("BA_MW_D_IDR_LOST.264");
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn malformed_ba_mw_d_p_lost() {
    check_stream("BA_MW_D_P_LOST.264");
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn malformed_error_i_p() {
    check_stream("Error_I_P.264");
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn malformed_narrow_16x16() {
    check_stream("narrow_16x16.264");
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn malformed_narrow_16x16_idr_lost() {
    check_stream("narrow_16x16_idr_lost.264");
}

fn check_degenerate_case(name: &str, feed: Feed, data: Vec<u8>, expected_rets: &[i32]) {
    let case = Case {
        name: name.to_string(),
        feed,
        data,
    };
    let run = decode_case(&case);
    let actual_rets: Vec<i32> = run.calls.iter().map(|&(r, _)| r).collect();
    assert_eq!(actual_rets, expected_rets, "case `{name}` return codes");
    assert_eq!(
        run.frames, 0,
        "case `{name}` should not emit complete frames"
    );
}

#[test]
fn malformed_degenerate_nals() {
    let ok = DECODING_STATE::dsErrorFree.0;
    let bs_err = DECODING_STATE::dsBitstreamError.0;
    let no_ps = DECODING_STATE::dsNoParamSets.0;

    // Empty input
    check_degenerate_case("raw.empty", Feed::Raw, vec![], &[ok, ok]);
    check_degenerate_case("annexb.empty", Feed::AnnexB, vec![], &[ok]);

    // Non-start-code bytes: Raw mode reports dsBitstreamError; AnnexB splitter drops them.
    for (label, bytes) in [
        ("z1", vec![0]),
        ("z2", vec![0, 0]),
        ("z3", vec![0, 0, 0]),
        ("epb_only", vec![0, 0, 3]),
    ] {
        check_degenerate_case(
            &format!("raw.{label}"),
            Feed::Raw,
            bytes.clone(),
            &[bs_err, ok],
        );
        check_degenerate_case(&format!("annexb.{label}"), Feed::AnnexB, bytes, &[ok]);
    }

    // Bare start codes and single non-SPS NAL header bytes -> dsNoParamSets.
    for (label, bytes) in [
        ("sc3_only", vec![0, 0, 1]),
        ("sc4_only", vec![0, 0, 0, 1]),
        ("sc3_slice_byte", vec![0, 0, 1, 0x65]),
        ("sc3_one_zero", vec![0, 0, 1, 0x00]),
        ("sc3_type31", vec![0, 0, 1, 0x1F]),
    ] {
        check_degenerate_case(
            &format!("raw.{label}"),
            Feed::Raw,
            bytes.clone(),
            &[no_ps, ok],
        );
        check_degenerate_case(
            &format!("annexb.{label}"),
            Feed::AnnexB,
            bytes,
            &[no_ps, ok],
        );
    }

    // Two back-to-back bare start codes.
    check_degenerate_case(
        "raw.sc3_zero_payload",
        Feed::Raw,
        vec![0, 0, 1, 0, 0, 1],
        &[no_ps, ok],
    );
    check_degenerate_case(
        "annexb.sc3_zero_payload",
        Feed::AnnexB,
        vec![0, 0, 1, 0, 0, 1],
        &[no_ps, no_ps, ok],
    );

    // Start code + lone SPS header byte (0x67).
    check_degenerate_case(
        "raw.sc3_sps_byte",
        Feed::Raw,
        vec![0, 0, 1, 0x67],
        &[ok, ok],
    );
    check_degenerate_case(
        "annexb.sc3_sps_byte",
        Feed::AnnexB,
        vec![0, 0, 1, 0x67],
        &[ok, ok],
    );
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn malformed_sarvui_parameter_sets() {
    let ok = DECODING_STATE::dsErrorFree.0;
    let bs_err = DECODING_STATE::dsBitstreamError.0;
    let no_ps = DECODING_STATE::dsNoParamSets.0;
    let concealed = DECODING_STATE::dsBitstreamError.0 | DECODING_STATE::dsDataErrorConcealed.0;

    // Parameter sets and slice lifted from SarVui.264.
    let data = common::read_stream("SarVui.264");
    let units = split_annexb_units(&data);
    let find = |ty: u8| -> Option<Vec<u8>> {
        units
            .clone()
            .find(|u| header_byte(u, 0).map(|h| h & 0x1F) == Some(ty))
            .map(|u| u.to_vec())
    };
    let sps = find(7).expect("SarVui.264 has an SPS");
    let pps = find(8).expect("SarVui.264 has a PPS");
    let slice = find(5).or_else(|| find(1)).expect("SarVui.264 has a slice");

    // Lone SPS is accepted without error; lone PPS or PPS-before-SPS reports dsNoParamSets.
    check_degenerate_case("raw.sps_only", Feed::Raw, sps.clone(), &[ok, ok]);
    check_degenerate_case("annexb.sps_only", Feed::AnnexB, sps.clone(), &[ok, ok]);
    check_degenerate_case("raw.pps_only", Feed::Raw, pps.clone(), &[no_ps, ok]);
    check_degenerate_case("annexb.pps_only", Feed::AnnexB, pps.clone(), &[no_ps, ok]);
    check_degenerate_case(
        "raw.pps_then_sps",
        Feed::Raw,
        [pps.clone(), sps.clone()].concat(),
        &[no_ps, ok],
    );
    check_degenerate_case(
        "annexb.pps_then_sps",
        Feed::AnnexB,
        [pps.clone(), sps.clone()].concat(),
        &[no_ps, ok, ok],
    );

    // Valid SPS + PPS followed by a slice truncated to 1..=5 bytes -> dsBitstreamError.
    for n in 1..=5 {
        let mut buf = [sps.clone(), pps.clone()].concat();
        buf.extend_from_slice(&slice[..(slice.len()).min(start_code_len(&slice, 0) + n)]);
        check_degenerate_case(
            &format!("raw.sps_pps_slice{n}"),
            Feed::Raw,
            buf.clone(),
            &[bs_err, ok],
        );
        check_degenerate_case(
            &format!("annexb.sps_pps_slice{n}"),
            Feed::AnnexB,
            buf,
            &[ok, ok, bs_err, ok],
        );
    }

    // Valid SPS + PPS followed by an 8-byte truncated slice -> reaches slice decode and
    // triggers error concealment at EOS (`dsBitstreamError | dsDataErrorConcealed`).
    {
        let mut buf = [sps.clone(), pps.clone()].concat();
        buf.extend_from_slice(&slice[..(slice.len()).min(start_code_len(&slice, 0) + 8)]);
        check_degenerate_case(
            "raw.sps_pps_slice8",
            Feed::Raw,
            buf.clone(),
            &[ok, concealed],
        );
        check_degenerate_case(
            "annexb.sps_pps_slice8",
            Feed::AnnexB,
            buf,
            &[ok, ok, ok, concealed],
        );
    }

    // Truncated SPS payloads: 1-byte (header only) is ignored; 2..=4 bytes report dsBitstreamError.
    let sps_cut1 = sps[..(start_code_len(&sps, 0) + 1).min(sps.len())].to_vec();
    check_degenerate_case("raw.sps_cut1", Feed::Raw, sps_cut1.clone(), &[ok, ok]);
    check_degenerate_case("annexb.sps_cut1", Feed::AnnexB, sps_cut1, &[ok, ok]);
    for n in 2..=4 {
        let cut = sps[..(start_code_len(&sps, 0) + n).min(sps.len())].to_vec();
        check_degenerate_case(
            &format!("raw.sps_cut{n}"),
            Feed::Raw,
            cut.clone(),
            &[bs_err, ok],
        );
        check_degenerate_case(
            &format!("annexb.sps_cut{n}"),
            Feed::AnnexB,
            cut,
            &[bs_err, ok],
        );
    }
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn mid_au_sps_dimension_increase_before_need_error_con_does_not_panic() {
    let small_data = common::read_stream("narrow_16x16.264");
    let large_data = common::read_stream("SarVui.264");

    let find_unit = |data: &[u8], ty: u8| -> Vec<u8> {
        split_annexb_units(data)
            .find(|u| header_byte(u, 0).map(|h| h & 0x1F) == Some(ty))
            .expect("expected NAL unit in stream")
            .to_vec()
    };

    let small_sps = find_unit(&small_data, 7);
    let small_pps = find_unit(&small_data, 8);
    let small_idr = find_unit(&small_data, 5);
    let large_sps = find_unit(&large_data, 7);

    // Step 1: Feed SPS(1x1 MB), PPS, and a duplicated IDR slice in the same access unit.
    // Both IDR slices share identical slice headers (`first_mb_in_slice == 0`), so they stay
    // in one AU (`iTotalNumMbRec == 2 != 1`) and `DecodeFrameNoDelay`'s instant-decode pass
    // returns `ERR_INFO_MB_NUM_INADEQUATE` while leaving `iTotalNumMbRec == 2` and
    // `pCurDqLayer.grid` allocated at 1x1 MB (`len == 1`) with its decoded flag `true`.
    let mut au1 = Vec::new();
    au1.extend_from_slice(&small_sps);
    au1.extend_from_slice(&small_pps);
    au1.extend_from_slice(&small_idr);
    au1.extend_from_slice(&small_idr);

    // Step 2: Feed a larger SPS(id=0) followed by an AUD NAL (`00 00 00 01 09 10`).
    // On the SPS NAL, `CheckAndFinishLastPic` runs while `sSpsBuffer[0]` is still 1x1 (1 MB),
    // so `NeedErrorCon` returns `false`, `iTotalNumMbRec` stays non-zero, and
    // `DecodeFinishUpdate` overwrites `sSpsBuffer[0]` with the larger SPS.
    // On the AUD NAL, `CheckAndFinishLastPic` calls `NeedErrorCon` with `active_sps` at the
    // larger dimensions while `pCurDqLayer.grid` is still 1x1 (`1` MB).
    let mut au2 = Vec::new();
    au2.extend_from_slice(&large_sps);
    au2.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x09, 0x10]);

    for ec in [
        ERROR_CON_IDC::ERROR_CON_DISABLE,
        ERROR_CON_IDC::ERROR_CON_FRAME_COPY,
        ERROR_CON_IDC::ERROR_CON_SLICE_COPY,
        ERROR_CON_IDC::ERROR_CON_SLICE_COPY_CROSS_IDR,
        ERROR_CON_IDC::ERROR_CON_SLICE_COPY_CROSS_IDR_FREEZE_RES_CHANGE,
        ERROR_CON_IDC::ERROR_CON_SLICE_MV_COPY_CROSS_IDR,
        ERROR_CON_IDC::ERROR_CON_SLICE_MV_COPY_CROSS_IDR_FREEZE_RES_CHANGE,
    ] {
        unsafe {
            let mut decoder: *mut ISVCDecoder = std::ptr::null_mut();
            assert_eq!(WelsCreateDecoder(&mut decoder), CM_RESULT_SUCCESS as i64);
            let mut dec_param = SDecodingParam::default();
            dec_param.uiTargetDqLayer = u8::MAX;
            dec_param.eEcActiveIdc = ec;
            dec_param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
            assert_eq!(
                ISVCDecoder::Initialize(decoder, &dec_param as *const SDecodingParam),
                CM_RESULT_SUCCESS as i64
            );

            let mut p_dst: [*mut u8; 3] = [std::ptr::null_mut(); 3];
            let mut buf_info = SBufferInfo::default();
            let _ = ISVCDecoder::DecodeFrameNoDelay(
                decoder,
                au1.as_ptr(),
                au1.len() as i32,
                p_dst.as_mut_ptr(),
                &mut buf_info,
            );
            let _ = ISVCDecoder::DecodeFrameNoDelay(
                decoder,
                au2.as_ptr(),
                au2.len() as i32,
                p_dst.as_mut_ptr(),
                &mut buf_info,
            );

            ISVCDecoder::Uninitialize(decoder);
            WelsDestroyDecoder(decoder);
        }
    }
}
