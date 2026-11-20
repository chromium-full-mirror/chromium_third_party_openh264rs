// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Shared decoder conformance drivers and Annex B access-unit splitter.
// `c_long` is i64 on 64-bit Unix and i32 on Windows; keep the explicit cast.
#![allow(clippy::useless_conversion)]

use super::Sha1Hasher;
use openh264rs::api::codec_api::*;
use openh264rs::split_annexb_units;

/// Outcome of decoding a bitstream through one of the conformance feed-mode drivers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeStreamResult {
    pub sha1: String,
    pub frames: usize,
    pub state_union: DECODING_STATE,
}

/// Appends the cropped I420 planes (`Y`, then `U`, then `V`) of `buf_info` into `hasher`.
pub fn update_hash_from_frame(hasher: &mut Sha1Hasher, dec: &Decoder, buf_info: &SBufferInfo) {
    for row in buf_info
        .y_rows(dec)
        .chain(buf_info.u_rows(dec))
        .chain(buf_info.v_rows(dec))
    {
        hasher.update(row);
    }
}

/// Splits an Annex B byte stream into access-unit slices built on [`split_annexb_units`].
///
/// A new access unit begins at the first NAL of type 6..=9 or 14..=18 following a slice,
/// or at a slice (`nal_unit_type` in `{1, 2, 5}`) with `first_mb_in_slice == 0` following
/// a slice. That simplified rule assumes no redundant pictures and no arbitrary slice
/// order (ASO), which holds for all committed conformance streams; callers should assert
/// that the resulting unit count matches the expected frame count.
pub fn split_access_units(bitstream: &[u8]) -> Vec<&[u8]> {
    let base = bitstream.as_ptr() as usize;
    let mut aus = Vec::new();
    let mut au_start: Option<usize> = None;
    let mut has_slice = false;

    for unit in split_annexb_units(bitstream) {
        let start = unit.as_ptr() as usize - base;
        let payload = if unit.starts_with(&[0, 0, 1]) {
            &unit[3..]
        } else if unit.starts_with(&[0, 0, 0, 1]) {
            &unit[4..]
        } else {
            unit
        };
        let au_s = *au_start.get_or_insert(start);
        if let Some(&hdr) = payload.first() {
            let nal_type = hdr & 0x1f;
            let is_slice_hdr = matches!(nal_type, 1 | 2 | 5);
            // In Exp-Golomb (`ue(v)`), `codeNum == 0` (`first_mb_in_slice == 0`) is a
            // single `1` bit at the MSB of the first RBSP byte after the 1-byte NAL
            // header (too early in the NAL for `00 00 03` emulation prevention).
            let first_mb_zero = is_slice_hdr && payload.get(1).is_some_and(|&b| b & 0x80 != 0);
            let starts_new_au = has_slice
                && (matches!(nal_type, 6..=9 | 14..=18) || (is_slice_hdr && first_mb_zero));
            if starts_new_au {
                aus.push(&bitstream[au_s..start]);
                au_start = Some(start);
                has_slice = matches!(nal_type, 1..=5);
            } else if matches!(nal_type, 1..=5) {
                has_slice = true;
            }
        }
    }
    if let Some(au_s) = au_start {
        aus.push(&bitstream[au_s..bitstream.len()]);
    }
    aus
}

fn init_conformance_decoder(label: &str) -> Decoder {
    let mut decoder = Decoder::new();
    let mut dec_param = SDecodingParam::default();
    dec_param.uiTargetDqLayer = u8::MAX;
    dec_param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
    dec_param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;

    let init_ret = decoder.initialize(&dec_param);
    assert_eq!(
        init_ret, CM_RESULT_SUCCESS as std::ffi::c_long,
        "Failed to initialize decoder for {label}"
    );
    decoder
}

#[inline]
fn record_step(
    decoder: &Decoder,
    dec_ret: DECODING_STATE,
    buf_info: &SBufferInfo,
    hash_concealed: bool,
    hasher: &mut Sha1Hasher,
    decoded_frames: &mut usize,
    state_union: &mut DECODING_STATE,
) {
    state_union.0 |= dec_ret.0;
    if (hash_concealed || dec_ret == DECODING_STATE::dsErrorFree) && buf_info.iBufferStatus == 1 {
        update_hash_from_frame(hasher, decoder, buf_info);
        *decoded_frames += 1;
    }
}

/// Decodes an Annex B bitstream one NAL at a time, following each NAL with a `None`
/// `decode` call (`h264dec.cpp:418-430`), then flushes at end of stream.
pub fn decode_bytes_per_nal(data: &[u8], label: &str, hash_concealed: bool) -> DecodeStreamResult {
    assert!(!data.is_empty(), "Bitstream asset {label} is empty");
    let mut decoder = init_conformance_decoder(label);
    let mut hasher = Sha1Hasher::new();
    let mut decoded_frames = 0;
    let mut state_union = DECODING_STATE::dsErrorFree;

    for unit in split_annexb_units(data) {
        let mut buf_info = SBufferInfo::default();
        let dec_ret = decoder.decode(Some(unit), &mut buf_info);
        record_step(
            &decoder,
            dec_ret,
            &buf_info,
            hash_concealed,
            &mut hasher,
            &mut decoded_frames,
            &mut state_union,
        );

        // A null `DecodeFrame2` after every NAL (`h264dec.cpp:418-430`)
        // constructs the access unit the NAL completed instead of leaving it
        // pending.
        let mut buf_info = SBufferInfo::default();
        let dec_ret = decoder.decode(None, &mut buf_info);
        record_step(
            &decoder,
            dec_ret,
            &buf_info,
            hash_concealed,
            &mut hasher,
            &mut decoded_frames,
            &mut state_union,
        );
    }

    decoder.set_end_of_stream(true);

    let mut buf_info = SBufferInfo::default();
    let dec_ret = decoder.decode(None, &mut buf_info);
    record_step(
        &decoder,
        dec_ret,
        &buf_info,
        hash_concealed,
        &mut hasher,
        &mut decoded_frames,
        &mut state_union,
    );

    let remaining_frames = decoder.frames_remaining();
    for _ in 0..remaining_frames {
        let mut buf_info = SBufferInfo::default();
        let flush_ret = decoder.flush(&mut buf_info);
        record_step(
            &decoder,
            flush_ret,
            &buf_info,
            hash_concealed,
            &mut hasher,
            &mut decoded_frames,
            &mut state_union,
        );
    }

    decoder.uninitialize();
    DecodeStreamResult {
        sha1: hasher.digest(),
        frames: decoded_frames,
        state_union,
    }
}

/// Reads a named stream from [`super::stream_dir`] and decodes it via
/// [`decode_bytes_per_nal`], returning the SHA-1 digest, frame count, and ORed states.
pub fn test_single_bitstream_asset_ex(file_name: &str, hash_concealed: bool) -> DecodeStreamResult {
    let data = super::read_stream(file_name);
    decode_bytes_per_nal(&data, file_name, hash_concealed)
}

/// Decodes pre-split access units one whole access unit per [`Decoder::decode`] call
/// (with no intermediate `None` calls), then flushes at end of stream.
pub fn decode_access_units(
    access_units: &[&[u8]],
    label: &str,
    hash_concealed: bool,
) -> DecodeStreamResult {
    assert!(
        !access_units.is_empty(),
        "No access units split for bitstream {label}"
    );
    let mut decoder = init_conformance_decoder(label);
    let mut hasher = Sha1Hasher::new();
    let mut decoded_frames = 0;
    let mut state_union = DECODING_STATE::dsErrorFree;

    for au in access_units {
        let mut buf_info = SBufferInfo::default();
        let dec_ret = decoder.decode(Some(au), &mut buf_info);
        record_step(
            &decoder,
            dec_ret,
            &buf_info,
            hash_concealed,
            &mut hasher,
            &mut decoded_frames,
            &mut state_union,
        );
    }

    decoder.set_end_of_stream(true);

    let mut buf_info = SBufferInfo::default();
    let dec_ret = decoder.decode(None, &mut buf_info);
    record_step(
        &decoder,
        dec_ret,
        &buf_info,
        hash_concealed,
        &mut hasher,
        &mut decoded_frames,
        &mut state_union,
    );

    let remaining_frames = decoder.frames_remaining();
    for _ in 0..remaining_frames {
        let mut buf_info = SBufferInfo::default();
        let flush_ret = decoder.flush(&mut buf_info);
        record_step(
            &decoder,
            flush_ret,
            &buf_info,
            hash_concealed,
            &mut hasher,
            &mut decoded_frames,
            &mut state_union,
        );
    }

    decoder.uninitialize();
    DecodeStreamResult {
        sha1: hasher.digest(),
        frames: decoded_frames,
        state_union,
    }
}

/// Decodes pre-split access units one access unit per [`Decoder::decode_no_delay`] call,
/// then flushes at end of stream.
pub fn decode_access_units_no_delay(
    access_units: &[&[u8]],
    label: &str,
    hash_concealed: bool,
) -> DecodeStreamResult {
    assert!(
        !access_units.is_empty(),
        "No access units split for bitstream {label}"
    );
    let mut decoder = init_conformance_decoder(label);
    let mut hasher = Sha1Hasher::new();
    let mut decoded_frames = 0;
    let mut state_union = DECODING_STATE::dsErrorFree;

    for au in access_units {
        let mut buf_info = SBufferInfo::default();
        let dec_ret = decoder.decode_no_delay(Some(au), &mut buf_info);
        record_step(
            &decoder,
            dec_ret,
            &buf_info,
            hash_concealed,
            &mut hasher,
            &mut decoded_frames,
            &mut state_union,
        );
    }

    decoder.set_end_of_stream(true);

    let mut buf_info = SBufferInfo::default();
    let dec_ret = decoder.decode_no_delay(None, &mut buf_info);
    record_step(
        &decoder,
        dec_ret,
        &buf_info,
        hash_concealed,
        &mut hasher,
        &mut decoded_frames,
        &mut state_union,
    );

    let remaining_frames = decoder.frames_remaining();
    for _ in 0..remaining_frames {
        let mut buf_info = SBufferInfo::default();
        let flush_ret = decoder.flush(&mut buf_info);
        record_step(
            &decoder,
            flush_ret,
            &buf_info,
            hash_concealed,
            &mut hasher,
            &mut decoded_frames,
            &mut state_union,
        );
    }

    decoder.uninitialize();
    DecodeStreamResult {
        sha1: hasher.digest(),
        frames: decoded_frames,
        state_union,
    }
}
