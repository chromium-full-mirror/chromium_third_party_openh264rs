// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! `DecodeFrameNoDelay` is a different entry point from `DecodeFrame2`.
//!
//! `welsDecoderExt.cpp:720–725`, the whole of the single-threaded body:
//!
//! ```c
//! iRet  = DecodeFrame2 (kpSrc, kiSrcLen, ppDst, pDstInfo);
//! iRet |= DecodeFrame2 (NULL, 0, ppDst, pDstInfo);
//! ```
//!
//! The second call is what "no delay" means: it forces reconstruction so the caller
//! gets its frame on the call that fed the access unit.
//!
//! The flow is an annex-B split, `ERROR_CON_SLICE_COPY`, one NAL per call, EOS, a final
//! `DecodeFrameNoDelay(NULL, 0)`, then `FlushFrame` for what `GetOption` reports
//! remaining, capped at 24.
//!
//! `Error_I_P.264` emits one frame here against `DecodeFrame2`'s five, and
//! `QCIF_2P_I_allIPCM.264` two against a different call pattern: when the first call
//! emits a picture and the second does not, the second call's `iBufferStatus = 0`
//! overwrites it and the frame is lost to that caller.

use openh264rs::api::codec_api::*;
use openh264rs::split_annexb_units;

#[path = "common/mod.rs"]
mod common;
use common::Sha1Hasher;

/// `(asset, frames, first-frame dims, sha1 over every emitted plane, codes, buffer
/// statuses)`.
struct Row {
    asset: &'static str,
    frames: usize,
    dims: (i32, i32),
    sha1: &'static str,
    codes: &'static [i32],
    bufs: &'static [i32],
}

/// Diversity over the axes that change emission timing: CAVLC, CABAC with B-frames, a
/// one-macroblock frame, all-IPCM, and the resolution-change stream.
const ROWS: &[Row] = &[
    Row {
        asset: "BA_MW_D.264",
        frames: 100,
        dims: (176, 144),
        sha1: "afd7a9765961ca241bb4bdf344b31397bec7465a",
        codes: &[0; 103],
        bufs: &[
            0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        ],
    },
    Row {
        asset: "Cisco_Men_whisper_640x320_CABAC_Bframe_9.264",
        frames: 9,
        dims: (640, 320),
        sha1: "2b349c1bc806b6e0412008747b2463d77b576476",
        codes: &[0; 20],
        bufs: &[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    },
    Row {
        asset: "narrow_16x16.264",
        frames: 24,
        dims: (16, 16),
        sha1: "6299ce8a7dc8a86d367dca65ca123eb499fc5ca8",
        codes: &[0; 31],
        bufs: &[
            0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            1, 0,
        ],
    },
    Row {
        asset: "QCIF_2P_I_allIPCM.264",
        frames: 2,
        dims: (176, 144),
        sha1: "8724c0866ebdba7ebb7209a0c0c3ae3ae38a0240",
        codes: &[0; 7],
        bufs: &[0, 0, 0, 0, 0, 1, 1],
    },
    Row {
        asset: "Error_I_P.264",
        frames: 1,
        dims: (640, 480),
        sha1: "5c1f742798b2c1061cb83ab2f2cddd7929b2fb4e",
        codes: &[
            0x0, 0x0, 0x0, 0x20, 0x0, 0x0, 0x20, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x20, 0x20, 0x20,
            0x0,
        ],
        bufs: &[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    },
];

/// Drives one asset through `DecodeFrameNoDelay`.
fn nodelay_row(data: &[u8]) -> (usize, (i32, i32), String, Vec<i32>, Vec<i32>) {
    let mut dec = Decoder::new();
    let mut param = SDecodingParam::default();
    param.uiTargetDqLayer = u8::MAX;
    param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
    param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
    assert_eq!(
        dec.initialize(&param),
        CM_RESULT_SUCCESS as std::ffi::c_long
    );

    let mut hasher = Sha1Hasher::new();
    let mut frames = 0usize;
    let mut first = (0i32, 0i32);
    let mut codes = Vec::new();
    let mut bufs = Vec::new();

    let take = |dec: &Decoder,
                info: &SBufferInfo,
                hasher: &mut Sha1Hasher,
                frames: &mut usize,
                first: &mut (i32, i32)| {
        if info.iBufferStatus != 1 {
            return;
        }
        let sys = *info.UsrData.sys();
        for row in info
            .y_rows(dec)
            .chain(info.u_rows(dec))
            .chain(info.v_rows(dec))
        {
            hasher.update(row);
        }
        *frames += 1;
        if first.0 == 0 {
            *first = (sys.iWidth, sys.iHeight);
        }
    };

    for unit in split_annexb_units(data) {
        let mut info = SBufferInfo::default();
        let st = dec.decode_no_delay(Some(unit), &mut info);
        codes.push(st.0);
        bufs.push(info.iBufferStatus);
        take(&dec, &info, &mut hasher, &mut frames, &mut first);
    }

    dec.set_end_of_stream(true);
    {
        let mut info = SBufferInfo::default();
        let st = dec.decode_no_delay(None, &mut info);
        codes.push(st.0);
        bufs.push(info.iBufferStatus);
        take(&dec, &info, &mut hasher, &mut frames, &mut first);
    }

    let remaining = dec.frames_remaining();
    for _ in 0..remaining.clamp(0, 24) {
        let mut info = SBufferInfo::default();
        let st = dec.flush(&mut info);
        codes.push(st.0);
        bufs.push(info.iBufferStatus);
        take(&dec, &info, &mut hasher, &mut frames, &mut first);
    }

    dec.uninitialize();
    (frames, first, hasher.digest(), codes, bufs)
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn decode_frame_no_delay_matches_the_reference_on_every_axis() {
    for row in ROWS {
        let data = common::read_stream(row.asset);
        let (frames, dims, sha1, codes, bufs) = nodelay_row(&data);
        // Counts before hashes: a mismatch whose count also moved is a different
        // defect from one whose count held.
        assert_eq!(frames, row.frames, "{}: emitted frame count", row.asset);
        assert_eq!(
            dims, row.dims,
            "{}: first emitted frame's dimensions",
            row.asset
        );
        assert_eq!(
            codes, row.codes,
            "{}: DecodeFrameNoDelay return codes, in call order",
            row.asset
        );
        assert_eq!(
            bufs, row.bufs,
            "{}: iBufferStatus, in call order",
            row.asset
        );
        assert_eq!(
            sha1, row.sha1,
            "{}: SHA-1 over every emitted plane",
            row.asset
        );
    }
}

/// Runs one full pass + EOS flush on an already-initialized `Decoder`, returning
/// `(emitted_frames, flushed_frames, yuv_sha1)`.
fn decode_and_flush_pass(dec: &mut Decoder, data: &[u8]) -> (usize, i32, String) {
    let mut hasher = Sha1Hasher::new();
    let mut frames = 0usize;

    let mut take = |dec: &Decoder, info: &SBufferInfo| {
        if info.iBufferStatus != 1 {
            return;
        }
        for row in info
            .y_rows(dec)
            .chain(info.u_rows(dec))
            .chain(info.v_rows(dec))
        {
            hasher.update(row);
        }
        frames += 1;
    };

    for (idx, unit) in split_annexb_units(data).enumerate() {
        let mut info = SBufferInfo::default();
        let st = dec.decode(Some(unit), &mut info);
        assert_eq!(st, DECODING_STATE::dsErrorFree, "decode(Some) failed at unit {idx}");
        take(dec, &info);

        let mut info = SBufferInfo::default();
        let st = dec.decode(None, &mut info);
        assert_eq!(st, DECODING_STATE::dsErrorFree, "decode(None) failed at unit {idx}");
        take(dec, &info);
    }

    dec.set_end_of_stream(true);
    let mut info = SBufferInfo::default();
    let st = dec.decode(None, &mut info);
    assert_eq!(st, DECODING_STATE::dsErrorFree, "EOS decode(None) failed");
    take(dec, &info);

    let remaining = dec.frames_remaining();
    for i in 0..remaining {
        let mut info = SBufferInfo::default();
        let st = dec.flush(&mut info);
        assert_eq!(st, DECODING_STATE::dsErrorFree, "flush {i} failed");
        assert_eq!(info.iBufferStatus, 1, "flush {i} must emit a buffered picture");
        take(dec, &info);
    }
    assert_eq!(dec.frames_remaining(), 0, "reordering buffer must be empty after flush");
    dec.set_end_of_stream(false);

    (frames, remaining, hasher.digest())
}

/// Regression test for https://issues.chromium.org/issues/586589228:
/// pictures emitted by `flush()` must release their reordering-buffer reference in
/// `PicPool` so the same `Decoder` can keep decoding across EOS boundaries without
/// exhausting the pool (`dsOutOfMemory`).
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn decoder_can_be_reused_across_multiple_eos_flushes() {
    let cases = [
        ("CABA3_SVA_B.264", "1b811ad90ac4ce9beb034cb8c993ad4e2935f46a"),
        ("CACQP3_Sony_D.jsv", "2ab41abd239c1c556e91ff027f47002001ee8605"),
        (
            "Cisco_Men_whisper_640x320_CABAC_Bframe_9.264",
            "2b349c1bc806b6e0412008747b2463d77b576476",
        ),
        (
            "Cisco_Adobe_PDF_sample_a_1024x768_CAVLC_Bframe_9.264",
            "9d758d9e6f4dead0d7b361f3ddf2ee009d0ea190",
        ),
    ];

    for (asset, expected_sha1) in cases {
        let data = common::read_stream(asset);
        let mut dec = Decoder::new();
        let mut param = SDecodingParam::default();
        param.uiTargetDqLayer = u8::MAX;
        param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
        param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
        assert_eq!(dec.initialize(&param), CM_RESULT_SUCCESS as std::ffi::c_long);

        let (frames0, flushed0, sha0) = decode_and_flush_pass(&mut dec, &data);
        assert!(flushed0 > 0, "{asset}: expected non-empty EOS flush");
        assert_eq!(sha0, expected_sha1, "{asset} pass 0 SHA-1");

        // Decode two more full passes on the same `Decoder` without re-initializing,
        // ensuring total flushed frames far exceed `PicPool` capacity.
        for pass in 1..=2 {
            let (frames, flushed, sha) = decode_and_flush_pass(&mut dec, &data);
            assert_eq!(frames, frames0, "{asset} pass {pass}: frame count");
            assert_eq!(flushed, flushed0, "{asset} pass {pass}: flushed count");
            assert_eq!(sha, expected_sha1, "{asset} pass {pass}: SHA-1");
        }

        dec.uninitialize();
    }

    // Also verify cross-stream reuse (switching streams & resolutions after EOS flush).
    let mut dec = Decoder::new();
    let mut param = SDecodingParam::default();
    param.uiTargetDqLayer = u8::MAX;
    param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
    param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
    assert_eq!(dec.initialize(&param), CM_RESULT_SUCCESS as std::ffi::c_long);

    for (asset, expected_sha1) in cases.iter().chain(cases.iter().rev()) {
        let data = common::read_stream(asset);
        let (_, flushed, sha) = decode_and_flush_pass(&mut dec, &data);
        assert!(flushed > 0, "{asset}: expected non-empty EOS flush");
        assert_eq!(sha, *expected_sha1, "cross-stream {asset}: SHA-1");
    }
    dec.uninitialize();
}

