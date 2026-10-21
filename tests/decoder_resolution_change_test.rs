// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! A mid-stream resolution change must reallocate the picture pool. The stream
//! decoded here runs 352x288 → 640x480 → 352x288.

use openh264rs::api::codec_api::*;
use openh264rs::split_annexb_units;

#[path = "common/mod.rs"]
mod common;
use common::Sha1Hasher;

/// Expected output per emitted frame: `(width, height, sha1-of-planes)`. The two
/// 640x480 frames are only reached once the pool is reallocated.
const CPP_FRAMES: &[(i32, i32, &str)] = &[
    (352, 288, "0f786183de107a429a903cdd838d08268ea34bbe"),
    (352, 288, "477e333c677615c2e3e369793358f327ca4fac61"),
    (640, 480, "5c1f742798b2c1061cb83ab2f2cddd7929b2fb4e"),
    (640, 480, "60270f979f8e2ccd662c05f58d0775271f90a4cd"),
    (352, 288, "b1e052919cf52ecff84fa100eeddaf8c762957c4"),
];

/// `DecodeFrame2` return code per call, in call order; the last call is the
/// end-of-stream drain. `0x20` is `dsDataErrorConcealed`.
const CPP_CODES: &[i32] = &[
    0x0, 0x0, 0x0, 0x20, 0x0, 0x0, 0x20, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x20, 0x20, 0x0, 0x20,
];

/// `iBufferStatus` per call.
const CPP_BUFS: &[i32] = &[0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0];

/// SHA-1 over the three planes at the strides `UsrData` reports.
///
/// # Safety
/// `dst` must be the plane pointers `DecodeFrame2` just wrote with
/// `iBufferStatus == 1`, valid for the dimensions and strides in `info`.
fn hash_frame(dec: &Decoder, info: &SBufferInfo) -> (i32, i32, String) {
    let sys = info.UsrData.sys();
    let mut hasher = Sha1Hasher::new();
    for row in info
        .y_rows(dec)
        .chain(info.u_rows(dec))
        .chain(info.v_rows(dec))
    {
        hasher.update(row);
    }
    (sys.iWidth, sys.iHeight, hasher.digest())
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn resolution_change_stream_matches_the_reference_and_does_not_abort() {
    let data = common::read_stream("Error_I_P.264");

    let mut codes = Vec::new();
    let mut bufs = Vec::new();
    let mut frames: Vec<(i32, i32, String)> = Vec::new();

    // Annex-B split, `ERROR_CON_SLICE_COPY`, one NAL per call, then the
    // end-of-stream drain.
    let mut decoder = Decoder::new();
    let mut param = SDecodingParam::default();
    param.uiTargetDqLayer = u8::MAX;
    param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
    param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
    assert_eq!(
        decoder.initialize(&param),
        CM_RESULT_SUCCESS as std::ffi::c_long
    );

    let mut feed = |decoder: &mut Decoder, unit: Option<&[u8]>| {
        let mut buf_info = SBufferInfo::default();
        let ret = decoder.decode(unit, &mut buf_info);
        codes.push(ret.0);
        bufs.push(buf_info.iBufferStatus);
        if buf_info.iBufferStatus == 1 {
            frames.push(hash_frame(decoder, &buf_info));
        }
    };
    for unit in split_annexb_units(&data) {
        feed(&mut decoder, Some(unit));
    }
    decoder.set_end_of_stream(true);
    feed(&mut decoder, None);

    decoder.uninitialize();

    assert_eq!(
        codes, CPP_CODES,
        "DecodeFrame2 return codes must match the C++ decoder's"
    );
    assert_eq!(
        bufs, CPP_BUFS,
        "iBufferStatus per call must match the C++ decoder's"
    );
    assert_eq!(frames.len(), CPP_FRAMES.len(), "emitted frame count");
    for (i, (got, want)) in frames.iter().zip(CPP_FRAMES).enumerate() {
        assert_eq!(
            (got.0, got.1, got.2.as_str()),
            (want.0, want.1, want.2),
            "frame {i}: dimensions and plane hash must match the C++ decoder's"
        );
    }
}
