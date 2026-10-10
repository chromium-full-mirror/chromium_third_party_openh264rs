// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Tests for `ISVCDecoder::DecodeParser` (`bParseOnly = true`).
//!
//! `DecodeParser` parses NAL units and emits normalized Annex-B access units in
//! `SParserBsInfo` (prepending active SPS/PPS in front of IDR slices) rather than
//! decoding macroblocks into YUV planes.

use openh264rs::api::codec_api::*;
use openh264rs::split_annexb_units;

#[path = "common/mod.rs"]
mod common;
use common::Sha1Hasher;

/// `(asset_stem, expected_calls, expected_emitted_aus, expected_bitstream_sha1)`
const EXPECTED_PARSEONLY: &[(&str, usize, usize, &str)] = &[
    (
        "BA_MW_D",
        103,
        100,
        "9b6dbd6ec5a4506b0ba0e5ae93772292b3e16a92",
    ),
    (
        "Cisco_Men_whisper_640x320_CABAC_Bframe_9",
        12,
        9,
        "78e021f4fa0e470c3d117647a4b643c74703a54d",
    ),
    (
        "QCIF_2P_I_allIPCM",
        5,
        2,
        "fe8af4d2d9bcdf7935e4b6eb312ccd596dc27f43",
    ),
    (
        "grid_48x32",
        10,
        6,
        "effba4ac9db56bf676edd01732bea75f60de1197",
    ),
    (
        "fmo_2groups_64x64",
        5,
        1,
        "f56770379754860404223a4f49ff08845cb78e59",
    ),
    (
        "sps_subsetsps_bothVUI",
        6,
        1,
        "549b30692d4e03b665da39913b770727719a353a",
    ),
    (
        "Error_I_P",
        17,
        1,
        "f50c3a36ac5960d75e6fc615af216cf9c30fbdbd",
    ),
];

/// Drives one asset through `DecodeParser`, one NAL per call, returning
/// `(total_calls, emitted_aus, cumulative_sha1)`.
unsafe fn run_parseonly(asset: &str, data: &[u8]) -> (usize, usize, String) {
    unsafe {
        let mut dec: *mut ISVCDecoder = std::ptr::null_mut();
        assert_eq!(WelsCreateDecoder(&mut dec), CM_RESULT_SUCCESS as i64);
        let mut param = SDecodingParam::default();
        param.uiTargetDqLayer = u8::MAX;
        param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
        param.bParseOnly = true;
        param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
        assert_eq!(
            ISVCDecoder::Initialize(dec, &param as *const SDecodingParam),
            CM_RESULT_SUCCESS as i64
        );

        let mut info = SParserBsInfo::default();
        let mut all = Sha1Hasher::new();
        let mut call = 0usize;
        let mut emitted = 0usize;

        let mut step = |buf: *const u8, len: i32| {
            info.uiInBsTimeStamp = call as u64 + 1;
            let _rv = ISVCDecoder::DecodeParser(dec, buf, len, &mut info).0;
            if info.iNalNum > 0 {
                assert!(!info.pDstBuff.is_null(), "{asset}: null pDstBuff on emit");
                assert!(
                    !info.pNalLenInByte.is_null(),
                    "{asset}: null pNalLenInByte on emit"
                );
                assert!(
                    info.iSpsWidthInPixel > 0 && info.iSpsHeightInPixel > 0,
                    "{asset}: invalid SPS dimensions {}x{}",
                    info.iSpsWidthInPixel,
                    info.iSpsHeightInPixel
                );
                assert_eq!(
                    info.uiInBsTimeStamp, 0,
                    "{asset}: emitting DecodeParser call zeroes uiInBsTimeStamp"
                );
                assert!(
                    info.uiOutBsTimeStamp > 0,
                    "{asset}: emitting DecodeParser call sets uiOutBsTimeStamp"
                );
                let mut total = 0usize;
                for i in 0..info.iNalNum {
                    let nal_len = *info.pNalLenInByte.add(i as usize);
                    assert!(nal_len > 0, "{asset}: non-positive NAL length {nal_len}");
                    total += nal_len as usize;
                }
                let bytes = std::slice::from_raw_parts(info.pDstBuff, total);
                all.update(bytes);
                emitted += 1;
            }
            call += 1;
        };

        for unit in split_annexb_units(data) {
            step(unit.as_ptr(), unit.len() as i32);
        }
        step(std::ptr::null(), 0);

        ISVCDecoder::Uninitialize(dec);
        WelsDestroyDecoder(dec);
        (call, emitted, all.digest())
    }
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
fn decode_parser_matches_expected_output_on_every_asset() {
    for &(asset, want_calls, want_emitted, want_sha1) in EXPECTED_PARSEONLY {
        let data = common::read_stream(&format!("{asset}.264"));
        let (calls, emitted, sha1) = unsafe { run_parseonly(asset, &data) };
        assert_eq!(calls, want_calls, "{asset}: call count");
        assert_eq!(emitted, want_emitted, "{asset}: emitted access unit count");
        assert_eq!(sha1, want_sha1, "{asset}: emitted bitstream SHA-1");
    }
}
