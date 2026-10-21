// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

#[path = "common/mod.rs"]
mod common;
use common::Sha1Hasher;
use openh264rs::api::codec_api::*;

fn update_hash_from_encoded_frame(hasher: &mut Sha1Hasher, enc: &Encoder, bs_info: &SFrameBSInfo) {
    for layer in bs_info.layers() {
        let slice = layer.bs_buf(enc);
        if !slice.is_empty() {
            hasher.update(slice);
        }
    }
}

use openh264rs::split_annexb_units;

#[test]
pub(crate) fn test_loopback_encode_and_decode_pipeline() {
    let test_resolutions = [(160, 120), (320, 240), (640, 360)];

    for (width, height) in test_resolutions {
        let mut enc = Encoder::new();

        let mut enc_param = SEncParamBase::default();
        enc_param.iPicWidth = width;
        enc_param.iPicHeight = height;
        enc_param.fMaxFrameRate = 30.0;
        enc_param.iTargetBitrate = 250000;
        enc_param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;

        let enc_init = enc.initialize(&enc_param);
        assert_eq!(enc_init, CM_RESULT_SUCCESS);

        let mut dec = Decoder::new();

        let mut dec_param = SDecodingParam::default();
        dec_param.uiTargetDqLayer = u8::MAX;

        let dec_init = dec.initialize(&dec_param);
        assert_eq!(dec_init, CM_RESULT_SUCCESS as std::ffi::c_long);

        let mut bs_info = SFrameBSInfo::default();
        let ps_ret = enc.encode_parameter_sets(&mut bs_info);
        assert_eq!(ps_ret, CM_RESULT_SUCCESS);

        let frame_size = (width * height * 3 / 2) as usize;
        let mut yuv_input = vec![128u8; frame_size];
        for y in 0..height as usize {
            for x in 0..width as usize {
                yuv_input[y * width as usize + x] = ((x * 4 + y * 4) % 256) as u8;
            }
        }

        let src_pic = SourcePicture::from_i420(&yuv_input, width, height);

        let enc_frame_ret = enc.encode_frame(&src_pic, &mut bs_info);
        assert_eq!(enc_frame_ret, CM_RESULT_SUCCESS);

        if bs_info.eFrameType != EVideoFrameType::videoFrameTypeInvalid
            && bs_info.eFrameType != EVideoFrameType::videoFrameTypeSkip
        {
            for layer in bs_info.layers() {
                let bs = layer.bs_buf(&enc);
                if !bs.is_empty() {
                    let mut dst_info = SBufferInfo::default();
                    let dec_ret = dec.decode(Some(bs), &mut dst_info);
                    assert_eq!(dec_ret, DECODING_STATE::dsErrorFree);
                }
            }
        }

        assert_eq!(enc.uninitialize(), CM_RESULT_SUCCESS);
        assert_eq!(dec.uninitialize(), CM_RESULT_SUCCESS as std::ffi::c_long);
    }
}

struct DecodeEncodeFileParam {
    file_name: &'static str,
    hash_str: &'static str,
    width: i32,
    height: i32,
    frame_rate: f32,
}

static K_DECODE_ENCODE_FILE_ARRAY: &[DecodeEncodeFileParam] = &[
    DecodeEncodeFileParam {
        file_name: "res/test_vd_1d.264",
        hash_str: "34fc3aee85cc0b0223c2701d810a536fe3818a00",
        width: 320,
        height: 192,
        frame_rate: 12.0,
    },
    DecodeEncodeFileParam {
        file_name: "res/test_vd_rc.264",
        hash_str: "9f15b0677b5f7daa922079ec4fa49e3f457fc998",
        width: 320,
        height: 192,
        frame_rate: 12.0,
    },
];

/// Upstream's `DecodeEncodeFile/DecodeEncodeTest.CompareOutput`
/// (`test/api/decode_encode_test.cpp`).
///
/// `Initialize` takes a bare `SEncParamBase`, which leaves every `FillDefault` value in
/// place: `iRCMode = RC_QUALITY_MODE`, and `bEnableSceneChangeDetect`,
/// `bEnableBackgroundDetection`, `bEnableAdaptiveQuant` and `bEnableFrameSkip` all
/// `true`.
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_decode_encode_full_cycle_sha1_parity() {
    for param in K_DECODE_ENCODE_FILE_ARRAY {
        let data = common::read_stream(param.file_name);
        assert!(
            !data.is_empty(),
            "Asset file {} must not be empty",
            param.file_name
        );

        let mut dec = Decoder::new();

        let mut dec_param = SDecodingParam::default();
        dec_param.uiTargetDqLayer = u8::MAX;
        dec_param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
        dec_param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;

        let dec_init = dec.initialize(&dec_param);
        assert_eq!(dec_init, CM_RESULT_SUCCESS as std::ffi::c_long);

        let mut enc = Encoder::new();

        // `hash_str` is upstream's expectation from
        // `test/api/decode_encode_test.cpp:133`: single slice, one spatial layer, no
        // denoise, no lossless link, no LTR, CAVLC, and a zeroed `SEncParamBase`
        // carrying only usage type, frame rate, width, height and the bitrate.
        let mut enc_param = SEncParamBase::default();
        enc_param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
        enc_param.fMaxFrameRate = param.frame_rate;
        enc_param.iPicWidth = param.width;
        enc_param.iPicHeight = param.height;
        enc_param.iTargetBitrate = 5_000_000;

        let enc_init = enc.initialize(&enc_param);
        assert_eq!(enc_init, CM_RESULT_SUCCESS);

        let mut hasher = Sha1Hasher::new();
        let units = split_annexb_units(&data);

        let encode_picture_frame =
            |enc: &mut Encoder, dec: &Decoder, buf_info: &SBufferInfo, hasher: &mut Sha1Hasher| {
                if let Some(src_pic) = buf_info.as_source_picture(dec) {
                    let mut bs_info = SFrameBSInfo::default();
                    let enc_ret = enc.encode_frame(&src_pic, &mut bs_info);
                    if enc_ret == CM_RESULT_SUCCESS {
                        update_hash_from_encoded_frame(hasher, enc, &bs_info);
                    }
                }
            };

        let mut timestamp = 0u64;
        for unit in units {
            let mut buf_info = SBufferInfo::default();
            timestamp += 1;
            buf_info.uiInBsTimeStamp = timestamp;
            let dec_ret = dec.decode_no_delay(Some(unit), &mut buf_info);
            if dec_ret == DECODING_STATE::dsErrorFree {
                encode_picture_frame(&mut enc, &dec, &buf_info, &mut hasher);
            }

            let mut null_buf_info = SBufferInfo::default();
            null_buf_info.uiInBsTimeStamp = timestamp;
            let recon_ret = dec.decode(None, &mut null_buf_info);
            if recon_ret == DECODING_STATE::dsErrorFree {
                encode_picture_frame(&mut enc, &dec, &null_buf_info, &mut hasher);
            }
        }

        // Flush remaining frames in decoder buffer
        dec.set_end_of_stream(true);

        let mut buf_info = SBufferInfo::default();
        let dec_ret = dec.decode(None, &mut buf_info);
        if dec_ret == DECODING_STATE::dsErrorFree {
            encode_picture_frame(&mut enc, &dec, &buf_info, &mut hasher);
        }

        let remaining_frames = dec.frames_remaining();

        for _ in 0..remaining_frames {
            let mut buf_info = SBufferInfo::default();
            let flush_ret = dec.flush(&mut buf_info);
            if flush_ret == DECODING_STATE::dsErrorFree {
                encode_picture_frame(&mut enc, &dec, &buf_info, &mut hasher);
            }
        }

        let calculated_hash = hasher.digest();
        assert_eq!(
            calculated_hash, param.hash_str,
            "SHA-1 hash mismatch in full decode-encode cycle for {}",
            param.file_name
        );

        enc.uninitialize();
        dec.uninitialize();
    }
}
