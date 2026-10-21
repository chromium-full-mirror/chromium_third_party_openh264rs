// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! `ForceIntraFrame` with LTR on — `ltr_test.cpp:14-42`.
//!
//! With long-term reference enabled (`iLTRRefNum = 1`, marking period 2) and
//! `ForceIntraFrame(true)` called after every frame, every frame the encoder produces
//! reports `videoFrameTypeIDR`: the first because it opens the stream, each one after
//! it because the caller asked.
//!
//! The configuration is `prepareParamDefault` + `prepareParam`
//! (`encode_decode_api_test.cpp`) with the fixture's first parameter row
//! (`decoder_ec_test.cpp:138`: 160x96, 6 fps, 2 slices) and its five `SetOption` calls.

use openh264rs::api::codec_api::*;

/// `VALID_SIZE` (`encode_decode_api_test.h`) — the fixture rounds odd dimensions up.
fn valid_size(x: i32) -> i32 {
    if x % 2 == 1 { x + 1 } else { x }
}

/// One run of the reference's loop. Returns the frame type of each encoded frame.
fn force_idr_frame_types(
    width: i32,
    height: i32,
    slices: i32,
    frames: usize,
    idr_interval: i32,
) -> Vec<i32> {
    let mut enc = Encoder::new();

    let mut param = SEncParamExt::default();
    assert_eq!(enc.default_params(&mut param), CM_RESULT_SUCCESS);
    // `EncodeDecodeTestBase::prepareParam`.
    let (w, h) = (valid_size(width), valid_size(height));
    param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
    param.iPicWidth = w;
    param.iPicHeight = h;
    param.fMaxFrameRate = 6.0;
    param.iRCMode = RC_MODES::RC_OFF_MODE;
    param.iMultipleThreadIdc = 1;
    param.iSpatialLayerNum = 1;
    param.iNumRefFrame = AUTO_REF_PIC_COUNT;
    param.sSpatialLayers[0].iVideoWidth = w;
    param.sSpatialLayers[0].iVideoHeight = h;
    param.sSpatialLayers[0].fFrameRate = 6.0;
    param.sSpatialLayers[0].sSliceArgument.uiSliceMode = SliceModeEnum::SM_FIXEDSLCNUM_SLICE;
    param.sSpatialLayers[0].sSliceArgument.uiSliceNum = slices as u32;

    assert_eq!(enc.initialize_ext(&param), CM_RESULT_SUCCESS);

    enc.set_trace_level(0); // WELS_LOG_QUIET
    assert_eq!(
        enc.set_sps_pps_id_strategy(EParameterSetStrategy::INCREASING_ID),
        CM_RESULT_SUCCESS
    );
    assert_eq!(enc.set_idr_interval(idr_interval), CM_RESULT_SUCCESS);
    assert_eq!(
        enc.set_ltr_config(&SLTRConfig {
            bEnableLongTermReference: true,
            iLTRRefNum: 1,
        }),
        CM_RESULT_SUCCESS
    );
    assert_eq!(enc.set_ltr_marking_period(2), CM_RESULT_SUCCESS);

    // `EncodeOneFrame`: a flat luma plane and a flat chroma plane, so the content
    // never decides the frame type — only the encoder's own state does.
    let luma = (w * h) as usize;
    let frame = luma * 3 / 2;
    let mut buf = vec![0u8; frame];
    buf[..luma].fill(0x2a);
    buf[luma..].fill(0x80);

    let pic = SourcePicture::from_i420(&buf, w, h);

    let mut types = Vec::with_capacity(frames);
    for _ in 0..frames {
        let mut info = SFrameBSInfo::default();
        let rv = enc.encode_frame(&pic, &mut info);
        assert_eq!(rv, CM_RESULT_SUCCESS, "EncodeFrame");
        types.push(info.eFrameType as i32);
        assert_eq!(
            enc.force_intra_frame(true, -1),
            CM_RESULT_SUCCESS,
            "ForceIntraFrame"
        );
    }

    enc.uninitialize();
    types
}

/// Every frame reports `videoFrameTypeIDR`, on the fixture's three parameter rows.
///
/// 32 frames rather than the fixture's 300: the assertion fails on the second frame
/// when it fails at all.
#[test]
pub(crate) fn force_intra_frame_with_ltr_gives_an_idr_every_time() {
    // `decoder_ec_test.cpp:138-140`, the three instantiations of
    // `EncodeDecodeTestAPIBase/EncodeDecodeTestAPI`.
    for (w, h, slices) in [(160, 96, 2), (140, 96, 4), (140, 96, 4)] {
        // `iIDRPeriod = 2^(iTemporalLayerNum - 1) * (rand() % 5 + 1)`; with the default
        // one temporal layer the range is 1..=5. All five run, because
        // `ForceIntraFrame` is supposed to make the interval irrelevant.
        for idr_interval in 1..=5 {
            let types = force_idr_frame_types(w, h, slices, 32, idr_interval);
            for (i, t) in types.iter().enumerate() {
                assert_eq!(
                    *t,
                    EVideoFrameType::videoFrameTypeIDR as i32,
                    "{w}x{h}, {slices} slices, IDR interval {idr_interval}: frame {i} is \
                     type {t}, not IDR ({}), after ForceIntraFrame(true) — ltr_test.cpp:39",
                    EVideoFrameType::videoFrameTypeIDR as i32
                );
            }
        }
    }
}

#[test]
pub(crate) fn test_high_bitrate_idr_target_bits() {
    // C++ test: EncoderInterfaceTest.HighBitrateIdrTargetBits
    let mut enc = Encoder::new();

    let mut param = SEncParamExt::default();
    assert_eq!(enc.default_params(&mut param), CM_RESULT_SUCCESS);
    param.iUsageType = EUsageType::CAMERA_VIDEO_REAL_TIME;
    param.iPicWidth = 1280;
    param.iPicHeight = 720;
    param.iTargetBitrate = 200_000_000;
    param.fMaxFrameRate = 30.0;
    param.iRCMode = RC_MODES::RC_BITRATE_MODE;
    param.iSpatialLayerNum = 1;
    param.sSpatialLayers[0].iVideoWidth = param.iPicWidth;
    param.sSpatialLayers[0].iVideoHeight = param.iPicHeight;
    param.sSpatialLayers[0].iSpatialBitrate = param.iTargetBitrate;
    param.sSpatialLayers[0].fFrameRate = param.fMaxFrameRate;

    assert_eq!(enc.initialize_ext(&param), CM_RESULT_SUCCESS);

    let y = vec![128u8; 1280 * 720];
    let u = vec![128u8; 640 * 360];
    let v = vec![128u8; 640 * 360];
    let mut src = SourcePicture::from_i420_planes(&y, &u, &v, 1280, 720);
    let mut bs = SFrameBSInfo::default();
    assert_eq!(enc.encode_frame(&src, &mut bs), CM_RESULT_SUCCESS);

    // Force subsequent IDR frame so that iIdrNum != 0 and RcDecideTargetBits uses iIdrBitrateRatio
    assert_eq!(enc.force_intra_frame(true, -1), CM_RESULT_SUCCESS);
    src.uiTimeStamp += 33;
    assert_eq!(enc.encode_frame(&src, &mut bs), CM_RESULT_SUCCESS);

    enc.uninitialize();
}
