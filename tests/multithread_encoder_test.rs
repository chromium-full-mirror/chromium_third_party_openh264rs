// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Parameterized integration test for multi-threaded encoding (`iMultipleThreadIdc > 1`)
//! across `SM_FIXEDSLCNUM_SLICE`, `SM_RASTER_SLICE`, `SM_SIZELIMITED_SLICE`, and
//! multi-spatial-layer SVC, verifying bitstream determinism and decode roundtrip.

use openh264rs::api::codec_api::{SliceModeEnum::*, *};

#[test]
pub(crate) fn test_multithread_encoder() {
    let run = |threads: u16, mode: SliceModeEnum, layers: usize, lb: bool| {
        let (w, h) = (320i32, 192i32);
        let mut enc = Encoder::new();

        let mut p = SEncParamExt::default();
        let rv = enc.default_params(&mut p);
        assert_eq!(rv, CM_RESULT_SUCCESS);
        p.iPicWidth = w;
        p.iPicHeight = h;
        p.iRCMode = RC_MODES::RC_OFF_MODE;
        p.iMultipleThreadIdc = threads;
        p.iSpatialLayerNum = layers as i32;
        p.iLoopFilterDisableIdc = 2;
        p.bUseLoadBalancing = lb;

        for i in 0..layers {
            let scale = 1 << (layers - 1 - i);
            let (lw, lh, slices) = (w / scale, h / scale, 4 / scale as u32);
            let s = &mut p.sSpatialLayers[i];
            s.iVideoWidth = lw;
            s.iVideoHeight = lh;
            s.sSliceArgument.uiSliceMode = mode;
            s.sSliceArgument.uiSliceNum = slices;
            s.sSliceArgument.uiSliceSizeConstraint = 600;
            s.sSliceArgument.uiSliceMbNum[..slices as usize]
                .fill(((lw / 16) * (lh / 16)) as u32 / slices);
        }
        assert_eq!(enc.initialize_ext(&p), CM_RESULT_SUCCESS);

        let mut dec = Decoder::new();
        let dp = SDecodingParam {
            uiTargetDqLayer: u8::MAX,
            ..Default::default()
        };
        assert_eq!(dec.initialize(&dp), CM_RESULT_SUCCESS as std::ffi::c_long);

        let luma = (w * h) as usize;
        let mut yuv = vec![0u8; luma * 3 / 2];

        let mut stream = Vec::new();
        for f in 0..8usize {
            for (i, b) in yuv.iter_mut().enumerate() {
                *b = ((i * 3 + f * 17) & 0xff) as u8;
            }
            let pic = SourcePicture::from_i420(&yuv, w, h).with_timestamp((f as i64) * 33);
            let mut bs = SFrameBSInfo::default();
            assert_eq!(enc.encode_frame(&pic, &mut bs), 0);
            assert!(bs.iFrameSizeInBytes > 0);

            let mut au = Vec::new();
            for l in bs.layers() {
                let bytes = l.bs_buf(&enc);
                stream.extend_from_slice(bytes);
                if l.uiSpatialId == 0 {
                    au.extend_from_slice(bytes);
                }
            }

            let mut info = SBufferInfo::default();
            let st = dec.decode_no_delay(Some(&au), &mut info);
            assert_eq!(st, DECODING_STATE::dsErrorFree);
            assert_eq!(info.iBufferStatus, 1);
            assert_eq!(info.UsrData.sSystemBuffer.iWidth, w >> (layers - 1));
            assert_eq!(info.UsrData.sSystemBuffer.iHeight, h >> (layers - 1));
        }

        assert_eq!(dec.uninitialize(), CM_RESULT_SUCCESS as std::ffi::c_long);
        assert_eq!(enc.uninitialize(), CM_RESULT_SUCCESS);
        stream
    };

    for (mode, layers, lb, deterministic) in [
        // (slice_mode, spatial_layers, load_balancing, check_determinism)
        (SM_FIXEDSLCNUM_SLICE, 1, false, true),
        (SM_FIXEDSLCNUM_SLICE, 1, true, false),
        (SM_RASTER_SLICE, 1, false, true),
        (SM_SIZELIMITED_SLICE, 1, false, false),
        (SM_FIXEDSLCNUM_SLICE, 2, false, true),
    ] {
        let baseline = run(1, mode, layers, lb);
        for threads in [2, 4] {
            let mt = run(threads, mode, layers, lb);
            if deterministic {
                assert_eq!(baseline, mt, "{mode:?} layers={layers} threads={threads}");
            }
        }
    }
}
