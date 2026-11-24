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

/// Custom mid-row `SM_RASTER_SLICE` and `SM_FIXEDSLCNUM_SLICE` layouts must stay
/// bit-exact between 1 thread and 2/4 threads: slices sharing an MB row are merged
/// onto one worker while disjoint MB-row groups still run in parallel.
#[test]
pub(crate) fn test_multithread_mid_row_fixed_and_raster_bit_exact() {
    let run = |threads: u16, mode: SliceModeEnum, mb_counts: &[u32]| {
        let (w, h) = (320i32, 192i32); // 20x12 = 240 MBs
        let mut enc = Encoder::new();
        let mut p = SEncParamExt::default();
        assert_eq!(enc.default_params(&mut p), CM_RESULT_SUCCESS);
        p.iPicWidth = w;
        p.iPicHeight = h;
        p.iRCMode = RC_MODES::RC_OFF_MODE;
        p.iMultipleThreadIdc = threads;
        p.iSpatialLayerNum = 1;
        p.iLoopFilterDisableIdc = 2;
        p.bUseLoadBalancing = false;

        let s = &mut p.sSpatialLayers[0];
        s.iVideoWidth = w;
        s.iVideoHeight = h;
        s.sSliceArgument.uiSliceMode = mode;
        s.sSliceArgument.uiSliceNum = mb_counts.len() as u32;
        if mode == SM_RASTER_SLICE {
            s.sSliceArgument.uiSliceMbNum[..mb_counts.len()].copy_from_slice(mb_counts);
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
        for f in 0..6usize {
            for (i, b) in yuv.iter_mut().enumerate() {
                *b = ((i * 5 + f * 23) & 0xff) as u8;
            }
            let pic = SourcePicture::from_i420(&yuv, w, h).with_timestamp((f as i64) * 33);
            let mut bs = SFrameBSInfo::default();
            assert_eq!(enc.encode_frame(&pic, &mut bs), 0);
            let mut au = Vec::new();
            for l in bs.layers() {
                let bytes = l.bs_buf(&enc);
                stream.extend_from_slice(bytes);
                au.extend_from_slice(bytes);
            }
            let mut info = SBufferInfo::default();
            assert_eq!(
                dec.decode_no_delay(Some(&au), &mut info),
                DECODING_STATE::dsErrorFree
            );
            assert_eq!(info.iBufferStatus, 1);
        }
        assert_eq!(dec.uninitialize(), CM_RESULT_SUCCESS as std::ffi::c_long);
        assert_eq!(enc.uninitialize(), CM_RESULT_SUCCESS);
        stream
    };

    // Mid-row raster split on 20x12 (240 MBs):
    // slice 0 = [0..53) (rows 0..3), slice 1 = [53..120) (rows 2..6) -> merged on worker 0;
    // slice 2 = [120..187) (rows 6..10), slice 3 = [187..240) (rows 9..12) -> merged on worker 1.
    let mid_row_raster = [53u32, 67, 67, 53];
    let baseline = run(1, SM_RASTER_SLICE, &mid_row_raster);
    for threads in [2, 4] {
        assert_eq!(
            baseline,
            run(threads, SM_RASTER_SLICE, &mid_row_raster),
            "mid-row SM_RASTER_SLICE must be bit-exact at {threads} threads"
        );
    }

    // All-mid-row raster split (5 slices of 48 MBs on 20x12, boundaries at 48/96/144/192):
    // exercises the 2-wave alternating schedule (`{0, 2, 4}` then `{1, 3}`), including
    // multi-group wave filtering when 3 groups map onto 2 threads.
    let five_mid_row_raster = [48u32; 5];
    let baseline_five_raster = run(1, SM_RASTER_SLICE, &five_mid_row_raster);
    for threads in [2, 4] {
        assert_eq!(
            baseline_five_raster,
            run(threads, SM_RASTER_SLICE, &five_mid_row_raster),
            "2-wave mid-row SM_RASTER_SLICE must be bit-exact at {threads} threads"
        );
    }

    // 5 fixed slices on 20x12 in RC_OFF_MODE: row-aligned to [60, 60, 40, 40, 40] MBs,
    // staying bit-exact between 1, 2, and 4 threads.
    let five_fixed = [0u32; 5];
    let baseline_fixed = run(1, SM_FIXEDSLCNUM_SLICE, &five_fixed);
    for threads in [2, 4] {
        assert_eq!(
            baseline_fixed,
            run(threads, SM_FIXEDSLCNUM_SLICE, &five_fixed),
            "row-aligned RC_OFF_MODE SM_FIXEDSLCNUM_SLICE must be bit-exact at {threads} threads"
        );
    }
}

/// Multithreaded encoding with background detection (`bEnableBackgroundDetection = true`,
/// exercising parallel `VaaBackgroundMbDataUpdate` writes into `EncPicBand`), parallel
/// deblocking (`iLoopFilterDisableIdc = 2`), and dynamic load balancing across both
/// `SM_FIXEDSLCNUM_SLICE` and `SM_SIZELIMITED_SLICE`.
#[test]
pub(crate) fn test_multithread_background_detection_and_parallel_deblocking() {
    let (w, h) = (320i32, 192i32);
    let luma = (w * h) as usize;

    for (mode, rc_mode, lb) in [
        (SM_FIXEDSLCNUM_SLICE, RC_MODES::RC_QUALITY_MODE, false),
        (SM_FIXEDSLCNUM_SLICE, RC_MODES::RC_QUALITY_MODE, true),
        (SM_FIXEDSLCNUM_SLICE, RC_MODES::RC_OFF_MODE, true),
        (SM_SIZELIMITED_SLICE, RC_MODES::RC_QUALITY_MODE, false),
    ] {
        let mut enc = Encoder::new();
        let mut p = SEncParamExt::default();
        assert_eq!(enc.default_params(&mut p), CM_RESULT_SUCCESS);
        p.iPicWidth = w;
        p.iPicHeight = h;
        p.iTargetBitrate = 500_000;
        p.iRCMode = rc_mode;
        p.iMultipleThreadIdc = 4;
        p.iSpatialLayerNum = 1;
        p.iLoopFilterDisableIdc = 2;
        p.bEnableBackgroundDetection = true;
        p.bUseLoadBalancing = lb;

        let s = &mut p.sSpatialLayers[0];
        s.iVideoWidth = w;
        s.iVideoHeight = h;
        s.iSpatialBitrate = 500_000;
        s.sSliceArgument.uiSliceMode = mode;
        s.sSliceArgument.uiSliceNum = 4;
        s.sSliceArgument.uiSliceSizeConstraint = 800;
        assert_eq!(enc.initialize_ext(&p), CM_RESULT_SUCCESS);

        let mut dec = Decoder::new();
        let dp = SDecodingParam {
            uiTargetDqLayer: u8::MAX,
            ..Default::default()
        };
        assert_eq!(dec.initialize(&dp), CM_RESULT_SUCCESS as std::ffi::c_long);

        // Static background with a small moving foreground patch so BGD flags background
        // MBs across every worker band and triggers `VaaBackgroundMbDataUpdate`.
        let mut yuv = vec![128u8; luma * 3 / 2];
        for y in 0..h as usize {
            for x in 0..w as usize {
                yuv[y * w as usize + x] = ((x / 4 + y / 4) & 0xff) as u8;
            }
        }
        for f in 0..8usize {
            let mut frame = yuv.clone();
            let px = (f * 16) % (w as usize - 32);
            for y in 64..96 {
                for x in px..px + 32 {
                    frame[y * w as usize + x] = (200 + f * 5) as u8;
                }
            }
            let pic = SourcePicture::from_i420(&frame, w, h).with_timestamp((f as i64) * 33);
            let mut bs = SFrameBSInfo::default();
            assert_eq!(enc.encode_frame(&pic, &mut bs), CM_RESULT_SUCCESS);
            if bs.eFrameType != EVideoFrameType::videoFrameTypeSkip {
                let mut au = Vec::new();
                for l in bs.layers() {
                    au.extend_from_slice(l.bs_buf(&enc));
                }
                let mut info = SBufferInfo::default();
                assert_eq!(
                    dec.decode_no_delay(Some(&au), &mut info),
                    DECODING_STATE::dsErrorFree,
                    "decode failed on frame {f} ({mode:?}, {rc_mode:?}, lb={lb})"
                );
                assert_eq!(info.iBufferStatus, 1);
            }
        }
        assert_eq!(dec.uninitialize(), CM_RESULT_SUCCESS as std::ffi::c_long);
        assert_eq!(enc.uninitialize(), CM_RESULT_SUCCESS);
    }
}

/// End-to-end verification of:
/// - Issue #1: `iMultipleThreadIdc = 0` + `SM_FIXEDSLCNUM_SLICE` (`uiSliceNum = 0`)
///   auto-detects host CPU cores instead of silently resolving to 1 thread / 1 slice.
/// - Issue #2: `iMultipleThreadIdc = 0` + multi-slice + `iLoopFilterDisableIdc = 2`
///   never silently skips deblocking (decoded frames stay clean with no reference drift).
/// - Issue #9: runtime `set_encode_param_ext` updates `iLoopFilterDisableIdc` and
///   `bUseLoadBalancing` cleanly on subsequent frames.
#[test]
pub(crate) fn test_auto_threads_and_runtime_reconfig() {
    let (w, h) = (320i32, 192i32);
    let luma = (w * h) as usize;
    let host_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let mut enc = Encoder::new();
    let mut p = SEncParamExt::default();
    assert_eq!(enc.default_params(&mut p), CM_RESULT_SUCCESS);
    p.iPicWidth = w;
    p.iPicHeight = h;
    p.iTargetBitrate = 500_000;
    p.fMaxFrameRate = 30.0;
    p.iRCMode = RC_MODES::RC_QUALITY_MODE;
    p.iMultipleThreadIdc = 0; // auto-detect cores
    p.iSpatialLayerNum = 1;
    p.iLoopFilterDisableIdc = 2;
    p.bUseLoadBalancing = false;
    p.sSpatialLayers[0].iVideoWidth = w;
    p.sSpatialLayers[0].iVideoHeight = h;
    p.sSpatialLayers[0].iSpatialBitrate = 500_000;
    p.sSpatialLayers[0].fFrameRate = 30.0;
    p.sSpatialLayers[0].sSliceArgument.uiSliceMode = SM_FIXEDSLCNUM_SLICE;
    p.sSpatialLayers[0].sSliceArgument.uiSliceNum = 0; // auto-detect slices
    assert_eq!(enc.initialize_ext(&p), CM_RESULT_SUCCESS);

    let active = enc.encode_param_ext().expect("initialized");
    if host_cores > 1 {
        assert!(
            active.iMultipleThreadIdc > 1,
            "auto thread detection must pick >1 thread on {host_cores}-core host"
        );
        assert_eq!(
            active.sSpatialLayers[0].sSliceArgument.uiSliceMode,
            SM_FIXEDSLCNUM_SLICE
        );
        assert!(active.sSpatialLayers[0].sSliceArgument.uiSliceNum > 1);
    } else {
        assert_eq!(active.iMultipleThreadIdc, 1);
    }

    let mut dec = Decoder::new();
    let dp = SDecodingParam {
        uiTargetDqLayer: u8::MAX,
        ..Default::default()
    };
    assert_eq!(dec.initialize(&dp), CM_RESULT_SUCCESS as std::ffi::c_long);

    let mut yuv = vec![128u8; luma * 3 / 2];
    let encode_and_decode =
        |enc: &mut Encoder, dec: &mut Decoder, yuv: &mut [u8], f: usize| -> Vec<u8> {
            for y in 0..h as usize {
                for x in 0..w as usize {
                    yuv[y * w as usize + x] = ((x * 3 + y * 5 + f * 17) & 0xff) as u8;
                }
            }
            let pic = SourcePicture::from_i420(yuv, w, h).with_timestamp((f as i64) * 33);
            let mut bs = SFrameBSInfo::default();
            assert_eq!(enc.encode_frame(&pic, &mut bs), CM_RESULT_SUCCESS);
            assert_ne!(bs.eFrameType, EVideoFrameType::videoFrameTypeSkip);
            let mut au = Vec::new();
            for l in bs.layers() {
                au.extend_from_slice(l.bs_buf(enc));
            }
            let mut info = SBufferInfo::default();
            assert_eq!(
                dec.decode_no_delay(Some(&au), &mut info),
                DECODING_STATE::dsErrorFree,
                "decode failed on frame {f}"
            );
            assert_eq!(info.iBufferStatus, 1);
            au
        };

    for f in 0..3 {
        let _ = encode_and_decode(&mut enc, &mut dec, &mut yuv, f);
    }

    // Toggle iLoopFilterDisableIdc to 1 (off) in-place, then back to 2, and toggle
    // bUseLoadBalancing on then off.
    let mut reconf = active;
    reconf.iLoopFilterDisableIdc = 1;
    assert_eq!(enc.set_encode_param_ext(&reconf), CM_RESULT_SUCCESS);
    assert_eq!(enc.encode_param_ext().unwrap().iLoopFilterDisableIdc, 1);
    let _ = encode_and_decode(&mut enc, &mut dec, &mut yuv, 3);

    reconf.iLoopFilterDisableIdc = 2;
    assert_eq!(enc.set_encode_param_ext(&reconf), CM_RESULT_SUCCESS);
    assert_eq!(enc.encode_param_ext().unwrap().iLoopFilterDisableIdc, 2);
    let _ = encode_and_decode(&mut enc, &mut dec, &mut yuv, 4);

    reconf.bUseLoadBalancing = true;
    assert_eq!(enc.set_encode_param_ext(&reconf), CM_RESULT_SUCCESS);
    assert!(enc.encode_param_ext().unwrap().bUseLoadBalancing);
    let _ = encode_and_decode(&mut enc, &mut dec, &mut yuv, 5);

    reconf.bUseLoadBalancing = false;
    assert_eq!(enc.set_encode_param_ext(&reconf), CM_RESULT_SUCCESS);
    assert!(!enc.encode_param_ext().unwrap().bUseLoadBalancing);
    let _ = encode_and_decode(&mut enc, &mut dec, &mut yuv, 6);

    assert_eq!(dec.uninitialize(), CM_RESULT_SUCCESS as std::ffi::c_long);
    assert_eq!(enc.uninitialize(), CM_RESULT_SUCCESS);

    // Also verify single-threaded runtime toggle of `iLoopFilterDisableIdc` actually
    // changes the emitted P-frame bitstream (not silently ignored).
    let run_st_pair = |toggle_off_on_frame_1: bool| -> Vec<u8> {
        let mut e = Encoder::new();
        let mut cfg = SEncParamExt::default();
        assert_eq!(e.default_params(&mut cfg), CM_RESULT_SUCCESS);
        cfg.iPicWidth = w;
        cfg.iPicHeight = h;
        cfg.iTargetBitrate = 500_000;
        cfg.fMaxFrameRate = 30.0;
        cfg.iRCMode = RC_MODES::RC_OFF_MODE;
        cfg.iMultipleThreadIdc = 1;
        cfg.iSpatialLayerNum = 1;
        cfg.iLoopFilterDisableIdc = 0;
        cfg.sSpatialLayers[0].iVideoWidth = w;
        cfg.sSpatialLayers[0].iVideoHeight = h;
        cfg.sSpatialLayers[0].iSpatialBitrate = 500_000;
        cfg.sSpatialLayers[0].fFrameRate = 30.0;
        cfg.sSpatialLayers[0].iDLayerQp = 30;
        assert_eq!(e.initialize_ext(&cfg), CM_RESULT_SUCCESS);

        let mut buf = vec![128u8; luma * 3 / 2];
        let mut last_au = Vec::new();
        for f in 0..2 {
            if f == 1 && toggle_off_on_frame_1 {
                let mut adj = e.encode_param_ext().unwrap();
                adj.iLoopFilterDisableIdc = 1;
                assert_eq!(e.set_encode_param_ext(&adj), CM_RESULT_SUCCESS);
            }
            for y in 0..h as usize {
                for x in 0..w as usize {
                    buf[y * w as usize + x] = ((x * 7 + y * 11 + f * 23) & 0xff) as u8;
                }
            }
            let pic = SourcePicture::from_i420(&buf, w, h).with_timestamp((f as i64) * 33);
            let mut bs = SFrameBSInfo::default();
            assert_eq!(e.encode_frame(&pic, &mut bs), CM_RESULT_SUCCESS);
            last_au.clear();
            for l in bs.layers() {
                last_au.extend_from_slice(l.bs_buf(&e));
            }
        }
        assert_eq!(e.uninitialize(), CM_RESULT_SUCCESS);
        last_au
    };
    assert_ne!(
        run_st_pair(false),
        run_st_pair(true),
        "runtime iLoopFilterDisableIdc change must take effect on subsequent frames"
    );
}
