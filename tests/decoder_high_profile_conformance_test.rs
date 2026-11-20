// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Progressive High Profile (ITU-T H.264 A.2.4 / A.2.4.1) decoder conformance tests.
//!
//! Each vector is verified against libavcodec's raw `yuv420p` output across three
//! feed modes:
//! - per-NAL (`decode(Some(nal))` + `decode(None)`),
//! - per-access-unit (`decode(Some(au))`, no intermediate `None` calls), and
//! - no-delay (`decode_no_delay(Some(au))`).
//!
//! ## Running the `#[ignore]` full-mirror FATE tests
//!
//! The `test_full_*` and `test_external_195_conformance_streams_never_panic` tests expect
//! `OPENH264_CONFORMANCE_DIR` to point at a local copy of FFmpeg's FATE `h264-conformance`
//! suite (~140 MB, 137 base vectors + 58 `FRext/` vectors). Download it via `rsync` and run:
//!
//! ```sh
//! rsync -vaLW rsync://fate-suite.ffmpeg.org/fate-suite/h264-conformance/ /tmp/h264-conformance/
//! OPENH264_CONFORMANCE_DIR=/tmp/h264-conformance \
//!   cargo test --test decoder_high_profile_conformance_test -- --ignored
//! ```

#[path = "common/mod.rs"]
mod common;

use openh264rs::api::codec_api::DECODING_STATE;
use std::path::{Path, PathBuf};

#[derive(Copy, Clone)]
struct VectorRow {
    file: &'static str,
    frames: usize,
    sha1: &'static str,
}

#[derive(Copy, Clone, Debug)]
enum FeedMode {
    PerNal,
    PerAu,
    NoDelay,
}

/// Small first-party Chromium-generated High Profile (`profile_idc == 100`) bitstreams
/// committed in `tests/data`.
///
/// Each file is 16 frames of `lavfi` `mandelbrot=size=176x144:rate=30` encoded with
/// `ffmpeg` + `libx264`:
///
/// ```sh
/// BASE="threads=1:keyint=10:min-keyint=10:scenecut=0"
/// ffmpeg -hide_banner -v error -f lavfi -i mandelbrot=size=176x144:rate=30 -frames:v 16 \
///   [EXTRA_VF] -c:v libx264 -profile:v high -pix_fmt yuv420p \
///   -x264-params "${BASE}:${EXTRA_PARAMS}" -bsf:v h264_mp4toannexb -f h264 -y <file>
/// ```
///
/// Reference SHA-1 digests are computed from `libavcodec` raw `yuv420p` output:
///
/// ```sh
/// ffmpeg -hide_banner -v error -f h264 -i <file> -fps_mode passthrough \
///   -f rawvideo -pix_fmt yuv420p - | sha1sum
/// ```
const COMMITTED_VECTORS: [VectorRow; 10] = [
    // High Profile CABAC, 8x8 transform (intra & inter), hierarchical B-pyramid with
    // max_num_ref_frames=6, multi-ref L0/L1, spatial direct B-slices, and L0 RPLR.
    // EXTRA_PARAMS: bframes=4:b-pyramid=normal:ref=6:b-adapt=2:direct=spatial
    VectorRow {
        file: "high_cabac_bpyramid_176x144.264",
        frames: 16,
        sha1: "48c43f4d3088b05dc0da34eab9b1e38a759c4fb2",
    },
    // High Profile CABAC, 8x8 transform, strict B-pyramid with MMCO 1 short-term unmarking,
    // temporal direct B-slices (direct_spatial_mv_pred_flag=0), and open-GOP non-IDR I-slice.
    // EXTRA_PARAMS: bframes=3:b-pyramid=strict:ref=4:direct=temporal:open-gop=1
    VectorRow {
        file: "high_cabac_temporal_mmco_176x144.264",
        frames: 16,
        sha1: "e449c996282ffa01b6785dc435349833e8a10342",
    },
    // High Profile CAVLC, 8x8 transform, PPS JVT scaling matrices (pic_scaling_matrix_present_flag=1),
    // B-pyramid reordering, and temporal direct B-slices.
    // EXTRA_PARAMS: cabac=0:cqm=jvt:bframes=3:b-pyramid=normal:ref=4:direct=temporal
    VectorRow {
        file: "high_cavlc_cqm_bpyramid_176x144.264",
        frames: 16,
        sha1: "c58c01c2b1d12d1dc1e030d1d8dd204f67658e3f",
    },
    // High Profile CABAC, 8x8 transform, custom 4x4 scaling lists (128..255) at QP 36 so coded
    // 4x4/I16x16 residuals dequantize through >u16::MAX entries (exercising u32 dequant tables).
    // EXTRA_PARAMS: cqm4i=128,160,192,224,160,192,224,240,192,224,240,255,224,240,255,255:cqm4p=128,160,192,224,160,192,224,240,192,224,240,255,224,240,255,255:bframes=2:ref=4:qp=36:ipratio=1.0:pbratio=1.0
    VectorRow {
        file: "high_cabac_cqm_highqp_176x144.264",
        frames: 16,
        sha1: "01e3624a57a2dede12ad7919f1cd5fbdcb8067c1",
    },
    // High Profile CABAC, 8x8 transform, explicit non-identity P-slice weighted prediction
    // (weighted_pred_flag=1 on fade) and implicit B-slice weighted biprediction (idc=2).
    // EXTRA_VF: -vf fade=in:0:6,fade=out:10:6
    // EXTRA_PARAMS: weightp=2:weightb=1:bframes=2:ref=4
    VectorRow {
        file: "high_weightp_weightb_fade_176x144.264",
        frames: 16,
        sha1: "1048f858db6e10191e07d6c1aac3aee303555ef4",
    },
    // High Profile CAVLC, 8x8 transform, explicit non-identity P-slice weighted prediction
    // on fade and implicit B-slice weighted biprediction (idc=2).
    // EXTRA_VF: -vf fade=in:0:6,fade=out:10:6
    // EXTRA_PARAMS: cabac=0:weightp=2:weightb=1:bframes=2:ref=4
    VectorRow {
        file: "high_cavlc_weight_fade_176x144.264",
        frames: 16,
        sha1: "2f2ae5d7fe496218a9b742afaa68bfd0e598fa57",
    },
    // High Profile CABAC, 8x8 transform, mixed refs, and all MB/sub-MB partition sizes
    // including sub-8x8 (P_8x4/P_4x8/P_4x4) clearing noSubMbPartSizeLessThan8x8Flag.
    // EXTRA_PARAMS: partitions=all:me=umh:subme=9:mixed-refs=1:ref=5:bframes=3
    VectorRow {
        file: "high_partitions_all_176x144.264",
        frames: 16,
        sha1: "e297fbbdf6b1103d50990b802559353ce6cea0b2",
    },
    // High Profile CAVLC, 8x8 transform, 4 slices per access unit (64 VCL slices total),
    // and sub-8x8 partitions exercising per-MB 8x8 flag resets across slice boundaries.
    // EXTRA_PARAMS: cabac=0:slices=4:partitions=all:bframes=2:ref=4
    VectorRow {
        file: "high_cavlc_slices4_partitions_176x144.264",
        frames: 16,
        sha1: "74fb1a5959f554c239cb9f7a2558e64ef511f2c7",
    },
    // High Profile CABAC, 8x8 transform, custom deblocking offsets (alpha=-3, beta=+3),
    // non-zero chroma_qp_index_offset (+4), constrained_intra_pred_flag=1, and AUD NALs.
    // EXTRA_PARAMS: deblock=-3,3:chroma-qp-offset=6:constrained-intra=1:aud=1:bframes=2
    VectorRow {
        file: "high_deblock_chroma_ci_176x144.264",
        frames: 16,
        sha1: "aed7631d04f0b5fcaf46937981dc03bd33973953",
    },
    // High Profile CABAC, 8x8 transform, non-mod16 frame cropping (176x144 -> 174x142),
    // and deblocking filter disabled on every slice (disable_deblocking_filter_idc=1).
    // EXTRA_VF: -vf crop=174:142:0:0
    // EXTRA_PARAMS: no-deblock=1:bframes=3:ref=4
    VectorRow {
        file: "high_crop_nodeblock_174x142.264",
        frames: 16,
        sha1: "c585cb1f242f5548ae305bbaf954aaf60dee73dc",
    },
];

/// Full-length progressive 8-bit 4:2:0 FRExt conformance files, loaded at runtime
/// from the directory named by `OPENH264_CONFORMANCE_DIR`.
const FULL_LENGTH_VECTORS: [VectorRow; 21] = [
    VectorRow {
        file: "Freh12_B.264",
        frames: 100,
        sha1: "f493647c8003a53eceef2b7d30cc7c4e2f9d1e7b",
    },
    VectorRow {
        file: "Freh1_B.264",
        frames: 100,
        sha1: "c310cde0ad0dee24809f02320bcfb574fad33157",
    },
    VectorRow {
        file: "Freh2_B.264",
        frames: 100,
        sha1: "b2b4721862c9df0d5a1db7004e654e93c7c7a1a1",
    },
    VectorRow {
        file: "freh3.264",
        frames: 100,
        sha1: "29c0db497f800ccade4b96d89da61c3f7612df69",
    },
    VectorRow {
        file: "freh8.264",
        frames: 100,
        sha1: "015c48738a2999f22dea3fc3bbfdca76805b3438",
    },
    VectorRow {
        file: "freh9.264",
        frames: 100,
        sha1: "d2279733c48003546531ef0dd8b84e79845bf6c8",
    },
    VectorRow {
        file: "FRExt1_Panasonic.avc",
        frames: 8,
        sha1: "3772f1e597ab50ec127729b494c3232025ac5323",
    },
    VectorRow {
        file: "FRExt3_Panasonic.avc",
        frames: 11,
        sha1: "b9439e4360dd44cd807bd4fc09389b1944ea7d12",
    },
    VectorRow {
        file: "FRExt_MMCO4_Sony_B.264",
        frames: 60,
        sha1: "9189c6b8444779288b8ccf6b935388bf82d018fc",
    },
    VectorRow {
        file: "HCAFR1_HHI.264",
        frames: 10,
        sha1: "5ddf8e86686718ff9fcc9d32ebf556a394edfc2b",
    },
    VectorRow {
        file: "HCAFR2_HHI.264",
        frames: 10,
        sha1: "2ff2c478c530b76abd8d4ca0ad5051c7b1723c66",
    },
    VectorRow {
        file: "HCAFR3_HHI.264",
        frames: 10,
        sha1: "3de48d57cae5fd4f79bcbd71b9f8852eb78de184",
    },
    VectorRow {
        file: "HCAFR4_HHI.264",
        frames: 10,
        sha1: "1371740ae19ba275e92b9ad8232b9eb6166a482d",
    },
    VectorRow {
        file: "HPCA_BRCM_C.264",
        frames: 300,
        sha1: "17f6b0f91b2af72603962314e72bf444df238446",
    },
    VectorRow {
        file: "HPCADQ_BRCM_B.264",
        frames: 300,
        sha1: "b5f49b2493143c63703feb91744be7172d8e5762",
    },
    VectorRow {
        file: "HPCALQ_BRCM_B.264",
        frames: 300,
        sha1: "b5f49b2493143c63703feb91744be7172d8e5762",
    },
    VectorRow {
        file: "HPCANL_BRCM_C.264",
        frames: 300,
        sha1: "e507bfb437c6b75c0ae8585ce6bc4c787fe80d66",
    },
    VectorRow {
        file: "HPCAQ2LQ_BRCM_B.264",
        frames: 100,
        sha1: "165e63aa51b308f42bfb7cb3df3ca7ecaa1e56bf",
    },
    VectorRow {
        file: "HPCV_BRCM_A.264",
        frames: 300,
        sha1: "17f6b0f91b2af72603962314e72bf444df238446",
    },
    VectorRow {
        file: "HPCVNL_BRCM_A.264",
        frames: 300,
        sha1: "e507bfb437c6b75c0ae8585ce6bc4c787fe80d66",
    },
    VectorRow {
        file: "test8b43.264",
        frames: 43,
        sha1: "1016e8222ff717a3f59f8e3647e200a676878ab7",
    },
];

fn assert_stream_in_mode(file: &str, frames: usize, sha1: &str, data: &[u8], mode: FeedMode) {
    let aus = common::split_access_units(data);
    assert_eq!(
        aus.len(),
        frames,
        "{file}: access-unit splitter produced {} units, expected {frames} frames",
        aus.len()
    );

    let run = match mode {
        FeedMode::PerNal => common::decode_bytes_per_nal(data, file, false),
        FeedMode::PerAu => common::decode_access_units(&aus, file, false),
        FeedMode::NoDelay => common::decode_access_units_no_delay(&aus, file, false),
    };

    assert_eq!(
        run.state_union,
        DECODING_STATE::dsErrorFree,
        "{file} ({mode:?}): expected dsErrorFree on every call, got {:?}",
        run.state_union
    );
    assert_eq!(
        run.frames, frames,
        "{file} ({mode:?}): decoded frame count mismatch"
    );
    assert_eq!(
        run.sha1, sha1,
        "{file} ({mode:?}): SHA-1 mismatch ({} frames decoded)",
        run.frames
    );
}

fn assert_row_in_mode(row: &VectorRow, data: &[u8], mode: FeedMode) {
    assert_stream_in_mode(row.file, row.frames, row.sha1, data, mode);
}

fn check_committed(idx: usize, mode: FeedMode) {
    let row = &COMMITTED_VECTORS[idx];
    let data = common::read_stream(row.file);
    assert_row_in_mode(row, &data, mode);
}

fn resolve_full_conformance_path(file: &str) -> PathBuf {
    let dir = std::env::var_os("OPENH264_CONFORMANCE_DIR").unwrap_or_else(|| {
        panic!("OPENH264_CONFORMANCE_DIR must be set to run full-length conformance tests")
    });
    let base = Path::new(&dir);
    let direct = base.join(file);
    if direct.exists() {
        return direct;
    }
    let sub = base.join("FRext").join(file);
    if sub.exists() {
        return sub;
    }
    panic!(
        "conformance file {file} not found under {} (or {}/FRext)",
        base.display(),
        base.display()
    );
}

fn check_full_length(idx: usize) {
    let row = &FULL_LENGTH_VECTORS[idx];
    let path = resolve_full_conformance_path(row.file);
    let data =
        std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    for mode in [FeedMode::PerNal, FeedMode::PerAu, FeedMode::NoDelay] {
        assert_row_in_mode(row, &data, mode);
    }
}

macro_rules! committed_row_tests {
    ($idx:expr, $per_nal:ident, $per_au:ident, $no_delay:ident $(, ignore = $reason:literal )?) => {
        #[test]
        #[cfg_attr(not(has_test_data), ignore = "test input files not found")]
        $( #[ignore = $reason] )?
        pub(crate) fn $per_nal() {
            check_committed($idx, FeedMode::PerNal);
        }

        #[test]
        #[cfg_attr(not(has_test_data), ignore = "test input files not found")]
        $( #[ignore = $reason] )?
        pub(crate) fn $per_au() {
            check_committed($idx, FeedMode::PerAu);
        }

        #[test]
        #[cfg_attr(not(has_test_data), ignore = "test input files not found")]
        $( #[ignore = $reason] )?
        pub(crate) fn $no_delay() {
            check_committed($idx, FeedMode::NoDelay);
        }
    };
}

committed_row_tests!(
    0,
    test_hp_cabac_bpyramid_per_nal,
    test_hp_cabac_bpyramid_per_au,
    test_hp_cabac_bpyramid_no_delay
);
committed_row_tests!(
    1,
    test_hp_cabac_temporal_mmco_per_nal,
    test_hp_cabac_temporal_mmco_per_au,
    test_hp_cabac_temporal_mmco_no_delay
);
committed_row_tests!(
    2,
    test_hp_cavlc_cqm_bpyramid_per_nal,
    test_hp_cavlc_cqm_bpyramid_per_au,
    test_hp_cavlc_cqm_bpyramid_no_delay
);
committed_row_tests!(
    3,
    test_hp_cabac_cqm_highqp_per_nal,
    test_hp_cabac_cqm_highqp_per_au,
    test_hp_cabac_cqm_highqp_no_delay
);
committed_row_tests!(
    4,
    test_hp_weightp_weightb_fade_per_nal,
    test_hp_weightp_weightb_fade_per_au,
    test_hp_weightp_weightb_fade_no_delay
);
committed_row_tests!(
    5,
    test_hp_cavlc_weight_fade_per_nal,
    test_hp_cavlc_weight_fade_per_au,
    test_hp_cavlc_weight_fade_no_delay
);
committed_row_tests!(
    6,
    test_hp_partitions_all_per_nal,
    test_hp_partitions_all_per_au,
    test_hp_partitions_all_no_delay
);
committed_row_tests!(
    7,
    test_hp_cavlc_slices4_partitions_per_nal,
    test_hp_cavlc_slices4_partitions_per_au,
    test_hp_cavlc_slices4_partitions_no_delay
);
committed_row_tests!(
    8,
    test_hp_deblock_chroma_ci_per_nal,
    test_hp_deblock_chroma_ci_per_au,
    test_hp_deblock_chroma_ci_no_delay
);
committed_row_tests!(
    9,
    test_hp_crop_nodeblock_per_nal,
    test_hp_crop_nodeblock_per_au,
    test_hp_crop_nodeblock_no_delay
);

macro_rules! full_length_row_test {
    ($idx:expr, $name:ident) => {
        #[test]
        #[ignore = "requires OPENH264_CONFORMANCE_DIR pointing to full FRExt vectors"]
        pub(crate) fn $name() {
            check_full_length($idx);
        }
    };
}

full_length_row_test!(0, test_full_frext_freh12_b);
full_length_row_test!(1, test_full_frext_freh1_b);
full_length_row_test!(2, test_full_frext_freh2_b);
full_length_row_test!(3, test_full_frext_freh3);
full_length_row_test!(4, test_full_frext_freh8);
full_length_row_test!(5, test_full_frext_freh9);
full_length_row_test!(6, test_full_frext_frext1_panasonic);
full_length_row_test!(7, test_full_frext_frext3_panasonic);
full_length_row_test!(8, test_full_frext_frext_mmco4_sony_b);
full_length_row_test!(9, test_full_frext_hcafr1_hhi);
full_length_row_test!(10, test_full_frext_hcafr2_hhi);
full_length_row_test!(11, test_full_frext_hcafr3_hhi);
full_length_row_test!(12, test_full_frext_hcafr4_hhi);
full_length_row_test!(13, test_full_frext_hpca_brcm_c);
full_length_row_test!(14, test_full_frext_hpcadq_brcm_b);
full_length_row_test!(15, test_full_frext_hpcalq_brcm_b);
full_length_row_test!(16, test_full_frext_hpcanl_brcm_c);
full_length_row_test!(17, test_full_frext_hpcaq2lq_brcm_b);
full_length_row_test!(18, test_full_frext_hpcv_brcm_a);
full_length_row_test!(19, test_full_frext_hpcvnl_brcm_a);
full_length_row_test!(20, test_full_frext_test8b43);

/// Full-length base-suite heavy-MMCO + frame_num-gap conformance streams
/// (`MR3_TANDBERG_B.264`, `MR4_TANDBERG_C.264`, `MR5_TANDBERG_C.264`), loaded from
/// `OPENH264_CONFORMANCE_DIR`. Because Baseline streams intentionally emit in decode
/// order rather than POC order, compare individual frame digests sorted or in output
/// order against `libavcodec`.
#[test]
#[ignore = "requires OPENH264_CONFORMANCE_DIR pointing to JVT base conformance vectors"]
pub(crate) fn test_full_base_mr3_mr4_mr5_tandberg() {
    let dir =
        std::env::var_os("OPENH264_CONFORMANCE_DIR").expect("OPENH264_CONFORMANCE_DIR must be set");
    let base = Path::new(&dir);

    // MR3_TANDBERG_B has monotonic POCs, so decode order equals display order directly.
    let mr3_path = if base.join("MR3_TANDBERG_B.264").exists() {
        base.join("MR3_TANDBERG_B.264")
    } else {
        base.parent().unwrap_or(base).join("MR3_TANDBERG_B.264")
    };
    let mr3_data = std::fs::read(&mr3_path).unwrap();
    let mr3_aus = common::split_access_units(&mr3_data);
    assert_eq!(mr3_aus.len(), 300);
    for mode in [FeedMode::PerNal, FeedMode::PerAu, FeedMode::NoDelay] {
        let run = match mode {
            FeedMode::PerNal => {
                common::decode_bytes_per_nal(&mr3_data, "MR3_TANDBERG_B.264", false)
            }
            FeedMode::PerAu => common::decode_access_units(&mr3_aus, "MR3_TANDBERG_B.264", false),
            FeedMode::NoDelay => {
                common::decode_access_units_no_delay(&mr3_aus, "MR3_TANDBERG_B.264", false)
            }
        };
        assert_eq!(run.state_union, DECODING_STATE::dsErrorFree);
        assert_eq!(run.frames, 300);
        assert_eq!(run.sha1, "b4ebf9daa9950f6c97b28fdf5ca9413aaaa53d58");
    }

    // MR4_TANDBERG_C and MR5_TANDBERG_C are Baseline streams with non-monotonic POCs
    // and MMCO 5 sequence resets: OpenH264rs intentionally emits Baseline in zero-latency
    // decode order, so verify that sorting the 300 per-frame SHA-1 digests matches
    // libavcodec's 300 per-frame digests byte-for-byte.
    for file in ["MR4_TANDBERG_C.264", "MR5_TANDBERG_C.264"] {
        let path = if base.join(file).exists() {
            base.join(file)
        } else {
            base.parent().unwrap_or(base).join(file)
        };
        let data = std::fs::read(&path).unwrap();
        let aus = common::split_access_units(&data);
        assert_eq!(aus.len(), 300, "{file}: AU count");

        // Decode with OpenH264rs and collect sorted per-frame SHA-1 digests.
        let mut dec = openh264rs::api::codec_api::Decoder::new();
        let mut param = openh264rs::api::codec_api::SDecodingParam::default();
        param.uiTargetDqLayer = u8::MAX;
        param.eEcActiveIdc = openh264rs::api::codec_api::ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
        param.sVideoProperty.eVideoBsType = openh264rs::api::codec_api::VIDEO_BITSTREAM_DEFAULT;
        assert_eq!(
            dec.initialize(&param),
            openh264rs::api::codec_api::CM_RESULT_SUCCESS as std::ffi::c_long
        );

        let mut frame_hashes = Vec::new();
        for au in &aus {
            let mut info = openh264rs::api::codec_api::SBufferInfo::default();
            let st = dec.decode_no_delay(Some(au), &mut info);
            assert_eq!(
                st,
                DECODING_STATE::dsErrorFree,
                "{file}: decode_no_delay state"
            );
            if info.iBufferStatus == 1 {
                let mut h = common::Sha1Hasher::new();
                common::update_hash_from_frame(&mut h, &dec, &info);
                frame_hashes.push(h.digest());
            }
        }
        dec.set_end_of_stream(true);
        let mut info = openh264rs::api::codec_api::SBufferInfo::default();
        let st = dec.decode_no_delay(None, &mut info);
        assert_eq!(st, DECODING_STATE::dsErrorFree);
        if info.iBufferStatus == 1 {
            let mut h = common::Sha1Hasher::new();
            common::update_hash_from_frame(&mut h, &dec, &info);
            frame_hashes.push(h.digest());
        }
        dec.uninitialize();

        assert_eq!(frame_hashes.len(), 300, "{file}: frame count");
        frame_hashes.sort();
        let mut combined_sorted = common::Sha1Hasher::new();
        for fh in &frame_hashes {
            combined_sorted.update(fh.as_bytes());
        }
        assert_eq!(
            combined_sorted.digest(),
            "5d7b7d6ce66d21ef00704d95d481adaafeb5cca4",
            "{file}: sorted per-frame digest"
        );
    }
}

/// Standalone regression test for ITU-T H.264 7.4.1.2.1: two 16x16 Baseline IDR access
/// units, each carrying its own PPS 0 with a different `pic_init_qp_minus26` (-14 => QP 12
/// vs +20 => QP 46) and `slice_qp_delta == 0`. When fed one access unit per `decode()`
/// call, AU 1 is still pending (with `active_pps == None`) when AU 2's PPS 0 arrives;
/// AU 1 must still reconstruct at QP 12 (`Y == 128`) rather than AU 2's QP 46 (`Y == 136`).
#[test]
pub(crate) fn test_pending_picture_preserves_pps_across_same_id_replacement_per_au() {
    const AU1: &[u8] = &[
        0, 0, 0, 1, 103, 66, 128, 10, 218, 113, // SPS 0 (16x16 Baseline, POC type 2)
        0, 0, 0, 1, 104, 206, 3, 188, 128, // PPS 0 (pic_init_qp_minus26 = -14 => QP 12)
        0, 0, 0, 1, 101, 136, 132, 162, 107, // IDR 0 (I_16x16 DC +1)
    ];
    const AU2: &[u8] = &[
        0, 0, 0, 1, 104, 206, 1, 71, 32, // PPS 0 (pic_init_qp_minus26 = +20 => QP 46)
        0, 0, 0, 1, 101, 136, 130, 40, 154, 192, // IDR 1 (I_16x16 DC +1)
    ];
    let mut combined = Vec::with_capacity(AU1.len() + AU2.len());
    combined.extend_from_slice(AU1);
    combined.extend_from_slice(AU2);

    let per_nal = common::decode_bytes_per_nal(&combined, "synthetic_pps0_2au", false);
    let per_au = common::decode_access_units(&[AU1, AU2], "synthetic_pps0_2au", false);
    let no_delay = common::decode_access_units_no_delay(&[AU1, AU2], "synthetic_pps0_2au", false);

    assert_eq!(per_nal.state_union, DECODING_STATE::dsErrorFree);
    assert_eq!(per_au.state_union, DECODING_STATE::dsErrorFree);
    assert_eq!(no_delay.state_union, DECODING_STATE::dsErrorFree);
    assert_eq!(per_nal.frames, 2);
    assert_eq!(per_au.frames, 2);
    assert_eq!(no_delay.frames, 2);
    assert_eq!(per_au.sha1, per_nal.sha1);
    assert_eq!(no_delay.sha1, per_nal.sha1);
}

/// Synthetic 16x16 High-Profile (`profile_idc == 100`) stream exercising `pic_order_cnt_type == 1`
/// with a B slice, display reordering, temporal direct prediction, and a cycle offset
/// (`offset_for_ref_frame[0] = 400`, `offset_for_non_ref_pic = -200`) outside `i8`:
/// - AU 0: IDR (`frame_num = 0, POC = 0`), `I_PCM` `Y = 40, U = 128, V = 128`
/// - AU 1: Ref P (`frame_num = 1, POC = 400`), `I_PCM` `Y = 200, U = 128, V = 128`
/// - AU 2: Non-ref B (`frame_num = 2, POC = 200`), temporal-direct `B_Skip` (`Y = 120`)
///
/// Expected `libavcodec` (`ffmpeg -fps_mode passthrough -pix_fmt yuv420p`) output order is
/// `[Y=40, Y=120, Y=200]` with SHA-1 `77dc767cc1eebdf872061f0a34136f7e995789bf`.
#[test]
pub(crate) fn test_high_profile_poc_type_1_b_slice_reordering_and_large_cycle_offset() {
    const SPS_NAL: &[u8] = &[
        0, 0, 0, 1, 103, 100, 0, 30, 172, 168, 6, 70, 128, 25, 3, 122, 1, 226, 197, 167,
    ];
    const PPS_NAL: &[u8] = &[0, 0, 0, 1, 104, 206, 60, 48];

    let mut au0 = Vec::new();
    au0.extend_from_slice(SPS_NAL);
    au0.extend_from_slice(PPS_NAL);
    au0.extend_from_slice(&[0, 0, 0, 1, 101, 184, 74, 13, 0]);
    au0.extend(std::iter::repeat_n(40u8, 256));
    au0.extend(std::iter::repeat_n(128u8, 128));
    au0.push(0x80);

    let mut au1 = Vec::new();
    au1.extend_from_slice(&[0, 0, 0, 1, 65, 226, 42, 31]);
    au1.extend(std::iter::repeat_n(200u8, 256));
    au1.extend(std::iter::repeat_n(128u8, 128));
    au1.push(0x80);

    let au2: &[u8] = &[0, 0, 0, 1, 1, 169, 5, 40];

    let mut combined = Vec::new();
    combined.extend_from_slice(&au0);
    combined.extend_from_slice(&au1);
    combined.extend_from_slice(au2);

    let row = VectorRow {
        file: "synthetic_high_poc_type1_b.264",
        frames: 3,
        sha1: "77dc767cc1eebdf872061f0a34136f7e995789bf",
    };
    for mode in [FeedMode::PerNal, FeedMode::PerAu, FeedMode::NoDelay] {
        assert_row_in_mode(&row, &combined, mode);
    }
}

fn assert_rejected_cleanly(data: &[u8], label: &str, expected_mask: i32) {
    let per_nal = common::decode_bytes_per_nal(data, label, false);
    assert_eq!(per_nal.frames, 0, "{label} (PerNal): expected 0 frames");
    assert_eq!(
        per_nal.state_union.0 & expected_mask,
        expected_mask,
        "{label} (PerNal): expected mask {expected_mask:#x} in {:?}",
        per_nal.state_union
    );

    let mut dec = openh264rs::api::codec_api::Decoder::new();
    let mut param = openh264rs::api::codec_api::SDecodingParam::default();
    param.uiTargetDqLayer = u8::MAX;
    param.eEcActiveIdc = openh264rs::api::codec_api::ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
    param.sVideoProperty.eVideoBsType = openh264rs::api::codec_api::VIDEO_BITSTREAM_DEFAULT;
    assert_eq!(
        dec.initialize(&param),
        openh264rs::api::codec_api::CM_RESULT_SUCCESS as std::ffi::c_long
    );
    let mut info = openh264rs::api::codec_api::SBufferInfo::default();
    let st = dec.decode(Some(data), &mut info);
    assert_eq!(
        info.iBufferStatus, 0,
        "{label} (WholeBuf): expected no output"
    );
    dec.set_end_of_stream(true);
    let st_eos = dec.decode(None, &mut info);
    assert_eq!(
        info.iBufferStatus, 0,
        "{label} (WholeBuf EOS): expected no output"
    );
    assert_eq!(
        (st.0 | st_eos.0) & expected_mask,
        expected_mask,
        "{label} (WholeBuf): expected mask {expected_mask:#x}, got {:?}",
        DECODING_STATE(st.0 | st_eos.0)
    );
    dec.uninitialize();

    let no_delay = common::decode_access_units_no_delay(&[data], label, false);
    assert_eq!(no_delay.frames, 0, "{label} (NoDelay): expected 0 frames");
    assert_eq!(
        no_delay.state_union.0 & expected_mask,
        expected_mask,
        "{label} (NoDelay): expected mask {expected_mask:#x} in {:?}",
        no_delay.state_union
    );
}

#[test]
pub(crate) fn test_out_of_scope_interlaced_mbaff_rejected_cleanly() {
    // Main Profile (`profile_idc == 77`, level 3.0) with `frame_mbs_only_flag == 0`
    // and `mb_adaptive_frame_field_flag == 1`, followed by PPS 0.
    // Fails with `ERR_INFO_UNSUPPORTED_MBAFF` (`dsBitstreamError`), leaving PPS 0 with `dsNoParamSets`.
    const INTERLACED_MAIN_SPS_PPS: &[u8] = &[
        0, 0, 0, 1, 103, 77, 0, 30, 218, 108, 128, // SPS 0 (interlaced MBAFF)
        0, 0, 0, 1, 104, 238, 60, 128, // PPS 0
    ];
    assert_rejected_cleanly(
        INTERLACED_MAIN_SPS_PPS,
        "synthetic_interlaced_main",
        DECODING_STATE::dsBitstreamError.0 | DECODING_STATE::dsNoParamSets.0,
    );
}

#[test]
pub(crate) fn test_out_of_scope_high10_profile_rejected_cleanly() {
    // High 10 Profile (`profile_idc == 110`, level 3.0, 4:2:0 10-bit), followed by PPS 0.
    // Unsupported `profile_idc` is not stored, so PPS 0 reports `dsNoParamSets`.
    const HIGH10_SPS_PPS: &[u8] = &[
        0, 0, 0, 1, 103, 110, 0, 30, 166, 203, 79, 32, // SPS 0 (High 10)
        0, 0, 0, 1, 104, 238, 60, 128, // PPS 0
    ];
    assert_rejected_cleanly(
        HIGH10_SPS_PPS,
        "synthetic_high10_profile",
        DECODING_STATE::dsNoParamSets.0,
    );
}

#[test]
pub(crate) fn test_out_of_scope_high422_profile_rejected_cleanly() {
    // High 4:2:2 Profile (`profile_idc == 122`, level 3.0, 4:2:2 8-bit), followed by PPS 0.
    // Unsupported `profile_idc` is not stored, so PPS 0 reports `dsNoParamSets`.
    const HIGH422_SPS_PPS: &[u8] = &[
        0, 0, 0, 1, 103, 122, 0, 30, 188, 180, 242, 0, // SPS 0 (High 4:2:2)
        0, 0, 0, 1, 104, 238, 60, 128, // PPS 0
    ];
    assert_rejected_cleanly(
        HIGH422_SPS_PPS,
        "synthetic_high422_profile",
        DECODING_STATE::dsNoParamSets.0,
    );
}

#[test]
pub(crate) fn test_out_of_scope_high_profile_10bit_422_and_level60_sps_errors() {
    // `profile_idc == 100` (High) with `chroma_format_idc == 2` (4:2:2) -> `ERR_INFO_UNSUPPORTED_NON_BASELINE` (`dsBitstreamError`).
    const HIGH_422_SPS_PPS: &[u8] = &[
        0, 0, 0, 1, 103, 100, 0, 30, 172, 52,
        200, // seq_id=0, chroma_format_idc=2 (ue=2 -> 011)
        0, 0, 0, 1, 104, 206, 60, 128,
    ];
    assert_rejected_cleanly(
        HIGH_422_SPS_PPS,
        "synthetic_high100_422",
        DECODING_STATE::dsBitstreamError.0 | DECODING_STATE::dsNoParamSets.0,
    );

    // `profile_idc == 100` (High) with `bit_depth_luma_minus8 == 2` (10-bit) -> `ERR_INFO_UNSUPPORTED_NON_BASELINE` (`dsBitstreamError`).
    const HIGH_10BIT_SPS_PPS: &[u8] = &[
        0, 0, 0, 1, 103, 100, 0, 30, 172, 155,
        32, // seq_id=0, chroma=1, bit_depth_luma_minus8=2
        0, 0, 0, 1, 104, 206, 60, 128,
    ];
    assert_rejected_cleanly(
        HIGH_10BIT_SPS_PPS,
        "synthetic_high100_10bit",
        DECODING_STATE::dsBitstreamError.0 | DECODING_STATE::dsNoParamSets.0,
    );

    // `profile_idc == 100` (High) at Level 6.0 (`level_idc == 60`) -> `GetLevelLimits` returns `None` (`dsBitstreamError`).
    const HIGH_LEVEL60_SPS_PPS: &[u8] = &[
        0, 0, 0, 1, 103, 100, 0, 60, 172, 168, 6, 70, 128, 0, 0, 0, 1, 104, 206, 60, 128,
    ];
    assert_rejected_cleanly(
        HIGH_LEVEL60_SPS_PPS,
        "synthetic_high100_level60",
        DECODING_STATE::dsBitstreamError.0 | DECODING_STATE::dsNoParamSets.0,
    );
}

#[test]
#[ignore = "requires OPENH264_CONFORMANCE_DIR pointing to full 195-file JVT conformance mirror"]
pub(crate) fn test_external_195_conformance_streams_never_panic() {
    let base = PathBuf::from(
        std::env::var_os("OPENH264_CONFORMANCE_DIR").expect("OPENH264_CONFORMANCE_DIR must be set"),
    );
    let mut files = Vec::new();
    for dir in [base.clone(), base.join("FRext")] {
        for entry in
            std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        {
            let p = entry.unwrap().path();
            if p.is_file() && p.file_name().is_some_and(|n| n != "md5sum") {
                files.push(p);
            }
        }
    }
    files.sort();
    assert_eq!(
        files.len(),
        195,
        "expected 195 JVT conformance streams (137 base + 58 FRExt)"
    );

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy();
        let data = std::fs::read(path).unwrap();
        let _ = common::decode_bytes_per_nal(&data, &name, false);
        let aus = common::split_access_units(&data);
        let _ = common::decode_access_units(&aus, &name, false);
        let _ = common::decode_access_units_no_delay(&aus, &name, false);
    }
}
