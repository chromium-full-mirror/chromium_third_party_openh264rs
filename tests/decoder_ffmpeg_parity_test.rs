// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Differential decoder parity tests against `ffmpeg` (`libx264` + `libavcodec`).
//!
//! Each test encodes a `mandelbrot` lavfi clip with a specific H.264 profile and encoder
//! feature combination, decodes it with `ffmpeg` (`libavcodec`) to obtain the reference raw
//! `yuv420p` SHA-1 digest, and verifies that `openh264rs` reproduces the exact same frame count
//! and SHA-1 across three feed modes:
//! - per-NAL (`decode(Some(nal))` + `decode(None)`),
//! - per-access-unit (`decode(Some(au))`, no intermediate `None` calls), and
//! - no-delay (`decode_no_delay(Some(au))`).
//!
//! When `ffmpeg` with `libx264` is not available at build time (`build.rs` does not emit
//! `cfg(has_ffmpeg_libx264)`), all tests in this file are marked `#[ignore]`. Set
//! `FFMPEG=/path/to/ffmpeg` to override binary discovery.

#[path = "common/mod.rs"]
mod common;

use openh264rs::api::codec_api::DECODING_STATE;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

const SRC_WIDTH: usize = 640;
const SRC_HEIGHT: usize = 360;
const SRC_FRAMES: usize = 20;
const BASE_X264_PARAMS: &str = "threads=1:keyint=10:min-keyint=10:scenecut=0";

#[derive(Copy, Clone, Debug)]
enum FeedMode {
    PerNal,
    PerAu,
    NoDelay,
}

struct StreamRecipe {
    name: &'static str,
    profile: &'static str,
    extra_ffmpeg_args: &'static [&'static str],
    extra_x264_params: &'static str,
    width: usize,
    height: usize,
    frames: usize,
}

struct TempFileGuard(PathBuf);

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn probe_ffmpeg_bin(bin: &OsStr) -> bool {
    Command::new(bin)
        .args(["-hide_banner", "-v", "error", "-h", "encoder=libx264"])
        .output()
        .is_ok_and(|out| out.status.success() && out.stdout.starts_with(b"Encoder libx264"))
}

fn resolve_ffmpeg() -> &'static OsStr {
    static FFMPEG_BIN: OnceLock<OsString> = OnceLock::new();
    FFMPEG_BIN
        .get_or_init(|| {
            if let Some(from_env) = std::env::var_os("FFMPEG")
                && !from_env.is_empty()
            {
                return from_env;
            }
            for candidate in [
                "ffmpeg",
                "/opt/homebrew/bin/ffmpeg",
                "/usr/local/bin/ffmpeg",
                "/usr/bin/ffmpeg",
            ] {
                if probe_ffmpeg_bin(OsStr::new(candidate)) {
                    return OsString::from(candidate);
                }
            }
            panic!("ffmpeg binary with libx264 not found; set FFMPEG or install ffmpeg")
        })
        .as_os_str()
}

fn work_dir() -> PathBuf {
    let dir = option_env!("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("openh264rs-ffmpeg-parity");
    std::fs::create_dir_all(&dir)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", dir.display()));
    dir
}

fn ensure_mandelbrot_source(ffmpeg: &OsStr) -> &'static PathBuf {
    static SOURCE_Y4M: OnceLock<PathBuf> = OnceLock::new();
    SOURCE_Y4M.get_or_init(|| {
        let dir = work_dir();
        let final_path = dir.join(format!(
            "mandelbrot_{SRC_WIDTH}x{SRC_HEIGHT}_{SRC_FRAMES}f.y4m"
        ));
        let min_bytes = (SRC_WIDTH * SRC_HEIGHT * 3 / 2 * SRC_FRAMES) as u64;
        if std::fs::metadata(&final_path).is_ok_and(|m| m.len() >= min_bytes) {
            return final_path;
        }

        let tmp_path = dir.join(format!(
            "mandelbrot_{SRC_WIDTH}x{SRC_HEIGHT}_{SRC_FRAMES}f.{}.y4m.tmp",
            std::process::id()
        ));
        let _guard = TempFileGuard(tmp_path.clone());
        let source = format!("mandelbrot=size={SRC_WIDTH}x{SRC_HEIGHT}:rate=30");
        let output = Command::new(ffmpeg)
            .args([
                "-hide_banner",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                &source,
                "-frames:v",
                &SRC_FRAMES.to_string(),
                "-pix_fmt",
                "yuv420p",
                "-f",
                "yuv4mpegpipe",
                "-y",
            ])
            .arg(&tmp_path)
            .output()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", Path::new(ffmpeg).display()));
        assert!(
            output.status.success(),
            "ffmpeg mandelbrot generation failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        std::fs::rename(&tmp_path, &final_path).unwrap_or_else(|e| {
            panic!(
                "failed to move {} to {}: {e}",
                tmp_path.display(),
                final_path.display()
            )
        });
        final_path
    })
}

fn encode_recipe(ffmpeg: &OsStr, src_y4m: &Path, recipe: &StreamRecipe, out_path: &Path) {
    let x264_params = if recipe.extra_x264_params.is_empty() {
        BASE_X264_PARAMS.to_string()
    } else {
        format!("{BASE_X264_PARAMS}:{}", recipe.extra_x264_params)
    };

    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-v", "error", "-i"])
        .arg(src_y4m)
        .args(recipe.extra_ffmpeg_args)
        .args([
            "-c:v",
            "libx264",
            "-profile:v",
            recipe.profile,
            "-pix_fmt",
            "yuv420p",
            "-x264-params",
            &x264_params,
            "-bsf:v",
            "h264_mp4toannexb",
            "-f",
            "h264",
            "-y",
        ])
        .arg(out_path)
        .output()
        .unwrap_or_else(|e| panic!("failed to run ffmpeg for {}: {e}", recipe.name));
    assert!(
        output.status.success(),
        "{}: ffmpeg encode failed ({}): {}",
        recipe.name,
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

fn decode_reference_sha1(ffmpeg: &OsStr, bitstream_path: &Path, recipe: &StreamRecipe) -> String {
    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-v", "error", "-f", "h264", "-i"])
        .arg(bitstream_path)
        .args([
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-",
        ])
        .output()
        .unwrap_or_else(|e| panic!("failed to run ffmpeg decode for {}: {e}", recipe.name));
    assert!(
        output.status.success(),
        "{}: ffmpeg decode failed ({}): {}",
        recipe.name,
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let frame_bytes = recipe.width * recipe.height * 3 / 2;
    let expected_bytes = frame_bytes * recipe.frames;
    assert_eq!(
        output.stdout.len(),
        expected_bytes,
        "{}: ffmpeg decoded {} raw bytes, expected {} ({} frames of {}x{})",
        recipe.name,
        output.stdout.len(),
        expected_bytes,
        recipe.frames,
        recipe.width,
        recipe.height
    );

    let mut hasher = common::Sha1Hasher::new();
    hasher.update(&output.stdout);
    hasher.digest()
}

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

fn run_x264_recipe(recipe: &StreamRecipe) {
    let ffmpeg = resolve_ffmpeg();
    let src_y4m = ensure_mandelbrot_source(ffmpeg);
    let bitstream_path = work_dir().join(format!("{}.{}.264", recipe.name, std::process::id()));
    let _guard = TempFileGuard(bitstream_path.clone());

    encode_recipe(ffmpeg, src_y4m, recipe, &bitstream_path);
    let expected_sha1 = decode_reference_sha1(ffmpeg, &bitstream_path, recipe);
    let data = std::fs::read(&bitstream_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", bitstream_path.display()));

    let label = format!("{}.264", recipe.name);
    for mode in [FeedMode::PerNal, FeedMode::PerAu, FeedMode::NoDelay] {
        assert_stream_in_mode(&label, recipe.frames, &expected_sha1, &data, mode);
    }
}

macro_rules! x264_stream_test {
    (
        $test_fn:ident,
        $name:literal,
        profile = $profile:literal,
        args = [$($arg:literal),*],
        params = $params:literal,
        size = ($w:expr, $h:expr),
        frames = $frames:expr
    ) => {
        #[test]
        #[cfg_attr(not(has_ffmpeg_libx264), ignore = "ffmpeg with libx264 not found")]
        pub(crate) fn $test_fn() {
            const RECIPE: StreamRecipe = StreamRecipe {
                name: $name,
                profile: $profile,
                extra_ffmpeg_args: &[$($arg),*],
                extra_x264_params: $params,
                width: $w,
                height: $h,
                frames: $frames,
            };
            run_x264_recipe(&RECIPE);
        }
    };
}

x264_stream_test!(
    test_x264_high_default,
    "high_default",
    profile = "high",
    args = [],
    params = "",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_cavlc,
    "high_cavlc",
    profile = "high",
    args = [],
    params = "cabac=0",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_no8x8dct,
    "high_no8x8dct",
    profile = "high",
    args = [],
    params = "8x8dct=0",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_cavlc_no_b,
    "high_cavlc_no_b",
    profile = "high",
    args = [],
    params = "cabac=0:bframes=0",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_no_b,
    "high_no_b",
    profile = "high",
    args = [],
    params = "bframes=0",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_bpyramid_ref8,
    "high_bpyramid_ref8",
    profile = "high",
    args = [],
    params = "bframes=5:b-pyramid=normal:ref=8:b-adapt=2",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_bpyramid_strict,
    "high_bpyramid_strict",
    profile = "high",
    args = [],
    params = "bframes=3:b-pyramid=strict:ref=4",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_bframes16,
    "high_bframes16",
    profile = "high",
    args = [],
    params = "keyint=30:min-keyint=30:bframes=16:b-adapt=0:ref=16:b-pyramid=normal",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_direct_temporal,
    "high_direct_temporal",
    profile = "high",
    args = [],
    params = "direct=temporal",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_direct_spatial,
    "high_direct_spatial",
    profile = "high",
    args = [],
    params = "direct=spatial",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_direct_temporal_cavlc,
    "high_direct_temporal_cavlc",
    profile = "high",
    args = [],
    params = "direct=temporal:cabac=0",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_weightp_fade,
    "high_weightp_fade",
    profile = "high",
    args = ["-vf", "fade=in:0:8,fade=out:12:8"],
    params = "weightp=2:weightb=1",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_weightb_cavlc_fade,
    "high_weightb_cavlc_fade",
    profile = "high",
    args = ["-vf", "fade=in:0:8,fade=out:12:8"],
    params = "weightp=2:weightb=1:cabac=0",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_cqm_jvt,
    "high_cqm_jvt",
    profile = "high",
    args = [],
    params = "cqm=jvt",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_cqm_jvt_cavlc,
    "high_cqm_jvt_cavlc",
    profile = "high",
    args = [],
    params = "cqm=jvt:cabac=0",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_slices4,
    "high_slices4",
    profile = "high",
    args = [],
    params = "slices=4",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_slices4_cavlc,
    "high_slices4_cavlc",
    profile = "high",
    args = [],
    params = "slices=4:cabac=0",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_intra_only,
    "high_intra_only",
    profile = "high",
    args = [],
    params = "keyint=1",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_intra_only_cavlc,
    "high_intra_only_cavlc",
    profile = "high",
    args = [],
    params = "keyint=1:cabac=0",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_partitions_all,
    "high_partitions_all",
    profile = "high",
    args = [],
    params = "partitions=all:me=umh:subme=9:mixed-refs=1:ref=5",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_partitions_all_cavlc,
    "high_partitions_all_cavlc",
    profile = "high",
    args = [],
    params = "partitions=all:me=umh:subme=9:mixed-refs=1:ref=5:cabac=0",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_constrained_intra,
    "high_constrained_intra",
    profile = "high",
    args = [],
    params = "constrained-intra=1",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_deblock_off,
    "high_deblock_off",
    profile = "high",
    args = [],
    params = "no-deblock=1",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_deblock_m3_3,
    "high_deblock_m3_3",
    profile = "high",
    args = [],
    params = "deblock=-3,3",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_chroma_qp_off,
    "high_chroma_qp_off",
    profile = "high",
    args = [],
    params = "chroma-qp-offset=6",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_open_gop,
    "high_open_gop",
    profile = "high",
    args = [],
    params = "open-gop=1:bframes=3",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_intra_refresh,
    "high_intra_refresh",
    profile = "high",
    args = [],
    params = "intra-refresh=1:bframes=0",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_aud_hrd,
    "high_aud_hrd",
    profile = "high",
    args = [],
    params = "aud=1:nal-hrd=vbr:vbv-maxrate=2000:vbv-bufsize=2000",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_crop_638x358,
    "high_crop_638x358",
    profile = "high",
    args = ["-vf", "crop=638:358:0:0"],
    params = "",
    size = (638, 358),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_crop_322x182,
    "high_crop_322x182",
    profile = "high",
    args = ["-vf", "scale=322:182"],
    params = "",
    size = (322, 182),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_qp10,
    "high_qp10",
    profile = "high",
    args = [],
    params = "qp=10",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_qp51,
    "high_qp51",
    profile = "high",
    args = [],
    params = "qp=51",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_trellis2_psy,
    "high_trellis2_psy",
    profile = "high",
    args = [],
    params = "trellis=2:psy-rd=1.0,0.2:aq-mode=2",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_l51_1080p,
    "high_l51_1080p",
    profile = "high",
    args = ["-vf", "scale=1920:1080", "-frames:v", "12"],
    params = "level=5.1:ref=4",
    size = (1920, 1080),
    frames = 12
);
x264_stream_test!(
    test_x264_high_tiny_16x16,
    "high_tiny_16x16",
    profile = "high",
    args = ["-vf", "scale=16:16"],
    params = "",
    size = (16, 16),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_b_refs_poc_wrap,
    "high_b_refs_poc_wrap",
    profile = "high",
    args = [],
    params = "bframes=3:keyint=250:min-keyint=250",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_high_dpb_interleave,
    "high_dpb_interleave",
    profile = "high",
    args = [],
    params = "bframes=2:ref=16:keyint=250:min-keyint=250:b-pyramid=none",
    size = (640, 360),
    frames = SRC_FRAMES
);
x264_stream_test!(
    test_x264_main_default,
    "main_default",
    profile = "main",
    args = [],
    params = "8x8dct=0",
    size = (640, 360),
    frames = SRC_FRAMES
);
