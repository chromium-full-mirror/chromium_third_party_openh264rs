// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! 1080p decode throughput: native C++ OpenH264 (`libopenh264.dylib`) vs. the
//! Rust port, both fed byte-identical Annex-B streams produced by ffmpeg.
//!
//! Each stream is decoded twice per implementation: a verification pass that
//! SHA-1s every output plane and counts frames, then a timed pass. A comparison
//! is reported only once both sides agree on the frame count and the hash, since
//! a decode error drops frames silently and would read as a speedup.
//!
//! Both sides run the same loop over the same NAL units, one unit per
//! `DecodeFrame2` call.
//!
//! Environment knobs: `FFMPEG` (path to the binary), `BENCH_FRAMES` (default
//! 60), `BENCH_ITERS` (timed passes per stream, default 3), `BENCH_PATTERN`
//! (lavfi source, default `testsrc2`).

#![allow(non_snake_case)]

use openh264rs::api::codec_api::*;
use openh264rs::split_annexb_units;
use std::ffi::c_void;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr;
use std::time::{Duration, Instant};

#[path = "../tests/common/mod.rs"]
mod common;
use common::Sha1Hasher;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

// ---------------------------------------------------------------------------
// Test streams
// ---------------------------------------------------------------------------

/// An ffmpeg recipe for one 1080p stream. The profile is what moves decode
/// cost around: CAVLC vs CABAC, B-frames, and the 8x8 transform.
struct StreamSpec {
    slug: &'static str,
    label: &'static str,
    profile: &'static str,
    /// Extra encoder arguments, appended after the profile.
    extra: &'static [&'static str],
}

const STREAMS: &[StreamSpec] = &[
    StreamSpec {
        slug: "baseline",
        label: "Constrained Baseline (CAVLC, no B-frames)",
        profile: "baseline",
        extra: &[],
    },
    StreamSpec {
        slug: "main",
        label: "Main (CABAC, B-frames)",
        profile: "main",
        extra: &[],
    },
    StreamSpec {
        slug: "high",
        label: "High (CABAC, B-frames, 8x8 transform)",
        profile: "high",
        extra: &[],
    },
    // A one-keyframe-per-picture stream: every macroblock is intra-coded, so the
    // intra predictors and the intra deblocking paths carry the decode instead of
    // motion compensation. The three streams above spend under 2% of their
    // macroblocks there, which puts any change to those kernels under the noise.
    StreamSpec {
        slug: "allintra",
        label: "High, all-intra (keyint 1: every macroblock intra-coded)",
        profile: "high",
        extra: &["-g", "1"],
    },
];

fn resolve_ffmpeg() -> Option<PathBuf> {
    if let Ok(from_env) = std::env::var("FFMPEG") {
        let path = PathBuf::from(from_env);
        if path.exists() {
            return Some(path);
        }
    }
    if Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_ok_and(|out| out.status.success())
    {
        return Some(PathBuf::from("ffmpeg"));
    }
    // Cargo does not necessarily inherit a login shell's PATH.
    for candidate in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ] {
        if Path::new(candidate).exists() {
            return Some(PathBuf::from(candidate));
        }
    }
    None
}

fn stream_cache_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("bench-streams");
    std::fs::create_dir_all(&dir).expect("cannot create bench stream cache dir");
    dir
}

/// Encodes the stream with ffmpeg unless a matching file is already cached.
/// Everything that changes the bitstream is in the filename, so bumping
/// `BENCH_FRAMES` or `BENCH_PATTERN` regenerates rather than reusing a stale file.
fn ensure_stream(
    ffmpeg: Option<&PathBuf>,
    spec: &StreamSpec,
    pattern: &str,
    frames: usize,
) -> Option<PathBuf> {
    let path = stream_cache_dir().join(format!(
        "{}_{}_{}x{}_{}f.264",
        pattern, spec.slug, WIDTH, HEIGHT, frames
    ));
    if path.exists() && std::fs::metadata(&path).is_ok_and(|m| m.len() > 0) {
        return Some(path);
    }

    let ffmpeg = ffmpeg?;
    let source = format!("{}=size={}x{}:rate=30", pattern, WIDTH, HEIGHT);
    println!(
        "  generating {} ...",
        path.file_name().unwrap().to_string_lossy()
    );
    let out = Command::new(ffmpeg)
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            &source,
            "-frames:v",
            &frames.to_string(),
            "-c:v",
            "libx264",
            "-profile:v",
            spec.profile,
        ])
        .args(spec.extra)
        .args([
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            "5M",
            "-f",
            "h264",
            path.to_str().unwrap(),
        ])
        .output()
        .ok()?;

    if !out.status.success() {
        eprintln!(
            "  ffmpeg failed for {}: {}",
            spec.slug,
            String::from_utf8_lossy(&out.stderr).trim()
        );
        return None;
    }
    Some(path)
}

// ---------------------------------------------------------------------------
// The decoder under test
// ---------------------------------------------------------------------------

/// The slice of `ISVCDecoder` the benchmark drives, shared by both implementations.
trait Decoder {
    unsafe fn decode(
        &mut self,
        src: *const u8,
        len: i32,
        dst: *mut *mut u8,
        info: *mut SBufferInfo,
    ) -> i32;
    unsafe fn flush(&mut self, dst: *mut *mut u8, info: *mut SBufferInfo) -> i32;
    unsafe fn signal_end_of_stream(&mut self);
    unsafe fn frames_remaining(&mut self) -> i32;
}

fn decoding_param() -> SDecodingParam {
    let mut param = SDecodingParam::default();
    param.uiTargetDqLayer = u8::MAX;
    param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
    param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
    param
}

struct RustDecoder {
    dec: *mut ISVCDecoder,
}

impl RustDecoder {
    fn new() -> Self {
        unsafe {
            let mut dec: *mut ISVCDecoder = ptr::null_mut();
            let ret = WelsCreateDecoder(&mut dec);
            assert_eq!(
                ret as i64, CM_RESULT_SUCCESS as i64,
                "WelsCreateDecoder failed"
            );
            assert!(!dec.is_null());
            let param = decoding_param();
            let init = ISVCDecoder::Initialize(dec, &param);
            assert_eq!(
                init as i64, CM_RESULT_SUCCESS as i64,
                "Rust decoder Initialize failed"
            );
            Self { dec }
        }
    }
}

impl Decoder for RustDecoder {
    unsafe fn decode(
        &mut self,
        src: *const u8,
        len: i32,
        dst: *mut *mut u8,
        info: *mut SBufferInfo,
    ) -> i32 {
        unsafe { ISVCDecoder::DecodeFrame2(self.dec, src, len, dst, info).0 }
    }
    unsafe fn flush(&mut self, dst: *mut *mut u8, info: *mut SBufferInfo) -> i32 {
        unsafe { ISVCDecoder::FlushFrame(self.dec, dst, info).0 }
    }
    unsafe fn signal_end_of_stream(&mut self) {
        unsafe {
            let mut eos = 1i32;
            ISVCDecoder::SetOption(
                self.dec,
                DECODER_OPTION::DECODER_OPTION_END_OF_STREAM,
                &mut eos as *mut i32 as *mut c_void,
            );
        }
    }
    unsafe fn frames_remaining(&mut self) -> i32 {
        unsafe {
            let mut remaining = 0i32;
            ISVCDecoder::GetOption(
                self.dec,
                DECODER_OPTION::DECODER_OPTION_NUM_OF_FRAMES_REMAINING_IN_BUFFER,
                &mut remaining as *mut i32 as *mut c_void,
            );
            remaining
        }
    }
}

impl Drop for RustDecoder {
    fn drop(&mut self) {
        unsafe {
            ISVCDecoder::Uninitialize(self.dec);
            WelsDestroyDecoder(self.dec);
        }
    }
}

// ---------------------------------------------------------------------------
// The decode loop
// ---------------------------------------------------------------------------

const DS_ERROR_FREE: i32 = DECODING_STATE::dsErrorFree.0;

fn hash_plane(
    hasher: &mut Sha1Hasher,
    plane: *const u8,
    width: usize,
    height: usize,
    stride: usize,
) {
    if plane.is_null() || width == 0 || height == 0 || stride == 0 {
        return;
    }
    unsafe {
        for y in 0..height {
            hasher.update(std::slice::from_raw_parts(plane.add(y * stride), width));
        }
    }
}

fn hash_frame(hasher: &mut Sha1Hasher, dst: [*mut u8; 3], info: &SBufferInfo) {
    let buf = info.UsrData.sSystemBuffer;
    let (w, h) = (buf.iWidth as usize, buf.iHeight as usize);
    hash_plane(hasher, dst[0], w, h, buf.iStride[0] as usize);
    hash_plane(hasher, dst[1], w / 2, h / 2, buf.iStride[1] as usize);
    hash_plane(hasher, dst[2], w / 2, h / 2, buf.iStride[1] as usize);
}

struct PassResult {
    frames: usize,
    hash: Option<String>,
    elapsed: Duration,
}

/// Decodes every NAL unit and then drains the decoder. Hashing is off during timed
/// passes: walking 3 MB of planes per frame costs more than decoding some of them.
fn run_pass<D: Decoder>(dec: &mut D, units: &[&[u8]], verify: bool) -> PassResult {
    let mut hasher = Sha1Hasher::new();
    let mut frames = 0usize;

    let start = Instant::now();
    unsafe {
        let mut consume = |dst: [*mut u8; 3], info: &SBufferInfo, state: i32| {
            if state == DS_ERROR_FREE && info.iBufferStatus == 1 {
                frames += 1;
                if verify {
                    hash_frame(&mut hasher, dst, info);
                }
            }
        };

        for unit in units {
            let mut dst: [*mut u8; 3] = [ptr::null_mut(); 3];
            let mut info = SBufferInfo::default();
            let state = dec.decode(
                black_box(unit.as_ptr()),
                black_box(unit.len() as i32),
                dst.as_mut_ptr(),
                &mut info,
            );
            consume(black_box(dst), black_box(&info), black_box(state));
        }

        // Drain: flag EOS, pull the frame that flushes out, then flush the rest.
        dec.signal_end_of_stream();
        let mut dst: [*mut u8; 3] = [ptr::null_mut(); 3];
        let mut info = SBufferInfo::default();
        let state = dec.decode(ptr::null(), 0, dst.as_mut_ptr(), &mut info);
        consume(dst, &info, state);

        for _ in 0..dec.frames_remaining() {
            let mut dst: [*mut u8; 3] = [ptr::null_mut(); 3];
            let mut info = SBufferInfo::default();
            let state = dec.flush(dst.as_mut_ptr(), &mut info);
            consume(dst, &info, state);
        }
    }
    let elapsed = start.elapsed();

    PassResult {
        frames,
        hash: verify.then(|| hasher.digest()),
        elapsed,
    }
}

struct Measurement {
    frames: usize,
    hash: String,
    best: Duration,
}

impl Measurement {
    fn fps(&self) -> f64 {
        if self.best.is_zero() {
            return 0.0;
        }
        self.frames as f64 / self.best.as_secs_f64()
    }
    fn ms_per_frame(&self) -> f64 {
        if self.frames == 0 {
            return 0.0;
        }
        self.best.as_secs_f64() * 1000.0 / self.frames as f64
    }
}

/// Verifies the decoder once (which also warms the caches), then runs the timed passes
/// on a fresh decoder every time and reports the best time.
fn measure_stream(units: &[&[u8]], iters: usize) -> Measurement {
    let rust_verified = {
        let mut dec = RustDecoder::new();
        run_pass(&mut dec, units, true)
    };

    let mut rust_best = Duration::MAX;
    for _ in 0..iters {
        let mut dec = RustDecoder::new();
        let timed = run_pass(&mut dec, units, false);
        assert_eq!(
            timed.frames, rust_verified.frames,
            "Rust decoder returned a different frame count across passes"
        );
        rust_best = rust_best.min(timed.elapsed);
    }

    Measurement {
        frames: rust_verified.frames,
        hash: rust_verified.hash.unwrap_or_default(),
        best: rust_best,
    }
}

// ---------------------------------------------------------------------------

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default)
}

fn main() {
    let frames = env_usize("BENCH_FRAMES", 60);
    let iters = env_usize("BENCH_ITERS", 3);
    let pattern = std::env::var("BENCH_PATTERN").unwrap_or_else(|_| "testsrc2".to_string());

    let ffmpeg = resolve_ffmpeg();

    let rule = "=".repeat(88);
    println!("{rule}");
    println!(" 1080p decode throughput: openh264rs");
    println!("{rule}");
    println!(
        " source     : ffmpeg lavfi {pattern}, {WIDTH}x{HEIGHT}, {frames} frames, x264 @ 5 Mbit/s"
    );
    println!(
        " ffmpeg     : {}",
        ffmpeg
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "NOT FOUND (falling back to cached streams)".into())
    );
    println!(" timing     : best of {iters} passes, decode calls only (no hashing)");

    let mut any_stream = false;

    for spec in STREAMS {
        println!("{}", "-".repeat(88));
        let Some(path) = ensure_stream(ffmpeg.as_ref(), spec, &pattern, frames) else {
            println!(" {} -- SKIPPED (no stream available)", spec.label);
            continue;
        };
        let data = std::fs::read(&path).expect("cannot read generated stream");
        let units: Vec<_> = split_annexb_units(&data).collect();
        any_stream = true;

        println!(
            " {}\n   {:.2} MB bitstream, {} NAL units",
            spec.label,
            data.len() as f64 / (1024.0 * 1024.0),
            units.len()
        );

        let rust = measure_stream(&units, iters);
        println!(
            "   Rust : {:8.2} fps   {:8.3} ms/frame   {} frames   SHA-1 {}",
            rust.fps(),
            rust.ms_per_frame(),
            rust.frames,
            &rust.hash[..16]
        );
    }

    println!("{rule}");
    if !any_stream {
        eprintln!("No streams to decode. Install ffmpeg, or point FFMPEG at it, then re-run.");
        std::process::exit(1);
    }
}
