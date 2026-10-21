// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! A consumer that installs no trace callback still gets output:
//! `welsCodecTrace::welsCodecTrace()` installs `welsStderrTrace` at
//! `WELS_LOG_DEFAULT`.
//!
//! The two constructors disagree on the level. `welsCodecTrace()` sets
//! `WELS_LOG_WARNING`; `CWelsDecoder::CWelsDecoder()` then calls
//! `SetTraceLevel (WELS_LOG_ERROR)` (`welsDecoderExt.cpp:164`) and
//! `CWelsH264SVCEncoder` does not (`welsEncoderExt.cpp:166`). So the decoder's default
//! is ERROR and the encoder's is WARNING.
//!
//! # Capturing stderr
//!
//! `welsStderrTrace` writes to fd 2, so libtest's output capture does not see it and
//! an in-process reader would need `dup2` surgery that races every other test in the
//! binary. Each case instead re-executes this test binary as a child, with an
//! environment variable selecting the case, and reads the child's stderr: one process
//! per case, no fd games, no ordering assumptions.

#[path = "common/mod.rs"]
mod common;

use openh264rs::api::codec_api::*;
use std::process::Command;

const CHILD_ENV: &str = "OPENH264_TRACE_SINK_CASE";

/// Runs `case` in a child copy of this binary and returns everything it wrote to
/// stderr.
fn stderr_of(case: &str) -> String {
    let exe = std::env::current_exe().expect("current_exe");
    #[cfg(is_gtest_unittests)]
    let args = [
        "--gtest_filter=OpenH264Rs*.the_child_case",
        "--single-process-tests",
    ];
    #[cfg(not(is_gtest_unittests))]
    let args = ["the_child_case", "--exact", "--nocapture"];
    let out = Command::new(exe)
        .args(args)
        .env(CHILD_ENV, case)
        .output()
        .expect("re-exec the test binary");
    assert!(
        out.status.success(),
        "child case {case} exited {:?}; stderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// The child's body. A no-op in the parent, where `CHILD_ENV` is unset.
#[test]
pub(crate) fn the_child_case() {
    let Ok(case) = std::env::var(CHILD_ENV) else {
        return;
    };
    unsafe {
        match case.as_str() {
            // `decode_parser` on an uninitialized decoder logs at `WELS_LOG_ERROR`
            // and returns `dsInitialOptExpected`.
            "decoder_default" => {
                let mut d = Decoder::new();
                let rc = d.decode_parser(None, &mut SParserBsInfo::default());
                assert_eq!(rc, DECODING_STATE::dsInitialOptExpected);
            }
            // `WelsBitRateVerification` logs at `WELS_LOG_ERROR` when bitrate < fps.
            "encoder_default" => {
                let mut e = Encoder::new();
                let param = SEncParamBase {
                    iPicWidth: 16,
                    iPicHeight: 16,
                    fMaxFrameRate: 30.0,
                    iTargetBitrate: 10,
                    ..Default::default()
                };
                let rc = e.initialize(&param);
                assert_eq!(rc, CM_INIT_PARA_ERROR);
            }
            // A consumer that wants silence installs a callback that drops the line.
            "decoder_quiet" => {
                let mut d = Decoder::new();
                d.set_trace_callback(Some(quiet_sink));
                d.decode_parser(None, &mut SParserBsInfo::default());
            }
            "encoder_quiet" => {
                let mut e = Encoder::new();
                e.set_trace_callback(Some(quiet_sink));
                let param = SEncParamBase {
                    iPicWidth: 16,
                    iPicHeight: 16,
                    fMaxFrameRate: 30.0,
                    iTargetBitrate: 10,
                    ..Default::default()
                };
                e.initialize(&param);
            }
            // The decoder's default level is `WELS_LOG_ERROR`, so a WARNING-level
            // message must not appear. `BA_MW_D_IDR_LOST.264` drives
            // `UpdateAccessUnit()`'s "Key frame lost" warning and
            // `WelsInitRefList`'s "referencing pictures lost due frame gaps exist",
            // and the reference prints nothing for this stream at its default. Then an
            // uninitialized `decode_parser`, so the same run proves the sink is live
            // rather than merely silent.
            "decoder_level" => {
                let path = common::stream_path("BA_MW_D_IDR_LOST.264");
                let data = std::fs::read(path).expect("asset");
                let mut d = Decoder::new();
                let mut param = SDecodingParam::default();
                param.uiTargetDqLayer = u8::MAX;
                param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
                param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
                d.initialize(&param);
                for unit in openh264rs::split_annexb_units(&data) {
                    let mut info = SBufferInfo::default();
                    d.decode(Some(unit), &mut info);
                }
                d.uninitialize();

                let mut d2 = Decoder::new();
                d2.decode_parser(None, &mut SParserBsInfo::default());
            }
            other => panic!("unknown case {other}"),
        }
    }
}

/// A sink that writes nowhere.
///
/// # Safety
/// Matches `WelsTraceCallback`; reads nothing.
unsafe extern "C" fn quiet_sink(
    _ctx: *mut std::ffi::c_void,
    _level: i32,
    _s: *const std::ffi::c_char,
) {
}

#[test]
pub(crate) fn a_fresh_decoder_writes_its_error_to_stderr_with_no_callback_installed() {
    let err = stderr_of("decoder_default");
    assert!(
        err.contains("[OpenH264rs]")
            && err.contains("Error:")
            && err.contains("Call DecodeParser without Initialize"),
        "the default sink must be upstream's stderr writer; got:\n{err}"
    );
}

#[test]
pub(crate) fn a_fresh_encoder_writes_its_error_to_stderr_with_no_callback_installed() {
    let err = stderr_of("encoder_default");
    assert!(
        err.contains("[OpenH264rs]")
            && err.contains("Error:")
            && err.contains("Invalid bitrate settings"),
        "the default sink must be upstream's stderr writer; got:\n{err}"
    );
}

#[test]
pub(crate) fn an_installed_callback_replaces_the_default_sink_on_both_codecs() {
    for case in ["decoder_quiet", "encoder_quiet"] {
        let err = stderr_of(case);
        assert!(
            !err.contains("[OpenH264rs]"),
            "{case}: a caller who installs a sink must not also get the default one; got:\n{err}"
        );
    }
}

/// The decoder's default level is `WELS_LOG_ERROR`, not the trace object's
/// `WELS_LOG_WARNING` — `welsDecoderExt.cpp:164`.
///
/// The level cannot be read back — neither codec's `GetOption` handles
/// `*_TRACE_LEVEL`, only `SetOption` does (`welsDecoderExt.cpp:541` /
/// `welsEncoderExt.cpp:1090`) — so it is asserted through what does and does not
/// reach stderr.
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn the_decoder_defaults_to_error_level_so_its_warnings_are_silent() {
    let path = common::stream_path("BA_MW_D_IDR_LOST.264");
    assert!(path.exists(), "test input {} not found", path.display());
    let err = stderr_of("decoder_level");
    assert!(
        !err.contains("Warning:"),
        "the decoder's default is WELS_LOG_ERROR and the reference prints nothing for \
         this stream; got:\n{err}"
    );
    assert!(
        err.contains("Error:") && err.contains("Call DecodeParser without Initialize"),
        "...and the sink is live, not merely silent; got:\n{err}"
    );
}
