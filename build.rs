// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

fn main() {
    #[cfg(feature = "cxx-api")]
    {
        cxx_build::bridge("src/api/cxx_api.rs")
            .std("c++14")
            .compile("openh264rs-cxx");
        println!("cargo:rerun-if-changed=src/api/cxx_api.rs");
    }
    println!("cargo:rerun-if-changed=tests/data");
    println!("cargo:rerun-if-env-changed=OPENH264_TEST_DATA_DIR");
    println!("cargo:rerun-if-env-changed=FFMPEG");
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rustc-check-cfg=cfg(has_test_data)");
    println!("cargo:rustc-check-cfg=cfg(has_ffmpeg_libx264)");
    println!("cargo:rustc-check-cfg=cfg(is_gtest_unittests)");

    let data_dir = std::env::var_os("OPENH264_TEST_DATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("tests/data")
        });
    let has_data = std::fs::read_dir(&data_dir)
        .map(|mut rd| rd.any(|e| e.is_ok()))
        .unwrap_or(false);
    if has_data {
        println!("cargo:rustc-cfg=has_test_data");
    }

    if probe_ffmpeg_libx264() {
        println!("cargo:rustc-cfg=has_ffmpeg_libx264");
    }
}

fn probe_ffmpeg_bin(bin: &std::ffi::OsStr) -> bool {
    std::process::Command::new(bin)
        .args(["-hide_banner", "-v", "error", "-h", "encoder=libx264"])
        .output()
        .is_ok_and(|out| out.status.success() && out.stdout.starts_with(b"Encoder libx264"))
}

fn probe_ffmpeg_libx264() -> bool {
    if let Some(from_env) = std::env::var_os("FFMPEG") {
        return !from_env.is_empty() && probe_ffmpeg_bin(&from_env);
    }
    for candidate in [
        "ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ] {
        if probe_ffmpeg_bin(std::ffi::OsStr::new(candidate)) {
            return true;
        }
    }
    false
}
