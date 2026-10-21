// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Common test helper utilities for integration tests.
//!
//! Note on test visibility: `#[test]` functions across `tests/*.rs` are declared
//! `pub(crate) fn` so that Chromium's GoogleTest harness (`third_party/openh264rs/unittests.rs`)
//! can include each test file as a submodule (`#[path = "src/tests"] mod integration_tests`)
//! and invoke each test function from a `#[gtest]` wrapper.

// `#[path]`-included into several test crates, each of which uses only some helpers.
#![allow(dead_code)]

pub mod frame_gen;
pub mod sha1;

/// The library's deterministic PRNG, so a seed printed by a unit test replays here.
#[path = "../../src/safe/prng.rs"]
pub mod prng;
#[allow(unused_imports)]
pub use sha1::Sha1Hasher;

/// Resolves `CARGO_MANIFEST_DIR` (absolute under Cargo; relative to the
/// Chromium checkout root `//` under GN).
pub fn manifest_dir() -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    if dir.exists() {
        return dir.to_path_buf();
    }
    if let Ok(exe) = std::env::current_exe() {
        for anc in exe.ancestors().skip(1) {
            let candidate = anc.join(dir);
            if candidate.exists() {
                return candidate;
            }
        }
    }
    dir.to_path_buf()
}

/// Directory containing test bitstream assets (`OPENH264_TEST_DATA_DIR` or `tests/data`).
pub fn stream_dir() -> std::path::PathBuf {
    std::env::var_os("OPENH264_TEST_DATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| manifest_dir().join("tests/data"))
}

/// Resolves a test bitstream filename (with or without a leading `"res/"` prefix)
/// inside [`stream_dir`].
pub fn stream_path(name: &str) -> std::path::PathBuf {
    let clean = name.strip_prefix("res/").unwrap_or(name);
    stream_dir().join(clean)
}

/// Reads a test bitstream from [`stream_dir`].
///
/// # Panics
/// Panics if the file does not exist or cannot be read.
pub fn read_stream(name: &str) -> Vec<u8> {
    let path = stream_path(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}
