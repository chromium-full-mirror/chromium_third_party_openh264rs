// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Common test helper utilities for integration tests.

// `#[path]`-included into several test crates, each of which uses only some helpers.
#![allow(dead_code)]

pub mod frame_gen;
pub mod sha1;

/// The library's deterministic PRNG, so a seed printed by a unit test replays here.
#[path = "../../src/safe/prng.rs"]
pub mod prng;
#[allow(unused_imports)]
pub use sha1::Sha1Hasher;

/// Directory containing test bitstream assets (`OPENH264_TEST_DATA_DIR` or `tests/data`).
pub fn stream_dir() -> std::path::PathBuf {
    std::env::var_os("OPENH264_TEST_DATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data"))
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
