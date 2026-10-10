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
    println!("cargo:rustc-check-cfg=cfg(has_test_data)");

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
}
