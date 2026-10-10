// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Standalone C (`c-api`) and C++ (`cxx-api`) interop tests.
//!
//! Compiles real C (`tests/interop/c_api_test.c`) and C++ (`tests/interop/cxx_api_test.cpp`)
//! programs with the system C/C++ compiler, links them against the built `libopenh264rs`
//! shared library (`cdylib`), and runs the resulting executables.

#![cfg(not(target_os = "windows"))]

extern crate openh264rs;

#[cfg(any(feature = "c-api", feature = "cxx-api"))]
use std::path::{Path, PathBuf};
#[cfg(any(feature = "c-api", feature = "cxx-api"))]
use std::process::Command;

#[cfg(any(feature = "c-api", feature = "cxx-api"))]
fn find_cdylib() -> PathBuf {
    let dylib_name = if cfg!(target_os = "macos") {
        "libopenh264rs.dylib"
    } else {
        "libopenh264rs.so"
    };

    let exe = std::env::current_exe().expect("failed to locate current test executable");
    let deps_dir = exe.parent().expect("test executable has no parent dir");
    let deps_cand = deps_dir.join(dylib_name);
    let profile_cand = deps_dir.parent().map(|p| p.join(dylib_name));

    match (deps_cand.exists(), profile_cand) {
        (true, Some(prof)) if prof.exists() => {
            let deps_mtime = std::fs::metadata(&deps_cand)
                .and_then(|m| m.modified())
                .ok();
            let prof_mtime = std::fs::metadata(&prof).and_then(|m| m.modified()).ok();
            if prof_mtime > deps_mtime {
                prof
            } else {
                deps_cand
            }
        }
        (true, _) => deps_cand,
        (false, Some(prof)) if prof.exists() => prof,
        _ => panic!(
            "could not find {dylib_name} in {} or its parent directory",
            deps_dir.display()
        ),
    }
}

#[cfg(any(feature = "c-api", feature = "cxx-api"))]
fn run_checked(cmd: &mut Command, step: &str) {
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {step}: {e}"));
    if !output.status.success() {
        panic!(
            "{step} failed with status {}:\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}

#[test]
#[cfg(feature = "c-api")]
fn test_c_api_interop() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out_dir = Path::new(env!("OUT_DIR")).join("interop_bin");
    std::fs::create_dir_all(&out_dir).expect("failed to create interop_bin dir");

    let c_src = manifest_dir.join("tests/interop/c_api_test.c");
    let c_include = manifest_dir.join("tests/interop/include");
    let dylib = find_cdylib();
    let dylib_dir = dylib.parent().unwrap();
    let bin_path = out_dir.join("c_api_test");

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut compile = Command::new(&cc);
    compile
        .arg("-std=c11")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg("-I")
        .arg(&c_include)
        .arg(&c_src)
        .arg(&dylib)
        .arg("-lm")
        .arg(format!("-Wl,-rpath,{}", dylib_dir.display()))
        .arg("-o")
        .arg(&bin_path);
    run_checked(&mut compile, "C API test compilation");

    let mut run = Command::new(&bin_path);
    run_checked(&mut run, "c_api_test execution");
}

#[test]
#[cfg(feature = "cxx-api")]
fn test_cxx_api_interop() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out_dir = Path::new(env!("OUT_DIR"));
    let bin_dir = out_dir.join("interop_bin");
    std::fs::create_dir_all(&bin_dir).expect("failed to create interop_bin dir");

    let cpp_src = manifest_dir.join("tests/interop/cxx_api_test.cpp");
    // Only pass the cxx-generated include directory — never the Cisco C headers.
    let cxx_include = out_dir.join("cxxbridge/include");
    let cxx_bridge_cc = out_dir.join("cxxbridge/sources/openh264rs/src/api/cxx_api.rs.cc");
    let dylib = find_cdylib();
    let dylib_dir = dylib.parent().unwrap();
    let bin_path = bin_dir.join("cxx_api_test");

    let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    let mut compile = Command::new(&cxx);
    compile
        .arg("-std=c++14")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg("-I")
        .arg(&cxx_include)
        .arg(&cxx_bridge_cc)
        .arg(&cpp_src)
        .arg(&dylib)
        .arg(format!("-Wl,-rpath,{}", dylib_dir.display()))
        .arg("-o")
        .arg(&bin_path);
    run_checked(&mut compile, "C++ cxx_api test compilation");

    let mut run = Command::new(&bin_path);
    run_checked(&mut run, "cxx_api_test execution");
}
