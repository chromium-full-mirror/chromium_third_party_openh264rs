// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Generates `sources.gni` for Chromium's GN build (`cargo run --bin gen_sources_gni`).

use std::path::Path;

fn collect_rs_files(dir: &Path, root: &Path, out: &mut Vec<String>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("failed to read directory {}: {e}", dir.display()));
    for entry in entries {
        let entry =
            entry.unwrap_or_else(|e| panic!("failed to read entry in {}: {e}", dir.display()));
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, root, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let rel = path
                .strip_prefix(root)
                .expect("file path should be under manifest root");
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
}

fn collect_path_attr_deps(dir: &Path, root: &Path, out: &mut Vec<String>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("failed to read directory {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("failed to read directory entry").path();
        if !path.is_file() || !path.extension().is_some_and(|ext| ext == "rs") {
            continue;
        }
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        for line in content.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("#[path") {
                if let Some((_, after_first_quote)) = rest.split_once('"') {
                    if let Some((rel_target, _)) = after_first_quote.split_once('"') {
                        let resolved = dir.join(rel_target);
                        if let Ok(canon) = resolved.canonicalize()
                            && let Ok(canon_root) = root.canonicalize()
                            && let Ok(rel) = canon.strip_prefix(&canon_root)
                        {
                            out.push(rel.to_string_lossy().replace('\\', "/"));
                        }
                    }
                }
            }
        }
    }
}

/// Integration tests that depend on Cargo's `cdylib` build output or host C/C++ toolchain
/// rather than the `rust_static_library` target used in Chromium's GN build.
const CARGO_CDYLIB_TESTS: &[&str] = &["tests/interop_test.rs"];

fn c_api_only_tests(manifest_dir: &Path) -> Vec<String> {
    let cargo_toml_path = manifest_dir.join("Cargo.toml");
    let cargo_toml = std::fs::read_to_string(&cargo_toml_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", cargo_toml_path.display()));
    let mut excluded: Vec<String> = CARGO_CDYLIB_TESTS
        .iter()
        .map(|&s| String::from(s))
        .collect();
    let mut in_test_section = false;
    let mut current_name: Option<String> = None;
    let mut requires_c_api = false;

    for line in cargo_toml.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            if in_test_section && requires_c_api {
                if let Some(name) = current_name.take() {
                    excluded.push(format!("tests/{name}.rs"));
                }
            }
            in_test_section = trimmed == "[[test]]";
            current_name = None;
            requires_c_api = false;
            continue;
        }
        if !in_test_section {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("name") {
            if let Some((_, after_quote)) = rest.split_once('"') {
                if let Some((name, _)) = after_quote.split_once('"') {
                    current_name = Some(String::from(name));
                }
            }
        } else if trimmed.starts_with("required-features") && trimmed.contains("\"c-api\"") {
            requires_c_api = true;
        }
    }
    if in_test_section && requires_c_api {
        if let Some(name) = current_name {
            excluded.push(format!("tests/{name}.rs"));
        }
    }
    excluded
}

fn generate_sources_gni(manifest_dir: &Path) -> String {
    let src_dir = manifest_dir.join("src");
    let mut sources = Vec::new();
    collect_rs_files(&src_dir, manifest_dir, &mut sources);
    sources.sort();

    let test_common_dir = manifest_dir.join("tests/common");
    let mut test_support_sources = Vec::new();
    collect_path_attr_deps(&test_common_dir, manifest_dir, &mut test_support_sources);
    collect_rs_files(&test_common_dir, manifest_dir, &mut test_support_sources);
    test_support_sources.sort();
    test_support_sources.dedup();

    let excluded_tests = c_api_only_tests(manifest_dir);
    let tests_dir = manifest_dir.join("tests");
    let mut integration_tests = Vec::new();
    let entries = std::fs::read_dir(&tests_dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", tests_dir.display()));
    for entry in entries {
        let path = entry.expect("failed to read tests/ entry").path();
        if path.is_file() && path.extension().is_some_and(|ext| ext == "rs") {
            let rel = path
                .strip_prefix(manifest_dir)
                .expect("test file should be under manifest root");
            let rel_str = rel.to_string_lossy().replace('\\', "/");
            if !excluded_tests.contains(&rel_str) {
                integration_tests.push(rel_str);
            }
        }
    }
    integration_tests.sort();

    let mut content = String::from(
        "# @generated by `cargo run --bin gen_sources_gni` — do not edit manually.\n\
         \n\
         openh264rs_crate_root = \"src/lib.rs\"\n\
         \n\
         openh264rs_cxx_bindings = [ \"src/api/cxx_api.rs\" ]\n\
         \n\
         openh264rs_sources = [\n",
    );
    for src in &sources {
        content.push_str(&format!("  \"{src}\",\n"));
    }
    content.push_str(
        "]\n\
         \n\
         openh264rs_test_support_sources = [\n",
    );
    for src in &test_support_sources {
        content.push_str(&format!("  \"{src}\",\n"));
    }
    content.push_str(
        "]\n\
         \n\
         openh264rs_integration_tests = [\n",
    );
    for test in &integration_tests {
        content.push_str(&format!("  \"{test}\",\n"));
    }
    content.push_str("]\n");
    content
}

fn main() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let content = generate_sources_gni(manifest_dir);
    let gni_path = manifest_dir.join("sources.gni");
    let existing = std::fs::read_to_string(&gni_path).unwrap_or_default();
    if existing != content {
        std::fs::write(&gni_path, content)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", gni_path.display()));
        println!("Updated {}", gni_path.display());
    } else {
        println!("{} is already up to date", gni_path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sources_gni_is_up_to_date() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let expected = generate_sources_gni(manifest_dir);
        let gni_path = manifest_dir.join("sources.gni");
        let actual = std::fs::read_to_string(&gni_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", gni_path.display()));
        assert_eq!(
            actual, expected,
            "sources.gni is out of date; run `cargo run --bin gen_sources_gni` to regenerate"
        );
    }
}
