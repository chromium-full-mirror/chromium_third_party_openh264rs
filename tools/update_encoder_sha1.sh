#!/usr/bin/env bash
# Copyright 2026 The Chromium Authors
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE_CHROMIUM file.
#
# Rebuilds C++ libopenh264.so, regenerates expected SHA-1 digests in
# tests/encoder_output_sha1.txt using ONLY the C++ encoder (fed by the Rust frame
# generator in tests/common/frame_gen.rs), and then runs the Rust parity test.
#
# Usage:
#   ./tools/update_encoder_sha1.sh [/path/to/openh264]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENH264_DIR="${1:-${OPENH264_DIR:-${REPO_ROOT}/../openh264}}"

if [[ ! -d "${OPENH264_DIR}" ]]; then
  echo "error: C++ OpenH264 directory not found at ${OPENH264_DIR}" >&2
  echo "usage: $0 [/path/to/openh264]" >&2
  exit 1
fi

if [[ -w "${OPENH264_DIR}" ]]; then
  echo "==> Building C++ libopenh264.so in ${OPENH264_DIR}..."
  make -C "${OPENH264_DIR}" -j"$(nproc)" libraries
elif [[ ! -f "${OPENH264_DIR}/libopenh264.so" ]]; then
  echo "error: ${OPENH264_DIR} is read-only and libopenh264.so does not exist" >&2
  exit 1
else
  echo "==> ${OPENH264_DIR} is read-only; using existing ${OPENH264_DIR}/libopenh264.so"
fi

export OPENH264_CPP_LIB="${OPENH264_DIR}/libopenh264.so"
cd "${REPO_ROOT}"

echo "==> Step 1: Regenerating tests/encoder_output_sha1.txt from C++ libopenh264.so..."
cargo test --test encoder_output_sha1_test regenerate_sha1_from_cpp_openh264 -- --ignored --nocapture

echo "==> Step 2: Verifying Rust encoder against updated tests/encoder_output_sha1.txt..."
cargo test --test encoder_output_sha1_test
