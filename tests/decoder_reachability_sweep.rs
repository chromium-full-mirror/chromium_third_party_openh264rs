// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! The reset-arm reachability sweep: every `res/*.264` stream under every
//! concealment mode under both bitstream declarations, in one pass, pinning which
//! error classes are reachable at all.
// `c_long` is i64 here and i32 on Windows; the widening is not dead there.
#![allow(clippy::useless_conversion)]

#[path = "common/mod.rs"]
mod common;

use openh264rs::api::codec_api::*;
use openh264rs::split_annexb_units;

/// Base and MV-copy concealment modes exercised across the corpus.
const EC_MODES: [ERROR_CON_IDC; 6] = [
    ERROR_CON_IDC::ERROR_CON_DISABLE,
    ERROR_CON_IDC::ERROR_CON_FRAME_COPY,
    ERROR_CON_IDC::ERROR_CON_SLICE_COPY,
    ERROR_CON_IDC::ERROR_CON_SLICE_COPY_CROSS_IDR,
    ERROR_CON_IDC::ERROR_CON_SLICE_MV_COPY_CROSS_IDR,
    ERROR_CON_IDC::ERROR_CON_SLICE_MV_COPY_CROSS_IDR_FREEZE_RES_CHANGE,
];

/// Both declarations a caller can make about the bitstream.
const BS_TYPES: [VIDEO_BITSTREAM_TYPE; 2] = [
    VIDEO_BITSTREAM_TYPE::VIDEO_BITSTREAM_AVC,
    VIDEO_BITSTREAM_TYPE::VIDEO_BITSTREAM_SVC,
];

/// The union of every `DECODING_STATE` bit the sweep produces.
///
/// `0x02 dsRefLost | 0x04 dsBitstreamError | 0x10 dsNoParamSets | 0x20
/// dsDataErrorConcealed`.
///
/// Pinned in both directions: a bit appearing means a decode path became reachable,
/// a bit disappearing means one stopped being reachable.
const EXPECTED_UNION: i32 = 0x36;

/// The two arms reachable from no stream in `res/`.
const UNREACHED: [(i32, &str); 2] = [(0x4000, "dsOutOfMemory"), (0x0040, "dsRefListNullPtrs")];

fn assets() -> Vec<std::path::PathBuf> {
    let dir = common::stream_dir();
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut v: Vec<_> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "264"))
        .collect();
    v.sort();
    v
}

/// One whole-stream decode; returns the OR of every state the decoder reported.
///
fn sweep_one(data: &[u8], ec: ERROR_CON_IDC, bs: VIDEO_BITSTREAM_TYPE) -> i32 {
    let mut decoder = Decoder::new();
    let mut param = SDecodingParam::default();
    param.uiTargetDqLayer = u8::MAX;
    param.eEcActiveIdc = ec;
    param.sVideoProperty.eVideoBsType = bs;
    assert_eq!(
        decoder.initialize(&param),
        CM_RESULT_SUCCESS as std::ffi::c_long
    );

    let mut union_bits = 0i32;
    let mut feed = |decoder: &mut Decoder, unit: Option<&[u8]>| {
        let mut info = SBufferInfo::default();
        let st = decoder.decode(unit, &mut info);
        union_bits |= st.0;
    };
    for unit in split_annexb_units(data) {
        feed(&mut decoder, Some(unit));
    }
    decoder.set_end_of_stream(true);
    feed(&mut decoder, None);
    decoder.uninitialize();
    union_bits
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn every_res_stream_under_every_concealment_mode_reaches_a_known_set_of_states() {
    let files = assets();
    assert!(
        files.len() >= 50,
        "skipping reachability sweep: found {} streams in tests/data (need >= 50)",
        files.len()
    );

    // Forked across the asset list: the decodes are independent — one decoder object
    // per (stream, mode, declaration), created and destroyed inside the thread that
    // uses it, and no `*mut ISVCDecoder` ever crosses a thread. Nothing here relies on
    // `Decoder` being `Send`, which it is not.
    let nthreads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(files.len().max(1));
    let (union_bits, offenders, decodes) = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for t in 0..nthreads {
            let slice: Vec<&std::path::PathBuf> = files.iter().skip(t).step_by(nthreads).collect();
            handles.push(scope.spawn(move || {
                let mut union_bits = 0i32;
                let mut offenders: Vec<String> = Vec::new();
                let mut decodes = 0usize;
                for path in slice {
                    let data = std::fs::read(path).unwrap();
                    let name = path.file_name().unwrap().to_string_lossy().into_owned();
                    for &ec in &EC_MODES {
                        for &bs in &BS_TYPES {
                            // The decode itself is the abort test: an `extern "C"`
                            // thunk that panics kills this process and no assertion
                            // below ever runs.
                            let bits = sweep_one(&data, ec, bs);
                            decodes += 1;
                            union_bits |= bits;
                            for (bit, label) in UNREACHED {
                                if bits & bit != 0 {
                                    offenders
                                        .push(format!("{name} ec={ec:?} bs={bs:?} -> {label}"));
                                }
                            }
                        }
                    }
                }
                (union_bits, offenders, decodes)
            }));
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("sweep worker"))
            .fold((0i32, Vec::new(), 0usize), |(u, mut o, d), (u2, o2, d2)| {
                o.extend(o2);
                (u | u2, o, d + d2)
            })
    });

    eprintln!(
        "reachability sweep: {} streams x {} modes x {} declarations = {decodes} decodes, state union 0x{union_bits:x}",
        files.len(),
        EC_MODES.len(),
        BS_TYPES.len()
    );

    assert!(
        offenders.is_empty(),
        "an arm documented as unreachable was reached — that is a fact worth a finding, \
         not a number to change:\n  {}",
        offenders.join("\n  ")
    );
    assert_eq!(
        union_bits, EXPECTED_UNION,
        "the reachable state set moved (0x{union_bits:x} vs 0x{EXPECTED_UNION:x}); \
         a decode path became reachable or stopped being reachable"
    );
}
