// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Bit-exact SHA-1 parity tests for the encoder (`EncoderOutputTest.CompareOutput` from
//! upstream `test/api/encoder_test.cpp` plus additional coverage configurations).
//!
//! Each entry in [`encoder_sha1_cases!`] expands into both its own `#[test]` function and
//! an entry in [`test_cases()`] for SHA-1 regeneration. Expected SHA-1 digests are stored in
//! `tests/encoder_output_sha1.txt`.
//!
//! To regenerate SHA-1 values in `tests/encoder_output_sha1.txt` from C++ OpenH264, run:
//! ```sh
//! ./tools/update_encoder_sha1.sh
//! ```

#[path = "common/mod.rs"]
mod common;

use common::Sha1Hasher;
use common::frame_gen::{fill_camera_frame, fill_screen_frame};
use openh264rs::api::codec_api::{
    ECOMPLEXITY_MODE::*, EProfileIdc::*, EUsageType::*, EVideoFormatType::*, EVideoFrameType::*,
    RC_MODES::*, SliceModeEnum::*, *,
};
use std::collections::HashMap;
#[cfg(feature = "c-api")]
use std::ffi::{CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[cfg(feature = "c-api")]
type CreateEncoderFn = unsafe extern "C" fn(pp_encoder: *mut *mut ISVCEncoder) -> i32;
#[cfg(feature = "c-api")]
type DestroyEncoderFn = unsafe extern "C" fn(p_encoder: *mut ISVCEncoder);

#[cfg(feature = "c-api")]
struct EncoderFactory {
    create: CreateEncoderFn,
    destroy: DestroyEncoderFn,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum InitStyle {
    /// Matches `InitWithParam` in upstream `test/api/BaseEncoderTest.cpp`.
    Upstream,
    /// Uses `GetDefaultParams` + `InitializeExt` with explicit RC/temporal/complexity/deblock fields.
    Ext,
}

struct EncodeCase {
    name: &'static str,
    init_style: InitStyle,
    use_screen_gen: bool,
    frames: usize,
    usage: EUsageType,
    w: i32,
    h: i32,
    fps: f32,
    bitrate: i32,
    rc_mode: RC_MODES,
    temporal_layers: i32,
    complexity: ECOMPLEXITY_MODE,
    loop_filter_idc: i32,
    frame_skip: bool,
    slice_mode: SliceModeEnum,
    denoise: bool,
    spatial_layers: i32,
    lossless: bool,
    ltr: bool,
    cabac: bool,
}

impl Default for EncodeCase {
    fn default() -> Self {
        Self {
            name: "",
            init_style: InitStyle::Upstream,
            use_screen_gen: false,
            frames: 9,
            usage: CAMERA_VIDEO_REAL_TIME,
            w: 320,
            h: 192,
            fps: 12.0,
            bitrate: 5_000_000,
            rc_mode: RC_QUALITY_MODE,
            temporal_layers: 1,
            complexity: LOW_COMPLEXITY,
            loop_filter_idc: 0,
            frame_skip: false,
            slice_mode: SM_SINGLE_SLICE,
            denoise: false,
            spatial_layers: 1,
            lossless: false,
            ltr: false,
            cabac: false,
        }
    }
}

impl EncodeCase {
    fn hd_720p() -> Self {
        Self {
            frames: 5,
            w: 1280,
            h: 720,
            fps: 30.0,
            ..Self::default()
        }
    }

    fn ext() -> Self {
        Self {
            init_style: InitStyle::Ext,
            bitrate: 500_000,
            ..Self::default()
        }
    }
}

macro_rules! encoder_sha1_cases {
    ($( $name:ident => $case:expr ),* $(,)?) => {
        #[cfg(feature = "c-api")]
        fn test_cases() -> Vec<EncodeCase> {
            vec![
                $(
                    {
                        let mut c = $case;
                        c.name = stringify!($name);
                        c
                    },
                )*
            ]
        }

        $(
            #[test]
            pub(crate) fn $name() {
                let mut c = $case;
                c.name = stringify!($name);
                run_single_case(&c);
            }
        )*
    };
}

encoder_sha1_cases! {
    // --- 16 upstream `EncoderOutputTest.CompareOutput` cases (`test/api/encoder_test.cpp`) ---
    camera_320x192_single => EncodeCase {
        ..Default::default()
    },
    camera_160x96_single => EncodeCase {
        frames: 5,
        w: 160,
        h: 96,
        fps: 6.0,
        ..Default::default()
    },
    static_152x100_single => EncodeCase {
        use_screen_gen: true,
        frames: 10,
        w: 152,
        h: 100,
        fps: 6.0,
        ..Default::default()
    },
    camera_320x192_raster => EncodeCase {
        slice_mode: SM_RASTER_SLICE,
        ..Default::default()
    },
    camera_320x192_denoise => EncodeCase {
        denoise: true,
        ..Default::default()
    },
    camera_320x192_2layers => EncodeCase {
        spatial_layers: 2,
        ..Default::default()
    },
    camera_1280x720_sizelimited => EncodeCase {
        slice_mode: SM_SIZELIMITED_SLICE,
        ..EncodeCase::hd_720p()
    },
    camera_1280x720_4layers => EncodeCase {
        spatial_layers: 4,
        ..EncodeCase::hd_720p()
    },
    screen_320x192_single => EncodeCase {
        usage: SCREEN_CONTENT_REAL_TIME,
        ..Default::default()
    },
    screen_160x96_single => EncodeCase {
        usage: SCREEN_CONTENT_REAL_TIME,
        frames: 5,
        w: 160,
        h: 96,
        fps: 6.0,
        ..Default::default()
    },
    screen_152x100_single => EncodeCase {
        usage: SCREEN_CONTENT_REAL_TIME,
        use_screen_gen: true,
        frames: 10,
        w: 152,
        h: 100,
        fps: 6.0,
        ..Default::default()
    },
    screen_1280x720_sizelimited => EncodeCase {
        usage: SCREEN_CONTENT_REAL_TIME,
        slice_mode: SM_SIZELIMITED_SLICE,
        ..EncodeCase::hd_720p()
    },
    screen_1280x720_lossless_ltr => EncodeCase {
        usage: SCREEN_CONTENT_REAL_TIME,
        use_screen_gen: true,
        slice_mode: SM_SIZELIMITED_SLICE,
        lossless: true,
        ltr: true,
        ..EncodeCase::hd_720p()
    },
    camera_320x192_cabac_single => EncodeCase {
        cabac: true,
        ..Default::default()
    },
    camera_1280x720_cabac_szlim => EncodeCase {
        slice_mode: SM_SIZELIMITED_SLICE,
        cabac: true,
        ..EncodeCase::hd_720p()
    },
    camera_1280x720_cabac_fixed => EncodeCase {
        slice_mode: SM_FIXEDSLCNUM_SLICE,
        cabac: true,
        ..EncodeCase::hd_720p()
    },

    // --- Additional feature parity cases ---
    screen_320x192_scrolling => EncodeCase {
        usage: SCREEN_CONTENT_REAL_TIME,
        use_screen_gen: true,
        ..EncodeCase::ext()
    },
    camera_320x192_temporal4 => EncodeCase {
        frames: 12,
        fps: 24.0,
        bitrate: 600_000,
        temporal_layers: 4,
        ..EncodeCase::ext()
    },
    camera_320x192_rc_bitrate_skip => EncodeCase {
        frames: 12,
        fps: 24.0,
        bitrate: 120_000,
        rc_mode: RC_BITRATE_MODE,
        frame_skip: true,
        ..EncodeCase::ext()
    },
    camera_320x192_rc_off_qp24 => EncodeCase {
        rc_mode: RC_OFF_MODE,
        ..EncodeCase::ext()
    },
    camera_320x192_deblock_off => EncodeCase {
        loop_filter_idc: 1,
        ..EncodeCase::ext()
    },
    camera_320x192_deblock_slice => EncodeCase {
        loop_filter_idc: 2,
        slice_mode: SM_FIXEDSLCNUM_SLICE,
        ..EncodeCase::ext()
    },
    camera_320x192_high_complexity => EncodeCase {
        complexity: HIGH_COMPLEXITY,
        ..EncodeCase::ext()
    },
    screen_320x192_high_comp_ltr => EncodeCase {
        usage: SCREEN_CONTENT_REAL_TIME,
        use_screen_gen: true,
        temporal_layers: 2,
        complexity: HIGH_COMPLEXITY,
        ltr: true,
        ..EncodeCase::ext()
    },
}

fn sha1_file_path() -> PathBuf {
    common::manifest_dir().join("tests/encoder_output_sha1.txt")
}

fn load_expected_sha1s(path: &Path) -> HashMap<String, String> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let mut map = HashMap::new();
    for (line_no, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, sha1) = line.split_once('=').unwrap_or_else(|| {
            panic!(
                "{}:{}: expected `name = sha1`, got `{line}`",
                path.display(),
                line_no + 1
            )
        });
        let name = name.trim().to_string();
        let sha1 = sha1.trim().to_string();
        assert_eq!(
            sha1.len(),
            40,
            "{}:{}: expected 40-char SHA-1 for `{name}`, got `{sha1}`",
            path.display(),
            line_no + 1
        );
        assert!(
            map.insert(name.clone(), sha1).is_none(),
            "duplicate entry `{name}` in {}",
            path.display()
        );
    }
    map
}

fn expected_sha1_map() -> &'static HashMap<String, String> {
    static MAP: OnceLock<HashMap<String, String>> = OnceLock::new();
    MAP.get_or_init(|| load_expected_sha1s(&sha1_file_path()))
}

fn build_ext_param_for_case(mut param: SEncParamExt, c: &EncodeCase) -> SEncParamExt {
    match c.init_style {
        InitStyle::Upstream => {
            param.iUsageType = c.usage;
            param.fMaxFrameRate = c.fps;
            param.iPicWidth = c.w;
            param.iPicHeight = c.h;
            param.iTargetBitrate = c.bitrate;
            param.bEnableDenoise = c.denoise;
            param.iSpatialLayerNum = c.spatial_layers;
            param.bIsLosslessLink = c.lossless;
            param.bEnableLongTermReference = c.ltr;
            param.iEntropyCodingModeFlag = if c.cabac { 1 } else { 0 };
            if c.slice_mode != SM_SINGLE_SLICE && c.slice_mode != SM_SIZELIMITED_SLICE {
                param.iMultipleThreadIdc = 2;
            }
            for i in 0..c.spatial_layers as usize {
                let shift = (c.spatial_layers - 1 - i as i32) as u32;
                param.sSpatialLayers[i].iVideoWidth = c.w >> shift;
                param.sSpatialLayers[i].iVideoHeight = c.h >> shift;
                param.sSpatialLayers[i].fFrameRate = c.fps;
                param.sSpatialLayers[i].iSpatialBitrate = param.iTargetBitrate;
                param.sSpatialLayers[i].sSliceArgument.uiSliceMode = c.slice_mode;
                if c.slice_mode == SM_SIZELIMITED_SLICE {
                    param.sSpatialLayers[i].sSliceArgument.uiSliceSizeConstraint = 600;
                    param.uiMaxNalSize = 1500;
                    param.iMultipleThreadIdc = 4;
                    param.bUseLoadBalancing = false;
                }
                if c.slice_mode == SM_FIXEDSLCNUM_SLICE {
                    param.sSpatialLayers[i].sSliceArgument.uiSliceNum = 4;
                    param.iMultipleThreadIdc = 4;
                    param.bUseLoadBalancing = false;
                }
                if param.iEntropyCodingModeFlag != 0 {
                    param.sSpatialLayers[i].uiProfileIdc = PRO_MAIN;
                }
            }
            param.iTargetBitrate *= c.spatial_layers;
            param
        }
        InitStyle::Ext => {
            param.iUsageType = c.usage;
            param.fMaxFrameRate = c.fps;
            param.iPicWidth = c.w;
            param.iPicHeight = c.h;
            param.iTargetBitrate = c.bitrate;
            param.iRCMode = c.rc_mode;
            param.iTemporalLayerNum = c.temporal_layers;
            param.iComplexityMode = c.complexity;
            param.iLoopFilterDisableIdc = c.loop_filter_idc;
            param.bEnableFrameSkip = c.frame_skip;
            param.bEnableDenoise = c.denoise;
            param.bIsLosslessLink = c.lossless;
            param.bEnableLongTermReference = c.ltr;
            param.iEntropyCodingModeFlag = if c.cabac { 1 } else { 0 };
            param.iSpatialLayerNum = c.spatial_layers;
            for i in 0..c.spatial_layers as usize {
                let shift = (c.spatial_layers - 1 - i as i32) as u32;
                param.sSpatialLayers[i].iVideoWidth = c.w >> shift;
                param.sSpatialLayers[i].iVideoHeight = c.h >> shift;
                param.sSpatialLayers[i].fFrameRate = c.fps;
                param.sSpatialLayers[i].iSpatialBitrate = c.bitrate;
                param.sSpatialLayers[i].iDLayerQp = 24;
                param.sSpatialLayers[i].sSliceArgument.uiSliceMode = c.slice_mode;
                if c.slice_mode == SM_FIXEDSLCNUM_SLICE {
                    param.sSpatialLayers[i].sSliceArgument.uiSliceNum = 4;
                    param.iMultipleThreadIdc = 4;
                    param.bUseLoadBalancing = false;
                }
                if param.iEntropyCodingModeFlag != 0 {
                    param.sSpatialLayers[i].uiProfileIdc = PRO_MAIN;
                }
            }
            param
        }
    }
}

fn is_base_init_case(c: &EncodeCase) -> bool {
    c.init_style == InitStyle::Upstream
        && c.slice_mode == SM_SINGLE_SLICE
        && !c.denoise
        && c.spatial_layers == 1
        && !c.lossless
        && !c.ltr
        && !c.cabac
}

fn init_rust_encoder_for_case(enc: &mut Encoder, c: &EncodeCase) -> i32 {
    if is_base_init_case(c) {
        let mut param = SEncParamBase::default();
        param.iUsageType = c.usage;
        param.fMaxFrameRate = c.fps;
        param.iPicWidth = c.w;
        param.iPicHeight = c.h;
        param.iTargetBitrate = c.bitrate;
        enc.initialize(&param)
    } else {
        let mut param = SEncParamExt::default();
        assert_eq!(enc.default_params(&mut param), CM_RESULT_SUCCESS);
        let param = build_ext_param_for_case(param, c);
        enc.initialize_ext(&param)
    }
}

fn compute_rust_case_sha1(c: &EncodeCase) -> String {
    let mut enc = Encoder::new();
    enc.set_trace_level(1); // WELS_LOG_ERROR

    assert_eq!(
        init_rust_encoder_for_case(&mut enc, c),
        CM_RESULT_SUCCESS,
        "Initialize failed for {}",
        c.name
    );

    let (w, h) = (c.w as usize, c.h as usize);
    let luma = w * h;
    let mut buf = vec![0u8; luma * 3 / 2];

    let mut hasher = Sha1Hasher::new();
    for f in 0..c.frames {
        if c.use_screen_gen {
            fill_screen_frame(&mut buf, w, h, f);
        } else {
            fill_camera_frame(&mut buf, w, h, f);
        }
        let mut pic = SourcePicture::from_i420(&buf, c.w, c.h);
        if c.init_style == InitStyle::Ext {
            pic.uiTimeStamp = (f as f32 * 1000.0 / c.fps) as i64;
        }

        let mut info = SFrameBSInfo::default();
        let rv = enc.encode_frame(&pic, &mut info);
        assert_eq!(rv, CM_RESULT_SUCCESS, "EncodeFrame failed for {}", c.name);
        if info.eFrameType != videoFrameTypeSkip {
            for layer in info.layers() {
                hasher.update(layer.bs_buf(&enc));
            }
        }
    }

    enc.uninitialize();
    hasher.digest()
}

fn run_single_case(c: &EncodeCase) {
    let path = sha1_file_path();
    let expected_sha1 = expected_sha1_map().get(c.name).unwrap_or_else(|| {
        panic!(
            "missing expected SHA-1 for `{}` in {}; run ./tools/update_encoder_sha1.sh",
            c.name,
            path.display()
        )
    });
    let actual_sha1 = compute_rust_case_sha1(c);
    assert_eq!(
        &actual_sha1,
        expected_sha1,
        "SHA-1 mismatch for case `{}` in {}",
        c.name,
        path.display()
    );
}

#[cfg(feature = "c-api")]
unsafe fn init_encoder_for_case(enc: *mut ISVCEncoder, c: &EncodeCase) -> i32 {
    if is_base_init_case(c) {
        let mut param = SEncParamBase::default();
        param.iUsageType = c.usage;
        param.fMaxFrameRate = c.fps;
        param.iPicWidth = c.w;
        param.iPicHeight = c.h;
        param.iTargetBitrate = c.bitrate;
        unsafe { ISVCEncoder::Initialize(enc, &param) }
    } else {
        let mut param = SEncParamExt::default();
        assert_eq!(
            unsafe { ISVCEncoder::GetDefaultParams(enc, &mut param) },
            CM_RESULT_SUCCESS
        );
        let param = build_ext_param_for_case(param, c);
        unsafe { ISVCEncoder::InitializeExt(enc, &param) }
    }
}

#[cfg(feature = "c-api")]
fn compute_case_sha1(factory: &EncoderFactory, c: &EncodeCase) -> String {
    unsafe {
        let mut enc: *mut ISVCEncoder = std::ptr::null_mut();
        assert_eq!((factory.create)(&mut enc), CM_RESULT_SUCCESS);
        assert!(!enc.is_null());

        let mut trace_level: u32 = 1; // WELS_LOG_ERROR
        ISVCEncoder::SetOption(
            enc,
            ENCODER_OPTION::ENCODER_OPTION_TRACE_LEVEL,
            (&raw mut trace_level).cast::<c_void>(),
        );

        assert_eq!(
            init_encoder_for_case(enc, c),
            CM_RESULT_SUCCESS,
            "Initialize failed for {}",
            c.name
        );

        let (w, h) = (c.w as usize, c.h as usize);
        let luma = w * h;
        let mut buf = vec![0u8; luma * 3 / 2];

        let mut pic = SSourcePicture::default();
        pic.iPicWidth = c.w;
        pic.iPicHeight = c.h;
        pic.iColorFormat = videoFormatI420 as i32;
        pic.iStride[0] = c.w;
        pic.iStride[1] = c.w >> 1;
        pic.iStride[2] = c.w >> 1;
        pic.pData[0] = buf.as_mut_ptr();
        pic.pData[1] = buf.as_mut_ptr().add(luma);
        pic.pData[2] = buf.as_mut_ptr().add(luma + (luma >> 2));

        let mut hasher = Sha1Hasher::new();
        for f in 0..c.frames {
            if c.use_screen_gen {
                fill_screen_frame(&mut buf, w, h, f);
            } else {
                fill_camera_frame(&mut buf, w, h, f);
            }
            if c.init_style == InitStyle::Ext {
                pic.uiTimeStamp = (f as f32 * 1000.0 / c.fps) as i64;
            }

            let mut info = SFrameBSInfo::default();
            let rv = ISVCEncoder::EncodeFrame(enc, &pic, &mut info);
            assert_eq!(rv, CM_RESULT_SUCCESS, "EncodeFrame failed for {}", c.name);
            if info.eFrameType != videoFrameTypeSkip {
                for layer in info.layers() {
                    if layer.iNalCount > 0
                        && !layer.nal_lengths_ptr().is_null()
                        && !layer.bs_buf_ptr().is_null()
                    {
                        let nal_lens = std::slice::from_raw_parts(
                            layer.nal_lengths_ptr(),
                            layer.iNalCount as usize,
                        );
                        let len: usize = nal_lens.iter().map(|&n| n.max(0) as usize).sum();
                        let bs = std::slice::from_raw_parts(layer.bs_buf_ptr(), len);
                        hasher.update(bs);
                    }
                }
            }
        }

        ISVCEncoder::Uninitialize(enc);
        (factory.destroy)(enc);
        hasher.digest()
    }
}

#[cfg(feature = "c-api")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

#[cfg(feature = "c-api")]
const RTLD_NOW: c_int = 2;
#[cfg(feature = "c-api")]
const RTLD_DEEPBIND: c_int = 8;

#[cfg(feature = "c-api")]
fn locate_cpp_openh264_so() -> PathBuf {
    if let Some(p) = std::env::var_os("OPENH264_CPP_LIB") {
        return PathBuf::from(p);
    }
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.join("../openh264/libopenh264.so")
}

#[cfg(feature = "c-api")]
fn update_sha1_file(path: &Path, new_hashes: &HashMap<String, String>) {
    let original = std::fs::read_to_string(path).expect("read SHA-1 file");
    let mut updated = String::with_capacity(original.len());
    let mut seen = std::collections::HashSet::new();

    for raw_line in original.split_inclusive('\n') {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            updated.push_str(raw_line);
            continue;
        }
        let eq_pos = raw_line.find('=').expect("'=' in SHA-1 file");
        let name = raw_line[..eq_pos].trim();
        if let Some(new_sha1) = new_hashes.get(name) {
            seen.insert(name.to_string());
            let after_eq = &raw_line[eq_pos + 1..];
            let old_sha1_trimmed = after_eq.trim();
            let sha1_offset = eq_pos + 1 + after_eq.find(old_sha1_trimmed).unwrap();
            updated.push_str(&raw_line[..sha1_offset]);
            updated.push_str(new_sha1);
            updated.push_str(&raw_line[sha1_offset + old_sha1_trimmed.len()..]);
        } else {
            updated.push_str(raw_line);
        }
    }

    // Append any newly added cases not yet present in the file.
    for c in test_cases() {
        if !seen.contains(c.name) {
            if let Some(new_sha1) = new_hashes.get(c.name) {
                if !updated.ends_with('\n') {
                    updated.push('\n');
                }
                updated.push_str(&format!("{:<31} = {new_sha1}\n", c.name));
            }
        }
    }

    std::fs::write(path, updated).expect("write updated SHA-1 file");
}

/// Invoked by `./tools/update_encoder_sha1.sh`: loads C++ `libopenh264.so` via `dlopen`
/// (`RTLD_NOW | RTLD_DEEPBIND`), computes SHA-1 digests for all cases in [`test_cases`]
/// using ONLY the C++ encoder, and writes them to `tests/encoder_output_sha1.txt`.
#[cfg(feature = "c-api")]
#[test]
#[ignore = "invoked by ./tools/update_encoder_sha1.sh to update tests/encoder_output_sha1.txt"]
fn regenerate_sha1_from_cpp_openh264() {
    let so_path = locate_cpp_openh264_so();
    assert!(
        so_path.exists(),
        "C++ shared library not found at {}. Run `./tools/update_encoder_sha1.sh` \
         or set OPENH264_CPP_LIB=/path/to/libopenh264.so",
        so_path.display()
    );

    let c_path = CString::new(so_path.to_str().unwrap()).unwrap();
    let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_NOW | RTLD_DEEPBIND) };
    assert!(!handle.is_null(), "dlopen failed for {}", so_path.display());

    let create_sym = CString::new("WelsCreateSVCEncoder").unwrap();
    let destroy_sym = CString::new("WelsDestroySVCEncoder").unwrap();
    let create_ptr = unsafe { dlsym(handle, create_sym.as_ptr()) };
    let destroy_ptr = unsafe { dlsym(handle, destroy_sym.as_ptr()) };
    assert!(!create_ptr.is_null(), "dlsym(WelsCreateSVCEncoder) failed");
    assert!(
        !destroy_ptr.is_null(),
        "dlsym(WelsDestroySVCEncoder) failed"
    );

    assert_ne!(
        create_ptr as usize, WelsCreateSVCEncoder as *const () as usize,
        "dlsym returned Rust WelsCreateSVCEncoder instead of C++ libopenh264.so"
    );

    let cpp_factory = unsafe {
        EncoderFactory {
            create: std::mem::transmute::<*mut c_void, CreateEncoderFn>(create_ptr),
            destroy: std::mem::transmute::<*mut c_void, DestroyEncoderFn>(destroy_ptr),
        }
    };

    let path = sha1_file_path();
    let cases = test_cases();
    let mut new_hashes = HashMap::new();

    for c in &cases {
        let cpp_sha1 = compute_case_sha1(&cpp_factory, c);
        println!("{:<32} C++ SHA-1 = {}", c.name, cpp_sha1);
        new_hashes.insert(c.name.to_string(), cpp_sha1);
    }

    unsafe {
        dlclose(handle);
    }

    update_sha1_file(&path, &new_hashes);
    println!("Updated SHA-1 values in {}", path.display());
}
