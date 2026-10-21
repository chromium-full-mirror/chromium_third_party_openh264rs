// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

// Copyright (c) 2013, Cisco Systems
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice, this
//   list of conditions and the following disclaimer in the documentation and/or
//   other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//! Shared codec types, constants, return codes, options, and enums (`codec_def.h`, `codec_app_def.h`).
//!
//! These types are used across the Rust core, the safe Rust API (`Encoder` / `Decoder`),
//! the `cxx` C++ bridge (`cxx_api`), and the raw C ABI (`c_api`).

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]
#![deny(unsafe_code)]

use std::ffi::{c_char, c_void};
use std::ptr;

pub const MAX_TEMPORAL_LAYER_NUM: usize = 4;
pub const MAX_SPATIAL_LAYER_NUM: usize = 4;
pub const MAX_QUALITY_LAYER_NUM: usize = 4;

pub const MAX_LAYER_NUM_OF_FRAME: usize = 128;
pub const MAX_NAL_UNITS_IN_LAYER: usize = 128;

pub const MAX_RTP_PAYLOAD_LEN: usize = 1000;
pub const AVERAGE_RTP_PAYLOAD_LEN: usize = 800;

pub const SAVED_NALUNIT_NUM_TMP: usize =
    (MAX_SPATIAL_LAYER_NUM * MAX_QUALITY_LAYER_NUM) + 1 + MAX_SPATIAL_LAYER_NUM;
pub const MAX_SLICES_NUM_TMP: usize = (MAX_NAL_UNITS_IN_LAYER - SAVED_NALUNIT_NUM_TMP) / 3;

pub const AUTO_REF_PIC_COUNT: i32 = -1;
pub const UNSPECIFIED_BIT_RATE: i32 = 0;

pub const FRAME_NUM_PARAM_SET: i32 = -1;
pub const FRAME_NUM_IDR: i32 = 0;

// Error & Return Codes
pub const CM_RESULT_SUCCESS: i32 = 0;
pub const CM_INIT_PARA_ERROR: i32 = 1;
pub const CM_UNKNOWN_REASON: i32 = 2;
pub const CM_MALLOC_MEM_ERROR: i32 = 3;
pub const CM_INIT_EXPECTED: i32 = 4;
pub const CM_UNSUPPORTED_DATA: i32 = 5;

/// Return codes enumeration (`CM_RETURN`).
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum CM_RETURN {
    #[default]
    cmResultSuccess = 0,
    cmInitParaError = 1,
    cmUnknownReason = 2,
    cmMallocMemeError = 3,
    cmInitExpected = 4,
    cmUnsupportedData = 5,
}

/// Enumerate video format types (`EVideoFormatType`).
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum EVideoFormatType {
    videoFormatRGB = 1,
    videoFormatRGBA = 2,
    videoFormatRGB555 = 3,
    videoFormatRGB565 = 4,
    videoFormatBGR = 5,
    videoFormatBGRA = 6,
    videoFormatABGR = 7,
    videoFormatARGB = 8,
    videoFormatYUY2 = 20,
    videoFormatYVYU = 21,
    videoFormatUYVY = 22,
    #[default]
    videoFormatI420 = 23,
    videoFormatYV12 = 24,
    videoFormatInternal = 25,
    videoFormatNV12 = 26,
    videoFormatVFlip = -0x80000000i32,
}

pub type VideoFormat = EVideoFormatType;

/// Enumerate video frame types (`EVideoFrameType`).
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum EVideoFrameType {
    #[default]
    videoFrameTypeInvalid = 0,
    videoFrameTypeIDR = 1,
    videoFrameTypeI = 2,
    videoFrameTypeP = 3,
    videoFrameTypeSkip = 4,
    videoFrameTypeIPMixed = 5,
}

impl EVideoFrameType {
    #[inline]
    pub const fn from_raw(v: i32) -> Self {
        match v {
            1 => Self::videoFrameTypeIDR,
            2 => Self::videoFrameTypeI,
            3 => Self::videoFrameTypeP,
            4 => Self::videoFrameTypeSkip,
            5 => Self::videoFrameTypeIPMixed,
            _ => Self::videoFrameTypeInvalid,
        }
    }
}

/// NAL unit types (`ENalUnitType`).
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum ENalUnitType {
    #[default]
    NAL_UNKNOWN = 0,
    NAL_SLICE = 1,
    NAL_SLICE_DPA = 2,
    NAL_SLICE_DPB = 3,
    NAL_SLICE_DPC = 4,
    NAL_SLICE_IDR = 5,
    NAL_SEI = 6,
    NAL_SPS = 7,
    NAL_PPS = 8,
}

/// NAL reference priority (`ENalPriority`).
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum ENalPriority {
    #[default]
    NAL_PRIORITY_DISPOSABLE = 0,
    NAL_PRIORITY_LOW = 1,
    NAL_PRIORITY_HIGH = 2,
    NAL_PRIORITY_HIGHEST = 3,
}

/// Decoding status bitmask (`DECODING_STATE`).
///
/// A bitmask, not an enumeration: the decoder accumulates into `iErrorCode` with
/// `|=` and returns the accumulator whole, so combined values such as
/// `dsBitstreamError | dsDataErrorConcealed` (`0x24`) name no variant.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Default)]
pub struct DECODING_STATE(pub i32);

#[allow(non_upper_case_globals)]
impl DECODING_STATE {
    pub const dsErrorFree: Self = Self(0x00);
    pub const dsFramePending: Self = Self(0x01);
    pub const dsRefLost: Self = Self(0x02);
    pub const dsBitstreamError: Self = Self(0x04);
    pub const dsDepLayerLost: Self = Self(0x08);
    pub const dsNoParamSets: Self = Self(0x10);
    pub const dsDataErrorConcealed: Self = Self(0x20);
    pub const dsRefListNullPtrs: Self = Self(0x40);

    pub const dsInvalidArgument: Self = Self(0x1000);
    pub const dsInitialOptExpected: Self = Self(0x2000);
    pub const dsOutOfMemory: Self = Self(0x4000);
    pub const dsDstBufNeedExpan: Self = Self(0x8000);

    /// The set bits, named, in header order, so `{:?}` of a combined value reads as
    /// names rather than a number.
    const NAMES: [(i32, &'static str); 12] = [
        (0x01, "dsFramePending"),
        (0x02, "dsRefLost"),
        (0x04, "dsBitstreamError"),
        (0x08, "dsDepLayerLost"),
        (0x10, "dsNoParamSets"),
        (0x20, "dsDataErrorConcealed"),
        (0x40, "dsRefListNullPtrs"),
        (0x1000, "dsInvalidArgument"),
        (0x2000, "dsInitialOptExpected"),
        (0x4000, "dsOutOfMemory"),
        (0x8000, "dsDstBufNeedExpan"),
        (0, ""),
    ];
}

impl core::fmt::Debug for DECODING_STATE {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.0 == 0 {
            return f.write_str("dsErrorFree");
        }
        let mut first = true;
        let mut rest = self.0;
        for (bit, name) in Self::NAMES {
            if bit != 0 && self.0 & bit != 0 {
                if !first {
                    f.write_str("|")?;
                }
                f.write_str(name)?;
                first = false;
                rest &= !bit;
            }
        }
        if rest != 0 {
            if !first {
                f.write_str("|")?;
            }
            write!(f, "{rest:#x}")?;
        }
        Ok(())
    }
}

/// Encoder option identifiers (`ENCODER_OPTION`).
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum ENCODER_OPTION {
    #[default]
    ENCODER_OPTION_DATAFORMAT = 0,
    ENCODER_OPTION_IDR_INTERVAL = 1,
    ENCODER_OPTION_SVC_ENCODE_PARAM_BASE = 2,
    ENCODER_OPTION_SVC_ENCODE_PARAM_EXT = 3,
    ENCODER_OPTION_FRAME_RATE = 4,
    ENCODER_OPTION_BITRATE = 5,
    ENCODER_OPTION_MAX_BITRATE = 6,
    ENCODER_OPTION_INTER_SPATIAL_PRED = 7,
    ENCODER_OPTION_RC_MODE = 8,
    ENCODER_OPTION_RC_FRAME_SKIP = 9,
    ENCODER_PADDING_PADDING = 10,
    ENCODER_OPTION_PROFILE = 11,
    ENCODER_OPTION_LEVEL = 12,
    ENCODER_OPTION_NUMBER_REF = 13,
    ENCODER_OPTION_DELIVERY_STATUS = 14,
    ENCODER_LTR_RECOVERY_REQUEST = 15,
    ENCODER_LTR_MARKING_FEEDBACK = 16,
    ENCODER_LTR_MARKING_PERIOD = 17,
    ENCODER_OPTION_LTR = 18,
    ENCODER_OPTION_COMPLEXITY = 19,
    ENCODER_OPTION_ENABLE_SSEI = 20,
    ENCODER_OPTION_ENABLE_PREFIX_NAL_ADDING = 21,
    ENCODER_OPTION_SPS_PPS_ID_STRATEGY = 22,
    ENCODER_OPTION_CURRENT_PATH = 23,
    ENCODER_OPTION_DUMP_FILE = 24,
    ENCODER_OPTION_TRACE_LEVEL = 25,
    ENCODER_OPTION_TRACE_CALLBACK = 26,
    ENCODER_OPTION_TRACE_CALLBACK_CONTEXT = 27,
    ENCODER_OPTION_GET_STATISTICS = 28,
    ENCODER_OPTION_STATISTICS_LOG_INTERVAL = 29,
    ENCODER_OPTION_IS_LOSSLESS_LINK = 30,
    ENCODER_OPTION_BITS_VARY_PERCENTAGE = 31,
}
pub type EncoderOption = ENCODER_OPTION;

impl ENCODER_OPTION {
    #[inline]
    pub const fn from_raw(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::ENCODER_OPTION_DATAFORMAT),
            1 => Some(Self::ENCODER_OPTION_IDR_INTERVAL),
            2 => Some(Self::ENCODER_OPTION_SVC_ENCODE_PARAM_BASE),
            3 => Some(Self::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT),
            4 => Some(Self::ENCODER_OPTION_FRAME_RATE),
            5 => Some(Self::ENCODER_OPTION_BITRATE),
            6 => Some(Self::ENCODER_OPTION_MAX_BITRATE),
            7 => Some(Self::ENCODER_OPTION_INTER_SPATIAL_PRED),
            8 => Some(Self::ENCODER_OPTION_RC_MODE),
            9 => Some(Self::ENCODER_OPTION_RC_FRAME_SKIP),
            10 => Some(Self::ENCODER_PADDING_PADDING),
            11 => Some(Self::ENCODER_OPTION_PROFILE),
            12 => Some(Self::ENCODER_OPTION_LEVEL),
            13 => Some(Self::ENCODER_OPTION_NUMBER_REF),
            14 => Some(Self::ENCODER_OPTION_DELIVERY_STATUS),
            15 => Some(Self::ENCODER_LTR_RECOVERY_REQUEST),
            16 => Some(Self::ENCODER_LTR_MARKING_FEEDBACK),
            17 => Some(Self::ENCODER_LTR_MARKING_PERIOD),
            18 => Some(Self::ENCODER_OPTION_LTR),
            19 => Some(Self::ENCODER_OPTION_COMPLEXITY),
            20 => Some(Self::ENCODER_OPTION_ENABLE_SSEI),
            21 => Some(Self::ENCODER_OPTION_ENABLE_PREFIX_NAL_ADDING),
            22 => Some(Self::ENCODER_OPTION_SPS_PPS_ID_STRATEGY),
            23 => Some(Self::ENCODER_OPTION_CURRENT_PATH),
            24 => Some(Self::ENCODER_OPTION_DUMP_FILE),
            25 => Some(Self::ENCODER_OPTION_TRACE_LEVEL),
            26 => Some(Self::ENCODER_OPTION_TRACE_CALLBACK),
            27 => Some(Self::ENCODER_OPTION_TRACE_CALLBACK_CONTEXT),
            28 => Some(Self::ENCODER_OPTION_GET_STATISTICS),
            29 => Some(Self::ENCODER_OPTION_STATISTICS_LOG_INTERVAL),
            30 => Some(Self::ENCODER_OPTION_IS_LOSSLESS_LINK),
            31 => Some(Self::ENCODER_OPTION_BITS_VARY_PERCENTAGE),
            _ => None,
        }
    }
}

/// Decoder option identifiers (`DECODER_OPTION`).
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum DECODER_OPTION {
    #[default]
    DECODER_OPTION_END_OF_STREAM = 1,
    DECODER_OPTION_VCL_NAL = 2,
    DECODER_OPTION_TEMPORAL_ID = 3,
    DECODER_OPTION_FRAME_NUM = 4,
    DECODER_OPTION_IDR_PIC_ID = 5,
    DECODER_OPTION_LTR_MARKING_FLAG = 6,
    DECODER_OPTION_LTR_MARKED_FRAME_NUM = 7,
    DECODER_OPTION_ERROR_CON_IDC = 8,
    DECODER_OPTION_TRACE_LEVEL = 9,
    DECODER_OPTION_TRACE_CALLBACK = 10,
    DECODER_OPTION_TRACE_CALLBACK_CONTEXT = 11,
    DECODER_OPTION_GET_STATISTICS = 12,
    DECODER_OPTION_GET_SAR_INFO = 13,
    DECODER_OPTION_PROFILE = 14,
    DECODER_OPTION_LEVEL = 15,
    DECODER_OPTION_STATISTICS_LOG_INTERVAL = 16,
    DECODER_OPTION_IS_REF_PIC = 17,
    DECODER_OPTION_NUM_OF_FRAMES_REMAINING_IN_BUFFER = 18,
    DECODER_OPTION_NUM_OF_THREADS = 19,
}

pub type DecoderOption = DECODER_OPTION;

impl DECODER_OPTION {
    #[inline]
    pub const fn from_raw(v: i32) -> Option<Self> {
        match v {
            1 => Some(Self::DECODER_OPTION_END_OF_STREAM),
            2 => Some(Self::DECODER_OPTION_VCL_NAL),
            3 => Some(Self::DECODER_OPTION_TEMPORAL_ID),
            4 => Some(Self::DECODER_OPTION_FRAME_NUM),
            5 => Some(Self::DECODER_OPTION_IDR_PIC_ID),
            6 => Some(Self::DECODER_OPTION_LTR_MARKING_FLAG),
            7 => Some(Self::DECODER_OPTION_LTR_MARKED_FRAME_NUM),
            8 => Some(Self::DECODER_OPTION_ERROR_CON_IDC),
            9 => Some(Self::DECODER_OPTION_TRACE_LEVEL),
            10 => Some(Self::DECODER_OPTION_TRACE_CALLBACK),
            11 => Some(Self::DECODER_OPTION_TRACE_CALLBACK_CONTEXT),
            12 => Some(Self::DECODER_OPTION_GET_STATISTICS),
            13 => Some(Self::DECODER_OPTION_GET_SAR_INFO),
            14 => Some(Self::DECODER_OPTION_PROFILE),
            15 => Some(Self::DECODER_OPTION_LEVEL),
            16 => Some(Self::DECODER_OPTION_STATISTICS_LOG_INTERVAL),
            17 => Some(Self::DECODER_OPTION_IS_REF_PIC),
            18 => Some(Self::DECODER_OPTION_NUM_OF_FRAMES_REMAINING_IN_BUFFER),
            19 => Some(Self::DECODER_OPTION_NUM_OF_THREADS),
            _ => None,
        }
    }
}

/// Error concealment modes (`ERROR_CON_IDC`).
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum ERROR_CON_IDC {
    #[default]
    ERROR_CON_DISABLE = 0,
    ERROR_CON_FRAME_COPY = 1,
    ERROR_CON_SLICE_COPY = 2,
    ERROR_CON_FRAME_COPY_CROSS_IDR = 3,
    ERROR_CON_SLICE_COPY_CROSS_IDR = 4,
    ERROR_CON_SLICE_COPY_CROSS_IDR_FREEZE_RES_CHANGE = 5,
    ERROR_CON_SLICE_MV_COPY_CROSS_IDR = 6,
    ERROR_CON_SLICE_MV_COPY_CROSS_IDR_FREEZE_RES_CHANGE = 7,
}

/// Feedback VCL NAL state (`FEEDBACK_VCL_NAL_IN_AU`).
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum FEEDBACK_VCL_NAL_IN_AU {
    #[default]
    FEEDBACK_NON_VCL_NAL = 0,
    FEEDBACK_VCL_NAL = 1,
    FEEDBACK_UNKNOWN_NAL = 2,
}

/// Layer type being encoded (`LAYER_TYPE`).
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum LAYER_TYPE {
    #[default]
    NON_VIDEO_CODING_LAYER = 0,
    VIDEO_CODING_LAYER = 1,
}

/// Spatial layer enumeration (`LAYER_NUM`).
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum LAYER_NUM {
    #[default]
    SPATIAL_LAYER_0 = 0,
    SPATIAL_LAYER_1 = 1,
    SPATIAL_LAYER_2 = 2,
    SPATIAL_LAYER_3 = 3,
    SPATIAL_LAYER_ALL = 4,
}

impl LAYER_NUM {
    #[inline]
    pub const fn from_raw(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::SPATIAL_LAYER_0),
            1 => Some(Self::SPATIAL_LAYER_1),
            2 => Some(Self::SPATIAL_LAYER_2),
            3 => Some(Self::SPATIAL_LAYER_3),
            4 => Some(Self::SPATIAL_LAYER_ALL),
            _ => None,
        }
    }
}

/// Video bitstream type (`VIDEO_BITSTREAM_TYPE`).
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum VIDEO_BITSTREAM_TYPE {
    VIDEO_BITSTREAM_AVC = 0,
    #[default]
    VIDEO_BITSTREAM_SVC = 1,
}

pub const VIDEO_BITSTREAM_DEFAULT: VIDEO_BITSTREAM_TYPE = VIDEO_BITSTREAM_TYPE::VIDEO_BITSTREAM_SVC;

/// Keyframe request type (`KEY_FRAME_REQUEST_TYPE`).
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum KEY_FRAME_REQUEST_TYPE {
    #[default]
    NO_RECOVERY_REQUSET = 0,
    LTR_RECOVERY_REQUEST = 1,
    IDR_RECOVERY_REQUEST = 2,
    NO_LTR_MARKING_FEEDBACK = 3,
    LTR_MARKING_SUCCESS = 4,
    LTR_MARKING_FAILED = 5,
}

/// Rate control modes (`RC_MODES`).
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum RC_MODES {
    #[default]
    RC_QUALITY_MODE = 0,
    RC_BITRATE_MODE = 1,
    RC_BUFFERBASED_MODE = 2,
    RC_TIMESTAMP_MODE = 3,
    RC_BITRATE_MODE_POST_SKIP = 4,
    RC_OFF_MODE = -1,
}

pub type RCMode = RC_MODES;

impl RC_MODES {
    /// Maps a raw C `int` from `SEncParamBase` / `SEncParamExt` to `RC_MODES`.
    /// Out-of-range values map to `RC_BITRATE_MODE_POST_SKIP`, which `ParamValidation`
    /// rejects with `ENC_RETURN_UNSUPPORTED_PARA`.
    #[inline]
    pub const fn from_raw_param(v: i32) -> Self {
        match v {
            -1 => Self::RC_OFF_MODE,
            0 => Self::RC_QUALITY_MODE,
            1 => Self::RC_BITRATE_MODE,
            2 => Self::RC_BUFFERBASED_MODE,
            3 => Self::RC_TIMESTAMP_MODE,
            _ => Self::RC_BITRATE_MODE_POST_SKIP,
        }
    }
}

/// Profile IDC enumeration (`EProfileIdc`).
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum EProfileIdc {
    #[default]
    PRO_UNKNOWN = 0,
    PRO_BASELINE = 66,
    PRO_MAIN = 77,
    PRO_EXTENDED = 88,
    PRO_HIGH = 100,
    PRO_HIGH10 = 110,
    PRO_HIGH422 = 122,
    PRO_HIGH444 = 144,
    PRO_CAVLC444 = 244,
    PRO_SCALABLE_BASELINE = 83,
    PRO_SCALABLE_HIGH = 86,
}

impl EProfileIdc {
    /// Maps a raw C `int` to `EProfileIdc`. Non-zero unknown values map to
    /// `PRO_EXTENDED` so `ParamTranscode` treats them as explicitly specified and
    /// `CheckProfileSetting` logs a warning and falls back just as C++ does.
    #[inline]
    pub const fn from_raw_param(v: i32) -> Self {
        match v {
            0 => Self::PRO_UNKNOWN,
            66 => Self::PRO_BASELINE,
            77 => Self::PRO_MAIN,
            88 => Self::PRO_EXTENDED,
            100 => Self::PRO_HIGH,
            110 => Self::PRO_HIGH10,
            122 => Self::PRO_HIGH422,
            144 => Self::PRO_HIGH444,
            244 => Self::PRO_CAVLC444,
            83 => Self::PRO_SCALABLE_BASELINE,
            86 => Self::PRO_SCALABLE_HIGH,
            _ => Self::PRO_EXTENDED,
        }
    }
}

/// Level IDC enumeration (`ELevelIdc`).
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum ELevelIdc {
    #[default]
    LEVEL_UNKNOWN = 0,
    LEVEL_1_0 = 10,
    LEVEL_1_B = 9,
    LEVEL_1_1 = 11,
    LEVEL_1_2 = 12,
    LEVEL_1_3 = 13,
    LEVEL_2_0 = 20,
    LEVEL_2_1 = 21,
    LEVEL_2_2 = 22,
    LEVEL_3_0 = 30,
    LEVEL_3_1 = 31,
    LEVEL_3_2 = 32,
    LEVEL_4_0 = 40,
    LEVEL_4_1 = 41,
    LEVEL_4_2 = 42,
    LEVEL_5_0 = 50,
    LEVEL_5_1 = 51,
    LEVEL_5_2 = 52,
}

impl ELevelIdc {
    #[inline]
    pub const fn from_raw(v: i32) -> Self {
        match v {
            9 => Self::LEVEL_1_B,
            10 => Self::LEVEL_1_0,
            11 => Self::LEVEL_1_1,
            12 => Self::LEVEL_1_2,
            13 => Self::LEVEL_1_3,
            20 => Self::LEVEL_2_0,
            21 => Self::LEVEL_2_1,
            22 => Self::LEVEL_2_2,
            30 => Self::LEVEL_3_0,
            31 => Self::LEVEL_3_1,
            32 => Self::LEVEL_3_2,
            40 => Self::LEVEL_4_0,
            41 => Self::LEVEL_4_1,
            42 => Self::LEVEL_4_2,
            50 => Self::LEVEL_5_0,
            51 => Self::LEVEL_5_1,
            52 => Self::LEVEL_5_2,
            _ => Self::LEVEL_UNKNOWN,
        }
    }
}

/// Slicing modes (`SliceModeEnum`).
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum SliceModeEnum {
    #[default]
    SM_SINGLE_SLICE = 0,
    SM_FIXEDSLCNUM_SLICE = 1,
    SM_RASTER_SLICE = 2,
    SM_SIZELIMITED_SLICE = 3,
    SM_RESERVED = 4,
}

pub type SliceMode = SliceModeEnum;

impl SliceModeEnum {
    #[inline]
    pub const fn from_raw(v: i32) -> Self {
        match v {
            0 => Self::SM_SINGLE_SLICE,
            1 => Self::SM_FIXEDSLCNUM_SLICE,
            2 => Self::SM_RASTER_SLICE,
            3 => Self::SM_SIZELIMITED_SLICE,
            _ => Self::SM_RESERVED,
        }
    }
}

/// Video format in SPS VUI (`EVideoFormatSPS`).
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum EVideoFormatSPS {
    #[default]
    VF_COMPONENT = 0,
    VF_PAL = 1,
    VF_NTSC = 2,
    VF_SECAM = 3,
    VF_MAC = 4,
    VF_UNDEF = 5,
    VF_NUM_ENUM = 6,
}

/// Color primaries (`EColorPrimaries`).
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum EColorPrimaries {
    #[default]
    CP_RESERVED0 = 0,
    CP_BT709 = 1,
    CP_UNDEF = 2,
    CP_RESERVED3 = 3,
    CP_BT470M = 4,
    CP_BT470BG = 5,
    CP_SMPTE170M = 6,
    CP_SMPTE240M = 7,
    CP_FILM = 8,
    CP_BT2020 = 9,
    CP_NUM_ENUM = 10,
}

/// Transfer characteristics (`ETransferCharacteristics`).
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum ETransferCharacteristics {
    #[default]
    TRC_RESERVED0 = 0,
    TRC_BT709 = 1,
    TRC_UNDEF = 2,
    TRC_RESERVED3 = 3,
    TRC_BT470M = 4,
    TRC_BT470BG = 5,
    TRC_SMPTE170M = 6,
    TRC_SMPTE240M = 7,
    TRC_LINEAR = 8,
    TRC_LOG100 = 9,
    TRC_LOG316 = 10,
    TRC_IEC61966_2_4 = 11,
    TRC_BT1361E = 12,
    TRC_IEC61966_2_1 = 13,
    TRC_BT2020_10 = 14,
    TRC_BT2020_12 = 15,
    TRC_NUM_ENUM = 16,
}

/// Color matrix (`EColorMatrix`).
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum EColorMatrix {
    #[default]
    CM_GBR = 0,
    CM_BT709 = 1,
    CM_UNDEF = 2,
    CM_RESERVED3 = 3,
    CM_FCC = 4,
    CM_BT470BG = 5,
    CM_SMPTE170M = 6,
    CM_SMPTE240M = 7,
    CM_YCGCO = 8,
    CM_BT2020NC = 9,
    CM_BT2020C = 10,
    CM_NUM_ENUM = 11,
}

/// Sample aspect ratio (`ESampleAspectRatio`).
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum ESampleAspectRatio {
    #[default]
    ASP_UNSPECIFIED = 0,
    ASP_1x1 = 1,
    ASP_12x11 = 2,
    ASP_10x11 = 3,
    ASP_16x11 = 4,
    ASP_40x33 = 5,
    ASP_24x11 = 6,
    ASP_20x11 = 7,
    ASP_32x11 = 8,
    ASP_80x33 = 9,
    ASP_18x11 = 10,
    ASP_15x11 = 11,
    ASP_64x33 = 12,
    ASP_160x99 = 13,
    ASP_EXT_SAR = 255,
}

impl ESampleAspectRatio {
    #[inline]
    pub const fn from_raw(v: i32) -> Self {
        match v {
            1 => Self::ASP_1x1,
            2 => Self::ASP_12x11,
            3 => Self::ASP_10x11,
            4 => Self::ASP_16x11,
            5 => Self::ASP_40x33,
            6 => Self::ASP_24x11,
            7 => Self::ASP_20x11,
            8 => Self::ASP_32x11,
            9 => Self::ASP_80x33,
            10 => Self::ASP_18x11,
            11 => Self::ASP_15x11,
            12 => Self::ASP_64x33,
            13 => Self::ASP_160x99,
            255 => Self::ASP_EXT_SAR,
            _ => Self::ASP_UNSPECIFIED,
        }
    }
}

/// Encoder application scenario / usage type (`EUsageType`).
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum EUsageType {
    #[default]
    CAMERA_VIDEO_REAL_TIME = 0,
    SCREEN_CONTENT_REAL_TIME = 1,
    CAMERA_VIDEO_NON_REAL_TIME = 2,
    SCREEN_CONTENT_NON_REAL_TIME = 3,
    INPUT_CONTENT_TYPE_ALL = 4,
}

impl EUsageType {
    #[inline]
    pub const fn from_raw(v: i32) -> Self {
        match v {
            0 => Self::CAMERA_VIDEO_REAL_TIME,
            1 => Self::SCREEN_CONTENT_REAL_TIME,
            2 => Self::CAMERA_VIDEO_NON_REAL_TIME,
            3 => Self::SCREEN_CONTENT_NON_REAL_TIME,
            _ => Self::INPUT_CONTENT_TYPE_ALL,
        }
    }
}

/// Encoder complexity modes (`ECOMPLEXITY_MODE`).
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum ECOMPLEXITY_MODE {
    #[default]
    LOW_COMPLEXITY = 0,
    MEDIUM_COMPLEXITY = 1,
    HIGH_COMPLEXITY = 2,
}

pub type EComplexityMode = ECOMPLEXITY_MODE;

impl ECOMPLEXITY_MODE {
    #[inline]
    pub const fn from_raw(v: i32) -> Self {
        match v {
            0 => Self::LOW_COMPLEXITY,
            1 => Self::MEDIUM_COMPLEXITY,
            _ => Self::HIGH_COMPLEXITY,
        }
    }
}

/// Parameter set strategy (`EParameterSetStrategy`).
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum EParameterSetStrategy {
    CONSTANT_ID = 0,
    #[default]
    INCREASING_ID = 0x01,
    SPS_LISTING = 0x02,
    SPS_LISTING_AND_PPS_INCREASING = 0x03,
    SPS_PPS_LISTING = 0x06,
}

impl EParameterSetStrategy {
    #[inline]
    pub const fn from_raw(v: i32) -> Self {
        match v {
            0 => Self::CONSTANT_ID,
            0x01 => Self::INCREASING_ID,
            0x02 => Self::SPS_LISTING,
            0x03 => Self::SPS_LISTING_AND_PPS_INCREASING,
            0x06 => Self::SPS_PPS_LISTING,
            _ => Self::INCREASING_ID,
        }
    }
}

/// OpenH264 version record (`OpenH264Version`).
#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct OpenH264Version {
    pub uMajor: u32,
    pub uMinor: u32,
    pub uRevision: u32,
    pub uReserved: u32,
}

/// Slice configuration (`SSliceArgument`).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SSliceArgument {
    pub uiSliceMode: SliceModeEnum,
    pub uiSliceNum: u32,
    pub uiSliceMbNum: [u32; MAX_SLICES_NUM_TMP],
    pub uiSliceSizeConstraint: u32,
}

impl Default for SSliceArgument {
    fn default() -> Self {
        Self {
            uiSliceMode: SliceModeEnum::SM_SINGLE_SLICE,
            uiSliceNum: 0,
            uiSliceMbNum: [0; MAX_SLICES_NUM_TMP],
            uiSliceSizeConstraint: 0,
        }
    }
}

/// Spatial layer configuration (`SSpatialLayerConfig`).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SSpatialLayerConfig {
    pub iVideoWidth: i32,
    pub iVideoHeight: i32,
    pub fFrameRate: f32,
    pub iSpatialBitrate: i32,
    pub iMaxSpatialBitrate: i32,
    pub uiProfileIdc: EProfileIdc,
    pub uiLevelIdc: ELevelIdc,
    pub iDLayerQp: i32,

    pub sSliceArgument: SSliceArgument,

    pub bVideoSignalTypePresent: bool,
    pub uiVideoFormat: u8,
    pub bFullRange: bool,
    pub bColorDescriptionPresent: bool,
    pub uiColorPrimaries: u8,
    pub uiTransferCharacteristics: u8,
    pub uiColorMatrix: u8,

    pub bAspectRatioPresent: bool,
    pub eAspectRatio: ESampleAspectRatio,
    pub sAspectRatioExtWidth: u16,
    pub sAspectRatioExtHeight: u16,
}

impl Default for SSpatialLayerConfig {
    fn default() -> Self {
        Self {
            iVideoWidth: 0,
            iVideoHeight: 0,
            fFrameRate: 0.0,
            iSpatialBitrate: 0,
            iMaxSpatialBitrate: 0,
            uiProfileIdc: EProfileIdc::PRO_UNKNOWN,
            uiLevelIdc: ELevelIdc::LEVEL_UNKNOWN,
            iDLayerQp: 0,
            sSliceArgument: SSliceArgument::default(),
            bVideoSignalTypePresent: false,
            uiVideoFormat: 0,
            bFullRange: false,
            bColorDescriptionPresent: false,
            uiColorPrimaries: 0,
            uiTransferCharacteristics: 0,
            uiColorMatrix: 0,
            bAspectRatioPresent: false,
            eAspectRatio: ESampleAspectRatio::ASP_UNSPECIFIED,
            sAspectRatioExtWidth: 0,
            sAspectRatioExtHeight: 0,
        }
    }
}

/// Basic encoder parameters (`SEncParamBase`).
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SEncParamBase {
    pub iUsageType: EUsageType,
    pub iPicWidth: i32,
    pub iPicHeight: i32,
    pub iTargetBitrate: i32,
    pub iRCMode: RC_MODES,
    pub fMaxFrameRate: f32,
}

/// Extended encoder parameters (`SEncParamExt`).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SEncParamExt {
    pub iUsageType: EUsageType,
    pub iPicWidth: i32,
    pub iPicHeight: i32,
    pub iTargetBitrate: i32,
    pub iRCMode: RC_MODES,
    pub fMaxFrameRate: f32,

    pub iTemporalLayerNum: i32,
    pub iSpatialLayerNum: i32,
    pub sSpatialLayers: [SSpatialLayerConfig; MAX_SPATIAL_LAYER_NUM],

    pub iComplexityMode: ECOMPLEXITY_MODE,
    pub uiIntraPeriod: u32,
    pub iNumRefFrame: i32,
    pub eSpsPpsIdStrategy: EParameterSetStrategy,
    pub bPrefixNalAddingCtrl: bool,
    pub bEnableSSEI: bool,
    pub bSimulcastAVC: bool,
    pub iPaddingFlag: i32,
    pub iEntropyCodingModeFlag: i32,

    pub bEnableFrameSkip: bool,
    pub iMaxBitrate: i32,
    pub iMaxQp: i32,
    pub iMinQp: i32,
    pub uiMaxNalSize: u32,

    pub bEnableLongTermReference: bool,
    pub iLTRRefNum: i32,
    pub iLtrMarkPeriod: u32,

    pub iMultipleThreadIdc: u16,
    pub bUseLoadBalancing: bool,

    pub iLoopFilterDisableIdc: i32,
    pub iLoopFilterAlphaC0Offset: i32,
    pub iLoopFilterBetaOffset: i32,

    pub bEnableDenoise: bool,
    pub bEnableBackgroundDetection: bool,
    pub bEnableAdaptiveQuant: bool,
    pub bEnableFrameCroppingFlag: bool,
    pub bEnableSceneChangeDetect: bool,

    pub bIsLosslessLink: bool,
    pub bFixRCOverShoot: bool,
    pub iIdrBitrateRatio: i32,
    pub bPsnrY: bool,
    pub bPsnrU: bool,
    pub bPsnrV: bool,
}

impl Default for SEncParamExt {
    fn default() -> Self {
        Self {
            iUsageType: EUsageType::CAMERA_VIDEO_REAL_TIME,
            iPicWidth: 0,
            iPicHeight: 0,
            iTargetBitrate: 0,
            iRCMode: RC_MODES::RC_QUALITY_MODE,
            fMaxFrameRate: 0.0,
            iTemporalLayerNum: 1,
            iSpatialLayerNum: 1,
            sSpatialLayers: [SSpatialLayerConfig::default(); MAX_SPATIAL_LAYER_NUM],
            iComplexityMode: ECOMPLEXITY_MODE::LOW_COMPLEXITY,
            uiIntraPeriod: 0,
            iNumRefFrame: 1,
            eSpsPpsIdStrategy: EParameterSetStrategy::INCREASING_ID,
            bPrefixNalAddingCtrl: false,
            bEnableSSEI: false,
            bSimulcastAVC: false,
            iPaddingFlag: 0,
            iEntropyCodingModeFlag: 0,
            bEnableFrameSkip: false,
            iMaxBitrate: UNSPECIFIED_BIT_RATE,
            iMaxQp: 51,
            iMinQp: 0,
            uiMaxNalSize: 0,
            bEnableLongTermReference: false,
            iLTRRefNum: 0,
            iLtrMarkPeriod: 0,
            iMultipleThreadIdc: 1,
            bUseLoadBalancing: false,
            iLoopFilterDisableIdc: 0,
            iLoopFilterAlphaC0Offset: 0,
            iLoopFilterBetaOffset: 0,
            bEnableDenoise: false,
            bEnableBackgroundDetection: true,
            bEnableAdaptiveQuant: true,
            bEnableFrameCroppingFlag: true,
            bEnableSceneChangeDetect: true,
            bIsLosslessLink: false,
            bFixRCOverShoot: false,
            iIdrBitrateRatio: 0,
            bPsnrY: false,
            bPsnrU: false,
            bPsnrV: false,
        }
    }
}

/// Uncompressed input picture (`SSourcePicture`).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SSourcePicture {
    pub iColorFormat: i32,
    pub iStride: [i32; 4],
    pub pData: [*mut u8; 4],
    pub iPicWidth: i32,
    pub iPicHeight: i32,
    pub uiTimeStamp: i64,
    pub bPsnrY: bool,
    pub bPsnrU: bool,
    pub bPsnrV: bool,
}

impl Default for SSourcePicture {
    fn default() -> Self {
        Self {
            iColorFormat: EVideoFormatType::videoFormatI420 as i32,
            iStride: [0; 4],
            pData: [ptr::null_mut(); 4],
            iPicWidth: 0,
            iPicHeight: 0,
            uiTimeStamp: 0,
            bPsnrY: false,
            bPsnrU: false,
            bPsnrV: false,
        }
    }
}

impl SSourcePicture {
    /// Borrows the raw plane pointers of this C [`SSourcePicture`] as a slice-backed
    /// [`SourcePicture`] for the duration of an `EncodeFrame` FFI call.
    ///
    /// Planes with null pointers or non-positive / insufficient dimensions or strides
    /// are mapped to `None`, which the encoder rejects with `cmInitParaError` or
    /// `cmUnsupportedData` just as it would for null or invalid raw inputs.
    ///
    /// # Safety
    ///
    /// For each non-null plane `i` in `0..3` where `iPicWidth >= 2`, `iPicHeight >= 2`,
    /// and `iStride[i]` is at least that plane's pixel width, `pData[i]` must point to a
    /// readable allocation of at least `(plane_height - 1) * iStride[i] + plane_width`
    /// initialized bytes valid for the lifetime `'_'` of `&self`.
    #[cfg(any(feature = "c-api", feature = "cxx-api", test))]
    #[allow(unsafe_code, dead_code)]
    pub(crate) unsafe fn as_source_picture(&self) -> SourcePicture<'_> {
        #[allow(unsafe_code)]
        unsafe fn plane_slice<'a>(
            ptr: *mut u8,
            width: i32,
            height: i32,
            stride: i32,
        ) -> Option<&'a [u8]> {
            if ptr.is_null() || width <= 0 || height <= 0 || stride < width {
                return None;
            }
            let w = width as usize;
            let h = height as usize;
            let s = stride as usize;
            let len = (h - 1).checked_mul(s)?.checked_add(w)?;
            if len > isize::MAX as usize {
                return None;
            }
            Some(unsafe { std::slice::from_raw_parts(ptr.cast_const(), len) })
        }

        let w_y = self.iPicWidth & !1;
        let h_y = self.iPicHeight & !1;
        let w_uv = self.iPicWidth >> 1;
        let h_uv = self.iPicHeight >> 1;

        SourcePicture {
            iColorFormat: self.iColorFormat,
            iStride: self.iStride,
            pData: [
                unsafe { plane_slice(self.pData[0], w_y, h_y, self.iStride[0]) },
                unsafe { plane_slice(self.pData[1], w_uv, h_uv, self.iStride[1]) },
                unsafe { plane_slice(self.pData[2], w_uv, h_uv, self.iStride[2]) },
                None,
            ],
            iPicWidth: self.iPicWidth,
            iPicHeight: self.iPicHeight,
            uiTimeStamp: self.uiTimeStamp,
            bPsnrY: self.bPsnrY,
            bPsnrU: self.bPsnrU,
            bPsnrV: self.bPsnrV,
        }
    }
}

/// Safe, slice-backed uncompressed input picture for [`crate::api::encoder::Encoder::encode_frame`].
///
/// Mirrors [`SSourcePicture`], replacing raw `pData: [*mut u8; 4]` pointers with
/// borrowed byte slices so the Rust encoder pipeline can ingest frames without `unsafe`.
#[derive(Debug, Copy, Clone)]
pub struct SourcePicture<'a> {
    pub iColorFormat: i32,
    pub iStride: [i32; 4],
    pub pData: [Option<&'a [u8]>; 4],
    pub iPicWidth: i32,
    pub iPicHeight: i32,
    pub uiTimeStamp: i64,
    pub bPsnrY: bool,
    pub bPsnrU: bool,
    pub bPsnrV: bool,
}

impl Default for SourcePicture<'_> {
    fn default() -> Self {
        Self {
            iColorFormat: EVideoFormatType::videoFormatI420 as i32,
            iStride: [0; 4],
            pData: [None; 4],
            iPicWidth: 0,
            iPicHeight: 0,
            uiTimeStamp: 0,
            bPsnrY: false,
            bPsnrU: false,
            bPsnrV: false,
        }
    }
}

impl<'a> SourcePicture<'a> {
    /// Constructs a packed I420 [`SourcePicture`] from a contiguous `Y || U || V` byte slice.
    pub fn from_i420(yuv: &'a [u8], width: i32, height: i32) -> Self {
        let w = width.max(0) as usize;
        let h = height.max(0) as usize;
        let luma = w.saturating_mul(h);
        let chroma = (w >> 1).saturating_mul(h >> 1);
        let (y, uv) = yuv.split_at(luma.min(yuv.len()));
        let (u, v) = uv.split_at(chroma.min(uv.len()));
        Self::from_i420_planes(y, u, v, width, height)
    }

    /// Constructs a tightly packed I420 [`SourcePicture`] from separate `Y`, `U`, and `V` planes
    /// with default strides `[width, width / 2, width / 2]`.
    pub fn from_i420_planes(
        y: &'a [u8],
        u: &'a [u8],
        v: &'a [u8],
        width: i32,
        height: i32,
    ) -> Self {
        Self::from_i420_strided(y, u, v, width, height, [width, width >> 1, width >> 1])
    }

    /// Constructs a strided I420 [`SourcePicture`] from separate `Y`, `U`, and `V` planes
    /// and explicit plane row strides `[stride_y, stride_u, stride_v]`.
    pub fn from_i420_strided(
        y: &'a [u8],
        u: &'a [u8],
        v: &'a [u8],
        width: i32,
        height: i32,
        strides: [i32; 3],
    ) -> Self {
        Self {
            iColorFormat: EVideoFormatType::videoFormatI420 as i32,
            iStride: [strides[0], strides[1], strides[2], 0],
            pData: [Some(y), Some(u), Some(v), None],
            iPicWidth: width,
            iPicHeight: height,
            uiTimeStamp: 0,
            bPsnrY: false,
            bPsnrU: false,
            bPsnrV: false,
        }
    }

    /// Sets the picture timestamp (`uiTimeStamp`) and returns the updated [`SourcePicture`].
    pub fn with_timestamp(mut self, timestamp: i64) -> Self {
        self.uiTimeStamp = timestamp;
        self
    }
}

/// Bitstream layer output descriptor (`SLayerBSInfo`).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SLayerBSInfo {
    pub uiTemporalId: u8,
    pub uiSpatialId: u8,
    pub uiQualityId: u8,
    pub eFrameType: EVideoFrameType,
    pub uiLayerType: u8,
    pub iSubSeqId: i32,
    pub iNalCount: i32,
    pub(crate) pNalLengthInByte: *mut i32,
    pub(crate) pBsBuf: *mut u8,
    pub rPsnr: [f32; 3],
}

impl Default for SLayerBSInfo {
    fn default() -> Self {
        Self {
            uiTemporalId: 0,
            uiSpatialId: 0,
            uiQualityId: 0,
            eFrameType: EVideoFrameType::videoFrameTypeInvalid,
            uiLayerType: 0,
            iSubSeqId: 0,
            iNalCount: 0,
            pNalLengthInByte: ptr::null_mut(),
            pBsBuf: ptr::null_mut(),
            rPsnr: [0.0; 3],
        }
    }
}

impl SLayerBSInfo {
    /// Raw pointer to the per-NAL byte length array (`pNalLengthInByte`).
    #[inline]
    pub const fn nal_lengths_ptr(&self) -> *const i32 {
        self.pNalLengthInByte.cast_const()
    }

    /// Raw pointer to the layer bitstream buffer (`pBsBuf`).
    #[inline]
    pub const fn bs_buf_ptr(&self) -> *const u8 {
        self.pBsBuf.cast_const()
    }

    /// Returns the per-NAL byte lengths (`pNalLengthInByte[..iNalCount]`) for this layer,
    /// borrowing the owning [`Encoder`](crate::api::encoder::Encoder).
    ///
    /// Valid only for the [`SFrameBSInfo`] produced by the most recent
    /// [`encode_frame`](crate::api::encoder::Encoder::encode_frame) or
    /// [`encode_parameter_sets`](crate::api::encoder::Encoder::encode_parameter_sets)
    /// call on `enc`. After a subsequent encode call, returns `&[]` if the layer/NAL
    /// layout changed, or reads the newly overwritten lengths if the layout matches.
    #[inline]
    pub fn nal_lengths<'a>(&self, enc: &'a crate::api::encoder::Encoder) -> &'a [i32] {
        let Ok(count) = usize::try_from(self.iNalCount) else {
            return &[];
        };
        enc.slice_layer_nal_lengths(self.pNalLengthInByte.cast_const(), count)
    }

    /// Returns the concatenated Annex-B bitstream bytes (`pBsBuf`) for all NAL units in this layer,
    /// borrowing the owning [`Encoder`](crate::api::encoder::Encoder).
    ///
    /// Subject to the same single-call validity window as [`Self::nal_lengths`].
    #[inline]
    pub fn bs_buf<'a>(&self, enc: &'a crate::api::encoder::Encoder) -> &'a [u8] {
        let len: usize = self
            .nal_lengths(enc)
            .iter()
            .map(|&n| n.max(0) as usize)
            .sum();
        enc.slice_layer_bs_buf(self.pBsBuf.cast_const(), len)
    }

    /// Iterates over each individual Annex-B NAL unit slice in this layer,
    /// borrowing the owning [`Encoder`](crate::api::encoder::Encoder).
    #[inline]
    pub fn nals<'a>(
        &self,
        enc: &'a crate::api::encoder::Encoder,
    ) -> impl Iterator<Item = &'a [u8]> {
        let bs = self.bs_buf(enc);
        let mut offset = 0usize;
        self.nal_lengths(enc).iter().filter_map(move |&len| {
            let n = usize::try_from(len).ok()?;
            let end = offset.checked_add(n)?;
            let slice = bs.get(offset..end)?;
            offset = end;
            Some(slice)
        })
    }
}

/// Bitstream frame output descriptor (`SFrameBSInfo`).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SFrameBSInfo {
    pub iLayerNum: i32,
    pub sLayerInfo: [SLayerBSInfo; MAX_LAYER_NUM_OF_FRAME],
    pub eFrameType: EVideoFrameType,
    pub iFrameSizeInBytes: i32,
    pub uiTimeStamp: i64,
}

impl Default for SFrameBSInfo {
    fn default() -> Self {
        Self {
            iLayerNum: 0,
            sLayerInfo: [SLayerBSInfo::default(); MAX_LAYER_NUM_OF_FRAME],
            eFrameType: EVideoFrameType::videoFrameTypeInvalid,
            iFrameSizeInBytes: 0,
            uiTimeStamp: 0,
        }
    }
}

impl SFrameBSInfo {
    /// Returns the active [`SLayerBSInfo`] slice (`sLayerInfo[..iLayerNum]`) for this frame.
    pub fn layers(&self) -> &[SLayerBSInfo] {
        let count = self.iLayerNum.clamp(0, MAX_LAYER_NUM_OF_FRAME as i32) as usize;
        &self.sLayerInfo[..count]
    }
}

pub type PEncParamBase = *mut SEncParamBase;
pub type PLayerBSInfo = *mut SLayerBSInfo;
pub type PFrameBSInfo = *mut SFrameBSInfo;

/// Video bitstream property descriptor (`SVideoProperty`).
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SVideoProperty {
    pub size: u32,
    pub eVideoBsType: VIDEO_BITSTREAM_TYPE,
}

/// Decoder initialization parameters (`SDecodingParam`).
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SDecodingParam {
    /// Legacy C-ABI slot (`char* pFileNameRestructed` in `codec_app_def.h`), unused
    /// by the decoder. Typed as `usize` so the struct retains identical size, alignment,
    /// and field offsets while deriving `Send` + `Sync` and `Default` without `unsafe`.
    pub pFileNameRestructed: usize,
    pub uiCpuLoad: u32,
    pub uiTargetDqLayer: u8,
    pub eEcActiveIdc: ERROR_CON_IDC,
    pub bParseOnly: bool,
    pub sVideoProperty: SVideoProperty,
}

pub type PDecodingParam = *mut SDecodingParam;

/// System memory buffer information (`SSysMEMBuffer`).
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SSysMEMBuffer {
    pub iWidth: i32,
    pub iHeight: i32,
    pub iFormat: i32,
    pub iStride: [i32; 2],
}

/// Decoded frame destination buffer information payload.
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SBufferInfoUsrData {
    pub sSystemBuffer: SSysMEMBuffer,
}

impl SBufferInfoUsrData {
    /// Returns a reference to the system buffer metadata.
    #[inline]
    pub fn sys(&self) -> &SSysMEMBuffer {
        &self.sSystemBuffer
    }

    /// Returns a mutable reference to the system buffer metadata.
    #[inline]
    pub fn sys_mut(&mut self) -> &mut SSysMEMBuffer {
        &mut self.sSystemBuffer
    }
}

/// Decoded frame destination buffer metadata (`SBufferInfo`).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SBufferInfo {
    pub iBufferStatus: i32,
    pub uiInBsTimeStamp: u64,
    pub uiOutYuvTimeStamp: u64,
    pub UsrData: SBufferInfoUsrData,
    pub(crate) pDst: [*mut u8; 3],
}

impl Default for SBufferInfo {
    fn default() -> Self {
        Self {
            iBufferStatus: 0,
            uiInBsTimeStamp: 0,
            uiOutYuvTimeStamp: 0,
            UsrData: SBufferInfoUsrData {
                sSystemBuffer: SSysMEMBuffer::default(),
            },
            pDst: [ptr::null_mut(); 3],
        }
    }
}

impl SBufferInfo {
    /// Returns a reference to the system buffer metadata (`UsrData.sSystemBuffer`).
    #[inline]
    pub fn system_buffer(&self) -> &SSysMEMBuffer {
        &self.UsrData.sSystemBuffer
    }

    /// Returns a mutable reference to the system buffer metadata (`UsrData.sSystemBuffer`).
    #[inline]
    pub fn system_buffer_mut(&mut self) -> &mut SSysMEMBuffer {
        &mut self.UsrData.sSystemBuffer
    }

    /// Returns the strided `(y, u, v)` plane byte slices when `iBufferStatus == 1`
    /// and plane pointers and dimensions match the most recent frame emitted by `dec`,
    /// or `None` otherwise, borrowing the owning [`Decoder`](crate::api::decoder::Decoder).
    ///
    /// Each slice spans from the first pixel of row 0 to the last active pixel of
    /// the final row (`(height - 1) * stride + width`).
    ///
    /// Valid only until the next decode/flush/uninitialize call on `dec`; a stale
    /// `SBufferInfo` from an earlier call returns `None` once `dec` advances to
    /// another DPB slot or produces no output (`iBufferStatus == 0`).
    #[inline]
    pub fn yuv_planes<'a>(
        &self,
        dec: &'a crate::api::decoder::Decoder,
    ) -> Option<(&'a [u8], &'a [u8], &'a [u8])> {
        if self.iBufferStatus != 1 {
            return None;
        }
        let sys = &self.UsrData.sSystemBuffer;
        let w = usize::try_from(sys.iWidth).ok()?;
        let h = usize::try_from(sys.iHeight).ok()?;
        let sy = usize::try_from(sys.iStride[0]).ok()?;
        let suv = usize::try_from(sys.iStride[1]).ok()?;
        let uv_w = w / 2;
        let uv_h = h / 2;
        if w == 0
            || h == 0
            || uv_w == 0
            || uv_h == 0
            || sy < w
            || suv < uv_w
            || self.pDst[0].is_null()
            || self.pDst[1].is_null()
            || self.pDst[2].is_null()
        {
            return None;
        }
        let y_len = (h - 1).checked_mul(sy)?.checked_add(w)?;
        let uv_len = (uv_h - 1).checked_mul(suv)?.checked_add(uv_w)?;
        dec.slice_yuv_planes(&self.pDst, y_len, uv_len)
    }

    /// Iterates over the active pixel rows (`iWidth` bytes each) of the Y plane
    /// when `iBufferStatus == 1` (empty iterator otherwise), borrowing the owning
    /// [`Decoder`](crate::api::decoder::Decoder).
    #[inline]
    pub fn y_rows<'a>(
        &self,
        dec: &'a crate::api::decoder::Decoder,
    ) -> impl Iterator<Item = &'a [u8]> {
        let sys = &self.UsrData.sSystemBuffer;
        let w = usize::try_from(sys.iWidth).unwrap_or(0);
        let h = usize::try_from(sys.iHeight).unwrap_or(0);
        let sy = usize::try_from(sys.iStride[0]).unwrap_or(0);
        let plane = self.yuv_planes(dec).map(|(y, _, _)| y).unwrap_or(&[]);
        (0..if plane.is_empty() { 0 } else { h }).map(move |row| &plane[row * sy..row * sy + w])
    }

    /// Iterates over the active pixel rows (`iWidth / 2` bytes each) of the U plane
    /// when `iBufferStatus == 1` (empty iterator otherwise), borrowing the owning
    /// [`Decoder`](crate::api::decoder::Decoder).
    #[inline]
    pub fn u_rows<'a>(
        &self,
        dec: &'a crate::api::decoder::Decoder,
    ) -> impl Iterator<Item = &'a [u8]> {
        let sys = &self.UsrData.sSystemBuffer;
        let uv_w = usize::try_from(sys.iWidth).unwrap_or(0) / 2;
        let uv_h = usize::try_from(sys.iHeight).unwrap_or(0) / 2;
        let suv = usize::try_from(sys.iStride[1]).unwrap_or(0);
        let plane = self.yuv_planes(dec).map(|(_, u, _)| u).unwrap_or(&[]);
        (0..if plane.is_empty() { 0 } else { uv_h })
            .map(move |row| &plane[row * suv..row * suv + uv_w])
    }

    /// Iterates over the active pixel rows (`iWidth / 2` bytes each) of the V plane
    /// when `iBufferStatus == 1` (empty iterator otherwise), borrowing the owning
    /// [`Decoder`](crate::api::decoder::Decoder).
    #[inline]
    pub fn v_rows<'a>(
        &self,
        dec: &'a crate::api::decoder::Decoder,
    ) -> impl Iterator<Item = &'a [u8]> {
        let sys = &self.UsrData.sSystemBuffer;
        let uv_w = usize::try_from(sys.iWidth).unwrap_or(0) / 2;
        let uv_h = usize::try_from(sys.iHeight).unwrap_or(0) / 2;
        let suv = usize::try_from(sys.iStride[1]).unwrap_or(0);
        let plane = self.yuv_planes(dec).map(|(_, _, v)| v).unwrap_or(&[]);
        (0..if plane.is_empty() { 0 } else { uv_h })
            .map(move |row| &plane[row * suv..row * suv + uv_w])
    }

    /// Borrows this decoded frame as a [`SourcePicture`] suitable for passing to
    /// [`crate::api::encoder::Encoder::encode_frame`], or returns `None` if
    /// `iBufferStatus != 1`, borrowing the owning [`Decoder`](crate::api::decoder::Decoder).
    #[inline]
    pub fn as_source_picture<'a>(
        &self,
        dec: &'a crate::api::decoder::Decoder,
    ) -> Option<SourcePicture<'a>> {
        let (y, u, v) = self.yuv_planes(dec)?;
        let sys = &self.UsrData.sSystemBuffer;
        Some(SourcePicture::from_i420_strided(
            y,
            u,
            v,
            sys.iWidth,
            sys.iHeight,
            [sys.iStride[0], sys.iStride[1], sys.iStride[1]],
        ))
    }
}

/// Parsed bitstream output descriptor (`SParserBsInfo`).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SParserBsInfo {
    pub iNalNum: i32,
    pub(crate) pNalLenInByte: *mut i32,
    pub(crate) pDstBuff: *mut u8,
    pub iSpsWidthInPixel: i32,
    pub iSpsHeightInPixel: i32,
    pub uiInBsTimeStamp: u64,
    pub uiOutBsTimeStamp: u64,
}

impl Default for SParserBsInfo {
    fn default() -> Self {
        Self {
            iNalNum: 0,
            pNalLenInByte: ptr::null_mut(),
            pDstBuff: ptr::null_mut(),
            iSpsWidthInPixel: 0,
            iSpsHeightInPixel: 0,
            uiInBsTimeStamp: 0,
            uiOutBsTimeStamp: 0,
        }
    }
}

impl SParserBsInfo {
    /// Returns the per-NAL byte lengths slice (`pNalLenInByte[..iNalNum]`),
    /// borrowing the owning [`Decoder`](crate::api::decoder::Decoder).
    ///
    /// Valid only until the next [`decode_parser`](crate::api::decoder::Decoder::decode_parser)
    /// or [`uninitialize`](crate::api::decoder::Decoder::uninitialize) call on `dec`;
    /// after a subsequent call, returns `&[]` if `iNalNum` no longer matches the
    /// latest completed frame, or reads the newly overwritten lengths if it matches.
    #[inline]
    pub fn nal_lengths<'a>(&self, dec: &'a crate::api::decoder::Decoder) -> &'a [i32] {
        let Ok(count) = usize::try_from(self.iNalNum) else {
            return &[];
        };
        dec.slice_parser_nal_lengths(self.pNalLenInByte.cast_const(), count)
    }

    /// Returns the concatenated parsed bitstream slice (`pDstBuff[..total_bytes]`),
    /// borrowing the owning [`Decoder`](crate::api::decoder::Decoder).
    #[inline]
    pub fn dst_buf<'a>(&self, dec: &'a crate::api::decoder::Decoder) -> &'a [u8] {
        let total: usize = self
            .nal_lengths(dec)
            .iter()
            .map(|&n| n.max(0) as usize)
            .sum();
        dec.slice_parser_dst_buf(self.pDstBuff.cast_const(), total)
    }

    /// Iterates over the individual parsed NAL byte slices in this access unit,
    /// borrowing the owning [`Decoder`](crate::api::decoder::Decoder).
    #[inline]
    pub fn nals<'a>(
        &self,
        dec: &'a crate::api::decoder::Decoder,
    ) -> impl Iterator<Item = &'a [u8]> {
        let buf = self.dst_buf(dec);
        let mut offset = 0usize;
        self.nal_lengths(dec).iter().filter_map(move |&len| {
            let n = usize::try_from(len).ok()?;
            let end = offset.checked_add(n)?;
            let slice = buf.get(offset..end)?;
            offset = end;
            Some(slice)
        })
    }
}

pub type PParserBsInfo = *mut SParserBsInfo;

/// Decoder capability descriptor (`SDecoderCapability`).
#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct SDecoderCapability {
    pub iProfileIdc: i32,
    pub iProfileIop: i32,
    pub iLevelIdc: i32,
    pub iMaxMbps: i32,
    pub iMaxFs: i32,
    pub iMaxCpb: i32,
    pub iMaxDpb: i32,
    pub iMaxBr: i32,
    pub bRedPicCap: bool,
}

/// Video encoder runtime statistics (`SEncoderStatistics`).
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SEncoderStatistics {
    pub uiWidth: u32,
    pub uiHeight: u32,
    pub fAverageFrameSpeedInMs: f32,
    pub fAverageFrameRate: f32,
    pub fLatestFrameRate: f32,
    pub uiBitRate: u32,
    pub uiAverageFrameQP: u32,
    pub uiInputFrameCount: u32,
    pub uiSkippedFrameCount: u32,
    pub uiResolutionChangeTimes: u32,
    pub uiIDRReqNum: u32,
    pub uiIDRSentNum: u32,
    pub uiLTRSentNum: u32,
    pub iStatisticsTs: i64,
    pub iTotalEncodedBytes: std::ffi::c_ulong,
    pub iLastStatisticsBytes: std::ffi::c_ulong,
    pub iLastStatisticsFrameCount: std::ffi::c_ulong,
}

/// Video decoder runtime statistics (`SDecoderStatistics`).
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SDecoderStatistics {
    pub uiWidth: u32,
    pub uiHeight: u32,
    pub fAverageFrameSpeedInMs: f32,
    pub fActualAverageFrameSpeedInMs: f32,
    pub uiDecodedFrameCount: u32,
    pub uiResolutionChangeTimes: u32,
    pub uiIDRCorrectNum: u32,
    pub uiAvgEcRatio: u32,
    pub uiAvgEcPropRatio: u32,
    pub uiEcIDRNum: u32,
    pub uiEcFrameNum: u32,
    pub uiIDRLostNum: u32,
    pub uiFreezingIDRNum: u32,
    pub uiFreezingNonIDRNum: u32,
    pub iAvgLumaQp: i32,
    pub iSpsReportErrorNum: i32,
    pub iSubSpsReportErrorNum: i32,
    pub iPpsReportErrorNum: i32,
    pub iSpsNoExistNalNum: i32,
    pub iSubSpsNoExistNalNum: i32,
    pub iPpsNoExistNalNum: i32,
    pub uiProfile: u32,
    pub uiLevel: u32,
    pub iCurrentActiveSpsId: i32,
    pub iCurrentActivePpsId: i32,
    pub iStatisticsLogInterval: u32,
}

/// VUI sample aspect ratio metadata (`SVuiSarInfo`).
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SVuiSarInfo {
    pub uiSarWidth: u32,
    pub uiSarHeight: u32,
    pub bOverscanAppropriateFlag: bool,
}

pub type PVuiSarInfo = *mut SVuiSarInfo;

/// Bitrate info per layer (`SBitrateInfo`).
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SBitrateInfo {
    pub iLayer: LAYER_NUM,
    pub iBitrate: i32,
}

pub use crate::encoder::ref_list_mgr_svc::{SLTRMarkingFeedback, SLTRRecoverRequest};
pub use crate::encoder::wels_encoder_ext::{SDeliveryStatus, SLTRConfig, SLevelInfo, SProfileInfo};

/// Logging trace callback prototype (`WelsTraceCallback`) — `codec_api.h:129`.
pub type WelsTraceCallback =
    Option<unsafe extern "C" fn(ctx: *mut c_void, level: i32, string: *const c_char)>;

/// The caller's opaque trace handle. Never dereferenced here; `deliver` hands it
/// straight back to the caller untouched.
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct TraceUserCtx(usize);

impl Default for TraceUserCtx {
    /// No handle installed.
    fn default() -> Self {
        Self(0)
    }
}

impl TraceUserCtx {
    #[inline]
    pub(crate) fn from_abi(p: *mut c_void) -> Self {
        Self(p.expose_provenance())
    }

    #[inline]
    #[allow(unsafe_code)]
    pub(crate) fn deliver(
        self,
        pfLog: unsafe extern "C" fn(ctx: *mut c_void, level: i32, string: *const c_char),
        level: i32,
        line: &std::ffi::CStr,
    ) {
        let ctx: *mut c_void = ptr::with_exposed_provenance_mut(self.0);
        unsafe { pfLog(ctx, level, line.as_ptr()) };
    }
}

/// The default trace sink — `welsCodecTrace.cpp`'s `welsStderrTrace`.
#[allow(unsafe_code)]
pub unsafe extern "C" fn welsStderrTrace(_ctx: *mut c_void, _level: i32, string: *const c_char) {
    if string.is_null() {
        return;
    }
    let s = unsafe { std::ffi::CStr::from_ptr(string) }.to_string_lossy();
    eprintln!("{s}");
}

// ===========================================================================
// Raw C-ABI struct sanitizers
// ===========================================================================

#[cfg(any(feature = "c-api", feature = "cxx-api", test))]
#[inline]
#[allow(unsafe_code, dead_code)]
unsafe fn sanitize_bool_at(p: *mut bool) {
    let raw = unsafe { p.cast::<u8>().read() };
    unsafe { p.cast::<u8>().write(u8::from(raw != 0)) };
}

#[cfg(any(feature = "c-api", feature = "cxx-api", test))]
#[allow(unsafe_code, dead_code)]
pub(crate) unsafe fn read_sanitized_enc_param_base(p: *const SEncParamBase) -> SEncParamBase {
    unsafe {
        let mut buf = std::mem::MaybeUninit::<SEncParamBase>::uninit();
        ptr::copy_nonoverlapping(
            p.cast::<u8>(),
            buf.as_mut_ptr().cast::<u8>(),
            size_of::<SEncParamBase>(),
        );
        let ptr = buf.as_mut_ptr();
        let usage = ptr::addr_of_mut!((*ptr).iUsageType).cast::<i32>();
        usage.write(EUsageType::from_raw(usage.read()) as i32);
        let rc = ptr::addr_of_mut!((*ptr).iRCMode).cast::<i32>();
        rc.write(RC_MODES::from_raw_param(rc.read()) as i32);
        buf.assume_init()
    }
}

#[cfg(any(feature = "c-api", feature = "cxx-api", test))]
#[allow(unsafe_code, dead_code)]
pub(crate) unsafe fn read_sanitized_enc_param_ext(p: *const SEncParamExt) -> SEncParamExt {
    unsafe {
        let mut buf = std::mem::MaybeUninit::<SEncParamExt>::uninit();
        ptr::copy_nonoverlapping(
            p.cast::<u8>(),
            buf.as_mut_ptr().cast::<u8>(),
            size_of::<SEncParamExt>(),
        );
        let ptr = buf.as_mut_ptr();
        let usage = ptr::addr_of_mut!((*ptr).iUsageType).cast::<i32>();
        usage.write(EUsageType::from_raw(usage.read()) as i32);
        let rc = ptr::addr_of_mut!((*ptr).iRCMode).cast::<i32>();
        rc.write(RC_MODES::from_raw_param(rc.read()) as i32);

        let layers_ptr = ptr::addr_of_mut!((*ptr).sSpatialLayers).cast::<SSpatialLayerConfig>();
        for i in 0..MAX_SPATIAL_LAYER_NUM {
            let l = layers_ptr.add(i);
            let prof = ptr::addr_of_mut!((*l).uiProfileIdc).cast::<i32>();
            prof.write(EProfileIdc::from_raw_param(prof.read()) as i32);
            let lvl = ptr::addr_of_mut!((*l).uiLevelIdc).cast::<i32>();
            lvl.write(ELevelIdc::from_raw(lvl.read()) as i32);
            let sm = ptr::addr_of_mut!((*l).sSliceArgument.uiSliceMode).cast::<i32>();
            sm.write(SliceModeEnum::from_raw(sm.read()) as i32);
            sanitize_bool_at(ptr::addr_of_mut!((*l).bVideoSignalTypePresent));
            sanitize_bool_at(ptr::addr_of_mut!((*l).bFullRange));
            sanitize_bool_at(ptr::addr_of_mut!((*l).bColorDescriptionPresent));
            sanitize_bool_at(ptr::addr_of_mut!((*l).bAspectRatioPresent));
            let sar = ptr::addr_of_mut!((*l).eAspectRatio).cast::<i32>();
            sar.write(ESampleAspectRatio::from_raw(sar.read()) as i32);
        }

        let cmplx = ptr::addr_of_mut!((*ptr).iComplexityMode).cast::<i32>();
        cmplx.write(ECOMPLEXITY_MODE::from_raw(cmplx.read()) as i32);
        let strat = ptr::addr_of_mut!((*ptr).eSpsPpsIdStrategy).cast::<i32>();
        strat.write(EParameterSetStrategy::from_raw(strat.read()) as i32);

        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bPrefixNalAddingCtrl));
        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bEnableSSEI));
        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bSimulcastAVC));
        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bEnableFrameSkip));
        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bEnableLongTermReference));
        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bUseLoadBalancing));
        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bEnableDenoise));
        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bEnableBackgroundDetection));
        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bEnableAdaptiveQuant));
        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bEnableFrameCroppingFlag));
        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bEnableSceneChangeDetect));
        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bIsLosslessLink));
        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bFixRCOverShoot));
        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bPsnrY));
        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bPsnrU));
        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bPsnrV));

        buf.assume_init()
    }
}

#[cfg(any(feature = "c-api", feature = "cxx-api"))]
#[allow(unsafe_code)]
pub(crate) unsafe fn read_sanitized_source_picture(p: *const SSourcePicture) -> SSourcePicture {
    unsafe {
        let mut buf = std::mem::MaybeUninit::<SSourcePicture>::uninit();
        ptr::copy_nonoverlapping(
            p.cast::<u8>(),
            buf.as_mut_ptr().cast::<u8>(),
            size_of::<SSourcePicture>(),
        );
        let ptr = buf.as_mut_ptr();
        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bPsnrY));
        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bPsnrU));
        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bPsnrV));
        buf.assume_init()
    }
}

#[cfg(any(feature = "c-api", feature = "cxx-api"))]
#[allow(unsafe_code)]
pub(crate) unsafe fn sanitize_frame_bs_info_in_place(p: *mut SFrameBSInfo) {
    unsafe {
        let ft = ptr::addr_of_mut!((*p).eFrameType).cast::<i32>();
        ft.write(EVideoFrameType::from_raw(ft.read()) as i32);
        let layers = ptr::addr_of_mut!((*p).sLayerInfo).cast::<SLayerBSInfo>();
        for i in 0..MAX_LAYER_NUM_OF_FRAME {
            let lft = ptr::addr_of_mut!((*layers.add(i)).eFrameType).cast::<i32>();
            lft.write(EVideoFrameType::from_raw(lft.read()) as i32);
        }
    }
}

#[cfg(any(feature = "c-api", feature = "cxx-api"))]
#[allow(unsafe_code)]
pub(crate) unsafe fn read_sanitized_decoding_param(p: *const SDecodingParam) -> SDecodingParam {
    unsafe {
        let mut buf = std::mem::MaybeUninit::<SDecodingParam>::uninit();
        ptr::copy_nonoverlapping(
            p.cast::<u8>(),
            buf.as_mut_ptr().cast::<u8>(),
            size_of::<SDecodingParam>(),
        );
        let ptr = buf.as_mut_ptr();
        ptr::addr_of_mut!((*ptr).pFileNameRestructed).write(0);
        let ec = ptr::addr_of_mut!((*ptr).eEcActiveIdc).cast::<i32>();
        ec.write(super::decoder::ec_idc_from_raw(ec.read()) as i32);
        sanitize_bool_at(ptr::addr_of_mut!((*ptr).bParseOnly));
        let bs = ptr::addr_of_mut!((*ptr).sVideoProperty.eVideoBsType).cast::<i32>();
        bs.write(super::decoder::video_bs_type_from_raw(bs.read()) as i32);
        buf.assume_init()
    }
}

/// Reports a caught panic through the trace at `WELS_LOG_ERROR`.
///
/// The message is included when the payload is `&'static str` or `String`; any other
/// payload is named as such.
#[cfg(any(feature = "c-api", feature = "cxx-api"))]
pub(crate) fn report_abi_panic(
    slot: &str,
    payload: Box<dyn std::any::Any + Send>,
    log: Option<crate::common::wels_trace::SLogContext>,
) {
    let what = payload
        .downcast_ref::<&'static str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("<non-string panic payload>");
    // `WelsLog` does nothing when there is no sink, which is the whole of the `None`
    // case.
    crate::common::wels_trace::WelsLog(
        log.unwrap_or_default(),
        crate::common::wels_trace::WELS_LOG_ERROR,
        &format!(
            "{slot}: a panic was caught at the C-ABI boundary and reported as a failure code instead of aborting the process. Panic message: {what}"
        ),
    );
}

/// One `catch_unwind` window per `extern "C"` / FFI entry point.
///
/// `$slot` names the entry for the log, `$log` is its trace log context read,
/// `$fail` is the code this slot returns when it cannot do its job, and `$body`
/// is the slot's own body.
///
/// Every `return` inside `$body` returns from the closure, so it is the value of the
/// whole expression.
#[cfg(any(feature = "c-api", feature = "cxx-api"))]
macro_rules! abi_guard {
    ($slot:literal, $log:expr, $fail:expr, $body:block) => {{
        let __log = $log;
        match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(move || $body)) {
            Ok(v) => v,
            Err(payload) => {
                $crate::api::types::report_abi_panic($slot, payload, __log);
                $fail
            }
        }
    }};
}
#[cfg(any(feature = "c-api", feature = "cxx-api"))]
pub(crate) use abi_guard;
