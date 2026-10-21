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

//! Safe Rust H.264 / SVC Encoder core (`Encoder`).
//!
//! Wraps [`crate::encoder::wels_encoder_ext::CWelsH264SVCEncoder`] without C ABI vtables
//! or raw pointers on the main encoding path. Used by both the `cxx` C++ API (`cxx_api`)
//! and the raw C ABI (`c_api`).

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]
#![deny(unsafe_code)]

use std::ffi::c_void;

#[cfg(any(feature = "c-api", feature = "cxx-api", test))]
use super::types::ENCODER_OPTION;
use super::types::{
    EComplexityMode, EParameterSetStrategy, EVideoFormatType, LAYER_NUM, RC_MODES, SBitrateInfo,
    SDeliveryStatus, SEncParamBase, SEncParamExt, SEncoderStatistics, SFrameBSInfo, SLTRConfig,
    SLTRMarkingFeedback, SLTRRecoverRequest, SLevelInfo, SProfileInfo, SourcePicture, TraceUserCtx,
    WelsTraceCallback,
};

/// The H.264 encoder, as a Rust type.
///
/// Wraps `CWelsH264SVCEncoder`. Every method is safe except the trace callback
/// and context setters, which cross an FFI lifetime boundary.
pub struct Encoder(pub(crate) crate::encoder::wels_encoder_ext::CWelsH264SVCEncoder);

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    /// Creates a new uninitialized [`Encoder`].
    pub fn new() -> Self {
        Self(crate::encoder::wels_encoder_ext::CWelsH264SVCEncoder::new())
    }

    /// Initializes the encoder with basic parameters (`SEncParamBase`).
    ///
    /// Returns `CM_RESULT_SUCCESS` or one of the `CM_RETURN` error codes.
    pub fn initialize(&mut self, param: &SEncParamBase) -> i32 {
        self.0.Initialize(Some(param))
    }

    /// Initializes the encoder with extended SVC parameters (`SEncParamExt`).
    ///
    /// Returns `CM_RESULT_SUCCESS` or one of the `CM_RETURN` error codes.
    pub fn initialize_ext(&mut self, param: &SEncParamExt) -> i32 {
        self.0.InitializeExt(Some(param))
    }

    /// Fills `param` with the encoder's defaults. Every field is written.
    pub fn default_params(&self, param: &mut SEncParamExt) -> i32 {
        self.0.GetDefaultParams(param)
    }

    /// Uninitializes the encoder and releases its internal context.
    pub fn uninitialize(&mut self) -> i32 {
        self.0.Uninitialize()
    }

    /// Encodes one frame.
    ///
    /// `src`'s `pData` plane slices are borrowed for this call only.
    /// On success `bs` describes layers inside this encoder's internal output
    /// buffer, readable via [`SLayerBSInfo::bs_buf`], [`SLayerBSInfo::nal_lengths`],
    /// and [`SLayerBSInfo::nals`].
    ///
    /// # Output Window
    ///
    /// The internal bitstream and NAL-length buffers are reused on the next
    /// [`Self::encode_frame`], [`Self::encode_parameter_sets`], [`Self::uninitialize`],
    /// or re-initialization call. A stale `SFrameBSInfo` saved across a subsequent
    /// encode call will return empty slices when its layer count or NAL count no
    /// longer matches the latest frame, or will read the newly overwritten buffer
    /// contents if the new frame happens to have the same layer/NAL layout.
    pub fn encode_frame(&mut self, src: &SourcePicture<'_>, bs: &mut SFrameBSInfo) -> i32 {
        self.0.EncodeFrame(src, bs)
    }

    /// Emits SPS/PPS into `bs`, with the same single-call output window as
    /// [`Self::encode_frame`].
    pub fn encode_parameter_sets(&mut self, bs: &mut SFrameBSInfo) -> i32 {
        self.0.EncodeParameterSets(bs)
    }

    /// Forces the next encoded frame on `layer_id` (or all layers) to be an IDR
    /// frame when `idr` is `true`. Returns `0` on success or `1` if uninitialized.
    pub fn force_intra_frame(&mut self, idr: bool, layer_id: i32) -> i32 {
        self.0.ForceIntraFrame(idr, layer_id)
    }

    /// The trace destination, as three typed setters rather than an option blob.
    pub fn set_trace_level(&mut self, level: u32) {
        self.0.m_pWelsTrace.SetTraceLevel(level);
        self.0.sync_log_ctx();
    }

    /// # Safety
    ///
    /// `callback` is entered on every delivered message until it is replaced or
    /// this encoder is dropped, with the context installed beside it by
    /// [`Self::set_trace_callback_context`]; it must be sound to enter for that
    /// whole window. Neither half can carry a lifetime across the C ABI.
    #[allow(unsafe_code)]
    pub unsafe fn set_trace_callback(&mut self, callback: WelsTraceCallback) {
        self.0.m_pWelsTrace.SetTraceCallback(callback);
        self.0.sync_log_ctx();
    }

    /// # Safety
    ///
    /// `ctx` is handed back to the trace callback on every message until it is
    /// replaced or this encoder is dropped, so it must stay valid for that long.
    /// It is the caller's, and this crate never dereferences it.
    #[allow(unsafe_code)]
    pub unsafe fn set_trace_callback_context(&mut self, ctx: *mut c_void) {
        self.0
            .m_pWelsTrace
            .SetTraceCallbackContext(TraceUserCtx::from_abi(ctx));
        self.0.sync_log_ctx();
    }

    /// Returns `true` when the encoder has been initialized with an active context.
    pub fn has_ctx(&self) -> bool {
        self.0.has_ctx()
    }

    /// Queries `ENCODER_OPTION_DATAFORMAT`, or `None` if uninitialized.
    pub fn data_format(&self) -> Option<i32> {
        self.0.data_format()
    }

    /// Sets `ENCODER_OPTION_DATAFORMAT`. Returns `CM_RESULT_SUCCESS`,
    /// `CM_INIT_EXPECTED`, or `CM_INIT_PARA_ERROR`.
    pub fn set_data_format(&mut self, format: EVideoFormatType) -> i32 {
        self.0.set_data_format(format)
    }

    /// Queries `ENCODER_OPTION_IDR_INTERVAL` (intra period), or `None` if uninitialized.
    pub fn idr_interval(&self) -> Option<i32> {
        self.0.idr_interval()
    }

    /// Sets `ENCODER_OPTION_IDR_INTERVAL` (negative values clamp to `0`).
    /// Returns `CM_RESULT_SUCCESS` or `CM_INIT_EXPECTED`.
    pub fn set_idr_interval(&mut self, interval: i32) -> i32 {
        self.0.set_idr_interval(interval)
    }

    /// Queries `ENCODER_OPTION_SVC_ENCODE_PARAM_BASE`, or `None` if uninitialized.
    pub fn encode_param_base(&self) -> Option<SEncParamBase> {
        self.0.encode_param_base()
    }

    /// Reconfigures the encoder via `ENCODER_OPTION_SVC_ENCODE_PARAM_BASE`.
    /// Returns `CM_RESULT_SUCCESS`, `CM_INIT_EXPECTED`, or `CM_INIT_PARA_ERROR`.
    pub fn set_encode_param_base(&mut self, param: &SEncParamBase) -> i32 {
        self.0.set_encode_param_base(param)
    }

    /// Queries `ENCODER_OPTION_SVC_ENCODE_PARAM_EXT`, or `None` if uninitialized.
    pub fn encode_param_ext(&self) -> Option<SEncParamExt> {
        self.0.encode_param_ext()
    }

    /// Reconfigures the encoder via `ENCODER_OPTION_SVC_ENCODE_PARAM_EXT`.
    /// Returns `CM_RESULT_SUCCESS`, `CM_INIT_EXPECTED`, or `CM_INIT_PARA_ERROR`.
    pub fn set_encode_param_ext(&mut self, param: &SEncParamExt) -> i32 {
        self.0.set_encode_param_ext(param)
    }

    /// Queries `ENCODER_OPTION_FRAME_RATE` (`fMaxFrameRate`), or `None` if uninitialized.
    pub fn frame_rate(&self) -> Option<f32> {
        self.0.frame_rate()
    }

    /// Sets `ENCODER_OPTION_FRAME_RATE` (must be `> 0.0`, clamped to `[1.0, 60.0]`).
    /// Returns `CM_RESULT_SUCCESS`, `CM_INIT_EXPECTED`, or `CM_INIT_PARA_ERROR`.
    pub fn set_frame_rate(&mut self, fps: f32) -> i32 {
        self.0.set_frame_rate(fps)
    }

    /// Queries `ENCODER_OPTION_BITRATE` for `layer`, or `None` if uninitialized.
    pub fn bitrate(&self, layer: LAYER_NUM) -> Option<i32> {
        self.0.bitrate(layer)
    }

    /// Sets `ENCODER_OPTION_BITRATE`. Returns `CM_RESULT_SUCCESS`,
    /// `CM_INIT_EXPECTED`, or `CM_INIT_PARA_ERROR`.
    pub fn set_bitrate(&mut self, info: &SBitrateInfo) -> i32 {
        self.0.set_bitrate(info)
    }

    /// Queries `ENCODER_OPTION_MAX_BITRATE` for `layer`, or `None` if uninitialized.
    pub fn max_bitrate(&self, layer: LAYER_NUM) -> Option<i32> {
        self.0.max_bitrate(layer)
    }

    /// Sets `ENCODER_OPTION_MAX_BITRATE`. Returns `CM_RESULT_SUCCESS`,
    /// `CM_INIT_EXPECTED`, or `CM_INIT_PARA_ERROR`.
    pub fn set_max_bitrate(&mut self, info: &SBitrateInfo) -> i32 {
        self.0.set_max_bitrate(info)
    }

    /// Sets `ENCODER_OPTION_RC_MODE` and updates rate-control dispatch functions.
    /// Returns `CM_RESULT_SUCCESS` or `CM_INIT_EXPECTED`.
    pub fn set_rc_mode(&mut self, mode: RC_MODES) -> i32 {
        self.0.set_rc_mode(mode)
    }

    /// Sets `ENCODER_OPTION_RC_FRAME_SKIP` (ignored when rate control is `RC_OFF_MODE`).
    /// Returns `CM_RESULT_SUCCESS` or `CM_INIT_EXPECTED`.
    pub fn set_rc_frame_skip(&mut self, enable: bool) -> i32 {
        self.0.set_rc_frame_skip(enable)
    }

    /// Sets `ENCODER_PADDING_PADDING` (`iPaddingFlag`).
    /// Returns `CM_RESULT_SUCCESS` or `CM_INIT_EXPECTED`.
    pub fn set_padding(&mut self, padding: i32) -> i32 {
        self.0.set_padding(padding)
    }

    /// Submits a long-term reference recovery request (`ENCODER_LTR_RECOVERY_REQUEST`).
    /// Returns `CM_RESULT_SUCCESS` or `CM_INIT_EXPECTED`.
    pub fn set_ltr_recovery_request(&mut self, req: &mut SLTRRecoverRequest) -> i32 {
        self.0.set_ltr_recovery_request(req)
    }

    /// Submits long-term reference marking feedback (`ENCODER_LTR_MARKING_FEEDBACK`).
    /// Returns `CM_RESULT_SUCCESS` or `CM_INIT_EXPECTED`.
    pub fn set_ltr_marking_feedback(&mut self, fb: &mut SLTRMarkingFeedback) -> i32 {
        self.0.set_ltr_marking_feedback(fb)
    }

    /// Sets `ENCODER_LTR_MARKING_PERIOD` (`iLtrMarkPeriod`).
    /// Returns `CM_RESULT_SUCCESS` or `CM_INIT_EXPECTED`.
    pub fn set_ltr_marking_period(&mut self, period: u32) -> i32 {
        self.0.set_ltr_marking_period(period)
    }

    /// Sets `ENCODER_OPTION_LTR` (`SLTRConfig`).
    /// Returns `CM_RESULT_SUCCESS`, `CM_INIT_EXPECTED`, or `CM_INIT_PARA_ERROR`.
    pub fn set_ltr_config(&mut self, cfg: &SLTRConfig) -> i32 {
        self.0.set_ltr_config(cfg)
    }

    /// Sets `ENCODER_OPTION_ENABLE_SSEI`.
    /// Returns `CM_RESULT_SUCCESS` or `CM_INIT_EXPECTED`.
    pub fn set_enable_ssei(&mut self, enable: bool) -> i32 {
        self.0.set_enable_ssei(enable)
    }

    /// Sets `ENCODER_OPTION_ENABLE_PREFIX_NAL_ADDING`.
    /// Returns `CM_RESULT_SUCCESS` or `CM_INIT_EXPECTED`.
    pub fn set_enable_prefix_nal_adding(&mut self, enable: bool) -> i32 {
        self.0.set_enable_prefix_nal_adding(enable)
    }

    /// Sets `ENCODER_OPTION_SPS_PPS_ID_STRATEGY`. Switching into or out of
    /// `SPS_LISTING`-based strategies mid-stream returns `CM_INIT_PARA_ERROR`.
    pub fn set_sps_pps_id_strategy(&mut self, strategy: EParameterSetStrategy) -> i32 {
        self.0.set_sps_pps_id_strategy(strategy)
    }

    /// Sets `ENCODER_OPTION_PROFILE` for a spatial layer (`SPATIAL_LAYER_0..=SPATIAL_LAYER_3`).
    /// Returns `CM_RESULT_SUCCESS`, `CM_INIT_EXPECTED`, or `CM_INIT_PARA_ERROR`.
    pub fn set_profile(&mut self, info: &SProfileInfo) -> i32 {
        self.0.set_profile(info)
    }

    /// Sets `ENCODER_OPTION_LEVEL` for a spatial layer (`SPATIAL_LAYER_0..=SPATIAL_LAYER_3`).
    /// Returns `CM_RESULT_SUCCESS`, `CM_INIT_EXPECTED`, or `CM_INIT_PARA_ERROR`.
    pub fn set_level(&mut self, info: &SLevelInfo) -> i32 {
        self.0.set_level(info)
    }

    /// Sets `ENCODER_OPTION_NUMBER_REF` (`iNumRefFrame`).
    /// Returns `CM_RESULT_SUCCESS` or `CM_INIT_EXPECTED`.
    pub fn set_number_ref(&mut self, num_ref: i32) -> i32 {
        self.0.set_number_ref(num_ref)
    }

    /// Sets `ENCODER_OPTION_DELIVERY_STATUS` (`bDeliveryFlag`).
    /// Returns `CM_RESULT_SUCCESS` or `CM_INIT_EXPECTED`.
    pub fn set_delivery_status(&mut self, status: &SDeliveryStatus) -> i32 {
        self.0.set_delivery_status(status)
    }

    /// Queries `ENCODER_OPTION_COMPLEXITY`, or `None` if uninitialized.
    pub fn complexity(&self) -> Option<EComplexityMode> {
        self.0.complexity()
    }

    /// Sets `ENCODER_OPTION_COMPLEXITY`.
    /// Returns `CM_RESULT_SUCCESS` or `CM_INIT_EXPECTED`.
    pub fn set_complexity(&mut self, mode: EComplexityMode) -> i32 {
        self.0.set_complexity(mode)
    }

    /// Queries `ENCODER_OPTION_GET_STATISTICS` for the highest active spatial layer,
    /// or `None` if uninitialized.
    pub fn statistics(&self) -> Option<SEncoderStatistics> {
        self.0.statistics()
    }

    /// Queries `ENCODER_OPTION_STATISTICS_LOG_INTERVAL`, or `None` if uninitialized.
    pub fn statistics_log_interval(&self) -> Option<i32> {
        self.0.statistics_log_interval()
    }

    /// Sets `ENCODER_OPTION_STATISTICS_LOG_INTERVAL`.
    /// Returns `CM_RESULT_SUCCESS` or `CM_INIT_EXPECTED`.
    pub fn set_statistics_log_interval(&mut self, interval: i32) -> i32 {
        self.0.set_statistics_log_interval(interval)
    }

    /// Sets `ENCODER_OPTION_IS_LOSSLESS_LINK`.
    /// Returns `CM_RESULT_SUCCESS` or `CM_INIT_EXPECTED`.
    pub fn set_is_lossless_link(&mut self, lossless: bool) -> i32 {
        self.0.set_is_lossless_link(lossless)
    }

    /// Sets `ENCODER_OPTION_BITS_VARY_PERCENTAGE` (clamped to `0..=100`).
    /// Returns `CM_RESULT_SUCCESS` or `CM_INIT_EXPECTED`.
    pub fn set_bits_vary_percentage(&mut self, pct: i32) -> i32 {
        self.0.set_bits_vary_percentage(pct)
    }

    #[allow(unsafe_code)]
    pub(crate) fn slice_layer_nal_lengths(&self, ptr: *const i32, count: usize) -> &[i32] {
        if !self.0.m_bInitialFlag || ptr.is_null() || count == 0 {
            return &[];
        }
        let active_layers = &self.0.m_lastBsLayers[..self.0.m_lastBsLayerNum];
        if !active_layers
            .iter()
            .any(|&(nal_ptr, nal_count, _)| nal_ptr == ptr as usize && nal_count == count)
        {
            return &[];
        }
        let Some(ctx) = self.0.m_pEncContext.as_deref() else {
            return &[];
        };
        let Some(out) = ctx.out_opt() else {
            return &[];
        };
        let nal_lens = out.sNalLen.as_slice();
        if nal_lens.is_empty() {
            return &[];
        }
        let Some(byte_off) = (ptr as usize).checked_sub(nal_lens.as_ptr() as usize) else {
            return &[];
        };
        if byte_off % size_of::<i32>() != 0 {
            return &[];
        }
        let elem_off = byte_off / size_of::<i32>();
        let Some(end) = elem_off.checked_add(count) else {
            return &[];
        };
        let Some(sub) = nal_lens.get(elem_off..end) else {
            return &[];
        };
        // SAFETY: `sub` is an in-bounds `&[AtomicI32]` slice borrowed from `&self`.
        // `AtomicI32` has the same size, alignment, and bit validity as `i32`, and
        // `&self` (`&Encoder`) prevents any concurrent or re-entrant mutation of
        // `out.sNalLen` for the lifetime of the returned slice.
        unsafe { std::slice::from_raw_parts(sub.as_ptr().cast::<i32>(), sub.len()) }
    }

    pub(crate) fn slice_layer_bs_buf(&self, ptr: *const u8, len: usize) -> &[u8] {
        if !self.0.m_bInitialFlag || ptr.is_null() || len == 0 {
            return &[];
        }
        let active_layers = &self.0.m_lastBsLayers[..self.0.m_lastBsLayerNum];
        if !active_layers
            .iter()
            .any(|&(_, _, bs_ptr)| bs_ptr == ptr as usize)
        {
            return &[];
        }
        let Some(ctx) = self.0.m_pEncContext.as_deref() else {
            return &[];
        };
        let frame_bs = ctx.pFrameBs.as_slice();
        if frame_bs.is_empty() {
            return &[];
        }
        let Some(off) = (ptr as usize).checked_sub(frame_bs.as_ptr() as usize) else {
            return &[];
        };
        let Some(end) = off.checked_add(len) else {
            return &[];
        };
        frame_bs.get(off..end).unwrap_or(&[])
    }

    /// `ENCODER_OPTION_*`, the type-erased pair.
    ///
    /// # Safety
    ///
    /// `option` must point at a readable, aligned object of the type `id` names,
    /// for the duration of the call.
    #[cfg(any(feature = "c-api", feature = "cxx-api", test))]
    #[allow(unsafe_code, dead_code)]
    pub(crate) unsafe fn set_option_raw(&mut self, id: ENCODER_OPTION, option: *mut c_void) -> i32 {
        unsafe { self.0.SetOptionRaw(id as i32, option) }
    }

    /// # Safety
    ///
    /// As [`Self::set_option_raw`], with `option` written.
    #[cfg(any(feature = "c-api", feature = "cxx-api", test))]
    #[allow(unsafe_code, dead_code)]
    pub(crate) unsafe fn get_option_raw(&mut self, id: ENCODER_OPTION, option: *mut c_void) -> i32 {
        unsafe { self.0.GetOptionRaw(id as i32, option) }
    }
}
