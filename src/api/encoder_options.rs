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

//! `CWelsH264SVCEncoder::SetOption` / `GetOption` — the encoder's untyped
//! `void* pOption` boundary. `pOption`'s real type is named by `eOptionId`.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]
#![deny(unsafe_code)]

#[cfg(any(feature = "c-api", feature = "cxx-api", test))]
use std::ffi::c_void;
#[cfg(any(feature = "c-api", feature = "cxx-api", test))]
use std::ptr;

use crate::api::codec_api::EParameterSetStrategy::*;
use crate::api::codec_api::LAYER_NUM::*;
use crate::api::codec_api::RC_MODES::*;
#[cfg(any(feature = "c-api", feature = "cxx-api", test))]
use crate::api::codec_api::TraceUserCtx;
use crate::api::codec_api::{
    EComplexityMode, EParameterSetStrategy, EVideoFormatType, LAYER_NUM, RC_MODES, SBitrateInfo,
    SEncParamBase, SEncParamExt,
};
#[cfg(any(feature = "c-api", feature = "cxx-api", test))]
use crate::api::codec_api::{ELevelIdc, EProfileIdc, EncoderOption};
#[cfg(any(feature = "c-api", feature = "cxx-api", test))]
use crate::api::types::{read_sanitized_enc_param_base, read_sanitized_enc_param_ext};
#[cfg(any(feature = "c-api", feature = "cxx-api", test))]
use crate::common::wels_trace::WelsTraceCallback;
use crate::common::wels_trace::{WELS_LOG_INFO, WelsLog};
use crate::encoder::encoder_context::sWelsEncCtx;
use crate::encoder::param_svc::{MAX_SPATIAL_LAYER_NUM, SWelsSvcCodingParam, WELS_CLIP3};
use crate::encoder::rc::WelsRcInitFuncPointers;
use crate::encoder::ref_list_mgr_svc::{
    FilterLTRMarkingFeedback, FilterLTRRecoveryRequest, SLTRMarkingFeedback, SLTRRecoverRequest,
};
#[cfg(any(feature = "c-api", feature = "cxx-api", test))]
use crate::encoder::wels_encoder_ext::rc_mode_from_raw;
use crate::encoder::wels_encoder_ext::{
    CWelsH264SVCEncoder, CheckLevelSetting, CheckProfileSetting, CheckReferenceNumSetting,
    MAX_BIT_RATE, MAX_FRAME_RATE, MIN_BIT_RATE, MIN_FRAME_RATE, SDeliveryStatus, SLTRConfig,
    SLevelInfo, SProfileInfo, WelsEncoderApplyBitRate, WelsEncoderApplyBitVaryRang,
    WelsEncoderApplyFrameRate, WelsEncoderApplyLTR, WelsEncoderParamAdjust, cmInitExpected,
    cmInitParaError, cmResultSuccess,
};

impl CWelsH264SVCEncoder {
    pub fn has_ctx(&self) -> bool {
        self.m_pEncContext.is_some() && self.m_bInitialFlag
    }

    #[inline]
    fn active_ctx(&self) -> Option<&sWelsEncCtx> {
        if !self.m_bInitialFlag {
            return None;
        }
        self.m_pEncContext.as_deref()
    }

    #[inline]
    fn active_ctx_mut(&mut self) -> Option<&mut sWelsEncCtx> {
        if !self.m_bInitialFlag {
            return None;
        }
        self.m_pEncContext.as_deref_mut()
    }

    pub fn data_format(&self) -> Option<i32> {
        if !self.has_ctx() {
            return None;
        }
        Some(self.m_iCspInternal)
    }

    pub fn set_data_format(&mut self, format: EVideoFormatType) -> i32 {
        self.set_data_format_raw(format as i32)
    }

    pub(crate) fn set_data_format_raw(&mut self, format: i32) -> i32 {
        if !self.has_ctx() {
            return cmInitExpected;
        }
        if format == 0 {
            return cmInitParaError;
        }
        self.m_iCspInternal = format;
        cmResultSuccess
    }

    pub fn idr_interval(&self) -> Option<i32> {
        self.active_ctx()
            .map(|ctx| ctx.param().uiIntraPeriod as i32)
    }

    pub fn set_idr_interval(&mut self, mut interval: i32) -> i32 {
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        if interval <= -1 {
            interval = 0;
        }
        if interval == ctx.param().uiIntraPeriod as i32 {
            return cmResultSuccess;
        }
        ctx.param_mut().uiIntraPeriod = interval as u32;
        cmResultSuccess
    }

    pub fn encode_param_base(&self) -> Option<SEncParamBase> {
        let pCtx = self.active_ctx()?;
        let mut out = SEncParamBase::default();
        pCtx.param().GetBaseParams(&mut out);
        Some(out)
    }

    pub fn set_encode_param_base(&mut self, sEncodingParam: &SEncParamBase) -> i32 {
        if !self.has_ctx() {
            return cmInitExpected;
        }
        let mut sConfig = SWelsSvcCodingParam::default();
        if sConfig.ParamBaseTranscode(sEncodingParam) != 0 {
            return cmInitParaError;
        }
        let iTargetWidth = sConfig.iPicWidth;
        let iTargetHeight = sConfig.iPicHeight;
        if self.m_iMaxPicWidth != iTargetWidth || self.m_iMaxPicHeight != iTargetHeight {
            self.m_iMaxPicWidth = iTargetWidth;
            self.m_iMaxPicHeight = iTargetHeight;
        }
        if sConfig.DetermineTemporalSettings() != 0 {
            return cmInitParaError;
        }
        if WelsEncoderParamAdjust(&mut self.m_pEncContext, &mut sConfig) != 0 {
            return cmInitParaError;
        }
        // `WelsEncoderParamAdjust` may replace the context, so the
        // timestamp is copied out before the `&mut self` logging calls.
        let ts = match self.m_pEncContext.as_deref() {
            Some(ctx) => ctx.iLastStatisticsLogTs,
            None => return cmInitExpected,
        };
        self.LogStatistics(ts, 0);
        cmResultSuccess
    }

    pub fn encode_param_ext(&self) -> Option<SEncParamExt> {
        Some(self.active_ctx()?.param().to_param_ext())
    }

    pub fn set_encode_param_ext(&mut self, sEncodingParam: &SEncParamExt) -> i32 {
        if !self.has_ctx() {
            return cmInitExpected;
        }
        // Traced before the spatial-layer check, so rejected parameters
        // are still echoed.
        self.TraceParamInfo(sEncodingParam);
        if sEncodingParam.iSpatialLayerNum < 1
            || sEncodingParam.iSpatialLayerNum > MAX_SPATIAL_LAYER_NUM as i32
        {
            return cmInitParaError;
        }
        let mut sConfig = SWelsSvcCodingParam::default();
        if sConfig.ParamTranscode(sEncodingParam) != 0 {
            return cmInitParaError;
        }
        if sConfig.iSpatialLayerNum < 1 {
            return cmInitParaError;
        }
        if sConfig.DetermineTemporalSettings() != 0 {
            return cmInitParaError;
        }
        let iTargetWidth = sConfig.iPicWidth;
        let iTargetHeight = sConfig.iPicHeight;
        if self.m_iMaxPicWidth != iTargetWidth || self.m_iMaxPicHeight != iTargetHeight {
            self.m_iMaxPicWidth = iTargetWidth;
            self.m_iMaxPicHeight = iTargetHeight;
        }
        if WelsEncoderParamAdjust(&mut self.m_pEncContext, &mut sConfig) != 0 {
            return cmInitParaError;
        }
        // `WelsEncoderParamAdjust` may replace the context, so the
        // timestamp is copied out before the `&mut self` logging calls.
        let ts = match self.m_pEncContext.as_deref() {
            Some(ctx) => ctx.iLastStatisticsLogTs,
            None => return cmInitExpected,
        };
        WelsLog(
            self.log_ctx(),
            WELS_LOG_INFO,
            "CWelsH264SVCEncoder::SetOption():ENCODER_OPTION_SVC_ENCODE_PARAM_EXT, LogStatisticsBeforeNewEncoding",
        );
        self.LogStatistics(ts, sEncodingParam.iSpatialLayerNum - 1);
        cmResultSuccess
    }

    pub fn frame_rate(&self) -> Option<f32> {
        self.active_ctx().map(|ctx| ctx.param().fMaxFrameRate)
    }

    pub fn set_frame_rate(&mut self, fps: f32) -> i32 {
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        if fps <= 0.0 {
            return cmInitParaError;
        }
        ctx.param_mut().fMaxFrameRate = WELS_CLIP3(fps, MIN_FRAME_RATE, MAX_FRAME_RATE);
        WelsEncoderApplyFrameRate(ctx.param_mut());
        cmResultSuccess
    }

    pub fn bitrate(&self, layer: LAYER_NUM) -> Option<i32> {
        let pCtx = self.active_ctx()?;
        Some(match layer {
            SPATIAL_LAYER_ALL => pCtx.param().iTargetBitrate,
            SPATIAL_LAYER_0 | SPATIAL_LAYER_1 | SPATIAL_LAYER_2 | SPATIAL_LAYER_3 => {
                pCtx.param().sSpatialLayers[layer as usize].iSpatialBitrate
            }
        })
    }

    pub fn set_bitrate(&mut self, info: &SBitrateInfo) -> i32 {
        let log_ctx = self.m_pWelsTrace.m_sLogCtx;
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        if info.iBitrate <= 0 {
            return cmInitParaError;
        }
        let iBitrate = WELS_CLIP3(info.iBitrate, MIN_BIT_RATE, MAX_BIT_RATE);
        match info.iLayer {
            SPATIAL_LAYER_ALL => {
                ctx.param_mut().iTargetBitrate = iBitrate;
            }
            SPATIAL_LAYER_0 | SPATIAL_LAYER_1 | SPATIAL_LAYER_2 | SPATIAL_LAYER_3 => {
                ctx.param_mut().sSpatialLayers[info.iLayer as usize].iSpatialBitrate = iBitrate;
            }
        }
        if WelsEncoderApplyBitRate(log_ctx, ctx.param_mut(), info.iLayer as i32) != 0 {
            return cmInitParaError;
        }
        cmResultSuccess
    }

    pub fn max_bitrate(&self, layer: LAYER_NUM) -> Option<i32> {
        let pCtx = self.active_ctx()?;
        Some(match layer {
            SPATIAL_LAYER_ALL => pCtx.param().iMaxBitrate,
            SPATIAL_LAYER_0 | SPATIAL_LAYER_1 | SPATIAL_LAYER_2 | SPATIAL_LAYER_3 => {
                pCtx.param().sSpatialLayers[layer as usize].iMaxSpatialBitrate
            }
        })
    }

    pub fn set_max_bitrate(&mut self, info: &SBitrateInfo) -> i32 {
        let log_ctx = self.m_pWelsTrace.m_sLogCtx;
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        if info.iBitrate <= 0 {
            return cmInitParaError;
        }
        let iBitrate = WELS_CLIP3(info.iBitrate, MIN_BIT_RATE, MAX_BIT_RATE);
        match info.iLayer {
            SPATIAL_LAYER_ALL => {
                ctx.param_mut().iMaxBitrate = iBitrate;
            }
            SPATIAL_LAYER_0 | SPATIAL_LAYER_1 | SPATIAL_LAYER_2 | SPATIAL_LAYER_3 => {
                ctx.param_mut().sSpatialLayers[info.iLayer as usize].iMaxSpatialBitrate = iBitrate;
            }
        }
        if WelsEncoderApplyBitRate(log_ctx, ctx.param_mut(), info.iLayer as i32) != 0 {
            return cmInitParaError;
        }
        cmResultSuccess
    }

    pub fn set_rc_mode(&mut self, mode: RC_MODES) -> i32 {
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        ctx.param_mut().iRCMode = mode;
        WelsRcInitFuncPointers(&mut ctx.func_list_mut().pfRc, mode);
        cmResultSuccess
    }

    pub fn set_rc_frame_skip(&mut self, enable: bool) -> i32 {
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        if ctx.param().iRCMode != RC_OFF_MODE {
            ctx.param_mut().bEnableFrameSkip = enable;
        }
        cmResultSuccess
    }

    pub fn set_padding(&mut self, padding: i32) -> i32 {
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        ctx.param_mut().iPaddingFlag = padding;
        cmResultSuccess
    }

    pub fn set_ltr_recovery_request(&mut self, req: &mut SLTRRecoverRequest) -> i32 {
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        FilterLTRRecoveryRequest(ctx, req);
        cmResultSuccess
    }

    pub fn set_ltr_marking_feedback(&mut self, fb: &mut SLTRMarkingFeedback) -> i32 {
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        FilterLTRMarkingFeedback(ctx, fb);
        cmResultSuccess
    }

    pub fn set_ltr_marking_period(&mut self, period: u32) -> i32 {
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        ctx.param_mut().iLtrMarkPeriod = period;
        cmResultSuccess
    }

    pub fn set_ltr_config(&mut self, cfg: &SLTRConfig) -> i32 {
        if !self.has_ctx() {
            return cmInitExpected;
        }
        let mut ltr_val = *cfg;
        let log_ctx = self.m_pWelsTrace.m_sLogCtx;
        if WelsEncoderApplyLTR(log_ctx, &mut self.m_pEncContext, &mut ltr_val) != 0 {
            return cmInitParaError;
        }
        cmResultSuccess
    }

    pub fn set_enable_ssei(&mut self, enable: bool) -> i32 {
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        ctx.param_mut().bEnableSSEI = enable;
        cmResultSuccess
    }

    pub fn set_enable_prefix_nal_adding(&mut self, enable: bool) -> i32 {
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        ctx.param_mut().bPrefixNalAddingCtrl = enable;
        cmResultSuccess
    }

    pub fn set_sps_pps_id_strategy(&mut self, strategy: EParameterSetStrategy) -> i32 {
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        let eOld = ctx.param().eSpsPpsIdStrategy;
        if ((strategy as i32 & SPS_LISTING as i32) != 0 || (eOld as i32 & SPS_LISTING as i32) != 0)
            && eOld != strategy
        {
            // changing in the middle of call is NOT allowed for
            // eSpsPpsIdStrategy > INCREASING_ID
            return cmInitParaError;
        }
        let mut sConfig: SWelsSvcCodingParam = *ctx.param();
        sConfig.eSpsPpsIdStrategy = strategy;

        if WelsEncoderParamAdjust(&mut self.m_pEncContext, &mut sConfig) != 0 {
            return cmInitParaError;
        }
        cmResultSuccess
    }

    pub fn set_profile(&mut self, info: &SProfileInfo) -> i32 {
        let log_ctx = self.m_pWelsTrace.m_sLogCtx;
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        if info.iLayer < SPATIAL_LAYER_0 as i32 || info.iLayer > SPATIAL_LAYER_3 as i32 {
            return cmInitParaError;
        }
        CheckProfileSetting(log_ctx, ctx.param_mut(), info.iLayer, info.uiProfileIdc);
        cmResultSuccess
    }

    pub fn set_level(&mut self, info: &SLevelInfo) -> i32 {
        let log_ctx = self.m_pWelsTrace.m_sLogCtx;
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        if info.iLayer < SPATIAL_LAYER_0 as i32 || info.iLayer > SPATIAL_LAYER_3 as i32 {
            return cmInitParaError;
        }
        CheckLevelSetting(log_ctx, ctx.param_mut(), info.iLayer, info.uiLevelIdc);
        cmResultSuccess
    }

    pub fn set_number_ref(&mut self, num_ref: i32) -> i32 {
        let log_ctx = self.m_pWelsTrace.m_sLogCtx;
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        CheckReferenceNumSetting(log_ctx, ctx.param_mut(), num_ref);
        cmResultSuccess
    }

    pub fn set_delivery_status(&mut self, status: &SDeliveryStatus) -> i32 {
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        ctx.bDeliveryFlag = status.bDeliveryFlag;
        cmResultSuccess
    }

    pub fn complexity(&self) -> Option<EComplexityMode> {
        self.active_ctx().map(|ctx| ctx.param().iComplexityMode)
    }

    pub fn set_complexity(&mut self, mode: EComplexityMode) -> i32 {
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        ctx.param_mut().iComplexityMode = mode;
        cmResultSuccess
    }

    pub fn statistics(&self) -> Option<crate::SEncoderStatistics> {
        let pCtx = self.active_ctx()?;
        let iLayerIdx = (pCtx.param().iSpatialLayerNum - 1) as usize;
        Some(pCtx.sEncoderStatistics[iLayerIdx])
    }

    pub fn statistics_log_interval(&self) -> Option<i32> {
        self.active_ctx().map(|ctx| ctx.iStatisticsLogInterval)
    }

    pub fn set_statistics_log_interval(&mut self, interval: i32) -> i32 {
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        ctx.iStatisticsLogInterval = interval;
        cmResultSuccess
    }

    pub fn set_is_lossless_link(&mut self, lossless: bool) -> i32 {
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        ctx.param_mut().bIsLosslessLink = lossless;
        cmResultSuccess
    }

    pub fn set_bits_vary_percentage(&mut self, pct: i32) -> i32 {
        let log_ctx = self.m_pWelsTrace.m_sLogCtx;
        let Some(ctx) = self.active_ctx_mut() else {
            return cmInitExpected;
        };
        ctx.param_mut().iBitsVaryPercentage = WELS_CLIP3(pct, 0, 100);
        let iRang = ctx.param().iBitsVaryPercentage;
        WelsEncoderApplyBitVaryRang(log_ctx, ctx.param_mut(), iRang);
        cmResultSuccess
    }

    #[cfg(any(feature = "c-api", feature = "cxx-api", test))]
    #[allow(unsafe_code, dead_code)]
    /// Raw C-ABI entry point that accepts an untrusted `i32` option id.
    pub(crate) unsafe fn SetOptionRaw(&mut self, eOptionId: i32, pOption: *mut c_void) -> i32 {
        if pOption.is_null() {
            return cmInitParaError;
        }
        let opt = EncoderOption::from_raw(eOptionId);
        if (self.m_pEncContext.is_none() || !self.m_bInitialFlag)
            && !matches!(
                opt,
                Some(
                    EncoderOption::ENCODER_OPTION_TRACE_LEVEL
                        | EncoderOption::ENCODER_OPTION_TRACE_CALLBACK
                        | EncoderOption::ENCODER_OPTION_TRACE_CALLBACK_CONTEXT
                )
            )
        {
            return cmInitExpected;
        }
        let Some(opt) = opt else {
            return cmInitParaError;
        };
        unsafe { self.SetOption(opt, pOption) }
    }

    #[cfg(any(feature = "c-api", feature = "cxx-api", test))]
    #[allow(unsafe_code, dead_code)]
    /// Raw C-ABI entry point that accepts an untrusted `i32` option id.
    pub(crate) unsafe fn GetOptionRaw(&mut self, eOptionId: i32, pOption: *mut c_void) -> i32 {
        if pOption.is_null() {
            return cmInitParaError;
        }
        if self.m_pEncContext.is_none() || !self.m_bInitialFlag {
            return cmInitExpected;
        }
        let Some(opt) = EncoderOption::from_raw(eOptionId) else {
            return cmInitParaError;
        };
        unsafe { self.GetOption(opt, pOption) }
    }

    #[cfg(any(feature = "c-api", feature = "cxx-api", test))]
    #[allow(unsafe_code, dead_code)]
    /// # Safety
    ///
    /// `pOption` must point at a readable, aligned object of the type `eOptionId`
    /// names, live for the call. The three trace ids retain what they read: the
    /// callback and context are used by every later message, under
    /// [`crate::api::codec_api::Encoder::set_trace_callback`]'s contract.
    pub(crate) unsafe fn SetOption(
        &mut self,
        eOptionId: EncoderOption,
        pOption: *mut c_void,
    ) -> i32 {
        if pOption.is_null() {
            return cmInitParaError;
        }
        if !self.has_ctx()
            && eOptionId != EncoderOption::ENCODER_OPTION_TRACE_LEVEL
            && eOptionId != EncoderOption::ENCODER_OPTION_TRACE_CALLBACK
            && eOptionId != EncoderOption::ENCODER_OPTION_TRACE_CALLBACK_CONTEXT
        {
            return cmInitExpected;
        }

        match eOptionId {
            EncoderOption::ENCODER_OPTION_INTER_SPATIAL_PRED => {
                // Unsupported feature: accepted, no effect.
            }
            EncoderOption::ENCODER_OPTION_DATAFORMAT => {
                let iValue = unsafe { *(pOption as *const i32) };
                return self.set_data_format_raw(iValue);
            }
            EncoderOption::ENCODER_OPTION_IDR_INTERVAL => {
                let iValue = unsafe { *(pOption as *const i32) };
                return self.set_idr_interval(iValue);
            }
            EncoderOption::ENCODER_OPTION_SVC_ENCODE_PARAM_BASE => {
                let sEncodingParam =
                    unsafe { read_sanitized_enc_param_base(pOption as *const SEncParamBase) };
                return self.set_encode_param_base(&sEncodingParam);
            }
            EncoderOption::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT => {
                let sEncodingParam =
                    unsafe { read_sanitized_enc_param_ext(pOption as *const SEncParamExt) };
                return self.set_encode_param_ext(&sEncodingParam);
            }
            EncoderOption::ENCODER_OPTION_FRAME_RATE => {
                let iValue = unsafe { *(pOption as *const f32) };
                return self.set_frame_rate(iValue);
            }
            EncoderOption::ENCODER_OPTION_BITRATE => {
                let pInfoPtr = pOption as *const SBitrateInfo;
                let raw_layer = unsafe { ptr::addr_of!((*pInfoPtr).iLayer).cast::<i32>().read() };
                let iBitrate = unsafe { ptr::addr_of!((*pInfoPtr).iBitrate).read() };
                let Some(iLayer) = LAYER_NUM::from_raw(raw_layer) else {
                    return cmInitParaError;
                };
                return self.set_bitrate(&SBitrateInfo { iLayer, iBitrate });
            }
            EncoderOption::ENCODER_OPTION_MAX_BITRATE => {
                let pInfoPtr = pOption as *const SBitrateInfo;
                let raw_layer = unsafe { ptr::addr_of!((*pInfoPtr).iLayer).cast::<i32>().read() };
                let iBitrate = unsafe { ptr::addr_of!((*pInfoPtr).iBitrate).read() };
                let Some(iLayer) = LAYER_NUM::from_raw(raw_layer) else {
                    return cmInitParaError;
                };
                return self.set_max_bitrate(&SBitrateInfo { iLayer, iBitrate });
            }
            EncoderOption::ENCODER_OPTION_RC_MODE => {
                let iValue = unsafe { *(pOption as *const i32) };
                return self.set_rc_mode(rc_mode_from_raw(iValue));
            }
            EncoderOption::ENCODER_OPTION_RC_FRAME_SKIP => {
                let bValue = unsafe { *(pOption as *const u8) != 0 };
                return self.set_rc_frame_skip(bValue);
            }
            EncoderOption::ENCODER_PADDING_PADDING => {
                let iValue = unsafe { *(pOption as *const i32) };
                return self.set_padding(iValue);
            }
            EncoderOption::ENCODER_LTR_RECOVERY_REQUEST => {
                let pLTR_Recover_Request = unsafe { &mut *(pOption as *mut SLTRRecoverRequest) };
                return self.set_ltr_recovery_request(pLTR_Recover_Request);
            }
            EncoderOption::ENCODER_LTR_MARKING_FEEDBACK => {
                let fb = unsafe { &mut *(pOption as *mut SLTRMarkingFeedback) };
                return self.set_ltr_marking_feedback(fb);
            }
            EncoderOption::ENCODER_LTR_MARKING_PERIOD => {
                let iValue = unsafe { *(pOption as *const u32) };
                return self.set_ltr_marking_period(iValue);
            }
            EncoderOption::ENCODER_OPTION_LTR => {
                let pLTRPtr = pOption as *const SLTRConfig;
                let ltr_val = SLTRConfig {
                    bEnableLongTermReference: unsafe {
                        ptr::addr_of!((*pLTRPtr).bEnableLongTermReference)
                            .cast::<u8>()
                            .read()
                            != 0
                    },
                    iLTRRefNum: unsafe { ptr::addr_of!((*pLTRPtr).iLTRRefNum).read() },
                };
                return self.set_ltr_config(&ltr_val);
            }
            EncoderOption::ENCODER_OPTION_ENABLE_SSEI => {
                let iValue = unsafe { *(pOption as *const u8) != 0 };
                return self.set_enable_ssei(iValue);
            }
            EncoderOption::ENCODER_OPTION_ENABLE_PREFIX_NAL_ADDING => {
                let iValue = unsafe { *(pOption as *const u8) != 0 };
                return self.set_enable_prefix_nal_adding(iValue);
            }
            EncoderOption::ENCODER_OPTION_SPS_PPS_ID_STRATEGY => {
                let iValue = unsafe { *(pOption as *const i32) };
                let eNewStrategy = match iValue {
                    0x01 => INCREASING_ID,
                    0x02 => SPS_LISTING,
                    0x03 => SPS_LISTING_AND_PPS_INCREASING,
                    0x06 => SPS_PPS_LISTING,
                    // 0 or out of range: eNewStrategy stays CONSTANT_ID, not an error.
                    _ => CONSTANT_ID,
                };
                return self.set_sps_pps_id_strategy(eNewStrategy);
            }
            EncoderOption::ENCODER_OPTION_CURRENT_PATH => {
                // Accepted and ignored: the path is never read.
            }
            EncoderOption::ENCODER_OPTION_DUMP_FILE => {
                // Frame dumping is not compiled in: accepted, no effect.
            }
            EncoderOption::ENCODER_OPTION_PROFILE => {
                let pProfilePtr = pOption as *const SProfileInfo;
                let iLayer = unsafe { ptr::addr_of!((*pProfilePtr).iLayer).read() };
                let raw_profile = unsafe {
                    ptr::addr_of!((*pProfilePtr).uiProfileIdc)
                        .cast::<i32>()
                        .read()
                };
                let uiProfileIdc = EProfileIdc::from_raw_param(raw_profile);
                return self.set_profile(&SProfileInfo {
                    iLayer,
                    uiProfileIdc,
                });
            }
            EncoderOption::ENCODER_OPTION_LEVEL => {
                let pLevelPtr = pOption as *const SLevelInfo;
                let iLayer = unsafe { ptr::addr_of!((*pLevelPtr).iLayer).read() };
                let raw_level =
                    unsafe { ptr::addr_of!((*pLevelPtr).uiLevelIdc).cast::<i32>().read() };
                let uiLevelIdc = ELevelIdc::from_raw(raw_level);
                return self.set_level(&SLevelInfo { iLayer, uiLevelIdc });
            }
            EncoderOption::ENCODER_OPTION_NUMBER_REF => {
                let iValue = unsafe { *(pOption as *const i32) };
                return self.set_number_ref(iValue);
            }
            EncoderOption::ENCODER_OPTION_DELIVERY_STATUS => {
                let pDelivPtr = pOption as *const SDeliveryStatus;
                let bDeliveryFlag = unsafe {
                    ptr::addr_of!((*pDelivPtr).bDeliveryFlag)
                        .cast::<u8>()
                        .read()
                        != 0
                };
                return self.set_delivery_status(&SDeliveryStatus {
                    bDeliveryFlag,
                    iDropFrameType: 0,
                    iDropFrameSize: 0,
                });
            }
            EncoderOption::ENCODER_OPTION_COMPLEXITY => {
                let iValue = unsafe { *(pOption as *const i32) };
                return self.set_complexity(EComplexityMode::from_raw(iValue));
            }
            EncoderOption::ENCODER_OPTION_GET_STATISTICS => {
                // Get-only: accepted, no effect.
            }
            EncoderOption::ENCODER_OPTION_STATISTICS_LOG_INTERVAL => {
                let iValue = unsafe { *(pOption as *const i32) };
                return self.set_statistics_log_interval(iValue);
            }
            EncoderOption::ENCODER_OPTION_IS_LOSSLESS_LINK => {
                let bValue = unsafe { *(pOption as *const u8) != 0 };
                return self.set_is_lossless_link(bValue);
            }
            EncoderOption::ENCODER_OPTION_BITS_VARY_PERCENTAGE => {
                let iValue = unsafe { *(pOption as *const i32) };
                return self.set_bits_vary_percentage(iValue);
            }
            EncoderOption::ENCODER_OPTION_TRACE_LEVEL => {
                let level = unsafe { pOption.cast::<u32>().read() };
                self.m_pWelsTrace.SetTraceLevel(level);
                self.sync_log_ctx();
            }
            EncoderOption::ENCODER_OPTION_TRACE_CALLBACK => {
                let callback = unsafe { pOption.cast::<WelsTraceCallback>().read() };
                self.m_pWelsTrace.SetTraceCallback(callback);
                self.sync_log_ctx();
            }
            EncoderOption::ENCODER_OPTION_TRACE_CALLBACK_CONTEXT => {
                // The caller's opaque trace context: kept until replaced, handed
                // back to the callback untouched, never dereferenced here.
                let ctx = unsafe { pOption.cast::<*mut c_void>().read() };
                self.m_pWelsTrace
                    .SetTraceCallbackContext(TraceUserCtx::from_abi(ctx));
                self.sync_log_ctx();
            } // Exhaustive: a new option id must be handled here.
        }
        0
    }

    #[cfg(any(feature = "c-api", feature = "cxx-api", test))]
    #[allow(unsafe_code, dead_code)]
    /// # Safety
    ///
    /// As [`Self::SetOption`], with `pOption` written rather than read.
    pub(crate) unsafe fn GetOption(
        &mut self,
        eOptionId: EncoderOption,
        pOption: *mut c_void,
    ) -> i32 {
        if pOption.is_null() {
            return cmInitParaError;
        }
        if !self.has_ctx() {
            return cmInitExpected;
        }

        match eOptionId {
            EncoderOption::ENCODER_OPTION_INTER_SPATIAL_PRED => {
                // Unsupported feature: accepted, no effect.
            }
            EncoderOption::ENCODER_OPTION_DATAFORMAT => {
                let Some(val) = self.data_format() else {
                    return cmInitExpected;
                };
                unsafe { *(pOption as *mut i32) = val };
            }
            EncoderOption::ENCODER_OPTION_IDR_INTERVAL => {
                let Some(val) = self.idr_interval() else {
                    return cmInitExpected;
                };
                unsafe { *(pOption as *mut i32) = val };
            }
            EncoderOption::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT => {
                let Some(param_ext) = self.encode_param_ext() else {
                    return cmInitExpected;
                };
                unsafe { (pOption as *mut SEncParamExt).write(param_ext) };
            }
            EncoderOption::ENCODER_OPTION_SVC_ENCODE_PARAM_BASE => {
                let Some(out) = self.encode_param_base() else {
                    return cmInitExpected;
                };
                unsafe { (pOption as *mut SEncParamBase).write(out) };
            }
            EncoderOption::ENCODER_OPTION_FRAME_RATE => {
                let Some(fps) = self.frame_rate() else {
                    return cmInitExpected;
                };
                unsafe { *(pOption as *mut f32) = fps };
            }
            EncoderOption::ENCODER_OPTION_BITRATE => {
                let pInfoPtr = pOption as *mut SBitrateInfo;
                let raw_layer = unsafe { ptr::addr_of!((*pInfoPtr).iLayer).cast::<i32>().read() };
                let Some(layer) = LAYER_NUM::from_raw(raw_layer) else {
                    return cmInitParaError;
                };
                let Some(bitrate) = self.bitrate(layer) else {
                    return cmInitExpected;
                };
                unsafe { ptr::addr_of_mut!((*pInfoPtr).iBitrate).write(bitrate) };
            }
            EncoderOption::ENCODER_OPTION_MAX_BITRATE => {
                let pInfoPtr = pOption as *mut SBitrateInfo;
                let raw_layer = unsafe { ptr::addr_of!((*pInfoPtr).iLayer).cast::<i32>().read() };
                let Some(layer) = LAYER_NUM::from_raw(raw_layer) else {
                    return cmInitParaError;
                };
                let Some(bitrate) = self.max_bitrate(layer) else {
                    return cmInitExpected;
                };
                unsafe { ptr::addr_of_mut!((*pInfoPtr).iBitrate).write(bitrate) };
            }
            EncoderOption::ENCODER_OPTION_GET_STATISTICS => {
                let Some(stats) = self.statistics() else {
                    return cmInitExpected;
                };
                unsafe {
                    (pOption as *mut crate::SEncoderStatistics).write(stats);
                }
            }
            EncoderOption::ENCODER_OPTION_STATISTICS_LOG_INTERVAL => {
                let Some(interval) = self.statistics_log_interval() else {
                    return cmInitExpected;
                };
                unsafe { *(pOption as *mut i32) = interval };
            }
            EncoderOption::ENCODER_OPTION_COMPLEXITY => {
                let Some(mode) = self.complexity() else {
                    return cmInitExpected;
                };
                unsafe { *(pOption as *mut i32) = mode as i32 };
            }
            // Unhandled or set-only options fall to the error arm below.
            _ => return cmInitParaError,
        }
        0
    }
}
