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

use std::ffi::c_void;
use std::ptr;

use crate::api::codec_api::EParameterSetStrategy::*;
use crate::api::codec_api::LAYER_NUM::*;
use crate::api::codec_api::RC_MODES::*;
use crate::api::codec_api::TraceUserCtx;
use crate::api::codec_api::{
    EComplexityMode, ELevelIdc, EProfileIdc, EncoderOption, LAYER_NUM, SBitrateInfo, SEncParamBase,
    SEncParamExt,
};
use crate::api::types::{read_sanitized_enc_param_base, read_sanitized_enc_param_ext};
use crate::common::wels_trace::{WELS_LOG_INFO, WelsLog, WelsTraceCallback};
use crate::encoder::param_svc::{MAX_SPATIAL_LAYER_NUM, SWelsSvcCodingParam, WELS_CLIP3};
use crate::encoder::rc::WelsRcInitFuncPointers;
use crate::encoder::ref_list_mgr_svc::{
    FilterLTRMarkingFeedback, FilterLTRRecoveryRequest, SLTRMarkingFeedback, SLTRRecoverRequest,
};
use crate::encoder::wels_encoder_ext::{
    CWelsH264SVCEncoder, CheckLevelSetting, CheckProfileSetting, CheckReferenceNumSetting,
    MAX_BIT_RATE, MAX_DEPENDENCY_LAYER, MAX_FRAME_RATE, MIN_BIT_RATE, MIN_FRAME_RATE,
    SDeliveryStatus, SLTRConfig, SLevelInfo, SProfileInfo, WelsEncoderApplyBitRate,
    WelsEncoderApplyBitVaryRang, WelsEncoderApplyFrameRate, WelsEncoderApplyLTR,
    WelsEncoderParamAdjust, cmInitExpected, cmInitParaError, cmResultSuccess, rc_mode_from_raw,
};

impl CWelsH264SVCEncoder {
    #[allow(unsafe_code)]
    /// Raw C-ABI entry point that accepts an untrusted `i32` option id.
    pub unsafe fn SetOptionRaw(&mut self, eOptionId: i32, pOption: *mut c_void) -> i32 {
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

    #[allow(unsafe_code)]
    /// Raw C-ABI entry point that accepts an untrusted `i32` option id.
    pub unsafe fn GetOptionRaw(&mut self, eOptionId: i32, pOption: *mut c_void) -> i32 {
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

    #[allow(unsafe_code)]
    /// # Safety
    ///
    /// `pOption` must point at a readable, aligned object of the type `eOptionId`
    /// names, live for the call. The three trace ids retain what they read: the
    /// callback and context are used by every later message, under
    /// [`crate::api::codec_api::Encoder::set_trace_callback`]'s contract.
    pub unsafe fn SetOption(&mut self, eOptionId: EncoderOption, pOption: *mut c_void) -> i32 {
        if pOption.is_null() {
            return cmInitParaError;
        }
        if (self.m_pEncContext.is_none() || !self.m_bInitialFlag)
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
                if iValue == 0 {
                    return cmInitParaError;
                }
                self.m_iCspInternal = iValue;
            }
            EncoderOption::ENCODER_OPTION_IDR_INTERVAL => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                let mut iValue = unsafe { *(pOption as *const i32) };
                if iValue <= -1 {
                    iValue = 0;
                }
                if iValue == ctx.param().uiIntraPeriod as i32 {
                    return cmResultSuccess;
                }
                ctx.param_mut().uiIntraPeriod = iValue as u32;
            }
            EncoderOption::ENCODER_OPTION_SVC_ENCODE_PARAM_BASE => {
                let sEncodingParam =
                    unsafe { read_sanitized_enc_param_base(pOption as *const SEncParamBase) };
                let mut sConfig = SWelsSvcCodingParam::default();
                if sConfig.ParamBaseTranscode(&sEncodingParam) != 0 {
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
            }
            EncoderOption::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT => {
                let sEncodingParam =
                    unsafe { read_sanitized_enc_param_ext(pOption as *const SEncParamExt) };
                // Traced before the spatial-layer check, so rejected parameters
                // are still echoed.
                self.TraceParamInfo(&sEncodingParam);
                if sEncodingParam.iSpatialLayerNum < 1
                    || sEncodingParam.iSpatialLayerNum > MAX_SPATIAL_LAYER_NUM as i32
                {
                    return cmInitParaError;
                }
                let mut sConfig = SWelsSvcCodingParam::default();
                if sConfig.ParamTranscode(&sEncodingParam) != 0 {
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
            }
            EncoderOption::ENCODER_OPTION_FRAME_RATE => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                let iValue = unsafe { *(pOption as *const f32) };
                if iValue <= 0.0 {
                    return cmInitParaError;
                }
                ctx.param_mut().fMaxFrameRate = WELS_CLIP3(iValue, MIN_FRAME_RATE, MAX_FRAME_RATE);
                WelsEncoderApplyFrameRate(ctx.param_mut());
            }
            EncoderOption::ENCODER_OPTION_BITRATE => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                let pInfoPtr = pOption as *const SBitrateInfo;
                let raw_layer = unsafe { ptr::addr_of!((*pInfoPtr).iLayer).cast::<i32>().read() };
                let mut iBitrate = unsafe { ptr::addr_of!((*pInfoPtr).iBitrate).read() };
                if iBitrate <= 0 {
                    return cmInitParaError;
                }
                let Some(iLayer) = LAYER_NUM::from_raw(raw_layer) else {
                    return cmInitParaError;
                };
                iBitrate = WELS_CLIP3(iBitrate, MIN_BIT_RATE, MAX_BIT_RATE);
                match iLayer {
                    SPATIAL_LAYER_ALL => {
                        ctx.param_mut().iTargetBitrate = iBitrate;
                    }
                    SPATIAL_LAYER_0 | SPATIAL_LAYER_1 | SPATIAL_LAYER_2 | SPATIAL_LAYER_3 => {
                        ctx.param_mut().sSpatialLayers[iLayer as usize].iSpatialBitrate = iBitrate;
                    }
                }
                let log_ctx = self.m_pWelsTrace.m_sLogCtx;
                if WelsEncoderApplyBitRate(log_ctx, ctx.param_mut(), iLayer as i32) != 0 {
                    return cmInitParaError;
                }
            }
            EncoderOption::ENCODER_OPTION_MAX_BITRATE => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                let pInfoPtr = pOption as *const SBitrateInfo;
                let raw_layer = unsafe { ptr::addr_of!((*pInfoPtr).iLayer).cast::<i32>().read() };
                let mut iBitrate = unsafe { ptr::addr_of!((*pInfoPtr).iBitrate).read() };
                if iBitrate <= 0 {
                    return cmInitParaError;
                }
                let Some(iLayer) = LAYER_NUM::from_raw(raw_layer) else {
                    return cmInitParaError;
                };
                iBitrate = WELS_CLIP3(iBitrate, MIN_BIT_RATE, MAX_BIT_RATE);
                match iLayer {
                    SPATIAL_LAYER_ALL => {
                        ctx.param_mut().iMaxBitrate = iBitrate;
                    }
                    SPATIAL_LAYER_0 | SPATIAL_LAYER_1 | SPATIAL_LAYER_2 | SPATIAL_LAYER_3 => {
                        ctx.param_mut().sSpatialLayers[iLayer as usize].iMaxSpatialBitrate =
                            iBitrate;
                    }
                }
                let log_ctx = self.m_pWelsTrace.m_sLogCtx;
                if WelsEncoderApplyBitRate(log_ctx, ctx.param_mut(), iLayer as i32) != 0 {
                    return cmInitParaError;
                }
            }
            EncoderOption::ENCODER_OPTION_RC_MODE => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                // 0:quality mode;1:bit-rate mode;2:bitrate limited mode
                let iValue = unsafe { *(pOption as *const i32) };
                ctx.param_mut().iRCMode = rc_mode_from_raw(iValue);
                // Re-point the dispatch table; the field alone leaves the
                // encoder running the previous mode's callbacks.
                let iRCMode = ctx.param().iRCMode;
                WelsRcInitFuncPointers(&mut ctx.func_list_mut().pfRc, iRCMode);
            }
            EncoderOption::ENCODER_OPTION_RC_FRAME_SKIP => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                // 0:FRAME-SKIP disabled;1:FRAME-SKIP enabled
                let bValue = unsafe { *(pOption as *const u8) != 0 };
                if ctx.param().iRCMode != RC_OFF_MODE {
                    ctx.param_mut().bEnableFrameSkip = bValue;
                }
                // rc off: the setting is accepted and ignored.
            }
            EncoderOption::ENCODER_PADDING_PADDING => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                // 0:disable padding;1:padding
                let iValue = unsafe { *(pOption as *const i32) };
                ctx.param_mut().iPaddingFlag = iValue;
            }
            EncoderOption::ENCODER_LTR_RECOVERY_REQUEST => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                let pLTR_Recover_Request = unsafe { &mut *(pOption as *mut SLTRRecoverRequest) };
                // The second argument points into the caller's memory, not the context.
                FilterLTRRecoveryRequest(ctx, pLTR_Recover_Request);
            }
            EncoderOption::ENCODER_LTR_MARKING_FEEDBACK => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                let fb = unsafe { &mut *(pOption as *mut SLTRMarkingFeedback) };
                // As the recovery-request arm above — `pOption` is the caller's.
                FilterLTRMarkingFeedback(ctx, fb);
            }
            EncoderOption::ENCODER_LTR_MARKING_PERIOD => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                let iValue = unsafe { *(pOption as *const u32) };
                ctx.param_mut().iLtrMarkPeriod = iValue;
            }
            EncoderOption::ENCODER_OPTION_LTR => {
                let pLTRPtr = pOption as *const SLTRConfig;
                let mut ltr_val = SLTRConfig {
                    bEnableLongTermReference: unsafe {
                        ptr::addr_of!((*pLTRPtr).bEnableLongTermReference)
                            .cast::<u8>()
                            .read()
                            != 0
                    },
                    iLTRRefNum: unsafe { ptr::addr_of!((*pLTRPtr).iLTRRefNum).read() },
                };
                let log_ctx = self.m_pWelsTrace.m_sLogCtx;
                if WelsEncoderApplyLTR(log_ctx, &mut self.m_pEncContext, &mut ltr_val) != 0 {
                    return cmInitParaError;
                }
            }
            EncoderOption::ENCODER_OPTION_ENABLE_SSEI => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                let iValue = unsafe { *(pOption as *const u8) != 0 };
                ctx.param_mut().bEnableSSEI = iValue;
            }
            EncoderOption::ENCODER_OPTION_ENABLE_PREFIX_NAL_ADDING => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                let iValue = unsafe { *(pOption as *const u8) != 0 };
                ctx.param_mut().bPrefixNalAddingCtrl = iValue;
            }
            EncoderOption::ENCODER_OPTION_SPS_PPS_ID_STRATEGY => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                let iValue = unsafe { *(pOption as *const i32) };
                let mut eNewStrategy = CONSTANT_ID;
                match iValue {
                    0 => eNewStrategy = CONSTANT_ID,
                    0x01 => eNewStrategy = INCREASING_ID,
                    0x02 => eNewStrategy = SPS_LISTING,
                    0x03 => eNewStrategy = SPS_LISTING_AND_PPS_INCREASING,
                    0x06 => eNewStrategy = SPS_PPS_LISTING,
                    // Out of range: eNewStrategy stays CONSTANT_ID, not an error.
                    _ => {}
                }

                let eOld = ctx.param().eSpsPpsIdStrategy;
                if ((eNewStrategy as i32 & SPS_LISTING as i32) != 0
                    || (eOld as i32 & SPS_LISTING as i32) != 0)
                    && eOld != eNewStrategy
                {
                    // changing in the middle of call is NOT allowed for
                    // eSpsPpsIdStrategy > INCREASING_ID
                    return cmInitParaError;
                }
                let mut sConfig: SWelsSvcCodingParam = *ctx.param();
                sConfig.eSpsPpsIdStrategy = eNewStrategy;

                if WelsEncoderParamAdjust(&mut self.m_pEncContext, &mut sConfig) != 0 {
                    return cmInitParaError;
                }
            }
            EncoderOption::ENCODER_OPTION_CURRENT_PATH => {
                // Accepted and ignored: the path is never read.
            }
            EncoderOption::ENCODER_OPTION_DUMP_FILE => {
                // Frame dumping is not compiled in: accepted, no effect.
            }
            EncoderOption::ENCODER_OPTION_PROFILE => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                let pProfilePtr = pOption as *const SProfileInfo;
                let iLayer = unsafe { ptr::addr_of!((*pProfilePtr).iLayer).read() };
                let raw_profile = unsafe {
                    ptr::addr_of!((*pProfilePtr).uiProfileIdc)
                        .cast::<i32>()
                        .read()
                };
                if iLayer < SPATIAL_LAYER_0 as i32 || iLayer > SPATIAL_LAYER_3 as i32 {
                    return cmInitParaError;
                }
                let uiProfileIdc = EProfileIdc::from_raw_param(raw_profile);
                let log_ctx = self.m_pWelsTrace.m_sLogCtx;
                CheckProfileSetting(log_ctx, ctx.param_mut(), iLayer, uiProfileIdc);
            }
            EncoderOption::ENCODER_OPTION_LEVEL => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                let pLevelPtr = pOption as *const SLevelInfo;
                let iLayer = unsafe { ptr::addr_of!((*pLevelPtr).iLayer).read() };
                let raw_level =
                    unsafe { ptr::addr_of!((*pLevelPtr).uiLevelIdc).cast::<i32>().read() };
                if iLayer < SPATIAL_LAYER_0 as i32 || iLayer > SPATIAL_LAYER_3 as i32 {
                    return cmInitParaError;
                }
                let uiLevelIdc = ELevelIdc::from_raw(raw_level);
                let log_ctx = self.m_pWelsTrace.m_sLogCtx;
                CheckLevelSetting(log_ctx, ctx.param_mut(), iLayer, uiLevelIdc);
            }
            EncoderOption::ENCODER_OPTION_NUMBER_REF => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                let iValue = unsafe { *(pOption as *const i32) };
                let log_ctx = self.m_pWelsTrace.m_sLogCtx;
                CheckReferenceNumSetting(log_ctx, ctx.param_mut(), iValue);
            }
            EncoderOption::ENCODER_OPTION_DELIVERY_STATUS => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                let pDelivPtr = pOption as *const SDeliveryStatus;
                ctx.bDeliveryFlag = unsafe {
                    ptr::addr_of!((*pDelivPtr).bDeliveryFlag)
                        .cast::<u8>()
                        .read()
                        != 0
                };
            }
            EncoderOption::ENCODER_OPTION_COMPLEXITY => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                let iValue = unsafe { *(pOption as *const i32) };
                ctx.param_mut().iComplexityMode = EComplexityMode::from_raw(iValue);
            }
            EncoderOption::ENCODER_OPTION_GET_STATISTICS => {
                // Get-only: accepted, no effect.
            }
            EncoderOption::ENCODER_OPTION_STATISTICS_LOG_INTERVAL => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                let iValue = unsafe { *(pOption as *const i32) };
                ctx.iStatisticsLogInterval = iValue;
            }
            EncoderOption::ENCODER_OPTION_IS_LOSSLESS_LINK => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                let bValue = unsafe { *(pOption as *const u8) != 0 };
                ctx.param_mut().bIsLosslessLink = bValue;
            }
            EncoderOption::ENCODER_OPTION_BITS_VARY_PERCENTAGE => {
                let Some(ctx) = self.m_pEncContext.as_deref_mut() else {
                    return cmInitExpected;
                };
                let iValue = unsafe { *(pOption as *const i32) };
                ctx.param_mut().iBitsVaryPercentage = WELS_CLIP3(iValue, 0, 100);
                let log_ctx = self.m_pWelsTrace.m_sLogCtx;
                let iRang = ctx.param().iBitsVaryPercentage;
                WelsEncoderApplyBitVaryRang(log_ctx, ctx.param_mut(), iRang);
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

    #[allow(unsafe_code)]
    /// # Safety
    ///
    /// As [`Self::SetOption`], with `pOption` written rather than read.
    pub unsafe fn GetOption(&mut self, eOptionId: EncoderOption, pOption: *mut c_void) -> i32 {
        if pOption.is_null() {
            return cmInitParaError;
        }
        let Some(pCtx) = self.m_pEncContext.as_deref() else {
            return cmInitExpected;
        };
        if !self.m_bInitialFlag {
            return cmInitExpected;
        }

        match eOptionId {
            EncoderOption::ENCODER_OPTION_INTER_SPATIAL_PRED => {
                // Unsupported feature: accepted, no effect.
            }
            EncoderOption::ENCODER_OPTION_DATAFORMAT => {
                unsafe { *(pOption as *mut i32) = self.m_iCspInternal };
            }
            EncoderOption::ENCODER_OPTION_IDR_INTERVAL => {
                unsafe { *(pOption as *mut i32) = pCtx.param().uiIntraPeriod as i32 };
            }
            EncoderOption::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT => {
                let param_ext = pCtx.param().to_param_ext();
                unsafe { (pOption as *mut SEncParamExt).write(param_ext) };
            }
            EncoderOption::ENCODER_OPTION_SVC_ENCODE_PARAM_BASE => {
                let mut out = SEncParamBase::default();
                pCtx.param().GetBaseParams(&mut out);
                unsafe { (pOption as *mut SEncParamBase).write(out) };
            }
            EncoderOption::ENCODER_OPTION_FRAME_RATE => {
                unsafe { *(pOption as *mut f32) = pCtx.param().fMaxFrameRate };
            }
            EncoderOption::ENCODER_OPTION_BITRATE => {
                let pInfoPtr = pOption as *mut SBitrateInfo;
                let raw_layer = unsafe { ptr::addr_of!((*pInfoPtr).iLayer).cast::<i32>().read() };
                let bitrate = if raw_layer == SPATIAL_LAYER_ALL as i32 {
                    pCtx.param().iTargetBitrate
                } else if raw_layer >= 0 && raw_layer < MAX_DEPENDENCY_LAYER {
                    pCtx.param().sSpatialLayers[raw_layer as usize].iSpatialBitrate
                } else {
                    return cmInitParaError;
                };
                unsafe { ptr::addr_of_mut!((*pInfoPtr).iBitrate).write(bitrate) };
            }
            EncoderOption::ENCODER_OPTION_MAX_BITRATE => {
                let pInfoPtr = pOption as *mut SBitrateInfo;
                let raw_layer = unsafe { ptr::addr_of!((*pInfoPtr).iLayer).cast::<i32>().read() };
                let bitrate = if raw_layer == SPATIAL_LAYER_ALL as i32 {
                    pCtx.param().iMaxBitrate
                } else if raw_layer >= 0 && raw_layer < MAX_DEPENDENCY_LAYER {
                    pCtx.param().sSpatialLayers[raw_layer as usize].iMaxSpatialBitrate
                } else {
                    return cmInitParaError;
                };
                unsafe { ptr::addr_of_mut!((*pInfoPtr).iBitrate).write(bitrate) };
            }
            EncoderOption::ENCODER_OPTION_GET_STATISTICS => {
                let iLayerIdx = (pCtx.param().iSpatialLayerNum - 1) as usize;
                unsafe {
                    (pOption as *mut crate::SEncoderStatistics)
                        .write(pCtx.sEncoderStatistics[iLayerIdx]);
                }
            }
            EncoderOption::ENCODER_OPTION_STATISTICS_LOG_INTERVAL => {
                unsafe { *(pOption as *mut i32) = pCtx.iStatisticsLogInterval };
            }
            EncoderOption::ENCODER_OPTION_COMPLEXITY => {
                unsafe { *(pOption as *mut i32) = pCtx.param().iComplexityMode as i32 };
            }
            // Unhandled or set-only options fall to the error arm below.
            _ => return cmInitParaError,
        }
        0
    }
}
