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

#![forbid(unsafe_code)]

//! # Multithreaded Slice Processing & Dynamic Workload Balancing
//!
//! C++: `codec/encoder/core/inc/slice_multi_threading.h`,
//! `codec/encoder/core/src/slice_multi_threading.cpp`.
//!
//! Implements OpenH264's slice-level multithreading architecture, Root-Mean-Square
//! Error (RMSE) dynamic load balancing, thread-local bitstream buffer binding, and
//! Annex B NAL bitstream aggregation.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::sync::atomic::{AtomicI32, AtomicU16, Ordering};

use crate::common::wels_common_defs::{EWelsNalRefIdc, EWelsNalUnitType};
use crate::encoder::nal_encap::{
    WelsEncodeNal, WelsLoadNalForSlice, WelsUnloadNalForSlice, WelsWriteSVCPrefixNal,
};
use crate::encoder::picture::SlicePicBands;
use crate::encoder::svc_encode_slice::{
    InitOneSliceInThread, SetSliceBoundaryInfo, TakenFramePics, WelsCodeOneSlice, current_layer_ref,
};
use crate::encoder::vlc_encoder::BsWriter;
use crate::encoder::wels_encoder_ext::WelsTime;
pub const ENC_RETURN_UNEXPECTED: i32 = 0x04;
pub use crate::encoder::encoder_context::sWelsEncCtx;
pub use crate::encoder::md::SMB;
pub use crate::encoder::nal_encap::SWelsSliceBs;
pub use crate::encoder::rc::SWelsSvcRc;
pub use crate::encoder::svc_encode_slice::SDqLayer;
pub use crate::encoder::svc_encode_slice::SLayerInfo;
pub use crate::encoder::svc_encode_slice::SSlice;
use crate::encoder::svc_encode_slice::{LayerIdx, set_current_layer};
use crate::encoder::svc_encode_slice::{
    ReallocateSliceInThread, SSliceBufferInfo, UpdateMbNeighbor, current_layer_expect,
    current_layer_expect_mut, slice_in_layer_mut,
};
use crate::{MAX_SPATIAL_LAYER_NUM, RCMode, SLayerBSInfo, SliceMode};

// ============================================================================
// Constants and Thresholds
// ============================================================================

pub const THRESHOLD_RMSE_CORE8: f32 = 0.0320;
pub const THRESHOLD_RMSE_CORE4: f32 = 0.0215;
pub const THRESHOLD_RMSE_CORE2: f32 = 0.0200;
pub const EPSN: f32 = 0.000001;
pub const INT_MULTIPLY: i32 = 100;
pub const SEM_NAME_MAX: usize = 32;
pub use crate::encoder::encoder_context::MAX_THREADS_NUM;
use crate::encoder::encoder_context::{dq_layer_mut, dq_layer_ref};
use crate::encoder::nal_encap::SWelsEncoderOutput;
use crate::encoder::param_svc::SWelsSvcCodingParam;
/// One definition, in `wels_encoder_ext` — `svc_enc_slice_segment.h:62`.
pub use crate::encoder::wels_encoder_ext::MAX_SLICES_NUM;
use crate::encoder::worker_pool::WorkerPool;
use crate::safe::mb_grid::MbArray;
pub const MAX_DEPENDENCY_LAYER: usize = 4;

pub const ENC_RETURN_SUCCESS: i32 = 0;
pub const ENC_RETURN_MEMALLOCERR: i32 = 0x01;

/// `DEFAULT_MAXPACKETSIZE_CONSTRAINT` — `svc_enc_slice_segment.h:67`, in bytes.
pub const DEFAULT_MAXPACKETSIZE_CONSTRAINT: u32 = 1200;

/// `WelsSetMemMultiplebytes_c` over the macroblock map, as a `fill` on a subslice
/// — `codec/common/inc/macros.h:312`.
#[inline]
pub fn fill_mb_map(map: &[AtomicU16], kiFirstMb: i32, kiCount: i32, uiValue: u16) {
    if kiFirstMb < 0 || kiCount <= 0 {
        return;
    }
    let a = kiFirstMb as usize;
    let b = a.saturating_add(kiCount as usize).min(map.len());
    if a < b {
        // `&[AtomicU16]` rather than `&mut [u16]`: `AddSliceBoundary` calls this
        // from inside the fork, where the other partitions are reading the map.
        for c in &map[a..b] {
            c.store(uiValue, Ordering::Relaxed);
        }
    }
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug)]
pub struct SSliceThreading {
    /// The `iSliceNumInFrame` lock. `DynSlcJudgeSliceBoundaryStepBack` holds it
    /// across `AddSliceBoundary` and the increment
    /// (`svc_encode_slice.cpp:1776-1791`).
    pub mutexSliceNumUpdate: std::sync::Mutex<()>,
    /// One bs scratch buffer per worker slot. The fork entries `mem::take` each
    /// worker's `Vec` out of this array before the spawn and restore it after the
    /// join, so no worker derives from the shared context at all.
    pub pThreadBsBuffer: [Vec<u8>; MAX_THREADS_NUM],
    /// How many of the `MAX_THREADS_NUM` slots actually have a buffer behind them:
    /// `min(iMultipleThreadIdc, MAX_THREADS_NUM)`. This is where the fork reads it
    /// (`ForkWidth`).
    pub uiThreadBsBufferNum: usize,
    /// The persistent background worker threads the three forks run on —
    /// `CWelsThreadPool`'s counterpart. Built with `uiThreadBsBufferNum - 1`
    /// background workers (the calling thread executes slot 0 directly via
    /// [`WorkerPool::for_each_mut`]) and dropped with the context at
    /// `WelsUninitEncoderExt`, so none outlives the encoder. `Default` is a pool
    /// of no threads, on which every job runs on the calling thread.
    pub pool: WorkerPool,
}

impl Default for SSliceThreading {
    fn default() -> Self {
        Self {
            mutexSliceNumUpdate: std::sync::Mutex::new(()),
            pThreadBsBuffer: std::array::from_fn(|_| Vec::new()),
            uiThreadBsBufferNum: 0,
            pool: WorkerPool::new(0),
        }
    }
}
pub type TagSliceThreading = SSliceThreading;

pub type TagWelsSliceBs = SWelsSliceBs;

pub type TagSlice = SSlice;

// The element type is `AtomicU16`: `AddSliceBoundary` rewrites this map from
// inside the fork under `SM_SIZELIMITED_SLICE` while the other partitions'
// workers read it — one entry per macroblock, partitions disjoint, nothing
// synchronising the storage itself. `Relaxed` throughout: the disjointness is the
// argument, and the scope join is the publication edge.
#[derive(Debug)]
pub struct SSliceCtx {
    pub uiSliceMode: SliceMode,
    pub iMbWidth: i16,
    pub iMbHeight: i16,
    /// `DynSlcJudgeSliceBoundaryStepBack` increments this from inside the fork
    /// under `mutexSliceNumUpdate`, but every *reader* takes no lock:
    /// `WelsGetNextMbOfSlice` reborrows the whole `SSliceCtx` per macroblock on
    /// each worker. The mutex is *not* redundant: it brackets `AddSliceBoundary`'s
    /// map rewrite *with* the increment, which no single atomic can do.
    pub iSliceNumInFrame: AtomicI32,
    pub iMbNumInFrame: i32,
    /// One slice index per macroblock, in raster order. Allocated by
    /// `InitSlicePEncCtx`, released by the layer's `Drop` where `UninitSlicePEncCtx`
    /// used to free it.
    pub pOverallMbMap: Vec<AtomicU16>,
    pub uiSliceSizeConstraint: u32,
    pub iMaxSliceNumConstraint: i32,
}

impl Default for SSliceCtx {
    fn default() -> Self {
        Self {
            uiSliceMode: SliceMode::SM_SINGLE_SLICE,
            iMbWidth: 0,
            iMbHeight: 0,
            iSliceNumInFrame: AtomicI32::new(0),
            iMbNumInFrame: 0,
            pOverallMbMap: Vec::new(),
            uiSliceSizeConstraint: 0,
            iMaxSliceNumConstraint: 0,
        }
    }
}
pub type SlicepEncCtx_s = SSliceCtx;

pub type TagDqLayer = SDqLayer;

pub type SWelsEncCtx = sWelsEncCtx;

// ============================================================================
// Arithmetic Helpers
// ============================================================================

#[inline]
pub fn WelsDivRound(x: i32, y: i32) -> i32 {
    if y == 0 { x / (y + 1) } else { (y / 2 + x) / y }
}

#[inline]
pub fn WelsDivRound64(x: i64, y: i64) -> i64 {
    if y == 0 { x / (y + 1) } else { (y / 2 + x) / y }
}

// ============================================================================
// Mutex helpers (`WelsMutexInit` / `WelsMutexLock` / `WelsMutexUnlock` /
// `WelsMutexDestroy` from `codec/common/inc/WelsThreadLib.h`)
// ============================================================================
//
// A `std::sync::Mutex` cannot be locked and unlocked through two separate calls
// the way pthreads can — the guard owns the lock — so each lock/unlock pair is
// expressed as one scoped call over a straight-line critical section.

/// Runs `f` holding `pMutex`, i.e. a `WelsMutexLock`/`WelsMutexUnlock` pair.
///
/// `None` runs `f` unlocked; only the single-threaded paths pass it, and they
/// never contend.
pub fn with_wels_mutex<R>(pMutex: Option<&std::sync::Mutex<()>>, f: impl FnOnce() -> R) -> R {
    let Some(m) = pMutex else {
        return f();
    };
    // A worker that panicked mid-slice leaves the mutex poisoned; the encoder
    // has no recovery path for that, so take the guard either way.
    let _guard = m.lock().unwrap_or_else(|e| e.into_inner());
    f()
}

// ============================================================================
// Core Multithreading Functions
// ============================================================================

/// Updates macroblock spatial neighbor availability bitmasks for all macroblocks
/// belonging to a specific slice partition in parallel. The window is the
/// caller's — carved out of the grid before the fork by [`UpdateMbMapForked`].
pub fn UpdateMbListNeighborParallel(
    mbs: &mut crate::safe::mb_grid::MbWindow<'_, SMB>,
    pSliceCtx: &SSliceCtx,
    kiMbWidth: i32,
    kiSliceIdc: i32,
    kiFirst: i32,
    kiCount: i32,
) {
    let kiEndMbInSlice = kiFirst + kiCount - 1;
    let mut iIdx = kiFirst;
    while iIdx <= kiEndMbInSlice {
        UpdateMbNeighbor(
            Some(pSliceCtx),
            mbs.at_mut(iIdx as usize),
            kiMbWidth,
            kiSliceIdc as u16,
        );
        iIdx += 1;
    }
}

/// Calculates the normalized computational complexity ratio (`iSliceComplexRatio`)
/// for each slice in a spatial layer based on measured CPU consumption time.
///
/// The producer half of the load-balancing loop, called from
/// `WelsEncoderEncodeExt` at the end of the per-layer body, under its four-term
/// guard (`encoder_ext.cpp:4064-4073`).
pub fn CalcSliceComplexRatio(pCurDq: &mut SDqLayer) {
    let pSliceCtx = &mut pCurDq.sSliceEncCtx;
    let mut iSumAv = 0i32;
    let kiSliceCount = pSliceCtx.iSliceNumInFrame.load(Ordering::Relaxed);
    let mut iSliceIdx = 0i32;
    let mut iAvI = [0i32; MAX_SLICES_NUM];

    if kiSliceCount > MAX_SLICES_NUM as i32 {
        return;
    }
    while iSliceIdx < kiSliceCount {
        let pSlice = slice_in_layer_mut(pCurDq, iSliceIdx);
        if let Some(pSlice) = pSlice {
            let consume_time = pSlice.uiSliceConsumeTime as i32;
            let mb_num = pSlice.iCountMbNumInSlice;
            iAvI[iSliceIdx as usize] = WelsDivRound(INT_MULTIPLY * mb_num, consume_time);
            iSumAv += iAvI[iSliceIdx as usize];
        }
        iSliceIdx += 1;
    }

    while iSliceIdx > 0 {
        iSliceIdx -= 1;
        let pSlice = slice_in_layer_mut(pCurDq, iSliceIdx);
        if let Some(pSlice) = pSlice {
            pSlice.iSliceComplexRatio =
                WelsDivRound(INT_MULTIPLY * iAvI[iSliceIdx as usize], iSumAv);
        }
    }
}

/// Statistical decision engine that evaluates whether the timing variance across
/// slices exceeds the core-dependent threshold to justify dynamic slicing.
pub fn NeedDynamicAdjust(pCurDq: &mut SDqLayer, iSliceNum: i32) -> i32 {
    if iSliceNum <= 0 {
        return 0;
    }

    let mut uiTotalConsume: u32 = 0;
    let mut iSliceIdx: i32 = 0;
    let mut iNeedAdj: i32 = 0;

    while iSliceIdx < iSliceNum {
        let Some(pSlice) = slice_in_layer_mut(pCurDq, iSliceIdx) else {
            return 0;
        };
        uiTotalConsume += pSlice.uiSliceConsumeTime;
        iSliceIdx += 1;
    }

    if uiTotalConsume == 0 {
        return 0;
    }

    iSliceIdx = 0;
    let mut fThr = EPSN;
    let mut fRmse = 0.0f32;
    let kfMeanRatio = 1.0f32 / (iSliceNum as f32);

    while let Some(pSlice) = slice_in_layer_mut(pCurDq, iSliceIdx) {
        let fRatio = pSlice.uiSliceConsumeTime as f32 / (uiTotalConsume as f32);
        let fDiffRatio = fRatio - kfMeanRatio;
        fRmse += fDiffRatio * fDiffRatio;
        iSliceIdx += 1;
        if iSliceIdx >= iSliceNum {
            break;
        }
    }

    fRmse = (fRmse / (iSliceNum as f32)).sqrt();
    if iSliceNum >= 8 {
        fThr += THRESHOLD_RMSE_CORE8;
    } else if iSliceNum >= 4 {
        fThr += THRESHOLD_RMSE_CORE4;
    } else if iSliceNum >= 2 {
        fThr += THRESHOLD_RMSE_CORE2;
    } else {
        fThr = 1.0f32;
    }

    if fRmse > fThr {
        iNeedAdj = 1;
    }

    iNeedAdj
}

/// Dynamically recalculates macroblock run-lengths assigned to each slice in a spatial layer.
pub fn DynamicAdjustSlicing(
    pSvcParam: &SWelsSvcCodingParam,
    kpRc: &[SWelsSvcRc],
    pCurDqLayer: &mut SDqLayer,
    iCurDid: i32,
) {
    let pSliceCtx = &mut pCurDqLayer.sSliceEncCtx;
    let kiCountSliceNum = pSliceCtx.iSliceNumInFrame.load(Ordering::Relaxed);
    let kiCountNumMb = pSliceCtx.iMbNumInFrame;
    let mut iMinimalMbNum = pSliceCtx.iMbWidth as i32;
    let mut iMaximalMbNum;
    let mut iMbNumLeft = kiCountNumMb;
    let mut iRunLen = [0i32; MAX_THREADS_NUM];
    let mut iSliceIdx: i32;

    let rc_mode = pSvcParam.iRCMode;
    let kiMbWidth = pSliceCtx.iMbWidth as i32;
    let mut iNumMbInEachGom = 0i32;
    let bAlignToMbRow = rc_mode == RCMode::RC_OFF_MODE
        && kiMbWidth > 0
        && kiMbWidth * kiCountSliceNum < kiCountNumMb;
    if rc_mode != RCMode::RC_OFF_MODE {
        if kpRc.is_empty() {
            return;
        }
        iNumMbInEachGom = kpRc[iCurDid as usize].iNumberMbGom;

        if iNumMbInEachGom <= 0 {
            return;
        }
        if iNumMbInEachGom * kiCountSliceNum >= kiCountNumMb {
            return;
        }
        iMinimalMbNum = iNumMbInEachGom;
    } else {
        if kiCountSliceNum >= kiCountNumMb {
            return;
        } else if iMinimalMbNum * kiCountSliceNum == kiCountNumMb {
            // Exactly one MB row per slice: shifting mid-row would break MB-row
            // alignment between concurrent workers without adding parallelism.
            return;
        } else if iMinimalMbNum * kiCountSliceNum > kiCountNumMb {
            iMinimalMbNum = 1;
        }
    }

    if kiCountSliceNum < 2 || (kiCountSliceNum & 0x01) != 0 {
        return;
    }

    iMaximalMbNum = kiCountNumMb - (kiCountSliceNum - 1) * iMinimalMbNum;
    iSliceIdx = 0;
    while iSliceIdx + 1 < kiCountSliceNum {
        let Some(pSlice) = slice_in_layer_mut(pCurDqLayer, iSliceIdx) else {
            return;
        };
        let mut iNumMbAssigning =
            WelsDivRound(kiCountNumMb * pSlice.iSliceComplexRatio, INT_MULTIPLY);

        if rc_mode != RCMode::RC_OFF_MODE {
            iNumMbAssigning = iNumMbAssigning / iNumMbInEachGom * iNumMbInEachGom;
        } else if bAlignToMbRow {
            iNumMbAssigning = iNumMbAssigning / kiMbWidth * kiMbWidth;
        }

        if iNumMbAssigning < iMinimalMbNum {
            iNumMbAssigning = iMinimalMbNum;
        } else if iNumMbAssigning > iMaximalMbNum {
            iNumMbAssigning = iMaximalMbNum;
        }

        iMbNumLeft -= iNumMbAssigning;
        if iMbNumLeft <= 0 {
            return;
        }
        if (iSliceIdx as usize) < MAX_THREADS_NUM {
            iRunLen[iSliceIdx as usize] = iNumMbAssigning;
        }

        iSliceIdx += 1;
        iMaximalMbNum = iMbNumLeft - (kiCountSliceNum - iSliceIdx - 1) * iMinimalMbNum;
    }

    if (iSliceIdx as usize) < MAX_THREADS_NUM {
        iRunLen[iSliceIdx as usize] = iMbNumLeft;
    }

    let ret = DynamicAdjustSlicePEncCtxAll(pCurDqLayer, &iRunLen);
    pCurDqLayer.bNeedAdjustingSlicing = ret == 0;
}

/// Applies newly calculated macroblock run-lengths to slice context structures.
pub fn DynamicAdjustSlicePEncCtxAll(pCurDq: &mut SDqLayer, pRunLength: &[i32]) -> i32 {
    let pSliceCtx = &mut pCurDq.sSliceEncCtx;
    let iCountNumMbInFrame = pSliceCtx.iMbNumInFrame;
    let iCountSliceNumInFrame = pSliceCtx.iSliceNumInFrame.load(Ordering::Relaxed);
    let mut iSameRunLenFlag = 1i32;
    let mut iFirstMbIdx = 0i32;
    let mut iSliceIdx = 0i32;

    while iSliceIdx < iCountSliceNumInFrame {
        let count: &[i32] = &pCurDq.pCountMbNumInSlice;
        if pRunLength[iSliceIdx as usize] != count[iSliceIdx as usize] {
            iSameRunLenFlag = 0;
            break;
        }
        iSliceIdx += 1;
    }
    if iSameRunLenFlag != 0 {
        return 1;
    }

    iSliceIdx = 0;
    while iSliceIdx < iCountSliceNumInFrame && iFirstMbIdx < iCountNumMbInFrame {
        let kiSliceRun = pRunLength[iSliceIdx as usize];
        {
            let first: &mut Vec<i32> = &mut pCurDq.pFirstMbIdxOfSlice;
            first[iSliceIdx as usize] = iFirstMbIdx;
            let count: &mut Vec<i32> = &mut pCurDq.pCountMbNumInSlice;
            count[iSliceIdx as usize] = kiSliceRun;
        }

        {
            let map: &[AtomicU16] = &pSliceCtx.pOverallMbMap;
            fill_mb_map(map, iFirstMbIdx, kiSliceRun, iSliceIdx as u16);
        }

        iFirstMbIdx += kiSliceRun;
        iSliceIdx += 1;
    }

    0
}

/// Allocates and initializes multithreading synchronization resources and thread-local bitstream buffers.
pub fn RequestMtResource(
    ctx: &mut sWelsEncCtx,
    iCountBsLen: i32,
    _iMaxSliceBufferSize: i32,
    bDynamicSlice: bool,
) -> i32 {
    if iCountBsLen <= 0 {
        return 1;
    }

    let iThreadNum = ctx.param().iMultipleThreadIdc as i32;

    if iThreadNum <= 0 {
        return 1;
    }

    let mut pSmt = Box::new(SSliceThreading::default());

    // The fork is as wide as the buffers, so the buffers are counted from
    // `iMultipleThreadIdc`, already clipped to `[1, MAX_THREADS_NUM]` by
    // `ParamValidationExt`.
    let iThreadBufferNum = (iThreadNum as usize).min(MAX_THREADS_NUM);
    let _ = bDynamicSlice;

    pSmt.uiThreadBsBufferNum = iThreadBufferNum;
    for i in 0..iThreadBufferNum {
        pSmt.pThreadBsBuffer[i] = vec![0u8; iCountBsLen as usize];
    }
    // Spawn `N - 1` background workers: the calling thread runs slot 0 directly
    // via `WorkerPool::for_each_mut`, avoiding oversubscription (`N + 1` runnable
    // threads for `N` jobs) and eliminating the spurious worker wake/re-park cycle
    // every frame. A thread the OS refuses is the same failure as a buffer it refuses.
    pSmt.pool = match WorkerPool::try_new(iThreadBufferNum.saturating_sub(1)) {
        Ok(pool) => pool,
        Err(_) => return 1,
    };
    ctx.pSliceThreading = Some(pSmt);

    0
}

/// Tears down and frees all multithreading objects and bitstream buffers.
pub fn ReleaseMtResource(ctx: &mut sWelsEncCtx) {
    // The take is the whole teardown: the box drops at the end of this function,
    // its `Vec`s, the mutex and the worker pool with it — the pool's `Drop` stops
    // and joins its threads.
    let Some(pSmt) = ctx.pSliceThreading.take() else {
        return;
    };

    drop(pSmt);
}

/// Aggregates individual thread-local slice bitstream buffers into the contiguous frame bitstream buffer.
pub fn AppendSliceToFrameBs(
    pCtx: &mut sWelsEncCtx,
    pLbi: &mut SLayerBSInfo,
    kiSliceCount: i32,
) -> i32 {
    if current_layer_ref(pCtx).is_none() {
        return 0;
    }

    let sWelsEncCtx {
        ppDqLayerList,
        iCurDqLayer,
        pFrameBs,
        iPosBsBuffer,
        iFrameBsSize,
        iEncoderError,
        pOut,
        ..
    } = &mut *pCtx;
    let Some(pOut) = pOut.as_deref_mut() else {
        return 0;
    };
    let SWelsEncoderOutput {
        sNalLen,
        iNalLenBase,
        ..
    } = pOut;
    let Some(pCurDq) = iCurDqLayer
        .and_then(|idx| ppDqLayerList.get_mut(idx.get()))
        .and_then(|l| l.as_deref_mut())
    else {
        return 0;
    };
    let mut iLayerSize = 0i32;
    let mut iNalIdxBase = 0i32;
    pLbi.iNalCount = 0;

    let mut iSliceIdx = 0i32;
    while iSliceIdx < kiSliceCount {
        let Some(pSlice) = slice_in_layer_mut(pCurDq, iSliceIdx) else {
            iSliceIdx += 1;
            continue;
        };
        {
            let pSliceBs = &mut pSlice.sSliceBs;
            if pSliceBs.uiBsPos > 0 {
                let iCountNal = pSliceBs.iNalIndex;

                if (*iPosBsBuffer as u64) + (pSliceBs.uiBsPos as u64) > (*iFrameBsSize as u64) {
                    *iEncoderError |= ENC_RETURN_MEMALLOCERR;
                    return 0;
                }

                if !pFrameBs.is_empty() {
                    if let Some(src) = pSliceBs.pBs.as_ref() {
                        let kiPos = *iPosBsBuffer as usize;
                        let kiLen = pSliceBs.uiBsPos as usize;
                        pFrameBs[kiPos..kiPos + kiLen].copy_from_slice(&src[..kiLen]);
                    }
                }

                *iPosBsBuffer += pSliceBs.uiBsPos as i32;
                iLayerSize += pSliceBs.uiBsPos as i32;

                // `pNalLengthInByte` is the reslice of `pOut.sNalLen` at this
                // layer's base, so the slot is that base plus the running NAL
                // index.
                let mut iNalIdx = 0i32;
                while iNalIdx < iCountNal {
                    let kiSlot = *iNalLenBase + (iNalIdxBase + iNalIdx).max(0) as usize;
                    if kiSlot < sNalLen.len() {
                        sNalLen[kiSlot]
                            .store(pSliceBs.iNalLen[iNalIdx as usize], Ordering::Relaxed);
                    }
                    iNalIdx += 1;
                }
                pLbi.iNalCount += iCountNal;
                iNalIdxBase += iCountNal;
            }
        }
        iSliceIdx += 1;
    }

    iLayerSize
}

/// Encapsulates Raw Byte Sequence Payload (RBSP) data into Annex B NAL units for a slice.
///
/// Takes the slice rather than its `sSliceBs`: the NAL list is offsets into the
/// thread buffer the slice was claimed into, `pThreadBsBuffer[pSlice->uiBufferIdx]`.
pub fn WriteSliceBs(
    pCtx: &sWelsEncCtx,
    pSlice: &mut SSlice,
    _iSliceIdx: i32,
    iSliceSize: &mut i32,
    pSliceBsBuf: &[u8],
) -> i32 {
    if current_layer_ref(pCtx).is_none() {
        return 0;
    }
    let pSliceBs = &mut pSlice.sSliceBs;

    let kiNalCnt = pSliceBs.iNalIndex;
    let mut iNalIdx = 0i32;
    let mut iReturn = ENC_RETURN_SUCCESS;
    let iTotalLeftLength = (pSliceBs.uiBsSize - pSliceBs.uiBsPos) as i32;
    let kpNalHdrExt = &current_layer_expect(pCtx).sLayerInfo.sNalHeaderExt;
    // The write cursor is the slice's own buffer, absent when the slice shares
    // the frame's; `WelsEncodeNal` takes the `INVALIDINPUT` arm for `None`.
    let bHasOwnBuffer = pSliceBs.pBs.is_some();
    let mut iDstPos = 0usize;

    if kiNalCnt > 2 {
        return 0;
    }

    *iSliceSize = 0;
    while iNalIdx < kiNalCnt {
        let mut iNalSize = 0i32;
        // The slice's own buffer is re-sliced from the running offset each
        // iteration; the bound `iTotalLeftLength - *iSliceSize` agrees with it by
        // construction, because `uiBsSize` is that buffer's length.
        let kNalEntry = pSliceBs.sNalList[iNalIdx as usize];
        let kiLeft = (iTotalLeftLength - *iSliceSize).max(0) as usize;
        let pDstTail = if let Some(buf) = pSliceBs.pBs.as_mut() {
            let kiEnd = (iDstPos + kiLeft).min(buf.len());
            Some(&mut buf[iDstPos.min(kiEnd)..kiEnd])
        } else {
            None
        };
        iReturn = WelsEncodeNal(
            &kNalEntry,
            pSliceBsBuf,
            Some(kpNalHdrExt),
            pDstTail,
            &mut iNalSize,
        );

        if iReturn != ENC_RETURN_SUCCESS {
            return iReturn;
        }

        pSliceBs.iNalLen[iNalIdx as usize] = iNalSize;
        *iSliceSize += iNalSize;
        if bHasOwnBuffer {
            iDstPos += iNalSize as usize;
        }
        iNalIdx += 1;
    }
    pSliceBs.uiBsPos = *iSliceSize as u32;

    iReturn
}

/// Queries the operating system for the number of active logical CPU processing cores.
pub fn DynamicDetectCpuCores() -> i32 {
    crate::decoder::decoder_core::GetCPUCount()
}

/// Evaluates load balance and dynamically adjusts slicing for the base spatial dependency layer.
pub fn AdjustBaseLayer(pCtx: &mut sWelsEncCtx) -> i32 {
    if dq_layer_mut(pCtx, 0).is_none() {
        return 0;
    }

    set_current_layer(pCtx, Some(LayerIdx(0)));

    let Some(pCurDq) = dq_layer_mut(pCtx, 0) else {
        return 0;
    };
    let kiSliceNumInFrame = pCurDq.sSliceEncCtx.iSliceNumInFrame.load(Ordering::Relaxed);
    let iNeedAdj = NeedDynamicAdjust(pCurDq, kiSliceNumInFrame);

    if iNeedAdj != 0 {
        let sWelsEncCtx {
            pSvcParam,
            pWelsSvcRc,
            ppDqLayerList,
            ..
        } = &mut *pCtx;
        let Some(pSvcParam) = pSvcParam.as_deref() else {
            return iNeedAdj;
        };
        let Some(pCurDq) = ppDqLayerList.get_mut(0).and_then(|l| l.as_deref_mut()) else {
            return iNeedAdj;
        };
        DynamicAdjustSlicing(pSvcParam, pWelsSvcRc, pCurDq, 0);
    }

    iNeedAdj
}

/// Evaluates load balance and dynamically adjusts slicing for spatial enhancement layers.
pub fn AdjustEnhanceLayer(pCtx: &mut sWelsEncCtx, iCurDid: i32) -> i32 {
    if dq_layer_ref(pCtx, 0).is_none() || current_layer_ref(pCtx).is_none() {
        return 0;
    }

    if pCtx.param_opt().is_none() {
        return 0;
    }
    let kPrevSliceArg = if iCurDid > 0 && (iCurDid as usize - 1) < MAX_SPATIAL_LAYER_NUM {
        let a = &pCtx.param().sSpatialLayers[iCurDid as usize - 1].sSliceArgument;
        Some((a.uiSliceMode, a.uiSliceNum))
    } else {
        None
    };
    let kiMultipleThreadIdc = pCtx.param().iMultipleThreadIdc;

    let kbModelingFromSpatial = current_layer_expect(pCtx).pRefLayer.is_some()
        && match kPrevSliceArg {
            Some((uiSliceMode, uiSliceNum)) => {
                uiSliceMode == SliceMode::SM_FIXEDSLCNUM_SLICE
                    && kiMultipleThreadIdc as u32 >= uiSliceNum
            }
            None => false,
        };

    let iNeedAdj: i32;
    if kbModelingFromSpatial {
        // Hoist the load above `dq_layer_mut` to avoid a conflicting borrow
        // on `pCtx` between `current_layer_expect` and `dq_layer_mut`.
        let kiSliceNumInFrame = current_layer_expect(pCtx)
            .sSliceEncCtx
            .iSliceNumInFrame
            .load(Ordering::Relaxed);
        let Some(pBaseLayer) = dq_layer_mut(pCtx, iCurDid as usize - 1) else {
            return 0;
        };
        iNeedAdj = NeedDynamicAdjust(pBaseLayer, kiSliceNumInFrame);
        if iNeedAdj != 0 {
            let sWelsEncCtx {
                pSvcParam,
                pWelsSvcRc,
                ppDqLayerList,
                iCurDqLayer,
                ..
            } = &mut *pCtx;
            let Some(pSvcParam) = pSvcParam.as_deref() else {
                return iNeedAdj;
            };
            let Some(pCurLayer) = iCurDqLayer
                .and_then(|idx| ppDqLayerList.get_mut(idx.get()))
                .and_then(|l| l.as_deref_mut())
            else {
                return iNeedAdj;
            };
            DynamicAdjustSlicing(pSvcParam, pWelsSvcRc, pCurLayer, iCurDid);
        }
    } else {
        let kiSliceNumInFrame = current_layer_expect(pCtx)
            .sSliceEncCtx
            .iSliceNumInFrame
            .load(Ordering::Relaxed);
        let Some(pCurLayer) = dq_layer_mut(pCtx, iCurDid as usize) else {
            return 0;
        };
        iNeedAdj = NeedDynamicAdjust(pCurLayer, kiSliceNumInFrame);
        if iNeedAdj != 0 {
            let sWelsEncCtx {
                pSvcParam,
                pWelsSvcRc,
                ppDqLayerList,
                iCurDqLayer,
                ..
            } = &mut *pCtx;
            let Some(pSvcParam) = pSvcParam.as_deref() else {
                return iNeedAdj;
            };
            let Some(pCurLayer) = iCurDqLayer
                .and_then(|idx| ppDqLayerList.get_mut(idx.get()))
                .and_then(|l| l.as_deref_mut())
            else {
                return iNeedAdj;
            };
            DynamicAdjustSlicing(pSvcParam, pWelsSvcRc, pCurLayer, iCurDid);
        }
    }

    iNeedAdj
}

// ============================================================================
// Slice Threading Execution Engine
// ============================================================================

/// One worker's share of a frame's slices in fixed/raster slice modes.
///
/// The worker owns bs scratch slot `iBsSlot`, contiguous slices
/// `iFirstSlice..iSliceEnd`, their paired MB windows, and disjoint MB-row
/// picture bands `bands` for the whole scope.
struct FixedSliceJob<'a, 'mb> {
    pCtx: &'a sWelsEncCtx,
    pBsBuf: &'a mut [u8],
    pSlices: &'a mut [SSlice],
    /// Paired 1:1 with `pSlices`; each entry is populated before the fork.
    pMbs: &'a mut [Option<crate::safe::mb_grid::MbWindow<'mb, SMB>>],
    bands: SlicePicBands<'a>,
    /// When executing a 2-wave schedule, `Some((tags, wave_id))` restricts this worker
    /// to slices within `iFirstSlice..iSliceEnd` whose wave tag matches `wave_id`.
    slice_wave_filter: Option<(&'a [u8], u8)>,
    iBsSlot: i32,
    iFirstSlice: i32,
    iSliceEnd: i32,
    bRecordsTime: bool,
    iResult: i32,
}

/// One worker's share of a frame in `SM_SIZELIMITED_SLICE` mode.
///
/// Worker `p` owns partition `p`, bank `p`, bs slot `p`, optional CABAC restore
/// scratch, optional MB window (`None` for empty partitions), and disjoint
/// MB-row picture bands `bands`.
struct SizeLimitedJob<'a> {
    pCtx: &'a sWelsEncCtx,
    pBsBuf: &'a mut [u8],
    pMb: Option<crate::safe::mb_grid::MbWindow<'a, SMB>>,
    bands: SlicePicBands<'a>,
    pDynBsBuf: Option<&'a mut [u8]>,
    pBank: &'a mut SSliceBufferInfo,
    iBsSlot: i32,
    iPartitionIdx: i32,
    iResult: i32,
}

/// RAII restore guard for [`EncodeFixedSlicesForked`]: moves the worker bs
/// scratch buffers, slice bank 0, MB grid, and mutable picture planes out of
/// `sWelsEncCtx` before the fork and restores them unconditionally on return or
/// panic unwind.
struct FixedForkGuard<'ctx> {
    pCtx: &'ctx mut sWelsEncCtx,
    bs_bufs: [Vec<u8>; MAX_THREADS_NUM],
    bs_count: usize,
    bank: Vec<SSlice>,
    mb_data: MbArray<SMB>,
    pics: TakenFramePics,
}

impl<'ctx> FixedForkGuard<'ctx> {
    fn take(pCtx: &'ctx mut sWelsEncCtx, width: usize) -> Self {
        let mut bs_bufs: [Vec<u8>; MAX_THREADS_NUM] = std::array::from_fn(|_| Vec::new());
        let mut bs_count = 0usize;
        if let Some(pSmt) = pCtx.pSliceThreading.as_deref_mut() {
            bs_count = width.min(MAX_THREADS_NUM);
            for (k, slot) in bs_bufs[..bs_count].iter_mut().enumerate() {
                let buf = std::mem::take(&mut pSmt.pThreadBsBuffer[k]);
                debug_assert!(!buf.is_empty(), "job slot {k} has no buffer behind it");
                *slot = buf;
            }
        }
        let (bank, mb_data) = {
            let pCurDq = current_layer_expect_mut(pCtx);
            (
                std::mem::take(&mut pCurDq.sSliceBufferInfo[0].pSliceBuffer),
                std::mem::replace(&mut pCurDq.sMbDataP, MbArray::empty()),
            )
        };
        let pics = TakenFramePics::take(pCtx);
        Self {
            pCtx,
            bs_bufs,
            bs_count,
            bank,
            mb_data,
            pics,
        }
    }
}

impl Drop for FixedForkGuard<'_> {
    fn drop(&mut self) {
        self.pics.restore_mut(self.pCtx);
        if let Some(pSmt) = self.pCtx.pSliceThreading.as_deref_mut() {
            for (k, slot) in self.bs_bufs[..self.bs_count].iter_mut().enumerate() {
                pSmt.pThreadBsBuffer[k] = std::mem::take(slot);
            }
        }
        let pCurDq = current_layer_expect_mut(self.pCtx);
        pCurDq.sSliceBufferInfo[0].pSliceBuffer = std::mem::take(&mut self.bank);
        pCurDq.sMbDataP = std::mem::replace(&mut self.mb_data, MbArray::empty());
    }
}

/// RAII restore guard for [`EncodeSizeLimitedSlicesForked`]: moves the worker
/// bs scratch buffers, CABAC restore buffers, per-partition slice banks, MB
/// grid, and mutable picture planes out of `sWelsEncCtx` and restores them
/// unconditionally on return or panic unwind.
struct SizeLimitedForkGuard<'ctx> {
    pCtx: &'ctx mut sWelsEncCtx,
    width: usize,
    bs_bufs: [Vec<u8>; MAX_THREADS_NUM],
    dyn_bufs: [Vec<u8>; MAX_THREADS_NUM],
    banks: [SSliceBufferInfo; MAX_THREADS_NUM],
    mb_data: MbArray<SMB>,
    pics: TakenFramePics,
}

impl<'ctx> SizeLimitedForkGuard<'ctx> {
    fn take(pCtx: &'ctx mut sWelsEncCtx, width: usize) -> Self {
        let width = width.min(MAX_THREADS_NUM);
        let mut bs_bufs: [Vec<u8>; MAX_THREADS_NUM] = std::array::from_fn(|_| Vec::new());
        if let Some(pSmt) = pCtx.pSliceThreading.as_deref_mut() {
            for (k, slot) in bs_bufs[..width].iter_mut().enumerate() {
                let buf = std::mem::take(&mut pSmt.pThreadBsBuffer[k]);
                debug_assert!(!buf.is_empty(), "job slot {k} has no buffer behind it");
                *slot = buf;
            }
        }
        let mut dyn_bufs: [Vec<u8>; MAX_THREADS_NUM] = std::array::from_fn(|_| Vec::new());
        for (k, slot) in dyn_bufs[..width].iter_mut().enumerate() {
            *slot = std::mem::take(&mut pCtx.pDynamicBsBuffer[k]);
        }
        let mut banks: [SSliceBufferInfo; MAX_THREADS_NUM] =
            std::array::from_fn(|_| SSliceBufferInfo::default());
        let mb_data = {
            let pCurDq = current_layer_expect_mut(pCtx);
            for (k, slot) in banks[..width].iter_mut().enumerate() {
                *slot = std::mem::take(&mut pCurDq.sSliceBufferInfo[k]);
            }
            std::mem::replace(&mut pCurDq.sMbDataP, MbArray::empty())
        };
        let pics = TakenFramePics::take(pCtx);
        Self {
            pCtx,
            width,
            bs_bufs,
            dyn_bufs,
            banks,
            mb_data,
            pics,
        }
    }
}

impl Drop for SizeLimitedForkGuard<'_> {
    fn drop(&mut self) {
        self.pics.restore_mut(self.pCtx);
        if let Some(pSmt) = self.pCtx.pSliceThreading.as_deref_mut() {
            for (k, slot) in self.bs_bufs[..self.width].iter_mut().enumerate() {
                pSmt.pThreadBsBuffer[k] = std::mem::take(slot);
            }
        }
        for (k, slot) in self.dyn_bufs[..self.width].iter_mut().enumerate() {
            self.pCtx.pDynamicBsBuffer[k] = std::mem::take(slot);
        }
        let pCurDq = current_layer_expect_mut(self.pCtx);
        pCurDq.sMbDataP = std::mem::replace(&mut self.mb_data, MbArray::empty());
        for (k, slot) in self.banks[..self.width].iter_mut().enumerate() {
            pCurDq.sSliceBufferInfo[k] = std::mem::take(slot);
        }
    }
}

/// The prefix-NAL pair both encode bodies open with
/// (`CWelsBaseTask::WritePrefixNal`).
fn WritePrefixNalForSlice(
    _pCtx: &sWelsEncCtx,
    pSlice: &mut SSlice,
    eNalRefIdc: EWelsNalRefIdc,
    eNalType: EWelsNalUnitType,
    pSliceBsBuf: &mut [u8],
) {
    if eNalRefIdc != EWelsNalRefIdc::NRI_PRI_LOWEST {
        WelsLoadNalForSlice(
            &mut pSlice.sSliceBs,
            EWelsNalUnitType::NAL_UNIT_PREFIX as i32,
            eNalRefIdc as i32,
        );
        let buf = pSliceBsBuf;
        WelsWriteSVCPrefixNal(
            buf,
            &mut pSlice.sSliceBs.sBsWrite,
            eNalRefIdc as i32,
            EWelsNalUnitType::NAL_UNIT_CODED_SLICE_IDR == eNalType,
        );
        WelsUnloadNalForSlice(&mut pSlice.sSliceBs);
    } else {
        // No Prefix NAL Unit RBSP syntax here, but need add NAL Unit Header extension
        WelsLoadNalForSlice(
            &mut pSlice.sSliceBs,
            EWelsNalUnitType::NAL_UNIT_PREFIX as i32,
            eNalRefIdc as i32,
        );
        WelsUnloadNalForSlice(&mut pSlice.sSliceBs);
    }
}

/// One slice, start to finish, on a worker thread: the `InitTask` / `ExecuteTask`
/// / `FinishTask` triple of `CWelsSliceEncodingTask`.
///
/// Returns `ENC_RETURN_SUCCESS` or the first error encountered during
/// initialization, coding, or bitstream encapsulation.
#[allow(clippy::too_many_arguments)]
fn EncodeOneSliceInJob(
    pCtx: &sWelsEncCtx,
    iSliceIdx: i32,
    iBsSlot: i32,
    bRecordsTime: bool,
    pSlotBuf: &mut [u8],
    pSlices: &mut [SSlice],
    // This worker's macroblock windows, one per slice it codes, indexed exactly
    // as `pSlices` is.
    pMbs: &mut [Option<crate::safe::mb_grid::MbWindow<'_, SMB>>],
    bands: &mut SlicePicBands<'_>,
    iFirstSlice: i32,
) -> i32 {
    // ---- CWelsSliceEncodingTask::InitTask
    let eNalType = pCtx.eNalType;
    let eNalRefIdc = pCtx.eNalPriority;
    let bNeedPrefix = pCtx.bNeedPrefixNalFlag;

    // The slice comes from this worker's contiguous range `iFirstSlice..iSliceEnd`.
    let kiLocal = (iSliceIdx - iFirstSlice) as usize;
    let Some(pSlice) = pSlices.get_mut(kiLocal) else {
        return ENC_RETURN_UNEXPECTED;
    };
    // Move the window onto this worker's stack rather than borrowing it in place
    // inside `sMbWindowStack`, avoiding cross-core false sharing when `set_cur`
    // mutates `cur` on every macroblock.
    let Some(mut mb_run) = pMbs.get_mut(kiLocal).and_then(Option::take) else {
        return ENC_RETURN_UNEXPECTED;
    };
    InitOneSliceInThread(pCtx, pSlice, iBsSlot, iSliceIdx);
    let iReturn = SetSliceBoundaryInfo(current_layer_ref(pCtx), pSlice, iSliceIdx);
    if iReturn != ENC_RETURN_SUCCESS {
        return iReturn;
    }
    pSlice.sSliceBs.sBsWrite = BsWriter::new();
    let iSliceStart = if bRecordsTime { WelsTime() } else { 0 };

    // The slice's bitstream buffer is the job's partition slot, subsliced to the
    // claimed size; `uiSize` cannot exceed the slot (both are `iCountBsLen`,
    // single writer). The `pOut` writer option is `None` here: every slice of a
    // forked layer has its own writer.
    debug_assert_eq!(
        pSlice.uiBufferIdx as i32, iBsSlot,
        "the slice's claimed slot is this job's"
    );
    let kuiSize = pSlice.sSliceBs.uiSize;
    let pSliceBsBuf = &mut pSlotBuf[..kuiSize as usize];
    let mut pCtxOutBs: Option<&mut BsWriter> = None;

    // ---- CWelsSliceEncodingTask::ExecuteTask
    let iResult = (|| {
        if bNeedPrefix {
            WritePrefixNalForSlice(pCtx, pSlice, eNalRefIdc, eNalType, &mut *pSliceBsBuf);
        }
        WelsLoadNalForSlice(&mut pSlice.sSliceBs, eNalType as i32, eNalRefIdc as i32);
        debug_assert_eq!(iSliceIdx, pSlice.iSliceIdx);
        // `None` restore scratch: the fixed loops never use it. `None`
        // next-slice too — the fixed modes never hit the dynamic boundary, so
        // `AddSliceBoundary` never fires here.
        let mut iReturn = WelsCodeOneSlice(
            pCtx,
            pSlice,
            eNalType as i32,
            &mut *pSliceBsBuf,
            &mut pCtxOutBs,
            &mut mb_run,
            bands,
            None,
            None,
        );
        if ENC_RETURN_SUCCESS != iReturn {
            return iReturn;
        }
        WelsUnloadNalForSlice(&mut pSlice.sSliceBs);

        let mut iSliceSize = 0i32;
        iReturn = WriteSliceBs(pCtx, pSlice, iSliceIdx, &mut iSliceSize, &*pSliceBsBuf);
        if ENC_RETURN_SUCCESS != iReturn {
            return iReturn;
        }

        if let Some(pfDeblockingFilterSlice) = pCtx.func_list().pfDeblocking.pfDeblockingFilterSlice
        {
            // The walker's window is the worker's own carved run, the same one
            // the coding chain just wrote through. `uiFilterIdc == 1` (MT
            // validation rewrites idc 0 -> 2) confines walk and neighbour reads
            // to this slice, hence inside this worker's MB-row band.
            let pCurDq = current_layer_expect(pCtx);
            pfDeblockingFilterSlice(
                &mut bands.rec,
                &pCurDq.sSliceEncCtx,
                &pCurDq.iCsStride,
                pSlice,
                &mut mb_run,
            );
        }
        ENC_RETURN_SUCCESS
    })();

    // ---- CWelsSliceEncodingTask::FinishTask (the load-balancing override's half)
    if bRecordsTime {
        pSlice.uiSliceConsumeTime = (WelsTime() - iSliceStart).max(0) as u32;
    }

    iResult
}

/// How many workers a fork gets: never more than there are slices to encode, and
/// never more than there are bs scratch buffers behind the slots.
fn ForkWidth(pCtx: &mut sWelsEncCtx, iItemCount: i32) -> i32 {
    let iBuffers = pCtx
        .pSliceThreading
        .as_deref()
        .map_or(1, |pSmt| pSmt.uiThreadBsBufferNum as i32);
    iItemCount.min(iBuffers.max(1)).max(1)
}

/// One worker's contiguous slice span and disjoint MB-row band within a wave:
/// `(slice_start, slice_end, row_start, row_end)`.
type WorkerPlan = (usize, usize, usize, usize);

#[derive(Clone, Copy)]
struct SliceGroup {
    slice_start: usize,
    slice_end: usize,
    row_start: usize,
    row_end: usize,
    mb_count: i32,
}

/// Fixed-slice execution schedule produced by [`plan_fixed_slice_schedule`]:
///
/// - **1 wave** (the fast path for all MB-row-aligned slices and any layout where
///   single-wave merging already matches or beats 2-wave critical-path work).
/// - **2 waves** (when mid-row boundaries — e.g. custom `SM_RASTER_SLICE` or tiny-height
///   `SM_FIXEDSLCNUM_SLICE` — would otherwise chain into fewer workers, alternating
///   non-adjacent slices into two disjoint waves cuts critical-path work while preserving
///   exact bitstream output).
struct FixedSliceSchedule {
    waves: [[WorkerPlan; MAX_THREADS_NUM]; 2],
    wave_widths: [usize; 2],
    num_waves: usize,
    slice_wave_stack: [u8; MAX_SLICES_NUM],
    slice_wave_heap: Vec<u8>,
}

impl FixedSliceSchedule {
    fn max_width(&self) -> usize {
        self.wave_widths[..self.num_waves]
            .iter()
            .copied()
            .max()
            .unwrap_or(1)
    }

    fn slice_waves(&self, slice_count: usize) -> &[u8] {
        if !self.slice_wave_heap.is_empty() {
            &self.slice_wave_heap[..slice_count]
        } else {
            &self.slice_wave_stack[..slice_count]
        }
    }
}

fn distribute_slice_groups(
    groups: &[SliceGroup],
    cap: usize,
) -> ([WorkerPlan; MAX_THREADS_NUM], usize, i32) {
    let g_len = groups.len();
    let num_workers = g_len.min(cap).max(1);
    let base = g_len / num_workers;
    let rem = g_len % num_workers;
    let mut plans = [(0usize, 0usize, 0usize, 0usize); MAX_THREADS_NUM];
    let mut g_idx = 0usize;
    let mut max_mbs = 0i32;
    for (w, slot) in plans[..num_workers].iter_mut().enumerate() {
        let g_cnt = base + usize::from(w < rem);
        let g_first = groups[g_idx];
        let g_last = groups[g_idx + g_cnt - 1];
        let mut w_mbs = 0i32;
        for g in &groups[g_idx..g_idx + g_cnt] {
            w_mbs += g.mb_count;
        }
        max_mbs = max_mbs.max(w_mbs);
        *slot = (
            g_first.slice_start,
            g_last.slice_end,
            g_first.row_start,
            g_last.row_end,
        );
        g_idx += g_cnt;
    }
    (plans, num_workers, max_mbs)
}

/// Plans 1 or 2 waves of disjoint MB-row-aligned worker bands for raster-ordered slices
/// across at most `max_workers` workers (`<= MAX_THREADS_NUM`).
///
/// Returns `None` if the slice ranges are empty, malformed, or not in disjoint raster
/// order. Avoids heap allocation whenever `first_mbs.len() <= MAX_SLICES_NUM`.
fn plan_fixed_slice_schedule(
    first_mbs: &[i32],
    counts: &[i32],
    mb_width: usize,
    max_workers: usize,
) -> Option<FixedSliceSchedule> {
    let n = first_mbs.len();
    if n == 0 || counts.len() < n || mb_width == 0 {
        return None;
    }
    let zero_group = SliceGroup {
        slice_start: 0,
        slice_end: 0,
        row_start: 0,
        row_end: 0,
        mb_count: 0,
    };
    let mut stack_groups = [zero_group; MAX_SLICES_NUM];
    let mut heap_groups: Vec<SliceGroup> = Vec::new();
    let groups_buf: &mut [SliceGroup] = if n <= MAX_SLICES_NUM {
        &mut stack_groups[..n]
    } else {
        heap_groups.resize(n, zero_group);
        &mut heap_groups[..]
    };

    // Pass 1: validate raster order and build single-wave merged groups.
    let mut group_len = 0usize;
    let mut cursor = 0i32;
    for i in 0..n {
        let first = first_mbs[i];
        let count = counts[i];
        if first < cursor || count <= 0 {
            return None;
        }
        let row_start = (first as usize) / mb_width;
        let row_end = ((first + count - 1) as usize) / mb_width + 1;
        if group_len > 0 && groups_buf[group_len - 1].row_end > row_start {
            let last = &mut groups_buf[group_len - 1];
            last.slice_end = i + 1;
            last.row_end = last.row_end.max(row_end);
            last.mb_count += count;
        } else {
            groups_buf[group_len] = SliceGroup {
                slice_start: i,
                slice_end: i + 1,
                row_start,
                row_end,
                mb_count: count,
            };
            group_len += 1;
        }
        cursor = first + count;
    }

    let cap = max_workers.clamp(1, MAX_THREADS_NUM);
    let (single_plans, single_workers, single_crit) =
        distribute_slice_groups(&groups_buf[..group_len], cap);

    let mut sched = FixedSliceSchedule {
        waves: [single_plans, [(0, 0, 0, 0); MAX_THREADS_NUM]],
        wave_widths: [single_workers, 0],
        num_waves: 1,
        slice_wave_stack: [0u8; MAX_SLICES_NUM],
        slice_wave_heap: Vec::new(),
    };

    // Fast path: no row overlaps, single wave already saturates `cap`, or 1 worker cap.
    if cap <= 1 || group_len == n || single_workers >= cap {
        return Some(sched);
    }

    // Pass 2: try a 2-wave alternating schedule so slices sharing a boundary row can run
    // in separate waves while non-adjacent slices run concurrently within each wave.
    let mut stack_w0 = [zero_group; MAX_SLICES_NUM];
    let mut stack_w1 = [zero_group; MAX_SLICES_NUM];
    let mut heap_w0: Vec<SliceGroup> = Vec::new();
    let mut heap_w1: Vec<SliceGroup> = Vec::new();
    let (w0_buf, w1_buf): (&mut [SliceGroup], &mut [SliceGroup]) = if n <= MAX_SLICES_NUM {
        (&mut stack_w0[..n], &mut stack_w1[..n])
    } else {
        heap_w0.resize(n, zero_group);
        heap_w1.resize(n, zero_group);
        (&mut heap_w0[..], &mut heap_w1[..])
    };
    let mut wave_tag_stack = [0u8; MAX_SLICES_NUM];
    let mut wave_tag_heap: Vec<u8> = Vec::new();
    let wave_tags: &mut [u8] = if n <= MAX_SLICES_NUM {
        &mut wave_tag_stack[..n]
    } else {
        wave_tag_heap.resize(n, 0);
        &mut wave_tag_heap[..]
    };

    let mut w_len = [0usize; 2];
    let mut last_wave = 0usize;
    for i in 0..n {
        let first = first_mbs[i];
        let count = counts[i];
        let row_start = (first as usize) / mb_width;
        let row_end = ((first + count - 1) as usize) / mb_width + 1;

        let fits0 = w_len[0] == 0 || w0_buf[w_len[0] - 1].row_end <= row_start;
        let fits1 = w_len[1] == 0 || w1_buf[w_len[1] - 1].row_end <= row_start;

        let chosen = if fits0 && fits1 {
            let w = usize::from(w_len[1] < w_len[0]);
            let buf = if w == 0 { &mut *w0_buf } else { &mut *w1_buf };
            buf[w_len[w]] = SliceGroup {
                slice_start: i,
                slice_end: i + 1,
                row_start,
                row_end,
                mb_count: count,
            };
            w_len[w] += 1;
            w
        } else if fits0 {
            w0_buf[w_len[0]] = SliceGroup {
                slice_start: i,
                slice_end: i + 1,
                row_start,
                row_end,
                mb_count: count,
            };
            w_len[0] += 1;
            0
        } else if fits1 {
            w1_buf[w_len[1]] = SliceGroup {
                slice_start: i,
                slice_end: i + 1,
                row_start,
                row_end,
                mb_count: count,
            };
            w_len[1] += 1;
            1
        } else {
            // Overlaps both waves (e.g. >=3 slices in the same MB row): merge into
            // `last_wave`'s last group, whose `slice_end == i`, keeping spans contiguous.
            let buf = if last_wave == 0 {
                &mut *w0_buf
            } else {
                &mut *w1_buf
            };
            let last = &mut buf[w_len[last_wave] - 1];
            last.slice_end = i + 1;
            last.row_end = last.row_end.max(row_end);
            last.mb_count += count;
            last_wave
        };
        wave_tags[i] = chosen as u8;
        last_wave = chosen;
    }

    if w_len[0] > 0 && w_len[1] > 0 {
        let (p0, nw0, crit0) = distribute_slice_groups(&w0_buf[..w_len[0]], cap);
        let (p1, nw1, crit1) = distribute_slice_groups(&w1_buf[..w_len[1]], cap);
        if crit0 + crit1 < single_crit {
            sched.waves = [p0, p1];
            sched.wave_widths = [nw0, nw1];
            sched.num_waves = 2;
            sched.slice_wave_stack = wave_tag_stack;
            sched.slice_wave_heap = wave_tag_heap;
        }
    }

    Some(sched)
}

#[cfg(test)]
fn plan_fixed_slice_workers(
    slice_ranges: &[(i32, i32)],
    mb_width: usize,
    max_workers: usize,
) -> Option<Vec<Vec<WorkerPlan>>> {
    let firsts: Vec<i32> = slice_ranges.iter().map(|&(f, _)| f).collect();
    let counts: Vec<i32> = slice_ranges.iter().map(|&(_, c)| c).collect();
    let sched = plan_fixed_slice_schedule(&firsts, &counts, mb_width, max_workers)?;
    Some(
        (0..sched.num_waves)
            .map(|w| sched.waves[w][..sched.wave_widths[w]].to_vec())
            .collect(),
    )
}

/// The fork/join for every fixed slice mode —
/// `pTaskManage->ExecuteTasks(WELS_ENC_TASK_ENCODING)` for
/// `uiSliceMode != SM_SIZELIMITED_SLICE`.
///
/// Returns the value `FinishTask` would have ORed into `pCtx->iEncoderError`; the
/// caller ORs it.
pub fn EncodeFixedSlicesForked(pCtx: &mut sWelsEncCtx, kiSliceCount: i32) -> i32 {
    if kiSliceCount <= 0 || pCtx.pSliceThreading.is_none() {
        return ENC_RETURN_SUCCESS;
    }
    let bRecordsTime = pCtx.param_opt().is_some() && pCtx.param().bUseLoadBalancing;
    let iMaxWorkers = ForkWidth(pCtx, kiSliceCount);
    if iMaxWorkers > 1 {
        if let Some(pCurDq) = current_layer_ref(pCtx) {
            debug_assert!(
                !pCurDq.bDeblockingParallelFlag || pCurDq.iLoopFilterDisableIdc != 0,
                "parallel slice deblocking requires slice-boundary filtering to be disabled (iLoopFilterDisableIdc != 0)"
            );
            if pCurDq.bDeblockingParallelFlag && pCurDq.iLoopFilterDisableIdc == 0 {
                return ENC_RETURN_UNEXPECTED;
            }
        }
    }

    // Read the finalized slice ranges and plan 1 or 2 waves of disjoint MB-row-aligned
    // worker bands before taking any buffers out of `pCtx`.
    let kiSliceCountU = kiSliceCount as usize;
    let (schedule, kiGridWidth, kiMbWidth) = {
        let pCurDq = current_layer_expect(pCtx);
        if pCurDq.pFirstMbIdxOfSlice.len() < kiSliceCountU
            || pCurDq.pCountMbNumInSlice.len() < kiSliceCountU
        {
            return ENC_RETURN_UNEXPECTED;
        }
        let kiMbWidth = pCurDq.iMbWidth as usize;
        let Some(sched) = plan_fixed_slice_schedule(
            &pCurDq.pFirstMbIdxOfSlice[..kiSliceCountU],
            &pCurDq.pCountMbNumInSlice[..kiSliceCountU],
            kiMbWidth,
            iMaxWorkers as usize,
        ) else {
            return ENC_RETURN_UNEXPECTED;
        };
        (sched, pCurDq.sMbDataP.dims().mb_width(), kiMbWidth)
    };

    // Hoisted out of the fork: `sLayerInfo.sNalHeaderExt.bIdrFlag` is the same
    // constant on every worker and no worker reads it before its own write, so it is
    // stamped once here, on the calling thread, before anything spawns.
    crate::encoder::svc_encode_slice::StampLayerIdrFlagForSliceType(pCtx);

    // Move out the worker bs scratch buffers, slice bank, MB grid, and mutable
    // picture planes under an RAII guard that restores all of them on return or
    // panic unwind.
    let mut guard = FixedForkGuard::take(pCtx, schedule.max_width());

    let mut iErr = ENC_RETURN_SUCCESS;
    {
        let FixedForkGuard {
            pCtx,
            bs_bufs,
            bank,
            mb_data,
            pics,
            ..
        } = &mut guard;
        let pCtx: &sWelsEncCtx = pCtx;
        let pCurDq = current_layer_expect(pCtx);

        // Peel one `MbWindow` per slice in raster order into a flat stack array
        // (no heap allocation for `kiSliceCount <= MAX_SLICES_NUM`).
        let mut sMbWindowStack: [Option<crate::safe::mb_grid::MbWindow<'_, SMB>>; MAX_SLICES_NUM] =
            std::array::from_fn(|_| None);
        let mut vMbWindowHeap: Vec<Option<crate::safe::mb_grid::MbWindow<'_, SMB>>> = Vec::new();
        let pAllMbWindows: &mut [Option<crate::safe::mb_grid::MbWindow<'_, SMB>>] =
            if kiSliceCountU <= MAX_SLICES_NUM {
                &mut sMbWindowStack[..kiSliceCountU]
            } else {
                vMbWindowHeap.resize_with(kiSliceCountU, || None);
                &mut vMbWindowHeap[..]
            };
        {
            let mut rest: &mut [SMB] = mb_data.as_mut_slice();
            let mut cursor = 0i32;
            for s_idx in 0..kiSliceCountU {
                let first = pCurDq.pFirstMbIdxOfSlice[s_idx];
                let count = pCurDq.pCountMbNumInSlice[s_idx];
                if first < cursor
                    || count <= 0
                    || (first - cursor) as usize + count as usize > rest.len()
                {
                    iErr |= ENC_RETURN_UNEXPECTED;
                    break;
                }
                let (_gap, tail) = rest.split_at_mut((first - cursor) as usize);
                let (chunk, tail) = tail.split_at_mut(count as usize);
                pAllMbWindows[s_idx] = Some(crate::safe::mb_grid::MbWindow::new(
                    chunk,
                    first as usize,
                    kiGridWidth,
                    first as usize,
                ));
                rest = tail;
                cursor = first + count;
            }
        }

        if iErr == ENC_RETURN_SUCCESS {
            let bank_len = bank.len().min(kiSliceCountU);
            for wave_idx in 0..schedule.num_waves {
                let w_count = schedule.wave_widths[wave_idx];
                let wave_plans = &schedule.waves[wave_idx][..w_count];
                let slice_wave_filter: Option<(&[u8], u8)> = if schedule.num_waves > 1 {
                    Some((schedule.slice_waves(kiSliceCountU), wave_idx as u8))
                } else {
                    None
                };

                let mut vPerWorker: [&mut [SSlice]; MAX_THREADS_NUM] =
                    std::array::from_fn(|_| &mut [][..]);
                {
                    let mut bank_rest = &mut bank[..bank_len];
                    let mut bank_cursor = 0usize;
                    for (&(s_start, s_end, _, _), slot) in
                        wave_plans.iter().zip(vPerWorker[..w_count].iter_mut())
                    {
                        let skip = s_start.saturating_sub(bank_cursor).min(bank_rest.len());
                        let (_, after_skip) = bank_rest.split_at_mut(skip);
                        let take_n = (s_end - s_start).min(after_skip.len());
                        let (chunk, tail) = after_skip.split_at_mut(take_n);
                        *slot = chunk;
                        bank_rest = tail;
                        bank_cursor = s_start + take_n;
                    }
                }

                let mut vRowRanges = [(0usize, 0usize); MAX_THREADS_NUM];
                for (dst, &(_, _, r0, r1)) in vRowRanges[..w_count].iter_mut().zip(wave_plans) {
                    *dst = (r0, r1);
                }
                let mut vBands: [SlicePicBands<'_>; MAX_THREADS_NUM] =
                    std::array::from_fn(|_| SlicePicBands::empty());
                pics.split_bands_into(&vRowRanges[..w_count], kiMbWidth, &mut vBands[..w_count]);

                let mut jobs: [Option<FixedSliceJob<'_, '_>>; MAX_THREADS_NUM] =
                    std::array::from_fn(|_| None);
                let mut mb_rest: &mut [Option<crate::safe::mb_grid::MbWindow<'_, SMB>>] =
                    &mut *pAllMbWindows;
                let mut mb_cursor = 0usize;
                for (k, ((((&plan, buf), slices), bands), job_slot)) in wave_plans
                    .iter()
                    .zip(bs_bufs[..w_count].iter_mut())
                    .zip(vPerWorker[..w_count].iter_mut())
                    .zip(vBands[..w_count].iter_mut())
                    .zip(jobs[..w_count].iter_mut())
                    .enumerate()
                {
                    let (s_start, s_end, _, _) = plan;
                    let skip = s_start.saturating_sub(mb_cursor);
                    let (_, after_skip) = mb_rest.split_at_mut(skip);
                    let (w_mbs, tail) = after_skip.split_at_mut(s_end - s_start);
                    mb_rest = tail;
                    mb_cursor = s_end;
                    debug_assert!(
                        pCtx.pSliceThreading
                            .as_deref()
                            .is_some_and(|pSmt| k < pSmt.uiThreadBsBufferNum),
                        "job slot {k} is outside the allocated bs buffers"
                    );
                    *job_slot = Some(FixedSliceJob {
                        pCtx,
                        pBsBuf: buf.as_mut_slice(),
                        pSlices: std::mem::take(slices),
                        pMbs: w_mbs,
                        bands: std::mem::replace(bands, SlicePicBands::empty()),
                        slice_wave_filter,
                        iBsSlot: k as i32,
                        iFirstSlice: s_start as i32,
                        iSliceEnd: s_end as i32,
                        bRecordsTime,
                        iResult: ENC_RETURN_SUCCESS,
                    });
                }

                if let Some(pSmt) = pCtx.pSliceThreading.as_deref() {
                    let all_ok = pSmt.pool.for_each_mut(&mut jobs[..w_count], |slot| {
                        let Some(job) = slot.as_mut() else {
                            return;
                        };
                        let mut iWorkerErr = ENC_RETURN_SUCCESS;
                        let mut iSliceIdx = job.iFirstSlice;
                        while iSliceIdx < job.iSliceEnd {
                            if let Some((tags, wave_id)) = job.slice_wave_filter {
                                if tags[iSliceIdx as usize] != wave_id {
                                    iSliceIdx += 1;
                                    continue;
                                }
                            }
                            let r = EncodeOneSliceInJob(
                                job.pCtx,
                                iSliceIdx,
                                job.iBsSlot,
                                job.bRecordsTime,
                                job.pBsBuf,
                                job.pSlices,
                                job.pMbs,
                                &mut job.bands,
                                job.iFirstSlice,
                            );
                            if r != ENC_RETURN_SUCCESS {
                                iWorkerErr |= r;
                            }
                            iSliceIdx += 1;
                        }
                        job.iResult = iWorkerErr;
                    });
                    if !all_ok {
                        iErr |= ENC_RETURN_UNEXPECTED;
                    }
                    for job in jobs[..w_count].iter().flatten() {
                        iErr |= job.iResult;
                    }
                }
                if iErr != ENC_RETURN_SUCCESS {
                    break;
                }
            }
        }
    }

    drop(guard);
    iErr
}

/// The fork/join for the macroblock-map update — `pTaskManage->InitFrame`'s
/// `WELS_ENC_TASK_UPDATEMBMAP`.
///
/// A separate fork/join that fully joins before the encoding one starts. The two
/// are not fused: fusing them would let a slice encode against a neighbour map
/// another worker had not finished writing.
///
/// It fires only when `bNeedAdjustingSlicing` is set, which only
/// `DynamicAdjustSlicing` does — so this path is reachable only with
/// `bUseLoadBalancing` on.
#[must_use]
pub fn UpdateMbMapForked(pCtx: &mut sWelsEncCtx, kiTaskCount: i32) -> i32 {
    if kiTaskCount <= 0 || current_layer_ref(pCtx).is_none() || pCtx.pSliceThreading.is_none() {
        return ENC_RETURN_SUCCESS;
    }
    let iWidth = ForkWidth(pCtx, kiTaskCount);
    // The layer and the pool are disjoint fields of the context, so the layer's
    // `&mut` and the pool's `&` coexist — `current_layer_mut`, spelled out.
    let sWelsEncCtx {
        ppDqLayerList,
        iCurDqLayer,
        pSliceThreading,
        ..
    } = &mut *pCtx;
    let Some(pSmt) = pSliceThreading.as_deref() else {
        return ENC_RETURN_SUCCESS;
    };
    let pool = &pSmt.pool;
    let Some(pCurDq) = iCurDqLayer
        .and_then(|idx| ppDqLayerList.get_mut(idx.get()))
        .and_then(|l| l.as_deref_mut())
    else {
        return ENC_RETURN_SUCCESS;
    };

    // The pre-fork partition: each slice's records are the contiguous run
    // `[pFirstMbIdxOfSlice[i] .. + pCountMbNumInSlice[i])`, and the runs are
    // disjoint, so a chain of `split_at_mut` carves them. The grid is carved here,
    // on the calling thread, while the layer is `&mut`, and each worker is handed
    // its own chunks. Nothing crosses the spawn but `&mut [SMB]` and a shared
    // `&SSliceCtx`.
    let SDqLayer {
        sMbDataP,
        sSliceEncCtx,
        pFirstMbIdxOfSlice,
        pCountMbNumInSlice,
        ..
    } = pCurDq;
    let kiMbWidth = sSliceEncCtx.iMbWidth as i32;
    let kiGridWidth = sMbDataP.dims().mb_width();

    // The ranges, in macroblock order: an empty slice contributes no chunk.
    // Avoids heap allocation when `kiTaskCount <= MAX_SLICES_NUM`.
    let kiTaskCountU = kiTaskCount as usize;
    let mut stack_ranges = [(0i32, 0i32, 0i32); MAX_SLICES_NUM];
    let mut heap_ranges: Vec<(i32, i32, i32)> = Vec::new();
    let ranges_buf: &mut [(i32, i32, i32)] = if kiTaskCountU <= MAX_SLICES_NUM {
        &mut stack_ranges[..kiTaskCountU]
    } else {
        heap_ranges.resize(kiTaskCountU, (0, 0, 0));
        &mut heap_ranges[..]
    };
    let mut range_len = 0usize;
    for idc in 0..kiTaskCount {
        let Some(&first) = pFirstMbIdxOfSlice.get(idc as usize) else {
            continue;
        };
        let Some(&count) = pCountMbNumInSlice.get(idc as usize) else {
            continue;
        };
        if count > 0 && first >= 0 {
            ranges_buf[range_len] = (first, count, idc);
            range_len += 1;
        }
    }
    if range_len == 0 {
        return ENC_RETURN_SUCCESS;
    }
    let ranges = &mut ranges_buf[..range_len];
    ranges.sort_unstable();

    // One `&mut [SMB]` per slice, peeled off the front in order into a flat stack
    // array (no heap allocation for `range_len <= MAX_SLICES_NUM`), then split
    // contiguously across workers.
    type MbMapTask<'a> = (i32, i32, i32, &'a mut [SMB]);
    let mut stack_tasks: [MbMapTask<'_>; MAX_SLICES_NUM] =
        std::array::from_fn(|_| (0, 0, 0, &mut [][..]));
    let mut heap_tasks: Vec<MbMapTask<'_>> = Vec::new();
    let tasks_buf: &mut [MbMapTask<'_>] = if range_len <= MAX_SLICES_NUM {
        &mut stack_tasks[..range_len]
    } else {
        heap_tasks.resize_with(range_len, || (0, 0, 0, &mut [][..]));
        &mut heap_tasks[..]
    };

    let mut rest: &mut [SMB] = sMbDataP.as_mut_slice();
    let mut cursor = 0i32;
    for (&(first, count, idc), task_slot) in ranges.iter().zip(tasks_buf.iter_mut()) {
        debug_assert!(
            first >= cursor,
            "slice {idc} starts at mb {first}, inside the previous slice's records \
             (which end at {cursor}) — the macroblock partition is not disjoint"
        );
        if first < cursor || count <= 0 || (first - cursor) as usize + count as usize > rest.len() {
            return ENC_RETURN_UNEXPECTED;
        }
        let (_gap, tail) = rest.split_at_mut((first - cursor) as usize);
        let (chunk, tail) = tail.split_at_mut(count as usize);
        *task_slot = (idc, first, count, chunk);
        rest = tail;
        cursor = first + count;
    }

    let n_workers = (iWidth as usize).min(MAX_THREADS_NUM).min(range_len).max(1);
    let mut per_worker: [&mut [MbMapTask<'_>]; MAX_THREADS_NUM] =
        std::array::from_fn(|_| &mut [][..]);
    let base = range_len / n_workers;
    let rem = range_len % n_workers;
    let mut task_rest = tasks_buf;
    for (w, slot) in per_worker[..n_workers].iter_mut().enumerate() {
        let take = base + usize::from(w < rem);
        let (chunk, tail) = task_rest.split_at_mut(take);
        *slot = chunk;
        task_rest = tail;
    }

    let pSliceCtx: &SSliceCtx = sSliceEncCtx;
    let all_ok = pool.for_each_mut(&mut per_worker[..n_workers], |group| {
        for &mut (idc, first, count, ref mut chunk) in group.iter_mut() {
            let mut mbs = crate::safe::mb_grid::MbWindow::new(
                chunk,
                first as usize,
                kiGridWidth,
                first as usize,
            );
            UpdateMbListNeighborParallel(&mut mbs, pSliceCtx, kiMbWidth, idc, first, count);
        }
    });
    if all_ok {
        ENC_RETURN_SUCCESS
    } else {
        ENC_RETURN_UNEXPECTED
    }
}

/// One partition's slices under `SM_SIZELIMITED_SLICE` —
/// `CWelsConstrainedSizeSlicingEncodingTask`'s `InitTask` / `ExecuteTask` /
/// `FinishTask`, without the two mutexes the fork/join makes unnecessary.
///
/// Nothing here depends on the schedule:
///
/// * there are exactly `iActiveThreadsNum` tasks, and task `t` works partition
///   `t % iActiveThreadsNum`, i.e. `t` — partition is a static function of the task
///   index;
/// * the slice indices a partition produces are `t`, `t + N`, `t + 2N`, ... with
///   `N = iActiveThreadsNum`, a static arithmetic progression stamped into
///   `SSlice::iSliceIdx`;
/// * `ReOrderSliceInLayer` (after the join, on the calling thread) recovers the
///   layer position from that stamp alone — `iSliceIdx % N` gives the partition and
///   `iSliceIdx / N` the position within it — never from which bank held the slice.
///
/// Worker `p` owns partition `p`, bank `p` and bs slot `p`, which also makes
/// `NumSliceCodedOfPartition[p]` and `LastCodedMbIdxOfPartition[p]` — written from
/// inside the encode — disjoint by construction.
#[allow(clippy::too_many_arguments)]
fn EncodeOnePartitionSizeLimited(
    pCtx: &sWelsEncCtx,
    iPartitionIdx: i32,
    iBsSlot: i32,
    pSlotBuf: &mut [u8],
    // This partition's records, moved onto the worker's stack so `set_cur`
    // mutates a thread-local `MbWindow` rather than the caller's job array.
    // A partition whose span is zero codes nothing and carries `None`:
    // `MbWindow`'s invariant is that a window always has a current macroblock,
    // so the empty partition cannot be given one. Every read below is past the
    // `iDiffMbIdx == 0` guard, where the partition is known to hold an MB window.
    pMb: Option<crate::safe::mb_grid::MbWindow<'_, SMB>>,
    bands: &mut SlicePicBands<'_>,
    // This partition's CABAC restore scratch, taken from the context beside the
    // bitstream slot — reborrowed per coded slice below.
    mut pRestoreBuf: Option<&mut [u8]>,
    // This worker's slice bank, owned: the size-limited mode grows its bank
    // in-fork. The bank leaves the layer before the spawn (`std::mem::take`,
    // like the grid and the scratch), the growth is an ordinary `Vec` resize on
    // an exclusive borrow, and every resolve is an index. Restored after the
    // join.
    pBank: &mut SSliceBufferInfo,
) -> i32 {
    // ---- InitTask (base), minus the slot claim
    let eNalType = pCtx.eNalType;
    let eNalRefIdc = pCtx.eNalPriority;
    let bNeedPrefix = pCtx.bNeedPrefixNalFlag;

    // The resolve is an index into the owned bank; the borrow is taken *after*
    // each growth, per iteration.
    {
        let kiCur = pBank.iCodedSliceNum as usize;
        let Some(pSlice) = pBank.pSliceBuffer.get_mut(kiCur) else {
            return ENC_RETURN_UNEXPECTED;
        };
        InitOneSliceInThread(pCtx, pSlice, iBsSlot, iPartitionIdx);
        let iReturn = SetSliceBoundaryInfo(current_layer_ref(pCtx), pSlice, iPartitionIdx);
        if iReturn != ENC_RETURN_SUCCESS {
            return iReturn;
        }
        pSlice.sSliceBs.sBsWrite = BsWriter::new();
    }
    // The last coded slice is a remembered *index*, stamped after the loop.
    let mut kiLastCodedSlot: Option<usize> = None;
    // The slice time is stamped unconditionally here: `bUseLoadBalancing` does not
    // gate this task (`wels_task_encoder.h:110`).
    let iSliceStart = WelsTime();

    // ---- ExecuteTaskConstrainedSize
    let iResult = (|| {
        let pCurDq = current_layer_expect(pCtx);
        let kiSliceIdxStep = pCtx.iActiveThreadsNum as i32;
        let kiPartitionId = iPartitionIdx % kiSliceIdxStep;
        let kiFirstMbInPartition = pCurDq.FirstMbIdxOfPartition[kiPartitionId as usize];
        let kiEndMbIdxInPartition = pCurDq.EndMbIdxOfPartition[kiPartitionId as usize];
        let kiCodedSliceNumByThread = pBank.iCodedSliceNum as usize;
        pBank.pSliceBuffer[kiCodedSliceNumByThread]
            .sSliceHeaderExt
            .sSliceHeader
            .iFirstMbInSlice = kiFirstMbInPartition;
        kiLastCodedSlot = Some(kiCodedSliceNumByThread);

        let iDiffMbIdx = kiEndMbIdxInPartition - kiFirstMbInPartition;
        if 0 == iDiffMbIdx {
            pBank.pSliceBuffer[kiCodedSliceNumByThread].iSliceIdx = -1;
            return ENC_RETURN_SUCCESS;
        }
        let Some(mut mb_window) = pMb else {
            return ENC_RETURN_UNEXPECTED;
        };

        let mut iAnyMbLeftInPartition = iDiffMbIdx + 1;
        let mut iLocalSliceIdx = iPartitionIdx;
        while iAnyMbLeftInPartition > 0 {
            let bNeedReallocate = pBank.iCodedSliceNum >= pBank.iMaxSliceNum - 1;
            if bNeedReallocate {
                let iRet = ReallocateSliceInThread(pCtx, pCtx.uiDependencyId as i32, pBank);
                if ENC_RETURN_SUCCESS != iRet {
                    return iRet;
                }
            }

            // The resolve is a split of the owned bank — the current slot
            // exclusively, the boundary's forward slot beside it, from one
            // borrow. The borrow is taken *after* any growth above;
            // `tail.first_mut()` is `None` at the slot past the bank's end.
            let kiCurSlot = pBank.iCodedSliceNum as usize;
            if kiCurSlot >= pBank.pSliceBuffer.len() {
                return ENC_RETURN_UNEXPECTED;
            }
            let (kpHead, kpTail) = pBank.pSliceBuffer.split_at_mut(kiCurSlot + 1);
            let pSlice = &mut kpHead[kiCurSlot];
            let pNextSlice = kpTail.first_mut();
            InitOneSliceInThread(pCtx, pSlice, iBsSlot, iLocalSliceIdx);
            kiLastCodedSlot = Some(kiCurSlot);
            pSlice.sSliceBs.sBsWrite = BsWriter::new();

            // The partition slot, subsliced per coded slice — re-taken each
            // iteration because `InitOneSliceInThread` re-resolves the slice
            // (and with it the claimed size) after every reallocation. See
            // `EncodeOneSliceInJob` for the slot/size invariants.
            debug_assert_eq!(
                pSlice.uiBufferIdx as i32, iBsSlot,
                "the slice's claimed slot is this job's"
            );
            let kuiSize = pSlice.sSliceBs.uiSize;
            let pSliceBsBuf = &mut pSlotBuf[..kuiSize as usize];
            let mut pCtxOutBs: Option<&mut BsWriter> = None;

            if bNeedPrefix {
                WritePrefixNalForSlice(pCtx, pSlice, eNalRefIdc, eNalType, &mut *pSliceBsBuf);
            }
            WelsLoadNalForSlice(&mut pSlice.sSliceBs, eNalType as i32, eNalRefIdc as i32);

            debug_assert_eq!(iLocalSliceIdx, pSlice.iSliceIdx);
            // The forward slot is the split's other half; `iCodedSliceNum + 1`
            // is the split point by construction.
            let mut iRet = WelsCodeOneSlice(
                pCtx,
                pSlice,
                eNalType as i32,
                &mut *pSliceBsBuf,
                &mut pCtxOutBs,
                &mut mb_window,
                bands,
                pRestoreBuf.as_deref_mut(),
                pNextSlice,
            );
            if ENC_RETURN_SUCCESS != iRet {
                return iRet;
            }
            WelsUnloadNalForSlice(&mut pSlice.sSliceBs);

            let mut iSliceSize = 0i32;
            iRet = WriteSliceBs(pCtx, pSlice, iLocalSliceIdx, &mut iSliceSize, &*pSliceBsBuf);
            if ENC_RETURN_SUCCESS != iRet {
                return iRet;
            }
            // The walker reuses the partition run the coding chain just wrote
            // through, carved before the fork. `uiFilterIdc == 1` keeps the walk
            // and the neighbour reads inside the slice, hence inside the partition's
            // MB-row band.
            if let Some(pfDeblockingFilterSlice) =
                pCtx.func_list().pfDeblocking.pfDeblockingFilterSlice
            {
                pfDeblockingFilterSlice(
                    &mut bands.rec,
                    &pCurDq.sSliceEncCtx,
                    &pCurDq.iCsStride,
                    pSlice,
                    &mut mb_window,
                );
            }

            iAnyMbLeftInPartition = kiEndMbIdxInPartition
                - pCurDq.LastCodedMbIdxOfPartition[kiPartitionId as usize].load(Ordering::Relaxed);
            iLocalSliceIdx += kiSliceIdxStep;
            pBank.iCodedSliceNum += 1;
        }
        ENC_RETURN_SUCCESS
    })();

    // ---- FinishTask
    if let Some(kiSlot) = kiLastCodedSlot {
        pBank.pSliceBuffer[kiSlot].uiSliceConsumeTime = (WelsTime() - iSliceStart).max(0) as u32;
    }

    iResult
}

/// The fork/join for `SM_SIZELIMITED_SLICE` —
/// `pTaskManage->ExecuteTasks(WELS_ENC_TASK_ENCODING)` on the dynamic path.
///
/// One worker per picture partition, which is what the task count already was
/// (`kiTaskCount = iActiveThreadsNum`, `wels_task_management.rs` `CreateTasks`).
/// Returns the value `FinishTask` would have ORed into `pCtx->iEncoderError`.
pub fn EncodeSizeLimitedSlicesForked(pCtx: &mut sWelsEncCtx, kiPartitionCnt: i32) -> i32 {
    if kiPartitionCnt <= 0 || pCtx.pSliceThreading.is_none() {
        return ENC_RETURN_SUCCESS;
    }
    // Every partition is its own worker: the partition count is bounded by
    // `iMultipleThreadIdc` (`PicPartitionNumDecision`), which is also the bs-buffer
    // count, so the bound holds with equality. `ForkWidth`'s `min` enforces it.
    let iWidth = ForkWidth(pCtx, kiPartitionCnt);
    debug_assert_eq!(
        iWidth, kiPartitionCnt,
        "a size-limited partition would go unencoded: {kiPartitionCnt} partitions, {iWidth} buffers"
    );
    if iWidth > 1 {
        if let Some(pCurDq) = current_layer_ref(pCtx) {
            debug_assert!(
                !pCurDq.bDeblockingParallelFlag || pCurDq.iLoopFilterDisableIdc != 0,
                "parallel slice deblocking requires slice-boundary filtering to be disabled (iLoopFilterDisableIdc != 0)"
            );
            if pCurDq.bDeblockingParallelFlag && pCurDq.iLoopFilterDisableIdc == 0 {
                return ENC_RETURN_UNEXPECTED;
            }
        }
    }

    // Hoisted out of the fork — see `EncodeFixedSlicesForked`.
    crate::encoder::svc_encode_slice::StampLayerIdrFlagForSliceType(pCtx);

    let iWidthU = (iWidth as usize).min(MAX_THREADS_NUM);
    // Snapshot partition ranges on the stack before taking buffers out of `pCtx`.
    let (vPartRanges, kiGridWidth, kiMbWidth) = {
        let pCurDq = current_layer_expect(pCtx);
        let mut ranges = [(0i32, 0i32); MAX_THREADS_NUM];
        for (p, slot) in ranges[..iWidthU].iter_mut().enumerate() {
            let first = pCurDq.FirstMbIdxOfPartition[p];
            let end = pCurDq.EndMbIdxOfPartition[p];
            // A partition whose span is zero codes nothing, and the count must
            // say so: `EncodeOnePartitionSizeLimited`'s `iDiffMbIdx == 0` guard
            // stamps `iSliceIdx = -1` and returns before it reads the window.
            // `UpdateSlicepEncCtxWithPartition` clamps `iPartitionNum` whenever
            // `kiMbHeight` or `base_rows * kiMbWidth` is too small, then zeroes every
            // remaining slot, while the fork still runs `iActiveThreadsNum`
            // workers: 32x16 with 3 threads is two macroblocks and three workers.
            let span = end - first;
            *slot = (first, if span == 0 { 0 } else { span + 1 });
        }
        (
            ranges,
            pCurDq.sMbDataP.dims().mb_width(),
            pCurDq.iMbWidth as usize,
        )
    };

    // Move out the worker bs scratch buffers, CABAC restore buffers, slice banks,
    // MB grid, and mutable picture planes under an RAII guard that restores all of
    // them unconditionally on return or panic unwind.
    let mut guard = SizeLimitedForkGuard::take(pCtx, iWidthU);

    let mut iErr = ENC_RETURN_SUCCESS;
    {
        let SizeLimitedForkGuard {
            pCtx,
            bs_bufs,
            dyn_bufs,
            banks,
            mb_data,
            pics,
            ..
        } = &mut guard;
        let pCtx: &sWelsEncCtx = pCtx;

        let mut vMbPerWorker: [Option<crate::safe::mb_grid::MbWindow<'_, SMB>>; MAX_THREADS_NUM] =
            std::array::from_fn(|_| None);
        let mut vRowRanges = [(0usize, 0usize); MAX_THREADS_NUM];
        {
            let mut rest: &mut [SMB] = mb_data.as_mut_slice();
            let mut cursor = 0i32;
            for (p, &(first, count)) in vPartRanges[..iWidthU].iter().enumerate() {
                if count == 0 {
                    // No window and empty picture band: `MbWindow` requires a current
                    // macroblock and this partition has none. The worker's own
                    // `iDiffMbIdx == 0` guard returns before touching either.
                    continue;
                }
                debug_assert!(
                    first >= cursor && count > 0,
                    "partition {p} claims mbs [{first}..{}) against a carve cursor at {cursor} — \
                     the partition map is not disjoint raster order",
                    first as i64 + count as i64,
                );
                let first_u = first as usize;
                let end_excl_u = (first + count) as usize;
                debug_assert!(
                    kiMbWidth > 0 && first_u % kiMbWidth == 0 && end_excl_u % kiMbWidth == 0,
                    "partition {p} [{first_u}..{end_excl_u}) is not MB-row aligned (mb_width={kiMbWidth})"
                );
                if first < cursor
                    || count <= 0
                    || (first - cursor) as usize + count as usize > rest.len()
                    || kiMbWidth == 0
                    || first_u % kiMbWidth != 0
                    || end_excl_u % kiMbWidth != 0
                {
                    iErr |= ENC_RETURN_UNEXPECTED;
                    break;
                }
                let (_gap, tail) = rest.split_at_mut((first - cursor) as usize);
                let (chunk, tail) = tail.split_at_mut(count as usize);
                vMbPerWorker[p] = Some(crate::safe::mb_grid::MbWindow::new(
                    chunk,
                    first_u,
                    kiGridWidth,
                    first_u,
                ));
                vRowRanges[p] = (first_u / kiMbWidth, end_excl_u / kiMbWidth);
                rest = tail;
                cursor = first + count;
            }
        }
        if iErr == ENC_RETURN_SUCCESS {
            let mut vBands: [SlicePicBands<'_>; MAX_THREADS_NUM] =
                std::array::from_fn(|_| SlicePicBands::empty());
            pics.split_bands_into(&vRowRanges[..iWidthU], kiMbWidth, &mut vBands[..iWidthU]);

            let mut jobs: [Option<SizeLimitedJob<'_>>; MAX_THREADS_NUM] =
                std::array::from_fn(|_| None);
            for (k, (((((buf, mb), bands), dynbuf), bank), job_slot)) in bs_bufs[..iWidthU]
                .iter_mut()
                .zip(vMbPerWorker[..iWidthU].iter_mut())
                .zip(vBands[..iWidthU].iter_mut())
                .zip(dyn_bufs[..iWidthU].iter_mut())
                .zip(banks[..iWidthU].iter_mut())
                .zip(jobs[..iWidthU].iter_mut())
                .enumerate()
            {
                let dynbuf = if dynbuf.is_empty() {
                    None
                } else {
                    Some(dynbuf.as_mut_slice())
                };
                debug_assert!(
                    pCtx.pSliceThreading
                        .as_deref()
                        .is_some_and(|pSmt| k < pSmt.uiThreadBsBufferNum),
                    "job slot {k} is outside the allocated bs buffers"
                );
                *job_slot = Some(SizeLimitedJob {
                    pCtx,
                    pBsBuf: buf.as_mut_slice(),
                    pMb: mb.take(),
                    bands: std::mem::replace(bands, SlicePicBands::empty()),
                    pDynBsBuf: dynbuf,
                    pBank: bank,
                    iBsSlot: k as i32,
                    iPartitionIdx: k as i32,
                    iResult: ENC_RETURN_SUCCESS,
                });
            }

            if let Some(pSmt) = pCtx.pSliceThreading.as_deref() {
                let all_ok = pSmt.pool.for_each_mut(&mut jobs[..iWidthU], |slot| {
                    let Some(job) = slot.as_mut() else {
                        return;
                    };
                    job.iResult = EncodeOnePartitionSizeLimited(
                        job.pCtx,
                        job.iPartitionIdx,
                        job.iBsSlot,
                        job.pBsBuf,
                        job.pMb.take(),
                        &mut job.bands,
                        job.pDynBsBuf.take(),
                        job.pBank,
                    );
                });
                if !all_ok {
                    iErr |= ENC_RETURN_UNEXPECTED;
                }
                for job in jobs[..iWidthU].iter().flatten() {
                    iErr |= job.iResult;
                }
            }
        }
    }

    drop(guard);
    iErr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_div_round() {
        assert_eq!(WelsDivRound(100, 10), 10);
        assert_eq!(WelsDivRound(105, 10), 11);
        assert_eq!(WelsDivRound(104, 10), 10);
    }

    #[test]
    fn plan_fixed_slice_workers_merges_shared_rows_and_keeps_bands_disjoint() {
        // 4 MB-row-aligned slices on 80x45: 1 wave, 4 independent workers.
        let aligned = [(0, 960), (960, 880), (1840, 880), (2720, 880)];
        let waves = plan_fixed_slice_workers(&aligned, 80, 4).expect("valid");
        assert_eq!(
            waves,
            vec![vec![
                (0, 1, 0, 12),
                (1, 2, 12, 23),
                (2, 3, 23, 34),
                (3, 4, 34, 45)
            ]]
        );

        // 8 MB-row-aligned slices across 4 workers: 1 wave, contiguous pairs.
        let eight = [
            (0, 16),
            (16, 16),
            (32, 16),
            (48, 16),
            (64, 16),
            (80, 16),
            (96, 16),
            (112, 16),
        ];
        let waves = plan_fixed_slice_workers(&eight, 16, 4).expect("valid");
        assert_eq!(
            waves,
            vec![vec![(0, 2, 0, 2), (2, 4, 2, 4), (4, 6, 4, 6), (6, 8, 6, 8)]]
        );

        // Unaligned 720p `[900; 4]` (e.g. custom mid-row `SM_RASTER_SLICE`): every adjacent
        // pair shares a boundary row (11, 22, 33), so a 1-wave merge collapses to 1 worker;
        // the 2-wave schedule runs `{0, 2}` then `{1, 3}` with 2 disjoint workers per wave.
        let unaligned_720p_4 = [(0, 900), (900, 900), (1800, 900), (2700, 900)];
        let waves = plan_fixed_slice_workers(&unaligned_720p_4, 80, 4).expect("valid");
        assert_eq!(
            waves,
            vec![
                vec![(0, 1, 0, 12), (2, 3, 22, 34)],
                vec![(1, 2, 11, 23), (3, 4, 33, 45)],
            ]
        );

        // Unaligned 720p `[450; 8]`: 2 waves of 4 disjoint workers each (full 4-way
        // parallelism despite every adjacent boundary falling mid-row).
        let unaligned_720p_8: Vec<(i32, i32)> = (0..8).map(|i| (i * 450, 450)).collect();
        let waves = plan_fixed_slice_workers(&unaligned_720p_8, 80, 4).expect("valid");
        assert_eq!(waves.len(), 2);
        assert_eq!(waves[0].len(), 4);
        assert_eq!(waves[1].len(), 4);
        for wave in &waves {
            for pair in wave.windows(2) {
                assert!(pair[0].3 <= pair[1].2, "wave row bands must be disjoint");
            }
        }

        // Custom mid-row split on 16x4: slices 0 ([0..20)) and 1 ([20..48)) share row 1,
        // so single-wave merge (crit = 48) ties 2-wave (crit = 20 + 28 = 48) and avoids
        // an unnecessary second wave barrier.
        let mid_row = [(0, 20), (20, 28), (48, 16)];
        let waves = plan_fixed_slice_workers(&mid_row, 16, 4).expect("valid");
        assert_eq!(waves, vec![vec![(0, 2, 0, 3), (2, 3, 3, 4)]]);

        // 3 slices inside a single MB row plus a later disjoint row: verifies sound merging
        // when a slice overlaps both waves, keeping every wave's spans and rows disjoint.
        let dense_row = [(0, 5), (5, 5), (10, 6), (16, 20), (36, 20), (56, 24)];
        let firsts: Vec<i32> = dense_row.iter().map(|&(f, _)| f).collect();
        let counts: Vec<i32> = dense_row.iter().map(|&(_, c)| c).collect();
        let sched = plan_fixed_slice_schedule(&firsts, &counts, 16, 4).expect("valid");
        let mut seen = vec![false; dense_row.len()];
        let tags = sched.slice_waves(dense_row.len());
        for w in 0..sched.num_waves {
            let plans = &sched.waves[w][..sched.wave_widths[w]];
            for pair in plans.windows(2) {
                assert!(pair[0].1 <= pair[1].0, "slice spans must be disjoint");
                assert!(pair[0].3 <= pair[1].2, "row bands must be disjoint");
            }
            for &(s0, s1, _, _) in plans {
                for s in s0..s1 {
                    if sched.num_waves == 1 || tags[s] == w as u8 {
                        assert!(!seen[s], "slice {s} scheduled twice");
                        seen[s] = true;
                    }
                }
            }
        }
        assert!(seen.into_iter().all(|x| x), "every slice must be scheduled");

        // Overlapping / out-of-order ranges are rejected.
        assert!(plan_fixed_slice_workers(&[(0, 20), (16, 16)], 16, 2).is_none());
    }

    #[test]
    fn dynamic_adjust_slicing_rc_off_stays_mb_row_aligned() {
        let mut dq_layer = layer_with_bank(4);
        dq_layer.sSliceEncCtx.iMbWidth = 16;
        dq_layer.sSliceEncCtx.iMbHeight = 12;
        dq_layer.sSliceEncCtx.iMbNumInFrame = 192;
        dq_layer
            .sSliceEncCtx
            .iSliceNumInFrame
            .store(4, Ordering::Relaxed);
        dq_layer.sSliceEncCtx.pOverallMbMap = (0..192).map(|_| AtomicU16::new(0)).collect();
        dq_layer.pFirstMbIdxOfSlice = vec![0, 48, 96, 144];
        dq_layer.pCountMbNumInSlice = vec![48, 48, 48, 48];
        // Skewed complexity ratios whose unrounded assignments would fall mid-row.
        dq_layer.sSliceBufferInfo[0].pSliceBuffer[0].iSliceComplexRatio = 37;
        dq_layer.sSliceBufferInfo[0].pSliceBuffer[1].iSliceComplexRatio = 19;
        dq_layer.sSliceBufferInfo[0].pSliceBuffer[2].iSliceComplexRatio = 23;
        dq_layer.sSliceBufferInfo[0].pSliceBuffer[3].iSliceComplexRatio = 21;

        let mut param = SWelsSvcCodingParam::default();
        param.iRCMode = RCMode::RC_OFF_MODE;

        DynamicAdjustSlicing(&param, &[], &mut dq_layer, 0);
        assert!(dq_layer.bNeedAdjustingSlicing);
        assert_eq!(dq_layer.pCountMbNumInSlice.iter().sum::<i32>(), 192);
        for s in 0..4 {
            assert_eq!(
                dq_layer.pFirstMbIdxOfSlice[s] % 16,
                0,
                "slice {s} start must remain MB-row aligned"
            );
            assert_eq!(
                dq_layer.pCountMbNumInSlice[s] % 16,
                0,
                "slice {s} count must remain a multiple of iMbWidth"
            );
            assert!(dq_layer.pCountMbNumInSlice[s] >= 16);
        }
    }

    /// Two workers write distinct patterns into their own taken slots while
    /// both hold a shared borrow of the pool's owner, and the patterns must
    /// land back in the owner's own slots.
    #[test]
    fn bs_pool_partition_carves_disjoint_slots_and_restores_them() {
        const WORKERS: usize = 2;
        const LEN: usize = 64;

        let mut pSmt = SSliceThreading::default();
        for k in 0..WORKERS {
            pSmt.pThreadBsBuffer[k] = vec![0u8; LEN];
        }
        pSmt.uiThreadBsBufferNum = WORKERS;
        pSmt.pool = WorkerPool::new(WORKERS);

        // The take — the fork entries' partition.
        let mut vTakenBsBufs: Vec<Vec<u8>> = (0..WORKERS)
            .map(|k| std::mem::take(&mut pSmt.pThreadBsBuffer[k]))
            .collect();

        {
            // The pool is reached through the same shared borrow of the owner
            // the workers hold.
            let pSmtShared: &SSliceThreading = &pSmt;
            pSmtShared.pool.scope(|s| {
                for (k, buf) in vTakenBsBufs.iter_mut().enumerate() {
                    s.spawn(move || {
                        // A shared borrow of the owner held beside this worker's
                        // `&mut` slot, both live across the writes.
                        assert_eq!(pSmtShared.uiThreadBsBufferNum, WORKERS);
                        for (i, b) in buf.iter_mut().enumerate() {
                            *b = (k as u8) ^ (i as u8);
                        }
                        assert_eq!(pSmtShared.uiThreadBsBufferNum, WORKERS);
                    });
                }
            });
        }

        // The restore — buffer `k` to slot `k`.
        for (k, buf) in vTakenBsBufs.into_iter().enumerate() {
            pSmt.pThreadBsBuffer[k] = buf;
        }

        for k in 0..WORKERS {
            for i in 0..LEN {
                assert_eq!(
                    pSmt.pThreadBsBuffer[k][i],
                    (k as u8) ^ (i as u8),
                    "slot {k} byte {i} did not come back from its own worker"
                );
            }
        }
    }

    /// A layer with one bank of `n` slices and `ppSliceInLayer` naming them in
    /// order — the shape `InitSliceInLayer` builds.
    fn layer_with_bank(n: usize) -> SDqLayer {
        let mut dq_layer = SDqLayer::default();
        dq_layer.sSliceBufferInfo[0].pSliceBuffer = (0..n).map(|_| SSlice::new()).collect();
        dq_layer.sSliceBufferInfo[0].iMaxSliceNum = n as i32;
        dq_layer.ppSliceInLayer = (0..n)
            .map(|i| crate::encoder::svc_encode_slice::SliceIdx {
                bank: 0,
                offset: i as i32,
            })
            .collect();
        dq_layer
    }

    #[test]
    fn test_need_dynamic_adjust_zero_consume() {
        let mut dq_layer = layer_with_bank(2);
        let ret = NeedDynamicAdjust(&mut dq_layer, 2);
        assert_eq!(ret, 0);
    }

    #[test]
    fn test_need_dynamic_adjust_includes_last_slice() {
        let mut dq_layer = layer_with_bank(4);
        // Slices 0..3 are near-mean (240 each); slice 3 (last slice) is skewed (280).
        // Total = 1000, mean ratio = 0.25.
        // Without slice 3, sum of squared diffs = 3 * (-0.01)^2 = 0.0003, RMSE = sqrt(0.000075) = 0.00866 < 0.0200 (THRESHOLD_RMSE_CORE4).
        // With slice 3 (+0.03), sum of squared diffs = 0.0003 + 0.0009 = 0.0012, RMSE = sqrt(0.0003) = 0.01732.
        // With slice 0..3 = [235, 235, 235, 295], without slice 3 RMSE = sqrt(3 * 0.015^2 / 4) = 0.01299 < 0.0200,
        // whereas with slice 3 (+0.045) RMSE = sqrt((3 * 0.000225 + 0.002025) / 4) = sqrt(0.000675) = 0.02598 > 0.0200.
        dq_layer.sSliceBufferInfo[0].pSliceBuffer[0].uiSliceConsumeTime = 235;
        dq_layer.sSliceBufferInfo[0].pSliceBuffer[1].uiSliceConsumeTime = 235;
        dq_layer.sSliceBufferInfo[0].pSliceBuffer[2].uiSliceConsumeTime = 235;
        dq_layer.sSliceBufferInfo[0].pSliceBuffer[3].uiSliceConsumeTime = 295;
        assert_eq!(NeedDynamicAdjust(&mut dq_layer, 4), 1);
    }

    #[test]
    fn test_dynamic_adjust_slice_penc_ctx_all_same_run_len_returns_early() {
        let mut dq_layer = layer_with_bank(2);
        dq_layer.sSliceEncCtx.iMbNumInFrame = 100;
        dq_layer
            .sSliceEncCtx
            .iSliceNumInFrame
            .store(2, Ordering::Relaxed);
        dq_layer.sSliceEncCtx.pOverallMbMap = (0..100).map(|_| AtomicU16::new(0)).collect();
        dq_layer.pFirstMbIdxOfSlice = vec![0, 50];
        dq_layer.pCountMbNumInSlice = vec![50, 50];

        // Same run lengths [50, 50]: should return 1 (no adjustment needed).
        assert_eq!(DynamicAdjustSlicePEncCtxAll(&mut dq_layer, &[50, 50]), 1);
        // Changed run lengths [60, 40]: should return 0 and update slice boundaries.
        assert_eq!(DynamicAdjustSlicePEncCtxAll(&mut dq_layer, &[60, 40]), 0);
        assert_eq!(dq_layer.pFirstMbIdxOfSlice, vec![0, 60]);
        assert_eq!(dq_layer.pCountMbNumInSlice, vec![60, 40]);
    }

    #[test]
    fn test_calc_slice_complex_ratio() {
        let mut dq_layer = layer_with_bank(2);
        for slice in dq_layer.sSliceBufferInfo[0].pSliceBuffer.iter_mut() {
            slice.iCountMbNumInSlice = 100;
            slice.uiSliceConsumeTime = 1000;
        }
        dq_layer
            .sSliceEncCtx
            .iSliceNumInFrame
            .store(2, Ordering::Relaxed);

        CalcSliceComplexRatio(&mut dq_layer);

        assert_eq!(
            dq_layer.sSliceBufferInfo[0].pSliceBuffer[0].iSliceComplexRatio,
            50
        );
        assert_eq!(
            dq_layer.sSliceBufferInfo[0].pSliceBuffer[1].iSliceComplexRatio,
            50
        );
    }

    /// Runs the whole encoder with `bUseLoadBalancing` on, four threads and four
    /// slices — the four-term guard `WelsEncoderEncodeExt` tests before it calls the
    /// producer — for enough frames that frame N+1's boundaries are computed from
    /// frame N's measured times. Boundaries depend on wall-clock times, so it
    /// asserts structure, never bytes.
    ///
    /// 256x192 is forced: `MIN_NUM_MB_PER_SLICE` is 48 and
    /// `SliceArgumentValidationFixedSliceMode` silently rewrites a request it cannot
    /// honour down to a mode that needs no threads, so four slices need at least
    /// 4 x 48 = 192 macroblocks and a 16x12 grid is exactly 192. The
    /// `vcl_nals == 4` assertion catches that rewrite: on a smaller picture every
    /// other assertion passes while the encoder runs single-slice,
    /// single-threaded.
    ///
    /// Ignored under Miri: 192 macroblocks x 4 frames x 4 threads.
    #[cfg(feature = "c-api")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn load_balancing_completes_frames_with_sane_slice_counts() {
        use crate::api::codec_api::abi_test_driver::{EncoderProbeOptions, drive_encoder_over};

        let (frames, dims) = drive_encoder_over(
            256,
            192,
            4,
            EncoderProbeOptions {
                slice_mode: crate::api::codec_api::SliceModeEnum::SM_FIXEDSLCNUM_SLICE,
                slice_num: 4,
                threads: 4,
                load_balancing: true,
                ..EncoderProbeOptions::default()
            },
        );

        assert_eq!(
            dims,
            (256, 192),
            "the encoder must be configured for a 16x12 grid"
        );
        assert_eq!(frames.len(), 4, "the encode loop did not run to the end");
        assert!(
            frames.iter().all(|f| f.bytes > 0),
            "a frame produced no NAL bytes, which is what a lost slice looks like from \
             here: {:?}",
            frames.iter().map(|f| (f.kind, f.bytes)).collect::<Vec<_>>()
        );
        // `DynamicAdjustSlicing` redistributes macroblocks across the slices; it never
        // changes how many there are.
        assert!(
            frames.iter().all(|f| f.vcl_nals == 4),
            "a frame did not carry four VCL NALs, so the slice count moved under the \
             rebalance or the mode was rewritten: {:?}",
            frames
                .iter()
                .map(|f| (f.kind, f.vcl_nals))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            frames[0].kind,
            crate::api::codec_api::EVideoFrameType::videoFrameTypeIDR,
            "the sequence must open on an IDR"
        );
        assert!(
            frames[1..]
                .iter()
                .all(|f| f.kind == crate::api::codec_api::EVideoFrameType::videoFrameTypeP),
            "frames 1..4 must be inter-coded, or the rebalance never sees a second \
             frame's times: {:?}",
            frames.iter().map(|f| f.kind).collect::<Vec<_>>()
        );
    }

    /// `RequestMtResource` spawns `N - 1` background workers (since the calling
    /// thread executes slot 0 directly via `WorkerPool::for_each_mut`).
    #[test]
    fn request_mt_resource_spawns_n_minus_1_background_workers() {
        for threads in [2u16, 3, 4] {
            let mut ctx = sWelsEncCtx::default();
            let mut param = crate::encoder::param_svc::NewCodingParam();
            param.iMultipleThreadIdc = threads;
            ctx.pSvcParam = Some(param);
            assert_eq!(RequestMtResource(&mut ctx, 1024, 512, false), 0);
            let smt = ctx.pSliceThreading.as_deref().expect("mt resource built");
            assert_eq!(smt.uiThreadBsBufferNum, threads as usize);
            assert_eq!(smt.pool.workers(), threads as usize - 1);
            ReleaseMtResource(&mut ctx);
            assert!(ctx.pSliceThreading.is_none());
        }
    }

    /// RAII fork guards (`FixedForkGuard` and `SizeLimitedForkGuard`) unconditionally
    /// restore `pThreadBsBuffer`, slice banks, `sMbDataP`, `pDynamicBsBuffer`, and
    /// picture planes back into `sWelsEncCtx` even when a panic unwinds.
    #[test]
    #[cfg(panic = "unwind")]
    fn fork_guards_restore_context_on_panic_unwind() {
        use crate::encoder::encoder_ext::{WelsInitEncoderExt, WelsUninitEncoderExt};
        use crate::encoder::svc_encode_slice::{LayerIdx, set_current_layer};

        let make_mt_ctx = |slice_mode: SliceMode| -> Box<sWelsEncCtx> {
            let mut ext = crate::SEncParamExt::default();
            SWelsSvcCodingParam::FillDefaultExt(&mut ext);
            ext.iPicWidth = 320;
            ext.iPicHeight = 192;
            ext.iTargetBitrate = 500_000;
            ext.fMaxFrameRate = 30.0;
            ext.iSpatialLayerNum = 1;
            ext.iMultipleThreadIdc = 2;
            ext.iEntropyCodingModeFlag = 1;
            ext.sSpatialLayers[0].iVideoWidth = 320;
            ext.sSpatialLayers[0].iVideoHeight = 192;
            ext.sSpatialLayers[0].iSpatialBitrate = 500_000;
            ext.sSpatialLayers[0].fFrameRate = 30.0;
            ext.sSpatialLayers[0].sSliceArgument.uiSliceMode = slice_mode;
            ext.sSpatialLayers[0].sSliceArgument.uiSliceNum = 2;
            ext.sSpatialLayers[0].sSliceArgument.uiSliceSizeConstraint = 800;

            let mut param = SWelsSvcCodingParam::default();
            assert_eq!(param.ParamTranscode(&ext), 0);
            let mut pCtx: Option<Box<sWelsEncCtx>> = None;
            assert_eq!(
                WelsInitEncoderExt(
                    &mut pCtx,
                    &mut param,
                    crate::common::wels_trace::SLogContext::default(),
                    None,
                ),
                0
            );
            let mut ctx = pCtx.unwrap();
            set_current_layer(&mut ctx, Some(LayerIdx(0)));
            let id_dec = ctx.ref_list(0).and_then(|rl| rl.pNextBuffer);
            current_layer_expect_mut(&mut ctx).pDecPic = id_dec;
            ctx
        };

        // 1. FixedForkGuard unwind restoration.
        let mut ctx = make_mt_ctx(SliceMode::SM_FIXEDSLCNUM_SLICE);
        let bs0_len = ctx.pSliceThreading.as_deref().unwrap().pThreadBsBuffer[0].len();
        let bs1_len = ctx.pSliceThreading.as_deref().unwrap().pThreadBsBuffer[1].len();
        let bank0_len = current_layer_expect(&ctx).sSliceBufferInfo[0]
            .pSliceBuffer
            .len();
        let mb_len = current_layer_expect(&ctx).sMbDataP.as_slice().len();
        assert!(bs0_len > 0 && bs1_len > 0 && bank0_len > 0 && mb_len > 0);

        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = FixedForkGuard::take(&mut ctx, 2);
            panic!("simulated fixed-fork setup panic");
        }));
        assert!(res.is_err());
        assert_eq!(
            ctx.pSliceThreading.as_deref().unwrap().pThreadBsBuffer[0].len(),
            bs0_len
        );
        assert_eq!(
            ctx.pSliceThreading.as_deref().unwrap().pThreadBsBuffer[1].len(),
            bs1_len
        );
        assert_eq!(
            current_layer_expect(&ctx).sSliceBufferInfo[0]
                .pSliceBuffer
                .len(),
            bank0_len
        );
        assert_eq!(current_layer_expect(&ctx).sMbDataP.as_slice().len(), mb_len);
        let id_dec = current_layer_expect(&ctx).pDecPic.unwrap();
        assert!(
            !ctx.ref_list(0)
                .unwrap()
                .pic(id_dec)
                .plane_tail(0)
                .is_empty()
        );
        WelsUninitEncoderExt(Some(ctx));

        // 2. SizeLimitedForkGuard unwind restoration.
        let mut ctx = make_mt_ctx(SliceMode::SM_SIZELIMITED_SLICE);
        let bs0_len = ctx.pSliceThreading.as_deref().unwrap().pThreadBsBuffer[0].len();
        let dyn0_len = ctx.pDynamicBsBuffer[0].len();
        let dyn1_len = ctx.pDynamicBsBuffer[1].len();
        let bank0_len = current_layer_expect(&ctx).sSliceBufferInfo[0]
            .pSliceBuffer
            .len();
        let bank1_len = current_layer_expect(&ctx).sSliceBufferInfo[1]
            .pSliceBuffer
            .len();
        let mb_len = current_layer_expect(&ctx).sMbDataP.as_slice().len();
        assert!(bs0_len > 0 && dyn0_len > 0 && dyn1_len > 0 && bank0_len > 0 && bank1_len > 0);

        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = SizeLimitedForkGuard::take(&mut ctx, 2);
            panic!("simulated size-limited-fork setup panic");
        }));
        assert!(res.is_err());
        assert_eq!(
            ctx.pSliceThreading.as_deref().unwrap().pThreadBsBuffer[0].len(),
            bs0_len
        );
        assert_eq!(ctx.pDynamicBsBuffer[0].len(), dyn0_len);
        assert_eq!(ctx.pDynamicBsBuffer[1].len(), dyn1_len);
        assert_eq!(
            current_layer_expect(&ctx).sSliceBufferInfo[0]
                .pSliceBuffer
                .len(),
            bank0_len
        );
        assert_eq!(
            current_layer_expect(&ctx).sSliceBufferInfo[1]
                .pSliceBuffer
                .len(),
            bank1_len
        );
        assert_eq!(current_layer_expect(&ctx).sMbDataP.as_slice().len(), mb_len);
        let id_dec = current_layer_expect(&ctx).pDecPic.unwrap();
        assert!(
            !ctx.ref_list(0)
                .unwrap()
                .pic(id_dec)
                .plane_tail(0)
                .is_empty()
        );
        WelsUninitEncoderExt(Some(ctx));
    }

    /// `InitTask` failures (`SetSliceBoundaryInfo` / missing slice slot) propagate as
    /// errors instead of being silently swallowed.
    #[test]
    fn init_task_failures_propagate_instead_of_being_swallowed() {
        let mut ctx = sWelsEncCtx::default();
        let mut param = crate::encoder::param_svc::NewCodingParam();
        param.iMultipleThreadIdc = 2;
        ctx.pSvcParam = Some(param);
        assert_eq!(RequestMtResource(&mut ctx, 1024, 512, false), 0);

        // Layer with empty `pFirstMbIdxOfSlice` causes `SetSliceBoundaryInfo` to fail.
        let mut dq = Box::new(layer_with_bank(2));
        dq.iMbWidth = 16;
        dq.iMbHeight = 12;
        ctx.ppDqLayerList = vec![Some(dq)];
        set_current_layer(&mut ctx, Some(LayerIdx(0)));

        // Direct `EncodeOneSliceInJob` with empty `pSlices` fails.
        let mut bs_buf = vec![0u8; 256];
        let mut bands = SlicePicBands::empty();
        assert_eq!(
            EncodeOneSliceInJob(
                &ctx,
                0,
                0,
                false,
                &mut bs_buf,
                &mut [],
                &mut [],
                &mut bands,
                0
            ),
            ENC_RETURN_UNEXPECTED
        );

        // `EncodeFixedSlicesForked` propagates missing boundary tables as an error.
        assert_eq!(EncodeFixedSlicesForked(&mut ctx, 2), ENC_RETURN_UNEXPECTED);

        // `EncodeOnePartitionSizeLimited` with empty slice bank fails.
        let mut empty_bank = SSliceBufferInfo::default();
        assert_eq!(
            EncodeOnePartitionSizeLimited(
                &ctx,
                0,
                0,
                &mut bs_buf,
                None,
                &mut bands,
                None,
                &mut empty_bank,
            ),
            ENC_RETURN_UNEXPECTED
        );

        // `EncodeOnePartitionSizeLimited` with valid bank slot but missing boundary
        // tables (`SetSliceBoundaryInfo` failure) propagates the error.
        let mut one_slot_bank = SSliceBufferInfo {
            pSliceBuffer: vec![SSlice::new()],
            iMaxSliceNum: 1,
            iCodedSliceNum: 0,
        };
        assert_eq!(
            EncodeOnePartitionSizeLimited(
                &ctx,
                0,
                0,
                &mut bs_buf,
                None,
                &mut bands,
                None,
                &mut one_slot_bank,
            ),
            ENC_RETURN_UNEXPECTED
        );

        ReleaseMtResource(&mut ctx);
    }
}
