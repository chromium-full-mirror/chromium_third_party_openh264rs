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

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

//! Encoder picture buffers and reference-picture state —
//! `codec/encoder/core/inc/picture.h`.

#![forbid(unsafe_code)]

use crate::encoder::encoder_context::{BLOCK_SIZE_ALL, MAX_THREADS_NUM, SMVUnitXY};
use crate::encoder::svc_motion_estimate::{LIST_SIZE_SUM_8x8, LIST_SIZE_SUM_16x16};
use crate::safe::mb_grid::MbSlice;
pub use crate::safe::plane::{PaddedPlane, PlaneBand, PlaneBandMut, PlaneCursor, PlaneCursorMut};
use crate::safe::pool::{Id, Pool};

/// `PADDING_LENGTH` — `codec/encoder/core/inc/wels_const.h`. The luma border
/// `AllocPicture` puts around every picture; chroma gets half of it.
const PADDING_LENGTH: usize = 32;

use crate::common::macros::align_up;

/// `LTR_MARKING_RECEIVE_STATE` — `codec/encoder/core/inc/wels_const.h:150`.
pub const RECIEVE_UNKOWN: u8 = 0;
pub const RECIEVE_SUCCESS: u8 = 1;
pub const RECIEVE_FAILED: u8 = 2;

/// `LIST_SIZE` — `picture.h:42`, `(256*256)`.
pub const LIST_SIZE: usize = 0x10000;

/// `SScreenBlockFeatureStorage` — `codec/encoder/core/inc/picture.h:43`.
/// Stored with a reference picture, one per frame.
///
/// `iActualListSize` bounds the first two tables; the cursor table is [`LIST_SIZE`]
/// (65,536) entries long so `u16` feature values index it without bounds checks.
#[derive(Debug)]
pub struct SScreenBlockFeatureStorage {
    pub iIs16x16: i32,
    pub uiFeatureStrategyIndex: u8,
    /// Histogram: how many block positions carry each feature value. `iActualListSize`
    /// entries.
    pub pTimesOfFeatureValue: Vec<u32>,
    /// Per feature value, the **start offset** of its group in `pLocationPointer`.
    /// `iActualListSize` entries.
    pub pLocationOfFeature: Vec<usize>,
    /// The arena of (x, y) qpel position pairs, grouped by feature value.
    pub pLocationPointer: Vec<u16>,
    pub iActualListSize: i32,
    pub uiSadCostThreshold: [u32; BLOCK_SIZE_ALL],
    pub bRefBlockFeatureCalculated: bool,
    /// Per feature value, the **roving write cursor** into `pLocationPointer` — starts
    /// at that value's base and advances by 2 per position written.
    pub pFeatureValuePointerList: Vec<usize>,
}

impl Default for SScreenBlockFeatureStorage {
    /// Every buffer empty and `uiSadCostThreshold` `UINT_MAX`-filled — not the zero a
    /// derived `Default` would give.
    fn default() -> Self {
        Self {
            iIs16x16: 0,
            uiFeatureStrategyIndex: 0,
            pTimesOfFeatureValue: Vec::new(),
            pLocationOfFeature: Vec::new(),
            pLocationPointer: Vec::new(),
            iActualListSize: 0,
            uiSadCostThreshold: [u32::MAX; BLOCK_SIZE_ALL],
            bRefBlockFeatureCalculated: false,
            pFeatureValuePointerList: Vec::new(),
        }
    }
}

impl SScreenBlockFeatureStorage {
    /// `svc_motion_estimate.cpp:690-721` as a constructor.
    ///
    /// The histogram and base table are `kiListSize` long, the arena is
    /// `2 * kiFrameSize`, and the cursor table is [`LIST_SIZE`] — not `kiListSize`.
    ///
    /// Called from `AllocPicture` for the last layer's reference pictures under
    /// `SCREEN_CONTENT_REAL_TIME`; `bIsBlock8x8` is `(kiMe8x8FME == ME_FME)`.
    pub fn for_frame(
        kiFrameWidth: i32,
        kiFrameHeight: i32,
        bIsBlock8x8: bool,
        kiFeatureStrategyIndex: u8,
    ) -> Self {
        let kiMarginSize = if bIsBlock8x8 { 8 } else { 16 };
        let kiFrameSize =
            ((kiFrameWidth - kiMarginSize).max(0) * (kiFrameHeight - kiMarginSize).max(0)) as usize;
        let kiListSize = if kiFeatureStrategyIndex == 0 {
            if bIsBlock8x8 {
                LIST_SIZE_SUM_8x8
            } else {
                LIST_SIZE_SUM_16x16
            }
        } else {
            256
        };
        Self {
            iIs16x16: i32::from(!bIsBlock8x8),
            uiFeatureStrategyIndex: kiFeatureStrategyIndex,
            pTimesOfFeatureValue: vec![0; kiListSize],
            pLocationOfFeature: vec![0; kiListSize],
            pLocationPointer: vec![0; 2 * kiFrameSize],
            iActualListSize: kiListSize as i32,
            uiSadCostThreshold: [u32::MAX; BLOCK_SIZE_ALL],
            bRefBlockFeatureCalculated: false,
            pFeatureValuePointerList: vec![0; LIST_SIZE],
        }
    }
}

/// `SPicture` — `codec/encoder/core/inc/picture.h:64`.
#[derive(Debug)]
pub struct SPicture {
    /// The three planes — Y, Cb, Cr. Each [`PaddedPlane`] owns its bytes.
    planes: [PaddedPlane; 3],

    // picture information, from pSps
    pub iWidthInPixel: i32,
    pub iHeightInPixel: i32,
    pub iPictureType: i32,
    pub iFramePoc: i32,

    pub fFrameRate: f32,
    pub iFrameNum: i32,

    /// The four per-macroblock side arrays, `kuiCountMbNum` entries each, or **empty**
    /// where `AllocPicture`'s `bNeedMbInfo` was false — the spatial-source and scaled
    /// pictures never carry them.
    pub uiRefMbType: Vec<u32>,
    pub pRefMbQp: Vec<u8>,
    pub pMbSkipSad: Vec<i32>,

    pub sMvList: Vec<SMVUnitXY>,

    // self-definition for misc use
    pub iMarkFrameNum: i32,
    pub iLongTermPicNum: i32,

    pub bUsedAsRef: bool,
    pub bIsLongRef: bool,
    pub bIsSceneLTR: bool,
    pub uiRecieveConfirmed: u8,
    pub uiTemporalId: u8,
    pub uiSpatialId: u8,
    pub iFrameAverageQp: i32,

    // for screen reference frames
    /// `Some` for the last layer's reference pictures under
    /// `SCREEN_CONTENT_REAL_TIME`, `None` for every other picture.
    pub pScreenBlockFeatureStorage: Option<Box<SScreenBlockFeatureStorage>>,
}

impl SPicture {
    /// `picture_handle.cpp:51`, everything that is not the plane allocator.
    ///
    /// `bNeedMbInfo` decides whether the four side arrays exist at all; when it is
    /// false they are empty `Vec`s, which every consumer tests with `is_empty()`.
    ///
    /// A fresh picture has `iFramePoc == 0` and `uiTemporalId == uiSpatialId == 0` —
    /// **not** the `-1`/`255` that [`SetUnref`](Self::SetUnref) leaves behind.
    pub fn new(kiWidth: i32, kiHeight: i32, bNeedMbInfo: bool) -> Box<SPicture> {
        let kuiCountMbNum = if bNeedMbInfo {
            (((15 + kiWidth) >> 4) * ((15 + kiHeight) >> 4)).max(0) as usize
        } else {
            0
        };

        // `picture_handle.cpp:60-74`'s geometry — the alignment is load-bearing
        // (`pStrideDecBlockOffset` is built from the same strides).
        let kuiAlignedWidth = align_up(kiWidth.max(0) as usize, 16);
        let kuiAlignedHeight = align_up(kiHeight.max(0) as usize, 16);
        let kuiLumaStride = align_up(kuiAlignedWidth + 2 * PADDING_LENGTH, 32);
        let kuiChromaStride = align_up((kuiAlignedWidth + 2 * PADDING_LENGTH) >> 1, 16);

        Box::new(SPicture {
            // `PaddedPlane::new` zeroes: on the first frame `VAACalcSad` reads the
            // visible luma of a reference picture nothing has written yet.
            planes: [
                PaddedPlane::new(
                    kuiAlignedWidth,
                    kuiAlignedHeight,
                    PADDING_LENGTH,
                    kuiLumaStride,
                ),
                PaddedPlane::new(
                    kuiAlignedWidth >> 1,
                    kuiAlignedHeight >> 1,
                    PADDING_LENGTH >> 1,
                    kuiChromaStride,
                ),
                PaddedPlane::new(
                    kuiAlignedWidth >> 1,
                    kuiAlignedHeight >> 1,
                    PADDING_LENGTH >> 1,
                    kuiChromaStride,
                ),
            ],

            iWidthInPixel: kiWidth,
            iHeightInPixel: kiHeight,
            iPictureType: 0,
            // zeroed, not -1: `AllocPicture` never writes `iFramePoc`.
            iFramePoc: 0,

            fFrameRate: 0.0,
            // `picture_handle.cpp:99`.
            iFrameNum: -1,

            uiRefMbType: vec![0u32; kuiCountMbNum],
            pRefMbQp: vec![0u8; kuiCountMbNum],
            pMbSkipSad: vec![0i32; kuiCountMbNum],
            sMvList: vec![SMVUnitXY { iMvX: 0, iMvY: 0 }; kuiCountMbNum],

            iMarkFrameNum: -1,
            iLongTermPicNum: -1,

            bUsedAsRef: false,
            bIsLongRef: false,
            bIsSceneLTR: false,
            uiRecieveConfirmed: RECIEVE_UNKOWN,
            // zeroed, not 255: `AllocPicture` never writes either id.
            uiTemporalId: 0,
            uiSpatialId: 0,
            iFrameAverageQp: 0,

            // `AllocPicture` attaches the storage after this.
            pScreenBlockFeatureStorage: None,
        })
    }

    /// `uiRefMbType` as a raw pointer to its **root**, or null where the picture
    /// carries no macroblock info.
    ///
    /// The address is the `Vec`'s own root, never an index into it, so the pointer's
    /// provenance covers the whole array. Its one consumer,
    /// `SComplexityAnalysisParam.uiRefMbType`, tests it for null.
    #[inline]
    pub fn ref_mb_type_root(&mut self) -> *mut u32 {
        if self.uiRefMbType.is_empty() {
            std::ptr::null_mut()
        } else {
            self.uiRefMbType.as_mut_ptr()
        }
    }

    /// Plane `i`'s stride — the C++'s `iLineSize[i]`.
    #[inline]
    pub fn stride(&self, i: usize) -> i32 {
        self.planes[i].stride() as i32
    }

    /// Plane `i`'s samples from the logical origin, as the borrow they are.
    /// Empty when the plane is unallocated.
    #[inline]
    pub fn plane_tail(&self, i: usize) -> &[u8] {
        let plane = &self.planes[i];
        if plane.is_empty() {
            return &[];
        }
        let origin = plane.origin();
        &plane.as_slice()[origin..]
    }

    /// [`plane_tail`](Self::plane_tail)'s write half.
    #[inline]
    pub fn plane_tail_mut(&mut self, i: usize) -> &mut [u8] {
        let plane = &mut self.planes[i];
        if plane.is_empty() {
            return &mut [];
        }
        let origin = plane.origin();
        &mut plane.as_mut_slice()[origin..]
    }

    /// Copy the top-left `kiWidth x kiHeight` samples of `kpSrc` into `self`,
    /// each plane at its own stride — the pool-to-pool half of `WelsMoveMemory_c`.
    ///
    /// Chroma takes half the geometry in each dimension (`iWidth >> 1`,
    /// `iHeight >> 1`); both pictures keep their own strides, so a copy between
    /// differently-padded allocations is the same walk.
    ///
    /// # Panics
    /// If either picture has a plane shorter than the geometry asks for.
    pub fn copy_planes_from(&mut self, kpSrc: &SPicture, kiWidth: i32, kiHeight: i32) {
        let (kuiW, kuiH) = (kiWidth.max(0) as usize, kiHeight.max(0) as usize);
        for i in 0..3 {
            let (kuiRow, kuiRows) = if i == 0 {
                (kuiW, kuiH)
            } else {
                (kuiW >> 1, kuiH >> 1)
            };
            if kuiRow == 0 || kuiRows == 0 {
                continue;
            }
            let kuiSrcStride = kpSrc.planes[i].stride();
            let kuiDstStride = self.planes[i].stride();
            let kpFrom = kpSrc.plane_tail(i);
            let pInto = self.plane_tail_mut(i);
            for y in 0..kuiRows {
                pInto[y * kuiDstStride..][..kuiRow]
                    .copy_from_slice(&kpFrom[y * kuiSrcStride..][..kuiRow]);
            }
        }
    }

    /// Plane `i`, for a converted caller that wants the safe view.
    #[inline]
    pub fn plane(&self, i: usize) -> &PaddedPlane {
        &self.planes[i]
    }

    /// Mutable form of [`plane`](Self::plane).
    #[inline]
    pub fn plane_mut(&mut self, i: usize) -> &mut PaddedPlane {
        &mut self.planes[i]
    }

    /// All three planes mutably at once, for a step that writes Y, U and V in one
    /// pass (`METHOD_DENOISE`, `METHOD_DOWNSAMPLE`).
    #[inline]
    pub fn planes_mut3(&mut self) -> [&mut PaddedPlane; 3] {
        let [y, u, v] = &mut self.planes;
        [y, u, v]
    }

    /// Read cursor anchored at macroblock `(mb_x, mb_y)` in plane `plane`
    /// (`0` = luma, `1` = Cb, `2` = Cr).
    #[inline(always)]
    pub fn mb_cursor(&self, plane: usize, mb_x: i32, mb_y: i32) -> PlaneCursor<'_> {
        let shift = if plane == 0 { 4 } else { 3 };
        self.planes[plane].cursor((mb_x as isize) << shift, (mb_y as isize) << shift)
    }

    /// `ExpandReferencingPicture` for a picture that owns its planes.
    pub fn expand_as_reference(&mut self) {
        let (kiWidthY, kiHeightY) = (self.iWidthInPixel, self.iHeightInPixel);
        let planes = [
            (0usize, kiWidthY, kiHeightY, PADDING_LENGTH),
            (1, kiWidthY >> 1, kiHeightY >> 1, PADDING_LENGTH >> 1),
            (2, kiWidthY >> 1, kiHeightY >> 1, PADDING_LENGTH >> 1),
        ];
        for (i, pic_w, pic_h, pad) in planes {
            let stride = self.stride(i) as usize;
            let plane = self.plane_mut(i);
            if plane.is_empty() || pic_w <= 0 || pic_h <= 0 {
                continue;
            }
            crate::common::expand_pic::expand_picture(
                plane.as_mut_slice(),
                stride,
                pic_w as usize,
                pic_h as usize,
                pad,
            );
        }
    }

    /// The picture's strides and visible geometry, copied out.
    #[inline]
    pub fn planes(&self) -> PicPlanes {
        PicPlanes {
            iLineSize: [self.stride(0), self.stride(1), self.stride(2)],
            iWidthInPixel: self.iWidthInPixel,
            iHeightInPixel: self.iHeightInPixel,
        }
    }

    /// Set picture as unreferenced. Matches `SPicture::SetUnref()`, `picture.h:106`.
    pub fn SetUnref(&mut self) {
        self.iFramePoc = -1;
        self.iFrameNum = -1;
        self.uiTemporalId = 255;
        self.uiSpatialId = 255;
        self.iLongTermPicNum = -1;
        self.bIsLongRef = false;
        self.uiRecieveConfirmed = RECIEVE_FAILED;
        self.iMarkFrameNum = -1;
        self.bUsedAsRef = false;

        // picture_handle.cpp:245.
        if let Some(storage) = self.pScreenBlockFeatureStorage.as_deref_mut() {
            storage.bRefBlockFeatureCalculated = false;
        }
    }

    /// Temporarily moves out the mutable pixel planes and per-MB side arrays so they can
    /// be carved into disjoint `&mut` horizontal bands while `&sWelsEncCtx` is borrowed
    /// shared across worker threads. Restored by [`restore_data`](Self::restore_data).
    ///
    /// Plane stride metadata is preserved on the empty placeholders.
    pub fn take_data(&mut self) -> TakenPicData {
        let [s0, s1, s2] = [
            self.planes[0].stride(),
            self.planes[1].stride(),
            self.planes[2].stride(),
        ];
        TakenPicData {
            planes: [
                std::mem::replace(&mut self.planes[0], PaddedPlane::empty(s0)),
                std::mem::replace(&mut self.planes[1], PaddedPlane::empty(s1)),
                std::mem::replace(&mut self.planes[2], PaddedPlane::empty(s2)),
            ],
            sMvList: std::mem::take(&mut self.sMvList),
            pRefMbQp: std::mem::take(&mut self.pRefMbQp),
            pMbSkipSad: std::mem::take(&mut self.pMbSkipSad),
            uiRefMbType: std::mem::take(&mut self.uiRefMbType),
        }
    }

    /// Restores pixel planes and per-MB side arrays previously moved out by
    /// [`take_data`](Self::take_data).
    pub fn restore_data(&mut self, data: TakenPicData) {
        self.planes = data.planes;
        self.sMvList = data.sMvList;
        self.pRefMbQp = data.pRefMbQp;
        self.pMbSkipSad = data.pMbSkipSad;
        self.uiRefMbType = data.uiRefMbType;
    }

    /// Temporarily moves out only the three pixel planes (leaving side arrays intact).
    pub fn take_planes(&mut self) -> [PaddedPlane; 3] {
        let [s0, s1, s2] = [
            self.planes[0].stride(),
            self.planes[1].stride(),
            self.planes[2].stride(),
        ];
        [
            std::mem::replace(&mut self.planes[0], PaddedPlane::empty(s0)),
            std::mem::replace(&mut self.planes[1], PaddedPlane::empty(s1)),
            std::mem::replace(&mut self.planes[2], PaddedPlane::empty(s2)),
        ]
    }

    /// Restores the three pixel planes previously moved out by
    /// [`take_planes`](Self::take_planes).
    pub fn restore_planes(&mut self, planes: [PaddedPlane; 3]) {
        self.planes = planes;
    }
}

/// Mutable storage moved out of an [`SPicture`] via [`SPicture::take_data`].
#[derive(Debug)]
pub struct TakenPicData {
    pub planes: [PaddedPlane; 3],
    pub sMvList: Vec<SMVUnitXY>,
    pub pRefMbQp: Vec<u8>,
    pub pMbSkipSad: Vec<i32>,
    pub uiRefMbType: Vec<u32>,
}

impl TakenPicData {
    /// Empty placeholder storage (no planes, no side arrays).
    pub fn empty() -> Self {
        Self {
            planes: [
                PaddedPlane::empty(0),
                PaddedPlane::empty(0),
                PaddedPlane::empty(0),
            ],
            sMvList: Vec::new(),
            pRefMbQp: Vec::new(),
            pMbSkipSad: Vec::new(),
            uiRefMbType: Vec::new(),
        }
    }

    /// Borrows the entire picture as a single [`RecPicBand`] (single-threaded path).
    pub fn whole_rec_band(&mut self) -> RecPicBand<'_> {
        let [py, pu, pv] = &mut self.planes;
        RecPicBand {
            planes: [
                py.whole_band_mut(),
                pu.whole_band_mut(),
                pv.whole_band_mut(),
            ],
            mb_base: 0,
            sMvList: self.sMvList.as_mut_slice(),
            pRefMbQp: self.pRefMbQp.as_mut_slice(),
            pMbSkipSad: self.pMbSkipSad.as_mut_slice(),
            uiRefMbType: self.uiRefMbType.as_mut_slice(),
        }
    }

    /// Splits this reconstruction picture into disjoint [`RecPicBand`]s for the given
    /// MB-row ranges `[(row_start, row_end), ...]`, writing into `out[..mb_row_ranges.len()]`
    /// without heap allocations for up to `MAX_THREADS_NUM` bands.
    pub fn split_rec_bands_into<'a>(
        &'a mut self,
        mb_row_ranges: &[(usize, usize)],
        mb_width: usize,
        out: &mut [RecPicBand<'a>],
    ) {
        let n = mb_row_ranges.len();
        assert!(out.len() >= n);
        if n <= MAX_THREADS_NUM {
            let [py, pu, pv] = &mut self.planes;
            let mut y_bands =
                std::array::from_fn::<_, MAX_THREADS_NUM, _>(|_| PlaneBandMut::empty());
            let mut u_bands =
                std::array::from_fn::<_, MAX_THREADS_NUM, _>(|_| PlaneBandMut::empty());
            let mut v_bands =
                std::array::from_fn::<_, MAX_THREADS_NUM, _>(|_| PlaneBandMut::empty());
            let mut mv_bands: [&mut [SMVUnitXY]; MAX_THREADS_NUM] =
                std::array::from_fn(|_| &mut [][..]);
            let mut qp_bands: [&mut [u8]; MAX_THREADS_NUM] = std::array::from_fn(|_| &mut [][..]);
            let mut sad_bands: [&mut [i32]; MAX_THREADS_NUM] = std::array::from_fn(|_| &mut [][..]);
            let mut type_bands: [&mut [u32]; MAX_THREADS_NUM] =
                std::array::from_fn(|_| &mut [][..]);
            py.split_scaled_bands_mut_into(mb_row_ranges, 16, &mut y_bands[..n]);
            pu.split_scaled_bands_mut_into(mb_row_ranges, 8, &mut u_bands[..n]);
            pv.split_scaled_bands_mut_into(mb_row_ranges, 8, &mut v_bands[..n]);
            split_side_array_into(
                &mut self.sMvList,
                mb_row_ranges,
                mb_width,
                &mut mv_bands[..n],
            );
            split_side_array_into(
                &mut self.pRefMbQp,
                mb_row_ranges,
                mb_width,
                &mut qp_bands[..n],
            );
            split_side_array_into(
                &mut self.pMbSkipSad,
                mb_row_ranges,
                mb_width,
                &mut sad_bands[..n],
            );
            split_side_array_into(
                &mut self.uiRefMbType,
                mb_row_ranges,
                mb_width,
                &mut type_bands[..n],
            );
            for i in 0..n {
                out[i] = RecPicBand {
                    planes: [
                        std::mem::replace(&mut y_bands[i], PlaneBandMut::empty()),
                        std::mem::replace(&mut u_bands[i], PlaneBandMut::empty()),
                        std::mem::replace(&mut v_bands[i], PlaneBandMut::empty()),
                    ],
                    mb_base: mb_row_ranges[i].0 * mb_width,
                    sMvList: std::mem::take(&mut mv_bands[i]),
                    pRefMbQp: std::mem::take(&mut qp_bands[i]),
                    pMbSkipSad: std::mem::take(&mut sad_bands[i]),
                    uiRefMbType: std::mem::take(&mut type_bands[i]),
                };
            }
        } else {
            let bands = self.split_rec_bands(mb_row_ranges, mb_width);
            for (slot, band) in out[..n].iter_mut().zip(bands) {
                *slot = band;
            }
        }
    }

    /// Splits this reconstruction picture into disjoint [`RecPicBand`]s for the given
    /// MB-row ranges `[(row_start, row_end), ...]`.
    pub fn split_rec_bands(
        &mut self,
        mb_row_ranges: &[(usize, usize)],
        mb_width: usize,
    ) -> Vec<RecPicBand<'_>> {
        let plane_bands = split_planes_by_mb_rows(&mut self.planes, mb_row_ranges);
        let mv_bands = split_side_array(&mut self.sMvList, mb_row_ranges, mb_width);
        let qp_bands = split_side_array(&mut self.pRefMbQp, mb_row_ranges, mb_width);
        let sad_bands = split_side_array(&mut self.pMbSkipSad, mb_row_ranges, mb_width);
        let type_bands = split_side_array(&mut self.uiRefMbType, mb_row_ranges, mb_width);

        plane_bands
            .into_iter()
            .zip(mb_row_ranges.iter())
            .zip(mv_bands)
            .zip(qp_bands)
            .zip(sad_bands)
            .zip(type_bands)
            .map(
                |(
                    ((((planes, &(row_start, _row_end)), sMvList), pRefMbQp), pMbSkipSad),
                    uiRefMbType,
                )| {
                    RecPicBand {
                        planes,
                        mb_base: row_start * mb_width,
                        sMvList,
                        pRefMbQp,
                        pMbSkipSad,
                        uiRefMbType,
                    }
                },
            )
            .collect()
    }
}

/// Splits three `[Y, Cb, Cr]` [`PaddedPlane`]s into disjoint horizontal bands aligned to
/// the given macroblock-row ranges `[(row_start, row_end), ...]`, writing into
/// `out[..mb_row_ranges.len()]` with no heap allocations for up to `MAX_THREADS_NUM` bands.
pub fn split_planes_by_mb_rows_into<'a>(
    planes: &'a mut [PaddedPlane; 3],
    mb_row_ranges: &[(usize, usize)],
    out: &mut [[PlaneBandMut<'a>; 3]],
) {
    let n = mb_row_ranges.len();
    assert!(out.len() >= n);
    if n <= MAX_THREADS_NUM {
        let [py, pu, pv] = planes;
        let mut ys = std::array::from_fn::<_, MAX_THREADS_NUM, _>(|_| PlaneBandMut::empty());
        let mut us = std::array::from_fn::<_, MAX_THREADS_NUM, _>(|_| PlaneBandMut::empty());
        let mut vs = std::array::from_fn::<_, MAX_THREADS_NUM, _>(|_| PlaneBandMut::empty());
        py.split_scaled_bands_mut_into(mb_row_ranges, 16, &mut ys[..n]);
        pu.split_scaled_bands_mut_into(mb_row_ranges, 8, &mut us[..n]);
        pv.split_scaled_bands_mut_into(mb_row_ranges, 8, &mut vs[..n]);
        for i in 0..n {
            out[i] = [
                std::mem::replace(&mut ys[i], PlaneBandMut::empty()),
                std::mem::replace(&mut us[i], PlaneBandMut::empty()),
                std::mem::replace(&mut vs[i], PlaneBandMut::empty()),
            ];
        }
    } else {
        let bands = split_planes_by_mb_rows(planes, mb_row_ranges);
        for (slot, band) in out[..n].iter_mut().zip(bands) {
            *slot = band;
        }
    }
}

/// Splits three `[Y, Cb, Cr]` [`PaddedPlane`]s into disjoint horizontal bands aligned to
/// the given macroblock-row ranges `[(row_start, row_end), ...]`.
pub fn split_planes_by_mb_rows<'a>(
    planes: &'a mut [PaddedPlane; 3],
    mb_row_ranges: &[(usize, usize)],
) -> Vec<[PlaneBandMut<'a>; 3]> {
    let n = mb_row_ranges.len();
    let [py, pu, pv] = planes;
    let mut ys = Vec::with_capacity(n);
    ys.resize_with(n, PlaneBandMut::empty);
    let mut us = Vec::with_capacity(n);
    us.resize_with(n, PlaneBandMut::empty);
    let mut vs = Vec::with_capacity(n);
    vs.resize_with(n, PlaneBandMut::empty);
    py.split_scaled_bands_mut_into(mb_row_ranges, 16, &mut ys);
    pu.split_scaled_bands_mut_into(mb_row_ranges, 8, &mut us);
    pv.split_scaled_bands_mut_into(mb_row_ranges, 8, &mut vs);
    ys.into_iter()
        .zip(us)
        .zip(vs)
        .map(|((y, u), v)| [y, u, v])
        .collect()
}

fn split_side_array_into<'a, T>(
    v: &'a mut [T],
    mb_row_ranges: &[(usize, usize)],
    mb_width: usize,
    out: &mut [&'a mut [T]],
) {
    let n = mb_row_ranges.len();
    assert!(out.len() >= n);
    if v.is_empty() {
        for slot in &mut out[..n] {
            *slot = &mut [];
        }
        return;
    }
    let mut rest = v;
    let mut cur_mb = 0usize;
    for (&(row_start, row_end), slot) in mb_row_ranges.iter().zip(out.iter_mut()) {
        if row_start == row_end {
            *slot = &mut [];
            continue;
        }
        let mb_start = row_start * mb_width;
        let mb_end = row_end * mb_width;
        assert!(
            mb_start >= cur_mb && mb_start < mb_end && mb_end - cur_mb <= rest.len(),
            "side array mb range [{mb_start}..{mb_end}) invalid against cursor {cur_mb} and remaining {}",
            rest.len()
        );
        let (_gap, tail) = rest.split_at_mut(mb_start - cur_mb);
        let (chunk, tail) = tail.split_at_mut(mb_end - mb_start);
        *slot = chunk;
        rest = tail;
        cur_mb = mb_end;
    }
}

fn split_side_array<'a, T>(
    v: &'a mut [T],
    mb_row_ranges: &[(usize, usize)],
    mb_width: usize,
) -> Vec<&'a mut [T]> {
    let mut out = Vec::with_capacity(mb_row_ranges.len());
    out.resize_with(mb_row_ranges.len(), || &mut [][..]);
    split_side_array_into(v, mb_row_ranges, mb_width, &mut out);
    out
}

/// One worker's exclusive mutable band of the reconstruction picture (`pDecPic`):
/// the three pixel planes for its MB-row band, and the four per-MB side arrays
/// for `[mb_base .. mb_base + len)`.
#[derive(Debug)]
pub struct RecPicBand<'a> {
    planes: [PlaneBandMut<'a>; 3],
    mb_base: usize,
    sMvList: &'a mut [SMVUnitXY],
    pRefMbQp: &'a mut [u8],
    pMbSkipSad: &'a mut [i32],
    uiRefMbType: &'a mut [u32],
}

impl<'a> RecPicBand<'a> {
    /// An empty reconstruction band (no pixels, no side arrays).
    pub fn empty() -> Self {
        Self {
            planes: [
                PlaneBandMut::empty(),
                PlaneBandMut::empty(),
                PlaneBandMut::empty(),
            ],
            mb_base: 0,
            sMvList: &mut [],
            pRefMbQp: &mut [],
            pMbSkipSad: &mut [],
            uiRefMbType: &mut [],
        }
    }

    /// Borrows an entire [`SPicture`] as one [`RecPicBand`].
    pub fn whole(pic: &'a mut SPicture) -> Self {
        let [py, pu, pv] = &mut pic.planes;
        Self {
            planes: [
                py.whole_band_mut(),
                pu.whole_band_mut(),
                pv.whole_band_mut(),
            ],
            mb_base: 0,
            sMvList: pic.sMvList.as_mut_slice(),
            pRefMbQp: pic.pRefMbQp.as_mut_slice(),
            pMbSkipSad: pic.pMbSkipSad.as_mut_slice(),
            uiRefMbType: pic.uiRefMbType.as_mut_slice(),
        }
    }

    #[inline(always)]
    pub fn plane(&self, i: usize) -> &PlaneBandMut<'a> {
        &self.planes[i]
    }

    /// Read-only [`PlaneBand`] for plane `i`.
    #[inline(always)]
    pub fn plane_band(&self, i: usize) -> PlaneBand<'_> {
        self.planes[i].as_band()
    }

    /// Read cursor anchored at macroblock `(mb_x, mb_y)` in plane `plane`.
    #[inline(always)]
    pub fn mb_cursor(&self, plane: usize, mb_x: i32, mb_y: i32) -> PlaneCursor<'_> {
        let shift = if plane == 0 { 4 } else { 3 };
        self.planes[plane].cursor((mb_x as isize) << shift, (mb_y as isize) << shift)
    }

    #[inline(always)]
    pub fn plane_mut(&mut self, i: usize) -> &mut PlaneBandMut<'a> {
        &mut self.planes[i]
    }

    #[inline(always)]
    pub fn planes_mut3(&mut self) -> [&mut PlaneBandMut<'a>; 3] {
        let [y, u, v] = &mut self.planes;
        [y, u, v]
    }

    /// Write cursor anchored at macroblock `(mb_x, mb_y)` in plane `plane`.
    #[inline(always)]
    pub fn mb_cursor_mut(&mut self, plane: usize, mb_x: i32, mb_y: i32) -> PlaneCursorMut<'_> {
        let shift = if plane == 0 { 4 } else { 3 };
        self.planes[plane].cursor_mut((mb_x as isize) << shift, (mb_y as isize) << shift)
    }

    /// Three write cursors anchored at macroblock `(mb_x, mb_y)` across Y, Cb, and Cr.
    #[inline(always)]
    pub fn mb_cursors_mut(
        &mut self,
        mb_x: i32,
        mb_y: i32,
    ) -> (PlaneCursorMut<'_>, PlaneCursorMut<'_>, PlaneCursorMut<'_>) {
        let (lx, ly) = ((mb_x as isize) << 4, (mb_y as isize) << 4);
        let (cx, cy) = ((mb_x as isize) << 3, (mb_y as isize) << 3);
        let [y, u, v] = &mut self.planes;
        (
            y.cursor_mut(lx, ly),
            u.cursor_mut(cx, cy),
            v.cursor_mut(cx, cy),
        )
    }

    /// Read-only view of this band's `pMbSkipSad` slice for neighbor-cache lookups.
    #[inline(always)]
    pub fn mb_skip_sad(&self) -> MbSlice<'_, i32> {
        MbSlice::new(self.pMbSkipSad, self.mb_base)
    }

    #[inline(always)]
    pub fn has_mv_list(&self) -> bool {
        !self.sMvList.is_empty()
    }

    #[inline(always)]
    pub fn set_mv(&mut self, mb_xy: usize, v: SMVUnitXY) {
        self.sMvList[mb_xy - self.mb_base] = v;
    }

    #[inline(always)]
    pub fn set_mv_if_present(&mut self, mb_xy: usize, v: SMVUnitXY) {
        if !self.sMvList.is_empty() {
            self.sMvList[mb_xy - self.mb_base] = v;
        }
    }

    #[inline(always)]
    pub fn set_ref_mb_qp(&mut self, mb_xy: usize, v: u8) {
        self.pRefMbQp[mb_xy - self.mb_base] = v;
    }

    #[inline(always)]
    pub fn set_mb_skip_sad(&mut self, mb_xy: usize, v: i32) {
        self.pMbSkipSad[mb_xy - self.mb_base] = v;
    }

    #[inline(always)]
    pub fn set_ref_mb_type(&mut self, mb_xy: usize, v: u32) {
        self.uiRefMbType[mb_xy - self.mb_base] = v;
    }
}

/// One worker's exclusive horizontal band of the current source picture (`pEncPic`),
/// plus optional separate VAA current-picture planes when `vaa.pCurPic != pEncPic`.
///
/// Mode decision and motion estimation borrow this shared (`&self`) for read cursors;
/// `VaaBackgroundMbDataUpdate` borrows it mutably (`&mut self`) to copy background MBs
/// from the read-only VAA reference picture.
#[derive(Debug)]
pub struct EncPicBand<'a> {
    planes: [PlaneBandMut<'a>; 3],
    vaa_updates_enc: bool,
    vaa_cur_separate: Option<[PlaneBandMut<'a>; 3]>,
}

/// Three read-only `[Y, Cb, Cr]` cursors anchored at a macroblock in the source band.
#[derive(Clone, Copy)]
pub struct EncMbCursors<'a> {
    pub y: PlaneCursor<'a>,
    pub cb: PlaneCursor<'a>,
    pub cr: PlaneCursor<'a>,
}

impl<'a> EncPicBand<'a> {
    /// An empty source band covering no rows.
    pub fn empty() -> Self {
        Self {
            planes: [
                PlaneBandMut::empty(),
                PlaneBandMut::empty(),
                PlaneBandMut::empty(),
            ],
            vaa_updates_enc: false,
            vaa_cur_separate: None,
        }
    }

    /// Wraps an entire [`SPicture`] as one source band (no separate VAA current picture).
    pub fn whole(pic: &'a mut SPicture, vaa_updates_enc: bool) -> Self {
        let [py, pu, pv] = &mut pic.planes;
        Self {
            planes: [
                py.whole_band_mut(),
                pu.whole_band_mut(),
                pv.whole_band_mut(),
            ],
            vaa_updates_enc,
            vaa_cur_separate: None,
        }
    }

    /// Constructs a band from carved `[Y, Cb, Cr]` plane bands.
    pub fn from_planes(
        planes: [PlaneBandMut<'a>; 3],
        vaa_updates_enc: bool,
        vaa_cur_separate: Option<[PlaneBandMut<'a>; 3]>,
    ) -> Self {
        Self {
            planes,
            vaa_updates_enc,
            vaa_cur_separate,
        }
    }

    #[inline(always)]
    pub fn plane(&self, i: usize) -> &PlaneBandMut<'a> {
        &self.planes[i]
    }

    /// Read-only [`PlaneBand`] for plane `i`.
    #[inline(always)]
    pub fn plane_band(&self, i: usize) -> PlaneBand<'_> {
        self.planes[i].as_band()
    }

    /// Read cursor anchored at macroblock `(mb_x, mb_y)` in plane `plane`.
    #[inline(always)]
    pub fn mb_cursor(&self, plane: usize, mb_x: i32, mb_y: i32) -> PlaneCursor<'_> {
        let shift = if plane == 0 { 4 } else { 3 };
        self.planes[plane].cursor((mb_x as isize) << shift, (mb_y as isize) << shift)
    }

    /// Three read cursors anchored at macroblock `(mb_x, mb_y)` across Y, Cb, and Cr.
    #[inline(always)]
    pub fn mb_cursors(&self, mb_x: i32, mb_y: i32) -> EncMbCursors<'_> {
        let (lx, ly) = ((mb_x as isize) << 4, (mb_y as isize) << 4);
        let (cx, cy) = ((mb_x as isize) << 3, (mb_y as isize) << 3);
        EncMbCursors {
            y: self.planes[0].cursor(lx, ly),
            cb: self.planes[1].cursor(cx, cy),
            cr: self.planes[2].cursor(cx, cy),
        }
    }

    /// Mutable `[Y, Cb, Cr]` bands for `VaaBackgroundMbDataUpdate`, or `None` when
    /// background updates are inactive for this frame.
    #[inline(always)]
    pub fn vaa_cur_planes_mut(&mut self) -> Option<&mut [PlaneBandMut<'a>; 3]> {
        if self.vaa_updates_enc {
            Some(&mut self.planes)
        } else {
            self.vaa_cur_separate.as_mut()
        }
    }
}

/// One worker's exclusive mutable picture bands for the current frame/layer:
/// the reconstruction picture (`rec`) and the source picture (`enc`).
///
/// Kept as two sibling fields so callers can borrow `&mut bands.rec` and `&bands.enc`
/// simultaneously without aliasing.
#[derive(Debug)]
pub struct SlicePicBands<'a> {
    pub rec: RecPicBand<'a>,
    pub enc: EncPicBand<'a>,
}

impl<'a> SlicePicBands<'a> {
    /// Empty placeholder bands (unit tests that do not access pixel buffers).
    pub fn empty() -> Self {
        Self {
            rec: RecPicBand::empty(),
            enc: EncPicBand::empty(),
        }
    }

    /// Whole-frame bands directly borrowing `rec_pic` and `enc_pic`.
    pub fn whole(
        rec_pic: &'a mut SPicture,
        enc_pic: &'a mut SPicture,
        vaa_updates_enc: bool,
    ) -> Self {
        Self {
            rec: RecPicBand::whole(rec_pic),
            enc: EncPicBand::whole(enc_pic, vaa_updates_enc),
        }
    }
}

/// A picture's strides and geometry, copied out of it — see [`SPicture::planes`].
#[derive(Clone, Copy, Debug)]
pub struct PicPlanes {
    pub iLineSize: [i32; 3],
    pub iWidthInPixel: i32,
    pub iHeightInPixel: i32,
}

impl Default for PicPlanes {
    fn default() -> Self {
        Self {
            iLineSize: [0; 3],
            iWidthInPixel: 0,
            iHeightInPixel: 0,
        }
    }
}

// ===========================================================================
// The two pools, and the two handle types that address them
// ===========================================================================

/// A reference-picture handle that may name **either** pool.
///
/// The encoder owns pictures in exactly **three** places:
///
/// * the **reconstruction pool**, one per dependency layer, in that layer's
///   `SRefList` — the pictures the encoder decodes into and then references;
/// * the **spatial source pool**, in `CWelsPreProcess` — the downsampled copies of
///   the caller's frames;
/// * **one scaled input picture**, a slot of its own in `Scaled_Picture`.
///
/// [`SrcPicId`] and [`RecPicId`] are distinct types with no conversion in either
/// direction, because `pEncPic` (source) and `pDecPic`/`pRefPic` (reconstruction)
/// meet in one `WelsEncoderEncodeExt` iteration and in `UpdateOriginalPicInfo`.
///
/// A handle is `Copy`, names a slot rather than an address, and does not own, so a
/// slot recycled under an old handle is still a hazard; a debug-build generation
/// counter catches it. See [`crate::safe::pool`].
///
/// A [`RecPicId`] names a slot in **one dependency layer's** `SRefList`, not a global
/// picture, and every consumer resolves it through the layer it came from —
/// `ppRefPicListExt[uiDependencyId]`, or `SDqLayer::pRefList`, which is that same
/// list. No `RecPicId` crosses a layer switch.
///
/// `SDqLayer::pRefOri` is the one field that holds both kinds: `WelsBuildRefList`
/// (camera) stores *reconstruction* pictures there, `WelsBuildRefListScreen` stores
/// *spatial source* pictures, and which it is depends on a usage-type branch taken
/// frames earlier. Its readers are `JudgeStaticSkip` and `JudgeScrollSkip`, both
/// screen-content paths, so on the camera path the `Rec` writes are dead stores.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PicRef {
    /// A slot of a dependency layer's reconstruction pool.
    Rec(RecPicId),
    /// A slot of the spatial source pool.
    Src(SrcPicId),
}

macro_rules! pic_pool {
    ($id:ident, $pool:ident, $what:literal) => {
        #[doc = concat!("A handle to a slot of the ", $what, " picture pool.")]
        #[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
        pub struct $id(Id);

        #[doc = concat!("The ", $what, " picture pool: the owner of its slots.")]
        #[derive(Debug)]
        pub struct $pool(Pool<Box<SPicture>>);

        impl $pool {
            /// Takes ownership of `slots`. The pool never grows: both picture sets
            /// are sized once at initialisation and recycled thereafter.
            pub fn new(slots: Vec<Box<SPicture>>) -> Self {
                Self(Pool::new(slots))
            }

            /// A pool with no slots — what a host holds before `RequestMemorySvc`
            /// runs and after the pictures are released.
            pub fn empty() -> Self {
                Self::new(Vec::new())
            }

            /// Number of slots.
            #[inline]
            pub fn len(&self) -> usize {
                self.0.len()
            }

            /// Whether the pool has no slots.
            #[inline]
            pub fn is_empty(&self) -> bool {
                self.0.is_empty()
            }

            /// A handle to slot `index`.
            ///
            /// # Panics
            /// If `index` is out of range.
            #[inline]
            pub fn at(&self, index: usize) -> $id {
                $id(self.0.id(index))
            }

            /// Handles to every slot, in order.
            pub fn ids(&self) -> impl Iterator<Item = $id> + '_ {
                self.0.ids().map($id)
            }

            /// The picture `id` names.
            #[inline]
            pub fn get(&self, id: $id) -> &SPicture {
                self.0.get(id.0)
            }

            /// Mutable form of [`get`](Self::get).
            #[inline]
            pub fn get_mut(&mut self, id: $id) -> &mut SPicture {
                self.0.get_mut(id.0)
            }

            /// Two *different* pictures mutably at once — the read-one-write-another
            /// shape (downsampling, motion-compensated analysis).
            ///
            /// # Panics
            /// If `a == b`.
            #[inline]
            pub fn pair_mut(&mut self, a: $id, b: $id) -> (&mut SPicture, &mut SPicture) {
                let (x, y) = self.0.pair_mut(a.0, b.0);
                (&mut **x, &mut **y)
            }
        }
    };
}

pic_pool!(SrcPicId, SrcPicPool, "spatial source");
pic_pool!(RecPicId, RecPicPool, "reconstruction");

#[cfg(test)]
mod tests {
    use super::*;

    /// [`SPicture::plane`] and [`SPicture::plane_tail`] cover the padding behind the
    /// logical origin and the aligned stride layout.
    #[test]
    fn plane_reaches_the_padding_behind_the_logical_origin() {
        // 176x144 QCIF as `SPicture::new` lays it out.
        let mut pic = SPicture::new(176, 144, false);
        let pad = PADDING_LENGTH;
        let stride = pic.plane(0).stride();
        assert_eq!(
            stride,
            align_up(align_up(176, 16) + 2 * PADDING_LENGTH, 32),
            "the C's WELS_ALIGN(WELS_ALIGN(w,16) + 2*PADDING_LENGTH, 32)"
        );
        assert_eq!(
            pic.plane(0).origin(),
            (1 + stride) * pad,
            "the C's (1 + iLineSize[0]) * PADDING_LENGTH"
        );

        pic.plane_mut(0).set(0, 0, 0x5A);
        pic.plane_mut(0).set(-1, -1, 0xC3);
        pic.plane_mut(0).set(-(pad as isize), -(pad as isize), 0x7E);

        let origin = pic.plane(0).origin();
        assert_eq!(pic.plane_tail(0)[0], 0x5A);
        assert_eq!(
            pic.plane(0).as_slice()[origin - (stride + 1)],
            0xC3,
            "one sample diagonally behind the origin — intra prediction's top-left read"
        );
        assert_eq!(
            pic.plane(0).as_slice()[0],
            0x7E,
            "the top-left corner of the padding"
        );

        // Chroma keeps half the padding and its own aligned stride.
        assert_eq!(
            pic.stride(1),
            align_up((align_up(176, 16) + 2 * PADDING_LENGTH) >> 1, 16) as i32
        );
        assert_eq!(pic.stride(1), pic.stride(2));
        assert_eq!(pic.plane(1).pad(), PADDING_LENGTH / 2);
    }

    /// The four per-macroblock side arrays exist exactly when `bNeedMbInfo` says so,
    /// and are sized as `picture_handle.cpp:105` sizes them.
    #[test]
    fn side_arrays_follow_need_mb_info() {
        let with = SPicture::new(176, 144, true);
        let n = ((176 + 15) >> 4) * ((144 + 15) >> 4);
        assert_eq!(with.uiRefMbType.len(), n as usize);
        assert_eq!(with.pRefMbQp.len(), n as usize);
        assert_eq!(with.pMbSkipSad.len(), n as usize);
        assert_eq!(with.sMvList.len(), n as usize);

        let without = SPicture::new(176, 144, false);
        assert!(without.uiRefMbType.is_empty());
        assert!(without.sMvList.is_empty());
    }

    /// A fresh picture holds zeros where [`SPicture::SetUnref`] leaves `-1`/`255`.
    #[test]
    fn a_fresh_picture_is_not_an_unreferenced_one() {
        let pic = SPicture::new(176, 144, false);
        assert_eq!(pic.iFramePoc, 0);
        assert_eq!(pic.uiTemporalId, 0);
        assert_eq!(pic.uiSpatialId, 0);
        assert_eq!(pic.uiRecieveConfirmed, RECIEVE_UNKOWN);
        // and the fields the constructor writes
        assert_eq!(pic.iWidthInPixel, 176);
        assert_eq!(pic.iHeightInPixel, 144);
        assert_eq!(pic.iFrameNum, -1);
        assert!(!pic.bIsLongRef);
        assert_eq!(pic.iLongTermPicNum, -1);
        assert_eq!(pic.iMarkFrameNum, -1);
    }

    /// [`SPicture::copy_planes_from`] copies the requested sub-rectangle across
    /// each plane at its own stride without disturbing padding bytes.
    #[test]
    fn copy_planes_from_copies_subrectangle_and_preserves_padding() {
        for &(sw, sh, dw, dh, w, h) in &[
            (176, 144, 176, 144, 176, 144), // same geometry
            (176, 144, 160, 128, 160, 128), // destination narrower: strides differ
            (320, 240, 176, 144, 176, 144),
            (176, 144, 176, 144, 32, 16), // a sub-rectangle of both
        ] {
            let mut src = SPicture::new(sw, sh, false);
            for i in 0..3 {
                let stride = src.planes[i].stride();
                let origin = src.planes[i].origin();
                let buf = src.planes[i].as_mut_slice();
                for (k, b) in buf.iter_mut().enumerate() {
                    *b = (k.wrapping_mul(31).wrapping_add(i * 7).wrapping_add(origin)
                        ^ (k / stride.max(1))) as u8;
                }
            }

            let mut dst = SPicture::new(dw, dh, false);
            for i in 0..3 {
                dst.planes[i].as_mut_slice().fill(0xA5);
            }

            dst.copy_planes_from(&src, w, h);

            for i in 0..3 {
                let (rw, rh) = if i == 0 {
                    (w as usize, h as usize)
                } else {
                    ((w >> 1) as usize, (h >> 1) as usize)
                };
                for y in 0..rh {
                    assert_eq!(
                        dst.plane(i).row(y as isize, 0, rw),
                        src.plane(i).row(y as isize, 0, rw),
                        "plane {i} row {y} differs for src {sw}x{sh} -> dst {dw}x{dh}"
                    );
                }
                // Leading padding stays untouched.
                let origin = dst.plane(i).origin();
                assert!(
                    dst.plane(i).as_slice()[..origin].iter().all(|&b| b == 0xA5),
                    "plane {i} padding overwritten"
                );
            }
        }
    }
}
