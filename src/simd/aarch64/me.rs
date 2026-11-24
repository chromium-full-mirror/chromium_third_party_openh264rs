// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Screen-content motion-estimation feature kernels — `SumOf8x8SingleBlock_AArch64_neon`,
//! `SumOf16x16SingleBlock_AArch64_neon`, `SumOf8x8BlockOfFrame_AArch64_neon`, and
//! `SumOf16x16BlockOfFrame_AArch64_neon`
//! (`codec/encoder/core/arm64/svc_motion_estimation_aarch64_neon.S`).

#![deny(unsafe_code)]

use core::arch::aarch64::*;

use super::lanes::{ld8, ld8_u16, ld16, st8_u16};
use super::neon_token;
use crate::encoder::svc_motion_estimate;
use crate::safe::plane::{BlockRows, PlaneCursor, RefSamples};
use crate::simd::{simd_entry, tokens::NeonToken};

#[inline]
#[target_feature(enable = "neon")]
fn sum_window8_u16(lo: uint16x8_t, hi: uint16x8_t) -> uint16x8_t {
    let s2_lo = vaddq_u16(lo, vextq_u16(lo, hi, 1));
    let s2_hi = vaddq_u16(hi, vextq_u16(hi, hi, 1));
    let s4_lo = vaddq_u16(s2_lo, vextq_u16(s2_lo, s2_hi, 2));
    let s4_hi = vaddq_u16(s2_hi, vextq_u16(s2_hi, s2_hi, 2));
    vaddq_u16(s4_lo, vextq_u16(s4_lo, s4_hi, 4))
}

#[inline]
#[target_feature(enable = "neon")]
fn sliding_sum8(row16: &[u8]) -> uint16x8_t {
    let r = ld16(row16);
    let lo = vmovl_u8(vget_low_u8(r));
    let hi = vmovl_high_u8(r);
    sum_window8_u16(lo, hi)
}

#[inline]
#[target_feature(enable = "neon")]
fn sliding_diff8(bot16: &[u8], top16: &[u8]) -> uint16x8_t {
    let b = ld16(bot16);
    let t = ld16(top16);
    let lo = vsubl_u8(vget_low_u8(b), vget_low_u8(t));
    let hi = vsubl_high_u8(b, t);
    sum_window8_u16(lo, hi)
}

simd_entry! {
    #[target_feature(enable = "neon")]
    fn sum_8x8_single(cRef: &PlaneCursor<'_>, _token: NeonToken) -> i32 {
        let s = cRef.span::<8, 8>(0, 0);
        let mut acc = vaddl_u8(ld8(&s.row::<8>(0, 0)), ld8(&s.row::<8>(1, 0)));
        acc = vaddw_u8(acc, ld8(&s.row::<8>(2, 0)));
        acc = vaddw_u8(acc, ld8(&s.row::<8>(3, 0)));
        acc = vaddw_u8(acc, ld8(&s.row::<8>(4, 0)));
        acc = vaddw_u8(acc, ld8(&s.row::<8>(5, 0)));
        acc = vaddw_u8(acc, ld8(&s.row::<8>(6, 0)));
        acc = vaddw_u8(acc, ld8(&s.row::<8>(7, 0)));
        vaddlvq_u16(acc) as i32
    }

    #[target_feature(enable = "neon")]
    fn sum_16x16_single(cRef: &PlaneCursor<'_>, _token: NeonToken) -> i32 {
        let s = cRef.span::<16, 16>(0, 0);
        let mut acc0 = vpaddlq_u8(ld16(&s.row::<16>(0, 0)));
        let mut acc1 = vpaddlq_u8(ld16(&s.row::<16>(1, 0)));
        let mut y = 2usize;
        while y < 16 {
            acc0 = vpadalq_u8(acc0, ld16(&s.row::<16>(y, 0)));
            acc1 = vpadalq_u8(acc1, ld16(&s.row::<16>(y + 1, 0)));
            y += 2;
        }
        vaddlvq_u16(vaddq_u16(acc0, acc1)) as i32
    }

    #[target_feature(enable = "neon")]
    fn sum_8x8_frame(
        kpRefPicture: &[u8],
        kiWidth: i32,
        kiHeight: i32,
        kiRefStride: i32,
        pFeatureOfBlock: &mut [u16],
        pTimesOfFeatureValue: &mut [u32],
        _token: NeonToken,
    ) {
        if kiWidth <= 0 || kiHeight <= 0 {
            return;
        }
        let width = kiWidth as usize;
        let height = kiHeight as usize;
        let stride = kiRefStride as usize;

        // Row 0
        {
            let row0_feat = &mut pFeatureOfBlock[..width];
            let mut x = 0usize;
            while x + 8 < width {
                let mut acc = sliding_sum8(&kpRefPicture[x..]);
                for r in 1..8 {
                    acc = vaddq_u16(acc, sliding_sum8(&kpRefPicture[r * stride + x..]));
                }
                st8_u16(&mut row0_feat[x..x + 8], acc);
                for &s in &row0_feat[x..x + 8] {
                    pTimesOfFeatureValue[s as usize] += 1;
                }
                x += 8;
            }
            let mut sum = if x == 0 {
                let mut s = 0i32;
                for r in 0..8 {
                    let row = &kpRefPicture[r * stride..r * stride + 8];
                    for &b in row {
                        s += b as i32;
                    }
                }
                row0_feat[0] = s as u16;
                pTimesOfFeatureValue[s as usize] += 1;
                x = 1;
                s
            } else {
                row0_feat[x - 1] as i32
            };
            for tx in x..width {
                for r in 0..8 {
                    let row = &kpRefPicture[r * stride..];
                    sum += row[tx + 7] as i32 - row[tx - 1] as i32;
                }
                let s = sum as u16;
                row0_feat[tx] = s;
                pTimesOfFeatureValue[s as usize] += 1;
            }
        }

        // Subsequent rows y = 1..height
        for y in 1..height {
            let (prev_part, cur_part) = pFeatureOfBlock.split_at_mut(y * width);
            let prev_row = &prev_part[(y - 1) * width..y * width];
            let cur_row = &mut cur_part[..width];
            let top_row = &kpRefPicture[(y - 1) * stride..(y - 1) * stride + width + 7];
            let bot_row = &kpRefPicture[(y + 7) * stride..(y + 7) * stride + width + 7];

            let mut x = 0usize;
            while x + 8 < width {
                let prev = ld8_u16(&prev_row[x..x + 8]);
                let diff = sliding_diff8(&bot_row[x..x + 16], &top_row[x..x + 16]);
                let cur = vaddq_u16(prev, diff);
                st8_u16(&mut cur_row[x..x + 8], cur);
                for &s in &cur_row[x..x + 8] {
                    pTimesOfFeatureValue[s as usize] += 1;
                }
                x += 8;
            }

            let mut diff_sum = if x == 0 {
                let mut d = 0i32;
                for c in 0..8 {
                    d += bot_row[c] as i32 - top_row[c] as i32;
                }
                let s = (prev_row[0] as i32 + d) as u16;
                cur_row[0] = s;
                pTimesOfFeatureValue[s as usize] += 1;
                x = 1;
                d
            } else {
                cur_row[x - 1] as i32 - prev_row[x - 1] as i32
            };

            for tx in x..width {
                diff_sum += (bot_row[tx + 7] as i32 - top_row[tx + 7] as i32)
                    - (bot_row[tx - 1] as i32 - top_row[tx - 1] as i32);
                let s = (prev_row[tx] as i32 + diff_sum) as u16;
                cur_row[tx] = s;
                pTimesOfFeatureValue[s as usize] += 1;
            }
        }
    }

    #[target_feature(enable = "neon")]
    fn sum_16x16_frame(
        kpRefPicture: &[u8],
        kiWidth: i32,
        kiHeight: i32,
        kiRefStride: i32,
        pFeatureOfBlock: &mut [u16],
        pTimesOfFeatureValue: &mut [u32],
        _token: NeonToken,
    ) {
        if kiWidth <= 0 || kiHeight <= 0 {
            return;
        }
        let width = kiWidth as usize;
        let height = kiHeight as usize;
        let stride = kiRefStride as usize;

        // Row 0
        {
            let row0_feat = &mut pFeatureOfBlock[..width];
            let mut sum = 0i32;
            for r in 0..16 {
                let row = &kpRefPicture[r * stride..r * stride + 16];
                for &b in row {
                    sum += b as i32;
                }
            }
            row0_feat[0] = sum as u16;
            pTimesOfFeatureValue[sum as usize] += 1;

            for tx in 1..width {
                for r in 0..16 {
                    let row = &kpRefPicture[r * stride..];
                    sum += row[tx + 15] as i32 - row[tx - 1] as i32;
                }
                let s = sum as u16;
                row0_feat[tx] = s;
                pTimesOfFeatureValue[s as usize] += 1;
            }
        }

        // Subsequent rows y = 1..height
        for y in 1..height {
            let (prev_part, cur_part) = pFeatureOfBlock.split_at_mut(y * width);
            let prev_row = &prev_part[(y - 1) * width..y * width];
            let cur_row = &mut cur_part[..width];
            let top_row = &kpRefPicture[(y - 1) * stride..(y - 1) * stride + width + 15];
            let bot_row = &kpRefPicture[(y + 15) * stride..(y + 15) * stride + width + 15];

            let mut x = 0usize;
            if x + 8 < width {
                let mut curr_diff8 = sliding_diff8(&bot_row[0..16], &top_row[0..16]);
                while x + 8 < width {
                    let next_diff8 = sliding_diff8(&bot_row[x + 8..x + 24], &top_row[x + 8..x + 24]);
                    let diff16 = vaddq_u16(curr_diff8, next_diff8);
                    let prev = ld8_u16(&prev_row[x..x + 8]);
                    let cur = vaddq_u16(prev, diff16);
                    st8_u16(&mut cur_row[x..x + 8], cur);
                    for &s in &cur_row[x..x + 8] {
                        pTimesOfFeatureValue[s as usize] += 1;
                    }
                    curr_diff8 = next_diff8;
                    x += 8;
                }
            }

            let mut diff_sum = if x == 0 {
                let mut d = 0i32;
                for c in 0..16 {
                    d += bot_row[c] as i32 - top_row[c] as i32;
                }
                let s = (prev_row[0] as i32 + d) as u16;
                cur_row[0] = s;
                pTimesOfFeatureValue[s as usize] += 1;
                x = 1;
                d
            } else {
                cur_row[x - 1] as i32 - prev_row[x - 1] as i32
            };

            for tx in x..width {
                diff_sum += (bot_row[tx + 15] as i32 - top_row[tx + 15] as i32)
                    - (bot_row[tx - 1] as i32 - top_row[tx - 1] as i32);
                let s = (prev_row[tx] as i32 + diff_sum) as u16;
                cur_row[tx] = s;
                pTimesOfFeatureValue[s as usize] += 1;
            }
        }
    }
}

#[inline]
pub fn sum_of_8x8_single_block(cRef: &PlaneCursor<'_>) -> i32 {
    sum_8x8_single(cRef, neon_token())
}

#[inline]
pub fn sum_of_16x16_single_block(cRef: &PlaneCursor<'_>) -> i32 {
    sum_16x16_single(cRef, neon_token())
}

#[inline]
pub fn sum_of_8x8_block_of_frame(
    kpRefPicture: &[u8],
    kiWidth: i32,
    kiHeight: i32,
    kiRefStride: i32,
    pFeatureOfBlock: &mut [u16],
    pTimesOfFeatureValue: &mut [u32],
) {
    sum_8x8_frame(
        kpRefPicture,
        kiWidth,
        kiHeight,
        kiRefStride,
        pFeatureOfBlock,
        pTimesOfFeatureValue,
        neon_token(),
    );
}

#[inline]
pub fn sum_of_16x16_block_of_frame(
    kpRefPicture: &[u8],
    kiWidth: i32,
    kiHeight: i32,
    kiRefStride: i32,
    pFeatureOfBlock: &mut [u16],
    pTimesOfFeatureValue: &mut [u32],
) {
    sum_16x16_frame(
        kpRefPicture,
        kiWidth,
        kiHeight,
        kiRefStride,
        pFeatureOfBlock,
        pTimesOfFeatureValue,
        neon_token(),
    );
}

// Successive-elimination kernels (`svc_motion_estimate::sea_line_search`).

simd_entry! {
    #[target_feature(enable = "neon")]
    fn sea_rows16(c: &PlaneCursor<'_>, a0: &mut [u16], a1: &mut [u16], _token: NeonToken) {
        let rows = a0.len();
        let a1 = &mut a1[..rows];
        let mut r = 0usize;
        while r + 8 <= rows {
            let s = c.span::<16, 8>(r as isize, 0);
            let v = |y: usize| vpaddlq_u8(ld16(&s.row::<16>(y, 0)));
            let p01 = vpaddq_u16(v(0), v(1));
            let p23 = vpaddq_u16(v(2), v(3));
            let p45 = vpaddq_u16(v(4), v(5));
            let p67 = vpaddq_u16(v(6), v(7));
            let q0 = vpaddq_u16(p01, p23);
            let q1 = vpaddq_u16(p45, p67);
            st8_u16(&mut a0[r..], vuzp1q_u16(q0, q1));
            st8_u16(&mut a1[r..], vuzp2q_u16(q0, q1));
            r += 8;
        }
        if r < rows {
            svc_motion_estimate::sea_row_half_sums_c::<16>(
                &c.advance(0, r as isize),
                &mut a0[r..],
                &mut a1[r..],
            );
        }
    }

    #[target_feature(enable = "neon")]
    fn sea_rows8(c: &PlaneCursor<'_>, a0: &mut [u16], a1: &mut [u16], _token: NeonToken) {
        let rows = a0.len();
        let a1 = &mut a1[..rows];
        let mut r = 0usize;
        while r + 8 <= rows {
            let s = c.span::<8, 8>(r as isize, 0);
            let v = |y: usize| {
                vpaddlq_u8(vcombine_u8(ld8(&s.row::<8>(y, 0)), ld8(&s.row::<8>(y + 1, 0))))
            };
            let q0 = vpaddq_u16(v(0), v(2));
            let q1 = vpaddq_u16(v(4), v(6));
            st8_u16(&mut a0[r..], vuzp1q_u16(q0, q1));
            st8_u16(&mut a1[r..], vuzp2q_u16(q0, q1));
            r += 8;
        }
        if r < rows {
            svc_motion_estimate::sea_row_half_sums_c::<8>(
                &c.advance(0, r as isize),
                &mut a0[r..],
                &mut a1[r..],
            );
        }
    }

    /// A ragged tail overlaps the previous chunk rather than reading past the end.
    #[target_feature(enable = "neon")]
    fn sea_cols<const N: usize>(
        c: &PlaneCursor<'_>,
        a0: &mut [u16],
        a1: &mut [u16],
        _token: NeonToken,
    ) {
        let cols = a0.len();
        let a1 = &mut a1[..cols];
        if cols < 16 {
            svc_motion_estimate::sea_col_half_sums_c::<N>(c, a0, a1);
            return;
        }
        let mut x = 0usize;
        loop {
            let x0 = x.min(cols - 16);
            let s = c.span::<16, N>(0, x0 as isize);
            let z = vdupq_n_u16(0);
            let (mut t_lo, mut t_hi, mut b_lo, mut b_hi) = (z, z, z, z);
            for y in 0..N / 2 {
                let r = ld16(&s.row::<16>(y, 0));
                t_lo = vaddw_u8(t_lo, vget_low_u8(r));
                t_hi = vaddw_high_u8(t_hi, r);
            }
            for y in N / 2..N {
                let r = ld16(&s.row::<16>(y, 0));
                b_lo = vaddw_u8(b_lo, vget_low_u8(r));
                b_hi = vaddw_high_u8(b_hi, r);
            }
            st8_u16(&mut a0[x0..], t_lo);
            st8_u16(&mut a0[x0 + 8..], t_hi);
            st8_u16(&mut a1[x0..], b_lo);
            st8_u16(&mut a1[x0 + 8..], b_hi);
            x = x0 + 16;
            if x >= cols {
                break;
            }
        }
    }

    /// MV costs are every fourth `mvd` entry, gathered with two rounds of `uzp1`.
    #[target_feature(enable = "neon")]
    fn sea_bounds_neon(
        w0: &[u16],
        w1: &[u16],
        half: usize,
        e: [u16; 4],
        iFixedMvd: u16,
        mvd: &[u16],
        bound: &mut [u16],
        _token: NeonToken,
    ) {
        let n = bound.len();
        if n == 0 {
            return;
        }
        let (w0a, w0b) = (&w0[..n], &w0[half..half + n]);
        let (w1a, w1b) = (&w1[..n], &w1[half..half + n]);
        let mvd = &mvd[..4 * (n - 1) + 1];
        let fixed = vdupq_n_u16(iFixedMvd);
        let (e0, e1) = (vdupq_n_u16(e[0]), vdupq_n_u16(e[1]));
        let (e2, e3) = (vdupq_n_u16(e[2]), vdupq_n_u16(e[3]));
        let mut i = 0usize;
        while 4 * i + 32 <= mvd.len() {
            let d0 = vabdq_u16(ld8_u16(&w0a[i..]), e0);
            let d1 = vabdq_u16(ld8_u16(&w0b[i..]), e1);
            let d2 = vabdq_u16(ld8_u16(&w1a[i..]), e2);
            let d3 = vabdq_u16(ld8_u16(&w1b[i..]), e3);
            let lb = vaddq_u16(vaddq_u16(d0, d1), vaddq_u16(d2, d3));
            let m = &mvd[4 * i..4 * i + 32];
            let m01 = vuzp1q_u16(ld8_u16(&m[0..]), ld8_u16(&m[8..]));
            let m23 = vuzp1q_u16(ld8_u16(&m[16..]), ld8_u16(&m[24..]));
            let mv = vqaddq_u16(vuzp1q_u16(m01, m23), fixed);
            st8_u16(&mut bound[i..], vqaddq_u16(lb, mv));
            i += 8;
        }
        if i < n {
            svc_motion_estimate::sea_bounds_c(
                &w0[i..],
                &w1[i..],
                half,
                e,
                iFixedMvd,
                &mvd[4 * i..],
                &mut bound[i..],
            );
        }
    }
}

#[inline]
pub fn sea_row_half_sums<const N: usize>(c: &PlaneCursor<'_>, a0: &mut [u16], a1: &mut [u16]) {
    match N {
        16 => sea_rows16(c, a0, a1, neon_token()),
        8 => sea_rows8(c, a0, a1, neon_token()),
        _ => svc_motion_estimate::sea_row_half_sums_c::<N>(c, a0, a1),
    }
}

#[inline]
pub fn sea_col_half_sums<const N: usize>(c: &PlaneCursor<'_>, a0: &mut [u16], a1: &mut [u16]) {
    sea_cols::<N>(c, a0, a1, neon_token());
}

#[inline]
pub fn sea_bounds(
    w0: &[u16],
    w1: &[u16],
    half: usize,
    e: [u16; 4],
    iFixedMvd: u16,
    mvd: &[u16],
    bound: &mut [u16],
) {
    sea_bounds_neon(w0, w1, half, e, iFixedMvd, mvd, bound, neon_token());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::svc_motion_estimate::{
        LIST_SIZE_SUM_8x8, LIST_SIZE_SUM_16x16, SumOf8x8BlockOfFrame_c, SumOf16x16BlockOfFrame_c,
        sum_of_8x8_single_block as scalar_8x8_single,
        sum_of_16x16_single_block as scalar_16x16_single,
    };

    #[test]
    fn neon_me_kernels_match_scalar_over_tight_spans_and_extreme_values() {
        // Test every width 1..=25 (covering <8, ==8, ==9, ==16, ==17, ==24, ==25)
        // and heights 1, 2, 5 with exact-fit buffers so any 1-byte over-read panics.
        for width in 1..=25i32 {
            for height in [1i32, 2, 5] {
                // 8x8 exact-fit buffer: (height + 7 - 1) * stride + (width + 7)
                let stride8 = width + 7;
                let exact_len8 = ((height + 6) * stride8 + width + 7) as usize;
                let mut pic8 = vec![0u8; exact_len8];
                for (i, b) in pic8.iter_mut().enumerate() {
                    // Alternate 0, 255, and pseudo-random values to stress max +/-255 diffs
                    *b = match i % 5 {
                        0 => 0,
                        1 => 255,
                        _ => ((i * 97 + 31) ^ (i >> 2)) as u8,
                    };
                }

                let n = (width * height) as usize;
                let (mut want_f8, mut got_f8) = (vec![0u16; n], vec![0u16; n]);
                let (mut want_t8, mut got_t8) =
                    (vec![0u32; LIST_SIZE_SUM_8x8], vec![0u32; LIST_SIZE_SUM_8x8]);
                SumOf8x8BlockOfFrame_c(&pic8, width, height, stride8, &mut want_f8, &mut want_t8);
                sum_of_8x8_block_of_frame(&pic8, width, height, stride8, &mut got_f8, &mut got_t8);
                assert_eq!(got_f8, want_f8, "8x8 frame feat at {width}x{height}");
                assert_eq!(got_t8, want_t8, "8x8 frame times at {width}x{height}");

                // 16x16 exact-fit buffer: (height + 15 - 1) * stride + (width + 15)
                let stride16 = width + 15;
                let exact_len16 = ((height + 14) * stride16 + width + 15) as usize;
                let mut pic16 = vec![0u8; exact_len16];
                for (i, b) in pic16.iter_mut().enumerate() {
                    *b = match i % 5 {
                        0 => 255,
                        1 => 0,
                        _ => ((i * 151 + 17) ^ (i >> 3)) as u8,
                    };
                }

                let (mut want_f16, mut got_f16) = (vec![0u16; n], vec![0u16; n]);
                let (mut want_t16, mut got_t16) = (
                    vec![0u32; LIST_SIZE_SUM_16x16],
                    vec![0u32; LIST_SIZE_SUM_16x16],
                );
                SumOf16x16BlockOfFrame_c(
                    &pic16,
                    width,
                    height,
                    stride16,
                    &mut want_f16,
                    &mut want_t16,
                );
                sum_of_16x16_block_of_frame(
                    &pic16,
                    width,
                    height,
                    stride16,
                    &mut got_f16,
                    &mut got_t16,
                );
                assert_eq!(got_f16, want_f16, "16x16 frame feat at {width}x{height}");
                assert_eq!(got_t16, want_t16, "16x16 frame times at {width}x{height}");
            }
        }

        // All-255 and all-0 single block & frame saturation tests (max sum 16320 / 65280)
        let all_ff = vec![255u8; 32 * 32];
        let cursor_ff = PlaneCursor::new(&all_ff, 0, 32);
        assert_eq!(sum_of_8x8_single_block(&cursor_ff), 64 * 255);
        assert_eq!(
            sum_of_8x8_single_block(&cursor_ff),
            scalar_8x8_single(&cursor_ff)
        );
        assert_eq!(sum_of_16x16_single_block(&cursor_ff), 256 * 255);
        assert_eq!(
            sum_of_16x16_single_block(&cursor_ff),
            scalar_16x16_single(&cursor_ff)
        );
    }

    #[test]
    #[cfg(not(feature = "scalar"))]
    fn wels_init_me_func_installs_simd_kernels_when_cpu_flags_set() {
        use crate::common::cpu_core::{WELS_CPU_NEON, WELS_CPU_SSE2};
        use crate::encoder::svc_motion_estimate::WelsInitMeFunc;
        use crate::encoder::wels_func_ptr_def::SWelsFuncPtrList;

        let mut scalar_list = SWelsFuncPtrList::default();
        WelsInitMeFunc(&mut scalar_list, 0, true);

        let mut simd_list = SWelsFuncPtrList::default();
        WelsInitMeFunc(&mut simd_list, WELS_CPU_NEON | WELS_CPU_SSE2, true);

        for i in 0..2 {
            assert!(scalar_list.pfCalculateBlockFeatureOfFrame[i].is_some());
            assert!(simd_list.pfCalculateBlockFeatureOfFrame[i].is_some());
            assert_ne!(
                simd_list.pfCalculateBlockFeatureOfFrame[i].map(|f| f as usize),
                scalar_list.pfCalculateBlockFeatureOfFrame[i].map(|f| f as usize),
                "pfCalculateBlockFeatureOfFrame[{i}] should switch from scalar to SIMD"
            );

            assert!(scalar_list.sMeFuncs.pfCalculateSingleBlockFeature[i].is_some());
            assert!(simd_list.sMeFuncs.pfCalculateSingleBlockFeature[i].is_some());
            assert_ne!(
                simd_list.sMeFuncs.pfCalculateSingleBlockFeature[i].map(|f| f as usize),
                scalar_list.sMeFuncs.pfCalculateSingleBlockFeature[i].map(|f| f as usize),
                "pfCalculateSingleBlockFeature[{i}] should switch from scalar to SIMD"
            );
        }
    }
}
