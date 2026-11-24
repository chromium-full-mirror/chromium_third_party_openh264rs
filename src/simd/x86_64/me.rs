// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! x86_64 SSE2 & AVX2 implementations of screen-content motion-estimation feature kernels
//! (`codec/encoder/core/x86/sample_sc.asm`).

#![deny(unsafe_code)]

use super::lanes::{ld8, ld8_u16, ld16, st8_u16};
use super::sse2_token;
use crate::encoder::svc_motion_estimate;
use crate::safe::plane::{BlockRows, PlaneCursor, RefSamples};
use crate::simd::{
    simd_entry,
    tokens::{Avx2Token, Sse2Token},
};
use core::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse2")]
fn to8_u16(v: __m128i) -> [u16; 8] {
    let lo = (_mm_cvtsi128_si64(v) as u64).to_ne_bytes();
    let hi = (_mm_cvtsi128_si64(_mm_srli_si128::<8>(v)) as u64).to_ne_bytes();
    [
        u16::from_ne_bytes([lo[0], lo[1]]),
        u16::from_ne_bytes([lo[2], lo[3]]),
        u16::from_ne_bytes([lo[4], lo[5]]),
        u16::from_ne_bytes([lo[6], lo[7]]),
        u16::from_ne_bytes([hi[0], hi[1]]),
        u16::from_ne_bytes([hi[2], hi[3]]),
        u16::from_ne_bytes([hi[4], hi[5]]),
        u16::from_ne_bytes([hi[6], hi[7]]),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
fn ld16_u16(r: &[u16; 16]) -> __m256i {
    _mm256_setr_epi16(
        r[0] as i16,
        r[1] as i16,
        r[2] as i16,
        r[3] as i16,
        r[4] as i16,
        r[5] as i16,
        r[6] as i16,
        r[7] as i16,
        r[8] as i16,
        r[9] as i16,
        r[10] as i16,
        r[11] as i16,
        r[12] as i16,
        r[13] as i16,
        r[14] as i16,
        r[15] as i16,
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn to16_u16(v: __m256i) -> [u16; 16] {
    let a = to8_u16(_mm256_castsi256_si128(v));
    let b = to8_u16(_mm256_extracti128_si256::<1>(v));
    [
        a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], b[0], b[1], b[2], b[3], b[4], b[5], b[6],
        b[7],
    ]
}

#[inline]
#[target_feature(enable = "sse2")]
fn sum_window8_epi16(lo: __m128i, hi: __m128i) -> __m128i {
    let s2_lo = _mm_add_epi16(
        lo,
        _mm_or_si128(_mm_srli_si128(lo, 2), _mm_slli_si128(hi, 14)),
    );
    let s2_hi = _mm_add_epi16(hi, _mm_srli_si128(hi, 2));
    let s4_lo = _mm_add_epi16(
        s2_lo,
        _mm_or_si128(_mm_srli_si128(s2_lo, 4), _mm_slli_si128(s2_hi, 12)),
    );
    let s4_hi = _mm_add_epi16(s2_hi, _mm_srli_si128(s2_hi, 4));
    _mm_add_epi16(
        s4_lo,
        _mm_or_si128(_mm_srli_si128(s4_lo, 8), _mm_slli_si128(s4_hi, 8)),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn sum_window8_epi16_256(lo: __m256i, hi: __m256i) -> __m256i {
    let s2_lo = _mm256_add_epi16(
        lo,
        _mm256_or_si256(_mm256_srli_si256::<2>(lo), _mm256_slli_si256::<14>(hi)),
    );
    let s2_hi = _mm256_add_epi16(hi, _mm256_srli_si256::<2>(hi));
    let s4_lo = _mm256_add_epi16(
        s2_lo,
        _mm256_or_si256(
            _mm256_srli_si256::<4>(s2_lo),
            _mm256_slli_si256::<12>(s2_hi),
        ),
    );
    let s4_hi = _mm256_add_epi16(s2_hi, _mm256_srli_si256::<4>(s2_hi));
    _mm256_add_epi16(
        s4_lo,
        _mm256_or_si256(_mm256_srli_si256::<8>(s4_lo), _mm256_slli_si256::<8>(s4_hi)),
    )
}

#[inline]
#[target_feature(enable = "sse2")]
fn sliding_sum8(v: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let lo = _mm_unpacklo_epi8(v, zero);
    let hi = _mm_unpackhi_epi8(v, zero);
    sum_window8_epi16(lo, hi)
}

#[inline]
#[target_feature(enable = "avx2")]
fn sliding_sum8_256(v: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let lo = _mm256_unpacklo_epi8(v, zero);
    let hi = _mm256_unpackhi_epi8(v, zero);
    sum_window8_epi16_256(lo, hi)
}

#[inline]
#[target_feature(enable = "sse2")]
fn sliding_diff8_vec(b: __m128i, t: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let lo = _mm_sub_epi16(_mm_unpacklo_epi8(b, zero), _mm_unpacklo_epi8(t, zero));
    let hi = _mm_sub_epi16(_mm_unpackhi_epi8(b, zero), _mm_unpackhi_epi8(t, zero));
    sum_window8_epi16(lo, hi)
}

#[inline]
#[target_feature(enable = "avx2")]
fn sliding_diff8_256(b: __m256i, t: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let lo = _mm256_sub_epi16(_mm256_unpacklo_epi8(b, zero), _mm256_unpacklo_epi8(t, zero));
    let hi = _mm256_sub_epi16(_mm256_unpackhi_epi8(b, zero), _mm256_unpackhi_epi8(t, zero));
    sum_window8_epi16_256(lo, hi)
}

simd_entry! {
    #[target_feature(enable = "sse2")]
    fn sum_8x8_single(cRef: &PlaneCursor<'_>, _token: Sse2Token) -> i32 {
        let s = cRef.span::<8, 8>(0, 0);
        let ([q0, q1], st) = s.all_2_quads::<8>();
        let zero = _mm_setzero_si128();
        let (r0, r1, r2, r3) = <PlaneCursor<'_> as RefSamples>::Span::read_quad::<8>(q0, st);
        let (r4, r5, r6, r7) = <PlaneCursor<'_> as RefSamples>::Span::read_quad::<8>(q1, st);
        let mut acc = _mm_sad_epu8(ld8(&r0), zero);
        acc = _mm_add_epi64(acc, _mm_sad_epu8(ld8(&r1), zero));
        acc = _mm_add_epi64(acc, _mm_sad_epu8(ld8(&r2), zero));
        acc = _mm_add_epi64(acc, _mm_sad_epu8(ld8(&r3), zero));
        acc = _mm_add_epi64(acc, _mm_sad_epu8(ld8(&r4), zero));
        acc = _mm_add_epi64(acc, _mm_sad_epu8(ld8(&r5), zero));
        acc = _mm_add_epi64(acc, _mm_sad_epu8(ld8(&r6), zero));
        acc = _mm_add_epi64(acc, _mm_sad_epu8(ld8(&r7), zero));
        _mm_cvtsi128_si32(acc)
    }

    #[target_feature(enable = "sse2")]
    fn sum_16x16_single(cRef: &PlaneCursor<'_>, _token: Sse2Token) -> i32 {
        let s = cRef.span::<16, 16>(0, 0);
        let (qs, st) = s.all_4_quads::<16>();
        let zero = _mm_setzero_si128();
        let mut acc = _mm_setzero_si128();
        for q in qs {
            let (r0, r1, r2, r3) = <PlaneCursor<'_> as RefSamples>::Span::read_quad::<16>(q, st);
            acc = _mm_add_epi64(acc, _mm_sad_epu8(ld16(&r0), zero));
            acc = _mm_add_epi64(acc, _mm_sad_epu8(ld16(&r1), zero));
            acc = _mm_add_epi64(acc, _mm_sad_epu8(ld16(&r2), zero));
            acc = _mm_add_epi64(acc, _mm_sad_epu8(ld16(&r3), zero));
        }
        let hi = _mm_srli_si128(acc, 8);
        _mm_cvtsi128_si32(_mm_add_epi32(acc, hi))
    }

    #[target_feature(enable = "avx2")]
    fn sum_16x16_single_avx2(cRef: &PlaneCursor<'_>, _token: Avx2Token) -> i32 {
        let s = cRef.span::<16, 16>(0, 0);
        let (qs, st) = s.all_4_quads::<16>();
        let zero = _mm256_setzero_si256();
        let mut acc = _mm256_setzero_si256();
        for q in qs {
            let (r0, r1, r2, r3) = <PlaneCursor<'_> as RefSamples>::Span::read_quad::<16>(q, st);
            let v0 = _mm256_set_m128i(ld16(&r1), ld16(&r0));
            let v1 = _mm256_set_m128i(ld16(&r3), ld16(&r2));
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(v0, zero));
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(v1, zero));
        }
        let lo = _mm256_castsi256_si128(acc);
        let hi = _mm256_extracti128_si256::<1>(acc);
        let sum128 = _mm_add_epi64(lo, hi);
        _mm_cvtsi128_si32(_mm_add_epi32(sum128, _mm_srli_si128(sum128, 8)))
    }

    #[target_feature(enable = "avx2")]
    fn sum_8x8_frame_avx2(
        kpRefPicture: &[u8],
        kiWidth: i32,
        kiHeight: i32,
        kiRefStride: i32,
        pFeatureOfBlock: &mut [u16],
        pTimesOfFeatureValue: &mut [u32],
        _token: Avx2Token,
    ) {
        const LIST_8X8: usize = svc_motion_estimate::LIST_SIZE_SUM_8x8;
        if kiWidth <= 0 || kiHeight <= 0 {
            return;
        }
        let Some(hist) = pTimesOfFeatureValue.first_chunk_mut::<LIST_8X8>() else {
            return sum_8x8_frame(
                kpRefPicture,
                kiWidth,
                kiHeight,
                kiRefStride,
                pFeatureOfBlock,
                pTimesOfFeatureValue,
                sse2_token(),
            );
        };
        let width = kiWidth as usize;
        let height = kiHeight as usize;
        let stride = kiRefStride as usize;

        let avx2_end = ((width - 1) / 16) * 16;
        let simd_end = ((width - 1) / 8) * 8;
        let pic = &kpRefPicture[..(height + 6) * stride + width + 7];
        let mut feat_rows = pFeatureOfBlock[..height * width].chunks_exact_mut(width);

        // Row 0
        let mut prev_row: &[u16] = {
            let row0_feat = feat_rows.next().expect("height >= 1");
            let pic0 = &pic[..7 * stride + width + 7];
            let mut x = 0usize;
            if avx2_end > 0 {
                for (k, feat_chunk) in row0_feat[..avx2_end]
                    .as_chunks_mut::<16>()
                    .0
                    .iter_mut()
                    .enumerate()
                {
                    let cx = k * 16;
                    let v0 = _mm256_set_m128i(
                        ld16(&pic0[cx + 8..cx + 24]),
                        ld16(&pic0[cx..cx + 16]),
                    );
                    let mut acc = sliding_sum8_256(v0);
                    for r in 1..8 {
                        let off = r * stride + cx;
                        let vr = _mm256_set_m128i(
                            ld16(&pic0[off + 8..off + 24]),
                            ld16(&pic0[off..off + 16]),
                        );
                        acc = _mm256_add_epi16(acc, sliding_sum8_256(vr));
                    }
                    let vals = to16_u16(acc);
                    *feat_chunk = vals;
                    for s in vals {
                        hist[s as usize] += 1;
                    }
                }
                x = avx2_end;
            }
            if simd_end > x {
                let mut acc = sliding_sum8(ld16(&pic0[x..x + 16]));
                for r in 1..8 {
                    let off = r * stride + x;
                    acc = _mm_add_epi16(acc, sliding_sum8(ld16(&pic0[off..off + 16])));
                }
                let vals = to8_u16(acc);
                *row0_feat[x..x + 8].first_chunk_mut::<8>().unwrap() = vals;
                for s in vals {
                    hist[s as usize] += 1;
                }
                x = simd_end;
            }
            let mut sum = if x == 0 {
                let mut s = 0i32;
                for r in 0..8 {
                    for &b in &pic0[r * stride..r * stride + 8] {
                        s += b as i32;
                    }
                }
                row0_feat[0] = s as u16;
                hist[s as usize] += 1;
                x = 1;
                s
            } else {
                row0_feat[x - 1] as i32
            };
            for tx in x..width {
                for r in 0..8 {
                    let row = &pic0[r * stride..];
                    sum += row[tx + 7] as i32 - row[tx - 1] as i32;
                }
                let s = sum as u16;
                row0_feat[tx] = s;
                hist[s as usize] += 1;
            }
            row0_feat
        };

        // Subsequent rows y = 1..height
        for (ym1, cur_row) in feat_rows.enumerate() {
            let top_row = &pic[ym1 * stride..ym1 * stride + width + 7];
            let bot_row = &pic[(ym1 + 8) * stride..(ym1 + 8) * stride + width + 7];

            let mut x = 0usize;
            if avx2_end > 0 {
                let prev_ch = prev_row[..avx2_end].as_chunks::<16>().0;
                let cur_ch = cur_row[..avx2_end].as_chunks_mut::<16>().0;
                let b_ev = bot_row[..avx2_end].as_chunks::<16>().0;
                let b_od = bot_row[8..avx2_end + 8].as_chunks::<16>().0;
                let t_ev = top_row[..avx2_end].as_chunks::<16>().0;
                let t_od = top_row[8..avx2_end + 8].as_chunks::<16>().0;
                let n = prev_ch
                    .len()
                    .min(cur_ch.len())
                    .min(b_ev.len())
                    .min(b_od.len())
                    .min(t_ev.len())
                    .min(t_od.len());

                for i in 0..n {
                    let b = _mm256_set_m128i(ld16(&b_od[i]), ld16(&b_ev[i]));
                    let t = _mm256_set_m128i(ld16(&t_od[i]), ld16(&t_ev[i]));
                    let diff = sliding_diff8_256(b, t);
                    let vals = to16_u16(_mm256_add_epi16(ld16_u16(&prev_ch[i]), diff));
                    cur_ch[i] = vals;
                    for s in vals {
                        hist[s as usize] += 1;
                    }
                }
                x = avx2_end;
            }
            if simd_end > x {
                let prev = ld8_u16(&prev_row[x..x + 8]);
                let diff = sliding_diff8_vec(ld16(&bot_row[x..x + 16]), ld16(&top_row[x..x + 16]));
                let vals = to8_u16(_mm_add_epi16(prev, diff));
                *cur_row[x..x + 8].first_chunk_mut::<8>().unwrap() = vals;
                for s in vals {
                    hist[s as usize] += 1;
                }
                x = simd_end;
            }

            let mut diff_sum = if x == 0 {
                let mut d = 0i32;
                for c in 0..8 {
                    d += bot_row[c] as i32 - top_row[c] as i32;
                }
                let s = (prev_row[0] as i32 + d) as u16;
                cur_row[0] = s;
                hist[s as usize] += 1;
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
                hist[s as usize] += 1;
            }
            prev_row = cur_row;
        }
    }

    #[target_feature(enable = "sse2")]
    fn sum_8x8_frame(
        kpRefPicture: &[u8],
        kiWidth: i32,
        kiHeight: i32,
        kiRefStride: i32,
        pFeatureOfBlock: &mut [u16],
        pTimesOfFeatureValue: &mut [u32],
        _token: Sse2Token,
    ) {
        if kiWidth <= 0 || kiHeight <= 0 {
            return;
        }
        let width = kiWidth as usize;
        let height = kiHeight as usize;
        let stride = kiRefStride as usize;
        let simd_end = ((width - 1) / 8) * 8;
        let pic = &kpRefPicture[..(height + 6) * stride + width + 7];
        let mut feat_rows = pFeatureOfBlock[..height * width].chunks_exact_mut(width);

        // Row 0
        let mut prev_row: &[u16] = {
            let row0_feat = feat_rows.next().expect("height >= 1");
            let pic0 = &pic[..7 * stride + width + 7];
            let mut x = 0usize;
            if simd_end > 0 {
                for (k, feat_chunk) in row0_feat[..simd_end]
                    .as_chunks_mut::<8>()
                    .0
                    .iter_mut()
                    .enumerate()
                {
                    let cx = k * 8;
                    let mut acc = sliding_sum8(ld16(&pic0[cx..cx + 16]));
                    for r in 1..8 {
                        let off = r * stride + cx;
                        acc = _mm_add_epi16(acc, sliding_sum8(ld16(&pic0[off..off + 16])));
                    }
                    let vals = to8_u16(acc);
                    *feat_chunk = vals;
                    for s in vals {
                        pTimesOfFeatureValue[s as usize] += 1;
                    }
                }
                x = simd_end;
            }
            let mut sum = if x == 0 {
                let mut s = 0i32;
                for r in 0..8 {
                    for &b in &pic0[r * stride..r * stride + 8] {
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
                    let row = &pic0[r * stride..];
                    sum += row[tx + 7] as i32 - row[tx - 1] as i32;
                }
                let s = sum as u16;
                row0_feat[tx] = s;
                pTimesOfFeatureValue[s as usize] += 1;
            }
            row0_feat
        };

        // Subsequent rows y = 1..height
        for (ym1, cur_row) in feat_rows.enumerate() {
            let top_row = &pic[ym1 * stride..ym1 * stride + width + 7];
            let bot_row = &pic[(ym1 + 8) * stride..(ym1 + 8) * stride + width + 7];

            let mut x = 0usize;
            if simd_end > 0 {
                let prev_chunks = prev_row[..simd_end].as_chunks::<8>().0;
                let cur_chunks = cur_row[..simd_end].as_chunks_mut::<8>().0;
                let bot_chunks = bot_row[..simd_end + 8].as_chunks::<8>().0;
                let top_chunks = top_row[..simd_end + 8].as_chunks::<8>().0;
                let n = prev_chunks
                    .len()
                    .min(cur_chunks.len())
                    .min(bot_chunks.len() - 1)
                    .min(top_chunks.len() - 1);

                let mut b_lo = ld8(&bot_chunks[0]);
                let mut t_lo = ld8(&top_chunks[0]);
                for i in 0..n {
                    let b_hi = ld8(&bot_chunks[i + 1]);
                    let t_hi = ld8(&top_chunks[i + 1]);
                    let prev = ld8_u16(&prev_chunks[i]);
                    let diff = sliding_diff8_vec(
                        _mm_unpacklo_epi64(b_lo, b_hi),
                        _mm_unpacklo_epi64(t_lo, t_hi),
                    );
                    let vals = to8_u16(_mm_add_epi16(prev, diff));
                    cur_chunks[i] = vals;
                    for s in vals {
                        pTimesOfFeatureValue[s as usize] += 1;
                    }
                    b_lo = b_hi;
                    t_lo = t_hi;
                }
                x = simd_end;
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
            prev_row = cur_row;
        }
    }

    #[target_feature(enable = "avx2")]
    fn sum_16x16_frame_avx2(
        kpRefPicture: &[u8],
        kiWidth: i32,
        kiHeight: i32,
        kiRefStride: i32,
        pFeatureOfBlock: &mut [u16],
        pTimesOfFeatureValue: &mut [u32],
        _token: Avx2Token,
    ) {
        const LIST_16X16: usize = svc_motion_estimate::LIST_SIZE_SUM_16x16;
        if kiWidth <= 0 || kiHeight <= 0 {
            return;
        }
        let Some(hist) = pTimesOfFeatureValue.first_chunk_mut::<LIST_16X16>() else {
            return sum_16x16_frame(
                kpRefPicture,
                kiWidth,
                kiHeight,
                kiRefStride,
                pFeatureOfBlock,
                pTimesOfFeatureValue,
                sse2_token(),
            );
        };
        let width = kiWidth as usize;
        let height = kiHeight as usize;
        let stride = kiRefStride as usize;
        let n_avx2 = width.saturating_sub(9) / 16;
        let avx2_end = n_avx2 * 16;
        let simd_end = ((width - 1) / 8) * 8;
        let pic = &kpRefPicture[..(height + 14) * stride + width + 15];
        let mut feat_rows = pFeatureOfBlock[..height * width].chunks_exact_mut(width);

        // Row 0
        let mut prev_row: &[u16] = {
            let row0_feat = feat_rows.next().expect("height >= 1");
            let pic0 = &pic[..15 * stride + width + 15];
            let mut sum = 0i32;
            for r in 0..16 {
                for &b in &pic0[r * stride..r * stride + 16] {
                    sum += b as i32;
                }
            }
            row0_feat[0] = sum as u16;
            hist[sum as usize] += 1;

            for tx in 1..width {
                for r in 0..16 {
                    let row = &pic0[r * stride..];
                    sum += row[tx + 15] as i32 - row[tx - 1] as i32;
                }
                let s = sum as u16;
                row0_feat[tx] = s;
                hist[s as usize] += 1;
            }
            row0_feat
        };

        // Subsequent rows y = 1..height
        for (ym1, cur_row) in feat_rows.enumerate() {
            let top_row = &pic[ym1 * stride..ym1 * stride + width + 15];
            let bot_row = &pic[(ym1 + 16) * stride..(ym1 + 16) * stride + width + 15];

            let mut x = 0usize;
            if n_avx2 > 0 {
                let prev_ch = prev_row[..avx2_end].as_chunks::<16>().0;
                let cur_ch = cur_row[..avx2_end].as_chunks_mut::<16>().0;
                let b_ev = bot_row[..avx2_end + 16].as_chunks::<16>().0;
                let b_od = bot_row[8..avx2_end + 24].as_chunks::<16>().0;
                let t_ev = top_row[..avx2_end + 16].as_chunks::<16>().0;
                let t_od = top_row[8..avx2_end + 24].as_chunks::<16>().0;
                let n = prev_ch
                    .len()
                    .min(cur_ch.len())
                    .min(b_ev.len() - 1)
                    .min(b_od.len() - 1)
                    .min(t_ev.len() - 1)
                    .min(t_od.len() - 1);

                let mut curr_diff8 = sliding_diff8_256(
                    _mm256_set_m128i(ld16(&b_od[0]), ld16(&b_ev[0])),
                    _mm256_set_m128i(ld16(&t_od[0]), ld16(&t_ev[0])),
                );
                for i in 0..n {
                    let next_diff8 = sliding_diff8_256(
                        _mm256_set_m128i(ld16(&b_od[i + 1]), ld16(&b_ev[i + 1])),
                        _mm256_set_m128i(ld16(&t_od[i + 1]), ld16(&t_ev[i + 1])),
                    );
                    let shifted = _mm256_permute2x128_si256::<0x21>(curr_diff8, next_diff8);
                    let diff16 = _mm256_add_epi16(curr_diff8, shifted);
                    let vals = to16_u16(_mm256_add_epi16(ld16_u16(&prev_ch[i]), diff16));
                    cur_ch[i] = vals;
                    for s in vals {
                        hist[s as usize] += 1;
                    }
                    curr_diff8 = next_diff8;
                }
                x = avx2_end;
            }

            while x < simd_end {
                let d0 = sliding_diff8_vec(ld16(&bot_row[x..x + 16]), ld16(&top_row[x..x + 16]));
                let d1 = sliding_diff8_vec(
                    ld16(&bot_row[x + 8..x + 24]),
                    ld16(&top_row[x + 8..x + 24]),
                );
                let diff16 = _mm_add_epi16(d0, d1);
                let prev = ld8_u16(&prev_row[x..x + 8]);
                let vals = to8_u16(_mm_add_epi16(prev, diff16));
                *cur_row[x..x + 8].first_chunk_mut::<8>().unwrap() = vals;
                for s in vals {
                    hist[s as usize] += 1;
                }
                x += 8;
            }

            let mut diff_sum = if x == 0 {
                let mut d = 0i32;
                for c in 0..16 {
                    d += bot_row[c] as i32 - top_row[c] as i32;
                }
                let s = (prev_row[0] as i32 + d) as u16;
                cur_row[0] = s;
                hist[s as usize] += 1;
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
                hist[s as usize] += 1;
            }
            prev_row = cur_row;
        }
    }

    #[target_feature(enable = "sse2")]
    fn sum_16x16_frame(
        kpRefPicture: &[u8],
        kiWidth: i32,
        kiHeight: i32,
        kiRefStride: i32,
        pFeatureOfBlock: &mut [u16],
        pTimesOfFeatureValue: &mut [u32],
        _token: Sse2Token,
    ) {
        if kiWidth <= 0 || kiHeight <= 0 {
            return;
        }
        let width = kiWidth as usize;
        let height = kiHeight as usize;
        let stride = kiRefStride as usize;
        let simd_end = ((width - 1) / 8) * 8;
        let pic = &kpRefPicture[..(height + 14) * stride + width + 15];
        let mut feat_rows = pFeatureOfBlock[..height * width].chunks_exact_mut(width);

        // Row 0
        let mut prev_row: &[u16] = {
            let row0_feat = feat_rows.next().expect("height >= 1");
            let pic0 = &pic[..15 * stride + width + 15];
            let mut sum = 0i32;
            for r in 0..16 {
                for &b in &pic0[r * stride..r * stride + 16] {
                    sum += b as i32;
                }
            }
            row0_feat[0] = sum as u16;
            pTimesOfFeatureValue[sum as usize] += 1;

            for tx in 1..width {
                for r in 0..16 {
                    let row = &pic0[r * stride..];
                    sum += row[tx + 15] as i32 - row[tx - 1] as i32;
                }
                let s = sum as u16;
                row0_feat[tx] = s;
                pTimesOfFeatureValue[s as usize] += 1;
            }
            row0_feat
        };

        // Subsequent rows y = 1..height
        for (ym1, cur_row) in feat_rows.enumerate() {
            let top_row = &pic[ym1 * stride..ym1 * stride + width + 15];
            let bot_row = &pic[(ym1 + 16) * stride..(ym1 + 16) * stride + width + 15];

            let mut x = 0usize;
            if simd_end > 0 {
                let prev_chunks = prev_row[..simd_end].as_chunks::<8>().0;
                let cur_chunks = cur_row[..simd_end].as_chunks_mut::<8>().0;
                let bot_chunks = bot_row[..simd_end + 16].as_chunks::<8>().0;
                let top_chunks = top_row[..simd_end + 16].as_chunks::<8>().0;
                let n = prev_chunks
                    .len()
                    .min(cur_chunks.len())
                    .min(bot_chunks.len() - 2)
                    .min(top_chunks.len() - 2);

                let b0 = ld8(&bot_chunks[0]);
                let mut b_mid = ld8(&bot_chunks[1]);
                let t0 = ld8(&top_chunks[0]);
                let mut t_mid = ld8(&top_chunks[1]);
                let mut curr_diff8 = sliding_diff8_vec(
                    _mm_unpacklo_epi64(b0, b_mid),
                    _mm_unpacklo_epi64(t0, t_mid),
                );
                for i in 0..n {
                    let b_hi = ld8(&bot_chunks[i + 2]);
                    let t_hi = ld8(&top_chunks[i + 2]);
                    let next_diff8 = sliding_diff8_vec(
                        _mm_unpacklo_epi64(b_mid, b_hi),
                        _mm_unpacklo_epi64(t_mid, t_hi),
                    );
                    let diff16 = _mm_add_epi16(curr_diff8, next_diff8);
                    let prev = ld8_u16(&prev_chunks[i]);
                    let vals = to8_u16(_mm_add_epi16(prev, diff16));
                    cur_chunks[i] = vals;
                    for s in vals {
                        pTimesOfFeatureValue[s as usize] += 1;
                    }
                    curr_diff8 = next_diff8;
                    b_mid = b_hi;
                    t_mid = t_hi;
                }
                x = simd_end;
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
            prev_row = cur_row;
        }
    }
}

#[cold]
#[inline(never)]
fn sum_of_16x16_single_block_fallback(cRef: &PlaneCursor<'_>) -> i32 {
    if let Some(avx2) = Avx2Token::try_new_enabled() {
        return sum_16x16_single_avx2(cRef, avx2);
    }
    sum_16x16_single(cRef, sse2_token())
}

#[cold]
#[inline(never)]
fn sum_of_8x8_block_of_frame_fallback(
    kpRefPicture: &[u8],
    kiWidth: i32,
    kiHeight: i32,
    kiRefStride: i32,
    pFeatureOfBlock: &mut [u16],
    pTimesOfFeatureValue: &mut [u32],
) {
    if let Some(avx2) = Avx2Token::try_new_enabled() {
        return sum_8x8_frame_avx2(
            kpRefPicture,
            kiWidth,
            kiHeight,
            kiRefStride,
            pFeatureOfBlock,
            pTimesOfFeatureValue,
            avx2,
        );
    }
    sum_8x8_frame(
        kpRefPicture,
        kiWidth,
        kiHeight,
        kiRefStride,
        pFeatureOfBlock,
        pTimesOfFeatureValue,
        sse2_token(),
    )
}

#[cold]
#[inline(never)]
fn sum_of_16x16_block_of_frame_fallback(
    kpRefPicture: &[u8],
    kiWidth: i32,
    kiHeight: i32,
    kiRefStride: i32,
    pFeatureOfBlock: &mut [u16],
    pTimesOfFeatureValue: &mut [u32],
) {
    if let Some(avx2) = Avx2Token::try_new_enabled() {
        return sum_16x16_frame_avx2(
            kpRefPicture,
            kiWidth,
            kiHeight,
            kiRefStride,
            pFeatureOfBlock,
            pTimesOfFeatureValue,
            avx2,
        );
    }
    sum_16x16_frame(
        kpRefPicture,
        kiWidth,
        kiHeight,
        kiRefStride,
        pFeatureOfBlock,
        pTimesOfFeatureValue,
        sse2_token(),
    )
}

#[inline]
pub fn sum_of_8x8_single_block(cRef: &PlaneCursor<'_>) -> i32 {
    sum_8x8_single(cRef, sse2_token())
}

#[inline]
pub fn sum_of_16x16_single_block(cRef: &PlaneCursor<'_>) -> i32 {
    if let Some(avx2) = Avx2Token::try_fast_enabled() {
        return sum_16x16_single_avx2(cRef, avx2);
    }
    sum_of_16x16_single_block_fallback(cRef)
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
    if let Some(avx2) = Avx2Token::try_fast_enabled() {
        return sum_8x8_frame_avx2(
            kpRefPicture,
            kiWidth,
            kiHeight,
            kiRefStride,
            pFeatureOfBlock,
            pTimesOfFeatureValue,
            avx2,
        );
    }
    sum_of_8x8_block_of_frame_fallback(
        kpRefPicture,
        kiWidth,
        kiHeight,
        kiRefStride,
        pFeatureOfBlock,
        pTimesOfFeatureValue,
    )
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
    if let Some(avx2) = Avx2Token::try_fast_enabled() {
        return sum_16x16_frame_avx2(
            kpRefPicture,
            kiWidth,
            kiHeight,
            kiRefStride,
            pFeatureOfBlock,
            pTimesOfFeatureValue,
            avx2,
        );
    }
    sum_of_16x16_block_of_frame_fallback(
        kpRefPicture,
        kiWidth,
        kiHeight,
        kiRefStride,
        pFeatureOfBlock,
        pTimesOfFeatureValue,
    )
}

// Successive-elimination kernels (`svc_motion_estimate::sea_line_search`).

/// Four `psadbw` results packed to `[L0, R0, .., L3, R3]` (sums never clamp).
#[inline]
#[target_feature(enable = "sse2")]
fn sea_pack4(s0: __m128i, s1: __m128i, s2: __m128i, s3: __m128i) -> __m128i {
    _mm_packs_epi32(_mm_packs_epi32(s0, s1), _mm_packs_epi32(s2, s3))
}

#[inline]
#[target_feature(enable = "sse2")]
fn sea_unzip(q0: __m128i, q1: __m128i) -> (__m128i, __m128i) {
    let lo = _mm_set1_epi32(0xFFFF);
    let l = _mm_packs_epi32(_mm_and_si128(q0, lo), _mm_and_si128(q1, lo));
    let r = _mm_packs_epi32(_mm_srli_epi32::<16>(q0), _mm_srli_epi32::<16>(q1));
    (l, r)
}

#[inline]
#[target_feature(enable = "sse2")]
fn sea_lanes04(v: __m128i) -> __m128i {
    _mm_shufflelo_epi16::<0b11_01_10_00>(_mm_shuffle_epi32::<0b10_00_10_00>(v))
}

simd_entry! {
    #[target_feature(enable = "sse2")]
    fn sea_rows16(c: &PlaneCursor<'_>, a0: &mut [u16], a1: &mut [u16], _token: Sse2Token) {
        let rows = a0.len();
        let a1 = &mut a1[..rows];
        let z = _mm_setzero_si128();
        let mut r = 0usize;
        while r + 8 <= rows {
            let s = c.span::<16, 8>(r as isize, 0);
            let v = |y: usize| _mm_sad_epu8(ld16(&s.row::<16>(y, 0)), z);
            let q0 = sea_pack4(v(0), v(1), v(2), v(3));
            let q1 = sea_pack4(v(4), v(5), v(6), v(7));
            let (l, rr) = sea_unzip(q0, q1);
            st8_u16(&mut a0[r..], l);
            st8_u16(&mut a1[r..], rr);
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

    #[target_feature(enable = "sse2")]
    fn sea_rows8(c: &PlaneCursor<'_>, a0: &mut [u16], a1: &mut [u16], _token: Sse2Token) {
        let rows = a0.len();
        let a1 = &mut a1[..rows];
        let z = _mm_setzero_si128();
        let mut r = 0usize;
        while r + 8 <= rows {
            let s = c.span::<8, 8>(r as isize, 0);
            let v = |y: usize| _mm_sad_epu8(_mm_unpacklo_epi32(ld8(&s.row::<8>(y, 0)), z), z);
            let q0 = sea_pack4(v(0), v(1), v(2), v(3));
            let q1 = sea_pack4(v(4), v(5), v(6), v(7));
            let (l, rr) = sea_unzip(q0, q1);
            st8_u16(&mut a0[r..], l);
            st8_u16(&mut a1[r..], rr);
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
    #[target_feature(enable = "sse2")]
    fn sea_cols<const N: usize>(
        c: &PlaneCursor<'_>,
        a0: &mut [u16],
        a1: &mut [u16],
        _token: Sse2Token,
    ) {
        let cols = a0.len();
        let a1 = &mut a1[..cols];
        if cols < 16 {
            svc_motion_estimate::sea_col_half_sums_c::<N>(c, a0, a1);
            return;
        }
        let z = _mm_setzero_si128();
        let mut x = 0usize;
        loop {
            let x0 = x.min(cols - 16);
            let s = c.span::<16, N>(0, x0 as isize);
            let (mut t_lo, mut t_hi, mut b_lo, mut b_hi) = (z, z, z, z);
            for y in 0..N / 2 {
                let r = ld16(&s.row::<16>(y, 0));
                t_lo = _mm_add_epi16(t_lo, _mm_unpacklo_epi8(r, z));
                t_hi = _mm_add_epi16(t_hi, _mm_unpackhi_epi8(r, z));
            }
            for y in N / 2..N {
                let r = ld16(&s.row::<16>(y, 0));
                b_lo = _mm_add_epi16(b_lo, _mm_unpacklo_epi8(r, z));
                b_hi = _mm_add_epi16(b_hi, _mm_unpackhi_epi8(r, z));
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

    #[target_feature(enable = "sse2")]
    fn sea_bounds_sse2(
        w0: &[u16],
        w1: &[u16],
        half: usize,
        e: [u16; 4],
        iFixedMvd: u16,
        mvd: &[u16],
        bound: &mut [u16],
        _token: Sse2Token,
    ) {
        let n = bound.len();
        if n == 0 {
            return;
        }
        let (w0a, w0b) = (&w0[..n], &w0[half..half + n]);
        let (w1a, w1b) = (&w1[..n], &w1[half..half + n]);
        let mvd = &mvd[..4 * (n - 1) + 1];
        let fixed = _mm_set1_epi16(iFixedMvd as i16);
        let (e0, e1) = (_mm_set1_epi16(e[0] as i16), _mm_set1_epi16(e[1] as i16));
        let (e2, e3) = (_mm_set1_epi16(e[2] as i16), _mm_set1_epi16(e[3] as i16));
        let absd = |a: __m128i, e: __m128i| _mm_or_si128(_mm_subs_epu16(a, e), _mm_subs_epu16(e, a));
        let mut i = 0usize;
        while 4 * i + 32 <= mvd.len() {
            let d0 = absd(ld8_u16(&w0a[i..]), e0);
            let d1 = absd(ld8_u16(&w0b[i..]), e1);
            let d2 = absd(ld8_u16(&w1a[i..]), e2);
            let d3 = absd(ld8_u16(&w1b[i..]), e3);
            let lb = _mm_add_epi16(_mm_add_epi16(d0, d1), _mm_add_epi16(d2, d3));
            let m = &mvd[4 * i..4 * i + 32];
            let m01 = _mm_unpacklo_epi32(sea_lanes04(ld8_u16(&m[0..])), sea_lanes04(ld8_u16(&m[8..])));
            let m23 =
                _mm_unpacklo_epi32(sea_lanes04(ld8_u16(&m[16..])), sea_lanes04(ld8_u16(&m[24..])));
            let mv = _mm_adds_epu16(_mm_unpacklo_epi64(m01, m23), fixed);
            st8_u16(&mut bound[i..], _mm_adds_epu16(lb, mv));
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
        16 => sea_rows16(c, a0, a1, sse2_token()),
        8 => sea_rows8(c, a0, a1, sse2_token()),
        _ => svc_motion_estimate::sea_row_half_sums_c::<N>(c, a0, a1),
    }
}

#[inline]
pub fn sea_col_half_sums<const N: usize>(c: &PlaneCursor<'_>, a0: &mut [u16], a1: &mut [u16]) {
    sea_cols::<N>(c, a0, a1, sse2_token());
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
    sea_bounds_sse2(w0, w1, half, e, iFixedMvd, mvd, bound, sse2_token());
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
    fn sse2_me_kernels_match_scalar_over_tight_spans_and_extreme_values() {
        for width in 1..=45i32 {
            for height in [1i32, 2, 5] {
                let stride8 = width + 7;
                let exact_len8 = ((height + 6) * stride8 + width + 7) as usize;
                let mut pic8 = vec![0u8; exact_len8];
                for (i, b) in pic8.iter_mut().enumerate() {
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

                let (mut sse_f8, mut sse_t8) = (vec![0u16; n], vec![0u32; LIST_SIZE_SUM_8x8]);
                sum_8x8_frame(
                    &pic8,
                    width,
                    height,
                    stride8,
                    &mut sse_f8,
                    &mut sse_t8,
                    sse2_token(),
                );
                assert_eq!(sse_f8, want_f8, "sse2 8x8 frame feat at {width}x{height}");
                assert_eq!(sse_t8, want_t8, "sse2 8x8 frame times at {width}x{height}");

                if let Some(t) = Avx2Token::try_new() {
                    let (mut avx_f8, mut avx_t8) = (vec![0u16; n], vec![0u32; LIST_SIZE_SUM_8x8]);
                    sum_8x8_frame_avx2(&pic8, width, height, stride8, &mut avx_f8, &mut avx_t8, t);
                    assert_eq!(avx_f8, want_f8, "avx2 8x8 frame feat at {width}x{height}");
                    assert_eq!(avx_t8, want_t8, "avx2 8x8 frame times at {width}x{height}");
                }

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

                let (mut sse_f16, mut sse_t16) = (vec![0u16; n], vec![0u32; LIST_SIZE_SUM_16x16]);
                sum_16x16_frame(
                    &pic16,
                    width,
                    height,
                    stride16,
                    &mut sse_f16,
                    &mut sse_t16,
                    sse2_token(),
                );
                assert_eq!(
                    sse_f16, want_f16,
                    "sse2 16x16 frame feat at {width}x{height}"
                );
                assert_eq!(
                    sse_t16, want_t16,
                    "sse2 16x16 frame times at {width}x{height}"
                );

                if let Some(t) = Avx2Token::try_new() {
                    let (mut avx_f16, mut avx_t16) =
                        (vec![0u16; n], vec![0u32; LIST_SIZE_SUM_16x16]);
                    sum_16x16_frame_avx2(
                        &pic16,
                        width,
                        height,
                        stride16,
                        &mut avx_f16,
                        &mut avx_t16,
                        t,
                    );
                    assert_eq!(
                        avx_f16, want_f16,
                        "avx2 16x16 frame feat at {width}x{height}"
                    );
                    assert_eq!(
                        avx_t16, want_t16,
                        "avx2 16x16 frame times at {width}x{height}"
                    );
                }
            }
        }

        // Exercise multi-chunk AVX2 accumulation with pre-populated histogram counts.
        {
            let (w, h) = (67i32, 35i32);
            let s8 = w + 7;
            let pic: Vec<u8> = (0..((h + 6) * s8 + w + 7) as usize)
                .map(|i| ((i * 73 + 19) ^ (i >> 4)) as u8)
                .collect();
            let n = (w * h) as usize;
            let (mut wf, mut gf) = (vec![0u16; n], vec![0u16; n]);
            let (mut wt, mut gt) = (vec![3u32; LIST_SIZE_SUM_8x8], vec![3u32; LIST_SIZE_SUM_8x8]);
            SumOf8x8BlockOfFrame_c(&pic, w, h, s8, &mut wf, &mut wt);
            sum_of_8x8_block_of_frame(&pic, w, h, s8, &mut gf, &mut gt);
            assert_eq!(gf, wf, "large 8x8 frame feat");
            assert_eq!(gt, wt, "large 8x8 frame times");
            if let Some(t) = Avx2Token::try_new() {
                let mut af = vec![0u16; n];
                let mut at = vec![3u32; LIST_SIZE_SUM_8x8];
                sum_8x8_frame_avx2(&pic, w, h, s8, &mut af, &mut at, t);
                assert_eq!(af, wf, "avx2 large 8x8 frame feat");
                assert_eq!(at, wt, "avx2 large 8x8 frame times");
            }
        }

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
        if let Some(t) = Avx2Token::try_new() {
            assert_eq!(sum_16x16_single_avx2(&cursor_ff, t), 256 * 255);
        }
    }
}
