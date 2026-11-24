// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

#![forbid(unsafe_code)]
//! Scalar forwards for the `intra_pred` kernels — see the module header.

use crate::decoder::get_intra_predictor::{
    chroma_pred_dc, chroma_pred_dc_top, chroma_pred_h, chroma_pred_plane, chroma_pred_v,
    i4x4_luma_pred_dc, i4x4_luma_pred_ddl, i4x4_luma_pred_ddr, i4x4_luma_pred_h, i4x4_luma_pred_hd,
    i4x4_luma_pred_hu, i4x4_luma_pred_v, i4x4_luma_pred_vl, i4x4_luma_pred_vr, i16x16_luma_pred_dc,
    i16x16_luma_pred_dc_left, i16x16_luma_pred_dc_na, i16x16_luma_pred_dc_top, i16x16_luma_pred_h,
    i16x16_luma_pred_plane, i16x16_luma_pred_v,
};
use crate::encoder::get_intra_predictor::{
    WelsI4x4LumaPredDDL_c, WelsI4x4LumaPredDDR_c, WelsI4x4LumaPredDc_c, WelsI4x4LumaPredH_c,
    WelsI4x4LumaPredHD_c, WelsI4x4LumaPredHU_c, WelsI4x4LumaPredV_c, WelsI4x4LumaPredVL_c,
    WelsI4x4LumaPredVR_c, WelsI16x16LumaPredDc_c, WelsI16x16LumaPredH_c, WelsI16x16LumaPredPlane_c,
    WelsI16x16LumaPredV_c, WelsIChromaPredDc_c, WelsIChromaPredH_c, WelsIChromaPredPlane_c,
    WelsIChromaPredV_c,
};
use crate::safe::plane::{PlaneCursor, PlaneCursorMut, RefSamples};

#[inline(always)]
pub fn enc_i16x16_luma_pred_v(pred: &mut [u8; 256], rec: &PlaneCursor<'_>) {
    WelsI16x16LumaPredV_c(pred, rec)
}

#[inline(always)]
pub fn dec_i16x16_luma_pred_v(pred: &mut PlaneCursorMut<'_>) {
    i16x16_luma_pred_v(pred)
}

#[inline(always)]
pub fn enc_i16x16_luma_pred_h(pred: &mut [u8; 256], rec: &PlaneCursor<'_>) {
    WelsI16x16LumaPredH_c(pred, rec)
}

#[inline(always)]
pub fn dec_i16x16_luma_pred_h(pred: &mut PlaneCursorMut<'_>) {
    i16x16_luma_pred_h(pred)
}

#[inline(always)]
pub fn enc_i16x16_luma_pred_dc(pred: &mut [u8; 256], rec: &PlaneCursor<'_>) {
    WelsI16x16LumaPredDc_c(pred, rec)
}

#[inline(always)]
pub fn dec_i16x16_luma_pred_dc(pred: &mut PlaneCursorMut<'_>) {
    i16x16_luma_pred_dc(pred)
}

#[inline(always)]
pub fn dec_i16x16_luma_pred_dc_top(pred: &mut PlaneCursorMut<'_>) {
    i16x16_luma_pred_dc_top(pred)
}

#[inline(always)]
pub fn dec_i16x16_luma_pred_dc_left(pred: &mut PlaneCursorMut<'_>) {
    i16x16_luma_pred_dc_left(pred)
}

#[inline(always)]
pub fn dec_i16x16_luma_pred_dc_na(pred: &mut PlaneCursorMut<'_>) {
    i16x16_luma_pred_dc_na(pred)
}

#[inline(always)]
pub fn enc_i16x16_luma_pred_plane(pred: &mut [u8; 256], rec: &PlaneCursor<'_>) {
    WelsI16x16LumaPredPlane_c(pred, rec)
}

#[inline(always)]
pub fn dec_i16x16_luma_pred_plane(pred: &mut PlaneCursorMut<'_>) {
    i16x16_luma_pred_plane(pred)
}

#[inline(always)]
pub fn enc_chroma_pred_v(pred: &mut [u8; 64], rec: &PlaneCursor<'_>) {
    WelsIChromaPredV_c(pred, rec)
}

#[inline(always)]
pub fn dec_chroma_pred_v(pred: &mut PlaneCursorMut<'_>) {
    chroma_pred_v(pred)
}

#[inline(always)]
pub fn enc_chroma_pred_h(pred: &mut [u8; 64], rec: &PlaneCursor<'_>) {
    WelsIChromaPredH_c(pred, rec)
}

#[inline(always)]
pub fn dec_chroma_pred_h(pred: &mut PlaneCursorMut<'_>) {
    chroma_pred_h(pred)
}

#[inline(always)]
pub fn enc_chroma_pred_dc(pred: &mut [u8; 64], rec: &PlaneCursor<'_>) {
    WelsIChromaPredDc_c(pred, rec)
}

#[inline(always)]
pub fn dec_chroma_pred_dc(pred: &mut PlaneCursorMut<'_>) {
    chroma_pred_dc(pred)
}

#[inline(always)]
pub fn dec_chroma_pred_dc_top(pred: &mut PlaneCursorMut<'_>) {
    chroma_pred_dc_top(pred)
}

#[inline(always)]
pub fn enc_chroma_pred_plane(pred: &mut [u8; 64], rec: &PlaneCursor<'_>) {
    WelsIChromaPredPlane_c(pred, rec)
}

#[inline(always)]
pub fn dec_chroma_pred_plane(pred: &mut PlaneCursorMut<'_>) {
    chroma_pred_plane(pred)
}

#[inline(always)]
pub fn enc_i4x4_luma_pred_v(pred: &mut [u8; 16], rec: &PlaneCursor<'_>) {
    WelsI4x4LumaPredV_c(pred, rec)
}

#[inline(always)]
pub fn dec_i4x4_luma_pred_v(pred: &mut PlaneCursorMut<'_>) {
    i4x4_luma_pred_v(pred)
}

#[inline(always)]
pub fn enc_i4x4_luma_pred_h(pred: &mut [u8; 16], rec: &PlaneCursor<'_>) {
    WelsI4x4LumaPredH_c(pred, rec)
}

#[inline(always)]
pub fn dec_i4x4_luma_pred_h(pred: &mut PlaneCursorMut<'_>) {
    i4x4_luma_pred_h(pred)
}

#[inline(always)]
pub fn enc_i4x4_luma_pred_dc(pred: &mut [u8; 16], rec: &PlaneCursor<'_>) {
    WelsI4x4LumaPredDc_c(pred, rec)
}

#[inline(always)]
pub fn dec_i4x4_luma_pred_dc(pred: &mut PlaneCursorMut<'_>) {
    i4x4_luma_pred_dc(pred)
}

#[inline(always)]
pub fn enc_i4x4_luma_pred_ddl(pred: &mut [u8; 16], rec: &PlaneCursor<'_>) {
    WelsI4x4LumaPredDDL_c(pred, rec)
}

#[inline(always)]
pub fn dec_i4x4_luma_pred_ddl(pred: &mut PlaneCursorMut<'_>) {
    i4x4_luma_pred_ddl(pred)
}

#[inline(always)]
pub fn enc_i4x4_luma_pred_ddr(pred: &mut [u8; 16], rec: &PlaneCursor<'_>) {
    WelsI4x4LumaPredDDR_c(pred, rec)
}

#[inline(always)]
pub fn dec_i4x4_luma_pred_ddr(pred: &mut PlaneCursorMut<'_>) {
    i4x4_luma_pred_ddr(pred)
}

#[inline(always)]
pub fn enc_i4x4_luma_pred_vr(pred: &mut [u8; 16], rec: &PlaneCursor<'_>) {
    WelsI4x4LumaPredVR_c(pred, rec)
}

#[inline(always)]
pub fn dec_i4x4_luma_pred_vr(pred: &mut PlaneCursorMut<'_>) {
    i4x4_luma_pred_vr(pred)
}

#[inline(always)]
pub fn enc_i4x4_luma_pred_hd(pred: &mut [u8; 16], rec: &PlaneCursor<'_>) {
    WelsI4x4LumaPredHD_c(pred, rec)
}

#[inline(always)]
pub fn dec_i4x4_luma_pred_hd(pred: &mut PlaneCursorMut<'_>) {
    i4x4_luma_pred_hd(pred)
}

#[inline(always)]
pub fn enc_i4x4_luma_pred_vl(pred: &mut [u8; 16], rec: &PlaneCursor<'_>) {
    WelsI4x4LumaPredVL_c(pred, rec)
}

#[inline(always)]
pub fn dec_i4x4_luma_pred_vl(pred: &mut PlaneCursorMut<'_>) {
    i4x4_luma_pred_vl(pred)
}

#[inline(always)]
pub fn enc_i4x4_luma_pred_hu(pred: &mut [u8; 16], rec: &PlaneCursor<'_>) {
    WelsI4x4LumaPredHU_c(pred, rec)
}

#[inline(always)]
pub fn dec_i4x4_luma_pred_hu(pred: &mut PlaneCursorMut<'_>) {
    i4x4_luma_pred_hu(pred)
}

#[inline(always)]
pub fn intra_16x16_combined3_sad(
    pred: &mut [u8; 256],
    rec: &PlaneCursor<'_>,
    enc: &PlaneCursor<'_>,
    lambda: i32,
) -> (u8, i32) {
    let top = rec.row_n::<16>(-1, 0);
    let mut sum_top: i32 = 0;
    for &x in &top {
        sum_top += x as i32;
    }
    let mut left = [0u8; 16];
    let mut sum_left: i32 = 0;
    for y in 0..16 {
        let val = rec.at(-1, y as isize);
        left[y] = val;
        sum_left += val as i32;
    }

    let dc_val = ((16 + sum_top + sum_left) >> 5) as u8;

    let mut sad_v: i32 = 0;
    let mut sad_h: i32 = 0;
    let mut sad_dc: i32 = 0;

    for y in 0..16 {
        let enc_row = enc.row_n::<16>(y as isize, 0);
        let h_val = left[y];
        for x in 0..16 {
            let p = enc_row[x] as i32;
            sad_v += (p - top[x] as i32).abs();
            sad_h += (p - h_val as i32).abs();
            sad_dc += (p - dc_val as i32).abs();
        }
    }

    let cost_v = sad_v + lambda * 1;
    let cost_h = sad_h + lambda * 3;
    let cost_dc = sad_dc + lambda * 3;

    let (best_mode, best_cost) = if cost_dc < cost_h && cost_dc < cost_v {
        (2u8, cost_dc)
    } else if cost_h < cost_v {
        (1u8, cost_h)
    } else {
        (0u8, cost_v)
    };

    match best_mode {
        0 => {
            for y in 0..16 {
                pred[y * 16..(y + 1) * 16].copy_from_slice(&top);
            }
        }
        1 => {
            for y in 0..16 {
                pred[y * 16..(y + 1) * 16].fill(left[y]);
            }
        }
        _ => {
            pred.fill(dc_val);
        }
    }

    (best_mode, best_cost)
}

#[inline(always)]
fn hdm4_1d(a0: i32, a1: i32, a2: i32, a3: i32) -> [i32; 4] {
    let p0 = a0 + a1;
    let p1 = a2 + a3;
    let m0 = a0 - a1;
    let m1 = a2 - a3;
    [p0 + p1, p0 - p1, m0 + m1, m0 - m1]
}

#[inline(always)]
fn satd_4x4_modes(
    enc: &PlaneCursor<'_>,
    dx: isize,
    dy: isize,
    ht: &[i32; 4],
    hl: &[i32; 4],
    dc16: i32,
) -> (i32, i32, i32) {
    let mut h_row = [[0i32; 4]; 4];
    for (y, dst) in h_row.iter_mut().enumerate() {
        let r = enc.row_n::<4>(dy + y as isize, dx);
        *dst = hdm4_1d(r[0] as i32, r[1] as i32, r[2] as i32, r[3] as i32);
    }
    let mut f = [[0i32; 4]; 4];
    for v in 0..4 {
        let col = hdm4_1d(h_row[0][v], h_row[1][v], h_row[2][v], h_row[3][v]);
        for u in 0..4 {
            f[u][v] = col[u];
        }
    }
    let mut sum_v = 0i32;
    let mut sum_h = 0i32;
    let mut sum_dc = (f[0][0] - dc16).abs();
    for v in 0..4 {
        sum_v += (f[0][v] - (ht[v] << 2)).abs();
    }
    for u in 1..4 {
        for v in 0..4 {
            sum_v += f[u][v].abs();
        }
        sum_dc += f[u][0].abs();
    }
    for u in 0..4 {
        sum_h += (f[u][0] - (hl[u] << 2)).abs();
        for v in 1..4 {
            let a = f[u][v].abs();
            sum_h += a;
            sum_dc += a;
        }
    }
    (sum_v >> 1, sum_h >> 1, sum_dc >> 1)
}

#[inline(always)]
pub fn intra_4x4_combined3_satd(
    pred: &mut [u8; 16],
    rec: &PlaneCursor<'_>,
    enc: &PlaneCursor<'_>,
    lambda_dc: i32,
    lambda_h: i32,
    lambda_v: i32,
) -> (i8, i32) {
    let top = rec.row_n::<4>(-1, 0);
    let left = [rec.at(-1, 0), rec.at(-1, 1), rec.at(-1, 2), rec.at(-1, 3)];
    let ht = hdm4_1d(top[0] as i32, top[1] as i32, top[2] as i32, top[3] as i32);
    let hl = hdm4_1d(
        left[0] as i32,
        left[1] as i32,
        left[2] as i32,
        left[3] as i32,
    );
    let dc_val = ((ht[0] + hl[0] + 4) >> 3) as u8;

    let (satd_v, satd_h, satd_dc) = satd_4x4_modes(enc, 0, 0, &ht, &hl, (dc_val as i32) << 4);
    let cost_dc = satd_dc + lambda_dc;
    let cost_h = satd_h + lambda_h;
    let cost_v = satd_v + lambda_v;

    if cost_dc <= cost_h && cost_dc <= cost_v {
        pred.fill(dc_val);
        (2i8, cost_dc)
    } else if cost_h <= cost_v {
        for y in 0..4 {
            pred[y * 4..(y + 1) * 4].fill(left[y]);
        }
        (1i8, cost_h)
    } else {
        for y in 0..4 {
            pred[y * 4..(y + 1) * 4].copy_from_slice(&top);
        }
        (0i8, cost_v)
    }
}

#[inline(always)]
pub fn intra_16x16_combined3_satd(
    pred: &mut [u8; 256],
    rec: &PlaneCursor<'_>,
    enc: &PlaneCursor<'_>,
    lambda: i32,
) -> (u8, i32) {
    let top = rec.row_n::<16>(-1, 0);
    let mut left = [0u8; 16];
    for (y, dst) in left.iter_mut().enumerate() {
        *dst = rec.at(-1, y as isize);
    }

    let mut ht = [[0i32; 4]; 4];
    let mut hl = [[0i32; 4]; 4];
    for b in 0..4 {
        let o = b * 4;
        ht[b] = hdm4_1d(
            top[o] as i32,
            top[o + 1] as i32,
            top[o + 2] as i32,
            top[o + 3] as i32,
        );
        hl[b] = hdm4_1d(
            left[o] as i32,
            left[o + 1] as i32,
            left[o + 2] as i32,
            left[o + 3] as i32,
        );
    }

    let sum_top = ht[0][0] + ht[1][0] + ht[2][0] + ht[3][0];
    let sum_left = hl[0][0] + hl[1][0] + hl[2][0] + hl[3][0];
    let dc_val = ((16 + sum_top + sum_left) >> 5) as u8;
    let dc16 = (dc_val as i32) << 4;

    let mut satd_v = 0i32;
    let mut satd_h = 0i32;
    let mut satd_dc = 0i32;
    for by in 0..4 {
        for bx in 0..4 {
            let (sv, sh, sdc) = satd_4x4_modes(
                enc,
                (bx * 4) as isize,
                (by * 4) as isize,
                &ht[bx],
                &hl[by],
                dc16,
            );
            satd_v += sv;
            satd_h += sh;
            satd_dc += sdc;
        }
    }

    let cost_v = satd_v + lambda;
    let cost_h = satd_h + lambda * 3;
    let cost_dc = satd_dc + lambda * 3;

    let (best_mode, best_cost) = if cost_dc < cost_h && cost_dc < cost_v {
        (2u8, cost_dc)
    } else if cost_h < cost_v {
        (1u8, cost_h)
    } else {
        (0u8, cost_v)
    };

    match best_mode {
        0 => {
            for y in 0..16 {
                pred[y * 16..(y + 1) * 16].copy_from_slice(&top);
            }
        }
        1 => {
            for y in 0..16 {
                pred[y * 16..(y + 1) * 16].fill(left[y]);
            }
        }
        _ => {
            pred.fill(dc_val);
        }
    }

    (best_mode, best_cost)
}

#[inline(always)]
fn chroma_dc_quads(top: &[u8; 8], left: &[u8; 8]) -> ([u8; 4], [u8; 8], [u8; 8]) {
    let t0 = top[0] as i32 + top[1] as i32 + top[2] as i32 + top[3] as i32;
    let t1 = top[4] as i32 + top[5] as i32 + top[6] as i32 + top[7] as i32;
    let l0 = left[0] as i32 + left[1] as i32 + left[2] as i32 + left[3] as i32;
    let l1 = left[4] as i32 + left[5] as i32 + left[6] as i32 + left[7] as i32;
    let m0 = ((t0 + l0 + 4) >> 3) as u8;
    let m1 = ((t1 + 2) >> 2) as u8;
    let m2 = ((l1 + 2) >> 2) as u8;
    let m3 = ((t1 + l1 + 4) >> 3) as u8;
    (
        [m0, m1, m2, m3],
        [m0, m0, m0, m0, m1, m1, m1, m1],
        [m2, m2, m2, m2, m3, m3, m3, m3],
    )
}

#[inline(always)]
fn write_chroma_pred(
    dst: &mut [u8; 64],
    mode: u8,
    top: &[u8; 8],
    left: &[u8; 8],
    dc_top: &[u8; 8],
    dc_bot: &[u8; 8],
) {
    match mode {
        2 => {
            for y in 0..8 {
                dst[y * 8..(y + 1) * 8].copy_from_slice(top);
            }
        }
        1 => {
            for y in 0..8 {
                dst[y * 8..(y + 1) * 8].fill(left[y]);
            }
        }
        _ => {
            for y in 0..4 {
                dst[y * 8..(y + 1) * 8].copy_from_slice(dc_top);
            }
            for y in 4..8 {
                dst[y * 8..(y + 1) * 8].copy_from_slice(dc_bot);
            }
        }
    }
}

#[inline(always)]
pub fn intra_chroma_8x8_combined3_sad(
    pred_cb_cr: &mut [u8; 128],
    rec_cb: &PlaneCursor<'_>,
    enc_cb: &PlaneCursor<'_>,
    rec_cr: &PlaneCursor<'_>,
    enc_cr: &PlaneCursor<'_>,
    lambda: i32,
) -> (u8, i32) {
    let top_cb = rec_cb.row_n::<8>(-1, 0);
    let top_cr = rec_cr.row_n::<8>(-1, 0);
    let mut left_cb = [0u8; 8];
    let mut left_cr = [0u8; 8];
    for y in 0..8 {
        left_cb[y] = rec_cb.at(-1, y as isize);
        left_cr[y] = rec_cr.at(-1, y as isize);
    }
    let (_, dc_top_cb, dc_bot_cb) = chroma_dc_quads(&top_cb, &left_cb);
    let (_, dc_top_cr, dc_bot_cr) = chroma_dc_quads(&top_cr, &left_cr);

    let mut sad_v = 0i32;
    let mut sad_h = 0i32;
    let mut sad_dc = 0i32;

    for y in 0..8 {
        let e_cb = enc_cb.row_n::<8>(y as isize, 0);
        let e_cr = enc_cr.row_n::<8>(y as isize, 0);
        let h_cb = left_cb[y] as i32;
        let h_cr = left_cr[y] as i32;
        let dc_cb = if y < 4 { &dc_top_cb } else { &dc_bot_cb };
        let dc_cr = if y < 4 { &dc_top_cr } else { &dc_bot_cr };
        for x in 0..8 {
            let pcb = e_cb[x] as i32;
            sad_v += (pcb - top_cb[x] as i32).abs();
            sad_h += (pcb - h_cb).abs();
            sad_dc += (pcb - dc_cb[x] as i32).abs();

            let pcr = e_cr[x] as i32;
            sad_v += (pcr - top_cr[x] as i32).abs();
            sad_h += (pcr - h_cr).abs();
            sad_dc += (pcr - dc_cr[x] as i32).abs();
        }
    }

    let cost_dc = sad_dc + lambda;
    let cost_h = sad_h + lambda * 3;
    let cost_v = sad_v + lambda * 3;

    let (best_mode, best_cost) = if cost_dc < cost_h && cost_dc < cost_v {
        (0u8, cost_dc)
    } else if cost_h < cost_v {
        (1u8, cost_h)
    } else {
        (2u8, cost_v)
    };

    let (cb_dst, cr_dst) = pred_cb_cr.split_at_mut(64);
    write_chroma_pred(
        cb_dst.first_chunk_mut::<64>().expect("cb 64"),
        best_mode,
        &top_cb,
        &left_cb,
        &dc_top_cb,
        &dc_bot_cb,
    );
    write_chroma_pred(
        cr_dst.first_chunk_mut::<64>().expect("cr 64"),
        best_mode,
        &top_cr,
        &left_cr,
        &dc_top_cr,
        &dc_bot_cr,
    );

    (best_mode, best_cost)
}

#[inline(always)]
pub fn intra_chroma_8x8_combined3_satd(
    pred_cb_cr: &mut [u8; 128],
    rec_cb: &PlaneCursor<'_>,
    enc_cb: &PlaneCursor<'_>,
    rec_cr: &PlaneCursor<'_>,
    enc_cr: &PlaneCursor<'_>,
    lambda: i32,
) -> (u8, i32) {
    let top_cb = rec_cb.row_n::<8>(-1, 0);
    let top_cr = rec_cr.row_n::<8>(-1, 0);
    let mut left_cb = [0u8; 8];
    let mut left_cr = [0u8; 8];
    for y in 0..8 {
        left_cb[y] = rec_cb.at(-1, y as isize);
        left_cr[y] = rec_cr.at(-1, y as isize);
    }
    let (dc_cb, dc_top_cb, dc_bot_cb) = chroma_dc_quads(&top_cb, &left_cb);
    let (dc_cr, dc_top_cr, dc_bot_cr) = chroma_dc_quads(&top_cr, &left_cr);

    let ht_cb = [
        hdm4_1d(
            top_cb[0] as i32,
            top_cb[1] as i32,
            top_cb[2] as i32,
            top_cb[3] as i32,
        ),
        hdm4_1d(
            top_cb[4] as i32,
            top_cb[5] as i32,
            top_cb[6] as i32,
            top_cb[7] as i32,
        ),
    ];
    let hl_cb = [
        hdm4_1d(
            left_cb[0] as i32,
            left_cb[1] as i32,
            left_cb[2] as i32,
            left_cb[3] as i32,
        ),
        hdm4_1d(
            left_cb[4] as i32,
            left_cb[5] as i32,
            left_cb[6] as i32,
            left_cb[7] as i32,
        ),
    ];
    let ht_cr = [
        hdm4_1d(
            top_cr[0] as i32,
            top_cr[1] as i32,
            top_cr[2] as i32,
            top_cr[3] as i32,
        ),
        hdm4_1d(
            top_cr[4] as i32,
            top_cr[5] as i32,
            top_cr[6] as i32,
            top_cr[7] as i32,
        ),
    ];
    let hl_cr = [
        hdm4_1d(
            left_cr[0] as i32,
            left_cr[1] as i32,
            left_cr[2] as i32,
            left_cr[3] as i32,
        ),
        hdm4_1d(
            left_cr[4] as i32,
            left_cr[5] as i32,
            left_cr[6] as i32,
            left_cr[7] as i32,
        ),
    ];

    let mut satd_v = 0i32;
    let mut satd_h = 0i32;
    let mut satd_dc = 0i32;
    for by in 0..2 {
        for bx in 0..2 {
            let q = by * 2 + bx;
            let (sv_b, sh_b, sdc_b) = satd_4x4_modes(
                enc_cb,
                (bx * 4) as isize,
                (by * 4) as isize,
                &ht_cb[bx],
                &hl_cb[by],
                (dc_cb[q] as i32) << 4,
            );
            let (sv_r, sh_r, sdc_r) = satd_4x4_modes(
                enc_cr,
                (bx * 4) as isize,
                (by * 4) as isize,
                &ht_cr[bx],
                &hl_cr[by],
                (dc_cr[q] as i32) << 4,
            );
            satd_v += sv_b + sv_r;
            satd_h += sh_b + sh_r;
            satd_dc += sdc_b + sdc_r;
        }
    }

    let cost_dc = satd_dc + lambda;
    let cost_h = satd_h + lambda * 3;
    let cost_v = satd_v + lambda * 3;

    let (best_mode, best_cost) = if cost_dc < cost_h && cost_dc < cost_v {
        (0u8, cost_dc)
    } else if cost_h < cost_v {
        (1u8, cost_h)
    } else {
        (2u8, cost_v)
    };

    let (cb_dst, cr_dst) = pred_cb_cr.split_at_mut(64);
    write_chroma_pred(
        cb_dst.first_chunk_mut::<64>().expect("cb 64"),
        best_mode,
        &top_cb,
        &left_cb,
        &dc_top_cb,
        &dc_bot_cb,
    );
    write_chroma_pred(
        cr_dst.first_chunk_mut::<64>().expect("cr 64"),
        best_mode,
        &top_cr,
        &left_cr,
        &dc_top_cr,
        &dc_bot_cr,
    );

    (best_mode, best_cost)
}
