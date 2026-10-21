// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! Deterministic procedural I420 frame generators for encoder tests.
//!
//! All generators use pure integer arithmetic and stateless/seeded bit-mixing so
//! they produce bit-identical YUV frames across platforms and match the C++
//! reference harness byte-for-byte.

/// Stateless 32-bit integer hash of `(x, y, frame)` coordinates.
#[inline]
pub fn pixel_hash(x: u32, y: u32, frame: u32) -> u32 {
    let mut h = x
        .wrapping_mul(374_761_393)
        .wrapping_add(y.wrapping_mul(668_265_263))
        .wrapping_add(frame.wrapping_mul(2_147_483_647));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^ (h >> 16)
}

/// Deterministic `XorShift32` PRNG step for tests that need a sequential byte stream
/// (e.g. replacing C `rand() % 256` in ported upstream API tests).
#[inline]
pub fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    if x == 0 {
        x = 0x9E37_79B9;
    }
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// Fills an I420 buffer (`w * h * 3 / 2` bytes) with a composite camera-like scene:
/// - **Static border + smooth gradient background** with subtle spatial/temporal dither
///   (exercises Intra 16x16, Background Detection, and Bilateral Denoise).
/// - **Moving Box 1** translating by `(+3, +2)` luma pixels/frame with sharp diagonal stripes
///   (exercises Diamond/Cross Motion Estimation, sub-pel interpolation, Intra 4x4, and
///   `16x8`/`8x16`/`8x8` inter partitions).
/// - **Counter-moving Box 2** translating by `(-4, +5)` luma pixels/frame with high-entropy
///   seeded texture (exercises CABAC context adaptation, Adaptive Quantization, and
///   `SM_SIZELIMITED_SLICE` slice splitting).
/// - **Periodic scene cut** every 6 frames (`(frame_idx / 6) & 1`, exercising Scene Change
///   Detection).
pub fn fill_camera_frame(buf: &mut [u8], w: usize, h: usize, frame_idx: usize) {
    let luma_size = w * h;
    let cw = w / 2;
    let ch = h / 2;
    let chroma_size = cw * ch;
    assert!(buf.len() >= luma_size + 2 * chroma_size);

    let (y_plane, rest) = buf.split_at_mut(luma_size);
    let (u_plane, rest) = rest.split_at_mut(chroma_size);
    let v_plane = &mut rest[..chroma_size];

    let scene = (frame_idx / 6) & 1;
    let box_w = (w / 4).max(16);
    let box_h = (h / 4).max(16);
    let span_x = w.saturating_sub(box_w).max(1);
    let span_y = h.saturating_sub(box_h).max(1);

    let box1_x = (w / 8 + frame_idx * 3) % span_x;
    let box1_y = (h / 6 + frame_idx * 2) % span_y;
    let box2_x = (span_x - (frame_idx * 4) % span_x) % span_x;
    let box2_y = (h / 2 + frame_idx * 5) % span_y;

    let border_x = w / 6;
    let border_y = h / 6;

    for y in 0..h {
        for x in 0..w {
            let val = if x >= box1_x && x < box1_x + box_w && y >= box1_y && y < box1_y + box_h {
                let rx = x - box1_x;
                let ry = y - box1_y;
                let stripe = if ((rx + ry * 2) & 8) != 0 {
                    180u32
                } else {
                    40u32
                };
                let edge = (((rx * 11) ^ (ry * 7)) & 31) as u32;
                (stripe + edge) as u8
            } else if x >= box2_x && x < box2_x + box_w && y >= box2_y && y < box2_y + box_h {
                let rx = (x - box2_x) as u32;
                let ry = (y - box2_y) as u32;
                let hval = pixel_hash(rx, ry, (frame_idx & 1) as u32);
                (20 + (hval % 200)) as u8
            } else {
                let grad = ((x * 120) / w.max(1) + (y * 60) / h.max(1)) as u32;
                let bg = if scene == 0 { 24 + grad } else { 216 - grad };
                let is_border =
                    x < border_x || x >= w - border_x || y < border_y || y >= h - border_y;
                let dither = if is_border {
                    ((x * 7 + y * 13) % 7) as u32
                } else {
                    ((x * 7 + y * 13 + frame_idx * 5) % 9) as u32
                };
                (bg + dither) as u8
            };
            y_plane[y * w + x] = val;
        }
    }

    for cy in 0..ch {
        for cx in 0..cw {
            let lx = cx * 2;
            let ly = cy * 2;
            let (u_val, v_val) =
                if lx >= box1_x && lx < box1_x + box_w && ly >= box1_y && ly < box1_y + box_h {
                    (180u8, 76u8)
                } else if lx >= box2_x && lx < box2_x + box_w && ly >= box2_y && ly < box2_y + box_h
                {
                    let u = (90 + (pixel_hash(cx as u32, cy as u32, 0) % 70)) as u8;
                    let v = (160 - (pixel_hash(cx as u32, cy as u32, 1) % 70)) as u8;
                    (u, v)
                } else {
                    let c_grad = ((cx * 60) / cw.max(1) + (cy * 40) / ch.max(1)) as u8;
                    if scene == 0 {
                        (80 + c_grad, 160 - c_grad)
                    } else {
                        (176 - c_grad, 80 + c_grad)
                    }
                };
            u_plane[cy * cw + cx] = u_val;
            v_plane[cy * cw + cx] = v_val;
        }
    }
}

/// Allocates and returns an I420 frame generated by [`fill_camera_frame`].
pub fn generate_camera_frame(w: usize, h: usize, frame_idx: usize) -> Vec<u8> {
    let mut buf = vec![0u8; w * h * 3 / 2];
    fill_camera_frame(&mut buf, w, h, frame_idx);
    buf
}

/// Fills an I420 buffer (`w * h * 3 / 2` bytes) with a synthetic scrolling UI/text scene:
/// - **Window title bars and high-contrast glyph cells** scrolling vertically by 8 luma rows
///   per frame (exercises `SCREEN_CONTENT_REAL_TIME` scroll detection and feature search).
/// - **Moving 16x16 cursor/tooltip box** translating across the screen.
pub fn fill_screen_frame(buf: &mut [u8], w: usize, h: usize, frame_idx: usize) {
    let luma_size = w * h;
    let cw = w / 2;
    let ch = h / 2;
    let chroma_size = cw * ch;
    assert!(buf.len() >= luma_size + 2 * chroma_size);

    let (y_plane, rest) = buf.split_at_mut(luma_size);
    let (u_plane, rest) = rest.split_at_mut(chroma_size);
    let v_plane = &mut rest[..chroma_size];

    let span_x = w.saturating_sub(16).max(1);
    let span_y = h.saturating_sub(16).max(1);
    let cur_x = (frame_idx * 6) % span_x;
    let cur_y = (frame_idx * 4) % span_y;

    for y in 0..h {
        let sy = (y + frame_idx * 8) % h;
        for x in 0..w {
            let val = if x >= cur_x && x < cur_x + 16 && y >= cur_y && y < cur_y + 16 {
                (128 ^ (((x ^ y) & 4) * 16)) as u8
            } else if (sy % 32) < 6 {
                48
            } else {
                let in_text_row = (sy % 8) >= 2 && (sy % 8) < 6;
                let in_margin = x >= 16 && x + 16 < w;
                if in_text_row && in_margin && (x / 8) % 7 != 6 {
                    let cell = pixel_hash((x / 2) as u32, (sy / 2) as u32, 42);
                    if (cell & 3) == 0 { 24 } else { 232 }
                } else {
                    232
                }
            };
            y_plane[y * w + x] = val;
        }
    }

    for cy in 0..ch {
        let scy = (cy + frame_idx * 4) % ch;
        let is_bar = ((scy * 2) % 32) < 6;
        for cx in 0..cw {
            u_plane[cy * cw + cx] = if is_bar { 170 } else { 128 };
            v_plane[cy * cw + cx] = if is_bar { 90 } else { 128 };
        }
    }
}

/// Allocates and returns an I420 frame generated by [`fill_screen_frame`].
pub fn generate_screen_frame(w: usize, h: usize, frame_idx: usize) -> Vec<u8> {
    let mut buf = vec![0u8; w * h * 3 / 2];
    fill_screen_frame(&mut buf, w, h, frame_idx);
    buf
}

/// Fills `buf` with deterministic pseudo-random bytes using [`xorshift32`].
pub fn fill_seeded_noise(buf: &mut [u8], state: &mut u32) {
    for b in buf.iter_mut() {
        *b = (xorshift32(state) & 0xff) as u8;
    }
}
