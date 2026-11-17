// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! The aarch64 NEON kernel set — every entry point of `super::x86_64`, from
//! upstream's arm64 assembly under `codec/common/arm64/`,
//! `codec/encoder/core/arm64/` and `codec/decoder/core/arm64/`.
//!
//! NEON is part of the AArch64 baseline, so no runtime feature test guards these
//! kernels. Each body is a `#[target_feature(enable = "neon")]` function: inside one
//! the value-only intrinsics are ordinary safe calls, loads and stores go through
//! [`lanes`], and the entry points enter `#[target_feature]` contexts safely via
//! [`crate::simd::tokens::NeonToken`] and [`crate::simd::simd_entry`].
//!
//! Every kernel is byte-exact with the scalar beside it, which in `quant.rs`,
//! `dct.rs` and `mc.rs` means widening an intermediate the asm holds at 16 bits.
//!
//! The module is compiled out under `cfg(miri)`, whose NEON shims stop short of the
//! byte-difference instructions used here; that lane takes the scalar forwards.

#![deny(unsafe_code)]

pub mod copy;
pub mod dct;
pub mod deblock;
pub mod intra_pred;
pub mod mc;
pub mod me;
pub mod quant;
pub mod sad;
pub mod satd;
pub mod score;
pub mod vaa;

pub(super) use crate::simd::tokens::neon_token;

/// Safe array/slice loads and stores shared by the NEON kernels, built entirely from
/// value-based NEON intrinsics (`vcreate_*`, `vsetq_lane_*`, `vget_lane_*`,
/// `vreinterpret*`) so no raw pointers or `unsafe` blocks are needed.
#[forbid(unsafe_code)]
#[allow(dead_code)]
mod lanes {
    use core::arch::aarch64::*;

    /// `ld1 {v.8b}` — eight bytes into eight lanes.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn ld8(r: &[u8]) -> uint8x8_t {
        let a = r.first_chunk::<8>().expect("8 bytes");
        vcreate_u8(u64::from_ne_bytes(*a))
    }

    /// `ld1 {v.16b}` — sixteen bytes into sixteen lanes.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn ld16(r: &[u8]) -> uint8x16_t {
        let a = r.first_chunk::<16>().expect("16 bytes");
        let lo = u64::from_ne_bytes(*a[..8].first_chunk::<8>().expect("lo 8 bytes"));
        let hi = u64::from_ne_bytes(*a[8..].first_chunk::<8>().expect("hi 8 bytes"));
        vreinterpretq_u8_u64(vsetq_lane_u64::<1>(
            hi,
            vsetq_lane_u64::<0>(lo, vdupq_n_u64(0)),
        ))
    }

    /// `ld1 {v.s}[0]` — four bytes into the low four lanes, the upper four zero.
    ///
    /// Zeroing the upper lanes lets every reduce here be the full-width one.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn ld4(r: &[u8]) -> uint8x8_t {
        let w = u32::from_ne_bytes(*r.first_chunk::<4>().expect("4 bytes"));
        vcreate_u8(w as u64)
    }

    /// `ld1 {v.8h}` — eight `u16` lanes.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn ld8_u16(r: &[u16]) -> uint16x8_t {
        let a = r.first_chunk::<8>().expect("8 u16s");
        let v = vdupq_n_u16(0);
        let v = vsetq_lane_u16::<0>(a[0], v);
        let v = vsetq_lane_u16::<1>(a[1], v);
        let v = vsetq_lane_u16::<2>(a[2], v);
        let v = vsetq_lane_u16::<3>(a[3], v);
        let v = vsetq_lane_u16::<4>(a[4], v);
        let v = vsetq_lane_u16::<5>(a[5], v);
        let v = vsetq_lane_u16::<6>(a[6], v);
        vsetq_lane_u16::<7>(a[7], v)
    }

    /// `ld1 {v.8h}` — eight coefficients.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn ld8_i16(r: &[i16]) -> int16x8_t {
        let a = r.first_chunk::<8>().expect("8 coefficients");
        let v = vdupq_n_s16(0);
        let v = vsetq_lane_s16::<0>(a[0], v);
        let v = vsetq_lane_s16::<1>(a[1], v);
        let v = vsetq_lane_s16::<2>(a[2], v);
        let v = vsetq_lane_s16::<3>(a[3], v);
        let v = vsetq_lane_s16::<4>(a[4], v);
        let v = vsetq_lane_s16::<5>(a[5], v);
        let v = vsetq_lane_s16::<6>(a[6], v);
        vsetq_lane_s16::<7>(a[7], v)
    }

    /// `ld1 {v.4h}` — four coefficients.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn ld4_i16(r: &[i16]) -> int16x4_t {
        let a = r.first_chunk::<4>().expect("4 coefficients");
        let v = vdup_n_s16(0);
        let v = vset_lane_s16::<0>(a[0], v);
        let v = vset_lane_s16::<1>(a[1], v);
        let v = vset_lane_s16::<2>(a[2], v);
        vset_lane_s16::<3>(a[3], v)
    }

    /// The low four lanes as an array.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn low4(v: uint8x8_t) -> [u8; 4] {
        vget_lane_u32::<0>(vreinterpret_u32_u8(v)).to_ne_bytes()
    }

    /// The eight lanes as an array.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn to8(v: uint8x8_t) -> [u8; 8] {
        vget_lane_u64::<0>(vreinterpret_u64_u8(v)).to_ne_bytes()
    }

    /// The sixteen lanes as an array.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn to16(v: uint8x16_t) -> [u8; 16] {
        let mut out = [0u8; 16];
        st16(&mut out, v);
        out
    }

    /// `st1 {v.16b}`.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn st16(out: &mut [u8], v: uint8x16_t) {
        let a = out.first_chunk_mut::<16>().expect("16 bytes");
        let u = vreinterpretq_u64_u8(v);
        *a[..8].first_chunk_mut::<8>().expect("lo 8 bytes") = vgetq_lane_u64::<0>(u).to_ne_bytes();
        *a[8..].first_chunk_mut::<8>().expect("hi 8 bytes") = vgetq_lane_u64::<1>(u).to_ne_bytes();
    }

    /// `st1 {v.8b}`.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn st8(out: &mut [u8], v: uint8x8_t) {
        *out.first_chunk_mut::<8>().expect("8 bytes") = to8(v);
    }

    /// `st1 {v.s}[0]` — the low four lanes.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn st4(out: &mut [u8], v: uint8x8_t) {
        *out.first_chunk_mut::<4>().expect("4 bytes") = low4(v);
    }

    /// `st1 {v.8h}` — eight `u16` lanes.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn st8_u16(out: &mut [u16], v: uint16x8_t) {
        *out.first_chunk_mut::<8>().expect("8 u16s") = [
            vgetq_lane_u16::<0>(v),
            vgetq_lane_u16::<1>(v),
            vgetq_lane_u16::<2>(v),
            vgetq_lane_u16::<3>(v),
            vgetq_lane_u16::<4>(v),
            vgetq_lane_u16::<5>(v),
            vgetq_lane_u16::<6>(v),
            vgetq_lane_u16::<7>(v),
        ];
    }

    /// `st1 {v.8h}`.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn st8_i16(out: &mut [i16], v: int16x8_t) {
        *out.first_chunk_mut::<8>().expect("8 coefficients") = [
            vgetq_lane_s16::<0>(v),
            vgetq_lane_s16::<1>(v),
            vgetq_lane_s16::<2>(v),
            vgetq_lane_s16::<3>(v),
            vgetq_lane_s16::<4>(v),
            vgetq_lane_s16::<5>(v),
            vgetq_lane_s16::<6>(v),
            vgetq_lane_s16::<7>(v),
        ];
    }

    /// `st1 {v.4h}`.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn st4_i16(out: &mut [i16], v: int16x4_t) {
        *out.first_chunk_mut::<4>().expect("4 coefficients") = [
            vget_lane_s16::<0>(v),
            vget_lane_s16::<1>(v),
            vget_lane_s16::<2>(v),
            vget_lane_s16::<3>(v),
        ];
    }

    /// `st1 {v.4s}`.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn st4_i32(out: &mut [i32; 4], v: int32x4_t) {
        *out = [
            vgetq_lane_s32::<0>(v),
            vgetq_lane_s32::<1>(v),
            vgetq_lane_s32::<2>(v),
            vgetq_lane_s32::<3>(v),
        ];
    }

    /// True when any lane of a byte mask is set — the asm's `ZERO_JUMP_END`.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn any_set(m: uint8x16_t) -> bool {
        vmaxvq_u8(m) != 0
    }
}
