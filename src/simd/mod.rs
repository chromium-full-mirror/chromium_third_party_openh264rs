// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

//! SIMD acceleration kernels and CPU feature detection for openh264rs.
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

/// The NEON kernels. Compiled out under Miri, which cannot interpret them; that build
/// uses the scalar forwards.
#[cfg(all(target_arch = "aarch64", not(miri)))]
pub mod aarch64;

/// The scalar forwards, compiled only where they are what [`kernels`] names.
#[cfg(any(
    feature = "scalar",
    not(any(target_arch = "x86_64", all(target_arch = "aarch64", not(miri))))
))]
pub mod scalar;

#[cfg(all(target_arch = "aarch64", not(miri), not(feature = "scalar")))]
pub use aarch64 as kernels;
#[cfg(any(
    feature = "scalar",
    not(any(target_arch = "x86_64", all(target_arch = "aarch64", not(miri))))
))]
pub use scalar as kernels;
/// The kernel set, and the whole of the dispatch. Every direct call site and every
/// `WELS_CPU_SSE2` table install names its kernel `kernels::<family>::<kernel>`, and
/// this alias decides what that resolves to. Dispatch files import it once at module
/// level; a kernel shares its name with the scalar body it replaces, and the module
/// qualifier is what tells them apart.
///
/// Selection is total and build-time: every build lands on exactly one arm, with no
/// runtime test in front of it.
///
/// | build | resolves to | what runs |
/// |---|---|---|
/// | x86_64, default | [`x86_64`] | `core::arch` SSE2, SSSE3, and AVX2 intrinsics |
/// | aarch64, default | [`aarch64`] | `core::arch` NEON intrinsics |
/// | `--features scalar` | [`scalar`] | forwards to the scalar body |
/// | no kernels for this target, or Miri on aarch64 | [`scalar`] | likewise |
///
/// Dispatch sites carry no `#[cfg]` and no branch.
///
/// The three modules export the same entry points with the same signatures, so a
/// dispatch site does not change when the selection does. The intrinsic set for the
/// host is compiled whenever it can be, `--features scalar` included.
#[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
pub use x86_64 as kernels;

use crate::common::cpu_core::*;

/// Detects available CPU SIMD features, once per process. The word says which vector
/// kernels this build has; `--features scalar` makes it `0`.
///
/// Latching keeps this to one probe per process; the initialiser is out of line so the
/// steady-state read is an acquire load and a compare.
#[inline]
pub fn detect_cpu_features() -> u32 {
    // `Acquire` here pairs with the `Release` in `latch_cpu_features`, so a thread that
    // sees the ready flag also sees the word stored before it was set.
    if CPU_FEATURES_READY.load(Ordering::Acquire) {
        return CPU_FEATURES.load(Ordering::Relaxed);
    }
    latch_cpu_features()
}

/// Runs once per process.
#[cold]
#[inline(never)]
fn latch_cpu_features() -> u32 {
    let flags = arch_cpu_features();
    #[cfg(target_arch = "x86_64")]
    tokens::warm_x86_feats();
    // Racing callers compute the same word from the same inputs, so both stores are
    // idempotent and neither needs a compare-exchange. The word goes first and the flag
    // second, under `Release`, so no reader can see the flag without the word.
    CPU_FEATURES.store(flags, Ordering::Relaxed);
    CPU_FEATURES_READY.store(true, Ordering::Release);
    flags
}

/// The x86_64 feature probe. MMX, SSE and SSE2 are part of the baseline x86_64
/// instruction set, and SSSE3 is required and verified once here.
#[cfg(target_arch = "x86_64")]
fn arch_cpu_features() -> u32 {
    // Under `--features scalar` there is no vector kernel for any slot, so no bit is
    // reported and the `pfXxx` tables install their scalar arms directly.
    if cfg!(feature = "scalar") {
        return 0;
    }

    assert!(
        std::is_x86_feature_detected!("ssse3"),
        "x86_64 SIMD kernels require SSSE3 support"
    );

    let mut flags = WELS_CPU_MMX
        | WELS_CPU_MMXEXT
        | WELS_CPU_SSE
        | WELS_CPU_SSE2
        | WELS_CPU_SSE3
        | WELS_CPU_SSSE3;

    if std::is_x86_feature_detected!("sse4.1") {
        flags |= WELS_CPU_SSE41;
    }
    if std::is_x86_feature_detected!("sse4.2") {
        flags |= WELS_CPU_SSE42;
    }
    if std::is_x86_feature_detected!("avx") {
        flags |= WELS_CPU_AVX;
    }
    if std::is_x86_feature_detected!("avx2") {
        flags |= WELS_CPU_AVX2;
    }
    if std::is_x86_feature_detected!("fma") {
        flags |= WELS_CPU_FMA;
    }

    flags
}

/// The aarch64 probe — `codec/common/src/cpu.cpp`'s `WelsCPUFeatureDetect` for
/// `HAVE_NEON_AARCH64`. No runtime detection: NEON is mandatory on every AArch64 CPU.
///
/// `WELS_CPU_SSE2` names a slot here, not an instruction set: the bit means "there is
/// a vector kernel for this slot". The `pfXxx` tables test that one bit on every
/// target, so the aarch64 kernel set reports it. `WELS_CPU_NEON` is set alongside;
/// nothing dispatches on it.
///
/// `WELS_CPU_AVX2` stays clear: no `_avx2` entry point is a different kernel on this
/// target, and there is no wider register file to pick one for.
///
/// The word is `0` under `--features scalar`, and under Miri, where the NEON module is
/// compiled out and `kernels` is the scalar set.
#[cfg(target_arch = "aarch64")]
fn arch_cpu_features() -> u32 {
    if cfg!(any(feature = "scalar", miri)) {
        return 0;
    }
    WELS_CPU_SSE2 | WELS_CPU_NEON
}

/// Off x86_64 and aarch64 there are no vector kernels: every bit stays clear and every
/// dispatch site takes its scalar fallback.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn arch_cpu_features() -> u32 {
    0
}

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// The process-wide feature word, and whether it has been computed yet.
///
/// Two cells rather than a sentinel bit inside the word: every bit of the `u32` is a
/// real flag (`WELS_CPU_CACHELINE_128` is `0x8000_0000`), and `0` is a legitimate
/// answer.
static CPU_FEATURES: AtomicU32 = AtomicU32::new(0);
static CPU_FEATURES_READY: AtomicBool = AtomicBool::new(false);

/// Returns true if this build has the AVX2 kernels and the CPU can run them.
///
/// Unlike SSE2 this is not x86_64 baseline, so the runtime test is required: the AVX2
/// SAD kernels execute `vpsadbw` and fault on any pre-Haswell Intel or pre-Excavator
/// AMD part.
///
/// The `cfg!` only folds the branch away for a build that already guarantees AVX2, and
/// is false by default on every `x86_64-*` target; it cannot replace the runtime test,
/// because `-C target-feature=+avx2` applies to the whole crate. Per-function AVX2
/// codegen is `#[target_feature(enable = "avx2")]`, which `sad_16x_avx2` carries.
#[inline(always)]
pub fn has_avx2() -> bool {
    !cfg!(feature = "scalar")
        && (cfg!(target_feature = "avx2") || (detect_cpu_features() & WELS_CPU_AVX2) != 0)
}

#[allow(dead_code)]
pub mod tokens {
    #[cfg(target_arch = "x86_64")]
    use std::sync::atomic::{AtomicU8, Ordering};

    #[cfg(target_arch = "x86_64")]
    const FEAT_SSSE3: u8 = 1 << 0;
    #[cfg(target_arch = "x86_64")]
    const FEAT_AVX2: u8 = 1 << 1;
    #[cfg(target_arch = "x86_64")]
    const FEAT_READY: u8 = 1 << 7;

    /// Single-byte hardware feature cache: `0` before first probe, `FEAT_READY | bits`
    /// afterward so a present feature is verified by a single `testb` + `jne` (separate
    /// from `CPU_FEATURES`, which zeros all flags under `--features scalar` and uses a
    /// second atomic ready flag).
    #[cfg(target_arch = "x86_64")]
    static X86_FEATS: AtomicU8 = AtomicU8::new(0);

    #[cfg(target_arch = "x86_64")]
    #[cold]
    #[inline(never)]
    fn latch_x86_feats() -> u8 {
        let mut f = FEAT_READY;
        if std::arch::is_x86_feature_detected!("ssse3") {
            f |= FEAT_SSSE3;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            f |= FEAT_AVX2;
        }
        X86_FEATS.store(f, Ordering::Relaxed);
        f
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub(super) fn warm_x86_feats() {
        let _ = latch_x86_feats();
    }

    #[cfg(target_arch = "x86_64")]
    #[cold]
    #[inline(never)]
    fn has_x86_feat_slow(f: u8, bit: u8) -> bool {
        if (f & FEAT_READY) != 0 {
            return false;
        }
        (latch_x86_feats() & bit) != 0
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn has_x86_feat(bit: u8) -> bool {
        let f = X86_FEATS.load(Ordering::Relaxed);
        if (f & bit) != 0 {
            return true;
        }
        has_x86_feat_slow(f, bit)
    }

    #[derive(Copy, Clone)]
    pub struct NeonToken(());
    impl NeonToken {
        #[inline(always)]
        pub fn try_new() -> Option<Self> {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                Some(Self(()))
            }
            #[cfg(all(target_arch = "aarch64", not(target_feature = "neon")))]
            {
                std::arch::is_aarch64_feature_detected!("neon").then_some(Self(()))
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                None
            }
        }
    }

    #[derive(Copy, Clone)]
    pub struct Sse2Token(());
    impl Sse2Token {
        #[inline(always)]
        pub fn try_new() -> Option<Self> {
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
            {
                Some(Self(()))
            }
            #[cfg(all(target_arch = "x86_64", not(target_feature = "sse2")))]
            {
                std::arch::is_x86_feature_detected!("sse2").then_some(Self(()))
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                None
            }
        }
    }

    macro_rules! def_x86_token {
        ($name:ident, $feat:tt, $bit:ident) => {
            #[derive(Copy, Clone)]
            pub struct $name(());
            impl $name {
                /// Single-load fast check against the latched feature byte (zero calls).
                /// Returns `None` if unlatched or if `$feat` is absent.
                #[inline(always)]
                pub fn try_fast() -> Option<Self> {
                    #[cfg(all(target_arch = "x86_64", target_feature = $feat))]
                    {
                        Some(Self(()))
                    }
                    #[cfg(all(target_arch = "x86_64", not(target_feature = $feat)))]
                    {
                        ((X86_FEATS.load(Ordering::Relaxed) & $bit) != 0).then_some(Self(()))
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        None
                    }
                }

                /// Fast dispatch check: returns `None` under `--features scalar`, if
                /// unlatched, or if `$feat` is absent (zero calls).
                #[inline(always)]
                pub fn try_fast_enabled() -> Option<Self> {
                    if cfg!(feature = "scalar") {
                        None
                    } else {
                        Self::try_fast()
                    }
                }

                /// Hardware capability check (ignores `--features scalar` so unit tests
                /// can still verify SIMD kernel parity under `--features scalar`).
                #[inline(always)]
                pub fn try_new() -> Option<Self> {
                    #[cfg(all(target_arch = "x86_64", target_feature = $feat))]
                    {
                        Some(Self(()))
                    }
                    #[cfg(all(target_arch = "x86_64", not(target_feature = $feat)))]
                    {
                        has_x86_feat($bit).then_some(Self(()))
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        None
                    }
                }

                /// Dispatch capability check: returns `None` when `--features scalar`
                /// is enabled, even on CPUs (or `-C target-feature` builds) with `$feat`.
                #[inline(always)]
                pub fn try_new_enabled() -> Option<Self> {
                    if cfg!(feature = "scalar") {
                        None
                    } else {
                        Self::try_new()
                    }
                }
            }
        };
    }

    def_x86_token!(Ssse3Token, "ssse3", FEAT_SSSE3);
    def_x86_token!(Avx2Token, "avx2", FEAT_AVX2);

    #[cfg(target_arch = "x86_64")]
    impl Avx2Token {
        /// Reifies a stateless [`super::Avx2RecSadKernel`] into a plain function pointer,
        /// amortizing the capability check across `SWelsSampleDealingFunc::pfSampleSad`.
        ///
        /// Sound because `self` witnesses that AVX2 is supported on the host CPU for the
        /// lifetime of the process.
        #[inline]
        pub(crate) fn into_rec_sad_fn<K: super::Avx2RecSadKernel>(
            self,
        ) -> fn(
            &crate::encoder::rec_view::RecCursor<'_>,
            &crate::encoder::rec_view::RecCursor<'_>,
        ) -> i32 {
            fn call<K: super::Avx2RecSadKernel>(
                a: &crate::encoder::rec_view::RecCursor<'_>,
                b: &crate::encoder::rec_view::RecCursor<'_>,
            ) -> i32 {
                K::run(a, b, Avx2Token(()))
            }
            let _ = self;
            call::<K>
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub fn neon_token() -> NeonToken {
        NeonToken::try_new().expect("NEON is baseline on aarch64")
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub fn sse2_token() -> Sse2Token {
        Sse2Token::try_new().expect("SSE2 is baseline on x86_64")
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub fn ssse3_token() -> Ssse3Token {
        #[cold]
        #[inline(never)]
        fn slow() -> Ssse3Token {
            Ssse3Token::try_new().expect("SSSE3 required on x86_64")
        }
        match Ssse3Token::try_fast() {
            Some(t) => t,
            None => slow(),
        }
    }
}

/// Wraps a `#[target_feature]` function whose **last** parameter is the matching
/// capability token (`NeonToken`, `Sse2Token`, `Ssse3Token`, or `Avx2Token`) into a
/// safe outer function + a `#[target_feature]` inner function where SIMD intrinsics are
/// safe. Compile-time type checking rejects missing or mismatched capability tokens.
#[cfg(any(target_arch = "x86_64", all(target_arch = "aarch64", not(miri))))]
macro_rules! simd_entry {
    (@id $kw:ident $id:ident) => { $id };
    (@id $id:ident) => { $id };
    (@last $p1:ident $($p2:ident)?) => { simd_entry!(@id $p1 $($p2)?) };
    (@last $p1:ident $($p2:ident)?, $($rest:tt)+) => { simd_entry!(@last $($rest)+) };
    (@token "neon") => { $crate::simd::tokens::NeonToken };
    (@token "sse2") => { $crate::simd::tokens::Sse2Token };
    (@token "ssse3") => { $crate::simd::tokens::Ssse3Token };
    (@token "avx2") => { $crate::simd::tokens::Avx2Token };
    ($(
        $(#[doc = $doc:expr])*
        $(#[allow($allow:meta)])*
        #[target_feature(enable = $feat:tt)]
        $vis:vis fn $name:ident $(< $( $g1:ident $($g2:ident)? $(: $gb:ident $(+ $gb2:ident)*)? ),+ $(,)? >)?
        ( $($p1:ident $($p2:ident)? : $pty:ty),+ $(,)? ) $(-> $ret:ty)? $body:block
    )+) => {$(
        $(#[doc = $doc])*
        #[inline(always)]
        #[allow(clippy::too_many_arguments $(, $allow)*)]
        $vis fn $name $(< $( $g1 $($g2)? $(: $gb $(+ $gb2)*)? ),+ >)? ( $(simd_entry!(@id $p1 $($p2)?) : $pty),+ ) $(-> $ret)? {
            let _: simd_entry!(@token $feat) = simd_entry!(@last $($p1 $($p2)?),+);
            #[target_feature(enable = $feat)]
            #[inline]
            #[allow(clippy::too_many_arguments $(, $allow)*)]
            fn inner $(< $( $g1 $($g2)? $(: $gb $(+ $gb2)*)? ),+ >)? ( $($p1 $($p2)? : $pty),+ ) $(-> $ret)? $body
            #[allow(unsafe_code)]
            unsafe { inner $(::<$(simd_entry!(@id $g1 $($g2)?)),+>)? ( $(simd_entry!(@id $p1 $($p2)?)),+ ) }
        }
    )+};
}
#[cfg(any(target_arch = "x86_64", all(target_arch = "aarch64", not(miri))))]
pub(crate) use simd_entry;

/// Stateless two-cursor AVX2 SAD kernel that can be reified into a higher-ranked
/// `fn(&RecCursor<'_>, &RecCursor<'_>) -> i32` pointer once an
/// [`Avx2Token`](tokens::Avx2Token) has been witnessed at table-init time.
#[cfg(target_arch = "x86_64")]
pub(crate) trait Avx2RecSadKernel {
    fn run(
        a: &crate::encoder::rec_view::RecCursor<'_>,
        b: &crate::encoder::rec_view::RecCursor<'_>,
        token: tokens::Avx2Token,
    ) -> i32;
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::{Path, PathBuf};

    fn src_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
    }

    fn rs_files(dir: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).expect("read_dir").flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push(p);
                }
            }
        }
        out.sort();
        out
    }

    /// The text before the file's test module, so a helper inside one is never mistaken
    /// for an entry point and a name used only by a test never counts as "reached".
    ///
    /// The cut is at `#[cfg(test)]\nmod `, not at `#[cfg(test)]` alone: that attribute
    /// also sits on test-only `use` items among a file's imports.
    fn without_tests(src: &str) -> &str {
        src.split("#[cfg(test)]\nmod ").next().unwrap_or(src)
    }

    /// The `pub`/`pub(crate)` fn names declared in one kernel module.
    fn entry_points(module: &str) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        for f in rs_files(&src_root().join("simd").join(module)) {
            let text = std::fs::read_to_string(&f).expect("read kernel file");
            let file = f.file_name().unwrap().to_string_lossy().into_owned();
            let mut rest = without_tests(&text);
            while let Some(i) = rest.find("pub ").or_else(|| rest.find("pub(crate) ")) {
                let tail = &rest[i..];
                let tail = tail
                    .strip_prefix("pub(crate) ")
                    .or_else(|| tail.strip_prefix("pub "))
                    .unwrap();
                let tail = tail.strip_prefix("unsafe ").unwrap_or(tail);
                if let Some(tail) = tail.strip_prefix("fn ") {
                    let name: String = tail
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    if !name.is_empty() {
                        out.insert(name, file.clone());
                    }
                }
                rest = &rest[i + 4..];
            }
        }
        out
    }

    /// Kernels internal to a module that are nonetheless `pub` (currently empty — all
    /// module-internal helpers are private `fn`). Maintained by hand; it fails closed —
    /// an unlisted `pub` kernel that nothing dispatches makes the test fail.
    const INTERNAL: &[&str] = &[];

    /// Every kernel entry point is either named somewhere outside `src/simd/` or listed
    /// in [`INTERNAL`]. A slot left holding the scalar passes every parity and
    /// conformance test, so an unreached kernel is invisible to everything else.
    ///
    /// The check matches source text rather than function pointers, so it does not
    /// claim that the site executes (a reference inside a disabled `#[cfg]` still
    /// counts) or that a slot holds the right kernel.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn every_kernel_is_named_by_a_dispatch_site() {
        // Reached means "named through the alias", not "this token appears": a kernel
        // shares its name with the scalar body it replaces, so a bare-token search
        // would report success for a slot that was never wired.
        //
        // The exception is the sites that glob-import a kernel module and then call its
        // entries bare; for those, every token in the file counts. Both are
        // `intra_pred`, whose `enc_`/`dec_` names no scalar shares.
        let mut used: BTreeSet<String> = BTreeSet::new();
        for f in rs_files(&src_root()) {
            if f.components().any(|c| c.as_os_str() == "simd") {
                continue;
            }
            let text = std::fs::read_to_string(&f).expect("read codec file");
            let text = without_tests(&text);

            let mut rest = text;
            while let Some(i) = rest.find("kernels::") {
                let tail = &rest[i + "kernels::".len()..];
                // `<module>::<kernel>` — take the second segment.
                let seg: String = tail
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                let after = &tail[seg.len()..];
                if let Some(after) = after.strip_prefix("::") {
                    let name: String = after
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    if !name.is_empty() {
                        used.insert(name);
                    }
                }
                rest = &rest[i + "kernels::".len()..];
            }

            if text.contains("use kernels::") && text.contains("::*;") {
                used.extend(
                    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
                        .filter(|t| !t.is_empty())
                        .map(str::to_owned),
                );
            }
        }

        let mut unreached = Vec::new();
        for module in ["x86_64", "aarch64"] {
            for (name, file) in entry_points(module) {
                if INTERNAL.contains(&name.as_str()) || used.contains(&name) {
                    continue;
                }
                unreached.push(format!("{module}/{file}: {name}"));
            }
        }
        assert!(
            unreached.is_empty(),
            "these kernels exist but no dispatch site names them — wire them, or add them \
             to `INTERNAL` if they are module-internal: {unreached:#?}"
        );
    }

    /// The three kernel sets have to agree on their entry points, or `simd::kernels`
    /// would resolve differently per build and a dispatch site would compile on one
    /// target and not another.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn the_kernel_sets_expose_the_same_entry_points() {
        let sets: Vec<(&str, BTreeSet<String>)> = ["x86_64", "aarch64", "scalar"]
            .iter()
            .map(|m| {
                let names = entry_points(m)
                    .into_keys()
                    .filter(|n| !INTERNAL.contains(&n.as_str()))
                    .collect();
                (*m, names)
            })
            .collect();
        for (name, set) in &sets[1..] {
            let (base_name, base) = &sets[0];
            let missing: Vec<_> = base.difference(set).collect();
            let extra: Vec<_> = set.difference(base).collect();
            assert!(
                missing.is_empty() && extra.is_empty(),
                "`simd::{name}` does not match `simd::{base_name}` — missing {missing:?}, extra {extra:?}"
            );
        }
    }
}
