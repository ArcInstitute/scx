//! Opt-in per-stage GPU timing profiler (ACC-RUST-OPT-V4 Phase 0.2 evidence).
//!
//! Enabled by setting `SCX_GPU_PROFILE=1` (any non-empty, non-`"0"` value).
//! Quantifies where the SCX GPU path spends wall-clock time so we can tell
//! whether the measured in-VRAM gap vs rapids-singlecell is dominated by
//! **decode / host-to-device upload** (→ justifies the Phase 4 decode-metadata
//! sidecar) or by **compute** (→ does not).
//!
//! Four buckets; the decode/upload buckets are split by codec class because
//! Scx1 decodes GPU-side (Rice/FOR-BP) while None/Zstd/Lz4Shuffle/Pcodec decode
//! on the host and then upload:
//! - `host_decode` — CPU-side codec decode (Delta-Golomb indptr, scipy fallback)
//! - `htod`        — host→device uploads (`htod_copy`)
//! - `gpu_decode`  — GPU-side Scx1 decode (FOR-BP indices — scalar + BitPacker4x
//!   rows, Task 4.4b — plus Rice values and the two u32 casts), all on-device
//! - `compute`     — cuSPARSE SpMM execution (and other instrumented kernels)
//!
//! When disabled (the default), every hook short-circuits on a single relaxed
//! atomic load via [`profile_enabled`]: [`start`] returns `None`, so no
//! `Instant` is taken and no stream sync is issued — zero measurable hot-path
//! cost.
//!
//! Compute timing requires a stream synchronize (cuSPARSE SpMM is asynchronous),
//! which the caller performs only when [`profile_enabled`] is true; see the
//! `spmm_impl_view` instrumentation in [`crate::cusparse`].

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// Codec class for the decode/upload per-codec split.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CodecClass {
    /// Scx1 — Rice values + FOR-BP indices, decoded GPU-side.
    Scx1,
    /// None / Zstd / Lz4Shuffle / Pcodec — decoded on the host, then uploaded.
    Generic,
}

impl CodecClass {
    #[inline]
    fn idx(self) -> usize {
        match self {
            CodecClass::Scx1 => 0,
            CodecClass::Generic => 1,
        }
    }
}

struct Counters {
    host_decode_ns: [AtomicU64; 2],
    host_decode_shards: [AtomicU64; 2],
    htod_ns: [AtomicU64; 2],
    htod_bytes: [AtomicU64; 2],
    htod_calls: [AtomicU64; 2],
    gpu_decode_ns: AtomicU64,
    gpu_decode_shards: AtomicU64,
    compute_ns: AtomicU64,
    compute_calls: AtomicU64,
}

static COUNTERS: Counters = Counters {
    host_decode_ns: [AtomicU64::new(0), AtomicU64::new(0)],
    host_decode_shards: [AtomicU64::new(0), AtomicU64::new(0)],
    htod_ns: [AtomicU64::new(0), AtomicU64::new(0)],
    htod_bytes: [AtomicU64::new(0), AtomicU64::new(0)],
    htod_calls: [AtomicU64::new(0), AtomicU64::new(0)],
    gpu_decode_ns: AtomicU64::new(0),
    gpu_decode_shards: AtomicU64::new(0),
    compute_ns: AtomicU64::new(0),
    compute_calls: AtomicU64::new(0),
};

/// Whether profiling is enabled. Resolved once from `SCX_GPU_PROFILE` and
/// cached; the cost when disabled is a single relaxed atomic load.
#[inline]
pub fn profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SCX_GPU_PROFILE")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    })
}

/// Start a timed region. Returns `None` (and takes no timestamp) when profiling
/// is disabled, so the hot path pays nothing.
#[inline]
pub fn start() -> Option<Instant> {
    if profile_enabled() {
        Some(Instant::now())
    } else {
        None
    }
}

/// Record a host-side codec-decode region. No-op if `start` was `None`.
#[inline]
pub fn record_host_decode_since(class: CodecClass, start: Option<Instant>) {
    if let Some(t) = start {
        let i = class.idx();
        COUNTERS.host_decode_ns[i].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        COUNTERS.host_decode_shards[i].fetch_add(1, Ordering::Relaxed);
    }
}

/// Record a host→device upload region of `bytes` bytes. No-op if `start` was `None`.
#[inline]
pub fn record_htod_since(class: CodecClass, start: Option<Instant>, bytes: usize) {
    if let Some(t) = start {
        let i = class.idx();
        COUNTERS.htod_ns[i].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        COUNTERS.htod_bytes[i].fetch_add(bytes as u64, Ordering::Relaxed);
        COUNTERS.htod_calls[i].fetch_add(1, Ordering::Relaxed);
    }
}

/// Record a GPU-side decode region (Scx1 Rice + casts, plus the FOR-BP indices
/// kernel only when it ran on-GPU). No-op if `start` was `None`. The caller must
/// have synchronized the stream before calling so the elapsed time reflects
/// completed device work, and must exclude any FOR-BP SIMD host fallback (already
/// recorded via [`record_host_decode_since`]/[`record_htod_since`]) so this bucket
/// stays disjoint from the host buckets.
#[inline]
pub fn record_gpu_decode_since(start: Option<Instant>) {
    if let Some(t) = start {
        COUNTERS
            .gpu_decode_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        COUNTERS.gpu_decode_shards.fetch_add(1, Ordering::Relaxed);
    }
}

/// Record a GPU compute region (e.g. one cuSPARSE SpMM). No-op if `start` was
/// `None`. The caller must have synchronized the stream before calling.
#[inline]
pub fn record_compute_since(start: Option<Instant>) {
    if let Some(t) = start {
        COUNTERS
            .compute_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        COUNTERS.compute_calls.fetch_add(1, Ordering::Relaxed);
    }
}

/// One bucket's accumulated stats.
#[derive(Clone, Copy, Debug, Default)]
pub struct StageStat {
    /// Total nanoseconds accumulated in this bucket.
    pub ns: u64,
    /// Number of regions recorded (shards / calls).
    pub count: u64,
    /// Bytes moved (uploads only; 0 otherwise).
    pub bytes: u64,
}

/// A point-in-time copy of all profiler counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProfileSnapshot {
    pub enabled: bool,
    pub host_decode_scx1: StageStat,
    pub host_decode_generic: StageStat,
    pub htod_scx1: StageStat,
    pub htod_generic: StageStat,
    pub gpu_decode: StageStat,
    pub compute: StageStat,
}

/// Snapshot the current counters (does not reset them).
pub fn snapshot() -> ProfileSnapshot {
    let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
    ProfileSnapshot {
        enabled: profile_enabled(),
        host_decode_scx1: StageStat {
            ns: load(&COUNTERS.host_decode_ns[0]),
            count: load(&COUNTERS.host_decode_shards[0]),
            bytes: 0,
        },
        host_decode_generic: StageStat {
            ns: load(&COUNTERS.host_decode_ns[1]),
            count: load(&COUNTERS.host_decode_shards[1]),
            bytes: 0,
        },
        htod_scx1: StageStat {
            ns: load(&COUNTERS.htod_ns[0]),
            count: load(&COUNTERS.htod_calls[0]),
            bytes: load(&COUNTERS.htod_bytes[0]),
        },
        htod_generic: StageStat {
            ns: load(&COUNTERS.htod_ns[1]),
            count: load(&COUNTERS.htod_calls[1]),
            bytes: load(&COUNTERS.htod_bytes[1]),
        },
        gpu_decode: StageStat {
            ns: load(&COUNTERS.gpu_decode_ns),
            count: load(&COUNTERS.gpu_decode_shards),
            bytes: 0,
        },
        compute: StageStat {
            ns: load(&COUNTERS.compute_ns),
            count: load(&COUNTERS.compute_calls),
            bytes: 0,
        },
    }
}

/// Reset all counters to zero (e.g. between benchmark runs).
pub fn reset() {
    for i in 0..2 {
        COUNTERS.host_decode_ns[i].store(0, Ordering::Relaxed);
        COUNTERS.host_decode_shards[i].store(0, Ordering::Relaxed);
        COUNTERS.htod_ns[i].store(0, Ordering::Relaxed);
        COUNTERS.htod_bytes[i].store(0, Ordering::Relaxed);
        COUNTERS.htod_calls[i].store(0, Ordering::Relaxed);
    }
    COUNTERS.gpu_decode_ns.store(0, Ordering::Relaxed);
    COUNTERS.gpu_decode_shards.store(0, Ordering::Relaxed);
    COUNTERS.compute_ns.store(0, Ordering::Relaxed);
    COUNTERS.compute_calls.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Both behaviours live in one test because the counters are a single
    // process-global; splitting them into separate `#[test]`s would let cargo's
    // parallel runner interleave their `reset()`/`record()` calls and clobber
    // the shared state.
    #[test]
    fn record_hooks_behave() {
        // `None` (disabled-profiling) starts record nothing.
        reset();
        record_host_decode_since(CodecClass::Scx1, None);
        record_htod_since(CodecClass::Generic, None, 1024);
        record_gpu_decode_since(None);
        record_compute_since(None);
        let snap = snapshot();
        assert_eq!(snap.host_decode_scx1.count, 0);
        assert_eq!(snap.htod_generic.bytes, 0);
        assert_eq!(snap.gpu_decode.count, 0);
        assert_eq!(snap.compute.count, 0);

        // A concrete `Some(Instant)` (what `start()` returns when enabled)
        // accumulates; `reset()` clears.
        let t = Some(Instant::now());
        record_htod_since(CodecClass::Scx1, t, 4096);
        let snap = snapshot();
        assert_eq!(snap.htod_scx1.count, 1);
        assert_eq!(snap.htod_scx1.bytes, 4096);
        reset();
        assert_eq!(snapshot().htod_scx1.count, 0);
    }
}
