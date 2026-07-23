//! Opt-in per-stage CPU timing profiler for the streaming accelerator path.
//!
//! Enabled by setting `SCX_CPU_PROFILE=1` (any non-empty, non-`"0"` value). This
//! is the CPU-path twin of [`scx_gpu::profile`](../../scx-gpu/src/profile.rs):
//! it quantifies where a streaming CPU accelerator op (PCA, HVG, DE, pflog, …)
//! spends wall-clock time so we can tell whether a hot path is dominated by
//! **decode / I-O** (→ codec / prefetch work) or by **reduction / marshalling**
//! (→ kernel / Python-handoff work). This is the *ranking oracle* for the
//! Phase-2 performance tasks: no `×` speedup claim is meaningful until we know
//! which bucket a given op is bound by at a given scale.
//!
//! Four buckets:
//! - `io`          — raw byte fetch of a shard section before decode
//!   ([`ScxReader::section_bytes`]). For a **local mmap** reader this is an O(1)
//!   bounds-checked slice, so `io` is ~0 and the page-fault cost of touching
//!   those bytes is attributed to `decode`; the bucket is meaningful chiefly on
//!   the cloud/range-read path. Documented, not a defect.
//! - `decode`      — host-side codec decode of a shard, split by codec class
//!   (`Scx1` vs `Generic` = None/Zstd/Lz4Shuffle/Pcodec/ShufDeltaZstd) because
//!   `auto` files are mixed-codec and the decode cost differs per codec.
//! - `reduction`   — per-shard compute/accumulation inside an accel kernel
//!   (recorded *after* the shard decode returns, so it stays disjoint from
//!   `decode`).
//! - `marshalling` — assembly of the Python result buffer (`Vec<Vec<f32>>` →
//!   `PyArray2`) at the pyscx write-back boundary.
//!
//! When disabled (the default), every hook short-circuits on a single relaxed
//! atomic load via [`profile_enabled`]: [`start`] returns `None`, so no
//! `Instant` is taken and no counter is touched — zero measurable hot-path cost.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use scx_codec::CodecId;

/// Codec class for the `decode` per-codec split.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CodecClass {
    /// Scx1 — Delta-Golomb indptr + FOR-BP indices + Rice values.
    Scx1,
    /// None / Zstd / Lz4Shuffle / Pcodec / ShufDeltaZstd.
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

    /// Map a concrete [`CodecId`] to its profiler class.
    #[inline]
    pub fn from_codec(codec: CodecId) -> Self {
        match codec {
            CodecId::Scx1 => CodecClass::Scx1,
            _ => CodecClass::Generic,
        }
    }
}

struct Counters {
    io_ns: AtomicU64,
    io_calls: AtomicU64,
    io_bytes: AtomicU64,
    decode_ns: [AtomicU64; 2],
    decode_shards: [AtomicU64; 2],
    decode_bytes: [AtomicU64; 2],
    reduction_ns: AtomicU64,
    reduction_calls: AtomicU64,
    marshalling_ns: AtomicU64,
    marshalling_calls: AtomicU64,
    marshalling_bytes: AtomicU64,
}

static COUNTERS: Counters = Counters {
    io_ns: AtomicU64::new(0),
    io_calls: AtomicU64::new(0),
    io_bytes: AtomicU64::new(0),
    decode_ns: [AtomicU64::new(0), AtomicU64::new(0)],
    decode_shards: [AtomicU64::new(0), AtomicU64::new(0)],
    decode_bytes: [AtomicU64::new(0), AtomicU64::new(0)],
    reduction_ns: AtomicU64::new(0),
    reduction_calls: AtomicU64::new(0),
    marshalling_ns: AtomicU64::new(0),
    marshalling_calls: AtomicU64::new(0),
    marshalling_bytes: AtomicU64::new(0),
};

/// Whether profiling is enabled. Resolved once from `SCX_CPU_PROFILE` and
/// cached; the cost when disabled is a single relaxed atomic load.
#[inline]
pub fn profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SCX_CPU_PROFILE")
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

/// Record a raw byte-fetch (`io`) region of `bytes` bytes. No-op if `start` was
/// `None`.
#[inline]
pub fn record_io_since(start: Option<Instant>, bytes: usize) {
    if let Some(t) = start {
        COUNTERS
            .io_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        COUNTERS.io_calls.fetch_add(1, Ordering::Relaxed);
        COUNTERS.io_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
    }
}

/// Record a host codec-decode region for one shard of `bytes` encoded bytes.
/// No-op if `start` was `None`.
#[inline]
pub fn record_decode_since(class: CodecClass, start: Option<Instant>, bytes: usize) {
    if let Some(t) = start {
        let i = class.idx();
        COUNTERS.decode_ns[i].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        COUNTERS.decode_shards[i].fetch_add(1, Ordering::Relaxed);
        COUNTERS.decode_bytes[i].fetch_add(bytes as u64, Ordering::Relaxed);
    }
}

/// Record a per-shard reduction (kernel accumulation) region. No-op if `start`
/// was `None`.
#[inline]
pub fn record_reduction_since(start: Option<Instant>) {
    if let Some(t) = start {
        COUNTERS
            .reduction_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        COUNTERS.reduction_calls.fetch_add(1, Ordering::Relaxed);
    }
}

/// RAII guard for a per-shard reduction region: records the elapsed time into
/// the `reduction` bucket when dropped. Zero-cost when profiling is disabled
/// (holds `None`, drop is a no-op). Bind it *after* the shard decode returns so
/// the region stays disjoint from `decode`; it then covers the accumulation
/// through the end of the enclosing scope (including an early `?` return).
pub struct ReductionGuard(Option<Instant>);

impl Drop for ReductionGuard {
    #[inline]
    fn drop(&mut self) {
        record_reduction_since(self.0.take());
    }
}

/// Begin a [`ReductionGuard`] timing region. Returns a guard holding `None`
/// (a no-op on drop) when profiling is disabled.
#[inline]
pub fn reduction_guard() -> ReductionGuard {
    ReductionGuard(start())
}

/// Record a Python result-marshalling region of `bytes` bytes. No-op if `start`
/// was `None`.
#[inline]
pub fn record_marshalling_since(start: Option<Instant>, bytes: usize) {
    if let Some(t) = start {
        COUNTERS
            .marshalling_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        COUNTERS.marshalling_calls.fetch_add(1, Ordering::Relaxed);
        COUNTERS
            .marshalling_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }
}

/// One bucket's accumulated stats.
#[derive(Clone, Copy, Debug, Default)]
pub struct StageStat {
    /// Total nanoseconds accumulated in this bucket.
    pub ns: u64,
    /// Number of regions recorded (shards / calls).
    pub count: u64,
    /// Bytes moved / decoded / marshalled (0 where not tracked).
    pub bytes: u64,
}

/// A point-in-time copy of all CPU profiler counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuProfileSnapshot {
    pub enabled: bool,
    pub io: StageStat,
    pub decode_scx1: StageStat,
    pub decode_generic: StageStat,
    pub reduction: StageStat,
    pub marshalling: StageStat,
}

/// Snapshot the current counters (does not reset them).
pub fn snapshot() -> CpuProfileSnapshot {
    let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
    CpuProfileSnapshot {
        enabled: profile_enabled(),
        io: StageStat {
            ns: load(&COUNTERS.io_ns),
            count: load(&COUNTERS.io_calls),
            bytes: load(&COUNTERS.io_bytes),
        },
        decode_scx1: StageStat {
            ns: load(&COUNTERS.decode_ns[0]),
            count: load(&COUNTERS.decode_shards[0]),
            bytes: load(&COUNTERS.decode_bytes[0]),
        },
        decode_generic: StageStat {
            ns: load(&COUNTERS.decode_ns[1]),
            count: load(&COUNTERS.decode_shards[1]),
            bytes: load(&COUNTERS.decode_bytes[1]),
        },
        reduction: StageStat {
            ns: load(&COUNTERS.reduction_ns),
            count: load(&COUNTERS.reduction_calls),
            bytes: 0,
        },
        marshalling: StageStat {
            ns: load(&COUNTERS.marshalling_ns),
            count: load(&COUNTERS.marshalling_calls),
            bytes: load(&COUNTERS.marshalling_bytes),
        },
    }
}

/// Reset all counters to zero (e.g. between benchmark runs).
pub fn reset() {
    COUNTERS.io_ns.store(0, Ordering::Relaxed);
    COUNTERS.io_calls.store(0, Ordering::Relaxed);
    COUNTERS.io_bytes.store(0, Ordering::Relaxed);
    for i in 0..2 {
        COUNTERS.decode_ns[i].store(0, Ordering::Relaxed);
        COUNTERS.decode_shards[i].store(0, Ordering::Relaxed);
        COUNTERS.decode_bytes[i].store(0, Ordering::Relaxed);
    }
    COUNTERS.reduction_ns.store(0, Ordering::Relaxed);
    COUNTERS.reduction_calls.store(0, Ordering::Relaxed);
    COUNTERS.marshalling_ns.store(0, Ordering::Relaxed);
    COUNTERS.marshalling_calls.store(0, Ordering::Relaxed);
    COUNTERS.marshalling_bytes.store(0, Ordering::Relaxed);
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
        record_io_since(None, 1024);
        record_decode_since(CodecClass::Scx1, None, 2048);
        record_reduction_since(None);
        record_marshalling_since(None, 4096);
        let snap = snapshot();
        assert_eq!(snap.io.count, 0);
        assert_eq!(snap.decode_scx1.count, 0);
        assert_eq!(snap.reduction.count, 0);
        assert_eq!(snap.marshalling.count, 0);

        // A concrete `Some(Instant)` (what `start()` returns when enabled)
        // accumulates into the right bucket; `reset()` clears.
        let t = Some(Instant::now());
        record_decode_since(CodecClass::Scx1, t, 4096);
        record_decode_since(CodecClass::Generic, Some(Instant::now()), 8192);
        record_io_since(Some(Instant::now()), 256);
        record_reduction_since(Some(Instant::now()));
        record_marshalling_since(Some(Instant::now()), 512);
        let snap = snapshot();
        assert_eq!(snap.decode_scx1.count, 1);
        assert_eq!(snap.decode_scx1.bytes, 4096);
        assert_eq!(snap.decode_generic.count, 1);
        assert_eq!(snap.decode_generic.bytes, 8192);
        assert_eq!(snap.io.count, 1);
        assert_eq!(snap.io.bytes, 256);
        assert_eq!(snap.reduction.count, 1);
        assert_eq!(snap.marshalling.count, 1);
        assert_eq!(snap.marshalling.bytes, 512);

        // CodecClass mapping.
        assert_eq!(CodecClass::from_codec(CodecId::Scx1), CodecClass::Scx1);
        assert_eq!(CodecClass::from_codec(CodecId::Zstd), CodecClass::Generic);
        assert_eq!(CodecClass::from_codec(CodecId::Pcodec), CodecClass::Generic);

        reset();
        assert_eq!(snapshot().decode_scx1.count, 0);
        assert_eq!(snapshot().marshalling.bytes, 0);
    }
}
