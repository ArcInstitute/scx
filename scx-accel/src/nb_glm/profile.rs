//! Env-gated per-phase wall-time profiler for the pseudobulk NB-GLM path
//! (Stage B0). Mirrors the `scx-gpu` `SCX_GPU_PROFILE` profiler
//! (`scx-gpu/src/profile.rs`) but spans the CPU and GPU NB-GLM orchestrators +
//! the shared post-fit tail, so a single profiled run reveals where the
//! per-target wall-time actually goes (fit vs the unaccelerated host tail).
//!
//! Zero-cost when `SCX_NBGLM_PROFILE` is unset: [`start`] returns `None` and the
//! `Option<Timer>` guard does nothing on drop. Accumulators are process-global
//! atomics keyed by [`Phase`] (no string hashing on the hot path). Surfaced to
//! Python via `pyscx.accel.nb_glm_profile_snapshot()` / `_reset()`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// One timed phase of the NB-GLM pipeline. `as usize` indexes the accumulators.
#[derive(Clone, Copy, Debug)]
pub enum Phase {
    /// pyscx per-target group-major → gene-major counts transpose.
    Transpose = 0,
    /// CPU orchestrator: per-gene MLE pass (rayon IRLS ↔ Cox–Reid).
    CpuMlePass = 1,
    /// CPU orchestrator: per-gene shrinkage refit pass (rayon).
    CpuShrinkPass = 2,
    /// GPU orchestrator: MLE fit launch (htod + kernel + dtoh).
    GpuMleFit = 3,
    /// GPU orchestrator: cross-gene trend fit + prior variance (host).
    GpuTrendPrior = 4,
    /// GPU orchestrator: shrinkage fit launch (htod + kernel + dtoh).
    GpuShrinkFit = 5,
    /// Shared tail: Wald inference + Cook's distance (rayon).
    WaldCooks = 6,
    /// Shared tail: multiple-testing correction (independent filter + BH).
    Mtc = 7,
    /// Shared tail: result + diagnostics assembly.
    Assembly = 8,
    /// pyscx: pseudobulk aggregation (`aggregate_pseudobulk`), once per call.
    Aggregate = 9,
    /// pyscx: output DataFrame construction (`build_de_dataframe`), once per call.
    Dataframe = 10,
    /// pyscx: per-(target,gene) result flatten into column vectors (incl. gene-name
    /// clones), once per call.
    Flatten = 11,
    /// Orchestrator prep: median-ratio size factors + base means (per target).
    SizeFactors = 12,
    /// GPU orchestrator: reconstruct `GeneState` vecs from the flat device fit.
    Reconstruct = 13,
}

const N_PHASES: usize = 14;
const PHASE_NAMES: [&str; N_PHASES] = [
    "transpose",
    "cpu_mle_pass",
    "cpu_shrink_pass",
    "gpu_mle_fit",
    "gpu_trend_prior",
    "gpu_shrink_fit",
    "wald_cooks",
    "mtc",
    "assembly",
    "aggregate",
    "dataframe",
    "flatten",
    "size_factors",
    "reconstruct",
];

static NS: [AtomicU64; N_PHASES] = [const { AtomicU64::new(0) }; N_PHASES];
static CALLS: [AtomicU64; N_PHASES] = [const { AtomicU64::new(0) }; N_PHASES];

/// Whether profiling is enabled (read once from `SCX_NBGLM_PROFILE`; any
/// non-empty value other than `"0"` enables).
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SCX_NBGLM_PROFILE")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    })
}

/// Add `ns` to a phase's accumulator (no-op when disabled).
pub fn record(phase: Phase, ns: u64) {
    if !enabled() {
        return;
    }
    NS[phase as usize].fetch_add(ns, Ordering::Relaxed);
    CALLS[phase as usize].fetch_add(1, Ordering::Relaxed);
}

/// RAII timer: records `phase`'s elapsed wall-time on drop. `None` (and thus a
/// no-op) when profiling is disabled, so call sites stay branch-free:
/// `let _t = profile::start(Phase::Mtc);`.
pub struct Timer {
    phase: Phase,
    start: Instant,
}

impl Drop for Timer {
    fn drop(&mut self) {
        record(self.phase, self.start.elapsed().as_nanos() as u64);
    }
}

/// Start timing `phase`, or `None` when profiling is disabled.
pub fn start(phase: Phase) -> Option<Timer> {
    if enabled() {
        Some(Timer {
            phase,
            start: Instant::now(),
        })
    } else {
        None
    }
}

/// One phase's accumulated stats.
#[derive(Clone, Copy, Debug)]
pub struct PhaseStat {
    pub name: &'static str,
    pub ms: f64,
    pub calls: u64,
}

/// Snapshot all phase accumulators (cumulative since the last [`reset`]).
pub fn snapshot() -> Vec<PhaseStat> {
    (0..N_PHASES)
        .map(|i| PhaseStat {
            name: PHASE_NAMES[i],
            ms: NS[i].load(Ordering::Relaxed) as f64 / 1.0e6,
            calls: CALLS[i].load(Ordering::Relaxed),
        })
        .collect()
}

/// Reset all accumulators to zero (call between bench runs).
pub fn reset() {
    for i in 0..N_PHASES {
        NS[i].store(0, Ordering::Relaxed);
        CALLS[i].store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_has_all_phases_and_reset_clears() {
        // `enabled()` is env-driven and cached; we test the structural surface
        // (names present, reset zeroes) without asserting accumulation, which
        // would require the env var set before first touch.
        reset();
        let s = snapshot();
        assert_eq!(s.len(), N_PHASES);
        assert_eq!(s[Phase::Mtc as usize].name, "mtc");
        assert!(s.iter().all(|p| p.ms == 0.0 && p.calls == 0));
    }
}
