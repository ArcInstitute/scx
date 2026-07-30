//! Shared CPU memory-budget clamps for accelerator working sets.
//!
//! Several CPU DE kernels densify a gene chunk into a row-major `f32` buffer of
//! `n_obs × chunk` before ranking (`diffexp/cpu.rs`, `csc/pdex.rs`). The chunk
//! size is caller-driven and was previously unclamped, so a large
//! `gene_chunk_size` on an atlas-scale `n_obs` (tens of millions) could request
//! hundreds of GB and OOM the host. This module provides the CPU analogue of
//! the GPU `clamp_chunk_for_budget` (`scx-gpu/src/gpu_diffexp.rs`): a pure,
//! unit-testable clamp plus a thin env-configurable budget reader.

use std::sync::OnceLock;

use crate::error::{AccelError, Result};

/// Default CPU dense-DE-workspace budget in bytes (4 GiB). Env-overridable via
/// `SCX_ACCEL_DE_MEMORY_BUDGET`.
pub const DEFAULT_DE_MEMORY_BUDGET: u64 = 4 * 1024 * 1024 * 1024;

/// Lowest gene chunk the clamp will fall to before declaring the dense
/// workspace un-fittable (the caller then surfaces a clear error rather than
/// letting the allocation OOM).
pub const MIN_DE_GENE_CHUNK: usize = 32;

/// Read the CPU dense-DE-workspace byte budget, honouring the
/// `SCX_ACCEL_DE_MEMORY_BUDGET` env override. Unparseable / zero values fall
/// back to [`DEFAULT_DE_MEMORY_BUDGET`]. Cached on first read (env knob, not
/// expected to change mid-run; avoids the `std::env` lock on the DE path).
pub fn de_memory_budget() -> u64 {
    static B: OnceLock<u64> = OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("SCX_ACCEL_DE_MEMORY_BUDGET")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|b| *b > 0)
            .unwrap_or(DEFAULT_DE_MEMORY_BUDGET)
    })
}

/// Pure parse of the `SCX_ACCEL_NUM_THREADS` value: `Some(n)` for a positive
/// integer, `None` for unset / zero / unparseable. Split out so the env-cached
/// [`accel_num_threads`] can be unit-tested without touching process env.
fn parse_accel_num_threads(raw: Option<String>) -> Option<usize> {
    raw.and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
}

/// Accelerator-wide thread ceiling from `SCX_ACCEL_NUM_THREADS`.
///
/// Returns `Some(n)` (n ≥ 1) when the env var is a positive integer, else
/// `None` — in which case callers keep their own default, so an unset knob
/// preserves today's behaviour exactly. Cached on first read.
///
/// This is a shared *policy* knob, not one literal pool object: it caps the two
/// private rayon pools the accelerators build — Harmony's integration pool and
/// the PCA covariance-accumulator pool. The PCA pool's memory-derived worker
/// cap still applies; this only lowers it further (a single shared pool could
/// not also honour that per-accumulator RAM bound). Ambient global-pool sizing
/// for the many `rayon::current_num_threads()` callers stays controlled by
/// `RAYON_NUM_THREADS`.
pub fn accel_num_threads() -> Option<usize> {
    static N: OnceLock<Option<usize>> = OnceLock::new();
    *N.get_or_init(|| parse_accel_num_threads(std::env::var("SCX_ACCEL_NUM_THREADS").ok()))
}

/// Pure clamp (no allocation, unit-testable): the largest gene chunk
/// `≤ requested` whose dense `f32` workspace (`n_obs × chunk × 4` bytes) fits
/// `budget_bytes`, floored at [`MIN_DE_GENE_CHUNK`].
///
/// Returns `(chunk, fits)`. `fits` is `false` only when even the floor chunk
/// exceeds the budget — the caller should then surface a descriptive error.
pub fn clamp_de_gene_chunk(requested: usize, n_obs: usize, budget_bytes: u64) -> (usize, bool) {
    let requested = requested.max(1);
    // Bytes for one gene column of the dense f32 scatter buffer.
    let per_gene = (n_obs as u64).saturating_mul(4).max(1);
    let max_chunk = (budget_bytes / per_gene) as usize; // may be 0
    if max_chunk >= requested {
        (requested, true)
    } else if max_chunk >= MIN_DE_GENE_CHUNK {
        (max_chunk, true) // clamped down, still fits
    } else {
        (MIN_DE_GENE_CHUNK, false) // even the floor won't fit
    }
}

/// Resolve a caller `gene_chunk_size` against the CPU DE workspace budget,
/// logging a clamp and erroring when even [`MIN_DE_GENE_CHUNK`] does not fit.
pub fn de_gene_chunk_or_err(requested: usize, n_obs: usize, context: &str) -> Result<usize> {
    let budget = de_memory_budget();
    let (chunk, fits) = clamp_de_gene_chunk(requested, n_obs, budget);
    if !fits {
        let floor_bytes = (n_obs as u64)
            .saturating_mul(4)
            .saturating_mul(MIN_DE_GENE_CHUNK as u64);
        return Err(AccelError::InvalidInput(format!(
            "{context}: the dense DE workspace for n_obs={n_obs} needs {floor_bytes} bytes \
             even at the floor chunk of {MIN_DE_GENE_CHUNK} genes, exceeding the CPU memory \
             budget of {budget} bytes. Raise SCX_ACCEL_DE_MEMORY_BUDGET, use a backed/CSC \
             route, or subset genes."
        )));
    }
    if chunk < requested {
        log::warn!(
            "{context}: clamped DE gene chunk {requested} -> {chunk} to fit the {budget}-byte \
             CPU workspace budget (n_obs={n_obs}); set SCX_ACCEL_DE_MEMORY_BUDGET to change."
        );
    }
    Ok(chunk)
}

/// Pure clamp for the decode-prefetch depth.
///
/// Moved to [`scx_format_io::prefetch`] in Phase 4.2 alongside the pipeline it
/// bounds; re-exported here so `mem_budget::clamp_prefetch_depth` still resolves
/// at the HVG call sites. See [`scx_format_io::prefetch::clamp_prefetch_depth`]
/// for the semantics and its unit test.
pub use scx_format_io::prefetch::clamp_prefetch_depth;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accel_num_threads_parses_positive_only() {
        assert_eq!(parse_accel_num_threads(Some("4".into())), Some(4));
        assert_eq!(parse_accel_num_threads(Some(" 8 ".into())), Some(8));
        assert_eq!(parse_accel_num_threads(None), None);
        assert_eq!(parse_accel_num_threads(Some("0".into())), None);
        assert_eq!(parse_accel_num_threads(Some("abc".into())), None);
        assert_eq!(parse_accel_num_threads(Some("-2".into())), None);
        assert_eq!(parse_accel_num_threads(Some("".into())), None);
    }

    #[test]
    fn clamp_passes_through_when_it_fits() {
        // 1 GiB budget, n_obs=1000 → per_gene=4000 B → max_chunk=268435.
        let (chunk, fits) = clamp_de_gene_chunk(500, 1000, 1024 * 1024 * 1024);
        assert_eq!(chunk, 500);
        assert!(fits);
    }

    #[test]
    fn clamp_reduces_but_still_fits() {
        // budget lets through ~100 genes at n_obs; request 500 → clamp to 100.
        let n_obs = 1000usize;
        let budget = (n_obs as u64) * 4 * 100; // exactly 100 gene columns
        let (chunk, fits) = clamp_de_gene_chunk(500, n_obs, budget);
        assert_eq!(chunk, 100);
        assert!(fits);
    }

    #[test]
    fn clamp_reports_unfittable_below_floor() {
        // Budget below MIN_DE_GENE_CHUNK columns → (floor, false).
        let n_obs = 1_000_000usize;
        let budget = (n_obs as u64) * 4 * (MIN_DE_GENE_CHUNK as u64 - 1);
        let (chunk, fits) = clamp_de_gene_chunk(500, n_obs, budget);
        assert_eq!(chunk, MIN_DE_GENE_CHUNK);
        assert!(!fits);
    }

    #[test]
    fn or_err_errors_when_unfittable() {
        // A budget of 1 byte can't hold any chunk at a large n_obs.
        let err = de_gene_chunk_or_err_with_budget(500, 10_000_000, "test", 1);
        assert!(err.is_err());
    }

    #[test]
    fn or_err_passes_when_fittable() {
        let ok = de_gene_chunk_or_err_with_budget(64, 1000, "test", 1024 * 1024 * 1024);
        assert_eq!(ok.unwrap(), 64);
    }

    // Test shim that injects the budget instead of reading the env var, so the
    // env-cached `de_memory_budget()` OnceLock cannot make the test order-dependent.
    fn de_gene_chunk_or_err_with_budget(
        requested: usize,
        n_obs: usize,
        context: &str,
        budget: u64,
    ) -> Result<usize> {
        let (chunk, fits) = clamp_de_gene_chunk(requested, n_obs, budget);
        if !fits {
            return Err(AccelError::InvalidInput(format!("{context}: unfittable")));
        }
        Ok(chunk)
    }
}
