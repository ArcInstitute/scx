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

/// Default byte budget for one pairwise-distance Gram block (256 MiB).
/// Env-overridable via `SCX_ACCEL_PAIRWISE_MEMORY_BUDGET`.
///
/// Sized so the common shapes stay a single block — the whole existing
/// `scx-accel` test and bench grid tops out at 80 MB — while the atlas-scale
/// self-distance that motivated the budget (100 K control cells at f32 is a
/// 40 GB Gram) blocks into a working set that fits a laptop.
pub const DEFAULT_PAIRWISE_MEMORY_BUDGET: u64 = 256 * 1024 * 1024;

/// Read the pairwise-Gram byte budget, honouring `SCX_ACCEL_PAIRWISE_MEMORY_BUDGET`.
/// Unparseable / zero values fall back to [`DEFAULT_PAIRWISE_MEMORY_BUDGET`].
/// Cached on first read.
///
/// This bounds **one** Gram block. `compute_energy_distance` evaluates
/// perturbations on a rayon `par_iter`, so the host-side Gram term is
/// `rayon::current_num_threads() × budget`, not `budget`.
pub fn pairwise_memory_budget() -> u64 {
    static B: OnceLock<u64> = OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("SCX_ACCEL_PAIRWISE_MEMORY_BUDGET")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|b| *b > 0)
            .unwrap_or(DEFAULT_PAIRWISE_MEMORY_BUDGET)
    })
}

/// Pure planner (no allocation, unit-testable): how many rows of `a` one Gram
/// block may cover so that `rows × n_b × elem_bytes` fits `budget`.
///
/// Two clamps carry the contract:
///
/// - **Never 0.** A single row whose Gram alone exceeds the budget still gets
///   its own block, so the caller always makes progress rather than looping
///   forever or erroring. Same rule as the GPU sibling
///   (`scx-gpu/src/gpu_pairwise.rs::gpu_mean_pairwise_distance_chunked`).
/// - **Never more than `n_a`.** Anything that already fits stays one block, and
///   a one-block run issues exactly the gemm the unblocked kernel did, over
///   exactly the same operands — so *blocking* contributes no drift below the
///   budget. (The self path's numbers still move once, because it switched from
///   the full square to the upper triangle. That is a separate change; this
///   clamp is what keeps the cross path bit-identical.)
pub fn plan_gram_row_block(n_a: usize, n_b: usize, elem_bytes: usize, budget: u64) -> usize {
    let per_row = (n_b as u64).saturating_mul(elem_bytes as u64).max(1);
    ((budget / per_row) as usize).clamp(1, n_a.max(1))
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
/// This is a shared *policy* knob, not one literal pool object. It caps Harmony's
/// private integration pool, and — since PCA's reductions started partitioning
/// their output — the number of column blocks those reductions split into
/// (`scx_accel::pca::colblocks::block_count`) plus PCA's decode-prefetch depth.
/// PCA no longer builds a private pool or a per-accumulator RAM cap: there is
/// one shared accumulator, so on that op this knob bounds speed and memory only
/// and provably cannot change the numbers. Ambient global-pool sizing for the
/// many `rayon::current_num_threads()` callers stays controlled by
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

    // ── plan_gram_row_block ───────────────────────────────────────────

    #[test]
    fn gram_block_is_the_whole_input_when_it_fits() {
        // The property the existing distance suite rests on: below the budget
        // there is exactly one block, so the computation is the untiled one and
        // its bits do not move. 2000x2000 f64 = 32 MB, well inside 256 MiB.
        assert_eq!(
            plan_gram_row_block(2000, 2000, 8, DEFAULT_PAIRWISE_MEMORY_BUDGET),
            2000
        );
        // The widest shape in `benches/distances.rs` (1000 x 10000, f64 = 80 MB).
        assert_eq!(
            plan_gram_row_block(1000, 10_000, 8, DEFAULT_PAIRWISE_MEMORY_BUDGET),
            1000
        );
    }

    #[test]
    fn gram_block_fits_the_budget_once_it_clamps() {
        // 100 K x 100 K f32 is the 40 GB case. At 256 MiB the block must be
        // small enough that one block's Gram fits.
        let n = 100_000usize;
        let budget = DEFAULT_PAIRWISE_MEMORY_BUDGET;
        let rows = plan_gram_row_block(n, n, 4, budget);
        assert!(rows < n, "expected clamping at 40 GB, got the whole input");
        assert!(
            (rows as u64) * (n as u64) * 4 <= budget,
            "block of {rows} rows needs {} bytes, over the {budget}-byte budget",
            (rows as u64) * (n as u64) * 4
        );
    }

    #[test]
    fn gram_block_never_reaches_zero() {
        // A single row's Gram over budget: still one row, so the caller makes
        // progress instead of spinning on an empty block.
        assert_eq!(plan_gram_row_block(1000, 1_000_000, 8, 4), 1);
        assert_eq!(plan_gram_row_block(1000, 1000, 4, 0), 1);
        // Degenerate inputs must not divide by zero or return 0.
        assert_eq!(plan_gram_row_block(0, 0, 4, 1024), 1);
        assert_eq!(plan_gram_row_block(5, 0, 4, 1024), 5);
    }

    #[test]
    fn gram_block_scales_with_the_budget() {
        // 10 000 columns of f32 = 40 000 B/row.
        assert_eq!(plan_gram_row_block(1_000_000, 10_000, 4, 40_000), 1);
        assert_eq!(plan_gram_row_block(1_000_000, 10_000, 4, 400_000), 10);
        assert_eq!(plan_gram_row_block(1_000_000, 10_000, 4, 4_000_000), 100);
        // f64 halves the rows for the same budget.
        assert_eq!(plan_gram_row_block(1_000_000, 10_000, 8, 4_000_000), 50);
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
