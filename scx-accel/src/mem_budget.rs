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

/// faer's leading dimension for a `Mat` of `n_rows` rows of `elem_bytes` each.
///
/// `faer::Mat` pads its column stride to a 64-byte boundary, so a `Mat` of
/// `n_rows` rows actually reserves `pad(n_rows) × n_cols × elem_bytes`. Measured
/// on faer 0.24: `Mat::<f32>::zeros(1, 1000)` asks the allocator for 64 000
/// bytes, not 4 000 — a 16× overshoot — and `Mat::<f64>::zeros(1, 1000)` for
/// 64 000 rather than 8 000. The rounding quantum is `64 / elem_bytes`
/// (f32 → 16 rows, f64 → 8).
///
/// [`plan_gram_row_block`] budgets against this rather than the logical size,
/// so a narrow `n_b` cannot blow past the byte ceiling the knob advertises.
pub fn faer_padded_rows(n_rows: usize, elem_bytes: usize) -> usize {
    let quantum = (64 / elem_bytes.max(1)).max(1);
    n_rows.div_ceil(quantum).saturating_mul(quantum)
}

/// Pure planner (no allocation, unit-testable): how many rows of `a` one Gram
/// block may cover so that faer's actual allocation for it fits `budget`.
///
/// Budgets [`faer_padded_rows`]`(n_b, elem_bytes) × rows × elem_bytes` — the
/// bytes the allocator is really asked for — not the logical `n_b × rows`.
///
/// Two clamps carry the contract:
///
/// - **Never 0.** A single row whose Gram alone exceeds the budget still gets
///   its own block, so the caller always makes progress rather than looping
///   forever or erroring. Same rule as the GPU sibling
///   (`scx-gpu/src/gpu_pairwise.rs::gpu_mean_pairwise_distance_chunked`). This
///   is the one case where the result can exceed `budget`, which is why the
///   documented bound is `max(budget, one padded Gram row)`.
/// - **Never more than `n_a`.** Anything that already fits stays one block, and
///   a one-block run issues exactly the gemm the unblocked kernel did, over
///   exactly the same operands — so *blocking* contributes no drift below the
///   budget. (The self path's numbers still move once, because it switched from
///   the full square to the upper triangle. That is a separate change; this
///   clamp is what keeps the cross path bit-identical.)
pub fn plan_gram_row_block(n_a: usize, n_b: usize, elem_bytes: usize, budget: u64) -> usize {
    let per_row = (faer_padded_rows(n_b, elem_bytes) as u64)
        .saturating_mul(elem_bytes as u64)
        .max(1);
    // Clamp in `u64` *before* narrowing: on a 32-bit target `budget / per_row`
    // can exceed `usize::MAX` and a direct cast would truncate — plausibly to a
    // tiny value or 0, silently shrinking the block instead of widening it.
    ((budget / per_row).clamp(1, n_a.max(1) as u64)) as usize
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

/// Decode-prefetch depth for the streaming DE kernels, budgeted against what
/// the dense gene-chunk workspace has already claimed.
///
/// The bounded pipeline holds up to `depth` decoded **full-width** shards where
/// the hand-rolled loop held one — the gene-chunk column projection happens in
/// the consumer, after the read — so the depth is a memory decision, not just a
/// concurrency one. `dense_bytes` is the workspace the caller already committed
/// to (`n_obs x chunk x 4`), so the prefetch only gets what is left of
/// [`de_memory_budget`].
///
/// Two consequences worth knowing, because they are the memory claim:
///
/// * On a budget-bound atlas file the dense workspace has already taken the
///   budget, so this collapses to 1 and the kernel behaves exactly as it did
///   before there was a pipeline — no extra resident bytes, no regression.
/// * On ordinary shapes there is room for the full depth: 100 k cells x 500
///   genes is 200 MB of a 4 GiB budget.
///
/// A source with no [`shard_size_hint`](scx_format_io::ShardSource::shard_size_hint)
/// falls to **depth 1**, not to the unclamped request. There is no per-shard
/// byte estimate to divide by, and this is reachable on real files rather than
/// only on a hypothetical third-party source: `BackedCsrReader::shard_size_hint`
/// answers `None` whenever the catalog carries no per-shard `nnz`, and
/// `LazyShardSource` forwards that. Granting the full depth there would stack
/// `depth` decoded shards on top of a full dense workspace on exactly the files
/// that cannot be measured — an unbounded term under a function whose contract
/// is a bound.
///
/// This deliberately diverges from
/// `scx_gpu::gpu_shard_source::resolve_staging_prefetch_depth_for`, which does
/// return the unclamped request without a hint. That is consistent there: its
/// budget (`SCX_GPU_STAGING_MEMORY_BUDGET`) is unset by default, so "no budget"
/// and "no hint" both mean "no clamp was asked for". Here the budget always
/// exists (4 GiB by default), so a missing hint is a measurement failure, not an
/// opt-out.
///
/// Deliberately **not** done: shrinking the gene chunk to make room for
/// prefetch. That would change chunk counts on exactly the files where the
/// clamp binds, trading a measured win for an unmeasured one.
pub(crate) fn de_prefetch_depth(
    hint: Option<scx_format_io::ShardSizeHint>,
    dense_bytes: u64,
) -> usize {
    de_prefetch_depth_for(
        scx_format_io::prefetch::prefetch_depth(),
        hint,
        dense_bytes,
        de_memory_budget(),
    )
}

/// The pure half of [`de_prefetch_depth`]: no env reads, no pool query.
///
/// Split out for the same reason `de_gene_chunk_or_err`'s tests inject a budget
/// — `de_memory_budget()` and `prefetch_depth()` are both `OnceLock`-cached env
/// reads, so a test that went through them would be order-dependent.
pub(crate) fn de_prefetch_depth_for(
    requested: usize,
    hint: Option<scx_format_io::ShardSizeHint>,
    dense_bytes: u64,
    budget: u64,
) -> usize {
    let Some(hint) = hint else {
        // Unmeasurable, so unbounded — take the floor rather than the request.
        return 1;
    };
    let remaining = budget.saturating_sub(dense_bytes);
    clamp_prefetch_depth(requested, hint.decoded_bytes(), remaining)
}

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
    fn faer_padding_is_measured_not_assumed() {
        // Quantum is 64 bytes / elem. Values pinned against a live
        // `Mat::zeros` allocation probe on faer 0.24.
        assert_eq!(faer_padded_rows(1, 4), 16);
        assert_eq!(faer_padded_rows(3, 4), 16);
        assert_eq!(faer_padded_rows(16, 4), 16);
        assert_eq!(faer_padded_rows(17, 4), 32);
        assert_eq!(faer_padded_rows(1, 8), 8);
        assert_eq!(faer_padded_rows(7, 8), 8);
        assert_eq!(faer_padded_rows(17, 8), 24);
        // A shape already on the boundary must not be inflated.
        assert_eq!(faer_padded_rows(4096, 4), 4096);
        assert_eq!(faer_padded_rows(0, 4), 0);
    }

    #[test]
    fn a_narrow_gram_still_fits_the_budget_once_padded() {
        // The case the logical-size planner got wrong: `n_b = 1` f32 pads to 16
        // rows, so budgeting `n_b * elem` overshoots the real allocation 16×.
        let budget = 1 << 20; // 1 MiB
        let rows = plan_gram_row_block(10_000_000, 1, 4, budget);
        let real_bytes = (faer_padded_rows(1, 4) as u64) * (rows as u64) * 4;
        assert!(
            real_bytes <= budget,
            "block of {rows} rows really allocates {real_bytes} bytes, over the {budget} budget"
        );
        // And the logical calculation is what would have overshot, so this test
        // is not vacuous.
        let logical_rows = (budget / 4) as usize;
        assert!(
            logical_rows > rows * 8,
            "expected the padded planner to be much tighter than the logical one \
             ({logical_rows} vs {rows})"
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

    // --- de_prefetch_depth ------------------------------------------------
    //
    // The depth is a memory decision: the pipeline holds `depth` decoded
    // full-width shards where the hand-rolled loop held one.

    fn hint(nnz: usize) -> Option<scx_format_io::ShardSizeHint> {
        Some(scx_format_io::ShardSizeHint {
            max_rows: 1,
            max_nnz: nnz,
        })
    }

    /// No per-shard estimate means the budget cannot be honoured, so the depth
    /// takes the floor rather than the request.
    ///
    /// Reachable on real files, not just a third-party source:
    /// `BackedCsrReader::shard_size_hint` answers `None` whenever the catalog
    /// carries no per-shard `nnz`. Returning the request there would put an
    /// unbounded term under a function whose contract is a bound.
    #[test]
    fn de_prefetch_depth_without_a_hint_falls_to_the_floor() {
        assert_eq!(de_prefetch_depth_for(4, None, 0, 1024), 1);
        assert_eq!(de_prefetch_depth_for(4, None, 1024, 1024), 1);
        // And a roomy budget does not buy back the depth: the shard size, not
        // the budget, is what is unknown.
        assert_eq!(de_prefetch_depth_for(4, None, 0, u64::MAX), 1);
    }

    /// A budget-bound file: the dense gene-chunk workspace already claimed the
    /// budget, so the prefetch gets nothing and the kernel reverts to the
    /// pre-pipeline sequential behaviour rather than adding resident bytes.
    #[test]
    fn de_prefetch_depth_collapses_to_one_when_the_workspace_took_the_budget() {
        let budget = 4 * 1024 * 1024;
        assert_eq!(de_prefetch_depth_for(4, hint(1 << 20), budget, budget), 1);
        // And past it — `saturating_sub`, not a wrap into a huge remainder.
        assert_eq!(
            de_prefetch_depth_for(4, hint(1 << 20), budget * 2, budget),
            1
        );
    }

    /// Room for everything: a small shard against a large remainder.
    #[test]
    fn de_prefetch_depth_keeps_the_request_when_there_is_room() {
        assert_eq!(de_prefetch_depth_for(4, hint(8), 0, 4 * 1024 * 1024), 4);
    }

    /// Charging the *requested* gene chunk rather than the one actually
    /// allocated collapses the depth on a narrow gene view — the regression
    /// this arm records, with the numbers that make it matter.
    ///
    /// 10 M cells projected to 10 genes: the caller asks for 500-gene chunks,
    /// `clamp_de_gene_chunk` leaves the request alone because it fits, and the
    /// buffer is `n_obs x min(500, 10)` — about 400 MB. Billing the request
    /// instead charges ~20 GB against a 4 GiB budget, leaves nothing, and pins
    /// the depth at 1 while depth 4 fits many times over.
    #[test]
    fn charging_the_request_instead_of_the_active_chunk_collapses_the_depth() {
        let n_obs = 10_000_000u64;
        let budget = 4 * 1024 * 1024 * 1024u64;
        let shard = Some(scx_format_io::ShardSizeHint {
            max_rows: 16_384,
            max_nnz: 16_384 * 10,
        });

        let allocated = n_obs * 10 * 4; // active chunk = min(500, 10)
        let as_requested = n_obs * 500 * 4; // what charging the request bills

        assert_eq!(
            de_prefetch_depth_for(4, shard, as_requested, budget),
            1,
            "charging the request must be what collapses the depth — \
             otherwise this arm proves nothing"
        );
        assert_eq!(
            de_prefetch_depth_for(4, shard, allocated, budget),
            4,
            "charging only the allocated workspace must leave room for the \
             full depth"
        );
    }

    /// The interesting middle, and the arm that would catch a clamp applied to
    /// the wrong quantity: exactly two shards fit the remainder, so the depth is
    /// two even though four were requested.
    #[test]
    fn de_prefetch_depth_grants_what_the_remainder_holds() {
        // `decoded_bytes()` for max_rows=1, max_nnz=nnz is (1+1)*8 + nnz*8.
        let per_shard = hint(126).unwrap().decoded_bytes();
        assert_eq!(per_shard, 1024);
        // Budget 3 KiB, workspace 1 KiB -> 2 KiB left -> two shards.
        assert_eq!(de_prefetch_depth_for(4, hint(126), 1024, 3 * 1024), 2);
    }
}
