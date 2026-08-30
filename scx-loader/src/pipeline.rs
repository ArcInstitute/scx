use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::record_batch::RecordBatch;
use scx_format_io::deletion_vectors::DeletionVectors;
use scx_format_io::reader::ScxReader;
use scx_format_io::BackedCsrReader;

use crate::batch::Batch;
use crate::budget::{profiling_enabled, BudgetBreakdown, BudgetModel, PYTHON_OVERHEAD_BYTES};
use crate::decode_stage::{build_category_dicts, decode_stage, CategoryDict};
use crate::error::{LoaderError, Result};
use crate::io_stage::io_stage;
use crate::projection::HvgProjection;
use crate::shuffle::ShardShuffler;

/// Bounded join deadline for `Drop` and `join_epoch_handles` shutdown.
/// If the I/O or decode thread does not finish within this window the join
/// is abandoned and the thread handle is detached — preferable to wedging
/// the worker process exit forever.
///
/// Also the deadline `crate::runtime::BoundedRuntime` is built with, so the
/// plan-driven loaders bound their teardown by the same window as the training
/// pipeline; two constants here would drift silently.
pub(crate) const SHUTDOWN_DEADLINE: Duration = Duration::from_secs(5);

/// Polling interval for `JoinHandle::is_finished()` waits during bounded
/// shutdown.
const SHUTDOWN_POLL: Duration = Duration::from_millis(20);

/// Configuration for the training data loader pipeline.
///
/// The memory budget model is computed by `memory_budget()` below.
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Mini-batch size (default: 1024).
    pub batch_size: usize,
    /// Number of shards read per I/O group (default: 8).
    ///
    /// This knob trades **I/O locality** against **shuffle quality**.
    /// Shards within a group are read together (mostly-sequential I/O, lower
    /// peak RSS), and the per-epoch shuffle mixes cells *only within* a
    /// group (see `shuffle.rs`). A larger group → better-mixed minibatches
    /// (higher shuffle entropy) at the cost of more resident decoded shards;
    /// a smaller group → cheaper I/O and lower memory, but cells from the
    /// same on-disk neighborhood co-occur in minibatches more often, which
    /// can bias SGD. Auto-tuning may shrink this to fit `max_memory_mb`; if
    /// it drops below [`MIN_SHUFFLE_QUALITY_SHARD_GROUP_SIZE`] the loader
    /// warns and sets [`MemoryBudget::shuffle_quality_degraded`].
    pub shard_group_size: usize,
    /// Ring buffer depth — number of pre-built batches to buffer (default: 4).
    pub prefetch_batches: usize,
    /// Gene indices for HVG projection. None = use all genes.
    pub hvg_indices: Option<Vec<u32>>,
    /// Obs metadata column names to include in each batch.
    pub obs_columns: Vec<String>,
    /// Apply total-count normalization (default: true).
    pub normalize: bool,
    /// Apply log1p transformation (default: true).
    pub log1p: bool,
    /// Normalization target sum (default: 1e4).
    pub target_sum: f64,
    /// Apply PFlog (v4) / shifted-log normalization on raw counts (Booeshaghi
    /// et al.) instead of normalize/log1p (default: false).
    ///
    /// PFlog is itself a normalization, so it is **mutually exclusive** with
    /// `normalize`/`log1p`: when `true` it takes precedence and those flags are
    /// ignored. Per cell it computes
    /// `z_ij = log1p(4α·x_ij) − (1/D)·Σ_k log1p(4α·x_ik)` on **raw counts** —
    /// no per-cell depth (it cancels under the Anscombe scale). The centering
    /// denominator `D` is over the **full transcriptome** (not the HVG-projected
    /// panel — see [`crate::projection::HvgProjection::scatter_pflog_row`]).
    pub pflog: bool,
    /// PFlog NB overdispersion `α` (only used when `pflog` is true). The
    /// matrix-wide Anscombe pseudocount is `1/(4α)`. `None` (default) →
    /// **estimate once at loader construction** via `scx_accel::estimate_alpha`
    /// over the dataset's raw CSR shards (single-modality only); `Some(α)` pins
    /// a value (e.g. a reference α from `pyscx.accel.pflog`). Resolved to
    /// `Some` before decoding.
    pub pflog_alpha: Option<f64>,
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Memory budget in MB (default: 512).
    /// Pipeline auto-tunes shard_group_size and prefetch_batches to fit.
    pub max_memory_mb: usize,
    /// When `true`, treat `max_memory_mb` as a *floor* rather than a hard
    /// ceiling: [`TrainingPipeline::new`] raises the effective budget to fit
    /// the file's requested configuration (so a full-width ~33k-gene file does
    /// not silently shrink `batch_size` or trip the shuffle-quality warning),
    /// clamped to [`ADAPTIVE_BUDGET_CAP_MB`]. Set by the Python bindings when
    /// the caller does **not** pass an explicit `max_memory_mb`; an explicit
    /// budget keeps `false` so the hard-ceiling auto-tune (and its warnings)
    /// behave exactly as before. Default `false`.
    pub auto_memory_budget: bool,
    /// Phase H.1: optional modality filter.
    ///
    /// `None` = legacy global / single-modality behaviour: load every CSR
    /// shard in the file (matches the v1 invariant).
    ///
    /// `Some(id)` (1-based) = restrict the I/O stage to shards stamped
    /// with the given `modality_id`. Cells (obs) are global across
    /// modalities, so `n_obs` is unchanged; only the X matrix is
    /// per-modality. Deletion vectors compose with this filter: v2 stores
    /// deletions as global obs rows, and `io_stage::reconstruct_deletion_map`
    /// re-buckets them into whichever modality's shard ranges are being read
    /// (`reconstruct_deletion_map_cross_modality` pins that).
    ///
    /// `None` on a **multimodal** file is rejected at construction: with no
    /// filter the I/O stage pools every modality's shards, and since each
    /// modality independently tiles `[0, n_obs)` two of them would claim the
    /// same global row. `Some(mid)` is checked too — that modality's own shards
    /// must tile the obs axis exactly once. See
    /// [`ensure_csr_ranges_are_readable`].
    pub modality_id: Option<u8>,
    /// Opt out of the `hvg_indices` range check against `n_vars`.
    ///
    /// Set **only** by `MultimodalTrainingDataset`, which fans one panel across
    /// every selected modality (one `TrainingPipeline` each). Those modalities
    /// have different widths — RNA ~33k, ADT ~100 — so a single shared panel
    /// cannot be in range for all of them, and range-checking it would reject
    /// an ordinary CITE-seq call outright. The cost of the opt-out is the
    /// behaviour [`crate::projection::HvgProjection::new`] exists to prevent:
    /// an index past a given modality's width yields an output column that is
    /// silently always zero for that modality.
    ///
    /// This is deliberately **not** keyed on `modality_id`. A modality id means
    /// "this pipeline reads one modality's shards", which is also true of
    /// `TrainingDataset(path, modality="adt")` and of the implicit
    /// alphabetically-first fallback `TrainingDataset` takes on a multimodal
    /// file. Those have exactly one panel and one unambiguous `n_vars`, so the
    /// shared-panel argument does not apply to them and they are checked like
    /// any single-modality loader. Default `false`.
    pub shared_hvg_panel: bool,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        LoaderConfig {
            batch_size: 1024,
            shard_group_size: 8,
            prefetch_batches: 4,
            hvg_indices: None,
            obs_columns: Vec::new(),
            normalize: true,
            log1p: true,
            target_sum: 1e4,
            pflog: false,
            pflog_alpha: None,
            seed: 42,
            max_memory_mb: 512,
            auto_memory_budget: false,
            modality_id: None,
            shared_hvg_panel: false,
        }
    }
}

/// Upper bound for the adaptive default memory budget (see
/// [`LoaderConfig::auto_memory_budget`]). A full-width single-cell file held in
/// one CSR shard needs roughly 1–1.5 GB (decoded-shard cache + mmap-resident
/// file + per-batch buffer); this cap covers the common full-width case while
/// still letting genuinely huge files fall back to the hard-ceiling auto-tune
/// (with its protective warnings) instead of silently reserving unbounded RAM.
pub const ADAPTIVE_BUDGET_CAP_MB: usize = 4096;

impl LoaderConfig {
    /// Validate the configuration, returning `ConfigError` for invalid settings.
    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(LoaderError::ConfigError {
                reason: "batch_size must be > 0".to_string(),
            });
        }
        if self.shard_group_size == 0 {
            return Err(LoaderError::ConfigError {
                reason: "shard_group_size must be > 0".to_string(),
            });
        }
        if self.prefetch_batches == 0 {
            return Err(LoaderError::ConfigError {
                reason: "prefetch_batches must be > 0".to_string(),
            });
        }
        if self.target_sum <= 0.0 {
            return Err(LoaderError::ConfigError {
                reason: "target_sum must be > 0.0".to_string(),
            });
        }
        if self.pflog {
            if let Some(a) = self.pflog_alpha {
                if a <= 0.0 || a.is_nan() || a.is_infinite() {
                    return Err(LoaderError::ConfigError {
                        reason: "pflog_alpha must be positive and finite".to_string(),
                    });
                }
            }
        }
        if self.pflog && (self.normalize || self.log1p) {
            log::warn!(
                "pflog=true takes precedence; normalize/log1p flags are ignored \
                 (pflog is itself a normalization)"
            );
        }
        if self.max_memory_mb < 64 {
            return Err(LoaderError::ConfigError {
                reason: "max_memory_mb must be >= 64 (minimum viable budget)".to_string(),
            });
        }
        Ok(())
    }
}

/// Result of memory budget computation. Contains the effective parameters
/// after auto-tuning to fit within `max_memory_mb`.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Effective shard_group_size (may be reduced to fit budget).
    pub shard_group_size: usize,
    /// Effective prefetch_batches (may be reduced to fit budget).
    pub prefetch_batches: usize,
    /// Effective batch_size (may be reduced to fit budget for large gene counts).
    pub batch_size: usize,
    /// Estimated total memory in bytes. Equal to `breakdown.total_bytes`, and
    /// **excludes** the mmap'd file (ORG-9.10-5) — do not subtract
    /// [`Self::mmap_bytes`] from it.
    pub estimated_bytes: usize,
    /// Size of the mmap'd SCX file in bytes. **Reported, never budgeted**: the
    /// kernel page cache is evictable under pressure, so no loader class counts
    /// it against `max_memory_mb`. Not included in [`Self::estimated_bytes`].
    pub mmap_bytes: usize,
    /// True if estimated memory exceeds the budget even at all minimums.
    pub budget_exceeded: bool,
    /// Per-component breakdown (cache, batch buffer, lookahead, transient,
    /// python overhead). Sequential paths report a zero `lookahead_overhead`
    /// and a zero `transient` since they don't carry plan-tuple staging or
    /// per-batch obs scratch. Mmap is tracked separately on
    /// `mmap_bytes`; the breakdown excludes mmap, matching the index-plan
    /// path's convention.
    pub breakdown: BudgetBreakdown,
    /// True when auto-tuning reduced `shard_group_size` below the
    /// shuffle-quality threshold ([`MIN_SHUFFLE_QUALITY_SHARD_GROUP_SIZE`])
    /// to fit `max_memory_mb`. Shuffling happens *within* a shard group
    /// (`shuffle.rs`), so a small group lowers shuffle entropy: cells from
    /// the same on-disk neighborhood co-occur in minibatches more often,
    /// which can bias SGD. Surfaced so callers (and tests) can detect the
    /// I/O-locality-vs-shuffle-quality tradeoff that a tight budget forced.
    pub shuffle_quality_degraded: bool,
}

/// `shard_group_size` at or above which within-group shuffling is
/// considered to provide adequate entropy. Below this, auto-tuning has
/// traded shuffle quality for a smaller memory footprint and
/// [`MemoryBudget::shuffle_quality_degraded`] is set with a `log::warn!`.
///
/// The loader shuffles only within a shard group (see `shuffle.rs`), so the
/// group size is the knob that trades I/O locality (small groups → mostly
/// sequential reads, lower RSS) against shuffle entropy (large groups →
/// better-mixed minibatches). A value of 4 keeps at least a few shards
/// mixing per group; raise `max_memory_mb` (or `shard_group_size`) to
/// recover full-entropy shuffling.
pub(crate) const MIN_SHUFFLE_QUALITY_SHARD_GROUP_SIZE: usize = 4;

/// Decide whether auto-tuning degraded shuffle quality and, if so, emit a
/// `log::warn!`. Returns the flag stored on [`MemoryBudget`]. Degraded means
/// the effective group size both dropped below the quality threshold **and**
/// was reduced from what the caller requested (so a caller that deliberately
/// asked for a tiny group isn't warned spuriously). Called once per
/// `compute_memory_budget`, so the warning fires once per budget computation.
fn shuffle_quality_degraded(requested: usize, effective: usize) -> bool {
    let degraded = effective < MIN_SHUFFLE_QUALITY_SHARD_GROUP_SIZE && effective < requested;
    if degraded {
        log::warn!(
            "loader auto-tune reduced shard_group_size from {} to {} to fit \
             max_memory_mb (below the shuffle-quality threshold of {}). \
             Shuffling is scoped to a shard group, so minibatch entropy is \
             reduced and SGD may see correlated cells together. Raise \
             max_memory_mb (or set hvg_indices to shrink per-batch memory) to \
             restore full-entropy shuffling.",
            requested,
            effective,
            MIN_SHUFFLE_QUALITY_SHARD_GROUP_SIZE,
        );
    }
    degraded
}

/// Refuse a file whose CSR shards do not tile the obs axis exactly once for the
/// reader that is about to consume them.
///
/// Every loader in this crate resolves a cell by its **global obs row**, so the
/// shard list it reads must claim each row exactly once. Two ways that fails:
///
/// * **A multimodal file read unscoped.** Each modality independently tiles
///   `[0, n_obs)`, so the flattened list claims every row once per modality.
///   `TrainingPipeline` would pool them into shard groups and drop or duplicate
///   cells depending on the epoch's shuffle; `IndexPlanLoader` and
///   `SparseCellSetLoader` go through `BackedCsrReader`, whose row index is
///   built from the same flattened ranges, so they silently answer from
///   whichever modality the index happened to keep — verified: a two-modality
///   file with equal widths returns one modality's rows with no warning, and
///   with differing widths it fails mid-iteration with a message blaming the
///   file. Neither of those is a modality *choice* the caller made.
/// * **A malformed single-modality tiling** — the merge / append / compact
///   defect. `ShardGroupIndex::build` catches the duplicate only when both
///   shards land in the same shard group, which the shuffle re-draws each epoch
///   (and never, at `shard_group_size == 1`).
///
/// `scoped_modality` is `Some(mid)` when the caller selected one, in which case
/// that modality's own tiling is what must hold; `None` means the whole
/// flattened list is being read.
pub(crate) fn ensure_csr_ranges_are_readable(
    reader: &ScxReader,
    scoped_modality: Option<u8>,
    who: &str,
) -> Result<()> {
    match scoped_modality {
        Some(mid) => {
            if !reader
                .catalog()
                .modality_csr_ranges_tile_obs(mid, reader.n_obs())
            {
                return Err(LoaderError::ConfigError {
                    reason: format!(
                        "{who}: modality {mid}'s CSR shards do not tile the obs axis \
                         [0, {}) exactly once — they overlap, leave a gap, or are \
                         missing row-range stats, so a cell cannot be resolved to a \
                         single shard. `scx info` lists the shard row ranges.",
                        reader.n_obs()
                    ),
                });
            }
        }
        None => {
            if reader.catalog().has_overlapping_csr_ranges() {
                let n_modalities = reader.n_modalities();
                let remedy = if n_modalities > 1 {
                    format!(
                        "This file has {n_modalities} modalities, each covering the whole \
                         obs axis, so the overlap is expected — but reading them pooled is \
                         not a modality choice, it is an arbitrary one. Select one \
                         (`modality_id` / `modality=` on TrainingDataset), use \
                         `MultimodalTrainingDataset` to read several at once, or extract a \
                         modality first with `scx subset --modality NAME` for the loaders \
                         that have no modality surface."
                    )
                } else {
                    "This file has a single modality, so its shards should tile the obs \
                     axis exactly; the overlap means the file is malformed. `scx info` \
                     lists the shard row ranges."
                        .to_string()
                };
                return Err(LoaderError::ConfigError {
                    reason: format!(
                        "{who}: this file's CSR shard row ranges overlap, so a cell \
                         cannot be attributed to a single shard. {remedy}"
                    ),
                });
            }
            // Overlap is only half of "exactly once". A cover with a *gap*, one
            // that stops short of `n_obs`, or one whose shards carry no
            // row-range stats passes `has_overlapping_csr_ranges` — it skips
            // stat-less entries outright — and then loses the uncovered rows:
            // `TrainingPipeline` emits a short epoch, and the plan-driven
            // loaders report an in-bounds row as "out of range" mid-iteration,
            // blaming the caller's plan for the file's defect.
            //
            // This walks the **flattened** list, which is what the unscoped path
            // actually reads — deliberately not
            // `modality_csr_ranges_tile_obs(0, n_obs)`. That looked equivalent
            // ("a single-modality file's shards are stamped `modality_id = 0`")
            // and is not: a file with a *one-entry modality table* — what
            // `from_mudata(MuData({"rna": adata}))` and a single-modality h5mu
            // ingest emit — stamps its only X with `modality_id = 1`, leaving
            // modality 0 owning no shards. Its flattened cover is perfectly
            // unambiguous, and keying on modality 0 rejected every such file
            // with a false "leave a gap".
            //
            // The overlap check above has already established the ranges are
            // disjoint and sorted, so contiguity from 0 to `n_obs` is all that
            // is left to prove.
            let n_obs = reader.n_obs();
            let mut expected: u64 = 0;
            for entry in reader.catalog().csr_shards_sorted() {
                let Some(stats) = entry.stats.as_ref() else {
                    return Err(LoaderError::ConfigError {
                        reason: format!(
                            "{who}: CSR shard '{}' carries no row-range stats, so the \
                             rows it holds cannot be located. `scx info` lists the \
                             shard row ranges.",
                            entry.name
                        ),
                    });
                };
                if stats.row_start != expected {
                    return Err(LoaderError::ConfigError {
                        reason: format!(
                            "{who}: this file's CSR shards do not cover the obs axis \
                             [0, {n_obs}) exactly once — shard '{}' starts at row {} \
                             where row {expected} was expected, so rows \
                             [{expected}, {}) belong to no shard. `scx info` lists the \
                             shard row ranges.",
                            entry.name, stats.row_start, stats.row_start
                        ),
                    });
                }
                expected = stats.row_end;
            }
            if expected != n_obs {
                return Err(LoaderError::ConfigError {
                    reason: format!(
                        "{who}: this file's CSR shards cover the obs axis only up to \
                         row {expected}, but n_obs is {n_obs} — rows \
                         [{expected}, {n_obs}) belong to no shard. `scx info` lists \
                         the shard row ranges."
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Knobs the sequential auto-tune reduces, in reduction order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SequentialParams {
    pub(crate) shard_group_size: usize,
    pub(crate) prefetch_batches: usize,
    pub(crate) batch_size: usize,
}

impl SequentialParams {
    fn from_config(config: &LoaderConfig) -> Self {
        SequentialParams {
            shard_group_size: config.shard_group_size,
            prefetch_batches: config.prefetch_batches,
            batch_size: config.batch_size,
        }
    }
}

/// The sequential (`TrainingPipeline`) arm of [`crate::budget::BudgetModel`].
///
/// Reduction order — `prefetch_batches` to 2, then `shard_group_size` to 1,
/// then `batch_size` **halved** to 64 — is deliberate and pinned
/// (`budget_reduces_prefetch_then_shard_group_then_batch`): `batch_size` is the
/// only one of the three a caller picks for statistical rather than memory
/// reasons, so it gives last.
pub(crate) struct SequentialBudgetModel {
    n_output_genes: usize,
    shard_target_rows: usize,
    avg_nnz_per_cell: f64,
}

impl SequentialBudgetModel {
    pub(crate) fn new(
        n_output_genes: usize,
        shard_target_rows: usize,
        avg_nnz_per_cell: f64,
    ) -> Self {
        SequentialBudgetModel {
            n_output_genes,
            shard_target_rows,
            avg_nnz_per_cell,
        }
    }
}

impl BudgetModel for SequentialBudgetModel {
    type Params = SequentialParams;

    fn estimate(&self, p: SequentialParams) -> BudgetBreakdown {
        estimate_breakdown(
            p.shard_group_size,
            p.prefetch_batches,
            p.batch_size,
            self.n_output_genes,
            self.shard_target_rows,
            self.avg_nnz_per_cell,
        )
    }

    fn reduce(&self, p: SequentialParams) -> Option<SequentialParams> {
        if p.prefetch_batches > 2 {
            Some(SequentialParams {
                prefetch_batches: p.prefetch_batches - 1,
                ..p
            })
        } else if p.shard_group_size > 1 {
            Some(SequentialParams {
                shard_group_size: p.shard_group_size - 1,
                ..p
            })
        } else if p.batch_size > 64 {
            Some(SequentialParams {
                batch_size: (p.batch_size / 2).max(64),
                ..p
            })
        } else {
            None
        }
    }
}

/// Number of output columns the batch is actually allocated at.
///
/// The **deduplicated** panel width, not `hvg.len()`: costing the raw length
/// over-reserves for any panel with duplicates and can auto-tune `batch_size`
/// down to afford memory that is never used.
fn n_output_genes_for(config: &LoaderConfig, n_vars: u64) -> usize {
    match &config.hvg_indices {
        Some(hvg) => HvgProjection::output_cols_for(hvg),
        None => n_vars as usize,
    }
}

/// Compute the memory budget for the training pipeline.
///
/// Memory model:
/// ```text
/// n_output_genes     = unique genes in hvg_indices if present, else n_vars
/// shard_buffer       = (shard_group_size + 1) × decoded_shard_bytes
/// batch_buffer       = (max(prefetch_batches, 2) + 1) × batch_size × n_output_genes × 4
/// overhead           = ~50 MB (Python interpreter, numpy, Arrow, thread stacks)
/// ```
///
/// If total exceeds `max_memory_mb`, reduces parameters in order:
/// 1. `prefetch_batches` (minimum 2)
/// 2. `shard_group_size` (minimum 1)
/// 3. `batch_size` (halve each step, minimum 64)
///
/// **`file_size_bytes` is reported, never budgeted** (ORG-9.10-5). The mmap'd
/// file lives in the kernel page cache, which is evictable under pressure; the
/// two plan-driven loaders have always excluded it, and counting it here made
/// this path mean something different by the same name. It also made the
/// auto-tune useless on exactly the files it matters for: any file above
/// [`ADAPTIVE_BUDGET_CAP_MB`] exceeded its budget on the mmap term alone,
/// collapsing to `batch_size=64, shard_group_size=1, prefetch_batches=2` with
/// `budget_exceeded` set, whatever the caller asked for.
pub fn compute_memory_budget(
    config: &LoaderConfig,
    n_vars: u64,
    shard_target_rows: u32,
    avg_nnz_per_cell: f64,
    file_size_bytes: usize,
) -> MemoryBudget {
    let model = SequentialBudgetModel::new(
        n_output_genes_for(config, n_vars),
        shard_target_rows as usize,
        avg_nnz_per_cell,
    );
    compute_memory_budget_with(&model, config, n_vars, file_size_bytes)
}

/// [`compute_memory_budget`] against a model the caller already built.
///
/// `TrainingPipeline::new` needs the same model twice — once to resolve an
/// adaptive budget from the requested config's own need, once to tune — and
/// then keeps it for [`TrainingPipeline::pin_effective_config`]. Building it
/// once is not just cheaper: the adaptive path reads the output width off the
/// `HvgProjection` while `n_output_genes_for` re-derives it from
/// `config.hvg_indices`, and those are two sources for one number that must
/// agree. Now there is one.
pub(crate) fn compute_memory_budget_with(
    model: &SequentialBudgetModel,
    config: &LoaderConfig,
    n_vars: u64,
    file_size_bytes: usize,
) -> MemoryBudget {
    let tuned = crate::budget::tune(
        model,
        SequentialParams::from_config(config),
        config.max_memory_mb.saturating_mul(1024 * 1024),
    );

    if tuned.exhausted {
        log::warn!(
            "estimated memory ({} MB) exceeds budget ({} MB) even at minimums \
             (batch_size={}, shard_group_size=1, prefetch_batches=2). \
             Consider setting hvg_indices to reduce n_output_genes from {}.",
            tuned.breakdown.total_bytes / (1024 * 1024),
            config.max_memory_mb,
            tuned.params.batch_size,
            n_vars,
        );
    }

    MemoryBudget {
        shard_group_size: tuned.params.shard_group_size,
        prefetch_batches: tuned.params.prefetch_batches,
        batch_size: tuned.params.batch_size,
        estimated_bytes: tuned.breakdown.total_bytes,
        mmap_bytes: file_size_bytes,
        budget_exceeded: tuned.exhausted,
        breakdown: tuned.breakdown,
        shuffle_quality_degraded: shuffle_quality_degraded(
            config.shard_group_size,
            tuned.params.shard_group_size,
        ),
    }
}

/// Estimate the per-component memory breakdown for given parameters.
///
/// Excludes mmap, like every other budget model. The file's pages do fault
/// into RSS during an epoch, but they are evictable page cache and are
/// reported on `MemoryBudget::mmap_bytes` rather than charged (ORG-9.10-5).
fn estimate_breakdown(
    shard_group_size: usize,
    prefetch_batches: usize,
    batch_size: usize,
    n_output_genes: usize,
    shard_target_rows: usize,
    avg_nnz_per_cell: f64,
) -> BudgetBreakdown {
    // Decoded shard stores i64 indptr + i32 indices + f32 values = 12 bytes/nnz
    const BYTES_PER_NNZ_DECODED: usize = 12;

    // Decoded shard size: CSR arrays. Every multiply is done with
    // `checked_mul`/`checked_add` so pathological configs (petabyte shard
    // sizes, UB-flavoured integer overflow on 32-bit builds) return
    // `usize::MAX` rather than silently wrap. Callers reading this as a
    // "fits in memory?" hint correctly see an over-budget answer.
    let decoded_shard_bytes = shard_target_rows
        .checked_mul(avg_nnz_per_cell as usize)
        .and_then(|v| v.checked_mul(BYTES_PER_NNZ_DECODED))
        .and_then(|v| v.checked_add(shard_target_rows.saturating_add(1).saturating_mul(8)))
        .unwrap_or(usize::MAX);

    // I/O pipeline overlap: shard_group_size in channel + 1 being decoded
    let shard_buffer = shard_group_size
        .checked_add(1)
        .and_then(|v| v.checked_mul(decoded_shard_bytes))
        .unwrap_or(usize::MAX);

    // Batch ring: channel capacity + 1 being consumed by Python
    let live_batches = prefetch_batches.max(2) + 1;
    let batch_bytes = batch_size
        .checked_mul(n_output_genes)
        .and_then(|v| v.checked_mul(4)) // f32
        .unwrap_or(usize::MAX);
    let batch_buffer = live_batches.saturating_mul(batch_bytes);

    BudgetBreakdown::new(
        shard_buffer,
        batch_buffer,
        /* lookahead_overhead_bytes */ 0,
        /* transient_bytes */ 0,
        PYTHON_OVERHEAD_BYTES,
    )
}

// ---------------------------------------------------------------------------
// E1: TrainingPipeline — triple-buffered pipeline coordinator
// ---------------------------------------------------------------------------

/// Triple-buffered training pipeline coordinator.
///
/// Orchestrates three concurrent stages:
/// 1. **I/O stage** (`std::thread` driving a `tokio::runtime::Builder::new_current_thread()`
///    runtime via `block_on`): reads shard groups from the SCX file
/// 2. **Decode stage** (`std::thread` + a per-pipeline `rayon::ThreadPool`):
///    shuffles, projects, densifies, normalizes
/// 3. **GPU stage** (caller): consumes pre-built `Batch`es via `next_batch()`
///
/// **Fork-safety contract**. Both the tokio current-thread runtime and the
/// per-pipeline rayon `ThreadPool` are constructed *lazily inside
/// `start_epoch`*, i.e. after any fork has happened, so a forked child that
/// constructs its own pipeline does not inherit fork-hostile state from the
/// parent. `pyscx.from_anndata` in the parent — which lazily initialises
/// rayon's *global* pool as a side-effect — is the historical wedge case; this
/// design eliminates dependence on that pool entirely.
///
/// The constructor does own threads, however: [`Self::new`] runs `read_obs`
/// and the PFlog α estimate inside [`crate::pool::cpu_pool`], which builds that
/// pool. They are this process's threads, created after the fork — which is the
/// property fork-safety actually needs — but the earlier claim that the value
/// holds no worker threads at construction is no longer true. See
/// `docs/multithreading.md` § Per-worker thread footprint.
///
/// See [docs/multithreading.md §Training data loader](../../docs/multithreading.md#training-data-loader-triple-buffered-pipeline).
pub struct TrainingPipeline {
    config: LoaderConfig,
    reader: Arc<ScxReader>,
    obs_metadata: RecordBatch,
    /// Stable global category dictionaries (one per categorical obs column),
    /// computed once at construction so categorical codes are identical across
    /// batches and epochs. See [`build_category_dicts`].
    cat_dicts: HashMap<String, CategoryDict>,
    deletion_vectors: Option<DeletionVectors>,
    n_vars: u64,
    #[allow(dead_code)]
    shard_target_rows: u32,
    projection: Option<HvgProjection>,
    memory_budget: MemoryBudget,
    /// The budget model this pipeline was tuned against. Kept so
    /// [`Self::pin_effective_config`] can re-estimate without re-opening the
    /// file — three numbers, no I/O.
    budget_model: SequentialBudgetModel,
    /// CSR shard count, for rebuilding the shuffler on a pin.
    n_csr_shards: usize,
    // Runtime state — all lazy / per-epoch:
    batch_rx: Option<crossbeam_channel::Receiver<Batch>>,
    /// I/O stage runs on a dedicated `std::thread` that owns a
    /// `tokio::runtime::Builder::new_current_thread()` runtime. The runtime
    /// lives only as long as the I/O thread; nothing in `TrainingPipeline`
    /// holds the runtime across fork boundaries.
    io_handle: Option<std::thread::JoinHandle<Result<()>>>,
    decode_handle: Option<std::thread::JoinHandle<Result<()>>>,
    /// Per-pipeline rayon thread pool. `None` before the first
    /// `start_epoch()`; once built, persists across epochs
    /// for the same `TrainingPipeline` and is dropped in `shutdown()` /
    /// `Drop`. Workers are *not* shared with rayon's global registry.
    decode_pool: Option<Arc<rayon::ThreadPool>>,
    shuffler: ShardShuffler,
}

impl TrainingPipeline {
    /// Create a new training pipeline from an SCX file path and config.
    ///
    /// Opens the file, reads metadata and deletion vectors, computes the
    /// memory budget, and prepares the pipeline. Does NOT start any pipeline
    /// stages — call `start_epoch()` to begin iteration.
    pub fn new(path: impl AsRef<Path>, mut config: LoaderConfig) -> Result<Self> {
        let profile = profiling_enabled();
        let t_start = Instant::now();

        let _new_span =
            tracing::trace_span!("TrainingPipeline::new", pid = std::process::id(),).entered();
        tracing::trace!("entry");

        config.validate()?;

        // Open SCX file
        let t0 = Instant::now();
        let reader = Arc::new(ScxReader::open(path.as_ref())?);
        tracing::trace!(
            elapsed_us = t0.elapsed().as_micros() as u64,
            "ScxReader::open"
        );
        if profile {
            eprintln!("[scx-loader profile] open: {:?}", t0.elapsed());
        }

        // Read header metadata
        let header = reader.header();
        let shard_target_rows = header.shard_target_rows;

        // Phase H.1: per-modality filtering. When `modality_id` is set,
        // n_vars and the shard count come from the modality table /
        // per-modality catalog filter, not the file-wide header.
        let (n_vars, n_csr_shards, modality_nnz) = if let Some(mid) = config.modality_id {
            if mid == 0 {
                return Err(LoaderError::ConfigError {
                    reason: "modality_id must be >= 1 (0 is reserved for global / legacy)"
                        .to_string(),
                });
            }
            let info = reader
                .modality_info(mid)
                .ok_or_else(|| LoaderError::ConfigError {
                    reason: format!(
                        "modality_id {mid} not found in file (n_modalities = {})",
                        reader.n_modalities()
                    ),
                })?;
            ensure_csr_ranges_are_readable(&reader, Some(mid), "TrainingPipeline")?;
            let n_shards = reader.catalog().csr_shards_for_modality(mid).len();
            (info.n_vars, n_shards, info.nnz)
        } else {
            // No modality selected, so the I/O stage pools `shards_sorted()`
            // (`io_stage.rs`) — every modality's shards at once. See
            // `ensure_csr_ranges_are_readable` for what that does to a cell.
            ensure_csr_ranges_are_readable(&reader, None, "TrainingPipeline")?;
            (
                header.n_vars,
                reader.catalog().shards_sorted().len(),
                header.nnz,
            )
        };

        // HVG projection, built here rather than just before it is stored,
        // because the memory budget below needs it: `adaptive_budget_mb` reads
        // `n_output_cols()` straight off it, and `compute_memory_budget` — which
        // only gets a `LoaderConfig` — recomputes the same width through
        // `HvgProjection::output_cols_for`. Building it first also means an
        // out-of-range panel is rejected before it can size anything.
        let projection = match &config.hvg_indices {
            // Range-checked — the default, including every modality-*scoped*
            // loader. An index >= n_vars matches no CSR column on either scatter
            // path, so it becomes an output column that is always exactly zero:
            // a dead input feature that trains to a zero weight and is never
            // diagnosed. `n_vars` here is already the per-modality width when a
            // modality is selected, so the bound is the right one to check.
            Some(idxs) if !config.shared_hvg_panel => {
                Some(HvgProjection::new(idxs.clone(), n_vars)?)
            }
            // Deliberately NOT range-checked, for the one caller that shares a
            // single panel across modalities of differing widths — see
            // `LoaderConfig::shared_hvg_panel` for why, and `docs/training.md`
            // for the user-facing statement of the trade-off.
            Some(idxs) => Some(HvgProjection::new_unchecked(idxs.clone())),
            None => None,
        };

        // v4 PFlog: resolve α once. `None` ⇒ estimate over the raw CSR shards
        // (single-modality only — modality-scoped estimation isn't wired, and a
        // whole-file `BackedCsrReader` on a multimodal file would pool every
        // modality's counts into one α, so a multimodal or modality-scoped
        // loader must pin `pflog_alpha`). After this, `config.pflog_alpha` is
        // `Some` whenever `config.pflog`.
        if config.pflog && config.pflog_alpha.is_none() {
            if config.modality_id.is_some() || reader.n_modalities() > 1 {
                return Err(LoaderError::ConfigError {
                    reason: "pflog_alpha must be set explicitly for a multimodal or \
                             modality-scoped loader (auto-estimation would pool modalities); \
                             estimate it once via pyscx.accel.pflog and pass pflog_alpha"
                        .to_string(),
                });
            }
            let est_reader = BackedCsrReader::new(ScxReader::open(path.as_ref())?, 0);
            // `estimate_alpha` walks shards through `scx_format_io::prefetch`,
            // which uses `rayon::in_place_scope`. Its "am I already on a worker?"
            // guard cannot see a fork — after one, `current_num_threads()` still
            // reports the parent's count and `current_thread_index()` is still
            // `None` — so on the global registry it spawns onto threads that do
            // not exist and the drain blocks forever. Inside `install` that same
            // guard sees a worker thread and takes the sequential path, which is
            // the right trade for a one-time construction-path estimate.
            let est = crate::pool::cpu_pool()
                .install(|| {
                    scx_accel::estimate_alpha(&est_reader, &scx_accel::AlphaOptions::default())
                })
                .map_err(|e| LoaderError::ConfigError {
                    reason: format!("pflog α estimation failed: {e}"),
                })?;
            log::info!(
                "pflog: estimated α={:.6} (pseudocount={:.6}, n_genes_used={}, fell_back={})",
                est.alpha,
                est.pseudocount,
                est.n_genes_used,
                est.fell_back
            );
            config.pflog_alpha = Some(est.alpha);
        }

        // Read obs metadata (full RecordBatch for column extraction).
        //
        // On a file with sharded obs metadata this is not a plain section read:
        // `read_sharded_layout_by_prefix` fans the shard decode out with
        // `par_iter`. This constructor is the lazily-constructed-in-`__iter__`
        // call a forked `DataLoader` worker makes, where rayon's inherited
        // global registry has no live threads — so without `install` a forked
        // worker hangs here on every atlas-scale file, which is every file with
        // `n_obs > shard_target_rows`. Same reason as `IndexPlanLoader::new`.
        let t0 = Instant::now();
        let obs_metadata = crate::pool::cpu_pool().install(|| reader.read_obs())?;
        if profile {
            eprintln!(
                "[scx-loader profile] read_obs: {:?} ({} rows)",
                t0.elapsed(),
                obs_metadata.num_rows()
            );
        }

        // Build stable global category dictionaries for categorical obs columns
        // once over the full obs table — so per-batch codes never drift with
        // batch composition. Also validates that requested columns exist.
        let cat_dicts = build_category_dicts(&obs_metadata, &config.obs_columns)?;

        // Load deletion vectors if present
        let t0 = Instant::now();
        let deletion_vectors = reader.read_deletion_vectors()?;
        if profile {
            eprintln!(
                "[scx-loader profile] read_deletion_vectors: {:?}",
                t0.elapsed()
            );
        }

        // Compute average nnz per cell for memory budget estimation.
        // Phase H.1: per-modality runs use the modality's nnz, not the
        // file-wide header.nnz which sums across modalities.
        let avg_nnz_per_cell = if header.n_obs > 0 {
            modality_nnz as f64 / header.n_obs as f64
        } else {
            0.0
        };

        // A shard group can never span more shards than the file holds. Clamp
        // the requested group to the shard count so (a) the decoded-shard cache
        // estimate is accurate for single-/few-shard files (otherwise a
        // 1-shard file is costed as if 8 shards were resident) and (b) the
        // shuffle-quality warning doesn't fire spuriously when the *data*, not
        // memory, caps the group below the threshold.
        config.shard_group_size = config.shard_group_size.min(n_csr_shards.max(1));

        // Compute memory budget and auto-tune parameters
        let file_size_bytes = reader.mmap().len();

        // Adaptive default budget: when the caller did not pin `max_memory_mb`,
        // raise the effective budget to fit the requested configuration so a
        // full-width file doesn't silently shrink `batch_size` (P3). The floor
        // is the configured default (small files are unchanged); the ceiling is
        // `ADAPTIVE_BUDGET_CAP_MB` (genuinely huge files still fall through to
        // the hard-ceiling auto-tune + warnings rather than reserving unbounded
        // RAM). Only ever raises, never lowers.
        // One model for the whole constructor: the adaptive raise below, the
        // tune after it, and `pin_effective_config` later all use this one.
        // Read the width off the projection built above rather than the raw
        // panel: they differ whenever the panel had duplicates, and the
        // projection's answer is the one the batch is allocated at.
        let budget_model = SequentialBudgetModel::new(
            match &projection {
                Some(proj) => proj.n_output_cols(),
                None => n_vars as usize,
            },
            shard_target_rows as usize,
            avg_nnz_per_cell,
        );

        if config.auto_memory_budget {
            // One `adaptive_budget_mb`, in `crate::budget` — this path used to
            // carry a second copy of the same arithmetic (ORG-9.10-5).
            let adaptive_mb = crate::budget::adaptive_budget_mb(
                budget_model
                    .estimate(SequentialParams::from_config(&config))
                    .total_bytes,
                config.max_memory_mb,
            );
            if adaptive_mb > config.max_memory_mb {
                log::info!(
                    "loader auto-budget: raised max_memory_mb {} -> {} MB to fit the \
                     requested configuration (batch_size={}, shard_group_size={}) \
                     without shrinking the batch. Pass an explicit max_memory_mb to pin \
                     a hard ceiling instead.",
                    config.max_memory_mb,
                    adaptive_mb,
                    config.batch_size,
                    config.shard_group_size,
                );
                config.max_memory_mb = adaptive_mb;
            }
        }

        let memory_budget =
            compute_memory_budget_with(&budget_model, &config, n_vars, file_size_bytes);

        // Propagate effective batch_size back into config
        config.batch_size = memory_budget.batch_size;

        // Create shard shuffler
        let shuffler =
            ShardShuffler::new(n_csr_shards, memory_budget.shard_group_size, config.seed)?;

        // NOTE: tokio runtime construction has moved to `start_epoch` and
        // the rayon thread pool to `ensure_decode_pool`. Both are built
        // lazily, post-fork, so a forked child that constructs its own
        // pipeline does not inherit any worker threads from the parent.

        if profile {
            eprintln!(
                "[scx-loader profile] TrainingPipeline::new total: {:?}",
                t_start.elapsed()
            );
            eprintln!("[scx-loader profile]   n_vars={n_vars}, n_shards={n_csr_shards}, shard_target_rows={shard_target_rows}");
            eprintln!("[scx-loader profile]   memory_budget: shard_group_size={}, prefetch_batches={}, estimated_bytes={}",
                memory_budget.shard_group_size, memory_budget.prefetch_batches, memory_budget.estimated_bytes);
        }

        Ok(TrainingPipeline {
            budget_model,
            n_csr_shards,
            config,
            reader,
            obs_metadata,
            cat_dicts,
            deletion_vectors,
            n_vars,
            shard_target_rows,
            projection,
            memory_budget,
            batch_rx: None,
            io_handle: None,
            decode_handle: None,
            decode_pool: None,
            shuffler,
        })
    }

    /// Lazily build the per-pipeline rayon `ThreadPool`. Called from
    /// `start_epoch` so the pool is constructed inside the worker
    /// process, after any fork. Reused across epochs.
    ///
    /// Sized through [`crate::pool::resolve_pool_threads`], the same function
    /// that sizes `cpu_pool()`, so `SCX_LOADER_CPU_THREADS` governs **both**
    /// pools a `TrainingPipeline` worker can hold. It previously read
    /// `num_cpus::get_physical()` directly and ignored the knob, which made
    /// per-worker thread count unpredictable once the constructor also began
    /// building a pool — see the footprint note in `docs/multithreading.md`.
    fn ensure_decode_pool(&mut self) -> Result<&Arc<rayon::ThreadPool>> {
        if self.decode_pool.is_none() {
            let n_threads = crate::pool::resolve_pool_threads(
                std::env::var(crate::pool::CPU_THREADS_ENV).ok().as_deref(),
            );
            let t = Instant::now();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .thread_name(|i| format!("scx-decode-rayon-{i}"))
                .build()
                .map_err(|e| {
                    LoaderError::ShutdownError(format!(
                        "failed to build per-pipeline rayon pool: {e}"
                    ))
                })?;
            tracing::trace!(
                elapsed_us = t.elapsed().as_micros() as u64,
                num_threads = n_threads,
                "per-pipeline rayon pool constructed",
            );
            self.decode_pool = Some(Arc::new(pool));
        }
        Ok(self.decode_pool.as_ref().unwrap())
    }

    /// Start a new training epoch.
    ///
    /// Generates a new shuffled shard ordering, creates bounded channels,
    /// and spawns the I/O and decode stages. Call `next_batch()` to consume
    /// batches from the pipeline.
    ///
    /// If a previous epoch is still active, joins its handles first.
    pub fn start_epoch(&mut self) -> Result<()> {
        let _start_span = tracing::trace_span!(
            "TrainingPipeline::start_epoch",
            pid = std::process::id(),
            epoch = self.shuffler.epoch(),
        )
        .entered();
        tracing::trace!("entry");

        // Join previous epoch handles if they exist
        self.join_epoch_handles()?;
        tracing::trace!("prior epoch handles joined");

        // Phase 3.1 invariant: after `join_epoch_handles`, all per-epoch
        // channel state must be reset. If this fires, a future edit broke
        // the contract that `join_epoch_handles` is the single owner of
        // `batch_rx` teardown — and `tx.send` from the decode stage that
        // about to be respawned would race with the prior epoch's receiver.
        debug_assert!(
            self.batch_rx.is_none(),
            "batch_rx must be None after join_epoch_handles before re-spawning"
        );
        debug_assert!(
            self.io_handle.is_none() && self.decode_handle.is_none(),
            "epoch handles must be None after join_epoch_handles before re-spawning"
        );

        // Generate offset-sorted shard groups for this epoch.
        // Shard groups are randomly composed (stochastic across epochs) but
        // sorted by file offset within and across groups for sequential I/O.
        // Collect into owned `Vec<u64>` first so the catalog borrow ends
        // before `ensure_decode_pool` (which needs `&mut self`).
        //
        // Phase H.1: when `modality_id` is set, the shard list is
        // filtered to that modality's CSR shards. The shuffler /
        // io_stage operate on the filtered list (per-modality positions
        // 0..N), which io_stage maps back to the original catalog
        // entries via the same filter applied internally.
        let shard_offsets: Vec<u64> = match self.config.modality_id {
            Some(mid) => self
                .reader
                .catalog()
                .csr_shards_for_modality(mid)
                .iter()
                .map(|e| e.offset)
                .collect(),
            None => self
                .reader
                .catalog()
                .shards_sorted()
                .iter()
                .map(|e| e.offset)
                .collect(),
        };

        // Per-position sort keys for the shuffle. Use each shard's `row_start`
        // (the order `csr_shards_for_modality`/`shards_sorted` already return
        // entries in) rather than the raw file offset, so the group ordering is
        // layout-invariant: every modality shares the same per-position
        // `row_start` (checked by `check_uniform_modality_layouts`), so the
        // per-modality shufflers produce identical group sequences and the
        // per-batch cell axis stays aligned even if a rewrite reordered shards on
        // disk. `shard_offsets` above stays byte offsets for the mmap smoke-read /
        // prefetch below. Missing-stats shards map to `u64::MAX` identically for
        // all modalities (stable sort keeps ties in RNG order). (H2)
        let sort_keys: Vec<u64> = match self.config.modality_id {
            Some(mid) => self
                .reader
                .catalog()
                .csr_shards_for_modality(mid)
                .iter()
                .map(|e| e.stats.as_ref().map_or(u64::MAX, |s| s.row_start))
                .collect(),
            None => self
                .reader
                .catalog()
                .shards_sorted()
                .iter()
                .map(|e| e.stats.as_ref().map_or(u64::MAX, |s| s.row_start))
                .collect(),
        };

        // Touch one byte of the first CSR shard's
        // backing region before spawning the I/O / decode threads. If the
        // mmap'd file has been poisoned (truncated, unmapped via outer
        // munmap, or unmapped because the underlying file was deleted by
        // another process and the kernel reaped the pages), the load below
        // raises SIGBUS *here*, on the consumer thread — far better than
        // hanging the I/O stage on an unreproducible read after the workers
        // have spawned. Cheap: a single byte fault on a region the I/O
        // stage was about to fault anyway.
        if let Some(&first_offset) = shard_offsets.first() {
            let mmap = self.reader.mmap();
            let off = first_offset as usize;
            if off >= mmap.len() {
                return Err(LoaderError::ShutdownError(format!(
                    "first shard offset {off} exceeds mmap length {}",
                    mmap.len()
                )));
            }
            // `black_box` forces the load to be observable and prevents the
            // optimizer from eliding the smoke-read as dead code. The slice
            // index is bounds-checked just above.
            let _ = std::hint::black_box(mmap[off]);
        }

        // Build the per-pipeline rayon pool (lazy, post-fork) before
        // spawning the decode thread that uses it.
        let decode_pool = Arc::clone(self.ensure_decode_pool()?);

        let shard_groups = self.shuffler.shuffle_epoch_sorted(&sort_keys);

        // Create bounded channels
        //
        // I/O → Decode: tokio mpsc channel. Each item is a ShardGroup containing
        // shard_group_size decoded shards. Cap at 2 for pipeline overlap (one
        // being decoded + one read-ahead), not shard_group_size which would allow
        // shard_group_size * shard_group_size decoded shards in flight.
        //
        // **Shutdown propagation chain**:
        // 1. Consumer drops `self.batch_rx` (in `join_epoch_handles`).
        // 2. Decode stage's `crossbeam tx.send(batch)` returns Err. Decode
        //    stage exits with `LoaderError::ChannelError`, dropping its
        //    end of the tokio mpsc (`io_rx`).
        // 3. I/O stage's `tokio tx.send(group).await` returns Err (receiver
        //    closed). `io_stage` exits with `LoaderError::ChannelError`.
        // 4. The current-thread runtime's `block_on(io_stage(...))` returns;
        //    the I/O thread exits, dropping its tokio runtime cleanly.
        //
        // Both stages' ChannelError return values are filtered out as
        // expected-on-shutdown by `join_epoch_handles` — the chain
        // only surfaces real errors (panics, ShutdownError, ConfigError).
        // No explicit cancellation token is needed; channel close is the
        // signal.
        let (io_tx, io_rx) = tokio::sync::mpsc::channel(2);

        // Decode → Consumer: crossbeam bounded channel, capacity = prefetch_batches
        // Use at least 2 to allow decode to run ahead of consumer.
        let batch_channel_cap = self.memory_budget.prefetch_batches.max(2);
        let (batch_tx, batch_rx) = crossbeam_channel::bounded(batch_channel_cap);

        // Spawn I/O stage as a dedicated `std::thread` that owns a tokio
        // current-thread runtime. The runtime lives only as
        // long as the I/O thread, so the `TrainingPipeline` value never
        // holds a long-lived multi-threaded tokio runtime that fork would
        // inherit. `spawn_blocking` inside `io_stage` still works because
        // the current-thread runtime drives a separate blocking thread pool.
        let t_spawn_io = Instant::now();
        let io_reader = Arc::clone(&self.reader);
        let io_dv = self.deletion_vectors.clone();
        let io_modality_id = self.config.modality_id;
        let io_handle = std::thread::Builder::new()
            .name("scx-io".to_string())
            .spawn(move || -> Result<()> {
                tracing::trace!(pid = std::process::id(), "I/O thread entered");
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| {
                        LoaderError::ShutdownError(format!(
                            "I/O thread: failed to build current-thread runtime: {e}"
                        ))
                    })?;
                let res = rt.block_on(io_stage(
                    io_reader,
                    shard_groups,
                    io_dv,
                    io_modality_id,
                    io_tx,
                ));
                tracing::trace!(
                    pid = std::process::id(),
                    ok = res.is_ok(),
                    "I/O thread exiting"
                );
                res
            })
            .map_err(|e| LoaderError::ShutdownError(format!("failed to spawn I/O thread: {e}")))?;
        tracing::trace!(
            elapsed_us = t_spawn_io.elapsed().as_micros() as u64,
            "I/O stage spawned",
        );

        // Spawn decode stage as a standard thread (CPU-bound work)
        let decode_config = self.config.clone();
        let decode_n_vars = self.n_vars;
        let decode_projection = self.projection.clone();
        let decode_obs = self.obs_metadata.clone();
        let decode_cat_dicts = self.cat_dicts.clone();
        let decode_epoch = self.shuffler.epoch().saturating_sub(1); // epoch was already incremented by shuffle_epoch()
        let t_spawn_decode = Instant::now();
        let decode_handle = std::thread::Builder::new()
            .name("scx-decode".to_string())
            .spawn(move || {
                tracing::trace!(pid = std::process::id(), "decode stage entered");
                decode_stage(
                    io_rx,
                    batch_tx,
                    &decode_config,
                    decode_n_vars,
                    decode_projection,
                    &decode_obs,
                    &decode_cat_dicts,
                    decode_epoch,
                    &decode_pool,
                )
            })
            .map_err(|e| {
                LoaderError::ShutdownError(format!("failed to spawn decode thread: {e}"))
            })?;
        tracing::trace!(
            elapsed_us = t_spawn_decode.elapsed().as_micros() as u64,
            "decode stage spawned",
        );

        self.batch_rx = Some(batch_rx);
        self.io_handle = Some(io_handle);
        self.decode_handle = Some(decode_handle);

        Ok(())
    }

    /// Get the next training batch from the pipeline.
    ///
    /// - `Ok(Some(batch))` — a batch is available.
    /// - `Ok(None)` — the epoch completed cleanly (all cells yielded, or no
    ///   epoch is active). Call `start_epoch()` again for the next epoch.
    /// - `Err(e)` — a mid-epoch I/O or decode fault (or a stage panic). The
    ///   epoch is torn down; the error is surfaced rather than silently
    ///   truncating the epoch.
    ///
    /// When the batch channel closes we cannot, on its own, tell a clean
    /// epoch end from a stage that faulted early and dropped its sender.
    /// `join_epoch_handles()` disambiguates: it returns `Ok(())` for a clean
    /// end (and for the `ChannelError` cancellation signal) and
    /// `Err(..)` for a real io/decode error or panic — so we propagate its
    /// result instead of discarding it.
    pub fn next_batch(&mut self) -> Result<Option<Batch>> {
        let Some(rx) = self.batch_rx.as_ref() else {
            return Ok(None);
        };
        let t_recv = Instant::now();
        match rx.recv() {
            Ok(batch) => {
                tracing::trace!(
                    pid = std::process::id(),
                    wait_us = t_recv.elapsed().as_micros() as u64,
                    n_rows = batch.n_rows(),
                    "next_batch: received",
                );
                Ok(Some(batch))
            }
            Err(_) => {
                // Channel closed — either a clean epoch end or a stage fault.
                // Join the handles and let their result decide which.
                tracing::trace!(
                    pid = std::process::id(),
                    "next_batch: channel closed (epoch end or fault)"
                );
                // The epoch is over either way; tear it down before deciding
                // clean-vs-fault. `join_epoch_handles` clears `batch_rx` as its
                // first statement, on every exit path including its early error
                // returns, so the teardown holds even when the join reports a
                // fault.
                self.join_epoch_handles().map(|()| None)
            }
        }
    }

    /// Total number of observations (cells) in the dataset.
    pub fn n_obs(&self) -> u64 {
        self.reader.n_obs()
    }

    /// Total number of variables (genes) in the dataset.
    pub fn n_vars(&self) -> u64 {
        self.n_vars
    }

    /// Number of output genes per batch (HVG count if projection is active,
    /// otherwise `n_vars`).
    pub fn n_output_genes(&self) -> usize {
        match &self.projection {
            Some(proj) => proj.n_output_cols(),
            None => self.n_vars as usize,
        }
    }

    /// Verdict on whether building the HVG projection changed the panel the
    /// caller passed — `None` when it was already ascending and unique.
    ///
    /// The pyo3 layer turns this into a `UserWarning`; pure-Rust callers can
    /// read the struct. See [`crate::projection::assess_hvg_panel`].
    pub fn hvg_panel(&self) -> Option<crate::projection::HvgPanelVerdict> {
        self.config
            .hvg_indices
            .as_deref()
            .and_then(crate::projection::assess_hvg_panel)
    }

    /// Effective batch_size after memory budget auto-tuning.
    /// May be less than the configured batch_size for large gene counts.
    pub fn effective_batch_size(&self) -> usize {
        self.memory_budget.batch_size
    }

    /// Memory budget diagnostics.
    /// Pin `(batch_size, shard_group_size)` **after** the budget auto-tune has
    /// already run.
    ///
    /// `MultimodalTrainingDataset` runs one pipeline per modality and must
    /// batch them in lockstep, so it forces every modality onto the
    /// cross-modality *minimum* effective config. It used to do that by
    /// building every pipeline a second time with the minimum pre-set and then
    /// `debug_assert`ing that the tuner had landed on the same answer — an
    /// assertion compiled out of the release wheels users run, guarding a
    /// property (`estimate` is monotone, so a smaller config always still fits)
    /// that nothing checked. Setting the values directly makes the property
    /// structural, and skips N full reconstructions — each of which re-opened
    /// the file and re-ran pflog α estimation over the raw CSR shards.
    ///
    /// Refuses to *raise* either knob: "the pinned config always fits" holds
    /// because the pin only ever shrinks and the model is monotone (asserted
    /// per step by [`crate::budget::tune`] and end to end by
    /// `the_sequential_reduction_chain_is_monotone`). Raising would need a
    /// fresh tune, so it is an error rather than a silent no-op.
    ///
    /// `pub(crate)`: the only caller is the multimodal wrapper in this crate,
    /// and it is not a documented Rust extension point — exporting it would
    /// commit to input contracts nothing external needs (review, round 2).
    ///
    /// `shuffle_quality_degraded` is deliberately **not** recomputed: it
    /// records what *this modality's own* tuner did, and recomputing it here
    /// would both erase that and emit a second `log::warn!` per modality, where
    /// the caller already emits one cross-modality `log::info!`.
    pub(crate) fn pin_effective_config(
        &mut self,
        batch_size: usize,
        shard_group_size: usize,
    ) -> Result<()> {
        if self.batch_rx.is_some() {
            return Err(LoaderError::ConfigError {
                reason: "pin_effective_config called with an epoch in flight".to_string(),
            });
        }
        if batch_size > self.memory_budget.batch_size
            || shard_group_size > self.memory_budget.shard_group_size
        {
            return Err(LoaderError::ConfigError {
                reason: format!(
                    "pin_effective_config may only shrink: asked for \
                     (batch_size={batch_size}, shard_group_size={shard_group_size}) \
                     against an effective (batch_size={}, shard_group_size={})",
                    self.memory_budget.batch_size, self.memory_budget.shard_group_size,
                ),
            });
        }

        // Everything that can fail is built *before* `self` is touched, so a
        // refused pin leaves the pipeline exactly as it was rather than
        // half-updated. `ShardShuffler::new` rejects a zero group size, and a
        // zero `batch_size` would yield an epoch of empty batches.
        if batch_size == 0 || shard_group_size == 0 {
            return Err(LoaderError::ConfigError {
                reason: format!(
                    "pin_effective_config requires non-zero knobs, got \
                     (batch_size={batch_size}, shard_group_size={shard_group_size})"
                ),
            });
        }
        let shuffler = ShardShuffler::new(self.n_csr_shards, shard_group_size, self.config.seed)?;

        let params = SequentialParams {
            shard_group_size,
            prefetch_batches: self.memory_budget.prefetch_batches,
            batch_size,
        };
        let breakdown = self.budget_model.estimate(params);
        self.config.batch_size = batch_size;
        self.config.shard_group_size = shard_group_size;
        self.memory_budget.batch_size = batch_size;
        self.memory_budget.shard_group_size = shard_group_size;
        self.memory_budget.estimated_bytes = breakdown.total_bytes;
        self.memory_budget.breakdown = breakdown;
        self.memory_budget.budget_exceeded = !breakdown.fits_within(self.config.max_memory_mb);
        self.shuffler = shuffler;
        Ok(())
    }

    /// The memory budget in force, in MB — the value the auto-tune ran
    /// against, which is the *resolved* one when `auto_memory_budget` raised
    /// it. Same accessor name and meaning as `IndexPlanLoader::max_memory_mb`.
    pub fn max_memory_mb(&self) -> usize {
        self.config.max_memory_mb
    }

    pub fn memory_budget_info(&self) -> &MemoryBudget {
        &self.memory_budget
    }

    /// Join the I/O and decode handles from a previous epoch, propagating errors.
    ///
    /// Drops the batch receiver first to propagate channel-close back through
    /// the decode and I/O threads, then joins each handle with a bounded
    /// deadline. A timed-out join is logged as a warning rather than
    /// returned as an error — once the consumer has dropped its receivers
    /// the only remaining sin is leaving worker threads alive, and that's a
    /// strictly better outcome than wedging the caller.
    fn join_epoch_handles(&mut self) -> Result<()> {
        // Drop the batch receiver first to unblock the decode stage
        // if it's trying to send. This propagates: decode → io_rx drop →
        // io_tx.send returns Err → io_stage exits → I/O thread's block_on
        // returns → I/O thread exits. End-to-end shutdown without
        // requiring a separate abort signal.
        self.batch_rx = None;

        // Join the I/O thread (bounded by SHUTDOWN_DEADLINE).
        //
        // ChannelError is treated as Ok on the shutdown path: it means the
        // I/O stage observed its downstream channel closing (because the
        // decode stage dropped `io_rx` after we dropped `batch_rx` above —
        // the natural propagation chain) and exited with that as its
        // return value. That is the *intended* shutdown signal, not a
        // failure mode worth surfacing.
        if let Some(handle) = self.io_handle.take() {
            match join_handle_bounded(handle, "I/O thread", SHUTDOWN_DEADLINE) {
                Ok(Some(Ok(()))) => {}
                Ok(Some(Err(LoaderError::ChannelError(_)))) => {}
                Ok(Some(Err(e))) => {
                    return Err(LoaderError::ShutdownError(format!("I/O stage error: {e}")));
                }
                Ok(None) => {
                    // Timed out — the thread is still running; we've detached
                    // its handle and accept the leak.
                }
                Err(()) => {
                    return Err(LoaderError::ShutdownError(
                        "I/O thread panicked".to_string(),
                    ));
                }
            }
        }

        // Join the decode thread (bounded). Same ChannelError-is-shutdown
        // semantics as the I/O thread above: if the consumer dropped
        // `batch_rx` mid-iteration the decode stage's `tx.send` returns
        // Err and the stage exits with `LoaderError::ChannelError`. That
        // is the cancellation path, not a fault.
        if let Some(handle) = self.decode_handle.take() {
            match join_handle_bounded(handle, "decode thread", SHUTDOWN_DEADLINE) {
                Ok(Some(Ok(()))) => {}
                Ok(Some(Err(LoaderError::ChannelError(_)))) => {}
                Ok(Some(Err(e))) => {
                    return Err(LoaderError::ShutdownError(format!(
                        "decode stage error: {e}"
                    )));
                }
                Ok(None) => {
                    // Timed out.
                }
                Err(()) => {
                    return Err(LoaderError::ShutdownError(
                        "decode stage panicked".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Explicit shutdown: drop channels, join I/O + decode threads (bounded),
    /// and release the per-pipeline rayon pool.
    ///
    /// Idempotent — safe to call multiple times. `Drop` calls this
    /// internally, but callers running under PyTorch DataLoader workers
    /// should call this from a `weakref.finalize` hook before interpreter
    /// teardown so shutdown happens while the GIL state is still healthy.
    pub fn shutdown(&mut self) {
        let _span =
            tracing::trace_span!("TrainingPipeline::shutdown", pid = std::process::id()).entered();
        tracing::trace!("entry");

        // Best-effort join of any in-flight epoch. Errors are logged, not
        // returned, since shutdown is best-effort.
        if let Err(e) = self.join_epoch_handles() {
            tracing::warn!(error = %e, "join_epoch_handles error during shutdown");
        }

        // Drop the rayon pool. Its workers exit when the last `Arc` is
        // dropped — `decode_pool` here is the only strong ref outside the
        // (now-joined) decode thread.
        if self.decode_pool.take().is_some() {
            tracing::trace!("rayon pool dropped");
        }

        tracing::trace!("exit");
    }
}

/// Join a `std::thread::JoinHandle<Result<T>>` with a bounded deadline.
///
/// - `Ok(Some(Ok(value)))` — thread finished cleanly with `Ok(value)`.
/// - `Ok(Some(Err(e)))` — thread finished cleanly returning `Err(e)`.
/// - `Ok(None)` — deadline expired; the handle has been detached
///   (the thread continues running until it exits naturally).
/// - `Err(())` — thread panicked.
///
/// `label` is used in the timeout-warning log line to identify which thread
/// was abandoned.
fn join_handle_bounded<T>(
    handle: std::thread::JoinHandle<T>,
    label: &str,
    deadline: Duration,
) -> std::result::Result<Option<T>, ()> {
    let started = Instant::now();
    while !handle.is_finished() {
        if started.elapsed() >= deadline {
            tracing::warn!(
                label = %label,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "thread did not finish within shutdown deadline; detaching",
            );
            // Drop the handle without joining → thread is detached.
            // The thread continues to run until it exits on its own; if
            // the worker process is exiting anyway, the OS reclaims it.
            return Ok(None);
        }
        std::thread::sleep(SHUTDOWN_POLL);
    }
    match handle.join() {
        Ok(value) => {
            tracing::trace!(
                label = %label,
                elapsed_us = started.elapsed().as_micros() as u64,
                "thread joined"
            );
            Ok(Some(value))
        }
        Err(_) => Err(()),
    }
}

impl Drop for TrainingPipeline {
    /// Best-effort shutdown when a pipeline is dropped without an explicit
    /// `shutdown()` call.
    ///
    /// **Interpreter-teardown probe**. When `Drop`
    /// runs during Python interpreter teardown — which can happen if a
    /// `TrainingDataset` outlives the worker process's normal lifecycle and
    /// is dropped during `_atexit` — the GIL state is partially gone and
    /// any code path that acquires it can crash the process. There is no
    /// pyo3 dependency in this crate, so we detect the hazard structurally:
    /// the I/O and decode threads are bounded by `SHUTDOWN_DEADLINE` and
    /// detach on timeout (`join_handle_bounded`), and we never call into
    /// Python during Drop. The rayon pool drops cleanly on its own; tokio
    /// has no surviving handles in `TrainingPipeline` after Phase 2's
    /// restructure.
    fn drop(&mut self) {
        let _drop_span =
            tracing::trace_span!("TrainingPipeline::drop", pid = std::process::id()).entered();
        tracing::trace!("entry");
        self.shutdown();
        tracing::trace!("exit");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_codec::CodecId;
    use scx_format_io::header::FileHeader;
    use scx_format_io::writer::ScxWriter;

    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc as StdArc;

    fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
        FileHeader::new_single_modality(n_obs, n_vars, nnz, 16384, 0, 0)
    }

    fn sample_obs(n: usize) -> RecordBatch {
        let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
        let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
        RecordBatch::try_new(
            StdArc::new(schema),
            vec![StdArc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    fn sample_var(n: usize) -> RecordBatch {
        let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
        let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
        RecordBatch::try_new(
            StdArc::new(schema),
            vec![StdArc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    fn sample_shard_data(n_rows: usize, n_vars: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for row in 0..n_rows {
            let col0 = (row * 2) % n_vars;
            let col1 = (row * 2 + 1) % n_vars;
            indices.push(col0 as u32);
            indices.push(col1 as u32);
            values.push(((row + 1) % 256) as u8);
            values.push(((row + 2) % 256) as u8);
            indptr.push(indptr.last().unwrap() + 2);
        }
        (indptr, indices, values)
    }

    /// Write a multi-shard test file.
    fn write_test_file(
        dir: &tempfile::TempDir,
        filename: &str,
        n_obs: usize,
        n_vars: usize,
        n_shards: usize,
    ) -> std::path::PathBuf {
        let path = dir.path().join(filename);
        let total_nnz = n_obs * 2;
        let header = sample_header(n_obs as u64, n_vars as u64, total_nnz as u64);
        let mut writer = ScxWriter::new(&path, header).unwrap();

        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();

        let rows_per_shard = n_obs / n_shards;
        for s in 0..n_shards {
            let shard_rows = if s == n_shards - 1 {
                n_obs - rows_per_shard * s
            } else {
                rows_per_shard
            };
            let (indptr, indices, values) = sample_shard_data(shard_rows, n_vars);
            writer
                .write_csr_shard(
                    &indptr,
                    &indices,
                    &values,
                    CodecId::None,
                    scx_codec::ValueEncoding::Uint8,
                    (s * rows_per_shard) as u64,
                )
                .unwrap();
        }

        writer.finish().unwrap();
        path
    }

    // -----------------------------------------------------------------------
    // A-series tests (LoaderConfig + MemoryBudget) — preserved
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_config_validates() {
        let config = LoaderConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_zero_batch_size_fails() {
        let config = LoaderConfig {
            batch_size: 0,
            ..LoaderConfig::default()
        };
        let err = config.validate().unwrap_err();
        match err {
            LoaderError::ConfigError { reason } => {
                assert!(reason.contains("batch_size"), "unexpected reason: {reason}");
            }
            _ => panic!("expected ConfigError, got: {err:?}"),
        }
    }

    #[test]
    fn test_zero_shard_group_size_fails() {
        let config = LoaderConfig {
            shard_group_size: 0,
            ..LoaderConfig::default()
        };
        let err = config.validate().unwrap_err();
        match err {
            LoaderError::ConfigError { reason } => {
                assert!(
                    reason.contains("shard_group_size"),
                    "unexpected reason: {reason}"
                );
            }
            _ => panic!("expected ConfigError, got: {err:?}"),
        }
    }

    #[test]
    fn test_zero_prefetch_batches_fails() {
        let config = LoaderConfig {
            prefetch_batches: 0,
            ..LoaderConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_zero_target_sum_fails() {
        let config = LoaderConfig {
            target_sum: 0.0,
            ..LoaderConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_negative_target_sum_fails() {
        let config = LoaderConfig {
            target_sum: -1.0,
            ..LoaderConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_small_memory_budget_fails() {
        let config = LoaderConfig {
            max_memory_mb: 32,
            ..LoaderConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_minimum_memory_budget_passes() {
        let config = LoaderConfig {
            max_memory_mb: 64,
            ..LoaderConfig::default()
        };
        assert!(config.validate().is_ok());
    }

    // --- Memory budget tests ---

    /// The four adaptive-budget tests below were written against
    /// `pipeline::adaptive_budget_mb`, the second copy of arithmetic
    /// `budget::adaptive_budget_mb` already owned. ORG-9.10-5 deleted the copy;
    /// this keeps the tests' call shape while routing them through the survivor,
    /// so what they assert is unchanged. `file_size_bytes` is gone from the
    /// signature because the sequential model no longer budgets mmap.
    fn adaptive_budget_mb(
        floor_mb: usize,
        shard_group_size: usize,
        prefetch_batches: usize,
        batch_size: usize,
        n_output_genes: usize,
        shard_target_rows: usize,
        avg_nnz_per_cell: f64,
    ) -> usize {
        let model = SequentialBudgetModel::new(n_output_genes, shard_target_rows, avg_nnz_per_cell);
        crate::budget::adaptive_budget_mb(
            model
                .estimate(SequentialParams {
                    shard_group_size,
                    prefetch_batches,
                    batch_size,
                })
                .total_bytes,
            floor_mb,
        )
    }

    #[test]
    fn test_memory_budget_2k_hvg_within_512mb() {
        let config = LoaderConfig {
            hvg_indices: Some((0..2000).collect()),
            ..LoaderConfig::default()
        };
        let budget = compute_memory_budget(&config, 30_000, 16_384, 10.0, 0);
        let budget_mb = budget.estimated_bytes / (1024 * 1024);
        assert!(
            budget_mb <= 512,
            "2K HVG budget {budget_mb} MB should be <= 512 MB"
        );
        assert_eq!(budget.shard_group_size, 8);
        assert_eq!(budget.prefetch_batches, 4);
        assert_eq!(budget.batch_size, 1024);
        assert!(!budget.budget_exceeded);
    }

    /// The budget must cost the panel's *deduplicated* width, because that is
    /// what the batch is allocated at (`HvgProjection::n_output_cols`). Costing
    /// the raw length over-reserves and can auto-tune `batch_size` down to
    /// afford memory the loader can never use.
    #[test]
    fn budget_costs_the_deduplicated_hvg_width() {
        let unique: Vec<u32> = (0..2000).collect();
        // Same 2000 genes, each written twice: identical batch, twice the len().
        let mut duplicated: Vec<u32> = unique.iter().chain(unique.iter()).copied().collect();
        duplicated.sort_unstable();

        let budget_of = |panel: Vec<u32>| {
            compute_memory_budget(
                &LoaderConfig {
                    hvg_indices: Some(panel),
                    ..LoaderConfig::default()
                },
                30_000,
                16_384,
                10.0,
                0,
            )
        };
        let plain = budget_of(unique);
        let dup = budget_of(duplicated);

        assert_eq!(
            dup.estimated_bytes, plain.estimated_bytes,
            "a duplicated panel projects to the same width, so it must cost the same"
        );
        assert_eq!(dup.batch_size, plain.batch_size);
        assert_eq!(dup.prefetch_batches, plain.prefetch_batches);
        assert_eq!(dup.shard_group_size, plain.shard_group_size);
    }

    /// The same claim where it bites: under a budget tight enough to force the
    /// auto-tune to act, an inflated width made it act *harder* than the real
    /// width needed.
    #[test]
    fn a_duplicated_panel_does_not_tighten_the_auto_tune() {
        let unique: Vec<u32> = (0..8000).collect();
        let mut duplicated: Vec<u32> = unique.iter().chain(unique.iter()).copied().collect();
        duplicated.sort_unstable();

        let budget_of = |panel: Vec<u32>| {
            compute_memory_budget(
                &LoaderConfig {
                    hvg_indices: Some(panel),
                    max_memory_mb: 128,
                    auto_memory_budget: false,
                    ..LoaderConfig::default()
                },
                30_000,
                16_384,
                10.0,
                0,
            )
        };
        let dup = budget_of(duplicated);
        let plain = budget_of(unique);
        // A budget this tight does reduce something — assert that first, or the
        // comparison below could pass by both arms being untouched.
        assert!(
            plain.prefetch_batches < 4 || plain.shard_group_size < 8 || plain.batch_size < 1024,
            "fixture must actually engage the auto-tune"
        );
        assert_eq!(
            (dup.shard_group_size, dup.prefetch_batches, dup.batch_size),
            (
                plain.shard_group_size,
                plain.prefetch_batches,
                plain.batch_size
            ),
            "the duplicated panel tuned to a different config than the identical batch needs"
        );
    }

    #[test]
    fn test_memory_budget_30k_genes_auto_tuned() {
        let config = LoaderConfig::default();
        let budget = compute_memory_budget(&config, 30_000, 16_384, 10.0, 0);
        assert!(budget.shard_group_size >= 1);
        assert!(budget.prefetch_batches >= 2);
        assert!(budget.batch_size >= 64);
        assert!(!budget.budget_exceeded);
    }

    #[test]
    fn test_memory_budget_128mb_reduced() {
        let config = LoaderConfig {
            max_memory_mb: 128,
            ..LoaderConfig::default()
        };
        let budget = compute_memory_budget(&config, 30_000, 16_384, 10.0, 0);
        assert!(
            budget.shard_group_size < 8 || budget.prefetch_batches < 4 || budget.batch_size < 1024,
            "128 MB budget should reduce at least one parameter: \
             shard_group_size={}, prefetch_batches={}, batch_size={}",
            budget.shard_group_size,
            budget.prefetch_batches,
            budget.batch_size
        );
    }

    #[test]
    fn test_memory_budget_64mb_all_minimums() {
        let config = LoaderConfig {
            max_memory_mb: 64,
            ..LoaderConfig::default()
        };
        let budget = compute_memory_budget(&config, 30_000, 16_384, 10.0, 0);
        assert_eq!(
            budget.shard_group_size, 1,
            "shard_group_size should be at minimum 1"
        );
        assert_eq!(
            budget.prefetch_batches, 2,
            "prefetch_batches should be at minimum 2"
        );
        assert!(budget.batch_size <= 1024, "batch_size should be reduced");
    }

    #[test]
    fn test_adaptive_budget_keeps_floor_for_small_file() {
        // A tiny file needs far less than the 512 MB floor → budget unchanged,
        // so small-file behaviour is identical to the fixed default.
        let mb = adaptive_budget_mb(
            512,  /* floor */
            1,    /* shard_group_size */
            4,    /* prefetch */
            64,   /* batch_size */
            100,  /* n_output_genes */
            1024, /* shard_target_rows */
            5.0,  /* avg_nnz_per_cell */
        );
        assert_eq!(mb, 512, "small file should keep the 512 MB floor");
    }

    #[test]
    fn test_adaptive_budget_raises_for_full_width_single_shard() {
        // The P3 scenario: ~33.5k-gene file in a single shard at batch_size=512.
        // The fixed 512 MB default cannot fit it (the decoded-shard cache alone
        // exceeds 512 MB), so the adaptive budget must raise above the floor —
        // and at the raised budget the auto-tune must NOT shrink batch_size or
        // flag budget_exceeded.
        let n_genes = 33_538usize;
        let shard_target_rows = 16_384usize;
        let avg_nnz = 2_246.0f64;
        let mb = adaptive_budget_mb(512, 1, 4, 512, n_genes, shard_target_rows, avg_nnz);
        assert!(
            mb > 512,
            "full-width file should raise above the 512 MB floor (got {mb})"
        );
        assert!(
            mb <= ADAPTIVE_BUDGET_CAP_MB,
            "must stay within the cap (got {mb})"
        );

        let config = LoaderConfig {
            batch_size: 512,
            shard_group_size: 1,
            max_memory_mb: mb,
            ..LoaderConfig::default()
        };
        let budget = compute_memory_budget(
            &config,
            n_genes as u64,
            shard_target_rows as u32,
            avg_nnz,
            /*file_size_bytes*/ 280 * 1024 * 1024,
        );
        assert_eq!(
            budget.batch_size, 512,
            "raised budget should preserve the requested batch_size"
        );
        assert!(
            !budget.budget_exceeded,
            "raised budget should fit without exceeding"
        );
        assert!(
            !budget.shuffle_quality_degraded,
            "single-shard group=1 should not be flagged as degraded"
        );
    }

    #[test]
    fn test_adaptive_budget_clamps_huge_file_to_cap() {
        // A fully dense huge shard needs many GB; the adaptive budget clamps to
        // the cap so we never silently reserve unbounded RAM — the hard-ceiling
        // auto-tune + warnings take over above the cap.
        let mb = adaptive_budget_mb(
            512, 8, 4, 1024, 33_538,   /* n_output_genes */
            16_384,   /* shard_target_rows */
            33_538.0, /* fully dense: avg_nnz == n_genes */
        );
        assert_eq!(
            mb, ADAPTIVE_BUDGET_CAP_MB,
            "huge need should clamp to the cap"
        );
    }

    #[test]
    fn test_adaptive_budget_floor_above_cap_does_not_panic() {
        // PR #247 review: a floor above ADAPTIVE_BUDGET_CAP_MB must not panic the
        // `clamp` (min > max). The floor wins as the effective budget.
        let mb = adaptive_budget_mb(
            ADAPTIVE_BUDGET_CAP_MB + 4096, /* floor far above the cap */
            8,
            4,
            1024,
            30_000,
            16_384,
            10.0,
        );
        assert!(
            mb >= ADAPTIVE_BUDGET_CAP_MB + 4096,
            "floor above the cap must be honored (no panic), got {mb}"
        );
    }

    #[test]
    fn test_shuffle_quality_degraded_under_tight_budget() {
        // 64 MB forces shard_group_size to 1 (below the quality threshold and
        // below the requested default of 8) → degraded flag set.
        let config = LoaderConfig {
            max_memory_mb: 64,
            ..LoaderConfig::default()
        };
        let budget = compute_memory_budget(&config, 30_000, 16_384, 10.0, 0);
        assert!(budget.shard_group_size < MIN_SHUFFLE_QUALITY_SHARD_GROUP_SIZE);
        assert!(
            budget.shuffle_quality_degraded,
            "tight budget that shrinks shard_group_size to {} (< {}) should flag \
             degraded shuffle quality",
            budget.shard_group_size, MIN_SHUFFLE_QUALITY_SHARD_GROUP_SIZE
        );
    }

    #[test]
    fn test_shuffle_quality_not_degraded_with_room() {
        // A generous budget keeps shard_group_size at the requested default,
        // so shuffle quality is not flagged.
        let config = LoaderConfig {
            max_memory_mb: 4096,
            ..LoaderConfig::default()
        };
        let budget = compute_memory_budget(&config, 2_000, 16_384, 10.0, 0);
        assert_eq!(budget.shard_group_size, config.shard_group_size);
        assert!(!budget.shuffle_quality_degraded);
    }

    #[test]
    fn test_memory_budget_61k_genes_reduces_batch_size() {
        // 61K genes without HVG projection — batch_size must be reduced
        let config = LoaderConfig::default(); // 512 MB budget, no HVG
        let budget = compute_memory_budget(&config, 61_497, 16_384, 10.0, 0);
        assert!(
            budget.batch_size < 1024,
            "61K genes should reduce batch_size from 1024, got {}",
            budget.batch_size
        );
        assert!(budget.batch_size >= 64, "batch_size should not go below 64");
        assert!(
            !budget.budget_exceeded,
            "512 MB budget should be achievable with reduced batch_size"
        );
    }

    #[test]
    fn test_memory_budget_61k_genes_tiny_budget_exceeded() {
        // 61K genes with 64 MB budget — impossible to fit
        let config = LoaderConfig {
            max_memory_mb: 64,
            ..LoaderConfig::default()
        };
        let budget = compute_memory_budget(&config, 61_497, 16_384, 10.0, 0);
        assert_eq!(budget.batch_size, 64);
        assert_eq!(budget.shard_group_size, 1);
        assert_eq!(budget.prefetch_batches, 2);
        assert!(
            budget.budget_exceeded,
            "64 MB budget with 61K genes should exceed budget"
        );
    }

    #[test]
    fn test_memory_budget_batch_size_halving() {
        // Verify batch_size halves (not decrements by 1)
        let config = LoaderConfig {
            max_memory_mb: 200,
            ..LoaderConfig::default()
        };
        let budget = compute_memory_budget(&config, 61_497, 16_384, 10.0, 0);
        // batch_size should be a power-of-2 fraction of 1024
        assert!(
            budget.batch_size == 64
                || budget.batch_size == 128
                || budget.batch_size == 256
                || budget.batch_size == 512,
            "batch_size should be a power-of-2 reduction of 1024, got {}",
            budget.batch_size
        );
    }

    /// **Inverted by ORG-9.10-5.** This test previously asserted the opposite —
    /// that a 2.5 GB file raised `estimated_bytes` by roughly the file size —
    /// because the sequential model counted mmap-resident pages against the
    /// budget and the two plan-driven models did not.
    ///
    /// `mmap_bytes` is still *reported*, because the pages are real and a
    /// caller reading `ru_maxrss` will see them. It is no longer *budgeted*:
    /// the kernel page cache is evictable under pressure, and counting it made
    /// every file above `ADAPTIVE_BUDGET_CAP_MB` exceed its budget on that term
    /// alone.
    #[test]
    fn mmap_is_reported_but_not_budgeted() {
        let config = LoaderConfig {
            hvg_indices: Some((0..2000).collect()),
            ..LoaderConfig::default()
        };
        let file_2gb = 2_500 * 1024 * 1024; // 2.5 GB
        let no_file = compute_memory_budget(&config, 30_000, 16_384, 10.0, 0);
        let large = compute_memory_budget(&config, 30_000, 16_384, 10.0, file_2gb);

        assert_eq!(
            large.mmap_bytes, file_2gb,
            "the file size is still reported"
        );
        assert_eq!(no_file.mmap_bytes, 0);
        assert_eq!(
            large.estimated_bytes, no_file.estimated_bytes,
            "the estimate must not move with the file size"
        );
        assert_eq!(
            large.estimated_bytes, large.breakdown.total_bytes,
            "`estimated_bytes` and the breakdown's total are now the same number \
             on every loader class"
        );
    }

    /// **Inverted by ORG-9.10-5**, and the reason the change is worth making:
    /// a 2.5 GB file used to force `batch_size` / `shard_group_size` /
    /// `prefetch_batches` down purely because of its size.
    /// `benchmarks/comprehensive/benchmarks/ml_loader.py` carried a
    /// SLURM-memory-scaling workaround for exactly this collapse.
    #[test]
    fn a_large_file_no_longer_forces_a_reduction() {
        let config = LoaderConfig {
            hvg_indices: Some((0..2000).collect()),
            ..LoaderConfig::default()
        };
        let file_2gb = 2_500 * 1024 * 1024;
        let budget = compute_memory_budget(&config, 30_000, 16_384, 10.0, file_2gb);
        assert_eq!(
            (
                budget.shard_group_size,
                budget.prefetch_batches,
                budget.batch_size
            ),
            (
                config.shard_group_size,
                config.prefetch_batches,
                config.batch_size
            ),
            "a large file must not shrink a config that fits in anonymous memory"
        );
        assert!(!budget.budget_exceeded);
        // The premise: this config genuinely fits the 512 MB default, so the
        // old failure really was the mmap term and nothing else.
        assert!(budget.breakdown.fits_within(config.max_memory_mb));
    }

    /// `pin_effective_config` must refuse to **raise** a knob.
    ///
    /// This is what replaced §9.6's `debug_assert!` and makes
    /// `MultimodalTrainingDataset`'s uniform pin structural rather than
    /// assumed: "the pinned config always fits" holds because the pin only ever
    /// shrinks and the model is monotone. A regression that allowed a raise
    /// would not fail the Python-level test, which only observes the happy path.
    #[test]
    fn pin_effective_config_refuses_to_raise_a_knob() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "pin.scx", 64, 20, 4);
        let config = LoaderConfig {
            batch_size: 16,
            shard_group_size: 2,
            normalize: false,
            log1p: false,
            ..LoaderConfig::default()
        };
        let mut p = TrainingPipeline::new(&path, config).unwrap();
        let (bs, sgs) = {
            let mb = p.memory_budget_info();
            (mb.batch_size, mb.shard_group_size)
        };

        // Shrinking is fine, and is what the multimodal pin does.
        p.pin_effective_config(bs / 2, sgs)
            .expect("shrinking must be allowed");
        assert_eq!(p.memory_budget_info().batch_size, bs / 2);
        assert_eq!(p.effective_batch_size(), bs / 2);

        // Raising either knob is an error, not a silent no-op.
        let err = p.pin_effective_config(bs, sgs).unwrap_err();
        assert!(
            err.to_string().contains("may only shrink"),
            "unexpected error: {err}"
        );
        let err = p.pin_effective_config(bs / 2, sgs + 1).unwrap_err();
        assert!(
            err.to_string().contains("may only shrink"),
            "unexpected error: {err}"
        );
    }

    /// Pinning mid-epoch would swap the shuffler out from under a live
    /// iteration, so it is refused rather than silently reordering rows.
    #[test]
    fn pin_effective_config_refuses_mid_epoch() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "pin_epoch.scx", 64, 20, 4);
        let config = LoaderConfig {
            batch_size: 16,
            normalize: false,
            log1p: false,
            ..LoaderConfig::default()
        };
        let mut p = TrainingPipeline::new(&path, config).unwrap();
        p.start_epoch().unwrap();
        let err = p.pin_effective_config(8, 1).unwrap_err();
        assert!(
            err.to_string().contains("epoch in flight"),
            "unexpected error: {err}"
        );
        p.shutdown();
    }

    // ---- ORG-9.10-5 pre-refactor pins ------------------------------------

    /// The reduction **order**, not merely that something was reduced.
    ///
    /// `test_memory_budget_128mb_reduced` asserts only that *some* knob moved
    /// and `test_memory_budget_64mb_all_minimums` only pins the terminal state,
    /// so a loop that halved `batch_size` before touching `prefetch_batches`
    /// would pass both — and `batch_size` is the one knob a caller picks for
    /// statistical rather than memory reasons, so it must give last.
    #[test]
    fn budget_reduces_prefetch_then_shard_group_then_batch() {
        let base = LoaderConfig::default();
        let (req_sgs, req_pf, req_bs) = (
            base.shard_group_size,
            base.prefetch_batches,
            base.batch_size,
        );
        assert_eq!(
            (req_sgs, req_pf, req_bs),
            (8, 4, 1024),
            "premise: the defaults this test reasons about"
        );

        let mut saw_prefetch_alone = false;
        let mut saw_group_without_batch = false;

        for mb in (64..=2048).step_by(8) {
            let b = compute_memory_budget(
                &LoaderConfig {
                    max_memory_mb: mb,
                    ..base.clone()
                },
                30_000,
                16_384,
                10.0,
                0,
            );

            if b.shard_group_size < req_sgs {
                assert_eq!(
                    b.prefetch_batches, 2,
                    "at {mb} MB shard_group_size fell to {} while prefetch_batches was \
                     still {} — prefetch must reach its floor first",
                    b.shard_group_size, b.prefetch_batches
                );
            }
            if b.batch_size < req_bs {
                assert_eq!(
                    (b.prefetch_batches, b.shard_group_size),
                    (2, 1),
                    "at {mb} MB batch_size fell to {} before prefetch_batches and \
                     shard_group_size reached their floors (got {}, {})",
                    b.batch_size,
                    b.prefetch_batches,
                    b.shard_group_size
                );
            }

            if b.prefetch_batches < req_pf
                && b.shard_group_size == req_sgs
                && b.batch_size == req_bs
            {
                saw_prefetch_alone = true;
            }
            if b.shard_group_size < req_sgs && b.batch_size == req_bs {
                saw_group_without_batch = true;
            }
        }

        // Without these the ordering assertions above are vacuously true.
        assert!(
            saw_prefetch_alone,
            "no budget in the sweep reduced prefetch_batches alone"
        );
        assert!(
            saw_group_without_batch,
            "no budget in the sweep reduced shard_group_size while batch_size survived"
        );
    }

    /// The whole reduction chain, through the shared harness — the same check
    /// `crate::budget::tune` `debug_assert`s per step, run end to end with no
    /// file and no fixture.
    #[test]
    fn the_sequential_reduction_chain_is_monotone() {
        for &(n_genes, rows, nnz) in &[
            (30_000usize, 16_384usize, 10.0f64),
            (2_000, 4_096, 3.0),
            (61_497, 16_384, 40.0),
        ] {
            crate::budget::assert_monotone_reduction_chain(
                &SequentialBudgetModel::new(n_genes, rows, nnz),
                SequentialParams {
                    shard_group_size: 8,
                    prefetch_batches: 4,
                    batch_size: 1024,
                },
            );
        }
    }

    /// `estimate_memory` must be monotone non-increasing in every knob the
    /// auto-tune reduces.
    ///
    /// This is the property `python.rs`'s multimodal guard rests on — "a
    /// smaller pinned config always fits within the same budget" — and which
    /// nothing asserted. `MultimodalTrainingDataset` pins every modality to the
    /// **minimum** effective `(batch_size, shard_group_size)` across
    /// modalities; a term that grew as a knob shrank would desync per-modality
    /// batching and surface mid-epoch as a `RuntimeError` blaming the file's
    /// sharding, which is the wrong diagnosis.
    #[test]
    fn estimate_is_monotone_in_every_tuned_knob() {
        for &(n_genes, rows, nnz) in &[
            (30_000usize, 16_384usize, 10.0f64),
            (2_000, 4_096, 3.0),
            (61_497, 16_384, 40.0),
        ] {
            let model = SequentialBudgetModel::new(n_genes, rows, nnz);
            let est = |sgs: usize, pf: usize, bs: usize| {
                model
                    .estimate(SequentialParams {
                        shard_group_size: sgs,
                        prefetch_batches: pf,
                        batch_size: bs,
                    })
                    .total_bytes
            };

            for sgs in (1..64usize).rev() {
                assert!(
                    est(sgs, 4, 1024) <= est(sgs + 1, 4, 1024),
                    "shard_group_size {sgs} estimates more than {} at {n_genes} genes",
                    sgs + 1
                );
            }
            for pf in (2..64usize).rev() {
                assert!(
                    est(8, pf, 1024) <= est(8, pf + 1, 1024),
                    "prefetch_batches {pf} estimates more than {} at {n_genes} genes",
                    pf + 1
                );
            }
            let mut bs = 1024usize;
            while bs > 64 {
                let half = bs / 2;
                assert!(
                    est(8, 4, half) <= est(8, 4, bs),
                    "batch_size {half} estimates more than {bs} at {n_genes} genes"
                );
                bs = half;
            }

            // The joint reduction the multimodal repin actually performs: both
            // knobs drop to the cross-modality minimum at once.
            assert!(
                est(1, 4, 64) <= est(8, 4, 1024),
                "the pinned (batch_size, shard_group_size) minimum estimates more \
                 than the requested config at {n_genes} genes"
            );
        }
    }

    /// A budget the tuner accepted must actually hold the breakdown it reports.
    ///
    /// `BudgetBreakdown::fits_within` exists and no loop uses it as its
    /// terminator; this pins the postcondition on the sequential path, matching
    /// `test_index_plan_dataset.py::test_memory_budget_matches_max_memory_mb`
    /// on the plan-driven one.
    #[test]
    fn a_budget_that_was_not_exceeded_fits_the_breakdown_it_reports() {
        let mut saw_tuned = false;
        for mb in (64..=2048).step_by(8) {
            let b = compute_memory_budget(
                &LoaderConfig {
                    max_memory_mb: mb,
                    ..LoaderConfig::default()
                },
                30_000,
                16_384,
                10.0,
                0,
            );
            if b.budget_exceeded {
                continue;
            }
            assert!(
                b.breakdown.fits_within(mb),
                "budget {mb} MB reported not-exceeded but its breakdown totals {} bytes",
                b.breakdown.total_bytes
            );
            if b.batch_size < 1024 || b.shard_group_size < 8 || b.prefetch_batches < 4 {
                saw_tuned = true;
            }
        }
        assert!(
            saw_tuned,
            "the sweep never engaged the auto-tune, so this proves nothing"
        );
    }

    // -----------------------------------------------------------------------
    // E1 Tests: TrainingPipeline
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_new_opens_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "test.scx", 30, 10, 3);

        let config = LoaderConfig {
            batch_size: 10,
            normalize: false,
            log1p: false,
            ..LoaderConfig::default()
        };

        let pipeline = TrainingPipeline::new(&path, config).unwrap();
        assert_eq!(pipeline.n_obs(), 30);
        assert_eq!(pipeline.n_vars(), 10);
        assert_eq!(pipeline.n_output_genes(), 10);
    }

    #[test]
    fn test_pipeline_all_cells_once() {
        let dir = tempfile::tempdir().unwrap();
        let n_obs = 30;
        let n_vars = 10;
        let path = write_test_file(&dir, "test.scx", n_obs, n_vars, 3);

        let config = LoaderConfig {
            batch_size: 10,
            normalize: false,
            log1p: false,
            ..LoaderConfig::default()
        };

        let mut pipeline = TrainingPipeline::new(&path, config).unwrap();
        pipeline.start_epoch().unwrap();

        let mut all_cells: Vec<u64> = Vec::new();
        while let Some(batch) = pipeline.next_batch().unwrap() {
            assert_eq!(batch.x_shape.1, n_vars);
            assert_eq!(batch.x.len(), batch.x_shape.0 * batch.x_shape.1);
            all_cells.extend_from_slice(&batch.cell_indices);
        }

        // All cells appear exactly once
        let cell_set: std::collections::HashSet<u64> = all_cells.iter().copied().collect();
        assert_eq!(cell_set.len(), n_obs, "all cells should be unique");
        assert_eq!(
            all_cells.len(),
            n_obs,
            "all cells should appear exactly once"
        );
        for i in 0..n_obs as u64 {
            assert!(cell_set.contains(&i), "cell {i} missing");
        }
    }

    #[test]
    fn test_pipeline_none_at_end() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "test.scx", 10, 5, 1);

        let config = LoaderConfig {
            batch_size: 100,
            normalize: false,
            log1p: false,
            ..LoaderConfig::default()
        };

        let mut pipeline = TrainingPipeline::new(&path, config).unwrap();
        pipeline.start_epoch().unwrap();

        // First batch should have all cells
        let batch = pipeline.next_batch().unwrap().unwrap();
        assert_eq!(batch.cell_indices.len(), 10);

        // Second call should return None
        assert!(pipeline.next_batch().unwrap().is_none());
    }

    /// L1 regression: a mid-epoch I/O/decode fault must surface as `Err` from
    /// `next_batch`, not be swallowed as a clean epoch end (which would train
    /// a truncated epoch with no exception).
    ///
    /// Drives the exact `rx.recv() == Err` → `join_epoch_handles` →
    /// propagate-error path with a hand-installed faulting decode handle, so
    /// the test is deterministic and needs no corrupt on-disk fixture.
    #[test]
    fn test_next_batch_surfaces_mid_epoch_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "test.scx", 10, 5, 1);

        let config = LoaderConfig {
            batch_size: 100,
            normalize: false,
            log1p: false,
            ..LoaderConfig::default()
        };

        let mut pipeline = TrainingPipeline::new(&path, config).unwrap();

        // Hand-install a faulting epoch: a decode handle that returns an error
        // (any variant other than ChannelError, which join_epoch_handles treats
        // as a clean cancellation) and immediately drops its sender so the
        // consumer's recv() sees the channel close.
        let (batch_tx, batch_rx) = crossbeam_channel::bounded::<Batch>(1);
        let decode_handle = std::thread::spawn(move || {
            drop(batch_tx);
            Err(LoaderError::ConfigError {
                reason: "injected decode fault".to_string(),
            })
        });
        pipeline.batch_rx = Some(batch_rx);
        pipeline.decode_handle = Some(decode_handle);
        pipeline.io_handle = None;

        let result = pipeline.next_batch();
        assert!(
            result.is_err(),
            "mid-epoch decode fault must surface as Err, got {result:?}"
        );
        // The epoch is torn down; the error message carries the stage context.
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("decode stage error") && msg.contains("injected decode fault"),
            "unexpected error message: {msg}"
        );
        assert!(
            pipeline.batch_rx.is_none(),
            "the epoch must be torn down after a fault"
        );
    }

    #[test]
    fn test_pipeline_different_shuffle_second_epoch() {
        let dir = tempfile::tempdir().unwrap();
        let n_obs = 20;
        let path = write_test_file(&dir, "test.scx", n_obs, 10, 2);

        let config = LoaderConfig {
            batch_size: 100,
            normalize: false,
            log1p: false,
            ..LoaderConfig::default()
        };

        let mut pipeline = TrainingPipeline::new(&path, config).unwrap();

        // Epoch 0
        pipeline.start_epoch().unwrap();
        let mut epoch0_cells = Vec::new();
        while let Some(batch) = pipeline.next_batch().unwrap() {
            epoch0_cells.extend_from_slice(&batch.cell_indices);
        }

        // Epoch 1
        pipeline.start_epoch().unwrap();
        let mut epoch1_cells = Vec::new();
        while let Some(batch) = pipeline.next_batch().unwrap() {
            epoch1_cells.extend_from_slice(&batch.cell_indices);
        }

        // Both epochs should have all cells
        let set0: std::collections::HashSet<u64> = epoch0_cells.iter().copied().collect();
        let set1: std::collections::HashSet<u64> = epoch1_cells.iter().copied().collect();
        assert_eq!(set0.len(), n_obs);
        assert_eq!(set1.len(), n_obs);

        // Order should differ (shuffle)
        assert_ne!(
            epoch0_cells, epoch1_cells,
            "different epochs should produce different orderings"
        );
    }

    #[test]
    fn test_pipeline_next_batch_before_start() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "test.scx", 10, 5, 1);

        let config = LoaderConfig {
            batch_size: 10,
            normalize: false,
            log1p: false,
            ..LoaderConfig::default()
        };

        let mut pipeline = TrainingPipeline::new(&path, config).unwrap();
        // next_batch before start_epoch should return None
        assert!(pipeline.next_batch().unwrap().is_none());
    }

    #[test]
    fn test_pipeline_drop_during_active_epoch() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "test.scx", 30, 10, 3);

        let config = LoaderConfig {
            batch_size: 5,
            normalize: false,
            log1p: false,
            ..LoaderConfig::default()
        };

        let mut pipeline = TrainingPipeline::new(&path, config).unwrap();
        pipeline.start_epoch().unwrap();

        // Consume just one batch, then drop
        let _batch = pipeline.next_batch();
        drop(pipeline);
        // If we get here without panic/deadlock, the test passes
    }

    // -----------------------------------------------------------------------
    // E2 Tests: Multi-epoch lifecycle
    // -----------------------------------------------------------------------

    #[test]
    fn test_three_complete_epochs() {
        let dir = tempfile::tempdir().unwrap();
        let n_obs = 20;
        let path = write_test_file(&dir, "test.scx", n_obs, 10, 2);

        let config = LoaderConfig {
            batch_size: 100,
            normalize: false,
            log1p: false,
            ..LoaderConfig::default()
        };

        let mut pipeline = TrainingPipeline::new(&path, config).unwrap();

        for epoch in 0..3 {
            pipeline.start_epoch().unwrap();
            let mut cells = Vec::new();
            while let Some(batch) = pipeline.next_batch().unwrap() {
                cells.extend_from_slice(&batch.cell_indices);
            }
            let set: std::collections::HashSet<u64> = cells.iter().copied().collect();
            assert_eq!(set.len(), n_obs, "epoch {epoch}: all cells should appear");
            assert_eq!(
                cells.len(),
                n_obs,
                "epoch {epoch}: each cell should appear exactly once"
            );
        }
    }

    #[test]
    fn test_epochs_different_order() {
        let dir = tempfile::tempdir().unwrap();
        let n_obs = 20;
        let path = write_test_file(&dir, "test.scx", n_obs, 10, 4);

        let config = LoaderConfig {
            batch_size: 100,
            normalize: false,
            log1p: false,
            ..LoaderConfig::default()
        };

        let mut pipeline = TrainingPipeline::new(&path, config).unwrap();

        let mut all_epoch_cells = Vec::new();
        for _ in 0..3 {
            pipeline.start_epoch().unwrap();
            let mut cells = Vec::new();
            while let Some(batch) = pipeline.next_batch().unwrap() {
                cells.extend_from_slice(&batch.cell_indices);
            }
            all_epoch_cells.push(cells);
        }

        // At least two of the three epochs should differ in order
        let any_differ = all_epoch_cells[0] != all_epoch_cells[1]
            || all_epoch_cells[1] != all_epoch_cells[2]
            || all_epoch_cells[0] != all_epoch_cells[2];
        assert!(any_differ, "epochs should have different orderings");
    }

    #[test]
    fn test_same_seed_reproducible() {
        let dir = tempfile::tempdir().unwrap();
        let n_obs = 20;
        let path = write_test_file(&dir, "test.scx", n_obs, 10, 2);

        let config = LoaderConfig {
            batch_size: 100,
            seed: 12345,
            normalize: false,
            log1p: false,
            ..LoaderConfig::default()
        };

        // First pipeline run
        let mut pipeline1 = TrainingPipeline::new(&path, config.clone()).unwrap();
        pipeline1.start_epoch().unwrap();
        let mut cells1 = Vec::new();
        while let Some(batch) = pipeline1.next_batch().unwrap() {
            cells1.extend_from_slice(&batch.cell_indices);
        }
        drop(pipeline1);

        // Second pipeline run with same seed
        let mut pipeline2 = TrainingPipeline::new(&path, config).unwrap();
        pipeline2.start_epoch().unwrap();
        let mut cells2 = Vec::new();
        while let Some(batch) = pipeline2.next_batch().unwrap() {
            cells2.extend_from_slice(&batch.cell_indices);
        }

        assert_eq!(
            cells1, cells2,
            "same seed should produce identical epoch 0 sequence"
        );
    }

    #[test]
    fn test_mid_epoch_drop_no_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "test.scx", 50, 10, 5);

        let config = LoaderConfig {
            batch_size: 5,
            normalize: false,
            log1p: false,
            ..LoaderConfig::default()
        };

        // Create, start, consume a few batches, then drop mid-epoch
        let mut pipeline = TrainingPipeline::new(&path, config).unwrap();
        pipeline.start_epoch().unwrap();

        // Consume 2 batches (out of ~10)
        let _ = pipeline.next_batch();
        let _ = pipeline.next_batch();

        // Drop — must not panic or deadlock
        drop(pipeline);
    }
    // -----------------------------------------------------------------------
    // ORG-9.10-6 / §9.7 — what a mid-epoch `break` does and does not release.
    //
    // The review asks for a test that "a mid-epoch `break` releases the ring".
    // It does not, and §9.7 is still open: nothing observes that the consumer
    // stopped, so `batch_rx` stays `Some`, the decode thread stays parked in
    // `tx.send` and the I/O thread behind it, holding the full high-water
    // allocation for as long as the object lives. These two tests pin what is
    // actually true on both sides of that line, so the §9.7 fix has a stated
    // baseline to invert rather than a blank page.
    // -----------------------------------------------------------------------

    /// `shutdown()` is the escape: it drops the batch receiver, joins both
    /// worker threads within the bounded deadline, and releases the rayon pool.
    #[test]
    fn shutdown_after_a_mid_epoch_break_releases_the_ring() {
        let dir = tempfile::tempdir().unwrap();
        // 40 shards at `shard_group_size = 2` → 20 shard groups against an
        // io→decode channel of capacity 2, so the I/O worker is genuinely
        // back-pressured rather than finished. With the 5-shard/`sgs = 8`
        // fixtures used elsewhere in this module there is exactly one group and
        // the I/O thread exits immediately — which is why §9.7's "both worker
        // threads" claim needs a fixture that can actually show it.
        let path = write_test_file(&dir, "test.scx", 400, 10, 40);

        let config = LoaderConfig {
            batch_size: 5,
            shard_group_size: 2,
            normalize: false,
            log1p: false,
            ..LoaderConfig::default()
        };
        let mut pipeline = TrainingPipeline::new(&path, config).unwrap();
        pipeline.start_epoch().unwrap();

        // Two of 80 batches, then stop consuming — the Python `break`.
        let _ = pipeline.next_batch().unwrap().expect("first batch");
        let _ = pipeline.next_batch().unwrap().expect("second batch");
        assert!(
            pipeline.decode_handle.is_some(),
            "premise: the epoch must still be live before shutdown"
        );

        pipeline.shutdown();

        assert!(
            pipeline.batch_rx.is_none(),
            "shutdown must drop the batch receiver — that is what unblocks the \
             decode stage's parked `tx.send`"
        );
        assert!(
            pipeline.io_handle.is_none() && pipeline.decode_handle.is_none(),
            "shutdown must join (and clear) both worker handles"
        );
        assert!(
            pipeline.decode_pool.is_none(),
            "shutdown must release the per-pipeline rayon pool"
        );

        // Idempotent, per its doc comment.
        pipeline.shutdown();
    }

    /// **§9.7, pinned as it stands.** A `break` on its own releases nothing:
    /// the ring, both worker threads and the rayon pool are all still held.
    ///
    /// This test states the defect, not the desired behaviour. **When §9.7 is
    /// fixed — by splitting the epoch iterator out of the dataset so dropping
    /// the iterator drops the epoch — this test is expected to fail, and the
    /// fix should invert it rather than delete it.**
    #[test]
    fn a_break_alone_does_not_release_the_ring() {
        let dir = tempfile::tempdir().unwrap();
        // 40 shards at `shard_group_size = 2` → 20 shard groups against an
        // io→decode channel of capacity 2, so the I/O worker is genuinely
        // back-pressured rather than finished. With the 5-shard/`sgs = 8`
        // fixtures used elsewhere in this module there is exactly one group and
        // the I/O thread exits immediately — which is why §9.7's "both worker
        // threads" claim needs a fixture that can actually show it.
        let path = write_test_file(&dir, "test.scx", 400, 10, 40);

        let config = LoaderConfig {
            batch_size: 5,
            shard_group_size: 2,
            normalize: false,
            log1p: false,
            ..LoaderConfig::default()
        };
        let mut pipeline = TrainingPipeline::new(&path, config).unwrap();
        pipeline.start_epoch().unwrap();

        let _ = pipeline.next_batch().unwrap().expect("first batch");
        let _ = pipeline.next_batch().unwrap().expect("second batch");
        // ... and now the consumer walks away.

        assert!(
            pipeline.batch_rx.is_some(),
            "§9.7: the batch ring is still held after a mid-epoch break"
        );
        assert!(
            pipeline.decode_pool.is_some(),
            "§9.7: the rayon pool is still held after a mid-epoch break"
        );

        let decode = pipeline
            .decode_handle
            .as_ref()
            .expect("§9.7: the decode worker handle is still held");
        let io = pipeline
            .io_handle
            .as_ref()
            .expect("§9.7: the I/O worker handle is still held");
        assert!(
            !decode.is_finished(),
            "§9.7: the decode worker is still alive — parked in `tx.send` against \
             a ring nobody is draining, not exited"
        );
        assert!(
            !io.is_finished(),
            "§9.7: the I/O worker is still alive, back-pressured behind the decode \
             stage"
        );
    }
}
