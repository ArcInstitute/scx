use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::record_batch::RecordBatch;
use scx_format_io::deletion_vectors::DeletionVectors;
use scx_format_io::reader::ScxReader;

use crate::batch::Batch;
use crate::budget::{profiling_enabled, BudgetBreakdown, PYTHON_OVERHEAD_BYTES};
use crate::decode_stage::decode_stage;
use crate::error::{LoaderError, Result};
use crate::io_stage::io_stage;
use crate::projection::HvgProjection;
use crate::shuffle::ShardShuffler;

/// Hard upper bound on per-pipeline rayon worker threads. On many-core hosts
/// the decode work is embarrassingly parallel but memory-bound; more than
/// ~8 workers does not pay off and increases the fork-hostile thread count
/// for downstream callers that use spawn-mode multiprocessing.
const DEFAULT_DECODE_POOL_MAX_THREADS: usize = 8;

/// Bounded join deadline for `Drop` and `join_epoch_handles` shutdown.
/// If the I/O or decode thread does not finish within this window the join
/// is abandoned and the thread handle is detached — preferable to wedging
/// the worker process exit forever.
const SHUTDOWN_DEADLINE: Duration = Duration::from_secs(5);

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
    /// Apply PFlog1pPF / shifted-CLR normalization (Booeshaghi et al. 2026)
    /// instead of normalize/log1p (default: false).
    ///
    /// PFlog1pPF is itself a normalization, so it is **mutually exclusive**
    /// with `normalize`/`log1p`: when `true` it takes precedence and those
    /// flags are ignored. Per cell it computes
    /// `z_ij = log1p(x_ij/(c·s_i)) − (1/D)·Σ_k log1p(x_ik/(c·s_i))`, where the
    /// depth `s_i` and centering denominator `D` are over the **full
    /// transcriptome** (not the HVG-projected panel — see
    /// [`crate::projection::HvgProjection::scatter_pflog1ppf_row`]).
    pub pflog1ppf: bool,
    /// PFlog1pPF shift / pseudocount `c` (default: 1.0; only used when
    /// `pflog1ppf` is true).
    pub pflog1ppf_c: f64,
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
    /// per-modality. Deletion vectors are not supported in conjunction
    /// with this filter (the per-shard bitmap keys reference the global
    /// shard index, not the per-modality index).
    pub modality_id: Option<u8>,
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
            pflog1ppf: false,
            pflog1ppf_c: 1.0,
            seed: 42,
            max_memory_mb: 512,
            auto_memory_budget: false,
            modality_id: None,
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
        if self.pflog1ppf
            && (self.pflog1ppf_c <= 0.0
                || self.pflog1ppf_c.is_nan()
                || self.pflog1ppf_c.is_infinite())
        {
            return Err(LoaderError::ConfigError {
                reason: "pflog1ppf_c must be positive and finite".to_string(),
            });
        }
        if self.pflog1ppf && (self.normalize || self.log1p) {
            log::warn!(
                "pflog1ppf=true takes precedence; normalize/log1p flags are ignored \
                 (pflog1ppf is itself a normalization)"
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
    /// Estimated total memory in bytes (includes mmap file size).
    pub estimated_bytes: usize,
    /// Size of the mmap'd SCX file in bytes (included in estimated_bytes).
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

/// Compute the memory budget for the training pipeline.
///
/// Memory model:
/// ```text
/// n_output_genes     = hvg_indices.len() if present, else n_vars
/// shard_buffer       = (shard_group_size + 1) × decoded_shard_bytes
/// batch_buffer       = (max(prefetch_batches, 2) + 1) × batch_size × n_output_genes × 4
/// mmap_resident      = file_size_bytes (entire file faulted into RSS during epoch)
/// overhead           = ~50 MB (Python interpreter, numpy, Arrow, thread stacks)
/// ```
///
/// If total exceeds `max_memory_mb`, reduces parameters in order:
/// 1. `prefetch_batches` (minimum 2)
/// 2. `shard_group_size` (minimum 1)
/// 3. `batch_size` (halve each step, minimum 64)
pub fn compute_memory_budget(
    config: &LoaderConfig,
    n_vars: u64,
    shard_target_rows: u32,
    avg_nnz_per_cell: f64,
    file_size_bytes: usize,
) -> MemoryBudget {
    let n_output_genes = match &config.hvg_indices {
        Some(hvg) => hvg.len(),
        None => n_vars as usize,
    };

    let max_bytes = config.max_memory_mb * 1024 * 1024;

    let mut shard_group_size = config.shard_group_size;
    let mut prefetch_batches = config.prefetch_batches;
    let mut batch_size = config.batch_size;

    loop {
        let estimated = estimate_memory(
            shard_group_size,
            prefetch_batches,
            batch_size,
            n_output_genes,
            shard_target_rows as usize,
            avg_nnz_per_cell,
            file_size_bytes,
        );

        if estimated <= max_bytes {
            let breakdown = estimate_breakdown(
                shard_group_size,
                prefetch_batches,
                batch_size,
                n_output_genes,
                shard_target_rows as usize,
                avg_nnz_per_cell,
            );
            return MemoryBudget {
                shard_group_size,
                prefetch_batches,
                batch_size,
                estimated_bytes: estimated,
                mmap_bytes: file_size_bytes,
                budget_exceeded: false,
                breakdown,
                shuffle_quality_degraded: shuffle_quality_degraded(
                    config.shard_group_size,
                    shard_group_size,
                ),
            };
        }

        // Reduce prefetch_batches first (to minimum 2)
        if prefetch_batches > 2 {
            prefetch_batches -= 1;
            continue;
        }

        // Then reduce shard_group_size (to minimum 1)
        if shard_group_size > 1 {
            shard_group_size -= 1;
            continue;
        }

        // Then halve batch_size (to minimum 64)
        if batch_size > 64 {
            batch_size = (batch_size / 2).max(64);
            continue;
        }

        // All at minimums — return best-effort estimate with warning
        let breakdown = estimate_breakdown(
            shard_group_size,
            prefetch_batches,
            batch_size,
            n_output_genes,
            shard_target_rows as usize,
            avg_nnz_per_cell,
        );
        return MemoryBudget {
            shard_group_size,
            prefetch_batches,
            batch_size,
            estimated_bytes: estimated,
            mmap_bytes: file_size_bytes,
            budget_exceeded: true,
            breakdown,
            shuffle_quality_degraded: shuffle_quality_degraded(
                config.shard_group_size,
                shard_group_size,
            ),
        };
    }
}

/// Estimate the per-component memory breakdown for given parameters.
///
/// Returns a `BudgetBreakdown` that follows the index-plan convention of
/// excluding mmap from the budget; the mmap term is returned separately so
/// the caller can include it in `MemoryBudget.estimated_bytes` for the
/// sequential path (where the entire file faults into RSS during an
/// epoch).
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

/// Estimate total memory usage including the mmap-resident term, used by
/// the auto-tune to decide when to reduce parameters.
///
/// **Mmap note**: the OS faults pages into RSS as shards are read
/// sequentially; with MADV_SEQUENTIAL the kernel may reclaim pages, but we
/// conservatively include the full file size since `ru_maxrss` captures
/// the high-water mark. Intentional over-estimate — a caller that sees
/// this fit will nearly always fit at runtime.
fn estimate_memory(
    shard_group_size: usize,
    prefetch_batches: usize,
    batch_size: usize,
    n_output_genes: usize,
    shard_target_rows: usize,
    avg_nnz_per_cell: f64,
    file_size_bytes: usize,
) -> usize {
    let breakdown = estimate_breakdown(
        shard_group_size,
        prefetch_batches,
        batch_size,
        n_output_genes,
        shard_target_rows,
        avg_nnz_per_cell,
    );
    breakdown.total_bytes.saturating_add(file_size_bytes)
}

/// Adaptive default budget (MB) for [`LoaderConfig::auto_memory_budget`].
///
/// Returns the memory the **requested** configuration needs (the same model
/// [`compute_memory_budget`] auto-tunes against) rounded up to whole MB with
/// ~12 % headroom, clamped to `[floor_mb, ADAPTIVE_BUDGET_CAP_MB]`. Because the
/// floor is the lower clamp bound this never lowers the budget — a small file
/// whose need is below `floor_mb` keeps `floor_mb`, while a full-width file is
/// raised just enough to fit without the auto-tune shrinking `batch_size` /
/// `shard_group_size`. A need above the cap is clamped to the cap, leaving the
/// hard-ceiling auto-tune (and its warnings) to handle genuinely huge files.
#[allow(clippy::too_many_arguments)]
fn adaptive_budget_mb(
    floor_mb: usize,
    shard_group_size: usize,
    prefetch_batches: usize,
    batch_size: usize,
    n_output_genes: usize,
    shard_target_rows: usize,
    avg_nnz_per_cell: f64,
    file_size_bytes: usize,
) -> usize {
    let requested_need = estimate_memory(
        shard_group_size,
        prefetch_batches,
        batch_size,
        n_output_genes,
        shard_target_rows,
        avg_nnz_per_cell,
        file_size_bytes,
    );
    let need_mb = requested_need.div_ceil(1024 * 1024);
    let with_headroom = need_mb.saturating_add(need_mb / 8);
    // `clamp` panics if min > max. Today the auto path always supplies
    // floor_mb = the 512 MB default (< cap), but guard defensively against a
    // caller that sets `auto_memory_budget` with a budget above the cap so a
    // misconfiguration never panics the interpreter.
    with_headroom.clamp(floor_mb, ADAPTIVE_BUDGET_CAP_MB.max(floor_mb))
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
/// **Fork-safety contract**. Both the tokio current-thread runtime and
/// the rayon `ThreadPool` are constructed *lazily inside `start_epoch`*,
/// i.e. after any fork has happened. The `TrainingPipeline` value itself
/// contains no live runtime,
/// rayon registry, or worker threads at construction time, so a forked child
/// that constructs its own pipeline does not inherit fork-hostile state from
/// the parent. `pyscx.from_anndata` in the parent — which lazily initialises
/// rayon's *global* pool as a side-effect — is the historical wedge case;
/// this design eliminates dependence on that pool entirely.
///
/// See [docs/multithreading.md §Training data loader](../../docs/multithreading.md#training-data-loader-triple-buffered-pipeline).
pub struct TrainingPipeline {
    config: LoaderConfig,
    reader: Arc<ScxReader>,
    obs_metadata: RecordBatch,
    deletion_vectors: Option<DeletionVectors>,
    n_vars: u64,
    #[allow(dead_code)]
    shard_target_rows: u32,
    projection: Option<HvgProjection>,
    memory_budget: MemoryBudget,
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
    epoch_active: bool,
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
        let reader = Arc::new(ScxReader::open(path)?);
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
            let n_shards = reader.catalog().csr_shards_for_modality(mid).len();
            (info.n_vars, n_shards, info.nnz)
        } else {
            (
                header.n_vars,
                reader.catalog().shards_sorted().len(),
                header.nnz,
            )
        };

        // Read obs metadata (full RecordBatch for column extraction)
        let t0 = Instant::now();
        let obs_metadata = reader.read_obs()?;
        if profile {
            eprintln!(
                "[scx-loader profile] read_obs: {:?} ({} rows)",
                t0.elapsed(),
                obs_metadata.num_rows()
            );
        }

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
        if config.auto_memory_budget {
            let n_output_genes = match &config.hvg_indices {
                Some(hvg) => hvg.len(),
                None => n_vars as usize,
            };
            let adaptive_mb = adaptive_budget_mb(
                config.max_memory_mb,
                config.shard_group_size,
                config.prefetch_batches,
                config.batch_size,
                n_output_genes,
                shard_target_rows as usize,
                avg_nnz_per_cell,
                file_size_bytes,
            );
            if adaptive_mb > config.max_memory_mb {
                log::info!(
                    "loader auto-budget: raised max_memory_mb {} -> {} MB to fit the \
                     requested configuration (batch_size={}, shard_group_size={}, \
                     n_output_genes={}) without shrinking the batch. Pass an explicit \
                     max_memory_mb to pin a hard ceiling instead.",
                    config.max_memory_mb,
                    adaptive_mb,
                    config.batch_size,
                    config.shard_group_size,
                    n_output_genes,
                );
                config.max_memory_mb = adaptive_mb;
            }
        }

        let memory_budget = compute_memory_budget(
            &config,
            n_vars,
            shard_target_rows,
            avg_nnz_per_cell,
            file_size_bytes,
        );

        // Propagate effective batch_size back into config
        config.batch_size = memory_budget.batch_size;

        if memory_budget.budget_exceeded {
            log::warn!(
                "estimated memory ({} MB) exceeds budget ({} MB) even at minimums \
                 (batch_size={}, shard_group_size=1, prefetch_batches=2). \
                 Consider setting hvg_indices to reduce n_output_genes from {}.",
                memory_budget.estimated_bytes / (1024 * 1024),
                config.max_memory_mb,
                memory_budget.batch_size,
                n_vars,
            );
        }

        // Create HVG projection if configured
        let projection = config
            .hvg_indices
            .as_ref()
            .map(|indices| HvgProjection::new(indices.clone()));

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
            config,
            reader,
            obs_metadata,
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
            epoch_active: false,
        })
    }

    /// Lazily build the per-pipeline rayon `ThreadPool`. Called from
    /// `start_epoch` so the pool is constructed inside the worker
    /// process, after any fork. Reused across epochs.
    fn ensure_decode_pool(&mut self) -> Result<&Arc<rayon::ThreadPool>> {
        if self.decode_pool.is_none() {
            let n_threads = num_cpus::get_physical().clamp(1, DEFAULT_DECODE_POOL_MAX_THREADS);
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

        // Phase 3.3 mmap smoke-read. Touch one byte of the first CSR shard's
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

        let shard_groups = self.shuffler.shuffle_epoch_sorted(&shard_offsets);

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
        self.epoch_active = true;

        Ok(())
    }

    /// Get the next training batch from the pipeline.
    ///
    /// Returns `Some(batch)` while batches are available, `None` when the
    /// epoch is complete (all cells have been yielded). After `None` is
    /// returned, call `start_epoch()` again for the next epoch.
    ///
    /// Returns `None` if no epoch is active.
    pub fn next_batch(&mut self) -> Option<Batch> {
        let rx = self.batch_rx.as_ref()?;
        let t_recv = Instant::now();
        match rx.recv() {
            Ok(batch) => {
                tracing::trace!(
                    pid = std::process::id(),
                    wait_us = t_recv.elapsed().as_micros() as u64,
                    n_rows = batch.n_rows(),
                    "next_batch: received",
                );
                Some(batch)
            }
            Err(_) => {
                // Channel closed — epoch is complete
                // Join handles to propagate any errors (logged, not returned)
                tracing::trace!(
                    pid = std::process::id(),
                    "next_batch: channel closed (epoch end)"
                );
                let _ = self.join_epoch_handles();
                self.epoch_active = false;
                None
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

    /// Effective batch_size after memory budget auto-tuning.
    /// May be less than the configured batch_size for large gene counts.
    pub fn effective_batch_size(&self) -> usize {
        self.memory_budget.batch_size
    }

    /// Memory budget diagnostics.
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

        self.epoch_active = false;
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
            512,         /* floor */
            1,           /* shard_group_size */
            4,           /* prefetch */
            64,          /* batch_size */
            100,         /* n_output_genes */
            1024,        /* shard_target_rows */
            5.0,         /* avg_nnz_per_cell */
            1024 * 1024, /* 1 MB file */
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
        let file_size = 280 * 1024 * 1024usize;
        let mb = adaptive_budget_mb(
            512,
            1,
            4,
            512,
            n_genes,
            shard_target_rows,
            avg_nnz,
            file_size,
        );
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
            file_size,
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
            512,
            8,
            4,
            1024,
            33_538,                 /* n_output_genes */
            16_384,                 /* shard_target_rows */
            33_538.0,               /* fully dense: avg_nnz == n_genes */
            2 * 1024 * 1024 * 1024, /* 2 GB file */
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
            0,
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

    #[test]
    fn test_memory_budget_includes_mmap_size() {
        // A large file should increase the estimate proportionally
        let config = LoaderConfig {
            hvg_indices: Some((0..2000).collect()),
            ..LoaderConfig::default()
        };
        let budget_no_file = compute_memory_budget(&config, 30_000, 16_384, 10.0, 0);
        let file_2gb = 2_500 * 1024 * 1024; // 2.5 GB
        let budget_large_file = compute_memory_budget(&config, 30_000, 16_384, 10.0, file_2gb);

        assert_eq!(budget_large_file.mmap_bytes, file_2gb);
        assert!(
            budget_large_file.estimated_bytes > budget_no_file.estimated_bytes + file_2gb / 2,
            "large file should significantly increase estimate"
        );
    }

    #[test]
    fn test_memory_budget_large_file_reduces_batch_size() {
        // A 2.5 GB file with 2K HVG should force parameter reductions
        let config = LoaderConfig {
            hvg_indices: Some((0..2000).collect()),
            ..LoaderConfig::default()
        };
        let file_2gb = 2_500 * 1024 * 1024;
        let budget = compute_memory_budget(&config, 30_000, 16_384, 10.0, file_2gb);
        assert!(
            budget.shard_group_size < 8 || budget.prefetch_batches < 4 || budget.batch_size < 1024,
            "2.5 GB file should force parameter reduction even with 2K HVG"
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
        while let Some(batch) = pipeline.next_batch() {
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
        let batch = pipeline.next_batch().unwrap();
        assert_eq!(batch.cell_indices.len(), 10);

        // Second call should return None
        assert!(pipeline.next_batch().is_none());
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
        while let Some(batch) = pipeline.next_batch() {
            epoch0_cells.extend_from_slice(&batch.cell_indices);
        }

        // Epoch 1
        pipeline.start_epoch().unwrap();
        let mut epoch1_cells = Vec::new();
        while let Some(batch) = pipeline.next_batch() {
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
        assert!(pipeline.next_batch().is_none());
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
            while let Some(batch) = pipeline.next_batch() {
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
            while let Some(batch) = pipeline.next_batch() {
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
        while let Some(batch) = pipeline1.next_batch() {
            cells1.extend_from_slice(&batch.cell_indices);
        }
        drop(pipeline1);

        // Second pipeline run with same seed
        let mut pipeline2 = TrainingPipeline::new(&path, config).unwrap();
        pipeline2.start_epoch().unwrap();
        let mut cells2 = Vec::new();
        while let Some(batch) = pipeline2.next_batch() {
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
}
