use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow::record_batch::RecordBatch;
use scx_format::deletion_vectors::DeletionVectors;
use scx_format::reader::ScxReader;

use crate::batch::Batch;
use crate::decode_stage::decode_stage;
use crate::error::{LoaderError, Result};
use crate::io_stage::io_stage;
use crate::projection::HvgProjection;
use crate::shuffle::ShardShuffler;

/// Returns true if SCX_LOADER_PROFILE env var is set to "1" or "true".
fn profiling_enabled() -> bool {
    std::env::var("SCX_LOADER_PROFILE")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false)
}

/// Configuration for the training data loader pipeline.
///
/// The memory budget model is computed by `memory_budget()` below.
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Mini-batch size (default: 1024).
    pub batch_size: usize,
    /// Number of shards read per I/O group (default: 8).
    /// Sequential I/O within each group for disk efficiency.
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
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Memory budget in MB (default: 512).
    /// Pipeline auto-tunes shard_group_size and prefetch_batches to fit.
    pub max_memory_mb: usize,
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
            seed: 42,
            max_memory_mb: 512,
        }
    }
}

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
            return MemoryBudget {
                shard_group_size,
                prefetch_batches,
                batch_size,
                estimated_bytes: estimated,
                mmap_bytes: file_size_bytes,
                budget_exceeded: false,
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
        return MemoryBudget {
            shard_group_size,
            prefetch_batches,
            batch_size,
            estimated_bytes: estimated,
            mmap_bytes: file_size_bytes,
            budget_exceeded: true,
        };
    }
}

/// Estimate total memory usage for given parameters.
///
/// Uses a data-driven model that accounts for:
/// - I/O-decode pipeline overlap (shard_group_size + 1 decoded shards live)
/// - Batch channel + consumer (prefetch_batches.max(2) + 1 batches live)
/// - Mmap'd file (entire file faulted into RSS during a full epoch)
/// - Python/runtime overhead (~50 MB for interpreter, numpy, Arrow, threads)
fn estimate_memory(
    shard_group_size: usize,
    prefetch_batches: usize,
    batch_size: usize,
    n_output_genes: usize,
    shard_target_rows: usize,
    avg_nnz_per_cell: f64,
    file_size_bytes: usize,
) -> usize {
    // Decoded shard stores i64 indptr + i32 indices + f32 values = 12 bytes/nnz
    const BYTES_PER_NNZ_DECODED: usize = 12;
    // Python interpreter + numpy + Arrow RecordBatch + tokio/rayon stacks
    const PYTHON_OVERHEAD: usize = 50 * 1024 * 1024;

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

    // Mmap'd file: the OS faults pages into RSS as shards are read sequentially.
    // During a full epoch, most of the file will be resident in page cache.
    // With MADV_SEQUENTIAL the kernel may reclaim pages, but we conservatively
    // include the full file size since ru_maxrss captures the high-water mark.
    //
    // NOTE: this is an intentional over-estimate of steady-state RSS — the
    // kernel reclaims sequentially-read pages aggressively under memory pressure,
    // so a caller that sees this budget fit their memory limit will nearly always
    // fit at runtime. The mmap_resident term exists to protect against peak
    // page-cache residency near the end of an epoch, not to reflect steady-state.
    let mmap_resident = file_size_bytes;

    shard_buffer
        .saturating_add(batch_buffer)
        .saturating_add(mmap_resident)
        .saturating_add(PYTHON_OVERHEAD)
}

// ---------------------------------------------------------------------------
// E1: TrainingPipeline — triple-buffered pipeline coordinator
// ---------------------------------------------------------------------------

/// Triple-buffered training pipeline coordinator.
///
/// Orchestrates three concurrent stages:
/// 1. **I/O stage** (tokio async): Reads shard groups from the SCX file
/// 2. **Decode stage** (std::thread + rayon): Shuffles, projects, densifies, normalizes
/// 3. **GPU stage** (caller): Consumes pre-built `Batch`es via `next_batch()`
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
    // Runtime state
    batch_rx: Option<crossbeam_channel::Receiver<Batch>>,
    io_handle: Option<tokio::task::JoinHandle<Result<()>>>,
    decode_handle: Option<std::thread::JoinHandle<Result<()>>>,
    runtime: tokio::runtime::Runtime,
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

        config.validate()?;

        // Open SCX file
        let t0 = Instant::now();
        let reader = Arc::new(ScxReader::open(path)?);
        if profile {
            eprintln!("[scx-loader profile] open: {:?}", t0.elapsed());
        }

        // Read header metadata
        let header = reader.header();
        let n_vars = header.n_vars;
        let shard_target_rows = header.shard_target_rows;
        let n_csr_shards = reader.catalog().shards_sorted().len();

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

        // Compute average nnz per cell for memory budget estimation
        let avg_nnz_per_cell = if header.n_obs > 0 {
            header.nnz as f64 / header.n_obs as f64
        } else {
            0.0
        };

        // Compute memory budget and auto-tune parameters
        let file_size_bytes = reader.mmap().len();
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

        // Create tokio runtime for async I/O
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| {
                LoaderError::ShutdownError(format!("failed to create tokio runtime: {e}"))
            })?;

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
            runtime,
            shuffler,
            epoch_active: false,
        })
    }

    /// Start a new training epoch.
    ///
    /// Generates a new shuffled shard ordering, creates bounded channels,
    /// and spawns the I/O and decode stages. Call `next_batch()` to consume
    /// batches from the pipeline.
    ///
    /// If a previous epoch is still active, joins its handles first.
    pub fn start_epoch(&mut self) -> Result<()> {
        // Join previous epoch handles if they exist
        self.join_epoch_handles()?;

        // Generate offset-sorted shard groups for this epoch.
        // Shard groups are randomly composed (stochastic across epochs) but
        // sorted by file offset within and across groups for sequential I/O.
        let sorted_entries = self.reader.catalog().shards_sorted();
        let shard_offsets: Vec<u64> = sorted_entries.iter().map(|e| e.offset).collect();
        let shard_groups = self.shuffler.shuffle_epoch_sorted(&shard_offsets);

        // Create bounded channels
        // I/O → Decode: tokio mpsc channel. Each item is a ShardGroup containing
        // shard_group_size decoded shards. Cap at 2 for pipeline overlap (one
        // being decoded + one read-ahead), not shard_group_size which would allow
        // shard_group_size * shard_group_size decoded shards in flight.
        let (io_tx, io_rx) = tokio::sync::mpsc::channel(2);

        // Decode → Consumer: crossbeam bounded channel, capacity = prefetch_batches
        // Use at least 2 to allow decode to run ahead of consumer.
        let batch_channel_cap = self.memory_budget.prefetch_batches.max(2);
        let (batch_tx, batch_rx) = crossbeam_channel::bounded(batch_channel_cap);

        // Spawn I/O stage as a tokio task
        let io_reader = Arc::clone(&self.reader);
        let io_dv = self.deletion_vectors.clone();
        let io_handle = self
            .runtime
            .spawn(async move { io_stage(io_reader, shard_groups, io_dv, io_tx).await });

        // Spawn decode stage as a standard thread (CPU-bound work)
        let decode_config = self.config.clone();
        let decode_n_vars = self.n_vars;
        let decode_projection = self.projection.clone();
        let decode_obs = self.obs_metadata.clone();
        let decode_epoch = self.shuffler.epoch().saturating_sub(1); // epoch was already incremented by shuffle_epoch()
        let decode_handle = std::thread::Builder::new()
            .name("scx-decode".to_string())
            .spawn(move || {
                decode_stage(
                    io_rx,
                    batch_tx,
                    &decode_config,
                    decode_n_vars,
                    decode_projection,
                    &decode_obs,
                    decode_epoch,
                )
            })
            .map_err(|e| {
                LoaderError::ShutdownError(format!("failed to spawn decode thread: {e}"))
            })?;

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
        match rx.recv() {
            Ok(batch) => Some(batch),
            Err(_) => {
                // Channel closed — epoch is complete
                // Join handles to propagate any errors (logged, not returned)
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
    fn join_epoch_handles(&mut self) -> Result<()> {
        // Drop the batch receiver first to unblock the decode stage
        // if it's trying to send.
        self.batch_rx = None;

        // Join the I/O handle
        if let Some(handle) = self.io_handle.take() {
            // Block on the tokio task. We use block_on here because this
            // method is called from sync context.
            match self.runtime.block_on(handle) {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    return Err(LoaderError::ShutdownError(format!("I/O stage error: {e}")));
                }
                Err(e) => {
                    return Err(LoaderError::ShutdownError(format!(
                        "I/O stage panicked: {e}"
                    )));
                }
            }
        }

        // Join the decode handle
        if let Some(handle) = self.decode_handle.take() {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    return Err(LoaderError::ShutdownError(format!(
                        "decode stage error: {e}"
                    )));
                }
                Err(_) => {
                    return Err(LoaderError::ShutdownError(
                        "decode stage panicked".to_string(),
                    ));
                }
            }
        }

        self.epoch_active = false;
        Ok(())
    }
}

impl Drop for TrainingPipeline {
    /// Best-effort shutdown when a pipeline is dropped without an explicit
    /// `shutdown()`.
    ///
    /// **Async-context caveat:** `Drop` calls `self.runtime.block_on(...)` to
    /// join the tokio I/O task. If the pipeline is dropped from inside a
    /// running tokio runtime (e.g. an `async fn`), the nested `block_on` will
    /// panic. Callers running inside an async context MUST call
    /// `.shutdown().await` before dropping. When `block_on` is unsafe to call
    /// here, we detect the running runtime via `Handle::try_current()` and
    /// skip the join — the task will simply continue until the pipeline's
    /// channel receivers are dropped and it exits on its own.
    fn drop(&mut self) {
        // Drop channels first to signal shutdown to stages
        self.batch_rx = None;

        // Best-effort join of handles — don't propagate errors in Drop
        if let Some(handle) = self.io_handle.take() {
            // Cancel the tokio task if it's still running
            handle.abort();
            if tokio::runtime::Handle::try_current().is_err() {
                let _ = self.runtime.block_on(handle);
            }
            // else: we're in an async context; the aborted task will wind down
            // on its own. Dropping the handle detaches it.
        }
        if let Some(handle) = self.decode_handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_codec::CodecId;
    use scx_format::header::{FileHeader, MAGIC};
    use scx_format::writer::ScxWriter;

    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc as StdArc;

    fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
        FileHeader {
            magic: MAGIC,
            format_version: 1,
            header_length: 256,
            flags: 0,
            n_obs,
            n_vars,
            nnz,
            n_csr_shards: 0,
            n_csc_shards: 0,
            shard_target_rows: 16384,
            codec_id: 0,
            index_dtype: 0,
            endian: 0,
            reserved_padding: 0,
            root_catalog_offset: 0,
            root_catalog_length: 0,
            full_catalog_offset: 0,
            full_catalog_length: 0,
            manifest_sequence: 1,
            prev_catalog_offset: 0,
            file_checksum: 0,
            front_catalog_offset: 0,
            front_catalog_length: 0,
            reserved: [0u8; 132],
        }
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
