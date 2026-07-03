//! Python bindings for the SCX training data loader.
//!
//! Provides `TrainingDataset`, a PyTorch-compatible iterable dataset that wraps
//! the triple-buffered `TrainingPipeline`.
//!
//! # Usage from Python
//!
//! ```python
//! from pyscx import TrainingDataset
//! import torch
//!
//! dataset = TrainingDataset("data.scx", batch_size=1024, hvg_indices=[0, 1, 2])
//! loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)
//!
//! for batch in loader:
//!     X = torch.from_numpy(batch["X"])
//!     # ... training step ...
//! ```

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyRuntimeError, PyStopIteration};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use scx_format_io::CacheMetrics;

use crate::batch::{Batch, ObsColumn};
use crate::error::LoaderError;
use crate::index_plan::{IndexPlanBatch, IndexPlanIter, IndexPlanLoader, IterMetrics};
use crate::pipeline::{LoaderConfig, TrainingPipeline};
use crate::sparse_cellset::{
    CollateScalars, CollatedCellSetBatch, SparseCellSetBatch, SparseCellSetLoader,
    SparseCellSetPlan,
};
use crate::sparse_cellset_collate::PreprocessMode;
use scx_format_io::ScxReader;

/// A PyTorch-compatible iterable dataset for SCX training data.
///
/// Wraps the Rust `TrainingPipeline` and exposes it as a Python iterator.
/// Each call to `__next__` returns a dict `{"X": ndarray, "obs": {...}, "cell_indices": ndarray}`.
///
/// # Default transforms (IMPORTANT)
///
/// **`normalize` and `log1p` both default to `True`** — by default every batch
/// is total-count normalized (`target_sum=1e4`) and `log1p`-transformed, even
/// though the SCX file stores raw counts. The yielded `X` is therefore
/// log-normalized, **not** raw counts.
///
/// Count-likelihood models (scVI, scANVI, count autoencoders, NB/ZINB
/// decoders) require **raw integer counts** — pass
/// `normalize=False, log1p=False` to disable both transforms and stream raw
/// counts through unchanged:
///
/// ```python
/// # log-normalized batches (default — for models that expect lognorm input)
/// ds = pyscx.TrainingDataset("counts.scx", batch_size=256)
///
/// # raw counts (for scVI / count-based likelihoods)
/// ds = pyscx.TrainingDataset("counts.scx", batch_size=256,
///                            normalize=False, log1p=False)
/// ```
///
/// See the `normalize` / `log1p` / `target_sum` / `pflog1ppf` constructor args
/// below for the full normalization surface.
///
/// # Fork safety
///
/// `TrainingDataset` is fork-safe under PyTorch
/// `DataLoader(num_workers > 0, start_method="fork")` **when the dataset is
/// constructed lazily inside the worker's `__iter__`** (the pattern used by
/// `cell-load-scx`'s `ScxTrainingDataset` and `state-scx`'s
/// `ScxStateAdapter`). The pipeline's internal tokio current-thread runtime
/// (per-epoch, owned by a dedicated I/O `std::thread`) and per-pipeline
/// `rayon::ThreadPool` (per-instance, lazily built on first `start_epoch()`)
/// are constructed *inside the worker process* and therefore never inherit
/// fork-hostile thread state from the parent.
///
/// **Constraints that still apply:**
/// - Eager-construct in parent + fork = unsupported. The PID check in
///   `__next__` raises `RuntimeError` if a `TrainingDataset` constructed
///   in the parent is used from a forked child.
/// - CUDA-initialised parent + fork = unsupported (PyTorch / driver
///   territory; no scx-side fix possible).
/// - Don't share a `TrainingDataset` across processes via pickle / Manager
///   handles — it owns thread handles that don't survive transfer.
///
/// **Recommended for environments that allow it:** use
/// `multiprocessing.set_start_method("spawn")`, which re-execs Python in
/// the child and eliminates fork hazards entirely.
///
/// **Recommended for clean shutdown:** call `dataset.close()` (or register
/// `weakref.finalize(dataset, dataset.close)` at construction time) before
/// process exit so the rayon pool and I/O thread shut down while the
/// interpreter is still healthy. `Drop` runs at interpreter teardown as a
/// fallback but is bounded by a 5-second deadline per thread to avoid
/// hangs.
#[pyclass]
pub struct TrainingDataset {
    pipeline: TrainingPipeline,
    epoch_started: bool,
    /// PID at construction time — used to detect forking (num_workers > 0).
    creation_pid: u32,
}

#[pymethods]
impl TrainingDataset {
    /// Create a new TrainingDataset from an SCX file.
    ///
    /// Args:
    ///     path: Path to the .scx file.
    ///     batch_size: Mini-batch size (default: 1024).
    ///     hvg_indices: Gene indices for HVG projection. None = all genes.
    ///     obs_columns: Obs metadata column names to include in each batch.
    ///     normalize: Apply total-count normalization (default: True).
    ///         NOTE: this is **on by default** — batches are normalized even
    ///         though the file stores raw counts. Pass `normalize=False`
    ///         (with `log1p=False`) for raw-count output (scVI / count
    ///         likelihoods). See "Default transforms" on the class docstring.
    ///     log1p: Apply log1p transformation (default: True). On by default;
    ///         pass `log1p=False` to disable. See `normalize`.
    ///     target_sum: Normalization target sum (default: 1e4).
    ///     pflog1ppf: Apply PFlog1pPF / shifted-CLR normalization
    ///         (Booeshaghi et al. 2026) instead of normalize/log1p
    ///         (default: False). PFlog1pPF is itself a normalization, so it is
    ///         mutually exclusive with `normalize`/`log1p`: when True it takes
    ///         precedence and those flags are ignored. Depth and the centering
    ///         denominator are computed over the full transcriptome even under
    ///         `hvg_indices` projection.
    ///     pflog1ppf_c: PFlog1pPF shift / pseudocount `c` (default: 1.0; only
    ///         used when `pflog1ppf=True`).
    ///     shard_group_size: Shards per I/O group (default: 8).
    ///     prefetch_batches: Ring buffer depth (default: 4).
    ///     seed: RNG seed for reproducibility (default: 42).
    ///     max_memory_mb: Memory budget in MB. When omitted (None) the budget
    ///         is adaptive: it scales up to fit the file's requested
    ///         configuration (floored at 512 MB, capped at 4096 MB) so a
    ///         full-width ~33k-gene file keeps its requested `batch_size`
    ///         instead of silently shrinking it. Pass an explicit value to pin
    ///         a hard ceiling — the pipeline then auto-tunes `shard_group_size`,
    ///         `prefetch_batches`, and `batch_size` down to fit it.
    ///     modality: Phase H.1 — name of the modality to load on a
    ///         multimodal v2 file. On a single-modality file this is
    ///         ignored. On a multimodal file with no `modality`
    ///         argument, the dataset emits a UserWarning and loads
    ///         the alphabetically-first modality only — use
    ///         `MultimodalTrainingDataset` for full coverage.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        path,
        batch_size=None,
        hvg_indices=None,
        obs_columns=None,
        normalize=None,
        log1p=None,
        target_sum=None,
        pflog1ppf=None,
        pflog1ppf_c=None,
        shard_group_size=None,
        prefetch_batches=None,
        seed=None,
        max_memory_mb=None,
        modality=None,
    ))]
    fn new(
        py: Python<'_>,
        path: &str,
        batch_size: Option<usize>,
        hvg_indices: Option<Vec<u32>>,
        obs_columns: Option<Vec<String>>,
        normalize: Option<bool>,
        log1p: Option<bool>,
        target_sum: Option<f64>,
        pflog1ppf: Option<bool>,
        pflog1ppf_c: Option<f64>,
        shard_group_size: Option<usize>,
        prefetch_batches: Option<usize>,
        seed: Option<u64>,
        max_memory_mb: Option<usize>,
        modality: Option<String>,
    ) -> PyResult<Self> {
        // Phase H.1: resolve modality_id. Single-modality files keep
        // the legacy global path. Multimodal files need either an
        // explicit `modality=` kwarg or fall back to the alphabetical
        // first modality (with a UserWarning telling the user to use
        // MultimodalTrainingDataset).
        let modality_id = resolve_modality_id_with_warning(py, path, modality.as_deref())?;

        let defaults = LoaderConfig::default();
        let config = LoaderConfig {
            batch_size: batch_size.unwrap_or(defaults.batch_size),
            shard_group_size: shard_group_size.unwrap_or(defaults.shard_group_size),
            prefetch_batches: prefetch_batches.unwrap_or(defaults.prefetch_batches),
            hvg_indices,
            obs_columns: obs_columns.unwrap_or_default(),
            normalize: normalize.unwrap_or(defaults.normalize),
            log1p: log1p.unwrap_or(defaults.log1p),
            target_sum: target_sum.unwrap_or(defaults.target_sum),
            pflog1ppf: pflog1ppf.unwrap_or(defaults.pflog1ppf),
            pflog1ppf_c: pflog1ppf_c.unwrap_or(defaults.pflog1ppf_c),
            seed: seed.unwrap_or(defaults.seed),
            max_memory_mb: max_memory_mb.unwrap_or(defaults.max_memory_mb),
            // No explicit budget → adaptive (treat the default as a floor and
            // raise to fit a full-width file). An explicit budget is a hard
            // ceiling (preserves the auto-tune-down + warning behaviour).
            auto_memory_budget: max_memory_mb.is_none(),
            modality_id,
        };

        let pipeline = TrainingPipeline::new(path, config)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(TrainingDataset {
            pipeline,
            epoch_started: false,
            creation_pid: std::process::id(),
        })
    }

    /// Start a new epoch. Called automatically by `for batch in dataset`.
    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        slf.pipeline
            .start_epoch()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        slf.epoch_started = true;
        Ok(slf)
    }

    /// Get the next batch from the pipeline.
    ///
    /// Returns a dict `{"X": ndarray, "obs": {...}, "cell_indices": ndarray}`,
    /// or raises `StopIteration` when the epoch is complete.
    ///
    /// The GIL is released while waiting for the next batch from the Rust
    /// pipeline, allowing PyTorch CUDA threads to run concurrently.
    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        // Fork detection: if current PID differs from creation PID, we've been
        // forked by DataLoader with num_workers > 0.
        if std::process::id() != self.creation_pid {
            return Err(PyRuntimeError::new_err(
                "scx.TrainingDataset requires num_workers=0. The Rust pipeline manages its \
                 own threads. Set: DataLoader(dataset, batch_size=None, num_workers=0)",
            ));
        }

        if !self.epoch_started {
            return Err(PyRuntimeError::new_err(
                "Must call __iter__ before __next__",
            ));
        }

        // Release the GIL while waiting for the next batch from the Rust
        // pipeline. This allows other Python threads (e.g., PyTorch CUDA
        // threads) to run while Stage 2 builds the next batch.
        let batch_opt = py.detach(|| self.pipeline.next_batch());

        match batch_opt {
            Some(batch) => {
                let dict = batch_to_dict(py, batch)?;
                Ok(Some(dict))
            }
            None => {
                self.epoch_started = false;
                Ok(None) // StopIteration
            }
        }
    }

    /// Total number of observations (cells) in the dataset.
    #[getter]
    fn n_obs(&self) -> u64 {
        self.pipeline.n_obs()
    }

    /// Total number of variables (genes) in the dataset.
    #[getter]
    fn n_vars(&self) -> u64 {
        self.pipeline.n_vars()
    }

    /// Number of output genes per batch (HVG count if projection active).
    #[getter]
    fn n_output_genes(&self) -> usize {
        self.pipeline.n_output_genes()
    }

    /// Effective batch_size after memory budget auto-tuning.
    /// May be less than the requested batch_size for large gene counts.
    #[getter]
    fn effective_batch_size(&self) -> usize {
        self.pipeline.effective_batch_size()
    }

    /// Memory budget diagnostics as a dict.
    ///
    /// Legacy fields (`shard_group_size`, `prefetch_batches`, `batch_size`,
    /// `estimated_mb`, `mmap_mb`, `budget_exceeded`) plus a nested
    /// `breakdown` dict matching `IndexPlanDataset.memory_budget()`'s
    /// per-component shape (`cache_bytes`, `batch_buffer_bytes`,
    /// `lookahead_overhead_bytes`, `transient_bytes`, `python_overhead_bytes`,
    /// `total_bytes`). Sequential paths report `0` for `lookahead_overhead`
    /// and `transient` since they don't carry plan-tuple staging or
    /// per-batch obs scratch.
    fn memory_budget<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let mb = self.pipeline.memory_budget_info();
        dict.set_item("shard_group_size", mb.shard_group_size)?;
        dict.set_item("prefetch_batches", mb.prefetch_batches)?;
        dict.set_item("batch_size", mb.batch_size)?;
        dict.set_item("estimated_mb", mb.estimated_bytes / (1024 * 1024))?;
        dict.set_item("mmap_mb", mb.mmap_bytes / (1024 * 1024))?;
        dict.set_item("budget_exceeded", mb.budget_exceeded)?;
        dict.set_item("breakdown", mb.breakdown.to_pydict(py)?)?;
        Ok(dict)
    }

    /// Explicitly shut the pipeline down: drop channels, join I/O + decode
    /// threads (bounded by `SHUTDOWN_DEADLINE`), and release the
    /// per-pipeline rayon pool.
    ///
    /// Idempotent. Safe to call multiple times. Once `close()` has been
    /// called, subsequent `__iter__` / `__next__` calls behave as if no
    /// epoch were active — the next `__iter__` re-builds the rayon pool
    /// and runtime lazily.
    ///
    /// Recommended pattern under PyTorch DataLoader workers: register a
    /// `weakref.finalize(self, lambda: ds.close())` at construction time
    /// so the pool / runtime are torn down before interpreter teardown
    /// (where `Drop`'s GIL probe might still be too late).
    fn close(&mut self, py: Python<'_>) {
        py.detach(|| self.pipeline.shutdown());
    }

    fn __repr__(&self) -> String {
        format!(
            "TrainingDataset(n_obs={}, n_vars={}, n_output_genes={})",
            self.pipeline.n_obs(),
            self.pipeline.n_vars(),
            self.pipeline.n_output_genes(),
        )
    }
}

/// Multimodal training dataset (Phase H.1 of MULTIMODAL-SUPPORT.md).
///
/// Wraps N independent `TrainingPipeline` instances — one per requested
/// modality — and yields per-batch dicts whose cell axes align across
/// modalities. Cells (obs) are global across modalities, so all
/// pipelines see the same `n_obs` and the same shuffler seed produces
/// the same row ordering when their per-modality shard layouts agree
/// (the standard CITE-seq / multiome writer guarantees this).
///
/// On `__next__`, returns
/// `{"X": {modality_name: ndarray, ...}, "obs": {...}, "cell_indices": ndarray}`
/// when constructed with `return_dict=True` (default), or a tuple
/// `(X_modality_0, X_modality_1, ...)` when `return_dict=False`. The
/// per-modality X arrays share the same `cell_indices` row ordering;
/// the wrapper validates this on each batch and raises
/// `RuntimeError` if the per-modality shufflers diverge (e.g. because
/// the modalities have different shard layouts on disk — typically a
/// writer / file-construction bug).
#[pyclass]
pub struct MultimodalTrainingDataset {
    /// One pipeline per requested modality, in the order the user
    /// supplied them.
    pipelines: Vec<TrainingPipeline>,
    /// Modality names parallel to `pipelines`.
    modality_names: Vec<String>,
    /// True → batches are dicts. False → batches are tuples of X arrays.
    return_dict: bool,
    epoch_started: bool,
    creation_pid: u32,
}

#[pymethods]
impl MultimodalTrainingDataset {
    /// Create a new `MultimodalTrainingDataset` from a multimodal SCX file.
    ///
    /// Args:
    ///     path: Path to the .scx file.
    ///     modalities: List of modality names to load (e.g.
    ///         `["rna", "adt"]`). All listed modalities must exist
    ///         in the file's modality table.
    ///     batch_size: Mini-batch size shared across modalities.
    ///     hvg_indices: Optional HVG projection. Currently applied
    ///         to every modality identically; for per-modality
    ///         projections, instantiate separate
    ///         `TrainingDataset(modality=…, hvg_indices=…)` instances
    ///         and zip in Python.
    ///     obs_columns: Obs columns to include in each batch (read
    ///         once from the global obs table).
    ///     return_dict: If True (default), yield
    ///         `{"X": {name: ndarray}, "obs": {...}, "cell_indices": ...}`.
    ///         If False, yield a tuple `(X_0, X_1, …)` aligned with
    ///         `modalities` order.
    ///     normalize, log1p, target_sum, pflog1ppf, pflog1ppf_c,
    ///     shard_group_size, prefetch_batches, seed, max_memory_mb: see
    ///     `TrainingDataset` for semantics — applied to every modality
    ///     uniformly. `pflog1ppf` (RNA-appropriate) replaces normalize/log1p
    ///     when set. An explicit `max_memory_mb` is divided across modalities
    ///     proportionally to per-modality nnz (modalities with denser X get a
    ///     larger share of the memory budget). When omitted, each modality's
    ///     per-modality share becomes an adaptive floor (raised to fit that
    ///     modality's full-width configuration), matching `TrainingDataset`'s
    ///     default behaviour.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        path,
        modalities,
        batch_size=None,
        hvg_indices=None,
        obs_columns=None,
        return_dict=None,
        normalize=None,
        log1p=None,
        target_sum=None,
        pflog1ppf=None,
        pflog1ppf_c=None,
        shard_group_size=None,
        prefetch_batches=None,
        seed=None,
        max_memory_mb=None,
    ))]
    fn new(
        path: &str,
        modalities: Vec<String>,
        batch_size: Option<usize>,
        hvg_indices: Option<Vec<u32>>,
        obs_columns: Option<Vec<String>>,
        return_dict: Option<bool>,
        normalize: Option<bool>,
        log1p: Option<bool>,
        target_sum: Option<f64>,
        pflog1ppf: Option<bool>,
        pflog1ppf_c: Option<f64>,
        shard_group_size: Option<usize>,
        prefetch_batches: Option<usize>,
        seed: Option<u64>,
        max_memory_mb: Option<usize>,
    ) -> PyResult<Self> {
        use scx_format_io::ScxReader;
        if modalities.is_empty() {
            return Err(PyRuntimeError::new_err(
                "MultimodalTrainingDataset: `modalities` must contain at least one name",
            ));
        }

        // Open the file once to resolve modality_id + per-modality nnz
        // for the memory-budget split. The inner `TrainingPipeline::new`
        // re-opens the file per modality.
        let reader = ScxReader::open(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if !reader.is_multimodal() {
            return Err(PyRuntimeError::new_err(format!(
                "MultimodalTrainingDataset: file '{path}' is single-modality. \
                 Use TrainingDataset(path) instead."
            )));
        }
        let mut resolved: Vec<(String, u8, u64)> = Vec::with_capacity(modalities.len());
        for name in &modalities {
            let mid = reader.modality_id(name).ok_or_else(|| {
                let registered: Vec<&str> = reader.modality_names();
                PyRuntimeError::new_err(format!(
                    "MultimodalTrainingDataset: file '{path}' does not have a \
                     modality named '{name}'. Registered modalities: {registered:?}"
                ))
            })?;
            let info = reader.modality_info(mid).ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "MultimodalTrainingDataset: failed to resolve modality_info for '{name}'"
                ))
            })?;
            resolved.push((name.clone(), mid, info.nnz));
        }

        // Memory budget split. The user-supplied `max_memory_mb` is
        // total across all modalities; allocate proportionally to
        // per-modality nnz. Each modality gets at least 64 MB so the
        // budget tuner has room to settle on viable shard_group_size /
        // prefetch_batches values.
        let defaults = LoaderConfig::default();
        let total_mb = max_memory_mb.unwrap_or(defaults.max_memory_mb);
        let total_nnz: u64 = resolved.iter().map(|(_, _, n)| *n).sum::<u64>().max(1);
        let per_modality_mb: Vec<usize> = resolved
            .iter()
            .map(|(_, _, nnz)| {
                let share = (total_mb as f64) * (*nnz as f64) / (total_nnz as f64);
                (share as usize).max(64)
            })
            .collect();

        // Build one pipeline per modality. All pipelines share the
        // same seed so the per-modality shufflers agree on row
        // ordering as long as the per-modality shard layouts align
        // (which the standard multimodal writer guarantees).
        drop(reader);
        let mut pipelines = Vec::with_capacity(resolved.len());
        let mut names = Vec::with_capacity(resolved.len());
        for ((name, mid, _), modality_mb) in resolved.into_iter().zip(per_modality_mb) {
            let config = LoaderConfig {
                batch_size: batch_size.unwrap_or(defaults.batch_size),
                shard_group_size: shard_group_size.unwrap_or(defaults.shard_group_size),
                prefetch_batches: prefetch_batches.unwrap_or(defaults.prefetch_batches),
                hvg_indices: hvg_indices.clone(),
                obs_columns: obs_columns.clone().unwrap_or_default(),
                normalize: normalize.unwrap_or(defaults.normalize),
                log1p: log1p.unwrap_or(defaults.log1p),
                target_sum: target_sum.unwrap_or(defaults.target_sum),
                pflog1ppf: pflog1ppf.unwrap_or(defaults.pflog1ppf),
                pflog1ppf_c: pflog1ppf_c.unwrap_or(defaults.pflog1ppf_c),
                seed: seed.unwrap_or(defaults.seed),
                max_memory_mb: modality_mb,
                // No explicit total budget → adaptive per-modality floor:
                // each modality's pipeline raises to fit its own full-width
                // configuration rather than shrinking the batch.
                auto_memory_budget: max_memory_mb.is_none(),
                modality_id: Some(mid),
            };
            let pipeline = TrainingPipeline::new(path, config)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            pipelines.push(pipeline);
            names.push(name);
        }

        Ok(MultimodalTrainingDataset {
            pipelines,
            modality_names: names,
            return_dict: return_dict.unwrap_or(true),
            epoch_started: false,
            creation_pid: std::process::id(),
        })
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        for p in slf.pipelines.iter_mut() {
            p.start_epoch()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        slf.epoch_started = true;
        Ok(slf)
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if std::process::id() != self.creation_pid {
            return Err(PyRuntimeError::new_err(
                "scx.MultimodalTrainingDataset requires num_workers=0. The Rust pipeline manages its \
                 own threads.",
            ));
        }
        if !self.epoch_started {
            return Err(PyRuntimeError::new_err(
                "Must call __iter__ before __next__",
            ));
        }

        // Pull one batch from each modality in lockstep. Releasing the
        // GIL once around all pulls is the simplest correct
        // implementation; with the same seed the pipelines should
        // produce in roughly aligned cadence.
        let batches_opt: Option<Vec<Batch>> = py.detach(|| {
            let mut out = Vec::with_capacity(self.pipelines.len());
            for p in self.pipelines.iter_mut() {
                match p.next_batch() {
                    Some(b) => out.push(b),
                    None => return None,
                }
            }
            Some(out)
        });

        let batches = match batches_opt {
            Some(b) => b,
            None => {
                self.epoch_started = false;
                return Ok(None);
            }
        };

        // Cross-modality cell-axis alignment check. The per-modality
        // shufflers should produce identical row orderings when their
        // shard layouts match; if they don't, surface a clear error
        // rather than silently emit mis-aligned batches.
        let first_indices = &batches[0].cell_indices;
        for (i, b) in batches.iter().enumerate().skip(1) {
            if b.cell_indices != *first_indices {
                return Err(PyRuntimeError::new_err(format!(
                    "MultimodalTrainingDataset: modalities '{}' and '{}' produced \
                     different cell_indices on the same batch — this typically means \
                     the per-modality shard layouts disagree (different shard counts \
                     or row ranges). Reshard the file with a uniform shard_target_rows \
                     before training.",
                    self.modality_names[0], self.modality_names[i],
                )));
            }
        }

        // Build the output. Both the dict and tuple paths share the
        // first batch's obs / cell_indices (cells are global, so the
        // obs record is identical across modalities).
        let dict = build_multimodal_batch_dict(py, &batches, &self.modality_names)?;
        if self.return_dict {
            Ok(Some(dict.into_any()))
        } else {
            // Tuple of X arrays in modality order.
            let x_dict = dict.get_item("X")?.expect("X key always present");
            let x_dict = x_dict.cast::<PyDict>()?;
            let mut tuple_items: Vec<Bound<'_, PyAny>> =
                Vec::with_capacity(self.modality_names.len());
            for name in &self.modality_names {
                let v = x_dict
                    .get_item(name)?
                    .expect("modality entry always present");
                tuple_items.push(v);
            }
            Ok(Some(PyTuple::new(py, &tuple_items)?.into_any()))
        }
    }

    /// Total observations (cells), shared across modalities (global obs axis).
    #[getter]
    fn n_obs(&self) -> u64 {
        self.pipelines.first().map(|p| p.n_obs()).unwrap_or(0)
    }

    /// List of modality names, in the same order as the constructor's
    /// `modalities` argument.
    #[getter]
    fn modality_names(&self) -> Vec<String> {
        self.modality_names.clone()
    }

    /// Per-modality `n_vars` as a dict.
    #[getter]
    fn n_vars(&self) -> HashMap<String, u64> {
        self.modality_names
            .iter()
            .zip(self.pipelines.iter())
            .map(|(n, p)| (n.clone(), p.n_vars()))
            .collect()
    }

    fn close(&mut self, py: Python<'_>) {
        py.detach(|| {
            for p in self.pipelines.iter_mut() {
                p.shutdown();
            }
        });
    }

    fn __repr__(&self) -> String {
        let names: Vec<&str> = self.modality_names.iter().map(|s| s.as_str()).collect();
        format!(
            "MultimodalTrainingDataset(n_obs={}, modalities={names:?})",
            self.n_obs(),
        )
    }
}

/// Phase H.2 helper: build a `{"X": {name: ndarray}, "obs": {...},
/// "cell_indices": ndarray}` dict from a slice of per-modality
/// `Batch`es. Uses the first batch's `obs` and `cell_indices` (cells
/// are global across modalities).
fn build_multimodal_batch_dict<'py>(
    py: Python<'py>,
    batches: &[Batch],
    modality_names: &[String],
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::IntoPyArray;
    let dict = PyDict::new(py);

    let x_dict = PyDict::new(py);
    for (name, batch) in modality_names.iter().zip(batches.iter()) {
        let n_genes = if batch.x_shape.0 > 0 {
            batch.x_shape.1
        } else {
            0
        };
        let x_owned: Vec<f32> = batch.x.clone();
        let x_arr = x_owned.into_pyarray(py);
        let x_2d = x_arr
            .reshape([batch.x_shape.0, n_genes])
            .map_err(|e| PyRuntimeError::new_err(format!("X reshape failed: {e}")))?;
        x_dict.set_item(name, x_2d)?;
    }
    dict.set_item("X", x_dict)?;

    // obs and cell_indices come from the first modality's batch
    // (all batches share the global obs table). Skip if obs_columns
    // were not requested — empty dict.
    let obs_dict = PyDict::new(py);
    for (col_name, col) in &batches[0].obs {
        let arr_obj = match col {
            ObsColumn::Float64(v) => v.clone().into_pyarray(py).into_any(),
            ObsColumn::Int64(v) => v.clone().into_pyarray(py).into_any(),
            ObsColumn::Categorical(codes, cats) => {
                // Decode codes to category strings; emit as a Python
                // list (numpy lacks a native variable-width string
                // dtype). Consumer can wrap in pandas.Categorical.
                let decoded: Vec<&str> = codes
                    .iter()
                    .map(|&c| cats.get(c as usize).map(|s| s.as_str()).unwrap_or(""))
                    .collect();
                let list = pyo3::types::PyList::new(py, decoded)?;
                list.into_any()
            }
        };
        obs_dict.set_item(col_name, arr_obj)?;
    }
    dict.set_item("obs", obs_dict)?;
    dict.set_item(
        "cell_indices",
        batches[0].cell_indices.clone().into_pyarray(py),
    )?;
    Ok(dict)
}

/// Plan-driven paired-batch reader for `(perturbed, control)` ML training.
///
/// Sibling to `TrainingDataset`: instead of streaming shards in catalog order,
/// `IndexPlanDataset` consumes caller-supplied `(pert_idx, ctrl_idx)` plans
/// and returns paired dense batches.
///
/// Surface: `next_batch(plan)` is the synchronous entry point; the streaming
/// `iter_with_plans` API exposes a tokio-driven prefetched iterator.
///
/// # Fork safety
///
/// `IndexPlanDataset` is fork-safe under PyTorch
/// `DataLoader(num_workers > 0, start_method="fork")` **when the dataset is
/// constructed lazily inside the worker's `__iter__`** — same contract as
/// `TrainingDataset`. Because `IndexPlanLoader` does *not* use rayon's
/// global pool (its prefetch goes via `tokio::spawn_blocking` and
/// `std::thread::spawn`), the rayon-after-fork hazard that motivated
/// Phase 2.0 for `TrainingDataset` does **not** apply here. The remaining
/// concern is the eager `tokio::runtime::Builder::new_multi_thread()` built
/// in `IndexPlanLoader::new()` (`scx-loader/src/index_plan.rs:276`); under
/// the lazy-construct-in-worker pattern that runtime is built fresh in the
/// worker process, so the parent's runtime threads are never inherited.
/// The eager-construct-then-fork case is caught by the PID check in
/// `iter_with_plans` / `next_batch_for_test`.
///
/// **Acceptance test**: `pyscx/tests/test_fork_safety.py::test_fork_index_plan_dataset`
/// pins this contract end-to-end (multiprocessing.fork + lazy worker
/// construction + plan iteration to completion).
///
/// Recommended: call `start_method="spawn"` for the same reason it is
/// recommended on `TrainingDataset` — no fork hazards at all.
#[pyclass]
pub struct IndexPlanDataset {
    loader: Arc<IndexPlanLoader>,
    /// PID at construction time — used to detect forking. The shard cache and
    /// mmap state are not fork-safe; consumers must lazily construct the
    /// dataset post-fork in each DataLoader worker.
    creation_pid: u32,
}

#[pymethods]
impl IndexPlanDataset {
    /// Construct a plan-driven paired-batch reader.
    ///
    /// Args:
    ///     path: Path to the .scx file.
    ///     hvg_indices: Gene indices for HVG projection. None = all genes.
    ///     obs_columns: Obs metadata column names to include in each batch.
    ///     normalize: Apply total-count normalization (default: True).
    ///     log1p: Apply ln(x+1) to each row (default: True). Independent of
    ///         `normalize`; all four (normalize, log1p) combinations are
    ///         honoured, matching `TrainingDataset` semantics.
    ///     target_sum: Normalization target sum (default: 1e4).
    ///     pflog1ppf: Apply PFlog1pPF / shifted-CLR normalization
    ///         (Booeshaghi et al. 2026) instead of normalize/log1p
    ///         (default: False). Mutually exclusive with `normalize`/`log1p`
    ///         (takes precedence when True). Depth and the centering
    ///         denominator are over the full transcriptome even under
    ///         `hvg_indices` projection. See `TrainingDataset` for semantics.
    ///     pflog1ppf_c: PFlog1pPF shift / pseudocount `c` (default: 1.0; only
    ///         used when `pflog1ppf=True`).
    ///     cache_shards: LRU shard cache count cap (default: 128). Must be
    ///         >= 1. Auto-tuned downward to fit `max_memory_mb`; check the
    ///         resolved value via `effective_cache_shards()`. The cache also
    ///         enforces a byte cap derived from the memory budget — see
    ///         `cache_metrics()` for runtime hit/miss/eviction observability.
    ///     sort_by_shard: Reorder each plan by shard-of-min-row before
    ///         gathering, so the returned rows of X/X_paired are in shard
    ///         locality order (default: True). Disable to preserve the
    ///         caller's input pair order.
    ///     lookahead: Default lookahead used by `iter_with_plans` when the
    ///         caller does not pass an explicit value (default: 4). 0 disables
    ///         shard prefetching. Auto-tuned downward to fit `max_memory_mb`;
    ///         check the resolved value via `effective_lookahead()`.
    ///     max_plan_size: Upper bound on rows-per-batch for the memory budget
    ///         calculation (default: 16384). Sets the ceiling on the dense
    ///         X / X_paired buffers so the loader can refuse pathological
    ///         plans early.
    ///     max_memory_mb: Memory budget in MB (default: 512). On overflow,
    ///         lookahead is reduced first (down to 1), then cache_shards
    ///         (down to 1); construction fails if neither fits.
    ///     scatter_sidecar: Per-dataset escape hatch (default: True) gating the
    ///         sidecar-aware prefetch skip. True lets cold sparse plans decode
    ///         O(rows) via the scx1 decode sidecar; False makes the prefetch
    ///         warm whole shards (legacy full-shard decode). `SCX_SCATTER_SIDECAR=0`
    ///         is the process-wide reader-layer kill-switch.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        path,
        hvg_indices=None,
        obs_columns=None,
        normalize=None,
        log1p=None,
        target_sum=None,
        pflog1ppf=None,
        pflog1ppf_c=None,
        cache_shards=None,
        sort_by_shard=None,
        lookahead=None,
        max_plan_size=None,
        max_memory_mb=None,
        scatter_sidecar=None,
    ))]
    fn new(
        path: &str,
        hvg_indices: Option<Vec<u32>>,
        obs_columns: Option<Vec<String>>,
        normalize: Option<bool>,
        log1p: Option<bool>,
        target_sum: Option<f64>,
        pflog1ppf: Option<bool>,
        pflog1ppf_c: Option<f64>,
        cache_shards: Option<usize>,
        sort_by_shard: Option<bool>,
        lookahead: Option<usize>,
        max_plan_size: Option<usize>,
        max_memory_mb: Option<usize>,
        scatter_sidecar: Option<bool>,
    ) -> PyResult<Self> {
        let mut config = LoaderConfig::default();
        if let Some(v) = hvg_indices {
            config.hvg_indices = Some(v);
        }
        if let Some(v) = obs_columns {
            config.obs_columns = v;
        }
        if let Some(v) = normalize {
            config.normalize = v;
        }
        if let Some(v) = log1p {
            config.log1p = v;
        }
        if let Some(v) = target_sum {
            config.target_sum = v;
        }
        if let Some(v) = pflog1ppf {
            config.pflog1ppf = v;
        }
        if let Some(v) = pflog1ppf_c {
            config.pflog1ppf_c = v;
        }
        if let Some(v) = max_memory_mb {
            config.max_memory_mb = v;
        }

        let cache_shards = cache_shards.unwrap_or(128);
        let sort_by_shard = sort_by_shard.unwrap_or(true);
        let lookahead = lookahead.unwrap_or(4);
        let max_plan_size = max_plan_size.unwrap_or(16384);
        let scatter_sidecar = scatter_sidecar.unwrap_or(true);

        let loader = IndexPlanLoader::new(
            path,
            config,
            cache_shards,
            sort_by_shard,
            lookahead,
            max_plan_size,
            scatter_sidecar,
        )
        .map_err(loader_err_to_py)?;

        Ok(Self {
            loader: Arc::new(loader),
            creation_pid: std::process::id(),
        })
    }

    /// Total number of observations (cells) in the dataset.
    #[getter]
    fn n_obs(&self) -> u64 {
        self.loader.n_obs()
    }

    /// Total number of variables (genes) in the dataset.
    #[getter]
    fn n_vars(&self) -> u64 {
        self.loader.n_vars()
    }

    /// Number of output genes per batch (HVG count if projection active).
    #[getter]
    fn n_output_genes(&self) -> usize {
        self.loader.n_output_cols()
    }

    /// Drive the loader from a Python iterable of `(pert_idx, ctrl_idx)`
    /// pair lists; return an iterator of paired-batch dicts.
    ///
    /// Args:
    ///     plans: any iterable yielding `list[tuple[int, int]]` (or any
    ///         sequence of (int, int) pairs).
    ///     lookahead: override the loader's default lookahead. None (the
    ///         default) uses `effective_lookahead()` — the constructor's
    ///         auto-tuned value. 0 disables prefetch entirely (decoded
    ///         synchronously per plan). Larger values trade RAM (~1
    ///         prefetch slot per lookahead) for I/O hiding.
    ///
    /// Returns: an `IndexPlanBatchIter` (Python iterator) whose `__next__`
    /// yields `{"X", "X_paired", "pairs", "obs", "obs_paired"}` dicts.
    /// Plan iteration is lazy: the loader pulls the next plan only when it
    /// is ready to schedule a prefetch for it.
    #[pyo3(signature = (plans, lookahead=None))]
    fn iter_with_plans(
        &self,
        plans: Py<PyAny>,
        lookahead: Option<usize>,
    ) -> PyResult<IndexPlanBatchIter> {
        if std::process::id() != self.creation_pid {
            return Err(PyRuntimeError::new_err(
                "scx.IndexPlanDataset requires num_workers=0 (or lazy per-worker \
                 construction). The Rust shard cache and mmap state are not fork-safe.",
            ));
        }
        let lookahead = lookahead.unwrap_or_else(|| self.loader.effective_lookahead());

        // Bind plans → its iter, hold an owned Py<PyAny> Send-safe handle.
        let py_iter: Py<PyAny> = Python::attach(|py| -> PyResult<Py<PyAny>> {
            Ok(plans.bind(py).call_method0("__iter__")?.unbind())
        })?;

        let plan_stream = PyPlanIterator { py_iter };
        let inner = Arc::clone(&self.loader).iter_with_plans(plan_stream, lookahead);
        let iter_metrics = inner.iter_metrics();
        let cache_metrics = self.loader.cache_metrics();
        Ok(IndexPlanBatchIter {
            inner: Some(inner),
            lookahead,
            iter_metrics,
            cache_metrics,
        })
    }

    /// Effective LRU shard cache size after auto-tuning to fit
    /// `max_memory_mb`. May be less than the user-requested `cache_shards`.
    fn effective_cache_shards(&self) -> usize {
        self.loader.effective_cache_shards()
    }

    /// Effective default lookahead after auto-tuning to fit `max_memory_mb`.
    /// May be less than the user-requested `lookahead`. Used by
    /// `iter_with_plans` when the caller does not pass an explicit override.
    fn effective_lookahead(&self) -> usize {
        self.loader.effective_lookahead()
    }

    /// Snapshot of the underlying `BackedCsrReader`'s shard-cache counters,
    /// cumulative since dataset construction. Returns a dict with keys:
    ///
    /// ```text
    /// hits                - cache lookups served without decode
    /// misses              - decodes that ran (each shard's first-leader)
    /// evictions           - LRU entries dropped to honour the byte/count cap
    /// bytes_inserted      - cumulative decoded bytes inserted into the cache
    /// duplicate_waiters   - waiters that found a peer leader in flight and
    ///                       skipped redundant decode work
    /// peak_bytes_in_cache - high-water mark of the cache's resident byte
    ///                       budget; useful for sizing `max_memory_mb`
    /// ```
    ///
    /// All values are `int`. Counters are atomic and read with `Relaxed`
    /// ordering. Sample as often as you want — there are no locks involved.
    fn cache_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        cache_metrics_to_pydict(py, &self.loader.cache_metrics())
    }

    /// Per-component memory breakdown estimated at construction. Returns a
    /// dict with the keys:
    ///
    /// ```text
    /// cache_bytes              - decoded shard cache budget
    /// batch_buffer_bytes       - dense X / X_paired buffers
    /// lookahead_overhead_bytes - plan-tuple staging in the iter
    /// transient_bytes          - per-batch obs Vecs + PairRequest scratch
    /// python_overhead_bytes    - constant Python/Arrow/numpy overhead
    /// total_bytes              - sum of the above
    /// max_memory_mb            - user-supplied budget (LoaderConfig)
    /// effective_cache_shards   - post-auto-tune cache count cap
    /// effective_lookahead      - post-auto-tune iter lookahead
    /// ```
    ///
    /// All byte values are `int`. Mirrors `TrainingDataset.memory_budget()`'s
    /// `breakdown` sub-dict shape, plus the index-plan-specific
    /// `effective_*` fields.
    fn memory_budget<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = self.loader.budget_breakdown().to_pydict(py)?;
        dict.set_item("max_memory_mb", self.loader.max_memory_mb())?;
        dict.set_item(
            "effective_cache_shards",
            self.loader.effective_cache_shards(),
        )?;
        dict.set_item("effective_lookahead", self.loader.effective_lookahead())?;
        Ok(dict)
    }

    /// Process one plan and return a paired dense batch dict.
    ///
    /// **Unstable / debug helper.** Superseded by `iter_with_plans`. Retained
    /// so unit tests can drive the loader synchronously without iterator
    /// setup; do not depend on this method from production code — it may be
    /// removed in a future release.
    #[pyo3(name = "_next_batch_for_test")]
    fn next_batch_for_test<'py>(
        &self,
        py: Python<'py>,
        plan: Vec<(u64, u64)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if std::process::id() != self.creation_pid {
            return Err(PyRuntimeError::new_err(
                "scx.IndexPlanDataset requires num_workers=0 (or lazy per-worker \
                 construction). The Rust shard cache and mmap state are not fork-safe.",
            ));
        }

        let loader = Arc::clone(&self.loader);
        let batch = py
            .detach(move || loader.process_plan(plan))
            .map_err(loader_err_to_py)?;

        index_plan_batch_to_dict(py, batch)
    }

    fn __repr__(&self) -> String {
        format!(
            "IndexPlanDataset(n_obs={}, n_vars={}, n_output_genes={})",
            self.loader.n_obs(),
            self.loader.n_vars(),
            self.loader.n_output_cols(),
        )
    }
}

/// Adapter: Python iterator → Rust `Iterator<Item = Result<Vec<(u64,u64)>, _>>`.
///
/// `Py<PyAny>` is `Send` + `Sync`, so this struct can cross the thread boundary
/// to the plan-pull worker. Each `next` reacquires the GIL just for the
/// `__next__` call so the GIL is freely available between pulls.
struct PyPlanIterator {
    py_iter: Py<PyAny>,
}

impl Iterator for PyPlanIterator {
    type Item = std::result::Result<Vec<(u64, u64)>, LoaderError>;

    fn next(&mut self) -> Option<Self::Item> {
        Python::attach(|py| {
            let bound = self.py_iter.bind(py);
            match bound.call_method0("__next__") {
                Ok(obj) => match obj.extract::<Vec<(u64, u64)>>() {
                    Ok(plan) => Some(Ok(plan)),
                    Err(e) => Some(Err(LoaderError::ConfigError {
                        reason: format!("plan extraction failed: {e}"),
                    })),
                },
                Err(e) => {
                    if e.is_instance_of::<PyStopIteration>(py) {
                        None
                    } else {
                        // Forward the Python exception text. We can't pass
                        // a PyErr through to the consumer's Result type, so
                        // wrap as a ChannelError carrying the message.
                        Some(Err(LoaderError::ChannelError(format!(
                            "plan iterator raised: {e}"
                        ))))
                    }
                }
            }
        })
    }
}

/// Python iterator that wraps an [`IndexPlanIter`] and converts each yielded
/// `IndexPlanBatch` into the standard pyscx batch dict.
#[pyclass]
pub struct IndexPlanBatchIter {
    inner: Option<IndexPlanIter>,
    #[allow(dead_code)] // surfaced via __repr__ / future docs
    lookahead: usize,
    /// Snapshot of the iter's prefetch counters, cloned at construction so
    /// the Python `metrics()` accessor stays valid after the iterator is
    /// drained and `inner` is dropped.
    iter_metrics: Arc<IterMetrics>,
    /// Loader-level cache counters, cloned at construction for the same
    /// post-drain stability.
    cache_metrics: Arc<CacheMetrics>,
}

#[pymethods]
impl IndexPlanBatchIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(inner) = self.inner.as_mut() else {
            return Ok(None);
        };
        // Release the GIL for both the prefetch await and the decode work,
        // so the plan-pull worker can call __next__ on the user iterator
        // without contention.
        let next = py.detach(|| inner.next());
        match next {
            Some(Ok(batch)) => Ok(Some(index_plan_batch_to_dict(py, batch)?)),
            Some(Err(e)) => Err(loader_err_to_py(e)),
            None => {
                // Drop the inner iterator to release the plan-pull thread
                // and tokio runtime references; subsequent next calls return
                // None without re-entering Rust.
                self.inner = None;
                Ok(None)
            }
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "IndexPlanBatchIter(lookahead={}, exhausted={})",
            self.lookahead,
            self.inner.is_none()
        )
    }

    /// Snapshot of cache- and prefetch-side counters as a dict-of-dicts:
    ///
    /// ```text
    /// {"cache": {hits, misses, evictions, bytes_inserted, duplicate_waiters,
    ///            peak_bytes_in_cache},
    ///  "prefetch": {prefetch_tasks_spawned,
    ///               prefetch_skipped_cache_hit,
    ///               prefetch_skipped_in_flight}}
    /// ```
    ///
    /// `cache` reflects loader-cumulative counters (shared with
    /// `IndexPlanDataset.cache_metrics()`). `prefetch` is per-iter — counters
    /// reset across `iter_with_plans` calls. Safe to call after the iterator
    /// is exhausted; the metrics handles are cloned at construction time.
    fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("cache", cache_metrics_to_pydict(py, &self.cache_metrics)?)?;
        dict.set_item("prefetch", iter_metrics_to_pydict(py, &self.iter_metrics)?)?;
        Ok(dict)
    }
}

/// Encode `CacheMetrics` (atomic counters from `BackedCsrReader`) as a Python
/// dict of `int` keys. Counters are loaded with `Relaxed` ordering — values
/// are statistical and not used for synchronization on the Python side.
fn cache_metrics_to_pydict<'py>(py: Python<'py>, m: &CacheMetrics) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("hits", m.hits.load(Ordering::Relaxed))?;
    dict.set_item("misses", m.misses.load(Ordering::Relaxed))?;
    dict.set_item("evictions", m.evictions.load(Ordering::Relaxed))?;
    dict.set_item("bytes_inserted", m.bytes_inserted.load(Ordering::Relaxed))?;
    dict.set_item(
        "duplicate_waiters",
        m.duplicate_waiters.load(Ordering::Relaxed),
    )?;
    dict.set_item(
        "peak_bytes_in_cache",
        m.peak_bytes_in_cache.load(Ordering::Relaxed),
    )?;
    dict.set_item("sidecar_groups", m.sidecar_groups.load(Ordering::Relaxed))?;
    dict.set_item(
        "full_shard_groups",
        m.full_shard_groups.load(Ordering::Relaxed),
    )?;
    dict.set_item(
        "block_index_groups",
        m.block_index_groups.load(Ordering::Relaxed),
    )?;
    Ok(dict)
}

/// Encode `IterMetrics` (per-iter prefetch counters) as a Python dict.
fn iter_metrics_to_pydict<'py>(py: Python<'py>, m: &IterMetrics) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        "prefetch_tasks_spawned",
        m.prefetch_tasks_spawned.load(Ordering::Relaxed),
    )?;
    dict.set_item(
        "prefetch_skipped_cache_hit",
        m.prefetch_skipped_cache_hit.load(Ordering::Relaxed),
    )?;
    dict.set_item(
        "prefetch_skipped_in_flight",
        m.prefetch_skipped_in_flight.load(Ordering::Relaxed),
    )?;
    dict.set_item(
        "prefetch_skipped_sidecar",
        m.prefetch_skipped_sidecar.load(Ordering::Relaxed),
    )?;
    Ok(dict)
}

/// Map `LoaderError` → Python exception, picking the most precise type.
fn loader_err_to_py(err: LoaderError) -> PyErr {
    match err {
        LoaderError::IndexOutOfRange { .. } => PyIndexError::new_err(err.to_string()),
        LoaderError::ObsColumnNotFound { .. } => PyKeyError::new_err(err.to_string()),
        _ => PyRuntimeError::new_err(err.to_string()),
    }
}

/// Encode a `HashMap<String, ObsColumn>` into a Python dict using the same
/// schema as `TrainingDataset`'s batch dict (numeric → ndarray; categorical →
/// `{"codes": ndarray, "categories": list[str]}`).
fn obs_to_pydict<'py>(
    py: Python<'py>,
    obs: HashMap<String, ObsColumn>,
) -> PyResult<Bound<'py, PyDict>> {
    let obs_dict = PyDict::new(py);
    for (name, column) in obs {
        match column {
            ObsColumn::Int64(values) => {
                let arr = PyArray1::from_vec(py, values);
                obs_dict.set_item(&name, arr)?;
            }
            ObsColumn::Float64(values) => {
                let arr = PyArray1::from_vec(py, values);
                obs_dict.set_item(&name, arr)?;
            }
            ObsColumn::Categorical(codes, categories) => {
                let cat_dict = PyDict::new(py);
                let codes_i32: Vec<i32> = codes.into_iter().map(|c| c as i32).collect();
                let codes_arr = PyArray1::from_vec(py, codes_i32);
                cat_dict.set_item("codes", codes_arr)?;
                let cat_list = PyList::new(py, categories.iter()).map_err(|e| {
                    PyRuntimeError::new_err(format!("failed to create category list: {e}"))
                })?;
                cat_dict.set_item("categories", cat_list)?;
                obs_dict.set_item(&name, cat_dict)?;
            }
        }
    }
    Ok(obs_dict)
}

/// Phase H.1 / H.3: resolve the optional `modality` kwarg to a
/// `modality_id: Option<u8>` for the loader pipeline.
///
/// Behaviour:
/// - Single-modality file + no `modality` arg → returns `None` (legacy
///   global path).
/// - Single-modality file + `modality` arg → error (the file has no
///   such modality).
/// - Multimodal file + explicit `modality` name → resolves the name;
///   error if not found.
/// - Multimodal file + no `modality` arg → emits `UserWarning` and
///   falls back to the alphabetically-first modality name.
///
/// Opens the SCX file briefly to inspect the modality table; the
/// inner `TrainingPipeline::new` re-opens it so this transient open
/// is cheap and serves only the modality resolution.
fn resolve_modality_id_with_warning(
    py: Python<'_>,
    path: &str,
    modality: Option<&str>,
) -> PyResult<Option<u8>> {
    use scx_format_io::ScxReader;
    let reader = ScxReader::open(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    if !reader.is_multimodal() {
        if let Some(name) = modality {
            return Err(PyRuntimeError::new_err(format!(
                "TrainingDataset: file '{path}' is single-modality but \
                 `modality='{name}'` was passed; remove the kwarg, or \
                 use a multimodal source."
            )));
        }
        return Ok(None);
    }

    if let Some(name) = modality {
        return match reader.modality_id(name) {
            Some(mid) => Ok(Some(mid)),
            None => {
                let names: Vec<&str> = reader.modality_names();
                Err(PyRuntimeError::new_err(format!(
                    "TrainingDataset: file '{path}' does not have a \
                     modality named '{name}'. Registered modalities: {names:?}"
                )))
            }
        };
    }

    // No modality arg on a multimodal file: fall back to the
    // alphabetically-first modality and emit a UserWarning telling
    // the user to use MultimodalTrainingDataset for full coverage.
    let mut names: Vec<String> = reader
        .modality_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    names.sort();
    let chosen = names
        .first()
        .ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "TrainingDataset: file '{path}' is flagged multimodal but \
                 the modality table is empty"
            ))
        })?
        .clone();
    let mid = reader.modality_id(&chosen).ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "TrainingDataset: failed to resolve modality_id for '{chosen}'"
        ))
    })?;

    let warnings = py.import("warnings")?;
    let user_warning = py.import("builtins")?.getattr("UserWarning")?;
    let msg = format!(
        "file is multimodal; loading modality '{chosen}' only — use \
         MultimodalTrainingDataset for full coverage"
    );
    warnings.call_method1("warn", (msg, user_warning))?;
    Ok(Some(mid))
}

/// Convert an `IndexPlanBatch` into the Python dict shape:
/// `{"X", "X_paired", "pairs", "obs", "obs_paired"}`.
fn index_plan_batch_to_dict<'py>(
    py: Python<'py>,
    batch: IndexPlanBatch,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);

    let n_pairs = batch.n_pairs();
    let n_cols = batch.x.len().checked_div(n_pairs).unwrap_or(0);

    let x_array = PyArray1::from_vec(py, batch.x)
        .reshape([n_pairs, n_cols])
        .map_err(|e| PyRuntimeError::new_err(format!("failed to reshape X array: {e}")))?;
    dict.set_item("X", x_array)?;

    let xp_array = PyArray1::from_vec(py, batch.x_paired)
        .reshape([n_pairs, n_cols])
        .map_err(|e| PyRuntimeError::new_err(format!("failed to reshape X_paired array: {e}")))?;
    dict.set_item("X_paired", xp_array)?;

    let pairs_list = PyList::empty(py);
    for (p, c) in &batch.pairs {
        let tup = PyTuple::new(py, [*p, *c])
            .map_err(|e| PyRuntimeError::new_err(format!("failed to build pair tuple: {e}")))?;
        pairs_list.append(tup)?;
    }
    dict.set_item("pairs", pairs_list)?;

    dict.set_item("obs", obs_to_pydict(py, batch.obs)?)?;
    dict.set_item("obs_paired", obs_to_pydict(py, batch.obs_paired)?)?;

    Ok(dict)
}

/// Convert a `Batch` into a Python dict: `{"X": ndarray, "obs": {...}, "cell_indices": ndarray}`.
///
/// Uses `PyArray::from_vec()` for zero-copy transfer of Rust Vecs to numpy arrays.
fn batch_to_dict<'py>(py: Python<'py>, batch: Batch) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);

    // X: dense expression matrix [n_rows × n_genes], dtype float32
    let (n_rows, n_genes) = batch.x_shape;
    let x_array = PyArray1::from_vec(py, batch.x)
        .reshape([n_rows, n_genes])
        .map_err(|e| PyRuntimeError::new_err(format!("failed to reshape X array: {e}")))?;
    dict.set_item("X", x_array)?;

    dict.set_item("obs", obs_to_pydict(py, batch.obs)?)?;

    // cell_indices: u64 → i64 for numpy compatibility
    let cell_indices_i64: Vec<i64> = batch
        .cell_indices
        .into_iter()
        .map(|idx| idx as i64)
        .collect();
    let cell_arr = PyArray1::from_vec(py, cell_indices_i64);
    dict.set_item("cell_indices", cell_arr)?;

    Ok(dict)
}

// ---------------------------------------------------------------------------
// SparseCellSetDataset — native sparse cell-set loader (SCX-DATA-LOADER §4)
// ---------------------------------------------------------------------------

/// Adapter: Python iterator of `(file_ids, rows, role_tags, set_offsets)`
/// tuples → Rust `Iterator<Item = Result<SparseCellSetPlan>>`. Mirrors
/// [`PyPlanIterator`]: `Py<PyAny>` is `Send`, and each `next` reacquires the
/// GIL only for the `__next__` call.
struct PySparseCellSetPlanIterator {
    py_iter: Py<PyAny>,
}

impl Iterator for PySparseCellSetPlanIterator {
    type Item = std::result::Result<SparseCellSetPlan, LoaderError>;

    fn next(&mut self) -> Option<Self::Item> {
        Python::attach(|py| {
            let bound = self.py_iter.bind(py);
            match bound.call_method0("__next__") {
                Ok(obj) => match obj.extract::<(Vec<u32>, Vec<u64>, Vec<i32>, Vec<i64>)>() {
                    Ok((file_ids, rows, role_tags, set_offsets)) => Some(Ok(SparseCellSetPlan {
                        file_ids,
                        rows,
                        role_tags,
                        set_offsets,
                    })),
                    Err(e) => Some(Err(LoaderError::ConfigError {
                        reason: format!(
                            "sparse plan extraction failed (expected a tuple \
                             (file_ids:u32[], rows:u64[], role_tags:i32[], set_offsets:i64[])): {e}"
                        ),
                    })),
                },
                Err(e) => {
                    if e.is_instance_of::<PyStopIteration>(py) {
                        None
                    } else {
                        Some(Err(LoaderError::ChannelError(format!(
                            "plan iterator raised: {e}"
                        ))))
                    }
                }
            }
        })
    }
}

/// Plan-driven native sparse cell-set reader. Each plan item is one batch of
/// cell sets (delimited by `set_offsets`); each yielded dict carries the
/// SCX-DATA-LOADER §4.4 sparse contract. Sibling to `IndexPlanDataset` but
/// emits sparse CSR (not dense pairs) and is multi-file.
#[pyclass]
pub struct SparseCellSetDataset {
    loader: Arc<SparseCellSetLoader>,
    default_lookahead: usize,
    /// PID at construction — the shard cache / mmap state is not fork-safe, so
    /// the dataset must be built post-fork in each worker (or `num_workers=0`).
    creation_pid: u32,
}

#[pymethods]
impl SparseCellSetDataset {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        paths,
        cache_shards=None,
        max_memory_mb=None,
        lookahead=None,
        remap_tables=None,
        n_global_genes=None,
        normalize=None,
        log1p=None,
        target_sum=None,
        scatter_sidecar=None,
    ))]
    fn new(
        paths: Vec<String>,
        cache_shards: Option<usize>,
        max_memory_mb: Option<usize>,
        lookahead: Option<usize>,
        remap_tables: Option<Vec<Vec<i32>>>,
        n_global_genes: Option<usize>,
        normalize: Option<bool>,
        log1p: Option<bool>,
        target_sum: Option<f64>,
        scatter_sidecar: Option<bool>,
    ) -> PyResult<Self> {
        if paths.is_empty() {
            return Err(PyRuntimeError::new_err(
                "SparseCellSetDataset requires at least one .scx path",
            ));
        }
        let cache_shards = cache_shards.unwrap_or(128);
        let lookahead = lookahead.unwrap_or(4);
        let bytes_budget = max_memory_mb
            .map(|mb| mb.saturating_mul(1024 * 1024))
            .unwrap_or(usize::MAX);
        let normalize = normalize.unwrap_or(false);
        let log1p = log1p.unwrap_or(false);
        let target_sum = target_sum.unwrap_or(1e4);
        // Default OFF: the cache-friendly sparse cell-set workload (sorted data
        // + control-pool cache → high shard reuse) wins by decoding each hot
        // shard once into the LRU rather than re-decoding it via the per-row
        // sidecar every batch. See SCX-CACHE-SHARDS.md §5.0 (Phase 0 measured
        // 2.80 → 4.63 steps/s). Pass `scatter_sidecar=True` to opt back into the
        // sidecar gather for genuinely cache-hostile (working-set ≫ cache) runs.
        let scatter_sidecar = scatter_sidecar.unwrap_or(false);

        let mut readers = Vec::with_capacity(paths.len());
        for p in &paths {
            readers.push(
                ScxReader::open(p)
                    .map_err(|e| PyRuntimeError::new_err(format!("failed to open {p}: {e}")))?,
            );
        }

        let loader = SparseCellSetLoader::new(
            readers,
            cache_shards,
            bytes_budget,
            lookahead,
            remap_tables,
            n_global_genes,
            normalize,
            log1p,
            target_sum,
            scatter_sidecar,
        )
        .map_err(loader_err_to_py)?;

        Ok(Self {
            loader,
            default_lookahead: lookahead,
            creation_pid: std::process::id(),
        })
    }

    /// Number of `.scx` files (the `file_id` range).
    #[getter]
    fn n_files(&self) -> usize {
        self.loader.n_files()
    }

    /// CSR column count of emitted batches (max per-file `n_vars`, or the
    /// global vocab size when remap tables were supplied).
    #[getter]
    fn n_cols(&self) -> usize {
        self.loader.n_cols()
    }

    /// Drive the loader from a Python iterable of batch plans, each a tuple
    /// `(file_ids, rows, role_tags, set_offsets)`; return an iterator of
    /// sparse batch dicts (the §4.4 contract).
    #[pyo3(signature = (plans, lookahead=None))]
    fn iter_with_plans(
        &self,
        plans: Py<PyAny>,
        lookahead: Option<usize>,
    ) -> PyResult<SparseCellSetBatchIter> {
        if std::process::id() != self.creation_pid {
            return Err(PyRuntimeError::new_err(
                "scx.SparseCellSetDataset requires num_workers=0 (or lazy per-worker \
                 construction). The Rust shard cache and mmap state are not fork-safe.",
            ));
        }
        let lookahead = lookahead.unwrap_or(self.default_lookahead);

        let py_iter: Py<PyAny> = Python::attach(|py| -> PyResult<Py<PyAny>> {
            Ok(plans.bind(py).call_method0("__iter__")?.unbind())
        })?;

        let plan_stream = PySparseCellSetPlanIterator { py_iter };
        let inner = Arc::clone(&self.loader).iter_with_plans(plan_stream, lookahead);
        Ok(SparseCellSetBatchIter { inner: Some(inner) })
    }

    /// Snapshot of the readers' shared shard-cache counters, cumulative since
    /// construction (the multi-file sibling of `IndexPlanDataset.cache_metrics`).
    /// Returns a dict with keys: `hits`, `misses` (= shard decodes), `evictions`,
    /// `bytes_inserted`, `duplicate_waiters`, `peak_bytes_in_cache`. All `int`;
    /// atomic, lock-free — sample as often as you like.
    fn cache_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        cache_metrics_to_pydict(py, &self.loader.cache_metrics())
    }

    fn __repr__(&self) -> String {
        format!(
            "SparseCellSetDataset(n_files={}, n_cols={})",
            self.loader.n_files(),
            self.loader.n_cols()
        )
    }
}

/// Python iterator wrapping the boxed engine iterator; converts each
/// `SparseCellSetBatch` into the §4.4 dict.
#[pyclass]
pub struct SparseCellSetBatchIter {
    inner: Option<Box<dyn Iterator<Item = crate::error::Result<SparseCellSetBatch>> + Send + Sync>>,
}

#[pymethods]
impl SparseCellSetBatchIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(inner) = self.inner.as_mut() else {
            return Ok(None);
        };
        // Release the GIL for the prefetch await + gather so the plan-pull
        // worker can call __next__ on the user iterator without contention.
        let next = py.detach(|| inner.next());
        match next {
            Some(Ok(batch)) => Ok(Some(sparse_cellset_batch_to_dict(py, batch)?)),
            Some(Err(e)) => Err(loader_err_to_py(e)),
            None => {
                self.inner = None;
                Ok(None)
            }
        }
    }

    fn __repr__(&self) -> String {
        format!("SparseCellSetBatchIter(exhausted={})", self.inner.is_none())
    }
}

/// Convert a `SparseCellSetBatch` into the §4.4 dict: `{indptr, indices, data,
/// shape, cell_indices, file_ids, set_offsets, role_tags}`.
fn sparse_cellset_batch_to_dict<'py>(
    py: Python<'py>,
    batch: SparseCellSetBatch,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("indptr", PyArray1::from_vec(py, batch.indptr))?;
    dict.set_item("indices", PyArray1::from_vec(py, batch.indices))?;
    dict.set_item("data", PyArray1::from_vec(py, batch.data))?;
    dict.set_item("shape", PyTuple::new(py, [batch.shape.0, batch.shape.1])?)?;
    dict.set_item("cell_indices", PyArray1::from_vec(py, batch.cell_indices))?;
    dict.set_item("file_ids", PyArray1::from_vec(py, batch.file_ids))?;
    dict.set_item("set_offsets", PyArray1::from_vec(py, batch.set_offsets))?;
    dict.set_item("role_tags", PyArray1::from_vec(py, batch.role_tags))?;
    Ok(dict)
}

/// Convert a `CollatedCellSetBatch` into a dict of flat stacked tensors plus
/// shape scalars (`n_rows`, `k_enc`, `k_dec`); state3 reshapes to `[B, S, K]`.
fn collated_cellset_batch_to_dict<'py>(
    py: Python<'py>,
    batch: CollatedCellSetBatch,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        "encoder_gene_ids",
        PyArray1::from_vec(py, batch.encoder_gene_ids),
    )?;
    dict.set_item(
        "encoder_counts",
        PyArray1::from_vec(py, batch.encoder_counts),
    )?;
    dict.set_item("encoder_mask", PyArray1::from_vec(py, batch.encoder_mask))?;
    dict.set_item(
        "encoder_pad_mask",
        PyArray1::from_vec(py, batch.encoder_pad_mask),
    )?;
    dict.set_item("target_counts", PyArray1::from_vec(py, batch.target_counts))?;
    dict.set_item("library_size", PyArray1::from_vec(py, batch.library_size))?;
    dict.set_item("cell_indices", PyArray1::from_vec(py, batch.cell_indices))?;
    dict.set_item("file_ids", PyArray1::from_vec(py, batch.file_ids))?;
    dict.set_item("set_offsets", PyArray1::from_vec(py, batch.set_offsets))?;
    dict.set_item("role_tags", PyArray1::from_vec(py, batch.role_tags))?;
    dict.set_item("n_rows", batch.n_rows)?;
    dict.set_item("k_enc", batch.k_enc)?;
    dict.set_item("k_dec", batch.k_dec)?;
    Ok(dict)
}

/// Collate an already-gathered, **global-vocab** CSR batch into stacked tensors
/// (state3 "3A hybrid"). Pure compute; releases the GIL. Python gathers (via
/// `iter_with_plans`) and samples the query, then collates here. `set_offsets`
/// delimits the sets; `enc_mask_positions` may be empty (perturbation path).
#[pyfunction]
#[pyo3(signature = (
    indptr, indices, data, set_offsets, cell_indices, file_ids, role_tags,
    k_dec, query_gene_ids, enc_mask_positions, hide_readout, n_measured,
    k_enc, mode, n_genes_total, target_sum=None, lib_size_redef=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn collate_cellset_gathered<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i32>,
    data: PyReadonlyArray1<'py, f32>,
    set_offsets: PyReadonlyArray1<'py, i64>,
    cell_indices: PyReadonlyArray1<'py, u64>,
    file_ids: PyReadonlyArray1<'py, u32>,
    role_tags: PyReadonlyArray1<'py, i32>,
    k_dec: usize,
    query_gene_ids: PyReadonlyArray1<'py, i32>,
    enc_mask_positions: PyReadonlyArray1<'py, u8>,
    hide_readout: PyReadonlyArray1<'py, u8>,
    n_measured: PyReadonlyArray1<'py, u32>,
    k_enc: usize,
    mode: String,
    n_genes_total: i64,
    target_sum: Option<f64>,
    lib_size_redef: Option<bool>,
) -> PyResult<Bound<'py, PyDict>> {
    let mode = PreprocessMode::parse(&mode).map_err(loader_err_to_py)?;
    let scalars = CollateScalars {
        k_enc,
        mode,
        target_sum: target_sum.unwrap_or(1e4),
        n_genes_total,
        lib_size_redef: lib_size_redef.unwrap_or(false),
    };
    let err = |e| PyRuntimeError::new_err(format!("array not contiguous: {e}"));
    let indptr = indptr.as_slice().map_err(err)?;
    let indices = indices.as_slice().map_err(err)?;
    let data = data.as_slice().map_err(err)?;
    let set_offsets = set_offsets.as_slice().map_err(err)?;
    let cell_indices_v = cell_indices.as_slice().map_err(err)?.to_vec();
    let file_ids_v = file_ids.as_slice().map_err(err)?.to_vec();
    let role_tags_v = role_tags.as_slice().map_err(err)?.to_vec();
    let query = query_gene_ids.as_slice().map_err(err)?;
    let encmask = enc_mask_positions.as_slice().map_err(err)?;
    let hide = hide_readout.as_slice().map_err(err)?;
    let nmeas = n_measured.as_slice().map_err(err)?;
    let batch = py
        .detach(|| {
            crate::sparse_cellset::collate_gathered(
                indptr,
                indices,
                data,
                set_offsets,
                cell_indices_v,
                file_ids_v,
                role_tags_v,
                k_dec,
                query,
                encmask,
                hide,
                nmeas,
                &scalars,
            )
        })
        .map_err(loader_err_to_py)?;
    collated_cellset_batch_to_dict(py, batch)
}
