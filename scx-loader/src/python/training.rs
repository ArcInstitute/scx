//! Sequential `TrainingDataset` and its batch conversion.
//!
//! Split out of `python.rs` by ORG-9.10-2; a pure move.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::batch::Batch;
use crate::pipeline::{resolve_loader_config, LoaderConfig, TrainingPipeline};

use super::*;

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
/// See the `normalize` / `log1p` / `target_sum` / `pflog` constructor args
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
/// fork-hostile thread state from the parent. The same holds for the
/// process-wide [`crate::pool::cpu_pool`] the constructor uses for `read_obs`
/// and the PFlog α estimate: it is keyed on the PID, so a child never draws on
/// a pool the parent built. Note that this means construction *does* create
/// worker threads — the guarantee is that they are the child's own, not that
/// there are none.
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
/// interpreter is still healthy. `Drop` runs on garbage-collection /
/// interpreter teardown as a fallback — bounded by a 5-second deadline per
/// thread to avoid hangs, and it **releases the GIL for the duration of the
/// join** (like `close()`) so a slow shutdown never freezes other Python
/// threads. Still prefer `close()` for prompt, deterministic teardown.
#[pyclass]
pub struct TrainingDataset {
    pipeline: TrainingPipeline,
    epoch_started: bool,
    /// `close()` was the last lifecycle action — see the `closed` getter. Not a
    /// terminal state on this class: `__iter__` clears it and rebuilds.
    closed: bool,
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
    ///         Every index must be < n_vars (the selected modality's
    ///         n_vars when `modality=` or the multimodal fallback is in
    ///         play); an out-of-range index is rejected at construction
    ///         rather than becoming an always-zero output column.
    ///         Sorted and deduplicated, so batch columns are in ascending
    ///         gene-index order regardless of the order passed and duplicates
    ///         shrink the batch (see `n_output_genes`); a panel that is not
    ///         already ascending-unique gets a UserWarning.
    ///     obs_columns: Obs metadata column names to include in each batch.
    ///     normalize: Apply total-count normalization (default: True).
    ///         NOTE: this is **on by default** — batches are normalized even
    ///         though the file stores raw counts. Pass `normalize=False`
    ///         (with `log1p=False`) for raw-count output (scVI / count
    ///         likelihoods). See "Default transforms" on the class docstring.
    ///     log1p: Apply log1p transformation (default: True). On by default;
    ///         pass `log1p=False` to disable. See `normalize`.
    ///     target_sum: Normalization target sum (default: 1e4).
    ///     pflog: Apply PFlog (v4) / shifted-log normalization on raw counts
    ///         (Booeshaghi et al.) instead of normalize/log1p (default: False).
    ///         PFlog is itself a normalization, so it is mutually exclusive with
    ///         `normalize`/`log1p`: when True it takes precedence and those flags
    ///         are ignored. The centering denominator is computed over the full
    ///         transcriptome even under `hvg_indices` projection.
    ///     pflog_alpha: PFlog NB overdispersion `α` (matrix-wide pseudocount
    ///         1/(4α); only used when `pflog=True`). None (default) estimates α
    ///         once at construction from the dataset's raw counts (single-modality
    ///         only; a modality-scoped loader must pin it); a float pins a
    ///         reference α (e.g. from `pyscx.accel.pflog`).
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
        pflog=None,
        pflog_alpha=None,
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
        pflog: Option<bool>,
        pflog_alpha: Option<f64>,
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

        // No explicit budget → adaptive (treat the default as a floor and raise
        // to fit a full-width file). An explicit budget is a hard ceiling
        // (preserves the auto-tune-down + warning behaviour).
        let config = resolve_loader_config(
            batch_size,
            shard_group_size,
            prefetch_batches,
            normalize,
            log1p,
            target_sum,
            pflog,
            pflog_alpha,
            seed,
            hvg_indices,
            obs_columns.unwrap_or_default(),
            max_memory_mb.unwrap_or_else(|| LoaderConfig::default().max_memory_mb),
            max_memory_mb.is_none(),
            modality_id,
            // One panel, one modality, one unambiguous `n_vars` — even when
            // `modality_id` is set by `modality=` or the implicit
            // alphabetically-first fallback. Range-check it.
            /*shared_hvg_panel=*/
            false,
        );

        let pipeline = TrainingPipeline::new(path, config)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        if let Some(v) = pipeline.hvg_panel() {
            warn_hvg_panel(py, "TrainingDataset", &v)?;
        }
        if pipeline.memory_budget_info().budget_exceeded {
            warn_budget_exceeded(
                py,
                "TrainingDataset",
                pipeline.max_memory_mb(),
                pipeline.memory_budget_info(),
            )?;
        }

        Ok(TrainingDataset {
            pipeline,
            epoch_started: false,
            closed: false,
            creation_pid: std::process::id(),
        })
    }

    /// Start a new epoch. Called automatically by `for batch in dataset`.
    ///
    /// Legal after `close()` — that is the difference from the two plan-driven
    /// classes, whose `close()` is terminal. `start_epoch` rebuilds the rayon
    /// pool through `ensure_decode_pool`, so this clears `closed`.
    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        slf.pipeline
            .start_epoch()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        slf.epoch_started = true;
        slf.closed = false;
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
        let batch_res = py.detach(|| self.pipeline.next_batch());

        match batch_res {
            Ok(Some(batch)) => {
                let dict = batch_to_dict(py, batch)?;
                Ok(Some(dict))
            }
            Ok(None) => {
                self.epoch_started = false;
                Ok(None) // StopIteration
            }
            Err(e) => {
                // A mid-epoch I/O/decode fault: end iteration and surface the
                // error rather than silently truncating the epoch.
                self.epoch_started = false;
                Err(loader_err_to_py(e))
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
        self.closed = true;
    }

    /// True from [`Self::close`] until the next `__iter__` rebuilds. Never raises.
    ///
    /// Deliberately **not** the terminal flag `IndexPlanDataset.closed` is.
    /// `close()` on this class releases the rayon pool and joins the epoch
    /// threads, and the next `__iter__` builds them again — so the honest
    /// reading is "torn down right now", not "unusable from here on". It cannot
    /// be derived from pipeline state either: a freshly constructed dataset has
    /// no pool yet and would report `True` before it had ever been closed.
    #[getter]
    fn closed(&self) -> bool {
        self.closed
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

impl Drop for TrainingDataset {
    /// Release the GIL around the pipeline's bounded thread joins on drop.
    ///
    /// A `#[pyclass]` is dropped with the GIL held, so `TrainingPipeline`'s
    /// `Drop` (which bound-joins the I/O + decode threads, up to
    /// `~2×SHUTDOWN_DEADLINE`) would otherwise stall every other Python thread.
    /// Mirror `close()` and detach the GIL around the join. `py.detach` only
    /// wraps the pure-Rust join — it never calls *into* Python — so this is
    /// safe even mid-finalization, where the finalizing thread holds the GIL
    /// while destructors run.
    ///
    /// `Py_IsInitialized` guards the case where we are dropped *after* the
    /// interpreter has fully finalized (no GIL to acquire): fall back to an
    /// in-place shutdown, preserving `TrainingPipeline::drop`'s teardown-safety
    /// contract. (`Py_IsFinalizing` would be the tighter predicate but is
    /// `Py_3_13`-gated, so it can't be used on the supported 3.11 baseline.)
    fn drop(&mut self) {
        if unsafe { pyo3::ffi::Py_IsInitialized() } != 0 {
            Python::attach(|py| py.detach(|| self.pipeline.shutdown()));
        } else {
            self.pipeline.shutdown();
        }
    }
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

    let warnings = crate::pyimport::import_module(py, "warnings")?;
    let user_warning = crate::pyimport::import_module(py, "builtins")?.getattr("UserWarning")?;
    let msg = format!(
        "file is multimodal; loading modality '{chosen}' only — use \
         MultimodalTrainingDataset for full coverage"
    );
    warnings.call_method1("warn", (msg, user_warning))?;
    Ok(Some(mid))
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
