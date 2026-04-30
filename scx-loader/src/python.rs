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
use std::sync::Arc;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::batch::{Batch, ObsColumn};
use crate::error::LoaderError;
use crate::index_plan::{IndexPlanBatch, IndexPlanLoader};
use crate::pipeline::{LoaderConfig, TrainingPipeline};

/// A PyTorch-compatible iterable dataset for SCX training data.
///
/// Wraps the Rust `TrainingPipeline` and exposes it as a Python iterator.
/// Each call to `__next__` returns a dict `{"X": ndarray, "obs": {...}, "cell_indices": ndarray}`.
///
/// **Important**: Must be used with `num_workers=0` in `torch.utils.data.DataLoader`.
/// The Rust pipeline manages its own threads; forking after CUDA initialization
/// causes deadlocks.
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
    ///     log1p: Apply log1p transformation (default: True).
    ///     target_sum: Normalization target sum (default: 1e4).
    ///     shard_group_size: Shards per I/O group (default: 8).
    ///     prefetch_batches: Ring buffer depth (default: 4).
    ///     seed: RNG seed for reproducibility (default: 42).
    ///     max_memory_mb: Memory budget in MB (default: 512).
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
        shard_group_size=None,
        prefetch_batches=None,
        seed=None,
        max_memory_mb=None,
    ))]
    fn new(
        path: &str,
        batch_size: Option<usize>,
        hvg_indices: Option<Vec<u32>>,
        obs_columns: Option<Vec<String>>,
        normalize: Option<bool>,
        log1p: Option<bool>,
        target_sum: Option<f64>,
        shard_group_size: Option<usize>,
        prefetch_batches: Option<usize>,
        seed: Option<u64>,
        max_memory_mb: Option<usize>,
    ) -> PyResult<Self> {
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
            seed: seed.unwrap_or(defaults.seed),
            max_memory_mb: max_memory_mb.unwrap_or(defaults.max_memory_mb),
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
        let batch_opt = py.allow_threads(|| self.pipeline.next_batch());

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
    fn memory_budget<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let mb = self.pipeline.memory_budget_info();
        dict.set_item("shard_group_size", mb.shard_group_size)?;
        dict.set_item("prefetch_batches", mb.prefetch_batches)?;
        dict.set_item("batch_size", mb.batch_size)?;
        dict.set_item("estimated_mb", mb.estimated_bytes / (1024 * 1024))?;
        dict.set_item("mmap_mb", mb.mmap_bytes / (1024 * 1024))?;
        dict.set_item("budget_exceeded", mb.budget_exceeded)?;
        Ok(dict)
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

/// Plan-driven paired-batch reader for `(perturbed, control)` ML training.
///
/// Sibling to `TrainingDataset`: instead of streaming shards in catalog order,
/// `IndexPlanDataset` consumes caller-supplied `(pert_idx, ctrl_idx)` plans
/// and returns paired dense batches. See `PER-CELL-CONTROL-PAIRING.md` at the
/// workspace root for the full design.
///
/// Phase 1 surface: `next_batch(plan)` is the only batch entry point. The
/// streaming `iter_with_plans` API lands in Phase 4.
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
    ///     log1p: Reserved (default: True). Currently fused with `normalize`
    ///         on this path, mirroring `TrainingDataset`.
    ///     target_sum: Normalization target sum (default: 1e4).
    ///     cache_shards: LRU shard cache budget (default: 128). Must be >= 1.
    ///     max_memory_mb: Memory budget in MB (default: 512). Currently
    ///         informational on this path; auto-tuning lands in Phase 5.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        path,
        hvg_indices=None,
        obs_columns=None,
        normalize=None,
        log1p=None,
        target_sum=None,
        cache_shards=None,
        max_memory_mb=None,
    ))]
    fn new(
        path: &str,
        hvg_indices: Option<Vec<u32>>,
        obs_columns: Option<Vec<String>>,
        normalize: Option<bool>,
        log1p: Option<bool>,
        target_sum: Option<f64>,
        cache_shards: Option<usize>,
        max_memory_mb: Option<usize>,
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
        if let Some(v) = max_memory_mb {
            config.max_memory_mb = v;
        }

        let cache_shards = cache_shards.unwrap_or(128);

        let loader = IndexPlanLoader::new(path, config, cache_shards).map_err(loader_err_to_py)?;

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

    /// Process one plan and return a paired dense batch dict.
    ///
    /// Returns: `{"X": ndarray[B, G], "X_paired": ndarray[B, G],
    /// "pairs": list[tuple[int, int]], "obs": {...}, "obs_paired": {...}}`.
    ///
    /// Phase 1 stepping-stone API; the iterator surface (`iter_with_plans`)
    /// lands in Phase 4.
    fn next_batch<'py>(
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
            .allow_threads(move || loader.process_plan(plan))
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

/// Map `LoaderError` → Python exception, picking the most precise type.
fn loader_err_to_py(err: LoaderError) -> PyErr {
    match err {
        LoaderError::IndexOutOfRange { .. } => PyIndexError::new_err(err.to_string()),
        LoaderError::ConfigError { ref reason } if reason.contains("not found in RecordBatch") => {
            PyKeyError::new_err(err.to_string())
        }
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
                let cat_list = PyList::new(py, &categories).map_err(|e| {
                    PyRuntimeError::new_err(format!("failed to create category list: {e}"))
                })?;
                cat_dict.set_item("categories", cat_list)?;
                obs_dict.set_item(&name, cat_dict)?;
            }
        }
    }
    Ok(obs_dict)
}

/// Convert an `IndexPlanBatch` into the Python dict shape:
/// `{"X", "X_paired", "pairs", "obs", "obs_paired"}`.
fn index_plan_batch_to_dict<'py>(
    py: Python<'py>,
    batch: IndexPlanBatch,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);

    let n_pairs = batch.n_pairs();
    let n_cols = if n_pairs == 0 {
        0
    } else {
        batch.x.len() / n_pairs
    };

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
        let tup = PyTuple::new(py, [*p, *c]).map_err(|e| {
            PyRuntimeError::new_err(format!("failed to build pair tuple: {e}"))
        })?;
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
