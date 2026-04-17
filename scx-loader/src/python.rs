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

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::batch::{Batch, ObsColumn};
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

    // obs: dict of metadata columns
    let obs_dict = PyDict::new(py);
    for (name, column) in batch.obs {
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
                // Codes as int32 (u32 → i32 for numpy compatibility)
                let codes_i32: Vec<i32> = codes.into_iter().map(|c| c as i32).collect();
                let codes_arr = PyArray1::from_vec(py, codes_i32);
                cat_dict.set_item("codes", codes_arr)?;
                // Categories as Python list of strings
                let cat_list = PyList::new(py, &categories).map_err(|e| {
                    PyRuntimeError::new_err(format!("failed to create category list: {e}"))
                })?;
                cat_dict.set_item("categories", cat_list)?;
                obs_dict.set_item(&name, cat_dict)?;
            }
        }
    }
    dict.set_item("obs", obs_dict)?;

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
