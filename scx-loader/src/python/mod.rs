//! Python bindings for the SCX training data loader.
//!
//! Split out of a single 3384-line `python.rs` by ORG-9.10-2 (Phase 9f). This
//! module keeps only what every child needs — the shared imports and the
//! `LoaderError` → Python exception mapping. Each child owns one binding:
//!
//! * [`training`] — `TrainingDataset`, the sequential loader, and its batch dict.
//! * [`multimodal`] — `MultimodalTrainingDataset` and the per-modality budget split.
//! * [`index_plan`] — `IndexPlanDataset` / `IndexPlanBatchIter` (paired plans).
//! * [`sparse_cellset`] — `SparseCellSetDataset` / `SparseCellSetBatchIter`.
//! * [`collate`] — the three free `#[pyfunction]`s.
//! * [`plan_iter`] — the Python-iterator adapter both plan loaders share.
//! * [`diagnostics`] — the `UserWarning` layer and the cache-thrash sampler.
//! * [`convert`] — the obs / metrics `PyDict` converters.
//!
//! The `pub use` globs below keep every registration path unchanged: the eight
//! names `lib.rs` re-exports still resolve at `scx_loader::<name>`, so
//! `pyscx/src/lib.rs` is untouched by the split.
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

use pyo3::exceptions::{PyIndexError, PyKeyError, PyRuntimeError};
use pyo3::prelude::*;

use crate::error::LoaderError;

// `pub use` on the five children that define the names `lib.rs` re-exports.
// The other three hold `pub(super)` items only — siblings reach them through
// their own `use super::*`, so a plain private re-export is the right
// visibility; `pub(crate) use` cannot re-export a `pub(super)` item and only
// earns a "doesn't reexport anything" warning.
pub mod collate;
pub mod convert;
pub mod diagnostics;
pub mod index_plan;
pub mod multimodal;
pub mod plan_iter;
pub mod sparse_cellset;
pub mod training;

pub use collate::*;
use convert::*;
use diagnostics::*;
pub use index_plan::*;
pub use multimodal::*;
use plan_iter::*;
pub use sparse_cellset::*;
pub use training::*;

/// Map `LoaderError` → Python exception, picking the most precise type.
fn loader_err_to_py(err: LoaderError) -> PyErr {
    match err {
        LoaderError::IndexOutOfRange { .. } => PyIndexError::new_err(err.to_string()),
        LoaderError::ObsColumnNotFound { .. } => PyKeyError::new_err(err.to_string()),
        _ => PyRuntimeError::new_err(err.to_string()),
    }
}
