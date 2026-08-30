//! Python bindings for the SCX training data loader.
//!
//! Split out of a single 3384-line `python.rs` by ORG-9.10-2 (Phase 9f). This
//! module keeps only what every child needs — the shared imports and the
//! `LoaderError` → Python exception mapping. Each child owns one binding:
//!
//! * `training` — `TrainingDataset`, the sequential loader, and its batch dict.
//! * `multimodal` — `MultimodalTrainingDataset` and the per-modality budget split.
//! * `index_plan` — `IndexPlanDataset` / `IndexPlanBatchIter` (paired plans).
//! * `sparse_cellset` — `SparseCellSetDataset` / `SparseCellSetBatchIter`.
//! * `collate` — the three free `#[pyfunction]`s.
//! * `plan_iter` — the Python-iterator adapter both plan loaders share.
//! * `diagnostics` — the `UserWarning` layer and the cache-thrash sampler.
//! * `convert` — the obs / metrics `PyDict` converters.
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

// The children are private `mod`s: making them `pub` would expose the split's
// layout as a second set of API paths (`python::training::TrainingDataset`)
// alongside the `python::TrainingDataset` this module preserves. `pub use` on
// the five that define names `lib.rs` re-exports.
//
// A child that does `use super::*` must NOT also `use pyo3::prelude::*`: the
// glob already supplies it, and rustc 1.98 (what CI runs) rejects the duplicate
// as an unused import while 1.95 does not.
mod collate;
mod convert;
mod diagnostics;
mod index_plan;
mod multimodal;
mod plan_iter;
mod sparse_cellset;
mod training;

// Named, never globbed: a `pub use child::*` would silently publish any `pub`
// item a later edit adds to a child. These nine are exactly the pre-split public
// surface — the eight `lib.rs` re-exports plus `IndexPlanBatchIter`, which is
// reachable at `python::` but not at the crate root.
pub use collate::{collate_cellset_gathered, downsample_counts_csr, downsample_file_identity};
pub use index_plan::{IndexPlanBatchIter, IndexPlanDataset};
pub use multimodal::MultimodalTrainingDataset;
pub use sparse_cellset::{SparseCellSetBatchIter, SparseCellSetDataset};
pub use training::TrainingDataset;

// The remaining three hold `pub(super)` items only, reached by siblings through
// their own `use super::*`; a private re-export is the right visibility, and
// `pub(crate) use` cannot re-export a `pub(super)` item at all.
use convert::*;
use diagnostics::*;
use plan_iter::*;

/// Map `LoaderError` → Python exception, picking the most precise type.
fn loader_err_to_py(err: LoaderError) -> PyErr {
    match err {
        LoaderError::IndexOutOfRange { .. } => PyIndexError::new_err(err.to_string()),
        LoaderError::ObsColumnNotFound { .. } => PyKeyError::new_err(err.to_string()),
        _ => PyRuntimeError::new_err(err.to_string()),
    }
}
