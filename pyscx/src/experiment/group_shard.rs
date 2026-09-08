//! `GroupShard` — one shard of a grouped read — and the shared
//! stale-handle / engine error adapters.

// PyExperiment — lazy handle for SCX files

use std::sync::Arc;

use scx_engine::QueryPipeline;

use super::*;

/// The one-clause "why" out of a `FileChangedOnDisk`, for the repr.
///
/// The full message ends with the how-to-recover sentence, which is right for
/// an exception and noise inside `<Experiment '…' [stale: …]>`.
pub(crate) fn stale_repr_detail(e: &scx_format_io::ScxError) -> String {
    match e {
        scx_format_io::ScxError::FileChangedOnDisk { detail, .. } => detail.clone(),
        other => other.to_string(),
    }
}

/// Map an `scx_engine::EngineError` to the right Python exception for the
/// grouped-read API: unknown label → `KeyError` (with close matches in the
/// message), not-grouped → `ValueError`, everything else → `RuntimeError`.
///
/// **A second mapper, deliberately not merged with `query::engine_to_pyerr`.**
/// The two own disjoint variants — this one the grouped-read errors, that one
/// the predicate / modality / cast-refusal ones — and they have always disagreed
/// about the variants neither names (a `SchemaError` is a `ValueError` through a
/// query and a `RuntimeError` through a grouped read). Unifying them would flip
/// exception types on grouped reads, which no caller of this API has asked for;
/// worth doing on its own, not as a side effect.
pub(crate) fn engine_to_pyerr(e: scx_engine::EngineError) -> PyErr {
    use scx_engine::EngineError as E;
    match e {
        E::UnknownGroupLabel { .. } => pyo3::exceptions::PyKeyError::new_err(e.to_string()),
        E::NotGrouped => pyo3::exceptions::PyValueError::new_err(e.to_string()),
        other => pyo3::exceptions::PyRuntimeError::new_err(other.to_string()),
    }
}

/// F2: a non-reference shard's grouped contents, with deferred I/O.
///
/// Holds a shared `Arc<QueryPipeline>` (cloned from the parent `Experiment`), so
/// streaming over `iter_group_shards()` opens/parses the file once rather than
/// re-opening per shard.
#[pyclass(name = "GroupShard")]
pub struct PyGroupShard {
    pipeline: Arc<QueryPipeline>,
    handle: scx_engine::GroupShardHandle,
}

impl PyGroupShard {
    /// Construct from a shared pipeline + engine handle (Rust-only). Shared by
    /// the local and cloud `iter_group_shards` surfaces.
    pub(crate) fn new(pipeline: Arc<QueryPipeline>, handle: scx_engine::GroupShardHandle) -> Self {
        Self { pipeline, handle }
    }
}

#[pymethods]
impl PyGroupShard {
    /// This shard's index in the grouped layout.
    #[getter]
    fn shard_index(&self) -> u32 {
        self.handle.shard_index
    }

    /// First global output row in this shard (inclusive).
    #[getter]
    fn global_start(&self) -> u64 {
        self.handle.global_start
    }

    /// One past the last global output row in this shard (exclusive).
    #[getter]
    fn global_stop(&self) -> u64 {
        self.handle.global_stop
    }

    /// Labels present in this shard.
    #[getter]
    fn labels(&self) -> Vec<String> {
        self.handle
            .groups
            .iter()
            .map(|(l, _, _)| l.clone())
            .collect()
    }

    /// Per-label **shard-local** `(start, stop)` row ranges as a dict
    /// `{label: (start, stop)}` (offsets relative to this shard's start).
    #[getter]
    fn groups(&self) -> std::collections::HashMap<String, (u64, u64)> {
        self.handle
            .groups
            .iter()
            .map(|(l, ls, le)| (l.clone(), (*ls, *le)))
            .collect()
    }

    /// Read this shard's rows as an AnnData (deferred I/O — decodes only this
    /// shard's range).
    fn to_anndata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let pipeline = Arc::clone(&self.pipeline);
        let (start, stop) = (self.handle.global_start, self.handle.global_stop);
        let result = py
            .detach(|| pipeline.read_row_range(start, stop))
            .map_err(engine_to_pyerr)?;
        crate::query::query_result_to_anndata(py, result)
    }

    /// Read just the cells of `label` within this shard as an AnnData (deferred
    /// I/O — decodes only the label's sub-range). Raises `KeyError` if the label
    /// is not resident in this shard.
    fn read_group<'py>(&self, py: Python<'py>, label: &str) -> PyResult<Bound<'py, PyAny>> {
        let (start, stop) = self.handle.range(label).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "label '{label}' is not in shard {}",
                self.handle.shard_index
            ))
        })?;
        let pipeline = Arc::clone(&self.pipeline);
        let result = py
            .detach(|| pipeline.read_row_range(start, stop))
            .map_err(engine_to_pyerr)?;
        crate::query::query_result_to_anndata(py, result)
    }

    fn __repr__(&self) -> String {
        format!(
            "GroupShard(shard_index={}, rows={}..{}, labels={})",
            self.handle.shard_index,
            self.handle.global_start,
            self.handle.global_stop,
            self.handle.groups.len()
        )
    }
}
