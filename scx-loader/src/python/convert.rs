//! Shared `->` `PyDict` converters for obs and the metrics dicts.
//!
//! Split out of `python.rs` by ORG-9.10-2; a pure move.

use std::collections::HashMap;
use std::sync::atomic::Ordering;

use numpy::PyArray1;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use scx_format_io::CacheMetrics;

use crate::batch::ObsColumn;
use crate::plan_engine::IterMetrics;

/// Add the reader-registry counters to an existing `cache_metrics()` dict.
///
/// A separate function, and a separate call, rather than four more lines inside
/// [`cache_metrics_to_pydict`]: that one is shared by five Python surfaces,
/// including the single-file `IndexPlanDataset`, which has no registry and
/// would gain four keys that are always zero. A key that can only ever read
/// zero is worse than no key — it invites a reader to conclude the registry
/// never opened anything.
pub(super) fn reader_metrics_into(
    dict: &Bound<'_, PyDict>,
    m: &crate::reader_registry::ReaderMetrics,
) -> PyResult<()> {
    dict.set_item("reader_opens", m.opens.load(Ordering::Relaxed))?;
    dict.set_item("reader_evictions", m.evictions.load(Ordering::Relaxed))?;
    dict.set_item("reader_resident", m.resident.load(Ordering::Relaxed))?;
    dict.set_item("reader_hwm", m.hwm.load(Ordering::Relaxed))?;
    Ok(())
}

/// Encode `CacheMetrics` (atomic counters from `BackedCsrReader`) as a Python
/// dict of `int` keys. Counters are loaded with `Relaxed` ordering — values
/// are statistical and not used for synchronization on the Python side.
pub(super) fn cache_metrics_to_pydict<'py>(
    py: Python<'py>,
    m: &CacheMetrics,
) -> PyResult<Bound<'py, PyDict>> {
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
    dict.set_item(
        "full_shard_groups",
        m.full_shard_groups.load(Ordering::Relaxed),
    )?;
    dict.set_item(
        "block_index_groups",
        m.block_index_groups.load(Ordering::Relaxed),
    )?;
    // The row-group LRU entries a framed scattered read retains
    // (OPT-FORMATIO-1). Separate from `hits` / `misses`, which keep meaning
    // the whole-shard LRU.
    dict.set_item("row_group_hits", m.row_group_hits.load(Ordering::Relaxed))?;
    dict.set_item(
        "row_group_misses",
        m.row_group_misses.load(Ordering::Relaxed),
    )?;
    dict.set_item(
        "row_group_evictions",
        m.row_group_evictions.load(Ordering::Relaxed),
    )?;
    dict.set_item(
        "row_group_bytes_inserted",
        m.row_group_bytes_inserted.load(Ordering::Relaxed),
    )?;
    dict.set_item(
        "row_group_duplicate_waiters",
        m.row_group_duplicate_waiters.load(Ordering::Relaxed),
    )?;
    Ok(dict)
}

/// Encode `IterMetrics` (per-iter prefetch counters) as a Python dict.
pub(super) fn iter_metrics_to_pydict<'py>(
    py: Python<'py>,
    m: &IterMetrics,
) -> PyResult<Bound<'py, PyDict>> {
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
        "prefetch_skipped_block_index",
        m.prefetch_skipped_block_index.load(Ordering::Relaxed),
    )?;
    Ok(dict)
}

/// Encode a `HashMap<String, ObsColumn>` into a Python dict using the same
/// schema as `TrainingDataset`'s batch dict (numeric → ndarray; categorical →
/// `{"codes": ndarray, "categories": list[str]}`).
pub(super) fn obs_to_pydict<'py>(
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
