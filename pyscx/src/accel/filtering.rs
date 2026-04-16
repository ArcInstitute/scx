//! Cell/gene filtering without materialization — filter_cells, filter_genes, subset_obs.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::{ScxBackedLayerDataset, ScxBackedSparseDataset};
use crate::lazy_transform::ScxLazyTransformedDataset;
use crate::projected_agg;

// ---------------------------------------------------------------------------
// Backed aggregation helpers
// ---------------------------------------------------------------------------

/// Helper: compute row NNZ for a backed dataset (respecting col_projection + deletions).
fn backed_row_nnz(backed: &ScxBackedSparseDataset) -> PyResult<Vec<i64>> {
    let all_nnz = if let Some(cols) = backed.col_projection() {
        projected_agg::row_nnz_projected(&backed.backed, cols)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    } else {
        backed
            .backed
            .row_nnz()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    };
    Ok(backed.filter_row_results(&all_nnz))
}

/// Helper: compute row sums for a backed dataset (respecting col_projection + deletions).
fn backed_row_sums(backed: &ScxBackedSparseDataset) -> PyResult<Vec<f64>> {
    let all_sums = if let Some(cols) = backed.col_projection() {
        projected_agg::row_sums_projected(&backed.backed, cols)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    } else {
        backed
            .backed
            .row_sums()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    };
    Ok(backed.filter_row_results(&all_sums))
}

/// Helper: compute row NNZ and sums in a single shard scan (fused).
///
/// Avoids the double I/O of `backed_row_nnz()` + `backed_row_sums()` when
/// `filter_cells` needs both `min_genes` and `min_counts`.
fn backed_row_nnz_and_sums(backed: &ScxBackedSparseDataset) -> PyResult<(Vec<i64>, Vec<f64>)> {
    let (all_nnz, all_sums) = if let Some(cols) = backed.col_projection() {
        projected_agg::row_nnz_and_sums_projected(&backed.backed, cols)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    } else {
        backed
            .backed
            .row_nnz_and_sums()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    };
    Ok((
        backed.filter_row_results(&all_nnz),
        backed.filter_row_results(&all_sums),
    ))
}

/// Helper: compute col NNZ for a backed dataset (4-way dispatch).
fn backed_col_nnz(backed: &ScxBackedSparseDataset) -> PyResult<Vec<i64>> {
    let counts = match (backed.col_projection(), &backed.kept_to_global) {
        (Some(cols), Some(kept)) => {
            projected_agg::col_nnz_masked_projected(&backed.backed, kept, cols)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        }
        (Some(cols), None) => projected_agg::col_nnz_projected(&backed.backed, cols)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        (None, Some(kept)) => {
            let f_counts = backed
                .backed
                .col_nnz_masked(kept)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            f_counts.iter().map(|&v| v as i64).collect()
        }
        (None, None) => backed
            .backed
            .col_nnz()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    };
    Ok(counts)
}

/// Helper: compute col sums for a backed dataset (4-way dispatch).
fn backed_col_sums(backed: &ScxBackedSparseDataset) -> PyResult<Vec<f64>> {
    let sums = match (backed.col_projection(), &backed.kept_to_global) {
        (Some(cols), Some(kept)) => {
            projected_agg::col_sums_masked_projected(&backed.backed, kept, cols)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        }
        (Some(cols), None) => projected_agg::col_sums_projected(&backed.backed, cols)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        (None, Some(kept)) => backed
            .backed
            .col_sums_masked(kept)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        (None, None) => backed
            .backed
            .col_sums()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    };
    Ok(sums)
}

// ---------------------------------------------------------------------------
// Shared mask/projection helpers (used by hvg.rs too)
// ---------------------------------------------------------------------------

/// Build a boolean keep-mask from optional min/max thresholds on two metrics.
pub(super) fn build_keep_mask(
    n: usize,
    nnz: Option<&[i64]>,
    sums: Option<&[f64]>,
    min_nnz: Option<i64>,
    max_nnz: Option<i64>,
    min_sum: Option<f64>,
    max_sum: Option<f64>,
) -> Vec<bool> {
    let mut keep = vec![true; n];
    if let Some(min_v) = min_nnz {
        if let Some(vals) = nnz {
            for (i, &v) in vals.iter().enumerate() {
                if v < min_v {
                    keep[i] = false;
                }
            }
        }
    }
    if let Some(max_v) = max_nnz {
        if let Some(vals) = nnz {
            for (i, &v) in vals.iter().enumerate() {
                if v > max_v {
                    keep[i] = false;
                }
            }
        }
    }
    if let Some(min_v) = min_sum {
        if let Some(vals) = sums {
            for (i, &v) in vals.iter().enumerate() {
                if v < min_v {
                    keep[i] = false;
                }
            }
        }
    }
    if let Some(max_v) = max_sum {
        if let Some(vals) = sums {
            for (i, &v) in vals.iter().enumerate() {
                if v > max_v {
                    keep[i] = false;
                }
            }
        }
    }
    keep
}

/// Compose a new deletion vector from a boolean mask and an existing kept_to_global.
pub(super) fn compose_kept_to_global(keep: &[bool], existing: Option<&[u64]>) -> Vec<u64> {
    match existing {
        Some(existing) => keep
            .iter()
            .enumerate()
            .filter(|(_, &k)| k)
            .map(|(i, _)| existing[i])
            .collect(),
        None => keep
            .iter()
            .enumerate()
            .filter(|(_, &k)| k)
            .map(|(i, _)| i as u64)
            .collect(),
    }
}

/// Slice `adata.obs` and `adata.obsm` to match a boolean keep-mask.
pub(super) fn slice_obs_and_obsm<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    keep: &[bool],
) -> PyResult<()> {
    let np = py.import("numpy")?;
    let mask_arr = numpy::PyArray::from_vec(py, keep.to_vec());

    // Slice obs — use _obs to bypass anndata's shape validation
    // (X.shape is already updated via set_kept_to_global before this call)
    let obs = adata.getattr("obs")?;
    let filtered_obs = obs.getattr("loc")?.get_item(&mask_arr)?;
    adata.setattr("_obs", filtered_obs)?;

    // Slice obsm entries
    let obsm = adata.getattr("obsm")?;
    // obsm may be empty or a dict-like; get keys safely
    let keys_result: PyResult<Vec<String>> = obsm
        .call_method0("keys")?
        .try_iter()?
        .map(|k| k.and_then(|k| k.extract()))
        .collect();
    if let Ok(keys) = keys_result {
        // Build numpy boolean array for indexing
        let np_mask = np.call_method1("array", (mask_arr,))?;
        for key in &keys {
            let arr = obsm.get_item(key)?;
            let sliced = arr.get_item(&np_mask)?;
            obsm.set_item(key, sliced)?;
        }
    }

    Ok(())
}

/// Update all backed layers with a new deletion vector.
pub(super) fn update_layers_kept_to_global(
    adata: &Bound<'_, PyAny>,
    new_kept: &[u64],
) -> PyResult<()> {
    let layers = adata.getattr("layers")?;
    let keys_result: PyResult<Vec<String>> = layers
        .call_method0("keys")?
        .try_iter()?
        .map(|k| k.and_then(|k| k.extract()))
        .collect();
    if let Ok(keys) = keys_result {
        for key in &keys {
            let layer_obj = layers.get_item(key)?;
            if let Ok(layer) = layer_obj.downcast::<ScxBackedLayerDataset>() {
                layer
                    .borrow_mut()
                    .inner
                    .set_kept_to_global(new_kept.to_vec());
            }
            // ScxLazyTransformedDataset layers are unlikely but handle them
            if let Ok(lazy_layer) = layer_obj.downcast::<ScxLazyTransformedDataset>() {
                lazy_layer
                    .borrow_mut()
                    .set_kept_to_global(new_kept.to_vec());
            }
        }
    }
    Ok(())
}

/// Update all backed layers with a new column projection.
pub(super) fn update_layers_col_projection(
    adata: &Bound<'_, PyAny>,
    new_cols: &[u32],
) -> PyResult<()> {
    let layers = adata.getattr("layers")?;
    let keys_result: PyResult<Vec<String>> = layers
        .call_method0("keys")?
        .try_iter()?
        .map(|k| k.and_then(|k| k.extract()))
        .collect();
    if let Ok(keys) = keys_result {
        for key in &keys {
            let layer_obj = layers.get_item(key)?;
            if let Ok(layer) = layer_obj.downcast::<ScxBackedLayerDataset>() {
                layer
                    .borrow_mut()
                    .inner
                    .set_col_projection(new_cols.to_vec());
            }
            if let Ok(lazy_layer) = layer_obj.downcast::<ScxLazyTransformedDataset>() {
                lazy_layer
                    .borrow_mut()
                    .set_col_projection(new_cols.to_vec());
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// filter_cells
// ---------------------------------------------------------------------------

/// Filter cells (rows) without materializing backed data.
///
/// Replacement for `sc.pp.filter_cells()` that works on backed and
/// lazy-transformed SCX data. Computes row metrics via streaming,
/// builds a boolean mask, and updates the deletion vector on X and
/// layers. Also slices `adata.obs` and `adata.obsm` to match.
///
/// Falls back to `sc.pp.filter_cells()` for regular scipy/dense matrices.
///
/// Args:
///     adata: AnnData object
///     min_genes: Minimum number of genes expressed (row NNZ >= threshold)
///     max_genes: Maximum number of genes expressed (row NNZ <= threshold)
///     min_counts: Minimum total counts per cell (row sum >= threshold)
///     max_counts: Maximum total counts per cell (row sum <= threshold)
#[pyfunction]
#[pyo3(signature = (adata, min_genes=None, max_genes=None, min_counts=None, max_counts=None))]
pub fn filter_cells(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    min_genes: Option<i64>,
    max_genes: Option<i64>,
    min_counts: Option<f64>,
    max_counts: Option<f64>,
) -> PyResult<()> {
    if min_genes.is_none() && max_genes.is_none() && min_counts.is_none() && max_counts.is_none() {
        return Ok(());
    }

    let x = adata.getattr("X")?;
    let need_nnz = min_genes.is_some() || max_genes.is_some();
    let need_sums = min_counts.is_some() || max_counts.is_some();

    // Case 1: X is ScxBackedSparseDataset
    if let Ok(backed) = x.downcast::<ScxBackedSparseDataset>() {
        let n_obs = backed.borrow().shape_val.0;
        // Fused: compute both NNZ and sums in a single shard scan when both are needed
        let (row_nnz, row_sums) = if need_nnz && need_sums {
            let (nnz, sums) = backed_row_nnz_and_sums(&backed.borrow())?;
            (Some(nnz), Some(sums))
        } else {
            (
                if need_nnz {
                    Some(backed_row_nnz(&backed.borrow())?)
                } else {
                    None
                },
                if need_sums {
                    Some(backed_row_sums(&backed.borrow())?)
                } else {
                    None
                },
            )
        };

        let keep = build_keep_mask(
            n_obs,
            row_nnz.as_deref(),
            row_sums.as_deref(),
            min_genes,
            max_genes,
            min_counts,
            max_counts,
        );

        let new_kept = compose_kept_to_global(
            &keep,
            backed
                .borrow()
                .kept_to_global
                .as_ref()
                .map(|v| v.as_slice()),
        );

        backed.borrow_mut().set_kept_to_global(new_kept.clone());
        slice_obs_and_obsm(py, adata, &keep)?;
        update_layers_kept_to_global(adata, &new_kept)?;
        return Ok(());
    }

    // Case 2: X is ScxLazyTransformedDataset
    if let Ok(lazy) = x.downcast::<ScxLazyTransformedDataset>() {
        let lazy_ref = lazy.borrow();
        let n_obs = lazy_ref.shape_val.0;

        // Fused: when both NNZ and sums are needed, compute them in a single
        // shard scan. NNZ is transform-invariant but we compute it from the
        // same decoded shard to avoid double I/O.
        let (row_nnz, row_sums) = if need_nnz && need_sums {
            let (all_nnz, all_sums) = lazy_ref.streaming_row_nnz_and_sums()?;
            (
                Some(lazy_ref.filter_row_results(&all_nnz)),
                Some(lazy_ref.filter_row_results(&all_sums)),
            )
        } else {
            let nnz = if need_nnz {
                let all_nnz = lazy_ref
                    .backed
                    .row_nnz()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Some(lazy_ref.filter_row_results(&all_nnz))
            } else {
                None
            };
            let sums = if need_sums {
                let all_sums = lazy_ref.streaming_row_sums()?;
                Some(lazy_ref.filter_row_results(&all_sums))
            } else {
                None
            };
            (nnz, sums)
        };

        let keep = build_keep_mask(
            n_obs,
            row_nnz.as_deref(),
            row_sums.as_deref(),
            min_genes,
            max_genes,
            min_counts,
            max_counts,
        );

        let new_kept = compose_kept_to_global(
            &keep,
            lazy_ref.kept_to_global.as_ref().map(|v| v.as_slice()),
        );

        drop(lazy_ref);
        lazy.borrow_mut().set_kept_to_global(new_kept.clone());
        slice_obs_and_obsm(py, adata, &keep)?;
        update_layers_kept_to_global(adata, &new_kept)?;
        return Ok(());
    }

    // Case 3: fallback to scanpy
    let sc = py.import("scanpy")?;
    let kwargs = PyDict::new(py);
    if let Some(v) = min_genes {
        kwargs.set_item("min_genes", v)?;
    }
    if let Some(v) = max_genes {
        kwargs.set_item("max_genes", v)?;
    }
    if let Some(v) = min_counts {
        kwargs.set_item("min_counts", v)?;
    }
    if let Some(v) = max_counts {
        kwargs.set_item("max_counts", v)?;
    }
    sc.getattr("pp")?
        .call_method("filter_cells", (adata,), Some(&kwargs))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// filter_genes
// ---------------------------------------------------------------------------

/// Filter genes (columns) without materializing backed data.
///
/// Replacement for `sc.pp.filter_genes()` that works on backed and
/// lazy-transformed SCX data. Computes column metrics via streaming,
/// builds a boolean mask, and sets `col_projection` on X and layers.
/// Also slices `adata.var` to match.
///
/// Falls back to `sc.pp.filter_genes()` for regular scipy/dense matrices.
///
/// **Note:** After `filter_genes`, column projection is active. Streaming
/// PCA via `as_shard_source()` applies the projection per-shard.
///
/// Args:
///     adata: AnnData object
///     min_cells: Minimum number of cells expressing gene (col NNZ >= threshold)
///     max_cells: Maximum number of cells expressing gene (col NNZ <= threshold)
///     min_counts: Minimum total counts per gene (col sum >= threshold)
///     max_counts: Maximum total counts per gene (col sum <= threshold)
#[pyfunction]
#[pyo3(signature = (adata, min_cells=None, max_cells=None, min_counts=None, max_counts=None))]
pub fn filter_genes(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    min_cells: Option<i64>,
    max_cells: Option<i64>,
    min_counts: Option<f64>,
    max_counts: Option<f64>,
) -> PyResult<()> {
    if min_cells.is_none() && max_cells.is_none() && min_counts.is_none() && max_counts.is_none() {
        return Ok(());
    }

    let x = adata.getattr("X")?;
    let need_nnz = min_cells.is_some() || max_cells.is_some();
    let need_sums = min_counts.is_some() || max_counts.is_some();

    // Case 1: X is ScxBackedSparseDataset
    if let Ok(backed) = x.downcast::<ScxBackedSparseDataset>() {
        let n_vars = backed.borrow().shape_val.1;
        let col_nnz = if need_nnz {
            Some(backed_col_nnz(&backed.borrow())?)
        } else {
            None
        };
        let col_sums = if need_sums {
            Some(backed_col_sums(&backed.borrow())?)
        } else {
            None
        };

        let keep = build_keep_mask(
            n_vars,
            col_nnz.as_deref(),
            col_sums.as_deref(),
            min_cells,
            max_cells,
            min_counts,
            max_counts,
        );

        // Compose with existing col_projection
        let new_col_indices: Vec<u32> = match backed.borrow().col_projection() {
            Some(existing) => keep
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| existing[i])
                .collect(),
            None => keep
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| i as u32)
                .collect(),
        };

        // Slice var BEFORE updating col_projection (AnnData validates
        // shape consistency on .var setter, so we use ._var to bypass).
        let mask_arr = numpy::PyArray::from_vec(py, keep);
        let var = adata.getattr("var")?;
        let filtered_var = var.getattr("loc")?.get_item(&mask_arr)?;
        // Use _var to skip shape validation (X.shape changes next)
        adata.setattr("_var", filtered_var)?;

        backed
            .borrow_mut()
            .set_col_projection(new_col_indices.clone());

        update_layers_col_projection(adata, &new_col_indices)?;
        return Ok(());
    }

    // Case 2: X is ScxLazyTransformedDataset
    if let Ok(lazy) = x.downcast::<ScxLazyTransformedDataset>() {
        let lazy_ref = lazy.borrow();
        let n_vars = lazy_ref.shape_val.1;

        // For lazy datasets, NNZ is transform-invariant: use underlying backed reader
        // with the lazy dataset's col_projection and kept_to_global
        let col_nnz = if need_nnz {
            let counts = match (lazy_ref.col_projection(), &lazy_ref.kept_to_global) {
                (Some(cols), Some(kept)) => {
                    projected_agg::col_nnz_masked_projected(&lazy_ref.backed, kept, cols)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                }
                (Some(cols), None) => projected_agg::col_nnz_projected(&lazy_ref.backed, cols)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                (None, Some(kept)) => {
                    let f_counts = lazy_ref
                        .backed
                        .col_nnz_masked(kept)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    f_counts.iter().map(|&v| v as i64).collect()
                }
                (None, None) => lazy_ref
                    .backed
                    .col_nnz()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            };
            Some(counts)
        } else {
            None
        };

        // Col sums through transforms (streaming).
        // streaming_col_sums returns full-width (all original columns).
        // When col_projection is active, extract only projected columns.
        let col_sums = if need_sums {
            let full_sums = if lazy_ref.kept_to_global.is_some() {
                lazy_ref.streaming_col_sums_masked()?
            } else {
                lazy_ref.streaming_col_sums()?
            };
            if let Some(cols) = lazy_ref.col_projection() {
                Some(
                    cols.iter()
                        .map(|&c| full_sums[c as usize])
                        .collect::<Vec<f64>>(),
                )
            } else {
                Some(full_sums)
            }
        } else {
            None
        };

        let keep = build_keep_mask(
            n_vars,
            col_nnz.as_deref(),
            col_sums.as_deref(),
            min_cells,
            max_cells,
            min_counts,
            max_counts,
        );

        let new_col_indices: Vec<u32> = match lazy_ref.col_projection() {
            Some(existing) => keep
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| existing[i])
                .collect(),
            None => keep
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| i as u32)
                .collect(),
        };

        drop(lazy_ref);

        // Slice var BEFORE updating col_projection (AnnData validates
        // shape consistency on .var setter, so we use ._var to bypass).
        let mask_arr = numpy::PyArray::from_vec(py, keep);
        let var = adata.getattr("var")?;
        let filtered_var = var.getattr("loc")?.get_item(&mask_arr)?;
        adata.setattr("_var", filtered_var)?;

        lazy.borrow_mut()
            .set_col_projection(new_col_indices.clone());

        update_layers_col_projection(adata, &new_col_indices)?;
        return Ok(());
    }

    // Case 3: fallback to scanpy
    let sc = py.import("scanpy")?;
    let kwargs = PyDict::new(py);
    if let Some(v) = min_cells {
        kwargs.set_item("min_cells", v)?;
    }
    if let Some(v) = max_cells {
        kwargs.set_item("max_cells", v)?;
    }
    if let Some(v) = min_counts {
        kwargs.set_item("min_counts", v)?;
    }
    if let Some(v) = max_counts {
        kwargs.set_item("max_counts", v)?;
    }
    sc.getattr("pp")?
        .call_method("filter_genes", (adata,), Some(&kwargs))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// subset_obs
// ---------------------------------------------------------------------------

/// Subset observations (rows) without materializing backed data.
///
/// Accepts a boolean mask (numpy array or list of bool) or integer index
/// array (numpy array or list of int) and updates the backed dataset's
/// deletion vector in-place. Also slices `adata.obs`, `adata.obsm`, and
/// updates all backed layers to match.
///
/// This is the general-purpose version of `filter_cells()` — use it for
/// custom QC logic, cluster-based filtering, or any arbitrary subsetting:
///
///     mask = adata.obs['doublet_score'] < 0.5
///     pyscx.accel.subset_obs(adata, mask)
///
///     # Or with integer indices (treated as set membership):
///     indices = [0, 5, 10, 15, 20]
///     pyscx.accel.subset_obs(adata, indices)
///
/// **Note:** Integer indices are converted to a boolean mask internally,
/// so they behave as set membership rather than ordered selection:
/// duplicate indices are silently collapsed, and the original order is
/// not preserved (`[5, 0, 10]` produces the same result as `[0, 5, 10]`).
/// This differs from NumPy fancy indexing. Use a boolean mask if you need
/// exact control over which rows are kept.
///
/// Falls back to standard numpy/pandas subsetting for non-SCX data
/// (materializes X via `adata._X = adata.X[mask]`).
///
/// Args:
///     adata: AnnData object
///     mask_or_indices: Boolean mask (length n_obs) or integer index array
#[pyfunction]
#[pyo3(signature = (adata, mask_or_indices))]
pub fn subset_obs(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    mask_or_indices: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let np = py.import("numpy")?;

    // Get n_obs from adata.shape[0]
    let shape = adata.getattr("shape")?;
    let n_obs: usize = shape.get_item(0)?.extract()?;

    // Convert input to a numpy array to determine dtype
    let arr = np.call_method1("asarray", (mask_or_indices,))?;
    let dtype_str: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;

    // Build boolean keep mask
    let keep: Vec<bool> = match dtype_str.as_str() {
        // Boolean mask
        "b" => {
            let bool_arr: numpy::PyReadonlyArray1<'_, bool> = arr.extract()?;
            let slice = bool_arr
                .as_slice()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            if slice.len() != n_obs {
                return Err(PyValueError::new_err(format!(
                    "Boolean mask length ({}) does not match n_obs ({})",
                    slice.len(),
                    n_obs
                )));
            }
            slice.to_vec()
        }
        // Integer indices
        "i" | "u" => {
            let idx_arr = arr.call_method1("astype", (np.getattr("int64")?,))?;
            let idx_ro: numpy::PyReadonlyArray1<'_, i64> = idx_arr.extract()?;
            let indices = idx_ro
                .as_slice()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            // Validate bounds
            for &idx in indices {
                if idx < 0 || (idx as usize) >= n_obs {
                    return Err(PyValueError::new_err(format!(
                        "Index {} is out of bounds for n_obs={}",
                        idx, n_obs
                    )));
                }
            }

            // Build boolean mask from indices
            let mut mask = vec![false; n_obs];
            for &idx in indices {
                mask[idx as usize] = true;
            }
            mask
        }
        _ => {
            return Err(PyValueError::new_err(
                "mask_or_indices must be a boolean mask or integer index array",
            ));
        }
    };

    // Check if any rows are kept
    let n_kept = keep.iter().filter(|&&k| k).count();
    if n_kept == n_obs {
        // All rows kept — nothing to do
        return Ok(());
    }

    let x = adata.getattr("X")?;

    // Case 1: X is ScxBackedSparseDataset
    if let Ok(backed) = x.downcast::<ScxBackedSparseDataset>() {
        let new_kept = compose_kept_to_global(
            &keep,
            backed
                .borrow()
                .kept_to_global
                .as_ref()
                .map(|v| v.as_slice()),
        );
        backed.borrow_mut().set_kept_to_global(new_kept.clone());
        slice_obs_and_obsm(py, adata, &keep)?;
        update_layers_kept_to_global(adata, &new_kept)?;
        return Ok(());
    }

    // Case 2: X is ScxLazyTransformedDataset
    if let Ok(lazy) = x.downcast::<ScxLazyTransformedDataset>() {
        let new_kept = compose_kept_to_global(
            &keep,
            lazy.borrow().kept_to_global.as_ref().map(|v| v.as_slice()),
        );
        lazy.borrow_mut().set_kept_to_global(new_kept.clone());
        slice_obs_and_obsm(py, adata, &keep)?;
        update_layers_kept_to_global(adata, &new_kept)?;
        return Ok(());
    }

    // Case 3: Fallback for non-SCX data — materialize subset
    let mask_arr = numpy::PyArray::from_vec(py, keep.clone());
    let np_mask = np.call_method1("array", (mask_arr,))?;

    // Slice X
    let x_sliced = x.get_item(&np_mask)?;
    adata.setattr("_X", x_sliced)?;

    // Slice obs and obsm
    slice_obs_and_obsm(py, adata, &keep)?;

    Ok(())
}
