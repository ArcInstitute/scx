//! Cell/gene filtering without materialization — filter_cells, filter_genes, subset_obs.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::axis_align::{subset_obs_axis, subset_var_axis};
use crate::backed::{detached, ScxBackedSparseDataset};
use crate::lazy_transform::ScxLazyTransformedDataset;
use crate::projected_agg;

// ---------------------------------------------------------------------------
// Backed aggregation helpers
// ---------------------------------------------------------------------------

/// Helper: compute row NNZ for a backed dataset (respecting col_projection + deletions).
///
/// Pure-Rust (`Result<_, String>`, no `PyErr`) so callers can run it through
/// `detached` with the GIL released.
fn backed_row_nnz(backed: &ScxBackedSparseDataset) -> Result<Vec<i64>, String> {
    let all_nnz = if let Some(cols) = backed.col_projection() {
        projected_agg::row_nnz_projected(&backed.backed, cols).map_err(|e| e.to_string())?
    } else {
        backed.backed.row_nnz().map_err(|e| e.to_string())?
    };
    Ok(backed.filter_row_results(&all_nnz))
}

/// Helper: compute row sums for a backed dataset (respecting col_projection + deletions).
///
/// Pure-Rust (`Result<_, String>`, no `PyErr`) so callers can run it through
/// `detached` with the GIL released.
fn backed_row_sums(backed: &ScxBackedSparseDataset) -> Result<Vec<f64>, String> {
    let all_sums = if let Some(cols) = backed.col_projection() {
        projected_agg::row_sums_projected(&backed.backed, cols).map_err(|e| e.to_string())?
    } else {
        backed.backed.row_sums().map_err(|e| e.to_string())?
    };
    Ok(backed.filter_row_results(&all_sums))
}

/// Helper: compute row NNZ and sums in a single shard scan (fused).
///
/// Avoids the double I/O of `backed_row_nnz()` + `backed_row_sums()` when
/// `filter_cells` needs both `min_genes` and `min_counts`. Pure-Rust
/// (`Result<_, String>`, no `PyErr`) so callers can run it through `detached`.
fn backed_row_nnz_and_sums(
    backed: &ScxBackedSparseDataset,
) -> Result<(Vec<i64>, Vec<f64>), String> {
    let (all_nnz, all_sums) = if let Some(cols) = backed.col_projection() {
        projected_agg::row_nnz_and_sums_projected(&backed.backed, cols)
            .map_err(|e| e.to_string())?
    } else {
        backed
            .backed
            .row_nnz_and_sums()
            .map_err(|e| e.to_string())?
    };
    Ok((
        backed.filter_row_results(&all_nnz),
        backed.filter_row_results(&all_sums),
    ))
}

/// Helper: compute col NNZ for a backed dataset (4-way dispatch).
///
/// Pure-Rust (`Result<_, String>`, no `PyErr`) so callers can run it through
/// `detached` with the GIL released.
fn backed_col_nnz(backed: &ScxBackedSparseDataset) -> Result<Vec<u32>, String> {
    let counts = match (backed.col_projection(), &backed.kept_to_global) {
        (Some(cols), Some(kept)) => {
            projected_agg::col_nnz_masked_projected(&backed.backed, kept, cols)
                .map_err(|e| e.to_string())?
        }
        (Some(cols), None) => {
            projected_agg::col_nnz_projected(&backed.backed, cols).map_err(|e| e.to_string())?
        }
        (None, Some(kept)) => {
            let f_counts = backed
                .backed
                .col_nnz_masked(kept)
                .map_err(|e| e.to_string())?;
            f_counts.iter().map(|&v| v as u32).collect()
        }
        (None, None) => backed.backed.col_nnz().map_err(|e| e.to_string())?,
    };
    // Reorder into presentation order so the keep-mask aligns with the
    // (presentation-ordered) var rows. No-op without preserve_var_order.
    Ok(backed.present_reorder(counts))
}

/// Helper: compute col sums for a backed dataset (4-way dispatch).
///
/// Pure-Rust (`Result<_, String>`, no `PyErr`) so callers can run it through
/// `detached` with the GIL released.
fn backed_col_sums(backed: &ScxBackedSparseDataset) -> Result<Vec<f64>, String> {
    let sums = match (backed.col_projection(), &backed.kept_to_global) {
        (Some(cols), Some(kept)) => {
            projected_agg::col_sums_masked_projected(&backed.backed, kept, cols)
                .map_err(|e| e.to_string())?
        }
        (Some(cols), None) => {
            projected_agg::col_sums_projected(&backed.backed, cols).map_err(|e| e.to_string())?
        }
        (None, Some(kept)) => backed
            .backed
            .col_sums_masked(kept)
            .map_err(|e| e.to_string())?,
        (None, None) => backed.backed.col_sums().map_err(|e| e.to_string())?,
    };
    // Reorder into presentation order so the keep-mask aligns with the
    // (presentation-ordered) var rows. No-op without preserve_var_order.
    Ok(backed.present_reorder(sums))
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

// Axis mutation — composing the new projection, re-pointing X, slicing the
// obs/var frames and every aligned member — lives in `crate::axis_align`
// behind `subset_obs_axis` / `subset_var_axis`. It used to be three helpers
// here (`compose_kept_to_global`, `slice_obs_and_obsm`,
// `update_layers_{kept_to_global,col_projection}`) that between them handled
// two of the five aligned members and reached them through the *validating*
// property, which raises the moment X changes width. See §9.18.

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
    if let Ok(backed) = x.cast::<ScxBackedSparseDataset>() {
        let n_obs = backed.borrow().shape_val.0;
        // Heavy shard scan runs off the GIL (`detached`). Rebind the dataset to a
        // plain `&Self` (Send) — the closure captures `b`, not the `!Send` PyRef.
        // The borrow is scoped to this block so it drops before `borrow_mut` below.
        let (row_nnz, row_sums) = {
            let bref = backed.borrow();
            let b: &ScxBackedSparseDataset = &bref;
            // Fused: compute both NNZ and sums in a single shard scan when both are needed
            if need_nnz && need_sums {
                let (nnz, sums) =
                    detached(py, || backed_row_nnz_and_sums(b)).map_err(PyRuntimeError::new_err)?;
                (Some(nnz), Some(sums))
            } else {
                (
                    if need_nnz {
                        Some(detached(py, || backed_row_nnz(b)).map_err(PyRuntimeError::new_err)?)
                    } else {
                        None
                    },
                    if need_sums {
                        Some(detached(py, || backed_row_sums(b)).map_err(PyRuntimeError::new_err)?)
                    } else {
                        None
                    },
                )
            }
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

        return subset_obs_axis(py, adata, &keep);
    }

    // Case 2: X is ScxLazyTransformedDataset
    if let Ok(lazy) = x.cast::<ScxLazyTransformedDataset>() {
        let lazy_ref = lazy.borrow();
        let n_obs = lazy_ref.shape_val.0;

        // Fused: when both NNZ and sums are needed, compute them in a single
        // shard scan. NNZ is transform-invariant but we compute it from the
        // same decoded shard to avoid double I/O.
        //
        // All three arms go through the visible-space `*_raw` API, so a column
        // projection is honored: a `filter_genes → normalize_total →
        // filter_cells` pipeline thresholds cells on totals over the *kept*
        // genes, as scanpy does on the sliced object. The physical-width
        // kernels these used to call summed the removed genes back in (§9.18).
        // Heavy shard scan + transforms run off the GIL (`detached`); rebind to a
        // plain `&Self` (Send) so the closure captures `l`, not the `!Send` PyRef.
        let l: &ScxLazyTransformedDataset = &lazy_ref;
        let (row_nnz, row_sums) = if need_nnz && need_sums {
            let (nnz, sums) = detached(py, || {
                let (all_nnz, all_sums) = l.row_nnz_and_sums_raw()?;
                Ok::<_, String>((
                    l.filter_row_results(&all_nnz),
                    l.filter_row_results(&all_sums),
                ))
            })
            .map_err(PyRuntimeError::new_err)?;
            (Some(nnz), Some(sums))
        } else {
            let nnz = if need_nnz {
                Some(
                    detached(py, || {
                        let all_nnz = l.row_nnz_raw()?;
                        Ok::<_, String>(l.filter_row_results(&all_nnz))
                    })
                    .map_err(PyRuntimeError::new_err)?,
                )
            } else {
                None
            };
            let sums = if need_sums {
                Some(
                    detached(py, || {
                        let all_sums = l.row_sums_raw()?;
                        Ok::<_, String>(l.filter_row_results(&all_sums))
                    })
                    .map_err(PyRuntimeError::new_err)?,
                )
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

        // Release the borrow before the funnel takes `borrow_mut` on the same X.
        drop(lazy_ref);
        return subset_obs_axis(py, adata, &keep);
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
    if let Ok(backed) = x.cast::<ScxBackedSparseDataset>() {
        let n_vars = backed.borrow().shape_val.1;
        // Heavy col scan runs off the GIL (`detached`). Rebind to a plain `&Self`
        // (Send) so the closure captures `b`, not the `!Send` PyRef; the borrow is
        // scoped to this block so it drops before the `borrow_mut` calls below.
        let (col_nnz, col_sums): (Option<Vec<i64>>, Option<Vec<f64>>) = {
            let bref = backed.borrow();
            let b: &ScxBackedSparseDataset = &bref;
            match (need_nnz, need_sums) {
                // Both thresholds: one fused scan instead of two full decodes.
                (true, true) => {
                    let (sums, nnz) = detached(py, || b.col_sums_and_nnz_raw())
                        .map_err(PyRuntimeError::new_err)?;
                    (Some(nnz.iter().map(|&v| v as i64).collect()), Some(sums))
                }
                (true, false) => (
                    Some(
                        detached(py, || backed_col_nnz(b))
                            .map_err(PyRuntimeError::new_err)?
                            .iter()
                            .map(|&v| v as i64)
                            .collect(),
                    ),
                    None,
                ),
                (false, true) => (
                    None,
                    Some(detached(py, || backed_col_sums(b)).map_err(PyRuntimeError::new_err)?),
                ),
                (false, false) => (None, None),
            }
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

        return subset_var_axis(py, adata, &keep);
    }

    // Case 2: X is ScxLazyTransformedDataset
    if let Ok(lazy) = x.cast::<ScxLazyTransformedDataset>() {
        let lazy_ref = lazy.borrow();
        let n_vars = lazy_ref.shape_val.1;

        // Heavy col scans run off the GIL (`detached`); rebind to a plain `&Self`
        // (Send) so the closures capture `l`, not the `!Send` PyRef. `l`'s borrow
        // ends before the direct `lazy_ref` use / `drop(lazy_ref)` below.
        let l: &ScxLazyTransformedDataset = &lazy_ref;
        let (col_nnz, col_sums): (Option<Vec<i64>>, Option<Vec<f64>>) = match (need_nnz, need_sums)
        {
            // Both thresholds: one fused scan through the transform chain,
            // mirroring the backed arm above. Without this the lazy path stayed
            // at two full decodes while backed dropped to one.
            (true, true) => {
                let (sums, nnz) =
                    detached(py, || l.col_sums_and_nnz_raw()).map_err(PyRuntimeError::new_err)?;
                (Some(nnz.iter().map(|&v| v as i64).collect()), Some(sums))
            }
            // NNZ is transform-invariant, so the single-statistic arm reads the
            // underlying backed reader with the lazy dataset's projection/mask.
            (true, false) => {
                let counts: Vec<u32> =
                    detached(py, || match (l.col_projection(), &l.kept_to_global) {
                        (Some(cols), Some(kept)) => {
                            projected_agg::col_nnz_masked_projected(&l.backed, kept, cols)
                                .map_err(|e| e.to_string())
                        }
                        (Some(cols), None) => projected_agg::col_nnz_projected(&l.backed, cols)
                            .map_err(|e| e.to_string()),
                        (None, Some(kept)) => l
                            .backed
                            .col_nnz_masked(kept)
                            .map(|f| f.iter().map(|&v| v as u32).collect())
                            .map_err(|e| e.to_string()),
                        (None, None) => l.backed.col_nnz().map_err(|e| e.to_string()),
                    })
                    .map_err(PyRuntimeError::new_err)?;
                (Some(counts.iter().map(|&v| v as i64).collect()), None)
            }
            (false, true) => (
                None,
                Some(detached(py, || l.col_sums_raw()).map_err(PyRuntimeError::new_err)?),
            ),
            (false, false) => (None, None),
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

        // Release the borrow before the funnel takes `borrow_mut` on the same X.
        drop(lazy_ref);
        return subset_var_axis(py, adata, &keep);
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
    let n_obs: usize = adata.getattr("shape")?.get_item(0)?.extract()?;
    let keep = keep_mask_from_selector(py, mask_or_indices, n_obs, "n_obs")?;
    subset_obs_axis(py, adata, &keep)
}

/// Subset genes (columns) by a boolean mask or an integer index array.
///
/// The var-axis twin of [`subset_obs`], and the general-purpose version of
/// `filter_genes()`:
///
///     mask = adata.var["highly_variable"]
///     pyscx.accel.subset_var(adata, mask)
///
/// Equivalent to `adata._inplace_subset_var(mask)`, which is also what
/// `adata[:, mask].copy()` does — both keep a backed / lazy `X` out of core.
/// The same set-membership caveat as [`subset_obs`] applies to integer
/// indices; pass a boolean mask, or use `adata[:, indices]`, if you need
/// ordered selection.
///
/// Args:
///     adata: AnnData object
///     mask_or_indices: Boolean mask (length n_vars) or integer index array
#[pyfunction]
#[pyo3(signature = (adata, mask_or_indices))]
pub fn subset_var(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    mask_or_indices: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let n_vars: usize = adata.getattr("shape")?.get_item(1)?.extract()?;
    let keep = keep_mask_from_selector(py, mask_or_indices, n_vars, "n_vars")?;
    subset_var_axis(py, adata, &keep)
}

/// Normalize a boolean mask / integer index array into a keep mask of length
/// `n`. Integer indices are set membership, not ordered selection — see
/// [`subset_obs`].
fn keep_mask_from_selector(
    py: Python<'_>,
    mask_or_indices: &Bound<'_, PyAny>,
    n: usize,
    axis_name: &str,
) -> PyResult<Vec<bool>> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (mask_or_indices,))?;
    let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;

    match kind.as_str() {
        "b" => {
            let bool_arr: numpy::PyReadonlyArray1<'_, bool> = arr.extract()?;
            let slice = bool_arr
                .as_slice()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            if slice.len() != n {
                return Err(PyValueError::new_err(format!(
                    "Boolean mask length ({}) does not match {axis_name} ({n})",
                    slice.len(),
                )));
            }
            Ok(slice.to_vec())
        }
        "i" | "u" => {
            let idx_arr = arr.call_method1("astype", (np.getattr("int64")?,))?;
            let idx_ro: numpy::PyReadonlyArray1<'_, i64> = idx_arr.extract()?;
            let indices = idx_ro
                .as_slice()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let mut mask = vec![false; n];
            for &idx in indices {
                if idx < 0 || (idx as usize) >= n {
                    return Err(PyValueError::new_err(format!(
                        "Index {idx} is out of bounds for {axis_name}={n}"
                    )));
                }
                mask[idx as usize] = true;
            }
            Ok(mask)
        }
        _ => Err(PyValueError::new_err(
            "mask_or_indices must be a boolean mask or integer index array",
        )),
    }
}
