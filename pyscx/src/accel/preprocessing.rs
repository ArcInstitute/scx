//! Preprocessing bindings — normalize_total, log1p, calculate_qc_metrics.

use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::{ScxLazyTransformedDataset, Transform};
use crate::projected_agg;

#[cfg(feature = "gpu")]
use super::gpu::resolve_device;

// Fusion marker key stashed on `adata.uns` by `normalize_total(device="gpu")`
// so that a subsequent `log1p(device="gpu")` can re-run a fused normalize+log1p
// pass over the ORIGINAL backed source instead of reading the already-
// materialized `adata.X`. See Phase 5 Task 5.4.
#[cfg(feature = "gpu")]
const GPU_NORMALIZE_MARKER_KEY: &str = "__scx_gpu_pending_normalize__";

/// Opaque marker stored on `adata.uns` after `normalize_total(device="gpu")`.
///
/// Carries everything needed to rebuild a `LazyShardSource` over the original
/// backed reader and re-run a fused normalize+log1p in a single GPU pass from
/// a later `log1p(device="gpu")` call.
///
/// The marker is consumed by `log1p(device="gpu")` on first hit; it is **not**
/// inspected by any other operation, so users who call `pca()` (or anything
/// else) between the two will get the non-fused (already-materialized) result.
/// Correct, just 1× redundant pass.
#[cfg(feature = "gpu")]
#[pyclass(name = "ScxGpuNormalizeMarker", module = "pyscx")]
pub struct ScxGpuNormalizeMarker {
    backed: Arc<scx_format::BackedCsrReader>,
    kept_to_global: Option<Arc<Vec<u64>>>,
    col_projection: Option<Arc<Vec<u32>>>,
    target_sum: f64,
    n_obs: usize,
    n_vars: usize,
}

/// Normalize total counts per cell.
///
/// By default (`device="auto"` or `device="cpu"`), replaces `adata.X` with a
/// lazy wrapper that applies row normalization during `__getitem__`. The row
/// sums are precomputed via streaming and cached in the wrapper.
///
/// With `device="gpu"`, the operation is **eager**: the normalized CSR is
/// streamed through GPU kernels shard-by-shard and materialized into a scipy
/// sparse CSR matrix that replaces `adata.X`. This breaks the lazy chain,
/// which is reported via a `UserWarning`.
///
/// Four cases (CPU path):
/// 1. X is `ScxBackedSparseDataset` → create new `ScxLazyTransformedDataset`
/// 2. X is `ScxLazyTransformedDataset` → append NormalizeTotal transform
/// 3. X is scipy sparse/dense → delegate to `sc.pp.normalize_total()`
///
/// GPU path (requires `pyscx` built with `--features gpu`):
/// * Backed/lazy X → stream through `scx_accel::gpu_preprocess_to_csr`, replace
///   `adata.X` with the materialized scipy CSR, and stash a fusion marker on
///   `adata.uns` so a subsequent `log1p(device="gpu")` can re-run a fused
///   pass over the ORIGINAL source.
/// * Scipy/dense X → defer to the CPU path (delegates to `sc.pp.normalize_total`).
///
/// Args:
///     adata: AnnData object
///     target_sum: Target total counts per cell (default: 1e4)
///     device: Device selection — "auto" (default), "cpu", or "gpu"
#[pyfunction]
#[pyo3(signature = (adata, target_sum=10000.0, device="auto"))]
pub fn normalize_total(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    target_sum: f64,
    device: &str,
) -> PyResult<()> {
    let _use_gpu = validate_device_or_default(device)?;

    #[cfg(feature = "gpu")]
    if _use_gpu {
        return gpu_normalize_total(py, adata, target_sum);
    }

    let x = adata.getattr("X")?;

    // Case 1: X is ScxBackedSparseDataset — create new lazy wrapper
    if let Ok(backed) = x.downcast::<ScxBackedSparseDataset>() {
        let backed_ref = backed.borrow();

        // Scanpy compat: sum only over projected (user-visible) genes.
        // After filter_genes(), col_projection restricts to kept genes.
        // Without col_projection, this sums all columns (same as before).
        // Must be in physical-row space because transforms are applied
        // per-shard before deletion vector filtering.
        let all_row_sums = if let Some(cols) = backed_ref.col_projection() {
            projected_agg::row_sums_projected(&backed_ref.backed, cols)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        } else {
            backed_ref
                .backed
                .row_sums()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        };

        let non_negative = backed_ref.non_negative;
        let lazy = ScxLazyTransformedDataset::new(
            Arc::clone(&backed_ref.backed),
            backed_ref.shape_val,
            backed_ref.kept_to_global.clone(),
            // Inherit col_projection: transforms operate on full columns,
            // projection is applied per-shard after transforms
            backed_ref.col_projection_arc(),
            vec![Transform::NormalizeTotal {
                row_sums: Arc::new(all_row_sums),
                target_sum,
            }],
            non_negative,
        );
        // Drop the borrow before setattr to avoid RefCell borrow conflict
        drop(backed_ref);
        adata.setattr("X", Bound::new(py, lazy)?)?;
        return Ok(());
    }

    // Case 2: X is already ScxLazyTransformedDataset — append transform
    if let Ok(lazy) = x.downcast::<ScxLazyTransformedDataset>() {
        let mut lazy_ref = lazy.borrow_mut();
        // Scanpy compat: sum only over projected (user-visible) genes.
        // streaming_row_sums_projected() applies transforms to the full-width
        // shard first (so prior NormalizeTotal sees correct denominator),
        // then project_csr restricts to projected genes before summing.
        // Returns a global-length vector (n_obs_global), which is what
        // apply_transforms_to_csr expects (indexes by global row).
        let sums = lazy_ref.streaming_row_sums_projected()?;
        lazy_ref.transforms.push(Transform::NormalizeTotal {
            row_sums: Arc::new(sums),
            target_sum,
        });
        return Ok(());
    }

    // Case 3: X is a regular scipy sparse or dense — delegate to scanpy
    let sc = py.import("scanpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("target_sum", target_sum)?;
    sc.getattr("pp")?
        .call_method("normalize_total", (adata,), Some(&kwargs))?;
    Ok(())
}

/// Apply log1p (ln(x + 1)) element-wise.
///
/// By default (`device="auto"` or `device="cpu"`), replaces `adata.X` with a
/// lazy wrapper that applies log1p during `__getitem__`. When chained after
/// `normalize_total`, the fused optimization in `ScxLazyTransformedDataset`
/// computes `ln(x * target_sum / row_sum + 1)` in a single pass.
///
/// With `device="gpu"`, the operation is **eager**. Three GPU sub-cases:
/// * If `adata.uns` carries a fusion marker from a preceding
///   `normalize_total(device="gpu")`, re-run a single fused normalize+log1p
///   pass over the original backed source and overwrite `adata.X` with the
///   fused result.
/// * Else if X is still backed/lazy, stream through `gpu_preprocess_to_csr`
///   with log1p only and materialize into scipy CSR.
/// * Else (X is already scipy/dense), upload data to GPU, apply log1p, and
///   write back into `adata.X`.
///
/// CPU path (three cases):
/// 1. X is `ScxBackedSparseDataset` → create new `ScxLazyTransformedDataset`
/// 2. X is `ScxLazyTransformedDataset` → append Log1p transform
/// 3. X is scipy sparse/dense → delegate to `sc.pp.log1p()`
///
/// Args:
///     adata: AnnData object
///     device: Device selection — "auto" (default), "cpu", or "gpu"
#[pyfunction]
#[pyo3(signature = (adata, device="auto"))]
pub fn log1p(py: Python<'_>, adata: &Bound<'_, PyAny>, device: &str) -> PyResult<()> {
    let _use_gpu = validate_device_or_default(device)?;

    #[cfg(feature = "gpu")]
    if _use_gpu {
        return gpu_log1p_dispatch(py, adata);
    }

    let x = adata.getattr("X")?;

    // Case 1: X is ScxBackedSparseDataset — create new lazy wrapper with Log1p
    if let Ok(backed) = x.downcast::<ScxBackedSparseDataset>() {
        let backed_ref = backed.borrow();

        let non_negative = backed_ref.non_negative;
        let lazy = ScxLazyTransformedDataset::new(
            Arc::clone(&backed_ref.backed),
            backed_ref.shape_val,
            backed_ref.kept_to_global.clone(),
            // Inherit col_projection so user-visible shape matches adata.var
            backed_ref.col_projection_arc(),
            vec![Transform::Log1p],
            non_negative,
        );
        // Drop the borrow before setattr to avoid RefCell borrow conflict
        drop(backed_ref);
        adata.setattr("X", Bound::new(py, lazy)?)?;
        return Ok(());
    }

    // Case 2: X is already ScxLazyTransformedDataset — append Log1p transform
    if let Ok(lazy) = x.downcast::<ScxLazyTransformedDataset>() {
        let mut lazy_ref = lazy.borrow_mut();
        lazy_ref.transforms.push(Transform::Log1p);
        return Ok(());
    }

    // Case 3: X is a regular scipy sparse or dense — delegate to scanpy
    let sc = py.import("scanpy")?;
    sc.getattr("pp")?.call_method1("log1p", (adata,))?;
    Ok(())
}

/// Calculate QC metrics natively using streaming aggregation.
///
/// Replacement for `sc.pp.calculate_qc_metrics()` that works on backed and
/// lazy-transformed SCX data without materialization.  Falls back to scanpy
/// for regular scipy/dense matrices.
///
/// Computes per-cell (obs) and per-gene (var) metrics and writes them to
/// `adata.obs` / `adata.var` columns, matching scanpy's naming convention.
///
/// Args:
///     adata: AnnData object
///     qc_vars: list of boolean column names in `adata.var` identifying gene
///              subsets (e.g. `["mt"]` for mitochondrial genes)
///     log1p: if True, also add log1p-transformed versions of count metrics
///     inplace: if True (default), write metrics to adata.obs/var;
///              if False, return (obs_df, var_df)
#[pyfunction]
#[pyo3(signature = (adata, qc_vars=None, log1p=true, inplace=true))]
pub fn calculate_qc_metrics<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    qc_vars: Option<Vec<String>>,
    log1p: bool,
    inplace: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let x = adata.getattr("X")?;
    let qc_vars = qc_vars.unwrap_or_default();

    // Detect backed or lazy-transformed SCX dataset
    let is_backed = x.downcast::<ScxBackedSparseDataset>().is_ok();
    let is_lazy = x.downcast::<ScxLazyTransformedDataset>().is_ok();

    if !is_backed && !is_lazy {
        // Delegate to scanpy for regular scipy/dense
        let sc = py.import("scanpy")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("inplace", inplace)?;
        kwargs.set_item("log1p", log1p)?;
        if !qc_vars.is_empty() {
            kwargs.set_item("qc_vars", &qc_vars)?;
        }
        return sc
            .getattr("pp")?
            .call_method("calculate_qc_metrics", (adata,), Some(&kwargs));
    }

    // --- Streaming path for SCX-backed / lazy data ---

    let np = py.import("numpy")?;
    let pd = py.import("pandas")?;

    // Compute per-cell total_counts and n_genes_by_counts
    let (total_counts, n_genes): (Vec<f64>, Vec<i64>) = if is_backed {
        let backed = x.extract::<PyRef<ScxBackedSparseDataset>>()?;
        let row_sums = backed
            .backed
            .row_sums()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let row_nnz = backed
            .backed
            .row_nnz()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        (
            backed.filter_row_results(&row_sums),
            backed.filter_row_results(&row_nnz),
        )
    } else {
        let lazy = x.extract::<PyRef<ScxLazyTransformedDataset>>()?;
        let row_sums = lazy.streaming_row_sums()?;
        let row_nnz = lazy
            .backed
            .row_nnz()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        (
            lazy.filter_row_results(&row_sums),
            lazy.filter_row_results(&row_nnz),
        )
    };

    // Compute per-gene total_counts and n_cells_by_counts
    let (gene_total_counts, n_cells): (Vec<f64>, Vec<i64>) = if is_backed {
        let backed = x.extract::<PyRef<ScxBackedSparseDataset>>()?;
        let col_sums = match &backed.kept_to_global {
            Some(kept) => backed
                .backed
                .col_sums_masked(kept)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            None => backed
                .backed
                .col_sums()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        };
        let col_nnz = match &backed.kept_to_global {
            Some(kept) => backed
                .backed
                .col_nnz_masked(kept)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .iter()
                .map(|&v| v as i64)
                .collect(),
            None => backed
                .backed
                .col_nnz()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        };
        (col_sums, col_nnz)
    } else {
        let lazy = x.extract::<PyRef<ScxLazyTransformedDataset>>()?;
        let col_sums = if lazy.kept_to_global.is_some() {
            lazy.streaming_col_sums_masked()?
        } else {
            lazy.streaming_col_sums()?
        };
        let col_nnz = if let Some(ref kept) = lazy.kept_to_global {
            lazy.backed
                .col_nnz_masked(kept)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .iter()
                .map(|&v| v as i64)
                .collect()
        } else {
            lazy.backed
                .col_nnz()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        };
        (col_sums, col_nnz)
    };

    // Build obs DataFrame
    let obs_dict = PyDict::new(py);
    obs_dict.set_item("n_genes_by_counts", numpy::PyArray::from_vec(py, n_genes))?;
    obs_dict.set_item(
        "total_counts",
        numpy::PyArray::from_vec(py, total_counts.clone()),
    )?;
    if log1p {
        let log_total: Vec<f64> = total_counts.iter().map(|&v| (v + 1.0).ln()).collect();
        obs_dict.set_item(
            "log1p_total_counts",
            numpy::PyArray::from_vec(py, log_total),
        )?;
    }

    // Build var DataFrame
    let var_dict = PyDict::new(py);
    var_dict.set_item("n_cells_by_counts", numpy::PyArray::from_vec(py, n_cells))?;
    var_dict.set_item(
        "total_counts",
        numpy::PyArray::from_vec(py, gene_total_counts.clone()),
    )?;
    if log1p {
        let log_gene_total: Vec<f64> = gene_total_counts.iter().map(|&v| (v + 1.0).ln()).collect();
        var_dict.set_item(
            "log1p_total_counts",
            numpy::PyArray::from_vec(py, log_gene_total),
        )?;
    }

    // Compute qc_var metrics (per-cell counts for gene subsets)
    for qc_var in &qc_vars {
        let var_df = adata.getattr("var")?;
        let mask_series = var_df.get_item(qc_var.as_str())?;
        let mask_values = mask_series.getattr("values")?;
        // Get column indices where mask is True
        let col_indices_py = np.call_method1("where", (&mask_values,))?;
        let col_indices_arr = col_indices_py.get_item(0)?;
        let col_indices: Vec<u32> = col_indices_arr
            .call_method1("astype", ("uint32",))?
            .extract()?;

        if col_indices.is_empty() {
            // No genes in this qc_var — fill with zeros
            let n_obs = total_counts.len();
            let zeros = vec![0.0f64; n_obs];
            obs_dict.set_item(
                format!("total_counts_{qc_var}"),
                numpy::PyArray::from_vec(py, zeros.clone()),
            )?;
            obs_dict.set_item(
                format!("pct_counts_{qc_var}"),
                numpy::PyArray::from_vec(py, zeros),
            )?;
            continue;
        }

        // Streaming projected row sums for the gene subset
        let subset_sums = if is_backed {
            let backed = x.extract::<PyRef<ScxBackedSparseDataset>>()?;
            let all_sums = projected_agg::row_sums_projected(&backed.backed, &col_indices)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            backed.filter_row_results(&all_sums)
        } else {
            let lazy = x.extract::<PyRef<ScxLazyTransformedDataset>>()?;
            // For lazy data, streaming through transforms with projection
            // is not yet supported. Fall back to the raw (pre-transform) sums
            // since QC metrics are typically computed before normalization.
            let all_sums = projected_agg::row_sums_projected(&lazy.backed, &col_indices)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            lazy.filter_row_results(&all_sums)
        };

        // pct_counts = subset_sum / total * 100
        let pct_counts: Vec<f64> = subset_sums
            .iter()
            .zip(total_counts.iter())
            .map(|(&s, &t)| if t > 0.0 { s / t * 100.0 } else { 0.0 })
            .collect();

        // Compute log1p before moving subset_sums
        let log1p_subset_sums: Option<Vec<f64>> = if log1p {
            Some(subset_sums.iter().map(|&v| (v + 1.0).ln()).collect())
        } else {
            None
        };

        obs_dict.set_item(
            format!("total_counts_{qc_var}"),
            numpy::PyArray::from_vec(py, subset_sums),
        )?;
        obs_dict.set_item(
            format!("pct_counts_{qc_var}"),
            numpy::PyArray::from_vec(py, pct_counts),
        )?;
        if let Some(log1p_sums) = log1p_subset_sums {
            obs_dict.set_item(
                format!("log1p_total_counts_{qc_var}"),
                numpy::PyArray::from_vec(py, log1p_sums),
            )?;
        }
    }

    // Create DataFrames
    let obs_index = adata.getattr("obs")?.getattr("index")?;
    let var_index = adata.getattr("var")?.getattr("index")?;
    let obs_df = pd.call_method1("DataFrame", (obs_dict,))?;
    obs_df.setattr("index", &obs_index)?;
    let var_df = pd.call_method1("DataFrame", (var_dict,))?;
    var_df.setattr("index", &var_index)?;

    if inplace {
        // Write columns to adata.obs / adata.var
        let adata_obs = adata.getattr("obs")?;
        let adata_var = adata.getattr("var")?;
        // Iterate obs_df columns
        let obs_columns: Vec<String> = obs_df
            .getattr("columns")?
            .call_method0("tolist")?
            .extract()?;
        for col in &obs_columns {
            let values = obs_df.get_item(col.as_str())?;
            adata_obs.set_item(col.as_str(), &values)?;
        }
        let var_columns: Vec<String> = var_df
            .getattr("columns")?
            .call_method0("tolist")?
            .extract()?;
        for col in &var_columns {
            let values = var_df.get_item(col.as_str())?;
            adata_var.set_item(col.as_str(), &values)?;
        }
        Ok(py.None().into_bound(py))
    } else {
        // Return (obs_df, var_df) tuple
        let tuple = pyo3::types::PyTuple::new(py, &[obs_df, var_df])?;
        Ok(tuple.into_any())
    }
}

// ---------------------------------------------------------------------------
// Device resolution + GPU eager dispatch (Phase 5)
// ---------------------------------------------------------------------------

/// Parse `device` into a bool. Returns `Ok(false)` on CPU-only builds even
/// when the user passes `device="gpu"` to avoid losing user input silently —
/// actually delegates to [`resolve_device`] when the GPU feature is enabled.
#[cfg(feature = "gpu")]
fn validate_device_or_default(device: &str) -> PyResult<bool> {
    resolve_device(device)
}

/// Fallback used when compiled without the `gpu` feature: only "cpu" / "auto"
/// are accepted. "gpu" raises a RuntimeError, matching `resolve_device`.
#[cfg(not(feature = "gpu"))]
fn validate_device_or_default(device: &str) -> PyResult<bool> {
    match device {
        "cpu" | "auto" => Ok(false),
        d if d.starts_with("gpu") => Err(PyRuntimeError::new_err(
            "device='gpu' requested but pyscx was built without the 'gpu' feature",
        )),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown device: '{}'. Use 'auto', 'cpu', or 'gpu'",
            device
        ))),
    }
}

#[cfg(feature = "gpu")]
fn emit_laziness_break_warning(py: Python<'_>) -> PyResult<()> {
    let warnings = py.import("warnings")?;
    warnings.call_method1(
        "warn",
        (
            "pyscx preprocessing device='gpu' is eager: adata.X will be \
             materialized as scipy.sparse.csr_matrix, breaking the lazy chain.",
            py.get_type::<pyo3::exceptions::PyUserWarning>(),
        ),
    )?;
    Ok(())
}

/// Build a scipy `csr_matrix((data, indices, indptr), shape=...)` on the Python side.
#[cfg(feature = "gpu")]
fn scx_csr_to_scipy<'py>(py: Python<'py>, csr: scx_sparse::ScxCsr) -> PyResult<Bound<'py, PyAny>> {
    let scipy_sparse = py.import("scipy.sparse")?;
    let data = numpy::PyArray::from_vec(py, csr.data);
    let indices = numpy::PyArray::from_vec(py, csr.indices);
    let indptr = numpy::PyArray::from_vec(py, csr.indptr);
    let args = pyo3::types::PyTuple::new(
        py,
        &[data.into_any(), indices.into_any(), indptr.into_any()],
    )?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("shape", csr.shape)?;
    scipy_sparse.call_method("csr_matrix", (args,), Some(&kwargs))
}

/// Everything `gpu_normalize_total` needs from an AnnData's `X`: a live
/// `LazyShardSource` plus the raw handles required to rebuild an equivalent
/// source later (fusion marker). Returned by [`source_from_x`].
#[cfg(feature = "gpu")]
struct GpuShardSource {
    source: crate::lazy_transform::LazyShardSource,
    backed: Arc<scx_format::BackedCsrReader>,
    kept_to_global: Option<Arc<Vec<u64>>>,
    col_projection: Option<Arc<Vec<u32>>>,
    n_obs: usize,
    n_vars: usize,
}

/// Extract a [`GpuShardSource`] from either a backed or lazy X. Returns
/// `None` when `x` is neither (i.e. scipy/dense), signalling that the caller
/// should fall back to the CPU path.
#[cfg(feature = "gpu")]
fn source_from_x(x: &Bound<'_, PyAny>) -> PyResult<Option<GpuShardSource>> {
    use crate::lazy_transform::LazyShardSource;

    if let Ok(backed) = x.downcast::<ScxBackedSparseDataset>() {
        let r = backed.borrow();
        let backed_arc = Arc::clone(&r.backed);
        let kept = r.kept_to_global.clone();
        let col_proj = r.col_projection_arc();
        let (n_obs, n_vars) = r.shape_val;
        let source = LazyShardSource::new(
            Arc::clone(&backed_arc),
            vec![],
            kept.clone(),
            col_proj.clone(),
            n_obs,
            n_vars,
        );
        return Ok(Some(GpuShardSource {
            source,
            backed: backed_arc,
            kept_to_global: kept,
            col_projection: col_proj,
            n_obs,
            n_vars,
        }));
    }

    if let Ok(lazy) = x.downcast::<ScxLazyTransformedDataset>() {
        let r = lazy.borrow();
        let source = r.as_shard_source();
        let backed_arc = Arc::clone(&r.backed);
        let kept = r.kept_to_global.clone();
        let col_proj = r.col_projection.clone();
        let (n_obs, n_vars) = r.shape_val;
        return Ok(Some(GpuShardSource {
            source,
            backed: backed_arc,
            kept_to_global: kept,
            col_projection: col_proj,
            n_obs,
            n_vars,
        }));
    }

    Ok(None)
}

/// GPU-eager `normalize_total` dispatch.
#[cfg(feature = "gpu")]
fn gpu_normalize_total(py: Python<'_>, adata: &Bound<'_, PyAny>, target_sum: f64) -> PyResult<()> {
    let x = adata.getattr("X")?;

    let Some(gs) = source_from_x(&x)? else {
        // Scipy/dense X: fall back to CPU path (scanpy's normalize_total).
        let sc = py.import("scanpy")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("target_sum", target_sum)?;
        sc.getattr("pp")?
            .call_method("normalize_total", (adata,), Some(&kwargs))?;
        return Ok(());
    };

    let was_lazy = x.downcast::<ScxLazyTransformedDataset>().is_ok();
    let GpuShardSource {
        source,
        backed,
        kept_to_global,
        col_projection,
        n_obs,
        n_vars,
    } = gs;

    let dev = scx_accel::GpuDevice::new(0)
        .map_err(|e| PyRuntimeError::new_err(format!("GPU init failed: {e}")))?;
    let csr = scx_accel::gpu_preprocess_to_csr(&dev, &source, Some(target_sum as f32), false)
        .map_err(|e| PyRuntimeError::new_err(format!("gpu_preprocess_to_csr: {e}")))?;
    drop(source);

    // Replace adata.X with materialized scipy CSR.
    let scipy_csr = scx_csr_to_scipy(py, csr)?;
    adata.setattr("X", scipy_csr)?;

    // Stash fusion marker on adata.uns so a subsequent log1p(device="gpu")
    // can re-run a fused normalize+log1p pass over the original source.
    let marker = ScxGpuNormalizeMarker {
        backed,
        kept_to_global,
        col_projection,
        target_sum,
        n_obs,
        n_vars,
    };
    let uns = adata.getattr("uns")?;
    uns.set_item(GPU_NORMALIZE_MARKER_KEY, Bound::new(py, marker)?)?;

    if was_lazy {
        emit_laziness_break_warning(py)?;
    }
    Ok(())
}

/// GPU-eager `log1p` dispatch — handles the fusion-marker path, backed/lazy
/// materialization, and in-place scipy CSR log1p.
#[cfg(feature = "gpu")]
fn gpu_log1p_dispatch(py: Python<'_>, adata: &Bound<'_, PyAny>) -> PyResult<()> {
    let uns = adata.getattr("uns")?;

    // 1) Fusion marker path: re-run fused normalize+log1p over ORIGINAL source.
    let marker_obj = uns.call_method1("get", (GPU_NORMALIZE_MARKER_KEY,))?;
    if !marker_obj.is_none() {
        if let Ok(marker_ref) = marker_obj.downcast::<ScxGpuNormalizeMarker>() {
            let m = marker_ref.borrow();
            let source = crate::lazy_transform::LazyShardSource::new(
                Arc::clone(&m.backed),
                vec![],
                m.kept_to_global.clone(),
                m.col_projection.clone(),
                m.n_obs,
                m.n_vars,
            );
            let target_sum = m.target_sum;
            drop(m);

            let dev = scx_accel::GpuDevice::new(0)
                .map_err(|e| PyRuntimeError::new_err(format!("GPU init failed: {e}")))?;
            let csr =
                scx_accel::gpu_preprocess_to_csr(&dev, &source, Some(target_sum as f32), true)
                    .map_err(|e| PyRuntimeError::new_err(format!("gpu_preprocess_to_csr: {e}")))?;
            drop(source);

            let scipy_csr = scx_csr_to_scipy(py, csr)?;
            adata.setattr("X", scipy_csr)?;

            // Invalidate marker after consuming it.
            uns.call_method1("pop", (GPU_NORMALIZE_MARKER_KEY,))?;
            return Ok(());
        }
    }

    // 2) Backed/lazy X: stream through gpu_preprocess_to_csr with log1p only.
    let x = adata.getattr("X")?;
    if let Some(gs) = source_from_x(&x)? {
        let was_lazy = x.downcast::<ScxLazyTransformedDataset>().is_ok();
        let dev = scx_accel::GpuDevice::new(0)
            .map_err(|e| PyRuntimeError::new_err(format!("GPU init failed: {e}")))?;
        let csr = scx_accel::gpu_preprocess_to_csr(&dev, &gs.source, None, true)
            .map_err(|e| PyRuntimeError::new_err(format!("gpu_preprocess_to_csr: {e}")))?;
        drop(gs);
        let scipy_csr = scx_csr_to_scipy(py, csr)?;
        adata.setattr("X", scipy_csr)?;
        if was_lazy {
            emit_laziness_break_warning(py)?;
        }
        return Ok(());
    }

    // 3) Scipy/dense X: apply gpu_log1p in place on the data buffer.
    gpu_log1p_on_materialized(py, adata)
}

/// Upload the `data` array of a materialized scipy/dense X, apply the
/// GPU log1p kernel, and write back.
#[cfg(feature = "gpu")]
fn gpu_log1p_on_materialized(py: Python<'_>, adata: &Bound<'_, PyAny>) -> PyResult<()> {
    let scipy_sparse = py.import("scipy.sparse")?;
    let x = adata.getattr("X")?;
    let is_sparse = scipy_sparse
        .call_method1("issparse", (&x,))?
        .extract::<bool>()?;
    if !is_sparse {
        // Dense X: defer to scanpy (small code path; not worth a GPU kernel).
        let sc = py.import("scanpy")?;
        sc.getattr("pp")?.call_method1("log1p", (adata,))?;
        return Ok(());
    }

    // Ensure CSR for predictable layout.
    let csr_py = scipy_sparse.call_method1("csr_matrix", (&x,))?;
    let np = py.import("numpy")?;
    let indptr: Vec<i64> = np
        .call_method1("asarray", (csr_py.getattr("indptr")?,))?
        .call_method1("astype", ("int64",))?
        .extract()?;
    let data: Vec<f32> = np
        .call_method1("asarray", (csr_py.getattr("data")?,))?
        .call_method1("astype", ("float32",))?
        .extract()?;
    let n_rows = indptr.len().saturating_sub(1);

    let dev = scx_accel::GpuDevice::new(0)
        .map_err(|e| PyRuntimeError::new_err(format!("GPU init failed: {e}")))?;
    let d_indptr = dev
        .htod_copy(&indptr)
        .map_err(|e| PyRuntimeError::new_err(format!("htod indptr: {e}")))?;
    let mut d_data = dev
        .htod_copy(&data)
        .map_err(|e| PyRuntimeError::new_err(format!("htod data: {e}")))?;
    scx_accel::gpu_log1p(&dev, &d_indptr, &mut d_data, n_rows)
        .map_err(|e| PyRuntimeError::new_err(format!("gpu_log1p: {e}")))?;
    dev.synchronize()
        .map_err(|e| PyRuntimeError::new_err(format!("sync: {e}")))?;
    let new_data: Vec<f32> = dev
        .dtoh_copy(&d_data)
        .map_err(|e| PyRuntimeError::new_err(format!("dtoh data: {e}")))?;

    // Write back: rebuild csr_matrix with the transformed data and the
    // existing structural arrays. This is cheaper than mutating
    // `csr_py.data[:] = new_data` because we avoid the Python-side
    // buffer-reshape machinery.
    let shape: (usize, usize) = csr_py.getattr("shape")?.extract()?;
    let indices = csr_py.getattr("indices")?;
    let new_data_np = numpy::PyArray::from_vec(py, new_data);
    let indptr_np = numpy::PyArray::from_vec(py, indptr);
    let args = pyo3::types::PyTuple::new(
        py,
        &[
            new_data_np.into_any(),
            indices.clone(),
            indptr_np.into_any(),
        ],
    )?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("shape", shape)?;
    let new_csr = scipy_sparse.call_method("csr_matrix", (args,), Some(&kwargs))?;
    adata.setattr("X", new_csr)?;
    Ok(())
}
