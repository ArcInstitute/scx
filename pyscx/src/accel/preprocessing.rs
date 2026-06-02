//! Preprocessing bindings — normalize_total, log1p, calculate_qc_metrics.

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::{ScxLazyTransformedDataset, Transform};
use crate::projected_agg;

// Fusion marker key stashed on `adata.uns` by `normalize_total(device="gpu")`
// so that a subsequent `log1p(device="gpu")` can re-run a fused normalize+log1p
// pass over the ORIGINAL backed source instead of reading the already-
// materialized `adata.X`. See Phase 5 Task 5.4.
//
// Lifecycle contract: the marker is **valid only while X and its row/col
// dimensions are unchanged**. It MUST be cleared by any op that mutates X or
// changes obs/var shape (normalize_total, CPU log1p, filter_*, subset_obs).
// Ops that don't mutate X (PCA, kNN, UMAP, Leiden, Harmony, HVG,
// calculate_qc_metrics) leave it intact. New call sites that mutate X must
// call `clear_gpu_normalize_marker`.
pub(crate) const GPU_NORMALIZE_MARKER_KEY: &str = "__scx_gpu_pending_normalize__";

/// Pop the GPU `normalize_total → log1p` fusion marker off `adata.uns` if
/// present. No-op if absent. Always available (independent of `gpu` feature)
/// so non-GPU builds can clear stale markers left by a prior GPU build.
pub(crate) fn clear_gpu_normalize_marker(adata: &Bound<'_, PyAny>) -> PyResult<()> {
    let uns = adata.getattr("uns")?;
    // dict.pop(key, None) returns None on absence rather than raising KeyError.
    let py = adata.py();
    uns.call_method1("pop", (GPU_NORMALIZE_MARKER_KEY, py.None()))?;
    Ok(())
}

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
///     device: Device selection — "auto" (default), "cpu", "gpu", or
///         "gpu:N" to target CUDA device N on multi-GPU systems.
#[pyfunction]
#[pyo3(signature = (adata, target_sum=10000.0, device="auto"))]
pub fn normalize_total(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    target_sum: f64,
    device: &str,
) -> PyResult<()> {
    let _device = validate_device_or_default(device)?;

    // Any prior fusion marker is now stale: this normalize call supersedes it.
    // The GPU success path will stash a fresh marker; CPU / scipy-fallback
    // paths leave it cleared.
    clear_gpu_normalize_marker(adata)?;

    // Record the planned route. The GPU shard-streaming kernel runs only for
    // backed / lazy X; a scipy/dense X has no GPU kernel and falls back to the
    // CPU path (UnsupportedInputLayout on a GPU host). The CPU path itself is
    // lazy (it wraps X), recorded as cpu_csr.
    let np_eligible = {
        let xp = adata.getattr("X")?;
        xp.cast::<ScxBackedSparseDataset>().is_ok()
            || xp.cast::<ScxLazyTransformedDataset>().is_ok()
    };
    super::route::write_accel_route(
        py,
        adata,
        "normalize_total",
        &super::route::simple_exec_info(
            device,
            np_eligible,
            scx_accel::AccelRoute::GpuCsr,
            scx_accel::AccelRoute::CpuCsr,
        ),
    )?;

    #[cfg(feature = "gpu")]
    if let Some(device_id) = _device.gpu_id() {
        return gpu_normalize_total(py, adata, target_sum, device_id, device);
    }

    let x = adata.getattr("X")?;

    // Case 1: X is ScxBackedSparseDataset — create new lazy wrapper
    if let Ok(backed) = x.cast::<ScxBackedSparseDataset>() {
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
        )
        // Forward CSC sidecar so a later log1p() preserves the
        // CSC-dispatch capability (NormalizeTotal itself is not
        // column-local, but a NormalizeTotal+Log1p chain is — the
        // gate at as_column_source() will reject it for now;
        // carrying the handle costs nothing).
        .with_csc_reader(backed_ref.backed_csc.clone())
        .with_source_path(backed_ref.source_path.clone());
        // Drop the borrow before setattr to avoid RefCell borrow conflict
        drop(backed_ref);
        adata.setattr("X", Bound::new(py, lazy)?)?;
        return Ok(());
    }

    // Case 2: X is already ScxLazyTransformedDataset — append transform
    if let Ok(lazy) = x.cast::<ScxLazyTransformedDataset>() {
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
/// * Else (X is already scipy/dense), GPU dispatch is slower than CPU
///   (H→D + D→H copies dominate log1p's trivial math), so `pyscx` emits a
///   `UserWarning` and falls back to `sc.pp.log1p`.
///
/// CPU path (three cases):
/// 1. X is `ScxBackedSparseDataset` → create new `ScxLazyTransformedDataset`
/// 2. X is `ScxLazyTransformedDataset` → append Log1p transform
/// 3. X is scipy sparse/dense → delegate to `sc.pp.log1p()`
///
/// Args:
///     adata: AnnData object
///     device: Device selection — "auto" (default), "cpu", "gpu", or
///         "gpu:N" to target CUDA device N on multi-GPU systems.
#[pyfunction]
#[pyo3(signature = (adata, device="auto"))]
pub fn log1p(py: Python<'_>, adata: &Bound<'_, PyAny>, device: &str) -> PyResult<()> {
    let _device = validate_device_or_default(device)?;

    // Record the planned route. The GPU kernel runs when a normalize+log1p
    // fusion marker is present (re-runs over the ORIGINAL source) or when X is
    // still backed / lazy; a materialized scipy/dense X falls back to CPU
    // (UnsupportedInputLayout on a GPU host). The CPU path is lazy (cpu_csr).
    let log1p_eligible = {
        let has_marker = !adata
            .getattr("uns")?
            .call_method1("get", (GPU_NORMALIZE_MARKER_KEY, py.None()))?
            .is_none();
        let xp = adata.getattr("X")?;
        has_marker
            || xp.cast::<ScxBackedSparseDataset>().is_ok()
            || xp.cast::<ScxLazyTransformedDataset>().is_ok()
    };
    super::route::write_accel_route(
        py,
        adata,
        "log1p",
        &super::route::simple_exec_info(
            device,
            log1p_eligible,
            scx_accel::AccelRoute::GpuCsr,
            scx_accel::AccelRoute::CpuCsr,
        ),
    )?;

    #[cfg(feature = "gpu")]
    if let Some(device_id) = _device.gpu_id() {
        return gpu_log1p_dispatch(py, adata, device_id, device);
    }

    // CPU log1p invalidates any pending GPU fusion: a later log1p(device="gpu")
    // must NOT replay a fused normalize+log1p over the original source (it
    // would silently overwrite the result of this CPU log1p call).
    clear_gpu_normalize_marker(adata)?;

    let x = adata.getattr("X")?;

    // Case 1: X is ScxBackedSparseDataset — create new lazy wrapper with Log1p
    if let Ok(backed) = x.cast::<ScxBackedSparseDataset>() {
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
        )
        // Forward CSC sidecar so the resulting log1p()-only chain
        // remains CSC-dispatchable via as_column_source().
        .with_csc_reader(backed_ref.backed_csc.clone())
        .with_source_path(backed_ref.source_path.clone());
        // Drop the borrow before setattr to avoid RefCell borrow conflict
        drop(backed_ref);
        adata.setattr("X", Bound::new(py, lazy)?)?;
        return Ok(());
    }

    // Case 2: X is already ScxLazyTransformedDataset — append Log1p transform
    if let Ok(lazy) = x.cast::<ScxLazyTransformedDataset>() {
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
///     prefer_format: "csr" (default) or "csc". When "csc", the gene-axis
///         (`total_counts`, `n_cells_by_counts`) accumulators read the CSC
///         sidecar instead of streaming CSR shards. The cell-axis stays
///         on CSR — row aggregations have no CSC win. Capability gate
///         applies (no row deletion vector, no non-column-local
///         transforms; raises on missing CSC sidecar).
#[pyfunction]
#[pyo3(signature = (adata, qc_vars=None, log1p=true, inplace=true, prefer_format="csr"))]
pub fn calculate_qc_metrics<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    qc_vars: Option<Vec<String>>,
    log1p: bool,
    inplace: bool,
    prefer_format: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if !matches!(prefer_format, "csr" | "csc") {
        return Err(PyValueError::new_err(format!(
            "Invalid prefer_format={prefer_format:?}; expected 'csr' or 'csc'"
        )));
    }

    // Record the planned route. QC metrics have no GPU kernel; the only route
    // choice is the gene-axis layout — cpu_csc when prefer_format="csc" (reads
    // the gene-major sidecar), cpu_csr otherwise. This is the route the CSC
    // dispatch gate asserts.
    let qc_route = if prefer_format == "csc" {
        scx_accel::AccelRoute::CpuCsc
    } else {
        scx_accel::AccelRoute::CpuCsr
    };
    super::route::write_accel_route(
        py,
        adata,
        "calculate_qc_metrics",
        &scx_accel::AccelExecutionInfo::new(qc_route, scx_accel::FallbackReason::None),
    )?;

    let x = adata.getattr("X")?;
    let qc_vars = qc_vars.unwrap_or_default();

    // emit qc-vars
    // advisories upfront so they cover BOTH the SCX-backed/lazy streaming
    // path and the scipy/dense → scanpy fallback path.
    emit_qc_advisories(py, adata, &qc_vars)?;

    // Detect backed or lazy-transformed SCX dataset
    let is_backed = x.cast::<ScxBackedSparseDataset>().is_ok();
    let is_lazy = x.cast::<ScxLazyTransformedDataset>().is_ok();

    if !is_backed && !is_lazy {
        // Delegate to scanpy for regular scipy/dense. CSC on a non-SCX
        // matrix has no meaningful interpretation — reject explicitly so
        // the user doesn't think it had any effect.
        if prefer_format == "csc" {
            return Err(PyRuntimeError::new_err(
                "prefer_format='csc' requires adata.X to be ScxBackedSparseDataset \
                 or ScxLazyTransformedDataset (got scipy/dense)",
            ));
        }
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

    // Compute per-cell total_counts and n_genes_by_counts. Row-axis
    // aggregations stay CSR regardless of prefer_format — CSC offers
    // no win for row sums (would require gathering per-shard column
    // contributions back into a row index).
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

    // Compute per-gene total_counts and n_cells_by_counts.
    let (gene_total_counts, n_cells): (Vec<f64>, Vec<u32>) = if prefer_format == "csc" {
        // CSC dispatch: capability gate + projected_agg twins.
        compute_gene_axis_csc(&x)?
    } else if is_backed {
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
                .map(|&v| v as u32)
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
                .map(|&v| v as u32)
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
            // B1-2026-05-20-Tier2: empty-mask advisory already emitted by
            // `emit_qc_advisories` upfront. Fall through to the zero-fill
            // so downstream code sees the expected obs columns.
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

/// Parse `device` into a [`ResolvedDevice`]. Delegates to [`resolve_device`]
/// in both feature configurations — the CPU-only build's parser already
/// rejects `gpu*` requests with a `RuntimeError`, so no separate
/// implementation is needed.
fn validate_device_or_default(device: &str) -> PyResult<super::gpu::ResolvedDevice> {
    super::gpu::resolve_device(device)
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

#[cfg(feature = "gpu")]
fn emit_gpu_log1p_fallback_warning(py: Python<'_>, device: &str) -> PyResult<()> {
    let warnings = py.import("warnings")?;
    warnings.call_method1(
        "warn",
        (
            format!(
                "pyscx.accel.log1p(device={device:?}) on a materialized scipy/dense \
                 X falls back to CPU: H→D and D→H copies dominate log1p's trivial \
                 math, making GPU dispatch 50–100× slower than scanpy.pp.log1p. The \
                 GPU fast path requires X to be ScxBackedSparseDataset or \
                 ScxLazyTransformedDataset at the time log1p runs — once an \
                 intermediate step (.copy(), adata[:, mask].copy(), etc.) materialises \
                 X to scipy CSR, the fast path is unreachable for the rest of the \
                 pipeline. To enable it, open via pyscx.open(...).to_anndata(backed=True) \
                 and avoid materialising between normalize_total and log1p."
            ),
            py.get_type::<pyo3::exceptions::PyUserWarning>(),
        ),
    )?;
    Ok(())
}

/// Symmetric helper for `gpu_normalize_total`'s scipy/dense fallback. Mirrors
/// `emit_gpu_log1p_fallback_warning`'s shape so a docs-skimmer reading the two
/// warnings side-by-side sees the same "open backed" recommendation.
#[cfg(feature = "gpu")]
fn emit_gpu_normalize_fallback_warning(py: Python<'_>, device: &str) -> PyResult<()> {
    let warnings = py.import("warnings")?;
    warnings.call_method1(
        "warn",
        (
            format!(
                "pyscx.accel.normalize_total(device={device:?}) on a materialized \
                 scipy/dense X falls back to CPU (scanpy.pp.normalize_total); the GPU \
                 shard-streaming path requires an ScxBackedSparseDataset or \
                 ScxLazyTransformedDataset. To get the GPU fast path (including the \
                 normalize+log1p fusion marker on adata.uns), open via \
                 pyscx.open(...).to_anndata(backed=True) or keep the source as \
                 ScxLazyTransformedDataset until after log1p."
            ),
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

    if let Ok(backed) = x.cast::<ScxBackedSparseDataset>() {
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

    if let Ok(lazy) = x.cast::<ScxLazyTransformedDataset>() {
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
fn gpu_normalize_total(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    target_sum: f64,
    device_id: usize,
    device: &str,
) -> PyResult<()> {
    let x = adata.getattr("X")?;

    let Some(gs) = source_from_x(&x)? else {
        // Scipy/dense X: fall back to CPU path (scanpy's normalize_total).
        // Emit a UserWarning symmetric with `gpu_log1p_dispatch`'s scipy/dense
        // fallback so the silent-fallback regression that the 2026-05-21
        // dogfood report flagged (B4) cannot recur.
        emit_gpu_normalize_fallback_warning(py, device)?;
        let sc = py.import("scanpy")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("target_sum", target_sum)?;
        sc.getattr("pp")?
            .call_method("normalize_total", (adata,), Some(&kwargs))?;
        return Ok(());
    };

    let was_lazy = x.cast::<ScxLazyTransformedDataset>().is_ok();
    let GpuShardSource {
        source,
        backed,
        kept_to_global,
        col_projection,
        n_obs,
        n_vars,
    } = gs;

    let dev = scx_accel::GpuDevice::new(device_id)
        .map_err(|e| PyRuntimeError::new_err(format!("GPU init failed: {e}")))?;
    let csr = scx_accel::gpu_preprocess_to_csr(&dev, &source, Some(target_sum as f32), false, None)
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

/// GPU-eager `log1p` dispatch — handles the fusion-marker path and backed/lazy
/// materialization. Materialised scipy/dense X warns and falls back to
/// `sc.pp.log1p` (the H→D / D→H round-trip dominates the kernel cost).
#[cfg(feature = "gpu")]
fn gpu_log1p_dispatch(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    device_id: usize,
    device: &str,
) -> PyResult<()> {
    let uns = adata.getattr("uns")?;

    // 1) Fusion marker path: re-run fused normalize+log1p over ORIGINAL source.
    let marker_obj = uns.call_method1("get", (GPU_NORMALIZE_MARKER_KEY,))?;
    if !marker_obj.is_none() {
        if let Ok(marker_ref) = marker_obj.cast::<ScxGpuNormalizeMarker>() {
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

            let dev = scx_accel::GpuDevice::new(device_id)
                .map_err(|e| PyRuntimeError::new_err(format!("GPU init failed: {e}")))?;
            let csr = scx_accel::gpu_preprocess_to_csr(
                &dev,
                &source,
                Some(target_sum as f32),
                true,
                None,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("gpu_preprocess_to_csr: {e}")))?;
            drop(source);

            let scipy_csr = scx_csr_to_scipy(py, csr)?;
            adata.setattr("X", scipy_csr)?;

            // Invalidate marker after consuming it.
            uns.call_method1("pop", (GPU_NORMALIZE_MARKER_KEY,))?;
            return Ok(());
        }
    }

    // Reaching here means Path 1 was not consumed (marker absent or downcast
    // failed). Any leftover marker is now stale — defensively clear before the
    // remaining paths mutate X.
    clear_gpu_normalize_marker(adata)?;

    // 2) Backed/lazy X: stream through gpu_preprocess_to_csr with log1p only.
    let x = adata.getattr("X")?;
    if let Some(gs) = source_from_x(&x)? {
        let was_lazy = x.cast::<ScxLazyTransformedDataset>().is_ok();
        let dev = scx_accel::GpuDevice::new(device_id)
            .map_err(|e| PyRuntimeError::new_err(format!("GPU init failed: {e}")))?;
        let csr = scx_accel::gpu_preprocess_to_csr(&dev, &gs.source, None, true, None)
            .map_err(|e| PyRuntimeError::new_err(format!("gpu_preprocess_to_csr: {e}")))?;
        drop(gs);
        let scipy_csr = scx_csr_to_scipy(py, csr)?;
        adata.setattr("X", scipy_csr)?;
        if was_lazy {
            emit_laziness_break_warning(py)?;
        }
        return Ok(());
    }

    // 3) Scipy/dense X: GPU dispatch is slower than CPU here (H→D + D→H
    //    copies dominate log1p's trivial math). Warn and fall back to scanpy.
    emit_gpu_log1p_fallback_warning(py, device)?;
    let sc = py.import("scanpy")?;
    sc.getattr("pp")?.call_method1("log1p", (adata,))?;
    Ok(())
}

/// Return whether `adata.var[qc_var]` is present and has any True entries.
/// `Some(true)` = mask has ≥1 True; `Some(false)` = mask is all False;
/// `None` = column absent or values not coercible to a Python bool array.
fn qc_var_mask_has_true(adata: &Bound<'_, PyAny>, qc_var: &str) -> Option<bool> {
    let var = adata.getattr("var").ok()?;
    let has_col = var
        .call_method1("__contains__", (qc_var,))
        .ok()
        .and_then(|v| v.is_truthy().ok())
        .unwrap_or(false);
    if !has_col {
        return None;
    }
    let series = var.get_item(qc_var).ok()?;
    // `Series.any()` returns a numpy.bool_ scalar. Extract via Python truthiness.
    let any_result = series.call_method0("any").ok()?;
    any_result.is_truthy().ok()
}

/// Emit qc-vars-related advisories before either the SCX-backed loop or
/// the scanpy fallback runs. Covers three cases:
///   * **F2-2026-05-19**: `qc_vars` is None AND adata.var_names contains
///     5+ MT-/mt- prefixed symbols. Without `qc_vars`, `pct_counts_mt`
///     is never computed and a downstream scanpy filter raises KeyError.
///   * **N1-2026-05-21-Tier2**: `qc_vars` is None AND var_names lacks MT
///     prefixes (e.g. CELLxGENE Census integer-string indexes) but
///     `var['feature_name']` has 5+ MT-/mt- prefixed symbols. Same
///     downstream symptom; the message points at feature_name as the
///     correct source. The two qc_vars=None branches are mutually
///     exclusive (else-arm) so var_names wins when both axes have MT.
///   * **B1-2026-05-20-Tier2 options 1+2**: a user-supplied qc_var mask
///     matches zero genes — `pct_counts_{qc_var}` is silently filled with
///     zeros. For `qc_var == "mt"`, when adata.var['feature_name'] has
///     MT-prefixed symbols (the CELLxGENE Census layout signature), emit
///     a CELLxGENE-specific message pointing at `feature_name` as the
///     right source; otherwise emit a generic empty-mask message.
fn emit_qc_advisories(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    qc_vars: &[String],
) -> PyResult<()> {
    let warnings = py.import("warnings")?;
    let builtins = py.import("builtins")?;
    let user_warning = builtins.getattr("UserWarning")?;

    // The MT-prefix counts cost a pandas
    // `.str.startswith(...).sum()` scan each — small but non-trivial on
    // 60k+ genes. Compute lazily and memoize so the common path (valid
    // non-empty `qc_vars=["mt"]` mask) does zero scans, and the warning
    // paths do at most one scan per axis.
    let mut n_mt_var_names: Option<usize> = None;
    let mut n_mt_feature_name: Option<usize> = None;

    // F2-2026-05-19: qc_vars=None + MT genes visible in var_names.
    // N1-2026-05-21-Tier2: extend to CELLxGENE Census layout (integer-
    // string var_names, MT symbols in var['feature_name']) via an
    // `else if` arm — var_names wins when both axes have MT prefixes so
    // the existing test_warns_when_mt_genes_present_and_qc_vars_none
    // fixture (no feature_name column) keeps its message verbatim.
    if qc_vars.is_empty() {
        let n_vn = *n_mt_var_names.get_or_insert_with(|| count_mt_prefix_in_var_names(adata));
        if n_vn >= 5 {
            let msg = format!(
                "calculate_qc_metrics: found {n_vn} MT-/mt- prefixed gene \
                 symbols in adata.var_names but `qc_vars` is None, so \
                 `pct_counts_mt` will NOT be computed. To compute it, run:\n  \
                 adata.var['mt'] = adata.var_names.str.upper().str.startswith('MT-')\n  \
                 pyscx.accel.calculate_qc_metrics(adata, qc_vars=['mt'])"
            );
            warnings.call_method1("warn", (msg, &user_warning))?;
        } else {
            let n_fn =
                *n_mt_feature_name.get_or_insert_with(|| count_mt_prefix_in_feature_name(adata));
            if n_fn >= 5 {
                let msg = format!(
                    "calculate_qc_metrics: `qc_vars` is None but \
                     adata.var['feature_name'] has {n_fn} MT-/mt- prefixed \
                     gene symbols (CELLxGENE Census layout: integer-string \
                     var_names, gene symbols in var['feature_name']). \
                     pct_counts_mt will NOT be computed. To compute it, run:\n  \
                     adata.var['mt'] = adata.var['feature_name'].str.upper().str.startswith('MT-')\n  \
                     pyscx.accel.calculate_qc_metrics(adata, qc_vars=['mt'])"
                );
                warnings.call_method1("warn", (msg, &user_warning))?;
            }
        }
    }

    // `qc_var_mask_has_true` returns None when the column is absent or
    // not introspectable — in either case we let the downstream code
    // raise its own error rather than guessing.
    for qc_var in qc_vars {
        if qc_var_mask_has_true(adata, qc_var) != Some(false) {
            continue;
        }
        let is_mt = qc_var.eq_ignore_ascii_case("mt");
        // CELLxGENE-shape branch only reachable when `is_mt` — gate the
        // count materialisation on that so non-mt qc_vars don't pay the
        // scan cost.
        let msg = if is_mt {
            let n_vn = *n_mt_var_names.get_or_insert_with(|| count_mt_prefix_in_var_names(adata));
            let n_fn =
                *n_mt_feature_name.get_or_insert_with(|| count_mt_prefix_in_feature_name(adata));
            if n_vn < 5 && n_fn >= 5 {
                format!(
                    "calculate_qc_metrics: adata.var['mt'] mask matches 0 \
                     genes, but adata.var['feature_name'] has \
                     {n_fn} MT-/mt- prefixed symbols — the \
                     CELLxGENE Census layout (integer-string var_names, \
                     gene symbols in var['feature_name']). \
                     pct_counts_mt will be 0 for every cell. Tag mt from \
                     feature_name instead:\n  \
                     adata.var['mt'] = adata.var['feature_name'].str.upper().str.startswith('MT-')\n  \
                     pyscx.accel.calculate_qc_metrics(adata, qc_vars=['mt'])"
                )
            } else {
                format!(
                    "calculate_qc_metrics: qc_var '{qc_var}' mask matches 0 \
                     genes in adata.var['{qc_var}']; total_counts_{qc_var} \
                     and pct_counts_{qc_var} will be 0 for every cell. \
                     Check that adata.var['{qc_var}'] is populated correctly \
                     — e.g. for CELLxGENE Census data, gene symbols live in \
                     adata.var['feature_name'], not in adata.var_names, so \
                     tag as `adata.var['{qc_var}'] = \
                     adata.var['feature_name'].str.upper().str.startswith('MT-')`."
                )
            }
        } else {
            format!(
                "calculate_qc_metrics: qc_var '{qc_var}' mask matches 0 \
                 genes in adata.var['{qc_var}']; total_counts_{qc_var} \
                 and pct_counts_{qc_var} will be 0 for every cell. \
                 Check that adata.var['{qc_var}'] is populated correctly."
            )
        };
        warnings.call_method1("warn", (msg, &user_warning))?;
    }

    Ok(())
}

/// Count gene symbols starting with `MT-` or `mt-` in `adata.var_names`.
/// Returns 0 on any access error (var_names missing, accessor failure,
/// etc.) so the caller can treat 0 as "no MT-tagged genes".
///
/// Uses the pandas `.str.startswith(...).sum()` vectorised accessor so
/// no per-name Python→Rust string copy is needed — the prefix scan runs
/// in C inside pandas, returning a `numpy.int64` scalar.
fn count_mt_prefix_in_var_names(adata: &Bound<'_, PyAny>) -> usize {
    let Ok(var_names) = adata.getattr("var_names") else {
        return 0;
    };
    var_names
        .getattr("str")
        .and_then(|s| s.call_method1("startswith", (("MT-", "mt-"),)))
        .and_then(|m| m.call_method0("sum"))
        .and_then(|v| v.extract::<usize>())
        .unwrap_or(0)
}

/// Count gene symbols starting with `MT-` or `mt-` in
/// `adata.var['feature_name']`. Used to detect the CELLxGENE Census layout
/// (integer-string var_names, real symbols under `feature_name`). Returns
/// 0 when `feature_name` is absent or the accessor chain fails.
///
/// pandas `.str.startswith` works directly on Categorical columns
/// (Census stores `feature_name` as Categorical) — no `.astype(str)` /
/// `Vec<String>` materialisation needed.
fn count_mt_prefix_in_feature_name(adata: &Bound<'_, PyAny>) -> usize {
    let Ok(var) = adata.getattr("var") else {
        return 0;
    };
    // pandas `feature_name in var.columns` check via __contains__.
    let has_col = var
        .call_method1("__contains__", ("feature_name",))
        .ok()
        .and_then(|v| v.is_truthy().ok())
        .unwrap_or(false);
    if !has_col {
        return 0;
    }
    let Ok(col) = var.get_item("feature_name") else {
        return 0;
    };
    col.getattr("str")
        .and_then(|s| s.call_method1("startswith", (("MT-", "mt-"),)))
        .and_then(|m| m.call_method0("sum"))
        .and_then(|v| v.extract::<usize>())
        .unwrap_or(0)
}

/// Compute per-gene `(total_counts, n_cells_by_counts)` via the CSC
/// path. Used by `calculate_qc_metrics(prefer_format="csc")`.
///
/// Capability gate (raised on missing CSC sidecar / non-column-local
/// transforms / row deletion vector active) is delegated to
/// `as_column_source()`. Honors `col_projection` if set on the
/// dataset.
fn compute_gene_axis_csc(x: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<u32>)> {
    if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        let source = backed.as_column_source().ok_or_else(|| {
            PyRuntimeError::new_err(
                "CSC requested but unavailable: file has no CSC sidecar, \
                 or a row deletion vector is active",
            )
        })?;
        let cols_owned: Vec<u32> = match backed.col_projection() {
            Some(c) => c.to_vec(),
            None => (0..backed.shape_val.1 as u32).collect(),
        };
        let sums = projected_agg::col_sums_projected_csc(source, &cols_owned)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let nnz = projected_agg::col_nnz_projected_csc(source, &cols_owned)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        return Ok((sums, nnz));
    }
    if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
        let lazy_src = lazy.as_column_source().ok_or_else(|| {
            PyRuntimeError::new_err(
                "CSC requested but unavailable: file has no CSC sidecar, \
                 the transform chain contains a non-column-local op, or a \
                 row deletion vector is active",
            )
        })?;
        let cols_owned: Vec<u32> = match lazy.col_projection() {
            Some(c) => c.to_vec(),
            None => (0..lazy.shape_val.1 as u32).collect(),
        };
        let sums = projected_agg::col_sums_projected_csc(&lazy_src, &cols_owned)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let nnz = projected_agg::col_nnz_projected_csc(&lazy_src, &cols_owned)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        return Ok((sums, nnz));
    }
    Err(PyRuntimeError::new_err(
        "prefer_format='csc' requires backed or lazy SCX dataset",
    ))
}
