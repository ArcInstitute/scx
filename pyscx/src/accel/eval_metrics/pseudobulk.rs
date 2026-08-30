//! Pseudobulk mean aggregation for the evaluation metrics — streaming /
//! in-memory / dense grouped means and the `pseudobulk_means` entry point.

use pyo3::exceptions::{PyRuntimeError, PyValueError};

use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::ScxLazyTransformedDataset;

// ──────────────────────────────────────────────────────────────────────────────
// device scaffolding (Phase 0 of CELL-EVAL-SCX-GPU-ACC.md)
// ──────────────────────────────────────────────────────────────────────────────
use super::*;

/// Streaming pseudobulk means over a CSR shard source — GPU when `gpu_dev` is
/// `Some`, else CPU.
///
/// Generic over the source so a backed handle can hand over its *view*:
/// `obs_groups` is one label per visible cell, so a subset handle streamed as
/// its raw reader would misalign (and trip the kernel's length guard).
fn agg_streaming<S: scx_format_io::ShardSource + Sync>(
    py: Python<'_>,
    gpu_dev: &EvalGpuDev,
    source: &S,
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    min_cells_per_group: usize,
) -> scx_accel::Result<scx_accel::PseudobulkResult> {
    #[cfg(feature = "gpu")]
    {
        if let Some(dev) = gpu_dev {
            return scx_accel::pseudobulk_means_gpu_streaming(
                dev,
                source,
                obs_groups,
                groupby_columns,
                gene_names,
                min_cells_per_group,
            );
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = gpu_dev;
    }
    py.detach(|| {
        scx_accel::pseudobulk_aggregate(
            source,
            obs_groups,
            groupby_columns,
            gene_names,
            scx_accel::AggregationMethod::Mean,
            min_cells_per_group,
        )
    })
}

/// In-memory CSR pseudobulk means — GPU when `gpu_dev` is `Some`, else CPU.
///
/// Takes an **owned** [`scx_sparse::ScxCsr`], never numpy-borrowed slices:
/// the CPU arm releases the GIL, and a numpy borrow is not safe to read once
/// other Python threads can run. `crate::convert::owned_csr` is how callers
/// get one.
///
/// `pseudobulk_aggregate_inmemory` is the line-for-line twin of the
/// `…_from_slices` entry point this replaced — same accumulation order, so
/// results are bit-identical.
fn agg_inmemory(
    py: Python<'_>,
    gpu_dev: &EvalGpuDev,
    csr: &scx_sparse::ScxCsr,
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    min_cells_per_group: usize,
) -> scx_accel::Result<scx_accel::PseudobulkResult> {
    #[cfg(feature = "gpu")]
    {
        if let Some(dev) = gpu_dev {
            return scx_accel::pseudobulk_means_gpu_from_slices(
                dev,
                csr.shape,
                &csr.indptr,
                &csr.indices,
                &csr.data,
                obs_groups,
                groupby_columns,
                gene_names,
                min_cells_per_group,
            );
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = gpu_dev;
    }
    py.detach(|| {
        scx_accel::pseudobulk_aggregate_inmemory(
            csr,
            obs_groups,
            groupby_columns,
            gene_names,
            scx_accel::AggregationMethod::Mean,
            min_cells_per_group,
        )
    })
}

/// Dense pseudobulk means (in-memory dense X or obsm embedding) — GPU when
/// `gpu_dev` is `Some`, else CPU. Note the GPU dense kernel is f32; for f64
/// embeddings the caller downcasts, so GPU/CPU parity on the `embed_key` path
/// is at the f32 bar (~1e-4), not the 1e-6 of the sparse gene-space paths.
///
/// The CPU arm releases the GIL, so `data` must not be a borrow of a buffer
/// Python can still reach. `crate::convert::owned_dense2_f32` is how callers
/// get one, and the in-memory dense arm of `pseudobulk_means` uses it.
///
/// `compute_obsm_pseudobulk`'s `#[cfg(feature = "gpu")]` branch is the one
/// caller that passes a borrow, and it is sound for two independent reasons:
/// the array it borrows is freshly allocated (`astype("float32")` defaults to
/// `copy=True`, so it aliases nothing the caller can name), and the GPU arm
/// returns above the `py.detach` below. Either alone would suffice; both would
/// have to be broken at once for it to matter. Routing that branch through
/// `owned_dense2_f32` as well would make the rule hold by construction rather
/// than by argument — deferred only because it is GPU-gated code that cannot be
/// exercised on a CPU host.
#[allow(clippy::too_many_arguments)]
fn agg_dense(
    py: Python<'_>,
    gpu_dev: &EvalGpuDev,
    data: &[f32],
    shape: (usize, usize),
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    min_cells_per_group: usize,
) -> scx_accel::Result<scx_accel::PseudobulkResult> {
    #[cfg(feature = "gpu")]
    {
        if let Some(dev) = gpu_dev {
            return scx_accel::pseudobulk_means_gpu_dense(
                dev,
                data,
                shape,
                obs_groups,
                groupby_columns,
                gene_names,
                min_cells_per_group,
            );
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = gpu_dev;
    }
    py.detach(|| {
        scx_accel::pseudobulk_aggregate_dense(
            data,
            shape,
            obs_groups,
            groupby_columns,
            gene_names,
            scx_accel::AggregationMethod::Mean,
            min_cells_per_group,
        )
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// pseudobulk_means
// ──────────────────────────────────────────────────────────────────────────────

/// Compute pseudobulk means (group-by mean on sparse X).
///
/// Aggregates single-cell expression into per-group means, returning the
/// dense means matrix and group names. This is the foundation for
/// perturbation evaluation metrics (pearson_delta, MSE, discrimination
/// score, etc.).
///
/// Supports backed SCX, lazy-transformed, scipy CSR, and dense numpy inputs.
///
/// Args:
///     adata: AnnData object with X and obs columns for groupby
///     groupby: Column name in adata.obs to group by (e.g., "perturbation")
///     min_cells_per_group: Skip groups with fewer cells (default: 1)
///
/// Returns:
///     Tuple of (means, group_names):
///     - means: numpy array of shape [P, G] (float64) — per-group mean expression
///     - group_names: list of str — group names in order
///
/// Example:
///     means, groups = pyscx.accel.pseudobulk_means(adata, "perturbation")
///     # means.shape == (n_perturbations, n_genes)
///     # groups == ["control", "drug_A", "drug_B", ...]
#[pyfunction]
#[pyo3(signature = (adata, groupby, min_cells_per_group=1, device="auto"))]
pub fn pseudobulk_means<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    groupby: &str,
    min_cells_per_group: usize,
    device: &str,
) -> PyResult<Py<PyAny>> {
    // Phase 2: GPU-accelerate the aggregation. Sparse X → gpu_csr, dense/obsm →
    // gpu_dense (route stamped GpuCsr either way — the op ran on GPU); CPU/auto-
    // on-CPU-host / non-gpu-build fall through to the CPU path.
    let resolved = crate::accel::gpu::resolve_device(device)?;
    // Before the stamp below: on a view the stamp is what would trigger
    // anndata's copy-on-write and gather a backed `X`. `_impl` repeats the
    // var-order half of this for its internal callers, where it is a no-op.
    crate::accel::prepare_target(py, adata, "pseudobulk_means")?;
    let info = crate::accel::route::simple_exec_info(
        device,
        cfg!(feature = "gpu"),
        scx_accel::route::AccelRoute::GpuCsr,
        scx_accel::route::AccelRoute::CpuCsr,
    );
    crate::accel::route::announce_route(py, "pseudobulk_means", device, &info);
    // Rolled back if the aggregation below raises — see RouteStamp.
    let route = crate::accel::route::RouteStamp::write(py, adata, "pseudobulk_means", &info)?;
    let gpu_id = if info.route.is_gpu() {
        resolved.gpu_id()
    } else {
        None
    };
    let gpu_dev = eval_make_gpu_dev(gpu_id)?;
    route.settle(pseudobulk_means_impl(
        py,
        adata,
        groupby,
        min_cells_per_group,
        &gpu_dev,
    ))
}

/// Body of [`pseudobulk_means`] without the device/route scaffolding, so
/// internal callers (e.g. `compute_aligned_pseudobulk_means`) reuse the
/// extraction + aggregation without re-stamping `uns["scx_accel"]` or
/// re-validating `device`.
fn pseudobulk_means_impl<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    groupby: &str,
    min_cells_per_group: usize,
    gpu_dev: &EvalGpuDev,
) -> PyResult<Py<PyAny>> {
    // A presentation-ordered backed `X` (`preserve_var_order=True`) has no
    // `ShardSource` spelling: the source emits columns in sorted on-disk order
    // while `adata.var` — and so `gene_names` — stays in request order. Now
    // that this op streams the handle's *view*, the two widths match, so the
    // mismatch would be a silent gene/column permutation instead of a shape
    // error. Refuse, as the other streaming accel ops do.
    crate::accel::reject_preserve_var_order(adata, "pseudobulk_means")?;

    // Extract groupby column from adata.obs as Vec<String>.
    let obs = adata.getattr("obs")?;
    let col = obs.get_item(groupby).map_err(|_| {
        PyValueError::new_err(format!(
            "groupby column '{}' not found in adata.obs",
            groupby
        ))
    })?;
    let labels: Vec<String> = col
        .call_method1("astype", ("str",))?
        .call_method0("tolist")?
        .extract()?;
    let obs_groups = vec![labels];
    let groupby_columns = vec![groupby.to_string()];

    // Get gene names.
    let var = adata.getattr("var")?;
    let var_names = var.getattr("index")?;
    let gene_names: Vec<String> = var_names.call_method0("tolist")?.extract()?;

    // Perform aggregation with Mean method: backed, lazy-transformed, or in-memory.
    // Each branch releases the GIL around the Rust kernel, so each branch first
    // puts the matrix into Rust-owned buffers. Keeping a `PyReadonlyArray`
    // guard alive is *not* enough: it keeps the numpy object alive but leaves it
    // writable from every other Python thread, and rust-numpy borrows carry no
    // synchronization. See `crate::convert::owned_csr`.
    let x = adata.getattr("X")?;
    let result = if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        // The handle's *view* — `obs_groups` / `gene_names` came off
        // `adata.obs` / `adata.var`, so they describe the visible axes.
        let source = backed.as_shard_source();
        drop(backed);
        agg_streaming(
            py,
            gpu_dev,
            &source,
            &obs_groups,
            &groupby_columns,
            &gene_names,
            min_cells_per_group,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    } else if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
        // Lazy-transformed datasets: materialize through the transform pipeline
        // (normalize, log1p, etc.), then aggregate in-memory.
        //
        // `materialize_csr` is what `to_memory_py` runs before wrapping the
        // result in a scipy object, so going to it directly skips the whole
        // numpy round-trip — cheaper than the previous borrow-the-scipy-arrays
        // path, and Rust-owned by construction.
        //
        // The decode stays detached, exactly as `to_memory_py` had it: a
        // `PyRef` cannot cross `detach`, but the `&ScxLazyTransformedDataset`
        // behind it can, and dropping that release here would have been a
        // silent regression on every lazy `pseudobulk_means`.
        let lazy_ref: &ScxLazyTransformedDataset = &lazy;
        let csr = py
            .detach(|| lazy_ref.materialize_csr())
            .map_err(PyRuntimeError::new_err)?;
        drop(lazy);

        agg_inmemory(
            py,
            gpu_dev,
            &csr,
            &obs_groups,
            &groupby_columns,
            &gene_names,
            min_cells_per_group,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    } else {
        // In-memory: dense fast-path for non-sparse inputs. At Replogle
        // scale (24K cells × 18K genes log-normalised) the old
        // `scipy.sparse.csr_matrix(dense)` path was 22 s — dominated by
        // densify-then-CSR-construct churn, not the actual aggregation.
        // The dense kernel skips that.
        let scipy_sparse = crate::pyimport::import_module(py, "scipy.sparse")?;
        let is_sparse = scipy_sparse
            .call_method1("issparse", (&x,))?
            .extract::<bool>()?;

        if !is_sparse {
            // Dense path: copy into an owned row-major f32 buffer and run
            // pseudobulk_aggregate_dense directly.
            //
            // The coercion inside `owned_dense2_f32` already materializes a
            // fresh array for f64, Fortran-order and non-array inputs, so only
            // an already-f32-C-contiguous `X` pays an extra copy — and that is
            // exactly the case where a borrow would have aliased `adata.X`.
            // The temporaries are dropped before `agg_dense`, so peak is
            // `X + copy`, not `X + astype-temp + copy`.
            let (data, shape) = crate::convert::owned_dense2_f32(py, &x, Some("pseudobulk_means"))?;

            agg_dense(
                py,
                gpu_dev,
                &data,
                shape,
                &obs_groups,
                &groupby_columns,
                &gene_names,
                min_cells_per_group,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        } else {
            // True sparse input. `csr_matrix(adata.X)` is an identity op on an
            // already-CSR `X`, so a borrow here would be `adata.X.data` /
            // `.indices` themselves — copy instead.
            let csr = crate::convert::owned_csr(py, &x, Some("pseudobulk_means"))?;

            agg_inmemory(
                py,
                gpu_dev,
                &csr,
                &obs_groups,
                &groupby_columns,
                &gene_names,
                min_cells_per_group,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        }
    };

    if result.n_groups == 0 {
        return Err(PyRuntimeError::new_err(
            "no groups passed the min_cells_per_group filter",
        ));
    }

    // Convert to numpy [P, G] float64 array. `from_vec` is a zero-copy move of
    // the Rust Vec into a numpy array — avoids the slow `np.array(list)` path
    // that allocates a Python float per element before copying back.
    let counts_array = numpy::PyArray::from_vec(py, result.counts).into_any();
    let means_2d = counts_array.call_method1("reshape", ((result.n_groups, result.n_vars),))?;

    // Extract group names (first column of group_labels, since we only have
    // one groupby column).
    let group_names: Vec<String> = result.group_labels.iter().map(|l| l[0].clone()).collect();

    // Return (means, group_names) tuple.
    let tuple = pyo3::types::PyTuple::new(
        py,
        &[
            means_2d.into_any(),
            pyo3::types::PyList::new(py, &group_names)?.into_any(),
        ],
    )?;

    Ok(tuple.into())
}

// ──────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Shared helper: compute pseudobulk means for both AnnData objects, align
/// them to a common set of perturbations (sorted), and return flat f64 arrays.
///
/// Returns: (means_real_flat, means_pred_flat, common_pert_names, n_genes, gene_names)
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(crate) fn compute_aligned_pseudobulk_means<'py>(
    py: Python<'py>,
    adata_real: &Bound<'py, PyAny>,
    adata_pred: &Bound<'py, PyAny>,
    pert_col: &str,
    control: &str,
    embed_key: Option<&str>,
    min_cells_per_group: usize,
    gpu_dev: &EvalGpuDev,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<String>, usize, Vec<String>)> {
    let np = crate::pyimport::import_module(py, "numpy")?;

    // Validate no NaN in perturbation labels (NaN → "nan" is silent and wrong).
    for (label, adata) in [("real", adata_real), ("pred", adata_pred)] {
        let obs = adata.getattr("obs")?;
        let series = obs.get_item(pert_col).map_err(|_| {
            PyValueError::new_err(format!("column '{}' not found in adata.obs", pert_col))
        })?;
        let pd = crate::pyimport::import_module(py, "pandas")?;
        let isna = pd.call_method1("isna", (&series,))?;
        let any_na: bool = isna.call_method0("any")?.extract()?;
        if any_na {
            let n_na: usize = isna.call_method0("sum")?.extract()?;
            return Err(PyValueError::new_err(format!(
                "{label} adata.obs['{}'] has {} NaN values. Remove or fill NaN before calling.",
                pert_col, n_na
            )));
        }
    }

    // For embed_key: use obsm-based pseudobulk (compute manually).
    // For X: use the existing pseudobulk_means infrastructure.
    let (means_real_np, groups_real, means_pred_np, groups_pred, gene_names) =
        if let Some(key) = embed_key {
            // Extract obsm embeddings and compute group means (GPU dense when a
            // device is set — see the f32/f64 precision note on `agg_dense`).
            let (means_r, groups_r) = compute_obsm_pseudobulk(
                py,
                &np,
                adata_real,
                pert_col,
                key,
                min_cells_per_group,
                gpu_dev,
            )?;
            let (means_p, groups_p) = compute_obsm_pseudobulk(
                py,
                &np,
                adata_pred,
                pert_col,
                key,
                min_cells_per_group,
                gpu_dev,
            )?;
            // No gene names when using embeddings
            let n_dims: usize = means_r.getattr("shape")?.extract::<(usize, usize)>()?.1;
            let empty_genes: Vec<String> = (0..n_dims).map(|i| format!("embed_{i}")).collect();
            (means_r, groups_r, means_p, groups_p, empty_genes)
        } else {
            // Use standard X-based pseudobulk means
            let means_real_obj =
                pseudobulk_means_impl(py, adata_real, pert_col, min_cells_per_group, gpu_dev)?;
            let means_pred_obj =
                pseudobulk_means_impl(py, adata_pred, pert_col, min_cells_per_group, gpu_dev)?;

            let real_tuple = means_real_obj.bind(py);
            let pred_tuple = means_pred_obj.bind(py);

            let means_r = real_tuple.get_item(0)?;
            let groups_r: Vec<String> = real_tuple.get_item(1)?.extract()?;
            let means_p = pred_tuple.get_item(0)?;
            let groups_p: Vec<String> = pred_tuple.get_item(1)?.extract()?;

            // Extract gene names from adata_real.var_names
            let var = adata_real.getattr("var")?;
            let var_names = var.getattr("index")?;
            let gene_names: Vec<String> = var_names.call_method0("tolist")?.extract()?;

            (means_r, groups_r, means_p, groups_p, gene_names)
        };

    // ── Align perturbation groups ───────────────────────────────────
    if groups_real.is_empty() || groups_pred.is_empty() {
        return Err(PyValueError::new_err(
            "no groups passed the min_cells_per_group filter",
        ));
    }

    let real_set: std::collections::HashSet<&str> =
        groups_real.iter().map(|s| s.as_str()).collect();
    let pred_set: std::collections::HashSet<&str> =
        groups_pred.iter().map(|s| s.as_str()).collect();

    let mut common: Vec<String> = real_set
        .intersection(&pred_set)
        .map(|s| s.to_string())
        .collect();
    common.sort();

    if common.is_empty() {
        return Err(PyValueError::new_err(
            "no common perturbation groups between real and predicted",
        ));
    }

    // Ensure control is in the common set
    if !common.contains(&control.to_string()) {
        return Err(PyValueError::new_err(format!(
            "control '{}' not found in common perturbation groups. Available: {:?}",
            control, common
        )));
    }

    // Reorder both matrices to common ordering
    let real_idx_map: std::collections::HashMap<&str, usize> = groups_real
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let pred_idx_map: std::collections::HashMap<&str, usize> = groups_pred
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    // `i64`, not `usize`. These are fancy indices, and the pre-4.3 spelling
    // `np.array(vec_of_usize)` went through a Python list, so numpy inferred
    // **int64**. `PyArray1::from_vec` preserves the Rust width instead, and a
    // `Vec<usize>` would land as uint64 — a silent dtype change on an array
    // that reaches numpy's indexing machinery. Collect at the target width.
    let real_indices: Vec<i64> = common
        .iter()
        .map(|s| real_idx_map[s.as_str()] as i64)
        .collect();
    let pred_indices: Vec<i64> = common
        .iter()
        .map(|s| pred_idx_map[s.as_str()] as i64)
        .collect();

    let real_idx_arr = numpy::PyArray1::from_vec(py, real_indices);
    let pred_idx_arr = numpy::PyArray1::from_vec(py, pred_indices);

    let means_real_ordered = means_real_np.get_item(&real_idx_arr)?;
    let means_pred_ordered = means_pred_np.get_item(&pred_idx_arr)?;

    let shape: (usize, usize) = means_real_ordered.getattr("shape")?.extract()?;
    let n_genes = shape.1;

    let means_real_flat: Vec<f64> = means_real_ordered
        .call_method0("ravel")?
        .call_method1("astype", ("float64",))?
        .extract()?;
    let means_pred_flat: Vec<f64> = means_pred_ordered
        .call_method0("ravel")?
        .call_method1("astype", ("float64",))?
        .extract()?;

    Ok((
        means_real_flat,
        means_pred_flat,
        common,
        n_genes,
        gene_names,
    ))
}

/// Compute pseudobulk means from adata.obsm[embed_key] using numpy group-by
/// (CPU) or the GPU dense kernel when `gpu_dev` is set.
#[allow(clippy::too_many_arguments)]
fn compute_obsm_pseudobulk<'py>(
    py: Python<'py>,
    np: &Bound<'py, PyModule>,
    adata: &Bound<'py, PyAny>,
    pert_col: &str,
    embed_key: &str,
    min_cells_per_group: usize,
    gpu_dev: &EvalGpuDev,
) -> PyResult<(Bound<'py, PyAny>, Vec<String>)> {
    // `gpu_dev` is only consumed by the `#[cfg(feature = "gpu")]` branch below.
    #[cfg(not(feature = "gpu"))]
    let _ = gpu_dev;
    let obsm = adata.getattr("obsm")?;
    let embeddings = obsm.get_item(embed_key).map_err(|_| {
        PyValueError::new_err(format!("embed_key '{}' not found in adata.obsm", embed_key))
    })?;
    let matrix = np
        .call_method1("asarray", (&embeddings,))?
        .call_method1("astype", ("float64",))?;
    let shape: (usize, usize) = matrix.getattr("shape")?.extract()?;
    let n_obs = shape.0;
    let n_dims = shape.1;

    // Warn for large materializations (consistent with extract_dense_matrix).
    let n_bytes = n_obs * n_dims * 8; // f64 = 8 bytes
    if n_bytes > 500_000_000 {
        let mb = n_bytes / (1024 * 1024);
        log::warn!("materializing obsm['{embed_key}'] ({n_obs}×{n_dims}) into {mb} MB of memory");
    }

    let labels = extract_obs_column(py, adata, pert_col)?;
    if labels.len() != n_obs {
        return Err(PyValueError::new_err(format!(
            "obs has {} rows but obsm['{embed_key}'] has {n_obs} rows",
            labels.len()
        )));
    }

    // GPU dense path: aggregate the embedding on the device. The DE dense
    // kernel is f32, so the f64 embedding is downcast — parity with the CPU
    // f64 path here is at the f32 bar (see `agg_dense`). Group ordering matches
    // the CPU `BTreeMap` (both lexicographically sorted via `build_group_mapping`).
    #[cfg(feature = "gpu")]
    {
        if gpu_dev.is_some() {
            let arr = np
                .call_method1("asarray", (&embeddings,))?
                .call_method1("astype", ("float32",))?;
            let arr = np.call_method1("ascontiguousarray", (&arr,))?;
            let arr_ro: numpy::PyReadonlyArray2<'_, f32> = arr.extract()?;
            let data_slice = arr_ro
                .as_slice()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let obs_groups = vec![labels.clone()];
            let groupby_columns = vec![pert_col.to_string()];
            let gene_names: Vec<String> = (0..n_dims).map(|i| format!("embed_{i}")).collect();
            let result = agg_dense(
                py,
                gpu_dev,
                data_slice,
                (n_obs, n_dims),
                &obs_groups,
                &groupby_columns,
                &gene_names,
                min_cells_per_group,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            if result.n_groups == 0 {
                return Err(PyRuntimeError::new_err(
                    "no groups passed the min_cells_per_group filter",
                ));
            }
            let means_arr = numpy::PyArray::from_vec(py, result.counts).into_any();
            let means_2d =
                means_arr.call_method1("reshape", ((result.n_groups, result.n_vars),))?;
            let group_names: Vec<String> =
                result.group_labels.iter().map(|l| l[0].clone()).collect();
            return Ok((means_2d, group_names));
        }
    }

    // Group by label and compute mean.
    // BTreeMap guarantees sorted iteration over keys, producing deterministic
    // group ordering. The downstream sort in compute_aligned_pseudobulk_means
    // is a harmless no-op but kept for defensive correctness.
    let mut group_map: std::collections::BTreeMap<String, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, label) in labels.iter().enumerate() {
        group_map.entry(label.clone()).or_default().push(i);
    }

    // Filter by min_cells_per_group
    let groups: Vec<(String, Vec<usize>)> = group_map
        .into_iter()
        .filter(|(_, indices)| indices.len() >= min_cells_per_group)
        .collect();

    if groups.is_empty() {
        return Err(PyRuntimeError::new_err(
            "no groups passed the min_cells_per_group filter",
        ));
    }

    let n_groups = groups.len();
    let mut means_data = vec![0.0f64; n_groups * n_dims];
    let matrix_flat: Vec<f64> = matrix.call_method0("ravel")?.extract()?;

    for (g, (_, indices)) in groups.iter().enumerate() {
        let n = indices.len() as f64;
        for &i in indices {
            for d in 0..n_dims {
                means_data[g * n_dims + d] += matrix_flat[i * n_dims + d];
            }
        }
        for d in 0..n_dims {
            means_data[g * n_dims + d] /= n;
        }
    }

    let group_names: Vec<String> = groups.iter().map(|(name, _)| name.clone()).collect();

    let means_arr = numpy::PyArray::from_vec(py, means_data).into_any();
    let means_2d = means_arr.call_method1("reshape", ((n_groups, n_dims),))?;

    Ok((means_2d, group_names))
}
