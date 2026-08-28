//! Perturbation evaluation metrics — pseudobulk means, perturbation_metrics,
//! energy_distance, discrimination_score, knockdown_efficiency, clustering_agreement.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::ScxLazyTransformedDataset;

// ──────────────────────────────────────────────────────────────────────────────
// device scaffolding (Phase 0 of CELL-EVAL-SCX-GPU-ACC.md)
// ──────────────────────────────────────────────────────────────────────────────

/// Validate the `device=` selector and record the resolved route on
/// `adata.uns["scx_accel"][op]` for a CPU-only eval metric.
///
/// None of the perturbation-evaluation metrics has a GPU kernel yet (the GPU
/// kernels land in later phases of `CELL-EVAL-SCX-GPU-ACC.md`), so this is the
/// no-op API surface that lets callers pass `device=` today and observe the
/// route via `uns["scx_accel"]`:
///
/// * [`resolve_device`](super::gpu::resolve_device) validates the string, so an
///   explicit `device="gpu"`/`"gpu:N"` errors loudly when pyscx was built
///   without the `gpu` feature or no CUDA device is visible. `"auto"`/`"cpu"`
///   always resolve to CPU here.
/// * The route is planned via [`cpu_only_exec_info`](super::route::cpu_only_exec_info):
///   `UserForcedCpu` for `device="cpu"`, `NoCuda` when no GPU is present, and
///   `UnsupportedInputLayout` for an explicit GPU request on a GPU host (there
///   is no GPU eval-metric kernel yet). An explicit `device="gpu"` landing on
///   CPU emits the standard one-shot [`announce_route`](super::route::announce_route)
///   `UserWarning`; `"auto"` stays quiet.
///
/// Stamp on the AnnData the caller would inspect for the route — the prediction
/// (`adata_pred`) for the pair metrics, the sole input for single-input ops.
///
/// Returns the [`RouteStamp`](super::route::RouteStamp) guard: these ops stamp
/// before their compute, so the caller must `commit()` on success or the stamp
/// is rolled back — `uns["scx_accel"][op]` is present iff the op completed.
#[must_use = "commit the returned RouteStamp on success, or the route stamp is rolled back"]
fn scaffold_device_route<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    op: &'static str,
    device: &str,
) -> PyResult<super::route::RouteStamp<'py>> {
    super::gpu::resolve_device(device)?;
    let info = super::route::cpu_only_exec_info(device);
    super::route::announce_route(py, op, device, &info);
    super::route::RouteStamp::write(py, adata, op, &info)
}

// ──────────────────────────────────────────────────────────────────────────────
// GPU dispatch for the pseudobulk aggregation.
//
// `perturbation_metrics` / `pseudobulk_means` GPU-accelerate the pseudobulk
// aggregation (the cell-count-scaling step) via `scx_accel::pseudobulk_means_gpu_*`,
// which wrap the DE pseudobulk kernels. The resulting `[P×G]` f64 means then flow
// through the unchanged host alignment + `compute_bulk_metrics`. The GPU device
// handle is `!Send`, so the GPU folds hold the GIL (they run on-device); the CPU
// fallbacks release it via `py.detach`.
// ──────────────────────────────────────────────────────────────────────────────

/// GPU device id from a resolved device (always `None` without the gpu feature).
fn eval_resolve_gpu_id(resolved: super::gpu::ResolvedDevice) -> Option<usize> {
    #[cfg(feature = "gpu")]
    {
        resolved.gpu_id()
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = resolved;
        None
    }
}

#[cfg(feature = "gpu")]
type EvalGpuDev = Option<scx_accel::GpuDevice>;
#[cfg(not(feature = "gpu"))]
type EvalGpuDev = Option<()>;

/// Build the reusable GPU device handle (`None` → CPU). Errors only on CUDA
/// context-creation failure.
fn eval_make_gpu_dev(gpu_device_id: Option<usize>) -> PyResult<EvalGpuDev> {
    #[cfg(feature = "gpu")]
    {
        match gpu_device_id {
            Some(id) => Ok(Some(scx_accel::GpuDevice::new(id).map_err(|e| {
                PyRuntimeError::new_err(format!("GPU eval-metrics device init failed: {e}"))
            })?)),
            None => Ok(None),
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = gpu_device_id;
        Ok(None)
    }
}

/// Resolve device + plan the route for `energy_distance` / `energy_distance_details`.
///
/// GPU `energy_distance` is gemm-based (f32) and covers **euclidean/cosine only**;
/// L1 (no gemm decomposition) and f64 stay on CPU. So the op is GPU-eligible only
/// when built with the gpu feature AND the metric isn't L1 AND the dtype is f32
/// (the default). Returns the (possibly `None`) device handle and the planned
/// `AccelExecutionInfo`; the caller stamps it via `write_accel_route` after
/// compute. Route pair: `gpu_dense` / `cpu_csr` (cells materialise dense).
fn edist_device_dispatch(
    py: Python<'_>,
    op: &'static str,
    device: &str,
    metric: &str,
    dtype: Option<&str>,
) -> PyResult<(EvalGpuDev, scx_accel::route::AccelExecutionInfo)> {
    let resolved = super::gpu::resolve_device(device)?;
    let metric_is_l1 = matches!(
        metric.to_ascii_lowercase().as_str(),
        "l1" | "manhattan" | "cityblock"
    );
    let dtype_is_f32 = matches!(
        dtype.map(|s| s.to_ascii_lowercase()).as_deref(),
        None | Some("f32") | Some("float32")
    );
    let gpu_eligible = cfg!(feature = "gpu") && !metric_is_l1 && dtype_is_f32;
    let info = super::route::simple_exec_info(
        device,
        gpu_eligible,
        scx_accel::route::AccelRoute::GpuDense,
        scx_accel::route::AccelRoute::CpuCsr,
    );
    super::route::announce_route(py, op, device, &info);
    let gpu_id = if info.route.is_gpu() {
        eval_resolve_gpu_id(resolved)
    } else {
        None
    };
    let gpu_dev = eval_make_gpu_dev(gpu_id)?;
    Ok((gpu_dev, info))
}

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
    let resolved = super::gpu::resolve_device(device)?;
    // Before the stamp below: on a view the stamp is what would trigger
    // anndata's copy-on-write and gather a backed `X`. `_impl` repeats the
    // var-order half of this for its internal callers, where it is a no-op.
    super::prepare_target(py, adata, "pseudobulk_means")?;
    let info = super::route::simple_exec_info(
        device,
        cfg!(feature = "gpu"),
        scx_accel::route::AccelRoute::GpuCsr,
        scx_accel::route::AccelRoute::CpuCsr,
    );
    super::route::announce_route(py, "pseudobulk_means", device, &info);
    // Rolled back if the aggregation below raises — see RouteStamp.
    let route = super::route::RouteStamp::write(py, adata, "pseudobulk_means", &info)?;
    let gpu_id = if info.route.is_gpu() {
        eval_resolve_gpu_id(resolved)
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
    super::reject_preserve_var_order(adata, "pseudobulk_means")?;

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
fn compute_aligned_pseudobulk_means<'py>(
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

/// Trait bridging Rust `f32`/`f64` to numpy dtype strings for
/// `materialize_dense`. Sealed in spirit (only impls in this module).
trait NumpyDtype: numpy::Element + Sized {
    /// numpy `dtype` string passed to `.astype(...)`.
    const NUMPY_NAME: &'static str;
    /// Bytes per element — used to gate the "large materialization" warning.
    const BYTES: usize;

    /// GPU energy distance (Phase 3), specialised by element type: `Some(..)`
    /// for f32 (the GPU path runs), `None` for f64 (no f64 cuBLAS gemm → caller
    /// falls back to the CPU path). Lets the generic `run_energy_distance_inner`
    /// dispatch to the GPU kernel only for f32 without runtime type juggling.
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)]
    fn edist_gpu(
        dev: &scx_accel::GpuDevice,
        real: &[Self],
        pred: &[Self],
        real_groups: &[u32],
        pred_groups: &[u32],
        ctrl_group_idx: u32,
        pert_names: &[String],
        pert_group_indices: &[u32],
        n_dims: usize,
        metric: scx_accel::DistanceMetric,
    ) -> Option<scx_accel::Result<scx_accel::EDistanceResult>>;
}

impl NumpyDtype for f32 {
    const NUMPY_NAME: &'static str = "float32";
    const BYTES: usize = 4;

    #[cfg(feature = "gpu")]
    fn edist_gpu(
        dev: &scx_accel::GpuDevice,
        real: &[f32],
        pred: &[f32],
        real_groups: &[u32],
        pred_groups: &[u32],
        ctrl_group_idx: u32,
        pert_names: &[String],
        pert_group_indices: &[u32],
        n_dims: usize,
        metric: scx_accel::DistanceMetric,
    ) -> Option<scx_accel::Result<scx_accel::EDistanceResult>> {
        Some(scx_accel::compute_energy_distance_gpu(
            dev,
            real,
            pred,
            real_groups,
            pred_groups,
            ctrl_group_idx,
            pert_names,
            pert_group_indices,
            n_dims,
            metric,
        ))
    }
}

impl NumpyDtype for f64 {
    const NUMPY_NAME: &'static str = "float64";
    const BYTES: usize = 8;

    #[cfg(feature = "gpu")]
    fn edist_gpu(
        _dev: &scx_accel::GpuDevice,
        _real: &[f64],
        _pred: &[f64],
        _real_groups: &[u32],
        _pred_groups: &[u32],
        _ctrl_group_idx: u32,
        _pert_names: &[String],
        _pert_group_indices: &[u32],
        _n_dims: usize,
        _metric: scx_accel::DistanceMetric,
    ) -> Option<scx_accel::Result<scx_accel::EDistanceResult>> {
        None // f64 has no cuBLAS gemm → CPU fallback
    }
}

/// Extract a dense `[N, D]` matrix from `adata.X` (or `adata.obsm[embed_key]`)
/// as a flat `Vec<F>`, narrowing to F via numpy's `.astype(...)`.
///
/// Handles scipy sparse (converts to dense), numpy arrays, and SCX backed types.
/// Emits a Python warning for large materializations (> 2 GB estimated).
///
/// Generic over `F: PairwiseFloat + NumpyDtype` — the dtype string passed to
/// `.astype(...)` is inferred from `F::NUMPY_NAME`. This stops the historical
/// unconditional upcast to f64 and lets f32-native callers stay in f32.
fn materialize_dense<'py, F>(
    py: Python<'py>,
    np: &Bound<'py, PyModule>,
    adata: &Bound<'py, PyAny>,
    embed_key: Option<&str>,
) -> PyResult<(Vec<F>, usize, usize)>
where
    F: scx_accel::PairwiseFloat + NumpyDtype,
{
    let matrix_obj = if let Some(key) = embed_key {
        let obsm = adata.getattr("obsm")?;
        obsm.get_item(key).map_err(|_| {
            PyValueError::new_err(format!("embed_key '{}' not found in adata.obsm", key))
        })?
    } else {
        adata.getattr("X")?
    };

    // Check if it's a SCX backed or lazy-transformed type — must materialize.
    let dense = if matrix_obj.is_instance_of::<ScxBackedSparseDataset>()
        || matrix_obj.is_instance_of::<ScxLazyTransformedDataset>()
    {
        let arr = matrix_obj.call_method0("toarray")?;
        arr.call_method1("astype", (F::NUMPY_NAME,))?
    } else {
        // Check for scipy sparse.
        let scipy_sparse = crate::pyimport::import_module(py, "scipy.sparse")?;
        let is_sparse: bool = scipy_sparse
            .call_method1("issparse", (&matrix_obj,))?
            .extract()?;

        if is_sparse {
            let arr = matrix_obj.call_method0("toarray")?;
            arr.call_method1("astype", (F::NUMPY_NAME,))?
        } else {
            np.call_method1("asarray", (&matrix_obj,))?
                .call_method1("astype", (F::NUMPY_NAME,))?
        }
    };

    let shape: (usize, usize) = dense.getattr("shape")?.extract()?;

    // Warn if the dense matrix is very large (> 2 GB).
    let estimated_bytes = shape.0 * shape.1 * F::BYTES;
    if estimated_bytes > 2_000_000_000 {
        let gb = estimated_bytes as f64 / 1e9;
        let warnings = crate::pyimport::import_module(py, "warnings")?;
        warnings.call_method1(
            "warn",
            (format!(
                "energy_distance: materializing a {:.1} GB dense matrix ({} × {} × {} bytes). \
                 Consider using embed_key='X_pca' or subsetting the data.",
                gb,
                shape.0,
                shape.1,
                F::BYTES,
            ),),
        )?;
    }

    // Use numpy's typed array protocol to extract a Vec<F> without going
    // through pyo3's `extract::<Vec<F>>` (which has no generic impl).
    let raveled = dense.call_method0("ravel")?;
    let arr: numpy::PyReadonlyArray1<F> = raveled.extract()?;
    let flat: Vec<F> = arr.as_slice()?.to_vec();

    Ok((flat, shape.0, shape.1))
}

/// Extract a column from adata.obs as Vec<String>.
///
/// Raises ValueError if the column contains NaN values (which would
/// silently become the string `"nan"` after `.astype(str)`).
fn extract_obs_column<'py>(
    _py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    col: &str,
) -> PyResult<Vec<String>> {
    let obs = adata.getattr("obs")?;
    let series = obs
        .get_item(col)
        .map_err(|_| PyValueError::new_err(format!("column '{}' not found in adata.obs", col)))?;

    // Detect NaN values before string conversion (NaN → "nan" is silent and wrong).
    let pd = crate::pyimport::import_module(_py, "pandas")?;
    let isna = pd.call_method1("isna", (&series,))?;
    let any_na: bool = isna.call_method0("any")?.extract()?;
    if any_na {
        let n_na: usize = isna.call_method0("sum")?.extract()?;
        return Err(PyValueError::new_err(format!(
            "column '{}' has {} NaN values. Remove or fill NaN before calling.",
            col, n_na
        )));
    }

    let labels: Vec<String> = series
        .call_method1("astype", ("str",))?
        .call_method0("tolist")?
        .extract()?;
    Ok(labels)
}

// ──────────────────────────────────────────────────────────────────────────────
// perturbation_metrics
// ──────────────────────────────────────────────────────────────────────────────

/// Compute bulk perturbation metrics between real and predicted AnnData objects.
///
/// First computes pseudobulk means for both inputs, then evaluates per-perturbation
/// metrics comparing real vs predicted expression profiles.
///
/// Available metrics:
/// - **pearson_delta**: Pearson correlation of perturbation effects (delta from control)
/// - **mse**: Mean squared error of pseudobulk means
/// - **mae**: Mean absolute error of pseudobulk means
/// - **mse_delta**: MSE of perturbation effects (delta from control)
/// - **mae_delta**: MAE of perturbation effects (delta from control)
///
/// Args:
///     adata_real: AnnData object with real (ground truth) data
///     adata_pred: AnnData object with predicted data
///     pert_col: Column name in obs for perturbation labels (default: "perturbation")
///     control: Label for control perturbation (default: "control")
///     metrics: List of metric names to compute (default: all five)
///     embed_key: If set, use adata.obsm[embed_key] instead of X (default: None).
///         When set, the per-perturbation means are computed in embedding space
///         rather than gene space. Both adatas must expose obsm[embed_key] with
///         the same trailing dimension.
///     min_cells_per_group: Skip groups with fewer cells (default: 1)
///
/// Returns:
///     dict[str, dict[str, float]] — {metric_name: {perturbation: value}}
///
/// Example:
///     results = pyscx.accel.perturbation_metrics(adata_real, adata_pred)
///     # results["pearson_delta"]["drug_A"] == 0.95
///     # results["mse"]["drug_A"] == 0.12
#[pyfunction]
#[pyo3(signature = (adata_real, adata_pred, pert_col="perturbation", control="control", metrics=None, embed_key=None, min_cells_per_group=1, device="auto"))]
#[allow(clippy::too_many_arguments)]
pub fn perturbation_metrics<'py>(
    py: Python<'py>,
    adata_real: &Bound<'py, PyAny>,
    adata_pred: &Bound<'py, PyAny>,
    pert_col: &str,
    control: &str,
    metrics: Option<Vec<String>>,
    embed_key: Option<&str>,
    min_cells_per_group: usize,
    device: &str,
) -> PyResult<Py<PyAny>> {
    // Rebuild an AnnData view as actual before the route stamp / result
    // writes below, so a backed X is not gathered by anndata's
    // copy-on-write. Gene-order agnostic, so no var-order guard.
    //
    // `adata_pred` only: it is the one this op writes to (the stamp and, for
    // the ops that have one, the result column). If a future change starts
    // writing into `adata_real`, that argument needs the same call or a view
    // handed in as `adata_real` will silently gather.
    super::prepare_target_no_var_guard(py, adata_pred, "perturbation_metrics")?;
    // Phase 2 GPU dispatch: the pseudobulk aggregation runs on the GPU (mirrors
    // pdex_nb_glm's skeleton); the five bulk metrics run on the host. `auto` on a
    // CPU host, `cpu`, and non-gpu builds fall through to the CPU path.
    let resolved = super::gpu::resolve_device(device)?;
    let info = super::route::simple_exec_info(
        device,
        cfg!(feature = "gpu"),
        scx_accel::route::AccelRoute::GpuCsr,
        scx_accel::route::AccelRoute::CpuCsr,
    );
    super::route::announce_route(py, "perturbation_metrics", device, &info);
    let gpu_id = if info.route.is_gpu() {
        eval_resolve_gpu_id(resolved)
    } else {
        None
    };
    let gpu_dev = eval_make_gpu_dev(gpu_id)?;

    // Determine which metrics to compute.
    let default_metrics = vec![
        "pearson_delta".to_string(),
        "mse".to_string(),
        "mae".to_string(),
        "mse_delta".to_string(),
        "mae_delta".to_string(),
    ];
    let metric_names = metrics.unwrap_or(default_metrics);

    let bulk_metrics: Vec<scx_accel::BulkMetric> = metric_names
        .iter()
        .map(|name| {
            scx_accel::BulkMetric::parse(name).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown metric '{}'. Valid: pearson_delta, mse, mae, mse_delta, mae_delta",
                    name
                ))
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

    // Use shared helper for pseudobulk computation, alignment, and NaN validation.
    let (means_real_flat, means_pred_flat, common, n_genes, _gene_names) =
        compute_aligned_pseudobulk_means(
            py,
            adata_real,
            adata_pred,
            pert_col,
            control,
            embed_key,
            min_cells_per_group,
            &gpu_dev,
        )?;

    let n_perts = common.len();

    // Find control index.
    let ctrl_idx = common.iter().position(|s| s == control).ok_or_else(|| {
        PyValueError::new_err(format!(
            "control '{}' not found in perturbation groups. Ensure control exists and passes min_cells_per_group filter. Available: {:?}",
            control, common
        ))
    })?;

    // Call Rust bulk metrics computation. Release the GIL — all inputs are
    // owned Vecs / plain scalars, so the closure is Ungil+Send.
    let result = py
        .detach(|| {
            scx_accel::compute_bulk_metrics(
                &means_real_flat,
                &means_pred_flat,
                ctrl_idx,
                n_perts,
                n_genes,
                &common,
                &bulk_metrics,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    super::route::write_accel_route(py, adata_pred, "perturbation_metrics", &info)?;

    // Convert to dict[str, dict[str, float]].
    let outer_dict = PyDict::new(py);
    for (metric_name, values) in &result.metrics {
        let inner_dict = PyDict::new(py);
        for (i, pert_name) in result.pert_names.iter().enumerate() {
            inner_dict.set_item(pert_name.as_str(), values[i])?;
        }
        outer_dict.set_item(metric_name.as_str(), inner_dict)?;
    }

    Ok(outer_dict.into_any().unbind())
}

// ──────────────────────────────────────────────────────────────────────────────
// energy_distance
// ──────────────────────────────────────────────────────────────────────────────

/// Parse `backend` kwarg into `DistanceBackend`.
fn parse_backend(backend: Option<&str>) -> PyResult<scx_accel::DistanceBackend> {
    match backend.map(|s| s.to_ascii_lowercase()).as_deref() {
        None | Some("auto") => Ok(scx_accel::DistanceBackend::Auto),
        Some("gemm") | Some("blas") => Ok(scx_accel::DistanceBackend::Gemm),
        Some("scalar") => Ok(scx_accel::DistanceBackend::Scalar),
        Some(other) => Err(PyValueError::new_err(format!(
            "unknown backend '{other}'. Valid: auto, gemm, scalar"
        ))),
    }
}

/// Dtype dispatch tag, picked out of the `dtype` kwarg.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dtype {
    F32,
    F64,
}

/// Parse `dtype` kwarg. Default is `f32` to maximise throughput.
fn parse_dtype(dtype: Option<&str>) -> PyResult<Dtype> {
    match dtype.map(|s| s.to_ascii_lowercase()).as_deref() {
        None | Some("f32") | Some("float32") => Ok(Dtype::F32),
        Some("f64") | Some("float64") => Ok(Dtype::F64),
        Some(other) => Err(PyValueError::new_err(format!(
            "unknown dtype '{other}'. Valid: f32, f64"
        ))),
    }
}

/// Inner generic implementation of `run_energy_distance`. Materialises both
/// AnnData matrices in the chosen `F` precision, then dispatches to the
/// generic `compute_energy_distance::<F>` kernel.
#[allow(clippy::too_many_arguments)]
fn run_energy_distance_inner<'py, F>(
    py: Python<'py>,
    np: &Bound<'py, PyModule>,
    adata_real: &Bound<'py, PyAny>,
    adata_pred: &Bound<'py, PyAny>,
    pert_col: &str,
    control: &str,
    dist_metric: scx_accel::DistanceMetric,
    embed_key: Option<&str>,
    dist_backend: scx_accel::DistanceBackend,
    gpu_dev: &EvalGpuDev,
) -> PyResult<scx_accel::EDistanceResult>
where
    F: scx_accel::PairwiseFloat + NumpyDtype,
{
    #[cfg(not(feature = "gpu"))]
    let _ = gpu_dev;
    // ── Extract dense matrices from both AnnData objects ────────────
    let (real_flat, n_real, n_dims_real) = materialize_dense::<F>(py, np, adata_real, embed_key)?;
    let (pred_flat, n_pred, n_dims_pred) = materialize_dense::<F>(py, np, adata_pred, embed_key)?;

    if n_dims_real != n_dims_pred {
        return Err(PyValueError::new_err(format!(
            "dimension mismatch: real has {} features but pred has {}",
            n_dims_real, n_dims_pred
        )));
    }
    let n_dims = n_dims_real;

    // ── Extract perturbation labels ─────────────────────────────────
    let real_labels = extract_obs_column(py, adata_real, pert_col)?;
    let pred_labels = extract_obs_column(py, adata_pred, pert_col)?;

    if real_labels.len() != n_real {
        return Err(PyValueError::new_err(format!(
            "real adata has {} obs but X has {} rows",
            real_labels.len(),
            n_real
        )));
    }
    if pred_labels.len() != n_pred {
        return Err(PyValueError::new_err(format!(
            "pred adata has {} obs but X has {} rows",
            pred_labels.len(),
            n_pred
        )));
    }

    // ── Build group → index mapping ─────────────────────────────────
    let mut all_groups = std::collections::BTreeSet::new();
    for l in real_labels.iter().chain(pred_labels.iter()) {
        all_groups.insert(l.as_str());
    }
    let group_to_idx: std::collections::HashMap<&str, u32> = all_groups
        .iter()
        .enumerate()
        .map(|(i, &g)| (g, i as u32))
        .collect();

    let real_groups: Vec<u32> = real_labels
        .iter()
        .map(|l| group_to_idx[l.as_str()])
        .collect();
    let pred_groups: Vec<u32> = pred_labels
        .iter()
        .map(|l| group_to_idx[l.as_str()])
        .collect();

    let ctrl_group_idx = *group_to_idx.get(control).ok_or_else(|| {
        PyValueError::new_err(format!(
            "control '{}' not found in perturbation labels. Available: {:?}",
            control,
            all_groups.iter().collect::<Vec<_>>()
        ))
    })?;

    let mut pert_names = Vec::new();
    let mut pert_group_indices = Vec::new();
    for &g in &all_groups {
        if g != control {
            pert_names.push(g.to_string());
            pert_group_indices.push(group_to_idx[g]);
        }
    }

    if pert_names.is_empty() {
        return Err(PyValueError::new_err(
            "no non-control perturbation groups found",
        ));
    }

    // GPU path (f32 + euclidean/cosine only): F::edist_gpu returns None for f64
    // so the generic dispatch falls through to CPU. GpuDevice is !Send, so the
    // GIL is held around the device work (mirrors agg_backed / pdex_nb_glm).
    #[cfg(feature = "gpu")]
    if let Some(dev) = gpu_dev.as_ref() {
        if !matches!(dist_metric, scx_accel::DistanceMetric::L1) {
            if let Some(res) = F::edist_gpu(
                dev,
                &real_flat,
                &pred_flat,
                &real_groups,
                &pred_groups,
                ctrl_group_idx,
                &pert_names,
                &pert_group_indices,
                n_dims,
                dist_metric,
            ) {
                return res.map_err(|e| PyRuntimeError::new_err(e.to_string()));
            }
        }
    }

    // CPU path. Release the GIL for the O(N²) rayon-parallel kernel. All
    // arguments are owned Vecs or plain scalars.
    py.detach(|| {
        scx_accel::compute_energy_distance::<F>(
            &real_flat,
            &pred_flat,
            &real_groups,
            &pred_groups,
            ctrl_group_idx,
            &pert_names,
            &pert_group_indices,
            n_dims,
            dist_metric,
            dist_backend,
        )
    })
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Shared prep + compute for energy_distance / energy_distance_details.
#[allow(clippy::too_many_arguments)]
fn run_energy_distance<'py>(
    py: Python<'py>,
    adata_real: &Bound<'py, PyAny>,
    adata_pred: &Bound<'py, PyAny>,
    pert_col: &str,
    control: &str,
    metric: &str,
    embed_key: Option<&str>,
    backend: Option<&str>,
    dtype: Option<&str>,
    gpu_dev: &EvalGpuDev,
) -> PyResult<scx_accel::EDistanceResult> {
    let np = crate::pyimport::import_module(py, "numpy")?;

    // Parse distance metric.
    let dist_metric = match metric.to_lowercase().as_str() {
        "euclidean" | "l2" => scx_accel::DistanceMetric::Euclidean,
        "l1" | "manhattan" | "cityblock" => scx_accel::DistanceMetric::L1,
        "cosine" => scx_accel::DistanceMetric::Cosine,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown metric '{}'. Valid: euclidean, l1, cosine",
                metric
            )))
        }
    };

    let dist_backend = parse_backend(backend)?;
    let dtype = parse_dtype(dtype)?;

    match dtype {
        Dtype::F32 => run_energy_distance_inner::<f32>(
            py,
            &np,
            adata_real,
            adata_pred,
            pert_col,
            control,
            dist_metric,
            embed_key,
            dist_backend,
            gpu_dev,
        ),
        Dtype::F64 => run_energy_distance_inner::<f64>(
            py,
            &np,
            adata_real,
            adata_pred,
            pert_col,
            control,
            dist_metric,
            embed_key,
            dist_backend,
            gpu_dev,
        ),
    }
}

/// Score a perturbation-effect prediction by the cell-eval energy-distance metric.
///
/// For each perturbation, computes the energy distance (e-distance) between
/// perturbation cells and control cells — separately on the real and the
/// predicted side — then returns the **Pearson correlation** of the
/// per-perturbation real-vs-predicted e-distance vectors.
///
/// IMPORTANT — polarity: despite the name, the returned scalar is a
/// *correlation*, not a distance. It lies in ``[-1.0, 1.0]`` where
/// **1.0 = perfect prediction and higher is better** (the opposite of a raw
/// distance, where lower is better). Use it directly as a score; do not invert
/// it. The per-perturbation e-distances themselves (where smaller = more
/// similar to control) are exposed via :func:`energy_distance_details`.
///
/// Returns ``nan`` when the correlation is undefined — i.e. when either side's
/// e-distance vector has zero variance (e.g. a constant or control-broadcast
/// predictor that gives every perturbation the same e-distance), or when there
/// is only a single non-control perturbation.
///
/// This is the most expensive cell-eval metric — O(N²) pairwise distances
/// per perturbation. The Rust implementation avoids materializing N×N
/// distance matrices (streaming accumulation), precomputes control
/// self-distances once, and parallelizes across perturbations with rayon.
///
/// Args:
///     adata_real: AnnData with real (ground truth) data
///     adata_pred: AnnData with predicted data
///     pert_col: Column name in obs for perturbation labels (default: "perturbation")
///     control: Label for control perturbation (default: "control")
///     metric: Distance metric — "euclidean" (default), "l1", or "cosine"
///     embed_key: If set, use adata.obsm[embed_key] instead of X (default: None)
///     backend: Distance kernel backend — "auto" (default), "gemm", or "scalar".
///         "auto" uses faer-backed gemm for euclidean/cosine and the scalar
///         row-by-row path for L1. "gemm" forces the gemm path (errors on L1).
///         "scalar" forces the scalar path (matches the historical implementation).
///     dtype: Element precision for the dense kernel — "f32" (default) or "f64".
///         Reductions always accumulate in f64; the dtype only controls the
///         matmul / per-pair compute precision. f32 is ~1.5–2× faster on AVX2
///         and matches f64 within atol=1e-4 on log-normalised inputs.
///
/// Returns:
///     float — Pearson correlation of per-perturbation real-vs-predicted
///     e-distances, in [-1.0, 1.0] (1.0 = perfect, higher = better; `nan` if
///     either side's e-distance vector is constant). For the underlying
///     per-perturbation e-distance vectors, use `energy_distance_details`.
///
/// Example:
///     corr = pyscx.accel.energy_distance(adata_real, adata_pred)
///     # corr ≈ 0.85 means real and predicted perturbation effects
///     # have similar relative magnitudes (higher = better, 1.0 = perfect)
#[pyfunction]
#[pyo3(signature = (adata_real, adata_pred, pert_col="perturbation", control="control", metric="euclidean", embed_key=None, backend=None, dtype=None, device="auto"))]
#[allow(clippy::too_many_arguments)]
pub fn energy_distance<'py>(
    py: Python<'py>,
    adata_real: &Bound<'py, PyAny>,
    adata_pred: &Bound<'py, PyAny>,
    pert_col: &str,
    control: &str,
    metric: &str,
    embed_key: Option<&str>,
    backend: Option<&str>,
    dtype: Option<&str>,
    device: &str,
) -> PyResult<f64> {
    // Rebuild an AnnData view as actual before the route stamp / result
    // writes below, so a backed X is not gathered by anndata's
    // copy-on-write. Gene-order agnostic, so no var-order guard.
    super::prepare_target_no_var_guard(py, adata_pred, "energy_distance")?;
    let (gpu_dev, info) = edist_device_dispatch(py, "energy_distance", device, metric, dtype)?;
    let result = run_energy_distance(
        py, adata_real, adata_pred, pert_col, control, metric, embed_key, backend, dtype, &gpu_dev,
    )?;
    // Stamped after the compute, so there is nothing to roll back — a raise
    // above never reaches this line. If this ever moves to a pre-dispatch
    // stamp (as the thirteen progress-reporting ops did), it must switch to
    // `RouteStamp::write` or it will leave a stamp behind on failure.
    super::route::write_accel_route(py, adata_pred, "energy_distance", &info)?;
    Ok(result.correlation)
}

/// Compute energy distance with per-perturbation details.
///
/// Same inputs, distance kernel, and `nan` behavior as `energy_distance` (see
/// that function for the full argument reference), but returns the full
/// per-perturbation e-distance vectors alongside the summary Pearson
/// correlation.
///
/// The two scalar conventions differ — keep them straight:
///   - `"correlation"` is the score returned by `energy_distance`: a Pearson
///     correlation in [-1.0, 1.0] where **1.0 = perfect, higher = better**
///     (`nan` if either side's e-distance vector is constant / zero-variance).
///   - `"d_real"` / `"d_pred"` are raw **e-distances**: 0 = identical to
///     control, larger = more perturbed (lower is *not* better — they are the
///     per-perturbation effect magnitudes being compared, not a score).
///
/// Returns:
///     dict with keys:
///     - `"correlation"`: float — Pearson correlation of real vs pred e-distances
///       (1.0 = perfect, higher = better; `nan` on a constant predictor)
///     - `"d_real"`: dict[str, float] — per-perturbation e-distance on real side
///     - `"d_pred"`: dict[str, float] — per-perturbation e-distance on pred side
///     - `"pert_names"`: list[str] — perturbation names in order
///
/// Example:
///     out = pyscx.accel.energy_distance_details(adata_real, adata_pred)
///     # out["correlation"] ≈ 0.85   (higher = better, 1.0 = perfect)
///     # out["d_real"]["drug_A"] == 12.34
#[pyfunction]
#[pyo3(signature = (adata_real, adata_pred, pert_col="perturbation", control="control", metric="euclidean", embed_key=None, backend=None, dtype=None, device="auto"))]
#[allow(clippy::too_many_arguments)]
pub fn energy_distance_details<'py>(
    py: Python<'py>,
    adata_real: &Bound<'py, PyAny>,
    adata_pred: &Bound<'py, PyAny>,
    pert_col: &str,
    control: &str,
    metric: &str,
    embed_key: Option<&str>,
    backend: Option<&str>,
    dtype: Option<&str>,
    device: &str,
) -> PyResult<Py<PyAny>> {
    // Rebuild an AnnData view as actual before the route stamp / result
    // writes below, so a backed X is not gathered by anndata's
    // copy-on-write. Gene-order agnostic, so no var-order guard.
    super::prepare_target_no_var_guard(py, adata_pred, "energy_distance_details")?;
    let (gpu_dev, info) =
        edist_device_dispatch(py, "energy_distance_details", device, metric, dtype)?;
    let result = run_energy_distance(
        py, adata_real, adata_pred, pert_col, control, metric, embed_key, backend, dtype, &gpu_dev,
    )?;
    super::route::write_accel_route(py, adata_pred, "energy_distance_details", &info)?;

    let d_real = PyDict::new(py);
    let d_pred = PyDict::new(py);
    for (i, name) in result.pert_names.iter().enumerate() {
        d_real.set_item(name.as_str(), result.d_real[i])?;
        d_pred.set_item(name.as_str(), result.d_pred[i])?;
    }

    let out = PyDict::new(py);
    out.set_item("correlation", result.correlation)?;
    out.set_item("d_real", d_real)?;
    out.set_item("d_pred", d_pred)?;
    out.set_item(
        "pert_names",
        pyo3::types::PyList::new(py, &result.pert_names)?,
    )?;

    Ok(out.into_any().unbind())
}

// ──────────────────────────────────────────────────────────────────────────────
// discrimination_score
// ──────────────────────────────────────────────────────────────────────────────

/// Compute discrimination score between real and predicted perturbation data.
///
/// For each perturbation, computes how well the predicted perturbation effect
/// ranks among all real perturbation effects by pairwise distance. A score of
/// 1.0 means the correct perturbation is the closest match; 0.0 means it is
/// the furthest.
///
/// This metric builds on pseudobulk means: effects are computed as
/// means[pert] - means[control] for each perturbation.
///
/// When `exclude_target_gene=True` (default) and not using embeddings, the
/// gene column matching each perturbation's name is excluded from the distance
/// computation, preventing trivially high scores from knockdown-gene dominance.
///
/// Args:
///     adata_real: AnnData with real (ground truth) data
///     adata_pred: AnnData with predicted data
///     pert_col: Column name in obs for perturbation labels (default: "perturbation")
///     control: Label for control perturbation (default: "control")
///     metric: Distance metric — "l1" (default), "l2"/"euclidean", or "cosine"
///     exclude_target_gene: Exclude gene named after perturbation (default: True)
///     embed_key: If set, use adata.obsm[embed_key] instead of X (default: None).
///         When set, exclude_target_gene is ignored (gene names don't apply to
///         embeddings). When metric is L1/manhattan/cityblock, embed_key is forced
///         to None (matching cell-eval behavior).
///     min_cells_per_group: Skip groups with fewer cells (default: 1)
///
/// Returns:
///     dict[str, float] — {perturbation_name: normalized_rank_score}
///
/// Example:
///     scores = pyscx.accel.discrimination_score(adata_real, adata_pred)
///     # scores["drug_A"] == 0.95  (high = good prediction)
#[pyfunction]
#[pyo3(signature = (adata_real, adata_pred, pert_col="perturbation", control="control", metric="l1", exclude_target_gene=true, embed_key=None, min_cells_per_group=1, device="auto"))]
#[allow(clippy::too_many_arguments)]
pub fn discrimination_score<'py>(
    py: Python<'py>,
    adata_real: &Bound<'py, PyAny>,
    adata_pred: &Bound<'py, PyAny>,
    pert_col: &str,
    control: &str,
    metric: &str,
    exclude_target_gene: bool,
    embed_key: Option<&str>,
    min_cells_per_group: usize,
    device: &str,
) -> PyResult<Py<PyAny>> {
    // Rebuild an AnnData view as actual before the route stamp / result
    // writes below, so a backed X is not gathered by anndata's
    // copy-on-write. Gene-order agnostic, so no var-order guard.
    super::prepare_target_no_var_guard(py, adata_pred, "discrimination_score")?;
    let route = scaffold_device_route(py, adata_pred, "discrimination_score", device)?;

    // Parse distance metric.
    let dist_metric = match metric.to_lowercase().as_str() {
        "euclidean" | "l2" => scx_accel::DistanceMetric::Euclidean,
        "l1" | "manhattan" | "cityblock" => scx_accel::DistanceMetric::L1,
        "cosine" => scx_accel::DistanceMetric::Cosine,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown metric '{}'. Valid: l1, l2, euclidean, cosine",
                metric
            )))
        }
    };

    // Cell-eval behavior: L1/manhattan/cityblock forces embed_key=None
    let effective_embed_key = if matches!(
        metric.to_lowercase().as_str(),
        "l1" | "manhattan" | "cityblock"
    ) {
        None
    } else {
        embed_key
    };

    // Determine if we're using embeddings (affects exclude_target_gene behavior).
    let using_embeddings = effective_embed_key.is_some();

    // ── Compute pseudobulk means for both real and predicted ────────
    let (means_real_flat, means_pred_flat, common, n_genes, gene_names) =
        compute_aligned_pseudobulk_means(
            py,
            adata_real,
            adata_pred,
            pert_col,
            control,
            effective_embed_key,
            min_cells_per_group,
            &None, // discrimination_score / clustering_agreement stay CPU (Phase 3)
        )?;

    let n_perts = common.len();

    // Find control index.
    let ctrl_idx = common.iter().position(|s| s == control).ok_or_else(|| {
        PyValueError::new_err(format!(
            "control '{}' not found in perturbation groups. Available: {:?}",
            control, common
        ))
    })?;

    // ── Compute perturbation effects: means[p] - means[ctrl] ────────
    let ctrl_real = &means_real_flat[ctrl_idx * n_genes..(ctrl_idx + 1) * n_genes];
    let ctrl_pred = &means_pred_flat[ctrl_idx * n_genes..(ctrl_idx + 1) * n_genes];

    // Build effect matrices (excluding control row).
    let n_output = n_perts - 1;
    let mut real_effects = Vec::with_capacity(n_output * n_genes);
    let mut pred_effects = Vec::with_capacity(n_output * n_genes);
    let mut output_pert_names = Vec::with_capacity(n_output);

    for p in 0..n_perts {
        if p == ctrl_idx {
            continue;
        }
        output_pert_names.push(common[p].clone());
        let row_real = &means_real_flat[p * n_genes..(p + 1) * n_genes];
        let row_pred = &means_pred_flat[p * n_genes..(p + 1) * n_genes];
        for g in 0..n_genes {
            real_effects.push(row_real[g] - ctrl_real[g]);
        }
        for g in 0..n_genes {
            pred_effects.push(row_pred[g] - ctrl_pred[g]);
        }
    }

    // ── Gene exclusion setup ────────────────────────────────────────
    // exclude_target_gene only applies when not using embeddings.
    let effective_exclude = exclude_target_gene && !using_embeddings;
    let gene_names_ref = if effective_exclude {
        Some(gene_names.as_slice())
    } else {
        None
    };

    // ── Call Rust discrimination score ───────────────────────────────
    // Release the GIL for the rayon-parallel inner loop.
    let result = py
        .detach(|| {
            scx_accel::compute_discrimination_score(
                &real_effects,
                &pred_effects,
                n_output,
                n_genes,
                &output_pert_names,
                gene_names_ref,
                dist_metric,
                effective_exclude,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // ── Convert to dict[str, float] ─────────────────────────────────
    let dict = PyDict::new(py);
    for (i, pert_name) in result.pert_names.iter().enumerate() {
        dict.set_item(pert_name.as_str(), result.scores[i])?;
    }

    route.commit();
    Ok(dict.into_any().unbind())
}

// ──────────────────────────────────────────────────────────────────────────────
// knockdown_efficiency
// ──────────────────────────────────────────────────────────────────────────────

/// Compute per-cell knockdown efficiency and log deviation.
///
/// For each perturbed cell, this function measures how effectively the
/// perturbation knocked down its target gene. Perturbation names must match
/// gene names (var_names) for the lookup to work.
///
/// **Knockdown efficiency** (computed on normalized, NOT log-transformed data):
///     KD = 1 - x_target / (μ_control[target_gene] + eps)
///
/// **Log fold change** (computed on log1p-transformed data):
///     FC = x_log[target_gene] - log1p(μ_control[target_gene])
///
/// Control cells and cells whose perturbation name doesn't match any gene
/// will have NaN in both output columns.
///
/// Args:
///     adata: AnnData object with sparse or dense X matrix.
///         Must have obs[pert_col] with perturbation labels where perturbation
///         names match gene names (var_names).
///     pert_col: Column name in adata.obs for perturbation labels (default: "perturbation")
///     control: Label for control perturbation (default: "control")
///     eps: Small value for numerical stability (default: 1e-8)
///
/// Returns:
///     None — writes two columns to adata.obs:
///     - "KnockDownEfficiency": per-cell knockdown efficiency (float32)
///     - "KnockDownGeneFC": per-cell log fold change (float32)
#[pyfunction]
#[pyo3(signature = (adata, pert_col="perturbation", control="control", eps=1e-8, device="auto"))]
pub fn knockdown_efficiency<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    pert_col: &str,
    control: &str,
    eps: f64,
    device: &str,
) -> PyResult<()> {
    // Rebuild an AnnData view as actual before the route stamp / result
    // writes below, so a backed X is not gathered by anndata's
    // copy-on-write. Gene-order agnostic, so no var-order guard.
    super::prepare_target_no_var_guard(py, adata, "knockdown_efficiency")?;
    let route = scaffold_device_route(py, adata, "knockdown_efficiency", device)?;
    let np = crate::pyimport::import_module(py, "numpy")?;

    // ── Extract perturbation labels ─────────────────────────────────
    let pert_labels = extract_obs_column(py, adata, pert_col)?;
    let n_obs = pert_labels.len();

    // ── Extract gene names ──────────────────────────────────────────
    let var = adata.getattr("var")?;
    let var_names = var.getattr("index")?;
    let gene_names: Vec<String> = var_names.call_method0("tolist")?.extract()?;
    let n_vars = gene_names.len();

    // ── Extract CSR data from adata.X ───────────────────────────────
    let x = adata.getattr("X")?;
    let scipy_sparse = crate::pyimport::import_module(py, "scipy.sparse")?;

    // Get CSR matrix — handle backed, lazy-transformed, sparse, and dense inputs.
    let csr_obj = if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        // Backed SCX: materialize as *sparse* scipy CSR via `to_memory`.
        // `toarray()` would densify the whole matrix (~240 GB at 1M × 30K).
        drop(backed);
        x.call_method0("to_memory")?
    } else if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
        // Lazy-transformed: materialize through transforms
        let scipy_csr = lazy.to_memory_py(py)?;
        drop(lazy);
        scipy_csr
    } else {
        let is_sparse: bool = scipy_sparse.call_method1("issparse", (&x,))?.extract()?;
        if is_sparse {
            scipy_sparse.call_method1("csr_matrix", (&x,))?
        } else {
            let arr = np
                .call_method1("asarray", (&x,))?
                .call_method1("astype", ("float32",))?;
            scipy_sparse.call_method1("csr_matrix", (&arr,))?
        }
    };

    // Enforce sorted column indices on the CSR before extracting raw arrays.
    // `scx_accel::compute_knockdown_efficiency` looks up the target gene's
    // value via `csr_get_value`'s `binary_search` over each row's column
    // indices (see `scx-accel/src/eval_metrics/knockdown.rs`); on an
    // unsorted CSR that binary search silently returns 0.0 whenever the
    // target column is positioned out-of-order, collapsing every
    // perturbed cell's `KnockDownEfficiency` to `1.0 - 0.0 / (baseline +
    // eps) ≈ 1.0` regardless of the actual expression — a uniformly
    // degenerate failure mode. Real h5ad files (e.g. `pbmc10k.h5ad`)
    // routinely arrive with `has_sorted_indices == False`. `ensure_csr`
    // short-circuits when sorted (cheap clone for the backed / lazy /
    // dense-derived CSR branches above) and calls `.sorted_indices()`
    // when not.
    let (csr_obj, _) = crate::convert::ensure_csr(py, &csr_obj, /* in_place */ false)?;

    let shape: (usize, usize) = csr_obj.getattr("shape")?.extract()?;
    if shape.0 != n_obs {
        return Err(PyValueError::new_err(format!(
            "X has {} rows but obs has {} rows",
            shape.0, n_obs
        )));
    }
    if shape.1 != n_vars {
        return Err(PyValueError::new_err(format!(
            "X has {} columns but var has {} genes",
            shape.1, n_vars
        )));
    }

    // ── Copy the CSR arrays into Rust-owned buffers ─────────────────
    // `ensure_csr` above returns the input unchanged when it is already sorted
    // CSR, so these arrays can be `adata.X`'s own. The kernels below run with
    // the GIL released, and a numpy borrow is not safe to read there: it is not
    // GIL-bound, does not clear numpy's WRITEABLE flag, and a raced `indices`
    // value is consumed as a column index. See `crate::convert::owned_csr`.
    let csr = crate::convert::owned_csr(py, &csr_obj, Some("knockdown_efficiency"))?;
    let indptr = csr.indptr.as_slice();
    let indices = csr.indices.as_slice();
    let data = csr.data.as_slice();

    // ── Compute control baseline + knockdown efficiency + log deviation ──
    // Release the GIL for all three kernels. The log-transform allocations
    // also happen inside the closure to avoid a round-trip.
    let (efficiency, log_fc) = py
        .detach(|| -> scx_accel::Result<_> {
            let baseline = scx_accel::compute_control_baseline(
                indptr,
                indices,
                data,
                &pert_labels,
                control,
                n_vars,
            )?;

            let efficiency = scx_accel::compute_knockdown_efficiency(
                indptr,
                indices,
                data,
                &pert_labels,
                control,
                &gene_names,
                &baseline,
                eps,
            )?;

            // Arc-bench computes log deviation AFTER log1p. Baseline is
            // precomputed as `log1p(mean)`; the kernel applies `log1p` to
            // each per-cell value during its binary-search column extraction,
            // so we avoid allocating a full second copy of `data`.
            let baseline_log: Vec<f64> = baseline.iter().map(|&v| v.ln_1p()).collect();

            let log_fc = scx_accel::compute_log_deviation(
                indptr,
                indices,
                data,
                &pert_labels,
                control,
                &gene_names,
                &baseline_log,
                true,
            )?;

            Ok((efficiency, log_fc))
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // ── Write results to adata.obs ──────────────────────────────────
    let obs = adata.getattr("obs")?;
    let eff_array = numpy::PyArray::from_vec(py, efficiency);
    obs.set_item("KnockDownEfficiency", eff_array)?;

    let fc_array = numpy::PyArray::from_vec(py, log_fc);
    obs.set_item("KnockDownGeneFC", fc_array)?;

    route.commit();
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// clustering_agreement
// ──────────────────────────────────────────────────────────────────────────────

/// Compute clustering agreement between real and predicted perturbation centroids.
///
/// Builds centroid matrices (pseudobulk means per perturbation, excluding
/// control), constructs kNN graphs, clusters via Leiden at multiple resolutions,
/// and scores the agreement between real and predicted cluster assignments
/// using AMI, NMI, or ARI.
///
/// Args:
///     adata_real: AnnData with real (ground truth) data
///     adata_pred: AnnData with predicted data
///     pert_col: Column name in obs for perturbation labels (default: "perturbation")
///     control: Label for control perturbation (default: "control")
///     metric: Agreement metric — "ami" (default), "nmi", or "ari"
///     real_resolution: Leiden resolution for real centroids (default: 1.0)
///     pred_resolutions: Tuple of Leiden resolutions to sweep for predicted
///         centroids (default: (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0))
///     n_neighbors: Number of neighbors for kNN graph (default: 15)
///     embed_key: If set, use adata.obsm[embed_key] instead of X (default: None)
///     min_cells_per_group: Skip groups with fewer cells (default: 1)
///
/// Returns:
///     float — Best clustering agreement score across predicted resolutions
#[pyfunction]
#[pyo3(signature = (adata_real, adata_pred, pert_col="perturbation", control="control", metric="ami", real_resolution=1.0, pred_resolutions=None, n_neighbors=15, embed_key=None, min_cells_per_group=1, device="auto"))]
#[allow(clippy::too_many_arguments)]
pub fn clustering_agreement<'py>(
    py: Python<'py>,
    adata_real: &Bound<'py, PyAny>,
    adata_pred: &Bound<'py, PyAny>,
    pert_col: &str,
    control: &str,
    metric: &str,
    real_resolution: f64,
    pred_resolutions: Option<Vec<f64>>,
    n_neighbors: usize,
    embed_key: Option<&str>,
    min_cells_per_group: usize,
    device: &str,
) -> PyResult<f64> {
    // Rebuild an AnnData view as actual before the route stamp / result
    // writes below, so a backed X is not gathered by anndata's
    // copy-on-write. Gene-order agnostic, so no var-order guard.
    super::prepare_target_no_var_guard(py, adata_pred, "clustering_agreement")?;
    let route = scaffold_device_route(py, adata_pred, "clustering_agreement", device)?;

    // Native-Rust path: kNN graph + Leiden clustering live entirely in
    // `scx_accel`, so the entire hot path runs under `py.detach`.
    // No scanpy / anndata / igraph imports — the Leiden defaults
    // (`seed=0`, `parallel=false`, `max_iterations=2`) are calibrated
    // against the C++ leidenalg / python-igraph references; HNSW defaults
    // (`ef_construction=200`, `ef_search=50`) are kept hidden inside the
    // implementation since scanpy's `sc.pp.neighbors` similarly hides the
    // exact-vs-approx knobs from this caller.

    // Parse clustering metric.
    let clustering_metric = scx_accel::ClusteringMetric::parse(metric).ok_or_else(|| {
        PyValueError::new_err(format!("unknown metric '{}'. Valid: ami, nmi, ari", metric))
    })?;

    let default_resolutions = vec![0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0];
    let resolutions = pred_resolutions.unwrap_or(default_resolutions);

    if resolutions.is_empty() {
        return Err(PyValueError::new_err("pred_resolutions must not be empty"));
    }

    // ── Compute pseudobulk means (centroids) for both sides ─────────
    let (means_real_flat, means_pred_flat, common, n_genes, _gene_names) =
        compute_aligned_pseudobulk_means(
            py,
            adata_real,
            adata_pred,
            pert_col,
            control,
            embed_key,
            min_cells_per_group,
            &None, // discrimination_score / clustering_agreement stay CPU (Phase 3)
        )?;

    let n_perts = common.len();

    // Find control index and filter it out.
    let ctrl_idx = common.iter().position(|s| s == control);

    // Build non-control perturbation names and centroid matrices.
    let mut pert_names: Vec<String> = Vec::with_capacity(n_perts);
    let mut centroids_real: Vec<f64> = Vec::with_capacity(n_perts * n_genes);
    let mut centroids_pred: Vec<f64> = Vec::with_capacity(n_perts * n_genes);

    for p in 0..n_perts {
        if Some(p) == ctrl_idx {
            continue;
        }
        pert_names.push(common[p].clone());
        centroids_real.extend_from_slice(&means_real_flat[p * n_genes..(p + 1) * n_genes]);
        centroids_pred.extend_from_slice(&means_pred_flat[p * n_genes..(p + 1) * n_genes]);
    }

    let n_output = pert_names.len();
    if n_output < 2 {
        return Err(PyValueError::new_err(format!(
            "need at least 2 non-control perturbations for clustering agreement, got {}",
            n_output
        )));
    }

    // Sort centroids by perturbation name to align between real and pred.
    let mut sorted_indices: Vec<usize> = (0..n_output).collect();
    sorted_indices.sort_by(|&a, &b| pert_names[a].cmp(&pert_names[b]));

    // `scx_accel::neighbors::build_knn_graph` takes `&[f32]` only. Cast
    // centroids row-by-row in the same step that reorders by pert name.
    // Log-normalised counts are well below `f32::MAX`, but flag overflow
    // defensively in case a caller passes raw counts via `embed_key`.
    let mut sorted_real_f32 = vec![0.0f32; n_output * n_genes];
    let mut sorted_pred_f32 = vec![0.0f32; n_output * n_genes];
    let mut overflow_seen = false;
    for (new_idx, &old_idx) in sorted_indices.iter().enumerate() {
        let dst_real = &mut sorted_real_f32[new_idx * n_genes..(new_idx + 1) * n_genes];
        let dst_pred = &mut sorted_pred_f32[new_idx * n_genes..(new_idx + 1) * n_genes];
        let src_real = &centroids_real[old_idx * n_genes..(old_idx + 1) * n_genes];
        let src_pred = &centroids_pred[old_idx * n_genes..(old_idx + 1) * n_genes];
        for (d, &s) in dst_real.iter_mut().zip(src_real.iter()) {
            if !overflow_seen && s.abs() > f32::MAX as f64 {
                overflow_seen = true;
            }
            *d = s as f32;
        }
        for (d, &s) in dst_pred.iter_mut().zip(src_pred.iter()) {
            if !overflow_seen && s.abs() > f32::MAX as f64 {
                overflow_seen = true;
            }
            *d = s as f32;
        }
    }
    if overflow_seen {
        let warnings = crate::pyimport::import_module(py, "warnings")?;
        warnings.call_method1(
            "warn",
            (
                "clustering_agreement: centroid value exceeds f32::MAX during \
              cast — affected entries become +/- infinity (f64-as-f32 in Rust \
              does not saturate), which will poison HNSW distance computations. \
              Consider supplying log-normalised counts via embed_key.",
            ),
        )?;
    }

    let effective_n_neighbors = n_neighbors.min(n_output - 1);

    // ── Build kNN + run Leiden, all in Rust, GIL released ───────────
    //
    // Per-phase profiling: timers log to
    // `pyscx::accel::eval_metrics::clustering_agreement` at debug.
    // Enable via `RUST_LOG=pyscx::accel::eval_metrics=debug`. Production
    // callers see no output; the cost of one `Instant::now()` per phase is
    // negligible (~30 ns total) compared to the kNN / Leiden work.
    use std::time::Instant;
    let resolutions_owned = resolutions.clone();
    let n_resolutions = resolutions_owned.len();
    let best_score: scx_accel::Result<f64> = py.detach(|| {
        let t_total = Instant::now();

        // Phase: real-side kNN graph build.
        let t = Instant::now();
        let real_knn = scx_accel::build_knn_graph(
            &sorted_real_f32,
            n_output,
            n_genes,
            effective_n_neighbors,
            /*ef_construction=*/ 200,
            /*ef_search=*/ 50,
            /*seed=*/ 0,
        )?;
        let real_knn_ms = t.elapsed().as_secs_f64() * 1000.0;

        // Phase: real-side Leiden.
        let t = Instant::now();
        let real_leiden = scx_accel::leiden(
            &real_knn.conn_indptr,
            &real_knn.conn_indices,
            &real_knn.conn_data,
            n_output,
            real_resolution,
            /*seed=*/ 0,
            /*max_iterations=*/ 2,
            /*parallel=*/ false,
        )?;
        let real_leiden_ms = t.elapsed().as_secs_f64() * 1000.0;
        // `LeidenResult.membership` is `Vec<usize>`; the AMI/NMI/ARI scoring
        // signature is `&[u32]`. Cast row-by-row — community counts in the
        // centroid graph (≤ 3000 nodes) cannot exceed u32::MAX.
        let real_labels_u32: Vec<u32> = real_leiden.membership.iter().map(|&c| c as u32).collect();

        // Phase: pred-side kNN graph build.
        let t = Instant::now();
        let pred_knn = scx_accel::build_knn_graph(
            &sorted_pred_f32,
            n_output,
            n_genes,
            effective_n_neighbors,
            200,
            50,
            0,
        )?;
        let pred_knn_ms = t.elapsed().as_secs_f64() * 1000.0;

        // Phase: pred-side resolution sweep (Leiden + AMI scoring fused).
        // Resolutions are independent — fan out across rayon. Inner Leiden
        // is `parallel=false`, so the only nested rayon use is matmul-free
        // graph-coloring; safe to parallelise across the (typically 7)
        // resolutions. `try_reduce` short-circuits on the first error.
        let t = Instant::now();
        use rayon::prelude::*;
        let best = resolutions_owned
            .par_iter()
            .map(|&r| -> scx_accel::Result<f64> {
                let pred_leiden = scx_accel::leiden(
                    &pred_knn.conn_indptr,
                    &pred_knn.conn_indices,
                    &pred_knn.conn_data,
                    n_output,
                    r,
                    /*seed=*/ 0,
                    /*max_iterations=*/ 2,
                    /*parallel=*/ false,
                )?;
                let pred_labels_u32: Vec<u32> =
                    pred_leiden.membership.iter().map(|&c| c as u32).collect();
                Ok(clustering_metric.score(&real_labels_u32, &pred_labels_u32))
            })
            .try_reduce(|| f64::NEG_INFINITY, |a, b| Ok(a.max(b)))?;
        let sweep_ms = t.elapsed().as_secs_f64() * 1000.0;

        let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            target: "pyscx::accel::eval_metrics::clustering_agreement",
            "n_obs={n_output} n_dims={n_genes} n_resolutions={n_resolutions} | \
             real_knn={real_knn_ms:.1}ms ({real_pct:.0}%) \
             real_leiden={real_leiden_ms:.1}ms ({real_leiden_pct:.0}%) \
             pred_knn={pred_knn_ms:.1}ms ({pred_pct:.0}%) \
             sweep={sweep_ms:.1}ms ({sweep_pct:.0}%) \
             total={total_ms:.1}ms",
            real_pct = 100.0 * real_knn_ms / total_ms,
            real_leiden_pct = 100.0 * real_leiden_ms / total_ms,
            pred_pct = 100.0 * pred_knn_ms / total_ms,
            sweep_pct = 100.0 * sweep_ms / total_ms,
        );
        Ok(best)
    });

    let best_score = best_score.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    route.commit();
    Ok(best_score)
}

// ──────────────────────────────────────────────────────────────────────────────
// Clustering scoring functions (AMI / NMI / ARI) on raw label vectors
// ──────────────────────────────────────────────────────────────────────────────

/// Convert a Python list/array of integer labels to Vec<u32>.
///
/// Extract cluster labels as contiguous `u32` codes.
///
/// Accepts integer arrays / pandas categorical codes / plain Python lists
/// directly, and — like sklearn's `adjusted_rand_score` & friends — also
/// accepts **string / categorical / object** labels, which are factorized to
/// integer codes via `pandas.factorize`. ARI/NMI/AMI depend only on the
/// partition each label vector induces (and each vector is factorized
/// independently), so this is exact. Negative codes are rejected (pandas uses
/// `-1` for NA).
fn extract_u32_labels(py: Python<'_>, labels: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    let np = crate::pyimport::import_module(py, "numpy")?;
    let arr = np.call_method1("asarray", (labels,))?;
    let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;

    // Integer / unsigned / bool labels: use the codes as-is (preserves exact
    // values and the NA-code check below). Everything else — object, string,
    // unicode, datetime, float — is factorized to contiguous integer codes,
    // mirroring `factorize_obs_column` in harmony.rs / lisi.rs.
    let codes = if matches!(kind.as_str(), "i" | "u" | "b") {
        arr.call_method1("astype", ("int64",))?
    } else {
        let pd = crate::pyimport::import_module(py, "pandas")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("sort", false)?;
        // factorize(arr, sort=False) -> (codes, uniques); codes are int64
        // with -1 for NaN/NA, rejected by the guard below. Pass the numpy
        // `arr` (not the raw `labels`, which may be a list — pandas warns on
        // non-ndarray/Series/Index inputs).
        let tup = pd.call_method("factorize", (&arr,), Some(&kwargs))?;
        tup.get_item(0)?.call_method1("astype", ("int64",))?
    };

    let vec: Vec<i64> = codes.call_method0("tolist")?.extract()?;
    if vec.iter().any(|&c| c < 0) {
        return Err(PyValueError::new_err(
            "label array contains negative values / NA codes; drop or fill missing labels before calling",
        ));
    }
    Ok(vec.iter().map(|&c| c as u32).collect())
}

/// Adjusted Mutual Information (sklearn arithmetic-mean convention).
///
/// Matches `sklearn.metrics.adjusted_mutual_info_score(labels_a, labels_b,
/// average_method="arithmetic")` to within 1e-10.
#[pyfunction]
pub fn adjusted_mutual_info(
    py: Python<'_>,
    labels_a: &Bound<'_, PyAny>,
    labels_b: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let a = extract_u32_labels(py, labels_a)?;
    let b = extract_u32_labels(py, labels_b)?;
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "label length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    Ok(py.detach(|| scx_accel::adjusted_mutual_info(&a, &b)))
}

/// Normalized Mutual Information (sklearn arithmetic-mean convention).
///
/// Matches `sklearn.metrics.normalized_mutual_info_score(labels_a, labels_b,
/// average_method="arithmetic")` to within 1e-10.
#[pyfunction]
pub fn normalized_mutual_info(
    py: Python<'_>,
    labels_a: &Bound<'_, PyAny>,
    labels_b: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let a = extract_u32_labels(py, labels_a)?;
    let b = extract_u32_labels(py, labels_b)?;
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "label length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    Ok(py.detach(|| scx_accel::normalized_mutual_info(&a, &b)))
}

/// Adjusted Rand Index.
///
/// By default (`rescaled=False`) this matches
/// `sklearn.metrics.adjusted_rand_score(labels_a, labels_b)` exactly: `1.0` for
/// identical clusterings, `~0` for random labelings, and negative for
/// worse-than-random — consistent with `normalized_mutual_info` /
/// `adjusted_mutual_info` in this module. Pass `rescaled=True` for cell-eval's
/// clustering-agreement convention `(ARI + 1) / 2` in `[0, 1]`.
///
/// Labels may be integer codes, strings, or pandas categoricals (factorized
/// internally; see `normalized_mutual_info`).
#[pyfunction]
#[pyo3(signature = (labels_a, labels_b, rescaled=false))]
pub fn adjusted_rand_index(
    py: Python<'_>,
    labels_a: &Bound<'_, PyAny>,
    labels_b: &Bound<'_, PyAny>,
    rescaled: bool,
) -> PyResult<f64> {
    let a = extract_u32_labels(py, labels_a)?;
    let b = extract_u32_labels(py, labels_b)?;
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "label length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    Ok(py.detach(|| {
        if rescaled {
            scx_accel::adjusted_rand_index_rescaled(&a, &b)
        } else {
            scx_accel::adjusted_rand_index(&a, &b)
        }
    }))
}
