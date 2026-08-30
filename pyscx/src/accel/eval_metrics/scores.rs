//! Effect scores: `perturbation_metrics`, `discrimination_score`,
//! `knockdown_efficiency`.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyDict;

use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::ScxLazyTransformedDataset;

// ──────────────────────────────────────────────────────────────────────────────
// device scaffolding (Phase 0 of CELL-EVAL-SCX-GPU-ACC.md)
// ──────────────────────────────────────────────────────────────────────────────
use super::*;

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
    crate::accel::prepare_target_no_var_guard(py, adata_pred, "perturbation_metrics")?;
    // Phase 2 GPU dispatch: the pseudobulk aggregation runs on the GPU (mirrors
    // pdex_nb_glm's skeleton); the five bulk metrics run on the host. `auto` on a
    // CPU host, `cpu`, and non-gpu builds fall through to the CPU path.
    let resolved = crate::accel::gpu::resolve_device(device)?;
    let info = crate::accel::route::simple_exec_info(
        device,
        cfg!(feature = "gpu"),
        scx_accel::route::AccelRoute::GpuCsr,
        scx_accel::route::AccelRoute::CpuCsr,
    );
    crate::accel::route::announce_route(py, "perturbation_metrics", device, &info);
    let gpu_id = if info.route.is_gpu() {
        resolved.gpu_id()
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

    crate::accel::route::write_accel_route(py, adata_pred, "perturbation_metrics", &info)?;

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
    crate::accel::prepare_target_no_var_guard(py, adata_pred, "discrimination_score")?;
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
    crate::accel::prepare_target_no_var_guard(py, adata, "knockdown_efficiency")?;
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
