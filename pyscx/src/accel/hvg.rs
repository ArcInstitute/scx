//! Highly variable gene selection — seurat_v3 and seurat flavors.

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::{ScxLazyTransformedDataset, Transform};

use super::filtering::update_layers_col_projection;

/// Streaming highly-variable gene selection without materialization.
///
/// Computes HVG statistics shard-by-shard via the `ShardSource` abstraction,
/// then selects the top `n_top_genes` by normalized variance (seurat_v3) or
/// normalized dispersion (seurat).
///
/// With `device="gpu"`, the per-column mean/variance and clipped-sum kernels
/// run on GPU (see `scx_gpu::gpu_streaming_mean_var` /
/// `gpu_streaming_clip_square_sum`). The loess fit, ranking, and result
/// writing stay on CPU. GPU dispatch is only used for **single-batch
/// seurat_v3** runs today; other configurations (`batch_key` set, or
/// `flavor="seurat"`) silently fall back to CPU even when `device="gpu"`.
///
/// Args:
///     adata: AnnData with X as ScxBackedSparseDataset or ScxLazyTransformedDataset
///     n_top_genes: Number of highly variable genes to select (default: 2000)
///     flavor: "seurat_v3" (raw counts) or "seurat" (log-normalized) (default: "seurat_v3")
///     batch_key: Column in adata.obs for batch-aware HVG (default: None)
///     span: Loess span for seurat_v3 (default: 0.3)
///     subset: If True, subset adata to HVG via column projection (default: False)
///     n_bins: Number of bins for seurat flavor (default: 20)
///     device: Device selection — "auto" (default), "cpu", or "gpu"
#[pyfunction]
#[pyo3(signature = (adata, n_top_genes=2000, flavor="seurat_v3", batch_key=None, span=0.3, subset=false, n_bins=20, device="auto"))]
#[allow(clippy::too_many_arguments)]
pub fn highly_variable_genes<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    n_top_genes: usize,
    flavor: &str,
    batch_key: Option<&str>,
    span: f64,
    subset: bool,
    n_bins: usize,
    device: &str,
) -> PyResult<()> {
    let resolved = super::gpu::resolve_device(device)?;
    // GPU path is only supported for single-batch seurat_v3; emit a warning
    // and fall back to CPU otherwise so the call succeeds with correct results.
    #[cfg(feature = "gpu")]
    let effective_gpu_id: Option<usize> = if let Some(gid) = resolved.gpu_id() {
        let seurat_v3 = matches!(flavor, "seurat_v3" | "seurat_v3_paper");
        if batch_key.is_some() || !seurat_v3 {
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                (
                    "highly_variable_genes(device=\"gpu\") is only implemented for \
                     single-batch seurat_v3 flavors; falling back to CPU.",
                    py.get_type::<pyo3::exceptions::PyUserWarning>(),
                ),
            )?;
            None
        } else {
            Some(gid)
        }
    } else {
        None
    };
    #[cfg(not(feature = "gpu"))]
    let effective_gpu_id: Option<usize> = {
        let _ = resolved;
        None
    };

    let x = adata.getattr("X")?;

    // ── Try SCX backed dataset ──────────────────────────────────────────
    if let Ok(backed) = x.downcast::<ScxBackedSparseDataset>() {
        let backed_ref = backed.borrow();
        let reader = Arc::clone(&backed_ref.backed);
        let n_vars = backed_ref.shape_val.1;
        let n_obs = backed_ref.shape_val.0;
        let kept = backed_ref.kept_to_global.clone();
        let col_proj = backed_ref.col_projection_arc();
        drop(backed_ref);

        return hvg_on_source(
            py,
            adata,
            &x,
            reader,
            vec![],
            kept,
            col_proj,
            n_obs,
            n_vars,
            n_top_genes,
            flavor,
            batch_key,
            span,
            subset,
            n_bins,
            effective_gpu_id,
        );
    }

    // ── Try SCX lazy-transformed dataset ────────────────────────────────
    if let Ok(lazy) = x.downcast::<ScxLazyTransformedDataset>() {
        let lazy_ref = lazy.borrow();
        let reader = Arc::clone(&lazy_ref.backed);
        let transforms = lazy_ref.transforms.clone();
        let n_vars = lazy_ref.shape_val.1;
        let n_obs = lazy_ref.shape_val.0;
        let kept = lazy_ref.kept_to_global.clone();
        let col_proj = lazy_ref.col_projection.clone();
        drop(lazy_ref);

        return hvg_on_source(
            py,
            adata,
            &x,
            reader,
            transforms,
            kept,
            col_proj,
            n_obs,
            n_vars,
            n_top_genes,
            flavor,
            batch_key,
            span,
            subset,
            n_bins,
            effective_gpu_id,
        );
    }

    // ── Fallback to scanpy ──────────────────────────────────────────────
    let sc = py.import("scanpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("n_top_genes", n_top_genes)?;
    kwargs.set_item("flavor", flavor)?;
    kwargs.set_item("span", span)?;
    kwargs.set_item("subset", subset)?;
    kwargs.set_item("n_bins", n_bins)?;
    if let Some(bk) = batch_key {
        kwargs.set_item("batch_key", bk)?;
    }
    sc.getattr("pp")?
        .call_method("highly_variable_genes", (adata,), Some(&kwargs))?;
    Ok(())
}

/// Dispatch to seurat_v3 or seurat HVG implementation.
#[allow(clippy::too_many_arguments)]
fn hvg_on_source<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    x_obj: &Bound<'py, PyAny>,
    reader: Arc<scx_format::BackedCsrReader>,
    transforms: Vec<Transform>,
    kept_to_global: Option<Arc<Vec<u64>>>,
    col_projection: Option<Arc<Vec<u32>>>,
    n_obs: usize,
    n_vars: usize,
    n_top_genes: usize,
    flavor: &str,
    batch_key: Option<&str>,
    span: f64,
    subset: bool,
    n_bins: usize,
    device_id: Option<usize>,
) -> PyResult<()> {
    match flavor {
        "seurat_v3" | "seurat_v3_paper" => hvg_seurat_v3(
            py,
            adata,
            x_obj,
            reader,
            transforms,
            kept_to_global,
            col_projection,
            n_obs,
            n_vars,
            n_top_genes,
            batch_key,
            span,
            subset,
            flavor,
            device_id,
        ),
        "seurat" => hvg_seurat(
            py,
            adata,
            x_obj,
            reader,
            transforms,
            kept_to_global,
            col_projection,
            n_obs,
            n_vars,
            n_top_genes,
            batch_key,
            subset,
            n_bins,
        ),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported HVG flavor '{flavor}'. Use 'seurat_v3' or 'seurat'."
        ))),
    }
}

/// Build a LazyShardSource, optionally filtered to a batch of cells.
fn build_shard_source(
    reader: &Arc<scx_format::BackedCsrReader>,
    transforms: &[Transform],
    kept_to_global: &Option<Arc<Vec<u64>>>,
    col_projection: &Option<Arc<Vec<u32>>>,
    n_vars: usize,
    batch_indices: Option<&[usize]>,
) -> crate::lazy_transform::LazyShardSource {
    use crate::lazy_transform::LazyShardSource;

    match batch_indices {
        Some(indices) => {
            // Compose batch indices with existing kept_to_global
            let global_rows: Vec<u64> = match kept_to_global {
                Some(existing) => indices.iter().map(|&i| existing[i]).collect(),
                None => indices.iter().map(|&i| i as u64).collect(),
            };
            LazyShardSource::with_kept_rows(
                Arc::clone(reader),
                transforms.to_vec(),
                global_rows,
                col_projection.clone(),
                n_vars,
            )
        }
        None => {
            // Full dataset (or existing kept_to_global).
            // Pass None when no filtering needed — avoids allocating a full
            // identity range and skips the deletion-vector path in read_shard.
            let n_obs = match kept_to_global {
                Some(ref k) => k.len(),
                None => reader.shape().0,
            };
            LazyShardSource::new(
                Arc::clone(reader),
                transforms.to_vec(),
                kept_to_global.as_ref().map(Arc::clone),
                col_projection.clone(),
                n_obs,
                n_vars,
            )
        }
    }
}

/// seurat_v3 flavor: raw count data, loess fit, clipped variance.
#[allow(clippy::too_many_arguments)]
fn hvg_seurat_v3<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    x_obj: &Bound<'py, PyAny>,
    reader: Arc<scx_format::BackedCsrReader>,
    transforms: Vec<Transform>,
    kept_to_global: Option<Arc<Vec<u64>>>,
    col_projection: Option<Arc<Vec<u32>>>,
    n_obs: usize,
    n_vars: usize,
    n_top_genes: usize,
    batch_key: Option<&str>,
    span: f64,
    subset: bool,
    flavor: &str,
    _device_id: Option<usize>,
) -> PyResult<()> {
    // ── 1. Determine batches ────────────────────────────────────────────
    let batches: Vec<Vec<usize>> = match batch_key {
        Some(bk) => {
            let obs = adata.getattr("obs")?;
            let batch_col = obs.get_item(bk)?;
            // Handle both categorical and non-categorical columns:
            // wrap in pd.Categorical() which is a no-op for already-categorical data.
            let pd = py.import("pandas")?;
            let np = py.import("numpy")?;
            let cat = pd.call_method1("Categorical", (&batch_col,))?;
            let codes = cat.getattr("codes")?;
            let cat_codes: Vec<i64> = np
                .call_method1("asarray", (&codes,))?
                .call_method1("astype", ("int64",))?
                .extract()?;
            let n_batches = *cat_codes.iter().max().unwrap_or(&0) as usize + 1;
            let mut groups = vec![vec![]; n_batches];
            for (i, &code) in cat_codes.iter().enumerate() {
                if code >= 0 {
                    groups[code as usize].push(i);
                }
            }
            groups.into_iter().filter(|g| !g.is_empty()).collect()
        }
        None => vec![(0..n_obs).collect()],
    };

    let n_batches_actual = batches.len();

    // Build cell-to-batch mapping for batched streaming
    let mut cell_batch = vec![-1i32; n_obs];
    for (batch_id, batch_cells) in batches.iter().enumerate() {
        for &cell_idx in batch_cells {
            cell_batch[cell_idx] = batch_id as i32;
        }
    }

    // ── 2. Batched streaming mean/var (single pass for ALL batches + global) ──
    let source = build_shard_source(
        &reader,
        &transforms,
        &kept_to_global,
        &col_projection,
        n_vars,
        None,
    );
    // Single-batch + GPU: route to `streaming_mean_var_with_device` for the
    // GPU atomicAdd accumulation path. Multi-batch stays on CPU because the
    // GPU kernel doesn't carry per-cell batch membership today.
    #[cfg(feature = "gpu")]
    let batched_stats = if let (Some(dev_id), 1) = (_device_id, n_batches_actual) {
        let single = scx_accel::streaming_mean_var_with_device(&source, "gpu", dev_id)
            .map_err(|e| PyRuntimeError::new_err(format!("gpu streaming_mean_var: {e}")))?;
        scx_accel::BatchedHvgStats {
            per_batch: vec![single.clone()],
            global: single,
            batch_counts: vec![n_obs],
        }
    } else {
        scx_accel::streaming_mean_var_batched(&source, &cell_batch, n_batches_actual)
            .map_err(|e| PyRuntimeError::new_err(format!("streaming_mean_var_batched: {e}")))?
    };
    #[cfg(not(feature = "gpu"))]
    let batched_stats =
        scx_accel::streaming_mean_var_batched(&source, &cell_batch, n_batches_actual)
            .map_err(|e| PyRuntimeError::new_err(format!("streaming_mean_var_batched: {e}")))?;

    let global_stats = batched_stats.global.clone();

    // ── 3. Per-batch: loess fit → clip_val (in-memory, no I/O) ───────────
    let mut all_clip_vals: Vec<Vec<f64>> = Vec::new();
    let mut batch_estimat_vars: Vec<Vec<f64>> = Vec::new();

    for (b, batch_cells) in batches.iter().enumerate() {
        let batch_n = batch_cells.len();
        if batch_n < 2 {
            all_clip_vals.push(vec![0.0; n_vars]);
            batch_estimat_vars.push(vec![0.0; n_vars]);
            continue;
        }

        let batch_stats = &batched_stats.per_batch[b];

        // Loess fit via Python (on non-constant genes)
        let mut estimat_var = vec![0.0f64; n_vars];
        let not_const: Vec<bool> = batch_stats.variances.iter().map(|&v| v > 0.0).collect();
        let x_vals: Vec<f64> = batch_stats
            .means
            .iter()
            .zip(not_const.iter())
            .filter(|(_, &nc)| nc)
            .map(|(&m, _)| m.max(1e-300).log10())
            .collect();
        let y_vals: Vec<f64> = batch_stats
            .variances
            .iter()
            .zip(not_const.iter())
            .filter(|(_, &nc)| nc)
            .map(|(&v, _)| v.max(1e-300).log10())
            .collect();

        if x_vals.len() >= 3 {
            let x_arr = numpy::PyArray::from_vec(py, x_vals);
            let y_arr = numpy::PyArray::from_vec(py, y_vals);

            let loess_mod = py.import("skmisc.loess")?;
            let loess_cls = loess_mod.getattr("loess")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("span", span)?;
            kwargs.set_item("degree", 2)?;
            let model = loess_cls.call((x_arr, y_arr), Some(&kwargs))?;
            model.call_method0("fit")?;
            let fitted: Vec<f64> = model
                .getattr("outputs")?
                .getattr("fitted_values")?
                .extract()?;

            let mut fi = 0;
            for (j, &nc) in not_const.iter().enumerate() {
                if nc {
                    estimat_var[j] = fitted[fi];
                    fi += 1;
                }
            }
        }

        // reg_std and clip_val
        let mut clip_val = vec![0.0f64; n_vars];
        let batch_n_f = batch_n as f64;
        let sqrt_n = batch_n_f.sqrt();
        for j in 0..n_vars {
            let reg_std = 10.0f64.powf(estimat_var[j]).sqrt();
            clip_val[j] = reg_std * sqrt_n + batch_stats.means[j];
        }

        all_clip_vals.push(clip_val);
        batch_estimat_vars.push(estimat_var);
    }

    // ── 4. Batched streaming clipped sums (single pass for ALL batches) ──
    #[cfg(feature = "gpu")]
    let all_clipped = if let (Some(dev_id), 1) = (_device_id, n_batches_actual) {
        let single = scx_accel::streaming_clip_square_sum_with_device(
            &source,
            &all_clip_vals[0],
            "gpu",
            dev_id,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("gpu streaming_clip_square_sum: {e}")))?;
        vec![single]
    } else {
        scx_accel::streaming_clip_square_sum_batched(
            &source,
            &cell_batch,
            n_batches_actual,
            &all_clip_vals,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("streaming_clip_square_sum_batched: {e}")))?
    };
    #[cfg(not(feature = "gpu"))]
    let all_clipped = scx_accel::streaming_clip_square_sum_batched(
        &source,
        &cell_batch,
        n_batches_actual,
        &all_clip_vals,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("streaming_clip_square_sum_batched: {e}")))?;

    // ── 5. Compute normalized variance per batch (in-memory) ─────────────
    let mut all_norm_vars: Vec<Vec<f64>> = Vec::new();
    for (b, batch_cells) in batches.iter().enumerate() {
        let batch_n = batch_cells.len();
        if batch_n < 2 {
            all_norm_vars.push(vec![0.0; n_vars]);
            continue;
        }

        let batch_stats = &batched_stats.per_batch[b];
        let (ref bcs, ref sbcs) = all_clipped[b];
        let estimat_var = &batch_estimat_vars[b];
        let batch_n_f = batch_n as f64;
        let denom_n = (batch_n_f - 1.0).max(1.0);

        let mut norm_gene_var = vec![0.0f64; n_vars];
        for j in 0..n_vars {
            let reg_std_sq = 10.0f64.powf(estimat_var[j]);
            if reg_std_sq > 0.0 {
                norm_gene_var[j] = (1.0 / (denom_n * reg_std_sq))
                    * (batch_n_f * batch_stats.means[j] * batch_stats.means[j] + sbcs[j]
                        - 2.0 * bcs[j] * batch_stats.means[j]);
            }
        }
        all_norm_vars.push(norm_gene_var);
    }

    // ── 4. Rank genes and select top N ──────────────────────────────────
    let n_batches = all_norm_vars.len();

    // Mean normalized variance across batches
    let mut mean_norm_var = vec![0.0f64; n_vars];
    for nv in &all_norm_vars {
        for (j, &v) in nv.iter().enumerate() {
            mean_norm_var[j] += v;
        }
    }
    for v in &mut mean_norm_var {
        *v /= n_batches as f64;
    }

    // For multi-batch: rank within each batch, then combine ranks
    let (hvg_mask, ranks) = if n_batches > 1 {
        // Per-batch ranks: for each batch, rank genes by normalized variance (descending)
        let mut batch_ranks: Vec<Vec<usize>> = Vec::new();
        for nv in &all_norm_vars {
            let mut indices: Vec<usize> = (0..n_vars).collect();
            indices.sort_by(|&a, &b| {
                nv[b]
                    .partial_cmp(&nv[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut rank = vec![0usize; n_vars];
            for (r, &idx) in indices.iter().enumerate() {
                rank[idx] = r;
            }
            batch_ranks.push(rank);
        }

        // Count in how many batches each gene is in top n_top_genes
        let mut nbatches_hv = vec![0usize; n_vars];
        let mut median_ranks = vec![f64::NAN; n_vars];
        for j in 0..n_vars {
            let ranks_j: Vec<usize> = batch_ranks.iter().map(|br| br[j]).collect();
            nbatches_hv[j] = ranks_j.iter().filter(|&&r| r < n_top_genes).count();
            // Median of ranks where gene is in top n_top_genes
            let mut valid: Vec<f64> = ranks_j
                .iter()
                .filter(|&&r| r < n_top_genes)
                .map(|&r| r as f64)
                .collect();
            if !valid.is_empty() {
                valid.sort_by(|a, b| a.partial_cmp(b).unwrap());
                median_ranks[j] = valid[valid.len() / 2];
            }
        }

        // Sort genes: by nbatches (desc), then median_rank (asc)
        let mut gene_order: Vec<usize> = (0..n_vars).collect();
        if flavor == "seurat_v3_paper" {
            gene_order.sort_by(|&a, &b| {
                nbatches_hv[b].cmp(&nbatches_hv[a]).then(
                    median_ranks[a]
                        .partial_cmp(&median_ranks[b])
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
            });
        } else {
            gene_order.sort_by(|&a, &b| {
                median_ranks[a]
                    .partial_cmp(&median_ranks[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(nbatches_hv[b].cmp(&nbatches_hv[a]))
            });
        }

        let mut mask = vec![false; n_vars];
        let mut rank_out = vec![f64::NAN; n_vars];
        for (r, &g) in gene_order.iter().enumerate().take(n_top_genes.min(n_vars)) {
            mask[g] = true;
            rank_out[g] = r as f64;
        }
        (mask, rank_out)
    } else {
        // Single batch: simple rank by normalized variance descending
        let mut indices: Vec<usize> = (0..n_vars).collect();
        indices.sort_by(|&a, &b| {
            mean_norm_var[b]
                .partial_cmp(&mean_norm_var[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut mask = vec![false; n_vars];
        let mut rank_out = vec![f64::NAN; n_vars];
        for (r, &g) in indices.iter().enumerate().take(n_top_genes.min(n_vars)) {
            mask[g] = true;
            rank_out[g] = r as f64;
        }
        (mask, rank_out)
    };

    // ── 5. Write results to adata.var ───────────────────────────────────
    let var = adata.getattr("var")?;
    var.set_item(
        "highly_variable",
        numpy::PyArray::from_vec(py, hvg_mask.clone()),
    )?;
    var.set_item("means", numpy::PyArray::from_vec(py, global_stats.means))?;
    var.set_item(
        "variances",
        numpy::PyArray::from_vec(py, global_stats.variances),
    )?;
    var.set_item(
        "variances_norm",
        numpy::PyArray::from_vec(py, mean_norm_var),
    )?;
    var.set_item("highly_variable_rank", numpy::PyArray::from_vec(py, ranks))?;

    // ── 6. Subset if requested ──────────────────────────────────────────
    if subset {
        apply_hvg_subset(py, adata, x_obj, &hvg_mask)?;
    }

    Ok(())
}

/// seurat flavor: log-normalized data, binned dispersion normalization.
#[allow(clippy::too_many_arguments)]
fn hvg_seurat<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    x_obj: &Bound<'py, PyAny>,
    reader: Arc<scx_format::BackedCsrReader>,
    transforms: Vec<Transform>,
    kept_to_global: Option<Arc<Vec<u64>>>,
    col_projection: Option<Arc<Vec<u32>>>,
    _n_obs: usize,
    n_vars: usize,
    n_top_genes: usize,
    batch_key: Option<&str>,
    subset: bool,
    n_bins: usize,
) -> PyResult<()> {
    // For batched seurat, fall back to scanpy (complex aggregation logic)
    if batch_key.is_some() {
        let sc = py.import("scanpy")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("n_top_genes", n_top_genes)?;
        kwargs.set_item("flavor", "seurat")?;
        kwargs.set_item("subset", subset)?;
        kwargs.set_item("n_bins", n_bins)?;
        kwargs.set_item("batch_key", batch_key)?;
        sc.getattr("pp")?
            .call_method("highly_variable_genes", (adata,), Some(&kwargs))?;
        return Ok(());
    }

    // ── 1. Streaming mean/var ───────────────────────────────────────────
    let source = build_shard_source(
        &reader,
        &transforms,
        &kept_to_global,
        &col_projection,
        n_vars,
        None,
    );
    let stats = scx_accel::streaming_mean_var(&source)
        .map_err(|e| PyRuntimeError::new_err(format!("streaming_mean_var: {e}")))?;

    // ── 2. Compute dispersion (matching scanpy's seurat flavor) ────────
    let mut dispersions = vec![0.0f64; n_vars];
    let mut log_dispersions = vec![f64::NAN; n_vars];
    let mut log_means = vec![0.0f64; n_vars];
    let mut means_for_disp = stats.means.clone();

    for j in 0..n_vars {
        // scanpy: mean[mean == 0] = 1e-12 (before dispersion computation)
        if means_for_disp[j] == 0.0 {
            means_for_disp[j] = 1e-12;
        }
        dispersions[j] = stats.variances[j] / means_for_disp[j];
        // scanpy: dispersion[dispersion == 0] = NaN, then log(dispersion)
        if dispersions[j] > 0.0 {
            log_dispersions[j] = dispersions[j].ln();
        } else {
            dispersions[j] = f64::NAN;
        }
        // scanpy: mean = log1p(mean) — overwrite mean with log1p for binning
        log_means[j] = (means_for_disp[j] + 1.0).ln();
    }

    // ── 3. Bin by mean, z-score dispersion within bins (via Python) ────
    let log_means_np = numpy::PyArray::from_vec(py, log_means);
    let log_disp_np = numpy::PyArray::from_vec(py, log_dispersions);

    let helpers = py.import("pyscx._hvg_helpers")?;
    let dispersions_norm: Vec<f64> = helpers
        .call_method1(
            "binned_dispersion_norm",
            (log_means_np, log_disp_np, n_bins),
        )?
        .extract()?;

    // ── 4. Select top genes by normalized dispersion ────────────────────
    let mut indices: Vec<usize> = (0..n_vars).collect();
    indices.sort_by(|&a, &b| {
        dispersions_norm[b]
            .partial_cmp(&dispersions_norm[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut mask = vec![false; n_vars];
    for &g in indices.iter().take(n_top_genes.min(n_vars)) {
        mask[g] = true;
    }

    // ── 5. Write results to adata.var ───────────────────────────────────
    let var = adata.getattr("var")?;
    var.set_item(
        "highly_variable",
        numpy::PyArray::from_vec(py, mask.clone()),
    )?;
    var.set_item("means", numpy::PyArray::from_vec(py, stats.means))?;
    var.set_item("dispersions", numpy::PyArray::from_vec(py, dispersions))?;
    var.set_item(
        "dispersions_norm",
        numpy::PyArray::from_vec(
            py,
            dispersions_norm
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<f32>>(),
        ),
    )?;

    // ── 6. Subset if requested ──────────────────────────────────────────
    if subset {
        apply_hvg_subset(py, adata, x_obj, &mask)?;
    }

    Ok(())
}

/// Apply HVG subset: set column projection on X (and layers), slice var.
///
/// Order: update X col_projection + layers FIRST so shapes match,
/// then set `_var` (bypassing AnnData shape validation).
fn apply_hvg_subset(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    x_obj: &Bound<'_, PyAny>,
    hvg_mask: &[bool],
) -> PyResult<()> {
    let mask_arr = numpy::PyArray::from_vec(py, hvg_mask.to_vec());

    if let Ok(backed) = x_obj.downcast::<ScxBackedSparseDataset>() {
        let new_col_indices: Vec<u32> = match backed.borrow().col_projection() {
            Some(existing) => hvg_mask
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| existing[i])
                .collect(),
            None => hvg_mask
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| i as u32)
                .collect(),
        };

        // Update X and layers FIRST so shapes are consistent
        backed
            .borrow_mut()
            .set_col_projection(new_col_indices.clone());
        update_layers_col_projection(adata, &new_col_indices)?;

        // Slice var via _var: AnnData's public var setter validates
        // len(value) == self.n_vars, where n_vars is derived from the current
        // _var DataFrame. Since we're changing the column count, the public
        // setter would reject the new (shorter) DataFrame. Setting _var
        // directly is the same approach used by anndata's own _inplace_subset_var.
        let var = adata.getattr("var")?;
        let filtered_var = var.getattr("loc")?.get_item(&mask_arr)?;
        adata.setattr("_var", filtered_var)?;
    } else if let Ok(lazy) = x_obj.downcast::<ScxLazyTransformedDataset>() {
        let new_col_indices: Vec<u32> = match lazy.borrow().col_projection() {
            Some(existing) => hvg_mask
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| existing[i])
                .collect(),
            None => hvg_mask
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| i as u32)
                .collect(),
        };

        lazy.borrow_mut()
            .set_col_projection(new_col_indices.clone());
        update_layers_col_projection(adata, &new_col_indices)?;

        // See comment above in backed branch for why _var is used.
        let var = adata.getattr("var")?;
        let filtered_var = var.getattr("loc")?.get_item(&mask_arr)?;
        adata.setattr("_var", filtered_var)?;
    }

    Ok(())
}
