//! The column-major (CSC sidecar) seurat_v3 path and its GPU dispatchers.

use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyDict;

use super::*;
use crate::optional_deps::{import_optional, EXTRA_HVG};

/// Loess fit on (log10 mean, log10 var) for non-constant genes — the
/// seurat_v3 dispersion regression. Returns per-gene `estimat_var`
/// (log10 fitted variance); constant genes and the too-few-points case
/// stay 0.0. Shared by the backed and lazy CSC pipelines.
fn csc_loess_estimat_var(
    py: Python<'_>,
    means: &[f64],
    variances: &[f64],
    span: f64,
) -> PyResult<Vec<f64>> {
    let n_vars = means.len();
    let mut estimat_var = vec![0.0f64; n_vars];
    let not_const: Vec<bool> = variances.iter().map(|&v| v > 0.0).collect();
    let x_vals: Vec<f64> = means
        .iter()
        .zip(not_const.iter())
        .filter(|(_, &nc)| nc)
        .map(|(&m, _)| m.max(1e-300).log10())
        .collect();
    let y_vals: Vec<f64> = variances
        .iter()
        .zip(not_const.iter())
        .filter(|(_, &nc)| nc)
        .map(|(&v, _)| v.max(1e-300).log10())
        .collect();

    if x_vals.len() >= 3 {
        let x_arr = numpy::PyArray::from_vec(py, x_vals);
        let y_arr = numpy::PyArray::from_vec(py, y_vals);
        let loess_mod = import_optional(
            py,
            "skmisc.loess",
            EXTRA_HVG,
            "pyscx.accel.highly_variable_genes(flavor=\"seurat_v3\")",
            "scikit-misc",
        )?;
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
    Ok(estimat_var)
}

/// clip_val per gene = `reg_std * sqrt(n) + mean` (seurat_v3).
fn csc_clip_val(means: &[f64], estimat_var: &[f64], n_obs: usize) -> Vec<f64> {
    let sqrt_n = (n_obs as f64).sqrt();
    means
        .iter()
        .zip(estimat_var.iter())
        .map(|(&mean, &ev)| 10.0f64.powf(ev).sqrt() * sqrt_n + mean)
        .collect()
}

/// Steps 5–8 of single-batch seurat_v3: normalized variance, rank, write
/// `adata.var`, optional subset. Shared by the backed and lazy CSC paths.
#[allow(clippy::too_many_arguments)]
fn csc_finish_seurat_v3(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    n_obs: usize,
    n_vars: usize,
    means: Vec<f64>,
    variances: Vec<f64>,
    estimat_var: Vec<f64>,
    bcs: Vec<f64>,
    sbcs: Vec<f64>,
    n_top_genes: usize,
    subset: bool,
) -> PyResult<()> {
    let n_f = n_obs as f64;
    let denom_n = (n_f - 1.0).max(1.0);
    let mut norm_gene_var = vec![0.0f64; n_vars];
    for j in 0..n_vars {
        let reg_std_sq = 10.0f64.powf(estimat_var[j]);
        if reg_std_sq > 0.0 {
            norm_gene_var[j] = (1.0 / (denom_n * reg_std_sq))
                * (n_f * means[j] * means[j] + sbcs[j] - 2.0 * bcs[j] * means[j]);
        }
    }

    // Single-batch ranking: sort by normalized variance desc.
    let mut indices: Vec<usize> = (0..n_vars).collect();
    indices.sort_by(|&a, &b| {
        norm_gene_var[b]
            .partial_cmp(&norm_gene_var[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut hvg_mask = vec![false; n_vars];
    let mut ranks = vec![f64::NAN; n_vars];
    for (r, &g) in indices.iter().enumerate().take(n_top_genes.min(n_vars)) {
        hvg_mask[g] = true;
        ranks[g] = r as f64;
    }

    // Write to adata.var (matches CSR seurat_v3 schema).
    let var = adata.getattr("var")?;
    var.set_item(
        "highly_variable",
        numpy::PyArray::from_vec(py, hvg_mask.clone()),
    )?;
    var.set_item("means", numpy::PyArray::from_vec(py, means))?;
    var.set_item("variances", numpy::PyArray::from_vec(py, variances))?;
    var.set_item(
        "variances_norm",
        numpy::PyArray::from_vec(py, norm_gene_var),
    )?;
    var.set_item("highly_variable_rank", numpy::PyArray::from_vec(py, ranks))?;

    if subset {
        apply_hvg_subset(py, adata, &hvg_mask)?;
    }
    Ok(())
}

/// Device-dispatched per-column mean/var on a `Sync` CSC source: GPU CSC
/// reduce when `device_id` is `Some` (no `atomicAdd`), else CPU CSC.
#[cfg(feature = "gpu")]
fn csc_mean_var_dispatch<S: scx_format_io::ColumnShardSource + Sync>(
    py: Python<'_>,
    source: &S,
    device_id: Option<usize>,
) -> PyResult<scx_accel::HvgStats> {
    match device_id {
        Some(id) => py
            .detach(|| scx_accel::streaming_mean_var_csc_with_device(source, "gpu", id))
            .map_err(|e| PyRuntimeError::new_err(format!("gpu streaming_mean_var_csc: {e}"))),
        None => scx_accel::streaming_mean_var_csc(source)
            .map_err(|e| PyRuntimeError::new_err(format!("streaming_mean_var_csc: {e}"))),
    }
}

/// Device-dispatched per-column clipped sums on a `Sync` CSC source.
#[cfg(feature = "gpu")]
fn csc_clip_dispatch<S: scx_format_io::ColumnShardSource + Sync>(
    py: Python<'_>,
    source: &S,
    clip_val: &[f64],
    device_id: Option<usize>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    match device_id {
        Some(id) => {
            py.detach(|| {
                scx_accel::streaming_clip_square_sum_csc_with_device(source, clip_val, "gpu", id)
            })
            .map_err(|e| PyRuntimeError::new_err(format!("gpu streaming_clip_square_sum_csc: {e}")))
        }
        None => scx_accel::streaming_clip_square_sum_csc(source, clip_val)
            .map_err(|e| PyRuntimeError::new_err(format!("streaming_clip_square_sum_csc: {e}"))),
    }
}

/// Single-batch seurat_v3 HVG via the CSC sidecar.
///
/// Same numerics as `hvg_seurat_v3` for the single-batch case, but pulls
/// per-column mean/var and clipped sums from the gene-major
/// `ColumnShardSource`. With `device_id = Some(gid)` the two reduction
/// passes run the GPU CSC reduce kernels (one block per gene, no
/// `atomicAdd`) — route `gpu_csc_v3`; `None` runs the CPU CSC kernels.
/// Both handle kinds reach either device: the column source is owned and
/// `Sync`, so the restriction that used to pin the lazy branch to CPU (a
/// borrowed `&dyn ColumnShardSource` is not `Sync`) is gone.
pub(crate) fn hvg_seurat_v3_csc(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    n_top_genes: usize,
    span: f64,
    subset: bool,
    flavor: &str,
    device_id: Option<usize>,
) -> PyResult<()> {
    use scx_format_io::ColumnShardSource;
    let _ = flavor; // single-batch: seurat_v3 / seurat_v3_paper share ordering.

    let x = adata.getattr("X")?;

    // ── One column source for either handle kind. `as_column_source()` hands
    //    back an owned `LazyShardSource` (Send + Sync), so the GPU CSC reduce
    //    path — which decodes on a worker thread and wants
    //    `S: ColumnShardSource + Sync` — is reachable from both. The lazy arm
    //    used to be pinned to CPU because it held a borrowed `&dyn
    //    ColumnShardSource`, which is not `Sync`; that was a property of the
    //    borrow, not of the data. ──
    if !crate::accel::is_scx_matrix_handle(&x) {
        return Err(PyRuntimeError::new_err(
            "prefer_format='csc' requires adata.X to be ScxBackedSparseDataset \
             or ScxLazyTransformedDataset (got a regular scipy/dense matrix)",
        ));
    }
    let source = crate::accel::csc_source_for(&x).ok_or_else(crate::accel::csc_unavailable)?;

    let n_obs = ColumnShardSource::n_obs(&source);
    let n_vars = ColumnShardSource::n_vars(&source);

    #[cfg(feature = "gpu")]
    let stats = csc_mean_var_dispatch(py, &source, device_id)?;
    #[cfg(not(feature = "gpu"))]
    let stats = {
        let _ = device_id;
        scx_accel::streaming_mean_var_csc(&source)
            .map_err(|e| PyRuntimeError::new_err(format!("streaming_mean_var_csc: {e}")))?
    };

    let estimat_var = csc_loess_estimat_var(py, &stats.means, &stats.variances, span)?;
    let clip_val = csc_clip_val(&stats.means, &estimat_var, n_obs);

    #[cfg(feature = "gpu")]
    let (bcs, sbcs) = csc_clip_dispatch(py, &source, &clip_val, device_id)?;
    #[cfg(not(feature = "gpu"))]
    let (bcs, sbcs) = scx_accel::streaming_clip_square_sum_csc(&source, &clip_val)
        .map_err(|e| PyRuntimeError::new_err(format!("streaming_clip_square_sum_csc: {e}")))?;

    csc_finish_seurat_v3(
        py,
        adata,
        n_obs,
        n_vars,
        stats.means,
        stats.variances,
        estimat_var,
        bcs,
        sbcs,
        n_top_genes,
        subset,
    )
}
