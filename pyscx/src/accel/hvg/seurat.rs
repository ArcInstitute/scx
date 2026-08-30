//! The streaming seurat (dispersion) kernel.

use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyDict;

use super::*;
use crate::optional_deps::{import_optional_with_hint, EXTRA_SCANPY};

/// seurat flavor: log-normalized data, binned dispersion normalization.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hvg_seurat<'py, S: scx_format_io::ShardSource + Sync>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    source: &S,
    _n_obs: usize,
    n_vars: usize,
    n_top_genes: usize,
    batch_key: Option<&str>,
    subset: bool,
    n_bins: usize,
) -> PyResult<()> {
    // For batched seurat, fall back to scanpy (complex aggregation logic)
    if batch_key.is_some() {
        let sc = import_optional_with_hint(
            py,
            "scanpy",
            EXTRA_SCANPY,
            "pyscx.accel.highly_variable_genes(flavor=\"seurat\", batch_key=...)",
            "scanpy",
            // The single-batch seurat kernel is scx-native; only the cross-batch
            // aggregation is delegated.
            Some(
                "Only the batched `seurat` flavor is delegated — \
                 flavor=\"seurat\" without `batch_key`, and flavor=\"seurat_v3\" \
                 with or without it, run the scx-native kernel.",
            ),
        )?;
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

    // Native-path validation, deliberately AFTER the batch_key delegation so
    // the scanpy-delegated branch keeps scanpy's own n_bins behavior (and the
    // in-memory rapids branch, which never reaches this function, keeps
    // rapids'). Two guards for the two failure modes the Rust kernel changed:
    // n_bins == 0 was pandas' opaque "Cannot cut empty array"-adjacent
    // ValueError (the kernel would silently answer all-NaN → zero genes
    // selected); an absurd n_bins was pandas' survivable MemoryError, where
    // Rust's infallible per-bin allocations would ABORT the process. 2^20 is
    // far beyond any meaningful binning of a gene axis (scanpy's default is
    // 20; a bin count above n_vars only adds empty bins). (Round-1 finding:
    // codex; scoped to the native path in round 2, also codex.)
    if n_bins == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_bins must be at least 1 (scanpy's default is 20)",
        ));
    }
    const N_BINS_MAX: usize = 1 << 20;
    if n_bins > N_BINS_MAX {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n_bins = {n_bins} is not a meaningful binning (max {N_BINS_MAX}); \
             scanpy's default is 20"
        )));
    }

    // ── 1. Streaming mean/var in COUNT space ───────────────────────────
    // scanpy's seurat flavor un-`log1p`s the matrix before computing moments:
    // `x *= ln(base)` (identity for natural-log / no recorded base), then
    // `expm1`. We stream the same count-space moments directly from the
    // (log-transformed) shards. `scale = ln(base)`, or 1.0 when no base is
    // recorded (matches scanpy's `uns.get("log1p", {}).get("base")`).
    let scale = log1p_base_scale(py, adata)?;
    let stats = py
        .detach(|| scx_accel::streaming_mean_var_expm1(source, scale))
        .map_err(|e| PyRuntimeError::new_err(format!("streaming_mean_var_expm1: {e}")))?;

    // ── 2. Compute dispersion (matching scanpy's seurat flavor) ────────
    // scanpy publishes the LOG dispersion and the LOG1P count-space mean:
    //   mean[mean==0] = 1e-12; dispersion = var/mean;
    //   dispersion[dispersion==0] = NaN; dispersion = log(dispersion);
    //   mean = log1p(mean).
    let mut log_dispersions = vec![f64::NAN; n_vars];
    let mut log_means = vec![0.0f64; n_vars];
    let mut means_for_disp = stats.means.clone();

    for j in 0..n_vars {
        // scanpy: mean[mean == 0] = 1e-12 (before dispersion computation)
        if means_for_disp[j] == 0.0 {
            means_for_disp[j] = 1e-12;
        }
        let disp = stats.variances[j] / means_for_disp[j];
        // scanpy: dispersion[dispersion == 0] = NaN, then log(dispersion)
        if disp > 0.0 {
            log_dispersions[j] = disp.ln();
        }
        // scanpy: mean = log1p(mean) — count-space mean, logged for binning
        // and for the published `means` column.
        log_means[j] = means_for_disp[j].ln_1p();
    }

    // `expm1` overflows to Inf around x ≈ 709, which is what running
    // flavor="seurat" on raw counts (a MALAT1-scale UMI value) produces. The
    // pandas reference RAISED on an Inf mean ("cannot specify integer `bins`
    // when input data contains infinity"); the Rust kernel instead answers
    // all-NaN, which the -inf selection floor below would turn into ZERO
    // genes selected — and subset=True would then drop the whole var axis,
    // silently. Keep the failure loud at the boundary, like n_bins == 0
    // above. (Round-1 finding: Cursor Agent.)
    if log_means.iter().any(|v| !v.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "count-space means overflowed to infinity: flavor=\"seurat\" \
             un-log1ps X before computing moments, so it expects \
             log-normalized input (sc.pp.log1p / pyscx.accel.log1p), not raw \
             counts",
        ));
    }

    // ── 3. Bin by mean, z-score dispersion within bins ──────────────────
    // Rust-native since ORG-10.16-5 (previously a Python callback into the
    // shipped `pyscx._hvg_helpers`, the one file a Rust accelerator's
    // correctness depended on): the scanpy `pd.cut` + groupby semantics —
    // right-closed equal-width bins, NaN-skipping ddof-1 stats, the
    // singleton-bin `exactly 1.0` rule, NaN preserved — live in
    // `scx_accel::binned_dispersion_norm`, golden-pinned against the pandas
    // reference and held to scanpy by `test_column_parity_with_scanpy`.
    let dispersions_norm: Vec<f64> =
        py.detach(|| scx_accel::binned_dispersion_norm(&log_means, &log_dispersions, n_bins));

    // ── 4. Select top genes by normalized dispersion ────────────────────
    // scanpy selects via `nan_to_num(dispersion_norm, nan=-inf) >= cutoff`, so
    // NaN dispersions (zero-dispersion genes) must sort LAST and never be
    // selected. Order NaN as -inf.
    let mut indices: Vec<usize> = (0..n_vars).collect();
    indices.sort_by(|&a, &b| {
        let va = if dispersions_norm[a].is_nan() {
            f64::NEG_INFINITY
        } else {
            dispersions_norm[a]
        };
        let vb = if dispersions_norm[b].is_nan() {
            f64::NEG_INFINITY
        } else {
            dispersions_norm[b]
        };
        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut mask = vec![false; n_vars];
    for &g in indices.iter().take(n_top_genes.min(n_vars)) {
        // Never select a NaN-dispersion gene (scanpy's -inf floor).
        if !dispersions_norm[g].is_nan() {
            mask[g] = true;
        }
    }

    // ── 5. Write results to adata.var ───────────────────────────────────
    // scanpy publishes `means = log1p(count-space mean)` and
    // `dispersions = log(var/mean)` (both may be NaN for zero-dispersion genes).
    let var = adata.getattr("var")?;
    var.set_item(
        "highly_variable",
        numpy::PyArray::from_vec(py, mask.clone()),
    )?;
    var.set_item("means", numpy::PyArray::from_vec(py, log_means))?;
    var.set_item("dispersions", numpy::PyArray::from_vec(py, log_dispersions))?;
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
        apply_hvg_subset(py, adata, &mask)?;
    }

    Ok(())
}
