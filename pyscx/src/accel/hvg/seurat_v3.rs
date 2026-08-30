//! The streaming seurat_v3 kernel (row-major / CSR shard sources).

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyDict;

use super::*;
use crate::optional_deps::{import_optional, EXTRA_HVG};

/// seurat_v3 flavor: raw count data, loess fit, clipped variance.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hvg_seurat_v3<'py, S: scx_format_io::ShardSource + Sync>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    source: &S,
    n_obs: usize,
    n_vars: usize,
    n_top_genes: usize,
    batch_key: Option<&str>,
    span: f64,
    subset: bool,
    flavor: &str,
    _device_id: Option<usize>,
) -> PyResult<()> {
    // Route-at-start visibility (report P1): the streaming mean/var passes below
    // run inside `py.detach`, and on backed / atlas-scale input the CPU-bound
    // shard decode dominates while the GPU waits — so a multi-minute run shows
    // no output and ~0% GPU util, which reads as a hang. Log the shape + device
    // at op start (INFO; enable via `logging.basicConfig(level=logging.INFO)`)
    // so the user can tell "GPU route, decode-bound" from "hung".
    #[cfg(feature = "gpu")]
    let on_gpu = _device_id.is_some();
    #[cfg(not(feature = "gpu"))]
    let on_gpu = false;
    log::info!(
        target: "pyscx.accel",
        "highly_variable_genes: {flavor} over {n_obs}\u{00d7}{n_vars} on {} — streaming \
         mean/var; large/backed input is CPU-decode-bound (GPU may show low utilization), \
         not hung",
        if on_gpu { "gpu" } else { "cpu" },
    );

    // ── 1. Determine batches ────────────────────────────────────────────
    // Category label per batch, in the same order as `batches`. `None` when
    // there is no `batch_key` (one implicit batch covering every cell).
    let mut batch_labels: Option<Vec<String>> = None;
    let batches: Vec<Vec<usize>> = match batch_key {
        Some(bk) => {
            let obs = adata.getattr("obs")?;
            let batch_col = obs.get_item(bk)?;
            // Handle both categorical and non-categorical columns:
            // wrap in pd.Categorical() which is a no-op for already-categorical data.
            let pd = crate::pyimport::import_module(py, "pandas")?;
            let np = crate::pyimport::import_module(py, "numpy")?;
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
            // Keep each group's category label with it. Empty groups are
            // dropped below, so a batch's position in `batches` is NOT an
            // index into `cat.categories` — which is exactly why reporting a
            // bare index in the loess-failure warning was unactionable
            // (user-report F4). Carrying the label is the only way to name
            // the batch afterwards.
            let labels: Vec<String> = cat
                .getattr("categories")?
                .call_method0("tolist")?
                .extract::<Vec<Bound<'_, PyAny>>>()?
                .iter()
                .map(|v| v.str().map(|s| s.to_string_lossy().into_owned()))
                .collect::<PyResult<Vec<String>>>()?;
            let kept: Vec<(String, Vec<usize>)> = groups
                .into_iter()
                .enumerate()
                .filter(|(_, g)| !g.is_empty())
                .map(|(i, g)| {
                    let label = labels
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("<code {i}>"));
                    (label, g)
                })
                .collect();
            let (labels, groups): (Vec<String>, Vec<Vec<usize>>) = kept.into_iter().unzip();
            batch_labels = Some(labels);
            groups
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
    // GPU path: per-batch streaming mean/var via the batched device wrapper.
    // The wrapper finalises Bessel-corrected means/variances and the global
    // accumulator on host, matching the CPU formula byte-for-byte.
    #[cfg(feature = "gpu")]
    let batched_stats = if let Some(dev_id) = _device_id {
        py.detach(|| {
            scx_accel::streaming_mean_var_batched_with_device(
                source,
                &cell_batch,
                n_batches_actual,
                "gpu",
                dev_id,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(format!("gpu streaming_mean_var_batched: {e}")))?
    } else {
        py.detach(|| scx_accel::streaming_mean_var_batched(source, &cell_batch, n_batches_actual))
            .map_err(|e| PyRuntimeError::new_err(format!("streaming_mean_var_batched: {e}")))?
    };
    #[cfg(not(feature = "gpu"))]
    let batched_stats = py
        .detach(|| scx_accel::streaming_mean_var_batched(source, &cell_batch, n_batches_actual))
        .map_err(|e| PyRuntimeError::new_err(format!("streaming_mean_var_batched: {e}")))?;

    let global_stats = batched_stats.global.clone();

    // ── 3. Per-batch: loess fit → clip_val (in-memory, no I/O) ───────────
    let mut all_clip_vals: Vec<Vec<f64>> = Vec::new();
    let mut batch_estimat_vars: Vec<Vec<f64>> = Vec::new();
    // Tracks which batches had their per-batch loess fit fail (singular /
    // under-determined). Failed batches are skipped in step 5's
    // per-batch normalised-variance computation and excluded from the
    // cross-batch mean and median-rank aggregations below — the contract
    // the UserWarning text promises.
    let mut batch_failed = vec![false; n_batches_actual];
    // Collected `(batch_idx, batch_n, error_string)` for batches whose loess
    // fit raised a singularity `ValueError`. Coalesced into a single summary
    // UserWarning after the loop (user-report F10) instead of one warning per
    // failing batch — a high-cardinality batch_key can fail dozens of batches.
    let mut loess_failed: Vec<LoessFailure> = Vec::new();

    // Hoist the loess import out of the per-batch closure so a missing
    // or broken `skmisc.loess` fails fast with its real type
    // (`ModuleNotFoundError` / `AttributeError`) instead of getting
    // swallowed by the narrow `PyValueError` catch in the closure below.
    let loess_mod = import_optional(
        py,
        "skmisc.loess",
        EXTRA_HVG,
        "pyscx.accel.highly_variable_genes(flavor=\"seurat_v3\")",
        "scikit-misc",
    )?;
    let loess_cls = loess_mod.getattr("loess")?;

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

        // Captured before `x_vals` is moved into the numpy array: the number of
        // points the regression actually got. Its gap to `n_vars` is what makes
        // a singular fit singular, so the diagnostic needs it (E3).
        let n_fit_points = x_vals.len();

        if n_fit_points >= 3 {
            let x_arr = numpy::PyArray::from_vec(py, x_vals);
            let y_arr = numpy::PyArray::from_vec(py, y_vals);

            let fit_result: PyResult<Vec<f64>> = (|| -> PyResult<Vec<f64>> {
                let kwargs = PyDict::new(py);
                kwargs.set_item("span", span)?;
                kwargs.set_item("degree", 2)?;
                let model = loess_cls.call((x_arr, y_arr), Some(&kwargs))?;
                model.call_method0("fit")?;
                model
                    .getattr("outputs")?
                    .getattr("fitted_values")?
                    .extract::<Vec<f64>>()
            })();

            match fit_result {
                Ok(fitted) => {
                    let mut fi = 0;
                    for (j, &nc) in not_const.iter().enumerate() {
                        if nc {
                            estimat_var[j] = fitted[fi];
                            fi += 1;
                        }
                    }
                }
                Err(e) if e.is_instance_of::<PyValueError>(py) => {
                    // Singular / under-determined LOESS — `skmisc.loess`
                    // raises `ValueError` ("There are other near
                    // singularities…") on Census-style degenerate
                    // batches. Record the batch and mark it failed so
                    // step 5 and the rank step exclude it from per-batch
                    // and cross-batch aggregation; a single coalesced
                    // summary warning is emitted after the loop. We
                    // cannot rely on `estimat_var == 0` to opt out — a
                    // successful loess fit can legitimately produce zero
                    // entries, and downstream `reg_std_sq = 10^0 = 1` so
                    // a zero `estimat_var` would still pass the
                    // `reg_std_sq > 0` guard.
                    loess_failed.push(LoessFailure {
                        batch_idx: b,
                        n_cells: batch_n,
                        n_fit_points,
                        error: e.to_string(),
                    });
                    batch_failed[b] = true;
                }
                Err(e) => {
                    // Any other PyErr (TypeError, RuntimeError,
                    // MemoryError, etc.) is a real environmental issue,
                    // not a benign singularity — propagate so the user
                    // sees the real cause instead of a misleading
                    // "this batch had a singular loess fit" warning.
                    return Err(e);
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

    // Coalesce per-batch loess singularities into ONE summary UserWarning
    // (user-report F10) — a high-cardinality batch_key can fail dozens of
    // batches, and one verbatim warning each buries the signal. The full
    // per-batch list is recorded on adata.uns["hvg"]["loess_failed_batches"]
    // for callers who want the complete detail.
    //
    // Record unconditionally: writing the current list every
    // run — `[]` when nothing failed — clears any stale `loess_failed_batches`
    // left on a reused AnnData by an earlier failed run, and merges into (rather
    // than clobbers) any pre-existing uns["hvg"]. The summary warning stays
    // guarded by a non-empty list, and is emitted before the all-failed check
    // below so the diagnostic survives even that error.
    record_hvg_loess_failed_batches(py, adata, &loess_failed, batch_labels.as_deref())?;
    if !loess_failed.is_empty() {
        emit_hvg_loess_singularity_warning(
            py,
            &loess_failed,
            n_batches_actual,
            n_vars,
            batch_key,
            batch_labels.as_deref(),
        )?;
    }

    // Surviving batches (per-batch loess fit succeeded). If all batches
    // failed there's nothing to rank against and dividing by zero in the
    // cross-batch average below would silently produce NaN HVGs — raise
    // so the user sees the summary UserWarning as the cause.
    let n_valid_batches = batch_failed.iter().filter(|&&f| !f).count();
    if n_valid_batches == 0 {
        // Phrase the unbatched case as the single fit it is. "all 1 batches
        // failed" invites the user to go looking for a batch_key they never
        // passed (E3), and points at "per-batch causes" that do not exist.
        return Err(PyRuntimeError::new_err(if batch_key.is_some() {
            format!(
                "highly_variable_genes(flavor=\"seurat_v3\"): all {n_batches_actual} \
                 batches failed skmisc.loess fitting; see the preceding summary \
                 UserWarning (and adata.uns[\"hvg\"][\"loess_failed_batches\"]) for \
                 per-batch causes."
            )
        } else {
            "highly_variable_genes(flavor=\"seurat_v3\"): the skmisc.loess fit failed, \
             so there is no variance trend to rank genes against; see the preceding \
             summary UserWarning for the cause and the remedy \
             (pyscx.accel.filter_genes(min_cells=10) is the usual one)."
                .to_string()
        }));
    }

    // ── 4. Batched streaming clipped sums (single pass for ALL batches) ──
    #[cfg(feature = "gpu")]
    let all_clipped = if let Some(dev_id) = _device_id {
        py.detach(|| {
            scx_accel::streaming_clip_square_sum_batched_with_device(
                source,
                &cell_batch,
                n_batches_actual,
                &all_clip_vals,
                "gpu",
                dev_id,
            )
        })
        .map_err(|e| {
            PyRuntimeError::new_err(format!("gpu streaming_clip_square_sum_batched: {e}"))
        })?
    } else {
        py.detach(|| {
            scx_accel::streaming_clip_square_sum_batched(
                source,
                &cell_batch,
                n_batches_actual,
                &all_clip_vals,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(format!("streaming_clip_square_sum_batched: {e}")))?
    };
    #[cfg(not(feature = "gpu"))]
    let all_clipped = py
        .detach(|| {
            scx_accel::streaming_clip_square_sum_batched(
                source,
                &cell_batch,
                n_batches_actual,
                &all_clip_vals,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(format!("streaming_clip_square_sum_batched: {e}")))?;

    // ── 5. Compute normalized variance per batch (in-memory) ─────────────
    let mut all_norm_vars: Vec<Vec<f64>> = Vec::new();
    for (b, batch_cells) in batches.iter().enumerate() {
        let batch_n = batch_cells.len();
        if batch_n < 2 || batch_failed[b] {
            // Push a zero row so `all_norm_vars` stays indexable by batch
            // id; the rank step skips these by index via `batch_failed`.
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
    // Mean normalized variance across **surviving** batches only. Failed
    // batches contributed a zero row to `all_norm_vars` (see step 5) but
    // must not enter the average — we divide by `n_valid_batches`, not
    // `all_norm_vars.len()`.
    let mut mean_norm_var = vec![0.0f64; n_vars];
    for (b, nv) in all_norm_vars.iter().enumerate() {
        if batch_failed[b] {
            continue;
        }
        for (j, &v) in nv.iter().enumerate() {
            mean_norm_var[j] += v;
        }
    }
    for v in &mut mean_norm_var {
        *v /= n_valid_batches as f64;
    }

    // Multi-batch ranking when more than one batch survived. When only
    // one batch survives (`n_valid_batches == 1`), `mean_norm_var` equals
    // that batch's `norm_gene_var`, so the single-batch ranking branch
    // produces the same HVG mask as a direct 1-batch run on that batch
    // alone — that's the contract the parity test in
    // tests/test_hvg_loess_singularity.py asserts.
    // Per-batch dense ranks (0 = most variable), matching scanpy's
    // `argsort(argsort(-norm_gene_vars))`. Failed batches cast no rank vote and
    // don't count toward `nbatches_hv` / `median_ranks`. A single surviving
    // batch is just the 1-row case of the same algorithm (its `mean_norm_var`
    // equals that batch's `norm_gene_var`), so we use one code path for both.
    let mut batch_ranks: Vec<Vec<usize>> = Vec::new();
    for (b, nv) in all_norm_vars.iter().enumerate() {
        if batch_failed[b] {
            continue;
        }
        let mut indices: Vec<usize> = (0..n_vars).collect();
        indices.sort_by(|&a, &c| {
            nv[c]
                .partial_cmp(&nv[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut rank = vec![0usize; n_vars];
        for (r, &idx) in indices.iter().enumerate() {
            rank[idx] = r;
        }
        batch_ranks.push(rank);
    }

    // scanpy: `num_batches_high_var = sum(rank < n_top_genes, axis=0)`;
    // ranks >= n_top_genes are masked to NaN, then `highly_variable_rank` is
    // the per-gene `np.ma.median` over surviving batches (NaN when the gene is
    // never in any batch's top-N). `np.ma.median` AVERAGES the two middle
    // values for an even count — not the upper-middle element.
    let mut nbatches_hv = vec![0u32; n_vars];
    let mut median_ranks = vec![f64::NAN; n_vars];
    for j in 0..n_vars {
        let mut valid: Vec<f64> = batch_ranks
            .iter()
            .map(|br| br[j])
            .filter(|&r| r < n_top_genes)
            .map(|r| r as f64)
            .collect();
        nbatches_hv[j] = valid.len() as u32;
        if !valid.is_empty() {
            valid.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let m = valid.len();
            median_ranks[j] = if m % 2 == 1 {
                valid[m / 2]
            } else {
                0.5 * (valid[m / 2 - 1] + valid[m / 2])
            };
        }
    }

    // Sort with scanpy's key + `na_position="last"`: NaN median-ranks sort
    // after all finite ranks (mapped to +inf here). seurat_v3 sorts by
    // (rank asc, nbatches desc); seurat_v3_paper by (nbatches desc, rank asc).
    let rank_key = |g: usize| {
        if median_ranks[g].is_nan() {
            f64::INFINITY
        } else {
            median_ranks[g]
        }
    };
    let mut gene_order: Vec<usize> = (0..n_vars).collect();
    if flavor == "seurat_v3_paper" {
        gene_order.sort_by(|&a, &b| {
            nbatches_hv[b].cmp(&nbatches_hv[a]).then(
                rank_key(a)
                    .partial_cmp(&rank_key(b))
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
    } else {
        gene_order.sort_by(|&a, &b| {
            rank_key(a)
                .partial_cmp(&rank_key(b))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(nbatches_hv[b].cmp(&nbatches_hv[a]))
        });
    }

    // scanpy: `highly_variable = sorted_index[:n_top_genes]`. The published
    // `highly_variable_rank` is the per-gene median rank (NaN-preserving), NOT
    // the final selection order.
    let mut hvg_mask = vec![false; n_vars];
    for &g in gene_order.iter().take(n_top_genes.min(n_vars)) {
        hvg_mask[g] = true;
    }
    let ranks = median_ranks;

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
    // scanpy writes `highly_variable_nbatches` only when a batch_key was given.
    if batch_key.is_some() {
        var.set_item(
            "highly_variable_nbatches",
            numpy::PyArray::from_vec(py, nbatches_hv),
        )?;
    }

    // ── 6. Subset if requested ──────────────────────────────────────────
    if subset {
        apply_hvg_subset(py, adata, &hvg_mask)?;
    }

    Ok(())
}
