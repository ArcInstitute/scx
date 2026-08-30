//! Clustering agreement and label-comparison metrics:
//! `clustering_agreement`, `adjusted_mutual_info`, `normalized_mutual_info`,
//! `adjusted_rand_index`.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyDict;

// ──────────────────────────────────────────────────────────────────────────────
// device scaffolding (Phase 0 of CELL-EVAL-SCX-GPU-ACC.md)
// ──────────────────────────────────────────────────────────────────────────────
use super::*;

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
    crate::accel::prepare_target_no_var_guard(py, adata_pred, "clustering_agreement")?;
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
