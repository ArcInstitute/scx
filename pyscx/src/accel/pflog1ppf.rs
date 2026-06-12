//! PFlog1pPF / shifted-CLR normalization — Booeshaghi et al. 2026 (CPU-native).
//!
//! The exact transform `Z = delta + baseline·1ᵀ` is dense, but it decomposes
//! into a sparse `delta` (= the lazy `NormalizeTotal{1/c}→Log1p` chain) plus a
//! per-cell `baseline`. This binding:
//!
//! * always writes the per-cell `baseline` to `adata.obs[baseline_key]`;
//! * `store ∈ {"pca","all"}` runs baseline-aware out-of-core randomized PCA
//!   (`scx_accel::pflog1ppf_pca`) → `adata.obsm[obsm_key]` + singular values in
//!   `adata.uns[f"{obsm_key}_singular_values"]`;
//! * `store ∈ {"dense","all"}` materializes the exact dense `Z` into
//!   `adata.layers[layer_out]` — guarded to in-memory-feasible sizes
//!   (`dense_max_elems`); the streamed-to-disk path is deferred (Phase 4c).
//!
//! CPU-only: `device` is accepted for API symmetry but there is no GPU kernel.
//! PFlog1pPF requires **raw counts**; an already-transformed (lazy) `X` is
//! rejected (the raw-count guard).

use std::sync::Arc;

use numpy::{PyArray, PyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use scx_accel::{pflog1ppf_baseline_from_delta, pflog1ppf_pca, PcaResult};
use scx_format_io::shard_source::SingleShardSource;
use scx_format_io::ShardSource;
use scx_sparse::ScxCsr;

use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::{ScxLazyTransformedDataset, Transform};

use super::hvg::build_shard_source;
use super::util::extract_materialized_csr;

/// Default guard for the in-memory dense layer (`n_obs · n_vars` elements).
const DEFAULT_DENSE_MAX_ELEMS: usize = 200_000_000;

/// Apply PFlog1pPF normalization to `adata`.
#[pyfunction]
#[pyo3(signature = (
    adata,
    c=1.0,
    layer=None,
    *,
    store="pca",
    n_components=50,
    n_oversamples=10,
    n_power_iterations=2,
    zero_center=true,
    random_state=0,
    obsm_key="X_pflog1ppf_pca",
    baseline_key="pflog1ppf_baseline",
    layer_out=None,
    dense_max_elems=DEFAULT_DENSE_MAX_ELEMS,
    device="auto",
))]
#[allow(clippy::too_many_arguments)]
pub fn pflog1ppf(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    c: f64,
    layer: Option<&str>,
    store: &str,
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    random_state: u64,
    obsm_key: &str,
    baseline_key: &str,
    layer_out: Option<&str>,
    dense_max_elems: usize,
    device: &str,
) -> PyResult<()> {
    if c <= 0.0 || c.is_nan() {
        return Err(PyValueError::new_err(format!(
            "pflog1ppf: shift c must be positive, got {c}"
        )));
    }
    let (want_pca, want_dense) = match store {
        "pca" => (true, false),
        "baseline" => (false, false),
        "dense" => (false, true),
        "all" => (true, true),
        other => {
            return Err(PyValueError::new_err(format!(
                "pflog1ppf: unknown store {other:?}; expected \"pca\", \"baseline\", \"dense\", or \"all\""
            )));
        }
    };

    // Select the source matrix (layer or X).
    let x = match layer {
        Some(name) => adata.getattr("layers")?.get_item(name)?,
        None => adata.getattr("X")?,
    };

    // ── Build the delta source (NormalizeTotal{1/c} → Log1p) ────────────────
    // Three X kinds: backed (out-of-core), lazy (raw-count guard), in-memory.
    let target_sum = 1.0 / c;

    if let Ok(backed) = x.cast::<ScxBackedSparseDataset>() {
        let backed_ref = backed.borrow();
        let reader = Arc::clone(&backed_ref.backed);
        let n_vars = backed_ref.shape_val.1;
        let kept = backed_ref.kept_to_global.clone();
        let col_proj = backed_ref.col_projection_arc();
        // Physical-indexed (projected-aware) raw row sums — what NormalizeTotal
        // divides by (transforms run pre-deletion, pre-projection per shard).
        let phys_row_sums = if let Some(cols) = backed_ref.col_projection() {
            crate::projected_agg::row_sums_projected(&backed_ref.backed, cols)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        } else {
            backed_ref
                .backed
                .row_sums()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        };
        drop(backed_ref);
        let transforms = vec![
            Transform::NormalizeTotal {
                row_sums: Arc::new(phys_row_sums),
                target_sum,
            },
            Transform::Log1p,
        ];
        let source = build_shard_source(&reader, &transforms, &kept, &col_proj, n_vars);
        return run_on_source(
            py,
            adata,
            &source,
            want_pca,
            want_dense,
            n_components,
            n_oversamples,
            n_power_iterations,
            zero_center,
            random_state,
            obsm_key,
            baseline_key,
            layer_out,
            dense_max_elems,
            device,
        );
    }

    if let Ok(lazy) = x.cast::<ScxLazyTransformedDataset>() {
        let lazy_ref = lazy.borrow();
        if !lazy_ref.transforms.is_empty() {
            return Err(PyValueError::new_err(
                "pflog1ppf requires raw counts, but X is a lazy-transformed dataset that \
                 already carries transforms (e.g. normalize_total / log1p). Run pflog1ppf on \
                 the raw-count X instead.",
            ));
        }
        let reader = Arc::clone(&lazy_ref.backed);
        let n_vars = lazy_ref.shape_val.1;
        let kept = lazy_ref.kept_to_global.clone();
        let col_proj = lazy_ref.col_projection.clone();
        drop(lazy_ref);
        let phys_row_sums = match &col_proj {
            Some(cols) => crate::projected_agg::row_sums_projected(&reader, cols)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            None => reader
                .row_sums()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        };
        let transforms = vec![
            Transform::NormalizeTotal {
                row_sums: Arc::new(phys_row_sums),
                target_sum,
            },
            Transform::Log1p,
        ];
        let source = build_shard_source(&reader, &transforms, &kept, &col_proj, n_vars);
        return run_on_source(
            py,
            adata,
            &source,
            want_pca,
            want_dense,
            n_components,
            n_oversamples,
            n_power_iterations,
            zero_center,
            random_state,
            obsm_key,
            baseline_key,
            layer_out,
            dense_max_elems,
            device,
        );
    }

    // In-memory scipy/dense X → build the delta CSR directly, one shard.
    let raw = extract_materialized_csr(py, &x)?;
    let delta = delta_from_raw_csr(&raw, c)?;
    let source = SingleShardSource { csr: &delta };
    run_on_source(
        py,
        adata,
        &source,
        want_pca,
        want_dense,
        n_components,
        n_oversamples,
        n_power_iterations,
        zero_center,
        random_state,
        obsm_key,
        baseline_key,
        layer_out,
        dense_max_elems,
        device,
    )
}

/// Build the `delta` CSR `log1p(x_ij / (c·s_i))` from a raw-count CSR.
#[allow(clippy::needless_range_loop)]
fn delta_from_raw_csr(raw: &ScxCsr, c: f64) -> PyResult<ScxCsr> {
    let depths = raw.row_sums();
    let mut delta = raw.clone();
    for r in 0..raw.n_rows() {
        let depth = depths[r];
        if depth <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "pflog1ppf: cell {r} has non-positive depth {depth}; filter empty cells first"
            )));
        }
        let start = delta.indptr[r] as usize;
        let end = delta.indptr[r + 1] as usize;
        for v in &mut delta.data[start..end] {
            if (*v as f64) < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "pflog1ppf: negative count {v} at cell {r}; counts must be non-negative"
                )));
            }
            *v = ((*v as f64) / (c * depth)).ln_1p() as f32;
        }
    }
    Ok(delta)
}

/// Compute baseline + (optionally) PCA / dense over a delta `ShardSource`,
/// writing results into `adata` and stamping the CPU route.
#[allow(clippy::too_many_arguments)]
fn run_on_source<S: ShardSource>(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    source: &S,
    want_pca: bool,
    want_dense: bool,
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    random_state: u64,
    obsm_key: &str,
    baseline_key: &str,
    layer_out: Option<&str>,
    dense_max_elems: usize,
    device: &str,
) -> PyResult<()> {
    let (n_obs, n_vars) = source.shape();

    // Baseline (always) → adata.obs[baseline_key].
    let baseline = pflog1ppf_baseline_from_delta(source)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let baseline_arr = PyArray::from_slice(py, &baseline);
    adata.getattr("obs")?.set_item(baseline_key, baseline_arr)?;

    if want_pca {
        let result: PcaResult = pflog1ppf_pca(
            source,
            &baseline,
            n_components,
            n_oversamples,
            n_power_iterations,
            zero_center,
            random_state,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        write_pca(py, adata, &result, obsm_key)?;
    }

    if want_dense {
        let n_elems = n_obs.saturating_mul(n_vars);
        if n_elems > dense_max_elems {
            return Err(PyRuntimeError::new_err(format!(
                "pflog1ppf: dense materialization is {n_obs}×{n_vars} = {n_elems} elements, \
                 over the dense_max_elems={dense_max_elems} guard. The exact PFlog1pPF transform \
                 is dense; use store=\"pca\" for an out-of-core embedding, or raise dense_max_elems \
                 if you truly have the RAM (streamed-to-disk materialization is a separate path)."
            )));
        }
        let dense = materialize_dense(source, &baseline)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let arr = PyArray2::<f32>::from_vec2(py, &dense)?;
        let key = layer_out.unwrap_or("pflog1ppf");
        adata.getattr("layers")?.set_item(key, arr)?;
    }

    super::route::write_accel_route(
        py,
        adata,
        "pflog1ppf",
        &super::route::cpu_only_exec_info(device),
    )?;
    Ok(())
}

/// Materialize the exact dense `Z = delta + baseline·1ᵀ` (f32, row-major).
fn materialize_dense<S: ShardSource>(
    source: &S,
    baseline: &[f64],
) -> Result<Vec<Vec<f32>>, scx_accel::AccelError> {
    let (n_obs, n_vars) = source.shape();
    let mut dense = vec![vec![0.0f32; n_vars]; n_obs];
    let mut row_base = 0usize;
    for shard_idx in 0..source.n_shards() {
        let csr = source.read_shard(shard_idx)?;
        let rows = csr.n_rows();
        for r in 0..rows {
            let cell = row_base + r;
            let b = baseline[cell] as f32;
            let out = &mut dense[cell];
            for v in out.iter_mut() {
                *v = b;
            }
            let start = csr.indptr[r] as usize;
            let end = csr.indptr[r + 1] as usize;
            for nz in start..end {
                let col = csr.indices[nz] as usize;
                out[col] += csr.data[nz];
            }
        }
        row_base += rows;
    }
    Ok(dense)
}

/// Write PFlog1pPF PCA results: `obsm[obsm_key]` (embeddings, f32) and
/// `uns[f"{obsm_key}_singular_values"]` (σ_i = sqrt(var_explained_i·(n−1))).
fn write_pca(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    result: &PcaResult,
    obsm_key: &str,
) -> PyResult<()> {
    let embeddings = PyArray2::<f32>::from_vec2(
        py,
        &(0..result.n_obs)
            .map(|i| {
                (0..result.n_components)
                    .map(|j| result.embeddings[i * result.n_components + j] as f32)
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>(),
    )?;
    adata.getattr("obsm")?.set_item(obsm_key, embeddings)?;

    let scale = (result.n_obs as f64 - 1.0).max(1.0);
    let singular: Vec<f64> = result
        .variance_explained
        .iter()
        .map(|&ve| (ve * scale).max(0.0).sqrt())
        .collect();
    let sv_arr = PyArray::from_slice(py, &singular);
    adata
        .getattr("uns")?
        .set_item(format!("{obsm_key}_singular_values"), sv_arr)?;
    Ok(())
}
