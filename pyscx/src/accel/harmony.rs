//! Harmony2 batch integration — Python binding.
//!
//! Reads PCA embeddings from `adata.obsm[basis]`, factorizes batch labels
//! from `adata.obs[key]`, runs the Rust Harmony2 core, and writes corrected
//! embeddings to `adata.obsm[adjusted_basis]` (default "X_pca_harmony",
//! preserving the input `basis`) with metadata at `adata.uns["harmony"]`.

use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use scx_accel::{BatchCovariate, HarmonyConfig};

use super::gpu::resolve_device;

/// Run Harmony2 batch integration on PCA embeddings stored in AnnData.
///
/// **Compatibility tier 2 — scanpy-shaped, documented divergence.** Same role
/// and signature as `scanpy.external.pp.harmony_integrate()`, but a distinct
/// clean-room **Harmony2 / R-harmony** integrator — NOT the harmonypy algorithm
/// scanpy wraps. The pinned reference is **R-harmony**, validated against cached
/// R fixtures (per-PC Pearson r; see `pyscx/tests/test_harmony_validation.py`);
/// there is no harmonypy-parity mode. Corrects batch effects in
/// `adata.obsm[basis]` via iterative soft clustering + ridge regression and
/// writes the corrected embeddings to `adjusted_basis` (a new obsm key),
/// preserving the input `basis`.
///
/// Defaults intentionally differ from harmonypy (documented divergence):
/// `epsilon_harmony=1e-2` (harmonypy `1e-4`), `lamb=None` → dynamic estimation
/// (harmonypy `lamb=1`), `theta=2.0` per covariate, `sigma=0.1`,
/// `max_iter_kmeans=6`. Do not treat harmonypy differences as bugs — the
/// R-harmony fixture is the acceptance oracle.
///
/// Args:
///     adata: AnnData with PCA embeddings at `adata.obsm[basis]`.
///     key: obs column name, or list of obs column names, identifying batch
///         variable(s). Each column is factorized to contiguous integer
///         labels internally.
///     basis: obsm key holding the input embeddings (default "X_pca").
///     adjusted_basis: obsm key for the corrected embeddings (default
///         "X_pca_harmony", matching scanpy — preserves the input `basis`).
///         Pass `adjusted_basis=basis` (e.g. "X_pca") to overwrite in place.
///     n_clusters: K (default min(N/30, 100), clamped to [2, N/2]).
///     theta: Diversity penalty per covariate (scalar broadcasts; default 2.0).
///     sigma: Soft assignment kernel bandwidth (default 0.1).
///     lamb: Ridge penalty. None = dynamic estimation (default).
///     alpha: Dynamic lambda scale factor (default 0.2).
///     max_iter: Maximum Harmony iterations (default 10).
///     max_iter_kmeans: Maximum k-means sub-iterations (default 6; must be
///         >= 2*window_size so the k-means convergence check can fire).
///     epsilon_harmony: Harmony convergence tolerance (default 1e-2).
///     epsilon_kmeans: K-means convergence tolerance (default 1e-3).
///     block_size: Stochastic block size as fraction of N (default 0.05).
///     batch_prop_cutoff: Minimum batch proportion per cluster (default 1e-5).
///     tau: Overcorrection protection (default 0.0).
///     random_state: RNG seed (default 0).
///     device: Device selection — "auto" (default), "cpu", "gpu", or
///         "gpu:N" to target CUDA device N on multi-GPU systems. GPU path
///         routes to `scx_accel::harmony_integrate_gpu` (cuBLAS + custom
///         CUDA kernels for distance / L2-normalize / batched
///         scatter-subtract); per-PC Pearson r ≥ 0.99 vs CPU on validation
///         fixtures (see `pyscx/tests/test_harmony_validation.py`).
#[pyfunction]
#[pyo3(signature = (
    adata,
    key,
    *,
    basis = "X_pca",
    adjusted_basis = "X_pca_harmony",
    n_clusters = None,
    theta = None,
    sigma = 0.1,
    lamb = None,
    alpha = 0.2,
    max_iter = 10,
    max_iter_kmeans = 6,
    epsilon_harmony = 1e-2,
    epsilon_kmeans = 1e-3,
    block_size = 0.05,
    batch_prop_cutoff = 1e-5,
    tau = 0.0,
    random_state = 0,
    device = "auto",
))]
#[allow(clippy::too_many_arguments)]
pub fn harmony_integrate(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    key: &Bound<'_, PyAny>,
    basis: &str,
    adjusted_basis: Option<&str>,
    n_clusters: Option<usize>,
    theta: Option<&Bound<'_, PyAny>>,
    sigma: f64,
    lamb: Option<&Bound<'_, PyAny>>,
    alpha: f64,
    max_iter: usize,
    max_iter_kmeans: usize,
    epsilon_harmony: f64,
    epsilon_kmeans: f64,
    block_size: f64,
    batch_prop_cutoff: f64,
    tau: f64,
    random_state: u64,
    device: &str,
) -> PyResult<()> {
    // Rebuild an AnnData view as actual before the first write below (obsm /
    // uns), so a backed X is not gathered by anndata's copy-on-write. No
    // var-order guard: this op reads obsm and is gene-order agnostic.
    super::prepare_target_no_var_guard(py, adata, "harmony_integrate")?;

    // Resolve device. `_device.gpu_id()` is the CUDA index that GPU dispatch
    // forwards to `scx_accel::harmony_integrate_gpu(device_id, ...)`.
    let _device = resolve_device(device)?;
    #[cfg(feature = "gpu")]
    let _gpu_id = _device.gpu_id();

    // --- Extract embeddings (N x d, f32 row-major) ---
    let obsm = adata.getattr("obsm")?;
    let emb_obj = obsm.get_item(basis).map_err(|_| {
        PyRuntimeError::new_err(format!(
            "'{basis}' not found in adata.obsm. Run PCA first: pyscx.accel.pca(adata)",
        ))
    })?;

    let np = py.import("numpy")?;
    let emb_f32 = np
        .call_method1("ascontiguousarray", (&emb_obj,))?
        .call_method1("astype", ("float32",))?;
    let emb_arr: &Bound<'_, PyArray2<f32>> = emb_f32.cast::<PyArray2<f32>>().map_err(|e| {
        PyRuntimeError::new_err(format!("failed to view '{basis}' as 2D float32: {e}"))
    })?;
    let shape = emb_arr.shape();
    if shape.len() != 2 {
        return Err(PyRuntimeError::new_err(format!(
            "adata.obsm['{basis}'] must be 2D, got shape {:?}",
            shape
        )));
    }
    let n_obs = shape[0];
    let n_pcs = shape[1];
    if n_obs == 0 || n_pcs == 0 {
        return Err(PyRuntimeError::new_err(format!(
            "adata.obsm['{basis}'] is empty (shape {}x{})",
            n_obs, n_pcs
        )));
    }

    let embeddings: Vec<f32> = {
        // Safe readonly view; copy into a Vec so we can release the GIL.
        let ro = emb_arr.readonly();
        let sl = ro.as_slice().map_err(|e| {
            PyRuntimeError::new_err(format!("adata.obsm['{basis}'] is not C-contiguous: {e}"))
        })?;
        sl.to_vec()
    };

    // Reject NaN / Inf early with an informative message. Rust core would
    // also catch these, but the error path there has less Python context.
    if !embeddings.iter().all(|v| v.is_finite()) {
        return Err(PyRuntimeError::new_err(format!(
            "adata.obsm['{basis}'] contains NaN or Inf"
        )));
    }

    // --- Extract batch covariate name(s) from `key` ---
    let key_names: Vec<String> = if let Ok(s) = key.extract::<String>() {
        vec![s]
    } else if let Ok(v) = key.extract::<Vec<String>>() {
        if v.is_empty() {
            return Err(PyValueError::new_err(
                "`key` must be a non-empty string or list of strings",
            ));
        }
        v
    } else {
        return Err(PyValueError::new_err(
            "`key` must be a string or a list of strings",
        ));
    };

    // Build BatchCovariate for each key by factorizing the obs column.
    let obs = adata.getattr("obs")?;
    let mut covariates: Vec<BatchCovariate> = Vec::with_capacity(key_names.len());
    for k in &key_names {
        let col = obs.get_item(k.as_str()).map_err(|_| {
            PyRuntimeError::new_err(format!("obs column '{k}' not found in adata.obs"))
        })?;
        let (labels, n_levels) = factorize_obs_column(py, &col, k)?;
        if labels.len() != n_obs {
            return Err(PyRuntimeError::new_err(format!(
                "obs column '{k}' length {} != n_obs {}",
                labels.len(),
                n_obs
            )));
        }
        covariates.push(BatchCovariate {
            labels,
            n_levels,
            name: Some(k.clone()),
        });
    }

    // --- Broadcast theta / lamb to per-covariate Vec ---
    let theta_vec = broadcast_per_covariate(theta, covariates.len(), "theta")?;
    let lambda_vec = broadcast_per_covariate(lamb, covariates.len(), "lamb")?;

    // --- Build Rust config ---
    let config = HarmonyConfig {
        n_clusters,
        theta: theta_vec,
        sigma,
        lambda: lambda_vec,
        alpha,
        max_iter,
        max_iter_kmeans,
        epsilon_harmony,
        epsilon_kmeans,
        window_size: 3,
        block_size,
        batch_prop_cutoff,
        tau,
        random_state,
        n_threads: None,
    };

    // --- Dispatch Rust core with GIL released ---
    // GPU path is only compiled when pyscx is built with `--features gpu`.
    #[cfg(feature = "gpu")]
    let (result, backend) = if let Some(device_id) = _gpu_id {
        let r = py
            .detach(|| {
                scx_accel::harmony_integrate_gpu(
                    device_id,
                    &embeddings,
                    n_obs,
                    n_pcs,
                    &covariates,
                    &config,
                )
            })
            .map_err(|e: scx_accel::AccelError| {
                PyRuntimeError::new_err(format!("harmony_integrate_gpu: {e}"))
            })?;
        (r, "scx-gpu")
    } else {
        let r = py
            .detach(|| {
                scx_accel::harmony_integrate(&embeddings, n_obs, n_pcs, &covariates, &config)
            })
            .map_err(|e: scx_accel::AccelError| {
                PyRuntimeError::new_err(format!("harmony_integrate: {e}"))
            })?;
        (r, "scx-accel-cpu")
    };

    #[cfg(not(feature = "gpu"))]
    let (result, backend) = {
        // Without the `gpu` feature, `resolve_device` rejects device="gpu";
        // "auto" collapses to CPU.
        let _ = _device;
        let r = py
            .detach(|| {
                scx_accel::harmony_integrate(&embeddings, n_obs, n_pcs, &covariates, &config)
            })
            .map_err(|e: scx_accel::AccelError| {
                PyRuntimeError::new_err(format!("harmony_integrate: {e}"))
            })?;
        (r, "scx-accel-cpu")
    };

    // --- Write corrected embeddings back as float32 (N x d) ---
    let out_f32: Vec<f32> = result.z_corrected.iter().map(|&v| v as f32).collect();
    let out_arr = unsafe { PyArray2::<f32>::new(py, [n_obs, n_pcs], false) };
    {
        let mut rw = out_arr.readwrite();
        let slice = rw
            .as_slice_mut()
            .map_err(|e| PyRuntimeError::new_err(format!("output array slice error: {e}")))?;
        slice.copy_from_slice(&out_f32);
    }

    let out_key = adjusted_basis.unwrap_or(basis);
    obsm.set_item(out_key, &out_arr)?;

    // --- Write metadata to adata.uns["harmony"] ---
    let uns = adata.getattr("uns")?;
    let info = PyDict::new(py);

    let params = PyDict::new(py);
    // Store `key` as either str or list[str] matching the caller shape.
    if key_names.len() == 1 {
        params.set_item("key", &key_names[0])?;
    } else {
        let list = PyList::new(py, &key_names)?;
        params.set_item("key", list)?;
    }
    params.set_item("basis", basis)?;
    params.set_item("adjusted_basis", out_key)?;
    params.set_item("n_clusters", result.n_clusters)?;
    params.set_item("sigma", sigma)?;
    params.set_item("alpha", alpha)?;
    params.set_item("max_iter", max_iter)?;
    params.set_item("max_iter_kmeans", max_iter_kmeans)?;
    params.set_item("epsilon_harmony", epsilon_harmony)?;
    params.set_item("epsilon_kmeans", epsilon_kmeans)?;
    params.set_item("block_size", block_size)?;
    params.set_item("batch_prop_cutoff", batch_prop_cutoff)?;
    params.set_item("tau", tau)?;
    params.set_item("random_state", random_state)?;
    info.set_item("params", params)?;

    info.set_item("converged", result.converged)?;
    info.set_item("n_iterations", result.n_iterations)?;
    let obj_arr = numpy::PyArray1::from_slice(py, &result.objective_harmony);
    info.set_item("objective_harmony", obj_arr)?;
    info.set_item("backend", backend)?;

    uns.set_item("harmony", info)?;

    // Stamp the canonical route envelope on adata.uns["scx_accel"]["harmony_integrate"]
    // so users can prove GPU-vs-CPU dispatch the same way PCA / kNN / UMAP do.
    // Harmony's GPU path is native (cuBLAS + custom kernels, no rapids dependency)
    // and operates on a dense embedding, so the route pair is GpuDense / CpuDense.
    // `gpu_eligible = true` because there is no input-layout restriction; the
    // planner resolves the actual route from the device intent + GPU availability,
    // matching the dispatch branch above (GPU branch runs iff resolve_device
    // returned a CUDA id, i.e. device="gpu"/"gpu:N" or "auto" with a GPU present).
    //
    // `graph_replay` is re-stamped from the result rather than left `None`:
    // Harmony is the only production CUDA-graph capture site in the tree, and
    // a capture that fails silently re-runs every k-means sub-iter directly
    // for the rest of the call. Without this field that slowdown had no
    // observable signal at all. `fallback_reason` deliberately stays `none` —
    // the GPU route did run, and the numbers are identical either way.
    let mut info = super::route::simple_exec_info(
        device,
        true,
        scx_accel::AccelRoute::GpuDense,
        scx_accel::AccelRoute::CpuDense,
    );
    if info.route.is_gpu() {
        info.graph_replay = result.graph_replay;
    }
    super::route::announce_route(py, "harmony_integrate", device, &info);
    super::route::write_accel_route(py, adata, "harmony_integrate", &info)?;

    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────

/// Factorize an obs column (categorical, object, or numeric) to
/// `(labels: Vec<u32>, n_levels: usize)`. Uses `pandas.factorize(sort=False)`
/// so level indices are stable under the input order.
fn factorize_obs_column(
    py: Python<'_>,
    col: &Bound<'_, PyAny>,
    col_name: &str,
) -> PyResult<(Vec<u32>, usize)> {
    let pd = py.import("pandas")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("sort", false)?;
    let tup = pd.call_method("factorize", (col,), Some(&kwargs))?;
    let codes = tup.get_item(0)?; // numpy array of int64 codes
    let uniques = tup.get_item(1)?;

    // codes may contain -1 for NaN values; reject those up front since
    // Harmony does not define batch membership for missing values.
    let has_nan: bool = codes
        .call_method1("__lt__", (0,))?
        .call_method0("any")?
        .extract()?;
    if has_nan {
        return Err(PyValueError::new_err(format!(
            "obs column '{col_name}' contains NaN / missing values; drop or impute them first",
        )));
    }

    let codes_i64: Vec<i64> = codes
        .call_method1("astype", ("int64",))?
        .call_method0("tolist")?
        .extract()?;
    let labels: Vec<u32> = codes_i64.iter().map(|&v| v as u32).collect();

    let n_levels: usize = uniques.len()?;
    if n_levels == 0 {
        return Err(PyValueError::new_err(format!(
            "obs column '{col_name}' has zero unique values",
        )));
    }
    Ok((labels, n_levels))
}

/// Convert a Python scalar or sequence into `Option<Vec<f64>>` of length
/// `n_cov` (broadcast scalar → repeated). None input stays None.
fn broadcast_per_covariate(
    value: Option<&Bound<'_, PyAny>>,
    n_cov: usize,
    param_name: &str,
) -> PyResult<Option<Vec<f64>>> {
    let Some(v) = value else { return Ok(None) };
    if v.is_none() {
        return Ok(None);
    }
    if let Ok(scalar) = v.extract::<f64>() {
        return Ok(Some(vec![scalar; n_cov]));
    }
    if let Ok(seq) = v.extract::<Vec<f64>>() {
        if seq.len() != n_cov {
            return Err(PyValueError::new_err(format!(
                "`{param_name}` length {} does not match number of covariates {}",
                seq.len(),
                n_cov
            )));
        }
        return Ok(Some(seq));
    }
    Err(PyValueError::new_err(format!(
        "`{param_name}` must be a float or a sequence of floats"
    )))
}
