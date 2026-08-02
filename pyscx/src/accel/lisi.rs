//! LISI (Local Inverse Simpson Index) — Python binding.
//!
//! Reads PCA embeddings from `adata.obsm[basis]`, factorises categorical
//! labels from `adata.obs[key]`, and computes per-cell LISI using the
//! exact kNN + Gaussian-kernel routine in `scx_accel::compute_lisi`.

use std::collections::HashMap;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use scx_accel::LisiConfig;

/// Factorise a Python object column (Categorical / string / numeric) into
/// `(labels: Vec<u32>, n_levels: usize)` via `pandas.factorize(sort=False)`.
fn factorize_obs_column(
    py: Python<'_>,
    col: &Bound<'_, PyAny>,
    col_name: &str,
) -> PyResult<(Vec<u32>, usize)> {
    let pd = py.import("pandas")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("sort", false)?;
    let tup = pd.call_method("factorize", (col,), Some(&kwargs))?;
    let codes = tup.get_item(0)?;
    let uniques = tup.get_item(1)?;

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

/// Compute per-cell Local Inverse Simpson Index.
///
/// LISI measures local category diversity: a value near 1 indicates
/// neighbourhoods dominated by a single label (poor mixing); values
/// approaching the number of categories indicate uniform mixing. For
/// batch-integration QC, run against the batch column before and after
/// `harmony_integrate` to quantify improvement.
///
/// Args:
///     adata: AnnData with embeddings at `adata.obsm[basis]`.
///     key: obs column with categorical labels to evaluate.
///     basis: obsm key holding embeddings (default "X_pca").
///     perplexity: target perplexity for the Gaussian kernel (default 30.0).
///     n_neighbors: number of neighbours to use. `None` (default) uses
///         `3 * perplexity`.
///     approximate_knn: when `True`, use an HNSW approximate kNN instead of
///         the exact O(N²) sweep. Trades small numerical drift (~0.01–0.05 on
///         mean-LISI) for an order-of-magnitude speed-up at N ≳ 100k. Default
///         `False` matches the R `lisi` reference byte-for-byte; the exact path
///         logs a hint to set this `True` once N exceeds the large-N threshold.
///
/// Writes the LISI vector to `adata.obs[f"lisi_{key}"]` and also returns
/// it as a 1-D numpy array of length N.
#[pyfunction]
#[pyo3(signature = (
    adata,
    key,
    *,
    basis = "X_pca",
    perplexity = 30.0,
    n_neighbors = None,
    approximate_knn = false,
))]
pub fn compute_lisi<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    key: &str,
    basis: &str,
    perplexity: f64,
    n_neighbors: Option<usize>,
    approximate_knn: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if perplexity <= 0.0 {
        return Err(PyValueError::new_err("perplexity must be > 0"));
    }

    // Returns the LISI vector *and* writes it to `adata.obs[f"lisi_{key}"]`
    // below, which makes this a write-back op: on a view that `obs` write goes
    // through anndata's copy-on-write and gathers a backed `X`. No var-order
    // guard — this op reads `obsm` and never touches the gene axis.
    super::prepare_target_no_var_guard(py, adata, "compute_lisi")?;

    // --- Embeddings (N x d, f32 row-major) ---
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
    let n_dims = shape[1];
    let embeddings: Vec<f32> = {
        let ro = emb_arr.readonly();
        let sl = ro.as_slice().map_err(|e| {
            PyRuntimeError::new_err(format!("adata.obsm['{basis}'] not C-contiguous: {e}"))
        })?;
        sl.to_vec()
    };
    if !embeddings.iter().all(|v| v.is_finite()) {
        return Err(PyRuntimeError::new_err(format!(
            "adata.obsm['{basis}'] contains NaN or Inf"
        )));
    }

    // --- Labels ---
    let obs = adata.getattr("obs")?;
    let col = obs.get_item(key).map_err(|_| {
        PyRuntimeError::new_err(format!("obs column '{key}' not found in adata.obs"))
    })?;
    let (labels, _n_levels) = factorize_obs_column(py, &col, key)?;
    if labels.len() != n_obs {
        return Err(PyRuntimeError::new_err(format!(
            "obs column '{key}' length {} != n_obs {}",
            labels.len(),
            n_obs
        )));
    }

    let k = n_neighbors.unwrap_or_else(|| (perplexity * 3.0).ceil() as usize);
    let config = LisiConfig {
        perplexity,
        n_neighbors: k,
        approximate_knn,
        ..Default::default()
    };

    let result = py
        .detach(|| scx_accel::compute_lisi(&embeddings, n_obs, n_dims, &labels, &config))
        .map_err(|e: scx_accel::AccelError| {
            PyRuntimeError::new_err(format!("compute_lisi: {e}"))
        })?;

    // Write to adata.obs[f"lisi_{key}"]. We round-trip through a pandas
    // Series so the column lands with a sensible index rather than being
    // a bare numpy array in AnnData's obs frame.
    let pd = py.import("pandas")?;
    let obs_index = obs.getattr("index")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("index", &obs_index)?;
    let arr_for_series: HashMap<&str, Vec<f64>> = HashMap::new();
    let _ = arr_for_series; // silence unused if compiler is picky
    let series = pd.call_method("Series", (result.lisi.clone(),), Some(&kwargs))?;
    obs.set_item(format!("lisi_{key}"), series)?;

    // Return as numpy array.
    Ok(PyArray1::<f64>::from_vec(py, result.lisi))
}
