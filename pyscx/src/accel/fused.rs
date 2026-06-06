//! Fused device-resident accelerator entry points.
//!
//! [`pca_neighbors`] runs GPU PCA then GPU CAGRA kNN in a single call, keeping
//! the PCA embedding resident on the GPU between the two stages (V3 plan
//! Phase 2.3). When the fully device-resident path is unavailable (no GPU, or a
//! missing cuSPARSE 12.5 / cuVS library), it transparently falls back to the
//! sequential [`super::pca::pca`] + [`super::neighbors::neighbors`] calls, which
//! preserve their own per-op routing and warnings.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::gpu::resolve_device;

#[cfg(feature = "gpu")]
use super::pca::{
    emit_cusparse_abi_warning, parse_qr_method, resolve_gpu_method, try_extract_borrowed_csr,
    write_pca_to_adata, BorrowedCsrSource, ScxCsrSource,
};
#[cfg(feature = "gpu")]
use super::umap::{write_umap_backend, write_umap_to_adata};
#[cfg(feature = "gpu")]
use super::util::extract_materialized_csr;
#[cfg(feature = "gpu")]
use crate::backed::ScxBackedSparseDataset;
#[cfg(feature = "gpu")]
use crate::lazy_transform::ScxLazyTransformedDataset;
#[cfg(feature = "gpu")]
use pyo3::exceptions::PyRuntimeError;
#[cfg(feature = "gpu")]
use scx_format::ShardSource;

/// GPU PCA → kNN in one call, with the embedding kept device-resident.
///
/// Equivalent to calling `pyscx.accel.pca(...)` followed by
/// `pyscx.accel.neighbors(...)`, but on a GPU host with cuSPARSE 12.5+ and cuVS
/// available the PCA embedding is fed straight into CAGRA without the
/// GPU → host → GPU round-trip the two separate calls incur. Writes the same
/// AnnData slots as the individual ops (`obsm["X_pca"]`, `varm["PCs"]`,
/// `uns["pca"]`, `obsp["distances"]`, `obsp["connectivities"]`,
/// `uns["neighbors"]`).
///
/// When the fully fused path runs, `adata.uns["scx_accel"]["pca"]`,
/// `["neighbors"]`, and `["pca_neighbors"]` are all stamped with route
/// `"gpu_device_resident"`. Otherwise the sequential fallback stamps each op's
/// usual route and records the (non-device-resident) summary on
/// `["pca_neighbors"]`.
///
/// Args:
///     adata: AnnData with `X` (backed SCX, lazy transform, or scipy/dense).
///     n_comps: number of principal components (default 50).
///     n_neighbors: number of nearest neighbors (default 15).
///     zero_center: mean-center before PCA (default True).
///     random_state: random seed (default 0).
///     n_oversamples / n_power_iterations: randomized-PCA accuracy knobs.
///     device: "auto" (default), "cpu", "gpu", or "gpu:N".
///     method: "auto" (default), "covariance", or "randomized".
///     qr_method: "householder" (default) or "cholesky".
///     use_rep: obsm key the neighbors step reads (default "X_pca"). The fused
///         device-resident path runs kNN on the freshly-computed PCA embedding,
///         so a non-default `use_rep` always takes the sequential path (which
///         reads `obsm[use_rep]`).
///     prefer_format: only "csr" is supported (PCA is row-major).
#[pyfunction]
#[pyo3(signature = (adata, n_comps=50, n_neighbors=15, zero_center=true, random_state=0, n_oversamples=10, n_power_iterations=2, device="auto", method="auto", qr_method="householder", use_rep="X_pca", prefer_format="csr"))]
#[allow(clippy::too_many_arguments)]
pub fn pca_neighbors(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    n_comps: usize,
    n_neighbors: usize,
    zero_center: bool,
    random_state: u64,
    n_oversamples: usize,
    n_power_iterations: usize,
    device: &str,
    method: &str,
    qr_method: &str,
    use_rep: &str,
    prefer_format: &str,
) -> PyResult<()> {
    let _device = resolve_device(device)?;

    // Validate user args regardless of device (mirrors pca()).
    if !matches!(method, "auto" | "covariance" | "randomized") {
        return Err(PyValueError::new_err(format!(
            "Invalid method={method:?}; expected 'auto', 'covariance', or 'randomized'"
        )));
    }
    if !matches!(qr_method, "householder" | "cholesky") {
        return Err(PyValueError::new_err(format!(
            "Invalid qr_method={qr_method:?}; expected 'householder' or 'cholesky'"
        )));
    }
    if prefer_format != "csr" {
        return Err(PyValueError::new_err(format!(
            "pyscx.accel.pca_neighbors only supports prefer_format='csr' (got \
             {prefer_format:?}); PCA's SpMM path is row-major and CSC is not implemented."
        )));
    }

    // ---- Fully fused device-resident GPU path ----
    // Gated on `use_rep == "X_pca"`: the fused path always feeds kNN the
    // freshly-computed PCA embedding, whereas the sequential `neighbors` reads
    // `obsm[use_rep]`. A non-default `use_rep` therefore must take the
    // sequential path (which honors `obsm[use_rep]`) — otherwise we'd compute
    // the graph from the new PCA embedding while claiming we used `use_rep`. So
    // a non-default `use_rep` falls straight through to the sequential delegate
    // below (no fused-specific cuSPARSE/cuVS warnings — `pca()` emits its own).
    #[cfg(feature = "gpu")]
    if let Some(device_id) = match (_device.gpu_id(), use_rep == "X_pca") {
        (Some(id), true) if scx_accel::cusparse_modern_abi_available() => Some(id),
        (Some(_), true) => {
            // libcusparse too old — warn and let the fallback (which may still
            // use GPU PCA via its own probe) handle it.
            emit_cusparse_abi_warning(py, device)?;
            None
        }
        _ => None,
    } {
        if scx_accel::cuvs_available() {
            let qr = parse_qr_method(qr_method)?;
            let x = adata.getattr("X")?;

            let ran = if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
                let reader = &*backed.backed;
                run_fused_gpu(
                    py,
                    adata,
                    device_id,
                    reader,
                    n_comps,
                    n_oversamples,
                    n_power_iterations,
                    zero_center,
                    random_state,
                    qr,
                    method,
                    n_neighbors,
                    use_rep,
                )?;
                true
            } else if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
                let source = lazy.as_shard_source();
                run_fused_gpu(
                    py,
                    adata,
                    device_id,
                    &source,
                    n_comps,
                    n_oversamples,
                    n_power_iterations,
                    zero_center,
                    random_state,
                    qr,
                    method,
                    n_neighbors,
                    use_rep,
                )?;
                true
            } else if let Some((slices, shape)) = try_extract_borrowed_csr(py, &x)? {
                let source = BorrowedCsrSource {
                    indptr: slices.indptr(),
                    indices: slices.indices(),
                    data: slices.data(),
                    shape,
                };
                run_fused_gpu(
                    py,
                    adata,
                    device_id,
                    &source,
                    n_comps,
                    n_oversamples,
                    n_power_iterations,
                    zero_center,
                    random_state,
                    qr,
                    method,
                    n_neighbors,
                    use_rep,
                )?;
                true
            } else {
                let csr = extract_materialized_csr(py, &x)?;
                let source = ScxCsrSource { csr: &csr };
                run_fused_gpu(
                    py,
                    adata,
                    device_id,
                    &source,
                    n_comps,
                    n_oversamples,
                    n_power_iterations,
                    zero_center,
                    random_state,
                    qr,
                    method,
                    n_neighbors,
                    use_rep,
                )?;
                true
            };

            if ran {
                // Stamp the device-resident route on both ops + the fused summary.
                let info = super::route::simple_exec_info(
                    device,
                    true,
                    scx_accel::AccelRoute::GpuDeviceResident,
                    scx_accel::AccelRoute::CpuCsr,
                );
                super::route::write_accel_route(py, adata, "pca", &info)?;
                super::route::write_accel_route(py, adata, "neighbors", &info)?;
                super::route::write_accel_route(py, adata, "pca_neighbors", &info)?;
                return Ok(());
            }
        } else {
            // GPU requested but cuVS missing — warn, then fall through to the
            // sequential path (GPU PCA + CPU HNSW neighbors).
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                (
                    format!(
                        "pca_neighbors(device={device:?}): cuVS not found — the fused \
                         device-resident path is unavailable; falling back to sequential \
                         pca + neighbors (CPU HNSW). Install cuVS: conda install -c rapidsai \
                         -c conda-forge libcuvs"
                    ),
                    py.get_type::<pyo3::exceptions::PyUserWarning>(),
                ),
            )?;
        }
    }

    // ---- Sequential fallback (no GPU / missing cuSPARSE 12.5 / missing cuVS) ----
    // Delegate to the standalone ops; each stamps its own route + warnings.
    super::pca::pca(
        py,
        adata,
        n_comps,
        zero_center,
        random_state,
        n_oversamples,
        n_power_iterations,
        device,
        method,
        qr_method,
        prefer_format,
    )?;
    super::neighbors::neighbors(
        py,
        adata,
        n_neighbors,
        use_rep,
        random_state,
        200,
        200,
        device,
    )?;

    // Summary route on pca_neighbors: the fused path returned above when it
    // ran, so we got here only on the sequential fallback. Mirror the actual
    // route the `neighbors` stage recorded (the stage whose GPU-vs-CPU outcome
    // determines whether fusion happened) rather than synthesizing a summary —
    // this stays accurate when GPU PCA ran but kNN fell back to CPU HNSW (cuVS
    // missing), where a synthesized `cpu_csr` would wrongly imply a full-CPU run.
    super::route::copy_accel_route(adata, "neighbors", "pca_neighbors")?;

    Ok(())
}

/// GPU PCA → kNN → UMAP in one call, with the embedding **and** the fuzzy
/// connectivity graph kept device-resident across the whole chain (V3 Phase 2.4).
///
/// Equivalent to `pyscx.accel.pca(...)` → `neighbors(...)` → `umap(...)`, but on
/// a GPU host with cuSPARSE 12.5+ and cuVS the fuzzy simplicial set is built on
/// the GPU (replacing the host symmetrization) and feeds UMAP directly, so the
/// connectivities are downloaded once (for `obsp` + spectral init) instead of
/// symmetrized on the CPU and re-uploaded for SGD. Writes the same AnnData slots
/// as the three individual ops, including `obsm["X_umap"]`.
///
/// When the fully fused path runs, `adata.uns["scx_accel"]` entries for `pca`,
/// `neighbors`, `umap`, and the `pca_neighbors_umap` summary are all stamped
/// with route `"gpu_device_resident"`. Otherwise the sequential fallback stamps
/// each op's usual route and mirrors the `umap` route onto the summary.
///
/// Args mirror [`pca_neighbors`] plus the UMAP knobs: `n_components` (UMAP output
/// dims, default 2), `n_epochs` (default 200), `min_dist` (default 0.1), `spread`
/// (default 1.0), `negative_sample_rate` (default 5), `umap_learning_rate`
/// (default 1.0).
#[pyfunction]
#[pyo3(signature = (adata, n_comps=50, n_neighbors=15, n_components=2, n_epochs=200, min_dist=0.1, spread=1.0, negative_sample_rate=5, umap_learning_rate=1.0, zero_center=true, random_state=0, n_oversamples=10, n_power_iterations=2, device="auto", method="auto", qr_method="householder", use_rep="X_pca", prefer_format="csr"))]
#[allow(clippy::too_many_arguments)]
pub fn pca_neighbors_umap(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    n_comps: usize,
    n_neighbors: usize,
    n_components: usize,
    n_epochs: usize,
    min_dist: f64,
    spread: f64,
    negative_sample_rate: usize,
    umap_learning_rate: f64,
    zero_center: bool,
    random_state: u64,
    n_oversamples: usize,
    n_power_iterations: usize,
    device: &str,
    method: &str,
    qr_method: &str,
    use_rep: &str,
    prefer_format: &str,
) -> PyResult<()> {
    let _device = resolve_device(device)?;

    if !matches!(method, "auto" | "covariance" | "randomized") {
        return Err(PyValueError::new_err(format!(
            "Invalid method={method:?}; expected 'auto', 'covariance', or 'randomized'"
        )));
    }
    if !matches!(qr_method, "householder" | "cholesky") {
        return Err(PyValueError::new_err(format!(
            "Invalid qr_method={qr_method:?}; expected 'householder' or 'cholesky'"
        )));
    }
    if prefer_format != "csr" {
        return Err(PyValueError::new_err(format!(
            "pyscx.accel.pca_neighbors_umap only supports prefer_format='csr' (got \
             {prefer_format:?}); PCA's SpMM path is row-major and CSC is not implemented."
        )));
    }

    // ---- Fully fused device-resident GPU path (gated on use_rep == "X_pca"). ----
    #[cfg(feature = "gpu")]
    // The fused fuzzy-graph kernel caps `n_neighbors` at `FUZZY_MAX_K` (256);
    // for larger `k` we fall through to the sequential path (which has no such
    // cap) rather than hard-erroring. Route the size-cap case through `_ => None`
    // so it does not also emit a spurious cuSPARSE-ABI warning.
    if let Some(device_id) = match (_device.gpu_id(), use_rep == "X_pca") {
        (Some(id), true)
            if n_neighbors <= scx_accel::FUZZY_MAX_K
                && scx_accel::cusparse_modern_abi_available() =>
        {
            Some(id)
        }
        (Some(_), true) if n_neighbors <= scx_accel::FUZZY_MAX_K => {
            emit_cusparse_abi_warning(py, device)?;
            None
        }
        _ => None,
    } {
        if scx_accel::cuvs_available() {
            let qr = parse_qr_method(qr_method)?;
            let x = adata.getattr("X")?;

            let umap_params = UmapFusedParams {
                n_components,
                n_epochs,
                min_dist,
                spread,
                negative_sample_rate,
                umap_learning_rate,
            };

            let ran = if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
                let reader = &*backed.backed;
                run_fused_gpu_umap(
                    py,
                    adata,
                    device_id,
                    reader,
                    n_comps,
                    n_oversamples,
                    n_power_iterations,
                    zero_center,
                    random_state,
                    qr,
                    method,
                    n_neighbors,
                    use_rep,
                    &umap_params,
                )?;
                true
            } else if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
                let source = lazy.as_shard_source();
                run_fused_gpu_umap(
                    py,
                    adata,
                    device_id,
                    &source,
                    n_comps,
                    n_oversamples,
                    n_power_iterations,
                    zero_center,
                    random_state,
                    qr,
                    method,
                    n_neighbors,
                    use_rep,
                    &umap_params,
                )?;
                true
            } else if let Some((slices, shape)) = try_extract_borrowed_csr(py, &x)? {
                let source = BorrowedCsrSource {
                    indptr: slices.indptr(),
                    indices: slices.indices(),
                    data: slices.data(),
                    shape,
                };
                run_fused_gpu_umap(
                    py,
                    adata,
                    device_id,
                    &source,
                    n_comps,
                    n_oversamples,
                    n_power_iterations,
                    zero_center,
                    random_state,
                    qr,
                    method,
                    n_neighbors,
                    use_rep,
                    &umap_params,
                )?;
                true
            } else {
                let csr = extract_materialized_csr(py, &x)?;
                let source = ScxCsrSource { csr: &csr };
                run_fused_gpu_umap(
                    py,
                    adata,
                    device_id,
                    &source,
                    n_comps,
                    n_oversamples,
                    n_power_iterations,
                    zero_center,
                    random_state,
                    qr,
                    method,
                    n_neighbors,
                    use_rep,
                    &umap_params,
                )?;
                true
            };

            if ran {
                let info = super::route::simple_exec_info(
                    device,
                    true,
                    scx_accel::AccelRoute::GpuDeviceResident,
                    scx_accel::AccelRoute::CpuCsr,
                );
                super::route::write_accel_route(py, adata, "pca", &info)?;
                super::route::write_accel_route(py, adata, "neighbors", &info)?;
                super::route::write_accel_route(py, adata, "umap", &info)?;
                super::route::write_accel_route(py, adata, "pca_neighbors_umap", &info)?;
                return Ok(());
            }
        } else {
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                (
                    format!(
                        "pca_neighbors_umap(device={device:?}): cuVS not found — the fused \
                         device-resident path is unavailable; falling back to sequential \
                         pca + neighbors + umap. Install cuVS: conda install -c rapidsai \
                         -c conda-forge libcuvs"
                    ),
                    py.get_type::<pyo3::exceptions::PyUserWarning>(),
                ),
            )?;
        }
    }

    // ---- Sequential fallback (no GPU / missing cuSPARSE 12.5 / missing cuVS). ----
    super::pca::pca(
        py,
        adata,
        n_comps,
        zero_center,
        random_state,
        n_oversamples,
        n_power_iterations,
        device,
        method,
        qr_method,
        prefer_format,
    )?;
    super::neighbors::neighbors(
        py,
        adata,
        n_neighbors,
        use_rep,
        random_state,
        200,
        200,
        device,
    )?;
    super::umap::umap(
        py,
        adata,
        n_components,
        n_epochs,
        min_dist,
        spread,
        negative_sample_rate,
        umap_learning_rate,
        random_state,
        device,
    )?;

    // Summary route: mirror the umap stage's recorded route (the last stage,
    // whose GPU-vs-CPU outcome determines whether the chain stayed on device).
    super::route::copy_accel_route(adata, "umap", "pca_neighbors_umap")?;

    Ok(())
}

/// UMAP knobs threaded through the fused GPU path.
#[cfg(feature = "gpu")]
struct UmapFusedParams {
    n_components: usize,
    n_epochs: usize,
    min_dist: f64,
    spread: f64,
    negative_sample_rate: usize,
    umap_learning_rate: f64,
}

/// Run the fused GPU PCA → kNN dispatch on a single `ShardSource` and write the
/// results into `adata`. Route stamping is done by the caller.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn run_fused_gpu<S: ShardSource + Sync>(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    device_id: usize,
    source: &S,
    n_comps: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    random_state: u64,
    qr: scx_accel::QrMethod,
    method: &str,
    n_neighbors: usize,
    use_rep: &str,
) -> PyResult<()> {
    let n_vars = source.n_vars();
    let use_covariance = resolve_gpu_method(method, n_vars)? == "covariance";

    let (pca_res, knn_res) = fused_dispatch_unwind_safe(
        device_id,
        source,
        n_comps,
        n_oversamples,
        n_power_iterations,
        zero_center,
        random_state,
        qr,
        use_covariance,
        n_neighbors,
    )
    .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?;

    write_pca_to_adata(py, adata, &pca_res, "scx-gpu-cusparse")?;
    super::neighbors::write_neighbors_to_adata(py, adata, &knn_res, n_neighbors, use_rep, "cagra")?;
    Ok(())
}

/// Catch any cudarc dlsym / FFI panic escaping `scx_accel::pca_then_knn_gpu`
/// (PCA's cuSPARSE path or cuVS CAGRA) and translate it to a normal
/// `AccelError`, so users never see a raw `pyo3_runtime.PanicException`.
/// Mirrors `pca::gpu_pca_dispatch_unwind_safe`.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn fused_dispatch_unwind_safe<S: ShardSource + Sync>(
    device_id: usize,
    source: &S,
    n_comps: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    random_state: u64,
    qr: scx_accel::QrMethod,
    use_covariance: bool,
    n_neighbors: usize,
) -> Result<(scx_accel::PcaResult, scx_accel::KnnResult), scx_accel::AccelError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        scx_accel::pca_then_knn_gpu(
            device_id,
            source,
            n_comps,
            n_oversamples,
            n_power_iterations,
            zero_center,
            random_state,
            qr,
            use_covariance,
            n_neighbors,
        )
    }))
    .unwrap_or_else(|panic_payload| {
        let msg = panic_payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| {
                panic_payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
            })
            .unwrap_or_else(|| "unknown panic payload".to_string());
        Err(scx_accel::AccelError::LinAlg(format!(
            "fused GPU PCA→kNN panicked: {msg}. If this mentions libcusparse / \
             cusparse* undefined symbol, set \
             LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH — see docs/gpu-setup.md."
        )))
    })
}

/// Run the fused GPU PCA → kNN → UMAP dispatch on a single `ShardSource` and
/// write all results (`X_pca`, `obsp`, `uns["neighbors"]`, `X_umap`) into
/// `adata`. Route stamping is done by the caller.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn run_fused_gpu_umap<S: ShardSource + Sync>(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    device_id: usize,
    source: &S,
    n_comps: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    random_state: u64,
    qr: scx_accel::QrMethod,
    method: &str,
    n_neighbors: usize,
    use_rep: &str,
    umap: &UmapFusedParams,
) -> PyResult<()> {
    let n_vars = source.n_vars();
    let use_covariance = resolve_gpu_method(method, n_vars)? == "covariance";

    let (pca_res, knn_res, umap_res) = fused_umap_dispatch_unwind_safe(
        device_id,
        source,
        n_comps,
        n_oversamples,
        n_power_iterations,
        zero_center,
        random_state,
        qr,
        use_covariance,
        n_neighbors,
        umap,
    )
    .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?;

    write_pca_to_adata(py, adata, &pca_res, "scx-gpu-cusparse")?;
    super::neighbors::write_neighbors_to_adata(py, adata, &knn_res, n_neighbors, use_rep, "cagra")?;
    write_umap_to_adata(py, adata, &umap_res)?;
    write_umap_backend(py, adata, "scx-gpu-cuda")?;
    Ok(())
}

/// `catch_unwind` wrapper around [`scx_accel::pca_then_knn_umap_gpu`], mirroring
/// [`fused_dispatch_unwind_safe`].
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn fused_umap_dispatch_unwind_safe<S: ShardSource + Sync>(
    device_id: usize,
    source: &S,
    n_comps: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    random_state: u64,
    qr: scx_accel::QrMethod,
    use_covariance: bool,
    n_neighbors: usize,
    umap: &UmapFusedParams,
) -> Result<
    (
        scx_accel::PcaResult,
        scx_accel::KnnResult,
        scx_accel::UmapResult,
    ),
    scx_accel::AccelError,
> {
    // `random_state` intentionally seeds both the PCA and UMAP stages — the
    // fused `pca_neighbors_umap` entry exposes a single `random_state` kwarg
    // (matching the standalone path), so independent PCA/UMAP seeds are not
    // surfaced here even though the underlying fn takes them separately.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        scx_accel::pca_then_knn_umap_gpu(
            device_id,
            source,
            n_comps,
            n_oversamples,
            n_power_iterations,
            zero_center,
            random_state, // PCA seed
            qr,
            use_covariance,
            n_neighbors,
            umap.n_components,
            umap.n_epochs,
            umap.min_dist,
            umap.spread,
            umap.negative_sample_rate,
            umap.umap_learning_rate,
            random_state, // UMAP seed (same value by design)
        )
    }))
    .unwrap_or_else(|panic_payload| {
        let msg = panic_payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| {
                panic_payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
            })
            .unwrap_or_else(|| "unknown panic payload".to_string());
        Err(scx_accel::AccelError::LinAlg(format!(
            "fused GPU PCA→kNN→UMAP panicked: {msg}. If this mentions libcusparse / \
             cusparse* undefined symbol, set \
             LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH — see docs/gpu-setup.md."
        )))
    })
}
