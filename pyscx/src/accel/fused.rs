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
#[cfg(feature = "gpu")]
use crate::backed::ScxBackedSparseDataset;
#[cfg(feature = "gpu")]
use crate::lazy_transform::ScxLazyTransformedDataset;
#[cfg(feature = "gpu")]
use pyo3::exceptions::PyRuntimeError;
#[cfg(feature = "gpu")]
use scx_format_io::ShardSource;

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

    // Same reason as `pca`: a presentation-ordered backed `X` has no
    // `ShardSource` representation, and the fused GPU path below would
    // otherwise skip the guard `pca` applies on the delegating path.
    super::prepare_target(py, adata, "pca_neighbors")?;

    // In-VRAM `device="gpu"` fused PCA→kNN routes to a full rapids pipeline
    // (rsc.pp.pca → rsc.pp.neighbors) on an in-memory X. backed/lazy X stays on
    // the native device-resident fused path (>VRAM moat). Gated on the default
    // `use_rep="X_pca"` like the native fused path.
    #[cfg(feature = "gpu")]
    if use_rep == "X_pca" {
        let x_in_memory = {
            let xp = adata.getattr("X")?;
            xp.extract::<PyRef<ScxBackedSparseDataset>>().is_err()
                && xp.extract::<PyRef<ScxLazyTransformedDataset>>().is_err()
        };
        if x_in_memory {
            match super::rapids::decide(py, _device, "pca_neighbors") {
                super::rapids::RapidsDecision::Rapids(gid) => {
                    super::rapids::run_fused(
                        py,
                        adata,
                        gid,
                        n_comps,
                        zero_center,
                        random_state,
                        n_neighbors,
                        use_rep,
                        None,
                    )?;
                    return Ok(());
                }
                super::rapids::RapidsDecision::NoRapidsCpu => {
                    pca_neighbors(
                        py,
                        adata,
                        n_comps,
                        n_neighbors,
                        zero_center,
                        random_state,
                        n_oversamples,
                        n_power_iterations,
                        "cpu",
                        method,
                        qr_method,
                        use_rep,
                        prefer_format,
                    )?;
                    super::rapids::stamp_no_rapids(
                        py,
                        adata,
                        "pca",
                        scx_accel::AccelRoute::CpuCsr,
                    )?;
                    super::rapids::stamp_no_rapids(
                        py,
                        adata,
                        "neighbors",
                        scx_accel::AccelRoute::CpuCsr,
                    )?;
                    return super::rapids::stamp_no_rapids(
                        py,
                        adata,
                        "pca_neighbors",
                        scx_accel::AccelRoute::CpuCsr,
                    );
                }
                super::rapids::RapidsDecision::Native => {}
            }
        }
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
                // The handle's *view*, not the raw reader — otherwise a
                // subset backed `X` feeds PCA every on-disk row/column and
                // the embedding misaligns against `adata.obs` (see `pca`).
                // Uncached for the same reason as `pca`'s GPU arm: the GPU
                // shard source stages through `read_shard`, so cached reads
                // would deep-clone each shard instead of decoding fresh.
                let source = backed.as_shard_source();
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
                    device,
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
                    device,
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
                    device,
                )?;
                true
            } else {
                let csr = crate::convert::owned_csr(py, &x, None)?;
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
                    device,
                )?;
                true
            };

            if ran {
                // Routes (pca / neighbors / pca_neighbors) were stamped inside
                // `run_fused_gpu` with the Task 2.5 PCA metadata.
                return Ok(());
            }
        } else {
            // GPU requested but cuVS missing — warn, then fall through to the
            // sequential path (GPU PCA + CPU HNSW neighbors).
            let warnings = crate::pyimport::import_module(py, "warnings")?;
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
        // Task 2.5 tuning knobs: the fused entry points keep PCA defaults.
        false,     // allow_tf32
        "default", // spmm_policy
        None,      // memory_budget (default PCA cache ceiling)
        // mask_var: this CPU/rapids fallback delegates to pca(), so mask_var=None
        // auto-consumes adata.var["highly_variable"] if present (scanpy semantics).
        // KNOWN LIMITATION: the native device-resident GPU fused path
        // (run_fused_gpu) does NOT mask and analyzes all genes, so the fused
        // PCA gene set is route-dependent when highly_variable is set. For a
        // deterministic HVG-masked pipeline, run pca(mask_var=...) then
        // neighbors()/umap() separately. Tracked for unification.
        None,
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
    super::route::copy_accel_route(py, adata, "neighbors", "pca_neighbors")?;

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

    // See `pca_neighbors`.
    super::prepare_target(py, adata, "pca_neighbors_umap")?;

    // In-VRAM `device="gpu"` fused PCA→kNN→UMAP routes to a full rapids pipeline
    // (rsc.pp.pca → rsc.pp.neighbors → rsc.tl.umap) on an in-memory X.
    // backed/lazy X stays on the native device-resident fused path.
    #[cfg(feature = "gpu")]
    if use_rep == "X_pca" {
        let x_in_memory = {
            let xp = adata.getattr("X")?;
            xp.extract::<PyRef<ScxBackedSparseDataset>>().is_err()
                && xp.extract::<PyRef<ScxLazyTransformedDataset>>().is_err()
        };
        if x_in_memory {
            match super::rapids::decide(py, _device, "pca_neighbors_umap") {
                super::rapids::RapidsDecision::Rapids(gid) => {
                    super::rapids::run_fused(
                        py,
                        adata,
                        gid,
                        n_comps,
                        zero_center,
                        random_state,
                        n_neighbors,
                        use_rep,
                        Some((
                            n_components,
                            min_dist,
                            spread,
                            negative_sample_rate,
                            n_epochs,
                            umap_learning_rate,
                        )),
                    )?;
                    return Ok(());
                }
                super::rapids::RapidsDecision::NoRapidsCpu => {
                    pca_neighbors_umap(
                        py,
                        adata,
                        n_comps,
                        n_neighbors,
                        n_components,
                        n_epochs,
                        min_dist,
                        spread,
                        negative_sample_rate,
                        umap_learning_rate,
                        zero_center,
                        random_state,
                        n_oversamples,
                        n_power_iterations,
                        "cpu",
                        method,
                        qr_method,
                        use_rep,
                        prefer_format,
                    )?;
                    for op in ["pca", "neighbors", "umap", "pca_neighbors_umap"] {
                        super::rapids::stamp_no_rapids(
                            py,
                            adata,
                            op,
                            scx_accel::AccelRoute::CpuCsr,
                        )?;
                    }
                    return Ok(());
                }
                super::rapids::RapidsDecision::Native => {}
            }
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
        // Task 2.5 tuning knobs: the fused entry points keep PCA defaults.
        false,     // allow_tf32
        "default", // spmm_policy
        None,      // memory_budget (default PCA cache ceiling)
        // mask_var: this CPU/rapids fallback delegates to pca(), so mask_var=None
        // auto-consumes adata.var["highly_variable"] if present (scanpy semantics).
        // KNOWN LIMITATION: the native device-resident GPU fused path
        // (run_fused_gpu) does NOT mask and analyzes all genes, so the fused
        // PCA gene set is route-dependent when highly_variable is set. For a
        // deterministic HVG-masked pipeline, run pca(mask_var=...) then
        // neighbors()/umap() separately. Tracked for unification.
        None,
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
    super::route::copy_accel_route(py, adata, "umap", "pca_neighbors_umap")?;

    Ok(())
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
    device: &str,
) -> PyResult<()> {
    let n_vars = source.n_vars();
    // The covariance GPU PCA path was removed; `resolve_gpu_method` validates
    // the method string and always resolves the native fused path to randomized.
    resolve_gpu_method(method, n_vars)?;

    let (pca_res, knn_res) = fused_dispatch_unwind_safe(
        device_id,
        source,
        n_comps,
        n_oversamples,
        n_power_iterations,
        zero_center,
        random_state,
        qr,
        n_neighbors,
    )
    .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?;

    write_pca_to_adata(py, adata, &pca_res, "scx-gpu-cusparse", None)?;
    super::neighbors::write_neighbors_to_adata(py, adata, knn_res, n_neighbors, use_rep, "cagra")?;
    // Stamp the device-resident route on pca / neighbors / pca_neighbors,
    // carrying the Task 2.5 metadata (finding 5): the math-mode / SpMM-policy
    // knobs (the randomized path consumes both — covariance, which recorded
    // neither, was removed in Phase 3.2).
    stamp_fused_route(
        py,
        adata,
        device,
        &["pca", "neighbors", "pca_neighbors"],
        pca_res.resident_csr,
    )
}

/// Stamp the fused `gpu_device_resident` route on each `op`, attaching the
/// Task 2.5 PCA metadata. The fused entry points use the default tuning
/// (strict-fp32 + heuristic SpMM); `math_mode` / `spmm_policy` are always
/// recorded now that the (SpMM-free) covariance path is gone.
#[cfg(feature = "gpu")]
fn stamp_fused_route(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    device: &str,
    ops: &[&str],
    resident_csr: Option<bool>,
) -> PyResult<()> {
    let mut info = super::route::simple_exec_info(
        device,
        true,
        scx_accel::AccelRoute::GpuDeviceResident,
        scx_accel::AccelRoute::CpuCsr,
    );
    if info.route.is_gpu() {
        info.resident_csr = resident_csr;
        info.math_mode = Some(scx_accel::GpuMathMode::default().as_str());
        info.spmm_policy = Some(scx_accel::SpmmAlgPolicy::default().as_str());
    }
    for op in ops {
        super::route::write_accel_route(py, adata, op, &info)?;
    }
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
            n_neighbors,
            // Task 2.5: the fused pyscx entry points do not expose the math-mode /
            // SpMM-policy knobs; use the strict-fp32 + heuristic-default tuning
            // (residency + capture still engage automatically).
            scx_accel::GpuPcaTuning::default(),
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
