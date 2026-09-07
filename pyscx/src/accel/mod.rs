//! Python bindings for SCX accelerators (PCA, kNN, UMAP, Leiden, DE, etc.).
//!
//! This module is split into submodules by functional domain:
//!
//! - `gpu` — GPU device info, memory estimation, device resolution
//! - `pca` — Randomized/covariance PCA (streaming + in-memory)
//! - `neighbors` — kNN graph construction (HNSW + cuVS CAGRA)
//! - `umap` — UMAP embedding (CPU SGD + GPU CUDA + cuML fallback)
//! - `de` — Wilcoxon rank-sum DE, stratified DE, cell-eval bridge
//! - `pseudobulk` — Pseudobulk differential expression via pydeseq2
//! - `leiden` — Leiden community detection (Rust-native, cuGraph, leidenalg)
//! - `preprocessing` — normalize_total, log1p, calculate_qc_metrics
//! - `filtering` — filter_cells, filter_genes, subset_obs (non-materializing)
//! - `hvg` — Highly variable gene selection (seurat_v3 + seurat flavors)
//! - `harmony` — Harmony2 batch integration (soft k-means + ridge correction)
//! - `lisi` — Local Inverse Simpson Index (exact-kNN + Gaussian bandwidth)
//! - `eval_metrics` — Perturbation evaluation metrics (pseudobulk means,
//!   energy distance, discrimination score, knockdown, clustering agreement)
//! - `util` — Shared CSR extraction helpers

pub mod col_aggs;
pub mod de;
pub mod eval_metrics;
pub mod filtering;
pub mod fused;
pub mod gpu;
pub mod gpu_handoff;
pub mod harmony;
pub mod hvg;
pub mod leiden;
pub mod lisi;
pub mod nb_glm;
pub mod neighbors;
pub mod pca;
pub mod pflog;
pub mod preprocessing;
pub mod profile;
pub mod pseudobulk;
#[cfg(feature = "gpu")]
pub mod rapids;
pub mod route;
pub mod score_genes;
pub mod umap;
pub mod util;

use pyo3::prelude::*;

/// Reject accel ops on a backed `X` opened with `preserve_var_order=True`.
///
/// A request-ordered gene axis lives only on `ScxBackedSparseDataset` (as
/// `col_presentation`), installed by `to_anndata(preserve_var_order=True)`,
/// by `adata[:, idx]` with a non-ascending `idx`, and — since 0.17 — by a
/// handle-level reorder such as `adata.X = adata.X[:, [7, 2, 11]]` or
/// `X[:, ::-1]`. Accel ops that build a `ScxLazyTransformedDataset` or a
/// `ShardSource` from the dataset's *sorted* `col_projection` would decode
/// columns in sorted order while `adata.var` stays in request order — a silent
/// X/var misalignment (or, for name-resolving ops like `score_genes`, the wrong
/// physical columns). Until the permutation is propagated into those paths,
/// refuse loudly rather than return wrong data.
///
/// No-op when `X` is not a presentation-ordered backed dataset.
///
/// Only the backed type is inspected, and deliberately so:
/// `ScxLazyTransformedDataset` has no `col_presentation` field at all, and its
/// `set_col_projection` always sorts and dedups, so a lazy `X`'s visible axis is
/// sorted by construction. (`normalize_total` / `log1p` additionally reject a
/// presentation-ordered backed `X` before wrapping it, so a lazy dataset can
/// never inherit one.) Callers that rely on sorted visible order — notably the
/// `calculate_qc_metrics` bitmask, which indexes `adata.var` positions directly
/// — are safe on both types because of this, not because both are checked.
pub(crate) fn reject_preserve_var_order(adata: &Bound<'_, PyAny>, op: &str) -> PyResult<()> {
    let Ok(x) = adata.getattr("X") else {
        return Ok(());
    };
    reject_presentation_ordered_source(&x, op)
}

/// Refuse a presentation-ordered gene axis on **the matrix an op will actually
/// read**, which is not always `adata.X`.
///
/// Two ways the `adata.X`-only check misses it, both of which returned silently
/// mislabelled genes rather than an error:
///
/// * a backed *layer* handle carries its own permutation, and
///   `cast::<ScxBackedSparseDataset>()` does not match `ScxBackedLayerDataset`;
/// * an op resolving `layer=` reads a matrix `adata.X` says nothing about — so
///   materialising `X` (`adata.X = adata.X.to_memory()`) disarmed the guard
///   while the layer stayed presentation-ordered.
///
/// The streaming kernels emit the *sorted* projection, so a request-ordered
/// `adata.var` would be labelled with the wrong genes: asking for the second
/// gene of `var_names=["g2", "g0"]` returned `g0`'s values under `g2`'s name.
/// Every op that resolves a source calls this on the resolved matrix.
pub(crate) fn reject_presentation_ordered_source(x: &Bound<'_, PyAny>, op: &str) -> PyResult<()> {
    let ordered = if let Ok(backed) = x.cast::<crate::backed::ScxBackedSparseDataset>() {
        backed.borrow().col_presentation_arc().is_some()
    } else if let Ok(layer) = x.cast::<crate::backed::ScxBackedLayerDataset>() {
        layer.borrow().inner.col_presentation_arc().is_some()
    } else {
        false
    };
    if ordered {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{op} is not supported on a backed matrix whose gene axis is in a \
             caller-requested order — opened with preserve_var_order=True, or \
             reordered through adata[:, idx] / X[:, [7, 2, 11]] / X[:, ::-1] \
             (the streaming kernels decode columns in sorted on-disk order, \
             which would misalign the result against adata.var). This covers a \
             named `layer=` and a layer handle assigned to X, not just X \
             itself. Run {op} before reordering, select with a sorted index or \
             a boolean mask, or materialise first with \
             `adata.X = adata.X.to_memory()`."
        )));
    }
    Ok(())
}

/// The shard-source inputs of a backed dataset, however it reached us.
///
/// `adata.X` on a backed file is `crate::backed::ScxBackedSparseDataset`; `adata.layers[name]`
/// is `crate::backed::ScxBackedLayerDataset`, a distinct `#[pyclass]` wrapping one. Every
/// accelerator that dispatched with a bare `cast::<crate::backed::ScxBackedSparseDataset>()`
/// therefore missed a layer handle and fell through to `owned_csr`, where
/// `scipy.sparse.csr_matrix(<handle>)` raises "unrecognized csr_matrix
/// constructor input" — so `layer=` was broken on exactly the files it exists
/// for, and every test for it used an in-memory scipy AnnData.
///
/// Returning the parts rather than a borrow keeps the `PyRef` guard local: the
/// layer's `inner` lives behind a `Ref`, so callers cannot hold a reference to
/// it across `build_shard_source`.
pub(crate) struct BackedShardParts {
    pub(crate) reader: std::sync::Arc<scx_format_io::BackedCsrReader>,
    pub(crate) n_obs: usize,
    pub(crate) n_vars: usize,
    pub(crate) kept: Option<std::sync::Arc<Vec<u64>>>,
    pub(crate) col_proj: Option<std::sync::Arc<Vec<u32>>>,
}

impl BackedShardParts {
    fn of_dataset(ds: &crate::backed::ScxBackedSparseDataset) -> Self {
        Self {
            reader: std::sync::Arc::clone(&ds.backed),
            n_obs: ds.shape_val.0,
            n_vars: ds.shape_val.1,
            kept: ds.kept_to_global.clone(),
            col_proj: ds.col_projection_arc(),
        }
    }
}

/// Extract [`BackedShardParts`] from `adata.X` **or** a backed layer handle.
///
/// `None` means "not a backed SCX matrix" — the caller falls on to its lazy
/// and in-memory arms as before.
pub(crate) fn backed_shard_parts(x: &Bound<'_, PyAny>) -> Option<BackedShardParts> {
    if let Ok(ds) = x.cast::<crate::backed::ScxBackedSparseDataset>() {
        return Some(BackedShardParts::of_dataset(&ds.borrow()));
    }
    if let Ok(layer) = x.cast::<crate::backed::ScxBackedLayerDataset>() {
        return Some(BackedShardParts::of_dataset(&layer.borrow().inner));
    }
    None
}

/// The prologue every `pyscx.accel.*` op that writes back to `adata` runs first.
///
/// Two jobs, in this order:
/// 1. [`reject_preserve_var_order`] — a guard must refuse *before* the object
///    is touched, so it comes first.
/// 2. [`crate::axis_align::devirtualize_scx_view`] — rebuild an anndata view
///    over a backed/lazy `X` as an actual `AnnData`, keeping `X` lazy.
///
/// # Ordering invariant
///
/// This must run **before the op's first mutation of `adata`** — `uns`
/// (including the `uns["scx_accel"]` route stamp), `obs`, `var`, `obsm`, or
/// `X`. On a view, the first write is what triggers anndata's copy-on-write,
/// and copy-on-write materializes a backed `X`. In particular it must precede
/// `write_accel_route` and `clear_gpu_normalize_marker`. Cheap argument
/// validation (`prefer_format`, `method`, …) still belongs above both.
///
/// Ops that write only to `obsm` / `obs` and are indifferent to gene order
/// (`neighbors`, `umap`, `leiden`, `harmony_integrate`, the eval metrics) do
/// not reject `preserve_var_order` today; they call
/// [`prepare_target_no_var_guard`] so this change adds no new rejections.
pub(crate) fn prepare_target(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    op: &'static str,
) -> PyResult<()> {
    reject_preserve_var_order(adata, op)?;
    crate::axis_align::devirtualize_scx_view(py, adata, op)?;
    Ok(())
}

/// [`prepare_target`] without the `preserve_var_order` guard, for write-back
/// ops that are indifferent to gene order (they consume `obsm` or write only
/// `obs` / `obsp`). Same ordering invariant.
pub(crate) fn prepare_target_no_var_guard(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    op: &'static str,
) -> PyResult<()> {
    crate::axis_align::devirtualize_scx_view(py, adata, op)?;
    Ok(())
}
