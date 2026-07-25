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
/// `preserve_var_order` lives only on `ScxBackedSparseDataset` (as
/// `col_presentation`). Accel ops that build a `ScxLazyTransformedDataset`
/// or a `ShardSource` from the dataset's *sorted* `col_projection` would
/// decode columns in sorted order while `adata.var` stays in request order —
/// a silent X/var misalignment (or, for name-resolving ops like
/// `score_genes`, the wrong physical columns). Until the permutation is
/// propagated into those paths, refuse loudly rather than return wrong data.
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
    if let Ok(backed) = x.cast::<crate::backed::ScxBackedSparseDataset>() {
        if backed.borrow().col_presentation_arc().is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{op} is not supported on a dataset opened with \
                 preserve_var_order=True (the gene axis is in request order, which \
                 would misalign the result against adata.var). Run {op} before \
                 projecting by name, or re-open without preserve_var_order."
            )));
        }
    }
    Ok(())
}
