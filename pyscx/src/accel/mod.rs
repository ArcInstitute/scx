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
//! - `eval_metrics` — Perturbation evaluation metrics (pseudobulk means,
//!   energy distance, discrimination score, knockdown, clustering agreement)
//! - `util` — Shared CSR extraction helpers

pub mod de;
pub mod eval_metrics;
pub mod filtering;
pub mod gpu;
pub mod harmony;
pub mod hvg;
pub mod leiden;
pub mod lisi;
pub mod neighbors;
pub mod pca;
pub mod preprocessing;
pub mod pseudobulk;
pub mod umap;
pub mod util;
