//! Rust-native analysis accelerators — R bindings via extendr.
//!
//! Thin wrappers around the `scx-accel` CPU kernels, mirroring the
//! [`crate::harmony`] / [`crate::lisi`] pattern: each `#[extendr]` function
//! takes plain R matrices/vectors, converts to the row-major `f32` / `ScxCsr`
//! layout the kernels expect, and returns a named R `list`. The pipe-friendly,
//! Seurat-aware front ends (`scx_pca()`, `scx_neighbors()`, …) live in
//! `R/accel.R` and assemble these into the slots Seurat users expect.
//!
//! All paths are CPU-only (rscx links no GPU feature). The full pipeline
//! Normalize → HVG → PCA → Neighbors → UMAP → Leiden → FindMarkers is
//! reachable from R and numerically consistent with the Python accelerators.

use extendr_api::prelude::*;

use scx_accel::{
    build_knn_graph, compute_umap, covariance_pca_inmemory, leiden, pflog1ppf_baseline_from_delta,
    pflog1ppf_pca, randomized_pca_inmemory, streaming_clip_square_sum, streaming_mean_var,
    wilcoxon_rank_sum, COVARIANCE_PCA_THRESHOLD,
};
use scx_sparse::ScxCsr;

use crate::util::factorize_chars;

/// Transpose a **genes × cells** `dgCMatrix` (Seurat's native orientation,
/// CSC over cells) into a **cells × genes** [`ScxCsr`] (the orientation the
/// `scx-accel` kernels expect: `n_obs` rows = cells, `n_vars` cols = genes).
///
/// Mirrors the transpose in [`crate::interop::csr_to_dgcmatrix`]'s inverse
/// (`dgcmatrix_to_csr`) but yields a validated `ScxCsr<f32>` directly with no
/// value-encoding step. Gene indices within each cell row come out ascending
/// because dgCMatrix stores `@i` sorted within each column.
fn dgc_genes_by_cells_to_csr(dgc: &Robj) -> Result<ScxCsr> {
    let dim_robj =
        R!("{{dgc}}@Dim").map_err(|e| Error::Other(format!("failed to get @Dim: {e}")))?;
    let dim: Vec<i32> = dim_robj
        .as_integer_slice()
        .ok_or_else(|| Error::Other("@Dim is not integer".into()))?
        .to_vec();
    if dim.len() != 2 {
        return Err(Error::Other(format!(
            "@Dim has {} elements, expected 2",
            dim.len()
        )));
    }
    let n_genes = dim[0] as usize; // rows of the dgCMatrix (genes)
    let n_cells = dim[1] as usize; // cols of the dgCMatrix (cells)

    let i_robj = R!("{{dgc}}@i").map_err(|e| Error::Other(format!("failed to get @i: {e}")))?;
    let csc_indices: Vec<i32> = i_robj
        .as_integer_slice()
        .ok_or_else(|| Error::Other("@i is not integer".into()))?
        .to_vec();
    let p_robj = R!("{{dgc}}@p").map_err(|e| Error::Other(format!("failed to get @p: {e}")))?;
    let csc_indptr: Vec<i32> = p_robj
        .as_integer_slice()
        .ok_or_else(|| Error::Other("@p is not integer".into()))?
        .to_vec();
    let x_robj = R!("{{dgc}}@x").map_err(|e| Error::Other(format!("failed to get @x: {e}")))?;
    let csc_values: Vec<f64> = x_robj
        .as_real_slice()
        .ok_or_else(|| Error::Other("@x is not double".into()))?
        .to_vec();

    if csc_indptr.len() != n_cells + 1 {
        return Err(Error::Other(format!(
            "@p length {} != n_cells + 1 ({})",
            csc_indptr.len(),
            n_cells + 1
        )));
    }
    let nnz = csc_values.len();

    // CSC column j (cell j) → CSR row j. Each CSC entry at gene-row i becomes
    // CSR (cell j, gene i). Per-row counts are just the per-column nnz.
    let mut csr_indptr = vec![0i64; n_cells + 1];
    for j in 0..n_cells {
        let count = (csc_indptr[j + 1] - csc_indptr[j]) as i64;
        csr_indptr[j + 1] = csr_indptr[j] + count;
    }
    let mut csr_indices = vec![0i32; nnz];
    let mut csr_data = vec![0.0f32; nnz];
    for j in 0..n_cells {
        let start = csc_indptr[j] as usize;
        let end = csc_indptr[j + 1] as usize;
        let row_start = csr_indptr[j] as usize;
        for (offset, k) in (start..end).enumerate() {
            csr_indices[row_start + offset] = csc_indices[k]; // gene index (0-based)
            csr_data[row_start + offset] = csc_values[k] as f32;
        }
    }

    ScxCsr::new((n_cells, n_genes), csr_indptr, csr_indices, csr_data)
        .map_err(|e| Error::Other(format!("CSR construction failed: {e}")))
}

// The single-shard in-memory `ShardSource` adapter now lives in scx-format as
// `SingleShardSource`, shared with pyscx.
use scx_format_io::shard_source::SingleShardSource;

/// Repack a column-major `RMatrix<f64>` (R layout) into row-major `f32`
/// of shape `n_rows × n_cols`, rejecting non-finite entries.
fn rmatrix_to_row_major_f32(m: &RMatrix<f64>) -> Result<(Vec<f32>, usize, usize)> {
    let n_rows = m.nrows();
    let n_cols = m.ncols();
    let col_major = m.data();
    let mut out = vec![0f32; n_rows * n_cols];
    for i in 0..n_rows {
        for j in 0..n_cols {
            let v = col_major[j * n_rows + i];
            if !v.is_finite() {
                return Err(Error::Other(format!(
                    "matrix[{}, {}] is NaN or Inf",
                    i + 1,
                    j + 1
                )));
            }
            out[i * n_cols + j] = v as f32;
        }
    }
    Ok((out, n_rows, n_cols))
}

/// Repack a row-major `f64` buffer (`n_rows × n_cols`) into a column-major R
/// matrix object.
fn row_major_to_rmatrix(data: &[f64], n_rows: usize, n_cols: usize) -> Result<Robj> {
    let mut col_major = vec![0f64; n_rows * n_cols];
    for i in 0..n_rows {
        for j in 0..n_cols {
            col_major[j * n_rows + i] = data[i * n_cols + j];
        }
    }
    let nr = n_rows as i32;
    let nc = n_cols as i32;
    R!("matrix({{col_major}}, nrow = {{nr}}, ncol = {{nc}})")
        .map_err(|e| Error::Other(e.to_string()))
}

// ─── PCA ──────────────────────────────────────────────────────────────

/// Run PCA on a counts/log-normalized expression matrix.
///
/// @param counts A **genes × cells** `dgCMatrix` (Seurat's native layout).
/// @param n_components Number of principal components.
/// @param zero_center Mean-center columns (genes) before decomposition.
/// @param n_oversamples Extra randomized dimensions for accuracy.
/// @param n_power_iterations Power iterations for spectral accuracy.
/// @param seed RNG seed.
///
/// Routes to the exact covariance solver when `n_vars <= 5000`
/// ([`COVARIANCE_PCA_THRESHOLD`]) and the randomized solver otherwise,
/// mirroring the Python accelerator.
///
/// @return list(`embeddings` = cells × n_components, `loadings` =
///   genes × n_components, `variance_explained`, `variance_ratio`,
///   `n_components`).
#[extendr]
fn scx_pca_matrix(
    counts: Robj,
    n_components: i32,
    zero_center: bool,
    n_oversamples: i32,
    n_power_iterations: i32,
    seed: f64,
) -> Result<Robj> {
    if n_components < 1 {
        return Err(Error::Other("n_components must be >= 1".into()));
    }
    let csr = dgc_genes_by_cells_to_csr(&counts)?;
    let n_comp = n_components as usize;
    let seed = seed as u64;

    let result = if csr.n_cols() <= COVARIANCE_PCA_THRESHOLD {
        covariance_pca_inmemory(&csr, n_comp, zero_center)
    } else {
        randomized_pca_inmemory(
            &csr,
            n_comp,
            n_oversamples.max(0) as usize,
            n_power_iterations.max(0) as usize,
            zero_center,
            seed,
        )
    }
    .map_err(|e| Error::Other(format!("pca: {e}")))?;

    let embeddings = row_major_to_rmatrix(&result.embeddings, result.n_obs, result.n_components)?;
    // `components` is row-major (n_components × n_vars); the genes × components
    // loadings are its transpose. Row-major (comp, gene) and column-major
    // (gene, comp) share the same flat layout, so the buffer needs no repack —
    // hand it straight to `matrix()`.
    let loadings = {
        let col_major = &result.components;
        let nr = result.n_vars as i32;
        let nc = result.n_components as i32;
        R!("matrix({{col_major}}, nrow = {{nr}}, ncol = {{nc}})")
            .map_err(|e| Error::Other(e.to_string()))?
    };
    let variance_explained = result.variance_explained.clone();
    let variance_ratio = result.variance_ratio.clone();
    let n_comp_i = result.n_components as i32;

    R!("list(
        embeddings = {{embeddings}},
        loadings = {{loadings}},
        variance_explained = {{variance_explained}},
        variance_ratio = {{variance_ratio}},
        n_components = {{n_comp_i}}
    )")
    .map_err(|e| Error::Other(e.to_string()))
}

// ─── PFlog1pPF (shifted-CLR) ────────────────────────────────────────────

/// Build the in-memory `delta` CSR `log1p(x_ij / (c·s_i))` from raw counts.
///
/// Mirrors `pyscx`'s `delta_from_raw_csr`: validates positive cell depths and
/// non-negative counts, then rewrites each nonzero in place (f64 math, f32
/// store). `delta` is the sparse, same-pattern part of `Z = delta + baseline`;
/// the centering `baseline` is recovered from it by
/// [`pflog1ppf_baseline_from_delta`].
#[allow(clippy::needless_range_loop)]
fn delta_from_raw_csr(raw: &ScxCsr, c: f64) -> Result<ScxCsr> {
    let depths = raw.row_sums();
    let mut delta = raw.clone();
    for r in 0..raw.n_rows() {
        let depth = depths[r];
        if depth <= 0.0 {
            return Err(Error::Other(format!(
                "pflog1ppf: cell {r} has non-positive depth {depth}; filter empty cells first"
            )));
        }
        let start = delta.indptr[r] as usize;
        let end = delta.indptr[r + 1] as usize;
        for v in &mut delta.data[start..end] {
            if (*v as f64) < 0.0 {
                return Err(Error::Other(format!(
                    "pflog1ppf: negative count {v} at cell {r}; counts must be non-negative"
                )));
            }
            *v = ((*v as f64) / (c * depth)).ln_1p() as f32;
        }
    }
    Ok(delta)
}

/// PFlog1pPF / shifted centered-log-ratio normalization (Booeshaghi et al.
/// 2026), returning a baseline-aware PCA embedding.
///
/// @param counts A **genes × cells** raw-counts `dgCMatrix` (Seurat's native
///   layout). PFlog1pPF is defined on raw counts, **not** log-normalized data.
/// @param c Shift / pseudocount (default `1.0` for PFlog1pPF).
/// @param n_components Number of principal components.
/// @param zero_center Mean-center columns (genes) before decomposition.
/// @param n_oversamples,n_power_iterations Randomized-SVD accuracy knobs.
/// @param seed RNG seed.
///
/// The exact transform `Z = delta + baseline·1ᵀ` is dense, but decomposes into
/// the sparse `delta` plus a per-cell `baseline`, so PCA never densifies. This
/// is the **in-memory** R path (materialized `dgCMatrix`); the streaming /
/// atlas-scale out-of-core path is `pyscx`-only.
///
/// @return list(`embeddings` = cells × n_components, `loadings` =
///   genes × n_components, `variance_explained`, `variance_ratio`,
///   `n_components`, `baseline` = per-cell length-`n_cells` vector).
#[extendr]
fn scx_pflog1ppf_matrix(
    counts: Robj,
    c: f64,
    n_components: i32,
    zero_center: bool,
    n_oversamples: i32,
    n_power_iterations: i32,
    seed: f64,
) -> Result<Robj> {
    if n_components < 1 {
        return Err(Error::Other("n_components must be >= 1".into()));
    }
    if c <= 0.0 || c.is_nan() {
        return Err(Error::Other(format!(
            "pflog1ppf: shift c must be positive, got {c}"
        )));
    }
    let raw = dgc_genes_by_cells_to_csr(&counts)?;
    let delta = delta_from_raw_csr(&raw, c)?;
    let source = SingleShardSource { csr: &delta };
    let n_comp = n_components as usize;

    let baseline = pflog1ppf_baseline_from_delta(&source)
        .map_err(|e| Error::Other(format!("pflog1ppf baseline: {e}")))?;
    let result = pflog1ppf_pca(
        &source,
        &baseline,
        n_comp,
        n_oversamples.max(0) as usize,
        n_power_iterations.max(0) as usize,
        zero_center,
        seed as u64,
    )
    .map_err(|e| Error::Other(format!("pflog1ppf pca: {e}")))?;

    let embeddings = row_major_to_rmatrix(&result.embeddings, result.n_obs, result.n_components)?;
    // `components` is row-major (n_components × n_vars); the genes × components
    // loadings are its transpose, which shares the same flat layout as a
    // column-major (gene, comp) matrix — hand the buffer straight to matrix().
    let loadings = {
        let col_major = &result.components;
        let nr = result.n_vars as i32;
        let nc = result.n_components as i32;
        R!("matrix({{col_major}}, nrow = {{nr}}, ncol = {{nc}})")
            .map_err(|e| Error::Other(e.to_string()))?
    };
    let variance_explained = result.variance_explained.clone();
    let variance_ratio = result.variance_ratio.clone();
    let n_comp_i = result.n_components as i32;

    R!("list(
        embeddings = {{embeddings}},
        loadings = {{loadings}},
        variance_explained = {{variance_explained}},
        variance_ratio = {{variance_ratio}},
        n_components = {{n_comp_i}},
        baseline = {{baseline}}
    )")
    .map_err(|e| Error::Other(e.to_string()))
}

// ─── Neighbors (kNN) ────────────────────────────────────────────────────

/// Build a kNN graph from a dense embedding (e.g. PCA cell coordinates).
///
/// @param embeddings A **cells × dims** numeric matrix (R column-major).
/// @param n_neighbors Number of nearest neighbors (k).
/// @param ef_construction HNSW build accuracy parameter.
/// @param ef_search HNSW search accuracy parameter.
/// @param seed RNG seed.
///
/// @return list(`n_obs`, `n_neighbors`, 0-based `indices` (cells × k) and
///   `distances` (cells × k) matrices, plus the connectivity CSR
///   (`conn_indptr`, 0-based `conn_indices`, `conn_data`)) ready to hand to
///   [`scx_umap_graph`] / [`scx_leiden_graph`].
#[extendr]
fn scx_knn_matrix(
    embeddings: RMatrix<f64>,
    n_neighbors: i32,
    ef_construction: i32,
    ef_search: i32,
    seed: f64,
) -> Result<Robj> {
    if n_neighbors < 1 {
        return Err(Error::Other("n_neighbors must be >= 1".into()));
    }
    let (data, n_obs, n_vars) = rmatrix_to_row_major_f32(&embeddings)?;
    let result = build_knn_graph(
        &data,
        n_obs,
        n_vars,
        n_neighbors as usize,
        ef_construction.max(1) as usize,
        ef_search.max(1) as usize,
        seed as u64,
    )
    .map_err(|e| Error::Other(format!("build_knn_graph: {e}")))?;

    // Neighbor indices/distances → cells × k matrices (row-major usize/f64).
    let k = result.n_neighbors;
    let idx_f64: Vec<f64> = result.indices.iter().map(|&v| v as f64).collect();
    let indices = row_major_to_rmatrix(&idx_f64, n_obs, k)?;
    let distances = row_major_to_rmatrix(&result.distances, n_obs, k)?;

    // Connectivity CSR — i64 indptr returned as R double (graph sizes fit).
    let conn_indptr: Vec<f64> = result.conn_indptr.iter().map(|&v| v as f64).collect();
    let conn_indices = result.conn_indices.clone();
    let conn_data = result.conn_data.clone();
    let n_obs_i = n_obs as i32;
    let k_i = k as i32;

    R!("list(
        n_obs = {{n_obs_i}},
        n_neighbors = {{k_i}},
        indices = {{indices}},
        distances = {{distances}},
        conn_indptr = {{conn_indptr}},
        conn_indices = {{conn_indices}},
        conn_data = {{conn_data}}
    )")
    .map_err(|e| Error::Other(e.to_string()))
}

// ─── UMAP ───────────────────────────────────────────────────────────────

/// Compute a UMAP embedding from a kNN connectivity graph.
///
/// @param conn_indptr,conn_indices,conn_data Connectivity CSR from
///   [`scx_knn_matrix`] (`conn_indptr` as double, `conn_indices` 0-based).
/// @param n_obs Number of cells.
/// @param n_components Output dimensions (typically 2).
/// @param n_epochs,min_dist,spread,negative_sample_rate,learning_rate,seed
///   Standard UMAP hyperparameters.
///
/// @return list(`embeddings` = cells × n_components matrix).
#[extendr]
#[allow(clippy::too_many_arguments)]
fn scx_umap_graph(
    conn_indptr: Vec<f64>,
    conn_indices: Vec<i32>,
    conn_data: Vec<f64>,
    n_obs: i32,
    n_components: i32,
    n_epochs: i32,
    min_dist: f64,
    spread: f64,
    negative_sample_rate: i32,
    learning_rate: f64,
    seed: f64,
) -> Result<Robj> {
    let n_obs = n_obs as usize;
    let indptr_i64: Vec<i64> = conn_indptr.iter().map(|&v| v as i64).collect();
    let result = compute_umap(
        &indptr_i64,
        &conn_indices,
        &conn_data,
        n_obs,
        n_components.max(1) as usize,
        n_epochs.max(1) as usize,
        min_dist,
        spread,
        negative_sample_rate.max(1) as usize,
        learning_rate,
        seed as u64,
        None,
    )
    .map_err(|e| Error::Other(format!("compute_umap: {e}")))?;

    let embeddings = row_major_to_rmatrix(&result.embeddings, result.n_obs, result.n_components)?;
    R!("list(embeddings = {{embeddings}})").map_err(|e| Error::Other(e.to_string()))
}

// ─── Leiden ─────────────────────────────────────────────────────────────

/// Run Leiden community detection on a graph (CSR connectivity).
///
/// @param indptr,indices,weights Graph CSR (`indptr` as double, `indices`
///   0-based) — typically the connectivity graph from [`scx_knn_matrix`].
/// @param n_nodes Number of cells.
/// @param resolution Resolution parameter γ (higher → more communities).
/// @param seed RNG seed.
/// @param max_iterations Maximum outer-loop iterations.
///
/// @return list(`membership` = 1-based integer cluster label per cell,
///   `modularity`, `n_communities`).
#[extendr]
fn scx_leiden_graph(
    indptr: Vec<f64>,
    indices: Vec<i32>,
    weights: Vec<f64>,
    n_nodes: i32,
    resolution: f64,
    seed: f64,
    max_iterations: i32,
) -> Result<Robj> {
    let n_nodes = n_nodes as usize;
    let indptr_i64: Vec<i64> = indptr.iter().map(|&v| v as i64).collect();
    let result = leiden(
        &indptr_i64,
        &indices,
        &weights,
        n_nodes,
        resolution,
        seed as u64,
        max_iterations.max(1) as usize,
        false,
    )
    .map_err(|e| Error::Other(format!("leiden: {e}")))?;

    // 0-based contiguous labels → 1-based for R.
    let membership: Vec<i32> = result.membership.iter().map(|&m| m as i32 + 1).collect();
    let modularity = result.modularity;
    let n_communities = result.n_communities as i32;
    R!("list(
        membership = {{membership}},
        modularity = {{modularity}},
        n_communities = {{n_communities}}
    )")
    .map_err(|e| Error::Other(e.to_string()))
}

// ─── Rank genes (Wilcoxon) ──────────────────────────────────────────────

/// Wilcoxon rank-sum differential expression (the scanpy
/// `rank_genes_groups` / Seurat `FindAllMarkers` equivalent).
///
/// @param counts A **genes × cells** `dgCMatrix` (log-normalized values
///   recommended); densified internally, so call on an HVG subset.
/// @param gene_names Character vector of length `n_vars` (gene IDs).
/// @param groups Character vector of length `n_cells` (per-cell group label).
/// @param reference Reference group name, or NULL to test each group vs rest.
/// @param log_transformed Whether the input is already log-transformed
///   (controls the logFC computation).
/// @param rankby_abs Rank by absolute score (scanpy `rankby_abs`).
/// @param tie_correct Apply tie correction to the rank-sum statistic. The R
///   front end defaults this to `FALSE` to match `pyscx` / scanpy
///   (`rank_genes_groups(..., tie_correct=False)`).
///
/// @return list(`group_names`, and per-group lists `names`, `scores`,
///   `pvals`, `pvals_adj`, `logfoldchanges`).
#[extendr]
fn scx_rank_genes(
    counts: Robj,
    gene_names: Strings,
    groups: Strings,
    reference: Nullable<String>,
    log_transformed: bool,
    rankby_abs: bool,
    tie_correct: bool,
) -> Result<Robj> {
    let csr = dgc_genes_by_cells_to_csr(&counts)?;
    let n_obs = csr.n_rows();
    let n_vars = csr.n_cols();

    let gene_names: Vec<String> = gene_names.iter().map(|s| s.to_string()).collect();
    if gene_names.len() != n_vars {
        return Err(Error::Other(format!(
            "gene_names length {} != n_vars {}",
            gene_names.len(),
            n_vars
        )));
    }
    let group_strs: Vec<String> = groups.iter().map(|s| s.to_string()).collect();
    if group_strs.len() != n_obs {
        return Err(Error::Other(format!(
            "groups length {} != n_cells {}",
            group_strs.len(),
            n_obs
        )));
    }
    let (codes, group_names) = factorize_chars(&group_strs);
    let groups_usize: Vec<usize> = codes.iter().map(|&c| c as usize).collect();

    let reference_idx = match reference {
        Nullable::NotNull(name) => match group_names.iter().position(|g| g == &name) {
            Some(i) => Some(i),
            None => {
                return Err(Error::Other(format!(
                    "reference group '{name}' not found among groups"
                )))
            }
        },
        Nullable::Null => None,
    };

    let dense = csr
        .to_dense()
        .map_err(|e| Error::Other(format!("densify failed: {e}")))?;

    let result = wilcoxon_rank_sum(
        &dense,
        n_obs,
        n_vars,
        &gene_names,
        &groups_usize,
        &group_names,
        reference_idx,
        log_transformed,
        rankby_abs,
        tie_correct,
        0,
    )
    .map_err(|e| Error::Other(format!("wilcoxon_rank_sum: {e}")))?;

    // Build per-group R lists. names[g] is a character vector; the numeric
    // vectors are parallel.
    let group_names_out = result.group_names.clone();
    let names_list = List::from_values(
        result
            .names
            .iter()
            .map(|v| v.iter().map(|s| s.as_str()).collect::<Vec<_>>().into_robj()),
    );
    let scores_list = List::from_values(result.scores.iter().map(|v| v.clone().into_robj()));
    let pvals_list = List::from_values(result.pvals.iter().map(|v| v.clone().into_robj()));
    let pvals_adj_list = List::from_values(result.pvals_adj.iter().map(|v| v.clone().into_robj()));
    let lfc_list = List::from_values(result.logfoldchanges.iter().map(|v| v.clone().into_robj()));

    let names_robj: Robj = names_list.into();
    let scores_robj: Robj = scores_list.into();
    let pvals_robj: Robj = pvals_list.into();
    let pvals_adj_robj: Robj = pvals_adj_list.into();
    let lfc_robj: Robj = lfc_list.into();

    R!("list(
        group_names = {{group_names_out}},
        names = {{names_robj}},
        scores = {{scores_robj}},
        pvals = {{pvals_robj}},
        pvals_adj = {{pvals_adj_robj}},
        logfoldchanges = {{lfc_robj}}
    )")
    .map_err(|e| Error::Other(e.to_string()))
}

// ─── Highly variable genes (seurat_v3 two-pass) ─────────────────────────

/// HVG pass 1: per-gene mean and variance (Bessel-corrected).
///
/// @param counts A **genes × cells** raw-counts `dgCMatrix`.
/// @return list(`means`, `variances`) — vectors of length `n_genes`.
#[extendr]
fn scx_hvg_mean_var(counts: Robj) -> Result<Robj> {
    let csr = dgc_genes_by_cells_to_csr(&counts)?;
    let source = SingleShardSource { csr: &csr };
    let stats =
        streaming_mean_var(&source).map_err(|e| Error::Other(format!("hvg pass 1: {e}")))?;
    let means = stats.means;
    let variances = stats.variances;
    R!("list(means = {{means}}, variances = {{variances}})")
        .map_err(|e| Error::Other(e.to_string()))
}

/// HVG pass 2: per-gene clipped sum and clipped sum-of-squares.
///
/// @param counts A **genes × cells** raw-counts `dgCMatrix`.
/// @param clip_val Per-gene clip threshold (length `n_genes`), derived in R
///   from the seurat_v3 loess fit on the pass-1 mean/variance. `streaming_clip_square_sum`
///   compares it against the f32 CSR values; the f64→f32 narrowing is safe
///   here because `clip_val = reg_std * sqrt(N) + mean` stays far below f32's
///   range for count data.
/// @return list(`counts_sum`, `sq_counts_sum`).
#[extendr]
fn scx_hvg_clipped_sums(counts: Robj, clip_val: Vec<f64>) -> Result<Robj> {
    let csr = dgc_genes_by_cells_to_csr(&counts)?;
    if clip_val.len() != csr.n_cols() {
        return Err(Error::Other(format!(
            "clip_val length {} != n_genes {}",
            clip_val.len(),
            csr.n_cols()
        )));
    }
    let source = SingleShardSource { csr: &csr };
    let (counts_sum, sq_counts_sum) = streaming_clip_square_sum(&source, &clip_val)
        .map_err(|e| Error::Other(format!("hvg pass 2: {e}")))?;
    R!("list(counts_sum = {{counts_sum}}, sq_counts_sum = {{sq_counts_sum}})")
        .map_err(|e| Error::Other(e.to_string()))
}

extendr_module! {
    mod accel;
    fn scx_pca_matrix;
    fn scx_pflog1ppf_matrix;
    fn scx_knn_matrix;
    fn scx_umap_graph;
    fn scx_leiden_graph;
    fn scx_rank_genes;
    fn scx_hvg_mean_var;
    fn scx_hvg_clipped_sums;
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    // SingleShardSource's accessors come from the ShardSource trait.
    use scx_format_io::shard_source::ShardSource;

    #[test]
    fn test_factorize_levels_first_seen() {
        let labels: Vec<String> = ["T", "B", "T", "NK", "B"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (codes, levels) = factorize_chars(&labels);
        assert_eq!(levels, vec!["T", "B", "NK"]);
        assert_eq!(codes, vec![0, 1, 0, 2, 1]);
    }

    #[test]
    fn test_in_memory_shard_source_roundtrips_csr() {
        // 2 cells × 3 genes.
        let csr = ScxCsr::new((2, 3), vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let src = SingleShardSource { csr: &csr };
        assert_eq!(src.n_shards(), 1);
        assert_eq!(src.n_obs(), 2);
        assert_eq!(src.n_vars(), 3);
        assert_eq!(src.max_shard_rows().unwrap(), 2);
        let stats = streaming_mean_var(&src).unwrap();
        assert_eq!(stats.means.len(), 3);
        // Column 0: values [1.0, 0.0] → mean 0.5.
        assert!((stats.means[0] - 0.5).abs() < 1e-9);
    }
}
