//! Streaming gene-set scoring — scanpy `sc.tl.score_genes` family.
//!
//! Three methods, all computed by streaming CSR shards through [`ShardSource`]
//! (so they work identically on in-memory, backed, and lazy data) without
//! materializing the full matrix or densifying the gene set:
//!
//! * [`ScoreMethod::Control`] — scanpy `score_genes`: per-cell
//!   `mean(gene_list) − mean(control)`, where the control set is sampled from
//!   expression-matched bins. Binning + control selection are Rust-native and
//!   deterministic given `random_state`, but do **not** bit-match numpy's RNG —
//!   so the chosen control genes differ from scanpy even though the algorithm
//!   is the same.
//! * [`ScoreMethod::Mean`] — per-cell mean over `gene_list` (no control set).
//! * [`ScoreMethod::Zscore`] — per-gene z-standardize across cells, then
//!   aggregate over `gene_list` (decoupler `mt.zscore`): `Σ z / sqrt(k)`.
//!
//! ## Algorithm shape
//!
//! Every method decomposes into at most two streaming passes:
//! 1. (Control / Zscore only) per-gene mean and variance via
//!    [`crate::hvg::streaming_mean_var`].
//! 2. per-cell weighted column sums over the selected gene set(s)
//!    ([`streaming_weighted_row_sums`]). Per-gene weights bake in the `1/k`
//!    (mean), `1/std` (zscore), or set-membership normalization so the final
//!    score is a cheap per-cell combination of the pass-2 outputs.

use crate::error::{AccelError, Result};
use crate::hvg::streaming_mean_var;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use scx_format_io::ShardSource;

/// Gene-set scoring method. See the module docs for definitions.
#[derive(Debug, Clone)]
pub enum ScoreMethod {
    /// scanpy `score_genes`: `mean(gene_list) − mean(control)` per cell.
    Control {
        /// Number of control genes to sample per occupied expression bin.
        ctrl_size: usize,
        /// Number of expression bins (scanpy `n_bins`, default 25).
        n_bins: usize,
        /// Seed for the deterministic (non-numpy) control sampler.
        random_state: u64,
    },
    /// Per-cell mean over `gene_list`.
    Mean,
    /// Per-cell decoupler `mt.zscore`: `Σ_{g} (x_g − mean_g)/std_g / sqrt(k)`.
    Zscore,
}

/// Reject non-finite values at the scoring boundary (mirrors the HVG/DE
/// boundary checks): a NaN/Inf would silently poison the per-cell sums.
fn ensure_finite(data: &[f32]) -> Result<()> {
    if let Some(pos) = data.iter().position(|v| !v.is_finite()) {
        return Err(AccelError::InvalidInput(format!(
            "gene scoring input contains a non-finite value ({}) at nonzero index {pos}; \
             score_genes requires finite input — filter/QC NaN and Inf before scoring",
            data[pos]
        )));
    }
    Ok(())
}

/// Per-cell weighted column sums for one or more weight vectors, in a single
/// streaming pass. `weights[k]` has length `n_vars` with `0.0` for columns not
/// in set `k`. Returns `out[k]` of length `n_obs` where
/// `out[k][cell] = Σ_col weights[k][col] · x[cell, col]`.
///
/// The inner loop touches every nonzero once and does `weights.len()` (1–2)
/// lookups per nonzero — CSR is row-major so a column subset cannot be reached
/// without scanning, matching scanpy's `X[:, genes].mean(axis=1)`.
fn streaming_weighted_row_sums<S: ShardSource>(
    source: &S,
    weights: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>> {
    let n_obs = source.n_obs();
    let n_vars = source.n_vars();
    for w in weights {
        debug_assert_eq!(w.len(), n_vars);
    }

    let mut out = vec![vec![0.0f64; n_obs]; weights.len()];
    let mut row_base = 0usize;
    for shard_idx in 0..source.n_shards() {
        let csr = source.read_shard(shard_idx)?;
        ensure_finite(&csr.data)?;
        let rows = csr.n_rows();
        for r in 0..rows {
            let start = csr.indptr[r] as usize;
            let end = csr.indptr[r + 1] as usize;
            let cell = row_base + r;
            for nz in start..end {
                let col = csr.indices[nz] as usize;
                let v = csr.data[nz] as f64;
                for (k, w) in weights.iter().enumerate() {
                    let wc = w[col];
                    if wc != 0.0 {
                        out[k][cell] += wc * v;
                    }
                }
            }
        }
        row_base += rows;
    }

    if row_base != n_obs {
        return Err(AccelError::ShapeError(format!(
            "gene scoring streamed {row_base} rows but source reports n_obs={n_obs}"
        )));
    }
    Ok(out)
}

/// Replicate scanpy's expression-matched control-gene selection.
///
/// `means` is the per-gene mean over all cells (length `n_vars`). Genes in
/// `gene_pool` (with finite mean) are binned by the rank of their mean (ties →
/// min rank, 1-based), with bin width `n_items = round(n_pool / (n_bins - 1))`
/// and `bin = rank / n_items` (integer division). For each bin occupied by a
/// `gene_list` gene, up to `ctrl_size` genes are sampled without replacement
/// (seeded by `random_state`), the scored genes are removed (scanpy
/// `ctrl_as_ref=True`), and the result is unioned across bins.
///
/// Deterministic given `random_state`, but not numpy-RNG-compatible.
fn select_control_genes(
    means: &[f64],
    gene_list: &[u32],
    gene_pool: &[u32],
    ctrl_size: usize,
    n_bins: usize,
    random_state: u64,
) -> Vec<u32> {
    use std::collections::{BTreeSet, HashMap};

    // Pool genes with finite means, paired with their mean.
    let mut pool: Vec<(u32, f64)> = gene_pool
        .iter()
        .copied()
        .filter(|&g| means.get(g as usize).is_some_and(|m| m.is_finite()))
        .map(|g| (g, means[g as usize]))
        .collect();
    if pool.is_empty() {
        return Vec::new();
    }

    // Sort by mean ascending; ties broken by gene index for reproducibility.
    pool.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    // Min-rank (1-based): genes with equal mean share the rank of the first in
    // the sorted run — matches pandas `rank(method="min")`.
    let n = pool.len();
    let mut rank = vec![0usize; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && pool[j].1 == pool[i].1 {
            j += 1;
        }
        let r = i + 1;
        for slot in rank.iter_mut().take(j).skip(i) {
            *slot = r;
        }
        i = j;
    }

    // scanpy: n_items = int(round(len(obs_avg) / (n_bins - 1))); guard against
    // a zero width when the pool is tiny relative to n_bins.
    let denom = (n_bins.max(2) - 1) as f64;
    let n_items = (((n as f64) / denom).round() as usize).max(1);

    let mut gene_bin: HashMap<u32, usize> = HashMap::with_capacity(n);
    let mut bin_members: HashMap<usize, Vec<u32>> = HashMap::new();
    for (pos, &(gene, _)) in pool.iter().enumerate() {
        let bin = rank[pos] / n_items;
        gene_bin.insert(gene, bin);
        bin_members.entry(bin).or_default().push(gene);
    }

    // Distinct bins occupied by gene_list genes that are in the pool. BTreeSet
    // keeps a deterministic bin-iteration order (mirrors scanpy's np.unique).
    let mut cut_bins: BTreeSet<usize> = BTreeSet::new();
    for &g in gene_list {
        if let Some(&b) = gene_bin.get(&g) {
            cut_bins.insert(b);
        }
    }

    let gene_list_set: BTreeSet<u32> = gene_list.iter().copied().collect();
    let mut rng = StdRng::seed_from_u64(random_state);
    let mut control: BTreeSet<u32> = BTreeSet::new();

    for bin in cut_bins {
        let members = &bin_members[&bin];
        let chosen: Vec<u32> = if ctrl_size < members.len() {
            members
                .choose_multiple(&mut rng, ctrl_size)
                .copied()
                .collect()
        } else {
            members.clone()
        };
        for g in chosen {
            if !gene_list_set.contains(&g) {
                control.insert(g);
            }
        }
    }

    control.into_iter().collect()
}

/// Per-cell gene-set scores for `gene_list` (length `n_obs`).
///
/// `gene_list` and `gene_pool` are 0-based column (var) indices; `gene_pool` is
/// only consulted by [`ScoreMethod::Control`] (the universe genes are binned
/// into). Callers should de-duplicate `gene_list` (duplicates inflate the `1/k`
/// normalization).
pub fn score_genes<S: ShardSource>(
    source: &S,
    gene_list: &[u32],
    gene_pool: &[u32],
    method: &ScoreMethod,
) -> Result<Vec<f64>> {
    let n_obs = source.n_obs();
    let n_vars = source.n_vars();

    if gene_list.is_empty() {
        return Err(AccelError::InvalidInput(
            "score_genes: gene_list is empty after resolving against var_names".into(),
        ));
    }
    for &g in gene_list {
        if g as usize >= n_vars {
            return Err(AccelError::InvalidInput(format!(
                "score_genes: gene index {g} out of range (n_vars={n_vars})"
            )));
        }
    }
    if n_obs == 0 {
        return Ok(Vec::new());
    }

    let k_list = gene_list.len() as f64;

    match method {
        ScoreMethod::Mean => {
            let mut w = vec![0.0f64; n_vars];
            let inv = 1.0 / k_list;
            for &g in gene_list {
                w[g as usize] = inv;
            }
            let out = streaming_weighted_row_sums(source, std::slice::from_ref(&w))?;
            Ok(out.into_iter().next().unwrap())
        }
        ScoreMethod::Zscore => {
            // decoupler `mt.zscore`: per-gene std uses ddof=1 (Bessel), which is
            // exactly what `streaming_mean_var` returns; score = Σ z / sqrt(k).
            let stats = streaming_mean_var(source)?;
            let mut w = vec![0.0f64; n_vars];
            let mut offset = 0.0f64; // Σ mean_g / std_g over genes with std > 0
            for &g in gene_list {
                let gi = g as usize;
                let std = stats.variances[gi].sqrt();
                if std > 0.0 {
                    let inv = 1.0 / std;
                    w[gi] = inv;
                    offset += stats.means[gi] * inv;
                }
                // std == 0 (constant gene) → z ≡ 0; contributes nothing.
            }
            let sqrt_k = k_list.sqrt();
            let out = streaming_weighted_row_sums(source, std::slice::from_ref(&w))?;
            let s = out.into_iter().next().unwrap();
            Ok(s.into_iter().map(|si| (si - offset) / sqrt_k).collect())
        }
        ScoreMethod::Control {
            ctrl_size,
            n_bins,
            random_state,
        } => {
            let stats = streaming_mean_var(source)?;
            let control = select_control_genes(
                &stats.means,
                gene_list,
                gene_pool,
                *ctrl_size,
                *n_bins,
                *random_state,
            );

            let mut w_list = vec![0.0f64; n_vars];
            let inv_list = 1.0 / k_list;
            for &g in gene_list {
                w_list[g as usize] = inv_list;
            }

            let mut w_ctrl = vec![0.0f64; n_vars];
            if !control.is_empty() {
                let inv_ctrl = 1.0 / control.len() as f64;
                for &g in &control {
                    w_ctrl[g as usize] = inv_ctrl;
                }
            }

            let out = streaming_weighted_row_sums(source, &[w_list, w_ctrl])?;
            let mut it = out.into_iter();
            let list_means = it.next().unwrap();
            let ctrl_means = it.next().unwrap();
            // Empty control set → ctrl_means is all-zero, so score = mean(list).
            Ok(list_means
                .into_iter()
                .zip(ctrl_means)
                .map(|(a, b)| a - b)
                .collect())
        }
    }
}

#[cfg(test)]
#[path = "gene_score_tests.rs"]
mod tests;
