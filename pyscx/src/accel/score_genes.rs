//! Gene-set scoring — scanpy `sc.tl.score_genes` equivalent (CPU-native).
//!
//! Three methods selectable via `method=`:
//! * `"control"` — scanpy `score_genes`: `mean(gene_list) − mean(control)`,
//!   control sampled from expression-matched bins. The binning + control
//!   sampling are Rust-native and deterministic given `random_state` but do NOT
//!   bit-match scanpy's numpy RNG, so absolute scores differ (rank correlation
//!   stays high).
//! * `"mean"` — per-cell mean over `gene_list`.
//! * `"zscore"` — per-gene z-standardize then aggregate (decoupler `mt.zscore`).
//!
//! Streams shard-by-shard through `ShardSource`, so it runs on in-memory,
//! backed, and lazy `X` with identical numerics and bounded memory. CPU-only:
//! `device` is accepted for API symmetry but there is no GPU kernel.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use scx_accel::{score_genes as accel_score_genes, ScoreMethod};
use scx_format_io::shard_source::SingleShardSource;
use scx_format_io::ShardSource;

use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::ScxLazyTransformedDataset;

use super::hvg::build_shard_source;
use super::util::extract_materialized_csr;

/// Score a set of genes per cell, writing the result to `adata.obs[score_name]`.
///
/// `gene_list` / `gene_pool` are gene symbols resolved against `adata.var.index`;
/// genes absent from `var_names` are dropped with a warning. `gene_pool` defaults
/// to all genes and is only used by `method="control"`.
#[pyfunction]
#[pyo3(signature = (
    adata,
    gene_list,
    ctrl_size=50,
    gene_pool=None,
    n_bins=25,
    score_name="score",
    random_state=0,
    method="control",
    layer=None,
    device="auto",
))]
#[allow(clippy::too_many_arguments)]
pub fn score_genes<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    gene_list: Vec<String>,
    ctrl_size: usize,
    gene_pool: Option<Vec<String>>,
    n_bins: usize,
    score_name: &str,
    random_state: u64,
    method: &str,
    layer: Option<&str>,
    device: &str,
) -> PyResult<()> {
    // gene_list is resolved against var_names (presentation order) but the
    // ShardSource gathers in sorted-projection order — a presentation-ordered
    // backed X would score the wrong physical columns. Reject loudly.
    super::reject_preserve_var_order(adata, "score_genes")?;

    // ── Parse the scoring method ────────────────────────────────────────
    let method_enum = match method {
        "control" => ScoreMethod::Control {
            ctrl_size,
            n_bins,
            random_state,
        },
        "mean" => ScoreMethod::Mean,
        "zscore" => ScoreMethod::Zscore,
        other => {
            return Err(PyValueError::new_err(format!(
                "score_genes: unknown method {other:?}; expected \"control\", \"mean\", or \"zscore\""
            )));
        }
    };

    // ── Resolve gene symbols → var-index positions ──────────────────────
    let var = adata.getattr("var")?;
    let var_index = var.getattr("index")?;
    let var_names: Vec<String> = var_index.call_method0("tolist")?.extract()?;
    // Borrow keys from `var_names` (which outlives this function) — no per-gene
    // String allocation.
    let mut name_to_idx: HashMap<&str, u32> = HashMap::with_capacity(var_names.len());
    for (i, name) in var_names.iter().enumerate() {
        // First occurrence wins for duplicate var names (matches pandas .loc).
        name_to_idx.entry(name.as_str()).or_insert(i as u32);
    }

    let mut seen: HashSet<u32> = HashSet::new();
    let mut gene_list_idx: Vec<u32> = Vec::with_capacity(gene_list.len());
    let mut missing: Vec<String> = Vec::new();
    for g in &gene_list {
        match name_to_idx.get(g.as_str()) {
            Some(&idx) => {
                if seen.insert(idx) {
                    gene_list_idx.push(idx);
                }
            }
            None => missing.push(g.clone()),
        }
    }
    if !missing.is_empty() {
        let shown: Vec<&String> = missing.iter().take(10).collect();
        let suffix = if missing.len() > 10 { ", …" } else { "" };
        let warnings = py.import("warnings")?;
        warnings.call_method1(
            "warn",
            (format!(
                "score_genes: {} of {} genes in gene_list are not in var_names and were dropped: {:?}{}",
                missing.len(),
                gene_list.len(),
                shown,
                suffix
            ),),
        )?;
    }
    if gene_list_idx.is_empty() {
        return Err(PyValueError::new_err(
            "score_genes: no genes from gene_list were found in adata.var_names",
        ));
    }

    let gene_pool_idx: Vec<u32> = match gene_pool {
        Some(pool) => {
            // De-duplicate resolved indices: a duplicate pool gene would be
            // binned twice in select_control_genes and skew control sampling.
            let mut seen_pool: HashSet<u32> = HashSet::new();
            let resolved: Vec<u32> = pool
                .iter()
                .filter_map(|g| name_to_idx.get(g.as_str()).copied())
                .filter(|&idx| seen_pool.insert(idx))
                .collect();
            // An explicit pool that mostly fails to resolve usually means
            // wrong/typo'd symbols — surface it (gene_list already warns).
            if !pool.is_empty() && resolved.len() * 2 < pool.len() {
                let warnings = py.import("warnings")?;
                warnings.call_method1(
                    "warn",
                    (format!(
                        "score_genes: only {} of {} genes in gene_pool resolved against \
                         var_names; check that gene_pool uses the same identifiers as adata.var_names",
                        resolved.len(),
                        pool.len()
                    ),),
                )?;
            }
            resolved
        }
        None => (0..var_names.len() as u32).collect(),
    };

    // ── Select the source matrix (layer or X) ───────────────────────────
    let x = match layer {
        Some(name) => adata.getattr("layers")?.get_item(name)?,
        None => adata.getattr("X")?,
    };

    // ── Dispatch: backed → lazy → in-memory, all via ShardSource ─────────
    if let Ok(backed) = x.cast::<ScxBackedSparseDataset>() {
        let backed_ref = backed.borrow();
        let reader = Arc::clone(&backed_ref.backed);
        let n_vars = backed_ref.shape_val.1;
        let kept = backed_ref.kept_to_global.clone();
        let col_proj = backed_ref.col_projection_arc();
        drop(backed_ref);
        let source = build_shard_source(&reader, &[], &kept, &col_proj, n_vars);
        return score_on_source(
            py,
            adata,
            &source,
            &gene_list_idx,
            &gene_pool_idx,
            &method_enum,
            score_name,
            device,
        );
    }

    if let Ok(lazy) = x.cast::<ScxLazyTransformedDataset>() {
        let lazy_ref = lazy.borrow();
        let reader = Arc::clone(&lazy_ref.backed);
        let transforms = lazy_ref.transforms.clone();
        let n_vars = lazy_ref.shape_val.1;
        let kept = lazy_ref.kept_to_global.clone();
        let col_proj = lazy_ref.col_projection.clone();
        drop(lazy_ref);
        let source = build_shard_source(&reader, &transforms, &kept, &col_proj, n_vars);
        return score_on_source(
            py,
            adata,
            &source,
            &gene_list_idx,
            &gene_pool_idx,
            &method_enum,
            score_name,
            device,
        );
    }

    // In-memory scipy/dense X → wrap the materialized CSR as a single shard.
    let csr = extract_materialized_csr(py, &x)?;
    let source = SingleShardSource { csr: &csr };
    score_on_source(
        py,
        adata,
        &source,
        &gene_list_idx,
        &gene_pool_idx,
        &method_enum,
        score_name,
        device,
    )
}

/// Run the kernel on a `ShardSource`, write per-cell scores to
/// `adata.obs[score_name]`, and stamp the CPU route metadata.
#[allow(clippy::too_many_arguments)]
fn score_on_source<S: ShardSource + Sync>(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    source: &S,
    gene_list_idx: &[u32],
    gene_pool_idx: &[u32],
    method: &ScoreMethod,
    score_name: &str,
    device: &str,
) -> PyResult<()> {
    let scores = accel_score_genes(source, gene_list_idx, gene_pool_idx, method)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // adata.obs[score_name] = float64 per-cell scores (positional, like scanpy).
    let arr = numpy::PyArray::from_vec(py, scores);
    adata.getattr("obs")?.set_item(score_name, arr)?;

    super::route::write_accel_route(
        py,
        adata,
        "score_genes",
        &super::route::cpu_only_exec_info(device),
    )?;
    Ok(())
}
