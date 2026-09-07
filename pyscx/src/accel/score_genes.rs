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

use crate::lazy_transform::ScxLazyTransformedDataset;

use super::backed_shard_parts;
use super::hvg::build_shard_source;

/// Resolve gene symbols to var-index positions, de-duplicated, first-seen order.
///
/// Returns `(resolved, missing)`. Duplicate *inputs* collapse to one index (a
/// repeat would inflate the `1/k` normalization); duplicate *var_names* resolve
/// to their first occurrence, matching pandas `.loc`.
///
/// The error policy is the caller's: `gene_list` and `ctrl_genes` warn on drops
/// and refuse an empty result, while `gene_pool` warns only when most of it
/// failed. Sharing the resolution but not the policy is deliberate — an
/// under-resolved pool still samples, an under-resolved control set silently
/// answers a different question.
fn resolve_var_indices(
    name_to_idx: &HashMap<&str, u32>,
    names: &[String],
) -> (Vec<u32>, Vec<String>) {
    let mut seen: HashSet<u32> = HashSet::new();
    let mut resolved: Vec<u32> = Vec::with_capacity(names.len());
    let mut missing: Vec<String> = Vec::new();
    for g in names {
        match name_to_idx.get(g.as_str()) {
            Some(&idx) => {
                if seen.insert(idx) {
                    resolved.push(idx);
                }
            }
            None => missing.push(g.clone()),
        }
    }
    (resolved, missing)
}

/// Score a set of genes per cell, writing the result to `adata.obs[score_name]`.
///
/// `gene_list` / `gene_pool` / `ctrl_genes` are gene symbols resolved against
/// `adata.var.index`; genes absent from `var_names` are dropped with a warning.
/// `gene_pool` defaults to all genes and is only used by `method="control"`.
///
/// `ctrl_genes` supplies the control set directly and skips the
/// expression-matched sampling, which is the exact-parity route: given the
/// controls scanpy used, the score matches `sc.tl.score_genes`. The sampler
/// behind `method="control"` is deterministic but is not numpy's, so it draws
/// different control genes and the absolute scores differ. `ctrl_genes` also
/// works on a backed `X`, which `sc.tl.score_genes` refuses outright. It
/// requires `method="control"` and rejects an explicit `gene_pool`, whose only
/// role is to be sampled from; `ctrl_size` / `n_bins` / `random_state` are
/// ignored, there being no sampling left to steer.
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
    *,
    ctrl_genes=None,
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
    ctrl_genes: Option<Vec<String>>,
) -> PyResult<()> {
    // gene_list is resolved against var_names (presentation order) but the
    // ShardSource gathers in sorted-projection order — a presentation-ordered
    // backed X would score the wrong physical columns. Reject loudly.
    // Deliberately the no-var-guard prologue: the
    // `reject_presentation_ordered_source` call below guards the matrix this
    // op will actually read, which is `adata.layers[layer]` when `layer=` was
    // given. Keeping the X-only check here made the documented remedy
    // impossible — materialise the layer, and the op still refused because
    // `adata.X` was presentation-ordered, while the matrix it was about to
    // read was fine.
    super::prepare_target_no_var_guard(py, adata, "score_genes")?;

    // Resolve the matrix and guard it **here**, before `adata.var` is read: a
    // caller whose gene axis is in a requested order should be told about the
    // axis, not handed a downstream symptom. This fixture makes the difference
    // concrete — `var.index` holds ENSG ids while `var_names=` selected through
    // a symbol column, so resolving the gene list first reports "no genes from
    // gene_list were found", which is true and useless.
    let x = match layer {
        // Same typed error as `select_de_matrix` / `calculate_qc_metrics`
        // rather than the mapping's bare `KeyError`.
        Some(name) => adata.getattr("layers")?.get_item(name).map_err(|_| {
            PyValueError::new_err(format!("layer '{name}' not found in adata.layers"))
        })?,
        None => adata.getattr("X")?,
    };
    super::reject_presentation_ordered_source(&x, "score_genes")?;

    // Validate `device=` like every other accel op (round-2 review: this op
    // previously skipped `resolve_device`, so `device="tpu"` was silently
    // treated as GPU intent in the route stamp instead of raising).
    let _device = super::gpu::resolve_device(device)?;

    // ── Parse the scoring method (shared vocabulary + error text) ───────
    let method_enum = ScoreMethod::parse(method, ctrl_size, n_bins, random_state)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // `ctrl_genes` replaces the control *selection*, so it is meaningful only
    // for the one method that has controls, and it makes `gene_pool` — which
    // exists solely to be binned and sampled from — dead. Silently ignoring
    // either is how a published number ends up computed from something other
    // than what the call says.
    if ctrl_genes.is_some() {
        if !matches!(method_enum, ScoreMethod::Control { .. }) {
            return Err(PyValueError::new_err(format!(
                "score_genes: ctrl_genes= is only meaningful for method=\"control\" \
                 (got method={method:?}); \"mean\" and \"zscore\" use no control set"
            )));
        }
        if gene_pool.is_some() {
            return Err(PyValueError::new_err(
                "score_genes: ctrl_genes= and gene_pool= are mutually exclusive — \
                 gene_pool only supplies the universe the control set is sampled \
                 from, and ctrl_genes= replaces that sampling entirely",
            ));
        }
    }

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

    let (gene_list_idx, missing) = resolve_var_indices(&name_to_idx, &gene_list);
    if !missing.is_empty() {
        let shown: Vec<&String> = missing.iter().take(10).collect();
        let suffix = if missing.len() > 10 { ", …" } else { "" };
        let warnings = crate::pyimport::import_module(py, "warnings")?;
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
                let warnings = crate::pyimport::import_module(py, "warnings")?;
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

    // ── Resolve an explicit control set, if one was given ───────────────
    //
    // Deliberately the `gene_list` policy (warn on drops, error if nothing
    // survives), not `gene_pool`'s lenient one: a control set that quietly
    // loses half its members still returns a number, and that number is no
    // longer the one the caller's controls define — which is the entire
    // reason to pass them.
    let method_enum = match &ctrl_genes {
        None => method_enum,
        Some(names) => {
            let (ctrl, missing) = resolve_var_indices(&name_to_idx, names);
            if !missing.is_empty() {
                let shown: Vec<&String> = missing.iter().take(10).collect();
                let suffix = if missing.len() > 10 { ", …" } else { "" };
                let warnings = crate::pyimport::import_module(py, "warnings")?;
                warnings.call_method1(
                    "warn",
                    (format!(
                        "score_genes: {} of {} genes in ctrl_genes are not in var_names \
                         and were dropped: {:?}{}",
                        missing.len(),
                        names.len(),
                        shown,
                        suffix
                    ),),
                )?;
            }
            if ctrl.is_empty() {
                return Err(PyValueError::new_err(
                    "score_genes: no genes from ctrl_genes were found in adata.var_names",
                ));
            }
            ScoreMethod::ControlSet { ctrl }
        }
    };

    // ── Select the source matrix (layer or X) ───────────────────────────
    // ── Dispatch: backed → lazy → in-memory, all via ShardSource ─────────
    if let Some(parts) = backed_shard_parts(&x) {
        let source = build_shard_source(
            &parts.reader,
            &[],
            &parts.kept,
            &parts.col_proj,
            parts.n_vars,
        );
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
    let csr = crate::convert::owned_csr(py, &x, None)?;
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
