//! The Wilcoxon rank-sum kernel driver, the `rank_genes_groups` entry
//! point, and the adata write-back.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyDict;

use super::*;
use crate::lazy_transform::ScxLazyTransformedDataset;

/// scanpy's `pts` / `pts_rest` tables and the column order they are written in.
pub(crate) struct PtsTables {
    /// `pts` frame columns: every group of the label universe (reference
    /// included, in category order), or the requested `groups` in request order
    /// with the reference appended when one is named — scanpy's `groups_order`.
    pub(crate) group_names: Vec<String>,
    /// `[group][gene]`, parallel to `group_names`, over every analysed gene.
    pub(crate) pts: Vec<Vec<f64>>,
    /// `[group][gene]`; `None` when a reference group was named (scanpy emits
    /// `pts_rest` only for `reference="rest"`).
    pub(crate) pts_rest: Option<Vec<Vec<f64>>>,
}

/// One `rank_genes_groups` run: the result (already restricted to `groups=`
/// when given), the analysed gene names, and the `pts` tables when asked for.
pub(crate) struct RankGenesRun {
    pub(crate) result: scx_accel::DiffExpResult,
    pub(crate) gene_names: Vec<String>,
    pub(crate) pts: Option<PtsTables>,
}

/// Resolve labels, reference and `groups=`, run the kernels, then apply the
/// `groups=` output filter and the `pts` pass.
///
/// `requested_groups` restricts which groups are *reported*, never which cells
/// take part: every group's statistic is computed against the same pool
/// (1-vs-rest keeps every other cell in "rest", unlabelled ones included;
/// pairwise compares against the named reference; BH is per group), so a
/// group's numbers are identical with or without the restriction — and
/// identical to scanpy's, whose `groups=` works the same way. Relabelling the
/// unselected groups as unlabelled before the kernel would have changed what
/// "rest" means.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_rank_genes_groups_inner(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    groupby: &str,
    reference: &str,
    gene_chunk_size: Option<usize>,
    rankby_abs: bool,
    tie_correct: bool,
    prefer_format: &str,
    device: &str,
    gpu_device_id: Option<usize>,
    use_raw: bool,
    layer: Option<&str>,
    pts: bool,
    requested_groups: Option<&[String]>,
    // The public entry point that called this, for error messages only. Two
    // ops share this kernel — `rank_genes_groups` and `rank_genes_groups_df`'s
    // compute mode — and a refusal naming the other one sends the caller
    // looking at a function they did not call.
    op: &str,
) -> PyResult<RankGenesRun> {
    // Extract group labels from adata.obs[groupby].
    let obs = adata.getattr("obs")?;
    let group_col = single_obs_column(obs.get_item(groupby)?, groupby)?;
    let group_labels: Vec<String> = group_col
        .call_method1("astype", ("str",))?
        .call_method0("tolist")?
        .extract()?;

    // Which cells have no label, and what the levels are. Both come from
    // `pandas` rather than from how a value prints — see `missing_label_mask`:
    // `astype("str")` turns `None` into `"None"` and `pd.NA` into `"<NA>"`, so
    // a spelling test would both miss those and steal a real group named
    // `"None"`. That matters more since X8, because the singlet guard counts
    // cells per level and a phantom level has none.
    let missing = missing_label_mask(py, &group_col, group_labels.len())?;
    let unique_groups = group_level_universe(&group_col, &group_labels, &missing)?;

    // Map labels → indices.
    let group_name_to_idx: std::collections::HashMap<&str, usize> = unique_groups
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();

    // Missing values, and any label outside the universe, map to a sentinel
    // >= n_groups so the Wilcoxon kernels give them no group of their own,
    // instead of contaminating group 0. They still rank, and still count as
    // "rest", in the 1-vs-rest arm. Mirrors resolve_groups_and_reference.
    let groups = encode_group_labels(
        &group_labels,
        &group_name_to_idx,
        unique_groups.len(),
        &missing,
    );
    warn_unlabelled_cells(
        py,
        &groups,
        unique_groups.len(),
        groupby,
        reference == "rest",
    );

    // Resolve reference.
    let ref_idx: Option<usize> = if reference == "rest" {
        None
    } else {
        Some(*group_name_to_idx.get(reference).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "reference group '{reference}' not found in adata.obs['{groupby}']"
            ))
        })?)
    };

    // `groups=`: which groups are *reported*. Validated against the label
    // universe now; applied to the result after the dispatch (see the doc
    // above). The reference is never a tested group — scanpy drops it from the
    // request silently and keeps it as a `pts` column, so do the same.
    let named_reference = (reference != "rest").then_some(reference);

    // scanpy refuses any *participating* group with fewer than two cells
    // ("… since they only contain one sample"), and a participating group is
    // every level when `groups=` is omitted — `select_groups(adata, "all", …)`
    // returns `cat.categories`, and `value_counts()` reports an unused level as
    // 0, which is also `< 2`. So the check cannot live inside the `groups=`
    // branch: doing so made the default call — the one nobody is watching —
    // return finite, plausible scores computed from one cell (X8).
    //
    // Sizes are counted over the encoded labels, so an off-category value is a
    // sentinel here and not a member of any level, exactly as it is downstream.
    let group_sizes = {
        let mut sizes = vec![0usize; unique_groups.len()];
        for &code in &groups {
            if code < unique_groups.len() {
                sizes[code] += 1;
            }
        }
        sizes
    };
    let refuse_singletons = |participating: &[&str]| -> PyResult<()> {
        let too_small: Vec<&str> = participating
            .iter()
            .copied()
            .filter(|name| group_sizes[group_name_to_idx[*name]] < 2)
            .collect();
        if too_small.is_empty() {
            return Ok(());
        }
        // Naming the remedy matters for the zero-cell case, which is almost
        // always a subset that kept its parent's categories rather than a
        // genuinely tiny group.
        let empties: Vec<&str> = too_small
            .iter()
            .copied()
            .filter(|name| group_sizes[group_name_to_idx[*name]] == 0)
            .collect();
        let remedy = if empties.is_empty() {
            String::new()
        } else {
            format!(
                " ({} has no cells at all; if that is a level left over from a subset, run \
                 adata.obs[{groupby:?}] = adata.obs[{groupby:?}].cat.remove_unused_categories())",
                empties.join(", ")
            )
        };
        Err(PyValueError::new_err(format!(
            "Could not calculate statistics for groups {} since they only contain one \
             sample.{remedy}",
            too_small.join(", ")
        )))
    };

    let tested_groups: Option<Vec<String>> = match requested_groups {
        None => {
            let all: Vec<&str> = unique_groups.iter().map(String::as_str).collect();
            refuse_singletons(&all)?;
            None
        }
        Some(req) => {
            // Shape (non-empty, no repeats) was checked at the entry point, so
            // the stratified path fails once, up front, not once per stratum.
            let mut tested = Vec::with_capacity(req.len());
            for name in req {
                if !unique_groups.iter().any(|g| g == name) {
                    return Err(PyValueError::new_err(format!(
                        "groups: {name:?} is not a level of adata.obs[{groupby:?}]; \
                         available: {unique_groups:?}"
                    )));
                }
                if named_reference == Some(name.as_str()) {
                    continue;
                }
                tested.push(name.clone());
            }
            if tested.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "groups={req:?} leaves no group to test: it names only the reference \
                     group {reference:?}"
                )));
            }
            // The requested set, not the whole universe: `groups=` is an output
            // filter, so a singleton level the caller did not ask about is not
            // a group this run reports on. Same rule as scanpy's, whose
            // `groups_order` is the request when one is given.
            let participating: Vec<&str> = tested
                .iter()
                .map(String::as_str)
                .chain(named_reference)
                .collect();
            refuse_singletons(&participating)?;
            Some(tested)
        }
    };
    // scanpy's `groups_order` for the `pts` frame: every level (reference in
    // place) when nothing was requested, else the request plus the reference.
    let pts_groups: Vec<String> = match requested_groups {
        None => unique_groups.clone(),
        Some(req) => {
            let mut cols = req.to_vec();
            if let Some(r) = named_reference {
                if !cols.iter().any(|g| g == r) {
                    cols.push(r.to_string());
                }
            }
            cols
        }
    };

    // Select the input matrix + gene names per the use_raw/layer contract
    // (adata.X by default; adata.raw.X with raw var names for use_raw; a named
    // layer otherwise). The selected matrix flows through the same dispatch.
    let (x, gene_names) = select_de_matrix(adata, use_raw, layer, op)?;

    // `pts` is indexed by var name and `rank_genes_groups_df` joins on it, so
    // the names must be unique: with a duplicate, any by-name lookup would hand
    // one gene's fraction to the other (and scanpy's own merge multiplies the
    // rows). Checked here, before the kernels run, so a bad index does not pay
    // for a full rank-sum first — and named for the matrix that was actually
    // selected: `adata.var_names_make_unique()` fixes `X` / a layer, but leaves
    // `adata.raw.var_names` alone.
    if pts {
        let mut seen = std::collections::HashSet::with_capacity(gene_names.len());
        if let Some(dup) = gene_names.iter().find(|name| !seen.insert(name.as_str())) {
            let remedy = if use_raw {
                "the names come from adata.raw.var_names, which \
                 adata.var_names_make_unique() does not touch — pass use_raw=False, or rebuild \
                 raw from an AnnData whose var_names are unique"
            } else {
                "run adata.var_names_make_unique() first"
            };
            return Err(PyValueError::new_err(format!(
                "pts=True needs unique var names but {dup:?} occurs more than once; {remedy}"
            )));
        }
    }

    // Auto-detect whether data has been log-transformed (sc.pp.log1p sets
    // adata.uns["log1p"], and so does pyscx.accel.log1p). When true, logFC uses
    // the expm1 back-transform to match scanpy's formula. Deliberately the
    // annotation alone, with no value heuristic: scanpy keys off `uns` too, so
    // adding one here would create a divergence rather than close one.
    let log_transformed = crate::accel::util::uns_log1p_present(adata);

    let mut result = dispatch_rank_genes_kernels(
        py,
        &x,
        &gene_names,
        &groups,
        &unique_groups,
        ref_idx,
        log_transformed,
        gene_chunk_size,
        rankby_abs,
        tie_correct,
        prefer_format,
        device,
        gpu_device_id,
    )?;

    if let Some(tested) = &tested_groups {
        result = result
            .restrict_to_groups(tested)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    }

    // `pts`: one more pass over the same matrix, counting nonzeros per
    // (group, gene). Route-independent by construction — the count never
    // enters a kernel, so CSC-direct and the GPU drivers report the same
    // number as the dense CPU path.
    let pts_tables = if pts {
        let counts =
            compute_group_nonzero_counts(py, &x, &groups, unique_groups.len(), gene_names.len())?;
        let fractions = counts.fractions(ref_idx);
        // `pts_groups` was validated against the label universe above, so the
        // lookup cannot miss.
        let pick = |table: &Vec<Vec<f64>>| -> Vec<Vec<f64>> {
            pts_groups
                .iter()
                .map(|g| table[group_name_to_idx[g.as_str()]].clone())
                .collect()
        };
        Some(PtsTables {
            pts: pick(&fractions.pts),
            pts_rest: fractions.pts_rest.as_ref().map(pick),
            group_names: pts_groups,
        })
    } else {
        None
    };

    Ok(RankGenesRun {
        result,
        gene_names,
        pts: pts_tables,
    })
}

/// Nonzero counts per (group, gene) over the matrix the DE ran on, derived
/// from `x` in the same order the dispatcher recognises it: backed handle,
/// lazy handle, scipy sparse, dense. The in-memory arms take an owned copy
/// (the sanctioned way to read a buffer with the GIL released — see
/// `crate::convert::owned_csr`); the streamed arms decode every shard again —
/// the default four-shard LRU does not keep a full sequential scan resident, so
/// this is a second read of `X`, not a cache hit.
fn compute_group_nonzero_counts(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    group_codes: &[usize],
    n_groups: usize,
    n_vars: usize,
) -> PyResult<scx_accel::GroupNonzeroCounts> {
    let to_err = |e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string());
    let counts = if let Some(handle) = crate::accel::backed_dataset_ref(x) {
        let backed = handle.get();
        let source = backed.as_shard_source().with_cached_reads();
        drop(handle);
        py.detach(|| scx_accel::group_nonzero_counts_streaming(&source, group_codes, n_groups))
            .map_err(to_err)?
    } else if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
        let source = lazy.as_shard_source().with_cached_reads();
        drop(lazy);
        py.detach(|| scx_accel::group_nonzero_counts_streaming(&source, group_codes, n_groups))
            .map_err(to_err)?
    } else {
        let scipy_sparse = crate::pyimport::import_module(py, "scipy.sparse")?;
        let is_sparse = scipy_sparse
            .call_method1("issparse", (x,))?
            .extract::<bool>()?;
        if is_sparse {
            let csr = crate::convert::owned_csr(py, x, Some("rank_genes_groups(pts=True)"))?;
            py.detach(|| scx_accel::group_nonzero_counts_csr(&csr, group_codes, n_groups))
                .map_err(to_err)?
        } else {
            let (data, (n_obs, n_cols)) =
                crate::convert::owned_dense2_f32(py, x, Some("rank_genes_groups(pts=True)"))?;
            py.detach(|| {
                scx_accel::group_nonzero_counts_dense(&data, n_obs, n_cols, group_codes, n_groups)
            })
            .map_err(to_err)?
        }
    };
    if counts.n_vars() != n_vars {
        return Err(PyRuntimeError::new_err(format!(
            "pts: the analysed matrix has {} columns but {n_vars} gene names",
            counts.n_vars()
        )));
    }
    Ok(counts)
}

/// The CPU CSC route stamp for `rank_genes_groups`.
///
/// `SCX_ACCEL_WILCOXON_NNZ=1` swaps in a structurally different kernel, but the
/// stamp used to read `cpu_csc` either way, so a benchmark timing the nnz path
/// could not tell from `uns["scx_accel"]` which kernel it had measured
/// (review §7.17). The kernel choice comes from
/// `scx_accel::csc_wilcoxon_uses_nnz_kernel`, the same predicate
/// `wilcoxon_rank_sum_streaming_csc` branches on — not a second copy of the
/// rule — so the two cannot disagree. Both CSC call sites below go through
/// here for the same reason.
fn csc_exec_info(
    device: &str,
    reference: Option<usize>,
    rankby_abs: bool,
    chunk_size: usize,
) -> scx_accel::route::AccelExecutionInfo {
    let mut info = crate::accel::route::cpu_exec_info(
        device,
        scx_accel::InputLayout::BackedCsc,
        false, // no GPU CSC kernel
        true,
        Some(chunk_size),
    );
    if scx_accel::csc_wilcoxon_uses_nnz_kernel(reference, rankby_abs) {
        info.route = scx_accel::AccelRoute::CpuCscNnz;
    }
    info
}

/// The kernel dispatch: pick the CPU / GPU, CSC-direct / CSR / in-memory arm
/// for `x` and run it. Knows nothing about `groups=` or `pts`.
#[allow(clippy::too_many_arguments)]
fn dispatch_rank_genes_kernels(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    gene_names: &[String],
    groups: &[usize],
    unique_groups: &[String],
    ref_idx: Option<usize>,
    log_transformed: bool,
    gene_chunk_size: Option<usize>,
    rankby_abs: bool,
    tie_correct: bool,
    prefer_format: &str,
    device: &str,
    gpu_device_id: Option<usize>,
) -> PyResult<scx_accel::DiffExpResult> {
    let numpy = crate::pyimport::import_module(py, "numpy")?;
    let scipy_sparse = crate::pyimport::import_module(py, "scipy.sparse")?;

    // Resolve the `"auto"` policy (§5.2) against the *selected* matrix: CSC-direct
    // on CPU when a valid sidecar is present, else CSR; CSR on GPU (the planner
    // routes gpu_csc_v3 from there). Explicit "csr"/"csc" pass through.
    let prefer_format = resolve_de_format(prefer_format, gpu_device_id, x);

    if prefer_format == "csc" {
        if gpu_device_id.is_some() {
            return Err(PyRuntimeError::new_err(
                "prefer_format='csc' selects the CPU column-major path and has no \
                 GPU kernel, so it cannot be combined with device='gpu'. For GPU \
                 CSC-direct DE (route gpu_csc_v3), keep prefer_format='csr' \
                 with device='gpu' (or 'auto'): when the file has a CSC sidecar the \
                 planner routes to gpu_csc_v3 automatically. For the CPU column-major \
                 path, use device='cpu'.",
            ));
        }
        // CSC dispatch: works on both backed and lazy datasets via
        // `as_column_source`, which hands back the same type for either — a
        // view that renumbers rows onto the live space and remaps columns into
        // the projected one. That is what lets a filtered or gene-subset
        // handle take this path; it used to be refused here outright. The
        // kernel reads each gene chunk as a CSC
        // slab once, scatters into a row-major dense buffer, and hands it to
        // the existing `wilcoxon_rank_sum` kernel.
        let chunk_size = gene_chunk_size.unwrap_or(500);

        // Resolve the source under the GIL; it is owned, so the kernel runs
        // detached. (`LazyShardSource` is `Send`, where a `&dyn
        // ColumnShardSource` was not — which is why the two arms used to
        // differ.)
        if !crate::accel::is_scx_matrix_handle(x) {
            return Err(PyRuntimeError::new_err(
                "prefer_format='csc' requires adata.X to be a backed or lazy SCX \
                 dataset; got a regular scipy/dense matrix",
            ));
        }
        let source = crate::accel::csc_source_for(x).ok_or_else(crate::accel::csc_unavailable)?;

        let result = py
            .detach(|| {
                scx_accel::wilcoxon_rank_sum_streaming_csc(
                    &source,
                    gene_names,
                    groups,
                    unique_groups,
                    ref_idx,
                    chunk_size,
                    log_transformed,
                    rankby_abs,
                    tie_correct,
                )
            })
            .map(|mut r| {
                r.exec_info = csc_exec_info(device, ref_idx, rankby_abs, chunk_size);
                r
            })
            .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?;
        return Ok(result);
    }

    let result = if let Some(handle) = crate::accel::backed_dataset_ref(x) {
        let backed = handle.get();
        // Backed mode: stream shards with gene-chunked DE. Clone the Arc
        // so the kernel runs without holding the GIL.
        let chunk_size = gene_chunk_size.unwrap_or(500);
        // The handle's *view*, not the file: `groups` is one label per
        // *visible* cell and `gene_names` one per visible gene, so a subset
        // handle has to stream its window. `with_cached_reads` because the
        // kernel walks every shard once per gene chunk — the same LRU the raw
        // reader served from.
        let source = backed.as_shard_source().with_cached_reads();
        // Only the GPU arm still needs the concrete reader (for the
        // CSC-direct `Backed` input); the CPU arm runs entirely off `source`.
        #[cfg(feature = "gpu")]
        let reader = std::sync::Arc::clone(&backed.backed);
        #[cfg(feature = "gpu")]
        let has_view = backed.has_axis_view();
        // If a CSC sidecar reader exists on the dataset, hand it to the GPU
        // streaming path so the default v3 route can dispatch to the
        // CSC-direct Wilcoxon driver. None falls through to the v3 CSR-direct
        // fallback. Mirrors the pdex_ref backed path. Only bind under the gpu
        // feature — the CPU branch takes no CSC.
        #[cfg(feature = "gpu")]
        let csc_reader = backed.backed_csc.as_ref().map(std::sync::Arc::clone);
        drop(handle);
        match gpu_device_id {
            #[cfg(feature = "gpu")]
            Some(device_id) => py
                .detach(|| {
                    // Which CSC source to hand over, not whether to hand
                    // one over. Neither CSC gate elsewhere protects this
                    // path: `csc_reader` comes straight off
                    // `backed.backed_csc`, bypassing `as_column_source()`,
                    // and `csc_route_available` is never consulted on GPU
                    // (`resolve_de_format` short-circuits to "csr" whenever
                    // `gpu_device_id.is_some()`). That reader is the *file* —
                    // full row axis, full column axis — so under a subset
                    // handle's visible-width `gene_names` it would read the
                    // wrong columns, and silently, since `Backed::shape()`
                    // takes `n_vars` from `gene_names.len()` and the widths
                    // agree. This used to be handled by giving the route up
                    // and passing the CSR-shaped `Lazy` input. It no longer
                    // has to be: `source` is the window on both axes — rows
                    // renumbered onto the live space, columns remapped into
                    // the projected one — and serves as both the CSR and the
                    // CSC side, so a filtered or gene-subset handle keeps
                    // `gpu_csc_v3`. `Lazy` remains for the case with no
                    // sidecar to offer — and for a window so narrow that CSC's
                    // full-height column reads would lose to the shard
                    // skipping the CSR path can do
                    // (`csc_preferred_for_auto`), since on GPU the route is
                    // chosen by the planner rather than requested.
                    let input = if !has_view {
                        scx_accel::GpuDeShardInput::Backed {
                            csr: reader.as_ref(),
                            csc: csc_reader
                                .as_deref()
                                .map(|c| c as &(dyn scx_format_io::ColumnShardSource + Sync)),
                        }
                    } else if source.csc_preferred_for_auto() {
                        scx_accel::GpuDeShardInput::Backed {
                            csr: &source,
                            csc: Some(&source),
                        }
                    } else {
                        scx_accel::GpuDeShardInput::Lazy(&source)
                    };
                    scx_accel::wilcoxon_rank_sum_gpu(
                        device_id,
                        input,
                        gene_names,
                        groups,
                        unique_groups,
                        ref_idx,
                        Some(chunk_size),
                        log_transformed,
                        rankby_abs,
                        tie_correct,
                    )
                })
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?,
            #[cfg(not(feature = "gpu"))]
            Some(_) => unreachable!("gpu_device_id is None when gpu feature is disabled"),
            None => py
                .detach(|| {
                    scx_accel::wilcoxon_rank_sum_streaming(
                        &source,
                        gene_names,
                        groups,
                        unique_groups,
                        ref_idx,
                        chunk_size,
                        log_transformed,
                        rankby_abs,
                        tie_correct,
                    )
                })
                .map(|mut r| {
                    r.exec_info = crate::accel::route::cpu_exec_info(
                        device,
                        scx_accel::InputLayout::BackedCsr,
                        true,
                        false,
                        Some(chunk_size),
                    );
                    r
                })
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?,
        }
    } else {
        // ScxLazyTransformedDataset — route through `wilcoxon_rank_sum_gpu`
        // (device-resident shard pipeline) before falling through to the CPU
        // lazy streamer.
        //
        // `Backed` when the chain carries a sidecar, `Lazy` when it does not.
        // This arm was CSR-only for as long as `Backed` named the concrete
        // full-axis readers, which is the shape a transformed handle can never
        // present — so `normalize_total → log1p` on GPU gave up `gpu_csc_v3`
        // even on a file with a sidecar, which is every real pipeline. The
        // source renumbers rows, remaps columns and applies the chain
        // column-major, so it can serve both sides.
        #[cfg(feature = "gpu")]
        {
            if let Some(device_id) = gpu_device_id {
                if let Ok(lazy) =
                    x.extract::<PyRef<crate::lazy_transform::ScxLazyTransformedDataset>>()
                {
                    let chunk_size = gene_chunk_size.unwrap_or(500);
                    let lazy_src = lazy.as_shard_source();
                    drop(lazy);
                    let input = if lazy_src.csc_preferred_for_auto() {
                        scx_accel::GpuDeShardInput::Backed {
                            csr: &lazy_src,
                            csc: Some(&lazy_src),
                        }
                    } else {
                        scx_accel::GpuDeShardInput::Lazy(&lazy_src)
                    };
                    let result = py
                        .detach(|| {
                            scx_accel::wilcoxon_rank_sum_gpu(
                                device_id,
                                input,
                                gene_names,
                                groups,
                                unique_groups,
                                ref_idx,
                                Some(chunk_size),
                                log_transformed,
                                rankby_abs,
                                tie_correct,
                            )
                        })
                        .map_err(|e: scx_accel::AccelError| {
                            PyRuntimeError::new_err(e.to_string())
                        })?;
                    return Ok(result);
                }
            }
        }

        // CPU lazy: stream the transformed shards, exactly as the backed arm
        // streams untransformed ones. Without this the chain falls through to
        // the scipy/numpy branch below, which cannot coerce a lazy dataset —
        // `open(...) → accel.log1p → rank_genes_groups(device="cpu")` died on
        // `ValueError: setting an array element with a sequence` rather than
        // running. (A lazy X with a usable CSC sidecar was already served by
        // the `prefer_format="csc"` branch above; this is the CSR case.)
        if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
            let chunk_size = gene_chunk_size.unwrap_or(500);
            let lazy_src = lazy.as_shard_source().with_cached_reads();
            drop(lazy);
            let result = py
                .detach(|| {
                    scx_accel::wilcoxon_rank_sum_streaming(
                        &lazy_src,
                        gene_names,
                        groups,
                        unique_groups,
                        ref_idx,
                        chunk_size,
                        log_transformed,
                        rankby_abs,
                        tie_correct,
                    )
                })
                .map(|mut r| {
                    r.exec_info = crate::accel::route::cpu_exec_info(
                        device,
                        scx_accel::InputLayout::LazyCsr,
                        true,
                        false,
                        Some(chunk_size),
                    );
                    r
                })
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?;
            return Ok(result);
        }

        let is_sparse = scipy_sparse
            .call_method1("issparse", (x,))?
            .extract::<bool>()?;

        if is_sparse {
            // Sparse in-memory: extract CSR arrays and use gene-chunked path
            // to avoid O(n_obs × n_vars) dense materialization.
            //
            // `ensure_csr` short-circuits when the input is already CSR with
            // sorted indices (the common h5ad case) and copies-then-sorts
            // only when the caller's CSR has `has_sorted_indices == False`.
            // Sorted indices are a precondition of
            // `scx_engine::project_csr_row`; without this step the CPU
            // sparse path silently returns U = n_g·n_ref/2 for every gene
            // on inputs with unsorted CSRs (e.g. pbmc10k.h5ad).
            let (csr_obj, _) = crate::convert::ensure_csr(py, x, /* in_place */ false)?;
            let shape: (usize, usize) = csr_obj.getattr("shape")?.extract()?;
            let np = crate::pyimport::import_module(py, "numpy")?;
            let indptr: Vec<i64> = np
                .call_method1("asarray", (csr_obj.getattr("indptr")?,))?
                .call_method1("astype", ("int64",))?
                .extract::<Vec<i64>>()?;
            let indices: Vec<i32> = np
                .call_method1("asarray", (csr_obj.getattr("indices")?,))?
                .call_method1("astype", ("int32",))?
                .extract::<Vec<i32>>()?;
            let data: Vec<f32> = np
                .call_method1("asarray", (csr_obj.getattr("data")?,))?
                .call_method1("astype", ("float32",))?
                .extract::<Vec<f32>>()?;

            let csr = scx_sparse::ScxCsr::new_unchecked(shape, indptr, indices, data);
            let chunk_size = gene_chunk_size.unwrap_or(500);
            match gpu_device_id {
                #[cfg(feature = "gpu")]
                Some(device_id) => py
                    .detach(|| {
                        scx_accel::wilcoxon_rank_sum_gpu(
                            device_id,
                            scx_accel::GpuDeShardInput::Csr(&csr),
                            gene_names,
                            groups,
                            unique_groups,
                            ref_idx,
                            Some(chunk_size),
                            log_transformed,
                            rankby_abs,
                            tie_correct,
                        )
                    })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                #[cfg(not(feature = "gpu"))]
                Some(_) => unreachable!("gpu_device_id is None when gpu feature is disabled"),
                None => py
                    .detach(|| {
                        scx_accel::wilcoxon_rank_sum_sparse(
                            &csr,
                            gene_names,
                            groups,
                            unique_groups,
                            ref_idx,
                            chunk_size,
                            log_transformed,
                            rankby_abs,
                            tie_correct,
                        )
                    })
                    .map(|mut r| {
                        r.exec_info = crate::accel::route::cpu_exec_info(
                            device,
                            scx_accel::InputLayout::CsrHost,
                            true,
                            false,
                            Some(chunk_size),
                        );
                        r
                    })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            }
        } else {
            // Dense numpy array: flatten and use direct wilcoxon_rank_sum
            let dense = numpy
                .call_method1("asarray", (x,))?
                .call_method1("astype", ("float32",))?;
            let shape: (usize, usize) = dense.getattr("shape")?.extract()?;
            let (n_obs, n_vars) = shape;

            let flat = dense.call_method0("ravel")?;
            let data: Vec<f32> = flat.extract()?;

            match gpu_device_id {
                #[cfg(feature = "gpu")]
                Some(device_id) => py
                    .detach(|| {
                        scx_accel::wilcoxon_rank_sum_gpu_dense(
                            device_id,
                            &data,
                            n_obs,
                            n_vars,
                            gene_names,
                            groups,
                            unique_groups,
                            ref_idx,
                            log_transformed,
                            rankby_abs,
                            tie_correct,
                        )
                    })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                #[cfg(not(feature = "gpu"))]
                Some(_) => unreachable!("gpu_device_id is None when gpu feature is disabled"),
                None => py
                    .detach(|| {
                        scx_accel::wilcoxon_rank_sum(
                            &data,
                            n_obs,
                            n_vars,
                            gene_names,
                            groups,
                            unique_groups,
                            ref_idx,
                            log_transformed,
                            rankby_abs,
                            tie_correct,
                            0,
                        )
                    })
                    .map(|mut r| {
                        r.exec_info = crate::accel::route::cpu_exec_info(
                            device,
                            scx_accel::InputLayout::DenseHost,
                            true,
                            false,
                            None,
                        );
                        r
                    })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            }
        }
    };

    Ok(result)
}

/// Convert a DiffExpResult into a pandas DataFrame.
///
/// Each group's genes become rows with columns: gene, scores, pvals, pvals_adj,
/// logfoldchanges, group.
pub(crate) fn de_result_to_dataframe<'py>(
    py: Python<'py>,
    result: &scx_accel::DiffExpResult,
    n_genes: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let pd = crate::pyimport::import_module(py, "pandas")?;
    let mut all_frames: Vec<Bound<'py, PyAny>> = Vec::new();

    for (i, group_name) in result.group_names.iter().enumerate() {
        let full_n_genes = result.names[i].len();
        let n = n_genes.unwrap_or(full_n_genes).min(full_n_genes);

        let names_slice = &result.names[i][..n];
        let scores_slice = &result.scores[i][..n];
        let pvals_slice = &result.pvals[i][..n];
        let pvals_adj_slice = &result.pvals_adj[i][..n];
        let logfc_slice = &result.logfoldchanges[i][..n];

        let dict = PyDict::new(py);
        dict.set_item("gene", names_slice.to_vec())?;
        dict.set_item("scores", scores_slice.to_vec())?;
        dict.set_item("pvals", pvals_slice.to_vec())?;
        dict.set_item("pvals_adj", pvals_adj_slice.to_vec())?;
        dict.set_item("logfoldchanges", logfc_slice.to_vec())?;
        dict.set_item("group", vec![group_name.clone(); n])?;

        let df = pd.call_method1("DataFrame", (dict,))?;
        all_frames.push(df);
    }

    if all_frames.is_empty() {
        // Return empty DataFrame with the right columns.
        let dict = PyDict::new(py);
        for col in [
            "gene",
            "scores",
            "pvals",
            "pvals_adj",
            "logfoldchanges",
            "group",
        ] {
            dict.set_item(col, pyo3::types::PyList::empty(py))?;
        }
        return pd.call_method1("DataFrame", (dict,));
    }

    let frame_list = pyo3::types::PyList::new(py, &all_frames)?;
    let combined = pd.call_method(
        "concat",
        (frame_list,),
        Some(&{
            let kw = PyDict::new(py);
            kw.set_item("ignore_index", true)?;
            kw
        }),
    )?;
    Ok(combined)
}

/// Wilcoxon rank-sum differential expression, scanpy-compatible.
///
/// Writes results to ``adata.uns["rank_genes_groups"]`` as scanpy-style
/// structured arrays. The chosen accelerator execution route is recorded both
/// as ``adata.uns["rank_genes_groups"]["scx_accel_route"]`` and under the
/// unified ``adata.uns["scx_accel"]["rank_genes_groups"]`` dict (keys:
/// ``route``, ``fallback_reason``, ``chunk_size``, ``csc_available``, ...).
/// Check ``route`` when comparing CPU vs GPU performance — GPU is fastest only
/// when the input layout matches the op. Returns ``None`` (results live on
/// ``adata.uns``); the stratified path (``stratify_by``) instead returns a
/// concatenated pandas DataFrame and does not write route metadata.
///
/// ``use_raw`` / ``layer`` select the analyzed matrix (scanpy semantics):
/// ``use_raw`` analyzes ``adata.raw.X`` (with ``adata.raw.var`` names),
/// ``layer`` analyzes ``adata.layers[layer]``, and they are mutually exclusive.
/// ``use_raw=None`` (default) resolves to ``True`` iff ``adata.raw`` exists and
/// ``layer`` is None, else ``False``. The resolved ``use_raw`` and ``layer`` are
/// written to ``adata.uns["rank_genes_groups"]["params"]``.
///
/// ``pts=True`` adds scanpy's ``uns[key]["pts"]`` — and ``["pts_rest"]`` when
/// ``reference="rest"`` — as ``genes × groups`` DataFrames (float64, indexed by
/// the analysed var names, every gene whatever ``n_genes`` says) of the fraction
/// of cells in the group with a nonzero value; ``rank_genes_groups_df`` then
/// appends ``pct_nz_group`` / ``pct_nz_reference``. It is one extra streaming
/// pass over the matrix and is route-independent (CSC-direct and GPU included).
/// ``pts_rest`` is scanpy's ``X[~mask_g]`` fraction — every other cell of the
/// matrix, cells with no ``groupby`` label included — so the table equals
/// scanpy's on partially labelled input too. Both frames round-trip through
/// ``from_anndata`` / ``to_anndata`` and h5ad via the uns DataFrame envelope.
///
/// ``groups`` restricts which groups are *reported*, in the given order; the
/// pool each group is compared against is unchanged, so its numbers equal the
/// unrestricted run's and scanpy's ``groups=``. Unknown names, repeats and an
/// empty list raise; the reference group is silently not tested (it stays a
/// ``pts`` column).
///
/// Any **participating** group with fewer than two cells raises scanpy's "only
/// contain one sample" error: every level when ``groups`` is omitted, the named
/// ones (plus a named reference) when it is given. An unused category counts as
/// zero cells and so raises too, as it does in scanpy — the message names
/// ``remove_unused_categories()``. Under ``stratify_by`` that, like any
/// per-stratum failure, warns and drops the stratum.
///
/// Cells with no ``groupby`` label — a pandas missing value (``NaN`` / ``None``
/// / ``pd.NA``, decided by ``pandas.isna`` and never by how the value prints)
/// or a value outside the column's categories — get no group, no result row and
/// no ``pts`` column, but for ``reference="rest"`` they are in the rank pool and
/// in every group's "rest", scanpy 1.12's rule. A pairwise run against a named
/// reference compares ``group ∪ reference``, so they take no part in it. A group
/// whose *name* merely looks like a missing value (``"nan"``, ``"None"``,
/// ``""``) is a real group and is kept.
///
/// ``pts=True`` refuses duplicate var names (its table is joined by
/// name). ``corr_method`` accepts only ``"benjamini-hochberg"`` and is
/// recorded in ``params``; any other value raises instead of silently
/// applying BH.
///
/// NOTE: the log-fold-change back-transform uses the ``adata.uns["log1p"]``
/// flag, which describes ``X``. For the conventional case (``.raw`` /
/// ``layer`` hold log-normalized data, like ``X``) this is correct; if ``.raw``
/// holds raw counts while ``X`` is log1p-transformed, the logFC is computed as
/// if the counts were log-space. Prefer ``pdex_ref`` (which exposes an explicit
/// ``is_log1p`` override) when analyzing a matrix whose transform state differs
/// from ``X``.
#[pyfunction]
#[pyo3(signature = (adata, groupby, reference="rest", n_genes=None, method="wilcoxon", gene_chunk_size=None, stratify_by=None, min_cells_per_stratum=50, rankby_abs=false, tie_correct=false, prefer_format="auto", device="auto", use_raw=None, layer=None, *, pts=false, groups=None, corr_method="benjamini-hochberg"))]
#[allow(clippy::too_many_arguments)]
pub fn rank_genes_groups(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    groupby: &str,
    reference: &str,
    n_genes: Option<usize>,
    method: &str,
    gene_chunk_size: Option<usize>,
    stratify_by: Option<Vec<String>>,
    min_cells_per_stratum: usize,
    rankby_abs: bool,
    tie_correct: bool,
    prefer_format: &str,
    device: &str,
    use_raw: Option<bool>,
    layer: Option<&str>,
    pts: bool,
    groups: Option<Vec<String>>,
    corr_method: &str,
) -> PyResult<Py<PyAny>> {
    if method != "wilcoxon" {
        return Err(PyRuntimeError::new_err(format!(
            "unsupported method '{method}': only 'wilcoxon' is currently supported"
        )));
    }
    // Fail loud rather than silently fall back to BH: a caller who asked for
    // Bonferroni and got BH-adjusted values would never find out from the
    // output. scanpy's Bonferroni is `np.minimum(pvals * n_genes, 1.0)`.
    if corr_method != "benjamini-hochberg" {
        return Err(PyValueError::new_err(format!(
            "corr_method={corr_method:?} is not supported: pyscx applies Benjamini-Hochberg \
             only (scanpy's default), so pass corr_method=\"benjamini-hochberg\" or omit it. \
             For Bonferroni compute np.minimum(uns[\"rank_genes_groups\"][\"pvals\"][group] * \
             n_genes, 1.0) from the raw p-values."
        )));
    }
    if pts && stratify_by.is_some() {
        return Err(PyValueError::new_err(
            "pts=True cannot be combined with stratify_by: the stratified path returns a \
             DataFrame and writes no adata.uns[\"rank_genes_groups\"], which is where the pts / \
             pts_rest tables live. Run per stratum without stratify_by, or drop pts.",
        ));
    }
    if let Some(req) = &groups {
        validate_groups_request(req)?;
    }
    if !matches!(prefer_format, "csr" | "csc" | "auto") {
        return Err(PyValueError::new_err(format!(
            "Invalid prefer_format={prefer_format:?}; expected 'auto', 'csr', or 'csc'"
        )));
    }
    // A presentation-ordered backed `X` (`preserve_var_order=True`) has no
    // `ShardSource` spelling: the source emits columns in sorted on-disk order
    // while `adata.var` — and so `gene_names` — stays in request order. Now
    // that this op streams the handle's *view*, the two widths match, so the
    // mismatch would be a silent gene/column permutation instead of a shape
    // error. Refuse, as the other streaming accel ops do.
    // Deliberately the no-var-guard prologue: `select_de_matrix` guards the
    // matrix this op will actually read, which is `adata.raw.X` or
    // `adata.layers[layer]` when either was asked for. Keeping the X-only
    // check here made the documented remedy impossible — materialise the
    // layer, and `rank_genes_groups(layer=…)` still refused because `adata.X` was
    // presentation-ordered, while the matrix it was about to read was fine.
    crate::accel::prepare_target_no_var_guard(py, adata, "rank_genes_groups")?;

    // Resolve scanpy's use_raw/layer contract once (mutual-exclusion + default).
    let resolved_use_raw = resolve_use_raw(adata, use_raw, layer)?;
    let resolved = crate::accel::gpu::resolve_device(device)?;
    #[cfg(feature = "gpu")]
    let gpu_device_id = resolved.gpu_id();
    #[cfg(not(feature = "gpu"))]
    let gpu_device_id: Option<usize> = {
        let _ = resolved;
        None
    };
    // `prefer_format="csc"` selects the CPU column-major path (it is a CPU-only
    // knob — the GPU CSC-direct route `gpu_csc_v3` is reached via the *default*
    // prefer_format="csr", chosen by the planner when a CSC sidecar is present).
    // Reject when the user explicitly asked for GPU; for device="auto" fall back
    // to CPU, but nudge on a GPU host so the silent CPU pin is not surprising.
    let gpu_device_id = if prefer_format == "csc" {
        if device.starts_with("gpu") {
            return Err(PyRuntimeError::new_err(
                "prefer_format='csc' selects the CPU column-major path and has no \
                 GPU kernel, so it cannot be combined with device='gpu'. For GPU \
                 CSC-direct DE (route gpu_csc_v3), keep prefer_format='csr' \
                 with device='gpu' (or 'auto'): when the file has a CSC sidecar the \
                 planner routes to gpu_csc_v3 automatically. For the CPU column-major \
                 path, use device='cpu'.",
            ));
        }
        if device == "auto" && crate::accel::route::gpu_available() {
            crate::pyimport::import_module(py, "warnings")?.call_method1(
                "warn",
                (
                    "rank_genes_groups(device=\"auto\", prefer_format=\"csc\") runs on the \
                     CPU: prefer_format=\"csc\" pins the CPU column-major path even on a GPU \
                     host. For GPU CSC-direct DE (route gpu_csc_v3), drop prefer_format \
                     (pass \"csr\" explicitly) with device=\"auto\"/\"gpu\" — the planner \
                     routes to gpu_csc_v3 automatically when a CSC sidecar is present.",
                    py.get_type::<pyo3::exceptions::PyUserWarning>(),
                ),
            )?;
        }
        None
    } else {
        gpu_device_id
    };

    // --- Stratified path ---
    if let Some(ref strat_cols) = stratify_by {
        let forbidden = vec![groupby];
        let (strata, masks) =
            extract_strata(py, adata, strat_cols, min_cells_per_stratum, &forbidden)?;

        let pd = crate::pyimport::import_module(py, "pandas")?;
        let warnings = crate::pyimport::import_module(py, "warnings")?;
        let mut all_frames: Vec<Bound<'_, PyAny>> = Vec::new();

        for (stratum, mask) in strata.iter().zip(masks.iter()) {
            // Subset adata by mask.
            let sub_adata = adata.get_item(mask)?;
            let sub_adata = sub_adata.call_method0("copy")?;

            // Run DE on the subset.
            match run_rank_genes_groups_inner(
                py,
                &sub_adata,
                groupby,
                reference,
                gene_chunk_size,
                rankby_abs,
                tie_correct,
                prefer_format,
                device,
                gpu_device_id,
                resolved_use_raw,
                layer,
                false,
                groups.as_deref(),
                "rank_genes_groups",
            ) {
                Ok(run) => {
                    let df = de_result_to_dataframe(py, &run.result, n_genes)?;
                    // Add stratum columns.
                    for (j, col_name) in strat_cols.iter().enumerate() {
                        df.set_item(col_name.as_str(), stratum.key[j].as_str())?;
                    }
                    all_frames.push(df);
                }
                Err(e) => {
                    let key_str = stratum.key.join(", ");
                    let msg = format!("DE failed for stratum [{}]: {}", key_str, e);
                    warnings.call_method1("warn", (msg,))?;
                }
            }
        }

        if all_frames.is_empty() {
            return Err(PyValueError::new_err(
                "all strata failed during stratified DE analysis",
            ));
        }

        let frame_list = pyo3::types::PyList::new(py, &all_frames)?;
        let combined = pd.call_method(
            "concat",
            (frame_list,),
            Some(&{
                let kw = PyDict::new(py);
                kw.set_item("ignore_index", true)?;
                kw
            }),
        )?;

        return Ok(combined.unbind());
    }

    // --- Non-stratified path (original behavior) ---
    let run = run_rank_genes_groups_inner(
        py,
        adata,
        groupby,
        reference,
        gene_chunk_size,
        rankby_abs,
        tie_correct,
        prefer_format,
        device,
        gpu_device_id,
        resolved_use_raw,
        layer,
        pts,
        groups.as_deref(),
        "rank_genes_groups",
    )?;
    let result = &run.result;

    // Write results to adata.uns["rank_genes_groups"] in scanpy format.
    write_de_to_adata(
        py,
        adata,
        result,
        groupby,
        reference,
        n_genes,
        resolved_use_raw,
        layer,
        corr_method,
        &run.gene_names,
        run.pts.as_ref(),
    )?;

    // Record the accelerator execution route: both inside the scanpy-style
    // rank_genes_groups dict (as `scx_accel_route`) and under the unified
    // adata.uns["scx_accel"]["rank_genes_groups"] lookup. `result.exec_info`
    // is already complete (route + reason) — set by the single planner on both
    // the CPU dispatch sites and inside scx-accel for GPU routes.
    if let Ok(rgg) = adata.getattr("uns")?.get_item("rank_genes_groups") {
        rgg.set_item("scx_accel_route", result.exec_info.route.as_str())?;
    }
    crate::accel::route::announce_route(py, "rank_genes_groups", device, &result.exec_info);
    crate::accel::route::warn_materialized_csc_sidecar(
        py,
        "rank_genes_groups",
        device,
        adata,
        &result.exec_info,
    );
    crate::accel::route::write_accel_route(py, adata, "rank_genes_groups", &result.exec_info)?;

    Ok(py.None())
}

/// Write DE results to adata.uns["rank_genes_groups"] matching scanpy's format.
///
/// Scanpy stores results as numpy structured arrays (rec.arrays) with one
/// field per group. Each field contains gene names/scores/p-values sorted
/// by the test statistic.
#[allow(clippy::too_many_arguments)]
fn write_de_to_adata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    result: &scx_accel::DiffExpResult,
    groupby: &str,
    reference: &str,
    n_genes: Option<usize>,
    use_raw: bool,
    layer: Option<&str>,
    corr_method: &str,
    gene_names: &[String],
    pts: Option<&PtsTables>,
) -> PyResult<()> {
    let numpy = crate::pyimport::import_module(py, "numpy")?;
    // The builder closures below iterate `result.group_names` (length n_groups)
    // and index `field_data[i]`, so every outer field vector must have at least
    // n_groups rows or that indexing panics. Rectangular by construction, but
    // validate explicitly so a malformed DiffExpResult returns an error instead
    // of panicking (the `full_n_genes` zip only bounds the *inner* lengths).
    let n_groups = result.group_names.len();
    if result.names.len() < n_groups
        || result.scores.len() < n_groups
        || result.pvals.len() < n_groups
        || result.pvals_adj.len() < n_groups
        || result.logfoldchanges.len() < n_groups
    {
        return Err(PyRuntimeError::new_err(
            "DiffExpResult has fewer per-field rows than groups (malformed result)",
        ));
    }
    // Rank depth = the SHORTEST per-group vector across all emitted fields, not
    // just group 0's. Groups are rectangular by construction
    // (`[n_groups][n_genes]`), so this is a no-op on well-formed results, but
    // clamping to the min keeps the numpy structured array rectangular and every
    // `[..n_genes]` slice in the builder closures below in-bounds even on a
    // malformed/degenerate DiffExpResult — avoids a PanicException. The zip also
    // stops at the shortest outer vec.
    let full_n_genes = result
        .names
        .iter()
        .zip(&result.scores)
        .zip(&result.pvals)
        .zip(&result.pvals_adj)
        .zip(&result.logfoldchanges)
        .map(|((((n, s), p), pa), l)| n.len().min(s.len()).min(p.len()).min(pa.len()).min(l.len()))
        .min()
        .unwrap_or(0);
    let n_genes = n_genes.unwrap_or(full_n_genes).min(full_n_genes);

    let rgg = PyDict::new(py);

    // params dict
    let params = PyDict::new(py);
    params.set_item("groupby", groupby)?;
    params.set_item("reference", reference)?;
    params.set_item("method", "wilcoxon")?;
    params.set_item("use_raw", use_raw)?;
    params.set_item("layer", layer)?;
    params.set_item("corr_method", corr_method)?;

    // scanpy's recarray dtype, built through the **dict** spelling
    // (`{"names": [...], "formats": [...]}`) rather than a list of
    // `(name, format)` tuples.
    //
    // The two are equivalent for every ordinary group name and differ on
    // exactly one: numpy silently renames an empty field name to a positional
    // `"f0"` in the tuple-list form, so a group legitimately labelled `""`
    // produced an array whose field could not be read back —
    // `ValueError: no field of name`, raised from inside the writer. The dict
    // form keeps the name. Reachable only since missing values stopped being
    // inferred from their printed form, which is what made `""` a real level.
    let build_dtype = |groups: &[String], format: &str| -> PyResult<Bound<'_, PyAny>> {
        let names = pyo3::types::PyList::empty(py);
        let formats = pyo3::types::PyList::empty(py);
        for gn in groups {
            names.append(gn.as_str())?;
            formats.append(format)?;
        }
        let spec = pyo3::types::PyDict::new(py);
        spec.set_item("names", names)?;
        spec.set_item("formats", formats)?;
        numpy.call_method1("dtype", (spec,))
    };

    // Helper to build structured array (like scanpy's recarray format).
    // Scanpy stores e.g. names as a structured array with dtype like:
    //   [('group_A', 'O'), ('group_B', 'O')]
    // Each row is one gene rank position.
    let build_structured =
        |field_data: &[Vec<String>], groups: &[String]| -> PyResult<Bound<'_, PyAny>> {
            // Object dtype, as scanpy writes it: no name is ever truncated (the
            // `pts` frame is indexed by the full var names and
            // `rank_genes_groups_df` joins the two by name — a fixed `U200`
            // silently produced NaN past 200 characters) and no single long
            // name widens every cell of every group (a fitted `U{max}` would
            // have cost `4 × max_len × n_groups × n_genes` bytes).
            let dtype = build_dtype(groups, "O")?;

            // Build empty structured array, then fill fields.
            let arr = numpy.call_method1("empty", (n_genes,))?;
            let arr = arr.call_method1("astype", (&dtype,))?;
            for (i, gn) in groups.iter().enumerate() {
                let vals = &field_data[i];
                let col = pyo3::types::PyList::new(py, &vals[..n_genes])?;
                arr.set_item(gn.as_str(), col)?;
            }
            Ok(arr.unbind().into_bound(py))
        };

    // One structured array per field, one column per group. `from_slice` hands
    // numpy the f64 buffer directly; the pre-4.3 spelling copied the slice into
    // a `Vec`, then pyo3 built a Python `list` of `PyFloat`s, then numpy parsed
    // that back — n_groups × n_genes objects per field, so a 30-group ×
    // 60k-gene run materialised millions of them purely in transit.
    //
    // The `names` builder above keeps its `PyList`: its object field genuinely
    // needs Python strings.
    let build_structured_f64 =
        |field_data: &[Vec<f64>], groups: &[String]| -> PyResult<Bound<'_, PyAny>> {
            let t_marshal = scx_accel::cpu_profile::start();
            let dtype = build_dtype(groups, "f8")?;

            let arr = numpy.call_method1("empty", (n_genes,))?;
            let arr = arr.call_method1("astype", (&dtype,))?;
            for (i, gn) in groups.iter().enumerate() {
                let np_vals = numpy::PyArray1::from_slice(py, &field_data[i][..n_genes]);
                arr.set_item(gn.as_str(), np_vals)?;
            }
            scx_accel::cpu_profile::record_marshalling_since(
                t_marshal,
                groups.len() * n_genes * std::mem::size_of::<f64>(),
            );
            Ok(arr.unbind().into_bound(py))
        };

    let names = build_structured(&result.names, &result.group_names)?;
    let scores = build_structured_f64(&result.scores, &result.group_names)?;
    let pvals = build_structured_f64(&result.pvals, &result.group_names)?;
    let pvals_adj = build_structured_f64(&result.pvals_adj, &result.group_names)?;
    let logfoldchanges = build_structured_f64(&result.logfoldchanges, &result.group_names)?;

    rgg.set_item("params", params)?;
    rgg.set_item("names", names)?;
    rgg.set_item("scores", scores)?;
    rgg.set_item("pvals", pvals)?;
    rgg.set_item("pvals_adj", pvals_adj)?;
    rgg.set_item("logfoldchanges", logfoldchanges)?;

    // `pts` / `pts_rest`: scanpy's shape exactly — a `genes × groups` DataFrame
    // indexed by the analysed var names (so `filter_rank_genes_groups`' `.loc`
    // lookups work), float64, every gene regardless of `n_genes`. The uns writer
    // has a `pandas.DataFrame` envelope, so these two keys survive
    // `from_anndata` and both h5ad directions unchanged.
    if let Some(tables) = pts {
        let pd = crate::pyimport::import_module(py, "pandas")?;
        let index = pyo3::types::PyList::new(py, gene_names)?;
        let frame = |values: &[Vec<f64>]| -> PyResult<Bound<'_, PyAny>> {
            let cols = PyDict::new(py);
            for (name, col) in tables.group_names.iter().zip(values) {
                if col.len() != gene_names.len() {
                    return Err(PyRuntimeError::new_err(format!(
                        "pts table for group {name:?} has {} entries but there are {} genes",
                        col.len(),
                        gene_names.len()
                    )));
                }
                cols.set_item(name.as_str(), numpy::PyArray1::from_slice(py, col))?;
            }
            let kw = PyDict::new(py);
            kw.set_item("index", &index)?;
            pd.call_method("DataFrame", (cols,), Some(&kw))
        };
        rgg.set_item("pts", frame(&tables.pts)?)?;
        if let Some(rest) = &tables.pts_rest {
            rgg.set_item("pts_rest", frame(rest)?)?;
        }
    }

    let uns = adata.getattr("uns")?;
    uns.set_item("rank_genes_groups", rgg)?;

    Ok(())
}
