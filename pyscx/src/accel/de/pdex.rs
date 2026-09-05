//! The pdex_ref parity kernel (MWU + pseudobulk log-fold-change) and its
//! `pdex_ref` entry point.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyDict;

use super::*;
use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::ScxLazyTransformedDataset;

/// Run pdex `mode="ref"` against an AnnData, dispatching to the SCX-backed
/// streaming, in-memory-CSR, or dense kernel based on `adata.X`.
#[allow(clippy::too_many_arguments)]
fn run_pdex_ref_inner(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    groupby: &str,
    reference: &str,
    geometric_mean: bool,
    is_log1p: Option<bool>,
    epsilon: f64,
    cpm_filter: Option<f64>,
    gene_chunk_size: Option<usize>,
    prefer_format: &str,
    device: &str,
    gpu_device_id: Option<usize>,
    use_raw: bool,
    layer: Option<&str>,
    requested_groups: Option<&[String]>,
) -> PyResult<scx_accel::PdexRefResult> {
    let numpy = crate::pyimport::import_module(py, "numpy")?;
    let scipy_sparse = crate::pyimport::import_module(py, "scipy.sparse")?;

    let (groups, unique_groups, ref_idx) =
        resolve_groups_and_reference(adata, groupby, reference, requested_groups)?;

    // Select the input matrix + gene names per the use_raw/layer contract.
    let (x, gene_names) = select_de_matrix(adata, use_raw, layer)?;

    // Resolve the `"auto"` policy (§5.2): CSC-direct on CPU when a valid sidecar
    // is present, else CSR; CSR on GPU (planner routes gpu_csc_v3 from there).
    let prefer_format = resolve_de_format(prefer_format, gpu_device_id, &x);

    let resolved_log1p = match is_log1p {
        Some(v) => v,
        None => detect_is_log1p(py, adata, &x)?,
    };
    let mode = scx_accel::GeomMeanMode::from_flags(geometric_mean, resolved_log1p);

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
        let chunk_size = gene_chunk_size.unwrap_or(500);

        if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
            reject_csc_on_subset(&backed)?;
            let csc_reader = backed
                .backed_csc
                .as_ref()
                .ok_or_else(|| {
                    PyRuntimeError::new_err(
                        "CSC requested but unavailable: file has no CSC sidecar",
                    )
                })?
                .clone();
            drop(backed);
            return py
                .detach(|| {
                    scx_accel::pdex_ref_streaming_csc(
                        csc_reader.as_ref(),
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        chunk_size,
                        mode,
                        epsilon,
                        cpm_filter,
                    )
                })
                .map(|mut r| {
                    r.exec_info = crate::accel::route::cpu_exec_info(
                        device,
                        scx_accel::InputLayout::BackedCsc,
                        false, // no GPU CSC kernel
                        true,
                        Some(chunk_size),
                    );
                    r
                })
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()));
        }
        if let Ok(lazy) = x.extract::<PyRef<crate::lazy_transform::ScxLazyTransformedDataset>>() {
            let lazy_src = lazy.as_column_source().ok_or_else(|| {
                PyRuntimeError::new_err(
                    "CSC requested but unavailable: file has no CSC sidecar, \
                     the transform chain contains a non-column-local op, or \
                     a row deletion vector is active",
                )
            })?;
            drop(lazy);
            return py
                .detach(|| {
                    scx_accel::pdex_ref_streaming_csc(
                        &lazy_src,
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        chunk_size,
                        mode,
                        epsilon,
                        cpm_filter,
                    )
                })
                .map(|mut r| {
                    r.exec_info = crate::accel::route::cpu_exec_info(
                        device,
                        scx_accel::InputLayout::BackedCsc,
                        false, // no GPU CSC kernel
                        true,
                        Some(chunk_size),
                    );
                    r
                })
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()));
        }
        return Err(PyRuntimeError::new_err(
            "prefer_format='csc' requires adata.X to be a backed or lazy SCX \
             dataset; got a regular scipy/dense matrix",
        ));
    }

    if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        let chunk_size = gene_chunk_size.unwrap_or(500);
        // The handle's *view* — see the matching branch in
        // `run_rank_genes_groups_inner` for why the raw reader is wrong here.
        let source = backed.as_shard_source().with_cached_reads();
        // GPU-only: the concrete reader backs the CSC-direct `Backed` input.
        #[cfg(feature = "gpu")]
        let reader = std::sync::Arc::clone(&backed.backed);
        #[cfg(feature = "gpu")]
        let has_view = backed.has_axis_view();
        // G4.3: if a CSC sidecar reader exists on the dataset, hand it to
        // the GPU streaming path so the default v3 route can dispatch to
        // the CSC-direct driver. None falls through to the v3 CSR-direct
        // fallback. Only bind under the gpu feature — the CPU branch doesn't
        // take a CSC reader.
        #[cfg(feature = "gpu")]
        let csc_reader = backed.backed_csc.as_ref().map(std::sync::Arc::clone);
        drop(backed);
        return match gpu_device_id {
            #[cfg(feature = "gpu")]
            Some(device_id) => py
                .detach(|| {
                    // Subset handle → generic `Lazy`; unsubset → `Backed`,
                    // preserving the CSC-direct `gpu_csc_v3` route. See the
                    // matching branch in `run_rank_genes_groups_inner`: this
                    // switch is what stops a subset handle running the
                    // CSC-direct kernel against on-disk columns.
                    let input = if has_view {
                        scx_accel::GpuDeShardInput::Lazy(&source)
                    } else {
                        scx_accel::GpuDeShardInput::Backed {
                            csr: &reader,
                            csc: csc_reader.as_deref(),
                        }
                    };
                    scx_accel::pdex_ref_gpu(
                        device_id,
                        input,
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        Some(chunk_size),
                        mode,
                        epsilon,
                        cpm_filter,
                    )
                })
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string())),
            #[cfg(not(feature = "gpu"))]
            Some(_) => unreachable!("gpu_device_id is None when gpu feature is disabled"),
            None => py
                .detach(|| {
                    scx_accel::pdex_ref_streaming(
                        &source,
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        chunk_size,
                        mode,
                        epsilon,
                        cpm_filter,
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
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string())),
        };
    }

    // G1.8: ScxLazyTransformedDataset (non-CSC GPU path). See the matching
    // branch in `run_rank_genes_groups_inner`.
    #[cfg(feature = "gpu")]
    if let Some(device_id) = gpu_device_id {
        if let Ok(lazy) = x.extract::<PyRef<crate::lazy_transform::ScxLazyTransformedDataset>>() {
            let chunk_size = gene_chunk_size.unwrap_or(500);
            let lazy_src = lazy.as_shard_source();
            drop(lazy);
            return py
                .detach(|| {
                    scx_accel::pdex_ref_gpu(
                        device_id,
                        scx_accel::GpuDeShardInput::Lazy(&lazy_src),
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        Some(chunk_size),
                        mode,
                        epsilon,
                        cpm_filter,
                    )
                })
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()));
        }
    }

    // CPU lazy: stream the transformed shards, as the backed arm above does
    // for untransformed ones. See the matching branch in
    // `run_rank_genes_groups_inner` — without it the review's own §7.2 repro
    // (`open(...) → accel.log1p → pdex_ref`) raised out of numpy instead of
    // producing a number to compare.
    if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
        let chunk_size = gene_chunk_size.unwrap_or(500);
        let lazy_src = lazy.as_shard_source().with_cached_reads();
        drop(lazy);
        return py
            .detach(|| {
                scx_accel::pdex_ref_streaming(
                    &lazy_src,
                    &gene_names,
                    &groups,
                    &unique_groups,
                    ref_idx,
                    chunk_size,
                    mode,
                    epsilon,
                    cpm_filter,
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
            .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()));
    }

    let is_sparse = scipy_sparse
        .call_method1("issparse", (&x,))?
        .extract::<bool>()?;

    if is_sparse {
        // `ensure_csr` enforces sorted column indices — required by
        // `scx_engine::project_csr_row`. Without this, unsorted scipy
        // CSR inputs silently return U = n_g·n_ref/2 for every gene.
        let (csr_obj, _) = crate::convert::ensure_csr(py, &x, /* in_place */ false)?;
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
                    scx_accel::pdex_ref_gpu(
                        device_id,
                        scx_accel::GpuDeShardInput::Csr(&csr),
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        Some(chunk_size),
                        mode,
                        epsilon,
                        cpm_filter,
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string())),
            #[cfg(not(feature = "gpu"))]
            Some(_) => unreachable!("gpu_device_id is None when gpu feature is disabled"),
            None => py
                .detach(|| {
                    scx_accel::pdex_ref_sparse(
                        &csr,
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        chunk_size,
                        mode,
                        epsilon,
                        cpm_filter,
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
                .map_err(|e| PyRuntimeError::new_err(e.to_string())),
        }
    } else {
        let dense = numpy
            .call_method1("asarray", (&x,))?
            .call_method1("astype", ("float32",))?;
        let shape: (usize, usize) = dense.getattr("shape")?.extract()?;
        let (n_obs, n_vars) = shape;
        let flat = dense.call_method0("ravel")?;
        let data: Vec<f32> = flat.extract()?;

        match gpu_device_id {
            #[cfg(feature = "gpu")]
            Some(device_id) => py
                .detach(|| {
                    scx_accel::pdex_ref_gpu_dense(
                        device_id,
                        &data,
                        n_obs,
                        n_vars,
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        mode,
                        epsilon,
                        cpm_filter,
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string())),
            #[cfg(not(feature = "gpu"))]
            Some(_) => unreachable!("gpu_device_id is None when gpu feature is disabled"),
            None => py
                .detach(|| {
                    let mut r = scx_accel::pdex_ref_core(
                        &data,
                        n_obs,
                        n_vars,
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        mode,
                        epsilon,
                        cpm_filter.is_some(),
                    )?;
                    // The dense kernel is not chunked, so apply the CPM filter +
                    // survivor-scoped FDR here (no-op when cpm_filter is None,
                    // beyond the redundant BH recompute we skip by guarding).
                    if cpm_filter.is_some() {
                        scx_accel::finalize_pdex(&mut r, cpm_filter);
                    }
                    Ok::<_, scx_accel::AccelError>(r)
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
                .map_err(|e| PyRuntimeError::new_err(e.to_string())),
        }
    }
}

/// Best-effort check for negative entries in `adata.X`, used only to warn when
/// `cpm_filter` is set (CPM assumes non-negative counts). Cheap in-memory paths
/// only: scipy sparse (`.data.min()`) and numpy ndarray (`.min()`). Backed/lazy
/// SCX datasets are skipped — SCX stores non-negative counts by construction, so
/// matching pdex's backed-array sampling here is unnecessary.
fn pdex_x_has_negative(py: Python<'_>, adata: &Bound<'_, PyAny>) -> bool {
    let Ok(x) = adata.getattr("X") else {
        return false;
    };
    if let Ok(scipy_sparse) = crate::pyimport::import_module(py, "scipy.sparse") {
        if let Ok(true) = scipy_sparse
            .call_method1("issparse", (&x,))
            .and_then(|r| r.extract::<bool>())
        {
            return x
                .getattr("data")
                .and_then(|d| d.call_method0("min"))
                .and_then(|m| m.extract::<f64>())
                .map(|m| m < 0.0)
                .unwrap_or(false);
        }
    }
    if let Ok(numpy) = crate::pyimport::import_module(py, "numpy") {
        if let Ok(ndarray_ty) = numpy.getattr("ndarray") {
            if let Ok(true) = x.is_instance(&ndarray_ty) {
                return x
                    .call_method0("min")
                    .and_then(|m| m.extract::<f64>())
                    .map(|m| m < 0.0)
                    .unwrap_or(false);
            }
        }
    }
    false
}

/// Convert a `PdexRefResult` to a polars DataFrame matching pdex's row schema.
///
/// Columns: `target`, `feature`, `target_mean`, `ref_mean`,
/// `target_membership`, `ref_membership`, `fold_change` (=log2_fold_change,
/// deprecated alias for migration), `log2_fold_change`, `percent_change`,
/// `p_value`, `statistic`, `fdr`.
fn pdex_ref_result_to_dataframe<'py>(
    py: Python<'py>,
    result: &scx_accel::PdexRefResult,
    output: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let n_genes = result.feature_names.len();
    let n_test = result.group_names.len();
    let total_rows = n_genes * n_test;

    let mut targets: Vec<String> = Vec::with_capacity(total_rows);
    let mut features: Vec<String> = Vec::with_capacity(total_rows);
    let mut target_means: Vec<f64> = Vec::with_capacity(total_rows);
    let mut ref_means: Vec<f64> = Vec::with_capacity(total_rows);
    let mut target_memberships: Vec<u64> = Vec::with_capacity(total_rows);
    let mut ref_memberships: Vec<u64> = Vec::with_capacity(total_rows);
    let mut log2_fcs: Vec<f64> = Vec::with_capacity(total_rows);
    let mut percent_changes: Vec<f64> = Vec::with_capacity(total_rows);
    let mut p_values: Vec<f64> = Vec::with_capacity(total_rows);
    let mut statistics: Vec<f64> = Vec::with_capacity(total_rows);
    let mut fdrs: Vec<f64> = Vec::with_capacity(total_rows);

    for (tg, group_name) in result.group_names.iter().enumerate() {
        // When `cpm_filter` dropped rows, `kept_indices[tg]` maps each surviving
        // per-group row back to its gene in the shared `feature_names` /
        // `ref_means` axis; the per-group vectors are already compacted to the
        // survivors. Without filtering, every group spans the full gene axis.
        let n_rows = result.target_means[tg].len();
        targets.extend(std::iter::repeat_n(group_name.clone(), n_rows));
        match result.kept_indices.as_ref() {
            Some(kept) => {
                for &gi in &kept[tg] {
                    features.push(result.feature_names[gi].clone());
                    ref_means.push(result.ref_means[gi]);
                }
            }
            None => {
                features.extend(result.feature_names.iter().cloned());
                ref_means.extend(result.ref_means.iter().copied());
            }
        }
        target_means.extend(result.target_means[tg].iter().copied());
        target_memberships.extend(std::iter::repeat_n(
            result.target_memberships[tg] as u64,
            n_rows,
        ));
        ref_memberships.extend(std::iter::repeat_n(result.ref_membership as u64, n_rows));
        log2_fcs.extend(result.log2_fold_changes[tg].iter().copied());
        percent_changes.extend(result.percent_changes[tg].iter().copied());
        p_values.extend(result.p_values[tg].iter().copied());
        statistics.extend(result.statistics[tg].iter().copied());
        fdrs.extend(result.fdrs[tg].iter().copied());
    }

    // pdex emits `fold_change` as a duplicate of `log2_fold_change` (a
    // deprecated alias retained for one release). Mirror that exactly so
    // downstream `DEResults` finds both columns.
    let fold_changes = log2_fcs.clone();

    let dict = PyDict::new(py);
    dict.set_item("target", targets)?;
    dict.set_item("feature", features)?;
    dict.set_item("target_mean", target_means)?;
    dict.set_item("ref_mean", ref_means)?;
    dict.set_item("target_membership", target_memberships)?;
    dict.set_item("ref_membership", ref_memberships)?;
    dict.set_item("fold_change", fold_changes)?;
    dict.set_item("log2_fold_change", log2_fcs)?;
    dict.set_item("percent_change", percent_changes)?;
    dict.set_item("p_value", p_values)?;
    dict.set_item("statistic", statistics)?;
    dict.set_item("fdr", fdrs)?;

    build_de_dataframe(
        py,
        &dict,
        &[
            "target",
            "feature",
            "target_mean",
            "ref_mean",
            "target_membership",
            "ref_membership",
            "fold_change",
            "log2_fold_change",
            "percent_change",
            "p_value",
            "statistic",
            "fdr",
        ],
        output,
    )
}

/// pdex `mode="ref"` differential expression on an SCX-backed or in-memory
/// AnnData, returned as a pandas DataFrame matching pdex's row schema
/// (`output="polars"` for the polars frame upstream pdex itself returns).
///
/// This is the SCX-native equivalent of `pdex.pdex(adata, groupby, mode="ref")`:
/// per (group, gene), reports pseudobulk means in natural (count) space, log2
/// fold change, percent change, Mann-Whitney U statistic, two-sided p-value,
/// and BH-adjusted FDR. The reference group is excluded from output.
///
/// Output columns: `target`, `feature`, `target_mean`, `ref_mean`,
/// `target_membership`, `ref_membership`, `fold_change` (=log2_fold_change,
/// deprecated alias retained for migration), `log2_fold_change`,
/// `percent_change`, `p_value`, `statistic`, `fdr`.
///
/// Args:
///     adata: AnnData object with X and obs[groupby]
///     groupby: Column in adata.obs to group cells by
///     reference: Reference group name (default: "non-targeting")
///     is_log1p: Whether adata.X contains log1p-transformed values. None
///         (default) auto-detects via adata.uns["log1p"] and a max-value
///         heuristic, matching pdex.
///     geometric_mean: If True (default), pseudobulk summary is the geometric
///         mean of expression values back-transformed to count space (matches
///         pdex's `geometric_mean=True`). If False, arithmetic mean is used.
///     epsilon: Pseudocount added to target_mean and ref_mean (count space)
///         before computing log2_fold_change and percent_change — NOT applied to
///         CPM or the Mann-Whitney U test. Default 1e-9 (a finite-guard matching
///         pdex >= 0.2.x): denominators stay strictly positive so no ±inf/NaN
///         arises. Pass epsilon=0.0 to recover the legacy behaviour where genes
///         undetected in the reference yield ±inf (0/0 collapses to 0.0 either
///         way).
///     cpm_filter: Optional per-gene expression floor T (counts-per-million).
///         When set, a gene is kept for a test group iff its pooled (arithmetic,
///         count-space) target CPM > T OR the reference CPM > T (strict >);
///         dropped rows are removed and FDR is recomputed over the surviving
///         genes only. The CPM view is mode-independent (always arithmetic) and
///         never reported. None (default) disables filtering.
///     gene_chunk_size: Genes per chunk for sparse/backed streaming
///         (default: 500). Ignored for dense input.
///     output: Return type — `"pandas"` (default) or `"polars"`. Columns are
///         identical either way; `"pandas"` builds a pandas DataFrame directly and
///         needs no optional dependency. `"polars"` requires the `eval` extra and
///         matches what upstream `pdex` and cell-eval use.
///     use_raw: Analyze `adata.raw.X` (with `adata.raw.var` names) instead of
///         `adata.X`. `None` (default) → `True` iff `adata.raw` is present and
///         `layer` is None (scanpy semantics), else `False`. Mutually exclusive
///         with `layer`. Recorded on `adata.uns["scx_accel"]["pdex_ref"]`.
///     layer: Analyze `adata.layers[layer]` instead of `adata.X`. Mutually
///         exclusive with `use_raw=True`.
///     groups: Restrict the tested targets to these `groupby` levels, reported
///         in this order (default: every level except `reference`, in category
///         order). Each selected target's row is identical to the unrestricted
///         run's — a target is only ever compared with the reference — while the
///         work (and GPU memory) scales with the number of targets asked for.
///         Unknown names, repeats, an empty list, and the reference itself are
///         errors. A pyscx extension: upstream pdex has no such knob.
///
/// The accelerator execution route is recorded on
/// ``adata.uns["scx_accel"]["pdex_ref"]`` (keys: ``route``,
/// ``fallback_reason``, ``chunk_size``, ``csc_available``, ...). The
/// CSC-direct GPU route (``route == "gpu_csc_v3"``) requires a backed SCX file
/// with a CSC sidecar (v3 is the default GPU DE route); in-memory CSR inputs
/// fall back to ``"gpu_csr_v3"`` with ``fallback_reason == "no_csc_sidecar"``. Check
/// ``route`` when comparing performance.
#[pyfunction]
#[pyo3(signature = (adata, groupby, *, reference="non-targeting", is_log1p=None, geometric_mean=true, epsilon=1e-9, cpm_filter=None, gene_chunk_size=None, prefer_format="auto", device="auto", output="pandas", use_raw=None, layer=None, groups=None))]
#[allow(clippy::too_many_arguments)]
pub fn pdex_ref(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    groupby: &str,
    reference: &str,
    is_log1p: Option<bool>,
    geometric_mean: bool,
    epsilon: f64,
    cpm_filter: Option<f64>,
    gene_chunk_size: Option<usize>,
    prefer_format: &str,
    device: &str,
    output: &str,
    use_raw: Option<bool>,
    layer: Option<&str>,
    groups: Option<Vec<String>>,
) -> PyResult<Py<PyAny>> {
    if let Some(req) = &groups {
        validate_groups_request(req)?;
    }
    if epsilon < 0.0 || !epsilon.is_finite() {
        return Err(PyValueError::new_err(format!(
            "epsilon must be non-negative and finite (got {epsilon})"
        )));
    }
    if let Some(t) = cpm_filter {
        if !t.is_finite() {
            return Err(PyValueError::new_err(format!(
                "cpm_filter must be finite (got {t})"
            )));
        }
        // pdex warns (does not error) when counts contain negatives, since CPM
        // assumes non-negative expression. Only checked for cheap in-memory X.
        if pdex_x_has_negative(py, adata) {
            crate::pyimport::import_module(py, "warnings")?.call_method1(
                "warn",
                (
                    "cpm_filter is set but adata.X contains negative values; \
                     counts-per-million assumes non-negative expression, so the \
                     filter may behave unexpectedly.",
                    py.get_type::<pyo3::exceptions::PyUserWarning>(),
                ),
            )?;
        }
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
    crate::accel::prepare_target(py, adata, "pdex_ref")?;

    if !matches!(output, "polars" | "pandas") {
        return Err(PyValueError::new_err(format!(
            "Invalid output={output:?}; expected 'polars' or 'pandas'"
        )));
    }
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
                    "pdex_ref(device=\"auto\", prefer_format=\"csc\") runs on the CPU: \
                     prefer_format=\"csc\" pins the CPU column-major path even on a GPU \
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
    let result = run_pdex_ref_inner(
        py,
        adata,
        groupby,
        reference,
        geometric_mean,
        is_log1p,
        epsilon,
        cpm_filter,
        gene_chunk_size,
        prefer_format,
        device,
        gpu_device_id,
        resolved_use_raw,
        layer,
        groups.as_deref(),
    )?;
    // Record the accelerator execution route on adata.uns["scx_accel"]["pdex_ref"].
    // `result.exec_info` is already complete (route + reason) from the single
    // planner — CPU sites via `cpu_exec_info`, GPU routes inside scx-accel.
    crate::accel::route::announce_route(py, "pdex_ref", device, &result.exec_info);
    crate::accel::route::warn_materialized_csc_sidecar(
        py,
        "pdex_ref",
        device,
        adata,
        &result.exec_info,
    );
    crate::accel::route::write_accel_route(py, adata, "pdex_ref", &result.exec_info)?;
    // Record the resolved data-selection contract alongside the route so callers
    // can see which matrix was analyzed (X / raw.X / a layer).
    // Defensive: `get_item` with `?` would raise KeyError if either key is
    // absent. `write_accel_route` just created both, but check both with
    // `if let Ok(...)` so a missing route entry never crashes the op.
    if let Ok(scx_accel) = adata.getattr("uns")?.get_item("scx_accel") {
        if let Ok(entry) = scx_accel.get_item("pdex_ref") {
            entry.set_item("use_raw", resolved_use_raw)?;
            entry.set_item("layer", layer)?;
        }
    }
    let df = pdex_ref_result_to_dataframe(py, &result, output)?;
    Ok(df.unbind())
}
