//! Eager materialization bodies for `Experiment` — `to_anndata` and the
//! detection-bitmap / row-gather readers, split out of the single
//! `#[pymethods]` block (ORG-10.16-6).

// PyExperiment — lazy handle for SCX files

use numpy::PyArray1;

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn to_anndata_impl<'py>(
    exp: &PyExperiment,
    py: Python<'py>,
    backed: bool,
    cache_shards: usize,
    var_names: Option<Vec<String>>,
    obs_filter: Option<&str>,
    layers: Option<Vec<String>>,
    preserve_slots: bool,
    modality: Option<String>,
    eager: bool,
    memory_budget: Option<Bound<'_, PyAny>>,
    obsm: Option<Vec<String>>,
    preserve_var_order: bool,
    strict_var_names: bool,
    container: &str,
    data_dtype: Option<&str>,
    index_dtype: Option<&str>,
    allow_lossy: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let memory_budget_bytes = convert::parse_memory_budget(memory_budget.as_ref())?;
    // F3: resolve the container/dtype materialization plan. The default
    // (csr / f32 / i32) keeps the exact zero-copy path; any non-default plan
    // triggers a post-assembly retype of X (and layers).
    let plan = convert::build_plan(py, container, data_dtype, index_dtype, allow_lossy)?;
    // F3 Phase 1: dtype/container materialization applies to the eager
    // (in-memory) path only. Backed X is a lazy dataset and the device path
    // stays f32-native for now — reject a non-default plan loudly rather
    // than silently ignoring it.
    if !plan.is_default_csr_f32() && backed {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "container / data_dtype / index_dtype are only supported with backed=False \
             (eager materialization); backed reads produce a lazy f32 dataset",
        ));
    }
    if let Some(name) = modality.as_deref() {
        if !backed {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "to_anndata(modality=...) currently requires backed=True; \
                 use to_mudata() for eager multimodal extraction",
            ));
        }
        if var_names.is_some() || obs_filter.is_some() || layers.is_some() || obsm.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "to_anndata(modality=..., backed=True) does not support \
                 var_names / obs_filter / layers / obsm; use \
                 `scx subset --modality NAME --filter ...` to materialise \
                 a filtered single-modality file first",
            ));
        }
        return convert::to_anndata_backed_for_modality(py, &exp.path, name, cache_shards);
    }
    // NOTE: backed reads are *not* gated for the u32→f32 decode loss.
    // A backed AnnData decodes lazily per-slice, so a whole-catalog check
    // here would spuriously error on a partial read that never touches the
    // big-value shard. Gating the lazy-slice decode in the backed sparse
    // dataset is a deferred follow-up.
    if backed {
        convert::to_anndata_backed(
            py,
            &exp.path,
            cache_shards,
            var_names.as_deref(),
            obs_filter,
            layers.as_deref(),
            obsm.as_deref(),
            eager,
            preserve_var_order,
            strict_var_names,
        )
    } else {
        let adata = convert::to_anndata_filtered(
            py,
            &exp.path,
            exp.reader()?,
            var_names.as_deref(),
            obs_filter,
            layers.as_deref(),
            obsm.as_deref(),
            preserve_slots,
            eager,
            memory_budget_bytes,
            false,
            preserve_var_order,
            strict_var_names,
            &plan,
        )?;
        // F10: a materialized (in-memory CSR) AnnData drops the on-disk CSC
        // sidecar, so a later GPU DE call silently falls back to the slower
        // gpu_csr_v3 route. Stamp a hint when the source file has a sidecar so
        // the DE op can point the user at `to_anndata(backed=True)` (which
        // preserves the sidecar and engages gpu_csc_v3). See
        // `accel::route::warn_materialized_csc_sidecar`.
        //
        // Authoritative for the file just opened: set the hint when this file
        // has a sidecar, and *remove* any stale flag inherited from a prior
        // round-trip when it does not — so a sidecar-less file can never carry
        // a leftover `True` that would trigger a misleading warning.
        let uns = adata.getattr("uns")?;
        if exp.has_csc()? {
            uns.set_item("scx_source_has_csc_sidecar", true)?;
        } else if uns.contains("scx_source_has_csc_sidecar").unwrap_or(false) {
            let _ = uns.del_item("scx_source_has_csc_sidecar");
        }

        // X (and raw) now narrow **in-decode** inside
        // `to_anndata_filtered` (the typed reader assembles directly at the
        // target dtype — see `read_all_csr_shards_typed`), so no post-assembly
        // X retype is needed. **Layers** still materialize to f32 and are
        // narrowed here via `retype_matrix` (a DV-filtered typed layer reader
        // is a Phase-5 follow-up). No-op for the default plan.
        if !plan.is_default_csr_f32() {
            // Layers go through f32 before this cast, so a layer count > 2²⁴
            // cannot be delivered losslessly regardless of the requested
            // dtype — fail loud here rather than silently round (the eager
            // path guards the same value_max inside `to_anndata_with_layers`;
            // this covers the non-eager narrow path, which materializes layers
            // lazily via the retype loop below). Matches the X/raw fail-loud
            // contract; the exact `>2²⁴` layer read awaits the Phase-5 typed
            // layer reader.
            convert::guard_decode_loss(
                exp.reader()?.catalog().layer_csr_max_value(0, None),
                plan.allow_lossy,
            )?;
            let layers_obj = adata.getattr("layers")?;
            let mut keys: Vec<String> = Vec::new();
            for k in layers_obj.call_method0("keys")?.try_iter()? {
                keys.push(k?.extract()?);
            }
            for key in keys {
                let layer = layers_obj.get_item(&key)?;
                let new_layer = convert::retype_matrix(py, layer, &plan)?;
                layers_obj.set_item(&key, new_layer)?;
            }
        }
        Ok(adata)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn detection_counts_impl<'py>(
    exp: &PyExperiment,
    py: Python<'py>,
    axis: &str,
    modality: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    if axis != "var" {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "detection_counts: only axis='var' is supported (got '{axis}')"
        )));
    }
    // Re-opens from `exp.path`, and a fresh reader is fresh by definition
    // — so without this the handle would answer here while refusing
    // everywhere else. Gate on *this* handle's view first.
    exp.reader()?;
    let backed = open_backed_csr(&exp.path, modality, 4)?;
    let counts: Vec<i64> = py
        .detach(|| backed.gene_detection_counts())
        .map_err(to_pyerr)?
        .into_iter()
        .map(|c| c as i64)
        .collect();
    Ok(PyArray1::from_vec(py, counts))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn cells_expressing_impl<'py>(
    exp: &PyExperiment,
    py: Python<'py>,
    gene: &Bound<'_, PyAny>,
    modality: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    // Re-opens from `exp.path`, and a fresh reader is fresh by definition
    // — so without this the handle would answer here while refusing
    // everywhere else. Gate on *this* handle's view first.
    exp.reader()?;
    let backed = open_backed_csr(&exp.path, modality, 4)?;
    // Resolve gene → gene_idx. Integer fast path; string falls
    // through to a var.index lookup.
    let gene_idx: u32 = if let Ok(idx) = gene.extract::<u32>() {
        idx
    } else {
        let name: String = gene.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("gene must be an integer index or a string name")
        })?;
        resolve_gene_name(exp.reader()?, modality, &name)?
    };
    let rows = py
        .detach(|| backed.cells_expressing_gene(gene_idx))
        .map_err(to_pyerr)?;
    Ok(PyArray1::from_vec(py, rows))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gather_rows_sparse_impl<'py>(
    exp: &PyExperiment,
    py: Python<'py>,
    rows: PyReadonlyArray1<'_, u64>,
    modality: Option<&str>,
    cache_shards: usize,
) -> PyResult<Bound<'py, PyAny>> {
    // Snapshot the row ids while the GIL is held. A `PyReadonlyArray`
    // borrow is not GIL-bound and does not clear numpy's WRITEABLE flag,
    // so reading it inside the `py.detach` below would race any other
    // Python thread that touches the caller's array — and would defeat
    // the bounds check two statements down, which is the whole reason
    // this method can promise an `IndexError`.
    let rows: Vec<u64> = rows.as_slice()?.to_vec();
    let n_rows = rows.len();
    // Re-opens from `exp.path`, and a fresh reader is fresh by definition
    // — so without this the handle would answer here while refusing
    // everywhere else. Gate on *this* handle's view first.
    exp.reader()?;
    let backed = open_backed_csr(&exp.path, modality, cache_shards)?;
    let n_vars = backed.n_vars();
    let n_obs = backed.n_obs() as u64;

    // Bounds-check up front so out-of-range ids surface as a clean
    // IndexError rather than a generic read error from the gather loop.
    if let Some(&bad) = rows.iter().find(|&&r| r >= n_obs) {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "row index {bad} out of range for {n_obs} cells"
        )));
    }

    // Scatter each row's CSR into its request position (GIL released).
    let mut per_row: Vec<Option<(Vec<i32>, Vec<f32>)>> = (0..n_rows).map(|_| None).collect();
    py.detach(|| {
        backed.read_rows_with(&rows, |orig_pos, indices, data| {
            per_row[orig_pos] = Some((indices.to_vec(), data.to_vec()));
            Ok(())
        })
    })
    .map_err(to_pyerr)?;

    // Assemble the CSR in request order.
    let nnz: usize = per_row
        .iter()
        .map(|r| r.as_ref().map_or(0, |(idx, _)| idx.len()))
        .sum();
    let mut indptr: Vec<i64> = Vec::with_capacity(n_rows + 1);
    indptr.push(0);
    let mut indices: Vec<i32> = Vec::with_capacity(nnz);
    let mut data: Vec<f32> = Vec::with_capacity(nnz);
    for row in &per_row {
        if let Some((idx, val)) = row {
            indices.extend_from_slice(idx);
            data.extend_from_slice(val);
        }
        indptr.push(indices.len() as i64);
    }

    let csr = scx_sparse::ScxCsr::new_unchecked((n_rows, n_vars), indptr, indices, data);
    convert::csr_to_scipy(py, csr)
}
