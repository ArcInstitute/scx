// SCX -> AnnData assembly (eager and backed).
//
// Extracted from the former pyscx/src/anndata.rs (T5.7).

use numpy::PyArray1;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use scx_format_io::section::SectionType;
use scx_format_io::ScxReader;

use crate::to_pyerr;

use super::*;

/// Default memory budget for the eager [`to_anndata`] full-assembly
/// path (Phase 4d). Estimated bytes above this threshold trigger a
/// `UserWarning` that recommends `to_anndata(backed=True)` or
/// `pyscx.open(path).query()`. Assembly still proceeds — the warning
/// is advisory.
const DEFAULT_EAGER_MEMORY_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Catalog-only estimate of the bytes required to assemble the full X
/// matrix plus obs / var metadata into an in-memory AnnData. Sums
/// `nnz × 16` for CSR shards (i32 indices + f32 data), `n_rows × 8`
/// for the assembled CSR indptr (i64), and the on-disk size of every
/// obs / var section (sharded or single). Walks `reader.catalog()`
/// only — no payload reads.
pub(crate) fn estimate_eager_assembly_bytes(reader: &ScxReader) -> u64 {
    let entries = &reader.catalog().entries;
    let mut nnz: u64 = 0;
    let mut x_rows: u64 = 0;
    for entry in entries {
        if entry.section_type != SectionType::CsrShard || entry.modality_id != 0 {
            continue;
        }
        if let Some(stats) = &entry.stats {
            nnz = nnz.saturating_add(stats.nnz);
            x_rows = x_rows.saturating_add(stats.row_end.saturating_sub(stats.row_start));
        }
    }
    let mut meta_bytes: u64 = 0;
    for entry in entries {
        match entry.section_type {
            SectionType::ObsMetadata
            | SectionType::VarMetadata
            | SectionType::ObsMetadataShard
            | SectionType::VarMetadataShard => {
                meta_bytes = meta_bytes.saturating_add(entry.length);
            }
            _ => {}
        }
    }
    nnz.saturating_mul(16)
        .saturating_add(x_rows.saturating_mul(8))
        .saturating_add(meta_bytes)
}

/// Build an AnnData object from an ScxReader with optional layer filtering.
///
/// `eager` controls how `obsp` / `varp` / `varm` / `layers` are
/// populated. When `false` (default for `pyscx.open(...).to_anndata()`),
/// these slots are wrapped in `ScxLazyPairwiseMapping` /
/// `ScxLazyVarmMapping` / `ScxLazyLayersMapping` and attached to the
/// AnnData's private `_obsp` / `_varp` / `_varm` / `_layers` storage,
/// deferring each section's decode until the consumer first accesses
/// `ad.obsp[…]` etc. Keeps peak RSS of `to_anndata()` bounded for
/// files that carry large kNN graphs / embeddings. When `true`, every
/// section is decoded up front and a plain `dict` is passed through
/// the AnnData constructor — matches pre-fix behaviour and detaches
/// the returned AnnData from the SCX file handle. See
/// [`crate::lazy_mapping`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn to_anndata_with_layers<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    reader: &ScxReader,
    layer_filter: Option<&[String]>,
    obsm_filter: Option<&[String]>,
    eager: bool,
    memory_budget: Option<u64>,
    skip_x: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use crate::lazy_mapping::{
        PairwiseAxis, ScxLazyLayersMapping, ScxLazyObsmMapping, ScxLazyPairwiseMapping,
        ScxLazyVarmMapping,
    };
    use std::sync::Arc;

    let anndata_mod = py.import("anndata")?;

    // Phase 4d: catalog-only estimate of the full-assembly bytes. If
    // the estimate exceeds the budget (caller's `memory_budget` kwarg,
    // or `DEFAULT_EAGER_MEMORY_BUDGET_BYTES` = 8 GiB when unset) emit a
    // `UserWarning` recommending the backed / query alternatives.
    // Assembly proceeds regardless — the warning is advisory.
    let budget = memory_budget.unwrap_or(DEFAULT_EAGER_MEMORY_BUDGET_BYTES);
    let est_bytes = estimate_eager_assembly_bytes(reader);
    if est_bytes > budget {
        warn_python_convert(
            py,
            &scx_convert::ConvertWarning::EagerAssemblyMemoryHigh {
                estimated_bytes: est_bytes,
                budget_bytes: budget,
            },
        )?;
    }

    // X — assemble all CSR shards (with deletion vector filtering).
    //
    // `skip_x` builds an X-less skeleton: obs/var define the shape and the
    // caller assigns `adata.X` afterwards. Used by `to_gpu_anndata`'s
    // device-resident streamed path, which decodes X straight onto the GPU
    // instead of materialising a host scipy CSR here. obs/var/obsm/uns/layers
    // are assembled identically either way.
    let x = if skip_x {
        None
    } else {
        let csr = reader.read_all_csr_shards_filtered().map_err(to_pyerr)?;
        Some(csr_to_scipy(py, csr)?)
    };

    // obs metadata — filter by deletion vectors if present
    let obs = match reader.read_obs() {
        Ok(batch) => {
            let filtered_batch = filter_obs_by_deletion_vectors(reader, batch)?;
            let table = record_batch_to_pyarrow(py, &filtered_batch)?;
            Some(pyarrow_table_to_pandas(&table)?)
        }
        Err(scx_format_io::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    // var metadata
    let var = match reader.read_var() {
        Ok(batch) => {
            let table = record_batch_to_pyarrow(py, &batch)?;
            Some(pyarrow_table_to_pandas(&table)?)
        }
        Err(scx_format_io::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    // obsm embeddings.
    //
    // `obsm` is eager by default (it tends to be small relative to
    // obsp/varp/varm). It only becomes a lazy bridge when the caller has
    // explicitly opted into selective loading via `obsm=[...]` AND not
    // forced `eager=True` — keeping default semantics byte-identical
    // while letting the random-access dataloader path defer (and skip)
    // per-key materialisation. See `ScxLazyObsmMapping`.
    //
    // `obsm_filter` restricts the loaded keys in either mode
    // (`to_anndata(obsm=[...])`).
    let lazy_obsm_requested = !eager && obsm_filter.is_some();
    let obsm_dict = pyo3::types::PyDict::new(py);
    if lazy_obsm_requested {
        // Lazy path: validate the requested keys exist now via a
        // catalog-only scan (no shard bytes read — preserves the
        // deferral guarantee), then let the bridge read each on first
        // access.
        validate_obsm_keys(reader, obsm_filter)?;
    } else {
        let obsm_map = read_obsm_selected(reader, obsm_filter)?;
        for (name, batch) in &obsm_map {
            let filtered = filter_obs_by_deletion_vectors(reader, batch.clone())?;
            let np_arr = obsm_batch_to_numpy(py, &filtered)?;
            obsm_dict.set_item(name, np_arr)?;
        }
    }

    // uns — reconstruct any `__scx_type__` envelopes back into NumPy
    // ndarrays / scalars / tuples / pandas Index/Series/Categorical /
    // structured recarrays. Plain JSON passes through unchanged.
    let uns_dict = read_uns_as_pyobject(py, reader)?;

    // Sibling reader for the lazy bridges (`_obsp`/`_varp`/`_varm`/
    // `_layers`). Independent mmap so the returned AnnData stays valid
    // after the caller's `ScxReader` drops. Skipped when no lazy slot
    // is needed.
    let has_obsp = !reader.list_obsp().is_empty();
    let has_varp = !reader.list_varp().is_empty();
    let has_varm = !reader.list_varm().is_empty();
    let layer_names = reader.layer_names();
    let has_layers = if let Some(filter) = layer_filter {
        layer_names.iter().any(|n| filter.iter().any(|f| f == n))
    } else {
        !layer_names.is_empty()
    };
    let need_lazy = has_obsp || has_varp || has_varm || has_layers || lazy_obsm_requested;

    let obsp_kept = if has_obsp {
        compute_kept_to_global(reader)?.map(Arc::new)
    } else {
        None
    };

    let lazy_reader: Option<Arc<ScxReader>> = if need_lazy {
        Some(Arc::new(
            ScxReader::open_with_shared_catalog(path, reader.catalog_arc()).map_err(to_pyerr)?,
        ))
    } else {
        None
    };
    let lazy_obsp = lazy_reader
        .as_ref()
        .filter(|_| has_obsp)
        .map(|r| ScxLazyPairwiseMapping::new(Arc::clone(r), PairwiseAxis::Obsp, obsp_kept.clone()));
    let lazy_varp = lazy_reader
        .as_ref()
        .filter(|_| has_varp)
        .map(|r| ScxLazyPairwiseMapping::new(Arc::clone(r), PairwiseAxis::Varp, None));
    let lazy_varm = lazy_reader
        .as_ref()
        .filter(|_| has_varm)
        .map(|r| ScxLazyVarmMapping::new(Arc::clone(r)));
    let lazy_layers = lazy_reader
        .as_ref()
        .filter(|_| has_layers)
        .map(|r| ScxLazyLayersMapping::new(Arc::clone(r), layer_filter));
    let lazy_obsm = lazy_reader
        .as_ref()
        .filter(|_| lazy_obsm_requested)
        .map(|r| ScxLazyObsmMapping::new(Arc::clone(r), obsm_filter));

    // Build AnnData kwargs.
    let kwargs = pyo3::types::PyDict::new(py);
    if let Some(x) = x {
        kwargs.set_item("X", x)?;
    }
    if let Some(obs) = obs {
        kwargs.set_item("obs", obs)?;
    }
    if let Some(var) = var {
        kwargs.set_item("var", var)?;
    }
    if !obsm_dict.is_empty() {
        kwargs.set_item("obsm", obsm_dict)?;
    }
    if let Some(uns) = uns_dict {
        kwargs.set_item("uns", uns)?;
    }
    if eager {
        // Materialize each lazy bridge up front; AnnData's __init__
        // receives plain dicts (same shape as pre-fix). Returned AnnData
        // is fully detached from the SCX file handle.
        if let Some(m) = &lazy_obsp {
            kwargs.set_item("obsp", m.materialize_all(py)?)?;
        }
        if let Some(m) = &lazy_varp {
            kwargs.set_item("varp", m.materialize_all(py)?)?;
        }
        if let Some(m) = &lazy_varm {
            kwargs.set_item("varm", m.materialize_all(py)?)?;
        }
        if let Some(m) = &lazy_layers {
            kwargs.set_item("layers", m.materialize_all(py)?)?;
        }
    }

    let adata = anndata_mod.call_method("AnnData", (), Some(&kwargs))?;

    // Reconstruct `adata.raw` if the file carries a raw count matrix.
    // Raw shares X's obs axis; when deletion vectors are active the raw
    // rows would need the same filtering as X, which this path does not
    // yet apply — warn and drop rather than emit a misaligned raw.
    if reader.has_raw() {
        if reader.header().has_deletion_vectors() {
            warn_python_convert(
                py,
                &scx_convert::ConvertWarning::DroppedRaw {
                    raw_n_vars: reader.raw_n_vars().unwrap_or(0),
                },
            )?;
        } else {
            let raw_csr = reader.read_all_raw_csr_shards().map_err(to_pyerr)?;
            let raw_x = csr_to_scipy(py, raw_csr)?;
            let raw_var_batch = reader.read_raw_var().map_err(to_pyerr)?;
            let raw_var_table = record_batch_to_pyarrow(py, &raw_var_batch)?;
            let raw_var = pyarrow_table_to_pandas(&raw_var_table)?;
            let raw_kwargs = pyo3::types::PyDict::new(py);
            raw_kwargs.set_item("X", raw_x)?;
            raw_kwargs.set_item("var", raw_var)?;
            let raw_adata = anndata_mod.call_method("AnnData", (), Some(&raw_kwargs))?;
            // `adata.raw = AnnData(X=..., var=...)` stores it as a Raw —
            // the canonical scanpy idiom.
            adata.setattr("raw", raw_adata)?;
        }
    }

    if !eager {
        // Lazy mode: attach each bridge to AnnData's private `_obsp` /
        // `_varp` / `_varm` / `_layers` storage. AnnData's
        // `AlignedMappingProperty` descriptor reads from these on
        // every public `.obsp` (etc.) access — the first access drives
        // the bridge's per-key materialization through AnnData's
        // validation loop; subsequent accesses hit the bridge's cache.
        // We bypass the property setter (which would otherwise iterate
        // and validate every entry up front, defeating the lazy point).
        if let Some(m) = lazy_obsp {
            adata.setattr("_obsp", m.into_pyobject(py)?)?;
        }
        if let Some(m) = lazy_varp {
            adata.setattr("_varp", m.into_pyobject(py)?)?;
        }
        if let Some(m) = lazy_varm {
            adata.setattr("_varm", m.into_pyobject(py)?)?;
        }
        if let Some(m) = lazy_layers {
            adata.setattr("_layers", m.into_pyobject(py)?)?;
        }
        // Lazy obsm is only built when the caller opted into selective
        // loading (`obsm=[...]`, eager=False); default behaviour keeps
        // obsm eager. Attach via `_obsm` to bypass AnnData's axis-length
        // validation, same as the other bridges.
        if let Some(m) = lazy_obsm {
            adata.setattr("_obsm", m.into_pyobject(py)?)?;
        }
    }

    Ok(adata)
}

/// Build an AnnData with optional var_names projection, obs_filter, and layers selection.
///
/// For obs_filter: delegates to the QueryPipeline for predicate pushdown.
/// For var_names: resolves gene names to column indices and applies column slicing.
/// For layers: filters which layers are loaded.
///
/// When `preserve_slots=true` and `obs_filter` is set, the eager path
/// `to_anndata_with_layers()` is used (loading X / obs / var / obsm /
/// layers with deletion vectors applied) and then sliced by a pandas.eval
/// boolean mask. This preserves obsm and layers at the cost of the query
/// engine's predicate-pushdown shard skipping. When `preserve_slots=false`
/// (default), the query-engine path runs and emits a warning if obsm or
/// layers exist on disk (since they are dropped from the result).
#[allow(clippy::too_many_arguments)]
pub fn to_anndata_filtered<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    reader: &ScxReader,
    var_names: Option<&[String]>,
    obs_filter: Option<&str>,
    layer_filter: Option<&[String]>,
    obsm_filter: Option<&[String]>,
    preserve_slots: bool,
    eager: bool,
    memory_budget: Option<u64>,
    skip_x: bool,
) -> PyResult<Bound<'py, PyAny>> {
    // `skip_x` (the X-less skeleton for `to_gpu_anndata`) is only meaningful on
    // the no-filter fast path — the caller guarantees no var_names / obs_filter /
    // layer projection when it sets it (those paths reshape X and must build it).
    debug_assert!(
        !skip_x || (var_names.is_none() && obs_filter.is_none() && layer_filter.is_none()),
        "skip_x requires no var_names / obs_filter / layer_filter"
    );

    // Fast path: no filtering → use existing implementation
    if var_names.is_none() && obs_filter.is_none() && layer_filter.is_none() {
        return to_anndata_with_layers(
            py,
            path,
            reader,
            None,
            obsm_filter,
            eager,
            memory_budget,
            skip_x,
        );
    }

    // preserve_slots=true with obs_filter: load full AnnData, then filter
    // rows via pandas.eval. Keeps obsm / layers / uns intact at the cost
    // of skipping query-engine predicate pushdown. Force eager so the
    // pandas-side __getitem__ slicing operates on real arrays rather
    // than lazy bridges (which AnnData iterates / validates during
    // `.copy()` anyway).
    if let (Some(expr), true) = (obs_filter, preserve_slots) {
        let full = to_anndata_with_layers(
            py,
            path,
            reader,
            layer_filter,
            obsm_filter,
            true,
            memory_budget,
            false,
        )?;

        let obs_attr = full.getattr("obs")?;
        let mask = obs_attr.call_method1("eval", (expr,)).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "preserve_slots=True parses obs_filter via pandas.eval; \
                 failed to evaluate {expr:?}: {e}"
            ))
        })?;

        // Reject non-boolean results: AnnData treats numeric arrays as
        // positional indices, which would silently reorder rows instead
        // of failing on a malformed predicate.
        let dtype_kind: String = mask.getattr("dtype")?.getattr("kind")?.extract()?;
        if dtype_kind != "b" {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "preserve_slots=True requires obs_filter to evaluate to a \
                 boolean mask (e.g. \"cell_type == 'T cell'\"); expression \
                 {expr:?} produced dtype kind {dtype_kind:?}"
            )));
        }

        // Surface the grammar shift: this path evaluates obs_filter via
        // pandas.eval, which does not match the SCX predicate engine
        // (e.g. pandas accepts `&` / `|` / `~`; SCX accepts only
        // `and` / `or` / `not`). Users opted into preserve_slots=True, so
        // one warning per call is appropriate.
        py.import("warnings")?.call_method1(
            "warn",
            (format!(
                "preserve_slots=True evaluated obs_filter {expr:?} via pandas.eval; \
                 grammar differs from the SCX predicate engine used by \
                 preserve_slots=False (see docs/scanpy.md \"Filter Expression Compatibility\")."
            ),),
        )?;

        let builtins = py.import("builtins")?;
        let slice_all = builtins.call_method1("slice", (py.None(),))?;
        let col_idx = if let Some(names) = var_names {
            let indices = resolve_var_names_to_indices(reader, names)?;
            PyArray1::from_vec(py, indices).into_any().unbind()
        } else {
            slice_all.unbind()
        };
        let idx = pyo3::types::PyTuple::new(py, &[mask.unbind(), col_idx])?;
        return full.get_item(idx)?.call_method0("copy");
    }

    // If obs_filter is specified, use the query engine for predicate pushdown
    if let Some(expr) = obs_filter {
        use scx_engine::QueryPipeline;

        let mut pipeline =
            QueryPipeline::open(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        pipeline = pipeline
            .filter_obs(expr)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // If var_names is also specified, resolve to gene indices
        if let Some(names) = var_names {
            let gene_indices = resolve_var_names_to_indices(reader, names)?;
            pipeline = pipeline.select_genes(gene_indices);
        }

        let result = pipeline
            .collect()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let anndata_mod = py.import("anndata")?;
        let x = csr_to_scipy(py, result.x)?;
        let obs_table = record_batch_to_pyarrow(py, &result.obs)?;
        let obs_df = pyarrow_table_to_pandas(&obs_table)?;
        let var_table = record_batch_to_pyarrow(py, &result.var)?;
        let var_df = pyarrow_table_to_pandas(&var_table)?;

        // uns (still loaded from reader; see read_uns_as_pyobject for the
        // tagged-envelope reconstruction).
        let uns_dict = read_uns_as_pyobject(py, reader)?;

        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("X", x)?;
        kwargs.set_item("obs", obs_df)?;
        kwargs.set_item("var", var_df)?;
        if let Some(uns) = uns_dict {
            kwargs.set_item("uns", uns)?;
        }
        // obsm, varm, obsp, varp, and layers are not available via QueryResult.
        // Warn if the source file contains them so users know they're being dropped.
        let has_obsm = reader
            .read_all_obsm()
            .map(|m| !m.is_empty())
            .unwrap_or(false);
        let has_varm = reader
            .read_all_varm()
            .map(|m| !m.is_empty())
            .unwrap_or(false);
        let has_obsp = reader
            .read_all_obsp()
            .map(|m| !m.is_empty())
            .unwrap_or(false);
        let has_varp = reader
            .read_all_varp()
            .map(|m| !m.is_empty())
            .unwrap_or(false);
        let has_layers = !reader.layer_names().is_empty();
        if has_obsm || has_varm || has_obsp || has_varp || has_layers {
            let warnings = py.import("warnings")?;
            let mut parts = Vec::new();
            if has_obsm {
                parts.push("obsm");
            }
            if has_varm {
                parts.push("varm");
            }
            if has_obsp {
                parts.push("obsp");
            }
            if has_varp {
                parts.push("varp");
            }
            if has_layers {
                parts.push("layers");
            }
            warnings.call_method1(
                "warn",
                (format!(
                    "obs_filter with non-backed mode uses the query engine, which does not \
                     load {}. Pass preserve_slots=True to materialize them (skips predicate \
                     pushdown), use backed=True, or load the full dataset and filter in Python.",
                    parts.join(", ")
                ),),
            )?;
        }

        // The obs-filtered query path does not subset the raw matrix's
        // obs axis — warn + drop rather than emit a misaligned raw.
        if reader.has_raw() {
            warn_python_convert(
                py,
                &scx_convert::ConvertWarning::DroppedRaw {
                    raw_n_vars: reader.raw_n_vars().unwrap_or(0),
                },
            )?;
        }

        let adata = anndata_mod.call_method("AnnData", (), Some(&kwargs))?;
        return Ok(adata);
    }

    // No obs_filter but var_names and/or layers specified
    // Load normally, then apply var_names column projection. Force eager
    // because slicing the AnnData by var_names triggers AlignedMapping
    // validation across all aligned slots (obsp / varp / varm), which
    // would materialize through the lazy bridges anyway — doing it up
    // front avoids fragmenting the cost across implicit slicing.
    let adata = to_anndata_with_layers(
        py,
        path,
        reader,
        layer_filter,
        obsm_filter,
        true,
        memory_budget,
        false,
    )?;

    if let Some(names) = var_names {
        // Resolve via the same path as backed / query-engine: scans all string
        // columns (so gene symbols in non-index columns work) and returns
        // sorted positional indices. Slicing adata[:, np_indices] then projects
        // X, layers, var, varm, and varp consistently.
        let indices = resolve_var_names_to_indices(reader, names)?;
        let np_indices = PyArray1::from_vec(py, indices);

        let builtins = py.import("builtins")?;
        let slice_all = builtins.call_method1("slice", (py.None(),))?;
        let idx =
            pyo3::types::PyTuple::new(py, &[slice_all.unbind(), np_indices.into_any().unbind()])?;
        let sliced = adata.get_item(idx)?;
        let copied = sliced.call_method0("copy")?;
        return Ok(copied);
    }

    Ok(adata)
}

/// Resolve gene names to column indices using the var metadata.
pub(crate) fn resolve_var_names_to_indices(
    reader: &ScxReader,
    names: &[String],
) -> PyResult<Vec<u32>> {
    let var_batch = match reader.read_var() {
        Ok(batch) => batch,
        Err(scx_format_io::ScxError::SectionNotFound(_)) => {
            return Err(PyRuntimeError::new_err(
                "Cannot resolve var_names: this SCX file has no var metadata. \
                 Open without var_names to load all genes."
                    .to_string(),
            ));
        }
        Err(e) => return Err(to_pyerr(e)),
    };

    // Try to find gene names in the var DataFrame index.
    // The index column is typically the first column (or named "gene_id").
    // We check all string columns.
    let mut name_to_idx: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();

    for col_idx in 0..var_batch.num_columns() {
        let col = var_batch.column(col_idx);
        if let Some(str_arr) = col.as_any().downcast_ref::<arrow::array::StringArray>() {
            for (row, val) in str_arr.iter().enumerate() {
                if let Some(v) = val {
                    name_to_idx.entry(v).or_insert(row as u32);
                }
            }
        }
    }

    let mut indices = Vec::with_capacity(names.len());
    let mut not_found = Vec::new();
    for name in names {
        match name_to_idx.get(name.as_str()) {
            Some(&idx) => indices.push(idx),
            None => not_found.push(name.as_str()),
        }
    }

    if indices.is_empty() {
        return Err(PyRuntimeError::new_err(format!(
            "None of the requested var_names were found in the var metadata: {:?}",
            not_found
        )));
    }

    // Sort + dedup so all callers produce var rows in sorted column-position
    // order, matching scx-engine::project_var(). Keeps eager / backed /
    // query-engine paths consistent under reordered or duplicated requests.
    indices.sort_unstable();
    indices.dedup();

    Ok(indices)
}

/// Build an AnnData object with backed (on-demand) X and layers.
///
/// Opens a new ScxReader (independent mmap) so the backed dataset can
/// outlive the PyExperiment that created it. obs/var/obsm/uns are loaded
/// eagerly (same as non-backed mode).
///
/// When deletion vectors are present, a `kept_to_global` mapping is
/// computed and passed to `ScxBackedSparseDataset` so that user-visible
/// row indices exclude deleted rows (matching non-backed behavior).
#[allow(clippy::too_many_arguments)]
pub fn to_anndata_backed<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    cache_shards: usize,
    var_names: Option<&[String]>,
    obs_filter: Option<&str>,
    layer_filter: Option<&[String]>,
    obsm_filter: Option<&[String]>,
    eager: bool,
) -> PyResult<Bound<'py, PyAny>> {
    to_anndata_backed_with_options(
        py,
        path,
        cache_shards,
        var_names,
        obs_filter,
        layer_filter,
        obsm_filter,
        true,
        eager,
    )
}

/// Internal entrypoint for the backed AnnData builder.
///
/// `apply_deletion_vectors`: when `true` (the default for the public
/// `to_anndata_backed`), the X / obs / obsm / obsp paths are filtered through
/// the file's global deletion vectors. When `false`, the function returns the
/// unfiltered axes — used by `mudata::to_mudata_backed` so that the inner
/// AnnData's `obs` row count matches the outer MuData's global `obs` (which
/// is also unfiltered, matching the eager `to_mudata` path's behaviour).
#[allow(clippy::too_many_arguments)]
pub(crate) fn to_anndata_backed_with_options<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    cache_shards: usize,
    var_names: Option<&[String]>,
    obs_filter: Option<&str>,
    layer_filter: Option<&[String]>,
    obsm_filter: Option<&[String]>,
    apply_deletion_vectors: bool,
    eager: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use crate::backed::{ScxBackedLayerDataset, ScxBackedSparseDataset};
    use crate::lazy_mapping::{
        PairwiseAxis, ScxLazyObsmMapping, ScxLazyPairwiseMapping, ScxLazyVarmMapping,
    };
    use scx_format_io::BackedCsrReader;
    use std::sync::Arc;

    let anndata_mod = py.import("anndata")?;
    let reader = ScxReader::open(path).map_err(to_pyerr)?;
    // Share one parsed `FullCatalog` across the N+3 `ScxReader`
    // instances this function constructs (main reader + X CSR + CSC
    // sidecar + one per backed layer). The catalog is bytes-identical
    // across all opens of the same file, so re-parsing it N+3 times
    // per worker is pure overhead — the worker-amplification path that
    // motivated the Arc-sharing change. The shard cache and
    // singleflight table stay per-instance; only the immutable
    // catalog is reused. See docs/multithreading.md for the
    // fork-safety contract.
    let shared_catalog = reader.catalog_arc();

    // --- Compute kept_to_global from deletion vectors (if present) ---
    // Cache the deletion-vector-only mapping; obs_filter may mutate kept_to_global
    // further, but obsm filtering needs the original DV-only version.
    // Skipped when `apply_deletion_vectors` is false (e.g. `to_mudata_backed`
    // single-modality wrap, where DVs are intentionally not applied to keep
    // inner-AnnData obs in lockstep with the unfiltered outer MuData obs).
    let dv_kept_to_global = if apply_deletion_vectors {
        compute_kept_to_global(&reader)?
    } else {
        None
    };
    let mut kept_to_global = dv_kept_to_global.clone();

    // --- obs (eager, optionally filtered by deletion vectors) ---
    let obs = match reader.read_obs() {
        Ok(batch) => {
            let filtered_batch = if apply_deletion_vectors {
                filter_obs_by_deletion_vectors(&reader, batch)?
            } else {
                batch
            };
            let table = record_batch_to_pyarrow(py, &filtered_batch)?;
            Some(pyarrow_table_to_pandas(&table)?)
        }
        Err(scx_format_io::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    // --- Apply obs_filter if specified ---
    // Evaluate on the pandas DataFrame rather than QueryPipeline. In backed
    // mode, shard-level pushdown has negligible benefit since X is lazy (only
    // accessed shards are decoded). Pandas .query() is simpler and supports
    // richer expressions.
    let obs = if let Some(expr) = obs_filter {
        if let Some(obs_df) = obs {
            // Use pandas query to filter
            let filtered = obs_df.call_method1("query", (expr,))?;
            let original_idx = obs_df.getattr("index")?;
            let filtered_idx = filtered.getattr("index")?;

            // Get positional indices of kept rows in the (already deletion-filtered) obs
            let np = py.import("numpy")?;
            let isin_mask = original_idx.call_method1("isin", (&filtered_idx,))?;
            let where_result = np.call_method1("where", (&isin_mask,))?;
            // np.where returns a tuple; first element is array of indices
            let pos_indices = where_result.get_item(0)?;
            let pos_arr: numpy::PyReadonlyArray1<'_, i64> = pos_indices
                .call_method1("astype", (np.getattr("int64")?,))?
                .extract()?;
            let pos_slice = pos_arr
                .as_slice()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            // Update kept_to_global to reflect the obs_filter
            match &kept_to_global {
                Some(existing) => {
                    // existing maps user-visible → global. Now further filter.
                    let new_kept: Vec<u64> =
                        pos_slice.iter().map(|&i| existing[i as usize]).collect();
                    kept_to_global = Some(new_kept);
                }
                None => {
                    // No prior deletions. pos_slice maps directly to global.
                    let new_kept: Vec<u64> = pos_slice.iter().map(|&i| i as u64).collect();
                    kept_to_global = Some(new_kept);
                }
            }

            Some(filtered)
        } else {
            None
        }
    } else {
        obs
    };

    // --- Resolve var_names to column indices ---
    let col_indices = if let Some(names) = var_names {
        Some(resolve_var_names_to_indices(&reader, names)?)
    } else {
        None
    };

    // --- X: backed ---
    let x_reader =
        ScxReader::open_with_shared_catalog(path, Arc::clone(&shared_catalog)).map_err(to_pyerr)?;
    let has_csc = x_reader.header().has_csc();
    let x_backed = Arc::new(BackedCsrReader::new(x_reader, cache_shards));
    let x_backed_csc: Option<Arc<scx_format_io::BackedCscReader>> = if has_csc {
        // Open a separate ScxReader for the CSC sidecar (BackedCscReader
        // takes ownership). Header check is cheap; the reader holds a
        // mmap and per-shard catalog, but no shards decode until we
        // actually call read_csc_shard(). Catalog parse is skipped via
        // the shared `Arc<FullCatalog>`.
        let csc_reader = ScxReader::open_with_shared_catalog(path, Arc::clone(&shared_catalog))
            .map_err(to_pyerr)?;
        Some(Arc::new(
            scx_format_io::BackedCscReader::new(csc_reader, cache_shards).map_err(to_pyerr)?,
        ))
    } else {
        None
    };
    let mut x_dataset = match &kept_to_global {
        Some(mapping) => ScxBackedSparseDataset::from_reader_with_deletions(
            Arc::clone(&x_backed),
            cache_shards,
            mapping.clone(),
        ),
        None => ScxBackedSparseDataset::from_reader(Arc::clone(&x_backed), cache_shards),
    };
    x_dataset.with_csc_reader(x_backed_csc);
    x_dataset.with_source_path(path);
    if let Some(ref indices) = col_indices {
        x_dataset.set_col_projection(indices.clone());
    }

    // --- var (eager, optionally filtered by var_names) ---
    let var = match reader.read_var() {
        Ok(batch) => {
            let table = record_batch_to_pyarrow(py, &batch)?;
            let df = pyarrow_table_to_pandas(&table)?;
            if let Some(ref indices) = col_indices {
                // Slice var positionally with the same indices used to project
                // X (set_col_projection above). Using df.iloc keeps var aligned
                // with X when names match a non-index column like gene_symbol;
                // the prior var.index.isin(names) approach produced an empty
                // var when symbols were resolved from non-index columns.
                let np_indices = PyArray1::from_slice(py, indices);
                let iloc = df.getattr("iloc")?;
                let filtered = iloc.get_item(np_indices)?;
                Some(filtered)
            } else {
                Some(df)
            }
        }
        Err(scx_format_io::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    // --- obsm ---
    //
    // Backed dense row-gather (`ScxBackedObsmDataset`) kicks in when the
    // caller selected obsm keys (`obsm=[...]`), did not force `eager`,
    // and did not pass `obs_filter`. Under `obs_filter` we fall back to
    // the eager path (composing a pandas-query row mask with shard
    // gather is deferred), and `obsm=None` keeps the historical
    // eager-all behaviour.
    let use_backed_obsm = obsm_filter.is_some() && !eager && obs_filter.is_none();
    // `obsm_filter` restricts the loaded keys; under backed mode we only
    // need to validate them (the bridge reads each lazily).
    let obsm_dict = pyo3::types::PyDict::new(py);
    if use_backed_obsm {
        validate_obsm_keys(&reader, obsm_filter)?;
    } else {
        build_eager_obsm_dict(
            py,
            &reader,
            obsm_filter,
            obs_filter,
            apply_deletion_vectors,
            &kept_to_global,
            &dv_kept_to_global,
            &obsm_dict,
        )?;
    }

    let lazy_obsm = if use_backed_obsm {
        let obsm_reader = Arc::new(
            ScxReader::open_with_shared_catalog(path, Arc::clone(&shared_catalog))
                .map_err(to_pyerr)?,
        );
        // obs_filter is None here, so kept_to_global == dv_kept_to_global
        // (deletion vectors only).
        let kept_arc = kept_to_global.as_ref().map(|k| Arc::new(k.clone()));
        let config = crate::lazy_mapping::BackedObsmConfig {
            path: path.to_path_buf(),
            cache_shards,
            shared_catalog: Arc::clone(&shared_catalog),
            kept_to_global: kept_arc,
        };
        Some(ScxLazyObsmMapping::new_backed(
            obsm_reader,
            obsm_filter,
            config,
        ))
    } else {
        None
    };

    // --- obsp / varp / varm — lazy bridges by default ---
    //
    // Same `Arc<ScxReader>` (sibling of the main one, sharing the
    // parsed catalog) backs all three bridges; refcount-only clones
    // when handing it to each `ScxLazyPairwiseMapping` /
    // `ScxLazyVarmMapping`. Each bridge decodes its sections on the
    // consumer's first `ad.obsp[…]` / `.varp[…]` / `.varm[…]` access.
    // See `to_anndata_with_layers` for the contract between
    // `eager=true/false` and AnnData's private `_obsp` / `_varp` /
    // `_varm` storage.
    let has_obsp = !reader.list_obsp().is_empty();
    let has_varp = !reader.list_varp().is_empty();
    let has_varm = !reader.list_varm().is_empty();
    let need_lazy_aligned = has_obsp || has_varp || has_varm;
    let lazy_reader: Option<Arc<ScxReader>> = if need_lazy_aligned {
        Some(Arc::new(
            ScxReader::open_with_shared_catalog(path, Arc::clone(&shared_catalog))
                .map_err(to_pyerr)?,
        ))
    } else {
        None
    };
    // Backed-path obsp filter mirrors the eager pre-fix logic in this
    // module (kept_to_global composes deletion vectors with
    // obs_filter); shared by Arc so the bridge holds its own ref.
    let kept_to_global_arc = kept_to_global.as_ref().map(|k| Arc::new(k.clone()));
    let lazy_obsp = lazy_reader.as_ref().filter(|_| has_obsp).map(|r| {
        ScxLazyPairwiseMapping::new(
            Arc::clone(r),
            PairwiseAxis::Obsp,
            kept_to_global_arc.clone(),
        )
    });
    let lazy_varp = lazy_reader
        .as_ref()
        .filter(|_| has_varp)
        .map(|r| ScxLazyPairwiseMapping::new(Arc::clone(r), PairwiseAxis::Varp, None));
    let lazy_varm = lazy_reader
        .as_ref()
        .filter(|_| has_varm)
        .map(|r| ScxLazyVarmMapping::new(Arc::clone(r)));

    // --- uns (eager; tagged envelopes reconstructed) ---
    let uns_dict = read_uns_as_pyobject(py, &reader)?;

    // --- layers (backed, with optional filtering) ---
    let all_layer_names = reader.layer_names();
    let layers_dict = pyo3::types::PyDict::new(py);
    for name in &all_layer_names {
        // Skip layers not in the filter list (if specified)
        if let Some(filter) = layer_filter {
            if !filter.iter().any(|f| f == name) {
                continue;
            }
        }
        let l_reader = ScxReader::open_with_shared_catalog(path, Arc::clone(&shared_catalog))
            .map_err(to_pyerr)?;
        let l_backed = Arc::new(BackedCsrReader::new_for_layer(l_reader, name, cache_shards));
        let mut l_dataset = match &kept_to_global {
            Some(mapping) => ScxBackedLayerDataset::from_reader_with_deletions(
                l_backed,
                cache_shards,
                name.clone(),
                mapping.clone(),
            ),
            None => ScxBackedLayerDataset::from_reader(l_backed, cache_shards, name.clone()),
        };
        if let Some(ref indices) = col_indices {
            l_dataset.inner.set_col_projection(indices.clone());
        }
        let l_py = l_dataset.into_pyobject(py)?;
        layers_dict.set_item(name, l_py)?;
    }

    // Build AnnData kwargs
    let kwargs = pyo3::types::PyDict::new(py);
    let x_py = x_dataset.into_pyobject(py)?;
    kwargs.set_item("X", x_py)?;
    if let Some(obs) = obs {
        kwargs.set_item("obs", obs)?;
    }
    if let Some(var) = var {
        kwargs.set_item("var", var)?;
    }
    if !obsm_dict.is_empty() {
        kwargs.set_item("obsm", obsm_dict)?;
    }
    if let Some(uns) = uns_dict {
        kwargs.set_item("uns", uns)?;
    }
    if !layers_dict.is_empty() {
        kwargs.set_item("layers", layers_dict)?;
    }
    if eager {
        // Eager mode: materialize lazy bridges up front and pass
        // through normal kwargs path. Caller receives an AnnData
        // detached from the SCX file handle.
        if let Some(m) = &lazy_obsp {
            kwargs.set_item("obsp", m.materialize_all(py)?)?;
        }
        if let Some(m) = &lazy_varp {
            kwargs.set_item("varp", m.materialize_all(py)?)?;
        }
        if let Some(m) = &lazy_varm {
            kwargs.set_item("varm", m.materialize_all(py)?)?;
        }
    }

    let adata = anndata_mod.call_method("AnnData", (), Some(&kwargs))?;

    // Backed mode does not reconstruct the raw matrix — warn + drop.
    if reader.has_raw() {
        warn_python_convert(
            py,
            &scx_convert::ConvertWarning::DroppedRaw {
                raw_n_vars: reader.raw_n_vars().unwrap_or(0),
            },
        )?;
    }

    if !eager {
        // Lazy mode: attach bridges directly to AnnData's private
        // storage to bypass `AlignedMappingProperty.__set__`'s eager
        // validation. See [`to_anndata_with_layers`].
        if let Some(m) = lazy_obsp {
            adata.setattr("_obsp", m.into_pyobject(py)?)?;
        }
        if let Some(m) = lazy_varp {
            adata.setattr("_varp", m.into_pyobject(py)?)?;
        }
        if let Some(m) = lazy_varm {
            adata.setattr("_varm", m.into_pyobject(py)?)?;
        }
        // Backed dense row-gather obsm (only built when obsm was
        // selected, not eager, no obs_filter). Attach via `_obsm` to
        // bypass AnnData's axis-length validation, like the other
        // bridges.
        if let Some(m) = lazy_obsm {
            adata.setattr("_obsm", m.into_pyobject(py)?)?;
        }
    }

    Ok(adata)
}

/// Compute `kept_to_global` mapping from deletion vectors.
///
/// Returns `None` if there are no deletions. Otherwise returns a Vec
/// where `kept_to_global[i]` is the global (file-level) row index for
/// user-visible row `i`.
pub(crate) fn compute_kept_to_global(reader: &ScxReader) -> PyResult<Option<Vec<u64>>> {
    // Reuse the shared obs-indexed keep mask, then compress to the global row
    // indices of the surviving rows (same source of truth as the reader CSR
    // filter, `scx compact`, and the streaming export).
    let keep = match reader.deletion_keep_mask().map_err(to_pyerr)? {
        Some(keep) => keep,
        None => return Ok(None),
    };

    let kept: Vec<u64> = (0..keep.len())
        .filter(|&i| keep[i])
        .map(|i| i as u64)
        .collect();

    Ok(Some(kept))
}

/// Filter an obs RecordBatch to exclude deleted rows.
///
/// Builds a boolean keep-mask from the deletion vectors (same logic
/// as `read_all_csr_shards_filtered`) and applies
/// `arrow::compute::filter_record_batch`.
pub(crate) fn filter_obs_by_deletion_vectors(
    reader: &ScxReader,
    obs: arrow::array::RecordBatch,
) -> PyResult<arrow::array::RecordBatch> {
    // Shared obs-indexed keep mask (same logic as the reader CSR filter,
    // `scx compact`, and the streaming export); see
    // `ScxReader::deletion_keep_mask`.
    let keep = match reader.deletion_keep_mask().map_err(to_pyerr)? {
        Some(keep) => keep,
        None => return Ok(obs), // No deletions — return as-is
    };

    let bool_array = arrow::array::BooleanArray::from(keep);
    arrow::compute::filter_record_batch(&obs, &bool_array)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to filter obs: {}", e)))
}
