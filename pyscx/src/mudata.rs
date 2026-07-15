// pyscx mudata bindings.
//
// Mirrors the existing `from_anndata` / `to_anndata` API:
//   - `pyscx.from_mudata(mu, path, ...)` writes a v2 multimodal SCX
//     file from a Python `mudata.MuData` object.
//   - `PyExperiment::to_mudata()` materialises the SCX file as a
//     `mudata.MuData`, building one AnnData per modality with the
//     shared global obs.
//
// Lazy imports on `mudata` — pyscx itself is not a hard dependency
// of MuData. `ImportError("install `mudata`")` surfaces only when a
// caller actually touches the multimodal API.

use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::RecordBatch;
use pyo3::exceptions::{PyImportError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scx_codec::CodecId;
use scx_format_io::header::FileHeader;
use scx_format_io::modality::ModalityType;
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::section::SectionType;
use scx_format_io::select_codec_for_modality;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;

use scx_sparse::{Container, MaterializePlan};

use crate::convert::{
    build_plan, csr_max_value, csr_to_scipy, guard_decode_loss_dtype, obsm_batch_to_numpy,
    pandas_to_record_batch, parse_uns_format, pyarrow_table_to_pandas, record_batch_to_pyarrow,
    typed_csr_to_scipy, typed_read_to_pyerr, uns_py_to_json, UnsFormat,
};
use crate::to_pyerr;

/// Lazy `import mudata` with a friendly error if the package is
/// missing. The dependency is optional — pyscx as a whole works
/// without it; only the multimodal API requires it.
fn import_mudata(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("mudata").map_err(|_| {
        PyImportError::new_err(
            "the `mudata` package is required for pyscx multimodal I/O. \
             Install it with `pip install mudata`.",
        )
    })
}

/// Convenience: same heuristic as the CLI's `h5mu/pipeline.rs` but
/// invoked from Python. Maps a modality name to a `ModalityType`
/// based on common naming conventions for CITE-seq / multiome.
fn infer_modality_type(name: &str) -> ModalityType {
    ModalityType::infer_from_name(name)
}

/// Materialise an SCX file as a `mudata.MuData` whose
/// per-modality `AnnData` objects are backed (X is
/// `ScxBackedSparseDataset`, not a materialised scipy CSR). All
/// modalities share the same global `obs` DataFrame.
///
/// Single-modality files (v1 or single-modality v2) are wrapped in a
/// one-modality `MuData({name_or_X: adata})` rather than raising — so
/// `reader.to_mudata(backed=True)` works uniformly across layouts.
pub fn to_mudata_backed<'py>(
    py: Python<'py>,
    path: &Path,
    reader: &ScxReader,
    cache_shards: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let mudata_mod = import_mudata(py)?;

    // Build the global obs once.
    let global_obs = match reader.read_obs() {
        Ok(batch) => {
            let table = record_batch_to_pyarrow(py, &batch)?;
            Some(pyarrow_table_to_pandas(&table)?)
        }
        Err(scx_format_io::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    let mod_dict = PyDict::new(py);

    if reader.is_multimodal() {
        let shared_catalog = reader.catalog_arc();
        for modality_id in 1..=reader.n_modalities() as u8 {
            let info = reader.modality_info(modality_id).ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "modality_info({modality_id}) returned None — modality table is corrupt"
                ))
            })?;
            let mname = info.name.clone();
            let adata = crate::convert::build_backed_anndata_for_modality(
                py,
                path,
                &shared_catalog,
                modality_id,
                &mname,
                cache_shards,
                global_obs.as_ref(),
            )?;
            mod_dict.set_item(&mname, adata)?;
        }
    } else {
        // Single-modality (v1 or single-modality v2): wrap the existing
        // single-AnnData backed factory in a one-modality MuData. Use the
        // sole modality name when present, else "X" for v1.
        //
        // Skip deletion-vector filtering on the inner AnnData so its `obs`
        // row count stays in lockstep with the unfiltered outer `global_obs`
        // attached to the MuData below — the multimodal branch above and the
        // eager `to_mudata` path both skip DVs, so this keeps every backed-
        // mudata path symmetric.
        let modality_key = if reader.n_modalities() >= 1 {
            reader.modality_names()[0].to_string()
        } else {
            "X".to_string()
        };
        let adata = crate::convert::to_anndata_backed_with_options(
            py,
            path,
            cache_shards,
            None,
            None,
            None,
            None,
            false,
            false,
            // var_names is None here, so preserve_var_order / strict_var_names
            // are inert.
            false,
            false,
        )?;
        mod_dict.set_item(&modality_key, adata)?;
    }

    let mu_kwargs = PyDict::new(py);
    if let Some(obs) = global_obs {
        mu_kwargs.set_item("obs", obs)?;
    }
    let mu = mudata_mod.call_method("MuData", (mod_dict,), Some(&mu_kwargs))?;
    Ok(mu)
}

/// Resolve one scalar-or-dict shaping argument to a per-modality string value.
///
/// `arg` is a Python `str` (applied to every modality), a `dict` keyed by
/// modality name (per-modality), or `None`. A dict key that names no modality in
/// the file is a `ValueError` (fail loud on typos, before any decode). Returns
/// `None` for a missing modality / `None` arg (caller applies its default).
fn resolve_shape_arg(
    arg: Option<&Bound<'_, PyAny>>,
    mname: &str,
    modality_names: &[&str],
    arg_label: &str,
) -> PyResult<Option<String>> {
    let Some(arg) = arg else { return Ok(None) };
    if let Ok(s) = arg.extract::<String>() {
        return Ok(Some(s));
    }
    if let Ok(dict) = arg.cast::<PyDict>() {
        // Validate every key names a real modality (once per lookup is cheap:
        // dicts are tiny; the alternative is threading a validated set through).
        for key in dict.keys() {
            let k = key.extract::<String>()?;
            if !modality_names.contains(&k.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "{arg_label}: '{k}' names no modality in this file; \
                     available modalities: {modality_names:?}"
                )));
            }
        }
        return match dict.get_item(mname)? {
            // An explicit `None` value (e.g. `{"rna": None}`) means "no override
            // for this modality" — same as omitting the key — not a type error.
            Some(v) if v.is_none() => Ok(None),
            Some(v) => Ok(Some(v.extract::<String>()?)),
            None => Ok(None),
        };
    }
    Err(PyValueError::new_err(format!(
        "{arg_label} must be a str (applied to all modalities) or a dict keyed by \
         modality name, got {}",
        arg.get_type().name()?
    )))
}

/// Build the per-modality [`MaterializePlan`] for `mname` from the scalar-or-dict
/// shaping args. Missing keys resolve to the default (`csr` / `f32` / `i32`);
/// `container="dense"` is rejected (deferred — CSR only for now).
fn resolve_modality_plan(
    py: Python<'_>,
    mname: &str,
    modality_names: &[&str],
    container: Option<&Bound<'_, PyAny>>,
    data_dtype: Option<&Bound<'_, PyAny>>,
    index_dtype: Option<&Bound<'_, PyAny>>,
    allow_lossy: bool,
) -> PyResult<MaterializePlan> {
    let container_str = resolve_shape_arg(container, mname, modality_names, "container")?
        .unwrap_or_else(|| "csr".to_string());
    let data_dtype_str = resolve_shape_arg(data_dtype, mname, modality_names, "data_dtype")?;
    let index_dtype_str = resolve_shape_arg(index_dtype, mname, modality_names, "index_dtype")?;
    let plan = build_plan(
        py,
        &container_str,
        data_dtype_str.as_deref(),
        index_dtype_str.as_deref(),
        allow_lossy,
    )?;
    if plan.container == Container::Dense {
        return Err(PyValueError::new_err(format!(
            "container='dense' is not yet supported for to_mudata (modality '{mname}'); \
             use container='csr' (the default)"
        )));
    }
    Ok(plan)
}

/// Materialise an `ScxReader` as a `mudata.MuData` object. Iterates
/// `modality_names()`, builds an AnnData per modality, and attaches them to a
/// `MuData(...)` with the shared global obs.
///
/// Each modality's `X` narrows **in-decode** to a caller-chosen target dtype
/// (`data_dtype` / `index_dtype`, scalar-or-dict per modality): a non-default
/// plan assembles directly at the target width via
/// `read_all_csr_shards_for_typed` (never building the intermediate f32 CSR),
/// while a default-plan modality keeps the untouched zero-copy
/// `read_all_csr_shards_for` → `csr_to_scipy` path (byte-identical).
pub fn to_mudata<'py>(
    py: Python<'py>,
    reader: &ScxReader,
    container: Option<&Bound<'_, PyAny>>,
    data_dtype: Option<&Bound<'_, PyAny>>,
    index_dtype: Option<&Bound<'_, PyAny>>,
    allow_lossy: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if !reader.is_multimodal() {
        return Err(PyRuntimeError::new_err(
            "to_mudata() requires a multimodal SCX file (n_modalities > 0); \
             use to_anndata() for single-modality files",
        ));
    }

    let modality_names = reader.modality_names();

    let anndata_mod = py.import("anndata")?;
    let mudata_mod = import_mudata(py)?;

    // Build the global obs once; per-modality AnnData objects share it.
    let global_obs = match reader.read_obs() {
        Ok(batch) => {
            let table = record_batch_to_pyarrow(py, &batch)?;
            Some(pyarrow_table_to_pandas(&table)?)
        }
        Err(scx_format_io::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    // Resolve every modality's materialization plan up front, before decoding
    // anything, so a validation error (bad dict key, dense request) fails fast
    // and uniformly rather than after some modalities have already been
    // decoded/built. `plans[modality_id - 1]` aligns with the 1-based loop below
    // (both `modality_names` and `modality_info(id).name` derive from the same
    // registration-order table).
    let plans: Vec<MaterializePlan> = modality_names
        .iter()
        .map(|mname| {
            resolve_modality_plan(
                py,
                mname,
                &modality_names,
                container,
                data_dtype,
                index_dtype,
                allow_lossy,
            )
        })
        .collect::<PyResult<_>>()?;

    // Iterate modalities in registration order. modality_id is
    // 1-based; index 0 is reserved for "global".
    let mod_dict = PyDict::new(py);
    for modality_id in 1..=reader.n_modalities() as u8 {
        let info = reader.modality_info(modality_id).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "modality_info({modality_id}) returned None — modality table is corrupt"
            ))
        })?;
        let mname = info.name.clone();
        let plan = &plans[(modality_id - 1) as usize];

        // X — fail loud on the decode loss for this modality's shards before
        // decoding, keyed on the target dtype (f32 for a default plan).
        guard_decode_loss_dtype(
            csr_max_value(reader, Some(modality_id)),
            plan.data_dtype,
            plan.allow_lossy,
        )?;
        let x = if plan.is_default_csr_f32() {
            // Untouched zero-copy f32 CSR path — byte-identical to pre-narrow.
            let csr = reader
                .read_all_csr_shards_for(modality_id)
                .map_err(to_pyerr)?;
            csr_to_scipy(py, csr)?
        } else {
            // In-decode narrow: assemble directly at the target dtype.
            let typed = reader
                .read_all_csr_shards_for_typed(modality_id, plan)
                .map_err(typed_read_to_pyerr)?;
            typed_csr_to_scipy(py, typed)?
        };

        // var
        let var = reader.read_var_for(modality_id).map_err(to_pyerr)?;
        let var_table = record_batch_to_pyarrow(py, &var)?;
        let var_df = pyarrow_table_to_pandas(&var_table)?;

        // Per-modality obsm. Catalog entries with this modality_id
        // and section_type ObsmEmbedding.
        let obsm_dict = PyDict::new(py);
        for entry in &reader.catalog().entries {
            if entry.section_type != SectionType::ObsmEmbedding {
                continue;
            }
            if entry.modality_id != modality_id {
                continue;
            }
            let prefix = format!("obsm/{mname}/");
            let key = entry
                .name
                .strip_prefix(&prefix)
                .unwrap_or(&entry.name)
                .to_string();
            let batch = reader.read_obsm_for(modality_id, &key).map_err(to_pyerr)?;
            let np_arr = obsm_batch_to_numpy(py, &batch)?;
            obsm_dict.set_item(&key, np_arr)?;
        }

        // Build per-modality AnnData. Pass the shared obs so the
        // modality's adata.obs reflects the global cell metadata.
        let kwargs = PyDict::new(py);
        kwargs.set_item("X", x)?;
        if let Some(ref obs) = global_obs {
            kwargs.set_item("obs", obs)?;
        }
        kwargs.set_item("var", var_df)?;
        if !obsm_dict.is_empty() {
            kwargs.set_item("obsm", obsm_dict)?;
        }
        let adata = anndata_mod.call_method("AnnData", (), Some(&kwargs))?;
        mod_dict.set_item(&mname, adata)?;
    }

    // Build the outer MuData. The keyword `obs` on MuData wires up
    // the shared global obs.
    let mu_kwargs = PyDict::new(py);
    if let Some(obs) = global_obs {
        mu_kwargs.set_item("obs", obs)?;
    }

    // mudata.MuData(modalities_dict, **mu_kwargs)
    let mu = mudata_mod.call_method("MuData", (mod_dict,), Some(&mu_kwargs))?;
    Ok(mu)
}

/// Implementation of `pyscx.from_h5mu(path, out, ...)` — path-based
/// streaming entry that delegates to
/// `scx_convert::h5mu_to_scx_streaming` (Phase 3). When
/// `stream=false`, falls back to the non-streaming `h5mu_to_scx`.
///
/// `modality_types`: dict mapping name → type string (`"rna"`,
/// `"protein"`, `"atac"`, `"spatial"`, `"methylation"`, `"custom"`).
#[cfg(feature = "hdf5")]
#[allow(clippy::too_many_arguments)]
pub fn from_h5mu_impl(
    py: Python<'_>,
    path: &str,
    out: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
    csc: &str,
    csc_cols_per_shard: usize,
    stream: bool,
    strict_uns: bool,
    memory_budget: Option<Bound<'_, PyAny>>,
    temp_dir: Option<&str>,
    modalities: Option<Vec<String>>,
    modality_types: Option<HashMap<String, String>>,
    index_obs: Vec<String>,
    index_var: Vec<String>,
    index_preset: Option<String>,
    index_auto_threshold: usize,
    bitmap: &str,
    reader_threads: Option<usize>,
    writer_queue_depth: usize,
    row_group_rows: u32,
) -> PyResult<()> {
    use pyo3::exceptions::PyValueError;
    // Missing input → FileNotFoundError (the converter opens the h5mu via
    // `hdf5::File::open`, which surfaces as `ConvertError::Hdf5`, so the
    // common wrong-path case would otherwise raise `RuntimeError`).
    if !std::path::Path::new(path).exists() {
        return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "no such file: '{path}'"
        )));
    }
    let explicit_codec = crate::convert::parse_codec_nonframed(codec)?;
    let csc_policy =
        scx_format_io::CscPolicy::parse(csc).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let bitmap_policy = scx_format_io::BitmapPolicy::parse(bitmap)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let shard_target_rows = shard_size.unwrap_or(scx_format_io::DEFAULT_SHARD_TARGET_ROWS);
    let memory_budget_bytes = crate::convert::parse_memory_budget(memory_budget.as_ref())?;

    // Translate the Python dict form into ConvertOptions::modality_types.
    let modality_types_vec: Vec<(String, ModalityType)> = match modality_types {
        None => Vec::new(),
        Some(map) => map
            .into_iter()
            .map(|(k, v)| {
                let mt = match v.to_lowercase().as_str() {
                    "rna" => ModalityType::Rna,
                    "protein" | "adt" => ModalityType::Protein,
                    "atac" => ModalityType::Atac,
                    "spatial" => ModalityType::Spatial,
                    "methylation" | "methyl" => ModalityType::Methylation,
                    "custom" => ModalityType::Custom,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unknown modality type '{other}' for '{k}'; \
                             valid: rna, protein, atac, spatial, methylation, custom"
                        )));
                    }
                };
                Ok((k, mt))
            })
            .collect::<PyResult<Vec<_>>>()?,
    };

    let opts = scx_convert::ConvertOptions {
        shard_target_rows,
        // Row-group framing (v4) default; `row_group_rows=0` opts out to v3.
        // Explicit here so the opt-out is reachable (Default would force
        // Some(DEFAULT_ROW_GROUP_ROWS) with no way to disable it).
        row_group_rows: (row_group_rows != 0).then_some(row_group_rows),
        codec: explicit_codec,
        csc: csc_policy,
        csc_cols_per_shard,
        tool: "pyscx".into(),
        memory_budget: memory_budget_bytes,
        stream,
        strict_uns,
        dense_zero_epsilon: 0.0,
        temp_dir: temp_dir.map(std::path::PathBuf::from),
        modalities,
        modality_types: modality_types_vec,
        // h5mu paths emit PredicateIndexSkippedMultimodal when these are
        // set — the engine read-side is unimodal-only today.
        index_obs,
        index_var,
        index_preset,
        index_auto_threshold,
        bitmap: bitmap_policy,
        reader_threads,
        writer_queue_depth,
        // Sort-on-convert and grouped convert are not supported for multimodal
        // h5mu inputs yet.
        sort_by: Vec::new(),
        sort_reverse: false,
        ..Default::default()
    };

    let input = std::path::PathBuf::from(path);
    let output = std::path::PathBuf::from(out);
    let mut sink = scx_convert::WarningSink::log();
    if stream {
        py.detach(|| scx_convert::h5mu_to_scx_streaming(&input, &output, &opts, &mut sink))
            .map_err(|e| crate::convert_to_pyerr_with_path(e, path))?;
    } else {
        py.detach(|| scx_convert::h5mu_to_scx(&input, &output, &opts, &mut sink))
            .map_err(|e| crate::convert_to_pyerr_with_path(e, path))?;
    }
    crate::convert::emit_python_warnings(py, &sink)?;
    Ok(())
}

/// Implementation of `pyscx.from_mudata(mu, path, ...)`. Mirrors
/// `from_anndata_impl` but iterates `mu.mod` and emits one modality
/// per AnnData.
///
/// `codec_per_modality` (default `true`) routes each modality through
/// `select_codec_for_modality` so RNA / Protein / ATAC each pick the
/// codec that best suits their value distribution. When `false`,
/// every modality is routed through the single-modality `select_codec`
/// helper instead — used by the multimodal compression benchmark to
/// quantify the gain from per-modality routing (Phase K.3.4).
#[allow(clippy::too_many_arguments)]
/// Serialize a Python `uns`-like dict to `serde_json::Value`. Returns
/// `None` when the dict is empty so an empty `uns` section is never
/// written (callers downstream of `read_uns` treat an absent section
/// as "no uns set", which is what users expect).
fn uns_dict_to_optional_json(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    uns_format: UnsFormat,
) -> PyResult<Option<serde_json::Value>> {
    // Python `None` (e.g. an exotic MuData/AnnData where `uns` was
    // deleted) must not be serialized as `Value::Null` — that would
    // write a useless null section to the catalog. Treat it like an
    // empty dict.
    if obj.is_none() {
        return Ok(None);
    }
    // `len(obj) == 0` is the cheap pre-check; falls back gracefully
    // for non-dict-like objects (we serialize and let the helper raise
    // a useful error there).
    if let Ok(n) = obj.len() {
        if n == 0 {
            return Ok(None);
        }
    }
    Ok(Some(uns_py_to_json(py, obj, uns_format)?))
}

#[allow(clippy::too_many_arguments)]
pub fn from_mudata_impl(
    py: Python<'_>,
    mu: &Bound<'_, PyAny>,
    path: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
    csc: &str,
    csc_cols_per_shard: usize,
    codec_per_modality: bool,
    uns_format: &str,
    row_group_rows: u32,
) -> PyResult<()> {
    let _ = csc_cols_per_shard; // CSC for h5mu input is a Phase D follow-on

    // Row-group framing (v4) default, matching the unimodal in-memory path;
    // `row_group_rows=0` opts out to the legacy unframed v3 layout. Unlike
    // `from_h5mu_impl` (which sets `ConvertOptions::row_group_rows` and lets
    // `ConvertOptions::framing()` build the config), this path bypasses
    // `ConvertOptions` entirely and drives `ScxWriter` directly, so it
    // constructs the `FramingConfig` here and calls `writer.set_framing`
    // itself. `..Default::default()` keeps `target_nnz`/`codec_trial` at their
    // defaults (compact-trial for in-memory MuData is a future follow-on).
    let framing = (row_group_rows != 0).then(|| scx_format_io::FramingConfig {
        row_group_rows,
        ..Default::default()
    });
    let csc_policy =
        scx_format_io::CscPolicy::parse(csc).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    // The in-memory MuData path cannot build per-modality CSC sidecars yet.
    // An explicit `always` request is rejected (the user asked for something
    // we cannot honour); `auto` degrades gracefully to no-CSC.
    if csc_policy == scx_format_io::CscPolicy::Always {
        return Err(PyRuntimeError::new_err(
            "from_mudata(csc='always') is a Phase D+ follow-on (auto-emit \
             per-modality CSC). Use from_mudata(csc='off') for now and run \
             `scx build-csc` afterwards.",
        ));
    }

    // Resolve mu.obs and mu.mod.
    let mu_obs = mu.getattr("obs")?;
    let mod_attr = mu.getattr("mod")?;
    // mu.mod is dict-like — iterate keys in insertion order.
    let modality_names: Vec<String> = mod_attr
        .call_method0("keys")?
        .try_iter()?
        .map(|item| item.and_then(|i| i.extract::<String>()))
        .collect::<PyResult<Vec<_>>>()?;
    if modality_names.is_empty() {
        return Err(PyRuntimeError::new_err(
            "MuData has no modalities (mu.mod is empty)",
        ));
    }

    // Eagerly extract uns now so a malformed dict surfaces a clean
    // error before we open the writer and create a half-written file.
    // `mu.uns` is always present on a valid MuData; per-modality
    // `adata.uns` likewise on AnnData. Empty dicts skip the write so
    // the catalog doesn't carry a useless empty section.
    let uns_format_parsed = parse_uns_format(uns_format)?;
    let mu_uns_attr = mu.getattr("uns")?;
    let global_uns_json = uns_dict_to_optional_json(py, &mu_uns_attr, uns_format_parsed)?;
    let per_modality_uns: Vec<Option<serde_json::Value>> = modality_names
        .iter()
        .map(|mname| {
            let adata = mod_attr.get_item(mname)?;
            let uns_attr = adata.getattr("uns")?;
            uns_dict_to_optional_json(py, &uns_attr, uns_format_parsed)
        })
        .collect::<PyResult<_>>()?;

    // Pre-scan: read each modality's X to learn shapes / nnz so we
    // can populate the file header before opening the writer. The
    // assumption (matches Phase D MVP scope) is cell-aligned
    // modalities — every modality has the same n_obs as mu.obs.
    let n_obs = mu_obs.getattr("shape")?.get_item(0)?.extract::<usize>()?;
    if n_obs == 0 {
        return Err(PyRuntimeError::new_err(
            "MuData outer obs is empty — cannot determine global cell count",
        ));
    }

    // Pull each modality's AnnData and collect (X, var, obsm, ...)
    // into Rust-side structures up-front. This mirrors what
    // from_anndata_impl does for a single AnnData; for now we use
    // scipy.sparse.csr_matrix as the X carrier.
    struct ModalityPayload {
        name: String,
        modality_type: ModalityType,
        // CSR arrays in scipy-compatible types.
        indptr: Vec<i64>,
        indices: Vec<i32>,
        data: Vec<f32>,
        n_vars: usize,
        nnz: u64,
        // Pyarrow-friendly var DataFrame
        var_batch: RecordBatch,
        // obsm: name -> RecordBatch
        obsm: HashMap<String, RecordBatch>,
    }

    let mut modalities: Vec<ModalityPayload> = Vec::with_capacity(modality_names.len());
    for mname in &modality_names {
        let adata = mod_attr.get_item(mname)?;

        // Verify n_obs alignment.
        let mod_n_obs = adata.getattr("n_obs")?.extract::<usize>()?;
        if mod_n_obs != n_obs {
            return Err(PyRuntimeError::new_err(format!(
                "modality '{mname}' has n_obs={mod_n_obs} but outer mu.obs has \
                 n_obs={n_obs} — Phase D currently requires cell-aligned modalities"
            )));
        }
        let mod_n_vars = adata.getattr("n_vars")?.extract::<usize>()?;

        // X — convert to scipy CSR via the AnnData object's
        // adata.X.tocsr() (or just use as-is if already CSR).
        let scipy_sparse = py.import("scipy.sparse")?;
        let x_attr = adata.getattr("X")?;
        // Materialise to CSR (covers dense AnnData too).
        let x_csr = scipy_sparse.call_method1("csr_matrix", (x_attr,))?;
        // sort_indices to match SCX's invariant
        x_csr.call_method0("sort_indices")?;

        let indptr_arr = x_csr.getattr("indptr")?;
        let indices_arr = x_csr.getattr("indices")?;
        let data_arr = x_csr.getattr("data")?;

        let indptr_np = indptr_arr.call_method1("astype", ("int64",))?;
        let indices_np = indices_arr.call_method1("astype", ("int32",))?;
        let data_np = data_arr.call_method1("astype", ("float32",))?;

        let indptr: Vec<i64> = indptr_np
            .extract::<numpy::PyReadonlyArray1<i64>>()?
            .as_slice()?
            .to_vec();
        let indices: Vec<i32> = indices_np
            .extract::<numpy::PyReadonlyArray1<i32>>()?
            .as_slice()?
            .to_vec();
        let data: Vec<f32> = data_np
            .extract::<numpy::PyReadonlyArray1<f32>>()?
            .as_slice()?
            .to_vec();
        let nnz = *indptr.last().unwrap_or(&0) as u64;

        // var
        let var_pd = adata.getattr("var")?;
        let var_batch = pandas_to_record_batch(py, &var_pd)?;

        // obsm — dict of (key -> ndarray)
        let mut obsm: HashMap<String, RecordBatch> = HashMap::new();
        let obsm_attr = adata.getattr("obsm")?;
        let keys: Vec<String> = obsm_attr
            .call_method0("keys")?
            .try_iter()?
            .map(|i| i.and_then(|x| x.extract::<String>()))
            .collect::<PyResult<Vec<_>>>()?;
        for key in keys {
            let arr = obsm_attr.get_item(&key)?;
            let pd_mod = py.import("pandas")?;
            let df = pd_mod.call_method1("DataFrame", (arr,))?;
            let batch = pandas_to_record_batch(py, &df)?;
            obsm.insert(key, batch);
        }

        let modality_type = infer_modality_type(mname);
        modalities.push(ModalityPayload {
            name: mname.clone(),
            modality_type,
            indptr,
            indices,
            data,
            n_vars: mod_n_vars,
            nnz,
            var_batch,
            obsm,
        });
    }

    // Build outer obs RecordBatch.
    let outer_obs_batch = pandas_to_record_batch(py, &mu_obs)?;

    let total_nnz: u64 = modalities.iter().map(|m| m.nnz).sum();
    let max_n_vars = modalities.iter().map(|m| m.n_vars).max().unwrap_or(0) as u64;
    let index_dtype: u8 = if max_n_vars <= 65535 { 0 } else { 1 };
    let shard_target_rows = shard_size.unwrap_or(16384);

    let mut header = FileHeader::new_single_modality(
        n_obs as u64,
        max_n_vars,
        total_nnz,
        shard_target_rows,
        0,
        index_dtype,
    );
    // A framed shard is only valid inside a v4 file, so bump the header
    // alongside set_framing (below); framing == None keeps the v3 layout.
    if framing.is_some() {
        header.format_version = scx_format_io::header::CURRENT_FORMAT_VERSION;
    }

    // Open writer + emit sections.
    let mut writer = ScxWriter::new(Path::new(path), header).map_err(to_pyerr)?;
    writer.set_framing(framing);
    writer.write_obs(&outer_obs_batch).map_err(to_pyerr)?;
    if let Some(json) = &global_uns_json {
        writer.write_uns(json).map_err(to_pyerr)?;
    }

    let explicit_codec = match codec {
        None | Some("auto") => None,
        Some("none") => Some(CodecId::None),
        Some("scx1") => Some(CodecId::Scx1),
        Some("zstd") => Some(CodecId::Zstd),
        Some("lz4") => Some(CodecId::Lz4Shuffle),
        Some("pcodec") => Some(CodecId::Pcodec),
        Some("shufdelta") => Some(CodecId::ShufDeltaZstd),
        Some(other) => {
            return Err(PyRuntimeError::new_err(format!(
                "unknown codec '{other}'; use auto, none, scx1, zstd, lz4, pcodec, or shufdelta"
            )));
        }
    };

    for (i, payload) in modalities.iter().enumerate() {
        // Integer-detect per-modality: small UMI / ADT / ATAC counts
        // compress dramatically better when stored as uint8/16/32
        // than as Float32. Mirrors `from_anndata`'s per-shard
        // `detect_value_encoding(shard_data)` path; without this,
        // every modality's X would land as Float32 → Pcodec
        // regardless of `select_codec_for_modality`'s biological
        // routing.
        let value_encoding = scx_codec::value_encoding::detect_value_encoding(&payload.data);
        let raw_values_bytes =
            scx_codec::value_encoding::values_to_raw_bytes(&payload.data, value_encoding)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let codec_id = match explicit_codec {
            Some(c) => {
                if c == CodecId::Scx1 && !value_encoding.is_integer() {
                    CodecId::Zstd
                } else {
                    c
                }
            }
            None => {
                if codec_per_modality {
                    select_codec_for_modality(
                        &raw_values_bytes,
                        value_encoding,
                        payload.modality_type,
                    )
                } else {
                    // Phase K.3.4: skip the per-modality routing and
                    // run every modality through the same
                    // single-modality `select_codec`. Used by the
                    // multimodal compression benchmark to compare the
                    // uniform-auto baseline against per-modality
                    // routing.
                    scx_format_io::select_codec(&raw_values_bytes, value_encoding)
                }
            }
        };

        let modality_id = writer
            .add_modality(
                &payload.name,
                payload.modality_type,
                codec_id,
                value_encoding,
                false, // build_csc: per-modality CSC handled by from_mudata's
                       // explicit csc='always' path, not the writer auto-emit.
            )
            .map_err(to_pyerr)?;
        writer
            .set_modality_n_vars(modality_id, payload.n_vars as u64)
            .map_err(to_pyerr)?;
        writer
            .write_var_for(modality_id, &payload.var_batch)
            .map_err(to_pyerr)?;
        if let Some(json) = &per_modality_uns[i] {
            writer.write_uns_for(modality_id, json).map_err(to_pyerr)?;
        }

        // Shard each modality's CSR by `shard_target_rows` so the
        // emitted file streams well at census scale. The single-modality
        // `from_anndata` writer already shards; mirroring that here
        // keeps multimodal training I/O on the same footing. Codec /
        // value_encoding are detected once per modality (see comment
        // above) and shared across all of that modality's shards —
        // mixing codecs within one modality would defeat the
        // biological routing rationale.
        let total_rows = payload.indptr.len().saturating_sub(1);
        let shard_target = shard_target_rows as usize;
        let value_byte_size = value_encoding.byte_width();
        let mut row_offset = 0usize;
        while row_offset < total_rows {
            let shard_rows = std::cmp::min(shard_target, total_rows - row_offset);
            let shard_indptr_base = payload.indptr[row_offset] as u64;
            let shard_indptr: Vec<u64> = payload.indptr[row_offset..=row_offset + shard_rows]
                .iter()
                .map(|&v| (v as u64) - shard_indptr_base)
                .collect();
            let shard_nnz = *shard_indptr.last().unwrap_or(&0);
            let idx_start = shard_indptr_base as usize;
            let idx_end = idx_start + shard_nnz as usize;

            let shard_indices: Vec<u32> = payload.indices[idx_start..idx_end]
                .iter()
                .map(|&v| {
                    u32::try_from(v)
                        .map_err(|_| PyRuntimeError::new_err(format!("negative column index {v}")))
                })
                .collect::<PyResult<Vec<_>>>()?;

            let val_start = idx_start * value_byte_size;
            let val_end = idx_end * value_byte_size;
            let shard_values = &raw_values_bytes[val_start..val_end];

            writer
                .write_csr_shard_for(
                    modality_id,
                    &shard_indptr,
                    &shard_indices,
                    shard_values,
                    codec_id,
                    value_encoding,
                    row_offset as u64,
                )
                .map_err(to_pyerr)?;

            row_offset += shard_rows;
        }

        for (key, batch) in &payload.obsm {
            writer
                .write_obsm_for(modality_id, key, batch)
                .map_err(to_pyerr)?;
        }
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp,
            action: "from_mudata".to_string(),
            tool: "pyscx".to_string(),
            params_json: format!("{{\"path\":\"{path}\"}}"),
            input_checksums: vec![],
        }])
        .map_err(to_pyerr)?;

    writer.finish().map_err(to_pyerr)?;
    Ok(())
}
