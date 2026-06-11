// Backed / HDF5 AnnData routing to the streaming converter.
//
// Extracted from the former pyscx/src/anndata.rs (T5.7).

use arrow::array::RecordBatch;
use pyo3::prelude::*;

#[cfg(feature = "hdf5")]
use pyo3::exceptions::{PyRuntimeError, PyValueError};
#[cfg(feature = "hdf5")]
use scx_codec::CodecId;

use super::*;

/// Route a backed AnnData object through the streaming converter.
/// Extracts in-memory `obs` / `var` / `uns` / `obsm` / `varm` /
/// `obsp` / `varp` into Rust types so any caller mutations are
/// preserved, then invokes `scx_convert::h5ad_to_scx_streaming` on
/// the backing h5ad file.
///
/// X and layers always come from disk via streaming — there's no
/// override hook for those (they're potentially too large to extract
/// from a backed AnnData into memory). Emits a `UserWarning` when
/// the backed AnnData has any layers, because the streaming reads
/// will overwrite any in-memory layer mutations.
#[cfg(feature = "hdf5")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn route_backed_anndata_to_streaming(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    path: &str,
    explicit_codec: Option<CodecId>,
    shard_target_rows: u32,
    csc_policy: scx_format::CscPolicy,
    csc_cols_per_shard: usize,
    uns_format_parsed: UnsFormat,
    stream: bool,
    strict_uns: bool,
    dense_zero_epsilon: f32,
    memory_budget: Option<u64>,
    temp_dir: Option<&str>,
    index_obs: Vec<String>,
    index_var: Vec<String>,
    index_preset: Option<String>,
    index_auto_threshold: usize,
    bitmap: &str,
    reader_threads: Option<usize>,
    writer_queue_depth: usize,
) -> PyResult<()> {
    let bitmap_policy = scx_format::BitmapPolicy::parse(bitmap)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    // Resolve the on-disk h5ad path. `anndata` 0.12 exposes both
    // `adata.filename` (preferred) and `adata.file.filename` (older
    // name); we try both. Recent anndata returns `pathlib.PosixPath`
    // rather than a bare `str`, so go through Python's `str(...)` —
    // it's a no-op on `str` and stringifies `Path` cleanly.
    fn fspath_str(v: &Bound<'_, PyAny>) -> Option<String> {
        if v.is_none() {
            return None;
        }
        v.str()
            .ok()
            .and_then(|s| s.extract::<String>().ok())
            .filter(|s| !s.is_empty())
    }
    let filename: String = adata
        .getattr("filename")
        .ok()
        .and_then(|v| fspath_str(&v))
        .or_else(|| {
            adata
                .getattr("file")
                .ok()
                .and_then(|f| f.getattr("filename").ok())
                .and_then(|v| fspath_str(&v))
        })
        .unwrap_or_default();
    if filename.is_empty() || !std::path::Path::new(&filename).exists() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "backed AnnData has no resolvable h5ad filename; use \
             pyscx.from_h5ad(path, out) or convert to a non-backed \
             AnnData first",
        ));
    }

    // Build the overrides from the in-memory AnnData. `obs` / `var`
    // are always extracted from Python — the pandas → Arrow conversion
    // carries the categorical and nullable-encoding metadata that
    // scx-convert's pure-Rust `read_dataframe_group` would lose.
    // `obsm` / `varm` / `obsp` / `varp` / `uns` are extracted only when
    // mutation detection sees a divergence from the on-disk h5ad —
    // otherwise the streaming pipeline reads them from disk one shard
    // at a time. This avoids the per-shard OOM that the wholesale
    // Python extraction causes for inputs with large embeddings
    // (e.g. Parse-PBMC obsm reaching tens of GB).
    let obs_override = pandas_to_record_batch(py, &adata.getattr("obs")?)?;
    let var_override = pandas_to_record_batch(py, &adata.getattr("var")?)?;

    // Open the source h5ad through h5py. We only ever read group
    // `.keys()` — never any dataset value — so anndata's lazy
    // `obsm[key]` materialisation path stays untriggered.
    let h5py = py.import("h5py")?;
    let h5_kwargs = pyo3::types::PyDict::new(py);
    h5_kwargs.set_item("mode", "r")?;
    let h5_file = h5py.call_method("File", (&filename,), Some(&h5_kwargs))?;

    let obsm_clean = section_keys_match(py, &h5_file, adata, "obsm")?;
    let varm_clean = section_keys_match(py, &h5_file, adata, "varm")?;
    let obsp_clean = section_keys_match(py, &h5_file, adata, "obsp")?;
    let varp_clean = section_keys_match(py, &h5_file, adata, "varp")?;
    let uns_clean = section_keys_match(py, &h5_file, adata, "uns")?;

    let obsm_override = if obsm_clean {
        None
    } else {
        Some(extract_dense_mapping(py, adata, "obsm")?)
    };
    let varm_override = if varm_clean {
        None
    } else {
        Some(extract_dense_mapping(py, adata, "varm")?)
    };
    let obsp_override = if obsp_clean {
        None
    } else {
        Some(extract_coo_mapping(py, adata, "obsp")?)
    };
    let varp_override = if varp_clean {
        None
    } else {
        Some(extract_coo_mapping(py, adata, "varp")?)
    };
    let uns_override = if uns_clean {
        None
    } else {
        extract_uns_value(py, adata, uns_format_parsed)?
    };

    // Close the h5py file handle before scx-convert opens the same path
    // via the Rust `hdf5` crate. Concurrent libhdf5 access from h5py
    // and the Rust crate on a single file isn't documented as safe.
    let _ = h5_file.call_method0("close");

    // Layer mutations on a backed AnnData are not propagated — the
    // streaming pipeline always reads layers from disk. Warn so the
    // user knows.
    if let Ok(layers) = adata.getattr("layers") {
        if let Ok(len_val) = layers.call_method0("__len__") {
            if let Ok(len) = len_val.extract::<usize>() {
                if len > 0 {
                    let msg = format!(
                        "backed AnnData has {len} layer(s); layer data will be read \
                         from the on-disk h5ad file. Any in-memory layer mutations \
                         will be lost. Use pyscx.from_h5ad(path, out) on a \
                         freshly-written h5ad if you need mutated layers preserved.",
                    );
                    let _ = py
                        .import("warnings")
                        .and_then(|w| w.call_method1("warn", (msg,)));
                }
            }
        }
    }

    // Breadcrumb for the corner case where a user replaced a value
    // under an existing key (the keys-only heuristic can't catch this).
    let routed_from_disk: Vec<&str> = [
        ("obsm", obsm_clean),
        ("varm", varm_clean),
        ("obsp", obsp_clean),
        ("varp", varp_clean),
        ("uns", uns_clean),
    ]
    .into_iter()
    .filter_map(|(name, clean)| if clean { Some(name) } else { None })
    .collect();
    if !routed_from_disk.is_empty() {
        let msg = format!(
            "pyscx: streaming {} from the on-disk h5ad (no Python-side mutation \
             detected via top-level key comparison). If you replaced a value \
             under an existing key in-place, the on-disk version wins; re-add \
             the key under a fresh name to force the Python value through.",
            routed_from_disk.join(", ")
        );
        let _ = py
            .import("warnings")
            .and_then(|w| w.call_method1("warn", (msg,)));
    }

    let overrides = scx_convert::StreamingOverrides {
        obs: Some(obs_override),
        var: Some(var_override),
        uns: uns_override,
        obsm: obsm_override,
        varm: varm_override,
        obsp: obsp_override,
        varp: varp_override,
    };

    let opts = scx_convert::ConvertOptions {
        shard_target_rows,
        codec: explicit_codec,
        csc: csc_policy,
        csc_cols_per_shard,
        tool: "pyscx".into(),
        memory_budget,
        stream,
        strict_uns,
        dense_zero_epsilon,
        temp_dir: temp_dir.map(std::path::PathBuf::from),
        modalities: None,
        modality_types: Vec::new(),
        index_obs,
        index_var,
        index_preset,
        index_auto_threshold,
        bitmap: bitmap_policy,
        reader_threads,
        writer_queue_depth,
    };
    let input = std::path::PathBuf::from(filename);
    let output = std::path::PathBuf::from(path);

    // Honour `stream=false` by routing to the non-streaming
    // `h5ad_to_scx` path. The backed-AnnData overrides for obs / var /
    // uns / obsm / varm / obsp / varp are dropped on this path —
    // `from_anndata` (non-streaming) is the canonical caller when
    // those mutations need preserving.
    let mut sink = scx_convert::WarningSink::log();
    if stream {
        py.detach(|| {
            scx_convert::h5ad_to_scx_streaming(&input, &output, &opts, &overrides, &mut sink)
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    } else {
        py.detach(|| scx_convert::h5ad_to_scx(&input, &output, &opts, &mut sink))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    }
    emit_python_warnings(py, &sink)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 8b: SCX → SCX streaming writer (backed and lazy `X`).
// ---------------------------------------------------------------------------

/// Extracted metadata + uns JSON for an AnnData object that wraps an
/// SCX-backed or lazy `X`. Returned by `extract_scx_overrides` so the
/// route functions below can interleave metadata writes with shard
/// reads.
pub(crate) struct ScxOverrides {
    pub(crate) obs: RecordBatch,
    pub(crate) var: RecordBatch,
    pub(crate) obsm: Vec<(String, RecordBatch)>,
    pub(crate) varm: Vec<(String, RecordBatch)>,
    pub(crate) obsp: Vec<(String, RecordBatch)>,
    pub(crate) varp: Vec<(String, RecordBatch)>,
    pub(crate) uns: Option<serde_json::Value>,
}

pub(crate) fn extract_scx_overrides(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    uns_format_parsed: UnsFormat,
) -> PyResult<ScxOverrides> {
    let obs = pandas_to_record_batch(py, &adata.getattr("obs")?)?;
    let var = pandas_to_record_batch(py, &adata.getattr("var")?)?;
    let obsm = extract_dense_mapping(py, adata, "obsm")?;
    let varm = extract_dense_mapping(py, adata, "varm")?;
    let obsp = extract_coo_mapping(py, adata, "obsp")?;
    let varp = extract_coo_mapping(py, adata, "varp")?;
    let uns = extract_uns_value(py, adata, uns_format_parsed)?;
    Ok(ScxOverrides {
        obs,
        var,
        obsm,
        varm,
        obsp,
        varp,
        uns,
    })
}
