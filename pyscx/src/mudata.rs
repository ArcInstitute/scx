// pyscx mudata bindings (Phase D.3 / D.4 of MULTIMODAL-SUPPORT.md).
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
use pyo3::exceptions::{PyImportError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scx_codec::CodecId;
use scx_format::header::{FileHeader, MAGIC};
use scx_format::modality::ModalityType;
use scx_format::provenance::ProvenanceEntry;
use scx_format::section::SectionType;
use scx_format::select_codec_for_modality;
use scx_format::writer::ScxWriter;
use scx_format::ScxReader;

use crate::anndata::{
    csr_to_scipy, obsm_batch_to_numpy, pandas_to_record_batch, pyarrow_table_to_pandas,
    record_batch_to_pyarrow,
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

/// Convenience: same heuristic as the CLI's `mudata_pipeline.rs` but
/// invoked from Python. Maps a modality name to a `ModalityType`
/// based on common naming conventions for CITE-seq / multiome.
fn infer_modality_type(name: &str) -> ModalityType {
    let lower = name.to_ascii_lowercase();
    if lower.contains("atac") || lower.contains("peak") {
        ModalityType::Atac
    } else if lower.contains("adt") || lower.contains("protein") || lower.contains("antibody") {
        ModalityType::Protein
    } else if lower.contains("spatial") {
        ModalityType::Spatial
    } else if lower.contains("methyl") {
        ModalityType::Methylation
    } else if lower == "rna" || lower == "gex" || lower.contains("expression") {
        ModalityType::Rna
    } else {
        ModalityType::Custom
    }
}

/// Materialise an `ScxReader` as a `mudata.MuData` object. Iterates
/// `modality_names()`, builds an AnnData per modality via the
/// existing zero-copy CSR path, and attaches them to a `MuData(...)`
/// with the shared global obs.
pub fn to_mudata<'py>(py: Python<'py>, reader: &ScxReader) -> PyResult<Bound<'py, PyAny>> {
    if !reader.is_multimodal() {
        return Err(PyRuntimeError::new_err(
            "to_mudata() requires a multimodal SCX file (n_modalities > 0); \
             use to_anndata() for single-modality files",
        ));
    }

    let anndata_mod = py.import("anndata")?;
    let mudata_mod = import_mudata(py)?;

    // Build the global obs once; per-modality AnnData objects share it.
    let global_obs = match reader.read_obs() {
        Ok(batch) => {
            let table = record_batch_to_pyarrow(py, &batch)?;
            Some(pyarrow_table_to_pandas(&table)?)
        }
        Err(scx_format::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

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

        // X
        let csr = reader
            .read_all_csr_shards_for(modality_id)
            .map_err(to_pyerr)?;
        let x = csr_to_scipy(py, csr)?;

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

/// Implementation of `pyscx.from_mudata(mu, path, ...)`. Mirrors
/// `from_anndata_impl` but iterates `mu.mod` and emits one modality
/// per AnnData.
#[allow(clippy::too_many_arguments)]
pub fn from_mudata_impl(
    py: Python<'_>,
    mu: &Bound<'_, PyAny>,
    path: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
    csc: &str,
    csc_cols_per_shard: usize,
) -> PyResult<()> {
    let _ = csc_cols_per_shard; // CSC for h5mu input is a Phase D follow-on
    let csc_always = match csc {
        "off" => false,
        "always" => true,
        other => {
            return Err(PyRuntimeError::new_err(format!(
                "invalid csc value: {other}; use 'off' or 'always'"
            )));
        }
    };
    if csc_always {
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

        let indptr: Vec<i64> = numpy::PyReadonlyArray1::extract_bound(&indptr_np)?
            .as_slice()?
            .to_vec();
        let indices: Vec<i32> = numpy::PyReadonlyArray1::extract_bound(&indices_np)?
            .as_slice()?
            .to_vec();
        let data: Vec<f32> = numpy::PyReadonlyArray1::extract_bound(&data_np)?
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

    let header = FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs: n_obs as u64,
        n_vars: max_n_vars,
        nnz: total_nnz,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows,
        codec_id: 0,
        index_dtype,
        endian: 0,
        reserved_padding: 0,
        root_catalog_offset: 0,
        root_catalog_length: 0,
        full_catalog_offset: 0,
        full_catalog_length: 0,
        manifest_sequence: 1,
        prev_catalog_offset: 0,
        file_checksum: 0,
        front_catalog_offset: 0,
        front_catalog_length: 0,
        n_modalities: 0,
        modality_table_offset: 0,
        modality_table_length: 0,
        reserved: [0u8; 112],
    };

    // Open writer + emit sections.
    let mut writer = ScxWriter::new(Path::new(path), header).map_err(to_pyerr)?;
    writer.write_obs(&outer_obs_batch).map_err(to_pyerr)?;

    let explicit_codec = match codec {
        None | Some("auto") => None,
        Some("none") => Some(CodecId::None),
        Some("scx1") => Some(CodecId::Scx1),
        Some("zstd") => Some(CodecId::Zstd),
        Some("lz4") => Some(CodecId::Lz4Shuffle),
        Some("pcodec") => Some(CodecId::Pcodec),
        Some(other) => {
            return Err(PyRuntimeError::new_err(format!(
                "unknown codec '{other}'; use auto, none, scx1, zstd, lz4, or pcodec"
            )));
        }
    };

    for payload in &modalities {
        // Integer-detect per-modality: small UMI / ADT / ATAC counts
        // compress dramatically better when stored as uint8/16/32
        // than as Float32. Mirrors `from_anndata`'s per-shard
        // `detect_value_encoding(shard_data)` path; without this,
        // every modality's X would land as Float32 → Pcodec
        // regardless of `select_codec_for_modality`'s biological
        // routing.
        let value_encoding = scx_codec::value_encoding::detect_value_encoding(&payload.data);
        let raw_values_bytes =
            scx_codec::value_encoding::values_to_raw_bytes(&payload.data, value_encoding);
        let codec_id = match explicit_codec {
            Some(c) => {
                if c == CodecId::Scx1 && !value_encoding.is_integer() {
                    CodecId::Zstd
                } else {
                    c
                }
            }
            None => {
                select_codec_for_modality(&raw_values_bytes, value_encoding, payload.modality_type)
            }
        };

        let modality_id = writer
            .add_modality(
                &payload.name,
                payload.modality_type,
                codec_id,
                value_encoding,
            )
            .map_err(to_pyerr)?;
        writer
            .set_modality_n_vars(modality_id, payload.n_vars as u64)
            .map_err(to_pyerr)?;
        writer
            .write_var_for(modality_id, &payload.var_batch)
            .map_err(to_pyerr)?;

        // Single CSR shard per modality for now (Phase D MVP). The
        // h5mu CLI pipeline shards by `shard_target_rows`; we should
        // do the same here in a follow-on, but a single shard is a
        // valid SCX layout and round-trips correctly.
        let _ = shard_target_rows; // currently single-shard
        let shard_indptr: Vec<u64> = payload.indptr.iter().map(|&v| v as u64).collect();
        let shard_indices: Vec<u32> = payload
            .indices
            .iter()
            .map(|&v| {
                u32::try_from(v)
                    .map_err(|_| PyRuntimeError::new_err(format!("negative column index {v}")))
            })
            .collect::<PyResult<Vec<_>>>()?;

        writer
            .write_csr_shard_for(
                modality_id,
                &shard_indptr,
                &shard_indices,
                &raw_values_bytes,
                codec_id,
                value_encoding,
                0,
            )
            .map_err(to_pyerr)?;

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
