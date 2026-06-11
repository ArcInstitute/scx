// SCX -> SCX streaming rewrite (backed / lazy sources).
//
// Extracted from the former pyscx/src/anndata.rs (T5.7).

use arrow::array::RecordBatch;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

use scx_codec::CodecId;
use scx_format::section::SectionType;
use scx_format::{ModalityType, ProvenanceEntry, ScxReader, ScxWriter};

use crate::to_pyerr;

use crate::convert::*;

/// Route an AnnData with `adata.X = ScxBackedSparseDataset` through
/// an SCX → SCX streaming writer.
///
/// Two modes:
///
/// * **Byte-passthrough**: when the source and target shard layouts
///   agree (same `shard_target_rows`, same codec, no row deletions,
///   no column projection, source is built from a single modality
///   with `modality_id == None`), pre-encoded CSR shards are copied
///   from the source file into the target writer via
///   [`ScxWriter::copy_section_verbatim`]. No decode + re-encode.
/// * **Decode + encode**: otherwise the writer iterates the
///   wrapper's user-visible shard boundaries, calls
///   `wrapper[start:end]` to materialise each shard as a scipy CSR
///   (deletions and column projection already applied by the
///   wrapper), then hands it to [`scx_format::encode_one_shard`]
///   plus [`ScxWriter::write_preencoded_shard`].
///
/// `shard_size_overridden` reflects whether the caller passed
/// `shard_size=…`. When `explicit_codec` is `None` and `shard_size`
/// was not overridden *and* every other precondition holds, the
/// rewrite is byte-faithful; otherwise it falls back to
/// decode + encode.
#[allow(clippy::too_many_arguments)]
pub(crate) fn route_scx_backed_to_scx(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    backed: &crate::backed::ScxBackedSparseDataset,
    out_path: &str,
    explicit_codec: Option<CodecId>,
    shard_target_rows: u32,
    csc_policy: scx_format::CscPolicy,
    csc_cols_per_shard: usize,
    uns_format_parsed: UnsFormat,
    shard_size_overridden: bool,
) -> PyResult<()> {
    let src_path = backed.source_path().ok_or_else(|| {
        pyo3::exceptions::PyNotImplementedError::new_err(
            "ScxBackedSparseDataset has no known source path; the SCX → SCX writer needs a \
             path-backed wrapper (constructed via pyscx.open(path).to_anndata(backed=True)).",
        )
    })?;
    let src_path_owned = src_path.to_path_buf();

    let src_reader = ScxReader::open(&src_path_owned).map_err(to_pyerr)?;
    let src_header = src_reader.header().clone();
    let src_n_obs = src_header.n_obs;
    let src_n_vars = src_header.n_vars;
    let src_codec_id = src_header.codec_id;
    let src_shard_rows = src_header.shard_target_rows;
    let src_has_csc = src_header.has_csc();
    let modality_id = backed.modality_id;

    // User-visible dimensions after any row deletions / column
    // projection. Used by the decode-encode path's output header so
    // the new file's shape matches what the AnnData wrapper exposes.
    let (out_n_obs_visible, out_n_vars_visible) =
        (backed.shape_val.0 as u64, backed.shape_val.1 as u64);

    // Passthrough preconditions. Any false → fall through to
    // decode-encode.
    let target_codec_for_passthrough = match explicit_codec {
        Some(c) => c as u8 == src_codec_id,
        None => true,
    };
    let target_shard_rows_matches = if shard_size_overridden {
        shard_target_rows == src_shard_rows
    } else {
        true
    };
    let no_deletions = backed.kept_to_global.is_none();
    let no_projection = backed.col_projection().is_none();
    let single_modality_source = modality_id.is_none();
    let passthrough_ok = target_codec_for_passthrough
        && target_shard_rows_matches
        && no_deletions
        && no_projection
        && single_modality_source;

    // Output header / writer setup. For passthrough, mirror the
    // source's codec / shard_target_rows / index_dtype so the
    // catalog and per-shard headers stay self-consistent. On the
    // decode-encode fallback, default to the source's codec choice
    // (preserves Scx1/Pcodec/etc — only override when the caller
    // passes `codec=`).
    let out_codec_id: u8 = if passthrough_ok {
        src_codec_id
    } else {
        match explicit_codec {
            Some(c) => c as u8,
            None => src_codec_id,
        }
    };
    let out_shard_rows = if passthrough_ok {
        src_shard_rows
    } else {
        shard_target_rows
    };
    // Key index_dtype off the user-visible n_vars (after any column
    // projection), not the source's. Allows u16 when projecting a
    // large source down to a small gene subset, and matches the
    // in-memory path which sees only the visible shape.
    let out_index_dtype = if passthrough_ok {
        src_header.index_dtype
    } else if out_n_vars_visible <= 65535 {
        0
    } else {
        1
    };

    let codec_for_header = CodecId::from_u8(out_codec_id).ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "unknown codec id {out_codec_id} from source SCX header"
        ))
    })?;
    // Output dimensions: passthrough mirrors the source header
    // (preconditions guarantee no deletions / projection). The
    // decode-encode path uses the wrapper's user-visible shape so
    // the rewrite drops any deleted rows and respects column
    // projection.
    let (out_n_obs, out_n_vars) = if passthrough_ok {
        (src_n_obs, src_n_vars)
    } else {
        (out_n_obs_visible, out_n_vars_visible)
    };
    let header = build_output_header(
        out_n_obs,
        out_n_vars,
        out_shard_rows,
        codec_for_header,
        out_index_dtype,
        src_header.format_version,
    );

    let mut writer = ScxWriter::new(out_path, header).map_err(to_pyerr)?;

    // Extract metadata overrides up front so the writer can interleave
    // metadata writes with shard I/O in the canonical order.
    let ov = extract_scx_overrides(py, adata, uns_format_parsed)?;

    // CSC sidecar policy. Resolve `Auto` against the output shape so a
    // CSC sidecar is (re)built only when the rewritten dataset is large
    // enough to benefit.
    let csc_build = csc_policy.should_build_csc(out_n_obs, out_n_vars);
    let csc_dropped = src_has_csc && !csc_build;
    if csc_dropped {
        warn_csc_dropped(py);
    }

    // Write obs/var first.
    py.detach(|| -> Result<(), scx_format::ScxError> {
        writer.write_obs(&ov.obs)?;
        writer.write_var(&ov.var)?;
        Ok(())
    })
    .map_err(to_pyerr)?;

    let n_vars_u32 = u32::try_from(out_n_vars)
        .map_err(|_| PyRuntimeError::new_err(format!("n_vars {out_n_vars} exceeds u32::MAX")))?;
    // Pin the per-shard encode codec to the header's codec so the
    // recorded `codec_id` and the actual shard encodings stay
    // consistent. When the user passed an explicit codec we use it;
    // otherwise we use the source's codec (which `out_codec_id` now
    // mirrors).
    let codec_for_encode = Some(codec_for_header);

    if passthrough_ok {
        // Byte-passthrough. Iterate source CSR shards in row order;
        // copy each verbatim. `modality_id == None` is already
        // enforced above, so we can write at the global modality
        // (current_modality_id == 0).
        let csr_shards = src_reader.catalog().csr_shards_sorted();
        py.detach(|| -> Result<(), scx_format::ScxError> {
            for entry in csr_shards {
                let bytes = src_reader.read_raw_shard_bytes(entry)?;
                writer.copy_section_verbatim(entry, bytes)?;
            }
            Ok(())
        })
        .map_err(to_pyerr)?;
    } else {
        // Decode + encode. Drive iteration over user-visible shard
        // boundaries so deletions / column projection already apply
        // via the wrapper's `__getitem__`.
        let bounds = compute_wrapper_boundaries_backed(backed, out_shard_rows);
        let adata_x = adata.getattr("X")?;
        for (i, (start, end)) in bounds.iter().enumerate() {
            let py_slice = pyo3::types::PySlice::new(py, *start as isize, *end as isize, 1);
            // Use the Python-visible wrapper to honour deletion /
            // projection semantics. Calling through PyAny gives us
            // the wrapper's __getitem__ (returns scipy CSR).
            let shard_obj = adata_x.call_method1("__getitem__", (py_slice,))?;
            let pre = decompose_scipy_csr_with(py, &shard_obj, |indptr, indices, data| {
                py.detach(|| {
                    scx_format::encode_one_shard(
                        indptr,
                        indices,
                        data,
                        codec_for_encode,
                        out_index_dtype,
                        n_vars_u32,
                        *start as u64,
                        SectionType::CsrShard,
                        ModalityType::Rna,
                        format!("X_shard_{i}"),
                    )
                })
                .map_err(to_pyerr)
            })?;
            py.detach(|| writer.write_preencoded_shard(pre))
                .map_err(to_pyerr)?;
        }
    }

    // Layers (decode-encode, never passthrough — keeps the byte
    // path bounded to X). `adata.layers` from a backed AnnData
    // contains `ScxBackedLayerDataset` instances which slice-via-
    // `__getitem__` exactly like X.
    stream_write_layers(
        py,
        adata,
        &mut writer,
        out_n_obs,
        out_n_vars,
        out_shard_rows,
        codec_for_encode,
        out_index_dtype,
    )?;

    // Write remaining metadata (obsm / varm / obsp / varp / uns) after
    // the X shards, matching the canonical layout. obsm / varm / obsp /
    // varp are emitted as row-sharded sections so the on-disk layout
    // matches what the streaming pipeline produces (readers handle
    // both sharded and legacy single-section layouts transparently).
    py.detach(|| -> Result<(), scx_format::ScxError> {
        for (k, b) in &ov.obsm {
            for_each_dense_shard(
                b,
                out_shard_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_obsm_shard(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        for (k, b) in &ov.varm {
            for_each_dense_shard(
                b,
                out_shard_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_varm_shard(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        for (k, b) in &ov.obsp {
            for_each_coo_shard(
                b,
                out_shard_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_obsp_shard_coo(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        for (k, b) in &ov.varp {
            for_each_coo_shard(
                b,
                out_shard_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_varp_shard_coo(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        if let Some(ref uns_json) = ov.uns {
            writer.write_uns(uns_json)?;
        }
        Ok(())
    })
    .map_err(to_pyerr)?;

    // Provenance: x_source / passthrough / source_path / csc_dropped.
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let params_json = serde_json::json!({
        "x_source": "backed",
        "passthrough": passthrough_ok,
        "source_path": src_path_owned.display().to_string(),
        "csc_dropped": csc_dropped,
    });
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp,
            action: "from_anndata".to_string(),
            tool: format!("pyscx {}", env!("CARGO_PKG_VERSION")),
            params_json: params_json.to_string(),
            input_checksums: vec![],
        }])
        .map_err(to_pyerr)?;

    writer.finish().map_err(to_pyerr)?;

    // Optional CSC sidecar rebuild over the just-written file.
    if csc_build {
        py.detach(|| {
            scx_ops::rebuild_csc_inplace(std::path::Path::new(out_path), csc_cols_per_shard, "4G")
                .map_err(|e| e.to_string())
        })
        .map_err(|e| PyRuntimeError::new_err(format!("rebuild_csc_inplace failed: {e}")))?;
    }
    Ok(())
}

/// Route an AnnData with `adata.X = ScxLazyTransformedDataset`
/// through an SCX → SCX streaming writer.
///
/// Always decode + encode. The wrapper's `__getitem__` returns the
/// transformed scipy CSR for the requested row slice; the writer
/// hands it to `encode_one_shard` and `write_preencoded_shard`.
/// Any source CSC sidecar is invalidated by the transforms and is
/// dropped with a `UserWarning` unless `csc="always"` is passed (in
/// which case it is rebuilt post-finalise).
#[allow(clippy::too_many_arguments)]
pub(crate) fn route_scx_lazy_to_scx(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    lazy: &crate::lazy_transform::ScxLazyTransformedDataset,
    out_path: &str,
    explicit_codec: Option<CodecId>,
    shard_target_rows: u32,
    csc_policy: scx_format::CscPolicy,
    csc_cols_per_shard: usize,
    uns_format_parsed: UnsFormat,
) -> PyResult<()> {
    let (n_obs_usize, n_vars_usize) = lazy.shape_val;
    let n_obs = n_obs_usize as u64;
    let n_vars = n_vars_usize as u64;
    // Resolve `Auto` against the (lazy) source shape.
    let csc_build = csc_policy.should_build_csc(n_obs, n_vars);

    // Default codec to the source SCX's choice when known (preserves
    // Scx1 / Pcodec / etc through the rewrite). Falls back to Zstd
    // when no source path is recorded (the lazy wrapper can in
    // principle be built without one).
    // Open the source once for both codec and format_version. The lazy
    // per-shard re-encode applies value transforms only (no column reorder),
    // so it preserves a canonical source but cannot canonicalize a pre-v3
    // one — gate the v3 stamp on the source version.
    let src_header_meta: Option<(Option<CodecId>, u16)> = lazy.source_path().and_then(|p| {
        ScxReader::open(p).ok().map(|r| {
            (
                CodecId::from_u8(r.header().codec_id),
                r.header().format_version,
            )
        })
    });
    let src_codec: Option<CodecId> = src_header_meta.and_then(|(c, _)| c);
    let src_format_version: u16 = src_header_meta
        .map(|(_, v)| v)
        .unwrap_or(scx_format::CURRENT_FORMAT_VERSION);
    let out_codec = explicit_codec.or(src_codec).unwrap_or(CodecId::Zstd);
    let index_dtype: u8 = if n_vars <= 65535 { 0 } else { 1 };
    let n_vars_u32 = u32::try_from(n_vars)
        .map_err(|_| PyRuntimeError::new_err(format!("n_vars {n_vars} exceeds u32::MAX")))?;

    let header = build_output_header(
        n_obs,
        n_vars,
        shard_target_rows,
        out_codec,
        index_dtype,
        src_format_version,
    );
    let mut writer = ScxWriter::new(out_path, header).map_err(to_pyerr)?;

    // Source CSC sidecar (if any) is always invalidated by the
    // transform chain. Warn unless the user opted into a rebuild.
    let src_has_csc = lazy.backed_csc.is_some();
    let csc_dropped = src_has_csc && !csc_build;
    if csc_dropped {
        warn_csc_dropped(py);
    }

    let ov = extract_scx_overrides(py, adata, uns_format_parsed)?;

    py.detach(|| -> Result<(), scx_format::ScxError> {
        writer.write_obs(&ov.obs)?;
        writer.write_var(&ov.var)?;
        Ok(())
    })
    .map_err(to_pyerr)?;

    // Lazy transforms always force decode + encode. Iterate the
    // wrapper's user-visible shard boundaries so any deletion vector
    // already applies. Pin per-shard encode codec to the header
    // codec so the recorded `codec_id` and the actual encodings
    // stay consistent.
    let codec_for_encode = Some(out_codec);
    let bounds = compute_wrapper_boundaries_lazy(lazy, shard_target_rows);
    let adata_x = adata.getattr("X")?;
    for (i, (start, end)) in bounds.iter().enumerate() {
        let py_slice = pyo3::types::PySlice::new(py, *start as isize, *end as isize, 1);
        let shard_obj = adata_x.call_method1("__getitem__", (py_slice,))?;
        let pre = decompose_scipy_csr_with(py, &shard_obj, |indptr, indices, data| {
            py.detach(|| {
                scx_format::encode_one_shard(
                    indptr,
                    indices,
                    data,
                    codec_for_encode,
                    index_dtype,
                    n_vars_u32,
                    *start as u64,
                    SectionType::CsrShard,
                    ModalityType::Rna,
                    format!("X_shard_{i}"),
                )
            })
            .map_err(to_pyerr)
        })?;
        py.detach(|| writer.write_preencoded_shard(pre))
            .map_err(to_pyerr)?;
    }

    // Layers — never transformed by the lazy X chain, so we just
    // stream them through the same decode-encode pipeline as X.
    stream_write_layers(
        py,
        adata,
        &mut writer,
        n_obs,
        n_vars,
        shard_target_rows,
        codec_for_encode,
        index_dtype,
    )?;

    py.detach(|| -> Result<(), scx_format::ScxError> {
        for (k, b) in &ov.obsm {
            for_each_dense_shard(
                b,
                shard_target_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_obsm_shard(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        for (k, b) in &ov.varm {
            for_each_dense_shard(
                b,
                shard_target_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_varm_shard(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        for (k, b) in &ov.obsp {
            for_each_coo_shard(
                b,
                shard_target_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_obsp_shard_coo(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        for (k, b) in &ov.varp {
            for_each_coo_shard(
                b,
                shard_target_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_varp_shard_coo(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        if let Some(ref uns_json) = ov.uns {
            writer.write_uns(uns_json)?;
        }
        Ok(())
    })
    .map_err(to_pyerr)?;

    // Provenance: lazy_transforms summary.
    let transforms_repr: Vec<serde_json::Value> = lazy
        .transforms()
        .iter()
        .map(|t| match t {
            crate::lazy_transform::Transform::NormalizeTotal {
                row_sums,
                target_sum,
            } => serde_json::json!({
                "name": "normalize_total",
                "params": { "row_sums_len": row_sums.len(), "target_sum": target_sum },
            }),
            crate::lazy_transform::Transform::Log1p => serde_json::json!({
                "name": "log1p",
                "params": {},
            }),
            crate::lazy_transform::Transform::RowScale { factors } => serde_json::json!({
                "name": "row_scale",
                "params": { "factors_len": factors.len() },
            }),
        })
        .collect();
    let source_path_json: serde_json::Value = lazy
        .source_path()
        .map(|p| serde_json::Value::String(p.display().to_string()))
        .unwrap_or(serde_json::Value::Null);
    let params_json = serde_json::json!({
        "x_source": "lazy",
        "passthrough": false,
        "source_path": source_path_json,
        "lazy_transforms": transforms_repr,
        "csc_dropped": csc_dropped,
    });
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp,
            action: "from_anndata".to_string(),
            tool: format!("pyscx {}", env!("CARGO_PKG_VERSION")),
            params_json: params_json.to_string(),
            input_checksums: vec![],
        }])
        .map_err(to_pyerr)?;

    writer.finish().map_err(to_pyerr)?;

    if csc_build {
        py.detach(|| {
            scx_ops::rebuild_csc_inplace(std::path::Path::new(out_path), csc_cols_per_shard, "4G")
                .map_err(|e| e.to_string())
        })
        .map_err(|e| PyRuntimeError::new_err(format!("rebuild_csc_inplace failed: {e}")))?;
    }
    Ok(())
}

/// Compute output shard boundaries for the backed decode-encode
/// path. Sized by the **target** `shard_target_rows` (not the
/// source's), since the user may have asked for a different chunk
/// size — and that's the only reason we're on the decode-encode path
/// instead of byte-passthrough.
pub(crate) fn compute_wrapper_boundaries_backed(
    backed: &crate::backed::ScxBackedSparseDataset,
    target_shard_rows: u32,
) -> Vec<(usize, usize)> {
    let n_obs = backed.shape_val.0;
    chunk_boundaries(n_obs, target_shard_rows as usize)
}

pub(crate) fn compute_wrapper_boundaries_lazy(
    lazy: &crate::lazy_transform::ScxLazyTransformedDataset,
    target_shard_rows: u32,
) -> Vec<(usize, usize)> {
    let n_obs = lazy.shape_val.0;
    chunk_boundaries(n_obs, target_shard_rows as usize)
}

pub(crate) fn chunk_boundaries(n_obs: usize, target_rows: usize) -> Vec<(usize, usize)> {
    if n_obs == 0 || target_rows == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n_obs.div_ceil(target_rows));
    let mut start = 0usize;
    while start < n_obs {
        let end = (start + target_rows).min(n_obs);
        out.push((start, end));
        start = end;
    }
    out
}

/// Stream `adata.layers` shard-by-shard into the writer using the
/// same decode-encode pattern as the X path. Shared by both the
/// backed and lazy SCX → SCX routes — neither transforms layers,
/// so the logic is identical.
///
/// Each layer wrapper (`ScxBackedLayerDataset` or a scipy CSR) must
/// support `__getitem__(slice)` and report `(n_obs, n_vars)` via
/// `.shape`. The shape must match the output X dims; otherwise we
/// raise a `ValueError` matching the in-memory path's contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stream_write_layers(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    writer: &mut ScxWriter,
    out_n_obs: u64,
    out_n_vars: u64,
    out_shard_rows: u32,
    codec_for_encode: Option<CodecId>,
    index_dtype: u8,
) -> PyResult<()> {
    let layers = match adata.getattr("layers") {
        Ok(l) => l,
        Err(_) => return Ok(()),
    };
    let keys: Vec<String> = py
        .import("builtins")?
        .call_method1("list", (layers.call_method0("keys")?,))?
        .extract()?;
    if keys.is_empty() {
        return Ok(());
    }

    let n_vars_u32 = u32::try_from(out_n_vars)
        .map_err(|_| PyRuntimeError::new_err(format!("n_vars {out_n_vars} exceeds u32::MAX")))?;
    let bounds = chunk_boundaries(out_n_obs as usize, out_shard_rows as usize);

    for layer_name in &keys {
        let layer = layers.call_method1("__getitem__", (layer_name,))?;
        let l_shape: (u64, u64) = layer.getattr("shape")?.extract()?;
        if l_shape != (out_n_obs, out_n_vars) {
            return Err(PyValueError::new_err(format!(
                "Layer '{layer_name}' has shape ({}, {}), expected ({}, {})",
                l_shape.0, l_shape.1, out_n_obs, out_n_vars
            )));
        }

        for (i, (start, end)) in bounds.iter().enumerate() {
            let py_slice = pyo3::types::PySlice::new(py, *start as isize, *end as isize, 1);
            let shard_obj = layer.call_method1("__getitem__", (py_slice,))?;
            let pre = decompose_scipy_csr_with(py, &shard_obj, |indptr, indices, data| {
                py.detach(|| {
                    scx_format::encode_one_shard(
                        indptr,
                        indices,
                        data,
                        codec_for_encode,
                        index_dtype,
                        n_vars_u32,
                        *start as u64,
                        SectionType::LayerCsrShard,
                        ModalityType::Rna,
                        format!("{layer_name}_shard_{i}"),
                    )
                })
                .map_err(to_pyerr)
            })?;
            py.detach(|| writer.write_preencoded_shard(pre))
                .map_err(to_pyerr)?;
        }
    }
    Ok(())
}

/// Mutation detection for the backed-routing path. Compares the
/// top-level key set of `getattr(adata, attr)` against the on-disk
/// h5py group at `/<attr>`. Returns `true` when the two key sets match
/// exactly (sender hasn't mutated this section in Python, so the
/// pipeline can stream from the source h5ad instead of materialising
/// the Python copy).
///
/// Cheapness invariants:
/// 1. `adata.obsm.keys()` on a backed AnnData lists the underlying
///    h5py group members without loading any dataset.
/// 2. `h5_file[attr].keys()` is a pure metadata read on h5py.
///
/// We never touch `adata.obsm[key]` here — that would force the very
/// h5py-to-numpy read we're trying to avoid.
///
/// Limitation: same-key replacements ("user did `adata.obsm['X_pca'] =
/// new_array`" without renaming) aren't detected. Document this with a
/// `UserWarning` on the routing path so users have a breadcrumb.
///
/// Gated behind the `hdf5` feature because the sole caller
/// (`route_backed_anndata_to_streaming`) is. Without `hdf5` the
/// streaming backed-AnnData path falls through to the in-memory branch
/// in `from_anndata_impl` and this helper would be dead code.
#[cfg(feature = "hdf5")]
pub(crate) fn section_keys_match(
    py: Python<'_>,
    h5_file: &Bound<'_, PyAny>,
    adata: &Bound<'_, PyAny>,
    attr: &str,
) -> PyResult<bool> {
    let py_keys: Vec<String> = match adata.getattr(attr) {
        Ok(section) => match section.call_method0("keys") {
            Ok(keys_obj) => py
                .import("builtins")?
                .call_method1("list", (keys_obj,))?
                .extract()
                .unwrap_or_default(),
            Err(_) => Vec::new(),
        },
        Err(_) => Vec::new(),
    };
    let disk_keys: Vec<String> = match h5_file.get_item(attr) {
        Ok(group) => match group.call_method0("keys") {
            Ok(keys_obj) => py
                .import("builtins")?
                .call_method1("list", (keys_obj,))?
                .extract()
                .unwrap_or_default(),
            Err(_) => Vec::new(),
        },
        Err(_) => Vec::new(),
    };
    let py_set: std::collections::BTreeSet<&String> = py_keys.iter().collect();
    let disk_set: std::collections::BTreeSet<&String> = disk_keys.iter().collect();
    Ok(py_set == disk_set)
}

/// Slice a dense obsm/varm `RecordBatch` into row-aligned shards and
/// emit each via `f`. Used by the SCX-backed / lazy / in-memory
/// `from_anndata` paths so all pyscx-produced SCX files share the
/// sharded on-disk layout the streaming pipeline emits.
///
/// The callback signature is `(shard_idx, row_start, n_shard_rows,
/// n_rows_total, batch)`; `n_shard_rows == batch.num_rows()` for dense
/// but the parameter is passed explicitly so the writer's contiguity
/// metadata is sourced from one place.
pub(crate) fn for_each_dense_shard<F>(
    batch: &RecordBatch,
    shard_target_rows: u32,
    mut f: F,
) -> std::result::Result<(), scx_format::ScxError>
where
    F: FnMut(u32, u64, u64, u64, &RecordBatch) -> std::result::Result<(), scx_format::ScxError>,
{
    let n_rows = batch.num_rows();
    let n_total = n_rows as u64;
    if n_rows == 0 {
        return f(0, 0, 0, 0, batch);
    }
    let step = shard_target_rows.max(1) as usize;
    let mut shard_idx = 0u32;
    let mut row_start = 0usize;
    while row_start < n_rows {
        let n = (n_rows - row_start).min(step);
        let shard = batch.slice(row_start, n);
        f(shard_idx, row_start as u64, n as u64, n_total, &shard)?;
        row_start += n;
        shard_idx += 1;
    }
    Ok(())
}

/// Slice a COO obsp/varp `RecordBatch` into row-shards keyed by the
/// `row` column. Buckets non-zero triples by `row / shard_target_rows`
/// then emits one shard per non-empty bucket. Used by the in-Python
/// override paths to keep on-disk obsp/varp layout symmetric with the
/// streaming pipeline. Returns `ScxError` (not `PyResult`) so callers
/// can drive it from inside `py.detach(...)`.
pub(crate) fn for_each_coo_shard<F>(
    batch: &RecordBatch,
    shard_target_rows: u32,
    mut f: F,
) -> std::result::Result<(), scx_format::ScxError>
where
    F: FnMut(u32, u64, u64, u64, &RecordBatch) -> std::result::Result<(), scx_format::ScxError>,
{
    use arrow::array::{Array, Float32Array, Int32Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};

    let invalid = |msg: String| {
        scx_format::ScxError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, msg))
    };

    // Width-generic: accept both v1 (Int32) and v2 (Int64) row/col columns,
    // and emit shards with the same coord dtype as the input. Reuse
    // `coo_coords_from_batch` so the inner bucketing loop dispatches on
    // `CooCoordsRef` (static match) rather than `Box<dyn Fn>` (per-element
    // vtable call + heap alloc).
    let coords = coo_coords_from_batch(batch).map_err(|e| invalid(e.to_string()))?;
    let coord_dt = match &coords {
        CooCoordsRef::Int32(_, _) => DataType::Int32,
        CooCoordsRef::Int64(_, _) => DataType::Int64,
    };
    let data_arr = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| invalid("sparse override: column 2 must be Float32".into()))?;
    let nnz = coords.len();

    let metadata = batch.schema_ref().metadata().clone();
    let n_rows: usize = metadata
        .get("n_rows")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| invalid("sparse override: missing 'n_rows' metadata".into()))?;
    let n_cols: usize = metadata
        .get("n_cols")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| invalid("sparse override: missing 'n_cols' metadata".into()))?;

    let step = shard_target_rows.max(1) as usize;
    if n_rows == 0 {
        return f(0, 0, 0, 0, batch);
    }

    // Bucket count scales with the logical row axis. For atlas-scale v2
    // obsp (axis ≥ 2^31) this would allocate hundreds of thousands of
    // empty `Vec`s up-front — fail fast with a clear message rather than
    // silently OOM. Real callers either route through the streaming
    // converter (which writes shards incrementally without this bucket
    // table) or use Phase 6's dedicated huge-obsp path.
    let n_shards = n_rows.div_ceil(step);
    const MAX_BUCKETS: usize = 1_000_000;
    if n_shards > MAX_BUCKETS {
        return Err(invalid(format!(
            "for_each_coo_shard: logical n_rows={n_rows} would require {n_shards} shard buckets \
             (cap {MAX_BUCKETS}). Route this obsp / varp through the streaming converter \
             or pre-split it into row bands before re-shading."
        )));
    }
    let mut bucket_row: Vec<Vec<i64>> = (0..n_shards).map(|_| Vec::new()).collect();
    let mut bucket_col: Vec<Vec<i64>> = (0..n_shards).map(|_| Vec::new()).collect();
    let mut bucket_data: Vec<Vec<f32>> = (0..n_shards).map(|_| Vec::new()).collect();
    for i in 0..nnz {
        let r = coords.row_i64(i);
        if r < 0 {
            return Err(invalid(format!("sparse override: negative row index {r}")));
        }
        let r_us = r as usize;
        let shard = r_us / step;
        if shard >= n_shards {
            return Err(invalid(format!(
                "sparse override: row {r} exceeds n_rows={n_rows}"
            )));
        }
        bucket_row[shard].push(r);
        bucket_col[shard].push(coords.col_i64(i));
        bucket_data[shard].push(data_arr.value(i));
    }

    let n_total = n_rows as u64;
    for shard_idx in 0..n_shards {
        let row_start = shard_idx * step;
        let n_shard_rows = step.min(n_rows - row_start);
        let schema = Arc::new(Schema::new_with_metadata(
            vec![
                Field::new("row", coord_dt.clone(), false),
                Field::new("col", coord_dt.clone(), false),
                Field::new("data", DataType::Float32, false),
            ],
            std::collections::HashMap::from([
                ("n_rows".to_string(), n_rows.to_string()),
                ("n_cols".to_string(), n_cols.to_string()),
            ]),
        ));
        let row_i64_taken = std::mem::take(&mut bucket_row[shard_idx]);
        let col_i64_taken = std::mem::take(&mut bucket_col[shard_idx]);
        let (row_array, col_array): (Arc<dyn Array>, Arc<dyn Array>) = match &coord_dt {
            DataType::Int32 => (
                Arc::new(Int32Array::from(
                    row_i64_taken
                        .into_iter()
                        .map(|v| v as i32)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Int32Array::from(
                    col_i64_taken
                        .into_iter()
                        .map(|v| v as i32)
                        .collect::<Vec<_>>(),
                )),
            ),
            DataType::Int64 => (
                Arc::new(Int64Array::from(row_i64_taken)),
                Arc::new(Int64Array::from(col_i64_taken)),
            ),
            _ => unreachable!(),
        };
        let shard_batch = RecordBatch::try_new(
            schema,
            vec![
                row_array,
                col_array,
                Arc::new(Float32Array::from(std::mem::take(
                    &mut bucket_data[shard_idx],
                ))),
            ],
        )
        .map_err(scx_format::ScxError::Arrow)?;
        f(
            shard_idx as u32,
            row_start as u64,
            n_shard_rows as u64,
            n_total,
            &shard_batch,
        )?;
    }
    Ok(())
}

/// Helper for the backed-routing path. Reads a dense mapping
/// (`obsm` / `varm`) from a Python AnnData and returns
/// `Vec<(name, RecordBatch)>`. Missing groups → empty Vec.
pub(crate) fn extract_dense_mapping(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    attr: &str,
) -> PyResult<Vec<(String, RecordBatch)>> {
    let group = match adata.getattr(attr) {
        Ok(g) => g,
        Err(_) => return Ok(Vec::new()),
    };
    let keys: Vec<String> = py
        .import("builtins")?
        .call_method1("list", (group.call_method0("keys")?,))?
        .extract()?;
    keys.iter()
        .map(|key| {
            let arr = group.call_method1("__getitem__", (key,))?;
            let pd = py.import("pandas")?;
            let df = pd.call_method1("DataFrame", (&arr,))?;
            let batch = pandas_to_record_batch(py, &df)?;
            Ok((key.clone(), batch))
        })
        .collect()
}

/// Helper for the backed-routing path. Reads a sparse pairwise
/// mapping (`obsp` / `varp`) as COO RecordBatches. Missing groups →
/// empty Vec.
pub(crate) fn extract_coo_mapping(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    attr: &str,
) -> PyResult<Vec<(String, RecordBatch)>> {
    let group = match adata.getattr(attr) {
        Ok(g) => g,
        Err(_) => return Ok(Vec::new()),
    };
    let keys: Vec<String> = py
        .import("builtins")?
        .call_method1("list", (group.call_method0("keys")?,))?
        .extract()?;
    keys.iter()
        .map(|key| {
            let mat = group.call_method1("__getitem__", (key,))?;
            let batch = sparse_to_coo_record_batch(py, &mat)?;
            Ok((key.clone(), batch))
        })
        .collect()
}

/// Helper for the backed-routing path. Extracts `uns` from a Python
/// AnnData into an optional `serde_json::Value`. Returns `None` if
/// `uns` is empty (no `__scx_uns__` section written), matching the
/// non-backed path.
pub(crate) fn extract_uns_value(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    uns_format_parsed: UnsFormat,
) -> PyResult<Option<serde_json::Value>> {
    let uns = adata.getattr("uns")?;
    let uns_len: usize = uns.call_method0("__len__")?.extract()?;
    if uns_len == 0 {
        return Ok(None);
    }
    let np = py.import("numpy")?;
    let np_generic = np.getattr("generic")?;
    let np_ndarray = np.getattr("ndarray")?;
    let mut ctx = UnsWriteCtx::new(uns_format_parsed, &np_generic, &np_ndarray);
    Ok(Some(normalize_uns_value(&uns, "uns", &mut ctx)?))
}
