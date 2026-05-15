use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use scx_codec::{CodecId, ValueEncoding};
use scx_format::encode_one_shard;
use scx_format::error::ScxError;
use scx_format::header::{FileHeader, MAGIC};
use scx_format::modality::ModalityType;
use scx_format::provenance::ProvenanceEntry;
use scx_format::section::SectionType;
use scx_format::writer::ScxWriter;
use scx_sparse::{drop_explicit_zeros_inplace, sort_csr_rows_in_place};

use super::detect::{detect_input_format, detect_matrix_format, InputFormat};
use super::dtype::{detect_value_encoding, values_to_raw_bytes};
use super::h5ad_read::{
    read_dataframe_group, read_layers, read_obsm, read_uns, read_varm, read_x_matrix,
};
use super::h5ad_stream::{open_layer_streaming, open_x_streaming};
use super::h5ad_write::write_scx_to_h5ad;
use super::tenx_read::read_tenx_h5;

#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5::Error),

    #[error("SCX error: {0}")]
    Scx(#[from] ScxError),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),

    #[error("format mismatch: expected {expected}, got {got}")]
    FormatMismatch { expected: String, got: String },

    #[error("streaming unsupported: {0}")]
    StreamingUnsupported(String),

    #[error("{0}")]
    Other(String),
}

pub struct ConvertOptions {
    pub shard_target_rows: u32,
    /// Explicit codec override. None = auto-select based on value distribution.
    pub codec: Option<CodecId>,
    /// When `true`, also emit a CSC sidecar at write time (multi-shard
    /// column-major layout). The CSR shards are still written first;
    /// CSC chunks are produced via streaming transpose over the
    /// in-memory CSR data.
    pub csc: bool,
    /// Columns per CSC shard when `csc == true`. `0` disables the
    /// cap (single CSC shard, memory permitting).
    pub csc_cols_per_shard: usize,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        ConvertOptions {
            shard_target_rows: 16384,
            codec: None,
            csc: false,
            csc_cols_per_shard: 5000,
        }
    }
}

pub fn h5ad_to_scx(input: &Path, output: &Path, opts: &ConvertOptions) -> Result<(), ConvertError> {
    let file = hdf5::File::open(input)?;

    // Validate format
    let format = detect_input_format(&file)?;
    if matches!(format, InputFormat::TenX) {
        return Err(ConvertError::FormatMismatch {
            expected: "h5ad".to_string(),
            got: "10x".to_string(),
        });
    }

    // Read X matrix
    let matrix_format = detect_matrix_format(&file)?;
    let (indptr, indices, data, n_obs, n_vars) = read_x_matrix(&file, matrix_format)?;
    let nnz = *indptr.last().unwrap_or(&0) as u64;

    // Detect encoding and codec
    let (value_encoding, codec_id) =
        detect_value_encoding(&data, opts.codec).map_err(ScxError::from)?;
    let index_dtype: u8 = if n_vars <= 65535 { 0 } else { 1 };

    // Build header
    let header = FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs: n_obs as u64,
        n_vars: n_vars as u64,
        nnz,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: opts.shard_target_rows,
        codec_id: codec_id as u8,
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

    let mut writer = ScxWriter::new(output, header)?;

    // Write obs/var
    let obs = read_dataframe_group(&file, "obs")?;
    let var = read_dataframe_group(&file, "var")?;
    writer.write_obs(&obs)?;
    writer.write_var(&var)?;

    // Write CSR shards
    write_csr_shards(
        &mut writer,
        &indptr,
        &indices,
        &data,
        n_obs,
        n_vars,
        opts.shard_target_rows as usize,
        value_encoding,
        codec_id,
        index_dtype,
    )?;

    // Optional CSC sidecar — streaming transpose over the in-memory
    // CSR data, one shard per chunk.
    if opts.csc {
        write_csc_shards_from_csr(
            &mut writer,
            &indptr,
            &indices,
            &data,
            n_obs,
            n_vars,
            value_encoding,
            codec_id,
            opts.csc_cols_per_shard,
        )?;
    }

    // Write optional sections
    if let Ok(obsm_map) = read_obsm(&file) {
        for (name, batch) in &obsm_map {
            writer.write_obsm(name, batch)?;
        }
    }

    if let Ok(uns) = read_uns(&file) {
        writer.write_uns(&uns)?;
    }

    if let Ok(layers) = read_layers(&file) {
        for (layer_name, (l_indptr, l_indices, l_data, l_nobs, l_nvars)) in &layers {
            let (l_enc, l_codec) =
                detect_value_encoding(l_data, opts.codec).map_err(ScxError::from)?;
            let l_index_dtype: u8 = if *l_nvars <= 65535 { 0 } else { 1 };
            write_layer_shards(
                &mut writer,
                l_indptr,
                l_indices,
                l_data,
                *l_nobs,
                *l_nvars,
                opts.shard_target_rows as usize,
                l_enc,
                l_codec,
                l_index_dtype,
                layer_name,
            )?;
        }
    }

    // Write provenance
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    writer.write_provenance(vec![ProvenanceEntry {
        timestamp,
        action: "convert".to_string(),
        tool: "scx-cli".to_string(),
        params_json: format!("{{\"input\":\"{}\",\"format\":\"h5ad\"}}", input.display()),
        input_checksums: vec![],
    }])?;

    writer.finish()?;
    Ok(())
}

pub fn tenx_to_scx(input: &Path, output: &Path, opts: &ConvertOptions) -> Result<(), ConvertError> {
    let file = hdf5::File::open(input)?;

    let format = detect_input_format(&file)?;
    if matches!(format, InputFormat::H5ad) {
        return Err(ConvertError::FormatMismatch {
            expected: "10x".to_string(),
            got: "h5ad".to_string(),
        });
    }

    let tenx = read_tenx_h5(&file)?;
    let nnz = *tenx.indptr.last().unwrap_or(&0) as u64;
    let (value_encoding, codec_id) =
        detect_value_encoding(&tenx.data, opts.codec).map_err(ScxError::from)?;
    let index_dtype: u8 = if tenx.n_genes <= 65535 { 0 } else { 1 };

    let header = FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs: tenx.n_cells as u64,
        n_vars: tenx.n_genes as u64,
        nnz,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: opts.shard_target_rows,
        codec_id: codec_id as u8,
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

    let mut writer = ScxWriter::new(output, header)?;
    writer.write_obs(&tenx.obs)?;
    writer.write_var(&tenx.var)?;

    write_csr_shards(
        &mut writer,
        &tenx.indptr,
        &tenx.indices,
        &tenx.data,
        tenx.n_cells,
        tenx.n_genes,
        opts.shard_target_rows as usize,
        value_encoding,
        codec_id,
        index_dtype,
    )?;

    // Optional CSC sidecar — same streaming transpose as h5ad.
    if opts.csc {
        write_csc_shards_from_csr(
            &mut writer,
            &tenx.indptr,
            &tenx.indices,
            &tenx.data,
            tenx.n_cells,
            tenx.n_genes,
            value_encoding,
            codec_id,
            opts.csc_cols_per_shard,
        )?;
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    writer.write_provenance(vec![ProvenanceEntry {
        timestamp,
        action: "convert".to_string(),
        tool: "scx-cli".to_string(),
        params_json: format!("{{\"input\":\"{}\",\"format\":\"10x\"}}", input.display()),
        input_checksums: vec![],
    }])?;

    writer.finish()?;
    Ok(())
}

pub fn scx_to_h5ad(scx_path: &Path, h5ad_path: &Path) -> Result<(), ConvertError> {
    write_scx_to_h5ad(scx_path, h5ad_path)
}

/// Streaming h5ad → SCX conversion. Reads the input one shard's worth
/// of rows at a time via [`super::h5ad_stream::XStreamReader`] so peak
/// memory is bounded by `shard_target_rows × n_vars × density × ~16
/// bytes` plus the always-resident indptr (`(n_obs + 1) × 8 bytes`).
///
/// Behaviour notes (Phase 4 MVP):
/// - **Sequential**: one shard read → sort → drop-zeros → encode →
///   write per iteration. No concurrent encoder pool yet. Worker
///   parallelism is a planned follow-on if benchmarks show I/O
///   starvation.
/// - **CSC sidecar (`opts.csc == true`)**: not yet supported via
///   streaming. The two-pass design (`rebuild_csc_inplace` on the
///   finished file) requires that helper to be reachable from
///   `scx-convert`; it currently lives in `scx-cli`. Returns
///   [`ConvertError::StreamingUnsupported`] until the move lands.
///   Users can still run `scx build-csc` on the streamed output.
/// - **`varm`** is read and written via [`read_varm`]; `obsp`/`varp`
///   readers are still missing and any present sections on the input
///   will be silently skipped (same gap the non-streaming CLI
///   converter had before Phase 4).
/// - **CSC-on-disk and dense X** are rejected by
///   `open_x_streaming` with [`ConvertError::StreamingUnsupported`].
pub fn h5ad_to_scx_streaming(
    input: &Path,
    output: &Path,
    opts: &ConvertOptions,
) -> Result<(), ConvertError> {
    if opts.csc {
        return Err(ConvertError::StreamingUnsupported(
            "csc sidecar via streaming is not yet implemented; \
             run `scx build-csc` on the streamed output, or use the \
             non-streaming converter"
                .into(),
        ));
    }

    let file = hdf5::File::open(input)?;

    // Format gating. `open_x_streaming` re-checks CSC/dense and the
    // encoding-type attribute; this branch only catches the 10x case
    // (which has no `X` group at all).
    let input_format = detect_input_format(&file)?;
    if matches!(input_format, InputFormat::TenX) {
        return Err(ConvertError::FormatMismatch {
            expected: "h5ad".to_string(),
            got: "10x".to_string(),
        });
    }
    let matrix_format = detect_matrix_format(&file)?;

    // Open the X reader first — it loads the full indptr and surfaces
    // shape via `n_obs` / `n_vars`, both of which the file header
    // needs before any section write.
    let mut x_reader = open_x_streaming(&file, "X", matrix_format)?;
    let n_obs = x_reader.n_obs;
    let n_vars = x_reader.n_vars;
    let n_vars_u32: u32 = u32::try_from(n_vars)
        .map_err(|_| ConvertError::Other(format!("n_vars {n_vars} exceeds u32::MAX")))?;
    let index_dtype: u8 = if n_vars <= 65535 { 0 } else { 1 };

    // Placeholder header. `nnz`, `n_csr_shards`, `n_csc_shards`, and
    // `codec_id` are overwritten by `ScxWriter::finish()` from
    // running accumulators (see scx-format/src/writer.rs).
    let header = FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs: n_obs as u64,
        n_vars: n_vars as u64,
        nnz: 0,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: opts.shard_target_rows,
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

    let mut writer = ScxWriter::new(output, header)?;

    // obs / var.
    let obs = read_dataframe_group(&file, "obs")?;
    let var = read_dataframe_group(&file, "var")?;
    writer.write_obs(&obs)?;
    writer.write_var(&var)?;

    // X shards (streaming).
    let target_rows = opts.shard_target_rows as usize;
    let mut shard_idx: u32 = 0;
    while let Some(slice_result) = x_reader.next_shard(target_rows) {
        let mut slice = slice_result?;
        sort_csr_rows_in_place(&slice.indptr, &mut slice.indices, &mut slice.values);
        drop_explicit_zeros_inplace(&mut slice.indptr, &mut slice.indices, &mut slice.values);
        let pre = encode_one_shard(
            &slice.indptr,
            &slice.indices,
            &slice.values,
            opts.codec,
            index_dtype,
            n_vars_u32,
            slice.row_start as u64,
            SectionType::CsrShard,
            ModalityType::Rna,
            format!("x_shard_{shard_idx}"),
        )?;
        writer.write_preencoded_shard(pre)?;
        shard_idx += 1;
    }
    drop(x_reader);

    // obsm / varm / uns. Small dense sections — non-streaming reads.
    if let Ok(obsm_map) = read_obsm(&file) {
        for (name, batch) in &obsm_map {
            writer.write_obsm(name, batch)?;
        }
    }
    if let Ok(varm_map) = read_varm(&file) {
        for (name, batch) in &varm_map {
            writer.write_varm(name, batch)?;
        }
    }
    if let Ok(uns) = read_uns(&file) {
        writer.write_uns(&uns)?;
    }

    // Layers (one streaming pass per layer).
    if let Ok(layers_group) = file.group("layers") {
        let layer_names = layers_group.member_names()?;
        for layer_name in &layer_names {
            let mut layer_reader = open_layer_streaming(&file, layer_name)?;
            let mut layer_shard_idx: u32 = 0;
            while let Some(slice_result) = layer_reader.next_shard(target_rows) {
                let mut slice = slice_result?;
                sort_csr_rows_in_place(&slice.indptr, &mut slice.indices, &mut slice.values);
                drop_explicit_zeros_inplace(
                    &mut slice.indptr,
                    &mut slice.indices,
                    &mut slice.values,
                );
                let pre = encode_one_shard(
                    &slice.indptr,
                    &slice.indices,
                    &slice.values,
                    opts.codec,
                    index_dtype,
                    n_vars_u32,
                    slice.row_start as u64,
                    SectionType::LayerCsrShard,
                    ModalityType::Rna,
                    format!("{layer_name}_shard_{layer_shard_idx}"),
                )?;
                writer.write_preencoded_shard(pre)?;
                layer_shard_idx += 1;
            }
        }
    }

    // Provenance carries the streaming flag so consumers can tell at
    // a glance how the file was produced.
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    writer.write_provenance(vec![ProvenanceEntry {
        timestamp,
        action: "convert".to_string(),
        tool: "scx-cli".to_string(),
        params_json: format!(
            "{{\"input\":\"{}\",\"format\":\"h5ad\",\"stream\":true}}",
            input.display()
        ),
        input_checksums: vec![],
    }])?;

    writer.finish()?;
    Ok(())
}

/// Memory budget for the streaming CSR→CSC transpose at convert time.
///
/// 4 GiB matches the `scx build-csc` default. The convert pipeline
/// already holds the full CSR matrix in RAM, so this only bounds
/// the per-chunk transpose working set. Large enough for typical
/// inputs; the user-facing knob is `csc_cols_per_shard`.
const CONVERT_CSC_MEMORY_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Streaming CSR → CSC transpose over the in-memory matrix, with
/// the result written shard-by-shard via `writer.write_csc_shard`.
///
/// Each emitted shard's column count is bounded by
/// `csc_cols_per_shard` (or the memory budget, whichever is
/// smaller). The CSR data already lives in `(indptr, indices, data)`
/// at this point in the pipeline — passed straight to the streaming
/// iterator without re-reading from disk.
#[allow(clippy::too_many_arguments)]
fn write_csc_shards_from_csr(
    writer: &mut ScxWriter,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    value_encoding: ValueEncoding,
    codec_id: CodecId,
    csc_cols_per_shard: usize,
) -> Result<(), ConvertError> {
    // Wrap the in-memory CSR as a single ScxCsr "shard" for the
    // transpose iterator. Use the unchecked constructor — these
    // arrays were just produced by validated readers, no need to
    // re-validate.
    let csr = scx_sparse::ScxCsr::new_unchecked(
        (n_obs, n_vars),
        indptr.to_vec(),
        indices.to_vec(),
        data.to_vec(),
    );
    let shards = std::slice::from_ref(&csr);

    let mut iter = scx_sparse::streaming_csr_to_csc_iter_with_cap(
        shards,
        n_obs,
        n_vars,
        CONVERT_CSC_MEMORY_BYTES,
        csc_cols_per_shard,
    )
    .map_err(|e| ConvertError::Other(format!("CSC transpose failed: {e}")))?;

    loop {
        let col_start = iter.current_col_start() as u64;
        let chunk = match iter.next() {
            Some(c) => c.map_err(|e| ConvertError::Other(format!("CSC chunk failed: {e}")))?,
            None => break,
        };

        let csc_indptr_u64: Vec<u64> = chunk.indptr.iter().map(|&v| v as u64).collect();
        let csc_indices_u32: Vec<u32> = chunk.indices.iter().map(|&i| i as u32).collect();
        let raw_values =
            values_to_raw_bytes(&chunk.data, value_encoding).map_err(ScxError::from)?;

        writer.write_csc_shard(
            &csc_indptr_u64,
            &csc_indices_u32,
            &raw_values,
            codec_id,
            value_encoding,
            col_start,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_csr_shards(
    writer: &mut ScxWriter,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    _n_vars: usize,
    shard_target_rows: usize,
    value_encoding: ValueEncoding,
    codec_id: CodecId,
    index_dtype: u8,
) -> Result<(), ConvertError> {
    let _ = index_dtype; // index dtype is set in the file header; writer reads it from there

    let mut row_start: usize = 0;
    while row_start < n_obs {
        let row_end = (row_start + shard_target_rows).min(n_obs);

        // Slice indptr for this shard
        let shard_indptr_slice = &indptr[row_start..=row_end];
        let base = shard_indptr_slice[0];
        // Validate indptr values are non-negative and >= base (finding 8.6).
        let shard_indptr: Vec<u64> = shard_indptr_slice
            .iter()
            .map(|&v| {
                if v < base {
                    Err(ConvertError::Other(format!(
                        "indptr value {v} less than base {base}"
                    )))
                } else {
                    Ok((v - base) as u64)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Slice indices and data
        let nnz_start = usize::try_from(base)
            .map_err(|_| ConvertError::Other(format!("negative indptr base {base}")))?;
        let nnz_end = usize::try_from(*shard_indptr_slice.last().unwrap()).map_err(|_| {
            ConvertError::Other(format!(
                "negative indptr value {}",
                shard_indptr_slice.last().unwrap()
            ))
        })?;
        // Validate indices are non-negative before casting to u32 (finding 8.5).
        let shard_indices: Vec<u32> = indices[nnz_start..nnz_end]
            .iter()
            .map(|&v| {
                u32::try_from(v)
                    .map_err(|_| ConvertError::Other(format!("negative column index {v}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shard_data = &data[nnz_start..nnz_end];
        let raw_values = values_to_raw_bytes(shard_data, value_encoding).map_err(ScxError::from)?;

        writer.write_csr_shard(
            &shard_indptr,
            &shard_indices,
            &raw_values,
            codec_id,
            value_encoding,
            row_start as u64,
        )?;

        row_start = row_end;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_layer_shards(
    writer: &mut ScxWriter,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    _n_vars: usize,
    shard_target_rows: usize,
    value_encoding: ValueEncoding,
    codec_id: CodecId,
    index_dtype: u8,
    layer_name: &str,
) -> Result<(), ConvertError> {
    let _ = index_dtype;
    let mut row_start: usize = 0;
    let mut shard_idx: u32 = 0;
    while row_start < n_obs {
        let row_end = (row_start + shard_target_rows).min(n_obs);

        let shard_indptr_slice = &indptr[row_start..=row_end];
        let base = shard_indptr_slice[0];
        // Validate indptr values are non-negative and >= base (finding 8.6).
        let shard_indptr: Vec<u64> = shard_indptr_slice
            .iter()
            .map(|&v| {
                if v < base {
                    Err(ConvertError::Other(format!(
                        "indptr value {v} less than base {base}"
                    )))
                } else {
                    Ok((v - base) as u64)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let nnz_start = usize::try_from(base)
            .map_err(|_| ConvertError::Other(format!("negative indptr base {base}")))?;
        let nnz_end = usize::try_from(*shard_indptr_slice.last().unwrap()).map_err(|_| {
            ConvertError::Other(format!(
                "negative indptr value {}",
                shard_indptr_slice.last().unwrap()
            ))
        })?;
        // Validate indices are non-negative before casting to u32 (finding 8.5).
        let shard_indices: Vec<u32> = indices[nnz_start..nnz_end]
            .iter()
            .map(|&v| {
                u32::try_from(v)
                    .map_err(|_| ConvertError::Other(format!("negative column index {v}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shard_data = &data[nnz_start..nnz_end];
        let raw_values = values_to_raw_bytes(shard_data, value_encoding).map_err(ScxError::from)?;

        writer.write_layer_csr_shard(
            &shard_indptr,
            &shard_indices,
            &raw_values,
            codec_id,
            value_encoding,
            row_start as u64,
            layer_name,
            shard_idx,
        )?;

        row_start = row_end;
        shard_idx += 1;
    }
    Ok(())
}
