//! Convert MTX directories to SCX files.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use scx_codec::{CodecId, ValueEncoding};
use scx_format::header::{FileHeader, MAGIC};
use scx_format::provenance::ProvenanceEntry;
use scx_format::select_codec;
use scx_format::writer::ScxWriter;

use crate::error::MtxError;

/// Convert an MTX directory to an SCX file.
///
/// Reads the MTX directory (matrix.mtx[.gz], barcodes.tsv[.gz], features.tsv[.gz])
/// and writes an SCX file with the specified codec and shard size.
///
/// `tool_name` is recorded in provenance (e.g. `"pyscx"` or `"scx-cli"`).
pub fn mtx_to_scx(
    input_dir: &Path,
    output: &Path,
    shard_target_rows: u32,
    codec_str: &str,
    tool_name: &str,
) -> Result<(), MtxError> {
    let mtx_data = crate::read_mtx_directory(input_dir)?;

    let explicit_codec = parse_codec_str(codec_str)?;
    let nnz = *mtx_data.indptr.last().unwrap_or(&0) as u64;
    let (value_encoding, codec_id) = detect_value_encoding(&mtx_data.data, explicit_codec);
    let index_dtype: u8 = if mtx_data.n_vars <= 65535 { 0 } else { 1 };

    let header = FileHeader {
        magic: MAGIC,
        format_version: 1,
        header_length: 256,
        flags: 0,
        n_obs: mtx_data.n_obs as u64,
        n_vars: mtx_data.n_vars as u64,
        nnz,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows,
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
        reserved: [0u8; 132],
    };

    let mut writer = ScxWriter::new(output, header)?;

    writer.write_obs(&mtx_data.obs)?;
    writer.write_var(&mtx_data.var)?;

    write_csr_shards(
        &mut writer,
        &mtx_data.indptr,
        &mtx_data.indices,
        &mtx_data.data,
        mtx_data.n_obs,
        shard_target_rows as usize,
        value_encoding,
        codec_id,
    )?;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let params_json = serde_json::json!({
        "input": input_dir.display().to_string(),
        "format": "mtx",
    })
    .to_string();
    writer.write_provenance(vec![ProvenanceEntry {
        timestamp,
        action: "convert".to_string(),
        tool: tool_name.to_string(),
        params_json,
        input_checksums: vec![],
    }])?;

    writer.finish()?;
    Ok(())
}

fn parse_codec_str(s: &str) -> Result<Option<CodecId>, MtxError> {
    match s {
        "auto" => Ok(None),
        "none" => Ok(Some(CodecId::None)),
        "scx1" => Ok(Some(CodecId::Scx1)),
        "zstd" => Ok(Some(CodecId::Zstd)),
        other => Err(MtxError::InvalidCodec(format!(
            "Unknown codec: '{}'. Use auto, none, scx1, or zstd.",
            other
        ))),
    }
}

fn detect_value_encoding(
    data: &[f32],
    explicit_codec: Option<CodecId>,
) -> (ValueEncoding, CodecId) {
    let is_integer = data
        .iter()
        .all(|&v| v.is_finite() && v >= 0.0 && v == v.floor());

    let encoding = if !is_integer {
        ValueEncoding::Float32
    } else {
        let max_val: f64 = data.iter().map(|&v| v as f64).fold(0.0f64, f64::max);
        if max_val <= 255.0 {
            ValueEncoding::Uint8
        } else if max_val <= 65535.0 {
            ValueEncoding::Uint16
        } else {
            ValueEncoding::Uint32
        }
    };

    let raw_bytes = values_to_raw_bytes(data, encoding);
    let codec = match explicit_codec {
        Some(codec_id) => {
            if codec_id == CodecId::Scx1 && !encoding.is_integer() {
                CodecId::Zstd
            } else {
                codec_id
            }
        }
        None => select_codec(&raw_bytes, encoding),
    };

    (encoding, codec)
}

fn values_to_raw_bytes(data: &[f32], encoding: ValueEncoding) -> Vec<u8> {
    match encoding {
        ValueEncoding::Uint8 => data.iter().map(|&v| v as u8).collect(),
        ValueEncoding::Uint16 => {
            let mut buf = Vec::with_capacity(data.len() * 2);
            for &v in data {
                buf.extend_from_slice(&(v as u16).to_le_bytes());
            }
            buf
        }
        ValueEncoding::Uint32 => {
            let mut buf = Vec::with_capacity(data.len() * 4);
            for &v in data {
                buf.extend_from_slice(&(v as u32).to_le_bytes());
            }
            buf
        }
        ValueEncoding::Float32 | ValueEncoding::Float16 => {
            let mut buf = Vec::with_capacity(data.len() * 4);
            for &v in data {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            buf
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_csr_shards(
    writer: &mut ScxWriter,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    shard_target_rows: usize,
    value_encoding: ValueEncoding,
    codec_id: CodecId,
) -> Result<(), MtxError> {
    let mut row_start: usize = 0;
    while row_start < n_obs {
        let row_end = (row_start + shard_target_rows).min(n_obs);

        let shard_indptr_slice = &indptr[row_start..=row_end];
        let base = shard_indptr_slice[0];
        let shard_indptr: Vec<u64> = shard_indptr_slice
            .iter()
            .map(|&v| (v - base) as u64)
            .collect();

        let nnz_start = base as usize;
        let nnz_end = *shard_indptr_slice.last().unwrap() as usize;
        let shard_indices: Vec<u32> = indices[nnz_start..nnz_end]
            .iter()
            .map(|&v| v as u32)
            .collect();
        let shard_data = &data[nnz_start..nnz_end];
        let raw_values = values_to_raw_bytes(shard_data, value_encoding);

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
