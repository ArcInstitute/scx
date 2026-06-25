//! Convert MTX directories to SCX files.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::error::ScxError;
use scx_format_io::header::FileHeader;
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::writer::ScxWriter;
use scx_format_io::{select_codec_for_modality, ModalityType};

use crate::error::MtxError;
use crate::read::MtxOrientation;

/// Convert an MTX directory to an SCX file.
///
/// Reads the MTX directory (matrix.mtx[.gz], barcodes.tsv[.gz], features.tsv[.gz])
/// and writes an SCX file with the specified codec and shard size.
///
/// `tool_name` is recorded in provenance (e.g. `"pyscx"` or `"scx"`).
///
/// Returns the detected on-disk [`MtxOrientation`] so callers (the CLI, pyscx)
/// can surface the ambiguous (square-matrix) case to the user; the orientation
/// is also recorded in provenance regardless of the caller.
pub fn mtx_to_scx(
    input_dir: &Path,
    output: &Path,
    shard_target_rows: u32,
    codec_str: &str,
    tool_name: &str,
) -> Result<MtxOrientation, MtxError> {
    let mtx_data = crate::read_mtx_directory(input_dir)?;
    let orientation = mtx_data.orientation;

    let explicit_codec = parse_codec_str(codec_str)?;
    let nnz = *mtx_data.indptr.last().unwrap_or(&0) as u64;
    let (value_encoding, codec_id) = detect_value_encoding(&mtx_data.data, explicit_codec)?;
    let index_dtype: u8 = if mtx_data.n_vars <= 65535 { 0 } else { 1 };

    let header = FileHeader::new_single_modality(
        mtx_data.n_obs as u64,
        mtx_data.n_vars as u64,
        nnz,
        shard_target_rows,
        codec_id as u8,
        index_dtype,
    );

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
        "mtx_orientation": orientation.as_str(),
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
    Ok(orientation)
}

fn parse_codec_str(s: &str) -> Result<Option<CodecId>, MtxError> {
    // Delegate to the single-source CLI codec vocabulary in scx-codec so this
    // path accepts the same set as the rest of the CLI (previously omitted
    // lz4/pcodec).
    CodecId::parse_cli(s).map_err(MtxError::InvalidCodec)
}

use scx_codec::value_encoding::{
    detect_value_encoding as detect_value_encoding_only, values_to_raw_bytes,
};

fn detect_value_encoding(
    data: &[f32],
    explicit_codec: Option<CodecId>,
) -> Result<(ValueEncoding, CodecId), MtxError> {
    let encoding = detect_value_encoding_only(data);
    let raw_bytes = values_to_raw_bytes(data, encoding).map_err(ScxError::from)?;
    let codec = match explicit_codec {
        Some(codec_id) => {
            if matches!(codec_id, CodecId::Scx1 | CodecId::Scx2) && !encoding.is_integer() {
                CodecId::Zstd
            } else {
                codec_id
            }
        }
        None => select_codec_for_modality(&raw_bytes, encoding, ModalityType::Rna),
    };

    Ok((encoding, codec))
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
