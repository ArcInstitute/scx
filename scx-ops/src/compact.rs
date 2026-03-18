// Compact operation: rewrite file without deleted rows or stale catalogs.

use std::path::Path;

use arrow::compute;
use scx_codec::{CodecId, ValueEncoding};
use scx_format::header::{FileHeader, MAGIC};
use scx_format::provenance::ProvenanceEntry;
use scx_format::writer::ScxWriter;
use scx_format::ScxReader;

use crate::error::Result;

/// Compact an SCX file: removes deleted rows, stale catalogs, and produces
/// a clean single-catalog file.
pub fn compact(input_path: &Path, output_path: &Path) -> Result<()> {
    let reader = ScxReader::open(input_path)?;
    let in_header = reader.header();

    // Load deletion vectors
    let dv = reader.read_deletion_vectors()?;

    // Read obs and build deletion mask
    let obs = reader.read_obs()?;
    let var = reader.read_var()?;

    let n_obs = in_header.n_obs as usize;
    let n_vars = in_header.n_vars;

    // Build a boolean array: true = keep, false = deleted
    let keep_mask = build_keep_mask(n_obs, &dv, reader.catalog());

    // Filter obs
    let filtered_obs = if let Some(ref mask) = keep_mask {
        let bool_array = arrow::array::BooleanArray::from(mask.clone());
        compute::filter_record_batch(&obs, &bool_array)?
    } else {
        obs
    };

    let new_n_obs = filtered_obs.num_rows();

    // Set up output header
    let out_header = FileHeader {
        magic: MAGIC,
        format_version: 1,
        header_length: 256,
        flags: 0, // clean file, no DV
        n_obs: new_n_obs as u64,
        n_vars,
        nnz: 0, // will be set by finish
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: in_header.shard_target_rows,
        codec_id: in_header.codec_id,
        index_dtype: in_header.index_dtype,
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

    // Copy obsm flag if present
    let has_obsm = in_header.has_obsm();

    let mut writer = ScxWriter::new(output_path, out_header)?;
    writer.write_obs(&filtered_obs)?;
    writer.write_var(&var)?;

    // Process CSR shards: decode, filter deleted rows, re-shard
    let shards = reader.catalog().shards_sorted();
    let codec_id = CodecId::from_u8(in_header.codec_id).unwrap_or(CodecId::None);
    let value_encoding_u8 = {
        // Read from first shard header to get value encoding
        if !shards.is_empty() {
            let section = reader.section_bytes(shards[0]);
            let sh = scx_format::ShardHeader::read_from(&mut std::io::Cursor::new(
                &section[..scx_format::SHARD_HEADER_SIZE],
            ))?;
            sh.value_encoding
        } else {
            0 // default to uint8
        }
    };
    let value_encoding = ValueEncoding::from_u8(value_encoding_u8).unwrap_or(ValueEncoding::Uint8);

    // Decode all shards and filter rows
    let mut global_row = 0u64;
    let shard_target = in_header.shard_target_rows;

    // Accumulate filtered CSR data
    let mut acc_indptr: Vec<u64> = vec![0];
    let mut acc_indices: Vec<u32> = Vec::new();
    let mut acc_values: Vec<u8> = Vec::new();
    let mut acc_row_count = 0u64;

    for shard_entry in &shards {
        let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;
        let shard_row_start = shard_entry
            .stats
            .as_ref()
            .map(|s| s.row_start)
            .unwrap_or(global_row);
        let shard_n_rows = indptr.len() - 1;

        for local_row in 0..shard_n_rows {
            let global_idx = shard_row_start + local_row as u64;

            // Check if deleted
            let deleted = keep_mask
                .as_ref()
                .is_some_and(|mask| !mask[global_idx as usize]);

            if deleted {
                continue;
            }

            let row_start_nnz = indptr[local_row] as usize;
            let row_end_nnz = indptr[local_row + 1] as usize;
            let row_nnz = row_end_nnz - row_start_nnz;

            for j in row_start_nnz..row_end_nnz {
                acc_indices.push(indices[j] as u32);
                // Convert f32 back to raw bytes per value_encoding
                encode_value(&mut acc_values, data[j], value_encoding);
            }

            let prev = *acc_indptr.last().unwrap();
            acc_indptr.push(prev + row_nnz as u64);
            acc_row_count += 1;

            // Flush accumulated rows as a shard when hitting target
            if acc_row_count >= shard_target as u64 {
                writer.write_csr_shard(
                    &acc_indptr,
                    &acc_indices,
                    &acc_values,
                    codec_id,
                    value_encoding,
                    global_row - acc_row_count + 1, // approximate row_start
                )?;
                acc_indptr = vec![0];
                acc_indices.clear();
                acc_values.clear();
                acc_row_count = 0;
            }
        }

        global_row = shard_row_start + shard_n_rows as u64;
    }

    // Flush remaining accumulated rows
    if acc_row_count > 0 {
        // Compute actual row_start for this final shard
        let row_start = new_n_obs as u64 - acc_row_count;
        writer.write_csr_shard(
            &acc_indptr,
            &acc_indices,
            &acc_values,
            codec_id,
            value_encoding,
            row_start,
        )?;
    }

    // Copy obsm (row-filtered)
    if has_obsm {
        let all_obsm = reader.read_all_obsm()?;
        for (name, batch) in &all_obsm {
            let filtered_batch = if let Some(ref mask) = keep_mask {
                let bool_array = arrow::array::BooleanArray::from(mask.clone());
                compute::filter_record_batch(batch, &bool_array)?
            } else {
                batch.clone()
            };
            writer.write_obsm(name, &filtered_batch)?;
        }
    }

    // Copy uns
    if let Ok(uns) = reader.read_uns() {
        writer.write_uns(&uns)?;
    }

    // Copy layers (row-filtered)
    let layer_names = reader.layer_names();
    for layer_name in &layer_names {
        let layer = reader.read_layer(layer_name)?;
        // Filter and write layer shards
        let mut layer_indptr: Vec<u64> = vec![0];
        let mut layer_indices: Vec<u32> = Vec::new();
        let mut layer_values: Vec<u8> = Vec::new();
        let mut layer_row_count = 0u64;
        let mut layer_shard_idx = 0u32;

        for row_idx in 0..layer.shape.0 {
            let deleted = keep_mask.as_ref().is_some_and(|mask| !mask[row_idx]);
            if deleted {
                continue;
            }

            let row_start = layer.indptr[row_idx] as usize;
            let row_end = layer.indptr[row_idx + 1] as usize;
            for j in row_start..row_end {
                layer_indices.push(layer.indices[j] as u32);
                encode_value(&mut layer_values, layer.data[j], value_encoding);
            }
            let prev = *layer_indptr.last().unwrap();
            layer_indptr.push(prev + (row_end - row_start) as u64);
            layer_row_count += 1;

            if layer_row_count >= shard_target as u64 {
                writer.write_layer_csr_shard(
                    &layer_indptr,
                    &layer_indices,
                    &layer_values,
                    codec_id,
                    value_encoding,
                    new_n_obs as u64 - layer_row_count,
                    layer_name,
                    layer_shard_idx,
                )?;
                layer_indptr = vec![0];
                layer_indices.clear();
                layer_values.clear();
                layer_row_count = 0;
                layer_shard_idx += 1;
            }
        }

        if layer_row_count > 0 {
            writer.write_layer_csr_shard(
                &layer_indptr,
                &layer_indices,
                &layer_values,
                codec_id,
                value_encoding,
                new_n_obs as u64 - layer_row_count,
                layer_name,
                layer_shard_idx,
            )?;
        }
    }

    // Add provenance
    let mut prov_entries = if let Ok(prov) = reader.read_provenance() {
        prov.operations
    } else {
        Vec::new()
    };
    prov_entries.push(ProvenanceEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        action: "compact".to_string(),
        tool: "scx-ops 0.1.0".to_string(),
        params_json: "{}".to_string(),
        input_checksums: vec![],
    });
    writer.write_provenance(prov_entries)?;

    writer.finish()?;
    Ok(())
}

/// Build a keep mask based on deletion vectors.
fn build_keep_mask(
    n_obs: usize,
    dv: &Option<scx_format::DeletionVectors>,
    catalog: &scx_format::FullCatalog,
) -> Option<Vec<bool>> {
    let dv = dv.as_ref()?;
    if dv.total_deleted() == 0 {
        return None;
    }

    let mut mask = vec![true; n_obs];

    // Get shard ranges
    let shards = catalog.shards_sorted();
    for shard_entry in &shards {
        if let Some(ref stats) = shard_entry.stats {
            let shard_idx = shards
                .iter()
                .position(|s| s.offset == shard_entry.offset)
                .unwrap_or(0) as u32;

            if let Some(sd) = dv.shards.iter().find(|sd| sd.shard_id == shard_idx) {
                for local_row in sd.bitmap.iter() {
                    let global_row = stats.row_start + local_row as u64;
                    if (global_row as usize) < n_obs {
                        mask[global_row as usize] = false;
                    }
                }
            }
        }
    }

    Some(mask)
}

/// Encode a single f32 value back to raw bytes according to the value encoding.
fn encode_value(buf: &mut Vec<u8>, value: f32, encoding: ValueEncoding) {
    match encoding {
        ValueEncoding::Uint8 => buf.push(value as u8),
        ValueEncoding::Uint16 => buf.extend_from_slice(&(value as u16).to_le_bytes()),
        ValueEncoding::Uint32 => buf.extend_from_slice(&(value as u32).to_le_bytes()),
        ValueEncoding::Float32 => buf.extend_from_slice(&value.to_le_bytes()),
        ValueEncoding::Float16 => {
            // Simplified: store as u16 bits
            buf.extend_from_slice(&(value as u16).to_le_bytes());
        }
    }
}
