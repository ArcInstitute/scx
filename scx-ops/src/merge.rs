// Merge operation: combine multiple SCX files into one.

use std::path::Path;

use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use scx_codec::{CodecId, ValueEncoding};
use scx_format::header::{FileHeader, MAGIC};
use scx_format::provenance::ProvenanceEntry;
use scx_format::section::SectionType;
use scx_format::writer::ScxWriter;
use scx_format::{ScxReader, ShardHeader, SHARD_HEADER_SIZE};

use crate::error::{OpsError, Result};

/// Merge multiple SCX files into a single output file.
/// All inputs must have the same n_vars.
pub fn merge(input_paths: &[&Path], output_path: &Path) -> Result<()> {
    if input_paths.is_empty() {
        return Err(OpsError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "no input files provided",
        )));
    }

    // Open all inputs
    let readers: Vec<ScxReader> = input_paths
        .iter()
        .map(|p| ScxReader::open(p).map_err(OpsError::Format))
        .collect::<Result<Vec<_>>>()?;

    // Validate n_vars consistency
    let n_vars = readers[0].n_vars();
    for (_i, reader) in readers.iter().enumerate().skip(1) {
        if reader.n_vars() != n_vars {
            return Err(OpsError::IncompatibleVars {
                expected: n_vars,
                found: reader.n_vars(),
            });
        }
    }

    let total_n_obs: u64 = readers.iter().map(|r| r.n_obs()).sum();

    // Get codec and encoding info from first file's first shard
    let first_header = readers[0].header();
    let codec_id = CodecId::from_u8(first_header.codec_id).unwrap_or(CodecId::None);
    let value_encoding_u8 = {
        let shards = readers[0].catalog().shards_sorted();
        if !shards.is_empty() {
            let section = readers[0].section_bytes(shards[0]);
            let sh = scx_format::ShardHeader::read_from(&mut std::io::Cursor::new(
                &section[..scx_format::SHARD_HEADER_SIZE],
            ))?;
            sh.value_encoding
        } else {
            0
        }
    };
    let value_encoding = ValueEncoding::from_u8(value_encoding_u8).unwrap_or(ValueEncoding::Uint8);

    // Build output header
    let out_header = FileHeader {
        magic: MAGIC,
        format_version: 1,
        header_length: 256,
        flags: 0,
        n_obs: total_n_obs,
        n_vars,
        nnz: 0,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: first_header.shard_target_rows,
        codec_id: first_header.codec_id,
        index_dtype: first_header.index_dtype,
        endian: 0,
        reserved_padding: 0,
        root_catalog_offset: 0,
        root_catalog_length: 0,
        full_catalog_offset: 0,
        full_catalog_length: 0,
        manifest_sequence: 0,
        prev_catalog_offset: 0,
        file_checksum: 0,
        front_catalog_offset: 0,
        front_catalog_length: 0,
        reserved: [0u8; 132],
    };

    let mut writer = ScxWriter::new(output_path, out_header)?;

    // Concatenate obs across all inputs
    let obs_batches: Vec<RecordBatch> = readers
        .iter()
        .map(|r| r.read_obs())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let merged_obs = concat_batches(&obs_batches[0].schema(), &obs_batches)?;
    writer.write_obs(&merged_obs)?;

    // Write var from first input
    let var = readers[0].read_var()?;
    writer.write_var(&var)?;

    // For each input: decode CSR shards and write to output with adjusted row_start
    let mut cumulative_rows = 0u64;
    for reader in &readers {
        let shards = reader.catalog().shards_sorted();
        for shard_entry in &shards {
            let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;
            let n_rows = indptr.len() - 1;

            // Convert back to on-disk format
            let indptr_u64: Vec<u64> = indptr.iter().map(|&v| v as u64).collect();
            let indices_u32: Vec<u32> = indices.iter().map(|&v| v as u32).collect();
            let mut values_bytes = Vec::new();
            for &v in &data {
                encode_value(&mut values_bytes, v, value_encoding);
            }

            writer.write_csr_shard(
                &indptr_u64,
                &indices_u32,
                &values_bytes,
                codec_id,
                value_encoding,
                cumulative_rows,
            )?;
            cumulative_rows += n_rows as u64;
        }
    }

    // Merge obsm
    let first_obsm = readers[0].read_all_obsm()?;
    for name in first_obsm.keys() {
        let mut batches = Vec::new();
        for reader in &readers {
            if let Ok(batch) = reader.read_obsm(name) {
                batches.push(batch);
            }
        }
        if batches.len() == readers.len() {
            let merged = concat_batches(&batches[0].schema(), &batches)?;
            writer.write_obsm(name, &merged)?;
        }
    }

    // Merge uns from first input
    if let Ok(uns) = readers[0].read_uns() {
        writer.write_uns(&uns)?;
    }

    // Merge layers — read per-layer value encoding from shard headers
    let shard_target = first_header.shard_target_rows;
    let layer_names = readers[0].layer_names();
    for layer_name in &layer_names {
        // Determine this layer's value encoding from its first shard header
        let layer_prefix = format!("{layer_name}_shard_");
        let layer_shard_entries: Vec<&scx_format::catalog::FullCatalogEntry> = readers[0]
            .catalog()
            .entries
            .iter()
            .filter(|e| {
                e.section_type == SectionType::LayerCsrShard && e.name.starts_with(&layer_prefix)
            })
            .collect();
        let layer_value_encoding = if let Some(first_entry) = layer_shard_entries.first() {
            let section = readers[0].section_bytes(first_entry);
            let sh =
                ShardHeader::read_from(&mut std::io::Cursor::new(&section[..SHARD_HEADER_SIZE]))?;
            ValueEncoding::from_u8(sh.value_encoding).unwrap_or(ValueEncoding::Uint8)
        } else {
            value_encoding
        };
        let layer_codec = if let Some(first_entry) = layer_shard_entries.first() {
            let section = readers[0].section_bytes(first_entry);
            let sh =
                ShardHeader::read_from(&mut std::io::Cursor::new(&section[..SHARD_HEADER_SIZE]))?;
            CodecId::from_u8(sh.codec_id).unwrap_or(CodecId::None)
        } else {
            codec_id
        };

        // Accumulate and flush at shard_target_rows, matching compact.rs pattern
        let mut layer_indptr: Vec<u64> = vec![0];
        let mut layer_indices: Vec<u32> = Vec::new();
        let mut layer_values: Vec<u8> = Vec::new();
        let mut layer_row_count = 0u64;
        let mut layer_shard_idx = 0u32;
        let mut emitted_layer_rows = 0u64;

        for reader in &readers {
            if let Ok(layer) = reader.read_layer(layer_name) {
                for row_idx in 0..layer.shape.0 {
                    let row_start = layer.indptr[row_idx] as usize;
                    let row_end = layer.indptr[row_idx + 1] as usize;
                    for j in row_start..row_end {
                        layer_indices.push(layer.indices[j] as u32);
                        encode_value(&mut layer_values, layer.data[j], layer_value_encoding);
                    }
                    let prev = *layer_indptr.last().unwrap();
                    layer_indptr.push(prev + (row_end - row_start) as u64);
                    layer_row_count += 1;

                    if layer_row_count >= shard_target as u64 {
                        writer.write_layer_csr_shard(
                            &layer_indptr,
                            &layer_indices,
                            &layer_values,
                            layer_codec,
                            layer_value_encoding,
                            emitted_layer_rows,
                            layer_name,
                            layer_shard_idx,
                        )?;
                        emitted_layer_rows += layer_row_count;
                        layer_indptr = vec![0];
                        layer_indices.clear();
                        layer_values.clear();
                        layer_row_count = 0;
                        layer_shard_idx += 1;
                    }
                }
            }
        }

        // Flush remaining layer rows
        if layer_row_count > 0 {
            writer.write_layer_csr_shard(
                &layer_indptr,
                &layer_indices,
                &layer_values,
                layer_codec,
                layer_value_encoding,
                emitted_layer_rows,
                layer_name,
                layer_shard_idx,
            )?;
        }
    }

    // Merge provenance
    let mut all_prov_entries = Vec::new();
    for reader in &readers {
        if let Ok(prov) = reader.read_provenance() {
            all_prov_entries.extend(prov.operations);
        }
    }
    all_prov_entries.push(ProvenanceEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        action: "merge".to_string(),
        tool: "scx-ops 0.1.0".to_string(),
        params_json: format!("{{\"n_inputs\":{}}}", input_paths.len()),
        input_checksums: readers
            .iter()
            .map(|r| {
                let mut cs = [0u8; 32];
                cs[..8].copy_from_slice(&r.header().file_checksum.to_le_bytes());
                cs
            })
            .collect(),
    });
    writer.write_provenance(all_prov_entries)?;

    writer.finish()?;
    Ok(())
}

fn encode_value(buf: &mut Vec<u8>, value: f32, encoding: ValueEncoding) {
    match encoding {
        ValueEncoding::Uint8 => buf.push(value as u8),
        ValueEncoding::Uint16 => buf.extend_from_slice(&(value as u16).to_le_bytes()),
        ValueEncoding::Uint32 => buf.extend_from_slice(&(value as u32).to_le_bytes()),
        ValueEncoding::Float32 => buf.extend_from_slice(&value.to_le_bytes()),
        ValueEncoding::Float16 => {
            let f16_val = half::f16::from_f32(value);
            buf.extend_from_slice(&f16_val.to_le_bytes());
        }
    }
}
