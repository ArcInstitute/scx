// Merge operation: combine multiple SCX files into one.

use std::path::Path;

use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use scx_codec::{CodecId, ValueEncoding};
use scx_format::codec_select::select_codec;
use scx_format::header::{FileHeader, MAGIC};
use scx_format::provenance::ProvenanceEntry;
use scx_format::section::SectionType;
use scx_format::writer::ScxWriter;
use scx_format::{ScxReader, ShardHeader, SHARD_HEADER_SIZE};

use crate::append::unify_dict_columns;
use crate::error::{OpsError, Result};
use crate::flock::SharedFileLock;
use crate::helpers::encode_value;

/// Merge multiple SCX files into a single output file.
/// All inputs must have the same n_vars.
pub fn merge(input_paths: &[&Path], output_path: &Path) -> Result<()> {
    if input_paths.is_empty() {
        return Err(OpsError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "no input files provided",
        )));
    }

    // Acquire shared locks on all inputs to prevent concurrent writers
    // from modifying files while we read them. Locks held until `_locks` dropped.
    let _locks: Vec<SharedFileLock> = input_paths
        .iter()
        .map(|p| SharedFileLock::acquire(p))
        .collect::<Result<Vec<_>>>()?;

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

    // Phase 6: validate modality structure consistency. Multimodal
    // merge requires every input to expose the same set of
    // modalities (name + type + n_vars). On mismatch, raise with a
    // clear error directing to extract-then-merge.
    let any_multimodal = readers.iter().any(|r| r.is_multimodal());
    if any_multimodal {
        let first = readers[0].modality_table();
        for (i, reader) in readers.iter().enumerate().skip(1) {
            let here = reader.modality_table();
            if !modality_tables_match(first, here) {
                return Err(OpsError::ModalityMismatch {
                    detail: format!(
                        "input 0 and input {i} have different modality structures \
                         (name / modality_type / n_vars must match across all inputs); \
                         use `scx subset --modality NAME` on each input to extract a \
                         single modality first, then `scx merge` the single-modality files"
                    ),
                });
            }
        }
        // Phase 6: dispatch to multimodal merge — concatenate the
        // global obs row axis and per-modality CSR shards in input
        // order, preserving each modality's var.
        return merge_multimodal(&readers, input_paths, output_path);
    }

    let total_n_obs: u64 = readers.iter().map(|r| r.n_obs()).sum();

    let first_header = readers[0].header();

    // CSC sidecars are dropped on merge: row layout is
    // re-concatenated across inputs, so any per-input CSC `indices`
    // arrays would reference stale row indices in the merged output.
    // Caller can opt back in via `--rebuild-csc` on the CLI.
    // Phase H.3.
    let any_input_had_csc = readers.iter().any(|r| r.header().has_csc());
    if any_input_had_csc {
        let inputs_str = input_paths
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        log::warn!(
            "merge dropped CSC shards from at least one input ({inputs_str}): \
             rerun `scx build-csc` (or pass --rebuild-csc) to restore the \
             column-major sidecar on the merged output"
        );
    }

    // Build output header. codec_id is 0 (None) because actual codec is
    // selected per-shard via select_codec(). `flags` stays 0 — merge
    // intentionally produces a clean output header rather than carrying
    // input flag state forward.
    let out_header = FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs: total_n_obs,
        n_vars,
        nnz: 0,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: first_header.shard_target_rows,
        codec_id: 0,
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
        n_modalities: 0,
        modality_table_offset: 0,
        modality_table_length: 0,
        reserved: [0u8; 112],
    };

    let mut writer = ScxWriter::new(output_path, out_header)?;

    // Concatenate obs across all inputs, unifying dictionary-encoded columns
    // to avoid corrupt categoricals when merging files with different dictionaries.
    let obs_batches: Vec<RecordBatch> = readers
        .iter()
        .map(|r| r.read_obs())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let unified_batches: Vec<RecordBatch> = obs_batches
        .iter()
        .map(unify_dict_columns)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let merged_obs = concat_batches(&unified_batches[0].schema(), &unified_batches)?;
    writer.write_obs(&merged_obs)?;

    // Write var from first input
    let var = readers[0].read_var()?;
    writer.write_var(&var)?;

    // For each input: decode CSR shards and write to output with adjusted row_start.
    // Per-shard codec is auto-selected via select_codec() on the re-encoded values,
    // and value_encoding is read from each shard's header for correctness.
    let mut cumulative_rows = 0u64;
    for reader in &readers {
        let shards = reader.catalog().shards_sorted();
        for shard_entry in &shards {
            // Read this shard's value_encoding from its header
            let shard_section = reader.section_bytes(shard_entry)?;
            let sh = ShardHeader::read_from(&mut std::io::Cursor::new(
                &shard_section[..SHARD_HEADER_SIZE],
            ))?;
            let shard_value_encoding = ValueEncoding::from_u8(sh.value_encoding)
                .ok_or(OpsError::UnknownValueEncoding(sh.value_encoding))?;

            let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;
            let n_rows = indptr.len() - 1;

            // Convert back to on-disk format using the shard's own value encoding
            let indptr_u64: Vec<u64> = indptr.iter().map(|&v| v as u64).collect();
            let indices_u32: Vec<u32> = indices.iter().map(|&v| v as u32).collect();
            let mut values_bytes = Vec::new();
            for &v in &data {
                encode_value(&mut values_bytes, v, shard_value_encoding)?;
            }

            // Auto-select optimal codec for this shard's data
            let shard_codec = select_codec(&values_bytes, shard_value_encoding);

            writer.write_csr_shard(
                &indptr_u64,
                &indices_u32,
                &values_bytes,
                shard_codec,
                shard_value_encoding,
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

    // Merge layers — collect layer names from ALL inputs, not just the first,
    // so layers present only in subsequent inputs are not silently dropped.
    let shard_target = first_header.shard_target_rows;
    let layer_names = {
        let mut all_names: Vec<String> = readers.iter().flat_map(|r| r.layer_names()).collect();
        all_names.sort();
        all_names.dedup();
        all_names
    };
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
            let section = readers[0].section_bytes(first_entry)?;
            let sh =
                ShardHeader::read_from(&mut std::io::Cursor::new(&section[..SHARD_HEADER_SIZE]))?;
            ValueEncoding::from_u8(sh.value_encoding)
                .ok_or(OpsError::UnknownValueEncoding(sh.value_encoding))?
        } else {
            ValueEncoding::Uint8
        };

        // Accumulate and flush at shard_target_rows, matching compact.rs pattern
        let mut layer_indptr: Vec<u64> = vec![0];
        let mut layer_indices: Vec<u32> = Vec::new();
        let mut layer_values: Vec<u8> = Vec::new();
        let mut layer_row_count = 0u64;
        let mut layer_shard_idx = 0u32;
        let mut emitted_layer_rows = 0u64;

        for (file_idx, reader) in readers.iter().enumerate() {
            let layer = reader
                .read_layer(layer_name)
                .map_err(|_| OpsError::LayerMissing {
                    name: layer_name.clone(),
                    file_index: file_idx,
                })?;
            for row_idx in 0..layer.shape.0 {
                let row_start = layer.indptr[row_idx] as usize;
                let row_end = layer.indptr[row_idx + 1] as usize;
                for j in row_start..row_end {
                    layer_indices.push(layer.indices[j] as u32);
                    encode_value(&mut layer_values, layer.data[j], layer_value_encoding)?;
                }
                let prev = *layer_indptr.last().unwrap();
                layer_indptr.push(prev + (row_end - row_start) as u64);
                layer_row_count += 1;

                if layer_row_count >= shard_target as u64 {
                    // Auto-select codec per layer shard
                    let layer_shard_codec = select_codec(&layer_values, layer_value_encoding);
                    writer.write_layer_csr_shard(
                        &layer_indptr,
                        &layer_indices,
                        &layer_values,
                        layer_shard_codec,
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

        // Flush remaining layer rows
        if layer_row_count > 0 {
            // Auto-select codec for remaining layer shard
            let layer_shard_codec = select_codec(&layer_values, layer_value_encoding);
            writer.write_layer_csr_shard(
                &layer_indptr,
                &layer_indices,
                &layer_values,
                layer_shard_codec,
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

/// Phase 6: merge multimodal SCX files with matching modality
/// structure. Per-modality CSR shards are concatenated in input order
/// with `row_start` adjusted for the cumulative global obs offset;
/// per-modality var is preserved from the first input (already
/// validated identical by the modality-table check). Per-modality CSC
/// sidecars are dropped (caller can `--rebuild-csc`).
fn merge_multimodal(
    readers: &[ScxReader],
    input_paths: &[&Path],
    output_path: &Path,
) -> Result<()> {
    let table = readers[0]
        .modality_table()
        .ok_or_else(|| {
            scx_format::ScxError::InvalidCatalog(
                "merge_multimodal: input has no modality table".to_string(),
            )
        })?
        .clone();

    let total_n_obs: u64 = readers.iter().map(|r| r.n_obs()).sum();
    let first_header = readers[0].header();

    let any_input_had_csc = readers.iter().any(|r| {
        r.modality_table()
            .map(|t| t.entries.iter().any(|info| info.flags.has_csc()))
            .unwrap_or(false)
    });
    if any_input_had_csc {
        let inputs_str = input_paths
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        log::warn!(
            "merge dropped per-modality CSC shards from at least one input ({inputs_str}): \
             rerun `scx build-csc` (or pass --rebuild-csc) to restore the \
             column-major sidecar on the merged output"
        );
    }

    let max_n_vars = table.entries.iter().map(|i| i.n_vars).max().unwrap_or(0);

    let out_header = FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs: total_n_obs,
        n_vars: max_n_vars,
        nnz: 0,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: first_header.shard_target_rows,
        codec_id: 0,
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
        n_modalities: 0,
        modality_table_offset: 0,
        modality_table_length: 0,
        reserved: [0u8; 112],
    };

    let mut writer = ScxWriter::new(output_path, out_header)?;

    // Global obs: concatenate with dict unification (same path as
    // single-modality merge).
    let obs_batches: Vec<RecordBatch> = readers
        .iter()
        .map(|r| r.read_obs())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let unified_batches: Vec<RecordBatch> = obs_batches
        .iter()
        .map(unify_dict_columns)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let merged_obs = concat_batches(&unified_batches[0].schema(), &unified_batches)?;
    writer.write_obs(&merged_obs)?;

    // Register modalities in input order.
    for info in &table.entries {
        let codec = CodecId::from_u8(info.default_codec_id)
            .ok_or(OpsError::UnknownCodec(info.default_codec_id))?;
        let value_encoding = ValueEncoding::from_u8(info.default_value_encoding)
            .ok_or(OpsError::UnknownValueEncoding(info.default_value_encoding))?;
        writer.add_modality(&info.name, info.modality_type, codec, value_encoding, false)?;
        writer.set_modality_n_vars(writer.n_modalities() as u8, info.n_vars)?;
    }

    // Per-modality var (from first input — already validated identical).
    for (idx, info) in table.entries.iter().enumerate() {
        let modality_id = (idx + 1) as u8;
        let var = readers[0].read_var_for(modality_id)?;
        writer.write_var_for(modality_id, &var)?;
        let _ = info; // suppress unused if no other field needed
    }

    // Per-modality CSR shards: concatenate across inputs with row_start
    // adjusted for the cumulative global obs offset. Within an input,
    // the modality's shards collectively cover the input's n_obs rows;
    // after one input we advance the offset by that input's n_obs so
    // the next input's shards line up against the merged obs.
    for (idx, _info) in table.entries.iter().enumerate() {
        let modality_id = (idx + 1) as u8;
        let mut input_offset: u64 = 0;
        for reader in readers {
            let entries = reader.catalog().csr_shards_for_modality(modality_id);
            for shard_entry in entries {
                let section = reader.section_bytes(shard_entry)?;
                let sh = ShardHeader::read_from(&mut std::io::Cursor::new(
                    &section[..SHARD_HEADER_SIZE],
                ))?;
                let shard_value_encoding = ValueEncoding::from_u8(sh.value_encoding)
                    .ok_or(OpsError::UnknownValueEncoding(sh.value_encoding))?;
                let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;
                let shard_local_row_start =
                    shard_entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);

                let indptr_u64: Vec<u64> = indptr.iter().map(|&v| v as u64).collect();
                let indices_u32: Vec<u32> = indices.iter().map(|&v| v as u32).collect();
                let mut values_bytes = Vec::new();
                for &v in &data {
                    encode_value(&mut values_bytes, v, shard_value_encoding)?;
                }
                let shard_codec = select_codec(&values_bytes, shard_value_encoding);
                writer.write_csr_shard_for(
                    modality_id,
                    &indptr_u64,
                    &indices_u32,
                    &values_bytes,
                    shard_codec,
                    shard_value_encoding,
                    input_offset + shard_local_row_start,
                )?;
            }
            input_offset += reader.n_obs();
        }
    }

    // Per-modality obsm: concatenate across inputs row-wise. Drop if
    // any input is missing it (consistent with single-modality merge).
    for (idx, info) in table.entries.iter().enumerate() {
        let modality_id = (idx + 1) as u8;
        let prefix = format!("obsm/{}/", info.name);
        let mut obsm_keys = std::collections::BTreeSet::new();
        for entry in &readers[0].catalog().entries {
            if entry.section_type == SectionType::ObsmEmbedding
                && entry.modality_id == modality_id
                && entry.name.starts_with(&prefix)
            {
                if let Some(k) = entry.name.strip_prefix(&prefix) {
                    obsm_keys.insert(k.to_string());
                }
            }
        }
        for key in obsm_keys {
            let mut batches = Vec::new();
            let mut all_present = true;
            for reader in readers {
                match reader.read_obsm_for(modality_id, &key) {
                    Ok(b) => batches.push(b),
                    Err(_) => {
                        all_present = false;
                        break;
                    }
                }
            }
            if all_present && !batches.is_empty() {
                let merged = concat_batches(&batches[0].schema(), &batches)?;
                writer.write_obsm_for(modality_id, &key, &merged)?;
            }
        }
    }

    // Per-modality uns: copy from first input (single source of truth).
    for (idx, _info) in table.entries.iter().enumerate() {
        let modality_id = (idx + 1) as u8;
        if let Ok(uns) = readers[0].read_uns_for(modality_id) {
            writer.write_uns_for(modality_id, &uns)?;
        }
    }

    // Global uns from first input.
    if let Ok(uns) = readers[0].read_uns() {
        writer.write_uns(&uns)?;
    }

    // Provenance: concatenate input chains, then stamp merge op.
    let mut all_prov = Vec::new();
    for reader in readers {
        if let Ok(prov) = reader.read_provenance() {
            all_prov.extend(prov.operations);
        }
    }
    all_prov.push(ProvenanceEntry {
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
    writer.write_provenance(all_prov)?;
    writer.finish()?;
    Ok(())
}

/// Phase F.4 helper: two modality tables match iff they have the same
/// length and every entry agrees on (name, modality_type, n_vars).
/// Two `None`s also match (both inputs single-modality). Mixed
/// `Some` / `None` does not match.
fn modality_tables_match(
    a: Option<&scx_format::ModalityTable>,
    b: Option<&scx_format::ModalityTable>,
) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(_), None) | (None, Some(_)) => false,
        (Some(x), Some(y)) => {
            if x.len() != y.len() {
                return false;
            }
            x.entries.iter().zip(y.entries.iter()).all(|(p, q)| {
                p.name == q.name && p.modality_type == q.modality_type && p.n_vars == q.n_vars
            })
        }
    }
}
