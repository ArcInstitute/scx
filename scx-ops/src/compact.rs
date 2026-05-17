// Compact operation: rewrite file without deleted rows or stale catalogs.

use std::path::Path;

use arrow::compute;
use scx_codec::{CodecId, ValueEncoding};
use scx_format::codec_select::select_codec;
use scx_format::header::{FileHeader, MAGIC};
use scx_format::provenance::ProvenanceEntry;
use scx_format::section::SectionType;
use scx_format::writer::ScxWriter;
use scx_format::ScxReader;

use crate::error::Result;
use crate::flock::SharedFileLock;
use crate::helpers::encode_value;

/// Compact an SCX file: removes deleted rows, stale catalogs, and produces
/// a clean single-catalog file.
pub fn compact(input_path: &Path, output_path: &Path) -> Result<()> {
    // Acquire shared lock to prevent concurrent writers from modifying the
    // file while we read it. The lock is held until `_lock` is dropped.
    let _lock = SharedFileLock::acquire(input_path)?;
    let reader = ScxReader::open(input_path)?;
    let in_header = reader.header().clone();

    // Phase 6: dispatch to the multimodal compact path. The single-modality
    // path below assumes one modality and would silently flatten the
    // ModalityTable.
    if reader.is_multimodal() {
        return compact_multimodal(reader, in_header, output_path);
    }
    let in_header = &in_header;

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

    // CSC sidecars (column-major shards) are dropped by `compact`: the
    // operation re-shards CSR rows on a different row layout, so any
    // input CSC shards would silently reference stale row indices.
    // Caller can opt back in via `--rebuild-csc` on the CLI to re-run
    // `build-csc` against the compacted output. Phase H.2.
    let had_csc = in_header.has_csc();
    if had_csc {
        log::warn!(
            "compact dropped CSC shards from {input}: rerun \
             `scx build-csc` (or pass --rebuild-csc) to restore the \
             column-major sidecar",
            input = input_path.display()
        );
    }
    // Carry all input flags except `has_deletion_vectors` (bit 5) — the
    // compacted output applies the deletion vector and drops it — and
    // `has_csc` (bit 0) — the CSC sidecar is dropped explicitly above.
    let out_flags = in_header.flags & !(1 << 5) & !(1 << 0);

    // Set up output header
    let out_header = FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: out_flags,
        n_obs: new_n_obs as u64,
        n_vars,
        nnz: 0, // will be set by finish
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: in_header.shard_target_rows,
        codec_id: 0,
        index_dtype: in_header.index_dtype,
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

    // Copy obsm flag if present
    let has_obsm = in_header.has_obsm();

    let mut writer = ScxWriter::new(output_path, out_header)?;
    writer.write_obs(&filtered_obs)?;
    writer.write_var(&var)?;

    // Process CSR shards: decode, filter deleted rows, re-shard.
    // Read value_encoding from first shard header for the main X matrix.
    let shards = reader.catalog().shards_sorted();
    let value_encoding = {
        if !shards.is_empty() {
            let section = reader.section_bytes(shards[0])?;
            let sh = scx_format::ShardHeader::read_from(&mut std::io::Cursor::new(
                &section[..scx_format::SHARD_HEADER_SIZE],
            ))?;
            ValueEncoding::from_u8(sh.value_encoding).ok_or(
                crate::error::OpsError::UnknownValueEncoding(sh.value_encoding),
            )?
        } else {
            ValueEncoding::Uint8
        }
    };
    let shard_target = in_header.shard_target_rows;

    // Accumulate filtered CSR data
    let mut acc_indptr: Vec<u64> = vec![0];
    let mut acc_indices: Vec<u32> = Vec::new();
    let mut acc_values: Vec<u8> = Vec::new();
    let mut acc_row_count = 0u64;
    let mut emitted_rows = 0u64; // tracks output row numbering

    for shard_entry in &shards {
        let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;
        let shard_row_start = shard_entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
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
                encode_value(&mut acc_values, data[j], value_encoding)?;
            }

            let prev = *acc_indptr.last().unwrap();
            acc_indptr.push(prev + row_nnz as u64);
            acc_row_count += 1;

            // Flush accumulated rows as a shard when hitting target
            if acc_row_count >= shard_target as u64 {
                let shard_row_start = emitted_rows;
                // Auto-select optimal codec for this shard's data
                let shard_codec = select_codec(&acc_values, value_encoding);
                writer.write_csr_shard(
                    &acc_indptr,
                    &acc_indices,
                    &acc_values,
                    shard_codec,
                    value_encoding,
                    shard_row_start,
                )?;
                emitted_rows += acc_row_count;
                acc_indptr = vec![0];
                acc_indices.clear();
                acc_values.clear();
                acc_row_count = 0;
            }
        }
    }

    // Flush remaining accumulated rows
    if acc_row_count > 0 {
        let shard_row_start = emitted_rows;
        // Auto-select optimal codec for remaining shard
        let shard_codec = select_codec(&acc_values, value_encoding);
        writer.write_csr_shard(
            &acc_indptr,
            &acc_indices,
            &acc_values,
            shard_codec,
            value_encoding,
            shard_row_start,
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
        // Determine this layer's value encoding from its first shard header
        let layer_prefix = format!("{layer_name}_shard_");
        let layer_shard_entries: Vec<&scx_format::FullCatalogEntry> = reader
            .catalog()
            .entries
            .iter()
            .filter(|e| {
                e.section_type == SectionType::LayerCsrShard && e.name.starts_with(&layer_prefix)
            })
            .collect();
        let layer_value_encoding = if let Some(first_entry) = layer_shard_entries.first() {
            let section = reader.section_bytes(first_entry)?;
            let sh = scx_format::ShardHeader::read_from(&mut std::io::Cursor::new(
                &section[..scx_format::SHARD_HEADER_SIZE],
            ))?;
            ValueEncoding::from_u8(sh.value_encoding).ok_or(
                crate::error::OpsError::UnknownValueEncoding(sh.value_encoding),
            )?
        } else {
            ValueEncoding::Uint8
        };

        let layer = reader.read_layer(layer_name)?;
        // Filter and write layer shards
        let mut layer_indptr: Vec<u64> = vec![0];
        let mut layer_indices: Vec<u32> = Vec::new();
        let mut layer_values: Vec<u8> = Vec::new();
        let mut layer_row_count = 0u64;
        let mut layer_shard_idx = 0u32;
        let mut emitted_layer_rows = 0u64;

        for row_idx in 0..layer.shape.0 {
            let deleted = keep_mask.as_ref().is_some_and(|mask| !mask[row_idx]);
            if deleted {
                continue;
            }

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

        if layer_row_count > 0 {
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

/// Phase 6: compact a multimodal SCX file. Applies the global keep
/// mask to every modality's CSR shards (and per-modality layers) while
/// preserving the ModalityTable and per-modality var / obsm / uns.
/// Per-modality CSC sidecars are dropped (rebuild via
/// `--rebuild-csc`).
fn compact_multimodal(reader: ScxReader, in_header: FileHeader, output_path: &Path) -> Result<()> {
    let dv = reader.read_deletion_vectors()?;

    let n_obs = in_header.n_obs as usize;
    let keep_mask = build_keep_mask(n_obs, &dv, reader.catalog());
    let obs = reader.read_obs()?;
    let filtered_obs = if let Some(ref mask) = keep_mask {
        let bool_array = arrow::array::BooleanArray::from(mask.clone());
        compute::filter_record_batch(&obs, &bool_array)?
    } else {
        obs
    };
    let new_n_obs = filtered_obs.num_rows();

    // Carry input flags except has_deletion_vectors (applied) and
    // has_csc (per-modality CSC sidecars are dropped on compact;
    // caller rebuilds via --rebuild-csc).
    let out_flags = in_header.flags & !(1 << 5) & !(1 << 0);
    let table = reader
        .modality_table()
        .ok_or_else(|| {
            scx_format::ScxError::InvalidCatalog(
                "compact_multimodal: file has no modality table".to_string(),
            )
        })?
        .clone();

    let any_input_had_csc = table.entries.iter().any(|info| info.flags.has_csc());
    if any_input_had_csc {
        log::warn!(
            "compact dropped per-modality CSC shards from {input}: \
             rerun `scx build-csc` (or pass --rebuild-csc) to restore the \
             column-major sidecar",
            input = output_path.display()
        );
    }

    let max_n_vars = table
        .entries
        .iter()
        .map(|info| info.n_vars)
        .max()
        .unwrap_or(0);

    let out_header = FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: out_flags,
        n_obs: new_n_obs as u64,
        n_vars: max_n_vars,
        nnz: 0,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: in_header.shard_target_rows,
        codec_id: 0,
        index_dtype: in_header.index_dtype,
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
    writer.write_obs(&filtered_obs)?;

    let shard_target = in_header.shard_target_rows;

    // Iterate modalities in registration order so the output's
    // modality_id assignment matches the input.
    for (idx, info) in table.entries.iter().enumerate() {
        let in_modality_id = (idx + 1) as u8;
        let default_codec = CodecId::from_u8(info.default_codec_id)
            .ok_or(crate::error::OpsError::UnknownCodec(info.default_codec_id))?;
        let default_value_encoding = ValueEncoding::from_u8(info.default_value_encoding).ok_or(
            crate::error::OpsError::UnknownValueEncoding(info.default_value_encoding),
        )?;
        let out_modality_id = writer.add_modality(
            &info.name,
            info.modality_type,
            default_codec,
            default_value_encoding,
            false,
        )?;
        debug_assert_eq!(in_modality_id, out_modality_id);
        writer.set_modality_n_vars(out_modality_id, info.n_vars)?;

        // Per-modality var.
        let var = reader.read_var_for(in_modality_id)?;
        writer.write_var_for(out_modality_id, &var)?;

        // Per-modality CSR shards. Filter by global keep_mask and
        // re-shard to shard_target_rows.
        let value_encoding = {
            let entries = reader.catalog().csr_shards_for_modality(in_modality_id);
            if let Some(first) = entries.first() {
                let section = reader.section_bytes(first)?;
                let sh = scx_format::ShardHeader::read_from(&mut std::io::Cursor::new(
                    &section[..scx_format::SHARD_HEADER_SIZE],
                ))?;
                ValueEncoding::from_u8(sh.value_encoding).ok_or(
                    crate::error::OpsError::UnknownValueEncoding(sh.value_encoding),
                )?
            } else {
                default_value_encoding
            }
        };

        let mut acc_indptr: Vec<u64> = vec![0];
        let mut acc_indices: Vec<u32> = Vec::new();
        let mut acc_values: Vec<u8> = Vec::new();
        let mut acc_rows = 0u64;
        let mut emitted_rows = 0u64;

        for shard_entry in reader.catalog().csr_shards_for_modality(in_modality_id) {
            let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;
            let shard_row_start = shard_entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
            let shard_n_rows = indptr.len() - 1;

            for local_row in 0..shard_n_rows {
                let global_idx = shard_row_start + local_row as u64;
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
                    encode_value(&mut acc_values, data[j], value_encoding)?;
                }
                let prev = *acc_indptr.last().unwrap();
                acc_indptr.push(prev + row_nnz as u64);
                acc_rows += 1;

                if acc_rows >= shard_target as u64 {
                    let codec = select_codec(&acc_values, value_encoding);
                    writer.write_csr_shard_for(
                        out_modality_id,
                        &acc_indptr,
                        &acc_indices,
                        &acc_values,
                        codec,
                        value_encoding,
                        emitted_rows,
                    )?;
                    emitted_rows += acc_rows;
                    acc_indptr = vec![0];
                    acc_indices.clear();
                    acc_values.clear();
                    acc_rows = 0;
                }
            }
        }

        if acc_rows > 0 {
            let codec = select_codec(&acc_values, value_encoding);
            writer.write_csr_shard_for(
                out_modality_id,
                &acc_indptr,
                &acc_indices,
                &acc_values,
                codec,
                value_encoding,
                emitted_rows,
            )?;
        }

        // Per-modality obsm — row-filter by keep_mask. Section names
        // are `obsm/{modality_name}/{key}`.
        let obsm_prefix = format!("obsm/{}/", info.name);
        let obsm_entries: Vec<scx_format::FullCatalogEntry> = reader
            .catalog()
            .entries
            .iter()
            .filter(|e| {
                e.section_type == SectionType::ObsmEmbedding
                    && e.modality_id == in_modality_id
                    && e.name.starts_with(&obsm_prefix)
            })
            .cloned()
            .collect();
        for entry in &obsm_entries {
            let key = entry.name.strip_prefix(&obsm_prefix).unwrap_or(&entry.name);
            let batch = reader.read_obsm_for(in_modality_id, key)?;
            let filtered = if let Some(ref mask) = keep_mask {
                let bool_array = arrow::array::BooleanArray::from(mask.clone());
                compute::filter_record_batch(&batch, &bool_array)?
            } else {
                batch
            };
            writer.write_obsm_for(out_modality_id, key, &filtered)?;
        }

        // Per-modality uns.
        if let Ok(uns) = reader.read_uns_for(in_modality_id) {
            writer.write_uns_for(out_modality_id, &uns)?;
        }

        // Per-modality layers. Catalog entries follow
        // `layer/{modality_name}/{layer_name}/shard_{idx}`. Re-shard
        // with the same keep_mask using `write_layer_csr_shard_for`.
        let layer_prefix = format!("layer/{}/", info.name);
        let mut layer_names = std::collections::BTreeSet::new();
        for entry in &reader.catalog().entries {
            if entry.section_type == SectionType::LayerCsrShard
                && entry.modality_id == in_modality_id
                && entry.name.starts_with(&layer_prefix)
            {
                if let Some(remainder) = entry.name.strip_prefix(&layer_prefix) {
                    if let Some(pos) = remainder.find("/shard_") {
                        layer_names.insert(remainder[..pos].to_string());
                    }
                }
            }
        }

        for layer_name in layer_names {
            let csr = reader.read_layer_for(in_modality_id, &layer_name)?;
            let layer_value_encoding = {
                let shards = reader
                    .catalog()
                    .layer_csr_shards_for_modality(in_modality_id, &layer_name);
                if let Some(first) = shards.first() {
                    let section = reader.section_bytes(first)?;
                    let sh = scx_format::ShardHeader::read_from(&mut std::io::Cursor::new(
                        &section[..scx_format::SHARD_HEADER_SIZE],
                    ))?;
                    ValueEncoding::from_u8(sh.value_encoding).ok_or(
                        crate::error::OpsError::UnknownValueEncoding(sh.value_encoding),
                    )?
                } else {
                    value_encoding
                }
            };

            let mut l_indptr: Vec<u64> = vec![0];
            let mut l_indices: Vec<u32> = Vec::new();
            let mut l_values: Vec<u8> = Vec::new();
            let mut l_rows = 0u64;
            let mut l_emitted = 0u64;
            let mut l_shard_idx = 0u32;

            for row_idx in 0..csr.shape.0 {
                let deleted = keep_mask.as_ref().is_some_and(|mask| !mask[row_idx]);
                if deleted {
                    continue;
                }
                let row_start = csr.indptr[row_idx] as usize;
                let row_end = csr.indptr[row_idx + 1] as usize;
                for j in row_start..row_end {
                    l_indices.push(csr.indices[j] as u32);
                    encode_value(&mut l_values, csr.data[j], layer_value_encoding)?;
                }
                let prev = *l_indptr.last().unwrap();
                l_indptr.push(prev + (row_end - row_start) as u64);
                l_rows += 1;

                if l_rows >= shard_target as u64 {
                    let codec = select_codec(&l_values, layer_value_encoding);
                    writer.write_layer_csr_shard_for(
                        out_modality_id,
                        &layer_name,
                        l_shard_idx,
                        &l_indptr,
                        &l_indices,
                        &l_values,
                        codec,
                        layer_value_encoding,
                        l_emitted,
                    )?;
                    l_emitted += l_rows;
                    l_indptr = vec![0];
                    l_indices.clear();
                    l_values.clear();
                    l_rows = 0;
                    l_shard_idx += 1;
                }
            }
            if l_rows > 0 {
                let codec = select_codec(&l_values, layer_value_encoding);
                writer.write_layer_csr_shard_for(
                    out_modality_id,
                    &layer_name,
                    l_shard_idx,
                    &l_indptr,
                    &l_indices,
                    &l_values,
                    codec,
                    layer_value_encoding,
                    l_emitted,
                )?;
            }
        }
    }

    // Global obsm (modality_id == 0). Single-modality compact handles
    // these via the `has_obsm` branch; mirror that path for multimodal.
    if in_header.has_obsm() {
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

    // Global uns.
    if let Ok(uns) = reader.read_uns() {
        writer.write_uns(&uns)?;
    }

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

    // Map shard_id (sort-order index from shards_sorted()) to deletion bitmaps.
    // This is consistent with delete.rs which uses the same sort-order index.
    let shards = catalog.shards_sorted();
    for (shard_idx, shard_entry) in shards.iter().enumerate() {
        if let Some(ref stats) = shard_entry.stats {
            let shard_idx = shard_idx as u32;

            if let Some(bitmap) = dv.shards.get(&shard_idx) {
                for local_row in bitmap.iter() {
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
