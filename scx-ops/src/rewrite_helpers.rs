// Shared helpers for copying auxiliary sections between SCX files.
//
// Used by build_csc and upgrade to avoid duplicating layer/obsm/uns/
// predicate-index/provenance copy logic.

use std::collections::HashMap;

use scx_codec::{CodecId, CodecSelection, ValueEncoding};
use scx_format::catalog::{FullCatalogEntry, ShardStats};
use scx_format::decode_sidecar::DecodeSidecar;
use scx_format::provenance::ProvenanceEntry;
use scx_format::section::SectionType;
use scx_format::shard::{ShardHeader, SHARD_HEADER_SIZE};
use scx_format::writer::ScxWriter;
use scx_format::{compute_shard_stats, MajorAxis, ScxReader};

use crate::error::{OpsError, Result as OpsResult};

/// Core eligibility for raw-copying a CSR shard's section bytes verbatim:
/// the source shard's index dtype, column extent, and codec all match what
/// the target would emit, so a byte-copy reproduces the decode/re-encode
/// output. Callers AND their own extra preconditions onto this:
/// `append` adds `sh.n_major <= shard_target_rows` (it may re-split shards);
/// `merge` adds `!assume_identical_var` (column indices are not guaranteed
/// identical when the var axis is assumed-but-not-verified equal).
pub(crate) fn raw_copy_csr_eligible(
    sh: &ShardHeader,
    target_index_dtype: u8,
    target_n_vars: u64,
    codec: CodecSelection,
) -> bool {
    sh.index_dtype == target_index_dtype
        && (sh.n_minor as u64) == target_n_vars
        && match codec {
            CodecSelection::Auto => true,
            CodecSelection::Explicit(c) => sh.codec_id == c as u8,
        }
}

/// Build the patched section bytes + stats for a raw-copied CSR shard,
/// sink-agnostic.
///
/// Patches only `ShardHeader.n_minor` (→ `target_n_vars`) and `global_offset`
/// (→ `global_row_start`); the indptr/indices/values/block-index payload is
/// byte-identical to the source. The caller writes the returned bytes through
/// its own sink — the in-place `FileLock` (append) or
/// [`ScxWriter::write_csr_shard_raw_copy`] (merge) — and records the catalog
/// entry. Reuses the source entry's stats (only the row range is
/// position-dependent); decodes to recompute stats only when the source entry
/// is missing them (format-permitted but not produced by the current writer).
pub(crate) fn build_raw_copied_csr_section(
    source: &ScxReader,
    entry: &FullCatalogEntry,
    sh: &ShardHeader,
    target_n_vars: u64,
    global_row_start: u64,
    value_encoding: ValueEncoding,
) -> OpsResult<(Vec<u8>, ShardStats)> {
    // `read_raw_shard_bytes` returns an mmap slice. We never mutate it: the
    // patched header is built in a separate `header_buf` and the payload is
    // copied read-only into `section_data`, so borrow it directly (no alloc).
    let src_bytes = source.read_raw_shard_bytes(entry)?;
    if src_bytes.len() < SHARD_HEADER_SIZE {
        return Err(OpsError::Format(scx_format::ScxError::InvalidCatalog(
            format!("source shard '{}' too small for header", entry.name),
        )));
    }

    let new_sh = ShardHeader {
        n_minor: target_n_vars as u32,
        global_offset: global_row_start,
        ..sh.clone()
    };
    let mut header_buf = Vec::with_capacity(SHARD_HEADER_SIZE);
    new_sh.write_to(&mut header_buf)?;

    let mut section_data = Vec::with_capacity(src_bytes.len());
    section_data.extend_from_slice(&header_buf);
    section_data.extend_from_slice(&src_bytes[SHARD_HEADER_SIZE..]);

    // Reuse the source entry's stats; only the row range is position-dependent.
    // `nnz`, `value_min/max/sum`, `col_start/col_end`, and `column_stats` are
    // invariant under raw copy because eligibility already requires
    // `sh.n_minor == target_n_vars` (so the column extent is preserved).
    let stats = match entry.stats.as_ref() {
        Some(src_stats) => {
            let mut s = src_stats.clone();
            s.row_start = global_row_start;
            s.row_end = global_row_start + sh.n_major as u64;
            s
        }
        None => {
            let codec_id =
                CodecId::from_u8(sh.codec_id).ok_or(OpsError::UnknownCodec(sh.codec_id))?;
            if codec_id == CodecId::None {
                let values_start = sh.values_rel_offset as usize;
                let values_end = values_start + sh.values_length as usize;
                // Defend against a malformed/corrupt header: error rather than
                // panic on an out-of-bounds values slice (readers never panic
                // on bad input).
                if values_start > values_end || values_end > src_bytes.len() {
                    return Err(OpsError::Format(scx_format::ScxError::InvalidCatalog(
                        format!(
                            "source shard '{}' has invalid values offset/length \
                             ({values_start}..{values_end} of {} bytes)",
                            entry.name,
                            src_bytes.len(),
                        ),
                    )));
                }
                compute_shard_stats(
                    &src_bytes[values_start..values_end],
                    value_encoding,
                    MajorAxis::Row,
                    global_row_start,
                    sh.n_major as u64,
                    target_n_vars,
                    sh.nnz,
                )
            } else {
                let (_, _, val_f32) = source.read_shard_from_entry(entry)?;
                let raw = scx_codec::values_to_raw_bytes(&val_f32, value_encoding)?;
                compute_shard_stats(
                    &raw,
                    value_encoding,
                    MajorAxis::Row,
                    global_row_start,
                    sh.n_major as u64,
                    target_n_vars,
                    sh.nnz,
                )
            }
        }
    };

    Ok((section_data, stats))
}

/// Index a reader's `DecodeMetadataShard` sidecar entries by their parent
/// shard name (the sidecar name with the `decode/` prefix stripped), so the
/// merge X loop can resolve a shard's sidecar in `O(1)` instead of scanning
/// the whole catalog per shard (avoids `O(shards × entries)` on atlas-scale
/// merges). Build once per reader.
pub(crate) fn decode_sidecar_index(reader: &ScxReader) -> HashMap<&str, &FullCatalogEntry> {
    reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::DecodeMetadataShard)
        .filter_map(|e| e.name.strip_prefix("decode/").map(|parent| (parent, e)))
        .collect()
}

/// Look up the decode sidecar the source file holds for `shard_entry`
/// (named `decode/{shard_name}`) via a prebuilt [`decode_sidecar_index`],
/// parsed into a [`DecodeSidecar`] with its `major_start` re-stamped to
/// `global_row_start`. Returns `None` when the source has no sidecar for this
/// shard (non-Scx1 codec, sidecar over the overhead budget, or a pre-v3
/// source). The remaining source-section identity fields are re-stamped by
/// [`ScxWriter::write_csr_shard_raw_copy`] once the shard's final position is
/// known.
pub(crate) fn source_decode_sidecar(
    source: &ScxReader,
    sidecar_index: &HashMap<&str, &FullCatalogEntry>,
    shard_entry: &FullCatalogEntry,
    global_row_start: u64,
) -> OpsResult<Option<DecodeSidecar>> {
    match sidecar_index.get(shard_entry.name.as_str()) {
        Some(e) => {
            let mut sc = source.read_decode_sidecar_from_entry(e)?;
            sc.major_start = global_row_start;
            Ok(Some(sc))
        }
        None => Ok(None),
    }
}

/// Copy all auxiliary sections (layers, obsm, uns, predicate indices) from
/// reader to writer, then append a new provenance entry.
pub fn copy_auxiliary_sections(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    action: &str,
    params_json: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    copy_layers(reader, writer)?;
    copy_obsm(reader, writer)?;
    copy_uns(reader, writer)?;
    copy_predicate_indices(reader, writer)?;
    append_provenance(reader, writer, action, params_json)?;
    Ok(())
}

/// Copy all layer CSR shards from reader to writer, preserving per-shard codec.
fn copy_layers(
    reader: &ScxReader,
    writer: &mut ScxWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    let layer_names = reader.layer_names();
    for layer_name in &layer_names {
        let layer_prefix = format!("{layer_name}_shard_");
        let layer_shard_entries: Vec<&scx_format::FullCatalogEntry> = reader
            .catalog()
            .entries
            .iter()
            .filter(|e| {
                e.section_type == SectionType::LayerCsrShard && e.name.starts_with(&layer_prefix)
            })
            .collect();

        let mut sorted_entries = layer_shard_entries;
        sorted_entries.sort_by_key(|e| e.stats.as_ref().map_or(u64::MAX, |s| s.row_start));

        for (shard_idx, entry) in sorted_entries.iter().enumerate() {
            let sh = reader.read_shard_header(entry)?;
            let ve = ValueEncoding::from_u8(sh.value_encoding)
                .ok_or(format!("unknown value encoding: {}", sh.value_encoding))?;
            let ci =
                CodecId::from_u8(sh.codec_id).ok_or(format!("unknown codec: {}", sh.codec_id))?;

            let (indptr, indices, data) = reader.read_shard_from_entry(entry)?;
            let row_start = entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
            let indptr_u64: Vec<u64> = indptr.iter().map(|&v| v as u64).collect();
            let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
            let mut raw_values = Vec::new();
            for &v in &data {
                ve.encode_f32(&mut raw_values, v)?;
            }
            writer.write_layer_csr_shard(
                &indptr_u64,
                &indices_u32,
                &raw_values,
                ci,
                ve,
                row_start,
                layer_name,
                shard_idx as u32,
            )?;
        }
    }
    Ok(())
}

/// Copy obsm sections from reader to writer.
fn copy_obsm(reader: &ScxReader, writer: &mut ScxWriter) -> Result<(), Box<dyn std::error::Error>> {
    if reader.header().has_obsm() {
        let all_obsm = reader.read_all_obsm()?;
        for (name, batch) in &all_obsm {
            writer.write_obsm(name, batch)?;
        }
    }
    Ok(())
}

/// Copy uns section from reader to writer.
fn copy_uns(reader: &ScxReader, writer: &mut ScxWriter) -> Result<(), Box<dyn std::error::Error>> {
    if let Ok(uns) = reader.read_uns() {
        writer.write_uns(&uns)?;
    }
    Ok(())
}

/// Copy predicate index sections from reader to writer.
pub(crate) fn copy_predicate_indices(
    reader: &ScxReader,
    writer: &mut ScxWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Ok(Some(data)) = reader.read_obs_predicate_index_bytes() {
        writer.write_obs_predicate_index(data)?;
    }
    if let Ok(Some(data)) = reader.read_var_predicate_index_bytes() {
        writer.write_var_predicate_index(data)?;
    }
    Ok(())
}

/// Read existing provenance, append a new entry, and write to writer.
pub(crate) fn append_provenance(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    action: &str,
    params_json: &str,
) -> Result<(), Box<dyn std::error::Error>> {
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
        action: action.to_string(),
        tool: format!("scx-cli {}", env!("CARGO_PKG_VERSION")),
        params_json: params_json.to_string(),
        input_checksums: vec![],
    });
    writer.write_provenance(prov_entries)?;
    Ok(())
}
