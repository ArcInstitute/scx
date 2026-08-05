// Shared helpers for copying auxiliary sections between SCX files.
//
// Used by build_csc and upgrade to avoid duplicating layer/obsm/uns/
// predicate-index/provenance copy logic.

use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::catalog::{FullCatalogEntry, ShardStats};
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::section::SectionType;
use scx_format_io::shard::{ShardHeader, DEFAULT_WRITE_SHARD_FORMAT_VERSION, SHARD_HEADER_SIZE};
use scx_format_io::writer::ScxWriter;
use scx_format_io::ResolvedCodec;
use scx_format_io::{compute_shard_stats, MajorAxis, ScxReader};

use crate::error::{OpsError, Result as OpsResult};

/// Core eligibility for raw-copying a CSR shard's section bytes verbatim:
/// the source shard's index dtype, column extent, and codec all match what
/// the target would emit, so a byte-copy reproduces the decode/re-encode
/// output. Callers AND their own extra preconditions onto this:
/// `append` adds `sh.n_major <= shard_target_rows` (it may re-split shards);
/// `merge` adds `!assume_identical_var` (column indices are not guaranteed
/// identical when the var axis is assumed-but-not-verified equal).
///
/// **Framing gate (`output_framed`).** A shard may only be byte-copied when its
/// framing matches the output file's: a framed (shard v2) shard into a v4 file,
/// or an unframed (v1) shard into a ≤v3 file. A framing mismatch forces the
/// decode-encode path, which re-emits the shard at the output's framing so the
/// file stays self-consistent (and passes
/// [`ScxWriter::guard_no_legacy_shard_in_v4`], which rejects an unframed CSR
/// shard in a v4 file). Byte-copying a v2 shard into a ≤v3 file would leave a
/// framed shard under a header a pre-framing reader accepts, which then
/// mis-decodes the per-group local-rebased indptr as global; byte-copying a v1
/// shard into a v4 file would advertise sub-shard random access it cannot honor.
/// **Codec gate.** A byte-copy preserves the source shard's codec exactly, so
/// whether it is eligible depends on what the caller's intent asked for:
///
/// | intent | raw-copy | why |
/// |---|---|---|
/// | explicit codec | only if `sh.codec_id` already matches | the copy would ignore the force |
/// | `auto` / `fast` | **yes** | preserving the source codec is size-neutral and free; re-encoding to "re-decide" a codec the source already chose adaptively is pure cost |
/// | `compact` / `compact-trial` | **no** | the caller explicitly asked for a size-max re-encode; a byte-copy would silently ignore it |
pub(crate) fn raw_copy_csr_eligible(
    sh: &ShardHeader,
    target_index_dtype: u8,
    target_n_vars: u64,
    codec: ResolvedCodec,
    output_framed: bool,
) -> bool {
    let shard_framed = sh.shard_format_version > DEFAULT_WRITE_SHARD_FORMAT_VERSION;
    let codec_ok = match codec.explicit_codec {
        Some(c) => sh.codec_id == c as u8,
        // `compact`/`compact-trial` request a size-max re-encode; honour it.
        // `auto`/`fast` are satisfied by keeping what the source already has.
        None => !codec.requires_framing && !codec.codec_trial,
    };
    shard_framed == output_framed
        && sh.index_dtype == target_index_dtype
        && (sh.n_minor as u64) == target_n_vars
        && codec_ok
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
        return Err(OpsError::Format(scx_format_io::ScxError::InvalidCatalog(
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
                    return Err(OpsError::Format(scx_format_io::ScxError::InvalidCatalog(
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
        let layer_shard_entries: Vec<&scx_format_io::FullCatalogEntry> = reader
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

#[cfg(test)]
mod tests {
    use super::*;
    use scx_format_io::shard::{ShardHeader, SHARD_MAGIC};

    fn header(shard_format_version: u8, codec_id: u8) -> ShardHeader {
        ShardHeader {
            magic: SHARD_MAGIC,
            shard_format_version,
            shard_type: 0,
            codec_id,
            value_encoding: ValueEncoding::Uint8 as u8,
            index_dtype: 0,
            reserved_flags: [0u8; 3],
            n_major: 4,
            n_minor: 10,
            nnz: 8,
            global_offset: 0,
            indptr_rel_offset: 0,
            indptr_length: 0,
            indices_rel_offset: 0,
            indices_length: 0,
            values_rel_offset: 0,
            values_length: 0,
            block_index_rel_offset: 0,
            block_index_length: 0,
            checksum: [0u8; 8],
        }
    }

    /// Framing gate: raw-copy requires the source shard's framing to match the
    /// output's framing exactly. A framed (v2) source is eligible only into a
    /// framed (v4) output; an unframed (v1) source only into a ≤v3 output.
    /// Byte-copying across a framing boundary would let a reader mis-decode the
    /// shard (a v2 shard under a pre-framing reader, or an unframed v1 shard
    /// under a v4 header whose readers expect frames).
    #[test]
    fn framed_shard_raw_copy_eligibility_tracks_output_framing() {
        let v1 = header(scx_format_io::shard::DEFAULT_WRITE_SHARD_FORMAT_VERSION, 0);
        let v2 = header(
            scx_format_io::shard::DEFAULT_WRITE_SHARD_FORMAT_VERSION + 1,
            0,
        );
        // Unframed (v1) shard: eligible ONLY into unframed output.
        assert!(
            raw_copy_csr_eligible(&v1, 0, 10, ResolvedCodec::AUTO, false),
            "unframed v1 shard must be eligible into unframed output"
        );
        assert!(
            !raw_copy_csr_eligible(&v1, 0, 10, ResolvedCodec::AUTO, true),
            "unframed v1 shard must NOT be raw-copy eligible into framed output"
        );
        // Framed (v2) shard: eligible ONLY into framed (v4) output.
        assert!(
            !raw_copy_csr_eligible(&v2, 0, 10, ResolvedCodec::AUTO, false),
            "framed v2 shard must NOT be raw-copy eligible into unframed output"
        );
        assert!(
            raw_copy_csr_eligible(&v2, 0, 10, ResolvedCodec::AUTO, true),
            "framed v2 shard must be raw-copy eligible into framed output"
        );
    }
}
