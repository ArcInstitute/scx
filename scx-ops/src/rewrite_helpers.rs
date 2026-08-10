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

/// Copy obs and var through a rewrite that preserves **both axes 1:1**,
/// keeping a row-sharded layout sharded.
///
/// This is the safe front door to `compact::write_obs_shards_streaming` /
/// `optimize::write_var_shards_streaming`, and the reason they are not exported
/// directly. The obs helper takes a `keep_mask` and a `total_kept`, and neither
/// is checkable from inside it: a mask shorter than the shards it is applied to
/// panics on the slice in `filtered_obs_shards`, and a wrong `total_kept` is
/// stamped into every output shard, producing a file whose own `n_rows_total`
/// disagrees with its cover. Both are fine for the in-crate callers that derive
/// them from `build_keep_mask(n_obs, …)`; neither is something a downstream
/// caller should have to know. So the filtering primitive stays crate-private
/// and this — the only case a rewrite outside `scx-ops` needs — takes no
/// arguments beyond the two files and derives the totals from the reader.
///
/// The alternative each caller reaches for otherwise is `read_obs()` +
/// `write_obs()`, which assembles the whole axis into one in-memory batch and
/// emits a single legacy section: peak RSS O(n_obs) — the OOM the sharded
/// layout exists to prevent — plus the silent loss of the row-sharded-obs
/// precondition Level-2 row-set pushdown depends on. `build_csc` and
/// `scx upgrade` had both written that by hand; this is the shared version.
///
/// A legacy single-section input has no per-shard reader and falls through to
/// the materialising path, which is what it already was.
pub fn copy_obs_var_preserving_layout(
    reader: &ScxReader,
    writer: &mut ScxWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    let (n_obs, n_vars) = (reader.header().n_obs, reader.header().n_vars);
    if reader.obs_metadata_shard_count() > 0 {
        crate::compact::write_obs_shards_streaming(reader, writer, None, n_obs as usize)?;
    } else {
        writer.write_obs(&reader.read_obs()?)?;
    }
    if reader.var_metadata_shard_count() > 0 {
        crate::optimize::write_var_shards_streaming(reader, writer, n_vars)?;
    } else {
        writer.write_var(&reader.read_var()?)?;
    }
    Ok(())
}

/// The value encoding to re-emit canonicalized values under, given the one the
/// source shard used.
///
/// Canonicalization **sums duplicate coordinates**, so it can produce a value
/// larger than any the source held — and the source's encoding was chosen to fit
/// the source's values. Re-encoding through it has two failure modes, and the
/// quiet one is the dangerous one: `Uint8` refuses `200 + 200 = 400` outright, so
/// the upgrade fails on exactly the non-canonical input it exists to repair,
/// while `Float16` is unchecked — `f16::from_f32` maps anything past 65504 to
/// **infinity** and the op reports success.
///
/// So widen to the narrowest encoding that actually holds the result. Integers
/// stay integers (canonicalizing sums counts; it never makes them fractional)
/// and floats stay floats. Returns the source encoding unchanged whenever it
/// still fits, which is the overwhelmingly common case — this only widens for a
/// shard canonicalization actually rewrote.
pub fn encoding_for_canonicalized(source: ValueEncoding, data: &[f32]) -> ValueEncoding {
    let max = data.iter().copied().fold(0.0f32, f32::max);
    match source {
        // f16's finite ceiling. Past it `from_f32` yields inf, silently.
        ValueEncoding::Float16 if max > 65504.0 => ValueEncoding::Float32,
        ValueEncoding::Uint8 if max > 255.0 => {
            encoding_for_canonicalized(ValueEncoding::Uint16, data)
        }
        ValueEncoding::Uint16 if max > 65535.0 => ValueEncoding::Uint32,
        // Uint32 saturates the integer ladder; a sum past u32::MAX would need
        // f32 and is not reachable from counts that fit u32 in a real matrix.
        other => other,
    }
}

/// Section families this file's copy helpers do **not** carry, checked against
/// the input so the loss can be reported rather than discovered later.
///
/// The list mirrors what `copy_auxiliary_sections` actually copies; keep the
/// two in step. Bitmaps and `.raw` are recoverable by re-running the op that
/// built them, but `varm` / `obsp` / `varp` are user data with no rebuild path.
const DROPPED_SECTION_FAMILIES: &[(SectionType, &str)] = &[
    (
        SectionType::BitmapShard,
        "detection bitmaps (rebuild: --bitmap)",
    ),
    (SectionType::VarmEmbedding, "varm"),
    (SectionType::VarmEmbeddingShard, "varm"),
    (SectionType::ObspCsrShard, "obsp"),
    (SectionType::ObspEmbedding, "obsp"),
    (SectionType::ObspEmbeddingShard, "obsp"),
    (SectionType::VarpEmbedding, "varp"),
    (SectionType::VarpEmbeddingShard, "varp"),
    (SectionType::RawCsrShard, "adata.raw"),
    (SectionType::RawVarMetadata, "adata.raw"),
    (SectionType::GroupIndex, "grouped-sort group index"),
    (
        SectionType::LayerCscShard,
        "layer CSC sidecars (rebuild: scx build-csc)",
    ),
];

/// Warn, once per family, about input sections this rewrite is about to drop.
///
/// `copy_auxiliary_sections` is an allowlist, so anything it does not name is
/// dropped — silently, until now. Both its callers rename a wholly new file
/// over the target with no prior catalog, so `scx rollback` cannot recover
/// what goes missing; a user who is not told loses `varm` / `obsp` / `varp`
/// with no way back and no record that it happened.
fn warn_dropped_sections(reader: &ScxReader, action: &str) {
    let dropped = dropped_section_labels(reader);
    if !dropped.is_empty() {
        // No rollback clause: this helper does not know whether the caller is
        // writing to a separate output (where the input is untouched) or
        // renaming over it. Stating the loss and the remedy is true of both;
        // claiming irreversibility on the copy-out form would be the same wrong
        // rationale for a right warning that `run_upgrade`'s decline message
        // had. The in-place hazard is documented in docs/operations.md.
        log::warn!(
            "scx {action}: the output will not carry {} — this rewrite copies only \
             layers, obsm, uns, predicate indexes and deletion vectors. Copy them \
             across from the input if you need them, or use `scx optimize`, which \
             carries obsm / varm / obsp / varp.",
            dropped.join(", ")
        );
    }
}

/// The distinct family labels this rewrite would drop from `reader`, in
/// [`DROPPED_SECTION_FAMILIES`] order. Split out from [`warn_dropped_sections`]
/// so the set can be asserted directly — a warning is only worth documenting if
/// it names the right things.
pub(crate) fn dropped_section_labels(reader: &ScxReader) -> Vec<&'static str> {
    let mut seen: Vec<&'static str> = Vec::new();
    for entry in &reader.catalog().entries {
        if let Some((_, label)) = DROPPED_SECTION_FAMILIES
            .iter()
            .find(|(ty, _)| *ty == entry.section_type)
        {
            if !seen.contains(label) {
                seen.push(label);
            }
        }
    }
    seen
}

/// Copy all auxiliary sections (layers, obsm, uns, predicate indices, deletion
/// vectors) from reader to writer, then append a new provenance entry.
///
/// **Only valid for a rewrite that preserves the global obs row space 1:1** —
/// its two callers, `build_csc` and `scx upgrade`, both do. The deletion-vector
/// carry below is what makes that a requirement: v2 vectors store global obs row
/// indices, so they stay valid exactly as long as the row order does. An op that
/// drops or reorders rows must apply the mask instead (see `compact` / `sort`).
///
/// Still **not** carried by this helper, and therefore dropped by both callers:
/// detection bitmaps, `varm`/`obsp`/`varp`, `adata.raw`, and the grouped-sort
/// group index. The list is an allowlist, so a section type added to the format
/// is dropped here silently until someone adds it — that is the shape of the
/// bug this carry was written to fix. A drop is unrecoverable on the in-place
/// form of either caller (a rename over the target, carrying no prior catalog),
/// so [`warn_dropped_sections`] reports it rather than leaving the user to
/// discover it: see [`DROPPED_SECTION_FAMILIES`], which must be kept in step
/// with what is actually copied below.
///
/// Layers are re-emitted as they are. For the canonicalizing variant — which
/// `scx upgrade` needs and `build_csc` must not have — see
/// [`copy_auxiliary_sections_canonicalizing`].
pub fn copy_auxiliary_sections(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    action: &str,
    params_json: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    copy_auxiliary_sections_canonicalizing(reader, writer, action, params_json, false)
}

/// [`copy_auxiliary_sections`] with control over layer canonicalization.
///
/// `canonicalize` re-sorts, dedup-sums and zero-drops every layer CSR shard
/// before re-encoding, widening the value encoding if the sums need it
/// ([`encoding_for_canonicalized`]). The two callers legitimately differ:
/// `scx upgrade` stamps the output `DEFAULT_WRITE_FORMAT_VERSION`, whose
/// contract *is* canonical CSR, so it must canonicalize what it re-emits;
/// `build_csc` deliberately clamps its output version to the source's
/// (SCX-005) precisely so it does **not** have to, and passing `true` there
/// would change the nnz of a file it promises to re-emit unchanged.
///
/// Split out rather than added as a fifth parameter to the existing function:
/// `canonicalize` is wanted by exactly one of the two callers, and widening a
/// published signature to say so would source-break every downstream caller for
/// a choice none of them are making. `false` is what the four-argument form has
/// always done.
pub fn copy_auxiliary_sections_canonicalizing(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    action: &str,
    params_json: &str,
    canonicalize: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    warn_dropped_sections(reader, action);
    copy_layers(reader, writer, canonicalize)?;
    copy_obsm(reader, writer)?;
    copy_uns(reader, writer)?;
    copy_predicate_indices(reader, writer)?;
    copy_deletion_vectors(reader, writer)?;
    append_provenance(reader, writer, action, params_json)?;
    Ok(())
}

/// Carry the deletion-vector section through a 1:1 rewrite.
///
/// Without this the rows come back. The output header is built by copying the
/// input's flags, so the `has_deletion_vectors` bit looks preserved — but
/// `FileHeader::sync_from_catalog` re-derives every flag from the sections that
/// were actually written, finds no deletion-vector section, and clears it. The
/// result is a file that has silently forgotten which cells were deleted, with
/// no dangling flag to notice it by.
fn copy_deletion_vectors(
    reader: &ScxReader,
    writer: &mut ScxWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(dv) = reader.read_deletion_vectors()? {
        writer.write_deletion_vectors(&dv)?;
    }
    Ok(())
}

/// Copy all layer CSR shards from reader to writer, preserving per-shard codec.
///
/// `canonicalize` sorts each row's column indices, sums duplicate coordinates
/// and drops explicit zeros before re-encoding — the v3 canonical-CSR contract.
/// See [`copy_auxiliary_sections`] for why it is the caller's choice.
fn copy_layers(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    canonicalize: bool,
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

            let (indptr, indices, mut data) = reader.read_shard_from_entry(entry)?;
            let row_start = entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
            let mut indptr_u64: Vec<u64> = indptr.iter().map(|&v| v as u64).collect();
            let mut indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
            // Canonicalizing sums duplicates, which can exceed what the source
            // encoding holds — see `encoding_for_canonicalized`.
            let ve = if canonicalize {
                scx_sparse::canonicalize_csr(&mut indptr_u64, &mut indices_u32, &mut data);
                encoding_for_canonicalized(ve, &data)
            } else {
                ve
            };
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

    /// `copy_auxiliary_sections` is an allowlist, so what it does not name is
    /// dropped — and both its callers rename over the target with no prior
    /// catalog, so the drop cannot be rolled back. The warning is the only
    /// notice a user gets, which makes "does it name the right families?" worth
    /// asserting rather than assuming: a stale [`DROPPED_SECTION_FAMILIES`]
    /// produces a *confidently wrong* warning, which is worse than none.
    #[test]
    fn dropped_families_are_detected_on_a_file_that_has_them() {
        use crate::test_utils::{sample_header, sample_obs, sample_var};
        use arrow::array::{Float32Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("with_varm.scx");
        let (n_obs, n_vars) = (4usize, 3usize);
        let mut w = ScxWriter::new(&path, sample_header(n_obs as u64, n_vars as u64)).unwrap();
        w.write_obs(&sample_obs(n_obs)).unwrap();
        w.write_var(&sample_var(n_vars)).unwrap();
        let varm = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "pc1",
                DataType::Float32,
                false,
            )])),
            vec![Arc::new(Float32Array::from(vec![0.1f32, 0.2, 0.3]))],
        )
        .unwrap();
        w.write_varm("loadings", &varm).unwrap();
        w.write_csr_shard(
            &vec![0u64; n_obs + 1],
            &[],
            &[],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
        w.finish().unwrap();

        let reader = ScxReader::open(&path).unwrap();
        assert_eq!(
            dropped_section_labels(&reader),
            vec!["varm"],
            "a file carrying varm must be reported as losing varm, and nothing else"
        );

        // And a file with none of them must warn about nothing — otherwise the
        // warning fires on every ordinary upgrade and stops being read.
        let plain = dir.path().join("plain.scx");
        let mut w = ScxWriter::new(&plain, sample_header(n_obs as u64, n_vars as u64)).unwrap();
        w.write_obs(&sample_obs(n_obs)).unwrap();
        w.write_var(&sample_var(n_vars)).unwrap();
        w.write_csr_shard(
            &vec![0u64; n_obs + 1],
            &[],
            &[],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
        w.finish().unwrap();
        assert!(dropped_section_labels(&ScxReader::open(&plain).unwrap()).is_empty());
    }
}
