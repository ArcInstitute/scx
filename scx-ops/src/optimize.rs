//! `scx optimize` — in-place upgrade of an existing single-modality SCX file:
//! re-encode every CSR shard (canonicalizing it) so the output carries decode
//! sidecars and legitimately claims the v3 canonical-CSR invariant, while
//! preserving the row layout, obs/var, obsm/varm/obsp/varp, uns, and predicate
//! indexes.
//!
//! Unlike `compact` (which targets deletion reclaim + re-sharding and keeps the
//! source `format_version`), `optimize` is a near-faithful upgrade: it does not
//! apply deletions (the deletion-vector section is carried through unchanged),
//! does not change CSR shard boundaries, and canonicalizes each shard so the
//! output is a real v3 file with sidecars. The one optional layout change is
//! obs-metadata: `ObsShardPolicy` may migrate a legacy single-section obs to the
//! sharded layout (row order/content preserved exactly). The CSC sidecar is
//! dropped (rerun `scx build-csc` / `--rebuild-csc`); a `decode/*` sidecar is
//! emitted for every Scx1 integer shard automatically by the encoder.

use std::path::Path;

use scx_codec::CodecId;
use scx_format_io::encoder::{encode_one_shard, FramingConfig};
use scx_format_io::header::{FileHeader, CURRENT_FORMAT_VERSION};
use scx_format_io::modality::ModalityType;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;
use scx_format_io::{ObsShardPolicy, ScxReader, DEFAULT_SHARD_TARGET_ROWS};
use scx_sparse::canonicalize_csr;

use crate::error::{OpsError, Result};
use crate::rewrite_helpers::{append_provenance, copy_predicate_indices};

/// Summary of what an [`optimize_with_framing`] pass did, for CLI reporting.
#[derive(Debug, Clone, Copy, Default)]
pub struct OptimizeStats {
    /// Total CSR-class shards (X + layers + obsp-CSR) re-encoded.
    pub shards_total: usize,
    /// Of those, how many were emitted row-group-framed (shard v2).
    pub shards_framed: usize,
    /// Of the framed shards, how many the encoder stored as `ShufDeltaZstd`
    /// (the `compact-trial` per-shard winner, or an explicit `--codec shufdelta`).
    pub shards_shufdelta: usize,
    /// The file `format_version` stamped on the output (3 unframed, 4 framed).
    pub format_version: u16,
}

/// Re-encode + canonicalize every CSR shard of `input_path` into `output_path`,
/// emitting decode sidecars and stamping `format_version = 3`. Single-modality
/// only (multimodal files should use `scx compact`).
///
/// `codec` selects the per-shard codec passed to the encoder: `None` keeps the
/// auto-codec (Scx1 for low-median integer counts, else Zstd), while
/// `Some(CodecId::Scx1)` forces Scx1 on every integer shard — guaranteeing a
/// `decode/*` sidecar (and thus the `to_gpu_anndata` device-decode route) even
/// for high-median shards that auto would route to Zstd. Non-integer shards
/// fall back to Zstd regardless (handled inside `encode_one_shard`).
///
/// `obs_shard_policy` controls whether a *single-section* legacy obs table is
/// migrated to the sharded `ObsMetadataShard` layout: `Auto` (default) shards
/// only when `n_obs > shard_target_rows` (the `from_anndata` threshold),
/// `Always` always shards, `Off` keeps the single section (the historical 1:1
/// behaviour). An already-sharded obs is always stream-preserved, regardless
/// of policy.
pub fn optimize(
    input_path: &Path,
    output_path: &Path,
    codec: Option<CodecId>,
    obs_shard_policy: ObsShardPolicy,
) -> Result<()> {
    optimize_with_framing(input_path, output_path, codec, obs_shard_policy, None)?;
    Ok(())
}

/// Like [`optimize`], but additionally row-group-frames every re-encoded CSR
/// shard when `framing` is `Some` (F5-b), producing a `format_version = 4`
/// file with a multi-entry `BlockIndex` per shard for codec-agnostic sub-shard
/// random access. When `framing` is `None` this is byte-identical to
/// [`optimize`] (v3, unframed). Returns an [`OptimizeStats`] describing the
/// per-shard codec/framing outcome (used by the CLI summary).
///
/// `framing.trial` (surfaced as `scx optimize --codec compact-trial`) picks the
/// smaller of {heuristic winner, ShufDeltaZstd} per shard; a framed shard
/// forgoes the Scx1 GPU/per-row decode sidecar (it uses the block index
/// instead), so `compact-trial` optimizes for size + random access.
pub fn optimize_with_framing(
    input_path: &Path,
    output_path: &Path,
    codec: Option<CodecId>,
    obs_shard_policy: ObsShardPolicy,
    framing: Option<FramingConfig>,
) -> Result<OptimizeStats> {
    let reader = ScxReader::open(input_path)?;
    if reader.is_multimodal() {
        return Err(OpsError::InvalidInput(
            "scx optimize does not support multimodal files yet; use `scx compact`".into(),
        ));
    }

    // Framing produces a v4/shard-v2 file; unframed stays v3/shard-v1.
    let framed = matches!(framing, Some(fc) if fc.row_group_rows > 0);

    let in_header = reader.header().clone();
    let index_dtype = in_header.index_dtype;

    // The CSC sidecar (bit 0) is dropped — re-canonicalizing may change nnz and
    // would leave the column-major sidecar referencing stale offsets. The
    // deletion-vector flag (bit 5) is cleared here and re-set below only if we
    // actually carry the `DeletionVectors` section through (otherwise the header
    // would claim deletions with no section, and the logically-deleted rows
    // would silently reappear).
    let had_csc = in_header.has_csc();
    if had_csc {
        log::warn!(
            "scx optimize dropped CSC shards from {}: rerun `scx build-csc` \
             to restore the column-major sidecar",
            input_path.display()
        );
    }
    let out_flags = in_header.flags & !(1 << 0) & !(1 << 5);

    // Resolve a malformed `shard_target_rows == 0` to the default up front and
    // stamp the resolved value into the output header, so the header agrees
    // with the obs reshape sizing below (an `Always` reshape on a `0`-header
    // input would otherwise emit default-sized shards while the header still
    // claimed `0`). Non-zero headers are preserved exactly.
    let shard_target_rows = if in_header.shard_target_rows == 0 {
        DEFAULT_SHARD_TARGET_ROWS
    } else {
        in_header.shard_target_rows
    };

    // We canonicalize every shard below, so the default v3 invariant is real.
    // Framing bumps the file to v4 (the multi-entry BlockIndex layout); unframed
    // output keeps the `Default` (v3) stamp.
    let out_header = FileHeader {
        flags: out_flags,
        n_obs: in_header.n_obs,
        n_vars: in_header.n_vars,
        shard_target_rows,
        // File-level codec hint for `scx info`. When the caller forces a codec,
        // reflect it; otherwise preserve the source hint (the real per-shard
        // codec is auto-selected by `encode_one_shard`).
        codec_id: codec.map(|c| c as u8).unwrap_or(in_header.codec_id),
        index_dtype,
        format_version: if framed {
            CURRENT_FORMAT_VERSION
        } else {
            FileHeader::default().format_version
        },
        ..Default::default()
    };

    let out_format_version = out_header.format_version;
    let mut writer = ScxWriter::new(output_path, out_header)?
        .with_data_generation(reader.catalog().data_generation + 1);

    // obs / var pass through unchanged (rows are preserved 1:1). Stream
    // shard-by-shard when the input is already sharded so the row-sharded
    // layout survives and peak memory stays at one shard: `read_obs()` would
    // assemble every `ObsMetadataShard` into one in-memory batch (atlas-scale
    // OOM) and `write_obs()` would collapse it back to a single legacy section,
    // breaking the "faithful 1:1 upgrade" contract. Legacy single-section input
    // (`*_metadata_shard_count() == 0`) has no per-shard reader and falls
    // through to the materialising path. Both counts are pure catalog scans.
    // Tracks whether a single-section obs was migrated to the sharded layout,
    // for the info log and provenance record below. `false` for already-sharded
    // input (preserved as-is) and for `Off`/sub-threshold single-section input.
    let mut obs_resharded = false;
    if reader.obs_metadata_shard_count() > 0 {
        crate::compact::write_obs_shards_streaming(
            &reader,
            &mut writer,
            None,
            in_header.n_obs as usize,
        )?;
    } else {
        // Single legacy `ObsMetadata` section. `read_obs()` materializes it
        // whole regardless (one Arrow IPC section is one batch), so there is no
        // optimize-side memory win from sharding — but emitting it as shards
        // gives downstream streaming/cloud/bounded-memory readers the bounded
        // layout.
        let reshape =
            obs_shard_policy.should_shard_single_section(in_header.n_obs, shard_target_rows);
        if reshape {
            let n_shards = in_header.n_obs.div_ceil(shard_target_rows as u64);
            log::info!(
                "scx optimize: migrating single-section obs ({} rows) to {} sharded \
                 section(s) (shard_target_rows={}, policy={:?})",
                in_header.n_obs,
                n_shards,
                shard_target_rows,
                obs_shard_policy,
            );
            obs_resharded = true;
        }
        crate::compact::write_obs_section(
            &mut writer,
            &reader.read_obs()?,
            reshape,
            shard_target_rows,
        )?;
    }
    if reader.var_metadata_shard_count() > 0 {
        write_var_shards_streaming(&reader, &mut writer, in_header.n_vars)?;
    } else {
        writer.write_var(&reader.read_var()?)?;
    }

    // Re-encode every CSR-backed shard (X first, then layers, then obs×obs
    // pairwise graphs) preserving its name and global row offset, canonicalizing
    // the triplet so the output is canonical and the encoder emits a decode
    // sidecar for Scx1 integer shards. `ObspCsrShard` is X-class CSR (minor axis
    // is `n_obs`, not `n_vars`), so it is re-encoded here too — dropping it would
    // silently lose the pairwise graph.
    let mut x_entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CsrShard)
        .collect();
    x_entries.sort_by_key(|e| e.stats.as_ref().map(|s| s.row_start).unwrap_or(0));
    let mut layer_entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::LayerCsrShard)
        .collect();
    layer_entries.sort_by(|a, b| a.name.cmp(&b.name));
    let mut obsp_csr_entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::ObspCsrShard)
        .collect();
    obsp_csr_entries.sort_by(|a, b| a.name.cmp(&b.name));

    let mut stats = OptimizeStats {
        format_version: out_format_version,
        ..Default::default()
    };
    for entry in x_entries
        .into_iter()
        .chain(layer_entries)
        .chain(obsp_csr_entries)
    {
        let row_start = entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
        // Use the shard's own minor dimension rather than the file-level
        // `n_vars`: correct for `ObspCsrShard` (minor axis = `n_obs`) and robust
        // against any per-shard width difference.
        let n_minor = reader.read_shard_header(entry)?.n_minor;
        let (indptr_i64, indices_i32, mut values) = reader.read_shard_from_entry(entry)?;
        let mut indptr: Vec<u64> = indptr_i64.iter().map(|&v| v as u64).collect();
        let mut indices: Vec<u32> = indices_i32.iter().map(|&v| v as u32).collect();
        canonicalize_csr(&mut indptr, &mut indices, &mut values);

        let pre = encode_one_shard(
            &indptr,
            &indices,
            &values,
            codec, // None = auto-codec; Some(Scx1) forces sidecars on every integer shard
            index_dtype,
            n_minor,
            row_start,
            entry.section_type,
            ModalityType::Rna,
            entry.name.clone(),
            framing, // Some => row-group-frame this shard (F5-b, shard v2)
        )?;
        stats.shards_total += 1;
        if pre.shard_format_version() > scx_format_io::DEFAULT_WRITE_SHARD_FORMAT_VERSION {
            stats.shards_framed += 1;
            if pre.codec_id() == CodecId::ShufDeltaZstd as u8 {
                stats.shards_shufdelta += 1;
            }
        }
        writer.write_preencoded_shard(pre)?;
    }

    // Auxiliary matrices (obsm/varm + COO obsp/varp) are unchanged by optimize,
    // so they are copied byte-for-byte. Verbatim copy preserves any sharded
    // layout (e.g. `ObsmEmbeddingShard`) and avoids decoding large `obsp`/`varp`
    // graphs into one section (OOM / the 2 GB Arrow IPC ceiling). `ObspCsrShard`
    // is excluded — it is the CSR-backed obsp graph, re-encoded in the loop
    // above. Sorted by name for deterministic output.
    let mut aux_entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| {
            matches!(
                e.section_type,
                SectionType::ObsmEmbedding
                    | SectionType::ObsmEmbeddingShard
                    | SectionType::VarmEmbedding
                    | SectionType::VarmEmbeddingShard
                    | SectionType::ObspEmbedding
                    | SectionType::ObspEmbeddingShard
                    | SectionType::VarpEmbedding
                    | SectionType::VarpEmbeddingShard
            )
        })
        .collect();
    aux_entries.sort_by(|a, b| a.name.cmp(&b.name));
    for entry in aux_entries {
        let bytes = reader.section_bytes(entry)?;
        writer.copy_section_verbatim(entry, bytes)?;
    }
    if let Ok(uns) = reader.read_uns() {
        writer.write_uns(&uns)?;
    }

    // Deletion vectors are carried through unchanged (optimize does not apply
    // them — that is `compact`'s job). The section references obs row ranges /
    // shard ids, both preserved here, so it stays valid; `write_deletion_vectors`
    // re-sets the header flag (cleared above) so it is set iff the section exists.
    if let Some(dv) = reader.read_deletion_vectors()? {
        writer.write_deletion_vectors(&dv)?;
    }

    // Predicate indexes reference obs row ranges; rows + shard boundaries are
    // preserved, so they stay valid — copy through. Then record provenance.
    copy_predicate_indices(&reader, &mut writer)
        .map_err(|e| OpsError::InvalidInput(format!("copy predicate indices: {e}")))?;
    // Record the obs-sharding decision so the layout change is auditable via
    // `scx info --history` (the obs layout is the one thing optimize can now
    // change beyond the CSR re-encode).
    let shard_obs_label = match obs_shard_policy {
        ObsShardPolicy::Off => "off",
        ObsShardPolicy::Auto => "auto",
        ObsShardPolicy::Always => "always",
    };
    let params = format!(
        "{{\"canonicalize\":true,\"sidecars\":true,\"shard_obs\":\"{shard_obs_label}\",\"obs_resharded\":{obs_resharded}}}"
    );
    append_provenance(&reader, &mut writer, "optimize", &params)
        .map_err(|e| OpsError::InvalidInput(format!("append provenance: {e}")))?;

    writer.finish()?;
    Ok(stats)
}

/// Stream var shards from `reader` straight to sharded output, one output shard
/// per non-empty input shard. Peak memory is one shard — `read_var()` (which
/// assembles every `VarMetadataShard` into one batch) is never called. Mirror of
/// compact's [`write_obs_shards_streaming`](crate::compact::write_obs_shards_streaming)
/// for the var axis; optimize applies no deletions, so (unlike the obs helper)
/// it takes no `keep_mask` and renumbers output shards over the non-empty inputs.
fn write_var_shards_streaming(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    n_vars_total: u64,
) -> Result<()> {
    let mut out_idx = 0u32;
    let mut row_start = 0u64;
    for res in reader.var_shards() {
        let batch = res?;
        let n = batch.num_rows() as u64;
        if n == 0 {
            continue;
        }
        writer.write_var_shard(out_idx, row_start, n, n_vars_total, &batch)?;
        out_idx += 1;
        row_start += n;
    }
    if out_idx == 0 {
        // Every input var shard was empty (degenerate `n_vars == 0` file with a
        // sharded var layout). Mirror `write_obs_shards_streaming`'s empty-section
        // fallback: emit one empty legacy var section so the output stays
        // well-formed instead of carrying no var section at all (which the old
        // `write_var(read_var())` path would also have written). Footer-only
        // schema read — no batch decode.
        let schema = std::sync::Arc::new(reader.read_var_schema_physical()?);
        writer.write_var(&arrow::array::RecordBatch::new_empty(schema))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{sample_header, sample_obs, sample_var};
    use scx_codec::{CodecId, ValueEncoding};

    #[test]
    fn optimize_adds_sidecars_canonicalizes_and_upgrades_to_v3() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        let output = dir.path().join("out.scx");

        // v2 input with a `None`-codec CSR shard (no decode sidecar). Rows are
        // dense (512 nnz) with moderate column gaps so the re-encoded Scx1
        // sidecar comfortably fits the 25% overhead budget; row 0 has ONE
        // duplicate column (non-canonical). u32 indices (n_vars > 65535).
        let n_obs = 40usize;
        let nnz_per_row = 512usize;
        let n_vars = 200_000usize;
        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 2;
        header.index_dtype = 1; // u32 indices

        let mut indptr = vec![0u64];
        let mut indices: Vec<u32> = Vec::new();
        let mut values: Vec<u8> = Vec::new();
        for r in 0..n_obs {
            let mut col = 0u32;
            for k in 0..nnz_per_row {
                if r == 0 && k == 1 {
                    // Duplicate the previous column → one non-canonical entry.
                    indices.push(*indices.last().unwrap());
                } else {
                    col += 1 + ((r * 13 + k * 7) % 250) as u32; // gaps → frame_bits ~8
                    indices.push(col);
                }
                values.push(1u8 + ((r + k) % 5) as u8); // small, non-zero → Scx1
            }
            indptr.push(indices.len() as u64);
        }
        let input_nnz = indices.len();
        {
            let mut w = ScxWriter::new(&input, header).unwrap();
            w.write_obs(&sample_obs(n_obs)).unwrap();
            w.write_var(&sample_var(n_vars)).unwrap();
            w.write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
            w.finish().unwrap();
        }

        // Input carries no decode sidecar (None codec).
        let in_reader = ScxReader::open(&input).unwrap();
        assert!(!in_reader
            .catalog()
            .entries
            .iter()
            .any(|e| e.section_type == SectionType::DecodeMetadataShard));

        optimize(&input, &output, None, ObsShardPolicy::Off).unwrap();

        let out = ScxReader::open(&output).unwrap();
        assert_eq!(out.header().format_version, 3, "optimize stamps v3");

        // A decode sidecar is emitted for the re-encoded (now Scx1) shard.
        let sidecars: Vec<_> = out
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::DecodeMetadataShard)
            .cloned()
            .collect();
        assert_eq!(sidecars.len(), 1, "decode sidecar added");
        // Sidecar decode-parity + source identity (the deep-validate check).
        for e in &sidecars {
            out.validate_decode_sidecar_entry(e).unwrap();
        }
        // Every CSR shard is canonical post-optimize.
        assert!(out
            .validate_canonical_csr_shards()
            .iter()
            .all(|(_, ok)| *ok));

        // Canonicalization summed the single duplicate → nnz drops by exactly 1.
        let (out_indptr, out_indices, _values) = out.read_csr_shard(0).unwrap();
        assert_eq!(out_indices.len(), input_nnz - 1);
        assert_eq!(*out_indptr.last().unwrap() as usize, input_nnz - 1);
        assert_eq!(out_indptr.len(), n_obs + 1);

        // obs/var preserved.
        assert_eq!(out.read_obs().unwrap().num_rows(), n_obs);
        assert_eq!(out.read_var().unwrap().num_rows(), n_vars);
    }

    /// T3.2: `optimize_with_framing` (the `scx optimize --codec compact-trial
    /// --row-group-rows N` path) upgrades a v2/v3 file to a v4/shard-v2 framed
    /// file, and the framed re-encode round-trips byte-identically.
    #[test]
    fn optimize_with_framing_upgrades_to_v4_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        let output = dir.path().join("out.scx");

        // Canonical v2 input (strictly increasing columns, no duplicates) so the
        // framed re-encode is byte-identical on read-back.
        let n_obs = 40usize;
        let nnz_per_row = 64usize;
        let n_vars = 200_000usize;
        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 2;
        header.index_dtype = 1; // u32 indices

        let mut indptr = vec![0u64];
        let mut indices: Vec<u32> = Vec::new();
        let mut values: Vec<u8> = Vec::new();
        for r in 0..n_obs {
            let mut col = 0u32;
            for k in 0..nnz_per_row {
                col += 1 + ((r * 13 + k * 7) % 250) as u32;
                indices.push(col);
                values.push(1u8 + ((r + k) % 5) as u8);
            }
            indptr.push(indices.len() as u64);
        }
        {
            let mut w = ScxWriter::new(&input, header).unwrap();
            w.write_obs(&sample_obs(n_obs)).unwrap();
            w.write_var(&sample_var(n_vars)).unwrap();
            w.write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
            w.finish().unwrap();
        }
        let (in_ip, in_ix, in_v) = ScxReader::open(&input).unwrap().read_csr_shard(0).unwrap();

        let stats = optimize_with_framing(
            &input,
            &output,
            None,
            ObsShardPolicy::Off,
            Some(FramingConfig {
                row_group_rows: 8,
                target_nnz: None,
                trial: true,
            }),
        )
        .unwrap();
        assert_eq!(stats.format_version, 4);
        assert!(stats.shards_framed >= 1, "at least one shard framed");
        assert_eq!(
            stats.shards_total, stats.shards_framed,
            "framing frames every re-encoded shard"
        );

        let out = ScxReader::open(&output).unwrap();
        assert_eq!(out.header().format_version, 4, "framed optimize stamps v4");
        let entry = out.catalog().csr_shards_sorted()[0];
        assert_eq!(
            out.read_shard_header(entry).unwrap().shard_format_version,
            scx_format_io::CURRENT_SHARD_FORMAT_VERSION,
            "framed shard is v2",
        );
        // Full decode is byte-identical to the (already-canonical) input.
        let (out_ip, out_ix, out_v) = out.read_csr_shard(0).unwrap();
        assert_eq!(out_ip, in_ip);
        assert_eq!(out_ix, in_ix);
        assert_eq!(out_v, in_v);
        assert_eq!(out.read_obs().unwrap().num_rows(), n_obs);
    }

    #[test]
    fn optimize_codec_scx1_forces_sidecar_on_high_median_shard() {
        // A high-median integer shard (all counts == 100) routes to Zstd under
        // the auto-codec → NO decode sidecar. `Some(Scx1)` must force Scx1 and
        // emit the sidecar, which `to_gpu_anndata`'s device-decode route needs.
        // Dense rows (512 nnz, moderate gaps, 200k vars) so the Scx1 decode
        // sidecar fits the 25% overhead budget — matching the dims the
        // auto-codec test uses; only the value magnitude differs.
        let n_obs = 40usize;
        let nnz_per_row = 512usize;
        let n_vars = 200_000usize;

        let build_input = |path: &Path| {
            let mut header = sample_header(n_obs as u64, n_vars as u64);
            header.format_version = 2;
            header.index_dtype = 1; // u32 indices
            let mut indptr = vec![0u64];
            let mut indices: Vec<u32> = Vec::new();
            let mut values: Vec<u8> = Vec::new();
            for r in 0..n_obs {
                let mut col = 0u32;
                for k in 0..nnz_per_row {
                    col += 1 + ((r * 13 + k * 7) % 250) as u32;
                    indices.push(col);
                    values.push(100u8); // median == 100 (> 8) → auto picks Zstd
                }
                indptr.push(indices.len() as u64);
            }
            let mut w = ScxWriter::new(path, header).unwrap();
            w.write_obs(&sample_obs(n_obs)).unwrap();
            w.write_var(&sample_var(n_vars)).unwrap();
            w.write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
            w.finish().unwrap();
        };

        let count_sidecars = |path: &Path| {
            ScxReader::open(path)
                .unwrap()
                .catalog()
                .entries
                .iter()
                .filter(|e| e.section_type == SectionType::DecodeMetadataShard)
                .count()
        };

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        build_input(&input);

        // Auto-codec: high-median shard → Zstd → no sidecar.
        let auto_out = dir.path().join("auto.scx");
        optimize(&input, &auto_out, None, ObsShardPolicy::Off).unwrap();
        assert_eq!(
            count_sidecars(&auto_out),
            0,
            "auto-codec leaves the high-median shard sidecar-less"
        );

        // Forced Scx1: sidecar present + decode-parity holds.
        let scx1_out = dir.path().join("scx1.scx");
        optimize(&input, &scx1_out, Some(CodecId::Scx1), ObsShardPolicy::Off).unwrap();
        let out = ScxReader::open(&scx1_out).unwrap();
        let sidecars: Vec<_> = out
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::DecodeMetadataShard)
            .cloned()
            .collect();
        assert_eq!(sidecars.len(), 1, "forced Scx1 emits a decode sidecar");
        for e in &sidecars {
            out.validate_decode_sidecar_entry(e).unwrap();
        }
        // The decoded matrix is unchanged by the codec choice.
        let (a_indptr, a_indices, a_values) = ScxReader::open(&auto_out)
            .unwrap()
            .read_csr_shard(0)
            .unwrap();
        let (s_indptr, s_indices, s_values) = out.read_csr_shard(0).unwrap();
        assert_eq!(a_indptr, s_indptr);
        assert_eq!(a_indices, s_indices);
        assert_eq!(a_values, s_values);
    }

    /// Build a small canonical Scx1-eligible CSR triplet (strictly-increasing
    /// per-row indices, small positive counts) with `n_obs` rows.
    fn small_csr(n_obs: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
        let mut indptr = vec![0u64];
        let mut indices: Vec<u32> = Vec::new();
        let mut values: Vec<u8> = Vec::new();
        for r in 0..n_obs {
            let mut col = 0u32;
            for k in 0..16usize {
                col += 1 + ((r + k) % 7) as u32;
                indices.push(col);
                values.push(1u8 + ((r + k) % 4) as u8);
            }
            indptr.push(indices.len() as u64);
        }
        (indptr, indices, values)
    }

    #[test]
    fn optimize_preserves_deletion_vectors() {
        use roaring::RoaringBitmap;
        use scx_format_io::deletion_vectors::DeletionVectors;

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        let output = dir.path().join("out.scx");

        let n_obs = 8usize;
        let n_vars = 1000usize;
        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 2;
        header.index_dtype = 1;
        let (indptr, indices, values) = small_csr(n_obs);
        {
            let mut w = ScxWriter::new(&input, header).unwrap();
            w.write_obs(&sample_obs(n_obs)).unwrap();
            w.write_var(&sample_var(n_vars)).unwrap();
            w.write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
            // Mark local row 1 of shard 0 as logically deleted.
            let mut dv = DeletionVectors::new();
            let mut bm = RoaringBitmap::new();
            bm.insert(1);
            dv.insert(0, bm);
            w.write_deletion_vectors(&dv).unwrap();
            w.finish().unwrap();
        }

        optimize(&input, &output, None, ObsShardPolicy::Off).unwrap();

        let out = ScxReader::open(&output).unwrap();
        assert_eq!(out.header().format_version, 3);
        // The deletion-vector section is carried through (flag re-set + section
        // present), not dropped — so the deleted row is not silently resurrected.
        assert!(
            out.header().has_deletion_vectors(),
            "deletion-vector flag preserved"
        );
        let dv = out
            .read_deletion_vectors()
            .unwrap()
            .expect("deletion-vector section present after optimize");
        assert!(dv.is_deleted(0, 1));
        assert_eq!(dv.total_deleted(), 1);
        // optimize does NOT apply deletions — rows stay (logical deletion).
        assert_eq!(out.read_obs().unwrap().num_rows(), n_obs);
        let (out_indptr, _, _) = out.read_csr_shard(0).unwrap();
        assert_eq!(out_indptr.len(), n_obs + 1);
    }

    #[test]
    fn optimize_preserves_and_canonicalizes_obsp_csr_shard() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        let output = dir.path().join("out.scx");

        let n_obs = 8usize;
        let n_vars = 1000usize;
        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 2;
        header.index_dtype = 1;
        let (indptr, indices, values) = small_csr(n_obs);
        // Canonical obs×obs graph: each row links to two distinct, sorted
        // neighbors (indices < n_obs).
        let mut op_indptr = vec![0u64];
        let mut op_indices: Vec<u32> = Vec::new();
        let mut op_values: Vec<u8> = Vec::new();
        for r in 0..n_obs {
            let a = ((r + 1) % n_obs) as u32;
            let b = ((r + 3) % n_obs) as u32;
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            op_indices.push(lo);
            op_values.push(1u8);
            op_indices.push(hi);
            op_values.push(1u8);
            op_indptr.push(op_indices.len() as u64);
        }
        {
            let mut w = ScxWriter::new(&input, header).unwrap();
            w.write_obs(&sample_obs(n_obs)).unwrap();
            w.write_var(&sample_var(n_vars)).unwrap();
            w.write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
            w.write_obsp_shard(
                &op_indptr,
                &op_indices,
                &op_values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
                "connectivities",
                0,
            )
            .unwrap();
            w.finish().unwrap();
        }

        optimize(&input, &output, None, ObsShardPolicy::Off).unwrap();

        let out = ScxReader::open(&output).unwrap();
        // The CSR-backed obsp graph survives (it would be silently dropped if
        // the re-encode loop only handled CsrShard/LayerCsrShard).
        let obsp_shards: Vec<_> = out
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::ObspCsrShard)
            .collect();
        assert_eq!(obsp_shards.len(), 1, "obsp CSR shard preserved");
        // Every CSR-class shard (X + obsp) is canonical post-optimize.
        assert!(out
            .validate_canonical_csr_shards()
            .iter()
            .all(|(_, ok)| *ok));
    }

    #[test]
    fn optimize_copies_sharded_obsm_verbatim() {
        use arrow::array::{Float32Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        let output = dir.path().join("out.scx");

        let n_obs = 6usize;
        let n_vars = 1000usize;
        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 2;
        header.index_dtype = 1;
        let (indptr, indices, values) = small_csr(n_obs);
        let schema = Arc::new(Schema::new(vec![
            Field::new("pc1", DataType::Float32, false),
            Field::new("pc2", DataType::Float32, false),
        ]));
        {
            let mut w = ScxWriter::new(&input, header).unwrap();
            w.write_obs(&sample_obs(n_obs)).unwrap();
            w.write_var(&sample_var(n_vars)).unwrap();
            w.write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
            // Two 3-row obsm shards (sharded layout to prove it is preserved).
            for shard_idx in 0u32..2 {
                let row_start = shard_idx as usize * 3;
                let rows: Vec<f32> = (row_start..row_start + 3).map(|r| r as f32).collect();
                let rows2: Vec<f32> = rows.iter().map(|r| r * 2.0).collect();
                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Float32Array::from(rows)),
                        Arc::new(Float32Array::from(rows2)),
                    ],
                )
                .unwrap();
                w.write_obsm_shard(
                    "X_pca",
                    shard_idx,
                    row_start as u64,
                    batch.num_rows() as u64,
                    n_obs as u64,
                    &batch,
                )
                .unwrap();
            }
            w.finish().unwrap();
        }

        let in_reader = ScxReader::open(&input).unwrap();
        let in_shards: Vec<(String, Vec<u8>)> = in_reader
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::ObsmEmbeddingShard)
            .map(|e| (e.name.clone(), in_reader.section_bytes(e).unwrap().to_vec()))
            .collect();
        assert_eq!(in_shards.len(), 2);
        drop(in_reader);

        optimize(&input, &output, None, ObsShardPolicy::Off).unwrap();

        let out = ScxReader::open(&output).unwrap();
        let out_shards: Vec<(String, Vec<u8>)> = out
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::ObsmEmbeddingShard)
            .map(|e| (e.name.clone(), out.section_bytes(e).unwrap().to_vec()))
            .collect();
        // Sharded layout preserved (not flattened) and copied byte-for-byte.
        assert_eq!(out_shards.len(), 2, "sharded obsm preserved as shards");
        assert_eq!(in_shards, out_shards, "obsm shards copied verbatim");
        assert!(
            !out.catalog()
                .entries
                .iter()
                .any(|e| e.section_type == SectionType::ObsmEmbedding),
            "no flattened single-section obsm emitted"
        );
    }

    #[test]
    fn optimize_preserves_already_sharded_obs_under_all_policies() {
        // A sharded `ObsMetadataShard` input must NOT collapse to a single
        // legacy `ObsMetadata` section (the atlas-scale read_obs() OOM +
        // layout regression this fix addresses) — and this is invariant to
        // `ObsShardPolicy`: the policy only governs single-section input, while
        // already-sharded obs always takes the stream-preserve path.
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");

        let n_obs = 6usize;
        let n_vars = 1000usize;
        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 2;
        header.index_dtype = 1;
        let (indptr, indices, values) = small_csr(n_obs);
        let obs = sample_obs(n_obs);
        {
            let mut w = ScxWriter::new(&input, header).unwrap();
            // Two 3-row obs shards (sharded layout to prove it is preserved).
            for shard_idx in 0u32..2 {
                let row_start = shard_idx as usize * 3;
                let slice = obs.slice(row_start, 3); // zero-copy
                w.write_obs_shard(shard_idx, row_start as u64, 3, n_obs as u64, &slice)
                    .unwrap();
            }
            w.write_var(&sample_var(n_vars)).unwrap();
            w.write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
            w.finish().unwrap();
        }

        for (i, policy) in [
            ObsShardPolicy::Off,
            ObsShardPolicy::Auto,
            ObsShardPolicy::Always,
        ]
        .into_iter()
        .enumerate()
        {
            let output = dir.path().join(format!("out_{i}.scx"));
            optimize(&input, &output, None, policy).unwrap();

            let out = ScxReader::open(&output).unwrap();
            // Sharded layout preserved (not flattened to a single section).
            assert_eq!(
                out.catalog()
                    .entries
                    .iter()
                    .filter(|e| e.section_type == SectionType::ObsMetadataShard)
                    .count(),
                2,
                "sharded obs preserved as shards under {policy:?}"
            );
            assert!(
                !out.catalog()
                    .entries
                    .iter()
                    .any(|e| e.section_type == SectionType::ObsMetadata),
                "no flattened single-section obs emitted under {policy:?}"
            );
            // Rows + values still readable and intact (assembled view).
            let out_obs = out.read_obs().unwrap();
            assert_eq!(out_obs.num_rows(), n_obs);
            let ids = out_obs
                .column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();
            assert_eq!(ids.value(0), "cell_0");
            assert_eq!(ids.value(n_obs - 1), &format!("cell_{}", n_obs - 1)[..]);
        }
    }

    #[test]
    fn optimize_preserves_sharded_var_layout() {
        // Mirror of the obs test for the var axis.
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        let output = dir.path().join("out.scx");

        let n_obs = 6usize;
        let n_vars = 8usize;
        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 2;
        header.index_dtype = 0; // n_vars <= 65535 → u16 indices
        let (indptr, indices, values) = small_csr(n_obs);
        // small_csr emits indices > n_vars; clamp into [0, n_vars) so the shard
        // is valid for this small gene axis.
        let indices: Vec<u32> = indices.iter().map(|&c| c % n_vars as u32).collect();
        let var = sample_var(n_vars);
        {
            let mut w = ScxWriter::new(&input, header).unwrap();
            w.write_obs(&sample_obs(n_obs)).unwrap();
            // Two 4-row var shards.
            for shard_idx in 0u32..2 {
                let row_start = shard_idx as usize * 4;
                let slice = var.slice(row_start, 4); // zero-copy
                w.write_var_shard(shard_idx, row_start as u64, 4, n_vars as u64, &slice)
                    .unwrap();
            }
            w.write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
            w.finish().unwrap();
        }

        optimize(&input, &output, None, ObsShardPolicy::Off).unwrap();

        let out = ScxReader::open(&output).unwrap();
        assert_eq!(
            out.catalog()
                .entries
                .iter()
                .filter(|e| e.section_type == SectionType::VarMetadataShard)
                .count(),
            2,
            "sharded var preserved as shards"
        );
        assert!(
            !out.catalog()
                .entries
                .iter()
                .any(|e| e.section_type == SectionType::VarMetadata),
            "no flattened single-section var emitted"
        );
        let out_var = out.read_var().unwrap();
        assert_eq!(out_var.num_rows(), n_vars);
        let ids = out_var
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        assert_eq!(ids.value(0), "gene_0");
        assert_eq!(ids.value(n_vars - 1), &format!("gene_{}", n_vars - 1)[..]);
    }

    // ---- `ObsShardPolicy` single-section migration (this feature) ----

    /// Build a single-section obs `RecordBatch` with a Utf8 id column and a
    /// categorical (`Dictionary<Int32, Utf8>`) `cell_type` column — exercises
    /// the per-shard re-dictionary path that sharding must round-trip (the
    /// riskiest interaction per the spec's Risks section).
    fn categorical_obs(n: usize) -> arrow::array::RecordBatch {
        use arrow::array::{DictionaryArray, Int32Array, StringArray};
        use arrow::datatypes::{DataType, Field, Int32Type, Schema};
        use std::sync::Arc;

        let labels = ["T cell", "B cell", "NK cell"];
        let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
        let keys = Int32Array::from((0..n).map(|i| (i % 3) as i32).collect::<Vec<_>>());
        let values = Arc::new(StringArray::from(labels.to_vec()));
        let cell_type = DictionaryArray::<Int32Type>::try_new(keys, values).unwrap();
        let schema = Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new(
                "cell_type",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            ),
        ]);
        arrow::array::RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from(
                    ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                )),
                Arc::new(cell_type),
            ],
        )
        .unwrap()
    }

    /// Read the `cell_type` column as plain strings regardless of whether it
    /// came back as `Dictionary` (single-section) or unified `Dictionary`/`Utf8`
    /// (assembled from shards) — cast to `Utf8` and collect.
    fn cell_type_strings(batch: &arrow::array::RecordBatch) -> Vec<String> {
        use arrow::array::Array;
        let idx = batch.schema().index_of("cell_type").unwrap();
        let utf8 =
            arrow::compute::cast(batch.column(idx), &arrow::datatypes::DataType::Utf8).unwrap();
        let arr = utf8
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
    }

    /// Write a single-section (legacy `ObsMetadata`) file with `obs`, a small
    /// var, and one CSR shard. `shard_target_rows` is stamped into the header so
    /// the `Auto` threshold can be exercised with tiny fixtures.
    fn build_single_section_file(
        path: &Path,
        n_obs: usize,
        shard_target_rows: u32,
        obs: &arrow::array::RecordBatch,
    ) {
        let n_vars = 1000usize;
        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 2;
        header.index_dtype = 1;
        header.shard_target_rows = shard_target_rows;
        let (indptr, indices, values) = small_csr(n_obs);
        let mut w = ScxWriter::new(path, header).unwrap();
        w.write_obs(obs).unwrap(); // single legacy ObsMetadata section
        w.write_var(&sample_var(n_vars)).unwrap();
        w.write_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
        w.finish().unwrap();
    }

    fn obs_shard_count(path: &Path) -> usize {
        ScxReader::open(path).unwrap().obs_metadata_shard_count()
    }

    fn has_single_section_obs(path: &Path) -> bool {
        ScxReader::open(path)
            .unwrap()
            .catalog()
            .entries
            .iter()
            .any(|e| e.section_type == SectionType::ObsMetadata)
    }

    #[test]
    fn optimize_auto_shards_large_single_section_obs() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        let output = dir.path().join("out.scx");

        let n_obs = 10usize; // 10 > shard_target_rows(4) → Auto shards
        let obs = categorical_obs(n_obs);
        build_single_section_file(&input, n_obs, 4, &obs);
        assert_eq!(obs_shard_count(&input), 0, "input is single-section");

        optimize(&input, &output, None, ObsShardPolicy::Auto).unwrap();

        assert!(
            obs_shard_count(&output) > 0,
            "auto shards a large single-section obs"
        );
        assert!(
            !has_single_section_obs(&output),
            "no single-section ObsMetadata remains after sharding"
        );
        // Round-trips row-for-row, including the categorical column.
        let out_obs = ScxReader::open(&output).unwrap().read_obs().unwrap();
        assert_eq!(out_obs.num_rows(), n_obs);
        assert_eq!(cell_type_strings(&out_obs), cell_type_strings(&obs));
    }

    #[test]
    fn optimize_auto_keeps_small_single_section_obs() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        let output = dir.path().join("out.scx");

        // n_obs == shard_target_rows → NOT sharded under Auto (strict `>`,
        // matching from_anndata's `obs_rows > step` boundary).
        let n_obs = 8usize;
        let obs = categorical_obs(n_obs);
        build_single_section_file(&input, n_obs, 8, &obs);

        optimize(&input, &output, None, ObsShardPolicy::Auto).unwrap();

        assert_eq!(
            obs_shard_count(&output),
            0,
            "auto keeps a small single-section obs as a single section"
        );
        assert!(has_single_section_obs(&output));
        let out_obs = ScxReader::open(&output).unwrap().read_obs().unwrap();
        assert_eq!(cell_type_strings(&out_obs), cell_type_strings(&obs));
    }

    #[test]
    fn optimize_off_keeps_single_section() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        let output = dir.path().join("out.scx");

        // 10 > 4 would shard under Auto; Off must keep the single section.
        let n_obs = 10usize;
        let obs = categorical_obs(n_obs);
        build_single_section_file(&input, n_obs, 4, &obs);

        optimize(&input, &output, None, ObsShardPolicy::Off).unwrap();

        assert_eq!(obs_shard_count(&output), 0, "off never shards");
        assert!(has_single_section_obs(&output));
    }

    #[test]
    fn optimize_always_shards_single_section() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        let output = dir.path().join("out.scx");

        // n_obs(6) < shard_target_rows(10000): Auto would NOT shard, but
        // Always must.
        let n_obs = 6usize;
        let obs = categorical_obs(n_obs);
        build_single_section_file(&input, n_obs, 10000, &obs);

        optimize(&input, &output, None, ObsShardPolicy::Always).unwrap();

        assert!(
            obs_shard_count(&output) > 0,
            "always shards regardless of size"
        );
        assert!(!has_single_section_obs(&output));
        let out_obs = ScxReader::open(&output).unwrap().read_obs().unwrap();
        assert_eq!(cell_type_strings(&out_obs), cell_type_strings(&obs));
    }

    #[test]
    fn optimize_always_on_empty_obs_stays_single_section() {
        // `write_obs_section` guards `n == 0` and never shards an empty obs,
        // even under `Always` — assert that end-to-end.
        use arrow::array::RecordBatch;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        let output = dir.path().join("out.scx");

        let n_vars = 1000usize;
        let mut header = sample_header(0, n_vars as u64);
        header.format_version = 2;
        header.index_dtype = 1;
        header.shard_target_rows = 4;
        let empty_obs = RecordBatch::new_empty(Arc::new(Schema::new(vec![Field::new(
            "cell_id",
            DataType::Utf8,
            false,
        )])));
        {
            let mut w = ScxWriter::new(&input, header).unwrap();
            w.write_obs(&empty_obs).unwrap();
            w.write_var(&sample_var(n_vars)).unwrap();
            // Zero-row CSR shard (indptr = [0], no entries).
            w.write_csr_shard(&[0u64], &[], &[], CodecId::None, ValueEncoding::Uint8, 0)
                .unwrap();
            w.finish().unwrap();
        }

        optimize(&input, &output, None, ObsShardPolicy::Always).unwrap();

        assert_eq!(
            obs_shard_count(&output),
            0,
            "an empty obs is never sharded, even under Always"
        );
        assert!(has_single_section_obs(&output));
        assert_eq!(
            ScxReader::open(&output)
                .unwrap()
                .read_obs()
                .unwrap()
                .num_rows(),
            0
        );
    }

    #[test]
    fn optimize_shard_obs_preserves_predicate_index() {
        use scx_engine::{
            build_and_write_conversion_predicate_indexes, ConversionPredicateIndexOptions,
            QueryPipeline,
        };

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        let output = dir.path().join("out.scx");

        let n_obs = 12usize;
        let n_vars = 1000usize;
        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 2;
        header.index_dtype = 1;
        header.shard_target_rows = 4; // Always reshapes obs into 3 shards
        let obs = sample_obs(n_obs); // cell_type cycles T/B/NK
        let var = sample_var(n_vars);
        let (indptr, indices, values) = small_csr(n_obs);
        {
            let mut w = ScxWriter::new(&input, header).unwrap();
            w.write_obs(&obs).unwrap(); // single legacy section
            w.write_var(&var).unwrap();
            w.write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
            // One CSR shard covers rows 0..n_obs; build a forced obs predicate
            // index on cell_type keyed to that range.
            let opts = ConversionPredicateIndexOptions {
                index_obs: vec!["cell_type".to_string()],
                index_var: vec![],
                index_preset: None,
                index_auto_threshold: 0,
            };
            build_and_write_conversion_predicate_indexes(
                &mut w,
                &obs,
                &var,
                &[(0, n_obs as u64)],
                n_vars,
                &opts,
            )
            .unwrap();
            w.finish().unwrap();
        }

        let expr = "cell_type == 'T cell'";
        let matched = |path: &Path| -> usize {
            let pipeline = QueryPipeline::open(path).unwrap().filter_obs(expr).unwrap();
            scx_engine::collect::count(&pipeline).unwrap().matched_rows
        };

        let before = matched(&input);
        assert!(before > 0, "baseline query matches some rows");
        assert!(
            ScxReader::open(&input)
                .unwrap()
                .catalog()
                .entries
                .iter()
                .any(|e| e.section_type == SectionType::ObsPredicateIndex),
            "input carries an obs predicate index"
        );

        // Always reshapes the single-section obs into shards; the predicate
        // index is keyed to CSR/output shards (unchanged), so it must remain
        // valid and the query must return identical rows.
        optimize(&input, &output, None, ObsShardPolicy::Always).unwrap();

        assert!(obs_shard_count(&output) > 0, "obs reshaped to shards");
        assert!(
            ScxReader::open(&output)
                .unwrap()
                .catalog()
                .entries
                .iter()
                .any(|e| e.section_type == SectionType::ObsPredicateIndex),
            "predicate index carried through optimize"
        );
        assert_eq!(
            matched(&output),
            before,
            "predicate-index query is stable across obs reshape"
        );
    }
}
