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

use rayon::prelude::*;
use scx_codec::CodecId;
use scx_format_io::encoder::{encode_one_shard, FramingConfig};
use scx_format_io::header::{FileHeader, CURRENT_FORMAT_VERSION};
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
/// stamping `format_version = 3` (unframed). Single-modality
/// only (multimodal files should use `scx compact`).
///
/// `codec` selects the per-shard codec passed to the encoder: `None` keeps the
/// auto-codec (Scx1 for low-median integer counts, else Zstd), while
/// `Some(CodecId::Scx1)` forces Scx1 on every integer shard — keeping the
/// `to_gpu_anndata` device-decode route (framed Scx1 decodes in VRAM) even for
/// high-median shards that auto would route to Zstd. Non-integer shards
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
    optimize_with_framing(input_path, output_path, codec, obs_shard_policy, None, None)?;
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
/// `memory_budget` bounds how many shards may be encoded concurrently: the
/// re-encode runs in chunks whose whole live phase fits the budget. `None` does
/// **not** mean unbounded — see `encode_budget::DEFAULT_IN_FLIGHT_BYTES`, which
/// keeps the default peak increase a constant instead of a multiple of the
/// host's core count. A budget smaller than one shard's phase still admits one
/// shard, because a single shard's encode is irreducible.
pub fn optimize_with_framing(
    input_path: &Path,
    output_path: &Path,
    codec: Option<CodecId>,
    obs_shard_policy: ObsShardPolicy,
    framing: Option<FramingConfig>,
    memory_budget: Option<u64>,
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
    // Raw (`adata.raw`) is not carried through optimize's section allowlist.
    // Warn loudly rather than drop it silently (SCX-002), matching
    // compact/sort/merge.
    if in_header.has_raw() {
        log::warn!(
            "scx optimize: input {} carries an adata.raw matrix, which is not yet \
             preserved through optimize — raw will be dropped from the output",
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
        scx_format_io::write_obs_section(
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
    // Set when canonicalising X actually changed the matrix — see the
    // detection-bitmap block far below, which is the only consumer.
    let mut x_was_rewritten = false;
    let entries: Vec<_> = x_entries
        .into_iter()
        .chain(layer_entries)
        .chain(obsp_csr_entries)
        .collect();

    // Read the 76-byte shard headers up front. The serial loop this replaces
    // read one per entry anyway (for `n_minor`); doing it here also gives the
    // per-shard `nnz` the chunk planner needs, so the header reads are the same
    // count, just hoisted.
    let headers = entries
        .iter()
        .map(|e| reader.read_shard_header(e))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let nnz_per_shard: Vec<u64> = headers.iter().map(|h| h.nnz).collect();

    // Re-encode in bounded parallel chunks. Each entry's work -- read,
    // canonicalize, encode -- is a pure function of that entry, and `ScxReader`
    // is shareable (mmap + `Arc<FullCatalog>`), so the *whole* body moves off
    // the calling thread rather than just the encode. That matters here: the
    // measured serial share of `scx optimize` at census_1m was 45 % of a 42.4 s
    // wall (87.0 s at one thread, 43.7 s at eight, 42.4 s at sixteen -- flat
    // past eight cores), and the source-shard decode is part of it.
    //
    // The writer stays serial and is fed in chunk order. That is not a
    // simplification: `current_offset`, the catalog entry order and
    // `first_csr_codec` (first CSR write wins, and `finish()` stamps it as the
    // file header's codec) are all positional, so parallel encode is safe and
    // parallel write is not.
    let chunk_lens = crate::encode_budget::plan_encode_chunks(
        &nnz_per_shard,
        rayon::current_num_threads().max(1),
        crate::encode_budget::resolve_in_flight_budget(memory_budget),
    );
    let mut at = 0usize;
    for len in chunk_lens {
        // `Vec<Result<_>>` rather than `collect::<Result<Vec<_>>>()`: rayon
        // leaves it undefined *which* error a short-circuiting collect returns,
        // and the op must fail with the first one in shard order however the
        // pool happened to schedule. Same rule as `encode_shard_framed`'s
        // per-group collect.
        let encoded: Vec<Result<(scx_format_io::PreEncodedSection, bool)>> = entries[at..at + len]
            .par_iter()
            .zip(headers[at..at + len].par_iter())
            .map(|(entry, sh)| {
                let row_start = entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
                let (indptr_i64, indices_i32, mut values) = reader.read_shard_from_entry(entry)?;
                let mut indptr: Vec<u64> = indptr_i64.iter().map(|&v| v as u64).collect();
                let mut indices: Vec<u32> = indices_i32.iter().map(|&v| v as u32).collect();
                // Whether canonicalisation rewrote **X** specifically, which is
                // what the detection bitmaps below are keyed to.
                // `canonicalize_csr` short-circuits on already-canonical input,
                // so this is the same test it runs internally and costs nothing
                // extra on the common path. Reported per shard and OR-ed in the
                // serial fold below, since a `&mut bool` cannot cross into the
                // pool.
                let rewrote = entry.section_type == SectionType::CsrShard
                    && !scx_sparse::is_canonical_csr(&indptr, &indices, &values);
                canonicalize_csr(&mut indptr, &mut indices, &mut values);

                // The shard's own minor dimension rather than the file-level
                // `n_vars`: correct for `ObspCsrShard` (minor axis = `n_obs`)
                // and robust against any per-shard width difference.
                let mut enc_opts = scx_format_io::EncodeShardOptions::new(
                    entry.name.clone(),
                    entry.section_type,
                    sh.n_minor as u64,
                    row_start,
                    index_dtype,
                );
                // Passed through as the caller gave it, `None` included: with
                // no explicit codec the encoder runs its own selection, and
                // substituting a pre-seeded codec here would change the pick.
                enc_opts.explicit_codec = codec;
                enc_opts.framing = framing;
                let pre = encode_one_shard(&indptr, &indices, &values, &enc_opts)?;
                Ok((pre, rewrote))
            })
            .collect();

        for result in encoded {
            let (pre, rewrote) = result?;
            x_was_rewritten |= rewrote;
            stats.shards_total += 1;
            if pre.shard_format_version() > scx_format_io::DEFAULT_WRITE_SHARD_FORMAT_VERSION {
                stats.shards_framed += 1;
                if pre.codec_id() == CodecId::ShufDeltaZstd as u8 {
                    stats.shards_shufdelta += 1;
                }
            }
            writer.write_preencoded_shard(pre)?;
        }
        at += len;
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

    // Detection-bitmap shards (SCXB) are keyed to CSR shard-local rows and are
    // read back in `row_start` order (`bitmap_shards_for_modality`). Optimize
    // preserves row order and CSR shard boundaries 1:1, so the keys stay valid
    // and the sidecar is copied verbatim (sorted by `row_start` for
    // deterministic output). Dropping it unconditionally would silently disable
    // `detection_counts` / `cells_expressing`.
    //
    // ⚠️ The clause that used to justify this was **false**, and it read
    // plausibly for two releases: "`canonicalize_csr` only reorders `(col,
    // val)` within a row (the set of expressed genes per row is unchanged)".
    // `canonicalize_csr` also calls `drop_explicit_zeros_inplace`, and
    // `BitmapShard::build_from_csr` records a gene for a row whenever the row
    // *stores* that column, regardless of the value in it. So an input holding
    // an explicit zero came out of `scx optimize` with a sidecar claiming a
    // gene the output's own X no longer stores, and `detection_counts`
    // over-reported it with nothing to notice by. Found in Phase 5b while
    // giving `upgrade` the same carry; `upgrade`'s cell is conditional for
    // exactly this reason, and now so is this one.
    //
    // Rare in practice — `canonicalize_csr` short-circuits via
    // `is_canonical_csr` and every file a current writer produces is canonical
    // — which is why the sidecar is dropped only when X was actually rewritten,
    // rather than dropped outright the way the CSC sidecar is.
    let mut bitmap_entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::BitmapShard)
        .collect();
    if x_was_rewritten && !bitmap_entries.is_empty() {
        log::warn!(
            "scx optimize: canonicalizing X changed which genes each row stores, so the \
             detection bitmaps built against the old matrix would over-report and have \
             been dropped. Rebuild them with `scx sort --bitmap always` or a re-convert \
             if you need `detection_counts` / `cells_expressing`."
        );
        bitmap_entries.clear();
    }
    bitmap_entries.sort_by_key(|e| e.stats.as_ref().map(|s| s.row_start).unwrap_or(0));
    for entry in bitmap_entries {
        let bytes = reader.section_bytes(entry)?;
        writer.copy_section_verbatim(entry, bytes)?;
    }

    if let Ok(uns) = reader.read_uns() {
        writer.write_uns(&uns)?;
    }

    // Deletion vectors are carried through (optimize does not apply them — that
    // is `compact`'s job). v2 deletion vectors store global obs row indices,
    // independent of shard layout, and optimize preserves the obs ordering, so
    // the section stays valid; a legacy v1 section is folded to v2 by
    // `read_deletion_vectors`, so `write_deletion_vectors` re-emits v2 bytes.
    // It also re-sets the header flag (cleared above) so it is set iff the
    // section exists.
    if let Some(dv) = reader.read_deletion_vectors()? {
        writer.write_deletion_vectors(&dv)?;
    }

    // The F1 grouped-sharding sidecar (`group_index`) records GLOBAL output-row
    // ranges + shard indices. Optimize preserves row order and CSR shard
    // boundaries 1:1, so those ranges/indices still point at the right rows and
    // shards — copy it verbatim to keep grouped reads (`read_group` /
    // `read_reference`) working. Contrast `append`, which appends rows over the
    // pre-append row universe and therefore must drop it (append.rs).
    if let Some(gi_entry) = reader
        .catalog()
        .entries
        .iter()
        .find(|e| e.section_type == SectionType::GroupIndex)
    {
        let bytes = reader.section_bytes(gi_entry)?;
        writer.copy_section_verbatim(gi_entry, bytes)?;
    }

    // Predicate indexes reference obs row ranges; rows + shard boundaries are
    // preserved, so they stay valid — copy through. Then record provenance.
    copy_predicate_indices(&reader, &mut writer)
        .map_err(|e| OpsError::InvalidInput(format!("copy predicate indices: {e}")))?;
    // Copying the section bytes is only half of carrying an index. Optimize
    // re-encodes every CSR shard above, and `compute_shard_stats` emits no
    // `column_stats` — so without this the index section survived and Level-1
    // shard pruning silently stopped, turning every `filter_obs` into a full scan
    // that still returned the right rows. `build-csc` had the same defect and the
    // same fix, and so does `upgrade` — three ops, not two. `build-csc` and
    // `optimize` declare `Carry::Verbatim` for this family in their own match
    // arms; `upgrade` reaches the same arm through `other => build_csc(other)`
    // and re-emits every CSR shard too. Counting only the explicit arms is how
    // the first pass missed it.
    //
    // Carried from the input rather than re-derived from the index: optimize
    // preserves row order and CSR shard boundaries 1:1, so the input's stats are
    // already correct for the output's shards, and re-deriving would have to
    // infer that the index is CSR-keyed — which the index bytes cannot establish.
    // See `scx_format_io::carry_csr_shard_column_stats`.
    writer.carry_csr_shard_column_stats_from(&reader.catalog().entries);

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

    crate::carry::audit_staged(
        crate::carry::RewriteOp::Optimize,
        &[reader.catalog()],
        &writer,
    )?;
    writer.finish()?;
    Ok(stats)
}

/// Stream var shards from `reader` straight to sharded output, one output shard
/// per non-empty input shard. Peak memory is one shard — `read_var()` (which
/// assembles every `VarMetadataShard` into one batch) is never called. Mirror of
/// compact's [`write_obs_shards_streaming`](crate::compact::write_obs_shards_streaming)
/// for the var axis; no op deletes columns, so (unlike the obs helper) it takes
/// no `keep_mask` and renumbers output shards over the non-empty inputs.
///
/// `pub(crate)` like its obs twin, and reached from outside the crate through
/// [`crate::rewrite_helpers::copy_obs_var_preserving_layout`] — `n_vars_total`
/// is stamped into every output shard, so a caller that passes the wrong one
/// writes a file whose var cover disagrees with its own stamp.
pub(crate) fn write_var_shards_streaming(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    n_vars_total: u64,
) -> Result<()> {
    let mut out_idx = 0u32;
    let mut row_start = 0u64;
    let mut cover = crate::compact::ShardCoverCheck::default();
    let n_shards = reader.var_metadata_shard_count();
    for res in reader.var_shards() {
        let batch = res?;
        // Same cover validation `read_var()` performs while assembling, which
        // the streaming path would otherwise trade away — see `ShardCoverCheck`.
        cover.visit("var_metadata", &batch)?;
        if cover.seen as usize == n_shards {
            cover.finish("var_metadata", n_vars_total)?;
        }
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
    fn optimize_canonicalizes_and_upgrades_to_v3() {
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

        optimize(&input, &output, None, ObsShardPolicy::Off).unwrap();

        let out = ScxReader::open(&output).unwrap();
        assert_eq!(out.header().format_version, 3, "optimize stamps v3");

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

    /// Optimize must carry the F1 `group_index` sidecar and the
    /// detection-bitmap shards through verbatim (CSR boundaries are preserved
    /// 1:1), so a grouped / bitmapped file stays grouped / bitmapped after
    /// optimize instead of silently losing those capabilities.
    #[test]
    fn optimize_preserves_group_index_and_bitmaps() {
        use scx_format::{GroupIndexPayload, GroupRecordWire};
        use scx_format_io::bitmap::BitmapShard;

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        let output = dir.path().join("out.scx");

        let n_obs = 8usize;
        let n_vars = 100usize;
        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 2;
        header.index_dtype = 0; // u16 indices (n_vars < 65536)

        // Simple canonical CSR: each row expresses one or two genes.
        let mut indptr = vec![0u64];
        let mut indices: Vec<u32> = Vec::new();
        let mut values: Vec<u8> = Vec::new();
        for r in 0..n_obs {
            let g0 = (r % n_vars) as u32;
            let g1 = ((r + 5) % n_vars) as u32;
            let (lo, hi) = if g0 <= g1 { (g0, g1) } else { (g1, g0) };
            indices.push(lo);
            values.push(1);
            if hi != lo {
                indices.push(hi);
                values.push(2);
            }
            indptr.push(indices.len() as u64);
        }

        // Realistic group_index (two contiguous groups over the single shard).
        let payload = GroupIndexPayload {
            group_by: "cell_type".to_string(),
            records: vec![
                GroupRecordWire {
                    label: "ref".to_string(),
                    role: "reference".to_string(),
                    row_start: 0,
                    row_stop: 4,
                    shard: 0,
                },
                GroupRecordWire {
                    label: "grp".to_string(),
                    role: "group".to_string(),
                    row_start: 4,
                    row_stop: n_obs as u64,
                    shard: 0,
                },
            ],
            reference_labels: vec!["ref".to_string()],
            reference_shard: Some(0),
        };
        let gi_bytes = serde_json::to_vec(&payload).unwrap();
        let bm = BitmapShard::build_from_csr(0, n_obs as u32, n_vars as u32, &indptr, &indices);

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
            w.write_bitmap_shard(&bm).unwrap();
            w.write_group_index(&gi_bytes).unwrap();
            w.finish().unwrap();
        }

        // Sanity: the input carries both sidecars.
        {
            let inp = ScxReader::open(&input).unwrap();
            assert_eq!(
                inp.read_group_index_bytes().unwrap(),
                Some(gi_bytes.as_slice())
            );
            assert_eq!(inp.bitmap_shard_count(0), 1);
        }

        optimize(&input, &output, None, ObsShardPolicy::Off).unwrap();

        let out = ScxReader::open(&output).unwrap();
        // group_index carried through byte-identically → grouped reads still work.
        assert_eq!(
            out.read_group_index_bytes().unwrap(),
            Some(gi_bytes.as_slice()),
            "optimize must preserve the group_index sidecar verbatim"
        );
        // Detection bitmap preserved and still associated with the shard.
        assert_eq!(
            out.bitmap_shard_count(0),
            1,
            "optimize must preserve detection-bitmap shards"
        );
        let out_bm = out.read_bitmap_shard(0).unwrap();
        assert_eq!(out_bm.n_rows, n_obs as u32);
        assert_eq!(out_bm.n_vars, n_vars as u32);
    }

    /// T3.2: `optimize_with_framing` (the `scx optimize --codec compact-trial
    /// --row-group-rows N` path) upgrades a v2/v3 file to a v4/shard-v2 framed
    /// file, and the framed re-encode round-trips byte-identically.
    /// An explicit codec must reach **every** shard, and the output must be
    /// byte-identical however many shards the re-encode holds in flight.
    ///
    /// Both halves exist because of what the twelve-arm `op_output_identity`
    /// golden cannot see here: every arm that drives `optimize` passes
    /// `codec: None`, so replacing the caller's codec with `None` inside the
    /// parallel map leaves the golden green. And a single-chunk run says nothing
    /// about chunk boundaries, so the budget is swept from "one shard at a time"
    /// to "all of them" and the bytes compared across the sweep.
    #[test]
    fn optimize_honours_an_explicit_codec_at_every_concurrency() {
        let dir = tempfile::tempdir().unwrap();
        // Several shards, so a chunk boundary can fall between them.
        let n_obs = 40usize;
        let shard_rows = 8usize;
        let n_vars = 500usize;
        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 2;
        header.shard_target_rows = shard_rows as u32;

        let input = dir.path().join("in.scx");
        {
            let mut w = ScxWriter::new(&input, header).unwrap();
            w.write_obs(&sample_obs(n_obs)).unwrap();
            w.write_var(&sample_var(n_vars)).unwrap();
            for shard in 0..(n_obs / shard_rows) {
                let mut indptr = vec![0u64];
                let mut indices: Vec<u32> = Vec::new();
                let mut values: Vec<u8> = Vec::new();
                for r in 0..shard_rows {
                    let mut col = 0u32;
                    for k in 0..16usize {
                        col += 1 + ((shard * 31 + r * 13 + k * 7) % 20) as u32;
                        indices.push(col);
                        // Median <= 8 **on purpose**: `select_codec` picks
                        // `Scx1` for this distribution, so forcing `Zstd` below
                        // is a choice the heuristic would not have made. With a
                        // high median both answers are `Zstd` and the assertion
                        // holds whether or not the forced codec was honoured.
                        values.push(1u8 + ((shard + r + k) % 4) as u8);
                    }
                    indptr.push(indices.len() as u64);
                }
                w.write_csr_shard(
                    &indptr,
                    &indices,
                    &values,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    (shard * shard_rows) as u64,
                )
                .unwrap();
            }
            w.finish().unwrap();
        }

        // `0` pins one shard per chunk; `u64::MAX` puts every shard in one.
        // Anything in between is a real boundary. All must agree byte for byte,
        // and every shard must carry the forced codec.
        type Layout = Vec<(String, u64)>;
        type Shards = Vec<(Vec<i64>, Vec<i32>, Vec<f32>)>;
        let mut reference: Option<(Layout, Shards)> = None;
        for budget in [Some(0u64), Some(1024), Some(64 * 1024 * 1024), None] {
            let output = dir.path().join(format!("out_{budget:?}.scx"));
            optimize_with_framing(
                &input,
                &output,
                Some(CodecId::Zstd),
                ObsShardPolicy::Off,
                None,
                budget,
            )
            .unwrap();

            let reader = ScxReader::open(&output).unwrap();
            let shards = reader.catalog().csr_shards_sorted();
            assert!(shards.len() > 1, "need several shards to have a boundary");
            for entry in &shards {
                let sh = reader.read_shard_header(entry).unwrap();
                assert_eq!(
                    sh.codec_id,
                    CodecId::Zstd as u8,
                    "shard {} lost the forced codec at budget {budget:?}",
                    entry.name
                );
            }

            // Contents **and** file order. `read_csr_shard(i)` goes through
            // `csr_shards_sorted`, which re-sorts by `row_start` and so hides a
            // mis-ordered write entirely; the catalog's own entry order is what
            // records where each shard actually landed. Both are compared
            // across the sweep.
            let layout: Vec<(String, u64)> = reader
                .catalog()
                .entries
                .iter()
                .filter(|e| e.section_type == SectionType::CsrShard)
                .map(|e| {
                    (
                        e.name.clone(),
                        e.stats.as_ref().map(|s| s.row_start).unwrap_or(0),
                    )
                })
                .collect();
            assert!(
                layout.windows(2).all(|w| w[0].1 < w[1].1),
                "shards must be written in ascending row order, got {layout:?} at budget {budget:?}"
            );
            let decoded: Vec<_> = (0..shards.len())
                .map(|i| reader.read_csr_shard(i).unwrap())
                .collect();
            match &reference {
                None => reference = Some((layout, decoded)),
                Some((want_layout, want_decoded)) => {
                    assert_eq!(
                        &layout, want_layout,
                        "shard layout differs at budget {budget:?}"
                    );
                    assert_eq!(
                        &decoded, want_decoded,
                        "shard contents differ at budget {budget:?}"
                    );
                }
            }
        }
    }

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
                // High median (>8) so the heuristic picks Zstd, not Scx1,
                // keeping this a clean "framing round-trips" test.
                values.push(50u8 + ((r + k) % 50) as u8);
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
                decode_target: None,
            }),
            None,
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
            // Mark global row 1 as logically deleted (single shard, row_start 0).
            let mut dv = DeletionVectors::new();
            dv.insert_global([1u32]);
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
        assert!(dv.is_deleted_global(1));
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
            let obsp_shard = scx_format_io::ShardBuffers::new(
                &op_indptr,
                &op_indices,
                &op_values,
                CodecId::None,
                ValueEncoding::Uint8,
            );
            w.write_obsp_shard("connectivities", 0, 0, obsp_shard)
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
    /// `optimize` declares `ObsPredicateIndex => Carry::Verbatim`, and it
    /// satisfies that by copying the section bytes. But it re-encodes every CSR
    /// shard through `encode_one_shard` / `write_preencoded_shard`, and
    /// `compute_shard_stats` does not produce `column_stats` — so the Level-1
    /// statistics the index's shard pruning reads were dropped on the way out.
    /// The query still returned the right rows, after a full scan.
    ///
    /// `optimize_shard_obs_preserves_predicate_index` above could not see this:
    /// it has one CSR shard, and Level-1 pruning is unobservable on a single
    /// shard. This fixture has three, with `cell_type` constant within each, so
    /// a query for one value can eliminate the other two.
    ///
    /// The same defect in `build-csc` and `upgrade`, the other ops declaring
    /// `Carry::Verbatim` here, is pinned at
    /// `scx-cli/tests/cli_ops_integration.rs::build_csc_carries_predicate_index_and_pushdown`.
    #[test]
    fn optimize_preserves_level1_shard_pruning() {
        use std::sync::Arc;

        use arrow::array::RecordBatch;
        use arrow::datatypes::{DataType, Field, Schema};
        use scx_engine::{
            build_and_write_conversion_predicate_indexes, ConversionPredicateIndexOptions,
            QueryPipeline,
        };

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        let output = dir.path().join("out.scx");

        // Three shards of four rows, `cell_type` constant per shard.
        const ROWS_PER_SHARD: usize = 4;
        const N_SHARDS: usize = 3;
        let n_obs = ROWS_PER_SHARD * N_SHARDS;
        let n_vars = 1000usize;
        let types = ["T cell", "B cell", "NK cell"];

        let obs = {
            let schema = Schema::new(vec![
                Field::new("cell_id", DataType::Utf8, false),
                Field::new("cell_type", DataType::Utf8, true),
            ]);
            let ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
            let ct: Vec<&str> = (0..n_obs).map(|i| types[i / ROWS_PER_SHARD]).collect();
            RecordBatch::try_new(
                Arc::new(schema),
                vec![
                    Arc::new(arrow::array::StringArray::from(
                        ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    )),
                    Arc::new(arrow::array::StringArray::from(ct)),
                ],
            )
            .unwrap()
        };
        let var = sample_var(n_vars);

        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 2;
        header.index_dtype = 1;
        {
            let mut w = ScxWriter::new(&input, header).unwrap();
            w.write_obs(&obs).unwrap();
            w.write_var(&var).unwrap();
            let mut ranges: Vec<(u64, u64)> = Vec::new();
            for s in 0..N_SHARDS {
                let (indptr, indices, values) = small_csr(ROWS_PER_SHARD);
                let row_start = (s * ROWS_PER_SHARD) as u64;
                w.write_csr_shard(
                    &indptr,
                    &indices,
                    &values,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    row_start,
                )
                .unwrap();
                ranges.push((row_start, row_start + ROWS_PER_SHARD as u64));
            }
            let opts = ConversionPredicateIndexOptions {
                index_obs: vec!["cell_type".to_string()],
                index_var: vec![],
                index_preset: None,
                index_auto_threshold: 0,
            };
            build_and_write_conversion_predicate_indexes(
                &mut w, &obs, &var, &ranges, n_vars, &opts,
            )
            .unwrap();
            w.finish().unwrap();
        }

        let probe = |path: &Path| -> (usize, usize) {
            let pipeline = QueryPipeline::open(path)
                .unwrap()
                .filter_obs("cell_type == 'T cell'")
                .unwrap();
            let c = scx_engine::collect::count(&pipeline).unwrap();
            (c.skipped_shards, c.matched_rows)
        };

        let (before_skipped, before_matched) = probe(&input);
        assert_eq!(
            before_skipped,
            N_SHARDS - 1,
            "fixture precondition: the input must prune {} of {N_SHARDS} shards, or the \
             comparison below proves nothing",
            N_SHARDS - 1
        );
        assert_eq!(before_matched, ROWS_PER_SHARD);

        optimize(&input, &output, None, ObsShardPolicy::Off).unwrap();

        let (after_skipped, after_matched) = probe(&output);
        assert_eq!(
            after_matched, before_matched,
            "optimize must not change which rows match"
        );
        assert_eq!(
            after_skipped, before_skipped,
            "optimize carries the obs predicate index verbatim, so it must also carry the \
             per-shard column_stats that index's Level-1 pruning reads — carrying the section \
             bytes alone leaves pruning off"
        );
    }
}
