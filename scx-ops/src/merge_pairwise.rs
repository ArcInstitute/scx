//! Carrying the pairwise graphs (`obsp`, `varp`) through a merge.
//!
//! Review §6.4: `merge` had no `obsp` or `varp` writer at all — only a comment
//! reading *"obsp dropped (as plain merge does)"* — so merging per-sample files
//! after computing kNN produced an atlas with no cell–cell graph, silently,
//! while `compact` remapped one through its keep-mask and `sort` through its
//! permutation. The Operations Matrix had no column for either.
//!
//! The two axes need opposite treatment, which is why they are one module
//! rather than one function:
//!
//! - **`obsp` is obs×obs**, and merge concatenates the obs axis, so every
//!   endpoint has to be rebased by its input's offset in the merged row space.
//!   That is a remap, and it is the same remap `compact` already owns
//!   ([`crate::compact::remap_obsp_coo_to_dim`]) — merge just supplies
//!   `old_to_new[r] = offset + r` instead of a keep-mask.
//! - **`varp` is var×var**, and merge validates that every input shares one var
//!   axis, so concatenating would duplicate the matrix. Input 0's is the
//!   canonical one, exactly as `var` and `varm` already are.
//!
//! ## Two things this module deliberately does not do
//!
//! **It never reads through `ScxReader::read_all_obsp`.** That helper keys off
//! the section *name* (`obsp/<key>`), and a modality-scoped graph written
//! through `with_modality` carries the same name with a non-zero `modality_id`
//! — so `read_all_obsp` on a multimodal file returns the per-modality graphs
//! indistinguishably from a file-scope one. Carrying that would silently promote
//! a graph scoped to `rna` into a file-wide one. Everything here filters on
//! `modality_id == 0`, and per-modality pairwise is a declared drop
//! (`carry::per_modality_override`) because the format has no per-modality
//! pairwise reader to round-trip it with.
//!
//! **It is not reachable under `--sort-by`.** A sorted merge interleaves rows
//! from every input, so "this shard covers output rows `[a, b)`" — which is what
//! `write_obsp_shard_coo` stamps and what the reader verifies as a contiguous
//! cover — has no meaning per input. The caller rejects COO obsp up front there,
//! the same way and for the same reason it already rejects obsm.
//!
//! ## It carries the COO encodings only
//!
//! `ObspCsrShard` — the CSR-backed obs×obs graph — is **not** carried by any
//! merge path and does **not** trigger the sorted-merge refusal. Nothing in the
//! crate reads one outside `optimize`'s own shard loop, so there is nothing to
//! rebase; `compact` and `sort` drop it for the same reason.
//!
//! That distinction has to be said out loud wherever this module's behaviour is
//! described, because "merge carries obsp" and "merge --sort-by refuses obsp"
//! are both **false of the CSR encoding**, and shipping them unqualified is how
//! a declared drop turns into a promise the code does not keep.
//! [`warn_dropped_csr_backed_obsp`] is what tells the user at run time.

use std::collections::BTreeSet;

use scx_format_io::catalog::FullCatalogEntry;
use scx_format_io::{ScxReader, ScxWriter, SectionType};

use crate::compact::remap_obsp_coo_to_dim;
use crate::error::{OpsError, Result};

/// True if any input carries a **file-scope** obs×obs graph, in the COO
/// encoding this module carries.
///
/// Used by the sorted-merge guard. Per-modality graphs do not count: they are
/// dropped on every merge path, so refusing a sorted merge over one would be
/// refusing to do something the caller was not going to get anyway.
///
/// `ObspCsrShard` deliberately does not count either, and that is a *narrower*
/// promise than "merge --sort-by refuses obsp" — see
/// [`warn_dropped_csr_backed_obsp`], which is what tells the user about it.
pub(crate) fn has_global_obsp(readers: &[ScxReader]) -> bool {
    readers.iter().any(|r| {
        r.catalog().entries.iter().any(|e| {
            e.modality_id == 0
                && matches!(
                    e.section_type,
                    SectionType::ObspEmbedding | SectionType::ObspEmbeddingShard
                )
        })
    })
}

/// Warn about a **CSR-backed** obs×obs graph, which no merge path carries.
///
/// `merge` reads and rebases the COO encodings only. Nothing in the crate reads
/// a CSR-backed pairwise graph outside `optimize`'s own shard loop, so there is
/// nothing to rebase — the same reason `compact` and `sort` drop it.
///
/// Before this warning it was a **silent** drop on the plain path, and on the
/// sorted path the guard did not fire at all, so a `merge --sort-by` over a
/// CSR-backed graph succeeded and lost it while the CLI help and
/// `docs/operations.md` both said obsp was refused. Saying "obsp" without
/// qualifying the encoding is what made that read as a promise.
///
/// Found by codex - gpt-5.6-sol.
pub(crate) fn warn_dropped_csr_backed_obsp(readers: &[ScxReader]) {
    let any = readers.iter().any(|r| {
        r.catalog()
            .entries
            .iter()
            .any(|e| e.modality_id == 0 && e.section_type == SectionType::ObspCsrShard)
    });
    if any {
        log::warn!(
            "scx merge: dropping the CSR-backed obsp graph — merge carries the COO \
             encodings only, and nothing outside `scx optimize` reads a CSR-backed \
             pairwise graph. Run `scx optimize` on the inputs if you need it preserved, \
             or recompute neighbours after the merge."
        );
    }
}

/// Warn about modality-scoped pairwise sections a merge is about to drop.
///
/// Separate from the carry itself because it is the *only* thing merge does
/// with them, and because saying so is the difference between this drop and the
/// one §6.4 was written about.
pub(crate) fn warn_dropped_per_modality_pairwise(readers: &[ScxReader]) {
    let any = readers.iter().any(|r| {
        r.catalog().entries.iter().any(|e| {
            e.modality_id != 0
                && matches!(
                    e.section_type,
                    SectionType::ObspEmbedding
                        | SectionType::ObspEmbeddingShard
                        | SectionType::ObspCsrShard
                        | SectionType::VarpEmbedding
                        | SectionType::VarpEmbeddingShard
                )
        })
    });
    if any {
        log::warn!(
            "scx merge: dropping modality-scoped obsp/varp graphs — the format has no \
             per-modality pairwise reader, so they cannot be round-tripped. File-scope \
             graphs are carried."
        );
    }
}

/// Carry `obsp` and `varp` through a concatenating merge.
///
/// `offsets[i]` is input `i`'s first row in the merged obs space;
/// `total_n_obs` is the merged row count, which becomes the output graph's
/// `n_rows`/`n_cols`.
///
/// Emits one obsp shard **per input shard**, so peak memory is one input shard
/// rather than the whole merged graph, and no single section approaches Arrow
/// IPC's 2 GB narrow-offset ceiling — the same reason the obsm path streams.
pub(crate) fn merge_pairwise_sections(
    readers: &[ScxReader],
    writer: &mut ScxWriter,
    offsets: &[u64],
    total_n_obs: u64,
) -> Result<()> {
    warn_dropped_per_modality_pairwise(readers);
    warn_dropped_csr_backed_obsp(readers);
    merge_obsp(readers, writer, offsets, total_n_obs)?;
    carry_varp_from_input_zero(readers, writer)?;
    Ok(())
}

/// The obs×obs half alone, for the multimodal path.
///
/// Multimodal merge carries a **file-scope** obsp — the obs axis is shared
/// across modalities, so an obs×obs graph over it is well-defined and the
/// concatenation offsets are the same ones every other obs-axis section uses.
/// It does not carry a file-scope `varp`, for the reason it does not carry a
/// file-scope `varm`: `n_vars` differs per modality, so there is no canonical
/// dimension to stamp on a global var-axis section.
pub(crate) fn merge_obsp_only(
    readers: &[ScxReader],
    writer: &mut ScxWriter,
    offsets: &[u64],
    total_n_obs: u64,
) -> Result<()> {
    warn_dropped_per_modality_pairwise(readers);
    warn_dropped_csr_backed_obsp(readers);
    merge_obsp(readers, writer, offsets, total_n_obs)
}

/// The obs×obs half: every input's graph, rebased and re-emitted as the next
/// output shard.
fn merge_obsp(
    readers: &[ScxReader],
    writer: &mut ScxWriter,
    offsets: &[u64],
    total_n_obs: u64,
) -> Result<()> {
    for key in global_pairwise_keys(
        readers,
        "obsp",
        SectionType::ObspEmbedding,
        SectionType::ObspEmbeddingShard,
    ) {
        // Presence is validated across every input before a byte is written,
        // for the same reason `LayerMissing` is: a graph one input lacks is a
        // graph the merged file cannot honestly claim to have.
        for (idx, reader) in readers.iter().enumerate() {
            if pairwise_entries(
                reader,
                "obsp",
                &key,
                SectionType::ObspEmbedding,
                SectionType::ObspEmbeddingShard,
            )
            .is_empty()
            {
                return Err(OpsError::DenseMappingMissing {
                    axis: "obsp",
                    key,
                    file_index: idx,
                    total: readers.len(),
                });
            }
        }

        validate_obsp_value_schemas(readers, &key)?;

        let mut out_shard_idx: u32 = 0;
        for (idx, reader) in readers.iter().enumerate() {
            let offset = offsets[idx];
            // `old_to_new` is the identity shifted by this input's offset: a
            // concatenating merge drops no rows, so every endpoint survives.
            let old_to_new: Vec<i64> = (0..reader.n_obs()).map(|r| (offset + r) as i64).collect();

            for entry in pairwise_entries(
                reader,
                "obsp",
                &key,
                SectionType::ObspEmbedding,
                SectionType::ObspEmbeddingShard,
            ) {
                let batch = reader
                    .read_dense_mapping_entry(entry)
                    .map_err(OpsError::Format)?;
                let remapped = remap_obsp_coo_to_dim(&batch, &old_to_new, total_n_obs as i64)?;
                // The row range this shard covers in the *merged* space. For a
                // legacy single-section input that is the input's whole axis;
                // for a sharded one it is that shard's range, shifted.
                let (row_start, n_shard_rows) = shard_row_range(&batch, reader.n_obs());
                writer.write_obsp_shard_coo(
                    &key,
                    out_shard_idx,
                    offset + row_start,
                    n_shard_rows,
                    total_n_obs,
                    &remapped,
                )?;
                out_shard_idx += 1;
            }
        }
    }
    Ok(())
}

/// Refuse a key whose `data` column disagrees across inputs, before any of it
/// is written.
///
/// The obsm path has had this guard since Phase 3b
/// (`validate_dense_mapping_schemas`) for exactly this reason: without it a
/// merge *succeeds* and the failure surfaces later, on read, in a file the user
/// now has to throw away. The obsp carry shipped with the missing-key half of
/// that parity and none of the schema half.
///
/// Only the `data` field is compared, and that is deliberate rather than lazy.
/// `remap_obsp_coo_to_dim` **rebuilds** `row`/`col` at a width chosen from the
/// merged dimension, so inputs that disagree there are normalised on the way
/// through and comparing them would reject merges that are actually fine. It
/// preserves `data`'s dtype *and* nullability, so those are what can still
/// differ between two shards written under one key — and
/// `read_all_obsp` → `concat_batches` requires one shared schema, where arrow's
/// `Schema` equality counts nullability.
///
/// Not `schema_field_diff` (the obsm helper) for the same two reasons: it would
/// compare the coordinate columns, and it does not compare nullability at all.
///
/// Found independently by codex - gpt-5.6-sol and Cursor Agent - Grok 4.6 High.
fn validate_obsp_value_schemas(readers: &[ScxReader], key: &str) -> Result<()> {
    if readers.len() < 2 {
        return Ok(());
    }
    let value_field = |reader: &ScxReader| -> Result<Option<(arrow::datatypes::DataType, bool)>> {
        let Some(entry) = pairwise_entries(
            reader,
            "obsp",
            key,
            SectionType::ObspEmbedding,
            SectionType::ObspEmbeddingShard,
        )
        .into_iter()
        .next() else {
            return Ok(None);
        };
        let batch = reader
            .read_dense_mapping_entry(entry)
            .map_err(OpsError::Format)?;
        Ok(batch
            .schema()
            .column_with_name("data")
            .map(|(_, f)| (f.data_type().clone(), f.is_nullable())))
    };

    let Some(first) = value_field(&readers[0])? else {
        return Ok(());
    };
    for (idx, reader) in readers.iter().enumerate().skip(1) {
        let Some(other) = value_field(reader)? else {
            continue;
        };
        if other != first {
            return Err(OpsError::DenseMappingMismatch {
                axis: "obsp",
                key: key.to_string(),
                detail: format!(
                    "input 0 vs input {idx}: 'data' column is {:?} (nullable={}) vs {:?} \
                     (nullable={}). The merged shards would be written under one key and \
                     `read_all_obsp` concatenates them under a single schema, so the graph \
                     would be unreadable. Cast them to one dtype before merging.",
                    first.0, first.1, other.0, other.1,
                ),
            });
        }
    }
    Ok(())
}

/// The var-axis half on its own, for the sorted-merge path.
///
/// A sorted merge refuses obsp (see the module doc) but `varp` is indexed by
/// `var`, which a sort does not touch — so refusing it too would be dropping
/// data for a reason that does not apply to it.
pub(crate) fn carry_varp_only(readers: &[ScxReader], writer: &mut ScxWriter) -> Result<()> {
    warn_dropped_per_modality_pairwise(readers);
    warn_dropped_csr_backed_obsp(readers);
    carry_varp_from_input_zero(readers, writer)
}

/// The var×var half: input 0's graphs, unchanged.
///
/// Not `Verbatim` in the carry table but `Conditional`, and the distinction is
/// load-bearing: a key only a later input carries is not carried, exactly as for
/// `varm`. Declaring it `Verbatim` is what turned `merge([no_varm, has_varm])`
/// into a hard audit failure on code that had always worked.
fn carry_varp_from_input_zero(readers: &[ScxReader], writer: &mut ScxWriter) -> Result<()> {
    let head = &readers[..1];
    for key in global_pairwise_keys(
        head,
        "varp",
        SectionType::VarpEmbedding,
        SectionType::VarpEmbeddingShard,
    ) {
        let entries = pairwise_entries(
            &readers[0],
            "varp",
            &key,
            SectionType::VarpEmbedding,
            SectionType::VarpEmbeddingShard,
        );
        let n_vars = readers[0].header().n_vars;
        for (out_shard_idx, entry) in (0u32..).zip(entries) {
            let batch = readers[0]
                .read_dense_mapping_entry(entry)
                .map_err(OpsError::Format)?;
            let (row_start, n_shard_rows) = shard_row_range(&batch, n_vars);
            writer.write_varp_shard_coo(
                &key,
                out_shard_idx,
                row_start,
                n_shard_rows,
                n_vars,
                &batch,
            )?;
        }
    }
    Ok(())
}

/// The union of file-scope pairwise keys across `readers`, sorted.
///
/// The **union**, not input 0's — that asymmetry is the other half of §6.4. A
/// key only a later input has used to be invisible; now it is considered, and
/// (for obsp) refused if some input lacks it.
fn global_pairwise_keys(
    readers: &[ScxReader],
    prefix: &str,
    single: SectionType,
    shard: SectionType,
) -> Vec<String> {
    let path_prefix = format!("{prefix}/");
    let mut keys: BTreeSet<String> = BTreeSet::new();
    for reader in readers {
        for entry in &reader.catalog().entries {
            if entry.modality_id != 0 {
                continue;
            }
            let Some(stem) = entry.name.strip_prefix(&path_prefix) else {
                continue;
            };
            if entry.section_type == shard {
                // The index is the FINAL segment, so `rfind` — a key may itself
                // contain `_shard_`. A non-numeric tail means this is not a
                // shard of anything (the reader skips it too), so no key.
                if let Some(pos) = stem.rfind("_shard_") {
                    if stem[pos + "_shard_".len()..].parse::<u32>().is_ok() {
                        keys.insert(stem[..pos].to_string());
                    }
                }
            } else if entry.section_type == single {
                // No `_shard_` exclusion. The section *type* already says this
                // is a legacy single section, and `read_all_sharded_or_single`
                // takes its stem verbatim — so excluding names containing
                // `_shard_` (as the obsm helper still does) silently drops a key
                // the reader will happily list and return.
                keys.insert(stem.to_string());
            }
        }
    }
    keys.into_iter().collect()
}

/// One key's file-scope entries in one input, in shard order.
///
/// Sharded entries win over a legacy single section of the same name, matching
/// `read_all_sharded_or_single` — a file carrying both is malformed, and
/// silently emitting the graph twice would be worse than picking one.
///
/// ⚠️ **The suffix must parse as a `u32`, not merely be present.** A bare
/// `starts_with("obsp/foo_shard_")` also matches `obsp/foo_shard_bar_shard_0`,
/// which belongs to the key `foo_shard_bar` — so looking up `foo` swallowed
/// another key's shard, and both landed under `foo` with overlapping row covers
/// that fail on read. `read_sharded_layout_by_prefix` has always parsed the
/// suffix; this now matches it exactly rather than approximating it.
///
/// Found by codex - gpt-5.6-sol.
fn pairwise_entries<'a>(
    reader: &'a ScxReader,
    prefix: &str,
    key: &str,
    single: SectionType,
    shard: SectionType,
) -> Vec<&'a FullCatalogEntry> {
    let shard_prefix = format!("{prefix}/{key}_shard_");
    let mut shards: Vec<(u32, &FullCatalogEntry)> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.modality_id == 0 && e.section_type == shard)
        .filter_map(|e| {
            let suffix = e.name.strip_prefix(&shard_prefix)?;
            let idx: u32 = suffix.parse().ok()?;
            Some((idx, e))
        })
        .collect();
    if !shards.is_empty() {
        shards.sort_by_key(|(idx, _)| *idx);
        return shards.into_iter().map(|(_, e)| e).collect();
    }
    let legacy_name = format!("{prefix}/{key}");
    reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.modality_id == 0 && e.section_type == single && e.name == legacy_name)
        .collect()
}

/// A pairwise shard's `(row_start, n_shard_rows)` in its own file's space.
///
/// `write_obsp_shard_coo` stamps these into the schema, so a sharded input
/// carries them; a legacy single section does not, and covers the whole axis.
/// Falling back to the whole axis rather than erroring is deliberate: the values
/// only describe the cover, and a legacy section genuinely is the whole cover.
fn shard_row_range(batch: &arrow::array::RecordBatch, axis_len: u64) -> (u64, u64) {
    let md = batch.schema();
    let md = md.metadata();
    let get = |k: &str| md.get(k).and_then(|v| v.parse::<u64>().ok());
    match (get("row_start"), get("n_shard_rows")) {
        (Some(start), Some(n)) => (start, n),
        _ => (0, axis_len),
    }
}
