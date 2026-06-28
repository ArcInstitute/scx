//! In-place metadata replacement.
//!
//! Replace SCX metadata sections (`uns` / `obs` / `var` / `obsm` / `varm`) of an
//! existing `.scx` file **without re-encoding `X`**. The op appends fresh
//! section bytes at EOF and atomically repoints the catalog — cost is O(size of
//! the replaced sections), the CSR/CSC shards are never read or rewritten.
//!
//! Because the matrix is untouched, `data_generation` and `csc_build_generation`
//! are left unchanged, so a pre-existing CSC sidecar stays valid (no
//! `--rebuild-csc`). `n_obs` / `n_vars` / `nnz` / `HAS_CSC` are invariants and
//! are validated rather than changed — to add cells/genes use `append`,
//! `subset`, or `from_*`.
//!
//! Replace semantics, **not merge**: a supplied field fully replaces the
//! existing section. For a shallow `uns` merge, read-modify-write in the caller.
//!
//! Multimodal (`modality_id != 0`) is not yet supported and returns
//! [`OpsError::MultimodalUnsupported`]; per-modality metadata replace is a
//! focused follow-on.

use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use serde_json::Value;

use scx_engine::ConversionPredicateIndexOptions;
use scx_format_io::catalog::{ColumnStat, FullCatalog, FullCatalogEntry};
use scx_format_io::checksum::blake3_hash;
use scx_format_io::provenance::{Provenance, ProvenanceEntry};
use scx_format_io::section::{write_alignment_padding, SectionType};
use scx_format_io::writer::ScxWriter;

use crate::append::{predicate_index_build_options_for_obs, unify_dict_columns};
use crate::error::{OpsError, Result};
use crate::flock::FileLock;
use crate::in_place::{commit_in_place, prepare_in_place};
use crate::predicate_index::{user_wants_index, validate_forced_columns};

/// A set of metadata replacements to apply atomically. Any `None` field is left
/// untouched (its existing catalog entries pass through verbatim). `obsm` /
/// `varm` replace only the named matrices; other keys pass through.
#[derive(Default)]
pub struct MetadataPatch {
    /// Replaces the whole `UnsBlob` section.
    pub uns: Option<Value>,
    /// Replaces obs metadata; `num_rows` must equal the file's `n_obs`.
    pub obs: Option<RecordBatch>,
    /// Replaces var metadata; `num_rows` must equal the file's `n_vars`.
    pub var: Option<RecordBatch>,
    /// Replace named obsm matrices; each `num_rows` must equal `n_obs`.
    pub obsm: Option<Vec<(String, RecordBatch)>>,
    /// Replace named varm matrices; each `num_rows` must equal `n_vars`.
    pub varm: Option<Vec<(String, RecordBatch)>>,
    /// Predicate-index rebuild policy (only consulted when `obs`/`var` change).
    pub index: ConversionPredicateIndexOptions,
    /// Modality to target. `0` = global / single-modality. Non-zero is not yet
    /// supported.
    pub modality_id: u8,
}

impl MetadataPatch {
    fn is_empty(&self) -> bool {
        self.uns.is_none()
            && self.obs.is_none()
            && self.var.is_none()
            && self.obsm.as_ref().is_none_or(|v| v.is_empty())
            && self.varm.as_ref().is_none_or(|v| v.is_empty())
    }
}

/// Apply `patch` to the file at `path` in place. O(size of replaced sections);
/// `X`/CSR shards are never read or rewritten. Atomic: a single header write
/// commits, and the change is rollback-able via the catalog chain.
pub fn modify_metadata(path: &Path, patch: &MetadataPatch) -> Result<()> {
    if patch.is_empty() {
        return Err(OpsError::InvalidInput(
            "modify_metadata: empty patch (set at least one of uns/obs/var/obsm/varm)".to_string(),
        ));
    }

    let (mut lock, mut prep) = prepare_in_place(path, patch.modality_id)?;

    // Per-modality metadata replace is deferred — bail before any write so the
    // file is left byte-identical.
    if patch.modality_id != 0 {
        return Err(OpsError::MultimodalUnsupported {
            op: "modify_metadata",
        });
    }

    // --- Shape validation (before any write) -------------------------------
    if let Some(obs) = &patch.obs {
        if obs.num_rows() as u64 != prep.old_n_obs {
            return Err(OpsError::ShapeMismatch {
                detail: format!(
                    "modify_metadata: obs has {} rows but file has n_obs={} \
                     (changing cell count is out of scope — use append/subset)",
                    obs.num_rows(),
                    prep.old_n_obs
                ),
            });
        }
    }
    if let Some(var) = &patch.var {
        if var.num_rows() as u64 != prep.target_n_vars {
            return Err(OpsError::ShapeMismatch {
                detail: format!(
                    "modify_metadata: var has {} rows but file has n_vars={} \
                     (changing gene count is out of scope)",
                    var.num_rows(),
                    prep.target_n_vars
                ),
            });
        }
    }
    if let Some(obsm) = &patch.obsm {
        for (k, b) in obsm {
            if b.num_rows() as u64 != prep.old_n_obs {
                return Err(OpsError::ShapeMismatch {
                    detail: format!(
                        "modify_metadata: obsm['{k}'] has {} rows but n_obs={}",
                        b.num_rows(),
                        prep.old_n_obs
                    ),
                });
            }
        }
    }
    if let Some(varm) = &patch.varm {
        for (k, b) in varm {
            if b.num_rows() as u64 != prep.target_n_vars {
                return Err(OpsError::ShapeMismatch {
                    detail: format!(
                        "modify_metadata: varm['{k}'] has {} rows but n_vars={}",
                        b.num_rows(),
                        prep.target_n_vars
                    ),
                });
            }
        }
    }

    // Predicate indexes are rebuilt only when the underlying axis is replaced
    // AND the caller asked for an index. A stale index over changed values is
    // always dropped (see `should_drop_old_entry`).
    let rebuild_obs_index = patch.obs.is_some() && user_wants_index(&patch.index);
    let rebuild_var_index = patch.var.is_some() && user_wants_index(&patch.index);
    if rebuild_obs_index || rebuild_var_index {
        let mut vopts = patch.index.clone();
        if patch.obs.is_none() {
            vopts.index_obs.clear();
        }
        if patch.var.is_none() {
            vopts.index_var.clear();
        }
        let empty = Schema::empty();
        let obs_schema = patch.obs.as_ref().map(|b| b.schema());
        let var_schema = patch.var.as_ref().map(|b| b.schema());
        validate_forced_columns(
            &vopts,
            obs_schema.as_deref().unwrap_or(&empty),
            var_schema.as_deref().unwrap_or(&empty),
        )?;
    }

    // --- Capture invariant fields before consuming prep.old_catalog --------
    let old_catalog_offset = prep.old_catalog_offset;
    let data_gen = prep.old_catalog.data_generation;
    let csc_gen = prep.old_catalog.csc_build_generation;
    let manifest = prep.header.manifest_sequence;
    let mt_off = prep.header.modality_table_offset;
    let mt_len = prep.header.modality_table_length;
    let shard_target_rows = (prep.header.shard_target_rows as usize).max(1);

    // CSR shard (modality 0) row ranges, sorted — the index is built over these
    // so query-time shard skipping aligns with the matrix shards.
    let mut csr_ranges: Vec<(u64, u64)> = prep
        .old_catalog
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CsrShard && e.modality_id == 0)
        .filter_map(|e| e.stats.as_ref().map(|s| (s.row_start, s.row_end)))
        .collect();
    csr_ranges.sort_by_key(|(s, _)| *s);

    // Read existing provenance ops through the lock before adopting the writer.
    let mut prov_ops = read_provenance_ops(&mut lock, &prep.old_catalog)?;

    // --- Emit replacement sections at EOF via an adopted writer ------------
    let write_offset = lock.seek(SeekFrom::End(0))?;
    let cloned = lock.file().try_clone()?;
    let mut writer =
        ScxWriter::adopt_in_place(cloned, prep.header.clone(), write_offset, Vec::new())?;

    if let Some(uns) = &patch.uns {
        writer.write_uns(uns)?;
    }
    if let Some(var) = &patch.var {
        writer.write_var(var)?;
    }

    let mut per_shard_obs_stats: Option<Vec<Vec<ColumnStat>>> = None;
    let mut indexed_obs_cols: Vec<String> = Vec::new();
    let mut indexed_var_cols: Vec<String> = Vec::new();

    if let Some(obs) = &patch.obs {
        // Unify dictionary columns to their value type, matching the obs
        // sharding the append/merge paths perform.
        let unified = unify_dict_columns(obs)?;
        let mut builder = if rebuild_obs_index {
            Some(
                scx_engine::ObsPredicateIndexBuilder::new(
                    unified.schema(),
                    &predicate_index_build_options_for_obs(&patch.index),
                )
                .map_err(OpsError::Engine)?,
            )
        } else {
            None
        };

        let n = unified.num_rows();
        let mut obs_shard_ranges: Vec<(u64, u64)> = Vec::new();
        let mut cursor = 0usize;
        let mut idx = 0u32;
        while cursor < n {
            let take = std::cmp::min(shard_target_rows, n - cursor);
            let chunk = unified.slice(cursor, take);
            let row_start = cursor as u64;
            if let Some(b) = builder.as_mut() {
                b.push_shard(&chunk, row_start).map_err(OpsError::Engine)?;
            }
            writer.write_obs_shard(idx, row_start, take as u64, prep.old_n_obs, &chunk)?;
            obs_shard_ranges.push((row_start, row_start + take as u64));
            idx += 1;
            cursor += take;
        }

        if let Some(builder) = builder {
            // Finish over CSR shard ranges so the index's shard space matches
            // the matrix shards. Fall back to the obs-shard ranges unless the
            // CSR ranges fully cover [0, n_obs): a partial range list (some
            // shards missing stats on older files) would misalign the index's
            // shard space with the matrix.
            let csr_covers_all = !csr_ranges.is_empty()
                && csr_ranges.iter().map(|(s, e)| e - s).sum::<u64>() == prep.old_n_obs;
            let use_csr = csr_covers_all;
            let ranges: &[(u64, u64)] = if use_csr {
                &csr_ranges
            } else {
                &obs_shard_ranges
            };
            let mut outcomes = Vec::new();
            let obs_bytes = builder
                .finish(ranges, &mut outcomes, &mut indexed_obs_cols)
                .map_err(OpsError::Engine)?;
            if let Some(bytes) = obs_bytes {
                writer.write_obs_predicate_index(&bytes)?;
                if use_csr {
                    let index = scx_engine::PredicateIndex::read_from(&mut Cursor::new(&bytes))
                        .map_err(OpsError::Engine)?;
                    per_shard_obs_stats = Some(scx_engine::derive_shard_column_stats(
                        &index,
                        csr_ranges.len(),
                    ));
                }
            }
        }
    }

    if rebuild_var_index {
        let var = patch
            .var
            .as_ref()
            .expect("rebuild_var_index implies patch.var is Some");
        let preset_var = match patch.index.index_preset.as_deref() {
            Some(name) => scx_engine::index_preset_columns(name)
                .map(|p| p.var_columns.iter().map(|s| (*s).to_string()).collect())
                .unwrap_or_default(),
            None => Vec::new(),
        };
        let var_build_opts = scx_engine::PredicateIndexBuildOptions {
            forced_columns: patch.index.index_var.clone(),
            preset_columns: preset_var,
            auto_threshold: patch.index.index_auto_threshold,
            high_cardinality_threshold: 100_000,
        };
        let mut outcomes = Vec::new();
        let var_bytes = scx_engine::build_var_predicate_index_bytes(
            var,
            &[(0, prep.target_n_vars)],
            &var_build_opts,
            &mut outcomes,
            &mut indexed_var_cols,
        )?;
        if let Some(bytes) = var_bytes {
            writer.write_var_predicate_index(&bytes)?;
        }
    }

    if let Some(obsm) = &patch.obsm {
        for (name, b) in obsm {
            writer.write_obsm(name, b)?;
        }
    }
    if let Some(varm) = &patch.varm {
        for (name, b) in varm {
            writer.write_varm(name, b)?;
        }
    }

    let (file, new_offset, new_section_entries) = writer.into_in_place_parts()?;
    drop(file);
    lock.seek(SeekFrom::Start(new_offset))?;

    // --- Append provenance (manual, mirrors finalize_append) ---------------
    let params = build_params_json(patch, &indexed_obs_cols, &indexed_var_cols);
    prov_ops.push(ProvenanceEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        action: "modify_metadata".to_string(),
        tool: concat!("scx-ops ", env!("CARGO_PKG_VERSION")).to_string(),
        params_json: params.to_string(),
        input_checksums: vec![],
    });
    let prov = Provenance {
        version: 1,
        operations: prov_ops,
    };
    let mut prov_bytes = Vec::new();
    prov.write_to(&mut prov_bytes)?;

    lock.seek(SeekFrom::End(0))?;
    let mut woff = lock.stream_position()?;
    let pad = write_alignment_padding(&mut *lock, woff)?;
    woff += pad as u64;
    let prov_offset = woff;
    lock.write_all(&prov_bytes)?;
    let prov_length = prov_bytes.len() as u64;
    let prov_checksum = blake3_hash(&prov_bytes);

    // --- Assemble the new catalog ------------------------------------------
    // Old entries minus the replaced section types (matrix shards + untouched
    // metadata pass through verbatim — their bytes never move). Unlike
    // `append`, CSC shards are NOT dropped: the matrix is unchanged.
    let mut entries: Vec<FullCatalogEntry> = prep
        .old_catalog
        .entries
        .into_iter()
        .filter(|e| !should_drop_old_entry(e, patch))
        .collect();
    entries.extend(new_section_entries);
    entries.push(FullCatalogEntry {
        name: "provenance".to_string(),
        offset: prov_offset,
        length: prov_length,
        section_type: SectionType::Provenance,
        checksum: prov_checksum,
        modality_id: 0,
        stats: None,
    });
    if let Some(per_shard) = per_shard_obs_stats {
        scx_format_io::assign_csr_shard_column_stats(&mut entries, per_shard)?;
    }

    let new_catalog = FullCatalog {
        catalog_version: scx_format_io::CURRENT_CATALOG_VERSION,
        manifest_sequence: manifest + 1,
        prev_catalog_offset: old_catalog_offset,
        n_obs: prep.old_n_obs, // UNCHANGED
        entries,
        data_generation: data_gen,     // UNCHANGED — no X mutation
        csc_build_generation: csc_gen, // UNCHANGED — CSC sidecar stays valid
    };

    commit_in_place(&mut lock, &mut prep.header, &new_catalog, mt_off, mt_len)?;
    Ok(())
}

/// Convenience wrapper: replace the whole `uns` block.
pub fn set_uns(path: &Path, uns: &Value) -> Result<()> {
    modify_metadata(
        path,
        &MetadataPatch {
            uns: Some(uns.clone()),
            ..Default::default()
        },
    )
}

/// Whether an old catalog entry is superseded by `patch` and must be dropped
/// from the new catalog (its bytes become orphans, reclaimable by `scx
/// compact`). Single-modality only — modality 0 entries.
fn should_drop_old_entry(e: &FullCatalogEntry, patch: &MetadataPatch) -> bool {
    use SectionType::*;
    // Provenance is always rewritten.
    if e.section_type == Provenance {
        return true;
    }
    // `set_uns` replaces the *global* uns only. Per-modality uns
    // (`uns/<modality>`, modality_id > 0) is left intact — `MetadataPatch`
    // has no per-modality uns slot, and dropping every UnsBlob would
    // silently erase per-modality metadata written by `from_mudata`.
    if patch.uns.is_some() && e.section_type == UnsBlob && e.modality_id == 0 {
        return true;
    }
    // obs/var changed → drop their metadata sections AND any predicate index
    // (the index covers now-stale values; a fresh one is re-emitted when the
    // caller requests a rebuild).
    if patch.var.is_some()
        && matches!(
            e.section_type,
            VarMetadata | VarMetadataShard | VarPredicateIndex
        )
    {
        return true;
    }
    if patch.obs.is_some()
        && matches!(
            e.section_type,
            ObsMetadata | ObsMetadataShard | ObsPredicateIndex
        )
    {
        return true;
    }
    if let Some(obsm) = &patch.obsm {
        if matches!(e.section_type, ObsmEmbedding | ObsmEmbeddingShard)
            && obsm
                .iter()
                .any(|(k, _)| entry_matches_key(&e.name, "obsm", k))
        {
            return true;
        }
    }
    if let Some(varm) = &patch.varm {
        if matches!(e.section_type, VarmEmbedding | VarmEmbeddingShard)
            && varm
                .iter()
                .any(|(k, _)| entry_matches_key(&e.name, "varm", k))
        {
            return true;
        }
    }
    false
}

/// Match an obsm/varm catalog entry name against a replaced key. Section names
/// are `{prefix}/{key}` (single) or `{prefix}/{key}_shard_{idx}` (sharded).
///
/// Uses `rfind("_shard_")` stem extraction (mirroring `merge`/`compact`) rather
/// than a `starts_with` prefix test, so a key like `pca` does not falsely match
/// the sharded sections of a distinct key `pca_shard` (`{prefix}/pca_shard_shard_0`).
fn entry_matches_key(name: &str, prefix: &str, key: &str) -> bool {
    let Some(rest) = name.strip_prefix(&format!("{prefix}/")) else {
        return false;
    };
    if rest == key {
        return true; // single, unsharded section: {prefix}/{key}
    }
    match rest.rfind("_shard_") {
        Some(pos) => &rest[..pos] == key, // {prefix}/{key}_shard_{idx}
        None => false,
    }
}

/// Read the existing `Provenance` operations (empty if the file has none).
fn read_provenance_ops(
    lock: &mut FileLock,
    old_catalog: &FullCatalog,
) -> Result<Vec<ProvenanceEntry>> {
    if let Some(e) = old_catalog
        .entries
        .iter()
        .find(|e| e.section_type == SectionType::Provenance)
    {
        lock.seek(SeekFrom::Start(e.offset))?;
        let mut buf = vec![0u8; e.length as usize];
        Read::read_exact(lock, &mut buf)?;
        let prov = Provenance::read_from(&mut Cursor::new(&buf), buf.len())?;
        Ok(prov.operations)
    } else {
        Ok(Vec::new())
    }
}

/// Provenance params recording which fields changed + any indexed columns.
fn build_params_json(patch: &MetadataPatch, obs_cols: &[String], var_cols: &[String]) -> Value {
    let mut changed: Vec<&str> = Vec::new();
    if patch.uns.is_some() {
        changed.push("uns");
    }
    if patch.obs.is_some() {
        changed.push("obs");
    }
    if patch.var.is_some() {
        changed.push("var");
    }
    if patch.obsm.as_ref().is_some_and(|v| !v.is_empty()) {
        changed.push("obsm");
    }
    if patch.varm.as_ref().is_some_and(|v| !v.is_empty()) {
        changed.push("varm");
    }
    let mut params = serde_json::json!({ "changed": changed });
    if !obs_cols.is_empty() || !var_cols.is_empty() {
        params["predicate_index"] = serde_json::json!({
            "obs_columns": obs_cols,
            "var_columns": var_cols,
        });
    }
    params
}

#[cfg(test)]
mod tests {
    use super::entry_matches_key;

    #[test]
    fn entry_matches_key_single_and_sharded() {
        // Single, unsharded section.
        assert!(entry_matches_key("obsm/pca", "obsm", "pca"));
        // Sharded sections of the same key.
        assert!(entry_matches_key("obsm/pca_shard_0", "obsm", "pca"));
        assert!(entry_matches_key("obsm/pca_shard_12", "obsm", "pca"));
    }

    #[test]
    fn entry_matches_key_no_prefix_false_positive() {
        // Regression: replacing key `pca` must NOT drop the sharded sections of
        // a distinct key `pca_shard` (`obsm/pca_shard_shard_0`). The old
        // `starts_with("obsm/pca_shard_")` test matched this by accident.
        assert!(!entry_matches_key("obsm/pca_shard_shard_0", "obsm", "pca"));
        // …but it IS the correct match for its own key.
        assert!(entry_matches_key(
            "obsm/pca_shard_shard_0",
            "obsm",
            "pca_shard"
        ));
        // The distinct single section likewise must not match.
        assert!(!entry_matches_key("obsm/pca_shard", "obsm", "pca"));
        // Numeric-suffix sibling key (e.g. `X_pca` vs `X_pca_2`).
        assert!(!entry_matches_key("obsm/X_pca_2_shard_0", "obsm", "X_pca"));
    }

    #[test]
    fn entry_matches_key_respects_axis_prefix() {
        // A varm section never matches an obsm replacement and vice versa.
        assert!(!entry_matches_key("varm/pca", "obsm", "pca"));
        assert!(!entry_matches_key("varm/pca_shard_0", "obsm", "pca"));
        assert!(entry_matches_key("varm/pca_shard_0", "varm", "pca"));
    }
}
