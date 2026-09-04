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
//! existing section. The one exception is opt-in: [`MetadataPatch::uns_merge`]
//! (surfaced as [`update_uns`], `pyscx.update_uns`, `scx set-uns --merge`)
//! shallow-merges the `uns` patch's top-level keys into the existing block.
//!
//! A replaced axis **keeps the predicate index it had**: the old section can
//! never survive verbatim (its shard ranges describe values that are gone), so
//! the op rebuilds one over the same columns the file already indexed unless the
//! caller names their own. Before that, replacing obs dropped the index and the
//! per-shard column stats with it — silently reverting `filter_obs` pushdown to
//! a full scan on the last step of the doublet workflow, which is a wholesale
//! obs replacement.
//!
//! Multimodal (`modality_id != 0`) is not yet supported and returns
//! [`OpsError::MultimodalUnsupported`]; per-modality metadata replace is a
//! focused follow-on.

use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use serde_json::Value;

use scx_engine::{BuildOutcome, ConversionPredicateIndexOptions, ConversionPredicateIndexResult};
use scx_format_io::catalog::{ColumnStat, FullCatalog, FullCatalogEntry};
use scx_format_io::checksum::blake3_hash;
use scx_format_io::provenance::{Provenance, ProvenanceEntry};
use scx_format_io::reader::ScxReader;
use scx_format_io::section::{write_alignment_padding, SectionType};
use scx_format_io::writer::ScxWriter;

use crate::error::{OpsError, Result};
use crate::external_obs::indexed_column_names;
use crate::in_place::{
    commit_in_place, entry_matches_key, prepare_in_place, read_provenance_ops, read_uns_blob,
};
use crate::predicate_index::{
    user_wants_index, validate_forced_columns, ObsVarIndexPass, PredicateIndexBuildSummary,
    StatsSink,
};

/// A set of metadata replacements to apply atomically. Any `None` field is left
/// untouched (its existing catalog entries pass through verbatim). `obsm` /
/// `varm` replace only the named matrices; other keys pass through.
#[derive(Default)]
pub struct MetadataPatch {
    /// Replaces the whole `UnsBlob` section — or, with [`Self::uns_merge`],
    /// is shallow-merged into it.
    pub uns: Option<Value>,
    /// When true, `uns` is a **shallow patch**: its top-level keys are merged
    /// into the file's existing `uns` — a patch key replaces a same-named key
    /// wholesale (no deep merge), every other key survives verbatim, and
    /// `null` sets null rather than deleting. Requires `uns` to be a JSON
    /// object, and the existing `uns` to be one too (or absent). False
    /// (default): `uns` replaces the whole block. See [`update_uns`].
    pub uns_merge: bool,
    /// Replaces obs metadata. `num_rows` must equal either the file's physical
    /// `n_obs` (`header.n_obs`, deleted rows in place — what
    /// `read_obs(logical=False)` returns) or its **live** row count (`n_obs`
    /// minus the deletion-vector popcount — what `read_obs()` returns since
    /// pyscx 0.17). A live-length frame is scattered onto the physical axis
    /// before it is written: a deleted row keeps its obs-index value (barcode)
    /// from the existing obs and is `null` in every other column, since the
    /// caller never saw it. The two lengths coincide on a file with no
    /// deletions. See [`modify_metadata`].
    pub obs: Option<RecordBatch>,
    /// Replaces var metadata; `num_rows` must equal the file's `n_vars`.
    pub var: Option<RecordBatch>,
    /// Replace named obsm matrices; each `num_rows` must equal `n_obs`.
    pub obsm: Option<Vec<(String, RecordBatch)>>,
    /// Replace named varm matrices; each `num_rows` must equal `n_vars`.
    pub varm: Option<Vec<(String, RecordBatch)>>,
    /// Predicate-index rebuild policy (only consulted when `obs`/`var` change).
    ///
    /// Empty means "keep what the file already indexes" — the replaced axis's
    /// index is rebuilt over its own columns. Naming columns here overrides
    /// that, and narrowing the set is reported rather than silent
    /// ([`ModifyMetadataSummary::obs_columns_not_carried`]).
    pub index: ConversionPredicateIndexOptions,
    /// Modality to target. `0` = global / single-modality. Non-zero is not yet
    /// supported.
    pub modality_id: u8,
}

/// Why an obs replacement could not re-derive the per-shard column statistics.
///
/// Only used to word the warning, but the wording matters: two of these three are
/// *not* fixed by passing `--index-obs`, and an earlier message told every caller
/// to do exactly that. On an under-covering file that advice loops forever.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NoStatsReason {
    /// No index was built, so there was nothing to derive from. Since a
    /// replaced axis carries its existing index forward, this now means the
    /// file had no index on that axis to carry — not merely that the caller
    /// omitted `--index-obs`.
    NoRebuildRequested,
    /// A rebuild ran and an index was written — but over the obs-shard ranges,
    /// because the CSR shards do not tile `[0, n_obs)`. `derive_shard_column_stats`
    /// would then be addressing a different shard space, so it is skipped.
    CsrRangesUnderCoverObsAxis,
    /// A rebuild was requested but no requested column could be indexed (a
    /// missing column, or an unsupported dtype such as `Boolean`).
    NoColumnWasIndexable,
}

impl NoStatsReason {
    fn what_happened(self) -> &'static str {
        match self {
            Self::NoRebuildRequested => {
                "obs was replaced and this file has no obs predicate index to carry forward"
            }
            Self::CsrRangesUnderCoverObsAxis => {
                "the predicate index was rebuilt, but this file's CSR shards do not cover \
                 every obs row, so the index could not be mapped onto them"
            }
            Self::NoColumnWasIndexable => {
                "an index rebuild was requested but no requested column could be indexed \
                 (missing, or an unsupported dtype)"
            }
        }
    }

    fn remedy(self) -> &'static str {
        match self {
            Self::NoRebuildRequested => {
                "pass --index-obs / --index-preset (or index_obs=/index_preset= in pyscx) \
                 to re-derive them."
            }
            // Deliberately NOT "--index-obs": that is what was just done, and on a
            // file with this shape it will skip the derive again.
            Self::CsrRangesUnderCoverObsAxis => {
                "another in-place rebuild will not help; rewrite the file with `scx sort` \
                 or `scx compact` plus --index-obs / --index-preset, which re-shards the \
                 matrix so its CSR shards cover the whole obs axis."
            }
            Self::NoColumnWasIndexable => {
                "name a column the index supports (a categorical or numeric obs column) \
                 in --index-obs / --index-preset."
            }
        }
    }
}

/// What [`modify_metadata`] did to the file's predicate indexes.
///
/// Replacing an axis wholesale used to drop its index outright and say nothing,
/// because the op cannot tell an *added* column from a *rewritten* one. It now
/// carries the index forward — rebuilds it over the columns it already covered —
/// and hands back what it could and could not carry, so the loss stops being
/// silent on the paths where one is unavoidable.
#[derive(Debug, Default)]
pub struct ModifyMetadataSummary {
    /// Per-column build outcomes in the same shape every other rewrite op
    /// reports them, so a caller can reuse one handler
    /// (`ForcedColumnError` → error, `PresetSkipped` → warning).
    pub index: PredicateIndexBuildSummary,
    /// Obs columns the file's index covered that the new index does not.
    ///
    /// On the carry-forward path each of these also has a `PresetSkipped`
    /// outcome carrying the precise reason; on the explicit-request path the
    /// builder knows nothing about the old index, so this list is the only
    /// signal. Callers that report both should skip a column already named by
    /// an outcome rather than warn about it twice.
    pub obs_columns_not_carried: Vec<String>,
    /// The var-axis counterpart of [`Self::obs_columns_not_carried`].
    pub var_columns_not_carried: Vec<String>,
    /// True when this op rebuilt the obs index the caller did not ask for,
    /// purely to keep the one the file already had, **and that rebuild
    /// produced an index**.
    ///
    /// Success-gated, and per axis, for two reasons a single attempted-carry
    /// flag got wrong: a carry whose every column turned out unindexable would
    /// report `carried_forward` beside `obs_predicate_index_dropped` on the
    /// same provenance entry, and one axis carrying while the other is
    /// explicitly rebuilt is reachable now that "explicit" is decided per axis.
    pub obs_carried_forward: bool,
    /// The var-axis counterpart of [`Self::obs_carried_forward`].
    pub var_carried_forward: bool,
    /// True when the file had an obs predicate index and the output has none.
    pub obs_predicate_index_dropped: bool,
    /// True when the file had a var predicate index and the output has none.
    pub var_predicate_index_dropped: bool,
}

impl ModifyMetadataSummary {
    /// [`Self::obs_columns_not_carried`] minus the columns a build outcome
    /// already names — i.e. the ones a caller must report itself.
    ///
    /// The two channels overlap by construction: on the carry path every
    /// column that could not be carried also has a `PresetSkipped` outcome
    /// carrying the precise reason, while on the explicit-request path the
    /// builder never sees the old index and emits nothing. A front end that
    /// renders both without this filter warns twice about one column — which
    /// is what the CLI did the moment it started rendering outcomes at all.
    /// Living here rather than in each front end is what keeps pyscx and the
    /// CLI from drifting on it.
    pub fn obs_not_carried_unreported(&self) -> Vec<String> {
        self.unreported(&self.obs_columns_not_carried, |r| &r.obs_outcomes)
    }

    /// The var-axis counterpart of [`Self::obs_not_carried_unreported`].
    pub fn var_not_carried_unreported(&self) -> Vec<String> {
        self.unreported(&self.var_columns_not_carried, |r| &r.var_outcomes)
    }

    fn unreported(
        &self,
        columns: &[String],
        pick: fn(&ConversionPredicateIndexResult) -> &[BuildOutcome],
    ) -> Vec<String> {
        let Some(result) = self.index.result.as_ref() else {
            return columns.to_vec();
        };

        let outcomes = pick(result);
        columns
            .iter()
            .filter(|c| {
                !outcomes.iter().any(|o| {
                    let named = match o {
                        BuildOutcome::ForcedColumnError { column, .. } => column,
                        BuildOutcome::PresetSkipped { column, .. } => column,
                    };
                    named == *c
                })
            })
            .cloned()
            .collect()
    }
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

/// Describe an index request `modify_metadata` cannot act on, if any.
///
/// Returns `None` when every index option given has an axis to apply to.
fn unhonourable_index_request(patch: &MetadataPatch) -> Option<String> {
    let idx = &patch.index;
    if !idx.index_obs.is_empty() && patch.obs.is_none() {
        return Some(format!(
            "index_obs={:?} was requested but no obs frame was supplied",
            idx.index_obs
        ));
    }
    if !idx.index_var.is_empty() && patch.var.is_none() {
        return Some(format!(
            "index_var={:?} was requested but no var frame was supplied",
            idx.index_var
        ));
    }
    // Cross-axis knobs need only *one* axis to be replaced to do something.
    if patch.obs.is_none() && patch.var.is_none() {
        if let Some(preset) = &idx.index_preset {
            return Some(format!(
                "index_preset={preset:?} was requested but neither obs nor var was supplied"
            ));
        }
        if idx.index_auto_threshold > 0 {
            return Some(format!(
                "index_auto_threshold={} was requested but neither obs nor var was supplied",
                idx.index_auto_threshold
            ));
        }
    }
    None
}

/// Apply `patch` to the file at `path` in place. O(size of replaced sections);
/// `X`/CSR shards are never read or rewritten. Atomic: a single header write
/// commits, and the change is rollback-able via the catalog chain.
pub fn modify_metadata(path: &Path, patch: &MetadataPatch) -> Result<ModifyMetadataSummary> {
    if patch.is_empty() {
        return Err(OpsError::InvalidInput(
            "modify_metadata: empty patch (set at least one of uns/obs/var/obsm/varm)".to_string(),
        ));
    }

    // An index request the op cannot honour is rejected here rather than
    // ignored. Index rebuilds are gated on the axis actually being replaced
    // (`rebuild_obs_index` / `rebuild_var_index` below), so `--index-obs
    // cell_type` with no `--obs` used to exit 0, print "Updated metadata..."
    // and build nothing at all.
    //
    // Decided per axis, because the knobs are not all axis-scoped:
    // `index_preset` and `index_auto_threshold` genuinely span both, so they
    // are only unhonourable when *neither* axis is being replaced. That
    // asymmetry is why this cannot be one `user_wants_index` check.
    if let Some(detail) = unhonourable_index_request(patch) {
        return Err(OpsError::InvalidInput(format!(
            "modify_metadata: {detail}. An index is rebuilt only over an axis this \
             call replaces, so the request would have been silently ignored; \
             pass the matching obs=/var= frame, or drop the index option."
        )));
    }

    if patch.uns_merge {
        match &patch.uns {
            None => {
                return Err(OpsError::InvalidInput(
                    "modify_metadata: uns_merge=true needs a uns patch to merge".to_string(),
                ))
            }
            Some(v) if !v.is_object() => {
                return Err(OpsError::InvalidInput(format!(
                    "modify_metadata: a uns merge patch must be a JSON object (its top-level \
                     keys are merged); got {}. To replace uns with a non-object, use set_uns.",
                    json_kind(v)
                )))
            }
            Some(_) => {}
        }
    }

    let (mut lock, mut prep) = prepare_in_place(path, patch.modality_id)?;

    // Per-modality metadata replace is deferred — bail before any write so the
    // file is left byte-identical.
    if patch.modality_id != 0 {
        return Err(OpsError::MultimodalUnsupported {
            op: "modify_metadata",
        });
    }

    // --- Resolve the uns payload ------------------------------------------
    // A merge reads the existing blob through the lock (no mmap — see
    // `read_uns_blob`) and lays the patch's top-level keys over it. Still
    // before any write, so a non-object existing `uns` is refused with the
    // file byte-identical.
    let uns_to_write: Option<Value> = match &patch.uns {
        Some(uns) if patch.uns_merge => {
            let entries = uns
                .as_object()
                .expect("uns_merge patch validated as an object above");
            let existing = read_uns_blob(&mut lock, &prep.old_catalog)?
                .unwrap_or_else(|| serde_json::json!({}));
            Some(crate::external_obs::merge_uns_entries(existing, entries)?)
        }
        Some(uns) => Some(uns.clone()),
        None => None,
    };

    // --- What does the file already index? ---------------------------------
    // Only when an axis is actually being replaced: a `uns`-only patch is the
    // headline cheap case and must not pay for an mmap + index parse.
    //
    // The reader takes no lock of its own; we only ever append, never truncate,
    // so holding it across the transaction is safe (same reasoning as
    // `attach_external_obs`). Opened before the shape gate because an obs
    // replacement needs the deletion keep mask to tell a live-length frame
    // from a wrong one.
    let reader = if patch.obs.is_some() || patch.var.is_some() {
        Some(ScxReader::open(path)?)
    } else {
        None
    };

    // --- Shape validation (before any write) -------------------------------
    //
    // `obs` is accepted in either row space. Physical (`header.n_obs` rows) is
    // written as handed in. Live (`n_obs - n_deleted` rows, what `read_obs()`
    // returns) is scattered onto the physical axis here — BEFORE the index
    // builder and the shard loop below, whose row ids and CSR ranges are
    // physical; scattering after them would index live rows against physical
    // shards and silently shorten every `filter_obs` on the new columns.
    let obs_to_write: Option<RecordBatch> = match &patch.obs {
        None => None,
        Some(obs) => {
            let reader = reader
                .as_ref()
                .expect("reader is opened whenever patch.obs is Some");
            Some(resolve_obs_row_space(reader, obs, prep.old_n_obs)?)
        }
    };
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
                        "modify_metadata: obsm['{k}'] has {} rows but n_obs_physical={}. \
                         A dense mapping is always physical-length (it has no null to \
                         stand in for a deleted row), so unlike `obs` a live-length \
                         array is not accepted here; pass one row per physical obs row",
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

    let (existing_obs_index, existing_var_index) = if let Some(reader) = reader.as_ref() {
        let obs = match patch.obs {
            Some(_) => indexed_column_names(reader.read_obs_predicate_index_bytes()?)?,
            None => Vec::new(),
        };
        let var = match patch.var {
            Some(_) => indexed_column_names(reader.read_var_predicate_index_bytes()?)?,
            None => Vec::new(),
        };
        (obs, var)
    } else {
        (Vec::new(), Vec::new())
    };

    // A replaced axis always invalidates its old index section (its ranges
    // describe values that are gone — see `should_drop_old_entry`), so keeping
    // the file's pushdown means *rebuilding* over the same columns, not
    // preserving bytes. An explicit `index_*` request still wins outright: it is
    // the caller naming what they want indexed, and narrowing it is allowed.
    //
    // "Explicit" is decided PER AXIS. `user_wants_index` is a whole-patch
    // question — true if *any* index knob is set — and using it here would let
    // `index_var=[…]` silently switch the obs axis off carry-forward and onto
    // auto-detect, which under the new "omit index_* to carry" contract is a
    // footgun rather than a quirk. `index_preset` and `index_auto_threshold`
    // genuinely span both axes, so they still count for both.
    let cross_axis = patch.index.index_preset.is_some() || patch.index.index_auto_threshold > 0;
    let explicit_obs = cross_axis || !patch.index.index_obs.is_empty();
    let explicit_var = cross_axis || !patch.index.index_var.is_empty();
    let carry_obs = !explicit_obs && !existing_obs_index.is_empty();
    let carry_var = !explicit_var && !existing_var_index.is_empty();
    let rebuild_obs_index = patch.obs.is_some() && (explicit_obs || carry_obs);
    let rebuild_var_index = patch.var.is_some() && (explicit_var || carry_var);

    // Two passes, because this op has two modes on the same axes. `carried`
    // enters the file's existing indexed columns as *preset*, never *forced*
    // (see `ObsVarIndexPass::carried` for why that distinction is load-bearing);
    // `requested` is the ordinary user request. Both resolve before the writer
    // is adopted, so an unknown `--index-preset` fails with nothing written.
    let carried_pass = ObsVarIndexPass::carried(&existing_obs_index, &existing_var_index);
    let requested_pass = ObsVarIndexPass::resolve(&patch.index)?;

    if user_wants_index(&patch.index) && (rebuild_obs_index || rebuild_var_index) {
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

    if let Some(uns) = &uns_to_write {
        writer.write_uns(uns)?;
    }
    if let Some(var) = &patch.var {
        writer.write_var(var)?;
    }

    let mut per_shard_obs_stats: Option<Vec<Vec<ColumnStat>>> = None;
    let mut index_result = ConversionPredicateIndexResult::default();
    // Why the per-shard column stats could not be re-derived, for the warning
    // below. The three arms are genuinely different remedies, and telling a user
    // to "pass --index-obs" when they just did — and when doing it again cannot
    // help — sends them round in a circle.
    let mut no_stats_reason = NoStatsReason::NoRebuildRequested;

    if let Some(obs) = &obs_to_write {
        // Written as handed in (or as scattered onto the physical axis, see
        // `resolve_obs_row_space`): a dictionary (categorical) column is sliced
        // per shard and lands as a dictionary carrying its field metadata, so
        // `read_obs()` gives the caller's categoricals back. (This used to run
        // `unify_dict_columns`, which cast every dictionary column to plain
        // strings; `append` / `merge` still do, after a concat.)
        let obs_pass = if carry_obs {
            &carried_pass
        } else {
            &requested_pass
        };
        let mut builder = if rebuild_obs_index {
            Some(obs_pass.obs_builder(obs.schema())?)
        } else {
            None
        };

        let n = obs.num_rows();
        let mut obs_shard_ranges: Vec<(u64, u64)> = Vec::new();
        let mut cursor = 0usize;
        let mut idx = 0u32;
        // The obs chunking below is by `shard_target_rows`; the index is
        // finished over the CSR shard ranges (see the `use_csr` choice after
        // the loop), and after a `compact` those two partitions diverge. Split
        // each chunk on the ranges the index will actually be keyed to, or the
        // numeric accumulator hands every CSR shard a chunk-wide `[min, max]`
        // and Level-1 pruning stops excluding shards that cannot match.
        let index_ranges: Vec<(u64, u64)> = if !csr_ranges.is_empty()
            && csr_ranges.iter().map(|(s, e)| e - s).sum::<u64>() == prep.old_n_obs
        {
            csr_ranges.clone()
        } else {
            Vec::new()
        };
        // A 0-row obs cannot be sharded: the loop would emit no shard and the
        // file would lose its obs section (`read_obs` → `SectionNotFound`).
        // Same rule as `scx_format_io::write_obs_section` — an empty batch is
        // written as the single legacy section. The index builder (if any)
        // finishes over zero rows and writes nothing.
        if n == 0 {
            writer.write_obs(obs)?;
        }
        while cursor < n {
            let take = std::cmp::min(shard_target_rows, n - cursor);
            let chunk = obs.slice(cursor, take);
            let row_start = cursor as u64;
            if let Some(b) = builder.as_mut() {
                b.push_shard_split(&chunk, row_start, &index_ranges)
                    .map_err(OpsError::Engine)?;
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
            //
            // `index_ranges` above encodes exactly that choice — empty means
            // "fall back" — and the pushes were split on it, so the two must
            // stay derived from the one predicate rather than recomputed.
            let ranges: &[(u64, u64)] = if index_ranges.is_empty() {
                &obs_shard_ranges
            } else {
                &index_ranges
            };
            // `ranges` is `index_ranges` (== `csr_ranges`) whenever that list
            // covers the obs axis, so `Deferred` derives over exactly the
            // CSR-shard space. In the fallback case it is the obs-shard space,
            // where Level-1 stats cannot be derived at all — hence `Skip`.
            let sink = if index_ranges.is_empty() {
                StatsSink::Skip
            } else {
                StatsSink::Deferred(&mut per_shard_obs_stats)
            };
            let wrote =
                obs_pass.finish_obs(builder, ranges, &mut writer, &mut index_result, sink)?;
            if !wrote {
                no_stats_reason = NoStatsReason::NoColumnWasIndexable;
            } else if index_ranges.is_empty() {
                // An index WAS written, over the obs-shard ranges. Another
                // in-place rebuild will land here again — the file's shape
                // is the problem, not the request.
                no_stats_reason = NoStatsReason::CsrRangesUnderCoverObsAxis;
            }
        }
    }

    if rebuild_var_index {
        let var = patch
            .var
            .as_ref()
            .expect("rebuild_var_index implies patch.var is Some");
        let var_pass = if carry_var {
            &carried_pass
        } else {
            &requested_pass
        };
        var_pass.write_var(var, prep.target_n_vars, &mut writer, &mut index_result)?;
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

    // --- What survived the replacement -------------------------------------
    // Computed from what the builder actually indexed rather than from what was
    // requested, so it stays true on every route into this function — including
    // the `index_preset` / `index_auto_threshold` routes, where the columns that
    // end up indexed are not a list the caller wrote down anywhere.
    let not_carried = |existing: &[String], indexed: &[String]| -> Vec<String> {
        existing
            .iter()
            .filter(|c| !indexed.contains(c))
            .cloned()
            .collect()
    };
    let summary = ModifyMetadataSummary {
        obs_columns_not_carried: not_carried(
            &existing_obs_index,
            &index_result.obs_indexed_columns,
        ),
        var_columns_not_carried: not_carried(
            &existing_var_index,
            &index_result.var_indexed_columns,
        ),
        obs_carried_forward: carry_obs && !index_result.obs_indexed_columns.is_empty(),
        var_carried_forward: carry_var && !index_result.var_indexed_columns.is_empty(),
        obs_predicate_index_dropped: !existing_obs_index.is_empty()
            && index_result.obs_indexed_columns.is_empty(),
        var_predicate_index_dropped: !existing_var_index.is_empty()
            && index_result.var_indexed_columns.is_empty(),
        index: PredicateIndexBuildSummary {
            result: Some(index_result),
            multimodal_skip: None,
        },
    };

    // --- Append provenance (manual, mirrors finalize_append) ---------------
    let params = build_params_json(patch, &summary);
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
    // Per-shard `column_stats` describe obs values, and `scx_engine`'s Level-1
    // pushdown prunes straight from them — for the numeric `MinMax` arm without
    // consulting the predicate index at all. Dropping the stale `ObsPredicateIndex`
    // (see `should_drop_old_entry`) is therefore only half the invariant: leave the
    // stats behind and every shard is excluded on a predicate the *new* values
    // satisfy, silently returning a short row set.
    //
    // `assign_csr_shard_column_stats` rewrites every modality-0 CSR entry, so it
    // is already total and needs no pre-clear. It is reached only when an index
    // was built (requested OR carried forward) AND produced index bytes AND the
    // CSR ranges cover [0, n_obs); the `None` arm is what closes the other three
    // routes to a replaced obs. Carrying the index forward is therefore also
    // what keeps Level-1 pruning alive across an ordinary obs edit — before it,
    // every `modify_metadata(obs=…)` landed in the `None` arm.
    match per_shard_obs_stats {
        Some(per_shard) => scx_format_io::assign_csr_shard_column_stats(&mut entries, per_shard)?,
        None if patch.obs.is_some() => {
            let cleared = scx_format_io::clear_all_csr_shard_column_stats(&mut entries);
            if cleared > 0 {
                log::warn!(
                    "modify_metadata dropped stale per-shard column statistics on {cleared} \
                     CSR shards of {}: {}. Level-1 shard pruning is off until they are \
                     re-derived — {}",
                    path.display(),
                    no_stats_reason.what_happened(),
                    no_stats_reason.remedy()
                );
            }
        }
        // A var-only or uns-only patch leaves every obs value intact, so the
        // stats stay true and the file keeps its pushdown.
        None => {}
    }

    // `write_obsm` does not set the header flag, and `commit_in_place`
    // deliberately writes the header verbatim without `sync_from_catalog`.
    // Without this, a first-ever in-place obsm silently disappears on the next
    // `compact` or `subset`, both of which gate obsm copying on
    // `header.has_obsm()` — and `build_csc` drops it the same way. The two
    // external-import ops stamp it by hand for the same reason; the durable
    // alternative is to have `commit_in_place` call `sync_from_catalog`, which
    // would subsume all three sites but changes header derivation for `append`
    // and `delete` too.
    //
    // `varm` needs no counterpart: it has no header flag and `compact` copies it
    // unconditionally.
    if patch.obsm.as_ref().is_some_and(|v| !v.is_empty()) {
        prep.header.set_obsm();
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
    Ok(summary)
}

/// Convenience wrapper: replace the whole `uns` block.
///
/// A uns-only patch touches no axis, so there is no index to carry and nothing
/// in the summary to report — hence the `()`. For a shallow merge see
/// [`update_uns`].
pub fn set_uns(path: &Path, uns: &Value) -> Result<()> {
    modify_metadata(
        path,
        &MetadataPatch {
            uns: Some(uns.clone()),
            ..Default::default()
        },
    )?;
    Ok(())
}

/// Shallow-merge `patch`'s top-level keys into the file's `uns`, in place.
///
/// A patch key replaces a same-named existing key wholesale (nested objects
/// are not deep-merged); every other key survives verbatim; `null` sets null
/// rather than deleting. A file with no `uns` section gets `patch` as-is. The
/// existing `uns` must be a JSON object — anything else is refused before any
/// write (replace it with [`set_uns`]). Same cost and guarantees as
/// [`set_uns`]: O(uns bytes), the matrix is untouched, one atomic commit,
/// undone by one `rollback`.
pub fn update_uns(path: &Path, patch: &Value) -> Result<()> {
    if !patch.is_object() {
        return Err(OpsError::InvalidInput(format!(
            "update_uns: the patch must be a JSON object whose top-level keys are merged; \
             got {}. To replace uns wholesale, use set_uns.",
            json_kind(patch)
        )));
    }
    modify_metadata(
        path,
        &MetadataPatch {
            uns: Some(patch.clone()),
            uns_merge: true,
            ..Default::default()
        },
    )?;
    Ok(())
}

/// JSON value kind for error messages (`null` / `bool` / `number` / …).
fn json_kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a bool",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
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
    // obs/var changed → drop their metadata sections AND any predicate index.
    // The old section's shard ranges describe values that are gone, so it can
    // never be kept verbatim; carrying the index forward means writing a fresh
    // one over the same columns, which the caller above has already done by the
    // time this runs.
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

/// Provenance params recording which fields changed + what happened to the
/// predicate indexes.
///
/// `obs_predicate_index_dropped` deliberately reuses the field name
/// `attach_external_obs` already stamps, so one `scx info` grep answers the
/// same question on either op.
fn build_params_json(patch: &MetadataPatch, summary: &ModifyMetadataSummary) -> Value {
    let (obs_cols, var_cols) = match summary.index.result.as_ref() {
        Some(r) => (
            r.obs_indexed_columns.as_slice(),
            r.var_indexed_columns.as_slice(),
        ),
        None => (&[] as &[String], &[] as &[String]),
    };
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
    if patch.uns_merge {
        params["uns_merge"] = true.into();
    }
    if patch.obs.is_some() {
        params["obs_predicate_index_dropped"] = summary.obs_predicate_index_dropped.into();
    }
    if patch.var.is_some() {
        params["var_predicate_index_dropped"] = summary.var_predicate_index_dropped.into();
    }
    if !obs_cols.is_empty()
        || !var_cols.is_empty()
        || !summary.obs_columns_not_carried.is_empty()
        || !summary.var_columns_not_carried.is_empty()
    {
        params["predicate_index"] = serde_json::json!({
            "obs_columns": obs_cols,
            "var_columns": var_cols,
            "obs_carried_forward": summary.obs_carried_forward,
            "var_carried_forward": summary.var_carried_forward,
            "obs_columns_not_carried": summary.obs_columns_not_carried,
            "var_columns_not_carried": summary.var_columns_not_carried,
        });
    }
    params
}

/// Accept an obs replacement in either row space and return it physical-length.
///
/// `n_obs_physical` is the header count. A frame of exactly that many rows is
/// returned as is (deleted rows in place). A frame of exactly the **live** count
/// — what `pyscx` `read_obs()` returns since 0.17 — is scattered onto the
/// physical axis through the deletion keep mask: a deleted row is `null` in
/// every column the caller supplied, except the obs index column, which keeps
/// the barcode the file already holds for it (a null key would become `""` in
/// every later keyed join and, twice over, a duplicate). Any other length is a
/// [`OpsError::ShapeMismatch`] naming both counts and the read that produces
/// each. When nothing is deleted the two counts coincide and the frame is
/// physical by construction.
fn resolve_obs_row_space(
    reader: &ScxReader,
    obs: &RecordBatch,
    n_obs_physical: u64,
) -> Result<RecordBatch> {
    let n_rows = obs.num_rows() as u64;
    if n_rows == n_obs_physical {
        return Ok(obs.clone());
    }
    let keep = reader.deletion_keep_mask()?;
    let n_live = keep
        .as_ref()
        .map_or(n_obs_physical, |k| k.iter().filter(|b| **b).count() as u64);
    let keep = match keep {
        Some(keep) if n_rows == n_live => keep,
        _ => {
            let deletions = if n_live == n_obs_physical {
                format!("n_obs={n_obs_physical} (no logical deletions)")
            } else {
                format!(
                    "n_obs={n_live} live rows (read_obs()) and n_obs_physical={n_obs_physical} \
                     (read_obs(logical=False); {} logically deleted)",
                    n_obs_physical - n_live
                )
            };
            return Err(OpsError::ShapeMismatch {
                detail: format!(
                    "modify_metadata: obs has {n_rows} rows but file has {deletions} \
                     (changing cell count is out of scope — use append/subset)"
                ),
            });
        }
    };

    let scattered = scx_format_io::scatter_batch_to_physical(obs, &keep)?;

    // Keep the deleted rows' identity. The index column is whatever the file's
    // pandas envelope (or the literal `__index_level_0__`) names, and it must
    // be a column of the new frame too — the barcode is a string, so a
    // dictionary-encoded index is cast to plain Utf8 first and back to the new
    // frame's type after.
    let existing_schema = reader.read_obs_schema_physical()?;
    let index_col = scx_format_io::resolve_index_columns(&existing_schema)
        .into_iter()
        .find(|c| existing_schema.index_of(c).is_ok() && scattered.schema().index_of(c).is_ok());
    let Some(index_col) = index_col else {
        return Ok(scattered);
    };
    let old_index = reader.read_obs_keys(std::slice::from_ref(&index_col))?;
    if old_index.num_rows() as u64 != n_obs_physical {
        return Err(OpsError::ShapeMismatch {
            detail: format!(
                "modify_metadata: the existing obs index column '{index_col}' has {} rows but the \
                 header declares n_obs={n_obs_physical}",
                old_index.num_rows()
            ),
        });
    }
    let pos = scattered.schema().index_of(&index_col)?;
    let target_type = scattered.column(pos).data_type().clone();
    let old_col = arrow::compute::cast(
        old_index
            .column_by_name(&index_col)
            .expect("projected column present"),
        &target_type,
    )?;
    // Where the mask is live take the caller's value, elsewhere the file's.
    let live_mask = arrow::array::BooleanArray::from(keep.clone());
    let merged = arrow::compute::kernels::zip::zip(&live_mask, scattered.column(pos), &old_col)?;
    let mut columns = scattered.columns().to_vec();
    columns[pos] = merged;
    Ok(RecordBatch::try_new_with_options(
        scattered.schema(),
        columns,
        &arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(keep.len())),
    )?)
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

    /// The under-covering route rebuilt the index already, so telling the caller
    /// to pass `--index-obs` sends them round a loop that cannot terminate: the
    /// next in-place rebuild hits the same `use_csr == false` branch and skips
    /// the derive again. Only a copy-out rewrite that re-shards the matrix helps.
    ///
    /// This is the exact wording bug this test exists to prevent, so it asserts
    /// on the property (does not offer an in-place rebuild; does name the
    /// copy-out ops) rather than on the sentence.
    #[test]
    fn the_under_covering_remedy_does_not_recommend_another_in_place_rebuild() {
        use super::NoStatsReason::*;

        let remedy = CsrRangesUnderCoverObsAxis.remedy();
        assert!(
            remedy.contains("sort") && remedy.contains("compact"),
            "must point at the copy-out rewrite that re-shards the matrix: {remedy}"
        );
        assert!(
            !remedy.starts_with("pass --index-obs"),
            "an in-place rebuild is what just failed here: {remedy}"
        );

        // The other two ARE fixed in place, and must keep saying so.
        assert!(NoRebuildRequested.remedy().contains("--index-obs"));
        assert!(NoColumnWasIndexable.remedy().contains("--index-obs"));

        // All three describe different situations — a copy-paste that collapsed
        // two of them would defeat the point.
        let all = [
            NoRebuildRequested,
            CsrRangesUnderCoverObsAxis,
            NoColumnWasIndexable,
        ];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a.what_happened(), b.what_happened());
                assert_ne!(a.remedy(), b.remedy());
            }
        }
    }
}
