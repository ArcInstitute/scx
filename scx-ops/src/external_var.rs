//! Attach externally-computed **per-gene annotations** to an existing SCX file,
//! in place.
//!
//! The var-axis sibling of [`crate::external_obs::attach_external_obs`], on the
//! same seam: the join is by key string and never by row position, the write is
//! in place through `prepare_in_place` / `commit_in_place`, and one
//! `scx rollback` undoes it. The motivating cases are gene-label normalisation
//! (a normalised symbol per gene, computed against a reference release) and
//! ATAC peak annotation — but nothing here knows about either, exactly as the
//! obs op knows nothing about doublets.
//!
//! # Why a separate op rather than `modify_metadata(var=)`
//!
//! `modify_metadata` replaces the whole var table, so a caller who wants to add
//! one column has to read var, do the join in pandas, and hand back every
//! column — which means being trusted with all of them, and doing the join
//! itself. This op writes named columns, joins them by key, reports what
//! matched, and previews with `dry_run`.
//!
//! # What is *not* mirrored from the obs op, and why
//!
//! * **The live/physical row-space dispatch.** Deletion vectors are an obs-axis
//!   concept; var has no logical/physical split, so `AxisJoinKey::Positional`
//!   on var means exactly one thing — `n_vars` rows, row `i` annotates gene `i`
//!   — and any other length is refused rather than reinterpreted.
//! * **`varm` payloads.** `varm` has no header flag (`has_obsm` / `set_obsm` is
//!   the only pair), so a first-ever in-place `varm` has a lifecycle question to
//!   answer before it can be written safely.
//! * **A streaming key read.** There is no `read_var_keys` /
//!   `read_var_shard_projected`, and at 10⁴–10⁵ genes there is nothing to save:
//!   [`ScxReader::read_var`] assembles the table (running the shard-cover walk
//!   while it does) and the streaming happens on the **write**, to preserve the
//!   input's shard boundaries.
//!
//! # Layout
//!
//! Whatever layout var arrived in, it leaves in: a sharded var is rewritten one
//! shard at a time with the same boundaries, a single `VarMetadata` section
//! stays a single section. This deliberately differs from the obs op, which
//! upgrades a legacy single section to shards — nothing in the ingest tree
//! creates var shards, and `merge` / `compact` / `modify_metadata` all collapse
//! var back to one section, so an attach is the wrong place to change a file's
//! layout. [`AttachVarSummary::var_streamed`] reports which path ran.
//!
//! # Predicate indexes
//!
//! A pure column *add* leaves the var predicate index valid, so it is carried
//! verbatim. An **overwrite** of an indexed column would leave it describing
//! values that are gone — and unlike obs, where a sharded index cannot be
//! rebuilt mid-append, var's index is one batch-mode build over `[(0, n_vars)]`
//! and the new table is already in memory. So it is **rebuilt**, matching
//! `modify_metadata`, and [`AttachVarSummary::var_columns_not_carried`] names
//! any column the rebuild could not cover.
//!
//! There is no var equivalent of the per-shard `ColumnStat`s that obs-axis
//! Level-1 pushdown reads, so this op clears none — clearing them would
//! silently disable pruning on obs.

use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use serde_json::Value;

use scx_format_io::catalog::{FullCatalog, FullCatalogEntry};
use scx_format_io::checksum::blake3_hash;
use scx_format_io::provenance::{Provenance, ProvenanceEntry};
use scx_format_io::reader::ScxReader;
use scx_format_io::section::{write_alignment_padding, SectionType};
use scx_format_io::writer::ScxWriter;

use crate::error::{OpsError, Result};
use crate::external_layer::{ExtraRowPolicy, MissingRowPolicy};
use crate::external_obs::{
    build_axis_row_join, build_new_axis_batch, check_uns_collisions, diagnose_axis_key,
    index_would_go_stale, indexed_column_names, materialize_target_keys, merge_uns_entries,
    planned_columns, read_uns_for_merge, resolve_target_key_spec, AxisJoinKey, AxisRowJoin,
    KeyDiagnosis,
};
use crate::in_place::{commit_in_place, prepare_in_place, read_provenance_ops};
use crate::predicate_index::ObsVarIndexPass;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Externally-computed per-gene annotations, keyed by string.
#[derive(Debug, Clone)]
pub struct ExternalVarData {
    /// One key per source row. `len == row_annotations.num_rows()`.
    ///
    /// For a composite key these are the fused values — build them with
    /// [`crate::build_composite_key_for`] so both sides of the join are
    /// constructed by the same code and the separator stays an implementation
    /// detail. Under [`AxisJoinKey::Positional`] they may be empty, or one per
    /// row as an order check.
    pub row_keys: Vec<String>,
    /// The columns to append to var. Fields are forced nullable on write:
    /// unmatched genes become `null`, never a fabricated zero.
    pub row_annotations: RecordBatch,
    /// Entries merged into the target's `uns` at top level, in the same commit
    /// as the columns. Empty leaves the existing `uns` section byte-identical.
    pub uns: serde_json::Map<String, Value>,
    /// BLAKE3 of the source file, recorded in the provenance entry.
    pub source_checksum: Option<[u8; 32]>,
    /// Display name of the source (usually a path basename).
    pub source_name: Option<String>,
}

impl ExternalVarData {
    /// Nest every uns entry under one key: `{a, b}` becomes `{key: {a, b}}`.
    /// A no-op on an empty payload, so a key with nothing to put under it never
    /// creates an empty record.
    pub fn nest_uns_under(&mut self, key: &str) {
        if self.uns.is_empty() {
            return;
        }
        let inner = std::mem::take(&mut self.uns);
        self.uns.insert(key.to_string(), Value::Object(inner));
    }
}

pub struct AttachVarOptions {
    pub join_key: AxisJoinKey,
    pub missing_row_policy: MissingRowPolicy,
    pub extra_row_policy: ExtraRowPolicy,
    /// Var column recording `"present"` / `"absent"` per gene. `None` omits it.
    pub status_column: Option<String>,
    /// When false (default), any pre-existing var column / uns key this op
    /// would replace is an error.
    ///
    /// **Replace, never merge.** With `overwrite = true` a colliding column is
    /// dropped and rebuilt from this source alone; values the previous import
    /// wrote are gone, so landing N partial annotation tables means
    /// concatenating them first and importing once.
    pub overwrite: bool,
    pub provenance_action: String,
    /// Merged into the provenance `params_json` object.
    pub provenance_params: Value,
    /// Run every validation and resolve the join, then return the summary
    /// **without writing anything**.
    pub dry_run: bool,
}

impl Default for AttachVarOptions {
    fn default() -> Self {
        Self {
            join_key: AxisJoinKey::Auto,
            missing_row_policy: MissingRowPolicy::default(),
            extra_row_policy: ExtraRowPolicy::default(),
            status_column: None,
            overwrite: false,
            provenance_action: "attach_external_var".to_string(),
            provenance_params: Value::Null,
            dry_run: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AttachVarSummary {
    pub n_vars: u64,
    pub n_matched: u64,
    pub n_target_rows_absent: u64,
    pub n_source_rows_absent: u64,
    /// The resolved key, or the comma-joined composite spec. The **physical**
    /// column name; consumers that show it to a user pass it through
    /// [`crate::display_key_name`] first.
    pub var_key_column: String,
    pub var_columns_added: Vec<String>,
    /// Whether a **replacement** `VarPredicateIndex` section was written
    /// because this import overwrote a column the old one covered. `false` on
    /// a pure add (the section is carried verbatim), on a file with no index,
    /// and when the rebuild could cover nothing — see
    /// [`Self::var_index_dropped`].
    pub var_index_rebuilt: bool,
    /// Whether the stale index section was removed with **no** replacement,
    /// because every column it covered became unindexable.
    ///
    /// Reported separately from [`Self::var_index_rebuilt`] because "rebuilt"
    /// and "gone" are different outcomes for a caller, and the planning
    /// boolean alone cannot tell them apart: whether a replacement can be
    /// built is only known once the new values are in hand.
    pub var_index_dropped: bool,
    /// Columns the old index covered that the rebuild could not — reported
    /// rather than silently dropped. Honest on a `dry_run` too.
    pub var_columns_not_carried: Vec<String>,
    /// `true` when var was rewritten shard by shard, preserving the input's
    /// boundaries; `false` when it was a single `VarMetadata` section, which
    /// stays a single section.
    pub var_streamed: bool,
}

// ---------------------------------------------------------------------------
// Key helpers, for the source-side readers
// ---------------------------------------------------------------------------

/// Resolve a var-style join-key column with the same preference order the
/// attach op uses: the pandas index column, then the gene-id fallbacks.
///
/// Exported for readers that build the **source** side of a join. Both sides
/// must resolve keys identically — if a reader picked `gene_id` while the op
/// picked `_index`, a file that should have joined would fail with zero overlap
/// and no obvious cause.
pub fn resolve_var_key_column(batch: &RecordBatch, requested: Option<&str>) -> Result<String> {
    crate::external_layer::resolve_key_column("var", &batch.schema(), requested)
}

/// Report which var columns could serve as a join key on `path`. Read-only.
pub fn diagnose_var_key(path: &Path, requested: Option<&AxisJoinKey>) -> Result<KeyDiagnosis> {
    let reader = ScxReader::open(path)?;
    let var = reader.read_var()?;
    Ok(diagnose_axis_key("var", &var, requested))
}

// ---------------------------------------------------------------------------
// The var write path
// ---------------------------------------------------------------------------

/// Rewrite var shard by shard, letting `build` append the import's columns to
/// each one, preserving the input's shard boundaries exactly.
///
/// The var-axis mirror of `write_obs_shards_appending`: one output shard per
/// input shard rather than a fresh split by `shard_target_rows`, so the output's
/// var tiles `[0, n_vars)` the same way the input did.
pub(crate) fn write_var_shards_appending<F>(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    n_vars: u64,
    mut build: F,
) -> Result<()>
where
    F: FnMut(&RecordBatch, usize) -> Result<RecordBatch>,
{
    let mut cover = crate::compact::ShardCoverCheck::default();
    let mut row_start = 0usize;
    for (out_idx, res) in reader.var_shards().enumerate() {
        let out_idx = out_idx as u32;
        let shard = res?;
        cover.visit("var_metadata", &shard)?;
        let n = shard.num_rows();
        // Written as built: a dictionary (categorical) column goes back to disk
        // as a dictionary, with its field metadata.
        let built = build(&shard, row_start)?;
        writer.write_var_shard(out_idx, row_start as u64, n as u64, n_vars, &built)?;
        row_start += n;
    }
    cover.finish("var_metadata", n_vars)?;
    Ok(())
}

/// Whether an old catalog entry is superseded and must be dropped.
///
/// Deliberately narrow. `obs`, CSR shards, layers, the CSC sidecar, `.raw`,
/// deletion vectors, detection bitmaps, `varm` / `varp` and the obs predicate
/// index are all untouched by a var-only attach and must survive verbatim.
fn should_drop_old_entry(e: &FullCatalogEntry, rewrote_uns: bool, retire_var_index: bool) -> bool {
    use SectionType::*;
    if e.section_type == Provenance {
        return true;
    }
    if rewrote_uns && e.section_type == UnsBlob && e.modality_id == 0 {
        return true;
    }
    if matches!(e.section_type, VarMetadata | VarMetadataShard) {
        return true;
    }
    // `retire_var_index` covers both outcomes: the stale section goes whether a
    // replacement was built from the new values or nothing could be.
    if retire_var_index && e.section_type == VarPredicateIndex {
        return true;
    }
    false
}

fn build_params_json(
    data: &ExternalVarData,
    opts: &AttachVarOptions,
    s: &AttachVarSummary,
) -> Value {
    let mut v = serde_json::json!({
        "source_file": data.source_name,
        "var_key_column": s.var_key_column,
        "n_vars": s.n_vars,
        "n_matched": s.n_matched,
        "n_target_rows_absent": s.n_target_rows_absent,
        "n_source_rows_absent": s.n_source_rows_absent,
        "var_columns": s.var_columns_added,
        "var_predicate_index_rebuilt": s.var_index_rebuilt,
        "var_predicate_index_dropped": s.var_index_dropped,
        "var_columns_not_carried": s.var_columns_not_carried,
        // Which layout the file had, and therefore kept. Not recoverable from
        // the output on its own: a single-section var and a one-shard sharded
        // var both hold every gene.
        "var_streamed": s.var_streamed,
        "uns_keys_merged": data.uns.keys().collect::<Vec<_>>(),
        "overwrite": opts.overwrite,
    });
    if let (Some(map), Some(extra)) = (v.as_object_mut(), opts.provenance_params.as_object()) {
        for (k, val) in extra {
            map.insert(k.clone(), val.clone());
        }
    }
    v
}

fn check_collisions(
    var_schema: &Schema,
    uns: &Value,
    data: &ExternalVarData,
    var_new: &[String],
) -> Result<()> {
    for name in var_new {
        if var_schema.field_with_name(name).is_ok() {
            return Err(OpsError::InvalidInput(format!(
                "var column '{name}' already exists; pass overwrite=true to replace it. \
                 Note that overwrite REPLACES rather than merges — importing several \
                 partial tables one after another would keep only the last."
            )));
        }
    }
    check_uns_collisions(uns, &data.uns)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Attach `data` to the SCX file at `path` as var columns plus optional `uns`,
/// joined to the target's own var axis by key.
///
/// `X`, layers, the CSC sidecar, `.raw`, deletion vectors, detection bitmaps,
/// `varm` / `varp` and `obs` are never read or rewritten; the whole import is
/// undoable with `scx rollback`.
///
/// A **multimodal** target is refused: each modality owns its own `var/{name}`
/// section, so "the var axis" is ambiguous there. Extract a modality with
/// `scx subset --modality`, attach, and `scx merge` back. This matches
/// `attach_external_layer`, the other op that writes var.
pub fn attach_external_var(
    path: &Path,
    data: &ExternalVarData,
    opts: &AttachVarOptions,
) -> Result<AttachVarSummary> {
    if data.row_keys.len() != data.row_annotations.num_rows()
        && !(opts.join_key == AxisJoinKey::Positional && data.row_keys.is_empty())
    {
        return Err(OpsError::ShapeMismatch {
            detail: format!(
                "row_annotations has {} rows but there are {} row keys",
                data.row_annotations.num_rows(),
                data.row_keys.len()
            ),
        });
    }
    let positional = opts.join_key == AxisJoinKey::Positional;
    if positional && opts.status_column.is_some() {
        return Err(OpsError::InvalidInput(
            "status_column is meaningless under a positional attach — every gene \
             matches by construction, so the marker would be a constant; drop \
             status_column or use a key join"
                .into(),
        ));
    }
    // A status column sharing a name with an annotation would produce TWO var
    // columns with the same name: `check_collisions` compares planned names
    // against the OLD schema only, and `build_new_axis_batch` pushes the status
    // field and the annotation field independently.
    if let Some(status) = &opts.status_column {
        if data
            .row_annotations
            .schema()
            .field_with_name(status)
            .is_ok()
        {
            return Err(OpsError::InvalidInput(format!(
                "status_column '{status}' is also an annotation column in the \
                 source; writing both would leave two var columns with the same \
                 name. Rename one of them."
            )));
        }
    }

    let planned_var = planned_columns(&data.row_annotations, opts.status_column.as_deref());
    if planned_var.is_empty() && data.uns.is_empty() {
        return Err(OpsError::InvalidInput(
            "nothing to attach: row_annotations has no columns, and no status \
             column or uns payload was supplied"
                .into(),
        ));
    }

    let (mut lock, mut prep) = prepare_in_place(path, 0)?;

    // Each modality owns a `var/{name}` section and the global `var` a
    // multimodal file carries is not any modality's gene axis, so there is no
    // single var table to attach to. Refused before any write.
    if prep.header.n_modalities > 0 {
        return Err(OpsError::MultimodalUnsupported {
            op: "attach_external_var",
        });
    }

    // The reader takes no lock of its own; we only ever append, never truncate,
    // so holding it across the transaction is safe.
    let reader = ScxReader::open(path)?;
    let uns = read_uns_for_merge(&reader, &data.uns)?;

    let n_vars = prep.target_n_vars;
    // Whole, not projected: there is no per-shard var key reader, and at
    // 10⁴–10⁵ genes there would be nothing to save if there were. `read_var`
    // runs the shard-cover walk while assembling, so a var whose shards do not
    // tile `[0, n_vars)` fails here rather than in the write loop.
    let var = reader.read_var()?;
    if var.num_rows() as u64 != n_vars {
        return Err(OpsError::ShapeMismatch {
            detail: format!(
                "var has {} rows but the header declares n_vars = {n_vars}",
                var.num_rows()
            ),
        });
    }
    let var_streamed = reader.var_metadata_shard_count() > 0;

    // --- Resolve the join --------------------------------------------------
    let (key_spec, row_join) = if positional {
        if data.row_annotations.num_rows() as u64 != n_vars {
            return Err(OpsError::ShapeMismatch {
                detail: format!(
                    "positional attach needs one row per gene: the frame has {} rows \
                     but the file has n_vars = {n_vars}",
                    data.row_annotations.num_rows()
                ),
            });
        }
        let join = AxisRowJoin {
            source_of_target: (0..n_vars as usize).map(|i| Some(i as u32)).collect(),
            n_matched: n_vars,
            n_target_absent: 0,
            n_source_absent: 0,
        };
        // Supplied keys are never joined on here — they are an *order check*.
        // A frame sorted or reindexed after `read_var()` has the right length
        // and every value on the wrong gene, which is the one way a positional
        // attach silently corrupts a file. Labels that are not this file's gene
        // names at all are ignored, as positional always ignored the index.
        if !data.row_keys.is_empty() {
            let index_cols = scx_format_io::resolve_index_columns(&var.schema());
            if !index_cols.is_empty() {
                let target = materialize_target_keys("var", &var, &index_cols)?;
                let mut a: Vec<&str> = target.iter().map(String::as_str).collect();
                let mut b: Vec<&str> = data.row_keys.iter().map(String::as_str).collect();
                a.sort_unstable();
                b.sort_unstable();
                if a == b && target != data.row_keys {
                    let i = (0..target.len())
                        .find(|i| target[*i] != data.row_keys[*i])
                        .expect("a != order ⇒ a mismatch exists");
                    return Err(OpsError::InvalidInput(format!(
                        "positional attach: the frame holds the file's own gene names in a \
                         different order — frame row {i} is '{}' but var row {i} is '{}'. A \
                         frame sorted or reindexed after read_var() cannot be landed \
                         positionally; restore the original order (or join by key instead)",
                        data.row_keys[i], target[i]
                    )));
                }
            }
        }
        (opts.join_key.describe(), join)
    } else {
        let (key_spec, key_columns) =
            resolve_target_key_spec("var", &var.schema(), &opts.join_key)?;
        let target_keys = materialize_target_keys("var", &var, &key_columns)?;
        let row_join = build_axis_row_join(
            "var",
            &reader,
            &key_spec,
            &target_keys,
            &data.row_keys,
            opts.missing_row_policy,
            opts.extra_row_policy,
        )?;
        (key_spec, row_join)
    };

    // --- Collision checks ---------------------------------------------------
    if !opts.overwrite {
        check_collisions(&var.schema(), &uns, data, &planned_var)?;
    }

    // --- Would this stale the var predicate index? --------------------------
    // A pure add keeps it verbatim; an overwrite of a column it covers is
    // rebuilt from the new table rather than dropped, because var's index is
    // one batch-mode build over `[(0, n_vars)]` and the table is in memory.
    let index_bytes = reader.read_var_predicate_index_bytes()?;
    let existing_var_index = indexed_column_names(index_bytes)?;
    let retire_var_index = index_would_go_stale(index_bytes, &planned_var)?;

    let rewrote_uns = !data.uns.is_empty();
    let new_uns = merge_uns_entries(uns, &data.uns)?;

    let old_catalog_offset = prep.old_catalog_offset;
    let data_gen = prep.old_catalog.data_generation;
    let csc_gen = prep.old_catalog.csc_build_generation;
    let manifest = prep.header.manifest_sequence;
    let mt_off = prep.header.modality_table_offset;
    let mt_len = prep.header.modality_table_length;

    let build = |batch: &RecordBatch, row_start: usize| {
        build_new_axis_batch(
            "var",
            batch,
            &data.row_annotations,
            &planned_var,
            opts.status_column.as_deref(),
            &row_join,
            row_start,
        )
    };

    // The whole new table, built before the writer exists so a failure still
    // leaves the file untouched.
    //
    // Needed on two paths and skipped otherwise: the single-section write
    // emits it directly, and an index rebuild indexes it. A *streamed* write
    // does **not** use it — see the write below for why it rebuilds per shard
    // instead of slicing this — so on a sharded var with no index to rebuild
    // it is never built.
    let need_whole = !var_streamed || retire_var_index;
    let new_var = if need_whole {
        Some(build(&var, 0)?)
    } else {
        None
    };

    // The replacement index, built here rather than inside the write, so a
    // `dry_run` reports what the real import would do. Which covered columns
    // survive is only knowable from the new *values* — a covered string column
    // overwritten by a float cannot be indexed — so the planning boolean alone
    // cannot fill `var_columns_not_carried`, and a preview that guessed `[]`
    // would disagree with the write it is previewing. The bytes are reused
    // below rather than rebuilt.
    let (new_index_bytes, var_columns_not_carried) = if retire_var_index {
        let whole = new_var
            .as_ref()
            .expect("retire_var_index implies need_whole");
        let pass = ObsVarIndexPass::carried(&[], &existing_var_index);
        let (bytes, covered) = pass.build_var_bytes(whole, n_vars)?;
        let missing: Vec<String> = existing_var_index
            .iter()
            .filter(|c| !covered.contains(c))
            .cloned()
            .collect();
        (bytes, missing)
    } else {
        (None, Vec::new())
    };

    let summary = AttachVarSummary {
        n_vars,
        n_matched: row_join.n_matched,
        n_target_rows_absent: row_join.n_target_absent,
        n_source_rows_absent: row_join.n_source_absent,
        var_key_column: key_spec,
        var_columns_added: planned_var.clone(),
        // "Rebuilt" means a replacement section exists; when the rebuild could
        // cover nothing, the stale section is retired and that is a *drop*.
        var_index_rebuilt: retire_var_index && new_index_bytes.is_some(),
        var_index_dropped: retire_var_index && new_index_bytes.is_none(),
        var_columns_not_carried,
        var_streamed,
    };

    if opts.dry_run {
        // Everything above is validation, the join, and the index decision, so
        // the caller gets a real `n_matched` and a real index outcome while the
        // file stays untouched.
        return Ok(summary);
    }

    let mut prov_ops = read_provenance_ops(&mut lock, &prep.old_catalog)?;

    // --- Emit sections at EOF through an adopted writer --------------------
    let write_offset = lock.seek(SeekFrom::End(0))?;
    let cloned = lock.file().try_clone()?;
    // Seeded with no entries, like the obs op: `into_in_place_parts` hands back
    // whatever the writer holds, so seeding it with the old catalog would
    // duplicate every entry in the new one. The layout is therefore decided
    // here rather than by the writer's own single-vs-sharded guard.
    let mut writer =
        ScxWriter::adopt_in_place(cloned, prep.header.clone(), write_offset, Vec::new())?;

    if rewrote_uns {
        writer.write_uns(&new_uns)?;
    }

    if var_streamed {
        // Rebuilt **from each shard as read**, not sliced out of `new_var`.
        // That is the difference between preserving a shard's own encoding and
        // re-encoding it: `read_var()` reconciles a var whose shards disagree
        // (one dictionary-encoded column, one plain `Utf8` — what an in-place
        // rewrite by an older writer leaves behind), so a slice of the
        // assembled table writes the *unified* encoding to every shard.
        // Measured on a two-shard mixed fixture: slicing turns
        // `[Dictionary(Int32, Utf8), Utf8]` into
        // `[Dictionary(Int8, Utf8), Dictionary(Int8, Utf8)]`, rewriting bytes
        // for a column this op was not asked to touch. Pinned by
        // `a_var_whose_shards_disagree_on_encoding_keeps_each_shards_own`.
        write_var_shards_appending(&reader, &mut writer, n_vars, |shard, row_start| {
            build(shard, row_start)
        })?;
    } else {
        writer.write_var(new_var.as_ref().expect("!var_streamed implies need_whole"))?;
    }

    if let Some(bytes) = &new_index_bytes {
        writer.write_var_predicate_index(bytes)?;
    }

    let (file, new_offset, new_section_entries) = writer.into_in_place_parts()?;
    drop(file);
    lock.seek(SeekFrom::Start(new_offset))?;

    // --- Append provenance --------------------------------------------------
    prov_ops.push(ProvenanceEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        action: opts.provenance_action.clone(),
        tool: concat!("scx-ops ", env!("CARGO_PKG_VERSION")).to_string(),
        params_json: build_params_json(data, opts, &summary).to_string(),
        input_checksums: data.source_checksum.map(|c| vec![c]).unwrap_or_default(),
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
    let prov_checksum = blake3_hash(&prov_bytes);

    // --- Assemble the new catalog ------------------------------------------
    let mut entries: Vec<FullCatalogEntry> = prep
        .old_catalog
        .entries
        .into_iter()
        .filter(|e| !should_drop_old_entry(e, rewrote_uns, retire_var_index))
        .collect();
    entries.extend(new_section_entries);
    entries.push(FullCatalogEntry {
        name: "provenance".to_string(),
        offset: prov_offset,
        length: prov_bytes.len() as u64,
        section_type: SectionType::Provenance,
        checksum: prov_checksum,
        modality_id: 0,
        stats: None,
    });

    // Deliberately NOT `clear_csr_shard_column_stats_for`: those stats are the
    // obs-axis Level-1 pushdown's input, keyed by obs column name, and this op
    // does not touch an obs column. Clearing them — the shape a copy from the
    // obs op would take — would silently disable pruning on obs.

    let new_catalog = FullCatalog {
        catalog_version: scx_format_io::CURRENT_CATALOG_VERSION,
        manifest_sequence: manifest + 1,
        prev_catalog_offset: old_catalog_offset,
        n_obs: prep.old_n_obs,
        entries,
        // X is never read or rewritten, so the CSC sidecar stays fresh. Bumping
        // either of these would silently invalidate it.
        data_generation: data_gen,
        csc_build_generation: csc_gen,
    };

    commit_in_place(&mut lock, &mut prep.header, &new_catalog, mt_off, mt_len)?;

    Ok(summary)
}

#[cfg(test)]
#[path = "external_var_tests.rs"]
mod tests;
