//! Attach externally-computed **per-cell annotations** to an existing SCX file,
//! in place.
//!
//! The obs-only sibling of [`crate::external_layer::attach_external_layer`].
//! The motivating case is doublet-caller interop — run scDblFinder / Scrublet /
//! solo externally, then land its score and call as `obs` columns — but nothing
//! here knows about doublets, exactly as the layer op knows nothing about
//! CellBender.
//!
//! # Why a separate op rather than a degenerate layer
//!
//! Every popular doublet caller emits a score and a call and *no matrix*.
//! [`crate::external_layer::ExternalLayerData`] requires a column axis plus a CSR
//! triple, so the only way to push annotations through it is a zero-nnz matrix —
//! which would leave a useless all-zero `LayerCsrShard` section on every imported
//! file. This op writes obs (and optionally obsm / uns) and nothing else.
//!
//! It is a strict subset of the layer op, and the half it drops is where both of
//! that op's documented writer traps live (the duplicate-`_shard_` re-import bug
//! and `write_layer_csr_shard` bypassing the v4 framing guard).
//!
//! # Why the join is by key and never by position
//!
//! Same reason as the layer op: the external tool's row order is its own
//! business, and a tool run per-library against a merged atlas returns rows in
//! whatever order it pleased. Rows the target has but the source lacks get
//! `null` annotations — never a fabricated `0.0`, which would be a scientific
//! claim the tool never made — and an `"absent"` status marker.
//!
//! # The join key on a real atlas is often not a barcode
//!
//! Measured on a 1M-cell CELLxGENE-derived file: no barcode column at all, a
//! stringified `RangeIndex` that is only 100,000-unique across 1,000,000 rows,
//! and exactly one unique column (`soma_joinid`) that no fallback list would
//! ever guess. Failing loud on a duplicate key is correct but strands the user,
//! so every key failure carries a [`KeyDiagnosis`] naming the columns that *are*
//! unique. See [`diagnose_obs_key`].
//!
//! # Predicate indexes
//!
//! A pure column *add* leaves the obs predicate index valid — it keys on column
//! name, and the indexed columns are untouched. An **overwrite** of an indexed
//! column does not: the index would describe values that no longer exist, and
//! query pushdown would silently return wrong rows. [`obs_index_would_go_stale`]
//! decides precisely, so pushdown survives every import that does not actually
//! invalidate it.
//!
//! # Memory
//!
//! The point of this op is landing a *small* annotation on a *large* file, so
//! the target side must not scale with the target. On a file whose obs is
//! sharded — every atlas — it does not: the schema is read from the Arrow IPC
//! footer, the join reads only the key column(s) through
//! [`ScxReader::read_obs_keys`], and the rewrite runs one obs shard at a time.
//! Peak is one obs shard plus the join arrays (one `Option<u32>` and one key
//! string per target row).
//!
//! Three things are still unbounded, deliberately:
//!
//! * **A legacy single-section `ObsMetadata` target.** One Arrow IPC section is
//!   one batch and there is no per-shard reader, so the whole table is
//!   assembled. The op warns and names `scx optimize` as the fix;
//!   [`AttachObsSummary::obs_streamed`] reports which path ran.
//! * **The key diagnosis on a failed join** ([`KeyDiagnosis`]) reads every obs
//!   column. Only reached when the import is already failing.
//! * **`obsm` embeddings**, which `write_obsm` emits as one `n_obs`-row
//!   section. Not reached unless the caller supplies them — no doublet caller
//!   does, but `cellbender_import(latent_embedding=True)` builds one, so this
//!   is reachable on the layer op.
//!
//! The **source** side is resident in full by construction —
//! [`ExternalObsData`] holds every key and every annotation — which is the
//! bargain the op is built on: the source is the small side.

use std::collections::{HashMap, HashSet};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use arrow::array::{Array, ArrayRef, RecordBatch, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::row::{RowConverter, SortField};
use serde_json::Value;
use std::sync::Arc;

use scx_format_io::catalog::{FullCatalog, FullCatalogEntry};
use scx_format_io::checksum::blake3_hash;
use scx_format_io::provenance::{Provenance, ProvenanceEntry};
use scx_format_io::reader::ScxReader;
use scx_format_io::section::{write_alignment_padding, SectionType};
use scx_format_io::writer::ScxWriter;

use crate::error::{OpsError, Result};
use crate::external_layer::{
    display_key_name, examples, first_duplicates, is_joinable_key_column, is_string_column,
    resolve_key_alias, resolve_key_column, string_column, unmatched_examples, ExtraRowPolicy,
    MissingRowPolicy, OBS_KEY_FALLBACKS,
};
use crate::in_place::{commit_in_place, entry_matches_key, prepare_in_place, read_provenance_ops};

/// Separator used to fuse a multi-column join key into one string.
///
/// ASCII unit separator. Deliberately **not** configurable: this op builds both
/// sides of the join from the same named columns, so the separator never
/// reaches a user and can be a byte no barcode, sample id or accession contains.
/// A configurable `_` — the obvious first choice — would collide constantly with
/// ordinary sample ids like `sample_1`.
pub const COMPOSITE_KEY_SEPARATOR: char = '\u{1f}';

/// How many unique-column candidates to report in a [`KeyDiagnosis`].
const DIAGNOSIS_MAX_COLUMNS: usize = 12;

/// Cap on the 2-column composite search. With `C` obs columns an exhaustive
/// search is `C²/2` full passes — 378 over a 28-column atlas — so restrict it to
/// the highest-cardinality columns, which are the only plausible key components
/// anyway. `capped` on the result says whether anything was skipped.
const DIAGNOSIS_PAIR_SEARCH_TOP_K: usize = 8;

/// How many unique pairs to report before stopping the search.
const DIAGNOSIS_MAX_PAIRS: usize = 3;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Externally-computed per-cell annotations, keyed by string.
///
/// The obs-only sibling of `ExternalLayerData`: no matrix, no column axis.
#[derive(Debug, Clone)]
pub struct ExternalObsData {
    /// One key per source row. `len == row_annotations.num_rows()`.
    ///
    /// For a composite key these are the fused values — build them with
    /// [`build_composite_key`] so both sides of the join are constructed by the
    /// same code and the separator stays an implementation detail.
    pub row_keys: Vec<String>,
    /// The columns to append to obs. Fields are forced nullable on write:
    /// unmatched target rows become `null`, never a fabricated zero.
    pub row_annotations: RecordBatch,
    /// Per-source-row dense matrices written to `obsm`. Usually empty for
    /// doublet callers.
    pub row_embeddings: Vec<(String, RecordBatch)>,
    /// Entries merged into the target's `uns` at top level, in the same commit
    /// as the columns. Empty leaves the existing `uns` section byte-identical
    /// (it is not even rewritten). To land everything under one key, build the
    /// payload and call [`ExternalObsData::nest_uns_under`].
    pub uns: serde_json::Map<String, Value>,
    /// BLAKE3 of the source file, recorded in the provenance entry.
    pub source_checksum: Option<[u8; 32]>,
    /// Display name of the source (usually a path basename).
    pub source_name: Option<String>,
}

impl ExternalObsData {
    /// Nest every uns entry under one key: `{a, b}` becomes `{key: {a, b}}`.
    /// What `obs_import --uns-key K` asks for. A no-op on an empty payload, so
    /// a key with nothing to put under it never creates an empty record.
    pub fn nest_uns_under(&mut self, key: &str) {
        if self.uns.is_empty() {
            return;
        }
        let inner = std::mem::take(&mut self.uns);
        self.uns.insert(key.to_string(), Value::Object(inner));
    }
}

/// How the join key is built from obs columns.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ObsJoinKey {
    /// Resolve automatically: the pandas index column, then `OBS_KEY_FALLBACKS`.
    #[default]
    Auto,
    /// One named obs column.
    Column(String),
    /// Ordered obs columns fused with [`COMPOSITE_KEY_SEPARATOR`]. The source
    /// must carry columns of the same names; both sides are built identically.
    Composite { columns: Vec<String> },
    /// No join at all: source row `i` annotates obs row `i`.
    ///
    /// **For frames computed in-process from this file's own obs axis** — e.g.
    /// columns derived from `read_obs()` output, which is in obs row order.
    /// External tool output must never use this: a tool returns rows in its own
    /// order, which is the whole reason the key join exists (see the module
    /// doc). What makes positional necessary at all is the file the key join
    /// *cannot* serve — a merged atlas whose obs index is duplicated with no
    /// unique column, where an in-process caller still needs to land columns it
    /// just computed.
    ///
    /// Requires [`ExternalObsData::row_keys`] to be empty and
    /// `row_annotations.num_rows()` to equal **either** the file's physical
    /// `n_obs` (`header.n_obs`; row `i` annotates physical row `i`, deleted rows
    /// included — `read_obs(logical=False)`) **or** its live row count (`n_obs`
    /// minus the deletion-vector popcount; row `i` annotates the `i`-th live
    /// row, deleted rows get `null` — `read_obs()` since pyscx 0.17). The two
    /// coincide when nothing is deleted. A live-length frame may carry no
    /// `row_embeddings` (a dense mapping has no null). A `status_column` is
    /// rejected: every row matches by construction, so the marker would be a
    /// constant.
    ///
    /// [`ExternalObsData::row_keys`] may be **empty or one per row**. Supplied,
    /// they are never joined on; they are an **order check**: when they are
    /// the file's own obs-index barcodes (of the rows the frame lands on) in a
    /// different order — a frame sorted or reindexed after `read_obs()`, right
    /// length, every value on the wrong cell — the attach is refused by row.
    /// Labels that are not the file's barcodes are ignored, as the index
    /// always was under positional. pyscx passes the frame's labelled pandas
    /// index; a RangeIndex frame checks nothing.
    Positional,
}

impl ObsJoinKey {
    /// Human-readable spec, as it would be typed on a command line.
    pub fn describe(&self) -> String {
        match self {
            ObsJoinKey::Auto => "<auto>".to_string(),
            ObsJoinKey::Column(c) => c.clone(),
            ObsJoinKey::Composite { columns } => columns.join(","),
            ObsJoinKey::Positional => "<positional>".to_string(),
        }
    }
}

pub struct AttachObsOptions {
    pub join_key: ObsJoinKey,
    pub missing_row_policy: MissingRowPolicy,
    pub extra_row_policy: ExtraRowPolicy,
    /// Obs column recording `"present"` / `"absent"` per row. `None` omits it.
    pub status_column: Option<String>,
    /// When false (default), any pre-existing obs column / obsm key / uns key
    /// this op would replace is an error.
    ///
    /// **Replace, never merge.** With `overwrite = true` a colliding column is
    /// dropped and rebuilt from this source alone; values the previous import
    /// wrote are gone. Importing N per-batch result tables therefore means
    /// concatenating them first and importing once — N successive overwriting
    /// imports would leave only the last batch. That is a correctness rule, not
    /// an ergonomic preference, and the loud failure is deliberate.
    pub overwrite: bool,
    /// Must be `0`. The file itself may be multimodal — obs is the shared global
    /// axis — but per-modality obs annotation is not a thing this op models.
    pub modality_id: u8,
    pub provenance_action: String,
    /// Merged into the provenance `params_json` object.
    pub provenance_params: Value,
    /// Run every validation and resolve the join, then return the summary
    /// **without writing anything**, so a caller can see `n_matched` before
    /// mutating a large file.
    pub dry_run: bool,
}

impl Default for AttachObsOptions {
    fn default() -> Self {
        Self {
            join_key: ObsJoinKey::Auto,
            missing_row_policy: MissingRowPolicy::default(),
            extra_row_policy: ExtraRowPolicy::default(),
            status_column: None,
            overwrite: false,
            modality_id: 0,
            provenance_action: "attach_external_obs".to_string(),
            provenance_params: Value::Null,
            dry_run: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AttachObsSummary {
    pub n_obs: u64,
    pub n_matched: u64,
    pub n_target_rows_absent: u64,
    pub n_source_rows_absent: u64,
    /// The resolved key, or the comma-joined composite spec.
    ///
    /// The **physical** column name, so the provenance entry records what
    /// actually keyed the join on disk. Consumers that show it to a user pass it
    /// through [`crate::display_key_name`] first — do not translate it here.
    pub obs_key_column: String,
    pub obs_columns_added: Vec<String>,
    pub obsm_keys_added: Vec<String>,
    /// Whether the obs predicate index had to be dropped because this import
    /// overwrote a column it covered. `true` means query pushdown on those
    /// columns is gone until the index is rebuilt.
    pub obs_index_dropped: bool,
    /// `true` when the obs axis was rewritten shard by shard (peak memory one
    /// shard); `false` when it had to be assembled whole, which is forced by a
    /// legacy single-section `ObsMetadata` and by nothing else. See
    /// [`ObsRewrite`].
    pub obs_streamed: bool,
}

/// What a key looks like on this file, and what would actually work.
///
/// Produced by [`diagnose_obs_key`] and folded into every key-related error, so
/// a failed import names the fix instead of leaving the caller to guess.
#[derive(Debug, Clone)]
pub struct KeyDiagnosis {
    pub n_obs: usize,
    /// The key that was resolved (or requested), if one could be resolved.
    pub resolved_key: Option<String>,
    /// Distinct values of `resolved_key`. Less than `n_obs` means it cannot join.
    pub resolved_cardinality: Option<usize>,
    /// Obs columns whose values are unique across all rows **and can serve as a
    /// join key** — the actionable list, best candidate first.
    ///
    /// Ordered: the obs index, then the [`OBS_KEY_FALLBACKS`] barcode spellings,
    /// then other string columns, then integers. Schema order is only the
    /// tiebreak — it used to be the whole ranking, which on a file carrying
    /// imported float score columns put a `*_score` first and the obs index last.
    pub unique_columns: Vec<String>,
    /// Unique 2-column composites, searched only when no single column is unique.
    pub unique_pairs: Vec<(String, String)>,
    /// Columns that are unique but cannot be a join key — in practice floats,
    /// plus any other type [`is_joinable_key_column`] rejects. (A nested column
    /// never reaches here: `distinct_count` cannot encode one, so it is dropped
    /// from the analysis entirely.)
    ///
    /// Reported separately rather than folded into [`Self::unique_columns`]:
    /// `resolve_key_column` refuses them, so offering one as the key to try is a
    /// dead end. Named rather than counted so a user who expected their unique
    /// float column to work learns which one was set aside.
    pub unusable_unique_columns: Vec<String>,
    /// Whether the pair search was capped (see [`DIAGNOSIS_PAIR_SEARCH_TOP_K`]).
    pub pair_search_capped: bool,
    /// A ready-to-paste key spec, when one exists.
    pub suggestion: Option<String>,
}

impl KeyDiagnosis {
    /// One-paragraph rendering appended to error messages.
    pub fn describe(&self) -> String {
        let mut s = String::new();
        if let (Some(k), Some(c)) = (&self.resolved_key, self.resolved_cardinality) {
            s.push_str(&format!(
                "key '{k}' has {c} distinct values over {} rows. ",
                self.n_obs
            ));
        }
        if !self.unique_columns.is_empty() {
            // "unique AND can key" rather than just "unique", because the list
            // is now filtered to keys the join accepts — a float column is
            // often unique per row and is deliberately absent from it.
            s.push_str(&format!(
                "Obs columns that ARE unique and can key a join: {:?}. ",
                self.unique_columns
            ));
        } else if !self.unique_pairs.is_empty() {
            let pairs: Vec<String> = self
                .unique_pairs
                .iter()
                .map(|(a, b)| format!("{a}+{b}"))
                .collect();
            s.push_str(&format!(
                "No single obs column is unique; unique 2-column composites: {pairs:?}. "
            ));
        } else {
            s.push_str(&format!(
                "No unique obs column{} was found — this file may have no usable join key. ",
                if self.pair_search_capped {
                    " (and the 2-column search was capped)"
                } else {
                    " or 2-column composite"
                }
            ));
        }
        // Without this the previous sentence reads as "nothing is unique" on a
        // file whose only unique column is a float, which is both false and
        // unactionable.
        if !self.unusable_unique_columns.is_empty() {
            s.push_str(&format!(
                "Unique but unusable as a key: {:?} — strings and integers work \
                 (both sides are fused as text), floats are refused because their \
                 text form is not guaranteed to agree across two independently \
                 written sides. ",
                self.unusable_unique_columns
            ));
        }
        if let Some(sug) = &self.suggestion {
            s.push_str(&format!("Try key = {sug}."));
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Composite keys
// ---------------------------------------------------------------------------

/// Fuse `columns` of `batch` into one key per row.
///
/// Exported so the readers in `scx-convert` build the source side with the
/// same code the op uses for the target side; the separator is then an
/// implementation detail neither side can disagree about.
pub fn build_composite_key(batch: &RecordBatch, columns: &[String]) -> Result<Vec<String>> {
    if columns.is_empty() {
        return Err(OpsError::InvalidInput(
            "composite join key needs at least one column".into(),
        ));
    }
    // Every component goes through the same alias resolution a single key does,
    // so `["sample_id", "obs_names"]` works and a composite can never disagree
    // with `ObsJoinKey::Column` about what a name means.
    let schema = batch.schema();
    let resolved: Vec<String> = columns
        .iter()
        .map(|c| resolve_key_alias("obs", &schema, c))
        .collect();
    for (name, requested) in resolved.iter().zip(columns) {
        if batch.column_by_name(name).is_none() {
            let present: Vec<String> = schema
                .fields()
                .iter()
                .map(|f| display_key_name("obs", f.name()))
                .collect();
            return Err(OpsError::KeyColumnUnresolved {
                axis: "obs",
                detail: format!(
                    "composite key column '{requested}' not found; columns present are {present:?}"
                ),
            });
        }
    }

    if resolved.len() == 1 {
        return string_column(batch, &resolved[0]);
    }

    let mut parts: Vec<Vec<String>> = Vec::with_capacity(resolved.len());
    for name in &resolved {
        parts.push(string_column(batch, name)?);
    }

    let n = batch.num_rows();
    let mut out = Vec::with_capacity(n);
    for row in 0..n {
        let mut key = String::new();
        for (ci, col) in parts.iter().enumerate() {
            // A component containing the separator would make the fused key
            // ambiguous: "a\x1fb" + "c" and "a" + "b\x1fc" collide. No real
            // barcode or accession contains a control character, so this is a
            // corrupt-input signal rather than a case to escape around.
            if col[row].contains(COMPOSITE_KEY_SEPARATOR) {
                return Err(OpsError::InvalidInput(format!(
                    "composite key component '{}' at row {row} contains the reserved \
                     separator (U+001F); the fused key would be ambiguous",
                    columns[ci]
                )));
            }
            if ci > 0 {
                key.push(COMPOSITE_KEY_SEPARATOR);
            }
            key.push_str(&col[row]);
        }
        out.push(key);
    }
    Ok(out)
}

/// Resolve an obs-style join-key column with the same preference order the
/// attach ops use: the pandas index column, then `OBS_KEY_FALLBACKS`.
///
/// Exported for readers that build the **source** side of a join (e.g.
/// `scx_convert::read_annotation_table`). Both sides must resolve keys
/// identically — if a reader picked `barcode` while the op picked `_index`, a
/// file that should have joined would fail with zero overlap and no obvious
/// cause.
pub fn resolve_obs_key_column(batch: &RecordBatch, requested: Option<&str>) -> Result<String> {
    resolve_key_column("obs", &batch.schema(), requested)
}

/// Materialize a column as owned `String`s, resolving dictionaries. Exported
/// alongside [`resolve_obs_key_column`] so a reader can build single-column
/// keys the same way the op does.
pub fn obs_key_values(batch: &RecordBatch, column: &str) -> Result<Vec<String>> {
    string_column(batch, column)
}

/// Remove the named columns from `batch`, forcing what remains nullable — the
/// shape [`ExternalObsData::row_annotations`] wants once the join-key columns
/// have been consumed (they are already on the target's obs axis, so
/// re-importing them would only duplicate them).
///
/// Exported for the bindings that build the source side from an in-memory
/// frame (pyscx `attach_obs_columns`, rscx `scx_attach_obs`), so they cannot
/// drift on which columns an attach keeps. A name in `drop` that no column
/// carries is ignored, deliberately: the physical `__index_level_0__` is
/// always dropped and not every frame has one.
pub fn drop_batch_columns(batch: &RecordBatch, drop: &[String]) -> Result<RecordBatch> {
    let schema = batch.schema();
    let mut fields = Vec::new();
    let mut arrays = Vec::new();
    for (i, f) in schema.fields().iter().enumerate() {
        if drop.iter().any(|d| d == f.name()) {
            continue;
        }
        // Nullable regardless: the attach op scatters nulls into every target
        // row the source does not cover. Field metadata rides along — it is
        // where a categorical's `ordered` bit lives.
        fields.push(
            Field::new(f.name(), f.data_type().clone(), true).with_metadata(f.metadata().clone()),
        );
        arrays.push(Arc::clone(batch.column(i)));
    }
    if fields.is_empty() {
        return Err(OpsError::InvalidInput(
            "no columns left to attach after removing the key column(s)".into(),
        ));
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .map_err(|e| OpsError::InvalidInput(format!("failed to build annotation columns: {e}")))
}

/// Resolve `join_key` against the obs **schema**, returning the spec to report
/// and the physical column(s) the keys are built from.
///
/// Split out from key materialization so the op can name the columns it needs
/// before deciding how to read them: with the physical names in hand it can ask
/// [`ScxReader::read_obs_keys`] for those columns alone instead of assembling
/// the whole obs table to reach one of them.
///
/// The reported spec is the physical name, not what the caller typed, so
/// `AttachObsSummary::obs_key_column` and the provenance entry stay in the
/// on-disk vocabulary even when the caller used `obs_names`.
fn resolve_target_key_spec(
    schema: &Schema,
    join_key: &ObsJoinKey,
) -> Result<(String, Vec<String>)> {
    match join_key {
        ObsJoinKey::Auto => {
            let col = resolve_key_column("obs", schema, None)?;
            Ok((col.clone(), vec![col]))
        }
        ObsJoinKey::Column(c) => {
            let col = resolve_key_column("obs", schema, Some(c))?;
            Ok((col.clone(), vec![col]))
        }
        ObsJoinKey::Composite { columns } => {
            if columns.is_empty() {
                return Err(OpsError::InvalidInput(
                    "composite join key needs at least one column".into(),
                ));
            }
            // Every component goes through the same alias resolution a single
            // key does, so `["sample_id", "obs_names"]` works — see
            // `build_composite_key`, which repeats this on the batch it is
            // handed and must not be able to disagree with it.
            let physical: Vec<String> = columns
                .iter()
                .map(|c| resolve_key_alias("obs", schema, c))
                .collect();
            // Existence is checked here, against the caller's own spelling.
            // `build_composite_key` also checks, but by the time it runs the
            // batch has been projected to `physical`, so its message would
            // quote a name the user never typed — and on the streaming path it
            // never runs at all, because the projected read fails first with a
            // bare "obs column not found". Single-column keys get this from
            // `resolve_key_column`; composites had nowhere else to get it.
            for (name, requested) in physical.iter().zip(columns) {
                if schema.field_with_name(name).is_err() {
                    let present: Vec<String> = schema
                        .fields()
                        .iter()
                        .map(|f| display_key_name("obs", f.name()))
                        .collect();
                    return Err(OpsError::KeyColumnUnresolved {
                        axis: "obs",
                        detail: format!(
                            "composite key column '{requested}' not found; \
                             columns present are {present:?}"
                        ),
                    });
                }
            }
            Ok((physical.join(","), physical))
        }
        // Handled by the positional branch in `attach_external_obs_inner`
        // before this is ever called; an error beats an unreachable!() panic if
        // a future caller reaches it anyway.
        ObsJoinKey::Positional => Err(OpsError::InvalidInput(
            "positional join has no key columns to resolve".into(),
        )),
    }
}

/// Build the target-side key strings from a batch already projected to
/// `physical` (or from the full obs table — either works, the columns are
/// addressed by name).
fn materialize_target_keys(batch: &RecordBatch, physical: &[String]) -> Result<Vec<String>> {
    if physical.len() == 1 {
        string_column(batch, &physical[0])
    } else {
        build_composite_key(batch, physical)
    }
}

/// Resolve the target-side keys for `join_key` from a materialized obs table,
/// returning `(spec, keys)`. Used by [`diagnose_obs_key`], which needs the whole
/// table anyway.
fn resolve_target_keys(obs: &RecordBatch, join_key: &ObsJoinKey) -> Result<(String, Vec<String>)> {
    let (spec, physical) = resolve_target_key_spec(&obs.schema(), join_key)?;
    let keys = materialize_target_keys(obs, &physical)?;
    Ok((spec, keys))
}

// ---------------------------------------------------------------------------
// Key diagnosis
// ---------------------------------------------------------------------------

/// Distinct-value count for a set of columns taken together.
///
/// Uses arrow's row format rather than casting to `Utf8`: it is dtype-agnostic
/// (dictionaries, numerics, booleans, timestamps all encode), avoids a String
/// allocation per cell, and handles multi-column keys in one pass. Returns
/// `None` for a column the row encoder does not support (nested types), so the
/// caller can report it as un-analysable rather than failing the diagnosis.
fn distinct_count(columns: &[ArrayRef]) -> Option<usize> {
    let fields: Vec<SortField> = columns
        .iter()
        .map(|c| SortField::new(c.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields).ok()?;
    let rows = converter.convert_columns(columns).ok()?;
    let mut seen: HashSet<Vec<u8>> = HashSet::with_capacity(rows.num_rows());
    for r in rows.iter() {
        seen.insert(r.as_ref().to_vec());
    }
    Some(seen.len())
}

/// Build a [`KeyDiagnosis`] from an already-read obs batch.
///
/// Only ever called on an error path (or from the public [`diagnose_obs_key`]),
/// so the per-column passes never cost anything on a successful import.
fn diagnose_from_obs(obs: &RecordBatch, resolved: Option<(&str, &[String])>) -> KeyDiagnosis {
    let n_obs = obs.num_rows();
    let schema = obs.schema();

    let (resolved_key, resolved_cardinality) = match resolved {
        Some((name, keys)) => {
            let uniq = keys.iter().collect::<HashSet<_>>().len();
            (Some(name.to_string()), Some(uniq))
        }
        None => (None, None),
    };

    // Per-column cardinality, cheapest signal first. Non-joinable columns
    // (floats) stay out of the candidate lists entirely — `resolve_key_column`
    // refuses them, so naming one as the key to try is a guaranteed dead end —
    // but a *unique* one is kept aside for the summary, because "no unique
    // column was found" would otherwise be a false statement about the file.
    let mut cardinalities: Vec<(String, usize)> = Vec::new();
    let mut unusable: Vec<String> = Vec::new();
    for (i, f) in schema.fields().iter().enumerate() {
        let Some(c) = distinct_count(&[obs.column(i).clone()]) else {
            continue;
        };
        if is_joinable_key_column(f.data_type()) {
            cardinalities.push((f.name().clone(), c));
        } else if c == n_obs && n_obs > 0 {
            unusable.push(f.name().clone());
        }
    }
    unusable.truncate(DIAGNOSIS_MAX_COLUMNS);

    // The axis index sorts first among usable keys: it is the identity every
    // external tool hands back, so preferring it is what lets `export_batches`
    // and `obs_import` agree on a key without either hard-coding one.
    let index_col = scx_format_io::resolve_index_columns(&schema)
        .into_iter()
        .next();
    let rank = |name: &str| -> (u8, usize) {
        if index_col.as_deref() == Some(name) {
            return (0, 0);
        }
        if let Some(pos) = OBS_KEY_FALLBACKS.iter().position(|c| *c == name) {
            return (1, pos);
        }
        let is_str = schema
            .field_with_name(name)
            .map(|f| is_string_column(f.data_type()))
            .unwrap_or(false);
        if is_str {
            (2, 0)
        } else {
            (3, 0)
        }
    };

    let mut unique_columns: Vec<String> = cardinalities
        .iter()
        .filter(|(_, c)| *c == n_obs && n_obs > 0)
        .map(|(n, _)| n.clone())
        .collect();
    // Stable, so schema order remains the tiebreak inside a tier.
    unique_columns.sort_by_key(|n| rank(n));
    unique_columns.truncate(DIAGNOSIS_MAX_COLUMNS);

    let mut unique_pairs = Vec::new();
    let mut pair_search_capped = false;
    if unique_columns.is_empty() && n_obs > 0 {
        // Only the highest-cardinality columns can combine into a unique key,
        // and an exhaustive C² sweep over a wide atlas obs is not worth it.
        let mut ranked = cardinalities.clone();
        ranked.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        // A single-valued column adds nothing to a composite.
        ranked.retain(|(_, c)| *c > 1);
        pair_search_capped = ranked.len() > DIAGNOSIS_PAIR_SEARCH_TOP_K;
        ranked.truncate(DIAGNOSIS_PAIR_SEARCH_TOP_K);

        'outer: for a in 0..ranked.len() {
            for b in (a + 1)..ranked.len() {
                let (ca, cb) = (&ranked[a].0, &ranked[b].0);
                let (Some(ia), Some(ib)) = (schema.index_of(ca).ok(), schema.index_of(cb).ok())
                else {
                    continue;
                };
                if distinct_count(&[obs.column(ia).clone(), obs.column(ib).clone()]) == Some(n_obs)
                {
                    unique_pairs.push((ca.clone(), cb.clone()));
                    if unique_pairs.len() >= DIAGNOSIS_MAX_PAIRS {
                        break 'outer;
                    }
                }
            }
        }
    }

    // Everything below is user-facing, so the physical index field name is
    // translated to `obs_names` here — the one place it can be done for the
    // Python dict, the CLI print and every error `describe()` feeds at once.
    // `resolve_key_column` accepts `obs_names` back, so each name reported is a
    // name that can be pasted into `key=`.
    let disp = |n: &String| display_key_name("obs", n);
    let unique_columns: Vec<String> = unique_columns.iter().map(disp).collect();
    let unique_pairs: Vec<(String, String)> = unique_pairs
        .iter()
        .map(|(a, b)| (disp(a), disp(b)))
        .collect();
    let unusable_unique_columns: Vec<String> = unusable.iter().map(disp).collect();

    let suggestion = unique_columns
        .first()
        .cloned()
        .or_else(|| unique_pairs.first().map(|(a, b)| format!("{a},{b}")));

    KeyDiagnosis {
        n_obs,
        resolved_key: resolved_key.as_ref().map(disp),
        resolved_cardinality,
        unique_columns,
        unique_pairs,
        unusable_unique_columns,
        pair_search_capped,
        suggestion,
    }
}

/// Report which obs columns could serve as a join key on `path`.
///
/// Read-only. Pass the key you intended to use as `requested` to have its
/// cardinality reported alongside the alternatives; pass `None` to just
/// enumerate what is available.
pub fn diagnose_obs_key(path: &Path, requested: Option<&ObsJoinKey>) -> Result<KeyDiagnosis> {
    let reader = ScxReader::open(path)?;
    let obs = reader.read_obs()?;
    // A bad requested key must not abort the diagnosis — reporting the
    // alternatives is the whole point.
    let resolved = requested.and_then(|k| resolve_target_keys(&obs, k).ok());
    Ok(match &resolved {
        Some((spec, keys)) => diagnose_from_obs(&obs, Some((spec.as_str(), keys))),
        None => diagnose_from_obs(&obs, None),
    })
}

// ---------------------------------------------------------------------------
// Join
// ---------------------------------------------------------------------------

struct ObsRowJoin {
    /// For each target row, the source row that supplies it.
    source_of_target: Vec<Option<u32>>,
    n_matched: u64,
    n_target_absent: u64,
    n_source_absent: u64,
}

/// The failure-path [`KeyDiagnosis`] text, read from `reader` on demand.
///
/// The diagnosis — "here are the columns that *would* work as a key" — is the
/// feature that stops a merged atlas with a duplicated obs index stranding the
/// user, so it is worth the whole obs table. But it needs every column and runs
/// a `RowConverter` + `HashSet` pass per column, so it must not be paid for on a
/// successful import. Calling it only from the arms that print it is what keeps
/// the read off the happy path, and it is the one unbounded read left in either
/// import.
///
/// Returns an empty string when obs cannot be read: a failed read must not
/// replace the join error the user actually needs with an I/O error raised by
/// the code trying to explain it.
fn describe_key_diagnosis(reader: &ScxReader, resolved: Option<(&str, &[String])>) -> String {
    match reader.read_obs() {
        Ok(obs) => diagnose_from_obs(&obs, resolved).describe(),
        Err(_) => String::new(),
    }
}

fn build_obs_row_join(
    reader: &ScxReader,
    key_spec: &str,
    target_keys: &[String],
    source_keys: &[String],
    opts: &AttachObsOptions,
) -> Result<ObsRowJoin> {
    let dups = first_duplicates(target_keys);
    if !dups.is_empty() {
        let diag = describe_key_diagnosis(reader, Some((key_spec, target_keys)));
        return Err(OpsError::DuplicateJoinKey {
            axis: "obs",
            detail: format!(
                "target obs key '{}' contains duplicates: {dups:?}. {diag}",
                display_key_name("obs", key_spec),
            ),
        });
    }
    let dups = first_duplicates(source_keys);
    if !dups.is_empty() {
        return Err(OpsError::DuplicateJoinKey {
            axis: "obs",
            detail: format!("source row keys contain duplicates: {dups:?}"),
        });
    }

    let index: HashMap<&str, u32> = source_keys
        .iter()
        .enumerate()
        .map(|(i, k)| (k.as_str(), i as u32))
        .collect();

    let mut source_of_target = Vec::with_capacity(target_keys.len());
    let mut used = vec![false; source_keys.len()];
    let mut n_matched = 0u64;
    let mut n_target_absent = 0u64;
    for key in target_keys {
        match index.get(key.as_str()) {
            Some(&src) => {
                used[src as usize] = true;
                n_matched += 1;
                source_of_target.push(Some(src));
            }
            None => {
                n_target_absent += 1;
                source_of_target.push(None);
            }
        }
    }

    if n_matched == 0 {
        let diag = describe_key_diagnosis(reader, Some((key_spec, target_keys)));
        return Err(OpsError::AxisMismatch {
            axis: "obs",
            detail: format!(
                "no target row key matched any source row key on '{}'. Target \
                 examples: {:?}; source examples: {:?}. Check for a sample-name prefix \
                 or a '-1' suffix difference. {diag}",
                display_key_name("obs", key_spec),
                examples(target_keys),
                examples(source_keys),
            ),
        });
    }

    let n_source_absent = used.iter().filter(|u| !**u).count() as u64;
    if n_source_absent > 0 && opts.extra_row_policy == ExtraRowPolicy::Error {
        return Err(OpsError::AxisMismatch {
            axis: "obs",
            detail: format!(
                "{n_source_absent} source rows have no matching target row \
                 (extra_row_policy = Error)"
            ),
        });
    }
    if n_target_absent > 0 && opts.missing_row_policy == MissingRowPolicy::Error {
        return Err(OpsError::AxisMismatch {
            axis: "obs",
            detail: format!(
                "{n_target_absent} target rows have no matching source row \
                 (missing_row_policy = Error)"
            ),
        });
    }
    if let Some((level, msg)) = obs_join_coverage_report(
        n_matched,
        target_keys.len() as u64,
        n_target_absent,
        n_source_absent,
        &display_key_name("obs", key_spec),
        || {
            // Examples from the keys that did NOT match — a target "100000"
            // beside a source "100000-1" names the suffix instantly, where head
            // examples would print two identical matching keys.
            let own = |v: Vec<&str>| v.into_iter().map(str::to_string).collect::<Vec<_>>();
            let target_matched: Vec<bool> = source_of_target.iter().map(|s| s.is_some()).collect();
            (
                own(unmatched_examples(target_keys, &target_matched)),
                own(unmatched_examples(source_keys, &used)),
            )
        },
    ) {
        log::log!(level, "{msg}");
    }

    Ok(ObsRowJoin {
        source_of_target,
        n_matched,
        n_target_absent,
        n_source_absent,
    })
}

/// How to report a join in which fewer than half the target rows matched, or
/// `None` when coverage is high enough to need no comment (dogfood E2).
///
/// An intentionally partial import and a broken join both leave most target rows
/// unmatched, but they are not the same event and must not read the same. The
/// discriminator is **`n_source_absent`**: when it is zero, every source row
/// found a home, so the source simply covers a subset of the target — the
/// documented per-batch workflow (run a caller on 6 of 116 batches, concatenate,
/// import once), and exactly what the user asked for.
///
/// That case reports *coverage* at `Info`, and deliberately **omits the
/// target/source example pair**. Both lists are valid keys drawn from the same
/// space, so presenting them side by side — the standard shape of a
/// key-format-mismatch diagnostic — reads as evidence of divergence and sends
/// the user hunting a bug that does not exist. The examples stay on the branch
/// where source rows genuinely failed to land, which is the branch where a
/// prefix / `-1`-suffix difference is actually plausible.
///
/// `examples` is a closure so the (allocating) example extraction is skipped
/// entirely on the coverage branch and on the no-report path.
fn obs_join_coverage_report(
    n_matched: u64,
    n_target_total: u64,
    n_target_absent: u64,
    n_source_absent: u64,
    key_name: &str,
    examples: impl FnOnce() -> (Vec<String>, Vec<String>),
) -> Option<(log::Level, String)> {
    if n_matched * 2 >= n_target_total {
        return None;
    }
    if n_source_absent == 0 {
        return Some((
            log::Level::Info,
            format!(
                "external obs join coverage: {n_matched} of {n_target_total} target rows \
                 matched on '{key_name}'; {n_target_absent} rows left null. Every source \
                 row matched, so this is a partial-coverage import, not a key mismatch."
            ),
        ));
    }
    let (target_examples, source_examples) = examples();
    Some((
        log::Level::Warn,
        format!(
            "external obs join matched only {n_matched} of {n_target_total} target rows \
             on '{key_name}', and {n_source_absent} source rows matched no target row — \
             check for a sample-name prefix or a '-1' suffix difference; target examples \
             {target_examples:?}, source examples {source_examples:?}"
        ),
    ))
}

/// Scatter a source-row-indexed array onto target rows `[row_start, row_end)`,
/// `null` for unmatched rows. `null`, not a zero: a score of `0.0` on a cell the
/// tool never saw is a claim it never made.
///
/// The row window is what lets the whole build run one obs shard at a time: the
/// output is shard-length, not `n_obs`-length.
fn scatter(
    col: &ArrayRef,
    join: &ObsRowJoin,
    row_start: usize,
    row_end: usize,
) -> Result<ArrayRef> {
    let take_idx: UInt32Array = join.source_of_target[row_start..row_end]
        .iter()
        .copied()
        .collect();
    arrow::compute::take(col.as_ref(), &take_idx, None)
        .map_err(|e| OpsError::InvalidInput(format!("failed to scatter annotation column: {e}")))
}

// ---------------------------------------------------------------------------
// Building the new sections
// ---------------------------------------------------------------------------

fn planned_obs_columns(data: &ExternalObsData, opts: &AttachObsOptions) -> Vec<String> {
    let mut cols = Vec::new();
    if let Some(name) = &opts.status_column {
        cols.push(name.clone());
    }
    cols.extend(
        data.row_annotations
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone()),
    );
    cols
}

/// Append the import's columns to `obs`, which covers global obs rows
/// `[row_start, row_start + obs.num_rows())`.
///
/// `row_start` is the only thing that distinguishes the streaming path from the
/// materializing one: the latter passes the whole table at `row_start = 0`. One
/// implementation, two callers — the alternative is two builders that agree
/// until one of them is edited.
fn build_new_obs(
    obs: &RecordBatch,
    data: &ExternalObsData,
    opts: &AttachObsOptions,
    join: &ObsRowJoin,
    row_start: usize,
) -> Result<RecordBatch> {
    let row_end = row_start + obs.num_rows();
    if row_end > join.source_of_target.len() {
        return Err(OpsError::ShapeMismatch {
            detail: format!(
                "obs rows [{row_start}, {row_end}) overrun the {}-row join built \
                 from the header's n_obs",
                join.source_of_target.len()
            ),
        });
    }
    let mut fields: Vec<Field> = obs
        .schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    let mut columns: Vec<ArrayRef> = obs.columns().to_vec();

    // Drop any column we are about to add (the overwrite path).
    let planned = planned_obs_columns(data, opts);
    let keep: Vec<bool> = fields.iter().map(|f| !planned.contains(f.name())).collect();
    fields = fields
        .into_iter()
        .zip(keep.iter())
        .filter_map(|(f, k)| k.then_some(f))
        .collect();
    columns = columns
        .into_iter()
        .zip(keep.iter())
        .filter_map(|(c, k)| k.then_some(c))
        .collect();

    if let Some(name) = &opts.status_column {
        let arr: StringArray = join.source_of_target[row_start..row_end]
            .iter()
            .map(|o| Some(if o.is_some() { "present" } else { "absent" }))
            .collect();
        fields.push(Field::new(name, DataType::Utf8, false));
        columns.push(Arc::new(arr) as ArrayRef);
    }

    for (i, f) in data.row_annotations.schema().fields().iter().enumerate() {
        let scattered = scatter(data.row_annotations.column(i), join, row_start, row_end)?;
        // Unmatched rows become null, so the field must admit nulls regardless
        // of how the source declared it. The type and the field metadata are
        // the source's: a categorical lands as a dictionary, `ordered` intact.
        fields.push(
            Field::new(f.name(), f.data_type().clone(), true).with_metadata(f.metadata().clone()),
        );
        columns.push(scattered);
    }

    let schema = Arc::new(Schema::new(fields).with_metadata(obs.schema().metadata().clone()));
    RecordBatch::try_new(schema, columns)
        .map_err(|e| OpsError::InvalidInput(format!("failed to build new obs: {e}")))
}

// ---------------------------------------------------------------------------
// The obs rewrite
// ---------------------------------------------------------------------------

/// Read the join key column(s) and validate the obs shard cover in one pass,
/// before anything is written.
///
/// [`ScxReader::read_obs_keys`] on its own is not enough, and the gap is narrow
/// enough to be worth spelling out. It walks the *stamps* — so a gap, a
/// reordering, or a total that disagrees with `n_obs` is refused — but it never
/// compares a shard's stamp with the number of rows that shard actually
/// carries. Those two agree on every file a writer produces, and they can
/// disagree in a way nothing downstream notices when the errors **cancel**: a
/// shard stamped 6 rows carrying 4, beside one stamped 4 carrying 6, tiles
/// `[0, 10)` by stamps and sums to 10 by payload. The caller's
/// `n_obs`-vs-header check passes, and the only thing left that would catch it
/// used to live inside the write loop — after the dry-run return, and after the
/// first bytes had been appended.
///
/// That made `dry_run` a liar on exactly this input: it reported a clean join
/// and the real import then failed. Doing the check here restores the contract
/// three separate surfaces promise — "`--dry-run` runs every validation", "a
/// rejected import leaves the file byte-identical" — at no extra I/O, because
/// this pass is the projected read the join needed anyway.
pub(crate) fn read_obs_keys_validated(
    reader: &ScxReader,
    obs_schema: &Schema,
    key_columns: &[String],
    n_obs: u64,
) -> Result<RecordBatch> {
    let n_shards = reader.obs_metadata_shard_count();
    if n_shards == 0 {
        // A legacy single section has no per-shard stamps to contradict, so
        // there is nothing extra to check — and no per-shard reader either.
        return Ok(reader.read_obs_keys(key_columns)?);
    }

    let projection: Vec<usize> = key_columns
        .iter()
        .map(|name| {
            obs_schema.index_of(name).map_err(|_| {
                OpsError::InvalidInput(format!("obs column '{name}' disappeared before the read"))
            })
        })
        .collect::<Result<_>>()?;

    let mut cover = crate::compact::ShardCoverCheck::default();
    let mut shards: Vec<(u32, RecordBatch)> = Vec::with_capacity(n_shards);
    for idx in 0..n_shards as u32 {
        let projected = reader.read_obs_shard_projected(idx, &projection)?;
        cover.visit("obs_metadata", &projected)?;
        // Arrow IPC column projection hands back a zero-copy slice into the
        // shard's whole message body, so retaining it would keep every
        // un-projected column resident and the projection would save nothing.
        // `compact_key_shard` rebuilds into fresh buffers — the same move
        // `read_obs_keys` makes, and the reason this loop is not a memory
        // regression over it.
        shards.push((idx, scx_format_io::compact_key_shard(&projected)?));
    }
    cover.finish("obs_metadata", n_obs)?;
    Ok(scx_format_io::assemble_sharded_metadata(
        "obs_metadata",
        shards,
    )?)
}

/// How the target's obs axis was rewritten. Reported on both attach summaries
/// and recorded in provenance, so "did this import hold the whole obs table"
/// is answerable after the fact rather than inferred from the file's layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObsRewrite {
    /// One output shard per input shard; peak memory is one shard. Requires a
    /// sharded input — the only layout with a per-shard reader.
    Streamed,
    /// The whole obs table was assembled, appended to, and re-sharded. The only
    /// option for a legacy single-section `ObsMetadata`: one Arrow IPC section
    /// is one batch, so there is nothing to stream.
    Materialized,
}

/// Rewrite the target's obs axis one shard at a time, appending the columns
/// `build` produces for each shard's global row range.
///
/// Shared by both external imports so they cannot drift on how the row offset
/// is derived — the running cursor here is the whole reason a shard's
/// annotations land on its own rows.
///
/// Emits one output shard per input shard, preserving the input's boundaries
/// rather than re-deriving them from `header.shard_target_rows`. That matches
/// `compact` and `optimize`, and it is what makes the rewrite streamable at all.
///
/// [`crate::compact::ShardCoverCheck`] runs here as well as in
/// [`read_obs_keys_validated`], which is what actually refuses a bad axis
/// before any byte is written. Keeping it in the write loop is not redundancy
/// for its own sake: this is the loop that derives each output shard's range
/// from the rows it can *see*, so it is the one place where a stamp the payload
/// does not honour turns into a well-formed file whose obs is bound to
/// different matrix rows than the input claimed. The pre-write pass makes that
/// unreachable; this makes it unreachable even if a future caller forgets the
/// pre-write pass.
pub(crate) fn write_obs_shards_appending<F>(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    n_obs: u64,
    mut build: F,
) -> Result<()>
where
    F: FnMut(&RecordBatch, usize) -> Result<RecordBatch>,
{
    let mut cover = crate::compact::ShardCoverCheck::default();
    let mut row_start = 0usize;
    for (out_idx, res) in reader.obs_shards().enumerate() {
        let out_idx = out_idx as u32;
        let shard = res?;
        cover.visit("obs_metadata", &shard)?;
        let n = shard.num_rows();
        // Written as built: a dictionary (categorical) column goes back to
        // disk as a dictionary, with its field metadata. The materializing
        // path slices one rebuilt table instead, and both produce the same
        // per-shard bytes for the columns they share.
        let built = build(&shard, row_start)?;
        writer.write_obs_shard(out_idx, row_start as u64, n as u64, n_obs, &built)?;
        row_start += n;
    }
    cover.finish("obs_metadata", n_obs)?;
    Ok(())
}

/// Write an already-assembled obs table as shards of `shard_target_rows`.
/// The legacy single-section path, kept byte-compatible with what both imports
/// have always emitted.
///
/// The row count is checked here rather than at each call site. `n_obs` is
/// stamped into every output shard as `n_rows_total`, so a short `obs` would
/// emit shards covering fewer rows than they claim — a malformed axis produced
/// by the code meant to preserve one. `build_new_obs`'s own bound catches an
/// over-run; this is the other direction, and putting it in the shared writer is
/// what stops the two ops from disagreeing about whether it is checked at all.
pub(crate) fn write_obs_shards_from_whole(
    writer: &mut ScxWriter,
    obs: &RecordBatch,
    n_obs: u64,
    shard_target_rows: usize,
) -> Result<()> {
    let n = obs.num_rows();
    if n as u64 != n_obs {
        return Err(OpsError::ShapeMismatch {
            detail: format!("obs has {n} rows but the header declares n_obs = {n_obs}"),
        });
    }
    // Both call sites already `.max(1)` the header value, so this is not
    // reachable today — but the loop below advances its cursor by `take`, so a
    // zero would hang rather than fail, and hanging is the one outcome a caller
    // cannot diagnose. Cheap insurance on new `pub(crate)` surface.
    // (Round-2 finding: Grok, Gemini.)
    if shard_target_rows == 0 {
        return Err(OpsError::InvalidInput(
            "shard_target_rows must be non-zero to write obs shards".into(),
        ));
    }
    let mut cursor = 0usize;
    let mut idx = 0u32;
    while cursor < n {
        let take = std::cmp::min(shard_target_rows, n - cursor);
        writer.write_obs_shard(
            idx,
            cursor as u64,
            take as u64,
            n_obs,
            &obs.slice(cursor, take),
        )?;
        idx += 1;
        cursor += take;
    }
    Ok(())
}

/// Warn once when a target's obs cannot be streamed, naming the remedy.
pub(crate) fn warn_unstreamable_obs(op: &str, n_obs: u64) {
    log::warn!(
        "{op}: the target's obs is a single legacy section ({n_obs} rows), which has no \
         per-shard reader — the whole obs table is held in memory for this import. \
         Run `scx optimize` on the file first to migrate it to the sharded layout."
    );
}

/// Merge `entries` into the file's existing `uns` at top level (shallow: an
/// entry replaces a same-named key wholesale). Shared by the obs and layer
/// attach ops so the two cannot drift.
///
/// The existing blob must be a JSON object — callers pass `{}` for a file with
/// no `uns` section. Anything else is refused *before any write*: "merge into
/// an array" has no meaning, and the single-key path this replaced used to
/// swap a non-object `uns` for `{}` in silence, which was data loss.
pub(crate) fn merge_uns_entries(
    mut uns: Value,
    entries: &serde_json::Map<String, Value>,
) -> Result<Value> {
    if entries.is_empty() {
        return Ok(uns);
    }
    let Some(map) = uns.as_object_mut() else {
        return Err(OpsError::InvalidInput(
            "the file's uns is not a JSON object, so there is nothing to merge \
             into; replace it with set_uns first"
                .to_string(),
        ));
    };
    for (k, v) in entries {
        map.insert(k.clone(), v.clone());
    }
    Ok(uns)
}

/// The file's existing `uns`, as the base `entries` are merged into.
///
/// Not read at all when there is nothing to merge, so a file whose `uns`
/// section cannot be read still takes a plain column attach. An **absent**
/// section is `{}`. Any other read error — checksum, JSON, nesting depth —
/// propagates: treating it as "absent" would let an unrelated attach commit
/// only its own keys and orphan metadata the op never saw.
pub(crate) fn read_uns_for_merge(
    reader: &ScxReader,
    entries: &serde_json::Map<String, Value>,
) -> Result<Value> {
    if entries.is_empty() {
        return Ok(serde_json::json!({}));
    }
    match reader.read_uns() {
        Ok(v) => Ok(v),
        Err(scx_format_io::ScxError::SectionNotFound(_)) => Ok(serde_json::json!({})),
        Err(e) => Err(e.into()),
    }
}

/// The `overwrite = false` half of the uns contract: every entry must be a key
/// the file does not have yet. Names the first collision.
pub(crate) fn check_uns_collisions(
    uns: &Value,
    entries: &serde_json::Map<String, Value>,
) -> Result<()> {
    for key in entries.keys() {
        if uns.get(key).is_some() {
            return Err(OpsError::InvalidInput(format!(
                "uns key '{key}' already exists; pass overwrite=true to replace it"
            )));
        }
    }
    Ok(())
}

/// Scatter the source embeddings onto the full target row axis.
///
/// Unlike obs, this is **not** bounded: `write_obsm` emits one section, so the
/// batch has to be `n_obs` rows. Only reached when the caller supplies
/// embeddings — no doublet caller does — and bounding it needs sharded
/// `ObsmEmbeddingShard` writes.
fn build_obsm(data: &ExternalObsData, join: &ObsRowJoin) -> Result<Vec<(String, RecordBatch)>> {
    let n_obs = join.source_of_target.len();
    let mut out = Vec::new();
    for (name, batch) in &data.row_embeddings {
        let mut fields = Vec::new();
        let mut columns = Vec::new();
        for (i, f) in batch.schema().fields().iter().enumerate() {
            fields.push(Field::new(f.name(), f.data_type().clone(), true));
            columns.push(scatter(batch.column(i), join, 0, n_obs)?);
        }
        let schema = Arc::new(Schema::new(fields));
        let scattered = RecordBatch::try_new(schema, columns)
            .map_err(|e| OpsError::InvalidInput(format!("failed to scatter obsm '{name}': {e}")))?;
        out.push((name.clone(), scattered));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Validation and catalog bookkeeping
// ---------------------------------------------------------------------------

/// Which obs row space a positional frame is in, decided by its length alone.
///
/// Shared by the positional `attach_external_obs` and `modify_metadata(obs=)`
/// so the two ops cannot drift on the rule or on the words. `Physical` is
/// `n_obs_physical` rows (`header.n_obs`; what `read_obs(logical=False)`
/// returns); `Live` is the live count (`n_obs_physical` minus the keep mask's
/// deleted rows; what `read_obs()` returns since pyscx 0.17) and only exists
/// when the file has deletions — with none the two coincide and the frame is
/// `Physical`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ObsFrameRowSpace {
    Physical,
    Live,
}

impl ObsFrameRowSpace {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            ObsFrameRowSpace::Physical => "physical",
            ObsFrameRowSpace::Live => "logical",
        }
    }
}

/// Classify an `n_rows`-row positional frame against a file with
/// `n_physical` obs rows and keep mask `keep` (`None` = nothing deleted).
/// Any other length is a [`OpsError::ShapeMismatch`] naming both counts and
/// the read that yields each. `what` names the frame in the message
/// (`"positional attach: row_annotations"`, `"modify_metadata: obs"`).
pub(crate) fn classify_obs_frame_length(
    what: &str,
    n_rows: u64,
    n_physical: u64,
    keep: Option<&[bool]>,
) -> Result<ObsFrameRowSpace> {
    let n_live = keep.map_or(n_physical, |k| k.iter().filter(|b| **b).count() as u64);
    if n_rows == n_physical {
        return Ok(ObsFrameRowSpace::Physical);
    }
    if keep.is_some() && n_rows == n_live {
        return Ok(ObsFrameRowSpace::Live);
    }
    let file = if n_live == n_physical {
        format!("n_obs = {n_physical} (no logical deletions)")
    } else {
        format!(
            "n_obs = {n_live} live rows and n_obs_physical = {n_physical} ({} logically \
             deleted). Pass {n_live} rows (read_obs()) to address the live rows, or \
             {n_physical} rows (read_obs(logical=False)) to address every physical row",
            n_physical - n_live
        )
    };
    Err(OpsError::ShapeMismatch {
        detail: format!("{what} has {n_rows} rows but the file has {file}"),
    })
}

fn validate_shape(data: &ExternalObsData, positional: bool) -> Result<()> {
    if positional
        && !data.row_keys.is_empty()
        && data.row_keys.len() != data.row_annotations.num_rows()
    {
        return Err(OpsError::ShapeMismatch {
            detail: format!(
                "positional attach: row_keys has {} entries but row_annotations has {} rows — \
                 under positional the keys are an alignment check on the frame, one per row",
                data.row_keys.len(),
                data.row_annotations.num_rows()
            ),
        });
    }
    // Under positional the annotations ARE the row count; keyed mode measures
    // everything against the keys.
    let n_rows = if positional {
        data.row_annotations.num_rows()
    } else {
        data.row_keys.len()
    };
    if data.row_annotations.num_rows() != n_rows {
        return Err(OpsError::ShapeMismatch {
            detail: format!(
                "row_annotations has {} rows but there are {n_rows} row keys",
                data.row_annotations.num_rows()
            ),
        });
    }
    for (name, batch) in &data.row_embeddings {
        if batch.num_rows() != n_rows {
            return Err(OpsError::ShapeMismatch {
                detail: format!(
                    "row embedding '{name}' has {} rows but there are {n_rows} row keys",
                    batch.num_rows()
                ),
            });
        }
    }
    Ok(())
}

/// Whether any column this op is about to write is covered by the existing obs
/// predicate index, which would leave the index describing stale values.
///
/// Precise rather than conservative on purpose: dropping the index on every
/// import would silently kill query pushdown for callers who only ever *add*
/// columns, which is the overwhelmingly common case.
pub(crate) fn obs_index_would_go_stale(reader: &ScxReader, planned: &[String]) -> Result<bool> {
    let existing = indexed_column_names(reader.read_obs_predicate_index_bytes()?)?;
    Ok(existing.iter().any(|name| planned.contains(name)))
}

/// The column names a serialised predicate index covers, in index order.
/// `None` bytes (no such section) yield an empty list.
///
/// Shared by [`obs_index_would_go_stale`] — which asks whether an import would
/// invalidate one — and `modify_metadata`, which asks what to rebuild so a
/// replaced axis keeps the index it had. Both need exactly this walk, and the
/// two drifting on which `IndexedColumn` arms carry a name is a silent
/// correctness bug on either side.
pub(crate) fn indexed_column_names(bytes: Option<&[u8]>) -> Result<Vec<String>> {
    let Some(bytes) = bytes else {
        return Ok(Vec::new());
    };
    let index = scx_engine::PredicateIndex::read_from(&mut std::io::Cursor::new(bytes))?;
    Ok(index
        .columns
        .iter()
        .map(|c| match c {
            scx_engine::index::IndexedColumn::Categorical(cat) => cat.column_name.clone(),
            scx_engine::index::IndexedColumn::Numeric(num) => num.column_name.clone(),
        })
        .collect())
}

fn check_collisions(
    reader: &ScxReader,
    obs_schema: &Schema,
    uns: &Value,
    data: &ExternalObsData,
    obs_new: &[String],
) -> Result<()> {
    for name in obs_new {
        if obs_schema.field_with_name(name).is_ok() {
            return Err(OpsError::InvalidInput(format!(
                "obs column '{name}' already exists; pass overwrite=true to replace it. \
                 Note that overwrite REPLACES rather than merges — importing several \
                 per-batch tables one after another would keep only the last."
            )));
        }
    }
    check_uns_collisions(uns, &data.uns)?;
    if !data.row_embeddings.is_empty() {
        // Names only — `read_all_obsm` would decode every existing embedding
        // just to look at its key, which on an atlas is the largest thing this
        // op touches. `list_obsm` is a catalog scan. (Round-1 finding: codex.)
        let existing = reader.list_obsm();
        for (name, _) in &data.row_embeddings {
            if existing.iter().any(|k| k == name) {
                return Err(OpsError::InvalidInput(format!(
                    "obsm key '{name}' already exists; pass overwrite=true to replace it"
                )));
            }
        }
    }
    Ok(())
}

/// Whether an old catalog entry is superseded and must be dropped.
///
/// Deliberately narrow. `var`, CSR shards, layers, the CSC sidecar, deletion
/// vectors and detection bitmaps are all untouched by an obs-only attach and
/// must survive verbatim.
fn should_drop_old_entry(
    e: &FullCatalogEntry,
    rewrote_uns: bool,
    drop_obs_index: bool,
    obsm: &[(String, RecordBatch)],
) -> bool {
    use SectionType::*;
    if e.section_type == Provenance {
        return true;
    }
    if rewrote_uns && e.section_type == UnsBlob && e.modality_id == 0 {
        return true;
    }
    if matches!(e.section_type, ObsMetadata | ObsMetadataShard) {
        return true;
    }
    if drop_obs_index && e.section_type == ObsPredicateIndex {
        return true;
    }
    if matches!(e.section_type, ObsmEmbedding | ObsmEmbeddingShard)
        && obsm
            .iter()
            .any(|(k, _)| entry_matches_key(&e.name, "obsm", k))
    {
        return true;
    }
    false
}

fn build_params_json(
    data: &ExternalObsData,
    opts: &AttachObsOptions,
    s: &AttachObsSummary,
) -> Value {
    let mut v = serde_json::json!({
        "source_file": data.source_name,
        "obs_key_column": s.obs_key_column,
        "n_obs": s.n_obs,
        "n_matched": s.n_matched,
        "n_target_rows_absent": s.n_target_rows_absent,
        "n_source_rows_absent": s.n_source_rows_absent,
        "obs_columns": s.obs_columns_added,
        "obsm_keys": s.obsm_keys_added,
        "obs_predicate_index_dropped": s.obs_index_dropped,
        // Whether the obs rewrite streamed shard-by-shard or assembled the
        // whole table. Recorded because it is not recoverable from the output:
        // both paths produce a sharded obs, so a file that cost its own obs
        // table in RAM looks exactly like one that did not.
        "obs_streamed": s.obs_streamed,
        "uns_keys_merged": data.uns.keys().collect::<Vec<_>>(),
        "overwrite": opts.overwrite,
    });
    // A positional attach is accepted in either row space, told apart only by
    // length — so which one landed is not recoverable from the output (both
    // write a physical-length obs). Record it: a live-length attach nulls the
    // deleted rows, a physical one writes them.
    if opts.join_key == ObsJoinKey::Positional {
        let row_space = if s.n_target_rows_absent > 0 {
            "logical"
        } else {
            "physical"
        };
        if let Some(map) = v.as_object_mut() {
            map.insert("row_space".to_string(), Value::from(row_space));
            map.insert(
                "positional_index_checked".to_string(),
                Value::from(!data.row_keys.is_empty()),
            );
        }
    }
    if let (Some(map), Some(extra)) = (v.as_object_mut(), opts.provenance_params.as_object()) {
        for (k, val) in extra {
            map.insert(k.clone(), val.clone());
        }
    }
    v
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Attach `data` to the SCX file at `path` as obs columns plus optional
/// `obsm` / `uns`, joined to the target's own obs axis by key.
///
/// # What is checked before the first byte is written
///
/// The shape, the join, and the obs shard cover — the last via
/// [`read_obs_keys_validated`], during the join's projected key read. That
/// placement is load-bearing rather than incidental: the same check inside the
/// write loop would sit past the `dry_run` return, so a preview could report a
/// clean join for a file the real import then refuses.
///
/// **The preflight reads only the key column(s).** Arrow IPC projection skips
/// decoding the rest, so a *non-key* obs column that fails to decode, or a
/// per-shard schema that disagrees with shard 0's, is not seen until the write
/// pass — where it aborts with obs shards already appended at EOF. The catalog
/// is only swapped by `commit_in_place`, so the file still reads as it did and
/// `scx compact` reclaims the orphans; but a `dry_run` cannot promise that case
/// away. Note such a file is already unreadable through `read_obs()`, whose
/// `concat_batches` needs one shared schema — this op declines to be the thing
/// that discovers it. Making the preflight total would mean decoding every
/// column of every shard twice per import; that trade is deliberately not made
/// here. (Round-2 finding: codex.)
///
/// `X`, layers, the CSC sidecar, `.raw`, deletion vectors, detection bitmaps
/// and `var` are never read or rewritten; the whole import is undoable with
/// `scx rollback`.
pub fn attach_external_obs(
    path: &Path,
    data: &ExternalObsData,
    opts: &AttachObsOptions,
) -> Result<AttachObsSummary> {
    attach_external_obs_inner(path, None, data, opts)
}

/// [`attach_external_obs`] against a caller-supplied reader, so a test can hold
/// the reader the op actually used and assert on its
/// [`ScxReader::debug_counts`] — the only way to prove the whole obs table was
/// never assembled.
///
/// Test-only because the ordering differs: production opens the reader *after*
/// `prepare_in_place` has taken the exclusive lock, so the mmap cannot be stale
/// with respect to a concurrent appender. A caller that opens first gives that
/// up, which is fine in a single-process test and is not something to offer
/// callers generally.
#[cfg(test)]
pub(crate) fn attach_external_obs_with_reader(
    path: &Path,
    reader: &ScxReader,
    data: &ExternalObsData,
    opts: &AttachObsOptions,
) -> Result<AttachObsSummary> {
    attach_external_obs_inner(path, Some(reader), data, opts)
}

fn attach_external_obs_inner(
    path: &Path,
    injected_reader: Option<&ScxReader>,
    data: &ExternalObsData,
    opts: &AttachObsOptions,
) -> Result<AttachObsSummary> {
    let positional = opts.join_key == ObsJoinKey::Positional;
    validate_shape(data, positional)?;

    if positional && opts.status_column.is_some() {
        return Err(OpsError::InvalidInput(
            "status_column is meaningless under a positional attach — every row \
             matches by construction, so the marker would be a constant; drop \
             status_column or use a key join"
                .into(),
        ));
    }
    // A status column that shares a name with an annotation would produce TWO
    // obs columns with the same name: `check_collisions` compares planned
    // names against the OLD schema only, and `build_new_obs` pushes the
    // status field and the annotation field independently. (Round-2 finding:
    // codex.)
    if let Some(status) = &opts.status_column {
        if data
            .row_annotations
            .schema()
            .field_with_name(status)
            .is_ok()
        {
            return Err(OpsError::InvalidInput(format!(
                "status_column '{status}' is also an annotation column in the \
                 source; writing both would leave two obs columns with the \
                 same name. Rename one of them."
            )));
        }
    }

    let planned_obs = planned_obs_columns(data, opts);
    if planned_obs.is_empty() && data.row_embeddings.is_empty() && data.uns.is_empty() {
        return Err(OpsError::InvalidInput(
            "nothing to attach: row_annotations has no columns, and no status column, \
             obsm embedding or uns payload was supplied"
                .into(),
        ));
    }

    let (mut lock, mut prep) = prepare_in_place(path, opts.modality_id)?;

    // The file may be multimodal — obs is the shared global axis, and
    // `prepare_in_place` only validates non-zero modality ids. Per-modality obs
    // annotation is a different feature and is not modelled here.
    if opts.modality_id != 0 {
        return Err(OpsError::MultimodalUnsupported {
            op: "attach_external_obs",
        });
    }

    // The reader takes no lock of its own; we only ever append, never truncate,
    // so holding it across the transaction is safe.
    let owned_reader;
    let reader = match injected_reader {
        Some(r) => r,
        None => {
            owned_reader = ScxReader::open(path)?;
            &owned_reader
        }
    };
    let uns = read_uns_for_merge(reader, &data.uns)?;

    // A single legacy `ObsMetadata` section is one Arrow IPC batch with no
    // per-shard reader, so there is nothing to stream and the whole table has
    // to be assembled. Decided once, here, and reported on the summary.
    let obs_rewrite = if reader.obs_metadata_shard_count() > 0 {
        ObsRewrite::Streamed
    } else {
        ObsRewrite::Materialized
    };

    let n_obs = prep.old_n_obs;

    // --- Resolve the join key and build the join ---------------------------
    //
    // Only the key column(s) are read: `read_obs_keys` projects each obs shard
    // to them and compacts it, so peak memory is the key data rather than the
    // whole table. It also runs the same contiguous-cover validation
    // `read_obs()` did, which is what keeps the op's "every validation runs
    // before the first byte is written" contract intact — including the
    // n_obs-vs-header shape check below.
    let obs_schema = reader.read_obs_schema_physical()?;
    let (key_spec, row_join) = if positional {
        // No key to read — but the keyed path's projected read doubles as the
        // payload-vs-stamp preflight that keeps `dry_run` honest (see
        // `read_obs_keys_validated`), and positional must not lose it. Project
        // obs column 0: obs always carries at least its index column, and one
        // column is the cheapest read that still walks every shard.
        // Either row space is accepted, told apart by length (the shared
        // `classify_obs_frame_length`). Physical (`header.n_obs` rows) is the
        // identity join. Live (`n_obs` minus the deletion-vector popcount —
        // what `read_obs()` returns since pyscx 0.17) is built straight from
        // the keep mask: the `i`-th live row is source row `i`, a deleted row
        // has no source and lands `null` through the same `scatter` the key
        // join uses. Not routed through `build_obs_row_join`: its coverage
        // WARN ("check for a sample-name prefix") and its
        // `MissingRowPolicy::Error` arm are about keys that failed to match,
        // and a deleted row is not a failed match.
        if n_obs == 0 {
            return Err(OpsError::InvalidInput(
                "positional attach: the file has no obs rows, so there is nothing to \
                 annotate"
                    .into(),
            ));
        }
        let n_rows = data.row_annotations.num_rows() as u64;
        let keep = if n_rows == n_obs {
            None
        } else {
            reader.deletion_keep_mask()?
        };
        let row_space = classify_obs_frame_length(
            "positional attach: row_annotations",
            n_rows,
            n_obs,
            keep.as_deref(),
        )?;
        let n_live = keep
            .as_ref()
            .map_or(n_obs, |k| k.iter().filter(|b| **b).count() as u64);
        if row_space == ObsFrameRowSpace::Live && !data.row_embeddings.is_empty() {
            return Err(OpsError::InvalidInput(format!(
                "positional attach of a live-length frame ({n_live} of {n_obs} physical rows) \
                 cannot carry obsm embeddings: a dense mapping has no null to stand in for a \
                 deleted row. Pass {n_obs} rows, or attach the embeddings separately"
            )));
        }
        if n_obs > u32::MAX as u64 {
            return Err(OpsError::InvalidInput(format!(
                "positional attach supports at most {} obs rows (source rows \
                 are indexed by u32); the file has {n_obs}",
                u32::MAX
            )));
        }
        if obs_schema.fields().is_empty() {
            return Err(OpsError::InvalidInput(
                "obs has no columns to validate the shard cover against".into(),
            ));
        }
        let join = match (row_space, keep) {
            (ObsFrameRowSpace::Live, Some(keep)) => {
                let mut next_live = 0u32;
                let source_of_target: Vec<Option<u32>> = keep
                    .iter()
                    .map(|&live| {
                        if live {
                            let i = next_live;
                            next_live += 1;
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect();
                ObsRowJoin {
                    source_of_target,
                    n_matched: n_live,
                    n_target_absent: n_obs - n_live,
                    n_source_absent: 0,
                }
            }
            _ => ObsRowJoin {
                source_of_target: (0..n_obs).map(|i| Some(i as u32)).collect(),
                n_matched: n_obs,
                n_target_absent: 0,
                n_source_absent: 0,
            },
        };

        // The keyed path's projected read doubles as the payload-vs-stamp
        // preflight that keeps `dry_run` honest (see `read_obs_keys_validated`),
        // and positional must not lose it. With no keys supplied, project obs
        // column 0 — obs always carries at least its index column, and one
        // column is the cheapest read that still walks every shard. With keys
        // supplied (pyscx passes the frame's labelled pandas index), project
        // the file's obs index column instead and use the same read as an
        // **order check**: positional dispatch is by length alone, so a frame
        // that was sorted or reindexed after `read_obs()` has the right length
        // and every value on the wrong cell. Keys are never joined on.
        let index_col = if data.row_keys.is_empty() {
            None
        } else {
            scx_format_io::resolve_index_columns(&obs_schema)
                .into_iter()
                .find(|c| obs_schema.index_of(c).is_ok())
        };
        if !data.row_keys.is_empty() && index_col.is_none() {
            return Err(OpsError::InvalidInput(
                "positional attach: the frame carries an index to check the row order against, \
                 but the file's obs has no index column (pandas envelope or `__index_level_0__`); \
                 pass a frame with a RangeIndex to attach by position without the check"
                    .into(),
            ));
        }
        let probe_columns = vec![index_col
            .clone()
            .unwrap_or_else(|| obs_schema.field(0).name().clone())];
        let probe = read_obs_keys_validated(reader, &obs_schema, &probe_columns, n_obs)?;
        if probe.num_rows() as u64 != n_obs {
            return Err(OpsError::ShapeMismatch {
                detail: format!(
                    "obs has {} rows but the header declares n_obs = {n_obs}",
                    probe.num_rows()
                ),
            });
        }
        if let Some(index_col) = index_col {
            // The check fires only when the frame's labels ARE the file's
            // barcodes (of the rows it lands on) in a different order — a
            // `sort_values` / `reindex` after `read_obs()`. Labels that are
            // not the file's barcodes at all are ignored, as positional has
            // always ignored the index: a frame built from another source
            // with its own row labels is still "row i annotates row i".
            let target = crate::external_layer::string_column(&probe, &index_col)?;
            let landed: Vec<(usize, u32)> = join
                .source_of_target
                .iter()
                .enumerate()
                .filter_map(|(phys, src)| src.map(|s| (phys, s)))
                .collect();
            let in_order = landed
                .iter()
                .all(|(phys, src)| data.row_keys[*src as usize] == target[*phys]);
            if !in_order {
                let mut a: Vec<&str> = landed.iter().map(|(p, _)| target[*p].as_str()).collect();
                let mut b: Vec<&str> = data.row_keys.iter().map(String::as_str).collect();
                a.sort_unstable();
                b.sort_unstable();
                if a == b {
                    let (phys, src) = landed
                        .iter()
                        .find(|(p, s)| data.row_keys[*s as usize] != target[*p])
                        .expect("not in order ⇒ a mismatch exists");
                    return Err(OpsError::InvalidInput(format!(
                        "positional attach: the frame holds the file's own barcodes in a different \
                         order — frame row {src} is '{}' but obs row {phys} ('{index_col}') is \
                         '{}'. A frame sorted or reindexed after read_obs() cannot be landed \
                         positionally; restore the original order (or join by key instead)",
                        data.row_keys[*src as usize], target[*phys]
                    )));
                }
            }
        }
        (opts.join_key.describe(), join)
    } else {
        let (key_spec, key_columns) = resolve_target_key_spec(&obs_schema, &opts.join_key)?;
        let key_batch = read_obs_keys_validated(reader, &obs_schema, &key_columns, n_obs)?;
        if key_batch.num_rows() as u64 != n_obs {
            return Err(OpsError::ShapeMismatch {
                detail: format!(
                    "obs has {} rows but the header declares n_obs = {n_obs}",
                    key_batch.num_rows()
                ),
            });
        }
        let target_keys = materialize_target_keys(&key_batch, &key_columns)?;
        // The key diagnosis costs the whole obs table, so `build_obs_row_join`
        // takes the reader and pays for it only on the arms that print it.
        let row_join = build_obs_row_join(reader, &key_spec, &target_keys, &data.row_keys, opts)?;
        (key_spec, row_join)
    };

    // --- Collision checks ---------------------------------------------------
    if !opts.overwrite {
        check_collisions(reader, &obs_schema, &uns, data, &planned_obs)?;
    }

    // --- Would this invalidate the obs predicate index? ---------------------
    let drop_obs_index = obs_index_would_go_stale(reader, &planned_obs)?;

    // --- Build the sections that are not the obs axis -----------------------
    let rewrote_uns = !data.uns.is_empty();
    let new_uns = merge_uns_entries(uns, &data.uns)?;
    let obsm_batches = build_obsm(data, &row_join)?;

    // --- Capture catalog invariants before consuming old_catalog -----------
    let old_catalog_offset = prep.old_catalog_offset;
    let data_gen = prep.old_catalog.data_generation;
    let csc_gen = prep.old_catalog.csc_build_generation;
    let manifest = prep.header.manifest_sequence;
    let mt_off = prep.header.modality_table_offset;
    let mt_len = prep.header.modality_table_length;
    let shard_target_rows = (prep.header.shard_target_rows as usize).max(1);

    let mut summary = AttachObsSummary {
        n_obs,
        n_matched: row_join.n_matched,
        n_target_rows_absent: row_join.n_target_absent,
        n_source_rows_absent: row_join.n_source_absent,
        obs_key_column: key_spec,
        obs_columns_added: planned_obs,
        obsm_keys_added: obsm_batches.iter().map(|(k, _)| k.clone()).collect(),
        obs_index_dropped: drop_obs_index,
        obs_streamed: obs_rewrite == ObsRewrite::Streamed,
    };

    if opts.dry_run {
        // Everything above is validation and the join itself, so the caller
        // gets a real `n_matched` while the file stays untouched. Unlike the
        // layer op there is no nnz to predict, so the preview is exact by
        // construction.
        return Ok(summary);
    }

    // On the materializing path the obs table is read and rebuilt *here*,
    // before the writer exists, so a failure still leaves the file untouched.
    // (After the dry-run return, though — a preview should not pay for it.)
    // The streaming path builds inside the write loop by construction.
    let materialized_obs = match obs_rewrite {
        ObsRewrite::Streamed => None,
        ObsRewrite::Materialized => {
            warn_unstreamable_obs("obs import", n_obs);
            let obs = reader.read_obs()?;
            let built = build_new_obs(&obs, data, opts, &row_join, 0)?;
            drop(obs);
            Some(built)
        }
    };

    let mut prov_ops = read_provenance_ops(&mut lock, &prep.old_catalog)?;

    // --- Emit sections at EOF through an adopted writer --------------------
    let write_offset = lock.seek(SeekFrom::End(0))?;
    let cloned = lock.file().try_clone()?;
    let mut writer =
        ScxWriter::adopt_in_place(cloned, prep.header.clone(), write_offset, Vec::new())?;

    if rewrote_uns {
        writer.write_uns(&new_uns)?;
    }

    match &materialized_obs {
        None => write_obs_shards_appending(reader, &mut writer, n_obs, |shard, row_start| {
            build_new_obs(shard, data, opts, &row_join, row_start)
        })?,
        Some(new_obs) => {
            write_obs_shards_from_whole(&mut writer, new_obs, n_obs, shard_target_rows)?
        }
    }

    for (name, batch) in &obsm_batches {
        writer.write_obsm(name, batch)?;
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
        .filter(|e| !should_drop_old_entry(e, rewrote_uns, drop_obs_index, &obsm_batches))
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

    // The other half of `drop_obs_index`: the catalog also carries a per-shard
    // `ColumnStat` for each indexed obs column, and `scx_engine`'s Level-1
    // pushdown prunes from those directly — the numeric `MinMax` arm never looks
    // at the index section. Overwriting a column whose stats exist would leave
    // bounds describing values that are gone, and shards holding matching rows
    // would be silently excluded.
    //
    // Scoped to the rewritten columns, not wholesale: this op joins by key and
    // leaves every other obs column byte-identical, so their stats stay true.
    // Clearing them all would quietly disable pruning on a `cell_type` index
    // every time someone lands doublet calls.
    //
    // Unconditional, NOT gated on `drop_obs_index`: that flag is false when the
    // file has no index section, and a file can carry stats with no index (what
    // `modify_metadata` leaves behind). Gating here is what would let those
    // survive a second rewrite.
    scx_format_io::clear_csr_shard_column_stats_for(&mut entries, &summary.obs_columns_added);

    // `write_obsm` does not set the header flag, and `commit_in_place` writes
    // the header verbatim without `sync_from_catalog`. Without this, a
    // first-ever in-place obsm silently disappears on the next `compact` or
    // `subset`, both of which gate obsm copying on `header.has_obsm()`.
    if !obsm_batches.is_empty() {
        prep.header.set_obsm();
    }

    let new_catalog = FullCatalog {
        catalog_version: scx_format_io::CURRENT_CATALOG_VERSION,
        manifest_sequence: manifest + 1,
        prev_catalog_offset: old_catalog_offset,
        n_obs,
        entries,
        // X is never read or rewritten, so the CSC sidecar stays fresh. Bumping
        // either of these would silently invalidate it and drop every
        // `prefer_format=csc` / `gpu_csc_v3` route to a fallback.
        data_generation: data_gen,
        csc_build_generation: csc_gen,
    };

    commit_in_place(&mut lock, &mut prep.header, &new_catalog, mt_off, mt_len)?;

    summary.obsm_keys_added = obsm_batches.into_iter().map(|(k, _)| k).collect();
    Ok(summary)
}

#[cfg(test)]
#[path = "external_obs_tests.rs"]
mod tests;
