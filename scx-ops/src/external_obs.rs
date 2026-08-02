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

use crate::append::unify_dict_columns;
use crate::error::{OpsError, Result};
use crate::external_layer::{
    examples, first_duplicates, resolve_key_column, string_column, ExtraRowPolicy, MissingRowPolicy,
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
    /// Merged into `uns` under [`AttachObsOptions::uns_key`].
    pub uns: Option<Value>,
    /// BLAKE3 of the source file, recorded in the provenance entry.
    pub source_checksum: Option<[u8; 32]>,
    /// Display name of the source (usually a path basename).
    pub source_name: Option<String>,
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
}

impl ObsJoinKey {
    /// Human-readable spec, as it would be typed on a command line.
    pub fn describe(&self) -> String {
        match self {
            ObsJoinKey::Auto => "<auto>".to_string(),
            ObsJoinKey::Column(c) => c.clone(),
            ObsJoinKey::Composite { columns } => columns.join(","),
        }
    }
}

pub struct AttachObsOptions {
    pub join_key: ObsJoinKey,
    pub missing_row_policy: MissingRowPolicy,
    pub extra_row_policy: ExtraRowPolicy,
    /// Obs column recording `"present"` / `"absent"` per row. `None` omits it.
    pub status_column: Option<String>,
    /// `uns` key for [`ExternalObsData::uns`]. `None` omits it — and then the
    /// existing `uns` section is left byte-identical rather than rewritten.
    pub uns_key: Option<String>,
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
            uns_key: None,
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
    pub obs_key_column: String,
    pub obs_columns_added: Vec<String>,
    pub obsm_keys_added: Vec<String>,
    /// Whether the obs predicate index had to be dropped because this import
    /// overwrote a column it covered. `true` means query pushdown on those
    /// columns is gone until the index is rebuilt.
    pub obs_index_dropped: bool,
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
    /// Obs columns whose values are unique across all rows — the actionable list.
    pub unique_columns: Vec<String>,
    /// Unique 2-column composites, searched only when no single column is unique.
    pub unique_pairs: Vec<(String, String)>,
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
            s.push_str(&format!(
                "Obs columns that ARE unique: {:?}. ",
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
    if columns.len() == 1 {
        return string_column(batch, &columns[0]);
    }

    let mut parts: Vec<Vec<String>> = Vec::with_capacity(columns.len());
    for name in columns {
        if batch.column_by_name(name).is_none() {
            let schema = batch.schema();
            let present: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
            return Err(OpsError::KeyColumnUnresolved {
                axis: "obs",
                detail: format!(
                    "composite key column '{name}' not found; columns present are {present:?}"
                ),
            });
        }
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

/// Resolve the target-side keys for `join_key`, returning `(spec, keys)`.
fn resolve_target_keys(obs: &RecordBatch, join_key: &ObsJoinKey) -> Result<(String, Vec<String>)> {
    match join_key {
        ObsJoinKey::Auto => {
            let col = resolve_key_column("obs", obs, None)?;
            let keys = string_column(obs, &col)?;
            Ok((col, keys))
        }
        ObsJoinKey::Column(c) => {
            let col = resolve_key_column("obs", obs, Some(c))?;
            let keys = string_column(obs, &col)?;
            Ok((col, keys))
        }
        ObsJoinKey::Composite { columns } => {
            let keys = build_composite_key(obs, columns)?;
            Ok((columns.join(","), keys))
        }
    }
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

    // Per-column cardinality, cheapest signal first.
    let mut cardinalities: Vec<(String, usize)> = Vec::new();
    for (i, f) in schema.fields().iter().enumerate() {
        if let Some(c) = distinct_count(&[obs.column(i).clone()]) {
            cardinalities.push((f.name().clone(), c));
        }
    }

    let mut unique_columns: Vec<String> = cardinalities
        .iter()
        .filter(|(_, c)| *c == n_obs && n_obs > 0)
        .map(|(n, _)| n.clone())
        .collect();
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

    let suggestion = unique_columns
        .first()
        .cloned()
        .or_else(|| unique_pairs.first().map(|(a, b)| format!("{a},{b}")));

    KeyDiagnosis {
        n_obs,
        resolved_key,
        resolved_cardinality,
        unique_columns,
        unique_pairs,
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

fn build_obs_row_join(
    obs: &RecordBatch,
    key_spec: &str,
    target_keys: &[String],
    source_keys: &[String],
    opts: &AttachObsOptions,
) -> Result<ObsRowJoin> {
    let dups = first_duplicates(target_keys);
    if !dups.is_empty() {
        let diag = diagnose_from_obs(obs, Some((key_spec, target_keys)));
        return Err(OpsError::DuplicateJoinKey {
            axis: "obs",
            detail: format!(
                "target obs key '{key_spec}' contains duplicates: {dups:?}. {}",
                diag.describe()
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
        let diag = diagnose_from_obs(obs, Some((key_spec, target_keys)));
        return Err(OpsError::AxisMismatch {
            axis: "obs",
            detail: format!(
                "no target row key matched any source row key on '{key_spec}'. Target \
                 examples: {:?}; source examples: {:?}. Check for a sample-name prefix \
                 or a '-1' suffix difference. {}",
                examples(target_keys),
                examples(source_keys),
                diag.describe()
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
    if (n_matched as usize) * 2 < target_keys.len() {
        log::warn!(
            "external obs join matched only {n_matched} of {} target rows on '{key_spec}'; \
             target examples {:?}, source examples {:?}",
            target_keys.len(),
            examples(target_keys),
            examples(source_keys),
        );
    }

    Ok(ObsRowJoin {
        source_of_target,
        n_matched,
        n_target_absent,
        n_source_absent,
    })
}

/// Scatter a source-row-indexed array onto the target row axis, `null` for
/// unmatched rows. `null`, not a zero: a score of `0.0` on a cell the tool never
/// saw is a claim it never made.
fn scatter(col: &ArrayRef, join: &ObsRowJoin) -> Result<ArrayRef> {
    let take_idx: UInt32Array = join.source_of_target.iter().copied().collect();
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

fn build_new_obs(
    obs: &RecordBatch,
    data: &ExternalObsData,
    opts: &AttachObsOptions,
    join: &ObsRowJoin,
) -> Result<RecordBatch> {
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
        let arr: StringArray = join
            .source_of_target
            .iter()
            .map(|o| Some(if o.is_some() { "present" } else { "absent" }))
            .collect();
        fields.push(Field::new(name, DataType::Utf8, false));
        columns.push(Arc::new(arr) as ArrayRef);
    }

    for (i, f) in data.row_annotations.schema().fields().iter().enumerate() {
        let scattered = scatter(data.row_annotations.column(i), join)?;
        // Unmatched rows become null, so the field must admit nulls regardless
        // of how the source declared it.
        fields.push(Field::new(f.name(), f.data_type().clone(), true));
        columns.push(scattered);
    }

    let schema = Arc::new(Schema::new(fields).with_metadata(obs.schema().metadata().clone()));
    RecordBatch::try_new(schema, columns)
        .map_err(|e| OpsError::InvalidInput(format!("failed to build new obs: {e}")))
}

fn build_new_uns(mut uns: Value, data: &ExternalObsData, opts: &AttachObsOptions) -> Value {
    let (Some(key), Some(note)) = (opts.uns_key.as_ref(), data.uns.as_ref()) else {
        return uns;
    };
    if !uns.is_object() {
        uns = serde_json::json!({});
    }
    if let Some(map) = uns.as_object_mut() {
        map.insert(key.clone(), note.clone());
    }
    uns
}

fn build_obsm(data: &ExternalObsData, join: &ObsRowJoin) -> Result<Vec<(String, RecordBatch)>> {
    let mut out = Vec::new();
    for (name, batch) in &data.row_embeddings {
        let mut fields = Vec::new();
        let mut columns = Vec::new();
        for (i, f) in batch.schema().fields().iter().enumerate() {
            fields.push(Field::new(f.name(), f.data_type().clone(), true));
            columns.push(scatter(batch.column(i), join)?);
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

fn validate_shape(data: &ExternalObsData) -> Result<()> {
    let n_rows = data.row_keys.len();
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
    let Some(bytes) = reader.read_obs_predicate_index_bytes()? else {
        return Ok(false);
    };
    let index = scx_engine::PredicateIndex::read_from(&mut std::io::Cursor::new(bytes))?;
    Ok(index.columns.iter().any(|c| {
        let name = match c {
            scx_engine::index::IndexedColumn::Categorical(cat) => &cat.column_name,
            scx_engine::index::IndexedColumn::Numeric(num) => &num.column_name,
        };
        planned.contains(name)
    }))
}

fn check_collisions(
    reader: &ScxReader,
    obs: &RecordBatch,
    uns: &Value,
    data: &ExternalObsData,
    opts: &AttachObsOptions,
    obs_new: &[String],
) -> Result<()> {
    let obs_schema = obs.schema();
    for name in obs_new {
        if obs_schema.field_with_name(name).is_ok() {
            return Err(OpsError::InvalidInput(format!(
                "obs column '{name}' already exists; pass overwrite=true to replace it. \
                 Note that overwrite REPLACES rather than merges — importing several \
                 per-batch tables one after another would keep only the last."
            )));
        }
    }
    if let Some(key) = &opts.uns_key {
        if data.uns.is_some() && uns.get(key).is_some() {
            return Err(OpsError::InvalidInput(format!(
                "uns key '{key}' already exists; pass overwrite=true to replace it"
            )));
        }
    }
    if !data.row_embeddings.is_empty() {
        let existing = reader.read_all_obsm().unwrap_or_default();
        for (name, _) in &data.row_embeddings {
            if existing.iter().any(|(k, _)| k == name) {
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
        "uns_key": opts.uns_key,
        "overwrite": opts.overwrite,
    });
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
/// Every validation runs before the first byte is written, so a rejected import
/// leaves the file byte-identical. `X`, layers, the CSC sidecar, `.raw`,
/// deletion vectors, detection bitmaps and `var` are never read or rewritten;
/// the whole import is undoable with `scx rollback`.
pub fn attach_external_obs(
    path: &Path,
    data: &ExternalObsData,
    opts: &AttachObsOptions,
) -> Result<AttachObsSummary> {
    validate_shape(data)?;

    let planned_obs = planned_obs_columns(data, opts);
    if planned_obs.is_empty() && data.row_embeddings.is_empty() && data.uns.is_none() {
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
    let reader = ScxReader::open(path)?;
    let obs = reader.read_obs()?;
    let uns = reader.read_uns().unwrap_or_else(|_| serde_json::json!({}));

    let n_obs = prep.old_n_obs;
    if obs.num_rows() as u64 != n_obs {
        return Err(OpsError::ShapeMismatch {
            detail: format!(
                "obs has {} rows but the header declares n_obs = {n_obs}",
                obs.num_rows()
            ),
        });
    }

    // --- Resolve the join key and build the join ---------------------------
    let (key_spec, target_keys) = resolve_target_keys(&obs, &opts.join_key)?;
    let row_join = build_obs_row_join(&obs, &key_spec, &target_keys, &data.row_keys, opts)?;

    // --- Collision checks ---------------------------------------------------
    if !opts.overwrite {
        check_collisions(&reader, &obs, &uns, data, opts, &planned_obs)?;
    }

    // --- Would this invalidate the obs predicate index? ---------------------
    let drop_obs_index = obs_index_would_go_stale(&reader, &planned_obs)?;

    // --- Build the new sections in memory ----------------------------------
    let rewrote_uns = opts.uns_key.is_some() && data.uns.is_some();
    let new_obs = build_new_obs(&obs, data, opts, &row_join)?;
    let new_obs = unify_dict_columns(&new_obs)?;
    let new_uns = build_new_uns(uns, data, opts);
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
    };

    if opts.dry_run {
        // Everything above is validation and the join itself, so the caller
        // gets a real `n_matched` while the file stays untouched. Unlike the
        // layer op there is no nnz to predict, so the preview is exact by
        // construction.
        return Ok(summary);
    }

    let mut prov_ops = read_provenance_ops(&mut lock, &prep.old_catalog)?;

    // --- Emit sections at EOF through an adopted writer --------------------
    let write_offset = lock.seek(SeekFrom::End(0))?;
    let cloned = lock.file().try_clone()?;
    let mut writer =
        ScxWriter::adopt_in_place(cloned, prep.header.clone(), write_offset, Vec::new())?;

    if rewrote_uns {
        writer.write_uns(&new_uns)?;
    }

    let n = new_obs.num_rows();
    let mut cursor = 0usize;
    let mut idx = 0u32;
    while cursor < n {
        let take = std::cmp::min(shard_target_rows, n - cursor);
        writer.write_obs_shard(
            idx,
            cursor as u64,
            take as u64,
            n_obs,
            &new_obs.slice(cursor, take),
        )?;
        idx += 1;
        cursor += take;
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
