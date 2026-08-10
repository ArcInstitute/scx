//! Attach an externally-computed layer to an existing SCX file, in place.
//!
//! The motivating case is CellBender interop — run `remove-background`
//! externally, then land its corrected counts as `layers["cellbender"]` — but
//! nothing here knows about CellBender. The op takes a matrix plus per-axis
//! annotations keyed by **string**, joins them onto the target's own axes, and
//! writes a layer, obs/var columns, optional `obsm`, `uns`, and an appended
//! provenance entry.
//!
//! # Why in place
//!
//! A layer is a *new* section, so appending sections at EOF and repointing the
//! catalog ([`prepare_in_place`] / [`commit_in_place`]) preserves `X`, the CSC
//! sidecar, `.raw`, deletion vectors, detection bitmaps and predicate indexes
//! **by construction** — there is no per-section copy policy to get wrong — and
//! the whole thing is undoable with `scx rollback`. `X` is never read or
//! rewritten, so `data_generation` and `csc_build_generation` stay put and a
//! pre-existing CSC sidecar remains valid.
//!
//! # Why the join is by key and never by position
//!
//! The external tool's row order is its own business. CellBender's filtered
//! output, for instance, is in descending-UMI order, not the input's row order,
//! so a positional assumption would silently scramble cells — every value lands
//! on the wrong barcode, and nothing downstream would notice. Rows the target
//! has but the source lacks become all-zero layer rows with a `"absent"` status
//! marker and `null` (not `0.0`) annotations.
//!
//! # Two writer traps this op works around
//!
//! * **The duplicate-section guard does not fire under an adopted writer.**
//!   `ScxWriter::write_section_bytes` checks only the writer's own entry list,
//!   and [`ScxWriter::adopt_in_place`] starts with an empty one. Re-importing a
//!   layer would therefore leave two `{layer}_shard_0` entries in the merged
//!   catalog, and `assemble_shards` concatenates every prefix match into a
//!   `2·n_obs`-row layer while `layer_names()` still reports one. Old entries
//!   are dropped explicitly, by `_shard_` stem.
//! * **`write_layer_csr_shard` bypasses the v4 framing guard.** `adopt_in_place`
//!   sets `framing: None`, and `guard_no_legacy_shard_in_v4` is wired into
//!   `write_preencoded_shard` but not `write_shard_inner`. Writing a layer the
//!   obvious way into a v4 file silently produces an unframed shard-v1 section,
//!   so this op encodes with [`encode_one_shard_with_value_encoding`] and writes
//!   through [`ScxWriter::write_preencoded_shard`] instead.
//!
//! # Memory
//!
//! Target-side obs is handled exactly as in the obs-only sibling — see
//! [`crate::external_obs`] for the full account. On a sharded target the join
//! reads only the obs key column and the rewrite runs one shard at a time; a
//! legacy single-section `ObsMetadata` target is assembled whole, warns, and
//! reports [`AttachLayerSummary::obs_streamed`] as `false`. `var` is read whole
//! regardless: it is the gene axis, and the column join retries across every
//! candidate key column.
//!
//! The **source** side is resident in full. `read_cellbender_h5` documents that
//! for its own input and the reason is the same here: the barcode-keyed join
//! reorders rows arbitrarily, so `ExternalLayerData`'s CSR triple has to be in
//! memory. Bounded in practice by CellBender's `total_droplets_included`.

use std::collections::HashMap;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use arrow::array::{Array, ArrayRef, Float32Array, RecordBatch, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use serde_json::Value;
use std::sync::Arc;

use scx_codec::value_encoding::detect_value_encoding;
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::catalog::{FullCatalog, FullCatalogEntry};
use scx_format_io::checksum::blake3_hash;
use scx_format_io::encoder::encode_one_shard_with_value_encoding;
use scx_format_io::provenance::{Provenance, ProvenanceEntry};
use scx_format_io::reader::ScxReader;
use scx_format_io::section::{write_alignment_padding, SectionType};
use scx_format_io::writer::ScxWriter;
use scx_format_io::FramingConfig;

use crate::append::unify_dict_columns;
use crate::error::{OpsError, Result};
use crate::external_obs::{
    warn_unstreamable_obs, write_obs_shards_appending, write_obs_shards_from_whole, ObsRewrite,
};
use crate::in_place::{
    commit_in_place, entry_matches_key, layer_entry_matches, prepare_in_place, read_provenance_ops,
};

/// Obs/var column names that resolve a join key when the caller does not name
/// one, in **preference order** — the first present column wins.
///
/// Ordered preference rather than "exactly one must be present": a
/// 10x-converted target carries both `id` and `name`, which is the standard
/// shape, not an ambiguity. A wrong pick is caught downstream by the
/// zero-overlap error (and, on the column axis, retried against the next
/// candidate), and the resolved column is reported in the summary.
///
/// `barcode` earns its place because `scx convert --from 10x` writes obs as a
/// bare `barcode` column with no pandas index metadata — exactly the files most
/// likely to be an import target.
pub(crate) const OBS_KEY_FALLBACKS: &[&str] = &[
    "barcode",
    "barcodes",
    "cell_id",
    "cell_barcode",
    "_index",
    "__index_level_0__",
    "index",
];

const VAR_KEY_FALLBACKS: &[&str] = &[
    "gene_id",
    "gene_ids",
    "id",
    "feature_id",
    "gene_name",
    "gene_symbol",
    "name",
    "_index",
    "__index_level_0__",
    "index",
];

/// Physical arrow field names that carry a frame's index. Kept in step with
/// the literal probe in [`scx_format_io::resolve_index_columns`].
const INDEX_FIELD_NAMES: &[&str] = &["__index_level_0__", "_index"];

/// The user-facing spelling of an axis's index, and the alias a caller may pass
/// to name it.
///
/// The physical field is `__index_level_0__` (pyarrow's serialization name for
/// a DataFrame index) or anndata's `_index` sentinel. Neither is addressable by
/// a user: `Experiment.read_obs()` hands that field back as the frame's
/// *unnamed index*, so `obs["__index_level_0__"]` is a `KeyError`. Diagnostics
/// and join messages print this spelling instead, and [`resolve_key_column`]
/// accepts it back.
///
/// Those two halves must ship together. Printing a name the resolver rejects
/// is the same dead end as suggesting a float key: the tooling names a key it
/// then refuses.
///
/// `axis` is the same two-valued string [`resolve_key_column`] takes — `"obs"`
/// or `"var"`. Asserted rather than left to an `else` branch because
/// [`candidate_key_columns`] defaults the *other* way, so a third spelling
/// would silently disagree between the two.
pub fn axis_index_alias(axis: &str) -> &'static str {
    debug_assert!(
        axis == "obs" || axis == "var",
        "unknown axis {axis:?}: expected \"obs\" or \"var\""
    );
    if axis == "var" {
        "var_names"
    } else {
        "obs_names"
    }
}

/// Render a resolved key column — or a comma-joined composite spec — for
/// display, translating the physical index field to [`axis_index_alias`].
///
/// Composite specs are translated component-wise, so
/// `sample_id,__index_level_0__` reads `sample_id,obs_names`.
pub fn display_key_name(axis: &str, spec: &str) -> String {
    let alias = axis_index_alias(axis);
    spec.split(',')
        .map(|c| {
            if INDEX_FIELD_NAMES.contains(&c) {
                alias
            } else {
                c
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// [`display_key_name`] over a list of column names, for the "columns present
/// are …" lists that error messages print.
fn display_columns(axis: &str, cols: &[&str]) -> Vec<String> {
    cols.iter().map(|c| display_key_name(axis, c)).collect()
}

/// Resolve an axis-index alias (`obs_names` / `var_names`, or the bare `index`
/// spelling) to the physical field that carries the index.
///
/// Consulted only when no field literally bears the requested name, so a real
/// column called `index` — or, absurdly, `obs_names` — still wins over the
/// alias. Returns `None` when the frame has no index field, letting the caller
/// fall through to its own not-found error.
fn index_field_for_alias(axis: &str, schema: &Schema, requested: &str) -> Option<String> {
    if requested != axis_index_alias(axis) && requested != "index" {
        return None;
    }
    scx_format_io::resolve_index_columns(schema)
        .into_iter()
        .next()
}

/// Map a requested key name to the physical field it names.
///
/// The literal name always wins; the axis-index alias is consulted only when no
/// field bears the requested name. Returns the input unchanged when neither
/// resolves, so the caller's own not-found error still quotes what the user
/// typed rather than a rewritten form of it.
pub fn resolve_key_alias(axis: &str, schema: &Schema, requested: &str) -> String {
    if schema.field_with_name(requested).is_ok() {
        return requested.to_string();
    }
    index_field_for_alias(axis, schema, requested).unwrap_or_else(|| requested.to_string())
}

/// Layer names that would collide with reserved section naming.
const RESERVED_LAYER_NAMES: &[&str] = &["X", "raw"];

/// How many example keys to show when a join fails, so a mismatch is
/// diagnosable in one shot rather than one round trip per guess.
pub(crate) const KEY_EXAMPLES: usize = 3;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// An externally-computed matrix plus axis annotations, keyed by string.
///
/// Produced by a reader (e.g. `scx_convert::read_cellbender_h5`) and consumed
/// by [`attach_external_layer`].
pub struct ExternalLayerData {
    /// One key per matrix row (e.g. a cell barcode). `len == matrix rows`.
    pub row_keys: Vec<String>,
    /// One key per matrix column (e.g. a gene id). `len == matrix cols`.
    pub col_keys: Vec<String>,
    /// CSR of `[row_keys.len() x col_keys.len()]`, 0-based indptr. Need not be
    /// canonical; each emitted shard is canonicalized.
    pub indptr: Vec<u64>,
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
    /// Per-source-row annotations appended to obs. `num_rows()` must equal
    /// `row_keys.len()`. Columns must be nullable if they are only meaningful
    /// for a subset of rows — unmatched target rows get `null`.
    pub row_annotations: Option<RecordBatch>,
    /// Per-source-row dense matrices written to `obsm`.
    pub row_embeddings: Vec<(String, RecordBatch)>,
    /// Per-source-column annotations appended to var. `num_rows()` must equal
    /// `col_keys.len()`.
    pub col_annotations: Option<RecordBatch>,
    /// Merged into `uns` under [`AttachLayerOptions::uns_key`].
    pub uns: Option<Value>,
    /// BLAKE3 of the source file, recorded in the provenance entry.
    pub source_checksum: Option<[u8; 32]>,
    /// Display name of the source (usually a path basename).
    pub source_name: Option<String>,
}

/// What to do with target rows that have no matching source row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingRowPolicy {
    /// Emit an empty layer row and mark it `"absent"`. The usual choice: the
    /// external tool legitimately analysed a subset.
    #[default]
    ZeroFill,
    /// Any unmatched target row is an error.
    Error,
}

/// What to do with source rows absent from the target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExtraRowPolicy {
    /// Warn and skip. The default because running a tool on raw droplets and
    /// importing into a filtered-cells file is a normal workflow that leaves
    /// hundreds of thousands of unmatched source rows.
    #[default]
    WarnSkip,
    /// Any unmatched source row is an error.
    Error,
}

/// How much column-axis divergence to tolerate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColumnAxisPolicy {
    /// Source and target column keys must be elementwise equal.
    #[default]
    RequireIdentical,
    /// Same set, any order. Requires an index remap, so the emitted shards are
    /// re-canonicalized.
    AllowReorder,
    /// Source keys may be a subset of the target's.
    AllowSubset,
}

/// How the source column axis actually related to the target's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnAxisMatch {
    Identical,
    Reordered,
    Subset,
}

/// Where the emitted layer's shard row ranges came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardRangeSource {
    /// Copied from the existing `X` CSR shards — the normal case, and what
    /// keeps the layer aligned with `X` for shard-wise readers.
    XShards,
    /// `X` shards lacked usable stats, so `[0, n_obs)` was chunked by the
    /// header's `shard_target_rows`.
    ShardTargetRowsFallback,
}

pub struct AttachLayerOptions {
    pub layer_name: String,
    /// Obs column holding the row key. `None` auto-resolves (see
    /// [`OBS_KEY_FALLBACKS`]); it never falls back to positional.
    pub obs_key_column: Option<String>,
    pub var_key_column: Option<String>,
    pub missing_row_policy: MissingRowPolicy,
    pub extra_row_policy: ExtraRowPolicy,
    pub column_axis_policy: ColumnAxisPolicy,
    /// Obs column recording `"present"` / `"absent"` per row. `None` omits it.
    pub status_column: Option<String>,
    /// Obs column holding each row's emitted layer sum. `None` omits it.
    pub row_sum_column: Option<String>,
    /// `uns` key for [`ExternalLayerData::uns`]. `None` omits it.
    pub uns_key: Option<String>,
    /// When false (default), any pre-existing layer / obs column / var column /
    /// obsm key / uns key this op would replace is an error.
    pub overwrite: bool,
    /// `None` detects the narrowest encoding that fits **all** values, once, so
    /// every shard of the layer shares one encoding.
    pub value_encoding: Option<ValueEncoding>,
    /// `None` lets the encoder pick per shard.
    pub codec: Option<CodecId>,
    pub modality_id: u8,
    pub provenance_action: String,
    /// Merged into the provenance `params_json` object.
    pub provenance_params: Value,
    /// Run every validation and resolve the join, then return the summary
    /// **without writing anything**. The point is to let a caller see
    /// `n_matched` before mutating a large file; a dry run that skipped the
    /// join would report nothing worth seeing.
    pub dry_run: bool,
}

impl Default for AttachLayerOptions {
    fn default() -> Self {
        Self {
            layer_name: "external".to_string(),
            obs_key_column: None,
            var_key_column: None,
            missing_row_policy: MissingRowPolicy::default(),
            extra_row_policy: ExtraRowPolicy::default(),
            column_axis_policy: ColumnAxisPolicy::default(),
            status_column: None,
            row_sum_column: None,
            uns_key: None,
            overwrite: false,
            value_encoding: None,
            codec: None,
            modality_id: 0,
            provenance_action: "attach_external_layer".to_string(),
            provenance_params: Value::Null,
            dry_run: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AttachLayerSummary {
    pub n_obs: u64,
    pub n_matched: u64,
    pub n_target_rows_absent: u64,
    pub n_source_rows_absent: u64,
    /// Of the skipped source rows, how many actually carried counts. Only this
    /// number means real data was discarded.
    pub n_source_rows_absent_nonzero: u64,
    pub obs_key_column: String,
    pub var_key_column: String,
    pub column_axis_match: ColumnAxisMatch,
    pub layer_nnz: u64,
    pub value_encoding: ValueEncoding,
    pub shard_ranges_from: ShardRangeSource,
    pub obs_columns_added: Vec<String>,
    pub var_columns_added: Vec<String>,
    pub obsm_keys_added: Vec<String>,
    /// `true` when the obs axis was rewritten shard by shard (peak memory one
    /// shard); `false` when it had to be assembled whole, which a legacy
    /// single-section `ObsMetadata` forces. See
    /// [`crate::external_obs::ObsRewrite`].
    pub obs_streamed: bool,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Attach `data` to the SCX file at `path` as a new layer plus annotations.
///
/// Every validation runs before the first byte is written, so a rejected import
/// leaves the file byte-identical.
pub fn attach_external_layer(
    path: &Path,
    data: &ExternalLayerData,
    opts: &AttachLayerOptions,
) -> Result<AttachLayerSummary> {
    attach_external_layer_inner(path, None, data, opts)
}

/// [`attach_external_layer`] against a caller-supplied reader. Test-only, for
/// the same reason as [`crate::external_obs::attach_external_obs_with_reader`]
/// — see that function for why production must open the reader after taking the
/// lock.
#[cfg(test)]
pub(crate) fn attach_external_layer_with_reader(
    path: &Path,
    reader: &ScxReader,
    data: &ExternalLayerData,
    opts: &AttachLayerOptions,
) -> Result<AttachLayerSummary> {
    attach_external_layer_inner(path, Some(reader), data, opts)
}

fn attach_external_layer_inner(
    path: &Path,
    injected_reader: Option<&ScxReader>,
    data: &ExternalLayerData,
    opts: &AttachLayerOptions,
) -> Result<AttachLayerSummary> {
    validate_layer_name(&opts.layer_name)?;
    validate_shape(data)?;

    let (mut lock, mut prep) = prepare_in_place(path, opts.modality_id)?;

    if opts.modality_id != 0 || prep.header.n_modalities > 0 {
        return Err(OpsError::MultimodalUnsupported {
            op: "attach_external_layer",
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
    // `var` is a gene-count axis (~10⁴ rows), so it is read whole: there is no
    // atlas-scale cost here, and the column join needs to retry across every
    // candidate key column.
    let var = reader.read_var()?;
    let uns = reader.read_uns().unwrap_or_else(|_| serde_json::json!({}));

    let obs_rewrite = if reader.obs_metadata_shard_count() > 0 {
        ObsRewrite::Streamed
    } else {
        ObsRewrite::Materialized
    };

    let n_obs = prep.old_n_obs;
    let n_vars = prep.target_n_vars;
    if var.num_rows() as u64 != n_vars {
        return Err(OpsError::ShapeMismatch {
            detail: format!(
                "var has {} rows but the modality declares n_vars = {n_vars}",
                var.num_rows()
            ),
        });
    }

    // --- Resolve join keys and build the maps ------------------------------
    //
    // The obs side reads the key column alone: `read_obs_schema_physical` is an
    // Arrow IPC footer read, and `read_obs_keys` projects + compacts each obs
    // shard rather than assembling the table. `read_obs_keys` also runs the
    // contiguous-cover validation `read_obs()` used to provide, so the
    // n_obs-vs-header check below is still a pre-write validation.
    let obs_schema = reader.read_obs_schema_physical()?;
    let obs_key_column = resolve_key_column("obs", &obs_schema, opts.obs_key_column.as_deref())?;
    let key_batch = reader.read_obs_keys(std::slice::from_ref(&obs_key_column))?;
    if key_batch.num_rows() as u64 != n_obs {
        return Err(OpsError::ShapeMismatch {
            detail: format!(
                "obs has {} rows but the header declares n_obs = {n_obs}",
                key_batch.num_rows()
            ),
        });
    }
    let target_row_keys = string_column(&key_batch, &obs_key_column)?;
    drop(key_batch);
    let row_join = build_row_join(&target_row_keys, data, opts)?;
    drop(target_row_keys);

    // Column axis: try the resolved key, then — only when the caller did not
    // name one — the remaining candidates, so an Ensembl-id target still joins
    // against a symbol-keyed source (and vice versa).
    let mut candidates = vec![resolve_key_column(
        "var",
        &var.schema(),
        opts.var_key_column.as_deref(),
    )?];
    if opts.var_key_column.is_none() {
        for c in candidate_key_columns("var", &var) {
            if !candidates.contains(&c) {
                candidates.push(c);
            }
        }
    }
    let mut attempt = None;
    let mut first_err = None;
    for key in &candidates {
        let target_col_keys = string_column(&var, key)?;
        match build_column_map(&target_col_keys, &data.col_keys, opts.column_axis_policy) {
            Ok(v) => {
                attempt = Some((key.clone(), v));
                break;
            }
            // Report the *first* candidate's failure if every one fails: that
            // is the caller's preferred key, and the reason it did not work is
            // more useful than why the last fallback did not.
            Err(e) => first_err = first_err.or(Some(e)),
        }
    }
    let Some((var_key_column, (col_map, column_axis_match))) = attempt else {
        return Err(first_err.expect("a non-empty candidate list either matches or errors"));
    };

    // --- Collision checks ---------------------------------------------------
    let obs_new_columns = planned_obs_columns(data, opts);
    let var_new_columns = planned_var_columns(data);
    // A pure column *add* leaves the obs predicate index valid — it keys on
    // column name and the indexed columns are untouched. An overwrite of an
    // indexed column does not: the index would describe values that no longer
    // exist and pushdown would silently return wrong rows.
    let drop_obs_index = crate::external_obs::obs_index_would_go_stale(reader, &obs_new_columns)?;
    if !opts.overwrite {
        check_collisions(
            reader,
            &prep.old_catalog,
            &obs_schema,
            &var,
            &uns,
            data,
            opts,
            &obs_new_columns,
            &var_new_columns,
        )?;
    }

    // --- Layer shard row ranges --------------------------------------------
    let (shard_ranges, shard_ranges_from) = derive_shard_ranges(&prep.old_catalog, &prep, n_obs);

    // --- Value encoding, detected once over the whole layer -----------------
    let value_encoding = match opts.value_encoding {
        Some(enc) => enc,
        None => {
            validate_values_finite_nonnegative(&data.values)?;
            detect_value_encoding(&data.values)
        }
    };

    // --- Build the new var / uns in memory ---------------------------------
    // The obs axis is built inside the write loop below, one shard at a time.
    let new_var = build_new_var(&var, data, &col_map, column_axis_match)?;
    let new_uns = build_new_uns(uns, data, opts);
    let obsm_batches = build_obsm(data, &row_join, n_obs as usize)?;

    // --- Capture catalog invariants before consuming old_catalog -----------
    let old_catalog_offset = prep.old_catalog_offset;
    let data_gen = prep.old_catalog.data_generation;
    let csc_gen = prep.old_catalog.csc_build_generation;
    let manifest = prep.header.manifest_sequence;
    let mt_off = prep.header.modality_table_offset;
    let mt_len = prep.header.modality_table_length;
    let shard_target_rows = (prep.header.shard_target_rows as usize).max(1);
    let index_dtype = prep.header_index_dtype;
    let modality_type = prep.modality_type;
    let framing = framing_for(&prep);

    let mut summary = AttachLayerSummary {
        n_obs,
        n_matched: row_join.n_matched,
        n_target_rows_absent: row_join.n_target_absent,
        n_source_rows_absent: row_join.n_source_absent,
        n_source_rows_absent_nonzero: row_join.n_source_absent_nonzero,
        obs_key_column,
        var_key_column,
        column_axis_match,
        // Filled in by the write loop.
        layer_nnz: 0,
        value_encoding,
        shard_ranges_from,
        obs_columns_added: obs_new_columns,
        var_columns_added: var_new_columns,
        obsm_keys_added: Vec::new(),
        obs_streamed: obs_rewrite == ObsRewrite::Streamed,
    };

    if opts.dry_run {
        // Everything above is validation and the join itself, so the caller
        // gets a real `n_matched` while the file stays untouched.
        //
        // nnz goes through the *same* gather+canonicalize the write path uses,
        // rather than summing raw source row lengths: otherwise a dry run
        // would overstate nnz exactly when the source has explicit zeros or
        // duplicate coordinates, and the preview would disagree with the
        // import it is meant to predict.
        summary.obsm_keys_added = obsm_batches.iter().map(|(k, _)| k.clone()).collect();
        for (row_start, row_end) in &shard_ranges {
            let (mut i, mut j, mut v) =
                gather_shard(data, &row_join, &col_map, *row_start, *row_end);
            scx_sparse::canonicalize_csr(&mut i, &mut j, &mut v);
            summary.layer_nnz += *i.last().unwrap_or(&0);
        }
        return Ok(summary);
    }

    // Built before the writer exists so a failure still leaves the file
    // untouched — and after the dry-run return so a preview does not pay for
    // it. The streaming path builds inside the write loop by construction.
    let materialized_obs = match obs_rewrite {
        ObsRewrite::Streamed => None,
        ObsRewrite::Materialized => {
            warn_unstreamable_obs("cellbender/layer import", n_obs);
            let obs = reader.read_obs()?;
            // The n_obs check lives in `write_obs_shards_from_whole`, shared
            // with the obs op so the two cannot diverge on whether it runs.
            let built = unify_dict_columns(&build_new_obs(&obs, data, opts, &row_join, 0)?)?;
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

    writer.write_uns(&new_uns)?;
    writer.write_var(&new_var)?;

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
        summary.obsm_keys_added.push(name.clone());
    }

    // Layer shards last, so the bulk bytes land contiguously at EOF.
    for (shard_idx, (row_start, row_end)) in shard_ranges.iter().enumerate() {
        let (mut s_indptr, mut s_indices, mut s_values) =
            gather_shard(data, &row_join, &col_map, *row_start, *row_end);

        // Shared canonicalizer: sort, dedup-summing duplicate coordinates,
        // then drop explicit zeros — in that order. A hand-rolled version here
        // got the ordering wrong and accumulated a duplicate onto the previous
        // column when the first occurrence was an explicit zero. Mandatory
        // after a gene-axis remap (indices come out unsorted), and `scx
        // validate --deep` enforces canonical CSR on layer shards regardless.
        scx_sparse::canonicalize_csr(&mut s_indptr, &mut s_indices, &mut s_values);

        // Counted after canonicalization, so the reported nnz matches what was
        // actually written rather than the pre-dedup input.
        summary.layer_nnz += *s_indptr.last().unwrap_or(&0);

        let section = encode_one_shard_with_value_encoding(
            &s_indptr,
            &s_indices,
            &s_values,
            opts.codec,
            index_dtype,
            n_vars as u32,
            *row_start,
            SectionType::LayerCsrShard,
            modality_type,
            format!("{}_shard_{}", opts.layer_name, shard_idx),
            framing,
            Some(value_encoding),
        )?;
        writer.write_preencoded_shard(section)?;
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
        .filter(|e| !should_drop_old_entry(e, opts, drop_obs_index, &obsm_batches))
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

    // The other half of `drop_obs_index` — see the matching comment in
    // `external_obs.rs`. The catalog's per-shard `ColumnStat`s are what Level-1
    // pushdown actually prunes on (the numeric `MinMax` arm never reads the index
    // section), so an overwrite has to clear them or shards holding matching rows
    // are silently excluded. Scoped to the rewritten columns, and unconditional
    // rather than gated on `drop_obs_index`, for the reasons given there.
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
        // X is untouched, so the CSC sidecar stays fresh. Bumping either of
        // these would silently invalidate it and drop every `prefer_format=csc`
        // / `gpu_csc_v3` route to a fallback.
        data_generation: data_gen,
        csc_build_generation: csc_gen,
    };

    commit_in_place(&mut lock, &mut prep.header, &new_catalog, mt_off, mt_len)?;

    Ok(summary)
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_layer_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(OpsError::InvalidInput(
            "layer name must not be empty".into(),
        ));
    }
    if RESERVED_LAYER_NAMES.contains(&name) {
        return Err(OpsError::InvalidInput(format!(
            "layer name '{name}' is reserved; a `{name}_shard_N` LayerCsrShard \
             would collide by name with the matrix sections"
        )));
    }
    if name.contains('/') {
        return Err(OpsError::InvalidInput(format!(
            "layer name '{name}' must not contain '/'"
        )));
    }
    Ok(())
}

fn validate_shape(data: &ExternalLayerData) -> Result<()> {
    let n_rows = data.row_keys.len();
    if data.indptr.len() != n_rows + 1 {
        return Err(OpsError::ShapeMismatch {
            detail: format!(
                "indptr has {} entries but there are {n_rows} row keys (expected {})",
                data.indptr.len(),
                n_rows + 1
            ),
        });
    }
    if data.indices.len() != data.values.len() {
        return Err(OpsError::ShapeMismatch {
            detail: format!(
                "indices ({}) and values ({}) lengths differ",
                data.indices.len(),
                data.values.len()
            ),
        });
    }
    // `ExternalLayerData` is a public seam, so a malformed value must not reach
    // `gather_shard` — the remap path indexes `col_map` by column index and
    // would panic, and the identity path would emit an out-of-range index.
    let n_cols = data.col_keys.len();
    if let Some(bad) = data.indices.iter().find(|c| **c as usize >= n_cols) {
        return Err(OpsError::ShapeMismatch {
            detail: format!("column index {bad} is out of range for {n_cols} column keys"),
        });
    }
    let nnz = *data.indptr.last().unwrap_or(&0) as usize;
    if nnz != data.indices.len() {
        return Err(OpsError::ShapeMismatch {
            detail: format!(
                "indptr terminates at {nnz} but there are {} indices",
                data.indices.len()
            ),
        });
    }
    if let Some(ann) = &data.row_annotations {
        if ann.num_rows() != n_rows {
            return Err(OpsError::ShapeMismatch {
                detail: format!(
                    "row_annotations has {} rows but there are {n_rows} row keys",
                    ann.num_rows()
                ),
            });
        }
    }
    if let Some(ann) = &data.col_annotations {
        if ann.num_rows() != data.col_keys.len() {
            return Err(OpsError::ShapeMismatch {
                detail: format!(
                    "col_annotations has {} rows but there are {} column keys",
                    ann.num_rows(),
                    data.col_keys.len()
                ),
            });
        }
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

/// Reject values that cannot be a count matrix. A negative or non-finite entry
/// means a corrupt source, and would otherwise silently force a `Float32` layer
/// that poisons every downstream sum.
fn validate_values_finite_nonnegative(values: &[f32]) -> Result<()> {
    for (i, v) in values.iter().enumerate() {
        if !v.is_finite() || *v < 0.0 {
            return Err(OpsError::InvalidInput(format!(
                "external layer value at nnz index {i} is {v}; values must be \
                 finite and non-negative"
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Key resolution and joins
// ---------------------------------------------------------------------------

pub(crate) fn is_string_column(dt: &DataType) -> bool {
    match dt {
        DataType::Utf8 | DataType::LargeUtf8 => true,
        DataType::Dictionary(_, v) => matches!(**v, DataType::Utf8 | DataType::LargeUtf8),
        _ => false,
    }
}

/// Whether an **explicitly requested** key column can serve as a join key.
///
/// Wider than [`is_string_column`] because both sides of the join go through
/// [`string_column`], which casts to `Utf8` — so an integer key fuses
/// identically on both sides and joins exactly.
///
/// This matters for the case the feature exists for: on a merged
/// CELLxGENE-derived atlas the obs index is duplicated and `soma_joinid` — an
/// `Int64` — is the *only* unique column. `diagnose_obs_key` suggests it, and
/// before this a user following that suggestion hit "column 'soma_joinid' has
/// type Int64, which is not a string column". The diagnosis pointed at a key
/// the join then refused.
///
/// Floats stay rejected: `f64 → Utf8` formatting is not guaranteed to agree
/// between two independently-produced sides, so a float key could silently
/// half-match. Auto-resolution is also unchanged — guessing that a numeric
/// column is the identity is a different and worse risk than honouring an
/// explicit request.
pub(crate) fn is_joinable_key_column(dt: &DataType) -> bool {
    if is_string_column(dt) {
        return true;
    }
    match dt {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => true,
        DataType::Dictionary(_, v) => is_joinable_key_column(v),
        _ => false,
    }
}

/// Resolve the join key column, never falling back to positional.
///
/// Takes a schema rather than a batch because that is genuinely all it needs,
/// and the difference matters: on an atlas the caller can hand it
/// `ScxReader::read_obs_schema_physical()` — an Arrow IPC footer read at
/// constant cost — instead of assembling the obs table to look at its field
/// names. The physical schema resolves identically to the batch's: the only
/// difference is `LargeUtf8` / `Dictionary(_, LargeUtf8)` left wide, and both
/// [`is_string_column`] and [`is_joinable_key_column`] accept those.
pub(crate) fn resolve_key_column(
    axis: &'static str,
    schema: &Schema,
    requested: Option<&str>,
) -> Result<String> {
    let string_cols: Vec<&str> = schema
        .fields()
        .iter()
        .filter(|f| is_string_column(f.data_type()))
        .map(|f| f.name().as_str())
        .collect();

    if let Some(requested_name) = requested {
        // `obs_names` / `var_names` name the axis index, whose physical field is
        // `__index_level_0__` or `_index`.
        let resolved = resolve_key_alias(axis, schema, requested_name);
        let name = resolved.as_str();
        return match schema.field_with_name(name) {
            // An explicit request accepts any type that fuses identically on
            // both sides — see `is_joinable_key_column`.
            Ok(f) if is_joinable_key_column(f.data_type()) => Ok(name.to_string()),
            Ok(f) => Err(OpsError::KeyColumnUnresolved {
                axis,
                detail: format!(
                    "column '{}' has type {:?}, which cannot be a join key. \
                     Strings and integers work (both sides are fused as text); \
                     floats are refused because their text form is not \
                     guaranteed to agree across two independently written sides",
                    display_key_name(axis, name),
                    f.data_type()
                ),
            }),
            // The listed columns are display names, so an index field shows up
            // as `obs_names` — which is both the spelling to pass back and, on
            // its own, all the hint this needs.
            Err(_) => Err(OpsError::KeyColumnUnresolved {
                axis,
                detail: format!(
                    "column '{requested_name}' not found; string columns are {:?}",
                    display_columns(axis, &string_cols),
                ),
            }),
        };
    }

    // The pandas index column, when the source stamped one.
    for candidate in scx_format_io::pandas_index_columns(schema) {
        if string_cols.contains(&candidate.as_str()) {
            return Ok(candidate);
        }
    }

    let fallbacks = if axis == "obs" {
        OBS_KEY_FALLBACKS
    } else {
        VAR_KEY_FALLBACKS
    };
    let present: Vec<&str> = fallbacks
        .iter()
        .copied()
        .filter(|c| string_cols.contains(c))
        .collect();
    present.first().map(|c| c.to_string()).ok_or_else(|| {
        // The fallback list is printed without its three index spellings.
        // They are one concept the user knows as `obs_names`, and naming the
        // physical fields invites keying on something `read_obs()` does not
        // expose as a column. Reaching here means no index field is present
        // either, so there is no alias worth suggesting.
        let named: Vec<&str> = fallbacks
            .iter()
            .copied()
            .filter(|c| !INDEX_FIELD_NAMES.contains(c) && *c != "index")
            .collect();
        OpsError::KeyColumnUnresolved {
            axis,
            detail: format!(
                "no {axis} index and none of {named:?} present; string columns \
                     are {:?}. Pass an explicit key column.",
                display_columns(axis, &string_cols)
            ),
        }
    })
}

/// Every candidate key column for an axis, in preference order.
///
/// Used to retry the column join when the preferred key has no overlap: a
/// target converted from 10x carries both Ensembl `id` and symbol `name`, and
/// the external tool may have keyed on the other one.
pub(crate) fn candidate_key_columns(axis: &'static str, batch: &RecordBatch) -> Vec<String> {
    let schema = batch.schema();
    let fallbacks = if axis == "obs" {
        OBS_KEY_FALLBACKS
    } else {
        VAR_KEY_FALLBACKS
    };
    fallbacks
        .iter()
        .filter(|c| {
            schema
                .field_with_name(c)
                .is_ok_and(|f| is_string_column(f.data_type()))
        })
        .map(|c| c.to_string())
        .collect()
}

/// Materialize a string column as owned `String`s, resolving dictionaries.
pub(crate) fn string_column(batch: &RecordBatch, name: &str) -> Result<Vec<String>> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| OpsError::InvalidInput(format!("column '{name}' disappeared")))?;
    let plain = arrow::compute::cast(col, &DataType::Utf8)
        .map_err(|e| OpsError::InvalidInput(format!("cannot read '{name}' as strings: {e}")))?;
    let arr = plain
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| OpsError::InvalidInput(format!("column '{name}' is not Utf8 after cast")))?;
    // A null key becomes "". Two or more nulls are caught by the duplicate-key
    // check; a single one can only mis-join against a literal empty-string key
    // on the other side, which no real barcode or gene id is.
    Ok((0..arr.len())
        .map(|i| {
            if arr.is_null(i) {
                String::new()
            } else {
                arr.value(i).to_string()
            }
        })
        .collect())
}

struct RowJoin {
    /// For each target row, the source row that supplies it.
    source_of_target: Vec<Option<u32>>,
    n_matched: u64,
    n_target_absent: u64,
    n_source_absent: u64,
    n_source_absent_nonzero: u64,
}

/// How to report a join in which fewer than half the target rows matched, or
/// `None` when coverage needs no comment. The layer-side twin of
/// `external_obs::obs_join_coverage_report` — see it for the full E2 rationale.
///
/// Partial coverage is the *normal* shape here, not an edge case: CellBender's
/// `_filtered.h5` holds only the droplets it called as cells, while the target is
/// the raw all-droplet file. Reporting that as a failed join, complete with a
/// target/source example pair that looks like evidence of divergence, sends the
/// user hunting a barcode-format bug that does not exist.
fn layer_join_coverage_report(
    n_matched: u64,
    n_target_total: u64,
    n_target_absent: u64,
    n_source_absent: u64,
    n_source_absent_nonzero: u64,
    examples: impl FnOnce() -> (Vec<String>, Vec<String>),
) -> Option<(log::Level, String)> {
    if n_matched * 2 >= n_target_total {
        return None;
    }
    if n_source_absent == 0 {
        return Some((
            log::Level::Info,
            format!(
                "external layer join coverage: {n_matched} of {n_target_total} target \
                 rows matched; {n_target_absent} rows left to the missing-row policy. \
                 Every source row matched, so this is a partial-coverage import \
                 (expected when the source is a cell-called subset of an all-droplet \
                 target), not a key mismatch."
            ),
        ));
    }
    let (target_examples, source_examples) = examples();
    Some((
        log::Level::Warn,
        format!(
            "external layer join matched only {n_matched} of {n_target_total} target \
             rows, and {n_source_absent} source rows matched no target row \
             ({n_source_absent_nonzero} of them carry counts) — check for a sample-name \
             prefix or a '-1' suffix difference; target examples {target_examples:?}, \
             source examples {source_examples:?}"
        ),
    ))
}

pub(crate) fn examples(keys: &[String]) -> Vec<&str> {
    keys.iter().take(KEY_EXAMPLES).map(|s| s.as_str()).collect()
}

/// Example keys drawn from the ones that did **not** match, falling back to the
/// head when everything matched.
///
/// The whole point of showing examples next to a partial-mismatch warning is to
/// let the user *see* the difference — a target `"100000"` beside a source
/// `"100000-1"` names the `-1` suffix instantly. Head examples cannot do that:
/// on a source whose first 100k keys match and whose last 150k carry a stray
/// suffix, `examples()` returns the matching prefix and both lists print
/// identically, which is the "both lists are valid keys from the same space"
/// confusion the dogfood report flagged (E2).
pub(crate) fn unmatched_examples<'a>(keys: &'a [String], matched: &[bool]) -> Vec<&'a str> {
    let picked: Vec<&str> = keys
        .iter()
        .zip(matched.iter())
        .filter(|(_, &m)| !m)
        .map(|(k, _)| k.as_str())
        .take(KEY_EXAMPLES)
        .collect();
    if picked.is_empty() {
        examples(keys)
    } else {
        picked
    }
}

pub(crate) fn first_duplicates(keys: &[String]) -> Vec<&str> {
    let mut seen = std::collections::HashSet::new();
    let mut dups = Vec::new();
    for k in keys {
        if !seen.insert(k.as_str()) && !dups.contains(&k.as_str()) {
            dups.push(k.as_str());
            if dups.len() == 5 {
                break;
            }
        }
    }
    dups
}

fn build_row_join(
    target_keys: &[String],
    data: &ExternalLayerData,
    opts: &AttachLayerOptions,
) -> Result<RowJoin> {
    let source_keys = &data.row_keys;
    let dups = first_duplicates(target_keys);
    if !dups.is_empty() {
        return Err(OpsError::DuplicateJoinKey {
            axis: "obs",
            detail: format!("target obs key column contains duplicates: {dups:?}"),
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
        return Err(OpsError::AxisMismatch {
            axis: "obs",
            detail: format!(
                "no target row key matched any source row key. Target examples: \
                 {:?}; source examples: {:?}. Check for a sample-name prefix or a \
                 '-1' suffix difference.",
                examples(target_keys),
                examples(source_keys)
            ),
        });
    }

    let n_source_absent = used.iter().filter(|u| !**u).count() as u64;
    // Only unmatched source rows that actually carried counts represent real
    // data being discarded; the rest are the all-zero rows an external tool
    // emits for droplets it did not analyse. Reporting them separately keeps a
    // routine raw-vs-filtered import from looking alarming.
    let n_source_absent_nonzero = used
        .iter()
        .enumerate()
        .filter(|(i, u)| !**u && data.indptr[*i + 1] > data.indptr[*i])
        .count() as u64;
    if n_source_absent > 0 && opts.extra_row_policy == ExtraRowPolicy::Error {
        return Err(OpsError::AxisMismatch {
            axis: "obs",
            detail: format!(
                "{n_source_absent} source rows have no matching target row \
                 ({n_source_absent_nonzero} of them carry counts) \
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
    if let Some((level, msg)) = layer_join_coverage_report(
        n_matched,
        target_keys.len() as u64,
        n_target_absent,
        n_source_absent,
        n_source_absent_nonzero,
        || {
            // Unmatched keys, for the same reason as the obs sibling: a
            // CellBender barcode that found no raw droplet is what shows the
            // format difference, not one that matched.
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

    Ok(RowJoin {
        source_of_target,
        n_matched,
        n_target_absent,
        n_source_absent,
        n_source_absent_nonzero,
    })
}

/// Map each source column index to a target column index.
///
/// `None` means the identity map (source and target keys are elementwise
/// equal), which lets the gather skip a per-nnz lookup entirely.
fn build_column_map(
    target_keys: &[String],
    source_keys: &[String],
    policy: ColumnAxisPolicy,
) -> Result<(Option<Vec<u32>>, ColumnAxisMatch)> {
    if target_keys == source_keys {
        return Ok((None, ColumnAxisMatch::Identical));
    }

    let dups = first_duplicates(target_keys);
    if !dups.is_empty() {
        return Err(OpsError::DuplicateJoinKey {
            axis: "var",
            detail: format!("target var key column contains duplicates: {dups:?}"),
        });
    }

    let index: HashMap<&str, u32> = target_keys
        .iter()
        .enumerate()
        .map(|(i, k)| (k.as_str(), i as u32))
        .collect();

    let mut map = Vec::with_capacity(source_keys.len());
    let mut missing = Vec::new();
    for key in source_keys {
        match index.get(key.as_str()) {
            Some(&t) => map.push(t),
            None => {
                if missing.len() < KEY_EXAMPLES {
                    missing.push(key.as_str());
                }
            }
        }
    }
    if !missing.is_empty() {
        // Always an error: importing would silently drop those columns' counts.
        return Err(OpsError::AxisMismatch {
            axis: "var",
            detail: format!(
                "source has {} column key(s) absent from the target var axis \
                 (e.g. {missing:?}); importing would silently drop their counts",
                source_keys.len() - map.len()
            ),
        });
    }

    let matched = if source_keys.len() == target_keys.len() {
        ColumnAxisMatch::Reordered
    } else {
        ColumnAxisMatch::Subset
    };
    let allowed = matches!(
        (matched, policy),
        (_, ColumnAxisPolicy::AllowSubset)
            | (ColumnAxisMatch::Reordered, ColumnAxisPolicy::AllowReorder)
    );
    if !allowed {
        return Err(OpsError::AxisMismatch {
            axis: "var",
            detail: format!(
                "source column axis is {matched:?} relative to the target but \
                 column_axis_policy is {policy:?}. A silent permutation would \
                 produce plausible-looking, biologically wrong values, so it must \
                 be opted into."
            ),
        });
    }
    Ok((Some(map), matched))
}

// ---------------------------------------------------------------------------
// Shard planning and gather
// ---------------------------------------------------------------------------

fn framing_for(prep: &crate::in_place::InPlacePrep) -> Option<FramingConfig> {
    // Match `X`'s framing: an unframed shard-v1 section inside a v4 file is
    // rejected by readers, and `adopt_in_place` defaults framing to None.
    (prep.header.format_version >= scx_format_io::CURRENT_FORMAT_VERSION)
        .then(FramingConfig::default)
}

/// Layer shards must tile `[0, n_obs)` exactly the way `X` does, or shard-wise
/// readers misalign. `assemble_shards` does not verify coverage, so a short or
/// gapped layer would only surface much later inside anndata.
fn derive_shard_ranges(
    catalog: &FullCatalog,
    prep: &crate::in_place::InPlacePrep,
    n_obs: u64,
) -> (Vec<(u64, u64)>, ShardRangeSource) {
    let mut ranges: Vec<(u64, u64)> = catalog
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CsrShard && e.modality_id == 0)
        .filter_map(|e| e.stats.as_ref().map(|s| (s.row_start, s.row_end)))
        .collect();
    ranges.sort_by_key(|(s, _)| *s);

    let contiguous = !ranges.is_empty()
        && ranges[0].0 == 0
        && ranges.last().map(|(_, e)| *e) == Some(n_obs)
        && ranges.windows(2).all(|w| w[0].1 == w[1].0);
    if contiguous {
        return (ranges, ShardRangeSource::XShards);
    }

    log::warn!(
        "X CSR shards do not form a gap-free cover of [0, {n_obs}); chunking the \
         layer by shard_target_rows instead"
    );
    let step = (prep.header.shard_target_rows as u64).max(1);
    let mut out = Vec::new();
    let mut start = 0u64;
    while start < n_obs {
        let end = (start + step).min(n_obs);
        out.push((start, end));
        start = end;
    }
    if out.is_empty() {
        out.push((0, 0));
    }
    (out, ShardRangeSource::ShardTargetRowsFallback)
}

/// Gather target rows `[row_start, row_end)` from the source matrix through the
/// join, remapping column indices when the axes differ.
fn gather_shard(
    data: &ExternalLayerData,
    join: &RowJoin,
    col_map: &Option<Vec<u32>>,
    row_start: u64,
    row_end: u64,
) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let mut indptr = Vec::with_capacity((row_end - row_start) as usize + 1);
    let mut indices = Vec::new();
    let mut values = Vec::new();
    indptr.push(0u64);

    for target_row in row_start..row_end {
        if let Some(src) = join.source_of_target[target_row as usize] {
            let s = data.indptr[src as usize] as usize;
            let e = data.indptr[src as usize + 1] as usize;
            match col_map {
                None => {
                    indices.extend_from_slice(&data.indices[s..e]);
                    values.extend_from_slice(&data.values[s..e]);
                }
                Some(map) => {
                    for k in s..e {
                        indices.push(map[data.indices[k] as usize]);
                        values.push(data.values[k]);
                    }
                }
            }
        }
        indptr.push(indices.len() as u64);
    }
    (indptr, indices, values)
}

// ---------------------------------------------------------------------------
// obs / var / uns / obsm construction
// ---------------------------------------------------------------------------

fn planned_obs_columns(data: &ExternalLayerData, opts: &AttachLayerOptions) -> Vec<String> {
    let mut cols = Vec::new();
    if let Some(name) = &opts.status_column {
        cols.push(name.clone());
    }
    if let Some(ann) = &data.row_annotations {
        cols.extend(ann.schema().fields().iter().map(|f| f.name().clone()));
    }
    if let Some(name) = &opts.row_sum_column {
        cols.push(name.clone());
    }
    cols
}

fn planned_var_columns(data: &ExternalLayerData) -> Vec<String> {
    data.col_annotations
        .as_ref()
        .map(|a| {
            a.schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect()
        })
        .unwrap_or_default()
}

/// Scatter a source-row-indexed array onto target rows `[row_start, row_end)`,
/// filling `null` for unmatched rows. `null`, not a zero: `cell_probability =
/// 0.0` on an un-analysed droplet is a scientific claim the tool never made.
///
/// The row window is what lets the build run one obs shard at a time.
fn scatter_column(
    col: &ArrayRef,
    join: &RowJoin,
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

/// Append the import's columns to `obs`, which covers global obs rows
/// `[row_start, row_start + obs.num_rows())`. See the obs op's twin: the whole
/// point of `row_start` is that the streaming and materializing callers share
/// one builder rather than two that agree until one is edited.
fn build_new_obs(
    obs: &RecordBatch,
    data: &ExternalLayerData,
    opts: &AttachLayerOptions,
    join: &RowJoin,
    row_start: usize,
) -> Result<RecordBatch> {
    let n = obs.num_rows();
    let row_end = row_start + n;
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

    // Drop any column we are about to add (overwrite path).
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

    if let Some(ann) = &data.row_annotations {
        for (i, f) in ann.schema().fields().iter().enumerate() {
            let scattered = scatter_column(ann.column(i), join, row_start, row_end)?;
            // Unmatched rows become null, so the field must admit nulls.
            fields.push(Field::new(f.name(), f.data_type().clone(), true));
            columns.push(scattered);
        }
    }

    // Row sums of the *emitted* layer, computed through the join so a
    // zero-filled row reads 0.0 rather than null. Sized to this shard, not to
    // n_obs — the sums are derived from the source, so any row window can be
    // computed independently.
    let mut row_sums = vec![0.0f32; n];
    #[allow(clippy::needless_range_loop)]
    for (t, src) in join.source_of_target[row_start..row_end].iter().enumerate() {
        if let Some(s) = src {
            let (a, b) = (
                data.indptr[*s as usize] as usize,
                data.indptr[*s as usize + 1] as usize,
            );
            row_sums[t] = data.values[a..b].iter().sum();
        }
    }
    if let Some(name) = &opts.row_sum_column {
        fields.push(Field::new(name, DataType::Float32, false));
        columns.push(Arc::new(Float32Array::from(row_sums)) as ArrayRef);
    }

    let schema = Arc::new(Schema::new(fields).with_metadata(obs.schema().metadata().clone()));
    RecordBatch::try_new(schema, columns)
        .map_err(|e| OpsError::InvalidInput(format!("failed to build new obs: {e}")))
}

fn build_new_var(
    var: &RecordBatch,
    data: &ExternalLayerData,
    col_map: &Option<Vec<u32>>,
    _matched: ColumnAxisMatch,
) -> Result<RecordBatch> {
    let Some(ann) = &data.col_annotations else {
        return Ok(var.clone());
    };
    let n_vars = var.num_rows();

    // Invert the source→target column map so annotations land on the right gene.
    let source_of_target: Vec<Option<u32>> = match col_map {
        None => (0..n_vars).map(|i| Some(i as u32)).collect(),
        Some(map) => {
            let mut inv = vec![None; n_vars];
            for (src, &tgt) in map.iter().enumerate() {
                inv[tgt as usize] = Some(src as u32);
            }
            inv
        }
    };
    let join = RowJoin {
        source_of_target,
        n_matched: 0,
        n_target_absent: 0,
        n_source_absent: 0,
        n_source_absent_nonzero: 0,
    };

    let planned = planned_var_columns(data);
    let mut fields: Vec<Field> = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();
    for (i, f) in var.schema().fields().iter().enumerate() {
        if !planned.contains(f.name()) {
            fields.push(f.as_ref().clone());
            columns.push(var.column(i).clone());
        }
    }
    // The whole var axis: this is the gene axis, always read and written whole.
    let n_var_rows = join.source_of_target.len();
    for (i, f) in ann.schema().fields().iter().enumerate() {
        fields.push(Field::new(f.name(), f.data_type().clone(), true));
        columns.push(scatter_column(ann.column(i), &join, 0, n_var_rows)?);
    }

    let schema = Arc::new(Schema::new(fields).with_metadata(var.schema().metadata().clone()));
    RecordBatch::try_new(schema, columns)
        .map_err(|e| OpsError::InvalidInput(format!("failed to build new var: {e}")))
}

fn build_new_uns(mut uns: Value, data: &ExternalLayerData, opts: &AttachLayerOptions) -> Value {
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

/// Scatter the source embeddings onto the full target row axis.
///
/// Unlike obs, this is **not** bounded: `write_obsm` emits one section, so the
/// batch has to be `n_obs` rows. Only reached when the caller supplies
/// embeddings; bounding it needs sharded `ObsmEmbeddingShard` writes.
fn build_obsm(
    data: &ExternalLayerData,
    join: &RowJoin,
    _n_obs: usize,
) -> Result<Vec<(String, RecordBatch)>> {
    let n_obs = join.source_of_target.len();
    let mut out = Vec::new();
    for (name, batch) in &data.row_embeddings {
        let mut fields = Vec::new();
        let mut columns = Vec::new();
        for (i, f) in batch.schema().fields().iter().enumerate() {
            fields.push(Field::new(f.name(), f.data_type().clone(), true));
            columns.push(scatter_column(batch.column(i), join, 0, n_obs)?);
        }
        let schema = Arc::new(Schema::new(fields));
        let scattered = RecordBatch::try_new(schema, columns)
            .map_err(|e| OpsError::InvalidInput(format!("failed to scatter obsm '{name}': {e}")))?;
        out.push((name.clone(), scattered));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Collision detection and catalog bookkeeping
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn check_collisions(
    reader: &ScxReader,
    _catalog: &FullCatalog,
    obs_schema: &Schema,
    var: &RecordBatch,
    uns: &Value,
    data: &ExternalLayerData,
    opts: &AttachLayerOptions,
    obs_new: &[String],
    var_new: &[String],
) -> Result<()> {
    let existing_layers = reader.layer_names();
    if existing_layers.iter().any(|l| l == &opts.layer_name) {
        return Err(OpsError::InvalidInput(format!(
            "layer '{}' already exists; pass overwrite=true to replace it",
            opts.layer_name
        )));
    }
    for name in obs_new {
        if obs_schema.field_with_name(name).is_ok() {
            return Err(OpsError::InvalidInput(format!(
                "obs column '{name}' already exists; pass overwrite=true to replace it"
            )));
        }
    }
    let var_schema = var.schema();
    for name in var_new {
        if var_schema.field_with_name(name).is_ok() {
            return Err(OpsError::InvalidInput(format!(
                "var column '{name}' already exists; pass overwrite=true to replace it"
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
/// Deliberately narrow. The predicate indexes are **kept** whenever this import
/// only *adds* columns: they key on column name and CSR shard range with no
/// schema hash, and this op never touches CSR shards, so dropping them (as
/// `modify_metadata` must) would silently kill query pushdown. The exception is
/// `drop_obs_index` — an overwrite of a column the obs index covers, where
/// keeping it would leave pushdown reading values that no longer exist.
fn should_drop_old_entry(
    e: &FullCatalogEntry,
    opts: &AttachLayerOptions,
    drop_obs_index: bool,
    obsm: &[(String, RecordBatch)],
) -> bool {
    use SectionType::*;
    if e.section_type == Provenance {
        return true;
    }
    // The one case where the index is NOT safe to keep: this import overwrites
    // an obs column the index covers, so its entries now describe stale values.
    if drop_obs_index && e.section_type == ObsPredicateIndex {
        return true;
    }
    if e.section_type == UnsBlob && e.modality_id == 0 {
        return true;
    }
    if matches!(
        e.section_type,
        ObsMetadata | ObsMetadataShard | VarMetadata | VarMetadataShard
    ) {
        return true;
    }
    // The re-import guard: without this the merged catalog keeps both shard
    // families and the layer reads back at 2x n_obs.
    if matches!(e.section_type, LayerCsrShard | LayerCscShard)
        && e.modality_id == 0
        && layer_entry_matches(&e.name, &opts.layer_name)
    {
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
    data: &ExternalLayerData,
    opts: &AttachLayerOptions,
    s: &AttachLayerSummary,
) -> Value {
    let mut v = serde_json::json!({
        "layer": opts.layer_name,
        "source_file": data.source_name,
        "obs_key_column": s.obs_key_column,
        "var_key_column": s.var_key_column,
        "column_axis_match": format!("{:?}", s.column_axis_match).to_lowercase(),
        "n_obs": s.n_obs,
        "n_matched": s.n_matched,
        "n_target_rows_absent": s.n_target_rows_absent,
        "n_source_rows_absent": s.n_source_rows_absent,
        "n_source_rows_absent_nonzero": s.n_source_rows_absent_nonzero,
        "layer_nnz": s.layer_nnz,
        "value_encoding": format!("{:?}", s.value_encoding).to_lowercase(),
        "shard_ranges_from": format!("{:?}", s.shard_ranges_from),
        "obs_columns": s.obs_columns_added,
        "var_columns": s.var_columns_added,
        "obsm_keys": s.obsm_keys_added,
        // See the obs op's twin: not recoverable from the output, because both
        // rewrite paths produce a sharded obs.
        "obs_streamed": s.obs_streamed,
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

#[cfg(test)]
#[path = "external_layer_tests.rs"]
mod tests;
