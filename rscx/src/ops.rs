// Phase E: File Operations R bindings
//
// Wraps scx-ops (append, mark_deleted, compact, rollback, merge) for R,
// plus scx_info and scx_validate from scx-format.
// Follows the same pattern as pyscx/src/ops.rs, adapted for extendr.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use extendr_api::prelude::*;

use scx_codec::{CodecId, CodecSelection};
use scx_engine::ConversionPredicateIndexOptions;
use scx_format_io::ScxReader;
use scx_ops::{AppendOptions, MergeOptions, UnsPolicy};

/// Build conversion predicate-index options from R inputs. Empty `Strings`
/// (R `character(0)`) mean "no forced columns"; `index_auto_threshold` 0
/// disables auto-detection.
fn index_options(
    index_obs: Strings,
    index_var: Strings,
    index_preset: Nullable<String>,
    index_auto_threshold: i32,
) -> ConversionPredicateIndexOptions {
    ConversionPredicateIndexOptions {
        index_obs: index_obs.iter().map(|s| s.to_string()).collect(),
        index_var: index_var.iter().map(|s| s.to_string()).collect(),
        index_preset: match index_preset {
            Nullable::NotNull(s) => Some(s),
            Nullable::Null => None,
        },
        index_auto_threshold: index_auto_threshold.max(0) as usize,
    }
}

/// Parse a codec string into a [`CodecSelection`]. `NULL` / `"auto"` ⇒ `Auto`;
/// otherwise an explicit codec (`none`/`scx1`/`zstd`/`lz4`/`pcodec`/`shufdelta`).
fn parse_codec(codec: Nullable<String>) -> Result<CodecSelection> {
    match codec {
        Nullable::Null => Ok(CodecSelection::Auto),
        Nullable::NotNull(s) => CodecId::parse_cli(&s)
            .map_err(Error::Other)
            .map(|o| o.map_or(CodecSelection::Auto, CodecSelection::Explicit)),
    }
}

// ---------------------------------------------------------------------------
// Append
// ---------------------------------------------------------------------------

/// Append cells from one SCX file to another (in place), with options.
///
/// Streams the input's X / obs straight from a reader (per-shard value
/// encodings preserved), mirroring `pyscx.append`.
///
/// @param target Path to the target SCX file (modified in place).
/// @param input Path to the input SCX file to append from.
/// @param codec Codec for the new shards: `NULL`/`"auto"` (per-shard
///   auto-selection) or one of
///   `"none"`/`"scx1"`/`"zstd"`/`"lz4"`/`"pcodec"`/`"shufdelta"`.
/// @param shard_size Target rows per CSR shard (default 16384; must be > 0).
/// @param index_obs,index_var Obs/var columns to build predicate indexes for.
/// @param index_preset Named index preset (`cellxgene`/`perturbseq`/`training`).
/// @param index_auto_threshold Auto-index cardinality cap (0 = disabled).
/// @param modality Modality NAME to append into on a multimodal file (`NULL`
///   for single-modality / the global axis).
///
/// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3): a
/// fallible `#[extendr]` fn would otherwise `unwrap()`-panic in extendr 0.8.0,
/// masking the real message behind "User function panicked".
#[extendr]
#[allow(clippy::too_many_arguments)]
fn scx_append(
    target: &str,
    input: &str,
    codec: Nullable<String>,
    shard_size: i32,
    index_obs: Strings,
    index_var: Strings,
    index_preset: Nullable<String>,
    index_auto_threshold: i32,
    modality: Nullable<String>,
) -> Robj {
    crate::util::throw_on_err(scx_append_impl(
        target,
        input,
        codec,
        shard_size,
        index_obs,
        index_var,
        index_preset,
        index_auto_threshold,
        modality,
    ))
}

#[allow(clippy::too_many_arguments)]
fn scx_append_impl(
    target: &str,
    input: &str,
    codec: Nullable<String>,
    shard_size: i32,
    index_obs: Strings,
    index_var: Strings,
    index_preset: Nullable<String>,
    index_auto_threshold: i32,
    modality: Nullable<String>,
) -> Result<()> {
    let input_reader =
        ScxReader::open(input).map_err(|e| Error::Other(format!("failed to open input: {e}")))?;
    let target_reader =
        ScxReader::open(target).map_err(|e| Error::Other(format!("failed to open target: {e}")))?;

    // Resolve the modality NAME (if any) against both readers → ids.
    let (target_modality_id, source_modality_id) = match modality {
        Nullable::Null => {
            // Single-modality / global append: modality_id 0 is the implicit
            // global axis for both target and source. Validate file-wide n_vars.
            if target_reader.n_vars() != input_reader.n_vars() {
                return Err(Error::Other(format!(
                    "n_vars mismatch: target has {}, input has {}",
                    target_reader.n_vars(),
                    input_reader.n_vars()
                )));
            }
            (0u8, 0u8)
        }
        Nullable::NotNull(name) => {
            let tid = target_reader.modality_id(&name).ok_or_else(|| {
                Error::Other(format!(
                    "target file has no modality named '{name}' (use scx_info / a multimodal file)"
                ))
            })?;
            let sid = input_reader.modality_id(&name).ok_or_else(|| {
                Error::Other(format!("input file has no modality named '{name}'"))
            })?;
            (tid, sid)
        }
    };

    let shard_target_rows = u32::try_from(shard_size)
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or_else(|| Error::Other("shard_size must be > 0".into()))?;

    let options = AppendOptions {
        codec: parse_codec(codec)?,
        shard_target_rows,
        modality_id: target_modality_id,
    };
    let idx = index_options(index_obs, index_var, index_preset, index_auto_threshold);

    let target_path = PathBuf::from(target);
    scx_ops::append_from_reader_with_index_options(
        &target_path,
        &input_reader,
        &options,
        source_modality_id,
        &idx,
    )
    .map(|_| ())
    .map_err(|e| Error::Other(e.to_string()))
}

// ---------------------------------------------------------------------------
// Delete
// ---------------------------------------------------------------------------

/// Mark specific cell indices as logically deleted.
///
/// Note: indices cross the FFI boundary as R doubles (`f64`), matching the
/// backed reader's row-index contract so cell indices beyond `.Machine$integer.max`
/// (2^31) remain addressable (an `i32` vector would truncate them to `NA`).
/// Each value is validated (finite, non-negative, integral) and converted to
/// `u64`. Returns the total number of deleted cells (including previously
/// deleted).
///
/// @param path Path to the SCX file.
/// @param cell_indices 0-based cell indices at this FFI boundary. The R-facing
///   `scx_delete()` wrapper takes **1-based** indices (R convention) and subtracts
///   1 before calling in, so values arrive here already 0-based.
/// @return Total number of deleted cells as numeric (f64 to avoid i32 overflow).
///
/// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3).
#[extendr]
fn scx_delete(path: &str, cell_indices: Vec<f64>) -> Robj {
    crate::util::throw_on_err(scx_delete_impl(path, cell_indices))
}

fn scx_delete_impl(path: &str, cell_indices: Vec<f64>) -> Result<Robj> {
    let indices: Vec<u64> = cell_indices
        .into_iter()
        .map(|v| {
            if !v.is_finite() {
                Err(Error::Other(format!("non-finite cell index: {v}")))
            } else if v < 0.0 {
                Err(Error::Other(format!("negative cell index: {v}")))
            } else if v.fract() != 0.0 {
                Err(Error::Other(format!("non-integer cell index: {v}")))
            } else {
                Ok(v as u64)
            }
        })
        .collect::<Result<Vec<u64>>>()?;

    let p = Path::new(path);
    let total = scx_ops::mark_deleted(p, &indices).map_err(|e| Error::Other(e.to_string()))?;
    Ok(Robj::from(total as f64))
}

// ---------------------------------------------------------------------------
// Compact
// ---------------------------------------------------------------------------

/// Rewrite an SCX file reclaiming deleted/orphaned space, with options.
///
/// @param input Path to the input SCX file.
/// @param output Path for the compacted output file.
/// @param index_obs,index_var Obs/var columns to build predicate indexes for.
/// @param index_preset Named index preset (`cellxgene`/`perturbseq`/`training`).
/// @param index_auto_threshold Auto-index cardinality cap (0 = disabled).
/// @param reshape_obs Migrate legacy single-section obs to row-sharded layout.
///
/// `shard_target_rows` is inherited from the input header; CSC sidecars are
/// dropped (rebuild separately).
///
/// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3).
#[extendr]
fn scx_compact(
    input: &str,
    output: &str,
    index_obs: Strings,
    index_var: Strings,
    index_preset: Nullable<String>,
    index_auto_threshold: i32,
    reshape_obs: bool,
) -> Robj {
    crate::util::throw_on_err(scx_compact_impl(
        input,
        output,
        index_obs,
        index_var,
        index_preset,
        index_auto_threshold,
        reshape_obs,
    ))
}

fn scx_compact_impl(
    input: &str,
    output: &str,
    index_obs: Strings,
    index_var: Strings,
    index_preset: Nullable<String>,
    index_auto_threshold: i32,
    reshape_obs: bool,
) -> Result<()> {
    let idx = index_options(index_obs, index_var, index_preset, index_auto_threshold);
    scx_ops::compact_with_index_options(Path::new(input), Path::new(output), &idx, reshape_obs)
        .map(|_| ())
        .map_err(|e| Error::Other(e.to_string()))
}

// ---------------------------------------------------------------------------
// Rollback
// ---------------------------------------------------------------------------

/// Roll back to a previous manifest version.
///
/// If `to_seq` is NULL or negative, rolls back one version.
/// If `to_seq` is a non-negative integer, rolls back to that specific
/// manifest sequence number.
///
/// @param path Path to the SCX file.
/// @param to_seq Optional manifest sequence number (integer or NULL).
///
/// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3).
#[extendr]
fn scx_rollback(path: &str, to_seq: Nullable<i32>) -> Robj {
    crate::util::throw_on_err(scx_rollback_impl(path, to_seq))
}

fn scx_rollback_impl(path: &str, to_seq: Nullable<i32>) -> Result<()> {
    let p = Path::new(path);
    match to_seq {
        Nullable::NotNull(seq) if seq >= 0 => {
            scx_ops::rollback_to(p, seq as u64).map_err(|e| Error::Other(e.to_string()))
        }
        _ => scx_ops::rollback(p).map_err(|e| Error::Other(e.to_string())),
    }
}

// ---------------------------------------------------------------------------
// Merge
// ---------------------------------------------------------------------------

/// Merge multiple SCX files into one, with options.
///
/// Requires at least 2 input files. Var identity is validated by default.
///
/// @param inputs Character vector of input file paths.
/// @param output Path for the merged output file.
/// @param index_obs,index_var Obs/var columns to build predicate indexes for.
/// @param index_preset Named index preset (`cellxgene`/`perturbseq`/`training`).
/// @param index_auto_threshold Auto-index cardinality cap (0 = disabled).
/// @param assume_identical_var,assume_identical_obs Skip the var / obs schema
///   identity checks across inputs.
/// @param uns_policy How to combine `uns`: `first` / `require-equal` /
///   `namespace` / `summary`.
/// @param sort_by Obs columns for a globally-ordered (sorted) k-way merge;
///   empty = legacy concatenation.
/// @param reverse Descending order when `sort_by` is set.
///
/// `shard_target_rows` is inherited from the first input; CSC sidecars are
/// dropped (rebuild separately).
///
/// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3).
#[extendr]
#[allow(clippy::too_many_arguments)]
fn scx_merge(
    inputs: Vec<String>,
    output: &str,
    index_obs: Strings,
    index_var: Strings,
    index_preset: Nullable<String>,
    index_auto_threshold: i32,
    assume_identical_var: bool,
    assume_identical_obs: bool,
    uns_policy: &str,
    sort_by: Strings,
    reverse: bool,
) -> Robj {
    crate::util::throw_on_err(scx_merge_impl(
        inputs,
        output,
        index_obs,
        index_var,
        index_preset,
        index_auto_threshold,
        assume_identical_var,
        assume_identical_obs,
        uns_policy,
        sort_by,
        reverse,
    ))
}

#[allow(clippy::too_many_arguments)]
fn scx_merge_impl(
    inputs: Vec<String>,
    output: &str,
    index_obs: Strings,
    index_var: Strings,
    index_preset: Nullable<String>,
    index_auto_threshold: i32,
    assume_identical_var: bool,
    assume_identical_obs: bool,
    uns_policy: &str,
    sort_by: Strings,
    reverse: bool,
) -> Result<()> {
    if inputs.len() < 2 {
        return Err(Error::Other("merge requires at least 2 input files".into()));
    }
    let input_paths: Vec<PathBuf> = inputs.iter().map(PathBuf::from).collect();
    let input_refs: Vec<&Path> = input_paths.iter().map(|p| p.as_path()).collect();

    let uns = UnsPolicy::parse(uns_policy).ok_or_else(|| {
        Error::Other(format!(
            "unknown uns_policy '{uns_policy}' (expected 'first', 'require-equal', \
             'namespace', or 'summary')"
        ))
    })?;

    let options = MergeOptions {
        index_options: index_options(index_obs, index_var, index_preset, index_auto_threshold),
        assume_identical_var,
        assume_identical_obs,
        // `scx_format_io::ResolvedCodec::AUTO` — the adaptive default, spelled
        // via `Default` so this file needs no new import. rscx's `scx_merge()`
        // exposes no `codec` argument yet; add one alongside the R docs when it
        // does, mirroring `pyscx.merge(codec=...)`.
        codec: Default::default(),
        uns_policy: uns,
        shard_target_rows: None,
        sort_by: sort_by.iter().map(|s| s.to_string()).collect(),
        sort_reverse: reverse,
    };

    scx_ops::merge_with_options(&input_refs, Path::new(output), &options)
        .map(|_| ())
        .map_err(|e| Error::Other(e.to_string()))
}

// ---------------------------------------------------------------------------
// Info
// ---------------------------------------------------------------------------

/// Return file information as a named list.
///
/// @param path Path to the SCX file.
/// @return Named list with n_obs, n_vars, nnz, format_version, n_shards.
///
/// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3).
#[extendr]
fn scx_info(path: &str) -> Robj {
    crate::util::throw_on_err(scx_info_impl(path))
}

fn scx_info_impl(path: &str) -> Result<Robj> {
    let reader = ScxReader::open(path).map_err(|e| Error::Other(format!("failed to open: {e}")))?;
    // Extract to local variables for R!() interpolation
    let n_obs = reader.n_obs() as f64;
    let n_vars = reader.n_vars() as f64;
    let nnz = reader.nnz() as f64;
    let format_version = reader.header().format_version as i32;
    let n_shards = reader.header().n_csr_shards as i32;
    R!("list(
        n_obs = {{n_obs}},
        n_vars = {{n_vars}},
        nnz = {{nnz}},
        format_version = {{format_version}},
        n_shards = {{n_shards}}
    )")
    .map_err(|e| Error::Other(e.to_string()))
}

// ---------------------------------------------------------------------------
// Validate
// ---------------------------------------------------------------------------

/// Validate an SCX file. Returns TRUE if valid, otherwise raises an error.
///
/// @param path Path to the SCX file.
/// @return TRUE if all checksums pass.
///
/// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3).
#[extendr]
fn scx_validate(path: &str) -> Robj {
    crate::util::throw_on_err(scx_validate_impl(path))
}

fn scx_validate_impl(path: &str) -> Result<bool> {
    let reader = ScxReader::open(path).map_err(|e| Error::Other(format!("failed to open: {e}")))?;
    let checks = reader.validate().map_err(|e| Error::Other(e.to_string()))?;
    let all_passed = checks.iter().all(|(_, ok)| *ok);
    if !all_passed {
        let failures: Vec<String> = checks
            .iter()
            .filter(|(_, ok)| !ok)
            .map(|(name, _)| name.clone())
            .collect();
        return Err(Error::Other(format!(
            "validation failed: {}",
            failures.join(", ")
        )));
    }
    Ok(true)
}

// ---------------------------------------------------------------------------
// Attach external obs
// ---------------------------------------------------------------------------

/// Attach an R `data.frame` of per-cell annotations to an SCX file, in place.
///
/// The R end of the doublet-caller interop. scDblFinder, DoubletFinder and
/// scds all run directly on an rscx-loaded object, so on this path there is no
/// export file in either direction — the caller hands back a `data.frame` and
/// it lands on the file.
///
/// @param path Path to the SCX file (modified in place).
/// @param df The annotations. Columns become obs columns; an R factor arrives
///   as a dictionary and is preserved as one.
/// @param key Character vector of one key per row of `df` — usually
///   `rownames(df)`. Empty means "resolve a key column from `df` instead".
/// @param key_columns Columns **of `df`** to fuse into a composite key,
///   joined against target obs columns of the same names. Mutually exclusive
///   with `key`.
/// @param key_column Target obs column to join on when `key` is given.
///   `NULL` auto-resolves it.
/// @param prefix Prepended to every imported column name.
/// @param status_column Obs column recording "present"/"absent" per row.
/// @param uns_key `uns` key for the run metadata.
/// @param overwrite Replace colliding columns. REPLACES, never merges.
/// @param on_missing_rows `"zero"` or `"error"`.
/// @param on_extra_rows `"warn"` or `"error"`.
/// @param dry_run Validate and join without writing.
/// @return Named list summarising the join.
///
/// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3).
#[extendr]
#[allow(clippy::too_many_arguments)]
fn scx_attach_obs(
    path: &str,
    df: Robj,
    key: Strings,
    key_columns: Strings,
    key_column: Nullable<String>,
    prefix: &str,
    status_column: Nullable<String>,
    uns_key: Nullable<String>,
    overwrite: bool,
    on_missing_rows: &str,
    on_extra_rows: &str,
    dry_run: bool,
) -> Robj {
    crate::util::throw_on_err(scx_attach_obs_impl(
        path,
        df,
        key,
        key_columns,
        key_column,
        prefix,
        status_column,
        uns_key,
        overwrite,
        on_missing_rows,
        on_extra_rows,
        dry_run,
    ))
}

#[allow(clippy::too_many_arguments)]
fn scx_attach_obs_impl(
    path: &str,
    df: Robj,
    key: Strings,
    key_columns: Strings,
    key_column: Nullable<String>,
    prefix: &str,
    status_column: Nullable<String>,
    uns_key: Nullable<String>,
    overwrite: bool,
    on_missing_rows: &str,
    on_extra_rows: &str,
    dry_run: bool,
) -> Result<Robj> {
    use scx_ops::{
        attach_external_obs, build_composite_key, obs_key_values, resolve_obs_key_column,
        AttachObsOptions, ExternalObsData, ExtraRowPolicy, MissingRowPolicy, ObsJoinKey,
    };

    let missing = match on_missing_rows {
        // "null" is what actually happens here (Arrow nulls, so NA in R), and is
        // the default. "zero" stays accepted: it names the shared
        // `MissingRowPolicy` enum, whose `zero` is literal only on the layer /
        // CellBender path, where a missing *matrix* row really is zeros.
        "null" | "zero" => MissingRowPolicy::ZeroFill,
        "error" => MissingRowPolicy::Error,
        other => {
            return Err(Error::Other(format!(
                "on_missing_rows must be \"null\", \"zero\" or \"error\"; got \"{other}\""
            )))
        }
    };
    let extra = match on_extra_rows {
        "warn" => ExtraRowPolicy::WarnSkip,
        "error" => ExtraRowPolicy::Error,
        other => {
            return Err(Error::Other(format!(
                "on_extra_rows must be \"warn\" or \"error\"; got \"{other}\""
            )))
        }
    };

    let batch = crate::interop::dataframe_to_record_batch(&df)?;
    let n_rows = batch.num_rows();
    if n_rows == 0 {
        return Err(Error::Other(
            "the data.frame has no rows; there is nothing to attach".into(),
        ));
    }

    let explicit_keys: Vec<String> = key.iter().map(|s| s.to_string()).collect();
    let key_cols: Vec<String> = key_columns.iter().map(|s| s.to_string()).collect();
    if !explicit_keys.is_empty() && !key_cols.is_empty() {
        return Err(Error::Other(
            "pass either `key` (one value per row) or `key_columns` (columns of \
             `df` to fuse), not both"
                .into(),
        ));
    }

    // Both sides of the join are built by the same ops-crate helpers the CSV
    // reader uses, so an R-attached table and an imported CSV cannot disagree
    // about what a key is. `drop` names the columns of `df` consumed by the
    // key — decided here rather than re-derived later, so the two can never
    // disagree about which columns the annotations keep.
    let (row_keys, join_key, drop): (Vec<String>, ObsJoinKey, Vec<String>) =
        if !explicit_keys.is_empty() {
            if explicit_keys.len() != n_rows {
                return Err(Error::Other(format!(
                    "`key` has {} values but `df` has {n_rows} rows",
                    explicit_keys.len()
                )));
            }
            let target = match key_column {
                Nullable::NotNull(c) => ObsJoinKey::Column(c),
                Nullable::Null => ObsJoinKey::Auto,
            };
            // The keys came from outside `df`, so every column is an annotation.
            (explicit_keys, target, Vec::new())
        } else if key_cols.len() > 1 {
            for c in &key_cols {
                if batch.schema().field_with_name(c).is_err() {
                    return Err(Error::Other(format!(
                        "key column \"{c}\" is not in `df`; columns are {:?}",
                        column_names(&batch)
                    )));
                }
            }
            let fused = build_composite_key(&batch, &key_cols).map_err(to_r_err)?;
            (
                fused,
                ObsJoinKey::Composite {
                    columns: key_cols.clone(),
                },
                key_cols.clone(),
            )
        } else {
            // One named column of `df`, or auto-resolve one the same way the
            // delimited-table reader does.
            let col = if key_cols.len() == 1 {
                if batch.schema().field_with_name(&key_cols[0]).is_err() {
                    return Err(Error::Other(format!(
                        "key column \"{}\" is not in `df`; columns are {:?}",
                        key_cols[0],
                        column_names(&batch)
                    )));
                }
                key_cols[0].clone()
            } else {
                resolve_obs_key_column(&batch, None).map_err(to_r_err)?
            };
            let values = obs_key_values(&batch, &col).map_err(to_r_err)?;
            let drop = vec![col.clone()];
            (values, ObsJoinKey::Column(col), drop)
        };

    // The key columns are already on the target's obs axis — that is what the
    // join matched against — so re-importing them would only duplicate them.
    //
    // `__index_level_0__` goes with them: `dataframe_to_record_batch`
    // synthesises it from `rownames(df)`, which in the flagship recipe *is*
    // the key. Keeping it would collide with the target's own index column on
    // every attach, and it is the same rule the delimited reader follows for
    // the pandas index it renames.
    let mut drop = drop;
    drop.push("__index_level_0__".to_string());
    let annotations = drop_columns(&batch, drop, prefix)?;

    let data = ExternalObsData {
        row_keys,
        row_annotations: annotations,
        row_embeddings: Vec::new(),
        uns: None,
        source_checksum: None,
        source_name: Some("<R data.frame>".to_string()),
    };

    let opts = AttachObsOptions {
        join_key,
        missing_row_policy: missing,
        extra_row_policy: extra,
        status_column: match status_column {
            Nullable::NotNull(s) => Some(s),
            Nullable::Null => None,
        },
        uns_key: match uns_key {
            Nullable::NotNull(s) => Some(s),
            Nullable::Null => None,
        },
        overwrite,
        provenance_action: "scx_attach_obs".to_string(),
        dry_run,
        ..Default::default()
    };

    let s = attach_external_obs(Path::new(path), &data, &opts).map_err(to_r_err)?;

    Ok(list!(
        n_obs = s.n_obs as f64,
        n_matched = s.n_matched as f64,
        n_target_rows_absent = s.n_target_rows_absent as f64,
        n_source_rows_absent = s.n_source_rows_absent as f64,
        obs_key_column = s.obs_key_column,
        obs_columns_added = s.obs_columns_added,
        obsm_keys_added = s.obsm_keys_added,
        obs_index_dropped = s.obs_index_dropped,
        dry_run = dry_run
    )
    .into())
}

fn column_names(batch: &arrow::array::RecordBatch) -> Vec<String> {
    batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

/// `scx_ops::OpsError` carries the key diagnosis in its message, so surfacing
/// the string verbatim is what makes a duplicate-key failure actionable in R.
fn to_r_err(e: scx_ops::OpsError) -> Error {
    Error::Other(e.to_string())
}

/// Drop the join-key columns and apply `prefix` to what remains.
fn drop_columns(
    batch: &arrow::array::RecordBatch,
    drop: Vec<String>,
    prefix: &str,
) -> Result<arrow::array::RecordBatch> {
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    let schema = batch.schema();
    let mut fields = Vec::new();
    let mut arrays = Vec::new();
    for (i, f) in schema.fields().iter().enumerate() {
        if drop.iter().any(|d| d == f.name()) {
            continue;
        }
        // Nullable regardless: the attach op scatters nulls into every target
        // row this table does not cover.
        fields.push(Field::new(
            format!("{prefix}{}", f.name()),
            f.data_type().clone(),
            true,
        ));
        arrays.push(Arc::clone(batch.column(i)));
    }
    if fields.is_empty() {
        return Err(Error::Other(
            "no columns left to attach after removing the key column(s)".into(),
        ));
    }
    arrow::array::RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .map_err(|e| Error::Other(format!("failed to build annotation columns: {e}")))
}

// ---------------------------------------------------------------------------
// Module registration — used by lib.rs extendr_module! via `use ops;`
// ---------------------------------------------------------------------------

extendr_module! {
    mod ops;
    fn scx_append;
    fn scx_delete;
    fn scx_compact;
    fn scx_rollback;
    fn scx_merge;
    fn scx_info;
    fn scx_validate;
    fn scx_attach_obs;
}
