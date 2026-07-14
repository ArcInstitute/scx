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
/// Note: indices are i32 from R (no unsigned int), converted to u64 internally.
/// Returns the total number of deleted cells (including previously deleted).
///
/// @param path Path to the SCX file.
/// @param cell_indices Integer vector of 0-based cell indices to delete.
/// @return Total number of deleted cells as numeric (f64 to avoid i32 overflow).
///
/// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3).
#[extendr]
fn scx_delete(path: &str, cell_indices: Vec<i32>) -> Robj {
    crate::util::throw_on_err(scx_delete_impl(path, cell_indices))
}

fn scx_delete_impl(path: &str, cell_indices: Vec<i32>) -> Result<Robj> {
    let indices: Vec<u64> = cell_indices
        .into_iter()
        .map(|i| {
            if i < 0 {
                Err(Error::Other(format!("negative cell index: {}", i)))
            } else {
                Ok(i as u64)
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
}
