// Phase E: File Operations R bindings
//
// Wraps scx-ops (append, mark_deleted, compact, rollback, merge) for R,
// plus scx_info and scx_validate from scx-format.
// Follows the same pattern as pyscx/src/ops.rs, adapted for extendr.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use extendr_api::prelude::*;

use scx_codec::ValueEncoding;
use scx_format::section::SectionType;
use scx_format::shard::{ShardHeader, SHARD_HEADER_SIZE};
use scx_format::{select_codec, ScxReader};

// ---------------------------------------------------------------------------
// Append
// ---------------------------------------------------------------------------

/// Append cells from one SCX file to another.
///
/// Reads the input file's CSR data and obs metadata, then appends them
/// to the target file using `scx_ops::append()`.
///
/// @param target Path to the target SCX file (will be modified in-place).
/// @param input Path to the input SCX file to append from.
#[extendr]
fn scx_append(target: &str, input: &str) -> Result<()> {
    // Open input file
    let input_reader =
        ScxReader::open(input).map_err(|e| Error::Other(format!("failed to open input: {e}")))?;

    // Validate n_vars match
    let target_reader = ScxReader::open(target)
        .map_err(|e| Error::Other(format!("failed to open target: {e}")))?;
    if target_reader.n_vars() != input_reader.n_vars() {
        return Err(Error::Other(format!(
            "n_vars mismatch: target has {}, input has {}",
            target_reader.n_vars(),
            input_reader.n_vars()
        )));
    }
    drop(target_reader);

    // Read CSR data
    let csr = input_reader
        .read_all_csr_shards()
        .map_err(|e| Error::Other(format!("failed to read CSR shards: {e}")))?;

    // Detect value encoding from first shard header
    let csr_entries = input_reader.catalog().shards(SectionType::CsrShard);
    let value_encoding = if let Some(first_entry) = csr_entries.first() {
        let bytes = input_reader
            .section_bytes(first_entry)
            .map_err(|e| Error::Other(e.to_string()))?;
        let sh = ShardHeader::read_from(&mut Cursor::new(&bytes[..SHARD_HEADER_SIZE]))
            .map_err(|e| Error::Other(e.to_string()))?;
        ValueEncoding::from_u8(sh.value_encoding)
            .ok_or_else(|| Error::Other(format!("unknown value encoding: {}", sh.value_encoding)))?
    } else {
        return Err(Error::Other("input file has no CSR shards".into()));
    };

    // Convert i64 → u64 indptr (validate non-negative)
    let indptr: Vec<u64> = csr
        .indptr
        .iter()
        .map(|&v| {
            if v < 0 {
                Err(Error::Other(format!("negative indptr value {v}")))
            } else {
                Ok(v as u64)
            }
        })
        .collect::<Result<Vec<u64>>>()?;

    // Convert i32 → u32 indices (validate non-negative)
    let indices: Vec<u32> = csr
        .indices
        .iter()
        .map(|&v| {
            if v < 0 {
                Err(Error::Other(format!("negative CSR index {v}")))
            } else {
                Ok(v as u32)
            }
        })
        .collect::<Result<Vec<u32>>>()?;

    // Encode f32 → raw LE bytes
    let values_bytes = encode_values(&csr.data, value_encoding);

    // Read obs metadata
    let obs = input_reader
        .read_obs()
        .map_err(|e| Error::Other(format!("failed to read obs: {e}")))?;

    // Auto-select codec
    let effective_codec = select_codec(&values_bytes, value_encoding);

    let target_path = PathBuf::from(target);
    scx_ops::append(
        &target_path,
        &obs,
        &indptr,
        &indices,
        &values_bytes,
        value_encoding,
        effective_codec,
        16384, // default shard target rows
    )
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
/// @return Total number of deleted cells as integer.
#[extendr]
fn scx_delete(path: &str, cell_indices: Vec<i32>) -> Result<i32> {
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
    Ok(total as i32)
}

// ---------------------------------------------------------------------------
// Compact
// ---------------------------------------------------------------------------

/// Rewrite an SCX file reclaiming deleted/orphaned space.
///
/// @param input Path to the input SCX file.
/// @param output Path for the compacted output file.
#[extendr]
fn scx_compact(input: &str, output: &str) -> Result<()> {
    scx_ops::compact(Path::new(input), Path::new(output))
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
#[extendr]
fn scx_rollback(path: &str, to_seq: Nullable<i32>) -> Result<()> {
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

/// Merge multiple SCX files into one.
///
/// Requires at least 2 input files. All must have the same n_vars.
///
/// @param inputs Character vector of input file paths.
/// @param output Path for the merged output file.
#[extendr]
fn scx_merge(inputs: Vec<String>, output: &str) -> Result<()> {
    if inputs.len() < 2 {
        return Err(Error::Other(
            "merge requires at least 2 input files".into(),
        ));
    }
    let input_paths: Vec<PathBuf> = inputs.iter().map(PathBuf::from).collect();
    let input_refs: Vec<&Path> = input_paths.iter().map(|p| p.as_path()).collect();

    scx_ops::merge(&input_refs, Path::new(output)).map_err(|e| Error::Other(e.to_string()))
}

// ---------------------------------------------------------------------------
// Info
// ---------------------------------------------------------------------------

/// Return file information as a named list.
///
/// @param path Path to the SCX file.
/// @return Named list with n_obs, n_vars, nnz, format_version, n_shards.
#[extendr]
fn scx_info(path: &str) -> Result<Robj> {
    let reader =
        ScxReader::open(path).map_err(|e| Error::Other(format!("failed to open: {e}")))?;
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
#[extendr]
fn scx_validate(path: &str) -> Result<bool> {
    let reader =
        ScxReader::open(path).map_err(|e| Error::Other(format!("failed to open: {e}")))?;
    let checks = reader
        .validate()
        .map_err(|e| Error::Other(e.to_string()))?;
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
// Helper: encode f32 values to raw LE bytes
// ---------------------------------------------------------------------------

/// Encode f32 values to raw little-endian bytes according to value encoding.
/// Same logic as pyscx::anndata::encode_values.
fn encode_values(data: &[f32], encoding: ValueEncoding) -> Vec<u8> {
    match encoding {
        ValueEncoding::Uint8 => data.iter().map(|&v| v as u8).collect(),
        ValueEncoding::Uint16 => data
            .iter()
            .flat_map(|&v| (v as u16).to_le_bytes())
            .collect(),
        ValueEncoding::Uint32 => data
            .iter()
            .flat_map(|&v| (v as u32).to_le_bytes())
            .collect(),
        ValueEncoding::Float32 => data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        ValueEncoding::Float16 => data
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect(),
    }
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
