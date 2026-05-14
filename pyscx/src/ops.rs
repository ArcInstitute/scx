// File operations Python bindings
//
// Wraps scx-ops (append, mark_deleted, compact, rollback, merge) for Python.

use std::io::Cursor;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use scx_codec::{CodecId, CodecSelection, ValueEncoding};
use scx_format::section::SectionType;
use scx_format::shard::{ShardHeader, SHARD_HEADER_SIZE};
use scx_format::ScxReader;

use scx_ops::OpsError;

use crate::anndata;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const DEFAULT_SHARD_SIZE: i64 = scx_format::DEFAULT_SHARD_TARGET_ROWS as i64;

/// Validate the Python-side `shard_size` kwarg and convert to NonZeroU32.
///
/// Accepts a signed i64 (rather than u32) so that negative values from
/// Python raise `ValueError` here instead of `OverflowError` during
/// pyo3 argument extraction. Per review P0 #9, all `shard_size <= 0`
/// inputs are rejected with `ValueError` before crossing into Rust.
fn validate_shard_size(shard_size: Option<i64>) -> PyResult<NonZeroU32> {
    let v = shard_size.unwrap_or(DEFAULT_SHARD_SIZE);
    if v <= 0 {
        return Err(PyValueError::new_err(format!(
            "shard_size must be > 0 (got {v})"
        )));
    }
    let v_u32: u32 = v.try_into().map_err(|_| {
        PyValueError::new_err(format!("shard_size {v} exceeds u32::MAX ({})", u32::MAX))
    })?;
    NonZeroU32::new(v_u32).ok_or_else(|| PyValueError::new_err("shard_size must be > 0"))
}

/// Map an Option<CodecId> (from anndata::parse_codec) plus the detected
/// value encoding into a CodecSelection. Preserves the legacy
/// Scx1+float silent fixup at the binding boundary.
fn resolve_codec_selection(
    explicit_codec: Option<CodecId>,
    value_encoding: ValueEncoding,
) -> CodecSelection {
    match explicit_codec {
        None => CodecSelection::Auto,
        Some(CodecId::Scx1) if !value_encoding.is_integer() => {
            CodecSelection::Explicit(CodecId::Zstd)
        }
        Some(c) => CodecSelection::Explicit(c),
    }
}

// ---------------------------------------------------------------------------
// Error conversion
// ---------------------------------------------------------------------------

/// Convert an OpsError to the most appropriate Python exception.
///
/// User-input validation errors → ValueError; missing files → FileNotFoundError;
/// permission errors → PermissionError; wrapped ScxError → delegates to
/// `crate::to_pyerr`; everything else → RuntimeError.
fn ops_to_pyerr(e: OpsError) -> PyErr {
    use pyo3::exceptions::{PyFileNotFoundError, PyPermissionError};
    let msg = e.to_string();
    match e {
        OpsError::Format(inner) => crate::to_pyerr(inner),
        OpsError::IncompatibleVars { .. }
        | OpsError::SchemaMismatch { .. }
        | OpsError::ShapeMismatch { .. }
        | OpsError::VarLengthMismatch { .. }
        | OpsError::IndexOutOfBounds { .. }
        | OpsError::ValueOutOfRange { .. }
        | OpsError::UnknownCodec(_)
        | OpsError::UnknownValueEncoding(_) => PyValueError::new_err(msg),
        OpsError::Io(ref io_err) if io_err.kind() == std::io::ErrorKind::NotFound => {
            PyFileNotFoundError::new_err(msg)
        }
        OpsError::Io(ref io_err) if io_err.kind() == std::io::ErrorKind::PermissionDenied => {
            PyPermissionError::new_err(msg)
        }
        _ => PyRuntimeError::new_err(msg),
    }
}

// ---------------------------------------------------------------------------
// C1. pyscx.append()
// ---------------------------------------------------------------------------

/// Append cells from one SCX file to another.
///
/// Returns None. Raises RuntimeError on failure.
///
/// Example:
///     pyscx.append("atlas.scx", "new_batch.scx")
///     pyscx.append("atlas.scx", "new_batch.scx", codec="auto", shard_size=10000)
#[pyfunction]
#[pyo3(signature = (target, input, codec=None, shard_size=None))]
pub fn append(
    py: Python<'_>,
    target: &str,
    input: &str,
    codec: Option<&str>,
    shard_size: Option<i64>,
) -> PyResult<()> {
    let explicit_codec = anndata::parse_codec(codec)?;
    let shard_target_rows = validate_shard_size(shard_size)?;

    // Open input file
    let input_reader =
        ScxReader::open(input).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Validate n_vars match
    let target_reader =
        ScxReader::open(target).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if target_reader.n_vars() != input_reader.n_vars() {
        return Err(PyValueError::new_err(format!(
            "n_vars mismatch: target has {}, input has {}",
            target_reader.n_vars(),
            input_reader.n_vars()
        )));
    }
    drop(target_reader);

    // Detect value encoding from the first source CSR shard header so we
    // can resolve Scx1+float fallback before crossing into Rust.
    let csr_entries = input_reader.catalog().shards(SectionType::CsrShard);
    let value_encoding = if let Some(first_entry) = csr_entries.first() {
        let bytes = input_reader
            .section_bytes(first_entry)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let sh = ShardHeader::read_from(&mut Cursor::new(&bytes[..SHARD_HEADER_SIZE]))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        ValueEncoding::from_u8(sh.value_encoding).ok_or_else(|| {
            PyRuntimeError::new_err(format!("unknown value encoding: {}", sh.value_encoding))
        })?
    } else {
        return Err(PyRuntimeError::new_err("input file has no CSR shards"));
    };

    // Resolve codec selection. None / "auto" → CodecSelection::Auto.
    // Explicit Scx1 with non-integer encoding falls back to Zstd
    // (Scx1 only encodes integers).
    let codec_selection = resolve_codec_selection(explicit_codec, value_encoding);

    let options = scx_ops::AppendOptions {
        codec: codec_selection,
        shard_target_rows,
        modality_id: 0,
    };

    let target_path = PathBuf::from(target);
    py.allow_threads(|| scx_ops::append_from_reader(&target_path, &input_reader, &options, 0))
        .map_err(ops_to_pyerr)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// C2. pyscx.append_from_anndata()
// ---------------------------------------------------------------------------

/// Append cells from a Python AnnData object to an existing SCX file.
///
/// By default `adata.X` is not mutated; CSR inputs with unsorted indices
/// are copied. Pass `in_place=True` to allow `ensure_csr()` to sort the
/// caller's CSR in place (one fewer allocation; mutates user input).
///
/// Example:
///     pyscx.append_from_anndata("atlas.scx", new_adata)
///     pyscx.append_from_anndata("atlas.scx", new_adata, codec="auto", shard_size=10000)
#[pyfunction]
#[pyo3(signature = (target, adata, codec=None, shard_size=None, in_place=false))]
pub fn append_from_anndata(
    py: Python<'_>,
    target: &str,
    adata: &Bound<'_, PyAny>,
    codec: Option<&str>,
    shard_size: Option<i64>,
    in_place: bool,
) -> PyResult<()> {
    let explicit_codec = anndata::parse_codec(codec)?;
    let shard_target_rows = validate_shard_size(shard_size)?;

    // Extract CSR from adata.X
    let x = adata.getattr("X")?;
    let (x_csr, _csr_validated) = anndata::ensure_csr(py, &x, in_place)?;

    // Get shape and validate n_vars match
    let shape: (u64, u64) = x_csr.getattr("shape")?.extract()?;
    let n_vars = shape.1;

    let target_reader =
        ScxReader::open(target).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if target_reader.n_vars() != n_vars {
        return Err(PyValueError::new_err(format!(
            "n_vars mismatch: target has {}, AnnData has {}",
            target_reader.n_vars(),
            n_vars
        )));
    }
    drop(target_reader);

    // Extract numpy arrays
    let np = py.import("numpy")?;

    let indptr_obj = x_csr.getattr("indptr")?;
    let indptr_arr = anndata::astype_if_needed(&indptr_obj, &np, "int64")?;
    let indptr_ro: PyReadonlyArray1<'_, i64> = indptr_arr.extract()?;
    let indptr_slice = indptr_ro
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let indices_obj = x_csr.getattr("indices")?;
    let indices_arr = anndata::astype_if_needed(&indices_obj, &np, "int32")?;
    let indices_ro: PyReadonlyArray1<'_, i32> = indices_arr.extract()?;
    let indices_slice = indices_ro
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let data_obj = x_csr.getattr("data")?;
    let data_arr = anndata::astype_if_needed(&data_obj, &np, "float32")?;
    let data_ro: PyReadonlyArray1<'_, f32> = data_arr.extract()?;
    let data_slice = data_ro
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Detect encoding and encode values
    let value_encoding = anndata::detect_value_encoding(data_slice);
    let values_bytes = anndata::encode_values(data_slice, value_encoding)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Convert indptr/indices to on-disk types (finding 9.2: validate non-negative).
    let indptr: Vec<u64> = indptr_slice
        .iter()
        .map(|&v| {
            if v < 0 {
                Err(PyRuntimeError::new_err(format!(
                    "negative indptr value {v}"
                )))
            } else {
                Ok(v as u64)
            }
        })
        .collect::<PyResult<Vec<u64>>>()?;
    let indices: Vec<u32> = indices_slice
        .iter()
        .map(|&v| {
            if v < 0 {
                Err(PyRuntimeError::new_err(format!("negative CSR index {v}")))
            } else {
                Ok(v as u32)
            }
        })
        .collect::<PyResult<Vec<u32>>>()?;

    // Read obs from AnnData
    let obs_df = adata.getattr("obs")?;
    let obs = anndata::pandas_to_record_batch(py, &obs_df)?;

    // Resolve codec selection. None / "auto" → CodecSelection::Auto.
    // Explicit Scx1 with non-integer encoding falls back to Zstd
    // (Scx1 only encodes integers).
    let codec_selection = resolve_codec_selection(explicit_codec, value_encoding);

    let options = scx_ops::AppendOptions {
        codec: codec_selection,
        shard_target_rows,
        modality_id: 0,
    };

    let target_path = PathBuf::from(target);
    py.allow_threads(|| {
        scx_ops::append(
            &target_path,
            &obs,
            &indptr,
            &indices,
            &values_bytes,
            value_encoding,
            &options,
        )
    })
    .map_err(ops_to_pyerr)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// C3. pyscx.mark_deleted()
// ---------------------------------------------------------------------------

/// Mark specific cell indices as logically deleted.
///
/// Returns the total number of deleted cells (including previously deleted).
///
/// Example:
///     total = pyscx.mark_deleted("experiment.scx", [0, 5, 10, 42])
#[pyfunction]
pub fn mark_deleted(path: &str, cell_indices: Vec<i64>) -> PyResult<u64> {
    // Convert i64 → u64 with overflow check
    let indices: Vec<u64> = cell_indices
        .iter()
        .map(|&v| {
            if v < 0 {
                Err(PyValueError::new_err(format!(
                    "cell index {} is negative",
                    v
                )))
            } else {
                Ok(v as u64)
            }
        })
        .collect::<PyResult<Vec<u64>>>()?;

    let p = Path::new(path);
    scx_ops::mark_deleted(p, &indices).map_err(ops_to_pyerr)
}

// ---------------------------------------------------------------------------
// C5. pyscx.compact()
// ---------------------------------------------------------------------------

/// Rewrite an SCX file reclaiming space from deleted and orphaned sections.
///
/// Example:
///     pyscx.compact("experiment.scx", "compacted.scx")
#[pyfunction]
pub fn compact(py: Python<'_>, input: &str, output: &str) -> PyResult<()> {
    let input_path = PathBuf::from(input);
    let output_path = PathBuf::from(output);
    py.allow_threads(|| scx_ops::compact(&input_path, &output_path))
        .map_err(ops_to_pyerr)
}

// ---------------------------------------------------------------------------
// C6. pyscx.rollback()
// ---------------------------------------------------------------------------

/// Roll back an SCX file to a previous manifest version.
///
/// If `to_seq` is None, rolls back one version.
/// If `to_seq` is given, rolls back to that specific sequence number.
///
/// Example:
///     pyscx.rollback("experiment.scx")           # roll back one version
///     pyscx.rollback("experiment.scx", to_seq=3)  # roll back to specific version
#[pyfunction]
#[pyo3(signature = (path, to_seq=None))]
pub fn rollback(path: &str, to_seq: Option<u64>) -> PyResult<()> {
    let p = Path::new(path);
    match to_seq {
        Some(seq) => scx_ops::rollback_to(p, seq).map_err(ops_to_pyerr),
        None => scx_ops::rollback(p).map_err(ops_to_pyerr),
    }
}

// ---------------------------------------------------------------------------
// C7. pyscx.merge()
// ---------------------------------------------------------------------------

/// Merge multiple SCX files into a single output file.
///
/// Requires at least 2 input files. All must have the same n_vars.
///
/// Example:
///     pyscx.merge(["batch1.scx", "batch2.scx", "batch3.scx"], "atlas.scx")
#[pyfunction]
pub fn merge(py: Python<'_>, inputs: Vec<String>, output: &str) -> PyResult<()> {
    if inputs.len() < 2 {
        return Err(PyValueError::new_err(
            "merge requires at least 2 input files",
        ));
    }

    let input_paths: Vec<PathBuf> = inputs.iter().map(PathBuf::from).collect();
    let input_refs: Vec<&Path> = input_paths.iter().map(|p| p.as_path()).collect();
    let output_path = PathBuf::from(output);

    py.allow_threads(|| scx_ops::merge(&input_refs, &output_path))
        .map_err(ops_to_pyerr)
}
