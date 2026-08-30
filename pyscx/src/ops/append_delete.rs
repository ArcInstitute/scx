//! Row-appending operations: `append`, `append_from_anndata`,
//! `mark_deleted`.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};

use scx_codec::ValueEncoding;
use scx_format_io::shard::{ShardHeader, SHARD_HEADER_SIZE};
use scx_format_io::ScxReader;

use super::*;
use crate::convert;

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
#[pyo3(signature = (
    target, input, codec=None, shard_size=None,
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None,
    modality=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn append(
    py: Python<'_>,
    target: &str,
    input: &str,
    codec: Option<&str>,
    shard_size: Option<i64>,
    index_obs: Option<Vec<String>>,
    index_var: Option<Vec<String>>,
    index_preset: Option<String>,
    index_auto_threshold: Option<usize>,
    modality: Option<&str>,
) -> PyResult<()> {
    let explicit_codec = convert::parse_codec_nonframed(codec)?;
    let shard_target_rows = validate_shard_size(shard_size)?;

    // Open input file
    let input_reader =
        ScxReader::open(input).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let target_reader =
        ScxReader::open(target).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Resolve modality routing (mirrors `scx append --modality`). The same
    // name addresses the target and the source modality.
    let target_modality_id = resolve_append_modality(&target_reader, "target", modality)?;
    let input_modality_id: u8 = if input_reader.is_multimodal() {
        if target_modality_id == 0 {
            return Err(PyValueError::new_err(
                "input file is multimodal but target is single-modality; \
                 subset the input to a single modality first",
            ));
        }
        // A multimodal target guarantees `modality` is Some here.
        let name = modality.expect("multimodal target requires modality");
        input_reader.modality_id(name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "input file has no modality named '{name}'; \
                 the input must expose the same modality as the target"
            ))
        })?
    } else {
        0
    };

    // Per-modality n_vars cross-check.
    let target_n_vars = modality_n_vars(&target_reader, target_modality_id);
    let input_n_vars = modality_n_vars(&input_reader, input_modality_id);
    if target_n_vars != input_n_vars {
        return Err(PyValueError::new_err(format!(
            "n_vars mismatch: target has {target_n_vars}, input has {input_n_vars}"
        )));
    }
    drop(target_reader);

    // Detect value encoding from the first source CSR shard header (for the
    // source modality) so we can resolve Scx1+float fallback before crossing
    // into Rust.
    let csr_entries = input_reader
        .catalog()
        .csr_shards_for_modality(input_modality_id);
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
        return Err(PyRuntimeError::new_err(format!(
            "input file has no CSR shards for modality {input_modality_id}"
        )));
    };

    // Resolve codec selection. None / "auto" → CodecSelection::Auto.
    // Explicit Scx1 with non-integer encoding falls back to Zstd
    // (Scx1 only encodes integers).
    let codec_selection = resolve_codec_selection(explicit_codec, value_encoding);

    let options = scx_ops::AppendOptions {
        codec: codec_selection,
        shard_target_rows,
        modality_id: target_modality_id,
    };

    let target_path = PathBuf::from(target);
    match build_index_options(index_obs, index_var, index_preset, index_auto_threshold) {
        Some(index_opts) => {
            let summary = py
                .detach(|| {
                    scx_ops::append_from_reader_with_index_options(
                        &target_path,
                        &input_reader,
                        &options,
                        input_modality_id,
                        &index_opts,
                    )
                })
                .map_err(ops_to_pyerr)?;
            process_index_summary(py, summary)?;
        }
        None => {
            py.detach(|| {
                scx_ops::append_from_reader(
                    &target_path,
                    &input_reader,
                    &options,
                    input_modality_id,
                )
            })
            .map_err(ops_to_pyerr)?;
        }
    }

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
#[pyo3(signature = (
    target, adata, codec=None, shard_size=None, in_place=false,
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None,
    modality=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn append_from_anndata(
    py: Python<'_>,
    target: &str,
    adata: &Bound<'_, PyAny>,
    codec: Option<&str>,
    shard_size: Option<i64>,
    in_place: bool,
    index_obs: Option<Vec<String>>,
    index_var: Option<Vec<String>>,
    index_preset: Option<String>,
    index_auto_threshold: Option<usize>,
    modality: Option<&str>,
) -> PyResult<()> {
    let explicit_codec = convert::parse_codec_nonframed(codec)?;
    let shard_target_rows = validate_shard_size(shard_size)?;

    // Extract CSR from adata.X
    let x = adata.getattr("X")?;
    let (x_csr, _csr_validated) = convert::ensure_csr(py, &x, in_place)?;

    // Get shape and validate n_vars match
    let shape: (u64, u64) = x_csr.getattr("shape")?.extract()?;
    let n_vars = shape.1;

    let target_reader =
        ScxReader::open(target).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    // Resolve the target modality (the AnnData source is a plain global axis).
    let target_modality_id = resolve_append_modality(&target_reader, "target", modality)?;
    let target_n_vars = modality_n_vars(&target_reader, target_modality_id);
    if target_n_vars != n_vars {
        return Err(PyValueError::new_err(format!(
            "n_vars mismatch: target has {target_n_vars}, AnnData has {n_vars}"
        )));
    }
    drop(target_reader);

    // Extract numpy arrays
    let np = crate::pyimport::import_module(py, "numpy")?;

    let indptr_obj = x_csr.getattr("indptr")?;
    let indptr_arr = convert::astype_if_needed(&indptr_obj, &np, "int64")?;
    let indptr_ro: PyReadonlyArray1<'_, i64> = indptr_arr.extract()?;
    let indptr_slice = indptr_ro
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let indices_obj = x_csr.getattr("indices")?;
    let indices_arr = convert::astype_if_needed(&indices_obj, &np, "int32")?;
    let indices_ro: PyReadonlyArray1<'_, i32> = indices_arr.extract()?;
    let indices_slice = indices_ro
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let data_obj = x_csr.getattr("data")?;
    let data_arr = convert::astype_if_needed(&data_obj, &np, "float32")?;
    let data_ro: PyReadonlyArray1<'_, f32> = data_arr.extract()?;
    let data_slice = data_ro
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Convert indptr/indices to on-disk types (finding 9.2: validate non-negative).
    let mut indptr: Vec<u64> = indptr_slice
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
    let mut indices: Vec<u32> = indices_slice
        .iter()
        .map(|&v| {
            if v < 0 {
                Err(PyRuntimeError::new_err(format!("negative CSR index {v}")))
            } else {
                Ok(v as u32)
            }
        })
        .collect::<PyResult<Vec<u32>>>()?;

    // Canonicalize the appended X before encoding so it upholds the v3
    // invariant (append preserves the base file's format_version, which may be
    // v3). Skip the f32 copy when the input is already canonical — the common
    // case. Detect encoding + encode from the (possibly canonicalized) values.
    let (value_encoding, values_bytes) =
        if scx_sparse::is_canonical_csr(&indptr, &indices, data_slice) {
            let ve = convert::detect_value_encoding(data_slice);
            let vb = convert::encode_values(data_slice, ve)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            (ve, vb)
        } else {
            let mut data_vec = data_slice.to_vec();
            scx_sparse::canonicalize_csr(&mut indptr, &mut indices, &mut data_vec);
            let ve = convert::detect_value_encoding(&data_vec);
            let vb = convert::encode_values(&data_vec, ve)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            (ve, vb)
        };

    // Read obs from AnnData
    let obs_df = adata.getattr("obs")?;
    let obs = convert::pandas_to_record_batch(py, &obs_df)?;

    // Resolve codec selection. None / "auto" → CodecSelection::Auto.
    // Explicit Scx1 with non-integer encoding falls back to Zstd
    // (Scx1 only encodes integers).
    let codec_selection = resolve_codec_selection(explicit_codec, value_encoding);

    let options = scx_ops::AppendOptions {
        codec: codec_selection,
        shard_target_rows,
        modality_id: target_modality_id,
    };

    let target_path = PathBuf::from(target);
    match build_index_options(index_obs, index_var, index_preset, index_auto_threshold) {
        Some(index_opts) => {
            let summary = py
                .detach(|| {
                    scx_ops::append_with_index_options(
                        &target_path,
                        &obs,
                        &indptr,
                        &indices,
                        &values_bytes,
                        value_encoding,
                        &options,
                        &index_opts,
                    )
                })
                .map_err(ops_to_pyerr)?;
            process_index_summary(py, summary)?;
        }
        None => {
            py.detach(|| {
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
        }
    }

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
