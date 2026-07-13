// File operations Python bindings
//
// Wraps scx-ops (append, mark_deleted, compact, rollback, merge) for Python.

use std::io::Cursor;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use arrow::record_batch::RecordBatch;

use scx_codec::{CodecId, CodecSelection, ValueEncoding};
use scx_engine::{BuildOutcome, ConversionPredicateIndexOptions, SkipReason};
use scx_format_io::shard::{ShardHeader, SHARD_HEADER_SIZE};
use scx_format_io::ScxReader;

use scx_ops::{OpsError, PredicateIndexBuildSummary, ReferenceSpec};

use crate::convert;

// ---------------------------------------------------------------------------
// Predicate-index kwarg helpers
// ---------------------------------------------------------------------------

/// Build a `ConversionPredicateIndexOptions` from the four pyscx kwargs
/// when at least one was supplied. Returns `None` when every kwarg is
/// `None`, signalling the caller should fall back to the legacy entry
/// point that emits no predicate index (pre-fix default behaviour).
///
/// `index_auto_threshold` reaching the engine ALONE (without the other
/// kwargs) is the recently-fixed bug — the caller previously dropped it
/// silently. Now any non-None kwarg opts the user into the engine path.
fn build_index_options(
    index_obs: Option<Vec<String>>,
    index_var: Option<Vec<String>>,
    index_preset: Option<String>,
    index_auto_threshold: Option<usize>,
) -> Option<ConversionPredicateIndexOptions> {
    let any_set = index_obs.is_some()
        || index_var.is_some()
        || index_preset.is_some()
        || index_auto_threshold.is_some();
    if !any_set {
        return None;
    }
    Some(ConversionPredicateIndexOptions {
        index_obs: index_obs.unwrap_or_default(),
        index_var: index_var.unwrap_or_default(),
        index_preset,
        // 1000 mirrors `scx-convert::pipeline::ConvertOptions::default`.
        index_auto_threshold: index_auto_threshold.unwrap_or(1000),
    })
}

/// Map a `PredicateIndexBuildSummary` from a `scx-ops` rewrite back
/// into pyscx-visible signals: forced-column errors become
/// `PyValueError`; preset skips become `warnings.warn(...)`; the
/// multimodal skip becomes a single `warnings.warn(...)`. Mirrors the
/// outcome handling in
/// `pyscx::convert::build_and_write_predicate_indexes_inline` so the
/// rewrite ops surface the same errors / warnings as `from_anndata`.
fn process_index_summary(py: Python<'_>, summary: PredicateIndexBuildSummary) -> PyResult<()> {
    let emit_warning = |msg: String| -> PyResult<()> {
        py.import("warnings")?.call_method1("warn", (msg,))?;
        Ok(())
    };

    if let Some(columns) = summary.multimodal_skip {
        emit_warning(format!(
            "predicate index skipped: target is multimodal but the engine \
             read-side is unimodal-only (requested columns: {columns:?})"
        ))?;
        return Ok(());
    }

    let Some(result) = summary.result else {
        return Ok(());
    };

    let process = |outcomes: Vec<BuildOutcome>, axis: &str| -> PyResult<()> {
        let mut forced_missing: Vec<String> = Vec::new();
        for outcome in outcomes {
            match outcome {
                BuildOutcome::ForcedColumnError { column, reason } => {
                    if matches!(reason, SkipReason::MissingColumn) {
                        forced_missing.push(column);
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "forced {axis} index column '{column}': {reason}"
                        )));
                    }
                }
                BuildOutcome::PresetSkipped { column, reason } => {
                    emit_warning(format!(
                        "predicate index skipped for {axis} column '{column}': {reason}"
                    ))?;
                }
            }
        }
        if !forced_missing.is_empty() {
            // No `available_columns` slice handy at this layer —
            // surface the missing list directly; the engine's
            // `forced_columns_missing_message` is only used by the
            // convert layer where the available columns slice is
            // cheap to compute. Sub-tier UX, but no surprises.
            return Err(PyValueError::new_err(format!(
                "forced {axis} index columns missing from \
                     output: {forced_missing:?}"
            )));
        }
        Ok(())
    };
    process(result.obs_outcomes, "obs")?;
    process(result.var_outcomes, "var")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const DEFAULT_SHARD_SIZE: i64 = scx_format_io::DEFAULT_SHARD_TARGET_ROWS as i64;

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

/// Map an Option<CodecId> (from convert::parse_codec_nonframed) plus the detected
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
        | OpsError::CellIndexOutOfBounds { .. }
        | OpsError::ValueOutOfRange { .. }
        | OpsError::UnknownCodec(_)
        | OpsError::UnknownValueEncoding(_)
        | OpsError::InvalidInput(_) => PyValueError::new_err(msg),
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
    let np = py.import("numpy")?;

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

// ---------------------------------------------------------------------------
// C5. pyscx.compact()
// ---------------------------------------------------------------------------

/// Rewrite an SCX file reclaiming space from deleted and orphaned sections.
///
/// Optionally rebuilds predicate indexes on the compacted output: pass
/// `index_obs=[...]` / `index_var=[...]` / `index_preset=...` to mirror
/// the `scx convert` surface. Without these kwargs the predicate-index
/// sections are dropped as before (the row layout is re-sharded against
/// the post-deletion row count).
///
/// `reshape_obs`: when True, migrate legacy single-section obs metadata
/// to the atlas-scale sharded `ObsMetadataShard` layout. Mirrors
/// `scx compact --reshape-obs`; useful after a backed `from_anndata`
/// conversion (which always writes single-section metadata regardless of
/// `n_obs`). Composes with the `index_*` kwargs.
///
/// Example:
///     pyscx.compact("experiment.scx", "compacted.scx")
///     pyscx.compact("experiment.scx", "compacted.scx",
///                   index_obs=["perturbation", "cell_type"])
///     pyscx.compact("experiment.scx", "compacted.scx", reshape_obs=True)
#[pyfunction]
#[pyo3(signature = (
    input, output,
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None,
    reshape_obs=false,
))]
#[allow(clippy::too_many_arguments)]
pub fn compact(
    py: Python<'_>,
    input: &str,
    output: &str,
    index_obs: Option<Vec<String>>,
    index_var: Option<Vec<String>>,
    index_preset: Option<String>,
    index_auto_threshold: Option<usize>,
    reshape_obs: bool,
) -> PyResult<()> {
    let input_path = PathBuf::from(input);
    let output_path = PathBuf::from(output);
    match build_index_options(index_obs, index_var, index_preset, index_auto_threshold) {
        Some(index_opts) => {
            let summary = py
                .detach(|| {
                    scx_ops::compact_with_index_options(
                        &input_path,
                        &output_path,
                        &index_opts,
                        reshape_obs,
                    )
                })
                .map_err(ops_to_pyerr)?;
            process_index_summary(py, summary)
        }
        // `reshape_obs` must reach the index-options path even with no
        // `index_*` kwarg set. The `index_auto_threshold = 0` sentinel
        // keeps `user_wants_index()` false (no index built), matching
        // bare `compact()`, while still migrating obs to sharded
        // sections. Mirrors `scx-cli/src/compact.rs`.
        None if reshape_obs => {
            let sentinel = ConversionPredicateIndexOptions {
                index_obs: Vec::new(),
                index_var: Vec::new(),
                index_preset: None,
                index_auto_threshold: 0,
            };
            let summary = py
                .detach(|| {
                    scx_ops::compact_with_index_options(&input_path, &output_path, &sentinel, true)
                })
                .map_err(ops_to_pyerr)?;
            process_index_summary(py, summary)
        }
        None => py
            .detach(|| scx_ops::compact(&input_path, &output_path))
            .map_err(ops_to_pyerr),
    }
}

// ---------------------------------------------------------------------------
// C5b. pyscx.optimize()
// ---------------------------------------------------------------------------

/// Re-encode + canonicalize CSR shards to add decode sidecars and upgrade
/// to format_version 3.
///
/// This is the Python equivalent of ``scx optimize``. It re-encodes every
/// CSR shard (X, layers, obsp graphs) so the output carries decode sidecars
/// (enabling ``to_gpu_anndata`` device-decode) and stamps ``format_version = 3``
/// (canonical-CSR invariant). Single-modality files only; use ``compact()``
/// for multimodal.
///
/// Args:
///     input:  Path to the source ``.scx`` file.
///     output: Path for the optimized file. May equal ``input`` for in-place
///             upgrade (writes to a sibling tempfile, then atomically renames).
///     codec:  Per-shard codec selection.
///
///             - ``"auto"`` (default): Scx1 for low-median integer counts,
///               Zstd otherwise. Some high-median shards will NOT get a
///               decode sidecar under auto.
///             - ``"scx1"``: Force Scx1 on every integer shard — guarantees
///               a decode sidecar on all integer shards (needed for full
///               ``to_gpu_anndata`` device-decode coverage).
///     shard_obs: Migrate a legacy single-section obs table to the sharded
///             ``ObsMetadataShard`` layout.
///
///             - ``"auto"`` (default): shard only when ``n_obs >
///               shard_target_rows`` (the ``from_anndata`` threshold) — small
///               files stay single-section, atlas-scale files get sharded obs.
///             - ``"always"``: always shard a single-section obs.
///             - ``"off"``: keep the single section (faithful 1:1 copy).
///
///             An already-sharded obs is preserved as shards regardless of
///             this setting.
///
/// Raises:
///     ValueError: If `codec` is not "auto"/"scx1" or `shard_obs` is not
///         "off"/"auto"/"always".
///     RuntimeError: If the file is multimodal, the input doesn't exist,
///         or the output already exists (no ``--force`` analogue; callers
///         should remove the target first or use ``output == input``).
///
/// Example:
///     pyscx.optimize("experiment.scx", "optimized.scx")
///     pyscx.optimize("experiment.scx", "experiment.scx")  # in-place
///     pyscx.optimize("experiment.scx", "optimized.scx", codec="scx1")
///     pyscx.optimize("atlas.scx", "atlas.opt.scx", shard_obs="always")
#[pyfunction]
#[pyo3(signature = (input, output, codec="auto", shard_obs="auto"))]
pub fn optimize(
    py: Python<'_>,
    input: &str,
    output: &str,
    codec: &str,
    shard_obs: &str,
) -> PyResult<()> {
    let input_path = PathBuf::from(input);
    let output_path = PathBuf::from(output);
    let codec_id = match codec {
        "auto" => None,
        "scx1" => Some(CodecId::Scx1),
        other => {
            return Err(PyValueError::new_err(format!(
                "codec must be 'auto' or 'scx1', got {other:?} \
                 (other codecs drop decode sidecars, defeating the purpose of optimize)"
            )));
        }
    };
    let obs_shard_policy =
        scx_format_io::ObsShardPolicy::parse(shard_obs).map_err(PyValueError::new_err)?;
    // No-clobber guard mirroring `scx optimize` (no `--force` analogue here).
    // An in-place upgrade (`output == input`) writes a sibling tempfile and
    // atomically renames, so only a *different* pre-existing output is
    // rejected. Compare canonicalized paths when both resolve, falling back to
    // a literal compare for a not-yet-created output.
    let same_file = match (
        std::fs::canonicalize(&input_path),
        std::fs::canonicalize(&output_path),
    ) {
        (Ok(a), Ok(b)) => a == b,
        _ => input_path == output_path,
    };
    if output_path.exists() && !same_file {
        return Err(PyRuntimeError::new_err(format!(
            "output file already exists: {} (no force analogue; remove the \
             target first or pass output == input for an in-place upgrade)",
            output_path.display()
        )));
    }
    py.detach(|| scx_ops::optimize(&input_path, &output_path, codec_id, obs_shard_policy))
        .map_err(ops_to_pyerr)
}

/// Globally reorder cells (the obs axis) of an SCX file by an obs key,
/// writing a new file with X-read locality and contiguous predicate-index
/// shard ranges for the sort key.
///
/// `by`: one or more obs columns, lexicographic in order (the leading key
/// gets the full X-read-locality benefit). `reverse`: descending on all keys.
/// Pass `memory_budget` (e.g. "4G") to force the bounded external partition
/// sort; without it the in-memory path is used. The detection bitmap and CSC
/// sidecar are dropped (the reorder invalidates them); pass `rebuild_csc=True`
/// to re-emit the column-major sidecar. obsp/multimodal sort are not yet
/// supported.
///
/// Example:
///     pyscx.sort("atlas.scx", "atlas.sorted.scx", by=["cell_type"])
/// Parse the `reference` kwarg of `sort` into a [`ReferenceSpec`].
///
/// Accepts `None`, a `str` (single label), a `list[str]` (label set), or a
/// dict `{"column": name}` (boolean obs column). Mirrors the CLI's
/// label-list / `col:NAME` forms. `str` is checked before `list[str]` because
/// pyo3 would otherwise iterate a string into single-character labels.
pub(crate) fn parse_reference_spec(
    v: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<ReferenceSpec>> {
    let Some(obj) = v else { return Ok(None) };
    if obj.is_none() {
        return Ok(None);
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        return match d.get_item("column")? {
            Some(col) => Ok(Some(ReferenceSpec::Column(col.extract::<String>()?))),
            None => Err(PyValueError::new_err(
                "reference dict must have a 'column' key, e.g. {'column': 'is_control'}",
            )),
        };
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Some(ReferenceSpec::Labels(vec![s])));
    }
    if let Ok(labels) = obj.extract::<Vec<String>>() {
        return Ok(Some(ReferenceSpec::Labels(labels)));
    }
    Err(PyValueError::new_err(
        "reference must be None, a str, a list[str], or {'column': name}",
    ))
}

#[pyfunction]
#[pyo3(signature = (
    input, output, by, reverse=false, shard_size=None,
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None,
    memory_budget=None, temp_dir=None, bitmap="off".to_string(), rebuild_csc=false,
    csc_cols_per_shard=5000, csc_memory_limit="4G".to_string(),
    group_by=None, reference=None, group_target_bytes=None, group_max_bytes=None,
    group_write_block_bytes=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn sort(
    py: Python<'_>,
    input: &str,
    output: &str,
    by: Vec<String>,
    reverse: bool,
    shard_size: Option<i64>,
    index_obs: Option<Vec<String>>,
    index_var: Option<Vec<String>>,
    index_preset: Option<String>,
    index_auto_threshold: Option<usize>,
    memory_budget: Option<String>,
    temp_dir: Option<String>,
    bitmap: String,
    rebuild_csc: bool,
    csc_cols_per_shard: usize,
    csc_memory_limit: String,
    group_by: Option<String>,
    reference: Option<Bound<'_, PyAny>>,
    group_target_bytes: Option<Bound<'_, PyAny>>,
    group_max_bytes: Option<Bound<'_, PyAny>>,
    group_write_block_bytes: Option<Bound<'_, PyAny>>,
) -> PyResult<()> {
    if by.is_empty() && group_by.is_none() {
        return Err(PyValueError::new_err(
            "sort requires at least one `by` column (or `group_by`)",
        ));
    }
    let input_path = PathBuf::from(input);
    let output_path = PathBuf::from(output);
    let memory_budget = match memory_budget {
        Some(s) => Some(scx_format_io::MemoryBudget::parse(&s).map_err(PyValueError::new_err)?),
        None => None,
    };
    // F1 grouped sharding (parse Python args into plain Rust before `py.detach`).
    let reference = parse_reference_spec(reference.as_ref())?;
    if reference.is_some() && group_by.is_none() {
        return Err(PyValueError::new_err(
            "reference requires group_by to be set",
        ));
    }
    let group_target_bytes = convert::parse_memory_budget(group_target_bytes.as_ref())?;
    let group_max_bytes = convert::parse_memory_budget(group_max_bytes.as_ref())?;
    let group_write_block_bytes = convert::parse_memory_budget(group_write_block_bytes.as_ref())?;
    let bitmap = scx_format_io::BitmapPolicy::parse(&bitmap).map_err(PyValueError::new_err)?;
    // Route shard_size through the shared validator (signed i64 so a negative
    // value is a clean `ValueError`, not pyo3 `OverflowError`), matching every
    // other op.
    let shard_target_rows = validate_shard_size(shard_size)?.get();
    let opts = scx_ops::SortOptions {
        by,
        reverse,
        shard_target_rows,
        codec: CodecSelection::Auto,
        index_options: ConversionPredicateIndexOptions {
            index_obs: index_obs.unwrap_or_default(),
            index_var: index_var.unwrap_or_default(),
            index_preset,
            // The sort key is auto-added regardless; this caps auto-detection
            // of other low-cardinality columns (0 = none).
            index_auto_threshold: index_auto_threshold.unwrap_or(0),
        },
        memory_budget,
        temp_dir: temp_dir.map(PathBuf::from),
        bitmap,
        // F1 grouped sharding (7.2a): thread the grouping options through so
        // Python can write grouped files without the `scx sort --group-by` CLI.
        group_by,
        reference,
        group_target_bytes,
        group_max_bytes,
        group_write_block_bytes,
    };
    py.detach(|| scx_ops::sort(&input_path, &output_path, &opts))
        .map_err(ops_to_pyerr)?;
    if rebuild_csc {
        // Run the heavy CSC rebuild off the GIL too. Its `Box<dyn Error>` is
        // not `Send`, so map it to a `String` inside the closure to cross
        // `py.detach`.
        py.detach(|| {
            scx_ops::rebuild_csc_inplace(&output_path, csc_cols_per_shard, &csc_memory_limit, None)
                .map_err(|e| e.to_string())
        })
        .map_err(PyRuntimeError::new_err)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// C5b. pyscx.build_csc()
// ---------------------------------------------------------------------------

/// Build a CSC (column-major) sidecar from an existing file's CSR shards.
///
/// Standalone equivalent of the `scx build-csc` CLI command: reads CSR
/// shards from `input` and writes both the CSR shards and a freshly built
/// CSC sidecar to `output`. CSC sidecars are the column-major substrate for
/// DE / HVG / per-gene QC / pseudobulk and the GPU `pdex_ref` CSC-direct
/// route.
///
/// For an in-place rebuild use `pyscx.sort(..., rebuild_csc=True)`; to emit
/// a sidecar at write time use `pyscx.from_anndata(..., csc="always")`.
///
/// Parameters:
///   input              — SCX file containing CSR shards.
///   output             — destination file (gets CSR + the new CSC shards).
///   memory_limit       — transpose working-set budget; accepts binary-
///                        prefixed sizes (`"4G"`, `"512MiB"`). Default "4G".
///   force              — overwrite `output` if it already exists.
///   csc_cols_per_shard — max columns per emitted CSC shard (0 = single
///                        shard, memory permitting). Default 5000.
///
/// Example:
///     pyscx.build_csc("counts.scx", "counts_csc.scx")
#[pyfunction]
#[pyo3(signature = (input, output, memory_limit="4G".to_string(), force=false, csc_cols_per_shard=5000))]
pub fn build_csc(
    py: Python<'_>,
    input: &str,
    output: &str,
    memory_limit: String,
    force: bool,
    csc_cols_per_shard: usize,
) -> PyResult<()> {
    // Fail fast with a clean ValueError on a malformed size string;
    // run_build_csc re-parses it internally with the same parser, so the
    // two cannot drift.
    scx_format_io::MemoryBudget::parse(&memory_limit).map_err(PyValueError::new_err)?;
    let input_path = PathBuf::from(input);
    let output_path = PathBuf::from(output);

    // Guard against `input == output`: run_build_csc removes `output` (when
    // `force`) before opening `input`, so an aliased path would delete the
    // source and then fail to open it. Compare canonicalized paths when both
    // resolve, falling back to a literal string compare for a not-yet-created
    // output.
    let same_file = match (
        std::fs::canonicalize(&input_path),
        std::fs::canonicalize(&output_path),
    ) {
        (Ok(a), Ok(b)) => a == b,
        _ => input_path == output_path,
    };
    if same_file {
        return Err(PyValueError::new_err(
            "input and output must be different files; build_csc writes the \
             CSR + new CSC sidecar to `output` (use a distinct path, or \
             sort(..., rebuild_csc=True) for an in-place rebuild)",
        ));
    }

    // `run_build_csc` is not modality-aware — it flattens every CSR shard
    // against the single top-level n_obs × n_vars shape, which would corrupt
    // the sidecar on a multimodal input. Reject it with a clear error.
    let input_reader =
        ScxReader::open(&input_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if input_reader.is_multimodal() {
        return Err(PyValueError::new_err(
            "build_csc does not support multimodal files; subset to a single \
             modality first (scx subset --modality NAME)",
        ));
    }
    drop(input_reader);

    // `run_build_csc` returns `Box<dyn Error>` (not `Send`), so stringify the
    // error inside the closure to cross `py.detach`, mirroring sort()'s CSC
    // rebuild path.
    py.detach(|| {
        scx_ops::run_build_csc(
            &input_path,
            &output_path,
            &memory_limit,
            force,
            csc_cols_per_shard,
            None,
        )
        .map_err(|e| e.to_string())
    })
    .map_err(PyRuntimeError::new_err)?;
    Ok(())
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
/// Optionally rebuilds predicate indexes on the merged output: pass
/// `index_obs=[...]` / `index_var=[...]` / `index_preset=...` to mirror
/// the `scx convert` surface. Without these kwargs the merged output
/// has NO predicate-index sections — query-time `filter_obs` pushdown
/// falls back to a full scan. This was the silent-data-loss bug
/// reported against multi-input atlas builds.
///
/// Validation kwargs (strict by default; opt back in to the
/// pre-strict behaviour explicitly):
///
/// * `assume_identical_var=False` — when `False` (default), merge
///   compares every input's var batch column-by-column against
///   input 0 and errors on mismatch. Set `True` if you've already
///   verified the gene axis upstream and want the count-only check.
/// * `assume_identical_obs=False` — when `False` (default), merge
///   compares every input's obs schema against input 0 (column names
///   + dtypes, normalised through the logical-lossy schema) and
///   errors on mismatch. Set `True` when the caller has already
///   validated obs columns.
/// * `uns_policy=None` — controls how the merged file's `uns`
///   section is built. `None` / `"first"` keeps input 0's payload
///   verbatim; `"require-equal"` errors on any disagreement;
///   `"namespace"` writes a `{"input_N": ...}` wrapper; `"summary"`
///   keeps input 0 and records a `_scx_uns_conflicts` array.
///   Applied independently at the global and per-modality levels
///   for multimodal inputs.
///
/// Example:
///     pyscx.merge(["batch1.scx", "batch2.scx", "batch3.scx"], "atlas.scx")
///     pyscx.merge(["a.scx", "b.scx"], "merged.scx",
///                 index_obs=["perturbation", "cell_type"])
///     pyscx.merge(["a.scx", "b.scx"], "merged.scx",
///                 assume_identical_var=True, uns_policy="namespace")
#[pyfunction]
#[pyo3(signature = (
    inputs, output,
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None,
    assume_identical_var=false, assume_identical_obs=false, uns_policy=None,
    sort_by=None, reverse=false,
))]
#[allow(clippy::too_many_arguments)]
pub fn merge(
    py: Python<'_>,
    inputs: Vec<String>,
    output: &str,
    index_obs: Option<Vec<String>>,
    index_var: Option<Vec<String>>,
    index_preset: Option<String>,
    index_auto_threshold: Option<usize>,
    assume_identical_var: bool,
    assume_identical_obs: bool,
    uns_policy: Option<String>,
    sort_by: Option<Vec<String>>,
    reverse: bool,
) -> PyResult<()> {
    if inputs.len() < 2 {
        return Err(PyValueError::new_err(
            "merge requires at least 2 input files",
        ));
    }

    let input_paths: Vec<PathBuf> = inputs.iter().map(PathBuf::from).collect();
    let input_refs: Vec<&Path> = input_paths.iter().map(|p| p.as_path()).collect();
    let output_path = PathBuf::from(output);

    // Parse the optional uns_policy kwarg into the enum. Default
    // (None) preserves `UnsPolicy::First` = today's behaviour: read
    // the first input's `uns` verbatim, drop the rest.
    let uns_policy_parsed = match uns_policy.as_deref() {
        Some(s) => scx_ops::UnsPolicy::parse(s).ok_or_else(|| {
            PyValueError::new_err(format!(
                "invalid uns_policy '{s}': expected one of \
                 first, require-equal, namespace, summary"
            ))
        })?,
        None => scx_ops::UnsPolicy::First,
    };

    let sort_by = sort_by.unwrap_or_default();
    let want_sort = !sort_by.is_empty();
    let want_policy = assume_identical_var
        || assume_identical_obs
        || uns_policy_parsed != scx_ops::UnsPolicy::First;
    match build_index_options(index_obs, index_var, index_preset, index_auto_threshold) {
        Some(index_opts) => {
            let merge_opts = scx_ops::MergeOptions {
                index_options: index_opts,
                assume_identical_var,
                assume_identical_obs,
                uns_policy: uns_policy_parsed,
                shard_target_rows: None,
                sort_by,
                sort_reverse: reverse,
            };
            let summary = py
                .detach(|| scx_ops::merge_with_options(&input_refs, &output_path, &merge_opts))
                .map_err(ops_to_pyerr)?;
            process_index_summary(py, summary)
        }
        None if want_policy || want_sort => {
            let merge_opts = scx_ops::MergeOptions {
                index_options: scx_engine::ConversionPredicateIndexOptions {
                    index_obs: Vec::new(),
                    index_var: Vec::new(),
                    index_preset: None,
                    index_auto_threshold: 0,
                },
                assume_identical_var,
                assume_identical_obs,
                uns_policy: uns_policy_parsed,
                shard_target_rows: None,
                sort_by,
                sort_reverse: reverse,
            };
            let summary = py
                .detach(|| scx_ops::merge_with_options(&input_refs, &output_path, &merge_opts))
                .map_err(ops_to_pyerr)?;
            process_index_summary(py, summary)
        }
        None => py
            .detach(|| scx_ops::merge(&input_refs, &output_path))
            .map_err(ops_to_pyerr),
    }
}

// ---------------------------------------------------------------------------
// In-place metadata replacement (set_uns / modify_metadata)
// ---------------------------------------------------------------------------

/// Convert an obs/var input (pandas `DataFrame` or pyarrow `Table`) to an
/// Arrow `RecordBatch`. A `Table` is routed through `to_pandas()` so the
/// shared `pandas_to_record_batch` IPC path handles both.
fn obs_var_to_record_batch(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    param: &str,
) -> PyResult<RecordBatch> {
    let pa = py.import("pyarrow")?;
    let table_cls = pa.getattr("Table")?;
    if obj.is_instance(&table_cls)? {
        let df = obj.call_method0("to_pandas")?;
        return convert::pandas_to_record_batch(py, &df);
    }
    // Require a pandas DataFrame. Without this guard a dict (a natural thing to
    // try) falls through to `pyarrow.Table.from_pandas` and surfaces an opaque
    // `AttributeError: 'dict' object has no attribute 'columns'` deep inside
    // pyarrow, with no mention of `modify_metadata`, the parameter, or the
    // expected type (report E2).
    let pd = py.import("pandas")?;
    let df_cls = pd.getattr("DataFrame")?;
    if !obj.is_instance(&df_cls)? {
        let got = obj
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "object".to_string());
        return Err(PyTypeError::new_err(format!(
            "modify_metadata({param}=...) expects a pandas DataFrame (got {got}); \
             wrap your columns with pd.DataFrame({{...}})."
        )));
    }
    convert::pandas_to_record_batch(py, obj)
}

/// Convert a `dict[str, ndarray]` (obsm/varm) into the named dense
/// `RecordBatch`es the Rust patch expects.
fn dense_dict_to_batches(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    axis: &str,
) -> PyResult<Vec<(String, RecordBatch)>> {
    let dict = obj.cast::<PyDict>().map_err(|_| {
        PyValueError::new_err(format!("{axis} must be a dict of {{name: ndarray}}"))
    })?;
    let mut out = Vec::with_capacity(dict.len());
    for (k, v) in dict.iter() {
        let name: String = k
            .extract()
            .map_err(|_| PyValueError::new_err(format!("{axis} keys must be strings")))?;
        let batch = convert::numpy_or_pandas_to_record_batch(py, &v)?;
        out.push((name, batch));
    }
    Ok(out)
}

/// Resolve a `modality` NAME against a reader to a `modality_id`, mirroring
/// the `scx append --modality` semantics: required on multimodal files,
/// rejected on single-modality files, and `0` (global / legacy) when omitted.
/// `label` ("target" / "input") is woven into error messages.
fn resolve_append_modality(
    reader: &ScxReader,
    label: &str,
    modality: Option<&str>,
) -> PyResult<u8> {
    if reader.is_multimodal() {
        match modality {
            Some(name) => reader.modality_id(name).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "{label} file has no modality named '{name}'; \
                     use Experiment.modality_names to list them"
                ))
            }),
            None => Err(PyValueError::new_err(format!(
                "{label} file is multimodal ({} modalities); pass modality=NAME",
                reader.n_modalities()
            ))),
        }
    } else if let Some(name) = modality {
        Err(PyValueError::new_err(format!(
            "{label} file is single-modality but modality='{name}' was passed; omit it"
        )))
    } else {
        Ok(0)
    }
}

/// Per-modality `n_vars` (file-wide `header.n_vars` for the global modality 0).
fn modality_n_vars(reader: &ScxReader, modality_id: u8) -> u64 {
    if modality_id == 0 {
        reader.header().n_vars
    } else {
        // Fallback to the file-wide `n_vars` when a non-zero id has no
        // modality descriptor — mirrors `scx-cli/src/append.rs` so the two
        // front-ends stay consistent (a broken descriptor falls back rather
        // than hard-failing here; the deeper `n_minor` check still guards).
        reader
            .modality_info(modality_id)
            .map(|i| i.n_vars)
            .unwrap_or(reader.header().n_vars)
    }
}

/// Resolve the optional `modality` kwarg to a `modality_id`. Only the
/// global modality (`0`) is supported today; non-zero ids reach the Rust
/// layer which returns a clear `MultimodalUnsupported` error.
///
/// See also `resolve_append_modality` above, which resolves a modality
/// *name* against a reader (used by `append` / `append_from_anndata`); this
/// integer-only variant is for `modify_metadata`.
fn resolve_modality_id(modality: Option<&Bound<'_, PyAny>>) -> PyResult<u8> {
    match modality {
        None => Ok(0),
        Some(m) => {
            if let Ok(i) = m.extract::<i64>() {
                if !(0..=255).contains(&i) {
                    return Err(PyValueError::new_err(
                        "modality id out of range (expected 0..=255)",
                    ));
                }
                Ok(i as u8)
            } else if m.extract::<String>().is_ok() {
                Err(PyValueError::new_err(
                    "named modality is not yet supported by modify_metadata; \
                     pass an integer modality id (only 0 / global is supported today)",
                ))
            } else {
                Err(PyValueError::new_err("modality must be an int or None"))
            }
        }
    }
}

/// Replace the whole `uns` block of an existing `.scx` file in place,
/// without re-encoding `X`.
///
/// **Replace semantics, not merge** — `uns` fully supersedes the existing
/// block (consistent with `from_h5ad(..., uns_override=)`). The matrix
/// (`X` / CSR / CSC shards) is never read or rewritten, so a pre-existing
/// CSC sidecar stays valid. The change is atomic and rollback-able
/// (`pyscx.rollback`).
///
/// For a shallow merge, read-modify-write::
///
///     adata = pyscx.open(path).to_anndata()
///     adata.uns["descriptions"] = {...}
///     pyscx.set_uns(path, dict(adata.uns))
#[pyfunction]
pub fn set_uns(py: Python<'_>, path: &str, uns: &Bound<'_, PyAny>) -> PyResult<()> {
    let json = convert::uns_py_to_json(py, uns, convert::UnsFormat::Tagged)?;
    let path_buf = PathBuf::from(path);
    py.detach(|| scx_ops::set_uns(&path_buf, &json))
        .map_err(ops_to_pyerr)?;
    Ok(())
}

/// Replace metadata sections (`uns` / `obs` / `var` / `obsm` / `varm`) of an
/// existing `.scx` file in place, without re-encoding `X`.
///
/// Any omitted argument is left untouched (its sections pass through
/// verbatim). Cost is O(size of the replaced sections); the matrix shards
/// are never read or rewritten, so a pre-existing CSC sidecar and
/// `data_generation` are preserved (no `--rebuild-csc` needed). One atomic
/// commit; rollback-able via `pyscx.rollback`.
///
/// **Replace semantics, not merge.** A supplied `obs`/`var` fully replaces
/// the section and `num_rows` must match the file's `n_obs` / `n_vars`
/// (changing cell/gene count is out of scope — use `append` / `subset`).
/// `obsm` / `varm` replace only the named matrices. Predicate indexes over a
/// replaced `obs`/`var` are dropped unless `index_obs` / `index_var` /
/// `index_preset` request a rebuild.
///
/// Args:
///     path: target `.scx` file.
///     uns: dict replacing the whole `uns` block.
///     obs / var: pandas `DataFrame` (or pyarrow `Table`); `num_rows` must
///         equal `n_obs` / `n_vars`.
///     obsm / varm: `dict[str, np.ndarray]` of named dense matrices.
///     index_obs / index_var / index_preset / index_auto_threshold:
///         predicate-index rebuild policy (only consulted when obs/var change).
///     modality: integer modality id (only `0` / global is supported today).
#[pyfunction]
#[pyo3(signature = (
    path, *, uns=None, obs=None, var=None, obsm=None, varm=None,
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None,
    modality=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn modify_metadata(
    py: Python<'_>,
    path: &str,
    uns: Option<&Bound<'_, PyAny>>,
    obs: Option<&Bound<'_, PyAny>>,
    var: Option<&Bound<'_, PyAny>>,
    obsm: Option<&Bound<'_, PyAny>>,
    varm: Option<&Bound<'_, PyAny>>,
    index_obs: Option<Vec<String>>,
    index_var: Option<Vec<String>>,
    index_preset: Option<String>,
    index_auto_threshold: Option<usize>,
    modality: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let uns_json = match uns {
        Some(u) => Some(convert::uns_py_to_json(py, u, convert::UnsFormat::Tagged)?),
        None => None,
    };
    let obs_batch = match obs {
        Some(o) => Some(obs_var_to_record_batch(py, o, "obs")?),
        None => None,
    };
    let var_batch = match var {
        Some(v) => Some(obs_var_to_record_batch(py, v, "var")?),
        None => None,
    };
    let obsm_batches = match obsm {
        Some(d) => Some(dense_dict_to_batches(py, d, "obsm")?),
        None => None,
    };
    let varm_batches = match varm {
        Some(d) => Some(dense_dict_to_batches(py, d, "varm")?),
        None => None,
    };
    let modality_id = resolve_modality_id(modality)?;
    let index = build_index_options(index_obs, index_var, index_preset, index_auto_threshold)
        .unwrap_or_default();

    let patch = scx_ops::MetadataPatch {
        uns: uns_json,
        obs: obs_batch,
        var: var_batch,
        obsm: obsm_batches,
        varm: varm_batches,
        index,
        modality_id,
    };
    let path_buf = PathBuf::from(path);
    py.detach(|| scx_ops::modify_metadata(&path_buf, &patch))
        .map_err(ops_to_pyerr)?;
    Ok(())
}
