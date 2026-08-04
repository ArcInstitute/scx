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
        | OpsError::InvalidInput(_)
        | OpsError::KeyColumnUnresolved { .. }
        | OpsError::DuplicateJoinKey { .. }
        | OpsError::AxisMismatch { .. }
        | OpsError::MultimodalUnsupported { .. } => PyValueError::new_err(msg),
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
    reshape_obs=false, codec="auto",
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
    codec: &str,
) -> PyResult<()> {
    let input_path = PathBuf::from(input);
    let output_path = PathBuf::from(output);
    let resolved_codec = parse_codec_intent(codec)?;
    match build_index_options(index_obs, index_var, index_preset, index_auto_threshold) {
        Some(index_opts) => {
            let summary = py
                .detach(|| {
                    scx_ops::compact_with_options(
                        &input_path,
                        &output_path,
                        &scx_ops::CompactOptions {
                            index_options: index_opts,
                            reshape_obs,
                            codec: resolved_codec,
                        },
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
            let summary = py
                .detach(|| {
                    scx_ops::compact_with_options(
                        &input_path,
                        &output_path,
                        &scx_ops::CompactOptions {
                            reshape_obs: true,
                            codec: resolved_codec,
                            ..Default::default()
                        },
                    )
                })
                .map_err(ops_to_pyerr)?;
            process_index_summary(py, summary)
        }
        // Still the options path so a lone `codec=` is not silently dropped;
        // `CompactOptions::default()` reproduces bare `compact()`.
        None => py
            .detach(|| {
                scx_ops::compact_with_options(
                    &input_path,
                    &output_path,
                    &scx_ops::CompactOptions {
                        codec: resolved_codec,
                        ..Default::default()
                    },
                )
            })
            .map(|_| ())
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

/// Parse a `codec=` kwarg into a [`CodecSelection`].
///
/// Both `sort` and `shuffle` hardcoded `Auto` before 1D, which left the Python
/// surface unable to express something the CLI has always had — and unable to
/// follow its *own* documented advice ("pin `--codec` if output size matters",
/// `docs/sharding.md`). It bit the 1D size benchmark first: sweeping the
/// per-codec fixtures produced byte-identical outputs for every variant,
/// because the writer re-selected `auto` each time, so the sweep measured
/// auto-reselection rather than whether a permutation grows that codec.
fn parse_codec_intent(codec: &str) -> PyResult<scx_format_io::ResolvedCodec> {
    scx_format_io::resolve_codec(Some(codec)).map_err(PyValueError::new_err)
}

/// Run a built [`scx_ops::SortOptions`] off the GIL, then optionally rebuild
/// the CSC sidecar off the GIL too.
///
/// Shared by `sort` and `shuffle` so the two cannot drift on the part that
/// matters — GIL handling, error mapping, and the post-write CSC rebuild. The
/// *option construction* stays in each pyfunction, because that is exactly
/// where they legitimately differ.
fn run_sort_engine(
    py: Python<'_>,
    input_path: &Path,
    output_path: &Path,
    opts: &scx_ops::SortOptions,
    rebuild_csc: bool,
    csc_cols_per_shard: usize,
    csc_memory_limit: &str,
) -> PyResult<()> {
    py.detach(|| scx_ops::sort(input_path, output_path, opts))
        .map_err(ops_to_pyerr)?;
    if rebuild_csc {
        // Run the heavy CSC rebuild off the GIL too. Its `Box<dyn Error>` is
        // not `Send`, so map it to a `String` inside the closure to cross
        // `py.detach`.
        // NOT `None`: on a v4 output that would rewrite CSR + CSC unframed and
        // strip the row-group framing the sort just wrote. See
        // `scx_ops::framing_for_csc_rebuild`.
        let csc_framing = scx_ops::framing_for_csc_rebuild(output_path);
        py.detach(|| {
            scx_ops::rebuild_csc_inplace(
                output_path,
                csc_cols_per_shard,
                csc_memory_limit,
                csc_framing,
            )
            .map_err(|e| e.to_string())
        })
        .map_err(PyRuntimeError::new_err)?;
    }
    Ok(())
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
/// to re-emit the column-major sidecar.
///
/// For a *random* reorder — training-batch diversity rather than query
/// locality — see `pyscx.shuffle`.
///
/// Example:
///     pyscx.sort("atlas.scx", "atlas.sorted.scx", by=["cell_type"])
#[pyfunction]
#[pyo3(signature = (
    input, output, by, reverse=false, shard_size=None, codec="auto".to_string(),
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
    codec: String,
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
        shuffle: None,
        shard_target_rows,
        codec: parse_codec_intent(&codec)?,
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
    run_sort_engine(
        py,
        &input_path,
        &output_path,
        &opts,
        rebuild_csc,
        csc_cols_per_shard,
        &csc_memory_limit,
    )
}

// ---------------------------------------------------------------------------
// 1D. pyscx.shuffle()
// ---------------------------------------------------------------------------

/// Globally reorder cells (the obs axis) of an SCX file by a **seeded random
/// permutation**, writing a new file whose row order carries no residual
/// structure.
///
/// This is the training-side counterpart of `pyscx.sort`. `TrainingDataset`
/// randomizes in two levels — shard order, then a Fisher-Yates shuffle within
/// each shard group — so on a file whose rows arrived clustered (by donor,
/// plate, or cell type) batch composition is capped by `shard_group_size`, and
/// widening it costs memory linearly. Permuting once, on disk, moves that cost
/// off the training loop.
///
/// `seed` is recorded in the output's provenance and is the **only** record of
/// the permutation: the same seed on the same input always reproduces the same
/// file, and nothing else can. Note the permutation runs over *live* rows, so
/// a file with deletion vectors shuffles differently from the same file without
/// them (deletions are materialized away, as in `sort`).
///
/// Two consequences worth knowing before a multi-hour rewrite:
///
/// - **Size is not quite neutral, and `codec="auto"` is still the right
///   choice.** A permutation genuinely loses some cross-row redundancy for
///   codecs whose compression spans rows — 6-12% for `zstd`, under 1% for
///   `lz4`/`shufdelta`. That is inherent to shuffling. `auto` runs the same
///   adaptive per-shard selection `scx convert` does, so pinning a codec
///   *chooses an encoding* rather than holding the file's size; reach for
///   `codec="scx1"` when you want a permutation-invariant layout or the GPU
///   device-decode route. (This used to say `auto` re-selects and grows X
///   1.86-2.09x, and to pin the input's own codec. That growth was a bug in
///   every derived-file op — `FramingConfig::default()` meant the `fast`
///   profile — not a property of shuffling, and it is fixed.)
/// - **Shard geometry is not preserved by default.** `shard_size=None` uses the
///   16,384-row default, so a file written with a different shard size is
///   re-sharded as well as reordered — and shard size is what quantises batch
///   composition. Pass `shard_size=<the input's value>` to reorder only.
/// - **It is the inverse of a sort for queries.** Sorting collapses each
///   category's predicate-index shard ranges to one contiguous run; shuffling
///   scatters every category across every shard.
///
/// Example:
///     pyscx.shuffle("atlas.scx", "atlas.shuffled.scx", seed=42)
#[pyfunction]
#[pyo3(signature = (
    input, output, seed=42, shard_size=None, codec="auto".to_string(),
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None,
    memory_budget=None, temp_dir=None, bitmap="off".to_string(), rebuild_csc=false,
    csc_cols_per_shard=5000, csc_memory_limit="4G".to_string(),
))]
#[allow(clippy::too_many_arguments)]
pub fn shuffle(
    py: Python<'_>,
    input: &str,
    output: &str,
    seed: u64,
    shard_size: Option<i64>,
    codec: String,
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
) -> PyResult<()> {
    let input_path = PathBuf::from(input);
    let output_path = PathBuf::from(output);
    let memory_budget = match memory_budget {
        Some(s) => Some(scx_format_io::MemoryBudget::parse(&s).map_err(PyValueError::new_err)?),
        None => None,
    };
    let bitmap = scx_format_io::BitmapPolicy::parse(&bitmap).map_err(PyValueError::new_err)?;
    let shard_target_rows = validate_shard_size(shard_size)?.get();
    let opts = scx_ops::SortOptions {
        // Shuffle is an order *source*, not a modifier: there is no key, and
        // the engine rejects `by` / `group_by` / `reverse` alongside it. This
        // surface simply does not expose them.
        by: Vec::new(),
        reverse: false,
        shuffle: Some(seed),
        shard_target_rows,
        codec: parse_codec_intent(&codec)?,
        index_options: ConversionPredicateIndexOptions {
            index_obs: index_obs.unwrap_or_default(),
            index_var: index_var.unwrap_or_default(),
            index_preset,
            index_auto_threshold: index_auto_threshold.unwrap_or(0),
        },
        memory_budget,
        temp_dir: temp_dir.map(PathBuf::from),
        bitmap,
        group_by: None,
        reference: None,
        group_target_bytes: None,
        group_max_bytes: None,
        group_write_block_bytes: None,
    };
    run_sort_engine(
        py,
        &input_path,
        &output_path,
        &opts,
        rebuild_csc,
        csc_cols_per_shard,
        &csc_memory_limit,
    )
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
    sort_by=None, reverse=false, codec="auto",
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
    codec: &str,
) -> PyResult<()> {
    let resolved_codec = parse_codec_intent(codec)?;
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
                codec: resolved_codec,
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
                codec: resolved_codec,
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
        // Still the options path so a lone `codec=` is not silently dropped;
        // `MergeOptions::default()` reproduces the bare `merge` wrapper.
        None => py
            .detach(|| {
                scx_ops::merge_with_options(
                    &input_refs,
                    &output_path,
                    &scx_ops::MergeOptions {
                        codec: resolved_codec,
                        ..Default::default()
                    },
                )
            })
            .map(|_| ())
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

// ---------------------------------------------------------------------------
// CellBender interop
// ---------------------------------------------------------------------------

/// Shared option parsing for `cellbender_import`, so the CLI and Python
/// surfaces cannot drift on the enum spellings.
#[cfg(feature = "hdf5")]
fn parse_missing_rows(s: &str) -> PyResult<scx_ops::MissingRowPolicy> {
    match s {
        "zero" => Ok(scx_ops::MissingRowPolicy::ZeroFill),
        "error" => Ok(scx_ops::MissingRowPolicy::Error),
        other => Err(PyValueError::new_err(format!(
            "on_missing_rows must be 'zero' or 'error'; got '{other}'"
        ))),
    }
}

#[cfg(feature = "hdf5")]
fn parse_extra_rows(s: &str) -> PyResult<scx_ops::ExtraRowPolicy> {
    match s {
        "warn" => Ok(scx_ops::ExtraRowPolicy::WarnSkip),
        "error" => Ok(scx_ops::ExtraRowPolicy::Error),
        other => Err(PyValueError::new_err(format!(
            "on_extra_rows must be 'warn' or 'error'; got '{other}'"
        ))),
    }
}

#[cfg(feature = "hdf5")]
fn parse_gene_axis(s: &str) -> PyResult<scx_ops::ColumnAxisPolicy> {
    match s {
        "identical" => Ok(scx_ops::ColumnAxisPolicy::RequireIdentical),
        "reorder" => Ok(scx_ops::ColumnAxisPolicy::AllowReorder),
        "subset" => Ok(scx_ops::ColumnAxisPolicy::AllowSubset),
        other => Err(PyValueError::new_err(format!(
            "gene_axis must be 'identical', 'reorder' or 'subset'; got '{other}'"
        ))),
    }
}

/// Import a CellBender `remove-background` output into an existing SCX file.
///
/// Reads the corrected count matrix and lands it as a new layer on `path`,
/// **in place**, joined to the target's own obs axis by barcode. X, the CSC
/// sidecar, `.raw`, deletion vectors and predicate indexes are preserved; the
/// whole import is undoable with `scx rollback`.
///
/// The join is always by barcode string, never by position: CellBender's
/// `_filtered.h5` is in descending-UMI order, so a positional import would
/// silently put every cell's corrected counts on the wrong barcode.
///
/// Returns a dict summarising the join — inspect `n_matched` before trusting
/// the result.
#[cfg(feature = "hdf5")]
#[pyfunction]
#[pyo3(signature = (
    path, cellbender_h5, *, layer="cellbender", obs_key=None, var_key=None,
    prefix="cellbender_", uns_key="cellbender", overwrite=false,
    on_missing_rows="zero", on_extra_rows="warn", gene_axis="identical",
    latent_embedding=false, dry_run=false
))]
#[allow(clippy::too_many_arguments)]
pub fn cellbender_import(
    py: Python<'_>,
    path: &str,
    cellbender_h5: &str,
    layer: &str,
    obs_key: Option<String>,
    var_key: Option<String>,
    prefix: &str,
    uns_key: Option<&str>,
    overwrite: bool,
    on_missing_rows: &str,
    on_extra_rows: &str,
    gene_axis: &str,
    latent_embedding: bool,
    dry_run: bool,
) -> PyResult<Py<pyo3::types::PyDict>> {
    use pyo3::types::PyDict;

    let read_opts = scx_convert::CellBenderReadOptions {
        column_prefix: prefix.to_string(),
        latent_embedding,
        ..Default::default()
    };
    let attach_opts = scx_ops::AttachLayerOptions {
        layer_name: layer.to_string(),
        obs_key_column: obs_key,
        var_key_column: var_key,
        missing_row_policy: parse_missing_rows(on_missing_rows)?,
        extra_row_policy: parse_extra_rows(on_extra_rows)?,
        column_axis_policy: parse_gene_axis(gene_axis)?,
        status_column: Some(format!("{prefix}status")),
        row_sum_column: Some(format!("{prefix}total_counts")),
        uns_key: uns_key.map(str::to_string),
        overwrite,
        provenance_action: "cellbender_import".to_string(),
        dry_run,
        ..Default::default()
    };

    let scx_path = PathBuf::from(path);
    let h5_path = PathBuf::from(cellbender_h5);

    // `dry_run` is honoured inside the op: it runs every validation and the
    // join, then returns without writing, so `n_matched` below is real.
    let (summary, info) = py.detach(|| -> PyResult<_> {
        let mut sink = scx_convert::WarningSink::log();
        let out = scx_convert::read_cellbender_h5(&h5_path, &read_opts, &mut sink)
            .map_err(crate::convert_to_pyerr)?;
        let summary = scx_ops::attach_external_layer(&scx_path, &out.data, &attach_opts)
            .map_err(ops_to_pyerr)?;
        Ok((summary, out.info))
    })?;

    let d = PyDict::new(py);
    d.set_item(
        "output_kind",
        format!("{:?}", info.output_kind).to_lowercase(),
    )?;
    d.set_item(
        "latent_alignment",
        format!("{:?}", info.latent_alignment).to_lowercase(),
    )?;
    d.set_item("n_rows_in_source", info.n_rows)?;
    d.set_item("n_features_in_source", info.n_features)?;
    d.set_item("estimator", info.estimator.clone())?;
    d.set_item("all_values_integer", info.all_values_integer)?;
    d.set_item("dry_run", dry_run)?;
    {
        let s = summary;
        d.set_item("layer", layer)?;
        d.set_item("n_obs", s.n_obs)?;
        d.set_item("n_matched", s.n_matched)?;
        d.set_item("n_target_rows_absent", s.n_target_rows_absent)?;
        d.set_item("n_source_rows_absent", s.n_source_rows_absent)?;
        d.set_item(
            "n_source_rows_absent_nonzero",
            s.n_source_rows_absent_nonzero,
        )?;
        d.set_item("obs_key_column", s.obs_key_column)?;
        d.set_item("var_key_column", s.var_key_column)?;
        d.set_item(
            "gene_axis_match",
            format!("{:?}", s.column_axis_match).to_lowercase(),
        )?;
        d.set_item("layer_nnz", s.layer_nnz)?;
        d.set_item(
            "value_encoding",
            format!("{:?}", s.value_encoding).to_lowercase(),
        )?;
        d.set_item("obs_columns_added", s.obs_columns_added)?;
        d.set_item("var_columns_added", s.var_columns_added)?;
        d.set_item("obsm_keys_added", s.obsm_keys_added)?;
    }
    Ok(d.into())
}

/// Probe whether a file looks like a CellBender `remove-background` output
/// (as opposed to a plain 10x CellRanger matrix).
#[cfg(feature = "hdf5")]
#[pyfunction]
pub fn is_cellbender_h5(path: &str) -> bool {
    scx_convert::is_cellbender_h5(std::path::Path::new(path))
}

// ---------------------------------------------------------------------------
// Generic external obs import (CSV / TSV annotation tables)
// ---------------------------------------------------------------------------

/// Shared option parsing, so the CLI and Python surfaces cannot drift on the
/// enum spellings. Ungated, unlike the CellBender pair above — a delimited
/// table reader has no business requiring libhdf5.
pub fn parse_obs_missing_rows(s: &str) -> PyResult<scx_ops::MissingRowPolicy> {
    match s {
        // The obs paths scatter Arrow *nulls*, not zeros, and null-aware
        // consensus depends on that — so "null" is the accurate spelling.
        // "zero" stays accepted: it is the name the CellBender-era policy
        // enum carries and what earlier callers pass.
        "null" | "zero" => Ok(scx_ops::MissingRowPolicy::ZeroFill),
        "error" => Ok(scx_ops::MissingRowPolicy::Error),
        other => Err(PyValueError::new_err(format!(
            "on_missing_rows must be 'null' (leave uncovered rows NULL, the \
             default), 'zero' (a legacy alias for the same thing) or 'error'; \
             got '{other}'"
        ))),
    }
}

pub fn parse_obs_extra_rows(s: &str) -> PyResult<scx_ops::ExtraRowPolicy> {
    match s {
        "warn" => Ok(scx_ops::ExtraRowPolicy::WarnSkip),
        "error" => Ok(scx_ops::ExtraRowPolicy::Error),
        other => Err(PyValueError::new_err(format!(
            "on_extra_rows must be 'warn' or 'error'; got '{other}'"
        ))),
    }
}

/// Turn the caller's `key` / `source_key` into the reader's column list and the
/// op's join-key spec.
///
/// Without `source_key` both sides are built from the same names, which is the
/// common case and the only one expressible before: a target obs keyed on
/// (`sample_id`, obs index) could not be joined to a tool output keyed on
/// (`sample_id`, `barcode`) without renaming a column in pandas first. The two
/// lists pair up **positionally**, mirroring pandas `left_on` / `right_on`.
///
/// Nothing downstream needs to know the names differ: `build_composite_key`
/// fuses each side from its own columns in the given order and the fused key
/// never carries a column name.
fn resolve_join_key(
    key: Option<Vec<String>>,
    source_key: Option<Vec<String>>,
) -> PyResult<(Vec<String>, scx_ops::ObsJoinKey)> {
    let target = key.unwrap_or_default();
    let source = source_key.unwrap_or_default();
    if !source.is_empty() && target.is_empty() {
        return Err(PyValueError::new_err(
            "source_key= needs key=: it names the source-side column for each \
             target-side key component, positionally. To key on the target's obs \
             index, pass key=\"obs_names\".",
        ));
    }
    if !source.is_empty() && source.len() != target.len() {
        return Err(PyValueError::new_err(format!(
            "key= has {} component(s) but source_key= has {}; they pair up \
             positionally, so the counts must match",
            target.len(),
            source.len()
        )));
    }
    let join_key = match target.len() {
        0 => scx_ops::ObsJoinKey::Auto,
        1 => scx_ops::ObsJoinKey::Column(target[0].clone()),
        _ => scx_ops::ObsJoinKey::Composite {
            columns: target.clone(),
        },
    };
    // Absent `source_key`, the reader gets the target names — the historical
    // behaviour, and still what makes the two sides impossible to desync.
    let source_columns = if source.is_empty() { target } else { source };
    Ok((source_columns, join_key))
}

fn key_diagnosis_dict<'py>(
    py: Python<'py>,
    d: &scx_ops::KeyDiagnosis,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("n_obs", d.n_obs)?;
    out.set_item("resolved_key", d.resolved_key.clone())?;
    out.set_item("resolved_cardinality", d.resolved_cardinality)?;
    out.set_item("unique_columns", d.unique_columns.clone())?;
    // Unique, but refused as a key — a float, or any other type that is not
    // guaranteed to render identically on two independently written sides. Kept
    // out of `unique_columns` so nothing in that list is a key the join rejects.
    out.set_item("unusable_unique_columns", d.unusable_unique_columns.clone())?;
    let pairs: Vec<(String, String)> = d.unique_pairs.clone();
    out.set_item("unique_pairs", pairs)?;
    out.set_item("pair_search_capped", d.pair_search_capped)?;
    out.set_item("suggestion", d.suggestion.clone())?;
    out.set_item("summary", d.describe())?;
    Ok(out)
}

/// Import a delimited annotation table (CSV / TSV) as `obs` columns.
///
/// Reads `table` and lands its columns on `path`, **in place**, joined to the
/// target's own obs axis by key. X, layers, `var`, the CSC sidecar, `.raw`,
/// deletion vectors and predicate indexes are preserved; the whole import is
/// undoable with `pyscx.rollback`.
///
/// The join is always by key string, never by position — a doublet caller run
/// per library returns rows in whatever order it pleased, and a positional
/// import would put every score on the wrong cell while still producing a
/// correctly-shaped column.
///
/// **`overwrite` replaces, it does not merge.** Importing several per-batch
/// tables one after another would keep only the last; concatenate them and
/// import once.
///
/// Returns a dict summarising the read and the join — inspect `n_matched`
/// before trusting the result, and prefer `dry_run=True` on a large file.
#[pyfunction]
#[pyo3(signature = (
    path, table, *, key=None, source_key=None, columns=None, rename=None, prefix="",
    keep_key_columns=false, delimiter=None, status_column=None, uns_key=None,
    uns_keys=None, overwrite=false, on_missing_rows="null", on_extra_rows="warn",
    dry_run=false
))]
#[allow(clippy::too_many_arguments)]
pub fn obs_import(
    py: Python<'_>,
    path: &str,
    table: &str,
    key: Option<Vec<String>>,
    source_key: Option<Vec<String>>,
    columns: Option<Vec<String>>,
    rename: Option<std::collections::HashMap<String, String>>,
    prefix: &str,
    keep_key_columns: bool,
    delimiter: Option<&str>,
    status_column: Option<&str>,
    uns_key: Option<&str>,
    uns_keys: Option<Vec<String>>,
    overwrite: bool,
    on_missing_rows: &str,
    on_extra_rows: &str,
    dry_run: bool,
) -> PyResult<Py<PyDict>> {
    let uns_keys = uns_keys.unwrap_or_default();
    let delimiter_byte = match delimiter {
        None => None,
        Some(s) => {
            let bytes = s.as_bytes();
            if bytes.len() != 1 {
                return Err(PyValueError::new_err(format!(
                    "delimiter must be exactly one byte; got {s:?}"
                )));
            }
            Some(bytes[0])
        }
    };

    let (key_columns, join_key) = resolve_join_key(key, source_key)?;

    let read_opts = scx_convert::AnnotationTableOptions {
        key_columns,
        delimiter: delimiter_byte,
        columns,
        rename: rename.unwrap_or_default(),
        prefix: prefix.to_string(),
        keep_key_columns,
        infer_max_records: None,
    };
    let attach_opts = scx_ops::AttachObsOptions {
        join_key,
        missing_row_policy: parse_obs_missing_rows(on_missing_rows)?,
        extra_row_policy: parse_obs_extra_rows(on_extra_rows)?,
        status_column: status_column.map(str::to_string),
        uns_key: uns_key.map(str::to_string),
        overwrite,
        provenance_action: "obs_import".to_string(),
        dry_run,
        ..Default::default()
    };

    let scx_path = PathBuf::from(path);
    let table_path = PathBuf::from(table);

    // Both halves return `OpsError`, so one mapping covers read-then-attach —
    // a malformed CSV surfaces as a clean `ValueError`, not a panic.
    let (summary, info, diagnosis) = py.detach(|| -> PyResult<_> {
        let (data, info) = scx_convert::read_obs_source(&table_path, &read_opts, &uns_keys)
            .map_err(ops_to_pyerr)?;
        let summary =
            scx_ops::attach_external_obs(&scx_path, &data, &attach_opts).map_err(ops_to_pyerr)?;
        // Only on a dry run: the diagnosis costs a pass per obs column, which
        // is not something a successful import should pay for.
        let diagnosis = if dry_run {
            scx_ops::diagnose_obs_key(&scx_path, Some(&attach_opts.join_key)).ok()
        } else {
            None
        };
        Ok((summary, info, diagnosis))
    })?;

    let d = PyDict::new(py);
    d.set_item("n_rows_in_source", info.n_rows)?;
    d.set_item("format", info.format.as_str())?;
    d.set_item("delimiter", info.delimiter.map(|b| (b as char).to_string()))?;
    d.set_item("uns_keys_imported", info.uns_keys_imported)?;
    d.set_item("renamed_index_column", info.renamed_index_column)?;
    d.set_item("source_key_columns", info.key_columns)?;
    d.set_item("columns_in_source", info.columns_imported)?;
    d.set_item("dry_run", dry_run)?;
    d.set_item("n_obs", summary.n_obs)?;
    d.set_item("n_matched", summary.n_matched)?;
    d.set_item("n_target_rows_absent", summary.n_target_rows_absent)?;
    d.set_item("n_source_rows_absent", summary.n_source_rows_absent)?;
    d.set_item(
        "obs_key_column",
        scx_ops::display_key_name("obs", &summary.obs_key_column),
    )?;
    d.set_item("obs_columns_added", summary.obs_columns_added)?;
    d.set_item("obsm_keys_added", summary.obsm_keys_added)?;
    d.set_item("obs_index_dropped", summary.obs_index_dropped)?;
    if let Some(diag) = diagnosis {
        d.set_item("key_diagnosis", key_diagnosis_dict(py, &diag)?)?;
    }
    Ok(d.into())
}

// ---------------------------------------------------------------------------
// Doublet-caller wrapper
// ---------------------------------------------------------------------------

/// Import a doublet caller's output table, normalising it to canonical columns.
///
/// The doublet-specific wrapper over [`obs_import`]. Everything the generic
/// importer does — key-joined, in place, `pyscx.rollback`-able, `null` for cells
/// the tool did not cover — plus the one thing that needs per-tool knowledge:
/// each caller names its score and call differently, and downstream consensus
/// code should not have to branch on which tool ran.
///
/// Emits, for `key_added="<K>"` (defaulting to the tool name):
///
/// * `obs["<K>_score"]` — f32, nullable, higher = more doublet-like
/// * `obs["<K>_predicted"]` — bool, nullable; **omitted** for a tool that emits
///   no call (scds), because thresholding a score is a scientific decision this
///   importer does not own
/// * `obs["<K>_status"]` — `"present"` / `"absent"`
/// * `obs["<K>_<native>"]` — every other source column, unchanged
/// * `uns["<K>"]` — tool, resolved source columns, join report
///
/// The canonical names match what a native SCX doublet run would write, so an
/// imported result and a native one are drop-in comparable.
#[pyfunction]
#[pyo3(signature = (
    path, table, *, tool, key=None, source_key=None, key_added=None, score_column=None,
    call_column=None, call_true=None, call_false=None, keep_native_columns=true,
    delimiter=None, uns_keys=None, overwrite=false, on_missing_rows="null",
    on_extra_rows="warn", dry_run=false
))]
#[allow(clippy::too_many_arguments)]
pub fn doublet_import(
    py: Python<'_>,
    path: &str,
    table: &str,
    tool: &str,
    key: Option<Vec<String>>,
    source_key: Option<Vec<String>>,
    key_added: Option<&str>,
    score_column: Option<&str>,
    call_column: Option<&str>,
    call_true: Option<&str>,
    call_false: Option<&str>,
    keep_native_columns: bool,
    delimiter: Option<&str>,
    uns_keys: Option<Vec<String>>,
    overwrite: bool,
    on_missing_rows: &str,
    on_extra_rows: &str,
    dry_run: bool,
) -> PyResult<Py<PyDict>> {
    let delimiter_byte = match delimiter {
        None => None,
        Some(s) => {
            let bytes = s.as_bytes();
            if bytes.len() != 1 {
                return Err(PyValueError::new_err(format!(
                    "delimiter must be exactly one byte; got {s:?}"
                )));
            }
            Some(bytes[0])
        }
    };

    // Resolve the profile up front so a typo'd tool name fails before any I/O,
    // and so `key_added` can default to the profile's own name.
    let profile = scx_convert::doublet_profile(tool).map_err(ops_to_pyerr)?;
    let resolved_key_added = key_added
        .filter(|s| !s.is_empty())
        .unwrap_or(profile.name)
        .to_string();

    let (key_columns, join_key) = resolve_join_key(key, source_key)?;

    let read_opts = scx_convert::DoubletImportOptions {
        tool: tool.to_string(),
        key_added: resolved_key_added.clone(),
        score_column: score_column.map(str::to_string),
        call_column: call_column.map(str::to_string),
        call_true: call_true.map(str::to_string),
        call_false: call_false.map(str::to_string),
        keep_native_columns,
        key_columns,
        delimiter: delimiter_byte,
        uns_keys: uns_keys.unwrap_or_default(),
    };
    let attach_opts = scx_ops::AttachObsOptions {
        join_key,
        missing_row_policy: parse_obs_missing_rows(on_missing_rows)?,
        extra_row_policy: parse_obs_extra_rows(on_extra_rows)?,
        // Both are part of the canonical contract rather than knobs: `_status`
        // says which cells the tool actually covered, and `uns` records what it
        // was. `obs_import` remains the surface where they are optional.
        status_column: Some(format!("{resolved_key_added}_status")),
        uns_key: Some(resolved_key_added.clone()),
        overwrite,
        provenance_action: "doublet_import".to_string(),
        dry_run,
        ..Default::default()
    };

    let scx_path = PathBuf::from(path);
    let table_path = PathBuf::from(table);

    let (summary, info, diagnosis) = py.detach(|| -> PyResult<_> {
        let (data, info) =
            scx_convert::read_doublet_table(&table_path, &read_opts).map_err(ops_to_pyerr)?;
        let summary =
            scx_ops::attach_external_obs(&scx_path, &data, &attach_opts).map_err(ops_to_pyerr)?;
        let diagnosis = if dry_run {
            scx_ops::diagnose_obs_key(&scx_path, Some(&attach_opts.join_key)).ok()
        } else {
            None
        };
        Ok((summary, info, diagnosis))
    })?;

    // The profile declares a call column and the table carried none of its
    // spellings, so `<K>_predicted` was NOT written even though this tool does
    // emit a call. Warn HERE (after `py.detach` closes — `warnings.warn` is
    // unreachable inside it) rather than leave the user to discover it at
    // `doublet_consensus`, which is where the dogfood run found it, under the
    // false claim that the tool emits no call column. In-module pattern: see
    // `process_index_summary`.
    if let Some(m) = &info.call_column_missing {
        // `m.expected` already renders aliases AND prefix in one phrase
        // (built by `resolve_column`), so do not re-append the prefix.
        let expected = &m.expected;
        // A near-miss is usually the user having named the wrong `--tool`, so
        // point at the actual column when exactly one unconsumed column is a
        // declared call spelling of some other profile.
        let candidates: Vec<String> = m
            .present_columns
            .iter()
            .filter(|c| {
                scx_convert::DOUBLET_PROFILE_NAMES.iter().any(|t| {
                    scx_convert::doublet_profile(t)
                        .map(|p| p.call_columns.contains(&c.as_str()))
                        .unwrap_or(false)
                })
            })
            .cloned()
            .collect();
        let suggestion = if candidates.len() == 1 {
            format!(
                " The table does carry {:?}, which is another tool's call column — \
                 pass call_column={:?} if that is your call.",
                candidates[0], candidates[0]
            )
        } else {
            String::new()
        };
        let key = &info.key_added;
        let msg = format!(
            "doublet_import: tool={:?} declares a call column ({expected}) but the table has \
             none of those names — columns present are {:?}. Imported SCORE ONLY: \
             obs[{:?}] was written, obs[{:?}] was NOT, so this tool cannot vote on a call in \
             pyscx.doublet_consensus. The unmatched column is preserved verbatim under the \
             {:?} prefix.{suggestion} Re-import with call_column=<your column>, or pass the \
             tool= whose profile matches this table.",
            info.tool,
            m.present_columns,
            format!("{key}_score"),
            format!("{key}_predicted"),
            key,
        );
        py.import("warnings")?.call_method1("warn", (msg,))?;
    }

    let d = PyDict::new(py);
    d.set_item("tool", &info.tool)?;
    d.set_item("key_added", &info.key_added)?;
    d.set_item("score_source_column", info.score_source_column)?;
    d.set_item("call_source_column", &info.call_source_column)?;
    // Lets a script branch without parsing the warning string. Mirrors the
    // `call_column_status` recorded in `uns["<K>"]`.
    d.set_item(
        "call_column_status",
        match (&info.call_source_column, &info.call_column_missing) {
            (Some(_), _) => "resolved",
            (None, None) => "not_declared",
            (None, Some(_)) => "declared_but_absent",
        },
    )?;
    d.set_item(
        "expected_call_columns",
        info.call_column_missing
            .as_ref()
            .map(|m| m.expected_columns.clone())
            .unwrap_or_default(),
    )?;
    d.set_item("canonical_columns", info.canonical_columns)?;
    d.set_item("native_columns", info.native_columns)?;
    d.set_item("dropped_alias_columns", info.dropped_alias_columns)?;
    d.set_item("n_rows_in_source", info.table.n_rows)?;
    d.set_item("format", info.table.format.as_str())?;
    d.set_item(
        "delimiter",
        info.table.delimiter.map(|b| (b as char).to_string()),
    )?;
    d.set_item("uns_keys_imported", info.table.uns_keys_imported)?;
    d.set_item("renamed_index_column", info.table.renamed_index_column)?;
    d.set_item("source_key_columns", info.table.key_columns)?;
    d.set_item("dry_run", dry_run)?;
    d.set_item("n_obs", summary.n_obs)?;
    d.set_item("n_matched", summary.n_matched)?;
    d.set_item("n_target_rows_absent", summary.n_target_rows_absent)?;
    d.set_item("n_source_rows_absent", summary.n_source_rows_absent)?;
    d.set_item(
        "obs_key_column",
        scx_ops::display_key_name("obs", &summary.obs_key_column),
    )?;
    d.set_item("obs_columns_added", summary.obs_columns_added)?;
    d.set_item("obs_index_dropped", summary.obs_index_dropped)?;
    if let Some(diag) = diagnosis {
        d.set_item("key_diagnosis", key_diagnosis_dict(py, &diag)?)?;
    }
    Ok(d.into())
}

/// The valid `tool=` values for [`doublet_import`], in table order.
#[pyfunction]
pub fn doublet_tools() -> Vec<String> {
    scx_convert::DOUBLET_PROFILE_NAMES
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Every `tool=` profile's column vocabulary, straight from the definitions.
///
/// Exists so the per-tool table in `docs/scanpy.md` is machine-checkable rather
/// than hand-maintained, and so a user surprised by an import can look up what
/// their `--tool` actually expects from a REPL instead of reading Rust. That
/// lookup being unavailable is what made a call-column name mismatch a
/// silent score-only import.
#[pyfunction]
pub fn doublet_profiles(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let out = PyDict::new(py);
    for name in scx_convert::DOUBLET_PROFILE_NAMES {
        let p = scx_convert::doublet_profile(name).map_err(ops_to_pyerr)?;
        let d = PyDict::new(py);
        d.set_item("score_columns", p.score_columns.to_vec())?;
        d.set_item("score_prefix", p.score_prefix)?;
        d.set_item("call_columns", p.call_columns.to_vec())?;
        d.set_item("call_prefix", p.call_prefix)?;
        d.set_item("call_tokens", p.call_tokens.map(|t| (t.doublet, t.singlet)))?;
        // `!call_columns.is_empty() || call_prefix.is_some()` — the one fact
        // that decides whether `<K>_predicted` can exist at all.
        d.set_item("emits_call", scx_convert::profile_has_call_column(p))?;
        out.set_item(*name, d)?;
    }
    Ok(out.into())
}

/// Report which obs columns could serve as a join key for `obs_import`.
///
/// Read-only. Useful when an import fails on a duplicated key: on a merged
/// atlas the obvious candidates are often *not* unique and the one that is may
/// be a column no fallback list would guess.
#[pyfunction]
#[pyo3(signature = (path, key=None))]
pub fn diagnose_obs_key(
    py: Python<'_>,
    path: &str,
    key: Option<Vec<String>>,
) -> PyResult<Py<PyDict>> {
    let (_, join_key) = resolve_join_key(key, None)?;
    let scx_path = PathBuf::from(path);
    let diag = py
        .detach(|| scx_ops::diagnose_obs_key(&scx_path, Some(&join_key)))
        .map_err(ops_to_pyerr)?;
    Ok(key_diagnosis_dict(py, &diag)?.into())
}
