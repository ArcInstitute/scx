//! File-operations Python bindings — wraps scx-ops for Python.
//!
//! Split into operation-family submodules (ORG-10.16-6): `append_delete` (append /
//! append_from_anndata / mark_deleted), `rewrite` (the copy-out rewrites:
//! compact, optimize, sort, shuffle, build_csc, rollback, merge),
//! `metadata` (in-place set_uns / modify_metadata), `obs_attach` (the
//! external-obs join family: obs_import, attach_obs_columns,
//! diagnose_obs_key), `tools` (CellBender + doublet-caller interop). This
//! module keeps the shared kwarg/error/arrow helpers and re-exports every
//! `#[pyfunction]` so lib.rs's `ops::<fn>` registration paths are unchanged.

use std::num::NonZeroU32;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use arrow::record_batch::RecordBatch;

use scx_codec::{CodecId, CodecSelection, ValueEncoding};
use scx_engine::{BuildOutcome, ConversionPredicateIndexOptions, SkipReason};
use scx_format_io::ScxReader;

use scx_ops::{OpsError, PredicateIndexBuildSummary};

use crate::convert;

pub(crate) mod append_delete;
pub(crate) mod metadata;
pub(crate) mod obs_attach;
pub(crate) mod rewrite;
pub(crate) mod tools;

pub(crate) use append_delete::*;
pub(crate) use metadata::*;
pub(crate) use obs_attach::*;
pub(crate) use rewrite::*;
pub(crate) use tools::*;

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
        // 1000 mirrors `scx-convert::pipeline::IngestOptions::default`.
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
        crate::pyimport::import_module(py, "warnings")?.call_method1("warn", (msg,))?;
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
pub(crate) fn ops_to_pyerr(e: OpsError) -> PyErr {
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

/// Convert an obs/var input (pandas `DataFrame` or pyarrow `Table`) to an
/// Arrow `RecordBatch`. A `Table` is routed through `to_pandas()` so the
/// shared `pandas_to_record_batch` IPC path handles both.
fn obs_var_to_record_batch(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    func: &str,
    param: &str,
) -> PyResult<RecordBatch> {
    let pa = crate::pyimport::import_module(py, "pyarrow")?;
    let table_cls = pa.getattr("Table")?;
    if obj.is_instance(&table_cls)? {
        let df = obj.call_method0("to_pandas")?;
        return convert::pandas_to_record_batch(py, &df);
    }
    // Require a pandas DataFrame. Without this guard a dict (a natural thing to
    // try) falls through to `pyarrow.Table.from_pandas` and surfaces an opaque
    // `AttributeError: 'dict' object has no attribute 'columns'` deep inside
    // pyarrow, with no mention of the calling function, the parameter, or the
    // expected type (report E2).
    let pd = crate::pyimport::import_module(py, "pandas")?;
    let df_cls = pd.getattr("DataFrame")?;
    if !obj.is_instance(&df_cls)? {
        let got = obj
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "object".to_string());
        return Err(PyTypeError::new_err(format!(
            "{func}({param}=...) expects a pandas DataFrame or pyarrow Table \
             (got {got}); wrap your columns with pd.DataFrame({{...}})."
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
