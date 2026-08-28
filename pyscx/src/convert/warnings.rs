// Conversion-warning bridging to Python warnings.
//
// Extracted from the former pyscx/src/anndata.rs (T5.7).

use pyo3::prelude::*;

/// Forward a [`scx_convert::ConvertWarning`] to Python's `warnings.warn`
/// as a `UserWarning`, with the category name prefixed so consumers
/// can filter on it. Mirrors the per-key warning surface used by the
/// streaming pipeline; the in-memory `from_anndata` path doesn't wire
/// a [`WarningSink`] today so we emit live instead of summarising.
pub(crate) fn warn_python_convert(py: Python<'_>, w: &scx_convert::ConvertWarning) -> PyResult<()> {
    let warnings_mod = crate::pyimport::import_module(py, "warnings")?;
    let user_warning = crate::pyimport::import_module(py, "builtins")?.getattr("UserWarning")?;
    let msg = format!("{}: {}", w.category(), w);
    warnings_mod.getattr("warn")?.call1((msg, user_warning))?;
    Ok(())
}

/// Forward each non-empty category in `sink` as a single
/// `warnings.warn(..., UserWarning)` call on the Python side.
///
/// The conversion itself runs under `py.detach`, so emission
/// happens after the GIL is reacquired. One Python-side warning per
/// category (with its aggregate count) is enough for Phase 0; per-
/// emission forwarding would require holding the GIL across the
/// whole conversion.
#[cfg(feature = "hdf5")]
pub(crate) fn emit_python_warnings(
    py: Python<'_>,
    sink: &scx_convert::WarningSink,
) -> PyResult<()> {
    if sink.total() == 0 {
        return Ok(());
    }
    let warnings_mod = crate::pyimport::import_module(py, "warnings")?;
    let warn = warnings_mod.getattr("warn")?;
    let user_warning = crate::pyimport::import_module(py, "builtins")?.getattr("UserWarning")?;
    for (cat, count) in sink.counts() {
        let msg = format!("scx conversion: {count} warning(s) of type '{cat}'");
        warn.call1((msg, user_warning.clone()))?;
    }
    Ok(())
}

/// Warn (Python `UserWarning`) that a CSC sidecar on the source is
/// being dropped on rewrite. Matches the convention documented in
/// `AGENTS.md`'s "CSC storage" bullet — mutating ops drop the
/// sidecar by default; callers opt into a rebuild via `csc="always"`.
pub(crate) fn warn_csc_dropped(py: Python<'_>) {
    let msg = "source SCX has a CSC sidecar; the rewrite drops it. \
               Pass csc=\"always\" to rebuild a fresh CSC sidecar over the new CSR shards.";
    let _ =
        crate::pyimport::import_module(py, "warnings").and_then(|w| w.call_method1("warn", (msg,)));
}
