// Python-surface parsing for the F3 materialization kwargs.
//
// Turns the `container` / `data_dtype` / `index_dtype` / `allow_lossy` read
// kwargs into a `scx_sparse::MaterializePlan`, with validation and actionable
// error messages.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use scx_sparse::{Container, IndexDtype, MaterializePlan, ValueDtype};

/// Parse the `container` kwarg into a `Container`.
pub(crate) fn parse_container(c: &str) -> PyResult<Container> {
    match c.to_ascii_lowercase().as_str() {
        "csr" => Ok(Container::Csr),
        "dense" => Ok(Container::Dense),
        other => Err(PyValueError::new_err(format!(
            "container must be 'csr' or 'dense'; got {other:?}"
        ))),
    }
}

/// Parse the `data_dtype` kwarg. `None` → `float32` (today's default).
pub(crate) fn parse_value_dtype(dt: Option<&str>) -> PyResult<ValueDtype> {
    let Some(dt) = dt else {
        return Ok(ValueDtype::F32);
    };
    match dt.to_ascii_lowercase().as_str() {
        "float16" | "f16" | "half" => Ok(ValueDtype::F16),
        "float32" | "f32" => Ok(ValueDtype::F32),
        "float64" | "f64" | "double" => Ok(ValueDtype::F64),
        "int8" | "i8" => Ok(ValueDtype::I8),
        "int16" | "i16" => Ok(ValueDtype::I16),
        "int32" | "i32" => Ok(ValueDtype::I32),
        "int64" | "i64" => Ok(ValueDtype::I64),
        "uint8" | "u8" => Ok(ValueDtype::U8),
        "uint16" | "u16" => Ok(ValueDtype::U16),
        "uint32" | "u32" => Ok(ValueDtype::U32),
        other => Err(PyValueError::new_err(format!(
            "data_dtype: unsupported dtype {other}; expected a numeric float/int/uint dtype \
             (float16/float32/float64/int8/int16/int32/int64/uint8/uint16/uint32)."
        ))),
    }
}

/// Parse the `data_dtype` kwarg, keeping "not given" distinguishable from
/// "float32".
///
/// [`parse_value_dtype`] resolves `None` to `f32`, which is right everywhere the
/// read *is* the decode. On a query result it is not: `collect(data_dtype=…)`
/// already decoded at some dtype, and resolving a later `data_dtype=None` to
/// `f32` would narrow a `float64` buffer back down and refuse the very read the
/// caller just asked for.
pub(crate) fn parse_value_dtype_opt(dt: Option<&str>) -> PyResult<Option<ValueDtype>> {
    match dt {
        None => Ok(None),
        Some(_) => parse_value_dtype(dt).map(Some),
    }
}

/// Parse the `index_dtype` kwarg (CSR only). `None` → `int32` (today's default).
pub(crate) fn parse_index_dtype(dt: Option<&str>) -> PyResult<IndexDtype> {
    let Some(dt) = dt else {
        return Ok(IndexDtype::I32);
    };
    match dt.to_ascii_lowercase().as_str() {
        "int16" | "i16" => Ok(IndexDtype::I16),
        "int32" | "i32" => Ok(IndexDtype::I32),
        "int64" | "i64" => Ok(IndexDtype::I64),
        other => Err(PyValueError::new_err(format!(
            "index_dtype: unsupported dtype {other}; expected int16/int32/int64."
        ))),
    }
}

/// The decode-loss guard for a **layer** read: `f32`-bound, because layers are
/// decoded as `f32` on every path.
///
/// Named for the axis rather than the dtype on purpose. It used to be the
/// generic `guard_decode_loss`, and that name is how it ended up guarding `X` on
/// the query path too — where the requested dtype does now change the answer.
/// A layer has no typed reader yet, so here the `f32` limit really is the limit
/// and no `data_dtype=` can move it; the message says so, which the shared codec
/// string no longer can (it is also `rscx`'s, and `rscx` has no dtype at all).
///
/// `max_value` is the max `ShardStats::value_max` over the layers in scope.
/// Float-encoded shards record `value_max = 0`, so they never trip this.
pub(crate) fn guard_decode_loss_layers(max_value: u32, allow_lossy: bool) -> PyResult<()> {
    scx_codec::guard_f32_decode_loss(max_value, allow_lossy).map_err(|e| {
        PyValueError::new_err(format!(
            "{e} Layers are decoded as float32 on every path (a typed layer reader is not \
             yet available), so no `data_dtype=` can make this read exact."
        ))
    })
}

/// Dtype-aware decode-loss guard for the in-decode narrow path: fails loud iff
/// the shards' `max_value` cannot be represented exactly in the *target*
/// `dtype`. Error-mapping wrapper over the shared
/// [`scx_format_io::guard_decode_loss_dtype`] dispatch. Used where the decode
/// itself narrows (the eager `X` / `raw` / per-modality readers), so one check
/// covers both loss steps; the query result's two-step rule is
/// [`materialize_guard`].
pub(crate) fn guard_decode_loss_dtype(
    max_value: u32,
    dtype: ValueDtype,
    allow_lossy: bool,
) -> PyResult<()> {
    scx_format_io::guard_decode_loss_dtype(max_value, dtype, allow_lossy)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Which of the two decode-loss steps a materialization request fails at.
///
/// A value can be lost twice on the way to a caller: once when the shards are
/// decoded into whatever buffer holds them, and again when that buffer is cast
/// to the dtype the caller named. Naming the step is what makes the error
/// actionable — "too narrow" wants a wider dtype, "too late" wants the dtype
/// declared before the decode instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GuardFailure {
    /// The requested dtype itself cannot hold `max_value`. No decode order helps.
    TooNarrow {
        max_value: u32,
        requested: ValueDtype,
    },
    /// The requested dtype could have held `max_value`, but the decode that
    /// already happened could not.
    TooLate {
        max_value: u32,
        assembled: ValueDtype,
        requested: ValueDtype,
    },
}

/// The whole decode-loss rule for a materialization, in one pure function.
///
/// `assembled` is the dtype the values are *already* stored at; `requested` is
/// what the caller wants back. Extracted rather than inlined at each call site
/// because the defect this replaces was four sites that each had to get the same
/// rule right and one of them being wrong was invisible — a rule with a test
/// beats four copies with none.
///
/// Order matters: **too-narrow beats too-late.** Asking for `float16` on a 55 M
/// count is a narrower problem than asking too late, and telling that caller to
/// "re-collect at float16" would be wrong advice.
///
/// Float-encoded shards record `value_max = 0`, so they never trip either step;
/// the per-element cast gate in `scx_codec::checked_cast_*` is what covers them.
pub(crate) fn materialize_guard(
    max_value: u32,
    assembled: ValueDtype,
    requested: ValueDtype,
    allow_lossy: bool,
) -> Result<(), GuardFailure> {
    if allow_lossy {
        return Ok(());
    }
    if max_value > max_lossless_u32(requested) {
        return Err(GuardFailure::TooNarrow {
            max_value,
            requested,
        });
    }
    if max_value > max_lossless_u32(assembled) {
        return Err(GuardFailure::TooLate {
            max_value,
            assembled,
            requested,
        });
    }
    Ok(())
}

/// Largest `u32` the dtype represents exactly, from `scx_codec`'s own
/// `CastFromU32::MAX_LOSSLESS_U32` — never a second table here.
fn max_lossless_u32(dtype: ValueDtype) -> u32 {
    use scx_codec::CastFromU32;
    match dtype {
        ValueDtype::F16 => <half::f16 as CastFromU32>::MAX_LOSSLESS_U32,
        ValueDtype::F32 => <f32 as CastFromU32>::MAX_LOSSLESS_U32,
        ValueDtype::F64 => <f64 as CastFromU32>::MAX_LOSSLESS_U32,
        ValueDtype::I8 => <i8 as CastFromU32>::MAX_LOSSLESS_U32,
        ValueDtype::I16 => <i16 as CastFromU32>::MAX_LOSSLESS_U32,
        ValueDtype::I32 => <i32 as CastFromU32>::MAX_LOSSLESS_U32,
        ValueDtype::I64 => <i64 as CastFromU32>::MAX_LOSSLESS_U32,
        ValueDtype::U8 => <u8 as CastFromU32>::MAX_LOSSLESS_U32,
        ValueDtype::U16 => <u16 as CastFromU32>::MAX_LOSSLESS_U32,
        ValueDtype::U32 => <u32 as CastFromU32>::MAX_LOSSLESS_U32,
    }
}

/// Render a [`GuardFailure`] as the `ValueError` a caller sees.
pub(crate) fn guard_failure_to_pyerr(f: GuardFailure) -> PyErr {
    PyValueError::new_err(guard_failure_message(f))
}

/// The text of a [`GuardFailure`], as a plain `String`.
///
/// Separate from the `PyErr` so it can be unit-tested: pyscx builds as an
/// extension module with no libpython to link, so a test that touches
/// `Python::attach` cannot run here — and a message that names the wrong remedy
/// is precisely the defect being fixed, so it needs a test.
///
/// The remedy sentence lives here, not in `scx-codec`: only the binding knows
/// whether the caller has a `collect()` to re-run. `scx-codec`'s own string is
/// also `rscx`'s, and `rscx` has no `data_dtype` at all.
pub(crate) fn guard_failure_message(f: GuardFailure) -> String {
    match f {
        GuardFailure::TooNarrow {
            max_value,
            requested,
        } => format!(
            "integer value {max_value} exceeds {limit}, the largest integer representable \
             exactly in {name}; this read would lose precision. Use a wider `data_dtype` or \
             pass `allow_lossy=True`.",
            limit = max_lossless_u32(requested),
            name = requested.numpy_name(),
        ),
        GuardFailure::TooLate {
            max_value,
            assembled,
            requested,
        } => format!(
            "integer value {max_value} exceeds {limit}, the largest integer representable \
             exactly in {assembled_name}; this result was decoded as {assembled_name}, so \
             returning it as {requested_name} would report {assembled_name}-rounded values at \
             the wider dtype. Re-run the query with \
             `.collect(data_dtype=\"{requested_name}\")` to decode losslessly, or pass \
             `allow_lossy=True` to accept the rounding.",
            limit = max_lossless_u32(assembled),
            assembled_name = assembled.numpy_name(),
            requested_name = requested.numpy_name(),
        ),
    }
}

/// Build a `MaterializePlan` from the four read kwargs, applying validation and
/// the "index_dtype ignored for dense" warning.
pub(crate) fn build_plan(
    py: Python<'_>,
    container: &str,
    data_dtype: Option<&str>,
    index_dtype: Option<&str>,
    allow_lossy: bool,
) -> PyResult<MaterializePlan> {
    build_plan_with_default(
        py,
        container,
        data_dtype,
        index_dtype,
        allow_lossy,
        ValueDtype::F32,
    )
}

/// [`build_plan`] with a caller-supplied meaning for `data_dtype=None`.
///
/// Only the query result needs this: its `None` means "the dtype this result was
/// decoded at", not `float32`. Every other caller wants `float32` and goes
/// through [`build_plan`].
pub(crate) fn build_plan_with_default(
    py: Python<'_>,
    container: &str,
    data_dtype: Option<&str>,
    index_dtype: Option<&str>,
    allow_lossy: bool,
    default_data_dtype: ValueDtype,
) -> PyResult<MaterializePlan> {
    let container = parse_container(container)?;
    let data_dtype = parse_value_dtype_opt(data_dtype)?.unwrap_or(default_data_dtype);
    let resolved_index = parse_index_dtype(index_dtype)?;

    // Dense output has no column-index array; warn if the caller asked for one.
    if container == Container::Dense && index_dtype.is_some() {
        let warnings = crate::pyimport::import_module(py, "warnings")?;
        warnings.call_method1(
            "warn",
            (
                "index_dtype is ignored for container='dense' (dense output has no column indices)",
                py.get_type::<pyo3::exceptions::PyRuntimeWarning>(),
            ),
        )?;
    }

    Ok(MaterializePlan {
        container,
        data_dtype,
        index_dtype: resolved_index,
        allow_lossy,
    })
}

/// Build a 1-D numpy array of the plan's `data_dtype` from an `f32` value stream,
/// applying the fail-loud cast gate. Zero-copy `from_vec` move for every dtype
/// (numpy's `half` feature gives native `f16`).
pub(crate) fn f32_values_to_numpy<'py>(
    py: Python<'py>,
    data: &[f32],
    dtype: ValueDtype,
    allow_lossy: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use numpy::PyArray1;
    use scx_codec::checked_cast_values;

    macro_rules! cast_arm {
        ($t:ty) => {{
            let v: Vec<$t> = checked_cast_values(data, allow_lossy).map_err(to_pyerr)?;
            Ok(PyArray1::from_vec(py, v).into_any())
        }};
    }
    match dtype {
        ValueDtype::F16 => cast_arm!(half::f16),
        ValueDtype::F32 => cast_arm!(f32),
        ValueDtype::F64 => cast_arm!(f64),
        ValueDtype::I8 => cast_arm!(i8),
        ValueDtype::I16 => cast_arm!(i16),
        ValueDtype::I32 => cast_arm!(i32),
        ValueDtype::I64 => cast_arm!(i64),
        ValueDtype::U8 => cast_arm!(u8),
        ValueDtype::U16 => cast_arm!(u16),
        ValueDtype::U32 => cast_arm!(u32),
    }
}

/// Build a 1-D numpy array of the plan's `index_dtype` from an `i32` index stream.
pub(crate) fn i32_indices_to_numpy<'py>(
    py: Python<'py>,
    indices: &[i32],
    dtype: IndexDtype,
    allow_lossy: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use numpy::PyArray1;
    use scx_codec::checked_cast_indices;

    match dtype {
        IndexDtype::I16 => {
            let v: Vec<i16> = checked_cast_indices(indices, allow_lossy).map_err(to_pyerr)?;
            Ok(PyArray1::from_vec(py, v).into_any())
        }
        IndexDtype::I32 => Ok(PyArray1::from_vec(py, indices.to_vec()).into_any()),
        IndexDtype::I64 => {
            let v: Vec<i64> = checked_cast_indices(indices, allow_lossy).map_err(to_pyerr)?;
            Ok(PyArray1::from_vec(py, v).into_any())
        }
    }
}

/// Reshape a flat 1-D numpy value array into a `(rows, cols)` 2-D dense array.
pub(crate) fn reshape_2d<'py>(
    flat: Bound<'py, PyAny>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let py = flat.py();
    let shape = PyTuple::new(py, [rows, cols])?;
    flat.call_method1("reshape", (shape,))
}

fn to_pyerr(e: scx_codec::CodecError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_codec::F32_MAX_EXACT_INT;

    /// Odd, above 2²⁴, and therefore **not** f32-exact. The even neighbour every
    /// Python fixture can write (`20_000_000`) is f32-exact, so it cannot tell a
    /// typed decode from an f32 detour — which is exactly why this rule needs a
    /// unit test rather than only end-to-end coverage.
    const BIG_ODD: u32 = (1 << 24) + 7;

    #[test]
    fn the_constant_is_not_f32_exact() {
        assert_ne!(BIG_ODD as f32 as u32, BIG_ODD);
        assert_eq!(20_000_000f32 as u32, 20_000_000);
    }

    #[test]
    fn a_wide_request_after_an_f32_decode_is_too_late() {
        // The defect this whole change exists to fix: `float64` *can* hold the
        // value, but the f32 decode already happened. Answering `Ok` here is
        // what the naive guard swap would have done.
        assert_eq!(
            materialize_guard(BIG_ODD, ValueDtype::F32, ValueDtype::F64, false),
            Err(GuardFailure::TooLate {
                max_value: BIG_ODD,
                assembled: ValueDtype::F32,
                requested: ValueDtype::F64,
            })
        );
        for requested in [ValueDtype::U32, ValueDtype::I64] {
            assert!(matches!(
                materialize_guard(BIG_ODD, ValueDtype::F32, requested, false),
                Err(GuardFailure::TooLate { .. })
            ));
        }
    }

    #[test]
    fn a_wide_request_on_a_wide_decode_passes() {
        for dtype in [ValueDtype::F64, ValueDtype::U32, ValueDtype::I64] {
            assert_eq!(materialize_guard(BIG_ODD, dtype, dtype, false), Ok(()));
        }
    }

    #[test]
    fn too_narrow_beats_too_late() {
        // Both steps fail. The caller must be told the dtype cannot hold the
        // value at all — advising them to re-collect at `float16` would be
        // wrong, and that is the whole reason the order is fixed.
        assert_eq!(
            materialize_guard(BIG_ODD, ValueDtype::F32, ValueDtype::F16, false),
            Err(GuardFailure::TooNarrow {
                max_value: BIG_ODD,
                requested: ValueDtype::F16,
            })
        );
        assert!(matches!(
            materialize_guard(BIG_ODD, ValueDtype::F64, ValueDtype::U16, false),
            Err(GuardFailure::TooNarrow { .. })
        ));
    }

    #[test]
    fn the_plain_default_is_too_narrow_not_too_late() {
        // `to_csr()` with no kwargs on a `>2²⁴` file: assembled and requested
        // are both f32, so there is no later decode to recommend — the honest
        // advice is a wider dtype, which `TooNarrow` gives.
        assert_eq!(
            materialize_guard(BIG_ODD, ValueDtype::F32, ValueDtype::F32, false),
            Err(GuardFailure::TooNarrow {
                max_value: BIG_ODD,
                requested: ValueDtype::F32,
            })
        );
    }

    #[test]
    fn the_f32_bound_is_inclusive() {
        assert_eq!(
            materialize_guard(F32_MAX_EXACT_INT, ValueDtype::F32, ValueDtype::F32, false),
            Ok(())
        );
        assert!(materialize_guard(
            F32_MAX_EXACT_INT + 1,
            ValueDtype::F32,
            ValueDtype::F32,
            false
        )
        .is_err());
    }

    #[test]
    fn a_float_encoded_result_never_trips_either_step() {
        // Float-encoded shards record `value_max = 0`; the per-element cast gate
        // is what covers them.
        for requested in [ValueDtype::U8, ValueDtype::F16, ValueDtype::I8] {
            assert_eq!(
                materialize_guard(0, ValueDtype::F32, requested, false),
                Ok(())
            );
        }
    }

    #[test]
    fn allow_lossy_bypasses_both_steps() {
        for requested in [ValueDtype::F16, ValueDtype::F64, ValueDtype::U16] {
            assert_eq!(
                materialize_guard(u32::MAX, ValueDtype::F32, requested, true),
                Ok(())
            );
        }
    }

    #[test]
    fn the_limits_come_from_the_codec_not_a_second_table() {
        // A local copy of these numbers is how the two halves drift apart.
        use scx_codec::CastFromU32;
        assert_eq!(max_lossless_u32(ValueDtype::F32), F32_MAX_EXACT_INT);
        assert_eq!(max_lossless_u32(ValueDtype::F16), 2048);
        assert_eq!(max_lossless_u32(ValueDtype::U16), u32::from(u16::MAX));
        for dtype in [ValueDtype::F64, ValueDtype::U32, ValueDtype::I64] {
            assert_eq!(
                max_lossless_u32(dtype),
                u32::MAX,
                "{dtype:?} holds every u32"
            );
        }
        assert_eq!(
            max_lossless_u32(ValueDtype::I32),
            <i32 as CastFromU32>::MAX_LOSSLESS_U32
        );
    }

    #[test]
    fn the_messages_name_what_the_caller_needs() {
        let msg = guard_failure_message(GuardFailure::TooLate {
            max_value: BIG_ODD,
            assembled: ValueDtype::F32,
            requested: ValueDtype::F64,
        });
        assert!(msg.contains("float64"), "{msg}");
        assert!(msg.contains("collect(data_dtype="), "{msg}");
        assert!(msg.contains("allow_lossy=True"), "{msg}");
        // The clause that is only true of a layer read must not appear here.
        assert!(!msg.contains("not yet available"), "{msg}");

        let msg = guard_failure_message(GuardFailure::TooNarrow {
            max_value: BIG_ODD,
            requested: ValueDtype::F16,
        });
        assert!(msg.contains("float16"), "{msg}");
        assert!(msg.contains("2048"), "{msg}");
        assert!(msg.contains("wider `data_dtype`"), "{msg}");
        // Nothing to re-collect: the dtype itself is the problem.
        assert!(!msg.contains("collect(data_dtype="), "{msg}");
    }
}
