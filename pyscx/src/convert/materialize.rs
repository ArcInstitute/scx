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

/// Fail loud (as a Python `ValueError`) when decoding an integer shard whose
/// `value_max` exceeds `f32`'s exact-integer range would silently round, and the
/// caller has not passed `allow_lossy=True`.
///
/// scx's Phase-1 read decodes every shard to an `f32` CSR before any dtype
/// materialization, so on-disk `u32` counts above 2²⁴ are silently rounded
/// regardless of the requested output dtype. `max_value` is the max
/// `ShardStats::value_max` over the shards in scope of the read (see
/// [`super::csr_max_value`] for the eager path, or `QueryResult::max_value` for
/// the query path). Float-encoded shards record `value_max = 0`, so they never
/// trip this.
pub(crate) fn guard_decode_loss(max_value: u32, allow_lossy: bool) -> PyResult<()> {
    scx_codec::guard_f32_decode_loss(max_value, allow_lossy)
        .map_err(|e| PyValueError::new_err(e.to_string()))
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
    let container = parse_container(container)?;
    let data_dtype = parse_value_dtype(data_dtype)?;
    let resolved_index = parse_index_dtype(index_dtype)?;

    // Dense output has no column-index array; warn if the caller asked for one.
    if container == Container::Dense && index_dtype.is_some() {
        let warnings = py.import("warnings")?;
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
