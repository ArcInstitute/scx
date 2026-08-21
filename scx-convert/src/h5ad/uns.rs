//! `/uns` export: JSON → HDF5.
//!
//! Split out of `h5ad/write.rs` as part of ORG-11.16-6. Nothing else in the
//! h5ad writer reaches into it and it reaches back only for
//! [`crate::h5_write_util::vlu`], so it is the cleanest seam in the file — the
//! recursive object walk, the `__scx_type__` envelope decode and the typed /
//! 2-D dataset emitters are one concern with one entry point,
//! [`write_uns_entries`].

use crate::h5_write_util::vlu;
use crate::pipeline::ConvertError;
use hdf5::types::VarLenUnicode;
use scx_format_io::SERDE_JSON_MAX_NESTING;

pub(crate) fn write_uns_entries_at(
    group: &hdf5::Group,
    value: &serde_json::Value,
) -> Result<(), ConvertError> {
    // `group` is the `/uns` group: container level 1.
    write_uns_entries(group, value, 1)
}

/// `depth` counts container levels already committed to, `group` included, so
/// the caller seeds it with 1 for the `/uns` group itself.
///
/// The bound is [`SERDE_JSON_MAX_NESTING`], not `MAX_UNS_DEPTH`: this is the
/// *consuming* side, and its input is either a tree `serde_json` already
/// parsed (so at most that deep) or one a capped producer built. Binding a
/// consumer to the producer cap instead would make an SCX file written before
/// that cap existed — legal, readable, up to 127 levels — suddenly
/// unexportable.
pub(super) fn write_uns_entries(
    group: &hdf5::Group,
    value: &serde_json::Value,
    depth: usize,
) -> Result<(), ConvertError> {
    if let serde_json::Value::Object(map) = value {
        for (key, val) in map {
            write_uns_value(group, key, val, depth)?;
        }
    }
    Ok(())
}

fn write_uns_value(
    group: &hdf5::Group,
    name: &str,
    value: &serde_json::Value,
    depth: usize,
) -> Result<(), ConvertError> {
    match value {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                group
                    .new_dataset::<i64>()
                    .shape(())
                    .create(name)?
                    .write_scalar(&i)?;
            } else if let Some(f) = n.as_f64() {
                group
                    .new_dataset::<f64>()
                    .shape(())
                    .create(name)?
                    .write_scalar(&f)?;
            }
        }
        serde_json::Value::String(s) => {
            let v = vlu(s);
            group
                .new_dataset::<VarLenUnicode>()
                .shape(())
                .create(name)?
                .write_scalar(&v)?;
        }
        serde_json::Value::Bool(b) => {
            group
                .new_dataset::<bool>()
                .shape(())
                .create(name)?
                .write_scalar(b)?;
        }
        serde_json::Value::Array(arr) => {
            // 2-D numeric arrays (e.g. color/contrast matrices) — mirror the
            // nested-JSON shape produced by `read_uns_entry`'s 2-D arm so they
            // round-trip rather than being silently dropped. Ragged / empty /
            // non-numeric nested arrays fall through to the no-op skip (a true
            // HDF5 round-trip is always rectangular, so ragged only arises from
            // synthetic JSON).
            if !arr.is_empty() && arr.iter().all(|v| v.is_array()) {
                write_uns_2d_array(group, name, arr)?;
            } else if arr.iter().all(|v| v.is_i64()) {
                let data: Vec<i64> = arr.iter().filter_map(|v| v.as_i64()).collect();
                group
                    .new_dataset::<i64>()
                    .shape([data.len()])
                    .create(name)?
                    .write(&data)?;
            } else if !arr.is_empty() && arr.iter().all(|v| v.is_u64()) {
                // Unsigned values above i64::MAX (C3): keep them exact instead
                // of coercing to f64 via the `is_number` arm below.
                let data: Vec<u64> = arr.iter().filter_map(|v| v.as_u64()).collect();
                group
                    .new_dataset::<u64>()
                    .shape([data.len()])
                    .create(name)?
                    .write(&data)?;
            } else if arr.iter().all(|v| v.is_number()) {
                let data: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
                group
                    .new_dataset::<f64>()
                    .shape([data.len()])
                    .create(name)?
                    .write(&data)?;
            } else if !arr.is_empty() && arr.iter().all(|v| v.is_boolean()) {
                // 1-D boolean (C3): the scalar bool arm above confirms the
                // `bool` H5Type; without this arm bool vectors are dropped.
                let data: Vec<bool> = arr.iter().filter_map(|v| v.as_bool()).collect();
                group
                    .new_dataset::<bool>()
                    .shape([data.len()])
                    .create(name)?
                    .write(&data)?;
            } else if arr.iter().all(|v| v.is_string()) {
                let data: Vec<VarLenUnicode> =
                    arr.iter().filter_map(|v| v.as_str()).map(vlu).collect();
                group
                    .new_dataset::<VarLenUnicode>()
                    .shape([data.len()])
                    .create(name)?
                    .write(&data)?;
            }
        }
        serde_json::Value::Object(map) => {
            // A tagged `__scx_type__` envelope (emitted by `read_uns_entry`'s
            // `uns_ndarray_envelope` for NaN/Inf-carrying floats and rank ≥ 3
            // arrays) decodes back to a native HDF5 dataset. Anything else —
            // a plain dict, or a pyscx-only envelope (recarray / tuple /
            // pandas.*) — falls through to the generic subgroup recursion.
            if !try_write_uns_envelope(group, name, map)? {
                // Refuse before descending: the subgroup would be level
                // `depth + 1`, and this walk is recursive, so an unbounded
                // tree overflows the stack and aborts the process.
                if depth >= SERDE_JSON_MAX_NESTING {
                    return Err(ConvertError::UnsTooDeep {
                        path: name.to_string(),
                        max_depth: SERDE_JSON_MAX_NESTING,
                    });
                }
                let subgroup = group.create_group(name)?;
                write_uns_entries(&subgroup, value, depth + 1)?;
            }
        }
        serde_json::Value::Null => {
            // anndata represents a Python `None` uns value as an `h5py.Empty`
            // dataset: an HDF5 null dataspace (0 elements) tagged
            // `encoding-type="null"`. Emit the same encoding so the value
            // round-trips as `None` rather than being silently dropped (and
            // read back as a bogus `0.0`, which breaks e.g. scanpy's
            // `uns['log1p']['base']`). Matches the null-dataspace detection
            // in `read_uns_entry`.
            let ds = group
                .new_dataset::<f32>()
                .shape(hdf5::Extents::null())
                .create(name)?;
            ds.new_attr::<VarLenUnicode>()
                .create("encoding-type")?
                .write_scalar(&vlu("null"))?;
            ds.new_attr::<VarLenUnicode>()
                .create("encoding-version")?
                .write_scalar(&vlu("0.1.0"))?;
        }
    }
    Ok(())
}

/// Decode a tagged `__scx_type__` `uns` envelope back to a native HDF5
/// dataset. Handles the numeric `ndarray` (`encoding == "base64le"`) and
/// `scalar` envelopes emitted by `read_uns_entry::uns_ndarray_envelope`
/// (the inverse of pyscx's `encode_ndarray_tagged` / `encode_np_scalar_tagged`).
/// Returns `Ok(false)` for a non-envelope object or any other tag/encoding so
/// the caller falls back to the generic subgroup recursion (no regression for
/// pyscx-only `json` / `recarray` / `tuple` / `pandas.*` envelopes).
fn try_write_uns_envelope(
    group: &hdf5::Group,
    name: &str,
    map: &serde_json::Map<String, serde_json::Value>,
) -> Result<bool, ConvertError> {
    let tag = match map
        .get(crate::h5ad::read::SCX_UNS_TYPE_KEY)
        .and_then(|v| v.as_str())
    {
        Some(t) => t,
        None => return Ok(false),
    };
    let (dtype, shape): (&str, Vec<usize>) = match tag {
        "ndarray" => {
            if map.get("encoding").and_then(|v| v.as_str()) != Some("base64le") {
                return Ok(false); // json-encoded string/object array — not ours
            }
            let dtype = match map.get("dtype").and_then(|v| v.as_str()) {
                Some(d) => d,
                None => return Ok(false),
            };
            // Strict: every dimension must be a valid non-negative integer.
            // A `null` / negative / fractional dim would otherwise be silently
            // dropped, producing a wrong-rank dataset with reinterpreted bytes.
            // Fall back to the generic subgroup write instead (preserves the
            // raw envelope, never corrupts).
            let shape: Vec<usize> = match map.get("shape").and_then(|v| v.as_array()) {
                Some(a) => match a
                    .iter()
                    .map(|v| v.as_u64().map(|n| n as usize))
                    .collect::<Option<Vec<usize>>>()
                {
                    Some(s) => s,
                    None => return Ok(false),
                },
                None => return Ok(false),
            };
            (dtype, shape)
        }
        // A `scalar` envelope is a 0-d value (empty shape).
        "scalar" => match map.get("dtype").and_then(|v| v.as_str()) {
            Some(d) => (d, Vec::new()),
            None => return Ok(false),
        },
        _ => return Ok(false),
    };

    let b64 = match map.get("data").and_then(|v| v.as_str()) {
        Some(d) => d,
        None => return Ok(false),
    };
    use base64::Engine;
    let bytes = match base64::engine::general_purpose::STANDARD.decode(b64) {
        Ok(b) => b,
        // Corrupt base64 — fall back rather than abort the whole export.
        Err(_) => return Ok(false),
    };

    // Returns false (→ generic fallback) for any dtype/order/itemsize we can't
    // faithfully decode; Err only on a genuine HDF5 write failure.
    write_uns_envelope_dataset(group, name, dtype, &shape, &bytes)
}

/// Reinterpret little-endian `bytes` as `dtype` and write a native HDF5
/// dataset of the given `shape` (empty `shape` → a 0-d scalar). Inverse of
/// the `to_le_bytes` packing in `uns_ndarray_envelope`.
///
/// Returns `Ok(true)` when the dataset was written, `Ok(false)` when the
/// envelope can't be faithfully decoded (unsupported byte order / dtype /
/// itemsize, or a byte count that isn't a multiple of itemsize) so the caller
/// falls back to writing the raw envelope as a subgroup — never aborting the
/// export and never reinterpreting bytes at the wrong width. `Err` is reserved
/// for a genuine HDF5 write failure.
fn write_uns_envelope_dataset(
    group: &hdf5::Group,
    name: &str,
    dtype: &str,
    shape: &[usize],
    bytes: &[u8],
) -> Result<bool, ConvertError> {
    // numpy `dtype.str`: byte-order char, kind char, then itemsize.
    let mut chars = dtype.chars();
    let order = chars.next();
    let kind = chars.next();
    let itemsize: usize = chars.as_str().parse().unwrap_or(0);
    // We only emit little-endian (`<`) or single-byte (`|`); `=` is native
    // (LE on supported hosts). Anything else (e.g. a big-endian `>f4` from a
    // pyscx-tagged AnnData uns) falls back to the generic subgroup write rather
    // than risking a byte-swapped misread or aborting the export.
    if !matches!(order, Some('<') | Some('|') | Some('=')) {
        return Ok(false);
    }

    macro_rules! emit {
        ($t:ty, $w:expr, $conv:expr) => {{
            let chunks = bytes.chunks_exact($w);
            if !chunks.remainder().is_empty() {
                // Truncated / corrupt payload — fall back instead of writing a
                // dataset from a partial buffer.
                return Ok(false);
            }
            let vals: Vec<$t> = chunks.map($conv).collect();
            write_typed_uns_dataset::<$t>(group, name, shape, vals)?;
            Ok(true)
        }};
    }

    match (kind, itemsize) {
        (Some('f'), 2) => emit!(half::f16, 2, |c: &[u8]| half::f16::from_le_bytes([
            c[0], c[1]
        ])),
        (Some('f'), 4) => emit!(f32, 4, |c: &[u8]| f32::from_le_bytes(c.try_into().unwrap())),
        (Some('f'), 8) => emit!(f64, 8, |c: &[u8]| f64::from_le_bytes(c.try_into().unwrap())),
        (Some('i'), 1) => emit!(i8, 1, |c: &[u8]| c[0] as i8),
        (Some('i'), 2) => emit!(i16, 2, |c: &[u8]| i16::from_le_bytes(c.try_into().unwrap())),
        (Some('i'), 4) => emit!(i32, 4, |c: &[u8]| i32::from_le_bytes(c.try_into().unwrap())),
        (Some('i'), 8) => emit!(i64, 8, |c: &[u8]| i64::from_le_bytes(c.try_into().unwrap())),
        (Some('u'), 1) => emit!(u8, 1, |c: &[u8]| c[0]),
        (Some('u'), 2) => emit!(u16, 2, |c: &[u8]| u16::from_le_bytes(c.try_into().unwrap())),
        (Some('u'), 4) => emit!(u32, 4, |c: &[u8]| u32::from_le_bytes(c.try_into().unwrap())),
        (Some('u'), 8) => emit!(u64, 8, |c: &[u8]| u64::from_le_bytes(c.try_into().unwrap())),
        (Some('b'), 1) => emit!(bool, 1, |c: &[u8]| c[0] != 0),
        // Unsupported kind/itemsize (datetime, complex, …) — fall back.
        _ => Ok(false),
    }
}

/// Write `vals` as a 0-d scalar (empty `shape`) or an N-D HDF5 dataset.
fn write_typed_uns_dataset<T: hdf5::H5Type>(
    group: &hdf5::Group,
    name: &str,
    shape: &[usize],
    vals: Vec<T>,
) -> Result<(), ConvertError> {
    if shape.is_empty() {
        if vals.len() != 1 {
            return Err(ConvertError::Other(format!(
                "uns envelope '{name}': scalar expected 1 element, got {}",
                vals.len()
            )));
        }
        group
            .new_dataset::<T>()
            .shape(())
            .create(name)?
            .write_scalar(&vals[0])?;
    } else {
        let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), vals)
            .map_err(|e| ConvertError::Other(format!("uns envelope '{name}': shape: {e}")))?;
        group
            .new_dataset::<T>()
            .shape(shape)
            .create(name)?
            .write(&arr)?;
    }
    Ok(())
}

/// Write a 2-D numeric `uns` array (nested JSON `[[..],[..]]`) as a rectangular
/// HDF5 dataset, mirroring the dtype set of `read_uns_entry`'s 2-D arm
/// (Integer/Unsigned/Float/Boolean). Ragged, empty, or non-numeric inputs are
/// skipped (no-op) rather than errored, matching the lenient behavior of the
/// surrounding scalar/1-D arms. The caller guarantees every element is an array.
fn write_uns_2d_array(
    group: &hdf5::Group,
    name: &str,
    arr: &[serde_json::Value],
) -> Result<(), ConvertError> {
    let rows: Vec<&Vec<serde_json::Value>> = arr.iter().filter_map(|v| v.as_array()).collect();
    let n_rows = rows.len();
    let n_cols = rows[0].len();
    // Rectangular and non-degenerate, else skip.
    if n_cols == 0 || rows.iter().any(|r| r.len() != n_cols) {
        return Ok(());
    }
    let cells = || rows.iter().flat_map(|r| r.iter());

    // Detect element dtype over all cells, mirroring the read-path ordering:
    // i64 first (catches negatives + small ints), then u64 (large unsigned),
    // then f64, then bool. Anything else (mixed / non-numeric) is skipped.
    if cells().all(|v| v.is_i64()) {
        let flat: Vec<i64> = cells().filter_map(|v| v.as_i64()).collect();
        write_2d_dataset::<i64>(group, name, n_rows, n_cols, flat)
    } else if cells().all(|v| v.is_u64()) {
        let flat: Vec<u64> = cells().filter_map(|v| v.as_u64()).collect();
        write_2d_dataset::<u64>(group, name, n_rows, n_cols, flat)
    } else if cells().all(|v| v.is_number()) {
        let flat: Vec<f64> = cells().filter_map(|v| v.as_f64()).collect();
        write_2d_dataset::<f64>(group, name, n_rows, n_cols, flat)
    } else if cells().all(|v| v.is_boolean()) {
        let flat: Vec<bool> = cells().filter_map(|v| v.as_bool()).collect();
        write_2d_dataset::<bool>(group, name, n_rows, n_cols, flat)
    } else {
        Ok(())
    }
}

/// Create and write a rectangular `n_rows × n_cols` HDF5 dataset from a
/// row-major flattened buffer. Shared by the `write_uns_2d_array` type arms;
/// mirrors the `ndarray::Array2::from_shape_vec` idiom in `write_obsm_entry`.
fn write_2d_dataset<T: hdf5::H5Type>(
    group: &hdf5::Group,
    name: &str,
    n_rows: usize,
    n_cols: usize,
    flat: Vec<T>,
) -> Result<(), ConvertError> {
    let nd = ndarray::Array2::from_shape_vec((n_rows, n_cols), flat)
        .map_err(|e| ConvertError::Other(format!("ndarray shape error: {e}")))?;
    group
        .new_dataset::<T>()
        .shape([n_rows, n_cols])
        .create(name)?
        .write(&nd)?;
    Ok(())
}

#[cfg(test)]
mod uns_envelope_tests {
    use super::*;

    /// A malformed or non-decodable `__scx_type__` envelope must fall back to
    /// the generic subgroup write (preserving the raw fields) rather than
    /// aborting the whole h5ad export — Antigravity #1 (bad shape) + Codex P2
    /// (unsupported byte order). `write_uns_value` must return `Ok` and create
    /// a subgroup for each case.
    #[test]
    fn malformed_envelopes_fall_back_to_subgroup() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("t.h5")).unwrap();
        let uns = file.create_group("uns").unwrap();

        let env = |extra: &[(&str, serde_json::Value)]| {
            let mut m = serde_json::Map::new();
            m.insert("__scx_type__".into(), "ndarray".into());
            m.insert("encoding".into(), "base64le".into());
            m.insert("dtype".into(), "<f8".into());
            m.insert("shape".into(), serde_json::json!([2]));
            m.insert("data".into(), serde_json::json!("AAAAAAAA8D8AAAAAAAAAQA==")); // [1.0, 2.0]
            for (k, v) in extra {
                m.insert((*k).into(), v.clone());
            }
            serde_json::Value::Object(m)
        };

        // (1) missing `data` → fallback.
        let mut no_data = env(&[]);
        no_data.as_object_mut().unwrap().remove("data");
        write_uns_value(&uns, "no_data", &no_data, 1).unwrap();

        // (2) non-integer shape dim → fallback (no wrong-rank dataset).
        write_uns_value(
            &uns,
            "bad_shape",
            &env(&[("shape", serde_json::json!([2, null]))]),
            1,
        )
        .unwrap();

        // (3) big-endian byte order → fallback (no abort, no byte-swap misread).
        write_uns_value(&uns, "big_endian", &env(&[("dtype", ">f8".into())]), 1).unwrap();

        // Each fell back to a subgroup that preserved the raw envelope fields.
        for key in ["no_data", "bad_shape", "big_endian"] {
            let g = uns
                .group(key)
                .unwrap_or_else(|_| panic!("'{key}' should fall back to a subgroup"));
            assert!(
                g.dataset("dtype").is_ok(),
                "'{key}' fallback should preserve the raw envelope fields"
            );
        }

        // A well-formed little-endian envelope still writes a real dataset
        // (not a subgroup), confirming the fallback is scoped to the bad cases.
        write_uns_value(&uns, "good", &env(&[]), 1).unwrap();
        assert!(
            uns.dataset("good").is_ok(),
            "valid envelope should write a dataset"
        );
        assert!(
            uns.group("good").is_err(),
            "valid envelope must not be a subgroup"
        );
    }
}
