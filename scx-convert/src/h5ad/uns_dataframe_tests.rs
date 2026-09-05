//! Tests for the `/uns` `pandas.DataFrame` export arm.
//!
//! Extracted to a sibling file per the crate convention for large `#[cfg(test)]`
//! modules; `#[path]`-included so `super::*` still reaches the private items.

use super::*;

/// The envelope a pyscx-written frame actually looks like, so the tests
/// exercise the shape the decoder meets in the wild rather than one invented
/// to suit it.
fn frame_envelope() -> serde_json::Value {
    serde_json::json!({
        "__scx_type__": "pandas.DataFrame",
        "index": {
            "__scx_type__": "pandas.Index",
            "name": "row",
            "data": {
                "__scx_type__": "ndarray", "dtype": "object", "shape": [2],
                "encoding": "json", "data": ["r0", "r1"],
            },
        },
        // Deliberately not alphabetical: `column-order` is the only carrier of
        // column order, so a writer that iterated `data`'s members instead
        // would reorder the frame silently.
        "columns": ["zed", "abe", "grade"],
        "data": {
            // [1.0, 2.0] as little-endian f64.
            "zed": {
                "__scx_type__": "ndarray", "dtype": "<f8", "shape": [2],
                "encoding": "base64le", "data": "AAAAAAAA8D8AAAAAAAAAQA==",
            },
            // [3, 4] as little-endian i64.
            "abe": {
                "__scx_type__": "ndarray", "dtype": "<i8", "shape": [2],
                "encoding": "base64le", "data": "AwAAAAAAAAAEAAAAAAAAAA==",
            },
            "grade": {
                "__scx_type__": "categorical",
                "ordered": true,
                // codes [1, 0] as int8.
                "codes": {
                    "__scx_type__": "ndarray", "dtype": "|i1", "shape": [2],
                    "encoding": "base64le", "data": "AQA=",
                },
                "categories": {
                    "__scx_type__": "ndarray", "dtype": "object", "shape": [3],
                    "encoding": "json", "data": ["lo", "mid", "hi"],
                },
            },
        },
    })
}

/// One mutation of the reference envelope, named so the case table does not
/// trip `clippy::type_complexity`.
type Mutate = fn(&mut serde_json::Value);

fn fresh_uns(dir: &tempfile::TempDir, name: &str) -> (hdf5::File, hdf5::Group) {
    let file = hdf5::File::create(dir.path().join(format!("{name}.h5"))).unwrap();
    let uns = file.create_group("uns").unwrap();
    (file, uns)
}

fn read_str_attr(obj: &hdf5::Group, name: &str) -> String {
    obj.attr(name)
        .unwrap()
        .read_scalar::<hdf5::types::VarLenUnicode>()
        .unwrap()
        .to_string()
}

/// The frame arm writes the anndata dataframe layout, not a raw subgroup.
///
/// Asserting the on-disk attributes rather than a Python round trip is the
/// point: `encoding-type`, `_index` and `column-order` are exactly what anndata
/// dispatches on, and they are what the pre-X6 generic path never wrote.
#[test]
fn frame_envelope_writes_an_anndata_dataframe_group() {
    let dir = tempfile::tempdir().unwrap();
    let (_f, uns) = fresh_uns(&dir, "ok");
    let mut sink = WarningSink::log();

    let env = frame_envelope();
    assert!(
        try_write_uns_dataframe(&uns, "tbl", env.as_object().unwrap(), &mut sink).unwrap(),
        "a well-formed frame envelope must be claimed by this arm"
    );

    let g = uns.group("tbl").unwrap();
    assert_eq!(read_str_attr(&g, "encoding-type"), "dataframe");
    assert_eq!(read_str_attr(&g, "_index"), "row");

    let order: Vec<String> = g
        .attr("column-order")
        .unwrap()
        .read_1d::<hdf5::types::VarLenUnicode>()
        .unwrap()
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(
        order,
        ["zed", "abe", "grade"],
        "column-order must follow the envelope's `columns`, not `data`'s member order"
    );

    assert_eq!(g.dataset("row").unwrap().shape(), [2]);
    assert_eq!(
        g.dataset("zed").unwrap().read_raw::<f64>().unwrap(),
        [1.0, 2.0]
    );
    assert_eq!(g.dataset("abe").unwrap().read_raw::<i64>().unwrap(), [3, 4]);

    let cat = g.group("grade").unwrap();
    assert_eq!(read_str_attr(&cat, "encoding-type"), "categorical");
    assert!(cat.attr("ordered").unwrap().read_scalar::<bool>().unwrap());
    assert_eq!(
        cat.dataset("codes").unwrap().read_raw::<i32>().unwrap(),
        [1, 0]
    );
    // Declared categories survive whole: no row uses "mid", and ingest applies
    // no row filter, so pruning it would be wrong.
    let cats: Vec<String> = cat
        .dataset("categories")
        .unwrap()
        .read_1d::<hdf5::types::VarLenUnicode>()
        .unwrap()
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(cats, ["lo", "mid", "hi"]);
}

/// Narrow dtypes keep their exact width, and `bool` lands as a plain `bool`
/// dataset.
///
/// Both are regressions from the first implementation of this module, which
/// routed the frame through the obs/var column writer: that widened `int8` to
/// `int32` and spelled every `bool` as `encoding-type: "nullable-boolean"` —
/// which this crate's own `uns` ingest cannot read, so `to_h5ad` → `from_h5ad`
/// dropped the column. Found by Cursor Agent.
#[test]
fn narrow_and_boolean_columns_keep_their_plain_dtypes() {
    let dir = tempfile::tempdir().unwrap();
    let (_f, uns) = fresh_uns(&dir, "narrow");
    let mut sink = WarningSink::log();

    let mut env = frame_envelope();
    env["columns"] = serde_json::json!(["i8", "flag"]);
    env["data"] = serde_json::json!({
        "i8":   {"__scx_type__": "ndarray", "dtype": "|i1", "shape": [2],
                 "encoding": "base64le", "data": "Af4="},          // [1, -2]
        "flag": {"__scx_type__": "ndarray", "dtype": "|b1", "shape": [2],
                 "encoding": "base64le", "data": "AQA="},          // [true, false]
    });
    assert!(try_write_uns_dataframe(&uns, "tbl", env.as_object().unwrap(), &mut sink).unwrap());

    let g = uns.group("tbl").unwrap();
    assert_eq!(g.dataset("i8").unwrap().read_raw::<i8>().unwrap(), [1, -2]);
    assert_eq!(
        g.dataset("flag").unwrap().read_raw::<bool>().unwrap(),
        [true, false]
    );
    // Plain datasets, not the nullable groups the obs/var writer emits.
    assert!(
        g.group("flag").is_err(),
        "bool must not become a nullable group"
    );
    assert!(g.group("i8").is_err());
}

/// An unnamed index must land as `_index`, anndata's own spelling.
#[test]
fn unnamed_index_lands_as_underscore_index() {
    let dir = tempfile::tempdir().unwrap();
    let (_f, uns) = fresh_uns(&dir, "unnamed");
    let mut sink = WarningSink::log();

    let mut env = frame_envelope();
    env["index"]["name"] = serde_json::Value::Null;
    assert!(try_write_uns_dataframe(&uns, "tbl", env.as_object().unwrap(), &mut sink).unwrap());

    let g = uns.group("tbl").unwrap();
    assert_eq!(read_str_attr(&g, "_index"), "_index");
    assert!(g.dataset("_index").is_ok());
}

/// A column-less frame still writes a length-0 `column-order`.
///
/// anndata raises `KeyError: 'column-order'` without the attribute, and treats
/// a scalar there as malformed — the same case the obs/var header writer
/// carries an explicit `.shape()` for.
#[test]
fn column_less_frame_writes_an_empty_column_order() {
    let dir = tempfile::tempdir().unwrap();
    let (_f, uns) = fresh_uns(&dir, "nocols");
    let mut sink = WarningSink::log();

    let mut env = frame_envelope();
    env["columns"] = serde_json::json!([]);
    env["data"] = serde_json::json!({});
    assert!(try_write_uns_dataframe(&uns, "tbl", env.as_object().unwrap(), &mut sink).unwrap());

    let g = uns.group("tbl").unwrap();
    assert_eq!(g.attr("column-order").unwrap().shape(), [0]);
    assert!(g.dataset("row").is_ok());
}

/// Everything this arm cannot spell must **decline**, write nothing, and warn.
///
/// Declining is the contract, not a failure mode: the caller then writes the
/// raw envelope subgroup, which preserves all the data. The pre-review version
/// got two of these wrong in the worst possible direction — a boolean
/// categorical was dropped while the arm reported success, and an index sharing
/// a column's name aborted the entire `to_h5ad` with an opaque HDF5 error.
#[test]
fn unwritable_frames_decline_write_nothing_and_warn() {
    let dir = tempfile::tempdir().unwrap();
    let (_f, uns) = fresh_uns(&dir, "decline");

    let cases: &[(&str, Mutate)] = &[
        ("wrong_tag", |e| {
            e["__scx_type__"] = "something.else".into();
        }),
        ("no_columns", |e| {
            e.as_object_mut().unwrap().remove("columns");
        }),
        ("column_not_in_data", |e| {
            e["columns"] = serde_json::json!(["zed", "ghost"]);
        }),
        ("length_mismatch", |e| {
            e["data"]["zed"]["shape"] = serde_json::json!([1]);
            e["data"]["zed"]["data"] = "AAAAAAAA8D8=".into();
            e["columns"] = serde_json::json!(["zed"]);
        }),
        ("big_endian_column", |e| {
            e["data"]["zed"]["dtype"] = ">f8".into();
            e["columns"] = serde_json::json!(["zed"]);
        }),
        ("two_dimensional_column", |e| {
            e["data"]["zed"]["shape"] = serde_json::json!([1, 2]);
            e["columns"] = serde_json::json!(["zed"]);
        }),
        // A plain h5ad string dataset has no null; writing `""` would read
        // back as a different value.
        ("null_in_string_column", |e| {
            e["columns"] = serde_json::json!(["s"]);
            e["data"] = serde_json::json!({
                "s": {"__scx_type__": "ndarray", "dtype": "object", "shape": [2],
                      "encoding": "json", "data": ["x", null]},
            });
        }),
        // `read_categorical_values` cannot read boolean categories back, so
        // writing them would produce a file this workspace cannot ingest.
        ("boolean_categorical", |e| {
            e["columns"] = serde_json::json!(["grade"]);
            e["data"]["grade"]["categories"] = serde_json::json!({
                "__scx_type__": "ndarray", "dtype": "|b1", "shape": [2],
                "encoding": "base64le", "data": "AAE=",
            });
        }),
        // Legal pandas (`df.index.name == "zed"` with a "zed" column); both
        // land in one HDF5 group, so the names would collide.
        ("index_name_collides_with_column", |e| {
            e["index"]["name"] = "zed".into();
        }),
    ];

    for (name, mutate) in cases {
        let mut env = frame_envelope();
        mutate(&mut env);
        let mut sink = WarningSink::log();
        assert!(
            !try_write_uns_dataframe(&uns, name, env.as_object().unwrap(), &mut sink).unwrap(),
            "'{name}' must decline, not claim the value"
        );
        assert!(
            uns.group(name).is_err() && uns.dataset(name).is_err(),
            "'{name}' must not have written anything before declining"
        );
        // Only a genuine frame that could not be written is worth a warning;
        // a value that was never a frame envelope is not this arm's business.
        let warned = sink.counts().get("uns_exported_as_raw_envelope").copied();
        if *name == "wrong_tag" {
            assert_eq!(warned, None, "a non-frame envelope must not warn here");
        } else {
            assert_eq!(warned, Some(1), "'{name}' must warn that it was demoted");
        }
    }
}
