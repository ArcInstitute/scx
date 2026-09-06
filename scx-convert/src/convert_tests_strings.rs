//! scx-convert integration tests — HDF5 string ingest (`h5ad/strings.rs`).
//!
//! Every HDF5 string column on the ingest side used to be copied four times
//! before it reached Arrow: HDF5's own allocation, `Array1::to_vec()`'s
//! deep-copying `VarLenUnicode` clone, a `Vec<String>`, and a `Vec<&str>`
//! built only to give `StringArray::from_iter_values` a `size_hint`. It now
//! goes through one traversal into a pre-sized Arrow builder.
//!
//! The oracle for that is the construction being replaced. It is kept here,
//! in-test, and every assertion below compares against it — the same shape
//! PR-04 used when it diffed the streaming `/raw` writer against the eager
//! one. A value-only comparison is **not** enough: the two Arrow
//! constructors differ in whether they emit a validity buffer at all
//! (`from_iter_values` never does, `FromIterator<Option<_>>` always does), and
//! a null must not advance the offset. So the assertions are on
//! `value_offsets`, `value_data` and `nulls` explicitly, not just `to_data`.
//!
//! Two of these tests also cover a **behaviour widening**.
//! `read_categorical_values` and `read_nullable_string_group` both matched
//! `FixedUnicode(_) | FixedAscii(_)` while reading their bodies as
//! `VarLenUnicode`, so a PyTables-written `categories` or nullable `values`
//! dataset failed with HDF5's opaque "no conversion paths found" despite the
//! arm accepting it. Routing them through the shared traversal fixes that;
//! `fixed_width_categories_and_nullable_values_read_at_all` is the accept-side
//! test, and it fails on the pre-change code.

use super::convert_tests_common::*;
use arrow::array::{DictionaryArray, StringArray};
use arrow::datatypes::Int32Type;
use hdf5::types::{FixedAscii, FixedUnicode};

use crate::h5ad::read::read_dataframe_group;
use crate::h5ad::strings::{read_masked_string_array, read_string_array, read_string_dataset};

/// Values with every property that has ever broken a string reader: an empty
/// string first (so a null-vs-empty confusion shows at index 0), multi-byte
/// UTF-8 at two different widths, an emoji (4 bytes), significant leading and
/// trailing whitespace (so a reader that normalises is caught -- without it a
/// `trim()` mutation stays green), and a trailing empty. No fixture in this
/// crate carried a non-ASCII payload before.
const TRICKY: &[&str] = &["", "café", "日本語", "🧬", "  padded  ", "ünïcödé", ""];

fn varlen_ds(group: &hdf5::Group, name: &str, values: &[&str]) {
    let vals: Vec<VarLenUnicode> = values.iter().map(|s| vlu(s)).collect();
    group
        .new_dataset::<VarLenUnicode>()
        .shape([vals.len()])
        .create(name)
        .unwrap()
        .write(&vals)
        .unwrap();
}

/// `FixedAscii<N>` — what PyTables `create_carray` emits (CellRanger,
/// CellBender). ASCII only by construction.
fn fixed_ascii_ds<const N: usize>(group: &hdf5::Group, name: &str, values: &[&str]) {
    let vals: Vec<FixedAscii<N>> = values
        .iter()
        .map(|s| FixedAscii::<N>::from_ascii(s.as_bytes()).unwrap())
        .collect();
    group
        .new_dataset::<FixedAscii<N>>()
        .shape([vals.len()])
        .create(name)
        .unwrap()
        .write(&vals)
        .unwrap();
}

/// `FixedUnicode<N>` where `N` is a **byte** width, which is the whole reason
/// the width ladder is keyed on bytes: `"日本語"` is 3 chars and 9 bytes.
fn fixed_unicode_ds<const N: usize>(group: &hdf5::Group, name: &str, values: &[&str]) {
    let vals: Vec<FixedUnicode<N>> = values
        .iter()
        .map(|s| s.parse::<FixedUnicode<N>>().unwrap())
        .collect();
    group
        .new_dataset::<FixedUnicode<N>>()
        .shape([vals.len()])
        .create(name)
        .unwrap()
        .write(&vals)
        .unwrap();
}

fn attr(group: &hdf5::Group, name: &str, value: &str) {
    group
        .new_attr::<VarLenUnicode>()
        .create(name)
        .unwrap()
        .write_scalar(&vlu(value))
        .unwrap();
}

fn str_attr_array(group: &hdf5::Group, name: &str, values: &[&str]) {
    let vals: Vec<VarLenUnicode> = values.iter().map(|s| vlu(s)).collect();
    group
        .new_attr::<VarLenUnicode>()
        .shape([vals.len()])
        .create(name)
        .unwrap()
        .write(&vals)
        .unwrap();
}

/// A minimal anndata `/obs` group: unnamed index plus the named columns.
fn obs_group(file: &hdf5::File, n_rows: usize, columns: &[&str]) -> hdf5::Group {
    let obs = file.create_group("obs").unwrap();
    attr(&obs, "encoding-type", "dataframe");
    attr(&obs, "encoding-version", "0.2.0");
    attr(&obs, "_index", "_index");
    str_attr_array(&obs, "column-order", columns);
    let ids: Vec<String> = (0..n_rows).map(|i| format!("cell_{i}")).collect();
    let refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
    varlen_ds(&obs, "_index", &refs);
    obs
}

/// The pre-change read, transcribed. **Deliberately does not call anything in
/// `h5ad/strings.rs`** — an oracle routed through the new traversal would agree
/// with a shared extraction bug in it. This is `read_1d::<VarLenUnicode>()` +
/// `to_vec()` + `Vec<String>` + `Vec<&str>` + `from_iter_values`, exactly as
/// `read_string_dataset` and `read_column_to_arrow` did before this change.
///
/// Var-length only, which is all the pre-change code supported on the two
/// group readers; the fixed-width arms are pinned by literal expectations
/// instead (see `fixed_width_categories_and_nullable_values_read_at_all`).
fn legacy_plain_varlen(ds: &hdf5::Dataset) -> StringArray {
    let data: Vec<VarLenUnicode> = ds.read_1d().unwrap().to_vec();
    let strings: Vec<String> = data.iter().map(|s| s.to_string()).collect();
    StringArray::from(strings.iter().map(|s| s.as_str()).collect::<Vec<_>>())
}

/// The pre-change nullable-group read, transcribed, on the same terms:
/// `read_1d` + `to_vec()` + `Vec<String>` + `Vec<Option<&str>>` +
/// `FromIterator<Option<_>>`, touching nothing in `h5ad/strings.rs`.
fn legacy_masked_varlen(ds: &hdf5::Dataset, mask: &[bool]) -> StringArray {
    let data: Vec<VarLenUnicode> = ds.read_1d().unwrap().to_vec();
    let strings: Vec<String> = data.iter().map(|s| s.to_string()).collect();
    StringArray::from(
        strings
            .iter()
            .zip(mask.iter())
            .map(|(v, m)| if *m { None } else { Some(v.as_str()) })
            .collect::<Vec<Option<&str>>>(),
    )
}

/// How much slack a "was this buffer pre-sized?" assertion must allow.
///
/// `MutableBuffer::with_capacity` rounds up to 64-byte alignment, and
/// `Buffer::capacity()` reports the allocation, not the request — neither is a
/// promised Arrow contract. So these assertions bound the capacity rather than
/// equating it: exact equality holds on arrow 58 today but would break on an
/// allocator or `finish()` change with no production defect. **If Arrow's
/// rounding changes, widen this constant — do not delete the assertion**: it is
/// the only observable trace that the value buffer was allocated once instead
/// of grown by doubling, and it is what the two route pins rest on.
const CAPACITY_SLACK: usize = 64;

/// The value buffer holds exactly the payload (a real Arrow contract) and was
/// allocated once rather than grown (bounded, see [`CAPACITY_SLACK`]).
///
/// **Carries its own premise check**, and that is not decoration. When this
/// assertion was first loosened from an exact equality to a bounded one, both
/// route pins silently stopped discriminating: their fixtures were small
/// enough (39 B and ~50 B of payload) that a buffer grown by doubling landed
/// at 64 B, *inside* the slack. The tests still passed with the call sites
/// routed back to the old construction. So the helper now builds a
/// deliberately grown buffer over the same payload and refuses to run unless
/// that one actually violates the bound.
///
/// For null-free arrays only — it reconstructs the payload via `value(i)`.
fn assert_value_buffer_was_pre_sized(got: &StringArray, payload: usize, what: &str) {
    assert_eq!(
        got.value_data().len(),
        payload,
        "{what}: value buffer length must equal the payload"
    );

    let mut grown = arrow::array::GenericStringBuilder::<i32>::with_capacity(got.len(), 0);
    for i in 0..got.len() {
        grown.append_value(got.value(i));
    }
    let grown_capacity = grown.finish().to_data().buffers()[1].capacity();
    assert!(
        grown_capacity > payload + CAPACITY_SLACK,
        "{what}: PREMISE FAILED — a buffer grown from zero over this payload \
         reaches only {grown_capacity} against payload {payload} \
         (+{CAPACITY_SLACK} slack), so the assertion below cannot tell a \
         pre-sized buffer from a grown one. Enlarge the fixture."
    );

    let capacity = got.to_data().buffers()[1].capacity();
    assert!(
        capacity >= payload && capacity <= payload + CAPACITY_SLACK,
        "{what}: value buffer capacity {capacity} is not one allocation of \
         payload {payload} (+{CAPACITY_SLACK} slack) — it was grown, so this \
         array did not come through the pre-sized builder"
    );
}

fn assert_arrays_identical(got: &StringArray, want: &StringArray, what: &str) {
    assert_eq!(
        got.value_offsets(),
        want.value_offsets(),
        "{what}: i32 offset buffer differs"
    );
    assert_eq!(
        got.value_data(),
        want.value_data(),
        "{what}: value buffer bytes differ"
    );
    assert_eq!(
        got.nulls().map(|n| n.iter().collect::<Vec<_>>()),
        want.nulls().map(|n| n.iter().collect::<Vec<_>>()),
        "{what}: validity buffer differs (present-vs-absent counts)"
    );
    assert_eq!(got.to_data(), want.to_data(), "{what}: ArrayData differs");
}

#[test]
fn a_plain_string_column_is_byte_identical_to_the_vec_str_construction() {
    let dir = tempfile::tempdir().unwrap();
    let file = hdf5::File::create(dir.path().join("s.h5")).unwrap();
    let root = file.as_group().unwrap();
    varlen_ds(&root, "note", TRICKY);
    let ds = file.dataset("note").unwrap();

    let got = read_string_array(&ds).unwrap();
    let want = legacy_plain_varlen(&ds);

    assert_arrays_identical(&got, &want, "plain string column");
    // Not implied by the comparison: `from_iter_values` emits no validity
    // buffer at all, so a builder that ever reached `append_null` -- e.g. by
    // treating `""` as missing -- would still match on values and offsets
    // while producing a different array.
    assert!(
        got.nulls().is_none(),
        "a plain string column has no null representation on disk; the array \
         must carry no validity buffer, not an all-valid one"
    );
    assert_eq!(got.value(0), "", "an empty string is a value, not a null");
    assert_eq!(got.value(2), "日本語");
    assert_eq!(got.value(3), "🧬");
    assert_eq!(
        got.value(4),
        "  padded  ",
        "whitespace is data, not padding"
    );
    assert_eq!(got.len(), TRICKY.len());
}

#[test]
fn a_nullable_string_group_is_byte_identical_and_a_null_does_not_advance_the_offset() {
    let dir = tempfile::tempdir().unwrap();
    let file = hdf5::File::create(dir.path().join("s.h5")).unwrap();
    let root = file.as_group().unwrap();
    // anndata fills null positions with "" on disk and marks them in `mask`.
    // Nulls at the first, an interior, and the last row: the three positions
    // an off-by-one in the mask walk gets wrong.
    let values = ["", "café", "", "日本語", "🧬", ""];
    let mask = [true, false, false, true, false, true];
    varlen_ds(&root, "values", &values);
    let ds = file.dataset("values").unwrap();

    let got = read_masked_string_array(&ds, &mask, "note").unwrap();
    let want = legacy_masked_varlen(&ds, &mask);

    assert_arrays_identical(&got, &want, "nullable-string group");
    assert!(got.is_null(0) && got.is_null(3) && got.is_null(5));
    assert!(got.is_valid(1) && got.is_valid(2) && got.is_valid(4));
    // `""` at index 2 is a real value while index 0 is null: the distinction
    // the mask carries and a values-only assertion cannot see.
    assert_eq!(got.value(2), "");
    assert_eq!(got.value(3), "", "a null slot is zero-length, not skipped");
    assert_eq!(got.value(4), "🧬");
    // The offsets must not advance across a null. Derive the expectation from
    // the payload rather than restating the buffer, so the assertion still
    // means something if the fixture changes.
    let mut expected = vec![0i32];
    for (v, m) in values.iter().zip(mask.iter()) {
        let last = *expected.last().unwrap();
        expected.push(last + if *m { 0 } else { v.len() as i32 });
    }
    assert_eq!(got.value_offsets(), &expected[..]);
}

#[test]
fn the_utf8_value_buffer_is_allocated_once_not_grown() {
    // The only observable trace of the allocation win. `with_capacity(items,
    // bytes)` lands the value buffer at exactly the payload size; the
    // construction this replaces starts at `MutableBuffer::new(0)` and grows
    // by doubling, so it overshoots (measured: 16384 for a 16000-byte payload
    // at 1000 rows). A regression that dropped the byte count would show here
    // and nowhere else -- capacity is invisible to every value assertion.
    let dir = tempfile::tempdir().unwrap();
    let file = hdf5::File::create(dir.path().join("s.h5")).unwrap();
    let root = file.as_group().unwrap();
    // Every third row is multi-byte, so a reservation computed from
    // `chars().count()` instead of `len()` under-reserves and the buffer has
    // to grow -- the mutation this fixture exists to catch.
    let ids: Vec<String> = (0..1000)
        .map(|i| match i % 3 {
            0 => format!("barcode_{i:08}"),
            1 => format!("バーコード_{i:08}"),
            _ => format!("étiquette_{i:08}"),
        })
        .collect();
    let refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
    varlen_ds(&root, "bc", &refs);
    let ds = file.dataset("bc").unwrap();
    assert!(
        refs.iter().map(|s| s.len()).sum::<usize>()
            > refs.iter().map(|s| s.chars().count()).sum::<usize>(),
        "premise: the fixture must carry multi-byte payload"
    );

    let payload: usize = refs.iter().map(|s| s.len()).sum();
    let got = read_string_array(&ds).unwrap();
    assert_value_buffer_was_pre_sized(&got, payload, "pre-sized value buffer");

    // Premise: the construction being replaced really does overshoot the
    // bound, so the assertion above is discriminating rather than trivially
    // true.
    let legacy_capacity = legacy_plain_varlen(&ds).to_data().buffers()[1].capacity();
    assert!(
        legacy_capacity > payload + CAPACITY_SLACK,
        "premise failed: the `Vec<&str>` construction no longer overshoots \
         ({legacy_capacity} vs {payload}+{CAPACITY_SLACK}), so this test \
         proves nothing"
    );
}

#[test]
fn fixed_width_categories_and_nullable_values_read_at_all() {
    // ACCEPT-SIDE test for a behaviour widening, and the pre-change failure
    // is worse than an error: both of these columns are **silently dropped**
    // on `main`. `read_categorical_values` and `read_nullable_string_group`
    // matched the fixed-width descriptors in their arms but read the body as
    // `VarLenUnicode`, HDF5 refused the conversion, and
    // `read_dataframe_group` swallowed each failure into a warning. Measured
    // on `1017703d`, this exact fixture:
    //
    //   SkippedColumn { group: "obs", name: "kind",
    //       reason: "categorical group: HDF5 error: no conversion paths found" }
    //   SkippedColumn { group: "obs", name: "note",
    //       reason: "nullable-string-array group: HDF5 error: no conversion paths found" }
    //   columns present: ["plainfix", "__index_level_0__"]
    //
    // A PyTables writer (CellRanger, CellBender) emits exactly this shape, so
    // that was obs data loss on ingest with nothing but a warning to show it.
    // `plainfix` is the control: the plain-string path already handled fixed
    // widths, which is what localises the defect to the two group readers.
    let dir = tempfile::tempdir().unwrap();
    let file = hdf5::File::create(dir.path().join("obs.h5")).unwrap();
    let obs = obs_group(&file, 4, &["kind", "note", "plainfix"]);

    // categorical with FIXED-ASCII categories
    let cat = obs.create_group("kind").unwrap();
    attr(&cat, "encoding-type", "categorical");
    fixed_ascii_ds::<16>(&cat, "categories", &["alpha", "beta"]);
    cat.new_dataset::<i32>()
        .shape([4])
        .create("codes")
        .unwrap()
        .write(&[0i32, 1, 1, 0])
        .unwrap();

    // nullable-string-array with FIXED-ASCII values
    let nul = obs.create_group("note").unwrap();
    attr(&nul, "encoding-type", "nullable-string-array");
    attr(&nul, "encoding-version", "0.1.0");
    fixed_ascii_ds::<16>(&nul, "values", &["a", "", "c", ""]);
    nul.new_dataset::<u8>()
        .shape([4])
        .create("mask")
        .unwrap()
        .write(&[0u8, 1, 0, 1])
        .unwrap();

    // and a plain fixed-ascii column, which already worked -- it is the
    // control that keeps this test honest about which arm was broken.
    fixed_ascii_ds::<16>(&obs, "plainfix", &["w", "x", "y", "z"]);

    let skipped = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let recorder = std::sync::Arc::clone(&skipped);
    let mut sink = WarningSink::with_handler(move |w| {
        if let crate::warnings::ConvertWarning::SkippedColumn { name, reason, .. } = w {
            recorder.lock().unwrap().push(format!("{name}: {reason}"));
        }
    });
    let batch = read_dataframe_group(&file, "obs", &mut sink).unwrap();

    // The sharp assertion: nothing may reach the skip channel. Column
    // presence alone would still pass if a future reader emitted the warning
    // and then recovered the column some other way.
    assert!(
        skipped.lock().unwrap().is_empty(),
        "columns were skipped instead of read: {:?}",
        skipped.lock().unwrap()
    );

    let kind = batch
        .column_by_name("kind")
        .expect("the fixed-ascii categorical column must survive, not be skipped")
        .as_any()
        .downcast_ref::<DictionaryArray<Int32Type>>()
        .expect("kind is a dictionary");
    let cats = kind
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(cats.value(0), "alpha");
    assert_eq!(cats.value(1), "beta");

    let note = batch
        .column_by_name("note")
        .expect("the fixed-ascii nullable-string column must survive")
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(note.value(0), "a");
    assert!(note.is_null(1) && note.is_null(3));
    assert_eq!(note.value(2), "c");

    let plain = batch
        .column_by_name("plainfix")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(plain.value(0), "w");
    assert_eq!(plain.value(3), "z");
}

#[test]
fn the_width_ladder_is_keyed_on_bytes_not_characters() {
    // `"日本語"` is 3 chars / 9 bytes and `"🧬"` is 1 char / 4 bytes, so a
    // ladder that sized on character count would pick too narrow a read type
    // and HDF5 would truncate. Two widths, either side of the 16-byte rung.
    let dir = tempfile::tempdir().unwrap();
    let file = hdf5::File::create(dir.path().join("s.h5")).unwrap();
    let root = file.as_group().unwrap();
    let narrow = ["日本語", "🧬", "ab"]; // max 9 bytes -> the 16-rung
    let wide = ["ünïcödé-ünïcödé", "x"]; // 23 bytes, 15 chars -> the 32-rung
    fixed_unicode_ds::<12>(&root, "narrow", &narrow);
    fixed_unicode_ds::<32>(&root, "wide", &wide);

    let got_narrow = read_string_array(&file.dataset("narrow").unwrap()).unwrap();
    let got_wide = read_string_array(&file.dataset("wide").unwrap()).unwrap();

    assert_eq!(got_narrow.value(0), "日本語");
    assert_eq!(got_narrow.value(1), "🧬");
    assert_eq!(got_narrow.value(2), "ab");
    assert_eq!(got_wide.value(0), "ünïcödé-ünïcödé");
    assert_eq!(got_wide.value(1), "x");
    // Byte lengths, not char counts, are what the offsets encode.
    assert_eq!(got_narrow.value_offsets(), &[0i32, 9, 13, 15]);
}

#[test]
fn read_string_dataset_returns_the_same_owned_strings_on_every_flavour() {
    // Five call sites still take the `Vec<String>` -- CellBender's barcode
    // and feature join keys, and two `serde_json::Value::String` builders --
    // so the owned shape is a contract, not an implementation detail.
    let dir = tempfile::tempdir().unwrap();
    let file = hdf5::File::create(dir.path().join("s.h5")).unwrap();
    let root = file.as_group().unwrap();
    let ascii = ["", "abc", "de", ""];
    varlen_ds(&root, "varlen", TRICKY);
    fixed_ascii_ds::<16>(&root, "fixed_ascii", &ascii);
    fixed_unicode_ds::<32>(&root, "fixed_unicode", TRICKY);

    assert_eq!(
        read_string_dataset(&file.dataset("varlen").unwrap()).unwrap(),
        TRICKY.iter().map(|s| s.to_string()).collect::<Vec<_>>()
    );
    assert_eq!(
        read_string_dataset(&file.dataset("fixed_ascii").unwrap()).unwrap(),
        ascii.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        "trailing and leading empties must survive the fixed-width read"
    );
    assert_eq!(
        read_string_dataset(&file.dataset("fixed_unicode").unwrap()).unwrap(),
        TRICKY.iter().map(|s| s.to_string()).collect::<Vec<_>>()
    );
}

#[test]
fn a_mask_that_disagrees_with_the_dataset_length_is_rejected_before_the_read() {
    // Nothing drove this error arm before. It now fires off the dataset
    // *shape*, so a mismatched pair costs a shape query rather than a full
    // decode -- and the message still names the column and both lengths.
    let dir = tempfile::tempdir().unwrap();
    let file = hdf5::File::create(dir.path().join("s.h5")).unwrap();
    let root = file.as_group().unwrap();
    varlen_ds(&root, "values", &["a", "b", "c"]);
    let ds = file.dataset("values").unwrap();

    let err = read_masked_string_array(&ds, &[true, false], "note").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("nullable-string-array 'note'")
            && msg.contains("values len 3")
            && msg.contains("mask len 2"),
        "unexpected message: {msg}"
    );
}

#[test]
fn tenx_obs_and_var_strings_survive_the_builder_route() {
    // `test_tenx_to_scx` asserts the shape and that nnz > 0; nothing asserted
    // that the barcodes and feature columns arrive intact, which is what the
    // two routed sites in `tenx_read.rs` produce.
    // Sized, not arbitrary: the route pins need a payload a doubling-grown
    // buffer overshoots (see `assert_value_buffer_was_pre_sized`). 5 x 3 left
    // both pins vacuous.
    let (n_cells, n_genes) = (3000usize, 800usize);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.h5");
    create_test_tenx_h5(&path, n_cells, n_genes);
    let file = hdf5::File::open(&path).unwrap();
    let data = crate::tenx_read::read_tenx_h5(&file).unwrap();

    let bc = data
        .obs
        .column_by_name("barcode")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(bc.len(), n_cells);
    assert!(bc.nulls().is_none(), "10x barcodes are non-nullable");
    let first = bc.value(0).to_string();
    assert!(!first.is_empty(), "barcode 0 must not be empty: {first:?}");
    // ROUTE PIN, as in the obs test below: values cannot distinguish the two
    // constructions, the value buffer's allocation can.
    let payload: usize = (0..bc.len()).map(|i| bc.value(i).len()).sum();
    assert_value_buffer_was_pre_sized(bc, payload, "10x barcodes");

    for col in ["id", "name", "feature_type"] {
        let arr = data
            .var
            .column_by_name(col)
            .unwrap_or_else(|| panic!("var column {col} missing"))
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap_or_else(|| panic!("var column {col} is not Utf8"));
        assert_eq!(arr.len(), n_genes, "{col} length");
        assert!(arr.nulls().is_none(), "{col} must carry no validity buffer");
        assert!(
            (0..arr.len()).all(|i| !arr.value(i).is_empty()),
            "{col} has an empty value"
        );
        let payload: usize = (0..arr.len()).map(|i| arr.value(i).len()).sum();
        assert_value_buffer_was_pre_sized(arr, payload, &format!("10x var column {col}"));
    }
}

#[test]
fn a_multi_byte_obs_column_round_trips_through_read_dataframe_group() {
    // End-to-end through the public reader, on the payload no fixture in this
    // crate carried before this test.
    // TRICKY repeated: the values are what matters, but the route pin below
    // needs a payload large enough that a doubling-grown buffer overshoots the
    // slack. At `TRICKY.len()` rows it does not, and the pin was silently
    // vacuous until the helper's premise check caught it.
    let rows: Vec<&str> = TRICKY.iter().copied().cycle().take(3000).collect();
    let dir = tempfile::tempdir().unwrap();
    let file = hdf5::File::create(dir.path().join("obs.h5")).unwrap();
    let obs = obs_group(&file, rows.len(), &["note"]);
    varlen_ds(&obs, "note", &rows);

    let batch = read_dataframe_group(&file, "obs", &mut WarningSink::log()).unwrap();
    let note = batch
        .column_by_name("note")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(
        (0..note.len()).map(|i| note.value(i)).collect::<Vec<_>>(),
        rows
    );
    assert!(note.nulls().is_none());

    // ROUTE PIN. Every value assertion above passes against the four-copy
    // construction too -- byte-identical output is the point of the change, so
    // nothing about the *values* can tell which path ran. The value buffer's
    // allocation can: the builder sizes it once, the `Vec<&str>` construction
    // grows it by doubling. This is the only assertion that fails if the site
    // is routed back.
    let payload: usize = rows.iter().map(|s| s.len()).sum();
    assert_value_buffer_was_pre_sized(note, payload, "obs string column");
}
