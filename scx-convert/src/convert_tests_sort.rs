//! scx-convert integration tests — sort-on-convert.
//!
//! Exercises `--sort-by` / `ConvertOptions::sort_by`: the obs axis (and X,
//! layers, obsm) is globally reordered by an obs key during conversion.

use super::convert_tests_common::*;
use arrow::array::{Int32Array, StringArray};
use std::path::Path;

/// Build a CSR/dense h5ad fixture with an unsorted `cell_type` column
/// (palette `A,B,A,B,A,…`) plus the standard `n_counts` (= row*10, a unique
/// per-row id). `extras` adds obsm `X_pca` and a CSR `raw` layer.
fn make_h5ad(path: &Path, n_obs: usize, n_vars: usize, fmt: &str, extras: bool) {
    create_test_h5ad(path, n_obs, n_vars, fmt, extras);
    let file = hdf5::File::append(path).unwrap();
    let obs = file.group("obs").unwrap();
    let labels = ["A", "B", "A", "B", "A"];
    let col: Vec<VarLenUnicode> = (0..n_obs).map(|i| vlu(labels[i % labels.len()])).collect();
    obs.new_dataset::<VarLenUnicode>()
        .shape([n_obs])
        .create("cell_type")
        .unwrap()
        .write(&col)
        .unwrap();
}

fn convert_sorted(h5ad: &Path, scx: &Path, by: &[&str], reverse: bool) {
    let opts = ConvertOptions {
        sort_by: by.iter().map(|s| s.to_string()).collect(),
        sort_reverse: reverse,
        ..Default::default()
    };
    h5ad_to_scx_streaming(
        h5ad,
        scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
}

fn convert_plain(h5ad: &Path, scx: &Path) {
    h5ad_to_scx_streaming(
        h5ad,
        scx,
        &ConvertOptions::default(),
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
}

fn dense_rows(csr: &scx_sparse::ScxCsr) -> Vec<Vec<f32>> {
    let (n, c) = csr.shape;
    let flat = csr.to_dense().unwrap();
    (0..n).map(|r| flat[r * c..(r + 1) * c].to_vec()).collect()
}

fn col_i32(batch: &arrow::array::RecordBatch, name: &str) -> Vec<i32> {
    let a = batch
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    (0..a.len()).map(|i| a.value(i)).collect()
}

fn col_str(batch: &arrow::array::RecordBatch, name: &str) -> Vec<String> {
    // Cast handles both plain Utf8 and dictionary-encoded categoricals.
    let col = batch.column_by_name(name).unwrap();
    let utf8 = arrow::compute::cast(col, &DataType::Utf8).unwrap();
    let a = utf8.as_any().downcast_ref::<StringArray>().unwrap();
    (0..a.len()).map(|i| a.value(i).to_string()).collect()
}

#[test]
fn sort_csr_reorders_and_preserves_rows() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let plain = dir.path().join("plain.scx");
    let sorted = dir.path().join("sorted.scx");
    let (n_obs, n_vars) = (10usize, 8usize);
    make_h5ad(&h5ad, n_obs, n_vars, "csr", false);

    convert_plain(&h5ad, &plain);
    convert_sorted(&h5ad, &sorted, &["cell_type"], false);

    let pr = ScxReader::open(&plain).unwrap();
    let sr = ScxReader::open(&sorted).unwrap();
    assert_eq!(sr.n_obs(), n_obs as u64);

    // n_counts == row*10 is a unique per-cell id; map it to the source row.
    let plain_x = dense_rows(&pr.read_all_csr_shards().unwrap());
    let sorted_x = dense_rows(&sr.read_all_csr_shards().unwrap());
    let plain_id = col_i32(&pr.read_obs().unwrap(), "n_counts");
    let sorted_obs = sr.read_obs().unwrap();
    let sorted_id = col_i32(&sorted_obs, "n_counts");
    let sorted_ct = col_str(&sorted_obs, "cell_type");

    // Output is sorted by cell_type (non-decreasing).
    assert!(sorted_ct.windows(2).all(|w| w[0] <= w[1]), "{sorted_ct:?}");

    // Each sorted row's X equals the source cell's X (aligned on n_counts).
    let src_row = |id: i32| plain_id.iter().position(|&v| v == id).unwrap();
    for k in 0..n_obs {
        assert_eq!(sorted_x[k], plain_x[src_row(sorted_id[k])], "row {k}");
    }
    // Same set of cells, reordered.
    let mut a = plain_id.clone();
    a.sort_unstable();
    let mut b = sorted_id.clone();
    b.sort_unstable();
    assert_eq!(a, b);
}

#[test]
fn sort_reverse_descends() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let sorted = dir.path().join("rev.scx");
    make_h5ad(&h5ad, 10, 6, "csr", false);
    convert_sorted(&h5ad, &sorted, &["cell_type"], true);
    let ct = col_str(
        &ScxReader::open(&sorted).unwrap().read_obs().unwrap(),
        "cell_type",
    );
    assert!(ct.windows(2).all(|w| w[0] >= w[1]), "{ct:?}");
}

#[test]
fn sort_is_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let a = dir.path().join("a.scx");
    let b = dir.path().join("b.scx");
    make_h5ad(&h5ad, 10, 8, "csr", false);
    convert_sorted(&h5ad, &a, &["cell_type"], false);
    convert_sorted(&h5ad, &b, &["cell_type"], false);
    let ra = ScxReader::open(&a).unwrap();
    let rb = ScxReader::open(&b).unwrap();
    // Determinism modulo the provenance timestamp: identical row order + X.
    assert_eq!(
        col_i32(&ra.read_obs().unwrap(), "n_counts"),
        col_i32(&rb.read_obs().unwrap(), "n_counts")
    );
    assert_eq!(
        dense_rows(&ra.read_all_csr_shards().unwrap()),
        dense_rows(&rb.read_all_csr_shards().unwrap())
    );
}

#[test]
fn sort_index_ranges_contiguous() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let sorted = dir.path().join("sorted.scx");
    make_h5ad(&h5ad, 10, 8, "csr", false);
    convert_sorted(&h5ad, &sorted, &["cell_type"], false);

    let reader = ScxReader::open(&sorted).unwrap();
    let bytes = reader.read_obs_predicate_index_bytes().unwrap().unwrap();
    let index = scx_engine::PredicateIndex::read_from(&mut std::io::Cursor::new(bytes)).unwrap();
    // Palette A,B,A,B,A over 10 rows → A=6, B=4; after sort A=[0,6), B=[6,10).
    for (val, lo, hi) in [("A", 0u32, 6u32), ("B", 6u32, 10u32)] {
        let ranges = index.categorical_eq("cell_type", val).expect("indexed");
        assert_eq!(ranges.len(), 1, "{val} ranges {ranges:?}");
        assert_eq!((ranges[0].row_start, ranges[0].row_end), (lo, hi), "{val}");
    }
}

#[test]
fn sort_dense_input() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let plain = dir.path().join("plain.scx");
    let sorted = dir.path().join("sorted.scx");
    let (n_obs, n_vars) = (10usize, 8usize);
    make_h5ad(&h5ad, n_obs, n_vars, "dense", false);
    convert_plain(&h5ad, &plain);
    convert_sorted(&h5ad, &sorted, &["cell_type"], false);

    let pr = ScxReader::open(&plain).unwrap();
    let sr = ScxReader::open(&sorted).unwrap();
    let plain_x = dense_rows(&pr.read_all_csr_shards().unwrap());
    let sorted_x = dense_rows(&sr.read_all_csr_shards().unwrap());
    let plain_id = col_i32(&pr.read_obs().unwrap(), "n_counts");
    let sorted_obs = sr.read_obs().unwrap();
    let sorted_id = col_i32(&sorted_obs, "n_counts");
    assert!(col_str(&sorted_obs, "cell_type")
        .windows(2)
        .all(|w| w[0] <= w[1]));
    let src_row = |id: i32| plain_id.iter().position(|&v| v == id).unwrap();
    for k in 0..n_obs {
        assert_eq!(sorted_x[k], plain_x[src_row(sorted_id[k])], "row {k}");
    }
}

#[test]
fn sort_tracks_layers_and_obsm() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let plain = dir.path().join("plain.scx");
    let sorted = dir.path().join("sorted.scx");
    let (n_obs, n_vars) = (10usize, 8usize);
    make_h5ad(&h5ad, n_obs, n_vars, "csr", true);
    convert_plain(&h5ad, &plain);
    convert_sorted(&h5ad, &sorted, &["cell_type"], false);

    let pr = ScxReader::open(&plain).unwrap();
    let sr = ScxReader::open(&sorted).unwrap();
    let plain_id = col_i32(&pr.read_obs().unwrap(), "n_counts");
    let sorted_id = col_i32(&sr.read_obs().unwrap(), "n_counts");
    let src_row = |id: i32| plain_id.iter().position(|&v| v == id).unwrap();

    // The "raw" layer (CSR) must track the same permutation as X.
    let plain_layer = dense_rows(&pr.read_layer("raw").unwrap());
    let sorted_layer = dense_rows(&sr.read_layer("raw").unwrap());
    for k in 0..n_obs {
        assert_eq!(
            sorted_layer[k],
            plain_layer[src_row(sorted_id[k])],
            "layer row {k}"
        );
    }
    // obsm survives the reorder with the full row count.
    assert_eq!(sr.read_obsm("X_pca").unwrap().num_rows(), n_obs);
}

#[test]
fn sort_csc_input_errors() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx = dir.path().join("out.scx");
    make_h5ad(&h5ad, 10, 8, "csc", false);
    let opts = ConvertOptions {
        sort_by: vec!["n_counts".to_string()],
        ..Default::default()
    };
    let err = h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    );
    assert!(err.is_err(), "CSC + sort must hard-error");
}

#[test]
fn sort_composite_key() {
    // Composite [cell_type, n_counts]: leading key dominates, n_counts breaks
    // ties within each category (here every n_counts is unique).
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let sorted = dir.path().join("sorted.scx");
    make_h5ad(&h5ad, 10, 8, "csr", false);
    convert_sorted(&h5ad, &sorted, &["cell_type", "n_counts"], false);
    let obs = ScxReader::open(&sorted).unwrap().read_obs().unwrap();
    let ct = col_str(&obs, "cell_type");
    let id = col_i32(&obs, "n_counts");
    assert!(ct.windows(2).all(|w| w[0] <= w[1]));
    // Within each cell_type block, n_counts ascending.
    for k in 1..ct.len() {
        if ct[k] == ct[k - 1] {
            assert!(id[k] > id[k - 1], "tie order at {k}");
        }
    }
}
