//! Tests for the CellBender `remove-background` output reader.
//!
//! The fixture builder deliberately reproduces the *awkward* parts of the real
//! layout, because those are what a naive reader gets wrong:
//!
//! * `/matrix/shape` as an **i32 dataset**, not an attribute.
//! * **Fixed-length ASCII** strings (PyTables `create_carray`), not h5py's
//!   variable-length UTF-8.
//! * **Rank-0** `/global_latents` scalars.
//! * Flattened `/metadata/learning_curve_*` keys and length-1 scalar arrays.
//! * The length asymmetry that makes the full output scatter-aligned and the
//!   filtered output positional — with `barcode_indices_for_latents` stale in
//!   the latter.

use std::path::Path;

use hdf5::types::{FixedAscii, VarLenUnicode};

use arrow::array::{Array, Float32Array, RecordBatch};

use super::*;
use crate::warnings::WarningSink;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// `<name>.h5`: every barcode; latents cover a subset, scattered by index.
    Full,
    /// `<name>_filtered.h5`: cells only; latents are one-per-row.
    Filtered,
}

#[derive(Clone, Copy, PartialEq)]
enum Strings {
    Fixed,
    VarLen,
}

const FIXED: usize = 24;

fn write_strings(group: &hdf5::Group, name: &str, values: &[String], flavor: Strings) {
    match flavor {
        Strings::VarLen => {
            let v: Vec<VarLenUnicode> = values.iter().map(|s| s.parse().unwrap()).collect();
            group
                .new_dataset::<VarLenUnicode>()
                .shape([v.len()])
                .create(name)
                .unwrap()
                .write(&v)
                .unwrap();
        }
        Strings::Fixed => {
            let v: Vec<FixedAscii<FIXED>> = values
                .iter()
                .map(|s| FixedAscii::<FIXED>::from_ascii(s.as_bytes()).unwrap())
                .collect();
            group
                .new_dataset::<FixedAscii<FIXED>>()
                .shape([v.len()])
                .create(name)
                .unwrap()
                .write(&v)
                .unwrap();
        }
    }
}

fn write_f32(group: &hdf5::Group, name: &str, values: &[f32]) {
    group
        .new_dataset::<f32>()
        .shape([values.len()])
        .create(name)
        .unwrap()
        .write(values)
        .unwrap();
}

fn write_i64(group: &hdf5::Group, name: &str, values: &[i64]) {
    group
        .new_dataset::<i64>()
        .shape([values.len()])
        .create(name)
        .unwrap()
        .write(values)
        .unwrap();
}

/// Rank-0 scalar, the shape `run.py` produces via `np.array(x.item())`.
fn write_scalar_f64(group: &hdf5::Group, name: &str, value: f64) {
    group
        .new_dataset::<f64>()
        .shape(())
        .create(name)
        .unwrap()
        .write_scalar(&value)
        .unwrap();
}

struct Fixture {
    n_cells: usize,
    n_genes: usize,
    /// Matrix row order, in the order the file stores them.
    barcodes: Vec<String>,
    /// Matrix rows that carry latents.
    analyzed_rows: Vec<usize>,
}

/// Build a CellBender-shaped `.h5`.
///
/// The stored matrix is CSC of `[n_genes x n_cells]`, i.e. CSR of
/// `[n_cells x n_genes]`; row `r` holds value `r + 1` at gene `r % n_genes`.
fn create_cellbender_h5(
    path: &Path,
    n_cells: usize,
    n_genes: usize,
    kind: Kind,
    strings: Strings,
    placeholder_ids: bool,
) -> Fixture {
    let file = hdf5::File::create(path).unwrap();
    let matrix = file.create_group("matrix").unwrap();

    // Filtered output is in descending-UMI order, so its stored row order is
    // NOT the natural barcode order. That is the whole reason the join must be
    // by key.
    let barcodes: Vec<String> = match kind {
        Kind::Full => (0..n_cells).map(|i| format!("cell_{i}")).collect(),
        Kind::Filtered => (0..n_cells).rev().map(|i| format!("cell_{i}")).collect(),
    };

    let mut indptr = vec![0i64];
    let (mut indices, mut data) = (Vec::<i32>::new(), Vec::<f32>::new());
    for (row, bc) in barcodes.iter().enumerate() {
        let ordinal: usize = bc.strip_prefix("cell_").unwrap().parse().unwrap();
        indices.push((row % n_genes) as i32);
        data.push((ordinal + 1) as f32);
        indptr.push(indices.len() as i64);
    }

    matrix
        .new_dataset::<i32>()
        .shape([2])
        .create("shape")
        .unwrap()
        .write(&[n_genes as i32, n_cells as i32])
        .unwrap();
    write_i64(&matrix, "indptr", &indptr);
    matrix
        .new_dataset::<i32>()
        .shape([indices.len()])
        .create("indices")
        .unwrap()
        .write(&indices)
        .unwrap();
    write_f32(&matrix, "data", &data);
    write_strings(&matrix, "barcodes", &barcodes, strings);

    let features = matrix.create_group("features").unwrap();
    let names: Vec<String> = (0..n_genes).map(|i| format!("GENE{i}")).collect();
    let ids: Vec<String> = if placeholder_ids {
        (0..n_genes).map(|i| format!("NA_{i}")).collect()
    } else {
        (0..n_genes).map(|i| format!("g{i}")).collect()
    };
    write_strings(&features, "name", &names, strings);
    write_strings(&features, "id", &ids, strings);
    write_strings(
        &features,
        "feature_type",
        &vec!["Gene Expression".to_string(); n_genes],
        strings,
    );

    // Latents. Full: a strict subset, scattered by row index. Filtered: one
    // per row, but `barcode_indices_for_latents` stays at the analyzed length.
    let analyzed_rows: Vec<usize> = match kind {
        Kind::Full => (0..n_cells).step_by(2).collect(),
        Kind::Filtered => (0..n_cells).collect(),
    };
    let n_latents = match kind {
        Kind::Full => analyzed_rows.len(),
        Kind::Filtered => n_cells,
    };

    let dl = file.create_group("droplet_latents").unwrap();
    write_f32(
        &dl,
        "cell_probability",
        &(0..n_latents)
            .map(|i| 0.5 + 0.01 * i as f32)
            .collect::<Vec<_>>(),
    );
    write_f32(
        &dl,
        "cell_size",
        &(0..n_latents).map(|i| 100.0 + i as f32).collect::<Vec<_>>(),
    );
    write_f32(&dl, "droplet_efficiency", &vec![0.9f32; n_latents]);
    write_f32(&dl, "background_fraction", &vec![0.1f32; n_latents]);
    // In the filtered file this is the STALE analyzed-length array.
    let idx_array: Vec<i64> = match kind {
        Kind::Full => analyzed_rows.iter().map(|&r| r as i64).collect(),
        Kind::Filtered => (0..n_cells).step_by(2).map(|r| r as i64).collect(),
    };
    write_i64(&dl, "barcode_indices_for_latents", &idx_array);

    let z_dim = 3usize;
    let z: Vec<f32> = (0..n_latents * z_dim).map(|i| i as f32 * 0.1).collect();
    let nd = ndarray::Array2::from_shape_vec((n_latents, z_dim), z).unwrap();
    dl.new_dataset::<f32>()
        .shape([n_latents, z_dim])
        .create("gene_expression_encoding")
        .unwrap()
        .write(&nd)
        .unwrap();

    let gl = file.create_group("global_latents").unwrap();
    write_f32(
        &gl,
        "ambient_expression",
        &(0..n_genes).map(|i| 0.01 * i as f32).collect::<Vec<_>>(),
    );
    write_scalar_f64(&gl, "empty_droplet_size_lognormal_loc", 4.21);
    write_scalar_f64(&gl, "cell_size_lognormal_std", 0.3);

    let md = file.create_group("metadata").unwrap();
    // `unravel_dict` flattens the nested learning curve into keys like these.
    write_f32(&md, "learning_curve_train_epoch", &[0.0, 1.0, 2.0]);
    write_f32(&md, "learning_curve_train_elbo", &[-9.0, -5.0, -3.0]);
    // Scalars are wrapped in length-1 arrays to dodge a scanpy loading bug.
    write_strings(&md, "estimator", &["mckp".to_string()], strings);
    write_f32(&md, "target_false_positive_rate", &[0.01]);
    let analyzed_i64: Vec<i64> = analyzed_rows.iter().map(|&r| r as i64).collect();
    write_i64(&md, "barcodes_analyzed_inds", &analyzed_i64);
    write_i64(
        &md,
        "features_analyzed_inds",
        &(0..n_genes as i64).collect::<Vec<_>>(),
    );
    let analyzed_barcodes: Vec<String> = match kind {
        Kind::Full => analyzed_rows.iter().map(|&r| barcodes[r].clone()).collect(),
        // Stale in the filtered file: the analyzed-length list, not the rows.
        Kind::Filtered => (0..n_cells)
            .step_by(2)
            .map(|r| barcodes[r].clone())
            .collect(),
    };
    write_strings(&md, "barcodes_analyzed", &analyzed_barcodes, strings);

    Fixture {
        n_cells,
        n_genes,
        barcodes,
        analyzed_rows,
    }
}

fn read(path: &Path) -> CellBenderOutput {
    read_cellbender_h5(
        path,
        &CellBenderReadOptions::default(),
        &mut WarningSink::log(),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Matrix
// ---------------------------------------------------------------------------

#[test]
fn reads_shape_from_a_dataset_and_needs_no_transpose() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.h5");
    let fx = create_cellbender_h5(&path, 6, 4, Kind::Full, Strings::VarLen, false);

    let out = read(&path);
    assert_eq!(out.info.n_rows, fx.n_cells);
    assert_eq!(out.info.n_features, fx.n_genes);
    assert_eq!(out.data.row_keys, fx.barcodes);
    assert_eq!(out.data.indptr.len(), fx.n_cells + 1);

    // Row r holds `ordinal + 1` at gene r % n_genes; on the full file the
    // stored order is the natural order.
    for row in 0..fx.n_cells {
        let (s, e) = (
            out.data.indptr[row] as usize,
            out.data.indptr[row + 1] as usize,
        );
        assert_eq!(out.data.indices[s..e], [(row % fx.n_genes) as u32]);
        assert_eq!(out.data.values[s..e], [(row + 1) as f32]);
    }
    assert!(out.info.all_values_integer);
}

#[test]
fn reads_fixed_length_ascii_strings() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixed.h5");
    create_cellbender_h5(&path, 4, 3, Kind::Full, Strings::Fixed, false);

    let out = read(&path);
    assert_eq!(out.data.row_keys[0], "cell_0");
    assert_eq!(out.data.col_keys[0], "g0");
}

#[test]
fn placeholder_ids_fall_back_to_feature_names() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("na.h5");
    create_cellbender_h5(&path, 4, 3, Kind::Full, Strings::VarLen, true);

    let out = read(&path);
    assert_eq!(out.info.feature_key_used, "name");
    assert_eq!(out.data.col_keys[0], "GENE0");
}

// ---------------------------------------------------------------------------
// Latent alignment
// ---------------------------------------------------------------------------

#[test]
fn full_output_scatters_latents_by_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.h5");
    let fx = create_cellbender_h5(&path, 6, 3, Kind::Full, Strings::VarLen, false);

    let out = read(&path);
    assert_eq!(out.info.latent_alignment, LatentAlignment::Scatter);
    assert_eq!(out.info.output_kind, CellBenderOutputKind::Full);

    let ann = out.data.row_annotations.expect("latents imported");
    assert_eq!(ann.num_rows(), fx.n_cells);
    let probs = ann
        .column_by_name("cellbender_cell_probability")
        .unwrap()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();

    // Rows 0, 2, 4 are analyzed (latents 0, 1, 2); the odd rows are null.
    for (latent, &row) in fx.analyzed_rows.iter().enumerate() {
        assert!(!probs.is_null(row));
        assert!((probs.value(row) - (0.5 + 0.01 * latent as f32)).abs() < 1e-6);
    }
    for row in (1..fx.n_cells).step_by(2) {
        assert!(probs.is_null(row), "row {row} was never analysed");
    }
}

/// In the filtered file `barcode_indices_for_latents` is a stale
/// analyzed-length array, so scatter must be rejected and positional chosen.
#[test]
fn filtered_output_uses_positional_alignment_despite_a_stale_index_array() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out_filtered.h5");
    let fx = create_cellbender_h5(&path, 6, 3, Kind::Filtered, Strings::VarLen, false);

    let out = read(&path);
    assert_eq!(out.info.latent_alignment, LatentAlignment::Positional);
    assert_eq!(out.info.output_kind, CellBenderOutputKind::Filtered);

    let ann = out.data.row_annotations.expect("latents imported");
    let probs = ann
        .column_by_name("cellbender_cell_probability")
        .unwrap()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    for row in 0..fx.n_cells {
        assert!(!probs.is_null(row), "every filtered row has a latent");
        assert!((probs.value(row) - (0.5 + 0.01 * row as f32)).abs() < 1e-6);
    }
    // And the row keys really are in reverse order — the case that breaks any
    // positional join between this file and an SCX target.
    assert_eq!(out.data.row_keys[0], format!("cell_{}", fx.n_cells - 1));
}

#[test]
fn unalignable_latents_are_skipped_with_a_warning_rather_than_guessed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.h5");
    create_cellbender_h5(&path, 6, 3, Kind::Full, Strings::VarLen, false);

    // Corrupt the index array so neither scatter nor positional validates.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        let dl = f.group("droplet_latents").unwrap();
        dl.unlink("barcode_indices_for_latents").unwrap();
        write_i64(&dl, "barcode_indices_for_latents", &[0, 1]);
    }

    let mut sink = WarningSink::with_handler(|_| {});
    let out = read_cellbender_h5(&path, &CellBenderReadOptions::default(), &mut sink).unwrap();
    assert_eq!(out.info.latent_alignment, LatentAlignment::None);
    assert!(
        sink.counts().contains_key("skipped_uns_key"),
        "an unalignable latent set must warn"
    );
    // The layer still imports; only the per-cell columns are omitted.
    let has_prob = out
        .data
        .row_annotations
        .as_ref()
        .is_some_and(|a: &RecordBatch| a.column_by_name("cellbender_cell_probability").is_some());
    assert!(!has_prob, "latent columns must be omitted, not guessed");
    assert_eq!(out.data.indptr.len(), 7, "the matrix still imports");
}

// ---------------------------------------------------------------------------
// Annotations and uns
// ---------------------------------------------------------------------------

#[test]
fn ambient_expression_lands_in_var_not_uns() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.h5");
    let fx = create_cellbender_h5(&path, 4, 5, Kind::Full, Strings::VarLen, false);

    let out = read(&path);
    let ann = out.data.col_annotations.expect("var annotations");
    assert_eq!(ann.num_rows(), fx.n_genes);
    let amb = ann
        .column_by_name("cellbender_ambient_expression")
        .unwrap()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert!((amb.value(2) - 0.02).abs() < 1e-6);

    let uns = out.data.uns.unwrap();
    assert!(
        uns["global_latents"].get("ambient_expression").is_none(),
        "a length-G vector must not be duplicated into uns"
    );
}

#[test]
fn uns_enumerates_flattened_metadata_and_rank0_scalars() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.h5");
    create_cellbender_h5(&path, 4, 3, Kind::Full, Strings::VarLen, false);

    let out = read(&path);
    let uns = out.data.uns.unwrap();

    // Flattened learning-curve keys are found by enumeration, not a key list.
    assert!(uns["metadata"]["learning_curve_train_elbo"].is_array());
    // Rank-0 global scalars read as numbers.
    assert!(
        (uns["global_latents"]["empty_droplet_size_lognormal_loc"]
            .as_f64()
            .unwrap()
            - 4.21)
            .abs()
            < 1e-9
    );
    // Length-1 wrapped scalars are unwrapped.
    assert_eq!(uns["estimator"], "mckp");
    assert!((uns["target_false_positive_rate"].as_f64().unwrap() - 0.01).abs() < 1e-9);
    // Per-barcode arrays stay out.
    assert!(uns["metadata"].get("barcodes_analyzed").is_none());
    assert!(uns["metadata"].get("barcodes_analyzed_inds").is_none());
}

#[test]
fn latent_embedding_is_opt_in() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.h5");
    create_cellbender_h5(&path, 6, 3, Kind::Full, Strings::VarLen, false);

    assert!(read(&path).data.row_embeddings.is_empty());

    let out = read_cellbender_h5(
        &path,
        &CellBenderReadOptions {
            latent_embedding: true,
            ..Default::default()
        },
        &mut WarningSink::log(),
    )
    .unwrap();
    let (name, batch) = &out.data.row_embeddings[0];
    assert_eq!(name, "X_cellbender_latent");
    assert_eq!(batch.num_columns(), 3);
    assert_eq!(batch.num_rows(), 6);
}

#[test]
fn source_checksum_is_recorded() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.h5");
    create_cellbender_h5(&path, 3, 2, Kind::Full, Strings::VarLen, false);

    let out = read(&path);
    assert!(out.data.source_checksum.is_some());
    assert_eq!(out.data.source_name.as_deref(), Some("out.h5"));
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

#[test]
fn is_cellbender_h5_accepts_a_real_output_and_rejects_a_plain_10x_file() {
    let dir = tempfile::tempdir().unwrap();
    let cb = dir.path().join("cb.h5");
    create_cellbender_h5(&cb, 3, 2, Kind::Full, Strings::VarLen, false);
    assert!(is_cellbender_h5(&cb));

    let tenx = dir.path().join("tenx.h5");
    crate::convert_tests_common::create_test_tenx_h5(&tenx, 3, 2);
    assert!(!is_cellbender_h5(&tenx));

    let msg = match read_cellbender_h5(
        &tenx,
        &CellBenderReadOptions::default(),
        &mut WarningSink::log(),
    ) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a plain 10x file must be rejected"),
    };
    assert!(msg.contains("plain 10x"), "{msg}");
    assert!(msg.contains("--from 10x"), "must redirect: {msg}");
}

/// The 10x reader now accepts `shape` as a dataset (as real CellRanger writes
/// it) as well as an attribute, and reads fixed-length ASCII strings.
#[test]
fn tenx_reader_still_works_after_the_string_and_shape_refactor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tenx.h5");
    crate::convert_tests_common::create_test_tenx_h5(&path, 4, 3);

    let file = hdf5::File::open(&path).unwrap();
    let data = crate::tenx_read::read_tenx_h5(&file).unwrap();
    assert_eq!(data.n_cells, 4);
    assert_eq!(data.n_genes, 3);
}
