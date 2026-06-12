//! scx-convert integration tests — h5ad (T5.6 split).

use super::convert_tests_common::*;

#[test]
fn test_h5ad_csr_to_scx_to_h5ad_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("test.h5ad");
    let scx_path = dir.path().join("test.scx");
    let h5ad_out = dir.path().join("out.h5ad");

    let n_obs = 20;
    let n_vars = 15;
    create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", true);

    // h5ad → scx
    let opts = ConvertOptions::default();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    // Verify scx
    let reader = ScxReader::open(&scx_path).unwrap();
    assert_eq!(reader.n_obs(), n_obs as u64);
    assert_eq!(reader.n_vars(), n_vars as u64);

    // scx → h5ad
    scx_to_h5ad(&scx_path, &h5ad_out, &mut WarningSink::log()).unwrap();

    // Verify round-trip: read back and compare X data
    let orig_file = hdf5::File::open(&h5ad_path).unwrap();
    let out_file = hdf5::File::open(&h5ad_out).unwrap();

    let orig_x = orig_file.group("X").unwrap();
    let out_x = out_file.group("X").unwrap();

    let orig_data: Vec<f32> = orig_x.dataset("data").unwrap().read_1d().unwrap().to_vec();
    let out_data: Vec<f32> = out_x.dataset("data").unwrap().read_1d().unwrap().to_vec();

    // Integer data should be bit-exact after round-trip
    assert_eq!(orig_data.len(), out_data.len(), "data length mismatch");
    for (i, (a, b)) in orig_data.iter().zip(out_data.iter()).enumerate() {
        assert_eq!(*a, *b, "data mismatch at index {i}: {a} != {b}");
    }
}

/// `adata.raw` (its own, wider var axis) round-trips h5ad → scx → h5ad
/// through both the eager and streaming convert paths.
#[test]
fn test_h5ad_raw_round_trip() {
    use super::pipeline::{h5ad_to_scx_streaming, StreamingOverrides};

    let dir = tempfile::tempdir().unwrap();
    let n_obs = 12;
    let n_vars = 15;
    let raw_n_vars = 23; // raw is WIDER than X

    // Small shard target so raw spans multiple shards on both paths,
    // exercising the streaming coordinator + multi-shard raw assembly.
    let opts = ConvertOptions {
        shard_target_rows: 4,
        ..ConvertOptions::default()
    };

    for streaming in [false, true] {
        let tag = if streaming { "stream" } else { "eager" };
        let h5ad_path = dir.path().join(format!("raw_{tag}.h5ad"));
        create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", false);
        let (raw_ip, raw_ix, raw_dt) = add_raw_group(&h5ad_path, n_obs, raw_n_vars);

        // h5ad → scx
        let scx_path = dir.path().join(format!("raw_{tag}.scx"));
        if streaming {
            h5ad_to_scx_streaming(
                &h5ad_path,
                &scx_path,
                &opts,
                &StreamingOverrides::default(),
                &mut WarningSink::log(),
            )
            .unwrap();
        } else {
            h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();
        }

        // Verify the raw section family in SCX.
        let reader = ScxReader::open(&scx_path).unwrap();
        assert!(reader.has_raw(), "{tag}: has_raw flag must be set");
        assert_eq!(reader.n_vars(), n_vars as u64, "{tag}: X n_vars unchanged");
        let raw = reader.read_all_raw_csr_shards().unwrap();
        assert_eq!(raw.shape, (n_obs, raw_n_vars), "{tag}: raw shape");
        assert_eq!(raw.indptr, raw_ip, "{tag}: raw indptr");
        assert_eq!(raw.indices, raw_ix, "{tag}: raw indices");
        assert_eq!(raw.data, raw_dt, "{tag}: raw data");
        let rv = reader.read_raw_var().unwrap();
        assert_eq!(rv.num_rows(), raw_n_vars, "{tag}: raw var rows");
        drop(reader);

        // scx → h5ad and verify /raw/X + /raw/var present & bit-exact.
        let out = dir.path().join(format!("raw_out_{tag}.h5ad"));
        scx_to_h5ad(&scx_path, &out, &mut WarningSink::log()).unwrap();
        let of = hdf5::File::open(&out).unwrap();
        let orx = of.group("raw/X").unwrap();
        let out_data: Vec<f32> = orx.dataset("data").unwrap().read_1d().unwrap().to_vec();
        assert_eq!(out_data, raw_dt, "{tag}: exported raw data");
        let shape: Vec<i64> = orx.attr("shape").unwrap().read_1d().unwrap().to_vec();
        assert_eq!(
            shape,
            vec![n_obs as i64, raw_n_vars as i64],
            "{tag}: exported raw shape"
        );
        assert!(of.group("raw/var").is_ok(), "{tag}: raw/var present");
    }
}

#[test]
fn test_tenx_to_scx() {
    let dir = tempfile::tempdir().unwrap();
    let tenx_path = dir.path().join("test_10x.h5");
    let scx_path = dir.path().join("test_10x.scx");

    let n_cells = 20;
    let n_genes = 10;
    create_test_tenx_h5(&tenx_path, n_cells, n_genes);

    let opts = ConvertOptions::default();
    tenx_to_scx(&tenx_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    assert_eq!(reader.n_obs(), n_cells as u64);
    assert_eq!(reader.n_vars(), n_genes as u64);

    // Read back and verify data is present
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, n_cells);
    assert_eq!(csr.shape.1, n_genes);
    assert!(csr.nnz() > 0);
}

#[test]
fn test_dense_x() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("dense.h5ad");
    let scx_path = dir.path().join("dense.scx");

    let n_obs = 10;
    let n_vars = 8;
    create_test_h5ad(&h5ad_path, n_obs, n_vars, "dense", false);

    let opts = ConvertOptions::default();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    assert_eq!(reader.n_obs(), n_obs as u64);
    assert_eq!(reader.n_vars(), n_vars as u64);

    let csr = reader.read_all_csr_shards().unwrap();
    assert!(csr.nnz() > 0);
}

#[test]
fn test_csc_x() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("csc.h5ad");
    let scx_path = dir.path().join("csc.scx");

    let n_obs = 10;
    let n_vars = 8;
    create_test_h5ad(&h5ad_path, n_obs, n_vars, "csc", false);

    let opts = ConvertOptions::default();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    // Compare with CSR version
    let csr_h5ad = dir.path().join("csr.h5ad");
    let csr_scx = dir.path().join("csr.scx");
    create_test_h5ad(&csr_h5ad, n_obs, n_vars, "csr", false);
    h5ad_to_scx(&csr_h5ad, &csr_scx, &opts, &mut WarningSink::log()).unwrap();

    let csc_reader = ScxReader::open(&scx_path).unwrap();
    let csr_reader = ScxReader::open(&csr_scx).unwrap();

    let csc_csr = csc_reader.read_all_csr_shards().unwrap();
    let csr_csr = csr_reader.read_all_csr_shards().unwrap();

    assert_eq!(csc_csr.shape, csr_csr.shape);
    assert_eq!(csc_csr.nnz(), csr_csr.nnz());
    // Data should match (same source matrix, just stored differently)
    assert_eq!(csc_csr.data, csr_csr.data);
}

// C1: a CSC group lacking the encoding-type attribute, with n_vars < n_obs,
// must be classified CSC (indptr.len() == n_vars+1) rather than misdetected as
// CSR (which would panic/corrupt downstream). Detection-only and full-convert.
#[test]
fn test_csc_without_encoding_type_detected_by_indptr_len() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("csc_no_enc.h5ad");

    let n_obs = 8usize;
    let n_vars = 3usize; // n_vars < n_obs is the panic-prone case

    // Build CSR arrays, then transpose to CSC for on-disk storage.
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for row in 0..n_obs {
        let col = row % n_vars;
        indices.push(col as i32);
        data.push((row + 1) as f32);
        indptr.push(data.len() as i64);
    }
    let (csc_indptr, csc_indices, csc_data) = csr_to_csc(&indptr, &indices, &data, n_obs, n_vars);
    assert_eq!(csc_indptr.len(), n_vars + 1);

    {
        let file = hdf5::File::create(&path).unwrap();
        // obs group so the file reads as h5ad; no X encoding-type attr.
        file.create_group("obs").unwrap();
        let x = file.create_group("X").unwrap();
        x.new_dataset::<i64>()
            .shape([csc_indptr.len()])
            .create("indptr")
            .unwrap()
            .write(&csc_indptr)
            .unwrap();
        x.new_dataset::<i32>()
            .shape([csc_indices.len()])
            .create("indices")
            .unwrap()
            .write(&csc_indices)
            .unwrap();
        x.new_dataset::<f32>()
            .shape([csc_data.len()])
            .create("data")
            .unwrap()
            .write(&csc_data)
            .unwrap();
        let shape = [n_obs as i64, n_vars as i64];
        x.new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&shape)
            .unwrap();
    }

    let file = hdf5::File::open(&path).unwrap();
    let fmt = detect_matrix_format(&file, &mut WarningSink::log()).unwrap();
    assert_eq!(
        fmt,
        MatrixFormat::Csc,
        "CSC group without encoding-type must be detected via indptr length"
    );
}

// C2: the eager (stream=false) ingest path must validate each layer's shape
// against X, matching the streaming path. A layer whose n_obs disagrees with
// /X is skipped with a LayerSkipped warning rather than written with diverging
// dimensions.
#[test]
fn test_eager_skips_layer_with_mismatched_shape() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("layer_mismatch.h5ad");
    let scx = dir.path().join("layer_mismatch.scx");

    let n_obs = 10usize;
    let n_vars = 8usize;
    create_test_h5ad(&path, n_obs, n_vars, "csr", false);

    // Add a layer whose row count (7) disagrees with X (10).
    {
        let file = hdf5::File::open_rw(&path).unwrap();
        let layers = file.create_group("layers").unwrap();
        let bad = layers.create_group("mismatch").unwrap();
        let l_obs = 7usize;
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for r in 0..l_obs {
            indices.push((r % n_vars) as i32);
            data.push((r + 1) as f32);
            indptr.push(data.len() as i64);
        }
        bad.new_dataset::<i64>()
            .shape([indptr.len()])
            .create("indptr")
            .unwrap()
            .write(&indptr)
            .unwrap();
        bad.new_dataset::<i32>()
            .shape([indices.len()])
            .create("indices")
            .unwrap()
            .write(&indices)
            .unwrap();
        bad.new_dataset::<f32>()
            .shape([data.len()])
            .create("data")
            .unwrap()
            .write(&data)
            .unwrap();
        bad.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("csr_matrix"))
            .unwrap();
        bad.new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[l_obs as i64, n_vars as i64])
            .unwrap();
    }

    let opts = ConvertOptions::default();
    let mut sink = WarningSink::log();
    h5ad_to_scx(&path, &scx, &opts, &mut sink).unwrap();

    assert!(
        sink.counts().get("layer_skipped").copied().unwrap_or(0) >= 1,
        "eager convert should skip the shape-mismatched layer with a warning"
    );

    let reader = ScxReader::open(&scx).unwrap();
    let layer_shards = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == FmtSectionType::LayerCsrShard)
        .count();
    assert_eq!(layer_shards, 0, "mismatched layer must not be written");
}

// A layer stored column-major (CSC) but missing its `encoding-type`
// attribute must be auto-detected from its shape and read as the correct
// cells×genes matrix — the same protection `/X` and `/raw/X` already have.
// The former inline layer reader defaulted to CSR whenever the attr was
// absent and would mis-ingest (or error on) such a layer; routing
// `read_layer_entry` through `detect_matrix_format_at` + `read_x_matrix_at`
// closes that gap.
#[test]
fn test_eager_layer_csc_without_encoding_type_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("layer_csc.h5ad");
    let scx = dir.path().join("layer_csc.scx");

    let n_obs = 6usize;
    let n_vars = 4usize;
    create_test_h5ad(&path, n_obs, n_vars, "csr", false);

    // Known non-square dense matrix (n_obs != n_vars so the CSC-vs-CSR
    // indptr-length discrimination is unambiguous).
    #[rustfmt::skip]
    let dense: Vec<f32> = vec![
        1.0, 0.0, 2.0, 0.0,
        0.0, 3.0, 0.0, 4.0,
        5.0, 0.0, 0.0, 6.0,
        0.0, 0.0, 7.0, 0.0,
        8.0, 9.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 10.0,
    ];

    // Build the CSR of `dense`, then transpose to CSC arrays for
    // column-major on-disk storage (mirrors the working `"csc"` X fixture).
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for row in 0..n_obs {
        for col in 0..n_vars {
            let v = dense[row * n_vars + col];
            if v != 0.0 {
                indices.push(col as i32);
                data.push(v);
            }
        }
        indptr.push(data.len() as i64);
    }
    let (csc_indptr, csc_indices, csc_data) = csr_to_csc(&indptr, &indices, &data, n_obs, n_vars);

    {
        let file = hdf5::File::open_rw(&path).unwrap();
        let layers = file.create_group("layers").unwrap();
        let lyr = layers.create_group("spliced").unwrap();
        lyr.new_dataset::<i64>()
            .shape([csc_indptr.len()])
            .create("indptr")
            .unwrap()
            .write(&csc_indptr)
            .unwrap();
        lyr.new_dataset::<i32>()
            .shape([csc_indices.len()])
            .create("indices")
            .unwrap()
            .write(&csc_indices)
            .unwrap();
        lyr.new_dataset::<f32>()
            .shape([csc_data.len()])
            .create("data")
            .unwrap()
            .write(&csc_data)
            .unwrap();
        // Deliberately NO encoding-type attr — force shape inference.
        lyr.new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[n_obs as i64, n_vars as i64])
            .unwrap();
    }

    let opts = ConvertOptions::default();
    let mut sink = WarningSink::log();
    h5ad_to_scx(&path, &scx, &opts, &mut sink).unwrap();

    assert_eq!(
        sink.counts().get("layer_skipped").copied().unwrap_or(0),
        0,
        "CSC layer should be ingested, not skipped"
    );

    let reader = ScxReader::open(&scx).unwrap();
    let csr = reader.read_layer("spliced").unwrap();
    assert_eq!(csr.shape, (n_obs, n_vars));
    assert_eq!(
        csr.to_dense().unwrap(),
        dense,
        "CSC layer must decode to the original cells×genes matrix"
    );
}

#[test]
fn test_uns_skip_non_serializable() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("uns_test.h5ad");
    let scx_path = dir.path().join("uns_test.scx");

    // Create h5ad with uns containing good data + opaque data
    let n_obs = 5;
    let n_vars = 4;
    create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", false);

    // Add uns with mixed content
    {
        let file = hdf5::File::open_rw(&h5ad_path).unwrap();
        let uns = file.create_group("uns").unwrap();
        let species = vlu("human");
        uns.new_dataset::<VarLenUnicode>()
            .shape(())
            .create("species")
            .unwrap()
            .write_scalar(&species)
            .unwrap();
        uns.new_dataset::<i64>()
            .shape(())
            .create("count")
            .unwrap()
            .write_scalar(&42i64)
            .unwrap();
    }

    let opts = ConvertOptions::default();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    let uns = reader.read_uns().unwrap();
    assert_eq!(uns["species"], "human");
    assert_eq!(uns["count"], 42);
}

// C3/C4 export: 2-D numeric `uns` arrays and 1-D boolean arrays must
// round-trip through h5ad → scx → h5ad. Before the writer fix they were
// silently dropped on export (the `write_uns_value` Array arm only handled
// 1-D i64/f64/string). Note: f32 input widens to f64 on round-trip (the read
// path coerces Float → f64), so the matrix here is authored as f64.
#[test]
fn test_uns_2d_and_bool_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("uns_nd.h5ad");
    let scx_path = dir.path().join("uns_nd.scx");
    let h5ad_out = dir.path().join("uns_nd_out.h5ad");

    create_test_h5ad(&h5ad_path, 5, 4, "csr", false);

    let colors =
        ndarray::Array2::<f64>::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
    let contrasts = ndarray::Array2::<i64>::from_shape_vec((2, 2), vec![1, -2, 3, -4]).unwrap();
    let mask2d =
        ndarray::Array2::<bool>::from_shape_vec((2, 2), vec![true, false, false, true]).unwrap();
    let flags: Vec<bool> = vec![true, false, true, true];

    {
        let file = hdf5::File::open_rw(&h5ad_path).unwrap();
        let uns = file.create_group("uns").unwrap();
        uns.new_dataset::<f64>()
            .shape([2, 3])
            .create("colors")
            .unwrap()
            .write(&colors)
            .unwrap();
        uns.new_dataset::<i64>()
            .shape([2, 2])
            .create("contrasts")
            .unwrap()
            .write(&contrasts)
            .unwrap();
        uns.new_dataset::<bool>()
            .shape([2, 2])
            .create("mask2d")
            .unwrap()
            .write(&mask2d)
            .unwrap();
        uns.new_dataset::<bool>()
            .shape([4])
            .create("flags")
            .unwrap()
            .write(&flags)
            .unwrap();
    }

    let opts = ConvertOptions::default();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();
    scx_to_h5ad(&scx_path, &h5ad_out, &mut WarningSink::log()).unwrap();

    let out = hdf5::File::open(&h5ad_out).unwrap();
    let uns = out.group("uns").unwrap();

    let out_colors = uns.dataset("colors").unwrap().read_2d::<f64>().unwrap();
    assert_eq!(
        out_colors, colors,
        "2-D float uns dropped/garbled on export"
    );

    let out_contrasts = uns.dataset("contrasts").unwrap().read_2d::<i64>().unwrap();
    assert_eq!(out_contrasts, contrasts, "2-D int uns dropped/garbled");

    let out_mask = uns.dataset("mask2d").unwrap().read_2d::<bool>().unwrap();
    assert_eq!(out_mask, mask2d, "2-D bool uns dropped/garbled");

    let out_flags: Vec<bool> = uns
        .dataset("flags")
        .unwrap()
        .read_1d::<bool>()
        .unwrap()
        .to_vec();
    assert_eq!(out_flags, flags, "1-D bool uns dropped on export");
}

#[test]
fn test_categorical_columns() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("cat.h5ad");
    let scx_path = dir.path().join("cat.scx");
    let h5ad_out = dir.path().join("cat_out.h5ad");

    let n_obs = 10;
    let n_vars = 5;

    // Create h5ad with categorical obs column
    create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", false);
    {
        let file = hdf5::File::open_rw(&h5ad_path).unwrap();
        let obs = file.group("obs").unwrap();

        // Add categorical column
        let codes: Vec<i32> = (0..n_obs).map(|i| (i % 3) as i32).collect();
        let ds = obs
            .new_dataset::<i32>()
            .shape([n_obs])
            .create("cell_type")
            .unwrap();
        ds.write(&codes).unwrap();

        let enc = vlu("categorical");
        ds.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&enc)
            .unwrap();

        let cats = vec![vlu("T-cell"), vlu("B-cell"), vlu("Monocyte")];
        ds.new_attr::<VarLenUnicode>()
            .shape([3])
            .create("categories")
            .unwrap()
            .write(&cats)
            .unwrap();

        // Add boolean column
        let bool_data: Vec<u8> = (0..n_obs).map(|i| (i % 2) as u8).collect();
        let bool_ds = obs
            .new_dataset::<u8>()
            .shape([n_obs])
            .create("is_doublet")
            .unwrap();
        bool_ds.write(&bool_data).unwrap();
        let bool_enc = vlu("boolean");
        bool_ds
            .new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&bool_enc)
            .unwrap();
    }

    let opts = ConvertOptions::default();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();
    scx_to_h5ad(&scx_path, &h5ad_out, &mut WarningSink::log()).unwrap();

    // Verify categorical survived
    let reader = ScxReader::open(&scx_path).unwrap();
    let obs = reader.read_obs().unwrap();

    // Find the cell_type column
    let schema = obs.schema();
    let ct_idx = schema.index_of("cell_type").unwrap();
    let ct_col = obs.column(ct_idx);
    assert!(matches!(ct_col.data_type(), DataType::Dictionary(_, _)));
}

/// Regression test for the `scx convert` blocker: real-world h5ad
/// files (e.g. `sc.read_10x_h5(...).write_h5ad(...)`,
/// `sc.datasets.pbmc3k().write_h5ad(...)`) store categorical codes as
/// **int8** whenever `len(categories) < 128`. The previous
/// `read_i32_dataset` had no `IntSize::U1` branch and fell through to
/// `ds.read_1d::<i32>()`, which hdf5-rust rejects with the opaque
/// `HDF5 error: no conversion paths found`. Both the non-streaming
/// (`h5ad_to_scx`) and streaming (`h5ad_to_scx_streaming`) convert
/// paths use the same `read_dataframe_group` → `read_categorical_group`
/// → `read_i32_dataset` chain, so both must accept int8 codes.
#[test]
fn h5ad_with_int8_categorical_codes_converts() {
    use super::pipeline::{h5ad_to_scx_streaming, StreamingOverrides};

    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("int8_codes.h5ad");
    let n_obs = 8;
    let n_vars = 4;
    create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", false);
    {
        let file = hdf5::File::open_rw(&h5ad_path).unwrap();
        let var = file.group("var").unwrap();

        // Write `var/feature_types` as a categorical with int8 codes
        // (mirrors the on-disk layout of pbmc10k.h5ad and every other
        // 10x-Genomics-derived h5ad on the planet).
        let codes: Vec<i8> = (0..n_vars).map(|i| (i % 2) as i8).collect();
        let ds = var
            .new_dataset::<i8>()
            .shape([n_vars])
            .create("feature_types")
            .unwrap();
        ds.write(&codes).unwrap();

        ds.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("categorical"))
            .unwrap();
        let cats = vec![vlu("Gene Expression"), vlu("Antibody Capture")];
        ds.new_attr::<VarLenUnicode>()
            .shape([2])
            .create("categories")
            .unwrap()
            .write(&cats)
            .unwrap();
    }

    // Streaming path — the one `scx convert` uses by default.
    let scx_stream = dir.path().join("stream.scx");
    h5ad_to_scx_streaming(
        &h5ad_path,
        &scx_stream,
        &ConvertOptions::default(),
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect("streaming convert must accept int8 categorical codes");

    // Non-streaming path — also needs to work.
    let scx_bulk = dir.path().join("bulk.scx");
    h5ad_to_scx(
        &h5ad_path,
        &scx_bulk,
        &ConvertOptions::default(),
        &mut WarningSink::log(),
    )
    .expect("non-streaming convert must accept int8 categorical codes");

    // Round-trip back and assert the categorical column survived in
    // both: same dictionary dtype, same category labels in the same
    // order.
    for scx in [&scx_stream, &scx_bulk] {
        let reader = ScxReader::open(scx).unwrap();
        let var = reader.read_var().unwrap();
        let ft_idx = var.schema().index_of("feature_types").unwrap();
        let ft_col = var.column(ft_idx);
        assert!(
            matches!(ft_col.data_type(), DataType::Dictionary(_, _)),
            "feature_types should round-trip as a Dictionary, got {:?}",
            ft_col.data_type()
        );
    }
}

/// Parametric coverage for categorical codes dtypes beyond the int8
/// regression case: int16, uint8, uint16. The on-disk anndata
/// categorical group always sets `codes` to a signed integer in
/// practice, but unsigned forms surface from some non-anndata writers
/// and should be accepted. Each width should round-trip the
/// categorical as a Dictionary array, identical to the int8 case.
macro_rules! categorical_codes_dtype_test {
    ($name:ident, $rust_ty:ty) => {
        #[test]
        fn $name() {
            use super::pipeline::{h5ad_to_scx_streaming, StreamingOverrides};

            let dir = tempfile::tempdir().unwrap();
            let h5ad_path = dir.path().join(concat!(stringify!($name), ".h5ad"));
            let n_obs = 8;
            let n_vars = 4;
            create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", false);
            {
                let file = hdf5::File::open_rw(&h5ad_path).unwrap();
                let var = file.group("var").unwrap();
                let codes: Vec<$rust_ty> = (0..n_vars).map(|i| (i % 2) as $rust_ty).collect();
                let ds = var
                    .new_dataset::<$rust_ty>()
                    .shape([n_vars])
                    .create("feature_types")
                    .unwrap();
                ds.write(&codes).unwrap();
                ds.new_attr::<VarLenUnicode>()
                    .create("encoding-type")
                    .unwrap()
                    .write_scalar(&vlu("categorical"))
                    .unwrap();
                let cats = vec![vlu("Gene Expression"), vlu("Antibody Capture")];
                ds.new_attr::<VarLenUnicode>()
                    .shape([2])
                    .create("categories")
                    .unwrap()
                    .write(&cats)
                    .unwrap();
            }
            let scx = dir.path().join("out.scx");
            h5ad_to_scx_streaming(
                &h5ad_path,
                &scx,
                &ConvertOptions::default(),
                &StreamingOverrides::default(),
                &mut WarningSink::log(),
            )
            .expect(concat!(
                "convert must accept ",
                stringify!($rust_ty),
                " categorical codes"
            ));
            let reader = ScxReader::open(&scx).unwrap();
            let var = reader.read_var().unwrap();
            let ft_idx = var.schema().index_of("feature_types").unwrap();
            assert!(matches!(
                var.column(ft_idx).data_type(),
                DataType::Dictionary(_, _)
            ));
        }
    };
}

categorical_codes_dtype_test!(h5ad_with_int16_categorical_codes_converts, i16);

categorical_codes_dtype_test!(h5ad_with_uint8_categorical_codes_converts, u8);

categorical_codes_dtype_test!(h5ad_with_uint16_categorical_codes_converts, u16);

/// Regression test for the second class of bug fixed by the
/// `HdfNumericDtype` migration in `read_column_to_arrow`. The pre-
/// refactor function fell through to `let data: Vec<i32> = ds.read_1d()?`
/// for any `Unsigned` width other than `U1` / `U4`, which hdf5-rust
/// rejects with the opaque `HDF5 error: no conversion paths found`
/// for `uint16` and `uint64` source dtypes. anndata writers do produce
/// such columns (e.g. integer count columns saved as uint16 to halve
/// disk footprint).
macro_rules! unsigned_dataframe_column_test {
    ($name:ident, $rust_ty:ty, $arrow_dtype:expr) => {
        #[test]
        fn $name() {
            use super::pipeline::{h5ad_to_scx_streaming, StreamingOverrides};

            let dir = tempfile::tempdir().unwrap();
            let h5ad_path = dir.path().join(concat!(stringify!($name), ".h5ad"));
            let n_obs = 8;
            let n_vars = 4;
            create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", false);
            {
                let file = hdf5::File::open_rw(&h5ad_path).unwrap();
                let var = file.group("var").unwrap();
                let col: Vec<$rust_ty> = (0..n_vars).map(|i| i as $rust_ty).collect();
                let ds = var
                    .new_dataset::<$rust_ty>()
                    .shape([n_vars])
                    .create("n_counts")
                    .unwrap();
                ds.write(&col).unwrap();
            }
            let scx = dir.path().join("out.scx");
            h5ad_to_scx_streaming(
                &h5ad_path,
                &scx,
                &ConvertOptions::default(),
                &StreamingOverrides::default(),
                &mut WarningSink::log(),
            )
            .expect(concat!(
                "convert must accept ",
                stringify!($rust_ty),
                " dataframe columns"
            ));
            let reader = ScxReader::open(&scx).unwrap();
            let var = reader.read_var().unwrap();
            let idx = var.schema().index_of("n_counts").unwrap();
            assert_eq!(var.column(idx).data_type(), &$arrow_dtype);
        }
    };
}

unsigned_dataframe_column_test!(
    h5ad_with_u16_dataframe_column_converts,
    u16,
    DataType::Int32
);

unsigned_dataframe_column_test!(
    h5ad_with_u64_dataframe_column_converts,
    u64,
    DataType::Int64
);

/// Companion regression test for the user-visible
/// `scx convert pbmc10k.h5ad` crash. pandas / anndata write an
/// *empty* `obs/@column-order` as a length-0 `float64` array (numpy's
/// default empty-array dtype). The old `read_dataframe_group`
/// unconditionally read the attribute as `Vec<VarLenUnicode>`, which
/// hdf5-rust rejects with `HDF5 error: no conversion paths found`.
/// This test installs that exact attribute on an h5ad with otherwise
/// valid obs and asserts the conversion succeeds.
#[test]
fn h5ad_with_empty_float64_column_order_converts() {
    use super::pipeline::{h5ad_to_scx_streaming, StreamingOverrides};

    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("empty_col_order.h5ad");
    create_test_h5ad(&h5ad_path, 8, 4, "csr", false);

    {
        let file = hdf5::File::open_rw(&h5ad_path).unwrap();
        let obs = file.group("obs").unwrap();

        // Install the anndata "empty dataframe" attribute shape: a
        // length-0 f64 array (numpy's default empty-array dtype).
        // Verbatim layout from pbmc10k.h5ad's `/obs/@column-order`.
        // `create_test_h5ad` does not write column-order itself, so
        // there is nothing to remove first.
        assert!(
            obs.attr("column-order").is_err(),
            "fixture helper should not write column-order"
        );
        let empty: [f64; 0] = [];
        obs.new_attr::<f64>()
            .shape([0usize])
            .create("column-order")
            .unwrap()
            .write_raw(&empty)
            .unwrap();
    }

    let scx_path = dir.path().join("out.scx");
    h5ad_to_scx_streaming(
        &h5ad_path,
        &scx_path,
        &ConvertOptions::default(),
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect("empty float64 column-order must not crash the convert");

    // And confirm we can also read the resulting scx — sanity that the
    // empty-obs path is internally consistent.
    let reader = ScxReader::open(&scx_path).unwrap();
    assert_eq!(reader.header().n_obs, 8);
    assert_eq!(reader.header().n_vars, 4);
}

#[test]
fn test_format_detection_mismatch() {
    let dir = tempfile::tempdir().unwrap();

    // Create 10x file
    let tenx_path = dir.path().join("tenx.h5");
    let scx_path = dir.path().join("out.scx");
    create_test_tenx_h5(&tenx_path, 10, 5);

    // Try converting as h5ad → should error with helpful message
    let opts = ConvertOptions::default();
    let result = h5ad_to_scx(&tenx_path, &scx_path, &opts, &mut WarningSink::log());
    assert!(result.is_err());
    match result.unwrap_err() {
        ConvertError::FormatMismatch { expected, got } => {
            assert_eq!(expected, "h5ad");
            assert_eq!(got, "10x");
        }
        e => panic!("expected FormatMismatch, got: {e}"),
    }

    // Create h5ad file
    let h5ad_path = dir.path().join("test.h5ad");
    create_test_h5ad(&h5ad_path, 10, 5, "csr", false);

    // Try converting as 10x → should error
    let result = tenx_to_scx(&h5ad_path, &scx_path, &opts, &mut WarningSink::log());
    assert!(result.is_err());
    match result.unwrap_err() {
        ConvertError::FormatMismatch { expected, got } => {
            assert_eq!(expected, "10x");
            assert_eq!(got, "h5ad");
        }
        e => panic!("expected FormatMismatch, got: {e}"),
    }
}

#[test]
fn test_integer_dtype_detection() {
    // All integer
    assert!(is_integer_data(&[0.0, 1.0, 255.0, 100.0]));

    // Float values
    assert!(!is_integer_data(&[0.5, 1.0, 2.0]));

    // Negative
    assert!(!is_integer_data(&[-1.0, 1.0, 2.0]));

    // NaN
    assert!(!is_integer_data(&[f32::NAN, 1.0]));

    // Infinity
    assert!(!is_integer_data(&[f32::INFINITY, 1.0]));

    // Empty
    assert!(is_integer_data(&[]));

    // detect_value_encoding (auto-codec selection: pass `None` for the
    // explicit-codec override).
    let (enc, codec) = detect_value_encoding(&[1.0, 2.0, 255.0], None).unwrap();
    assert_eq!(enc, ValueEncoding::Uint8);
    assert_eq!(codec, CodecId::Scx1);

    let (enc, codec) = detect_value_encoding(&[1.0, 256.0], None).unwrap();
    assert_eq!(enc, ValueEncoding::Uint16);
    assert_eq!(codec, CodecId::Scx1);

    let (enc, codec) = detect_value_encoding(&[1.0, 70000.0], None).unwrap();
    assert_eq!(enc, ValueEncoding::Uint32);
    assert_eq!(codec, CodecId::Scx1);

    // Float data auto-routes to Pcodec (see select_codec at
    // scx-format/src/codec_select.rs:50). Older versions selected
    // Zstd for floats — Pcodec landed as the float-data default
    // because it compresses log-normalised / PCA-style data ~4-7%
    // better than Zstd.
    let (enc, codec) = detect_value_encoding(&[0.5, 1.5], None).unwrap();
    assert_eq!(enc, ValueEncoding::Float32);
    assert_eq!(codec, CodecId::Pcodec);
}

#[test]
fn test_csc_to_csr_transpose() {
    // 3x4 matrix:
    // [[0, 1, 0, 2],
    //  [3, 0, 0, 0],
    //  [0, 4, 5, 0]]
    //
    // CSC (col-major):
    // col 0: row 1, val 3
    // col 1: rows 0,2, vals 1,4
    // col 2: row 2, val 5
    // col 3: row 0, val 2
    let csc_indptr = vec![0i64, 1, 3, 4, 5];
    let csc_indices = vec![1i32, 0, 2, 2, 0];
    let csc_data = vec![3.0f32, 1.0, 4.0, 5.0, 2.0];

    let (csr_indptr, csr_indices, csr_data) =
        csc_to_csr(&csc_indptr, &csc_indices, &csc_data, 3, 4).unwrap();

    assert_eq!(csr_indptr, vec![0, 2, 3, 5]);
    assert_eq!(csr_indices, vec![1, 3, 0, 1, 2]);
    assert_eq!(csr_data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_multi_shard() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("shard.h5ad");
    let scx_path = dir.path().join("shard.scx");

    let n_obs = 25;
    let n_vars = 10;
    create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", false);

    let opts = ConvertOptions {
        shard_target_rows: 10,
        ..ConvertOptions::default()
    };
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    assert_eq!(reader.header().n_csr_shards, 3); // 10 + 10 + 5

    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 25);
    assert_eq!(csr.shape.1, 10);
}

#[test]
fn test_float_data_uses_zstd() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("float.h5ad");
    let scx_path = dir.path().join("float.scx");

    // Create h5ad with float data
    let n_obs = 10;
    let n_vars = 5;
    {
        let file = hdf5::File::create(&h5ad_path).unwrap();

        let x = file.create_group("X").unwrap();
        let indptr = vec![0i64, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20];
        let indices: Vec<i32> = (0..20).map(|i| (i % n_vars) as i32).collect();
        // Use actual float values (not integers)
        let data: Vec<f32> = (0..20).map(|i| i as f32 * 0.1 + 0.05).collect();

        x.new_dataset::<i64>()
            .shape([indptr.len()])
            .create("indptr")
            .unwrap()
            .write(&indptr)
            .unwrap();
        x.new_dataset::<i32>()
            .shape([indices.len()])
            .create("indices")
            .unwrap()
            .write(&indices)
            .unwrap();
        x.new_dataset::<f32>()
            .shape([data.len()])
            .create("data")
            .unwrap()
            .write(&data)
            .unwrap();

        let enc = vlu("csr_matrix");
        x.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&enc)
            .unwrap();
        let shape = [n_obs as i64, n_vars as i64];
        x.new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&shape)
            .unwrap();

        // Minimal obs/var
        let obs = file.create_group("obs").unwrap();
        let obs_idx: Vec<VarLenUnicode> = (0..n_obs).map(|i| vlu(&format!("c{i}"))).collect();
        obs.new_dataset::<VarLenUnicode>()
            .shape([n_obs])
            .create("_index")
            .unwrap()
            .write(&obs_idx)
            .unwrap();
        let idx = vlu("_index");
        obs.new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&idx)
            .unwrap();

        let var = file.create_group("var").unwrap();
        let var_idx: Vec<VarLenUnicode> = (0..n_vars).map(|i| vlu(&format!("g{i}"))).collect();
        var.new_dataset::<VarLenUnicode>()
            .shape([n_vars])
            .create("_index")
            .unwrap()
            .write(&var_idx)
            .unwrap();
        let vidx = vlu("_index");
        var.new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&vidx)
            .unwrap();
    }

    let opts = ConvertOptions::default();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    // Float data auto-routes to Pcodec (see `select_codec` at
    // scx-format/src/codec_select.rs:50). Pcodec landed as the
    // float-data default after this test was written; the test
    // name is now historical.
    assert_eq!(reader.header().codec_id, CodecId::Pcodec as u8);
}

/// convert h5ad → scx with `csc=always`, verify the output
/// has `has_csc()`, the expected CSC shard count, contiguous column
/// ranges, and densified contents matching the CSR data.
#[test]
fn test_h5ad_to_scx_csc_always() {
    use scx_format_io::section::SectionType;

    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("with_csc.scx");

    let n_obs = 12;
    let n_vars = 10;
    create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", true);

    let opts = ConvertOptions {
        csc: super::pipeline::CscPolicy::Always,
        csc_cols_per_shard: 4, // → ceil(10/4) = 3 CSC shards
        ..ConvertOptions::default()
    };
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    let hdr = reader.header();
    assert!(hdr.has_csc(), "has_csc must be set after csc=always");
    assert_eq!(hdr.n_csc_shards, 3, "expected ceil(10/4) = 3 CSC shards");

    // Catalog: contiguous coverage of [0, n_vars).
    let csc_entries = reader.catalog().csc_shards_sorted();
    let ranges: Vec<std::ops::Range<u64>> = csc_entries
        .iter()
        .map(|e| e.stats.as_ref().unwrap().col_range())
        .collect();
    assert_eq!(ranges, vec![0..4, 4..8, 8..10]);
    for w in ranges.windows(2) {
        assert_eq!(w[0].end, w[1].start);
    }

    // On-disk shard_type byte is 1 for every CSC shard.
    let bytes = std::fs::read(&scx_path).unwrap();
    for entry in &csc_entries {
        assert_eq!(entry.section_type, SectionType::CscShard);
        let section = &bytes[entry.offset as usize..][..entry.length as usize];
        let sh = scx_format_io::shard::ShardHeader::read_from(&mut std::io::Cursor::new(
            &section[..scx_format_io::shard::SHARD_HEADER_SIZE],
        ))
        .unwrap();
        assert_eq!(sh.shard_type, 1);
    }

    // Densified CSC == densified CSR.
    let dense_csr = reader.read_all_csr_shards().unwrap().to_dense().unwrap();
    let dense_csc = reader.read_all_csc_shards().unwrap().to_dense().unwrap();
    assert_eq!(dense_csc, dense_csr);
}

/// same shape of test for the 10x path.
#[test]
fn test_tenx_to_scx_csc_always() {
    let dir = tempfile::tempdir().unwrap();
    let tenx_path = dir.path().join("input.h5");
    let scx_path = dir.path().join("with_csc.scx");

    let n_cells = 8;
    let n_genes = 12;
    create_test_tenx_h5(&tenx_path, n_cells, n_genes);

    let opts = ConvertOptions {
        csc: super::pipeline::CscPolicy::Always,
        csc_cols_per_shard: 5, // → ceil(12/5) = 3 CSC shards
        ..ConvertOptions::default()
    };
    tenx_to_scx(&tenx_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    assert!(reader.header().has_csc());
    assert_eq!(reader.header().n_csc_shards, 3);
    let dense_csr = reader.read_all_csr_shards().unwrap().to_dense().unwrap();
    let dense_csc = reader.read_all_csc_shards().unwrap().to_dense().unwrap();
    assert_eq!(dense_csc, dense_csr);
}

/// default `ConvertOptions` (csc=false) emits no CSC sidecar.
#[test]
fn test_h5ad_default_csc_off() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("csr_only.scx");
    create_test_h5ad(&h5ad_path, 10, 8, "csr", true);

    let opts = ConvertOptions::default();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    assert!(!reader.header().has_csc());
    assert_eq!(reader.header().n_csc_shards, 0);
}

#[test]
fn test_format_detection() {
    let dir = tempfile::tempdir().unwrap();

    // h5ad
    let h5ad_path = dir.path().join("det.h5ad");
    create_test_h5ad(&h5ad_path, 5, 3, "csr", false);
    let file = hdf5::File::open(&h5ad_path).unwrap();
    assert_eq!(detect_input_format(&file).unwrap(), InputFormat::H5ad);
    assert_eq!(
        detect_matrix_format(&file, &mut WarningSink::log()).unwrap(),
        MatrixFormat::Csr
    );

    // 10x
    let tenx_path = dir.path().join("det.h5");
    create_test_tenx_h5(&tenx_path, 5, 3);
    let file = hdf5::File::open(&tenx_path).unwrap();
    assert_eq!(detect_input_format(&file).unwrap(), InputFormat::TenX);
}

/// Round-trip: create an h5mu fixture → h5mu_to_scx → ScxReader
/// reports two modalities with the right names, var counts, and
/// CSR shard counts. Per-modality reads return non-empty data.
#[test]
fn test_h5mu_round_trip() {
    use crate::h5mu::pipeline::h5mu_to_scx;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_path = dir.path().join("cite.h5mu");
    let scx_path = dir.path().join("cite.scx");
    create_test_h5mu(&h5mu_path, 12, 50, 10);

    let opts = ConvertOptions::default();
    h5mu_to_scx(&h5mu_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    assert!(reader.is_multimodal());
    assert_eq!(reader.n_modalities(), 2);
    // HDF5 returns /mod members in alphabetical order, so the
    // registered order is ["adt", "rna"] rather than the
    // insertion order ["rna", "adt"]. Order-insensitive check.
    let mut names: Vec<String> = reader
        .modality_names()
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    names.sort();
    assert_eq!(names, vec!["adt".to_string(), "rna".to_string()]);
    assert!(reader.header().has_modalities());

    let rna_id = reader.modality_id("rna").unwrap();
    let adt_id = reader.modality_id("adt").unwrap();
    assert_eq!(reader.modality_info(rna_id).unwrap().n_vars, 50);
    assert_eq!(reader.modality_info(adt_id).unwrap().n_vars, 10);

    // Per-modality var read.
    let var_rna = reader.read_var_for(rna_id).unwrap();
    assert_eq!(var_rna.num_rows(), 50);
    let var_adt = reader.read_var_for(adt_id).unwrap();
    assert_eq!(var_adt.num_rows(), 10);

    // Per-modality CSR read.
    assert!(reader.csr_shard_count_for(rna_id) >= 1);
    assert!(reader.csr_shard_count_for(adt_id) >= 1);
    let csr_rna = reader.read_all_csr_shards_for(rna_id).unwrap();
    assert_eq!(csr_rna.shape, (12, 50));
    let csr_adt = reader.read_all_csr_shards_for(adt_id).unwrap();
    assert_eq!(csr_adt.shape, (12, 10));
}

/// Phase E: per-modality codec routing fires on the h5mu pipeline.
/// The `rna` modality (small UMI-style integer counts) should use
/// Scx1; the `adt` modality (Protein → Zstd override) should use
/// Zstd, even though the underlying byte distribution is similar.
#[test]
fn test_h5mu_per_modality_codec_routing() {
    use crate::h5mu::pipeline::h5mu_to_scx;
    use scx_format_io::section::SectionType;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_path = dir.path().join("cite.h5mu");
    let scx_path = dir.path().join("cite.scx");
    create_test_h5mu(&h5mu_path, 12, 50, 10);

    let opts = ConvertOptions::default();
    h5mu_to_scx(&h5mu_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    let rna_id = reader.modality_id("rna").unwrap();
    let adt_id = reader.modality_id("adt").unwrap();

    let bytes = std::fs::read(&scx_path).unwrap();
    let mut rna_codecs: Vec<u8> = Vec::new();
    let mut adt_codecs: Vec<u8> = Vec::new();
    for entry in &reader.catalog().entries {
        if entry.section_type != SectionType::CsrShard {
            continue;
        }
        let section = &bytes[entry.offset as usize..][..entry.length as usize];
        let sh = scx_format_io::shard::ShardHeader::read_from(&mut std::io::Cursor::new(
            &section[..scx_format_io::shard::SHARD_HEADER_SIZE],
        ))
        .unwrap();
        if entry.modality_id == rna_id {
            rna_codecs.push(sh.codec_id);
        } else if entry.modality_id == adt_id {
            adt_codecs.push(sh.codec_id);
        }
    }
    assert!(!rna_codecs.is_empty(), "expected at least one RNA shard");
    assert!(!adt_codecs.is_empty(), "expected at least one ADT shard");
    for c in &rna_codecs {
        assert_eq!(
            *c,
            CodecId::Scx1 as u8,
            "RNA shard codec should be Scx1 (small UMI median, RNA modality)"
        );
    }
    for c in &adt_codecs {
        assert_eq!(
            *c,
            CodecId::Zstd as u8,
            "ADT shard codec should be Zstd (Protein modality override)"
        );
    }

    // Modality table should also remember the resolved per-modality
    // default codecs, since the h5mu pipeline registers each modality
    // with the resolved codec.
    let rna_info = reader.modality_info(rna_id).unwrap();
    let adt_info = reader.modality_info(adt_id).unwrap();
    assert_eq!(rna_info.default_codec_id, CodecId::Scx1 as u8);
    assert_eq!(adt_info.default_codec_id, CodecId::Zstd as u8);
}

/// `scx_to_h5mu` round-trip: convert h5mu → SCX → h5mu and verify
/// the resulting h5mu reports two modalities with the right
/// per-modality shapes and that the outer obs is preserved.
#[test]
fn test_scx_to_h5mu_round_trip() {
    use crate::h5mu::pipeline::h5mu_to_scx;
    use crate::h5mu::write::scx_to_h5mu;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_in = dir.path().join("in.h5mu");
    let scx_path = dir.path().join("mid.scx");
    let h5mu_out = dir.path().join("out.h5mu");
    create_test_h5mu(&h5mu_in, 8, 30, 5);

    let opts = ConvertOptions::default();
    h5mu_to_scx(&h5mu_in, &scx_path, &opts, &mut WarningSink::log()).unwrap();
    scx_to_h5mu(&scx_path, &h5mu_out, &mut WarningSink::log()).unwrap();

    let file = hdf5::File::open(&h5mu_out).unwrap();
    // /mod/rna/X and /mod/adt/X exist.
    assert!(file.group("mod").is_ok());
    assert!(file.group("mod/rna").is_ok());
    assert!(file.group("mod/adt").is_ok());
    assert!(file.group("mod/rna/X").is_ok());
    assert!(file.group("mod/adt/X").is_ok());
    // Outer obs preserved.
    assert!(file.group("obs").is_ok());
    // Per-modality var preserved.
    let rna_var = file.group("mod/rna/var").unwrap();
    assert!(rna_var.dataset("_index").is_ok());
}

/// Single-modality extract: --to h5ad with --modality NAME on a
/// multi-modality file produces a valid h5ad of just that
/// modality.
#[test]
fn test_modality_extract_to_h5ad() {
    use crate::h5mu::pipeline::h5mu_to_scx;
    use crate::h5mu::write::scx_modality_to_h5ad;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_in = dir.path().join("in.h5mu");
    let scx_path = dir.path().join("mid.scx");
    let h5ad_out = dir.path().join("rna.h5ad");
    create_test_h5mu(&h5mu_in, 6, 40, 7);

    let opts = ConvertOptions::default();
    h5mu_to_scx(&h5mu_in, &scx_path, &opts, &mut WarningSink::log()).unwrap();
    scx_modality_to_h5ad(&scx_path, &h5ad_out, "rna", &mut WarningSink::log()).unwrap();

    let file = hdf5::File::open(&h5ad_out).unwrap();
    assert!(file.group("X").is_ok());
    assert!(file.group("obs").is_ok());
    assert!(file.group("var").is_ok());
    // var should have 40 entries (rna's count, not adt's 7).
    let var_idx = file.dataset("var/_index").unwrap();
    assert_eq!(var_idx.shape()[0], 40);
}

/// Phase F.1: `scx info` exposes per-modality counts via the
/// modality table accessor. We don't capture stdout here — instead
/// we verify the underlying accessors (which `run_info` formats).
#[test]
fn test_info_modality_table_exposed() {
    use crate::h5mu::pipeline::h5mu_to_scx;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_path = dir.path().join("cite.h5mu");
    let scx_path = dir.path().join("cite.scx");
    create_test_h5mu(&h5mu_path, 8, 30, 5);

    let opts = ConvertOptions::default();
    h5mu_to_scx(&h5mu_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    let table = reader.modality_table().expect("modality table present");
    assert_eq!(table.len(), 2);
    let names: Vec<&str> = table.entries.iter().map(|i| i.name.as_str()).collect();
    assert!(names.contains(&"rna"));
    assert!(names.contains(&"adt"));
    for info in &table.entries {
        assert!(info.n_csr_shards >= 1);
        // Default codec must not be 0/None — auto resolution always
        // picks a concrete codec, even on empty data.
        assert_ne!(info.default_codec_id, 0);
    }
}

/// Phase F.4: `scx merge` rejects two multimodal files with mismatched
/// modality structures. The user-facing error directs to extract-then-
/// merge.
#[test]
fn test_merge_multimodal_mismatch_raises() {
    use crate::h5mu::pipeline::h5mu_to_scx;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_a = dir.path().join("a.h5mu");
    let h5mu_b = dir.path().join("b.h5mu");
    let scx_a = dir.path().join("a.scx");
    let scx_b = dir.path().join("b.scx");
    let merged = dir.path().join("merged.scx");

    // Two multimodal files where modality b's RNA n_vars differs.
    create_test_h5mu(&h5mu_a, 6, 30, 5);
    create_test_h5mu(&h5mu_b, 6, 50, 5); // different rna n_vars
    let opts = ConvertOptions::default();
    h5mu_to_scx(&h5mu_a, &scx_a, &opts, &mut WarningSink::log()).unwrap();
    h5mu_to_scx(&h5mu_b, &scx_b, &opts, &mut WarningSink::log()).unwrap();

    // The pre-existing n_vars mismatch trips first (header.n_vars
    // is the per-file max). Either way the merge must fail with a
    // clear error rather than producing a corrupt single-modality
    // output.
    let err = scx_ops::merge(&[&scx_a, &scx_b], &merged).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("incompatible n_vars") || msg.contains("modality"),
        "merge error should mention n_vars or modality mismatch; got: {msg}"
    );
}

/// Phase 6: `scx merge` of two multimodal files that match in
/// every modality concatenates rows per-modality and preserves the
/// modality table. Previously this case was rejected (the test
/// name retains the historical `still_unsupported` prefix); merge
/// has since been implemented for matching multimodal structures
/// — assert the positive path.
#[test]
fn test_merge_multimodal_match_still_unsupported() {
    use crate::h5mu::pipeline::h5mu_to_scx;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_a = dir.path().join("a.h5mu");
    let h5mu_b = dir.path().join("b.h5mu");
    let scx_a = dir.path().join("a.scx");
    let scx_b = dir.path().join("b.scx");
    let merged = dir.path().join("merged.scx");

    create_test_h5mu(&h5mu_a, 6, 30, 5);
    create_test_h5mu(&h5mu_b, 6, 30, 5); // matching modality structure
    let opts = ConvertOptions::default();
    h5mu_to_scx(&h5mu_a, &scx_a, &opts, &mut WarningSink::log()).unwrap();
    h5mu_to_scx(&h5mu_b, &scx_b, &opts, &mut WarningSink::log()).unwrap();

    scx_ops::merge(&[&scx_a, &scx_b], &merged).unwrap();
    let reader = ScxReader::open(&merged).unwrap();
    assert!(reader.is_multimodal(), "merged file should be multimodal");
    assert_eq!(reader.n_obs(), 12, "merge should concatenate the 6+6 rows");
    let mod_names: Vec<&str> = reader.modality_names();
    let mut sorted = mod_names.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        vec!["adt", "rna"],
        "both modalities should survive merge"
    );
}

/// Phase 6: `scx compact` on a multimodal file now succeeds —
/// `compact_multimodal` applies the global keep-mask across every
/// modality and preserves the modality table. Previously this case
/// was rejected (the test name retains the historical
/// `_unsupported` suffix); assert the positive path.
#[test]
fn test_compact_multimodal_unsupported() {
    use crate::h5mu::pipeline::h5mu_to_scx;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_in = dir.path().join("in.h5mu");
    let scx_in = dir.path().join("multi.scx");
    let scx_out = dir.path().join("compacted.scx");
    create_test_h5mu(&h5mu_in, 6, 20, 5);

    let opts = ConvertOptions::default();
    h5mu_to_scx(&h5mu_in, &scx_in, &opts, &mut WarningSink::log()).unwrap();

    scx_ops::compact(&scx_in, &scx_out).unwrap();
    let reader = ScxReader::open(&scx_out).unwrap();
    assert!(
        reader.is_multimodal(),
        "compacted multimodal file should remain multimodal"
    );
    assert_eq!(
        reader.n_obs(),
        6,
        "no deletion vectors → all rows preserved"
    );
}

/// Phase F.3: `scx_ops::append` (with `modality_id` set) stamps shards with the
/// chosen modality_id and updates the modality table's per-modality
/// counts. We exercise the full path: build a multimodal file from
/// h5mu, append to its rna modality, verify the rna modality's
/// counts went up while the adt modality is untouched.
#[test]
fn test_append_for_modality_updates_table() {
    use crate::h5mu::pipeline::h5mu_to_scx;
    use scx_format_io::section::SectionType;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_path = dir.path().join("cite.h5mu");
    let scx_path = dir.path().join("cite.scx");
    create_test_h5mu(&h5mu_path, 8, 20, 5);

    let opts = ConvertOptions::default();
    h5mu_to_scx(&h5mu_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    // Capture pre-append per-modality state.
    let pre = ScxReader::open(&scx_path).unwrap();
    let rna_id = pre.modality_id("rna").unwrap();
    let adt_id = pre.modality_id("adt").unwrap();
    let pre_rna_csr = pre.modality_info(rna_id).unwrap().n_csr_shards;
    let pre_adt_csr = pre.modality_info(adt_id).unwrap().n_csr_shards;
    let pre_rna_nnz = pre.modality_info(rna_id).unwrap().nnz;
    let pre_adt_nnz = pre.modality_info(adt_id).unwrap().nnz;
    let pre_obs = pre.read_obs().unwrap();
    drop(pre);

    // Build a fresh batch of CSR data appropriate for rna's vars
    // (n_vars = 20). One non-zero per row at column 0.
    let n_new_rows = 4u64;
    let new_indptr: Vec<u64> = (0..=n_new_rows).collect();
    let new_indices: Vec<u32> = vec![0u32; n_new_rows as usize];
    let new_values: Vec<u8> = vec![1u8; n_new_rows as usize]; // uint8

    // Build a new obs batch by truncating the existing obs to 4 rows.
    let new_obs_batch = pre_obs.slice(0, n_new_rows as usize);

    scx_ops::append(
        &scx_path,
        &new_obs_batch,
        &new_indptr,
        &new_indices,
        &new_values,
        scx_codec::ValueEncoding::Uint8,
        &scx_ops::AppendOptions {
            codec: scx_codec::CodecSelection::Explicit(scx_codec::CodecId::Scx1),
            shard_target_rows: NonZeroU32::new(10000).unwrap(),
            modality_id: rna_id,
        },
    )
    .unwrap();

    // Re-open and verify post-append state.
    let post = ScxReader::open(&scx_path).unwrap();
    assert_eq!(post.header().n_obs, 8 + n_new_rows);
    let rna_post = post.modality_info(rna_id).unwrap();
    let adt_post = post.modality_info(adt_id).unwrap();
    assert!(
        rna_post.n_csr_shards > pre_rna_csr,
        "rna n_csr_shards should grow"
    );
    assert!(rna_post.nnz > pre_rna_nnz, "rna nnz should grow");
    assert_eq!(adt_post.n_csr_shards, pre_adt_csr, "adt untouched");
    assert_eq!(adt_post.nnz, pre_adt_nnz, "adt untouched");

    // The new shard's catalog entry must be stamped with rna_id.
    let new_shards: Vec<_> = post
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CsrShard && e.modality_id == rna_id)
        .collect();
    assert!(new_shards.len() as u32 >= rna_post.n_csr_shards);
}

/// Report B1: integer- and float-keyed categorical obs columns (valid
/// pandas/anndata — e.g. integer cluster labels, dose levels) were
/// previously dropped with `categorical group: HDF5 error: no conversion
/// paths found` because the reader assumed string categories. They must
/// now round-trip h5ad → scx → h5ad with the category dtype preserved.
#[test]
fn numeric_categorical_columns_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("numcat.h5ad");
    let scx_path = dir.path().join("numcat.scx");
    let h5ad_out = dir.path().join("numcat_out.h5ad");

    let n_obs = 9;
    create_test_h5ad(&h5ad_path, n_obs, 4, "csr", false);

    // Modern anndata categorical *group* form (encoding-version 0.2.0):
    // a subgroup with `codes` + a numeric `categories` dataset.
    let int_cats: Vec<i64> = vec![10, 20, 30];
    let float_cats: Vec<f64> = vec![0.1, 0.5, 0.9];
    let codes: Vec<i32> = (0..n_obs).map(|i| (i % 3) as i32).collect();
    {
        let file = hdf5::File::open_rw(&h5ad_path).unwrap();
        let obs = file.group("obs").unwrap();
        for (name, write_cats) in [
            ("int_cat", true), // integer categories
            ("lvl", false),    // float categories
        ] {
            let g = obs.create_group(name).unwrap();
            g.new_dataset::<i32>()
                .shape([n_obs])
                .create("codes")
                .unwrap()
                .write(&codes)
                .unwrap();
            if write_cats {
                g.new_dataset::<i64>()
                    .shape([int_cats.len()])
                    .create("categories")
                    .unwrap()
                    .write(&int_cats)
                    .unwrap();
            } else {
                g.new_dataset::<f64>()
                    .shape([float_cats.len()])
                    .create("categories")
                    .unwrap()
                    .write(&float_cats)
                    .unwrap();
            }
            g.new_attr::<VarLenUnicode>()
                .create("encoding-type")
                .unwrap()
                .write_scalar(&vlu("categorical"))
                .unwrap();
            g.new_attr::<VarLenUnicode>()
                .create("encoding-version")
                .unwrap()
                .write_scalar(&vlu("0.2.0"))
                .unwrap();
            g.new_attr::<bool>()
                .create("ordered")
                .unwrap()
                .write_scalar(&false)
                .unwrap();
        }
    }

    // No warnings expected: both columns must convert, not be skipped.
    use std::sync::{Arc, Mutex};
    let warnings: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let w = Arc::clone(&warnings);
    let mut sink = WarningSink::with_handler(move |x| w.lock().unwrap().push(format!("{x:?}")));
    h5ad_to_scx(&h5ad_path, &scx_path, &ConvertOptions::default(), &mut sink).unwrap();
    let warns = warnings.lock().unwrap().join("\n");
    assert!(
        !warns.contains("int_cat") && !warns.contains("\"lvl\""),
        "numeric categoricals were skipped: {warns}"
    );

    // scx stores the dictionaries with their numeric value type.
    let reader = ScxReader::open(&scx_path).unwrap();
    let obs = reader.read_obs().unwrap();
    let schema = obs.schema();
    let int_col = obs.column(schema.index_of("int_cat").unwrap());
    let lvl_col = obs.column(schema.index_of("lvl").unwrap());
    assert!(
        matches!(int_col.data_type(), DataType::Dictionary(k, v)
            if **k == DataType::Int32 && **v == DataType::Int64),
        "int_cat should be Dictionary(Int32, Int64), got {:?}",
        int_col.data_type()
    );
    assert!(
        matches!(lvl_col.data_type(), DataType::Dictionary(k, v)
            if **k == DataType::Int32 && **v == DataType::Float64),
        "lvl should be Dictionary(Int32, Float64), got {:?}",
        lvl_col.data_type()
    );

    // h5ad export re-emits numeric `categories` datasets with matching values.
    scx_to_h5ad(&scx_path, &h5ad_out, &mut WarningSink::log()).unwrap();
    let out = hdf5::File::open(&h5ad_out).unwrap();
    let out_obs = out.group("obs").unwrap();

    let out_int_cats: Vec<i64> = out_obs
        .group("int_cat")
        .unwrap()
        .dataset("categories")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(
        out_int_cats, int_cats,
        "int categories drifted on round-trip"
    );

    let out_float_cats: Vec<f64> = out_obs
        .group("lvl")
        .unwrap()
        .dataset("categories")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(out_float_cats, float_cats, "float categories drifted");

    // Codes preserved too (so the decoded values match the source).
    let out_codes: Vec<i32> = out_obs
        .group("int_cat")
        .unwrap()
        .dataset("codes")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(out_codes, codes, "codes drifted on round-trip");
}

/// Report B2 / E1: scanpy `rank_genes_groups` is stored as an HDF5
/// *compound* (structured) array. It is not preserved, but the skip
/// warning must be a short actionable line — not a ~1 KB
/// `CompoundType { fields: [..] }` Debug dump.
#[test]
fn compound_uns_array_emits_short_actionable_warning() {
    // A 2-field compound row mimics a `rank_genes_groups` recarray with
    // two cluster groups.
    #[repr(C)]
    #[derive(hdf5::H5Type, Clone, Copy)]
    struct RggRow {
        cl0: f32,
        cl1: f32,
    }

    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("rgg.h5ad");
    create_test_h5ad(&h5ad_path, 5, 4, "csr", false);
    {
        let file = hdf5::File::open_rw(&h5ad_path).unwrap();
        let uns = file
            .group("uns")
            .unwrap_or_else(|_| file.create_group("uns").unwrap());
        let rgg = uns.create_group("rank_genes_groups").unwrap();
        let rows = vec![RggRow { cl0: 1.0, cl1: 2.0 }, RggRow { cl0: 3.0, cl1: 4.0 }];
        rgg.new_dataset::<RggRow>()
            .shape([rows.len()])
            .create("scores")
            .unwrap()
            .write(&rows)
            .unwrap();
    }

    use std::sync::{Arc, Mutex};
    let warnings: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let w = Arc::clone(&warnings);
    let mut sink = WarningSink::with_handler(move |x| w.lock().unwrap().push(format!("{x:?}")));
    let file = hdf5::File::open(&h5ad_path).unwrap();
    crate::h5ad::read::read_uns(&file, false, &mut sink).unwrap();
    let joined = warnings.lock().unwrap().join("\n");

    assert!(
        joined.contains("compound/structured array with 2 fields"),
        "expected a short compound message, got: {joined}"
    );
    assert!(
        joined.contains("rank_genes_groups"),
        "message should mention rank_genes_groups: {joined}"
    );
    // The noisy ~1 KB dump must be gone.
    assert!(
        !joined.contains("CompoundField") && !joined.contains("fields: ["),
        "compound type dump leaked into the warning: {joined}"
    );
}

/// Report B3: a Python `None` uns scalar (anndata writes it as an
/// `h5py.Empty` null-dataspace dataset) previously round-tripped as a
/// bogus float `0.0` — silently corrupting e.g. `uns['log1p']['base']`,
/// which scanpy feeds to `log(x)/log(base)` → `-inf`. It must now
/// round-trip as JSON null (→ Python `None`), at the top level and nested.
#[test]
fn none_uns_scalar_round_trips_as_null() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("none.h5ad");
    let scx_path = dir.path().join("none.scx");
    let h5ad_out = dir.path().join("none_out.h5ad");

    create_test_h5ad(&h5ad_path, 5, 4, "csr", false);

    // Write anndata's null encoding: a null-dataspace (h5py.Empty) dataset
    // tagged encoding-type="null", at top level and nested under `log1p`.
    let write_null = |g: &hdf5::Group, name: &str| {
        let ds = g
            .new_dataset::<f32>()
            .shape(hdf5::Extents::null())
            .create(name)
            .unwrap();
        ds.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("null"))
            .unwrap();
    };
    {
        let file = hdf5::File::open_rw(&h5ad_path).unwrap();
        let uns = file
            .group("uns")
            .unwrap_or_else(|_| file.create_group("uns").unwrap());
        write_null(&uns, "none_scalar");
        let log1p = uns.create_group("log1p").unwrap();
        write_null(&log1p, "base");
    }

    h5ad_to_scx(
        &h5ad_path,
        &scx_path,
        &ConvertOptions::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    scx_to_h5ad(&scx_path, &h5ad_out, &mut WarningSink::log()).unwrap();

    // Read the round-tripped uns and confirm the nulls survived (NOT 0.0).
    let out = hdf5::File::open(&h5ad_out).unwrap();
    let uns = crate::h5ad::read::read_uns(&out, false, &mut WarningSink::log()).unwrap();

    assert_eq!(
        uns.get("none_scalar"),
        Some(&serde_json::Value::Null),
        "top-level None coerced/lost: {uns:?}"
    );
    assert_eq!(
        uns.get("log1p").and_then(|l| l.get("base")),
        Some(&serde_json::Value::Null),
        "nested uns['log1p']['base']=None corrupted: {uns:?}"
    );
}

/// B4 + B5: DataFrame-valued and sparse-valued obsm members are warned
/// (`SkippedObsm`) instead of silently dropped, a float16 obsm dataset is
/// upcast to f32 instead of aborting the whole conversion, and plain dense
/// obsm still round-trips. All on the default (disk-streaming) convert path.
#[test]
fn test_obsm_dataframe_sparse_warn_and_f16_upcast() {
    use half::f16;

    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("obsm_edge.h5ad");
    let scx_path = dir.path().join("obsm_edge.scx");
    let h5ad_out = dir.path().join("obsm_edge_out.h5ad");

    let n_obs = 20;
    let n_vars = 15;
    // include_extras=true creates a plain f32 obsm/X_pca we expect to survive.
    create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", true);

    // f16 values chosen to be exactly representable (multiples of 0.5), so the
    // f16 → f32 upcast is bit-exact and we can assert equality.
    let f16_cols = 3usize;
    let f16_vals_f32: Vec<f32> = (0..n_obs * f16_cols).map(|i| i as f32 * 0.5).collect();
    {
        let file = hdf5::File::open_rw(&h5ad_path).unwrap();
        let obsm = file.group("obsm").unwrap();

        // (B5) float16 dense obsm.
        let f16_vals: Vec<f16> = f16_vals_f32.iter().map(|&v| f16::from_f32(v)).collect();
        let nd = ndarray::Array2::from_shape_vec((n_obs, f16_cols), f16_vals).unwrap();
        obsm.new_dataset::<f16>()
            .shape([n_obs, f16_cols])
            .create("X_f16")
            .unwrap()
            .write(&nd)
            .unwrap();

        // (B4) DataFrame-encoded obsm (an HDF5 group, not a 2D dataset).
        let df = obsm.create_group("df_embed").unwrap();
        let enc = vlu("dataframe");
        df.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&enc)
            .unwrap();
        let col: Vec<f32> = (0..n_obs).map(|i| i as f32).collect();
        df.new_dataset::<f32>()
            .shape([n_obs])
            .create("d1")
            .unwrap()
            .write(&col)
            .unwrap();

        // (B4) sparse-matrix-valued obsm (CSR subgroup under /obsm).
        let sp = obsm.create_group("sparse_obsm").unwrap();
        let enc = vlu("csr_matrix");
        sp.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&enc)
            .unwrap();
        let indptr: Vec<i64> = (0..=n_obs as i64).collect(); // one nnz per row
        let indices: Vec<i32> = vec![0; n_obs];
        let data: Vec<f32> = vec![1.0; n_obs];
        sp.new_dataset::<i64>()
            .shape([indptr.len()])
            .create("indptr")
            .unwrap()
            .write(&indptr)
            .unwrap();
        sp.new_dataset::<i32>()
            .shape([indices.len()])
            .create("indices")
            .unwrap()
            .write(&indices)
            .unwrap();
        sp.new_dataset::<f32>()
            .shape([data.len()])
            .create("data")
            .unwrap()
            .write(&data)
            .unwrap();
        let shape = [n_obs as i64, 4i64];
        sp.new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&shape)
            .unwrap();
    }

    // h5ad → scx must SUCCEED (B5: no fatal abort) and warn for the two
    // unconvertible obsm members (B4).
    let opts = ConvertOptions::default();
    let mut sink = WarningSink::log();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap();
    assert_eq!(
        sink.counts().get("skipped_obsm").copied(),
        Some(2),
        "expected SkippedObsm for df_embed + sparse_obsm; counts: {:?}",
        sink.counts()
    );

    // scx → h5ad and inspect the surviving obsm.
    scx_to_h5ad(&scx_path, &h5ad_out, &mut WarningSink::log()).unwrap();
    let out = hdf5::File::open(&h5ad_out).unwrap();
    let out_obsm = out.group("obsm").unwrap();
    let members = out_obsm.member_names().unwrap();
    assert!(
        members.iter().any(|m| m == "X_pca"),
        "plain dense obsm dropped: {members:?}"
    );
    assert!(
        members.iter().any(|m| m == "X_f16"),
        "f16 obsm not preserved: {members:?}"
    );
    assert!(
        !members
            .iter()
            .any(|m| m == "df_embed" || m == "sparse_obsm"),
        "unconvertible obsm unexpectedly present: {members:?}"
    );

    // f16 → f32 values are bit-exact for the chosen inputs.
    let got: ndarray::Array2<f32> = out_obsm.dataset("X_f16").unwrap().read_2d().unwrap();
    let (gv, _) = got.into_raw_vec_and_offset();
    assert_eq!(gv, f16_vals_f32, "f16 obsm values corrupted on upcast");
}

/// B5 (column twin): a float16 obs column is upcast to f32 and preserved
/// rather than warn-skipped.
#[test]
fn test_obs_float16_column_upcasts_to_f32() {
    use half::f16;

    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("obs_f16.h5ad");
    let scx_path = dir.path().join("obs_f16.scx");
    let h5ad_out = dir.path().join("obs_f16_out.h5ad");

    let n_obs = 12;
    let n_vars = 5;
    create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", false);

    let vals_f32: Vec<f32> = (0..n_obs).map(|i| i as f32 * 0.25).collect();
    {
        let file = hdf5::File::open_rw(&h5ad_path).unwrap();
        let obs = file.group("obs").unwrap();
        let vals: Vec<f16> = vals_f32.iter().map(|&v| f16::from_f32(v)).collect();
        obs.new_dataset::<f16>()
            .shape([n_obs])
            .create("score_f16")
            .unwrap()
            .write(&vals)
            .unwrap();
    }

    let opts = ConvertOptions::default();
    let mut sink = WarningSink::log();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap();
    assert_eq!(
        sink.counts().get("skipped_column").copied().unwrap_or(0),
        0,
        "f16 obs column should be upcast, not skipped; counts: {:?}",
        sink.counts()
    );

    scx_to_h5ad(&scx_path, &h5ad_out, &mut WarningSink::log()).unwrap();
    let out = hdf5::File::open(&h5ad_out).unwrap();
    let got: Vec<f32> = out
        .group("obs")
        .unwrap()
        .dataset("score_f16")
        .unwrap()
        .read_1d::<f32>()
        .unwrap()
        .to_vec();
    assert_eq!(got, vals_f32, "f16 obs column corrupted on upcast");
}

/// B6: `uns` float scalars/arrays containing NaN/Inf round-trip exactly
/// (via the tagged base64 envelope) instead of silently degrading to
/// `null`/`0.0`; the source dtype width is preserved too.
#[test]
fn uns_nonfinite_floats_round_trip() {
    use hdf5::types::{FloatSize, TypeDescriptor};

    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("naninf.h5ad");
    let scx_path = dir.path().join("naninf.scx");
    let h5ad_out = dir.path().join("naninf_out.h5ad");

    create_test_h5ad(&h5ad_path, 5, 4, "csr", false);

    let arr_f64 = vec![1.0_f64, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 2.0];
    let arr_f32 = vec![1.5_f32, f32::NAN, -3.25];
    {
        let file = hdf5::File::open_rw(&h5ad_path).unwrap();
        let uns = file.create_group("uns").unwrap();
        uns.new_dataset::<f64>()
            .shape(())
            .create("inf_scalar")
            .unwrap()
            .write_scalar(&f64::INFINITY)
            .unwrap();
        uns.new_dataset::<f64>()
            .shape(())
            .create("nan_scalar")
            .unwrap()
            .write_scalar(&f64::NAN)
            .unwrap();
        uns.new_dataset::<f64>()
            .shape([arr_f64.len()])
            .create("naninf_1d")
            .unwrap()
            .write(&arr_f64)
            .unwrap();
        uns.new_dataset::<f32>()
            .shape([arr_f32.len()])
            .create("naninf_f32")
            .unwrap()
            .write(&arr_f32)
            .unwrap();
    }

    let opts = ConvertOptions::default();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();
    scx_to_h5ad(&scx_path, &h5ad_out, &mut WarningSink::log()).unwrap();

    let out = hdf5::File::open(&h5ad_out).unwrap();
    let uns = out.group("uns").unwrap();

    let inf: f64 = uns.dataset("inf_scalar").unwrap().read_scalar().unwrap();
    assert!(inf.is_infinite() && inf > 0.0, "inf scalar lost: {inf}");
    let nan: f64 = uns.dataset("nan_scalar").unwrap().read_scalar().unwrap();
    assert!(nan.is_nan(), "nan scalar lost: {nan}");

    let got: Vec<f64> = uns
        .dataset("naninf_1d")
        .unwrap()
        .read_1d::<f64>()
        .unwrap()
        .to_vec();
    assert_eq!(got.len(), arr_f64.len());
    assert_eq!(got[0], 1.0);
    assert!(got[1].is_nan(), "elem 1 should be NaN: {got:?}");
    assert!(got[2].is_infinite() && got[2] > 0.0, "elem 2 +Inf: {got:?}");
    assert!(got[3].is_infinite() && got[3] < 0.0, "elem 3 -Inf: {got:?}");
    assert_eq!(got[4], 2.0);

    // f32 source keeps its width (envelope carries the exact dtype).
    let f32_ds = uns.dataset("naninf_f32").unwrap();
    assert_eq!(
        f32_ds.dtype().unwrap().to_descriptor().unwrap(),
        TypeDescriptor::Float(FloatSize::U4),
        "f32 uns array width not preserved"
    );
    let got_f32: Vec<f32> = f32_ds.read_1d::<f32>().unwrap().to_vec();
    assert_eq!(got_f32[0], 1.5);
    assert!(got_f32[1].is_nan());
    assert_eq!(got_f32[2], -3.25);
}

/// B7: a 3-D `uns` array (no 1-D/2-D JSON form) round-trips with exact
/// shape and values via the tagged envelope, instead of being dropped.
#[test]
fn uns_3d_arrays_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("nd.h5ad");
    let scx_path = dir.path().join("nd.scx");
    let h5ad_out = dir.path().join("nd_out.h5ad");

    create_test_h5ad(&h5ad_path, 5, 4, "csr", false);

    let i64_3d = ndarray::Array::from_shape_vec((2, 3, 4), (0..24i64).collect::<Vec<_>>()).unwrap();
    let f64_3d = ndarray::Array::from_shape_vec(
        (2, 2, 2),
        (0..8).map(|i| i as f64 * 0.5).collect::<Vec<_>>(),
    )
    .unwrap();
    {
        let file = hdf5::File::open_rw(&h5ad_path).unwrap();
        let uns = file.create_group("uns").unwrap();
        uns.new_dataset::<i64>()
            .shape([2, 3, 4])
            .create("arr_3d_i64")
            .unwrap()
            .write(&i64_3d)
            .unwrap();
        uns.new_dataset::<f64>()
            .shape([2, 2, 2])
            .create("arr_3d_f64")
            .unwrap()
            .write(&f64_3d)
            .unwrap();
    }

    let opts = ConvertOptions::default();
    let mut sink = WarningSink::log();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap();
    assert_eq!(
        sink.counts().get("skipped_uns_key").copied().unwrap_or(0),
        0,
        "3-D uns arrays should be preserved, not skipped: {:?}",
        sink.counts()
    );
    scx_to_h5ad(&scx_path, &h5ad_out, &mut WarningSink::log()).unwrap();

    let out = hdf5::File::open(&h5ad_out).unwrap();
    let uns = out.group("uns").unwrap();

    let got_i: ndarray::ArrayD<i64> = uns.dataset("arr_3d_i64").unwrap().read_dyn().unwrap();
    assert_eq!(got_i.shape(), &[2, 3, 4], "3-D int shape lost");
    assert_eq!(got_i, i64_3d.into_dyn(), "3-D int values garbled");

    let got_f: ndarray::ArrayD<f64> = uns.dataset("arr_3d_f64").unwrap().read_dyn().unwrap();
    assert_eq!(got_f.shape(), &[2, 2, 2], "3-D float shape lost");
    assert_eq!(got_f, f64_3d.into_dyn(), "3-D float values garbled");
}

/// B8: a scipy-sparse matrix in `uns` (encoding-type="csr_matrix") emits a
/// `FlattenedUnsSparse` warning (no longer silent); its arrays still survive
/// as a dict.
#[test]
fn uns_sparse_matrix_emits_warning() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("sp.h5ad");
    let scx_path = dir.path().join("sp.scx");
    let h5ad_out = dir.path().join("sp_out.h5ad");

    create_test_h5ad(&h5ad_path, 5, 4, "csr", false);
    {
        let file = hdf5::File::open_rw(&h5ad_path).unwrap();
        let uns = file.create_group("uns").unwrap();
        let sp = uns.create_group("sparse_uns").unwrap();
        sp.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("csr_matrix"))
            .unwrap();
        let indptr: Vec<i32> = vec![0, 1, 2, 2, 3];
        let indices: Vec<i32> = vec![0, 2, 1];
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        sp.new_dataset::<i32>()
            .shape([indptr.len()])
            .create("indptr")
            .unwrap()
            .write(&indptr)
            .unwrap();
        sp.new_dataset::<i32>()
            .shape([indices.len()])
            .create("indices")
            .unwrap()
            .write(&indices)
            .unwrap();
        sp.new_dataset::<f64>()
            .shape([data.len()])
            .create("data")
            .unwrap()
            .write(&data)
            .unwrap();
        let shape = [4i64, 3i64];
        sp.new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&shape)
            .unwrap();
    }

    let opts = ConvertOptions::default();
    let mut sink = WarningSink::log();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap();
    assert_eq!(
        sink.counts()
            .get("flattened_uns_sparse")
            .copied()
            .unwrap_or(0),
        1,
        "scipy-sparse uns should emit one FlattenedUnsSparse warning: {:?}",
        sink.counts()
    );
    scx_to_h5ad(&scx_path, &h5ad_out, &mut WarningSink::log()).unwrap();

    // The component arrays still survive (as a plain dict / subgroup).
    let out = hdf5::File::open(&h5ad_out).unwrap();
    let recovered: Vec<f64> = out
        .group("uns")
        .unwrap()
        .group("sparse_uns")
        .unwrap()
        .dataset("data")
        .unwrap()
        .read_1d::<f64>()
        .unwrap()
        .to_vec();
    assert_eq!(recovered, vec![1.0, 2.0, 3.0], "sparse data arrays lost");
}
