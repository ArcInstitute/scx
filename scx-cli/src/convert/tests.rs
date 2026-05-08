// Integration tests for h5ad/10x conversion
// All tests gated behind #[cfg(feature = "hdf5")] (in mod.rs)

use std::path::Path;

use arrow::array::Array;
use arrow::datatypes::DataType;
use hdf5::types::VarLenUnicode;

/// Convert &str to VarLenUnicode, stripping NUL bytes if present.
fn vlu(s: &str) -> VarLenUnicode {
    s.parse::<VarLenUnicode>().unwrap_or_else(|_| {
        let cleaned: String = s.chars().filter(|&c| c != '\0').collect();
        cleaned
            .parse::<VarLenUnicode>()
            .expect("cleaned string should have no NUL bytes")
    })
}
use scx_codec::{CodecId, ValueEncoding};
use scx_format::reader::ScxReader;

use super::csc_transpose::csc_to_csr;
use super::detect::{detect_input_format, detect_matrix_format, InputFormat, MatrixFormat};
use scx_codec::value_encoding::is_integer_data;

use super::dtype::detect_value_encoding;
use super::pipeline::{h5ad_to_scx, scx_to_h5ad, tenx_to_scx, ConvertError, ConvertOptions};

// -----------------------------------------------------------------------
// Test helpers: create synthetic h5ad and 10x files
// -----------------------------------------------------------------------

/// Create a minimal h5ad file with known data.
fn create_test_h5ad(
    path: &Path,
    n_obs: usize,
    n_vars: usize,
    matrix_format: &str, // "csr", "csc", or "dense"
    include_extras: bool,
) {
    let file = hdf5::File::create(path).unwrap();

    // Build CSR data
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut data = Vec::new();

    for row in 0..n_obs {
        // Each row has 2-3 nonzeros with integer values, sorted by column
        let nnz_in_row = 2 + (row % 2);
        let mut row_entries: Vec<(usize, f32)> = Vec::new();
        for j in 0..nnz_in_row {
            let col = (row * 3 + j) % n_vars;
            let val = ((row * 7 + j * 3 + 1) % 200 + 1) as f32;
            row_entries.push((col, val));
        }
        // Sort by column and deduplicate
        row_entries.sort_by_key(|&(c, _)| c);
        row_entries.dedup_by_key(|e| e.0);
        for (col, val) in &row_entries {
            indices.push(*col as i32);
            data.push(*val);
        }
        indptr.push(data.len() as i64);
    }

    match matrix_format {
        "csr" => {
            let x = file.create_group("X").unwrap();
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
        }
        "csc" => {
            // Convert CSR to CSC for writing
            let (csc_indptr, csc_indices, csc_data) =
                csr_to_csc(&indptr, &indices, &data, n_obs, n_vars);

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

            let enc = vlu("csc_matrix");
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
        }
        "dense" => {
            // Build dense matrix from CSR
            let mut dense = vec![0.0f32; n_obs * n_vars];
            for row in 0..n_obs {
                let start = indptr[row] as usize;
                let end = indptr[row + 1] as usize;
                for idx in start..end {
                    let col = indices[idx] as usize;
                    dense[row * n_vars + col] = data[idx];
                }
            }
            let nd_arr = ndarray::Array2::from_shape_vec((n_obs, n_vars), dense).unwrap();
            file.new_dataset::<f32>()
                .shape([n_obs, n_vars])
                .create("X")
                .unwrap()
                .write(&nd_arr)
                .unwrap();
        }
        _ => panic!("unknown matrix format: {matrix_format}"),
    }

    // Write obs
    let obs = file.create_group("obs").unwrap();
    let obs_index: Vec<VarLenUnicode> = (0..n_obs).map(|i| vlu(&format!("cell_{i}"))).collect();
    obs.new_dataset::<VarLenUnicode>()
        .shape([n_obs])
        .create("_index")
        .unwrap()
        .write(&obs_index)
        .unwrap();

    let idx_name = vlu("_index");
    obs.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&idx_name)
        .unwrap();

    // Add numeric column
    let numeric_col: Vec<i32> = (0..n_obs).map(|i| (i * 10) as i32).collect();
    obs.new_dataset::<i32>()
        .shape([n_obs])
        .create("n_counts")
        .unwrap()
        .write(&numeric_col)
        .unwrap();

    // Write var
    let var = file.create_group("var").unwrap();
    let var_index: Vec<VarLenUnicode> = (0..n_vars).map(|i| vlu(&format!("gene_{i}"))).collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([n_vars])
        .create("_index")
        .unwrap()
        .write(&var_index)
        .unwrap();

    let var_idx_name = vlu("_index");
    var.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&var_idx_name)
        .unwrap();

    if include_extras {
        // obsm
        let obsm = file.create_group("obsm").unwrap();
        let pca_data: Vec<f32> = (0..n_obs * 2).map(|i| i as f32 * 0.1).collect();
        let nd_pca = ndarray::Array2::from_shape_vec((n_obs, 2), pca_data).unwrap();
        obsm.new_dataset::<f32>()
            .shape([n_obs, 2])
            .create("X_pca")
            .unwrap()
            .write(&nd_pca)
            .unwrap();

        // uns
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
            .create("version")
            .unwrap()
            .write_scalar(&2i64)
            .unwrap();

        // layers
        let layers = file.create_group("layers").unwrap();
        let raw_layer = layers.create_group("raw").unwrap();
        raw_layer
            .new_dataset::<i64>()
            .shape([indptr.len()])
            .create("indptr")
            .unwrap()
            .write(&indptr)
            .unwrap();
        raw_layer
            .new_dataset::<i32>()
            .shape([indices.len()])
            .create("indices")
            .unwrap()
            .write(&indices)
            .unwrap();
        raw_layer
            .new_dataset::<f32>()
            .shape([data.len()])
            .create("data")
            .unwrap()
            .write(&data)
            .unwrap();

        let enc = vlu("csr_matrix");
        raw_layer
            .new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&enc)
            .unwrap();
        let shape = [n_obs as i64, n_vars as i64];
        raw_layer
            .new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&shape)
            .unwrap();
    }
}

/// Convert CSR to CSC (helper for test data creation).
fn csr_to_csc(
    csr_indptr: &[i64],
    csr_indices: &[i32],
    csr_data: &[f32],
    n_rows: usize,
    n_cols: usize,
) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
    // Use the same scatter algorithm but row↔col swapped
    csc_to_csr(csr_indptr, csr_indices, csr_data, n_cols, n_rows).unwrap()
}

fn create_test_tenx_h5(path: &Path, n_cells: usize, n_genes: usize) {
    let file = hdf5::File::create(path).unwrap();
    let matrix = file.create_group("matrix").unwrap();

    // Shape: [n_genes, n_cells] (10x convention)
    let shape = [n_genes as i64, n_cells as i64];
    matrix
        .new_attr::<i64>()
        .shape([2])
        .create("shape")
        .unwrap()
        .write(&shape)
        .unwrap();

    // Build CSC data (genes as major axis)
    let mut csc_indptr = vec![0i64];
    let mut csc_indices = Vec::new();
    let mut csc_data = Vec::new();

    for col in 0..n_cells {
        // Each cell has 2 nonzero gene counts
        let g0 = (col * 2) % n_genes;
        let g1 = (col * 2 + 1) % n_genes;
        let mut row_indices = vec![g0, g1];
        row_indices.sort();
        row_indices.dedup();
        for &g in &row_indices {
            csc_indices.push(g as i32);
            csc_data.push(((col + g + 1) % 100 + 1) as f32);
        }
        csc_indptr.push(csc_data.len() as i64);
    }

    matrix
        .new_dataset::<i64>()
        .shape([csc_indptr.len()])
        .create("indptr")
        .unwrap()
        .write(&csc_indptr)
        .unwrap();
    matrix
        .new_dataset::<i32>()
        .shape([csc_indices.len()])
        .create("indices")
        .unwrap()
        .write(&csc_indices)
        .unwrap();
    matrix
        .new_dataset::<f32>()
        .shape([csc_data.len()])
        .create("data")
        .unwrap()
        .write(&csc_data)
        .unwrap();

    // Barcodes
    let barcodes: Vec<VarLenUnicode> = (0..n_cells).map(|i| vlu(&format!("AAAA-{i}"))).collect();
    matrix
        .new_dataset::<VarLenUnicode>()
        .shape([n_cells])
        .create("barcodes")
        .unwrap()
        .write(&barcodes)
        .unwrap();

    // Features
    let features = matrix.create_group("features").unwrap();
    let gene_ids: Vec<VarLenUnicode> = (0..n_genes).map(|i| vlu(&format!("ENSG{i:08}"))).collect();
    let gene_names: Vec<VarLenUnicode> = (0..n_genes).map(|i| vlu(&format!("Gene{i}"))).collect();
    let feature_types: Vec<VarLenUnicode> = (0..n_genes).map(|_| vlu("Gene Expression")).collect();

    features
        .new_dataset::<VarLenUnicode>()
        .shape([n_genes])
        .create("id")
        .unwrap()
        .write(&gene_ids)
        .unwrap();
    features
        .new_dataset::<VarLenUnicode>()
        .shape([n_genes])
        .create("name")
        .unwrap()
        .write(&gene_names)
        .unwrap();
    features
        .new_dataset::<VarLenUnicode>()
        .shape([n_genes])
        .create("feature_type")
        .unwrap()
        .write(&feature_types)
        .unwrap();
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

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
    h5ad_to_scx(&h5ad_path, &scx_path, &opts).unwrap();

    // Verify scx
    let reader = ScxReader::open(&scx_path).unwrap();
    assert_eq!(reader.n_obs(), n_obs as u64);
    assert_eq!(reader.n_vars(), n_vars as u64);

    // scx → h5ad
    scx_to_h5ad(&scx_path, &h5ad_out).unwrap();

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

#[test]
fn test_tenx_to_scx() {
    let dir = tempfile::tempdir().unwrap();
    let tenx_path = dir.path().join("test_10x.h5");
    let scx_path = dir.path().join("test_10x.scx");

    let n_cells = 20;
    let n_genes = 10;
    create_test_tenx_h5(&tenx_path, n_cells, n_genes);

    let opts = ConvertOptions::default();
    tenx_to_scx(&tenx_path, &scx_path, &opts).unwrap();

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
    h5ad_to_scx(&h5ad_path, &scx_path, &opts).unwrap();

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
    h5ad_to_scx(&h5ad_path, &scx_path, &opts).unwrap();

    // Compare with CSR version
    let csr_h5ad = dir.path().join("csr.h5ad");
    let csr_scx = dir.path().join("csr.scx");
    create_test_h5ad(&csr_h5ad, n_obs, n_vars, "csr", false);
    h5ad_to_scx(&csr_h5ad, &csr_scx, &opts).unwrap();

    let csc_reader = ScxReader::open(&scx_path).unwrap();
    let csr_reader = ScxReader::open(&csr_scx).unwrap();

    let csc_csr = csc_reader.read_all_csr_shards().unwrap();
    let csr_csr = csr_reader.read_all_csr_shards().unwrap();

    assert_eq!(csc_csr.shape, csr_csr.shape);
    assert_eq!(csc_csr.nnz(), csr_csr.nnz());
    // Data should match (same source matrix, just stored differently)
    assert_eq!(csc_csr.data, csr_csr.data);
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
    h5ad_to_scx(&h5ad_path, &scx_path, &opts).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    let uns = reader.read_uns().unwrap();
    assert_eq!(uns["species"], "human");
    assert_eq!(uns["count"], 42);
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
    h5ad_to_scx(&h5ad_path, &scx_path, &opts).unwrap();
    scx_to_h5ad(&scx_path, &h5ad_out).unwrap();

    // Verify categorical survived
    let reader = ScxReader::open(&scx_path).unwrap();
    let obs = reader.read_obs().unwrap();

    // Find the cell_type column
    let schema = obs.schema();
    let ct_idx = schema.index_of("cell_type").unwrap();
    let ct_col = obs.column(ct_idx);
    assert!(matches!(ct_col.data_type(), DataType::Dictionary(_, _)));
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
    let result = h5ad_to_scx(&tenx_path, &scx_path, &opts);
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
    let result = tenx_to_scx(&h5ad_path, &scx_path, &opts);
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
    let (enc, codec) = detect_value_encoding(&[1.0, 2.0, 255.0], None);
    assert_eq!(enc, ValueEncoding::Uint8);
    assert_eq!(codec, CodecId::Scx1);

    let (enc, codec) = detect_value_encoding(&[1.0, 256.0], None);
    assert_eq!(enc, ValueEncoding::Uint16);
    assert_eq!(codec, CodecId::Scx1);

    let (enc, codec) = detect_value_encoding(&[1.0, 70000.0], None);
    assert_eq!(enc, ValueEncoding::Uint32);
    assert_eq!(codec, CodecId::Scx1);

    let (enc, codec) = detect_value_encoding(&[0.5, 1.5], None);
    assert_eq!(enc, ValueEncoding::Float32);
    assert_eq!(codec, CodecId::Zstd);
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
    h5ad_to_scx(&h5ad_path, &scx_path, &opts).unwrap();

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
    h5ad_to_scx(&h5ad_path, &scx_path, &opts).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    // Should use Zstd for float data
    assert_eq!(reader.header().codec_id, CodecId::Zstd as u8);
}

/// convert h5ad → scx with `csc=always`, verify the output
/// has `has_csc()`, the expected CSC shard count, contiguous column
/// ranges, and densified contents matching the CSR data.
#[test]
fn test_h5ad_to_scx_csc_always() {
    use scx_format::section::SectionType;

    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("with_csc.scx");

    let n_obs = 12;
    let n_vars = 10;
    create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", true);

    let opts = ConvertOptions {
        csc: true,
        csc_cols_per_shard: 4, // → ceil(10/4) = 3 CSC shards
        ..ConvertOptions::default()
    };
    h5ad_to_scx(&h5ad_path, &scx_path, &opts).unwrap();

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
        let sh = scx_format::shard::ShardHeader::read_from(&mut std::io::Cursor::new(
            &section[..scx_format::shard::SHARD_HEADER_SIZE],
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
        csc: true,
        csc_cols_per_shard: 5, // → ceil(12/5) = 3 CSC shards
        ..ConvertOptions::default()
    };
    tenx_to_scx(&tenx_path, &scx_path, &opts).unwrap();

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
    h5ad_to_scx(&h5ad_path, &scx_path, &opts).unwrap();

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
    assert_eq!(detect_matrix_format(&file).unwrap(), MatrixFormat::Csr);

    // 10x
    let tenx_path = dir.path().join("det.h5");
    create_test_tenx_h5(&tenx_path, 5, 3);
    let file = hdf5::File::open(&tenx_path).unwrap();
    assert_eq!(detect_input_format(&file).unwrap(), InputFormat::TenX);
}
