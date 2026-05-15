// Integration tests for h5ad/10x conversion
// All tests gated behind #[cfg(feature = "hdf5")] (in mod.rs)

use std::num::NonZeroU32;
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
use super::stream::{CsrShardStream, StreamedCsrShard};
use super::warnings::WarningSink;

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
    let (enc, codec) = detect_value_encoding(&[1.0, 2.0, 255.0], None).unwrap();
    assert_eq!(enc, ValueEncoding::Uint8);
    assert_eq!(codec, CodecId::Scx1);

    let (enc, codec) = detect_value_encoding(&[1.0, 256.0], None).unwrap();
    assert_eq!(enc, ValueEncoding::Uint16);
    assert_eq!(codec, CodecId::Scx1);

    let (enc, codec) = detect_value_encoding(&[1.0, 70000.0], None).unwrap();
    assert_eq!(enc, ValueEncoding::Uint32);
    assert_eq!(codec, CodecId::Scx1);

    let (enc, codec) = detect_value_encoding(&[0.5, 1.5], None).unwrap();
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
    assert_eq!(detect_matrix_format(&file).unwrap(), MatrixFormat::Csr);

    // 10x
    let tenx_path = dir.path().join("det.h5");
    create_test_tenx_h5(&tenx_path, 5, 3);
    let file = hdf5::File::open(&tenx_path).unwrap();
    assert_eq!(detect_input_format(&file).unwrap(), InputFormat::TenX);
}

// -----------------------------------------------------------------------
// Phase D: h5mu / MuData round-trip tests
// -----------------------------------------------------------------------

/// Create a minimal h5mu file with `n_obs` cells across two
/// modalities (rna with `rna_n_vars` features and adt with
/// `adt_n_vars` features). Mirrors `create_test_h5ad` for the per-
/// modality `/mod/{name}/X` and `/mod/{name}/var` blocks plus an
/// outer `/obs`.
fn create_test_h5mu(path: &Path, n_obs: usize, rna_n_vars: usize, adt_n_vars: usize) {
    let file = hdf5::File::create(path).unwrap();

    // Outer obs.
    let obs = file.create_group("obs").unwrap();
    let obs_index: Vec<VarLenUnicode> = (0..n_obs).map(|i| vlu(&format!("cell_{i}"))).collect();
    obs.new_dataset::<VarLenUnicode>()
        .shape([n_obs])
        .create("_index")
        .unwrap()
        .write(&obs_index)
        .unwrap();
    obs.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&vlu("_index"))
        .unwrap();

    // /mod group.
    let mod_group = file.create_group("mod").unwrap();

    let write_modality = |group: &hdf5::Group, n_vars: usize| {
        // Build trivial CSR data: one nonzero per row at column (row % n_vars).
        let mut indptr = vec![0i64];
        let mut indices: Vec<i32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        for row in 0..n_obs {
            indices.push((row % n_vars) as i32);
            data.push((row + 1) as f32);
            indptr.push(data.len() as i64);
        }

        let x = group.create_group("X").unwrap();
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
        x.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("csr_matrix"))
            .unwrap();
        x.new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[n_obs as i64, n_vars as i64])
            .unwrap();

        // var
        let var = group.create_group("var").unwrap();
        let var_index: Vec<VarLenUnicode> =
            (0..n_vars).map(|i| vlu(&format!("feat_{i}"))).collect();
        var.new_dataset::<VarLenUnicode>()
            .shape([n_vars])
            .create("_index")
            .unwrap()
            .write(&var_index)
            .unwrap();
        var.new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&vlu("_index"))
            .unwrap();
    };

    let rna = mod_group.create_group("rna").unwrap();
    write_modality(&rna, rna_n_vars);
    let adt = mod_group.create_group("adt").unwrap();
    write_modality(&adt, adt_n_vars);
}

/// Round-trip: create an h5mu fixture → h5mu_to_scx → ScxReader
/// reports two modalities with the right names, var counts, and
/// CSR shard counts. Per-modality reads return non-empty data.
#[test]
fn test_h5mu_round_trip() {
    use super::mudata_pipeline::h5mu_to_scx;

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
    use super::mudata_pipeline::h5mu_to_scx;
    use scx_format::section::SectionType;

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
        let sh = scx_format::shard::ShardHeader::read_from(&mut std::io::Cursor::new(
            &section[..scx_format::shard::SHARD_HEADER_SIZE],
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
    use super::mudata_pipeline::h5mu_to_scx;
    use super::mudata_write::scx_to_h5mu;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_in = dir.path().join("in.h5mu");
    let scx_path = dir.path().join("mid.scx");
    let h5mu_out = dir.path().join("out.h5mu");
    create_test_h5mu(&h5mu_in, 8, 30, 5);

    let opts = ConvertOptions::default();
    h5mu_to_scx(&h5mu_in, &scx_path, &opts, &mut WarningSink::log()).unwrap();
    scx_to_h5mu(&scx_path, &h5mu_out).unwrap();

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
    use super::mudata_pipeline::h5mu_to_scx;
    use super::mudata_write::scx_modality_to_h5ad;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_in = dir.path().join("in.h5mu");
    let scx_path = dir.path().join("mid.scx");
    let h5ad_out = dir.path().join("rna.h5ad");
    create_test_h5mu(&h5mu_in, 6, 40, 7);

    let opts = ConvertOptions::default();
    h5mu_to_scx(&h5mu_in, &scx_path, &opts, &mut WarningSink::log()).unwrap();
    scx_modality_to_h5ad(&scx_path, &h5ad_out, "rna").unwrap();

    let file = hdf5::File::open(&h5ad_out).unwrap();
    assert!(file.group("X").is_ok());
    assert!(file.group("obs").is_ok());
    assert!(file.group("var").is_ok());
    // var should have 40 entries (rna's count, not adt's 7).
    let var_idx = file.dataset("var/_index").unwrap();
    assert_eq!(var_idx.shape()[0], 40);
}

// --- Phase F coverage ---------------------------------------------------

/// Phase F.1: `scx info` exposes per-modality counts via the
/// modality table accessor. We don't capture stdout here — instead
/// we verify the underlying accessors (which `run_info` formats).
#[test]
fn test_info_modality_table_exposed() {
    use super::mudata_pipeline::h5mu_to_scx;

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
    use super::mudata_pipeline::h5mu_to_scx;

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

/// Phase F.4: `scx merge` of two multimodal files that match in
/// every modality is also explicitly rejected (multimodal merge is
/// not yet implemented). Refusing is safer than silently producing
/// a flattened output.
#[test]
fn test_merge_multimodal_match_still_unsupported() {
    use super::mudata_pipeline::h5mu_to_scx;

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

    let err = scx_ops::merge(&[&scx_a, &scx_b], &merged).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("multimodal") || msg.contains("modality"),
        "merge error should mention multimodal limitation; got: {msg}"
    );
}

/// Phase F.4: `scx compact` on a multimodal file is rejected with a
/// clear error directing to subset-then-compact.
#[test]
fn test_compact_multimodal_unsupported() {
    use super::mudata_pipeline::h5mu_to_scx;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_in = dir.path().join("in.h5mu");
    let scx_in = dir.path().join("multi.scx");
    let scx_out = dir.path().join("compacted.scx");
    create_test_h5mu(&h5mu_in, 6, 20, 5);

    let opts = ConvertOptions::default();
    h5mu_to_scx(&h5mu_in, &scx_in, &opts, &mut WarningSink::log()).unwrap();

    let err = scx_ops::compact(&scx_in, &scx_out).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("multimodal"),
        "compact error should mention multimodal limitation; got: {msg}"
    );
}

/// Phase F.3: `scx_ops::append` (with `modality_id` set) stamps shards with the
/// chosen modality_id and updates the modality table's per-modality
/// counts. We exercise the full path: build a multimodal file from
/// h5mu, append to its rna modality, verify the rna modality's
/// counts went up while the adt modality is untouched.
#[test]
fn test_append_for_modality_updates_table() {
    use super::mudata_pipeline::h5mu_to_scx;
    use scx_format::section::SectionType;

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

// -----------------------------------------------------------------------
// Phase 1 — streaming reader (XStreamReader / open_x_streaming /
// open_layer_streaming). Verifies the row-range reader yields the same
// data as the existing bulk `read_x_matrix` path and refuses
// unsupported on-disk layouts (CSC, dense).
// -----------------------------------------------------------------------

use super::h5ad_read::read_x_matrix;
use super::h5ad_stream::{open_layer_streaming, open_x_streaming};

/// Drive `next_shard` to exhaustion, returning the concatenated
/// (indptr, indices, values) in their full-matrix layout. Re-bases
/// the per-shard local indptr to a global running total.
fn drain_streaming(
    reader: &mut super::h5ad_stream::XStreamReader,
    target_rows: usize,
) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let n_obs = reader.n_obs;
    let mut full_indptr: Vec<u64> = Vec::with_capacity(n_obs + 1);
    full_indptr.push(0);
    let mut full_indices: Vec<u32> = Vec::new();
    let mut full_values: Vec<f32> = Vec::new();
    let mut row_count = 0usize;

    while let Some(shard) = reader.next_shard(target_rows) {
        let shard = shard.expect("shard read failed");
        assert_eq!(
            shard.row_start, row_count,
            "row_start must equal cumulative row count"
        );
        let base = *full_indptr.last().unwrap();
        for &v in &shard.indptr[1..] {
            full_indptr.push(base + v);
        }
        full_indices.extend_from_slice(&shard.indices);
        full_values.extend_from_slice(&shard.values);
        row_count += shard.n_rows;
    }
    assert_eq!(row_count, n_obs);
    (full_indptr, full_indices, full_values)
}

#[test]
fn streaming_csr_round_trip_matches_bulk_reader() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stream_csr.h5ad");
    let n_obs = 37;
    let n_vars = 15;
    create_test_h5ad(&path, n_obs, n_vars, "csr", false);

    let (bulk_indptr, bulk_indices, bulk_data, bulk_n_obs, bulk_n_vars) = {
        let f = hdf5::File::open(&path).unwrap();
        read_x_matrix(&f, MatrixFormat::Csr).unwrap()
    };
    assert_eq!(bulk_n_obs, n_obs);
    assert_eq!(bulk_n_vars, n_vars);

    let file = hdf5::File::open(&path).unwrap();
    let mut reader = open_x_streaming(&file, "X", MatrixFormat::Csr).unwrap();
    assert_eq!(reader.n_obs, n_obs);
    assert_eq!(reader.n_vars, n_vars);

    let (stream_indptr, stream_indices, stream_values) = drain_streaming(&mut reader, 10);

    let bulk_indptr_u64: Vec<u64> = bulk_indptr.iter().map(|&v| v as u64).collect();
    let bulk_indices_u32: Vec<u32> = bulk_indices.iter().map(|&v| v as u32).collect();
    assert_eq!(stream_indptr, bulk_indptr_u64);
    assert_eq!(stream_indices, bulk_indices_u32);
    assert_eq!(stream_values, bulk_data);
}

#[test]
fn streaming_layer_matches_x_for_synthetic_fixture() {
    // create_test_h5ad with include_extras=true writes a "raw" layer
    // that mirrors X exactly. Streaming the layer must therefore
    // yield the same content as streaming X.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stream_layer.h5ad");
    let n_obs = 24;
    let n_vars = 10;
    create_test_h5ad(&path, n_obs, n_vars, "csr", true);

    let file = hdf5::File::open(&path).unwrap();
    let mut x_reader = open_x_streaming(&file, "X", MatrixFormat::Csr).unwrap();
    let (x_indptr, x_indices, x_values) = drain_streaming(&mut x_reader, 7);

    let mut layer_reader = open_layer_streaming(&file, "raw").unwrap();
    assert_eq!(layer_reader.n_obs, n_obs);
    assert_eq!(layer_reader.n_vars, n_vars);
    let (l_indptr, l_indices, l_values) = drain_streaming(&mut layer_reader, 7);

    assert_eq!(x_indptr, l_indptr);
    assert_eq!(x_indices, l_indices);
    assert_eq!(x_values, l_values);
}

#[test]
fn streaming_rejects_csc_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stream_csc.h5ad");
    create_test_h5ad(&path, 6, 5, "csc", false);

    let file = hdf5::File::open(&path).unwrap();
    let err = open_x_streaming(&file, "X", MatrixFormat::Csc).unwrap_err();
    match err {
        ConvertError::StreamingUnsupported(msg) => {
            assert!(
                msg.contains("CSC"),
                "error message should mention CSC; got: {msg}"
            );
        }
        other => panic!("expected StreamingUnsupported, got {other:?}"),
    }
}

#[test]
fn streaming_rejects_dense_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stream_dense.h5ad");
    create_test_h5ad(&path, 6, 5, "dense", false);

    let file = hdf5::File::open(&path).unwrap();
    let err = open_x_streaming(&file, "X", MatrixFormat::Dense).unwrap_err();
    match err {
        ConvertError::StreamingUnsupported(msg) => {
            assert!(
                msg.contains("dense"),
                "error message should mention dense; got: {msg}"
            );
        }
        other => panic!("expected StreamingUnsupported, got {other:?}"),
    }
}

#[test]
fn streaming_empty_matrix_yields_no_shards() {
    // Build a minimal h5ad with n_obs = 0 by hand — create_test_h5ad's
    // loop is unbounded but works at zero, producing indptr = [0] and
    // empty indices / data datasets.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stream_empty.h5ad");
    let n_vars = 5;
    {
        let file = hdf5::File::create(&path).unwrap();
        let x = file.create_group("X").unwrap();
        let indptr: Vec<i64> = vec![0];
        x.new_dataset::<i64>()
            .shape([1])
            .create("indptr")
            .unwrap()
            .write(&indptr)
            .unwrap();
        // hdf5 doesn't allow zero-sized writes via the typed builder;
        // create empty datasets via shape=[0] and skip the .write().
        x.new_dataset::<i32>().shape([0]).create("indices").unwrap();
        x.new_dataset::<f32>().shape([0]).create("data").unwrap();
        x.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("csr_matrix"))
            .unwrap();
        x.new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[0i64, n_vars as i64])
            .unwrap();
    }

    let file = hdf5::File::open(&path).unwrap();
    let mut reader = open_x_streaming(&file, "X", MatrixFormat::Csr).unwrap();
    assert_eq!(reader.n_obs, 0);
    assert_eq!(reader.n_vars, n_vars);
    assert!(reader.next_shard(16).is_none());
}

#[test]
fn streaming_handles_empty_rows() {
    // Hand-build a CSR fixture where some rows have zero nnz. The
    // streaming reader must yield row counts unchanged and nnz == 4.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stream_empty_rows.h5ad");
    let n_vars: usize = 4;
    let indptr: Vec<i64> = vec![0, 2, 2, 2, 4, 4]; // 5 rows, rows 1/2/4 empty
    let indices: Vec<i32> = vec![0, 1, 2, 3];
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    {
        let file = hdf5::File::create(&path).unwrap();
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
        x.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("csr_matrix"))
            .unwrap();
        x.new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[5i64, n_vars as i64])
            .unwrap();
    }

    let file = hdf5::File::open(&path).unwrap();
    let mut reader = open_x_streaming(&file, "X", MatrixFormat::Csr).unwrap();
    let (full_indptr, full_indices, full_values) = drain_streaming(&mut reader, 2);
    assert_eq!(full_indptr, vec![0, 2, 2, 2, 4, 4]);
    assert_eq!(full_indices, vec![0u32, 1, 2, 3]);
    assert_eq!(full_values, vec![1.0, 2.0, 3.0, 4.0]);
}

// -----------------------------------------------------------------------
// Phase 8 — end-to-end streaming pipeline (h5ad_to_scx_streaming).
//
// These differ from the Phase 1 streaming reader tests above: they
// exercise the full converter (open input → write obs/var → stream
// shards → write metadata → finish) and assert the resulting SCX
// file matches the materialising path. CSC sidecar parity and
// multi-layer round-trips are the load-bearing cases.
// -----------------------------------------------------------------------

use super::pipeline::{h5ad_to_scx_streaming, StreamingOverrides};
use scx_format::section::SectionType as FmtSectionType;

fn streaming_opts(shard_size: u32) -> ConvertOptions {
    ConvertOptions {
        shard_target_rows: shard_size,
        codec: None,
        csc: false,
        csc_cols_per_shard: 5000,
        tool: "scx-cli".into(),
        ..ConvertOptions::default()
    }
}

#[test]
fn streaming_round_trip_matches_non_streaming() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("rt.h5ad");
    create_test_h5ad(&h5ad, 53, 17, "csr", false);

    let scx_stream = dir.path().join("stream.scx");
    let scx_bulk = dir.path().join("bulk.scx");

    let opts = streaming_opts(16);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_stream,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    h5ad_to_scx(&h5ad, &scx_bulk, &opts, &mut WarningSink::log()).unwrap();

    let a = ScxReader::open(&scx_stream).unwrap();
    let b = ScxReader::open(&scx_bulk).unwrap();
    assert_eq!(a.header().n_obs, b.header().n_obs);
    assert_eq!(a.header().n_vars, b.header().n_vars);
    assert_eq!(a.header().nnz, b.header().nnz);
    assert_eq!(a.header().n_csr_shards, b.header().n_csr_shards);

    let csr_a = a.read_all_csr_shards().unwrap();
    let csr_b = b.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.shape, csr_b.shape);
    assert_eq!(csr_a.indptr, csr_b.indptr);
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data);
}

#[test]
fn streaming_empty_n_obs_produces_valid_scx() {
    // Hand-build a 0-row CSR h5ad — `create_test_h5ad`'s loop assumes
    // n_obs > 0 so we skip it.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("empty.h5ad");
    let n_vars: usize = 7;
    {
        let file = hdf5::File::create(&h5ad).unwrap();
        let x = file.create_group("X").unwrap();
        x.new_dataset::<i64>()
            .shape([1])
            .create("indptr")
            .unwrap()
            .write(&[0i64])
            .unwrap();
        x.new_dataset::<i32>().shape([0]).create("indices").unwrap();
        x.new_dataset::<f32>().shape([0]).create("data").unwrap();
        x.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("csr_matrix"))
            .unwrap();
        x.new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[0i64, n_vars as i64])
            .unwrap();

        // Minimal obs / var so write_obs / write_var don't error.
        let obs = file.create_group("obs").unwrap();
        let obs_index: Vec<VarLenUnicode> = Vec::new();
        obs.new_dataset::<VarLenUnicode>()
            .shape([0])
            .create("_index")
            .unwrap();
        let _ = obs_index;
        obs.new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&vlu("_index"))
            .unwrap();
        let var = file.create_group("var").unwrap();
        let var_idx: Vec<VarLenUnicode> = (0..n_vars).map(|i| vlu(&format!("g{i}"))).collect();
        var.new_dataset::<VarLenUnicode>()
            .shape([n_vars])
            .create("_index")
            .unwrap()
            .write(&var_idx)
            .unwrap();
        var.new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&vlu("_index"))
            .unwrap();
    }

    let scx = dir.path().join("empty.scx");
    let opts = streaming_opts(16);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(reader.header().n_obs, 0);
    assert_eq!(reader.header().n_vars, n_vars as u64);
    assert_eq!(reader.header().n_csr_shards, 0);
    assert_eq!(reader.header().nnz, 0);
}

#[test]
fn streaming_sets_index_dtype_1_when_n_vars_above_u16() {
    // 70_000 vars > u16::MAX (65535) → index_dtype must be 1 (u32 indices).
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("wide.h5ad");
    let n_obs: usize = 8;
    let n_vars: usize = 70_000;
    {
        let file = hdf5::File::create(&h5ad).unwrap();
        let x = file.create_group("X").unwrap();

        // One nnz per row, evenly spread across columns up to and
        // beyond the u16 limit so the on-disk index dtype must be u32
        // (i32 in scipy's CSR layout).
        let mut indptr = vec![0i64];
        let mut indices: Vec<i32> = Vec::with_capacity(n_obs);
        let mut data: Vec<f32> = Vec::with_capacity(n_obs);
        let stride = n_vars / n_obs;
        for row in 0..n_obs {
            indices.push((row * stride) as i32);
            data.push((row + 1) as f32);
            indptr.push(data.len() as i64);
        }

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
        x.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("csr_matrix"))
            .unwrap();
        x.new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[n_obs as i64, n_vars as i64])
            .unwrap();

        let obs = file.create_group("obs").unwrap();
        let obs_index: Vec<VarLenUnicode> = (0..n_obs).map(|i| vlu(&format!("c{i}"))).collect();
        obs.new_dataset::<VarLenUnicode>()
            .shape([n_obs])
            .create("_index")
            .unwrap()
            .write(&obs_index)
            .unwrap();
        obs.new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&vlu("_index"))
            .unwrap();

        let var = file.create_group("var").unwrap();
        // Wide var index: writing 70k VarLenUnicode strings is fast
        // enough for a test and exercises the read_dataframe_group
        // path on a large dimension.
        let var_index: Vec<VarLenUnicode> = (0..n_vars).map(|i| vlu(&format!("g{i}"))).collect();
        var.new_dataset::<VarLenUnicode>()
            .shape([n_vars])
            .create("_index")
            .unwrap()
            .write(&var_index)
            .unwrap();
        var.new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&vlu("_index"))
            .unwrap();
    }

    let scx = dir.path().join("wide.scx");
    let opts = streaming_opts(16);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(reader.header().n_vars, n_vars as u64);
    assert_eq!(
        reader.header().index_dtype,
        1,
        "n_vars > u16::MAX requires index_dtype = 1 (u32 indices)"
    );
}

#[test]
fn streaming_csc_on_disk_errors_at_pipeline_level() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("csc.h5ad");
    create_test_h5ad(&h5ad, 6, 5, "csc", false);

    let scx = dir.path().join("csc.scx");
    let opts = streaming_opts(16);
    let err = h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect_err("CSC-on-disk h5ad must be rejected by the streaming pipeline");
    match err {
        ConvertError::StreamingUnsupported(msg) => {
            assert!(
                msg.contains("CSC"),
                "expected CSC-on-disk error message; got: {msg}"
            );
        }
        other => panic!("expected StreamingUnsupported, got {other:?}"),
    }
}

#[test]
fn streaming_csc_always_emits_sidecar_matching_non_streaming() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("rt_csc.h5ad");
    create_test_h5ad(&h5ad, 41, 13, "csr", false);

    let scx_stream = dir.path().join("stream_csc.scx");
    let scx_bulk = dir.path().join("bulk_csc.scx");
    let opts = ConvertOptions {
        shard_target_rows: 16,
        codec: None,
        csc: true,
        csc_cols_per_shard: 5,
        tool: "scx-cli".into(),
        ..ConvertOptions::default()
    };
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_stream,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    h5ad_to_scx(&h5ad, &scx_bulk, &opts, &mut WarningSink::log()).unwrap();

    let a = ScxReader::open(&scx_stream).unwrap();
    let b = ScxReader::open(&scx_bulk).unwrap();

    // CSC sidecars present on both files, same shape.
    assert!(
        a.header().n_csc_shards >= 1,
        "streaming run should emit at least one CSC shard, got {}",
        a.header().n_csc_shards
    );
    assert_eq!(a.header().n_csc_shards, b.header().n_csc_shards);

    // Catalog CSC entries align.
    let csc_a: Vec<_> = a
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == FmtSectionType::CscShard)
        .collect();
    let csc_b: Vec<_> = b
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == FmtSectionType::CscShard)
        .collect();
    assert_eq!(csc_a.len(), csc_b.len());
    // Section lengths should match — `rebuild_csc_inplace` deterministic
    // on the same CSR.
    for (ea, eb) in csc_a.iter().zip(csc_b.iter()) {
        assert_eq!(ea.length, eb.length, "CSC shard length parity");
    }
}

#[test]
fn streaming_two_layer_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("layers.h5ad");
    // `create_test_h5ad` with include_extras=true writes a single
    // "raw" layer. Add a second layer by hand so we exercise the
    // multi-layer streaming loop.
    create_test_h5ad(&h5ad, 33, 9, "csr", true);
    {
        let file = hdf5::File::open_rw(&h5ad).unwrap();
        let layers = file.group("layers").unwrap();
        let counts = layers.create_group("counts").unwrap();
        // Reuse the same CSR triplet as "raw" — same shape, different
        // catalog name is enough to verify the multi-layer path.
        let raw = layers.group("raw").unwrap();
        let raw_indptr: Vec<i64> = raw.dataset("indptr").unwrap().read_1d().unwrap().to_vec();
        let raw_indices: Vec<i32> = raw.dataset("indices").unwrap().read_1d().unwrap().to_vec();
        let raw_data: Vec<f32> = raw.dataset("data").unwrap().read_1d().unwrap().to_vec();
        counts
            .new_dataset::<i64>()
            .shape([raw_indptr.len()])
            .create("indptr")
            .unwrap()
            .write(&raw_indptr)
            .unwrap();
        counts
            .new_dataset::<i32>()
            .shape([raw_indices.len()])
            .create("indices")
            .unwrap()
            .write(&raw_indices)
            .unwrap();
        counts
            .new_dataset::<f32>()
            .shape([raw_data.len()])
            .create("data")
            .unwrap()
            .write(&raw_data)
            .unwrap();
        counts
            .new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("csr_matrix"))
            .unwrap();
        counts
            .new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[33i64, 9])
            .unwrap();
    }

    let scx = dir.path().join("layers.scx");
    let opts = streaming_opts(8);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    let layer_entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == FmtSectionType::LayerCsrShard)
        .collect();
    let raw_count = layer_entries
        .iter()
        .filter(|e| e.name.starts_with("raw_shard_"))
        .count();
    let counts_count = layer_entries
        .iter()
        .filter(|e| e.name.starts_with("counts_shard_"))
        .count();
    assert!(raw_count >= 1, "expected at least one 'raw' layer shard");
    assert!(
        counts_count >= 1,
        "expected at least one 'counts' layer shard"
    );
}

#[test]
fn streaming_skips_unreadable_layer() {
    // A dense `/layers/{name}` group must not abort the streaming
    // convert — `open_layer_streaming` rejects dense layers, the
    // pipeline should warn and continue. Mirrors the non-streaming
    // `read_layers` best-effort behaviour.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("bad_layer.h5ad");
    let n_obs = 12;
    let n_vars = 5;
    create_test_h5ad(&h5ad, n_obs, n_vars, "csr", true);
    {
        let file = hdf5::File::open_rw(&h5ad).unwrap();
        let layers = file.group("layers").unwrap();
        let dense = layers.create_group("dense_bad").unwrap();
        // Mark as a dense matrix so `open_layer_streaming` rejects it.
        dense
            .new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("array"))
            .unwrap();
        dense
            .new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[n_obs as i64, n_vars as i64])
            .unwrap();
    }

    let scx = dir.path().join("bad_layer.scx");
    let opts = streaming_opts(8);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect("streaming convert must skip the bad layer, not abort");

    let reader = ScxReader::open(&scx).unwrap();
    let layer_entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == FmtSectionType::LayerCsrShard)
        .collect();
    let raw_count = layer_entries
        .iter()
        .filter(|e| e.name.starts_with("raw_shard_"))
        .count();
    let dense_count = layer_entries
        .iter()
        .filter(|e| e.name.starts_with("dense_bad_shard_"))
        .count();
    assert!(
        raw_count >= 1,
        "expected the valid 'raw' layer to still be converted"
    );
    assert_eq!(
        dense_count, 0,
        "the dense 'dense_bad' layer must be silently skipped, not emit shards"
    );
}

#[test]
fn streaming_provenance_escapes_path_quotes() {
    // A path containing a `"` would break the previous
    // `format!`-built JSON; the `serde_json::json!` construction
    // must produce parseable JSON for any path.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("has\"quote.h5ad");
    create_test_h5ad(&h5ad, 8, 4, "csr", false);

    let scx = dir.path().join("out.scx");
    let opts = streaming_opts(4);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    let prov = reader.read_provenance().unwrap();
    assert_eq!(prov.operations.len(), 1);
    let entry = &prov.operations[0];
    assert_eq!(entry.action, "convert");
    assert_eq!(entry.tool, "scx-cli");
    // `params_json` must round-trip through serde_json::from_str —
    // proves the path was escaped correctly.
    let parsed: serde_json::Value =
        serde_json::from_str(&entry.params_json).expect("params_json must be valid JSON");
    assert_eq!(parsed["format"], "h5ad");
    assert_eq!(parsed["stream"], true);
    let recorded = parsed["input"]
        .as_str()
        .expect("input field must be a string");
    assert!(
        recorded.ends_with("has\"quote.h5ad"),
        "input path must contain the literal quote, got {recorded:?}"
    );
}

#[test]
fn streaming_provenance_uses_configured_tool_name() {
    // `ConvertOptions::tool` must flow through to the provenance
    // entry verbatim — `pyscx` overrides it to "pyscx" so the
    // recorded provenance reflects the actual caller.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("rt.h5ad");
    create_test_h5ad(&h5ad, 6, 3, "csr", false);

    let scx = dir.path().join("out.scx");
    let mut opts = streaming_opts(4);
    opts.tool = "pyscx".into();
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    let prov = reader.read_provenance().unwrap();
    assert_eq!(prov.operations.len(), 1);
    assert_eq!(prov.operations[0].tool, "pyscx");
}

#[test]
fn streaming_through_trait_object() {
    // Phase 0 acceptance criterion 5: the existing concrete reader
    // must drive the streaming pipeline equivalently when accessed
    // through `&mut dyn CsrShardStream`. Drains a fixture twice —
    // once through the inherent `next_shard` (which the in-tree
    // pipeline uses today) and once through `next_csr_shard` on the
    // trait — and asserts every emitted shard matches.
    use super::h5ad_stream::open_x_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("rt.h5ad");
    create_test_h5ad(&h5ad, 53, 7, "csr", false);
    let file = hdf5::File::open(&h5ad).unwrap();

    let mut concrete = open_x_streaming(&file, "X", MatrixFormat::Csr).unwrap();
    let mut trait_reader = open_x_streaming(&file, "X", MatrixFormat::Csr).unwrap();
    let dyn_reader: &mut dyn CsrShardStream = &mut trait_reader;

    let target = 16usize;
    let mut total_concrete_rows: usize = 0;
    let mut total_trait_rows: u64 = 0;

    loop {
        let lhs = concrete.next_shard(target);
        let rhs: Option<StreamedCsrShard> = dyn_reader.next_csr_shard(target).unwrap();
        match (lhs, rhs) {
            (None, None) => break,
            (None, Some(_)) | (Some(_), None) => {
                panic!("inherent and trait drains disagreed on termination")
            }
            (Some(Ok(l)), Some(r)) => {
                assert_eq!(l.row_start as u64, r.row_start);
                assert_eq!(l.n_rows as u32, r.n_rows);
                assert_eq!(l.indptr, r.indptr);
                assert_eq!(l.indices, r.indices);
                assert_eq!(l.values, r.values);
                total_concrete_rows += l.n_rows;
                total_trait_rows += r.n_rows as u64;
            }
            (Some(Err(e)), _) => panic!("inherent reader returned error: {e}"),
        }
    }
    assert_eq!(total_concrete_rows, 53);
    assert_eq!(total_trait_rows, 53);

    // Trait-object exposes shape and source name.
    assert_eq!(dyn_reader.n_obs(), 53);
    assert_eq!(dyn_reader.n_vars(), 7);
    assert_eq!(dyn_reader.source_matrix_name(), "X");
}
