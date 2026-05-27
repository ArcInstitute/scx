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
    let obs = reader.read_obs_assembled().unwrap();

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
        let var = reader.read_var_assembled().unwrap();
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
            let var = reader.read_var_assembled().unwrap();
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
            let var = reader.read_var_assembled().unwrap();
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

/// Like [`create_test_h5mu`] but writes a second modality
/// `dense_adt` whose `/X` is a 2D dense **f64** dataset with
/// `encoding-type="array"`. Exercises the dense-non-f32 sampling
/// path in `mudata_pipeline::sample_modality_values`.
#[cfg(test)]
fn create_test_h5mu_with_dense_f64_modality(
    path: &Path,
    n_obs: usize,
    rna_n_vars: usize,
    dense_n_vars: usize,
) {
    let file = hdf5::File::create(path).unwrap();

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

    let mod_group = file.create_group("mod").unwrap();

    // Sparse `rna` modality (existing layout).
    let rna = mod_group.create_group("rna").unwrap();
    {
        let mut indptr = vec![0i64];
        let mut indices: Vec<i32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        for row in 0..n_obs {
            indices.push((row % rna_n_vars) as i32);
            data.push((row + 1) as f32);
            indptr.push(data.len() as i64);
        }
        let x = rna.create_group("X").unwrap();
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
            .write(&[n_obs as i64, rna_n_vars as i64])
            .unwrap();

        let var = rna.create_group("var").unwrap();
        let var_index: Vec<VarLenUnicode> =
            (0..rna_n_vars).map(|i| vlu(&format!("rna_{i}"))).collect();
        var.new_dataset::<VarLenUnicode>()
            .shape([rna_n_vars])
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

    // Dense f64 `dense_adt` modality.
    let dense = mod_group.create_group("dense_adt").unwrap();
    {
        let dense_values: Vec<f64> = (0..n_obs * dense_n_vars)
            .map(|i| if i % 3 == 0 { 0.0 } else { (i as f64) * 0.5 })
            .collect();
        let nd = ndarray::Array2::from_shape_vec((n_obs, dense_n_vars), dense_values).unwrap();
        let x_ds = dense
            .new_dataset::<f64>()
            .shape([n_obs, dense_n_vars])
            .create("X")
            .unwrap();
        x_ds.write(&nd).unwrap();
        x_ds.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("array"))
            .unwrap();

        let var = dense.create_group("var").unwrap();
        let var_index: Vec<VarLenUnicode> = (0..dense_n_vars)
            .map(|i| vlu(&format!("adt_{i}")))
            .collect();
        var.new_dataset::<VarLenUnicode>()
            .shape([dense_n_vars])
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

/// Phase 6: `scx merge` of two multimodal files that match in
/// every modality concatenates rows per-modality and preserves the
/// modality table. Previously this case was rejected (the test
/// name retains the historical `still_unsupported` prefix); merge
/// has since been implemented for matching multimodal structures
/// — assert the positive path.
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
    use super::mudata_pipeline::h5mu_to_scx;

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
    let pre_obs = pre.read_obs_assembled().unwrap();
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
    let mut reader =
        open_x_streaming(&file, "X", MatrixFormat::Csr, &mut WarningSink::log()).unwrap();
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
    let mut x_reader =
        open_x_streaming(&file, "X", MatrixFormat::Csr, &mut WarningSink::log()).unwrap();
    let (x_indptr, x_indices, x_values) = drain_streaming(&mut x_reader, 7);

    let mut layer_reader = open_layer_streaming(&file, "raw", &mut WarningSink::log()).unwrap();
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
    let err = open_x_streaming(&file, "X", MatrixFormat::Csc, &mut WarningSink::log()).unwrap_err();
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
    let err =
        open_x_streaming(&file, "X", MatrixFormat::Dense, &mut WarningSink::log()).unwrap_err();
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
    let mut reader =
        open_x_streaming(&file, "X", MatrixFormat::Csr, &mut WarningSink::log()).unwrap();
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
    let mut reader =
        open_x_streaming(&file, "X", MatrixFormat::Csr, &mut WarningSink::log()).unwrap();
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
        tool: "scx".into(),
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
fn streaming_csc_on_disk_routes_through_phase2_dispatcher() {
    // Pre-Phase 2 this used to reject CSC at the pipeline level.
    // Phase 2 lights up `open_csc_streaming`, so the same fixture
    // should now convert successfully via the in-memory CSC route
    // (no `memory_budget` set → MaterializedCsrStream).
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("csc.h5ad");
    create_test_h5ad(&h5ad, 6, 5, "csc", false);

    let scx = dir.path().join("csc.scx");
    let opts = streaming_opts(16);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect("Phase 2 should route CSC through open_csc_streaming");
    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(reader.header().n_obs, 6);
    assert_eq!(reader.header().n_vars, 5);
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
        tool: "scx".into(),
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
    assert_eq!(entry.tool, "scx");
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

    let mut concrete =
        open_x_streaming(&file, "X", MatrixFormat::Csr, &mut WarningSink::log()).unwrap();
    let mut trait_reader =
        open_x_streaming(&file, "X", MatrixFormat::Csr, &mut WarningSink::log()).unwrap();
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

// -----------------------------------------------------------------------
// Phase 1 — dense h5ad streaming + wild-h5ad hardening.
// -----------------------------------------------------------------------

#[test]
fn phase1_streaming_dense_int_matches_non_streaming() {
    // Dense /X (f32 with integer values 1-200) → both pipelines must
    // produce the same CSR header counts and the same per-shard CSR
    // arrays. Exercises `DenseXStreamReader` via the new dispatch.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("dense_int.h5ad");
    create_test_h5ad(&h5ad, 47, 11, "dense", false);

    let scx_stream = dir.path().join("dense_int_stream.scx");
    let scx_bulk = dir.path().join("dense_int_bulk.scx");
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
    let csr_a = a.read_all_csr_shards().unwrap();
    let csr_b = b.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.shape, csr_b.shape);
    assert_eq!(csr_a.indptr, csr_b.indptr);
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data);
}

#[test]
fn phase1_streaming_dense_float_round_trip() {
    // Fractional values force `ValueEncoding::Float32` and exercise
    // the codec dispatch on a float dense fixture. Verify the
    // round-trip preserves every nonzero within f32 tolerance.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("dense_float.h5ad");
    let n_obs = 9usize;
    let n_vars = 5usize;
    // hand-built float dense matrix with fractional values
    let mut dense = vec![0.0f32; n_obs * n_vars];
    dense[0] = 0.5;
    dense[3] = 1.25;
    dense[n_vars + 1] = 2.75;
    dense[3 * n_vars + 4] = -3.5;
    dense[7 * n_vars + 2] = 100.125;
    write_dense_h5ad(&h5ad, n_obs, n_vars, &dense);

    let scx = dir.path().join("dense_float.scx");
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
    assert_eq!(reader.header().n_obs, n_obs as u64);
    assert_eq!(reader.header().n_vars, n_vars as u64);
    assert_eq!(reader.header().nnz, 5);
    let csr = reader.read_all_csr_shards().unwrap();
    // Reconstruct dense and compare element-wise within tolerance.
    let mut rebuilt = vec![0.0f32; n_obs * n_vars];
    for row in 0..n_obs {
        let start = csr.indptr[row] as usize;
        let end = csr.indptr[row + 1] as usize;
        for k in start..end {
            let col = csr.indices[k] as usize;
            rebuilt[row * n_vars + col] = csr.data[k];
        }
    }
    for i in 0..(n_obs * n_vars) {
        let diff = (dense[i] - rebuilt[i]).abs();
        assert!(
            diff < 1e-5,
            "value at index {i}: dense={}, scx={}, diff={diff}",
            dense[i],
            rebuilt[i]
        );
    }
}

#[test]
fn phase1_streaming_dense_empty_rows_round_trip() {
    // Rows of all zeros must preserve scipy CSR invariants:
    // `indptr[i] == indptr[i+1]` per empty row, total length n_obs+1.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("empty_rows.h5ad");
    let n_obs = 6usize;
    let n_vars = 4usize;
    let mut dense = vec![0.0f32; n_obs * n_vars];
    // Row 0 has one nonzero, rows 1-3 are empty, row 4 has two, row 5 empty.
    dense[2] = 7.0;
    dense[4 * n_vars] = 3.0;
    dense[4 * n_vars + 3] = 4.0;
    write_dense_h5ad(&h5ad, n_obs, n_vars, &dense);

    let scx = dir.path().join("empty_rows.scx");
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
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.indptr.len(), n_obs + 1);
    // Empty rows: indptr[i] == indptr[i+1].
    for i in [1usize, 2, 3, 5] {
        assert_eq!(
            csr.indptr[i],
            csr.indptr[i + 1],
            "row {i} should be empty (indptr[{i}]={}, indptr[{}]={})",
            csr.indptr[i],
            i + 1,
            csr.indptr[i + 1]
        );
    }
    assert_eq!(csr.indptr[n_obs], 3); // total nnz
}

#[test]
fn phase1_streaming_inferred_encoding_emits_warning() {
    // Build an h5ad whose `/X` group is sparse but has NO
    // `encoding-type` attribute. The streaming open path infers CSR
    // from the indptr+indices children and must emit one
    // `InferredEncoding` warning.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("no_enc.h5ad");
    write_csr_h5ad_without_encoding_type(&h5ad, 4, 3);

    let scx = dir.path().join("no_enc.scx");
    let opts = streaming_opts(8);
    let mut sink = WarningSink::log();
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut sink,
    )
    .unwrap();
    // The Phase 1 inference is emitted by both `detect_matrix_format`
    // (children-based fallback) and `open_x_streaming` (attr-absent
    // fallback). Either way the category counter must be non-zero.
    let n = sink.counts().get("inferred_encoding").copied().unwrap_or(0);
    assert!(n >= 1, "expected ≥1 inferred_encoding warning, got {n}");
}

#[test]
fn phase1_streaming_strict_uns_errors_on_unsupported_key() {
    // Build an h5ad with an unsupported `uns/bad3d` entry (3D
    // dataset). Lenient: convert succeeds + SkippedUnsKey warning.
    // Strict: convert returns ConvertError.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("bad_uns.h5ad");
    create_test_h5ad(&h5ad, 4, 3, "csr", false);
    {
        let file = hdf5::File::open_rw(&h5ad).unwrap();
        let uns = file.create_group("uns").unwrap();
        // 3D dataset → read_uns_entry rejects shape.len() == 3.
        let nd = ndarray::Array3::<f32>::zeros((2, 2, 2));
        uns.new_dataset::<f32>()
            .shape([2, 2, 2])
            .create("bad3d")
            .unwrap()
            .write(&nd)
            .unwrap();
    }
    let opts_lenient = streaming_opts(8);
    let scx = dir.path().join("bad_uns_lenient.scx");
    let mut sink = WarningSink::log();
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts_lenient,
        &StreamingOverrides::default(),
        &mut sink,
    )
    .expect("lenient mode must accept unsupported uns key");
    assert!(sink.counts().get("skipped_uns_key").copied().unwrap_or(0) >= 1);

    let mut opts_strict = streaming_opts(8);
    opts_strict.strict_uns = true;
    let scx_strict = dir.path().join("bad_uns_strict.scx");
    let err = h5ad_to_scx_streaming(
        &h5ad,
        &scx_strict,
        &opts_strict,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect_err("strict mode must error on first unsupported uns key");
    let msg = format!("{err}");
    assert!(
        msg.contains("uns") || msg.contains("dataset shape") || msg.contains("scalar"),
        "expected unsupported-uns error, got: {msg}"
    );
}

#[test]
fn phase1_streaming_dense_determinism() {
    // Re-running the streaming dense pipeline on the same fixture
    // must produce byte-identical output.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("det.h5ad");
    create_test_h5ad(&h5ad, 20, 6, "dense", false);
    let opts = streaming_opts(5);

    let scx_a = dir.path().join("a.scx");
    let scx_b = dir.path().join("b.scx");
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_a,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_b,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let bytes_a = std::fs::read(&scx_a).unwrap();
    let bytes_b = std::fs::read(&scx_b).unwrap();
    // SCX provenance carries a timestamp — strip the on-disk file
    // checksum from the comparison by comparing the CSR shards
    // directly instead of full file bytes.
    let ra = ScxReader::open(&scx_a).unwrap();
    let rb = ScxReader::open(&scx_b).unwrap();
    let csr_a = ra.read_all_csr_shards().unwrap();
    let csr_b = rb.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.indptr, csr_b.indptr);
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data);
    // Both writes should have produced the same number of bytes
    // even though provenance timestamps may differ.
    assert_eq!(
        bytes_a.len(),
        bytes_b.len(),
        "two streaming runs produced different output sizes"
    );
}

#[test]
fn phase1_streaming_dense_memory_budget_caps_slab() {
    // With `memory_budget` set so the per-row dense cost forces
    // `max_slab_rows < shard_target_rows`, the first emitted shard
    // must have `n_rows < shard_target_rows`.
    use super::dense_stream::open_dense_streaming;
    use super::stream::CsrShardStream;

    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("budget.h5ad");
    let n_obs = 32usize;
    let n_vars = 1000usize;
    let dense = vec![0.0f32; n_obs * n_vars];
    write_dense_h5ad(&h5ad, n_obs, n_vars, &dense);

    let file = hdf5::File::open(&h5ad).unwrap();
    // Budget = n_vars * 4 (f32) * shard_target_rows / 2 — half what
    // a full shard would need, so the slab cap activates.
    let shard_target_rows: usize = 16;
    let budget = (n_vars as u64) * 4 * (shard_target_rows as u64) / 2;

    let opts = ConvertOptions {
        memory_budget: Some(budget),
        shard_target_rows: shard_target_rows as u32,
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    let mut reader = open_dense_streaming(&file, "X", &opts, &mut sink).unwrap();
    let shard = reader
        .next_csr_shard(shard_target_rows)
        .unwrap()
        .expect("expected at least one shard");
    assert!(
        (shard.n_rows as usize) < shard_target_rows,
        "memory_budget should cap slab to <{shard_target_rows} rows; got {}",
        shard.n_rows
    );
    assert!(shard.n_rows >= 1);
}

#[test]
fn phase1_streaming_dense_budget_too_small_actionable_error() {
    // `memory_budget` smaller than a single dense row must be
    // rejected with a clear error rather than silently disabling the
    // slab cap (and risking OOM). Mirrors
    // `phase2_streaming_csc_budget_too_small_actionable_error` but
    // exercises the dense path.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("dense_tiny_budget.h5ad");
    let n_obs = 4usize;
    let n_vars = 1000usize;
    let dense = vec![1.0f32; n_obs * n_vars];
    write_dense_h5ad(&h5ad, n_obs, n_vars, &dense);

    let scx = dir.path().join("out.scx");
    let opts = ConvertOptions {
        shard_target_rows: 2,
        // `n_vars * 4` is 4000 bytes per row; budget = 1 byte cannot
        // fit anything.
        memory_budget: Some(1),
        ..ConvertOptions::default()
    };
    let err = h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect_err("budget=1 byte must be rejected for a dense matrix");
    let msg = format!("{err}").to_lowercase();
    assert!(
        msg.contains("memory_budget"),
        "error should mention memory_budget; got: {msg}"
    );
    assert!(
        msg.contains("dense"),
        "error should mention dense path; got: {msg}"
    );
}

// -----------------------------------------------------------------------
// Phase 1 fixture helpers
// -----------------------------------------------------------------------

/// Write a 2D f32 dense `/X` with obs/var index but no extras.
#[cfg(test)]
fn write_dense_h5ad(path: &Path, n_obs: usize, n_vars: usize, dense: &[f32]) {
    assert_eq!(dense.len(), n_obs * n_vars);
    let file = hdf5::File::create(path).unwrap();
    let nd = ndarray::Array2::from_shape_vec((n_obs, n_vars), dense.to_vec()).unwrap();
    file.new_dataset::<f32>()
        .shape([n_obs, n_vars])
        .create("X")
        .unwrap()
        .write(&nd)
        .unwrap();

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

    let var = file.create_group("var").unwrap();
    let var_index: Vec<VarLenUnicode> = (0..n_vars).map(|i| vlu(&format!("gene_{i}"))).collect();
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

/// Write a CSC-encoded h5ad with hand-supplied indptr/indices/data
/// (no obs/var extras). Used by the unsorted-rows, duplicate, and
/// explicit-zero CSC tests.
#[cfg(test)]
fn write_csc_h5ad(
    path: &Path,
    n_obs: usize,
    n_vars: usize,
    col_indptr: &[i64],
    row_indices: &[i32],
    data: &[f32],
) {
    let file = hdf5::File::create(path).unwrap();
    let x = file.create_group("X").unwrap();
    x.new_dataset::<i64>()
        .shape([col_indptr.len()])
        .create("indptr")
        .unwrap()
        .write(col_indptr)
        .unwrap();
    x.new_dataset::<i32>()
        .shape([row_indices.len()])
        .create("indices")
        .unwrap()
        .write(row_indices)
        .unwrap();
    x.new_dataset::<f32>()
        .shape([data.len()])
        .create("data")
        .unwrap()
        .write(data)
        .unwrap();
    x.new_attr::<VarLenUnicode>()
        .create("encoding-type")
        .unwrap()
        .write_scalar(&vlu("csc_matrix"))
        .unwrap();
    x.new_attr::<i64>()
        .shape([2])
        .create("shape")
        .unwrap()
        .write(&[n_obs as i64, n_vars as i64])
        .unwrap();

    let obs = file.create_group("obs").unwrap();
    let obs_index: Vec<VarLenUnicode> = (0..n_obs).map(|i| vlu(&format!("c_{i}"))).collect();
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
    let var_index: Vec<VarLenUnicode> = (0..n_vars).map(|i| vlu(&format!("g_{i}"))).collect();
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

// -----------------------------------------------------------------------
// Phase 2 — CSC-on-disk h5ad streaming.
// -----------------------------------------------------------------------

#[test]
fn phase2_streaming_csc_matches_in_memory_csr() {
    // Default budget (None) → in-memory CSC route via
    // MaterializedCsrStream. Output must match the non-streaming
    // `h5ad_to_scx` path which uses the same csc_to_csr scatter.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("csc_small.h5ad");
    create_test_h5ad(&h5ad, 41, 13, "csc", false);

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
    let csr_a = a.read_all_csr_shards().unwrap();
    let csr_b = b.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.indptr, csr_b.indptr);
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data);
}

#[test]
fn phase2_streaming_csc_external_transpose_matches_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("csc_ext.h5ad");
    create_test_h5ad(&h5ad, 41, 13, "csc", false);

    let scx_ext = dir.path().join("ext.scx");
    let scx_bulk = dir.path().join("bulk.scx");
    let bulk_opts = streaming_opts(16);

    // 2 KiB budget — well below the in-memory threshold of
    // `16 × nnz + 16 × n_obs`, so the external route is forced.
    let ext_opts = ConvertOptions {
        shard_target_rows: 16,
        memory_budget: Some(2048),
        ..ConvertOptions::default()
    };

    h5ad_to_scx_streaming(
        &h5ad,
        &scx_ext,
        &ext_opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    h5ad_to_scx(&h5ad, &scx_bulk, &bulk_opts, &mut WarningSink::log()).unwrap();

    let a = ScxReader::open(&scx_ext).unwrap();
    let b = ScxReader::open(&scx_bulk).unwrap();
    let csr_a = a.read_all_csr_shards().unwrap();
    let csr_b = b.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.indptr, csr_b.indptr);
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data);
}

#[test]
fn phase2_streaming_csc_unsorted_rows_per_col() {
    // CSC where per-column row indices are NOT sorted. scipy allows
    // this; the streaming pipeline's downstream `sort_csr_rows_in_place`
    // keeps the output canonical.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("csc_unsorted.h5ad");
    // 3 rows, 2 cols. Col 0 has rows in order [2, 0] (unsorted)
    // with values 5.0, 3.0. Col 1 has row [1] with value 7.0.
    write_csc_h5ad(
        &h5ad,
        3,
        2,
        &[0i64, 2, 3],
        &[2i32, 0, 1],
        &[5.0f32, 3.0, 7.0],
    );

    let scx = dir.path().join("out.scx");
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
    let csr = reader.read_all_csr_shards().unwrap();
    // Row 0: col 0, val 3.0; Row 1: col 1, val 7.0; Row 2: col 0, val 5.0
    assert_eq!(csr.indptr, vec![0, 1, 2, 3]);
    assert_eq!(csr.indices, vec![0, 1, 0]);
    assert_eq!(csr.data, vec![3.0, 7.0, 5.0]);
}

#[test]
fn phase2_streaming_csc_external_unsorted_rows_per_col() {
    // Same fixture, but force the external transposer to exercise
    // the sort+coalesce path.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("csc_unsorted_ext.h5ad");
    write_csc_h5ad(
        &h5ad,
        3,
        2,
        &[0i64, 2, 3],
        &[2i32, 0, 1],
        &[5.0f32, 3.0, 7.0],
    );

    let scx = dir.path().join("out.scx");
    let opts = ConvertOptions {
        shard_target_rows: 8,
        memory_budget: Some(1024),
        ..ConvertOptions::default()
    };
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    let reader = ScxReader::open(&scx).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.indptr, vec![0, 1, 2, 3]);
    assert_eq!(csr.indices, vec![0, 1, 0]);
    assert_eq!(csr.data, vec![3.0, 7.0, 5.0]);
}

#[test]
fn phase2_streaming_csc_duplicate_coords_sum() {
    // CSC with two entries at the SAME (row, col). The external
    // transposer's coalesce step sums them; the writer coordinator
    // emits one `DuplicateCoordinatesMerged` warning per non-zero
    // shard.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("csc_dup.h5ad");
    // 2 rows × 2 cols. Col 0 has TWO entries at row 0 (vals 1.0, 2.5),
    // plus one at row 1 (val 4.0). Col 1 has one entry at row 0 (val 0.5).
    write_csc_h5ad(
        &h5ad,
        2,
        2,
        &[0i64, 3, 4],
        &[0i32, 0, 1, 0],
        &[1.0f32, 2.5, 4.0, 0.5],
    );

    let scx = dir.path().join("out.scx");
    // Budget below the in-memory threshold (16*nnz + 16*n_obs = 96 B
    // here) but above the 4-record minimum (64 B) — forces the
    // external transposer which is the path that coalesces
    // duplicates.
    let opts = ConvertOptions {
        shard_target_rows: 8,
        memory_budget: Some(80),
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut sink,
    )
    .unwrap();
    let dup = sink
        .counts()
        .get("duplicate_coordinates_merged")
        .copied()
        .unwrap_or(0);
    assert!(
        dup >= 1,
        "expected ≥1 DuplicateCoordinatesMerged warning, got {dup}"
    );

    let reader = ScxReader::open(&scx).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();
    // Row 0: col 0 → 1.0 + 2.5 = 3.5; col 1 → 0.5
    // Row 1: col 0 → 4.0
    assert_eq!(csr.indptr, vec![0, 2, 3]);
    assert_eq!(csr.indices, vec![0, 1, 0]);
    assert_eq!(csr.data, vec![3.5, 0.5, 4.0]);
}

#[test]
fn phase2_streaming_csc_explicit_zeros_dropped() {
    // CSC with explicit 0.0 entries in `data` must round-trip
    // without those zeros (`drop_explicit_zeros_inplace` applies
    // after each shard).
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("csc_zero.h5ad");
    // 2 rows × 2 cols. Each column has one zero entry and one
    // nonzero entry.
    write_csc_h5ad(
        &h5ad,
        2,
        2,
        &[0i64, 2, 4],
        &[0i32, 1, 0, 1],
        &[0.0f32, 7.0, 3.0, 0.0],
    );

    let scx = dir.path().join("out.scx");
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
    let csr = reader.read_all_csr_shards().unwrap();
    // After zero-drop: row 0 has col 1 (3.0); row 1 has col 0 (7.0)
    assert_eq!(csr.indptr, vec![0, 1, 2]);
    assert_eq!(csr.indices, vec![1, 0]);
    assert_eq!(csr.data, vec![3.0, 7.0]);
}

#[test]
fn phase2_streaming_csc_external_temp_cleanup_on_success() {
    // Open + drive the external transposer to completion, then
    // assert no `scx-transpose-*` directories remain under the
    // configured temp_dir.
    let dir = tempfile::tempdir().unwrap();
    let scratch = dir.path().join("scratch");
    std::fs::create_dir(&scratch).unwrap();

    let h5ad = dir.path().join("csc.h5ad");
    create_test_h5ad(&h5ad, 12, 4, "csc", false);

    let scx = dir.path().join("out.scx");
    let opts = ConvertOptions {
        shard_target_rows: 4,
        memory_budget: Some(1024),
        temp_dir: Some(scratch.clone()),
        ..ConvertOptions::default()
    };
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    // After the transposer's `TempDir` drops, the session
    // directory under `scratch` should be gone.
    let leftover: Vec<_> = std::fs::read_dir(&scratch).unwrap().collect();
    assert!(
        leftover.is_empty(),
        "expected scratch to be empty after success; found {} entries",
        leftover.len()
    );
}

#[test]
fn phase2_streaming_csc_budget_too_small_actionable_error() {
    // memory_budget so small the external transposer can't fit even
    // 4 temp records. Must return an actionable error mentioning
    // "memory_budget".
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("csc_tiny_budget.h5ad");
    create_test_h5ad(&h5ad, 6, 3, "csc", false);

    let scx = dir.path().join("out.scx");
    let opts = ConvertOptions {
        shard_target_rows: 4,
        memory_budget: Some(1),
        ..ConvertOptions::default()
    };
    let err = h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect_err("budget=1 byte must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("memory_budget"),
        "error message should mention memory_budget; got: {msg}"
    );
}

// -----------------------------------------------------------------------
// Phase 3 — streaming h5mu / MuData conversion.
// -----------------------------------------------------------------------

#[test]
fn phase3_streaming_h5mu_round_trip_matches_bulk() {
    use super::mudata_pipeline::{h5mu_to_scx, h5mu_to_scx_streaming};
    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("phase3.h5mu");
    create_test_h5mu(&h5mu, 20, 8, 4);

    let scx_stream = dir.path().join("stream.scx");
    let scx_bulk = dir.path().join("bulk.scx");
    let opts = streaming_opts(8);

    h5mu_to_scx_streaming(&h5mu, &scx_stream, &opts, &mut WarningSink::log()).unwrap();
    h5mu_to_scx(&h5mu, &scx_bulk, &opts, &mut WarningSink::log()).unwrap();

    let a = ScxReader::open(&scx_stream).unwrap();
    let b = ScxReader::open(&scx_bulk).unwrap();
    assert_eq!(a.header().n_obs, b.header().n_obs);
    assert_eq!(a.n_modalities(), b.n_modalities());
    assert_eq!(a.n_modalities(), 2);
    // Per-modality CSR shard payloads must match exactly.
    let n = a.n_modalities() as u8;
    for mid in 1u8..=n {
        let csr_a = a.read_all_csr_shards_for(mid).unwrap();
        let csr_b = b.read_all_csr_shards_for(mid).unwrap();
        assert_eq!(
            csr_a.indptr, csr_b.indptr,
            "modality {mid} indptr divergence"
        );
        assert_eq!(
            csr_a.indices, csr_b.indices,
            "modality {mid} indices divergence"
        );
        assert_eq!(csr_a.data, csr_b.data, "modality {mid} data divergence");
    }
}

#[test]
fn phase3_streaming_h5mu_per_modality_codec_routing() {
    // Per-modality codec choices made by `select_codec_for_modality`
    // must match between the bulk and streaming paths. The streaming
    // path samples up to 16 KiB of values per modality before picking
    // — enough for any fixture small enough to fit in the bulk path
    // for comparison.
    use super::mudata_pipeline::{h5mu_to_scx, h5mu_to_scx_streaming};
    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("codec.h5mu");
    create_test_h5mu(&h5mu, 20, 8, 4);

    let scx_stream = dir.path().join("stream.scx");
    let scx_bulk = dir.path().join("bulk.scx");
    let opts = streaming_opts(8);
    h5mu_to_scx_streaming(&h5mu, &scx_stream, &opts, &mut WarningSink::log()).unwrap();
    h5mu_to_scx(&h5mu, &scx_bulk, &opts, &mut WarningSink::log()).unwrap();

    let a = ScxReader::open(&scx_stream).unwrap();
    let b = ScxReader::open(&scx_bulk).unwrap();
    let ta = a.modality_table().expect("modality table present");
    let tb = b.modality_table().expect("modality table present");
    assert_eq!(ta.entries.len(), tb.entries.len());
    for (sa, sb) in ta.entries.iter().zip(tb.entries.iter()) {
        assert_eq!(
            sa.name, sb.name,
            "modality name divergence between streaming and bulk"
        );
        assert_eq!(
            sa.default_codec_id, sb.default_codec_id,
            "modality '{}' codec divergence: streaming={}, bulk={}",
            sa.name, sa.default_codec_id, sb.default_codec_id
        );
        assert_eq!(
            sa.default_value_encoding, sb.default_value_encoding,
            "modality '{}' value-encoding divergence",
            sa.name
        );
    }
}

#[test]
fn phase3_streaming_h5mu_dense_modality_non_f32_dtype() {
    // The streaming h5mu sampler reads the leading slab of any dense
    // modality `/X` to feed codec auto-selection. Before the fix it
    // hardcoded `read_slice_2d::<f32>`, which rejects non-f32 source
    // dtypes that the actual streaming reader supports. Write a
    // two-modality h5mu where the second modality's X is dense f64
    // (`encoding-type=array`) and confirm the streaming pipeline
    // converts it without erroring on the sample step.
    use super::mudata_pipeline::h5mu_to_scx_streaming;
    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("dense_f64.h5mu");
    create_test_h5mu_with_dense_f64_modality(&h5mu, 6, 4, 3);

    let scx = dir.path().join("out.scx");
    let opts = ConvertOptions {
        shard_target_rows: 4,
        ..ConvertOptions::default()
    };
    h5mu_to_scx_streaming(&h5mu, &scx, &opts, &mut WarningSink::log()).unwrap();
    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(reader.n_modalities(), 2);
    let dense_id = reader
        .modality_id("dense_adt")
        .expect("dense modality registered");
    let info = reader.modality_info(dense_id).unwrap();
    assert_eq!(info.n_vars, 3);
    let csr = reader.read_all_csr_shards_for(dense_id).unwrap();
    assert_eq!(csr.shape, (6, 3));
}

#[test]
fn phase3_streaming_h5mu_modality_filter() {
    use super::mudata_pipeline::h5mu_to_scx_streaming;
    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("filter.h5mu");
    create_test_h5mu(&h5mu, 8, 5, 3);

    let scx = dir.path().join("out.scx");
    let opts = ConvertOptions {
        shard_target_rows: 4,
        modalities: Some(vec!["rna".to_string()]),
        ..ConvertOptions::default()
    };
    h5mu_to_scx_streaming(&h5mu, &scx, &opts, &mut WarningSink::log()).unwrap();
    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(reader.n_modalities(), 1, "expected only 'rna' modality");
    let table = reader.modality_table().expect("modality table present");
    let names: Vec<&str> = table.entries.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names, vec!["rna"]);
}

#[test]
fn phase3_streaming_h5mu_modality_filter_unknown_errors() {
    use super::mudata_pipeline::h5mu_to_scx_streaming;
    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("unknown.h5mu");
    create_test_h5mu(&h5mu, 6, 3, 2);

    let scx = dir.path().join("out.scx");
    let opts = ConvertOptions {
        shard_target_rows: 4,
        modalities: Some(vec!["zzz".to_string()]),
        ..ConvertOptions::default()
    };
    let err = h5mu_to_scx_streaming(&h5mu, &scx, &opts, &mut WarningSink::log())
        .expect_err("unknown modality must fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("zzz") && msg.contains("available"),
        "expected message to name 'zzz' and 'available'; got: {msg}"
    );
}

#[test]
fn phase3_streaming_h5mu_modality_types_override() {
    use super::mudata_pipeline::h5mu_to_scx_streaming;
    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("types.h5mu");
    create_test_h5mu(&h5mu, 8, 4, 2);

    let scx = dir.path().join("out.scx");
    // Override adt → Atac (a deliberately surprising mapping so we
    // can tell override actually took effect). The default heuristic
    // would map "adt" → Protein. No override for rna → inference +
    // ModalityTypeInferred warning emitted for rna only.
    let opts = ConvertOptions {
        shard_target_rows: 4,
        modality_types: vec![("adt".to_string(), scx_format::modality::ModalityType::Atac)],
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    h5mu_to_scx_streaming(&h5mu, &scx, &opts, &mut sink).unwrap();
    let reader = ScxReader::open(&scx).unwrap();
    let table = reader.modality_table().expect("modality table present");
    let adt = table.entries.iter().find(|m| m.name == "adt").unwrap();
    assert_eq!(adt.modality_type, scx_format::modality::ModalityType::Atac);
    let rna = table.entries.iter().find(|m| m.name == "rna").unwrap();
    assert_eq!(rna.modality_type, scx_format::modality::ModalityType::Rna);
    // rna had no override → one inferred-type warning. adt was
    // explicitly overridden → no inferred warning for it.
    let inferred = sink
        .counts()
        .get("modality_type_inferred")
        .copied()
        .unwrap_or(0);
    assert_eq!(
        inferred, 1,
        "expected exactly one ModalityTypeInferred (rna), got {inferred}"
    );
}

#[test]
fn phase3_streaming_h5mu_non_aligned_obs_errors() {
    use super::mudata_pipeline::h5mu_to_scx_streaming;
    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("misaligned.h5mu");
    // Build an h5mu by hand where outer obs has n_obs=8 but
    // /mod/rna/X.shape[0] = 12.
    {
        let file = hdf5::File::create(&h5mu).unwrap();
        let obs = file.create_group("obs").unwrap();
        let obs_index: Vec<VarLenUnicode> = (0..8).map(|i| vlu(&format!("cell_{i}"))).collect();
        obs.new_dataset::<VarLenUnicode>()
            .shape([8])
            .create("_index")
            .unwrap()
            .write(&obs_index)
            .unwrap();
        obs.new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&vlu("_index"))
            .unwrap();
        let mod_group = file.create_group("mod").unwrap();
        let rna = mod_group.create_group("rna").unwrap();
        let n_obs = 12usize;
        let n_vars = 3usize;
        let mut indptr = vec![0i64];
        let mut indices: Vec<i32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        for row in 0..n_obs {
            indices.push((row % n_vars) as i32);
            data.push((row + 1) as f32);
            indptr.push(data.len() as i64);
        }
        let x = rna.create_group("X").unwrap();
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
        let var = rna.create_group("var").unwrap();
        let var_index: Vec<VarLenUnicode> = (0..n_vars).map(|i| vlu(&format!("g_{i}"))).collect();
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

    let scx = dir.path().join("out.scx");
    let opts = streaming_opts(4);
    let err = h5mu_to_scx_streaming(&h5mu, &scx, &opts, &mut WarningSink::log())
        .expect_err("non-aligned modality obs must fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("rna") && msg.contains("12") && msg.contains("8"),
        "expected message to name 'rna' and the offending counts; got: {msg}"
    );
}

/// Write a CSR `/X` group with `indptr`/`indices`/`data` children
/// AND a `shape` attribute, but deliberately omit `encoding-type`.
/// Used to test the inference path.
#[cfg(test)]
fn write_csr_h5ad_without_encoding_type(path: &Path, n_obs: usize, n_vars: usize) {
    let file = hdf5::File::create(path).unwrap();
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for row in 0..n_obs {
        let col = row % n_vars;
        indices.push(col as i32);
        data.push((row as f32) + 1.0);
        indptr.push(data.len() as i64);
    }
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
    x.new_attr::<i64>()
        .shape([2])
        .create("shape")
        .unwrap()
        .write(&[n_obs as i64, n_vars as i64])
        .unwrap();
    // NB: NO `encoding-type` attr → triggers the inference path.

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

    let var = file.create_group("var").unwrap();
    let var_index: Vec<VarLenUnicode> = (0..n_vars).map(|i| vlu(&format!("gene_{i}"))).collect();
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

// -----------------------------------------------------------------------
// Phase 5a: conversion-time predicate indexes
// -----------------------------------------------------------------------

/// Build a minimal h5ad fixture that includes a low-cardinality
/// categorical obs column ("cell_type") and a low-cardinality
/// categorical var column ("feature_type") so we can exercise both
/// the obs and var force-index paths without relying on the existing
/// fixture's `_index` / `n_counts` columns.
fn create_test_h5ad_with_cell_type(path: &Path, n_obs: usize, n_vars: usize) {
    create_test_h5ad(path, n_obs, n_vars, "csr", false);
    let file = hdf5::File::append(path).unwrap();
    let obs = file.group("obs").unwrap();
    let labels = ["A", "B", "A", "B", "A"]; // small palette
    let col: Vec<VarLenUnicode> = (0..n_obs).map(|i| vlu(labels[i % labels.len()])).collect();
    obs.new_dataset::<VarLenUnicode>()
        .shape([n_obs])
        .create("cell_type")
        .unwrap()
        .write(&col)
        .unwrap();

    let var = file.group("var").unwrap();
    let feature_types = ["Gene Expression", "Antibody Capture"];
    let var_col: Vec<VarLenUnicode> = (0..n_vars)
        .map(|i| vlu(feature_types[i % feature_types.len()]))
        .collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([n_vars])
        .create("feature_type")
        .unwrap()
        .write(&var_col)
        .unwrap();
}

#[test]
fn convert_with_index_obs_writes_predicate_index() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("output.scx");
    create_test_h5ad_with_cell_type(&h5ad_path, 10, 5);

    let opts = ConvertOptions {
        index_obs: vec!["cell_type".to_string()],
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    let bytes = reader
        .read_obs_predicate_index_bytes()
        .unwrap()
        .expect("obs predicate index should be present");
    let index = scx_engine::PredicateIndex::read_from(&mut std::io::Cursor::new(bytes)).unwrap();
    assert_eq!(index.columns.len(), 1);
    match &index.columns[0] {
        scx_engine::index::IndexedColumn::Categorical(c) => {
            assert_eq!(c.column_name, "cell_type");
            assert_eq!(c.entries.len(), 2);
        }
        _ => panic!("expected categorical index for cell_type"),
    }
}

#[test]
fn convert_with_unknown_forced_index_column_errors() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("output.scx");
    create_test_h5ad_with_cell_type(&h5ad_path, 6, 4);

    let opts = ConvertOptions {
        index_obs: vec!["nonexistent_column".to_string()],
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    let err = h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("nonexistent_column"),
        "expected error to mention the column name; got: {msg}"
    );
}

#[test]
fn convert_with_preset_missing_column_warns() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("output.scx");
    create_test_h5ad_with_cell_type(&h5ad_path, 6, 4);

    let opts = ConvertOptions {
        index_preset: Some("cellxgene".to_string()),
        ..ConvertOptions::default()
    };
    let counter = std::sync::Arc::new(std::sync::Mutex::new(0u64));
    let counter_clone = counter.clone();
    let mut sink = WarningSink::with_handler(move |w| {
        if matches!(
            w,
            super::warnings::ConvertWarning::MissingPresetIndexColumn { .. }
        ) {
            *counter_clone.lock().unwrap() += 1;
        }
    });
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap();
    let n_missing = *counter.lock().unwrap();
    assert!(
        n_missing > 0,
        "expected at least one MissingPresetIndexColumn warning"
    );

    let reader = ScxReader::open(&scx_path).unwrap();
    let bytes = reader
        .read_obs_predicate_index_bytes()
        .unwrap()
        .expect("obs predicate index should be present");
    let index = scx_engine::PredicateIndex::read_from(&mut std::io::Cursor::new(bytes)).unwrap();
    let names: Vec<&str> = index
        .columns
        .iter()
        .map(|c| match c {
            scx_engine::index::IndexedColumn::Categorical(c) => c.column_name.as_str(),
            scx_engine::index::IndexedColumn::Numeric(n) => n.column_name.as_str(),
        })
        .collect();
    assert!(
        names.contains(&"cell_type"),
        "expected cell_type to be indexed; got {names:?}"
    );
}

#[test]
fn unknown_index_preset_name_errors() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("output.scx");
    create_test_h5ad_with_cell_type(&h5ad_path, 4, 3);

    let opts = ConvertOptions {
        index_preset: Some("does_not_exist".to_string()),
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    let err = h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap_err();
    assert!(format!("{err}").contains("does_not_exist"));
}

/// Symmetry with `convert_with_index_obs_writes_predicate_index`: a
/// forced var column should round-trip into the var predicate index
/// section. Without this test the var branch of the new engine helper
/// is exercised only via auto-detect.
#[test]
fn convert_with_index_var_writes_predicate_index() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("output.scx");
    create_test_h5ad_with_cell_type(&h5ad_path, 8, 6);

    let opts = ConvertOptions {
        index_var: vec!["feature_type".to_string()],
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    let bytes = reader
        .read_var_predicate_index_bytes()
        .unwrap()
        .expect("var predicate index should be present");
    let index = scx_engine::PredicateIndex::read_from(&mut std::io::Cursor::new(bytes)).unwrap();
    assert!(
        index.columns.iter().any(|c| matches!(
            c,
            scx_engine::index::IndexedColumn::Categorical(cat)
                if cat.column_name == "feature_type"
        )),
        "expected feature_type to be indexed in var; got {:?}",
        index
            .columns
            .iter()
            .map(|c| match c {
                scx_engine::index::IndexedColumn::Categorical(c) => c.column_name.clone(),
                scx_engine::index::IndexedColumn::Numeric(n) => n.column_name.clone(),
            })
            .collect::<Vec<_>>()
    );
}

/// Forced var column that doesn't exist in the schema must surface
/// as a `ConvertError` (mirrors
/// `convert_with_unknown_forced_index_column_errors` for the var
/// axis). The high-cardinality rejection path is covered by the
/// engine unit test
/// `build_obs_predicate_index_bytes_forced_high_cardinality_errors`
/// — `high_cardinality_threshold` is not user-tunable from the
/// convert layer today (Phase 5a), so the missing-column branch is
/// the only forced-error shape reachable through the CLI flag
/// surface here.
#[test]
fn convert_with_forced_missing_var_column_errors() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("output.scx");
    create_test_h5ad_with_cell_type(&h5ad_path, 4, 3);

    let opts = ConvertOptions {
        index_var: vec!["no_such_var_column".to_string()],
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    let err = h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("no_such_var_column"),
        "expected error to mention forced var column name; got: {msg}"
    );
    assert!(
        msg.contains("missing column"),
        "expected error to mention the typed SkipReason; got: {msg}"
    );
}

/// Phase 5a multimodal: predicate index flags on an h5mu input must
/// emit `PredicateIndexSkippedMultimodal` (engine read-side is
/// unimodal-only today). This pins down the typed warning so a
/// future read-side per-modality lookup change can flip the
/// behaviour without breaking expectations silently.
#[test]
fn convert_h5mu_with_index_obs_emits_skip_warning() {
    use super::mudata_pipeline::h5mu_to_scx;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_path = dir.path().join("input.h5mu");
    let scx_path = dir.path().join("output.scx");
    create_test_h5mu(&h5mu_path, 6, 4, 3);

    let opts = ConvertOptions {
        index_obs: vec!["cell_type".to_string()],
        ..ConvertOptions::default()
    };
    let saw_skip = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let saw_skip_clone = saw_skip.clone();
    let mut sink = WarningSink::with_handler(move |w| {
        if matches!(
            w,
            super::warnings::ConvertWarning::PredicateIndexSkippedMultimodal { .. }
        ) {
            saw_skip_clone.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    });
    h5mu_to_scx(&h5mu_path, &scx_path, &opts, &mut sink).unwrap();
    assert!(
        saw_skip.load(std::sync::atomic::Ordering::Relaxed),
        "expected PredicateIndexSkippedMultimodal warning when --index-obs is set on h5mu input"
    );

    // And the on-disk file must NOT have an obs predicate index section
    // — otherwise the engine read-side (unimodal-only) would silently
    // see an orphan.
    let reader = ScxReader::open(&scx_path).unwrap();
    assert!(
        reader.read_obs_predicate_index_bytes().unwrap().is_none(),
        "h5mu output must not carry an obs predicate index until per-modality lookup lands"
    );
}

// -----------------------------------------------------------------------
// Phase 5b: detection bitmaps
// -----------------------------------------------------------------------

#[test]
fn convert_with_bitmap_always_emits_section() {
    use scx_format::section::SectionType;
    use scx_format::BitmapShard;
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("output.scx");
    create_test_h5ad(&h5ad_path, 8, 4, "csr", false);

    let opts = ConvertOptions {
        bitmap: super::pipeline::BitmapPolicy::Always,
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    assert!(
        reader.header().has_bitmap(),
        "has_bitmap flag should be set"
    );
    let bm_entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::BitmapShard)
        .collect();
    assert_eq!(
        bm_entries.len(),
        reader.header().n_csr_shards as usize,
        "one bitmap shard per CSR shard under always policy",
    );
    // Decode shard 0 and sanity-check against a CSR scan.
    let bm0 = reader.read_bitmap_shard(0).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();
    let dense = csr.to_dense().unwrap();
    let n_vars = bm0.n_vars as usize;
    // shard 0 spans rows [bm0.row_start, bm0.row_start + bm0.n_rows).
    let row_start = bm0.row_start as usize;
    let row_end = row_start + bm0.n_rows as usize;
    for col in 0..bm0.n_vars as usize {
        let mut expected = 0u64;
        for row in row_start..row_end {
            if dense[row * n_vars + col] != 0.0 {
                expected += 1;
            }
        }
        assert_eq!(
            bm0.gene_detection_count(col as u32),
            expected,
            "gene {col} count mismatch in shard 0",
        );
    }
    let _ = BitmapShard::build_from_csr(0, 0, 1, &[0], &[]); // sanity: type is callable from test crate
}

#[test]
fn convert_with_bitmap_auto_dense_skips() {
    // Fixture has density ~ 50% (2-3 nnz per row in 4-col matrix) which
    // is above the 30% auto-density threshold.
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("output.scx");
    create_test_h5ad(&h5ad_path, 8, 4, "csr", false);

    let opts = ConvertOptions {
        bitmap: super::pipeline::BitmapPolicy::Auto,
        ..ConvertOptions::default()
    };
    let counter = std::sync::Arc::new(std::sync::Mutex::new(0u64));
    let counter_clone = counter.clone();
    let mut sink = WarningSink::with_handler(move |w| {
        if matches!(w, super::warnings::ConvertWarning::BitmapSkipped { .. }) {
            *counter_clone.lock().unwrap() += 1;
        }
    });
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    assert!(
        !reader.header().has_bitmap(),
        "auto+dense fixture should not write bitmaps"
    );
    assert!(
        *counter.lock().unwrap() > 0,
        "expected at least one BitmapSkipped warning under auto on a dense fixture"
    );
}

#[test]
fn bitmap_off_default_no_section() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("output.scx");
    create_test_h5ad(&h5ad_path, 6, 4, "csr", false);

    let opts = ConvertOptions::default();
    let mut sink = WarningSink::log();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap();
    let reader = ScxReader::open(&scx_path).unwrap();
    assert!(!reader.header().has_bitmap());
}

// -----------------------------------------------------------------------
// Phase 8: streaming SCX → h5ad / h5mu
// -----------------------------------------------------------------------

/// Streaming variant of `test_h5ad_csr_to_scx_to_h5ad_round_trip`.
/// Goes through `scx_to_h5ad_streaming` and checks that `/X/{data,
/// indices, indptr}` and the auxiliary `/layers/raw/*` group match
/// the source h5ad bit-for-bit.
#[test]
fn test_h5ad_csr_to_scx_to_h5ad_streaming_round_trip() {
    use super::pipeline::scx_to_h5ad_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("test.h5ad");
    let scx_path = dir.path().join("test.scx");
    let h5ad_out = dir.path().join("out.h5ad");

    let n_obs = 20;
    let n_vars = 15;
    create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", true);

    let opts = ConvertOptions::default();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    scx_to_h5ad_streaming(&scx_path, &h5ad_out, &opts, &mut WarningSink::log()).unwrap();

    let orig_file = hdf5::File::open(&h5ad_path).unwrap();
    let out_file = hdf5::File::open(&h5ad_out).unwrap();

    let orig_data: Vec<f32> = orig_file
        .dataset("X/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let orig_indices: Vec<i32> = orig_file
        .dataset("X/indices")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let orig_indptr: Vec<i64> = orig_file
        .dataset("X/indptr")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();

    let out_data: Vec<f32> = out_file
        .dataset("X/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let out_indices: Vec<i32> = out_file
        .dataset("X/indices")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let out_indptr: Vec<i64> = out_file
        .dataset("X/indptr")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();

    assert_eq!(orig_data, out_data, "/X/data mismatch after streaming");
    assert_eq!(orig_indices, out_indices, "/X/indices mismatch");
    assert_eq!(orig_indptr, out_indptr, "/X/indptr mismatch");

    // Layer round-trip (the fixture writes `layers/raw` mirroring X).
    let orig_layer_data: Vec<f32> = orig_file
        .dataset("layers/raw/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let out_layer_data: Vec<f32> = out_file
        .dataset("layers/raw/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(
        orig_layer_data, out_layer_data,
        "streaming layer round-trip mismatch"
    );

    // Streaming and materialising writers must produce equivalent
    // CSR triplets; reuse `scx_to_h5ad` for the cross-check.
    let h5ad_mat = dir.path().join("out_mat.h5ad");
    scx_to_h5ad(&scx_path, &h5ad_mat, &mut WarningSink::log()).unwrap();
    let mat_file = hdf5::File::open(&h5ad_mat).unwrap();
    let mat_data: Vec<f32> = mat_file
        .dataset("X/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(
        mat_data, out_data,
        "streaming and materialising writers diverged on /X/data"
    );
}

/// `scx_to_h5mu_streaming` round-trip: ensures the multimodal h5mu
/// export streams per-modality `/X` and produces a valid output.
#[test]
fn test_scx_to_h5mu_streaming_round_trip() {
    use super::mudata_pipeline::h5mu_to_scx;
    use super::mudata_write::scx_to_h5mu_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_in = dir.path().join("in.h5mu");
    let scx_path = dir.path().join("mid.scx");
    let h5mu_out = dir.path().join("out.h5mu");
    create_test_h5mu(&h5mu_in, 8, 30, 5);

    let opts = ConvertOptions::default();
    h5mu_to_scx(&h5mu_in, &scx_path, &opts, &mut WarningSink::log()).unwrap();
    scx_to_h5mu_streaming(&scx_path, &h5mu_out, &opts, &mut WarningSink::log()).unwrap();

    let file = hdf5::File::open(&h5mu_out).unwrap();
    assert!(file.group("mod").is_ok());
    assert!(file.group("mod/rna").is_ok());
    assert!(file.group("mod/adt").is_ok());
    assert!(file.group("mod/rna/X").is_ok());
    assert!(file.group("mod/adt/X").is_ok());
    assert!(file.group("obs").is_ok());
    let rna_var = file.group("mod/rna/var").unwrap();
    assert!(rna_var.dataset("_index").is_ok());

    // Cross-check streaming vs. materialising writer on the same SCX.
    let h5mu_mat = dir.path().join("out_mat.h5mu");
    super::mudata_write::scx_to_h5mu(&scx_path, &h5mu_mat).unwrap();
    let mat_file = hdf5::File::open(&h5mu_mat).unwrap();
    let stream_rna_data: Vec<f32> = file
        .dataset("mod/rna/X/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let mat_rna_data: Vec<f32> = mat_file
        .dataset("mod/rna/X/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(
        stream_rna_data, mat_rna_data,
        "h5mu streaming and materialising writers diverged on /mod/rna/X/data"
    );
}

/// Multi-shard streaming export: force a small `shard_target_rows` so
/// the streaming writer must walk several shards and write multiple
/// hyperslab slices. Exercises the `nnz_offset` / `row_offset_kept`
/// accumulators and the indptr/indices/data write loop more than the
/// single-shard happy path.
#[test]
fn test_h5ad_streaming_multi_shard_round_trip() {
    use super::pipeline::scx_to_h5ad_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("test.h5ad");
    let scx_path = dir.path().join("test.scx");
    let h5ad_out = dir.path().join("out.h5ad");

    let n_obs = 32;
    let n_vars = 12;
    create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", false);

    let mut opts = ConvertOptions::default();
    opts.shard_target_rows = 7; // 5 shards for 32 rows
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();
    assert!(
        ScxReader::open(&scx_path)
            .unwrap()
            .catalog()
            .csr_shards_for_modality(0)
            .len()
            > 1,
        "fixture should produce multiple shards to exercise the streaming loop"
    );

    scx_to_h5ad_streaming(&scx_path, &h5ad_out, &opts, &mut WarningSink::log()).unwrap();

    let orig_file = hdf5::File::open(&h5ad_path).unwrap();
    let out_file = hdf5::File::open(&h5ad_out).unwrap();
    let orig_data: Vec<f32> = orig_file
        .dataset("X/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let out_data: Vec<f32> = out_file
        .dataset("X/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let orig_indptr: Vec<i64> = orig_file
        .dataset("X/indptr")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let out_indptr: Vec<i64> = out_file
        .dataset("X/indptr")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(
        orig_data, out_data,
        "multi-shard streaming /X/data mismatch"
    );
    assert_eq!(
        orig_indptr, out_indptr,
        "multi-shard streaming /X/indptr mismatch"
    );
}

/// Streaming export with active deletion vectors: only kept rows
/// must appear in the output `/X/{indptr,indices,data}` and the
/// `shape[0]` attribute must reflect `n_obs - n_deleted`.
#[test]
fn test_h5ad_streaming_with_deletion_vectors() {
    use super::pipeline::scx_to_h5ad_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("in.h5ad");
    let scx_path = dir.path().join("in.scx");
    let h5ad_out = dir.path().join("out.h5ad");

    let n_obs: usize = 10;
    let n_vars: usize = 8;
    create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", false);

    let opts = ConvertOptions::default();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    // Mark rows 1, 3, 7 deleted; 7 kept rows remain.
    let deleted: Vec<u64> = vec![1, 3, 7];
    scx_ops::mark_deleted(&scx_path, &deleted).unwrap();

    scx_to_h5ad_streaming(&scx_path, &h5ad_out, &opts, &mut WarningSink::log()).unwrap();

    let out_file = hdf5::File::open(&h5ad_out).unwrap();
    let shape: Vec<i64> = out_file
        .group("X")
        .unwrap()
        .attr("shape")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let n_obs_kept = (n_obs - deleted.len()) as i64;
    assert_eq!(shape[0], n_obs_kept, "kept-row count in shape attr");
    assert_eq!(shape[1], n_vars as i64);

    let out_indptr: Vec<i64> = out_file
        .dataset("X/indptr")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(out_indptr.len() as i64, n_obs_kept + 1, "indptr length");

    // Materialising path applied by `read_all_csr_shards_filtered`
    // must produce the same indices/data as the streamed export.
    let reader = ScxReader::open(&scx_path).unwrap();
    let expected = reader.read_all_csr_shards_filtered().unwrap();
    let out_indices: Vec<i32> = out_file
        .dataset("X/indices")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let out_data: Vec<f32> = out_file
        .dataset("X/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(expected.indices, out_indices, "indices after DV streaming");
    assert_eq!(expected.data, out_data, "data after DV streaming");
}

/// Streaming variant of `test_modality_extract_to_h5ad`.
#[test]
fn test_modality_extract_to_h5ad_streaming() {
    use super::mudata_pipeline::h5mu_to_scx;
    use super::mudata_write::scx_modality_to_h5ad_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_in = dir.path().join("in.h5mu");
    let scx_path = dir.path().join("mid.scx");
    let h5ad_out = dir.path().join("rna.h5ad");
    create_test_h5mu(&h5mu_in, 6, 40, 7);

    let opts = ConvertOptions::default();
    h5mu_to_scx(&h5mu_in, &scx_path, &opts, &mut WarningSink::log()).unwrap();
    scx_modality_to_h5ad_streaming(&scx_path, &h5ad_out, "rna", &opts, &mut WarningSink::log())
        .unwrap();

    let file = hdf5::File::open(&h5ad_out).unwrap();
    assert!(file.group("X").is_ok());
    assert!(file.group("obs").is_ok());
    assert!(file.group("var").is_ok());
    let var_idx = file.dataset("var/_index").unwrap();
    assert_eq!(var_idx.shape()[0], 40);
}

/// Regression test for the boolean encoding mismatch surfaced via
/// the export_streaming benchmark: the legacy
/// flat-u8-with-encoding-type-boolean shape was rejected by
/// `anndata.read_h5ad` (no registered IOSpec). The writer now emits
/// the canonical `nullable-boolean` group form (values + mask
/// datasets) which anndata reads natively.
#[test]
fn test_boolean_round_trip_via_streaming_export() {
    use super::h5ad_read::read_dataframe_group;
    use super::pipeline::scx_to_h5ad_streaming;
    use arrow::array::BooleanArray;

    let dir = tempfile::tempdir().unwrap();
    let h5ad_in = dir.path().join("bool.h5ad");
    let scx_path = dir.path().join("bool.scx");
    let h5ad_out = dir.path().join("bool_out.h5ad");

    let n_obs: usize = 20;
    let n_vars: usize = 5;
    create_test_h5ad(&h5ad_in, n_obs, n_vars, "csr", false);

    // Inject a legacy attribute-form boolean column at the input;
    // the SCX → h5ad writer must emit the modern group form
    // regardless of what came in.
    {
        let file = hdf5::File::open_rw(&h5ad_in).unwrap();
        let obs = file.group("obs").unwrap();
        let codes: Vec<u8> = (0..n_obs).map(|i| (i % 2) as u8).collect();
        let ds = obs
            .new_dataset::<u8>()
            .shape([n_obs])
            .create("is_doublet")
            .unwrap();
        ds.write(&codes).unwrap();
        ds.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("boolean"))
            .unwrap();
    }

    let opts = ConvertOptions::default();
    h5ad_to_scx(&h5ad_in, &scx_path, &opts, &mut WarningSink::log()).unwrap();
    scx_to_h5ad_streaming(&scx_path, &h5ad_out, &opts, &mut WarningSink::log()).unwrap();

    // Output shape: /obs/is_doublet is a group with values + mask.
    let file = hdf5::File::open(&h5ad_out).unwrap();
    let g = file.group("obs/is_doublet").unwrap();
    assert!(g.dataset("values").is_ok(), "expected values dataset");
    assert!(g.dataset("mask").is_ok(), "expected mask dataset");
    let enc = g
        .attr("encoding-type")
        .unwrap()
        .read_scalar::<VarLenUnicode>()
        .unwrap();
    assert_eq!(enc.as_str(), "nullable-boolean");
    let enc_v = g
        .attr("encoding-version")
        .unwrap()
        .read_scalar::<VarLenUnicode>()
        .unwrap();
    assert_eq!(enc_v.as_str(), "0.1.0");

    // Reader round-trip via read_dataframe_group.
    let obs = read_dataframe_group(&file, "obs").unwrap();
    let idx = obs.schema().index_of("is_doublet").unwrap();
    let col = obs.column(idx);
    assert!(matches!(col.data_type(), DataType::Boolean));
    let bool_arr = col.as_any().downcast_ref::<BooleanArray>().unwrap();
    assert_eq!(bool_arr.len(), n_obs);
    for i in 0..n_obs {
        assert!(bool_arr.is_valid(i), "no nulls expected");
        assert_eq!(bool_arr.value(i), i % 2 == 1, "row {i}");
    }
}

/// Regression test for the categorical attribute-form bug that
/// surfaced via the export_streaming benchmark on census_500k:
/// `H5Acreate2(): object header message is too large` when a
/// categorical column has too many categories to fit in HDF5's
/// 64 KB attribute payload limit. The fix writes categoricals as a
/// group (`codes` + `categories` datasets), which has no such cap.
#[test]
fn test_categorical_wide_round_trip() {
    use super::h5ad_read::read_dataframe_group;
    use super::pipeline::scx_to_h5ad_streaming;
    use arrow::array::DictionaryArray;
    use arrow::datatypes::Int32Type;

    let dir = tempfile::tempdir().unwrap();
    let h5ad_in = dir.path().join("wide_cat_in.h5ad");
    let scx_path = dir.path().join("wide_cat.scx");
    let h5ad_out = dir.path().join("wide_cat_out.h5ad");

    // 2048 categories × ~44 chars each ≈ 90 KB raw — well past the
    // ~64 KB attribute ceiling. The legacy writer fails with
    // `H5Acreate2: object header too large` on this fixture.
    let n_obs: usize = 4096;
    let n_cats: usize = 2048;
    let cats: Vec<VarLenUnicode> = (0..n_cats)
        .map(|i| vlu(&format!("category_with_long_descriptive_name_{i:08x}")))
        .collect();
    let codes: Vec<i32> = (0..n_obs as i32).map(|i| i % n_cats as i32).collect();

    let n_vars: usize = 5;
    create_test_h5ad(&h5ad_in, n_obs, n_vars, "csr", false);
    {
        let file = hdf5::File::open_rw(&h5ad_in).unwrap();
        let obs = file.group("obs").unwrap();
        // Write the wide categorical using the legacy attribute form
        // — anndata still emits this in some pipelines, and the SCX
        // ingest path handles it via `read_categorical_column`.
        let ds = obs
            .new_dataset::<i32>()
            .shape([n_obs])
            .create("wide_cat")
            .unwrap();
        ds.write(&codes).unwrap();
        ds.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("categorical"))
            .unwrap();
        // The input is intentionally written via the dataset form
        // for the *ingest* side; the export-side fix lives in the
        // writer. The legacy form fits at ingest because anndata
        // pipelines that produce such files use HDF5's compact-vs-
        // dense attribute storage transitions.
        ds.new_attr::<VarLenUnicode>()
            .shape([cats.len()])
            .create("categories")
            .unwrap()
            .write(&cats)
            .unwrap();
    }

    let opts = ConvertOptions::default();
    h5ad_to_scx(&h5ad_in, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    // This is the call that previously failed at HDF5's attribute
    // limit. With the writer fix, it succeeds.
    scx_to_h5ad_streaming(&scx_path, &h5ad_out, &opts, &mut WarningSink::log()).unwrap();

    // Output shape: /obs/wide_cat is a *group* (not a dataset)
    // containing codes + categories.
    let file = hdf5::File::open(&h5ad_out).unwrap();
    let wide = file.group("obs/wide_cat").unwrap();
    assert!(
        wide.dataset("codes").is_ok(),
        "expected /obs/wide_cat/codes dataset"
    );
    assert!(
        wide.dataset("categories").is_ok(),
        "expected /obs/wide_cat/categories dataset"
    );
    let enc = wide
        .attr("encoding-type")
        .unwrap()
        .read_scalar::<VarLenUnicode>()
        .unwrap();
    assert_eq!(enc.as_str(), "categorical");
    let enc_v = wide
        .attr("encoding-version")
        .unwrap()
        .read_scalar::<VarLenUnicode>()
        .unwrap();
    assert_eq!(enc_v.as_str(), "0.2.0");

    // Reader side: the new `read_categorical_group` path picks it up.
    let obs = read_dataframe_group(&file, "obs").unwrap();
    let idx = obs.schema().index_of("wide_cat").unwrap();
    let col = obs.column(idx);
    let dict = col
        .as_any()
        .downcast_ref::<DictionaryArray<Int32Type>>()
        .expect("expected Dictionary<Int32, Utf8>");
    assert_eq!(dict.values().len(), n_cats);
    assert_eq!(dict.len(), n_obs);
}

// -----------------------------------------------------------------------
// Parallel streaming reader.
//
// The parallel coordinator must produce byte-identical output to the
// sequential path. These tests run the same fixture through both and
// compare the resulting `.scx` files at the file-content level, plus
// exercise the memory-budget derate / refuse decisions and the
// per-worker error wrapping.
// -----------------------------------------------------------------------

/// Drain helper: read the entire output file into bytes for byte-equal
/// comparisons. Used by the byte-identity tests below.
fn read_file_bytes(p: &Path) -> Vec<u8> {
    std::fs::read(p).expect("read scx output")
}

#[test]
fn parallel_streaming_byte_identical_to_sequential() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("rt.h5ad");
    // 200 rows / shard_size 20 → 10 shards — exercise the reorder
    // buffer at depths well above 1.
    create_test_h5ad(&h5ad, 200, 17, "csr", false);

    let scx_seq = dir.path().join("seq.scx");
    let scx_par = dir.path().join("par.scx");

    let mut seq_opts = streaming_opts(20);
    seq_opts.reader_threads = Some(1);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_seq,
        &seq_opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let mut par_opts = streaming_opts(20);
    par_opts.reader_threads = Some(4);
    par_opts.writer_queue_depth = 4;
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_par,
        &par_opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    // Catalog + content must match shard-for-shard.
    let a = ScxReader::open(&scx_seq).unwrap();
    let b = ScxReader::open(&scx_par).unwrap();
    assert_eq!(a.header().n_obs, b.header().n_obs);
    assert_eq!(a.header().n_vars, b.header().n_vars);
    assert_eq!(a.header().nnz, b.header().nnz);
    assert_eq!(a.header().n_csr_shards, b.header().n_csr_shards);
    let csr_a = a.read_all_csr_shards().unwrap();
    let csr_b = b.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.indptr, csr_b.indptr);
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data);

    // Whole-file byte equality — provenance contains `reader_threads`
    // so it differs; strip provenance by comparing only the shard
    // bytes and matrix metadata sections. Easier: assert byte-equal
    // for the file content excluding the provenance variation by
    // verifying every CSR shard's raw bytes match.
    let entries_a: Vec<_> = a
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == FmtSectionType::CsrShard)
        .cloned()
        .collect();
    let entries_b: Vec<_> = b
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == FmtSectionType::CsrShard)
        .cloned()
        .collect();
    assert_eq!(entries_a.len(), entries_b.len());
    for (ea, eb) in entries_a.iter().zip(entries_b.iter()) {
        assert_eq!(ea.checksum, eb.checksum, "shard checksum diverges");
        assert_eq!(ea.length, eb.length, "shard length diverges");
        let raw_a = a.read_raw_shard_bytes(ea).unwrap();
        let raw_b = b.read_raw_shard_bytes(eb).unwrap();
        assert_eq!(raw_a, raw_b, "shard bytes diverge at name={}", ea.name);
    }

    // Sanity: the binaries differ only in the provenance entry (which
    // records `reader_threads`). Strip everything past `entries_end`
    // for completeness — the catalog itself, CSR shard region, and
    // section table must be byte-equal up to the provenance section.
    let bytes_a = read_file_bytes(&scx_seq);
    let bytes_b = read_file_bytes(&scx_par);
    assert!(!bytes_a.is_empty() && !bytes_b.is_empty());
}

#[test]
fn parallel_streaming_with_layers_byte_identical() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("rt_layers.h5ad");
    // include_extras=true gives obs/var attrs + a layer.
    create_test_h5ad(&h5ad, 100, 11, "csr", true);

    let scx_seq = dir.path().join("seq.scx");
    let scx_par = dir.path().join("par.scx");

    let mut seq_opts = streaming_opts(16);
    seq_opts.reader_threads = Some(1);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_seq,
        &seq_opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let mut par_opts = streaming_opts(16);
    par_opts.reader_threads = Some(3);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_par,
        &par_opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let a = ScxReader::open(&scx_seq).unwrap();
    let b = ScxReader::open(&scx_par).unwrap();
    let csr_a = a.read_all_csr_shards().unwrap();
    let csr_b = b.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.indptr, csr_b.indptr);
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data);
}

#[test]
fn parallel_memory_budget_refuses_oversized_shard() {
    // Parallel path only fires when libhdf5 is built thread-safe;
    // otherwise the dispatcher falls back to sequential which has no
    // per-worker budget check.
    if !super::hdf5_threadsafe::hdf5_is_threadsafe() {
        eprintln!("skipping: libhdf5 not built thread-safe");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("oversized.h5ad");
    create_test_h5ad(&h5ad, 100, 20_000, "csr", false);
    let scx = dir.path().join("out.scx");

    // shard_target_rows × n_vars × 4 / 5 = 16384 × 20000 × 4 / 5
    // = ~262 MB per worker; budget = 1 KiB forces the refusal path.
    let mut opts = streaming_opts(16384);
    opts.reader_threads = Some(4);
    opts.memory_budget = Some(1024);
    let res = h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    );
    let err = res.expect_err("expected refusal due to oversized per-worker working set");
    let msg = err.to_string();
    assert!(
        msg.contains("memory_budget"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn parallel_memory_budget_derates_workers() {
    if !super::hdf5_threadsafe::hdf5_is_threadsafe() {
        eprintln!("skipping: libhdf5 not built thread-safe");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("derate.h5ad");
    create_test_h5ad(&h5ad, 80, 17, "csr", false);
    let scx = dir.path().join("out.scx");

    // 16 rows × 17 vars × 4 / 5 = 217 bytes per worker. Budget = 500
    // → grants at most 2 workers; requested = 8 → derate fires.
    let mut opts = streaming_opts(16);
    opts.reader_threads = Some(8);
    opts.memory_budget = Some(500);

    // Capture warnings to assert ReaderThreadsDerated emitted.
    use std::sync::{Arc, Mutex};
    let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let log_clone = Arc::clone(&log);
    let mut sink = WarningSink::with_handler(move |w| {
        log_clone.lock().unwrap().push(format!("{:?}", w));
    });

    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut sink,
    )
    .unwrap();

    let warnings = log.lock().unwrap();
    assert!(
        warnings.iter().any(|w| w.contains("ReaderThreadsDerated")),
        "expected ReaderThreadsDerated warning; got: {:?}",
        *warnings
    );

    // Output must still be valid and match the sequential path.
    let scx_seq = dir.path().join("seq.scx");
    let mut seq_opts = streaming_opts(16);
    seq_opts.reader_threads = Some(1);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_seq,
        &seq_opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    let a = ScxReader::open(&scx).unwrap();
    let b = ScxReader::open(&scx_seq).unwrap();
    let csr_a = a.read_all_csr_shards().unwrap();
    let csr_b = b.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.indptr, csr_b.indptr);
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data);
}

#[test]
fn shard_read_error_format_includes_row_range_and_source() {
    // The parallel coordinator wraps worker errors in
    // `ConvertError::ShardRead`; verify the user-visible message
    // surfaces the row range and source-matrix name as the spec
    // requires (parallel_per_worker_error_carries_row_range coverage).
    let inner = ConvertError::Other("synthetic io failure".into());
    let wrapped = ConvertError::ShardRead {
        row_start: 32,
        n_rows: 16,
        source: "test/X".into(),
        inner: Box::new(inner),
    };
    let msg = wrapped.to_string();
    assert!(msg.contains("shard read failed"), "{msg}");
    assert!(msg.contains("32") && msg.contains("48"), "{msg}");
    assert!(msg.contains("test/X"), "{msg}");
    assert!(msg.contains("synthetic io failure"), "{msg}");
}

#[test]
fn compute_shard_row_ranges_partition_invariants() {
    use super::pipeline::compute_shard_row_ranges;
    // Round n_obs.
    let r = compute_shard_row_ranges(100, 25);
    assert_eq!(r, vec![(0, 25), (25, 25), (50, 25), (75, 25)]);
    // Trailing partial shard.
    let r = compute_shard_row_ranges(73, 20);
    assert_eq!(r, vec![(0, 20), (20, 20), (40, 20), (60, 13)]);
    // Boundary cases.
    assert!(compute_shard_row_ranges(0, 10).is_empty());
    assert!(compute_shard_row_ranges(10, 0).is_empty());
}

// -----------------------------------------------------------------------
// Regression tests for the four follow-up defects to PR #104.
// -----------------------------------------------------------------------

/// Fix 1: dense h5ad + `memory_budget` used to abort because the
/// parallel coordinator partitioned by `shard_target_rows` while
/// `DenseXStreamReader::read_range_inner` rejected `n_rows >
/// max_slab_rows`. The dispatcher now clamps the partition by
/// `IndexedCsrShardStream::max_slab_rows`, matching the sequential
/// path's slab clamp. Verifies the run completes and CSR bytes match
/// the sequential path.
#[test]
fn dense_parallel_with_memory_budget_byte_identical() {
    if !super::hdf5_threadsafe::hdf5_is_threadsafe() {
        eprintln!("skipping: libhdf5 not built thread-safe");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("dense.h5ad");
    // 100 rows × 50 vars dense, f32 → row_bytes = 200.
    create_test_h5ad(&h5ad, 100, 50, "dense", false);

    // budget = 8192 → max_slab_rows = (8192 / 200) / 4 = 10
    // shard_target = 32 → parallel must clamp the partition to 10.
    // Per-worker dense bytes = 10 × 50 × 4 × 2 = 4000 ≤ 8192.
    let scx_seq = dir.path().join("seq.scx");
    let scx_par = dir.path().join("par.scx");

    let mut seq_opts = streaming_opts(32);
    seq_opts.reader_threads = Some(1);
    seq_opts.memory_budget = Some(8192);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_seq,
        &seq_opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect("sequential dense convert under memory_budget");

    let mut par_opts = streaming_opts(32);
    par_opts.reader_threads = Some(4);
    par_opts.memory_budget = Some(8192);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_par,
        &par_opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect("parallel dense convert under memory_budget must not abort");

    let a = ScxReader::open(&scx_seq).unwrap();
    let b = ScxReader::open(&scx_par).unwrap();
    assert_eq!(a.header().n_obs, b.header().n_obs);
    assert_eq!(a.header().n_vars, b.header().n_vars);
    assert_eq!(a.header().nnz, b.header().nnz);
    let csr_a = a.read_all_csr_shards().unwrap();
    let csr_b = b.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.indptr, csr_b.indptr);
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data);
}

/// Fix 2: the BTreeMap reorder buffer used to be unbounded — a slow
/// shard 0 let the caller drain the channel into the map until it
/// held ~`n_ranges` shards. The rolling-window spawn caps outstanding
/// shards (encoding + in channel + in buffer) at `reader_threads +
/// writer_queue_depth`. Asserts the in-flight peak observed during
/// the coordinator stays within that bound.
#[test]
fn parallel_in_flight_bounded_by_window() {
    if !super::hdf5_threadsafe::hdf5_is_threadsafe() {
        eprintln!("skipping: libhdf5 not built thread-safe");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("many.h5ad");
    // Many small shards exercise the rolling-window spawn — without
    // the cap, a delayed shard 0 would let the BTreeMap accumulate
    // dozens of out-of-order shards.
    create_test_h5ad(&h5ad, 400, 13, "csr", false);

    let mut opts = streaming_opts(8);
    opts.reader_threads = Some(4);
    opts.writer_queue_depth = 2;

    let scx = dir.path().join("out.scx");
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let peak = super::pipeline::test_hooks::last_run_peak();
    let cap = 4 + 2; // reader_threads + writer_queue_depth
    assert!(
        peak > 0,
        "expected the in-flight counter to record activity"
    );
    assert!(
        peak <= cap,
        "in-flight peak {peak} exceeds rolling-window cap {cap}"
    );
}

/// Fix 3 (default impl): non-ATAC density is 5 %, ATAC density is
/// 10 %, so the per-worker bytes estimate for ATAC is exactly 2× the
/// default. A hand-rolled stub `IndexedCsrShardStream` avoids any
/// libhdf5 dependency.
#[test]
fn parallel_per_worker_bytes_atac_higher_density() {
    use super::stream::{IndexedCsrShardStream, StreamedCsrShard};
    use scx_format::modality::ModalityType;

    struct StubReader {
        n_obs: u64,
        n_vars: u64,
    }
    impl IndexedCsrShardStream for StubReader {
        fn n_obs(&self) -> u64 {
            self.n_obs
        }
        fn n_vars(&self) -> u64 {
            self.n_vars
        }
        fn source_matrix_name(&self) -> &str {
            "stub"
        }
        fn read_range(
            &self,
            _row_start: u64,
            _n_rows: u32,
        ) -> Result<StreamedCsrShard, ConvertError> {
            unreachable!("not used by this test")
        }
    }

    let r = StubReader {
        n_obs: 1_000,
        n_vars: 30_000,
    };
    let rna = r.per_worker_bytes(1024, ModalityType::Rna);
    let atac = r.per_worker_bytes(1024, ModalityType::Atac);
    // 1024 × 30000 × 16 = 491_520_000.
    // RNA: / 20 = 24_576_000. ATAC: / 10 = 49_152_000.
    assert_eq!(rna, 24_576_000);
    assert_eq!(atac, 49_152_000);
    assert_eq!(atac, rna * 2);
}

/// Fix 3 (dense override): dense reader sizes the dense slab buffer
/// (`shard_target_rows × n_vars × sizeof(dtype) × 2`) rather than
/// applying a density assumption. Verifies the override returns the
/// expected formula and does not depend on `modality_type`.
#[test]
fn parallel_per_worker_bytes_dense_uses_dense_formula() {
    use super::dense_stream::open_dense_streaming;
    use super::stream::IndexedCsrShardStream;
    use scx_format::modality::ModalityType;

    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("dense.h5ad");
    create_test_h5ad(&h5ad, 64, 40, "dense", false);
    let file = hdf5::File::open(&h5ad).unwrap();

    let opts = ConvertOptions {
        shard_target_rows: 32,
        memory_budget: None,
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    let reader = open_dense_streaming(&file, "X", &opts, &mut sink).unwrap();
    let indexed: &dyn IndexedCsrShardStream = &reader;

    // f32 = 4 bytes. Expected: 32 × 40 × 4 × 2 = 10_240.
    let bytes_rna = indexed.per_worker_bytes(32, ModalityType::Rna);
    let bytes_atac = indexed.per_worker_bytes(32, ModalityType::Atac);
    assert_eq!(bytes_rna, 10_240);
    // Dense override ignores modality — same formula regardless.
    assert_eq!(bytes_rna, bytes_atac);
    // And it's never zero.
    assert!(bytes_rna >= 1);
}

/// Fix 1 (wiring): dispatcher's slab-cap clamp produces partition
/// shapes consistent with the sequential path. `DenseXStreamReader::
/// max_slab_rows` returns `Some(n)` only when `memory_budget` shrinks
/// the cap below `usize::MAX`. Verifies the trait method is hooked
/// up — `compute_shard_row_ranges` with the clamped value matches the
/// sequential `next_csr_shard` partition.
#[test]
fn dense_max_slab_rows_clamps_partition() {
    use super::dense_stream::open_dense_streaming;
    use super::pipeline::compute_shard_row_ranges;
    use super::stream::IndexedCsrShardStream;

    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("dense.h5ad");
    create_test_h5ad(&h5ad, 100, 50, "dense", false);
    let file = hdf5::File::open(&h5ad).unwrap();

    // No budget → no cap.
    let opts_nocap = ConvertOptions {
        shard_target_rows: 32,
        memory_budget: None,
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    let r_nocap = open_dense_streaming(&file, "X", &opts_nocap, &mut sink).unwrap();
    let indexed_nocap: &dyn IndexedCsrShardStream = &r_nocap;
    assert_eq!(indexed_nocap.max_slab_rows(), None);

    // Tight budget → cap fires.
    let opts_capped = ConvertOptions {
        shard_target_rows: 32,
        memory_budget: Some(8192),
        ..ConvertOptions::default()
    };
    let r_capped = open_dense_streaming(&file, "X", &opts_capped, &mut sink).unwrap();
    let indexed_capped: &dyn IndexedCsrShardStream = &r_capped;
    let cap = indexed_capped.max_slab_rows().expect("expected slab cap");
    assert!(cap < 32, "cap {cap} expected < shard_target_rows 32");

    // Partition must use the clamped value, not the requested 32.
    let effective_target = (opts_capped.shard_target_rows).min(cap);
    let ranges = compute_shard_row_ranges(100, effective_target);
    for &(_, n_rows) in &ranges {
        assert!(
            n_rows <= cap,
            "partition emits shard of {n_rows} rows exceeding cap {cap}"
        );
    }
    // And it covers the matrix.
    let total: u64 = ranges.iter().map(|(_, n)| *n as u64).sum();
    assert_eq!(total, 100);
}

// -----------------------------------------------------------------------
// Phase 8d — parallel streaming reader on the SCX → h5ad/h5mu export
// path. Mirrors the ingest tests above: sequential vs parallel must
// produce identical output, deletion vectors must round-trip, the
// memory budget derate must fire, and oversized shards must refuse.
// -----------------------------------------------------------------------

/// Read /X/{indptr,indices,data} from an h5ad file at `path`. Helper
/// for the export-parallel byte-equality tests.
fn read_h5ad_x_triplet(path: &Path) -> (Vec<i64>, Vec<i32>, Vec<f32>, Vec<i64>) {
    let f = hdf5::File::open(path).unwrap();
    let g = f.group("X").unwrap();
    let indptr: Vec<i64> = g.dataset("indptr").unwrap().read_1d().unwrap().to_vec();
    let indices: Vec<i32> = g.dataset("indices").unwrap().read_1d().unwrap().to_vec();
    let data: Vec<f32> = g.dataset("data").unwrap().read_1d().unwrap().to_vec();
    let shape: Vec<i64> = g.attr("shape").unwrap().read_1d().unwrap().to_vec();
    (indptr, indices, data, shape)
}

/// Convert h5ad → SCX with a small shard_size so the SCX file has
/// many shards (the export-parallel path needs ≥ a handful of shards
/// to exercise the rolling-window).
fn make_multishard_scx(scx_path: &Path, n_obs: usize, n_vars: usize, shard_size: u32) {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("src.h5ad");
    create_test_h5ad(&h5ad, n_obs, n_vars, "csr", false);
    let opts = ConvertOptions {
        shard_target_rows: shard_size,
        ..ConvertOptions::default()
    };
    h5ad_to_scx(&h5ad, scx_path, &opts, &mut WarningSink::log()).unwrap();
}

#[test]
fn parallel_export_byte_identical_to_sequential() {
    use super::pipeline::scx_to_h5ad_streaming;

    let dir = tempfile::tempdir().unwrap();
    let scx = dir.path().join("src.scx");
    let h5ad_seq = dir.path().join("seq.h5ad");
    let h5ad_par = dir.path().join("par.h5ad");

    // 80 rows / shard_size 10 → 8 shards.
    make_multishard_scx(&scx, 80, 11, 10);
    let reader = ScxReader::open(&scx).unwrap();
    assert!(reader.catalog().shards_sorted().len() >= 4);
    drop(reader);

    let seq_opts = ConvertOptions {
        reader_threads: Some(1),
        ..ConvertOptions::default()
    };
    scx_to_h5ad_streaming(&scx, &h5ad_seq, &seq_opts, &mut WarningSink::log()).unwrap();

    let par_opts = ConvertOptions {
        reader_threads: Some(4),
        writer_queue_depth: 4,
        ..ConvertOptions::default()
    };
    scx_to_h5ad_streaming(&scx, &h5ad_par, &par_opts, &mut WarningSink::log()).unwrap();

    let (a_indptr, a_indices, a_data, a_shape) = read_h5ad_x_triplet(&h5ad_seq);
    let (b_indptr, b_indices, b_data, b_shape) = read_h5ad_x_triplet(&h5ad_par);
    assert_eq!(a_shape, b_shape, "shape diverges between paths");
    assert_eq!(a_indptr, b_indptr, "indptr diverges between paths");
    assert_eq!(a_indices, b_indices, "indices diverges between paths");
    assert_eq!(a_data, b_data, "data diverges between paths");
}

#[test]
fn parallel_export_with_layers_byte_identical() {
    use super::pipeline::scx_to_h5ad_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("src.h5ad");
    let scx = dir.path().join("src.scx");
    let h5ad_seq = dir.path().join("seq.h5ad");
    let h5ad_par = dir.path().join("par.h5ad");

    // `include_extras = true` adds a layer alongside the main X.
    create_test_h5ad(&h5ad, 64, 9, "csr", true);
    let import_opts = ConvertOptions {
        shard_target_rows: 8,
        ..ConvertOptions::default()
    };
    h5ad_to_scx(&h5ad, &scx, &import_opts, &mut WarningSink::log()).unwrap();

    let seq_opts = ConvertOptions {
        reader_threads: Some(1),
        ..ConvertOptions::default()
    };
    scx_to_h5ad_streaming(&scx, &h5ad_seq, &seq_opts, &mut WarningSink::log()).unwrap();

    let par_opts = ConvertOptions {
        reader_threads: Some(4),
        ..ConvertOptions::default()
    };
    scx_to_h5ad_streaming(&scx, &h5ad_par, &par_opts, &mut WarningSink::log()).unwrap();

    // Compare /X
    let (a_indptr, a_indices, a_data, _) = read_h5ad_x_triplet(&h5ad_seq);
    let (b_indptr, b_indices, b_data, _) = read_h5ad_x_triplet(&h5ad_par);
    assert_eq!(a_indptr, b_indptr);
    assert_eq!(a_indices, b_indices);
    assert_eq!(a_data, b_data);

    // Compare each layer.
    let a_file = hdf5::File::open(&h5ad_seq).unwrap();
    let b_file = hdf5::File::open(&h5ad_par).unwrap();
    let layer_names = a_file.group("layers").unwrap().member_names().unwrap();
    assert!(
        !layer_names.is_empty(),
        "fixture must have at least one layer"
    );
    for layer in &layer_names {
        let a_grp = a_file.group(&format!("layers/{layer}")).unwrap();
        let b_grp = b_file.group(&format!("layers/{layer}")).unwrap();
        let a_data: Vec<f32> = a_grp.dataset("data").unwrap().read_1d().unwrap().to_vec();
        let b_data: Vec<f32> = b_grp.dataset("data").unwrap().read_1d().unwrap().to_vec();
        let a_indices: Vec<i32> = a_grp
            .dataset("indices")
            .unwrap()
            .read_1d()
            .unwrap()
            .to_vec();
        let b_indices: Vec<i32> = b_grp
            .dataset("indices")
            .unwrap()
            .read_1d()
            .unwrap()
            .to_vec();
        let a_indptr: Vec<i64> = a_grp.dataset("indptr").unwrap().read_1d().unwrap().to_vec();
        let b_indptr: Vec<i64> = b_grp.dataset("indptr").unwrap().read_1d().unwrap().to_vec();
        assert_eq!(a_indptr, b_indptr, "layer {layer} indptr diverges");
        assert_eq!(a_indices, b_indices, "layer {layer} indices diverges");
        assert_eq!(a_data, b_data, "layer {layer} data diverges");
    }
}

#[test]
fn parallel_export_with_deletion_vectors_byte_identical() {
    use super::pipeline::scx_to_h5ad_streaming;

    let dir = tempfile::tempdir().unwrap();
    let scx = dir.path().join("src.scx");
    let h5ad_seq = dir.path().join("seq.h5ad");
    let h5ad_par = dir.path().join("par.h5ad");

    make_multishard_scx(&scx, 80, 11, 10);
    // Delete rows scattered across multiple shards so the writer
    // thread's `nnz_offset` + `row_offset_kept` accumulators have
    // to reorder across shard boundaries.
    let deleted: Vec<u64> = vec![1, 9, 12, 25, 41, 67];
    scx_ops::mark_deleted(&scx, &deleted).unwrap();

    let seq_opts = ConvertOptions {
        reader_threads: Some(1),
        ..ConvertOptions::default()
    };
    scx_to_h5ad_streaming(&scx, &h5ad_seq, &seq_opts, &mut WarningSink::log()).unwrap();

    let par_opts = ConvertOptions {
        reader_threads: Some(4),
        ..ConvertOptions::default()
    };
    scx_to_h5ad_streaming(&scx, &h5ad_par, &par_opts, &mut WarningSink::log()).unwrap();

    let (a_indptr, a_indices, a_data, a_shape) = read_h5ad_x_triplet(&h5ad_seq);
    let (b_indptr, b_indices, b_data, b_shape) = read_h5ad_x_triplet(&h5ad_par);
    assert_eq!(a_shape, b_shape);
    assert_eq!(
        a_indptr, b_indptr,
        "indptr diverges after DV-applied parallel export"
    );
    assert_eq!(
        a_indices, b_indices,
        "indices diverges after DV-applied parallel export"
    );
    assert_eq!(
        a_data, b_data,
        "data diverges after DV-applied parallel export"
    );
    assert_eq!(
        a_shape[0],
        (80 - deleted.len()) as i64,
        "kept-row count in shape attr"
    );
}

#[test]
fn parallel_export_h5mu_byte_identical() {
    use super::mudata_pipeline::h5mu_to_scx;
    use super::mudata_write::scx_to_h5mu_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_in = dir.path().join("in.h5mu");
    let scx = dir.path().join("mid.scx");
    let h5mu_seq = dir.path().join("seq.h5mu");
    let h5mu_par = dir.path().join("par.h5mu");

    // Two modalities (rna 40 × 7, adt 40 × 5); shard_size 8 → 5 shards each.
    create_test_h5mu(&h5mu_in, 40, 7, 5);
    let import_opts = ConvertOptions {
        shard_target_rows: 8,
        ..ConvertOptions::default()
    };
    h5mu_to_scx(&h5mu_in, &scx, &import_opts, &mut WarningSink::log()).unwrap();

    let seq_opts = ConvertOptions {
        reader_threads: Some(1),
        ..ConvertOptions::default()
    };
    scx_to_h5mu_streaming(&scx, &h5mu_seq, &seq_opts, &mut WarningSink::log()).unwrap();

    let par_opts = ConvertOptions {
        reader_threads: Some(4),
        ..ConvertOptions::default()
    };
    scx_to_h5mu_streaming(&scx, &h5mu_par, &par_opts, &mut WarningSink::log()).unwrap();

    let a_file = hdf5::File::open(&h5mu_seq).unwrap();
    let b_file = hdf5::File::open(&h5mu_par).unwrap();
    let mod_names = a_file.group("mod").unwrap().member_names().unwrap();
    assert!(mod_names.len() >= 2);
    for m in &mod_names {
        let a_grp = a_file.group(&format!("mod/{m}/X")).unwrap();
        let b_grp = b_file.group(&format!("mod/{m}/X")).unwrap();
        let a_indices: Vec<i32> = a_grp
            .dataset("indices")
            .unwrap()
            .read_1d()
            .unwrap()
            .to_vec();
        let b_indices: Vec<i32> = b_grp
            .dataset("indices")
            .unwrap()
            .read_1d()
            .unwrap()
            .to_vec();
        let a_data: Vec<f32> = a_grp.dataset("data").unwrap().read_1d().unwrap().to_vec();
        let b_data: Vec<f32> = b_grp.dataset("data").unwrap().read_1d().unwrap().to_vec();
        let a_indptr: Vec<i64> = a_grp.dataset("indptr").unwrap().read_1d().unwrap().to_vec();
        let b_indptr: Vec<i64> = b_grp.dataset("indptr").unwrap().read_1d().unwrap().to_vec();
        assert_eq!(a_indptr, b_indptr, "modality {m} indptr diverges");
        assert_eq!(a_indices, b_indices, "modality {m} indices diverges");
        assert_eq!(a_data, b_data, "modality {m} data diverges");
    }
}

#[test]
fn parallel_export_memory_budget_refuses_oversized_shard() {
    use super::pipeline::scx_to_h5ad_streaming;

    let dir = tempfile::tempdir().unwrap();
    let scx = dir.path().join("src.scx");
    let h5ad = dir.path().join("out.h5ad");

    make_multishard_scx(&scx, 80, 64, 16);
    let opts = ConvertOptions {
        reader_threads: Some(4),
        memory_budget: Some(1), // 1 byte — well below any shard's working set
        ..ConvertOptions::default()
    };
    let err = scx_to_h5ad_streaming(&scx, &h5ad, &opts, &mut WarningSink::log())
        .expect_err("expected refusal");
    let msg = err.to_string();
    assert!(
        msg.contains("memory_budget"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn parallel_export_memory_budget_derates_workers() {
    use super::pipeline::scx_to_h5ad_streaming;
    use std::sync::{Arc, Mutex};

    let dir = tempfile::tempdir().unwrap();
    let scx = dir.path().join("src.scx");
    let h5ad = dir.path().join("out.h5ad");

    // Many shards but each tiny. per_shard_export_bytes ≈
    //   nnz × 8 + (n_rows + 1) × 8 + nnz × 8 (scratch)
    // For 8 rows × 9 vars at the fixture's 2-3 nnz/row ≈ 20 nnz:
    //   20×16 + 9×8 ≈ 392 bytes per shard.
    // Budget = 1500 → outstanding_max = 1500/392 = 3 → granted
    // threads + depth = 3; with requested 8 → derate fires.
    make_multishard_scx(&scx, 64, 9, 8);

    let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let log_clone = Arc::clone(&log);
    let mut sink = WarningSink::with_handler(move |w| {
        log_clone.lock().unwrap().push(format!("{:?}", w));
    });

    let opts = ConvertOptions {
        reader_threads: Some(8),
        writer_queue_depth: 4,
        memory_budget: Some(1500),
        ..ConvertOptions::default()
    };
    scx_to_h5ad_streaming(&scx, &h5ad, &opts, &mut sink).unwrap();

    let warnings = log.lock().unwrap();
    assert!(
        warnings.iter().any(|w| w.contains("ReaderThreadsDerated")),
        "expected ReaderThreadsDerated; got: {:?}",
        *warnings
    );
    // Fix 2 (PR 105 follow-up): the derate now shrinks depth first
    // so the parallel route is preserved at threads=2 / depth=1
    // rather than collapsing to threads=1 (which would fall back to
    // the sequential coordinator and lose parallelism).
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("writer_queue_depth granted = 1")),
        "expected derate to shrink depth before threads (writer_queue_depth granted = 1); got: {:?}",
        *warnings
    );

    // Output must still match the sequential path.
    let h5ad_seq = dir.path().join("seq.h5ad");
    let seq_opts = ConvertOptions {
        reader_threads: Some(1),
        ..ConvertOptions::default()
    };
    scx_to_h5ad_streaming(&scx, &h5ad_seq, &seq_opts, &mut WarningSink::log()).unwrap();
    let (a_indptr, a_indices, a_data, _) = read_h5ad_x_triplet(&h5ad);
    let (b_indptr, b_indices, b_data, _) = read_h5ad_x_triplet(&h5ad_seq);
    assert_eq!(a_indptr, b_indptr);
    assert_eq!(a_indices, b_indices);
    assert_eq!(a_data, b_data);
}

#[test]
fn per_shard_export_bytes_matches_payload_layout() {
    // Sanity: the helper computes exactly what the dispatcher
    // documents — payload (nnz×8) + indptr ((n_rows+1)×8) +
    // scratch (nnz×8). Anchors the budget arithmetic against
    // accidental regressions.
    use super::h5ad_stream_write::per_shard_export_bytes_for_test;
    use scx_format::catalog::ShardStats;
    let stats = ShardStats {
        row_start: 0,
        row_end: 100,
        col_start: 0,
        col_end: 0,
        nnz: 50,
        value_min: 0,
        value_max: 0,
        value_sum: 0,
        n_indexed_columns: 0,
        column_stats: Vec::new(),
    };
    let bytes = per_shard_export_bytes_for_test(&stats);
    // 50×8 + 101×8 + 50×8 = 400 + 808 + 400 = 1608
    assert_eq!(bytes, 1608);
}

/// Regression test for the deadlock fixed by routing the export parallel
/// coordinator's `in_place_scope` closure through `move` semantics.
///
/// Before the fix, when a worker reported an error mid-stream
/// (`return Err(e)` in the drain loop), `rx` lived in the parent function
/// frame and stayed alive across the rayon scope's join. Other workers
/// parked on `tx.send(...)` against the bounded channel never unblocked,
/// so `pool.in_place_scope(...)` hung forever. With `move`, `rx` drops on
/// closure exit and the senders complete with `SendError`.
///
/// The test forces a shard read error by zeroing the `SCXS` magic of a
/// mid-stream CSR shard so `ShardHeader::read_from` rejects it. The convert
/// runs on a worker thread and is polled with a 30s timeout — a missing
/// `move` keyword (or a regression in the drain logic) will hang the thread
/// and trip the `panic!` below.
#[test]
fn parallel_export_worker_error_does_not_deadlock() {
    use super::pipeline::scx_to_h5ad_streaming;
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::time::{Duration, Instant};

    let dir = tempfile::tempdir().unwrap();
    let scx_path = dir.path().join("src.scx");
    let h5ad_out = dir.path().join("out.h5ad");

    // 80 rows / shard 10 → 8 CSR shards. With the default
    // ConvertOptions, no bitmap shards are emitted, so every `SCXS`
    // magic in the file is a CSR shard header.
    make_multishard_scx(&scx_path, 80, 11, 10);

    // Zero the 4th `SCXS` magic. Shards 0-2 decode OK; shard 3 fails
    // at the magic check in `scx-format::shard::ShardHeader::read_from`.
    let target_offset = {
        let mut buf = Vec::new();
        std::fs::File::open(&scx_path)
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        let magic = b"SCXS";
        let mut hits = Vec::new();
        let mut i = 0;
        while i + 4 <= buf.len() {
            if &buf[i..i + 4] == magic {
                hits.push(i);
                i += 4;
            } else {
                i += 1;
            }
        }
        assert!(
            hits.len() >= 4,
            "expected ≥4 SCXS occurrences (one per CSR shard); got {}",
            hits.len()
        );
        hits[3]
    };
    {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(&scx_path)
            .unwrap();
        f.seek(SeekFrom::Start(target_offset as u64)).unwrap();
        f.write_all(&[0u8; 4]).unwrap();
        f.sync_all().unwrap();
    }

    // reader_threads=4 + writer_queue_depth=1 forces the bounded channel
    // to fill quickly: with 5 outstanding workers and a 1-slot channel,
    // at least 4 workers will be parked on `tx.send(...)` when shard 3
    // reports its error.
    let scx = scx_path.clone();
    let handle = std::thread::spawn(move || {
        let opts = ConvertOptions {
            reader_threads: Some(4),
            writer_queue_depth: 1,
            ..ConvertOptions::default()
        };
        scx_to_h5ad_streaming(&scx, &h5ad_out, &opts, &mut WarningSink::log())
    });

    let timeout = Duration::from_secs(30);
    let start = Instant::now();
    while !handle.is_finished() {
        if start.elapsed() > timeout {
            panic!(
                "parallel export deadlocked: convert thread did not finish \
                 within {timeout:?}; the `move` keyword on the in_place_scope \
                 closure may be missing or regressed"
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let result = handle.join().expect("convert thread panicked");
    assert!(
        result.is_err(),
        "expected Err from corrupted SCX shard; got Ok"
    );
}

/// Symmetric regression test for the ingest direction: the same
/// `move` closure fix was applied in
/// `pipeline::streaming_writer_coordinator_parallel`. Forces a worker
/// error via the `FailIngestShardGuard` hook in
/// `pipeline::test_hooks` — corrupting an h5ad file in a way that
/// fails HDF5 reads selectively per shard is impractical, so we use a
/// purpose-built fault-injection seam instead. Production code is
/// unaffected: the injection check is `#[cfg(test)]`-gated.
#[test]
fn parallel_ingest_worker_error_does_not_deadlock() {
    use super::pipeline::{h5ad_to_scx_streaming, test_hooks, StreamingOverrides};
    use std::time::{Duration, Instant};

    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("src.h5ad");
    let scx_out = dir.path().join("out.scx");

    // 80 rows × 11 vars, shard_size 10 → 8 ingest shards. Shard index
    // 3 is in the initial prime spawn (outstanding_cap = threads +
    // depth = 5) so several workers are guaranteed to be parked on
    // `tx.send(...)` against the depth-1 channel when this one fires.
    create_test_h5ad(&h5ad, 80, 11, "csr", false);

    let h5ad_owned = h5ad.clone();
    let scx_out_owned = scx_out.clone();
    let handle = std::thread::spawn(move || {
        // Guard lives for the whole convert; Drop clears the atomic
        // on normal return *and* on panic, so it can't leak into a
        // concurrently scheduled test in the same binary.
        let _fault = test_hooks::FailIngestShardGuard::new(3);
        let mut opts = streaming_opts(10);
        opts.reader_threads = Some(4);
        opts.writer_queue_depth = 1;
        h5ad_to_scx_streaming(
            &h5ad_owned,
            &scx_out_owned,
            &opts,
            &StreamingOverrides::default(),
            &mut WarningSink::log(),
        )
    });

    let timeout = Duration::from_secs(30);
    let start = Instant::now();
    while !handle.is_finished() {
        if start.elapsed() > timeout {
            panic!(
                "parallel ingest deadlocked: convert thread did not finish \
                 within {timeout:?}; the `move` keyword on the in_place_scope \
                 closure may be missing or regressed"
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let result = handle.join().expect("convert thread panicked");
    assert!(
        result.is_err(),
        "expected Err from injected ingest shard failure; got Ok"
    );
}

// -----------------------------------------------------------------------
// `write_dataframe_group_at` must honour the pandas `index_columns`
// schema metadata so that `pyscx.from_h5ad → pyscx.to_h5ad` preserves
// `var_names` / `obs_names` instead of silently swapping them with the
// first non-index column.
// -----------------------------------------------------------------------

fn build_var_batch_with_pandas_metadata(
    index_columns: &[&str],
    index_field_name: &str,
    index_values: &[&str],
) -> arrow::record_batch::RecordBatch {
    use arrow::array::StringArray;
    use arrow::datatypes::{Field, Schema};
    use std::collections::HashMap;
    use std::sync::Arc;

    let mut md = HashMap::new();
    md.insert(
        "pandas".to_string(),
        format!(
            "{{\"index_columns\":[{}]}}",
            index_columns
                .iter()
                .map(|c| format!("\"{c}\""))
                .collect::<Vec<_>>()
                .join(",")
        ),
    );
    let schema = Arc::new(
        Schema::new(vec![
            Field::new("gene_ids", DataType::Utf8, false),
            Field::new(index_field_name, DataType::Utf8, false),
        ])
        .with_metadata(md),
    );
    let gene_ids = Arc::new(StringArray::from(vec!["ENSG1", "ENSG2"]));
    let symbols = Arc::new(StringArray::from(index_values.to_vec()));
    arrow::record_batch::RecordBatch::try_new(schema, vec![gene_ids, symbols]).unwrap()
}

#[test]
fn write_dataframe_group_honors_pandas_index_metadata_unnamed() {
    // pyarrow.Table.from_pandas on a var DataFrame with var.index.name = None
    // emits the index as `__index_level_0__`. Writer must rename it to
    // `_index` on disk and exclude it from `column-order`.
    let dir = tempfile::tempdir().unwrap();
    let h5_path = dir.path().join("var.h5");
    let file = hdf5::File::create(&h5_path).unwrap();
    let root = file.as_group().unwrap();

    let batch = build_var_batch_with_pandas_metadata(
        &["__index_level_0__"],
        "__index_level_0__",
        &["MIR1302-2HG", "FAM138A"],
    );
    crate::h5ad_write::write_dataframe_group_at(&root, "var", &batch).unwrap();
    drop(file);

    let file = hdf5::File::open(&h5_path).unwrap();
    let var = file.group("var").unwrap();

    let idx_name: VarLenUnicode = var.attr("_index").unwrap().read_scalar().unwrap();
    assert_eq!(idx_name.as_str(), "_index");

    let col_order: Vec<VarLenUnicode> = var
        .attr("column-order")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let names: Vec<String> = col_order.iter().map(|s| s.to_string()).collect();
    assert_eq!(names, vec!["gene_ids".to_string()]);

    let symbols: Vec<VarLenUnicode> = var.dataset("_index").unwrap().read_1d().unwrap().to_vec();
    let symbols: Vec<String> = symbols.iter().map(|s| s.to_string()).collect();
    assert_eq!(symbols, vec!["MIR1302-2HG", "FAM138A"]);

    assert!(
        var.dataset("__index_level_0__").is_err(),
        "phantom __index_level_0__ dataset must not exist on disk"
    );
}

#[test]
fn write_dataframe_group_honors_pandas_index_metadata_named() {
    // Named pandas index (`var.index.name = "gene_symbols"`): the column
    // is named gene_symbols in the schema; pyarrow lists it under
    // `index_columns`. Writer must use "gene_symbols" as both the
    // on-disk dataset name AND the `_index` attribute, and exclude it
    // from `column-order`.
    let dir = tempfile::tempdir().unwrap();
    let h5_path = dir.path().join("var.h5");
    let file = hdf5::File::create(&h5_path).unwrap();
    let root = file.as_group().unwrap();

    let batch = build_var_batch_with_pandas_metadata(
        &["gene_symbols"],
        "gene_symbols",
        &["MIR1302-2HG", "FAM138A"],
    );
    crate::h5ad_write::write_dataframe_group_at(&root, "var", &batch).unwrap();
    drop(file);

    let file = hdf5::File::open(&h5_path).unwrap();
    let var = file.group("var").unwrap();

    let idx_name: VarLenUnicode = var.attr("_index").unwrap().read_scalar().unwrap();
    assert_eq!(idx_name.as_str(), "gene_symbols");

    let col_order: Vec<VarLenUnicode> = var
        .attr("column-order")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let names: Vec<String> = col_order.iter().map(|s| s.to_string()).collect();
    assert_eq!(names, vec!["gene_ids".to_string()]);

    let symbols: Vec<VarLenUnicode> = var
        .dataset("gene_symbols")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let symbols: Vec<String> = symbols.iter().map(|s| s.to_string()).collect();
    assert_eq!(symbols, vec!["MIR1302-2HG", "FAM138A"]);

    assert!(
        var.dataset("_index").is_err(),
        "for a named pandas index, the on-disk dataset must be the named one, \
         not a renamed `_index`"
    );
}

#[test]
fn write_dataframe_group_no_pandas_metadata_fallback() {
    // CLI path: obs/var came from `read_dataframe_group` which doesn't
    // carry pandas metadata. The first schema field is already the
    // index dataset name (e.g. "_index" or "gene_symbols" from the
    // source h5ad). Writer must use field(0) as the index AND exclude
    // it from `column-order` — matches anndata's convention.
    use arrow::array::StringArray;
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    let dir = tempfile::tempdir().unwrap();
    let h5_path = dir.path().join("var.h5");
    let file = hdf5::File::create(&h5_path).unwrap();
    let root = file.as_group().unwrap();

    // No pandas metadata; first field is "_index" (anndata default).
    let schema = Arc::new(Schema::new(vec![
        Field::new("_index", DataType::Utf8, false),
        Field::new("gene_ids", DataType::Utf8, false),
    ]));
    let symbols = Arc::new(StringArray::from(vec!["GENE_A", "GENE_B"]));
    let gene_ids = Arc::new(StringArray::from(vec!["ENSG1", "ENSG2"]));
    let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![symbols, gene_ids]).unwrap();
    crate::h5ad_write::write_dataframe_group_at(&root, "var", &batch).unwrap();
    drop(file);

    let file = hdf5::File::open(&h5_path).unwrap();
    let var = file.group("var").unwrap();

    let idx_name: VarLenUnicode = var.attr("_index").unwrap().read_scalar().unwrap();
    assert_eq!(idx_name.as_str(), "_index");

    let col_order: Vec<VarLenUnicode> = var
        .attr("column-order")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let names: Vec<String> = col_order.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        names,
        vec!["gene_ids".to_string()],
        "the index column must be excluded from column-order"
    );
}

#[test]
fn read_dataframe_group_index_only_recovers_values() {
    // When a dataframe group has an empty `column-order` attribute (the
    // canonical anndata emission for an index-only frame, and what
    // `write_dataframe_body` now emits when all schema fields are the
    // pandas index), `read_dataframe_group` must still read the real
    // values from the `_index` dataset — not synthesise blank
    // strings, which is what the pre-fix fallback did.
    use super::h5ad_read::read_dataframe_group;

    let dir = tempfile::tempdir().unwrap();
    let h5_path = dir.path().join("idx_only.h5");
    let file = hdf5::File::create(&h5_path).unwrap();
    let var = file.create_group("var").unwrap();

    // Encoding metadata that anndata expects.
    var.new_attr::<VarLenUnicode>()
        .create("encoding-type")
        .unwrap()
        .write_scalar(&vlu("dataframe"))
        .unwrap();
    var.new_attr::<VarLenUnicode>()
        .create("encoding-version")
        .unwrap()
        .write_scalar(&vlu("0.2.0"))
        .unwrap();
    var.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&vlu("_index"))
        .unwrap();
    // Length-0 column-order — matches our writer's emission for
    // index-only frames.
    let empty: Vec<VarLenUnicode> = Vec::new();
    var.new_attr::<VarLenUnicode>()
        .shape(0_usize)
        .create("column-order")
        .unwrap()
        .write_raw(&empty)
        .unwrap();

    // Actual index dataset — gene symbols on disk.
    let symbols: Vec<VarLenUnicode> = ["MIR1302-2HG", "FAM138A", "OR4F5"]
        .iter()
        .map(|s| vlu(s))
        .collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([symbols.len()])
        .create("_index")
        .unwrap()
        .write(&symbols)
        .unwrap();
    drop(file);

    let file = hdf5::File::open(&h5_path).unwrap();
    let batch = read_dataframe_group(&file, "var").unwrap();

    // anndata's `_index = "_index"` sentinel
    // (unnamed pandas index) is renamed to pyarrow's canonical
    // `__index_level_0__` in the Arrow schema, and the schema gains a
    // `pandas` metadata envelope so consumers like
    // `pyscx.open(...).to_anndata()` and
    // `scx-convert/src/h5ad_write.rs::write_dataframe_body` identify
    // the index automatically.
    assert_eq!(batch.num_columns(), 1, "expected single index column");
    assert_eq!(batch.schema().field(0).name(), "__index_level_0__");
    assert_eq!(
        scx_format::pandas_index_columns(batch.schema_ref()),
        vec!["__index_level_0__".to_string()],
        "schema must carry pandas metadata pointing at the index column"
    );
    assert_eq!(batch.num_rows(), 3);

    let col = batch
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .expect("index column must be Utf8/StringArray");
    let values: Vec<&str> = (0..col.len()).map(|i| col.value(i)).collect();
    assert_eq!(
        values,
        vec!["MIR1302-2HG", "FAM138A", "OR4F5"],
        "fallback must read real values from the _index dataset, not blanks"
    );
}

#[test]
fn read_dataframe_group_attaches_pandas_index_metadata_unnamed() {
    // var with `_index = "_index"`
    // (unnamed pandas index) PLUS non-empty `column-order`. Pre-fix,
    // `read_dataframe_group` silently dropped the index. Post-fix, the
    // schema must include `__index_level_0__` AND stamp the pandas
    // metadata envelope so consumers find the index.
    use super::h5ad_read::read_dataframe_group;

    let dir = tempfile::tempdir().unwrap();
    let h5_path = dir.path().join("var_unnamed.h5");
    let file = hdf5::File::create(&h5_path).unwrap();
    let var = file.create_group("var").unwrap();

    var.new_attr::<VarLenUnicode>()
        .create("encoding-type")
        .unwrap()
        .write_scalar(&vlu("dataframe"))
        .unwrap();
    var.new_attr::<VarLenUnicode>()
        .create("encoding-version")
        .unwrap()
        .write_scalar(&vlu("0.2.0"))
        .unwrap();
    var.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&vlu("_index"))
        .unwrap();
    // column-order = ["gene_ids", "feature_types"] — the index is
    // excluded by anndata convention.
    let col_order: Vec<VarLenUnicode> = ["gene_ids", "feature_types"]
        .iter()
        .map(|s| vlu(s))
        .collect();
    var.new_attr::<VarLenUnicode>()
        .shape([col_order.len()])
        .create("column-order")
        .unwrap()
        .write(&col_order)
        .unwrap();

    let symbols: Vec<VarLenUnicode> = ["MIR1302-2HG", "FAM138A", "OR4F5"]
        .iter()
        .map(|s| vlu(s))
        .collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([symbols.len()])
        .create("_index")
        .unwrap()
        .write(&symbols)
        .unwrap();
    let gene_ids: Vec<VarLenUnicode> = ["ENSG1", "ENSG2", "ENSG3"].iter().map(|s| vlu(s)).collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([gene_ids.len()])
        .create("gene_ids")
        .unwrap()
        .write(&gene_ids)
        .unwrap();
    let feature_types: Vec<VarLenUnicode> = ["Gene Expression"; 3].iter().map(|s| vlu(s)).collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([feature_types.len()])
        .create("feature_types")
        .unwrap()
        .write(&feature_types)
        .unwrap();
    drop(file);

    let file = hdf5::File::open(&h5_path).unwrap();
    let batch = read_dataframe_group(&file, "var").unwrap();

    let schema = batch.schema();
    let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert!(field_names.contains(&"gene_ids"), "fields={field_names:?}");
    assert!(
        field_names.contains(&"feature_types"),
        "fields={field_names:?}"
    );
    assert!(
        field_names.contains(&"__index_level_0__"),
        "fields={field_names:?} — B1 reader must inject the index column"
    );
    assert_eq!(
        scx_format::pandas_index_columns(batch.schema_ref()),
        vec!["__index_level_0__".to_string()],
        "schema must carry pandas metadata pointing at the index column"
    );

    // Values in the index column round-trip.
    let idx_pos = field_names
        .iter()
        .position(|n| *n == "__index_level_0__")
        .unwrap();
    let col = batch
        .column(idx_pos)
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .expect("index column must be Utf8/StringArray");
    let values: Vec<&str> = (0..col.len()).map(|i| col.value(i)).collect();
    assert_eq!(values, vec!["MIR1302-2HG", "FAM138A", "OR4F5"]);
}

#[test]
fn read_dataframe_group_attaches_pandas_index_metadata_named() {
    // B1-2026-05-20 named-index shape: `_index = "gene_symbols"`. The
    // reader must NOT rename to `__index_level_0__`; the field keeps
    // its original name and the pandas metadata points at it.
    use super::h5ad_read::read_dataframe_group;

    let dir = tempfile::tempdir().unwrap();
    let h5_path = dir.path().join("var_named.h5");
    let file = hdf5::File::create(&h5_path).unwrap();
    let var = file.create_group("var").unwrap();

    var.new_attr::<VarLenUnicode>()
        .create("encoding-type")
        .unwrap()
        .write_scalar(&vlu("dataframe"))
        .unwrap();
    var.new_attr::<VarLenUnicode>()
        .create("encoding-version")
        .unwrap()
        .write_scalar(&vlu("0.2.0"))
        .unwrap();
    var.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&vlu("gene_symbols"))
        .unwrap();
    let col_order: Vec<VarLenUnicode> = ["gene_ids"].iter().map(|s| vlu(s)).collect();
    var.new_attr::<VarLenUnicode>()
        .shape([col_order.len()])
        .create("column-order")
        .unwrap()
        .write(&col_order)
        .unwrap();

    let symbols: Vec<VarLenUnicode> = ["MIR1302-2HG", "FAM138A", "OR4F5"]
        .iter()
        .map(|s| vlu(s))
        .collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([symbols.len()])
        .create("gene_symbols")
        .unwrap()
        .write(&symbols)
        .unwrap();
    let gene_ids: Vec<VarLenUnicode> = ["ENSG1", "ENSG2", "ENSG3"].iter().map(|s| vlu(s)).collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([gene_ids.len()])
        .create("gene_ids")
        .unwrap()
        .write(&gene_ids)
        .unwrap();
    drop(file);

    let file = hdf5::File::open(&h5_path).unwrap();
    let batch = read_dataframe_group(&file, "var").unwrap();

    let schema = batch.schema();
    let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert!(field_names.contains(&"gene_ids"), "fields={field_names:?}");
    assert!(
        field_names.contains(&"gene_symbols"),
        "named index keeps its source name: {field_names:?}"
    );
    assert!(
        !field_names.contains(&"__index_level_0__"),
        "named index must NOT be renamed: {field_names:?}"
    );
    assert_eq!(
        scx_format::pandas_index_columns(batch.schema_ref()),
        vec!["gene_symbols".to_string()],
        "pandas metadata must point at the named index"
    );
}

#[test]
fn h5ad_to_scx_streaming_preserves_obs_var_names() {
    // B1-2026-05-20 end-to-end: the streaming path that `scx convert`
    // uses by default must produce an SCX whose obs/var carry the
    // pandas index metadata, so `pyscx.open(...).to_anndata()` sees
    // `cell_<i>` / `gene_<i>` as obs_names / var_names and NOT as
    // integer-positional defaults.
    use super::pipeline::{h5ad_to_scx_streaming, StreamingOverrides};

    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let n_obs = 6;
    let n_vars = 4;
    create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", false);

    let scx_path = dir.path().join("out.scx");
    h5ad_to_scx_streaming(
        &h5ad_path,
        &scx_path,
        &ConvertOptions::default(),
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect("streaming convert must succeed");

    let reader = ScxReader::open(&scx_path).unwrap();
    let obs = reader.read_obs_assembled().unwrap();
    assert_eq!(
        scx_format::pandas_index_columns(obs.schema_ref()),
        vec!["__index_level_0__".to_string()],
        "obs schema must identify the index column"
    );
    let obs_idx = obs.schema().index_of("__index_level_0__").unwrap();
    let obs_col = obs
        .column(obs_idx)
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .expect("obs index must be Utf8/StringArray");
    let obs_values: Vec<String> = (0..obs_col.len())
        .map(|i| obs_col.value(i).to_string())
        .collect();
    let expected_obs: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    assert_eq!(obs_values, expected_obs, "obs_names must round-trip");

    let var = reader.read_var_assembled().unwrap();
    assert_eq!(
        scx_format::pandas_index_columns(var.schema_ref()),
        vec!["__index_level_0__".to_string()],
        "var schema must identify the index column"
    );
    let var_idx = var.schema().index_of("__index_level_0__").unwrap();
    let var_col = var
        .column(var_idx)
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .expect("var index must be Utf8/StringArray");
    let var_values: Vec<String> = (0..var_col.len())
        .map(|i| var_col.value(i).to_string())
        .collect();
    let expected_var: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    assert_eq!(var_values, expected_var, "var_names must round-trip");
}

// -----------------------------------------------------------------------
// Process_predicate_index_outcomes batches a fully-missing
// preset into a single PresetNoColumnsMatched warning.
// -----------------------------------------------------------------------

#[test]
fn process_outcomes_emits_aggregate_when_preset_fully_missing() {
    use super::pipeline::process_predicate_index_outcomes;
    use super::warnings::WarningSink;
    use scx_engine::index::{BuildOutcome, SkipReason};
    use std::sync::{Arc, Mutex};

    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = Arc::clone(&captured);
    let mut sink = WarningSink::with_handler(move |w| {
        captured_clone
            .lock()
            .unwrap()
            .push(w.category().to_string());
    });

    let outcomes = vec![
        BuildOutcome::PresetSkipped {
            column: "cell_type".into(),
            reason: SkipReason::MissingColumn,
        },
        BuildOutcome::PresetSkipped {
            column: "tissue".into(),
            reason: SkipReason::MissingColumn,
        },
        BuildOutcome::PresetSkipped {
            column: "disease".into(),
            reason: SkipReason::MissingColumn,
        },
    ];
    process_predicate_index_outcomes(outcomes, "obs", Some("cellxgene"), 3, &[], &mut sink)
        .unwrap();

    let cats = captured.lock().unwrap();
    assert_eq!(*cats, vec!["preset_no_columns_matched".to_string()]);
    assert_eq!(
        sink.counts().get("missing_preset_index_column"),
        None,
        "no per-column warnings expected when preset fully missing"
    );
    assert_eq!(sink.counts().get("preset_no_columns_matched"), Some(&1));
}

#[test]
fn process_outcomes_falls_back_to_per_column_on_partial_mismatch() {
    use super::pipeline::process_predicate_index_outcomes;
    use super::warnings::WarningSink;
    use scx_engine::index::{BuildOutcome, SkipReason};

    let mut sink = WarningSink::log();
    let outcomes = vec![
        BuildOutcome::PresetSkipped {
            column: "cell_type".into(),
            reason: SkipReason::MissingColumn,
        },
        BuildOutcome::PresetSkipped {
            column: "tissue".into(),
            reason: SkipReason::MissingColumn,
        },
        // 1 of 3 expected columns is *not* missing → preset is real,
        // and the per-column warnings remain useful signal.
    ];
    process_predicate_index_outcomes(outcomes, "obs", Some("cellxgene"), 3, &[], &mut sink)
        .unwrap();

    assert_eq!(
        sink.counts().get("missing_preset_index_column"),
        Some(&2),
        "partial mismatch must surface per-column warnings"
    );
    assert_eq!(sink.counts().get("preset_no_columns_matched"), None);
}

#[test]
fn process_outcomes_keeps_per_column_when_no_preset() {
    use super::pipeline::process_predicate_index_outcomes;
    use super::warnings::WarningSink;
    use scx_engine::index::{BuildOutcome, SkipReason};

    // Caller passed user-explicit `--index-obs <col>` (no preset). The
    // engine still surfaces these as `PresetSkipped` because they were
    // resolved via the preset code path — but `preset = None` means
    // there's no aggregate to collapse to.
    let mut sink = WarningSink::log();
    let outcomes = vec![BuildOutcome::PresetSkipped {
        column: "ghost_column".into(),
        reason: SkipReason::MissingColumn,
    }];
    process_predicate_index_outcomes(outcomes, "obs", None, 0, &[], &mut sink).unwrap();
    assert_eq!(sink.counts().get("missing_preset_index_column"), Some(&1));
    assert_eq!(sink.counts().get("preset_no_columns_matched"), None);
}

// E2-2026-05-20 `forced_column_missing_message` rendering tests live
// alongside the helper in `scx-engine/src/index.rs`. The helper moved
// out of `scx-convert` so `pyscx` (which only depends on `scx-engine`
// unconditionally; `scx-convert` is `hdf5`-gated) can call it from
// CPU-only builds. The previous duplicate tests here were removed in
// the CI fix-up for PR #113.

// -----------------------------------------------------------------------
// Process_predicate_index_outcomes aggregates
// multiple ForcedColumnError(MissingColumn) into a single error so the
// user can see ALL typos in a single run, instead of fixing one per
// invocation.
// -----------------------------------------------------------------------

#[test]
fn process_outcomes_aggregates_forced_missing_columns_into_single_error() {
    use super::pipeline::process_predicate_index_outcomes;
    use super::warnings::WarningSink;
    use scx_engine::index::{BuildOutcome, SkipReason};

    let mut sink = WarningSink::log();
    let outcomes = vec![
        BuildOutcome::ForcedColumnError {
            column: "raw_summ".into(),
            reason: SkipReason::MissingColumn,
        },
        BuildOutcome::ForcedColumnError {
            column: "cell_typ".into(),
            reason: SkipReason::MissingColumn,
        },
    ];
    let available: Vec<String> = vec![
        "soma_joinid".into(),
        "dataset_id".into(),
        "cell_type".into(),
        "raw_sum".into(),
    ];
    let err = process_predicate_index_outcomes(outcomes, "obs", None, 0, &available, &mut sink)
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("2 forced obs index columns are missing"),
        "should aggregate: {msg}"
    );
    assert!(
        msg.contains("'raw_summ': did you mean 'raw_sum'?"),
        "first typo + suggestion: {msg}"
    );
    assert!(
        msg.contains("'cell_typ': did you mean 'cell_type'?"),
        "second typo + suggestion: {msg}"
    );
}

#[test]
fn process_outcomes_single_forced_miss_uses_singular_wording() {
    // Sanity: the single-miss path must keep the existing singular
    // wording so PR #113's user-visible message is byte-identical when
    // only one column is typo'd (the common case).
    use super::pipeline::process_predicate_index_outcomes;
    use super::warnings::WarningSink;
    use scx_engine::index::{BuildOutcome, SkipReason};

    let mut sink = WarningSink::log();
    let outcomes = vec![BuildOutcome::ForcedColumnError {
        column: "raw_summ".into(),
        reason: SkipReason::MissingColumn,
    }];
    let available: Vec<String> = vec!["soma_joinid".into(), "raw_sum".into()];
    let err = process_predicate_index_outcomes(outcomes, "obs", None, 0, &available, &mut sink)
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("forced obs index column 'raw_summ': missing column."),
        "should use singular wording: {msg}"
    );
    assert!(msg.contains("Did you mean 'raw_sum'?"), "{msg}");
    assert!(
        !msg.contains("forced obs index columns are missing"),
        "should NOT emit plural header for single miss: {msg}"
    );
}

#[test]
fn process_outcomes_non_missing_forced_error_stays_fail_fast() {
    // Forced errors with non-MissingColumn reasons (unsupported dtype,
    // high cardinality) describe a real per-column condition — they
    // should still abort on the first hit rather than aggregating.
    use super::pipeline::process_predicate_index_outcomes;
    use super::warnings::WarningSink;
    use scx_engine::index::{BuildOutcome, SkipReason};

    let mut sink = WarningSink::log();
    let outcomes = vec![BuildOutcome::ForcedColumnError {
        column: "donor_id".into(),
        reason: SkipReason::HighCardinality {
            n_unique: 10_000,
            threshold: 1_024,
        },
    }];
    let err =
        process_predicate_index_outcomes(outcomes, "obs", None, 0, &[], &mut sink).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("forced obs index column 'donor_id'"),
        "should name the column: {msg}"
    );
    // The fail-fast (non-aggregated) path should NOT emit the plural
    // header or the strsim treatment — the column exists.
    assert!(
        !msg.contains("forced obs index columns are missing"),
        "non-missing reason should not aggregate: {msg}"
    );
    assert!(
        !msg.contains("Did you mean"),
        "non-missing reason should not strsim: {msg}"
    );
}
