//! Integration tests: MTX ↔ SCX round-trip parity.
//!
//! Verifies that `read_mtx_directory → mtx_to_scx → write_scx_to_mtx →
//! read_mtx_directory` preserves matrix data and obs/var metadata.

use std::io::Write;

use arrow::array::AsArray;
use flate2::write::GzEncoder;
use flate2::Compression;
use tempfile::TempDir;

use scx_mtx::{mtx_to_scx, read_mtx_directory, write_scx_to_mtx, MtxError, MtxOrientation};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a Cell Ranger v3–style MTX directory (features.tsv.gz).
///
/// Cell Ranger writes `matrix.mtx` as **features × barcodes** (size line
/// `<genes> <cells> <nnz>`), so this fixture is a genuine 4-genes × 3-cells
/// layout. On read it is detected as `FeaturesByBarcodes` and transposed to a
/// 3-cells × 4-genes CSR. The entries are the transpose of the logical
/// cells×genes matrix `{(c0,g1)=1, (c0,g3)=2, (c1,g0)=3, (c2,g1)=4, (c2,g2)=5}`.
fn create_v3_mtx_dir(dir: &std::path::Path) {
    // matrix.mtx.gz — 4 genes × 3 cells (features × barcodes), 5 nonzeros
    let mtx = "\
%%MatrixMarket matrix coordinate integer general
4 3 5
1 2 3
2 1 1
2 3 4
3 3 5
4 1 2
";
    write_gz(dir, "matrix.mtx.gz", mtx.as_bytes());

    // barcodes.tsv.gz
    let barcodes = "AAACCCAA-1\nBBBDDDBB-1\nCCCEEECC-1\n";
    write_gz(dir, "barcodes.tsv.gz", barcodes.as_bytes());

    // features.tsv.gz (v3: 3-column, with feature_type)
    let features = "ENSG001\tGeneA\tGene Expression\n\
                    ENSG002\tGeneB\tGene Expression\n\
                    ENSG003\tGeneC\tGene Expression\n\
                    ENSG004\tGeneD\tGene Expression\n";
    write_gz(dir, "features.tsv.gz", features.as_bytes());
}

/// Create a Cell Ranger v2–style MTX directory (genes.tsv.gz, no feature_type).
fn create_v2_mtx_dir(dir: &std::path::Path) {
    // Same matrix data — 4 genes × 3 cells (features × barcodes).
    let mtx = "\
%%MatrixMarket matrix coordinate integer general
4 3 5
1 2 3
2 1 1
2 3 4
3 3 5
4 1 2
";
    write_gz(dir, "matrix.mtx.gz", mtx.as_bytes());

    // barcodes.tsv.gz
    let barcodes = "AAACCCAA-1\nBBBDDDBB-1\nCCCEEECC-1\n";
    write_gz(dir, "barcodes.tsv.gz", barcodes.as_bytes());

    // genes.tsv.gz (v2: 2-column, no feature_type)
    let genes = "ENSG001\tGeneA\nENSG002\tGeneB\nENSG003\tGeneC\nENSG004\tGeneD\n";
    write_gz(dir, "genes.tsv.gz", genes.as_bytes());
}

fn write_gz(dir: &std::path::Path, name: &str, data: &[u8]) {
    let path = dir.join(name);
    let file = std::fs::File::create(path).unwrap();
    let mut enc = GzEncoder::new(file, Compression::fast());
    enc.write_all(data).unwrap();
    enc.finish().unwrap();
}

fn extract_string_column(batch: &arrow::array::RecordBatch, col: usize) -> Vec<String> {
    batch
        .column(col)
        .as_string::<i32>()
        .iter()
        .map(|v| v.unwrap().to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Full round-trip: MTX v3 → SCX → MTX → read, asserting data + metadata parity.
#[test]
fn round_trip_v3_features() {
    let tmp = TempDir::new().unwrap();
    let mtx_in = tmp.path().join("mtx_in");
    std::fs::create_dir_all(&mtx_in).unwrap();
    create_v3_mtx_dir(&mtx_in);

    // 1. Read the original MTX
    let original = read_mtx_directory(&mtx_in).unwrap();

    // 2. Convert MTX → SCX
    let scx_path = tmp.path().join("test.scx");
    mtx_to_scx(&mtx_in, &scx_path, 1000, "auto", "round-trip-test").unwrap();

    // 3. Convert SCX → MTX
    let mtx_out = tmp.path().join("mtx_out");
    write_scx_to_mtx(&scx_path, &mtx_out).unwrap();

    // 4. Read the round-tripped MTX
    let round_tripped = read_mtx_directory(&mtx_out).unwrap();

    // Assert matrix shape parity
    assert_eq!(original.n_obs, round_tripped.n_obs, "n_obs mismatch");
    assert_eq!(original.n_vars, round_tripped.n_vars, "n_vars mismatch");

    // Assert CSR data parity
    assert_eq!(original.indptr, round_tripped.indptr, "indptr mismatch");
    assert_eq!(original.indices, round_tripped.indices, "indices mismatch");
    assert_eq!(original.data, round_tripped.data, "data mismatch");

    // Assert obs (barcodes) parity
    let orig_barcodes = extract_string_column(&original.obs, 0);
    let rt_barcodes = extract_string_column(&round_tripped.obs, 0);
    assert_eq!(orig_barcodes, rt_barcodes, "barcodes mismatch");

    // Assert var (features) parity — v3 should have 3 columns
    assert_eq!(
        original.var.num_columns(),
        3,
        "v3 features should have 3 columns"
    );
    let orig_gene_ids = extract_string_column(&original.var, 0);
    let rt_gene_ids = extract_string_column(&round_tripped.var, 0);
    assert_eq!(orig_gene_ids, rt_gene_ids, "gene_ids mismatch");

    let orig_gene_names = extract_string_column(&original.var, 1);
    let rt_gene_names = extract_string_column(&round_tripped.var, 1);
    assert_eq!(orig_gene_names, rt_gene_names, "gene_names mismatch");

    let orig_feature_types = extract_string_column(&original.var, 2);
    let rt_feature_types = extract_string_column(&round_tripped.var, 2);
    assert_eq!(
        orig_feature_types, rt_feature_types,
        "feature_types mismatch"
    );
}

/// Full round-trip: MTX v2 → SCX → MTX → read, asserting data + metadata parity.
#[test]
fn round_trip_v2_genes() {
    let tmp = TempDir::new().unwrap();
    let mtx_in = tmp.path().join("mtx_in");
    std::fs::create_dir_all(&mtx_in).unwrap();
    create_v2_mtx_dir(&mtx_in);

    // 1. Read the original MTX
    let original = read_mtx_directory(&mtx_in).unwrap();

    // 2. Convert MTX → SCX
    let scx_path = tmp.path().join("test.scx");
    mtx_to_scx(&mtx_in, &scx_path, 1000, "auto", "round-trip-test").unwrap();

    // 3. Convert SCX → MTX
    let mtx_out = tmp.path().join("mtx_out");
    write_scx_to_mtx(&scx_path, &mtx_out).unwrap();

    // 4. Read the round-tripped MTX
    let round_tripped = read_mtx_directory(&mtx_out).unwrap();

    // Assert matrix shape parity
    assert_eq!(original.n_obs, round_tripped.n_obs, "n_obs mismatch");
    assert_eq!(original.n_vars, round_tripped.n_vars, "n_vars mismatch");

    // Assert CSR data parity
    assert_eq!(original.indptr, round_tripped.indptr, "indptr mismatch");
    assert_eq!(original.indices, round_tripped.indices, "indices mismatch");
    assert_eq!(original.data, round_tripped.data, "data mismatch");

    // Assert obs (barcodes) parity
    let orig_barcodes = extract_string_column(&original.obs, 0);
    let rt_barcodes = extract_string_column(&round_tripped.obs, 0);
    assert_eq!(orig_barcodes, rt_barcodes, "barcodes mismatch");

    // Assert var (genes) parity — v2 should have 2 columns (no feature_type)
    assert_eq!(
        original.var.num_columns(),
        2,
        "v2 genes should have 2 columns"
    );
    let orig_gene_ids = extract_string_column(&original.var, 0);
    let rt_gene_ids = extract_string_column(&round_tripped.var, 0);
    assert_eq!(orig_gene_ids, rt_gene_ids, "gene_ids mismatch");

    let orig_gene_names = extract_string_column(&original.var, 1);
    let rt_gene_names = extract_string_column(&round_tripped.var, 1);
    assert_eq!(orig_gene_names, rt_gene_names, "gene_names mismatch");
}

/// Round-trip with all codec variants: none, scx1, zstd.
#[test]
fn round_trip_all_codecs() {
    for codec in &["none", "scx1", "zstd"] {
        let tmp = TempDir::new().unwrap();
        let mtx_in = tmp.path().join("mtx_in");
        std::fs::create_dir_all(&mtx_in).unwrap();
        create_v3_mtx_dir(&mtx_in);

        let original = read_mtx_directory(&mtx_in).unwrap();

        let scx_path = tmp.path().join("test.scx");
        mtx_to_scx(&mtx_in, &scx_path, 2, codec, "codec-test").unwrap();

        let mtx_out = tmp.path().join("mtx_out");
        write_scx_to_mtx(&scx_path, &mtx_out).unwrap();

        let round_tripped = read_mtx_directory(&mtx_out).unwrap();

        assert_eq!(
            original.indptr, round_tripped.indptr,
            "indptr mismatch with codec={codec}"
        );
        assert_eq!(
            original.indices, round_tripped.indices,
            "indices mismatch with codec={codec}"
        );
        assert_eq!(
            original.data, round_tripped.data,
            "data mismatch with codec={codec}"
        );
    }
}

/// Round-trip with small shard size to exercise multi-shard writes.
#[test]
fn round_trip_multi_shard() {
    let tmp = TempDir::new().unwrap();
    let mtx_in = tmp.path().join("mtx_in");
    std::fs::create_dir_all(&mtx_in).unwrap();
    create_v3_mtx_dir(&mtx_in);

    let original = read_mtx_directory(&mtx_in).unwrap();

    // shard_target_rows = 1 → each row is its own shard
    let scx_path = tmp.path().join("test.scx");
    mtx_to_scx(&mtx_in, &scx_path, 1, "auto", "shard-test").unwrap();

    let mtx_out = tmp.path().join("mtx_out");
    write_scx_to_mtx(&scx_path, &mtx_out).unwrap();

    let round_tripped = read_mtx_directory(&mtx_out).unwrap();

    assert_eq!(original.n_obs, round_tripped.n_obs);
    assert_eq!(original.n_vars, round_tripped.n_vars);
    assert_eq!(original.indptr, round_tripped.indptr);
    assert_eq!(original.indices, round_tripped.indices);
    assert_eq!(original.data, round_tripped.data);
}

/// Round-trip with a real-valued (float) MTX matrix.
#[test]
fn round_trip_real_values() {
    let tmp = TempDir::new().unwrap();
    let mtx_in = tmp.path().join("mtx_in");
    std::fs::create_dir_all(&mtx_in).unwrap();

    // Real-valued matrix
    let mtx = "\
%%MatrixMarket matrix coordinate real general
2 3 3
1 1 1.5
1 3 2.7
2 2 0.001
";
    write_gz(&mtx_in, "matrix.mtx.gz", mtx.as_bytes());
    write_gz(&mtx_in, "barcodes.tsv.gz", b"CELL_A-1\nCELL_B-1\n");
    write_gz(
        &mtx_in,
        "features.tsv.gz",
        b"G1\tGene1\tGene Expression\nG2\tGene2\tGene Expression\nG3\tGene3\tGene Expression\n",
    );

    let original = read_mtx_directory(&mtx_in).unwrap();

    // Float values should use zstd codec (scx1 is integer-only)
    let scx_path = tmp.path().join("test.scx");
    mtx_to_scx(&mtx_in, &scx_path, 1000, "auto", "real-test").unwrap();

    let mtx_out = tmp.path().join("mtx_out");
    write_scx_to_mtx(&scx_path, &mtx_out).unwrap();

    let round_tripped = read_mtx_directory(&mtx_out).unwrap();

    assert_eq!(original.n_obs, round_tripped.n_obs);
    assert_eq!(original.n_vars, round_tripped.n_vars);
    assert_eq!(original.indptr, round_tripped.indptr);
    assert_eq!(original.indices, round_tripped.indices);

    // Float round-trip: check bitwise equality (f32 → f32, no precision loss)
    for (i, (o, r)) in original
        .data
        .iter()
        .zip(round_tripped.data.iter())
        .enumerate()
    {
        assert!(
            o.to_bits() == r.to_bits(),
            "data[{i}] bitwise mismatch: {o} vs {r}"
        );
    }
}

// ---------------------------------------------------------------------------
// B1: Cell Ranger orientation (features × barcodes → cells × genes)
// ---------------------------------------------------------------------------

/// A genuine Cell Ranger `filtered_feature_bc_matrix/` (features × barcodes,
/// non-square) must read back as a cells×genes matrix: 3 cells × 4 genes, with
/// the matrix transposed into the SCX/scipy orientation.
#[test]
fn cell_ranger_features_by_barcodes_is_transposed() {
    let tmp = TempDir::new().unwrap();
    let mtx_in = tmp.path().join("mtx_in");
    std::fs::create_dir_all(&mtx_in).unwrap();
    create_v3_mtx_dir(&mtx_in); // 4 genes × 3 cells on disk

    let data = read_mtx_directory(&mtx_in).unwrap();

    // Orientation detected and matrix transposed to cells×genes.
    assert_eq!(data.orientation, MtxOrientation::FeaturesByBarcodes);
    assert_eq!(data.n_obs, 3, "n_obs should be the 3 barcodes (cells)");
    assert_eq!(data.n_vars, 4, "n_vars should be the 4 features (genes)");

    // obs = barcodes (cells), var = features (genes) — unaffected by transpose.
    assert_eq!(data.obs.num_rows(), 3);
    assert_eq!(data.var.num_rows(), 4);
    let barcodes = extract_string_column(&data.obs, 0);
    assert_eq!(barcodes, vec!["AAACCCAA-1", "BBBDDDBB-1", "CCCEEECC-1"]);

    // Values survive the transpose, landing in the correct cells×genes CSR.
    // Logical matrix: (c0,g1)=1, (c0,g3)=2, (c1,g0)=3, (c2,g1)=4, (c2,g2)=5.
    assert_eq!(data.indptr, vec![0, 2, 3, 5]);
    assert_eq!(data.indices, vec![1, 3, 0, 1, 2]);
    assert_eq!(data.data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

/// The detector must discriminate a features×barcodes (4×3) layout from a
/// cells×genes (3×4) layout that encode the *same* logical matrix: one is
/// transposed, one is kept, and both yield identical cells×genes CSR.
#[test]
fn discriminates_features_by_barcodes_from_cells_by_genes() {
    // (a) Cell Ranger layout: 4 genes × 3 cells, transposed on read.
    let tmp_a = TempDir::new().unwrap();
    let dir_a = tmp_a.path().join("mtx");
    std::fs::create_dir_all(&dir_a).unwrap();
    create_v3_mtx_dir(&dir_a);
    let a = read_mtx_directory(&dir_a).unwrap();
    assert_eq!(a.orientation, MtxOrientation::FeaturesByBarcodes);

    // (b) cells×genes layout: 3 cells × 4 genes, kept as-is. Same logical data.
    let tmp_b = TempDir::new().unwrap();
    let dir_b = tmp_b.path().join("mtx");
    std::fs::create_dir_all(&dir_b).unwrap();
    let mtx_cells_by_genes = "\
%%MatrixMarket matrix coordinate integer general
3 4 5
1 2 1
1 4 2
2 1 3
3 2 4
3 3 5
";
    write_gz(&dir_b, "matrix.mtx.gz", mtx_cells_by_genes.as_bytes());
    write_gz(
        &dir_b,
        "barcodes.tsv.gz",
        b"AAACCCAA-1\nBBBDDDBB-1\nCCCEEECC-1\n",
    );
    write_gz(
        &dir_b,
        "features.tsv.gz",
        b"ENSG001\tGeneA\tGene Expression\nENSG002\tGeneB\tGene Expression\n\
          ENSG003\tGeneC\tGene Expression\nENSG004\tGeneD\tGene Expression\n",
    );
    let b = read_mtx_directory(&dir_b).unwrap();
    assert_eq!(b.orientation, MtxOrientation::BarcodesByFeatures);

    // Both resolve to the same cells×genes CSR.
    assert_eq!(a.n_obs, b.n_obs);
    assert_eq!(a.n_vars, b.n_vars);
    assert_eq!(a.indptr, b.indptr);
    assert_eq!(a.indices, b.indices);
    assert_eq!(a.data, b.data);
}

/// A matrix whose dimensions match neither (features×barcodes) nor
/// (barcodes×features) must fail loud with `OrientationMismatch`.
#[test]
fn orientation_mismatch_is_fail_loud() {
    let tmp = TempDir::new().unwrap();
    let mtx_in = tmp.path().join("mtx_in");
    std::fs::create_dir_all(&mtx_in).unwrap();

    // 3 barcodes, 4 features, but a 2×2 matrix — matches neither layout.
    let mtx = "\
%%MatrixMarket matrix coordinate integer general
2 2 1
1 1 7
";
    write_gz(&mtx_in, "matrix.mtx.gz", mtx.as_bytes());
    write_gz(&mtx_in, "barcodes.tsv.gz", b"A-1\nB-1\nC-1\n");
    write_gz(
        &mtx_in,
        "features.tsv.gz",
        b"G1\tg1\tGE\nG2\tg2\tGE\nG3\tg3\tGE\nG4\tg4\tGE\n",
    );

    let err = match read_mtx_directory(&mtx_in) {
        Err(e) => e,
        Ok(_) => panic!("expected OrientationMismatch, got Ok"),
    };
    assert!(
        matches!(
            err,
            MtxError::OrientationMismatch {
                n_rows: 2,
                n_cols: 2,
                n_barcodes: 3,
                n_features: 4,
            }
        ),
        "expected OrientationMismatch, got: {err:?}"
    );
}

/// A COO coordinate outside the declared size line must fail loud with a
/// `Parse` error rather than panicking in the CSR build or the transpose.
#[test]
fn out_of_range_coordinate_is_fail_loud() {
    let tmp = TempDir::new().unwrap();
    let mtx_in = tmp.path().join("mtx_in");
    std::fs::create_dir_all(&mtx_in).unwrap();

    // Declares 4 genes × 3 cells, but a triplet references gene 5 (row 5).
    let mtx = "\
%%MatrixMarket matrix coordinate integer general
4 3 1
5 1 7
";
    write_gz(&mtx_in, "matrix.mtx.gz", mtx.as_bytes());
    write_gz(&mtx_in, "barcodes.tsv.gz", b"A-1\nB-1\nC-1\n");
    write_gz(
        &mtx_in,
        "features.tsv.gz",
        b"G1\tg1\tGE\nG2\tg2\tGE\nG3\tg3\tGE\nG4\tg4\tGE\n",
    );

    match read_mtx_directory(&mtx_in) {
        Err(MtxError::Parse(msg)) => assert!(
            msg.contains("row index 5") && msg.contains("out of range"),
            "expected a row-out-of-range Parse error, got: {msg}"
        ),
        Err(other) => panic!("expected MtxError::Parse, got: {other:?}"),
        Ok(_) => panic!("expected MtxError::Parse, got Ok"),
    }
}

/// Verify that empty matrix round-trips cleanly (edge case).
#[test]
fn round_trip_empty_matrix() {
    let tmp = TempDir::new().unwrap();
    let mtx_in = tmp.path().join("mtx_in");
    std::fs::create_dir_all(&mtx_in).unwrap();

    let mtx = "\
%%MatrixMarket matrix coordinate integer general
3 4 0
";
    write_gz(&mtx_in, "matrix.mtx.gz", mtx.as_bytes());
    write_gz(&mtx_in, "barcodes.tsv.gz", b"A-1\nB-1\nC-1\n");
    write_gz(
        &mtx_in,
        "features.tsv.gz",
        b"G1\tg1\tGE\nG2\tg2\tGE\nG3\tg3\tGE\nG4\tg4\tGE\n",
    );

    let original = read_mtx_directory(&mtx_in).unwrap();
    assert_eq!(original.data.len(), 0);

    let scx_path = tmp.path().join("test.scx");
    mtx_to_scx(&mtx_in, &scx_path, 1000, "auto", "empty-test").unwrap();

    let mtx_out = tmp.path().join("mtx_out");
    write_scx_to_mtx(&scx_path, &mtx_out).unwrap();

    let round_tripped = read_mtx_directory(&mtx_out).unwrap();

    assert_eq!(original.n_obs, round_tripped.n_obs);
    assert_eq!(original.n_vars, round_tripped.n_vars);
    assert_eq!(round_tripped.data.len(), 0);
    assert_eq!(original.indptr, round_tripped.indptr);
}

/// SCX-010: exported `matrix.mtx` must be features × barcodes (Cell Ranger /
/// 10x convention), not the internal cells × genes orientation. Assert the
/// raw size line directly so a self-round-trip (which the reader would silently
/// re-transpose) cannot mask a wrong orientation.
#[test]
fn export_is_features_by_barcodes() {
    use std::io::Read;

    let tmp = TempDir::new().unwrap();
    let mtx_in = tmp.path().join("mtx_in");
    std::fs::create_dir_all(&mtx_in).unwrap();
    // 4 genes × 3 cells, 5 nnz. On read → SCX with n_obs=3 (cells),
    // n_vars=4 (genes).
    create_v3_mtx_dir(&mtx_in);

    let scx_path = tmp.path().join("test.scx");
    mtx_to_scx(&mtx_in, &scx_path, 1000, "auto", "orient-test").unwrap();

    let mtx_out = tmp.path().join("mtx_out");
    write_scx_to_mtx(&scx_path, &mtx_out).unwrap();

    // Decompress matrix.mtx.gz and find the size line (first non-comment line).
    let gz = std::fs::File::open(mtx_out.join("matrix.mtx.gz")).unwrap();
    let mut dec = flate2::read::GzDecoder::new(gz);
    let mut body = String::new();
    dec.read_to_string(&mut body).unwrap();
    let size_line = body
        .lines()
        .find(|l| !l.starts_with('%'))
        .expect("size line");
    // features × barcodes = n_vars n_obs nnz = 4 3 5, NOT 3 4 5.
    assert_eq!(
        size_line.trim(),
        "4 3 5",
        "export must be features × barcodes (n_vars n_obs nnz)"
    );
}
