// MTX-pipeline round-trip tests. Don't need hdf5 — only MTX I/O.

use scx_format::reader::ScxReader;
use std::io::Write;

/// Create a synthetic Cell Ranger–style MTX directory for testing.
fn create_test_mtx_dir(dir: &std::path::Path, n_obs: usize, n_vars: usize) {
    std::fs::create_dir_all(dir).unwrap();

    // Build CSR data
    let mut coo_entries = Vec::new();
    for row in 0..n_obs {
        let nnz_in_row = 2 + (row % 2);
        for j in 0..nnz_in_row {
            let col = (row * 3 + j) % n_vars;
            let val = ((row * 7 + j * 3 + 1) % 200 + 1) as f32;
            coo_entries.push((row, col, val));
        }
    }
    // Deduplicate (row, col) pairs, keeping last value
    coo_entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    coo_entries.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

    // Write matrix.mtx (uncompressed for simplicity in test)
    let mtx_path = dir.join("matrix.mtx");
    let mut f = std::fs::File::create(&mtx_path).unwrap();
    writeln!(f, "%%MatrixMarket matrix coordinate integer general").unwrap();
    writeln!(f, "% test data").unwrap();
    writeln!(f, "{} {} {}", n_obs, n_vars, coo_entries.len()).unwrap();
    for (row, col, val) in &coo_entries {
        writeln!(f, "{} {} {}", row + 1, col + 1, *val as i64).unwrap();
    }

    // Write barcodes.tsv
    let barcodes_path = dir.join("barcodes.tsv");
    let mut f = std::fs::File::create(&barcodes_path).unwrap();
    for i in 0..n_obs {
        writeln!(f, "cell_{}", i).unwrap();
    }

    // Write features.tsv
    let features_path = dir.join("features.tsv");
    let mut f = std::fs::File::create(&features_path).unwrap();
    for i in 0..n_vars {
        writeln!(f, "ENSG{:08}\tGene{}\tGene Expression", i, i).unwrap();
    }
}

/// Create a gzipped Cell Ranger–style MTX directory.
fn create_test_mtx_dir_gz(dir: &std::path::Path, n_obs: usize, n_vars: usize) {
    use flate2::write::GzEncoder;
    use flate2::Compression;

    std::fs::create_dir_all(dir).unwrap();

    let mut coo_entries = Vec::new();
    for row in 0..n_obs {
        let nnz_in_row = 2 + (row % 2);
        for j in 0..nnz_in_row {
            let col = (row * 3 + j) % n_vars;
            let val = ((row * 7 + j * 3 + 1) % 200 + 1) as f32;
            coo_entries.push((row, col, val));
        }
    }
    coo_entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    coo_entries.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

    // matrix.mtx.gz
    let file = std::fs::File::create(dir.join("matrix.mtx.gz")).unwrap();
    let mut gz = GzEncoder::new(file, Compression::default());
    writeln!(gz, "%%MatrixMarket matrix coordinate integer general").unwrap();
    writeln!(gz, "{} {} {}", n_obs, n_vars, coo_entries.len()).unwrap();
    for (row, col, val) in &coo_entries {
        writeln!(gz, "{} {} {}", row + 1, col + 1, *val as i64).unwrap();
    }
    gz.finish().unwrap();

    // barcodes.tsv.gz
    let file = std::fs::File::create(dir.join("barcodes.tsv.gz")).unwrap();
    let mut gz = GzEncoder::new(file, Compression::default());
    for i in 0..n_obs {
        writeln!(gz, "cell_{}", i).unwrap();
    }
    gz.finish().unwrap();

    // features.tsv.gz
    let file = std::fs::File::create(dir.join("features.tsv.gz")).unwrap();
    let mut gz = GzEncoder::new(file, Compression::default());
    for i in 0..n_vars {
        writeln!(gz, "ENSG{:08}\tGene{}\tGene Expression", i, i).unwrap();
    }
    gz.finish().unwrap();
}

#[test]
fn test_mtx_to_scx_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("mtx_input");
    let scx_path = dir.path().join("test.scx");
    let mtx_out_dir = dir.path().join("mtx_output");

    let n_obs = 20;
    let n_vars = 15;
    create_test_mtx_dir(&mtx_dir, n_obs, n_vars);

    // MTX → SCX
    crate::mtx_pipeline::mtx_to_scx(&mtx_dir, &scx_path, 10000, "auto").unwrap();

    // Verify SCX
    let reader = ScxReader::open(&scx_path).unwrap();
    assert_eq!(reader.n_obs(), n_obs as u64);
    assert_eq!(reader.n_vars(), n_vars as u64);

    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, n_obs);
    assert_eq!(csr.shape.1, n_vars);
    assert!(csr.nnz() > 0);

    // SCX → MTX
    scx_mtx::write_scx_to_mtx(&scx_path, &mtx_out_dir).unwrap();

    // Verify output files exist
    assert!(
        mtx_out_dir.join("matrix.mtx.gz").exists(),
        "matrix.mtx.gz should exist"
    );
    assert!(
        mtx_out_dir.join("barcodes.tsv.gz").exists(),
        "barcodes.tsv.gz should exist"
    );
    assert!(
        mtx_out_dir.join("features.tsv.gz").exists(),
        "features.tsv.gz should exist"
    );

    // Read back the output MTX and convert again
    let scx_path2 = dir.path().join("test2.scx");
    crate::mtx_pipeline::mtx_to_scx(&mtx_out_dir, &scx_path2, 10000, "auto").unwrap();

    let reader2 = ScxReader::open(&scx_path2).unwrap();
    assert_eq!(reader2.n_obs(), n_obs as u64);
    assert_eq!(reader2.n_vars(), n_vars as u64);

    let csr2 = reader2.read_all_csr_shards().unwrap();
    assert_eq!(csr.nnz(), csr2.nnz(), "nnz mismatch after round-trip");

    // Verify data matches
    assert_eq!(csr.data, csr2.data, "data mismatch after round-trip");
}

#[test]
fn test_mtx_gzipped() {
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("mtx_gz");
    let scx_path = dir.path().join("test_gz.scx");

    create_test_mtx_dir_gz(&mtx_dir, 15, 10);

    crate::mtx_pipeline::mtx_to_scx(&mtx_dir, &scx_path, 10000, "auto").unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    assert_eq!(reader.n_obs(), 15);
    assert_eq!(reader.n_vars(), 10);
}

#[test]
fn test_mtx_old_genes_tsv() {
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("mtx_genes");
    let scx_path = dir.path().join("test_genes.scx");

    // Create with genes.tsv instead of features.tsv
    create_test_mtx_dir(&mtx_dir, 10, 8);
    // Rename features.tsv to genes.tsv
    std::fs::rename(mtx_dir.join("features.tsv"), mtx_dir.join("genes.tsv")).unwrap();

    crate::mtx_pipeline::mtx_to_scx(&mtx_dir, &scx_path, 10000, "auto").unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    assert_eq!(reader.n_obs(), 10);
    assert_eq!(reader.n_vars(), 8);
}

#[test]
fn test_mtx_missing_sidecars() {
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("mtx_no_barcodes");
    let scx_path = dir.path().join("test_missing.scx");

    // Create only matrix.mtx, no barcodes or features
    std::fs::create_dir_all(&mtx_dir).unwrap();
    let mut f = std::fs::File::create(mtx_dir.join("matrix.mtx")).unwrap();
    writeln!(f, "%%MatrixMarket matrix coordinate integer general").unwrap();
    writeln!(f, "2 2 1").unwrap();
    writeln!(f, "1 1 1").unwrap();

    let result = crate::mtx_pipeline::mtx_to_scx(&mtx_dir, &scx_path, 10000, "auto");
    assert!(result.is_err(), "should fail without barcodes.tsv");
}
