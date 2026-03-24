// h5ad/10x -> scx, scx -> h5ad conversion

#[cfg(feature = "hdf5")]
mod csc_transpose;
#[cfg(feature = "hdf5")]
mod detect;
#[cfg(feature = "hdf5")]
mod dtype;
#[cfg(feature = "hdf5")]
mod h5ad_read;
#[cfg(feature = "hdf5")]
mod h5ad_write;
#[cfg(feature = "hdf5")]
mod tenx_read;

#[cfg(feature = "hdf5")]
mod pipeline {
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    use scx_codec::{CodecId, ValueEncoding};
    use scx_format::error::ScxError;
    use scx_format::header::{FileHeader, MAGIC};
    use scx_format::provenance::ProvenanceEntry;
    use scx_format::writer::ScxWriter;

    use super::detect::{detect_input_format, detect_matrix_format, InputFormat};
    use super::dtype::{detect_value_encoding, values_to_raw_bytes};
    use super::h5ad_read::{read_dataframe_group, read_layers, read_obsm, read_uns, read_x_matrix};
    use super::h5ad_write::write_scx_to_h5ad;
    use super::tenx_read::read_tenx_h5;

    #[derive(Debug, thiserror::Error)]
    pub enum ConvertError {
        #[error("HDF5 error: {0}")]
        Hdf5(#[from] hdf5::Error),

        #[error("SCX error: {0}")]
        Scx(#[from] ScxError),

        #[error("Arrow error: {0}")]
        Arrow(#[from] arrow::error::ArrowError),

        #[error("JSON error: {0}")]
        Json(#[from] serde_json::Error),

        #[error("I/O error: {0}")]
        Io(#[from] std::io::Error),

        #[error("unsupported dtype: {0}")]
        UnsupportedDtype(String),

        #[error("format mismatch: expected {expected}, got {got}")]
        FormatMismatch { expected: String, got: String },

        #[error("{0}")]
        Other(String),
    }

    pub struct ConvertOptions {
        pub shard_target_rows: u32,
        /// Explicit codec override. None = auto-select based on value distribution.
        pub codec: Option<CodecId>,
    }

    impl Default for ConvertOptions {
        fn default() -> Self {
            ConvertOptions {
                shard_target_rows: 16384,
                codec: None,
            }
        }
    }

    pub fn h5ad_to_scx(
        input: &Path,
        output: &Path,
        opts: &ConvertOptions,
    ) -> Result<(), ConvertError> {
        let file = hdf5::File::open(input)?;

        // Validate format
        let format = detect_input_format(&file)?;
        if matches!(format, InputFormat::TenX) {
            return Err(ConvertError::FormatMismatch {
                expected: "h5ad".to_string(),
                got: "10x".to_string(),
            });
        }

        // Read X matrix
        let matrix_format = detect_matrix_format(&file)?;
        let (indptr, indices, data, n_obs, n_vars) = read_x_matrix(&file, matrix_format)?;
        let nnz = *indptr.last().unwrap_or(&0) as u64;

        // Detect encoding and codec
        let (value_encoding, codec_id) = detect_value_encoding(&data, opts.codec);
        let index_dtype: u8 = if n_vars <= 65535 { 0 } else { 1 };

        // Build header
        let header = FileHeader {
            magic: MAGIC,
            format_version: 1,
            header_length: 256,
            flags: 0,
            n_obs: n_obs as u64,
            n_vars: n_vars as u64,
            nnz,
            n_csr_shards: 0,
            n_csc_shards: 0,
            shard_target_rows: opts.shard_target_rows,
            codec_id: codec_id as u8,
            index_dtype,
            endian: 0,
            reserved_padding: 0,
            root_catalog_offset: 0,
            root_catalog_length: 0,
            full_catalog_offset: 0,
            full_catalog_length: 0,
            manifest_sequence: 1,
            prev_catalog_offset: 0,
            file_checksum: 0,
            front_catalog_offset: 0,
            front_catalog_length: 0,
            reserved: [0u8; 132],
        };

        let mut writer = ScxWriter::new(output, header)?;

        // Write obs/var
        let obs = read_dataframe_group(&file, "obs")?;
        let var = read_dataframe_group(&file, "var")?;
        writer.write_obs(&obs)?;
        writer.write_var(&var)?;

        // Write CSR shards
        write_csr_shards(
            &mut writer,
            &indptr,
            &indices,
            &data,
            n_obs,
            n_vars,
            opts.shard_target_rows as usize,
            value_encoding,
            codec_id,
            index_dtype,
        )?;

        // Write optional sections
        if let Ok(obsm_map) = read_obsm(&file) {
            for (name, batch) in &obsm_map {
                writer.write_obsm(name, batch)?;
            }
        }

        if let Ok(uns) = read_uns(&file) {
            writer.write_uns(&uns)?;
        }

        if let Ok(layers) = read_layers(&file) {
            for (layer_name, (l_indptr, l_indices, l_data, l_nobs, l_nvars)) in &layers {
                let (l_enc, l_codec) = detect_value_encoding(l_data, opts.codec);
                let l_index_dtype: u8 = if *l_nvars <= 65535 { 0 } else { 1 };
                write_layer_shards(
                    &mut writer,
                    l_indptr,
                    l_indices,
                    l_data,
                    *l_nobs,
                    *l_nvars,
                    opts.shard_target_rows as usize,
                    l_enc,
                    l_codec,
                    l_index_dtype,
                    layer_name,
                )?;
            }
        }

        // Write provenance
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        writer.write_provenance(vec![ProvenanceEntry {
            timestamp,
            action: "convert".to_string(),
            tool: "scx-cli".to_string(),
            params_json: format!("{{\"input\":\"{}\",\"format\":\"h5ad\"}}", input.display()),
            input_checksums: vec![],
        }])?;

        writer.finish()?;
        Ok(())
    }

    pub fn tenx_to_scx(
        input: &Path,
        output: &Path,
        opts: &ConvertOptions,
    ) -> Result<(), ConvertError> {
        let file = hdf5::File::open(input)?;

        let format = detect_input_format(&file)?;
        if matches!(format, InputFormat::H5ad) {
            return Err(ConvertError::FormatMismatch {
                expected: "10x".to_string(),
                got: "h5ad".to_string(),
            });
        }

        let tenx = read_tenx_h5(&file)?;
        let nnz = *tenx.indptr.last().unwrap_or(&0) as u64;
        let (value_encoding, codec_id) = detect_value_encoding(&tenx.data, opts.codec);
        let index_dtype: u8 = if tenx.n_genes <= 65535 { 0 } else { 1 };

        let header = FileHeader {
            magic: MAGIC,
            format_version: 1,
            header_length: 256,
            flags: 0,
            n_obs: tenx.n_cells as u64,
            n_vars: tenx.n_genes as u64,
            nnz,
            n_csr_shards: 0,
            n_csc_shards: 0,
            shard_target_rows: opts.shard_target_rows,
            codec_id: codec_id as u8,
            index_dtype,
            endian: 0,
            reserved_padding: 0,
            root_catalog_offset: 0,
            root_catalog_length: 0,
            full_catalog_offset: 0,
            full_catalog_length: 0,
            manifest_sequence: 1,
            prev_catalog_offset: 0,
            file_checksum: 0,
            front_catalog_offset: 0,
            front_catalog_length: 0,
            reserved: [0u8; 132],
        };

        let mut writer = ScxWriter::new(output, header)?;
        writer.write_obs(&tenx.obs)?;
        writer.write_var(&tenx.var)?;

        write_csr_shards(
            &mut writer,
            &tenx.indptr,
            &tenx.indices,
            &tenx.data,
            tenx.n_cells,
            tenx.n_genes,
            opts.shard_target_rows as usize,
            value_encoding,
            codec_id,
            index_dtype,
        )?;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        writer.write_provenance(vec![ProvenanceEntry {
            timestamp,
            action: "convert".to_string(),
            tool: "scx-cli".to_string(),
            params_json: format!("{{\"input\":\"{}\",\"format\":\"10x\"}}", input.display()),
            input_checksums: vec![],
        }])?;

        writer.finish()?;
        Ok(())
    }

    pub fn scx_to_h5ad(scx_path: &Path, h5ad_path: &Path) -> Result<(), ConvertError> {
        write_scx_to_h5ad(scx_path, h5ad_path)
    }

    #[allow(clippy::too_many_arguments)]
    fn write_csr_shards(
        writer: &mut ScxWriter,
        indptr: &[i64],
        indices: &[i32],
        data: &[f32],
        n_obs: usize,
        _n_vars: usize,
        shard_target_rows: usize,
        value_encoding: ValueEncoding,
        codec_id: CodecId,
        index_dtype: u8,
    ) -> Result<(), ConvertError> {
        let _ = index_dtype; // index dtype is set in the file header; writer reads it from there

        let mut row_start: usize = 0;
        while row_start < n_obs {
            let row_end = (row_start + shard_target_rows).min(n_obs);

            // Slice indptr for this shard
            let shard_indptr_slice = &indptr[row_start..=row_end];
            let base = shard_indptr_slice[0];
            // Validate indptr values are non-negative and >= base (finding 8.6).
            let shard_indptr: Vec<u64> = shard_indptr_slice
                .iter()
                .map(|&v| {
                    if v < base {
                        Err(ConvertError::Other(format!(
                            "indptr value {v} less than base {base}"
                        )))
                    } else {
                        Ok((v - base) as u64)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Slice indices and data
            let nnz_start = usize::try_from(base)
                .map_err(|_| ConvertError::Other(format!("negative indptr base {base}")))?;
            let nnz_end = usize::try_from(*shard_indptr_slice.last().unwrap()).map_err(|_| {
                ConvertError::Other(format!(
                    "negative indptr value {}",
                    shard_indptr_slice.last().unwrap()
                ))
            })?;
            // Validate indices are non-negative before casting to u32 (finding 8.5).
            let shard_indices: Vec<u32> = indices[nnz_start..nnz_end]
                .iter()
                .map(|&v| {
                    u32::try_from(v)
                        .map_err(|_| ConvertError::Other(format!("negative column index {v}")))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let shard_data = &data[nnz_start..nnz_end];
            let raw_values = values_to_raw_bytes(shard_data, value_encoding);

            writer.write_csr_shard(
                &shard_indptr,
                &shard_indices,
                &raw_values,
                codec_id,
                value_encoding,
                row_start as u64,
            )?;

            row_start = row_end;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn write_layer_shards(
        writer: &mut ScxWriter,
        indptr: &[i64],
        indices: &[i32],
        data: &[f32],
        n_obs: usize,
        _n_vars: usize,
        shard_target_rows: usize,
        value_encoding: ValueEncoding,
        codec_id: CodecId,
        index_dtype: u8,
        layer_name: &str,
    ) -> Result<(), ConvertError> {
        let _ = index_dtype;
        let mut row_start: usize = 0;
        let mut shard_idx: u32 = 0;
        while row_start < n_obs {
            let row_end = (row_start + shard_target_rows).min(n_obs);

            let shard_indptr_slice = &indptr[row_start..=row_end];
            let base = shard_indptr_slice[0];
            // Validate indptr values are non-negative and >= base (finding 8.6).
            let shard_indptr: Vec<u64> = shard_indptr_slice
                .iter()
                .map(|&v| {
                    if v < base {
                        Err(ConvertError::Other(format!(
                            "indptr value {v} less than base {base}"
                        )))
                    } else {
                        Ok((v - base) as u64)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;

            let nnz_start = usize::try_from(base)
                .map_err(|_| ConvertError::Other(format!("negative indptr base {base}")))?;
            let nnz_end = usize::try_from(*shard_indptr_slice.last().unwrap()).map_err(|_| {
                ConvertError::Other(format!(
                    "negative indptr value {}",
                    shard_indptr_slice.last().unwrap()
                ))
            })?;
            // Validate indices are non-negative before casting to u32 (finding 8.5).
            let shard_indices: Vec<u32> = indices[nnz_start..nnz_end]
                .iter()
                .map(|&v| {
                    u32::try_from(v)
                        .map_err(|_| ConvertError::Other(format!("negative column index {v}")))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let shard_data = &data[nnz_start..nnz_end];
            let raw_values = values_to_raw_bytes(shard_data, value_encoding);

            writer.write_layer_csr_shard(
                &shard_indptr,
                &shard_indices,
                &raw_values,
                codec_id,
                value_encoding,
                row_start as u64,
                layer_name,
                shard_idx,
            )?;

            row_start = row_end;
            shard_idx += 1;
        }
        Ok(())
    }
}

#[cfg(feature = "hdf5")]
#[allow(unused_imports)]
pub use pipeline::{h5ad_to_scx, scx_to_h5ad, tenx_to_scx, ConvertError, ConvertOptions};

#[cfg(all(test, feature = "hdf5"))]
mod tests;

// MTX pipeline — always available, no hdf5 feature needed
pub mod mtx_pipeline {
    use std::path::Path;

    /// Convert an MTX directory to an SCX file.
    pub fn mtx_to_scx(
        input_dir: &Path,
        output: &Path,
        shard_target_rows: u32,
        codec_str: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        scx_mtx::mtx_to_scx(input_dir, output, shard_target_rows, codec_str, "scx-cli")?;
        Ok(())
    }
}

#[cfg(test)]
mod mtx_tests {
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
        super::mtx_pipeline::mtx_to_scx(&mtx_dir, &scx_path, 10000, "auto").unwrap();

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
        super::mtx_pipeline::mtx_to_scx(&mtx_out_dir, &scx_path2, 10000, "auto").unwrap();

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

        super::mtx_pipeline::mtx_to_scx(&mtx_dir, &scx_path, 10000, "auto").unwrap();

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

        super::mtx_pipeline::mtx_to_scx(&mtx_dir, &scx_path, 10000, "auto").unwrap();

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

        let result = super::mtx_pipeline::mtx_to_scx(&mtx_dir, &scx_path, 10000, "auto");
        assert!(result.is_err(), "should fail without barcodes.tsv");
    }
}
