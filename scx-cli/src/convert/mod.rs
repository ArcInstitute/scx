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
    }

    impl Default for ConvertOptions {
        fn default() -> Self {
            ConvertOptions {
                shard_target_rows: 16384,
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

        // Detect encoding
        let (value_encoding, codec_id) = detect_value_encoding(&data);
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
                let (l_enc, l_codec) = detect_value_encoding(l_data);
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
        let (value_encoding, codec_id) = detect_value_encoding(&tenx.data);
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
            let shard_indptr: Vec<u64> = shard_indptr_slice
                .iter()
                .map(|&v| (v - base) as u64)
                .collect();

            // Slice indices and data
            let nnz_start = base as usize;
            let nnz_end = *shard_indptr_slice.last().unwrap() as usize;
            let shard_indices: Vec<u32> = indices[nnz_start..nnz_end]
                .iter()
                .map(|&v| v as u32)
                .collect();
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
            let shard_indptr: Vec<u64> = shard_indptr_slice
                .iter()
                .map(|&v| (v - base) as u64)
                .collect();

            let nnz_start = base as usize;
            let nnz_end = *shard_indptr_slice.last().unwrap() as usize;
            let shard_indices: Vec<u32> = indices[nnz_start..nnz_end]
                .iter()
                .map(|&v| v as u32)
                .collect();
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
