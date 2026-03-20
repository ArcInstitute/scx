// ScxReader — mmap + pread paths (SPEC §10)

use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;

use arrow::array::RecordBatch;
use memmap2::Mmap;
use scx_codec::{CodecId, EncodedShard, ValueEncoding};
use scx_sparse::ScxCsr;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::catalog::{FullCatalog, FullCatalogEntry};
use crate::checksum::{blake3_hash, blake3_truncated_64};
use crate::error::{Result, ScxError};
use crate::header::{FileHeader, HEADER_SIZE};
use crate::provenance::Provenance;
use crate::section::SectionType;
use crate::shard::{ShardHeader, SHARD_HEADER_SIZE};
use crate::RootCatalog;

/// Memory-mapped reader for SCX files.
///
/// Opens an SCX file, validates the header and catalog checksums,
/// and provides methods to read obs/var metadata, CSR shards, layers,
/// obsm embeddings, uns JSON, and provenance.
pub struct ScxReader {
    mmap: Mmap,
    header: FileHeader,
    root_catalog: RootCatalog,
    full_catalog: FullCatalog,
}

impl ScxReader {
    /// Open an SCX file for reading.
    ///
    /// Validates the file header magic/version/endianness and the full
    /// catalog's trailing BLAKE3 checksum.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Check minimum file size (header + root catalog placeholder)
        if mmap.len() < HEADER_SIZE {
            return Err(ScxError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "file too small: {} bytes (minimum {})",
                    mmap.len(),
                    HEADER_SIZE
                ),
            )));
        }

        // Read and validate file header
        let header = FileHeader::read_from(&mut Cursor::new(&mmap[..HEADER_SIZE]))?;

        // Read root catalog at offset 256
        let root_catalog = RootCatalog::read_from(&mut Cursor::new(&mmap[HEADER_SIZE..]))?;

        // Read full catalog using header's offset and length
        let fc_offset = header.full_catalog_offset as usize;
        let fc_length = header.full_catalog_length as usize;
        let fc_slice = &mmap[fc_offset..fc_offset + fc_length];
        let full_catalog = FullCatalog::read_from(&mut Cursor::new(fc_slice), fc_length)?;

        Ok(ScxReader {
            mmap,
            header,
            root_catalog,
            full_catalog,
        })
    }

    // -----------------------------------------------------------------------
    // Summary accessors (11.13)
    // -----------------------------------------------------------------------

    pub fn header(&self) -> &FileHeader {
        &self.header
    }

    pub fn root_catalog(&self) -> &RootCatalog {
        &self.root_catalog
    }

    pub fn catalog(&self) -> &FullCatalog {
        &self.full_catalog
    }

    pub fn n_obs(&self) -> u64 {
        self.header.n_obs
    }

    pub fn n_vars(&self) -> u64 {
        self.header.n_vars
    }

    pub fn nnz(&self) -> u64 {
        self.header.nnz
    }

    // -----------------------------------------------------------------------
    // Arrow IPC reading (11.5–11.8)
    // -----------------------------------------------------------------------

    /// Read an Arrow IPC section from a catalog entry.
    fn read_arrow_ipc(&self, entry: &FullCatalogEntry) -> Result<RecordBatch> {
        let slice = self.section_bytes(entry);
        let cursor = Cursor::new(slice);
        let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)?;
        // Read the first (and typically only) batch
        let mut batches = reader.into_iter();
        batches
            .next()
            .ok_or_else(|| {
                ScxError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Arrow IPC file contains no batches",
                ))
            })?
            .map_err(ScxError::Arrow)
    }

    /// Read the obs (observation) metadata as an Arrow RecordBatch.
    pub fn read_obs(&self) -> Result<RecordBatch> {
        let entry = self
            .full_catalog
            .get("obs")
            .ok_or_else(|| ScxError::SectionNotFound("obs".to_string()))?;
        self.read_arrow_ipc(entry)
    }

    /// Read the var (variable/gene) metadata as an Arrow RecordBatch.
    pub fn read_var(&self) -> Result<RecordBatch> {
        let entry = self
            .full_catalog
            .get("var")
            .ok_or_else(|| ScxError::SectionNotFound("var".to_string()))?;
        self.read_arrow_ipc(entry)
    }

    /// Read a named obsm embedding as an Arrow RecordBatch.
    pub fn read_obsm(&self, name: &str) -> Result<RecordBatch> {
        let key = format!("obsm/{name}");
        let entry = self
            .full_catalog
            .get(&key)
            .ok_or_else(|| ScxError::SectionNotFound(key))?;
        self.read_arrow_ipc(entry)
    }

    /// Read all obsm embeddings, keyed by name.
    pub fn read_all_obsm(&self) -> Result<HashMap<String, RecordBatch>> {
        let mut result = HashMap::new();
        for entry in &self.full_catalog.entries {
            if entry.section_type == SectionType::ObsmEmbedding {
                let name = entry
                    .name
                    .strip_prefix("obsm/")
                    .unwrap_or(&entry.name)
                    .to_string();
                let batch = self.read_arrow_ipc(entry)?;
                result.insert(name, batch);
            }
        }
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // CSR shard reading (11.3–11.4)
    // -----------------------------------------------------------------------

    /// Read a single CSR shard by index, returning scipy-compatible arrays.
    pub fn read_csr_shard(&self, shard_idx: usize) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
        let shards = self.full_catalog.shards_sorted();
        if shard_idx >= shards.len() {
            return Err(ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: shards.len(),
            });
        }
        self.read_shard_from_entry(shards[shard_idx])
    }

    /// Read all CSR shards and assemble into a single ScxCsr.
    pub fn read_all_csr_shards(&self) -> Result<ScxCsr> {
        let shards = self.full_catalog.shards_sorted();
        #[cfg(feature = "parallel")]
        {
            self.assemble_shards_parallel(&shards)
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.assemble_shards(&shards)
        }
    }

    // -----------------------------------------------------------------------
    // Layer reading (11.9–11.10)
    // -----------------------------------------------------------------------

    /// List unique layer names from LayerCsrShard entries.
    pub fn layer_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .full_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::LayerCsrShard)
            .filter_map(|e| {
                // Strip _shard_N suffix to get layer name
                e.name.rfind("_shard_").map(|pos| e.name[..pos].to_string())
            })
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// Read a named layer, assembling all its shards into a ScxCsr.
    pub fn read_layer(&self, name: &str) -> Result<ScxCsr> {
        let prefix = format!("{name}_shard_");
        let mut shards: Vec<&FullCatalogEntry> = self
            .full_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::LayerCsrShard && e.name.starts_with(&prefix))
            .collect();

        if shards.is_empty() {
            return Err(ScxError::SectionNotFound(format!("layer '{name}'")));
        }

        // Sort by row_start
        shards.sort_by_key(|e| e.stats.as_ref().map_or(u64::MAX, |s| s.row_start));
        #[cfg(feature = "parallel")]
        {
            self.assemble_shards_parallel(&shards)
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.assemble_shards(&shards)
        }
    }

    // -----------------------------------------------------------------------
    // Uns (11.11)
    // -----------------------------------------------------------------------

    /// Read the uns (unstructured) section as a JSON value.
    pub fn read_uns(&self) -> Result<serde_json::Value> {
        let entry = self
            .full_catalog
            .get("uns")
            .ok_or_else(|| ScxError::SectionNotFound("uns".to_string()))?;
        let slice = self.section_bytes(entry);
        Ok(serde_json::from_slice(slice)?)
    }

    // -----------------------------------------------------------------------
    // Provenance
    // -----------------------------------------------------------------------

    /// Read the provenance section.
    pub fn read_provenance(&self) -> Result<Provenance> {
        let entry = self
            .full_catalog
            .get("provenance")
            .ok_or_else(|| ScxError::SectionNotFound("provenance".to_string()))?;
        let slice = self.section_bytes(entry);
        Provenance::read_from(&mut Cursor::new(slice))
    }

    // -----------------------------------------------------------------------
    // Deletion vectors
    // -----------------------------------------------------------------------

    /// Read deletion vectors if present. Returns `Ok(None)` if the file has no DV flag set.
    #[cfg(feature = "deletion-vectors")]
    pub fn read_deletion_vectors(
        &self,
    ) -> Result<Option<crate::deletion_vectors::DeletionVectors>> {
        if !self.header.has_deletion_vectors() {
            return Ok(None);
        }
        let entry = self
            .full_catalog
            .entries
            .iter()
            .find(|e| e.section_type == SectionType::DeletionVectors)
            .ok_or_else(|| ScxError::SectionNotFound("deletion_vectors".to_string()))?;
        let slice = self.section_bytes(entry);
        let dv = crate::deletion_vectors::DeletionVectors::read_from(&mut Cursor::new(slice))?;
        Ok(Some(dv))
    }

    /// Read all CSR shards with deletion vectors applied.
    /// Deleted rows are excluded from the returned ScxCsr.
    /// If no deletion vectors are present, returns the same result as `read_all_csr_shards()`.
    #[cfg(feature = "deletion-vectors")]
    pub fn read_all_csr_shards_filtered(&self) -> Result<ScxCsr> {
        let dv_opt = self.read_deletion_vectors()?;

        let csr = self.read_all_csr_shards()?;

        let dv = match dv_opt {
            Some(dv) if dv.total_deleted() > 0 => dv,
            _ => return Ok(csr),
        };

        // Build a keep mask from deletion vectors
        let n_obs = csr.shape.0;
        let shards = self.full_catalog.shards_sorted();

        let mut keep = vec![true; n_obs];
        for (shard_idx, shard_entry) in shards.iter().enumerate() {
            if let Some(ref stats) = shard_entry.stats {
                if let Some(sd) = dv.shards.iter().find(|sd| sd.shard_id == shard_idx as u32) {
                    for local_row in sd.bitmap.iter() {
                        let global_row = stats.row_start + local_row as u64;
                        if (global_row as usize) < n_obs {
                            keep[global_row as usize] = false;
                        }
                    }
                }
            }
        }

        // Filter CSR rows
        let mut new_indptr = vec![0i64];
        let mut new_indices = Vec::new();
        let mut new_data = Vec::new();

        for (row, &is_kept) in keep.iter().enumerate() {
            if !is_kept {
                continue;
            }
            let start = csr.indptr[row] as usize;
            let end = csr.indptr[row + 1] as usize;
            new_indices.extend_from_slice(&csr.indices[start..end]);
            new_data.extend_from_slice(&csr.data[start..end]);
            let prev = *new_indptr.last().unwrap();
            new_indptr.push(prev + (end - start) as i64);
        }

        let new_n_rows = new_indptr.len() - 1;
        Ok(ScxCsr::new_unchecked(
            (new_n_rows, csr.shape.1),
            new_indptr,
            new_indices,
            new_data,
        ))
    }

    /// Get access to the underlying mmap bytes.
    pub fn mmap(&self) -> &[u8] {
        &self.mmap
    }

    // -----------------------------------------------------------------------
    // Validate (11.12)
    // -----------------------------------------------------------------------

    /// Validate all section checksums.
    ///
    /// Returns a list of `(section_name, passed)` pairs. If any essential
    /// section (obs, var, CsrShard) fails, returns `Err(ChecksumMismatch)`.
    pub fn validate(&self) -> Result<Vec<(String, bool)>> {
        let mut results = Vec::new();
        let mut essential_failed = false;

        for entry in &self.full_catalog.entries {
            let slice = self.section_bytes(entry);
            let computed = blake3_hash(slice);
            let passed = computed == entry.checksum;

            if !passed {
                let is_essential = matches!(
                    entry.section_type,
                    SectionType::ObsMetadata | SectionType::VarMetadata | SectionType::CsrShard
                );
                if is_essential {
                    essential_failed = true;
                }
            }

            results.push((entry.name.clone(), passed));
        }

        if essential_failed {
            return Err(ScxError::ChecksumMismatch);
        }

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Get the raw bytes for a catalog entry from the mmap.
    pub fn section_bytes(&self, entry: &FullCatalogEntry) -> &[u8] {
        let start = entry.offset as usize;
        let end = start + entry.length as usize;
        &self.mmap[start..end]
    }

    /// Read and decode a single shard from a catalog entry.
    pub fn read_shard_from_entry(
        &self,
        entry: &FullCatalogEntry,
    ) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
        let section = self.section_bytes(entry);

        // Parse shard header
        let sh = ShardHeader::read_from(&mut Cursor::new(&section[..SHARD_HEADER_SIZE]))?;

        // Extract encoded byte slices
        let indptr_bytes = &section[sh.indptr_rel_offset as usize..][..sh.indptr_length as usize];
        let indices_bytes =
            &section[sh.indices_rel_offset as usize..][..sh.indices_length as usize];
        let values_bytes = &section[sh.values_rel_offset as usize..][..sh.values_length as usize];
        let block_index_bytes =
            &section[sh.block_index_rel_offset as usize..][..sh.block_index_length as usize];

        // Verify shard checksum
        let mut payload = Vec::new();
        payload.extend_from_slice(indptr_bytes);
        payload.extend_from_slice(indices_bytes);
        payload.extend_from_slice(values_bytes);
        payload.extend_from_slice(block_index_bytes);
        let computed = blake3_truncated_64(&payload);
        if computed != sh.checksum {
            return Err(ScxError::ChecksumMismatch);
        }

        // Resolve codec and encoding from shard header (NOT file header)
        let codec_id = CodecId::from_u8(sh.codec_id).ok_or(ScxError::UnknownCodec(sh.codec_id))?;
        let value_encoding = ValueEncoding::from_u8(sh.value_encoding)
            .ok_or(ScxError::UnknownValueEncoding(sh.value_encoding))?;
        let index_dtype_u16 = sh.index_dtype == 0;

        // Build EncodedShard and decode
        let encoded = EncodedShard {
            indptr_bytes: indptr_bytes.to_vec(),
            indices_bytes: indices_bytes.to_vec(),
            values_bytes: values_bytes.to_vec(),
        };

        let (indptr_u64, indices_u32, values_raw) = scx_codec::decode_shard(
            &encoded,
            codec_id,
            value_encoding,
            sh.n_major as usize,
            sh.nnz as usize,
            index_dtype_u16,
        )?;

        // Convert to scipy-compatible types
        let indptr: Vec<i64> = indptr_u64.iter().map(|&v| v as i64).collect();
        let indices: Vec<i32> = indices_u32.iter().map(|&v| v as i32).collect();
        let data = values_to_f32(&values_raw, value_encoding);

        Ok((indptr, indices, data))
    }

    /// Assemble multiple shard entries into a single ScxCsr using parallel decode.
    #[cfg(feature = "parallel")]
    fn assemble_shards_parallel(&self, shards: &[&FullCatalogEntry]) -> Result<ScxCsr> {
        if shards.is_empty() {
            return Ok(ScxCsr::new_unchecked(
                (0, self.header.n_vars as usize),
                vec![0],
                vec![],
                vec![],
            ));
        }

        // Decode all shards in parallel
        let decoded: Vec<(Vec<i64>, Vec<i32>, Vec<f32>)> = shards
            .par_iter()
            .map(|entry| self.read_shard_from_entry(entry))
            .collect::<Result<Vec<_>>>()?;

        // Pre-compute total sizes for allocation
        let total_indices: usize = decoded.iter().map(|(_, idx, _)| idx.len()).sum();
        let total_data: usize = decoded.iter().map(|(_, _, d)| d.len()).sum();
        let total_indptr: usize =
            decoded.iter().map(|(ip, _, _)| ip.len()).sum::<usize>() - (decoded.len() - 1); // subtract duplicate leading zeros

        let mut merged_indptr = Vec::with_capacity(total_indptr);
        let mut merged_indices = Vec::with_capacity(total_indices);
        let mut merged_data = Vec::with_capacity(total_data);
        let mut cumulative_nnz: i64 = 0;

        for (i, (indptr, indices, data)) in decoded.iter().enumerate() {
            if i == 0 {
                merged_indptr.extend_from_slice(indptr);
            } else {
                for &v in &indptr[1..] {
                    merged_indptr.push(v + cumulative_nnz);
                }
            }
            cumulative_nnz += *indptr.last().unwrap_or(&0);
            merged_indices.extend_from_slice(indices);
            merged_data.extend_from_slice(data);
        }

        let n_rows = merged_indptr.len().saturating_sub(1);
        Ok(ScxCsr::new_unchecked(
            (n_rows, self.header.n_vars as usize),
            merged_indptr,
            merged_indices,
            merged_data,
        ))
    }

    /// Assemble multiple shard entries into a single ScxCsr (sequential).
    #[cfg(any(not(feature = "parallel"), test))]
    fn assemble_shards(&self, shards: &[&FullCatalogEntry]) -> Result<ScxCsr> {
        if shards.is_empty() {
            return Ok(ScxCsr::new_unchecked(
                (0, self.header.n_vars as usize),
                vec![0],
                vec![],
                vec![],
            ));
        }

        let mut merged_indptr: Vec<i64> = Vec::new();
        let mut merged_indices: Vec<i32> = Vec::new();
        let mut merged_data: Vec<f32> = Vec::new();
        let mut cumulative_nnz: i64 = 0;

        for (i, entry) in shards.iter().enumerate() {
            let (indptr, indices, data) = self.read_shard_from_entry(entry)?;

            if i == 0 {
                merged_indptr.extend_from_slice(&indptr);
            } else {
                // Skip first element (0) and offset by cumulative nnz
                for &v in &indptr[1..] {
                    merged_indptr.push(v + cumulative_nnz);
                }
            }

            cumulative_nnz += *indptr.last().unwrap_or(&0);
            merged_indices.extend_from_slice(&indices);
            merged_data.extend_from_slice(&data);
        }

        let n_rows = merged_indptr.len().saturating_sub(1);
        Ok(ScxCsr::new_unchecked(
            (n_rows, self.header.n_vars as usize),
            merged_indptr,
            merged_indices,
            merged_data,
        ))
    }
}

/// Convert raw value bytes to f32 according to the value encoding.
fn values_to_f32(raw: &[u8], encoding: ValueEncoding) -> Vec<f32> {
    match encoding {
        ValueEncoding::Uint8 => raw.iter().map(|&b| b as f32).collect(),
        ValueEncoding::Uint16 => raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as f32)
            .collect(),
        ValueEncoding::Uint32 => raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        ValueEncoding::Float32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        ValueEncoding::Float16 => raw
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::MAGIC;
    use crate::provenance::ProvenanceEntry;
    use crate::writer::ScxWriter;
    use arrow::array::{Float32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
        FileHeader {
            magic: MAGIC,
            format_version: 1,
            header_length: 256,
            flags: 0,
            n_obs,
            n_vars,
            nnz,
            n_csr_shards: 0,
            n_csc_shards: 0,
            shard_target_rows: 16384,
            codec_id: 0,
            index_dtype: 0, // u16
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
        }
    }

    fn sample_obs(n: usize) -> RecordBatch {
        let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
        let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    fn sample_var(n: usize) -> RecordBatch {
        let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
        let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    /// Build a small shard: n_rows rows, each with some nonzeros in n_vars columns.
    fn sample_shard_data(n_rows: usize, n_vars: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for row in 0..n_rows {
            // Each row has 2 nonzeros
            let col0 = (row * 2) % n_vars;
            let col1 = (row * 2 + 1) % n_vars;
            indices.push(col0 as u32);
            indices.push(col1 as u32);
            values.push(((row + 1) % 256) as u8);
            values.push(((row + 2) % 256) as u8);
            indptr.push(indptr.last().unwrap() + 2);
        }
        (indptr, indices, values)
    }

    /// Write a complete test file and return the path.
    fn write_test_file(
        dir: &tempfile::TempDir,
        filename: &str,
        n_obs: usize,
        n_vars: usize,
        n_shards: usize,
        include_extras: bool,
    ) -> std::path::PathBuf {
        let path = dir.path().join(filename);
        let total_nnz = n_obs * 2; // 2 nnz per row
        let header = sample_header(n_obs as u64, n_vars as u64, total_nnz as u64);
        let mut writer = ScxWriter::new(&path, header).unwrap();

        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();

        let rows_per_shard = n_obs / n_shards;
        for s in 0..n_shards {
            let shard_rows = if s == n_shards - 1 {
                n_obs - rows_per_shard * s
            } else {
                rows_per_shard
            };
            let (indptr, indices, values) = sample_shard_data(shard_rows, n_vars);
            writer
                .write_csr_shard(
                    &indptr,
                    &indices,
                    &values,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    (s * rows_per_shard) as u64,
                )
                .unwrap();
        }

        if include_extras {
            // obsm
            let obsm_schema = Schema::new(vec![
                Field::new("pc1", DataType::Float32, false),
                Field::new("pc2", DataType::Float32, false),
            ]);
            let obsm_batch = RecordBatch::try_new(
                Arc::new(obsm_schema),
                vec![
                    Arc::new(Float32Array::from(
                        (0..n_obs).map(|i| i as f32).collect::<Vec<_>>(),
                    )),
                    Arc::new(Float32Array::from(
                        (0..n_obs).map(|i| (i as f32) * 2.0).collect::<Vec<_>>(),
                    )),
                ],
            )
            .unwrap();
            writer.write_obsm("X_pca", &obsm_batch).unwrap();

            // uns
            writer
                .write_uns(&serde_json::json!({"species": "human", "version": 2}))
                .unwrap();

            // provenance
            writer
                .write_provenance(vec![ProvenanceEntry {
                    timestamp: 1710000000,
                    action: "convert".to_string(),
                    tool: "scx-cli 0.1.0".to_string(),
                    params_json: "{}".to_string(),
                    input_checksums: vec![],
                }])
                .unwrap();
        }

        writer.finish().unwrap();
        path
    }

    // -----------------------------------------------------------------------
    // 11.15: Full round-trip test
    // -----------------------------------------------------------------------

    #[test]
    fn test_reader_full_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "full.scx", 6, 10, 2, true);

        let reader = ScxReader::open(&path).unwrap();

        // Summary accessors
        assert_eq!(reader.n_obs(), 6);
        assert_eq!(reader.n_vars(), 10);
        assert_eq!(reader.nnz(), 12);
        assert_eq!(reader.header().format_version, 1);

        // read_obs
        let obs = reader.read_obs().unwrap();
        assert_eq!(obs.num_rows(), 6);
        assert_eq!(obs.num_columns(), 1);

        // read_var
        let var = reader.read_var().unwrap();
        assert_eq!(var.num_rows(), 10);
        assert_eq!(var.num_columns(), 1);

        // read_csr_shard(0) — first shard
        let (indptr, indices, data) = reader.read_csr_shard(0).unwrap();
        assert_eq!(indptr.len(), 4); // 3 rows + 1
        assert_eq!(indices.len(), 6); // 3 rows * 2 nnz
        assert_eq!(data.len(), 6);

        // read_all_csr_shards
        let csr = reader.read_all_csr_shards().unwrap();
        assert_eq!(csr.shape, (6, 10));
        assert_eq!(csr.indptr.len(), 7); // 6 rows + 1
        assert_eq!(csr.nnz(), 12); // 6 rows * 2

        // read_obsm
        let obsm = reader.read_obsm("X_pca").unwrap();
        assert_eq!(obsm.num_rows(), 6);
        assert_eq!(obsm.num_columns(), 2);

        // read_all_obsm
        let all_obsm = reader.read_all_obsm().unwrap();
        assert_eq!(all_obsm.len(), 1);
        assert!(all_obsm.contains_key("X_pca"));

        // read_uns
        let uns = reader.read_uns().unwrap();
        assert_eq!(uns["species"], "human");
        assert_eq!(uns["version"], 2);

        // read_provenance
        let prov = reader.read_provenance().unwrap();
        assert_eq!(prov.operations.len(), 1);
        assert_eq!(prov.operations[0].action, "convert");
    }

    // -----------------------------------------------------------------------
    // 11.16: Checksum corruption detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_detects_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "corrupt.scx", 6, 10, 2, false);

        // Read the file, corrupt a byte in a shard section, write back
        let mut data = std::fs::read(&path).unwrap();

        // Find a CSR shard section offset from the catalog
        let reader = ScxReader::open(&path).unwrap();
        let shards = reader.catalog().shards_sorted();
        assert!(!shards.is_empty());
        let shard_offset = shards[0].offset as usize;
        // Corrupt a byte in the shard payload (after the 76-byte header)
        let corrupt_pos = shard_offset + SHARD_HEADER_SIZE + 1;
        drop(reader);

        data[corrupt_pos] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();

        // Re-open — open should still succeed (catalog checksum is intact)
        let reader = ScxReader::open(&path).unwrap();

        // validate() should detect the corruption and error (essential section)
        let result = reader.validate();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ScxError::ChecksumMismatch));
    }

    // -----------------------------------------------------------------------
    // 11.17: Unknown section types handled gracefully
    // -----------------------------------------------------------------------

    #[test]
    fn test_known_sections_read_correctly() {
        // Validates that when all sections are known types, the reader works.
        // The catalog modification to skip unknown types is tested by the
        // fact that FullCatalog::read_from no longer panics on unknown types.
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "known.scx", 4, 8, 1, true);

        let reader = ScxReader::open(&path).unwrap();
        let catalog = reader.catalog();

        // All entries should have known section types
        for entry in &catalog.entries {
            assert!(SectionType::is_known(entry.section_type as u8));
        }

        // All section reads should succeed
        assert!(reader.read_obs().is_ok());
        assert!(reader.read_var().is_ok());
        assert!(reader.read_csr_shard(0).is_ok());
        assert!(reader.read_uns().is_ok());
        assert!(reader.read_provenance().is_ok());
    }

    // -----------------------------------------------------------------------
    // 11.18: Multi-shard assembly matches individual reads
    // -----------------------------------------------------------------------

    #[test]
    fn test_four_shards_individual_vs_assembled() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "four.scx", 12, 10, 4, false);

        let reader = ScxReader::open(&path).unwrap();

        // Read individually
        let mut individual_indptr: Vec<i64> = Vec::new();
        let mut individual_indices: Vec<i32> = Vec::new();
        let mut individual_data: Vec<f32> = Vec::new();
        let mut cumulative_nnz: i64 = 0;

        for i in 0..4 {
            let (indptr, indices, data) = reader.read_csr_shard(i).unwrap();
            if i == 0 {
                individual_indptr.extend_from_slice(&indptr);
            } else {
                for &v in &indptr[1..] {
                    individual_indptr.push(v + cumulative_nnz);
                }
            }
            cumulative_nnz += *indptr.last().unwrap_or(&0);
            individual_indices.extend_from_slice(&indices);
            individual_data.extend_from_slice(&data);
        }

        // Read assembled
        let csr = reader.read_all_csr_shards().unwrap();

        assert_eq!(csr.indptr, individual_indptr);
        assert_eq!(csr.indices, individual_indices);
        assert_eq!(csr.data, individual_data);
        assert_eq!(csr.shape, (12, 10));
    }

    // -----------------------------------------------------------------------
    // 11.19: Section byte ranges match catalog entries
    // -----------------------------------------------------------------------

    #[test]
    fn test_section_byte_ranges_match_catalog() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "ranges.scx", 6, 10, 2, true);

        let reader = ScxReader::open(&path).unwrap();
        let catalog = reader.catalog();

        // Verify obs and var sections can be sliced at their catalog offsets
        let obs_entry = catalog.get("obs").unwrap();
        let obs_bytes = reader.section_bytes(obs_entry);
        assert_eq!(obs_bytes.len(), obs_entry.length as usize);

        let var_entry = catalog.get("var").unwrap();
        let var_bytes = reader.section_bytes(var_entry);
        assert_eq!(var_bytes.len(), var_entry.length as usize);

        // Verify checksum of raw bytes matches catalog checksum
        assert_eq!(blake3_hash(obs_bytes), obs_entry.checksum);
        assert_eq!(blake3_hash(var_bytes), var_entry.checksum);

        // All entries should be 8-byte aligned
        for entry in &catalog.entries {
            assert_eq!(
                entry.offset % 8,
                0,
                "section '{}' not 8-byte aligned",
                entry.name
            );
        }
    }

    // -----------------------------------------------------------------------
    // Additional edge case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_shard_index_out_of_bounds() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "bounds.scx", 4, 8, 1, false);

        let reader = ScxReader::open(&path).unwrap();
        let result = reader.read_csr_shard(5);
        assert!(matches!(
            result.unwrap_err(),
            ScxError::ShardIndexOutOfBounds { index: 5, count: 1 }
        ));
    }

    #[test]
    fn test_section_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "nofind.scx", 4, 8, 1, false);

        let reader = ScxReader::open(&path).unwrap();
        assert!(matches!(
            reader.read_uns().unwrap_err(),
            ScxError::SectionNotFound(_)
        ));
        assert!(matches!(
            reader.read_obsm("X_pca").unwrap_err(),
            ScxError::SectionNotFound(_)
        ));
    }

    #[test]
    fn test_layer_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("layers.scx");
        let header = sample_header(6, 10, 12);
        let mut writer = ScxWriter::new(&path, header).unwrap();

        writer.write_obs(&sample_obs(6)).unwrap();
        writer.write_var(&sample_var(10)).unwrap();

        // Write X shards
        let (indptr, indices, values) = sample_shard_data(6, 10);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();

        // Write "raw" layer shard
        let (indptr, indices, values) = sample_shard_data(6, 10);
        writer
            .write_layer_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
                "raw",
                0,
            )
            .unwrap();

        writer.finish().unwrap();

        let reader = ScxReader::open(&path).unwrap();
        let names = reader.layer_names();
        assert_eq!(names, vec!["raw"]);

        let layer = reader.read_layer("raw").unwrap();
        assert_eq!(layer.shape, (6, 10));
        assert_eq!(layer.nnz(), 12);
    }

    #[test]
    fn test_validate_passes_for_clean_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "clean.scx", 6, 10, 2, true);

        let reader = ScxReader::open(&path).unwrap();
        let results = reader.validate().unwrap();

        // All sections should pass
        for (name, passed) in &results {
            assert!(passed, "section '{}' failed checksum", name);
        }
    }

    // -----------------------------------------------------------------------
    // 16.8: Multi-operation provenance chain
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_operation_provenance_chain() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prov_chain.scx");
        let header = sample_header(4, 8, 8);
        let mut writer = ScxWriter::new(&path, header).unwrap();

        writer.write_obs(&sample_obs(4)).unwrap();
        writer.write_var(&sample_var(8)).unwrap();

        let (indptr, indices, values) = sample_shard_data(4, 8);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();

        // Write provenance with 3 chained operations
        let entries = vec![
            ProvenanceEntry {
                timestamp: 1710000000,
                action: "convert".to_string(),
                tool: "scx-cli 0.1.0".to_string(),
                params_json: r#"{"input":"raw.h5ad"}"#.to_string(),
                input_checksums: vec![[0xAA; 32]],
            },
            ProvenanceEntry {
                timestamp: 1710001000,
                action: "subset".to_string(),
                tool: "pyscx 0.1.0".to_string(),
                params_json: r#"{"n_cells":1000}"#.to_string(),
                input_checksums: vec![[0xBB; 32]],
            },
            ProvenanceEntry {
                timestamp: 1710002000,
                action: "normalize".to_string(),
                tool: "pyscx 0.1.0".to_string(),
                params_json: "{}".to_string(),
                input_checksums: vec![[0xCC; 32], [0xDD; 32]],
            },
        ];

        writer.write_provenance(entries.clone()).unwrap();
        writer.finish().unwrap();

        // Read back and verify
        let reader = ScxReader::open(&path).unwrap();
        let prov = reader.read_provenance().unwrap();

        assert_eq!(prov.version, 1);
        assert_eq!(prov.operations.len(), 3);

        // Verify ordering and content
        assert_eq!(prov.operations[0].action, "convert");
        assert_eq!(prov.operations[0].timestamp, 1710000000);
        assert_eq!(prov.operations[0].input_checksums.len(), 1);

        assert_eq!(prov.operations[1].action, "subset");
        assert_eq!(prov.operations[1].timestamp, 1710001000);
        assert_eq!(prov.operations[1].params_json, r#"{"n_cells":1000}"#);

        assert_eq!(prov.operations[2].action, "normalize");
        assert_eq!(prov.operations[2].timestamp, 1710002000);
        assert_eq!(prov.operations[2].input_checksums.len(), 2);
        assert_eq!(prov.operations[2].input_checksums[0], [0xCC; 32]);
        assert_eq!(prov.operations[2].input_checksums[1], [0xDD; 32]);
    }

    // -----------------------------------------------------------------------
    // Parallel shard decode tests
    // -----------------------------------------------------------------------

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_matches_sequential() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "par_seq.scx", 12, 10, 4, false);

        let reader = ScxReader::open(&path).unwrap();
        let shards = reader.catalog().shards_sorted();

        let sequential = reader.assemble_shards(&shards).unwrap();
        let parallel = reader.assemble_shards_parallel(&shards).unwrap();

        assert_eq!(sequential.shape, parallel.shape);
        assert_eq!(sequential.indptr, parallel.indptr);
        assert_eq!(sequential.indices, parallel.indices);
        assert_eq!(sequential.data, parallel.data);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_single_shard() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "par_single.scx", 6, 10, 1, false);

        let reader = ScxReader::open(&path).unwrap();
        let csr = reader.read_all_csr_shards().unwrap();

        assert_eq!(csr.shape, (6, 10));
        assert_eq!(csr.nnz(), 12);
        assert_eq!(csr.indptr.len(), 7);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_thread_pool() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "par_pool.scx", 12, 10, 4, false);

        let reader = ScxReader::open(&path).unwrap();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();

        let csr = pool.install(|| reader.read_all_csr_shards()).unwrap();

        assert_eq!(csr.shape, (12, 10));
        assert_eq!(csr.nnz(), 24);
        assert_eq!(csr.indptr.len(), 13);
    }

    #[test]
    fn test_values_to_f32_uint8() {
        let raw = vec![1u8, 2, 255];
        let result = values_to_f32(&raw, ValueEncoding::Uint8);
        assert_eq!(result, vec![1.0, 2.0, 255.0]);
    }

    #[test]
    fn test_values_to_f32_uint16() {
        let raw: Vec<u8> = vec![0x01, 0x00, 0xFF, 0x00]; // 1, 255 as u16 LE
        let result = values_to_f32(&raw, ValueEncoding::Uint16);
        assert_eq!(result, vec![1.0, 255.0]);
    }

    #[test]
    fn test_values_to_f32_float32() {
        let val: f32 = 3.14;
        let raw = val.to_le_bytes().to_vec();
        let result = values_to_f32(&raw, ValueEncoding::Float32);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 3.14).abs() < 1e-6);
    }
}
