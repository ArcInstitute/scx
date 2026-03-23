//! Cloud Reader: open SCX files directly from cloud storage.
//!
//! Supports three access patterns:
//!   - Exploded `.scxd/` directories → GET `_catalog.bin` + individual object reads
//!   - Cloud-ready packed `.scx` → range read first 256KB (header + front catalog)
//!   - Non-cloud-ready packed `.scx` → HEAD + range read header + range read catalog at EOF
//!
//! Implements SPEC §12 cloud access patterns.

use std::io::Cursor;
use std::sync::Arc;

use arrow::array::RecordBatch;
use object_store::path::Path as ObjPath;
use object_store::ObjectStore;

use scx_format::catalog::FullCatalog;
use scx_format::header::{FileHeader, HEADER_SIZE};

use crate::backend::CloudLocation;
use crate::error::{CloudError, Result};
use crate::explode::section_name_to_path;

/// Layout of the cloud reader source.
enum ReaderLayout {
    /// Exploded .scxd directory — each section is a separate object.
    Exploded(CloudLocation),
    /// Packed .scx file — sections are read via range reads.
    Packed(ObjPath),
}

/// A reader that can access SCX data from cloud object stores
/// without downloading the entire file.
pub struct CloudReader {
    backend: Arc<dyn ObjectStore>,
    layout: ReaderLayout,
    header: FileHeader,
    catalog: FullCatalog,
}

impl CloudReader {
    /// Number of observations (cells).
    pub fn n_obs(&self) -> u64 {
        self.header.n_obs
    }

    /// Number of variables (genes).
    pub fn n_vars(&self) -> u64 {
        self.header.n_vars
    }

    /// Total non-zero entries.
    pub fn nnz(&self) -> u64 {
        self.header.nnz
    }

    /// Number of CSR shards.
    pub fn n_shards(&self) -> u32 {
        self.header.n_csr_shards
    }

    /// Reference to the file header.
    pub fn header(&self) -> &FileHeader {
        &self.header
    }

    /// Reference to the full catalog.
    pub fn catalog(&self) -> &FullCatalog {
        &self.catalog
    }

    /// Read a specific section by name.
    pub async fn read_section(&self, name: &str) -> Result<Vec<u8>> {
        let entry = self
            .catalog
            .entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| {
                CloudError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("section not found: {name}"),
                ))
            })?;

        match &self.layout {
            ReaderLayout::Exploded(location) => {
                let make_path = crate::pull::build_path_fn(location);
                let rel_path = section_name_to_path(&entry.name, entry.section_type)
                    .map_err(|e| CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
                let obj_path = make_path(&rel_path);
                let data = self.backend.get(&obj_path).await?.bytes().await?;
                Ok(data.to_vec())
            }
            ReaderLayout::Packed(file_path) => {
                let end = entry.offset.checked_add(entry.length).ok_or_else(|| {
                    CloudError::SliceBoundsExceeded {
                        offset: entry.offset as usize,
                        length: entry.length as usize,
                        data_len: 0, // remote file; actual size unknown
                    }
                })?;
                let data = self.backend.get_range(file_path, entry.offset..end).await?;
                Ok(data.to_vec())
            }
        }
    }

    /// Read obs metadata as an Arrow RecordBatch.
    pub async fn read_obs(&self) -> Result<RecordBatch> {
        let obs_data = self.read_section("obs").await?;
        let cursor = Cursor::new(&obs_data);
        let reader = arrow::ipc::reader::FileReader::try_new(cursor, None).map_err(|e| {
            CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        })?;
        let batch = reader
            .into_iter()
            .next()
            .ok_or_else(|| {
                CloudError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "obs Arrow IPC contains no batches",
                ))
            })?
            .map_err(|e| CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
        Ok(batch)
    }

    /// Read var metadata as an Arrow RecordBatch.
    pub async fn read_var(&self) -> Result<RecordBatch> {
        let var_data = self.read_section("var").await?;
        let cursor = Cursor::new(&var_data);
        let reader = arrow::ipc::reader::FileReader::try_new(cursor, None).map_err(|e| {
            CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        })?;
        let batch = reader
            .into_iter()
            .next()
            .ok_or_else(|| {
                CloudError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "var Arrow IPC contains no batches",
                ))
            })?
            .map_err(|e| CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
        Ok(batch)
    }

    /// Read multiple shard sections in parallel.
    pub async fn read_shards(&self, shard_names: &[&str]) -> Result<Vec<Vec<u8>>> {
        let handles: Vec<_> = shard_names
            .iter()
            .map(|name| self.read_section(name))
            .collect();
        let results = futures::future::join_all(handles).await;
        results.into_iter().collect()
    }
}

/// Open an SCX file or directory from cloud/local storage.
///
/// Detects the layout automatically:
///   - If `_catalog.bin` is found → exploded directory
///   - Otherwise → packed file (auto-detects cloud-ready vs not)
pub async fn open_cloud(url: &str) -> Result<CloudReader> {
    let location = crate::backend::parse_location(url)?;
    let backend: Arc<dyn ObjectStore> =
        Arc::from(crate::backend::create_backend(&location).await?);

    // Try exploded layout first: look for _catalog.bin
    let catalog_path = {
        let f = crate::pull::build_path_fn(&location);
        f("_catalog.bin")
    };

    match backend.get(&catalog_path).await {
        Ok(get_result) => {
            // Exploded directory
            let catalog_bytes = get_result.bytes().await?.to_vec();
            let catalog = FullCatalog::read_from(
                &mut Cursor::new(&catalog_bytes),
                catalog_bytes.len(),
            )?;

            let header_path = {
                let f = crate::pull::build_path_fn(&location);
                f("_header.bin")
            };
            let header_data = backend.get(&header_path).await?.bytes().await?;
            let header = FileHeader::read_from(&mut Cursor::new(&header_data))?;

            Ok(CloudReader {
                backend,
                layout: ReaderLayout::Exploded(location),
                header,
                catalog,
            })
        }
        Err(e) => {
            // Only fall through to packed-file path for NotFound errors.
            // Auth, network, and other errors should propagate immediately.
            if !matches!(e, object_store::Error::NotFound { .. }) {
                return Err(CloudError::ObjectStore(e));
            }

            // Packed file — build the object path for range reads
            let file_path = match &location {
                CloudLocation::Local(p) => {
                    // LocalFileSystem is rooted at the parent directory,
                    // so the object path is just the filename.
                    let name = p
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    ObjPath::from(name.as_str())
                }
                CloudLocation::Gcs { prefix, .. }
                | CloudLocation::S3 { prefix, .. }
                | CloudLocation::Azure { prefix, .. } => ObjPath::from(prefix.as_str()),
            };

            let first_chunk_size = (HEADER_SIZE + 4096) as u64;
            let data = backend
                .get_range(&file_path, 0..first_chunk_size)
                .await?;
            let first_bytes = data.to_vec();

            if first_bytes.len() < HEADER_SIZE {
                return Err(CloudError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "file too small to contain SCX header",
                )));
            }

            let header =
                FileHeader::read_from(&mut Cursor::new(&first_bytes[..HEADER_SIZE]))?;

            if header.has_front_catalog()
                && header.front_catalog_offset > 0
                && header.front_catalog_length > 0
            {
                // Cloud-ready: front catalog is at start of file
                let fc_offset = header.front_catalog_offset;
                let fc_end = fc_offset.checked_add(header.front_catalog_length).ok_or_else(|| {
                    CloudError::SliceBoundsExceeded {
                        offset: fc_offset as usize,
                        length: header.front_catalog_length as usize,
                        data_len: 0,
                    }
                })?;

                let fc_bytes = if (fc_end as usize) <= first_bytes.len() {
                    first_bytes[fc_offset as usize..fc_end as usize].to_vec()
                } else {
                    backend
                        .get_range(&file_path, fc_offset..fc_end)
                        .await?
                        .to_vec()
                };

                let catalog = FullCatalog::read_from(
                    &mut Cursor::new(&fc_bytes),
                    fc_bytes.len(),
                )?;

                Ok(CloudReader {
                    backend,
                    layout: ReaderLayout::Packed(file_path),
                    header,
                    catalog,
                })
            } else {
                // Not cloud-ready: read full catalog at EOF
                let fc_offset = header.full_catalog_offset;
                let fc_end = fc_offset.checked_add(header.full_catalog_length).ok_or_else(|| {
                    CloudError::SliceBoundsExceeded {
                        offset: fc_offset as usize,
                        length: header.full_catalog_length as usize,
                        data_len: 0,
                    }
                })?;

                let fc_bytes = backend
                    .get_range(&file_path, fc_offset..fc_end)
                    .await?
                    .to_vec();

                let catalog = FullCatalog::read_from(
                    &mut Cursor::new(&fc_bytes),
                    fc_bytes.len(),
                )?;

                Ok(CloudReader {
                    backend,
                    layout: ReaderLayout::Packed(file_path),
                    header,
                    catalog,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use scx_codec::{CodecId, ValueEncoding};
    use scx_format::header::MAGIC;
    use scx_format::section::SectionType;
    use scx_format::writer::ScxWriter;
    use std::sync::Arc as StdArc;

    fn sample_header(n_obs: u64, n_vars: u64) -> FileHeader {
        FileHeader {
            magic: MAGIC,
            format_version: 1,
            header_length: 256,
            flags: 0,
            n_obs,
            n_vars,
            nnz: 0,
            n_csr_shards: 0,
            n_csc_shards: 0,
            shard_target_rows: 16384,
            codec_id: 0,
            index_dtype: 0,
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
        let types: Vec<String> = (0..n)
            .map(|i| match i % 3 {
                0 => "T_cell".to_string(),
                1 => "B_cell".to_string(),
                _ => "Monocyte".to_string(),
            })
            .collect();
        let schema = Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("cell_type", DataType::Utf8, false),
        ]);
        RecordBatch::try_new(
            StdArc::new(schema),
            vec![
                StdArc::new(StringArray::from(
                    ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                )),
                StdArc::new(StringArray::from(
                    types.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    }

    fn sample_var(n: usize) -> RecordBatch {
        let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
        let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
        RecordBatch::try_new(
            StdArc::new(schema),
            vec![StdArc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    fn sample_shard_data(n_rows: usize, n_vars: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for row in 0..n_rows {
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

    fn write_test_file(
        dir: &tempfile::TempDir,
        n_obs: usize,
        n_vars: usize,
    ) -> std::path::PathBuf {
        let path = dir.path().join("test.scx");
        let header = sample_header(n_obs as u64, n_vars as u64);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();

        let rows_per_shard = 50;
        let mut row_offset = 0;
        while row_offset < n_obs {
            let shard_rows = std::cmp::min(rows_per_shard, n_obs - row_offset);
            let (indptr, indices, values) = sample_shard_data(shard_rows, n_vars);
            writer
                .write_csr_shard(
                    &indptr,
                    &indices,
                    &values,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    row_offset as u64,
                )
                .unwrap();
            row_offset += shard_rows;
        }
        writer.finish().unwrap();
        path
    }

    #[tokio::test]
    async fn test_open_exploded_directory() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);
        let exploded = dir.path().join("test.scxd");
        crate::explode::explode(&input, &exploded).unwrap();

        let reader = open_cloud(&exploded.to_string_lossy()).await.unwrap();
        assert_eq!(reader.n_obs(), 100);
        assert_eq!(reader.n_vars(), 50);
    }

    #[tokio::test]
    async fn test_open_cloud_ready_packed_file() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);
        let optimized = dir.path().join("cloud_ready.scx");
        crate::cloud_optimize::cloud_optimize(&input, &optimized).unwrap();

        let reader = open_cloud(&optimized.to_string_lossy()).await.unwrap();
        assert_eq!(reader.n_obs(), 100);
        assert_eq!(reader.n_vars(), 50);
    }

    #[tokio::test]
    async fn test_open_non_cloud_ready_packed_file() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);

        let reader = open_cloud(&input.to_string_lossy()).await.unwrap();
        assert_eq!(reader.n_obs(), 100);
        assert_eq!(reader.n_vars(), 50);
    }

    #[tokio::test]
    async fn test_read_obs_from_exploded() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);
        let exploded = dir.path().join("test.scxd");
        crate::explode::explode(&input, &exploded).unwrap();

        let reader = open_cloud(&exploded.to_string_lossy()).await.unwrap();
        let obs = reader.read_obs().await.unwrap();
        assert_eq!(obs.num_rows(), 100);
        assert_eq!(obs.num_columns(), 2);

        let cell_type = obs
            .column_by_name("cell_type")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(cell_type.value(0), "T_cell");
    }

    #[tokio::test]
    async fn test_read_var_from_exploded() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);
        let exploded = dir.path().join("test.scxd");
        crate::explode::explode(&input, &exploded).unwrap();

        let reader = open_cloud(&exploded.to_string_lossy()).await.unwrap();
        let var = reader.read_var().await.unwrap();
        assert_eq!(var.num_rows(), 50);
    }

    #[tokio::test]
    async fn test_read_shards_from_exploded() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);
        let exploded = dir.path().join("test.scxd");
        crate::explode::explode(&input, &exploded).unwrap();

        let reader = open_cloud(&exploded.to_string_lossy()).await.unwrap();

        let shard_names: Vec<String> = reader
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CsrShard)
            .map(|e| e.name.clone())
            .collect();

        assert_eq!(shard_names.len(), 2);

        let shard_refs: Vec<&str> = shard_names.iter().map(|s| s.as_str()).collect();
        let shard_data = reader.read_shards(&shard_refs).await.unwrap();
        assert_eq!(shard_data.len(), 2);
        assert!(!shard_data[0].is_empty());
        assert!(!shard_data[1].is_empty());
    }
}
