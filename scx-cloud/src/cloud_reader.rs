//! Cloud Reader: open SCX files directly from cloud storage.
//!
//! Supports three access patterns:
//!   - Exploded `.scxd/` directories → GET `_catalog.bin` + individual object reads
//!   - Cloud-ready packed `.scx` → range read first 256KB (header + front catalog)
//!   - Non-cloud-ready packed `.scx` → HEAD + range read header + range read catalog at EOF
//!
//! Implements docs/cloud.md cloud access patterns.

use std::io::Cursor;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use futures::stream::{StreamExt, TryStreamExt};
use object_store::path::Path as ObjPath;
use object_store::ObjectStore;

use scx_format::catalog::{FullCatalog, FullCatalogEntry};
use scx_format::header::{FileHeader, HEADER_SIZE};
use scx_format::section::SectionType;

use crate::backend::CloudLocation;
use crate::error::{CloudError, Result};
use crate::explode::section_name_to_path;

/// Maximum number of metadata-shard range reads issued concurrently when
/// assembling sharded obs/var. Caps inflight requests so atlas-scale files
/// (thousands of `ObsMetadataShard` / `VarMetadataShard` sections) don't
/// fire an unbounded number of simultaneous fetches. Matches the default
/// `PullOptions::parallelism`; a configurable `CloudQueryOptions` is a
/// deferred follow-on (see docs/cloud.md).
const METADATA_SHARD_FETCH_CONCURRENCY: usize = 8;

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
    /// One-shot caches for the obs and var section bytes. These
    /// sections are read twice per `QueryPipeline`: once for schema
    /// (eager validation in `from_reader`) and once for the batch
    /// itself (`collect`). On census-scale data each is O(100 MB);
    /// caching elides the duplicate range read. Other sections
    /// (predicate indexes, shards, catalog) are single-fetch per
    /// query and are not cached.
    obs_bytes_cache: tokio::sync::OnceCell<Vec<u8>>,
    var_bytes_cache: tokio::sync::OnceCell<Vec<u8>>,
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

    /// Number of [`SectionType::ObsMetadataShard`] sections — non-zero
    /// only for Phase 2 sharded-obs files. Mirror of
    /// `ScxReader::obs_metadata_shard_count`.
    pub fn obs_metadata_shard_count(&self) -> usize {
        self.catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::ObsMetadataShard)
            .count()
    }

    /// Number of [`SectionType::VarMetadataShard`] sections. Mirror of
    /// [`Self::obs_metadata_shard_count`].
    pub fn var_metadata_shard_count(&self) -> usize {
        self.catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::VarMetadataShard)
            .count()
    }

    /// Reference to the file header.
    pub fn header(&self) -> &FileHeader {
        &self.header
    }

    /// Reference to the full catalog.
    pub fn catalog(&self) -> &FullCatalog {
        &self.catalog
    }

    /// Read the obs/var section bytes, caching the first fetch.
    ///
    /// Routes through `read_section(name)` on first call and stores
    /// the result in the per-section `OnceCell`. Subsequent calls
    /// return the cached bytes without touching the network. Used for
    /// obs/var only; other sections call `read_section` directly.
    async fn read_metadata_section(&self, name: &str) -> Result<Vec<u8>> {
        let cell = match name {
            "obs" => &self.obs_bytes_cache,
            "var" => &self.var_bytes_cache,
            _ => return self.read_section(name).await,
        };
        cell.get_or_try_init(|| async { self.read_section(name).await })
            .await
            .cloned()
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
                let rel_path =
                    section_name_to_path(&entry.name, entry.section_type).map_err(|e| {
                        CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
                    })?;
                let obj_path = make_path(&rel_path);
                let data = self.backend.get(&obj_path).await?.bytes().await?;
                Ok(data.to_vec())
            }
            ReaderLayout::Packed(file_path) => {
                let end = entry.offset.checked_add(entry.length).ok_or_else(|| {
                    CloudError::CatalogOffsetOverflow {
                        name: entry.name.clone(),
                        offset: entry.offset,
                        length: entry.length,
                    }
                })?;
                let data = self.backend.get_range(file_path, entry.offset..end).await?;
                Ok(data.to_vec())
            }
        }
    }

    /// Read obs metadata as an Arrow RecordBatch.
    ///
    /// Transparently handles both layouts: assembles every
    /// [`SectionType::ObsMetadataShard`] section in `shard_idx` order on
    /// Phase 2 sharded files (parallel range reads → shared
    /// [`scx_format::assemble_sharded_metadata`]), or reads the single
    /// legacy [`SectionType::ObsMetadata`] section otherwise. Bypasses
    /// `ScxReader::read_arrow_ipc`, so applies
    /// `scx_format::downcast_large_types` explicitly to surface the
    /// canonical narrow `Utf8` / `Binary` types regardless of the
    /// on-disk encoding.
    pub async fn read_obs(&self) -> Result<RecordBatch> {
        if self.obs_metadata_shard_count() > 0 {
            return self
                .read_sharded_metadata(
                    SectionType::ObsMetadataShard,
                    "obs_metadata/shard_",
                    "obs_metadata",
                )
                .await;
        }
        let obs_data = self.read_metadata_section("obs").await?;
        Ok(scx_format::downcast_large_types(&decode_arrow_ipc_batch(
            &obs_data, "obs",
        )?)?)
    }

    /// Read var metadata as an Arrow RecordBatch. Mirror of
    /// [`Self::read_obs`] for the var axis — same dual-layout handling.
    pub async fn read_var(&self) -> Result<RecordBatch> {
        if self.var_metadata_shard_count() > 0 {
            return self
                .read_sharded_metadata(
                    SectionType::VarMetadataShard,
                    "var_metadata/shard_",
                    "var_metadata",
                )
                .await;
        }
        let var_data = self.read_metadata_section("var").await?;
        Ok(scx_format::downcast_large_types(&decode_arrow_ipc_batch(
            &var_data, "var",
        )?)?)
    }

    /// Fetch every metadata shard of `shard_type` whose name starts with
    /// `prefix` (e.g. `"obs_metadata/shard_"`), decode each Arrow IPC
    /// batch **without** downcast, and assemble into one logical batch
    /// via [`scx_format::assemble_sharded_metadata`] — the same
    /// upcast → cover-validation → concat → downcast pipeline the local
    /// `ScxReader` uses, so the cloud and local read paths return
    /// byte-identical batches. Shard reads are issued concurrently, capped
    /// at [`METADATA_SHARD_FETCH_CONCURRENCY`] inflight requests.
    async fn read_sharded_metadata(
        &self,
        shard_type: SectionType,
        prefix: &str,
        logical: &str,
    ) -> Result<RecordBatch> {
        let entries: Vec<(u32, &FullCatalogEntry)> = self
            .catalog
            .entries
            .iter()
            .filter(|e| e.section_type == shard_type && e.name.starts_with(prefix))
            .filter_map(|e| {
                let idx: u32 = e.name.strip_prefix(prefix)?.parse().ok()?;
                Some((idx, e))
            })
            .collect();

        // Fetch + decode each shard with bounded concurrency.
        // `assemble_sharded_metadata` re-sorts by `shard_idx`, so the
        // out-of-order completion from `buffer_unordered` is fine.
        let raw_batches: Vec<(u32, RecordBatch)> = futures::stream::iter(entries)
            .map(|(idx, e)| async move {
                let bytes = self.read_section_for_entry(e).await?;
                Ok::<(u32, RecordBatch), CloudError>((
                    idx,
                    decode_arrow_ipc_batch(&bytes, logical)?,
                ))
            })
            .buffer_unordered(METADATA_SHARD_FETCH_CONCURRENCY)
            .try_collect()
            .await?;

        Ok(scx_format::assemble_sharded_metadata(logical, raw_batches)?)
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

    /// Read a section by its catalog entry. Equivalent to
    /// `read_section(&entry.name)` but skips the catalog lookup.
    pub async fn read_section_for_entry(&self, entry: &FullCatalogEntry) -> Result<Vec<u8>> {
        match &self.layout {
            ReaderLayout::Exploded(location) => {
                let make_path = crate::pull::build_path_fn(location);
                let rel_path =
                    section_name_to_path(&entry.name, entry.section_type).map_err(|e| {
                        CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
                    })?;
                let obj_path = make_path(&rel_path);
                let data = self.backend.get(&obj_path).await?.bytes().await?;
                Ok(data.to_vec())
            }
            ReaderLayout::Packed(file_path) => {
                let end = entry.offset.checked_add(entry.length).ok_or_else(|| {
                    CloudError::CatalogOffsetOverflow {
                        name: entry.name.clone(),
                        offset: entry.offset,
                        length: entry.length,
                    }
                })?;
                let data = self.backend.get_range(file_path, entry.offset..end).await?;
                Ok(data.to_vec())
            }
        }
    }

    /// Read the obs schema without materialising the full RecordBatch.
    ///
    /// Decodes only the Arrow IPC footer. On sharded files the schema is
    /// read from `obs_metadata/shard_0` (every shard shares one schema),
    /// mirroring `ScxReader::read_obs_schema`.
    pub async fn read_obs_schema(&self) -> Result<Schema> {
        let data = if self.obs_metadata_shard_count() > 0 {
            self.read_section("obs_metadata/shard_0").await?
        } else {
            self.read_metadata_section("obs").await?
        };
        decode_arrow_ipc_schema(&data)
    }

    /// Read the var schema without materialising the full RecordBatch.
    /// Mirror of [`Self::read_obs_schema`].
    pub async fn read_var_schema(&self) -> Result<Schema> {
        let data = if self.var_metadata_shard_count() > 0 {
            self.read_section("var_metadata/shard_0").await?
        } else {
            self.read_metadata_section("var").await?
        };
        decode_arrow_ipc_schema(&data)
    }

    /// Raw bytes of the obs predicate index section, if present.
    pub async fn read_obs_predicate_index_bytes(&self) -> Result<Option<Vec<u8>>> {
        self.read_predicate_index_bytes(SectionType::ObsPredicateIndex)
            .await
    }

    /// Raw bytes of the var predicate index section, if present.
    pub async fn read_var_predicate_index_bytes(&self) -> Result<Option<Vec<u8>>> {
        self.read_predicate_index_bytes(SectionType::VarPredicateIndex)
            .await
    }

    async fn read_predicate_index_bytes(&self, kind: SectionType) -> Result<Option<Vec<u8>>> {
        let entry = self
            .catalog
            .entries
            .iter()
            .find(|e| e.section_type == kind)
            .cloned();
        match entry {
            Some(e) => Ok(Some(self.read_section_for_entry(&e).await?)),
            None => Ok(None),
        }
    }

    /// Deletion vectors, if present in the file.
    pub async fn read_deletion_vectors(&self) -> Result<Option<scx_format::DeletionVectors>> {
        if !self.header.has_deletion_vectors() {
            return Ok(None);
        }
        let entry = self
            .catalog
            .entries
            .iter()
            .find(|e| e.section_type == SectionType::DeletionVectors)
            .cloned();
        let entry = match entry {
            Some(e) => e,
            None => return Ok(None),
        };
        let bytes = self.read_section_for_entry(&entry).await?;
        let dv = scx_format::DeletionVectors::read_from(&mut Cursor::new(&bytes), bytes.len())
            .map_err(CloudError::from)?;
        Ok(Some(dv))
    }
}

/// Decode just the schema from an Arrow IPC file's footer.
///
/// Delegates to [`scx_format::downcast_large_types_schema`] so the
/// schema matches what `ScxReader::read_obs_schema` returns even when
/// the on-disk Arrow IPC encodes `LargeUtf8` / `LargeBinary` or
/// `Dictionary(_, Large*)`. (Local fast path preserves narrow types as
/// data fits; cloud schemas are eagerly narrowed since we don't have
/// the offsets to inspect.)
/// Decode the first `RecordBatch` from an Arrow IPC file **without** any
/// wide→narrow downcast — used for both single-section reads (caller
/// downcasts afterward) and per-shard reads (the shared assembler upcasts
/// then downcasts the concatenated result). `logical` names the section
/// for error messages.
fn decode_arrow_ipc_batch(bytes: &[u8], logical: &str) -> Result<RecordBatch> {
    let cursor = Cursor::new(bytes);
    let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)
        .map_err(|e| CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
    reader
        .into_iter()
        .next()
        .ok_or_else(|| {
            CloudError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{logical} Arrow IPC contains no batches"),
            ))
        })?
        .map_err(|e| CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))
}

fn decode_arrow_ipc_schema(bytes: &[u8]) -> Result<Schema> {
    let cursor = Cursor::new(bytes);
    let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)
        .map_err(|e| CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
    Ok(scx_format::downcast_large_types_schema(
        reader.schema().as_ref(),
    ))
}

/// Open an SCX file or directory from cloud/local storage.
///
/// Detects the layout automatically:
///   - If `_catalog.bin` is found → exploded directory
///   - Otherwise → packed file (auto-detects cloud-ready vs not)
pub async fn open_cloud(url: &str) -> Result<CloudReader> {
    let location = crate::backend::parse_location(url)?;
    let backend: Arc<dyn ObjectStore> = Arc::from(crate::backend::create_backend(&location).await?);

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
                true,
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
                obs_bytes_cache: tokio::sync::OnceCell::new(),
                var_bytes_cache: tokio::sync::OnceCell::new(),
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
            let data = backend.get_range(&file_path, 0..first_chunk_size).await?;
            let first_bytes = data.to_vec();

            if first_bytes.len() < HEADER_SIZE {
                return Err(CloudError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "file too small to contain SCX header",
                )));
            }

            let header = FileHeader::read_from(&mut Cursor::new(&first_bytes[..HEADER_SIZE]))?;

            if header.has_front_catalog()
                && header.front_catalog_offset > 0
                && header.front_catalog_length > 0
            {
                // Cloud-ready: front catalog is at start of file
                let fc_offset = header.front_catalog_offset;
                let fc_end = fc_offset
                    .checked_add(header.front_catalog_length)
                    .ok_or_else(|| CloudError::CatalogOffsetOverflow {
                        name: "front_catalog".into(),
                        offset: fc_offset,
                        length: header.front_catalog_length,
                    })?;

                let fc_bytes = if (fc_end as usize) <= first_bytes.len() {
                    first_bytes[fc_offset as usize..fc_end as usize].to_vec()
                } else {
                    backend
                        .get_range(&file_path, fc_offset..fc_end)
                        .await?
                        .to_vec()
                };

                let catalog =
                    FullCatalog::read_from(&mut Cursor::new(&fc_bytes), fc_bytes.len(), true)?;

                Ok(CloudReader {
                    backend,
                    layout: ReaderLayout::Packed(file_path),
                    header,
                    catalog,
                    obs_bytes_cache: tokio::sync::OnceCell::new(),
                    var_bytes_cache: tokio::sync::OnceCell::new(),
                })
            } else {
                // Not cloud-ready: read full catalog at EOF
                let fc_offset = header.full_catalog_offset;
                let fc_end = fc_offset
                    .checked_add(header.full_catalog_length)
                    .ok_or_else(|| CloudError::CatalogOffsetOverflow {
                        name: "full_catalog".into(),
                        offset: fc_offset,
                        length: header.full_catalog_length,
                    })?;

                let fc_bytes = backend
                    .get_range(&file_path, fc_offset..fc_end)
                    .await?
                    .to_vec();

                let catalog =
                    FullCatalog::read_from(&mut Cursor::new(&fc_bytes), fc_bytes.len(), true)?;

                Ok(CloudReader {
                    backend,
                    layout: ReaderLayout::Packed(file_path),
                    header,
                    catalog,
                    obs_bytes_cache: tokio::sync::OnceCell::new(),
                    var_bytes_cache: tokio::sync::OnceCell::new(),
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
            format_version: scx_format::CURRENT_FORMAT_VERSION,
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
            n_modalities: 0,
            modality_table_offset: 0,
            modality_table_length: 0,
            reserved: [0u8; 112],
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

    fn write_test_file(dir: &tempfile::TempDir, n_obs: usize, n_vars: usize) -> std::path::PathBuf {
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

    /// Write a Phase 2 sharded-metadata `.scx`: obs is emitted as
    /// multiple `ObsMetadataShard` sections and var as multiple
    /// `VarMetadataShard` sections (both small enough to fit a single
    /// section, but split to exercise the assembly path).
    fn write_sharded_test_file(
        dir: &tempfile::TempDir,
        n_obs: usize,
        n_vars: usize,
    ) -> std::path::PathBuf {
        let path = dir.path().join("sharded.scx");
        let header = sample_header(n_obs as u64, n_vars as u64);
        let mut writer = ScxWriter::new(&path, header).unwrap();

        // Sharded obs: 40 rows per shard.
        let obs = sample_obs(n_obs);
        let obs_rows_per_shard = 40usize;
        let mut shard_idx = 0u32;
        let mut off = 0usize;
        while off < n_obs {
            let len = std::cmp::min(obs_rows_per_shard, n_obs - off);
            let chunk = obs.slice(off, len);
            writer
                .write_obs_shard(shard_idx, off as u64, len as u64, n_obs as u64, &chunk)
                .unwrap();
            off += len;
            shard_idx += 1;
        }

        // Sharded var: 20 rows per shard.
        let var = sample_var(n_vars);
        let var_rows_per_shard = 20usize;
        let mut vshard_idx = 0u32;
        let mut voff = 0usize;
        while voff < n_vars {
            let len = std::cmp::min(var_rows_per_shard, n_vars - voff);
            let chunk = var.slice(voff, len);
            writer
                .write_var_shard(vshard_idx, voff as u64, len as u64, n_vars as u64, &chunk)
                .unwrap();
            voff += len;
            vshard_idx += 1;
        }

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

    /// Assert a reader opened over a sharded-obs file returns the fully
    /// assembled obs/var batches and correct schemas. Shared by the
    /// three-layout tests below.
    async fn assert_sharded_reads(reader: &CloudReader, n_obs: usize, n_vars: usize) {
        assert!(reader.obs_metadata_shard_count() > 1);
        assert!(reader.var_metadata_shard_count() > 1);

        let obs = reader.read_obs().await.unwrap();
        assert_eq!(obs.num_rows(), n_obs);
        assert_eq!(obs.num_columns(), 2);
        let cell_id = obs
            .column_by_name("cell_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        // Values must round-trip across the shard boundaries (40 rows each).
        assert_eq!(cell_id.value(0), "cell_0");
        assert_eq!(cell_id.value(39), "cell_39");
        assert_eq!(cell_id.value(40), "cell_40");
        assert_eq!(cell_id.value(n_obs - 1), format!("cell_{}", n_obs - 1));

        let var = reader.read_var().await.unwrap();
        assert_eq!(var.num_rows(), n_vars);
        let gene_id = var
            .column_by_name("gene_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(gene_id.value(0), "gene_0");
        assert_eq!(gene_id.value(n_vars - 1), format!("gene_{}", n_vars - 1));

        // Schema reads must not require materialising the full batch.
        let obs_schema = reader.read_obs_schema().await.unwrap();
        assert!(obs_schema.field_with_name("cell_type").is_ok());
        let var_schema = reader.read_var_schema().await.unwrap();
        assert!(var_schema.field_with_name("gene_id").is_ok());
    }

    #[tokio::test]
    async fn test_sharded_obs_from_exploded() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_sharded_test_file(&dir, 100, 50);
        let exploded = dir.path().join("sharded.scxd");
        crate::explode::explode(&input, &exploded).unwrap();

        let reader = open_cloud(&exploded.to_string_lossy()).await.unwrap();
        assert_sharded_reads(&reader, 100, 50).await;
    }

    #[tokio::test]
    async fn test_sharded_obs_from_cloud_ready_packed() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_sharded_test_file(&dir, 100, 50);
        let optimized = dir.path().join("sharded_ready.scx");
        crate::cloud_optimize::cloud_optimize(&input, &optimized).unwrap();

        let reader = open_cloud(&optimized.to_string_lossy()).await.unwrap();
        assert_sharded_reads(&reader, 100, 50).await;
    }

    #[tokio::test]
    async fn test_sharded_obs_from_non_cloud_ready_packed() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_sharded_test_file(&dir, 100, 50);

        let reader = open_cloud(&input.to_string_lossy()).await.unwrap();
        assert_sharded_reads(&reader, 100, 50).await;
    }
}
