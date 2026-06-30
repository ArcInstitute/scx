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

use scx_format_io::catalog::{FullCatalog, FullCatalogEntry};
use scx_format_io::header::{FileHeader, HEADER_SIZE};
use scx_format_io::section::SectionType;

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

/// Chunk size for ranged reads of a single packed section. A monolithic GET of
/// a multi-GB section (e.g. the `ObsPredicateIndex`, which scales with `n_obs`
/// — ~2.4 GB at 149 M cells, ~9 GB at 561 M) streams for minutes and is killed
/// by a single mid-stream HTTP/2 body reset, with every retry re-downloading
/// from zero. Splitting the read into bounded chunks makes each chunk
/// independently retryable: a transient body error re-fetches only one
/// `SECTION_READ_CHUNK_BYTES` window, not the whole section.
const SECTION_READ_CHUNK_BYTES: u64 = 128 * 1024 * 1024;

/// Concurrency for the chunked section read above. Bounded so a large section
/// fetch can't fan out an unbounded number of simultaneous range GETs against
/// the same connection pool (the saturation the chunking is meant to avoid).
const SECTION_READ_CHUNK_CONCURRENCY: usize = 8;

/// Range-read `start..end` of `path`, splitting reads larger than `chunk_bytes`
/// into bounded, independently-retryable chunks fetched with `concurrency`-way
/// parallelism and reassembled in offset order. Small reads take the single-GET
/// fast path. A monolithic multi-GB GET (e.g. the `ObsPredicateIndex`, ~2.4 GB
/// at 149 M cells, ~9 GB at 561 M) is killed by a single mid-stream HTTP/2 body
/// reset and every retry restarts from zero; chunking bounds the blast radius of
/// a transient to one chunk.
///
/// The reassembled buffer length is verified against the requested span: cloud
/// predicate-index / section bytes are not independently checksum-verified
/// here, so a backend that returned a short read would otherwise truncate the
/// section silently.
pub(crate) async fn read_range_chunked(
    backend: &dyn ObjectStore,
    path: &ObjPath,
    start: u64,
    end: u64,
    chunk_bytes: u64,
    concurrency: usize,
) -> Result<Vec<u8>> {
    let total = end.saturating_sub(start);
    let chunk_bytes = chunk_bytes.max(1);

    let buf = if total <= chunk_bytes {
        crate::pull::get_range_with_retry(backend, path, start..end)
            .await?
            .to_vec()
    } else {
        // Ordered chunk ranges. `buffered` (not `buffer_unordered`) preserves
        // order, so the concatenation below reassembles the section
        // byte-for-byte.
        let mut ranges: Vec<std::ops::Range<u64>> = Vec::new();
        let mut off = start;
        while off < end {
            let chunk_end = off.saturating_add(chunk_bytes).min(end);
            ranges.push(off..chunk_end);
            off = chunk_end;
        }

        let chunks: Vec<bytes::Bytes> = futures::stream::iter(
            ranges
                .into_iter()
                .map(|r| crate::pull::get_range_with_retry(backend, path, r)),
        )
        .buffered(concurrency.max(1))
        .try_collect()
        .await?;

        let mut buf = Vec::with_capacity(total as usize);
        for c in &chunks {
            buf.extend_from_slice(c);
        }
        buf
    };

    if buf.len() as u64 != total {
        return Err(CloudError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!(
                "short read for {path}: expected {total} bytes ({start}..{end}), got {}",
                buf.len()
            ),
        )));
    }
    Ok(buf)
}

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
    /// Logical-level caches for the assembled obs/var `RecordBatch` (LC4).
    /// On atlas-scale sharded files, `read_obs`/`read_var` would otherwise
    /// re-fetch and re-assemble every `ObsMetadataShard`/`VarMetadataShard`
    /// section on each call (and `read_*_schema` fetch `shard_0` separately).
    /// Caching the assembled batch makes obs/var single-assembly per reader,
    /// and lets the schema derive from it without a redundant GET.
    obs_assembled: tokio::sync::OnceCell<RecordBatch>,
    var_assembled: tokio::sync::OnceCell<RecordBatch>,
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
                let data = crate::pull::get_with_retry(self.backend.as_ref(), &obj_path).await?;
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
                self.read_packed_range_chunked(file_path, entry.offset, end)
                    .await
            }
        }
    }

    /// Read obs metadata as an Arrow RecordBatch.
    ///
    /// Transparently handles both layouts: assembles every
    /// [`SectionType::ObsMetadataShard`] section in `shard_idx` order on
    /// Phase 2 sharded files (parallel range reads → shared
    /// [`scx_format_io::assemble_sharded_metadata`]), or reads the single
    /// legacy [`SectionType::ObsMetadata`] section otherwise. Bypasses
    /// `ScxReader::read_arrow_ipc`, so applies
    /// `scx_format_io::downcast_large_types` explicitly to surface the
    /// canonical narrow `Utf8` / `Binary` types regardless of the
    /// on-disk encoding.
    pub async fn read_obs(&self) -> Result<RecordBatch> {
        self.obs_assembled
            .get_or_try_init(|| self.load_obs())
            .await
            .cloned()
    }

    /// Assemble the obs batch (uncached). See [`Self::read_obs`].
    async fn load_obs(&self) -> Result<RecordBatch> {
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
        Ok(scx_format_io::downcast_large_types(
            &decode_arrow_ipc_batch(&obs_data, "obs")?,
        )?)
    }

    /// Read one obs metadata row-shard by index, decoded and narrowed to
    /// canonical `Utf8` / `Binary` types (parity with
    /// `ScxReader::read_obs_shard`), with the per-shard `row_start` /
    /// `n_shard_rows` schema metadata preserved. Bounded by one shard —
    /// does NOT assemble the full obs table, so it never re-OOMs the way
    /// [`Self::read_obs`] can at atlas scale.
    pub async fn read_obs_shard(&self, shard_idx: u32) -> Result<RecordBatch> {
        let name = format!("obs_metadata/shard_{shard_idx}");
        let entry = self
            .catalog
            .entries
            .iter()
            .find(|e| e.section_type == SectionType::ObsMetadataShard && e.name == name)
            .ok_or_else(|| CloudError::SectionNotFound(name.clone()))?;
        let bytes = self.read_section_for_entry(entry).await?;
        Ok(scx_format_io::downcast_large_types(
            &decode_arrow_ipc_batch(&bytes, "obs_metadata")?,
        )?)
    }

    /// Read var metadata as an Arrow RecordBatch. Mirror of
    /// [`Self::read_obs`] for the var axis — same dual-layout handling and
    /// per-axis assembled-batch caching (LC4).
    pub async fn read_var(&self) -> Result<RecordBatch> {
        self.var_assembled
            .get_or_try_init(|| self.load_var())
            .await
            .cloned()
    }

    /// Assemble the var batch (uncached). See [`Self::read_var`].
    async fn load_var(&self) -> Result<RecordBatch> {
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
        Ok(scx_format_io::downcast_large_types(
            &decode_arrow_ipc_batch(&var_data, "var")?,
        )?)
    }

    /// Fetch every metadata shard of `shard_type` whose name starts with
    /// `prefix` (e.g. `"obs_metadata/shard_"`), decode each Arrow IPC
    /// batch **without** downcast, and assemble into one logical batch
    /// via [`scx_format_io::assemble_sharded_metadata`] — the same
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

        Ok(scx_format_io::assemble_sharded_metadata(
            logical,
            raw_batches,
        )?)
    }

    /// Read multiple shard sections in parallel, capping inflight requests at
    /// [`METADATA_SHARD_FETCH_CONCURRENCY`] so atlas-scale files don't fire an
    /// unbounded number of simultaneous fetches. Output order matches
    /// `shard_names` (`buffered`, not `buffer_unordered`).
    pub async fn read_shards(&self, shard_names: &[&str]) -> Result<Vec<Vec<u8>>> {
        futures::stream::iter(shard_names.iter().copied())
            .map(|name| self.read_section(name))
            .buffered(METADATA_SHARD_FETCH_CONCURRENCY)
            .try_collect()
            .await
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
                let data = crate::pull::get_with_retry(self.backend.as_ref(), &obj_path).await?;
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
                self.read_packed_range_chunked(file_path, entry.offset, end)
                    .await
            }
        }
    }

    /// Range-read `start..end` of a packed file, splitting reads larger than
    /// [`SECTION_READ_CHUNK_BYTES`] into bounded, independently-retryable
    /// chunks. Small sections take the single-GET fast path. Chunks are fetched
    /// with bounded concurrency and reassembled in offset order — a monolithic
    /// multi-GB GET is fatal on a flaky link (a single mid-stream body reset
    /// fails the whole download, and every retry restarts from zero).
    async fn read_packed_range_chunked(
        &self,
        file_path: &ObjPath,
        start: u64,
        end: u64,
    ) -> Result<Vec<u8>> {
        read_range_chunked(
            self.backend.as_ref(),
            file_path,
            start,
            end,
            SECTION_READ_CHUNK_BYTES,
            SECTION_READ_CHUNK_CONCURRENCY,
        )
        .await
    }

    /// Read the obs schema — **bounded to a single section**, never
    /// assembling the full obs table.
    ///
    /// Fetches only the first obs section (`obs_metadata/shard_0` on sharded
    /// files, else the legacy `obs` section) and decodes its Arrow IPC
    /// schema via the shared [`scx_format_io::decode_arrow_ipc_schema`], the
    /// same core `ScxReader::read_obs_schema` uses — so the cloud schema is
    /// byte-identical to the local path. Does **not** touch [`Self::read_obs`]
    /// / `obs_assembled` (which would fetch and concatenate every
    /// `ObsMetadataShard` and OOM at atlas scale); the lazy full-assembly
    /// path stays available for callers that genuinely need the whole batch.
    pub async fn read_obs_schema(&self) -> Result<Schema> {
        let entry = self.first_obs_section_entry()?;
        let bytes = self.read_section_for_entry(entry).await?;
        Ok(scx_format_io::decode_arrow_ipc_schema(&bytes)?)
    }

    /// Read the var schema. Mirror of [`Self::read_obs_schema`].
    pub async fn read_var_schema(&self) -> Result<Schema> {
        let entry = self.first_var_section_entry()?;
        let bytes = self.read_section_for_entry(entry).await?;
        Ok(scx_format_io::decode_arrow_ipc_schema(&bytes)?)
    }

    /// Resolve the first obs section catalog entry: `obs_metadata/shard_0`
    /// when obs is sharded, else the legacy single `obs` section. Mirror of
    /// `ScxReader::first_obs_section_entry`.
    fn first_obs_section_entry(&self) -> Result<&FullCatalogEntry> {
        let (ty, name) = if self.obs_metadata_shard_count() > 0 {
            (SectionType::ObsMetadataShard, "obs_metadata/shard_0")
        } else {
            (SectionType::ObsMetadata, "obs")
        };
        self.catalog
            .entries
            .iter()
            .find(|e| e.section_type == ty && e.name == name)
            .ok_or_else(|| CloudError::SectionNotFound(name.into()))
    }

    /// Mirror of [`Self::first_obs_section_entry`] for the var axis.
    fn first_var_section_entry(&self) -> Result<&FullCatalogEntry> {
        let (ty, name) = if self.var_metadata_shard_count() > 0 {
            (SectionType::VarMetadataShard, "var_metadata/shard_0")
        } else {
            (SectionType::VarMetadata, "var")
        };
        self.catalog
            .entries
            .iter()
            .find(|e| e.section_type == ty && e.name == name)
            .ok_or_else(|| CloudError::SectionNotFound(name.into()))
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

    /// Raw bytes of the F1 `group_index` sidecar section, if present.
    pub async fn read_group_index_bytes(&self) -> Result<Option<Vec<u8>>> {
        self.read_predicate_index_bytes(SectionType::GroupIndex)
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
    pub async fn read_deletion_vectors(&self) -> Result<Option<scx_format_io::DeletionVectors>> {
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
        let dv = scx_format_io::DeletionVectors::read_from(&mut Cursor::new(&bytes), bytes.len())
            .map_err(CloudError::from)?;
        Ok(Some(dv))
    }

    /// Whether the source is an exploded `.scxd/` directory (each section a
    /// separate object) rather than a packed `.scx` file. `scx info` uses
    /// this to report live-bytes (not a single file size) and to suppress the
    /// "Orphaned bytes" line, which is meaningless for an exploded directory.
    pub fn is_exploded(&self) -> bool {
        matches!(self.layout, ReaderLayout::Exploded(_))
    }

    /// Total byte footprint of the source.
    ///
    /// Packed: the object size via a single `HEAD`. Exploded: the summed live
    /// bytes (`SECTIONS_START_OFFSET` + full-catalog length + Σ section
    /// lengths), since there is no single file to stat — this matches the
    /// "live" region `scx info` uses for its orphaned-bytes accounting.
    pub async fn object_size(&self) -> Result<u64> {
        match &self.layout {
            ReaderLayout::Packed(file_path) => {
                let meta = self
                    .backend
                    .head(file_path)
                    .await
                    .map_err(|e| CloudError::from_store_error(e, file_path))?;
                Ok(meta.size)
            }
            ReaderLayout::Exploded(_) => {
                let section_bytes: u64 = self.catalog.entries.iter().map(|e| e.length).sum();
                Ok(scx_format_io::SECTIONS_START_OFFSET
                    + self.header.full_catalog_length
                    + section_bytes)
            }
        }
    }

    /// The parsed modality table, if the file is multimodal.
    ///
    /// `None` when `n_modalities == 0`. Reads the `ModalityTable` section
    /// (catalog-entry driven, so it works for both packed and exploded
    /// layouts) and parses it identically to the local `ScxReader` path.
    pub async fn modality_table(&self) -> Result<Option<scx_format_io::modality::ModalityTable>> {
        if self.header.n_modalities == 0 {
            return Ok(None);
        }
        let entry = self
            .catalog
            .entries
            .iter()
            .find(|e| e.section_type == SectionType::ModalityTable)
            .cloned();
        let entry = match entry {
            Some(e) => e,
            None => return Ok(None),
        };
        let bytes = self.read_section_for_entry(&entry).await?;
        let table = scx_format_io::modality::ModalityTable::read_from(
            &mut Cursor::new(&bytes),
            bytes.len(),
        )
        .map_err(CloudError::from)?;
        Ok(Some(table))
    }

    /// Provenance operations, if present.
    pub async fn read_provenance(&self) -> Result<Option<scx_format_io::Provenance>> {
        let entry = self
            .catalog
            .entries
            .iter()
            .find(|e| e.section_type == SectionType::Provenance)
            .cloned();
        let entry = match entry {
            Some(e) => e,
            None => return Ok(None),
        };
        let bytes = self.read_section_for_entry(&entry).await?;
        let prov = scx_format_io::Provenance::read_from(&mut Cursor::new(&bytes), bytes.len())
            .map_err(CloudError::from)?;
        Ok(Some(prov))
    }

    /// Read the global `uns` JSON section. `Ok(None)` if the file has none.
    ///
    /// Mirrors the local `ScxReader::read_uns`, but follows the cloud
    /// convention (`read_provenance`) of returning `Option` for an absent
    /// section rather than a `SectionNotFound` error.
    pub async fn read_uns(&self) -> Result<Option<serde_json::Value>> {
        self.read_uns_for(0).await
    }

    /// Read the `uns` section for a modality. `modality_id == 0` reads the
    /// global `"uns"` section; `> 0` reads the per-modality `"uns/<name>"`
    /// section. `Ok(None)` if the section is absent.
    pub async fn read_uns_for(&self, modality_id: u8) -> Result<Option<serde_json::Value>> {
        let section_name = if modality_id == 0 {
            "uns".to_string()
        } else {
            let table = self.modality_table().await?;
            let name = table
                .as_ref()
                .and_then(|t| t.info_of(modality_id))
                .map(|m| m.name.clone())
                .ok_or_else(|| {
                    CloudError::SectionNotFound(format!("uns/<modality_id {modality_id}>"))
                })?;
            format!("uns/{name}")
        };
        // Catalog lookup by name (like `read_provenance`), so an absent
        // section short-circuits to `Ok(None)` without a wasted range-GET.
        if !self.catalog.entries.iter().any(|e| e.name == section_name) {
            return Ok(None);
        }
        let bytes = self.read_section(&section_name).await?;
        // `CloudError` has no direct `From<serde_json::Error>`; route the
        // parse error through `ScxError` (which does), lifting it to
        // `CloudError::Format` via `?`.
        let val = serde_json::from_slice(&bytes).map_err(scx_format_io::ScxError::from)?;
        Ok(Some(val))
    }

    /// De-duplicated `obsm` embedding names. Pure in-memory catalog scan
    /// (the catalog is loaded at open), no network I/O. Mirrors the local
    /// `ScxReader::list_obsm`.
    pub fn list_obsm(&self) -> Vec<String> {
        self.catalog.list_logical_names(
            "obsm",
            SectionType::ObsmEmbedding,
            SectionType::ObsmEmbeddingShard,
        )
    }

    /// De-duplicated `varm` embedding names. Pure in-memory catalog scan.
    /// Mirrors the local `ScxReader::list_varm`.
    pub fn list_varm(&self) -> Vec<String> {
        self.catalog.list_logical_names(
            "varm",
            SectionType::VarmEmbedding,
            SectionType::VarmEmbeddingShard,
        )
    }

    /// De-duplicated layer names. Pure in-memory catalog scan. Mirrors the
    /// local `ScxReader::layer_names`.
    pub fn layer_names(&self) -> Vec<String> {
        self.catalog.layer_names()
    }

    /// Distinct codec ids and value encodings across **all** CSR shards,
    /// each sorted ascending. Mirrors the local
    /// `distinct_sorted_shard_field` summary `scx info` prints, but range-reads
    /// only each shard's 76-byte header over the network (in parallel). Empty
    /// vecs when there are no CSR shards.
    pub async fn csr_shard_field_summaries(&self) -> Result<(Vec<u8>, Vec<u8>)> {
        let shards: Vec<FullCatalogEntry> = self
            .catalog
            .shards(SectionType::CsrShard)
            .into_iter()
            .cloned()
            .collect();
        if shards.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let headers: Vec<(u8, u8)> = futures::stream::iter(shards.into_iter().map(|entry| {
            let this = &self;
            async move {
                let hdr = this.read_shard_header(&entry).await?;
                Ok::<(u8, u8), CloudError>((hdr.codec_id, hdr.value_encoding))
            }
        }))
        .buffer_unordered(METADATA_SHARD_FETCH_CONCURRENCY)
        .try_collect()
        .await?;

        let mut codecs: Vec<u8> = headers.iter().map(|(c, _)| *c).collect();
        let mut encs: Vec<u8> = headers.iter().map(|(_, e)| *e).collect();
        codecs.sort_unstable();
        codecs.dedup();
        encs.sort_unstable();
        encs.dedup();
        Ok((codecs, encs))
    }

    /// Range-read and parse a single shard's 76-byte header.
    async fn read_shard_header(
        &self,
        entry: &FullCatalogEntry,
    ) -> Result<scx_format_io::shard::ShardHeader> {
        use scx_format_io::shard::SHARD_HEADER_SIZE;
        let len = SHARD_HEADER_SIZE as u64;
        let bytes = match &self.layout {
            ReaderLayout::Exploded(location) => {
                let make_path = crate::pull::build_path_fn(location);
                let rel_path =
                    section_name_to_path(&entry.name, entry.section_type).map_err(|e| {
                        CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
                    })?;
                let obj_path = make_path(&rel_path);
                crate::pull::get_range_with_retry(self.backend.as_ref(), &obj_path, 0..len).await?
            }
            ReaderLayout::Packed(file_path) => {
                let end = entry.offset.checked_add(len).ok_or_else(|| {
                    CloudError::CatalogOffsetOverflow {
                        name: entry.name.clone(),
                        offset: entry.offset,
                        length: entry.length,
                    }
                })?;
                crate::pull::get_range_with_retry(
                    self.backend.as_ref(),
                    file_path,
                    entry.offset..end,
                )
                .await?
            }
        };
        if (bytes.len() as u64) < len {
            return Err(CloudError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "shard '{}' header truncated: got {} bytes, need {}",
                    entry.name,
                    bytes.len(),
                    SHARD_HEADER_SIZE
                ),
            )));
        }
        scx_format_io::shard::ShardHeader::read_from(&mut Cursor::new(&bytes))
            .map_err(CloudError::from)
    }
}

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

/// Whether an `object_store` error means "the object isn't there".
///
/// Covers both the canonical `NotFound` variant (remote stores like GCS/S3)
/// and the `LocalFileSystem` case, which surfaces a missing file as a
/// `Generic` error wrapping a `std::io::Error` of kind `NotFound` (e.g.
/// `UnableToCanonicalize`) — detected by walking the error source chain.
/// Used to decide layout fallback and, ultimately, to emit the actionable
/// `CatalogNotFound` instead of leaking a raw 404 / canonicalize error —
/// while leaving auth/network errors verbose.
pub(crate) fn is_missing_object(e: &object_store::Error) -> bool {
    // Remote stores (GCS/S3/Azure) report a missing object as the canonical
    // `NotFound` variant.
    if matches!(e, object_store::Error::NotFound { .. }) {
        return true;
    }
    // `LocalFileSystem` reports a missing file as a `Generic` wrapping (e.g.)
    // `UnableToCanonicalize`, which carries a `std::io::Error` of kind
    // `NotFound`. Walk the source chain and downcast to `io::Error`.
    if let object_store::Error::Generic { source, .. } = e {
        let mut current: &(dyn std::error::Error + 'static) = source.as_ref();
        loop {
            if let Some(io) = current.downcast_ref::<std::io::Error>() {
                if io.kind() == std::io::ErrorKind::NotFound {
                    return true;
                }
            }
            match current.source() {
                Some(next) => current = next,
                None => break,
            }
        }
    }
    false
}

/// Open an SCX file or directory from cloud/local storage.
///
/// Detects the layout automatically:
///   - If `_catalog.bin` is found → exploded directory
///   - Otherwise → packed file (auto-detects cloud-ready vs not)
pub async fn open_cloud(url: &str) -> Result<CloudReader> {
    let location = crate::backend::parse_location(url)?;
    // For a missing LOCAL path, backend creation itself fails (the
    // `LocalFileSystem` prefix can't be canonicalized) before any GET — map
    // that to the actionable `CatalogNotFound` too. Remote backends build
    // lazily, so their missing-object error surfaces at the GETs below.
    let backend: Arc<dyn ObjectStore> = match crate::backend::create_backend(&location).await {
        Ok(b) => Arc::from(b),
        Err(CloudError::ObjectStore(e)) if is_missing_object(&e) => {
            return Err(CloudError::CatalogNotFound(url.to_string()));
        }
        Err(e) => return Err(e),
    };

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
                obs_assembled: tokio::sync::OnceCell::new(),
                var_assembled: tokio::sync::OnceCell::new(),
            })
        }
        Err(e) => {
            // Only fall through to packed-file path when `_catalog.bin` is
            // simply absent (so this isn't an exploded dir). Auth, network,
            // and other errors should propagate immediately.
            if !is_missing_object(&e) {
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
            // Neither the exploded `_catalog.bin` nor (here) the packed
            // file resolved. A `NotFound` at this point means the path
            // points at no SCX data at all — emit the actionable
            // `CatalogNotFound` rather than leaking the raw object_store
            // 404 (which is percent-encoded and dumps the provider's XML).
            // Auth/network/other errors still propagate verbatim.
            let data = match backend.get_range(&file_path, 0..first_chunk_size).await {
                Ok(d) => d,
                Err(e) if is_missing_object(&e) => {
                    return Err(CloudError::CatalogNotFound(url.to_string()));
                }
                Err(e) => return Err(CloudError::ObjectStore(e)),
            };
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
                    obs_assembled: tokio::sync::OnceCell::new(),
                    var_assembled: tokio::sync::OnceCell::new(),
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
                    obs_assembled: tokio::sync::OnceCell::new(),
                    var_assembled: tokio::sync::OnceCell::new(),
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
    use scx_format_io::section::SectionType;
    use scx_format_io::writer::ScxWriter;
    use std::sync::Arc as StdArc;

    fn sample_header(n_obs: u64, n_vars: u64) -> FileHeader {
        FileHeader::new_single_modality(n_obs, n_vars, 0, 16384, 0, 0)
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
    async fn test_open_missing_path_returns_catalog_not_found() {
        // A path that resolves to neither an exploded .scxd/ (`_catalog.bin`)
        // nor a packed .scx must surface the actionable `CatalogNotFound`,
        // not a raw object_store NotFound (E1).
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does_not_exist.scx");
        // `CloudReader` (the Ok type) isn't Debug, so match rather than
        // `unwrap_err()`.
        match open_cloud(&missing.to_string_lossy()).await {
            Err(CloudError::CatalogNotFound(_)) => {}
            Err(other) => panic!("expected CatalogNotFound, got {other:?}"),
            Ok(_) => panic!("expected CatalogNotFound, got Ok"),
        }
    }

    /// Write a two-modality (rna + adt) file so the modality-table cloud path
    /// has something to parse, with distinct per-modality shard counts/nnz.
    fn write_multimodal_file(dir: &tempfile::TempDir) -> std::path::PathBuf {
        use scx_format_io::modality::ModalityType;
        let path = dir.path().join("multimodal.scx");
        let header = sample_header(20, 50);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(20)).unwrap();

        let rna_id = writer
            .add_modality(
                "rna",
                ModalityType::Rna,
                CodecId::None,
                ValueEncoding::Uint8,
                false,
            )
            .unwrap();
        let adt_id = writer
            .add_modality(
                "adt",
                ModalityType::Protein,
                CodecId::None,
                ValueEncoding::Uint8,
                false,
            )
            .unwrap();
        writer.write_var_for(rna_id, &sample_var(50)).unwrap();
        writer.write_var_for(adt_id, &sample_var(50)).unwrap();
        writer.set_modality_n_vars(rna_id, 50).unwrap();
        writer.set_modality_n_vars(adt_id, 50).unwrap();

        let (indptr, indices, values) = sample_shard_data(20, 50);
        writer
            .write_csr_shard_for(
                rna_id,
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        let (indptr2, indices2, values2) = sample_shard_data(10, 50);
        writer
            .write_csr_shard_for(
                adt_id,
                &indptr2,
                &indices2,
                &values2,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        writer.finish().unwrap();
        path
    }

    #[tokio::test]
    async fn object_size_packed_matches_file_len() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);
        let expected = std::fs::metadata(&input).unwrap().len();

        let reader = open_cloud(&input.to_string_lossy()).await.unwrap();
        assert!(!reader.is_exploded());
        assert_eq!(reader.object_size().await.unwrap(), expected);
    }

    #[tokio::test]
    async fn object_size_exploded_reports_live_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);
        let exploded = dir.path().join("test.scxd");
        crate::explode::explode(&input, &exploded).unwrap();

        let reader = open_cloud(&exploded.to_string_lossy()).await.unwrap();
        assert!(reader.is_exploded());
        let section_bytes: u64 = reader.catalog().entries.iter().map(|e| e.length).sum();
        let expected = scx_format_io::SECTIONS_START_OFFSET
            + reader.header().full_catalog_length
            + section_bytes;
        assert_eq!(reader.object_size().await.unwrap(), expected);
    }

    #[tokio::test]
    async fn csr_shard_summaries_match_local() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50); // 2 shards, None/Uint8
        let exploded = dir.path().join("test.scxd");
        crate::explode::explode(&input, &exploded).unwrap();

        for src in [
            input.to_string_lossy().to_string(),
            exploded.to_string_lossy().to_string(),
        ] {
            let reader = open_cloud(&src).await.unwrap();
            let (codecs, encs) = reader.csr_shard_field_summaries().await.unwrap();
            assert_eq!(codecs, vec![CodecId::None as u8], "codecs for {src}");
            assert_eq!(
                encs,
                vec![ValueEncoding::Uint8 as u8],
                "encodings for {src}"
            );
        }
    }

    fn write_test_file_with_uns(
        dir: &tempfile::TempDir,
        uns: Option<&serde_json::Value>,
    ) -> std::path::PathBuf {
        let n_obs = 50;
        let n_vars = 10;
        let path = dir.path().join("test_uns.scx");
        let header = sample_header(n_obs as u64, n_vars as u64);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();
        let (indptr, indices, values) = sample_shard_data(n_obs, n_vars);
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
        if let Some(u) = uns {
            writer.write_uns(u).unwrap();
        }
        writer.finish().unwrap();
        path
    }

    #[tokio::test]
    async fn read_uns_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let uns = serde_json::json!({"species": "human", "version": 2});
        let input = write_test_file_with_uns(&dir, Some(&uns));
        let exploded = dir.path().join("test_uns.scxd");
        crate::explode::explode(&input, &exploded).unwrap();

        for src in [
            input.to_string_lossy().to_string(),
            exploded.to_string_lossy().to_string(),
        ] {
            let reader = open_cloud(&src).await.unwrap();
            let got = reader.read_uns().await.unwrap().expect("uns present");
            assert_eq!(got, uns, "uns mismatch for {src}");
        }
    }

    #[tokio::test]
    async fn read_uns_absent_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file_with_uns(&dir, None);
        let reader = open_cloud(&input.to_string_lossy()).await.unwrap();
        assert!(reader.read_uns().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn modality_table_matches_local() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_multimodal_file(&dir);
        let local = scx_format_io::ScxReader::open(&input).unwrap();
        let local_table = local.modality_table().expect("local modality table");

        let exploded = dir.path().join("multimodal.scxd");
        crate::explode::explode(&input, &exploded).unwrap();

        for src in [
            input.to_string_lossy().to_string(),
            exploded.to_string_lossy().to_string(),
        ] {
            let reader = open_cloud(&src).await.unwrap();
            let table = reader
                .modality_table()
                .await
                .unwrap()
                .expect("cloud modality table");
            assert_eq!(table.entries.len(), 2, "src {src}");
            for (got, want) in table.entries.iter().zip(local_table.entries.iter()) {
                assert_eq!(got.name, want.name, "src {src}");
                assert_eq!(got.nnz, want.nnz, "nnz for {} ({src})", got.name);
                assert_eq!(
                    got.n_csr_shards, want.n_csr_shards,
                    "csr for {} ({src})",
                    got.name
                );
                assert!(got.nnz > 0, "modality {} nnz should be populated", got.name);
            }
        }
    }

    #[tokio::test]
    async fn single_modality_file_has_no_modality_table() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);
        let reader = open_cloud(&input.to_string_lossy()).await.unwrap();
        assert!(reader.modality_table().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn read_provenance_matches_local() {
        use scx_format_io::provenance::ProvenanceEntry;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prov.scx");
        let mut writer = ScxWriter::new(&path, sample_header(6, 5)).unwrap();
        writer.write_obs(&sample_obs(6)).unwrap();
        writer.write_var(&sample_var(5)).unwrap();
        let (indptr, indices, values) = sample_shard_data(6, 5);
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
        writer
            .write_provenance(vec![ProvenanceEntry {
                timestamp: 1_710_000_000,
                action: "convert".to_string(),
                tool: "scx-test".to_string(),
                params_json: "{}".to_string(),
                input_checksums: vec![],
            }])
            .unwrap();
        writer.finish().unwrap();

        let local = scx_format_io::ScxReader::open(&path).unwrap();
        let local_prov = local.read_provenance().unwrap();

        let exploded = dir.path().join("prov.scxd");
        crate::explode::explode(&path, &exploded).unwrap();

        for src in [
            path.to_string_lossy().to_string(),
            exploded.to_string_lossy().to_string(),
        ] {
            let reader = open_cloud(&src).await.unwrap();
            let prov = reader
                .read_provenance()
                .await
                .unwrap()
                .expect("cloud provenance present");
            assert_eq!(
                prov.operations.len(),
                local_prov.operations.len(),
                "src {src}"
            );
            assert_eq!(prov.operations[0].action, "convert", "src {src}");
            assert_eq!(
                prov.operations[0].tool, local_prov.operations[0].tool,
                "src {src}"
            );
        }
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

        // Schema is derived from the (cached) assembled batch (LC4).
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

    /// LC4: the assembled obs/var batch is cached in a `OnceCell`, so repeated
    /// reads return identical data without re-fetching/re-assembling the
    /// shards (the `get_or_try_init` closure runs at most once per axis), and
    /// `read_*_schema` derives from the cached batch rather than a separate
    /// `shard_0` GET.
    #[tokio::test]
    async fn read_obs_var_assembled_batches_are_cached() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_sharded_test_file(&dir, 100, 50);
        let reader = open_cloud(&input.to_string_lossy()).await.unwrap();

        // Schema first (bounded shard_0 footer read — does NOT populate the
        // assembled-batch cache), then two batch reads: all three must agree.
        let obs_schema = reader.read_obs_schema().await.unwrap();
        let obs1 = reader.read_obs().await.unwrap();
        let obs2 = reader.read_obs().await.unwrap();
        assert_eq!(
            obs1, obs2,
            "repeated read_obs must return identical batches"
        );
        // Schema is read from shard_0's IPC footer (parity with the local
        // `ScxReader::read_obs_schema`), so its *fields* match the assembled
        // batch. The schema-level metadata map legitimately differs — the
        // footer carries per-shard keys (`row_start` / `shard_idx` / …) that
        // the assembler consolidates to `n_rows_total` only.
        assert_eq!(
            obs_schema.fields(),
            obs1.schema().fields(),
            "read_obs_schema fields must equal the assembled batch fields"
        );

        let var_schema = reader.read_var_schema().await.unwrap();
        let var1 = reader.read_var().await.unwrap();
        let var2 = reader.read_var().await.unwrap();
        assert_eq!(
            var1, var2,
            "repeated read_var must return identical batches"
        );
        assert_eq!(var_schema.fields(), var1.schema().fields());
    }

    /// Regression for CLOUD-READ-OOM2: reading the obs/var schema must
    /// stay bounded to a single section and must NOT trigger full-obs/var
    /// assembly (which fetches + concatenates every metadata shard and
    /// OOMs at atlas scale). The deterministic signal is that the
    /// assembled-batch `OnceCell`s remain uninitialised after a schema read.
    #[tokio::test]
    async fn read_obs_schema_does_not_assemble_full_obs() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_sharded_test_file(&dir, 100, 50);
        let reader = open_cloud(&input.to_string_lossy()).await.unwrap();

        let _obs_schema = reader.read_obs_schema().await.unwrap();
        let _var_schema = reader.read_var_schema().await.unwrap();

        assert!(
            reader.obs_assembled.get().is_none(),
            "read_obs_schema must not assemble the full obs batch"
        );
        assert!(
            reader.var_assembled.get().is_none(),
            "read_var_schema must not assemble the full var batch"
        );

        // And the schema fields must still match the assembled batch once it
        // IS genuinely requested (parity with the local path on narrow data).
        let obs = reader.read_obs().await.unwrap();
        assert_eq!(_obs_schema.fields(), obs.schema().fields());
    }
}
