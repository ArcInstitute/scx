//! `CloudSectionReader` — implements `scx_engine::SectionReader` over
//! `CloudReader`.
//!
//! Glue between the cloud range-read backend (`CloudReader`) and the
//! sync query engine (`scx_engine::QueryPipeline`). Each
//! `SectionReader` method calls `Handle::block_on` to drive the
//! underlying async `CloudReader` method. The runtime handle must be
//! supplied at construction time and outlive the reader; it does NOT
//! need to be a multi-threaded runtime, but rayon-parallel shard
//! decode (`scx_engine::collect`) calls `block_on` from worker threads
//! concurrently, so a multi-threaded runtime gets better cloud
//! concurrency.

use std::any::Any;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use scx_engine::error::EngineError;
use scx_engine::SectionReader;
use scx_format_io::catalog::FullCatalogEntry;
use scx_format_io::header::FileHeader;
use scx_format_io::modality::ModalityTable;
use scx_format_io::{DeletionVectors, FullCatalog};
use tokio::runtime::Runtime;

use crate::cloud_reader::CloudReader;
use crate::error::CloudError;

/// Sync `SectionReader` adapter wrapping a `CloudReader`.
///
/// Holds an `Arc<CloudReader>` and an `Arc<Runtime>` so the runtime
/// the adapter `block_on`s against stays alive for at least as long
/// as the adapter itself. The runtime can be shared across multiple
/// adapters / `QueryPipeline`s (e.g. multiple `.query()` calls on the
/// same `PyCloudExperiment`).
pub struct CloudSectionReader {
    inner: Arc<CloudReader>,
    rt: Arc<Runtime>,
    /// Lazily-fetched modality table, memoized so the sync modality accessors
    /// (`modality_names` / `modality_id_by_name`) don't re-read the section per
    /// call. `None` for single-modality files (or if the read failed — the
    /// modality-scoped `read_var_*` methods surface the real error instead).
    modalities: std::sync::OnceLock<Option<ModalityTable>>,
}

impl CloudSectionReader {
    /// Build a new adapter. The adapter clones the `Arc<Runtime>` so
    /// it stays alive for the adapter's lifetime; callers may share
    /// the same runtime across many adapters.
    pub fn new(inner: Arc<CloudReader>, rt: Arc<Runtime>) -> Self {
        Self {
            inner,
            rt,
            modalities: std::sync::OnceLock::new(),
        }
    }

    /// Underlying `CloudReader` (for callers that want to issue
    /// additional async calls outside the `SectionReader` surface).
    pub fn cloud_reader(&self) -> &Arc<CloudReader> {
        &self.inner
    }

    /// Memoized parsed modality table (`None` for single-modality files or on
    /// read error — best-effort; the fallible `read_var_*` / `modality_n_vars`
    /// methods block_on the reader directly and surface real errors).
    fn modalities(&self) -> Option<&ModalityTable> {
        self.modalities
            .get_or_init(|| self.rt.block_on(self.inner.modality_table()).ok().flatten())
            .as_ref()
    }
}

fn cloud_to_engine(e: CloudError) -> EngineError {
    EngineError::IoError(std::io::Error::other(e.to_string()))
}

impl SectionReader for CloudSectionReader {
    fn header(&self) -> &FileHeader {
        self.inner.header()
    }

    fn catalog(&self) -> &FullCatalog {
        self.inner.catalog()
    }

    fn read_obs_schema(&self) -> scx_engine::Result<Schema> {
        self.rt
            .block_on(self.inner.read_obs_schema())
            .map_err(cloud_to_engine)
    }

    fn read_var_schema(&self) -> scx_engine::Result<Schema> {
        self.rt
            .block_on(self.inner.read_var_schema())
            .map_err(cloud_to_engine)
    }

    fn read_obs(&self) -> scx_engine::Result<RecordBatch> {
        // `CloudReader::read_obs` transparently assembles Phase 2 sharded
        // obs (`ObsMetadataShard` sections) and falls back to the legacy
        // single-section read otherwise, so both layouts open over the
        // cloud path. The file-scope predicate index is read unchanged;
        // see `read_obs_predicate_index_bytes`.
        self.rt
            .block_on(self.inner.read_obs())
            .map_err(cloud_to_engine)
    }

    fn obs_metadata_shard_count(&self) -> usize {
        self.inner.obs_metadata_shard_count()
    }

    fn read_obs_shard(&self, shard_idx: u32) -> scx_engine::Result<RecordBatch> {
        self.rt
            .block_on(self.inner.read_obs_shard(shard_idx))
            .map_err(cloud_to_engine)
    }

    fn read_var(&self) -> scx_engine::Result<RecordBatch> {
        self.rt
            .block_on(self.inner.read_var())
            .map_err(cloud_to_engine)
    }

    // --- Modality-aware surface (enables `query(modality=…)` over cloud). ---

    fn n_modalities(&self) -> u32 {
        self.inner.header().n_modalities
    }

    fn is_multimodal(&self) -> bool {
        self.inner.header().n_modalities > 0
    }

    fn modality_names(&self) -> Vec<String> {
        self.modalities()
            .map(|t| t.entries.iter().map(|m| m.name.clone()).collect())
            .unwrap_or_default()
    }

    fn modality_id_by_name(&self, name: &str) -> Option<u8> {
        self.modalities().and_then(|t| t.id_of(name))
    }

    fn read_var_schema_for(&self, modality_id: u8) -> scx_engine::Result<Schema> {
        self.rt
            .block_on(self.inner.read_var_schema_for(modality_id))
            .map_err(cloud_to_engine)
    }

    fn read_var_for(&self, modality_id: u8) -> scx_engine::Result<RecordBatch> {
        self.rt
            .block_on(self.inner.read_var_for(modality_id))
            .map_err(cloud_to_engine)
    }

    fn modality_n_vars(&self, modality_id: u8) -> scx_engine::Result<u64> {
        self.rt
            .block_on(self.inner.modality_n_vars(modality_id))
            .map_err(cloud_to_engine)
    }

    fn read_obs_predicate_index_bytes(&self) -> scx_engine::Result<Option<Vec<u8>>> {
        self.rt
            .block_on(self.inner.read_obs_predicate_index_bytes())
            .map_err(cloud_to_engine)
    }

    fn read_var_predicate_index_bytes(&self) -> scx_engine::Result<Option<Vec<u8>>> {
        self.rt
            .block_on(self.inner.read_var_predicate_index_bytes())
            .map_err(cloud_to_engine)
    }

    fn read_group_index_bytes(&self) -> scx_engine::Result<Option<Vec<u8>>> {
        self.rt
            .block_on(self.inner.read_group_index_bytes())
            .map_err(cloud_to_engine)
    }

    fn read_deletion_vectors(&self) -> scx_engine::Result<Option<DeletionVectors>> {
        self.rt
            .block_on(self.inner.read_deletion_vectors())
            .map_err(cloud_to_engine)
    }

    fn read_shard_from_entry(
        &self,
        entry: &FullCatalogEntry,
    ) -> scx_engine::Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
        let bytes = self
            .rt
            .block_on(self.inner.read_section_for_entry(entry))
            .map_err(cloud_to_engine)?;
        let catalog_version = self.inner.catalog().catalog_version;
        Ok(scx_format_io::decode_shard_bytes(
            &bytes,
            entry,
            catalog_version,
            false,
        )?)
    }

    fn read_shard_from_entry_native(
        &self,
        entry: &FullCatalogEntry,
    ) -> scx_engine::Result<(Vec<i64>, Vec<u32>, scx_codec::ShardValuesNative)> {
        let bytes = self
            .rt
            .block_on(self.inner.read_section_for_entry(entry))
            .map_err(cloud_to_engine)?;
        let catalog_version = self.inner.catalog().catalog_version;
        Ok(scx_format_io::decode_shard_bytes_native(
            &bytes,
            entry,
            catalog_version,
            false,
        )?)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
