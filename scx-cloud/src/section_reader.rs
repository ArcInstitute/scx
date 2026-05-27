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
use scx_format::catalog::FullCatalogEntry;
use scx_format::header::FileHeader;
use scx_format::{DeletionVectors, FullCatalog};
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
}

impl CloudSectionReader {
    /// Build a new adapter. The adapter clones the `Arc<Runtime>` so
    /// it stays alive for the adapter's lifetime; callers may share
    /// the same runtime across many adapters.
    pub fn new(inner: Arc<CloudReader>, rt: Arc<Runtime>) -> Self {
        Self { inner, rt }
    }

    /// Underlying `CloudReader` (for callers that want to issue
    /// additional async calls outside the `SectionReader` surface).
    pub fn cloud_reader(&self) -> &Arc<CloudReader> {
        &self.inner
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
        // Cloud reader doesn't yet implement sharded obs assembly
        // (that's part of MERGE-OBS-OFFSET-OVERFLOW.md Phase 6b —
        // cloud-native incremental predicate index across appended
        // shards). For now the cloud path stays on the single-section
        // read; cloud files written before Phase 2 are guaranteed
        // single-section, and Phase 2-written sharded files won't open
        // via the cloud path until Phase 6b.
        self.rt
            .block_on(self.inner.read_obs())
            .map_err(cloud_to_engine)
    }

    fn read_var(&self) -> scx_engine::Result<RecordBatch> {
        self.rt
            .block_on(self.inner.read_var())
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
        Ok(scx_format::decode_shard_bytes(
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
