//! `SectionReader` — the I/O surface `QueryPipeline` runs against.
//!
//! Abstracts over the local mmap-backed `ScxReader` (`scx-format`) and
//! cloud `CloudSectionReader` (`scx-cloud`) so the query engine can
//! drive either without knowing which it has. The trait is sync —
//! cloud implementations drive async object-store I/O via `block_on`
//! inside their methods.
//!
//! See `docs/architecture.md` § `SectionReader` for the dependency
//! graph and `docs/cloud.md` § Cloud-native query for the motivating
//! cloud-native query path.

use std::any::Any;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use scx_format::catalog::FullCatalogEntry;
use scx_format::header::FileHeader;
use scx_format::reader::ScxReader;
use scx_format::DeletionVectors;
use scx_format::FullCatalog;

use crate::error::Result;

/// Read SCX sections in whatever way the underlying storage supports.
///
/// Implementors guarantee:
///  - `header()` and `catalog()` are O(1) and parsed up front
///    (no I/O at call time).
///  - All `read_*` methods may perform I/O each time they are called;
///    callers should cache the result if invoked more than once.
///  - Methods returning owned `Vec<u8>` / `RecordBatch` may allocate
///    even when the underlying storage is mmap-backed (the local
///    `ScxReader` impl copies the predicate-index slice into a Vec to
///    satisfy the owned return type — predicate indexes are small and
///    read once per query, so the cost is negligible).
pub trait SectionReader: Send + Sync {
    /// File header (in-memory, cached at open time).
    fn header(&self) -> &FileHeader;

    /// Full catalog (in-memory, cached at open time).
    fn catalog(&self) -> &FullCatalog;

    /// Obs schema without materialising the full RecordBatch.
    fn read_obs_schema(&self) -> Result<Schema>;

    /// Var schema without materialising the full RecordBatch.
    fn read_var_schema(&self) -> Result<Schema>;

    /// Obs metadata as an Arrow RecordBatch.
    fn read_obs(&self) -> Result<RecordBatch>;

    /// Var metadata as an Arrow RecordBatch.
    fn read_var(&self) -> Result<RecordBatch>;

    /// Raw bytes of the obs predicate index section, if present.
    fn read_obs_predicate_index_bytes(&self) -> Result<Option<Vec<u8>>>;

    /// Raw bytes of the var predicate index section, if present.
    fn read_var_predicate_index_bytes(&self) -> Result<Option<Vec<u8>>>;

    /// Deletion vectors, if present.
    fn read_deletion_vectors(&self) -> Result<Option<DeletionVectors>>;

    /// Decode a single CSR shard from its catalog entry into scipy
    /// types (`indptr`, `indices`, `data`).
    fn read_shard_from_entry(
        &self,
        entry: &FullCatalogEntry,
    ) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)>;

    /// Downcast escape hatch for callers that need backend-specific
    /// methods not on the trait (e.g. `scx subset` reading uns / layer
    /// names from a local `ScxReader`). Cloud-only callers do not use
    /// this.
    fn as_any(&self) -> &dyn Any;
}

impl SectionReader for ScxReader {
    fn header(&self) -> &FileHeader {
        ScxReader::header(self)
    }

    fn catalog(&self) -> &FullCatalog {
        ScxReader::catalog(self)
    }

    fn read_obs_schema(&self) -> Result<Schema> {
        Ok(ScxReader::read_obs_schema(self)?)
    }

    fn read_var_schema(&self) -> Result<Schema> {
        Ok(ScxReader::read_var_schema(self)?)
    }

    fn read_obs(&self) -> Result<RecordBatch> {
        Ok(ScxReader::read_obs(self)?)
    }

    fn read_var(&self) -> Result<RecordBatch> {
        Ok(ScxReader::read_var(self)?)
    }

    fn read_obs_predicate_index_bytes(&self) -> Result<Option<Vec<u8>>> {
        // ScxReader returns an mmap slice; copy into an owned Vec to
        // satisfy the trait. Predicate indexes are small (typically a
        // few KB) and read once per query, so the extra allocation is
        // not a hot path concern.
        Ok(ScxReader::read_obs_predicate_index_bytes(self)?.map(|s| s.to_vec()))
    }

    fn read_var_predicate_index_bytes(&self) -> Result<Option<Vec<u8>>> {
        Ok(ScxReader::read_var_predicate_index_bytes(self)?.map(|s| s.to_vec()))
    }

    fn read_deletion_vectors(&self) -> Result<Option<DeletionVectors>> {
        Ok(ScxReader::read_deletion_vectors(self)?)
    }

    fn read_shard_from_entry(
        &self,
        entry: &FullCatalogEntry,
    ) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
        Ok(ScxReader::read_shard_from_entry(self, entry)?)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Convenience alias for boxed `SectionReader` trait objects used
/// throughout the query engine.
pub type BoxedSectionReader = Box<dyn SectionReader>;
