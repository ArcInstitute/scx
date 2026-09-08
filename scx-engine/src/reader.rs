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
use scx_format_io::catalog::FullCatalogEntry;
use scx_format_io::header::FileHeader;
use scx_format_io::reader::ScxReader;
use scx_format_io::DeletionVectors;
use scx_format_io::FullCatalog;

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
    ///
    /// **Memory cost:** on row-sharded (atlas-scale) files this decodes
    /// and concatenates every obs metadata shard. Prefer the streaming
    /// [`Self::obs_metadata_shard_count`] + [`Self::read_obs_shard`] pair
    /// when you can process obs per-shard (the query engine does — see
    /// `scx-engine/src/collect/`).
    fn read_obs(&self) -> Result<RecordBatch>;

    /// Number of `ObsMetadataShard` sections (pure catalog scan, no I/O).
    /// Zero on legacy single-section files.
    fn obs_metadata_shard_count(&self) -> usize;

    /// Read one obs metadata row-shard by index. Returns the per-shard
    /// `RecordBatch` (narrow/canonical types) with its stamped
    /// `row_start` / `n_shard_rows` schema metadata preserved. Bounded by
    /// one shard's worth of obs, regardless of the logical obs size.
    fn read_obs_shard(&self, shard_idx: u32) -> Result<RecordBatch>;

    /// Var metadata as an Arrow RecordBatch.
    fn read_var(&self) -> Result<RecordBatch>;

    // -----------------------------------------------------------------------
    // Modality-aware surface.
    //
    // Default impls model a single-modality (`modality_id = 0`) file, so
    // existing backends (e.g. cloud) compile unchanged and reject non-zero
    // modality ids until they implement per-modality section routing. The
    // local `ScxReader` overrides all of these.
    // -----------------------------------------------------------------------

    /// Number of registered modalities (0 for single-modality / v1 files).
    fn n_modalities(&self) -> u32 {
        0
    }

    /// Whether this file carries a modality table (v2 multimodal).
    fn is_multimodal(&self) -> bool {
        false
    }

    /// Ordered modality names, in registration order. Empty for
    /// single-modality files. Used to render `available: …` in errors.
    fn modality_names(&self) -> Vec<String> {
        Vec::new()
    }

    /// Resolve a modality name to its 1-based `modality_id`, or `None` for
    /// an unknown name / single-modality file.
    fn modality_id_by_name(&self, _name: &str) -> Option<u8> {
        None
    }

    /// Per-modality var schema. `modality_id == 0` is the global /
    /// single-modality `var` section.
    fn read_var_schema_for(&self, modality_id: u8) -> Result<Schema> {
        if modality_id == 0 {
            self.read_var_schema()
        } else {
            Err(crate::error::EngineError::UnknownModality {
                requested: modality_id.to_string(),
                available: self.modality_names(),
            })
        }
    }

    /// Per-modality var RecordBatch. `modality_id == 0` is the global /
    /// single-modality `var` section.
    fn read_var_for(&self, modality_id: u8) -> Result<RecordBatch> {
        if modality_id == 0 {
            self.read_var()
        } else {
            Err(crate::error::EngineError::UnknownModality {
                requested: modality_id.to_string(),
                available: self.modality_names(),
            })
        }
    }

    /// Per-modality variable count. `modality_id == 0` returns the
    /// file-wide `header().n_vars` (single-modality semantics).
    fn modality_n_vars(&self, modality_id: u8) -> Result<u64> {
        if modality_id == 0 {
            Ok(self.header().n_vars)
        } else {
            Err(crate::error::EngineError::UnknownModality {
                requested: modality_id.to_string(),
                available: self.modality_names(),
            })
        }
    }

    /// Raw bytes of the obs predicate index section, if present.
    fn read_obs_predicate_index_bytes(&self) -> Result<Option<Vec<u8>>>;

    /// Raw bytes of the var predicate index section, if present.
    fn read_var_predicate_index_bytes(&self) -> Result<Option<Vec<u8>>>;

    /// Raw bytes of the F1 `group_index` sidecar section, if present.
    /// `None` means the archive was not written with `--group-by`.
    fn read_group_index_bytes(&self) -> Result<Option<Vec<u8>>>;

    /// Deletion vectors, if present.
    fn read_deletion_vectors(&self) -> Result<Option<DeletionVectors>>;

    /// Decode a single CSR shard from its catalog entry into scipy
    /// types (`indptr`, `indices`, `data`).
    fn read_shard_from_entry(
        &self,
        entry: &FullCatalogEntry,
    ) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)>;

    /// Decode a single CSR shard to **native** types (`i64` indptr, `u32`
    /// indices, [`scx_codec::ShardValuesNative`] values): integer-encoded
    /// shards keep their `u32` stream instead of rounding through `f32`.
    ///
    /// Backs the typed (dtype-selected) collect. Deliberately **not**
    /// defaulted: a default would have to either round through `f32` — the
    /// exact loss the typed path exists to avoid — or fail at run time, and
    /// every implementor is in this workspace (the local reader, the cloud
    /// reader, and one fault-injecting test reader), so requiring it costs
    /// nothing and keeps a future backend from landing without it.
    fn read_shard_from_entry_native(
        &self,
        entry: &FullCatalogEntry,
    ) -> Result<(Vec<i64>, Vec<u32>, scx_codec::ShardValuesNative)>;

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

    fn obs_metadata_shard_count(&self) -> usize {
        ScxReader::obs_metadata_shard_count(self)
    }

    fn read_obs_shard(&self, shard_idx: u32) -> Result<RecordBatch> {
        Ok(ScxReader::read_obs_shard(self, shard_idx)?)
    }

    fn read_var(&self) -> Result<RecordBatch> {
        Ok(ScxReader::read_var(self)?)
    }

    fn n_modalities(&self) -> u32 {
        ScxReader::n_modalities(self)
    }

    fn is_multimodal(&self) -> bool {
        ScxReader::is_multimodal(self)
    }

    fn modality_names(&self) -> Vec<String> {
        ScxReader::modality_names(self)
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn modality_id_by_name(&self, name: &str) -> Option<u8> {
        ScxReader::modality_id(self, name)
    }

    fn read_var_schema_for(&self, modality_id: u8) -> Result<Schema> {
        Ok(ScxReader::read_var_schema_for(self, modality_id)?)
    }

    fn read_var_for(&self, modality_id: u8) -> Result<RecordBatch> {
        Ok(ScxReader::read_var_for(self, modality_id)?)
    }

    fn modality_n_vars(&self, modality_id: u8) -> Result<u64> {
        if modality_id == 0 {
            return Ok(ScxReader::header(self).n_vars);
        }
        ScxReader::modality_info(self, modality_id)
            .map(|m| m.n_vars)
            .ok_or_else(|| crate::error::EngineError::UnknownModality {
                requested: modality_id.to_string(),
                available: SectionReader::modality_names(self),
            })
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

    fn read_group_index_bytes(&self) -> Result<Option<Vec<u8>>> {
        Ok(ScxReader::read_group_index_bytes(self)?.map(|s| s.to_vec()))
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

    fn read_shard_from_entry_native(
        &self,
        entry: &FullCatalogEntry,
    ) -> Result<(Vec<i64>, Vec<u32>, scx_codec::ShardValuesNative)> {
        Ok(ScxReader::read_shard_from_entry_native(self, entry)?)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Convenience alias for boxed `SectionReader` trait objects used
/// throughout the query engine.
pub type BoxedSectionReader = Box<dyn SectionReader>;
