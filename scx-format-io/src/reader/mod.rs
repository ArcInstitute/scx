// ScxReader — mmap + pread paths (docs/architecture.md)

use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use std::sync::atomic::AtomicU64;
#[cfg(debug_assertions)]
use std::sync::atomic::Ordering;
use std::sync::Arc;

use arrow::array::RecordBatch;
use memmap2::Mmap;
use scx_codec::{CodecId, ValueEncoding};
use scx_sparse::{ScxCsc, ScxCsr};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::catalog::{FullCatalog, FullCatalogEntry};
use crate::categorical::GlobalCategoryAccum;
use crate::checksum::blake3_hash;
use crate::distinct::DistinctAccumulator;
use crate::error::{Result, ScxError};
use crate::header::{FileHeader, HEADER_SIZE};
use crate::modality::{ModalityInfo, ModalityTable};
use crate::provenance::Provenance;
use crate::section::SectionType;
use crate::shard::ShardHeader;
use crate::RootCatalog;

/// Memory-mapped reader for SCX files.
///
/// Opens an SCX file, validates the header and catalog checksums,
/// and provides methods to read obs/var metadata, CSR shards, layers,
/// obsm embeddings, uns JSON, and provenance.
/// Phase 3c: debug-only call counters used by the streaming-merge test suite
/// to assert that whole-batch materialising paths (`read_layer`, `read_obsm`,
/// `read_all_obsm`, etc.) are not invoked during merge. In release builds the
/// `fetch_add` sites are `cfg(debug_assertions)`-gated and compile away; the
/// (small) struct itself remains so the public accessor stays available across
/// build profiles for cross-crate tests.
#[derive(Default, Debug)]
pub struct ReaderDebugCounts {
    pub read_obs: AtomicU64,
    /// Per-shard obs reads. The streaming query path increments this
    /// instead of `read_obs`; tests assert `read_obs == 0` (no full
    /// materialisation) and that this count stays bounded by the surviving
    /// shards (I/O skip).
    pub read_obs_shard: AtomicU64,
    /// Per-shard **projected** obs reads (`read_obs_shard_projected`), the
    /// column-scoped path behind `read_obs_keys`, `distinct_obs_values`, and
    /// `obs_categorical`.
    ///
    /// Counted separately from `read_obs_shard` so a test can assert the
    /// projected path was *taken*, not merely that the materialising ones were
    /// avoided: `read_obs == 0` alone is satisfied by every column-scoped path
    /// and by several that do far more work, so on its own it is close to
    /// vacuous.
    pub read_obs_shard_projected: AtomicU64,
    /// Per-shard X (CSR) decodes via `read_shard_from_entry[_verified]`. The
    /// query `materialize` path increments this once per decoded shard; tests
    /// assert it stays bounded by the prefix needed to satisfy `.limit(N)`
    /// rather than scaling with the candidate-shard count.
    pub read_shard_from_entry: AtomicU64,
    pub read_layer: AtomicU64,
    pub read_layer_for: AtomicU64,
    pub read_obsm: AtomicU64,
    pub read_all_obsm: AtomicU64,
    pub read_obsm_for: AtomicU64,
    pub read_varm: AtomicU64,
    pub read_all_varm: AtomicU64,
    pub read_varm_for: AtomicU64,
    /// Whole-`adata.raw` materialising reads (`read_all_raw_csr_shards`).
    ///
    /// Paired with `read_shard_from_entry` this is what lets a test tell the
    /// streaming h5ad export of `/raw` from the eager one. Both produce
    /// identical values, so no assertion on the OUTPUT can distinguish them —
    /// only the absence of this call plus the presence of the per-shard ones.
    pub read_all_raw_csr_shards: AtomicU64,
}

pub struct ScxReader {
    mmap: Mmap,
    /// Which inode `mmap` is a mapping of, read from the same `File` that
    /// created it. Kept so a later identity check cannot pair this reader's
    /// catalog with some other file's inode — see `map_with_path_context`.
    inode: crate::freshness::InodeIdentity,
    /// The path this reader was opened from. Kept unconditionally: it is what
    /// [`Self::watching`] stamps against, and it lets errors name the file
    /// without every caller threading the path back in alongside the reader.
    path: std::path::PathBuf,
    /// `Some` only when the caller opted in via [`Self::watching`]. Every
    /// [`Self::section_bytes`] then re-checks that the file at `path` is still
    /// the one that was opened — see [`crate::freshness`]. `None` (the
    /// default) makes the check compile down to a single null test, which is
    /// what keeps `scx-ops` — whose readers deliberately bracket its own
    /// mutations — and the training loader's hot path unaffected.
    freshness: Option<crate::freshness::FreshnessGuard>,
    header: FileHeader,
    root_catalog: RootCatalog,
    /// Stored as `Arc<FullCatalog>` so the same parsed catalog can
    /// back multiple `ScxReader` instances opened against the same
    /// file — see `ScxReader::open_with_shared_catalog` and the
    /// N+3 amplification path in `pyscx::to_anndata_backed`. The Arc
    /// is immutable after construction (`FullCatalog` has no interior
    /// mutability), so sharing across threads and forked workers is
    /// safe without synchronisation.
    full_catalog: Arc<FullCatalog>,
    /// `Some(table)` for v2 multimodal files; `None` for
    /// single-modality v2 files (`n_modalities == 0`) and all v1
    /// files. Parsed lazily-eagerly: the table is parsed once during
    /// `open()` so subsequent `modality_*` accessors are zero-cost.
    modality_table: Option<ModalityTable>,
    /// Phase 3c: per-instance call counters for whole-batch materialising
    /// reader methods (`read_layer`, `read_obsm`, ...). Always present so the
    /// `debug_counts()` accessor is stable across build profiles, but the
    /// increment sites are `cfg(debug_assertions)`-gated.
    debug_counts: ReaderDebugCounts,
}

// `metadata`, not `arrow`: a module named `arrow` inside this crate shadows
// the `arrow` crate for every path in the subtree.
// Gated as a whole rather than item-by-item: every item in it is a deletion-
// vector or detection-bitmap read. Eight of the eleven carried their own
// `#[cfg]` and three did not, so the module's own "feature-gated in its
// entirety" header was false and `--no-default-features` did not build.
#[cfg(feature = "deletion-vectors")]
mod filtered;
mod framed_layout;
mod integrity;
mod matrix;
mod metadata;
mod open;

// `reader::<name>` is an import path in ~60 places across eight crates, and
// five of these are also flat-re-exported from `lib.rs`. The split has to keep
// every one of them resolving, so everything reachable at `reader::<name>`
// before is re-exported here at the same visibility.
pub(crate) use framed_layout::{assemble_row_run, FramedShardLayout};
pub(crate) use matrix::{check_decoded_lengths, plan_row_major_layout, X_LABELS};
pub use metadata::{
    assemble_filtered_metadata, assemble_sharded_metadata, compact_key_shard,
    concat_prepared_metadata_batches, decode_arrow_ipc_schema, filter_batch_by_keep_mask,
    prepare_metadata_batches_for_concat, prune_unused_dictionary_values,
    reconcile_and_share_metadata_batches, scatter_batch_to_physical,
    widen_metadata_batch_for_concat,
};
pub(crate) use metadata::{LegacyRowCount, MappingLayout, MappingShardLayoutEntry};

// Reached from `typed_read.rs` (the typed assembler takes the same strategy) and
// from `reader_tests.rs`, which is attached to this module and so cannot see
// `matrix`'s items directly.
pub(crate) use matrix::RowMajorStrategy;

impl ScxReader {
    /// The inode this reader's mapping was taken from, captured at open time
    /// from the same `File`. The only sound source for a file identity — see
    /// [`crate::freshness::FileIdentity::of`].
    pub(crate) fn inode_identity(&self) -> crate::freshness::InodeIdentity {
        self.inode
    }

    /// The path this reader was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Per-instance counters for whole-batch reader entry points
    /// (`read_layer`, `read_obsm`, ...). Increments are
    /// `cfg(debug_assertions)`-gated and compile away in release builds —
    /// the test suite uses these to assert that streaming merge / append
    /// never reaches a materialising read path.
    pub fn debug_counts(&self) -> &ReaderDebugCounts {
        &self.debug_counts
    }

    pub fn header(&self) -> &FileHeader {
        &self.header
    }

    pub fn root_catalog(&self) -> &RootCatalog {
        &self.root_catalog
    }

    pub fn catalog(&self) -> &FullCatalog {
        self.full_catalog.as_ref()
    }

    /// Resolve the full catalog entry for a section at byte `offset` of the
    /// given type. The backed reader's `ShardEntryLite` table deliberately
    /// drops the section `name` and `checksum` to shrink its per-shard
    /// footprint, but sidecar resolution needs them (`decode/<name>` lookup +
    /// the freshness `checksum`), so the row-range fast path recovers the real
    /// entry through this accessor. Offsets are unique across sections, so the
    /// `(offset, section_type)` match is unambiguous. Linear scan — callers
    /// gate it to the small-window path where it is negligible against decode.
    pub fn full_entry_at_offset(
        &self,
        offset: u64,
        section_type: SectionType,
    ) -> Option<&FullCatalogEntry> {
        self.full_catalog
            .entries
            .iter()
            .find(|e| e.offset == offset && e.section_type == section_type)
    }

    /// Clone the internal `Arc<FullCatalog>` for cheap reuse across
    /// sibling `ScxReader` instances opened with
    /// [`open_with_shared_catalog`](Self::open_with_shared_catalog).
    /// Cloning an `Arc` is one atomic refcount bump — the catalog
    /// itself is not copied.
    pub fn catalog_arc(&self) -> Arc<FullCatalog> {
        Arc::clone(&self.full_catalog)
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

    /// Number of registered modalities. Returns `0` for v1 files and
    /// single-modality v2 files (semantically equivalent).
    pub fn n_modalities(&self) -> u32 {
        self.header.n_modalities
    }

    /// Returns true when the file has a `ModalityTable` section
    /// (v2 multimodal). Mirrors `header.has_modalities()`.
    pub fn is_multimodal(&self) -> bool {
        self.modality_table.is_some()
    }

    /// Returns the ordered list of modality names, in registration
    /// order. Empty for single-modality files.
    pub fn modality_names(&self) -> Vec<&str> {
        self.modality_table
            .as_ref()
            .map(|t| t.entries.iter().map(|m| m.name.as_str()).collect())
            .unwrap_or_default()
    }

    /// Resolve a modality name to its 1-based `modality_id`. Returns
    /// `None` for unknown names or for single-modality files.
    pub fn modality_id(&self, name: &str) -> Option<u8> {
        self.modality_table.as_ref().and_then(|t| t.id_of(name))
    }

    /// Look up modality metadata by 1-based id. Returns `None` for
    /// `id == 0` (global) and for single-modality files.
    pub fn modality_info(&self, modality_id: u8) -> Option<&ModalityInfo> {
        self.modality_table
            .as_ref()
            .and_then(|t| t.info_of(modality_id))
    }

    /// Returns the parsed `ModalityTable`, or `None` for
    /// single-modality files. Useful for tooling that wants to walk
    /// the table directly (e.g. `scx info`).
    pub fn modality_table(&self) -> Option<&ModalityTable> {
        self.modality_table.as_ref()
    }

    fn modality_name_for_id(&self, modality_id: u8) -> Result<String> {
        if modality_id == 0 {
            return Err(ScxError::InvalidCatalog(
                "modality_id 0 is reserved for global entries".to_string(),
            ));
        }
        self.modality_info(modality_id)
            .map(|m| m.name.clone())
            .ok_or_else(|| ScxError::InvalidCatalog(format!("modality_id {modality_id} not found")))
    }

    /// Read the uns (unstructured) section as a JSON value.
    pub fn read_uns(&self) -> Result<serde_json::Value> {
        let entry = self
            .full_catalog
            .get("uns")
            .ok_or_else(|| ScxError::SectionNotFound("uns".to_string()))?;
        let slice = self.section_bytes(entry)?;
        scx_format::parse_uns_json(slice)
    }

    /// Read the provenance section.
    pub fn read_provenance(&self) -> Result<Provenance> {
        let entry = self
            .full_catalog
            .get("provenance")
            .ok_or_else(|| ScxError::SectionNotFound("provenance".to_string()))?;
        let slice = self.section_bytes(entry)?;
        Provenance::read_from(&mut Cursor::new(slice), slice.len())
    }

    /// Read the raw bytes of the obs predicate index section, if present.
    /// Returns `Ok(None)` if the file contains no obs predicate index.
    pub fn read_obs_predicate_index_bytes(&self) -> Result<Option<&[u8]>> {
        let entry = self
            .full_catalog
            .entries
            .iter()
            .find(|e| e.section_type == SectionType::ObsPredicateIndex);
        match entry {
            Some(e) => Ok(Some(self.section_bytes(e)?)),
            None => Ok(None),
        }
    }

    /// Read the raw bytes of the var predicate index section, if present.
    /// Returns `Ok(None)` if the file contains no var predicate index.
    pub fn read_var_predicate_index_bytes(&self) -> Result<Option<&[u8]>> {
        let entry = self
            .full_catalog
            .entries
            .iter()
            .find(|e| e.section_type == SectionType::VarPredicateIndex);
        match entry {
            Some(e) => Ok(Some(self.section_bytes(e)?)),
            None => Ok(None),
        }
    }

    /// Raw bytes of the F1 `group_index` sidecar section, if present.
    pub fn read_group_index_bytes(&self) -> Result<Option<&[u8]>> {
        let entry = self
            .full_catalog
            .entries
            .iter()
            .find(|e| e.section_type == SectionType::GroupIndex);
        match entry {
            Some(e) => Ok(Some(self.section_bytes(e)?)),
            None => Ok(None),
        }
    }

    /// Get access to the underlying mmap bytes.
    pub fn mmap(&self) -> &[u8] {
        &self.mmap
    }

    /// Expose the underlying `Mmap` for range-specific madvise calls.
    pub(crate) fn mmap_ref(&self) -> &Mmap {
        &self.mmap
    }

    /// Hint sequential access for a byte range (`MADV_SEQUENTIAL`).
    #[cfg(unix)]
    pub(crate) fn advise_sequential(&self, offset: usize, len: usize) {
        use memmap2::Advice;
        let _ = self.mmap.advise_range(Advice::Sequential, offset, len);
    }

    /// Hint that a byte range will be needed soon (`MADV_WILLNEED`).
    ///
    /// Used by the training loader to prefetch upcoming shard byte ranges.
    #[cfg(unix)]
    pub fn advise_willneed(&self, offset: usize, len: usize) {
        use memmap2::Advice;
        let _ = self.mmap.advise_range(Advice::WillNeed, offset, len);
    }

    /// Get the raw bytes for a catalog entry from the mmap.
    ///
    /// The single point at which section payload bytes leave the mapping —
    /// every other read in the workspace, in this crate and outside it, comes
    /// through here. That is why the freshness check lives at this line rather
    /// than at the ~230 handle methods above it: a read path added tomorrow
    /// inherits it without anyone remembering to ask.
    pub fn section_bytes(&self, entry: &FullCatalogEntry) -> Result<&[u8]> {
        self.check_fresh()?;
        let start = entry.offset as usize;
        let end = start
            .checked_add(entry.length as usize)
            .ok_or(ScxError::SectionOutOfBounds {
                offset: entry.offset,
                length: entry.length,
                file_size: self.mmap.len(),
            })?;
        if end > self.mmap.len() {
            return Err(ScxError::SectionOutOfBounds {
                offset: entry.offset,
                length: entry.length,
                file_size: self.mmap.len(),
            });
        }
        Ok(&self.mmap[start..end])
    }
}

/// Convert raw value bytes to f32 according to the value encoding.
#[cfg(test)]
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
#[path = "../reader_tests.rs"]
mod tests;
