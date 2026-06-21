// ScxReader — mmap + pread paths (docs/architecture.md)

use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::atomic::AtomicU64;
#[cfg(debug_assertions)]
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use arrow::array::RecordBatch;
use lru::LruCache;
use memmap2::Mmap;
use scx_codec::{CodecId, ValueEncoding};
use scx_sparse::{ScxCsc, ScxCsr};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::catalog::{FullCatalog, FullCatalogEntry};
use crate::checksum::blake3_hash;
use crate::decode_sidecar::DecodeSidecar;
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
    /// Per-shard X (CSR) decodes via `read_shard_from_entry[_verified]`. The
    /// query `materialize` path increments this once per decoded shard; tests
    /// assert it stays bounded by the prefix needed to satisfy `.limit(N)`
    /// rather than scaling with the candidate-shard count.
    pub read_shard_from_entry: AtomicU64,
    /// Per-shard sidecar-driven row-range decodes via
    /// [`ScxReader::decode_scx1_row_range`] that actually resolved a fresh Scx1
    /// sidecar (the `Some` path). Backed-reader tests assert this is nonzero to
    /// prove the row-range fast path was taken rather than silently falling
    /// back to a full-shard decode.
    pub decode_scx1_row_range: AtomicU64,
    /// Number of times the per-shard Scx1 sidecar metadata was actually parsed
    /// (sidecar read + `to_scx1_metadata`) rather than served from
    /// `sidecar_meta_cache` (lever S). Tests assert this stays at 1 across
    /// repeated touches of one shard to prove the cache eliminates the
    /// per-batch O(shard-rows) reparse.
    pub sidecar_meta_parse: AtomicU64,
    pub read_layer: AtomicU64,
    pub read_layer_for: AtomicU64,
    pub read_obsm: AtomicU64,
    pub read_all_obsm: AtomicU64,
    pub read_obsm_for: AtomicU64,
    pub read_varm: AtomicU64,
    pub read_all_varm: AtomicU64,
    pub read_varm_for: AtomicU64,
}

pub struct ScxReader {
    mmap: Mmap,
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
    /// Per-shard Scx1 decode-sidecar metadata cache (lever S). Keyed by the
    /// CSR shard's `entry.offset`
    /// (unique per shard — see [`Self::full_entry_at_offset`]) → the parsed
    /// [`scx_codec::Scx1DecodeMetadata`]. The scattered cell-set gather touches
    /// one shard with many small runs across many batches; without this every
    /// touch re-ran [`DecodeSidecar::read_from`] (BLAKE3 over the whole sidecar)
    /// and `to_scx1_metadata()` (a second O(shard-rows) clone). Memoizing the
    /// parsed metadata makes those O(rows) — byte-identical, since the sidecar
    /// is a deterministic function of the read-only mmap bytes. `Mutex` gives
    /// interior mutability (mirrors the `SharedShardCache` Mutex in `backed.rs`);
    /// per-instance, so fork-safe; bounded LRU so a many-shard file stays
    /// memory-capped.
    sidecar_meta_cache: Mutex<LruCache<u64, Arc<scx_codec::Scx1DecodeMetadata>>>,
}

/// Bound on the per-reader sidecar-metadata LRU (lever S). Parsed metadata is
/// ~0.3–0.5 MB per 16k-row shard, so this caps the cache near ~0.5 GB worst
/// case; a per-file reader rarely holds more than a few dozen shards.
const SIDECAR_META_CACHE_CAP: usize = 1024;

impl ScxReader {
    /// Open an SCX file for reading.
    ///
    /// Validates the file header magic/version/endianness and the full
    /// catalog's trailing BLAKE3 checksum.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_inner(path, true)
    }

    /// Open an SCX file without verifying the catalog checksum.
    ///
    /// Skips the full catalog BLAKE3 verification for performance-sensitive
    /// paths where the file is trusted (e.g., `scx info`, repeated reads of
    /// a file that was already validated). The header magic/version/endianness
    /// are still checked.
    pub fn open_unchecked(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_inner(path, false)
    }

    fn open_inner(path: impl AsRef<Path>, verify_catalog: bool) -> Result<Self> {
        let path = path.as_ref();
        // Echo the offending path in the error — a bare "No such file or
        // directory (os error 2)" forces the caller to guess which file failed.
        let file = File::open(path).map_err(|e| {
            ScxError::Io(std::io::Error::new(
                e.kind(),
                format!("cannot open '{}': {}", path.display(), e),
            ))
        })?;
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| {
                ScxError::Io(std::io::Error::new(
                    e.kind(),
                    format!("cannot mmap '{}': {}", path.display(), e),
                ))
            })?
        };

        // Start with Normal advice (kernel default heuristic). Access-pattern-
        // specific hints (Sequential, WillNeed, DontNeed) are issued at each
        // call site — see assemble_shards*() and BackedCsrReader.
        #[cfg(unix)]
        {
            use memmap2::Advice;
            let _ = mmap.advise(Advice::Normal);
        }

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
        let fc_end = fc_offset
            .checked_add(fc_length)
            .ok_or(ScxError::SectionOutOfBounds {
                offset: header.full_catalog_offset,
                length: header.full_catalog_length,
                file_size: mmap.len(),
            })?;
        if fc_end > mmap.len() {
            return Err(ScxError::SectionOutOfBounds {
                offset: header.full_catalog_offset,
                length: header.full_catalog_length,
                file_size: mmap.len(),
            });
        }
        let fc_slice = &mmap[fc_offset..fc_end];
        let mut full_catalog =
            FullCatalog::read_from(&mut Cursor::new(fc_slice), fc_length, verify_catalog)?;

        // v1 → v2 reconciliation: populate `col_start`/`col_end` for
        // row-major shard entries from `n_vars`. CSC entries are
        // already reconciled inside `FullCatalog::read_from`. No-op on
        // v2 catalogs.
        full_catalog.reconcile_v1_csr_col_range(header.n_vars);

        // Phase B: parse the ModalityTable section if the header
        // points at one. The pointer is `0/0` for single-modality
        // files (legacy shape and v1 files alike).
        let modality_table = if header.n_modalities > 0
            && header.modality_table_offset != 0
            && header.modality_table_length != 0
        {
            let mt_off = header.modality_table_offset as usize;
            let mt_len = header.modality_table_length as usize;
            let mt_end = mt_off
                .checked_add(mt_len)
                .ok_or(ScxError::SectionOutOfBounds {
                    offset: header.modality_table_offset,
                    length: header.modality_table_length,
                    file_size: mmap.len(),
                })?;
            if mt_end > mmap.len() {
                return Err(ScxError::SectionOutOfBounds {
                    offset: header.modality_table_offset,
                    length: header.modality_table_length,
                    file_size: mmap.len(),
                });
            }
            let mt_slice = &mmap[mt_off..mt_end];
            let table = ModalityTable::read_from(&mut Cursor::new(mt_slice), mt_len)?;
            // Cross-check header.n_modalities against the table's
            // embedded count. Disagreement is corruption, not a v1/v2
            // mismatch.
            if table.len() as u32 != header.n_modalities {
                return Err(ScxError::InvalidCatalog(format!(
                    "header.n_modalities ({}) != ModalityTable.len() ({})",
                    header.n_modalities,
                    table.len()
                )));
            }
            Some(table)
        } else {
            None
        };

        Ok(ScxReader {
            mmap,
            header,
            root_catalog,
            full_catalog: Arc::new(full_catalog),
            modality_table,
            debug_counts: ReaderDebugCounts::default(),
            sidecar_meta_cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(SIDECAR_META_CACHE_CAP).unwrap(),
            )),
        })
    }

    /// Open an SCX file reusing an already-parsed `FullCatalog` from a
    /// sibling `ScxReader` against the same file. Skips
    /// `FullCatalog::read_from` entirely — the expensive part of
    /// `open()` for files with thousands of catalog entries.
    ///
    /// Worker-amplification use case: `to_anndata_backed` opens N+3
    /// `ScxReader` instances per call (main reader + X CSR + CSC
    /// sidecar + N backed layers). The catalog is identical
    /// bytes-for-bytes across all of them; sharing one parsed copy
    /// collapses the per-call parse cost from `(N+3) ×` to `1×` on
    /// the worker construction path that `cell-load-scx` /
    /// `state-scx` hit inside their DataLoader iterators.
    ///
    /// The header magic / version / endianness and the root catalog
    /// at offset 256 are still validated against the fresh mmap, and
    /// the fresh header's `manifest_sequence` is compared against the
    /// shared catalog's — any divergence means the file was mutated
    /// between opens (`scx-ops` append / compact / rollback bumps the
    /// sequence) and the shared catalog no longer describes this mmap.
    /// The function does not re-verify the trailing BLAKE3 catalog
    /// checksum: the source `ScxReader` already did that at its own
    /// open, and the sequence check covers the mutation case the
    /// checksum would otherwise have to catch.
    ///
    /// # Fork safety
    ///
    /// `Arc<FullCatalog>` is `Send + Sync` and has no interior
    /// mutability. When pyscx's worker iterators fork after the
    /// parent has constructed a `BackedCsrReader` ladder, the
    /// shared catalog is COW-duplicated into each child — no locks,
    /// no mutexes, no shared mutable state.
    pub fn open_with_shared_catalog(
        path: impl AsRef<Path>,
        catalog: Arc<FullCatalog>,
    ) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };

        #[cfg(unix)]
        {
            use memmap2::Advice;
            let _ = mmap.advise(Advice::Normal);
        }

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

        let header = FileHeader::read_from(&mut Cursor::new(&mmap[..HEADER_SIZE]))?;
        if header.manifest_sequence != catalog.manifest_sequence {
            return Err(ScxError::InvalidCatalog(format!(
                "shared catalog manifest_sequence ({}) does not match file header ({}) — \
                 file was mutated between opens",
                catalog.manifest_sequence, header.manifest_sequence,
            )));
        }
        let root_catalog = RootCatalog::read_from(&mut Cursor::new(&mmap[HEADER_SIZE..]))?;

        // Re-parse the (small) modality table from this instance's mmap.
        // It's only present on multimodal v2 files and is small; the
        // parse cost is negligible vs the full catalog.
        let modality_table = if header.n_modalities > 0
            && header.modality_table_offset != 0
            && header.modality_table_length != 0
        {
            let mt_off = header.modality_table_offset as usize;
            let mt_len = header.modality_table_length as usize;
            let mt_end = mt_off
                .checked_add(mt_len)
                .ok_or(ScxError::SectionOutOfBounds {
                    offset: header.modality_table_offset,
                    length: header.modality_table_length,
                    file_size: mmap.len(),
                })?;
            if mt_end > mmap.len() {
                return Err(ScxError::SectionOutOfBounds {
                    offset: header.modality_table_offset,
                    length: header.modality_table_length,
                    file_size: mmap.len(),
                });
            }
            let mt_slice = &mmap[mt_off..mt_end];
            let table = ModalityTable::read_from(&mut Cursor::new(mt_slice), mt_len)?;
            if table.len() as u32 != header.n_modalities {
                return Err(ScxError::InvalidCatalog(format!(
                    "header.n_modalities ({}) != ModalityTable.len() ({})",
                    header.n_modalities,
                    table.len()
                )));
            }
            Some(table)
        } else {
            None
        };

        Ok(ScxReader {
            mmap,
            header,
            root_catalog,
            full_catalog: catalog,
            modality_table,
            debug_counts: ReaderDebugCounts::default(),
            sidecar_meta_cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(SIDECAR_META_CACHE_CAP).unwrap(),
            )),
        })
    }

    /// Per-instance counters for whole-batch reader entry points
    /// (`read_layer`, `read_obsm`, ...). Increments are
    /// `cfg(debug_assertions)`-gated and compile away in release builds —
    /// the test suite uses these to assert that streaming merge / append
    /// never reaches a materialising read path.
    pub fn debug_counts(&self) -> &ReaderDebugCounts {
        &self.debug_counts
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

    // -----------------------------------------------------------------------
    // Arrow IPC reading (11.5–11.8)
    // -----------------------------------------------------------------------

    /// Read the Arrow IPC schema from a catalog entry.
    ///
    /// Stays in lockstep with [`Self::read_arrow_ipc`] under the
    /// opportunistic downcast in [`crate::arrow_compat`]: the
    /// canonical schema depends on whether columns' actual offsets fit
    /// back in `i32`, which can only be determined by inspecting the
    /// data. So:
    ///
    /// - **Fast path** (no `LargeUtf8` / `LargeBinary` /
    ///   `Dictionary(_, Large*)` on disk): return the IPC footer
    ///   schema directly. No data deserialization. ~KB of work.
    /// - **Slow path** (any wide type on disk): re-read the first
    ///   batch and run `downcast_large_types` so the returned schema
    ///   matches what `read_arrow_ipc` would produce — narrow types
    ///   when offsets fit, wide types when they overflow.
    fn read_arrow_ipc_schema(&self, entry: &FullCatalogEntry) -> Result<arrow::datatypes::Schema> {
        decode_arrow_ipc_schema(self.section_bytes(entry)?)
    }

    /// Read an Arrow IPC section from a catalog entry.
    ///
    /// Downcasts `LargeUtf8 → Utf8` / `LargeBinary → Binary` so callers
    /// always see canonical narrow types regardless of the on-disk
    /// encoding (see [`crate::arrow_compat`]).
    fn read_arrow_ipc(&self, entry: &FullCatalogEntry) -> Result<RecordBatch> {
        let batch = self.read_arrow_ipc_raw(entry)?;
        crate::arrow_compat::downcast_large_types(&batch)
    }

    /// Decode a single Arrow IPC entry **without** the wide→narrow
    /// downcast. The writer always upcasts to `LargeUtf8`/`LargeBinary`
    /// before serialising (so the bytes-on-disk are typically wide), but
    /// columns whose payload fits in narrow offsets may still come back
    /// downcast-eligible. This helper preserves the on-disk encoding so
    /// callers that need to concatenate batches across shards can defer
    /// the narrow choice until after [`arrow::compute::concat_batches`]
    /// — concatenating on narrow offsets reproduces the original
    /// `Offset overflow error` once the combined per-column string
    /// payload exceeds `i32::MAX` (the same failure mode the streaming
    /// merge-write path eliminated).
    fn read_arrow_ipc_raw(&self, entry: &FullCatalogEntry) -> Result<RecordBatch> {
        let slice = self.section_bytes(entry)?;
        let cursor = Cursor::new(slice);
        let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)?;
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

    /// Phase 3b: decode a single Arrow IPC dense-mapping section by
    /// catalog entry, applying the same wide→narrow downcast as the
    /// whole-batch readers. Public counterpart to
    /// [`Self::read_shard_from_entry`] (which targets encoded CSR
    /// shards): used by the streaming merge path to walk
    /// `Obsm/VarmEmbeddingShard` (and their legacy single-section
    /// counterparts) one shard at a time without going through
    /// `read_obsm` / `read_varm` (which reassemble the full mapping).
    pub fn read_dense_mapping_entry(&self, entry: &FullCatalogEntry) -> Result<RecordBatch> {
        self.read_arrow_ipc(entry)
    }

    /// Resolve the per-shard physical layout of a row-sharded dense
    /// mapping (e.g. `obsm/<name>`) for the backed dense row-gather
    /// reader ([`crate::BackedDenseReader`]).
    ///
    /// Unlike CSR shards, `ObsmEmbeddingShard` catalog entries carry no
    /// `stats` block, so the per-shard row ranges live only in each
    /// shard's Arrow schema metadata (`row_start` / `n_shard_rows` /
    /// `n_rows_total`, stamped by `writer::stamp_dense_shard_metadata`).
    /// We read each shard's IPC **footer schema only** (no batch
    /// deserialisation) and validate a contiguous, ordered cover with
    /// the same invariant as [`assemble_sharded_metadata`].
    ///
    /// Falls back to the legacy single-section layout (`single_type`)
    /// treated as one shard spanning `[0, num_rows)` — that path
    /// deserialises the one batch to learn its row count.
    pub(crate) fn dense_mapping_layout(
        &self,
        prefix: &str,
        name: &str,
        shard_type: SectionType,
        single_type: SectionType,
    ) -> Result<DenseMappingLayout> {
        let shard_name_prefix = format!("{prefix}/{name}_shard_");
        let logical = format!("{prefix}/{name}");

        let mut shards: Vec<(u32, &FullCatalogEntry)> = self
            .full_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == shard_type && e.name.starts_with(&shard_name_prefix))
            .filter_map(|e| {
                let suffix = e.name.strip_prefix(&shard_name_prefix)?;
                let idx: u32 = suffix.parse().ok()?;
                Some((idx, e))
            })
            .collect();

        if !shards.is_empty() {
            shards.sort_by_key(|(idx, _)| *idx);
            let mut entries: Vec<DenseShardLayoutEntry> = Vec::with_capacity(shards.len());
            let mut n_cols = 0usize;
            let mut dtype = arrow::datatypes::DataType::Float32;
            let mut fields: arrow::datatypes::Fields = Default::default();
            let mut prev_n_rows_total = 0u64;
            let mut next_expected_row_start = 0u64;

            for (i, (idx, entry)) in shards.iter().enumerate() {
                let schema = self.read_arrow_ipc_schema_physical(entry)?;
                let hdr = parse_shard_metadata_md(&logical, schema.metadata())?;
                let expected_idx = i as u32;
                if hdr.shard_idx != expected_idx {
                    return Err(ScxError::InvalidCatalog(format!(
                        "{logical}: shard at position {i} has shard_idx={} (expected {expected_idx})",
                        hdr.shard_idx
                    )));
                }
                if i == 0 {
                    if hdr.row_start != 0 {
                        return Err(ScxError::InvalidCatalog(format!(
                            "{logical}: first shard has row_start={} (expected 0)",
                            hdr.row_start
                        )));
                    }
                    n_cols = schema.fields().len();
                    if n_cols > 0 {
                        dtype = schema.field(0).data_type().clone();
                    }
                    fields = schema.fields().clone();
                    prev_n_rows_total = hdr.n_rows_total;
                    next_expected_row_start = hdr.n_shard_rows;
                } else {
                    if hdr.n_rows_total < prev_n_rows_total {
                        return Err(ScxError::InvalidCatalog(format!(
                            "{logical}: shard {i} has n_rows_total={} which contracts the prior \
                             shard's stamp of {prev_n_rows_total}",
                            hdr.n_rows_total
                        )));
                    }
                    if hdr.row_start != next_expected_row_start {
                        return Err(ScxError::InvalidCatalog(format!(
                            "{logical}: shard {i} has row_start={} (expected {next_expected_row_start})",
                            hdr.row_start
                        )));
                    }
                    next_expected_row_start =
                        next_expected_row_start.saturating_add(hdr.n_shard_rows);
                    prev_n_rows_total = hdr.n_rows_total;
                }
                let _ = idx;
                entries.push(DenseShardLayoutEntry {
                    offset: entry.offset,
                    length: entry.length,
                    section_type: entry.section_type,
                    modality_id: entry.modality_id,
                    row_start: hdr.row_start,
                    n_shard_rows: hdr.n_shard_rows,
                });
            }
            if next_expected_row_start != prev_n_rows_total {
                return Err(ScxError::InvalidCatalog(format!(
                    "{logical}: shards cover {next_expected_row_start} rows but the last shard's \
                     n_rows_total is {prev_n_rows_total}"
                )));
            }
            return Ok(DenseMappingLayout {
                entries,
                n_rows: prev_n_rows_total,
                n_cols,
                dtype,
                fields,
            });
        }

        // Legacy single section — one batch, no shard metadata.
        let entry = self
            .full_catalog
            .get(&logical)
            .filter(|e| e.section_type == single_type)
            .ok_or_else(|| ScxError::SectionNotFound(logical.clone()))?;
        let batch = self.read_arrow_ipc(entry)?;
        let n_rows = batch.num_rows() as u64;
        let n_cols = batch.num_columns();
        let dtype = if n_cols > 0 {
            batch.column(0).data_type().clone()
        } else {
            arrow::datatypes::DataType::Float32
        };
        let fields = batch.schema_ref().fields().clone();
        Ok(DenseMappingLayout {
            entries: vec![DenseShardLayoutEntry {
                offset: entry.offset,
                length: entry.length,
                section_type: entry.section_type,
                modality_id: entry.modality_id,
                row_start: 0,
                n_shard_rows: n_rows,
            }],
            n_rows,
            n_cols,
            dtype,
            fields,
        })
    }

    /// Read the obs schema. Uses the Arrow IPC footer fast path, falling
    /// back to a per-column wide-vs-narrow refinement (via
    /// [`crate::arrow_compat::downcast_large_types`]) when any column on
    /// disk is `LargeUtf8` / `LargeBinary` / `Dictionary(_, Large*)`.
    /// For atlas-scale obs that legitimately remain wide on disk, the
    /// refinement deserialises the first batch — for sharded files this
    /// is bounded to one shard rather than the whole obs table, but is
    /// still O(MB). Use [`Self::read_obs_schema_physical`] (no
    /// deserialisation) or [`Self::read_obs_schema_logical_lossy`]
    /// (unconditional schema-level narrowing) for the cheap paths.
    pub fn read_obs_schema(&self) -> Result<arrow::datatypes::Schema> {
        let entry = self.first_obs_section_entry()?;
        self.read_arrow_ipc_schema(entry)
    }

    /// Read the var schema. Mirror of [`Self::read_obs_schema`].
    pub fn read_var_schema(&self) -> Result<arrow::datatypes::Schema> {
        let entry = self.first_var_section_entry()?;
        self.read_arrow_ipc_schema(entry)
    }

    /// Return the obs schema **exactly as stored on disk** — including
    /// any `LargeUtf8` / `LargeBinary` columns left wide for >2 GB
    /// payloads. Pure Arrow IPC footer read; no batch deserialisation.
    /// Constant cost regardless of obs size.
    ///
    /// Use this when your code can handle wide types and you want a
    /// faithful picture of what the writer emitted. Pair with
    /// [`Self::read_obs_schema_logical_lossy`] if you'd rather always
    /// see narrow types and don't mind the lossy conversion.
    pub fn read_obs_schema_physical(&self) -> Result<arrow::datatypes::Schema> {
        let entry = self.first_obs_section_entry()?;
        self.read_arrow_ipc_schema_physical(entry)
    }

    /// Read var schema as on disk. Mirror of
    /// [`Self::read_obs_schema_physical`].
    pub fn read_var_schema_physical(&self) -> Result<arrow::datatypes::Schema> {
        let entry = self.first_var_section_entry()?;
        self.read_arrow_ipc_schema_physical(entry)
    }

    /// Return the obs schema with `LargeUtf8 → Utf8` / `LargeBinary →
    /// Binary` (and `Dictionary` variants) **unconditionally narrowed**
    /// at the schema level. Pure Arrow IPC footer read; no batch
    /// deserialisation. Lossy in the technical sense — a column the
    /// reader reports as `Utf8` may, on the data path, still come back
    /// as `LargeUtf8` if its actual offsets overflow `i32::MAX`. The
    /// trade-off is constant-cost schema reads for predicate parsing
    /// and validation paths that prefer the historical narrow types.
    pub fn read_obs_schema_logical_lossy(&self) -> Result<arrow::datatypes::Schema> {
        let physical = self.read_obs_schema_physical()?;
        Ok(crate::arrow_compat::downcast_large_types_schema(&physical))
    }

    /// Lossy logical schema for var. Mirror of
    /// [`Self::read_obs_schema_logical_lossy`].
    pub fn read_var_schema_logical_lossy(&self) -> Result<arrow::datatypes::Schema> {
        let physical = self.read_var_schema_physical()?;
        Ok(crate::arrow_compat::downcast_large_types_schema(&physical))
    }

    /// Resolve the first obs section catalog entry: shard 0 if obs is
    /// sharded, else the legacy single section. Used by both schema
    /// APIs and the assembled-batch fallback.
    fn first_obs_section_entry(&self) -> Result<&FullCatalogEntry> {
        if self.obs_metadata_shard_count() > 0 {
            let key = "obs_metadata/shard_0";
            self.full_catalog
                .get(key)
                .ok_or_else(|| ScxError::SectionNotFound(key.to_string()))
        } else {
            self.full_catalog
                .get("obs")
                .ok_or_else(|| ScxError::SectionNotFound("obs".to_string()))
        }
    }

    /// Mirror of [`Self::first_obs_section_entry`] for var.
    fn first_var_section_entry(&self) -> Result<&FullCatalogEntry> {
        if self.var_metadata_shard_count() > 0 {
            let key = "var_metadata/shard_0";
            self.full_catalog
                .get(key)
                .ok_or_else(|| ScxError::SectionNotFound(key.to_string()))
        } else {
            self.full_catalog
                .get("var")
                .ok_or_else(|| ScxError::SectionNotFound("var".to_string()))
        }
    }

    /// Pure Arrow IPC footer read: no batch deserialisation, no wide-vs-
    /// narrow refinement. Returns whatever the writer recorded in the
    /// footer schema. Counterpart to [`Self::read_arrow_ipc_schema`]
    /// which does the conditional first-batch re-read.
    fn read_arrow_ipc_schema_physical(
        &self,
        entry: &FullCatalogEntry,
    ) -> Result<arrow::datatypes::Schema> {
        let slice = self.section_bytes(entry)?;
        let cursor = Cursor::new(slice);
        let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)?;
        Ok(reader.schema().as_ref().clone())
    }

    /// Read the obs (observation) metadata as an Arrow RecordBatch.
    ///
    /// Transparently handles both file layouts: returns the single
    /// [`SectionType::ObsMetadata`] section on legacy files, or
    /// reassembles every [`SectionType::ObsMetadataShard`] section in
    /// `shard_idx` order on Phase 2 sharded files (with a contiguous-
    /// cover verification — any gap, duplicate, or shrinking
    /// `n_rows_total` is rejected as [`ScxError::InvalidCatalog`]).
    ///
    /// **Memory cost:** allocates a buffer sized to the entire logical
    /// obs table. For atlas-scale files (tens of GB) this can dominate
    /// peak RSS. Prefer the streaming [`Self::obs_shards`] iterator or
    /// per-shard [`Self::read_obs_shard`] when you can process the
    /// table in chunks.
    pub fn read_obs(&self) -> Result<RecordBatch> {
        #[cfg(debug_assertions)]
        self.debug_counts.read_obs.fetch_add(1, Ordering::Relaxed);
        if self.obs_metadata_shard_count() > 0 {
            self.read_sharded_layout_by_prefix(
                "obs_metadata/shard_",
                "obs_metadata",
                SectionType::ObsMetadataShard,
            )?
            .ok_or_else(|| ScxError::SectionNotFound("obs_metadata/shard_*".to_string()))
        } else {
            let entry = self
                .full_catalog
                .get("obs")
                .ok_or_else(|| ScxError::SectionNotFound("obs".to_string()))?;
            self.read_arrow_ipc(entry)
        }
    }

    /// Read the var (variable/gene) metadata as an Arrow RecordBatch.
    /// Mirror of [`Self::read_obs`] for the var axis — same dual-layout
    /// handling, same memory cost caveat, and same streaming
    /// alternatives ([`Self::var_shards`], [`Self::read_var_shard`]).
    pub fn read_var(&self) -> Result<RecordBatch> {
        if self.var_metadata_shard_count() > 0 {
            self.read_sharded_layout_by_prefix(
                "var_metadata/shard_",
                "var_metadata",
                SectionType::VarMetadataShard,
            )?
            .ok_or_else(|| ScxError::SectionNotFound("var_metadata/shard_*".to_string()))
        } else {
            let entry = self
                .full_catalog
                .get("var")
                .ok_or_else(|| ScxError::SectionNotFound("var".to_string()))?;
            self.read_arrow_ipc(entry)
        }
    }

    /// Number of [`SectionType::ObsMetadataShard`] sections in the
    /// catalog. Pure catalog scan — no payload read. Zero on legacy
    /// single-section ([`SectionType::ObsMetadata`]) files.
    pub fn obs_metadata_shard_count(&self) -> usize {
        self.full_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::ObsMetadataShard)
            .count()
    }

    /// Number of [`SectionType::VarMetadataShard`] sections in the
    /// catalog. Mirror of [`Self::obs_metadata_shard_count`].
    pub fn var_metadata_shard_count(&self) -> usize {
        self.full_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::VarMetadataShard)
            .count()
    }

    /// Read one row-shard of obs metadata by index. Returns the on-disk
    /// `RecordBatch` with its stamped shard schema metadata
    /// (`shard_idx`, `row_start`, `n_shard_rows`, `n_rows_total`)
    /// preserved — callers may consult those fields directly.
    pub fn read_obs_shard(&self, shard_idx: u32) -> Result<RecordBatch> {
        #[cfg(debug_assertions)]
        self.debug_counts
            .read_obs_shard
            .fetch_add(1, Ordering::Relaxed);
        let key = format!("obs_metadata/shard_{shard_idx}");
        let entry = self
            .full_catalog
            .get(&key)
            .ok_or(ScxError::SectionNotFound(key))?;
        self.read_arrow_ipc(entry)
    }

    /// Read one row-shard of var metadata. Mirror of
    /// [`Self::read_obs_shard`].
    pub fn read_var_shard(&self, shard_idx: u32) -> Result<RecordBatch> {
        let key = format!("var_metadata/shard_{shard_idx}");
        let entry = self
            .full_catalog
            .get(&key)
            .ok_or(ScxError::SectionNotFound(key))?;
        self.read_arrow_ipc(entry)
    }

    /// Iterate obs metadata shards in `shard_idx` order, yielding one
    /// `RecordBatch` per shard. The iterator never materialises more
    /// than one shard at a time, so peak memory is bounded by the
    /// largest single shard regardless of the logical obs size.
    ///
    /// Returns an empty iterator on legacy single-section files; call
    /// [`Self::read_obs`] for those (and for any caller that genuinely
    /// needs the full obs table).
    pub fn obs_shards(&self) -> impl Iterator<Item = Result<RecordBatch>> + '_ {
        self.metadata_shards_iter(SectionType::ObsMetadataShard, "obs_metadata/shard_")
    }

    /// Iterate var metadata shards in `shard_idx` order. Mirror of
    /// [`Self::obs_shards`].
    pub fn var_shards(&self) -> impl Iterator<Item = Result<RecordBatch>> + '_ {
        self.metadata_shards_iter(SectionType::VarMetadataShard, "var_metadata/shard_")
    }

    /// Shared iterator builder for [`Self::obs_shards`] /
    /// [`Self::var_shards`]. Materialises only the sorted catalog
    /// entries up front (cheap pointer slice); each shard's payload is
    /// fetched lazily as the consumer advances the iterator.
    fn metadata_shards_iter(
        &self,
        shard_type: SectionType,
        name_prefix: &'static str,
    ) -> impl Iterator<Item = Result<RecordBatch>> + '_ {
        let mut entries: Vec<(u32, &FullCatalogEntry)> = self
            .full_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == shard_type && e.name.starts_with(name_prefix))
            .filter_map(|e| {
                let suffix = e.name.strip_prefix(name_prefix)?;
                let idx: u32 = suffix.parse().ok()?;
                Some((idx, e))
            })
            .collect();
        entries.sort_by_key(|(idx, _)| *idx);
        entries
            .into_iter()
            .map(move |(_, e)| self.read_arrow_ipc(e))
    }

    /// Read a named obsm embedding as an Arrow RecordBatch.
    ///
    /// Prefers the sharded on-disk layout (`obsm/<name>_shard_<idx>`,
    /// section type [`SectionType::ObsmEmbeddingShard`]) and falls back
    /// to the legacy single-section [`SectionType::ObsmEmbedding`]
    /// layout for files written before sharding was introduced.
    pub fn read_obsm(&self, name: &str) -> Result<RecordBatch> {
        #[cfg(debug_assertions)]
        self.debug_counts.read_obsm.fetch_add(1, Ordering::Relaxed);
        if let Some(batch) =
            self.read_sharded_layout("obsm", name, SectionType::ObsmEmbeddingShard)?
        {
            return Ok(batch);
        }
        let key = format!("obsm/{name}");
        let entry = self
            .full_catalog
            .get(&key)
            .ok_or_else(|| ScxError::SectionNotFound(key))?;
        self.read_arrow_ipc(entry)
    }

    /// Read all obsm embeddings, keyed by name.
    pub fn read_all_obsm(&self) -> Result<HashMap<String, RecordBatch>> {
        #[cfg(debug_assertions)]
        self.debug_counts
            .read_all_obsm
            .fetch_add(1, Ordering::Relaxed);
        self.read_all_sharded_or_single(
            "obsm",
            SectionType::ObsmEmbedding,
            SectionType::ObsmEmbeddingShard,
        )
    }

    /// Read a named varm embedding as an Arrow RecordBatch.
    ///
    /// See [`Self::read_obsm`] for the sharded / legacy layout handling.
    pub fn read_varm(&self, name: &str) -> Result<RecordBatch> {
        #[cfg(debug_assertions)]
        self.debug_counts.read_varm.fetch_add(1, Ordering::Relaxed);
        if let Some(batch) =
            self.read_sharded_layout("varm", name, SectionType::VarmEmbeddingShard)?
        {
            return Ok(batch);
        }
        let key = format!("varm/{name}");
        let entry = self
            .full_catalog
            .get(&key)
            .ok_or_else(|| ScxError::SectionNotFound(key))?;
        self.read_arrow_ipc(entry)
    }

    /// Read all varm embeddings, keyed by name.
    pub fn read_all_varm(&self) -> Result<HashMap<String, RecordBatch>> {
        #[cfg(debug_assertions)]
        self.debug_counts
            .read_all_varm
            .fetch_add(1, Ordering::Relaxed);
        self.read_all_sharded_or_single(
            "varm",
            SectionType::VarmEmbedding,
            SectionType::VarmEmbeddingShard,
        )
    }

    /// Read a named obsp pairwise sparse matrix (COO format).
    ///
    /// Lazy counterpart to [`Self::read_all_obsp`]: used by
    /// `pyscx::lazy_mapping::ScxLazyPairwiseMapping` so `to_anndata()`
    /// can defer obsp materialization until the consumer actually
    /// accesses `ad.obsp[name]`. Handles both sharded and legacy
    /// single-section layouts; see [`Self::read_obsm`].
    pub fn read_obsp(&self, name: &str) -> Result<RecordBatch> {
        if let Some(batch) =
            self.read_sharded_layout("obsp", name, SectionType::ObspEmbeddingShard)?
        {
            return Ok(batch);
        }
        let key = format!("obsp/{name}");
        let entry = self
            .full_catalog
            .get(&key)
            .ok_or_else(|| ScxError::SectionNotFound(key))?;
        self.read_arrow_ipc(entry)
    }

    /// Read all obsp pairwise sparse matrices (COO format), keyed by name.
    pub fn read_all_obsp(&self) -> Result<HashMap<String, RecordBatch>> {
        self.read_all_sharded_or_single(
            "obsp",
            SectionType::ObspEmbedding,
            SectionType::ObspEmbeddingShard,
        )
    }

    /// Read a named varp pairwise sparse matrix (COO format).
    ///
    /// Lazy counterpart to [`Self::read_all_varp`]; see [`Self::read_obsp`].
    pub fn read_varp(&self, name: &str) -> Result<RecordBatch> {
        if let Some(batch) =
            self.read_sharded_layout("varp", name, SectionType::VarpEmbeddingShard)?
        {
            return Ok(batch);
        }
        let key = format!("varp/{name}");
        let entry = self
            .full_catalog
            .get(&key)
            .ok_or_else(|| ScxError::SectionNotFound(key))?;
        self.read_arrow_ipc(entry)
    }

    /// Read all varp pairwise sparse matrices (COO format), keyed by name.
    pub fn read_all_varp(&self) -> Result<HashMap<String, RecordBatch>> {
        self.read_all_sharded_or_single(
            "varp",
            SectionType::VarpEmbedding,
            SectionType::VarpEmbeddingShard,
        )
    }

    /// List the keys (entry names with their section prefix stripped) of
    /// every `obsp/*` catalog entry. Pure catalog scan — no section
    /// bytes are read. Used by `pyscx::lazy_mapping` to pre-populate the
    /// key set of the lazy obsp wrapper at `to_anndata()` time. Includes
    /// both sharded (`obsp/<name>_shard_<idx>`) and legacy single-section
    /// keys, deduplicating shard names back to their logical key.
    pub fn list_obsp(&self) -> Vec<String> {
        self.list_logical_names(
            "obsp",
            SectionType::ObspEmbedding,
            SectionType::ObspEmbeddingShard,
        )
    }

    /// List the keys of every `varp/*` catalog entry. See [`Self::list_obsp`].
    pub fn list_varp(&self) -> Vec<String> {
        self.list_logical_names(
            "varp",
            SectionType::VarpEmbedding,
            SectionType::VarpEmbeddingShard,
        )
    }

    /// List the keys of every `varm/*` catalog entry. See [`Self::list_obsp`].
    pub fn list_varm(&self) -> Vec<String> {
        self.list_logical_names(
            "varm",
            SectionType::VarmEmbedding,
            SectionType::VarmEmbeddingShard,
        )
    }

    /// List the keys of every `obsm/*` catalog entry. See [`Self::list_obsp`].
    pub fn list_obsm(&self) -> Vec<String> {
        self.list_logical_names(
            "obsm",
            SectionType::ObsmEmbedding,
            SectionType::ObsmEmbeddingShard,
        )
    }

    /// Read a sharded `<prefix>/<name>` section, concatenating shards in
    /// `shard_idx` order. Returns `Ok(None)` if no shards exist for
    /// `<name>` (caller can fall back to the legacy single-section path).
    ///
    /// Concatenation is row-axis. The per-shard `shard_idx` /
    /// `row_start` / `n_shard_rows` / `n_rows_total` metadata stamped
    /// by the writer (see `stamp_dense_shard_metadata`) is used to
    /// verify that the catalog entries form a contiguous, ordered cover
    /// of the logical matrix; any gap, duplicate, mismatch, or missing
    /// metadata returns `ScxError::InvalidCatalog` rather than silently
    /// producing a truncated matrix.
    ///
    /// The returned `RecordBatch`'s schema metadata has the per-shard
    /// fields stripped (`shard_idx` / `row_start` / `n_shard_rows`) so
    /// downstream consumers don't see misleading first-shard values;
    /// `n_rows_total` and any payload-level metadata (e.g. sparse
    /// `n_rows` / `n_cols`) are preserved.
    fn read_sharded_layout(
        &self,
        prefix: &str,
        name: &str,
        shard_type: SectionType,
    ) -> Result<Option<RecordBatch>> {
        let shard_name_prefix = format!("{prefix}/{name}_shard_");
        let logical = format!("{prefix}/{name}");
        self.read_sharded_layout_by_prefix(&shard_name_prefix, &logical, shard_type)
    }

    /// Generic worker shared by [`Self::read_sharded_layout`] (which
    /// handles `<prefix>/<name>_shard_<idx>` naming) and the obs/var
    /// metadata shard readers (which use the flatter
    /// `<axis>/shard_<idx>` naming because there is no logical
    /// sub-name). All cover-verification and metadata-stripping logic
    /// lives here; callers just supply the shard-name prefix to scan
    /// for and the display string used in error messages.
    fn read_sharded_layout_by_prefix(
        &self,
        shard_name_prefix: &str,
        logical: &str,
        shard_type: SectionType,
    ) -> Result<Option<RecordBatch>> {
        let mut shards: Vec<(u32, &FullCatalogEntry)> = self
            .full_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == shard_type && e.name.starts_with(shard_name_prefix))
            .filter_map(|e| {
                let suffix = e.name.strip_prefix(shard_name_prefix)?;
                let idx: u32 = suffix.parse().ok()?;
                Some((idx, e))
            })
            .collect();
        if shards.is_empty() {
            return Ok(None);
        }
        shards.sort_by_key(|(idx, _)| *idx);

        // Decode shards **without** the per-shard wide→narrow downcast;
        // [`assemble_sharded_metadata`] handles the upcast → cover
        // validation → concat → downcast pipeline (shared with the cloud
        // reader so both paths produce byte-identical results).
        let raw_batches: Vec<(u32, RecordBatch)> = shards
            .iter()
            .map(|(idx, entry)| Ok((*idx, self.read_arrow_ipc_raw(entry)?)))
            .collect::<Result<_>>()?;
        Ok(Some(assemble_sharded_metadata(logical, raw_batches)?))
    }

    /// Walk the catalog for both single-section and sharded entries
    /// under `<prefix>/`, returning a `name -> RecordBatch` map. Sharded
    /// entries are concatenated via [`Self::read_sharded_layout`];
    /// single-section entries fall through unchanged.
    fn read_all_sharded_or_single(
        &self,
        prefix: &str,
        single_type: SectionType,
        shard_type: SectionType,
    ) -> Result<HashMap<String, RecordBatch>> {
        let mut result = HashMap::new();
        let path_prefix = format!("{prefix}/");
        let shard_marker = "_shard_";

        // Sharded keys: collect unique logical names, then read each.
        let mut sharded_names: std::collections::BTreeSet<String> = Default::default();
        for entry in &self.full_catalog.entries {
            if entry.section_type != shard_type {
                continue;
            }
            if let Some(rest) = entry.name.strip_prefix(&path_prefix) {
                if let Some(idx) = rest.rfind(shard_marker) {
                    sharded_names.insert(rest[..idx].to_string());
                }
            }
        }
        for name in &sharded_names {
            if let Some(batch) = self.read_sharded_layout(prefix, name, shard_type)? {
                result.insert(name.clone(), batch);
            }
        }

        // Legacy single-section entries (only included if the same name
        // wasn't already produced from shards — shards win).
        for entry in &self.full_catalog.entries {
            if entry.section_type != single_type {
                continue;
            }
            let name = entry
                .name
                .strip_prefix(&path_prefix)
                .unwrap_or(&entry.name)
                .to_string();
            if result.contains_key(&name) {
                continue;
            }
            result.insert(name, self.read_arrow_ipc(entry)?);
        }
        Ok(result)
    }

    /// Pure catalog scan: return the de-duplicated set of logical names
    /// under `<prefix>/`, considering both legacy single-section and
    /// sharded entries. Used by `list_obsm` / `list_obsp` / etc.
    fn list_logical_names(
        &self,
        prefix: &str,
        single_type: SectionType,
        shard_type: SectionType,
    ) -> Vec<String> {
        let path_prefix = format!("{prefix}/");
        let shard_marker = "_shard_";
        let mut names: std::collections::BTreeSet<String> = Default::default();
        for entry in &self.full_catalog.entries {
            if entry.section_type == single_type {
                if let Some(name) = entry.name.strip_prefix(&path_prefix) {
                    names.insert(name.to_string());
                }
            } else if entry.section_type == shard_type {
                if let Some(rest) = entry.name.strip_prefix(&path_prefix) {
                    if let Some(idx) = rest.rfind(shard_marker) {
                        names.insert(rest[..idx].to_string());
                    }
                }
            }
        }
        names.into_iter().collect()
    }

    // -----------------------------------------------------------------------
    // Phase B: per-modality accessors
    // -----------------------------------------------------------------------

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

    /// Read the `var` metadata batch for a specific modality.
    /// `modality_id == 0` reads the global / single-modality `var`
    /// section (matches `read_var()`).
    pub fn read_var_for(&self, modality_id: u8) -> Result<RecordBatch> {
        let key = if modality_id == 0 {
            "var".to_string()
        } else {
            let mname = self.modality_name_for_id(modality_id)?;
            format!("var/{mname}")
        };
        let entry = self
            .full_catalog
            .get(&key)
            .ok_or_else(|| ScxError::SectionNotFound(key))?;
        self.read_arrow_ipc(entry)
    }

    /// Number of CSR shards belonging to the given modality.
    /// `modality_id == 0` returns the global CSR shard count
    /// (matches the legacy single-modality semantics).
    pub fn csr_shard_count_for(&self, modality_id: u8) -> u32 {
        self.full_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CsrShard && e.modality_id == modality_id)
            .count() as u32
    }

    /// Number of CSC shards belonging to the given modality.
    pub fn csc_shard_count_for(&self, modality_id: u8) -> u32 {
        self.full_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CscShard && e.modality_id == modality_id)
            .count() as u32
    }

    /// Read a single CSR shard for the given modality, by 0-based
    /// index in catalog order (sorted by `row_start`). Returns
    /// scipy-compatible arrays.
    pub fn read_csr_shard_for(
        &self,
        modality_id: u8,
        shard_idx: usize,
    ) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
        let shards = self.full_catalog.csr_shards_for_modality(modality_id);
        if shard_idx >= shards.len() {
            return Err(ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: shards.len(),
            });
        }
        self.read_shard_from_entry(shards[shard_idx])
    }

    /// Read the raw bytes (76-byte header + encoded payload) of a single CSR
    /// shard for the given modality, by 0-based index in catalog order.
    ///
    /// Mirrors [`Self::read_csr_shard_for`] but skips decoding — the returned
    /// slice (borrowed from the mmap) can be fed directly to a GPU-side shard
    /// decoder such as `scx_gpu::decode_shard_gpu`.
    pub fn read_raw_csr_shard_bytes_for(&self, modality_id: u8, shard_idx: usize) -> Result<&[u8]> {
        let shards = self.full_catalog.csr_shards_for_modality(modality_id);
        if shard_idx >= shards.len() {
            return Err(ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: shards.len(),
            });
        }
        self.read_raw_shard_bytes(shards[shard_idx])
    }

    /// Resolve the decode-metadata sidecar for a CSR shard by modality + 0-based
    /// index (parallel to [`Self::read_raw_csr_shard_bytes_for`]). Returns `None`
    /// when no fresh Scx1 sidecar exists (non-Scx1 codec, stale, or absent —
    /// never an error). Used by the GPU device-decode handoff to drive
    /// `scx_gpu::decode_csr_shards_to_device_with_metadata` from the
    /// encoder-emitted offsets instead of a CPU prescan (ACC-RUST-OPT-V4 4.4a).
    pub fn scx1_metadata_for_csr_shard(
        &self,
        modality_id: u8,
        shard_idx: usize,
    ) -> Result<Option<scx_codec::Scx1DecodeMetadata>> {
        let shards = self.full_catalog.csr_shards_for_modality(modality_id);
        if shard_idx >= shards.len() {
            return Err(ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: shards.len(),
            });
        }
        self.scx1_metadata_for(shards[shard_idx])
    }

    /// Indptr-only variant of [`Self::read_csr_shard_for`]. Decodes only
    /// the row-pointer region; cheap path for callers that need just
    /// per-row nnz counts.
    pub fn read_csr_shard_indptr_for(&self, modality_id: u8, shard_idx: usize) -> Result<Vec<i64>> {
        let shards = self.full_catalog.csr_shards_for_modality(modality_id);
        if shard_idx >= shards.len() {
            return Err(ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: shards.len(),
            });
        }
        self.read_shard_indptr_from_entry(shards[shard_idx])
    }

    /// Read a single CSC shard for the given modality.
    pub fn read_csc_shard_for(&self, modality_id: u8, shard_idx: usize) -> Result<ScxCsc> {
        let shards = self.full_catalog.csc_shards_for_modality(modality_id);
        if shard_idx >= shards.len() {
            return Err(ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: shards.len(),
            });
        }
        self.read_csc_from_entry(shards[shard_idx])
    }

    /// Read and assemble all CSR shards for the given modality into a
    /// single `ScxCsr`. Mirrors `read_all_csr_shards()` (the global
    /// path) but filters catalog entries by `modality_id` and prefers
    /// the modality table's `n_vars` over the assembled shard extent
    /// for the returned `n_cols`.
    pub fn read_all_csr_shards_for(&self, modality_id: u8) -> Result<ScxCsr> {
        let shards = self.full_catalog.csr_shards_for_modality(modality_id);
        #[cfg(feature = "parallel")]
        let assembled = self.assemble_shards_parallel(&shards)?;
        #[cfg(not(feature = "parallel"))]
        let assembled = self.assemble_shards(&shards)?;

        // Writers now stamp `ShardHeader.n_minor` with the per-modality
        // `n_vars` (see `ScxWriter::write_shard_inner`), so the
        // assembled extent should already match `modality_info.n_vars`.
        // The modality_info preference here is defensive — it lets us
        // recover the correct shape from older multimodal files that
        // pre-date that fix and stamped the file-wide max.
        let n_cols = match self.modality_info(modality_id) {
            Some(info) => info.n_vars as usize,
            None => assembled.shape.1,
        };
        Ok(ScxCsr::new_unchecked(
            (assembled.shape.0, n_cols),
            assembled.indptr,
            assembled.indices,
            assembled.data,
        ))
    }

    /// Read and assemble all CSC shards for the given modality into a
    /// single `ScxCsc`. Mirrors the per-modality CSR reader; falls
    /// back to a sequential per-shard concat (the parallel CSC
    /// assembler can be added later if hot).
    pub fn read_all_csc_shards_for(&self, modality_id: u8) -> Result<ScxCsc> {
        let n_rows = self.header.n_obs as usize;
        let shards = self.full_catalog.csc_shards_for_modality(modality_id);
        if shards.is_empty() {
            return Ok(ScxCsc::new_unchecked(
                (n_rows, 0),
                vec![0],
                Vec::new(),
                Vec::new(),
            ));
        }
        self.assemble_csc_shards(&shards)
    }

    /// Phase B.4: read a contiguous column range from a modality's
    /// CSC shards. Same shape as `read_csc_columns(col_range)` but
    /// scoped to one modality via the catalog's
    /// `csc_shards_for_modality` filter. Shard intersection /
    /// `col_slice` semantics match the single-modality version.
    pub fn read_csc_columns_for(
        &self,
        modality_id: u8,
        col_range: std::ops::Range<u32>,
    ) -> Result<ScxCsc> {
        let c_lo = col_range.start as u64;
        let c_hi = col_range.end as u64;
        let n_rows = self.header.n_obs as usize;

        if c_lo >= c_hi {
            return Ok(ScxCsc::new_unchecked(
                (n_rows, 0),
                vec![0],
                Vec::new(),
                Vec::new(),
            ));
        }

        // Filter to the modality's CSC shards, then intersect with
        // the column range. We can't use the file-wide
        // `csc_shards_for_col_range` helper because it doesn't filter
        // by modality_id.
        let modality_shards = self.full_catalog.csc_shards_for_modality(modality_id);
        let shards: Vec<&FullCatalogEntry> = modality_shards
            .into_iter()
            .filter(|e| match e.stats.as_ref() {
                Some(s) => {
                    let shard_lo = s.major_start(e.section_type);
                    let shard_hi = s.major_end(e.section_type);
                    shard_lo < c_hi && c_lo < shard_hi
                }
                None => false,
            })
            .collect();

        let mut decoded: Vec<ScxCsc> = Vec::with_capacity(shards.len());
        for entry in &shards {
            let csc = self.read_csc_from_entry(entry)?;
            let stats = entry.stats.as_ref().ok_or_else(|| {
                ScxError::InvalidCatalog(format!("CSC shard '{}' missing stats block", entry.name))
            })?;
            let shard_lo = stats.major_start(entry.section_type);
            let shard_hi = stats.major_end(entry.section_type);
            let lo_in_shard = c_lo.saturating_sub(shard_lo) as usize;
            let hi_in_shard = (c_hi.min(shard_hi).saturating_sub(shard_lo)) as usize;
            let sliced = if lo_in_shard == 0 && hi_in_shard == csc.n_cols() {
                csc
            } else {
                csc.col_slice(lo_in_shard, hi_in_shard).map_err(|e| {
                    ScxError::InvalidCatalog(format!(
                        "CSC col_slice failed for shard '{}': {e}",
                        entry.name
                    ))
                })?
            };
            decoded.push(sliced);
        }

        concatenate_csc_along_cols(decoded, n_rows)
    }

    /// Phase B.4: read a sorted column subset from a modality's CSC
    /// shards. Same shape as `read_csc_columns_subset(cols)` —
    /// collapses contiguous runs and concatenates per-run
    /// `read_csc_columns_for` results.
    pub fn read_csc_columns_subset_for(&self, modality_id: u8, cols: &[u32]) -> Result<ScxCsc> {
        let n_rows = self.header.n_obs as usize;
        if cols.is_empty() {
            return Ok(ScxCsc::new_unchecked(
                (n_rows, 0),
                vec![0],
                Vec::new(),
                Vec::new(),
            ));
        }
        let mut runs: Vec<ScxCsc> = Vec::new();
        let mut run_start = cols[0];
        let mut run_end = cols[0] + 1;
        for &c in &cols[1..] {
            if c == run_end {
                run_end = c + 1;
            } else {
                runs.push(self.read_csc_columns_for(modality_id, run_start..run_end)?);
                run_start = c;
                run_end = c + 1;
            }
        }
        runs.push(self.read_csc_columns_for(modality_id, run_start..run_end)?);
        if runs.len() == 1 {
            return Ok(runs.pop().unwrap());
        }
        concatenate_csc_along_cols(runs, n_rows)
    }

    /// Phase B.4: read a per-modality named layer (CSR), assembling
    /// all its shards into a single `ScxCsr`. Mirrors `read_layer`
    /// but filters via `catalog.layer_csr_shards_for_modality(...)`.
    /// Output `n_cols` is patched from `modality_info(id).n_vars`
    /// (matching `read_all_csr_shards_for`).
    pub fn read_layer_for(&self, modality_id: u8, layer_name: &str) -> Result<ScxCsr> {
        #[cfg(debug_assertions)]
        self.debug_counts
            .read_layer_for
            .fetch_add(1, Ordering::Relaxed);
        let shards = self
            .full_catalog
            .layer_csr_shards_for_modality(modality_id, layer_name);
        if shards.is_empty() {
            return Err(ScxError::SectionNotFound(format!(
                "layer '{layer_name}' for modality_id {modality_id}"
            )));
        }
        let mut assembled = {
            #[cfg(feature = "parallel")]
            {
                self.assemble_shards_parallel(&shards)?
            }
            #[cfg(not(feature = "parallel"))]
            {
                self.assemble_shards(&shards)?
            }
        };
        // Patch n_cols from the modality's per-modality n_vars (the
        // assembler used header.n_vars which is the file-wide max).
        if let Some(info) = self.modality_info(modality_id) {
            assembled.shape.1 = info.n_vars as usize;
        }
        Ok(assembled)
    }

    /// Phase B.4: read a per-modality named layer's CSC shards,
    /// concatenated along the column axis. Mirrors
    /// `read_all_csc_shards_for` but filters by layer name via
    /// `catalog.layer_csc_shards_for_modality(...)`.
    pub fn read_layer_csc_for(&self, modality_id: u8, layer_name: &str) -> Result<ScxCsc> {
        let n_rows = self.header.n_obs as usize;
        let shards = self
            .full_catalog
            .layer_csc_shards_for_modality(modality_id, layer_name);
        if shards.is_empty() {
            return Err(ScxError::SectionNotFound(format!(
                "layer-csc '{layer_name}' for modality_id {modality_id}"
            )));
        }
        let decoded: Vec<ScxCsc> = shards
            .iter()
            .map(|e| self.read_csc_from_entry(e))
            .collect::<Result<Vec<_>>>()?;
        concatenate_csc_along_cols(decoded, n_rows)
    }

    /// Read an obsm batch keyed by `(modality_id, key)`. Section
    /// names are `obsm/{modality_name}/{key}` for `modality_id >= 1`
    /// and `obsm/{key}` for `modality_id == 0` (global).
    pub fn read_obsm_for(&self, modality_id: u8, key: &str) -> Result<RecordBatch> {
        #[cfg(debug_assertions)]
        self.debug_counts
            .read_obsm_for
            .fetch_add(1, Ordering::Relaxed);
        let (shard_prefix, logical) = if modality_id == 0 {
            (format!("obsm/{key}_shard_"), format!("obsm/{key}"))
        } else {
            let mname = self.modality_name_for_id(modality_id)?;
            (
                format!("obsm/{mname}/{key}_shard_"),
                format!("obsm/{mname}/{key}"),
            )
        };
        if let Some(batch) = self.read_sharded_layout_by_prefix(
            &shard_prefix,
            &logical,
            SectionType::ObsmEmbeddingShard,
        )? {
            return Ok(batch);
        }
        let entry = self
            .full_catalog
            .get(&logical)
            .ok_or_else(|| ScxError::SectionNotFound(logical))?;
        self.read_arrow_ipc(entry)
    }

    /// Read a per-modality varm embedding as an Arrow RecordBatch.
    ///
    /// Mirrors [`Self::read_obsm_for`]: tries the sharded layout
    /// (`varm/{modality_name}/{key}_shard_<idx>`, section type
    /// [`SectionType::VarmEmbeddingShard`]) first, falling back to the
    /// legacy single-section [`SectionType::VarmEmbedding`].
    pub fn read_varm_for(&self, modality_id: u8, key: &str) -> Result<RecordBatch> {
        #[cfg(debug_assertions)]
        self.debug_counts
            .read_varm_for
            .fetch_add(1, Ordering::Relaxed);
        let (shard_prefix, logical) = if modality_id == 0 {
            (format!("varm/{key}_shard_"), format!("varm/{key}"))
        } else {
            let mname = self.modality_name_for_id(modality_id)?;
            (
                format!("varm/{mname}/{key}_shard_"),
                format!("varm/{mname}/{key}"),
            )
        };
        if let Some(batch) = self.read_sharded_layout_by_prefix(
            &shard_prefix,
            &logical,
            SectionType::VarmEmbeddingShard,
        )? {
            return Ok(batch);
        }
        let entry = self
            .full_catalog
            .get(&logical)
            .ok_or_else(|| ScxError::SectionNotFound(logical))?;
        self.read_arrow_ipc(entry)
    }

    /// Read the per-modality `uns` JSON, decoded to `serde_json::Value`.
    pub fn read_uns_for(&self, modality_id: u8) -> Result<serde_json::Value> {
        let section_name = if modality_id == 0 {
            "uns".to_string()
        } else {
            let mname = self.modality_name_for_id(modality_id)?;
            format!("uns/{mname}")
        };
        let entry = self
            .full_catalog
            .get(&section_name)
            .ok_or_else(|| ScxError::SectionNotFound(section_name))?;
        let bytes = self.section_bytes(entry)?;
        Ok(serde_json::from_slice(bytes)?)
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
    // adata.raw reading
    // -----------------------------------------------------------------------

    /// True if this file carries an `adata.raw` count matrix
    /// ([`SectionType::RawCsrShard`] + `raw/var`), per the `has_raw`
    /// header flag.
    pub fn has_raw(&self) -> bool {
        self.header.has_raw()
    }

    /// The raw matrix column count (`raw.n_vars`), read from the first
    /// raw shard's stats without decoding any payload. `None` when the
    /// file has no raw matrix.
    ///
    /// For a row-major shard `compute_shard_stats` stores the minor-axis
    /// extent (the full column count, passed as `raw_n_vars` in
    /// `write_shard_inner`) in `col_end` — NOT a per-shard max index — so
    /// `col_end` is the total raw column count and is identical on every
    /// raw shard.
    pub fn raw_n_vars(&self) -> Option<usize> {
        self.full_catalog
            .raw_csr_shards_sorted()
            .first()
            .and_then(|e| e.stats.as_ref())
            .map(|s| s.col_end as usize)
    }

    /// Read the `adata.raw.var` DataFrame ([`SectionType::RawVarMetadata`]).
    pub fn read_raw_var(&self) -> Result<RecordBatch> {
        let entry = self
            .full_catalog
            .get("raw/var")
            .ok_or_else(|| ScxError::SectionNotFound("raw/var".to_string()))?;
        self.read_arrow_ipc(entry)
    }

    /// Read all `adata.raw` CSR shards and concatenate along the row
    /// (obs) axis. The result's column count is the raw matrix's OWN
    /// `raw_n_vars` (recovered from the shards' stats), which may exceed
    /// the main matrix `n_vars`. Mirrors [`Self::read_all_csr_shards`]
    /// but over the raw section family.
    pub fn read_all_raw_csr_shards(&self) -> Result<ScxCsr> {
        let shards = self.full_catalog.raw_csr_shards_sorted();
        if shards.is_empty() {
            return Ok(ScxCsr::new_unchecked(
                (self.header.n_obs as usize, 0),
                vec![0],
                vec![],
                vec![],
            ));
        }

        // raw_n_vars is the minor extent of any raw shard (row-major
        // stats store it as col_end). All shards share it.
        let raw_n_vars = shards[0]
            .stats
            .as_ref()
            .map(|s| s.col_end as usize)
            .ok_or_else(|| {
                ScxError::InvalidCatalog(format!(
                    "raw CSR shard '{}' has no stats block",
                    shards[0].name
                ))
            })?;

        let shard_sizes: Vec<(usize, usize)> = shards
            .iter()
            .map(|e| {
                let stats = e.stats.as_ref().ok_or_else(|| {
                    ScxError::InvalidCatalog(format!(
                        "raw CSR shard '{}' at offset {} has no stats block",
                        e.name, e.offset
                    ))
                })?;
                Ok::<_, ScxError>((
                    (stats.row_end - stats.row_start) as usize,
                    stats.nnz as usize,
                ))
            })
            .collect::<Result<_>>()?;
        let total_rows: usize = shard_sizes.iter().map(|(r, _)| *r).sum();
        let total_nnz: usize = shard_sizes.iter().map(|(_, n)| *n).sum();

        let mut indptr = vec![0i64; total_rows + 1];
        let mut indices = vec![0i32; total_nnz];
        let mut data = vec![0f32; total_nnz];

        let mut cum_rows = 0usize;
        let mut cum_nnz = 0usize;
        for (i, entry) in shards.iter().enumerate() {
            let (n_rows, nnz) = shard_sizes[i];
            let (shard_ip, shard_ix, shard_data) = self.read_shard_from_entry(entry)?;
            indices[cum_nnz..cum_nnz + nnz].copy_from_slice(&shard_ix);
            data[cum_nnz..cum_nnz + nnz].copy_from_slice(&shard_data);
            if i == 0 {
                indptr[0..n_rows + 1].copy_from_slice(&shard_ip);
            } else {
                let nnz_off_i64 = cum_nnz as i64;
                for j in 0..n_rows {
                    indptr[cum_rows + 1 + j] = shard_ip[j + 1] + nnz_off_i64;
                }
            }
            cum_rows += n_rows;
            cum_nnz += nnz;
        }

        let n_rows = indptr.len().saturating_sub(1);
        Ok(ScxCsr::new_unchecked(
            (n_rows, raw_n_vars),
            indptr,
            indices,
            data,
        ))
    }

    // -----------------------------------------------------------------------
    // CSC shard reading (Phase A.3)
    // -----------------------------------------------------------------------

    /// Number of CSC shards in the file (from the file header).
    pub fn csc_shard_count(&self) -> u32 {
        self.header.n_csc_shards
    }

    /// Read a single CSC shard by index in catalog order, returning a
    /// fully-validated `ScxCsc`.
    ///
    /// The shard's `[col_start, col_end)` is taken from the catalog
    /// `ShardStats` (axis-overloaded `row_start`/`row_end`); within the
    /// shard, `indices` are global row indices.
    pub fn read_csc_shard(&self, shard_idx: usize) -> Result<ScxCsc> {
        let shards = self.full_catalog.csc_shards_sorted();
        if shard_idx >= shards.len() {
            return Err(ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: shards.len(),
            });
        }
        self.read_csc_from_entry(shards[shard_idx])
    }

    /// Read all CSC shards and concatenate them along the column axis.
    pub fn read_all_csc_shards(&self) -> Result<ScxCsc> {
        let shards = self.full_catalog.csc_shards_sorted();
        self.assemble_csc_shards(&shards)
    }

    /// Read a contiguous range of columns. Skips CSC shards whose
    /// `[col_start, col_end)` does not intersect `col_range`. Partially
    /// overlapping shards are decoded and `col_slice`d post-decode.
    pub fn read_csc_columns(&self, col_range: std::ops::Range<u32>) -> Result<ScxCsc> {
        let c_lo = col_range.start as u64;
        let c_hi = col_range.end as u64;
        let n_rows = self.header.n_obs as usize;

        if c_lo >= c_hi {
            return Ok(ScxCsc::new_unchecked(
                (n_rows, 0),
                vec![0],
                Vec::new(),
                Vec::new(),
            ));
        }

        let shards = self.full_catalog.csc_shards_for_col_range(c_lo, c_hi);

        // Decode each shard, then column-slice partial overlaps to the
        // intersection with [c_lo, c_hi).
        let mut decoded: Vec<ScxCsc> = Vec::with_capacity(shards.len());
        for entry in &shards {
            let csc = self.read_csc_from_entry(entry)?;
            let stats = entry.stats.as_ref().ok_or_else(|| {
                ScxError::InvalidCatalog(format!("CSC shard '{}' missing stats block", entry.name))
            })?;
            let shard_lo = stats.major_start(entry.section_type);
            let shard_hi = stats.major_end(entry.section_type);
            let lo_in_shard = c_lo.saturating_sub(shard_lo) as usize;
            let hi_in_shard = (c_hi.min(shard_hi).saturating_sub(shard_lo)) as usize;
            let sliced = if lo_in_shard == 0 && hi_in_shard == csc.n_cols() {
                csc
            } else {
                csc.col_slice(lo_in_shard, hi_in_shard).map_err(|e| {
                    ScxError::InvalidCatalog(format!(
                        "CSC col_slice failed for shard '{}': {e}",
                        entry.name
                    ))
                })?
            };
            decoded.push(sliced);
        }

        concatenate_csc_along_cols(decoded, n_rows)
    }

    /// Read an arbitrary sorted column subset by collapsing it to
    /// contiguous runs and concatenating per-run `read_csc_columns`
    /// results. The caller is responsible for sorting `cols`; duplicates
    /// are not deduplicated.
    pub fn read_csc_columns_subset(&self, cols: &[u32]) -> Result<ScxCsc> {
        let n_rows = self.header.n_obs as usize;
        if cols.is_empty() {
            return Ok(ScxCsc::new_unchecked(
                (n_rows, 0),
                vec![0],
                Vec::new(),
                Vec::new(),
            ));
        }

        // Detect contiguous runs and read each as one column slice.
        let mut runs: Vec<ScxCsc> = Vec::new();
        let mut run_start = cols[0];
        let mut run_end = cols[0] + 1;
        for &c in &cols[1..] {
            if c == run_end {
                run_end = c + 1;
            } else if c < run_end {
                // Non-monotonic input — fall through to a single-column
                // slice rather than silently re-using the run buffer.
                runs.push(self.read_csc_columns(run_start..run_end)?);
                run_start = c;
                run_end = c + 1;
            } else {
                runs.push(self.read_csc_columns(run_start..run_end)?);
                run_start = c;
                run_end = c + 1;
            }
        }
        runs.push(self.read_csc_columns(run_start..run_end)?);

        if runs.len() == 1 {
            return Ok(runs.pop().unwrap());
        }
        concatenate_csc_along_cols(runs, n_rows)
    }

    /// Decode a single CSC shard from a catalog entry. The decoded
    /// arrays are validated and wrapped in `ScxCsc::new_unchecked` (the
    /// shard payload was BLAKE3-checksummed when the catalog was
    /// verified at `open()`).
    fn read_csc_from_entry(&self, entry: &FullCatalogEntry) -> Result<ScxCsc> {
        let (indptr, indices, data) = self.read_shard_from_entry(entry)?;
        // For CSC: n_major == n_cols_in_shard, indices are global row
        // indices in [0, n_obs). The shard header's n_minor field
        // carries the file-wide `n_obs` (the unbound minor axis for
        // CSC); the actual column count is `len(indptr) - 1`.
        let n_cols_in_shard = indptr.len().saturating_sub(1);
        let n_rows = self.header.n_obs as usize;
        Ok(ScxCsc::new_unchecked(
            (n_rows, n_cols_in_shard),
            indptr,
            indices,
            data,
        ))
    }

    /// Concatenate a sorted list of CSC shards along the column axis.
    /// Each shard contributes its columns in order; indptr offsets are
    /// rebased via cumulative-nnz prefix accumulation. CSC `indices` are
    /// already global row IDs and need no offsetting.
    fn assemble_csc_shards(&self, shards: &[&FullCatalogEntry]) -> Result<ScxCsc> {
        let n_rows = self.header.n_obs as usize;
        if shards.is_empty() {
            return Ok(ScxCsc::new_unchecked(
                (n_rows, 0),
                vec![0],
                Vec::new(),
                Vec::new(),
            ));
        }

        let decoded: Vec<ScxCsc> = shards
            .iter()
            .map(|entry| self.read_csc_from_entry(entry))
            .collect::<Result<_>>()?;

        concatenate_csc_along_cols(decoded, n_rows)
    }
}

/// Per-shard metadata stamped by the writer
/// (see `crate::writer::stamp_dense_shard_metadata`). Parsed by the
/// sharded reader path to verify a contiguous, ordered cover of the
/// logical matrix. Distinct from `crate::shard::ShardHeader`, which is
/// the on-disk 76-byte CSR/CSC shard header.
struct ObsmShardMetadata {
    shard_idx: u32,
    row_start: u64,
    n_shard_rows: u64,
    n_rows_total: u64,
}

/// Physical layout of a row-sharded dense mapping, resolved by
/// [`ScxReader::dense_mapping_layout`] and consumed by
/// [`crate::BackedDenseReader`].
pub(crate) struct DenseMappingLayout {
    /// Per-shard rows, ordered by `shard_idx` (== sorted by `row_start`).
    pub(crate) entries: Vec<DenseShardLayoutEntry>,
    /// Total logical row count (last shard's `n_rows_total`).
    pub(crate) n_rows: u64,
    /// Embedding dimensionality (number of dense columns).
    pub(crate) n_cols: usize,
    /// Column-0 dtype, as a representative for the whole mapping.
    pub(crate) dtype: arrow::datatypes::DataType,
    /// Canonical column fields (per-shard schema metadata stripped) —
    /// the row-gather output schema. Taken from the first shard.
    pub(crate) fields: arrow::datatypes::Fields,
}

/// One shard's catalog offset + stamped row range. Mirrors the fields
/// `BackedDenseReader` needs (no `nnz`, since dense shards aren't CSR).
pub(crate) struct DenseShardLayoutEntry {
    pub(crate) offset: u64,
    pub(crate) length: u64,
    pub(crate) section_type: SectionType,
    pub(crate) modality_id: u8,
    pub(crate) row_start: u64,
    pub(crate) n_shard_rows: u64,
}

/// Pull `shard_idx` / `row_start` / `n_shard_rows` / `n_rows_total`
/// off a sharded batch's schema metadata. Returns
/// `ScxError::InvalidCatalog` if any field is missing or unparseable,
/// naming the logical section so the caller can produce a useful error.
fn parse_shard_metadata(logical: &str, batch: &RecordBatch) -> Result<ObsmShardMetadata> {
    parse_shard_metadata_md(logical, batch.schema_ref().metadata())
}

/// Like [`parse_shard_metadata`] but reads from a schema metadata map
/// directly, so the backed dense reader can pull row ranges from an
/// Arrow IPC **footer schema** (no batch deserialisation) at open time.
fn parse_shard_metadata_md(
    logical: &str,
    md: &std::collections::HashMap<String, String>,
) -> Result<ObsmShardMetadata> {
    let get = |key: &str| -> Result<u64> {
        md.get(key)
            .ok_or_else(|| {
                ScxError::InvalidCatalog(format!("{logical}: shard schema missing '{key}'"))
            })
            .and_then(|s| {
                s.parse::<u64>().map_err(|_| {
                    ScxError::InvalidCatalog(format!(
                        "{logical}: shard schema '{key}'='{s}' is not a u64"
                    ))
                })
            })
    };
    let shard_idx = u32::try_from(get("shard_idx")?)
        .map_err(|_| ScxError::InvalidCatalog(format!("{logical}: shard_idx exceeds u32::MAX")))?;
    Ok(ObsmShardMetadata {
        shard_idx,
        row_start: get("row_start")?,
        n_shard_rows: get("n_shard_rows")?,
        n_rows_total: get("n_rows_total")?,
    })
}

/// Assemble a set of raw (un-downcast) metadata-shard `RecordBatch`es into
/// one logical batch.
///
/// `raw_batches` are `(shard_idx, batch)` pairs decoded from disk or an
/// object store **without** any wide→narrow downcast — the caller is
/// responsible only for fetching and Arrow-IPC-decoding the shard bytes.
/// The shared assembly steps live here so the local mmap reader
/// ([`ScxReader::read_sharded_layout_by_prefix`]) and the cloud reader
/// produce byte-identical results:
///
/// 1. upcast every batch to `LargeUtf8` / `LargeBinary` (no-op when the
///    writer already serialised wide types);
/// 2. validate the shards form a contiguous, ordered cover via the stamped
///    `shard_idx` / `row_start` / `n_shard_rows` / `n_rows_total` metadata
///    (monotonic non-decreasing `n_rows_total`, the last shard's total
///    equals the cumulative cover);
/// 3. concat on the wide schema (safe to `i64::MAX` offsets);
/// 4. downcast back to narrow `Utf8` / `Binary` for columns whose combined
///    offsets fit, leaving over-`i32::MAX` columns wide;
/// 5. strip the per-shard metadata keys from the result schema.
///
/// Errors with `InvalidCatalog` on any cover violation and
/// `SectionNotFound` when `raw_batches` is empty.
/// Collapse every dictionary (categorical) column in `batch` to a unified
/// dictionary with distinct values.
///
/// `arrow::compute::concat` concatenates per-shard dictionary value arrays
/// without deduplicating, so concatenating N shards that each hold the same
/// category produces a dictionary with that category repeated N times. Such a
/// batch round-trips through Arrow IPC fine, but `pyarrow.Table.to_pandas()`
/// raises `ValueError: Categorical categories must be unique`. Casting each
/// dictionary column to its value type (decode) and back to the original
/// dictionary type (re-encode) rebuilds a deduplicated dictionary with keys
/// remapped to the surviving values. Non-dictionary columns pass through
/// untouched; the schema (and its `pandas` index metadata) is preserved.
fn unify_dictionary_columns(batch: &RecordBatch) -> Result<RecordBatch> {
    use arrow::datatypes::{DataType, Field, Schema};
    let schema = batch.schema();
    if !schema
        .fields()
        .iter()
        .any(|f| matches!(f.data_type(), DataType::Dictionary(_, _)))
    {
        return Ok(batch.clone());
    }
    let mut new_fields: Vec<Field> = Vec::with_capacity(schema.fields().len());
    let mut new_columns: Vec<arrow::array::ArrayRef> = Vec::with_capacity(batch.num_columns());
    for (i, field) in schema.fields().iter().enumerate() {
        let col = batch.column(i);
        match field.data_type() {
            DataType::Dictionary(_, value_type) => {
                // Decode to the plain value array (drops the per-shard,
                // possibly-duplicated dictionary), then re-encode to a fresh
                // unified dictionary. Encode once with a wide Int32 key to learn
                // the deduplicated cardinality, then re-encode with the minimal
                // signed key type that fits it so atlas-scale categoricals don't
                // carry needlessly wide codes.
                let values = arrow::compute::cast(col, value_type.as_ref())?;
                let wide_dt = DataType::Dictionary(Box::new(DataType::Int32), value_type.clone());
                let wide = arrow::compute::cast(&values, &wide_dt)?;
                let n_distinct = wide
                    .as_any()
                    .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::Int32Type>>()
                    .map(|d| d.values().len())
                    .unwrap_or(usize::MAX);
                let key_type = min_dictionary_key_type(n_distinct);
                let final_dt = DataType::Dictionary(Box::new(key_type), value_type.clone());
                let encoded = if final_dt == wide_dt {
                    wide
                } else {
                    arrow::compute::cast(&values, &final_dt)?
                };
                new_columns.push(encoded);
                new_fields.push(
                    Field::new(field.name(), final_dt, field.is_nullable())
                        .with_metadata(field.metadata().clone()),
                );
            }
            _ => {
                new_columns.push(col.clone());
                new_fields.push(field.as_ref().clone());
            }
        }
    }
    let new_schema = Schema::new(new_fields).with_metadata(schema.metadata().clone());
    Ok(RecordBatch::try_new(Arc::new(new_schema), new_columns)?)
}

/// Smallest signed Arrow dictionary key (index) type that can address
/// `n_distinct` values: `Int8` for ≤ `i8::MAX`, `Int16` for ≤ `i16::MAX`,
/// else `Int32`. Keys are non-negative indices, so the signed maxima are the
/// addressable counts. Mirrors the compact code widths anndata/pandas use for
/// categoricals while guaranteeing no overflow.
fn min_dictionary_key_type(n_distinct: usize) -> arrow::datatypes::DataType {
    use arrow::datatypes::DataType;
    if n_distinct <= i8::MAX as usize {
        DataType::Int8
    } else if n_distinct <= i16::MAX as usize {
        DataType::Int16
    } else {
        DataType::Int32
    }
}

/// Decode an Arrow IPC schema from a raw section byte slice.
///
/// - **Fast path** (no `LargeUtf8` / `LargeBinary` / `Dictionary(_, Large*)`
///   in the footer): return the IPC footer schema directly. No data
///   deserialization — ~KB of work.
/// - **Slow path** (any wide type present): deserialize only the *first*
///   batch and run [`crate::arrow_compat::downcast_large_types`] so the
///   returned schema matches what the data path produces — narrow types
///   when offsets fit, wide types when they overflow.
///
/// Cost is bounded by the bytes passed in (one section / one shard), never
/// the full logical table. This is the shared core of
/// [`ScxReader::read_obs_schema`] / [`ScxReader::read_var_schema`]; the
/// cloud reader calls it on a single fetched shard so its schema reads stay
/// byte-identical to the local path without assembling the whole obs/var.
pub fn decode_arrow_ipc_schema(bytes: &[u8]) -> Result<arrow::datatypes::Schema> {
    let cursor = Cursor::new(bytes);
    let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)?;
    let on_disk = reader.schema();

    let has_wide = on_disk.fields().iter().any(|f| {
        use arrow::datatypes::DataType;
        matches!(f.data_type(), DataType::LargeUtf8 | DataType::LargeBinary)
            || matches!(
                f.data_type(),
                DataType::Dictionary(_, v)
                    if matches!(v.as_ref(), DataType::LargeUtf8 | DataType::LargeBinary)
            )
    });
    if !has_wide {
        return Ok(on_disk.as_ref().clone());
    }

    // Wide types present — drive the slow path through the same logic the
    // data path uses, so the schema reflects whether offsets actually
    // overflow per column.
    let cursor = Cursor::new(bytes);
    let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)?;
    let mut batches = reader.into_iter();
    let batch = batches
        .next()
        .ok_or_else(|| {
            ScxError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Arrow IPC file contains no batches",
            ))
        })?
        .map_err(ScxError::Arrow)?;
    let normalized = crate::arrow_compat::downcast_large_types(&batch)?;
    Ok(normalized.schema().as_ref().clone())
}

pub fn assemble_sharded_metadata(
    logical: &str,
    mut raw_batches: Vec<(u32, RecordBatch)>,
) -> Result<RecordBatch> {
    if raw_batches.is_empty() {
        return Err(ScxError::SectionNotFound(format!("{logical} (no shards)")));
    }
    raw_batches.sort_by_key(|(idx, _)| *idx);

    // Force every batch to the wide encoding before concat. The writer's
    // `write_arrow_ipc` always upcasts to LargeUtf8 / LargeBinary before
    // serialising, so per-shard reads typically come back wide already;
    // upcasting is a no-op in that case but covers shards whose individual
    // payload was narrow on disk. Concatenating on narrow offsets would
    // otherwise reproduce the original `Offset overflow error` once the
    // combined string payload exceeds `i32::MAX`.
    //
    // Also widen every categorical (dictionary) column's KEY type to Int32
    // before concat. Per-shard categoricals are written with a key sized to
    // each shard's *local* vocabulary (e.g. Int8 for ≤127 local categories);
    // `concat_batches` appends the per-shard dictionaries and offsets their
    // keys, so once the *combined* vocabulary across shards exceeds the narrow
    // key's range the key overflows with `Dictionary key bigger than the key
    // type`. Int32 keys can't overflow at any realistic scale (combined
    // pre-dedup dictionary length ≤ total rows). `unify_dictionary_columns`
    // narrows the key back to the minimal fit after deduplication.
    let batches: Vec<RecordBatch> = raw_batches
        .iter()
        .map(|(_, b)| {
            crate::arrow_compat::upcast_to_large_types(b)
                .and_then(|b| crate::arrow_compat::widen_dictionary_keys(&b))
        })
        .collect::<Result<_>>()?;

    // Reconcile columns that disagree on Dictionary-vs-plain encoding across
    // shards (an append writes obs categoricals as plain Utf8 while
    // `from_anndata` writes them as Dictionary, so a sharded axis can carry both
    // representations). `concat_batches` requires one shared schema, so encode
    // the plain shards' columns to Dictionary before concat. No-op when every
    // shard already agrees.
    let batches = crate::arrow_compat::reconcile_dictionary_representations(batches)?;

    // Verify the shards form a contiguous, ordered cover by walking their
    // stamped metadata. Each shard's `n_rows_total` is the file's logical
    // row count *at the time that shard was written* — for single-pass
    // writes every shard carries the same value, but for append-grown files
    // older shards carry their smaller original stamps while later-appended
    // shards carry the bumped total. So the invariant is: `n_rows_total` is
    // monotonically non-decreasing across shards, and the **last shard's**
    // `n_rows_total` equals the cumulative row cover.
    let first_hdr = parse_shard_metadata(logical, &batches[0])?;
    if first_hdr.shard_idx != 0 {
        return Err(ScxError::InvalidCatalog(format!(
            "{logical}: first shard has shard_idx={} (expected 0)",
            first_hdr.shard_idx
        )));
    }
    if first_hdr.row_start != 0 {
        return Err(ScxError::InvalidCatalog(format!(
            "{logical}: first shard has row_start={} (expected 0)",
            first_hdr.row_start
        )));
    }
    let mut prev_n_rows_total = first_hdr.n_rows_total;
    let mut next_expected_row_start = first_hdr.n_shard_rows;
    for (i, batch) in batches.iter().enumerate().skip(1) {
        let hdr = parse_shard_metadata(logical, batch)?;
        let expected_idx = i as u32;
        if hdr.shard_idx != expected_idx {
            return Err(ScxError::InvalidCatalog(format!(
                "{logical}: shard at position {i} has shard_idx={} (expected {expected_idx})",
                hdr.shard_idx
            )));
        }
        if hdr.n_rows_total < prev_n_rows_total {
            return Err(ScxError::InvalidCatalog(format!(
                "{logical}: shard {i} has n_rows_total={} which contracts the prior \
                 shard's stamp of {prev_n_rows_total} — append-grown obs must stamp \
                 monotonically non-decreasing totals",
                hdr.n_rows_total
            )));
        }
        if hdr.row_start != next_expected_row_start {
            return Err(ScxError::InvalidCatalog(format!(
                "{logical}: shard {i} has row_start={} (expected {next_expected_row_start})",
                hdr.row_start
            )));
        }
        next_expected_row_start = next_expected_row_start.saturating_add(hdr.n_shard_rows);
        prev_n_rows_total = hdr.n_rows_total;
    }
    if next_expected_row_start != prev_n_rows_total {
        return Err(ScxError::InvalidCatalog(format!(
            "{logical}: shards cover {next_expected_row_start} rows but the last shard's \
             n_rows_total is {prev_n_rows_total}"
        )));
    }

    // Concat on the wide schema (every batch was upcast above), then
    // opportunistically narrow back to `Utf8`/`Binary` for columns whose
    // combined offsets still fit in `i32::MAX`.
    let wide_schema = batches[0].schema();
    let concatenated = arrow::compute::concat_batches(&wide_schema, batches.iter())?;
    // Arrow's `concat` appends each shard's dictionary verbatim without
    // deduplicating, so a categorical column that is `["batch1"]` in every
    // one of N shards comes back with a dictionary of `["batch1"; N]`. pandas
    // (`pyarrow.Table.to_pandas()`) rejects non-unique categories, so collapse
    // every dictionary column to a unified dictionary before narrowing.
    let unified = unify_dictionary_columns(&concatenated)?;
    let narrowed = crate::arrow_compat::downcast_large_types(&unified)?;

    // Strip the per-shard metadata (shard_idx / row_start / n_shard_rows)
    // from the merged batch's schema. Keep n_rows_total and any
    // payload-level metadata.
    let narrowed_schema = narrowed.schema();
    let mut clean_metadata = narrowed_schema.metadata().clone();
    clean_metadata.remove("shard_idx");
    clean_metadata.remove("row_start");
    clean_metadata.remove("n_shard_rows");
    let clean_schema = Arc::new(arrow::datatypes::Schema::new_with_metadata(
        narrowed_schema.fields().clone(),
        clean_metadata,
    ));
    Ok(RecordBatch::try_new(
        clean_schema,
        narrowed.columns().to_vec(),
    )?)
}

/// Assemble an **arbitrary, already-row-filtered** subset of metadata
/// shard batches into one logical batch.
///
/// Runs the same `upcast → widen-dict → concat → unify-dict → downcast →
/// strip-per-shard-metadata` pipeline as [`assemble_sharded_metadata`],
/// but WITHOUT the contiguous-cover validation and WITHOUT requiring the
/// per-shard `shard_idx` / `row_start` stamps — the input batches are an
/// arbitrary subset (any order, possibly empty), already filtered to the
/// rows the caller wants. Callers are responsible for passing the batches
/// in the final row order they want concatenated.
///
/// `template_schema` is only consulted when `batches` is empty, to build a
/// correctly-typed 0-row result; pass the schema a normal read of this
/// axis would produce (e.g. a single shard run through this same pipeline,
/// or a prior assembled batch's schema).
///
/// Used by the query engine to rebuild the filtered obs metadata for a
/// `filter_obs(...).collect()` result while only ever holding the matching
/// rows in memory — see `scx-engine/src/collect.rs`.
pub fn assemble_filtered_metadata(
    template_schema: &Arc<arrow::datatypes::Schema>,
    batches: Vec<RecordBatch>,
) -> Result<RecordBatch> {
    if batches.is_empty() {
        return Ok(RecordBatch::new_empty(template_schema.clone()));
    }

    // Widen exactly as `assemble_sharded_metadata` does so concat can't
    // overflow narrow string offsets or per-shard dictionary key widths.
    let wide: Vec<RecordBatch> = batches
        .iter()
        .map(|b| {
            crate::arrow_compat::upcast_to_large_types(b)
                .and_then(|b| crate::arrow_compat::widen_dictionary_keys(&b))
        })
        .collect::<Result<_>>()?;

    // Reconcile Dictionary-vs-plain disagreement across the filtered shards
    // (see `assemble_sharded_metadata`) so concat can't reject a mixed-encoding
    // column produced by an append. No-op when shards already agree.
    let wide = crate::arrow_compat::reconcile_dictionary_representations(wide)?;

    let wide_schema = wide[0].schema();
    let concatenated = arrow::compute::concat_batches(&wide_schema, wide.iter())?;
    let unified = unify_dictionary_columns(&concatenated)?;
    let narrowed = crate::arrow_compat::downcast_large_types(&unified)?;

    // Strip the per-shard metadata keys so the result schema matches a
    // normal (full) read's assembled batch.
    let narrowed_schema = narrowed.schema();
    let mut clean_metadata = narrowed_schema.metadata().clone();
    clean_metadata.remove("shard_idx");
    clean_metadata.remove("row_start");
    clean_metadata.remove("n_shard_rows");
    let clean_schema = Arc::new(arrow::datatypes::Schema::new_with_metadata(
        narrowed_schema.fields().clone(),
        clean_metadata,
    ));
    Ok(RecordBatch::try_new(
        clean_schema,
        narrowed.columns().to_vec(),
    )?)
}

/// Concatenate a list of CSC shards along the column axis.
///
/// Each shard contributes its columns in order; the result's indptr is
/// length `1 + Σ n_cols_in_shard` with cumulative-nnz prefix sums.
/// Indices and data are concatenated verbatim (CSC indices are global
/// row IDs).
///
/// `n_rows` is the global row count (every shard must share this; the
/// caller is responsible for the invariant).
fn concatenate_csc_along_cols(parts: Vec<ScxCsc>, n_rows: usize) -> Result<ScxCsc> {
    if parts.is_empty() {
        return Ok(ScxCsc::new_unchecked(
            (n_rows, 0),
            vec![0],
            Vec::new(),
            Vec::new(),
        ));
    }

    let total_cols: usize = parts.iter().map(|p| p.n_cols()).sum();
    let total_nnz: usize = parts.iter().map(|p| p.nnz()).sum();

    let mut indptr = Vec::with_capacity(total_cols + 1);
    let mut indices = Vec::with_capacity(total_nnz);
    let mut data = Vec::with_capacity(total_nnz);

    indptr.push(0i64);
    let mut cum_nnz: i64 = 0;

    for part in parts {
        let part_n_cols = part.n_cols();
        // Append indptr[1..] with cumulative offset; the leading 0 is
        // already in `indptr` (or is replaced by the previous part's
        // last entry).
        for i in 1..=part_n_cols {
            indptr.push(part.indptr[i] + cum_nnz);
        }
        cum_nnz += part.indptr[part_n_cols];
        indices.extend_from_slice(&part.indices);
        data.extend_from_slice(&part.data);
    }

    Ok(ScxCsc::new_unchecked(
        (n_rows, total_cols),
        indptr,
        indices,
        data,
    ))
}

impl ScxReader {
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
        #[cfg(debug_assertions)]
        self.debug_counts.read_layer.fetch_add(1, Ordering::Relaxed);
        let mut shards = self.legacy_layer_shards(name);

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

    fn legacy_layer_shards(&self, name: &str) -> Vec<&FullCatalogEntry> {
        let prefix = format!("{name}_shard_");
        self.full_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::LayerCsrShard && e.name.starts_with(&prefix))
            .collect()
    }

    /// Read a single LayerCsrShard by `(layer_name, shard_idx)` for the
    /// legacy single-modality naming pattern `{layer_name}_shard_{idx}`.
    /// Shards are sorted by `row_start` before indexing so the caller
    /// can walk them in row order.
    pub fn read_layer_csr_shard(
        &self,
        layer_name: &str,
        shard_idx: usize,
    ) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
        let mut shards = self.legacy_layer_shards(layer_name);
        if shards.is_empty() {
            return Err(ScxError::SectionNotFound(format!("layer '{layer_name}'")));
        }
        shards.sort_by_key(|e| e.stats.as_ref().map_or(u64::MAX, |s| s.row_start));
        if shard_idx >= shards.len() {
            return Err(ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: shards.len(),
            });
        }
        self.read_shard_from_entry(shards[shard_idx])
    }

    /// Per-modality variant of [`Self::read_layer_csr_shard`]: read a
    /// single LayerCsrShard for `(modality_id, layer_name, shard_idx)`
    /// using the multimodal naming pattern
    /// `layer/{mname}/{layer_name}/shard_{idx}`. Shards are returned in
    /// `row_start` order (matching the catalog accessor's sort).
    pub fn read_layer_csr_shard_for(
        &self,
        modality_id: u8,
        layer_name: &str,
        shard_idx: usize,
    ) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
        let shards = self
            .full_catalog
            .layer_csr_shards_for_modality(modality_id, layer_name);
        if shards.is_empty() {
            return Err(ScxError::SectionNotFound(format!(
                "layer '{layer_name}' for modality_id {modality_id}"
            )));
        }
        if shard_idx >= shards.len() {
            return Err(ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: shards.len(),
            });
        }
        self.read_shard_from_entry(shards[shard_idx])
    }

    /// Indptr-only variant of [`Self::read_layer_csr_shard`].
    pub fn read_layer_csr_shard_indptr(
        &self,
        layer_name: &str,
        shard_idx: usize,
    ) -> Result<Vec<i64>> {
        let mut shards = self.legacy_layer_shards(layer_name);
        if shards.is_empty() {
            return Err(ScxError::SectionNotFound(format!("layer '{layer_name}'")));
        }
        shards.sort_by_key(|e| e.stats.as_ref().map_or(u64::MAX, |s| s.row_start));
        if shard_idx >= shards.len() {
            return Err(ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: shards.len(),
            });
        }
        self.read_shard_indptr_from_entry(shards[shard_idx])
    }

    /// Indptr-only variant of [`Self::read_layer_csr_shard_for`].
    pub fn read_layer_csr_shard_indptr_for(
        &self,
        modality_id: u8,
        layer_name: &str,
        shard_idx: usize,
    ) -> Result<Vec<i64>> {
        let shards = self
            .full_catalog
            .layer_csr_shards_for_modality(modality_id, layer_name);
        if shards.is_empty() {
            return Err(ScxError::SectionNotFound(format!(
                "layer '{layer_name}' for modality_id {modality_id}"
            )));
        }
        if shard_idx >= shards.len() {
            return Err(ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: shards.len(),
            });
        }
        self.read_shard_indptr_from_entry(shards[shard_idx])
    }

    /// Number of legacy single-modality LayerCsrShard entries for
    /// `layer_name` (`{layer_name}_shard_{idx}`).
    pub fn layer_csr_shard_count(&self, layer_name: &str) -> usize {
        self.legacy_layer_shards(layer_name).len()
    }

    /// Number of per-modality LayerCsrShard entries for
    /// `(modality_id, layer_name)` (`layer/{mname}/{layer_name}/shard_{idx}`).
    pub fn layer_csr_shard_count_for(&self, modality_id: u8, layer_name: &str) -> usize {
        self.full_catalog
            .layer_csr_shards_for_modality(modality_id, layer_name)
            .len()
    }

    /// Per-modality layer names. For `modality_id == 0`, equivalent to
    /// [`Self::layer_names`] (legacy single-modality pattern). For
    /// `modality_id > 0`, returns the deduplicated layer names parsed
    /// from `layer/{mname}/{layer_name}/shard_{idx}` entries belonging
    /// to that modality.
    pub fn layer_names_for(&self, modality_id: u8) -> Vec<String> {
        if modality_id == 0 {
            return self.layer_names();
        }
        let mut names: Vec<String> = self
            .full_catalog
            .entries
            .iter()
            .filter(|e| {
                e.section_type == SectionType::LayerCsrShard && e.modality_id == modality_id
            })
            .filter_map(|e| {
                // `layer/{mname}/{layer_name}/shard_{idx}` —
                // the `{layer_name}` segment lives between the second
                // `/` and the trailing `/shard_{idx}`.
                let after_first = e.name.strip_prefix("layer/")?;
                let (_mname, rest) = after_first.split_once('/')?;
                let pos = rest.rfind("/shard_")?;
                Some(rest[..pos].to_string())
            })
            .collect();
        names.sort();
        names.dedup();
        names
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
        let slice = self.section_bytes(entry)?;
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
        let slice = self.section_bytes(entry)?;
        Provenance::read_from(&mut Cursor::new(slice), slice.len())
    }

    // -----------------------------------------------------------------------
    // Predicate indexes (docs/format.md (Predicate Indexes), Phase 2)
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Deletion vectors
    // -----------------------------------------------------------------------

    /// Read deletion vectors if present. Returns `Ok(None)` if the file has no DV flag set.
    #[cfg(feature = "deletion-vectors")]
    /// Phase 5b: read a single detection-bitmap shard by index, scoped
    /// to a modality (`modality_id == 0` for unimodal files). Shards
    /// are ordered by `row_start`, matching the CSR shard order.
    #[cfg(feature = "deletion-vectors")]
    pub fn read_bitmap_shard_for(
        &self,
        modality_id: u8,
        shard_idx: usize,
    ) -> Result<crate::bitmap::BitmapShard> {
        let entries = self.full_catalog.bitmap_shards_for_modality(modality_id);
        let entry = entries.get(shard_idx).ok_or_else(|| {
            ScxError::SectionNotFound(format!(
                "X/bitmap/shard_{shard_idx} (modality_id={modality_id})"
            ))
        })?;
        let slice = self.section_bytes(entry)?;
        crate::bitmap::BitmapShard::read_from(&mut Cursor::new(slice), slice.len())
    }

    /// Phase 5b: unimodal helper — equivalent to
    /// `read_bitmap_shard_for(0, shard_idx)`.
    #[cfg(feature = "deletion-vectors")]
    pub fn read_bitmap_shard(&self, shard_idx: usize) -> Result<crate::bitmap::BitmapShard> {
        self.read_bitmap_shard_for(0, shard_idx)
    }

    /// Number of bitmap shards available for a modality (0 if none).
    #[cfg(feature = "deletion-vectors")]
    pub fn bitmap_shard_count(&self, modality_id: u8) -> usize {
        self.full_catalog
            .bitmap_shards_for_modality(modality_id)
            .len()
    }

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
        let slice = self.section_bytes(entry)?;
        let dv = crate::deletion_vectors::DeletionVectors::read_from(
            &mut Cursor::new(slice),
            slice.len(),
        )?;
        Ok(Some(dv))
    }

    /// Build the obs-indexed keep mask (`true` = retained) implied by this
    /// file's deletion vectors, or `None` when the file has no deletion
    /// vectors or nothing is deleted.
    ///
    /// This is the single entry point that the reader's CSR filter, the
    /// h5ad/h5mu streaming export, `scx compact`, and the `pyscx` obs filter
    /// all share, so the row-keep semantics stay identical everywhere. The
    /// mask construction itself lives in
    /// [`crate::DeletionVectors::build_keep_mask`].
    pub fn deletion_keep_mask(&self) -> Result<Option<Vec<bool>>> {
        if !self.header.has_deletion_vectors() {
            return Ok(None);
        }
        let dv = match self.read_deletion_vectors()? {
            Some(dv) if dv.total_deleted() > 0 => dv,
            _ => return Ok(None),
        };
        Ok(Some(dv.build_keep_mask(
            self.n_obs() as usize,
            &self.full_catalog,
        )))
    }

    /// Read all CSR shards with deletion vectors applied.
    /// Deleted rows are excluded from the returned ScxCsr.
    /// If no deletion vectors are present, returns the same result as `read_all_csr_shards()`.
    #[cfg(feature = "deletion-vectors")]
    pub fn read_all_csr_shards_filtered(&self) -> Result<ScxCsr> {
        let csr = self.read_all_csr_shards()?;
        self.filter_csr_rows_by_deletion_vectors(csr)
    }

    /// Per-modality counterpart of [`Self::read_all_csr_shards_filtered`].
    /// Assembles the modality's CSR via [`Self::read_all_csr_shards_for`]
    /// and applies the file-wide deletion-vector keep mask. The mask is
    /// obs-indexed and shared across modalities (h5mu invariant), so
    /// each modality's filtered CSR has `n_obs - n_deleted` rows.
    #[cfg(feature = "deletion-vectors")]
    pub fn read_all_csr_shards_for_filtered(&self, modality_id: u8) -> Result<ScxCsr> {
        let csr = self.read_all_csr_shards_for(modality_id)?;
        self.filter_csr_rows_by_deletion_vectors(csr)
    }

    /// Read a named layer with deletion vectors applied.
    /// Deleted rows are excluded from the returned ScxCsr.
    /// If no deletion vectors are present, returns the same result as `read_layer()`.
    ///
    /// Layers must share X's row count (an AnnData invariant); the row-keep
    /// mask is built from the X-shard layout and applied to the layer CSR.
    #[cfg(feature = "deletion-vectors")]
    pub fn read_layer_filtered(&self, name: &str) -> Result<ScxCsr> {
        let csr = self.read_layer(name)?;
        self.filter_csr_rows_by_deletion_vectors(csr)
    }

    /// Apply deletion vectors to an already-assembled CSR (X or layer).
    /// The keep mask is derived from the X-shard layout in the full catalog,
    /// so the input CSR must share X's row count.
    #[cfg(feature = "deletion-vectors")]
    fn filter_csr_rows_by_deletion_vectors(&self, csr: ScxCsr) -> Result<ScxCsr> {
        // The keep mask is obs-indexed; the input CSR (X or a layer) shares
        // X's row count, so the mask aligns with its rows.
        let keep = match self.deletion_keep_mask()? {
            Some(keep) => keep,
            None => return Ok(csr),
        };

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

    /// Expose the underlying `Mmap` for range-specific madvise calls.
    pub(crate) fn mmap_ref(&self) -> &Mmap {
        &self.mmap
    }

    /// Hint sequential access for a byte range (`MADV_SEQUENTIAL`).
    #[cfg(unix)]
    fn advise_sequential(&self, offset: usize, len: usize) {
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

    /// Hint that a byte range is no longer needed (`MADV_DONTNEED`).
    ///
    /// # Safety
    /// `UncheckedAdvice::DontNeed` may discard dirty pages on some platforms,
    /// but our mmap is read-only so this is safe.
    #[cfg(unix)]
    #[allow(dead_code)]
    unsafe fn advise_dontneed(&self, offset: usize, len: usize) {
        use memmap2::UncheckedAdvice;
        let _ = self
            .mmap
            .unchecked_advise_range(UncheckedAdvice::DontNeed, offset, len);
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
            let slice = self.section_bytes(entry)?;
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
            return Err(ScxError::ChecksumMismatch {
                section: "essential section(s)".to_string(),
            });
        }

        Ok(results)
    }

    /// Read and parse a `DecodeMetadataShard` section.
    pub fn read_decode_sidecar_from_entry(
        &self,
        entry: &FullCatalogEntry,
    ) -> Result<DecodeSidecar> {
        if entry.section_type != SectionType::DecodeMetadataShard {
            return Err(ScxError::InvalidCatalog(format!(
                "entry {} is {:?}, not DecodeMetadataShard",
                entry.name, entry.section_type
            )));
        }
        let section = self.section_bytes(entry)?;
        DecodeSidecar::read_from(&mut Cursor::new(section), section.len())
    }

    /// Validate one decode sidecar against its source catalog entry and shard header.
    pub fn validate_decode_sidecar_entry(&self, entry: &FullCatalogEntry) -> Result<()> {
        let sidecar = self.read_decode_sidecar_from_entry(entry)?;
        if !Self::is_canonical_csr_section(sidecar.target_section_type) {
            return Err(ScxError::InvalidCatalog(format!(
                "decode sidecar {} targets non-CSR section type {:?}",
                entry.name, sidecar.target_section_type
            )));
        }

        let source = self
            .full_catalog
            .entries
            .iter()
            .find(|source| {
                source.offset == sidecar.source_section_offset
                    && source.length == sidecar.source_section_length
                    && source.checksum == sidecar.source_section_checksum
            })
            .ok_or_else(|| {
                ScxError::InvalidCatalog(format!(
                    "decode sidecar {} references missing source section at offset {} length {}",
                    entry.name, sidecar.source_section_offset, sidecar.source_section_length
                ))
            })?;

        if source.section_type != sidecar.target_section_type {
            return Err(ScxError::InvalidCatalog(format!(
                "decode sidecar {} target type {:?} != source {} type {:?}",
                entry.name, sidecar.target_section_type, source.name, source.section_type
            )));
        }
        if source.modality_id != entry.modality_id {
            return Err(ScxError::InvalidCatalog(format!(
                "decode sidecar {} modality {} != source {} modality {}",
                entry.name, entry.modality_id, source.name, source.modality_id
            )));
        }

        let shard_header = self.read_shard_header(source)?;
        let codec_id = CodecId::from_u8(shard_header.codec_id)
            .ok_or(ScxError::UnknownCodec(shard_header.codec_id))?;
        if codec_id != CodecId::Scx1 {
            return Err(ScxError::InvalidCatalog(format!(
                "decode sidecar {} source {} uses codec {:?}, expected Scx1",
                entry.name, source.name, codec_id
            )));
        }
        let value_encoding = ValueEncoding::from_u8(shard_header.value_encoding)
            .ok_or(ScxError::UnknownValueEncoding(shard_header.value_encoding))?;
        if !value_encoding.is_integer() {
            return Err(ScxError::InvalidCatalog(format!(
                "decode sidecar {} source {} uses non-integer value encoding {:?}",
                entry.name, source.name, value_encoding
            )));
        }
        if shard_header.codec_id != sidecar.codec_id
            || shard_header.value_encoding != sidecar.value_encoding
            || shard_header.index_dtype != sidecar.index_dtype
            || shard_header.n_major != sidecar.n_rows
            || shard_header.n_minor != sidecar.n_cols
            || shard_header.nnz != sidecar.nnz
            || shard_header.global_offset != sidecar.major_start
        {
            return Err(ScxError::InvalidCatalog(format!(
                "decode sidecar {} shape/header metadata does not match source {}",
                entry.name, source.name
            )));
        }

        if let Some(stats) = source.stats.as_ref() {
            let expected_end = sidecar
                .major_start
                .checked_add(sidecar.n_rows as u64)
                .ok_or_else(|| {
                    ScxError::InvalidCatalog(format!(
                        "decode sidecar {} row range overflows u64",
                        entry.name
                    ))
                })?;
            if stats.nnz != sidecar.nnz
                || stats.major_start(source.section_type) != sidecar.major_start
                || stats.major_end(source.section_type) != expected_end
            {
                return Err(ScxError::InvalidCatalog(format!(
                    "decode sidecar {} stats do not match source {}",
                    entry.name, source.name
                )));
            }
        }

        Self::validate_decode_sidecar_entries(&entry.name, &sidecar)?;

        // Decode-parity: decode the source shard **through** the sidecar's
        // recorded row/Rice-block offsets and assert byte-equality with the
        // canonical decode. This is the safety net the encoder-emitted offsets
        // depend on — a sidecar is only trustworthy if a consumer seeking to its
        // offsets reproduces exactly what the normal decoder produces.
        let section = self.section_bytes(source)?;
        let slice_stream = |rel: u32, len: u32, label: &str| -> Result<&[u8]> {
            let start = rel as usize;
            let end = start
                .checked_add(len as usize)
                .filter(|&e| e <= section.len())
                .ok_or_else(|| {
                    ScxError::InvalidCatalog(format!(
                        "decode sidecar {} source {} {label} stream out of bounds",
                        entry.name, source.name
                    ))
                })?;
            Ok(&section[start..end])
        };
        let encoded_ref = scx_codec::EncodedShardRef {
            indptr_bytes: slice_stream(
                shard_header.indptr_rel_offset,
                shard_header.indptr_length,
                "indptr",
            )?,
            indices_bytes: slice_stream(
                shard_header.indices_rel_offset,
                shard_header.indices_length,
                "indices",
            )?,
            values_bytes: slice_stream(
                shard_header.values_rel_offset,
                shard_header.values_length,
                "values",
            )?,
        };
        let n_rows = shard_header.n_major as usize;
        let nnz = shard_header.nnz as usize;
        let index_dtype_u16 = shard_header.index_dtype == 0;
        let via_offsets = scx_codec::decode_scx1_with_metadata(
            &encoded_ref,
            value_encoding,
            n_rows,
            nnz,
            &sidecar.to_scx1_metadata(),
        )
        .map_err(|e| {
            ScxError::InvalidCatalog(format!(
                "decode sidecar {} parity decode (via offsets) failed: {e}",
                entry.name
            ))
        })?;
        let canonical = scx_codec::decode_shard_ref(
            &encoded_ref,
            codec_id,
            value_encoding,
            n_rows,
            nnz,
            index_dtype_u16,
        )
        .map_err(|e| {
            ScxError::InvalidCatalog(format!(
                "decode sidecar {} canonical decode of source {} failed: {e}",
                entry.name, source.name
            ))
        })?;
        if via_offsets != canonical {
            return Err(ScxError::InvalidCatalog(format!(
                "decode sidecar {} parity mismatch: decode-via-offsets != canonical decode of source {}",
                entry.name, source.name
            )));
        }

        Ok(())
    }

    /// Validate every decode sidecar section.
    pub fn validate_decode_sidecars(&self) -> Vec<(String, bool)> {
        self.full_catalog
            .entries
            .iter()
            .filter(|entry| entry.section_type == SectionType::DecodeMetadataShard)
            .map(|entry| {
                (
                    entry.name.clone(),
                    self.validate_decode_sidecar_entry(entry).is_ok(),
                )
            })
            .collect()
    }

    /// Validate that a row-major sparse shard obeys the v3 canonical CSR invariant.
    pub fn validate_canonical_csr_entry(&self, entry: &FullCatalogEntry) -> Result<()> {
        if !Self::is_canonical_csr_section(entry.section_type) {
            return Err(ScxError::InvalidCatalog(format!(
                "entry {} is {:?}, not a canonical CSR shard type",
                entry.name, entry.section_type
            )));
        }
        let shard_header = self.read_shard_header(entry)?;
        let (indptr, indices, data) = self.read_shard_from_entry_verified(entry)?;
        Self::validate_decoded_canonical_csr(&entry.name, &shard_header, &indptr, &indices, &data)
    }

    /// Validate all row-major sparse shards against the v3 canonical CSR invariant.
    pub fn validate_canonical_csr_shards(&self) -> Vec<(String, bool)> {
        self.full_catalog
            .entries
            .iter()
            .filter(|entry| Self::is_canonical_csr_section(entry.section_type))
            .map(|entry| {
                (
                    entry.name.clone(),
                    self.validate_canonical_csr_entry(entry).is_ok(),
                )
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Get the raw bytes for a catalog entry from the mmap.
    pub fn section_bytes(&self, entry: &FullCatalogEntry) -> Result<&[u8]> {
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

    /// Read the shard header from a catalog entry without decoding the shard data.
    ///
    /// Useful when callers need per-shard codec/encoding info before or alongside
    /// `read_shard_from_entry`.
    pub fn read_shard_header(&self, entry: &FullCatalogEntry) -> Result<ShardHeader> {
        let section = self.section_bytes(entry)?;
        let vs = crate::validated_section::ValidatedSection::new(section);
        ShardHeader::read_from(&mut Cursor::new(vs.header()?))
    }

    /// Read raw shard bytes (header + compressed payload) without decoding.
    ///
    /// Returns the entire section as a byte slice from the mmap. Useful for
    /// verbatim shard copying (e.g., `streaming_save_layer` copying X shards
    /// unchanged) where decoding and re-encoding would be wasteful.
    pub fn read_raw_shard_bytes(&self, entry: &FullCatalogEntry) -> Result<&[u8]> {
        self.section_bytes(entry)
    }

    /// Read and decode a single shard from a catalog entry.
    ///
    /// Skips per-shard checksum verification for performance. The catalog
    /// checksum verified at `ScxReader::open()` authenticates the catalog
    /// payload (offsets, lengths, per-section checksums) but does **not**
    /// re-hash section bytes — a corrupted shard payload will not be
    /// detected here. Use [`read_shard_from_entry_verified`] or
    /// [`ScxReader::validate`] when section-level integrity must be
    /// confirmed (e.g., `scx validate`).
    pub fn read_shard_from_entry(
        &self,
        entry: &FullCatalogEntry,
    ) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
        self.read_shard_from_entry_inner(entry, false)
    }

    /// Read and decode a single shard with explicit checksum verification.
    ///
    /// Computes the BLAKE3 hash of the shard payload and compares it to the
    /// truncated 8-byte checksum in the shard header. Use this for `scx validate`
    /// or when data integrity must be confirmed per-shard.
    pub fn read_shard_from_entry_verified(
        &self,
        entry: &FullCatalogEntry,
    ) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
        self.read_shard_from_entry_inner(entry, true)
    }

    /// Resolve + load the decode sidecar for a CSR shard `entry` (named
    /// `decode/<name>`), returning `None` unless a matching, **fresh**, Scx1
    /// sidecar exists. Freshness is the `(offset, length, checksum)` triple — a
    /// stale sidecar (source rewritten) is ignored, never trusted.
    fn scx1_sidecar_for(&self, entry: &FullCatalogEntry) -> Result<Option<DecodeSidecar>> {
        let name = format!("decode/{}", entry.name);
        let Some(sc_entry) = self.full_catalog.get(&name) else {
            return Ok(None);
        };
        if sc_entry.section_type != SectionType::DecodeMetadataShard {
            return Ok(None);
        }
        let sidecar = self.read_decode_sidecar_from_entry(sc_entry)?;
        if sidecar.source_section_offset != entry.offset
            || sidecar.source_section_length != entry.length
            || sidecar.source_section_checksum != entry.checksum
            || sidecar.codec_id != CodecId::Scx1 as u8
        {
            return Ok(None);
        }
        Ok(Some(sidecar))
    }

    /// Resolve the codec-level [`scx_codec::Scx1DecodeMetadata`] for a CSR shard
    /// `entry`, or `None` when no fresh Scx1 decode sidecar exists (non-Scx1
    /// codec, stale, or absent — never an error). This is the public seam GPU
    /// device-decode consumers (`to_gpu_anndata`) call to drive
    /// `scx_gpu::decode_csr_shards_to_device_with_metadata` directly from the
    /// encoder-emitted offsets instead of a CPU prescan (ACC-RUST-OPT-V4 4.4a).
    pub fn scx1_metadata_for(
        &self,
        entry: &FullCatalogEntry,
    ) -> Result<Option<scx_codec::Scx1DecodeMetadata>> {
        Ok(self
            .scx1_sidecar_for(entry)?
            .map(|sidecar| sidecar.to_scx1_metadata()))
    }

    /// Memoized [`Self::scx1_metadata_for`] keyed by `entry.offset` (lever S).
    /// Returns a shared `Arc` so the parsed metadata
    /// is resolved once per shard and reused across the many row-range decodes
    /// of the scattered cell-set gather, instead of re-running the sidecar
    /// read (BLAKE3) + `to_scx1_metadata` (O(shard-rows) clone) every call.
    /// `None` (no fresh Scx1 sidecar) behaves exactly as the uncached path.
    /// Byte-identical: the metadata is a pure function of the read-only bytes.
    fn scx1_metadata_cached(
        &self,
        entry: &FullCatalogEntry,
    ) -> Result<Option<Arc<scx_codec::Scx1DecodeMetadata>>> {
        if let Some(meta) = self.sidecar_meta_cache.lock().unwrap().get(&entry.offset) {
            return Ok(Some(Arc::clone(meta)));
        }
        let Some(sidecar) = self.scx1_sidecar_for(entry)? else {
            return Ok(None);
        };
        #[cfg(debug_assertions)]
        self.debug_counts
            .sidecar_meta_parse
            .fetch_add(1, Ordering::Relaxed);
        let meta = Arc::new(sidecar.to_scx1_metadata());
        // A racing peer may have inserted/parsed the same shard meanwhile; `put`
        // is idempotent (same deterministic bytes), so overwrite is harmless.
        self.sidecar_meta_cache
            .lock()
            .unwrap()
            .put(entry.offset, Arc::clone(&meta));
        Ok(Some(meta))
    }

    /// Slice a shard section's three encoded streams into an `EncodedShardRef`
    /// using the `ShardHeader` relative offsets (mirrors the slicing in
    /// `validate_decode_sidecar_entry`).
    fn encoded_ref_from_section<'a>(
        section: &'a [u8],
        header: &ShardHeader,
        name: &str,
    ) -> Result<scx_codec::EncodedShardRef<'a>> {
        let slice = |rel: u32, len: u32, label: &str| -> Result<&'a [u8]> {
            let start = rel as usize;
            let end = start
                .checked_add(len as usize)
                .filter(|&e| e <= section.len())
                .ok_or_else(|| {
                    ScxError::InvalidCatalog(format!("shard {name} {label} stream out of bounds"))
                })?;
            Ok(&section[start..end])
        };
        Ok(scx_codec::EncodedShardRef {
            indptr_bytes: slice(header.indptr_rel_offset, header.indptr_length, "indptr")?,
            indices_bytes: slice(header.indices_rel_offset, header.indices_length, "indices")?,
            values_bytes: slice(header.values_rel_offset, header.values_length, "values")?,
        })
    }

    /// Random-access decode of a contiguous row range `[row_start, row_start +
    /// n_rows)` of a CSR shard via its decode sidecar (O(window), not O(shard)).
    ///
    /// Returns `Ok(None)` when the shard has no usable Scx1 sidecar (no sidecar,
    /// stale, or a non-Scx1 codec) — callers then fall back to a full
    /// [`read_shard_from_entry`](Self::read_shard_from_entry) + slice. The
    /// `Some` result is byte-identical to that fallback's matching row slice.
    pub fn decode_scx1_row_range(
        &self,
        entry: &FullCatalogEntry,
        row_start: usize,
        n_rows: usize,
    ) -> Result<Option<scx_codec::ScipyShard>> {
        let Some(meta) = self.scx1_metadata_cached(entry)? else {
            return Ok(None);
        };
        let section = self.section_bytes(entry)?;
        let header = self.read_shard_header(entry)?;
        let venc = ValueEncoding::from_u8(header.value_encoding)
            .ok_or(ScxError::UnknownValueEncoding(header.value_encoding))?;
        let encoded = Self::encoded_ref_from_section(section, &header, &entry.name)?;
        let decoded = scx_codec::decode_scx1_row_range(&encoded, venc, &meta, row_start, n_rows)
            .map_err(|e| {
                ScxError::InvalidCatalog(format!("sidecar row-range decode of {}: {e}", entry.name))
            })?;
        let scipy = scx_codec::decoded_shard_to_scipy(decoded, venc).map_err(|e| {
            ScxError::InvalidCatalog(format!("sidecar row-range convert of {}: {e}", entry.name))
        })?;
        #[cfg(debug_assertions)]
        self.debug_counts
            .decode_scx1_row_range
            .fetch_add(1, Ordering::Relaxed);
        Ok(Some(scipy))
    }

    /// Decode several contiguous row `runs` (`(row_start, n_rows)`) of one CSR
    /// shard `entry` via its decode sidecar, resolving the sidecar metadata,
    /// section bytes, header, and encoded streams **once** for the whole shard
    /// (vs once per run in [`Self::decode_scx1_row_range`]). This is the hot
    /// path for the scattered cell-set gather, where a single shard is touched
    /// by many small runs in one batch — repeating the O(shard-rows) sidecar
    /// metadata deserialization per run dominated the gather (~3 ms/cell).
    /// Returns `Ok(None)` (no fresh Scx1 sidecar)
    /// so the caller falls back to a full-shard decode; each run's result is
    /// byte-identical to that fallback's matching slice.
    pub fn decode_scx1_row_runs(
        &self,
        entry: &FullCatalogEntry,
        runs: &[(usize, usize)],
    ) -> Result<Option<Vec<scx_codec::ScipyShard>>> {
        let Some(meta) = self.scx1_metadata_cached(entry)? else {
            return Ok(None);
        };
        let section = self.section_bytes(entry)?;
        let header = self.read_shard_header(entry)?;
        let venc = ValueEncoding::from_u8(header.value_encoding)
            .ok_or(ScxError::UnknownValueEncoding(header.value_encoding))?;
        let encoded = Self::encoded_ref_from_section(section, &header, &entry.name)?;
        let mut out = Vec::with_capacity(runs.len());
        for &(row_start, n_rows) in runs {
            let decoded =
                scx_codec::decode_scx1_row_range(&encoded, venc, &meta, row_start, n_rows)
                    .map_err(|e| {
                        ScxError::InvalidCatalog(format!(
                            "sidecar row-range decode of {}: {e}",
                            entry.name
                        ))
                    })?;
            let scipy = scx_codec::decoded_shard_to_scipy(decoded, venc).map_err(|e| {
                ScxError::InvalidCatalog(format!(
                    "sidecar row-range convert of {}: {e}",
                    entry.name
                ))
            })?;
            out.push(scipy);
            #[cfg(debug_assertions)]
            self.debug_counts
                .decode_scx1_row_range
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(Some(out))
    }

    fn read_shard_from_entry_inner(
        &self,
        entry: &FullCatalogEntry,
        verify_checksum: bool,
    ) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
        #[cfg(debug_assertions)]
        self.debug_counts
            .read_shard_from_entry
            .fetch_add(1, Ordering::Relaxed);
        let section = self.section_bytes(entry)?;
        // Parallel sidecar-driven decode (Task 4.3): for large Scx1 shards that
        // carry a fresh decode sidecar, decode the rows concurrently via the
        // recorded offsets — byte-identical to the sequential path. Only on the
        // non-verifying read path (the verifying path keeps its single-pass
        // checksum+decode). Falls through when no sidecar / small / feature off.
        #[cfg(feature = "parallel")]
        if !verify_checksum {
            if let Some(scipy) = self.try_parallel_sidecar_decode(entry, section)? {
                return Ok(scipy);
            }
        }
        crate::shard_decode::decode_shard_bytes(
            section,
            entry,
            self.full_catalog.catalog_version,
            verify_checksum,
        )
    }

    /// Minimum shard rows before the parallel sidecar decode is worth its setup.
    #[cfg(feature = "parallel")]
    const PARALLEL_SIDECAR_DECODE_MIN_ROWS: u32 = 4096;

    #[cfg(feature = "parallel")]
    fn try_parallel_sidecar_decode(
        &self,
        entry: &FullCatalogEntry,
        section: &[u8],
    ) -> Result<Option<scx_codec::ScipyShard>> {
        let header = self.read_shard_header(entry)?;
        if header.n_major < Self::PARALLEL_SIDECAR_DECODE_MIN_ROWS {
            return Ok(None);
        }
        let Some(sidecar) = self.scx1_sidecar_for(entry)? else {
            return Ok(None);
        };
        let venc = ValueEncoding::from_u8(header.value_encoding)
            .ok_or(ScxError::UnknownValueEncoding(header.value_encoding))?;
        let encoded = Self::encoded_ref_from_section(section, &header, &entry.name)?;
        let chunk_rows = (header.n_major as usize)
            .div_ceil(rayon::current_num_threads().max(1) * 4)
            .max(1024);
        let decoded = crate::decode_sidecar::decode_scx1_parallel(
            &encoded,
            venc,
            &sidecar.to_scx1_metadata(),
            chunk_rows,
        )
        .map_err(|e| {
            ScxError::InvalidCatalog(format!("parallel sidecar decode of {}: {e}", entry.name))
        })?;
        let scipy = scx_codec::decoded_shard_to_scipy(decoded, venc).map_err(|e| {
            ScxError::InvalidCatalog(format!("parallel sidecar convert of {}: {e}", entry.name))
        })?;
        Ok(Some(scipy))
    }

    /// Read only the indptr (row-pointer) array of a shard, skipping
    /// indices/data decode entirely. For callers that need just the
    /// per-row nnz counts (e.g. the streaming SCX → h5ad export's
    /// `precompute_total_nnz` when a deletion vector is active).
    pub fn read_shard_indptr_from_entry(&self, entry: &FullCatalogEntry) -> Result<Vec<i64>> {
        let section = self.section_bytes(entry)?;
        crate::shard_decode::decode_shard_indptr_bytes(
            section,
            entry,
            self.full_catalog.catalog_version,
        )
    }

    fn is_canonical_csr_section(section_type: SectionType) -> bool {
        matches!(
            section_type,
            SectionType::CsrShard | SectionType::LayerCsrShard | SectionType::ObspCsrShard
        )
    }

    fn validate_decode_sidecar_entries(name: &str, sidecar: &DecodeSidecar) -> Result<()> {
        let mut expected_value_start = 0u64;
        for (row_idx, row) in sidecar.rows.iter().enumerate() {
            if row.value_start != expected_value_start {
                return Err(ScxError::InvalidCatalog(format!(
                    "decode sidecar {name} row {row_idx} value_start {} != expected {}",
                    row.value_start, expected_value_start
                )));
            }
            if row.frame_bits > 32 {
                return Err(ScxError::InvalidCatalog(format!(
                    "decode sidecar {name} row {row_idx} frame_bits {} exceeds 32",
                    row.frame_bits
                )));
            }
            if row.index_packing > 2 || (row.nnz == 0 && row.index_packing != 0) {
                return Err(ScxError::InvalidCatalog(format!(
                    "decode sidecar {name} row {row_idx} has invalid index_packing {}",
                    row.index_packing
                )));
            }
            expected_value_start = expected_value_start
                .checked_add(row.nnz as u64)
                .ok_or_else(|| {
                    ScxError::InvalidCatalog(format!(
                        "decode sidecar {name} row nnz sum overflows u64"
                    ))
                })?;
        }
        if expected_value_start != sidecar.nnz {
            return Err(ScxError::InvalidCatalog(format!(
                "decode sidecar {name} row coverage {} != nnz {}",
                expected_value_start, sidecar.nnz
            )));
        }

        let mut expected_value_start = 0u64;
        for (block_idx, block) in sidecar.rice_blocks.iter().enumerate() {
            if block.value_start != expected_value_start {
                return Err(ScxError::InvalidCatalog(format!(
                    "decode sidecar {name} Rice block {block_idx} value_start {} != expected {}",
                    block.value_start, expected_value_start
                )));
            }
            if block.n_values == 0 || block.n_values as usize > scx_codec::rice::B_VAL {
                return Err(ScxError::InvalidCatalog(format!(
                    "decode sidecar {name} Rice block {block_idx} has invalid n_values {}",
                    block.n_values
                )));
            }
            if block.k > scx_codec::rice::MAX_RICE_K {
                return Err(ScxError::InvalidCatalog(format!(
                    "decode sidecar {name} Rice block {block_idx} k {} exceeds {}",
                    block.k,
                    scx_codec::rice::MAX_RICE_K
                )));
            }
            expected_value_start = expected_value_start
                .checked_add(block.n_values as u64)
                .ok_or_else(|| {
                    ScxError::InvalidCatalog(format!(
                        "decode sidecar {name} Rice block coverage overflows u64"
                    ))
                })?;
        }
        if expected_value_start != sidecar.nnz {
            return Err(ScxError::InvalidCatalog(format!(
                "decode sidecar {name} Rice block coverage {} != nnz {}",
                expected_value_start, sidecar.nnz
            )));
        }

        Ok(())
    }

    fn validate_decoded_canonical_csr(
        name: &str,
        header: &ShardHeader,
        indptr: &[i64],
        indices: &[i32],
        data: &[f32],
    ) -> Result<()> {
        let expected_indptr_len = header.n_major as usize + 1;
        if indptr.len() != expected_indptr_len {
            return Err(ScxError::InvalidCatalog(format!(
                "CSR shard {name} indptr len {} != n_major + 1 {}",
                indptr.len(),
                expected_indptr_len
            )));
        }
        if indptr.first().copied() != Some(0) {
            return Err(ScxError::InvalidCatalog(format!(
                "CSR shard {name} indptr must start at 0"
            )));
        }
        if indices.len() != data.len() {
            return Err(ScxError::InvalidCatalog(format!(
                "CSR shard {name} indices len {} != data len {}",
                indices.len(),
                data.len()
            )));
        }
        if header.nnz != indices.len() as u64 {
            return Err(ScxError::InvalidCatalog(format!(
                "CSR shard {name} header nnz {} != decoded nnz {}",
                header.nnz,
                indices.len()
            )));
        }

        for (row, window) in indptr.windows(2).enumerate() {
            let start_raw = window[0];
            let end_raw = window[1];
            if start_raw < 0 || end_raw < 0 || end_raw < start_raw {
                return Err(ScxError::InvalidCatalog(format!(
                    "CSR shard {name} invalid indptr window at row {row}: {start_raw}..{end_raw}"
                )));
            }
            let start = usize::try_from(start_raw).map_err(|_| {
                ScxError::InvalidCatalog(format!(
                    "CSR shard {name} row {row} start overflows usize"
                ))
            })?;
            let end = usize::try_from(end_raw).map_err(|_| {
                ScxError::InvalidCatalog(format!("CSR shard {name} row {row} end overflows usize"))
            })?;
            if end > indices.len() {
                return Err(ScxError::InvalidCatalog(format!(
                    "CSR shard {name} row {row} end {end} exceeds decoded nnz {}",
                    indices.len()
                )));
            }

            let mut prev: Option<i32> = None;
            for pos in start..end {
                let idx = indices[pos];
                if idx < 0 || idx as u64 >= header.n_minor as u64 {
                    return Err(ScxError::InvalidCatalog(format!(
                        "CSR shard {name} row {row} index {idx} out of range [0, {})",
                        header.n_minor
                    )));
                }
                if let Some(prev_idx) = prev {
                    if idx <= prev_idx {
                        return Err(ScxError::InvalidCatalog(format!(
                        "CSR shard {name} row {row} indices are not strictly increasing: {idx} after {prev_idx}"
                    )));
                    }
                }
                if data[pos] == 0.0 {
                    return Err(ScxError::InvalidCatalog(format!(
                        "CSR shard {name} row {row} stores explicit zero at ordinal {pos}"
                    )));
                }
                prev = Some(idx);
            }
        }

        let last = *indptr.last().unwrap_or(&-1);
        if last < 0 || last as usize != indices.len() {
            return Err(ScxError::InvalidCatalog(format!(
                "CSR shard {name} indptr last {last} != decoded nnz {}",
                indices.len()
            )));
        }

        Ok(())
    }

    /// Assemble multiple shard entries into a single ScxCsr using parallel decode.
    ///
    /// Pre-allocates the final merged arrays to exact sizes using catalog stats,
    /// then decodes each shard in parallel directly into its non-overlapping region.
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

        // Hint aggressive readahead across the shard region.
        // Use min/max of file offsets since shards are sorted by row_start,
        // not file offset — they may not be contiguous after append/compact.
        #[cfg(unix)]
        {
            let min_offset = shards.iter().map(|e| e.offset as usize).min().unwrap();
            let max_end = shards
                .iter()
                .map(|e| (e.offset + e.length) as usize)
                .max()
                .unwrap();
            self.advise_sequential(min_offset, max_end - min_offset);
        }

        // Pre-compute per-shard (n_rows, nnz) from catalog stats
        let shard_sizes: Vec<(usize, usize)> = shards
            .iter()
            .map(|e| {
                let stats = e.stats.as_ref().ok_or_else(|| {
                    ScxError::InvalidCatalog(format!(
                        "shard entry '{}' at offset {} has no stats block",
                        e.name, e.offset
                    ))
                })?;
                Ok::<_, ScxError>((
                    (stats.row_end - stats.row_start) as usize,
                    stats.nnz as usize,
                ))
            })
            .collect::<Result<_>>()?;
        let total_rows: usize = shard_sizes.iter().map(|(r, _)| *r).sum();
        let total_nnz: usize = shard_sizes.iter().map(|(_, n)| *n).sum();

        // Compute per-shard cumulative offsets
        let mut row_offsets = Vec::with_capacity(shards.len());
        let mut nnz_offsets = Vec::with_capacity(shards.len());
        let (mut cum_rows, mut cum_nnz) = (0usize, 0usize);
        for &(n_rows, nnz) in &shard_sizes {
            row_offsets.push(cum_rows);
            nnz_offsets.push(cum_nnz);
            cum_rows += n_rows;
            cum_nnz += nnz;
        }

        // Single allocation for final merged arrays
        let mut indptr = vec![0i64; total_rows + 1];
        let mut indices = vec![0i32; total_nnz];
        let mut data = vec![0f32; total_nnz];

        // Parallel decode + copy into non-overlapping regions.
        // Store base addresses as usize so they can cross thread boundaries
        // (usize is Send+Sync; raw pointers are not).
        // SAFETY: each rayon task writes to a disjoint region determined by
        // pre-computed offsets, so there are no data races.
        let indptr_base = indptr.as_mut_ptr() as usize;
        let indices_base = indices.as_mut_ptr() as usize;
        let data_base = data.as_mut_ptr() as usize;

        shards.par_iter().enumerate().try_for_each(|(i, entry)| {
            let (n_rows, nnz) = shard_sizes[i];
            let row_off = row_offsets[i];
            let nnz_off = nnz_offsets[i];

            // Release-mode bounds guards — catch catalog corruption / stat
            // drift before dereferencing raw pointers below.  `assert!` (not
            // `debug_assert!`) because these invariants are the ONLY thing
            // keeping the unsafe block below from writing past the allocated
            // region; stripping them in release would silently corrupt the
            // heap on malformed inputs.
            assert!(
                nnz_off + nnz <= total_nnz,
                "shard {i}: nnz range {nnz_off}..{} exceeds total_nnz {total_nnz}",
                nnz_off + nnz
            );
            assert!(
                row_off + n_rows <= total_rows,
                "shard {i}: row range {row_off}..{} exceeds total_rows {total_rows}",
                row_off + n_rows
            );

            let (shard_ip, shard_ix, shard_data) = self.read_shard_from_entry(entry)?;
            debug_assert_eq!(
                shard_ip.len(),
                n_rows + 1,
                "shard {i} indptr length mismatch: catalog says {}, got {}",
                n_rows + 1,
                shard_ip.len()
            );
            debug_assert_eq!(
                shard_ix.len(),
                nnz,
                "shard {i} indices length mismatch: catalog says {nnz}, got {}",
                shard_ix.len()
            );
            debug_assert_eq!(
                shard_data.len(),
                nnz,
                "shard {i} data length mismatch: catalog says {nnz}, got {}",
                shard_data.len()
            );

            // SAFETY: each shard writes to [nnz_off..nnz_off+nnz], non-overlapping.
            // The non-overlap invariant is enforced by the monotonic `nnz_offsets`
            // prefix scan at L725–733 combined with the `nnz_off + nnz <= total_nnz`
            // guard above.
            let ix_out = unsafe {
                std::slice::from_raw_parts_mut((indices_base as *mut i32).add(nnz_off), nnz)
            };
            let d_out = unsafe {
                std::slice::from_raw_parts_mut((data_base as *mut f32).add(nnz_off), nnz)
            };
            ix_out.copy_from_slice(&shard_ix);
            d_out.copy_from_slice(&shard_data);

            // Indptr: shard 0 copies all n_rows+1 values as-is;
            // shard i>0 copies [1..] with cumulative nnz offset.
            if i == 0 {
                // SAFETY: shard 0 writes to [0..n_rows+1], non-overlapping with i>0.
                // Bounded by `row_off + n_rows <= total_rows` guard above.
                let ip_out =
                    unsafe { std::slice::from_raw_parts_mut(indptr_base as *mut i64, n_rows + 1) };
                ip_out.copy_from_slice(&shard_ip);
            } else {
                // SAFETY: shard i writes to [row_off+1..row_off+1+n_rows], non-overlapping.
                // Bounded by `row_off + n_rows <= total_rows` guard above.
                let ip_out = unsafe {
                    std::slice::from_raw_parts_mut(
                        (indptr_base as *mut i64).add(row_off + 1),
                        n_rows,
                    )
                };
                let nnz_off_i64 = nnz_off as i64;
                for j in 0..n_rows {
                    ip_out[j] = shard_ip[j + 1] + nnz_off_i64;
                }
            }

            Ok::<_, ScxError>(())
        })?;

        let n_rows = indptr.len().saturating_sub(1);
        Ok(ScxCsr::new_unchecked(
            (n_rows, self.header.n_vars as usize),
            indptr,
            indices,
            data,
        ))
    }

    /// Assemble multiple shard entries into a single ScxCsr (sequential).
    ///
    /// Pre-allocates the final merged arrays to exact sizes using catalog stats,
    /// then decodes each shard sequentially into its target region.
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

        // Hint aggressive readahead across the shard region.
        // Use min/max of file offsets since shards are sorted by row_start,
        // not file offset — they may not be contiguous after append/compact.
        #[cfg(unix)]
        {
            let min_offset = shards.iter().map(|e| e.offset as usize).min().unwrap();
            let max_end = shards
                .iter()
                .map(|e| (e.offset + e.length) as usize)
                .max()
                .unwrap();
            self.advise_sequential(min_offset, max_end - min_offset);
        }

        // Pre-compute per-shard (n_rows, nnz) from catalog stats
        let shard_sizes: Vec<(usize, usize)> = shards
            .iter()
            .map(|e| {
                let stats = e.stats.as_ref().ok_or_else(|| {
                    ScxError::InvalidCatalog(format!(
                        "shard entry '{}' at offset {} has no stats block",
                        e.name, e.offset
                    ))
                })?;
                Ok::<_, ScxError>((
                    (stats.row_end - stats.row_start) as usize,
                    stats.nnz as usize,
                ))
            })
            .collect::<Result<_>>()?;
        let total_rows: usize = shard_sizes.iter().map(|(r, _)| *r).sum();
        let total_nnz: usize = shard_sizes.iter().map(|(_, n)| *n).sum();

        // Single allocation for final merged arrays
        let mut indptr = vec![0i64; total_rows + 1];
        let mut indices = vec![0i32; total_nnz];
        let mut data = vec![0f32; total_nnz];

        let mut cum_rows = 0usize;
        let mut cum_nnz = 0usize;

        for (i, entry) in shards.iter().enumerate() {
            let (n_rows, nnz) = shard_sizes[i];
            let (shard_ip, shard_ix, shard_data) = self.read_shard_from_entry(entry)?;
            debug_assert_eq!(shard_ip.len(), n_rows + 1);
            debug_assert_eq!(shard_ix.len(), nnz);
            debug_assert_eq!(shard_data.len(), nnz);

            // Copy indices and data into their target region
            indices[cum_nnz..cum_nnz + nnz].copy_from_slice(&shard_ix);
            data[cum_nnz..cum_nnz + nnz].copy_from_slice(&shard_data);

            // Copy indptr with cumulative nnz offset
            if i == 0 {
                indptr[0..n_rows + 1].copy_from_slice(&shard_ip);
            } else {
                let nnz_off_i64 = cum_nnz as i64;
                for j in 0..n_rows {
                    indptr[cum_rows + 1 + j] = shard_ip[j + 1] + nnz_off_i64;
                }
            }

            cum_rows += n_rows;
            cum_nnz += nnz;
        }

        let n_rows = indptr.len().saturating_sub(1);
        Ok(ScxCsr::new_unchecked(
            (n_rows, self.header.n_vars as usize),
            indptr,
            indices,
            data,
        ))
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
#[path = "reader_tests.rs"]
mod tests;
