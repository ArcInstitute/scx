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
use scx_sparse::{ScxCsc, ScxCsr};

// `CodecId` / `ValueEncoding` are only referenced from the in-module
// `#[cfg(test)]` block + the test-only `values_to_f32` helper; gating
// the imports keeps the release build warning-clean.
#[cfg(test)]
use scx_codec::{CodecId, ValueEncoding};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::catalog::{FullCatalog, FullCatalogEntry};
use crate::checksum::blake3_hash;
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
}

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
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };

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
        let slice = self.section_bytes(entry)?;
        let cursor = Cursor::new(slice);
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

        // Wide types present — drive the slow path through the same
        // logic the data path uses, so schema reflects whether offsets
        // actually overflow per column.
        let cursor = Cursor::new(self.section_bytes(entry)?);
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
        let dv_opt = self.read_deletion_vectors()?;

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
                if let Some(bitmap) = dv.shards.get(&(shard_idx as u32)) {
                    for local_row in bitmap.iter() {
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

    fn read_shard_from_entry_inner(
        &self,
        entry: &FullCatalogEntry,
        verify_checksum: bool,
    ) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
        let section = self.section_bytes(entry)?;
        crate::shard_decode::decode_shard_bytes(
            section,
            entry,
            self.full_catalog.catalog_version,
            verify_checksum,
        )
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
mod tests {
    use super::*;
    use crate::header::{CURRENT_FORMAT_VERSION, MAGIC};
    use crate::provenance::ProvenanceEntry;
    use crate::shard::SHARD_HEADER_SIZE;
    use crate::writer::ScxWriter;
    use arrow::array::{Float32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
        FileHeader {
            magic: MAGIC,
            format_version: CURRENT_FORMAT_VERSION,
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
            n_modalities: 0,
            modality_table_offset: 0,
            modality_table_length: 0,
            reserved: [0u8; 112],
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
        assert_eq!(reader.header().format_version, CURRENT_FORMAT_VERSION);

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

    /// Sharded obsm round-trip: three row-shards of the same logical
    /// matrix should reassemble byte-equal to the unsharded equivalent.
    #[test]
    fn test_sharded_obsm_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sharded_obsm.scx");

        let n_obs: usize = 9;
        let n_vars: usize = 4;
        let header = FileHeader {
            magic: MAGIC,
            format_version: CURRENT_FORMAT_VERSION,
            header_length: 256,
            flags: 0,
            n_obs: n_obs as u64,
            n_vars: n_vars as u64,
            nnz: 0,
            n_csr_shards: 0,
            n_csc_shards: 0,
            shard_target_rows: 3,
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
        };
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();

        // Single CSR shard so the file passes its catalog invariants.
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

        // 3 row-shards of (3 × 2) obsm, values pc1 = row index, pc2 = 2*row.
        let obsm_schema = Arc::new(Schema::new(vec![
            Field::new("pc1", DataType::Float32, false),
            Field::new("pc2", DataType::Float32, false),
        ]));
        for shard_idx in 0u32..3 {
            let row_start = shard_idx as usize * 3;
            let rows: Vec<f32> = (row_start..row_start + 3).map(|r| r as f32).collect();
            let rows2: Vec<f32> = rows.iter().map(|r| r * 2.0).collect();
            let batch = RecordBatch::try_new(
                obsm_schema.clone(),
                vec![
                    Arc::new(Float32Array::from(rows)),
                    Arc::new(Float32Array::from(rows2)),
                ],
            )
            .unwrap();
            writer
                .write_obsm_shard(
                    "X_pca",
                    shard_idx,
                    row_start as u64,
                    batch.num_rows() as u64,
                    n_obs as u64,
                    &batch,
                )
                .unwrap();
        }

        writer.finish().unwrap();

        let reader = ScxReader::open(&path).unwrap();
        let pca = reader.read_obsm("X_pca").unwrap();
        assert_eq!(pca.num_rows(), n_obs);
        assert_eq!(pca.num_columns(), 2);
        let pc1 = pca
            .column(0)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        for (i, v) in pc1.values().iter().enumerate() {
            assert_eq!(*v, i as f32);
        }
        let all = reader.read_all_obsm().unwrap();
        assert_eq!(all.len(), 1);
        assert!(all.contains_key("X_pca"));
        let names = reader.list_obsm();
        assert_eq!(names, vec!["X_pca".to_string()]);
    }

    /// A file written with the legacy single-section obsm path must
    /// keep reading correctly after the sharded-aware reader changes.
    /// This is the backward-compatibility guarantee for files written
    /// before sharding was introduced.
    #[test]
    fn test_legacy_single_section_obsm_still_reads() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "legacy_obsm.scx", 6, 10, 2, true);
        let reader = ScxReader::open(&path).unwrap();

        let obsm = reader.read_obsm("X_pca").unwrap();
        assert_eq!(obsm.num_rows(), 6);
        assert_eq!(obsm.num_columns(), 2);

        let all = reader.read_all_obsm().unwrap();
        assert_eq!(all.len(), 1);
        assert!(all.contains_key("X_pca"));
    }

    /// Helper for the sharded-reader regression tests: writes a minimal
    /// SCX file with one CSR shard and `obsm/X_pca` split into
    /// `n_obs / shard_rows` dense shards, then hands the caller the
    /// `ScxWriter` mid-flight so it can override the obsm shard layout
    /// (skip a shard, duplicate a `shard_idx`, etc.) before `finish()`.
    fn build_obsm_test_writer(
        path: &std::path::Path,
        n_obs: usize,
        n_vars: usize,
        shard_rows: u32,
    ) -> ScxWriter {
        let header = FileHeader {
            magic: MAGIC,
            format_version: CURRENT_FORMAT_VERSION,
            header_length: 256,
            flags: 0,
            n_obs: n_obs as u64,
            n_vars: n_vars as u64,
            nnz: 0,
            n_csr_shards: 0,
            n_csc_shards: 0,
            shard_target_rows: shard_rows,
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
        };
        let mut writer = ScxWriter::new(path, header).unwrap();
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
        writer
    }

    fn dense_obsm_shard_batch(row_start: usize, n_rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("pc1", DataType::Float32, false),
            Field::new("pc2", DataType::Float32, false),
        ]));
        let rows: Vec<f32> = (row_start..row_start + n_rows).map(|r| r as f32).collect();
        let rows2: Vec<f32> = rows.iter().map(|r| r * 2.0).collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float32Array::from(rows)),
                Arc::new(Float32Array::from(rows2)),
            ],
        )
        .unwrap()
    }

    /// A sharded `obsm/X_pca` whose middle shard is missing must fail
    /// reads with `InvalidCatalog`, not silently return a truncated
    /// matrix. Regression guard for the contiguity-check fix.
    #[test]
    fn test_sharded_read_rejects_missing_shard() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing_shard.scx");
        let n_obs = 9usize;
        let mut writer = build_obsm_test_writer(&path, n_obs, 4, 3);

        // Write shard 0 and shard 2 only — shard 1 is missing.
        for &shard_idx in &[0u32, 2u32] {
            let row_start = shard_idx as usize * 3;
            let batch = dense_obsm_shard_batch(row_start, 3);
            writer
                .write_obsm_shard(
                    "X_pca",
                    shard_idx,
                    row_start as u64,
                    batch.num_rows() as u64,
                    n_obs as u64,
                    &batch,
                )
                .unwrap();
        }
        writer.finish().unwrap();

        let reader = ScxReader::open(&path).unwrap();
        let err = reader.read_obsm("X_pca").unwrap_err();
        assert!(
            matches!(err, ScxError::InvalidCatalog(_)),
            "expected InvalidCatalog, got {err:?}"
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("X_pca"),
            "error should name the logical section: {msg}"
        );
    }

    /// Two shards with the same `shard_idx` must be rejected as
    /// `InvalidCatalog` — the second shard's stamped `shard_idx`
    /// won't match its position after sorting.
    #[test]
    fn test_sharded_read_rejects_duplicate_shard_idx() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dup_shard.scx");
        let n_obs = 6usize;
        let mut writer = build_obsm_test_writer(&path, n_obs, 4, 3);

        // Two physical shards both stamped with shard_idx = 0. The
        // second one's catalog name is `..._shard_1` (so the catalog
        // walk picks both up), but its schema-stamped `shard_idx` is
        // still 0 — the reader should reject the position/stamp
        // mismatch.
        let batch0 = dense_obsm_shard_batch(0, 3);
        writer
            .write_obsm_shard(
                "X_pca",
                0,
                0,
                batch0.num_rows() as u64,
                n_obs as u64,
                &batch0,
            )
            .unwrap();
        let batch1 = dense_obsm_shard_batch(3, 3);
        // Stamp shard_idx = 0 on the second shard by re-using the
        // first shard's logical position in the metadata; section name
        // still carries `_shard_1` so it lands in the catalog.
        writer
            .write_obsm_shard(
                "X_pca",
                1, // section-name index — chosen so the catalog has _shard_1
                0, // row_start = 0 deliberately duplicates the first shard
                batch1.num_rows() as u64,
                n_obs as u64,
                &batch1,
            )
            .unwrap();
        writer.finish().unwrap();

        let reader = ScxReader::open(&path).unwrap();
        let err = reader.read_obsm("X_pca").unwrap_err();
        assert!(
            matches!(err, ScxError::InvalidCatalog(_)),
            "expected InvalidCatalog, got {err:?}"
        );
    }

    /// A zero-row `obsm` shard must round-trip — `n_rows == 0` is the
    /// edge case that disappeared from the disk-streaming path before
    /// this fix landed. The writer-side override path and the
    /// scx-convert disk-streaming branch both emit a single zero-row
    /// shard; this test asserts the reader reassembles it as a
    /// zero-row batch (not a `SectionNotFound`).
    #[test]
    fn test_sharded_read_zero_row_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("zero_row.scx");
        let mut writer = build_obsm_test_writer(&path, 6, 4, 3);
        let empty = dense_obsm_shard_batch(0, 0);
        writer
            .write_obsm_shard("X_empty", 0, 0, 0, 0, &empty)
            .unwrap();
        writer.finish().unwrap();

        let reader = ScxReader::open(&path).unwrap();
        let names = reader.list_obsm();
        assert_eq!(names, vec!["X_empty".to_string()]);
        let batch = reader.read_obsm("X_empty").unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 2);
    }

    /// The merged `RecordBatch` returned by `read_obsm` must NOT carry
    /// per-shard metadata (`shard_idx`, `row_start`, `n_shard_rows`) —
    /// those describe a single shard, not the reassembled matrix.
    /// Stripping them prevents downstream consumers from being misled.
    #[test]
    fn test_sharded_read_strips_shard_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("strip_meta.scx");
        let n_obs = 6usize;
        let mut writer = build_obsm_test_writer(&path, n_obs, 4, 3);
        for shard_idx in 0u32..2 {
            let row_start = shard_idx as usize * 3;
            let batch = dense_obsm_shard_batch(row_start, 3);
            writer
                .write_obsm_shard(
                    "X_pca",
                    shard_idx,
                    row_start as u64,
                    batch.num_rows() as u64,
                    n_obs as u64,
                    &batch,
                )
                .unwrap();
        }
        writer.finish().unwrap();

        let reader = ScxReader::open(&path).unwrap();
        let pca = reader.read_obsm("X_pca").unwrap();
        let md = pca.schema_ref().metadata();
        assert!(
            !md.contains_key("shard_idx"),
            "merged batch should not carry shard_idx (got metadata: {md:?})"
        );
        assert!(
            !md.contains_key("row_start"),
            "merged batch should not carry row_start (got metadata: {md:?})"
        );
        assert!(
            !md.contains_key("n_shard_rows"),
            "merged batch should not carry n_shard_rows (got metadata: {md:?})"
        );
        // n_rows_total describes the logical matrix and is preserved.
        assert_eq!(
            md.get("n_rows_total").map(String::as_str),
            Some("6"),
            "n_rows_total should survive (got metadata: {md:?})"
        );
    }

    /// Regression: when every shard carries the *same* categorical value,
    /// Arrow's `concat` appends each shard's one-element dictionary, yielding
    /// `["batch1", "batch1", "batch1", "batch1"]`. `to_pandas()` then raises
    /// `ValueError: Categorical categories must be unique`. `assemble_sharded_metadata`
    /// must collapse dictionary columns to a unified dictionary with distinct
    /// values.
    #[test]
    fn test_assemble_unifies_duplicate_dictionary_categories() {
        use arrow::array::{Array, DictionaryArray};
        use arrow::datatypes::Int8Type;

        let dict_dt = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        let n_shards = 4u32;

        let raw_batches: Vec<(u32, RecordBatch)> = (0..n_shards)
            .map(|shard_idx| {
                // One row per shard, all the same category — the duplicate
                // dictionary the writer's per-shard categoricals produce.
                let strs = StringArray::from(vec!["batch1"]);
                let dict =
                    arrow::compute::cast(&(Arc::new(strs) as arrow::array::ArrayRef), &dict_dt)
                        .unwrap();
                let metadata = std::collections::HashMap::from([
                    ("shard_idx".to_string(), shard_idx.to_string()),
                    ("row_start".to_string(), shard_idx.to_string()),
                    ("n_shard_rows".to_string(), "1".to_string()),
                    ("n_rows_total".to_string(), n_shards.to_string()),
                ]);
                let schema = Arc::new(
                    Schema::new(vec![Field::new("gem_group", dict_dt.clone(), false)])
                        .with_metadata(metadata),
                );
                let batch = RecordBatch::try_new(schema, vec![dict]).unwrap();
                (shard_idx, batch)
            })
            .collect();

        let merged = assemble_sharded_metadata("obs", raw_batches).unwrap();
        assert_eq!(merged.num_rows(), n_shards as usize);

        let col = merged.column(0);
        // After unification the single surviving category fits an Int8 key —
        // `unify_dictionary_columns` narrows to the minimal key type.
        assert_eq!(
            col.data_type(),
            &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            "1 distinct category should narrow to an Int8 key"
        );
        let dict = col
            .as_any()
            .downcast_ref::<DictionaryArray<Int8Type>>()
            .expect("gem_group should remain dictionary-encoded");
        let values = dict
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("dictionary values should be Utf8");
        assert_eq!(
            values.len(),
            1,
            "duplicate categories must be collapsed (got {:?})",
            (0..values.len())
                .map(|i| values.value(i))
                .collect::<Vec<_>>()
        );
        assert_eq!(values.value(0), "batch1");
        // Every row still resolves to the single surviving category.
        for i in 0..dict.len() {
            assert_eq!(values.value(dict.keys().value(i) as usize), "batch1");
        }
    }

    /// Regression: a categorical column whose per-shard vocabularies are
    /// disjoint and sum to more than a narrow per-shard key can address.
    /// Each shard here holds 50 unique categories encoded with an `Int8` key
    /// (50 ≤ 127, the per-shard write is valid), but the union across 3 shards
    /// is 150 distinct. Before the fix, `concat_batches` / the unify re-encode
    /// overflowed the `Int8` key with `Dictionary key bigger than the key
    /// type`; now the keys are widened to `Int32` before concat and narrowed
    /// to the minimal fit (`Int16` for 150 distinct) after deduplication.
    #[test]
    fn test_assemble_high_cardinality_dictionary_widens_key() {
        use arrow::array::{Array, DictionaryArray};
        use arrow::datatypes::Int16Type;

        let per_shard = 50usize;
        let n_shards = 3u32;
        let total = per_shard * n_shards as usize;
        let narrow_dt = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));

        let raw_batches: Vec<(u32, RecordBatch)> = (0..n_shards)
            .map(|shard_idx| {
                let cats: Vec<String> = (0..per_shard)
                    .map(|j| format!("s{shard_idx}_c{j}"))
                    .collect();
                let strs = StringArray::from(cats.iter().map(|s| s.as_str()).collect::<Vec<_>>());
                // Per-shard categorical with a narrow Int8 key — valid locally
                // (50 ≤ 127) but disjoint across shards.
                let dict =
                    arrow::compute::cast(&(Arc::new(strs) as arrow::array::ArrayRef), &narrow_dt)
                        .unwrap();
                let row_start = shard_idx as usize * per_shard;
                let metadata = std::collections::HashMap::from([
                    ("shard_idx".to_string(), shard_idx.to_string()),
                    ("row_start".to_string(), row_start.to_string()),
                    ("n_shard_rows".to_string(), per_shard.to_string()),
                    ("n_rows_total".to_string(), total.to_string()),
                ]);
                let schema = Arc::new(
                    Schema::new(vec![Field::new("cell_type", narrow_dt.clone(), false)])
                        .with_metadata(metadata),
                );
                let batch = RecordBatch::try_new(schema, vec![dict]).unwrap();
                (shard_idx, batch)
            })
            .collect();

        // Pre-fix this returned `Err(Arrow("Dictionary key bigger than the key type"))`.
        let merged = assemble_sharded_metadata("obs", raw_batches).unwrap();
        assert_eq!(merged.num_rows(), total);

        let col = merged.column(0);
        assert_eq!(
            col.data_type(),
            &DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8)),
            "150 distinct categories should narrow to an Int16 key"
        );
        let dict = col
            .as_any()
            .downcast_ref::<DictionaryArray<Int16Type>>()
            .expect("cell_type should remain dictionary-encoded");
        let values = dict
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("dictionary values should be Utf8");
        assert_eq!(values.len(), total, "all 150 categories must survive");
        // Every row resolves to its original "s{shard}_c{j}" category.
        for shard in 0..n_shards as usize {
            for j in 0..per_shard {
                let row = shard * per_shard + j;
                let got = values.value(dict.keys().value(row) as usize);
                assert_eq!(got, format!("s{shard}_c{j}"));
            }
        }
    }

    #[test]
    fn test_read_obs_schema_matches_full() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "schema.scx", 6, 10, 2, false);
        let reader = ScxReader::open(&path).unwrap();

        let obs_schema = reader.read_obs_schema().unwrap();
        let obs_batch = reader.read_obs().unwrap();
        assert_eq!(&obs_schema, obs_batch.schema().as_ref());

        let var_schema = reader.read_var_schema().unwrap();
        let var_batch = reader.read_var().unwrap();
        assert_eq!(&var_schema, var_batch.schema().as_ref());
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
        assert!(matches!(
            result.unwrap_err(),
            ScxError::ChecksumMismatch { .. }
        ));
    }

    /// Regression guard for Phase 2A: verified path catches corruption,
    /// unchecked path (default) does not error on corrupted shard payload.
    #[test]
    fn test_verified_vs_unchecked_shard_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "verify_guard.scx", 6, 10, 2, false);

        // Read the file, corrupt a byte in a shard payload, write back
        let mut data = std::fs::read(&path).unwrap();
        let reader = ScxReader::open(&path).unwrap();
        let shards = reader.catalog().shards_sorted();
        assert!(!shards.is_empty());
        let shard_entry = shards[0].clone();
        let shard_offset = shard_entry.offset as usize;
        let corrupt_pos = shard_offset + SHARD_HEADER_SIZE + 1;
        drop(reader);

        data[corrupt_pos] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();

        // Re-open (catalog checksum is still intact since we only corrupted
        // shard payload bytes, not the catalog region)
        let reader = ScxReader::open(&path).unwrap();

        // Verified path should detect the corruption
        let verified_result = reader.read_shard_from_entry_verified(&shard_entry);
        assert!(verified_result.is_err());
        assert!(matches!(
            verified_result.unwrap_err(),
            ScxError::ChecksumMismatch { .. }
        ));

        // Unchecked path (default) should not error — it skips the checksum
        let unchecked_result = reader.read_shard_from_entry(&shard_entry);
        assert!(unchecked_result.is_ok());
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
            assert!(SectionType::from_u8(entry.section_type as u8).is_some());
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
    // 2B.7: Single-allocation assembly matches individual shard merge
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_alloc_assembly_1_shard() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "one_shard.scx", 8, 10, 1, false);
        let reader = ScxReader::open(&path).unwrap();

        let csr = reader.read_all_csr_shards().unwrap();
        // 1 shard: indptr should start at 0 and be monotonic
        assert_eq!(csr.indptr[0], 0);
        assert_eq!(csr.shape, (8, 10));
        for w in csr.indptr.windows(2) {
            assert!(w[1] >= w[0], "indptr not monotonic: {} > {}", w[0], w[1]);
        }
        for &idx in &csr.indices {
            assert!((0..10).contains(&idx), "index {} out of bounds", idx);
        }
    }

    #[test]
    fn test_single_alloc_assembly_2_shards() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "two_shards.scx", 8, 10, 2, false);
        let reader = ScxReader::open(&path).unwrap();

        // Read individually (old merge pattern)
        let mut individual_indptr: Vec<i64> = Vec::new();
        let mut individual_indices: Vec<i32> = Vec::new();
        let mut individual_data: Vec<f32> = Vec::new();
        let mut cumulative_nnz: i64 = 0;

        for i in 0..2 {
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

        // Read assembled (single-allocation path)
        let csr = reader.read_all_csr_shards().unwrap();

        assert_eq!(csr.indptr, individual_indptr);
        assert_eq!(csr.indices, individual_indices);
        assert_eq!(csr.data, individual_data);
        assert_eq!(csr.shape, (8, 10));
    }

    #[test]
    fn test_single_alloc_assembly_many_shards() {
        let dir = tempfile::tempdir().unwrap();
        // 100 rows, 20 vars, 10 shards = 10 rows per shard
        let path = write_test_file(&dir, "many_shards.scx", 100, 20, 10, false);
        let reader = ScxReader::open(&path).unwrap();

        // Read individually (old merge pattern)
        let mut individual_indptr: Vec<i64> = Vec::new();
        let mut individual_indices: Vec<i32> = Vec::new();
        let mut individual_data: Vec<f32> = Vec::new();
        let mut cumulative_nnz: i64 = 0;

        for i in 0..10 {
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

        // Read assembled (single-allocation path)
        let csr = reader.read_all_csr_shards().unwrap();

        assert_eq!(csr.indptr, individual_indptr);
        assert_eq!(csr.indices, individual_indices);
        assert_eq!(csr.data, individual_data);
        assert_eq!(csr.shape, (100, 20));

        // Verify invariants
        for w in csr.indptr.windows(2) {
            assert!(w[1] >= w[0], "indptr not monotonic");
        }
        for &idx in &csr.indices {
            assert!((0..20).contains(&idx), "index {} out of bounds", idx);
        }
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
        let obs_bytes = reader.section_bytes(obs_entry).unwrap();
        assert_eq!(obs_bytes.len(), obs_entry.length as usize);

        let var_entry = catalog.get("var").unwrap();
        let var_bytes = reader.section_bytes(var_entry).unwrap();
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

    /// Phase J.2: validate() re-checks BLAKE3 for CSC shards via the
    /// generic catalog walk. Build a CSR + CSC test file, verify all
    /// CSC sections appear in the results and pass.
    #[test]
    fn test_validate_csc_shards_pass_on_clean_file() {
        use crate::section::SectionType;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("csc_clean.scx");

        // Build a small CSR + CSC file directly.
        let n_obs = 8usize;
        let n_vars = 6usize;
        let header = sample_header(n_obs as u64, n_vars as u64, 0);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();

        // CSR (single shard)
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

        // CSC: two shards of 3 cols each.
        for chunk_start in (0..n_vars).step_by(3) {
            let chunk_end = (chunk_start + 3).min(n_vars);
            let mut ip = vec![0u64];
            let mut ix: Vec<u32> = Vec::new();
            let mut vb: Vec<u8> = Vec::new();
            for c in chunk_start..chunk_end {
                ix.push((c % n_obs) as u32);
                vb.push((c as u8) + 1);
                ip.push(ix.len() as u64);
            }
            writer
                .write_csc_shard(
                    &ip,
                    &ix,
                    &vb,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    chunk_start as u64,
                )
                .unwrap();
        }
        writer.finish().unwrap();

        let reader = ScxReader::open(&path).unwrap();
        let results = reader.validate().unwrap();

        // Confirm the CSC shards are in the results AND all pass.
        let csc_results: Vec<_> = results
            .iter()
            .filter(|(name, _)| name.starts_with("X_csc_shard_"))
            .collect();
        assert_eq!(csc_results.len(), 2, "expected 2 CSC shards in validate()");
        for (name, passed) in &csc_results {
            assert!(*passed, "CSC section '{}' failed checksum", name);
        }

        // Also confirm `validate()` walks every catalog entry: total
        // results count >= number of catalog entries with CscShard
        // type, so no CSC entry was silently skipped.
        let n_csc_entries = reader
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CscShard)
            .count();
        let n_csc_in_results = results
            .iter()
            .filter(|(name, _)| name.starts_with("X_csc_shard_"))
            .count();
        assert_eq!(n_csc_entries, n_csc_in_results);
    }

    /// Phase J.2: validate() flags corruption inside a CSC shard.
    /// CSC is not in the "essential" set (corrupting only CSC
    /// shouldn't fail the full file), so we expect Ok with a
    /// `passed=false` row for the corrupted CSC section.
    #[test]
    fn test_validate_detects_csc_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("csc_corrupt.scx");

        let n_obs = 6usize;
        let n_vars = 4usize;
        let header = sample_header(n_obs as u64, n_vars as u64, 0);
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
        // One CSC shard.
        let ip = vec![0u64, 1, 2, 3, 4];
        let ix: Vec<u32> = vec![0, 1, 2, 3];
        let vb: Vec<u8> = vec![10, 20, 30, 40];
        writer
            .write_csc_shard(&ip, &ix, &vb, CodecId::None, ValueEncoding::Uint8, 0)
            .unwrap();
        writer.finish().unwrap();

        // Find the CSC shard offset in the catalog.
        let reader = ScxReader::open(&path).unwrap();
        let csc_entry = reader
            .catalog()
            .entries
            .iter()
            .find(|e| e.section_type == crate::section::SectionType::CscShard)
            .unwrap()
            .clone();
        let csc_offset = csc_entry.offset as usize;
        drop(reader);

        // Flip a byte in the CSC payload (after the 76-byte header).
        let mut data = std::fs::read(&path).unwrap();
        let corrupt_pos = csc_offset + SHARD_HEADER_SIZE + 1;
        data[corrupt_pos] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();

        // validate() should NOT return Err (CSC is not "essential"),
        // but should report the CSC section as failed.
        let reader = ScxReader::open(&path).unwrap();
        let results = reader.validate().expect(
            "validate() should not error on CSC corruption since CSC is not \
             in the essential-section set",
        );
        let csc_passed: bool = results
            .iter()
            .find(|(name, _)| name == &csc_entry.name)
            .map(|(_, p)| *p)
            .expect("CSC entry should appear in validate() results");
        assert!(!csc_passed, "validate() should flag corrupted CSC section");
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
        let val: f32 = 1.23456;
        let raw = val.to_le_bytes().to_vec();
        let result = values_to_f32(&raw, ValueEncoding::Float32);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.23456).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Shared-catalog open (open_with_shared_catalog)
    // -----------------------------------------------------------------------

    /// `open_with_shared_catalog` must produce a reader whose
    /// metadata (header, root catalog, full catalog, shard reads) is
    /// indistinguishable from a fresh `open()` against the same file.
    /// This is the load-bearing correctness check for the
    /// `to_anndata_backed` catalog-sharing path.
    #[test]
    fn test_open_with_shared_catalog_matches_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "shared_catalog.scx", 8, 12, 2, false);

        let primary = ScxReader::open(&path).unwrap();
        let primary_catalog = primary.catalog_arc();

        let shared = ScxReader::open_with_shared_catalog(&path, primary_catalog).unwrap();

        // Header / root catalog must be identical (parsed fresh from
        // the secondary mmap, but the file is the same).
        assert_eq!(shared.header().n_obs, primary.header().n_obs);
        assert_eq!(shared.header().n_vars, primary.header().n_vars);
        assert_eq!(
            shared.header().full_catalog_offset,
            primary.header().full_catalog_offset
        );

        // Catalog entries must match field-for-field — the shared
        // path didn't re-parse, so this verifies the Arc shared
        // through.
        let primary_entries = &primary.catalog().entries;
        let shared_entries = &shared.catalog().entries;
        assert_eq!(primary_entries.len(), shared_entries.len());
        for (a, b) in primary_entries.iter().zip(shared_entries.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.offset, b.offset);
            assert_eq!(a.length, b.length);
            assert_eq!(a.section_type, b.section_type);
            assert_eq!(a.modality_id, b.modality_id);
            assert_eq!(a.checksum, b.checksum);
        }

        // Shard reads against the shared reader must produce the same
        // bytes as against the primary — confirms the mmap path is
        // independent and the cached catalog still drives correct
        // section addressing.
        let csr_shards = primary.catalog().shards_sorted();
        for entry in &csr_shards {
            let (ip_a, ix_a, dv_a) = primary.read_shard_from_entry(entry).unwrap();
            let (ip_b, ix_b, dv_b) = shared.read_shard_from_entry(entry).unwrap();
            assert_eq!(ip_a, ip_b);
            assert_eq!(ix_a, ix_b);
            assert_eq!(dv_a, dv_b);
        }
    }

    /// Smoke test: many readers can share a single `Arc<FullCatalog>`
    /// without contention. Mirrors the `to_anndata_backed` shape (one
    /// primary reader + several secondary readers sharing its catalog).
    #[test]
    fn test_shared_catalog_n_plus_3_pattern() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "n_plus_3.scx", 6, 8, 2, false);

        let primary = ScxReader::open(&path).unwrap();
        let shared = primary.catalog_arc();

        // Strong refcount before the secondaries: 1 (held by primary).
        assert_eq!(Arc::strong_count(&shared), 2); // primary + this binding

        let secondaries: Vec<ScxReader> = (0..5)
            .map(|_| ScxReader::open_with_shared_catalog(&path, Arc::clone(&shared)).unwrap())
            .collect();

        // Each secondary holds a refcount; primary + binding + 5 = 7.
        assert_eq!(Arc::strong_count(&shared), 7);

        // All secondaries see the same catalog content.
        for s in &secondaries {
            assert_eq!(s.catalog().entries.len(), primary.catalog().entries.len());
        }

        // Dropping a secondary decrements the refcount.
        drop(secondaries);
        assert_eq!(Arc::strong_count(&shared), 2);
    }

    /// `open_with_shared_catalog` must reject a catalog whose
    /// `manifest_sequence` disagrees with the freshly-read file
    /// header. This is the guardrail against silently combining a
    /// stale catalog with a mutated file (append / compact / rollback
    /// in `scx-ops` bumps the sequence on every mutation).
    #[test]
    fn test_open_with_shared_catalog_rejects_manifest_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "manifest_mismatch.scx", 6, 8, 2, false);

        let primary = ScxReader::open(&path).unwrap();
        let mut tweaked = (*primary.catalog_arc()).clone();
        tweaked.manifest_sequence = primary.header().manifest_sequence.wrapping_add(1);

        match ScxReader::open_with_shared_catalog(&path, Arc::new(tweaked)) {
            Err(ScxError::InvalidCatalog(msg)) => {
                assert!(
                    msg.contains("manifest_sequence"),
                    "error must name the mismatched field, got: {msg}",
                );
            }
            Err(other) => panic!("expected InvalidCatalog, got {other:?}"),
            Ok(_) => panic!("manifest_sequence mismatch must surface as an error"),
        }
    }
}
