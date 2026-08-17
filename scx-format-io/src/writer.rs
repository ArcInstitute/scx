// ScxWriter — atomic rename path (docs/architecture.md)

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use byteorder::{LittleEndian, ReadBytesExt};
use scx_codec::{CodecId, ValueEncoding};

use crate::catalog::{FullCatalog, FullCatalogEntry, RootCatalog, RootCatalogEntry, ShardStats};
use crate::checksum::blake3_hash;
use crate::error::{Result, ScxError};
use crate::header::{FileHeader, HEADER_SIZE};
use crate::modality::{ModalityFlags, ModalityInfo, ModalityTable, ModalityType, MAX_MODALITIES};
use crate::section::{align_to_8, SectionType};
use crate::shard::{derive_shard_type, ShardHeader, SHARD_HEADER_SIZE};

use crate::provenance::{Provenance, ProvenanceEntry};

/// Offset where sections begin: 256 (header) + 4096 (root catalog placeholder).
pub const SECTIONS_START_OFFSET: u64 = 4352;

/// Atomic file writer for SCX files.
///
/// Writes sections sequentially to a temp file starting at offset 4352,
/// accumulates catalog entries, then finalizes: full catalog at EOF,
/// root catalog at 256, header at 0, fsync, atomic rename.
///
/// # Section-Ordering Convention
///
/// The writer accepts sections in **any order**, but callers should follow
/// the canonical layout described in docs/format.md (File Header) for maximum compatibility
/// with inspection tools and downstream readers:
///
/// 1. `write_obs` — cell metadata (Arrow IPC)
/// 2. `write_obs_predicate_index` — obs predicate indexes
/// 3. `write_var` — gene metadata (Arrow IPC)
/// 4. `write_var_predicate_index` — var predicate indexes
/// 5. `write_csr_shard` (repeated) — X matrix CSR shards
/// 6. `write_csc_shard` (repeated, optional) — X matrix CSC shards
/// 7. `write_layer_csr_shard` (repeated, optional) — layer shards
/// 8. `write_obsm` (repeated, optional) — embeddings
/// 9. `write_obsp_shard` (repeated, optional) — pairwise graphs
/// 10. `write_uns` (optional) — unstructured metadata (JSON)
/// 11. `write_provenance` (optional) — provenance chain
/// 12. `write_deletion_vectors` (optional) — logical deletions
///
/// This ordering is **not enforced** — `finish()` will produce a valid
/// file regardless of write order. However, deviating from it may produce
/// non-standard layouts that confuse inspection tools or yield sub-optimal
/// sequential read performance.
pub struct ScxWriter {
    final_path: PathBuf,
    tmp_path: Option<tempfile::TempPath>,
    file: Option<BufWriter<File>>,
    current_offset: u64,
    header: FileHeader,
    entries: Vec<FullCatalogEntry>,
    csr_shard_count: u32,
    csc_shard_count: u32,
    /// Count of `raw/X` (`RawCsrShard`) shards written so far — drives
    /// the `raw/X_shard_<idx>` naming. Kept separate from
    /// `csr_shard_count` so the main matrix's `n_csr_shards` header count
    /// is unaffected by the raw matrix.
    raw_csr_shard_count: u32,
    /// Column count (`raw.n_vars`) of the raw matrix. Set via
    /// [`Self::set_raw_n_vars`] before writing `RawCsrShard` shards so
    /// `write_shard_inner` stamps the correct minor-axis extent and picks
    /// the right per-shard `index_dtype` (raw has its OWN var axis,
    /// independent of `header.n_vars`).
    raw_n_vars: u64,
    /// Phase 5b: count of bitmap sidecar shards written so far. Used by
    /// `finish()` to flip `FileHeader::set_bitmap()` and (in the
    /// unimodal case) by `write_bitmap_shard` to derive the next
    /// shard's index.
    #[cfg(feature = "deletion-vectors")]
    bitmap_shard_count: u32,
    /// Phase 5b: per-modality bitmap shard counter. Indexed by
    /// `modality_id - 1`. Writer-only state (mirrors
    /// [`Self::modality_build_csc`]) — the actual count is reconstructed
    /// at read time from `FullCatalog::bitmap_shards_for_modality`.
    #[cfg(feature = "deletion-vectors")]
    modality_bitmap_counts: Vec<u32>,
    total_nnz: u64,
    has_obsm: bool,
    has_obsp: bool,
    /// Per-axis layout state for obs / var metadata. Mutually exclusive
    /// — calling [`Self::write_obs`] after [`Self::write_obs_shard`] (or
    /// vice versa) on the same writer returns
    /// `ScxError::ObsLayoutConflict`. Tracked here so the guard runs in
    /// O(1) without scanning [`Self::entries`] on every write.
    obs_layout: ObsVarLayout,
    var_layout: ObsVarLayout,
    /// The `modality_id` stamped on every catalog entry created by
    /// the next write call. Defaults to `0` (global / single-modality
    /// shape). Public `*_for` methods set this for the duration of
    /// the call and reset it to `0` afterwards via the
    /// `ModalityScope` RAII guard.
    current_modality_id: u8,
    /// Modalities registered via `add_modality()`. When empty, the
    /// file is single-modality and `finish()` emits no
    /// `ModalityTable` section.
    modalities: Vec<crate::modality::ModalityInfo>,
    /// Phase B.3: per-modality `build_csc` flags (writer-only — not
    /// persisted in the on-disk modality table). When `true`,
    /// `finish()` reads the modality's CSR shards back from the
    /// temp file, runs a streaming CSR→CSC transpose, and emits CSC
    /// sidecar shards via `write_csc_shard_for` before catalog
    /// assembly. The resulting CSC presence is recorded in the
    /// `ModalityFlags::HAS_CSC` bit on disk.
    modality_build_csc: Vec<bool>,
    /// CSR-data generation counter stamped into the output catalog
    /// (`FullCatalog::data_generation`). Defaults to `1` for a fresh
    /// write; CSR-mutating ops set it to `source + 1` via
    /// [`Self::with_data_generation`] so a later read can detect a CSC
    /// sidecar built against an earlier generation.
    data_generation: u64,
    /// The `data_generation` a CSC sidecar was built against, set by
    /// `finish()` when it auto-emits CSC (and left `None` otherwise so
    /// the catalog records `0`). Stamped into
    /// `FullCatalog::csc_build_generation`.
    csc_build_generation: Option<u64>,
    /// When set, every sparse shard written via `write_shard_inner`
    /// (CSC sidecars, layers, obsp — CSR X shards on this writer path too) is
    /// row-group-framed (shard v2). Set via [`Self::set_framing`]; `None` keeps
    /// the legacy unframed (v1) layout. The file `format_version` must be bumped
    /// to v4 separately by the caller when framing.
    framing: Option<crate::encoder::FramingConfig>,
}

/// Output of parallel shard encoding, ready for sequential write.
///
/// Contains all bytes and metadata needed to write a complete shard section
/// without any re-encoding or re-computation. Built in parallel (e.g., via
/// rayon) and consumed sequentially by [`ScxWriter::write_preencoded_shard`].
pub struct PreEncodedSection {
    /// The codec-compressed shard arrays (from `scx_codec::encode_shard`).
    pub encoded: scx_codec::EncodedShard,
    /// Serialized block index bytes.
    pub block_index_bytes: Vec<u8>,
    /// Serialized 76-byte shard header.
    pub header_buf: Vec<u8>,
    /// Full 32-byte BLAKE3 section checksum (header + all payload).
    pub section_checksum: [u8; 32],
    /// Total section length in bytes (header + indptr + indices + values + block_index).
    pub section_length: u64,
    /// Shard statistics computed from raw values.
    pub stats: ShardStats,
    /// Section name (e.g., "X_shard_0", "layer_name_shard_1").
    pub name: String,
    /// Section type (CsrShard, LayerCsrShard, etc.).
    pub section_type: SectionType,
    /// NNZ count for this shard.
    pub nnz: u64,
}

impl PreEncodedSection {
    /// The per-shard codec id stamped in the serialized `ShardHeader`
    /// (`header_buf`). Used by callers (e.g. `scx optimize`) to report which
    /// codec the encoder actually selected — notably the `compact-trial`
    /// per-shard winner. Returns `0` (`CodecId::None`) if the header cannot be
    /// parsed, which never happens for a section this crate just produced.
    pub fn codec_id(&self) -> u8 {
        ShardHeader::read_from(&mut &self.header_buf[..])
            .map(|h| h.codec_id)
            .unwrap_or(0)
    }

    /// The `shard_format_version` stamped in the serialized `ShardHeader`
    /// (`1` = unframed/legacy, `2` = row-group-framed). Lets callers count how
    /// many shards were emitted framed.
    pub fn shard_format_version(&self) -> u8 {
        ShardHeader::read_from(&mut &self.header_buf[..])
            .map(|h| h.shard_format_version)
            .unwrap_or(0)
    }
}

/// Per-axis layout state used by [`ScxWriter`] to enforce that obs (and
/// var) metadata is written either as a single Arrow IPC section
/// ([`SectionType::ObsMetadata`] / [`SectionType::VarMetadata`]) or as
/// a sequence of row-shard sections ([`SectionType::ObsMetadataShard`]
/// / [`SectionType::VarMetadataShard`]), but never both for the same
/// axis. Mixing would leave readers without a deterministic way to
/// reconstruct the logical batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObsVarLayout {
    /// No obs/var section of either flavour has been written yet for
    /// this axis. The next write picks the layout.
    Pending,
    /// A single-section batch has been written via `write_obs` /
    /// `write_var`. Subsequent `write_obs_shard` / `write_var_shard`
    /// calls return [`ScxError::ObsLayoutConflict`].
    Single,
    /// Some number of shards has been written via `write_obs_shard` /
    /// `write_var_shard`. The counter is consulted by writer-side
    /// asserts; it is not consulted by readers (which enumerate via
    /// the catalog).
    Sharded(u32),
}

/// Reconstruct an [`ObsVarLayout`] from a catalog-entry slice. Used by
/// [`ScxWriter::adopt_in_place`] so the mixed-mode guard survives an
/// adopt → write transition (e.g. append-time predicate-index emit
/// against an already-sharded file).
fn obs_var_layout_from_entries(
    entries: &[FullCatalogEntry],
    single: SectionType,
    sharded: SectionType,
) -> ObsVarLayout {
    let has_single = entries.iter().any(|e| e.section_type == single);
    let shard_count = entries.iter().filter(|e| e.section_type == sharded).count() as u32;
    match (has_single, shard_count) {
        (false, 0) => ObsVarLayout::Pending,
        (true, 0) => ObsVarLayout::Single,
        (false, n) => ObsVarLayout::Sharded(n),
        // Catalogs are not supposed to carry both. Surface as
        // `Sharded(n)` so the next write picks the safer shard path;
        // the mixed-mode condition is caught loudly at read time by
        // `read_obs()` (which prefers shards when both are present).
        (true, n) => ObsVarLayout::Sharded(n),
    }
}

/// Stamp a sharded `obsm` / `varm` / `obsp` / `varp` batch with the
/// per-shard metadata the reader uses to verify a contiguous, ordered
/// cover of the logical matrix and reassemble it. Stamped fields
/// (`shard_idx`, `row_start`, `n_shard_rows`, `n_rows_total`) ride on
/// the Arrow schema metadata so the reader can recover them without a
/// separate sidecar section.
///
/// For dense shards `n_shard_rows == batch.num_rows()` and the field
/// is redundant; for sparse shards `batch.num_rows()` counts COO
/// triples (= nnz in the shard's row range) rather than the row span,
/// so the stamped value is the only authoritative source.
fn stamp_dense_shard_metadata(
    batch: &RecordBatch,
    shard_idx: u32,
    row_start: u64,
    n_shard_rows: u64,
    n_rows_total: u64,
) -> RecordBatch {
    let mut metadata = batch.schema_ref().metadata().clone();
    metadata.insert("shard_idx".to_string(), shard_idx.to_string());
    metadata.insert("row_start".to_string(), row_start.to_string());
    metadata.insert("n_shard_rows".to_string(), n_shard_rows.to_string());
    metadata.insert("n_rows_total".to_string(), n_rows_total.to_string());
    let new_schema = Arc::new(Schema::new_with_metadata(
        batch.schema_ref().fields().clone(),
        metadata,
    ));
    // Columns and field count are unchanged — this can only fail if the
    // input batch is itself malformed, in which case `write_arrow_ipc`
    // would have failed too. We unwrap to keep the API ergonomic.
    RecordBatch::try_new(new_schema, batch.columns().to_vec())
        .expect("shard metadata stamping preserves schema fields")
}

impl ScxWriter {
    /// Create a new ScxWriter that will write to `path`.
    ///
    /// A temporary file is created alongside the final path. The header
    /// template is used for metadata (n_obs, n_vars, codec_id, etc.) but
    /// catalog offsets and shard count are filled in during `finish()`.
    ///
    /// See [`ScxWriter`] struct-level docs for the recommended section
    /// write order.
    pub fn new(path: impl AsRef<Path>, header: FileHeader) -> Result<Self> {
        let final_path = path.as_ref().to_path_buf();

        // Collision-safe sibling temp file (random suffix, intra-filesystem
        // rename target). The returned `TempPath` auto-deletes on drop if
        // `finish()` is never reached.
        let (file, tmp_path) = make_sibling_tempfile(&final_path)?;
        let mut writer = BufWriter::new(file);

        // Write 4352 zero bytes as placeholder for header + root catalog.
        // Stream zeros from `io::repeat` rather than allocating a 4 KB heap
        // buffer — same result with no allocation.
        std::io::copy(
            &mut std::io::repeat(0).take(SECTIONS_START_OFFSET),
            &mut writer,
        )?;

        Ok(ScxWriter {
            final_path,
            tmp_path: Some(tmp_path),
            file: Some(writer),
            current_offset: SECTIONS_START_OFFSET,
            header,
            entries: Vec::new(),
            csr_shard_count: 0,
            csc_shard_count: 0,
            raw_csr_shard_count: 0,
            raw_n_vars: 0,
            #[cfg(feature = "deletion-vectors")]
            bitmap_shard_count: 0,
            #[cfg(feature = "deletion-vectors")]
            modality_bitmap_counts: Vec::new(),
            total_nnz: 0,
            has_obsm: false,
            has_obsp: false,
            obs_layout: ObsVarLayout::Pending,
            var_layout: ObsVarLayout::Pending,
            current_modality_id: 0,
            modalities: Vec::new(),
            modality_build_csc: Vec::new(),
            data_generation: 1,
            csc_build_generation: None,
            framing: None,
        })
    }

    /// Override the CSR-data generation stamped into the output catalog.
    ///
    /// Defaults to `1` for a fresh write. CSR-mutating ops (`compact`,
    /// `merge`) call this with `source + 1`; CSC-only rewrites
    /// (`build-csc`) call it with the *unchanged* source generation so the
    /// freshly-emitted sidecar reads as fresh
    /// (`csc_build_generation == data_generation`). (`append` builds its
    /// catalog manually rather than via `finish()`, so it bumps the
    /// generation there, not through this setter; `subset` writes a fresh
    /// file at the default generation.)
    pub fn with_data_generation(mut self, data_generation: u64) -> Self {
        self.data_generation = data_generation;
        self
    }

    /// Adopt an already-open file mid-write so that section-emit methods
    /// (notably [`Self::write_obs_predicate_index`] /
    /// [`Self::write_var_predicate_index`]) can be reused from in-place
    /// rewrite paths that don't use `ScxWriter`'s standard temp-file →
    /// rename flow.
    ///
    /// Used by `scx-ops::append::finalize_append`: that function writes
    /// sections directly through a `FileLock`, so this constructor lets it
    /// hand the file off to `ScxWriter` for the predicate-index writes,
    /// then take the file + updated offset + new catalog entries back via
    /// [`Self::into_in_place_parts`]. The caller remains responsible for
    /// finalising the file (header, root catalog, full catalog, checksum,
    /// fsync) — [`Self::finish`] must NOT be called on an adopted writer.
    ///
    /// `current_offset` must match the file's actual cursor position; this
    /// method seeks the file there defensively. `existing_entries` should
    /// be the catalog entries already written ahead of `current_offset`;
    /// new section writes append to that vector.
    pub fn adopt_in_place(
        mut file: File,
        header: FileHeader,
        current_offset: u64,
        existing_entries: Vec<FullCatalogEntry>,
    ) -> Result<Self> {
        file.seek(SeekFrom::Start(current_offset))?;
        let writer = BufWriter::new(file);
        // Reconstruct obs/var layout state from existing entries so the
        // in-place append path doesn't allow a sharded file to suddenly
        // gain a single-section obs (or vice versa) via the adopted
        // writer's [`Self::write_obs`] / [`Self::write_obs_shard`].
        let obs_layout = obs_var_layout_from_entries(
            &existing_entries,
            SectionType::ObsMetadata,
            SectionType::ObsMetadataShard,
        );
        let var_layout = obs_var_layout_from_entries(
            &existing_entries,
            SectionType::VarMetadata,
            SectionType::VarMetadataShard,
        );
        Ok(ScxWriter {
            // `final_path` is only consulted by `finish()`, which adopted
            // writers must not call. Use an empty path to make accidental
            // use loud (it will surface as an obvious I/O error).
            final_path: PathBuf::new(),
            tmp_path: None,
            file: Some(writer),
            current_offset,
            header,
            entries: existing_entries,
            csr_shard_count: 0,
            csc_shard_count: 0,
            raw_csr_shard_count: 0,
            raw_n_vars: 0,
            #[cfg(feature = "deletion-vectors")]
            bitmap_shard_count: 0,
            #[cfg(feature = "deletion-vectors")]
            modality_bitmap_counts: Vec::new(),
            total_nnz: 0,
            has_obsm: false,
            has_obsp: false,
            obs_layout,
            var_layout,
            current_modality_id: 0,
            modalities: Vec::new(),
            modality_build_csc: Vec::new(),
            // Adopted writers (append in-place) never call `finish()`;
            // the caller stamps the catalog generations itself. Defaults
            // here are inert.
            data_generation: 1,
            csc_build_generation: None,
            framing: None,
        })
    }

    /// Tear down an adopted writer: flush the buffer and return the open
    /// file handle, the post-write offset, and the catalog entry list
    /// (existing entries from `adopt_in_place` plus any new entries
    /// pushed by section writes performed in between). The caller
    /// continues from there with its own header/catalog finalisation.
    ///
    /// Pairs with [`Self::adopt_in_place`]; not valid on writers created
    /// via [`Self::new`].
    pub fn into_in_place_parts(mut self) -> Result<(File, u64, Vec<FullCatalogEntry>)> {
        let buf_writer = self.file.take().ok_or(ScxError::WriterAlreadyFinished)?;
        let file = buf_writer.into_inner().map_err(std::io::Error::from)?;
        let entries = std::mem::take(&mut self.entries);
        let current_offset = self.current_offset;
        // `Drop` runs next: `tmp_path` is `None` for adopted writers, so
        // no temp-file cleanup occurs; `self.file` is `None`, so no
        // double-close. The caller now owns `file`.
        Ok((file, current_offset, entries))
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Get a mutable reference to the inner writer.
    fn writer(&mut self) -> Result<&mut BufWriter<File>> {
        self.file.as_mut().ok_or(ScxError::WriterAlreadyFinished)
    }

    /// Write zero-byte padding to reach 8-byte alignment.
    fn write_padding(&mut self) -> Result<()> {
        let aligned = align_to_8(self.current_offset);
        let pad = aligned - self.current_offset;
        if pad > 0 {
            // At most 7 bytes of padding — stack-allocated zero buffer avoids
            // the heap allocation that `vec![0u8; pad]` would incur.
            const ZEROS: [u8; 7] = [0; 7];
            self.writer()?.write_all(&ZEROS[..pad as usize])?;
            self.current_offset = aligned;
        }
        Ok(())
    }

    /// Core helper: pad, record offset, write data, compute checksum, push catalog entry.
    fn write_section_bytes(
        &mut self,
        name: impl Into<String>,
        section_type: SectionType,
        data: &[u8],
        stats: Option<ShardStats>,
    ) -> Result<()> {
        let name = name.into();

        // Central uniqueness guard (SCX-015): readers resolve a section name to
        // the first catalog match, so a duplicate (modality_id, section_type,
        // name) triple produces a file whose extra section is silently
        // unreachable. Reject it at the single write choke-point. Legitimate
        // shard families are unaffected because their names carry distinct
        // `_shard_N` suffixes.
        let modality_id = self.current_modality_id;
        if self.entries.iter().any(|e| {
            e.name == name && e.section_type == section_type && e.modality_id == modality_id
        }) {
            return Err(ScxError::DuplicateSection {
                name,
                section_type: format!("{section_type:?}"),
                modality_id,
            });
        }

        self.write_padding()?;

        let offset = self.current_offset;
        let length = data.len() as u64;
        let checksum = blake3_hash(data);

        self.writer()?.write_all(data)?;
        self.current_offset += length;

        self.entries.push(FullCatalogEntry {
            name,
            offset,
            length,
            section_type,
            checksum,
            modality_id,
            stats,
        });

        Ok(())
    }

    /// Serialize a RecordBatch to Arrow IPC file format bytes.
    ///
    /// Upcasts `Utf8 → LargeUtf8` and `Binary → LargeBinary` so columns
    /// larger than 2 GB do not overflow Arrow IPC's 32-bit offset limit.
    /// See [`crate::arrow_compat`] for details.
    fn write_arrow_ipc(batch: &RecordBatch) -> Result<Vec<u8>> {
        let batch = crate::arrow_compat::upcast_to_large_types(batch)?;
        let mut buf = Vec::new();
        {
            let mut writer = arrow::ipc::writer::FileWriter::try_new(&mut buf, batch.schema_ref())?;
            writer.write(&batch)?;
            writer.finish()?;
        }
        Ok(buf)
    }

    // -----------------------------------------------------------------------
    // Public section write methods
    // -----------------------------------------------------------------------

    /// Write the obs metadata section (Arrow IPC, single batch).
    ///
    /// Returns [`ScxError::ObsLayoutConflict`] if [`Self::write_obs_shard`]
    /// has already been called on this writer — the two layouts are
    /// mutually exclusive within one file. Use the sharded API for files
    /// whose obs may exceed Arrow IPC's 2 GB narrow-offset ceiling
    /// (~`i32::MAX` cumulative string-buffer bytes per column).
    pub fn write_obs(&mut self, obs: &RecordBatch) -> Result<()> {
        if let ObsVarLayout::Sharded(_) = self.obs_layout {
            return Err(ScxError::ObsLayoutConflict {
                attempted: "write_obs",
                existing: "write_obs_shard",
                single_kind: "ObsMetadata",
                sharded_kind: "ObsMetadataShard",
            });
        }
        let data = Self::write_arrow_ipc(obs)?;
        self.write_section_bytes("obs", SectionType::ObsMetadata, &data, None)?;
        self.obs_layout = ObsVarLayout::Single;
        Ok(())
    }

    /// Write the var metadata section (Arrow IPC, single batch).
    ///
    /// Returns [`ScxError::ObsLayoutConflict`] if [`Self::write_var_shard`]
    /// has already been called on this writer.
    pub fn write_var(&mut self, var: &RecordBatch) -> Result<()> {
        if let ObsVarLayout::Sharded(_) = self.var_layout {
            return Err(ScxError::ObsLayoutConflict {
                attempted: "write_var",
                existing: "write_var_shard",
                single_kind: "VarMetadata",
                sharded_kind: "VarMetadataShard",
            });
        }
        let data = Self::write_arrow_ipc(var)?;
        self.write_section_bytes("var", SectionType::VarMetadata, &data, None)?;
        self.var_layout = ObsVarLayout::Single;
        Ok(())
    }

    /// Write one row-shard of the obs metadata section ([`SectionType::ObsMetadataShard`]).
    ///
    /// Section name: `obs_metadata/shard_<shard_idx>`. Mirror of
    /// [`Self::write_obsm_shard`] for the obs metadata axis. The batch's
    /// `shard_idx` / `row_start` / `n_shard_rows` / `n_rows_total` are
    /// stamped into the schema metadata before encoding so the reader
    /// can verify a contiguous, ordered cover of the logical obs table.
    /// `write_arrow_ipc` (called internally) upcasts narrow `Utf8` /
    /// `Binary` to `LargeUtf8` / `LargeBinary` per shard, so each shard
    /// individually is safe from the 2 GB offset ceiling regardless of
    /// the merged total.
    ///
    /// Returns [`ScxError::ObsLayoutConflict`] if [`Self::write_obs`]
    /// has already been called on this writer.
    pub fn write_obs_shard(
        &mut self,
        shard_idx: u32,
        row_start: u64,
        n_shard_rows: u64,
        n_rows_total: u64,
        batch: &RecordBatch,
    ) -> Result<()> {
        match self.obs_layout {
            ObsVarLayout::Single => {
                return Err(ScxError::ObsLayoutConflict {
                    attempted: "write_obs_shard",
                    existing: "write_obs",
                    single_kind: "ObsMetadata",
                    sharded_kind: "ObsMetadataShard",
                });
            }
            ObsVarLayout::Pending | ObsVarLayout::Sharded(_) => {}
        }
        let stamped =
            stamp_dense_shard_metadata(batch, shard_idx, row_start, n_shard_rows, n_rows_total);
        let data = Self::write_arrow_ipc(&stamped)?;
        // Stamp the shard's global row range into the catalog so the query
        // engine can map this shard to its rows — and skip decoding it when
        // it doesn't overlap any surviving CSR shard — without reading the
        // payload. See `ShardStats::row_range_only`.
        self.write_section_bytes(
            format!("obs_metadata/shard_{shard_idx}"),
            SectionType::ObsMetadataShard,
            &data,
            Some(ShardStats::row_range_only(row_start, n_shard_rows)),
        )?;
        let next = match self.obs_layout {
            ObsVarLayout::Sharded(n) => n.saturating_add(1),
            _ => 1,
        };
        self.obs_layout = ObsVarLayout::Sharded(next);
        Ok(())
    }

    /// Write one row-shard of the var metadata section ([`SectionType::VarMetadataShard`]).
    ///
    /// Mirror of [`Self::write_obs_shard`] for the var axis. Section
    /// name: `var_metadata/shard_<shard_idx>`. Var rarely overflows the
    /// 2 GB ceiling (gene-count rather than cell-count axis), but the
    /// sharded API is symmetric for catalog ergonomics.
    pub fn write_var_shard(
        &mut self,
        shard_idx: u32,
        row_start: u64,
        n_shard_rows: u64,
        n_rows_total: u64,
        batch: &RecordBatch,
    ) -> Result<()> {
        match self.var_layout {
            ObsVarLayout::Single => {
                return Err(ScxError::ObsLayoutConflict {
                    attempted: "write_var_shard",
                    existing: "write_var",
                    single_kind: "VarMetadata",
                    sharded_kind: "VarMetadataShard",
                });
            }
            ObsVarLayout::Pending | ObsVarLayout::Sharded(_) => {}
        }
        let stamped =
            stamp_dense_shard_metadata(batch, shard_idx, row_start, n_shard_rows, n_rows_total);
        let data = Self::write_arrow_ipc(&stamped)?;
        // Mirror of `write_obs_shard`: stamp the row range for symmetry and
        // future var-axis pushdown.
        self.write_section_bytes(
            format!("var_metadata/shard_{shard_idx}"),
            SectionType::VarMetadataShard,
            &data,
            Some(ShardStats::row_range_only(row_start, n_shard_rows)),
        )?;
        let next = match self.var_layout {
            ObsVarLayout::Sharded(n) => n.saturating_add(1),
            _ => 1,
        };
        self.var_layout = ObsVarLayout::Sharded(next);
        Ok(())
    }

    /// Write the uns (unstructured) section as JSON.
    pub fn write_uns(&mut self, json: &serde_json::Value) -> Result<()> {
        // Last gate before the bytes exist: `serde_json::to_vec` has no depth
        // limit while the parser stops at 127 levels, so without this a caller
        // could store a section no reader — including ours — can take back.
        scx_format::validate_uns_depth(json)?;
        let data = serde_json::to_vec(json)?;
        self.write_section_bytes("uns", SectionType::UnsBlob, &data, None)
    }

    /// Write an obsm embedding section (Arrow IPC).
    pub fn write_obsm(&mut self, name: &str, batch: &RecordBatch) -> Result<()> {
        let data = Self::write_arrow_ipc(batch)?;
        self.has_obsm = true;
        self.write_section_bytes(
            format!("obsm/{name}"),
            SectionType::ObsmEmbedding,
            &data,
            None,
        )
    }

    /// Write a varm embedding section (Arrow IPC) — dense (n_vars × n_components).
    pub fn write_varm(&mut self, name: &str, batch: &RecordBatch) -> Result<()> {
        let data = Self::write_arrow_ipc(batch)?;
        self.write_section_bytes(
            format!("varm/{name}"),
            SectionType::VarmEmbedding,
            &data,
            None,
        )
    }

    /// Write an obsp pairwise sparse matrix (Arrow IPC, COO format).
    ///
    /// The batch must have columns `row: Int32`, `col: Int32`, `data: Float32`
    /// and schema metadata `n_rows` and `n_cols`.
    pub fn write_obsp(&mut self, name: &str, batch: &RecordBatch) -> Result<()> {
        let data = Self::write_arrow_ipc(batch)?;
        self.write_section_bytes(
            format!("obsp/{name}"),
            SectionType::ObspEmbedding,
            &data,
            None,
        )
    }

    /// Write a varp pairwise sparse matrix (Arrow IPC, COO format).
    ///
    /// Same wire format as `write_obsp`.
    pub fn write_varp(&mut self, name: &str, batch: &RecordBatch) -> Result<()> {
        let data = Self::write_arrow_ipc(batch)?;
        self.write_section_bytes(
            format!("varp/{name}"),
            SectionType::VarpEmbedding,
            &data,
            None,
        )
    }

    /// Write a row-shard of an `obsm/<name>` dense embedding (Arrow IPC).
    ///
    /// Section name: `obsm/<name>_shard_<shard_idx>`. The batch's
    /// `shard_idx` / `row_start` / `n_shard_rows` / `n_rows_total` are
    /// stamped into the schema metadata before encoding so the reader
    /// can verify a contiguous, ordered cover of the logical matrix
    /// when reassembling shards. For dense shards
    /// `n_shard_rows == batch.num_rows()`; the field is redundant
    /// but stamped uniformly so the reader's verification path is the
    /// same for dense and sparse. Readers (`ScxReader::read_obsm`)
    /// concatenate shards in `shard_idx` order; legacy single-section
    /// `ObsmEmbedding` files remain readable.
    pub fn write_obsm_shard(
        &mut self,
        name: &str,
        shard_idx: u32,
        row_start: u64,
        n_shard_rows: u64,
        n_rows_total: u64,
        batch: &RecordBatch,
    ) -> Result<()> {
        self.has_obsm = true;
        let stamped =
            stamp_dense_shard_metadata(batch, shard_idx, row_start, n_shard_rows, n_rows_total);
        let data = Self::write_arrow_ipc(&stamped)?;
        self.write_section_bytes(
            format!("obsm/{name}_shard_{shard_idx}"),
            SectionType::ObsmEmbeddingShard,
            &data,
            None,
        )
    }

    /// Write a row-shard of a `varm/<name>` dense embedding (Arrow IPC).
    ///
    /// Mirror of [`Self::write_obsm_shard`] for the `varm/` axis.
    pub fn write_varm_shard(
        &mut self,
        name: &str,
        shard_idx: u32,
        row_start: u64,
        n_shard_rows: u64,
        n_rows_total: u64,
        batch: &RecordBatch,
    ) -> Result<()> {
        let stamped =
            stamp_dense_shard_metadata(batch, shard_idx, row_start, n_shard_rows, n_rows_total);
        let data = Self::write_arrow_ipc(&stamped)?;
        self.write_section_bytes(
            format!("varm/{name}_shard_{shard_idx}"),
            SectionType::VarmEmbeddingShard,
            &data,
            None,
        )
    }

    /// Write a row-shard of an `obsp/<name>` pairwise sparse matrix
    /// (Arrow IPC COO).
    ///
    /// The batch must have the same column schema as the single-section
    /// [`Self::write_obsp`] (`row: Int32`, `col: Int32`, `data: Float32`)
    /// and contain only the non-zero triples whose `row` is in
    /// `[row_start, row_start + n_rows_in_shard)`. `row` values are
    /// stored as **global** indices (no shard-local renumbering); the
    /// reader concatenates shards without offset application. Schema
    /// metadata is augmented with `shard_idx`, `row_start`,
    /// `n_shard_rows`, and `n_rows_total`; the original `n_rows`/`n_cols`
    /// (the logical matrix shape) is preserved.
    pub fn write_obsp_shard_coo(
        &mut self,
        name: &str,
        shard_idx: u32,
        row_start: u64,
        n_shard_rows: u64,
        n_rows_total: u64,
        batch: &RecordBatch,
    ) -> Result<()> {
        self.has_obsp = true;
        let stamped =
            stamp_dense_shard_metadata(batch, shard_idx, row_start, n_shard_rows, n_rows_total);
        let data = Self::write_arrow_ipc(&stamped)?;
        self.write_section_bytes(
            format!("obsp/{name}_shard_{shard_idx}"),
            SectionType::ObspEmbeddingShard,
            &data,
            None,
        )
    }

    /// Write a row-shard of a `varp/<name>` pairwise sparse matrix
    /// (Arrow IPC COO). Mirror of [`Self::write_obsp_shard_coo`].
    pub fn write_varp_shard_coo(
        &mut self,
        name: &str,
        shard_idx: u32,
        row_start: u64,
        n_shard_rows: u64,
        n_rows_total: u64,
        batch: &RecordBatch,
    ) -> Result<()> {
        let stamped =
            stamp_dense_shard_metadata(batch, shard_idx, row_start, n_shard_rows, n_rows_total);
        let data = Self::write_arrow_ipc(&stamped)?;
        self.write_section_bytes(
            format!("varp/{name}_shard_{shard_idx}"),
            SectionType::VarpEmbeddingShard,
            &data,
            None,
        )
    }

    /// Write the provenance section.
    pub fn write_provenance(&mut self, operations: Vec<ProvenanceEntry>) -> Result<()> {
        let prov = Provenance {
            version: 1,
            operations,
        };
        let mut data = Vec::new();
        prov.write_to(&mut data)?;
        self.write_section_bytes("provenance", SectionType::Provenance, &data, None)
    }

    /// Write a CSR shard (X matrix).
    ///
    /// - `indptr`: the indptr array (length = n_rows + 1), on-disk u64 values.
    /// - `indices`: column indices (length = nnz), as u32.
    /// - `values`: raw LE bytes of the value array.
    /// - `codec_id`: compression codec to use.
    /// - `value_encoding`: how values are encoded.
    /// - `row_start`: global row index where this shard begins.
    pub fn write_csr_shard(
        &mut self,
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        codec_id: CodecId,
        value_encoding: ValueEncoding,
        row_start: u64,
    ) -> Result<()> {
        let shard_idx = self.csr_shard_count;
        let name = format!("X_shard_{shard_idx}");
        let nnz = *indptr.last().unwrap_or(&0);
        self.write_shard_inner(
            indptr,
            indices,
            values,
            codec_id,
            value_encoding,
            row_start,
            &name,
            SectionType::CsrShard,
        )?;
        self.csr_shard_count += 1;
        self.total_nnz += nnz;
        Ok(())
    }

    /// Set the raw matrix column count (`raw.n_vars`). MUST be called
    /// before the first [`Self::write_raw_csr_shard`] so the shard
    /// machinery stamps the correct minor-axis extent and per-shard
    /// `index_dtype` for the raw matrix's own (independent) var axis.
    pub fn set_raw_n_vars(&mut self, raw_n_vars: u64) {
        self.raw_n_vars = raw_n_vars;
    }

    /// Write one row-shard of the `adata.raw` count matrix
    /// ([`SectionType::RawCsrShard`]). Structurally identical to
    /// [`Self::write_csr_shard`] but emits the raw section type and is
    /// named `raw/X_shard_<idx>`. Does NOT touch `csr_shard_count` /
    /// `total_nnz` (those track the main matrix). Call
    /// [`Self::set_raw_n_vars`] first.
    pub fn write_raw_csr_shard(
        &mut self,
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        codec_id: CodecId,
        value_encoding: ValueEncoding,
        row_start: u64,
    ) -> Result<()> {
        let shard_idx = self.raw_csr_shard_count;
        let name = format!("raw/X_shard_{shard_idx}");
        self.write_shard_inner(
            indptr,
            indices,
            values,
            codec_id,
            value_encoding,
            row_start,
            &name,
            SectionType::RawCsrShard,
        )?;
        self.raw_csr_shard_count += 1;
        Ok(())
    }

    /// Write the raw var metadata section ([`SectionType::RawVarMetadata`],
    /// Arrow IPC, single batch). Companion to the `raw/X` shards.
    pub fn write_raw_var(&mut self, raw_var: &RecordBatch) -> Result<()> {
        let data = Self::write_arrow_ipc(raw_var)?;
        self.write_section_bytes("raw/var", SectionType::RawVarMetadata, &data, None)?;
        Ok(())
    }

    /// Enable/disable F5-b row-group framing for subsequent sparse-shard writes
    /// through this writer (CSC sidecars, layers, obsp, and CSR X shards written
    /// via `write_csr_shard`). `None` restores the legacy unframed layout. The
    /// caller bumps the file `format_version` to v4 when framing (the convert
    /// path does this alongside setting this).
    ///
    /// A config whose `decode_target` is `Some` (or whose `trial` is set)
    /// authorises this writer to **re-select the per-shard integer codec**,
    /// overriding the `codec_id` passed to `write_csr_shard`. Pass
    /// `FramingConfig::default()` (`decode_target: None`) when the intent is to
    /// preserve a source codec. See `FramingConfig`'s contract section.
    pub fn set_framing(&mut self, framing: Option<crate::encoder::FramingConfig>) {
        self.framing = framing;
    }

    /// The writer's current framing setting. Lets a helper temporarily override
    /// framing for a scoped batch (e.g. [`crate::csc_sidecar::write_csc_sidecar`])
    /// and restore the prior value afterward.
    pub fn framing(&self) -> Option<crate::encoder::FramingConfig> {
        self.framing
    }

    /// Write a CSC shard (column-major sparse matrix).
    ///
    /// Structurally identical to a CSR shard but uses `SectionType::CscShard (5)`.
    /// - `indptr`: the column pointer array (length = n_cols_in_shard + 1), on-disk u64 values.
    /// - `indices`: row indices (length = nnz), as u32.
    /// - `values`: raw LE bytes of the value array.
    /// - `codec_id`: compression codec to use.
    /// - `value_encoding`: how values are encoded.
    /// - `col_start`: global column index where this shard begins.
    pub fn write_csc_shard(
        &mut self,
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        codec_id: CodecId,
        value_encoding: ValueEncoding,
        col_start: u64,
    ) -> Result<()> {
        let shard_idx = self.csc_shard_count;
        let name = format!("X_csc_shard_{shard_idx}");
        self.write_shard_inner(
            indptr,
            indices,
            values,
            codec_id,
            value_encoding,
            col_start,
            &name,
            SectionType::CscShard,
        )?;
        self.csc_shard_count += 1;
        Ok(())
    }

    /// Write a layer CSR shard.
    #[allow(clippy::too_many_arguments)]
    pub fn write_layer_csr_shard(
        &mut self,
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        codec_id: CodecId,
        value_encoding: ValueEncoding,
        row_start: u64,
        layer_name: &str,
        shard_idx: u32,
    ) -> Result<()> {
        let name = format!("{layer_name}_shard_{shard_idx}");
        self.write_shard_inner(
            indptr,
            indices,
            values,
            codec_id,
            value_encoding,
            row_start,
            &name,
            SectionType::LayerCsrShard,
        )
    }

    /// Write an obsp CSR shard.
    ///
    /// Emits a [`SectionType::ObspCsrShard`], which `scx validate --deep`
    /// checks against the v3 canonical-CSR invariant. The caller MUST pass
    /// canonical CSR (per-row indices strictly increasing, duplicate
    /// coordinates summed, explicit zeros dropped — see
    /// [`scx_sparse::canonicalize_csr`]); this writer does not canonicalize.
    #[allow(clippy::too_many_arguments)]
    pub fn write_obsp_shard(
        &mut self,
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        codec_id: CodecId,
        value_encoding: ValueEncoding,
        row_start: u64,
        obsp_name: &str,
        shard_idx: u32,
    ) -> Result<()> {
        self.has_obsp = true;
        let name = format!("obsp/{obsp_name}_shard_{shard_idx}");
        self.write_shard_inner(
            indptr,
            indices,
            values,
            codec_id,
            value_encoding,
            row_start,
            &name,
            SectionType::ObspCsrShard,
        )
    }

    /// Core shard writing logic shared by write_csr_shard, write_layer_csr_shard, write_obsp_shard.
    #[allow(clippy::too_many_arguments)]
    fn write_shard_inner(
        &mut self,
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        codec_id: CodecId,
        value_encoding: ValueEncoding,
        row_start: u64,
        name: &str,
        section_type: SectionType,
    ) -> Result<()> {
        // Any CSC sidecar shard (X or layer, single- or multi-modality)
        // funnels through here, so this is the one place to record that
        // the sidecar was built against the current `data_generation`.
        // `finish()` stamps it into `FullCatalog::csc_build_generation`;
        // the reader rejects a sidecar whose recorded generation does not
        // match `data_generation` (staleness guard). Mutating ops that
        // drop CSC never reach this branch, so the field stays `0`.
        if crate::shard::is_column_major(section_type) {
            self.csc_build_generation = Some(self.data_generation);
        }

        self.write_padding()?;

        let shard_global_offset = self.current_offset;
        if indptr.is_empty() {
            return Err(ScxError::EmptyIndptr);
        }
        let n_major = (indptr.len() - 1) as u32;
        let nnz = *indptr.last().unwrap_or(&0);

        // Resolve the unbound-minor extent for this shard. For CSC shards
        // the minor axis is rows (file-wide `n_obs`, shared across
        // modalities). For row-major shards (CSR / LayerCsr / ObspCsr)
        // the minor axis is columns, which is per-modality: when
        // `current_modality_id > 0` we MUST use that modality's `n_vars`
        // rather than the file-wide max (`self.header.n_vars`), otherwise
        // every shard in a multi-modality file gets stamped with the max
        // and column-range pruning / bounds checks break.
        let row_major_n_minor: u64 = if self.current_modality_id > 0 {
            self.modalities
                .get((self.current_modality_id - 1) as usize)
                .map(|m| m.n_vars)
                .unwrap_or(self.header.n_vars)
        } else {
            self.header.n_vars
        };

        // Decide indices encoding width per shard, based on the actual
        // index-value bound for THIS shard layout. CSC stores row
        // indices (bounded by `n_obs`); CSR / LayerCsr / ObspCsr store
        // column indices (bounded by `row_major_n_minor`). The file-
        // level `header.index_dtype` was set at file creation from
        // `n_vars` (CSR-correct); using it unconditionally for CSC
        // breaks files with n_obs > 65535 and n_vars ≤ 65535 (e.g.
        // census_500k / census_1m). The per-shard `index_dtype` field
        // in `ShardHeader` is the source of truth at read time
        // (see `sh.index_dtype == 0` in the reader), so widening to
        // u32 here on a CSC shard is read-correct even when the file
        // header says u16.
        let index_max_value: u64 = match section_type {
            // Raw is row-major but has its OWN column axis (`raw_n_vars`),
            // independent of `header.n_vars`.
            SectionType::RawCsrShard => self.raw_n_vars.saturating_sub(1),
            st if crate::shard::is_column_major(st) => self.header.n_obs.saturating_sub(1),
            _ => row_major_n_minor.saturating_sub(1),
        };
        let index_dtype_u16 = index_max_value <= u16::MAX as u64;
        let shard_index_dtype: u8 = if index_dtype_u16 { 0 } else { 1 };

        // Encode the shard data. When row-group framing is enabled on the writer
        // (F5-b), every sparse shard funneling through here — CSC sidecars,
        // layers, obsp — is emitted row-group-framed (shard v2) for
        // codec-agnostic sub-shard random access; else the monolithic layout
        // (shard v1, byte-identical to legacy).
        //
        // `codec_id` is the caller's *candidate*. When `self.framing` carries a
        // `decode_target` (`auto`/`compact`) or `trial`, an integer shard is
        // dual-encoded against ShufDeltaZstd and `chosen_codec` may differ — see
        // `encode_shard_adaptive`. Before that call existed this path ignored
        // both fields, so every shard written through here (layers, the CSC
        // sidecar, multimodal X) silently got the `fast` heuristic even when the
        // caller asked for `auto`.
        let (encoded, block_index, shard_version, chosen_codec) =
            crate::encoder::encode_shard_adaptive(
                indptr,
                indices,
                values,
                codec_id,
                value_encoding,
                index_dtype_u16,
                self.framing,
            )?;
        let mut block_index_bytes = Vec::new();
        block_index.write_to(&mut block_index_bytes)?;

        // Compute shard-level checksum: streaming blake3 of (indptr + indices + values + block_index), truncated to 8 bytes
        let mut shard_hasher = blake3::Hasher::new();
        shard_hasher.update(&encoded.indptr_bytes);
        shard_hasher.update(&encoded.indices_bytes);
        shard_hasher.update(&encoded.values_bytes);
        shard_hasher.update(&block_index_bytes);
        let shard_hash = shard_hasher.finalize();
        let mut shard_checksum = [0u8; 8];
        shard_checksum.copy_from_slice(&shard_hash.as_bytes()[..8]);

        // Build shard header with relative offsets. u32 fields; fail loud on a
        // >4 GiB sub-stream / cumulative offset instead of silently wrapping (F-c).
        let len_u32 = |n: usize, what: &str| -> Result<u32> {
            u32::try_from(n).map_err(|_| {
                ScxError::ShardStreamTooLarge(format!("{what} length {n} exceeds u32::MAX"))
            })
        };
        let add_u32 = |a: u32, b: u32, what: &str| -> Result<u32> {
            a.checked_add(b).ok_or_else(|| {
                ScxError::ShardStreamTooLarge(format!("{what} relative offset exceeds u32::MAX"))
            })
        };
        let indptr_rel_offset = SHARD_HEADER_SIZE as u32;
        let indptr_length = len_u32(encoded.indptr_bytes.len(), "indptr")?;
        let indices_rel_offset = add_u32(indptr_rel_offset, indptr_length, "indices")?;
        let indices_length = len_u32(encoded.indices_bytes.len(), "indices")?;
        let values_rel_offset = add_u32(indices_rel_offset, indices_length, "values")?;
        let values_length = len_u32(encoded.values_bytes.len(), "values")?;
        let block_index_rel_offset = add_u32(values_rel_offset, values_length, "block_index")?;
        let block_index_length = len_u32(block_index_bytes.len(), "block_index")?;

        let shard_header = ShardHeader {
            magic: crate::shard::SHARD_MAGIC,
            shard_format_version: shard_version,
            shard_type: derive_shard_type(section_type),
            // The adaptively-chosen codec, NOT the caller's candidate — the
            // reader dispatches on this byte.
            codec_id: chosen_codec as u8,
            value_encoding: value_encoding as u8,
            index_dtype: shard_index_dtype,
            reserved_flags: [0; 3],
            n_major,
            n_minor: {
                // CSC shards have n_obs as their minor axis (and
                // `row_major_n_minor` is unused for those); row-major
                // shards use the per-modality column count resolved
                // above so multimodal files stamp the correct extent.
                let header_n_minor = match section_type {
                    SectionType::RawCsrShard => self.raw_n_vars,
                    st if crate::shard::is_column_major(st) => self.header.n_obs,
                    _ => row_major_n_minor,
                };
                if header_n_minor > u32::MAX as u64 {
                    return Err(ScxError::NVarsOverflow(header_n_minor));
                }
                header_n_minor as u32
            },
            nnz,
            global_offset: row_start,
            indptr_rel_offset,
            indptr_length,
            indices_rel_offset,
            indices_length,
            values_rel_offset,
            values_length,
            block_index_rel_offset,
            block_index_length,
            checksum: shard_checksum,
        };

        // Serialize shard header to buffer
        let mut header_buf = Vec::with_capacity(SHARD_HEADER_SIZE);
        shard_header.write_to(&mut header_buf)?;

        // Compute section-level BLAKE3 (full 32-byte) via streaming hasher — no section_data Vec needed
        let mut section_hasher = blake3::Hasher::new();

        section_hasher.update(&header_buf);
        self.writer()?.write_all(&header_buf)?;

        section_hasher.update(&encoded.indptr_bytes);
        self.writer()?.write_all(&encoded.indptr_bytes)?;

        section_hasher.update(&encoded.indices_bytes);
        self.writer()?.write_all(&encoded.indices_bytes)?;

        section_hasher.update(&encoded.values_bytes);
        self.writer()?.write_all(&encoded.values_bytes)?;

        section_hasher.update(&block_index_bytes);
        self.writer()?.write_all(&block_index_bytes)?;

        let section_checksum = *section_hasher.finalize().as_bytes();
        let section_length = (header_buf.len()
            + encoded.indptr_bytes.len()
            + encoded.indices_bytes.len()
            + encoded.values_bytes.len()
            + block_index_bytes.len()) as u64;
        self.current_offset += section_length;

        // Compute shard stats from raw values. Dispatch on section
        // type: column-major shards use the column-major axis (the
        // file-wide `n_obs` is the unbound row range, shared across
        // modalities), all others use the row-major axis with the
        // per-modality column extent resolved at the top of this
        // function. `row_start` here is interpreted on the major axis —
        // for column-major paths it carries `col_start` (see
        // `write_csc_shard`, which passes its `col_start` argument as
        // the inner `row_start`).
        //
        // `is_column_major` rather than a bare `CscShard` arm, here and in
        // the two dispatches above: `LayerCscShard` is a CSC sidecar too,
        // and matching only `CscShard` wrote its stats on the row axis
        // while the readers looked for them on the column axis.
        let (major_kind, n_minor) = match section_type {
            SectionType::RawCsrShard => (MajorAxis::Row, self.raw_n_vars),
            st if crate::shard::is_column_major(st) => (MajorAxis::Col, self.header.n_obs),
            _ => (MajorAxis::Row, row_major_n_minor),
        };
        let stats = compute_shard_stats(
            values,
            value_encoding,
            major_kind,
            row_start,
            n_major as u64,
            n_minor,
            nnz,
        );

        self.entries.push(FullCatalogEntry {
            name: name.to_string(),
            offset: shard_global_offset,
            length: section_length,
            section_type,
            checksum: section_checksum,
            modality_id: self.current_modality_id,
            stats: Some(stats),
        });

        Ok(())
    }

    /// Write a pre-encoded shard section verbatim (no encode/compress).
    ///
    /// The provided `raw_bytes` must be a complete shard section (76-byte
    /// header + encoded payload) as returned by `ScxReader::read_raw_shard_bytes`.
    /// Used for fast shard copying where the data does not need transformation.
    ///
    /// The section-level BLAKE3 checksum is recomputed from the raw bytes for
    /// the catalog entry. The inner shard-level checksum (in the 76-byte
    /// header) is preserved as-is.
    #[allow(clippy::too_many_arguments)]
    pub fn write_raw_shard(
        &mut self,
        raw_bytes: &[u8],
        section_type: SectionType,
        name: &str,
        stats: ShardStats,
        nnz: u64,
    ) -> Result<()> {
        self.write_padding()?;

        let shard_global_offset = self.current_offset;
        let section_length = raw_bytes.len() as u64;
        let section_checksum = blake3_hash(raw_bytes);

        self.writer()?.write_all(raw_bytes)?;
        self.current_offset += section_length;

        self.entries.push(FullCatalogEntry {
            name: name.to_string(),
            offset: shard_global_offset,
            length: section_length,
            section_type,
            checksum: section_checksum,
            modality_id: self.current_modality_id,
            stats: Some(stats),
        });

        match section_type {
            SectionType::CsrShard => {
                self.csr_shard_count += 1;
                self.total_nnz += nnz;
            }
            SectionType::CscShard => {
                // CSC shards count toward `n_csc_shards` so that
                // `finish()` populates the header's `n_csc_shards` and
                // `has_csc` flag bit. Without this, cloud
                // pass-through paths (`cloud_optimize`, `pack`,
                // `push`, `pull`) silently dropped CSC sidecars on
                // copy.
                self.csc_shard_count += 1;
                // Don't add to total_nnz: CSC shards mirror the same
                // values as CSR shards (different layout, same
                // entries). Adding here would double-count nnz.
            }
            // Raw sections bump no counter here — see the note in
            // `write_preencoded_shard`. A future raw-aware verbatim path
            // must advance `self.raw_csr_shard_count` for `raw/X_shard`
            // naming.
            _ => {}
        }

        Ok(())
    }

    /// Write a raw-copied CSR shard section (single-modality).
    ///
    /// `section_bytes` is a complete shard section (76-byte header + payload)
    /// whose header has already been patched for the new `global_offset` /
    /// `n_minor` (see the merge / append raw-copy fast paths). This auto-names
    /// `X_shard_<idx>` exactly like [`Self::write_csr_shard`] and advances the
    /// same counters, so a file mixing raw-copied and re-encoded shards stays
    /// catalog-consistent.
    pub fn write_csr_shard_raw_copy(
        &mut self,
        section_bytes: &[u8],
        stats: ShardStats,
        nnz: u64,
    ) -> Result<()> {
        let name = format!("X_shard_{}", self.csr_shard_count);
        self.write_csr_shard_raw_copy_inner(section_bytes, &name, stats)?;
        self.csr_shard_count += 1;
        self.total_nnz += nnz;
        Ok(())
    }

    /// Per-modality [`Self::write_csr_shard_raw_copy`]. Section name is
    /// `X/{modality_name}/shard_{idx}`, mirroring [`Self::write_csr_shard_for`].
    pub fn write_csr_shard_raw_copy_for(
        &mut self,
        modality_id: u8,
        section_bytes: &[u8],
        stats: ShardStats,
        nnz: u64,
    ) -> Result<()> {
        let mname = self.modality_name_for(modality_id)?;
        let shard_idx = self
            .modalities
            .get((modality_id - 1) as usize)
            .map(|m| m.n_csr_shards)
            .unwrap_or(0);
        let name = format!("X/{mname}/shard_{shard_idx}");
        self.with_modality(modality_id, |this| {
            this.write_csr_shard_raw_copy_inner(section_bytes, &name, stats)
        })?;
        if let Some(info) = self.modalities.get_mut((modality_id - 1) as usize) {
            info.n_csr_shards += 1;
            info.nnz += nnz;
        }
        Ok(())
    }

    /// Defensive guard: refuse to place an **unframed** (shard v1) CSR-class
    /// shard into a file that claims the v4 layout.
    ///
    /// A file `format_version` of [`CURRENT_FORMAT_VERSION`](scx_format::CURRENT_FORMAT_VERSION)
    /// (v4) advertises "every sparse shard supports sub-shard random access."
    /// Only a **framed** (shard v2) shard satisfies that, via its group
    /// `BlockIndex`. Legacy rewrite paths stamp ≤ v3, so the guard is inert for
    /// them; only CSR-class shards carry the layout, so obs/var/aux verbatim
    /// copies are exempt.
    fn guard_no_legacy_shard_in_v4(
        &self,
        section_type: SectionType,
        section_bytes: &[u8],
    ) -> Result<()> {
        if self.header.format_version < scx_format::CURRENT_FORMAT_VERSION {
            return Ok(());
        }
        if !matches!(
            section_type,
            SectionType::CsrShard | SectionType::LayerCsrShard | SectionType::ObspCsrShard
        ) {
            return Ok(());
        }
        let sh = ShardHeader::read_from(&mut &section_bytes[..])?;
        if sh.shard_format_version <= crate::shard::DEFAULT_WRITE_SHARD_FORMAT_VERSION {
            return Err(ScxError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "refusing to place an unframed (shard v{}) shard into a v{} file: it \
                     would advertise sub-shard random access it cannot honor. Re-encode it \
                     row-group-framed.",
                    sh.shard_format_version, self.header.format_version,
                ),
            )));
        }
        Ok(())
    }

    fn write_csr_shard_raw_copy_inner(
        &mut self,
        section_bytes: &[u8],
        name: &str,
        stats: ShardStats,
    ) -> Result<()> {
        self.guard_no_legacy_shard_in_v4(SectionType::CsrShard, section_bytes)?;
        self.write_padding()?;
        let shard_global_offset = self.current_offset;
        let section_length = section_bytes.len() as u64;
        let section_checksum = blake3_hash(section_bytes);
        self.writer()?.write_all(section_bytes)?;
        self.current_offset += section_length;

        self.entries.push(FullCatalogEntry {
            name: name.to_string(),
            offset: shard_global_offset,
            length: section_length,
            section_type: SectionType::CsrShard,
            checksum: section_checksum,
            modality_id: self.current_modality_id,
            stats: Some(stats),
        });
        Ok(())
    }

    /// Write a pre-encoded shard section produced by parallel encoding.
    ///
    /// The encoding, checksums, and stats have all been computed in advance
    /// (typically in parallel via rayon). This method only performs the
    /// sequential I/O write and catalog entry bookkeeping.
    ///
    /// Note: for a `CsrShard` this bumps the *global* `csr_shard_count` /
    /// `total_nnz` unconditionally, even inside a [`Self::with_modality`] scope
    /// (unlike [`Self::write_csr_shard_for`], which only touches the per-modality
    /// counters). That is inert today — the header's `n_csr_shards` / `nnz` are
    /// re-derived from the catalog in [`Self::finish`], and the global counter is
    /// only read by the single-modality `X_shard_{n}` auto-namer — but a writer
    /// that interleaves modality-scoped `write_preencoded_shard` calls with a
    /// later global `write_csr_shard` would see a skewed auto-name index.
    pub fn write_preencoded_shard(&mut self, section: PreEncodedSection) -> Result<()> {
        self.guard_no_legacy_shard_in_v4(section.section_type, &section.header_buf)?;
        self.write_padding()?;

        let shard_global_offset = self.current_offset;

        let w = self.writer()?;
        w.write_all(&section.header_buf)?;
        w.write_all(&section.encoded.indptr_bytes)?;
        w.write_all(&section.encoded.indices_bytes)?;
        w.write_all(&section.encoded.values_bytes)?;
        w.write_all(&section.block_index_bytes)?;

        self.current_offset += section.section_length;

        self.entries.push(FullCatalogEntry {
            name: section.name,
            offset: shard_global_offset,
            length: section.section_length,
            section_type: section.section_type,
            checksum: section.section_checksum,
            modality_id: self.current_modality_id,
            stats: Some(section.stats),
        });

        // Per-modality bookkeeping: mirror `write_csr_shard_for` /
        // `write_csc_shard_for` so the modality table `finish()` emits carries
        // non-zero `nnz`/`n_csr_shards`/`n_csc_shards`/`has_csc`. No-op outside
        // a `with_modality` scope (single-modality files). See
        // `record_modality_shard`.
        self.record_modality_shard(section.section_type, section.nnz);
        match section.section_type {
            SectionType::CsrShard => {
                self.csr_shard_count += 1;
                self.total_nnz += section.nnz;
            }
            SectionType::CscShard => {
                // CSC shards count toward `n_csc_shards` so that
                // `finish()` populates the header's `n_csc_shards` and
                // `has_csc` flag bit. Without this, cloud
                // pass-through paths (`cloud_optimize`, `pack`,
                // `push`, `pull`) silently dropped CSC sidecars on
                // copy.
                self.csc_shard_count += 1;
                // Don't add to total_nnz: CSC shards mirror the same
                // values as CSR shards (different layout, same
                // entries). Adding here would double-count nnz.
            }
            // `RawCsrShard` (and metadata sections) deliberately bump no
            // counter: raw must not perturb the main matrix's
            // `n_csr_shards`/`total_nnz`, and `has_raw` is derived from the
            // catalog by `FileHeader::sync_from_catalog` rather than from a
            // counter. `raw_csr_shard_count` is not advanced either, and does
            // not need to be: it exists only to name `raw/X_shard_<idx>` in
            // `write_raw_csr_shard`, whereas every caller on this path
            // (`pipeline::ingest_raw_streaming` via the streaming
            // coordinator, and pyscx's in-memory `from_anndata` raw write)
            // names its own shards from its own shard index before encoding.
            _ => {}
        }

        Ok(())
    }

    /// Copy a pre-encoded section verbatim from another SCX file into
    /// this writer.
    ///
    /// Used by the Phase 8b SCX → SCX byte-passthrough path
    /// (`pyscx.from_anndata` with `ScxBackedSparseDataset` input):
    /// when the source and target shard layouts agree (same
    /// `shard_target_rows`, same codec, same value encoding), the
    /// source shard bytes — header, indptr, indices, values,
    /// block-index — can be written into the new file without
    /// decode + re-encode. Also used by `scx optimize` to carry
    /// non-CSR sections through unchanged — obsm/varm/obsp/varp
    /// matrices and the `BitmapShard` / `GroupIndex` sidecars —
    /// where the source layout is preserved 1:1.
    ///
    /// `src_entry` carries the source section's name, length,
    /// checksum, `section_type`, modality routing key, and shard
    /// stats. This method writes `raw_bytes` (the full section
    /// payload as returned by [`ScxReader::read_raw_shard_bytes`]),
    /// pushes a new [`FullCatalogEntry`] with the file offset
    /// adjusted to the writer's current cursor (every other field
    /// cloned from `src_entry`), and updates the writer's shard
    /// counters and `total_nnz` accumulator to mirror
    /// [`Self::write_preencoded_shard`]'s bookkeeping.
    ///
    /// `raw_bytes.len()` must equal `src_entry.length`.
    pub fn copy_section_verbatim(
        &mut self,
        src_entry: &FullCatalogEntry,
        raw_bytes: &[u8],
    ) -> Result<()> {
        if raw_bytes.len() as u64 != src_entry.length {
            return Err(ScxError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "copy_section_verbatim: raw_bytes length {} does not match catalog entry length {}",
                    raw_bytes.len(),
                    src_entry.length,
                ),
            )));
        }

        // A v4 file requires framed (shard v2) CSR-class shards; reject an
        // unframed verbatim copy into one.
        self.guard_no_legacy_shard_in_v4(src_entry.section_type, raw_bytes)?;

        self.write_padding()?;

        let shard_global_offset = self.current_offset;
        self.writer()?.write_all(raw_bytes)?;
        self.current_offset += src_entry.length;

        let nnz = src_entry.stats.as_ref().map(|s| s.nnz).unwrap_or(0);

        self.entries.push(FullCatalogEntry {
            name: src_entry.name.clone(),
            offset: shard_global_offset,
            length: src_entry.length,
            section_type: src_entry.section_type,
            checksum: src_entry.checksum,
            modality_id: self.current_modality_id,
            stats: src_entry.stats.clone(),
        });

        // Per-modality bookkeeping — see `write_preencoded_shard`.
        self.record_modality_shard(src_entry.section_type, nnz);
        match src_entry.section_type {
            SectionType::CsrShard => {
                self.csr_shard_count += 1;
                self.total_nnz += nnz;
            }
            SectionType::CscShard => {
                self.csc_shard_count += 1;
                // Don't double-count nnz — CSC mirrors CSR (see
                // `write_preencoded_shard`).
            }
            // Raw sections bump no counter here — see the note in
            // `write_preencoded_shard`. A future raw-aware verbatim copy
            // must advance `self.raw_csr_shard_count` for `raw/X_shard`
            // naming.
            _ => {}
        }

        Ok(())
    }

    /// Set per-column statistics on the last written shard entry.
    ///
    /// This must be called immediately after `write_csr_shard()` (or its
    /// layer/obsp variants) to attach `CategoryBitset` or `MinMax` column
    /// stats that enable catalog-level predicate pushdown (docs/format.md (Dual Catalog)).
    ///
    /// If the last entry is not a shard (no stats), this is a no-op.
    pub fn set_shard_column_stats(
        &mut self,
        column_stats: Vec<crate::catalog::ColumnStat>,
    ) -> Result<()> {
        if column_stats.len() > u8::MAX as usize {
            return Err(ScxError::ColumnStatsOverflow(column_stats.len()));
        }
        if let Some(entry) = self.entries.last_mut() {
            if let Some(ref mut stats) = entry.stats {
                stats.n_indexed_columns = column_stats.len() as u8;
                stats.column_stats = column_stats;
            }
        }
        Ok(())
    }

    /// Attach per-shard column statistics to every CSR shard at once.
    ///
    /// `per_shard[k]` is the `Vec<ColumnStat>` for the k-th CSR shard in
    /// `FullCatalog::csr_shards_sorted` order (sorted by `row_start`). This is
    /// the same `shard_id` space the obs predicate index is built against
    /// (`shard_row_ranges` index == catalog CSR-shard sorted position), so the
    /// per-shard `CategoryBitset` / `MinMax` stats derived from that index line
    /// up with the shards `prune_shards_by_catalog_with_dict` iterates at query
    /// time. Unlike [`Self::set_shard_column_stats`] (which only touches the
    /// last-written entry) this addresses shards by position, in a single sort
    /// pass — O(n log n) rather than O(n²) on atlas-scale shard counts.
    ///
    /// `per_shard.len()` must equal the number of **`modality_id == 0`** CSR
    /// shards written so far — not the total. On a multimodal file every CSR
    /// shard carries a nonzero modality id, so that count is zero and any
    /// non-empty `per_shard` is rejected with
    /// `ColumnStatsShardCountMismatch`; see
    /// [`assign_csr_shard_column_stats`] for why failing closed is the right
    /// answer there.
    pub fn set_csr_shard_column_stats_bulk(
        &mut self,
        per_shard: Vec<Vec<crate::catalog::ColumnStat>>,
    ) -> Result<()> {
        assign_csr_shard_column_stats(&mut self.entries, per_shard)
    }

    // -----------------------------------------------------------------------
    // Phase B: per-modality writer API
    // -----------------------------------------------------------------------

    /// Register a modality and return its 1-based `modality_id`.
    ///
    /// Modalities must be registered up-front (before any per-modality
    /// section writes) so that subsequent `*_for(modality_id, …)`
    /// calls have a valid id to stamp on catalog entries. The order
    /// of registration is significant — the on-disk `ModalityTable`
    /// preserves insertion order, and the assigned `modality_id`
    /// equals position-in-table + 1.
    ///
    /// `modality_id = 0` is reserved for "global" entries (obs,
    /// obs_index, provenance, …) and cannot be allocated by this
    /// method.
    pub fn add_modality(
        &mut self,
        name: &str,
        modality_type: ModalityType,
        default_codec: CodecId,
        default_value_encoding: ValueEncoding,
        build_csc: bool,
    ) -> Result<u8> {
        ModalityTable::validate_name(name)?;
        if self.modalities.iter().any(|m| m.name == name) {
            return Err(ScxError::InvalidCatalog(format!(
                "modality name '{name}' already registered"
            )));
        }
        if self.modalities.len() >= MAX_MODALITIES as usize {
            return Err(ScxError::InvalidCatalog(format!(
                "cannot register more than {MAX_MODALITIES} modalities"
            )));
        }
        let info = ModalityInfo {
            name: name.to_string(),
            modality_type,
            default_codec_id: default_codec as u8,
            default_value_encoding: default_value_encoding as u8,
            n_vars: 0,
            nnz: 0,
            n_csr_shards: 0,
            n_csc_shards: 0,
            flags: ModalityFlags::empty(),
        };
        self.modalities.push(info);
        self.modality_build_csc.push(build_csc);
        #[cfg(feature = "deletion-vectors")]
        self.modality_bitmap_counts.push(0);
        Ok(self.modalities.len() as u8)
    }

    /// Resolve a registered modality name to its `modality_id`.
    /// Returns `None` for unknown names.
    pub fn modality_id(&self, name: &str) -> Option<u8> {
        self.modalities
            .iter()
            .position(|m| m.name == name)
            .map(|idx| (idx + 1) as u8)
    }

    /// The catalog entries written so far, and the header row count they
    /// describe.
    ///
    /// Exists so a caller can check *what it is about to persist* before
    /// [`Self::finish`] does. `finish` consumes `self` and ends with an atomic
    /// rename over the final path, so anything checked after it returns is a
    /// post-mortem: on an in-place rewrite (`scx optimize --output == input`,
    /// `scx build-csc` with no `<OUTPUT>`) the original is already gone by then
    /// and an error can only report the loss, not prevent it.
    ///
    /// **Two sections are written by `finish` itself and are therefore absent
    /// here**: the `ModalityTable`, and any CSC sidecar auto-emitted for a
    /// modality registered with `build_csc = true`. A caller that asserts on
    /// either must do so after `finish`, not through this.
    ///
    /// Read-only by design — mutating the entries from outside would let a
    /// caller desynchronise them from the offsets already written to disk.
    pub fn staged_catalog(&self) -> (&[FullCatalogEntry], u64) {
        (&self.entries, self.header.n_obs)
    }

    /// Number of modalities registered so far.
    pub fn n_modalities(&self) -> usize {
        self.modalities.len()
    }

    /// Run `body` with `current_modality_id` temporarily set to `id`.
    /// Public so external pipelines (e.g. the Phase 3 h5mu streaming
    /// coordinator) can stamp per-modality shards without going
    /// through the per-method `_for` wrappers.
    /// Used internally by every `*_for` method to stamp the right
    /// modality id on catalog entries created by `body`.
    pub fn with_modality<F, R, E>(&mut self, id: u8, body: F) -> std::result::Result<R, E>
    where
        E: From<ScxError>,
        F: FnOnce(&mut Self) -> std::result::Result<R, E>,
    {
        if id == 0 {
            return Err(ScxError::InvalidCatalog(
                "modality_id 0 is reserved for global entries".to_string(),
            )
            .into());
        }
        if id as usize > self.modalities.len() {
            return Err(ScxError::InvalidCatalog(format!(
                "modality_id {id} out of range (registered: {})",
                self.modalities.len()
            ))
            .into());
        }
        let prev = self.current_modality_id;
        self.current_modality_id = id;
        let result = body(self);
        self.current_modality_id = prev;
        result
    }

    /// Per-modality `write_var`. Catalog entry stamped with
    /// `modality_id`. The section name is `var/{modality_name}` to
    /// avoid colliding with the global "var" section (used for
    /// modality_id == 0 / single-modality files).
    pub fn write_var_for(&mut self, modality_id: u8, var: &RecordBatch) -> Result<()> {
        let name = self.var_section_name(modality_id)?;
        let data = Self::write_arrow_ipc(var)?;
        self.with_modality(modality_id, |this| {
            this.write_section_bytes(name, SectionType::VarMetadata, &data, None)
        })
    }

    fn var_section_name(&self, modality_id: u8) -> Result<String> {
        if modality_id == 0 {
            return Ok("var".to_string());
        }
        let idx = (modality_id - 1) as usize;
        let mname = self
            .modalities
            .get(idx)
            .map(|m| m.name.clone())
            .ok_or_else(|| {
                ScxError::InvalidCatalog(format!("modality_id {modality_id} not registered"))
            })?;
        Ok(format!("var/{mname}"))
    }

    /// Accumulate per-modality shard stats on the registered `ModalityInfo`
    /// for the current `with_modality` scope, flushed to the modality table at
    /// `finish()`. No-op when `current_modality_id == 0` (outside any
    /// `with_modality` scope — i.e. single-modality files, which carry no
    /// modality table); the `> 0` guard also avoids the `0 - 1` index
    /// underflow. CSC bumps the `has_csc` flag but not `nnz` (CSC mirrors the
    /// same entries as CSR — adding would double-count). Used by the
    /// pre-encoded (`write_preencoded_shard`) and verbatim-copy
    /// (`copy_section_verbatim`) paths. Only `CsrShard` / `CscShard` update the
    /// modality stats; any other section the verbatim-copy path may carry (e.g.
    /// `optimize` routing `BitmapShard` / `GroupIndex` / obsm/varm through
    /// `copy_section_verbatim`) falls through the `_ => {}` arm and is ignored.
    fn record_modality_shard(&mut self, section_type: SectionType, nnz: u64) {
        if self.current_modality_id == 0 {
            return;
        }
        if let Some(info) = self
            .modalities
            .get_mut((self.current_modality_id - 1) as usize)
        {
            match section_type {
                SectionType::CsrShard => {
                    info.n_csr_shards += 1;
                    info.nnz += nnz;
                }
                SectionType::CscShard => {
                    info.n_csc_shards += 1;
                    info.flags.set_csc();
                }
                _ => {}
            }
        }
    }

    /// Per-modality `write_csr_shard`. Section names are
    /// `X/{modality_name}/shard_{idx}` to avoid the global "X_shard_*"
    /// namespace. Shard counts accumulate on the registered
    /// `ModalityInfo` and are flushed to the modality table at
    /// `finish()` time.
    #[allow(clippy::too_many_arguments)]
    pub fn write_csr_shard_for(
        &mut self,
        modality_id: u8,
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        codec_id: CodecId,
        value_encoding: ValueEncoding,
        row_start: u64,
    ) -> Result<()> {
        let mname = self.modality_name_for(modality_id)?;
        let shard_idx = self
            .modalities
            .get((modality_id - 1) as usize)
            .map(|m| m.n_csr_shards)
            .unwrap_or(0);
        let name = format!("X/{mname}/shard_{shard_idx}");
        let nnz = *indptr.last().unwrap_or(&0);
        self.with_modality(modality_id, |this| {
            this.write_shard_inner(
                indptr,
                indices,
                values,
                codec_id,
                value_encoding,
                row_start,
                &name,
                SectionType::CsrShard,
            )
        })?;
        if let Some(info) = self.modalities.get_mut((modality_id - 1) as usize) {
            info.n_csr_shards += 1;
            info.nnz += nnz;
        }
        Ok(())
    }

    /// Per-modality `write_csc_shard`. Section names are
    /// `X_csc/{modality_name}/shard_{idx}`.
    #[allow(clippy::too_many_arguments)]
    pub fn write_csc_shard_for(
        &mut self,
        modality_id: u8,
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        codec_id: CodecId,
        value_encoding: ValueEncoding,
        col_start: u64,
    ) -> Result<()> {
        let mname = self.modality_name_for(modality_id)?;
        let shard_idx = self
            .modalities
            .get((modality_id - 1) as usize)
            .map(|m| m.n_csc_shards)
            .unwrap_or(0);
        let name = format!("X_csc/{mname}/shard_{shard_idx}");
        self.with_modality(modality_id, |this| {
            this.write_shard_inner(
                indptr,
                indices,
                values,
                codec_id,
                value_encoding,
                col_start,
                &name,
                SectionType::CscShard,
            )
        })?;
        // Phase 6: increment the file-wide CSC counter so `finish()`
        // sets the header `has_csc` flag and `n_csc_shards` field.
        // `write_shard_inner` itself does not touch these counters; the
        // single-modality `write_csc_shard` increments them after the
        // inner call, so mirror that here for the multimodal path.
        self.csc_shard_count += 1;
        if let Some(info) = self.modalities.get_mut((modality_id - 1) as usize) {
            info.n_csc_shards += 1;
            info.flags.set_csc();
        }
        Ok(())
    }

    /// Per-modality `write_layer_csr_shard`. Section name is
    /// `layer/{modality_name}/{layer_name}/shard_{idx}`.
    #[allow(clippy::too_many_arguments)]
    pub fn write_layer_csr_shard_for(
        &mut self,
        modality_id: u8,
        layer_name: &str,
        shard_idx: u32,
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        codec_id: CodecId,
        value_encoding: ValueEncoding,
        row_start: u64,
    ) -> Result<()> {
        let mname = self.modality_name_for(modality_id)?;
        let name = format!("layer/{mname}/{layer_name}/shard_{shard_idx}");
        self.with_modality(modality_id, |this| {
            this.write_shard_inner(
                indptr,
                indices,
                values,
                codec_id,
                value_encoding,
                row_start,
                &name,
                SectionType::LayerCsrShard,
            )
        })?;
        if let Some(info) = self.modalities.get_mut((modality_id - 1) as usize) {
            info.flags.set_layers();
        }
        Ok(())
    }

    /// Per-modality `write_layer_csc_shard`. Section name is
    /// `layer_csc/{modality_name}/{layer_name}/shard_{idx}`.
    /// Emits `SectionType::LayerCscShard` (id 16, new in v2).
    #[allow(clippy::too_many_arguments)]
    pub fn write_layer_csc_shard_for(
        &mut self,
        modality_id: u8,
        layer_name: &str,
        shard_idx: u32,
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        codec_id: CodecId,
        value_encoding: ValueEncoding,
        col_start: u64,
    ) -> Result<()> {
        let mname = self.modality_name_for(modality_id)?;
        let name = format!("layer_csc/{mname}/{layer_name}/shard_{shard_idx}");
        self.with_modality(modality_id, |this| {
            this.write_shard_inner(
                indptr,
                indices,
                values,
                codec_id,
                value_encoding,
                col_start,
                &name,
                SectionType::LayerCscShard,
            )
        })?;
        if let Some(info) = self.modalities.get_mut((modality_id - 1) as usize) {
            info.flags.set_layers();
            info.flags.set_csc();
        }
        Ok(())
    }

    /// Per-modality `write_obsm`. Section name is
    /// `obsm/{modality_name}/{key}`.
    pub fn write_obsm_for(
        &mut self,
        modality_id: u8,
        key: &str,
        batch: &RecordBatch,
    ) -> Result<()> {
        let mname = self.modality_name_for(modality_id)?;
        let name = format!("obsm/{mname}/{key}");
        let data = Self::write_arrow_ipc(batch)?;
        self.with_modality(modality_id, |this| {
            this.has_obsm = true;
            this.write_section_bytes(name, SectionType::ObsmEmbedding, &data, None)
        })?;
        if let Some(info) = self.modalities.get_mut((modality_id - 1) as usize) {
            info.flags.set_obsm();
        }
        Ok(())
    }

    /// Per-modality row-shard of an `obsm/{modality_name}/{key}` dense
    /// embedding. Section name is `obsm/{modality_name}/{key}_shard_{shard_idx}`
    /// with `SectionType::ObsmEmbeddingShard`. Mirrors the single-modality
    /// [`Self::write_obsm_shard`] but stamps the modality_id on the catalog
    /// entry via [`Self::with_modality`].
    #[allow(clippy::too_many_arguments)]
    pub fn write_obsm_shard_for(
        &mut self,
        modality_id: u8,
        key: &str,
        shard_idx: u32,
        row_start: u64,
        n_shard_rows: u64,
        n_rows_total: u64,
        batch: &RecordBatch,
    ) -> Result<()> {
        let mname = self.modality_name_for(modality_id)?;
        let stamped =
            stamp_dense_shard_metadata(batch, shard_idx, row_start, n_shard_rows, n_rows_total);
        let data = Self::write_arrow_ipc(&stamped)?;
        let name = format!("obsm/{mname}/{key}_shard_{shard_idx}");
        self.with_modality(modality_id, |this| {
            this.has_obsm = true;
            this.write_section_bytes(name, SectionType::ObsmEmbeddingShard, &data, None)
        })?;
        if let Some(info) = self.modalities.get_mut((modality_id - 1) as usize) {
            info.flags.set_obsm();
        }
        Ok(())
    }

    /// Per-modality row-shard of a `varm/{modality_name}/{key}` dense
    /// embedding. Section name is `varm/{modality_name}/{key}_shard_{shard_idx}`
    /// with `SectionType::VarmEmbeddingShard`. Mirrors the single-modality
    /// [`Self::write_varm_shard`].
    #[allow(clippy::too_many_arguments)]
    pub fn write_varm_shard_for(
        &mut self,
        modality_id: u8,
        key: &str,
        shard_idx: u32,
        row_start: u64,
        n_shard_rows: u64,
        n_rows_total: u64,
        batch: &RecordBatch,
    ) -> Result<()> {
        let mname = self.modality_name_for(modality_id)?;
        let stamped =
            stamp_dense_shard_metadata(batch, shard_idx, row_start, n_shard_rows, n_rows_total);
        let data = Self::write_arrow_ipc(&stamped)?;
        let name = format!("varm/{mname}/{key}_shard_{shard_idx}");
        self.with_modality(modality_id, |this| {
            this.write_section_bytes(name, SectionType::VarmEmbeddingShard, &data, None)
        })
    }

    /// Per-modality `write_obsp_shard`. Section name is
    /// `obsp/{modality_name}/{name}/shard_{shard_idx}`.
    ///
    /// Emits a [`SectionType::ObspCsrShard`] checked by `scx validate --deep`
    /// against the v3 canonical-CSR invariant; the caller MUST pass canonical
    /// CSR (see [`Self::write_obsp_shard`]). This writer does not canonicalize.
    #[allow(clippy::too_many_arguments)]
    pub fn write_obsp_shard_for(
        &mut self,
        modality_id: u8,
        obsp_name: &str,
        shard_idx: u32,
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        codec_id: CodecId,
        value_encoding: ValueEncoding,
        row_start: u64,
    ) -> Result<()> {
        let mname = self.modality_name_for(modality_id)?;
        let name = format!("obsp/{mname}/{obsp_name}/shard_{shard_idx}");
        self.with_modality(modality_id, |this| {
            this.has_obsp = true;
            this.write_shard_inner(
                indptr,
                indices,
                values,
                codec_id,
                value_encoding,
                row_start,
                &name,
                SectionType::ObspCsrShard,
            )
        })?;
        if let Some(info) = self.modalities.get_mut((modality_id - 1) as usize) {
            info.flags.set_obsp();
        }
        Ok(())
    }

    /// Per-modality `write_uns`. Section name is
    /// `uns/{modality_name}`.
    pub fn write_uns_for(&mut self, modality_id: u8, json: &serde_json::Value) -> Result<()> {
        let mname = self.modality_name_for(modality_id)?;
        let name = format!("uns/{mname}");
        scx_format::validate_uns_depth(json)?;
        let data = serde_json::to_vec(json)?;
        self.with_modality(modality_id, |this| {
            this.write_section_bytes(name, SectionType::UnsBlob, &data, None)
        })?;
        if let Some(info) = self.modalities.get_mut((modality_id - 1) as usize) {
            info.flags.set_uns();
        }
        Ok(())
    }

    /// Set the `n_vars` count for an already-registered modality.
    /// Required before `finish()` so the on-disk `ModalityTable`
    /// records the correct per-modality variable count.
    pub fn set_modality_n_vars(&mut self, modality_id: u8, n_vars: u64) -> Result<()> {
        let idx = (modality_id
            .checked_sub(1)
            .ok_or_else(|| ScxError::InvalidCatalog("modality_id must be >= 1".to_string()))?)
            as usize;
        let info = self.modalities.get_mut(idx).ok_or_else(|| {
            ScxError::InvalidCatalog(format!("modality_id {modality_id} not registered"))
        })?;
        info.n_vars = n_vars;
        Ok(())
    }

    fn modality_name_for(&self, modality_id: u8) -> Result<String> {
        if modality_id == 0 {
            return Err(ScxError::InvalidCatalog(
                "modality_id 0 is reserved for global entries".to_string(),
            ));
        }
        self.modalities
            .get((modality_id - 1) as usize)
            .map(|m| m.name.clone())
            .ok_or_else(|| {
                ScxError::InvalidCatalog(format!("modality_id {modality_id} not registered"))
            })
    }

    /// Phase B.3: streaming CSR→CSC transpose pass for modalities
    /// registered with `add_modality(..., build_csc=true)`.
    ///
    /// Reads each modality's CSR shards back from the writer's own
    /// temp file (decoded via `scx_codec::decode_shard_scipy`), runs
    /// `streaming_csr_to_csc_iter_with_cap`, and emits CSC sidecars
    /// via `write_csc_shard_for`. The latter increments
    /// `info.n_csc_shards` and sets `ModalityFlags::HAS_CSC` so the
    /// modality table emitted just after this pass records CSC
    /// presence correctly.
    ///
    /// Memory cap matches `scx-cli/src/build_csc.rs`'s 4 GiB default.
    /// CSC sharding granularity matches the CLI's
    /// `csc_cols_per_shard = 5000`.
    fn auto_emit_csc_for_marked_modalities(&mut self) -> Result<()> {
        const FINISH_TIME_CSC_MEMORY_BYTES: usize = 4 * 1024 * 1024 * 1024;
        const DEFAULT_CSC_COLS_PER_SHARD: usize = 5000;

        if self.modality_build_csc.iter().all(|&b| !b) {
            return Ok(());
        }

        let n_obs = self.header.n_obs as usize;

        // Collect per-modality work: (modality_id, csr_entries cloned, n_vars).
        // Cloning entries is cheap (Vec<u8> name + a few fields) and lets us
        // borrow `self` mutably for `write_csc_shard_for` calls below.
        let mut work: Vec<(u8, Vec<FullCatalogEntry>, usize)> = Vec::new();
        for (idx, build_csc) in self.modality_build_csc.iter().enumerate() {
            if !*build_csc {
                continue;
            }
            let modality_id = (idx + 1) as u8;
            let n_vars = self.modalities[idx].n_vars as usize;
            let csr_entries: Vec<FullCatalogEntry> = self
                .entries
                .iter()
                .filter(|e| e.section_type == SectionType::CsrShard && e.modality_id == modality_id)
                .cloned()
                .collect();
            if csr_entries.is_empty() {
                continue;
            }
            work.push((modality_id, csr_entries, n_vars));
        }
        if work.is_empty() {
            return Ok(());
        }

        // Flush BufWriter so all CSR shard bytes are durable in the
        // underlying File before we read them back.
        self.writer()?.flush()?;

        for (modality_id, csr_entries, n_vars) in work {
            // Decode each CSR shard back to ScxCsr by reading from the
            // open temp file. Mirrors the `read_shard_from_entry` flow
            // in `scx-format-io/src/reader/matrix.rs::read_shard_from_entry_inner`,
            // adapted to a `File` (no mmap).
            let mut csr_shards: Vec<scx_sparse::ScxCsr> = Vec::with_capacity(csr_entries.len());
            // The CSC sidecar encoding must cover EVERY shard, not just the
            // first (SCX-004): a Uint8 first shard followed by a Float32 shard
            // would truncate the float values. Scan all shards for the
            // float/integer kind; float ⇒ Float32 + Pcodec (float-safe codec),
            // else widen the integer width to fit the max value.
            let mut any_float = false;
            let mut max_int_val: u32 = 0;
            let mut first_codec: Option<CodecId> = None;
            // Floor the integer width on each shard's declared encoding too, so
            // a wide integer shard that lacks stats is not under-picked as
            // Uint8 (see build_csc.rs for the same guard).
            let mut header_int_enc = ValueEncoding::Uint8;
            let enc_width = |e: ValueEncoding| match e {
                ValueEncoding::Uint8 => 1u8,
                ValueEncoding::Uint16 => 2,
                _ => 4,
            };
            for entry in &csr_entries {
                let (sh, indptr, indices, data) = self.decode_csr_entry(entry, n_vars)?;
                let enc = ValueEncoding::from_u8(sh.value_encoding)
                    .ok_or(ScxError::UnknownValueEncoding(sh.value_encoding))?;
                if matches!(enc, ValueEncoding::Float32 | ValueEncoding::Float16) {
                    any_float = true;
                } else if enc_width(enc) > enc_width(header_int_enc) {
                    header_int_enc = enc;
                }
                if first_codec.is_none() {
                    first_codec = Some(
                        CodecId::from_u8(sh.codec_id).ok_or(ScxError::UnknownCodec(sh.codec_id))?,
                    );
                }
                if let Some(stats) = entry.stats.as_ref() {
                    max_int_val = max_int_val.max(stats.value_max);
                }
                let n_shard_rows = indptr.len() - 1;
                csr_shards.push(scx_sparse::ScxCsr::new_unchecked(
                    (n_shard_rows, n_vars),
                    indptr,
                    indices,
                    data,
                ));
            }

            let (value_encoding, codec) = if any_float {
                (ValueEncoding::Float32, CodecId::Pcodec)
            } else {
                let by_value = if max_int_val <= u8::MAX as u32 {
                    ValueEncoding::Uint8
                } else if max_int_val <= u16::MAX as u32 {
                    ValueEncoding::Uint16
                } else {
                    ValueEncoding::Uint32
                };
                let enc = if enc_width(header_int_enc) > enc_width(by_value) {
                    header_int_enc
                } else {
                    by_value
                };
                (enc, first_codec.expect("at least one CSR entry processed"))
            };

            // Streaming CSR→CSC transpose. Mirrors
            // `scx-cli/src/build_csc.rs:159`.
            let mut iter = scx_sparse::streaming_csr_to_csc_iter_with_cap(
                &csr_shards,
                n_obs,
                n_vars,
                FINISH_TIME_CSC_MEMORY_BYTES,
                DEFAULT_CSC_COLS_PER_SHARD,
            )
            .map_err(|e| {
                ScxError::InvalidCatalog(format!("auto_emit_csc transpose failed: {e}"))
            })?;

            loop {
                let col_start = iter.current_col_start() as u64;
                let chunk = match iter.next() {
                    Some(c) => c.map_err(|e| {
                        ScxError::InvalidCatalog(format!("auto_emit_csc chunk decode failed: {e}"))
                    })?,
                    None => break,
                };
                let csc_indptr_u64: Vec<u64> = chunk.indptr.iter().map(|&v| v as u64).collect();
                let csc_indices_u32: Vec<u32> = chunk.indices.iter().map(|&i| i as u32).collect();
                let csc_raw_values = value_encoding.encode_f32_batch(&chunk.data).map_err(|e| {
                    ScxError::InvalidCatalog(format!("auto_emit_csc encode_f32_batch failed: {e}"))
                })?;

                self.write_csc_shard_for(
                    modality_id,
                    &csc_indptr_u64,
                    &csc_indices_u32,
                    &csc_raw_values,
                    codec,
                    value_encoding,
                    col_start,
                )?;
            }
        }
        // `csc_build_generation` is recorded inside `write_shard_inner`
        // for every CSC shard emitted above (single chokepoint), so no
        // explicit stamp is needed here.
        Ok(())
    }

    /// Phase B.3 helper: read a CSR catalog entry back from the
    /// writer's temp file and decode it via `scx_codec::decode_shard_scipy`.
    /// Returns the parsed `ShardHeader` plus the scipy-shape
    /// `(indptr, indices, data)` triple.
    #[allow(clippy::type_complexity)]
    fn decode_csr_entry(
        &mut self,
        entry: &FullCatalogEntry,
        _n_vars: usize,
    ) -> Result<(ShardHeader, Vec<i64>, Vec<i32>, Vec<f32>)> {
        use std::io::Cursor;

        // Flush any pending BufWriter bytes before seeking the inner
        // file: BufWriter calls `write_all` on its inner File at the
        // file's current cursor, so leaving buffered data while we
        // seek would interleave when the buffer next flushes.
        let writer = self.writer()?;
        writer.flush()?;
        let file = writer.get_mut();
        file.seek(SeekFrom::Start(entry.offset))?;
        let mut section = vec![0u8; entry.length as usize];
        file.read_exact(&mut section)?;

        let vs = crate::validated_section::ValidatedSection::new(&section);
        let sh = ShardHeader::read_from(&mut Cursor::new(vs.header()?))?;

        let indptr_bytes = vs.subslice(sh.indptr_rel_offset, sh.indptr_length)?;
        let indices_bytes = vs.subslice(sh.indices_rel_offset, sh.indices_length)?;
        let values_bytes = vs.subslice(sh.values_rel_offset, sh.values_length)?;

        let codec_id = CodecId::from_u8(sh.codec_id).ok_or(ScxError::UnknownCodec(sh.codec_id))?;
        let value_encoding = ValueEncoding::from_u8(sh.value_encoding)
            .ok_or(ScxError::UnknownValueEncoding(sh.value_encoding))?;
        let index_dtype_u16 = sh.index_dtype == 0;

        let encoded = scx_codec::EncodedShardRef {
            indptr_bytes,
            indices_bytes,
            values_bytes,
        };

        let (indptr, indices, data) = scx_codec::decode_shard_scipy(
            &encoded,
            codec_id,
            value_encoding,
            sh.n_major as usize,
            sh.nnz as usize,
            index_dtype_u16,
            // This reads back a shard *this writer* just wrote, so the bound is
            // a self-check rather than a defence against a hostile file — but it
            // costs nothing and catches an encode bug at the point it happens.
            scx_codec::clamp_index_bound(sh.n_minor),
        )
        // `ScxError::from`, not `ScxError::Codec`: the latter constructs the
        // variant directly and bypasses the promoting `From`, so an
        // `IndexOutOfRange` from the bound above would stay `Codec` here while
        // every other seam reports `ShardIndexOutOfRange`.
        .map_err(ScxError::from)?;

        // Restore the file cursor to EOF so subsequent
        // `write_csc_shard_for` calls append from the right position.
        // The BufWriter wraps the same underlying file; seek the inner
        // File explicitly.
        let writer = self.writer()?;
        let file = writer.get_mut();
        file.seek(SeekFrom::End(0))?;
        Ok((sh, indptr, indices, data))
    }

    /// Finalize the file: write catalogs, header, fsync, atomic rename.
    ///
    /// Returns the final file path on success.
    pub fn finish(mut self) -> Result<PathBuf> {
        // Phase B.3: auto-emit CSC sidecars for any modality registered
        // with `build_csc=true`. Runs before the modality table is
        // serialised so the table picks up the resulting
        // `n_csc_shards` / `flags.has_csc()` state.
        self.auto_emit_csc_for_marked_modalities()?;

        // Flush BufWriter and take inner File
        let buf_writer = self.file.take().ok_or(ScxError::WriterAlreadyFinished)?;
        let mut file = buf_writer.into_inner().map_err(std::io::Error::from)?;

        // 1a. Emit the ModalityTable section (Phase B) if any
        // modalities were registered. The table is just a normal
        // section: it has a catalog entry, lives between the last
        // regular section and the full catalog, and gets covered by
        // the file checksum like everything else. The header records
        // its offset/length and bit-7 capability flag.
        let (modality_table_offset, modality_table_length) = if self.modalities.is_empty() {
            (0u64, 0u64)
        } else {
            let aligned = align_to_8(self.current_offset);
            let pad = (aligned - self.current_offset) as usize;
            if pad > 0 {
                file.write_all(&vec![0u8; pad])?;
            }
            self.current_offset = aligned;

            let table = ModalityTable::new(self.modalities.clone());
            let mut table_buf = Vec::new();
            table.write_to(&mut table_buf)?;
            let mt_offset = self.current_offset;
            let mt_length = table_buf.len() as u64;
            let mt_checksum = blake3_hash(&table_buf);
            file.write_all(&table_buf)?;
            self.current_offset += mt_length;

            self.entries.push(FullCatalogEntry {
                name: "modality_table".to_string(),
                offset: mt_offset,
                length: mt_length,
                section_type: SectionType::ModalityTable,
                checksum: mt_checksum,
                modality_id: 0, // global section
                stats: None,
            });

            self.header.n_modalities = self.modalities.len() as u32;
            self.header.modality_table_offset = mt_offset;
            self.header.modality_table_length = mt_length;
            self.header.set_modalities();

            (mt_offset, mt_length)
        };
        let _ = (modality_table_offset, modality_table_length); // header already set

        // 1. Write full catalog at EOF
        let aligned_offset = align_to_8(self.current_offset);
        let pad = (aligned_offset - self.current_offset) as usize;
        if pad > 0 {
            file.write_all(&vec![0u8; pad])?;
        }

        let full_catalog_offset = aligned_offset;

        let full_catalog = FullCatalog {
            catalog_version: crate::catalog::CURRENT_CATALOG_VERSION,
            manifest_sequence: self.header.manifest_sequence,
            prev_catalog_offset: 0,
            n_obs: self.header.n_obs,
            entries: self.entries.clone(),
            data_generation: self.data_generation,
            // `auto_emit_csc_for_marked_modalities` sets this to
            // `Some(data_generation)` when it emits a sidecar; absent a
            // sidecar it stays `None` → recorded as `0`.
            csc_build_generation: self.csc_build_generation.unwrap_or(0),
        };
        let mut catalog_buf = Vec::new();
        full_catalog.write_to(&mut catalog_buf)?;
        file.write_all(&catalog_buf)?;
        let full_catalog_length = catalog_buf.len() as u64;

        // 2. Build root catalog from entries
        let mut groups: BTreeMap<u8, Vec<&FullCatalogEntry>> = BTreeMap::new();
        for entry in &self.entries {
            groups
                .entry(entry.section_type as u8)
                .or_default()
                .push(entry);
        }

        let mut root_entries = Vec::new();
        for (&group_type, entries) in &groups {
            let first_offset = entries.iter().map(|e| e.offset).min().unwrap_or(0);
            // Compute span from first section start to end of last section (includes padding)
            let last_end = entries
                .iter()
                .map(|e| e.offset + e.length)
                .max()
                .unwrap_or(0);
            let total_length: u64 = last_end.saturating_sub(first_offset);
            let n_sections = entries.len() as u32;
            root_entries.push(RootCatalogEntry {
                group_type,
                first_section_offset: first_offset,
                total_group_length: total_length,
                n_sections,
                summary: [0u8; 32],
            });
        }

        let root_catalog = RootCatalog {
            n_section_groups: root_entries.len() as u16,
            entries: root_entries,
        };
        let mut root_buf = Vec::new();
        root_catalog.write_to(&mut root_buf)?;
        let root_catalog_length = root_buf.len() as u64;
        // Pad to 4096 bytes
        root_buf.resize(4096, 0);

        // 3. Update header fields (checksum=0 placeholder for hashing)
        self.header.root_catalog_offset = HEADER_SIZE as u64;
        self.header.root_catalog_length = root_catalog_length;
        self.header.full_catalog_offset = full_catalog_offset;
        self.header.full_catalog_length = full_catalog_length;
        // Derive n_csr/n_csc/nnz + the CSC/obsm/obsp/bitmap/DV flags
        // from the catalog through the single source of truth shared
        // with `scx_ops::rollback` (abstraction 3 / OE1). Byte-identical
        // to the previous hand-rolled accumulators: `n_csr_shards` /
        // `n_csc_shards` count the same section types, `nnz` sums the
        // CsrShard `stats.nnz` (== each shard's `*indptr.last()`), and
        // each flag is presence-derived exactly as before.
        self.header.sync_from_catalog(&full_catalog);
        self.header.file_checksum = 0;

        // 4. Compute file checksum: hash header + root catalog from memory,
        //    re-read only section bytes from file, hash full catalog from memory.
        //    This avoids re-reading the entire file and writes the header only once.
        let mut header_bytes = Vec::with_capacity(HEADER_SIZE);
        self.header.write_to(&mut header_bytes)?;

        let mut final_hasher = blake3::Hasher::new();
        final_hasher.update(&header_bytes); // 256 bytes from memory
        final_hasher.update(&root_buf); // 4096 bytes (padded) from memory

        // Re-read section bytes (offset 4352 to full_catalog_offset) from file
        file.flush()?;
        file.seek(SeekFrom::Start(SECTIONS_START_OFFSET))?;
        let mut remaining = (full_catalog_offset - SECTIONS_START_OFFSET) as usize;
        let mut chunk = [0u8; 65536];
        while remaining > 0 {
            use std::io::Read;
            let to_read = remaining.min(chunk.len());
            file.read_exact(&mut chunk[..to_read])?;
            final_hasher.update(&chunk[..to_read]);
            remaining -= to_read;
        }

        final_hasher.update(&catalog_buf); // full catalog from memory

        let file_hash = final_hasher.finalize();
        let file_checksum = crate::checksum::truncate_hash_to_u64(&file_hash);

        // 5. Write header with correct checksum (written only once)
        self.header.file_checksum = file_checksum;
        file.seek(SeekFrom::Start(0))?;
        self.header.write_to(&mut file)?;

        // 6. Write root catalog at offset 256
        file.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
        file.write_all(&root_buf)?;

        // 7. fsync
        file.sync_all()?;
        drop(file);

        // 8. Atomic rename via TempPath::persist
        let tmp_path = self
            .tmp_path
            .take()
            .ok_or(ScxError::WriterAlreadyFinished)?;
        tmp_path
            .persist(&self.final_path)
            .map_err(|e| ScxError::Io(e.error))?;

        // 9. Restore umask-respecting permissions on the persisted file.
        //    `tempfile::NamedTempFile` always creates files with `0600`;
        //    this widens to `0o666 & !umask` so SCX outputs in shared
        //    directories remain group/world readable.  No-op on non-Unix.
        chmod_to_umask(&self.final_path)?;

        // 10. fsync the parent directory so the new directory entry is
        //     durable on POSIX.  Without this, a power loss after rename
        //     can lose the directory entry even though the file data is
        //     intact.  No-op on non-Unix platforms.
        fsync_parent_dir(&self.final_path)?;

        Ok(self.final_path.clone())
    }

    /// Write the obs predicate index section (pre-serialized bytes).
    pub fn write_obs_predicate_index(&mut self, data: &[u8]) -> Result<()> {
        self.write_section_bytes(
            "obs_predicate_index",
            SectionType::ObsPredicateIndex,
            data,
            None,
        )
    }

    /// Write the var predicate index section (pre-serialized bytes).
    pub fn write_var_predicate_index(&mut self, data: &[u8]) -> Result<()> {
        self.write_section_bytes(
            "var_predicate_index",
            SectionType::VarPredicateIndex,
            data,
            None,
        )
    }

    /// F1: write the condition/label-grouped sharding sidecar
    /// ([`SectionType::GroupIndex`], section name `group_index`). `data` is the
    /// pre-serialized JSON payload
    /// `{group_by, reference_shard, reference_labels, records[]}`. One per file;
    /// written after the last CSR shard flush by the grouped sort path.
    pub fn write_group_index(&mut self, data: &[u8]) -> Result<()> {
        self.write_section_bytes("group_index", SectionType::GroupIndex, data, None)
    }

    /// Phase 5b: write a detection-bitmap shard.
    ///
    /// Modality-aware: if [`Self::with_modality`] has set
    /// `current_modality_id` to a non-zero value, the section name is
    /// `X/bitmap/{modality_name}/shard_{idx}` and the per-modality
    /// counter is incremented (`ModalityFlags::HAS_BITMAP` is flipped).
    /// Otherwise the unimodal naming `X/bitmap/shard_{idx}` is used
    /// and the global counter is incremented.
    ///
    /// Auto-derives the next shard index from the writer's own
    /// counter (mirrors `write_csr_shard`); callers don't need to
    /// track it.
    #[cfg(feature = "deletion-vectors")]
    pub fn write_bitmap_shard(&mut self, shard: &crate::bitmap::BitmapShard) -> Result<()> {
        if self.current_modality_id > 0 {
            // We're inside a `with_modality(id, ...)` scope — defer to
            // the per-modality path so the section name + counter match
            // the multimodal naming convention.
            let modality_id = self.current_modality_id;
            return self.write_bitmap_shard_for_inner(modality_id, shard);
        }
        let shard_idx = self.bitmap_shard_count;
        let name = format!("X/bitmap/shard_{shard_idx}");
        let mut data = Vec::new();
        shard.write_to(&mut data)?;
        let stats = ShardStats {
            row_start: shard.row_start,
            row_end: shard.row_start + shard.n_rows as u64,
            col_start: 0,
            col_end: shard.n_vars as u64,
            nnz: 0,
            value_min: 0,
            value_max: 0,
            value_sum: 0,
            n_indexed_columns: 0,
            column_stats: Vec::new(),
        };
        self.write_section_bytes(name, SectionType::BitmapShard, &data, Some(stats))?;
        self.bitmap_shard_count += 1;
        Ok(())
    }

    /// Phase 5b: per-modality bitmap shard. Section name
    /// `X/bitmap/{modality_name}/shard_{idx}`.
    ///
    /// Auto-derives the per-modality shard index from
    /// `modality_bitmap_counts`. Also flips
    /// [`ModalityFlags::HAS_BITMAP`] on the modality's flags.
    ///
    /// Wraps the body in [`Self::with_modality`] so the catalog entry
    /// is stamped with the right `modality_id`. Callers that are
    /// already inside `with_modality` should call [`Self::write_bitmap_shard`]
    /// instead — it dispatches to the same per-modality path
    /// automatically via `current_modality_id`.
    #[cfg(feature = "deletion-vectors")]
    pub fn write_bitmap_shard_for(
        &mut self,
        modality_id: u8,
        shard: &crate::bitmap::BitmapShard,
    ) -> Result<()> {
        // Re-enter `with_modality` even if we're already in scope:
        // it stacks correctly (saves/restores `current_modality_id`).
        self.with_modality(modality_id, |this| {
            this.write_bitmap_shard_for_inner(modality_id, shard)
        })
    }

    /// Internal: the actual write. Assumes `current_modality_id` is
    /// already set to `modality_id` (either via [`Self::with_modality`]
    /// wrapper above or via the dispatcher in [`Self::write_bitmap_shard`]).
    #[cfg(feature = "deletion-vectors")]
    fn write_bitmap_shard_for_inner(
        &mut self,
        modality_id: u8,
        shard: &crate::bitmap::BitmapShard,
    ) -> Result<()> {
        let mname = self.modality_name_for(modality_id)?;
        let idx = (modality_id as usize)
            .checked_sub(1)
            .ok_or_else(|| ScxError::InvalidCatalog("modality_id must be >= 1".to_string()))?;
        let shard_idx = self.modality_bitmap_counts.get(idx).copied().unwrap_or(0);
        let name = format!("X/bitmap/{mname}/shard_{shard_idx}");
        let mut data = Vec::new();
        shard.write_to(&mut data)?;
        let stats = ShardStats {
            row_start: shard.row_start,
            row_end: shard.row_start + shard.n_rows as u64,
            col_start: 0,
            col_end: shard.n_vars as u64,
            nnz: 0,
            value_min: 0,
            value_max: 0,
            value_sum: 0,
            n_indexed_columns: 0,
            column_stats: Vec::new(),
        };
        self.write_section_bytes(name, SectionType::BitmapShard, &data, Some(stats))?;
        if let Some(slot) = self.modality_bitmap_counts.get_mut(idx) {
            *slot += 1;
        }
        if let Some(info) = self.modalities.get_mut(idx) {
            info.flags.set_bitmap();
        }
        Ok(())
    }

    /// Write the deletion vectors section.
    ///
    /// Automatically sets the deletion-vectors flag (bit 5) in the file header
    /// so that `ScxReader::read_deletion_vectors()` will find the section.
    #[cfg(feature = "deletion-vectors")]
    pub fn write_deletion_vectors(
        &mut self,
        dv: &crate::deletion_vectors::DeletionVectors,
    ) -> Result<()> {
        self.header.set_deletion_vectors();
        let mut data = Vec::new();
        dv.write_to(&mut data)?;
        self.write_section_bytes(
            "deletion_vectors",
            SectionType::DeletionVectors,
            &data,
            None,
        )
    }
}

impl Drop for ScxWriter {
    fn drop(&mut self) {
        // Close file handle first so the temp file is not held open.
        drop(self.file.take());
        // TempPath::drop auto-deletes the temp file if `persist` was never
        // called (i.e., `finish()` was not reached).  After a successful
        // `finish()`, `tmp_path` is `None` so this is a no-op.
        drop(self.tmp_path.take());
    }
}

/// Fsync the parent directory of `path` so the directory entry is durable.
///
/// On POSIX, `rename()` is atomic but the directory entry may not survive
/// a power loss unless the directory itself is fsynced.  This function
/// opens the parent directory and calls `sync_all()` on Unix; on
/// non-Unix platforms it is a no-op.
pub fn fsync_parent_dir(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let dir = File::open(parent)?;
        dir.sync_all()?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

/// Create a collision-safe sibling temp file for an atomic write to
/// `final_path`. Names the temp file `.{stem}_<random>.tmp` in the same
/// directory as `final_path`, so the eventual `rename` is intra-filesystem
/// and therefore atomic. The returned `TempPath` auto-deletes the file on
/// drop if `persist` is never called, so an interrupted write leaves no
/// orphan beyond the lifetime of the writer.
///
/// Used by `ScxWriter` and by the `scx-cloud` `pull` / `pull_filtered` /
/// `pack` / `cloud_optimize` paths so all five sites share one naming
/// policy.
pub fn make_sibling_tempfile(final_path: &Path) -> Result<(File, tempfile::TempPath)> {
    let parent = final_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let stem = final_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("scx");
    let named_tmp = tempfile::Builder::new()
        .prefix(&format!(".{stem}_"))
        .suffix(".tmp")
        .tempfile_in(parent)
        .map_err(ScxError::Io)?;
    Ok(named_tmp.into_parts())
}

/// Re-apply the process umask to `path` so the persisted file ends up
/// with `0o666 & !umask` — matching the permissions a plain
/// `OpenOptions::create()` would have produced. `tempfile::NamedTempFile`
/// always creates files with `0600` for security; on shared filesystems
/// (e.g. HPC group dirs) this is too restrictive, so atomic-write paths
/// call this immediately after `persist()` to restore the conventional
/// umask-driven mode.
///
/// No-op on non-Unix platforms (Windows permission model is unrelated).
#[cfg(unix)]
pub fn chmod_to_umask(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mode = 0o666 & !current_umask();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))?;
    Ok(())
}

#[cfg(not(unix))]
pub fn chmod_to_umask(_path: &Path) -> Result<()> {
    Ok(())
}

/// Read the process umask once, via the POSIX `umask(0)` / `umask(saved)`
/// dance, and cache it in a `OnceLock`. POSIX provides no race-free way
/// to read umask without temporarily clearing it; caching on first call
/// keeps the window to a single brief interval at process startup rather
/// than reopening it on every write.
#[cfg(unix)]
fn current_umask() -> u32 {
    use std::sync::OnceLock;
    static UMASK: OnceLock<u32> = OnceLock::new();
    *UMASK.get_or_init(|| {
        // SAFETY: `umask` is async-signal-safe and the value we pass
        // (`0o022`) is a no-op placeholder we immediately overwrite with
        // the saved value. Racy with concurrent `umask` callers but the
        // worst case is a one-time misread on the first invocation; the
        // cached value is stable thereafter.
        unsafe {
            let saved = libc::umask(0o022);
            libc::umask(saved);
            saved as u32
        }
    })
}

/// Major axis of a shard: row-major (CSR/Layer/Obsp) or column-major
/// (CSC). Tells `compute_shard_stats` which pair (`row_*` or `col_*`)
/// carries the shard's primary index range; the other pair is filled
/// with the full extent of the unbound axis (`n_minor`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MajorAxis {
    /// Row-major shard. The `row_start`/`row_end` pair carries the
    /// primary range; `col_start`/`col_end` cover `[0, n_minor)`.
    Row,
    /// Column-major shard. The `col_start`/`col_end` pair carries the
    /// primary range; `row_start`/`row_end` cover `[0, n_minor)`.
    Col,
}

/// Attach per-shard column statistics to the modality-0 CSR shard entries in
/// `entries`, addressing them by `csr_shards_sorted` position.
///
/// `per_shard[k]` is the `Vec<ColumnStat>` for the k-th modality-0 CSR shard in
/// `row_start` order — the same `shard_id` space an obs predicate index is
/// built against (`shard_row_ranges` index == catalog CSR-shard sorted
/// position). Shared by [`ScxWriter::set_csr_shard_column_stats_bulk`] (on the
/// writer's own entries) and `scx-ops::append` (on its assembled catalog
/// entries, which include both pre-existing and freshly-appended CSR shards).
///
/// `per_shard.len()` must equal the number of modality-0 CSR shard entries;
/// returns [`ScxError::ColumnStatsShardCountMismatch`] otherwise.
///
/// Precondition: every modality-0 CSR shard entry must carry `Some(stats)`. The
/// by-position mapping relies on sorting those entries by `major_start`; an entry
/// with `stats == None` sorts to the end (`u64::MAX`) and has its stats dropped,
/// either of which would silently misalign the mapping. Trips a `debug_assert`.
pub fn assign_csr_shard_column_stats(
    entries: &mut [FullCatalogEntry],
    per_shard: Vec<Vec<crate::catalog::ColumnStat>>,
) -> Result<()> {
    let mut shard_entry_indices: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.section_type == SectionType::CsrShard && e.modality_id == 0)
        .map(|(i, _)| i)
        .collect();
    if per_shard.len() != shard_entry_indices.len() {
        return Err(ScxError::ColumnStatsShardCountMismatch {
            got: per_shard.len(),
            expected: shard_entry_indices.len(),
        });
    }
    shard_entry_indices.sort_by_key(|&i| {
        entries[i]
            .stats
            .as_ref()
            .map_or(u64::MAX, |s| s.major_start(SectionType::CsrShard))
    });
    for (column_stats, &entry_idx) in per_shard.into_iter().zip(shard_entry_indices.iter()) {
        if column_stats.len() > u8::MAX as usize {
            return Err(ScxError::ColumnStatsOverflow(column_stats.len()));
        }
        debug_assert!(
            entries[entry_idx].stats.is_some(),
            "CSR shard entry missing stats — by-position column-stats mapping would misalign"
        );
        if let Some(ref mut stats) = entries[entry_idx].stats {
            stats.n_indexed_columns = column_stats.len() as u8;
            stats.column_stats = column_stats;
        }
    }
    Ok(())
}

/// Drop the per-shard column statistics for **every** indexed obs column from
/// the CSR shard entries in `entries`. Returns the number of entries changed.
///
/// The counterpart to [`assign_csr_shard_column_stats`], and the invariant they
/// jointly maintain: a `ColumnStat` must never outlive the obs values it
/// describes. `scx_engine`'s Level-1 pushdown prunes a shard straight from
/// `stats.column_stats` and consults the `ObsPredicateIndex` only for the
/// *categorical* arm's value dictionary — so an in-place op that replaces obs
/// and merely drops the index section leaves the numeric `MinMax` bounds live
/// and authoritative. Every shard is then excluded on a predicate the new values
/// satisfy, and the query returns a short row set with no error.
///
/// Clearing costs Level-1 pruning (a slower query) until an index rebuild
/// re-derives the stats; keeping a stale one costs rows. Use
/// [`clear_csr_shard_column_stats_for`] when only some columns are rewritten.
///
/// Deliberately **broader** than `assign_csr_shard_column_stats`, which addresses
/// modality-0 entries only: this spans every modality. Clearing is a pure safety
/// operation with no case where retaining a stale stat is preferable, and
/// nothing writes CSR column stats for a non-zero modality today (every writer
/// routes through `assign_csr_shard_column_stats`), so the wider scope is a
/// no-op now and correct the day multimodal indexing ships.
pub fn clear_all_csr_shard_column_stats(entries: &mut [FullCatalogEntry]) -> usize {
    clear_csr_shard_column_stats_inner(entries, None)
}

/// Drop the per-shard column statistics for `columns` only, leaving every other
/// indexed column's stats intact. Returns the number of entries changed.
///
/// For ops that rewrite *named* obs columns by key join — `obs_import`,
/// `doublet_import`, `cellbender_import` — and never reorder rows: an untouched
/// column's `MinMax` / `CategoryBitset` is still true of the file, and clearing
/// it would silently disable Level-1 pruning on, say, a headline `cell_type`
/// index every time someone lands doublet calls on an atlas.
///
/// Matching is by [`crate::catalog::column_name_hash`], the same hash
/// `scx_engine::derive_shard_column_stats` writes into each stat. Naming a
/// column the file has no stats for is a no-op, so callers may pass the whole
/// planned-column list unconditionally — in particular **without** first
/// checking whether an `ObsPredicateIndex` exists. It need not: a file whose
/// index was already dropped by an earlier op can still be carrying that op's
/// stats, and gating on the index is what lets those survive a second rewrite.
pub fn clear_csr_shard_column_stats_for(
    entries: &mut [FullCatalogEntry],
    columns: &[String],
) -> usize {
    if columns.is_empty() {
        return 0;
    }
    let hashes: Vec<u64> = columns
        .iter()
        .map(|name| crate::catalog::column_name_hash(name))
        .collect();
    clear_csr_shard_column_stats_inner(entries, Some(&hashes))
}

/// `None` clears every column; `Some(hashes)` clears only those.
///
/// `n_indexed_columns` is rewritten from `column_stats.len()` rather than
/// decremented: [`ShardStats::read_from`] loops exactly `n_indexed_columns`
/// times over the stats payload, so a count that disagrees with the vector
/// desynchronises the decode — it stops early or runs into the next field.
///
/// The `as u8` cannot truncate, and deliberately does not become a `try_from`
/// that panics — this crate returns errors rather than panicking, and there is
/// no error to return. `clear`/`retain` only ever *shrink* the vector, and the
/// pre-clear length is already ≤ `u8::MAX` from both of its only two producers:
/// [`ShardStats::read_from`] pushes exactly the `u8` count it read off the wire,
/// and [`assign_csr_shard_column_stats`] rejects anything longer with
/// [`ScxError::ColumnStatsOverflow`] (as does `ShardStats::write_to`, so an
/// over-long vector could never have been serialised in the first place). The
/// `debug_assert` pins that reasoning where it is used.
fn clear_csr_shard_column_stats_inner(
    entries: &mut [FullCatalogEntry],
    hashes: Option<&[u64]>,
) -> usize {
    let mut changed = 0usize;
    for entry in entries.iter_mut() {
        if entry.section_type != SectionType::CsrShard {
            continue;
        }
        let Some(stats) = entry.stats.as_mut() else {
            continue;
        };
        if stats.column_stats.is_empty() {
            continue;
        }
        let before = stats.column_stats.len();
        match hashes {
            None => stats.column_stats.clear(),
            Some(h) => stats
                .column_stats
                .retain(|cs| !h.contains(&cs.column_name_hash())),
        }
        if stats.column_stats.len() != before {
            changed += 1;
        }
        debug_assert!(
            stats.column_stats.len() <= u8::MAX as usize,
            "column_stats longer than the u8 wire count could never have been \
             written or read — see the note on this function"
        );
        stats.n_indexed_columns = stats.column_stats.len() as u8;
    }
    changed
}

/// Compute shard statistics from raw value bytes.
///
/// `major_kind` distinguishes row-major (CSR/Layer/Obsp) and
/// column-major (CSC) shards. `major_start` is the global index where
/// this shard begins on its primary axis; `n_major` is the count of
/// major-axis entries in the shard. `n_minor` is the count of entries
/// on the OTHER axis (file-wide `n_vars` for row-major shards or
/// file-wide `n_obs` for column-major shards) — used to populate the
/// "full range" pair for v2 symmetry.
pub fn compute_shard_stats(
    values: &[u8],
    value_encoding: ValueEncoding,
    major_kind: MajorAxis,
    major_start: u64,
    n_major: u64,
    n_minor: u64,
    nnz: u64,
) -> ShardStats {
    let (value_min, value_max, value_sum) = match value_encoding {
        ValueEncoding::Uint8 => {
            let mut min = u32::MAX;
            let mut max = 0u32;
            let mut sum = 0u64;
            for &b in values {
                let v = b as u32;
                min = min.min(v);
                max = max.max(v);
                sum += v as u64;
            }
            if values.is_empty() {
                (0, 0, 0)
            } else {
                (min, max, sum)
            }
        }
        ValueEncoding::Uint16 => {
            let mut min = u32::MAX;
            let mut max = 0u32;
            let mut sum = 0u64;
            let mut cursor = std::io::Cursor::new(values);
            while let Ok(v) = cursor.read_u16::<LittleEndian>() {
                let v = v as u32;
                min = min.min(v);
                max = max.max(v);
                sum += v as u64;
            }
            if values.is_empty() {
                (0, 0, 0)
            } else {
                (min, max, sum)
            }
        }
        ValueEncoding::Uint32 => {
            let mut min = u32::MAX;
            let mut max = 0u32;
            let mut sum = 0u64;
            let mut cursor = std::io::Cursor::new(values);
            while let Ok(v) = cursor.read_u32::<LittleEndian>() {
                min = min.min(v);
                max = max.max(v);
                sum += v as u64;
            }
            if values.is_empty() {
                (0, 0, 0)
            } else {
                (min, max, sum)
            }
        }
        // Float types: value_min/value_max are u32 and value_sum is u64, which
        // cannot represent float statistics. Return zeros; these fields are
        // documented as undefined for float value encodings (see docs/format.md (Dual Catalog)).
        ValueEncoding::Float32 | ValueEncoding::Float16 => (0, 0, 0),
    };

    let (row_start, row_end, col_start, col_end) = match major_kind {
        MajorAxis::Row => (major_start, major_start + n_major, 0, n_minor),
        MajorAxis::Col => (0, n_minor, major_start, major_start + n_major),
    };

    ShardStats {
        row_start,
        row_end,
        col_start,
        col_end,
        nnz,
        value_min,
        value_max,
        value_sum,
        n_indexed_columns: 0,
        column_stats: vec![],
    }
}

#[cfg(test)]
#[path = "writer_tests.rs"]
mod tests;
