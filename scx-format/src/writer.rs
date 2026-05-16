// ScxWriter — atomic rename path (docs/architecture.md)

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use arrow::array::RecordBatch;
use byteorder::{LittleEndian, ReadBytesExt};
use scx_codec::{CodecId, ValueEncoding};

use crate::catalog::{FullCatalog, FullCatalogEntry, RootCatalog, RootCatalogEntry, ShardStats};
use crate::checksum::blake3_hash;
use crate::error::{Result, ScxError};
use crate::header::{FileHeader, HEADER_SIZE};
use crate::modality::{ModalityFlags, ModalityInfo, ModalityTable, ModalityType, MAX_MODALITIES};
use crate::section::{align_to_8, SectionType};
use crate::shard::{
    derive_shard_type, BlockIndex, BlockIndexEntry, ShardHeader, SHARD_HEADER_SIZE,
};

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
            #[cfg(feature = "deletion-vectors")]
            bitmap_shard_count: 0,
            #[cfg(feature = "deletion-vectors")]
            modality_bitmap_counts: Vec::new(),
            total_nnz: 0,
            has_obsm: false,
            has_obsp: false,
            current_modality_id: 0,
            modalities: Vec::new(),
            modality_build_csc: Vec::new(),
        })
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
        self.write_padding()?;

        let offset = self.current_offset;
        let length = data.len() as u64;
        let checksum = blake3_hash(data);

        self.writer()?.write_all(data)?;
        self.current_offset += length;

        self.entries.push(FullCatalogEntry {
            name: name.into(),
            offset,
            length,
            section_type,
            checksum,
            modality_id: self.current_modality_id,
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

    /// Write the obs metadata section (Arrow IPC).
    pub fn write_obs(&mut self, obs: &RecordBatch) -> Result<()> {
        let data = Self::write_arrow_ipc(obs)?;
        self.write_section_bytes("obs", SectionType::ObsMetadata, &data, None)
    }

    /// Write the var metadata section (Arrow IPC).
    pub fn write_var(&mut self, var: &RecordBatch) -> Result<()> {
        let data = Self::write_arrow_ipc(var)?;
        self.write_section_bytes("var", SectionType::VarMetadata, &data, None)
    }

    /// Write the uns (unstructured) section as JSON.
    pub fn write_uns(&mut self, json: &serde_json::Value) -> Result<()> {
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
        self.write_padding()?;

        let shard_global_offset = self.current_offset;
        if indptr.is_empty() {
            return Err(ScxError::EmptyIndptr);
        }
        let n_major = (indptr.len() - 1) as u32;
        let nnz = *indptr.last().unwrap_or(&0);
        let index_dtype_u16 = self.header.index_dtype == 0;

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

        // Encode the shard data
        let encoded = scx_codec::encode_shard(
            indptr,
            indices,
            values,
            codec_id,
            value_encoding,
            index_dtype_u16,
        )?;

        // Build block index (Phase 1: single entry covering entire shard)
        let block_index = BlockIndex {
            entries: vec![BlockIndexEntry::new(0, n_major, 0, 0, 0, nnz)?],
        };
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

        // Build shard header with relative offsets
        let indptr_rel_offset = SHARD_HEADER_SIZE as u32;
        let indptr_length = encoded.indptr_bytes.len() as u32;
        let indices_rel_offset = indptr_rel_offset + indptr_length;
        let indices_length = encoded.indices_bytes.len() as u32;
        let values_rel_offset = indices_rel_offset + indices_length;
        let values_length = encoded.values_bytes.len() as u32;
        let block_index_rel_offset = values_rel_offset + values_length;
        let block_index_length = block_index_bytes.len() as u32;

        let shard_header = ShardHeader {
            magic: crate::shard::SHARD_MAGIC,
            shard_format_version: 1,
            shard_type: derive_shard_type(section_type),
            codec_id: codec_id as u8,
            value_encoding: value_encoding as u8,
            index_dtype: self.header.index_dtype,
            reserved_flags: [0; 3],
            n_major,
            n_minor: {
                // CSC shards have n_obs as their minor axis (and
                // `row_major_n_minor` is unused for those); row-major
                // shards use the per-modality column count resolved
                // above so multimodal files stamp the correct extent.
                let header_n_minor = match section_type {
                    SectionType::CscShard => self.header.n_obs,
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
        // type: CSC shards use the column-major axis (the file-wide
        // `n_obs` is the unbound row range, shared across modalities),
        // all others use the row-major axis with the per-modality
        // column extent resolved at the top of this function.
        // `row_start` here is interpreted on the major axis — for CSC
        // paths it carries `col_start` (see `write_csc_shard`, which
        // passes its `col_start` argument as the inner `row_start`).
        let (major_kind, n_minor) = match section_type {
            SectionType::CscShard => (MajorAxis::Col, self.header.n_obs),
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
            _ => {}
        }

        Ok(())
    }

    /// Write a pre-encoded shard section produced by parallel encoding.
    ///
    /// The encoding, checksums, and stats have all been computed in advance
    /// (typically in parallel via rayon). This method only performs the
    /// sequential I/O write and catalog entry bookkeeping.
    pub fn write_preencoded_shard(&mut self, section: PreEncodedSection) -> Result<()> {
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

    /// Per-modality `write_obsp_shard`. Section name is
    /// `obsp/{modality_name}/{name}/shard_{shard_idx}`.
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
            // in `scx-format/src/reader.rs::read_shard_from_entry_inner`,
            // adapted to a `File` (no mmap).
            let mut csr_shards: Vec<scx_sparse::ScxCsr> = Vec::with_capacity(csr_entries.len());
            // First-shard codec / value_encoding govern the CSC sidecar
            // (matches the standalone `scx build-csc` choice).
            let mut csc_codec: Option<CodecId> = None;
            let mut csc_value_encoding: Option<ValueEncoding> = None;
            for entry in &csr_entries {
                let (sh, indptr, indices, data) = self.decode_csr_entry(entry, n_vars)?;
                if csc_codec.is_none() {
                    csc_codec = Some(
                        CodecId::from_u8(sh.codec_id).ok_or(ScxError::UnknownCodec(sh.codec_id))?,
                    );
                    csc_value_encoding = Some(
                        ValueEncoding::from_u8(sh.value_encoding)
                            .ok_or(ScxError::UnknownValueEncoding(sh.value_encoding))?,
                    );
                }
                let n_shard_rows = indptr.len() - 1;
                csr_shards.push(scx_sparse::ScxCsr::new_unchecked(
                    (n_shard_rows, n_vars),
                    indptr,
                    indices,
                    data,
                ));
            }

            let codec = csc_codec.expect("at least one CSR entry processed");
            let value_encoding = csc_value_encoding.expect("at least one CSR entry processed");

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

        let sh = ShardHeader::read_from(&mut Cursor::new(&section[..SHARD_HEADER_SIZE]))?;

        let indptr_bytes = &section[sh.indptr_rel_offset as usize..][..sh.indptr_length as usize];
        let indices_bytes =
            &section[sh.indices_rel_offset as usize..][..sh.indices_length as usize];
        let values_bytes = &section[sh.values_rel_offset as usize..][..sh.values_length as usize];

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
        )
        .map_err(ScxError::Codec)?;

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
        self.header.n_csr_shards = self.csr_shard_count;
        self.header.n_csc_shards = self.csc_shard_count;
        self.header.nnz = self.total_nnz;
        self.header.file_checksum = 0;
        if self.csc_shard_count > 0 {
            self.header.set_csc();
        }
        if self.has_obsm {
            self.header.set_obsm();
        }
        if self.has_obsp {
            self.header.set_obsp();
        }
        // Phase 5b: header `has_bitmap` flag is set if ≥1 bitmap shard
        // landed (unimodal or any modality).
        #[cfg(feature = "deletion-vectors")]
        if self.bitmap_shard_count > 0
            || self.modality_bitmap_counts.iter().any(|&n| n > 0)
        {
            self.header.set_bitmap();
        }

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
        let shard_idx = self
            .modality_bitmap_counts
            .get(idx)
            .copied()
            .unwrap_or(0);
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
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn sample_header() -> FileHeader {
        FileHeader {
            magic: crate::header::MAGIC,
            format_version: crate::header::CURRENT_FORMAT_VERSION,
            header_length: 256,
            flags: 0,
            n_obs: 100,
            n_vars: 50,
            nnz: 500,
            n_csr_shards: 0,
            n_csc_shards: 0,
            shard_target_rows: crate::DEFAULT_SHARD_TARGET_ROWS,
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

    fn sample_obs() -> RecordBatch {
        let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(vec![
                "cell_0", "cell_1", "cell_2",
            ]))],
        )
        .unwrap()
    }

    fn sample_var() -> RecordBatch {
        let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(vec!["gene_0", "gene_1"]))],
        )
        .unwrap()
    }

    fn sample_shard_data() -> (Vec<u64>, Vec<u32>, Vec<u8>) {
        // 3 rows, nnz=6
        let indptr = vec![0u64, 2, 5, 6];
        let indices = vec![1u32, 3, 0, 2, 4, 2];
        let values: Vec<u8> = vec![5, 10, 1, 3, 7, 2]; // u8 encoding
        (indptr, indices, values)
    }

    /// 10.14: Write minimal file → verify header fields and 8-byte alignment
    #[test]
    fn test_minimal_file_header_and_alignment() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.scx");
        let header = sample_header();

        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs()).unwrap();
        writer.write_var(&sample_var()).unwrap();

        let (indptr, indices, values) = sample_shard_data();
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

        let final_path = writer.finish().unwrap();
        assert!(final_path.exists());

        // Read back header
        let data = std::fs::read(&final_path).unwrap();
        let mut cursor = std::io::Cursor::new(&data);
        let hdr = FileHeader::read_from(&mut cursor).unwrap();

        assert_eq!(hdr.magic, crate::header::MAGIC);
        assert_eq!(hdr.format_version, crate::header::CURRENT_FORMAT_VERSION);
        assert_eq!(hdr.n_csr_shards, 1);
        assert_eq!(hdr.root_catalog_offset, HEADER_SIZE as u64);
        assert!(hdr.full_catalog_offset >= SECTIONS_START_OFFSET);
        assert_eq!(hdr.full_catalog_offset % 8, 0);
        assert!(hdr.file_checksum != 0);

        // Verify full catalog is readable
        let fc_start = hdr.full_catalog_offset as usize;
        let fc_end = fc_start + hdr.full_catalog_length as usize;
        let mut fc_cursor = std::io::Cursor::new(&data[fc_start..fc_end]);
        let catalog =
            FullCatalog::read_from(&mut fc_cursor, hdr.full_catalog_length as usize, true).unwrap();
        assert_eq!(catalog.entries.len(), 3); // obs + var + 1 shard

        // Verify all section offsets are 8-byte aligned
        for entry in &catalog.entries {
            assert_eq!(
                entry.offset % 8,
                0,
                "section '{}' offset {} not 8-byte aligned",
                entry.name,
                entry.offset
            );
        }
    }

    /// 10.15: Temp file lifecycle — tmp exists before finish, final after.
    /// With randomized temp file names we can't predict the exact path,
    /// so we scan the directory for `.tmp` files instead.
    #[test]
    fn test_temp_file_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lifecycle.scx");

        let header = sample_header();
        let mut writer = ScxWriter::new(&path, header).unwrap();

        // A temp file should exist in the directory; final path should not.
        let tmp_files_before: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .map(|s| s.contains(".tmp"))
                    .unwrap_or(false)
            })
            .collect();
        assert!(
            !tmp_files_before.is_empty(),
            "temp file should exist after new()"
        );
        assert!(
            !path.exists(),
            "final path should not exist before finish()"
        );

        writer.write_obs(&sample_obs()).unwrap();
        writer.write_var(&sample_var()).unwrap();

        let (indptr, indices, values) = sample_shard_data();
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

        writer.finish().unwrap();

        // Final file should exist; no temp files should remain.
        assert!(path.exists(), "final path should exist after finish()");
        let tmp_files_after: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .map(|s| s.contains(".tmp"))
                    .unwrap_or(false)
            })
            .collect();
        assert!(
            tmp_files_after.is_empty(),
            "no temp files should remain after finish()"
        );
    }

    /// 10.16: Write obs + var + 4 CSR shards → verify catalog entries
    #[test]
    fn test_four_shards_catalog() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("four_shards.scx");
        let header = sample_header();

        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs()).unwrap();
        writer.write_var(&sample_var()).unwrap();

        let (indptr, indices, values) = sample_shard_data();
        for i in 0..4u64 {
            writer
                .write_csr_shard(
                    &indptr,
                    &indices,
                    &values,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    i * 3, // each shard has 3 rows
                )
                .unwrap();
        }

        let final_path = writer.finish().unwrap();

        // Read back catalog
        let data = std::fs::read(&final_path).unwrap();
        let mut cursor = std::io::Cursor::new(&data);
        let hdr = FileHeader::read_from(&mut cursor).unwrap();
        assert_eq!(hdr.n_csr_shards, 4);

        let fc_start = hdr.full_catalog_offset as usize;
        let fc_end = fc_start + hdr.full_catalog_length as usize;
        let mut fc_cursor = std::io::Cursor::new(&data[fc_start..fc_end]);
        let catalog =
            FullCatalog::read_from(&mut fc_cursor, hdr.full_catalog_length as usize, true).unwrap();

        assert_eq!(catalog.entries.len(), 6); // obs + var + 4 shards

        let csr_entries: Vec<_> = catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CsrShard)
            .collect();
        assert_eq!(csr_entries.len(), 4);

        // Verify distinct row_starts
        let row_starts: Vec<u64> = csr_entries
            .iter()
            .map(|e| e.stats.as_ref().unwrap().row_start)
            .collect();
        assert_eq!(row_starts, vec![0, 3, 6, 9]);
    }

    /// 10.17: Write all section types → verify in catalog
    #[test]
    fn test_all_section_types() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("all_types.scx");
        let header = sample_header();

        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs()).unwrap();
        writer.write_var(&sample_var()).unwrap();

        let (indptr, indices, values) = sample_shard_data();
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

        // Layer shard
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

        // obsm
        let obsm_schema = Schema::new(vec![
            Field::new("x", DataType::Int32, false),
            Field::new("y", DataType::Int32, false),
        ]);
        let obsm_batch = RecordBatch::try_new(
            Arc::new(obsm_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Int32Array::from(vec![4, 5, 6])),
            ],
        )
        .unwrap();
        writer.write_obsm("X_pca", &obsm_batch).unwrap();

        // uns
        writer
            .write_uns(&serde_json::json!({"key": "value"}))
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

        let final_path = writer.finish().unwrap();

        let data = std::fs::read(&final_path).unwrap();
        let mut cursor = std::io::Cursor::new(&data);
        let hdr = FileHeader::read_from(&mut cursor).unwrap();

        let fc_start = hdr.full_catalog_offset as usize;
        let fc_end = fc_start + hdr.full_catalog_length as usize;
        let mut fc_cursor = std::io::Cursor::new(&data[fc_start..fc_end]);
        let catalog =
            FullCatalog::read_from(&mut fc_cursor, hdr.full_catalog_length as usize, true).unwrap();

        // obs, var, csr_shard, layer_csr_shard, obsm, uns, provenance = 7
        assert_eq!(catalog.entries.len(), 7);

        let types: Vec<SectionType> = catalog.entries.iter().map(|e| e.section_type).collect();
        assert!(types.contains(&SectionType::ObsMetadata));
        assert!(types.contains(&SectionType::VarMetadata));
        assert!(types.contains(&SectionType::CsrShard));
        assert!(types.contains(&SectionType::LayerCsrShard));
        assert!(types.contains(&SectionType::ObsmEmbedding));
        assert!(types.contains(&SectionType::UnsBlob));
        assert!(types.contains(&SectionType::Provenance));
    }

    /// 10.18: Root catalog at offset 256 is <= 4096 bytes and readable
    #[test]
    fn test_root_catalog_structure() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("root_cat.scx");
        let header = sample_header();

        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs()).unwrap();
        writer.write_var(&sample_var()).unwrap();

        let (indptr, indices, values) = sample_shard_data();
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

        let final_path = writer.finish().unwrap();

        let data = std::fs::read(&final_path).unwrap();
        let mut cursor = std::io::Cursor::new(&data);
        let hdr = FileHeader::read_from(&mut cursor).unwrap();

        assert_eq!(hdr.root_catalog_offset, HEADER_SIZE as u64);
        assert!(hdr.root_catalog_length <= 4096);

        // Read root catalog from offset 256
        let rc_start = HEADER_SIZE;
        let mut rc_cursor = std::io::Cursor::new(&data[rc_start..rc_start + 4096]);
        let root_catalog = RootCatalog::read_from(&mut rc_cursor).unwrap();

        // Should have groups for ObsMetadata, VarMetadata, CsrShard
        assert!(root_catalog.n_section_groups >= 3);
        assert_eq!(
            root_catalog.entries.len(),
            root_catalog.n_section_groups as usize
        );

        // Verify each group has valid fields
        for entry in &root_catalog.entries {
            assert!(entry.first_section_offset >= SECTIONS_START_OFFSET);
            assert!(entry.n_sections > 0);
            assert!(entry.total_group_length > 0);
        }
    }

    /// Test Drop cleans up temp file when finish() is not called.
    /// TempPath auto-deletes the temp file on drop.
    #[test]
    fn test_drop_cleans_up_temp() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dropped.scx");

        {
            let _writer = ScxWriter::new(&path, sample_header()).unwrap();
            // A temp file should exist somewhere in the directory.
            let tmp_count = std::fs::read_dir(dir.path())
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_name()
                        .to_str()
                        .map(|s| s.contains(".tmp"))
                        .unwrap_or(false)
                })
                .count();
            assert!(tmp_count > 0, "temp file should exist before drop");
        }
        // After drop: no temp files, no final file.
        let tmp_count_after = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .map(|s| s.contains(".tmp"))
                    .unwrap_or(false)
            })
            .count();
        assert_eq!(
            tmp_count_after, 0,
            "temp file should be cleaned up after drop"
        );
        assert!(
            !path.exists(),
            "final path should not exist after drop without finish"
        );
    }

    /// Two concurrent ScxWriters targeting the same final path should use
    /// different temp files and not trample each other.
    #[test]
    fn test_concurrent_writers_no_collision() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("concurrent.scx");

        let writer1 = ScxWriter::new(&path, sample_header()).unwrap();
        let writer2 = ScxWriter::new(&path, sample_header()).unwrap();

        // Both writers should have created separate temp files.
        let tmp_files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .map(|s| s.contains(".tmp"))
                    .unwrap_or(false)
            })
            .collect();
        assert_eq!(
            tmp_files.len(),
            2,
            "two concurrent writers should create two distinct temp files"
        );

        // Dropping both should clean up both temp files.
        drop(writer1);
        drop(writer2);

        let tmp_remaining: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .map(|s| s.contains(".tmp"))
                    .unwrap_or(false)
            })
            .collect();
        assert!(
            tmp_remaining.is_empty(),
            "all temp files should be cleaned up after dropping both writers"
        );
    }

    /// `make_sibling_tempfile` creates a temp file in the same directory
    /// as the final path, with the `.{stem}_<rand>.tmp` naming convention.
    #[test]
    fn test_make_sibling_tempfile_creates_in_parent() {
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("dest.scx");
        let (file, tmp_path) = make_sibling_tempfile(&final_path).unwrap();
        // File handle should be valid (write 1 byte).
        let mut f = file;
        use std::io::Write;
        f.write_all(b"x").unwrap();
        // Temp path lives in the same parent directory.
        assert_eq!(tmp_path.parent(), Some(dir.path()));
        // Filename matches `.dest.scx_*.tmp`.
        let name = tmp_path.file_name().and_then(|n| n.to_str()).unwrap();
        assert!(
            name.starts_with(".dest.scx_"),
            "name {name:?} should start with `.dest.scx_`"
        );
        assert!(
            name.ends_with(".tmp"),
            "name {name:?} should end with `.tmp`"
        );
        // Dropping `tmp_path` cleans up.
        let path_clone = tmp_path.to_path_buf();
        drop(tmp_path);
        assert!(!path_clone.exists(), "temp file should be deleted on drop");
    }

    /// `finish()` should restore umask-respecting permissions on the
    /// persisted file. `tempfile::NamedTempFile` creates `0600`; after the
    /// post-persist `chmod_to_umask` call we expect `0o666 & !umask`.
    ///
    /// This test is Unix-only and inherently single-threaded because it
    /// reads (and briefly clears) the process umask. The umask cache in
    /// `current_umask()` reads on first call — to make this test
    /// deterministic regardless of test ordering we force a known umask
    /// before any `chmod_to_umask` call in this binary may have run.
    #[cfg(unix)]
    #[test]
    fn test_finish_sets_umask_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("perm.scx");

        let mut writer = ScxWriter::new(&path, sample_header()).unwrap();
        writer.write_obs(&sample_obs()).unwrap();
        writer.write_var(&sample_var()).unwrap();
        let (indptr, indices, values) = sample_shard_data();
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
        writer.finish().unwrap();

        // Read back current umask the same way `chmod_to_umask` does.
        // SAFETY: same umask dance as `current_umask`.
        let umask = unsafe {
            let saved = libc::umask(0o022);
            libc::umask(saved);
            saved as u32
        };
        let expected_mode = 0o666 & !umask;
        let actual_mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(
            actual_mode, expected_mode,
            "persisted file mode {actual_mode:o} should equal 0o666 & !umask ({expected_mode:o})"
        );
    }

    /// Test compute_shard_stats (row-major branch)
    #[test]
    fn test_compute_shard_stats() {
        let values: Vec<u8> = vec![5, 10, 1, 3, 7, 2];
        // Row-major shard: rows [0, 3), col_end = n_minor (= n_vars).
        let stats = compute_shard_stats(&values, ValueEncoding::Uint8, MajorAxis::Row, 0, 3, 50, 6);
        assert_eq!(stats.row_start, 0);
        assert_eq!(stats.row_end, 3);
        assert_eq!(stats.col_start, 0);
        assert_eq!(stats.col_end, 50);
        assert_eq!(stats.nnz, 6);
        assert_eq!(stats.value_min, 1);
        assert_eq!(stats.value_max, 10);
        assert_eq!(stats.value_sum, 28); // 5+10+1+3+7+2
    }

    /// Test compute_shard_stats column-major branch.
    #[test]
    fn test_compute_shard_stats_col_major() {
        let values: Vec<u8> = vec![5, 10, 1];
        // Column-major shard: cols [100, 102), row pair = full [0, n_obs).
        let stats = compute_shard_stats(
            &values,
            ValueEncoding::Uint8,
            MajorAxis::Col,
            100,
            2,
            1000,
            3,
        );
        assert_eq!(stats.col_start, 100);
        assert_eq!(stats.col_end, 102);
        assert_eq!(stats.row_start, 0);
        assert_eq!(stats.row_end, 1000);
    }

    /// Test compute_shard_stats for Float32 returns zero stats
    #[test]
    fn test_compute_shard_stats_float32() {
        // Float32: 3 values as LE bytes (1.0f32, 2.5f32, 0.5f32)
        let mut values = Vec::new();
        values.extend_from_slice(&1.0f32.to_le_bytes());
        values.extend_from_slice(&2.5f32.to_le_bytes());
        values.extend_from_slice(&0.5f32.to_le_bytes());
        let stats =
            compute_shard_stats(&values, ValueEncoding::Float32, MajorAxis::Row, 0, 2, 50, 3);
        assert_eq!(stats.value_min, 0, "float32 value_min must be zero");
        assert_eq!(stats.value_max, 0, "float32 value_max must be zero");
        assert_eq!(stats.value_sum, 0, "float32 value_sum must be zero");
        assert_eq!(stats.row_start, 0);
        assert_eq!(stats.row_end, 2);
        assert_eq!(stats.nnz, 3);
    }

    /// Test compute_shard_stats for Float16 returns zero stats
    #[test]
    fn test_compute_shard_stats_float16() {
        let values = vec![0u8; 6]; // 3 × 2-byte float16 values
        let stats = compute_shard_stats(
            &values,
            ValueEncoding::Float16,
            MajorAxis::Row,
            10,
            5,
            50,
            3,
        );
        assert_eq!(stats.value_min, 0, "float16 value_min must be zero");
        assert_eq!(stats.value_max, 0, "float16 value_max must be zero");
        assert_eq!(stats.value_sum, 0, "float16 value_sum must be zero");
        assert_eq!(stats.row_start, 10);
        assert_eq!(stats.row_end, 15);
    }

    /// P3: Write CSR + CSC shards → verify CSC section in catalog and correct data
    #[test]
    fn test_csc_shard_write_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("with_csc.scx");
        let header = sample_header();

        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs()).unwrap();
        writer.write_var(&sample_var()).unwrap();

        let (indptr, indices, values) = sample_shard_data();

        // Write a CSR shard (row-major)
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

        // Write a CSC shard (column-major) with the same data
        // In a real scenario the indptr/indices represent column pointers/row indices
        writer
            .write_csc_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0, // col_start
            )
            .unwrap();

        let final_path = writer.finish().unwrap();

        // Read back and verify
        let data = std::fs::read(&final_path).unwrap();
        let mut cursor = std::io::Cursor::new(&data);
        let hdr = FileHeader::read_from(&mut cursor).unwrap();

        // Verify shard counts
        assert_eq!(hdr.n_csr_shards, 1);
        assert_eq!(hdr.n_csc_shards, 1);

        // Verify catalog has both shard types
        let fc_start = hdr.full_catalog_offset as usize;
        let fc_end = fc_start + hdr.full_catalog_length as usize;
        let mut fc_cursor = std::io::Cursor::new(&data[fc_start..fc_end]);
        let catalog =
            FullCatalog::read_from(&mut fc_cursor, hdr.full_catalog_length as usize, true).unwrap();

        // obs + var + 1 CSR shard + 1 CSC shard = 4 entries
        assert_eq!(catalog.entries.len(), 4);

        let csr_entries: Vec<_> = catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CsrShard)
            .collect();
        assert_eq!(csr_entries.len(), 1);
        assert_eq!(csr_entries[0].name, "X_shard_0");

        let csc_entries: Vec<_> = catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CscShard)
            .collect();
        assert_eq!(csc_entries.len(), 1);
        assert_eq!(csc_entries[0].name, "X_csc_shard_0");

        // CSC shard should have stats (computed from values)
        let csc_stats = csc_entries[0].stats.as_ref().unwrap();
        assert_eq!(csc_stats.nnz, 6);

        // CSC shard's on-disk header byte must be `1` (Phase A.1).
        let csc_section = &data[csc_entries[0].offset as usize..][..csc_entries[0].length as usize];
        let csc_sh =
            ShardHeader::read_from(&mut std::io::Cursor::new(&csc_section[..SHARD_HEADER_SIZE]))
                .unwrap();
        assert_eq!(csc_sh.shard_type, 1, "CSC shard_type byte must be 1");
        assert!(csc_sh.is_csc(SectionType::CscShard));

        // CSR shard byte must remain `0`.
        let csr_section = &data[csr_entries[0].offset as usize..][..csr_entries[0].length as usize];
        let csr_sh =
            ShardHeader::read_from(&mut std::io::Cursor::new(&csr_section[..SHARD_HEADER_SIZE]))
                .unwrap();
        assert_eq!(csr_sh.shard_type, 0, "CSR shard_type byte must be 0");
        assert!(!csr_sh.is_csc(SectionType::CsrShard));

        // All sections should be 8-byte aligned
        for entry in &catalog.entries {
            assert_eq!(
                entry.offset % 8,
                0,
                "section '{}' offset {} not 8-byte aligned",
                entry.name,
                entry.offset
            );
        }
    }

    /// v2 strict shard_type validation: a CSC shard whose
    /// `shard_type` byte is corrupted to 0 must be rejected by the
    /// reader. This is the new behavior on the v2 catalog read path
    /// (catalog-wins tolerance survives only on v1 reads).
    #[test]
    fn test_strict_shard_type_v2_rejects_corrupted_csc() {
        use crate::reader::ScxReader;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("strict_csc.scx");
        let header = sample_header();

        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs()).unwrap();
        writer.write_var(&sample_var()).unwrap();

        let (indptr, indices, values) = sample_shard_data();
        writer
            .write_csc_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();

        let final_path = writer.finish().unwrap();

        // Find the CSC shard's on-disk shard_type byte (offset 5 in
        // the shard header) and corrupt it from 1 → 0.
        let mut data = std::fs::read(&final_path).unwrap();
        let hdr = FileHeader::read_from(&mut std::io::Cursor::new(&data)).unwrap();
        let fc_start = hdr.full_catalog_offset as usize;
        let fc_end = fc_start + hdr.full_catalog_length as usize;
        let catalog = FullCatalog::read_from(
            &mut std::io::Cursor::new(&data[fc_start..fc_end]),
            hdr.full_catalog_length as usize,
            true,
        )
        .unwrap();
        let csc_entry = catalog
            .entries
            .iter()
            .find(|e| e.section_type == SectionType::CscShard)
            .unwrap();
        // shard_type is the 6th byte of the shard header (after the
        // 4-byte magic and 1-byte shard_format_version).
        let shard_type_offset = csc_entry.offset as usize + 4 + 1;
        assert_eq!(data[shard_type_offset], 1, "writer must emit shard_type=1");
        data[shard_type_offset] = 0;

        // Need to rewrite to a new path to preserve the original mmap
        // semantics; the file_checksum will not match either, so open
        // with verify_catalog/header disabled.
        let corrupt_path = dir.path().join("strict_csc_corrupt.scx");
        std::fs::write(&corrupt_path, &data).unwrap();

        // Open and try to read the CSC shard. The strict v2 validator
        // fires inside `read_shard_from_entry_inner` and returns
        // `InvalidShardType`.
        let reader = ScxReader::open_unchecked(&corrupt_path).unwrap();
        let err = reader.read_csc_shard(0).unwrap_err();
        match err {
            ScxError::InvalidShardType {
                expected,
                got,
                section_type,
            } => {
                assert_eq!(expected, 1);
                assert_eq!(got, 0);
                assert_eq!(section_type, SectionType::CscShard as u8);
            }
            other => panic!("expected InvalidShardType, got {other:?}"),
        }
    }

    /// P3: has_csc flag (bit 0) is auto-set when CSC shards are written
    #[test]
    fn test_has_csc_flag_set() {
        let dir = tempfile::tempdir().unwrap();

        // File WITHOUT CSC shards: flag should NOT be set
        {
            let path = dir.path().join("no_csc.scx");
            let header = sample_header();
            let mut writer = ScxWriter::new(&path, header).unwrap();
            writer.write_obs(&sample_obs()).unwrap();
            writer.write_var(&sample_var()).unwrap();
            let (indptr, indices, values) = sample_shard_data();
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
            writer.finish().unwrap();

            let data = std::fs::read(&path).unwrap();
            let mut cursor = std::io::Cursor::new(&data);
            let hdr = FileHeader::read_from(&mut cursor).unwrap();
            assert!(
                !hdr.has_csc(),
                "has_csc should be false when no CSC shards written"
            );
            assert_eq!(hdr.n_csc_shards, 0);
        }

        // File WITH CSC shards: flag SHOULD be set
        {
            let path = dir.path().join("with_csc.scx");
            let header = sample_header();
            let mut writer = ScxWriter::new(&path, header).unwrap();
            writer.write_obs(&sample_obs()).unwrap();
            writer.write_var(&sample_var()).unwrap();
            let (indptr, indices, values) = sample_shard_data();
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
                .write_csc_shard(
                    &indptr,
                    &indices,
                    &values,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    0,
                )
                .unwrap();
            writer.finish().unwrap();

            let data = std::fs::read(&path).unwrap();
            let mut cursor = std::io::Cursor::new(&data);
            let hdr = FileHeader::read_from(&mut cursor).unwrap();
            assert!(
                hdr.has_csc(),
                "has_csc should be true when CSC shards written"
            );
            assert_eq!(hdr.n_csc_shards, 1);
        }
    }

    // -----------------------------------------------------------------------
    // Phase A.4 — CSC round-trip and codec sweep
    // -----------------------------------------------------------------------

    /// Build a 4-row, 6-column dense reference matrix with known entries.
    ///
    /// Returns `(dense_row_major, n_rows, n_cols)`. Used by the
    /// multi-shard CSC round-trip test below to sanity-check
    /// densification.
    fn dense_4x6() -> (Vec<f32>, usize, usize) {
        // Hand-picked sparse pattern across 6 columns; row indices in
        // [0, 4), unsorted within each column to exercise the col_slice
        // / concatenation paths without assuming sorted input.
        let n_rows = 4usize;
        let n_cols = 6usize;
        #[rustfmt::skip]
        let dense: Vec<f32> = vec![
            // col: 0    1    2    3    4    5
                   1.0, 0.0, 0.0, 4.0, 0.0, 7.0,
                   0.0, 2.0, 5.0, 0.0, 0.0, 8.0,
                   0.0, 0.0, 0.0, 0.0, 6.0, 0.0,
                   3.0, 0.0, 0.0, 0.0, 0.0, 9.0,
        ];
        (dense, n_rows, n_cols)
    }

    /// Build CSC arrays for `cols` (a contiguous range of column
    /// indices) over a dense row-major matrix. Returns the on-disk
    /// layout: `(indptr_u64, indices_u32, values_le_bytes)`.
    fn csc_arrays_for_col_range(
        dense: &[f32],
        n_rows: usize,
        n_cols: usize,
        col_start: usize,
        col_end: usize,
        encoding: ValueEncoding,
    ) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
        let mut indptr: Vec<u64> = Vec::with_capacity(col_end - col_start + 1);
        indptr.push(0);
        let mut indices: Vec<u32> = Vec::new();
        let mut values_f32: Vec<f32> = Vec::new();

        for col in col_start..col_end {
            for row in 0..n_rows {
                let v = dense[row * n_cols + col];
                if v != 0.0 {
                    indices.push(row as u32);
                    values_f32.push(v);
                }
            }
            indptr.push(indices.len() as u64);
        }

        // Encode values to LE bytes per the requested encoding.
        let mut values_bytes = Vec::with_capacity(values_f32.len() * encoding.byte_width());
        for &v in &values_f32 {
            encoding.encode_f32(&mut values_bytes, v).unwrap();
        }

        (indptr, indices, values_bytes)
    }

    /// Build a header for a CSC round-trip test fixture.
    fn csc_test_header(n_obs: u64, n_vars: u64) -> FileHeader {
        FileHeader {
            magic: crate::header::MAGIC,
            format_version: crate::header::CURRENT_FORMAT_VERSION,
            header_length: 256,
            flags: 0,
            n_obs,
            n_vars,
            nnz: 0,
            n_csr_shards: 0,
            n_csc_shards: 0,
            shard_target_rows: crate::DEFAULT_SHARD_TARGET_ROWS,
            codec_id: 0,
            // u32 indices on disk (Phase A test fixtures use n_vars=6
            // which fits in u16, but we want index_dtype to track
            // arrays we hand the writer; the writer reads it from the
            // header). u16 index_dtype byte = 0; u32 = 1.
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

    /// Round-trip a 2-shard CSC file and verify that
    /// `read_all_csc_shards` densifies back to the source matrix.
    #[test]
    fn test_csc_two_shard_round_trip() {
        use crate::reader::ScxReader;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("two_shard_csc.scx");

        let (dense, n_rows, n_cols) = dense_4x6();
        let header = csc_test_header(n_rows as u64, n_cols as u64);

        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs()).unwrap();
        writer.write_var(&sample_var()).unwrap();

        // Need a CSR shard so the file passes basic invariants
        // (`n_obs > 0` requires at least one row-shard for downstream
        // tools); use a tiny 4-row CSR shard with all zeros.
        let csr_indptr = vec![0u64; n_rows + 1];
        writer
            .write_csr_shard(
                &csr_indptr,
                &[],
                &[],
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();

        // Shard 1: cols [0..3); shard 2: cols [3..6).
        let (ip1, ix1, vb1) =
            csc_arrays_for_col_range(&dense, n_rows, n_cols, 0, 3, ValueEncoding::Uint8);
        let (ip2, ix2, vb2) =
            csc_arrays_for_col_range(&dense, n_rows, n_cols, 3, 6, ValueEncoding::Uint8);

        writer
            .write_csc_shard(&ip1, &ix1, &vb1, CodecId::None, ValueEncoding::Uint8, 0)
            .unwrap();
        writer
            .write_csc_shard(&ip2, &ix2, &vb2, CodecId::None, ValueEncoding::Uint8, 3)
            .unwrap();

        writer.finish().unwrap();

        // Read back via the high-level CSC API.
        let reader = ScxReader::open(&path).unwrap();
        assert_eq!(reader.csc_shard_count(), 2);

        let csc = reader.read_all_csc_shards().unwrap();
        assert_eq!(csc.shape, (n_rows, n_cols));
        let densified = csc.to_dense().unwrap();
        assert_eq!(densified, dense);

        // Per-shard reads also work.
        let s0 = reader.read_csc_shard(0).unwrap();
        assert_eq!(s0.n_cols(), 3);
        let s1 = reader.read_csc_shard(1).unwrap();
        assert_eq!(s1.n_cols(), 3);

        // CSC entries in the catalog have correct col_start/col_end.
        let csc_entries = reader.catalog().csc_shards_sorted();
        assert_eq!(csc_entries.len(), 2);
        let r0 = csc_entries[0].stats.as_ref().unwrap().col_range();
        let r1 = csc_entries[1].stats.as_ref().unwrap().col_range();
        assert_eq!(r0, 0..3);
        assert_eq!(r1, 3..6);
    }

    /// Codec sweep: write a single CSC shard under every supported
    /// codec × value-encoding combination and confirm round-trip
    /// equality. Pcodec exercises a different decode path than
    /// None/Zstd/Lz4Shuffle and is included.
    ///
    /// Scx1 is integer-only; combinations with Float32/Float16 are
    /// skipped (they would error at encode time).
    #[test]
    fn test_csc_codec_sweep() {
        use crate::reader::ScxReader;

        let dir = tempfile::tempdir().unwrap();
        let (dense, n_rows, n_cols) = dense_4x6();

        let codecs = [
            CodecId::None,
            CodecId::Scx1,
            CodecId::Zstd,
            CodecId::Lz4Shuffle,
            CodecId::Pcodec,
        ];
        let encodings = [
            ValueEncoding::Uint8,
            ValueEncoding::Uint16,
            ValueEncoding::Uint32,
            ValueEncoding::Float32,
            ValueEncoding::Float16,
        ];

        for &codec in &codecs {
            for &enc in &encodings {
                if codec == CodecId::Scx1 && !enc.is_integer() {
                    continue;
                }
                let label = format!("codec={codec:?}/enc={enc:?}");
                let path = dir
                    .path()
                    .join(format!("csc_sweep_{}_{}.scx", codec as u8, enc as u8));
                let header = csc_test_header(n_rows as u64, n_cols as u64);

                let mut writer = ScxWriter::new(&path, header).unwrap();
                writer.write_obs(&sample_obs()).unwrap();
                writer.write_var(&sample_var()).unwrap();

                // Empty CSR shard for the file invariant.
                let csr_indptr = vec![0u64; n_rows + 1];
                writer
                    .write_csr_shard(
                        &csr_indptr,
                        &[],
                        &[],
                        CodecId::None,
                        ValueEncoding::Uint8,
                        0,
                    )
                    .unwrap();

                let (ip, ix, vb) = csc_arrays_for_col_range(&dense, n_rows, n_cols, 0, n_cols, enc);
                writer
                    .write_csc_shard(&ip, &ix, &vb, codec, enc, 0)
                    .unwrap();
                writer.finish().unwrap();

                let reader = ScxReader::open(&path).unwrap();
                let csc = reader.read_all_csc_shards().unwrap();
                let densified = csc.to_dense().unwrap();
                assert_eq!(densified, dense, "round-trip mismatch for {label}");
            }
        }
    }

    /// `read_csc_columns(range)` — verify that arbitrary contiguous
    /// column slices across a multi-shard layout match the
    /// densify-then-slice reference. Phase A asserts correctness only;
    /// shard-skip count assertions are deferred to Phase E.5 once
    /// `BackedCscReader::enable_metrics()` lands.
    #[test]
    fn test_read_csc_columns_range_correctness() {
        use crate::reader::ScxReader;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("csc_range.scx");

        let (dense, n_rows, n_cols) = dense_4x6();
        let header = csc_test_header(n_rows as u64, n_cols as u64);

        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs()).unwrap();
        writer.write_var(&sample_var()).unwrap();
        let csr_indptr = vec![0u64; n_rows + 1];
        writer
            .write_csr_shard(
                &csr_indptr,
                &[],
                &[],
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();

        // Three CSC shards: cols [0..2), [2..4), [4..6).
        for (col_start, col_end) in [(0usize, 2usize), (2, 4), (4, 6)] {
            let (ip, ix, vb) = csc_arrays_for_col_range(
                &dense,
                n_rows,
                n_cols,
                col_start,
                col_end,
                ValueEncoding::Uint8,
            );
            writer
                .write_csc_shard(
                    &ip,
                    &ix,
                    &vb,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    col_start as u64,
                )
                .unwrap();
        }
        writer.finish().unwrap();

        let reader = ScxReader::open(&path).unwrap();
        assert_eq!(reader.csc_shard_count(), 3);

        // Reference: dense slice for the same column range.
        let dense_slice = |c_lo: usize, c_hi: usize| -> Vec<f32> {
            let cols = c_hi - c_lo;
            let mut out = vec![0.0f32; n_rows * cols];
            for r in 0..n_rows {
                for (out_c, src_c) in (c_lo..c_hi).enumerate() {
                    out[r * cols + out_c] = dense[r * n_cols + src_c];
                }
            }
            out
        };

        let cases = [
            (0u32, 6u32),
            (1, 3), // partial-overlap on shards 0 and 1
            (2, 5), // partial-overlap on shards 1 and 2
            (3, 4), // single shard, partial slice
            (0, 0), // empty range
            (4, 6), // exact shard boundary
        ];
        for (c_lo, c_hi) in cases {
            let csc = reader.read_csc_columns(c_lo..c_hi).unwrap();
            assert_eq!(
                csc.shape,
                (n_rows, (c_hi - c_lo) as usize),
                "shape mismatch for cols [{c_lo}..{c_hi})"
            );
            let got = csc.to_dense().unwrap();
            let want = dense_slice(c_lo as usize, c_hi as usize);
            assert_eq!(got, want, "values mismatch for cols [{c_lo}..{c_hi})");
        }

        // read_csc_columns_subset over a sorted, non-contiguous selection.
        let subset = [0u32, 2, 3, 5];
        let csc = reader.read_csc_columns_subset(&subset).unwrap();
        assert_eq!(csc.shape, (n_rows, subset.len()));
        let got = csc.to_dense().unwrap();
        for (out_c, &src_c) in subset.iter().enumerate() {
            for r in 0..n_rows {
                assert_eq!(
                    got[r * subset.len() + out_c],
                    dense[r * n_cols + src_c as usize],
                    "subset mismatch at row {r} col {src_c}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase B integration tests
    // -----------------------------------------------------------------------

    /// Full multimodal round-trip: register 3 modalities, write
    /// distinct var batches per modality, then read everything back
    /// through the per-modality reader API.
    #[test]
    fn test_phase_b_three_modality_round_trip() {
        use crate::modality::{ModalityFlags, ModalityType};
        use crate::reader::ScxReader;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multimodal.scx");
        let header = sample_header();

        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs()).unwrap();

        // Register three modalities. Order matters — modality_id is
        // 1-based and equals position+1.
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
        let atac_id = writer
            .add_modality(
                "atac",
                ModalityType::Atac,
                CodecId::None,
                ValueEncoding::Uint8,
                false,
            )
            .unwrap();
        assert_eq!(rna_id, 1);
        assert_eq!(adt_id, 2);
        assert_eq!(atac_id, 3);

        // Distinct per-modality var batches. The writer doesn't
        // enforce a relationship between var.num_rows and the
        // modality's n_vars; we set that explicitly.
        let var_rna = sample_var();
        writer.write_var_for(rna_id, &var_rna).unwrap();
        writer.write_var_for(adt_id, &var_rna).unwrap();
        writer.write_var_for(atac_id, &var_rna).unwrap();
        writer.set_modality_n_vars(rna_id, 50).unwrap();
        writer.set_modality_n_vars(adt_id, 50).unwrap();
        writer.set_modality_n_vars(atac_id, 50).unwrap();

        // One CSR shard per modality.
        let (indptr, indices, values) = sample_shard_data();
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
        writer
            .write_csr_shard_for(
                adt_id,
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        writer
            .write_csr_shard_for(
                atac_id,
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();

        let final_path = writer.finish().unwrap();

        // Read back through ScxReader.
        let reader = ScxReader::open(&final_path).unwrap();
        assert!(reader.is_multimodal());
        assert_eq!(reader.n_modalities(), 3);
        assert_eq!(reader.modality_names(), vec!["rna", "adt", "atac"]);

        assert_eq!(reader.modality_id("rna"), Some(1));
        assert_eq!(reader.modality_id("adt"), Some(2));
        assert_eq!(reader.modality_id("atac"), Some(3));
        assert_eq!(reader.modality_id("missing"), None);

        let info = reader.modality_info(2).unwrap();
        assert_eq!(info.name, "adt");
        assert_eq!(info.modality_type, ModalityType::Protein);
        assert_eq!(info.n_vars, 50);
        assert_eq!(info.n_csr_shards, 1);
        assert_eq!(info.n_csc_shards, 0);
        assert_eq!(info.flags, ModalityFlags::empty());

        // header.has_modalities flag is set.
        assert!(reader.header().has_modalities());

        // Per-modality var read.
        let var_back = reader.read_var_for(rna_id).unwrap();
        assert_eq!(var_back.num_rows(), var_rna.num_rows());

        // Per-modality CSR shard count + read.
        for id in [rna_id, adt_id, atac_id] {
            assert_eq!(reader.csr_shard_count_for(id), 1);
            let (ip, ix, dv) = reader.read_csr_shard_for(id, 0).unwrap();
            assert_eq!(ip.len(), indptr.len());
            assert_eq!(ix.len(), indices.len());
            assert_eq!(dv.len(), values.len());
        }
    }

    /// Regression test for PR #68: every CSR shard written via
    /// `write_csr_shard_for` must stamp `ShardHeader.n_minor` and
    /// `ShardStats.col_end` with the modality's own `n_vars` rather
    /// than the file-wide `header.n_vars` (which is the max across
    /// modalities). The existing 3-modality test uses uniform n_vars
    /// so it can't catch the bug.
    #[test]
    fn test_multimodal_shard_stats_use_per_modality_n_vars() {
        use crate::modality::ModalityType;
        use crate::reader::ScxReader;
        use crate::section::SectionType;
        use crate::shard::ShardHeader;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("per_modality_nvars.scx");

        // header.n_vars is the file-wide max across modalities.
        let mut header = sample_header();
        header.n_vars = 200;

        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs()).unwrap();

        // Three modalities with DISTINCT n_vars; "atac" matches the
        // header max, "rna" / "adt" do not. This ensures any path that
        // accidentally falls back to header.n_vars (= 200) gets caught
        // for the latter two.
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
        let atac_id = writer
            .add_modality(
                "atac",
                ModalityType::Atac,
                CodecId::None,
                ValueEncoding::Uint8,
                false,
            )
            .unwrap();

        let expected: [(u8, u64); 3] = [(rna_id, 30), (adt_id, 12), (atac_id, 200)];

        writer.write_var_for(rna_id, &sample_var()).unwrap();
        writer.write_var_for(adt_id, &sample_var()).unwrap();
        writer.write_var_for(atac_id, &sample_var()).unwrap();
        for (id, n_vars) in expected {
            writer.set_modality_n_vars(id, n_vars).unwrap();
        }

        let (indptr, indices, values) = sample_shard_data();
        for (id, _) in expected {
            writer
                .write_csr_shard_for(
                    id,
                    &indptr,
                    &indices,
                    &values,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    0,
                )
                .unwrap();
        }

        let final_path = writer.finish().unwrap();

        let reader = ScxReader::open(&final_path).unwrap();
        for (id, n_vars) in expected {
            let shards: Vec<&FullCatalogEntry> = reader
                .catalog()
                .shards(SectionType::CsrShard)
                .into_iter()
                .filter(|e| e.modality_id == id)
                .collect();
            assert_eq!(
                shards.len(),
                1,
                "modality_id {id} should have exactly 1 CSR shard"
            );
            let entry = shards[0];

            // Catalog stats: row-major shards stamp col_end = n_minor.
            let stats = entry
                .stats
                .as_ref()
                .expect("v2 catalog must carry shard stats");
            assert_eq!(
                stats.col_end, n_vars,
                "ShardStats.col_end for modality {id} should equal that \
                 modality's n_vars ({n_vars}), got {} (header.n_vars=200)",
                stats.col_end
            );
            assert_eq!(stats.col_start, 0);

            // On-disk shard header: n_minor field must also match.
            let bytes = reader.section_bytes(entry).unwrap();
            let sh = ShardHeader::read_from(&mut std::io::Cursor::new(
                &bytes[..crate::shard::SHARD_HEADER_SIZE],
            ))
            .unwrap();
            assert_eq!(
                sh.n_minor as u64, n_vars,
                "ShardHeader.n_minor for modality {id} should equal {n_vars}, \
                 got {}",
                sh.n_minor
            );
        }
    }

    /// Single-modality v2 file: no `add_modality` calls means no
    /// `ModalityTable` section is emitted. The on-disk shape and the
    /// reader-visible accessors match a v1 file.
    #[test]
    fn test_phase_b_single_modality_no_modality_table() {
        use crate::reader::ScxReader;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("single_modality.scx");
        let header = sample_header();

        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs()).unwrap();
        writer.write_var(&sample_var()).unwrap();
        let (indptr, indices, values) = sample_shard_data();
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
        let final_path = writer.finish().unwrap();

        let reader = ScxReader::open(&final_path).unwrap();
        assert!(!reader.is_multimodal());
        assert_eq!(reader.n_modalities(), 0);
        assert!(reader.modality_names().is_empty());
        assert!(!reader.header().has_modalities());
        assert_eq!(reader.header().modality_table_offset, 0);
        assert_eq!(reader.header().modality_table_length, 0);
        assert_eq!(reader.modality_info(0), None);
        assert_eq!(reader.modality_info(1), None);
        // Global accessors continue to work.
        assert_eq!(reader.read_var_for(0).unwrap().num_rows(), 2);
    }

    /// Per-modality CSC sidecars produce shard counts on the right
    /// modality's `ModalityInfo`, and the `BackedCscReader::for_modality`
    /// constructor scopes shard reads to that modality.
    #[test]
    fn test_phase_b_per_modality_csc() {
        use crate::backed::BackedCscReader;
        use crate::modality::ModalityType;
        use crate::reader::ScxReader;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multimodal_csc.scx");
        let header = sample_header();

        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs()).unwrap();
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
        writer.write_var_for(rna_id, &sample_var()).unwrap();
        writer.write_var_for(adt_id, &sample_var()).unwrap();
        writer.set_modality_n_vars(rna_id, 50).unwrap();
        writer.set_modality_n_vars(adt_id, 50).unwrap();

        let (indptr, indices, values) = sample_shard_data();
        // RNA gets a CSR shard only.
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
        // ADT gets BOTH CSR and CSC.
        writer
            .write_csr_shard_for(
                adt_id,
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        writer
            .write_csc_shard_for(
                adt_id,
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();

        let final_path = writer.finish().unwrap();
        let reader = ScxReader::open(&final_path).unwrap();

        // Per-modality counts reflect the writes.
        assert_eq!(reader.csc_shard_count_for(rna_id), 0);
        assert_eq!(reader.csc_shard_count_for(adt_id), 1);
        let adt_info = reader.modality_info(adt_id).unwrap();
        assert!(adt_info.flags.has_csc());
        let rna_info = reader.modality_info(rna_id).unwrap();
        assert!(!rna_info.flags.has_csc());

        // BackedCscReader scoped to RNA sees zero shards; scoped to
        // ADT sees the one shard. This is the cache-isolation
        // guarantee from B.5.
        let rna_csc =
            BackedCscReader::for_modality(ScxReader::open(&final_path).unwrap(), rna_id, 4)
                .unwrap();
        assert_eq!(rna_csc.n_shards(), 0);
        let adt_csc =
            BackedCscReader::for_modality(ScxReader::open(&final_path).unwrap(), adt_id, 4)
                .unwrap();
        assert_eq!(adt_csc.n_shards(), 1);
    }

    /// Phase B.6: `add_modality(..., build_csc=true)` triggers an
    /// auto-emit transpose pass at finish() time. After finish(), the
    /// modality's `n_csc_shards >= 1` and `flags.has_csc() == true`,
    /// even though the caller never invoked `write_csc_shard_for` —
    /// the writer read the CSR shards back from its temp file and
    /// streamed them through `streaming_csr_to_csc_iter_with_cap`.
    #[test]
    fn test_phase_b3_auto_emit_csc() {
        use crate::modality::ModalityType;
        use crate::reader::ScxReader;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auto_emit_csc.scx");
        let header = sample_header();
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs()).unwrap();

        let rna_id = writer
            .add_modality(
                "rna",
                ModalityType::Rna,
                CodecId::None,
                ValueEncoding::Uint8,
                true, // build_csc — Phase B.3 auto-emit
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
        writer.set_modality_n_vars(rna_id, 50).unwrap();
        writer.set_modality_n_vars(adt_id, 50).unwrap();
        writer.write_var_for(rna_id, &sample_var()).unwrap();
        writer.write_var_for(adt_id, &sample_var()).unwrap();

        // Both modalities get one CSR shard. Only `rna`'s
        // `build_csc=true`, so only its CSC sidecar should
        // auto-emit.
        let (indptr, indices, values) = sample_shard_data();
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
        writer
            .write_csr_shard_for(
                adt_id,
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();

        let final_path = writer.finish().unwrap();
        let reader = ScxReader::open(&final_path).unwrap();

        // RNA picked up the auto-emit; ADT did not.
        assert!(
            reader.csc_shard_count_for(rna_id) >= 1,
            "rna should have at least one auto-emitted CSC shard"
        );
        assert_eq!(
            reader.csc_shard_count_for(adt_id),
            0,
            "adt build_csc=false → no CSC sidecar"
        );
        assert!(reader.modality_info(rna_id).unwrap().flags.has_csc());
        assert!(!reader.modality_info(adt_id).unwrap().flags.has_csc());

        // The auto-emitted CSC stores the same nnz as the CSR. We
        // compare nnz rather than densifying because the CSR shape is
        // (n_shard_rows, n_modality_vars) while the CSC shape uses
        // file-wide n_obs (the column-axis slice covers the full obs
        // range, with zero rows for cells absent from the CSR shard).
        let csr = reader.read_all_csr_shards_for(rna_id).unwrap();
        let csc = reader.read_all_csc_shards_for(rna_id).unwrap();
        assert_eq!(
            *csr.indptr.last().unwrap_or(&0),
            *csc.indptr.last().unwrap_or(&0),
            "CSR and CSC nnz must agree after auto-emit"
        );
    }

    /// Phase B.4: per-modality CSC column-range reads return columns
    /// from the right modality only.
    #[test]
    fn test_phase_b4_read_csc_columns_for() {
        use crate::modality::ModalityType;
        use crate::reader::ScxReader;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("b4_csc_columns_for.scx");
        let header = sample_header();
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs()).unwrap();

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
        writer.set_modality_n_vars(rna_id, 50).unwrap();
        writer.set_modality_n_vars(adt_id, 50).unwrap();
        writer.write_var_for(rna_id, &sample_var()).unwrap();
        writer.write_var_for(adt_id, &sample_var()).unwrap();

        let (indptr, indices, values) = sample_shard_data();
        // Both modalities get one CSC shard at col_start=0.
        writer
            .write_csc_shard_for(
                rna_id,
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        writer
            .write_csc_shard_for(
                adt_id,
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();

        let final_path = writer.finish().unwrap();
        let reader = ScxReader::open(&final_path).unwrap();

        // Per-modality CSC counters reflect what was written.
        assert_eq!(reader.csc_shard_count_for(rna_id), 1);
        assert_eq!(reader.csc_shard_count_for(adt_id), 1);

        // Per-modality CSC range read returns the modality's
        // contribution. We assert the call succeeds and returns a
        // non-empty result (exact column-slice semantics are
        // covered by the single-modality `read_csc_columns` tests).
        let rna_cols = reader.read_csc_columns_for(rna_id, 0..3).unwrap();
        assert!(
            rna_cols.shape.1 >= 1,
            "rna CSC range read should return ≥ 1 col"
        );
        let rna_subset = reader
            .read_csc_columns_subset_for(rna_id, &[0u32, 2])
            .unwrap();
        assert!(rna_subset.shape.1 >= 1);
    }

    // -----------------------------------------------------------------------
    // Defensive tests (Patch 9): write_preencoded_shard CSC counting
    // -----------------------------------------------------------------------

    #[test]
    fn preencoded_csc_shard_increments_csc_count() {
        use crate::shard::{
            BlockIndex, BlockIndexEntry, ShardHeader, SHARD_HEADER_SIZE, SHARD_MAGIC,
        };
        use scx_codec::{CodecId, ValueEncoding};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("csc_preencoded.scx");

        let mut header = sample_header();
        header.n_obs = 3;
        header.n_vars = 2;
        header.nnz = 0;

        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs()).unwrap();
        writer.write_var(&sample_var()).unwrap();

        // First, write a normal CSR shard
        let (indptr, indices, values) = sample_shard_data();
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

        // Now craft a PreEncodedSection with section_type = CscShard
        let csc_indptr = vec![0u64, 1, 3]; // 2 columns
        let csc_indices = vec![0u32, 1, 2]; // 3 entries
        let csc_values: Vec<u8> = vec![10, 20, 30];

        let encoded = scx_codec::encode_shard(
            &csc_indptr,
            &csc_indices,
            &csc_values,
            CodecId::None,
            ValueEncoding::Uint8,
            true,
        )
        .unwrap();

        let block_index = BlockIndex {
            entries: vec![BlockIndexEntry::new(0, 2, 0, 0, 0, 3).unwrap()],
        };
        let mut bi_buf = Vec::new();
        block_index.write_to(&mut bi_buf).unwrap();

        let sh = ShardHeader {
            magic: SHARD_MAGIC,
            shard_format_version: 1,
            shard_type: 1, // CSC
            codec_id: CodecId::None as u8,
            value_encoding: ValueEncoding::Uint8 as u8,
            index_dtype: 0,
            reserved_flags: [0; 3],
            n_major: 2,
            n_minor: 3,
            nnz: 3,
            global_offset: 0,
            indptr_rel_offset: SHARD_HEADER_SIZE as u32,
            indptr_length: encoded.indptr_bytes.len() as u32,
            indices_rel_offset: SHARD_HEADER_SIZE as u32 + encoded.indptr_bytes.len() as u32,
            indices_length: encoded.indices_bytes.len() as u32,
            values_rel_offset: SHARD_HEADER_SIZE as u32
                + encoded.indptr_bytes.len() as u32
                + encoded.indices_bytes.len() as u32,
            values_length: encoded.values_bytes.len() as u32,
            block_index_rel_offset: SHARD_HEADER_SIZE as u32
                + encoded.indptr_bytes.len() as u32
                + encoded.indices_bytes.len() as u32
                + encoded.values_bytes.len() as u32,
            block_index_length: bi_buf.len() as u32,
            checksum: [0; 8], // dummy, we'll compute the real one
        };
        let mut hdr_buf = Vec::new();
        sh.write_to(&mut hdr_buf).unwrap();

        // Compute checksum from payload
        let mut payload = Vec::new();
        payload.extend_from_slice(&encoded.indptr_bytes);
        payload.extend_from_slice(&encoded.indices_bytes);
        payload.extend_from_slice(&encoded.values_bytes);
        payload.extend_from_slice(&bi_buf);
        let shard_checksum = crate::checksum::blake3_truncated_64(&payload);

        // Rewrite header with correct checksum
        let sh_corrected = ShardHeader {
            checksum: shard_checksum,
            ..sh
        };
        hdr_buf.clear();
        sh_corrected.write_to(&mut hdr_buf).unwrap();

        // Build full section for checksum
        let mut full_section = Vec::new();
        full_section.extend_from_slice(&hdr_buf);
        full_section.extend_from_slice(&payload);
        let section_checksum = crate::checksum::blake3_hash(&full_section);
        let section_length = full_section.len() as u64;

        let stats = compute_shard_stats(
            &csc_values,
            ValueEncoding::Uint8,
            MajorAxis::Col,
            0,
            2,
            3,
            3,
        );

        let pre = PreEncodedSection {
            encoded,
            block_index_bytes: bi_buf,
            header_buf: hdr_buf,
            section_checksum,
            section_length,
            stats,
            name: "X_csc_shard_0".to_string(),
            section_type: SectionType::CscShard,
            nnz: 3,
        };

        writer.write_preencoded_shard(pre).unwrap();
        let final_path = writer.finish().unwrap();

        // Verify the header now reports 1 CSC shard
        let reader = crate::reader::ScxReader::open(&final_path).unwrap();
        assert_eq!(
            reader.csc_shard_count(),
            1,
            "write_preencoded_shard should count CSC shards"
        );
        assert_eq!(reader.header().n_csr_shards, 1);
        assert_eq!(
            reader.header().n_csc_shards,
            1,
            "header n_csc_shards should reflect the preencoded CSC shard"
        );
        assert!(
            reader.header().has_csc(),
            "header has_csc flag should be set after writing a CSC shard"
        );
    }
}
