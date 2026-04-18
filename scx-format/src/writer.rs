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
use crate::section::{align_to_8, SectionType};
use crate::shard::{BlockIndex, BlockIndexEntry, ShardHeader, SHARD_HEADER_SIZE};

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
    tmp_path: PathBuf,
    file: Option<BufWriter<File>>,
    current_offset: u64,
    header: FileHeader,
    entries: Vec<FullCatalogEntry>,
    csr_shard_count: u32,
    csc_shard_count: u32,
    total_nnz: u64,
    has_obsm: bool,
    has_obsp: bool,
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
        let tmp_path = PathBuf::from(format!(
            "{}.tmp.{}",
            final_path.display(),
            std::process::id()
        ));

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;
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
            tmp_path,
            file: Some(writer),
            current_offset: SECTIONS_START_OFFSET,
            header,
            entries: Vec::new(),
            csr_shard_count: 0,
            csc_shard_count: 0,
            total_nnz: 0,
            has_obsm: false,
            has_obsp: false,
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
            stats,
        });

        Ok(())
    }

    /// Serialize a RecordBatch to Arrow IPC file format bytes.
    fn write_arrow_ipc(batch: &RecordBatch) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        {
            let mut writer = arrow::ipc::writer::FileWriter::try_new(&mut buf, batch.schema_ref())?;
            writer.write(batch)?;
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
            shard_type: 0, // CSR
            codec_id: codec_id as u8,
            value_encoding: value_encoding as u8,
            index_dtype: self.header.index_dtype,
            reserved_flags: [0; 3],
            n_major,
            n_minor: {
                if self.header.n_vars > u32::MAX as u64 {
                    return Err(ScxError::NVarsOverflow(self.header.n_vars));
                }
                self.header.n_vars as u32
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

        // Compute shard stats from raw values
        let stats = compute_shard_stats(values, value_encoding, row_start, n_major as u64, nnz);

        self.entries.push(FullCatalogEntry {
            name: name.to_string(),
            offset: shard_global_offset,
            length: section_length,
            section_type,
            checksum: section_checksum,
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
            stats: Some(stats),
        });

        if section_type == SectionType::CsrShard {
            self.csr_shard_count += 1;
            self.total_nnz += nnz;
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
            stats: Some(section.stats),
        });

        if section.section_type == SectionType::CsrShard {
            self.csr_shard_count += 1;
            self.total_nnz += section.nnz;
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

    /// Finalize the file: write catalogs, header, fsync, atomic rename.
    ///
    /// Returns the final file path on success.
    pub fn finish(mut self) -> Result<PathBuf> {
        // Flush BufWriter and take inner File
        let buf_writer = self.file.take().ok_or(ScxError::WriterAlreadyFinished)?;
        let mut file = buf_writer.into_inner().map_err(std::io::Error::from)?;

        // 1. Write full catalog at EOF
        let aligned_offset = align_to_8(self.current_offset);
        let pad = (aligned_offset - self.current_offset) as usize;
        if pad > 0 {
            file.write_all(&vec![0u8; pad])?;
        }

        let full_catalog_offset = aligned_offset;

        let full_catalog = FullCatalog {
            catalog_version: 1,
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

        // 8. Atomic rename
        std::fs::rename(&self.tmp_path, &self.final_path)?;

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
        // Close file handle first
        drop(self.file.take());
        // Remove temp file (harmless no-op after successful finish+rename)
        let _ = std::fs::remove_file(&self.tmp_path);
    }
}

/// Compute shard statistics from raw value bytes.
pub fn compute_shard_stats(
    values: &[u8],
    value_encoding: ValueEncoding,
    row_start: u64,
    n_rows: u64,
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

    ShardStats {
        row_start,
        row_end: row_start + n_rows,
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
            format_version: 1,
            header_length: 256,
            flags: 0,
            n_obs: 100,
            n_vars: 50,
            nnz: 500,
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
            reserved: [0u8; 132],
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
        assert_eq!(hdr.format_version, 1);
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

    /// 10.15: Temp file lifecycle — tmp exists before finish, final after
    #[test]
    fn test_temp_file_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lifecycle.scx");
        let tmp_path = PathBuf::from(format!("{}.tmp.{}", path.display(), std::process::id()));

        let header = sample_header();
        let mut writer = ScxWriter::new(&path, header).unwrap();

        // Tmp file exists, final doesn't
        assert!(tmp_path.exists());
        assert!(!path.exists());

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

        // Final exists, tmp doesn't
        assert!(path.exists());
        assert!(!tmp_path.exists());
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

    /// Test Drop cleans up temp file when finish() is not called
    #[test]
    fn test_drop_cleans_up_temp() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dropped.scx");
        let tmp_path = PathBuf::from(format!("{}.tmp.{}", path.display(), std::process::id()));

        {
            let _writer = ScxWriter::new(&path, sample_header()).unwrap();
            assert!(tmp_path.exists());
        }
        // After drop
        assert!(!tmp_path.exists());
        assert!(!path.exists());
    }

    /// Test compute_shard_stats
    #[test]
    fn test_compute_shard_stats() {
        let values: Vec<u8> = vec![5, 10, 1, 3, 7, 2];
        let stats = compute_shard_stats(&values, ValueEncoding::Uint8, 0, 3, 6);
        assert_eq!(stats.row_start, 0);
        assert_eq!(stats.row_end, 3);
        assert_eq!(stats.nnz, 6);
        assert_eq!(stats.value_min, 1);
        assert_eq!(stats.value_max, 10);
        assert_eq!(stats.value_sum, 28); // 5+10+1+3+7+2
    }

    /// Test compute_shard_stats for Float32 returns zero stats
    #[test]
    fn test_compute_shard_stats_float32() {
        // Float32: 3 values as LE bytes (1.0f32, 2.5f32, 0.5f32)
        let mut values = Vec::new();
        values.extend_from_slice(&1.0f32.to_le_bytes());
        values.extend_from_slice(&2.5f32.to_le_bytes());
        values.extend_from_slice(&0.5f32.to_le_bytes());
        let stats = compute_shard_stats(&values, ValueEncoding::Float32, 0, 2, 3);
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
        let stats = compute_shard_stats(&values, ValueEncoding::Float16, 10, 5, 3);
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
}
