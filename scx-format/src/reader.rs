// ScxReader — mmap + pread paths (docs/architecture.md)

use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;

use arrow::array::RecordBatch;
use memmap2::Mmap;
use scx_codec::{CodecId, EncodedShardRef, ValueEncoding};
use scx_sparse::{ScxCsc, ScxCsr};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::catalog::{FullCatalog, FullCatalogEntry};
use crate::checksum::blake3_hash;
use crate::error::{Result, ScxError};
use crate::header::{FileHeader, HEADER_SIZE};
use crate::modality::{ModalityInfo, ModalityTable};
use crate::provenance::Provenance;
use crate::section::SectionType;
use crate::shard::{ShardHeader, SHARD_HEADER_SIZE};
use crate::RootCatalog;

/// Memory-mapped reader for SCX files.
///
/// Opens an SCX file, validates the header and catalog checksums,
/// and provides methods to read obs/var metadata, CSR shards, layers,
/// obsm embeddings, uns JSON, and provenance.
pub struct ScxReader {
    mmap: Mmap,
    header: FileHeader,
    root_catalog: RootCatalog,
    full_catalog: FullCatalog,
    /// `Some(table)` for v2 multimodal files; `None` for
    /// single-modality v2 files (`n_modalities == 0`) and all v1
    /// files. Parsed lazily-eagerly: the table is parsed once during
    /// `open()` so subsequent `modality_*` accessors are zero-cost.
    modality_table: Option<ModalityTable>,
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
            full_catalog,
            modality_table,
        })
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
        &self.full_catalog
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
        let slice = self.section_bytes(entry)?;
        let cursor = Cursor::new(slice);
        let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)?;
        // Read the first (and typically only) batch
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
        crate::arrow_compat::downcast_large_types(&batch)
    }

    /// Read the obs schema without deserializing the full RecordBatch.
    /// Uses the Arrow IPC footer to extract field names and types.
    pub fn read_obs_schema(&self) -> Result<arrow::datatypes::Schema> {
        let entry = self
            .full_catalog
            .get("obs")
            .ok_or_else(|| ScxError::SectionNotFound("obs".to_string()))?;
        self.read_arrow_ipc_schema(entry)
    }

    /// Read the var schema without deserializing the full RecordBatch.
    /// Uses the Arrow IPC footer to extract field names and types.
    pub fn read_var_schema(&self) -> Result<arrow::datatypes::Schema> {
        let entry = self
            .full_catalog
            .get("var")
            .ok_or_else(|| ScxError::SectionNotFound("var".to_string()))?;
        self.read_arrow_ipc_schema(entry)
    }

    /// Read the obs (observation) metadata as an Arrow RecordBatch.
    pub fn read_obs(&self) -> Result<RecordBatch> {
        let entry = self
            .full_catalog
            .get("obs")
            .ok_or_else(|| ScxError::SectionNotFound("obs".to_string()))?;
        self.read_arrow_ipc(entry)
    }

    /// Read the var (variable/gene) metadata as an Arrow RecordBatch.
    pub fn read_var(&self) -> Result<RecordBatch> {
        let entry = self
            .full_catalog
            .get("var")
            .ok_or_else(|| ScxError::SectionNotFound("var".to_string()))?;
        self.read_arrow_ipc(entry)
    }

    /// Read a named obsm embedding as an Arrow RecordBatch.
    pub fn read_obsm(&self, name: &str) -> Result<RecordBatch> {
        let key = format!("obsm/{name}");
        let entry = self
            .full_catalog
            .get(&key)
            .ok_or_else(|| ScxError::SectionNotFound(key))?;
        self.read_arrow_ipc(entry)
    }

    /// Read all obsm embeddings, keyed by name.
    pub fn read_all_obsm(&self) -> Result<HashMap<String, RecordBatch>> {
        let mut result = HashMap::new();
        for entry in &self.full_catalog.entries {
            if entry.section_type == SectionType::ObsmEmbedding {
                let name = entry
                    .name
                    .strip_prefix("obsm/")
                    .unwrap_or(&entry.name)
                    .to_string();
                let batch = self.read_arrow_ipc(entry)?;
                result.insert(name, batch);
            }
        }
        Ok(result)
    }

    /// Read a named varm embedding as an Arrow RecordBatch.
    pub fn read_varm(&self, name: &str) -> Result<RecordBatch> {
        let key = format!("varm/{name}");
        let entry = self
            .full_catalog
            .get(&key)
            .ok_or_else(|| ScxError::SectionNotFound(key))?;
        self.read_arrow_ipc(entry)
    }

    /// Read all varm embeddings, keyed by name.
    pub fn read_all_varm(&self) -> Result<HashMap<String, RecordBatch>> {
        let mut result = HashMap::new();
        for entry in &self.full_catalog.entries {
            if entry.section_type == SectionType::VarmEmbedding {
                let name = entry
                    .name
                    .strip_prefix("varm/")
                    .unwrap_or(&entry.name)
                    .to_string();
                let batch = self.read_arrow_ipc(entry)?;
                result.insert(name, batch);
            }
        }
        Ok(result)
    }

    /// Read all obsp pairwise sparse matrices (COO format), keyed by name.
    pub fn read_all_obsp(&self) -> Result<HashMap<String, RecordBatch>> {
        let mut result = HashMap::new();
        for entry in &self.full_catalog.entries {
            if entry.section_type == SectionType::ObspEmbedding {
                let name = entry
                    .name
                    .strip_prefix("obsp/")
                    .unwrap_or(&entry.name)
                    .to_string();
                let batch = self.read_arrow_ipc(entry)?;
                result.insert(name, batch);
            }
        }
        Ok(result)
    }

    /// Read all varp pairwise sparse matrices (COO format), keyed by name.
    pub fn read_all_varp(&self) -> Result<HashMap<String, RecordBatch>> {
        let mut result = HashMap::new();
        for entry in &self.full_catalog.entries {
            if entry.section_type == SectionType::VarpEmbedding {
                let name = entry
                    .name
                    .strip_prefix("varp/")
                    .unwrap_or(&entry.name)
                    .to_string();
                let batch = self.read_arrow_ipc(entry)?;
                result.insert(name, batch);
            }
        }
        Ok(result)
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
        let section_name = if modality_id == 0 {
            format!("obsm/{key}")
        } else {
            let mname = self.modality_name_for_id(modality_id)?;
            format!("obsm/{mname}/{key}")
        };
        let entry = self
            .full_catalog
            .get(&section_name)
            .ok_or_else(|| ScxError::SectionNotFound(section_name))?;
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
        let prefix = format!("{name}_shard_");
        let mut shards: Vec<&FullCatalogEntry> = self
            .full_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::LayerCsrShard && e.name.starts_with(&prefix))
            .collect();

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
        Provenance::read_from(&mut Cursor::new(slice))
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
        let dv = crate::deletion_vectors::DeletionVectors::read_from(&mut Cursor::new(slice))?;
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
            return Err(ScxError::ChecksumMismatch);
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
        ShardHeader::read_from(&mut Cursor::new(&section[..SHARD_HEADER_SIZE]))
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

    /// Read and decode a single shard without verifying checksums.
    ///
    /// Alias for [`read_shard_from_entry`] — both skip checksums.
    /// Retained for call-site clarity (e.g., in the training loader).
    #[deprecated(note = "call `read_shard_from_entry` (identical behavior) or \
                `read_shard_from_entry_verified` when per-shard checksum \
                verification is required.")]
    pub fn read_shard_from_entry_unchecked(
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

        // Parse shard header
        let sh = ShardHeader::read_from(&mut Cursor::new(&section[..SHARD_HEADER_SIZE]))?;

        // v2 strict shard_type validation: a v2 catalog must not carry
        // CSC entries with shard_type != 1. v1 catalogs preserve the
        // legacy catalog-wins tolerance (the writer hardcoded
        // shard_type = 0 for CSC pre-CSC-SUPPORT).
        if self.full_catalog.catalog_version >= 2 {
            sh.validate_csc_strict(entry.section_type)?;
        }

        // Extract encoded byte slices
        let indptr_bytes = &section[sh.indptr_rel_offset as usize..][..sh.indptr_length as usize];
        let indices_bytes =
            &section[sh.indices_rel_offset as usize..][..sh.indices_length as usize];
        let values_bytes = &section[sh.values_rel_offset as usize..][..sh.values_length as usize];
        let block_index_bytes =
            &section[sh.block_index_rel_offset as usize..][..sh.block_index_length as usize];

        if verify_checksum {
            // Verify shard checksum via streaming hasher (no payload Vec allocation)
            let mut shard_hasher = blake3::Hasher::new();
            shard_hasher.update(indptr_bytes);
            shard_hasher.update(indices_bytes);
            shard_hasher.update(values_bytes);
            shard_hasher.update(block_index_bytes);
            let hash = shard_hasher.finalize();
            let mut computed = [0u8; 8];
            computed.copy_from_slice(&hash.as_bytes()[..8]);
            if computed != sh.checksum {
                return Err(ScxError::ChecksumMismatch);
            }
        }

        // Resolve codec and encoding from shard header (NOT file header)
        let codec_id = CodecId::from_u8(sh.codec_id).ok_or(ScxError::UnknownCodec(sh.codec_id))?;
        let value_encoding = ValueEncoding::from_u8(sh.value_encoding)
            .ok_or(ScxError::UnknownValueEncoding(sh.value_encoding))?;
        let index_dtype_u16 = sh.index_dtype == 0;

        // Build EncodedShardRef (zero-copy from mmap) and decode directly to scipy types
        let encoded = EncodedShardRef {
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
        )?;

        Ok((indptr, indices, data))
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
        assert!(matches!(result.unwrap_err(), ScxError::ChecksumMismatch));
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
            ScxError::ChecksumMismatch
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
            assert!(SectionType::is_known(entry.section_type as u8));
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
}
