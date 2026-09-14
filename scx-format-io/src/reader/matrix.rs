//! The matrices: CSR and CSC shard decode, whole-matrix assembly, layers,
//! and `adata.raw`.
//!
//! One row-major assembler serves X, per-modality X, layers and raw --
//! see [`ScxReader::assemble_row_major`].

use super::*;

/// Which nouns the row-major assembler uses in its diagnostics. The `X` and
/// `adata.raw` families produce textually different errors, and the difference
/// is worth keeping: it tells a reader which matrix failed on a file that has
/// both.
#[derive(Clone, Copy)]
pub(crate) struct RowMajorLabels {
    /// Names an offending *catalog entry* ("... has no stats block").
    entry: &'static str,
    /// Names an offending *shard* ("... length mismatch"). `pub(crate)` so
    /// `typed_read.rs` can build its own error text from the same label.
    pub(crate) shard: &'static str,
}

/// Labels for `X`, a modality's `X`, and named layers.
pub(crate) const X_LABELS: RowMajorLabels = RowMajorLabels {
    entry: "shard entry",
    shard: "CSR shard",
};

/// Labels for the `adata.raw` matrix.
pub(crate) const RAW_LABELS: RowMajorLabels = RowMajorLabels {
    entry: "raw CSR shard",
    shard: "raw CSR shard",
};

/// One shard's exclusive slices of the merged output buffers: `(indptr,
/// indices, data)`. Carved by `split_at_mut` before any decode starts, which
/// is what lets the parallel path drop its raw-pointer scatter.
type ShardOutputSlices<'a> = (&'a mut [i64], &'a mut [i32], &'a mut [f32]);

/// Whether the row-major assembler fans its per-shard decode out across the
/// rayon pool.
///
/// A parameter rather than a `#[cfg]` inside the assembler so
/// `parallel_matches_sequential` can still run both against one file. After
/// the unification the two share everything but the iterator, so be precise
/// about what that differential now proves: not that two independent
/// implementations agree — they are one implementation — but that fanning the
/// writes out across threads does not reorder or overlap them, which is
/// exactly the property the `split_at_mut` carve-up is responsible for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RowMajorStrategy {
    /// Constructed by [`Self::for_build`] when the `parallel` feature is off,
    /// and by `parallel_matches_sequential`, which runs both arms against one
    /// file. With `parallel` on, production never picks it — hence the
    /// feature-conditional allow rather than a blanket one.
    #[cfg_attr(feature = "parallel", allow(dead_code))]
    Sequential,
    #[cfg(feature = "parallel")]
    Parallel,
}

impl RowMajorStrategy {
    /// What production reads use: parallel when the feature is on.
    pub(crate) fn for_build() -> Self {
        #[cfg(feature = "parallel")]
        {
            RowMajorStrategy::Parallel
        }
        #[cfg(not(feature = "parallel"))]
        {
            RowMajorStrategy::Sequential
        }
    }
}

/// Per-shard `(n_rows, nnz)` from catalog stats, without decoding anything.
///
/// One of the two halves the review's "three near-identical assemblers" finding
/// is really about: this `checked_sub` existed in four places (three assemblers
/// plus the typed reader's own), so a fix to it could land in three of four.
///
/// `checked_sub` and not a bare `-`: a corrupt catalog with
/// `row_end < row_start` would otherwise underflow-panic in debug or wrap to a
/// huge `usize` in release, driving a giant allocation.
pub(crate) fn plan_row_major_layout(
    shards: &[&FullCatalogEntry],
    labels: RowMajorLabels,
) -> Result<Vec<(usize, usize)>> {
    shards
        .iter()
        .map(|e| {
            let stats = e.stats.as_ref().ok_or_else(|| {
                ScxError::InvalidCatalog(format!(
                    "{} '{}' at offset {} has no stats block",
                    labels.entry, e.name, e.offset
                ))
            })?;
            let n_rows = stats.row_end.checked_sub(stats.row_start).ok_or_else(|| {
                ScxError::InvalidCatalog(format!(
                    "shard '{}' has row_end {} < row_start {}",
                    e.name, stats.row_end, stats.row_start
                ))
            })? as usize;
            Ok::<_, ScxError>((n_rows, stats.nnz as usize))
        })
        .collect()
}

/// The three decoded-vs-catalog length checks, in one place.
///
/// The other half of the duplication: twelve copies of these three messages
/// existed across four assembly bodies. Returned errors and not
/// `debug_assert!`, because they are reachable on a corrupt or stat-drifted
/// catalog and a mismatch would otherwise panic in the `copy_from_slice` /
/// `shard_ip[j + 1]` indexing that follows every caller — in release, where
/// `debug_assert!` is gone.
pub(crate) fn check_decoded_lengths(
    labels: RowMajorLabels,
    i: usize,
    n_rows: usize,
    nnz: usize,
    indptr_len: usize,
    indices_len: usize,
    values_len: usize,
) -> Result<()> {
    let kind = labels.shard;
    if indptr_len != n_rows + 1 {
        return Err(ScxError::InvalidCatalog(format!(
            "{kind} {i} indptr length mismatch: catalog stats say {}, decoded {indptr_len}",
            n_rows + 1,
        )));
    }
    if indices_len != nnz {
        return Err(ScxError::InvalidCatalog(format!(
            "{kind} {i} indices length mismatch: catalog stats say {nnz}, decoded {indices_len}"
        )));
    }
    if values_len != nnz {
        return Err(ScxError::InvalidCatalog(format!(
            "{kind} {i} data length mismatch: catalog stats say {nnz}, decoded {values_len}"
        )));
    }
    Ok(())
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
        let assembled = self.assemble_x_shards(&shards)?;

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
        let mut assembled = self.assemble_x_shards(&shards)?;
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
        self.assemble_x_shards(&shards)
    }

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

    /// Read all `adata.raw` CSR shards and concatenate along the row
    /// (obs) axis. The result's column count is the raw matrix's OWN
    /// `raw_n_vars` (recovered from the shards' stats), which may exceed
    /// the main matrix `n_vars`. Mirrors [`Self::read_all_csr_shards`]
    /// but over the raw section family.
    pub fn read_all_raw_csr_shards(&self) -> Result<ScxCsr> {
        #[cfg(debug_assertions)]
        self.debug_counts
            .read_all_raw_csr_shards
            .fetch_add(1, Ordering::Relaxed);
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

        // Same strategy as every other whole-matrix read. Until this was
        // unified, raw was the one path that stayed serial even in a parallel
        // build — not by decision, but because it was a third inlined copy
        // that nobody updated when the other two gained the rayon fan-out.
        self.assemble_row_major(
            &shards,
            raw_n_vars,
            (self.header.n_obs as usize, 0),
            RAW_LABELS,
            RowMajorStrategy::for_build(),
        )
    }

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
            let stats = entry.stats.as_ref().ok_or_else(|| {
                ScxError::InvalidCatalog(format!("CSC shard '{}' missing stats block", entry.name))
            })?;
            let shard_lo = stats.major_start(entry.section_type);
            let shard_hi = stats.major_end(entry.section_type);
            let lo_in_shard = c_lo.saturating_sub(shard_lo) as usize;
            let hi_in_shard = (c_hi.min(shard_hi).saturating_sub(shard_lo)) as usize;

            // A row-group-framed (v2) CSC shard is column-group indexed
            // by its `BlockIndex` (major axis = columns), so a gene-subset read
            // decodes only the touched column-groups instead of the whole shard.
            // `decode_block_index_row_runs` is axis-agnostic ("row" = major line);
            // one contiguous column run yields one CSC fragment (indptr over the
            // run's columns, indices = global row ids). Non-framed shards return
            // `None` → the full-decode + `col_slice` fallback below.
            if hi_in_shard > lo_in_shard {
                let header = self.read_shard_header(entry)?;
                if header.shard_format_version > crate::shard::DEFAULT_WRITE_SHARD_FORMAT_VERSION {
                    let run = (lo_in_shard, hi_in_shard - lo_in_shard);
                    if let Some(mut runs) = self.decode_block_index_row_runs(entry, &[run])? {
                        if let Some((indptr, indices, data)) = runs.pop() {
                            decoded.push(ScxCsc::new_unchecked(
                                (n_rows, hi_in_shard - lo_in_shard),
                                indptr,
                                indices,
                                data,
                            ));
                            continue;
                        }
                    }
                }
            }

            let csc = self.read_csc_from_entry(entry)?;
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
        // Freshness is enforced by `guard_csc_sidecar_fresh` inside the decode
        // entry point below, not here — see that function for why the CSC
        // wrapper is the wrong place for it.
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

    /// List unique layer names from LayerCsrShard entries.
    ///
    /// Delegates to [`FullCatalog::layer_names`] so the cloud reader shares
    /// the same logic.
    pub fn layer_names(&self) -> Vec<String> {
        self.full_catalog.layer_names()
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
        self.assemble_x_shards(&shards)
    }

    pub(crate) fn legacy_layer_shards(&self, name: &str) -> Vec<&FullCatalogEntry> {
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

    /// Decode several shard-local row `runs` (`(row_start, n_rows)`) of a
    /// **row-group-framed (v2)** shard by resolving the `BlockIndex`, decoding
    /// only the touched row-groups (each at most once), and slicing the requested
    /// rows out of them. Returns `Ok(None)` for a non-framed (v1) shard so the
    /// caller falls back to a full-shard decode. Each run's result is a run-local
    /// CSR (`indptr[0] == 0`) byte-identical to the matching slice of a full
    /// decode. Cost is O(touched-groups + touched-rows), not O(shard).
    ///
    /// This is the **uncached** form: the layout is resolved and the groups are
    /// decoded per call and dropped. The CSR gather path in `BackedCsrReader`
    /// memoizes the layout per shard and retains the groups in its LRU
    /// ([`super::FramedShardLayout`], OPT-FORMATIO-1); the column-major
    /// (`read_csc_columns`) callers and tests use this one.
    pub fn decode_block_index_row_runs(
        &self,
        entry: &FullCatalogEntry,
        runs: &[(usize, usize)],
    ) -> Result<Option<Vec<scx_codec::ScipyShard>>> {
        let Some(layout) = self.framed_shard_layout(entry)? else {
            return Ok(None);
        };
        for &(run_start, run_len) in runs {
            layout.check_run(run_start, run_len)?;
        }
        // Each touched group is decoded once per call and shared by every run
        // that lands in it.
        let mut groups: HashMap<usize, Arc<ScxCsr>> = HashMap::new();
        let mut out = Vec::with_capacity(runs.len());
        for &(run_start, run_len) in runs {
            out.push(super::assemble_row_run(&layout, run_start, run_len, |g| {
                if let Some(rg) = groups.get(&g) {
                    return Ok(Arc::clone(rg));
                }
                let rg = Arc::new(self.decode_framed_row_group(&layout, g)?);
                groups.insert(g, Arc::clone(&rg));
                Ok(rg)
            })?);
        }
        Ok(Some(out))
    }

    /// Reject a decode of a column-major shard whose sidecar was built against
    /// an earlier generation of the CSR data (review §4.7).
    ///
    /// Keyed on the **entry's section type**, so the CSR side of a file with a
    /// stale sidecar still reads: staleness is a statement about the sidecar,
    /// not about the file.
    ///
    /// # Why here and not on the CSC read wrappers
    ///
    /// It was on `read_csc_from_entry` first, on the reasoning that every
    /// `ScxReader` CSC read funnels through it. Two things funnel around it:
    ///
    /// - the framed (v2) arm of [`Self::read_csc_columns`] decodes through
    ///   [`Self::decode_block_index_row_runs`] and `continue`s, so the whole
    ///   gene-subset scatter path — the hottest CSC read there is — skipped it;
    /// - `scx upgrade` copies a sidecar forward with
    ///   [`Self::read_shard_from_entry`] and then re-stamps it at the output's
    ///   generation, which converts a *detectable* stale sidecar into an
    ///   undetectable one. That is strictly worse than not checking.
    ///
    /// So the guard belongs at the point where shard payload becomes values,
    /// which is the four functions below. `section_bytes` would be the broader
    /// chokepoint but is deliberately not used: `scx info` / `scx validate`
    /// must still be able to inspect and checksum a file whose sidecar is
    /// stale, and a byte fetch is not a decode.
    pub(super) fn guard_csc_sidecar_fresh(&self, entry: &FullCatalogEntry) -> Result<()> {
        if crate::shard::is_column_major(entry.section_type)
            && !self.full_catalog.csc_sidecar_is_fresh()
        {
            return Err(ScxError::StaleCscSidecar {
                built_generation: self.full_catalog.csc_build_generation,
                data_generation: self.full_catalog.data_generation,
            });
        }
        Ok(())
    }

    fn read_shard_from_entry_inner(
        &self,
        entry: &FullCatalogEntry,
        verify_checksum: bool,
    ) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
        self.guard_csc_sidecar_fresh(entry)?;
        #[cfg(debug_assertions)]
        self.debug_counts
            .read_shard_from_entry
            .fetch_add(1, Ordering::Relaxed);
        // `io` bucket: raw byte fetch. On a local mmap reader this is an O(1)
        // bounds-checked slice (≈0); the page-fault cost of touching the bytes
        // is attributed to `decode` below. On a cloud/range-read path the fetch
        // is real. See `crate::profile` module docs.
        let io_start = crate::profile::start();
        let section = self.section_bytes(entry)?;
        crate::profile::record_io_since(io_start, section.len());
        // `decode` bucket: resolve the codec class only when profiling, parsing
        // the header out of the section bytes we already fetched (no second
        // `section_bytes`/`read_shard_header` round-trip — that would be
        // untracked I/O and inflate the observer effect on the oracle wall).
        let class = if crate::profile::profile_enabled() {
            crate::validated_section::ValidatedSection::new(section)
                .header()
                .ok()
                .and_then(|h| ShardHeader::read_from(&mut Cursor::new(h)).ok())
                .and_then(|h| CodecId::from_u8(h.codec_id))
                .map(crate::profile::CodecClass::from_codec)
                .unwrap_or(crate::profile::CodecClass::Generic)
        } else {
            crate::profile::CodecClass::Generic
        };
        let decode_start = crate::profile::start();
        let decoded = crate::shard_decode::decode_shard_bytes(
            section,
            entry,
            self.full_catalog.catalog_version,
            verify_checksum,
        );
        crate::profile::record_decode_since(class, decode_start, section.len());
        decoded
    }

    /// Read and decode a single shard to **native** types (`i64` indptr, `u32`
    /// indices, [`scx_codec::ShardValuesNative`] values) — the in-assembly narrow
    /// twin of [`read_shard_from_entry`](Self::read_shard_from_entry). Integer
    /// values stay `u32` (never rounded through `f32`); float values are `f32`.
    /// Used by the typed whole-matrix reader in `typed_read.rs` and by the query
    /// engine's typed collect (via `SectionReader::read_shard_from_entry_native`),
    /// which is why this is `pub` rather than `pub(crate)`.
    pub fn read_shard_from_entry_native(
        &self,
        entry: &FullCatalogEntry,
    ) -> Result<(Vec<i64>, Vec<u32>, scx_codec::ShardValuesNative)> {
        self.guard_csc_sidecar_fresh(entry)?;
        // Instrumented identically to the f32 twin above. It was not, and the
        // omission made a typed read report zero shard decodes and zero I/O —
        // a diagnostic that lies rather than one that is merely absent, since
        // `debug_counts` is what "did we read the whole table" assertions read.
        #[cfg(debug_assertions)]
        self.debug_counts
            .read_shard_from_entry
            .fetch_add(1, Ordering::Relaxed);
        let io_start = crate::profile::start();
        let section = self.section_bytes(entry)?;
        crate::profile::record_io_since(io_start, section.len());
        let class = if crate::profile::profile_enabled() {
            crate::validated_section::ValidatedSection::new(section)
                .header()
                .ok()
                .and_then(|h| ShardHeader::read_from(&mut Cursor::new(h)).ok())
                .and_then(|h| CodecId::from_u8(h.codec_id))
                .map(crate::profile::CodecClass::from_codec)
                .unwrap_or(crate::profile::CodecClass::Generic)
        } else {
            crate::profile::CodecClass::Generic
        };
        let decode_start = crate::profile::start();
        let decoded = crate::shard_decode::decode_shard_bytes_native(
            section,
            entry,
            self.full_catalog.catalog_version,
            false,
        );
        crate::profile::record_decode_since(class, decode_start, section.len());
        decoded
    }

    /// Read only the indptr (row-pointer) array of a shard, skipping
    /// indices/data decode entirely. For callers that need just the
    /// per-row nnz counts (e.g. the streaming SCX → h5ad export's
    /// `precompute_total_nnz` when a deletion vector is active).
    pub fn read_shard_indptr_from_entry(&self, entry: &FullCatalogEntry) -> Result<Vec<i64>> {
        self.guard_csc_sidecar_fresh(entry)?;
        let section = self.section_bytes(entry)?;
        crate::shard_decode::decode_shard_indptr_bytes(
            section,
            entry,
            self.full_catalog.catalog_version,
        )
    }

    /// Assemble a set of row-major CSR shard entries into one [`ScxCsr`].
    ///
    /// The single implementation behind every whole-matrix read: `X`, a
    /// per-modality `X`, a named layer, and (since the raw migration) the
    /// `adata.raw` matrix. It replaced three hand-rolled copies that each
    /// carried their own `checked_sub` and their own three decoded-vs-catalog
    /// length checks — the shape where a bounds fix lands in two of three
    /// places.
    ///
    /// `n_cols` is passed rather than read off the header because the raw
    /// matrix has its own, independent column count. `empty_shape` is passed
    /// for the same reason and is not derivable from `n_cols`: an empty `X`
    /// read answers `(0, n_vars)` while an empty raw read answers
    /// `(n_obs, 0)`, and collapsing the two is a silent change to what a
    /// caller gets back for a file with no shards.
    ///
    /// # Why there is no `unsafe` here
    ///
    /// The parallel path used to launder three base pointers through `usize`
    /// and rebuild `&mut` slices inside each rayon task, guarded by two
    /// release-mode `assert!`s whose own comment called them "the ONLY thing
    /// keeping the unsafe block below from writing past the allocated region".
    /// Pre-splitting the output buffers with [`slice::split_at_mut`] hands
    /// each task an exclusive slice the borrow checker has already proved
    /// disjoint, at no cost — same slices, same writes — and takes both the
    /// `unsafe` and the two panics-on-malformed-input with it.
    pub(super) fn assemble_row_major(
        &self,
        shards: &[&FullCatalogEntry],
        n_cols: usize,
        empty_shape: (usize, usize),
        labels: RowMajorLabels,
        strategy: RowMajorStrategy,
    ) -> Result<ScxCsr> {
        if shards.is_empty() {
            return Ok(ScxCsr::new_unchecked(empty_shape, vec![0], vec![], vec![]));
        }

        // Hint aggressive readahead across the shard region.
        // Use min/max of file offsets since shards are sorted by row_start,
        // not file offset — they may not be contiguous after append/compact.
        //
        // Entirely checked, and silent on failure. This runs on raw catalog
        // values *before* `section_bytes` has had a chance to reject them, so
        // `offset + length` on a hostile catalog overflowed `u64` (panic in
        // debug, wrap in release, then an underflowing `max_end - min_offset`)
        // — a panic on malformed input, in a reader. A readahead hint is
        // advisory, so skipping it is the correct degradation: the bad entry is
        // still rejected a moment later by `section_bytes`, which is where an
        // out-of-range offset should surface.
        #[cfg(unix)]
        {
            let min_offset = shards.iter().map(|e| e.offset).min().unwrap_or(0);
            let max_end = shards
                .iter()
                .filter_map(|e| e.offset.checked_add(e.length))
                .max()
                .unwrap_or(0);
            if let (Ok(start), Some(len)) = (
                usize::try_from(min_offset),
                max_end
                    .checked_sub(min_offset)
                    .and_then(|n| usize::try_from(n).ok()),
            ) {
                self.advise_sequential(start, len);
            }
        }

        let shard_sizes = plan_row_major_layout(shards, labels)?;
        let total_rows: usize = shard_sizes.iter().map(|(r, _)| *r).sum();
        let total_nnz: usize = shard_sizes.iter().map(|(_, n)| *n).sum();

        // Per-shard cumulative nnz, needed to rebase each shard's indptr.
        let mut nnz_offsets = Vec::with_capacity(shard_sizes.len());
        let mut cum_nnz = 0usize;
        for &(_, nnz) in &shard_sizes {
            nnz_offsets.push(cum_nnz);
            cum_nnz += nnz;
        }

        // Single allocation for the final merged arrays.
        let mut indptr = vec![0i64; total_rows + 1];
        let mut indices = vec![0i32; total_nnz];
        let mut data = vec![0f32; total_nnz];

        // Carve the outputs into per-shard exclusive slices up front. Shard 0
        // takes `n_rows + 1` indptr slots (it owns the leading 0); every later
        // shard takes `n_rows`, landing it at `row_offset + 1` — the same
        // regions the pointer arithmetic used to compute. The sizes sum to the
        // buffer lengths exactly, by construction of `total_rows` / `total_nnz`
        // above, so the splits below cannot fail; `expect` rather than a
        // silent `split_at_mut` panic in case that ever stops being true.
        let mut chunks: Vec<ShardOutputSlices<'_>> = Vec::with_capacity(shard_sizes.len());
        {
            let mut ip_rest: &mut [i64] = &mut indptr;
            let mut ix_rest: &mut [i32] = &mut indices;
            let mut d_rest: &mut [f32] = &mut data;
            for (i, &(n_rows, nnz)) in shard_sizes.iter().enumerate() {
                let ip_take = if i == 0 { n_rows + 1 } else { n_rows };
                let (ip_head, ip_tail) = ip_rest
                    .split_at_mut_checked(ip_take)
                    .ok_or_else(|| Self::split_overflow(labels, i, "indptr"))?;
                let (ix_head, ix_tail) = ix_rest
                    .split_at_mut_checked(nnz)
                    .ok_or_else(|| Self::split_overflow(labels, i, "indices"))?;
                let (d_head, d_tail) = d_rest
                    .split_at_mut_checked(nnz)
                    .ok_or_else(|| Self::split_overflow(labels, i, "data"))?;
                chunks.push((ip_head, ix_head, d_head));
                ip_rest = ip_tail;
                ix_rest = ix_tail;
                d_rest = d_tail;
            }
        }

        let decode_into =
            |(i, (ip_out, ix_out, d_out)): (usize, ShardOutputSlices<'_>)| -> Result<()> {
                let (n_rows, nnz) = shard_sizes[i];
                let (shard_ip, shard_ix, shard_data) = self.read_shard_from_entry(shards[i])?;
                check_decoded_lengths(
                    labels,
                    i,
                    n_rows,
                    nnz,
                    shard_ip.len(),
                    shard_ix.len(),
                    shard_data.len(),
                )?;

                ix_out.copy_from_slice(&shard_ix);
                d_out.copy_from_slice(&shard_data);

                // Shard 0 owns indptr[0] and copies its decoded indptr verbatim
                // (its nnz offset is 0); every later shard copies `[1..]` rebased
                // by the running nnz.
                if i == 0 {
                    ip_out.copy_from_slice(&shard_ip);
                } else {
                    let nnz_off_i64 = nnz_offsets[i] as i64;
                    for (j, slot) in ip_out.iter_mut().enumerate() {
                        *slot = shard_ip[j + 1] + nnz_off_i64;
                    }
                }
                Ok(())
            };

        match strategy {
            #[cfg(feature = "parallel")]
            RowMajorStrategy::Parallel => chunks
                .into_par_iter()
                .enumerate()
                .try_for_each(decode_into)?,
            RowMajorStrategy::Sequential => {
                chunks.into_iter().enumerate().try_for_each(decode_into)?
            }
        }

        let n_rows = indptr.len().saturating_sub(1);
        Ok(ScxCsr::new_unchecked(
            (n_rows, n_cols),
            indptr,
            indices,
            data,
        ))
    }

    /// Unreachable given the prefix sums above, but a returned error rather
    /// than a `split_at_mut` panic: readers return errors on malformed input.
    fn split_overflow(labels: RowMajorLabels, i: usize, buffer: &str) -> ScxError {
        ScxError::InvalidCatalog(format!(
            "{} {i}: {buffer} output region exceeds the allocation implied by catalog stats",
            labels.shard
        ))
    }

    /// Assemble row-major `X`-family shards (X, a modality's X, or a layer)
    /// with this build's default strategy.
    fn assemble_x_shards(&self, shards: &[&FullCatalogEntry]) -> Result<ScxCsr> {
        let n_vars = self.header.n_vars as usize;
        self.assemble_row_major(
            shards,
            n_vars,
            (0, n_vars),
            X_LABELS,
            RowMajorStrategy::for_build(),
        )
    }
}
