//! Shard-range index: the O(log n) row-range → shard lookup every backed
//! reader is built on.
//!
//! [`BackedCsrIndex`] is shared by the CSR reader and the dense (obsm) reader —
//! a dense mapping is row-sharded on the same axis, so the same binary search
//! serves both.

use super::*;

// ---------------------------------------------------------------------------
// ShardEntryLite — internal per-shard row
// ---------------------------------------------------------------------------

/// Per-shard catalog row retained by `BackedCsrReader` for read-path
/// dispatch. Drops the `String` name and 32-byte BLAKE3 `checksum`
/// from `FullCatalogEntry`, plus the `value_*` / `col_*` /
/// `column_stats` fields the read path never reads. The remaining
/// 25 bytes (with alignment padding to 32) carry exactly what
/// `read_shard_from_entry`, the MADV_WILLNEED prefetch, and
/// `total_nnz` consume.
///
/// The `to_anndata_backed` worker amplification — N+3 reader opens
/// across W workers × thousands of catalog entries — used to clone
/// ~250-byte `FullCatalogEntry` values per shard. With `ShardEntryLite`
/// the per-shard retained footprint drops ~8× and the per-entry
/// `String` / `Vec<ColumnStat>` allocations disappear from the
/// construction path entirely.
#[derive(Debug, Clone, Copy)]
pub(super) struct ShardEntryLite {
    pub(super) offset: u64,
    pub(super) length: u64,
    /// `nnz` for `total_nnz()` aggregation. Stored even though
    /// `read_shard_from_entry` doesn't need it — it's cheaper to
    /// retain the `u64` than to re-walk the catalog when total_nnz
    /// is called.
    pub(super) nnz: u64,
    pub(super) section_type: SectionType,
    pub(super) modality_id: u8,
}

impl ShardEntryLite {
    /// Build from a `CatalogView` entry whose `stats` is `Some`. The
    /// view's `ShardStatsLite` already carries the dispatched major
    /// axis range; we only need `nnz` here, since row-range lookups
    /// go through `BackedCsrIndex`.
    pub(super) fn from_view_entry(e: &CatalogViewEntry) -> Self {
        Self {
            offset: e.offset,
            length: e.length,
            nnz: e.stats.as_ref().map_or(0, |s| s.nnz),
            section_type: e.section_type,
            modality_id: e.modality_id,
        }
    }

    /// Synthesise a transient `FullCatalogEntry` for the few reader
    /// APIs (`ScxReader::read_shard_from_entry`,
    /// `ScxReader::section_bytes`) that still take it. `String::new()`
    /// is heap-free and `[0u8; 32]` is a stack array — total cost is
    /// a small stack copy per call, with no allocation. Used by
    /// `read_shard_uncached` and `decode_and_cache` to bridge into
    /// the existing reader API without paying for retained
    /// `FullCatalogEntry` clones.
    pub(super) fn into_transient_full_entry(self) -> FullCatalogEntry {
        FullCatalogEntry {
            name: String::new(),
            offset: self.offset,
            length: self.length,
            section_type: self.section_type,
            checksum: [0u8; 32],
            modality_id: self.modality_id,
            stats: None,
        }
    }
}

// ---------------------------------------------------------------------------
// BackedCsrIndex
// ---------------------------------------------------------------------------

/// Precomputed shard index for O(log n) row-range lookups.
///
/// Built once from a [`FullCatalog`] at open time.  Each entry stores
/// `row_start`, `row_end`, and a `sorted_shard_idx` — the position of the
/// shard after sort-by-`row_start` and filter-by-section-type, **not** the
/// original index in the catalog's `entries` vec. This distinction matters
/// because `BackedCsrReader::read_shard_cached` indexes
/// `self.sorted_entries[sorted_shard_idx]`, which is already in that sorted
/// order.
#[derive(Debug, Clone, Copy)]
pub(super) struct ShardRange {
    row_start: u64,
    row_end: u64,
    /// Position in the sorted-and-filtered shard list (not the raw catalog
    /// entry index).
    sorted_shard_idx: usize,
}

/// Precomputed shard index for O(log n) row-range lookups.
///
/// Built once from a [`FullCatalog`] at open time.
#[derive(Debug, Clone)]
pub struct BackedCsrIndex {
    /// Sorted by `row_start`.
    shard_ranges: Vec<ShardRange>,
}

impl BackedCsrIndex {
    /// Build from a [`FullCatalog`].
    ///
    /// Extracts CSR shard entries, sorts by `row_start`, and records their
    /// position in the sorted order (which is the index used by
    /// `ScxReader::read_csr_shard`).
    pub fn from_catalog(catalog: &FullCatalog) -> Self {
        Self::from_catalog_filtered(catalog, SectionType::CsrShard, None)
    }

    /// Build from a [`FullCatalog`] for a specific layer.
    ///
    /// Extracts `LayerCsrShard` entries whose name starts with
    /// `"{layer_name}_shard_"`, sorts by `row_start`, and records their
    /// position in sorted order.
    pub fn from_catalog_layer(catalog: &FullCatalog, layer_name: &str) -> Self {
        let prefix = format!("{layer_name}_shard_");
        Self::from_catalog_filtered(catalog, SectionType::LayerCsrShard, Some(&prefix))
    }

    /// Phase B.5 / D.4: build from a [`FullCatalog`] for a specific
    /// modality. Mirrors `BackedCscIndex::from_catalog_for_modality`
    /// — filters CSR entries by `(SectionType::CsrShard, modality_id)`
    /// and indexes the per-modality position-to-row mapping.
    pub fn from_catalog_for_modality(catalog: &FullCatalog, modality_id: u8) -> Self {
        let mut shard_entries: Vec<ShardRange> = catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CsrShard && e.modality_id == modality_id)
            .filter_map(|e| {
                e.stats.as_ref().map(|s| ShardRange {
                    row_start: s.row_start,
                    row_end: s.row_end,
                    sorted_shard_idx: 0,
                })
            })
            .collect();
        shard_entries.sort_by_key(|r| r.row_start);
        for (i, entry) in shard_entries.iter_mut().enumerate() {
            entry.sorted_shard_idx = i;
        }
        BackedCsrIndex {
            shard_ranges: shard_entries,
        }
    }

    /// Internal: build from catalog filtering by section type and optional name prefix.
    fn from_catalog_filtered(
        catalog: &FullCatalog,
        section_type: SectionType,
        name_prefix: Option<&str>,
    ) -> Self {
        let mut shard_entries: Vec<ShardRange> = catalog
            .entries
            .iter()
            .filter(|e| {
                e.section_type == section_type && name_prefix.is_none_or(|p| e.name.starts_with(p))
            })
            .filter_map(|e| {
                e.stats.as_ref().map(|s| ShardRange {
                    row_start: s.row_start,
                    row_end: s.row_end,
                    sorted_shard_idx: 0, // filled below
                })
            })
            .collect();

        // Sort by row_start (deterministic ordering)
        shard_entries.sort_by_key(|r| r.row_start);

        // Assign sorted indices
        for (i, entry) in shard_entries.iter_mut().enumerate() {
            entry.sorted_shard_idx = i;
        }

        BackedCsrIndex {
            shard_ranges: shard_entries,
        }
    }

    /// Build the row-range index directly from a pre-sorted
    /// `&[&CatalogViewEntry]`. The caller is responsible for sorting
    /// by `stats.major_start`; we just zip the row range pair into
    /// `ShardRange` and stamp the sequential `sorted_shard_idx`. This
    /// is the function that pairs with the `ShardEntryLite::from_view_entry`
    /// builder — both index and lightweight entry list are produced
    /// in a single pass over the catalog view.
    pub(super) fn from_view_sorted(sorted_view_entries: &[&CatalogViewEntry]) -> Self {
        let shard_ranges: Vec<ShardRange> = sorted_view_entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                e.stats.as_ref().map(|s| ShardRange {
                    row_start: s.major_start,
                    row_end: s.major_end,
                    sorted_shard_idx: i,
                })
            })
            .collect();
        BackedCsrIndex { shard_ranges }
    }

    /// Build the index directly from pre-sorted `(row_start, row_end)`
    /// pairs. Used by [`BackedDenseReader`], whose obsm shard catalog
    /// entries carry no `stats` block (row ranges come from each shard's
    /// Arrow schema metadata, not the catalog stats the `from_catalog*` /
    /// `from_view_sorted` constructors read). The caller must pass the
    /// ranges already sorted by `row_start`; `sorted_shard_idx` is
    /// stamped sequentially to index the caller's sorted shard-entry list.
    pub fn from_ranges(ranges: &[(u64, u64)]) -> Self {
        let shard_ranges = ranges
            .iter()
            .enumerate()
            .map(|(i, &(row_start, row_end))| ShardRange {
                row_start,
                row_end,
                sorted_shard_idx: i,
            })
            .collect();
        BackedCsrIndex { shard_ranges }
    }

    /// Number of shards in the index.
    pub fn n_shards(&self) -> usize {
        self.shard_ranges.len()
    }

    /// Find all shard indices that overlap `[row_start, row_end)`.
    ///
    /// Uses binary search — O(log n) in the number of shards.
    /// Returns a `Vec<usize>` of sorted shard indices.
    pub fn shards_for_range(&self, row_start: u64, row_end: u64) -> Vec<usize> {
        if row_start >= row_end || self.shard_ranges.is_empty() {
            return Vec::new();
        }

        // Binary search: find the first shard whose row_end > row_start.
        // A shard (s_start, s_end) overlaps [row_start, row_end) iff
        //   s_start < row_end  AND  s_end > row_start
        //
        // We scan from the first candidate shard onwards.
        let first = self
            .shard_ranges
            .partition_point(|r| r.row_end <= row_start);

        let mut result = Vec::new();
        for r in &self.shard_ranges[first..] {
            if r.row_start >= row_end {
                break; // no more overlapping shards
            }
            result.push(r.sorted_shard_idx);
        }
        result
    }

    /// Find all shard indices needed for a set of row indices.
    ///
    /// Sorts indices, deduplicates, then uses range lookups.
    pub fn shards_for_indices(&self, rows: &[u64]) -> Vec<usize> {
        if rows.is_empty() {
            return Vec::new();
        }

        let mut sorted_rows = rows.to_vec();
        sorted_rows.sort_unstable();
        sorted_rows.dedup();

        let mut result = Vec::new();
        let mut last_shard: Option<usize> = None;

        for &row in &sorted_rows {
            // Find the shard containing this row: shard where row_start <= row < row_end
            let pos = self.shard_ranges.partition_point(|r| r.row_start <= row);
            if pos == 0 {
                continue; // row is before all shards
            }
            let r = self.shard_ranges[pos - 1];
            if row >= r.row_start && row < r.row_end && last_shard != Some(r.sorted_shard_idx) {
                result.push(r.sorted_shard_idx);
                last_shard = Some(r.sorted_shard_idx);
            }
        }
        result
    }

    /// Get the shard range `(row_start, row_end)` for a given shard index.
    ///
    /// O(1) — shard indices are assigned sequentially during construction,
    /// so `shard_idx` is the position in the sorted `shard_ranges` vec.
    pub fn shard_range(&self, shard_idx: usize) -> Option<(u64, u64)> {
        self.shard_ranges
            .get(shard_idx)
            .map(|r| (r.row_start, r.row_end))
    }

    /// Find the shard index containing a single row, or `None` if the row
    /// falls outside every shard's range.
    ///
    /// O(log n) over `shard_ranges` via `partition_point`. Equivalent to a
    /// `shards_for_indices(&[row])` call without the sort/dedup overhead —
    /// useful when sorting plans by shard locality on a per-row basis.
    pub fn shard_for_row(&self, row: u64) -> Option<usize> {
        let pos = self.shard_ranges.partition_point(|r| r.row_start <= row);
        if pos == 0 {
            return None;
        }
        let r = &self.shard_ranges[pos - 1];
        if row >= r.row_start && row < r.row_end {
            Some(r.sorted_shard_idx)
        } else {
            None
        }
    }
}
