//! Backed (on-demand) CSR access for SCX files.
//!
//! Provides [`BackedCsrIndex`] for O(log n) shard lookups and
//! [`BackedCsrReader`] for on-demand shard decoding with optional LRU caching.
//! Used by pyscx's backed mode to implement AnnData-compatible lazy access.

use std::num::NonZeroUsize;
use std::sync::Mutex;

use lru::LruCache;
use scx_sparse::ScxCsr;

use crate::catalog::{FullCatalog, FullCatalogEntry};
use crate::error::{Result, ScxError};
use crate::reader::ScxReader;
use crate::section::SectionType;

// ---------------------------------------------------------------------------
// BackedCsrIndex
// ---------------------------------------------------------------------------

/// Precomputed shard index for O(log n) row-range lookups.
///
/// Built once from a [`FullCatalog`] at open time.  Each entry stores
/// `(row_start, row_end, catalog_entry_index)` sorted by `row_start`.
#[derive(Debug, Clone)]
pub struct BackedCsrIndex {
    /// Sorted by `row_start`.  Each entry: `(row_start, row_end, entry_index_in_sorted_shards)`.
    shard_ranges: Vec<(u64, u64, usize)>,
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

    /// Internal: build from catalog filtering by section type and optional name prefix.
    fn from_catalog_filtered(
        catalog: &FullCatalog,
        section_type: SectionType,
        name_prefix: Option<&str>,
    ) -> Self {
        let mut shard_entries: Vec<(u64, u64, usize)> = catalog
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                e.section_type == section_type && name_prefix.is_none_or(|p| e.name.starts_with(p))
            })
            .filter_map(|(_, e)| {
                e.stats.as_ref().map(|s| (s.row_start, s.row_end, 0usize)) // index filled below
            })
            .collect();

        // Sort by row_start (deterministic ordering)
        shard_entries.sort_by_key(|&(rs, _, _)| rs);

        // Assign sorted indices
        for (i, entry) in shard_entries.iter_mut().enumerate() {
            entry.2 = i;
        }

        BackedCsrIndex {
            shard_ranges: shard_entries,
        }
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
            .partition_point(|&(_, s_end, _)| s_end <= row_start);

        let mut result = Vec::new();
        for &(s_start, _s_end, idx) in &self.shard_ranges[first..] {
            if s_start >= row_end {
                break; // no more overlapping shards
            }
            result.push(idx);
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
            let pos = self
                .shard_ranges
                .partition_point(|&(s_start, _, _)| s_start <= row);
            if pos == 0 {
                continue; // row is before all shards
            }
            let (s_start, s_end, idx) = self.shard_ranges[pos - 1];
            if row >= s_start && row < s_end && last_shard != Some(idx) {
                result.push(idx);
                last_shard = Some(idx);
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
            .map(|&(rs, re, _)| (rs, re))
    }
}

// ---------------------------------------------------------------------------
// Aggregation operation enum (internal)
// ---------------------------------------------------------------------------

/// Internal enum for masked column aggregation dispatch.
#[derive(Clone, Copy)]
enum AggOp {
    Sum,
    Nnz,
    Max,
    Min,
}

// ---------------------------------------------------------------------------
// BackedCsrReader
// ---------------------------------------------------------------------------

/// On-demand CSR reader with optional shard caching.
///
/// Wraps an [`ScxReader`] with a [`BackedCsrIndex`] for efficient row-range
/// lookups and an optional LRU cache for decoded shards.
///
/// # Thread Safety
///
/// `ScxReader` contains a `memmap2::Mmap` which is `Send + Sync`.
/// The reader itself is safe to share immutably.  Only the LRU cache
/// requires interior mutability via [`Mutex`].
pub struct BackedCsrReader {
    reader: ScxReader,
    index: BackedCsrIndex,
    n_vars: usize,
    n_obs: usize,
    /// If set, this reader targets a specific layer rather than X.
    layer_name: Option<String>,
    /// Pre-sorted catalog entries for layer shards (empty for X shards).
    sorted_entries: Vec<FullCatalogEntry>,
    cache: Option<Mutex<LruCache<usize, ScxCsr>>>,
}

impl BackedCsrReader {
    /// Create a new backed reader for X shards.
    ///
    /// `cache_shards`: number of decoded shards to cache (0 = no cache).
    pub fn new(reader: ScxReader, cache_shards: usize) -> Self {
        let index = BackedCsrIndex::from_catalog(reader.catalog());
        let n_vars = reader.n_vars() as usize;
        let n_obs = reader.n_obs() as usize;
        let cache = Self::make_cache(cache_shards);
        BackedCsrReader {
            reader,
            index,
            n_vars,
            n_obs,
            layer_name: None,
            sorted_entries: Vec::new(),
            cache,
        }
    }

    /// Create a new backed reader for a specific layer's shards.
    ///
    /// `layer_name`: the layer name (e.g., `"raw"`).
    /// `cache_shards`: number of decoded shards to cache (0 = no cache).
    pub fn new_for_layer(reader: ScxReader, layer_name: &str, cache_shards: usize) -> Self {
        let index = BackedCsrIndex::from_catalog_layer(reader.catalog(), layer_name);
        let n_vars = reader.n_vars() as usize;
        // n_obs for layers is the same as for X — layer shards cover the same rows.
        let n_obs = reader.n_obs() as usize;
        let cache = Self::make_cache(cache_shards);

        // Pre-compute sorted layer shard entries to avoid re-scanning catalog on every access
        let prefix = format!("{layer_name}_shard_");
        let mut sorted_entries: Vec<FullCatalogEntry> = reader
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::LayerCsrShard && e.name.starts_with(&prefix))
            .cloned()
            .collect();
        sorted_entries.sort_by_key(|e| e.stats.as_ref().map_or(u64::MAX, |s| s.row_start));

        BackedCsrReader {
            reader,
            index,
            n_vars,
            n_obs,
            layer_name: Some(layer_name.to_string()),
            sorted_entries,
            cache,
        }
    }

    fn make_cache(cache_shards: usize) -> Option<Mutex<LruCache<usize, ScxCsr>>> {
        if cache_shards > 0 {
            Some(Mutex::new(LruCache::new(
                NonZeroUsize::new(cache_shards).unwrap(),
            )))
        } else {
            None
        }
    }

    /// Shape of the full matrix `(n_obs, n_vars)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.n_obs, self.n_vars)
    }

    /// Number of variables (columns).
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Number of observations (rows).
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// Access the underlying shard index.
    pub fn index(&self) -> &BackedCsrIndex {
        &self.index
    }

    /// Read rows `[start, end)` as a scipy-compatible `ScxCsr`.
    ///
    /// Decompresses only overlapping shards.  Takes `&self` — cache
    /// mutation is handled via interior mutability (Mutex).
    pub fn read_rows(&self, start: u64, end: u64) -> Result<ScxCsr> {
        if start >= end {
            return Ok(ScxCsr::new_unchecked(
                (0, self.n_vars),
                vec![0],
                vec![],
                vec![],
            ));
        }

        let shard_indices = self.index.shards_for_range(start, end);
        if shard_indices.is_empty() {
            return Ok(ScxCsr::new_unchecked(
                (0, self.n_vars),
                vec![0],
                vec![],
                vec![],
            ));
        }

        let mut slices = Vec::with_capacity(shard_indices.len());
        for &shard_idx in &shard_indices {
            let shard_csr = self.read_shard_cached(shard_idx)?;

            // Compute local row range within this shard
            let (s_start, s_end) =
                self.index
                    .shard_range(shard_idx)
                    .ok_or(ScxError::ShardIndexOutOfBounds {
                        index: shard_idx,
                        count: self.index.n_shards(),
                    })?;

            let local_start = (start.max(s_start) - s_start) as usize;
            let local_end = (end.min(s_end) - s_start) as usize;

            let sliced = shard_csr
                .row_slice(local_start, local_end)
                .map_err(|e| ScxError::Io(std::io::Error::other(e)))?;
            slices.push(sliced);
        }

        concatenate_csr(&slices, self.n_vars)
    }

    /// Read specific row indices as a scipy-compatible `ScxCsr`.
    ///
    /// Decompresses only shards containing requested rows.
    pub fn read_row_indices(&self, indices: &[u64]) -> Result<ScxCsr> {
        if indices.is_empty() {
            return Ok(ScxCsr::new_unchecked(
                (0, self.n_vars),
                vec![0],
                vec![],
                vec![],
            ));
        }

        // Sort the indices but keep track of the original ordering
        // so we output rows in the requested order.
        let mut sorted_pairs: Vec<(u64, usize)> =
            indices.iter().enumerate().map(|(i, &r)| (r, i)).collect();
        sorted_pairs.sort_by_key(|&(r, _)| r);

        // Group by shard
        let shard_indices = self
            .index
            .shards_for_indices(&sorted_pairs.iter().map(|&(r, _)| r).collect::<Vec<_>>());

        // For each shard, extract the needed rows
        let mut row_csrs: Vec<(usize, ScxCsr)> = Vec::new();

        for &shard_idx in &shard_indices {
            let shard_csr = self.read_shard_cached(shard_idx)?;
            let (s_start, s_end) =
                self.index
                    .shard_range(shard_idx)
                    .ok_or(ScxError::ShardIndexOutOfBounds {
                        index: shard_idx,
                        count: self.index.n_shards(),
                    })?;

            // Extract individual rows from this shard
            for &(row, orig_idx) in &sorted_pairs {
                if row >= s_start && row < s_end {
                    let local_row = (row - s_start) as usize;
                    let sliced = shard_csr
                        .row_slice(local_row, local_row + 1)
                        .map_err(|e| ScxError::Io(std::io::Error::other(e)))?;
                    row_csrs.push((orig_idx, sliced));
                }
            }
        }

        // Sort by original index to preserve requested ordering
        row_csrs.sort_by_key(|&(idx, _)| idx);

        let ordered: Vec<ScxCsr> = row_csrs.into_iter().map(|(_, csr)| csr).collect();
        concatenate_csr(&ordered, self.n_vars)
    }

    /// Read all rows — materializes the full matrix.
    ///
    /// Used by `to_memory()` on the Python side.
    pub fn read_all(&self) -> Result<ScxCsr> {
        match &self.layer_name {
            None => self.reader.read_all_csr_shards(),
            Some(name) => self.reader.read_layer(name),
        }
    }

    /// Read and optionally cache a single decoded shard.
    fn read_shard_cached(&self, shard_idx: usize) -> Result<ScxCsr> {
        // Check cache first
        if let Some(ref cache_mutex) = self.cache {
            let mut cache = cache_mutex.lock().unwrap();
            if let Some(cached) = cache.get(&shard_idx) {
                return Ok(cached.clone());
            }
        }

        // Cache miss: decode the shard. For layer shards, we look up the
        // sorted layer entries in the catalog; for X shards, we use
        // the standard read_csr_shard path.
        let (indptr, indices, data) = match &self.layer_name {
            None => self.reader.read_csr_shard(shard_idx)?,
            Some(_) => {
                let entry =
                    self.sorted_entries
                        .get(shard_idx)
                        .ok_or(ScxError::ShardIndexOutOfBounds {
                            index: shard_idx,
                            count: self.sorted_entries.len(),
                        })?;
                self.reader.read_shard_from_entry(entry)?
            }
        };
        let n_rows = indptr.len().saturating_sub(1);
        let csr = ScxCsr::new_unchecked((n_rows, self.n_vars), indptr, indices, data);

        // Insert into cache
        if let Some(ref cache_mutex) = self.cache {
            let mut cache = cache_mutex.lock().unwrap();
            cache.put(shard_idx, csr.clone());
        }

        Ok(csr)
    }

    // --- Native shard-by-shard aggregation ---
    //
    // These methods compute statistics without materializing the full
    // concatenated CSR. Peak memory = one decoded shard at a time
    // (plus the output vector).

    /// Compute per-row sums without materializing the full matrix.
    ///
    /// Iterates shards in order, computes row sums from each shard's
    /// CSR arrays, and concatenates the results.
    pub fn row_sums(&self) -> Result<Vec<f64>> {
        let n_shards = self.index.n_shards();
        let mut all_sums = Vec::with_capacity(self.n_obs);
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_cached(shard_idx)?;
            all_sums.extend(csr.row_sums());
        }
        Ok(all_sums)
    }

    /// Compute per-column sums without materializing the full matrix.
    ///
    /// Iterates shards, accumulates column sums into a single `n_vars`-length vector.
    pub fn col_sums(&self) -> Result<Vec<f64>> {
        let n_shards = self.index.n_shards();
        let mut sums = vec![0.0f64; self.n_vars];
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_cached(shard_idx)?;
            let partial = csr.col_sums();
            for (s, p) in sums.iter_mut().zip(partial.iter()) {
                *s += p;
            }
        }
        Ok(sums)
    }

    /// Compute per-row NNZ counts without materializing the full matrix.
    pub fn row_nnz(&self) -> Result<Vec<i64>> {
        let n_shards = self.index.n_shards();
        let mut all_nnz = Vec::with_capacity(self.n_obs);
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_cached(shard_idx)?;
            all_nnz.extend(csr.row_nnz());
        }
        Ok(all_nnz)
    }

    /// Compute per-column NNZ counts without materializing the full matrix.
    pub fn col_nnz(&self) -> Result<Vec<i64>> {
        let n_shards = self.index.n_shards();
        let mut counts = vec![0i64; self.n_vars];
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_cached(shard_idx)?;
            let partial = csr.col_nnz();
            for (c, p) in counts.iter_mut().zip(partial.iter()) {
                *c += p;
            }
        }
        Ok(counts)
    }

    /// Total NNZ across all shards without materializing.
    pub fn total_nnz(&self) -> Result<usize> {
        let n_shards = self.index.n_shards();
        let mut total = 0usize;
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_cached(shard_idx)?;
            total += csr.nnz();
        }
        Ok(total)
    }

    // --- Variance ---

    /// Streaming per-row variance without materializing the full matrix.
    ///
    /// Each shard independently computes row variances (one row = one shard's row).
    pub fn row_var(&self) -> Result<Vec<f64>> {
        let n_shards = self.index.n_shards();
        let mut all_var = Vec::with_capacity(self.n_obs);
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_cached(shard_idx)?;
            all_var.extend(csr.row_var());
        }
        Ok(all_var)
    }

    /// Streaming per-column variance (population variance, ddof=0).
    ///
    /// Two-pass algorithm:
    ///   1. Compute column means via `col_sums() / n_obs`
    ///   2. Stream shards, accumulating `(x - mean)²` for stored values
    ///   3. Add zero-entry contributions: `(n_obs - col_nnz) * mean²`
    pub fn col_var(&self) -> Result<Vec<f64>> {
        let n_obs = self.n_obs;
        if n_obs == 0 {
            return Ok(vec![0.0f64; self.n_vars]);
        }

        // Pass 1: column means
        let col_sums = self.col_sums()?;
        let col_means: Vec<f64> = col_sums.iter().map(|&s| s / n_obs as f64).collect();

        // Pass 2: accumulate (val - mean)² for stored entries
        let n_shards = self.index.n_shards();
        let mut sq_devs = vec![0.0f64; self.n_vars];
        let mut col_nnz = vec![0usize; self.n_vars];

        for shard_idx in 0..n_shards {
            let csr = self.read_shard_cached(shard_idx)?;
            let partial = csr.col_var_partial(&col_means);
            for (s, p) in sq_devs.iter_mut().zip(partial.iter()) {
                *s += p;
            }
            let nnz = csr.col_nnz();
            for (c, &n) in col_nnz.iter_mut().zip(nnz.iter()) {
                *c += n as usize;
            }
        }

        // Add contribution from implicit zeros: (n_obs - col_nnz[c]) * mean[c]²
        let mut variances = vec![0.0f64; self.n_vars];
        for c in 0..self.n_vars {
            let n_zeros = n_obs - col_nnz[c];
            let total_sq_dev = sq_devs[c] + n_zeros as f64 * col_means[c] * col_means[c];
            variances[c] = total_sq_dev / n_obs as f64;
        }

        Ok(variances)
    }

    // --- Max / Min ---

    /// Streaming per-row max without materializing the full matrix.
    pub fn row_max(&self) -> Result<Vec<f64>> {
        let n_shards = self.index.n_shards();
        let mut all_max = Vec::with_capacity(self.n_obs);
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_cached(shard_idx)?;
            all_max.extend(csr.row_max());
        }
        Ok(all_max)
    }

    /// Streaming per-column max without materializing the full matrix.
    ///
    /// Merges per-shard column maxes. Accounts for implicit zeros:
    /// if any column has fewer stored entries than `n_obs`, the max is
    /// at least 0.0.
    pub fn col_max(&self) -> Result<Vec<f64>> {
        let n_shards = self.index.n_shards();
        let mut maxes = vec![f64::NEG_INFINITY; self.n_vars];
        let mut col_nnz = vec![0usize; self.n_vars];

        for shard_idx in 0..n_shards {
            let csr = self.read_shard_cached(shard_idx)?;
            // Get per-column max within this shard (using n_rows of shard, not global n_obs)
            // We need the raw stored max, so we pass n_rows = shard.n_rows()
            // But we want the global implicit-zero correction at the end,
            // so we track NNZ ourselves and compute raw stored max.
            for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                let c = col as usize;
                let v = val as f64;
                maxes[c] = maxes[c].max(v);
                col_nnz[c] += 1;
            }
        }

        // Apply implicit-zero correction at the global level
        for c in 0..self.n_vars {
            if col_nnz[c] < self.n_obs {
                if maxes[c] == f64::NEG_INFINITY {
                    maxes[c] = 0.0;
                } else {
                    maxes[c] = maxes[c].max(0.0);
                }
            }
        }
        Ok(maxes)
    }

    /// Streaming per-row min without materializing the full matrix.
    pub fn row_min(&self) -> Result<Vec<f64>> {
        let n_shards = self.index.n_shards();
        let mut all_min = Vec::with_capacity(self.n_obs);
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_cached(shard_idx)?;
            all_min.extend(csr.row_min());
        }
        Ok(all_min)
    }

    /// Streaming per-column min without materializing the full matrix.
    pub fn col_min(&self) -> Result<Vec<f64>> {
        let n_shards = self.index.n_shards();
        let mut mins = vec![f64::INFINITY; self.n_vars];
        let mut col_nnz = vec![0usize; self.n_vars];

        for shard_idx in 0..n_shards {
            let csr = self.read_shard_cached(shard_idx)?;
            for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                let c = col as usize;
                let v = val as f64;
                mins[c] = mins[c].min(v);
                col_nnz[c] += 1;
            }
        }

        for c in 0..self.n_vars {
            if col_nnz[c] < self.n_obs {
                if mins[c] == f64::INFINITY {
                    mins[c] = 0.0;
                } else {
                    mins[c] = mins[c].min(0.0);
                }
            }
        }
        Ok(mins)
    }

    // --- Masked column aggregation (deletion-vector aware) ---
    //
    // These variants accept a `kept_rows` set and only aggregate values from
    // rows in that set. Used when deletion vectors are present.

    /// Column sums considering only the kept rows.
    ///
    /// Iterates shards, intersects with the kept set, and accumulates.
    pub fn col_sums_masked(&self, kept_rows: &[u64]) -> Result<Vec<f64>> {
        self.col_aggregate_masked(kept_rows, AggOp::Sum)
    }

    /// Column NNZ considering only the kept rows.
    pub fn col_nnz_masked(&self, kept_rows: &[u64]) -> Result<Vec<f64>> {
        self.col_aggregate_masked(kept_rows, AggOp::Nnz)
    }

    /// Column max considering only the kept rows.
    pub fn col_max_masked(&self, kept_rows: &[u64]) -> Result<Vec<f64>> {
        self.col_aggregate_masked(kept_rows, AggOp::Max)
    }

    /// Column min considering only the kept rows.
    pub fn col_min_masked(&self, kept_rows: &[u64]) -> Result<Vec<f64>> {
        self.col_aggregate_masked(kept_rows, AggOp::Min)
    }

    /// Column variance considering only the kept rows.
    pub fn col_var_masked(&self, kept_rows: &[u64]) -> Result<Vec<f64>> {
        let n_kept = kept_rows.len();
        if n_kept == 0 {
            return Ok(vec![0.0f64; self.n_vars]);
        }

        // Pass 1: column sums over kept rows → means
        let col_sums = self.col_sums_masked(kept_rows)?;
        let col_means: Vec<f64> = col_sums.iter().map(|&s| s / n_kept as f64).collect();

        // Pass 2: accumulate (val - mean)² for stored entries in kept rows
        let mut sq_devs = vec![0.0f64; self.n_vars];
        let mut col_nnz = vec![0usize; self.n_vars];

        let n_shards = self.index.n_shards();
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_cached(shard_idx)?;
            let (s_start, s_end) = match self.index.shard_range(shard_idx) {
                Some(r) => r,
                None => continue,
            };

            // Find kept rows that fall in this shard
            for &global_row in kept_rows {
                if global_row >= s_start && global_row < s_end {
                    let local_row = (global_row - s_start) as usize;
                    let row_start = csr.indptr[local_row] as usize;
                    let row_end = csr.indptr[local_row + 1] as usize;
                    for j in row_start..row_end {
                        let c = csr.indices[j] as usize;
                        let diff = csr.data[j] as f64 - col_means[c];
                        sq_devs[c] += diff * diff;
                        col_nnz[c] += 1;
                    }
                }
            }
        }

        // Add zero-entry contributions
        let mut variances = vec![0.0f64; self.n_vars];
        for c in 0..self.n_vars {
            let n_zeros = n_kept - col_nnz[c];
            let total = sq_devs[c] + n_zeros as f64 * col_means[c] * col_means[c];
            variances[c] = total / n_kept as f64;
        }
        Ok(variances)
    }

    /// Internal: masked column aggregation over kept rows.
    fn col_aggregate_masked(&self, kept_rows: &[u64], op: AggOp) -> Result<Vec<f64>> {
        let n_kept = kept_rows.len();
        let mut result = match op {
            AggOp::Sum | AggOp::Nnz => vec![0.0f64; self.n_vars],
            AggOp::Max => vec![f64::NEG_INFINITY; self.n_vars],
            AggOp::Min => vec![f64::INFINITY; self.n_vars],
        };
        let mut col_nnz = vec![0usize; self.n_vars];

        let n_shards = self.index.n_shards();
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_cached(shard_idx)?;
            let (s_start, s_end) = match self.index.shard_range(shard_idx) {
                Some(r) => r,
                None => continue,
            };

            for &global_row in kept_rows {
                if global_row >= s_start && global_row < s_end {
                    let local_row = (global_row - s_start) as usize;
                    let row_start = csr.indptr[local_row] as usize;
                    let row_end = csr.indptr[local_row + 1] as usize;
                    for j in row_start..row_end {
                        let c = csr.indices[j] as usize;
                        let v = csr.data[j] as f64;
                        match op {
                            AggOp::Sum => result[c] += v,
                            AggOp::Nnz => result[c] += 1.0,
                            AggOp::Max => result[c] = result[c].max(v),
                            AggOp::Min => result[c] = result[c].min(v),
                        }
                        col_nnz[c] += 1;
                    }
                }
            }
        }

        // Handle implicit zeros for max/min
        match op {
            AggOp::Max => {
                for c in 0..self.n_vars {
                    if col_nnz[c] < n_kept {
                        if result[c] == f64::NEG_INFINITY {
                            result[c] = 0.0;
                        } else {
                            result[c] = result[c].max(0.0);
                        }
                    }
                }
            }
            AggOp::Min => {
                for c in 0..self.n_vars {
                    if col_nnz[c] < n_kept {
                        if result[c] == f64::INFINITY {
                            result[c] = 0.0;
                        } else {
                            result[c] = result[c].min(0.0);
                        }
                    }
                }
            }
            _ => {}
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// CSR concatenation helper
// ---------------------------------------------------------------------------

/// Merge multiple `ScxCsr` values into one, rebasing indptr.
///
/// All CSRs must have the same number of columns (`n_vars`).
pub fn concatenate_csr(csrs: &[ScxCsr], n_vars: usize) -> Result<ScxCsr> {
    if csrs.is_empty() {
        return Ok(ScxCsr::new_unchecked((0, n_vars), vec![0], vec![], vec![]));
    }

    if csrs.len() == 1 {
        return Ok(csrs[0].clone());
    }

    // Pre-compute total sizes for allocation
    let total_rows: usize = csrs.iter().map(|c| c.n_rows()).sum();
    let total_nnz: usize = csrs.iter().map(|c| c.nnz()).sum();

    let mut merged_indptr = Vec::with_capacity(total_rows + 1);
    let mut merged_indices = Vec::with_capacity(total_nnz);
    let mut merged_data = Vec::with_capacity(total_nnz);
    let mut cumulative_nnz: i64 = 0;

    for (i, csr) in csrs.iter().enumerate() {
        if i == 0 {
            merged_indptr.extend_from_slice(&csr.indptr);
        } else {
            // Skip first element (0) and offset by cumulative nnz
            for &v in &csr.indptr[1..] {
                merged_indptr.push(v + cumulative_nnz);
            }
        }
        cumulative_nnz += *csr.indptr.last().unwrap_or(&0);
        merged_indices.extend_from_slice(&csr.indices);
        merged_data.extend_from_slice(&csr.data);
    }

    Ok(ScxCsr::new_unchecked(
        (total_rows, n_vars),
        merged_indptr,
        merged_indices,
        merged_data,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::{FileHeader, MAGIC};
    use crate::writer::ScxWriter;
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use scx_codec::{CodecId, ValueEncoding};
    use std::sync::Arc;
    use tempfile::TempDir;

    // --- Test helpers (same as reader.rs test helpers) ---

    fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
        FileHeader {
            magic: MAGIC,
            format_version: 1,
            header_length: 256,
            flags: 0,
            n_obs,
            n_vars,
            nnz,
            n_csr_shards: 0,
            n_csc_shards: 0,
            shard_target_rows: 16384,
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
            reserved: [0u8; 132],
        }
    }

    fn sample_obs(n: usize) -> arrow::array::RecordBatch {
        let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
        let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
        arrow::array::RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    fn sample_var(n: usize) -> arrow::array::RecordBatch {
        let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
        let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
        arrow::array::RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    fn sample_shard_data(n_rows: usize, n_vars: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for row in 0..n_rows {
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

    /// Write a test file with the specified number of shards and return
    /// a `BackedCsrReader` along with the reference full CSR.
    fn write_test_file_and_open(
        dir: &TempDir,
        n_obs: usize,
        n_vars: usize,
        n_shards: usize,
        cache_shards: usize,
    ) -> (BackedCsrReader, ScxCsr) {
        let path = dir.path().join("test.scx");
        let total_nnz = n_obs * 2;
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

        writer.finish().unwrap();

        let reader = ScxReader::open(&path).unwrap();
        let full_csr = {
            let r2 = ScxReader::open(&path).unwrap();
            r2.read_all_csr_shards().unwrap()
        };
        let backed = BackedCsrReader::new(reader, cache_shards);
        (backed, full_csr)
    }

    // -----------------------------------------------------------------------
    // BackedCsrIndex tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_index_from_catalog() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        assert_eq!(backed.index().n_shards(), 4);
    }

    #[test]
    fn test_shards_for_range_single_shard() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        // 4 shards of 3 rows each: [0,3), [3,6), [6,9), [9,12)
        let shards = backed.index().shards_for_range(0, 3);
        assert_eq!(shards, vec![0]);
    }

    #[test]
    fn test_shards_for_range_two_shards() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        // Spans shard 0 [0,3) and shard 1 [3,6)
        let shards = backed.index().shards_for_range(1, 5);
        assert_eq!(shards, vec![0, 1]);
    }

    #[test]
    fn test_shards_for_range_all_shards() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        let shards = backed.index().shards_for_range(0, 12);
        assert_eq!(shards, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_shards_for_range_empty() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        let shards = backed.index().shards_for_range(5, 5);
        assert!(shards.is_empty());
    }

    #[test]
    fn test_shards_for_range_at_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        // Exact shard boundary: [3,6) should be shard 1 only
        let shards = backed.index().shards_for_range(3, 6);
        assert_eq!(shards, vec![1]);
    }

    #[test]
    fn test_shards_for_indices_scattered() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        // Row 0 in shard 0, row 5 in shard 1, row 11 in shard 3
        let shards = backed.index().shards_for_indices(&[0, 5, 11]);
        assert_eq!(shards, vec![0, 1, 3]);
    }

    #[test]
    fn test_shards_for_indices_all_one_shard() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        let shards = backed.index().shards_for_indices(&[0, 1, 2]);
        assert_eq!(shards, vec![0]);
    }

    #[test]
    fn test_shards_for_indices_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        let shards = backed.index().shards_for_indices(&[0, 0, 1, 1]);
        assert_eq!(shards, vec![0]);
    }

    #[test]
    fn test_shards_for_indices_unsorted() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        let shards = backed.index().shards_for_indices(&[11, 0, 5]);
        assert_eq!(shards, vec![0, 1, 3]);
    }

    // -----------------------------------------------------------------------
    // BackedCsrReader::read_rows tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_rows_full_range() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        let result = backed.read_rows(0, 12).unwrap();
        assert_eq!(result.shape, full.shape);
        assert_eq!(result.indptr, full.indptr);
        assert_eq!(result.indices, full.indices);
        assert_eq!(result.data, full.data);
    }

    #[test]
    fn test_read_rows_single_row() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        for row in 0..12 {
            let backed_row = backed.read_rows(row, row + 1).unwrap();
            let full_row = full.row_slice(row as usize, row as usize + 1).unwrap();
            assert_eq!(
                backed_row.indptr, full_row.indptr,
                "indptr mismatch at row {row}"
            );
            assert_eq!(
                backed_row.indices, full_row.indices,
                "indices mismatch at row {row}"
            );
            assert_eq!(backed_row.data, full_row.data, "data mismatch at row {row}");
        }
    }

    #[test]
    fn test_read_rows_cross_shard_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        // Spans shards 0 and 1 (rows [2,5))
        let result = backed.read_rows(2, 5).unwrap();
        let expected = full.row_slice(2, 5).unwrap();
        assert_eq!(result.shape, expected.shape);
        assert_eq!(result.indptr, expected.indptr);
        assert_eq!(result.indices, expected.indices);
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_read_rows_every_contiguous_range() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        // Test every possible contiguous range
        for start in 0..12u64 {
            for end in (start + 1)..=12 {
                let result = backed.read_rows(start, end).unwrap();
                let expected = full.row_slice(start as usize, end as usize).unwrap();
                assert_eq!(
                    result.indptr, expected.indptr,
                    "indptr mismatch at [{start}, {end})"
                );
                assert_eq!(
                    result.indices, expected.indices,
                    "indices mismatch at [{start}, {end})"
                );
                assert_eq!(
                    result.data, expected.data,
                    "data mismatch at [{start}, {end})"
                );
            }
        }
    }

    #[test]
    fn test_read_rows_empty_range() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        let result = backed.read_rows(5, 5).unwrap();
        assert_eq!(result.n_rows(), 0);
        assert_eq!(result.nnz(), 0);
    }

    // -----------------------------------------------------------------------
    // BackedCsrReader::read_row_indices tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_row_indices_scattered() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        let indices = [0u64, 5, 11];
        let result = backed.read_row_indices(&indices).unwrap();
        assert_eq!(result.n_rows(), 3);

        // Each row should match the corresponding row from the full CSR
        for (i, &row) in indices.iter().enumerate() {
            let expected = full.row_slice(row as usize, row as usize + 1).unwrap();
            let actual = result.row_slice(i, i + 1).unwrap();
            assert_eq!(actual.indices, expected.indices, "row {row} indices");
            assert_eq!(actual.data, expected.data, "row {row} data");
        }
    }

    #[test]
    fn test_read_row_indices_empty() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        let result = backed.read_row_indices(&[]).unwrap();
        assert_eq!(result.n_rows(), 0);
    }

    // -----------------------------------------------------------------------
    // LRU cache behavior tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 0);

        // Should still work without cache
        let result = backed.read_rows(0, 12).unwrap();
        assert_eq!(result.indptr, full.indptr);
    }

    #[test]
    fn test_cache_hit() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        // First read (cold)
        let r1 = backed.read_rows(0, 3).unwrap();
        // Second read (cached)
        let r2 = backed.read_rows(0, 3).unwrap();
        assert_eq!(r1.indptr, r2.indptr);
        assert_eq!(r1.indices, r2.indices);
        assert_eq!(r1.data, r2.data);
    }

    #[test]
    fn test_cache_eviction() {
        let dir = tempfile::tempdir().unwrap();
        // Cache can hold 2 shards, file has 4
        let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 2);

        // Access shards 0, 1, 2 — shard 0 should be evicted
        let _ = backed.read_rows(0, 3).unwrap(); // shard 0
        let _ = backed.read_rows(3, 6).unwrap(); // shard 1
        let _ = backed.read_rows(6, 9).unwrap(); // shard 2 (evicts shard 0)

        // Re-read shard 0 — should still produce correct results
        let result = backed.read_rows(0, 3).unwrap();
        let expected = full.row_slice(0, 3).unwrap();
        assert_eq!(result.indptr, expected.indptr);
        assert_eq!(result.indices, expected.indices);
        assert_eq!(result.data, expected.data);
    }

    // -----------------------------------------------------------------------
    // CSR concatenation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_concatenate_empty() {
        let result = concatenate_csr(&[], 5).unwrap();
        assert_eq!(result.n_rows(), 0);
        assert_eq!(result.nnz(), 0);
        assert_eq!(result.shape.1, 5);
    }

    #[test]
    fn test_concatenate_single() {
        let csr = ScxCsr::new_unchecked((2, 5), vec![0, 2, 3], vec![1, 3, 2], vec![5.0, 10.0, 2.0]);
        let result = concatenate_csr(&[csr.clone()], 5).unwrap();
        assert_eq!(result.indptr, csr.indptr);
        assert_eq!(result.indices, csr.indices);
        assert_eq!(result.data, csr.data);
    }

    #[test]
    fn test_concatenate_two() {
        let csr1 = ScxCsr::new_unchecked((1, 5), vec![0, 2], vec![1, 3], vec![5.0, 10.0]);

        let csr2 = ScxCsr::new_unchecked((1, 5), vec![0, 1], vec![2], vec![7.0]);

        let result = concatenate_csr(&[csr1, csr2], 5).unwrap();
        assert_eq!(result.shape, (2, 5));
        assert_eq!(result.indptr, vec![0, 2, 3]);
        assert_eq!(result.indices, vec![1, 3, 2]);
        assert_eq!(result.data, vec![5.0, 10.0, 7.0]);
    }

    #[test]
    fn test_concatenate_empty_plus_nonempty() {
        let empty = ScxCsr::new_unchecked((0, 5), vec![0], vec![], vec![]);
        let nonempty = ScxCsr::new_unchecked((1, 5), vec![0, 2], vec![1, 3], vec![5.0, 10.0]);

        let result = concatenate_csr(&[empty, nonempty], 5).unwrap();
        assert_eq!(result.shape, (1, 5));
        assert_eq!(result.indptr, vec![0, 2]);
        assert_eq!(result.indices, vec![1, 3]);
        assert_eq!(result.data, vec![5.0, 10.0]);
    }

    #[test]
    fn test_shape_accessors() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 4);
        assert_eq!(backed.shape(), (12, 10));
        assert_eq!(backed.n_obs(), 12);
        assert_eq!(backed.n_vars(), 10);
    }
}
