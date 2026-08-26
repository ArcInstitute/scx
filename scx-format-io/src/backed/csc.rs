//! [`BackedCscReader`] and [`BackedCscIndex`]: column-major access to the
//! optional gene-major CSC sidecar.

use super::*;

// ---------------------------------------------------------------------------
// BackedCscIndex — column-major counterpart to BackedCsrIndex
// ---------------------------------------------------------------------------

/// Per-shard column range for a CSC sidecar.
///
/// Mirrors `ShardRange` (CSR) but the major axis is columns. The
/// on-disk fields are still `row_start` / `row_end` in `ShardStats`
/// (axis-overload — for `CscShard` entries those fields hold
/// `col_start` / `col_end`); we read them via
/// `ShardStats::major_start()` / `major_end()`.
#[derive(Debug, Clone, Copy)]
struct CscShardRange {
    col_start: u64,
    col_end: u64,
    /// Nonzeros in this shard, from the same `ShardStats` block the column
    /// range is read from. Carried so `ColumnShardSource::csc_shard_size_hint`
    /// can price a decoded shard without opening one — the row-major side has
    /// had this since 4.5; the column-major side read the field and threw it
    /// away.
    nnz: u64,
    /// Position in the sorted-and-filtered CSC shard list. Equal to the
    /// shard index used by `ScxReader::read_csc_shard`.
    sorted_shard_idx: usize,
}

/// Precomputed column-shard index for O(log n) col-range lookups.
///
/// Built once from a [`FullCatalog`] at open time. Filters
/// `SectionType::CscShard` only — `LayerCscShard` does not exist in
/// the current format version (deferred until layer-level CSC
/// support is needed).
#[derive(Debug, Clone)]
pub struct BackedCscIndex {
    /// Sorted by `col_start`.
    shard_ranges: Vec<CscShardRange>,
}

impl BackedCscIndex {
    /// Build from a [`FullCatalog`], scoped to global / single-modality
    /// CSC shards (`modality_id == 0`). Backwards-compatible entry
    /// point for v1 files and single-modality v2 files.
    pub fn from_catalog(catalog: &FullCatalog) -> Self {
        Self::from_catalog_for_modality(catalog, 0)
    }

    /// Build from a [`FullCatalog`], scoped to a specific modality.
    /// Filters CSC shards by `(SectionType::CscShard, modality_id)`,
    /// sorts by `col_start`, and records each shard's position in
    /// sorted order. v2 multimodal files use this constructor with
    /// `modality_id >= 1`.
    pub fn from_catalog_for_modality(catalog: &FullCatalog, modality_id: u8) -> Self {
        let mut shard_entries: Vec<CscShardRange> = catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CscShard && e.modality_id == modality_id)
            .filter_map(|e| {
                e.stats.as_ref().map(|s| CscShardRange {
                    col_start: s.major_start(e.section_type),
                    col_end: s.major_end(e.section_type),
                    nnz: s.nnz,
                    sorted_shard_idx: 0,
                })
            })
            .collect();

        shard_entries.sort_by_key(|r| r.col_start);
        for (i, entry) in shard_entries.iter_mut().enumerate() {
            entry.sorted_shard_idx = i;
        }
        BackedCscIndex {
            shard_ranges: shard_entries,
        }
    }

    /// Build from a [`FullCatalog`], scoped to a layer's CSC shards
    /// for a given modality. Filters by
    /// `(SectionType::LayerCscShard, modality_id, layer_name)`. The
    /// `layer_name` matches by substring `/{layer_name}/` in the
    /// section name (mirrors `FullCatalog::layer_csc_shards_for_modality`).
    pub fn from_catalog_for_layer(
        catalog: &FullCatalog,
        modality_id: u8,
        layer_name: &str,
    ) -> Self {
        let needle = format!("/{layer_name}/");
        let mut shard_entries: Vec<CscShardRange> = catalog
            .entries
            .iter()
            .filter(|e| {
                e.section_type == SectionType::LayerCscShard
                    && e.modality_id == modality_id
                    && e.name.contains(&needle)
            })
            .filter_map(|e| {
                e.stats.as_ref().map(|s| CscShardRange {
                    col_start: s.major_start(e.section_type),
                    col_end: s.major_end(e.section_type),
                    nnz: s.nnz,
                    sorted_shard_idx: 0,
                })
            })
            .collect();
        shard_entries.sort_by_key(|r| r.col_start);
        for (i, entry) in shard_entries.iter_mut().enumerate() {
            entry.sorted_shard_idx = i;
        }
        BackedCscIndex {
            shard_ranges: shard_entries,
        }
    }

    /// Number of CSC shards.
    pub fn n_shards(&self) -> usize {
        self.shard_ranges.len()
    }

    /// Find the CSC shard containing `col`, or `None` if `col` falls
    /// outside every shard's `[col_start, col_end)`.
    pub fn shard_for_col(&self, col: u64) -> Option<usize> {
        let pos = self.shard_ranges.partition_point(|r| r.col_start <= col);
        if pos == 0 {
            return None;
        }
        let r = &self.shard_ranges[pos - 1];
        if col >= r.col_start && col < r.col_end {
            Some(r.sorted_shard_idx)
        } else {
            None
        }
    }

    /// All shard indices whose `[col_start, col_end)` intersects
    /// `[c_lo, c_hi)`. Returned in ascending order.
    pub fn shards_for_col_range(&self, c_lo: u64, c_hi: u64) -> Vec<usize> {
        if c_lo >= c_hi || self.shard_ranges.is_empty() {
            return Vec::new();
        }
        let first = self.shard_ranges.partition_point(|r| r.col_end <= c_lo);
        let mut result = Vec::new();
        for r in &self.shard_ranges[first..] {
            if r.col_start >= c_hi {
                break;
            }
            result.push(r.sorted_shard_idx);
        }
        result
    }

    /// Upper bounds on the largest CSC shard, or `None` when the catalog
    /// carries nothing to bound with.
    ///
    /// `max_rows` is the **major axis** — columns on this layout; see
    /// [`ColumnShardSource::csc_shard_size_hint`](crate::ColumnShardSource::csc_shard_size_hint).
    ///
    /// Declines on `max_nnz == 0` for the same reason the row-major override
    /// does: reporting zero would be an *under*-estimate, which the hint
    /// contract forbids, and the caller then grows its buffers on demand
    /// exactly as it did before. A genuinely all-empty sidecar is
    /// indistinguishable here and also declines — harmless.
    fn size_hint(&self) -> Option<crate::ShardSizeHint> {
        let max_nnz = self.shard_ranges.iter().map(|r| r.nnz).max().unwrap_or(0);
        if max_nnz == 0 {
            return None;
        }
        let max_cols = self
            .shard_ranges
            .iter()
            .map(|r| r.col_end.saturating_sub(r.col_start))
            .max()
            .unwrap_or(0);
        Some(crate::ShardSizeHint {
            max_rows: usize::try_from(max_cols).ok()?,
            max_nnz: usize::try_from(max_nnz).ok()?,
        })
    }

    /// Get the shard column range `(col_start, col_end)`.
    pub fn shard_col_range(&self, shard_idx: usize) -> Option<(u64, u64)> {
        self.shard_ranges
            .get(shard_idx)
            .map(|r| (r.col_start, r.col_end))
    }
}

// ---------------------------------------------------------------------------
// BackedCscReader — column-major counterpart to BackedCsrReader
// ---------------------------------------------------------------------------

/// Reject a CSC sidecar that was built against an earlier generation of
/// the CSR data (the freshness guard introduced with `catalog_version`
/// v4). The rule itself lives on [`FullCatalog::csc_sidecar_is_fresh`];
/// this adds the "is there even a sidecar" half, which the constructors
/// need because they are handed a possibly-empty entry list.
///
/// The check is skipped when `csc_entries` is empty (no sidecar to
/// validate) and is a no-op for v1–v3 files (both counters default to
/// `0`, so `0 == 0`), so existing valid sidecars are never rejected.
pub(super) fn check_csc_sidecar_fresh(
    catalog: &FullCatalog,
    csc_entries: &[FullCatalogEntry],
) -> Result<()> {
    if csc_entries.is_empty() {
        return Ok(());
    }
    if !catalog.csc_sidecar_is_fresh() {
        return Err(ScxError::StaleCscSidecar {
            built_generation: catalog.csc_build_generation,
            data_generation: catalog.data_generation,
        });
    }
    Ok(())
}

/// On-demand CSC reader with optional decoded-shard caching.
///
/// Wraps an [`ScxReader`] with a [`BackedCscIndex`] for O(log n)
/// column-range lookups and a [`ShardCache`] of decoded `ScxCsc` shards.
/// Implements [`crate::ColumnShardSource`].
///
/// The cache is the same byte-budgeted, singleflighted one the CSR and dense
/// readers use. It was a separate count-only implementation with neither, on
/// the reasoning that CSC analytical kernels (DE on gene chunks, projected
/// `col_*`) access shards in column-range order with limited reuse. The reuse
/// argument holds and is why [`Self::new`] still opens **count-only** — a
/// `usize::MAX` byte budget, so only the shard count bounds it, exactly as
/// before. What it did not justify was a third copy of the eviction loop, or
/// concurrent readers of one cold shard each decoding their own copy.
///
/// [`Self::with_byte_budget`] bounds the cache in bytes instead.
pub struct BackedCscReader {
    reader: ScxReader,
    index: BackedCscIndex,
    n_obs: usize,
    n_vars: usize,
    /// Sorted CSC shard catalog entries (catalog index == sorted shard
    /// index). Pre-cached at construction so we don't re-scan the
    /// catalog on every read.
    sorted_entries: Vec<FullCatalogEntry>,
    /// Decoded-shard cache + singleflight, keyed by shard index.
    ///
    /// `BackedCscReader::new` / `for_modality` / `for_layer` open it with a
    /// `usize::MAX` byte budget, i.e. count-only — the behaviour the
    /// hand-rolled cache this replaced had no way to do otherwise.
    /// [`Self::with_byte_budget`] is the way to bound it in bytes.
    cache: ShardCache<usize, ScxCsc>,
}

impl BackedCscReader {
    /// `Ok(())` unless this reader is watching its file and the file has
    /// changed since it was opened. See [`ScxReader::check_fresh`].
    ///
    /// Section reads are already covered — they funnel through
    /// `ScxReader::section_bytes`. This exists for the answers that never
    /// touch a section: the cached `shape` / shard-count scalars a caller
    /// reads straight off the index.
    pub fn check_fresh(&self) -> Result<()> {
        self.reader.check_fresh()
    }

    /// Create a new backed CSC reader from an [`ScxReader`], scoped
    /// to the global / single-modality CSC shards
    /// (`modality_id == 0`). Convenience wrapper around
    /// [`Self::for_modality`] kept for backward compatibility with
    /// v1 / single-modality v2 callers.
    ///
    /// `cache_shards`: number of decoded CSC shards to cache (0 = no
    /// cache). The underlying file must have CSC sidecar shards
    /// (`reader.header().has_csc()`); otherwise `read_csc_shard` will
    /// always return an out-of-bounds error.
    pub fn new(reader: ScxReader, cache_shards: usize) -> Result<Self> {
        Self::for_modality(reader, 0, cache_shards)
    }

    /// Create a backed CSC reader scoped to a specific modality.
    /// Filters CSC shards by `(SectionType::CscShard, modality_id)`
    /// so each modality gets its own LRU cache and shard range —
    /// avoids cache thrashing under interleaved access patterns
    /// (e.g. totalVI training touching RNA + ADT in the same step).
    pub fn for_modality(reader: ScxReader, modality_id: u8, cache_shards: usize) -> Result<Self> {
        Self::for_modality_with_byte_budget(reader, modality_id, cache_shards, usize::MAX)
    }

    /// Global-modality CSC reader with both a count cap and a byte cap on the
    /// decoded-shard LRU. The CSR and dense readers have had a byte budget
    /// since their caches did; CSC could not express one until all three
    /// shared an implementation.
    ///
    /// `bytes_budget` counts a decoded `ScxCsc` by its components
    /// (`indptr.len()*8 + indices.len()*4 + data.len()*4`), the same model the
    /// loader's memory-budget auto-tune uses.
    pub fn with_byte_budget(
        reader: ScxReader,
        cache_shards: usize,
        bytes_budget: usize,
    ) -> Result<Self> {
        Self::for_modality_with_byte_budget(reader, 0, cache_shards, bytes_budget)
    }

    fn for_modality_with_byte_budget(
        reader: ScxReader,
        modality_id: u8,
        cache_shards: usize,
        bytes_budget: usize,
    ) -> Result<Self> {
        let index = BackedCscIndex::from_catalog_for_modality(reader.catalog(), modality_id);
        let n_obs = reader.n_obs() as usize;
        // For multimodal files the per-modality `n_vars` lives on
        // the modality table; the file-level `header.n_vars` is the
        // primary modality's count or aggregate. Prefer the modality
        // info when available.
        let n_vars = match reader.modality_info(modality_id) {
            Some(info) => info.n_vars as usize,
            None => reader.n_vars() as usize,
        };
        let sorted_entries: Vec<FullCatalogEntry> = reader
            .catalog()
            .csc_shards_for_modality(modality_id)
            .into_iter()
            .cloned()
            .collect();
        check_csc_sidecar_fresh(reader.catalog(), &sorted_entries)?;
        Ok(BackedCscReader {
            reader,
            index,
            n_obs,
            n_vars,
            sorted_entries,
            cache: ShardCache::inline(cache_shards, bytes_budget),
        })
    }

    /// Create a backed CSC reader scoped to a layer's CSC sidecar
    /// for a given modality. Filters by
    /// `(SectionType::LayerCscShard, modality_id, layer_name)`.
    pub fn for_layer(
        reader: ScxReader,
        modality_id: u8,
        layer_name: &str,
        cache_shards: usize,
    ) -> Result<Self> {
        let index =
            BackedCscIndex::from_catalog_for_layer(reader.catalog(), modality_id, layer_name);
        let n_obs = reader.n_obs() as usize;
        let n_vars = match reader.modality_info(modality_id) {
            Some(info) => info.n_vars as usize,
            None => reader.n_vars() as usize,
        };
        let sorted_entries: Vec<FullCatalogEntry> = reader
            .catalog()
            .layer_csc_shards_for_modality(modality_id, layer_name)
            .into_iter()
            .cloned()
            .collect();
        check_csc_sidecar_fresh(reader.catalog(), &sorted_entries)?;
        Ok(BackedCscReader {
            reader,
            index,
            n_obs,
            n_vars,
            sorted_entries,
            cache: ShardCache::inline(cache_shards, usize::MAX),
        })
    }

    /// Borrow the CSC shard index.
    pub fn index(&self) -> &BackedCscIndex {
        &self.index
    }

    /// Enable shard-cache metrics. Returns a cloneable
    /// `Arc<CacheMetrics>` so callers can sample counters without going
    /// through the cache lock. The returned handle's `hits` /
    /// `misses` / `evictions` reflect CSC-side activity; CSR metrics
    /// live on `BackedCsrReader::enable_metrics()` separately.
    ///
    /// **Idempotent, and that is a change.** This used to install a fresh
    /// `CacheMetrics` on every call, so a second call reset the counters to
    /// zero and orphaned the handle the first caller was holding. It now
    /// returns the same handle every time, with counters that accumulate for
    /// the life of the reader — matching `BackedCsrReader`, which has always
    /// behaved this way via the shared cache's `OnceLock`.
    ///
    /// A caller that wants a fresh measurement window must therefore snapshot
    /// the counters and subtract, rather than re-enabling. Nothing in-tree did
    /// the latter, but this is public API and the old behaviour was reachable.
    pub fn enable_metrics(&mut self) -> Arc<CacheMetrics> {
        self.cache.enable_metrics()
    }

    /// Borrow the metrics handle, if enabled.
    pub fn metrics(&self) -> Option<&Arc<CacheMetrics>> {
        self.cache.metrics()
    }

    /// Number of CSC shards.
    pub fn n_shards(&self) -> usize {
        self.index.n_shards()
    }

    /// Total number of observations (rows). CSC shards span the full
    /// row axis, so this equals the file's `n_obs`.
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// Number of variables (columns).
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Read and decode a CSC shard without consulting the cache.
    /// Useful for one-shot streaming passes where caching would only
    /// add overhead.
    pub fn read_shard_uncached(&self, shard_idx: usize) -> Result<ScxCsc> {
        let entry = self
            .sorted_entries
            .get(shard_idx)
            .ok_or(ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: self.sorted_entries.len(),
            })?;
        let (indptr, indices, data) = self.reader.read_shard_from_entry(entry)?;
        let n_cols_in_shard = indptr.len().saturating_sub(1);
        Ok(ScxCsc::new_unchecked(
            (self.n_obs, n_cols_in_shard),
            indptr,
            indices,
            data,
        ))
    }

    /// Read and optionally cache a CSC shard, returning a shared
    /// `Arc<ScxCsc>`. On cache hit increments `metrics.hits`; on miss
    /// (or no cache) increments `metrics.misses` and decodes.
    ///
    /// Singleflighted as of the shared cache: concurrent readers of the same
    /// cold shard decode it once and the rest wait, where before each decoded
    /// its own copy. On a gene-major DE sweep that is the difference between
    /// N decodes of a 5000-column shard and one.
    pub fn read_shard_cached(&self, shard_idx: usize) -> Result<Arc<ScxCsc>> {
        // See `BackedCsrReader::read_shard_cached_arc`: a cache hit bypasses
        // `section_bytes` entirely.
        self.check_fresh()?;
        self.cache.get_or_decode(shard_idx, || {
            Ok(Arc::new(self.read_shard_uncached(shard_idx)?))
        })
    }

    /// Read a contiguous column slice across CSC shards.
    ///
    /// Skips shards whose `[col_start, col_end)` does not intersect
    /// `col_range`; `col_slice`s partial-overlap shards post-decode.
    /// Cached: each overlapping shard is fetched via
    /// [`Self::read_shard_cached`].
    pub fn read_csc_columns(&self, col_range: std::ops::Range<u32>) -> Result<ScxCsc> {
        let c_lo = col_range.start as u64;
        let c_hi = col_range.end as u64;
        if c_lo >= c_hi {
            return Ok(ScxCsc::new_unchecked(
                (self.n_obs, 0),
                vec![0],
                Vec::new(),
                Vec::new(),
            ));
        }
        let shard_indices = self.index.shards_for_col_range(c_lo, c_hi);

        let mut decoded: Vec<ScxCsc> = Vec::with_capacity(shard_indices.len());
        for shard_idx in shard_indices {
            let (shard_lo, shard_hi) = self.index.shard_col_range(shard_idx).ok_or_else(|| {
                ScxError::InvalidCatalog(format!(
                    "BackedCscIndex missing range for shard {shard_idx}"
                ))
            })?;
            let lo_in_shard = c_lo.saturating_sub(shard_lo) as usize;
            let hi_in_shard = (c_hi.min(shard_hi).saturating_sub(shard_lo)) as usize;

            // A row-group-framed (v2) CSC shard is column-group indexed,
            // so a gene-subset read decodes only the touched column-groups via the
            // block index (major axis = columns) instead of full-decoding the whole
            // shard through the LRU cache. Non-framed shards fall back to the cached
            // full-decode + `col_slice` path below.
            if hi_in_shard > lo_in_shard {
                if let Some(entry) = self.sorted_entries.get(shard_idx) {
                    let header = self.reader.read_shard_header(entry)?;
                    if header.shard_format_version
                        > crate::shard::DEFAULT_WRITE_SHARD_FORMAT_VERSION
                    {
                        let run = (lo_in_shard, hi_in_shard - lo_in_shard);
                        if let Some(mut runs) =
                            self.reader.decode_block_index_row_runs(entry, &[run])?
                        {
                            if let Some((indptr, indices, data)) = runs.pop() {
                                decoded.push(ScxCsc::new_unchecked(
                                    (self.n_obs, hi_in_shard - lo_in_shard),
                                    indptr,
                                    indices,
                                    data,
                                ));
                                continue;
                            }
                        }
                    }
                }
            }

            let csc = self.read_shard_cached(shard_idx)?;
            let sliced = if lo_in_shard == 0 && hi_in_shard == csc.n_cols() {
                (*csc).clone()
            } else {
                csc.col_slice(lo_in_shard, hi_in_shard).map_err(|e| {
                    ScxError::InvalidCatalog(format!(
                        "BackedCscReader col_slice failed for shard {shard_idx}: {e}"
                    ))
                })?
            };
            decoded.push(sliced);
        }
        concatenate_csc_along_cols(decoded, self.n_obs)
    }

    /// Read an arbitrary sorted column subset by collapsing it to
    /// contiguous runs and concatenating per-run `read_csc_columns`
    /// results. Mirrors `ScxReader::read_csc_columns_subset` but uses
    /// the cached shard reads.
    pub fn read_csc_columns_subset(&self, cols: &[u32]) -> Result<ScxCsc> {
        if cols.is_empty() {
            return Ok(ScxCsc::new_unchecked(
                (self.n_obs, 0),
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
                runs.push(self.read_csc_columns(run_start..run_end)?);
                run_start = c;
                run_end = c + 1;
            }
        }
        runs.push(self.read_csc_columns(run_start..run_end)?);
        if runs.len() == 1 {
            return Ok(runs.pop().unwrap());
        }
        concatenate_csc_along_cols(runs, self.n_obs)
    }
}

/// Concatenate a list of CSC parts along the column axis.
/// Internal helper — the same logic also lives in `reader/matrix.rs` as a
/// private free function. Local copy avoids cross-module visibility
/// changes.
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

impl crate::shard_source::ColumnShardSource for BackedCscReader {
    fn n_csc_shards(&self) -> usize {
        BackedCscReader::n_shards(self)
    }

    fn n_obs(&self) -> usize {
        BackedCscReader::n_obs(self)
    }

    fn n_vars(&self) -> usize {
        BackedCscReader::n_vars(self)
    }

    fn read_csc_shard(&self, shard_idx: usize) -> Result<ScxCsc> {
        // Owned clone — trait return type is `ScxCsc`, not `Arc<_>`.
        Ok((*self.read_shard_cached(shard_idx)?).clone())
    }

    fn read_csc_columns(&self, col_range: std::ops::Range<u32>) -> Result<ScxCsc> {
        BackedCscReader::read_csc_columns(self, col_range)
    }

    fn csc_shard_col_range(&self, shard_idx: usize) -> Option<(u32, u32)> {
        let (lo, hi) = self.index.shard_col_range(shard_idx)?;
        let lo = u32::try_from(lo).ok()?;
        let hi = u32::try_from(hi).ok()?;
        Some((lo, hi))
    }

    fn csc_shard_size_hint(&self) -> Option<crate::ShardSizeHint> {
        self.index.size_hint()
    }

    /// Delegates to the precomputed column index's binary search rather than
    /// re-running the trait default's O(n_shards) scan. Same boundary
    /// condition — `shards_for_col_range` is half-open on both ends — which is
    /// what the shared-predicate test in `shard_source.rs` pins.
    fn csc_shards_for_col_range(&self, col_range: std::ops::Range<u32>) -> Vec<usize> {
        if col_range.start >= col_range.end {
            return Vec::new();
        }
        self.index
            .shards_for_col_range(col_range.start as u64, col_range.end as u64)
    }
}

// `concatenate_csr` moved to `scx_sparse::concatenate_csr` (operates purely on
// `ScxCsr`, not format I/O).
