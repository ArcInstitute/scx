//! [`BackedPairwiseReader`]: bounded row-range reads over a row-sharded
//! pairwise mapping (`obsp/<name>`), the COO counterpart of
//! [`BackedDenseReader`].
//!
//! # Why this is not the dense reader with a different section type
//!
//! An `obsp` section is **Arrow IPC COO**, not CSR and not dense: three
//! columns `row` / `col` / `data`, one Arrow row per non-zero triple, with
//! `row` stored as a **global** index (`writer::write_obsp_shard_coo`). Three
//! consequences shape everything below.
//!
//! 1. **`batch.num_rows()` is nnz, not a row span.** [`BackedDenseReader`]'s
//!    gather does `take(row - shard_row_start)` over batch rows; on a COO shard
//!    that indexes a *triple*. The row span lives only in the stamped
//!    `n_shard_rows` (sharded) or the `n_rows` schema metadata key (legacy) —
//!    which is what [`crate::reader::LegacyRowCount`] exists to distinguish.
//! 2. **There is no indptr on disk.** A CSR view of a row range has to be
//!    *derived*, by counting triples per row. That is what `read_rows_range`
//!    does, and it is why a caller cannot get a row's degree without reading
//!    the row.
//! 3. **Triples are not guaranteed sorted by `row` within a shard.**
//!    The shared emitter (`crate::mapping_shards::for_each_coo_mapping_shard`)
//!    buckets by `row / step` in one linear pass preserving *input* order; only
//!    the h5py streaming path happens to come out row-sorted. So the row
//!    grouping is a counting sort, never a binary search, and the column order
//!    within a row is established here rather than assumed.
//!
//! # Retention
//!
//! One decoded shard is memoised, not an LRU. A neighbourhood plan build is a
//! single ascending pass over the row axis: consecutive ranges land in the same
//! shard, and no shard is ever revisited. Holding more would cost peak RSS for
//! zero additional hits — the same reasoning the row-group admission verdict
//! uses on the CSR side.
//!
//! # Bounded-memory caveat
//!
//! Bounded reads need a **sharded** obsp. A legacy single-section
//! `ObspEmbedding` is one Arrow batch and must be deserialised whole whatever
//! range is asked for; the range is then applied to the decoded triples.
//!
//! That branch is no longer what the rewrite ops produce — since phase 9,
//! `scx sort` and `scx compact` emit all four mapping families as shards — but
//! it is still reachable, and not hypothetically: a file written before
//! phase 9 has an unsharded graph, and `scx subset` drops obsp entirely.
//!
//! # Row space
//!
//! Everything here is **physical**. Deletion vectors are not consulted, exactly
//! as [`ScxReader::read_obs`] is physical and `read_obs_filtered` layers the
//! policy on top. The neighbourhood plan builder applies the keep mask itself,
//! because "drop an edge whose either endpoint is deleted" is a decision about
//! graphs, not about bytes.

use super::*;

use crate::reader::{LegacyRowCount, MappingLayout};

/// One decoded pairwise shard: the COO triples, converted once out of Arrow
/// into the widest coordinate form so the row-range scan does not re-dispatch
/// on dtype per triple.
struct PairwiseShard {
    rows: Vec<i64>,
    cols: Vec<i64>,
    vals: Vec<f32>,
}

/// One row range of a pairwise mapping, as CSR.
///
/// `indptr` has `n_rows + 1` entries and starts at 0; row `i` of the range is
/// global row `row_start + i`. `indices` are **global** column ids, ascending
/// within each row, and `data` is parallel to them. Duplicate `(row, col)`
/// pairs are passed through rather than summed — every in-tree writer emits a
/// canonical matrix, so a duplicate means the source had one.
#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseRows {
    pub indptr: Vec<i64>,
    pub indices: Vec<i64>,
    pub data: Vec<f32>,
    /// Global index of this range's first row.
    pub row_start: u64,
    /// Rows in this range (`indptr.len() - 1`).
    pub n_rows: u64,
    /// The **matrix's** column count, not this range's.
    pub n_cols: u64,
}

impl PairwiseRows {
    /// Non-zeros in the range.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Columns and values of range-local row `i`.
    ///
    /// # Panics
    ///
    /// If `i >= n_rows`. Asserted rather than left to the slice index so the
    /// message names the contract: this is a caller bug, not malformed input,
    /// and the reader's error path is for the latter.
    pub fn row(&self, i: usize) -> (&[i64], &[f32]) {
        assert!(
            i < self.n_rows as usize,
            "PairwiseRows::row({i}) on a range of {} rows",
            self.n_rows
        );
        let lo = self.indptr[i] as usize;
        let hi = self.indptr[i + 1] as usize;
        (&self.indices[lo..hi], &self.data[lo..hi])
    }
}

/// Bounded row-range reader over `obsp/<name>`.
///
/// # Fork safety
///
/// The shard memo is per-instance, never global — same contract as
/// [`BackedDenseReader`] and [`BackedCsrReader`].
pub struct BackedPairwiseReader {
    reader: ScxReader,
    name: String,
    n_rows: u64,
    n_cols: u64,
    /// Ordered by `row_start`; a legacy single section is one entry.
    sorted_entries: Vec<crate::reader::MappingShardLayoutEntry>,
    /// Single-entry memo: `(shard_idx, decoded)`. See the module docs for why
    /// this is not an LRU.
    memo: Mutex<Option<(usize, Arc<PairwiseShard>)>>,
    /// Decoded-shard reads served from the memo, for tests and diagnostics.
    memo_hits: AtomicU64,
    memo_misses: AtomicU64,
}

impl BackedPairwiseReader {
    /// Open a bounded reader over `obsp/<name>`.
    pub fn new_obsp(reader: ScxReader, name: &str) -> Result<Self> {
        let n_obs = reader.n_obs();
        let layout = reader.row_sharded_mapping_layout(
            "obsp",
            name,
            SectionType::ObspEmbeddingShard,
            SectionType::ObspEmbedding,
            LegacyRowCount::SchemaNRows,
        )?;
        Self::from_layout(reader, name, layout, n_obs)
    }

    fn from_layout(
        reader: ScxReader,
        name: &str,
        layout: MappingLayout,
        n_obs: u64,
    ) -> Result<Self> {
        // The COO schema is exactly `row` / `col` / `data`. Anything else is a
        // section written by something this reader does not understand, and
        // reading it as COO would answer confidently wrong.
        // Three columns, and the right three. A batch that happens to have
        // three of something else would be read as coordinates and answer
        // confidently wrong; the names are the only thing distinguishing a COO
        // section from an arbitrary triple.
        let names: Vec<&str> = layout.fields.iter().map(|f| f.name().as_str()).collect();
        if names != ["row", "col", "data"] {
            return Err(ScxError::InvalidCatalog(format!(
                "obsp/{name}: expected the 3-column COO schema row/col/data, found {names:?}"
            )));
        }
        let n_cols = layout.matrix_n_cols.ok_or_else(|| {
            ScxError::InvalidCatalog(format!(
                "obsp/{name}: no parseable 'n_cols' schema metadata, so the matrix's column \
                 extent is unknown"
            ))
        })?;
        // `obsp` is obs x obs by definition, and the plan builders read a
        // column index as a row id — so a non-square one hands the gather rows
        // that do not exist. Refused here, where the shape is known, rather
        // than at gather time where it is someone else's error message.
        // Square, AND the file's own obs axis. Squareness alone lets a 500x500
        // graph open on a 1,000-cell file, and nothing downstream would say so:
        // the plan builder would emit 500 centres, `read_obsp_rows(logical=False)`
        // would report a 500-column physical graph against an `n_obs_physical`
        // of 1,000, and the only symptom is half the cells quietly missing.
        if n_cols != layout.n_rows || layout.n_rows != n_obs {
            return Err(ScxError::InvalidCatalog(format!(
                "obsp/{name}: {} rows x {n_cols} columns on a file with {n_obs} observations — \
                 a pairwise obs mapping must be square and on the file's own obs axis",
                layout.n_rows
            )));
        }
        let sorted_entries = layout.entries;
        Ok(BackedPairwiseReader {
            reader,
            name: name.to_string(),
            n_rows: layout.n_rows,
            n_cols,
            sorted_entries,
            memo: Mutex::new(None),
            memo_hits: AtomicU64::new(0),
            memo_misses: AtomicU64::new(0),
        })
    }

    /// `Ok(())` unless this reader is watching its file and the file changed.
    pub fn check_fresh(&self) -> Result<()> {
        self.reader.check_fresh()
    }

    /// Logical mapping name (e.g. `"connectivities"`).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Rows of the pairwise matrix (== `n_obs`).
    pub fn n_rows(&self) -> u64 {
        self.n_rows
    }

    /// Columns of the pairwise matrix (== `n_obs` for `obsp`).
    pub fn n_cols(&self) -> u64 {
        self.n_cols
    }

    /// Number of row-shards. `1` for a legacy single section.
    pub fn shard_count(&self) -> usize {
        self.sorted_entries.len()
    }

    /// `true` when the mapping is stored as one unsharded section, which is
    /// what makes a read unbounded. Since phase 9 no rewrite op writes one;
    /// a file predating it does.
    pub fn is_legacy_single_section(&self) -> bool {
        self.sorted_entries
            .iter()
            .any(|e| e.section_type == SectionType::ObspEmbedding)
    }

    /// `(memo hits, memo misses)` over the life of the reader.
    pub fn memo_metrics(&self) -> (u64, u64) {
        (
            self.memo_hits.load(Ordering::Relaxed),
            self.memo_misses.load(Ordering::Relaxed),
        )
    }

    /// Positions in `sorted_entries` whose stamped row span intersects
    /// `[start, end)`.
    ///
    /// A linear scan, deliberately: obsp shard counts are in the tens, and a
    /// binary search here would need the cover invariant the layout resolver
    /// has already checked, stated a second time.
    fn shards_overlapping(&self, start: u64, end: u64) -> Vec<usize> {
        self.sorted_entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                let s = e.row_start;
                let t = e.row_start.saturating_add(e.n_shard_rows);
                s < end && start < t
            })
            .map(|(i, _)| i)
            .collect()
    }

    fn shard(&self, shard_idx: usize) -> Result<Arc<PairwiseShard>> {
        self.check_fresh()?;
        {
            let memo = self.memo.lock().expect("pairwise shard memo poisoned");
            if let Some((idx, shard)) = memo.as_ref() {
                if *idx == shard_idx {
                    self.memo_hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(Arc::clone(shard));
                }
            }
        }
        self.memo_misses.fetch_add(1, Ordering::Relaxed);
        let decoded = Arc::new(self.decode_shard(shard_idx)?);
        let mut memo = self.memo.lock().expect("pairwise shard memo poisoned");
        *memo = Some((shard_idx, Arc::clone(&decoded)));
        Ok(decoded)
    }

    fn decode_shard(&self, shard_idx: usize) -> Result<PairwiseShard> {
        let lite = *self
            .sorted_entries
            .get(shard_idx)
            .ok_or(ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: self.sorted_entries.len(),
            })?;
        // `read_dense_mapping_entry` is Arrow-IPC-generic despite its name —
        // `scx-ops::merge_pairwise` already reads COO obsp shards through it.
        let batch = self
            .reader
            // The layout entry is retained as-is (it carries exactly these
            // fields); the section reader wants a catalog entry, so one is
            // built here at the single call site.
            .read_dense_mapping_entry(&FullCatalogEntry {
                name: String::new(),
                offset: lite.offset,
                length: lite.length,
                section_type: lite.section_type,
                checksum: [0u8; 32],
                modality_id: lite.modality_id,
                stats: None,
            })?;
        // Per shard, not once at open. The layout resolver takes `n_cols` from
        // shard 0 only, so a later shard with a different schema reached
        // `batch.column(2)` — and arrow's `column` **panics** out of bounds
        // rather than erroring, which on a truncated file is a process abort
        // where the format's rule is "readers return errors, not panics".
        if batch.num_columns() != 3 {
            return Err(ScxError::InvalidCatalog(format!(
                "obsp/{}: shard {shard_idx} has {} columns, expected the 3-column COO \
                 (row/col/data)",
                self.name,
                batch.num_columns()
            )));
        }
        let logical = format!("obsp/{}", self.name);
        let rows = coo_coord_column(&batch, 0, &logical)?;
        let cols = coo_coord_column(&batch, 1, &logical)?;
        let vals = coo_data_column(&batch, &self.name)?;
        if rows.len() != cols.len() || rows.len() != vals.len() {
            return Err(ScxError::InvalidCatalog(format!(
                "obsp/{}: COO columns disagree in length ({}, {}, {})",
                self.name,
                rows.len(),
                cols.len(),
                vals.len()
            )));
        }
        // Every triple must belong to the span this shard is stamped with.
        // Filtering silently — which the range scan below would otherwise do —
        // makes a misfiled triple invisible **twice**: skipped here, and never
        // looked for in the shard that covers its row. Two wrong answers, no
        // error. A legacy single section is stamped `[0, n_rows)`, so this
        // checks the whole matrix's extent there.
        let lo = lite.row_start;
        let hi = lite.row_start.saturating_add(lite.n_shard_rows);
        for &r in &rows {
            if r < 0 || (r as u64) < lo || (r as u64) >= hi {
                return Err(ScxError::InvalidCatalog(format!(
                    "obsp/{}: shard {shard_idx} is stamped [{lo}, {hi}) but carries a triple at \
                     row {r}",
                    self.name
                )));
            }
        }
        Ok(PairwiseShard { rows, cols, vals })
    }

    /// Read `[start, end)` of the row axis as CSR, in physical row space.
    ///
    /// Decodes only the shards whose stamped span intersects the range. Peak
    /// memory is one shard plus the range's own non-zeros.
    pub fn read_rows_range(&self, start: u64, end: u64) -> Result<PairwiseRows> {
        // All three conditions here, not just `end > n_rows`. An inverted
        // range (`start > end`) or an out-of-range `start` used to fall through
        // to the empty-range arm below and come back `Ok` with a `row_start`
        // that names no row — a caller looping over blocks then reads nothing
        // and has no way to tell that from a genuinely empty graph.
        if start > end || start > self.n_rows || end > self.n_rows {
            return Err(ScxError::InvalidCatalog(format!(
                "obsp/{}: invalid row range [{start}, {end}) over {} rows",
                self.name, self.n_rows
            )));
        }
        if start == end {
            return Ok(PairwiseRows {
                indptr: vec![0],
                indices: Vec::new(),
                data: Vec::new(),
                row_start: start,
                n_rows: 0,
                n_cols: self.n_cols,
            });
        }
        let n = (end - start) as usize;

        // One pass over the overlapping shards, collecting only in-range
        // triples. A second pass would re-decode, since only one shard is
        // memoised; and the collected triples are the output's own size.
        let mut local_rows: Vec<u64> = Vec::new();
        let mut pairs: Vec<(i64, f32)> = Vec::new();
        for shard_idx in self.shards_overlapping(start, end) {
            let shard = self.shard(shard_idx)?;
            for i in 0..shard.rows.len() {
                // Non-negative and inside this shard's stamped span: checked
                // once per shard in `decode_shard`, not once per triple here.
                let r = shard.rows[i];
                let r = r as u64;
                if r < start || r >= end {
                    // In range of the shard (decode_shard proved that) but not
                    // of the request — the ordinary case for a shard the range
                    // only partly covers.
                    continue;
                }
                let c = shard.cols[i];
                if c < 0 || c as u64 >= self.n_cols {
                    return Err(ScxError::InvalidCatalog(format!(
                        "obsp/{}: COO column index {c} out of range [0, {})",
                        self.name, self.n_cols
                    )));
                }
                local_rows.push(r - start);
                pairs.push((c, shard.vals[i]));
            }
        }

        // Counting sort into CSR. The triples arrive in whatever order the
        // writer left them (module docs, point 3), so this is what establishes
        // the row grouping rather than confirming it.
        let mut indptr = vec![0i64; n + 1];
        for &lr in &local_rows {
            indptr[lr as usize + 1] += 1;
        }
        for i in 0..n {
            indptr[i + 1] += indptr[i];
        }
        let nnz = pairs.len();
        let mut scattered: Vec<(i64, f32)> = vec![(0, 0.0); nnz];
        let mut cursor: Vec<i64> = indptr[..n].to_vec();
        for (k, &lr) in local_rows.iter().enumerate() {
            let slot = &mut cursor[lr as usize];
            scattered[*slot as usize] = pairs[k];
            *slot += 1;
        }

        // Column order within a row is established here, not inherited.
        for i in 0..n {
            let lo = indptr[i] as usize;
            let hi = indptr[i + 1] as usize;
            scattered[lo..hi].sort_by_key(|&(c, _)| c);
        }

        let mut indices = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);
        for (c, v) in scattered {
            indices.push(c);
            data.push(v);
        }
        Ok(PairwiseRows {
            indptr,
            indices,
            data,
            row_start: start,
            n_rows: n as u64,
            n_cols: self.n_cols,
        })
    }
}

/// A **borrowed** view of one COO coordinate column, over either the `Int32`
/// (v1) or `Int64` (v2 wide-axis) form the writer picks between.
///
/// Exists so a caller that only needs to *scan* the column does not have to
/// materialise it. [`crate::mapping_shards`] buckets by row, one pass, on
/// batches that can carry hundreds of millions of triples — at 8 B per
/// coordinate a copy there is gigabytes for nothing.
pub(crate) enum CooCoordColumn<'a> {
    I32(&'a arrow::array::Int32Array),
    I64(&'a arrow::array::Int64Array),
}

impl CooCoordColumn<'_> {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        use arrow::array::Array;
        match self {
            Self::I32(a) => a.len(),
            Self::I64(a) => a.len(),
        }
    }

    #[inline]
    pub(crate) fn at(&self, i: usize) -> i64 {
        match self {
            Self::I32(a) => a.values()[i] as i64,
            Self::I64(a) => a.values()[i],
        }
    }
}

/// Borrow COO coordinate column `col_idx`, accepting both the `Int32` (v1) and
/// `Int64` (v2 wide-axis) forms.
///
/// `logical` is the fully qualified section label (`"obsp/connectivities"`),
/// not the bare key: this is shared with [`crate::mapping_shards`], which
/// buckets `varp` too, so the family cannot be assumed here.
pub(crate) fn coo_coord_borrow<'a>(
    batch: &'a arrow::array::RecordBatch,
    col_idx: usize,
    logical: &str,
) -> Result<CooCoordColumn<'a>> {
    use arrow::array::{Int32Array, Int64Array};
    let col = batch.column(col_idx);
    // `values()` returns the raw buffer, so a null slot reads back as whatever
    // happens to be there — a plausible-looking coordinate with nothing to
    // mark it. Nothing in-tree writes a nullable COO column; a file that has
    // one was not written by this workspace and is refused rather than read.
    if col.null_count() > 0 {
        return Err(ScxError::InvalidCatalog(format!(
            "{logical}: COO coordinate column {col_idx} has {} null entries; a COO triple \
             has no meaning with a missing coordinate",
            col.null_count()
        )));
    }
    if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
        return Ok(CooCoordColumn::I32(a));
    }
    if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        return Ok(CooCoordColumn::I64(a));
    }
    Err(ScxError::InvalidCatalog(format!(
        "{logical}: COO coordinate column {col_idx} has dtype {:?}, expected Int32 or Int64",
        col.data_type()
    )))
}

/// [`coo_coord_borrow`] materialised. The reader wants an owned `Vec<i64>`
/// (it keeps decoded shards in the memo); the emitter does not.
pub(crate) fn coo_coord_column(
    batch: &arrow::array::RecordBatch,
    col_idx: usize,
    logical: &str,
) -> Result<Vec<i64>> {
    let col = coo_coord_borrow(batch, col_idx, logical)?;
    Ok(match col {
        CooCoordColumn::I32(a) => a.values().iter().map(|&v| v as i64).collect(),
        CooCoordColumn::I64(a) => a.values().to_vec(),
    })
}

/// Read the COO `data` column as `f32`. `Float64` is accepted because
/// `scx-ops::compact::remap_obsp_coo` preserves a source matrix's `Float64`
/// values; the loader's contract is f32 and the narrow is where it happens.
fn coo_data_column(batch: &arrow::array::RecordBatch, name: &str) -> Result<Vec<f32>> {
    use arrow::array::{Float32Array, Float64Array};
    let col = batch.column(2);
    if col.null_count() > 0 {
        return Err(ScxError::InvalidCatalog(format!(
            "obsp/{name}: COO data column has {} null entries",
            col.null_count()
        )));
    }
    if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
        return Ok(a.values().to_vec());
    }
    if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        return Ok(a.values().iter().map(|&v| v as f32).collect());
    }
    Err(ScxError::InvalidCatalog(format!(
        "obsp/{name}: COO data column has dtype {:?}, expected Float32 or Float64",
        col.data_type()
    )))
}

#[cfg(test)]
#[path = "pairwise_tests.rs"]
mod tests;
