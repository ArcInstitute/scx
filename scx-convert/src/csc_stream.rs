// Streaming CSC-on-disk h5ad → CSR shard reader.
//
// Phase 2 of REAL-WORLD-UX-FEATS. Two routes share the
// [`open_csc_streaming`] dispatcher:
//
// 1. **In-memory** ([`MaterializedCsrStream`]): when the budget arithmetic
//    allows, load the full CSC, run the existing
//    [`crate::csc_transpose::csc_to_csr`] scatter, and yield CSR shards
//    from the materialised buffers.
// 2. **External-memory** ([`CscToCsrExternalTransposer`]): for files that
//    don't fit in `memory_budget`, do a single column-chunk pass writing
//    `(row, col, value)` triples to per-bucket temp files, then yield
//    shards by loading one bucket at a time, sorting, summing duplicates,
//    and slicing off `target_rows` rows per `next_csr_shard` call.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use byteorder::{LittleEndian, WriteBytesExt};

use super::h5ad_read::{read_i64_dataset, read_x_matrix_at};
use super::h5ad_stream::{read_slice_f32, read_slice_i32};
use super::pipeline::{ConvertError, ConvertOptions};
use super::stream::{CsrShardStream, StreamedCsrShard};
use super::warnings::WarningSink;

/// Open a CSC-on-disk matrix as a [`CsrShardStream`]. Picks the
/// in-memory or external-memory route based on
/// [`ConvertOptions::memory_budget`]:
///
/// - `None` or `Some(budget)` where the materialised CSR working set
///   (≈ `16 × nnz + 16 × n_obs` bytes) fits → in-memory.
/// - Otherwise → external bucketed transpose. Temp files live under
///   `<opts.temp_dir>/scx-transpose-<pid>-<random>/`; the session
///   directory is auto-deleted on success and on most failure modes
///   (`tempfile::TempDir` drop). Set `SCX_KEEP_TEMP=1` to persist
///   them for forensic inspection.
pub fn open_csc_streaming(
    file: &hdf5::File,
    path: &str,
    opts: &ConvertOptions,
    sink: &mut WarningSink,
) -> Result<Box<dyn CsrShardStream>, ConvertError> {
    let group = file.group(path)?;
    let shape: Vec<i64> = group.attr("shape")?.read_1d()?.to_vec();
    if shape.len() != 2 {
        return Err(ConvertError::Other(format!(
            "expected 2D shape attr on '{path}', got {}-D",
            shape.len()
        )));
    }
    let n_obs = shape[0] as u64;
    let n_vars = shape[1] as u64;

    // Resident col_indptr — (n_vars + 1) × 8 bytes.
    let col_indptr_i64 = read_i64_dataset(&group.dataset("indptr")?)?;
    if col_indptr_i64.len() as u64 != n_vars + 1 {
        return Err(ConvertError::Other(format!(
            "CSC indptr length {} != n_vars + 1 ({})",
            col_indptr_i64.len(),
            n_vars + 1
        )));
    }
    let nnz: u64 = *col_indptr_i64.last().unwrap_or(&0) as u64;

    // Routing arithmetic. The in-memory threshold is the working
    // set of the materialised CSR (indptr + indices + values + an
    // equal-sized scratch buffer from the in-memory scatter
    // transpose); the spec calls out `16 × nnz + 16 × n_obs`.
    const RECORD_BYTES: u64 = 16; // u64 row + u32 col + f32 value
    let in_memory_threshold = nnz
        .saturating_mul(RECORD_BYTES)
        .saturating_add(n_obs.saturating_mul(16));
    let use_external = matches!(opts.memory_budget, Some(b) if b < in_memory_threshold);

    if !use_external {
        return Ok(Box::new(MaterializedCsrStream::open(file, path)?));
    }

    let budget = opts
        .memory_budget
        .expect("external route implies budget set");
    if budget < 4 * RECORD_BYTES {
        return Err(ConvertError::Other(format!(
            "memory_budget {budget} bytes too small for CSC external transpose; \
             need at least {} bytes (≥ 4 records of {RECORD_BYTES} bytes each)",
            4 * RECORD_BYTES
        )));
    }

    let indices_ds = group.dataset("indices")?;
    let data_ds = group.dataset("data")?;
    CscToCsrExternalTransposer::open(
        path,
        n_obs,
        n_vars,
        nnz,
        col_indptr_i64,
        indices_ds,
        data_ds,
        budget,
        opts.temp_dir.clone(),
        sink,
    )
    .map(|r| Box::new(r) as Box<dyn CsrShardStream>)
}

/// Convenience wrapper for `/layers/{layer_name}` CSC matrices.
pub fn open_csc_layer_streaming(
    file: &hdf5::File,
    layer_name: &str,
    opts: &ConvertOptions,
    sink: &mut WarningSink,
) -> Result<Box<dyn CsrShardStream>, ConvertError> {
    open_csc_streaming(file, &format!("layers/{layer_name}"), opts, sink)
}

// ---------------------------------------------------------------------
// In-memory route
// ---------------------------------------------------------------------

/// In-memory CSR cursor. Loads the full CSC up front, runs the
/// existing [`csc_to_csr`] scatter (with `drop_explicit_zeros`), then
/// yields shards by slicing successive row ranges.
pub(crate) struct MaterializedCsrStream {
    n_obs: u64,
    n_vars: u64,
    source_name: String,
    indptr: Vec<i64>,
    indices: Vec<i32>,
    data: Vec<f32>,
    cursor: u64,
}

impl MaterializedCsrStream {
    fn open(file: &hdf5::File, path: &str) -> Result<Self, ConvertError> {
        // `read_x_matrix_at(.., Csc)` already runs csc_to_csr +
        // drop_explicit_zeros. Reuses the bulk-path code so the
        // in-memory CSC route is byte-identical to `h5ad_to_scx`.
        let (indptr, indices, data, n_obs, n_vars) =
            read_x_matrix_at(file, path, super::detect::MatrixFormat::Csc)?;
        Ok(MaterializedCsrStream {
            n_obs: n_obs as u64,
            n_vars: n_vars as u64,
            source_name: path.to_string(),
            indptr,
            indices,
            data,
            cursor: 0,
        })
    }
}

impl CsrShardStream for MaterializedCsrStream {
    fn n_obs(&self) -> u64 {
        self.n_obs
    }
    fn n_vars(&self) -> u64 {
        self.n_vars
    }
    fn source_matrix_name(&self) -> &str {
        &self.source_name
    }

    fn next_csr_shard(
        &mut self,
        target_rows: usize,
    ) -> Result<Option<StreamedCsrShard>, ConvertError> {
        if self.cursor >= self.n_obs {
            return Ok(None);
        }
        if target_rows == 0 {
            return Err(ConvertError::Other("target_rows must be > 0".into()));
        }
        let row_start = self.cursor as usize;
        let row_end = (row_start + target_rows).min(self.n_obs as usize);
        let n_rows = row_end - row_start;

        let base = self.indptr[row_start];
        let end_val = self.indptr[row_end];
        let nnz_start = usize::try_from(base)
            .map_err(|_| ConvertError::Other(format!("negative indptr base {base}")))?;
        let nnz_end = usize::try_from(end_val)
            .map_err(|_| ConvertError::Other(format!("negative indptr end {end_val}")))?;

        // Rebase the local indptr to start at zero (StreamedCsrShard
        // contract — first element always 0).
        let mut shard_indptr: Vec<u64> = Vec::with_capacity(n_rows + 1);
        for &v in &self.indptr[row_start..=row_end] {
            shard_indptr.push((v - base) as u64);
        }
        let shard_indices: Vec<u32> = self.indices[nnz_start..nnz_end]
            .iter()
            .map(|&v| v as u32)
            .collect();
        let shard_values: Vec<f32> = self.data[nnz_start..nnz_end].to_vec();

        self.cursor = row_end as u64;
        Ok(Some(StreamedCsrShard {
            row_start: row_start as u64,
            n_rows: n_rows as u32,
            n_cols: self.n_vars as u32,
            indptr: shard_indptr,
            indices: shard_indices,
            values: shard_values,
            source_name: Some(self.source_name.clone()),
            duplicates_merged: 0,
        }))
    }
}

// ---------------------------------------------------------------------
// External-memory route
// ---------------------------------------------------------------------

const RECORD_BYTES_USIZE: usize = 16;

/// Bucketed external transpose. Pass 1 (runs at `open` time) reads
/// the source CSC by column chunks and distributes
/// `(row, col, value)` triples to per-bucket temp files. Subsequent
/// `next_csr_shard` calls load the current bucket, sort by
/// `(row, col)`, sum duplicate coordinates, and slice off
/// `target_rows` rows per call.
pub(crate) struct CscToCsrExternalTransposer {
    n_obs: u64,
    n_vars: u64,
    source_name: String,
    bucket_rows: u64,
    /// Auto-deleted on drop unless `SCX_KEEP_TEMP=1`. Holds the
    /// per-bucket temp files we wrote in pass 1. `bucket_paths`
    /// indexes into this directory.
    _temp_dir: tempfile::TempDir,
    bucket_paths: Vec<PathBuf>,
    /// Next global row index to emit. Monotonic across shards and
    /// across bucket boundaries.
    row_cursor: u64,
    /// Index of the bucket currently sitting in `current_records`
    /// (i.e. records whose row falls in
    /// `[loaded_bucket × bucket_rows, (loaded_bucket+1) × bucket_rows)`).
    /// `usize::MAX` means "no bucket loaded yet".
    loaded_bucket: usize,
    /// Sorted + duplicate-summed `(row, col, value)` triples for the
    /// currently loaded bucket.
    current_records: Vec<(u64, u32, f32)>,
    /// Position within `current_records` (advances as the row cursor
    /// walks past records).
    record_cursor: usize,
    /// Duplicates merged in the current bucket that haven't yet been
    /// reported through a `StreamedCsrShard::duplicates_merged`
    /// field. Drained onto the next shard emitted from this bucket
    /// so the writer coordinator can route it through the sink.
    pending_duplicates: u64,
}

impl CscToCsrExternalTransposer {
    #[allow(clippy::too_many_arguments)]
    fn open(
        path: &str,
        n_obs: u64,
        n_vars: u64,
        nnz: u64,
        col_indptr: Vec<i64>,
        indices_ds: hdf5::Dataset,
        data_ds: hdf5::Dataset,
        budget: u64,
        temp_dir: Option<PathBuf>,
        sink: &mut WarningSink,
    ) -> Result<Self, ConvertError> {
        let _ = sink; // duplicates warning is emitted lazily on first
                      // bucket flush; sink is held by the writer
                      // coordinator, not threaded through here.

        // Bucket sizing. Reserve 1/4 of the budget for the in-memory
        // bucket buffer (sort scratch, dup-coalesce). Floor at 1
        // bucket-row to make progress even on tiny budgets.
        let bucket_record_cap = ((budget / 4) / RECORD_BYTES_USIZE as u64).max(1);
        let nnz_per_obs = nnz.div_ceil(n_obs.max(1)).max(1);
        let bucket_rows: u64 = (bucket_record_cap / nnz_per_obs).max(1);
        let n_buckets = n_obs.div_ceil(bucket_rows) as usize;

        // Session temp directory; auto-deleted on drop.
        let temp_root = temp_dir.unwrap_or_else(std::env::temp_dir);
        let session = tempfile::Builder::new()
            .prefix("scx-transpose-")
            .tempdir_in(&temp_root)
            .map_err(|e| {
                ConvertError::Other(format!(
                    "failed to create temp dir under {}: {e}",
                    temp_root.display()
                ))
            })?;

        // One BufWriter per bucket. Bounded by `n_buckets`; we let the
        // OS manage file descriptors. Typical n_buckets is single-digit
        // for census-scale inputs.
        let bucket_paths: Vec<PathBuf> = (0..n_buckets)
            .map(|b| session.path().join(format!("bucket_{b}.bin")))
            .collect();
        let mut writers: Vec<BufWriter<File>> = bucket_paths
            .iter()
            .map(|p| Ok::<_, ConvertError>(BufWriter::new(File::create(p)?)))
            .collect::<Result<_, _>>()?;

        // Pass 1: column-chunk scan. The chunk size is derived from
        // the half-budget allocated to "in-flight column data"
        // (indices: i32 + data: f32 = 8 bytes per nnz). Floor at one
        // column at a time.
        let col_bytes_per_nnz: u64 = 8;
        let cols_budget = (budget / 2).max(col_bytes_per_nnz * 8);
        let n_vars_usize = n_vars as usize;
        let mut col_start = 0usize;
        while col_start < n_vars_usize {
            // Pick the largest chunk whose nnz × 8 fits in
            // `cols_budget`. Walk forwards over `col_indptr`.
            let base_pos = col_indptr[col_start] as u64;
            let mut col_end = col_start + 1;
            while col_end < n_vars_usize {
                let pos = col_indptr[col_end] as u64;
                if (pos - base_pos) * col_bytes_per_nnz > cols_budget {
                    break;
                }
                col_end += 1;
            }

            let nnz_start = col_indptr[col_start] as usize;
            let nnz_end = col_indptr[col_end] as usize;
            if nnz_end > nnz_start {
                let chunk_indices = read_slice_i32(&indices_ds, nnz_start, nnz_end)?;
                let chunk_values = read_slice_f32(&data_ds, nnz_start, nnz_end)?;

                // Walk the chunk one column at a time and distribute
                // each nonzero into its bucket file.
                for col in col_start..col_end {
                    let col_pos_start = col_indptr[col] as usize;
                    let col_pos_end = col_indptr[col + 1] as usize;
                    for pos in col_pos_start..col_pos_end {
                        let local = pos - nnz_start;
                        let row = chunk_indices[local];
                        if row < 0 {
                            return Err(ConvertError::Other(format!(
                                "negative CSC row index {row}"
                            )));
                        }
                        let row_u = row as u64;
                        if row_u >= n_obs {
                            return Err(ConvertError::Other(format!(
                                "CSC row index {row} >= n_obs {n_obs}"
                            )));
                        }
                        let bucket = (row_u / bucket_rows) as usize;
                        let value = chunk_values[local];
                        write_record(&mut writers[bucket], row_u, col as u32, value)?;
                    }
                }
            }

            col_start = col_end;
        }

        for mut w in writers {
            w.flush().map_err(ConvertError::from)?;
        }

        Ok(CscToCsrExternalTransposer {
            n_obs,
            n_vars,
            source_name: path.to_string(),
            bucket_rows,
            _temp_dir: session,
            bucket_paths,
            row_cursor: 0,
            loaded_bucket: usize::MAX,
            current_records: Vec::new(),
            record_cursor: 0,
            pending_duplicates: 0,
        })
    }

    /// Load the bucket containing `self.row_cursor`. If the bucket
    /// is already loaded, this is a no-op. Sets
    /// `self.pending_duplicates` to the number of `(row, col)`
    /// duplicates merged in the bucket (drained onto the next
    /// emitted shard via `StreamedCsrShard::duplicates_merged`).
    fn load_bucket_for_cursor(&mut self) -> Result<(), ConvertError> {
        let needed = (self.row_cursor / self.bucket_rows) as usize;
        if self.loaded_bucket == needed {
            return Ok(());
        }
        let mut records = read_bucket(&self.bucket_paths[needed])?;
        // Sort by (row, col); coalesce duplicates by summing values.
        // scipy `sum_duplicates()` semantics. An empty bucket
        // produces zero records and zero duplicates.
        if !records.is_empty() {
            records.sort_unstable_by_key(|r| (r.0, r.1));
            let dup_count = coalesce_duplicates(&mut records);
            self.pending_duplicates = self.pending_duplicates.saturating_add(dup_count);
        }
        self.current_records = records;
        self.record_cursor = 0;
        self.loaded_bucket = needed;
        Ok(())
    }

    /// Boundary (exclusive) of the currently loaded bucket. Records
    /// in `current_records` all satisfy `row < bucket_end_exclusive`.
    fn bucket_end_exclusive(&self) -> u64 {
        let end = (self.loaded_bucket as u64 + 1).saturating_mul(self.bucket_rows);
        end.min(self.n_obs)
    }
}

impl CsrShardStream for CscToCsrExternalTransposer {
    fn n_obs(&self) -> u64 {
        self.n_obs
    }
    fn n_vars(&self) -> u64 {
        self.n_vars
    }
    fn source_matrix_name(&self) -> &str {
        &self.source_name
    }

    fn next_csr_shard(
        &mut self,
        target_rows: usize,
    ) -> Result<Option<StreamedCsrShard>, ConvertError> {
        if target_rows == 0 {
            return Err(ConvertError::Other("target_rows must be > 0".into()));
        }
        if self.row_cursor >= self.n_obs {
            return Ok(None);
        }
        // Ensure the bucket containing `row_cursor` is loaded.
        self.load_bucket_for_cursor()?;

        let row_start = self.row_cursor;
        // Shard ends at the smaller of: target_rows past the cursor,
        // the current bucket's end, or `n_obs`. Capping at the bucket
        // end guarantees the loop below doesn't need to span a bucket
        // boundary in a single shard — the next call will load the
        // adjacent bucket.
        let row_end = (row_start + target_rows as u64)
            .min(self.bucket_end_exclusive())
            .min(self.n_obs);
        let n_rows = (row_end - row_start) as u32;

        // Walk records starting at `record_cursor`, emitting all
        // entries whose row falls in `[row_start, row_end)`.
        let mut indptr: Vec<u64> = Vec::with_capacity(n_rows as usize + 1);
        indptr.push(0);
        let mut indices: Vec<u32> = Vec::new();
        let mut values: Vec<f32> = Vec::new();
        let mut next_expected_row = row_start;

        while self.record_cursor < self.current_records.len() {
            let (r, c, v) = self.current_records[self.record_cursor];
            if r >= row_end {
                break;
            }
            // Push empty-row markers up to the current record's row.
            while next_expected_row < r {
                indptr.push(values.len() as u64);
                next_expected_row += 1;
            }
            indices.push(c);
            values.push(v);
            self.record_cursor += 1;
        }
        // Pad trailing empty rows up to `row_end`.
        while next_expected_row < row_end {
            indptr.push(values.len() as u64);
            next_expected_row += 1;
        }
        debug_assert_eq!(indptr.len(), n_rows as usize + 1);

        // Drain any pending duplicate-merge counts onto this shard.
        let duplicates_merged = std::mem::take(&mut self.pending_duplicates);

        self.row_cursor = row_end;
        Ok(Some(StreamedCsrShard {
            row_start,
            n_rows,
            n_cols: self.n_vars as u32,
            indptr,
            indices,
            values,
            source_name: Some(self.source_name.clone()),
            duplicates_merged,
        }))
    }
}

// ---------------------------------------------------------------------
// Temp-record helpers (16-byte LE: u64 row + u32 col + f32 value).
// Kept inline — the format is local to this module.
// ---------------------------------------------------------------------

fn write_record(
    w: &mut BufWriter<File>,
    row: u64,
    col: u32,
    value: f32,
) -> Result<(), ConvertError> {
    w.write_u64::<LittleEndian>(row)?;
    w.write_u32::<LittleEndian>(col)?;
    w.write_f32::<LittleEndian>(value)?;
    Ok(())
}

fn read_bucket(path: &PathBuf) -> Result<Vec<(u64, u32, f32)>, ConvertError> {
    let mut reader = BufReader::new(File::open(path)?);
    let metadata = std::fs::metadata(path)?;
    let len = metadata.len() as usize;
    if !len.is_multiple_of(RECORD_BYTES_USIZE) {
        return Err(ConvertError::Other(format!(
            "bucket temp file {} has size {} not aligned to {}-byte records",
            path.display(),
            len,
            RECORD_BYTES_USIZE
        )));
    }
    let count = len / RECORD_BYTES_USIZE;
    let mut out = Vec::with_capacity(count);
    let mut buf = [0u8; RECORD_BYTES_USIZE];
    for _ in 0..count {
        reader.read_exact(&mut buf)?;
        let row = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let col = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let value = f32::from_le_bytes(buf[12..16].try_into().unwrap());
        out.push((row, col, value));
    }
    Ok(out)
}

/// Sum entries that share the same `(row, col)` coordinate. The
/// input must be sorted by `(row, col)`. Returns the number of
/// duplicate pairs that were merged (i.e. `dup_count == 0` means no
/// duplicates).
///
/// Resulting `0.0` values stay in the vector — `drop_explicit_zeros_inplace`
/// (applied downstream by `streaming_writer_coordinator`) removes them.
fn coalesce_duplicates(records: &mut Vec<(u64, u32, f32)>) -> u64 {
    if records.is_empty() {
        return 0;
    }
    let mut dup_count: u64 = 0;
    let mut write = 0usize;
    for read in 1..records.len() {
        let (r, c, v) = records[read];
        let (pr, pc, pv) = records[write];
        if r == pr && c == pc {
            // Merge into the previous slot.
            records[write] = (pr, pc, pv + v);
            dup_count += 1;
        } else {
            write += 1;
            records[write] = (r, c, v);
        }
    }
    records.truncate(write + 1);
    dup_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesce_no_duplicates() {
        let mut v = vec![(0u64, 0u32, 1.0f32), (0, 1, 2.0), (1, 0, 3.0)];
        let n = coalesce_duplicates(&mut v);
        assert_eq!(n, 0);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn coalesce_sums_duplicates() {
        let mut v = vec![(0u64, 0u32, 1.0f32), (0, 0, 2.5), (0, 0, 0.5), (1, 0, 3.0)];
        let n = coalesce_duplicates(&mut v);
        assert_eq!(n, 2);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], (0, 0, 4.0));
        assert_eq!(v[1], (1, 0, 3.0));
    }
}
