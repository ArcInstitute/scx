//! The backed row gather must assemble its output **once**, into pre-sized
//! buffers, with a live-memory peak bounded by the result plus the shard cache
//! plus the shards a warm or bulk decode holds in flight — at most
//! `2 × cache_shards` decoded shards beside the result on a full cache, never a
//! second copy of the result.
//!
//! `BackedCsrReader::read_row_indices` used to build one `ScxCsr` (three `Vec`s)
//! per requested row and then `concatenate_csr` them — a second full copy of the
//! result held while the first was still alive — and `read_rows` did the same
//! per shard window. Both peaked at ≈ 2× the result on top of the decoded-shard
//! LRU. The fix (PR C / REC-1) prescans each touched shard's indptr, allocates
//! the exact output, and copies every row straight into its window.
//!
//! # Why the allocator is the oracle
//!
//! A `to_memory()`-style parity assertion is green against the old code — the
//! values were always right, the transient was the defect. RSS is too noisy a
//! measure for a bound this tight (the allocator retains freed pages), so this
//! test tracks **live heap bytes** through a recording `#[global_allocator]` and
//! asserts the high-water mark of the gather against a budget derived from the
//! result and the shard cache. Structure follows
//! `framed_decode_allocation.rs`; unlike that file it tracks live bytes (alloc
//! minus dealloc) rather than the largest single request, because the defect
//! here is two result-sized buffers alive at once, not one oversized request.
//!
//! A `#[global_allocator]` is per binary, so this file is one `#[test]` — rayon's
//! worker arenas are paid up front by an unmeasured warm-up read.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::{
    encode_one_shard, BackedCsrReader, EncodeShardOptions, FileHeader, FramingConfig, ScxReader,
    ScxWriter, SectionType, CURRENT_FORMAT_VERSION,
};
use scx_sparse::ScxCsr;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Live-bytes recording allocator
// ---------------------------------------------------------------------------

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ARMED: AtomicBool = AtomicBool::new(false);

struct LiveAlloc;

impl LiveAlloc {
    #[inline]
    fn grow(by: usize) {
        let now = LIVE.fetch_add(by, Ordering::Relaxed) + by;
        if ARMED.load(Ordering::Relaxed) {
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
    }
    #[inline]
    fn shrink(by: usize) {
        LIVE.fetch_sub(by, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for LiveAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            Self::grow(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            Self::grow(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        Self::shrink(layout.size());
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            if new_size >= layout.size() {
                Self::grow(new_size - layout.size());
            } else {
                Self::shrink(layout.size() - new_size);
            }
        }
        p
    }
}

#[global_allocator]
static ALLOC: LiveAlloc = LiveAlloc;

/// Run `f` and return `(result, peak_live_bytes_above_the_entry_baseline)`.
fn measure_live_peak<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let baseline = LIVE.load(Ordering::SeqCst);
    PEAK.store(baseline, Ordering::SeqCst);
    ARMED.store(true, Ordering::SeqCst);
    let out = f();
    ARMED.store(false, Ordering::SeqCst);
    let peak = PEAK.load(Ordering::SeqCst);
    (out, peak.saturating_sub(baseline))
}

// ---------------------------------------------------------------------------
// Fixture: 40 shards × 256 rows × 512 cols, 64 nnz per row
// ---------------------------------------------------------------------------

const N_SHARDS: usize = 40;
const ROWS_PER_SHARD: usize = 256;
const N_VARS: usize = 512;
const NNZ_PER_ROW: usize = 64;
const ROW_GROUP_ROWS: u32 = 8;
const CACHE_SHARDS: usize = 4;
/// Decode-pool width pinned on every measured reader.
///
/// Since the block-index gather decodes its row groups a **chunk** at a time,
/// the chunk is the pool's width and the transient it holds is
/// `GROUP_DECODE_WIDTH × group bytes`. Left to rayon's global registry that
/// width is the machine's core count, so the budget below would be a different
/// number on a laptop and on a 184-core node — the bound would still hold and
/// the test would still pass, but it would have stopped measuring anything.
const GROUP_DECODE_WIDTH: usize = 4;

fn shard_data(shard: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr = vec![0u64];
    let mut indices = Vec::with_capacity(ROWS_PER_SHARD * NNZ_PER_ROW);
    let mut values = Vec::with_capacity(ROWS_PER_SHARD * NNZ_PER_ROW);
    for r in 0..ROWS_PER_SHARD {
        let row = shard * ROWS_PER_SHARD + r;
        for k in 0..NNZ_PER_ROW {
            // Strictly increasing within the row, shifted per row so no two
            // rows are identical.
            indices.push(((row % 7) + k * (N_VARS / NNZ_PER_ROW)) as u32);
            values.push(((row + k) % 255 + 1) as u8);
        }
        indptr.push(indptr.last().unwrap() + NNZ_PER_ROW as u64);
    }
    (indptr, indices, values)
}

fn obs_batch(n: usize) -> arrow::array::RecordBatch {
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
    arrow::array::RecordBatch::try_new(
        std::sync::Arc::new(schema),
        vec![std::sync::Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn var_batch(n: usize) -> arrow::array::RecordBatch {
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    arrow::array::RecordBatch::try_new(
        std::sync::Arc::new(schema),
        vec![std::sync::Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

/// Write the fixture; `framed` selects row-group framing (the layout pyscx
/// writes by default, which the block-index gather path serves) or the plain
/// unframed layout (full-shard decode only).
fn write_fixture(dir: &TempDir, framed: bool) -> std::path::PathBuf {
    let n_obs = N_SHARDS * ROWS_PER_SHARD;
    let nnz = (n_obs * NNZ_PER_ROW) as u64;
    let path = dir
        .path()
        .join(if framed { "framed.scx" } else { "plain.scx" });
    let mut header = FileHeader::new_single_modality(
        n_obs as u64,
        N_VARS as u64,
        nnz,
        ROWS_PER_SHARD as u32,
        0,
        0,
    );
    if framed {
        header.format_version = CURRENT_FORMAT_VERSION;
    }
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&obs_batch(n_obs)).unwrap();
    writer.write_var(&var_batch(N_VARS)).unwrap();
    for s in 0..N_SHARDS {
        let (indptr, indices, values) = shard_data(s);
        let row_start = (s * ROWS_PER_SHARD) as u64;
        if framed {
            let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
            let mut opts = EncodeShardOptions::new(
                format!("X_shard_{s}"),
                SectionType::CsrShard,
                N_VARS as u64,
                row_start,
                0,
            );
            opts.explicit_codec = Some(CodecId::None);
            opts.framing = Some(FramingConfig {
                row_group_rows: ROW_GROUP_ROWS,
                target_nnz: None,
                trial: false,
                decode_target: None,
            });
            let pre = encode_one_shard(&indptr, &indices, &values_f32, &opts).unwrap();
            writer.write_preencoded_shard(pre).unwrap();
        } else {
            writer
                .write_csr_shard(
                    &indptr,
                    &indices,
                    &values,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    row_start,
                )
                .unwrap();
        }
    }
    writer.finish().unwrap();
    path
}

fn open(path: &std::path::Path) -> BackedCsrReader {
    // `mut` only under `parallel` — without it nothing reconfigures the reader
    // and clippy's `unused_mut` is denied in the feature-matrix legs.
    #[cfg(feature = "parallel")]
    let mut backed = BackedCsrReader::new(ScxReader::open(path).unwrap(), CACHE_SHARDS);
    #[cfg(not(feature = "parallel"))]
    let backed = BackedCsrReader::new(ScxReader::open(path).unwrap(), CACHE_SHARDS);
    // Only the pool pin is feature-gated, not the test: without `parallel`,
    // `group_decode_chunk()` is already 1, so the peak bound below is *tighter*
    // and every other assertion still means something. `set_cpu_pool` and
    // `rayon` do not exist in that build.
    #[cfg(feature = "parallel")]
    backed.set_cpu_pool(std::sync::Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(GROUP_DECODE_WIDTH)
            .build()
            .unwrap(),
    ));
    backed
}

/// Decoded byte size of a CSR as the shard cache accounts for it.
fn csr_bytes(rows: usize, nnz: usize) -> usize {
    (rows + 1) * 8 + nnz * 8
}

fn shard_bytes() -> usize {
    csr_bytes(ROWS_PER_SHARD, ROWS_PER_SHARD * NNZ_PER_ROW)
}

fn assert_rows_match(out: &ScxCsr, full: &ScxCsr, rows: &[u64]) {
    assert_eq!(out.n_rows(), rows.len());
    for (i, &row) in rows.iter().enumerate() {
        let (lo, hi) = (
            full.indptr[row as usize] as usize,
            full.indptr[row as usize + 1] as usize,
        );
        let (olo, ohi) = (out.indptr[i] as usize, out.indptr[i + 1] as usize);
        assert_eq!(
            &out.indices[olo..ohi],
            &full.indices[lo..hi],
            "row {row} indices"
        );
        assert_eq!(&out.data[olo..ohi], &full.data[lo..hi], "row {row} data");
    }
}

#[test]
fn gather_paths_assemble_once_within_the_cache_bound() {
    let dir = TempDir::new().unwrap();
    let n_obs = N_SHARDS * ROWS_PER_SHARD;
    let eighth: Vec<u64> = (0..n_obs as u64).step_by(8).collect();
    let shard = shard_bytes();

    for framed in [false, true] {
        let path = write_fixture(&dir, framed);
        let full = ScxReader::open(&path)
            .unwrap()
            .read_all_csr_shards()
            .unwrap();

        // Pay rayon's worker-thread arenas and every lazily-initialised
        // reader-side table before arming. Throwaway reader: the measured
        // readers below start with an empty LRU.
        {
            let warm = open(&path);
            warm.read_rows(0, n_obs as u64).unwrap();
            warm.read_row_indices(&eighth[..64]).unwrap();
        }

        // --- read_row_indices: an interleaved eighth touches every shard ---
        let backed = open(&path);
        let (out, peak) = measure_live_peak(|| backed.read_row_indices(&eighth).unwrap());
        assert_rows_match(&out, &full, &eighth);
        let result = csr_bytes(out.n_rows(), out.nnz());
        // Exact output + the LRU (warm fills up to `cache_shards`; the
        // sequential tail decodes one shard into the cache at a time, so at
        // most one extra is alive during an insert) + the block-index
        // transient on a framed file + 10 % for bookkeeping.
        //
        // That transient is `GROUP_DECODE_WIDTH` row groups, not the gather's
        // whole set of them: the chunked decode holds one chunk and drops it
        // before decoding the next, which is what keeps an over-budget gather —
        // the one with the most groups to overlap — from holding all of them.
        // Generously over-counted as a quarter shard here (a group is
        // `ROW_GROUP_ROWS / ROWS_PER_SHARD` = 1/32 of a shard, so four of them
        // are an eighth), plus one small triple per single-row run.
        let budget = result
            + (CACHE_SHARDS + 1) * shard
            + shard
            + shard / 4
            + GROUP_DECODE_WIDTH * (shard / 32)
            + eighth.len() * 3 * 32;
        let budget = budget + budget / 10;
        eprintln!(
            "framed={framed} read_row_indices(eighth): result={result} peak={peak} budget={budget} ({:.2}× result)",
            peak as f64 / result as f64
        );
        assert!(
            peak <= budget,
            "framed={framed}: read_row_indices peaked at {peak} live bytes for a {result}-byte \
             result (budget {budget}) — the gather is not assembling once"
        );
        drop(out);
        drop(backed);

        // --- read_row_indices with a FULL LRU going in ---
        // `scatter_groups` warms up to `cache_shards` cold full-path shards in
        // parallel; a full LRU is evicted only as each new shard lands, so up
        // to `2 × cache_shards` decoded shards can sit beside the result. Armed
        // before the warming read so the resident shards count.
        let backed = open(&path);
        let (out, peak) = measure_live_peak(|| {
            let warm = backed
                .read_rows(0, (CACHE_SHARDS * ROWS_PER_SHARD) as u64)
                .unwrap();
            drop(warm);
            backed.read_row_indices(&eighth).unwrap()
        });
        assert_rows_match(&out, &full, &eighth);
        let result = csr_bytes(out.n_rows(), out.nnz());
        let budget = result + 2 * CACHE_SHARDS * shard + shard + shard / 4 + eighth.len() * 3 * 32;
        let budget = budget + budget / 10;
        eprintln!(
            "framed={framed} read_row_indices(eighth) warm LRU: result={result} peak={peak} budget={budget} ({:.2}× result)",
            peak as f64 / result as f64
        );
        assert!(
            peak <= budget,
            "framed={framed}: read_row_indices on a full LRU peaked at {peak} live bytes for a \
             {result}-byte result (budget {budget}: result + 2 × cache_shards shards + transient)"
        );
        drop(out);
        drop(backed);

        // --- read_rows(0, n): the whole matrix through the range path ---
        let backed = open(&path);
        let (out, peak) = measure_live_peak(|| backed.read_rows(0, n_obs as u64).unwrap());
        assert_eq!(out.indptr, full.indptr);
        assert_eq!(out.indices, full.indices);
        assert_eq!(out.data, full.data);
        let result = csr_bytes(out.n_rows(), out.nnz());
        // Exact output + one chunk of `cache_shards` shards decoded uncached in
        // parallel (the LRU is empty here and stays empty — a bulk range never
        // enters it) + one shard of decode bookkeeping.
        let budget = result + CACHE_SHARDS * shard + shard;
        let budget = budget + budget / 10;
        eprintln!(
            "framed={framed} read_rows(0, n): result={result} peak={peak} budget={budget} ({:.2}× result)",
            peak as f64 / result as f64
        );
        assert!(
            peak <= budget,
            "framed={framed}: read_rows(0, n) peaked at {peak} live bytes for a {result}-byte \
             result (budget {budget}) — the range read is not assembling once"
        );
        drop(out);
        drop(backed);

        // --- read_rows(0, n) with a FULL LRU going in ---
        // The resident shards are the caller's budget and stay resident; the
        // bulk read adds at most one chunk of `cache_shards` uncached decodes
        // on top, so the documented bound is result + 2 × cache_shards shards.
        // Arm *before* the warming read so the four resident shards count
        // toward the peak — the entry baseline of a later arm would already
        // contain them and hide exactly the term this arm exists to bound.
        let backed = open(&path);
        let (out, peak) = measure_live_peak(|| {
            let warm = backed
                .read_rows(0, (CACHE_SHARDS * ROWS_PER_SHARD) as u64)
                .unwrap();
            assert_eq!(warm.n_rows(), CACHE_SHARDS * ROWS_PER_SHARD);
            drop(warm);
            backed.read_rows(0, n_obs as u64).unwrap()
        });
        assert_eq!(out.indices, full.indices);
        let result = csr_bytes(out.n_rows(), out.nnz());
        let budget = result + 2 * CACHE_SHARDS * shard + shard;
        let budget = budget + budget / 10;
        eprintln!(
            "framed={framed} read_rows(0, n) warm LRU: result={result} peak={peak} budget={budget} ({:.2}× result)",
            peak as f64 / result as f64
        );
        assert!(
            peak <= budget,
            "framed={framed}: read_rows(0, n) on a full LRU peaked at {peak} live bytes for a \
             {result}-byte result (budget {budget}: result + 2 × cache_shards shards)"
        );
    }
}
