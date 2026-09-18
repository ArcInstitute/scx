//! The cell-set batch gather must allocate its output **once**.
//!
//! Before W11 the gather walked the plan one set at a time: a
//! `Vec<Option<(Vec<i32>, Vec<f32>)>>` per set, two owned `Vec`s per row inside
//! `transform_row`, and an `extend_from_slice` copy of every row into the batch
//! — three allocations and two copies per row, on top of an `indices`/`data`
//! pair pre-sized from an estimate. The batch executor reads the plan's unique
//! rows through `BackedCsrReader::read_row_indices_with_admission`, which
//! prescans each touched shard's indptr and carves the output exactly, and on a
//! single-file plan with no repeats hands that allocation straight out as the
//! batch.
//!
//! # Why the allocator is the oracle
//!
//! Every output assertion in `sparse_cellset_tests.rs` is green against the old
//! code — the values were always right, the allocations were the cost. RSS is
//! too noisy for a bound this tight, so this tracks **live heap bytes** and the
//! **allocation count** through a recording `#[global_allocator]`, exactly as
//! `scx-format-io/tests/gather_allocation.rs` does for the reader below it.
//!
//! A `#[global_allocator]` is per binary, so this file is one `#[test]`, and the
//! `SCX_CELLSET_EXECUTOR` switch is read once per process — so the pre-W11 arm
//! is a second **run** of this same binary, not a second test:
//!
//! ```text
//! cargo test -p scx-loader --test gather_allocation -- --nocapture
//! SCX_CELLSET_EXECUTOR=set \
//!   cargo test -p scx-loader --test gather_allocation -- --nocapture
//! ```
//!
//! Both print their measurement; only the default arm asserts the budget, since
//! the per-set walk cannot meet it and failing it would only re-state that.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;
use scx_loader::sparse_cellset::{SparseCellSetLoader, SparseCellSetPlan};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Live-bytes / allocation-count recording allocator
// ---------------------------------------------------------------------------

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static ARMED: AtomicBool = AtomicBool::new(false);

struct LiveAlloc;

impl LiveAlloc {
    #[inline]
    fn grow(by: usize) {
        let now = LIVE.fetch_add(by, Ordering::Relaxed) + by;
        if ARMED.load(Ordering::Relaxed) {
            PEAK.fetch_max(now, Ordering::Relaxed);
            ALLOCS.fetch_add(1, Ordering::Relaxed);
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

/// Run `f` and return `(result, peak_live_bytes, allocations)`, both above the
/// entry baseline.
fn measure<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    let baseline = LIVE.load(Ordering::SeqCst);
    PEAK.store(baseline, Ordering::SeqCst);
    ALLOCS.store(0, Ordering::SeqCst);
    ARMED.store(true, Ordering::SeqCst);
    let out = f();
    ARMED.store(false, Ordering::SeqCst);
    (
        out,
        PEAK.load(Ordering::SeqCst).saturating_sub(baseline),
        ALLOCS.load(Ordering::SeqCst),
    )
}

// ---------------------------------------------------------------------------
// Fixture: 4 shards × 256 rows × 512 cols, 32 nnz per row
// ---------------------------------------------------------------------------

const N_SHARDS: usize = 4;
const ROWS_PER_SHARD: usize = 256;
const N_OBS: usize = N_SHARDS * ROWS_PER_SHARD;
const N_VARS: usize = 512;
const NNZ_PER_ROW: usize = 32;

/// The plan shape: `ML_BATCH_SIZE`-equivalent, 16 sets of 64, every row distinct
/// so the raw-local fast path is the one under test.
const SETS: usize = 16;
const SET_SIZE: usize = 64;
const PLAN_ROWS: usize = SETS * SET_SIZE;

fn write_fixture(path: &Path) {
    let header = FileHeader::new_single_modality(
        N_OBS as u64,
        N_VARS as u64,
        (N_OBS * NNZ_PER_ROW) as u64,
        ROWS_PER_SHARD as u32,
        0,
        0,
    );
    let mut writer = ScxWriter::new(path, header).unwrap();
    let cells: Vec<String> = (0..N_OBS).map(|i| format!("cell_{i}")).collect();
    writer
        .write_obs(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "cell_id",
                    DataType::Utf8,
                    false,
                )])),
                vec![Arc::new(StringArray::from(
                    cells.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                ))],
            )
            .unwrap(),
        )
        .unwrap();
    let genes: Vec<String> = (0..N_VARS).map(|i| format!("gene_{i}")).collect();
    writer
        .write_var(
            &RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "gene_id",
                    DataType::Utf8,
                    false,
                )])),
                vec![Arc::new(StringArray::from(
                    genes.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                ))],
            )
            .unwrap(),
        )
        .unwrap();
    for s in 0..N_SHARDS {
        let row_start = s * ROWS_PER_SHARD;
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for local in 0..ROWS_PER_SHARD {
            let row = row_start + local;
            // Ascending and unique within the row, as CSR requires, and
            // row-dependent so a row served from the wrong span is visible.
            let stride = N_VARS / NNZ_PER_ROW;
            for k in 0..NNZ_PER_ROW {
                indices.push((k * stride + row % stride) as u32);
                values.push(((row + k + 1) & 0xFF) as u8);
            }
            indptr.push(*indptr.last().unwrap() + NNZ_PER_ROW as u64);
        }
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_start as u64,
            )
            .unwrap();
    }
    writer.finish().unwrap();
}

#[test]
fn the_cell_set_gather_allocates_its_output_once() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("alloc.scx");
    write_fixture(&path);

    let loader = SparseCellSetLoader::new(
        vec![scx_format_io::ScxReader::open(&path).unwrap()],
        /*cache_shards*/ N_SHARDS,
        None,
        /*lookahead*/ 4,
        /*remap*/ None,
        /*n_global_genes*/ None,
        false,
        false,
        0.0,
        /*downsample*/ None,
        /*scatter_block_index*/ false,
        /*max_plan_rows*/ None,
    )
    .unwrap();

    // Distinct rows, spread over every shard, in an order no shard walk would
    // produce. `PLAN_ROWS` of `N_OBS`, so no row repeats.
    let rows: Vec<u64> = (0..PLAN_ROWS).map(|i| ((i * 37) % N_OBS) as u64).collect();
    let mut seen = rows.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(
        seen.len(),
        PLAN_ROWS,
        "premise: the plan names no row twice"
    );
    let set_offsets: Vec<i64> = (0..=SETS).map(|s| (s * SET_SIZE) as i64).collect();
    let plan = SparseCellSetPlan {
        file_ids: vec![0; PLAN_ROWS],
        rows,
        role_tags: vec![0; PLAN_ROWS],
        set_offsets,
    };

    // Unmeasured: pays the rayon worker arenas, the lazy catalog tables and the
    // shard decodes, so what the measured pass sees is the assembly alone.
    let warm = loader.gather(&plan).unwrap();
    let nnz = warm.indices.len();
    drop(warm);

    let (batch, peak, allocs) = measure(|| loader.gather(&plan).unwrap());

    assert_eq!(batch.indices.len(), nnz, "same gather, twice");
    assert_eq!(batch.shape.0, PLAN_ROWS);

    // `indices` + `data` + `indptr`, plus the three small plan-order vectors.
    let result_bytes = nnz * 4 + nnz * 4 + (PLAN_ROWS + 1) * 8;
    let arm = std::env::var("SCX_CELLSET_EXECUTOR").unwrap_or_else(|_| "plan".into());
    eprintln!(
        "[{arm}] {PLAN_ROWS} rows / {nnz} nnz: peak live {peak} B ({:.2}x the result's \
         {result_bytes} B), {allocs} allocations ({:.2} per row)",
        peak as f64 / result_bytes as f64,
        allocs as f64 / PLAN_ROWS as f64,
    );

    if arm.eq_ignore_ascii_case("set") {
        // The pre-W11 arm is here to be read, not to be bounded.
        return;
    }

    // The claim is **result + O(rows)**, never a second O(nnz). The batch is
    // handed out whole, so everything above the result is per-row bookkeeping:
    // the plan's dedup set and per-shard buckets (`bucket_plan_rows`), the
    // occurrence table (`slots` 16 B, `slot_of_pos` 4 B, `slot_loc` 8 B, its
    // index map ~28 B at load factor), the per-file row list (8 B) and the
    // reader's `(row, orig_pos)` sort (16 B). 192 B/row covers those with
    // headroom and is an order of magnitude below the ~264 B/row a second copy
    // of this fixture's rows would cost — so an executor that assembled the
    // batch twice cannot pass by being frugal elsewhere.
    let bookkeeping = PLAN_ROWS * 192;
    assert!(
        peak < result_bytes + bookkeeping,
        "peak live {peak} B exceeds the {result_bytes} B result plus {bookkeeping} B \
         of per-row bookkeeping — the gather is assembling the batch more than once",
    );

    // The count is the claim that survives a change of fixture: the per-set walk
    // allocated three `Vec`s per row plus one per set, so it scales with the
    // plan. The executor's allocations are a fixed handful of buffers per file.
    assert!(
        allocs < PLAN_ROWS / 4,
        "{allocs} allocations for {PLAN_ROWS} rows — that is per-row growth, not \
         a fixed set of buffers"
    );
}
