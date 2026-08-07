//! A malformed framed shard must not drive an unbounded pre-allocation.
//!
//! `docs/format.md` requires that "the reader must not crash, panic, or exhibit
//! UB on any input", and the catalog's BLAKE3 authenticates *catalog* bytes, not
//! shard payloads — so `ShardHeader.nnz` / `.n_major` and every `nnz_in_block`
//! reaching the decoder are attacker-controlled. `Vec::with_capacity` calls
//! `handle_alloc_error` on failure, which **aborts** the process rather than
//! unwinding, so an eager reservation off those fields is a remote kill switch.
//!
//! # Why the allocator is the oracle
//!
//! The obvious test — `assert!(decode_shard_bytes(..).is_err())` — is **green
//! against the unfixed code**. Linux with `vm.overcommit_memory = 0` happily
//! satisfies a 17 GB `Vec::with_capacity` on a large-memory host (the dev node
//! here has 1007 GB); the decode then fails inside the row-group loop and
//! returns the `Err` the assertion wanted. The reservation is the defect, so the
//! reservation is what has to be measured. The recording allocator below makes
//! the pre-fix behaviour observable as a 17.18 GB / 1.07 GB request.
//!
//! Both fixtures use `codec_id = None` so the eventual failure is an unambiguous
//! sub-stream length mismatch rather than a codec-internal error.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use scx_format_io::{
    decode_shard_bytes, decode_shard_bytes_native, decode_shard_indptr_bytes, BlockIndex,
    BlockIndexEntry, FullCatalogEntry, SectionType, ShardHeader, SHARD_HEADER_SIZE, SHARD_MAGIC,
};

// ---------------------------------------------------------------------------
// Recording allocator
// ---------------------------------------------------------------------------

static PEAK_REQUEST: AtomicUsize = AtomicUsize::new(0);
static ARMED: AtomicBool = AtomicBool::new(false);

/// Passes every allocation through to the system allocator, recording the
/// largest single request made while armed. `realloc` is included because
/// `Vec` growth goes through it.
struct RecordingAlloc;

impl RecordingAlloc {
    #[inline]
    fn note(size: usize) {
        if ARMED.load(Ordering::Relaxed) {
            PEAK_REQUEST.fetch_max(size, Ordering::Relaxed);
        }
    }
}

unsafe impl GlobalAlloc for RecordingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::note(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::note(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::note(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: RecordingAlloc = RecordingAlloc;

/// Run `f` with allocation recording on, returning `(result, peak_request_bytes)`.
///
/// The whole file is a single `#[test]`, so no other test thread can pollute the
/// counter. Arming is deliberately as tight as possible around the call.
fn measure_peak_alloc<T>(f: impl FnOnce() -> T) -> (T, usize) {
    PEAK_REQUEST.store(0, Ordering::SeqCst);
    ARMED.store(true, Ordering::SeqCst);
    let out = f();
    ARMED.store(false, Ordering::SeqCst);
    (out, PEAK_REQUEST.load(Ordering::SeqCst))
}

/// Generous enough that ordinary decode bookkeeping never trips it, tiny next to
/// the 17.18 GB / 1.07 GB an unclamped reader requests.
const MAX_ALLOWED_REQUEST: usize = 16 << 20;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn entry() -> FullCatalogEntry {
    FullCatalogEntry {
        name: "X_shard_0".to_string(),
        offset: 0,
        length: 0,
        section_type: SectionType::CsrShard,
        checksum: [0u8; 32],
        modality_id: 0,
        stats: None,
    }
}

/// Assemble `header ++ indptr ++ indices ++ values ++ block_index`, matching the
/// relative offsets the header declares.
fn section(
    header: &ShardHeader,
    indptr: &[u8],
    indices: &[u8],
    values: &[u8],
    bi: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();
    header.write_to(&mut out).expect("write shard header");
    assert_eq!(out.len(), SHARD_HEADER_SIZE);
    out.extend_from_slice(indptr);
    out.extend_from_slice(indices);
    out.extend_from_slice(values);
    out.extend_from_slice(bi);
    out
}

fn framed_header(
    n_major: u32,
    n_minor: u32,
    nnz: u64,
    indptr_len: u32,
    indices_len: u32,
    values_len: u32,
    bi_len: u32,
) -> ShardHeader {
    let ip = SHARD_HEADER_SIZE as u32;
    ShardHeader {
        magic: SHARD_MAGIC,
        shard_format_version: 2, // framed
        shard_type: 0,           // CSR
        codec_id: 0,             // None
        value_encoding: 2,       // u32
        index_dtype: 0,          // u16
        reserved_flags: [0u8; 3],
        n_major,
        n_minor,
        nnz,
        global_offset: 0,
        indptr_rel_offset: ip,
        indptr_length: indptr_len,
        indices_rel_offset: ip + indptr_len,
        indices_length: indices_len,
        values_rel_offset: ip + indptr_len + indices_len,
        values_length: values_len,
        block_index_rel_offset: ip + indptr_len + indices_len + values_len,
        block_index_length: bi_len,
        checksum: [0u8; 8],
    }
}

fn block_index_bytes(entries: Vec<BlockIndexEntry>) -> Vec<u8> {
    let mut out = Vec::new();
    BlockIndex { entries }
        .write_to(&mut out)
        .expect("write block index");
    out
}

/// **Fixture A — the `nnz` side, i.e. the reported repro.** A 105-byte section
/// whose single block declares `nnz_in_block = u32::MAX` against 1-byte
/// indices/values frames. Every structural check in `resolve_block_index`
/// passes: coverage is exact, offsets are monotonic and start at zero,
/// `Σ nnz_in_block == header.nnz`, and the indices/values ranges are non-empty
/// as required when `nnz > 0`.
///
/// `n_minor` is deliberately `u32::MAX` so the group-capacity check
/// (`nnz_in_block <= n_rows * n_minor`) cannot fire — this fixture must exercise
/// the reservation clamp specifically, not the geometry guard.
///
/// Unclamped, the two `Vec::with_capacity(4_294_967_295)` calls request
/// 17.18 GB each.
fn fixture_nnz() -> Vec<u8> {
    let nnz = u32::MAX;
    let bi = block_index_bytes(vec![
        BlockIndexEntry::new(0, 1, 0, 0, 0, nnz as u64).expect("block index entry")
    ]);
    let header = framed_header(1, u32::MAX, nnz as u64, 1, 1, 1, bi.len() as u32);
    section(&header, &[0u8], &[0u8], &[0u8], &bi)
}

/// **Fixture B — the `n_major` side.** 2049 groups of 65535 rows each declare
/// `n_major = 134_281_215`, so the reassembled `Vec<i64>` indptr is 1.07 GB
/// before a single frame is decoded. Reaching `u32::MAX` rows needs only a
/// ~1.4 MB block index; 2049 groups keeps the fixture at ~45 KB while still
/// separating pre-fix from post-fix by three orders of magnitude.
///
/// Every group carries `nnz_in_block = 0`, which keeps the geometry check
/// trivially satisfied and lets the indices/values streams be empty (the
/// "non-empty range" rule only applies when `nnz > 0`). The indptr offsets step
/// by one byte per group so no group gets an empty indptr range.
fn fixture_n_major() -> (Vec<u8>, usize) {
    const GROUPS: u32 = 2049;
    const ROWS_PER_GROUP: u32 = u16::MAX as u32;
    let n_major = GROUPS * ROWS_PER_GROUP;

    let entries: Vec<BlockIndexEntry> = (0..GROUPS)
        .map(|g| {
            BlockIndexEntry::new(g * ROWS_PER_GROUP, ROWS_PER_GROUP, g, 0, 0, 0)
                .expect("block index entry")
        })
        .collect();
    let bi = block_index_bytes(entries);
    let indptr = vec![0u8; GROUPS as usize];
    let header = framed_header(n_major, 30_000, 0, GROUPS, 0, 0, bi.len() as u32);
    (
        section(&header, &indptr, &[], &[], &bi),
        n_major as usize + 1,
    )
}

// ---------------------------------------------------------------------------

/// Every framed decode entry point must reject these sections while keeping its
/// largest single allocation request under [`MAX_ALLOWED_REQUEST`].
///
/// One `#[test]` per file on purpose: `PEAK_REQUEST` is process-global, so a
/// second concurrently-running test would pollute the measurement.
#[test]
fn framed_decode_bounds_allocation_on_hostile_headers() {
    let e = entry();

    // --- Fixture A: nnz -------------------------------------------------
    let a = fixture_nnz();
    assert_eq!(a.len(), 105, "the reported repro is a ~100-byte file");

    let (scipy, peak) = measure_peak_alloc(|| decode_shard_bytes(&a, &e, 2, false));
    let err = scipy.expect_err("a shard declaring nnz = u32::MAX from 1 byte must not decode");
    assert!(
        peak <= MAX_ALLOWED_REQUEST,
        "decode_shard_bytes reserved {peak} bytes for a 105-byte section \
         (limit {MAX_ALLOWED_REQUEST}); error was {err:?}"
    );

    // `ShardValuesNative` is not `Debug`, so drop the Ok payload before unwrapping.
    let (native, peak) =
        measure_peak_alloc(|| decode_shard_bytes_native(&a, &e, 2, false).map(|_| ()));
    let err = native.expect_err("native twin must reject the same section");
    assert!(
        peak <= MAX_ALLOWED_REQUEST,
        "decode_shard_bytes_native reserved {peak} bytes for a 105-byte section \
         (limit {MAX_ALLOWED_REQUEST}); error was {err:?}"
    );

    // --- Fixture B: n_major ---------------------------------------------
    let (b, declared_indptr_len) = fixture_n_major();
    assert!(
        declared_indptr_len * 8 > 1 << 30,
        "fixture must declare an indptr over 1 GB to be worth measuring \
         (got {} elements)",
        declared_indptr_len
    );

    let (ip_only, peak) = measure_peak_alloc(|| decode_shard_indptr_bytes(&b, &e, 2));
    let err = ip_only.expect_err("134M rows backed by 2048 bytes of indptr must not decode");
    assert!(
        peak <= MAX_ALLOWED_REQUEST,
        "decode_shard_indptr_bytes reserved {peak} bytes (limit {MAX_ALLOWED_REQUEST}); \
         error was {err:?}"
    );

    let (full, peak) = measure_peak_alloc(|| decode_shard_bytes(&b, &e, 2, false));
    let err = full.expect_err("the full decode must reject it too");
    assert!(
        peak <= MAX_ALLOWED_REQUEST,
        "decode_shard_bytes reserved {peak} bytes on the n_major fixture \
         (limit {MAX_ALLOWED_REQUEST}); error was {err:?}"
    );
}
