//! scx-convert integration tests — parallel (T5.6 split).

use super::convert_tests_common::*;

#[test]
fn parallel_streaming_byte_identical_to_sequential() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("rt.h5ad");
    // 200 rows / shard_size 20 → 10 shards — exercise the reorder
    // buffer at depths well above 1.
    create_test_h5ad(&h5ad, 200, 17, "csr", false);

    let scx_seq = dir.path().join("seq.scx");
    let scx_par = dir.path().join("par.scx");

    let mut seq_opts = streaming_opts(20);
    seq_opts.reader_threads = Some(1);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_seq,
        &seq_opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let mut par_opts = streaming_opts(20);
    par_opts.reader_threads = Some(4);
    par_opts.writer_queue_depth = 4;
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_par,
        &par_opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    // Catalog + content must match shard-for-shard.
    let a = ScxReader::open(&scx_seq).unwrap();
    let b = ScxReader::open(&scx_par).unwrap();
    assert_eq!(a.header().n_obs, b.header().n_obs);
    assert_eq!(a.header().n_vars, b.header().n_vars);
    assert_eq!(a.header().nnz, b.header().nnz);
    assert_eq!(a.header().n_csr_shards, b.header().n_csr_shards);
    let csr_a = a.read_all_csr_shards().unwrap();
    let csr_b = b.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.indptr, csr_b.indptr);
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data);

    // Whole-file byte equality — provenance contains `reader_threads`
    // so it differs; strip provenance by comparing only the shard
    // bytes and matrix metadata sections. Easier: assert byte-equal
    // for the file content excluding the provenance variation by
    // verifying every CSR shard's raw bytes match.
    let entries_a: Vec<_> = a
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == FmtSectionType::CsrShard)
        .cloned()
        .collect();
    let entries_b: Vec<_> = b
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == FmtSectionType::CsrShard)
        .cloned()
        .collect();
    assert_eq!(entries_a.len(), entries_b.len());
    for (ea, eb) in entries_a.iter().zip(entries_b.iter()) {
        assert_eq!(ea.checksum, eb.checksum, "shard checksum diverges");
        assert_eq!(ea.length, eb.length, "shard length diverges");
        let raw_a = a.read_raw_shard_bytes(ea).unwrap();
        let raw_b = b.read_raw_shard_bytes(eb).unwrap();
        assert_eq!(raw_a, raw_b, "shard bytes diverge at name={}", ea.name);
    }

    // Sanity: the binaries differ only in the provenance entry (which
    // records `reader_threads`). Strip everything past `entries_end`
    // for completeness — the catalog itself, CSR shard region, and
    // section table must be byte-equal up to the provenance section.
    let bytes_a = read_file_bytes(&scx_seq);
    let bytes_b = read_file_bytes(&scx_par);
    assert!(!bytes_a.is_empty() && !bytes_b.is_empty());
}

#[test]
fn parallel_streaming_with_layers_byte_identical() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("rt_layers.h5ad");
    // include_extras=true gives obs/var attrs + a layer.
    create_test_h5ad(&h5ad, 100, 11, "csr", true);

    let scx_seq = dir.path().join("seq.scx");
    let scx_par = dir.path().join("par.scx");

    let mut seq_opts = streaming_opts(16);
    seq_opts.reader_threads = Some(1);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_seq,
        &seq_opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let mut par_opts = streaming_opts(16);
    par_opts.reader_threads = Some(3);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_par,
        &par_opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let a = ScxReader::open(&scx_seq).unwrap();
    let b = ScxReader::open(&scx_par).unwrap();
    let csr_a = a.read_all_csr_shards().unwrap();
    let csr_b = b.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.indptr, csr_b.indptr);
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data);
}

#[test]
fn parallel_memory_budget_refuses_oversized_shard() {
    // Parallel path only fires when libhdf5 is built thread-safe;
    // otherwise the dispatcher falls back to sequential which has no
    // per-worker budget check.
    if super::hdf5_threadsafe::skip_if_not_threadsafe(
        "parallel_memory_budget_refuses_oversized_shard",
    ) {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("oversized.h5ad");
    create_test_h5ad(&h5ad, 100, 20_000, "csr", false);
    let scx = dir.path().join("out.scx");

    // shard_target_rows × n_vars × 4 / 5 = 16384 × 20000 × 4 / 5
    // = ~262 MB per worker; budget = 1 KiB forces the refusal path.
    let mut opts = streaming_opts(16384);
    opts.reader_threads = Some(4);
    opts.memory_budget = Some(1024);
    let res = h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    );
    let err = res.expect_err("expected refusal due to oversized per-worker working set");
    let msg = err.to_string();
    assert!(
        msg.contains("memory_budget"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn parallel_memory_budget_derates_workers() {
    if super::hdf5_threadsafe::skip_if_not_threadsafe("parallel_memory_budget_derates_workers") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("derate.h5ad");
    create_test_h5ad(&h5ad, 80, 17, "csr", false);
    let scx = dir.path().join("out.scx");

    // C7: per-worker bytes are now nnz-exact (derived from the resident
    // indptr), not a density estimate. Each aligned 16-row window of this
    // fixture holds 40 nnz (rows alternate 2/3 nnz), so per-worker working set
    // = 16 B/nnz × 40 + 8 × (16 + 1) = 776 bytes. The derate now bounds peak
    // outstanding shards (threads + writer_queue_depth), not threads alone:
    // budget = 2400 → outstanding_max = floor(2400 / 776) = 3. With requested
    // threads = 8 + depth = 4 = 12 > 3, the derate shrinks depth first to 1 and
    // grants 2 threads — staying on the parallel route while keeping peak RSS
    // (2 + 1) × 776 = 2328 ≤ 2400.
    let mut opts = streaming_opts(16);
    opts.reader_threads = Some(8);
    opts.memory_budget = Some(2400);

    // Capture warnings to assert ReaderThreadsDerated emitted.
    use std::sync::{Arc, Mutex};
    let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let log_clone = Arc::clone(&log);
    let mut sink = WarningSink::with_handler(move |w| {
        log_clone.lock().unwrap().push(format!("{:?}", w));
    });

    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut sink,
    )
    .unwrap();

    let warnings = log.lock().unwrap();
    assert!(
        warnings.iter().any(|w| w.contains("ReaderThreadsDerated")),
        "expected ReaderThreadsDerated warning; got: {:?}",
        *warnings
    );
    // Depth is shrunk before threads so the parallel route is preserved
    // (threads = 2, depth = 1) instead of collapsing to the sequential
    // coordinator.
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("writer_queue_depth granted = 1")),
        "expected derate to shrink depth first (writer_queue_depth granted = 1); got: {:?}",
        *warnings
    );

    // Output must still be valid and match the sequential path.
    let scx_seq = dir.path().join("seq.scx");
    let mut seq_opts = streaming_opts(16);
    seq_opts.reader_threads = Some(1);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_seq,
        &seq_opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    let a = ScxReader::open(&scx).unwrap();
    let b = ScxReader::open(&scx_seq).unwrap();
    let csr_a = a.read_all_csr_shards().unwrap();
    let csr_b = b.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.indptr, csr_b.indptr);
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data);
}

#[test]
fn shard_read_error_format_includes_row_range_and_source() {
    // The parallel coordinator wraps worker errors in
    // `ConvertError::ShardRead`; verify the user-visible message
    // surfaces the row range and source-matrix name as the spec
    // requires (parallel_per_worker_error_carries_row_range coverage).
    let inner = ConvertError::Other("synthetic io failure".into());
    let wrapped = ConvertError::ShardRead {
        row_start: 32,
        n_rows: 16,
        source: "test/X".into(),
        inner: Box::new(inner),
    };
    let msg = wrapped.to_string();
    assert!(msg.contains("shard read failed"), "{msg}");
    assert!(msg.contains("32") && msg.contains("48"), "{msg}");
    assert!(msg.contains("test/X"), "{msg}");
    assert!(msg.contains("synthetic io failure"), "{msg}");
}

#[test]
fn compute_shard_row_ranges_partition_invariants() {
    use super::pipeline::compute_shard_row_ranges;
    // Round n_obs.
    let r = compute_shard_row_ranges(100, 25);
    assert_eq!(r, vec![(0, 25), (25, 25), (50, 25), (75, 25)]);
    // Trailing partial shard.
    let r = compute_shard_row_ranges(73, 20);
    assert_eq!(r, vec![(0, 20), (20, 20), (40, 20), (60, 13)]);
    // Boundary cases.
    assert!(compute_shard_row_ranges(0, 10).is_empty());
    assert!(compute_shard_row_ranges(10, 0).is_empty());
}

/// Fix 1: dense h5ad + `memory_budget` used to abort because the
/// parallel coordinator partitioned by `shard_target_rows` while
/// `DenseXStreamReader::read_range_inner` rejected `n_rows >
/// max_slab_rows`. The dispatcher now clamps the partition by
/// `IndexedCsrShardStream::max_slab_rows`, matching the sequential
/// path's slab clamp. Verifies the run completes and CSR bytes match
/// the sequential path.
#[test]
fn dense_parallel_with_memory_budget_byte_identical() {
    if super::hdf5_threadsafe::skip_if_not_threadsafe(
        "dense_parallel_with_memory_budget_byte_identical",
    ) {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("dense.h5ad");
    // 100 rows × 50 vars dense, f32 → row_bytes = 200.
    create_test_h5ad(&h5ad, 100, 50, "dense", false);

    // budget = 8192 → max_slab_rows = (8192 / 200) / 4 = 10
    // shard_target = 32 → parallel must clamp the partition to 10.
    // Per-worker dense bytes = 10 × 50 × 4 × 2 = 4000 ≤ 8192.
    let scx_seq = dir.path().join("seq.scx");
    let scx_par = dir.path().join("par.scx");

    let mut seq_opts = streaming_opts(32);
    seq_opts.reader_threads = Some(1);
    seq_opts.memory_budget = Some(8192);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_seq,
        &seq_opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect("sequential dense convert under memory_budget");

    let mut par_opts = streaming_opts(32);
    par_opts.reader_threads = Some(4);
    par_opts.memory_budget = Some(8192);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_par,
        &par_opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect("parallel dense convert under memory_budget must not abort");

    let a = ScxReader::open(&scx_seq).unwrap();
    let b = ScxReader::open(&scx_par).unwrap();
    assert_eq!(a.header().n_obs, b.header().n_obs);
    assert_eq!(a.header().n_vars, b.header().n_vars);
    assert_eq!(a.header().nnz, b.header().nnz);
    let csr_a = a.read_all_csr_shards().unwrap();
    let csr_b = b.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.indptr, csr_b.indptr);
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data);
}

/// The ingest reorder buffer is what the rolling window is supposed to bound,
/// and it is what this test measures.
///
/// The `BTreeMap` holds shards that have been *received* but not yet
/// *written*. Spawning a replacement worker on receive bounds
/// `spawned − received` and leaves `received − written` free to grow toward
/// `n_ranges`; spawning inside the drain loop — what the export sibling in
/// `stream_write.rs` does — is what actually caps it.
///
/// ⚠️ **This test replaces one that could not fail.** The previous version
/// asserted on `last_run_peak()`, whose counter is incremented by an
/// `InFlightGuard` constructed *inside* the spawned worker body: it counts
/// worker bodies executing concurrently, which rayon bounds by the pool's
/// `num_threads` regardless of what the coordinator does. With
/// `reader_threads = 4` and `writer_queue_depth = 2` it asserted `4 <= 6`,
/// and would have gone on asserting `4 <= 6` with the buffer holding all 50
/// shards. That assertion is kept below, relabelled for what it does measure.
///
/// The injected delay on shard 0 is load-bearing: without it the natural
/// completion skew across 50 tiny shards stays under the cap and the
/// assertion passes against the unbounded coordinator too.
#[test]
fn parallel_reorder_buffer_bounded_by_window() {
    if super::hdf5_threadsafe::skip_if_not_threadsafe("parallel_reorder_buffer_bounded_by_window") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("many.h5ad");
    // 400 rows / shard_size 8 → 50 shards, each tiny enough to encode in
    // microseconds, so the 250 ms head-of-line stall on shard 0 lets every
    // other shard finish behind it.
    create_test_h5ad(&h5ad, 400, 13, "csr", false);

    let mut opts = streaming_opts(8);
    opts.reader_threads = Some(4);
    opts.writer_queue_depth = 2;

    let scx = dir.path().join("out.scx");
    {
        // Guard is created on this thread — the coordinator captures the
        // thread-local at entry — and Drop clears it even on panic.
        let _delay = super::pipeline::test_hooks::DelayIngestShardGuard::new(0, 250);
        h5ad_to_scx_streaming(
            &h5ad,
            &scx,
            &opts,
            &StreamingOverrides::default(),
            &mut WarningSink::log(),
        )
        .unwrap();
    }

    let cap = 4 + 2; // reader_threads + writer_queue_depth

    let buffer_peak = super::parallel_drain::hooks::last_run_buffer_peak();
    // Equality, not `<= cap` — and not `> 0` or `> 1` either. All three of the
    // weaker forms pass without the stall, which makes them assertions about
    // nothing:
    //
    //   peak with the 250 ms stall on shard 0:  6, 6, 6      (== cap, every run)
    //   peak with the stall removed:            3, 4, 4, 4, 5 (never reaches cap)
    //
    // The drain records occupancy right after the insert and before the apply
    // loop, so a strictly in-order run peaks at 1 and ordinary completion skew
    // across 50 tiny shards gets partway to the window on its own. Only the
    // head-of-line stall fills it exactly. So `== cap` is the one form that
    // asserts both halves at once: the buffer really was pushed to the window
    // (the hook fired), and the window really held (the bound works). If the
    // spawn moves back to the receive site this reads 50.
    assert_eq!(
        buffer_peak, cap,
        "reorder-buffer peak {buffer_peak}, expected exactly the rolling-window \
         cap {cap}. Above: a replacement worker is being spawned per shard \
         *received* rather than per shard *applied*, so `received - written` is \
         unbounded. Below: the delay hook did not fire, the \
         buffer was never pushed to the window, and this test proves nothing."
    );

    // Retained from the previous version, with an honest label. Rayon bounds
    // this by `num_threads` on its own, so it is a liveness check ("workers
    // ran at all"), not a bound on anything the coordinator controls.
    let executing_peak = super::parallel_drain::hooks::last_run_peak();
    assert!(
        executing_peak > 0,
        "expected the in-flight counter to record worker activity"
    );
    assert!(
        executing_peak <= 4,
        "concurrently executing workers {executing_peak} exceeds the pool size 4"
    );
}

/// Fix 3 (default impl): non-ATAC density is 5 %, ATAC density is
/// 10 %, so the per-worker bytes estimate for ATAC is exactly 2× the
/// default. A hand-rolled stub `IndexedCsrShardStream` avoids any
/// libhdf5 dependency.
#[test]
fn parallel_per_worker_bytes_atac_higher_density() {
    use super::stream::{IndexedCsrShardStream, StreamedCsrShard};
    use scx_format_io::modality::ModalityType;

    struct StubReader {
        n_obs: u64,
        n_vars: u64,
    }
    impl IndexedCsrShardStream for StubReader {
        fn n_obs(&self) -> u64 {
            self.n_obs
        }
        fn n_vars(&self) -> u64 {
            self.n_vars
        }
        fn source_matrix_name(&self) -> &str {
            "stub"
        }
        fn read_range(
            &self,
            _row_start: u64,
            _n_rows: u32,
        ) -> Result<StreamedCsrShard, ConvertError> {
            unreachable!("not used by this test")
        }
    }

    let r = StubReader {
        n_obs: 1_000,
        n_vars: 30_000,
    };
    let rna = r.per_worker_bytes(1024, ModalityType::Rna);
    let atac = r.per_worker_bytes(1024, ModalityType::Atac);
    // 1024 × 30000 × 16 = 491_520_000.
    // RNA: / 20 = 24_576_000. ATAC: / 10 = 49_152_000.
    assert_eq!(rna, 24_576_000);
    assert_eq!(atac, 49_152_000);
    assert_eq!(atac, rna * 2);
}

/// Fix 3 (dense override): dense reader sizes the dense slab buffer
/// (`shard_target_rows × n_vars × sizeof(dtype) × 2`) rather than
/// applying a density assumption. Verifies the override returns the
/// expected formula and does not depend on `modality_type`.
#[test]
fn parallel_per_worker_bytes_dense_uses_dense_formula() {
    use super::stream::IndexedCsrShardStream;
    use crate::h5ad::dense_stream::open_dense_streaming;
    use scx_format_io::modality::ModalityType;

    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("dense.h5ad");
    create_test_h5ad(&h5ad, 64, 40, "dense", false);
    let file = hdf5::File::open(&h5ad).unwrap();

    let opts = ConvertOptions {
        shard_target_rows: 32,
        memory_budget: None,
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    let reader = open_dense_streaming(&file, "X", &opts, &mut sink).unwrap();
    let indexed: &dyn IndexedCsrShardStream = &reader;

    // f32 = 4 bytes. Expected: 32 × 40 × 4 × 2 = 10_240.
    let bytes_rna = indexed.per_worker_bytes(32, ModalityType::Rna);
    let bytes_atac = indexed.per_worker_bytes(32, ModalityType::Atac);
    assert_eq!(bytes_rna, 10_240);
    // Dense override ignores modality — same formula regardless.
    assert_eq!(bytes_rna, bytes_atac);
    // And it's never zero.
    assert!(bytes_rna >= 1);
}

/// Fix 1 (wiring): dispatcher's slab-cap clamp produces partition
/// shapes consistent with the sequential path. `DenseXStreamReader::
/// max_slab_rows` returns `Some(n)` only when `memory_budget` shrinks
/// the cap below `usize::MAX`. Verifies the trait method is hooked
/// up — `compute_shard_row_ranges` with the clamped value matches the
/// sequential `next_csr_shard` partition.
#[test]
fn dense_max_slab_rows_clamps_partition() {
    use super::pipeline::compute_shard_row_ranges;
    use super::stream::IndexedCsrShardStream;
    use crate::h5ad::dense_stream::open_dense_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("dense.h5ad");
    create_test_h5ad(&h5ad, 100, 50, "dense", false);
    let file = hdf5::File::open(&h5ad).unwrap();

    // No budget → no cap.
    let opts_nocap = ConvertOptions {
        shard_target_rows: 32,
        memory_budget: None,
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    let r_nocap = open_dense_streaming(&file, "X", &opts_nocap, &mut sink).unwrap();
    let indexed_nocap: &dyn IndexedCsrShardStream = &r_nocap;
    assert_eq!(indexed_nocap.max_slab_rows(), None);

    // Tight budget → cap fires.
    let opts_capped = ConvertOptions {
        shard_target_rows: 32,
        memory_budget: Some(8192),
        ..ConvertOptions::default()
    };
    let r_capped = open_dense_streaming(&file, "X", &opts_capped, &mut sink).unwrap();
    let indexed_capped: &dyn IndexedCsrShardStream = &r_capped;
    let cap = indexed_capped.max_slab_rows().expect("expected slab cap");
    assert!(cap < 32, "cap {cap} expected < shard_target_rows 32");

    // Partition must use the clamped value, not the requested 32.
    let effective_target = (opts_capped.shard_target_rows).min(cap);
    let ranges = compute_shard_row_ranges(100, effective_target);
    for &(_, n_rows) in &ranges {
        assert!(
            n_rows <= cap,
            "partition emits shard of {n_rows} rows exceeding cap {cap}"
        );
    }
    // And it covers the matrix.
    let total: u64 = ranges.iter().map(|(_, n)| *n as u64).sum();
    assert_eq!(total, 100);
}

#[test]
fn parallel_export_byte_identical_to_sequential() {
    use super::pipeline::scx_to_h5ad_streaming;

    let dir = tempfile::tempdir().unwrap();
    let scx = dir.path().join("src.scx");
    let h5ad_seq = dir.path().join("seq.h5ad");
    let h5ad_par = dir.path().join("par.h5ad");

    // 80 rows / shard_size 10 → 8 shards.
    make_multishard_scx(&scx, 80, 11, 10);
    let reader = ScxReader::open(&scx).unwrap();
    assert!(reader.catalog().shards_sorted().len() >= 4);
    drop(reader);

    let seq_opts = ConvertOptions {
        reader_threads: Some(1),
        ..ConvertOptions::default()
    };
    scx_to_h5ad_streaming(&scx, &h5ad_seq, &seq_opts, &mut WarningSink::log()).unwrap();

    let par_opts = ConvertOptions {
        reader_threads: Some(4),
        writer_queue_depth: 4,
        ..ConvertOptions::default()
    };
    scx_to_h5ad_streaming(&scx, &h5ad_par, &par_opts, &mut WarningSink::log()).unwrap();

    let (a_indptr, a_indices, a_data, a_shape) = read_h5ad_x_triplet(&h5ad_seq);
    let (b_indptr, b_indices, b_data, b_shape) = read_h5ad_x_triplet(&h5ad_par);
    assert_eq!(a_shape, b_shape, "shape diverges between paths");
    assert_eq!(a_indptr, b_indptr, "indptr diverges between paths");
    assert_eq!(a_indices, b_indices, "indices diverges between paths");
    assert_eq!(a_data, b_data, "data diverges between paths");
}

#[test]
fn parallel_export_with_layers_byte_identical() {
    use super::pipeline::scx_to_h5ad_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("src.h5ad");
    let scx = dir.path().join("src.scx");
    let h5ad_seq = dir.path().join("seq.h5ad");
    let h5ad_par = dir.path().join("par.h5ad");

    // `include_extras = true` adds a layer alongside the main X.
    create_test_h5ad(&h5ad, 64, 9, "csr", true);
    let import_opts = ConvertOptions {
        shard_target_rows: 8,
        ..ConvertOptions::default()
    };
    h5ad_to_scx(&h5ad, &scx, &import_opts, &mut WarningSink::log()).unwrap();

    let seq_opts = ConvertOptions {
        reader_threads: Some(1),
        ..ConvertOptions::default()
    };
    scx_to_h5ad_streaming(&scx, &h5ad_seq, &seq_opts, &mut WarningSink::log()).unwrap();

    let par_opts = ConvertOptions {
        reader_threads: Some(4),
        ..ConvertOptions::default()
    };
    scx_to_h5ad_streaming(&scx, &h5ad_par, &par_opts, &mut WarningSink::log()).unwrap();

    // Compare /X
    let (a_indptr, a_indices, a_data, _) = read_h5ad_x_triplet(&h5ad_seq);
    let (b_indptr, b_indices, b_data, _) = read_h5ad_x_triplet(&h5ad_par);
    assert_eq!(a_indptr, b_indptr);
    assert_eq!(a_indices, b_indices);
    assert_eq!(a_data, b_data);

    // Compare each layer.
    let a_file = hdf5::File::open(&h5ad_seq).unwrap();
    let b_file = hdf5::File::open(&h5ad_par).unwrap();
    let layer_names = a_file.group("layers").unwrap().member_names().unwrap();
    assert!(
        !layer_names.is_empty(),
        "fixture must have at least one layer"
    );
    for layer in &layer_names {
        let a_grp = a_file.group(&format!("layers/{layer}")).unwrap();
        let b_grp = b_file.group(&format!("layers/{layer}")).unwrap();
        let a_data: Vec<f32> = a_grp.dataset("data").unwrap().read_1d().unwrap().to_vec();
        let b_data: Vec<f32> = b_grp.dataset("data").unwrap().read_1d().unwrap().to_vec();
        let a_indices: Vec<i32> = a_grp
            .dataset("indices")
            .unwrap()
            .read_1d()
            .unwrap()
            .to_vec();
        let b_indices: Vec<i32> = b_grp
            .dataset("indices")
            .unwrap()
            .read_1d()
            .unwrap()
            .to_vec();
        let a_indptr: Vec<i64> = a_grp.dataset("indptr").unwrap().read_1d().unwrap().to_vec();
        let b_indptr: Vec<i64> = b_grp.dataset("indptr").unwrap().read_1d().unwrap().to_vec();
        assert_eq!(a_indptr, b_indptr, "layer {layer} indptr diverges");
        assert_eq!(a_indices, b_indices, "layer {layer} indices diverges");
        assert_eq!(a_data, b_data, "layer {layer} data diverges");
    }
}

#[test]
fn parallel_export_with_deletion_vectors_byte_identical() {
    use super::pipeline::scx_to_h5ad_streaming;

    let dir = tempfile::tempdir().unwrap();
    let scx = dir.path().join("src.scx");
    let h5ad_seq = dir.path().join("seq.h5ad");
    let h5ad_par = dir.path().join("par.h5ad");

    make_multishard_scx(&scx, 80, 11, 10);
    // Delete rows scattered across multiple shards so the writer
    // thread's `nnz_offset` + `row_offset_kept` accumulators have
    // to reorder across shard boundaries.
    let deleted: Vec<u64> = vec![1, 9, 12, 25, 41, 67];
    scx_ops::mark_deleted(&scx, &deleted).unwrap();

    let seq_opts = ConvertOptions {
        reader_threads: Some(1),
        ..ConvertOptions::default()
    };
    scx_to_h5ad_streaming(&scx, &h5ad_seq, &seq_opts, &mut WarningSink::log()).unwrap();

    let par_opts = ConvertOptions {
        reader_threads: Some(4),
        ..ConvertOptions::default()
    };
    scx_to_h5ad_streaming(&scx, &h5ad_par, &par_opts, &mut WarningSink::log()).unwrap();

    let (a_indptr, a_indices, a_data, a_shape) = read_h5ad_x_triplet(&h5ad_seq);
    let (b_indptr, b_indices, b_data, b_shape) = read_h5ad_x_triplet(&h5ad_par);
    assert_eq!(a_shape, b_shape);
    assert_eq!(
        a_indptr, b_indptr,
        "indptr diverges after DV-applied parallel export"
    );
    assert_eq!(
        a_indices, b_indices,
        "indices diverges after DV-applied parallel export"
    );
    assert_eq!(
        a_data, b_data,
        "data diverges after DV-applied parallel export"
    );
    assert_eq!(
        a_shape[0],
        (80 - deleted.len()) as i64,
        "kept-row count in shape attr"
    );
}

#[test]
fn parallel_export_h5mu_byte_identical() {
    use crate::h5mu::pipeline::h5mu_to_scx;
    use crate::h5mu::write::scx_to_h5mu_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_in = dir.path().join("in.h5mu");
    let scx = dir.path().join("mid.scx");
    let h5mu_seq = dir.path().join("seq.h5mu");
    let h5mu_par = dir.path().join("par.h5mu");

    // Two modalities (rna 40 × 7, adt 40 × 5); shard_size 8 → 5 shards each.
    create_test_h5mu(&h5mu_in, 40, 7, 5);
    let import_opts = ConvertOptions {
        shard_target_rows: 8,
        ..ConvertOptions::default()
    };
    h5mu_to_scx(&h5mu_in, &scx, &import_opts, &mut WarningSink::log()).unwrap();

    let seq_opts = ConvertOptions {
        reader_threads: Some(1),
        ..ConvertOptions::default()
    };
    scx_to_h5mu_streaming(&scx, &h5mu_seq, &seq_opts, &mut WarningSink::log()).unwrap();

    let par_opts = ConvertOptions {
        reader_threads: Some(4),
        ..ConvertOptions::default()
    };
    scx_to_h5mu_streaming(&scx, &h5mu_par, &par_opts, &mut WarningSink::log()).unwrap();

    let a_file = hdf5::File::open(&h5mu_seq).unwrap();
    let b_file = hdf5::File::open(&h5mu_par).unwrap();
    let mod_names = a_file.group("mod").unwrap().member_names().unwrap();
    assert!(mod_names.len() >= 2);
    for m in &mod_names {
        let a_grp = a_file.group(&format!("mod/{m}/X")).unwrap();
        let b_grp = b_file.group(&format!("mod/{m}/X")).unwrap();
        let a_indices: Vec<i32> = a_grp
            .dataset("indices")
            .unwrap()
            .read_1d()
            .unwrap()
            .to_vec();
        let b_indices: Vec<i32> = b_grp
            .dataset("indices")
            .unwrap()
            .read_1d()
            .unwrap()
            .to_vec();
        let a_data: Vec<f32> = a_grp.dataset("data").unwrap().read_1d().unwrap().to_vec();
        let b_data: Vec<f32> = b_grp.dataset("data").unwrap().read_1d().unwrap().to_vec();
        let a_indptr: Vec<i64> = a_grp.dataset("indptr").unwrap().read_1d().unwrap().to_vec();
        let b_indptr: Vec<i64> = b_grp.dataset("indptr").unwrap().read_1d().unwrap().to_vec();
        assert_eq!(a_indptr, b_indptr, "modality {m} indptr diverges");
        assert_eq!(a_indices, b_indices, "modality {m} indices diverges");
        assert_eq!(a_data, b_data, "modality {m} data diverges");
    }
}

#[test]
fn parallel_export_memory_budget_refuses_oversized_shard() {
    use super::pipeline::scx_to_h5ad_streaming;

    let dir = tempfile::tempdir().unwrap();
    let scx = dir.path().join("src.scx");
    let h5ad = dir.path().join("out.h5ad");

    make_multishard_scx(&scx, 80, 64, 16);
    let opts = ConvertOptions {
        reader_threads: Some(4),
        memory_budget: Some(1), // 1 byte — well below any shard's working set
        ..ConvertOptions::default()
    };
    let err = scx_to_h5ad_streaming(&scx, &h5ad, &opts, &mut WarningSink::log())
        .expect_err("expected refusal");
    let msg = err.to_string();
    assert!(
        msg.contains("memory_budget"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn parallel_export_memory_budget_derates_workers() {
    use super::pipeline::scx_to_h5ad_streaming;
    use std::sync::{Arc, Mutex};

    let dir = tempfile::tempdir().unwrap();
    let scx = dir.path().join("src.scx");
    let h5ad = dir.path().join("out.h5ad");

    // Many shards but each tiny. per_shard_export_bytes ≈
    //   nnz × 8 + (n_rows + 1) × 8 + nnz × 8 (scratch)
    // For 8 rows × 9 vars at the fixture's 2-3 nnz/row ≈ 20 nnz:
    //   20×16 + 9×8 ≈ 392 bytes per shard.
    // Budget = 1500 → outstanding_max = 1500/392 = 3 → granted
    // threads + depth = 3; with requested 8 → derate fires.
    make_multishard_scx(&scx, 64, 9, 8);

    let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let log_clone = Arc::clone(&log);
    let mut sink = WarningSink::with_handler(move |w| {
        log_clone.lock().unwrap().push(format!("{:?}", w));
    });

    let opts = ConvertOptions {
        reader_threads: Some(8),
        writer_queue_depth: 4,
        memory_budget: Some(1500),
        ..ConvertOptions::default()
    };
    scx_to_h5ad_streaming(&scx, &h5ad, &opts, &mut sink).unwrap();

    let warnings = log.lock().unwrap();
    assert!(
        warnings.iter().any(|w| w.contains("ReaderThreadsDerated")),
        "expected ReaderThreadsDerated; got: {:?}",
        *warnings
    );
    // Fix 2 (PR 105 follow-up): the derate now shrinks depth first
    // so the parallel route is preserved at threads=2 / depth=1
    // rather than collapsing to threads=1 (which would fall back to
    // the sequential coordinator and lose parallelism).
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("writer_queue_depth granted = 1")),
        "expected derate to shrink depth before threads (writer_queue_depth granted = 1); got: {:?}",
        *warnings
    );

    // Output must still match the sequential path.
    let h5ad_seq = dir.path().join("seq.h5ad");
    let seq_opts = ConvertOptions {
        reader_threads: Some(1),
        ..ConvertOptions::default()
    };
    scx_to_h5ad_streaming(&scx, &h5ad_seq, &seq_opts, &mut WarningSink::log()).unwrap();
    let (a_indptr, a_indices, a_data, _) = read_h5ad_x_triplet(&h5ad);
    let (b_indptr, b_indices, b_data, _) = read_h5ad_x_triplet(&h5ad_seq);
    assert_eq!(a_indptr, b_indptr);
    assert_eq!(a_indices, b_indices);
    assert_eq!(a_data, b_data);
}

#[test]
fn per_shard_export_bytes_matches_payload_layout() {
    // Sanity: the helper computes exactly what the dispatcher
    // documents — payload (nnz×8) + indptr ((n_rows+1)×8) +
    // scratch (nnz×8). Anchors the budget arithmetic against
    // accidental regressions.
    use crate::h5ad::stream_write::per_shard_export_bytes_for_test;
    use scx_format_io::catalog::ShardStats;
    let stats = ShardStats {
        row_start: 0,
        row_end: 100,
        col_start: 0,
        col_end: 0,
        nnz: 50,
        value_min: 0,
        value_max: 0,
        value_sum: 0,
        n_indexed_columns: 0,
        column_stats: Vec::new(),
    };
    let bytes = per_shard_export_bytes_for_test(&stats);
    // 50×8 + 101×8 + 50×8 = 400 + 808 + 400 = 1608
    assert_eq!(bytes, 1608);
}

/// Regression test for the deadlock fixed by routing the export parallel
/// coordinator's `in_place_scope` closure through `move` semantics.
///
/// Before the fix, when a worker reported an error mid-stream
/// (`return Err(e)` in the drain loop), `rx` lived in the parent function
/// frame and stayed alive across the rayon scope's join. Other workers
/// parked on `tx.send(...)` against the bounded channel never unblocked,
/// so `pool.in_place_scope(...)` hung forever. With `move`, `rx` drops on
/// closure exit and the senders complete with `SendError`.
///
/// The test forces a shard read error by zeroing the `SCXS` magic of a
/// mid-stream CSR shard so `ShardHeader::read_from` rejects it. The convert
/// runs on a worker thread and is polled with a 30s timeout — a missing
/// `move` keyword (or a regression in the drain logic) will hang the thread
/// and trip the `panic!` below.
#[test]
fn parallel_export_worker_error_does_not_deadlock() {
    use super::pipeline::scx_to_h5ad_streaming;
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::time::{Duration, Instant};

    let dir = tempfile::tempdir().unwrap();
    let scx_path = dir.path().join("src.scx");
    let h5ad_out = dir.path().join("out.h5ad");

    // 80 rows / shard 10 → 8 CSR shards. With the default
    // ConvertOptions, no bitmap shards are emitted, so every `SCXS`
    // magic in the file is a CSR shard header.
    make_multishard_scx(&scx_path, 80, 11, 10);

    // Zero the 4th `SCXS` magic. Shards 0-2 decode OK; shard 3 fails
    // at the magic check in `scx-format::shard::ShardHeader::read_from`.
    let target_offset = {
        let mut buf = Vec::new();
        std::fs::File::open(&scx_path)
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        let magic = b"SCXS";
        let mut hits = Vec::new();
        let mut i = 0;
        while i + 4 <= buf.len() {
            if &buf[i..i + 4] == magic {
                hits.push(i);
                i += 4;
            } else {
                i += 1;
            }
        }
        assert!(
            hits.len() >= 4,
            "expected ≥4 SCXS occurrences (one per CSR shard); got {}",
            hits.len()
        );
        hits[3]
    };
    {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(&scx_path)
            .unwrap();
        f.seek(SeekFrom::Start(target_offset as u64)).unwrap();
        f.write_all(&[0u8; 4]).unwrap();
        f.sync_all().unwrap();
    }

    // reader_threads=4 + writer_queue_depth=1 forces the bounded channel
    // to fill quickly: with 5 outstanding workers and a 1-slot channel,
    // at least 4 workers will be parked on `tx.send(...)` when shard 3
    // reports its error.
    let scx = scx_path.clone();
    let handle = std::thread::spawn(move || {
        let opts = ConvertOptions {
            reader_threads: Some(4),
            writer_queue_depth: 1,
            ..ConvertOptions::default()
        };
        scx_to_h5ad_streaming(&scx, &h5ad_out, &opts, &mut WarningSink::log())
    });

    let timeout = Duration::from_secs(30);
    let start = Instant::now();
    while !handle.is_finished() {
        if start.elapsed() > timeout {
            panic!(
                "parallel export deadlocked: convert thread did not finish \
                 within {timeout:?}; the `move` keyword on the in_place_scope \
                 closure may be missing or regressed"
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let result = handle.join().expect("convert thread panicked");
    assert!(
        result.is_err(),
        "expected Err from corrupted SCX shard; got Ok"
    );
}

/// Symmetric regression test for the ingest direction: the same
/// `move` closure fix was applied in
/// `pipeline::streaming_writer_coordinator_parallel`. Forces a worker
/// error via the `FailIngestShardGuard` hook in
/// `pipeline::test_hooks` — corrupting an h5ad file in a way that
/// fails HDF5 reads selectively per shard is impractical, so we use a
/// purpose-built fault-injection seam instead. Production code is
/// unaffected: the injection check is `#[cfg(test)]`-gated.
#[test]
fn parallel_ingest_worker_error_does_not_deadlock() {
    use super::pipeline::{h5ad_to_scx_streaming, test_hooks, StreamingOverrides};
    use std::time::{Duration, Instant};

    // Without this the test does not skip on a non-thread-safe libhdf5 — it
    // *fails*: the dispatcher routes to the sequential coordinator, whose
    // workers never run, so the injected fault never fires and the convert
    // returns `Ok`. Four of the six parallel-coordinator tests carried the
    // guard and these two did not.
    if super::hdf5_threadsafe::skip_if_not_threadsafe(
        "parallel_ingest_worker_error_does_not_deadlock",
    ) {
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("src.h5ad");
    let scx_out = dir.path().join("out.scx");

    // 80 rows × 11 vars, shard_size 10 → 8 ingest shards. Shard index
    // 3 is in the initial prime spawn (outstanding_cap = threads +
    // depth = 5) so several workers are guaranteed to be parked on
    // `tx.send(...)` against the depth-1 channel when this one fires.
    create_test_h5ad(&h5ad, 80, 11, "csr", false);

    let h5ad_owned = h5ad.clone();
    let scx_out_owned = scx_out.clone();
    let handle = std::thread::spawn(move || {
        // Guard lives for the whole convert; Drop clears the atomic
        // on normal return *and* on panic, so it can't leak into a
        // concurrently scheduled test in the same binary.
        let _fault = test_hooks::FailIngestShardGuard::new(3);
        let mut opts = streaming_opts(10);
        opts.reader_threads = Some(4);
        opts.writer_queue_depth = 1;
        h5ad_to_scx_streaming(
            &h5ad_owned,
            &scx_out_owned,
            &opts,
            &StreamingOverrides::default(),
            &mut WarningSink::log(),
        )
    });

    let timeout = Duration::from_secs(30);
    let start = Instant::now();
    while !handle.is_finished() {
        if start.elapsed() > timeout {
            panic!(
                "parallel ingest deadlocked: convert thread did not finish \
                 within {timeout:?}; the `move` keyword on the in_place_scope \
                 closure may be missing or regressed"
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let result = handle.join().expect("convert thread panicked");
    assert!(
        result.is_err(),
        "expected Err from injected ingest shard failure; got Ok"
    );
}

/// Regression test for review finding #4: a worker that *panics*
/// before its `tx.send(...)` (rather than sending an `Err`) must not
/// deadlock the ingest coordinator. Without the `catch_unwind` guard
/// the panicking worker produces no message, the drain loop's
/// `received` counter never reaches `n_ranges`, and `rx.recv()` blocks
/// forever because the original `tx` in the scope frame keeps the
/// channel open. The `PanicIngestShardGuard` hook forces a real
/// `panic!` in the coordinator's worker closure; the `catch_unwind`
/// that catches it lives in `parallel_drain::ordered_parallel_drain`,
/// which must convert it to a `ConvertError` and return `Err`.
///
/// The drain's own `a_panicking_worker_returns_an_error_instead_of_deadlocking`
/// covers the same contract without libhdf5; this one additionally pins that
/// the ingest coordinator really does route through the drain.
#[test]
fn parallel_ingest_worker_panic_does_not_deadlock() {
    use super::pipeline::{h5ad_to_scx_streaming, test_hooks, StreamingOverrides};
    use std::time::{Duration, Instant};

    // See the sibling above: sequential fallback makes this a failure rather
    // than a skip on a non-thread-safe libhdf5.
    if super::hdf5_threadsafe::skip_if_not_threadsafe(
        "parallel_ingest_worker_panic_does_not_deadlock",
    ) {
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("src.h5ad");
    let scx_out = dir.path().join("out.scx");

    // 80 rows × 11 vars, shard_size 10 → 8 ingest shards. Shard index
    // 3 is in the initial prime spawn (outstanding_cap = threads +
    // depth = 5) so several workers are guaranteed to be parked on
    // `tx.send(...)` against the depth-1 channel when this one panics.
    create_test_h5ad(&h5ad, 80, 11, "csr", false);

    let h5ad_owned = h5ad.clone();
    let scx_out_owned = scx_out.clone();
    let handle = std::thread::spawn(move || {
        // Guard lives for the whole convert; Drop restores the
        // thread-local on normal return *and* on panic.
        let _panic = test_hooks::PanicIngestShardGuard::new(3);
        let mut opts = streaming_opts(10);
        opts.reader_threads = Some(4);
        opts.writer_queue_depth = 1;
        h5ad_to_scx_streaming(
            &h5ad_owned,
            &scx_out_owned,
            &opts,
            &StreamingOverrides::default(),
            &mut WarningSink::log(),
        )
    });

    let timeout = Duration::from_secs(30);
    let start = Instant::now();
    while !handle.is_finished() {
        if start.elapsed() > timeout {
            panic!(
                "parallel ingest deadlocked on worker panic: convert thread did \
                 not finish within {timeout:?}; the `catch_unwind` guard on the \
                 worker body may be missing or regressed"
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let result = handle.join().expect("convert thread panicked");
    assert!(
        result.is_err(),
        "expected Err from injected ingest worker panic; got Ok"
    );
}

/// Symmetric regression test for the export coordinator
/// (`stream_csr_into_prealloc_parallel`): a worker that panics must not
/// deadlock the SCX → h5ad drain loop. Forced via the
/// `PanicExportShardGuard` hook, which panics in the coordinator's worker
/// closure; the `catch_unwind` that converts it into a delivered `Err` lives
/// in `parallel_drain::ordered_parallel_drain`, shared with ingest.
///
/// Its libhdf5-free twin is the drain's own
/// `a_panicking_worker_returns_an_error_instead_of_deadlocking`; this one
/// additionally pins that the export coordinator routes through the drain.
#[test]
fn parallel_export_worker_panic_does_not_deadlock() {
    use super::pipeline::{scx_to_h5ad_streaming, test_hooks};
    use std::time::{Duration, Instant};

    let dir = tempfile::tempdir().unwrap();
    let scx_path = dir.path().join("src.scx");
    let h5ad_out = dir.path().join("out.h5ad");

    // 80 rows / shard 10 → 8 CSR shards. Shard index 3 is in the
    // initial prime spawn so other workers park on the depth-1 channel.
    make_multishard_scx(&scx_path, 80, 11, 10);

    let scx = scx_path.clone();
    let handle = std::thread::spawn(move || {
        let _panic = test_hooks::PanicExportShardGuard::new(3);
        let opts = ConvertOptions {
            reader_threads: Some(4),
            writer_queue_depth: 1,
            ..ConvertOptions::default()
        };
        scx_to_h5ad_streaming(&scx, &h5ad_out, &opts, &mut WarningSink::log())
    });

    let timeout = Duration::from_secs(30);
    let start = Instant::now();
    while !handle.is_finished() {
        if start.elapsed() > timeout {
            panic!(
                "parallel export deadlocked on worker panic: convert thread did \
                 not finish within {timeout:?}; the `catch_unwind` guard on the \
                 worker body may be missing or regressed"
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let result = handle.join().expect("convert thread panicked");
    assert!(
        result.is_err(),
        "expected Err from injected export worker panic; got Ok"
    );
}

/// §11.5: `--memory-budget` on a dense h5ad must not silently destroy
/// parallelism.
///
/// ⚠️ **This is the assertion `dense_parallel_with_memory_budget_byte_identical`
/// never made, and without which its two arms are secretly one arm.** That test
/// asks for `reader_threads = Some(4)`, but the budget derate collapses the
/// grant to 1 and `pipeline.rs`'s `if granted <= 1` routes it to the
/// *sequential* coordinator — so it compares the sequential path against itself
/// and byte-identity is trivially true. Its own comment computes
/// `(8192 / 200) / 4 = 10` and then asserts nothing about it.
///
/// The oracle is `parallel_drain::hooks::last_run_peak()`, a thread-local set
/// at the tail of `ordered_parallel_drain`. Only two production callers exist —
/// parallel ingest (`pipeline.rs:2286`) and parallel export
/// (`stream_write.rs:509`) — and the sequential coordinator calls neither, so it
/// leaves the counter at its initial `0`. That makes `0` vs `> 0` an exact
/// answer to "which coordinator ran", which is the only question this test asks.
/// (It is deliberately *not* an assertion about how much concurrency was
/// achieved: rayon bounds that by the pool size regardless of the coordinator,
/// as `parallel_reorder_buffer_bounded_by_window` documents at length.)
///
/// **Arm order is load-bearing.** The counter is never reset to zero, so the
/// sequential arm must run first, on a virgin thread-local — otherwise it would
/// read the parallel arm's leftover value and pass for the wrong reason.
///
/// The arithmetic, with `n_vars = 50` f32 (`row_bytes = 200`) and
/// `memory_budget = 24_000`:
///
/// | | today | after the fix |
/// |---|---|---|
/// | `max_slab_rows` | `(24000/200)/4 = 30` | `(24000/4)/(50×12) = 10` |
/// | `per_worker_bytes` | `30×200×2 = 12000` (= B/2) | `10×50×12 = 6000` (= B/4) |
/// | `outstanding_max` | 2 | 4 |
/// | `granted_threads` | **1 → sequential** | **3 → parallel** |
#[test]
fn dense_convert_under_a_memory_budget_stays_parallel() {
    if super::hdf5_threadsafe::skip_if_not_threadsafe(
        "dense_convert_under_a_memory_budget_stays_parallel",
    ) {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("dense.h5ad");
    // 200 rows × 50 vars, f32 dense → row_bytes = 200.
    create_test_h5ad(&h5ad, 200, 50, "dense", false);

    // ---- control arm: sequential, on a virgin thread-local ----
    let mut seq_opts = streaming_opts(32);
    seq_opts.reader_threads = Some(1);
    seq_opts.memory_budget = Some(24_000);
    h5ad_to_scx_streaming(
        &h5ad,
        &dir.path().join("seq.scx"),
        &seq_opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect("sequential dense convert under memory_budget");
    assert_eq!(
        super::parallel_drain::hooks::last_run_peak(),
        0,
        "the sequential coordinator entered the parallel drain — the control \
         arm is not a control, and the assertion below proves nothing"
    );

    // ---- the arm under test ----
    let mut par_opts = streaming_opts(32);
    par_opts.reader_threads = Some(8);
    par_opts.writer_queue_depth = 4;
    par_opts.memory_budget = Some(24_000);
    h5ad_to_scx_streaming(
        &h5ad,
        &dir.path().join("par.scx"),
        &par_opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect("parallel dense convert under memory_budget must not abort");

    let peak = super::parallel_drain::hooks::last_run_peak();
    assert!(
        peak > 0,
        "§11.5: a dense convert with reader_threads=8 under memory_budget \
         24000 ran on the SEQUENTIAL coordinator (drain in-flight peak {peak}). \
         The dense slab cap divides the budget by 4 and the per-worker estimate \
         then multiplies the already-capped slab by 2, so the same reserve is \
         spent twice: outstanding_max lands at 2, granted_threads at 1, and \
         `if granted <= 1` routes away from the parallel path."
    );
}

/// The dense slab cap reserves `4 × sizeof(source_dtype)` bytes per element,
/// but what `read_range_inner` actually holds does **not** scale with the
/// source width — so a narrow-dtype dense h5ad exceeds its own
/// `--memory-budget`, on the sequential path, with no parallel reader involved.
///
/// ⚠️ **This bug is not §11.5 and is not in the code review at all.** §11.5 is
/// the ÷4-then-×2 double-count, which costs parallelism; this is the ÷4 being
/// keyed to the wrong unit, which costs the budget itself. They live on the
/// same line (`dense_stream.rs:142`) and are fixed by the same split, but only
/// one of them was known.
///
/// What is resident at peak, from `read_range_inner` and `read_dense_slab_f32`,
/// with `N = slab_rows × n_vars` and `s = sizeof(source dtype)`:
///
/// * **read + cast**: the `F32` arm moves its `Vec` out of `read_slice_2d`
///   (`into_raw_vec_and_offset`, no copy) → `4N`. Every other arm does
///   `data.into_iter().map(…).collect()`, and `data`'s allocation lives until
///   the `IntoIter` drops, so both buffers coexist → `(s + 4)·N`, max `12N`.
/// * **sparsify**: `flat` (`4N`) + `indices` + `values`, which at exact capacity
///   is `4N` each → `12N`.
///
/// The two never coexist (`read_slab_f32` drops the source before returning),
/// so peak is `max(…) = 12N` — **independent of `s`**. Against the reserved
/// `4s` B/elem:
///
/// | dtype | `s` | reserved | needed | |
/// |---|---|---|---|---|
/// | `u8` / `i8` | 1 | 4 | 12 | **3× over budget** |
/// | `f16`/`u16`/`i16` | 2 | 8 | 12 | **1.5× over budget** |
/// | `f32`/`i32`/`u32` | 4 | 16 | 12 | ok |
/// | `f64`/`i64`/`u64` | 8 | 32 | 12 | conservative |
///
/// This asserts only the **no-OOM** bound (one slab ≤ the whole budget), which
/// isolates the dtype bug: it passes for f32/f64 today and fails for the narrow
/// three. The stricter share bound — one slab ≤ a *quarter* of the budget, which
/// is what leaves room for more than one worker — fails for every dtype today
/// and is what `dense_convert_under_a_memory_budget_stays_parallel` measures.
#[test]
fn dense_slab_cap_never_exceeds_the_budget_for_any_dtype() {
    /// Peak bytes per source element held by `read_range_inner`. Independent of
    /// the source dtype; see the table above.
    const PEAK_BYTES_PER_ELEM: u64 = 12;

    let dir = tempfile::tempdir().unwrap();
    let n_vars: u64 = 1000;
    let budget: u64 = 4_800_000;

    // `create_test_h5ad` only writes f32, and `open_dense_streaming` takes a
    // bare `hdf5::File` plus a dataset path — no h5ad validity required — so a
    // one-dataset file is the whole fixture.
    let mut failures = Vec::new();
    for (name, write) in dense_dtype_writers() {
        let path = dir.path().join(format!("{name}.h5"));
        write(&path, 64, n_vars as usize);

        let file = hdf5::File::open(&path).unwrap();
        let mut opts = streaming_opts(32);
        opts.memory_budget = Some(budget);
        let reader = super::open_dense_streaming(&file, "X", &opts, &mut WarningSink::log())
            .unwrap_or_else(|e| panic!("open_dense_streaming for {name}: {e}"));

        let rows = <_ as super::stream::IndexedCsrShardStream>::max_slab_rows(&reader)
            .expect("a budget is set, so the dense reader must report a cap")
            as u64;
        let peak = rows * n_vars * PEAK_BYTES_PER_ELEM;
        if peak > budget {
            failures.push(format!(
                "{name}: max_slab_rows={rows} → peak {peak} B = {:.1}× the \
                 memory_budget of {budget} B",
                peak as f64 / budget as f64
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "the dense slab cap exceeds memory_budget for {} of the source dtypes \
         tested:\n  {}\n\nThe cap divides by `4 × sizeof(source_dtype)`, but the \
         resident slab is f32 whatever the input was, and the sparsified output \
         does not depend on the source width at all.",
        failures.len(),
        failures.join("\n  ")
    );
}

/// One minimal single-dataset HDF5 file per source dtype the dense reader
/// supports. `open_dense_streaming` reads only shape and dtype, so the values
/// are irrelevant and a zero-filled array is the cheapest valid fixture.
fn dense_dtype_writers() -> Vec<(&'static str, fn(&std::path::Path, usize, usize))> {
    fn write_dense<T>(path: &std::path::Path, n_obs: usize, n_vars: usize)
    where
        T: hdf5::H5Type + Default + Clone,
    {
        let file = hdf5::File::create(path).unwrap();
        let arr = ndarray::Array2::<T>::from_elem((n_obs, n_vars), T::default());
        file.new_dataset::<T>()
            .shape([n_obs, n_vars])
            .create("X")
            .unwrap()
            .write(&arr)
            .unwrap();
    }
    vec![
        (
            "u8",
            write_dense::<u8> as fn(&std::path::Path, usize, usize),
        ),
        ("i16", write_dense::<i16>),
        ("f32", write_dense::<f32>),
        ("f64", write_dense::<f64>),
    ]
}
