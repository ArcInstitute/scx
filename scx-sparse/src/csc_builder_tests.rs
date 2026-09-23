//! Tests for the push-based CSC builder.
//!
//! The load-bearing one is [`builder_matches_the_reference_chunk_for_chunk`]:
//! the builder's whole claim is that it produces *the same bytes* as
//! `transpose::transpose_column_chunk` in `2 * nnz` work instead of
//! `2 * nnz * n_chunks`, so the predecessor is kept as the oracle and every
//! other test here is a named corner of that comparison.
//!
//! Why the comparison has to be element-for-element rather than dense-parity:
//! a wrong within-column order is a *permutation*, and every dense-parity,
//! `to_dense` and `col_sums` check in the tree stays green under one. Only an
//! array comparison, the two golden blake3s, and `scx-gpu`'s policy-gated
//! `validate_csc` can see it.

use super::*;
use crate::transpose::{csr_to_csc, reference_chunks, CscArrays};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Widen an emitted shard back to `CscArrays` so it can be compared with the
/// reference. Test-only: production consumes the on-disk widths directly,
/// which is the point of `next_shard_into`.
fn to_csc_arrays(n_rows: usize, indptr: &[u64], indices: &[u32], data: &[f32]) -> CscArrays {
    CscArrays {
        shape: (n_rows, indptr.len() - 1),
        indptr: indptr.iter().map(|&v| v as i64).collect(),
        indices: indices.iter().map(|&v| v as i32).collect(),
        data: data.to_vec(),
    }
}

fn cfg(cols_per_shard: usize, memory_bytes: usize, spill_after_bytes: usize) -> CscBuilderConfig {
    CscBuilderConfig {
        cols_per_shard,
        memory_bytes,
        spill_after_bytes,
        target_buckets: DEFAULT_TARGET_BUCKETS,
        // Tiny blocks so a test-sized matrix actually seals several of them
        // and reaches the sealed/partial seam; 1 MiB would make every test a
        // single-block build.
        block_bytes: 64,
    }
}

/// Build through the serial push and, with `parallel`, again through the
/// parallel one, requiring the two to agree; returns the serial result.
///
/// Every test that goes through here is therefore also a test of the parallel
/// push. The comparison is the emitted arrays, the error text when either
/// fails, and every statistic except the two that depend on *when* blocks were
/// spilled (`spilled_bytes`, `peak_in_memory_bytes`) — the parallel push picks
/// its spill victim differently by design.
fn run_with(
    shards: &[ScxCsr],
    n_rows: usize,
    n_cols: usize,
    cfg: CscBuilderConfig,
    store: Box<dyn SpillStore>,
) -> Result<(Vec<(u64, CscArrays)>, CscBuilderStats), CscBuilderError> {
    let serial = run_mode(shards, n_rows, n_cols, cfg, store, usize::MAX, None);
    #[cfg(feature = "parallel")]
    {
        let par = run_mode(
            shards,
            n_rows,
            n_cols,
            cfg,
            Box::new(MemSpillStore::new()),
            0,
            // Small batches, so a drain spans several of them and a batch can
            // hold more than one group.
            Some(7),
        );
        match (&serial, &par) {
            (Ok((s_out, s_stats)), Ok((p_out, p_stats))) => {
                assert_same(p_out, s_out);
                assert_eq!(p_stats.nnz, s_stats.nnz, "parallel push nnz");
                assert_eq!(
                    p_stats.n_buckets, s_stats.n_buckets,
                    "parallel push buckets"
                );
                assert_eq!(
                    p_stats.first_non_strict_column, s_stats.first_non_strict_column,
                    "parallel push first_non_strict_column"
                );
            }
            (Err(s), Err(p)) => assert_eq!(p.to_string(), s.to_string(), "parallel push error"),
            (s, p) => panic!(
                "serial and parallel pushes disagree on success: serial ok={}, parallel ok={}",
                s.is_ok(),
                p.is_ok()
            ),
        }
    }
    serial
}

fn run_mode(
    shards: &[ScxCsr],
    n_rows: usize,
    n_cols: usize,
    cfg: CscBuilderConfig,
    store: Box<dyn SpillStore>,
    parallel_min_nnz: usize,
    batch_nnz: Option<u64>,
) -> Result<(Vec<(u64, CscArrays)>, CscBuilderStats), CscBuilderError> {
    let mut b = CscBuilder::new(n_rows, n_cols, cfg, store)?;
    b.parallel_min_nnz = parallel_min_nnz;
    let mut row_start = 0u64;
    for s in shards {
        b.push_shard(row_start, s)?;
        row_start += s.n_rows() as u64;
    }
    let mut em = b.finish()?;
    let mut out = Vec::new();
    if let Some(n) = batch_nnz {
        loop {
            let batch = em.next_batch(n)?;
            if batch.is_empty() {
                break;
            }
            for a in batch {
                out.push((
                    a.col_start,
                    to_csc_arrays(n_rows, &a.indptr, &a.indices, &a.data),
                ));
            }
        }
    } else {
        let (mut ip, mut ix, mut dt) = (Vec::new(), Vec::new(), Vec::new());
        while let Some(col_start) = em.next_shard_into(&mut ip, &mut ix, &mut dt)? {
            out.push((col_start, to_csc_arrays(n_rows, &ip, &ix, &dt)));
        }
    }
    Ok((out, em.stats().clone()))
}

fn run(
    shards: &[ScxCsr],
    n_rows: usize,
    n_cols: usize,
    c: CscBuilderConfig,
) -> Result<Vec<(u64, CscArrays)>, CscBuilderError> {
    run_with(shards, n_rows, n_cols, c, Box::new(MemSpillStore::new())).map(|(o, _)| o)
}

/// The predecessor's per-chunk transpose, driven over the same column
/// boundaries production uses.
///
/// `transpose::reference_chunks` is `transpose_column_chunk` plus the
/// `compute_chunk_cols_with_cap` walk that used to wrap it — kept as
/// `#[cfg(test)]` when the public streaming API was deleted, precisely so it
/// can go on being the oracle here.
fn reference(
    shards: &[ScxCsr],
    n_rows: usize,
    n_cols: usize,
    memory_bytes: usize,
    cols_per_shard: usize,
) -> Vec<(u64, CscArrays)> {
    reference_chunks(shards, n_rows, n_cols, memory_bytes, cols_per_shard)
        .expect("reference chunks")
        .into_iter()
        .map(|(col_start, chunk)| (col_start as u64, chunk))
        .collect()
}

fn assert_same(got: &[(u64, CscArrays)], want: &[(u64, CscArrays)]) {
    assert_eq!(
        got.len(),
        want.len(),
        "shard count: the emitted layout must match the predecessor's"
    );
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.0, w.0, "shard {i} col_start");
        assert_eq!(g.1.shape, w.1.shape, "shard {i} shape");
        assert_eq!(g.1.indptr, w.1.indptr, "shard {i} indptr");
        assert_eq!(g.1.indices, w.1.indices, "shard {i} indices");
        assert_eq!(g.1.data, w.1.data, "shard {i} data");
    }
}

/// Build shards from an explicit `(row, col, value)` list, split at `cuts`.
fn shards_from(
    n_rows: usize,
    n_cols: usize,
    entries: &[(usize, usize, f32)],
    cuts: &[usize],
) -> Vec<ScxCsr> {
    let mut bounds = vec![0usize];
    bounds.extend_from_slice(cuts);
    bounds.push(n_rows);
    bounds.dedup();
    let mut out = Vec::new();
    for w in bounds.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for r in lo..hi {
            let mut row: Vec<_> = entries.iter().filter(|e| e.0 == r).collect();
            row.sort_by_key(|e| e.1);
            for e in row {
                indices.push(e.1 as i32);
                data.push(e.2);
            }
            indptr.push(indices.len() as i64);
        }
        out.push(ScxCsr::new_unchecked(
            (hi - lo, n_cols),
            indptr,
            indices,
            data,
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// The proof, executable
// ---------------------------------------------------------------------------

/// Rows split into 1..=4 shards, parts allowed to be **empty** and **size-1**.
///
/// The existing `proptest_csc_roundtrip.rs` feeds a single-shard slice and says
/// so; multi-shard reassembly is exactly where a `row_offset` advanced by the
/// wrong amount lives, and a size-1 or empty part is its strongest detector.
fn arb_shards() -> impl Strategy<Value = (usize, usize, Vec<ScxCsr>)> {
    (1usize..=18, 1usize..=14)
        .prop_flat_map(|(n_rows, n_cols)| {
            let rows = prop::collection::vec(
                prop::collection::hash_set(0usize..n_cols, 0..=std::cmp::min(6, n_cols)),
                n_rows..=n_rows,
            );
            let cuts = prop::collection::vec(0usize..=n_rows, 0..=3);
            (Just(n_rows), Just(n_cols), rows, cuts)
        })
        .prop_map(|(n_rows, n_cols, rows, mut cuts)| {
            let entries: Vec<(usize, usize, f32)> = rows
                .iter()
                .enumerate()
                .flat_map(|(r, cols)| {
                    let mut cs: Vec<usize> = cols.iter().copied().collect();
                    cs.sort_unstable();
                    // Distinct per (row, col) so a permutation shows in `data`
                    // and not only in `indices`.
                    cs.into_iter()
                        .map(move |c| (r, c, (r * 64 + c + 1) as f32))
                        .collect::<Vec<_>>()
                })
                .collect();
            cuts.sort_unstable();
            (n_rows, n_cols, shards_from(n_rows, n_cols, &entries, &cuts))
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Every emitted shard equals `transpose_column_chunk` over the same
    /// column range, **and** the boundary sequence equals the reference
    /// iterator's. The second half is the layout-unchanged claim: this PR
    /// changes what the transpose costs, not what it writes.
    #[test]
    fn builder_matches_the_reference_chunk_for_chunk(
        (n_rows, n_cols, shards) in arb_shards(),
        cols_per_shard in 1usize..=16,
        // A multiple of the predecessor's own per-column reserve
        // (`n_rows * 12`): below one column it refuses outright, and the point
        // here is to compare the two where both answer, across chunk widths
        // from one column to all of them.
        budget_cols in 1usize..=20,
        spill_after_bytes in 0usize..=4096,
    ) {
        let memory_bytes = n_rows * 12 * budget_cols;
        let want = reference(&shards, n_rows, n_cols, memory_bytes, cols_per_shard);
        let got = run(
            &shards,
            n_rows,
            n_cols,
            cfg(cols_per_shard, memory_bytes, spill_after_bytes),
        ).expect("builder");

        prop_assert_eq!(got.len(), want.len());
        for ((gs, ga), (ws, wa)) in got.iter().zip(&want) {
            prop_assert_eq!(gs, ws);
            prop_assert_eq!(ga.shape, wa.shape);
            prop_assert_eq!(&ga.indptr, &wa.indptr);
            prop_assert_eq!(&ga.indices, &wa.indices);
            prop_assert_eq!(&ga.data, &wa.data);
        }
    }

    /// Concatenating the emitted shards reproduces the whole-matrix in-memory
    /// transpose. A second, independent oracle: `transpose_column_chunk` and
    /// `csr_to_csc` are different code, so a change that broke both in the
    /// same way would still have to survive this.
    #[test]
    fn concatenated_shards_match_the_in_memory_transpose(
        (n_rows, n_cols, shards) in arb_shards(),
        cols_per_shard in 1usize..=16,
    ) {
        let whole = {
            let mut indptr = vec![0i64];
            let mut indices = Vec::new();
            let mut data = Vec::new();
            for s in &shards {
                for r in 0..s.n_rows() {
                    let (a, z) = (s.indptr[r] as usize, s.indptr[r + 1] as usize);
                    indices.extend_from_slice(&s.indices[a..z]);
                    data.extend_from_slice(&s.data[a..z]);
                    indptr.push(indices.len() as i64);
                }
            }
            ScxCsr::new_unchecked((n_rows, n_cols), indptr, indices, data)
        };
        let want = csr_to_csc(&whole);
        let got = run(&shards, n_rows, n_cols, cfg(cols_per_shard, 1 << 20, 1 << 20))
            .expect("builder");

        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for (_, chunk) in &got {
            let base = *indptr.last().expect("seeded");
            indptr.extend(chunk.indptr[1..].iter().map(|v| v + base));
            indices.extend_from_slice(&chunk.indices);
            data.extend_from_slice(&chunk.data);
        }
        prop_assert_eq!(indptr, want.indptr);
        prop_assert_eq!(indices, want.indices);
        prop_assert_eq!(data, want.data);
    }

    /// The in-RAM and spilled paths are one encoder, one parser and one
    /// scatter over two byte sources, so all three settings must agree.
    ///
    /// The middle setting is the one that earns its keep: it leaves a bucket
    /// **part on disk and part in RAM**, so a single column's scatter crosses
    /// the seam. `never` and `always` both miss that.
    #[test]
    fn spill_never_always_and_partial_all_agree(
        (n_rows, n_cols, shards) in arb_shards(),
        cols_per_shard in 1usize..=8,
    ) {
        // `block_bytes: 8` is below one row block (16 bytes minimum), so every
        // row seals. Without that, `spill_after_bytes = 0` spills *nothing* on
        // a matrix too small to fill one block — the bound is over sealed
        // blocks, and the partial tail is the declared slack — and the
        // "always" arm would pass while exercising the in-RAM path.
        let arm = |spill_after_bytes| CscBuilderConfig {
            block_bytes: 8,
            ..cfg(cols_per_shard, 1 << 20, spill_after_bytes)
        };
        let nnz: u64 = shards.iter().map(|s| s.nnz() as u64).sum();

        let (never, s_never) = run_with(
            &shards, n_rows, n_cols, arm(usize::MAX), Box::new(MemSpillStore::new()),
        ).expect("no spill");
        let (always, s_always) = run_with(
            &shards, n_rows, n_cols, arm(0), Box::new(MemSpillStore::new()),
        ).expect("all spilled");
        let (partial, s_partial) = run_with(
            &shards, n_rows, n_cols, arm(64), Box::new(MemSpillStore::new()),
        ).expect("partial spill");

        assert_same(&always, &never);
        assert_same(&partial, &never);
        // Premises, so none of the three arms can pass vacuously: the "never"
        // arm must touch no store at all, and the "always" arm must have
        // spilled whenever there was a single nonzero to seal.
        prop_assert_eq!(s_never.spilled_bytes, 0);
        prop_assert!(nnz == 0 || s_always.spilled_bytes > 0);
        prop_assert!(s_partial.spilled_bytes <= s_always.spilled_bytes);
    }

    /// The resident source and the pushed one must be indistinguishable.
    ///
    /// They exist separately because an eager caller already holds the CSR and
    /// bucketing it would be a second copy — but that is a *memory* argument,
    /// and it buys nothing if the two can disagree about bytes. This is the
    /// test that makes keeping both honest; without it they are two
    /// implementations rather than two sources.
    #[test]
    fn the_resident_and_pushed_sources_agree(
        (n_rows, n_cols, shards) in arb_shards(),
        cols_per_shard in 1usize..=16,
        budget_cols in 1usize..=20,
    ) {
        let memory_bytes = n_rows * 12 * budget_cols;
        let pushed = run(
            &shards, n_rows, n_cols,
            CscBuilderConfig { block_bytes: 8, ..cfg(cols_per_shard, memory_bytes, 0) },
        ).expect("pushed");

        let mut src = ResidentCscSource::new(&shards, n_rows, n_cols, cols_per_shard, memory_bytes)
            .expect("resident");
        let (mut ip, mut ix, mut dt) = (Vec::new(), Vec::new(), Vec::new());
        let mut resident = Vec::new();
        while let Some(col_start) = src.next_shard_into(&mut ip, &mut ix, &mut dt).expect("emit") {
            resident.push((col_start, to_csc_arrays(n_rows, &ip, &ix, &dt)));
        }
        assert_same(&resident, &pushed);
    }
}

// ---------------------------------------------------------------------------
// One test per corner of the identity claim
// ---------------------------------------------------------------------------

/// A zero-nnz column still gets its `indptr` entry: the emit loops over the
/// column *range*, never over the columns that happen to be non-empty.
#[test]
fn empty_columns_still_get_indptr_entries() {
    let shards = shards_from(3, 5, &[(0, 1, 1.0), (1, 3, 2.0), (2, 1, 3.0)], &[]);
    let got = run(&shards, 3, 5, cfg(5, 1 << 20, 1 << 20)).expect("builder");
    assert_same(&got, &reference(&shards, 3, 5, 1 << 20, 5));
    assert_eq!(got[0].1.indptr, vec![0, 0, 2, 2, 3, 3]);
}

/// `n_cols == 0` emits **zero** shards, not one empty one — `run_build_csc`'s
/// `BuildCscOutcome::NoSidecar` is decided by that count.
#[test]
fn zero_columns_emits_no_shards() {
    let shards = vec![ScxCsr::new_unchecked((4, 0), vec![0; 5], vec![], vec![])];
    let got = run(&shards, 4, 0, cfg(5000, 1 << 20, 1 << 20)).expect("builder");
    assert!(got.is_empty(), "got {} shards", got.len());
    assert_same(&got, &reference(&shards, 4, 0, 1 << 20, 5000));
}

/// `n_rows == 0` takes the predecessor's `usize::MAX` chunk-width sentinel, so
/// both agree on one all-columns shard with an all-zero `indptr`.
#[test]
fn zero_rows_emits_one_empty_shard() {
    let got = run(&[], 0, 3, cfg(5000, 1 << 20, 1 << 20)).expect("builder");
    assert_same(&got, &reference(&[], 0, 3, 1 << 20, 5000));
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].1.indptr, vec![0i64; 4]);
}

/// A pushed shard with no rows, and rows with no nonzeros, at the start, the
/// middle and the end of a shard.
///
/// This is the case that decides the record format: a row contributing nothing
/// to a bucket writes no block at all, so a design that inferred the row from
/// block *position* would drift by exactly the number of empty rows. Empty
/// rows in all three positions is what makes the drift visible whichever way
/// it goes.
#[test]
fn empty_rows_and_empty_shards_do_not_shift_the_row_axis() {
    // rows 0 and 4 empty, row 2 empty, plus a zero-row shard in the middle.
    let entries = [
        (1, 0, 1.0),
        (1, 2, 2.0),
        (3, 1, 3.0),
        (5, 0, 4.0),
        (5, 2, 5.0),
    ];
    let mut shards = shards_from(6, 3, &entries, &[2, 4]);
    shards.insert(1, ScxCsr::new_unchecked((0, 3), vec![0], vec![], vec![]));
    let got = run(&shards, 6, 3, cfg(2, 1 << 20, 0)).expect("builder");
    // The reference cannot be handed a zero-row shard in the middle and still
    // be the same matrix, so compare against the same shard list minus it.
    let flat = shards_from(6, 3, &entries, &[2, 4]);
    assert_same(&got, &reference(&flat, 6, 3, 1 << 20, 2));
    assert_eq!(got[0].1.indices, vec![1, 5, 3]);
}

/// A column with exactly one nonzero, which is where a prefix-sum off-by-one
/// is visible and nowhere else in a uniform fixture.
#[test]
fn a_single_nonzero_column_round_trips() {
    let shards = shards_from(4, 3, &[(2, 1, 7.5)], &[1, 3]);
    let got = run(&shards, 4, 3, cfg(1, 1 << 20, 0)).expect("builder");
    assert_same(&got, &reference(&shards, 4, 3, 1 << 20, 1));
    assert_eq!(got[1].1.indices, vec![2]);
    assert_eq!(got[1].1.data, vec![7.5]);
}

/// A source row whose column indices are **descending**.
///
/// Within-row order cannot affect within-column order — each row contributes
/// at most one nonzero per column — so the output is unchanged, and the only
/// casualty is a non-monotone `cols` array inside a spill record. That is the
/// concrete reason the record format stores plain `u32` columns rather than
/// delta-coding them, and why this test asserts equality with the reference
/// rather than "the output is canonical".
#[test]
fn a_descending_source_row_still_matches_the_reference() {
    let shards = vec![ScxCsr::new_unchecked(
        (2, 4),
        vec![0, 3, 4],
        vec![3, 1, 0, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    )];
    let (got, stats) = run_with(
        &shards,
        2,
        4,
        cfg(2, 1 << 20, 0),
        Box::new(MemSpillStore::new()),
    )
    .expect("builder");
    assert_same(&got, &reference(&shards, 2, 4, 1 << 20, 2));
    assert_eq!(stats.first_non_strict_column, None);
}

/// A duplicated `(row, col)` is **preserved**, not coalesced, and reported —
/// in both build profiles.
///
/// Coalescing would move bytes relative to the predecessor, which emits both
/// entries adjacent in `j` order. It is still a defect in the *source* — the
/// resulting sidecar violates `validate_csc`'s `sorted` check and the GPU will
/// reject it — so the builder reports the column rather than repairing it.
///
/// This used to be two tests, a `#[should_panic]` one under `debug_assertions`
/// and a reporting one under `not(debug_assertions)`, because the builder
/// carried a `debug_assert!(false, ..)` on a non-strict column. That made the
/// contract depend on the build profile for input `run_build_csc` is
/// documented to accept: `ScxCsr::new` does not reject duplicates (its own
/// docs say so) and pre-v3 input is deliberately not canonicalised. The panic
/// is gone and the behaviour is the same either way.
#[test]
fn a_duplicate_row_col_is_preserved_and_reported() {
    let shards = vec![ScxCsr::new_unchecked(
        (1, 2),
        vec![0, 2],
        vec![1, 1],
        vec![3.0, 4.0],
    )];
    let (got, stats) = run_with(
        &shards,
        1,
        2,
        cfg(2, 1 << 20, 0),
        Box::new(MemSpillStore::new()),
    )
    .expect("builder");
    assert_same(&got, &reference(&shards, 1, 2, 1 << 20, 2));
    assert_eq!(got[0].1.data, vec![3.0, 4.0]);
    assert_eq!(stats.first_non_strict_column, Some(1));
}

/// Enough duplicates in one row overflow what `block_capacity` reserves (one
/// record per column the bucket owns), so the staging block reallocates.
///
/// The accounting must then charge what was really allocated. It used to add
/// the nominal `block_capacity` and assert the block had not grown — which
/// panics in debug on input the op accepts, and silently under-charges the
/// budget in release, on the one row the allocation table calls `enforced`.
#[test]
fn duplicate_heavy_rows_do_not_break_the_staging_accounting() {
    // One row, one column, 4,000 duplicate entries — far past what a
    // single-column bucket reserves.
    let n = 4_000usize;
    let shards = vec![ScxCsr::new_unchecked(
        (1, 1),
        vec![0, n as i64],
        vec![0; n],
        (0..n).map(|k| (k + 1) as f32).collect(),
    )];
    // A real spill budget, not `usize::MAX`: the point is that the declared
    // bound HOLDS for this input, not merely that the overshoot is measured
    // after it has happened.
    let block_bytes = 64usize;
    let spill_after = 256usize;
    let c = CscBuilderConfig {
        block_bytes,
        ..cfg(1, 1 << 20, spill_after)
    };
    let (got, stats) = run_with(&shards, 1, 1, c, Box::new(MemSpillStore::new())).expect("builder");

    assert_same(&got, &reference(&shards, 1, 1, 1 << 20, 1));
    assert_eq!(got[0].1.indices.len(), n);
    assert_eq!(stats.first_non_strict_column, Some(0));

    // 32 kB of payload in ONE row against a 256-byte staging budget. The
    // bound holds because `push_shard` seals AND spills at the push that
    // crossed `block_bytes`, inside the row. Sealing alone is not enough — it
    // only moves bytes from `cur` into `sealed` — and running either at the
    // row's END leaves the peak unbounded in `n`, which is what this measured
    // at 45,760 bytes before the split. (`Bucket::push` itself never cuts; it
    // starts a fresh row block only because `seal_bucket` called `close_row`.)
    let block_capacity = block_bytes + 8 + SPILL_BYTES_PER_NNZ;
    let bound = spill_after + 2 * stats.n_buckets * block_capacity;
    assert!(
        stats.peak_in_memory_bytes <= bound as u64,
        "peak {} exceeds the declared bound {bound} on duplicate-heavy input \
         ({} bytes of payload in a single row)",
        stats.peak_in_memory_bytes,
        n * SPILL_BYTES_PER_NNZ,
    );
    assert!(
        stats.spilled_bytes > 0,
        "premise: the row must have spilled rather than staying resident"
    );
}

// ---------------------------------------------------------------------------
// Contract errors — each one a condition the predecessor handled worse
// ---------------------------------------------------------------------------

/// The predecessor derived the global row from *slice position* and had no way
/// to notice a permuted, repeated or skipped shard. For the rewrite ops this
/// would shift the whole row axis silently, so the sink checks it.
#[test]
fn an_out_of_order_push_is_refused() {
    let shards = shards_from(4, 2, &[(0, 0, 1.0), (2, 1, 2.0)], &[2]);
    let mut b =
        CscBuilder::new(4, 2, cfg(2, 1 << 20, 0), Box::new(MemSpillStore::new())).expect("new");
    b.push_shard(0, &shards[0]).expect("first");
    let err = b.push_shard(0, &shards[1]).expect_err("repeated row_start");
    assert!(
        matches!(
            err,
            CscBuilderError::RowStartMismatch {
                expected: 2,
                got: 0
            }
        ),
        "{err}"
    );
}

/// The predecessor used `n_rows_total` for `shape` and the shard sum for
/// `indices`, so a mismatch produced arrays whose shape and contents
/// disagreed. Silent then, named now.
#[test]
fn pushing_the_wrong_number_of_rows_is_refused() {
    let shards = shards_from(2, 2, &[(0, 0, 1.0)], &[]);
    let mut b =
        CscBuilder::new(5, 2, cfg(2, 1 << 20, 0), Box::new(MemSpillStore::new())).expect("new");
    b.push_shard(0, &shards[0]).expect("push");
    let err = b.finish().err().expect("short");
    assert!(
        matches!(
            err,
            CscBuilderError::RowCountMismatch {
                pushed: 2,
                declared: 5
            }
        ),
        "{err}"
    );
}

#[test]
fn a_shard_with_the_wrong_column_count_is_refused() {
    let wrong = ScxCsr::new_unchecked((1, 3), vec![0, 0], vec![], vec![]);
    let mut b =
        CscBuilder::new(1, 2, cfg(2, 1 << 20, 0), Box::new(MemSpillStore::new())).expect("new");
    let err = b.push_shard(0, &wrong).expect_err("shape");
    assert!(
        matches!(
            err,
            CscBuilderError::ShapeMismatch {
                shard_cols: 3,
                expected_cols: 2
            }
        ),
        "{err}"
    );
}

/// A deliberate non-identity. The predecessor casts `(row_offset + row) as
/// i32` and lets it wrap negative, after which `validate_csc` rejects the file
/// at the GPU or a CPU kernel's `row >= n_obs` guard silently drops the value.
#[test]
fn more_rows_than_i32_can_index_is_refused_up_front() {
    let made = CscBuilder::new(
        i32::MAX as usize + 1,
        2,
        CscBuilderConfig::default(),
        Box::new(MemSpillStore::new()),
    );
    match made {
        Err(CscBuilderError::RowCountOverflow(n)) => assert_eq!(n, i32::MAX as usize + 1),
        Err(other) => panic!("wrong error: {other}"),
        Ok(_) => panic!("2^31 rows must be refused: the on-disk CSC row index is i32"),
    }
}

/// A builder given nowhere to spill fails with a budget message, not an I/O
/// one, and fails before it has written anything.
#[test]
fn a_no_spill_builder_refuses_rather_than_touching_disk() {
    let entries: Vec<(usize, usize, f32)> = (0..64).map(|r| (r, r % 4, (r + 1) as f32)).collect();
    let shards = shards_from(64, 4, &entries, &[]);
    let mut b = CscBuilder::in_memory(64, 4, cfg(4, 1 << 20, 0)).expect("new");
    let err = b.push_shard(0, &shards[0]).expect_err("must refuse");
    assert!(matches!(err, CscBuilderError::SpillRefused { .. }), "{err}");
}

/// ...and the same builder succeeds when the staging fits, so the test above
/// is about the budget and not about `NoSpillStore` being broken.
#[test]
fn a_no_spill_builder_succeeds_within_its_budget() {
    let entries: Vec<(usize, usize, f32)> = (0..64).map(|r| (r, r % 4, (r + 1) as f32)).collect();
    let shards = shards_from(64, 4, &entries, &[]);
    let mut b = CscBuilder::in_memory(64, 4, cfg(4, 1 << 20, usize::MAX)).expect("new");
    b.push_shard(0, &shards[0]).expect("push");
    let em = b.finish().expect("finish");
    assert_eq!(em.stats().spilled_bytes, 0);
    assert_eq!(em.plan().len(), 1);
}

// ---------------------------------------------------------------------------
// The plan, and the bucket cap
// ---------------------------------------------------------------------------

/// The plan is exact before a byte is read back — which is what finally gives
/// `build-csc`'s progress bar a real denominator.
#[test]
fn the_plan_is_exact_before_any_record_is_read() {
    // `(r % 7, (r + 1) % 7)` can never collide, where `(r % 7, (r * 3) % 7)`
    // does at r = 0 — a duplicate (row, col), which is a defect in the
    // fixture and not something to assert about.
    let entries: Vec<(usize, usize, f32)> = (0..10)
        .flat_map(|r| [(r, r % 7, 1.0), (r, (r + 1) % 7, 2.0)])
        .collect();
    let shards = shards_from(10, 7, &entries, &[3, 7]);
    let mut b =
        CscBuilder::new(10, 7, cfg(3, 1 << 20, 0), Box::new(MemSpillStore::new())).expect("new");
    let mut row_start = 0u64;
    for s in &shards {
        b.push_shard(row_start, s).expect("push");
        row_start += s.n_rows() as u64;
    }
    let mut em = b.finish().expect("finish");
    let plan: Vec<CscShardSpec> = em.plan().to_vec();
    assert_eq!(plan.len(), 3);
    assert_eq!(
        plan.iter()
            .map(|s| (s.col_start, s.col_end))
            .collect::<Vec<_>>(),
        vec![(0, 3), (3, 6), (6, 7)]
    );
    let (mut ip, mut ix, mut dt) = (Vec::new(), Vec::new(), Vec::new());
    for spec in &plan {
        em.next_shard_into(&mut ip, &mut ix, &mut dt).expect("emit");
        assert_eq!(ix.len() as u64, spec.nnz, "planned nnz for {spec:?}");
    }
}

/// More shards than `MAX_BUCKETS` means buckets coarser than one shard, which
/// is the degenerate arm: a bucket then serves several shards and the emit's
/// column dispatch has to place each record in the right one. Output must
/// still be identical.
#[test]
fn more_shards_than_buckets_still_matches_the_reference() {
    let n_cols = 600;
    let entries: Vec<(usize, usize, f32)> = (0..8)
        .flat_map(|r| {
            (0..n_cols)
                .step_by(7)
                .map(move |c| (r, c, (r * 601 + c) as f32))
        })
        .collect();
    let shards = shards_from(8, n_cols, &entries, &[3]);
    let c = CscBuilderConfig {
        target_buckets: 4,
        ..cfg(1, 1 << 20, 0)
    };
    let got = run(&shards, 8, n_cols, c).expect("builder");
    assert_eq!(
        got.len(),
        n_cols,
        "one shard per column at cols_per_shard=1"
    );
    assert_same(&got, &reference(&shards, 8, n_cols, 1 << 20, 1));
}

/// `cols_per_shard == 0` and `usize::MAX` both mean "no cap", matching
/// `compute_chunk_cols_with_cap`'s convention.
#[test]
fn zero_and_usize_max_cols_per_shard_both_mean_no_cap() {
    let shards = shards_from(3, 6, &[(0, 0, 1.0), (1, 5, 2.0), (2, 3, 3.0)], &[1]);
    for cps in [0usize, usize::MAX] {
        let got = run(&shards, 3, 6, cfg(cps, 1 << 30, 0)).expect("builder");
        assert_eq!(got.len(), 1, "cols_per_shard={cps}");
        assert_same(&got, &reference(&shards, 3, 6, 1 << 30, cps));
    }
}

// ---------------------------------------------------------------------------
// At a shape where the spill actually engages
// ---------------------------------------------------------------------------

/// The proptest above runs at 18 x 14. This runs at a shape where the spill is
/// doing real work — thousands of row blocks across dozens of buckets, several
/// sealed blocks per bucket, and a budget that forces most of them out — and
/// still demands element-for-element equality with the reference.
///
/// It is the one test that would notice a divergence which only appears once
/// there is more than a handful of blocks per bucket: a sealed-block ordering
/// bug, a bucket whose disk prefix and RAM tail are stitched the wrong way
/// round at depth, or a `u32` row that only overflows past 65k. None of those
/// can arise at proptest scale.
///
/// 3,000 x 600 at ~4 % density is ~72k nonzeros: large enough for all of that,
/// small enough to stay a unit test.
#[test]
fn a_spilling_build_at_scale_matches_the_reference() {
    const N_ROWS: usize = 3_000;
    const N_COLS: usize = 600;

    // Deterministic, and deliberately *skewed*: column `c` is hit roughly in
    // proportion to `600 - c`, so the buckets are far from equal and the
    // "spill the largest bucket" rule is actually exercised rather than
    // draining a uniform set.
    let mut shards = Vec::new();
    let mut row = 0usize;
    for rows in [777usize, 1, 1_222, 1_000] {
        let mut indptr = vec![0i64];
        let mut indices: Vec<i32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        for r in row..row + rows {
            let mut cols: Vec<usize> = (0..24)
                .map(|k| (r * 31 + k * k * 7) % N_COLS)
                .filter(|c| (r + c) % 3 != 0)
                .collect();
            cols.sort_unstable();
            cols.dedup();
            for c in &cols {
                indices.push(*c as i32);
                data.push((r * N_COLS + c + 1) as f32);
            }
            indptr.push(indices.len() as i64);
        }
        shards.push(ScxCsr::new_unchecked((rows, N_COLS), indptr, indices, data));
        row += rows;
    }
    assert_eq!(row, N_ROWS);
    let nnz: usize = shards.iter().map(|s| s.nnz()).sum();
    assert!(nnz > 40_000, "premise: {nnz} nonzeros is too few to spill");

    let memory_bytes = N_ROWS * 12 * 40; // 40-column shards -> 15 of them
    let cfg = CscBuilderConfig {
        cols_per_shard: 64,
        memory_bytes,
        // A tenth of the payload, so most buckets spill and the ones that do
        // not leave a RAM tail for a column's scatter to cross.
        spill_after_bytes: nnz * SPILL_BYTES_PER_NNZ / 10,
        target_buckets: DEFAULT_TARGET_BUCKETS,
        block_bytes: 4096,
    };

    let (got, stats) =
        run_with(&shards, N_ROWS, N_COLS, cfg, Box::new(MemSpillStore::new())).expect("builder");

    // Premises, so a green run cannot mean "it never spilled": most of the
    // payload left RAM, and it left in many blocks rather than one.
    assert!(
        stats.spilled_bytes > (nnz * SPILL_BYTES_PER_NNZ / 2) as u64,
        "only {} of ~{} payload bytes spilled",
        stats.spilled_bytes,
        nnz * SPILL_BYTES_PER_NNZ
    );
    assert!(stats.n_buckets > 4, "{} buckets", stats.n_buckets);
    assert!(got.len() > 8, "{} emitted shards", got.len());

    assert_same(&got, &reference(&shards, N_ROWS, N_COLS, memory_bytes, 64));
}

/// The staging bound is a **promise**, so assert it rather than the code that
/// is supposed to keep it.
///
/// This test did not exist when the builder was written, and its absence cost
/// a measurement: `in_memory_bytes` counted the bytes *written* into each
/// staging block while the process pays for each block's **capacity**, and a
/// `Vec` grown by `extend_from_slice` doubles — so the realized bound was up
/// to 2x the declared one. The census_1m capture showed 1.92x of a 4 GiB
/// budget and tabula's peak got 10 % *worse* than the code being replaced,
/// which is how it was found. A green suite said nothing, because nothing
/// asserted the number.
///
/// The bound is `spill_after_bytes + 2 * n_buckets * block_capacity`: the
/// share the builder spills against, plus the slack `CscBuilderConfig`
/// declares. The factor of two is the second thing this test found — sealing
/// runs once per pushed row across every bucket that row touched, and the
/// spill loop runs after that sweep, so a row writing into all of them leaves
/// a sealed block *and* a fresh live block per bucket before anything is
/// released. The first draft of this test declared `1 *` and failed at
/// 9,072 against 8,256, which is the gap being described rather than a
/// tolerance being widened to fit.
#[test]
fn staged_bytes_never_exceed_the_declared_bound() {
    let entries: Vec<(usize, usize, f32)> = (0..400)
        .flat_map(|r| (0..40).map(move |k| (r, (r * 13 + k * 7) % 96, (r + k + 1) as f32)))
        .collect();
    let shards = shards_from(400, 96, &entries, &[100, 101, 250]);

    // Both pushes: the parallel one spills a different victim, and the bound
    // is the claim that choice must not break.
    for (spill_after_bytes, parallel_min_nnz) in [0usize, 1 << 10, 1 << 14, usize::MAX]
        .into_iter()
        .flat_map(|s| [(s, usize::MAX), (s, 0)])
    {
        let c = CscBuilderConfig {
            cols_per_shard: 8,
            memory_bytes: 400 * 12 * 8,
            spill_after_bytes,
            target_buckets: 8,
            block_bytes: 512,
        };
        let mut b = CscBuilder::new(400, 96, c, Box::new(MemSpillStore::new())).expect("new");
        b.parallel_min_nnz = parallel_min_nnz;
        let mut row_start = 0u64;
        for sh in &shards {
            b.push_shard(row_start, sh).expect("push");
            row_start += sh.n_rows() as u64;
        }
        let em = b.finish().expect("finish");
        let stats = em.stats();

        // The same arithmetic the config documents, recomputed here rather
        // than read back from the builder, so a change to how the builder
        // sizes a block has to be reflected in both.
        //
        // 96 columns at 8 per shard is 12 shards; `target_buckets: 8` caps the
        // bucket count at 8, so each bucket owns ceil(12/8) = 2 shards = 16
        // columns. One header plus one nonzero per owned column is the widest
        // row block a bucket can hold.
        let bucket_cols = 2 * 8;
        let max_row_block = 8 + SPILL_BYTES_PER_NNZ * bucket_cols;
        let block_capacity = 512 + max_row_block;
        let bound = spill_after_bytes.saturating_add(2 * stats.n_buckets * block_capacity);
        assert!(
            stats.peak_in_memory_bytes <= bound as u64,
            "spill_after_bytes={spill_after_bytes}, parallel_min_nnz={parallel_min_nnz}: \
             peak {} exceeds the declared bound {bound} ({} buckets x {block_capacity} B \
             of slack)",
            stats.peak_in_memory_bytes,
            stats.n_buckets,
        );
        // Premise: at the tight settings the builder really did have to spill,
        // so the bound above is not being met by never filling anything.
        if spill_after_bytes <= 1 << 14 {
            assert!(
                stats.spilled_bytes > 0,
                "spill_after_bytes={spill_after_bytes} should have forced a spill"
            );
        }
    }
}

/// Every bucket the layout allocates must be reachable.
///
/// `n_buckets` used to be `target.min(n_shards)` and was *not* recomputed after
/// `shards_per_bucket` rounded up, so a shard count just above a multiple of the
/// target produced buckets no column could route to — 31 of them at
/// `n_shards = 65, target = 64`. Each one still allocated a live block and was
/// charged against the staging budget, and `bucket_shards` gave it an inverted
/// range.
#[test]
fn every_allocated_bucket_is_reachable() {
    // 65 columns at 1 per shard against a 64-bucket target: the exact shape.
    for (n_cols, target) in [(65usize, 64usize), (173, 64), (12, 8), (129, 64), (7, 64)] {
        let entries: Vec<(usize, usize, f32)> = (0..4)
            .flat_map(|r| (0..n_cols).map(move |c| (r, c, (r * n_cols + c + 1) as f32)))
            .collect();
        let shards = shards_from(4, n_cols, &entries, &[2]);
        let c = CscBuilderConfig {
            target_buckets: target,
            ..cfg(1, 1 << 30, 0)
        };
        let (got, stats) =
            run_with(&shards, 4, n_cols, c, Box::new(MemSpillStore::new())).expect("builder");

        // Buckets are exactly ceil(n_shards / spb) — which is the assertion
        // that matters, because a phantom bucket is one the layout allocates
        // and no column can route to.
        //
        // (An earlier draft looped `for b in 0..n_buckets` asserting the
        // *global* `spilled_bytes > 0` and discarding `b`, which passes if
        // bucket 0 spills and every other bucket is empty — the exact
        // condition it claimed to rule out.)
        let n_shards = n_cols; // cols_per_shard = 1
        let want = target.min(n_shards);
        let spb = n_shards.div_ceil(want);
        assert_eq!(
            stats.n_buckets,
            n_shards.div_ceil(spb),
            "n_cols={n_cols} target={target}: allocated {} buckets for {n_shards} shards \
             at {spb} shards each",
            stats.n_buckets,
        );
        assert_same(&got, &reference(&shards, 4, n_cols, 1 << 30, 1));
    }
}

/// A corrupt spill stream must come back as `SpillCorrupt`, not a panic.
///
/// The first version recorded the bad column and returned from that one row
/// block, leaving `drain_bucket` to walk the rest — so a later record could
/// index past the planned slots and panic before the note was read. The
/// callback is fallible now and stops at the first bad record.
#[test]
fn a_corrupt_spill_stream_errors_instead_of_panicking() {
    /// A store that hands back bytes nobody wrote.
    #[derive(Default)]
    struct EvilStore {
        inner: MemSpillStore,
        payload: Vec<u8>,
    }
    impl SpillStore for EvilStore {
        fn append(&mut self, bucket: usize, block: &[u8]) -> std::io::Result<()> {
            self.inner.append(bucket, block)
        }
        fn reader(&self, _bucket: usize) -> std::io::Result<Option<Box<dyn std::io::Read + '_>>> {
            Ok(Some(Box::new(self.payload.as_slice())))
        }
        fn spilled_bytes(&self, _bucket: usize) -> u64 {
            self.payload.len() as u64
        }
    }

    // One row block claiming a column far outside any bucket's range.
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_le_bytes()); // row 0
    payload.extend_from_slice(&1u32.to_le_bytes()); // one record
    payload.extend_from_slice(&9_999u32.to_le_bytes()); // column 9999
    payload.extend_from_slice(&1.0f32.to_le_bytes());

    let shards = shards_from(2, 4, &[(0, 0, 1.0), (1, 3, 2.0)], &[]);
    let store = EvilStore {
        payload,
        ..Default::default()
    };
    let mut b = CscBuilder::new(2, 4, cfg(2, 1 << 20, 0), Box::new(store)).expect("new");
    b.push_shard(0, &shards[0]).expect("push");
    let mut em = b.finish().expect("finish");
    let (mut ip, mut ix, mut dt) = (Vec::new(), Vec::new(), Vec::new());
    let err = em
        .next_shard_into(&mut ip, &mut ix, &mut dt)
        .expect_err("a column outside the bucket must be refused");
    assert!(
        matches!(err, CscBuilderError::SpillCorrupt { .. }),
        "expected SpillCorrupt, got {err}"
    );
}

/// With `parallel`, fewer shards than `target_buckets` split each shard into
/// equal buckets: every bucket lies inside one shard, every allocated bucket
/// is reachable, and a shard width with no divisor in range stays coarse.
#[cfg(feature = "parallel")]
#[test]
fn fine_buckets_never_straddle_a_shard() {
    for (n_rows, n_cols, cols_per_shard, target) in [
        (10usize, 61_497usize, 5000usize, 64usize),
        (10, 100, 12, 64),
        (10, 100, 7, 64), // prime width: coarse
        (10, 40, 0, 16),  // no cap: one shard
        (10, 13, 13, 4),
        (0, 50, 10, 64), // no rows: one unbounded shard
    ] {
        let cfg = CscBuilderConfig {
            cols_per_shard,
            memory_bytes: 1 << 30,
            spill_after_bytes: usize::MAX,
            target_buckets: target,
            block_bytes: 64,
        };
        let l = Layout::plan(n_rows, n_cols, &cfg).expect("plan");
        let label = format!("{n_rows}x{n_cols} cols_per_shard={cols_per_shard} target={target}");
        if l.buckets_per_shard > 1 {
            assert!(
                l.n_shards < target,
                "{label}: fine with {} shards",
                l.n_shards
            );
            assert!(l.n_buckets <= target.max(l.n_shards), "{label}");
            for b in 0..l.n_buckets {
                let lo = b * l.bucket_cols;
                let hi = (lo + l.bucket_cols).min(n_cols) - 1;
                assert!(
                    lo < n_cols,
                    "{label}: bucket {b} starts past the last column"
                );
                assert_eq!(
                    lo / l.shard_cols,
                    hi / l.shard_cols,
                    "{label}: bucket {b} straddles"
                );
                assert_eq!(l.bucket_of(lo), b, "{label}");
                assert_eq!(l.bucket_of(hi), b, "{label}");
            }
            // Each group's buckets are exactly its shard's.
            for g in 0..l.n_groups() {
                let ((s0, s1), (b0, b1)) = l.group(g);
                assert_eq!((s0, s1), (g, g + 1), "{label}");
                let (c0, c1) = l.shard_range(g, n_cols);
                assert_eq!(b0 * l.bucket_cols, c0, "{label}: group {g}");
                assert!((b1 * l.bucket_cols).min(n_cols) >= c1, "{label}: group {g}");
            }
        } else {
            assert_eq!(l.n_groups(), l.n_buckets, "{label}");
        }
        if cols_per_shard == 7 {
            assert_eq!(
                l.buckets_per_shard, 1,
                "{label}: a prime width has no divisor"
            );
        }
        if cols_per_shard == 5000 {
            assert!(
                l.buckets_per_shard > 1,
                "{label}: the census layout must go fine"
            );
        }
    }
}
