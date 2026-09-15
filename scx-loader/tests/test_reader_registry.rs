//! Bounded reader registry (W8): byte identity, deferred eviction, reopen
//! identity, and residency.
//!
//! # What is being bounded here
//!
//! Not file descriptors. `ScxReader::open` mmaps and lets the `File` drop, and
//! `scx-loader` never calls `ScxReader::watching`, so a manifest of N files
//! holds N mappings and **zero** descriptors — measured on the real Python
//! constructor, a 5,000-file manifest constructs and gathers under
//! `ulimit -n 1024` with the process's descriptor count flat. What an open
//! reader costs is ~104 kB resident, over 90 % of it the parsed `FullCatalog`.
//! `reader_limit` bounds how many of those exist at once.
//!
//! Every file here carries a distinct `tag` in its values, so a registry that
//! answered `file_id` 7 from file 3's mapping produces different bytes rather
//! than the same ones. A fixture whose files are byte-identical would let that
//! defect pass, which is the whole point of the mutation in
//! `bounded_and_unbounded_gathers_are_byte_identical`.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;
use scx_loader::sparse_cellset::{SparseCellSetBatch, SparseCellSetLoader, SparseCellSetPlan};

const N_VARS: usize = 8;
const ROWS: usize = 8;
const SHARDS: usize = 2;

/// One tiny fixture whose every value is `tag`, so its bytes name the file.
fn write_tagged(path: &Path, tag: u8) -> PathBuf {
    let rows_per_shard = ROWS / SHARDS;
    let header = FileHeader::new_single_modality(
        ROWS as u64,
        N_VARS as u64,
        ROWS as u64,
        rows_per_shard as u32,
        0,
        0,
    );
    let mut writer = ScxWriter::new(path, header).unwrap();
    let cells: Vec<String> = (0..ROWS).map(|i| format!("cell_{i}")).collect();
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
    for s in 0..SHARDS {
        let row_start = s * rows_per_shard;
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for local in 0..rows_per_shard {
            indices.push(((row_start + local) % N_VARS) as u32);
            values.push(tag);
            indptr.push(*indptr.last().unwrap() + 1);
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
    path.to_path_buf()
}

/// `n` files, file `i` tagged `i + 1` (tag 0 would be an implicit zero).
fn manifest(dir: &Path, n: usize) -> Vec<PathBuf> {
    (0..n)
        .map(|i| write_tagged(&dir.join(format!("f{i}.scx")), (i + 1) as u8))
        .collect()
}

fn open_loader(paths: Vec<PathBuf>, reader_limit: Option<usize>) -> Arc<SparseCellSetLoader> {
    open_loader_lookahead(paths, reader_limit, 0)
}

/// The same loader with a caller-chosen `lookahead`, so a test can drive the
/// prefetching path rather than only the synchronous gather.
#[allow(clippy::too_many_arguments)]
fn open_loader_lookahead(
    paths: Vec<PathBuf>,
    reader_limit: Option<usize>,
    lookahead: usize,
) -> Arc<SparseCellSetLoader> {
    SparseCellSetLoader::open(
        paths,
        /* cache_shards */ 4,
        /* bytes_budget */ Some(1 << 20),
        lookahead,
        /* remap */ None,
        /* n_global_genes */ None,
        /* normalize */ false,
        /* log1p */ false,
        /* target_sum */ 1e4,
        /* downsample */ None,
        /* scatter_block_index */ false,
        /* max_plan_rows */ None,
        reader_limit,
    )
    .expect("loader opens")
}

/// One single-file set per file, in manifest order — the access pattern that
/// makes a small `reader_limit` evict on every step.
fn one_set_per_file(n: usize) -> SparseCellSetPlan {
    SparseCellSetPlan {
        file_ids: (0..n as u32).collect(),
        rows: (0..n).map(|i| (i % ROWS) as u64).collect(),
        role_tags: vec![0; n],
        set_offsets: (0..=n as i64).collect(),
    }
}

/// Every field of a batch that a registry could plausibly corrupt, compared as
/// one value so a diverging one names itself in the assertion.
#[derive(PartialEq, Debug)]
struct Fields {
    indptr: Vec<i64>,
    indices: Vec<i32>,
    data: Vec<f32>,
    cell_indices: Vec<u64>,
    file_ids: Vec<u32>,
}

fn fields(b: &SparseCellSetBatch) -> Fields {
    Fields {
        indptr: b.indptr.clone(),
        indices: b.indices.clone(),
        data: b.data.clone(),
        cell_indices: b.cell_indices.clone(),
        file_ids: b.file_ids.clone(),
    }
}

/// **The identity claim.** A bounded registry must change residency and nothing
/// else.
///
/// Mutation (applied, seen to fail): in `ReaderRegistry::lease`, reopen the
/// slot of `file_id 0` instead of the requested one — i.e. reuse a vacated
/// `file_id` for another path. The decoded-shard cache is keyed
/// `CacheKey::Shard(file_id, shard)` with no path and no generation, so the
/// wrong file's bytes come back as a hit and `data` diverges.
#[test]
fn bounded_and_unbounded_gathers_are_byte_identical() {
    let dir = tempfile::tempdir().unwrap();
    let paths = manifest(dir.path(), 64);
    let plan = one_set_per_file(64);

    let unbounded = open_loader(paths.clone(), None).gather(&plan).unwrap();
    for limit in [1usize, 8, 63, 64, 128] {
        let bounded = open_loader(paths.clone(), Some(limit))
            .gather(&plan)
            .unwrap();
        assert_eq!(
            fields(&bounded),
            fields(&unbounded),
            "reader_limit={limit} changed the gathered bytes"
        );
    }

    // The premise the identity rests on: the files really are distinguishable,
    // so an identical result is evidence and not an artefact of a fixture whose
    // 64 files are the same bytes.
    let mut seen: Vec<f32> = unbounded.data.clone();
    seen.sort_by(|a, b| a.partial_cmp(b).unwrap());
    seen.dedup();
    assert_eq!(seen.len(), 64, "fixture files are not distinguishable");
}

/// `reader_limit` is what bounds residency, and at `None` nothing moves.
#[test]
fn residency_is_bounded_by_reader_limit() {
    let dir = tempfile::tempdir().unwrap();
    let paths = manifest(dir.path(), 64);
    let plan = one_set_per_file(64);

    let unbounded = open_loader(paths.clone(), None);
    unbounded.gather(&plan).unwrap();
    let m = unbounded.reader_metrics();
    assert_eq!(m.opens.load(std::sync::atomic::Ordering::Relaxed), 64);
    assert_eq!(m.evictions.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(m.hwm.load(std::sync::atomic::Ordering::Relaxed), 64);

    let bounded = open_loader(paths, Some(8));
    bounded.gather(&plan).unwrap();
    let m = bounded.reader_metrics();
    let hwm = m.hwm.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        hwm <= 8,
        "high-water {hwm} exceeded reader_limit 8 with no lease held across files"
    );
    // The manifest is 8x the limit and the plan visits every file, so the
    // registry must have reopened: 64 opens' worth of work for 8 slots.
    assert!(
        m.opens.load(std::sync::atomic::Ordering::Relaxed) >= 64,
        "a plan over 64 files at reader_limit=8 cannot have opened fewer than 64 times"
    );
    assert!(m.evictions.load(std::sync::atomic::Ordering::Relaxed) > 0);
}

/// **Residency on the path consumers actually run.**
///
/// Every other residency assertion in this file drives `gather`, the
/// synchronous one-plan entry point. Production drives `iter_with_plans`, which
/// goes through `PlanPrefetchIter::spawn_prefetches` — a different function
/// with its own leases. Review on #536 (Cursor Agent) pointed out that the
/// gather-path tests would keep reporting the cap as held while the iterator
/// blew through it.
///
/// **The plans here are WIDE, and that is the whole point.** The first version
/// of this test used one file per plan, which passes with the defect still in
/// place — a "bulk lease" of one file is not a spike. Watched: with the
/// pre-fix order restored (bulk-lease + eligibility map before the
/// `lookahead == 0` early return) this fails at 32 against a ceiling of 5,
/// and the narrow version does not notice at all.
#[test]
fn iterator_at_lookahead_zero_does_not_lease_the_plans_width() {
    let dir = tempfile::tempdir().unwrap();
    let paths = manifest(dir.path(), 32);
    let loader = open_loader(paths, Some(4));
    let metrics = loader.reader_metrics();

    // Two plans, each touching all 32 files. At `lookahead == 0` there is no
    // prefetch to launch, so nothing should need more than the sizing and the
    // gather — both of which handle one file at a time.
    let plans: Vec<_> = (0..2).map(|_| Ok(one_set_per_file(32))).collect();
    let iter = Arc::clone(&loader).iter_with_plans(plans.into_iter(), 0);
    let iter_metrics = iter.iter_metrics();
    let n = iter.count();
    assert_eq!(n, 2);

    // The counter, not just the residency. At `lookahead == 0` the declined
    // path and the no-prefetch path now do the *same* sizing work, so residency
    // and open counts cannot tell them apart — only this can. Leaving it
    // unasserted is what let the decline sit unreachable behind the
    // `lookahead == 0` early return for a whole round. **Review on #536
    // (Antigravity).**
    assert_eq!(
        iter_metrics
            .prefetch_skipped_reader_limit
            .load(std::sync::atomic::Ordering::Relaxed),
        2,
        "both 32-file plans exceed reader_limit=4 and must be counted as \
         declined, whatever the lookahead"
    );

    let hwm = metrics.hwm.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        hwm <= 5,
        "a 32-file plan at reader_limit=4, lookahead=0 held {hwm} readers; the \
         prefetcher has no tasks to launch and must not lease the plan's width"
    );
    // `opens` as well as `hwm`, because they fail independently. Asserting
    // residency alone is how the `lookahead == 0` decline stayed unreachable
    // through a whole review round: the bulk lease was gone, so `hwm` looked
    // right, while the plan still paid a sizing pass nothing counted.
    // **Review on #536 (Antigravity).**
    let opens = metrics.opens.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        opens <= 2 * (2 * 32) + 8,
        "{opens} opens for two 32-file plans at reader_limit=4, lookahead=0: the \
         budget is one sizing pass and one gather pass per plan, and a third \
         pass means the prefetcher leased the width it has no tasks for"
    );
}

/// The same iterator with prefetch on: narrow plans stay inside the cap.
///
/// Separate from the wide case because the two bound different things. Here the
/// in-flight lookahead window is the only thing that can raise residency, and
/// each plan touches one file, so the ceiling is the cap plus the window.
#[test]
fn iterator_residency_is_bounded_on_narrow_plans() {
    for lookahead in [0usize, 4] {
        let dir = tempfile::tempdir().unwrap();
        let paths = manifest(dir.path(), 64);
        let loader = open_loader(paths, Some(8));
        let plans: Vec<_> = (0..64u32)
            .map(|f| {
                Ok(SparseCellSetPlan {
                    file_ids: vec![f],
                    rows: vec![0],
                    role_tags: vec![0],
                    set_offsets: vec![0, 1],
                })
            })
            .collect();
        let metrics = loader.reader_metrics();
        let n = Arc::clone(&loader)
            .iter_with_plans(plans.into_iter(), lookahead)
            .count();
        assert_eq!(n, 64, "lookahead={lookahead}");

        let hwm = metrics.hwm.load(std::sync::atomic::Ordering::Relaxed);
        let ceiling = 8 + lookahead as u64 + 1;
        assert!(
            hwm <= ceiling,
            "lookahead={lookahead}: residency high-water {hwm} exceeded {ceiling} \
             on 64 single-file plans at reader_limit=8"
        );
    }
}

/// A plan wider than the limit completes, and the synchronous gather stays
/// inside the cap while doing it.
///
/// Renamed from `a_plan_wider_than_the_limit_exceeds_it_rather_than_blocking`,
/// which is not what it checked. `gather` sizes and reads one file at a time,
/// so residency here is *naturally* bounded — the old name and doc claimed it
/// pinned the soft cap giving way, and the assertion said the opposite.
/// **Review on #536 (Cursor Agent, codex, Antigravity — all three.)** The
/// contract it was supposed to pin is now in
/// `a_wide_plan_declines_prefetch_rather_than_exceeding_the_limit`.
#[test]
fn a_wide_plan_gathers_without_exceeding_the_cap() {
    let dir = tempfile::tempdir().unwrap();
    let paths = manifest(dir.path(), 32);
    let loader = open_loader(paths, Some(4));
    let plan = one_set_per_file(32);

    let batch = loader.gather(&plan).expect("a wide plan must not deadlock");
    assert_eq!(batch.cell_indices.len(), 32);
    let hwm = loader
        .reader_metrics()
        .hwm
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        hwm <= 4 + 1,
        "the synchronous gather sizes and reads one file at a time, so even a \
         32-file plan should not need more than the cap plus one; got {hwm}"
    );
}

/// **The production-path bound, with prefetch on.**
///
/// A plan touching more distinct files than `reader_limit` cannot be
/// prefetched: the prefetcher holds a lease per launched file, so it would pin
/// the plan's width and the cap would stop meaning anything. It declines
/// instead, and the gather reads one file at a time.
///
/// codex - gpt-5.6-sol measured the unfixed behaviour at head `d5edc405`:
/// one 32-file plan at `reader_limit=2, lookahead=4` moved `opens/hwm` from
/// `2/2` to `65/32`. Both numbers are asserted here, because either alone
/// misses half the defect — residency was 16x the cap AND the catalogs were
/// reparsed.
#[test]
fn a_wide_plan_declines_prefetch_rather_than_exceeding_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let paths = manifest(dir.path(), 32);
    let loader = open_loader_lookahead(paths, Some(2), 4);
    let metrics = loader.reader_metrics();

    let plans: Vec<_> = (0..2).map(|_| Ok(one_set_per_file(32))).collect();
    let n = Arc::clone(&loader)
        .iter_with_plans(plans.into_iter(), 4)
        .count();
    assert_eq!(n, 2);

    let hwm = metrics.hwm.load(std::sync::atomic::Ordering::Relaxed);
    let opens = metrics.opens.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        hwm <= 3,
        "residency high-water {hwm} against reader_limit=2: the prefetcher must \
         decline a plan it cannot hold, not pin its width"
    );
    // Two plans over 32 files at a cap of 2 reopen a lot — that is the cost of
    // the shape and it is the caller's to avoid. The budget is one sizing pass
    // plus one gather pass per plan; the defect this pins is the *third*, from
    // the bulk lease, which codex measured at 65 opens for a single plan.
    //
    // The sizing pass is deliberately kept rather than skipped: it is what
    // produces a real admission verdict, and a plan whose readers cannot all
    // stay resident can still have its row groups fit the byte budget — the
    // two live in different places and `CacheKey::Group` outlives a handle
    // eviction. Trading that for a lower open count was the wrong trade.
    assert!(
        opens <= 2 * 32 * 2 + 4,
        "{opens} opens for two 32-file plans at reader_limit=2; more than a \
         sizing pass and a gather pass each means something is reopening what \
         it just closed"
    );
}

/// **A relative manifest path survives a `chdir`.**
///
/// Before this PR every mmap was established in the constructor and held, so
/// the process's cwd could not affect a later batch. A registry that evicts
/// reopens by path, so storing the caller's spelling made a bounded dataset
/// break on `set_current_dir` — a regression against `main` that only bounded
/// mode has. Reproduced on #536 by codex - gpt-5.6-sol and confirmed by Cursor
/// Agent and Antigravity; the scan now stores `std::path::absolute`.
///
/// `set_current_dir` is process-wide and `cargo test` runs these in parallel,
/// so this is only safe because every other test in this binary opens absolute
/// tempdir paths and is therefore immune to the cwd moving. It is restored
/// before the assertion. Do not add a test here that resolves a relative path
/// without reading this first.
#[test]
fn a_relative_manifest_path_survives_a_chdir() {
    let dir = tempfile::tempdir().unwrap();
    let paths = manifest(dir.path(), 8);
    let names: Vec<PathBuf> = paths
        .iter()
        .map(|p| PathBuf::from(p.file_name().unwrap()))
        .collect();

    // Relative spellings, resolved against the fixture dir at construction.
    let prev = std::env::current_dir().unwrap();
    std::env::set_current_dir(dir.path()).unwrap();
    let loader = open_loader(names, Some(2));
    // Somewhere the relative names cannot resolve.
    let elsewhere = tempfile::tempdir().unwrap();
    std::env::set_current_dir(elsewhere.path()).unwrap();

    let got = loader.gather(&one_set_per_file(8));
    std::env::set_current_dir(prev).unwrap();

    let batch = got.expect("a bounded reopen must not depend on the process cwd");
    assert_eq!(batch.cell_indices.len(), 8);
}

/// A file replaced between gathers is refused on reopen rather than served.
///
/// Mutation (applied, seen to fail): drop the `FileIdentity::ensure_same` call
/// in `ReaderRegistry::lease`. The replaced file is then read as though it were
/// the one that was scanned, and the gather returns the new file's bytes under
/// the old file's `file_id`.
#[test]
fn a_file_replaced_between_gathers_is_refused_on_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let paths = manifest(dir.path(), 16);
    let loader = open_loader(paths.clone(), Some(2));
    let plan = one_set_per_file(16);
    loader.gather(&plan).unwrap();

    // Rewrite file 0 in place with different content. `reader_limit = 2` over a
    // 16-file plan guarantees its handle was evicted, so the next gather
    // reopens it.
    write_tagged(&paths[0], 200);

    let err = loader
        .gather(&plan)
        .expect_err("a replaced file must be refused, not served");
    let msg = err.to_string();
    assert!(
        msg.contains("changed on disk") || msg.contains("replaced on disk"),
        "expected a file-changed error, got: {msg}"
    );
    assert!(msg.contains("f0.scx"), "error must name the file: {msg}");
}

/// An unbounded registry never reopens, so it cannot notice a replacement —
/// the companion to the test above, pinning that the default path is unchanged
/// rather than newly strict.
#[test]
fn an_unbounded_registry_keeps_serving_a_replaced_file_from_its_mapping() {
    let dir = tempfile::tempdir().unwrap();
    let paths = manifest(dir.path(), 4);
    let loader = open_loader(paths.clone(), None);
    let plan = one_set_per_file(4);
    let before = loader.gather(&plan).unwrap();

    write_tagged(&paths[0], 200);

    let after = loader
        .gather(&plan)
        .expect("an unbounded registry holds the original mapping and reads on");
    assert_eq!(
        fields(&before),
        fields(&after),
        "the default path must behave exactly as it did before reader_limit existed"
    );
}
