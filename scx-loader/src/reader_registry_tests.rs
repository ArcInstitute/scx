use super::*;

use std::sync::atomic::Ordering;
use std::sync::Arc as StdArc;

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;

/// The smallest `.scx` that opens: four rows, one non-zero each, two shards.
/// Content is irrelevant here — these tests are about handle lifetime, and the
/// byte-level claims live in `tests/test_reader_registry.rs`, whose fixture
/// tags each file distinctly.
fn write_tiny(path: &std::path::Path) {
    let (n_obs, n_vars, n_shards) = (4usize, 4usize, 2usize);
    let rows_per_shard = n_obs / n_shards;
    let header = FileHeader::new_single_modality(
        n_obs as u64,
        n_vars as u64,
        n_obs as u64,
        rows_per_shard as u32,
        0,
        0,
    );
    let mut writer = ScxWriter::new(path, header).unwrap();
    let names = |n: usize, p: &str| -> Vec<String> { (0..n).map(|i| format!("{p}{i}")).collect() };
    let batch = |col: &str, v: &[String]| {
        arrow::record_batch::RecordBatch::try_new(
            StdArc::new(Schema::new(vec![Field::new(col, DataType::Utf8, false)])),
            vec![StdArc::new(StringArray::from(
                v.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    };
    writer
        .write_obs(&batch("cell_id", &names(n_obs, "c")))
        .unwrap();
    writer
        .write_var(&batch("gene_id", &names(n_vars, "g")))
        .unwrap();
    for s in 0..n_shards {
        let row_start = s * rows_per_shard;
        let indptr: Vec<u64> = (0..=rows_per_shard as u64).collect();
        let indices: Vec<u32> = (0..rows_per_shard)
            .map(|l| ((row_start + l) % n_vars) as u32)
            .collect();
        let values: Vec<u8> = vec![1u8; rows_per_shard];
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

/// A bounded registry over `n` copies of one tiny file.
fn bounded_registry(dir: &std::path::Path, n: usize, limit: usize) -> StdArc<ReaderRegistry> {
    let shared = SharedShardCache::new(4, 1 << 20);
    let mut slots = Vec::new();
    let mut retained = Vec::new();
    for i in 0..n {
        let path = dir.join(format!("f{i}.scx"));
        write_tiny(&path);
        let reader = ScxReader::open(&path).unwrap();
        slots.push(FileSlot {
            path: path.clone(),
            n_obs: reader.n_obs(),
            index: BackedCsrIndex::from_catalog(reader.catalog()),
            identity: Some(FileIdentity::of(&reader)),
        });
        if i < limit {
            retained.push((i as u32, wrap_reader(reader, i as u32, &shared, false)));
        }
    }
    ReaderRegistry::from_scan(slots, retained, Some(limit), shared, false, None)
}

/// **The safety property the design rests on.** A handle with an outstanding
/// lease must not be evicted, whatever `reader_limit` says: a read holds its
/// receiver across a parallel decode and a single-flight wait, and an
/// already-started `spawn_blocking` prefetch cannot be aborted, so unmapping a
/// leased file is a use-after-free of its mmap.
///
/// `reader_limit` therefore bounds *idle-retainable* handles, not live ones —
/// this asserts that shape directly rather than asserting a cap it does not
/// have.
///
/// Mutation applied and seen to fail exactly this test: drop the
/// `Arc::strong_count(&h.reader) == 1` filter in `evict_down_to`.
#[test]
fn a_leased_handle_is_never_evicted() {
    let dir = tempfile::tempdir().unwrap();
    let reg = bounded_registry(dir.path(), 16, 2);

    let leases: Vec<_> = (0..16u32).map(|f| reg.lease(f).unwrap()).collect();
    assert_eq!(
        reg.metrics().resident.load(Ordering::Relaxed),
        16,
        "16 live leases must all still be resident despite a limit of 2"
    );
    // Distinct files, so the residency count means what it says.
    let mut ptrs: Vec<usize> = leases.iter().map(|a| StdArc::as_ptr(a) as usize).collect();
    ptrs.sort_unstable();
    ptrs.dedup();
    assert_eq!(ptrs.len(), 16);

    // Released, the registry is free to come back under the cap on the next miss.
    drop(leases);
    reg.lease(0).unwrap();
    let resident = reg.metrics().resident.load(Ordering::Relaxed);
    assert!(
        resident <= 2,
        "with every lease dropped the next miss must evict back to the limit, got {resident}"
    );
}

/// A lease that is still held is skipped, and an idle one is taken instead —
/// the accept-side of the test above, which on its own would pass on a registry
/// that simply never evicted anything.
#[test]
fn eviction_picks_an_idle_handle_over_a_leased_one() {
    let dir = tempfile::tempdir().unwrap();
    let reg = bounded_registry(dir.path(), 4, 2);

    // Slot 0 pinned, slot 1 idle and least-recently-used.
    let pinned = reg.lease(0).unwrap();
    let _ = reg.lease(1).unwrap();
    let before = reg.metrics().evictions.load(Ordering::Relaxed);

    // A miss on slot 2 must evict slot 1, not slot 0.
    let _ = reg.lease(2).unwrap();
    assert_eq!(reg.metrics().evictions.load(Ordering::Relaxed), before + 1);
    assert!(
        StdArc::ptr_eq(&pinned, &reg.lease(0).unwrap()),
        "the pinned handle was replaced, so it had been evicted while leased"
    );
}

/// An unbounded registry seeded from open readers has no reopen recipe, so a
/// lease for a slot it does not hold is a programming error rather than a
/// silent reopen. Unreachable through the loader (it retains everything), and
/// pinned so it stays that way.
#[test]
fn an_unbounded_registry_refuses_a_lease_it_cannot_serve() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f.scx");
    write_tiny(&path);
    let shared = SharedShardCache::new(4, 1 << 20);
    let reader = wrap_reader(ScxReader::open(&path).unwrap(), 0, &shared, false);
    let reg = ReaderRegistry::from_open(vec![reader]);

    assert!(reg.lease(0).is_ok());
    let err = match reg.lease(1) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("file_id 1 does not exist in a one-file registry"),
    };
    assert!(err.contains("out of range"), "{err}");
}

/// A reopening registry whose slot carries no identity refuses rather than
/// serving the file unverified.
///
/// Unreachable through either constructor — the manifest scan stamps every slot
/// it hands to a registry that can reopen — and pinned precisely so it stays
/// unreachable. Falling through instead of refusing would make a future
/// constructor that forgot to stamp serve a replaced file silently, which is
/// the failure the identity check exists for.
#[test]
fn a_reopenable_slot_without_an_identity_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f.scx");
    write_tiny(&path);
    let shared = SharedShardCache::new(4, 1 << 20);
    let reader = ScxReader::open(&path).unwrap();
    let slots = vec![FileSlot {
        path: path.clone(),
        n_obs: reader.n_obs(),
        index: BackedCsrIndex::from_catalog(reader.catalog()),
        identity: None,
    }];
    // Retain nothing, so the first lease must go through the reopen path.
    let reg = ReaderRegistry::from_scan(slots, Vec::new(), Some(1), shared, false, None);

    let err = match reg.lease(0) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("an unverifiable reopen must be refused, not served"),
    };
    assert!(err.contains("carries no identity"), "{err}");
}

/// Residency is bounded when nothing is leased across calls — the plain
/// sequential-access case, which is what a plan-per-file iteration looks like.
#[test]
fn sequential_leases_stay_within_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let reg = bounded_registry(dir.path(), 32, 4);
    for f in 0..32u32 {
        let _ = reg.lease(f).unwrap();
    }
    let m = reg.metrics();
    assert!(m.hwm.load(Ordering::Relaxed) <= 4);
    assert_eq!(m.opens.load(Ordering::Relaxed), 32);
    assert_eq!(m.evictions.load(Ordering::Relaxed), 32 - 4);
}
