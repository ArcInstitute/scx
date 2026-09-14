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

#[allow(clippy::too_many_arguments)]
fn open_loader(paths: Vec<PathBuf>, reader_limit: Option<usize>) -> Arc<SparseCellSetLoader> {
    SparseCellSetLoader::open(
        paths,
        /* cache_shards */ 4,
        /* bytes_budget */ Some(1 << 20),
        /* lookahead */ 0,
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
