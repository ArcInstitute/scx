//! Gate tests for the standalone `scx sort` engine. Pure SCX; built on the
//! fixtures in `crate::test_utils`.

use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::path::Path;

use std::sync::Arc;

use arrow::array::{Array, DictionaryArray, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Int8Type, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_engine::index::PredicateIndex;
use scx_format_io::header::FileHeader;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;
use scx_format_io::{BitmapPolicy, ScxReader};

use super::{sort, sort_with_strategy};
use crate::sort::{ReferenceSpec, SortOptions, SortStrategy};
use crate::test_utils::{
    fixture_composite, fixture_deletion, fixture_multimodal, fixture_null_key, fixture_numeric,
    fixture_obsp_layers, fixture_plain, fixture_skewed,
};

// --- helpers ---------------------------------------------------------------

fn opts(by: &[&str]) -> SortOptions {
    SortOptions {
        by: by.iter().map(|s| s.to_string()).collect(),
        // Small shards so tiny fixtures still exercise multi-shard re-sharding.
        shard_target_rows: 2,
        ..Default::default()
    }
}

/// (cell_ids, per-row sparse `(col, value)`) in output row order.
fn content(path: &Path) -> (Vec<String>, Vec<Vec<(i32, f32)>>) {
    let r = ScxReader::open(path).unwrap();
    let ids = str_col(&r.read_obs().unwrap(), "cell_id");
    let csr = r.read_all_csr_shards().unwrap();
    let mut rows = Vec::new();
    for i in 0..csr.shape.0 {
        let s = csr.indptr[i] as usize;
        let e = csr.indptr[i + 1] as usize;
        rows.push((s..e).map(|j| (csr.indices[j], csr.data[j])).collect());
    }
    (ids, rows)
}

fn str_col(batch: &arrow::array::RecordBatch, name: &str) -> Vec<String> {
    let col = batch.column_by_name(name).unwrap();
    let utf8 = arrow::compute::cast(col, &DataType::Utf8).unwrap();
    let arr = utf8.as_any().downcast_ref::<StringArray>().unwrap();
    (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
}

fn col_of(path: &Path, name: &str) -> Vec<String> {
    str_col(&ScxReader::open(path).unwrap().read_obs().unwrap(), name)
}

fn is_sorted_asc(v: &[String]) -> bool {
    v.windows(2).all(|w| w[0] <= w[1])
}

// --- T4.1/round-trip: in-memory default ------------------------------------

#[test]
fn round_trip_equivalence_categorical() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    let summary = sort(&inp, &out, &opts(&["cell_type"])).unwrap();

    // No budget set → in-memory fast path.
    assert_eq!(summary.strategy, SortStrategy::InMemory);
    assert!(summary.indexed_columns.iter().any(|c| c == "cell_type"));

    let (in_ids, in_rows) = content(&inp);
    let (out_ids, out_rows) = content(&out);
    assert_eq!(in_ids.len(), out_ids.len());

    // Same set of cells, X content preserved per cell.
    let in_map: HashMap<&String, &Vec<(i32, f32)>> = in_ids.iter().zip(&in_rows).collect();
    for (id, row) in out_ids.iter().zip(&out_rows) {
        assert_eq!(in_map[id], row, "X row for {id} must survive the reorder");
    }
    assert!(is_sorted_asc(&col_of(&out, "cell_type")));
}

// --- Strategy differential: a/b/c byte-identical (content) -----------------

#[test]
fn strategy_differential_identical() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let mut o = opts(&["cell_type"]);
    o.memory_budget = Some(64); // forces the external path to actually partition

    let mut results = Vec::new();
    for strat in [
        SortStrategy::InMemory,
        SortStrategy::KPassByCategory,
        SortStrategy::ExternalPartition,
    ] {
        let out = dir.path().join(format!("out_{strat:?}.scx"));
        let summary = sort_with_strategy(&inp, &out, &o, Some(strat)).unwrap();
        assert_eq!(summary.strategy, strat);
        results.push(content(&out));
    }
    assert_eq!(results[0], results[1], "in-memory vs K-pass must match");
    assert_eq!(results[1], results[2], "K-pass vs external must match");
}

// --- Part 1 OOM fix: sharded-obs input via projected key-only pass 0 -------

/// `scx sort` always writes obs via `write_obs_sharded`, so sorting a fixture
/// once yields a multi-`ObsMetadataShard` file. Re-sorting that file drives
/// pass 0 through `read_obs_keys`'s sharded assembly path (the atlas-scale
/// path the OOM fix targets); the result must be byte-identical across the
/// in-memory and external (spill) strategies and correctly ordered.
#[test]
fn sharded_obs_input_sorts_identically() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);

    // First sort → sharded-obs .scx (12 obs / shard_target_rows 2 = 6 shards).
    let sharded = dir.path().join("sharded.scx");
    sort(&inp, &sharded, &opts(&["cell_type"])).unwrap();
    let shard_count = ScxReader::open(&sharded)
        .unwrap()
        .obs_metadata_shard_count();
    assert!(
        shard_count >= 2,
        "expected a multi-shard obs input, got {shard_count}"
    );

    // Re-sort the sharded-obs file under both strategies; the projected
    // key-only pass-0 read must yield identical output. A large budget keeps
    // obs on the in-memory path (Part 2's obs spill is exercised separately)
    // while still letting a forced external X strategy run.
    let mut o = opts(&["cell_type"]);
    o.memory_budget = Some(1 << 30);
    let in_mem = dir.path().join("in_mem.scx");
    let external = dir.path().join("external.scx");
    sort_with_strategy(&sharded, &in_mem, &o, Some(SortStrategy::InMemory)).unwrap();
    sort_with_strategy(
        &sharded,
        &external,
        &o,
        Some(SortStrategy::ExternalPartition),
    )
    .unwrap();

    assert_eq!(
        content(&in_mem),
        content(&external),
        "sharded-obs input: in-memory vs external must match"
    );
    assert!(is_sorted_asc(&col_of(&in_mem, "cell_type")));
}

/// obs bytes-per-row from a sharded-obs file's first shard (for budget
/// calibration).
fn obs_bytes_per_row(path: &Path) -> u64 {
    let s0 = ScxReader::open(path).unwrap().read_obs_shard(0).unwrap();
    let bytes: usize = s0.columns().iter().map(|c| c.get_array_memory_size()).sum();
    (bytes / s0.num_rows().max(1)) as u64
}

/// The obs spill-scatter path must produce output logically identical to the
/// in-memory `take` path. X is forced to `InMemory` for both so only the obs
/// write strategy differs; the budget is calibrated to hold ~3 rows/partition
/// (≥ one shard, < all rows) so partitions don't align to `shard_target` —
/// exercising cross-partition shard chunking.
#[test]
fn obs_spill_matches_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let sharded = dir.path().join("sharded.scx");
    sort(&inp, &sharded, &opts(&["cell_type"])).unwrap();
    assert!(
        ScxReader::open(&sharded)
            .unwrap()
            .obs_metadata_shard_count()
            >= 2
    );

    let bpr = obs_bytes_per_row(&sharded);

    // In-memory baseline (no budget → obs in-memory).
    let mem = dir.path().join("mem.scx");
    let mem_s = sort_with_strategy(
        &sharded,
        &mem,
        &opts(&["cell_type"]),
        Some(SortStrategy::InMemory),
    )
    .unwrap();
    assert!(!mem_s.obs_spilled, "no budget must keep obs in memory");

    // Spill (budget holds ~3 rows/partition; shard_target is 2 → shards span
    // partitions). X forced InMemory so only the obs strategy differs.
    let mut o = opts(&["cell_type"]);
    o.memory_budget = Some(bpr * 3);
    let spilled = dir.path().join("spilled.scx");
    let sp_s = sort_with_strategy(&sharded, &spilled, &o, Some(SortStrategy::InMemory)).unwrap();
    assert!(
        sp_s.obs_spilled,
        "budget {} should force obs spill",
        bpr * 3
    );
    assert!(
        sp_s.obs_partitions >= 2,
        "expected multiple obs partitions, got {}",
        sp_s.obs_partitions
    );

    assert_eq!(
        content(&mem),
        content(&spilled),
        "obs spill vs in-memory content must match"
    );
    assert!(is_sorted_asc(&col_of(&spilled, "cell_type")));
}

/// Build a sharded-obs `.scx` whose `cell_type` is a `Dictionary` column with a
/// distinct local vocabulary per shard (exercises decode→re-encode + the
/// reader's cross-shard unify). 9 obs across 3 shards, one CSR shard.
fn write_dict_obs_fixture(dir: &tempfile::TempDir) -> std::path::PathBuf {
    let path = dir.path().join("dict_obs.scx");
    let (n_obs, n_vars) = (9usize, 4usize);
    let header =
        FileHeader::new_single_modality(n_obs as u64, n_vars as u64, (n_obs * 2) as u64, 3, 0, 0);
    let mut w = ScxWriter::new(&path, header).unwrap();

    let mut indptr = vec![0u64];
    let (mut indices, mut values) = (Vec::new(), Vec::new());
    for r in 0..n_obs {
        indices.push(((r * 2) % n_vars) as u32);
        indices.push(((r * 2 + 1) % n_vars) as u32);
        values.push(((r + 1) % 256) as u8);
        values.push(((r + 2) % 256) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    w.write_csr_shard(
        &indptr,
        &indices,
        &values,
        CodecId::None,
        ValueEncoding::Uint8,
        0,
    )
    .unwrap();
    let var = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "gene_id",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(
            (0..n_vars).map(|i| format!("g{i}")).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    w.write_var(&var).unwrap();

    let cts = [["b", "a", "b"], ["c", "a", "b"], ["a", "c", "a"]];
    let obs_schema = Arc::new(Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new(
            "cell_type",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            false,
        ),
    ]));
    for (si, types) in cts.iter().enumerate() {
        let rs = si * 3;
        let ids: Vec<String> = (rs..rs + 3).map(|i| format!("cell_{i}")).collect();
        let dict: DictionaryArray<Int8Type> = types.iter().copied().map(Some).collect();
        let batch = RecordBatch::try_new(
            obs_schema.clone(),
            vec![
                Arc::new(StringArray::from(
                    ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                )),
                Arc::new(dict),
            ],
        )
        .unwrap();
        w.write_obs_shard(si as u32, rs as u64, 3, n_obs as u64, &batch)
            .unwrap();
    }
    w.finish().unwrap();
    path
}

/// On a `Dictionary`-typed categorical sort key, the spill path must (a)
/// preserve the categorical dtype on read (re-encode works), and (b) produce
/// content + values identical to the in-memory path.
#[test]
fn obs_spill_preserves_categorical_dtype() {
    let dir = tempfile::tempdir().unwrap();
    let inp = write_dict_obs_fixture(&dir);
    let bpr = obs_bytes_per_row(&inp);

    let mem = dir.path().join("mem.scx");
    sort_with_strategy(
        &inp,
        &mem,
        &opts(&["cell_type"]),
        Some(SortStrategy::InMemory),
    )
    .unwrap();

    let mut o = opts(&["cell_type"]);
    o.memory_budget = Some(bpr * 3);
    let out = dir.path().join("out.scx");
    let s = sort_with_strategy(&inp, &out, &o, Some(SortStrategy::InMemory)).unwrap();
    assert!(s.obs_spilled);
    assert!(s.obs_partitions >= 2);

    let obs = ScxReader::open(&out).unwrap().read_obs().unwrap();
    let ct = obs.column_by_name("cell_type").unwrap();
    assert!(
        matches!(ct.data_type(), DataType::Dictionary(_, _)),
        "spill path must preserve categorical dtype, got {:?}",
        ct.data_type()
    );

    assert_eq!(content(&mem), content(&out), "spill vs in-memory content");
    assert_eq!(col_of(&out, "cell_type"), col_of(&mem, "cell_type"));
    assert!(is_sorted_asc(&col_of(&out, "cell_type")));
}

/// Deletions + obs spill: the scatter skips `new_pos < 0` (deleted) rows, so a
/// sharded-obs input carrying a deletion vector must spill-sort to the same
/// dense, deletion-free output as the in-memory path. Builds the fixture by
/// sorting (→ sharded obs) then marking deletions on that file.
#[test]
fn obs_spill_with_deletions() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let sharded = dir.path().join("sharded.scx");
    sort(&inp, &sharded, &opts(&["cell_type"])).unwrap();
    // Add a deletion vector to the (now sharded-obs) file.
    crate::delete::mark_deleted(&sharded, &[1u64, 4, 7]).unwrap();
    assert!(
        ScxReader::open(&sharded)
            .unwrap()
            .obs_metadata_shard_count()
            >= 2
    );

    let bpr = obs_bytes_per_row(&sharded);

    // In-memory baseline (no budget) vs forced obs spill.
    let mem = dir.path().join("mem.scx");
    let mem_s = sort_with_strategy(
        &sharded,
        &mem,
        &opts(&["cell_type"]),
        Some(SortStrategy::InMemory),
    )
    .unwrap();
    assert!(!mem_s.obs_spilled);

    let mut o = opts(&["cell_type"]);
    o.memory_budget = Some(bpr * 3);
    let spilled = dir.path().join("spilled.scx");
    let sp_s = sort_with_strategy(&sharded, &spilled, &o, Some(SortStrategy::InMemory)).unwrap();
    assert!(sp_s.obs_spilled, "budget should force obs spill");
    assert!(sp_s.obs_partitions >= 2);

    // 3 of 12 deleted → 9 live, identical content on both paths, deletion-free.
    assert_eq!(sp_s.n_obs, 9);
    assert_eq!(mem_s.n_obs, 9);
    assert_eq!(
        content(&mem),
        content(&spilled),
        "deletion + spill must match the in-memory path"
    );
    assert!(is_sorted_asc(&col_of(&spilled, "cell_type")));
    let clean = ScxReader::open(&spilled)
        .unwrap()
        .deletion_keep_mask()
        .unwrap()
        .map(|m| m.iter().all(|&k| k))
        .unwrap_or(true);
    assert!(clean, "spilled output must be deletion-free");
}

// --- Determinism (modulo provenance timestamp) -----------------------------

#[test]
fn deterministic_output() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let a = dir.path().join("a.scx");
    let b = dir.path().join("b.scx");
    sort(&inp, &a, &opts(&["cell_type"])).unwrap();
    sort(&inp, &b, &opts(&["cell_type"])).unwrap();
    assert_eq!(content(&a), content(&b));
    assert_eq!(col_of(&a, "cell_type"), col_of(&b, "cell_type"));
}

// --- Stability under the external scatter path -----------------------------

#[test]
fn stable_tie_order_external() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    let mut o = opts(&["cell_type"]);
    o.memory_budget = Some(64);
    sort_with_strategy(&inp, &out, &o, Some(SortStrategy::ExternalPartition)).unwrap();

    // Input position of each cell.
    let (in_ids, _) = content(&inp);
    let pos: HashMap<&String, usize> = in_ids.iter().enumerate().map(|(i, s)| (s, i)).collect();

    let out_ids = col_of(&out, "cell_id");
    let out_types = col_of(&out, "cell_type");
    // Within each equal-key run, input positions must be strictly increasing
    // (stability: equal keys keep their original global order).
    for w in 0..out_ids.len().saturating_sub(1) {
        if out_types[w] == out_types[w + 1] {
            assert!(
                pos[&out_ids[w]] < pos[&out_ids[w + 1]],
                "tie order broke at output rows {w}/{}",
                w + 1
            );
        }
    }
}

// --- Skew: bounded partitions, no sub-split needed (new_pos design) --------

#[test]
fn skew_partitions_bounded() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_skewed(&dir);
    let out = dir.path().join("out.scx");
    let mut o = opts(&["cell_type"]);
    o.memory_budget = Some(64); // P == 2 rows → 10 partitions over 20 rows

    let summary =
        sort_with_strategy(&inp, &out, &o, Some(SortStrategy::ExternalPartition)).unwrap();
    assert!(summary.partitions > 1, "skew must span multiple partitions");
    assert!(summary.spill_bytes > 0, "external path must spill");
    // new_pos-range partitions are inherently balanced regardless of the
    // dominant category: 20 rows / 2-per-partition.
    assert_eq!(summary.partitions, 10);
    assert!(is_sorted_asc(&col_of(&out, "cell_type")));
}

// --- Spill-size telemetry + refuse-on-overflow (T4.7) ----------------------

#[test]
fn spill_size_matches_formula() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    let mut o = opts(&["cell_type"]);
    o.memory_budget = Some(64);
    let summary =
        sort_with_strategy(&inp, &out, &o, Some(SortStrategy::ExternalPartition)).unwrap();
    // 12 rows × (8 new_pos + 4 nnz + 2 nnz × 8 bytes) = 12 × 28.
    assert_eq!(summary.spill_bytes, 12 * 28);
}

#[test]
fn external_refuses_when_budget_below_one_shard() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    let mut o = opts(&["cell_type"]);
    // per-shard bytes = 2 rows × 8 vars × 0.25 density × 16 = 64; budget below it.
    o.memory_budget = Some(32);
    let err = sort_with_strategy(&inp, &out, &o, Some(SortStrategy::ExternalPartition));
    assert!(err.is_err(), "must refuse a budget too small for one shard");
}

// --- Index correctness: contiguous shard ranges ----------------------------

#[test]
fn index_ranges_contiguous() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    sort(&inp, &out, &opts(&["cell_type"])).unwrap();

    let r = ScxReader::open(&out).unwrap();
    let bytes = r
        .read_obs_predicate_index_bytes()
        .unwrap()
        .expect("sort must (re)build the obs predicate index for the sort key");
    let index = PredicateIndex::read_from(&mut Cursor::new(bytes)).unwrap();

    let ranges = index
        .categorical_eq("cell_type", "B cell")
        .expect("cell_type must be an indexed categorical");
    assert!(!ranges.is_empty(), "B cell must occupy some shard range");
    // After sort, the category is a contiguous block → its shard ids form a
    // consecutive run.
    let mut shard_ids: Vec<u32> = ranges.iter().map(|r| r.shard_id).collect();
    shard_ids.sort_unstable();
    shard_ids.dedup();
    for w in shard_ids.windows(2) {
        assert_eq!(w[1], w[0] + 1, "B cell shard ids must be contiguous");
    }
}

// --- Composite + reverse + numeric keys ------------------------------------

#[test]
fn composite_key_lexicographic() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_composite(&dir);
    let out = dir.path().join("out.scx");
    sort(&inp, &out, &opts(&["cell_type", "donor"])).unwrap();

    let types = col_of(&out, "cell_type");
    let donors = col_of(&out, "donor");
    assert!(is_sorted_asc(&types));
    // Within each cell_type block, donor is non-decreasing.
    for w in 0..types.len().saturating_sub(1) {
        if types[w] == types[w + 1] {
            assert!(donors[w] <= donors[w + 1], "donor order within a cell_type");
        }
    }
}

#[test]
fn reverse_key_descending() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    let mut o = opts(&["cell_type"]);
    o.reverse = true;
    sort(&inp, &out, &o).unwrap();
    let types = col_of(&out, "cell_type");
    assert!(
        types.windows(2).all(|w| w[0] >= w[1]),
        "reverse → descending"
    );
}

#[test]
fn numeric_key_ascending() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_numeric(&dir);
    let out = dir.path().join("out.scx");
    sort(&inp, &out, &opts(&["n_genes"])).unwrap();

    let r = ScxReader::open(&out).unwrap();
    let obs = r.read_obs().unwrap();
    let col = obs.column_by_name("n_genes").unwrap();
    let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
    let vals: Vec<i64> = (0..arr.len()).map(|i| arr.value(i)).collect();
    assert!(vals.windows(2).all(|w| w[0] <= w[1]), "numeric ascending");
}

// --- Deletion vector input → dense, deletion-free --------------------------

#[test]
fn deletion_input_materialized_away() {
    let dir = tempfile::tempdir().unwrap();
    let (inp, n_deleted) = fixture_deletion(&dir);
    let out = dir.path().join("out.scx");
    let summary = sort(&inp, &out, &opts(&["cell_type"])).unwrap();

    assert_eq!(summary.n_obs, (12 - n_deleted) as u64);

    let r = ScxReader::open(&out).unwrap();
    // Output carries no live deletions.
    let clean = r
        .deletion_keep_mask()
        .unwrap()
        .map(|m| m.iter().all(|&k| k))
        .unwrap_or(true);
    assert!(clean, "sorted output must be deletion-free");

    // Deleted cells (rows 1,3,5) are gone; everyone else survives.
    let out_ids: HashSet<String> = col_of(&out, "cell_id").into_iter().collect();
    let expected: HashSet<String> = (0..12)
        .filter(|i| ![1usize, 3, 5].contains(i))
        .map(|i| format!("cell_{i}"))
        .collect();
    assert_eq!(out_ids, expected);
    assert!(is_sorted_asc(&col_of(&out, "cell_type")));
}

// ===========================================================================
// Multimodal, obsp remap, bitmap rebuild
// ===========================================================================

fn cell_idx(id: &str) -> usize {
    id.strip_prefix("cell_").unwrap().parse().unwrap()
}

/// `old global row -> new position` from the output's cell_id order (`-1` if a
/// row is absent, e.g. deleted).
fn new_pos_map(out_ids: &[String], n_obs: usize) -> Vec<i64> {
    let mut m = vec![-1i64; n_obs];
    for (new, id) in out_ids.iter().enumerate() {
        m[cell_idx(id)] = new as i64;
    }
    m
}

fn csr_row(indptr: &[i64], indices: &[i32], data: &[f32], i: usize) -> Vec<(i32, f32)> {
    let s = indptr[i] as usize;
    let e = indptr[i + 1] as usize;
    (s..e).map(|j| (indices[j], data[j])).collect()
}

/// COO edges `(row, col, data)` as i64/i64/f32 regardless of on-disk width.
fn obsp_edges(b: &RecordBatch) -> Vec<(i64, i64, f32)> {
    use arrow::array::Float32Array;
    let row = arrow::compute::cast(b.column_by_name("row").unwrap(), &DataType::Int64).unwrap();
    let col = arrow::compute::cast(b.column_by_name("col").unwrap(), &DataType::Int64).unwrap();
    let data = arrow::compute::cast(b.column_by_name("data").unwrap(), &DataType::Float32).unwrap();
    let row = row.as_any().downcast_ref::<Int64Array>().unwrap();
    let col = col.as_any().downcast_ref::<Int64Array>().unwrap();
    let data = data.as_any().downcast_ref::<Float32Array>().unwrap();
    (0..b.num_rows())
        .map(|i| (row.value(i), col.value(i), data.value(i)))
        .collect()
}

fn obsp_dim(b: &RecordBatch, key: &str) -> i64 {
    b.schema().metadata().get(key).unwrap().parse().unwrap()
}

// --- T5.1 multimodal -------------------------------------------------------

#[test]
fn multimodal_reorders_every_modality() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_multimodal(&dir); // 12 obs, rna(8) + adt(4), shared cell_type
    let out = dir.path().join("out.scx");
    let summary = sort(&inp, &out, &opts(&["cell_type"])).unwrap();
    assert_eq!(summary.strategy, SortStrategy::InMemory);

    let ri = ScxReader::open(&inp).unwrap();
    let ro = ScxReader::open(&out).unwrap();
    assert!(ro.is_multimodal());
    assert_eq!(ro.n_modalities(), 2);
    let names = ro.modality_names();
    assert!(names.contains(&"rna") && names.contains(&"adt"));
    assert!(is_sorted_asc(&col_of(&out, "cell_type")));

    let in_ids = str_col(&ri.read_obs().unwrap(), "cell_id");
    let out_ids = str_col(&ro.read_obs().unwrap(), "cell_id");
    for mid in [1u8, 2] {
        let ci = ri.read_all_csr_shards_for(mid).unwrap();
        let co = ro.read_all_csr_shards_for(mid).unwrap();
        assert_eq!(ci.shape.1, co.shape.1, "modality {mid} n_vars preserved");
        let in_map: HashMap<&String, Vec<(i32, f32)>> = in_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id, csr_row(&ci.indptr, &ci.indices, &ci.data, i)))
            .collect();
        for (k, id) in out_ids.iter().enumerate() {
            assert_eq!(
                csr_row(&co.indptr, &co.indices, &co.data, k),
                in_map[id],
                "modality {mid} X row for {id}"
            );
        }
    }
}

#[test]
fn multimodal_is_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_multimodal(&dir);
    let a = dir.path().join("a.scx");
    let b = dir.path().join("b.scx");
    sort(&inp, &a, &opts(&["cell_type"])).unwrap();
    sort(&inp, &b, &opts(&["cell_type"])).unwrap();
    let (ra, rb) = (ScxReader::open(&a).unwrap(), ScxReader::open(&b).unwrap());
    assert_eq!(col_of(&a, "cell_id"), col_of(&b, "cell_id"));
    for mid in [1u8, 2] {
        let ca = ra.read_all_csr_shards_for(mid).unwrap();
        let cb = rb.read_all_csr_shards_for(mid).unwrap();
        assert_eq!(ca.indptr, cb.indptr);
        assert_eq!(ca.indices, cb.indices);
        assert_eq!(ca.data, cb.data);
    }
}

// --- T5.2 / T5.3 obsp remap ------------------------------------------------

#[test]
fn obsp_remapped_through_permutation() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_obsp_layers(&dir); // 8 obs; edge r -> (r+1)%8, data r+1; "raw" layer
    let out = dir.path().join("out.scx");
    sort(&inp, &out, &opts(&["cell_type"])).unwrap();

    let ro = ScxReader::open(&out).unwrap();
    let out_ids = col_of(&out, "cell_id");
    let np = new_pos_map(&out_ids, 8);

    let obsp = ro.read_obsp("connectivities").unwrap();
    assert_eq!(obsp_dim(&obsp, "n_rows"), 8, "no deletions → dim unchanged");
    assert_eq!(obsp_dim(&obsp, "n_cols"), 8);

    // Every original edge (r -> (r+1)%8, r+1) maps to (np[r] -> np[(r+1)%8]).
    let expected: HashSet<(i64, i64, u32)> = (0..8i64)
        .map(|r| {
            let c = (r + 1) % 8;
            (np[r as usize], np[c as usize], (r + 1) as u32)
        })
        .collect();
    let got: HashSet<(i64, i64, u32)> = obsp_edges(&obsp)
        .into_iter()
        .map(|(r, c, d)| (r, c, d as u32))
        .collect();
    assert_eq!(got, expected, "obsp edges remapped through the sort order");

    // The `raw` layer is reordered like X.
    let ri = ScxReader::open(&inp).unwrap();
    let li = ri.read_layer("raw").unwrap();
    let lo = ro.read_layer("raw").unwrap();
    let in_ids = str_col(&ri.read_obs().unwrap(), "cell_id");
    let in_map: HashMap<&String, Vec<(i32, f32)>> = in_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id, csr_row(&li.indptr, &li.indices, &li.data, i)))
        .collect();
    for (k, id) in out_ids.iter().enumerate() {
        assert_eq!(csr_row(&lo.indptr, &lo.indices, &lo.data, k), in_map[id]);
    }
}

#[test]
fn obsp_drops_edges_to_deleted_endpoints() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_obsp_layers(&dir); // 8 obs
    crate::delete::mark_deleted(&inp, &[2u64, 5]).unwrap();
    let out = dir.path().join("out.scx");
    let summary = sort(&inp, &out, &opts(&["cell_type"])).unwrap();
    assert_eq!(summary.n_obs, 6);

    let ro = ScxReader::open(&out).unwrap();
    let np = new_pos_map(&col_of(&out, "cell_id"), 8);
    let obsp = ro.read_obsp("connectivities").unwrap();
    assert_eq!(obsp_dim(&obsp, "n_rows"), 6, "dims collapse to live count");
    assert_eq!(obsp_dim(&obsp, "n_cols"), 6);

    // Original edges touching old row 2 or 5 (as row or col) are dropped; the
    // rest are remapped. Edge r -> (r+1)%8: deleted endpoints are {2,5}, so
    // edges with r in {2,5} or (r+1)%8 in {2,5} (i.e. r in {1,4}) drop.
    let expected: HashSet<(i64, i64)> = (0..8i64)
        .filter(|&r| {
            let c = (r + 1) % 8;
            np[r as usize] >= 0 && np[c as usize] >= 0
        })
        .map(|r| (np[r as usize], np[((r + 1) % 8) as usize]))
        .collect();
    let got: HashSet<(i64, i64)> = obsp_edges(&obsp)
        .into_iter()
        .map(|(r, c, _)| (r, c))
        .collect();
    assert_eq!(got, expected);
    // No surviving edge references an out-of-range (deleted) endpoint.
    for (r, c) in &got {
        assert!(*r >= 0 && *r < 6 && *c >= 0 && *c < 6);
    }
}

// --- T5.4 bitmap rebuild ---------------------------------------------------

/// Sum per-gene detection counts across all bitmap shards in `path`.
fn bitmap_total_counts(path: &Path, n_vars: usize) -> (usize, Vec<u64>) {
    let r = ScxReader::open(path).unwrap();
    let n_bm = r
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::BitmapShard)
        .count();
    let mut total = vec![0u64; n_vars];
    for i in 0..n_bm {
        let bm = r.read_bitmap_shard(i).unwrap();
        for (g, c) in bm.per_gene_counts().iter().enumerate() {
            total[g] += c;
        }
    }
    (n_bm, total)
}

/// Direct per-gene nnz recount over a file's main X.
fn recount_genes(path: &Path, n_vars: usize) -> Vec<u64> {
    let csr = ScxReader::open(path)
        .unwrap()
        .read_all_csr_shards()
        .unwrap();
    let mut counts = vec![0u64; n_vars];
    for &g in &csr.indices {
        counts[g as usize] += 1;
    }
    counts
}

#[test]
fn bitmap_always_rebuilds_and_matches_recount() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir); // 12 obs, 8 vars
    let out = dir.path().join("out.scx");
    let mut o = opts(&["cell_type"]); // shard_size 2 → multiple X shards + bitmaps
    o.bitmap = BitmapPolicy::Always;
    sort(&inp, &out, &o).unwrap();

    let (n_bm, total) = bitmap_total_counts(&out, 8);
    assert_eq!(n_bm, 6, "one bitmap per X shard (12 rows / 2)");
    assert_eq!(
        total,
        recount_genes(&out, 8),
        "bitmap counts == sorted-X recount"
    );
}

#[test]
fn bitmap_auto_builds_on_sparse_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir); // density 2/8 = 0.25 ≤ 0.30 → auto builds
    let out = dir.path().join("out.scx");
    let mut o = opts(&["cell_type"]);
    o.bitmap = BitmapPolicy::Auto;
    sort(&inp, &out, &o).unwrap();
    let (n_bm, total) = bitmap_total_counts(&out, 8);
    assert!(n_bm > 0, "auto must build on a sparse fixture");
    assert_eq!(total, recount_genes(&out, 8));
}

#[test]
fn bitmap_off_writes_none() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    sort(&inp, &out, &opts(&["cell_type"])).unwrap(); // default Off
    let (n_bm, _) = bitmap_total_counts(&out, 8);
    assert_eq!(n_bm, 0, "default bitmap policy drops the sidecar");
}

// --- null sort-key handling (review fix) -----------------------------------

#[test]
fn null_key_in_memory_matches_external() {
    // The in-memory and external strategies both derive order from the full
    // `stable_argsort` (nulls placed via nulls_first), so they must agree even
    // when the sort key contains nulls — and no row may be dropped.
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_null_key(&dir); // 12 obs, cell_type with 3 nulls
    let a = dir.path().join("a.scx");
    let b = dir.path().join("b.scx");
    let mut o = opts(&["cell_type"]);
    o.memory_budget = Some(64);
    sort_with_strategy(&inp, &a, &o, Some(SortStrategy::InMemory)).unwrap();
    sort_with_strategy(&inp, &b, &o, Some(SortStrategy::ExternalPartition)).unwrap();

    let (a_ids, a_rows) = content(&a);
    assert_eq!(a_ids.len(), 12, "no rows dropped (incl. null-key rows)");
    assert_eq!(
        (a_ids, a_rows),
        content(&b),
        "in-memory == external on nulls"
    );

    // Round-trips the full cell set including the null-key cells.
    let (in_ids, _) = content(&inp);
    let got: HashSet<String> = col_of(&a, "cell_id").into_iter().collect();
    assert_eq!(got, in_ids.into_iter().collect::<HashSet<_>>());
}

#[test]
fn kpass_rejects_null_key() {
    // Forcing K-pass on a null-containing key must error rather than silently
    // drop the null-key rows.
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_null_key(&dir);
    let out = dir.path().join("out.scx");
    let mut o = opts(&["cell_type"]);
    o.memory_budget = Some(64);
    let err = sort_with_strategy(&inp, &out, &o, Some(SortStrategy::KPassByCategory));
    assert!(
        err.is_err(),
        "K-pass must reject a null-containing sort key"
    );
}

#[test]
fn auto_selection_avoids_kpass_on_null_key() {
    // With a budget that would otherwise pick K-pass (small categorical key),
    // a null in the key must route to a null-safe strategy and preserve all
    // rows.
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_null_key(&dir);
    let out = dir.path().join("out.scx");
    let mut o = opts(&["cell_type"]);
    o.memory_budget = Some(64);
    let summary = sort(&inp, &out, &o).unwrap();
    assert_ne!(summary.strategy, SortStrategy::KPassByCategory);
    assert_eq!(summary.n_obs, 12);
}

// ===========================================================================
// F1 — grouped sharding integration tests
// ===========================================================================

/// Build a small grouped-screen fixture: `cell_id`, a `target_gene` group
/// column, and a boolean `is_control` column. `genes[i]` / `control[i]` give
/// row i's values. CSR is deterministic 2-nnz-per-row, single input shard.
fn write_grouped_fixture(
    dir: &tempfile::TempDir,
    name: &str,
    genes: &[&str],
    control: &[bool],
) -> std::path::PathBuf {
    let n_obs = genes.len();
    let n_vars = 4usize;
    assert_eq!(control.len(), n_obs);
    let path = dir.path().join(name);
    let mut writer = ScxWriter::new(
        &path,
        super::super::test_utils::sample_header(n_obs as u64, n_vars as u64),
    )
    .unwrap();

    let ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("target_gene", DataType::Utf8, true),
        Field::new("is_control", DataType::Boolean, true),
    ]);
    let obs = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(genes.to_vec())),
            Arc::new(arrow::array::BooleanArray::from(control.to_vec())),
        ],
    )
    .unwrap();
    writer.write_obs(&obs).unwrap();
    writer
        .write_var(&super::super::test_utils::sample_var(n_vars))
        .unwrap();

    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..n_obs {
        indices.push((row * 2 % n_vars) as u32);
        indices.push(((row * 2 + 1) % n_vars) as u32);
        values.push(((row + 1) % 256) as u8);
        values.push(((row + 2) % 256) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer.finish().unwrap();
    path
}

fn read_group_index(path: &Path) -> serde_json::Value {
    let r = ScxReader::open(path).unwrap();
    let entry = r
        .catalog()
        .get("group_index")
        .expect("group_index section present");
    let bytes = r.section_bytes(entry).unwrap();
    serde_json::from_slice(bytes).unwrap()
}

fn csr_shard_ranges(path: &Path) -> Vec<(u64, u64)> {
    let r = ScxReader::open(path).unwrap();
    r.catalog()
        .shards_sorted()
        .iter()
        .map(|e| {
            let s = e.stats.as_ref().unwrap();
            (s.row_start, s.row_end)
        })
        .collect()
}

/// Structural invariants every grouped output must satisfy, asserted against
/// the output obs `target_gene` order, the GroupIndex sidecar, and the CSR
/// shard ranges.
fn assert_grouped_invariants(path: &Path, reference_labels: &[&str]) -> serde_json::Value {
    let gi = read_group_index(path);
    let genes_out = col_of(path, "target_gene");
    let ranges = csr_shard_ranges(path);

    // Reference rows (by label) cluster first.
    let ref_set: HashSet<&str> = reference_labels.iter().copied().collect();
    if !ref_set.is_empty() {
        let first_non_ref = genes_out.iter().position(|g| !ref_set.contains(g.as_str()));
        if let Some(fnr) = first_non_ref {
            assert!(
                genes_out[fnr..]
                    .iter()
                    .all(|g| !ref_set.contains(g.as_str())),
                "reference rows must all precede non-reference rows"
            );
        }
    }

    // Each sidecar record's global range is uniform in its label, role matches
    // the reference set, and lies entirely within one CSR shard (never-split).
    for rec in gi["records"].as_array().unwrap() {
        let label = rec["label"].as_str().unwrap();
        let start = rec["row_start"].as_u64().unwrap();
        let stop = rec["row_stop"].as_u64().unwrap();
        let role = rec["role"].as_str().unwrap();
        for r in start..stop {
            assert_eq!(
                genes_out[r as usize], label,
                "record range must be uniform in label"
            );
        }
        // For a label-based reference spec (`ref_set` non-empty), reference and
        // group labels are disjoint, so role must agree with set membership.
        // The `Column` split-label case passes an empty `ref_set` and is exempt
        // (a label can carry both roles there).
        if !ref_set.is_empty() {
            assert_eq!(
                role == "reference",
                ref_set.contains(label),
                "record role for {label:?} must match reference-set membership"
            );
        }
        assert!(
            ranges.iter().any(|&(s, e)| s <= start && stop <= e),
            "record [{start},{stop}) for {label:?} escapes every CSR shard range {ranges:?}"
        );
    }
    gi
}

#[test]
fn grouped_sort_reference_first_and_clustered_inmemory() {
    let dir = tempfile::tempdir().unwrap();
    // "nt" is the reference label, scattered through the input.
    let genes = [
        "nt", "MYC", "nt", "TP53", "MYC", "GATA1", "nt", "MYC", "GATA1", "GATA1",
    ];
    let control = genes.iter().map(|g| *g == "nt").collect::<Vec<_>>();
    let inp = write_grouped_fixture(&dir, "screen.scx", &genes, &control);
    let out = dir.path().join("grouped.scx");

    let o = SortOptions {
        group_by: Some("target_gene".to_string()),
        reference: Some(ReferenceSpec::Labels(vec!["nt".to_string()])),
        shard_target_rows: 3,
        ..Default::default()
    };
    let summary = sort(&inp, &out, &o).unwrap();
    assert_eq!(summary.n_obs, 10);

    let gi = assert_grouped_invariants(&out, &["nt"]);
    assert_eq!(gi["group_by"], "target_gene");
    assert_eq!(gi["reference_shard"], 0);
    // 3 reference rows lead the output.
    let genes_out = col_of(&out, "target_gene");
    assert_eq!(&genes_out[0..3], &["nt", "nt", "nt"]);
    // Every non-"nt" record is role "group".
    for rec in gi["records"].as_array().unwrap() {
        let label = rec["label"].as_str().unwrap();
        let role = rec["role"].as_str().unwrap();
        assert_eq!(role == "reference", label == "nt");
    }
}

#[test]
fn grouped_single_shard_plan_does_not_split_group() {
    // Regression: when the plan collapses to a single shard, the planner returns
    // an EMPTY `shard_starts`. The emitter must still be in grouped mode (breaks
    // = Some(empty)) and NOT fall back to the legacy `shard_target_rows` cap,
    // which would split the group across shards and desync the sidecar. Here one
    // group of 6 rows with `shard_target_rows: 2` fits one planner shard; a
    // pre-fix emitter would emit 3 X shards.
    let dir = tempfile::tempdir().unwrap();
    let genes = ["MYC", "MYC", "MYC", "MYC", "MYC", "MYC"];
    let control = [false; 6];
    let inp = write_grouped_fixture(&dir, "single.scx", &genes, &control);
    let out = dir.path().join("single_out.scx");

    let o = SortOptions {
        group_by: Some("target_gene".to_string()),
        shard_target_rows: 2,
        ..Default::default()
    };
    sort(&inp, &out, &o).unwrap();

    // Exactly one CSR shard covering all 6 rows — the group was not split.
    let ranges = csr_shard_ranges(&out);
    assert_eq!(ranges, vec![(0, 6)], "single group must land in one shard");

    // One record spanning the whole file, in that single shard.
    let gi = assert_grouped_invariants(&out, &[]);
    let records = gi["records"].as_array().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["label"], "MYC");
    assert_eq!(records[0]["shard"], 0);
    assert_eq!(records[0]["row_start"].as_u64().unwrap(), 0);
    assert_eq!(records[0]["row_stop"].as_u64().unwrap(), 6);
}

// --- M1: oversized group vs --memory-budget --------------------------------

#[test]
fn max_grouped_shard_footprint_row_and_byte_mode() {
    use crate::group_plan::{GroupPlan, GroupRecord, Role};
    // Two shards: shard 0 = one 2-row group "A", shard 1 = one 6-row group "B".
    let plan = GroupPlan {
        shard_starts: vec![2],
        records: vec![
            GroupRecord {
                label: "A".to_string(),
                shard: 0,
                row_start: 0,
                row_stop: 2,
                role: Role::Group,
            },
            GroupRecord {
                label: "B".to_string(),
                shard: 1,
                row_start: 2,
                row_stop: 8,
                role: Role::Group,
            },
        ],
        reference_shard: None,
        n_shards: 2,
    };

    // Row-count mode (empty per_row_nnz): per-row est = ceil(n_vars*density*16).
    // n_vars=4, density=0.5 -> 32 B/row. Shard 1 (6 rows) = 192 B dominates.
    let (bytes, shard, label) = super::max_grouped_shard_footprint(&plan, &[], 4, 0.5);
    assert_eq!((bytes, shard, label.as_str()), (192, 1, "B"));

    // Byte mode: per_row_nnz supplied (emission order). Shard 1 rows [2,8) have
    // nnz summing to 30 -> 30*16 + 6*8 = 528 B; shard 0 rows [0,2) -> 4 nnz ->
    // 4*16 + 2*8 = 80 B. Shard 1 dominates.
    let per_row_nnz = vec![1, 3, 5, 5, 5, 5, 5, 5];
    let (bytes, shard, label) = super::max_grouped_shard_footprint(&plan, &per_row_nnz, 4, 0.5);
    assert_eq!((bytes, shard, label.as_str()), (528, 1, "B"));
}

/// F6 Phase 0 relaxed the M1 guard. A dominant group whose buffered footprint
/// exceeds `--memory-budget` now hard-errors **only when the block sub-flush is
/// explicitly disabled** (`group_write_block_bytes = Some(0)`); with the default
/// cap the guard downgrades to a warning and the sort proceeds (the sub-flush
/// bounds RSS instead of buffering the group whole). A sufficient budget still
/// round-trips regardless.
#[test]
fn grouped_sort_oversized_group_budget_guard() {
    let dir = tempfile::tempdir().unwrap();
    // One group of 8 rows; n_vars=4, 2 nnz/row -> density 0.5 -> 32 B/row ->
    // shard footprint 256 B.
    let genes = ["MYC"; 8];
    let control = [false; 8];
    let inp = write_grouped_fixture(&dir, "big_group.scx", &genes, &control);

    // Sub-flush disabled + tiny budget: 256 B needed, 100 B allowed -> refuse.
    let out_err = dir.path().join("err.scx");
    let o_err = SortOptions {
        group_by: Some("target_gene".to_string()),
        memory_budget: Some(100),
        shard_target_rows: 2,
        group_write_block_bytes: Some(0),
        ..Default::default()
    };
    let err = sort(&inp, &out_err, &o_err).expect_err("disabled sub-flush must exceed the budget");
    let msg = err.to_string();
    assert!(
        msg.contains("memory-budget") && msg.contains("MYC") && msg.contains("sub-flush"),
        "error must name the group and the disabled sub-flush: {msg}"
    );

    // Default sub-flush + same tiny budget: the guard warns and the sort proceeds.
    let out_warn = dir.path().join("warn.scx");
    let o_warn = SortOptions {
        group_write_block_bytes: None,
        ..o_err.clone()
    };
    sort(&inp, &out_warn, &o_warn).expect("default sub-flush must not hard-error on the budget");
    assert!(col_of(&out_warn, "target_gene").iter().all(|g| g == "MYC"));

    // Ample budget: same layout succeeds and round-trips.
    let ok = dir.path().join("ok.scx");
    let o_ok = SortOptions {
        memory_budget: Some(1 << 20),
        ..o_err
    };
    sort(&inp, &ok, &o_ok).unwrap();
    let genes_out = col_of(&ok, "target_gene");
    assert_eq!(genes_out.len(), 8);
    assert!(genes_out.iter().all(|g| g == "MYC"));
}

#[test]
fn grouped_sort_external_matches_inmemory_layout() {
    let dir = tempfile::tempdir().unwrap();
    let genes = [
        "nt", "MYC", "nt", "TP53", "MYC", "GATA1", "nt", "MYC", "GATA1", "GATA1",
    ];
    let control = genes.iter().map(|g| *g == "nt").collect::<Vec<_>>();
    let inp = write_grouped_fixture(&dir, "screen.scx", &genes, &control);

    let mk = || SortOptions {
        group_by: Some("target_gene".to_string()),
        reference: Some(ReferenceSpec::Labels(vec!["nt".to_string()])),
        shard_target_rows: 3,
        ..Default::default()
    };

    let out_mem = dir.path().join("mem.scx");
    sort_with_strategy(&inp, &out_mem, &mk(), Some(SortStrategy::InMemory)).unwrap();
    let out_ext = dir.path().join("ext.scx");
    sort_with_strategy(&inp, &out_ext, &mk(), Some(SortStrategy::ExternalPartition)).unwrap();

    // Identical group order/roles/global ranges (and here, same tool → same
    // shard assignment too).
    assert_eq!(read_group_index(&out_mem), read_group_index(&out_ext));
    assert_eq!(
        col_of(&out_mem, "target_gene"),
        col_of(&out_ext, "target_gene")
    );
}

#[test]
fn grouped_sort_split_label_column_reference() {
    // ReferenceSpec::Column where label "shared" has both control and
    // non-control rows → it must split into two records (reference + group).
    let dir = tempfile::tempdir().unwrap();
    let genes = ["shared", "shared", "MYC", "shared", "shared", "MYC"];
    let control = [true, true, false, false, false, false];
    let inp = write_grouped_fixture(&dir, "split.scx", &genes, &control);
    let out = dir.path().join("split_out.scx");

    let o = SortOptions {
        group_by: Some("target_gene".to_string()),
        reference: Some(ReferenceSpec::Column("is_control".to_string())),
        shard_target_rows: 100,
        ..Default::default()
    };
    sort(&inp, &out, &o).unwrap();

    let gi = assert_grouped_invariants(&out, &[]);
    assert_eq!(gi["reference_shard"], 0);
    let recs = gi["records"].as_array().unwrap();
    let shared: Vec<&serde_json::Value> = recs.iter().filter(|r| r["label"] == "shared").collect();
    assert_eq!(shared.len(), 2, "split label must yield two records");
    assert!(shared.iter().any(|r| r["role"] == "reference"));
    assert!(shared.iter().any(|r| r["role"] == "group"));
    // The two reference control rows lead the output.
    let genes_out = col_of(&out, "target_gene");
    assert_eq!(&genes_out[0..2], &["shared", "shared"]);
}

#[test]
fn grouped_sort_reverse_is_ignored() {
    // --reverse with --group-by must be forced off (reference sorts first).
    let dir = tempfile::tempdir().unwrap();
    let genes = ["nt", "MYC", "nt", "ABC"];
    let control = [true, false, true, false];
    let inp = write_grouped_fixture(&dir, "rev.scx", &genes, &control);
    let out = dir.path().join("rev_out.scx");
    let o = SortOptions {
        group_by: Some("target_gene".to_string()),
        reference: Some(ReferenceSpec::Labels(vec!["nt".to_string()])),
        reverse: true,
        shard_target_rows: 100,
        ..Default::default()
    };
    sort(&inp, &out, &o).unwrap();
    let genes_out = col_of(&out, "target_gene");
    assert_eq!(
        &genes_out[0..2],
        &["nt", "nt"],
        "reference must lead despite --reverse"
    );
}

#[test]
fn non_grouped_sort_writes_no_group_index() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("plain_sorted.scx");
    sort(&inp, &out, &opts(&["cell_type"])).unwrap();
    let r = ScxReader::open(&out).unwrap();
    assert!(r.catalog().get("group_index").is_none());
}

#[test]
fn grouped_read_back_matches_full_scan() {
    use scx_engine::QueryPipeline;
    let dir = tempfile::tempdir().unwrap();
    let genes = [
        "nt", "MYC", "nt", "TP53", "MYC", "GATA1", "nt", "MYC", "GATA1", "GATA1",
    ];
    let control = genes.iter().map(|g| *g == "nt").collect::<Vec<_>>();
    let inp = write_grouped_fixture(&dir, "screen.scx", &genes, &control);
    let out = dir.path().join("grouped.scx");
    let o = SortOptions {
        group_by: Some("target_gene".to_string()),
        reference: Some(ReferenceSpec::Labels(vec!["nt".to_string()])),
        shard_target_rows: 3,
        ..Default::default()
    };
    sort(&inp, &out, &o).unwrap();

    // Full sorted output for cross-checking X rows.
    let (_ids, full_rows) = content(&out);
    let genes_out = col_of(&out, "target_gene");

    let pipe = QueryPipeline::open(&out).unwrap();

    // read_group("MYC"): obs all MYC; X rows equal the corresponding slice of
    // the full output; only the group's shard(s) decoded.
    let qr = pipe.read_group("MYC").unwrap();
    let myc_global: Vec<usize> = genes_out
        .iter()
        .enumerate()
        .filter(|(_, g)| g.as_str() == "MYC")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(qr.x.shape.0, myc_global.len());
    assert!(qr.skipped_shards > 0, "read_group must prune shards");
    let qr_genes = str_col(&qr.obs, "target_gene");
    assert!(qr_genes.iter().all(|g| g == "MYC"));
    // X content row-for-row.
    for (local, &g) in myc_global.iter().enumerate() {
        let s = qr.x.indptr[local] as usize;
        let e = qr.x.indptr[local + 1] as usize;
        let got: Vec<(i32, f32)> = (s..e).map(|j| (qr.x.indices[j], qr.x.data[j])).collect();
        assert_eq!(got, full_rows[g], "X row mismatch for MYC local {local}");
    }

    // read_reference: all rows are "nt".
    let refq = pipe.read_reference().unwrap().unwrap();
    let ref_genes = str_col(&refq.obs, "target_gene");
    assert!(ref_genes.iter().all(|g| g == "nt"));
    assert_eq!(refq.x.shape.0, genes.iter().filter(|g| **g == "nt").count());

    // group_labels excludes nothing structural; iter_group_shards covers every
    // non-reference label exactly once.
    let labels = pipe.group_labels().unwrap();
    for l in ["GATA1", "MYC", "TP53", "nt"] {
        assert!(labels.contains(&l.to_string()), "labels missing {l}");
    }
    let handles = pipe.iter_group_shards().unwrap();
    let mut seen = HashSet::new();
    for h in &handles {
        for (lab, _, _) in &h.groups {
            assert!(
                seen.insert(lab.clone()),
                "label {lab} appeared in two shards"
            );
            assert_ne!(lab, "nt", "reference label must not appear in group shards");
        }
    }
    assert_eq!(
        seen,
        ["GATA1", "MYC", "TP53"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    );

    // Unknown label errors with suggestions.
    let err = pipe.read_group("MYCN").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("unknown group label"), "got: {msg}");
}

#[test]
fn grouped_read_is_obs_shard_scoped() {
    // 7.1a: on a row-sharded file `read_group` must read only the obs shards
    // overlapping the group's range, never the full obs table. The
    // `read_obs` / `read_obs_shard` debug counters distinguish the two paths.
    use scx_engine::QueryPipeline;
    use std::sync::atomic::Ordering::Relaxed;
    let dir = tempfile::tempdir().unwrap();
    let genes = [
        "nt", "MYC", "nt", "TP53", "MYC", "GATA1", "nt", "MYC", "GATA1", "GATA1",
    ];
    let control = genes.iter().map(|g| *g == "nt").collect::<Vec<_>>();
    let inp = write_grouped_fixture(&dir, "screen.scx", &genes, &control);
    let out = dir.path().join("grouped.scx");
    let o = SortOptions {
        group_by: Some("target_gene".to_string()),
        reference: Some(ReferenceSpec::Labels(vec!["nt".to_string()])),
        shard_target_rows: 3,
        ..Default::default()
    };
    sort(&inp, &out, &o).unwrap();

    // Confirm the output really is multi-obs-shard (otherwise the assertion is
    // vacuous).
    assert!(
        ScxReader::open(&out).unwrap().obs_metadata_shard_count() >= 2,
        "fixture must produce a multi-obs-shard file"
    );

    let pipe = QueryPipeline::open(&out).unwrap();
    let _ = pipe.read_group("MYC").unwrap();
    let _ = pipe.read_reference().unwrap();

    let counts = pipe
        .local_reader()
        .expect("local file pipeline")
        .debug_counts();
    assert_eq!(
        counts.read_obs.load(Relaxed),
        0,
        "grouped read on a sharded file must not materialize the full obs table"
    );
    assert!(
        counts.read_obs_shard.load(Relaxed) >= 1,
        "grouped read must take the shard-scoped obs path"
    );
}

#[test]
fn non_grouped_read_group_errors_not_grouped() {
    use scx_engine::QueryPipeline;
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("plain_sorted.scx");
    sort(&inp, &out, &opts(&["cell_type"])).unwrap();
    let pipe = QueryPipeline::open(&out).unwrap();
    assert!(pipe.read_group("anything").is_err());
    assert!(pipe.require_grouped().is_err());
}

#[test]
fn grouped_sort_is_deterministic() {
    // Two grouped sorts of the same input must produce byte-identical X content
    // and an identical group_index sidecar (guards against layout drift — the
    // role a committed golden file would play, without the binary artifact).
    let dir = tempfile::tempdir().unwrap();
    let genes = [
        "nt", "MYC", "nt", "TP53", "MYC", "GATA1", "nt", "MYC", "GATA1", "GATA1",
    ];
    let control = genes.iter().map(|g| *g == "nt").collect::<Vec<_>>();
    let inp = write_grouped_fixture(&dir, "screen.scx", &genes, &control);
    let o = SortOptions {
        group_by: Some("target_gene".to_string()),
        reference: Some(ReferenceSpec::Labels(vec!["nt".to_string()])),
        shard_target_rows: 3,
        ..Default::default()
    };
    let a = dir.path().join("a.scx");
    let b = dir.path().join("b.scx");
    sort(&inp, &a, &o).unwrap();
    sort(&inp, &b, &o).unwrap();
    assert_eq!(content(&a), content(&b));
    assert_eq!(read_group_index(&a), read_group_index(&b));
}

#[test]
fn grouped_read_respects_deletion_vectors() {
    // After a post-sort mark_deleted, grouped reads must drop the deleted rows
    // (closes the read_row_range deletion-vector gap).
    use scx_engine::QueryPipeline;
    let dir = tempfile::tempdir().unwrap();
    let genes = [
        "nt", "MYC", "nt", "TP53", "MYC", "GATA1", "nt", "MYC", "GATA1", "GATA1",
    ];
    let control = genes.iter().map(|g| *g == "nt").collect::<Vec<_>>();
    let inp = write_grouped_fixture(&dir, "screen.scx", &genes, &control);
    let out = dir.path().join("grouped.scx");
    let o = SortOptions {
        group_by: Some("target_gene".to_string()),
        reference: Some(ReferenceSpec::Labels(vec!["nt".to_string()])),
        shard_target_rows: 100,
        ..Default::default()
    };
    sort(&inp, &out, &o).unwrap();

    let genes_out = col_of(&out, "target_gene");
    let myc_global: Vec<u64> = genes_out
        .iter()
        .enumerate()
        .filter(|(_, g)| g.as_str() == "MYC")
        .map(|(i, _)| i as u64)
        .collect();
    // Mark the first MYC cell deleted (by its post-sort global row index).
    crate::mark_deleted(&out, &[myc_global[0]]).unwrap();

    let pipe = QueryPipeline::open(&out).unwrap();
    let qr = pipe.read_group("MYC").unwrap();
    assert_eq!(
        qr.x.shape.0,
        myc_global.len() - 1,
        "deleted MYC cell must be excluded from read_group"
    );
    let qr_genes = str_col(&qr.obs, "target_gene");
    assert_eq!(qr_genes.len(), myc_global.len() - 1);
    assert!(qr_genes.iter().all(|g| g == "MYC"));

    // Equivalence with the predicate path, which also respects deletions.
    let viaq = QueryPipeline::open(&out)
        .unwrap()
        .filter_obs("target_gene == 'MYC'")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(viaq.x.shape.0, qr.x.shape.0);

    // read_reference still returns all (undeleted) nt cells.
    let refq = pipe.read_reference().unwrap().unwrap();
    assert_eq!(refq.x.shape.0, genes.iter().filter(|g| **g == "nt").count());
}

#[test]
fn read_row_range_rejects_out_of_bounds() {
    use scx_engine::QueryPipeline;
    let dir = tempfile::tempdir().unwrap();
    let genes = ["nt", "MYC", "TP53"];
    let control = [true, false, false];
    let inp = write_grouped_fixture(&dir, "screen.scx", &genes, &control);
    let out = dir.path().join("grouped.scx");
    let o = SortOptions {
        group_by: Some("target_gene".to_string()),
        reference: Some(ReferenceSpec::Labels(vec!["nt".to_string()])),
        ..Default::default()
    };
    sort(&inp, &out, &o).unwrap();
    let n = ScxReader::open(&out).unwrap().n_obs();
    let pipe = QueryPipeline::open(&out).unwrap();
    assert!(
        pipe.read_row_range(0, n + 5).is_err(),
        "stop past n_obs must error rather than panic / corrupt"
    );
    // Valid full-range read still works.
    assert_eq!(pipe.read_row_range(0, n).unwrap().x.shape.0 as u64, n);
}

#[test]
fn require_grouped_caches_parsed_index() {
    // 7.1b: the sidecar is parsed at most once per pipeline; repeated calls
    // return the same cached `GroupIndex` (proven by pointer identity).
    use scx_engine::QueryPipeline;
    let dir = tempfile::tempdir().unwrap();
    let genes = ["nt", "MYC", "nt", "TP53", "MYC"];
    let control = genes.iter().map(|g| *g == "nt").collect::<Vec<_>>();
    let inp = write_grouped_fixture(&dir, "screen.scx", &genes, &control);
    let out = dir.path().join("grouped.scx");
    let o = SortOptions {
        group_by: Some("target_gene".to_string()),
        reference: Some(ReferenceSpec::Labels(vec!["nt".to_string()])),
        ..Default::default()
    };
    sort(&inp, &out, &o).unwrap();

    let pipe = QueryPipeline::open(&out).unwrap();
    let a = pipe.require_grouped().unwrap();
    let b = pipe.require_grouped().unwrap();
    assert!(
        std::ptr::eq(a, b),
        "require_grouped must return the same cached GroupIndex on repeat calls"
    );
    // And grouped reads in a loop stay correct over the cache.
    for _ in 0..3 {
        assert!(pipe.read_group("MYC").is_ok());
    }
}

#[test]
fn append_drops_group_index_sidecar() {
    use scx_engine::QueryPipeline;
    let dir = tempfile::tempdir().unwrap();
    let genes = ["nt", "MYC", "nt", "TP53", "MYC"];
    let control = genes.iter().map(|g| *g == "nt").collect::<Vec<_>>();
    let inp = write_grouped_fixture(&dir, "screen.scx", &genes, &control);
    let out = dir.path().join("grouped.scx");
    let o = SortOptions {
        group_by: Some("target_gene".to_string()),
        reference: Some(ReferenceSpec::Labels(vec!["nt".to_string()])),
        ..Default::default()
    };
    sort(&inp, &out, &o).unwrap();
    assert!(QueryPipeline::open(&out).unwrap().require_grouped().is_ok());

    // Append 2 rows with a matching obs schema (cell_id, target_gene, is_control).
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("target_gene", DataType::Utf8, true),
        Field::new("is_control", DataType::Boolean, true),
    ]);
    let new_obs = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(vec!["new_0", "new_1"])),
            Arc::new(StringArray::from(vec!["MYC", "TP53"])),
            Arc::new(arrow::array::BooleanArray::from(vec![false, false])),
        ],
    )
    .unwrap();
    let n_vars = 4usize; // write_grouped_fixture uses 4 vars
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..2usize {
        indices.push((row * 2 % n_vars) as u32);
        indices.push(((row * 2 + 1) % n_vars) as u32);
        values.push(1u8);
        values.push(2u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    crate::append(
        &out,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        &crate::AppendOptions::default(),
    )
    .unwrap();

    // The stale grouped sidecar must be gone; grouped reads cleanly report it.
    let r = ScxReader::open(&out).unwrap();
    assert!(
        r.catalog().get("group_index").is_none(),
        "append must drop the stale group_index sidecar"
    );
    assert!(QueryPipeline::open(&out)
        .unwrap()
        .require_grouped()
        .is_err());
}

// ---------------------------------------------------------------------------
// F6 Phase 0 — block-level sub-flush of oversized groups (T0.4 / T0.5)
// ---------------------------------------------------------------------------
//
// The fixture writer emits 2 nnz/row with Uint8 values, so the emitter's
// accumulation-byte estimate is `2*4 (indices) + 2 (values) = 10 bytes/row`.
// A `group_write_block_bytes` of 120 therefore sub-flushes every 12 rows.

/// T0.4 — an oversized group is sub-flushed across multiple output shards, and
/// `read_group` unions them back into the full, byte-correct group. This is the
/// OOM fix: instead of buffering the whole group, the emitter caps at one block.
/// It also exercises a block boundary landing exactly on a group edge (the BIG
/// group ends flush against the start of the next shard).
#[test]
fn grouped_sort_subflush_splits_oversized_group() {
    use scx_engine::QueryPipeline;
    let dir = tempfile::tempdir().unwrap();
    // One dominant 60-row group + two singletons, no reference.
    let mut genes: Vec<&str> = vec!["BIG"; 60];
    genes.push("A");
    genes.push("B");
    let control = vec![false; genes.len()];
    let inp = write_grouped_fixture(&dir, "big.scx", &genes, &control);
    let out = dir.path().join("big_out.scx");
    let o = SortOptions {
        group_by: Some("target_gene".to_string()),
        shard_target_rows: 1000, // large row cap; the sub-flush is byte-driven
        group_write_block_bytes: Some(120), // ~12 rows/block
        ..Default::default()
    };
    sort(&inp, &out, &o).unwrap();

    // BIG must span multiple records (one per sub-flushed shard).
    let gi = assert_grouped_invariants(&out, &[]);
    let big_recs: Vec<_> = gi["records"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["label"] == "BIG")
        .collect();
    assert!(
        big_recs.len() > 1,
        "oversized group must sub-flush into multiple records, got {}",
        big_recs.len()
    );
    assert!(
        csr_shard_ranges(&out).len() > 1,
        "oversized group must produce multiple output shards"
    );

    // read_group("BIG") returns all 60 rows, row-for-row correct.
    let (_ids, full_rows) = content(&out);
    let genes_out = col_of(&out, "target_gene");
    let pipe = QueryPipeline::open(&out).unwrap();
    let qr = pipe.read_group("BIG").unwrap();
    let big_global: Vec<usize> = genes_out
        .iter()
        .enumerate()
        .filter(|(_, g)| g.as_str() == "BIG")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(qr.x.shape.0, 60);
    assert_eq!(qr.x.shape.0, big_global.len());
    assert!(str_col(&qr.obs, "target_gene").iter().all(|g| g == "BIG"));
    for (local, &g) in big_global.iter().enumerate() {
        let s = qr.x.indptr[local] as usize;
        let e = qr.x.indptr[local + 1] as usize;
        let got: Vec<(i32, f32)> = (s..e).map(|j| (qr.x.indices[j], qr.x.data[j])).collect();
        assert_eq!(got, full_rows[g], "X row mismatch for BIG local {local}");
    }
    // The trailing singletons still read back correctly (they share a shard).
    assert_eq!(pipe.read_group("A").unwrap().x.shape.0, 1);
    assert_eq!(pipe.read_group("B").unwrap().x.shape.0, 1);
}

/// T0.5 — an oversized *reference* group (the chemogenetic OOM scenario) is
/// sub-flushed into multiple reference records that still tile `[0, k)`, and
/// `read_reference` unions them back. Normal groups still read correctly.
#[test]
fn grouped_sort_subflush_reference_group_unions_on_read() {
    use scx_engine::QueryPipeline;
    let dir = tempfile::tempdir().unwrap();
    let mut genes: Vec<&str> = vec!["nt"; 40];
    genes.extend(["MYC"; 8]);
    genes.extend(["TP53"; 8]);
    let control: Vec<bool> = genes.iter().map(|g| *g == "nt").collect();
    let inp = write_grouped_fixture(&dir, "bigref.scx", &genes, &control);
    let out = dir.path().join("bigref_out.scx");
    let o = SortOptions {
        group_by: Some("target_gene".to_string()),
        reference: Some(ReferenceSpec::Labels(vec!["nt".to_string()])),
        shard_target_rows: 1000,
        group_write_block_bytes: Some(120),
        ..Default::default()
    };
    sort(&inp, &out, &o).unwrap();

    let gi = assert_grouped_invariants(&out, &["nt"]);
    let ref_recs: Vec<_> = gi["records"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["role"] == "reference")
        .collect();
    assert!(
        ref_recs.len() > 1,
        "oversized reference must sub-flush into multiple records, got {}",
        ref_recs.len()
    );

    let pipe = QueryPipeline::open(&out).unwrap();
    let refq = pipe.read_reference().unwrap().unwrap();
    assert_eq!(refq.x.shape.0, 40);
    assert!(str_col(&refq.obs, "target_gene").iter().all(|g| g == "nt"));
    assert_eq!(pipe.read_group("MYC").unwrap().x.shape.0, 8);
    assert_eq!(pipe.read_group("TP53").unwrap().x.shape.0, 8);
}

/// T0.5 — a group whose row count is not a multiple of the block size leaves a
/// single-row final block; it must still round-trip correctly.
#[test]
fn grouped_sort_subflush_single_row_last_block() {
    use scx_engine::QueryPipeline;
    let dir = tempfile::tempdir().unwrap();
    // 25 rows @ 12 rows/block => blocks of 12, 12, 1 (single-row tail).
    let mut genes: Vec<&str> = vec!["BIG"; 25];
    genes.push("A");
    let control = vec![false; genes.len()];
    let inp = write_grouped_fixture(&dir, "tail.scx", &genes, &control);
    let out = dir.path().join("tail_out.scx");
    let o = SortOptions {
        group_by: Some("target_gene".to_string()),
        shard_target_rows: 1000,
        group_write_block_bytes: Some(120),
        ..Default::default()
    };
    sort(&inp, &out, &o).unwrap();

    assert_grouped_invariants(&out, &[]);
    let (_ids, full_rows) = content(&out);
    let genes_out = col_of(&out, "target_gene");
    let pipe = QueryPipeline::open(&out).unwrap();
    let qr = pipe.read_group("BIG").unwrap();
    assert_eq!(qr.x.shape.0, 25);
    let big_global: Vec<usize> = genes_out
        .iter()
        .enumerate()
        .filter(|(_, g)| g.as_str() == "BIG")
        .map(|(i, _)| i)
        .collect();
    for (local, &g) in big_global.iter().enumerate() {
        let s = qr.x.indptr[local] as usize;
        let e = qr.x.indptr[local + 1] as usize;
        let got: Vec<(i32, f32)> = (s..e).map(|j| (qr.x.indices[j], qr.x.data[j])).collect();
        assert_eq!(got, full_rows[g], "X row mismatch for BIG local {local}");
    }
}

/// T0.5 — regression: when every group fits under the block cap the sidecar and
/// shard layout are unchanged (byte-identical to the sub-flush-disabled path).
#[test]
fn grouped_sort_subflush_inert_below_cap_matches_disabled() {
    let dir = tempfile::tempdir().unwrap();
    let genes = [
        "nt", "MYC", "nt", "TP53", "MYC", "GATA1", "nt", "MYC", "GATA1", "GATA1",
    ];
    let control: Vec<bool> = genes.iter().map(|g| *g == "nt").collect();
    let inp = write_grouped_fixture(&dir, "screen.scx", &genes, &control);
    let mk = |cap: Option<u64>| SortOptions {
        group_by: Some("target_gene".to_string()),
        reference: Some(ReferenceSpec::Labels(vec!["nt".to_string()])),
        shard_target_rows: 3,
        group_write_block_bytes: cap,
        ..Default::default()
    };
    let out_disabled = dir.path().join("disabled.scx");
    sort(&inp, &out_disabled, &mk(Some(0))).unwrap();
    let out_huge = dir.path().join("huge.scx");
    sort(&inp, &out_huge, &mk(Some(1 << 30))).unwrap();
    assert_eq!(
        read_group_index(&out_disabled),
        read_group_index(&out_huge),
        "a block cap above the data size must not change the sidecar"
    );
    assert_eq!(
        csr_shard_ranges(&out_disabled),
        csr_shard_ranges(&out_huge),
        "a block cap above the data size must not change the shard layout"
    );
    // No label split into multiple records (one (label, role) run each).
    let gi = read_group_index(&out_huge);
    let mut seen = HashSet::new();
    for rec in gi["records"].as_array().unwrap() {
        let key = format!("{}:{}", rec["label"], rec["role"]);
        assert!(
            seen.insert(key),
            "unexpected split record without sub-flush"
        );
    }
}

// ---------------------------------------------------------------------------
// In-memory grouped-write fast path (byte parity + layers)
// ---------------------------------------------------------------------------

/// Raw section bytes of every CSR X shard, in shard order — for byte-identity
/// checks between the parallel fast path and the row-by-row `CsrEmitter`.
fn x_shard_section_bytes(path: &Path) -> Vec<Vec<u8>> {
    let r = ScxReader::open(path).unwrap();
    r.catalog()
        .shards_sorted()
        .iter()
        .map(|e| r.section_bytes(e).unwrap().to_vec())
        .collect()
}

/// Raw section bytes of every detection-bitmap shard, in catalog order — the
/// fast path builds bitmaps in parallel, so this guards bitmap-section parity
/// with the emitter (which builds them in `flush`).
fn bitmap_section_bytes(path: &Path) -> Vec<Vec<u8>> {
    let r = ScxReader::open(path).unwrap();
    r.catalog()
        .shards(SectionType::BitmapShard)
        .iter()
        .map(|e| r.section_bytes(e).unwrap().to_vec())
        .collect()
}

/// The parallel fast path (grouped `InMemory`) must be **byte-identical**
/// to the row-by-row `CsrEmitter` path. We drive the emitter via
/// `ExternalPartition` (which pushes rows through the same `CsrEmitter` with the
/// same group breaks + block sub-flush), so this is also a strategy-independence
/// check. Covered both without sub-flush (default cap) and with a small cap that
/// splits groups across multiple blocks.
#[test]
fn grouped_fast_path_byte_identical_to_emitter() {
    // Cross both the block cap (no sub-flush / splitting) and the bitmap policy
    // (Off / Always) so the parallel bitmap build is byte-compared too. The
    // fixture header is v4, so the sort output is row-group-framed — framing
    // parity is exercised in every case.
    for cap in [None, Some(120u64)] {
        for bmp in [BitmapPolicy::Off, BitmapPolicy::Always] {
            let dir = tempfile::tempdir().unwrap();
            let mut genes: Vec<&str> = vec!["nt"; 20];
            genes.extend(["MYC"; 15]);
            genes.extend(["TP53"; 12]);
            genes.extend(["GATA1"; 9]);
            let control: Vec<bool> = genes.iter().map(|g| *g == "nt").collect();
            let inp = write_grouped_fixture(&dir, "screen.scx", &genes, &control);
            let mk = || SortOptions {
                group_by: Some("target_gene".to_string()),
                reference: Some(ReferenceSpec::Labels(vec!["nt".to_string()])),
                shard_target_rows: 8,
                bitmap: bmp,
                group_write_block_bytes: cap,
                ..Default::default()
            };
            let fast = dir.path().join("fast.scx");
            sort_with_strategy(&inp, &fast, &mk(), Some(SortStrategy::InMemory)).unwrap();
            let emit = dir.path().join("emit.scx");
            sort_with_strategy(&inp, &emit, &mk(), Some(SortStrategy::ExternalPartition)).unwrap();

            let tag = format!("cap={cap:?}, bitmap={bmp:?}");
            assert_eq!(
                x_shard_section_bytes(&fast),
                x_shard_section_bytes(&emit),
                "fast-path X shard bytes must equal the emitter path ({tag})"
            );
            assert_eq!(
                bitmap_section_bytes(&fast),
                bitmap_section_bytes(&emit),
                "detection-bitmap sections must match the emitter path ({tag})"
            );
            if bmp == BitmapPolicy::Always {
                assert!(
                    !bitmap_section_bytes(&fast).is_empty(),
                    "bitmap=Always must emit bitmap sections ({tag})"
                );
            }
            assert_eq!(
                read_group_index(&fast),
                read_group_index(&emit),
                "group_index must match the emitter path ({tag})"
            );
            assert_eq!(
                csr_shard_ranges(&fast),
                csr_shard_ranges(&emit),
                "shard ranges must match the emitter path ({tag})"
            );
        }
    }
}

/// The parallel fast path is deterministic: two runs (same options) produce
/// byte-identical X shards and group index regardless of rayon scheduling.
#[test]
fn grouped_fast_path_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let mut genes: Vec<&str> = vec!["nt"; 30];
    genes.extend(["MYC"; 20]);
    genes.extend(["TP53"; 10]);
    let control: Vec<bool> = genes.iter().map(|g| *g == "nt").collect();
    let inp = write_grouped_fixture(&dir, "screen.scx", &genes, &control);
    let mk = || SortOptions {
        group_by: Some("target_gene".to_string()),
        reference: Some(ReferenceSpec::Labels(vec!["nt".to_string()])),
        shard_target_rows: 1000,
        group_write_block_bytes: Some(120), // sub-flush → many parallel blocks
        ..Default::default()
    };
    let a = dir.path().join("a.scx");
    sort(&inp, &a, &mk()).unwrap();
    let b = dir.path().join("b.scx");
    sort(&inp, &b, &mk()).unwrap();
    assert_eq!(x_shard_section_bytes(&a), x_shard_section_bytes(&b));
    assert_eq!(read_group_index(&a), read_group_index(&b));
}

/// H1: the parallel fast path's concurrency is capped by `--memory-budget` so its
/// total peak (`concurrency × per-block transient`) stays within the budget,
/// instead of scaling with core count.
#[test]
fn grouped_fast_concurrency_honors_budget() {
    use super::{grouped_fast_concurrency, GROUP_BYTES_PER_NNZ};

    // f32 encoding: per_nnz_bytes = 4 (index) + 4 (value) = 8.
    let per_nnz_bytes: u64 = 8;
    let block_byte_cap: u64 = 256 * 1024 * 1024; // default 256 MB cap
    let per_block = (2 * GROUP_BYTES_PER_NNZ) * (block_byte_cap / per_nnz_bytes); // ≈ 2× cap = 512 MB
    let threads = 192;

    // No budget → full thread count.
    assert_eq!(
        grouped_fast_concurrency(threads, block_byte_cap, per_nnz_bytes, None),
        threads
    );

    // 8 GB budget on a 192-core host → capped to budget/per_block (= 16), NOT 192.
    let budget = 8u64 * 1024 * 1024 * 1024;
    let c = grouped_fast_concurrency(threads, block_byte_cap, per_nnz_bytes, Some(budget));
    assert_eq!(c, (budget / per_block) as usize);
    assert!(c < threads, "budget must cap concurrency below core count");
    assert!(
        (c as u64) * per_block <= budget,
        "concurrency × per-block transient ({}) must fit budget ({budget})",
        (c as u64) * per_block
    );

    // Tiny budget → at least 1 block (parity with the one-block-at-a-time emitter).
    assert_eq!(
        grouped_fast_concurrency(threads, block_byte_cap, per_nnz_bytes, Some(1)),
        1
    );

    // Sub-flush disabled (cap 0) with a budget → whole-shard blocks, size unknown
    // here, so bound to one at a time (NOT left uncapped at `threads`).
    assert_eq!(
        grouped_fast_concurrency(threads, 0, per_nnz_bytes, Some(budget)),
        1
    );
    // Sub-flush disabled (cap 0) with no budget → full threads (unchanged).
    assert_eq!(
        grouped_fast_concurrency(threads, 0, per_nnz_bytes, None),
        threads
    );
}

/// T1.4 — the fast path is X-only; layers and obsp continue through the existing
/// row-by-row path and must still be reordered correctly under a grouped sort
/// (with X sub-flushing across blocks).
#[test]
fn grouped_fast_path_preserves_layers_and_obsp() {
    use scx_engine::QueryPipeline;
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_obsp_layers(&dir); // 8 obs; "raw" layer; "cell_type" obs
    let out = dir.path().join("grouped_layers.scx");
    let o = SortOptions {
        group_by: Some("cell_type".to_string()),
        group_write_block_bytes: Some(16), // force X sub-flush; layers unaffected
        ..Default::default()
    };
    sort(&inp, &out, &o).unwrap();

    // Layer "raw" is reordered row-for-row like X (matched by cell_id).
    let ri = ScxReader::open(&inp).unwrap();
    let ro = ScxReader::open(&out).unwrap();
    let li = ri.read_layer("raw").unwrap();
    let lo = ro.read_layer("raw").unwrap();
    let in_ids = str_col(&ri.read_obs().unwrap(), "cell_id");
    let out_ids = col_of(&out, "cell_id");
    let in_map: HashMap<&String, Vec<(i32, f32)>> = in_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id, csr_row(&li.indptr, &li.indices, &li.data, i)))
        .collect();
    for (k, id) in out_ids.iter().enumerate() {
        assert_eq!(
            csr_row(&lo.indptr, &lo.indices, &lo.data, k),
            in_map[id],
            "layer row mismatch at output row {k}"
        );
    }

    // X round-trips per group under the fast path.
    let pipe = QueryPipeline::open(&out).unwrap();
    for label in pipe.group_labels().unwrap() {
        let qr = pipe.read_group(&label).unwrap();
        assert!(str_col(&qr.obs, "cell_type").iter().all(|c| *c == label));
    }
}

// ---------------------------------------------------------------------------
// 1D — `scx sort --shuffle`: the seeded permutation as a third pass-0 producer
// ---------------------------------------------------------------------------

/// Shuffle-mode options. Note `by` stays empty — shuffle and `--by` are
/// mutually exclusive order sources, not a key plus a modifier.
fn shuffle_opts(seed: u64) -> SortOptions {
    SortOptions {
        by: Vec::new(),
        shuffle: Some(seed),
        shard_target_rows: 2,
        ..Default::default()
    }
}

/// The alignment test, and the one that matters most: a permutation that
/// desyncs obs from X produces a correctly *shaped* file with every row
/// mislabelled. Joining on `cell_id` is what catches it.
#[test]
fn shuffle_preserves_every_row_and_its_obs_alignment() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    let summary = sort(&inp, &out, &shuffle_opts(42)).unwrap();

    assert_eq!(summary.strategy, SortStrategy::InMemory);
    assert_eq!(summary.n_obs, 12);

    let (in_ids, in_rows) = content(&inp);
    let (out_ids, out_rows) = content(&out);

    // Row multiset identical: nothing gained, lost, or duplicated.
    assert_eq!(
        in_ids.iter().collect::<HashSet<_>>(),
        out_ids.iter().collect::<HashSet<_>>()
    );
    assert_eq!(in_ids.len(), out_ids.len());

    // ...and each cell still carries its own X row.
    let in_map: HashMap<&String, &Vec<(i32, f32)>> = in_ids.iter().zip(&in_rows).collect();
    for (id, row) in out_ids.iter().zip(&out_rows) {
        assert_eq!(in_map[id], row, "X row for {id} must survive the shuffle");
    }
}

/// The anti-tautology half of the round-trip test: every assertion above holds
/// for the identity permutation too, so a shuffle that silently became a no-op
/// would pass all of them.
#[test]
fn shuffle_actually_permutes() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    sort(&inp, &out, &shuffle_opts(42)).unwrap();

    assert_ne!(
        col_of(&inp, "cell_id"),
        col_of(&out, "cell_id"),
        "shuffle must reorder rows, not just rewrite them"
    );
}

/// The strategy-differential gate, 2-way. K-pass is excluded on purpose: it
/// emits grouped by category and cannot express an arbitrary permutation
/// (`shuffle_refuses_a_forced_kpass_strategy` pins the refusal).
#[test]
fn shuffle_strategy_differential_identical() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let mut o = shuffle_opts(42);
    o.memory_budget = Some(64); // forces the external path to actually partition

    let mut results = Vec::new();
    for strat in [SortStrategy::InMemory, SortStrategy::ExternalPartition] {
        let out = dir.path().join(format!("out_{strat:?}.scx"));
        let summary = sort_with_strategy(&inp, &out, &o, Some(strat)).unwrap();
        assert_eq!(summary.strategy, strat);
        results.push(content(&out));
    }
    assert_eq!(
        results[0], results[1],
        "in-memory vs external must produce the same shuffled file"
    );
}

/// Auto-selection must never pick K-pass in shuffle mode — there is no
/// categorical key for it to enumerate. A budget below the estimate is what
/// makes K-pass a candidate for a key sort.
#[test]
fn shuffle_auto_selection_avoids_kpass() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    let mut o = shuffle_opts(42);
    o.memory_budget = Some(64);
    let summary = sort(&inp, &out, &o).unwrap();
    assert_eq!(summary.strategy, SortStrategy::ExternalPartition);
}

#[test]
fn shuffle_refuses_a_forced_kpass_strategy() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    let mut o = shuffle_opts(42);
    o.memory_budget = Some(64);
    let err = sort_with_strategy(&inp, &out, &o, Some(SortStrategy::KPassByCategory))
        .expect_err("forced K-pass under --shuffle must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("K-pass") && msg.contains("random permutation"),
        "message should name both the strategy and why it cannot work: {msg}"
    );
}

#[test]
fn shuffle_is_deterministic_at_a_fixed_seed() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let a = dir.path().join("a.scx");
    let b = dir.path().join("b.scx");
    sort(&inp, &a, &shuffle_opts(1234)).unwrap();
    sort(&inp, &b, &shuffle_opts(1234)).unwrap();
    assert_eq!(content(&a), content(&b));
}

#[test]
fn shuffle_seed_changes_the_order() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let a = dir.path().join("a.scx");
    let b = dir.path().join("b.scx");
    sort(&inp, &a, &shuffle_opts(1234)).unwrap();
    sort(&inp, &b, &shuffle_opts(1235)).unwrap();
    assert_ne!(col_of(&a, "cell_id"), col_of(&b, "cell_id"));
}

/// The seed is the *only* record of the permutation — there is no key to
/// re-derive it from and no sidecar holding it — so a shuffled file that did
/// not carry its seed in provenance would be unreproducible.
#[test]
fn shuffle_records_its_seed_in_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    sort(&inp, &out, &shuffle_opts(7)).unwrap();

    let prov = ScxReader::open(&out).unwrap().read_provenance().unwrap();
    let entry = prov
        .operations
        .iter()
        .rev()
        .find(|e| e.action == "sort")
        .expect("shuffle records a sort provenance entry");
    let v: serde_json::Value = serde_json::from_str(&entry.params_json).unwrap();
    assert_eq!(v["shuffle"]["seed"], serde_json::json!(7));
    assert_eq!(v["by"], serde_json::json!([]));
}

// --- cross-flag rejections: reject, don't silently prefer one order source ---

#[test]
fn shuffle_rejects_a_sort_key() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    let mut o = shuffle_opts(42);
    o.by = vec!["cell_type".to_string()];
    let msg = sort(&inp, &out, &o).expect_err("must reject").to_string();
    assert!(msg.contains("--shuffle and --by"), "{msg}");
}

#[test]
fn shuffle_rejects_group_by() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    let mut o = shuffle_opts(42);
    o.group_by = Some("cell_type".to_string());
    let msg = sort(&inp, &out, &o).expect_err("must reject").to_string();
    assert!(msg.contains("--shuffle and --group-by"), "{msg}");
}

#[test]
fn shuffle_rejects_reverse() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    let mut o = shuffle_opts(42);
    o.reverse = true;
    let msg = sort(&inp, &out, &o).expect_err("must reject").to_string();
    assert!(msg.contains("--reverse is meaningless"), "{msg}");
}

// --- inherited engine behaviour, re-checked on the shuffle path -------------

/// Deletions are materialized away, exactly as for a key sort. The consequence
/// worth pinning: the permutation is over the *live* rows, so it cannot be
/// compared against a shuffle of the undeleted file.
#[test]
fn shuffle_materializes_deletions_away() {
    let dir = tempfile::tempdir().unwrap();
    let (inp, n_deleted) = fixture_deletion(&dir);
    let out = dir.path().join("out.scx");
    let summary = sort(&inp, &out, &shuffle_opts(42)).unwrap();
    assert_eq!(summary.n_obs, (12 - n_deleted) as u64);

    let clean = ScxReader::open(&out)
        .unwrap()
        .deletion_keep_mask()
        .unwrap()
        .map(|m| m.iter().all(|&k| k))
        .unwrap_or(true);
    assert!(clean, "shuffled output must be deletion-free");

    let out_ids: HashSet<String> = col_of(&out, "cell_id").into_iter().collect();
    let expected: HashSet<String> = (0..12)
        .filter(|i| ![1usize, 3, 5].contains(i))
        .map(|i| format!("cell_{i}"))
        .collect();
    assert_eq!(out_ids, expected);
}

#[test]
fn shuffle_remaps_obsp_and_carries_layers() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_obsp_layers(&dir); // 8 obs; edge r -> (r+1)%8, data r+1; "raw" layer
    let out = dir.path().join("out.scx");
    sort(&inp, &out, &shuffle_opts(42)).unwrap();

    let ro = ScxReader::open(&out).unwrap();
    let out_ids = col_of(&out, "cell_id");
    let np = new_pos_map(&out_ids, 8);

    let obsp = ro.read_obsp("connectivities").unwrap();
    let expected: HashSet<(i64, i64, u32)> = (0..8i64)
        .map(|r| {
            let c = (r + 1) % 8;
            (np[r as usize], np[c as usize], (r + 1) as u32)
        })
        .collect();
    let got: HashSet<(i64, i64, u32)> = obsp_edges(&obsp)
        .into_iter()
        .map(|(r, c, d)| (r, c, d as u32))
        .collect();
    assert_eq!(got, expected, "obsp edges remapped through the shuffle");

    // The `raw` layer follows X.
    let ri = ScxReader::open(&inp).unwrap();
    let li = ri.read_layer("raw").unwrap();
    let lo = ro.read_layer("raw").unwrap();
    let in_ids = str_col(&ri.read_obs().unwrap(), "cell_id");
    let in_map: HashMap<&String, Vec<(i32, f32)>> = in_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id, csr_row(&li.indptr, &li.indices, &li.data, i)))
        .collect();
    for (k, id) in out_ids.iter().enumerate() {
        assert_eq!(csr_row(&lo.indptr, &lo.indices, &lo.data, k), in_map[id]);
    }
}

/// Multimodal works because `sort_multimodal` consumes only `order_old` /
/// `new_pos_of_old` and never the keys — but "works by construction" is what
/// this test exists to disprove or confirm. The failure it guards is a
/// per-modality desync, which yields correctly shaped modalities whose rows
/// describe different cells.
#[test]
fn shuffle_reorders_every_modality_in_lockstep() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_multimodal(&dir); // 12 obs, rna(8) + adt(4)
    let out = dir.path().join("out.scx");
    sort(&inp, &out, &shuffle_opts(42)).unwrap();

    let ri = ScxReader::open(&inp).unwrap();
    let ro = ScxReader::open(&out).unwrap();
    assert!(ro.is_multimodal());
    assert_eq!(ro.n_modalities(), 2);

    let in_ids = str_col(&ri.read_obs().unwrap(), "cell_id");
    let out_ids = str_col(&ro.read_obs().unwrap(), "cell_id");
    assert_ne!(in_ids, out_ids, "the shuffle must have done something");

    for mid in [1u8, 2] {
        let ci = ri.read_all_csr_shards_for(mid).unwrap();
        let co = ro.read_all_csr_shards_for(mid).unwrap();
        assert_eq!(ci.shape.1, co.shape.1, "modality {mid} n_vars preserved");
        let in_map: HashMap<&String, Vec<(i32, f32)>> = in_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id, csr_row(&ci.indptr, &ci.indices, &ci.data, i)))
            .collect();
        for (k, id) in out_ids.iter().enumerate() {
            assert_eq!(
                csr_row(&co.indptr, &co.indices, &co.data, k),
                in_map[id],
                "modality {mid} X row for {id}"
            );
        }
    }
}

/// A shuffle on a sharded-obs input drives the same projected/assembled obs
/// path an atlas-scale file takes — and, unlike a key sort, reads *no* key
/// columns at all. Re-shuffling a shuffled file is the cheapest way to build
/// that input.
#[test]
fn shuffle_handles_a_sharded_obs_input() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let once = dir.path().join("once.scx");
    sort(&inp, &once, &shuffle_opts(1)).unwrap();

    let shard_count = ScxReader::open(&once).unwrap().obs_metadata_shard_count();
    assert!(
        shard_count >= 2,
        "expected a multi-shard obs input, got {shard_count}"
    );

    // A large budget keeps obs on the in-memory path (a 64-byte budget cannot
    // fit even one obs shard) while still letting a forced external X strategy
    // run — the same split `sharded_obs_input_sorts_identically` uses.
    let a = dir.path().join("a.scx");
    let b = dir.path().join("b.scx");
    let mut budgeted = shuffle_opts(2);
    budgeted.memory_budget = Some(1 << 30);
    sort(&once, &a, &shuffle_opts(2)).unwrap();
    sort_with_strategy(&once, &b, &budgeted, Some(SortStrategy::ExternalPartition)).unwrap();
    assert_eq!(content(&a), content(&b));

    // And the twice-shuffled file still holds exactly the original cells.
    assert_eq!(
        col_of(&inp, "cell_id").into_iter().collect::<HashSet<_>>(),
        col_of(&a, "cell_id").into_iter().collect::<HashSet<_>>()
    );
}

/// The codec-mix probe behind the `--shuffle` size warning. Pinning the
/// *decision* rather than the log line: the warning exists to tell a user, in
/// advance of a multi-hour rewrite, that their file is in the class that grows.
/// A probe that always answered "0 cross-row shards" would silence it forever
/// and nothing else would notice.
#[test]
fn cross_row_codec_probe_distinguishes_scx1_from_zstd() {
    use scx_codec::CodecSelection;

    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);

    for (codec, expect_cross_row) in [
        (CodecId::Scx1, false),
        (CodecId::Zstd, true),
        (CodecId::ShufDeltaZstd, true),
        (CodecId::None, false),
    ] {
        let out = dir.path().join(format!("out_{codec:?}.scx"));
        let mut o = opts(&["cell_type"]);
        o.codec = CodecSelection::Explicit(codec);
        sort(&inp, &out, &o).unwrap();

        let reader = ScxReader::open(&out).unwrap();
        let (cross_row, total, dominant) = super::cross_row_coded_shard_counts(&reader).unwrap();
        assert!(total > 0, "{codec:?}: fixture must have X shards");
        if expect_cross_row {
            assert_eq!(cross_row, total, "{codec:?} compresses across rows");
        } else {
            assert_eq!(cross_row, 0, "{codec:?} codes each row independently");
        }
        // The warning names this codec as the size-preserving `--codec` pin, so
        // a wrong answer here sends the user to a flag that reproduces the very
        // blowup being warned about.
        assert_eq!(dominant, Some(codec), "dominant codec must be reported");
    }
}
