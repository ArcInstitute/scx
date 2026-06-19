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
use crate::sort::{SortOptions, SortStrategy};
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
