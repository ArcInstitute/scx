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
    fixture_all_mapping_families, fixture_composite, fixture_deletion, fixture_multimodal,
    fixture_multimodal_global_mappings, fixture_null_key, fixture_numeric, fixture_obsp_layers,
    fixture_plain, fixture_skewed,
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
/// preserve the categorical dtype, its declared vocabulary and its key width
/// on read, and (b) produce content + values identical to the in-memory path.
///
/// The declared-list and key-width halves were added with §6.15: asserting
/// only `matches!(dt, Dictionary(_, _))` passed against a re-encode that
/// rebuilt the vocabulary per shard and forced every key to `Int32`.
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
    // The fixture's three shards declare `a`/`b`/`c` between them; the output
    // carries their union at the minimal key width, as the in-memory path does.
    assert_eq!(
        declared_levels(&obs, "cell_type"),
        declared_levels(
            &ScxReader::open(&mem).unwrap().read_obs().unwrap(),
            "cell_type"
        ),
    );
    assert_eq!(declared_levels(&obs, "cell_type").0, DataType::Int8);

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
fn grouped_block_byte_cap_keeps_one_block_inside_the_budget() {
    use super::{
        block_cut_bytes_per_nnz, grouped_block_byte_cap, grouped_fast_concurrency,
        ENCODE_PHASE_MULTIPLE, GROUP_BYTES_PER_NNZ,
    };
    use scx_codec::ValueEncoding;

    // `grouped_fast_concurrency_honors_budget` passes the *default* 256 MB cap,
    // so it never exercises the clamp. This drives the production pair — the
    // clamp and the concurrency cut — at every value encoding, which is where
    // the two were denominated differently: the cap counted narrowed
    // accumulator bytes and the gather held `i32 + f32`.
    let mib = 1024 * 1024;
    let default_cap = 256 * mib;
    let threads = 192;

    // Several budgets, because the two spellings differ only where the clamp
    // *binds*: at 8 GiB the default cap is the smaller of the two and even the
    // old arithmetic held, at 2 GiB the old clamp left the cap at 256 MB and a
    // single Uint8 block cost 1.2x the budget, and at 256 MiB / 1 GiB the clamp
    // is what sets the cap.
    for &budget in &[256 * mib, 1024 * mib, 2048 * mib, 8 * 1024 * mib] {
        for enc in [
            ValueEncoding::Uint8,
            ValueEncoding::Uint16,
            ValueEncoding::Float16,
            ValueEncoding::Uint32,
            ValueEncoding::Float32,
        ] {
            let per_nnz_bytes = block_cut_bytes_per_nnz(enc);
            let cap = grouped_block_byte_cap(default_cap, Some(budget), enc);
            let c = grouped_fast_concurrency(threads, cap, per_nnz_bytes, Some(budget)) as u64;
            let per_block =
                (ENCODE_PHASE_MULTIPLE * GROUP_BYTES_PER_NNZ) * (cap / per_nnz_bytes).max(1);
            assert!(
                c * per_block <= budget,
                "{enc:?} at budget {budget}: concurrency {c} x per_block {per_block} = {} \
                 exceeds it (cap {cap}, per_nnz_bytes {per_nnz_bytes})",
                c * per_block
            );
            assert!(c >= 1, "{enc:?}: concurrency must never fall below 1");
        }
    }

    // No budget → the cap is passed through, and `0` (sub-flush disabled) stays
    // `0` at every encoding rather than being floored to 1.
    let budget = 8 * 1024 * mib;
    assert_eq!(
        grouped_block_byte_cap(default_cap, None, ValueEncoding::Uint8),
        default_cap
    );
    assert_eq!(
        grouped_block_byte_cap(0, Some(budget), ValueEncoding::Uint8),
        0
    );

    // f32 is the arm the previous `budget / ENCODE_PHASE_MULTIPLE`
    // spelling got right, so it must be byte-for-byte unchanged — the clamp is
    // a narrow-encoding fix, not a re-tuning of the default path.
    assert_eq!(
        grouped_block_byte_cap(default_cap, Some(budget), ValueEncoding::Float32),
        default_cap.min(budget / ENCODE_PHASE_MULTIPLE).max(1)
    );
    // And `Uint8` is where it moved: 5/8 of the f32 cap.
    assert_eq!(
        grouped_block_byte_cap(default_cap, Some(budget), ValueEncoding::Uint8),
        default_cap.min(budget * 5 / 48).max(1)
    );
}

#[test]
fn grouped_fast_concurrency_honors_budget() {
    use super::{grouped_fast_concurrency, ENCODE_PHASE_MULTIPLE, GROUP_BYTES_PER_NNZ};

    // f32 encoding: per_nnz_bytes = 4 (index) + 4 (value) = 8.
    let per_nnz_bytes: u64 = 8;
    let block_byte_cap: u64 = 256 * 1024 * 1024; // default 256 MB cap
                                                 // The whole block phase: gather buffers + the encoder's value copy + the
                                                 // framed encode's two candidates. It was `2 *`, which charged the encode
                                                 // side at ~1x and made the `<= budget` assertion below a statement about
                                                 // the gather stage only.
    let per_block =
        (ENCODE_PHASE_MULTIPLE * GROUP_BYTES_PER_NNZ) * (block_byte_cap / per_nnz_bytes);
    let threads = 192;

    // No budget → full thread count.
    assert_eq!(
        grouped_fast_concurrency(threads, block_byte_cap, per_nnz_bytes, None),
        threads
    );

    // 8 GB budget on a 192-core host → capped to budget/per_block, NOT 192.
    // (5 at the whole-phase cost, where the gather-only model gave 16.)
    let budget = 8u64 * 1024 * 1024 * 1024;
    let c = grouped_fast_concurrency(threads, block_byte_cap, per_nnz_bytes, Some(budget));
    assert_eq!(c, (budget / per_block) as usize);
    // A **literal**, independent of the constant above. The assertion on the
    // line before re-derives `per_block` from `ENCODE_PHASE_MULTIPLE`,
    // so it moves with the production formula and cannot see a change to it —
    // measured: it passes at both 2 and 6. This is what pins the charge:
    //   per_block = 6 × 8 × (256 MiB / 8) = 1_610_612_736
    //   8 GiB / that = 5   (the gather-only 2× model gave 16)
    assert_eq!(
        c, 5,
        "concurrency at the whole-phase block cost; 16 would mean the encode \
         term is uncharged again"
    );
    assert_eq!(
        ENCODE_PHASE_MULTIPLE, 6,
        "gather (1) + the encoder's value copy (1) + the framed encode's two \
         candidates (4); see `scx-convert/src/budget.rs` for the derivation"
    );
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
        o.codec = crate::codec_intent::intent_from_codec_selection(
            scx_codec::CodecSelection::Explicit(codec),
        );
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

// --- Phase 9: the four mapping families are re-emitted as row-shards --------
//
// Why these are assertions on the *catalog* and not on content: `scx-ops`'
// carry table folds `ObsmEmbedding` and `ObsmEmbeddingShard` into one
// `SectionFamily` (`carry.rs`), so every `*_matches_the_table` test is blind to
// the difference, and every other obsp/obsm test in this crate reads through
// `read_obsp` / `read_all_obsm`, which assemble either layout transparently.
// Nothing else here can see a collapse back to one section.

/// Every `obsm/`, `varm/`, `obsp/`, `varp/` catalog entry, as
/// `(name, section_type)`, for `modality_id == 0`.
fn mapping_sections(path: &Path) -> Vec<(String, SectionType)> {
    let reader = ScxReader::open(path).unwrap();
    let mut v: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| {
            e.modality_id == 0
                && ["obsm/", "varm/", "obsp/", "varp/"]
                    .iter()
                    .any(|p| e.name.starts_with(p))
        })
        .map(|e| (e.name.clone(), e.section_type))
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

/// Assert all four families came back sharded, and name the offender if not.
fn assert_all_four_sharded(path: &Path) {
    let got = mapping_sections(path);
    for prefix in ["obsm/", "varm/", "obsp/", "varp/"] {
        assert!(
            got.iter().any(|(n, _)| n.starts_with(prefix)),
            "no {prefix} section in the output at all; got {got:?}"
        );
    }
    for (name, ty) in &got {
        assert!(
            matches!(
                ty,
                SectionType::ObsmEmbeddingShard
                    | SectionType::VarmEmbeddingShard
                    | SectionType::ObspEmbeddingShard
                    | SectionType::VarpEmbeddingShard
            ),
            "{name} came back as {ty:?}, not a row-sharded section"
        );
        assert!(
            name.contains("_shard_"),
            "{name} has a sharded section type but an unsharded section name"
        );
    }
}

#[test]
fn sort_emits_all_four_mapping_families_as_shards() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_all_mapping_families(&dir);
    let out = dir.path().join("out.scx");
    sort(&inp, &out, &opts(&["cell_type"])).unwrap();
    assert_all_four_sharded(&out);
}

#[test]
fn sort_multimodal_emits_all_four_global_families_as_shards() {
    // The modality-0 block in `sort_multimodal` is a separate call site from
    // the unimodal tail, so neither test implies the other.
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_multimodal_global_mappings(&dir);
    let out = dir.path().join("out_mm.scx");
    sort(&inp, &out, &opts(&["cell_type"])).unwrap();
    let ro = ScxReader::open(&out).unwrap();
    assert!(ro.is_multimodal());
    assert_all_four_sharded(&out);
    // Per-modality obsm is a third call site, deliberately left unsharded
    // (`write_obsm_for`), and must not have been swept into the globals.
    assert_eq!(
        ro.read_obsm_for(1, "X_umap").unwrap().num_rows(),
        12,
        "per-modality obsm still round-trips"
    );
}

#[test]
fn sort_shards_a_legacy_unsharded_input_too() {
    // The emit rule is the op's, not the input's: `fixture_multimodal_global_mappings`
    // writes all four globals through the unsharded writers. A rule that
    // preserved the input layout would leave a legacy file legacy — and it is
    // a legacy file that most needs the bounded read.
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_multimodal_global_mappings(&dir);
    let before = mapping_sections(&inp);
    assert!(
        before
            .iter()
            .any(|(_, t)| matches!(t, SectionType::ObspEmbedding | SectionType::ObsmEmbedding)),
        "premise: the input must be unsharded, got {before:?}"
    );
    let out = dir.path().join("out_legacy_in.scx");
    sort(&inp, &out, &opts(&["cell_type"])).unwrap();
    assert_all_four_sharded(&out);
}

#[test]
fn sharding_obsp_does_not_change_which_edges_survive() {
    // The content claim, stated at the strength it actually holds.
    //
    // Bucketing groups triples by `row / step` while `remap_obsp_coo` emits
    // them in input order, so the assembled batch is NOT byte-identical to the
    // pre-phase-9 single section — it is the same MULTISET of triples. No
    // reader's matrix semantics depend on that order: `BackedPairwiseReader`
    // counting-sorts and then sorts columns within a row, the h5ad exporter
    // sorts by `(row, col)` for a canonical CSR, and scipy's `coo_matrix`
    // imposes none. A caller reading raw triples through `read_obsp` and
    // relying on their sequence does see it.
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_obsp_layers(&dir); // 8 obs, ring edge r -> (r+1)%8, data r+1
    let out = dir.path().join("out.scx");
    sort(&inp, &out, &opts(&["cell_type"])).unwrap();

    let ro = ScxReader::open(&out).unwrap();
    let out_ids = col_of(&out, "cell_id");
    let np = new_pos_map(&out_ids, 8);
    let obsp = ro.read_obsp("connectivities").unwrap();

    assert_eq!(
        obsp_dim(&obsp, "n_rows"),
        8,
        "no deletions -> dim unchanged"
    );
    assert_eq!(obsp_dim(&obsp, "n_cols"), 8);
    // The extent the assembled shards declare must equal the output's own obs
    // axis, or `BackedPairwiseReader::new_obsp` refuses to open the file (it
    // requires the graph be square AND on the file's obs axis).
    assert_eq!(
        obsp_dim(&obsp, "n_rows") as u64,
        ro.n_obs(),
        "the graph's declared extent must match the output header's n_obs"
    );

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
    assert_eq!(got, expected, "every edge survives, remapped, exactly once");
    assert_eq!(
        obsp_edges(&obsp).len(),
        expected.len(),
        "and no edge is duplicated across shards"
    );
}

#[test]
fn a_sorted_graph_reads_bounded() {
    // The point of the phase. Before it, `sort` wrote one section, the layout
    // resolver produced a single entry spanning the whole axis, and every
    // `read_rows_range` decoded the entire graph.
    //
    // Asserted on the decode count (`memo_metrics`), not on wall time: this is
    // a claim about how many shards were touched, and a timing assertion would
    // be a claim about a machine.
    use scx_format_io::BackedPairwiseReader;

    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_obsp_layers(&dir);
    let out = dir.path().join("out.scx");
    sort(&inp, &out, &opts(&["cell_type"])).unwrap(); // shard_target_rows = 2
    let backed =
        BackedPairwiseReader::new_obsp(ScxReader::open(&out).unwrap(), "connectivities").unwrap();

    assert!(
        !backed.is_legacy_single_section(),
        "a sorted graph must not resolve to the legacy whole-axis layout"
    );
    assert_eq!(backed.shard_count(), 4, "8 obs at a target of 2 rows");

    // Two reads inside shard 0: one decode, then a memo hit. A single section
    // would also report one decode here — but it would be a decode of the
    // whole graph, which `shard_count` above is what distinguishes.
    backed.read_rows_range(0, 1).unwrap();
    backed.read_rows_range(1, 2).unwrap();
    assert_eq!(
        backed.memo_metrics(),
        (1, 1),
        "(hits, misses): the second read inside shard 0 must not decode again"
    );

    // A range confined to the last shard decodes that shard and no other.
    let fresh =
        BackedPairwiseReader::new_obsp(ScxReader::open(&out).unwrap(), "connectivities").unwrap();
    fresh.read_rows_range(6, 8).unwrap();
    assert_eq!(
        fresh.memo_metrics(),
        (0, 1),
        "one shard decoded for a range inside one shard, not four"
    );
}

// --- §6.15: the obs spill path must not rebuild the vocabulary -------------

/// Build a sharded-obs `.scx` carrying `cell_id` (the sort key) plus one extra
/// obs column supplied per shard, so a test can control the declared
/// vocabulary, its order, its unused levels and its value type exactly.
///
/// One CSR shard, `n_vars = 4`, three rows per obs shard. The obs shards are
/// written with `write_obs_shard` directly, which is what makes
/// `obs_metadata_shard_count() > 0` and so makes the spill path reachable.
fn write_obs_column_fixture(
    dir: &tempfile::TempDir,
    name: &str,
    col_name: &str,
    per_shard: Vec<arrow::array::ArrayRef>,
    field_metadata: HashMap<String, String>,
) -> std::path::PathBuf {
    let rows_per_shard = per_shard[0].len();
    let n_obs = per_shard.iter().map(|c| c.len()).sum::<usize>();
    let n_vars = 4usize;
    let path = dir.path().join(name);
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

    for (si, col) in per_shard.iter().enumerate() {
        let rs = si * rows_per_shard;
        // Descending ids so the sort actually reorders rows across shards.
        let ids: Vec<String> = (rs..rs + col.len())
            .map(|i| format!("cell_{:03}", n_obs - 1 - i))
            .collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new(col_name, col.data_type().clone(), true)
                .with_metadata(field_metadata.clone()),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(
                    ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                )),
                col.clone(),
            ],
        )
        .unwrap();
        w.write_obs_shard(si as u32, rs as u64, col.len() as u64, n_obs as u64, &batch)
            .unwrap();
    }
    w.finish().unwrap();
    path
}

fn str_dict(declared: &[&str], keys: &[i8]) -> arrow::array::ArrayRef {
    use arrow::array::Int8Array;
    Arc::new(
        DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(keys.to_vec()),
            Arc::new(StringArray::from(declared.to_vec())) as arrow::array::ArrayRef,
        )
        .unwrap(),
    )
}

fn ordered_metadata() -> HashMap<String, String> {
    let mut md = HashMap::new();
    md.insert(
        scx_format_io::CATEGORICAL_ORDERED_KEY.to_string(),
        "true".to_string(),
    );
    md
}

/// The declared values of a dictionary column, as strings, plus its key type.
fn declared_levels(batch: &RecordBatch, name: &str) -> (DataType, Vec<String>) {
    use arrow::array::{Array, ArrayRef};
    use arrow::datatypes::{Int16Type, Int32Type};
    let col = batch.column_by_name(name).unwrap();
    let DataType::Dictionary(key_type, _) = col.data_type() else {
        panic!("column '{name}' is {:?}, not a dictionary", col.data_type());
    };
    let any = col.as_any();
    let values: ArrayRef = if let Some(d) = any.downcast_ref::<DictionaryArray<Int8Type>>() {
        d.values().clone()
    } else if let Some(d) = any.downcast_ref::<DictionaryArray<Int16Type>>() {
        d.values().clone()
    } else if let Some(d) = any.downcast_ref::<DictionaryArray<Int32Type>>() {
        d.values().clone()
    } else {
        panic!("unexpected dictionary key width {key_type:?}");
    };
    let s = arrow::compute::cast(&values, &DataType::Utf8).unwrap();
    let s = s.as_any().downcast_ref::<StringArray>().unwrap();
    (
        (**key_type).clone(),
        (0..s.len()).map(|i| s.value(i).to_string()).collect(),
    )
}

/// Sort `inp` twice with X pinned to `InMemory` — once with no budget (obs
/// in-memory) and once with a budget small enough to force the obs spill — and
/// return `(in_memory_path, spilled_path)`. Panics unless the second actually
/// spilled, so a test can never silently assert against two in-memory runs.
fn sort_both_obs_paths(
    dir: &tempfile::TempDir,
    inp: &Path,
    by: &[&str],
) -> (std::path::PathBuf, std::path::PathBuf) {
    let bpr = obs_bytes_per_row(inp);
    let mem = dir.path().join("mem.scx");
    let mem_s = sort_with_strategy(inp, &mem, &opts(by), Some(SortStrategy::InMemory)).unwrap();
    assert!(!mem_s.obs_spilled, "no budget must keep obs in memory");

    let mut o = opts(by);
    o.memory_budget = Some(bpr * 3);
    let spilled = dir.path().join("spilled.scx");
    let sp_s = sort_with_strategy(inp, &spilled, &o, Some(SortStrategy::InMemory)).unwrap();
    assert!(
        sp_s.obs_spilled,
        "budget {} should force obs spill",
        bpr * 3
    );
    assert!(sp_s.obs_partitions >= 2, "expected multiple obs partitions");
    (mem, spilled)
}

/// A file whose `cell_type` declares `["z", "a", "m"]` — an order that is
/// neither sorted nor first-occurrence — and whose rows never reference `"z"`.
fn unused_level_fixture(dir: &tempfile::TempDir) -> std::path::PathBuf {
    write_obs_column_fixture(
        dir,
        "unused_level.scx",
        "cell_type",
        vec![
            str_dict(&["z", "a", "m"], &[1, 2, 1]),
            str_dict(&["z", "a", "m"], &[2, 1, 2]),
            str_dict(&["z", "a", "m"], &[1, 2, 1]),
        ],
        ordered_metadata(),
    )
}

/// The headline §6.15 regression test: the obs sections a spilled sort writes
/// must be exactly the ones the in-memory `take` path writes. Subsumes declared
/// levels, declared order, key width, vocabulary sharing across shards, field
/// metadata and the pandas schema envelope in one assertion.
#[test]
fn obs_spill_obs_matches_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    let inp = unused_level_fixture(&dir);
    let (mem, spilled) = sort_both_obs_paths(&dir, &inp, &["cell_id"]);

    let rm = ScxReader::open(&mem).unwrap();
    let rs = ScxReader::open(&spilled).unwrap();
    assert_eq!(
        rm.obs_metadata_shard_count(),
        rs.obs_metadata_shard_count(),
        "shard count"
    );
    assert!(rm.obs_metadata_shard_count() >= 2, "need multiple shards");
    for i in 0..rm.obs_metadata_shard_count() as u32 {
        let a = rm.read_obs_shard(i).unwrap();
        let b = rs.read_obs_shard(i).unwrap();
        assert_eq!(a.schema(), b.schema(), "obs shard {i} schema");
        assert_eq!(a, b, "obs shard {i} contents");
    }
    assert_eq!(rm.read_obs().unwrap(), rs.read_obs().unwrap());
}

/// A `pd.Categorical` level no row uses is declared on disk and must survive the
/// spill. It cannot be recovered on read — the re-encode never wrote it to any
/// shard — so this is the failure §6.15 is about.
#[test]
fn obs_spill_keeps_a_declared_but_unused_level() {
    let dir = tempfile::tempdir().unwrap();
    let inp = unused_level_fixture(&dir);
    let (_mem, spilled) = sort_both_obs_paths(&dir, &inp, &["cell_id"]);

    let obs = ScxReader::open(&spilled).unwrap().read_obs().unwrap();
    let (key_type, levels) = declared_levels(&obs, "cell_type");
    assert_eq!(
        levels,
        vec!["z", "a", "m"],
        "declared levels and their order must survive the spill"
    );
    assert_eq!(key_type, DataType::Int8, "minimal key width for 3 levels");
}

/// The `scx.categorical.ordered` stamp is only meaningful alongside the
/// declared order, and the spill used to keep the stamp while replacing the
/// order with a per-shard first-occurrence one.
#[test]
fn obs_spill_keeps_the_ordered_flag_with_its_order() {
    let dir = tempfile::tempdir().unwrap();
    let inp = unused_level_fixture(&dir);
    let (_mem, spilled) = sort_both_obs_paths(&dir, &inp, &["cell_id"]);

    let obs = ScxReader::open(&spilled).unwrap().read_obs().unwrap();
    let schema = obs.schema();
    let (_, ct) = schema.column_with_name("cell_type").unwrap();
    assert_eq!(
        ct.metadata().get(scx_format_io::CATEGORICAL_ORDERED_KEY),
        Some(&"true".to_string()),
    );
    assert_eq!(declared_levels(&obs, "cell_type").1, vec!["z", "a", "m"]);
}

/// Every output shard must declare the *same* vocabulary. The assembler unions
/// divergent per-shard lists on the way back out, so `read_obs()` cannot see
/// this — only the raw per-shard read can.
#[test]
fn obs_spill_shares_one_vocabulary_across_output_shards() {
    let dir = tempfile::tempdir().unwrap();
    let inp = unused_level_fixture(&dir);
    let (_mem, spilled) = sort_both_obs_paths(&dir, &inp, &["cell_id"]);

    let r = ScxReader::open(&spilled).unwrap();
    let n = r.obs_metadata_shard_count();
    assert!(n >= 2, "need multiple output shards, got {n}");
    for i in 0..n as u32 {
        assert_eq!(
            declared_levels(&r.read_obs_shard(i).unwrap(), "cell_type"),
            (DataType::Int8, vec!["z".into(), "a".into(), "m".into()]),
            "obs shard {i} declares a different vocabulary",
        );
    }
}

/// Arrow has no boolean dictionary packing, so the old
/// `cast(col, Dictionary(Int32, Boolean))` re-encode failed the sort outright
/// (`Unsupported output type for dictionary packing: Boolean`). Nothing packs
/// now, so a boolean categorical spills like any other.
#[test]
fn obs_spill_handles_a_boolean_categorical() {
    use arrow::array::{BooleanArray, Int8Array};
    let dir = tempfile::tempdir().unwrap();
    let bool_dict = |keys: Vec<i8>| -> arrow::array::ArrayRef {
        Arc::new(
            DictionaryArray::<Int8Type>::try_new(
                Int8Array::from(keys),
                Arc::new(BooleanArray::from(vec![true, false])) as arrow::array::ArrayRef,
            )
            .unwrap(),
        )
    };
    let inp = write_obs_column_fixture(
        &dir,
        "bool_cat.scx",
        "is_doublet",
        vec![bool_dict(vec![0, 1, 0]), bool_dict(vec![1, 1, 0])],
        HashMap::new(),
    );
    let (mem, spilled) = sort_both_obs_paths(&dir, &inp, &["cell_id"]);

    let rm = ScxReader::open(&mem).unwrap();
    let rs = ScxReader::open(&spilled).unwrap();
    for i in 0..rs.obs_metadata_shard_count() as u32 {
        assert_eq!(rm.read_obs_shard(i).unwrap(), rs.read_obs_shard(i).unwrap());
    }
    let obs = rs.read_obs().unwrap();
    assert!(matches!(
        obs.column_by_name("is_doublet").unwrap().data_type(),
        DataType::Dictionary(_, _)
    ));
}

/// Per-shard vocabularies that genuinely differ (each shard declaring levels
/// the others do not) union to one list, in first-occurrence-over-shards order,
/// shared by every output shard.
#[test]
fn obs_spill_unions_per_shard_vocabularies() {
    let dir = tempfile::tempdir().unwrap();
    let inp = write_obs_column_fixture(
        &dir,
        "per_shard_vocab.scx",
        "cell_type",
        vec![
            str_dict(&["b", "a", "unused_0"], &[0, 1, 0]),
            str_dict(&["c", "a"], &[0, 1, 0]),
            str_dict(&["unused_2", "b"], &[1, 1, 1]),
        ],
        HashMap::new(),
    );
    let (mem, spilled) = sort_both_obs_paths(&dir, &inp, &["cell_id"]);

    let rs = ScxReader::open(&spilled).unwrap();
    let want: Vec<String> = ["b", "a", "unused_0", "c", "unused_2"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    for i in 0..rs.obs_metadata_shard_count() as u32 {
        assert_eq!(
            declared_levels(&rs.read_obs_shard(i).unwrap(), "cell_type"),
            (DataType::Int8, want.clone()),
            "obs shard {i}",
        );
    }
    let rm = ScxReader::open(&mem).unwrap();
    assert_eq!(rm.read_obs().unwrap(), rs.read_obs().unwrap());
}

/// A file whose first shard is a dictionary and whose later shards store the
/// column plain — the shape an `append` from a plain source onto a
/// dictionary-encoded base leaves. The plain shard's values join the union and
/// its rows come back keyed against it.
#[test]
fn obs_spill_promotes_a_plain_shard_against_a_dictionary_base() {
    let dir = tempfile::tempdir().unwrap();
    let plain: arrow::array::ArrayRef =
        Arc::new(StringArray::from(vec!["a", "appended", "appended"]));
    let inp = write_obs_column_fixture(
        &dir,
        "mixed.scx",
        "cell_type",
        vec![str_dict(&["b", "a", "unused"], &[0, 1, 0]), plain],
        HashMap::new(),
    );
    let (mem, spilled) = sort_both_obs_paths(&dir, &inp, &["cell_id"]);

    let rs = ScxReader::open(&spilled).unwrap();
    let obs = rs.read_obs().unwrap();
    assert_eq!(
        declared_levels(&obs, "cell_type").1,
        vec!["b", "a", "unused", "appended"],
    );
    assert_eq!(ScxReader::open(&mem).unwrap().read_obs().unwrap(), obs);
}

/// Negative control: the ops stay representation-preserving. A plain-`Utf8` obs
/// column spills and comes back plain — never promoted to a categorical — and
/// still matches the in-memory path shard for shard.
#[test]
fn obs_spill_leaves_a_plain_column_plain() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let sharded = dir.path().join("sharded.scx");
    sort(&inp, &sharded, &opts(&["cell_type"])).unwrap();
    let (mem, spilled) = sort_both_obs_paths(&dir, &sharded, &["cell_type"]);

    let rm = ScxReader::open(&mem).unwrap();
    let rs = ScxReader::open(&spilled).unwrap();
    for i in 0..rs.obs_metadata_shard_count() as u32 {
        let b = rs.read_obs_shard(i).unwrap();
        assert!(
            b.schema()
                .fields()
                .iter()
                .all(|f| !matches!(f.data_type(), DataType::Dictionary(_, _))),
            "obs shard {i} promoted a plain column to a categorical",
        );
        assert_eq!(rm.read_obs_shard(i).unwrap(), b);
    }
}

/// The mirror of [`obs_spill_promotes_a_plain_shard_against_a_dictionary_base`]:
/// the **first** obs shard stores the column plain and a later one stores it as
/// a dictionary.
///
/// Deciding which columns are categorical from shard 0 alone got this
/// direction wrong — the spill wrote the column plain while `read_obs()`
/// promotes a field any shard declares a dictionary, so the two writers
/// disagreed on a layout the docs say is reachable. Found by
/// **codex - gpt-5.6-sol** on PR #547.
#[test]
fn obs_spill_promotes_when_only_a_later_shard_is_a_dictionary() {
    let dir = tempfile::tempdir().unwrap();
    let plain: arrow::array::ArrayRef = Arc::new(StringArray::from(vec!["a", "base", "base"]));
    let inp = write_obs_column_fixture(
        &dir,
        "plain_first.scx",
        "cell_type",
        vec![plain, str_dict(&["b", "a", "unused"], &[0, 1, 0])],
        HashMap::new(),
    );
    let (mem, spilled) = sort_both_obs_paths(&dir, &inp, &["cell_id"]);

    let rm = ScxReader::open(&mem).unwrap();
    let rs = ScxReader::open(&spilled).unwrap();
    let obs = rs.read_obs().unwrap();
    assert!(
        matches!(
            obs.column_by_name("cell_type").unwrap().data_type(),
            DataType::Dictionary(_, _)
        ),
        "a column any shard declares categorical must come out categorical",
    );
    assert_eq!(
        declared_levels(&obs, "cell_type").1,
        vec!["a", "base", "b", "unused"],
        "the plain first shard's values lead the union, then the later shard's",
    );
    for i in 0..rs.obs_metadata_shard_count() as u32 {
        assert_eq!(rm.read_obs_shard(i).unwrap(), rs.read_obs_shard(i).unwrap());
    }
}

/// Two or more **plain** shards ahead of the first dictionary shard.
///
/// The pairwise fold carries the accumulator forward as a 0-row slice, which is
/// only lossless once a column is dictionary-typed: an arrow slice of a
/// `DictionaryArray` keeps its values array, a slice of a plain array keeps
/// nothing. With a plain run at the front there was no dictionary in the pair
/// yet, so `reconcile_dictionary_representations` had nothing to promote
/// against and every plain shard but the last one before the first dictionary
/// lost its values — which pass 1 then found again, tripping the
/// "gained categories while spilling" guard. Found by
/// **Antigravity - Gemini 3.8 Flash** on PR #547.
#[test]
fn obs_spill_keeps_values_from_a_plain_run_before_the_first_dictionary() {
    let dir = tempfile::tempdir().unwrap();
    let plain0: arrow::array::ArrayRef = Arc::new(StringArray::from(vec!["p0", "shared", "p0b"]));
    let plain1: arrow::array::ArrayRef = Arc::new(StringArray::from(vec!["p1", "shared", "p1b"]));
    let inp = write_obs_column_fixture(
        &dir,
        "plain_run.scx",
        "cell_type",
        vec![
            plain0,
            plain1,
            str_dict(&["d", "shared", "unused"], &[0, 1, 0]),
        ],
        HashMap::new(),
    );
    let (mem, spilled) = sort_both_obs_paths(&dir, &inp, &["cell_id"]);

    let rs = ScxReader::open(&spilled).unwrap();
    let obs = rs.read_obs().unwrap();
    assert_eq!(
        declared_levels(&obs, "cell_type").1,
        vec!["p0", "shared", "p0b", "p1", "p1b", "d", "unused"],
        "every shard's values must reach the union, in read_obs() order",
    );
    let rm = ScxReader::open(&mem).unwrap();
    for i in 0..rs.obs_metadata_shard_count() as u32 {
        assert_eq!(rm.read_obs_shard(i).unwrap(), rs.read_obs_shard(i).unwrap());
    }
}

/// A later obs shard carrying a column the spill schema does not name is
/// rejected, not silently dropped. Pins the half of the by-name correction that
/// the reverse-mix test does not cover. Found by
/// **Cursor Agent - Grok 4.6 High** on PR #547.
#[test]
fn obs_spill_rejects_a_shard_with_an_extra_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("extra_col.scx");
    let (n_obs, n_vars) = (6usize, 4usize);
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
    w.write_var(
        &RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "gene_id",
                DataType::Utf8,
                false,
            )])),
            vec![Arc::new(StringArray::from(
                (0..n_vars).map(|i| format!("g{i}")).collect::<Vec<_>>(),
            ))],
        )
        .unwrap(),
    )
    .unwrap();
    for si in 0..2usize {
        let rs = si * 3;
        let ids: Vec<String> = (rs..rs + 3)
            .map(|i| format!("cell_{:03}", n_obs - 1 - i))
            .collect();
        let mut fields = vec![Field::new("cell_id", DataType::Utf8, false)];
        let mut cols: Vec<arrow::array::ArrayRef> = vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))];
        if si == 1 {
            fields.push(Field::new("surprise", DataType::Utf8, true));
            cols.push(Arc::new(StringArray::from(vec!["x", "y", "z"])));
        }
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap();
        w.write_obs_shard(si as u32, rs as u64, 3, n_obs as u64, &batch)
            .unwrap();
    }
    w.finish().unwrap();

    let bpr = obs_bytes_per_row(&path);
    let mut o = opts(&["cell_id"]);
    o.memory_budget = Some(bpr * 3);
    let out = dir.path().join("out.scx");
    let err = sort_with_strategy(&path, &out, &o, Some(SortStrategy::InMemory)).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("the spill schema names"),
        "expected the column-count rejection, got: {msg}"
    );
}

/// A categorical whose **values are a nested type** (`Dictionary(_, List<Utf8>)`).
///
/// Raised on PR #547 by **codex - gpt-5.6-sol** as a case the seeded fold would
/// lose, on the premise that `intern_declared_values` declines nested value
/// types and so leaves the empty seed as the accumulator. It does not:
/// `RowConverter::supports_fields` is `true` for `List`, `LargeList`,
/// `FixedSizeList` and `Struct`, so the union is interned exactly as for a
/// string. Of the nested types only `Map` is declined — and there neither
/// writer works, because the decode→re-encode fallback both would fall back to
/// raises `Unsupported output type for dictionary packing: Map(...)`. Pinned
/// here so the supported half stays covered.
#[test]
fn obs_spill_handles_a_nested_dictionary_value_type() {
    use arrow::array::{Int32Array, ListArray};
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::Int32Type;

    let dir = tempfile::tempdir().unwrap();
    let list_dict = |keys: Vec<i32>| -> arrow::array::ArrayRef {
        let inner = StringArray::from(vec!["a", "b", "c", "d"]);
        let values = ListArray::new(
            Arc::new(Field::new("item", DataType::Utf8, true)),
            OffsetBuffer::new(vec![0, 2, 3, 4].into()),
            Arc::new(inner) as arrow::array::ArrayRef,
            None,
        );
        Arc::new(
            DictionaryArray::<Int32Type>::try_new(
                Int32Array::from(keys),
                Arc::new(values) as arrow::array::ArrayRef,
            )
            .unwrap(),
        )
    };
    let inp = write_obs_column_fixture(
        &dir,
        "nested_dict.scx",
        "tags",
        vec![list_dict(vec![0, 1, 0]), list_dict(vec![1, 1, 0])],
        HashMap::new(),
    );
    let (mem, spilled) = sort_both_obs_paths(&dir, &inp, &["cell_id"]);

    let rm = ScxReader::open(&mem).unwrap();
    let rs = ScxReader::open(&spilled).unwrap();
    assert!(matches!(
        rs.read_obs()
            .unwrap()
            .column_by_name("tags")
            .unwrap()
            .data_type(),
        DataType::Dictionary(_, _)
    ));
    for i in 0..rs.obs_metadata_shard_count() as u32 {
        assert_eq!(rm.read_obs_shard(i).unwrap(), rs.read_obs_shard(i).unwrap());
    }
}

/// The count-mismatch rejection's *missing* arm, the mirror of
/// [`obs_spill_rejects_a_shard_with_an_extra_column`]. Noted as untested by
/// **Cursor Agent - Grok 4.6 High** on PR #547.
#[test]
fn obs_spill_rejects_a_shard_with_a_missing_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("missing_col.scx");
    let (n_obs, n_vars) = (6usize, 4usize);
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
    w.write_var(
        &RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "gene_id",
                DataType::Utf8,
                false,
            )])),
            vec![Arc::new(StringArray::from(
                (0..n_vars).map(|i| format!("g{i}")).collect::<Vec<_>>(),
            ))],
        )
        .unwrap(),
    )
    .unwrap();
    // Shard 0 has two columns, shard 1 only `cell_id`.
    for si in 0..2usize {
        let rs = si * 3;
        let ids: Vec<String> = (rs..rs + 3)
            .map(|i| format!("cell_{:03}", n_obs - 1 - i))
            .collect();
        let mut fields = vec![Field::new("cell_id", DataType::Utf8, false)];
        let mut cols: Vec<arrow::array::ArrayRef> = vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))];
        if si == 0 {
            fields.push(Field::new("extra", DataType::Utf8, true));
            cols.push(Arc::new(StringArray::from(vec!["x", "y", "z"])));
        }
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap();
        w.write_obs_shard(si as u32, rs as u64, 3, n_obs as u64, &batch)
            .unwrap();
    }
    w.finish().unwrap();

    let bpr = obs_bytes_per_row(&path);
    let mut o = opts(&["cell_id"]);
    o.memory_budget = Some(bpr * 3);
    let out = dir.path().join("out.scx");
    let err = sort_with_strategy(&path, &out, &o, Some(SortStrategy::InMemory)).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("missing: [\"extra\"]"),
        "expected the missing-column arm of the rejection, got: {msg}"
    );
}
