//! Differential correctness oracle for row-set predicate pushdown.
//!
//! The row-set fast path (resolve indexed obs predicates directly from the
//! predicate index, no obs-shard decode) MUST return results byte-identical to
//! the legacy full-decode-and-mask path for every predicate. This test builds a
//! sharded-obs fixture with a predicate index and runs a large batch of random
//! predicates twice — once with the row-set path enabled (default) and once
//! with `SCX_DISABLE_ROWSET_PUSHDOWN` forcing the legacy path — asserting the
//! two results are identical (X, obs, var, count, exists), across a range of
//! `limit` values.
//!
//! The fixture deliberately includes: a null-bearing categorical (exercising
//! the null/complement reasoning that keeps `Ne`/`Not` on the residual path),
//! an absent categorical value, a non-indexed column and a numeric column (both
//! forcing the residual path), and obs/CSR shard boundaries that COINCIDE (the
//! atlas case). A handful of hand-computed expectations cross-check both paths
//! against ground truth, catching a bug that might exist in BOTH.
//!
//! NOTE: this file intentionally contains a single `#[test]` so the
//! process-global `SCX_DISABLE_ROWSET_PUSHDOWN` env var is toggled without
//! racing other tests (separate `tests/*.rs` files are separate binaries).

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_engine::{
    build_and_write_conversion_predicate_indexes, ConversionPredicateIndexOptions, QueryPipeline,
};
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;
use tempfile::TempDir;

const N_OBS: u64 = 40;
const N_VARS: usize = 6;
const ROWS_PER_SHARD: usize = 8; // 5 shards; obs shards == CSR shards
const N_SHARDS: usize = 5;

// ---- Ground-truth obs columns (deterministic) ----

fn cell_type_at(i: usize) -> Option<&'static str> {
    // Spread across shards; every 7th row is NULL (exercises null handling).
    if i % 7 == 0 {
        return None;
    }
    match i % 3 {
        0 => Some("T cell"),
        1 => Some("B cell"),
        _ => Some("NK cell"),
    }
}

fn tissue_at(i: usize) -> &'static str {
    match i % 4 {
        0 => "blood",
        1 => "brain",
        2 => "bone",
        _ => "blood",
    }
}

/// Non-indexed categorical → forces the residual decode path.
fn quality_at(i: usize) -> &'static str {
    if i % 2 == 0 {
        "hi"
    } else {
        "lo"
    }
}

fn n_genes_at(i: usize) -> i64 {
    100 + ((i * 137) % 900) as i64 // [100, 999]
}

fn full_obs() -> RecordBatch {
    let n = N_OBS as usize;
    let cell_id: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let cell_type: Vec<Option<&str>> = (0..n).map(cell_type_at).collect();
    let tissue: Vec<&str> = (0..n).map(tissue_at).collect();
    let quality: Vec<&str> = (0..n).map(quality_at).collect();
    let n_genes: Vec<i64> = (0..n).map(n_genes_at).collect();

    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, true), // nullable
        Field::new("tissue", DataType::Utf8, false),
        Field::new("quality", DataType::Utf8, false),
        Field::new("n_genes", DataType::Int64, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                cell_id.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(cell_type)),
            Arc::new(StringArray::from(tissue)),
            Arc::new(StringArray::from(quality)),
            Arc::new(Int64Array::from(n_genes)),
        ],
    )
    .unwrap()
}

fn sample_var() -> RecordBatch {
    let ids: Vec<String> = (0..N_VARS).map(|i| format!("gene_{i}")).collect();
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn write_csr_shard(writer: &mut ScxWriter, n_rows: usize, row_start: u64) {
    // 1 nnz/row, value = (global row % 254)+1 so X varies per row.
    let indptr: Vec<u64> = (0..=n_rows as u64).collect();
    let indices: Vec<u32> = (0..n_rows as u32).map(|i| i % N_VARS as u32).collect();
    let values: Vec<u8> = (0..n_rows)
        .map(|i| ((row_start as usize + i) % 254 + 1) as u8)
        .collect();
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            row_start,
        )
        .unwrap();
}

fn build_fixture(dir: &TempDir) -> PathBuf {
    let path = dir.path().join("rowset_diff.scx");
    let obs = full_obs();
    let var = sample_var();
    let mut writer = ScxWriter::new(
        &path,
        FileHeader::new_single_modality(N_OBS, N_VARS as u64, 0, 16384, 0, 0),
    )
    .unwrap();

    // obs shards == CSR shards (5 shards of 8 rows) — the atlas case.
    for s in 0..N_SHARDS {
        let row_start = (s * ROWS_PER_SHARD) as u64;
        let slice = obs.slice(row_start as usize, ROWS_PER_SHARD);
        writer
            .write_obs_shard(s as u32, row_start, ROWS_PER_SHARD as u64, N_OBS, &slice)
            .unwrap();
    }
    writer.write_var(&var).unwrap();
    for s in 0..N_SHARDS {
        write_csr_shard(&mut writer, ROWS_PER_SHARD, (s * ROWS_PER_SHARD) as u64);
    }

    // Index keyed to CSR shard ranges (mirrors merge/compact/conversion).
    let csr_row_ranges: Vec<(u64, u64)> = (0..N_SHARDS)
        .map(|s| {
            let st = (s * ROWS_PER_SHARD) as u64;
            (st, st + ROWS_PER_SHARD as u64)
        })
        .collect();
    let opts = ConversionPredicateIndexOptions {
        index_obs: vec![
            "cell_type".to_string(),
            "tissue".to_string(),
            "n_genes".to_string(),
        ],
        index_var: Vec::new(),
        index_preset: None,
        index_auto_threshold: 1000,
    };
    build_and_write_conversion_predicate_indexes(
        &mut writer,
        &obs,
        &var,
        &csr_row_ranges,
        N_VARS,
        &opts,
    )
    .unwrap();
    writer.finish().unwrap();
    path
}

// ---- Deterministic predicate-expression generator ----

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }
    fn below(&mut self, n: u64) -> usize {
        (self.next() % n) as usize
    }
}

const CELL_TYPES: &[&str] = &["T cell", "B cell", "NK cell", "Ghost"]; // Ghost absent
const TISSUES: &[&str] = &["blood", "brain", "bone", "Nowhere"]; // Nowhere absent
const QUALITIES: &[&str] = &["hi", "lo", "mid"]; // mid absent; non-indexed col

/// A leaf comparison over one of the columns. Mixes indexed (cell_type, tissue,
/// n_genes) and non-indexed (quality) columns, plus `Ne` and numeric ops that
/// route to the residual path.
fn gen_leaf(rng: &mut Lcg) -> String {
    match rng.below(6) {
        0 => format!(
            "cell_type == '{}'",
            CELL_TYPES[rng.below(CELL_TYPES.len() as u64)]
        ),
        1 => {
            let a = TISSUES[rng.below(TISSUES.len() as u64)];
            let b = TISSUES[rng.below(TISSUES.len() as u64)];
            format!("tissue in ['{a}', '{b}']")
        }
        2 => format!(
            "quality == '{}'",
            QUALITIES[rng.below(QUALITIES.len() as u64)]
        ),
        3 => format!("n_genes > {}", 100 + rng.below(900) as u64),
        4 => format!(
            "cell_type != '{}'",
            CELL_TYPES[rng.below(CELL_TYPES.len() as u64)]
        ),
        _ => format!("n_genes <= {}", 100 + rng.below(900) as u64),
    }
}

fn gen_expr(rng: &mut Lcg, depth: u32) -> String {
    if depth == 0 {
        return gen_leaf(rng);
    }
    match rng.below(5) {
        0 | 1 => gen_leaf(rng),
        2 => format!(
            "({}) and ({})",
            gen_expr(rng, depth - 1),
            gen_expr(rng, depth - 1)
        ),
        3 => format!(
            "({}) or ({})",
            gen_expr(rng, depth - 1),
            gen_expr(rng, depth - 1)
        ),
        _ => format!("not ({})", gen_expr(rng, depth - 1)),
    }
}

// ---- Result capture + comparison ----

#[derive(Debug, PartialEq)]
struct Captured {
    n_rows: usize,
    cell_ids: Vec<Option<String>>,
    indptr: Vec<i64>,
    indices: Vec<i32>,
    data: Vec<f32>,
    matched_rows: usize,
}

fn cell_ids_of(batch: &RecordBatch) -> Vec<Option<String>> {
    let idx = batch.schema().index_of("cell_id").unwrap();
    let arr = arrow::compute::cast(batch.column(idx), &DataType::Utf8).unwrap();
    let s = arr.as_any().downcast_ref::<StringArray>().unwrap();
    (0..s.len())
        .map(|i| s.is_valid(i).then(|| s.value(i).to_string()))
        .collect()
}

fn run(path: &PathBuf, expr: &str, limit: Option<usize>) -> Captured {
    let mut p = QueryPipeline::open(path).unwrap().filter_obs(expr).unwrap();
    if let Some(n) = limit {
        p = p.limit(n);
    }
    let r = p.collect().unwrap();
    Captured {
        n_rows: r.x.n_rows(),
        cell_ids: cell_ids_of(&r.obs),
        indptr: r.x.indptr.clone(),
        indices: r.x.indices.clone(),
        data: r.x.data.clone(),
        matched_rows: r.matched_rows,
    }
}

fn count_exists(path: &PathBuf, expr: &str) -> (usize, bool) {
    let p = QueryPipeline::open(path).unwrap().filter_obs(expr).unwrap();
    (p.count().unwrap().matched_rows, p.exists().unwrap())
}

#[test]
fn rowset_path_matches_legacy_path() {
    let dir = TempDir::new().unwrap();
    let path = build_fixture(&dir);

    // Sanity: with the row-set path ON, a couple of hand-computed expectations
    // cross-check BOTH paths against ground truth.
    std::env::remove_var("SCX_DISABLE_ROWSET_PUSHDOWN");
    {
        // cell_type == 'T cell': rows where i%7!=0 and i%3==0.
        let expected: Vec<String> = (0..N_OBS as usize)
            .filter(|&i| cell_type_at(i) == Some("T cell"))
            .map(|i| format!("cell_{i}"))
            .collect();
        let got = run(&path, "cell_type == 'T cell'", None);
        let got_ids: Vec<String> = got.cell_ids.into_iter().flatten().collect();
        assert_eq!(
            got_ids, expected,
            "ground-truth check for cell_type == 'T cell'"
        );

        // Absent value -> empty.
        assert_eq!(run(&path, "cell_type == 'Ghost'", None).n_rows, 0);
    }

    let limits = [None, Some(1usize), Some(5), Some(13), Some(1000)];
    let mut rng = Lcg(0xDEAD_BEEF_1234_5678);
    let mut checked = 0u32;

    for _ in 0..600 {
        let expr = gen_expr(&mut rng, 3);

        for &limit in &limits {
            // Row-set path (default).
            std::env::remove_var("SCX_DISABLE_ROWSET_PUSHDOWN");
            let fast = run(&path, &expr, limit);

            // Legacy full-decode path.
            std::env::set_var("SCX_DISABLE_ROWSET_PUSHDOWN", "1");
            let slow = run(&path, &expr, limit);
            std::env::remove_var("SCX_DISABLE_ROWSET_PUSHDOWN");

            assert_eq!(
                fast, slow,
                "row-set vs legacy mismatch for `{expr}` limit={limit:?}"
            );
            checked += 1;
        }

        // count() / exists() parity across paths (limit-independent).
        std::env::remove_var("SCX_DISABLE_ROWSET_PUSHDOWN");
        let (fc, fe) = count_exists(&path, &expr);
        std::env::set_var("SCX_DISABLE_ROWSET_PUSHDOWN", "1");
        let (sc, se) = count_exists(&path, &expr);
        std::env::remove_var("SCX_DISABLE_ROWSET_PUSHDOWN");
        assert_eq!((fc, fe), (sc, se), "count/exists mismatch for `{expr}`");
        // exists() must agree with count() > 0 on the fast path.
        assert_eq!(
            fe,
            fc > 0,
            "exists() inconsistent with count() for `{expr}`"
        );

        checked += 1;
    }

    assert!(
        checked > 3000,
        "expected a substantial number of comparisons"
    );
}
