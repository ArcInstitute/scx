//! Ground-truth correctness oracle for the obs predicate engine, plus a
//! differential check of the two evaluation paths against each other.
//!
//! Every generated predicate is checked **three** ways:
//!
//! 1. against an independent per-row three-valued (Kleene) evaluator written in
//!    this file, which never calls `scx-engine` — the ground truth;
//! 2. on the row-set fast path (resolve indexed obs predicates straight from
//!    the predicate index, no obs-shard decode); and
//! 3. on the legacy full-decode-and-mask path, forced with
//!    `SCX_DISABLE_ROWSET_PUSHDOWN`.
//!
//! Layers 2 and 3 must agree with each other (X, obs, var, count, exists,
//! across a range of `limit` values) *and* with layer 1. The differential half
//! alone can only prove path-parity, and that is not enough: measured on the
//! commit before this one, layers 2 and 3 agreed on every generated predicate
//! while both dropped rows where `null OR true` should have matched — the
//! row-set evaluator had been deliberately restricted (`Or` refused on any
//! nullable column) so the fast path would reproduce the slow path's wrong
//! answer. The independent oracle is what catches a bug that lives in BOTH.
//!
//! The fixture deliberately includes: a null-bearing categorical (`cell_type`
//! is NULL on every 7th row — the `null OR true` case, and the
//! null/complement reasoning that keeps `Ne`/`Not` on the residual path), an
//! absent categorical value, a non-indexed column and a numeric column (both
//! forcing the residual path), and obs/CSR shard boundaries that COINCIDE (the
//! atlas case).
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
use scx_format_io::DeletionVectors;
use tempfile::TempDir;

const N_OBS: u64 = 40;
const N_VARS: usize = 6;
const ROWS_PER_SHARD: usize = 8; // 5 shards; obs shards == CSR shards
const N_SHARDS: usize = 5;

/// Deleted global rows (exercises the deletion-vector path in BOTH the row-set
/// `deletion_rowset` and the legacy obs_mask loop). Written below as CSR-shard
/// -local deletions: shard 0 local {1,3} -> global {1,3}; shard 2 local {0,2}
/// -> global {16,18}.
const DELETED_GLOBAL: &[usize] = &[1, 3, 16, 18];

fn is_deleted(i: usize) -> bool {
    DELETED_GLOBAL.contains(&i)
}

// ---- Ground-truth obs columns (deterministic) ----

fn cell_type_at(i: usize) -> Option<&'static str> {
    // Spread across shards; every 7th row is NULL (exercises null handling).
    if i.is_multiple_of(7) {
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
    if i.is_multiple_of(2) {
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

    // Deletion vectors as v2 global obs rows (== DELETED_GLOBAL).
    let mut dv = DeletionVectors::new();
    dv.insert_global(DELETED_GLOBAL.iter().map(|&i| i as u32));
    writer.write_deletion_vectors(&dv).unwrap();

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

/// A generated predicate, rendered two ways: to the engine's expression syntax
/// (`render`) and to a per-row truth value (`eval`). Keeping one AST behind
/// both means the oracle and the engine are always asked the same question.
#[derive(Clone)]
enum Expr {
    /// Indexed nullable categorical — `null` on every 7th row.
    CellTypeEq(&'static str),
    /// Indexed non-null categorical, via `in [...]`.
    TissueIn(&'static str, &'static str),
    /// Non-indexed non-null categorical → residual path.
    QualityEq(&'static str),
    /// Non-null numeric → residual path.
    NGenesGt(i64),
    NGenesLe(i64),
    /// `Ne` on the nullable categorical → residual path.
    CellTypeNe(&'static str),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
}

impl Expr {
    fn render(&self) -> String {
        match self {
            Self::CellTypeEq(v) => format!("cell_type == '{v}'"),
            Self::TissueIn(a, b) => format!("tissue in ['{a}', '{b}']"),
            Self::QualityEq(v) => format!("quality == '{v}'"),
            Self::NGenesGt(n) => format!("n_genes > {n}"),
            Self::NGenesLe(n) => format!("n_genes <= {n}"),
            Self::CellTypeNe(v) => format!("cell_type != '{v}'"),
            Self::And(a, b) => format!("({}) and ({})", a.render(), b.render()),
            Self::Or(a, b) => format!("({}) or ({})", a.render(), b.render()),
            Self::Not(a) => format!("not ({})", a.render()),
        }
    }

    /// Three-valued (Kleene / SQL) truth of this expression for obs row `i`,
    /// computed straight from the fixture's ground-truth column functions.
    /// `None` is UNKNOWN: a comparison against a NULL cell.
    ///
    /// This is a deliberately independent implementation — it must not call
    /// into `scx-engine`, or it would only prove the engine agrees with itself.
    fn eval(&self, i: usize) -> Option<bool> {
        match self {
            Self::CellTypeEq(v) => cell_type_at(i).map(|c| c == *v),
            Self::CellTypeNe(v) => cell_type_at(i).map(|c| c != *v),
            Self::TissueIn(a, b) => {
                let t = tissue_at(i);
                Some(t == *a || t == *b)
            }
            Self::QualityEq(v) => Some(quality_at(i) == *v),
            Self::NGenesGt(n) => Some(n_genes_at(i) > *n),
            Self::NGenesLe(n) => Some(n_genes_at(i) <= *n),
            // AND is FALSE as soon as either side is FALSE, even if the other
            // is UNKNOWN; TRUE only when both are TRUE.
            Self::And(a, b) => match (a.eval(i), b.eval(i)) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            },
            // OR is TRUE as soon as either side is TRUE, even if the other is
            // UNKNOWN — this is the case the non-Kleene kernel got wrong.
            Self::Or(a, b) => match (a.eval(i), b.eval(i)) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            },
            Self::Not(a) => a.eval(i).map(|v| !v),
        }
    }

    /// Truth of this expression for row `i` under arrow's **non-Kleene**
    /// kernels — the pre-fix behaviour, where `and`/`or` return UNKNOWN
    /// whenever *either* operand is UNKNOWN. Leaves and `not` are identical to
    /// [`Self::eval`]; only the two combinators differ.
    ///
    /// This exists solely so the generator's coverage canary can measure the
    /// thing that actually matters — expressions the two semantics answer
    /// *differently* — rather than the much weaker "expression happens to match
    /// a row that has a NULL somewhere in it".
    fn eval_non_kleene(&self, i: usize) -> Option<bool> {
        match self {
            Self::And(a, b) => match (a.eval_non_kleene(i), b.eval_non_kleene(i)) {
                (Some(x), Some(y)) => Some(x && y),
                _ => None,
            },
            Self::Or(a, b) => match (a.eval_non_kleene(i), b.eval_non_kleene(i)) {
                (Some(x), Some(y)) => Some(x || y),
                _ => None,
            },
            Self::Not(a) => a.eval_non_kleene(i).map(|v| !v),
            leaf => leaf.eval(i),
        }
    }

    /// True if Kleene and non-Kleene disagree about whether *any* row matches —
    /// i.e. this expression would have been answered wrong before the fix.
    fn is_kleene_sensitive(&self) -> bool {
        (0..N_OBS as usize)
            .any(|i| (self.eval(i) == Some(true)) != (self.eval_non_kleene(i) == Some(true)))
    }
}

/// The rows a `filter_obs(expr)` must return: UNKNOWN is not a match (the
/// top-level `WHERE`-clause coalesce), and deleted rows never surface.
fn expected_ids(e: &Expr) -> Vec<String> {
    (0..N_OBS as usize)
        .filter(|&i| e.eval(i) == Some(true) && !is_deleted(i))
        .map(|i| format!("cell_{i}"))
        .collect()
}

/// A leaf comparison over one of the columns. Mixes indexed (cell_type, tissue,
/// n_genes) and non-indexed (quality) columns, plus `Ne` and numeric ops that
/// route to the residual path.
fn gen_leaf(rng: &mut Lcg) -> Expr {
    match rng.below(6) {
        0 => Expr::CellTypeEq(CELL_TYPES[rng.below(CELL_TYPES.len() as u64)]),
        1 => Expr::TissueIn(
            TISSUES[rng.below(TISSUES.len() as u64)],
            TISSUES[rng.below(TISSUES.len() as u64)],
        ),
        2 => Expr::QualityEq(QUALITIES[rng.below(QUALITIES.len() as u64)]),
        3 => Expr::NGenesGt(100 + rng.below(900) as i64),
        4 => Expr::CellTypeNe(CELL_TYPES[rng.below(CELL_TYPES.len() as u64)]),
        _ => Expr::NGenesLe(100 + rng.below(900) as i64),
    }
}

fn gen_expr(rng: &mut Lcg, depth: u32) -> Expr {
    if depth == 0 {
        return gen_leaf(rng);
    }
    match rng.below(5) {
        0 | 1 => gen_leaf(rng),
        2 => Expr::And(
            Box::new(gen_expr(rng, depth - 1)),
            Box::new(gen_expr(rng, depth - 1)),
        ),
        3 => Expr::Or(
            Box::new(gen_expr(rng, depth - 1)),
            Box::new(gen_expr(rng, depth - 1)),
        ),
        _ => Expr::Not(Box::new(gen_expr(rng, depth - 1))),
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
        // cell_type == 'T cell': rows where i%7!=0 and i%3==0, MINUS deleted rows
        // (cell_3, cell_18 are deleted T cells) — proves deletion_rowset applies.
        let expected: Vec<String> = (0..N_OBS as usize)
            .filter(|&i| cell_type_at(i) == Some("T cell") && !is_deleted(i))
            .map(|i| format!("cell_{i}"))
            .collect();
        let got = run(&path, "cell_type == 'T cell'", None);
        let got_ids: Vec<String> = got.cell_ids.into_iter().flatten().collect();
        assert_eq!(
            got_ids, expected,
            "ground-truth check for cell_type == 'T cell' (deletions applied)"
        );
        assert!(
            !got_ids.contains(&"cell_3".to_string()) && !got_ids.contains(&"cell_18".to_string()),
            "deleted T cells must be excluded on the row-set path"
        );

        // A deleted B cell (cell_1) must be absent too.
        let b = run(&path, "cell_type == 'B cell'", None);
        let b_ids: Vec<String> = b.cell_ids.into_iter().flatten().collect();
        assert!(
            !b_ids.contains(&"cell_1".to_string()) && !b_ids.contains(&"cell_16".to_string()),
            "deleted B cells must be excluded on the row-set path"
        );

        // Absent value -> empty.
        assert_eq!(run(&path, "cell_type == 'Ghost'", None).n_rows, 0);
    }

    let limits = [None, Some(1usize), Some(5), Some(13), Some(1000)];
    let mut rng = Lcg(0xDEAD_BEEF_1234_5678);
    let mut checked = 0u32;
    let mut null_sensitive = 0u32;

    for _ in 0..600 {
        let ast = gen_expr(&mut rng, 3);
        let expr = ast.render();
        // Ground truth from the independent Kleene evaluator.
        let want = expected_ids(&ast);
        // How many generated expressions this oracle would answer differently
        // under the two semantics — i.e. how many actually exercise the bug.
        // Deliberately NOT "matches a row that has a NULL somewhere": a bare
        // `n_genes > 100` matching row 0 satisfies that while being completely
        // insensitive to null handling, so such a counter could stay green even
        // if `Or`-over-`cell_type` generation disappeared entirely.
        if ast.is_kleene_sensitive() {
            null_sensitive += 1;
        }

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

            // ...and both must match ground truth, not merely each other.
            let got: Vec<String> = fast.cell_ids.iter().flatten().cloned().collect();
            match limit {
                None => {
                    assert_eq!(got, want, "ground-truth mismatch for `{expr}`");
                    assert_eq!(
                        fast.matched_rows,
                        want.len(),
                        "matched_rows mismatch for `{expr}`"
                    );
                }
                // A limit truncates the (ascending) match list; it must never
                // reorder it or admit a row ground truth excludes.
                Some(n) => {
                    assert_eq!(
                        got,
                        want[..got.len().min(want.len())],
                        "limited result is not a prefix of ground truth for `{expr}` limit={limit:?}"
                    );
                    // A prefix check alone would let an empty result pass
                    // vacuously — `[]` is a prefix of everything. A non-empty
                    // ground truth with a non-zero limit must return rows, and
                    // at least min(limit, |want|) of them.
                    assert!(
                        got.len() >= n.min(want.len()),
                        "limit={n} returned {} rows but ground truth has {} \
                         for `{expr}`",
                        got.len(),
                        want.len()
                    );
                    assert!(
                        got.len() <= n,
                        "limit={n} but got {} rows for `{expr}`",
                        got.len()
                    );
                }
            }
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
        assert_eq!(fc, want.len(), "count() vs ground truth for `{expr}`");
        assert_eq!(
            fe,
            !want.is_empty(),
            "exists() vs ground truth for `{expr}`"
        );

        checked += 1;
    }

    assert!(
        checked > 3000,
        "expected a substantial number of comparisons"
    );
    assert!(
        null_sensitive > 50,
        "only {null_sensitive} generated expressions are answered differently by \
         Kleene vs non-Kleene logic; the semantics this oracle exists to pin \
         would be barely exercised"
    );
}
