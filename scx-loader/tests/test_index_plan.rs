//! Integration tests for `scx_loader::IndexPlanLoader`.
//!
//! Covers:
//! - `process_plan` shape correctness (HVG on/off).
//! - Normalize + log1p semantics.
//! - Empty plan + out-of-range index error paths.
//! - sort_by_shard: pairs/rows alignment + same-plan-different-order parity.
//! - HVG-projected paired scatter end-to-end via process_plan.

mod common;

use common::{
    open_loader, open_loader_hvg, open_loader_normalized, open_loader_with_flags,
    write_multi_shard_fixture,
};
use scx_loader::LoaderError;

const N_OBS: usize = 100;
const N_VARS: usize = 50;
const N_SHARDS: usize = 5;

fn fixture(dir: &tempfile::TempDir) -> std::path::PathBuf {
    write_multi_shard_fixture(&dir.path().join("fixture.scx"), N_OBS, N_VARS, N_SHARDS)
}

/// Spec: "process_plan on a 100-cell test fixture, plan [(0,1),(2,3),(4,5)],
/// output shapes [3, n_cols] and [3, n_cols]."
#[test]
fn process_plan_basic_shape() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);
    let loader = open_loader(&path, /*sort_by_shard=*/ false);

    let plan = vec![(0u64, 1u64), (2, 3), (4, 5)];
    let batch = loader.process_plan(plan.clone()).unwrap();

    assert_eq!(batch.pairs.len(), 3);
    assert_eq!(batch.x.len(), 3 * loader.n_output_cols());
    assert_eq!(batch.x_paired.len(), 3 * loader.n_output_cols());
    assert_eq!(batch.pairs, plan); // sort_by_shard off → input order preserved
    assert_eq!(loader.n_output_cols(), N_VARS);
    // obs is gathered for both sides
    assert!(batch.obs.contains_key("cell_id"));
    assert!(batch.obs_paired.contains_key("cell_id"));
}

/// Spec: "HVG projection on/off: when hvg_indices=Some([0,5,10]), output
/// cols == 3; when None, output cols == n_vars."
#[test]
fn process_plan_hvg_projection_shapes() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    let loader_none = open_loader(&path, false);
    assert_eq!(loader_none.n_output_cols(), N_VARS);
    let b_none = loader_none.process_plan(vec![(0, 1)]).unwrap();
    assert_eq!(b_none.x.len(), N_VARS);

    let loader_hvg = open_loader_hvg(&path, vec![0, 5, 10], false);
    assert_eq!(loader_hvg.n_output_cols(), 3);
    let b_hvg = loader_hvg.process_plan(vec![(0, 1)]).unwrap();
    assert_eq!(b_hvg.x.len(), 3);
    assert_eq!(b_hvg.x_paired.len(), 3);
}

/// Spec: "Normalize+log1p on/off: row sums match target_sum after normalize;
/// log1p applied if flag set."
///
/// In our fixture each row has exactly one nonzero, so post-normalize that
/// nonzero equals `target_sum` (the row sum), and post-log1p it equals
/// `ln(1 + target_sum)`.
#[test]
fn process_plan_normalize_log1p_semantics() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);
    let target_sum = 1.0e4_f64;

    let loader = open_loader_normalized(&path, /*sort_by_shard=*/ false, target_sum);
    let batch = loader.process_plan(vec![(7u64, 8u64)]).unwrap();

    // Row 7's single nonzero is at col 7%50=7 with raw value 8.
    // After normalize: 8 × (1e4 / 8) = 1e4. After log1p: ln(1 + 1e4).
    let row = &batch.x[..N_VARS];
    let expected = (1.0f64 + target_sum).ln() as f32;
    let nonzero_count = row.iter().filter(|&&v| v != 0.0).count();
    assert_eq!(nonzero_count, 1, "exactly one nonzero per row");
    let nonzero = row[7 % N_VARS];
    assert!(
        (nonzero - expected).abs() < 1e-3,
        "log1p(target_sum) ≈ {expected}, got {nonzero}"
    );
}

/// Spec: `(normalize=true, log1p=false)` — row sum equals `target_sum`,
/// no log1p applied. Regression test for the bug where `IndexPlanLoader`
/// always called `fused_normalize_log1p_dense` whenever `normalize=true`,
/// silently applying log1p that the caller did not request.
#[test]
fn process_plan_normalize_only_no_log1p() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);
    let target_sum = 1.0e4_f64;

    let loader = open_loader_with_flags(
        &path, /*normalize=*/ true, /*log1p=*/ false, target_sum,
        /*sort_by_shard=*/ false,
    );
    // Pert row 7 -> nonzero at col 7, raw value 8.
    // Ctrl row 8 -> nonzero at col 8, raw value 9.
    let batch = loader.process_plan(vec![(7u64, 8u64)]).unwrap();

    let p_row = &batch.x[..N_VARS];
    let c_row = &batch.x_paired[..N_VARS];

    // Both pert and ctrl rows should sum to target_sum (normalize only).
    let p_sum: f64 = p_row.iter().map(|&v| v as f64).sum();
    let c_sum: f64 = c_row.iter().map(|&v| v as f64).sum();
    assert!(
        (p_sum - target_sum).abs() < 1e-2,
        "pert row sum {p_sum} != target {target_sum}"
    );
    assert!(
        (c_sum - target_sum).abs() < 1e-2,
        "ctrl row sum {c_sum} != target {target_sum}"
    );

    // Single nonzero per row, equal to target_sum (no log1p).
    assert_eq!(p_row.iter().filter(|&&v| v != 0.0).count(), 1);
    assert_eq!(c_row.iter().filter(|&&v| v != 0.0).count(), 1);
    assert!((p_row[7 % N_VARS] as f64 - target_sum).abs() < 1e-2);
    assert!((c_row[8 % N_VARS] as f64 - target_sum).abs() < 1e-2);
}

/// Spec: `(normalize=false, log1p=true)` — `ln(1 + raw)` per element, no
/// scaling. Regression test for the bug where `IndexPlanLoader` skipped
/// the transform entirely when `normalize=false`, returning raw rows even
/// though the caller requested log1p (the state-scx ST default).
#[test]
fn process_plan_log1p_only_no_normalize() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    let loader = open_loader_with_flags(
        &path, /*normalize=*/ false, /*log1p=*/ true, /*target_sum=*/ 0.0,
        /*sort_by_shard=*/ false,
    );
    // Pert row 7 -> col 7, raw value 8 -> ln(9).
    // Ctrl row 8 -> col 8, raw value 9 -> ln(10).
    let batch = loader.process_plan(vec![(7u64, 8u64)]).unwrap();

    let p_row = &batch.x[..N_VARS];
    let c_row = &batch.x_paired[..N_VARS];

    let p_expected = ((8.0_f32) + 1.0).ln();
    let c_expected = ((9.0_f32) + 1.0).ln();

    assert!(
        (p_row[7 % N_VARS] - p_expected).abs() < 1e-5,
        "pert log1p mismatch: got {} expected {}",
        p_row[7 % N_VARS],
        p_expected
    );
    assert!(
        (c_row[8 % N_VARS] - c_expected).abs() < 1e-5,
        "ctrl log1p mismatch: got {} expected {}",
        c_row[8 % N_VARS],
        c_expected
    );
    // Other columns should be ln(0+1) = 0 — but we wrote zeros into the
    // dense buffer, so the log1p applies element-wise and yields exactly 0.
    for (i, &v) in p_row.iter().enumerate() {
        if i != 7 % N_VARS {
            assert!(v.abs() < 1e-7);
        }
    }
}

/// Spec: `(normalize=false, log1p=false)` — pass-through, raw values.
#[test]
fn process_plan_no_normalize_no_log1p_returns_raw() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    let loader = open_loader(&path, /*sort_by_shard=*/ false);
    let batch = loader.process_plan(vec![(7u64, 8u64)]).unwrap();

    let p_row = &batch.x[..N_VARS];
    let c_row = &batch.x_paired[..N_VARS];

    assert_eq!(p_row[7 % N_VARS], 8.0);
    assert_eq!(c_row[8 % N_VARS], 9.0);
    assert_eq!(p_row.iter().filter(|&&v| v != 0.0).count(), 1);
    assert_eq!(c_row.iter().filter(|&&v| v != 0.0).count(), 1);
}

/// Spec: "Empty plan: yields a zero-row batch."
#[test]
fn process_plan_empty_yields_zero_row_batch() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);
    let loader = open_loader(&path, false);

    let batch = loader.process_plan(Vec::new()).unwrap();
    assert!(batch.x.is_empty());
    assert!(batch.x_paired.is_empty());
    assert!(batch.pairs.is_empty());
    assert!(batch.obs.is_empty());
    assert!(batch.obs_paired.is_empty());
}

/// Spec: "Out-of-range index: returns IndexError-shaped error."
#[test]
fn process_plan_out_of_range_returns_index_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);
    let loader = open_loader(&path, false);

    // pert side OOR
    let r = loader.process_plan(vec![(N_OBS as u64 + 5, 0)]);
    match r {
        Err(LoaderError::IndexOutOfRange { idx, n_obs }) => {
            assert_eq!(idx, N_OBS as u64 + 5);
            assert_eq!(n_obs, N_OBS);
        }
        Err(other) => panic!("expected IndexOutOfRange, got {other}"),
        Ok(_) => panic!("expected error, got Ok"),
    }

    // ctrl side OOR
    let r = loader.process_plan(vec![(0, N_OBS as u64 + 1)]);
    assert!(matches!(r, Err(LoaderError::IndexOutOfRange { .. })));
}

/// Spec: "Sort-by-shard preserves pair semantics: the post-sort pairs field,
/// when iterated in returned order, equals row order in x / x_paired."
///
/// Our fixture's row `r` has a single nonzero at col `r % n_vars` with value
/// `((r+1) & 0xFF)`. We pin the alignment by checking each row's reconstructed
/// (col, value) against `pairs[i]`.
#[test]
fn sort_by_shard_pairs_align_with_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);
    let loader = open_loader(&path, /*sort_by_shard=*/ true);
    let n_cols = loader.n_output_cols();

    // Deliberately scrambled across shards.
    let plan: Vec<(u64, u64)> = vec![
        (95, 0),
        (40, 50),
        (80, 90),
        (3, 12),
        (60, 11),
        (1, 2),
        (25, 35),
    ];
    let batch = loader.process_plan(plan).unwrap();

    for i in 0..batch.pairs.len() {
        let (p, c) = batch.pairs[i];
        let p_row = &batch.x[i * n_cols..(i + 1) * n_cols];
        let c_row = &batch.x_paired[i * n_cols..(i + 1) * n_cols];

        let p_col = (p as usize) % N_VARS;
        let c_col = (c as usize) % N_VARS;
        assert_eq!(p_row[p_col], ((p as usize + 1) & 0xFF) as f32);
        assert_eq!(c_row[c_col], ((c as usize + 1) & 0xFF) as f32);
        // Single nonzero per row.
        assert_eq!(p_row.iter().filter(|&&v| v != 0.0).count(), 1);
        assert_eq!(c_row.iter().filter(|&&v| v != 0.0).count(), 1);
    }
}

/// Spec: "same-plan-different-order parity invariant (output equality up to
/// row permutation when sort_by_shard is on vs off)."
#[test]
fn sort_by_shard_is_pure_permutation() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    let plan: Vec<(u64, u64)> = vec![(95, 0), (40, 50), (80, 90), (3, 12), (60, 11), (1, 2)];

    let loader_unsorted = open_loader(&path, false);
    let loader_sorted = open_loader(&path, true);

    let b_un = loader_unsorted.process_plan(plan.clone()).unwrap();
    let b_so = loader_sorted.process_plan(plan.clone()).unwrap();

    let n_cols = loader_unsorted.n_output_cols();
    let triples = |b: &scx_loader::IndexPlanBatch| -> Vec<((u64, u64), Vec<u32>, Vec<u32>)> {
        (0..b.pairs.len())
            .map(|i| {
                let r: Vec<u32> = b.x[i * n_cols..(i + 1) * n_cols]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                let pr: Vec<u32> = b.x_paired[i * n_cols..(i + 1) * n_cols]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                (b.pairs[i], r, pr)
            })
            .collect()
    };
    let mut un = triples(&b_un);
    let mut so = triples(&b_so);
    un.sort_by_key(|t| t.0);
    so.sort_by_key(|t| t.0);
    assert_eq!(un, so);
}

/// HVG-projected paired scatter via `process_plan` reproduces the expected
/// `(col, value)` tuple per row on both sides of the pair, exercising the
/// `read_rows_with` + `HvgProjection::scatter_row` dense-gather path.
#[test]
fn scatter_pair_rows_via_process_plan() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);
    // Hits HVG positions {0, 1, 2} via cols {7, 17, 27} for rows 7, 17, 27.
    let hvg = vec![7u32, 17, 27];
    let loader = open_loader_hvg(&path, hvg.clone(), false);

    let plan = vec![(7u64, 17u64), (27, 7), (17, 27)];
    let batch = loader.process_plan(plan.clone()).unwrap();

    // For row 7 → col 7, value 8 → HVG position 0.
    let n_cols = 3;
    for i in 0..plan.len() {
        let (p, c) = plan[i];
        let p_row = &batch.x[i * n_cols..(i + 1) * n_cols];
        let c_row = &batch.x_paired[i * n_cols..(i + 1) * n_cols];

        let p_pos = hvg
            .iter()
            .position(|&g| g as usize == (p as usize) % N_VARS);
        let c_pos = hvg
            .iter()
            .position(|&g| g as usize == (c as usize) % N_VARS);

        match p_pos {
            Some(pos) => {
                assert_eq!(p_row[pos], ((p as usize + 1) & 0xFF) as f32);
                assert_eq!(p_row.iter().filter(|&&v| v != 0.0).count(), 1);
            }
            None => assert!(p_row.iter().all(|&v| v == 0.0)),
        }
        match c_pos {
            Some(pos) => {
                assert_eq!(c_row[pos], ((c as usize + 1) & 0xFF) as f32);
                assert_eq!(c_row.iter().filter(|&&v| v != 0.0).count(), 1);
            }
            None => assert!(c_row.iter().all(|&v| v == 0.0)),
        }
    }
}

/// Constructor rejects `cache_shards < 1`.
#[test]
fn ctor_rejects_zero_cache_shards() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    let mut config = scx_loader::LoaderConfig::default();
    config.max_memory_mb = 1024;
    let r = scx_loader::IndexPlanLoader::new(&path, config, 0, true, 4, 1024);
    assert!(matches!(r, Err(LoaderError::ConfigError { .. })));
}

/// Constructor surfaces missing-obs-column as the typed `ObsColumnNotFound`
/// variant carrying the requested name and the available columns. Python
/// wrapper maps this to KeyError; that's covered in the Python integration
/// tests.
#[test]
fn ctor_rejects_missing_obs_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    let mut config = scx_loader::LoaderConfig::default();
    config.obs_columns = vec!["nonexistent_column".to_string()];
    config.max_memory_mb = 1024;
    let r = scx_loader::IndexPlanLoader::new(&path, config, 4, true, 4, 1024);
    match r {
        Err(LoaderError::ObsColumnNotFound { name, available }) => {
            assert_eq!(name, "nonexistent_column");
            assert!(
                available.iter().any(|c| c == "cell_id"),
                "available list should include the fixture's cell_id column; got {available:?}"
            );
        }
        Err(other) => panic!("expected ObsColumnNotFound, got {other}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

/// Constructor rejects HVG indices >= n_vars.
#[test]
fn ctor_rejects_hvg_out_of_range() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    let mut config = scx_loader::LoaderConfig::default();
    config.hvg_indices = Some(vec![0, 5, N_VARS as u32]); // last one is OOR
    config.max_memory_mb = 1024;
    let r = scx_loader::IndexPlanLoader::new(&path, config, 4, true, 4, 1024);
    match r {
        Err(LoaderError::ConfigError { reason }) => {
            assert!(reason.contains("HVG index"));
            assert!(reason.contains("out of range"));
        }
        Err(other) => panic!("expected ConfigError, got {other}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}
