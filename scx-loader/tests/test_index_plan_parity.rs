//! Phase 6.2 — parity tests for `IndexPlanLoader::process_plan` against
//! manual `read_row_indices` + scatter + normalize.
//!
//! For each (pert_idx, ctrl_idx) pair in a plan, gather the row by:
//!   1. The full pipeline: `IndexPlanLoader::process_plan(plan)`.
//!   2. Manually: `BackedCsrReader::read_row_indices(&[idx])` → CSR row →
//!      `scatter_row_full` (or `scatter_row`) → optional
//!      `fused_normalize_log1p_dense`.
//!
//! The two paths must agree to within `rtol=1e-5, atol=1e-6` (per spec). For
//! our deterministic fixture they are in fact bit-identical because both
//! paths call the same primitives, but the spec's allclose tolerance is the
//! contract.

mod common;

use common::{write_known_multinnz_fixture, write_multi_shard_fixture, KNOWN_MULTINNZ_N_VARS};
use scx_format_io::{BackedCsrReader, ScxReader};
use scx_loader::{
    fused_normalize_log1p_dense_with_depth, scatter_row_full, HvgProjection, IndexPlanLoader,
    LoaderConfig,
};

const N_OBS: usize = 200;
const N_VARS: usize = 64;
const N_SHARDS: usize = 4;

/// Allclose check matching the spec's tolerance.
fn allclose(a: &[f32], b: &[f32], rtol: f32, atol: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b)
        .all(|(x, y)| (x - y).abs() <= atol + rtol * y.abs())
}

/// Open a fresh `BackedCsrReader` for manual parity computations. Independent
/// of the `IndexPlanLoader` under test (separate file handle).
fn open_backed(path: &std::path::Path) -> BackedCsrReader {
    let reader = ScxReader::open(path).unwrap();
    BackedCsrReader::new(reader, /*cache_shards=*/ 4)
}

/// Manually compute the expected dense row vector for a given source row,
/// going through the same primitives `process_plan` uses.
fn manual_dense_row(
    backed: &BackedCsrReader,
    row: u64,
    n_output_cols: usize,
    hvg: Option<&HvgProjection>,
    normalize: bool,
    target_sum: f64,
) -> Vec<f32> {
    let csr = backed.read_row_indices(&[row]).unwrap();
    let lo = csr.indptr[0] as usize;
    let hi = csr.indptr[1] as usize;
    let mut out = vec![0f32; n_output_cols];
    match hvg {
        Some(hvg) => hvg.scatter_row(&csr.indices[lo..hi], &csr.data[lo..hi], &mut out),
        None => scatter_row_full(&csr.indices[lo..hi], &csr.data[lo..hi], &mut out).unwrap(),
    }
    if normalize {
        // Normalize by the FULL pre-projection row depth (matches the loader's
        // corrected semantics), NOT the panel-local sum of `out`.
        let depth: f64 = csr.data[lo..hi].iter().map(|&v| v as f64).sum();
        fused_normalize_log1p_dense_with_depth(&mut out, target_sum, depth);
    }
    out
}

fn build_loader(
    path: &std::path::Path,
    hvg: Option<Vec<u32>>,
    normalize: bool,
    target_sum: f64,
    sort_by_shard: bool,
) -> IndexPlanLoader {
    let config = LoaderConfig {
        normalize,
        log1p: normalize,
        target_sum,
        hvg_indices: hvg,
        obs_columns: vec!["cell_id".to_string()],
        max_memory_mb: 1024,
        ..Default::default()
    };
    IndexPlanLoader::new(path, config, 4, sort_by_shard, 4, 16384).unwrap()
}

/// Spec: full-gene path with no normalize. process_plan rows match manual.
#[test]
fn parity_full_gene_no_normalize() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), N_OBS, N_VARS, N_SHARDS);

    // sort_by_shard=false so batch.pairs[i] aligns with input plan[i].
    let loader = build_loader(&path, None, false, 1e4, false);
    let backed = open_backed(&path);

    let plan: Vec<(u64, u64)> = vec![(0, 5), (50, 99), (123, 1), (180, 75), (10, 130), (199, 0)];
    let batch = loader.process_plan(plan.clone()).unwrap();
    let n_cols = loader.n_output_cols();

    for (i, &(p, c)) in plan.iter().enumerate() {
        let p_actual = &batch.x[i * n_cols..(i + 1) * n_cols];
        let c_actual = &batch.x_paired[i * n_cols..(i + 1) * n_cols];

        let p_expected = manual_dense_row(&backed, p, n_cols, None, false, 0.0);
        let c_expected = manual_dense_row(&backed, c, n_cols, None, false, 0.0);

        assert!(
            allclose(p_actual, &p_expected, 1e-5, 1e-6),
            "pert row {p} mismatch at plan index {i}"
        );
        assert!(
            allclose(c_actual, &c_expected, 1e-5, 1e-6),
            "ctrl row {c} mismatch at plan index {i}"
        );
    }
}

/// Spec: full-gene path WITH normalize+log1p. process_plan rows match manual.
#[test]
fn parity_full_gene_normalize_log1p() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), N_OBS, N_VARS, N_SHARDS);

    let target_sum = 1.0e4_f64;
    let loader = build_loader(&path, None, true, target_sum, false);
    let backed = open_backed(&path);

    let plan: Vec<(u64, u64)> = vec![(0, 5), (50, 99), (123, 1), (180, 75)];
    let batch = loader.process_plan(plan.clone()).unwrap();
    let n_cols = loader.n_output_cols();

    for (i, &(p, c)) in plan.iter().enumerate() {
        let p_actual = &batch.x[i * n_cols..(i + 1) * n_cols];
        let c_actual = &batch.x_paired[i * n_cols..(i + 1) * n_cols];

        let p_expected = manual_dense_row(&backed, p, n_cols, None, true, target_sum);
        let c_expected = manual_dense_row(&backed, c, n_cols, None, true, target_sum);

        assert!(
            allclose(p_actual, &p_expected, 1e-5, 1e-6),
            "pert row {p} (normalized) mismatch"
        );
        assert!(
            allclose(c_actual, &c_expected, 1e-5, 1e-6),
            "ctrl row {c} (normalized) mismatch"
        );
    }
}

/// Spec: HVG-projected path. process_plan rows match manual `scatter_row` +
/// optional normalize.
#[test]
fn parity_hvg_projected_no_normalize() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), N_OBS, N_VARS, N_SHARDS);

    let hvg_indices = vec![0u32, 5, 10, 17, 23, 30, 39, 50, 60];
    let hvg = HvgProjection::new(hvg_indices.clone(), N_VARS as u64).unwrap();
    let n_hvg = hvg.n_output_cols();

    let loader = build_loader(&path, Some(hvg_indices), false, 1e4, false);
    let backed = open_backed(&path);
    assert_eq!(loader.n_output_cols(), n_hvg);

    let plan: Vec<(u64, u64)> = vec![(7, 17), (30, 60), (100, 50), (190, 0), (45, 70)];
    let batch = loader.process_plan(plan.clone()).unwrap();

    for (i, &(p, c)) in plan.iter().enumerate() {
        let p_actual = &batch.x[i * n_hvg..(i + 1) * n_hvg];
        let c_actual = &batch.x_paired[i * n_hvg..(i + 1) * n_hvg];

        let p_expected = manual_dense_row(&backed, p, n_hvg, Some(&hvg), false, 0.0);
        let c_expected = manual_dense_row(&backed, c, n_hvg, Some(&hvg), false, 0.0);

        assert!(
            allclose(p_actual, &p_expected, 1e-5, 1e-6),
            "HVG pert row {p} mismatch at plan index {i}"
        );
        assert!(
            allclose(c_actual, &c_expected, 1e-5, 1e-6),
            "HVG ctrl row {c} mismatch at plan index {i}"
        );
    }
}

/// Spec: HVG-projected + normalize+log1p. Most expressive parity check.
#[test]
fn parity_hvg_projected_normalize_log1p() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), N_OBS, N_VARS, N_SHARDS);

    let hvg_indices = vec![0u32, 5, 10, 17, 23, 30, 39, 50, 60];
    let hvg = HvgProjection::new(hvg_indices.clone(), N_VARS as u64).unwrap();
    let n_hvg = hvg.n_output_cols();

    let target_sum = 5.0e3_f64;
    let loader = build_loader(&path, Some(hvg_indices), true, target_sum, false);
    let backed = open_backed(&path);

    let plan: Vec<(u64, u64)> = vec![(7, 17), (30, 60), (100, 50), (190, 0)];
    let batch = loader.process_plan(plan.clone()).unwrap();

    for (i, &(p, c)) in plan.iter().enumerate() {
        let p_actual = &batch.x[i * n_hvg..(i + 1) * n_hvg];
        let c_actual = &batch.x_paired[i * n_hvg..(i + 1) * n_hvg];

        let p_expected = manual_dense_row(&backed, p, n_hvg, Some(&hvg), true, target_sum);
        let c_expected = manual_dense_row(&backed, c, n_hvg, Some(&hvg), true, target_sum);

        assert!(
            allclose(p_actual, &p_expected, 1e-5, 1e-6),
            "HVG+norm pert row {p} mismatch"
        );
        assert!(
            allclose(c_actual, &c_expected, 1e-5, 1e-6),
            "HVG+norm ctrl row {c} mismatch"
        );
    }
}

/// L2 regression: with an HVG projection, `normalize_total` must use the cell's
/// FULL transcriptome depth as the denominator (scanpy's normalize-then-subset),
/// NOT the panel-local sum. Uses a fixture where every row has mass both inside
/// and outside the panel, so the two depths genuinely differ — the fixture in
/// the other tests has one nonzero per row, which cannot exercise this.
#[test]
fn parity_hvg_normalize_uses_full_depth_not_panel() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_known_multinnz_fixture(&dir.path().join("f.scx"), 40, 4);

    // Panel captures genes 2 and 7 (present in every row) but not 33 or 58, so
    // panel-local depth = v2+v7 is strictly below full depth v2+v7+v33+v58.
    let hvg_indices = vec![2u32, 7, 40];
    let hvg = HvgProjection::new(hvg_indices.clone(), KNOWN_MULTINNZ_N_VARS).unwrap();
    let n_hvg = hvg.n_output_cols();

    let target_sum = 1.0e4_f64;
    let loader = build_loader(&path, Some(hvg_indices), true, target_sum, false);
    let backed = open_backed(&path);

    let plan: Vec<(u64, u64)> = vec![(0, 5), (13, 27), (38, 2), (19, 31)];
    let batch = loader.process_plan(plan.clone()).unwrap();

    let mut saw_divergence = false;
    for (i, &(p, c)) in plan.iter().enumerate() {
        let p_actual = &batch.x[i * n_hvg..(i + 1) * n_hvg];
        let c_actual = &batch.x_paired[i * n_hvg..(i + 1) * n_hvg];

        // Correct reference: normalize by full transcriptome depth.
        let p_expected = manual_dense_row(&backed, p, n_hvg, Some(&hvg), true, target_sum);
        let c_expected = manual_dense_row(&backed, c, n_hvg, Some(&hvg), true, target_sum);

        assert!(
            allclose(p_actual, &p_expected, 1e-5, 1e-6),
            "pert row {p}: loader must match full-depth normalize"
        );
        assert!(
            allclose(c_actual, &c_expected, 1e-5, 1e-6),
            "ctrl row {c}: loader must match full-depth normalize"
        );

        // The (incorrect) panel-local normalize must differ — otherwise the
        // test would be vacuous and wouldn't catch a regression.
        let panel_local = panel_local_normalize(&backed, p, n_hvg, &hvg, target_sum);
        if !allclose(p_actual, &panel_local, 1e-5, 1e-6) {
            saw_divergence = true;
        }
    }
    assert!(
        saw_divergence,
        "fixture/panel failed to exercise full-depth vs panel-local divergence"
    );
}

/// The buggy panel-local computation: scatter into the HVG panel, then normalize
/// using the PANEL's own sum as depth. Used only to prove the fix diverges from it.
fn panel_local_normalize(
    backed: &BackedCsrReader,
    row: u64,
    n_output_cols: usize,
    hvg: &HvgProjection,
    target_sum: f64,
) -> Vec<f32> {
    let csr = backed.read_row_indices(&[row]).unwrap();
    let lo = csr.indptr[0] as usize;
    let hi = csr.indptr[1] as usize;
    let mut out = vec![0f32; n_output_cols];
    hvg.scatter_row(&csr.indices[lo..hi], &csr.data[lo..hi], &mut out);
    let panel_sum: f64 = out.iter().map(|&v| v as f64).sum();
    fused_normalize_log1p_dense_with_depth(&mut out, target_sum, panel_sum);
    out
}

/// Sort-by-shard reorders the plan; `pairs` reflects the post-sort order, so
/// `pairs[i]` (NOT input plan[i]) names the row that sits in `x[i]`. Verify
/// parity in this realigned indexing.
#[test]
fn parity_sorted_uses_post_sort_pairs() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), N_OBS, N_VARS, N_SHARDS);

    let loader = build_loader(&path, None, false, 1e4, /*sort_by_shard=*/ true);
    let backed = open_backed(&path);

    // Deliberately scrambled across shards.
    let plan: Vec<(u64, u64)> = vec![(199, 0), (150, 50), (100, 1), (49, 100), (10, 150)];
    let batch = loader.process_plan(plan).unwrap();
    let n_cols = loader.n_output_cols();

    for (i, &(p, c)) in batch.pairs.iter().enumerate() {
        let p_actual = &batch.x[i * n_cols..(i + 1) * n_cols];
        let c_actual = &batch.x_paired[i * n_cols..(i + 1) * n_cols];

        let p_expected = manual_dense_row(&backed, p, n_cols, None, false, 0.0);
        let c_expected = manual_dense_row(&backed, c, n_cols, None, false, 0.0);

        assert!(allclose(p_actual, &p_expected, 1e-5, 1e-6));
        assert!(allclose(c_actual, &c_expected, 1e-5, 1e-6));
    }
}

/// Iterator path: `iter_with_plans` yields the same per-plan batches as
/// repeated `process_plan` calls.
#[test]
fn parity_iter_vs_process_plan() {
    use std::sync::Arc;
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), N_OBS, N_VARS, N_SHARDS);

    let plans = vec![
        vec![(0u64, 5u64), (50, 99)],
        vec![(123, 1), (180, 75)],
        vec![(10, 130), (199, 0), (45, 70)],
    ];

    let loader_direct = build_loader(&path, None, true, 1e4, false);
    let direct: Vec<_> = plans
        .iter()
        .map(|p| loader_direct.process_plan(p.clone()).unwrap())
        .collect();

    // Fresh loader for the iter path so caches don't bleed assertions about
    // determinism across runs.
    let loader_iter = Arc::new(build_loader(&path, None, true, 1e4, false));
    let iter_results: Vec<_> = loader_iter
        .iter_with_plans(plans.clone().into_iter().map(Ok), 4)
        .map(|r| r.unwrap())
        .collect();

    assert_eq!(iter_results.len(), direct.len());
    for (i, (a, b)) in iter_results.iter().zip(direct.iter()).enumerate() {
        assert_eq!(a.pairs, b.pairs, "plan {i}: pair order mismatch");
        assert!(
            allclose(&a.x, &b.x, 1e-5, 1e-6),
            "plan {i}: x mismatch between iter and direct"
        );
        assert!(
            allclose(&a.x_paired, &b.x_paired, 1e-5, 1e-6),
            "plan {i}: x_paired mismatch between iter and direct"
        );
    }
}
