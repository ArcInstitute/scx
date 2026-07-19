//! L2 regression on the **sequential** `TrainingPipeline` / `decode_stage`
//! route (the default `TrainingDataset` path in pyscx).
//!
//! The `test_index_plan_parity.rs` suite covers the paired `IndexPlanLoader`
//! path and `normalize.rs` covers the primitives, but nothing exercised the
//! `decode_stage` scatter+normalize path end-to-end with a fixture where the
//! HVG panel captures only part of each cell's mass. This pins that the
//! projected `normalize`/`log1p` uses the cell's **full transcriptome depth**
//! (scanpy's normalize-then-subset), not the panel-local sum.

mod common;

use common::write_known_multinnz_fixture;
use scx_format_io::{BackedCsrReader, ScxReader};
use scx_loader::{
    fused_normalize_log1p_dense_with_depth, HvgProjection, LoaderConfig, TrainingPipeline,
};

fn allclose(a: &[f32], b: &[f32], rtol: f32, atol: f32) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= atol + rtol * y.abs())
}

fn open_backed(path: &std::path::Path) -> BackedCsrReader {
    BackedCsrReader::new(ScxReader::open(path).unwrap(), 4)
}

/// Correct reference: scatter into the HVG panel, then normalize+log1p by the
/// cell's FULL pre-projection depth.
fn expected_full_depth(
    backed: &BackedCsrReader,
    cell: u64,
    hvg: &HvgProjection,
    n_hvg: usize,
    target_sum: f64,
) -> Vec<f32> {
    let csr = backed.read_row_indices(&[cell]).unwrap();
    let (lo, hi) = (csr.indptr[0] as usize, csr.indptr[1] as usize);
    let mut out = vec![0f32; n_hvg];
    hvg.scatter_row(&csr.indices[lo..hi], &csr.data[lo..hi], &mut out);
    let depth: f64 = csr.data[lo..hi].iter().map(|&v| v as f64).sum();
    fused_normalize_log1p_dense_with_depth(&mut out, target_sum, depth);
    out
}

/// The (incorrect) panel-local normalize — used only to prove the pipeline's
/// output genuinely diverges from it, so the assertion is not vacuous.
fn panel_local(
    backed: &BackedCsrReader,
    cell: u64,
    hvg: &HvgProjection,
    n_hvg: usize,
    target_sum: f64,
) -> Vec<f32> {
    let csr = backed.read_row_indices(&[cell]).unwrap();
    let (lo, hi) = (csr.indptr[0] as usize, csr.indptr[1] as usize);
    let mut out = vec![0f32; n_hvg];
    hvg.scatter_row(&csr.indices[lo..hi], &csr.data[lo..hi], &mut out);
    let panel_sum: f64 = out.iter().map(|&v| v as f64).sum();
    fused_normalize_log1p_dense_with_depth(&mut out, target_sum, panel_sum);
    out
}

#[test]
fn training_pipeline_hvg_normalize_uses_full_depth() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_known_multinnz_fixture(&dir.path().join("f.scx"), 40, 4);

    // Panel captures genes 2 and 7 (present in every row) but not 33 or 58, so
    // panel-local depth is strictly below full depth for every cell.
    let hvg_indices = vec![2u32, 7, 40];
    let hvg = HvgProjection::new(hvg_indices.clone());
    let n_hvg = hvg.n_output_cols();
    let target_sum = 1.0e4_f64;

    let config = LoaderConfig {
        batch_size: 100, // one batch covers all 40 cells
        normalize: true,
        log1p: true,
        target_sum,
        hvg_indices: Some(hvg_indices),
        max_memory_mb: 1024,
        ..LoaderConfig::default()
    };

    let mut pipeline = TrainingPipeline::new(&path, config).unwrap();
    let backed = open_backed(&path);
    pipeline.start_epoch().unwrap();

    let mut n_checked = 0usize;
    let mut saw_divergence = false;
    while let Some(batch) = pipeline.next_batch().unwrap() {
        assert_eq!(batch.x_shape.1, n_hvg);
        for (r, &cell) in batch.cell_indices.iter().enumerate() {
            let actual = &batch.x[r * n_hvg..(r + 1) * n_hvg];
            let expected = expected_full_depth(&backed, cell, &hvg, n_hvg, target_sum);
            assert!(
                allclose(actual, &expected, 1e-5, 1e-6),
                "cell {cell}: decode_stage normalize must use full depth"
            );
            if !allclose(
                actual,
                &panel_local(&backed, cell, &hvg, n_hvg, target_sum),
                1e-5,
                1e-6,
            ) {
                saw_divergence = true;
            }
            n_checked += 1;
        }
    }

    assert_eq!(n_checked, 40, "all cells should be yielded exactly once");
    assert!(
        saw_divergence,
        "fixture/panel failed to exercise full-depth vs panel-local divergence"
    );
}
