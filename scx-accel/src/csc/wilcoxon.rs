//! CSC streaming Wilcoxon rank-sum DE.
//!
//! Mirrors [`crate::diffexp::wilcoxon_rank_sum_streaming`] (CSR path)
//! but processes gene chunks as CSC column ranges. For each chunk:
//!   1. `source.read_csc_columns(chunk_start..chunk_end)` materialises
//!      one CSC slab covering the chunk's columns.
//!   2. Scatter the slab's `(row, col, value)` triples into a row-major
//!      `[n_obs × chunk_size]` dense buffer.
//!   3. Call the existing `wilcoxon_rank_sum()` kernel on that buffer.
//!
//! The chunk-merge step (`merge_diff_exp_results`) runs unchanged on
//! per-chunk results, so the numerics are identical to the CSR path.

use crate::diffexp::{merge_diff_exp_results, wilcoxon_rank_sum, DiffExpResult};
use crate::error::{AccelError, Result};
use scx_format::ColumnShardSource;

/// Gene-chunked Wilcoxon rank-sum DE driven by a CSC source.
///
/// Returns the same [`DiffExpResult`] as the CSR equivalent. Chunk
/// boundaries are aligned to columns (genes), not rows.
#[allow(clippy::too_many_arguments)]
pub fn wilcoxon_rank_sum_streaming_csc<S: ColumnShardSource + ?Sized>(
    source: &S,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: Option<usize>,
    gene_chunk_size: usize,
    log_transformed: bool,
    rankby_abs: bool,
    tie_correct: bool,
) -> Result<DiffExpResult> {
    let n_obs = source.n_obs();
    let n_vars = gene_names.len();

    if groups.len() != n_obs {
        return Err(AccelError::InvalidInput(format!(
            "groups length {} != n_obs {}",
            groups.len(),
            n_obs
        )));
    }
    if gene_chunk_size == 0 {
        return Err(AccelError::InvalidInput(
            "gene_chunk_size must be > 0".to_string(),
        ));
    }
    if n_vars != source.n_vars() {
        return Err(AccelError::ShapeError(format!(
            "gene_names length {} != source.n_vars() {}",
            n_vars,
            source.n_vars()
        )));
    }

    let mut all_chunk_results = Vec::new();
    // Allocate the row-major dense scatter buffer once and reuse across
    // chunks. The trailing chunk may be smaller; we slice down and zero
    // only the active sub-range, avoiding the per-chunk full allocation.
    let max_chunk = gene_chunk_size.min(n_vars);
    let mut dense = vec![0.0f32; n_obs.saturating_mul(max_chunk)];

    for chunk_start in (0..n_vars).step_by(gene_chunk_size) {
        let chunk_end = (chunk_start + gene_chunk_size).min(n_vars);
        let chunk_size = chunk_end - chunk_start;

        // Read this chunk's columns as a single CSC slab.
        let csc = source
            .read_csc_columns(chunk_start as u32..chunk_end as u32)
            .map_err(AccelError::Scx)?;

        // Scatter into row-major dense buffer [n_obs × chunk_size].
        let dense_view = &mut dense[..n_obs * chunk_size];
        dense_view.fill(0.0);
        for local_col in 0..chunk_size {
            let s = csc.indptr[local_col] as usize;
            let e = csc.indptr[local_col + 1] as usize;
            for j in s..e {
                let row = csc.indices[j] as usize;
                if row < n_obs {
                    dense_view[row * chunk_size + local_col] = csc.data[j];
                }
            }
        }

        let chunk_result = wilcoxon_rank_sum(
            dense_view,
            n_obs,
            chunk_size,
            &gene_names[chunk_start..chunk_end],
            groups,
            group_names,
            reference,
            log_transformed,
            rankby_abs,
            tie_correct,
            chunk_start,
        )?;
        all_chunk_results.push(chunk_result);
    }

    merge_diff_exp_results(all_chunk_results, rankby_abs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csc::test_helpers::{deterministic_dense, write_csr_csc_test_file};
    use crate::diffexp::wilcoxon_rank_sum_streaming;
    use scx_format::{BackedCscReader, BackedCsrReader, ScxReader};
    use tempfile::tempdir;

    #[test]
    fn wilcoxon_csc_matches_csr() {
        let dir = tempdir().unwrap();
        let n_obs = 16usize;
        let n_vars = 8usize;
        let dense = deterministic_dense(n_obs, n_vars);
        let path = write_csr_csc_test_file(dir.path(), "wilcox", n_obs, n_vars, &dense, 4);

        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        let group_names = vec!["A".to_string(), "B".to_string()];
        // Half of cells in group 0, half in group 1.
        let groups: Vec<usize> = (0..n_obs)
            .map(|i| if i < n_obs / 2 { 0 } else { 1 })
            .collect();

        let csr_reader = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);
        let csr_res = wilcoxon_rank_sum_streaming(
            &csr_reader,
            &gene_names,
            &groups,
            &group_names,
            None,
            4, // gene_chunk_size
            false,
            false,
            false,
        )
        .unwrap();

        let csc_reader = BackedCscReader::new(ScxReader::open(&path).unwrap(), 0).unwrap();
        let csc_res = wilcoxon_rank_sum_streaming_csc(
            &csc_reader,
            &gene_names,
            &groups,
            &group_names,
            None,
            4,
            false,
            false,
            false,
        )
        .unwrap();

        // Compare per-group sorted score vectors. Names ordering must
        // match because the underlying sort is by score and inputs are
        // identical.
        assert_eq!(csr_res.group_names, csc_res.group_names);
        for g in 0..csr_res.group_names.len() {
            assert_eq!(csr_res.names[g], csc_res.names[g], "group {g} gene order");
            for k in 0..csr_res.scores[g].len() {
                assert!(
                    (csr_res.scores[g][k] - csc_res.scores[g][k]).abs() < 1e-9,
                    "group {g} score[{k}] mismatch"
                );
                assert!(
                    (csr_res.pvals[g][k] - csc_res.pvals[g][k]).abs() < 1e-9,
                    "group {g} pval[{k}] mismatch"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // GPU CSC-direct / CSR-direct v3 Wilcoxon parity (§C.9). All gated on a
    // CUDA device; the binary still links without one. Compared by gene name
    // (not slot index) because `DiffExpResult` is score-sorted and GPU fp32
    // vs CPU f64 can re-order ties.
    // -----------------------------------------------------------------------
    #[cfg(feature = "gpu")]
    use crate::diffexp::DiffExpResult;
    #[cfg(feature = "gpu")]
    use crate::diffexp_gpu::{wilcoxon_rank_sum_gpu, GpuDeShardInput};
    #[cfg(feature = "gpu")]
    use crate::route::AccelRoute;
    #[cfg(feature = "gpu")]
    use std::collections::HashMap;

    /// `(group, gene) -> (score, pval, logfc)` lookup, order-independent.
    #[cfg(feature = "gpu")]
    fn de_by_gene(res: &DiffExpResult) -> HashMap<(String, String), (f64, f64, f64)> {
        let mut m = HashMap::new();
        for g in 0..res.group_names.len() {
            for k in 0..res.names[g].len() {
                m.insert(
                    (res.group_names[g].clone(), res.names[g][k].clone()),
                    (res.scores[g][k], res.pvals[g][k], res.logfoldchanges[g][k]),
                );
            }
        }
        m
    }

    /// Assert GPU vs CPU Wilcoxon results agree within fp32-GPU tolerance,
    /// matching by `(group, gene)` and handling NaN (empty groups / undefined
    /// logFC) symmetrically.
    #[cfg(feature = "gpu")]
    fn assert_de_close(cpu: &DiffExpResult, gpu: &DiffExpResult, ctx: &str) {
        assert_eq!(cpu.group_names, gpu.group_names, "{ctx}: group_names");
        let c = de_by_gene(cpu);
        let g = de_by_gene(gpu);
        assert_eq!(c.len(), g.len(), "{ctx}: result entry count");
        for (key, &(sc, pc, lc)) in &c {
            let &(sg, pg, lg) = g
                .get(key)
                .unwrap_or_else(|| panic!("{ctx}: GPU missing {key:?}"));
            // Z-score (signed). NaN-aware.
            if sc.is_finite() && sg.is_finite() {
                let d = (sc - sg).abs();
                assert!(
                    d < 1e-3 || d / sc.abs().max(1e-12) < 1e-3,
                    "{ctx}: score {key:?} cpu={sc} gpu={sg}"
                );
            } else {
                assert_eq!(sc.is_nan(), sg.is_nan(), "{ctx}: score NaN-ness {key:?}");
            }
            // Two-sided p-value.
            let pdiff = (pc - pg).abs();
            assert!(
                pdiff < 1e-6 || pdiff / pc.abs().max(1e-30) < 1e-3,
                "{ctx}: pval {key:?} cpu={pc} gpu={pg}"
            );
            // log2 fold-change. NaN-aware.
            if lc.is_finite() && lg.is_finite() {
                let d = (lc - lg).abs();
                assert!(
                    d < 1e-3 || d / lc.abs().max(1e-12) < 1e-3,
                    "{ctx}: logfc {key:?} cpu={lc} gpu={lg}"
                );
            } else {
                assert_eq!(lc.is_nan(), lg.is_nan(), "{ctx}: logfc NaN-ness {key:?}");
            }
        }
    }

    /// Skip helper: returns true (and prints) when no CUDA device is present.
    #[cfg(feature = "gpu")]
    fn no_gpu() -> bool {
        if scx_gpu::device::GpuDevice::new(0).is_err() {
            eprintln!("CUDA not available — skipping GPU Wilcoxon v3 parity test");
            true
        } else {
            false
        }
    }

    /// Run the CPU CSR streaming baseline for the given fixture/params.
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)]
    fn cpu_baseline(
        path: &std::path::Path,
        gene_names: &[String],
        groups: &[usize],
        group_names: &[String],
        reference: Option<usize>,
        chunk: usize,
    ) -> DiffExpResult {
        let csr_reader = BackedCsrReader::new(ScxReader::open(path).unwrap(), 0);
        wilcoxon_rank_sum_streaming(
            &csr_reader,
            gene_names,
            groups,
            group_names,
            reference,
            chunk,
            false,
            false,
            true,
        )
        .expect("CPU streaming Wilcoxon failed")
    }

    /// Run the GPU backed Wilcoxon path with v3 forced on. `with_csc` selects
    /// the CSC-direct route (sidecar provided) vs the CSR-direct fallback.
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)]
    fn gpu_v3(
        path: &std::path::Path,
        gene_names: &[String],
        groups: &[usize],
        group_names: &[String],
        reference: Option<usize>,
        chunk: usize,
        with_csc: bool,
    ) -> DiffExpResult {
        let csr_reader = BackedCsrReader::new(ScxReader::open(path).unwrap(), 0);
        let csc_reader = if with_csc {
            Some(BackedCscReader::new(ScxReader::open(path).unwrap(), 0).unwrap())
        } else {
            None
        };
        // v3 is the unconditional default since the V1b flip — no override.
        wilcoxon_rank_sum_gpu(
            0,
            GpuDeShardInput::Backed {
                csr: &csr_reader,
                csc: csc_reader.as_ref(),
            },
            gene_names,
            groups,
            group_names,
            reference,
            Some(chunk),
            false,
            false,
            true,
        )
        .expect("GPU v3 Wilcoxon failed")
    }

    #[cfg(feature = "gpu")]
    fn three_group_fixture(
        dir: &std::path::Path,
        name: &str,
        cols_per_csc_shard: usize,
    ) -> (std::path::PathBuf, Vec<String>, Vec<usize>, Vec<String>) {
        let n_obs = 64usize;
        let n_vars = 20usize;
        let dense = deterministic_dense(n_obs, n_vars);
        let path = write_csr_csc_test_file(dir, name, n_obs, n_vars, &dense, cols_per_csc_shard);
        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        let groups: Vec<usize> = (0..n_obs).map(|i| (i * 3) / n_obs).collect();
        let group_names = vec!["ref".to_string(), "ko_a".to_string(), "ko_b".to_string()];
        (path, gene_names, groups, group_names)
    }

    /// (1) CSC-direct vs CPU streaming — 1-vs-rest.
    #[cfg(feature = "gpu")]
    #[test]
    fn wilcoxon_gpu_v3_csc_one_vs_rest() {
        if no_gpu() {
            return;
        }
        let dir = tempdir().unwrap();
        let (path, gn, gr, names) = three_group_fixture(dir.path(), "wil_csc_ovr", 7);
        let cpu = cpu_baseline(&path, &gn, &gr, &names, None, 7);
        let gpu = gpu_v3(&path, &gn, &gr, &names, None, 7, true);
        assert_de_close(&cpu, &gpu, "csc 1-vs-rest");
        assert_eq!(gpu.exec_info.route, AccelRoute::GpuCscV3);
    }

    /// (2) CSC-direct vs CPU streaming — ref-mode.
    #[cfg(feature = "gpu")]
    #[test]
    fn wilcoxon_gpu_v3_csc_ref_mode() {
        if no_gpu() {
            return;
        }
        let dir = tempdir().unwrap();
        let (path, gn, gr, names) = three_group_fixture(dir.path(), "wil_csc_ref", 7);
        let cpu = cpu_baseline(&path, &gn, &gr, &names, Some(0), 7);
        let gpu = gpu_v3(&path, &gn, &gr, &names, Some(0), 7, true);
        assert_de_close(&cpu, &gpu, "csc ref-mode");
        assert_eq!(gpu.exec_info.route, AccelRoute::GpuCscV3);
    }

    /// (3) CSR-direct fallback (no CSC sidecar) vs CPU streaming — 1-vs-rest.
    #[cfg(feature = "gpu")]
    #[test]
    fn wilcoxon_gpu_v3_csr_fallback() {
        if no_gpu() {
            return;
        }
        let dir = tempdir().unwrap();
        let (path, gn, gr, names) = three_group_fixture(dir.path(), "wil_csr_fb", 7);
        let cpu = cpu_baseline(&path, &gn, &gr, &names, None, 7);
        let gpu = gpu_v3(&path, &gn, &gr, &names, None, 7, false);
        assert_de_close(&cpu, &gpu, "csr fallback 1-vs-rest");
        assert_eq!(gpu.exec_info.route, AccelRoute::GpuCsrV3);
    }

    /// (4) Route metadata: CSC sidecar → GpuCscV3; none → GpuCsrV3 (v3 on).
    #[cfg(feature = "gpu")]
    #[test]
    fn wilcoxon_gpu_v3_route_metadata() {
        if no_gpu() {
            return;
        }
        let dir = tempdir().unwrap();
        let (path, gn, gr, names) = three_group_fixture(dir.path(), "wil_route", 7);
        let csc = gpu_v3(&path, &gn, &gr, &names, Some(0), 7, true);
        assert_eq!(
            csc.exec_info.route,
            AccelRoute::GpuCscV3,
            "csc → gpu_csc_v3"
        );
        let csr = gpu_v3(&path, &gn, &gr, &names, Some(0), 7, false);
        assert_eq!(
            csr.exec_info.route,
            AccelRoute::GpuCsrV3,
            "no csc → gpu_csr_v3"
        );
    }

    /// (5) Empty test group — must produce NaN score / p=1 matching CPU.
    #[cfg(feature = "gpu")]
    #[test]
    fn wilcoxon_gpu_v3_csc_empty_group() {
        if no_gpu() {
            return;
        }
        let dir = tempdir().unwrap();
        let (path, gn, gr, _names) = three_group_fixture(dir.path(), "wil_empty", 7);
        // 4th group label with zero members.
        let names = vec![
            "ref".to_string(),
            "ko_a".to_string(),
            "ko_b".to_string(),
            "empty".to_string(),
        ];
        let cpu = cpu_baseline(&path, &gn, &gr, &names, Some(0), 7);
        let gpu = gpu_v3(&path, &gn, &gr, &names, Some(0), 7, true);
        assert_de_close(&cpu, &gpu, "csc empty group");
    }

    /// (6) All-zero gene column — defined U / p=1 for the constant column.
    #[cfg(feature = "gpu")]
    #[test]
    fn wilcoxon_gpu_v3_csc_all_zero_gene() {
        if no_gpu() {
            return;
        }
        let n_obs = 48usize;
        let n_vars = 12usize;
        let mut dense = deterministic_dense(n_obs, n_vars);
        // Force gene 5 to all zeros.
        for r in 0..n_obs {
            dense[r * n_vars + 5] = 0;
        }
        let dir = tempdir().unwrap();
        let path = write_csr_csc_test_file(dir.path(), "wil_zero", n_obs, n_vars, &dense, 5);
        let gn: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        let gr: Vec<usize> = (0..n_obs).map(|i| (i * 2) / n_obs).collect();
        let names = vec!["a".to_string(), "b".to_string()];
        let cpu = cpu_baseline(&path, &gn, &gr, &names, None, 5);
        let gpu = gpu_v3(&path, &gn, &gr, &names, None, 5, true);
        assert_de_close(&cpu, &gpu, "csc all-zero gene");
    }

    /// (7) Tie-spanning gene — constant value across all cells (max ties).
    #[cfg(feature = "gpu")]
    #[test]
    fn wilcoxon_gpu_v3_csc_tie_spanning_gene() {
        if no_gpu() {
            return;
        }
        let n_obs = 48usize;
        let n_vars = 12usize;
        let mut dense = deterministic_dense(n_obs, n_vars);
        // Gene 3 constant (every cell = 5) → fully tied column.
        for r in 0..n_obs {
            dense[r * n_vars + 3] = 5;
        }
        let dir = tempdir().unwrap();
        let path = write_csr_csc_test_file(dir.path(), "wil_tie", n_obs, n_vars, &dense, 5);
        let gn: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        let gr: Vec<usize> = (0..n_obs).map(|i| (i * 2) / n_obs).collect();
        let names = vec!["a".to_string(), "b".to_string()];
        let cpu = cpu_baseline(&path, &gn, &gr, &names, Some(0), 5);
        let gpu = gpu_v3(&path, &gn, &gr, &names, Some(0), 5, true);
        assert_de_close(&cpu, &gpu, "csc tie-spanning gene");
    }

    /// (8) Multi-shard CSC: `cols_per_csc_shard=4`, `n_vars=20` → 5 shards;
    /// `gene_chunk_size=7` spans shard boundaries (exercises range prefilter).
    #[cfg(feature = "gpu")]
    #[test]
    fn wilcoxon_gpu_v3_csc_multi_shard() {
        if no_gpu() {
            return;
        }
        let dir = tempdir().unwrap();
        let (path, gn, gr, names) = three_group_fixture(dir.path(), "wil_multishard", 4);
        let cpu = cpu_baseline(&path, &gn, &gr, &names, None, 7);
        let gpu = gpu_v3(&path, &gn, &gr, &names, None, 7, true);
        assert_de_close(&cpu, &gpu, "csc multi-shard");

        // §B.10 criterion 4: range prefiltering decoded strictly fewer CSC
        // shards than the no-prefilter worst case (n_csc_shards × n_chunks).
        let n_vars = gn.len();
        let n_chunks = n_vars.div_ceil(7); // gene_chunk_size = 7
        let n_csc_shards = n_vars.div_ceil(4); // cols_per_csc_shard = 4
        let decoded = gpu
            .exec_info
            .shards_decoded
            .expect("shards_decoded recorded on v3 CSC route");
        assert!(
            decoded > 0 && decoded < n_csc_shards * n_chunks,
            "expected prefiltered shard count in (0, {}), got {decoded}",
            n_csc_shards * n_chunks
        );
    }

    // (Former test (9) "v3 CSC-direct vs v1 dense-chunk" was removed with the
    // V1b default-flip: v1 is no longer reachable through `plan_de_route`, so it
    // cannot be selected in-process for an A/B comparison. v3-vs-CPU parity
    // across modes/edge-cases is covered by the tests above.)
}
