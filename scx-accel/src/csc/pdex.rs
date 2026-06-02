//! CSC streaming `pdex_ref` differential expression.
//!
//! Mirrors [`crate::diffexp::pdex_ref_streaming`] (CSR path) but processes
//! gene chunks as CSC column ranges. For each chunk:
//!   1. `source.read_csc_columns(chunk_start..chunk_end)` materialises one
//!      CSC slab covering the chunk's columns.
//!   2. Scatter the slab's `(row, col, value)` triples into a row-major
//!      `[n_obs × chunk_size]` dense buffer.
//!   3. Call the existing `pdex_ref()` kernel on that buffer.
//!
//! The chunk-merge step (`merge_pdex_chunk_into` + `recompute_pdex_fdrs`)
//! runs unchanged on per-chunk results, so the numerics are identical to
//! the CSR path.
//!
//! Same shape as [`crate::csc::wilcoxon::wilcoxon_rank_sum_streaming_csc`];
//! the only difference is the per-chunk kernel call and the merge logic.

use crate::diffexp::{
    empty_pdex_result, merge_pdex_chunk_into, pdex_ref, recompute_pdex_fdrs, PdexRefResult,
};
use crate::error::{AccelError, Result};
use crate::pseudobulk::GeomMeanMode;
use scx_format::ColumnShardSource;

/// Gene-chunked `pdex_ref` driven by a CSC source.
///
/// Returns the same [`PdexRefResult`] as the CSR equivalent
/// (`pdex_ref_streaming`). Chunk boundaries are aligned to columns
/// (genes), not rows.
#[allow(clippy::too_many_arguments)]
pub fn pdex_ref_streaming_csc<S: ColumnShardSource + ?Sized>(
    source: &S,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: usize,
    gene_chunk_size: usize,
    mode: GeomMeanMode,
    epsilon: f64,
) -> Result<PdexRefResult> {
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

    let mut combined: Option<PdexRefResult> = None;

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

        let chunk_result = pdex_ref(
            dense_view,
            n_obs,
            chunk_size,
            &gene_names[chunk_start..chunk_end],
            groups,
            group_names,
            reference,
            mode,
            epsilon,
        )?;

        combined = Some(match combined.take() {
            None => chunk_result,
            Some(mut acc) => {
                merge_pdex_chunk_into(&mut acc, chunk_result);
                acc
            }
        });
    }

    let mut result = combined.unwrap_or_else(|| empty_pdex_result(group_names, reference));
    recompute_pdex_fdrs(&mut result);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csc::test_helpers::{deterministic_dense, write_csr_csc_test_file};
    use crate::diffexp::pdex_ref_streaming;
    use scx_format::{BackedCscReader, BackedCsrReader, ScxReader};
    use tempfile::tempdir;

    /// CSR↔CSC parity: `pdex_ref_streaming` (CSR) and `pdex_ref_streaming_csc`
    /// (CSC) must produce identical `PdexRefResult` on the same fixture.
    #[test]
    fn pdex_ref_csc_matches_csr() {
        let n_obs = 32usize;
        let n_vars = 16usize;
        let cols_per_csc_shard = 8usize;
        let dense = deterministic_dense(n_obs, n_vars);
        let dir = tempdir().unwrap();
        let path = write_csr_csc_test_file(
            dir.path(),
            "pdex_csc_parity",
            n_obs,
            n_vars,
            &dense,
            cols_per_csc_shard,
        );

        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        // 3 groups: ref + 2 KOs, ~equal sizes.
        let groups: Vec<usize> = (0..n_obs).map(|i| (i * 3) / n_obs).collect();
        let group_names = vec!["ref".to_string(), "ko_a".to_string(), "ko_b".to_string()];
        let reference = 0usize;
        let mode = GeomMeanMode::ArithRaw;
        let epsilon = 1e-6;

        let csr_reader = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);
        let csc_reader = BackedCscReader::new(ScxReader::open(&path).unwrap(), 0).unwrap();

        let csr_res = pdex_ref_streaming(
            &csr_reader,
            &gene_names,
            &groups,
            &group_names,
            reference,
            5, // gene_chunk_size — forces multi-chunk on n_vars=16
            mode,
            epsilon,
        )
        .expect("CSR streaming pdex_ref failed");

        let csc_res = pdex_ref_streaming_csc(
            &csc_reader,
            &gene_names,
            &groups,
            &group_names,
            reference,
            5,
            mode,
            epsilon,
        )
        .expect("CSC streaming pdex_ref failed");

        // Structural equality.
        assert_eq!(csr_res.group_names, csc_res.group_names);
        assert_eq!(csr_res.feature_names, csc_res.feature_names);
        assert_eq!(csr_res.ref_membership, csc_res.ref_membership);
        assert_eq!(csr_res.target_memberships, csc_res.target_memberships);

        // Numeric equality: U statistic exact; p / means within fp64 ulps.
        let n_test = csr_res.group_names.len();
        for tg in 0..n_test {
            for var in 0..n_vars {
                let u_csr = csr_res.statistics[tg][var];
                let u_csc = csc_res.statistics[tg][var];
                if u_csr.is_finite() && u_csc.is_finite() {
                    assert!(
                        (u_csr - u_csc).abs() < 1e-9,
                        "U mismatch tg={tg} gene={var}: csr={u_csr}, csc={u_csc}"
                    );
                }
                let p_csr = csr_res.p_values[tg][var];
                let p_csc = csc_res.p_values[tg][var];
                assert!(
                    (p_csr - p_csc).abs() < 1e-12,
                    "p mismatch tg={tg} gene={var}: csr={p_csr}, csc={p_csc}"
                );
                let tm_csr = csr_res.target_means[tg][var];
                let tm_csc = csc_res.target_means[tg][var];
                if tm_csr.is_finite() && tm_csc.is_finite() {
                    assert!(
                        (tm_csr - tm_csc).abs() < 1e-12,
                        "target_mean mismatch tg={tg} gene={var}: csr={tm_csr}, csc={tm_csc}"
                    );
                }
                let r_csr = csr_res.ref_means[var];
                let r_csc = csc_res.ref_means[var];
                if r_csr.is_finite() && r_csc.is_finite() {
                    assert!(
                        (r_csr - r_csc).abs() < 1e-12,
                        "ref_mean mismatch var={var}: csr={r_csr}, csc={r_csc}"
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // G4.3: GPU v3 CSC-direct parity vs CPU CSR baseline.
    // -----------------------------------------------------------------
    //
    // Two tests exercise the v3 dispatch in `pdex_ref_gpu` (Backed variant):
    //   (a) v3 CSC-direct path (CSC sidecar provided)
    //   (b) v3 CSR-direct fallback (no CSC sidecar)
    //
    // v3 is the unconditional default since the V1b flip, so no override is
    // needed. Tolerance is fp32-tight on U statistic / means since the v3
    // kernels use f64 atomicAdd (CSR fallback) or f64 shared-mem tree-reduce
    // (CSC) for the pseudobulk fold — same precision as `gpu_de_pseudobulk_all_groups`.
    //
    // Skips via `require_gpu_or_skip!()` if no CUDA device is available
    // (the test binary still links; the test just returns).
    #[cfg(feature = "gpu")]
    #[test]
    fn test_pdex_ref_gpu_v3_csc_matches_cpu_streaming() {
        use scx_gpu::device::GpuDevice;
        if GpuDevice::new(0).is_err() {
            eprintln!("CUDA not available — skipping GPU CSC parity test");
            return;
        }
        let n_obs = 64usize;
        let n_vars = 20usize;
        let cols_per_csc_shard = 7usize;
        let dense = deterministic_dense(n_obs, n_vars);
        let dir = tempdir().unwrap();
        let path = write_csr_csc_test_file(
            dir.path(),
            "pdex_gpu_v3_csc_parity",
            n_obs,
            n_vars,
            &dense,
            cols_per_csc_shard,
        );

        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        let groups: Vec<usize> = (0..n_obs).map(|i| (i * 3) / n_obs).collect();
        let group_names = vec!["ref".to_string(), "ko_a".to_string(), "ko_b".to_string()];
        let reference = 0usize;
        let mode = GeomMeanMode::ArithRaw;
        let epsilon = 1e-6;

        let csr_reader = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);
        let csc_reader = BackedCscReader::new(ScxReader::open(&path).unwrap(), 0).unwrap();

        // CPU CSR baseline.
        let cpu_res = pdex_ref_streaming(
            &csr_reader,
            &gene_names,
            &groups,
            &group_names,
            reference,
            7, // gene_chunk_size — multi-chunk on n_vars=20
            mode,
            epsilon,
        )
        .expect("CPU streaming pdex_ref failed");

        // GPU v3 CSC-direct path (v3 is the unconditional default).
        let gpu_res = crate::diffexp_gpu::pdex_ref_gpu(
            0,
            crate::diffexp_gpu::GpuDeShardInput::Backed {
                csr: &csr_reader,
                csc: Some(&csc_reader),
            },
            &gene_names,
            &groups,
            &group_names,
            reference,
            Some(7),
            mode,
            epsilon,
        )
        .expect("GPU v3 CSC streaming pdex_ref failed");

        // Structural equality.
        assert_eq!(cpu_res.group_names, gpu_res.group_names);
        assert_eq!(cpu_res.feature_names, gpu_res.feature_names);
        assert_eq!(cpu_res.ref_membership, gpu_res.ref_membership);
        assert_eq!(cpu_res.target_memberships, gpu_res.target_memberships);

        // §B.10 criterion 4: CSC range prefiltering decoded strictly fewer
        // shards than the no-prefilter worst case (n_csc_shards × n_chunks).
        let n_chunks = n_vars.div_ceil(7); // gene_chunk_size = 7
        let n_csc_shards = n_vars.div_ceil(cols_per_csc_shard);
        let decoded = gpu_res
            .exec_info
            .shards_decoded
            .expect("shards_decoded recorded on v3 CSC route");
        assert!(
            decoded > 0 && decoded < n_csc_shards * n_chunks,
            "expected prefiltered shard count in (0, {}), got {decoded}",
            n_csc_shards * n_chunks
        );

        let n_test = cpu_res.group_names.len();
        for tg in 0..n_test {
            for var in 0..n_vars {
                let u_cpu = cpu_res.statistics[tg][var];
                let u_gpu = gpu_res.statistics[tg][var];
                if u_cpu.is_finite() && u_gpu.is_finite() {
                    assert!(
                        (u_cpu - u_gpu).abs() < 1e-3,
                        "U mismatch (v3 CSC) tg={tg} gene={var}: cpu={u_cpu}, gpu={u_gpu}"
                    );
                }
                let p_cpu = cpu_res.p_values[tg][var];
                let p_gpu = gpu_res.p_values[tg][var];
                let pdiff = (p_cpu - p_gpu).abs();
                let prel = pdiff / p_cpu.abs().max(1e-30);
                assert!(
                    pdiff < 1e-6 || prel < 1e-3,
                    "p mismatch (v3 CSC) tg={tg} gene={var}: cpu={p_cpu}, gpu={p_gpu}"
                );
                let tm_cpu = cpu_res.target_means[tg][var];
                let tm_gpu = gpu_res.target_means[tg][var];
                if tm_cpu.is_finite() && tm_gpu.is_finite() {
                    let diff = (tm_cpu - tm_gpu).abs();
                    let rel = diff / tm_cpu.abs().max(1e-12);
                    assert!(
                        diff < 1e-4 || rel < 1e-4,
                        "target_mean mismatch (v3 CSC) tg={tg} gene={var}: cpu={tm_cpu}, gpu={tm_gpu}"
                    );
                }
                let r_cpu = cpu_res.ref_means[var];
                let r_gpu = gpu_res.ref_means[var];
                if r_cpu.is_finite() && r_gpu.is_finite() {
                    let diff = (r_cpu - r_gpu).abs();
                    let rel = diff / r_cpu.abs().max(1e-12);
                    assert!(
                        diff < 1e-4 || rel < 1e-4,
                        "ref_mean mismatch (v3 CSC) var={var}: cpu={r_cpu}, gpu={r_gpu}"
                    );
                }
            }
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_pdex_ref_gpu_v3_csr_fallback_matches_cpu_streaming() {
        use scx_gpu::device::GpuDevice;
        if GpuDevice::new(0).is_err() {
            eprintln!("CUDA not available — skipping GPU CSR fallback parity test");
            return;
        }
        let n_obs = 64usize;
        let n_vars = 20usize;
        let cols_per_csc_shard = 7usize;
        let dense = deterministic_dense(n_obs, n_vars);
        let dir = tempdir().unwrap();
        let path = write_csr_csc_test_file(
            dir.path(),
            "pdex_gpu_v3_csr_parity",
            n_obs,
            n_vars,
            &dense,
            cols_per_csc_shard,
        );

        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        let groups: Vec<usize> = (0..n_obs).map(|i| (i * 3) / n_obs).collect();
        let group_names = vec!["ref".to_string(), "ko_a".to_string(), "ko_b".to_string()];
        let reference = 0usize;
        let mode = GeomMeanMode::ArithRaw;
        let epsilon = 1e-6;

        let csr_reader = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);

        let cpu_res = pdex_ref_streaming(
            &csr_reader,
            &gene_names,
            &groups,
            &group_names,
            reference,
            7,
            mode,
            epsilon,
        )
        .expect("CPU streaming pdex_ref failed");

        // GPU v3 CSR-direct fallback (csc_reader = None; v3 is the default).
        let gpu_res = crate::diffexp_gpu::pdex_ref_gpu(
            0,
            crate::diffexp_gpu::GpuDeShardInput::Backed {
                csr: &csr_reader,
                csc: None,
            },
            &gene_names,
            &groups,
            &group_names,
            reference,
            Some(7),
            mode,
            epsilon,
        )
        .expect("GPU v3 CSR-fallback streaming pdex_ref failed");

        assert_eq!(cpu_res.group_names, gpu_res.group_names);
        assert_eq!(cpu_res.feature_names, gpu_res.feature_names);

        let n_test = cpu_res.group_names.len();
        for tg in 0..n_test {
            for var in 0..n_vars {
                let u_cpu = cpu_res.statistics[tg][var];
                let u_gpu = gpu_res.statistics[tg][var];
                if u_cpu.is_finite() && u_gpu.is_finite() {
                    assert!(
                        (u_cpu - u_gpu).abs() < 1e-3,
                        "U mismatch (v3 CSR fallback) tg={tg} gene={var}: cpu={u_cpu}, gpu={u_gpu}"
                    );
                }
                let p_cpu = cpu_res.p_values[tg][var];
                let p_gpu = gpu_res.p_values[tg][var];
                let pdiff = (p_cpu - p_gpu).abs();
                let prel = pdiff / p_cpu.abs().max(1e-30);
                assert!(
                    pdiff < 1e-6 || prel < 1e-3,
                    "p mismatch (v3 CSR fallback) tg={tg} gene={var}: cpu={p_cpu}, gpu={p_gpu}"
                );
                let tm_cpu = cpu_res.target_means[tg][var];
                let tm_gpu = gpu_res.target_means[tg][var];
                if tm_cpu.is_finite() && tm_gpu.is_finite() {
                    let diff = (tm_cpu - tm_gpu).abs();
                    let rel = diff / tm_cpu.abs().max(1e-12);
                    assert!(
                        diff < 1e-4 || rel < 1e-4,
                        "target_mean mismatch (v3 CSR fallback) tg={tg} gene={var}: cpu={tm_cpu}, gpu={tm_gpu}"
                    );
                }
            }
        }
    }

    /// Regression guard for the CUDA-graph-capture htod bug: the GPU pdex path
    /// over a **backed multi-shard** reader (no CSC sidecar → CSR-direct
    /// `gpu_csr_v3`), **multi-chunk**, with graph capture **forced on**. Before
    /// the device-side-scatter fix, the captured per-chunk sequence did a
    /// host→device copy of the cell permutations, invalidating the capture
    /// (`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`) and erroring out. `csc: None`.
    /// Must complete and match the CPU streaming reference. (The in-memory
    /// single-shard multi-chunk tests did not catch this — the failure needs
    /// the backed streaming path.)
    #[cfg(feature = "gpu")]
    #[test]
    fn test_pdex_ref_gpu_v1_backed_multichunk_graph_capture() {
        use scx_gpu::device::GpuDevice;
        if GpuDevice::new(0).is_err() {
            eprintln!("CUDA not available — skipping GPU graph-capture regression test");
            return;
        }
        let n_obs = 64usize;
        let n_vars = 20usize;
        let cols_per_csc_shard = 7usize;
        let dense = deterministic_dense(n_obs, n_vars);
        let dir = tempdir().unwrap();
        let path = write_csr_csc_test_file(
            dir.path(),
            "pdex_gpu_v1_graph_capture",
            n_obs,
            n_vars,
            &dense,
            cols_per_csc_shard,
        );

        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        let groups: Vec<usize> = (0..n_obs).map(|i| (i * 3) / n_obs).collect();
        let group_names = vec!["ref".to_string(), "ko_a".to_string(), "ko_b".to_string()];
        let reference = 0usize;
        let mode = GeomMeanMode::ArithRaw;
        let epsilon = 1e-6;

        let csr_reader = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);
        let cpu_res = pdex_ref_streaming(
            &csr_reader,
            &gene_names,
            &groups,
            &group_names,
            reference,
            7, // gene_chunk_size — multi-chunk on n_vars=20 (3 chunks)
            mode,
            epsilon,
        )
        .expect("CPU streaming pdex_ref failed");

        // Force graph capture ON (deterministic) so the captured CSR-direct v3
        // sequence runs over the backed multi-shard reader.
        let prev_graphs = scx_gpu::set_cuda_graphs_enabled_override(Some(true));
        let csr_reader_gpu = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);
        let gpu_res = crate::diffexp_gpu::pdex_ref_gpu(
            0,
            crate::diffexp_gpu::GpuDeShardInput::Backed {
                csr: &csr_reader_gpu,
                csc: None,
            },
            &gene_names,
            &groups,
            &group_names,
            reference,
            Some(7),
            mode,
            epsilon,
        );
        scx_gpu::set_cuda_graphs_enabled_override(prev_graphs);
        let gpu_res =
            gpu_res.expect("GPU v1 backed multi-chunk pdex_ref under graph capture failed");

        assert_eq!(cpu_res.group_names, gpu_res.group_names);
        assert_eq!(cpu_res.feature_names, gpu_res.feature_names);
        let n_test = cpu_res.group_names.len();
        for tg in 0..n_test {
            for var in 0..n_vars {
                let u_cpu = cpu_res.statistics[tg][var];
                let u_gpu = gpu_res.statistics[tg][var];
                if u_cpu.is_finite() && u_gpu.is_finite() {
                    assert!(
                        (u_cpu - u_gpu).abs() < 1e-3,
                        "U mismatch (v1 graph) tg={tg} gene={var}: cpu={u_cpu}, gpu={u_gpu}"
                    );
                }
                let p_cpu = cpu_res.p_values[tg][var];
                let p_gpu = gpu_res.p_values[tg][var];
                let pdiff = (p_cpu - p_gpu).abs();
                assert!(
                    pdiff < 1e-6 || pdiff / p_cpu.abs().max(1e-30) < 1e-3,
                    "p mismatch (v1 graph) tg={tg} gene={var}: cpu={p_cpu}, gpu={p_gpu}"
                );
            }
        }
    }

    /// Wilcoxon counterpart of `test_pdex_ref_gpu_v1_backed_multichunk_graph_capture`.
    /// The Wilcoxon GPU chunk sequence had the same CUDA-graph-capture htod bug
    /// (its scatter calls copied the pool + per-test-group permutations
    /// host→device inside the captured region), fixed by the pre-upload-once +
    /// device-side scatter change. A backed input with no CSC sidecar runs the
    /// CSR-direct `gpu_csr_v3` driver. Exercised in
    /// **ref-mode** (`reference = Some(0)`) to cover the per-test-group
    /// combined-tie + `tie_per_group` staging branch of the captured sequence.
    /// Backed multi-shard + multi-chunk (`gene_chunk_size=7`, `n_vars=20`) +
    /// graphs forced ON; must match the CPU streaming reference. (The dense
    /// `test_wilcoxon_gpu_*_graph_vs_direct_parity` tests are single-chunk and
    /// never trigger capture.)
    #[cfg(feature = "gpu")]
    #[test]
    fn test_wilcoxon_gpu_v1_backed_multichunk_graph_capture() {
        use crate::diffexp::wilcoxon_rank_sum_streaming;
        use scx_gpu::device::GpuDevice;
        if GpuDevice::new(0).is_err() {
            eprintln!("CUDA not available — skipping GPU graph-capture regression test");
            return;
        }
        let n_obs = 64usize;
        let n_vars = 20usize;
        let cols_per_csc_shard = 7usize;
        let dense = deterministic_dense(n_obs, n_vars);
        let dir = tempdir().unwrap();
        let path = write_csr_csc_test_file(
            dir.path(),
            "wilcoxon_gpu_v1_graph_capture",
            n_obs,
            n_vars,
            &dense,
            cols_per_csc_shard,
        );

        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        let groups: Vec<usize> = (0..n_obs).map(|i| (i * 3) / n_obs).collect();
        let group_names = vec!["ref".to_string(), "ko_a".to_string(), "ko_b".to_string()];
        let reference = Some(0usize);
        let log_transformed = false;
        let rankby_abs = false;
        let tie_correct = true;

        let csr_reader = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);
        let cpu_res = wilcoxon_rank_sum_streaming(
            &csr_reader,
            &gene_names,
            &groups,
            &group_names,
            reference,
            7, // gene_chunk_size — multi-chunk on n_vars=20 (3 chunks)
            log_transformed,
            rankby_abs,
            tie_correct,
        )
        .expect("CPU streaming wilcoxon failed");

        // Force graph capture ON over the backed multi-shard reader (Wilcoxon
        // is always the v1 dense-chunk driver — no v2/v3/CSC override needed).
        let prev_graphs = scx_gpu::set_cuda_graphs_enabled_override(Some(true));
        let csr_reader_gpu = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);
        let gpu_res = crate::diffexp_gpu::wilcoxon_rank_sum_gpu(
            0,
            crate::diffexp_gpu::GpuDeShardInput::Backed {
                csr: &csr_reader_gpu,
                csc: None,
            },
            &gene_names,
            &groups,
            &group_names,
            reference,
            Some(7),
            log_transformed,
            rankby_abs,
            tie_correct,
        );
        scx_gpu::set_cuda_graphs_enabled_override(prev_graphs);
        let gpu_res =
            gpu_res.expect("GPU v1 backed multi-chunk wilcoxon under graph capture failed");

        assert_eq!(cpu_res.group_names, gpu_res.group_names);

        use std::collections::HashMap;
        let group_to_map =
            |res: &crate::diffexp::DiffExpResult, g: usize| -> HashMap<String, (f64, f64)> {
                res.names[g]
                    .iter()
                    .zip(res.scores[g].iter().zip(res.pvals[g].iter()))
                    .map(|(n, (&s, &p))| (n.clone(), (s, p)))
                    .collect()
            };

        for g in 0..cpu_res.group_names.len() {
            let c_map = group_to_map(&cpu_res, g);
            let gpu_map = group_to_map(&gpu_res, g);
            for gene in &gene_names {
                let (s_c, p_c) = c_map.get(gene).copied().unwrap_or((f64::NAN, 1.0));
                let (s_g, p_g) = gpu_map.get(gene).copied().unwrap_or((f64::NAN, 1.0));
                if s_c.is_finite() && s_g.is_finite() {
                    assert!(
                        (s_c - s_g).abs() < 1e-3,
                        "score mismatch (v1 graph) group={} gene={gene}: cpu={s_c}, gpu={s_g}",
                        cpu_res.group_names[g]
                    );
                }
                let pdiff = (p_c - p_g).abs();
                assert!(
                    pdiff < 1e-6 || pdiff / p_c.abs().max(1e-30) < 1e-3,
                    "p mismatch (v1 graph) group={} gene={gene}: cpu={p_c}, gpu={p_g}",
                    cpu_res.group_names[g]
                );
            }
        }
    }
}
