//! CSC pseudobulk aggregation restricted to a gene subset.
//!
//! The full-gene CSR pseudobulk path already streams shards row-major
//! and accumulates per-group sums; CSC offers no measurable win
//! against it. Where CSC pays off is the *projected* case: a caller
//! has narrowed the gene axis (`col_indices`) and wants to read only
//! those columns. With CSC we skip non-overlapping shards entirely
//! and decode just the requested columns; with CSR we'd decode every
//! row and project per-shard.
//!
//! The kernel signature reflects this: `col_indices` is required.
//! Passing the full gene set works but defeats the purpose; pyscx
//! gates on this and rejects empty `col_indices` for the CSC path.
//!
//! A contiguous run of requested columns wide enough to split is scattered in
//! parallel over its columns (each task owns whole columns, into a tile-local
//! scratch of at most `CSC_RUN_TILE_COLS` columns); a single-column run — the
//! scattered-gene-subset production shape — keeps the serial loop straight into
//! its output column. Both are
//! bit-identical to the serial per-run loop they replaced: a column's rows are
//! visited in the ascending order the CSC stores them. See `pseudobulk.rs` for
//! the CSR twin and the reasoning.

use rayon::prelude::*;

use crate::error::{AccelError, Result};
use crate::pca::colblocks;
use crate::pseudobulk::{apply_mean, filter_and_build_result, AggregationMethod, PseudobulkResult};
use scx_format_io::ColumnShardSource;
use scx_sparse::ScxCsc;

/// Columns per tile of a wide contiguous run: the run-local scratch is at most
/// `CSC_RUN_TILE_COLS × n_groups` f64s (20 MB at 10k groups) however wide the
/// projection, never a second copy of the result.
const CSC_RUN_TILE_COLS: usize = 256;

/// Add one decoded column's nonzeros into `dst(group, value)`, rows in the
/// ascending order the CSC stores them, skipping rows and group ids out of
/// range exactly as the serial loop did. Generic so both call sites inline it.
#[inline]
fn accumulate_column<F: FnMut(usize, f64)>(
    run: &ScxCsc,
    local_col: usize,
    n_obs: usize,
    cell_to_group: &[usize],
    n_groups: usize,
    mut dst: F,
) {
    let s = run.indptr[local_col] as usize;
    let e = run.indptr[local_col + 1] as usize;
    for k in s..e {
        let row = run.indices[k] as usize;
        if row >= n_obs {
            continue;
        }
        let group_idx = cell_to_group[row];
        if group_idx >= n_groups {
            continue;
        }
        dst(group_idx, run.data[k] as f64);
    }
}

/// Aggregate counts per group on a CSC source, restricted to the
/// `col_indices` gene subset.
///
/// `cell_to_group[cell] -> group_idx in 0..n_groups`. Groups absent
/// from `cell_to_group` (no contributing cell) end up at zero, but are
/// filtered out by `min_cells_per_group` if requested.
///
/// Output `PseudobulkResult.counts` has shape `[n_groups × col_indices.len()]`
/// (row-major). `gene_names` should match the *projected* gene set —
/// the caller is responsible for slicing the original gene name vector
/// by `col_indices` before passing it in.
#[allow(clippy::too_many_arguments)]
pub fn pseudobulk_aggregate_csc<S: ColumnShardSource + ?Sized>(
    source: &S,
    cell_to_group: &[usize],
    n_groups: usize,
    group_labels: Vec<Vec<String>>,
    groupby_columns: &[String],
    gene_names: &[String],
    col_indices: &[u32],
    method: AggregationMethod,
    min_cells_per_group: usize,
) -> Result<PseudobulkResult> {
    let n_obs = source.n_obs();
    let n_proj = col_indices.len();

    if cell_to_group.len() != n_obs {
        return Err(AccelError::InvalidInput(format!(
            "cell_to_group length {} != source.n_obs() {}",
            cell_to_group.len(),
            n_obs
        )));
    }
    if n_proj == 0 {
        return Err(AccelError::InvalidInput(
            "col_indices must be non-empty for the CSC pseudobulk path".to_string(),
        ));
    }
    if gene_names.len() != n_proj {
        return Err(AccelError::ShapeError(format!(
            "gene_names length {} != col_indices length {}",
            gene_names.len(),
            n_proj
        )));
    }
    if group_labels.len() != n_groups {
        return Err(AccelError::ShapeError(format!(
            "group_labels has {} entries but n_groups = {}",
            group_labels.len(),
            n_groups
        )));
    }

    let mut cell_counts = vec![0usize; n_groups];
    for &g in cell_to_group {
        if g < n_groups {
            cell_counts[g] += 1;
        }
    }
    // The old serial loop's `min_cells_per_group <= 1` shortcut kept every
    // group, zero-cell ones included; a floor of 0 reproduces that through the
    // shared builder.
    let min_cells = if min_cells_per_group <= 1 {
        0
    } else {
        min_cells_per_group
    };
    if n_groups == 0 {
        // Nothing to accumulate into, and `par_chunks_mut(0)` would panic
        // below (reachable from pyscx on an empty AnnData with a gene subset).
        return filter_and_build_result(
            Vec::new(),
            group_labels,
            groupby_columns,
            cell_counts,
            gene_names,
            0,
            n_proj,
            min_cells,
        );
    }

    // `read_csc_columns_subset` is on `BackedCscReader` only, not on
    // the trait. Walk `col_indices` in sorted contiguous-run order and
    // call the trait's `read_csc_columns(Range<u32>)` per run.
    //
    // Sort col_indices and remember their original output positions.
    let mut sorted_with_pos: Vec<(u32, usize)> = col_indices
        .iter()
        .copied()
        .enumerate()
        .map(|(i, c)| (c, i))
        .collect();
    sorted_with_pos.sort_by_key(|(c, _)| *c);

    let mut counts = vec![0.0f64; n_groups * n_proj];
    let mut i = 0;
    while i < sorted_with_pos.len() {
        let mut j = i + 1;
        while j < sorted_with_pos.len() && sorted_with_pos[j].0 == sorted_with_pos[j - 1].0 + 1 {
            j += 1;
        }
        let run_start = sorted_with_pos[i].0;
        let run_end = sorted_with_pos[j - 1].0 + 1;

        let csc_run = source
            .read_csc_columns(run_start..run_end)
            .map_err(AccelError::Scx)?;

        // Consecutive sorted columns differ by exactly one, so the run's width
        // is its span in `sorted_with_pos`; a duplicated column index breaks a
        // run, so each output position still gets its own copy.
        let run_n_cols = (run_end - run_start) as usize;
        debug_assert_eq!(run_n_cols, j - i);
        if csc_run.indptr.len() < run_n_cols + 1 {
            return Err(AccelError::ShapeError(format!(
                "read_csc_columns({run_start}..{run_end}) returned {} columns, expected {run_n_cols}",
                csc_run.indptr.len().saturating_sub(1)
            )));
        }
        if run_n_cols == 1 {
            // The production shape — a scattered gene subset decodes one column
            // per run — has nothing to parallelise across and pays no scratch:
            // the serial loop straight into its output column, as before.
            let output_col = sorted_with_pos[i].1;
            accumulate_column(&csc_run, 0, n_obs, cell_to_group, n_groups, |g, v| {
                counts[g * n_proj + output_col] += v
            });
        } else {
            // A contiguous run wide enough to split, in tiles of at most
            // `CSC_RUN_TILE_COLS` columns: a tile-local column-major scratch
            // (bounded whatever the run's width — a contiguous projection of
            // every gene is a valid input and must not hold the result twice),
            // one whole column per task with the fan-out capped like the CSR
            // blocks', then a transpose of the tile into its requested output
            // positions. Within a column the rows are visited in the ascending
            // order the CSC stores them, exactly as the serial loop did, so
            // every `(group, column)` sum is formed from the same operands in
            // the same order — bit-identical on any thread count.
            let tile = run_n_cols.min(CSC_RUN_TILE_COLS);
            let mut scratch = vec![0.0f64; tile * n_groups];
            let mut lo = 0usize;
            while lo < run_n_cols {
                let hi = (lo + tile).min(run_n_cols);
                let width = hi - lo;
                let scratch = &mut scratch[..width * n_groups];
                scratch.fill(0.0);
                let per_task = width.div_ceil(colblocks::block_count(width));
                scratch
                    .par_chunks_mut(n_groups * per_task)
                    .enumerate()
                    .for_each(|(t, cols)| {
                        for (k, dst) in cols.chunks_mut(n_groups).enumerate() {
                            let local_col = lo + t * per_task + k;
                            accumulate_column(
                                &csc_run,
                                local_col,
                                n_obs,
                                cell_to_group,
                                n_groups,
                                |g, v| dst[g] += v,
                            );
                        }
                    });
                for (k, &(_, output_col)) in sorted_with_pos[i + lo..i + hi].iter().enumerate() {
                    let col = &scratch[k * n_groups..(k + 1) * n_groups];
                    for (g, &v) in col.iter().enumerate() {
                        counts[g * n_proj + output_col] = v;
                    }
                }
                lo = hi;
            }
        }

        i = j;
    }

    if method == AggregationMethod::Mean {
        apply_mean(&mut counts, &cell_counts, n_proj);
    }

    filter_and_build_result(
        counts,
        group_labels,
        groupby_columns,
        cell_counts,
        gene_names,
        n_groups,
        n_proj,
        min_cells,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csc::test_helpers::{deterministic_dense, write_csr_csc_test_file};
    use crate::pseudobulk::pseudobulk_aggregate;
    use scx_format_io::{BackedCscReader, BackedCsrReader, ScxReader};
    use tempfile::tempdir;

    #[test]
    fn pseudobulk_csc_matches_csr_on_projected_subset() {
        let dir = tempdir().unwrap();
        let n_obs = 12usize;
        let n_vars = 8usize;
        let dense = deterministic_dense(n_obs, n_vars);
        let path = write_csr_csc_test_file(dir.path(), "pb", n_obs, n_vars, &dense, 4);

        let groupby_columns = vec!["batch".to_string()];
        let labels: Vec<String> = (0..n_obs)
            .map(|i| {
                if i < n_obs / 2 {
                    "A".to_string()
                } else {
                    "B".to_string()
                }
            })
            .collect();
        let obs_groups = vec![labels];
        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();

        let csr_reader = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);
        let csr_full = pseudobulk_aggregate(
            &csr_reader,
            &obs_groups,
            &groupby_columns,
            &gene_names,
            AggregationMethod::Sum,
            0,
        )
        .unwrap();

        // Project to a subset of genes.
        let col_indices: Vec<u32> = vec![1, 3, 5, 7];
        let projected_gene_names: Vec<String> = col_indices
            .iter()
            .map(|&c| gene_names[c as usize].clone())
            .collect();

        // Build cell_to_group/group_labels matching the CSR full result's
        // group ordering.
        let n_groups = csr_full.n_groups;
        let group_labels = csr_full.group_labels.clone();
        // Build cell_to_group from obs_groups using the same key->index
        // map the CSR path produced. We already know n_groups == 2 with
        // group_labels[[A]], [[B]] in lexicographic order.
        let cell_to_group: Vec<usize> = (0..n_obs)
            .map(|i| {
                let label = if i < n_obs / 2 { "A" } else { "B" };
                group_labels.iter().position(|gl| gl[0] == label).unwrap()
            })
            .collect();

        let csc_reader = BackedCscReader::new(ScxReader::open(&path).unwrap(), 0).unwrap();
        let csc_proj = pseudobulk_aggregate_csc(
            &csc_reader,
            &cell_to_group,
            n_groups,
            group_labels.clone(),
            &groupby_columns,
            &projected_gene_names,
            &col_indices,
            AggregationMethod::Sum,
            0,
        )
        .unwrap();

        // Compare slice-by-slice: CSR full result projected to the
        // selected columns should match the CSC projected result.
        for g in 0..n_groups {
            for (out_col, &orig_col) in col_indices.iter().enumerate() {
                let csr_val = csr_full.counts[g * csr_full.n_vars + orig_col as usize];
                let csc_val = csc_proj.counts[g * csc_proj.n_vars + out_col];
                assert!(
                    (csr_val - csc_val).abs() < 1e-9,
                    "g={g} out_col={out_col} (orig {orig_col}): csr={csr_val} csc={csc_val}"
                );
            }
        }
    }

    /// An in-memory column source over one `ScxCsc`, so the kernel can be
    /// driven on **float** values (the file helper writes `Uint8`, and small
    /// integers sum exactly in any order — a reordering could not be seen).
    struct SingleCsc {
        csc: scx_sparse::ScxCsc,
    }

    impl ColumnShardSource for SingleCsc {
        fn n_csc_shards(&self) -> usize {
            1
        }
        fn n_obs(&self) -> usize {
            self.csc.n_rows()
        }
        fn n_vars(&self) -> usize {
            self.csc.n_cols()
        }
        fn read_csc_shard(&self, _shard_idx: usize) -> scx_format_io::Result<scx_sparse::ScxCsc> {
            Ok(self.csc.clone())
        }
        fn read_csc_columns(
            &self,
            col_range: std::ops::Range<u32>,
        ) -> scx_format_io::Result<scx_sparse::ScxCsc> {
            Ok(self
                .csc
                .col_slice(col_range.start as usize, col_range.end as usize)
                .expect("test double: in-range column slice"))
        }
        fn csc_shard_col_range(&self, shard_idx: usize) -> Option<(u32, u32)> {
            (shard_idx == 0).then(|| (0, self.csc.n_cols() as u32))
        }
    }

    /// Order-sensitive CSC: ~half the rows of each column populated, values
    /// with a mantissa in `[1, 2)` and an exponent cycling over `10^{-5..5}`.
    fn reassociating_csc(n_obs: usize, n_vars: usize, seed: u64) -> scx_sparse::ScxCsc {
        let mut state = 0x2545_F491_4F6C_DD1Du64 ^ seed.wrapping_mul(0x9E37_79B9);
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        for c in 0..n_vars {
            for r in 0..n_obs {
                let bits = next();
                if bits & 1 == 0 {
                    continue;
                }
                let mant = 1.0 + (bits >> 40) as f32 / 16_777_216.0;
                let exp = 10f32.powi((((r + c) % 11) as i32) - 5);
                indices.push(r as i32);
                data.push(mant * exp);
            }
            indptr.push(indices.len() as i64);
        }
        scx_sparse::ScxCsc::new_unchecked((n_obs, n_vars), indptr, indices, data)
    }

    /// The per-run serial loop OPT-ACCEL-4 replaced, as the oracle.
    fn serial_csc_oracle(
        source: &SingleCsc,
        cell_to_group: &[usize],
        n_groups: usize,
        col_indices: &[u32],
        mean: bool,
    ) -> Vec<f64> {
        let n_obs = source.n_obs();
        let n_proj = col_indices.len();
        let mut counts = vec![0.0f64; n_groups * n_proj];
        let mut cell_counts = vec![0usize; n_groups];
        for &g in cell_to_group {
            if g < n_groups {
                cell_counts[g] += 1;
            }
        }
        let mut sorted_with_pos: Vec<(u32, usize)> = col_indices
            .iter()
            .copied()
            .enumerate()
            .map(|(i, c)| (c, i))
            .collect();
        sorted_with_pos.sort_by_key(|(c, _)| *c);
        let mut i = 0;
        while i < sorted_with_pos.len() {
            let mut j = i + 1;
            while j < sorted_with_pos.len() && sorted_with_pos[j].0 == sorted_with_pos[j - 1].0 + 1
            {
                j += 1;
            }
            let (run_start, run_end) = (sorted_with_pos[i].0, sorted_with_pos[j - 1].0 + 1);
            let run = source.read_csc_columns(run_start..run_end).unwrap();
            for local_col in 0..(run_end - run_start) as usize {
                let output_col = sorted_with_pos[i + local_col].1;
                let (s, e) = (
                    run.indptr[local_col] as usize,
                    run.indptr[local_col + 1] as usize,
                );
                for k in s..e {
                    let row = run.indices[k] as usize;
                    if row >= n_obs {
                        continue;
                    }
                    let g = cell_to_group[row];
                    if g >= n_groups {
                        continue;
                    }
                    counts[g * n_proj + output_col] += run.data[k] as f64;
                }
            }
            i = j;
        }
        if mean {
            for g in 0..n_groups {
                if cell_counts[g] > 0 {
                    let cc = cell_counts[g] as f64;
                    for v in &mut counts[g * n_proj..(g + 1) * n_proj] {
                        *v /= cc;
                    }
                }
            }
        }
        counts
    }

    /// The parallel-within-run scatter equals the serial per-run loop bit for
    /// bit: a scattered subset (single-column runs), a contiguous run wide
    /// enough to split, a duplicated column, a group nobody belongs to, and a
    /// cell whose group index is out of range (skipped on both sides).
    #[test]
    fn csc_scatter_is_bit_identical_to_the_serial_per_run_loop() {
        let (n_obs, n_vars) = (600usize, 40usize);
        let source = SingleCsc {
            csc: reassociating_csc(n_obs, n_vars, 3),
        };
        let n_groups = 5usize;
        // cell → group i % 4, group 4 stays empty, one cell out of range.
        let mut cell_to_group: Vec<usize> = (0..n_obs).map(|i| i % 4).collect();
        cell_to_group[17] = 99;
        let group_labels: Vec<Vec<String>> = (0..n_groups).map(|g| vec![format!("g{g}")]).collect();
        // Runs: [3], [7..=15] (nine wide), [20], [22, 22] (duplicate), [39].
        let col_indices: Vec<u32> = vec![20, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 39, 22, 22];
        let gene_names: Vec<String> = col_indices.iter().map(|c| format!("g{c}")).collect();

        for method in [AggregationMethod::Sum, AggregationMethod::Mean] {
            let want = serial_csc_oracle(
                &source,
                &cell_to_group,
                n_groups,
                &col_indices,
                method == AggregationMethod::Mean,
            );
            // Premise: the fixture can tell a reordering apart — summing the
            // rows of one column backwards must move a bit somewhere.
            if method == AggregationMethod::Sum {
                let rev = {
                    let n_proj = col_indices.len();
                    let mut counts = vec![0.0f64; n_groups * n_proj];
                    for (out_col, &c) in col_indices.iter().enumerate() {
                        let run = source.read_csc_columns(c..c + 1).unwrap();
                        for k in (0..run.indices.len()).rev() {
                            let g = cell_to_group[run.indices[k] as usize];
                            if g < n_groups {
                                counts[g * n_proj + out_col] += run.data[k] as f64;
                            }
                        }
                    }
                    counts
                };
                assert!(
                    rev.iter()
                        .zip(&want)
                        .any(|(a, b)| a.to_bits() != b.to_bits()),
                    "premise: the float fixture must be order-sensitive"
                );
            }
            for threads in [1usize, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let got = pool.install(|| {
                    pseudobulk_aggregate_csc(
                        &source,
                        &cell_to_group,
                        n_groups,
                        group_labels.clone(),
                        &["g".to_string()],
                        &gene_names,
                        &col_indices,
                        method,
                        0,
                    )
                    .unwrap()
                });
                assert_eq!(got.n_groups, n_groups);
                // 600 cells cycle over four groups; cell 17 (group 1) is out of range.
                assert_eq!(got.cell_counts, vec![150, 149, 150, 150, 0]);
                assert_eq!(got.counts.len(), want.len());
                for (i, (a, b)) in got.counts.iter().zip(&want).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{method:?} on {threads} threads, [{i}]: {a} != {b}"
                    );
                }
            }
        }
    }

    /// Zero groups (an empty AnnData with `prefer_format="csc"` and a gene
    /// subset reaches this from pyscx) is an empty result, not a
    /// `par_chunks_mut(0)` panic — for a single-column run and a wide one.
    #[test]
    fn zero_groups_give_an_empty_result_instead_of_panicking() {
        let source = SingleCsc {
            csc: reassociating_csc(50, 12, 4),
        };
        let cell_to_group = vec![0usize; 50]; // every id out of range for n_groups = 0
        for col_indices in [vec![3u32], vec![2u32, 3, 4, 5]] {
            let gene_names: Vec<String> = col_indices.iter().map(|c| format!("g{c}")).collect();
            for method in [AggregationMethod::Sum, AggregationMethod::Mean] {
                let got = pseudobulk_aggregate_csc(
                    &source,
                    &cell_to_group,
                    0,
                    Vec::new(),
                    &["g".to_string()],
                    &gene_names,
                    &col_indices,
                    method,
                    0,
                )
                .unwrap();
                assert_eq!(
                    (got.n_groups, got.n_vars, got.counts.len()),
                    (0, col_indices.len(), 0)
                );
                assert!(got.group_labels.is_empty() && got.cell_counts.is_empty());
            }
        }
    }

    /// A contiguous run wider than one tile is scattered tile by tile into a
    /// bounded scratch; the result is still the serial per-run loop's, bit for
    /// bit, with the tile boundaries landing mid-run.
    #[test]
    fn a_run_wider_than_the_tile_matches_the_serial_loop_bitwise() {
        let (n_obs, n_vars) = (200usize, 640usize);
        let source = SingleCsc {
            csc: reassociating_csc(n_obs, n_vars, 5),
        };
        let n_groups = 3usize;
        let cell_to_group: Vec<usize> = (0..n_obs).map(|i| i % n_groups).collect();
        let group_labels: Vec<Vec<String>> = (0..n_groups).map(|g| vec![format!("g{g}")]).collect();
        // 600 contiguous columns: two full tiles and a partial third.
        let col_indices: Vec<u32> = (20u32..620).collect();
        assert!(
            col_indices.len() > 2 * CSC_RUN_TILE_COLS,
            "premise: wider than two tiles"
        );
        let gene_names: Vec<String> = col_indices.iter().map(|c| format!("g{c}")).collect();
        for method in [AggregationMethod::Sum, AggregationMethod::Mean] {
            let want = serial_csc_oracle(
                &source,
                &cell_to_group,
                n_groups,
                &col_indices,
                method == AggregationMethod::Mean,
            );
            let got = pseudobulk_aggregate_csc(
                &source,
                &cell_to_group,
                n_groups,
                group_labels.clone(),
                &["g".to_string()],
                &gene_names,
                &col_indices,
                method,
                0,
            )
            .unwrap();
            assert_eq!(got.counts.len(), want.len());
            for (i, (a, b)) in got.counts.iter().zip(&want).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{method:?} [{i}]: {a} != {b}");
            }
        }
    }
}
