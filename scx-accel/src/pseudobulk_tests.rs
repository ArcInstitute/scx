//! Tests for [`super`] — the pseudobulk aggregation kernels.
//!
//! The bit-identity tests use a **float** fixture whose f64 sums are
//! order-sensitive; the original integer fixture (`make_test_csr`) cannot see a
//! reordering, because small integers sum exactly in any order.

use super::*;

fn make_test_csr() -> scx_sparse::ScxCsr {
    // 6 cells × 4 genes
    // Cell 0: gene0=1, gene1=2
    // Cell 1: gene0=3, gene2=4
    // Cell 2: gene1=5, gene3=6
    // Cell 3: gene0=7, gene1=8
    // Cell 4: gene2=9, gene3=10
    // Cell 5: gene0=11
    let indptr = vec![0i64, 2, 4, 6, 8, 10, 11];
    let indices = vec![0i32, 1, 0, 2, 1, 3, 0, 1, 2, 3, 0];
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    scx_sparse::ScxCsr::new_unchecked((6, 4), indptr, indices, data)
}

#[test]
fn test_pseudobulk_sum_inmemory() {
    let csr = make_test_csr();
    // Groups: cells 0,1,2 → "A", cells 3,4,5 → "B"
    let obs_groups = vec![vec![
        "A".to_string(),
        "A".to_string(),
        "A".to_string(),
        "B".to_string(),
        "B".to_string(),
        "B".to_string(),
    ]];
    let groupby = vec!["group".to_string()];
    let genes = vec![
        "g0".to_string(),
        "g1".to_string(),
        "g2".to_string(),
        "g3".to_string(),
    ];

    let result = pseudobulk_aggregate_inmemory(
        &csr,
        &obs_groups,
        &groupby,
        &genes,
        AggregationMethod::Sum,
        0,
    )
    .unwrap();

    assert_eq!(result.n_groups, 2);
    assert_eq!(result.n_vars, 4);
    assert_eq!(result.cell_counts, vec![3, 3]);

    // Group A (cells 0,1,2): g0=1+3=4, g1=2+5=7, g2=4, g3=6
    let a_idx = result
        .group_labels
        .iter()
        .position(|l| l[0] == "A")
        .unwrap();
    let a_row = &result.counts[a_idx * 4..(a_idx + 1) * 4];
    assert_eq!(a_row, &[4.0, 7.0, 4.0, 6.0]);

    // Group B (cells 3,4,5): g0=7+11=18, g1=8, g2=9, g3=10
    let b_idx = result
        .group_labels
        .iter()
        .position(|l| l[0] == "B")
        .unwrap();
    let b_row = &result.counts[b_idx * 4..(b_idx + 1) * 4];
    assert_eq!(b_row, &[18.0, 8.0, 9.0, 10.0]);
}

#[test]
fn test_pseudobulk_mean_inmemory() {
    let csr = make_test_csr();
    let obs_groups = vec![vec![
        "A".to_string(),
        "A".to_string(),
        "A".to_string(),
        "B".to_string(),
        "B".to_string(),
        "B".to_string(),
    ]];
    let groupby = vec!["group".to_string()];
    let genes = vec![
        "g0".to_string(),
        "g1".to_string(),
        "g2".to_string(),
        "g3".to_string(),
    ];

    let result = pseudobulk_aggregate_inmemory(
        &csr,
        &obs_groups,
        &groupby,
        &genes,
        AggregationMethod::Mean,
        0,
    )
    .unwrap();

    let a_idx = result
        .group_labels
        .iter()
        .position(|l| l[0] == "A")
        .unwrap();
    let a_row = &result.counts[a_idx * 4..(a_idx + 1) * 4];
    // Mean of group A: sum / 3
    assert!((a_row[0] - 4.0 / 3.0).abs() < 1e-10);
    assert!((a_row[1] - 7.0 / 3.0).abs() < 1e-10);
}

#[test]
fn test_min_cells_filter() {
    let csr = make_test_csr();
    // 3 groups: A (cells 0,1), B (cell 2), C (cells 3,4,5)
    let obs_groups = vec![vec![
        "A".to_string(),
        "A".to_string(),
        "B".to_string(),
        "C".to_string(),
        "C".to_string(),
        "C".to_string(),
    ]];
    let groupby = vec!["group".to_string()];
    let genes = vec![
        "g0".to_string(),
        "g1".to_string(),
        "g2".to_string(),
        "g3".to_string(),
    ];

    // min_cells=2 → B (1 cell) should be excluded
    let result = pseudobulk_aggregate_inmemory(
        &csr,
        &obs_groups,
        &groupby,
        &genes,
        AggregationMethod::Sum,
        2,
    )
    .unwrap();

    assert_eq!(result.n_groups, 2);
    let labels: Vec<&str> = result.group_labels.iter().map(|l| l[0].as_str()).collect();
    assert!(labels.contains(&"A"));
    assert!(labels.contains(&"C"));
    assert!(!labels.contains(&"B"));
}

#[test]
fn test_multi_column_groupby() {
    let csr = make_test_csr();
    // Two groupby columns: perturbation and donor
    let obs_groups = vec![
        vec![
            "drug".to_string(),
            "drug".to_string(),
            "ctrl".to_string(),
            "ctrl".to_string(),
            "drug".to_string(),
            "drug".to_string(),
        ],
        vec![
            "d1".to_string(),
            "d1".to_string(),
            "d1".to_string(),
            "d2".to_string(),
            "d2".to_string(),
            "d2".to_string(),
        ],
    ];
    let groupby = vec!["perturbation".to_string(), "donor".to_string()];
    let genes = vec![
        "g0".to_string(),
        "g1".to_string(),
        "g2".to_string(),
        "g3".to_string(),
    ];

    let result = pseudobulk_aggregate_inmemory(
        &csr,
        &obs_groups,
        &groupby,
        &genes,
        AggregationMethod::Sum,
        0,
    )
    .unwrap();

    // Groups: (ctrl, d1)→cell2, (ctrl, d2)→cell3, (drug, d1)→cells0,1, (drug, d2)→cells4,5
    assert_eq!(result.n_groups, 4);
    assert_eq!(result.groupby_columns, vec!["perturbation", "donor"]);

    // Check (drug, d1): cells 0,1 → g0=1+3=4, g1=2, g2=4, g3=0
    let drug_d1_idx = result
        .group_labels
        .iter()
        .position(|l| l[0] == "drug" && l[1] == "d1")
        .unwrap();
    let row = &result.counts[drug_d1_idx * 4..(drug_d1_idx + 1) * 4];
    assert_eq!(row, &[4.0, 2.0, 4.0, 0.0]);
    assert_eq!(result.cell_counts[drug_d1_idx], 2);
}

#[test]
fn test_validation_errors() {
    let csr = make_test_csr();
    let genes = vec![
        "g0".to_string(),
        "g1".to_string(),
        "g2".to_string(),
        "g3".to_string(),
    ];

    // Empty obs_groups
    let err = pseudobulk_aggregate_inmemory(&csr, &[], &[], &genes, AggregationMethod::Sum, 0);
    assert!(err.is_err());

    // Wrong number of cells
    let bad_groups = vec![vec!["A".to_string(), "B".to_string()]]; // only 2 cells, need 6
    let err = pseudobulk_aggregate_inmemory(
        &csr,
        &bad_groups,
        &["group".to_string()],
        &genes,
        AggregationMethod::Sum,
        0,
    );
    assert!(err.is_err());

    // Wrong number of genes
    let obs = vec![vec!["A".to_string(); 6]];
    let bad_genes = vec!["g0".to_string(), "g1".to_string()]; // only 2, need 4
    let err = pseudobulk_aggregate_inmemory(
        &csr,
        &obs,
        &["group".to_string()],
        &bad_genes,
        AggregationMethod::Sum,
        0,
    );
    assert!(err.is_err());
}

#[test]
fn test_geom_mean_mode_transforms() {
    // Each row: (mode, x, expected_pre, mean (sum/2), expected_post(mean))
    let cases: &[(GeomMeanMode, f64)] = &[
        (GeomMeanMode::ArithRaw, 2.5),
        (GeomMeanMode::ArithLog1pExpand, 1.5),
        (GeomMeanMode::GeomRaw, 1.5),
        (GeomMeanMode::GeomLog1p, 0.5),
    ];

    for &(mode, x) in cases {
        // f(0) == 0 invariant — required for CSR aggregation correctness.
        assert!(
            mode.pre(0.0).abs() < 1e-15,
            "{:?}.pre(0.0) must equal 0 (got {})",
            mode,
            mode.pre(0.0)
        );

        // Match pdex's _math.pseudobulk reference behavior on a single value.
        let pre = mode.pre(x);
        let post = mode.post(pre);
        let expected = match mode {
            GeomMeanMode::ArithRaw => x,
            GeomMeanMode::ArithLog1pExpand => x.exp_m1(),
            GeomMeanMode::GeomRaw => x.ln_1p().exp_m1(), // = x for x > -1
            GeomMeanMode::GeomLog1p => x.exp_m1(),
        };
        assert!(
            (post - expected).abs() < 1e-12,
            "{:?} round-trip: post(pre({})) = {} != {}",
            mode,
            x,
            post,
            expected
        );
    }
}

#[test]
fn test_geom_mean_mode_from_flags() {
    assert_eq!(
        GeomMeanMode::from_flags(false, false),
        GeomMeanMode::ArithRaw
    );
    assert_eq!(
        GeomMeanMode::from_flags(false, true),
        GeomMeanMode::ArithLog1pExpand
    );
    assert_eq!(GeomMeanMode::from_flags(true, false), GeomMeanMode::GeomRaw);
    assert_eq!(
        GeomMeanMode::from_flags(true, true),
        GeomMeanMode::GeomLog1p
    );
}

#[test]
fn test_pseudobulk_dense_matches_csr_inmemory() {
    // The dense kernel should produce bit-identical sums to the CSR
    // kernel when given the dense expansion of the same matrix.
    let csr = make_test_csr();
    let (n_obs, n_vars) = csr.shape;
    let mut dense = vec![0.0f32; n_obs * n_vars];
    for row in 0..n_obs {
        let s = csr.indptr[row] as usize;
        let e = csr.indptr[row + 1] as usize;
        for j in s..e {
            let col = csr.indices[j] as usize;
            dense[row * n_vars + col] = csr.data[j];
        }
    }

    let obs_groups = vec![vec![
        "A".to_string(),
        "A".to_string(),
        "A".to_string(),
        "B".to_string(),
        "B".to_string(),
        "B".to_string(),
    ]];
    let groupby = vec!["group".to_string()];
    let genes = vec![
        "g0".to_string(),
        "g1".to_string(),
        "g2".to_string(),
        "g3".to_string(),
    ];

    for method in [AggregationMethod::Sum, AggregationMethod::Mean] {
        let csr_res =
            pseudobulk_aggregate_inmemory(&csr, &obs_groups, &groupby, &genes, method, 0).unwrap();
        let dense_res = pseudobulk_aggregate_dense(
            &dense,
            (n_obs, n_vars),
            &obs_groups,
            &groupby,
            &genes,
            method,
            0,
        )
        .unwrap();
        assert_eq!(csr_res.n_groups, dense_res.n_groups);
        assert_eq!(csr_res.cell_counts, dense_res.cell_counts);
        assert_eq!(csr_res.group_labels, dense_res.group_labels);
        // f64 sums of the exact same f32 values in the same ascending-cell
        // order on both partitions; bit-identical, not merely close.
        for (a, b) in csr_res.counts.iter().zip(dense_res.counts.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "method={method:?} mismatch: csr={a} dense={b}"
            );
        }
    }
}
/// The parse vocabulary and its exact error text are a cross-binding
/// contract (pyscx surfaces the message as `RuntimeError`, rscx as an R
/// error) — pinned here so `cargo test -p scx-accel` catches drift.
#[test]
fn aggregation_method_parse_vocabulary_and_error_text() {
    assert!(matches!(
        AggregationMethod::parse("sum"),
        Ok(AggregationMethod::Sum)
    ));
    assert!(matches!(
        AggregationMethod::parse("mean"),
        Ok(AggregationMethod::Mean)
    ));
    assert_eq!(
        AggregationMethod::parse("median").unwrap_err().to_string(),
        "unsupported aggr_method 'median': use 'sum' or 'mean'"
    );
}

// ──────────────────────────────────────────────────────────────────────────
// Bit-identity of the partitioned scatter against the serial loop.
// ──────────────────────────────────────────────────────────────────────────

use scx_sparse::ScxCsr;

/// The pre-partition loop, kept as the oracle: cells ascending, each row's
/// nonzeros in stored order, one dependent `+=` per nonzero.
fn serial_oracle(csr: &ScxCsr, cell_to_group: &[usize], n_groups: usize) -> Vec<f64> {
    let n_vars = csr.shape.1;
    let mut counts = vec![0.0f64; n_groups * n_vars];
    for (row, &g) in cell_to_group.iter().enumerate() {
        let (s, e) = (csr.indptr[row] as usize, csr.indptr[row + 1] as usize);
        for j in s..e {
            counts[g * n_vars + csr.indices[j] as usize] += csr.data[j] as f64;
        }
    }
    counts
}

/// The same sums formed with the cells in **descending** order — what a kernel
/// that visits rows backwards computes. Exists to prove the fixture can tell
/// the two apart.
fn descending_oracle(csr: &ScxCsr, cell_to_group: &[usize], n_groups: usize) -> Vec<f64> {
    let n_vars = csr.shape.1;
    let mut counts = vec![0.0f64; n_groups * n_vars];
    for (row, &g) in cell_to_group.iter().enumerate().rev() {
        let (s, e) = (csr.indptr[row] as usize, csr.indptr[row + 1] as usize);
        for j in s..e {
            counts[g * n_vars + csr.indices[j] as usize] += csr.data[j] as f64;
        }
    }
    counts
}

fn assert_bits_eq(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "{what}[{i}]: {a} != {b}");
    }
}

/// A sparse CSR whose f64 sums are order-sensitive: ~half the columns of each
/// row populated, positive values with a mantissa in `[1, 2)` and an exponent
/// cycling over `10^{-5..5}`, so a handful of terms spans more than f64's 53
/// bits and every reassociation lands on different low bits. Canonical rows.
fn reassociating_csr(n_rows: usize, n_vars: usize, seed: u64) -> ScxCsr {
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
    for r in 0..n_rows {
        for c in 0..n_vars {
            let bits = next();
            if bits & 1 == 0 {
                continue;
            }
            let mant = 1.0 + (bits >> 40) as f32 / 16_777_216.0;
            let exp = 10f32.powi((((r + c) % 11) as i32) - 5);
            indices.push(c as i32);
            data.push(mant * exp);
        }
        indptr.push(indices.len() as i64);
    }
    ScxCsr::new_unchecked((n_rows, n_vars), indptr, indices, data)
}

/// `cell i → group i % n_groups`, labelled so lexicographic order is numeric
/// order (`build_group_mapping` sorts labels).
fn cyclic_groups(n_rows: usize, n_groups: usize) -> Vec<Vec<String>> {
    vec![(0..n_rows)
        .map(|i| format!("g{:04}", i % n_groups))
        .collect()]
}

fn gene_names(n_vars: usize) -> Vec<String> {
    (0..n_vars).map(|j| format!("gene{j}")).collect()
}

/// Two descending rows and a duplicated coordinate — the shape a scipy CSR
/// arrives in when nobody called `sort_indices()` / `sum_duplicates()`.
fn non_canonical_csr() -> ScxCsr {
    // Row 0: cols 3, 1, 0 (descending).  Row 1: col 2 twice, then 0.
    // Row 2: canonical.                   Row 3: cols 3, 3 (duplicate only).
    ScxCsr::new_unchecked(
        (4, 4),
        vec![0, 3, 6, 8, 10],
        vec![3, 1, 0, 2, 2, 0, 1, 3, 3, 3],
        vec![
            2.5e3, 1.5e-4, 4.0, 0.5e5, 0.25, 3.0e-3, 7.0, 1.0e4, 1.0e-3, 9.0e2,
        ],
    )
}

#[test]
fn the_float_fixture_is_order_sensitive() {
    let csr = reassociating_csr(240, 16, 1);
    let (ctg, labels) = build_group_mapping(&cyclic_groups(240, 3), 240);
    let asc = serial_oracle(&csr, &ctg, labels.len());
    let desc = descending_oracle(&csr, &ctg, labels.len());
    let moved = asc
        .iter()
        .zip(&desc)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(
        moved > 0,
        "premise: summing the fixture backwards must move at least one bit, or no test \
         here can see a reordering"
    );
    // And the integer fixture cannot: this is the trap the float one exists for.
    let int_csr = make_test_csr();
    let (ictg, ilabels) = build_group_mapping(&cyclic_groups(6, 2), 6);
    assert_bits_eq(
        &descending_oracle(&int_csr, &ictg, ilabels.len()),
        &serial_oracle(&int_csr, &ictg, ilabels.len()),
        "integer fixture is order-blind",
    );
}

/// Drive `scatter_rows` at fixed block counts, inside fixed-width pools, and
/// compare bits with the serial oracle. A kernel walking rows backwards fails
/// here and passes on `make_test_csr`.
#[test]
fn column_blocks_are_bit_identical_to_the_serial_oracle_for_every_block_count() {
    let (n_rows, n_vars) = (400usize, 64usize);
    let csr = reassociating_csr(n_rows, n_vars, 2);
    let (ctg, labels) = build_group_mapping(&cyclic_groups(n_rows, 5), n_rows);
    let n_groups = labels.len();
    let want = serial_oracle(&csr, &ctg, n_groups);
    assert!(
        rows_strictly_increasing_par(CsrRows::of(&csr)),
        "premise: the fixture is canonical"
    );

    for threads in [1usize, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            for n_blocks in [1usize, 2, 3, 5, 17, 64, 100] {
                let mut counts = vec![0.0f64; n_groups * n_vars];
                let mut weights = vec![0u64; n_vars];
                scatter_rows(
                    CsrRows::of(&csr),
                    &ctg,
                    &mut counts,
                    n_vars,
                    n_groups,
                    ScatterPartition::ColumnBlocks(n_blocks),
                    Some(&mut weights),
                );
                assert_bits_eq(
                    &counts,
                    &want,
                    &format!("{n_blocks} blocks on {threads} threads"),
                );
                // Without a histogram (the in-memory paths): even blocks,
                // no bump, same bits.
                let mut counts_even = vec![0.0f64; n_groups * n_vars];
                scatter_rows(
                    CsrRows::of(&csr),
                    &ctg,
                    &mut counts_even,
                    n_vars,
                    n_groups,
                    ScatterPartition::ColumnBlocks(n_blocks),
                    None,
                );
                assert_bits_eq(
                    &counts_even,
                    &want,
                    &format!("{n_blocks} even blocks on {threads} threads"),
                );
                // The one-block walk skips the histogram (a second store per
                // nonzero on a memory-bound loop); every split bumps each
                // column's weight exactly once.
                if n_blocks > 1 {
                    assert_eq!(
                        weights.iter().sum::<u64>() as usize,
                        csr.indices.len(),
                        "every nonzero bumps its column's weight exactly once"
                    );
                } else {
                    assert!(weights.iter().all(|&w| w == 0));
                }
            }
        });
    }
}

/// A weights histogram carried in from earlier rows changes the block plan and
/// nothing else.
#[test]
fn a_carried_weights_histogram_moves_the_plan_but_not_the_bits() {
    let (n_rows, n_vars) = (300usize, 32usize);
    let csr = reassociating_csr(n_rows, n_vars, 3);
    let (ctg, labels) = build_group_mapping(&cyclic_groups(n_rows, 4), n_rows);
    let n_groups = labels.len();
    let want = serial_oracle(&csr, &ctg, n_groups);

    // Skewed prior: the first four columns look enormously heavy.
    let mut skewed = vec![1u64; n_vars];
    skewed[..4].iter_mut().for_each(|w| *w = 1_000_000);
    let even = vec![0u64; n_vars];
    assert_ne!(
        colblocks::plan_blocks(&skewed, 4),
        colblocks::plan_blocks(&even, 4),
        "premise: the two histograms plan differently"
    );
    for mut weights in [skewed, even] {
        let mut counts = vec![0.0f64; n_groups * n_vars];
        scatter_rows(
            CsrRows::of(&csr),
            &ctg,
            &mut counts,
            n_vars,
            n_groups,
            ScatterPartition::ColumnBlocks(4),
            Some(&mut weights),
        );
        assert_bits_eq(&counts, &want, "skewed vs even prior");
    }
}

#[test]
fn the_group_partition_is_bit_identical_on_non_canonical_rows() {
    let csr = non_canonical_csr();
    assert!(
        !rows_strictly_increasing_par(CsrRows::of(&csr)),
        "premise: the fixture must be non-canonical"
    );
    let (ctg, labels) = build_group_mapping(&cyclic_groups(4, 2), 4);
    let n_groups = labels.len();
    let want = serial_oracle(&csr, &ctg, n_groups);
    // Pin the duplicate's value too: row 1 (group 1) holds col 2 twice, and a
    // kernel that coalesced by overwriting would answer 0.25 or 0.5e5, not
    // their sum. Row 3 (group 1) holds col 3 twice.
    assert_eq!(want[4 + 2], 0.5e5f32 as f64 + 0.25f32 as f64);
    assert_eq!(want[4 + 3], 1.0e-3f32 as f64 + 9.0e2f32 as f64);

    let mut counts = vec![0.0f64; n_groups * 4];
    let mut weights = vec![0u64; 4];
    scatter_rows(
        CsrRows::of(&csr),
        &ctg,
        &mut counts,
        4,
        n_groups,
        ScatterPartition::ByGroup,
        Some(&mut weights),
    );
    assert_bits_eq(&counts, &want, "group partition on non-canonical rows");
    assert!(
        weights.iter().all(|&w| w == 0),
        "ByGroup leaves the histogram alone"
    );
}

#[test]
fn the_group_partition_matches_the_oracle_on_the_float_fixture_too() {
    let (n_rows, n_vars) = (200usize, 24usize);
    let csr = reassociating_csr(n_rows, n_vars, 4);
    let (ctg, labels) = build_group_mapping(&cyclic_groups(n_rows, 7), n_rows);
    let n_groups = labels.len();
    let want = serial_oracle(&csr, &ctg, n_groups);
    let mut counts = vec![0.0f64; n_groups * n_vars];
    scatter_rows(
        CsrRows::of(&csr),
        &ctg,
        &mut counts,
        n_vars,
        n_groups,
        ScatterPartition::ByGroup,
        Some(&mut vec![0u64; n_vars]),
    );
    assert_bits_eq(&counts, &want, "group partition on canonical float rows");
}

#[test]
fn choose_partition_routes_by_balance_canonicality_and_size() {
    // Too little work to split: one block, whatever the grouping.
    let small = reassociating_csr(10, 8, 5);
    assert!(small.indices.len() < MIN_NNZ_FOR_BLOCKS);
    let (ctg, labels) = build_group_mapping(&cyclic_groups(10, 2), 10);
    assert_eq!(
        choose_partition(CsrRows::of(&small), &ctg, labels.len(), 8),
        ScatterPartition::ColumnBlocks(1)
    );

    // Long rows (~512 nonzeros): a four-block window is 128 wide, above the
    // floor. Short rows (~32 nonzeros) are the serial walk whatever the pool.
    let (n_rows, n_vars) = (400usize, 1024usize);
    let big = reassociating_csr(n_rows, n_vars, 6);
    assert!(big.indices.len() >= MIN_NNZ_FOR_BLOCKS);
    assert!(
        big.indices.len() / n_rows / 4 >= MIN_BLOCK_WINDOW,
        "premise: long rows"
    );
    let short = reassociating_csr(n_rows, 64, 6);
    assert!(short.indices.len() >= MIN_NNZ_FOR_BLOCKS);
    assert!(
        short.indices.len() / n_rows / 2 < MIN_BLOCK_WINDOW,
        "premise: short rows"
    );
    let one_group = vec![vec!["a".to_string(); n_rows]];
    // A four-wide pool pins `block_count` at 4.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        assert_eq!(colblocks::block_count(n_vars), 4, "premise: four blocks");
        // Balanced groups, the largest within two pool-shares: by group.
        let (ctg, labels) = build_group_mapping(&cyclic_groups(n_rows, 5), n_rows);
        assert_eq!(
            choose_partition(CsrRows::of(&big), &ctg, labels.len(), n_vars),
            ScatterPartition::ByGroup
        );
        // One group holding everything: the blocks, when canonical and long ...
        let (ctg1, labels1) = build_group_mapping(&one_group, n_rows);
        assert_eq!(
            choose_partition(CsrRows::of(&big), &ctg1, labels1.len(), n_vars),
            ScatterPartition::ColumnBlocks(4)
        );
        // ... one block when the rows are too short to window ...
        assert_eq!(
            choose_partition(CsrRows::of(&short), &ctg1, labels1.len(), 64),
            ScatterPartition::ColumnBlocks(1)
        );
        // ... and by group again when the rows are not canonical.
        let unsorted = reversed_rows(&big);
        assert_eq!(
            choose_partition(CsrRows::of(&unsorted), &ctg1, labels1.len(), n_vars),
            ScatterPartition::ByGroup
        );
        assert_eq!(
            choose_partition(CsrRows::of(&non_canonical_csr()), &[0, 0, 0, 0], 1, 4),
            ScatterPartition::ColumnBlocks(1),
            "a tiny non-canonical matrix is one block — a whole-row walk needs no window"
        );
    });
}

/// Through the public entry point: a non-canonical matrix gives the serial
/// loop's bits, duplicates included.
#[test]
fn inmemory_aggregation_on_a_non_canonical_matrix_matches_the_oracle() {
    let csr = non_canonical_csr();
    let obs = cyclic_groups(4, 2);
    let (ctg, labels) = build_group_mapping(&obs, 4);
    let want = serial_oracle(&csr, &ctg, labels.len());
    let got = pseudobulk_aggregate_inmemory(
        &csr,
        &obs,
        &["g".to_string()],
        &gene_names(4),
        AggregationMethod::Sum,
        0,
    )
    .unwrap();
    assert_bits_eq(&got.counts, &want, "public in-memory path, non-canonical");
}

/// The streaming path over uneven shards — one of them a single row, two of
/// them wide enough for several blocks — equals the in-memory path over the
/// concatenation and the oracle, bit for bit, for `Sum` and `Mean`. Exercises
/// the `global_row` cursor and the weights carried between shards.
#[test]
fn streaming_matches_inmemory_bitwise_across_uneven_shards() {
    // 1 024 columns at half density: ~512 nonzeros per row, wide enough for the
    // one-group arm to split into blocks on the two big shards.
    let n_vars = 1024usize;
    let sizes = [700usize, 1, 4000, 1200];
    let n_rows: usize = sizes.iter().sum();
    let full = reassociating_csr(n_rows, n_vars, 8);
    let mut shards = Vec::new();
    let mut lo = 0usize;
    for &n in &sizes {
        let hi = lo + n;
        let (s, e) = (full.indptr[lo] as usize, full.indptr[hi] as usize);
        let indptr: Vec<i64> = full.indptr[lo..=hi].iter().map(|&p| p - s as i64).collect();
        shards.push(ScxCsr::new_unchecked(
            (n, n_vars),
            indptr,
            full.indices[s..e].to_vec(),
            full.data[s..e].to_vec(),
        ));
        lo = hi;
    }
    let source = crate::test_support::GaugedSource::new(shards, n_rows, n_vars);
    let cols = ["g".to_string()];
    let genes = gene_names(n_vars);
    // Nine cyclic groups take the group partition on every shard (the
    // single-row shard's cell lands in a different group than a per-shard
    // cursor reset would put it); one group takes the column blocks on the
    // wide shards, with the weights histogram carried between them.
    for (n_groups, method) in [
        (9usize, AggregationMethod::Sum),
        (9, AggregationMethod::Mean),
        (1, AggregationMethod::Sum),
        (1, AggregationMethod::Mean),
    ] {
        let obs = cyclic_groups(n_rows, n_groups);
        let (ctg, labels) = build_group_mapping(&obs, n_rows);
        let want = serial_oracle(&full, &ctg, labels.len());
        source.reset();
        let streamed = pseudobulk_aggregate(&source, &obs, &cols, &genes, method, 0).unwrap();
        // The consume closure now runs a nested parallel scatter; the decode
        // prefetch must still overlap it rather than collapse to
        // decode-then-scatter (a single-thread pool takes the sequential
        // fallback by design and is skipped here).
        if crate::test_support::pool_can_prefetch() {
            crate::test_support::assert_prefetch_engaged(
                &source,
                &format!("streaming pseudobulk, {n_groups} groups, {method:?}"),
            );
        }
        let inmem = pseudobulk_aggregate_inmemory(&full, &obs, &cols, &genes, method, 0).unwrap();
        assert_eq!(streamed.cell_counts, inmem.cell_counts);
        assert_eq!(streamed.group_labels, inmem.group_labels);
        assert_bits_eq(
            &streamed.counts,
            &inmem.counts,
            &format!("streaming vs in-memory, {n_groups} groups, {method:?}"),
        );
        if method == AggregationMethod::Sum {
            assert_bits_eq(
                &inmem.counts,
                &want,
                &format!("in-memory vs oracle, {n_groups} groups"),
            );
        } else {
            let mut divided = want.clone();
            for (g, row) in divided.chunks_mut(n_vars).enumerate() {
                let cc = inmem.cell_counts[g] as f64;
                row.iter_mut().for_each(|v| *v /= cc);
            }
            assert_bits_eq(&inmem.counts, &divided, "mean vs serial divide");
        }
    }
}

#[test]
fn from_slices_matches_inmemory_bitwise() {
    let (n_rows, n_vars) = (350usize, 40usize);
    let csr = reassociating_csr(n_rows, n_vars, 9);
    let obs = cyclic_groups(n_rows, 3);
    let cols = ["g".to_string()];
    let genes = gene_names(n_vars);
    for method in [AggregationMethod::Sum, AggregationMethod::Mean] {
        let a = pseudobulk_aggregate_inmemory(&csr, &obs, &cols, &genes, method, 0).unwrap();
        let b = pseudobulk_aggregate_from_slices(
            csr.shape,
            &csr.indptr,
            &csr.indices,
            &csr.data,
            &obs,
            &cols,
            &genes,
            method,
            0,
        )
        .unwrap();
        assert_bits_eq(&a.counts, &b.counts, &format!("from_slices, {method:?}"));
    }
}

#[test]
fn from_slices_rejects_a_short_indptr() {
    let csr = make_test_csr();
    let err = pseudobulk_aggregate_from_slices(
        csr.shape,
        &csr.indptr[..3],
        &csr.indices,
        &csr.data,
        &cyclic_groups(6, 2),
        &["g".to_string()],
        &gene_names(4),
        AggregationMethod::Sum,
        0,
    );
    assert!(
        matches!(err, Err(crate::AccelError::ShapeError(_))),
        "{err:?}"
    );
}

#[test]
fn a_shard_source_overrunning_n_obs_is_a_shape_error() {
    // Two shards of 3 rows on a source that claims 5.
    let csr = make_test_csr();
    let split = |lo: usize, hi: usize| {
        let (s, e) = (csr.indptr[lo] as usize, csr.indptr[hi] as usize);
        ScxCsr::new_unchecked(
            (hi - lo, 4),
            csr.indptr[lo..=hi].iter().map(|&p| p - s as i64).collect(),
            csr.indices[s..e].to_vec(),
            csr.data[s..e].to_vec(),
        )
    };
    let source = crate::test_support::GaugedSource::new(vec![split(0, 3), split(3, 6)], 5, 4);
    let err = pseudobulk_aggregate(
        &source,
        &cyclic_groups(5, 2),
        &["g".to_string()],
        &gene_names(4),
        AggregationMethod::Sum,
        0,
    );
    assert!(
        matches!(err, Err(crate::AccelError::ShapeError(_))),
        "{err:?}"
    );
}

/// Reverse each row of the float fixture so the rows walk col_max → col_min
/// with no duplicates — the shape a scipy CSR arrives in when nobody called
/// `sort_indices()`. Large enough for several blocks, so a kernel that
/// windowed these rows by binary search would drop and misplace nonzeros; a
/// one-block window over every column is right by accident, which is why the
/// four-row fixture above cannot carry this claim.
fn reversed_rows(csr: &ScxCsr) -> ScxCsr {
    let mut indices = csr.indices.clone();
    let mut data = csr.data.clone();
    for r in 0..csr.n_rows() {
        let (s, e) = (csr.indptr[r] as usize, csr.indptr[r + 1] as usize);
        indices[s..e].reverse();
        data[s..e].reverse();
    }
    ScxCsr::new_unchecked(csr.shape, csr.indptr.clone(), indices, data)
}

#[test]
fn a_large_unsorted_matrix_matches_the_oracle_through_the_public_path() {
    // Long rows, so the sorted twin takes four blocks in a four-wide pool.
    let (n_rows, n_vars) = (400usize, 1024usize);
    let sorted = reassociating_csr(n_rows, n_vars, 12);
    let unsorted = reversed_rows(&sorted);
    assert!(
        unsorted.indices.len() >= MIN_NNZ_FOR_BLOCKS,
        "premise: wide enough to split"
    );
    assert!(
        !rows_strictly_increasing_par(CsrRows::of(&unsorted)),
        "premise: the rows are not canonical"
    );
    // One group: the balance rule alone would already pick the group
    // partition for balanced groups, and this test is about the canonical
    // check deciding — a whole-matrix group is the case where only that check
    // stands between these rows and the column blocks.
    let obs = vec![vec!["all".to_string(); n_rows]];
    let (ctg, labels) = build_group_mapping(&obs, n_rows);
    // Same operands per (group, gene) in the same cell order, so the unsorted
    // oracle equals the sorted one — and the kernel must equal both.
    let want = serial_oracle(&unsorted, &ctg, labels.len());
    assert_bits_eq(
        &want,
        &serial_oracle(&sorted, &ctg, labels.len()),
        "oracle is order-blind within a row",
    );
    let cols = ["g".to_string()];
    let genes = gene_names(n_vars);
    // A four-wide pool, so `block_count` would plan several blocks if the
    // routing let the column-block kernel near these rows.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        assert_eq!(
            choose_partition(CsrRows::of(&unsorted), &ctg, labels.len(), n_vars),
            ScatterPartition::ByGroup
        );
        assert_eq!(
            choose_partition(CsrRows::of(&sorted), &ctg, labels.len(), n_vars),
            ScatterPartition::ColumnBlocks(4),
            "premise: the sorted twin does take the blocks"
        );
        let got = pseudobulk_aggregate_inmemory(
            &unsorted,
            &obs,
            &cols,
            &genes,
            AggregationMethod::Sum,
            0,
        )
        .unwrap();
        assert_bits_eq(&got.counts, &want, "public in-memory path on reversed rows");
        let got_sorted =
            pseudobulk_aggregate_inmemory(&sorted, &obs, &cols, &genes, AggregationMethod::Sum, 0)
                .unwrap();
        assert_bits_eq(
            &got.counts,
            &got_sorted.counts,
            "sorted and reversed input agree bitwise",
        );
    });
}

#[test]
fn zero_groups_give_an_empty_result_on_every_csr_path() {
    // An empty matrix: every obs column has zero entries, so the mapping has
    // no groups and the kernels must return rather than index into nothing.
    let csr = ScxCsr::new_unchecked((0, 4), vec![0], vec![], vec![]);
    let obs: Vec<Vec<String>> = vec![vec![]];
    let cols = ["g".to_string()];
    let genes = gene_names(4);
    for method in [AggregationMethod::Sum, AggregationMethod::Mean] {
        let a = pseudobulk_aggregate_inmemory(&csr, &obs, &cols, &genes, method, 0).unwrap();
        assert_eq!((a.n_groups, a.n_vars, a.counts.len()), (0, 4, 0));
        let b = pseudobulk_aggregate_from_slices(
            csr.shape,
            &csr.indptr,
            &csr.indices,
            &csr.data,
            &obs,
            &cols,
            &genes,
            method,
            0,
        )
        .unwrap();
        assert_eq!((b.n_groups, b.counts.len()), (0, 0));
        let source = crate::test_support::GaugedSource::new(vec![], 0, 4);
        let c = pseudobulk_aggregate(&source, &obs, &cols, &genes, method, 0).unwrap();
        assert_eq!((c.n_groups, c.counts.len()), (0, 0));
    }
    // And `choose_partition` itself, handed zero groups on a non-empty chunk.
    let big = reassociating_csr(400, 64, 6);
    assert_eq!(
        choose_partition(CsrRows::of(&big), &[], 0, 64),
        ScatterPartition::ColumnBlocks(1)
    );
}

/// A source whose shards cover fewer rows than it claims: every cell was
/// counted in `cell_counts`, so a mean over the partial sums would be silently
/// wrong. The overrun twin is `a_shard_source_overrunning_n_obs_is_a_shape_error`.
#[test]
fn a_shard_source_underrunning_n_obs_is_a_shape_error() {
    let csr = make_test_csr();
    let (s, e) = (csr.indptr[0] as usize, csr.indptr[3] as usize);
    let first_half = ScxCsr::new_unchecked(
        (3, 4),
        csr.indptr[0..=3].to_vec(),
        csr.indices[s..e].to_vec(),
        csr.data[s..e].to_vec(),
    );
    let source = crate::test_support::GaugedSource::new(vec![first_half], 6, 4);
    let err = pseudobulk_aggregate(
        &source,
        &cyclic_groups(6, 2),
        &["g".to_string()],
        &gene_names(4),
        AggregationMethod::Mean,
        0,
    );
    assert!(
        matches!(err, Err(crate::AccelError::ShapeError(_))),
        "{err:?}"
    );
}

/// The borrowed-slices entry point is public and its callers hand it raw
/// numpy buffers; every malformed triple is a `ShapeError`, never a panic on
/// a rayon worker or a silently dropped tail.
#[test]
fn from_slices_rejects_malformed_csr_triples() {
    let csr = make_test_csr();
    let obs = cyclic_groups(6, 2);
    let cols = ["g".to_string()];
    let genes = gene_names(4);
    let run = |indptr: &[i64], indices: &[i32], data: &[f32]| {
        pseudobulk_aggregate_from_slices(
            csr.shape,
            indptr,
            indices,
            data,
            &obs,
            &cols,
            &genes,
            AggregationMethod::Sum,
            0,
        )
    };
    let shape_error =
        |r: Result<PseudobulkResult>| matches!(r, Err(crate::AccelError::ShapeError(_)));
    // indices / data length mismatch (a short `data` would otherwise zip-drop).
    assert!(shape_error(run(
        &csr.indptr,
        &csr.indices,
        &csr.data[..csr.data.len() - 1]
    )));
    // indptr past the end of indices.
    let mut long = csr.indptr.clone();
    *long.last_mut().unwrap() += 5;
    assert!(shape_error(run(&long, &csr.indices, &csr.data)));
    // inverted offsets.
    let mut inverted = csr.indptr.clone();
    inverted[2] = inverted[3] + 1;
    assert!(shape_error(run(&inverted, &csr.indices, &csr.data)));
    // negative start.
    let mut neg = csr.indptr.clone();
    neg[0] = -1;
    assert!(shape_error(run(&neg, &csr.indices, &csr.data)));
    // positive start (would silently drop the leading nonzeros).
    let mut late = csr.indptr.clone();
    late[0] = 1;
    assert!(shape_error(run(&late, &csr.indices, &csr.data)));
    // trailing nonzeros past the last offset (would silently drop the tail).
    let mut idx_long = csr.indices.clone();
    idx_long.push(0);
    let mut dat_long = csr.data.clone();
    dat_long.push(1.0);
    assert!(shape_error(run(&csr.indptr, &idx_long, &dat_long)));
    // a column out of range, high and negative (would panic on a worker).
    let mut high = csr.indices.clone();
    high[5] = 4;
    assert!(shape_error(run(&csr.indptr, &high, &csr.data)));
    let mut negative = csr.indices.clone();
    negative[5] = -1;
    assert!(shape_error(run(&csr.indptr, &negative, &csr.data)));
    // The well-formed triple still goes through.
    assert!(run(&csr.indptr, &csr.indices, &csr.data).is_ok());
}
