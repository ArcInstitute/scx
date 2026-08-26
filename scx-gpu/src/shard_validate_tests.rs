//! CPU-runnable tests for the shared shard validators.
//!
//! `scx-gpu` is a default workspace member that compiles without CUDA, so these
//! run under a plain `cargo test --workspace` and in CI. They cover the
//! validator's arithmetic; that it is actually *wired* to the CSC staging path
//! is a separate, GPU-gated test in `gpu_csc_shard_source.rs`, and the
//! end-to-end route coverage lives in `scx-accel`.

use super::*;
use scx_sparse::ScxCsc;

/// The pre-§8.14 CSC entry point, as a test shim at the default policy.
///
/// The twenty scanner tests below predate `ValidationLevel` and assert its
/// arithmetic, not its gating: which offender is named, that columns are
/// scanned independently, that the parallel and serial arms agree. Routing them
/// through the default policy — which runs every rung, exactly as the old
/// entry point did — keeps them covering what they covered. The gating itself
/// is tested separately, in `validation_level_tests.rs`.
fn validate_csc_shard_for_gpu(
    csc: &ScxCsc,
    n_obs: usize,
    col_start: usize,
) -> Result<(), GpuError> {
    validate_shard(
        ShardToValidate::Csc {
            csc,
            n_obs,
            col_start,
        },
        &ValidationPolicy::default(),
    )
}

/// `n_cols` columns, each holding rows `0..per_col` in strictly increasing
/// order — the shape `scx_sparse::transpose` emits. Every column restarts at
/// row 0, so a validator that ran `windows(2)` over the flat `indices` array
/// instead of per column would reject it.
fn make_csc(n_cols: usize, per_col: usize) -> ScxCsc {
    let mut indptr = Vec::with_capacity(n_cols + 1);
    indptr.push(0i64);
    let mut indices = Vec::with_capacity(n_cols * per_col);
    let mut data = Vec::with_capacity(n_cols * per_col);
    for c in 0..n_cols {
        for r in 0..per_col {
            indices.push(r as i32);
            data.push((c * per_col + r) as f32);
        }
        indptr.push(indices.len() as i64);
    }
    ScxCsc::new_unchecked((per_col, n_cols), indptr, indices, data)
}

/// Keep the first `n_cols` columns, so the same corruption can be replayed
/// below the parallel threshold and the two scans compared.
fn truncate_csc(csc: &ScxCsc, n_cols: usize) -> ScxCsc {
    let nnz = csc.indptr[n_cols] as usize;
    ScxCsc::new_unchecked(
        (csc.shape.0, n_cols),
        csc.indptr[..=n_cols].to_vec(),
        csc.indices[..nnz].to_vec(),
        csc.data[..nnz].to_vec(),
    )
}

/// Skip rather than fail when `SCX_GPU_VALIDATE_PAR_MIN_NNZ` puts the parallel
/// path out of reach for a fixture of `nnz`. A premise assertion should fire
/// when the *fixture* is wrong, not when the environment deliberately disables
/// the feature under test.
fn parallel_path_reachable(nnz: usize) -> bool {
    if nnz >= validate_par_min_nnz() {
        return true;
    }
    eprintln!(
        "SCX_GPU_VALIDATE_PAR_MIN_NNZ={} puts the parallel scan out of reach for a \
         {nnz}-nnz fixture — skipping",
        validate_par_min_nnz()
    );
    false
}

fn expect_invalid(csc: &ScxCsc, n_obs: usize, col_start: usize) -> String {
    match validate_csc_shard_for_gpu(csc, n_obs, col_start) {
        Err(GpuError::InvalidShard(msg)) => msg,
        other => panic!("expected InvalidShard, got {other:?}"),
    }
}

#[test]
fn accepts_a_clean_shard() {
    let csc = make_csc(4, 8);
    assert!(validate_csc_shard_for_gpu(&csc, 8, 0).is_ok());
    // Empty shards are legal — the staging path skips them before this point,
    // but the validator must not invent an offender for one.
    let empty = ScxCsc::new_unchecked((8, 0), vec![0], Vec::new(), Vec::new());
    assert!(validate_csc_shard_for_gpu(&empty, 8, 0).is_ok());
}

/// A NaN reaching `block_radix_sort_per_gene_kernel` sorts on its raw IEEE-754
/// bit pattern, above `+INF`, and silently corrupts U / tie counts / p-values.
#[test]
fn rejects_non_finite_values() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut csc = make_csc(4, 8);
        csc.data[13] = bad;
        let msg = expect_invalid(&csc, 8, 0);
        assert!(
            msg.contains("non-finite") && msg.contains("at nonzero index 13"),
            "must name the offending position: {msg}"
        );
    }
}

/// Two nonzeros at the same `(cell, gene)` are two threads writing one
/// `slab[gene, pos]` cell in `csc_shard_to_gene_major_kernel`.
///
/// The fixture starts at global column 500 on purpose: a validator that
/// reported the shard-local column would say 2, and the assertion below would
/// fail. Without that, `col_start` could be deleted and every test would pass.
#[test]
fn rejects_duplicate_row_in_a_column_and_names_the_global_column() {
    let mut csc = make_csc(4, 8);
    let s = csc.indptr[2] as usize;
    csc.indices[s + 3] = csc.indices[s + 2]; // duplicate row within column 2

    let msg = expect_invalid(&csc, 8, 500);
    assert_eq!(
        msg,
        "ScxCsc column 502 has unsorted or duplicate row indices (2 >= 2): the GPU shard \
         scatter behind this GPU operation requires strictly-increasing per-column row \
         indices so every (gene, cell) has exactly one writer (SCX-written CSC sidecars are \
         strictly increasing by construction)"
    );
}

/// Distinct-but-unordered rows are rejected too. Documented fail-closed false
/// negative: the kernels need only distinctness, but SCX's transpose never
/// emits this shape, and proving distinctness on unordered input costs a
/// seen-set over `n_obs`.
#[test]
fn rejects_unordered_but_distinct_rows() {
    let mut csc = make_csc(4, 8);
    let s = csc.indptr[1] as usize;
    csc.indices.swap(s + 2, s + 5);
    let msg = expect_invalid(&csc, 8, 0);
    assert!(msg.contains("unsorted or duplicate"), "{msg}");
}

/// `row_indices[e] >= n_obs` is an unguarded `cell_to_group[cell]` device read.
#[test]
fn rejects_out_of_range_rows() {
    let mut high = make_csc(4, 8);
    high.indices[9] = 8; // == n_obs
    let msg = expect_invalid(&high, 8, 0);
    assert!(
        msg.contains("row index 8") && msg.contains("outside [0, 8)"),
        "{msg}"
    );

    let mut negative = make_csc(4, 8);
    negative.indices[9] = -1;
    let msg = expect_invalid(&negative, 8, 0);
    assert!(msg.contains("row index -1"), "{msg}");
}

/// Range is checked first: it is the memory-safety invariant, and an
/// out-of-range row also breaks ordering, so the other message would otherwise
/// mask it.
#[test]
fn out_of_range_is_reported_before_disorder() {
    let mut csc = make_csc(4, 8);
    let s0 = csc.indptr[0] as usize;
    csc.indices[s0 + 3] = csc.indices[s0 + 2]; // duplicate in column 0
    let s3 = csc.indptr[3] as usize;
    csc.indices[s3 + 1] = 99; // out of range, in a later column
    let msg = expect_invalid(&csc, 8, 0);
    assert!(msg.contains("row index 99"), "range must win: {msg}");
}

/// The parallel scan must name the **first** offending column, not whichever a
/// worker reaches first, and must agree with the serial scan.
#[test]
fn parallel_reports_the_minimum_offending_column() {
    let per_col = 16;
    let mut csc = make_csc(4096, per_col); // 65 536 nnz
    if !parallel_path_reachable(csc.data.len()) {
        return;
    }

    // Corrupt columns 3000, 977 and 2500 (inserted out of order on purpose).
    for &c in &[3000usize, 977, 2500] {
        let s = csc.indptr[c] as usize;
        csc.indices[s + 5] = csc.indices[s + 4];
    }

    let msg = expect_invalid(&csc, per_col, 0);
    assert!(
        msg.starts_with("ScxCsc column 977 has unsorted or duplicate row indices (4 >= 4)"),
        "parallel scan must report the lowest offending column: {msg}"
    );

    let small = truncate_csc(&csc, 1000);
    assert!(
        small.data.len() < validate_par_min_nnz(),
        "premise: the truncated fixture must take the serial path"
    );
    let small_msg = expect_invalid(&small, per_col, 0);
    assert_eq!(small_msg, msg, "serial and parallel scans must agree");
}

#[test]
fn parallel_reports_the_first_non_finite_position() {
    let mut csc = make_csc(4096, 16);
    if !parallel_path_reachable(csc.data.len()) {
        return;
    }
    csc.data[60_000] = f32::INFINITY;
    csc.data[12_345] = f32::NAN;
    csc.data[40_000] = f32::NEG_INFINITY;

    let msg = expect_invalid(&csc, 16, 0);
    assert!(
        msg.contains("at nonzero index 12345"),
        "must name the first non-finite position: {msg}"
    );
}

#[test]
fn parallel_reports_the_first_out_of_range_position() {
    let mut csc = make_csc(4096, 16);
    if !parallel_path_reachable(csc.data.len()) {
        return;
    }
    csc.indices[60_000] = 16;
    csc.indices[12_345] = 4242;
    csc.indices[40_000] = -7;

    let msg = expect_invalid(&csc, 16, 0);
    assert!(
        msg.contains("row index 4242 at nonzero index 12345"),
        "must name the first out-of-range position: {msg}"
    );
}

/// Columns are scanned independently. `indices` is one flat array, so a naive
/// `windows(2)` over the whole of it would see the boundary pair (last row of
/// column `c`, first row of column `c + 1`) and reject this legal shard.
#[test]
fn accepts_a_large_clean_shard_whose_columns_restart_at_row_zero() {
    let csc = make_csc(4096, 16);
    if !parallel_path_reachable(csc.data.len()) {
        return;
    }
    assert!(validate_csc_shard_for_gpu(&csc, 16, 0).is_ok());
}

// ---------------------------------------------------------------------------
// ValidationLevel — the §8.14 fix
// ---------------------------------------------------------------------------

use scx_sparse::ScxCsr;

/// One CSR row `[0, 1, 2]`, with `data[1]` replaced by `bad`.
fn csr_with(bad: f32) -> ScxCsr {
    ScxCsr::new_unchecked((1, 4), vec![0i64, 3], vec![0i32, 1, 2], vec![1.0, bad, 3.0])
}

fn policy(level: ValidationLevel, op: &'static str) -> ValidationPolicy {
    ValidationPolicy::new(level, op)
}

/// The structural half of the fix: a consumer at `Bounds` never reaches the
/// finiteness scan, so a NaN is simply not its problem.
///
/// This is the behaviour change the PR body has to declare — a NaN now reaches
/// the HVG / preprocess / pseudobulk kernels — and it is the intended one: the
/// CPU paths accept it, and HVG still rejects post-accumulation with a better
/// message.
#[test]
fn bounds_and_scatter_accept_a_non_finite_csr_shard() {
    let nan = csr_with(f32::NAN);
    for level in [ValidationLevel::Bounds, ValidationLevel::Scatter] {
        assert!(
            validate_shard(
                ShardToValidate::Csr(&nan),
                &policy(level, "normalize_total")
            )
            .is_ok(),
            "{level:?} must not run the finiteness scan"
        );
    }
    assert!(validate_shard(
        ShardToValidate::Csr(&nan),
        &policy(ValidationLevel::Ranking, "rank_genes_groups")
    )
    .is_err());
}

/// The textual half: even at the rung that *does* reject, the message names the
/// operation the user called.
///
/// Both halves are needed. A message can be reworded back and a level can be
/// raised back, and each failure is invisible to a test that only checks the
/// other.
#[test]
fn every_message_names_the_calling_op_and_never_says_de() {
    let cases: Vec<(String, &str)> = vec![
        // CSR, unsorted -> Scatter rung
        {
            let csr =
                ScxCsr::new_unchecked((1, 4), vec![0i64, 2], vec![1i32, 1], vec![1.0f32, 2.0]);
            let e = validate_shard(
                ShardToValidate::Csr(&csr),
                &policy(ValidationLevel::Scatter, "pca"),
            )
            .unwrap_err();
            (format!("{e}"), "pca")
        },
        // CSR, non-finite -> Ranking rung
        {
            let e = validate_shard(
                ShardToValidate::Csr(&csr_with(f32::NAN)),
                &policy(ValidationLevel::Ranking, "rank_genes_groups"),
            )
            .unwrap_err();
            (format!("{e}"), "rank_genes_groups")
        },
        // CSC, out of range -> Bounds rung
        {
            let mut csc = make_csc(4, 8);
            csc.indices[3] = 99;
            let e = validate_shard(
                ShardToValidate::Csc {
                    csc: &csc,
                    n_obs: 8,
                    col_start: 0,
                },
                &policy(ValidationLevel::Bounds, "highly_variable_genes"),
            )
            .unwrap_err();
            (format!("{e}"), "highly_variable_genes")
        },
        // CSC, non-finite -> Ranking rung
        {
            let mut csc = make_csc(4, 8);
            csc.data[5] = f32::NAN;
            let e = validate_shard(
                ShardToValidate::Csc {
                    csc: &csc,
                    n_obs: 8,
                    col_start: 0,
                },
                &policy(ValidationLevel::Ranking, "pseudobulk"),
            )
            .unwrap_err();
            (format!("{e}"), "pseudobulk")
        },
    ];

    for (msg, op) in &cases {
        assert!(
            msg.contains(op),
            "message does not name the calling op `{op}`: {msg}"
        );
        // The literals review §8.14 named. A `normalize_total` caller was told
        // that "GPU DE ranking requires finite input (… sanitise/QC before DE)".
        for banned in ["GPU DE ranking requires", "before DE", "GPU DE "] {
            assert!(
                !msg.contains(banned),
                "message still contains the DE-specific literal `{banned}`: {msg}"
            );
        }
    }
    assert_eq!(cases.len(), 4, "all four rung/layout combinations covered");
}

/// The rungs are cumulative and ordered, so `runs()` is a `>=` and not a match.
///
/// Pinned because the whole dispatcher is written as `if policy.runs(rung)`:
/// were the ordering to change, `Ranking` would silently stop running the
/// bounds check and a CSC consumer would lose its out-of-bounds guard while
/// asking for *more* validation.
#[test]
fn levels_are_ordered_and_cumulative() {
    use ValidationLevel::*;
    assert!(Bounds < Scatter && Scatter < Ranking);
    assert!(policy(Ranking, "x").runs(Bounds));
    assert!(policy(Ranking, "x").runs(Scatter));
    assert!(policy(Scatter, "x").runs(Bounds));
    assert!(!policy(Bounds, "x").runs(Scatter));
    assert!(!policy(Scatter, "x").runs(Ranking));
    // The default must be the strictest rung: a consumer that forgets to choose
    // fails closed, at the cost of work rather than of a kernel precondition.
    assert_eq!(ValidationPolicy::default().level, Ranking);
}

/// `Bounds` is a no-op on CSR and a real check on CSC — the asymmetry that
/// makes "every consumer at Bounds" the wrong mental model.
#[test]
fn bounds_is_a_no_op_on_csr_and_a_real_check_on_csc() {
    // A CSR shard that is unsorted AND non-finite passes at `Bounds`, because
    // CSR has no minor-axis range check to run.
    let mut csr = ScxCsr::new_unchecked(
        (1, 4),
        vec![0i64, 3],
        vec![2i32, 1, 0],
        vec![1.0f32, f32::NAN, 3.0],
    );
    csr.indices[0] = 2;
    assert!(validate_shard(
        ShardToValidate::Csr(&csr),
        &policy(ValidationLevel::Bounds, "normalize_total")
    )
    .is_ok());

    // The same rung on CSC does run: an out-of-range row is an out-of-bounds
    // device read, which poisons the CUDA context.
    let mut csc = make_csc(4, 8);
    csc.indices[3] = 99;
    assert!(validate_shard(
        ShardToValidate::Csc {
            csc: &csc,
            n_obs: 8,
            col_start: 0
        },
        &policy(ValidationLevel::Bounds, "highly_variable_genes")
    )
    .is_err());
}
