//! CPU-runnable tests for the shared shard validators.
//!
//! `scx-gpu` is a default workspace member that compiles without CUDA, so these
//! run under a plain `cargo test --workspace` and in CI. They cover the
//! validator's arithmetic; that it is actually *wired* to the CSC staging path
//! is a separate, GPU-gated test in `gpu_csc_shard_source.rs`, and the
//! end-to-end route coverage lives in `scx-accel`.

use super::*;
use scx_sparse::ScxCsc;

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
        "ScxCsc column 502 has unsorted or duplicate row indices (2 >= 2): GPU shard \
         scatter requires strictly-increasing per-column row indices so every (gene, cell) \
         has exactly one writer (SCX-written CSC sidecars are strictly increasing by \
         construction)"
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
