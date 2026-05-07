//! Phase L.4 — proptest round-trips for the CSR↔CSC transpose.
//!
//! Generates random small CSR matrices and verifies that:
//!
//! 1. `csr_to_csc(csr).to_csr() == csr` (round-trip exact).
//! 2. The streaming chunked transpose
//!    (`streaming_csr_to_csc_iter_with_cap`) produces the same
//!    column-axis arrays (indptr / indices / data, modulo column-
//!    range slicing) as the in-memory `csr_to_csc(...)`.
//! 3. Densifying both CSR and CSC views produces the same dense
//!    matrix (parity with a row-major dense reference).
//!
//! Parallel to the codec round-trip proptest in
//! `scx-codec/tests/proptest_roundtrip.rs`.

use proptest::prelude::*;

use scx_sparse::transpose::{csr_to_csc, streaming_csr_to_csc_iter_with_cap, CscArrays};
use scx_sparse::{ScxCsc, ScxCsr};

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Generate a small CSR matrix as `(n_rows, n_cols, indptr, indices, data)`.
/// The sparsity pattern is a per-row count drawn uniformly in
/// `[0, max_nnz_per_row]`, with sorted unique column indices.
fn arb_csr(
    max_rows: usize,
    max_cols: usize,
    max_nnz_per_row: usize,
) -> impl Strategy<Value = ScxCsr> {
    (1..=max_rows, 1..=max_cols).prop_flat_map(move |(n_rows, n_cols)| {
        let nnz_per_row_strat = prop::collection::vec(0..=max_nnz_per_row.min(n_cols), n_rows);
        nnz_per_row_strat.prop_flat_map(move |row_nnzs| {
            let total_nnz: usize = row_nnzs.iter().sum();

            let idx_strats: Vec<_> = row_nnzs
                .iter()
                .map(|&nnz| {
                    if nnz == 0 {
                        Just(vec![]).boxed()
                    } else {
                        prop::collection::hash_set(0u32..(n_cols as u32), nnz)
                            .prop_map(|set| {
                                let mut v: Vec<u32> = set.into_iter().collect();
                                v.sort_unstable();
                                v
                            })
                            .boxed()
                    }
                })
                .collect();

            (idx_strats, prop::collection::vec(1..200u32, total_nnz)).prop_map(
                move |(per_row, values_u32)| {
                    let mut indptr = Vec::with_capacity(n_rows + 1);
                    indptr.push(0i64);
                    let mut indices: Vec<i32> = Vec::new();
                    for row in &per_row {
                        for &c in row {
                            indices.push(c as i32);
                        }
                        indptr.push(indices.len() as i64);
                    }
                    let data: Vec<f32> = values_u32.iter().map(|&v| v as f32).collect();
                    ScxCsr::new_unchecked((n_rows, n_cols), indptr, indices, data)
                },
            )
        })
    })
}

// Densify helpers. Small fixtures only; we cap n_rows × n_cols at
// 50 × 50 in arb_csr so the dense buffers stay tiny.
fn csr_to_dense(csr: &ScxCsr) -> Vec<f32> {
    let (n_rows, n_cols) = csr.shape;
    let mut dense = vec![0.0f32; n_rows * n_cols];
    for r in 0..n_rows {
        let s = csr.indptr[r] as usize;
        let e = csr.indptr[r + 1] as usize;
        for j in s..e {
            dense[r * n_cols + csr.indices[j] as usize] = csr.data[j];
        }
    }
    dense
}

fn csc_arrays_to_dense(c: &CscArrays) -> Vec<f32> {
    let (n_rows, n_cols) = c.shape;
    let mut dense = vec![0.0f32; n_rows * n_cols];
    for col in 0..n_cols {
        let s = c.indptr[col] as usize;
        let e = c.indptr[col + 1] as usize;
        for j in s..e {
            dense[c.indices[j] as usize * n_cols + col] = c.data[j];
        }
    }
    dense
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// `csr_to_csc(csr)` then back to CSR via `ScxCsc::to_csr()`
    /// reproduces the input row-for-row.
    #[test]
    fn csr_csc_csr_roundtrip(csr in arb_csr(20, 20, 8)) {
        let csc_arrays = csr_to_csc(&csr);
        let csc = ScxCsc::new_unchecked(
            csc_arrays.shape,
            csc_arrays.indptr.clone(),
            csc_arrays.indices.clone(),
            csc_arrays.data.clone(),
        );
        let back = csc.to_csr().unwrap();
        prop_assert_eq!(back.shape, csr.shape);
        prop_assert_eq!(back.indptr, csr.indptr);
        prop_assert_eq!(back.indices, csr.indices);
        prop_assert_eq!(back.data, csr.data);
    }

    /// CSR and CSC views of the same data densify to the same matrix.
    #[test]
    fn csr_csc_dense_parity(csr in arb_csr(20, 20, 8)) {
        let csc = csr_to_csc(&csr);
        let csr_dense = csr_to_dense(&csr);
        let csc_dense = csc_arrays_to_dense(&csc);
        prop_assert_eq!(csr_dense, csc_dense);
    }

    /// The streaming-with-cap transpose produces the same column-axis
    /// arrays as the in-memory `csr_to_csc(...)`. We chain the
    /// emitted shards back together (concatenating indices / data and
    /// shifting indptr by the running nnz offset) and compare.
    #[test]
    fn streaming_chunked_matches_in_memory(
        csr in arb_csr(20, 20, 8),
        memory_cap_kb in 1u64..16,
        max_cols in 1usize..32,
    ) {
        let in_memory = csr_to_csc(&csr);
        let memory_cap = (memory_cap_kb * 1024) as usize;
        let n_rows = csr.shape.0;
        let n_cols = csr.shape.1;

        // The iterator takes a `&[ScxCsr]` (treats each as a shard).
        // We feed it as a single-shard slice; the test isn't about
        // multi-shard reassembly (that's covered by Phase A tests).
        let shards = vec![csr];
        let it = streaming_csr_to_csc_iter_with_cap(
            &shards,
            n_rows,
            n_cols,
            memory_cap,
            max_cols,
        ).unwrap();

        let mut concat_indptr: Vec<i64> = vec![0];
        let mut concat_indices: Vec<i32> = Vec::new();
        let mut concat_data: Vec<f32> = Vec::new();
        let mut total_cols_emitted: usize = 0;
        for chunk_res in it {
            let chunk = chunk_res.unwrap();
            // chunk.shape == (n_rows, chunk_n_cols)
            prop_assert_eq!(chunk.shape.0, n_rows);
            let chunk_cols = chunk.shape.1;
            let base_nnz = *concat_indptr.last().unwrap();
            for &p in &chunk.indptr[1..] {
                concat_indptr.push(p + base_nnz);
            }
            concat_indices.extend_from_slice(&chunk.indices);
            concat_data.extend_from_slice(&chunk.data);
            total_cols_emitted += chunk_cols;
        }
        prop_assert_eq!(total_cols_emitted, n_cols);

        prop_assert_eq!(concat_indptr, in_memory.indptr);
        prop_assert_eq!(concat_indices, in_memory.indices);
        prop_assert_eq!(concat_data, in_memory.data);
    }
}
