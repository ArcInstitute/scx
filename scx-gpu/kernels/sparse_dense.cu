// GPU CSR → dense conversion + HVG projection kernel.
//
// Warp-cooperative row scatter: each warp (32 threads) handles one row.
// Threads cooperatively read the row's non-zeros and scatter them into the
// dense output matrix. If an hvg_map is provided, only mapped columns are
// written (gene projection).
//
// The output matrix must be pre-zeroed by the caller (alloc_zeros).
// Layout: row-major, output[row * n_output_cols + mapped_col] = value.
//
// hvg_map semantics:
//   NULL → identity mapping (col_idx → col_idx, n_output_cols == n_vars)
//   non-NULL → hvg_map[col_idx] gives the output column index, or
//              0xFFFFFFFF to skip (gene not in subset).
//
// INVARIANT: hvg_map MUST be injective over its non-sentinel entries — each
// output column index may appear at most once. The scatter below writes one
// nonzero per lane with a plain (non-atomic) store, so two input columns of the
// same row mapping to the same output column would race with an undefined
// winner (silent corruption). Callers are responsible for guaranteeing
// injectivity; the Rust `upload_hvg_map` helper validates this before upload.

extern "C" __global__ void sparse_to_dense_kernel(
    const long long*     __restrict__ indptr,       // [n_rows + 1], i64
    const int*           __restrict__ indices,      // [nnz], i32
    const float*         __restrict__ data,         // [nnz], f32
    float*               __restrict__ output,       // [n_rows × n_output_cols], pre-zeroed
    const unsigned int*  __restrict__ hvg_map,      // [n_vars] → output col, or NULL
    int n_rows,
    int n_output_cols
) {
    // Each warp handles one row. Warp lane = threadIdx.x % 32.
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= n_rows) return;

    // Row boundaries from indptr (i64 → unsigned range)
    long long row_start = indptr[warp_id];
    long long row_end   = indptr[warp_id + 1];
    long long row_nnz   = row_end - row_start;

    // Bounds check: skip if row is empty or indptr is invalid
    if (row_nnz <= 0) return;

    // Base pointer into the output dense row
    float* out_row = output + (long long)warp_id * n_output_cols;

    // Warp-cooperative scatter: each lane handles elements at
    // lane_id, lane_id+32, lane_id+64, ...
    for (long long j = lane_id; j < row_nnz; j += 32) {
        long long nz_idx = row_start + j;
        int col = indices[nz_idx];
        float val = data[nz_idx];

        // Apply HVG mapping if provided
        int out_col;
        if (hvg_map != NULL) {
            unsigned int mapped = hvg_map[col];
            if (mapped == 0xFFFFFFFFu) continue;  // gene not in subset
            out_col = (int)mapped;
        } else {
            out_col = col;
        }

        // Bounds check on output column
        if (out_col >= 0 && out_col < n_output_cols) {
            out_row[out_col] = val;
        }
    }
}
