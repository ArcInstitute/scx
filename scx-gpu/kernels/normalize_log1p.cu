// Fused per-row normalize_total + log1p on GPU-resident CSR.
//
// Each thread handles one row: divides by row sum, multiplies by target_sum,
// applies log1p. Modifies data[] in-place.
//
// Matches the CPU implementation in scx-engine/src/fused_ops.rs:
//   fused_normalize_log1p(data, indptr, row_idx, target_sum)
//
// NOTE: The CPU path computes `(value_f64 * factor) as f32).ln_1p()`.
// The GPU path uses plain f32 arithmetic: `log1pf(value * scale)`.
// For NLP (normalize→log1p), both paths converge because:
//   - Row sums are accumulated in f32 on GPU vs f64 on CPU
//   - The scaling factor is computed in f32 on GPU vs f64 on CPU
// Small rounding differences (~1e-6 relative) are expected.
// Use rtol=1e-6 for validation, not bitwise identity.

extern "C" __global__ void normalize_log1p_kernel(
    const long long* __restrict__ indptr,   // [n_rows + 1], int64_t
    float* __restrict__ data,               // [nnz], modified in-place
    int n_rows,
    float target_sum
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    long long start = indptr[row];
    long long end = indptr[row + 1];

    // Compute row sum
    float sum = 0.0f;
    for (long long i = start; i < end; i++) {
        sum += data[i];
    }
    if (sum == 0.0f) return;

    // Normalize + log1p
    float scale = target_sum / sum;
    for (long long i = start; i < end; i++) {
        data[i] = log1pf(data[i] * scale);
    }
}

// Normalize-only variant: per-row normalize_total without log1p.
// data[i] = data[i] / row_sum * target_sum
extern "C" __global__ void normalize_kernel(
    const long long* __restrict__ indptr,
    float* __restrict__ data,
    int n_rows,
    float target_sum
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    long long start = indptr[row];
    long long end = indptr[row + 1];

    float sum = 0.0f;
    for (long long i = start; i < end; i++) {
        sum += data[i];
    }
    if (sum == 0.0f) return;

    float scale = target_sum / sum;
    for (long long i = start; i < end; i++) {
        data[i] *= scale;
    }
}

// Log1p-only variant: applies log1p to all non-zero values.
// data[i] = log1p(data[i])
extern "C" __global__ void log1p_kernel(
    const long long* __restrict__ indptr,
    float* __restrict__ data,
    int n_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    long long start = indptr[row];
    long long end = indptr[row + 1];

    for (long long i = start; i < end; i++) {
        data[i] = log1pf(data[i]);
    }
}

// Row-scale variant: multiplies each row's non-zero values by an explicit
// per-row factor. data[i] *= factors[row_offset + row].
//
// Unlike normalize_kernel (which computes the factor from the row sum), the
// factor is supplied. `factors` is a global per-row vector indexed in the
// source's iteration row order; `row_offset` is this shard's first global
// row, so one device factor vector serves every shard in a streamed source.
extern "C" __global__ void row_scale_kernel(
    const long long* __restrict__ indptr,
    float* __restrict__ data,
    int n_rows,
    int row_offset,
    const float* __restrict__ factors
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    long long start = indptr[row];
    long long end = indptr[row + 1];

    float factor = factors[row_offset + row];
    for (long long i = start; i < end; i++) {
        data[i] *= factor;
    }
}
