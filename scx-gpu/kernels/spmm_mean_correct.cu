// Mean-centering correction kernel for GPU PCA.
//
// Subtracts a per-column correction vector from each row of a matrix Y.
// Used in randomized PCA: Y = X @ Ω - 1_n × (μ^T @ Ω)
// where mc = μ^T @ Ω is a (1 × k) vector.
//
// Y: row-major [n_obs × k], mc: [k]
// After kernel: Y[row * k + col] -= mc[col]  for all (row, col)

extern "C" __global__ void mean_correct_kernel(
    float* __restrict__ Y,         // [n_obs × k], row-major
    const float* __restrict__ mc,  // [k], mean correction vector
    int n_obs, int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_obs * k;
    if (idx >= total) return;

    int col = idx % k;
    Y[idx] -= mc[col];
}
