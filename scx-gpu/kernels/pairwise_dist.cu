// Pairwise-distance primitives for GPU energy_distance (Phase 3 of
// CELL-EVAL-SCX-GPU-ACC).
//
// Points are the COLUMNS of a col-major [d x n] matrix — a row-major [n x d]
// host matrix (one point per row) is the same bytes as col-major [d x n].
// Distances derive from a cuBLAS gram matrix G[na x nb] (col-major, element
// (i,j) at i + j*na) where G[i,j] = a_i · b_j:
//   euclidean: d(a_i,b_j) = sqrt(max(0, ||a_i||^2 + ||b_j||^2 - 2*G[i,j]))
//   cosine   : inputs pre-L2-normalized, d = clamp(1 - G[i,j], 0, 2)
// Squared norms and the reduction accumulate in `double` to match the CPU
// f64 path (`scx-accel/src/eval_metrics/distances.rs`); the gemm itself is f32.

// Per-column squared L2 norm of a col-major [d x n] matrix.
// One thread per column (point); serial reduction over d in double.
extern "C" __global__ void col_sqnorm_kernel(
    const float* __restrict__ M,   // [d x n] col-major (column i = point i)
    double* __restrict__ sqnorm,   // [n] output
    int d,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* col = M + (long long)i * d;
    double s = 0.0;
    for (int t = 0; t < d; ++t) {
        double v = (double)col[t];
        s += v * v;
    }
    sqnorm[i] = s;
}

// Sum of all na*nb pairwise distances derived from the gram matrix.
// mode 0 = euclidean (uses a_sq/b_sq), mode 1 = cosine (ignores a_sq/b_sq).
// Grid-stride over the gram elements; block-reduce in double, then a single
// atomicAdd(double) of the block partial into *out_sum (native on CC >= 6.0).
extern "C" __global__ void pairwise_dist_sum_kernel(
    const float* __restrict__ gram,   // [na x nb] col-major: (i,j) at i + j*na
    const double* __restrict__ a_sq,  // [na] (euclidean only)
    const double* __restrict__ b_sq,  // [nb] (euclidean only)
    long long na,
    long long nb,
    int mode,
    double* __restrict__ out_sum
) {
    extern __shared__ double sdata[];
    long long total = na * nb;
    double local = 0.0;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        // idx = i + j*na. Derive (i,j) with one 64-bit division and a
        // mul-subtract — avoids the second slow 64-bit modulo (GPUs emulate
        // 64-bit div/mod in software). Kept 64-bit: na*nb can exceed 2^31 for
        // large control groups, so a 32-bit cast would overflow.
        long long j = idx / na;
        long long i = idx - j * na;
        double g = (double)gram[idx];
        double dist;
        if (mode == 0) {
            double dsq = a_sq[i] + b_sq[j] - 2.0 * g;
            if (dsq < 0.0) dsq = 0.0;   // clamp FP cancellation (near-identical rows)
            dist = sqrt(dsq);
        } else {
            dist = 1.0 - g;
            if (dist < 0.0) dist = 0.0;
            if (dist > 2.0) dist = 2.0;
        }
        local += dist;
    }
    int tid = threadIdx.x;
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out_sum, sdata[0]);
}
