// UMAP SGD optimization kernel for GPU UMAP.
//
// Parallelizes edge processing: each thread handles one edge per kernel launch.
// Uses atomicAdd for concurrent embedding updates (intentional — UMAP SGD is
// stochastic and tolerant of noisy gradient updates, matching cuML's approach).
//
// Per-thread negative sampling uses a Philox-style hash function for
// reproducible-per-thread (but globally stochastic due to atomicAdd races)
// random number generation.

// Simple hash function for per-thread RNG (Philox-style mixing).
// Produces decent uniformity for negative sample selection.
__device__ unsigned int gpu_hash(unsigned int x) {
    x ^= x >> 16;
    x *= 0x45d9f3bU;
    x ^= x >> 16;
    x *= 0x45d9f3bU;
    x ^= x >> 16;
    return x;
}

extern "C" __global__ void umap_sgd_kernel(
    float* __restrict__ embedding,          // [n_obs × n_components], row-major
    const int* __restrict__ head,           // edge head indices [n_active_edges]
    const int* __restrict__ tail,           // edge tail indices [n_active_edges]
    int n_active_edges,
    int n_obs,
    int n_components,
    float a, float b,                       // UMAP curve parameters
    float alpha,                            // learning rate for this epoch
    int negative_sample_rate,
    unsigned long long seed,
    int epoch                               // current epoch (used for RNG seeding)
) {
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_idx >= n_active_edges) return;

    int i = head[edge_idx];
    int j = tail[edge_idx];

    // --- Attractive force ---
    float dist_sq = 0.0f;
    for (int d = 0; d < n_components; d++) {
        float diff = embedding[i * n_components + d] - embedding[j * n_components + d];
        dist_sq += diff * diff;
    }
    // Floor to avoid division by zero
    dist_sq = fmaxf(dist_sq, 1e-10f);

    // Gradient coefficient for attractive force:
    //   -2ab * dist^(2(b-1)) / (1 + a * dist^(2b))
    float pow_b = powf(dist_sq, b);              // dist_sq^b = dist^(2b)
    float pow_bm1 = powf(dist_sq, b - 1.0f);    // dist_sq^(b-1) = dist^(2(b-1))
    float grad_coeff = -2.0f * a * b * pow_bm1 / (1.0f + a * pow_b);

    for (int d = 0; d < n_components; d++) {
        float diff = embedding[i * n_components + d] - embedding[j * n_components + d];
        float grad = grad_coeff * diff;
        // Clip gradient to prevent divergence
        grad = fmaxf(fminf(grad, 4.0f), -4.0f);
        atomicAdd(&embedding[i * n_components + d], alpha * grad);
        atomicAdd(&embedding[j * n_components + d], -alpha * grad);
    }

    // --- Repulsive forces via negative sampling ---
    // Per-thread RNG state: combine edge_idx, epoch, and seed for uniqueness
    unsigned int rng_state = gpu_hash((unsigned int)(edge_idx ^ (epoch * 1000003) ^ (unsigned int)(seed & 0xFFFFFFFF)));

    for (int neg = 0; neg < negative_sample_rate; neg++) {
        // Generate random negative sample index
        rng_state = gpu_hash(rng_state + (unsigned int)(neg + 1));
        int k = (int)(rng_state % (unsigned int)n_obs);

        // Skip self
        if (k == i) continue;

        float neg_dist_sq = 0.0f;
        for (int d = 0; d < n_components; d++) {
            float diff = embedding[i * n_components + d] - embedding[k * n_components + d];
            neg_dist_sq += diff * diff;
        }
        neg_dist_sq = fmaxf(neg_dist_sq, 1e-10f);

        // Gradient coefficient for repulsive force:
        //   2b / ((0.001 + dist²)(1 + a * dist²^b))
        float neg_pow_b = powf(neg_dist_sq, b);
        float neg_grad_coeff = 2.0f * b / ((0.001f + neg_dist_sq) * (1.0f + a * neg_pow_b));

        for (int d = 0; d < n_components; d++) {
            float diff = embedding[i * n_components + d] - embedding[k * n_components + d];
            float grad = neg_grad_coeff * diff;
            grad = fmaxf(fminf(grad, 4.0f), -4.0f);
            // Only update the head node for repulsive forces (matching umap-learn)
            atomicAdd(&embedding[i * n_components + d], alpha * grad);
        }
    }
}
