// GPU type-cast kernels for eliminating host round-trips.
//
// Simple element-wise casts run at full GPU memory bandwidth,
// avoiding the GPU→CPU→GPU transfers needed for type conversion.

/// Cast u32 → i32 (element-wise, one thread per element).
/// Safe for all values < 2^31 (always true for column indices).
extern "C" __global__ void cast_u32_to_i32(
    const unsigned int* __restrict__ input,
    int*               __restrict__ output,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = (int)input[i];
}

/// Cast u32 → f32 (element-wise, one thread per element).
/// Exact for all values ≤ 2^24 (always true for UMI counts).
extern "C" __global__ void cast_u32_to_f32(
    const unsigned int* __restrict__ input,
    float*             __restrict__ output,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = (float)input[i];
}
