// GPU Rice decoder kernel — stub for Phase C implementation.
//
// Primary design: one CUDA thread per block of 256 values (sequential decode,
// parallel across blocks). See Phase3-Step2.md Phase C for full specification.
//
// Stretch-goal: warp-cooperative decode using __ballot_sync() and __shfl_sync().

// Placeholder — will be implemented in Phase C.
