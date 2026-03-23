// GPU FOR-BP decoder kernel — stub for Phase C implementation.
//
// Architecture: one CUDA block per FOR-BP block (128 rows), threads decode
// rows in parallel. CPU pre-parses LEB128 block headers to avoid serial
// bottleneck on device. See Phase3-Step2.md Phase C for full specification.

// Placeholder — will be implemented in Phase C.
