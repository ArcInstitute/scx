// GPU CSR → dense conversion + HVG projection kernel — stub for Phase G.
//
// Each warp (32 threads) cooperates on one row: threads split the row's nnz
// values for parallel scatter into the dense output. Falls back to
// one-thread-per-row for rows with nnz < 32.
// See Phase3-Step2.md Phase G for full specification.

// Placeholder — will be implemented in Phase G.
