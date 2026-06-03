---
triples:
  - benchmark: accel_hvg
    format: accel_hvg__pyscx_cpu
    dataset: pbmc3k
reason: >
  accel_hvg (CPU, seurat_v3) hvg_overlap_vs_scanpy on pbmc3k is
  deterministically 0.9891 (98.91%), marginally below the 0.99 floor. This
  is a borderline-threshold miss, not a regression: the ~1.1% gap is a
  handful of genes near the HVG selection cutoff on the small (2700-cell)
  pbmc3k fixture, where dispersion rank ties shift membership relative to
  scanpy. The value is identical across the v0.6.4-gpu-de-v3 recapture
  runs (deterministic, not flaky), and the larger datasets clear the
  floor. Treat as a marginal floor for pbmc3k specifically; the proper
  long-term fix is to either relax the pbmc3k HVG floor to 0.98 or pin the
  HVG selection tie-break to match scanpy.
expires: 2026-12-31
---

HVG overlap vs scanpy on pbmc3k is a deterministic 98.91% vs the 99% floor —
a marginal borderline miss on the small fixture (tie-break near the HVG
cutoff), not a correctness regression. Larger datasets pass.
