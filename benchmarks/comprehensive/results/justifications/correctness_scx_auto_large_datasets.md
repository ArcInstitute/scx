---
triples:
  - benchmark: correctness
    format: scx_auto
    dataset: smartseq2
    metric: overall_passed_int
  - benchmark: correctness
    format: scx_auto
    dataset: tabula_sapiens_100k
    metric: overall_passed_int
  - benchmark: correctness
    format: scx_auto
    dataset: census_500k
    metric: overall_passed_int
  - benchmark: correctness
    format: scx_auto
    dataset: census_1m
    metric: overall_passed_int
reason: >
  The correctness validation suite materialises 4 reference fixtures
  simultaneously (scanpy equivalence, backed-mode equivalence,
  preprocessing path cross-validation, SLAF round-trip). On datasets
  with high n_obs × n_vars (smartseq2 50K × 61K, tabula_sapiens_100k
  100K × 61K, census_500k 500K × 61K, census_1m 1M × 61K), the stacked
  dense-X working sets exceed even the 1 TB MEM_CEILING_GB high-mem
  node allocation. Empirically (2026-05-10 gate runs #4–#6):
    smartseq2:          OOM at 88 GB allocated  (~7–8× dense)
    tabula_sapiens_100k: OOM at 176 GB
    census_500k:        OOM at 864 GB
    census_1m:          OOM at 1 TB (capped)
  pbmc3k / pbmc10k pass at 24 GB. The suite is functionally a
  small-dataset smoke test — running it at census scale provides no
  additional signal beyond pbmc10k coverage. Suppressing these four
  triples until either the suite is refactored to release each
  reference between checks or `correctness` is split into per-suite
  benchmarks that can be run independently.
expires: 2026-08-01
---

Captured 2026-05-10. Bumping `correctness` memory estimator alone
(now `max(base_mb * 4, dense_mb * 5.0, 12 GB)`) is insufficient on
high-`n_vars` datasets because the suites materialise reference X
copies in parallel, and the `MEM_CEILING_GB = 1 TB` cap binds before
the sum-of-references peak is allocated.

Re-evaluate when the suite is restructured (each validator releases
its own AnnData / SCX handle before the next loads), or when
`correctness` is decomposed into 4 sibling benchmarks
(`correctness_scanpy`, `correctness_backed`, `correctness_preproc`,
`correctness_slaf`) so each can be sized independently.
