---
triples:
  - benchmark: roundtrip
    format: scx_auto
    dataset: pbmc10k
    metric: median_wall_s
reason: >
  Intended cost of the auto→adaptive flip (commit 472df42). The default
  `codec="auto"` now dual-encodes each framed integer shard (heuristic vs
  ShufDeltaZstd) and picks by size, vs the old single-encode Scx1 heuristic.
  The `roundtrip` benchmark re-encodes on write, so the dual-encode adds
  ~4.9% wall (1.42s → 1.49s on pbmc10k). This is the write-time cost that
  buys the 1.3–2× smaller files (amortized over every read); `codec="fast"`
  keeps the single-encode path for write-latency-critical use.
expires: 2027-01-12
---

Accepted tradeoff, not a regression. See docs/codec.md § "The codec intent
axis" and AUTO-V2-CONVERGE.md G2 (accept the 2× dual-encode; the
statistics-gated estimator is the deferred optimization if atlas-scale
encode time becomes a measured problem).
