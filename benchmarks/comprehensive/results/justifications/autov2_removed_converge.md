---
triples:
  - benchmark: compression
    format: scx_auto_v2
    dataset: census_1m
  - benchmark: compression
    format: scx_auto_v2
    dataset: census_500k
  - benchmark: compression
    format: scx_auto_v2
    dataset: pbmc10k
  - benchmark: compression
    format: scx_auto_v2
    dataset: pbmc3k
  - benchmark: compression
    format: scx_auto_v2
    dataset: smartseq2
  - benchmark: compression
    format: scx_auto_v2
    dataset: tabula_sapiens_100k
  - benchmark: memory
    format: scx_auto_v2
    dataset: pbmc10k
  - benchmark: memory
    format: scx_auto_v2
    dataset: pbmc3k
  - benchmark: ml_loader
    format: scx_auto_v2
    dataset: pbmc10k
  - benchmark: ml_loader
    format: scx_auto_v2
    dataset: pbmc3k
  - benchmark: read_full
    format: scx_auto_v2
    dataset: census_1m
  - benchmark: read_full
    format: scx_auto_v2
    dataset: census_500k
  - benchmark: read_full
    format: scx_auto_v2
    dataset: pbmc10k
  - benchmark: read_full
    format: scx_auto_v2
    dataset: pbmc3k
  - benchmark: read_full
    format: scx_auto_v2
    dataset: smartseq2
  - benchmark: read_full
    format: scx_auto_v2
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: scx_auto_v2
    dataset: pbmc10k
  - benchmark: read_selective
    format: scx_auto_v2
    dataset: pbmc3k
  - benchmark: read_selective
    format: scx_auto_v2
    dataset: smartseq2
  - benchmark: read_selective
    format: scx_auto_v2
    dataset: tabula_sapiens_100k
  - benchmark: read_streaming_vs_inmemory
    format: scx_auto_v2
    dataset: pbmc10k
  - benchmark: read_streaming_vs_inmemory
    format: scx_auto_v2
    dataset: pbmc3k
  - benchmark: write
    format: scx_auto_v2
    dataset: pbmc3k
reason: >
  The `scx_auto_v2` codec profile was removed in the auto→adaptive codec
  convergence (commit 472df42): `codec="auto"` is now itself the cost-aware
  adaptive profile that `auto_v2` used to provide, and `decode_target` was
  retired. These rows are present in the pre-flip LATEST baseline but are
  intentionally absent from the candidate (the format no longer exists), so
  the gate flags them as "missing". They are removed coverage, not a
  regression, and disappear once this baseline is promoted.
expires: 2027-01-12
---

`scx_auto_v2` (workload-aware ShufDeltaZstd profile, opt-in) folded into the
default `codec="auto"`. See docs/codec.md § "The codec intent axis
(auto / fast / compact)". The equivalent size-optimizing behavior is now the
default `auto`; `fast` provides the old decode-max heuristic. Once this
candidate is promoted to LATEST, the auto_v2 rows are gone from the baseline
and this justification becomes inert.
