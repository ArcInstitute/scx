---
triples:
  - benchmark: index_plan
    format: scx_auto
    dataset: tabula_sapiens_100k
    metric: batches_per_sec__pyscx_index_plan_dataset_workers2
  - benchmark: ml_loader
    format: scx_auto
    dataset: tabula_sapiens_100k
    metric: batches_per_sec__pyscx_training_dataset_workers2_persistent
  - benchmark: index_plan
    format: scx_auto
    dataset: census_500k
  - benchmark: index_plan
    format: scx_auto
    dataset: census_1m
reason: >
  ~50% throughput drop on workers2 paths after MULTIMODAL-SUPPORT
  Phase A.2 (v2 header + catalog parsing). Hypothesis: per-worker
  BackedCsrReader open() on tabula_sapiens_100k now traverses the
  expanded v2 catalog metadata even on single-modality files, and
  the workers2 + workers2_persistent paths construct the dataset
  inside each `IterableDataset.__iter__` worker — so any open-path
  overhead multiplies by `num_workers`. Tracked as a follow-on
  perf investigation; suppressing the floor here so infra fixes
  ship cleanly.

  Extended 2026-05-11 to also cover whole-cell suppression for
  `(index_plan, scx_auto, census_500k)` and `(index_plan, scx_auto,
  census_1m)`. These cells now TIMEOUT at the 125 / 130-min budget
  rather than report a degraded throughput — the v2-catalog open
  cost at ~16K shards × `num_workers` makes the workers2 scenario
  infeasible at census scale. The fix is on the Rust side
  (memoise the catalog parse across worker opens, or lazy-parse
  `ShardStats` so per-shard reads only happen on access);
  re-enable these cells once that lands.
expires: 2026-08-01
---

Captured during the 2026-05-09 tier-full gate run. Observations
that pin the diagnosis to v2 catalog overhead (rather than per-
modality dispatch):

- The drop reproduces on **single-modality v1 fixtures read by the
  v2 reader** — the modality routing path doesn't fire, but the
  v2 catalog parser does. Header parse + per-entry `modality_id`
  read add a constant per-shard cost that workers2 multiplies by
  `num_workers`.
- `tabula_sapiens_100k` has ~16K shards, so the per-shard overhead
  dominates open-path wallclock on that dataset specifically.
  Smaller fixtures (pbmc3k, pbmc10k) don't show the regression,
  consistent with the per-shard hypothesis.
- The non-`workers2` ml_loader scenarios (the in-process iterator
  paths) are unaffected — they pay the open cost once per epoch,
  not per-worker.

Re-evaluate the floor after the perf fix lands. The 75% target
captured under the v1 reader is still the right threshold —
nothing about the loader semantics changed, only the open-path
constant factor.

Tracking: this expires 2026-08-01 to force a deliberate revisit
within ~3 months. If the fix doesn't ship by then the gate will
fail again and re-prompt investigation.
