---
# Scoped to the two metrics these floors name. The `cellstream` triples have no
# other floors today, so the scope changes nothing now — it is here so that
# adding one later does not silently inherit this suppression, which is the
# defect `test_no_floor_is_fully_suppressed_by_an_active_justification` exists
# to catch.
metrics: [batches_per_sec__raw, batches_per_sec__hvg_norm]
triples:
  - benchmark: ml_loader
    format: cellstream
    dataset: pbmc3k
  - benchmark: ml_loader
    format: cellstream
    dataset: pbmc10k
  - benchmark: ml_loader
    format: cellstream
    dataset: smartseq2
  - benchmark: ml_loader
    format: cellstream
    dataset: tabula_sapiens_100k
  - benchmark: ml_loader
    format: cellstream
    dataset: census_500k
  - benchmark: ml_loader
    format: cellstream
    dataset: census_1m
reason: >
  The cellstream runner cannot run: the upstream package restructured and its
  API moved. Not a throughput regression — no measurement was taken.
expires: 2027-03-01
---

`cellstream` is a competitor loader, installed as an editable checkout at
`~/dev/python/cellstream`. Upstream restructured it: `cellstream.format` and
`cellstream.writer` no longer exist and `CellStream` is now `CellStore`, so
`runners/cellstream_runner.py`'s member imports fail while the package itself
imports fine. Every `cellstream` cell in the 2026-09-03 tier-full capture
failed this way — 33 of them across all benchmarks, 12 of which are these
floored `ml_loader` triples.

**The floors are not wrong and the values are not stale.** They are floors on a
format whose runner is broken, so the metric is `missing` rather than low, and
`check_absolute_floors` counts a present-but-metricless triple as a violation
(correctly — that is what stops a silently-skipped arm from reading as a pass).
Suppressing is the honest record: nothing was measured, so nothing can be
compared.

Porting the runner to the restructured API is a separate change. It is a
comparison format rather than an SCX surface, and guessing at new upstream
semantics is its own piece of work with its own correctness question. Until
then `LATEST` carries no `cellstream` rows where the previous baseline
(`v0.11.2-multimodal-loader-fix`) carried 47.

The error message now names the real cause. It used to say "cellstream is not
installed in this env. Install it into scx-bench with: … pip install -e
~/dev/python/cellstream" — false, and it sent me looking for an install problem
for ten minutes. `runners/base.probe_optional` now distinguishes absent from
moved and reports "cellstream IS installed (<path>) but its API has moved: No
module named 'cellstream.format'. The runner needs porting — installing the
package will not help."

**When the runner is ported, delete this file rather than renewing it** — the
floors should go back to being measured. The expiry is deliberately long
because the blocker is upstream and unscheduled; a short date would just churn.
