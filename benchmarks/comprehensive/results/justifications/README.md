# Regression justifications

Under `compare_against_baseline.py --gate --justifications <this-dir>`,
markdown files here suppress matching `(benchmark, format, dataset)`
regressions from the gate's failure tally. Use for known-bad
regressions that have been investigated and deliberately accepted
(upstream dependency change, intentional tuning tradeoff, etc).

## File format

One markdown file per justification. Front-matter is YAML, body is
free-form prose shown in the gate's report:

```markdown
---
triples:
  - benchmark: cloud_push
    format: scx_auto
    dataset: pbmc3k
  - benchmark: cloud_pull
    format: scx_auto
    dataset: pbmc3k
reason: Upstream GCS client added a mandatory HTTP/2 header (issue #1234).
expires: 2026-06-01
---

The 2025.9.0 → 2025.10.0 gcsfs bump added a synchronous header
canonicalization pass that adds ~4% per-request latency. We accept it
because rolling back gcsfs would block the zarr 3.1.6 upgrade.
```

`expires` is optional. When present and in the past relative to
`date.today()`, the justification is treated as inactive and its
triples are no longer suppressed — the gate will fail again until the
justification is updated or removed.

## Metric scope — read this before justifying a pooled median

By default a justification suppresses **every metric** on each triple it names:
relative regressions *and* `thresholds.yaml` absolute floors. That is the right
default when the triple is wholly known-bad (a suite that OOMs, a dispatch path
that collapses both a throughput floor and a timing row).

It is the wrong default for the commonest case in practice. `summary.json`
carries one `peak_rss_mb_median` and one `median_wall_s` per triple, pooled
across **every** run in the file — so adding an arm to an existing benchmark
shifts both by construction and needs a justification. Left unscoped, that
justification also switches off every absolute floor on the triple. Two
committed files were in exactly that state and disarmed 16 floors between them,
one of which was added in the same commit as its own justification.

So name what the file explains:

```markdown
---
metrics: [peak_rss_mb_median, median_wall_s]
triples:
  - benchmark: cellset_gather
    format: scx_auto
    dataset: census_1m
  - benchmark: index_plan
    format: scx_auto
    dataset: tabula_sapiens_100k
    metric: batches_per_sec__pyscx_index_plan_dataset_workers2
---
```

- A file-level `metrics:` list applies to every triple in the file.
- A per-entry `metric:` (singular, one name) narrows that triple alone and
  **adds to** the file-level list rather than replacing it.
- Omitting both keeps the whole-triple default.
- Where two active files name the same triple, the scopes union and an unscoped
  file wins — a merge can only ever suppress more than either file asked for.

**The rule:** *a justification for a pooled summary metric must name it, or it
silently disarms every floor on the triple.* Enforced by
`tests/test_floor_reachability.py::test_no_floor_is_fully_suppressed_by_an_active_justification`,
which fails on any floor sitting under an unscoped suppression; genuinely
whole-triple cases go in that test's `_DELIBERATE_WHOLE_TRIPLE_SUPPRESSIONS`
with the reason written down.

Each justification must list at least one triple. Malformed
front-matter fails loud (logged + skipped, gate proceeds without
suppression).

## Workflow

1. Gate fails in a PR; one or more triples are flagged.
2. Investigator reviews; if the regression is acceptable, commit a
   new file here with the triples + prose explanation.
3. Re-run the gate; the previously-failing triples now appear in the
   "suppressed" column and the gate passes.
4. Review expiry dates periodically — a stale suppression should
   either be renewed (new file, new expiry) or retired (delete the
   file, accept the gate will fail).
5. Before committing, check what *floors* the triples carry:

   ```bash
   .venv/bin/python -m pytest \
       benchmarks/comprehensive/tests/test_floor_reachability.py -k suppressed
   ```

Documented end-to-end in `benchmarks/README.md` under "Regression
Gating".
