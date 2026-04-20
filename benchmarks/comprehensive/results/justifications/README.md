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

Documented end-to-end in `benchmarks/README.md` under "Regression
Gating".
