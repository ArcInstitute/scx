# Flakiness ledger

Under `compare_against_baseline.py --gate --flakiness <this-dir>`,
markdown files here declare per-row **relaxed tolerances** for the
regression gate. Each entry raises the bound for one
`(benchmark, format, dataset, metric)` row without weakening the rest
of the gate. A regression that crosses the relaxed bound still trips.

This is distinct from `../justifications/`:

- A **justification** fully suppresses a row from the regression tally
  (use for an investigated, deliberately-accepted regression with a
  fix-or-revisit deadline).
- A **flakiness override** keeps the gate active for the row but at a
  higher per-row tolerance (use for a row whose underlying performance
  is fine but whose run-to-run wall-time has measurable RSD on this
  hardware — preemption, shared cache, NUMA placement, oversubscribed
  threads).

If you find yourself reaching for `--timing-tolerance 0.05` to make a
PR pass, add a flakiness entry for the noisy row(s) instead — the
global flag weakens every row simultaneously.

## File format

```markdown
---
overrides:
  - benchmark: cloud_pull
    format: scx_auto
    dataset: tabula_sapiens_100k
    metric: median_wall_s   # optional; defaults to median_wall_s
    tolerance: 0.08         # required; relaxed bound for this row
reason: Shared SLURM node — 7% wall-time RSD across 10 runs (issue #1234).
expires: 2026-07-01
---

Optional prose explaining what would let the override expire — e.g.
``--exclusive`` SLURM allocation, NUMA pinning, switching the queue.
```

Field semantics:

- `metric` is optional and defaults to `median_wall_s` (the dominant
  noise source). Set it explicitly to `peak_rss_mb_median` or
  `file_size_bytes` to relax those rows.
- `tolerance` is the relative bound applied to that row, replacing the
  global `--timing-tolerance` / `--rss-tolerance` / `--size-tolerance`
  for the matching `(benchmark, format, dataset, metric)` quad only.
  Must be non-negative — overrides relax, they don't tighten.
- `expires` follows the same inclusive-date semantics as
  justifications. Active on the listed date, inactive the day after.
  Stale entries should be renewed (new expiry) or retired (delete).
- When two files declare overrides for the same quad, the **stricter
  (lower)** tolerance wins and the conflict is logged. Collapse
  overlapping entries during cleanup.

The gate report annotates every `median_wall_s` row with its observed
coefficient of variation (`stdev/mean`) computed from the candidate's
`runs[].wall_s`. A "High-variance rows" section highlights rows whose
CV exceeds 5% — those are the natural candidates for a ledger entry.

## Workflow

1. Gate fails in a PR with a regression on a row whose CV is high
   (visible in the report's High-variance section).
2. Inspect the underlying noise source. If the perf is genuinely fine
   and the noise comes from the runtime environment, commit a new
   markdown here with a tolerance reflecting observed RSD plus a
   small safety margin (e.g. CV ≈ 7% → tolerance ≈ 0.10).
3. Re-run the gate. The row now appears with status `relaxed` and the
   gate passes if the observed delta is within the new bound.
4. Periodically renew or retire entries — the goal is to fix the
   underlying noise, not to keep relaxing tolerances forever.

Documented end-to-end in `benchmarks/README.md` under "Regression
Gating".
