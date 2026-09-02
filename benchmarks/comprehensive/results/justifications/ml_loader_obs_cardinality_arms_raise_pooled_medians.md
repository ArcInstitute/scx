---
# Scoped, not whole-triple. `ml_loader / scx_auto / census_1m` carries no
# absolute floors today, but `ml_loader` carries 39 of them across other
# triples and this file must not become the reason a future one is unreachable.
# See results/justifications/README.md § Metric scope.
metrics: [peak_rss_mb_median, median_wall_s]
triples:
  - benchmark: ml_loader
    format: scx_auto
    dataset: census_1m
  - benchmark: ml_loader
    format: scx_fast
    dataset: census_1m
reason: "Two new obs-cardinality arms (raw_obs_lowcard / raw_obs_highcard) add runs to a triple whose summary medians are pooled across every run; the high-cardinality arm is intentionally ~109 ms/batch slower. Not a regression."
---

# `ml_loader` pooled medians rose when the obs-cardinality arms landed

## What changed

`ml_loader` gained two SCX-only scenarios on census_1m, and only on census_1m:

| scenario | `obs_columns=` | categories |
|---|---|---|
| `raw_obs_lowcard` | `["sex"]` | 3 |
| `raw_obs_highcard` | `["observation_joinid"]` | 957,955 |

They exist because **`obs_columns=` was passed nowhere in the suite**. Every
other scenario leaves it unset, so `batch["obs"] == {}` and
`batches_per_sec__raw` is a pure-X number. That made OPT-LOADER-6 invisible:
`obs_to_pydict` (`scx-loader/src/python/convert.rs`) rebuilds a categorical
column's entire category list as fresh `PyUnicode` objects **per batch**, and
nothing in the suite ever asked it to.

census_1m is the only dataset in the arm's spec because it is the only fixture
with a high-cardinality categorical. The review doc prescribed
`obs_columns=["soma_joinid"]`; that column is `int64`, so it takes
`obs_to_pydict`'s `PyArray1::from_vec` branch and builds no dictionary at all —
a million distinct values down the cheap path. `observation_joinid` is the
categorical one.

## Why the medians move, and why that is not a regression

`capture_baseline.archive_raw_results` writes one `median_wall_s` and one
`peak_rss_mb_median` per `(benchmark, format, dataset)` triple, taken across
**every** run in the file — it does not partition by `extra["scenario"]`.
`ml_loader`'s existing four scenarios (`raw`, `hvg`, `norm`, `hvg_norm`) all
run at similar rates; the new pair adds runs that are deliberately slower, and
the high-cardinality one deliberately much slower.

Measured 2026-09-02, page-cache-cold, 40 batches at `batch_size=1024`:

| `obs_columns=` | batches/s | ms/batch |
|---|---|---|
| `None` (the `raw` scenario) | 1.15 | 872 |
| `["sex"]` (3 categories) | 1.14 | 879 |
| `["observation_joinid"]` (958k) | 1.02 | 981 |

**+109 ms/batch** for the high-cardinality column — squarely inside the
60–120 ms/batch the review estimated from the code. Note it is only **1.13×**
as a ratio, because a cold epoch is I/O-bound at ~1.15 batches/s; this
benchmark normally reports warm, page-cache-resident rates (~48 batches/s),
where the same 109 ms is several-fold. That is exactly why the arm records
`obs_highcard_overhead_ms_per_batch` beside the ratio: the per-batch delta is
what the fix changes, and the ratio is a property of the host.

The per-scenario keys are where the signal is —
`batches_per_sec__raw_obs_highcard` and friends are sparse, so a threshold on
one medians that arm alone. The pooled row is the only thing this file
suppresses.

## Scope

Both `scx_auto` and `scx_fast` are listed: the arm is gated on
`loader_type == "scx"`, not on a single codec key, so whichever SCX variants a
capture schedules on census_1m will carry the new runs.

Deliberately **not** listed: `ml_loader / scx_auto / tabula_sapiens_100k`,
which carries two live `batches_per_sec__pyscx_training_dataset_workers2*`
floors. The arm does not run there (no high-cardinality categorical), so it has
nothing to justify and must not shadow those floors.

## When to remove this file

Once a promoted baseline was captured **with** both arms present, the
composition matches and the pooled medians line up again. Delete it in the same
change that promotes that baseline. If the flag persists against such a
baseline, it is a real regression.
