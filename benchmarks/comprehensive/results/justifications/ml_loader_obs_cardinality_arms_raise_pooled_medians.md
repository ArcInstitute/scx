---
# Scoped, not whole-triple. `ml_loader / scx_auto / census_1m` carries no
# absolute floors today, but `ml_loader` carries 39 of them across other
# triples and this file must not become the reason a future one is unreachable.
# See results/justifications/README.md § Metric scope.
metrics: [peak_rss_mb_median, median_wall_s]
# Expires deliberately. Every "when to remove this file" recipe below waits on
# the PR-03 recapture; without a date, a missed cleanup leaves the suppression
# in place forever and silently. 2026-12-31 gives the recapture margin and then
# fails loud.
expires: 2026-12-31
triples:
  - benchmark: ml_loader
    format: scx_auto
    dataset: census_1m
reason: "Two new obs-cardinality arms (raw_obs_lowcard / raw_obs_highcard) add runs to a triple whose summary medians are pooled across every run; the high-cardinality arm is intentionally slower (measured 7.86 ms/batch). Not a regression."
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

Measured through the arm at census_1m, 2 runs of a full 977-batch epoch
(SLURM job 2891005, warm):

| scenario | `obs_columns=` | categories | batches/s |
|---|---|---|---|
| `raw_obs_lowcard` | `["sex"]` | 3 | 4.2 |
| `raw_obs_highcard` | `["observation_joinid"]` | 957,955 | 4.1 |

**7.86 ms/batch**, a 1.033× slowdown.

> **Correction.** An earlier version of this file reported **+109 ms/batch**,
> from a standalone 40-batch cold probe — a number that agreed with the review
> doc's 60–120 ms/batch estimate and is ~14× too high. 40 batches cannot
> separate an 8 s construction cost from a per-batch one. The full-epoch figure
> is the one to use.
>
> The mechanism is real and confirmed live, so the small number is a *finding*:
> a batch's `obs["observation_joinid"]` arrives as `{codes, categories}` with
> 957,955 categories, and the list is a fresh object every batch (checked by
> identity). ~8 ns per category on a 238 ms batch is ~3% of an epoch — so
> OPT-LOADER-6 is worth much less than estimated, and this arm is what says so.

The arm records `obs_highcard_overhead_ms_per_batch` beside the ratio because
the per-batch delta is what a fix changes and the ratio is a property of how
I/O-bound the host is.

The per-scenario keys are where the signal is —
`batches_per_sec__raw_obs_highcard` and friends are sparse, so a threshold on
one medians that arm alone. The pooled row is the only thing this file
suppresses.

## Scope

One triple, because the arm runs on **`scx_auto` only**. It was first gated on
`loader_type == "scx"`, which covers all seven `_SCX_KEYS` variants — and
`results/baselines/LATEST` carries census_1m rows for all seven, so five of them
would have acquired two deliberately slower scenarios with nothing here
explaining the shift. A reviewer caught the mismatch between the code, this
front matter and the prose.

Pinning a single trigger is also the right answer on its own terms: the cost
this arm prices is Python-side (`obs_to_pydict` rebuilding a category list) and
codec-independent, so seven variants would pay seven times for one number. It
is the same reasoning `build_csc` and `fragment_ops` give for their own
`scx_auto` pins.

Deliberately **not** listed: `ml_loader / scx_auto / tabula_sapiens_100k`,
which carries two live `batches_per_sec__pyscx_training_dataset_workers2*`
floors. The arm does not run there (no high-cardinality categorical), so it has
nothing to justify and must not shadow those floors.

## When to remove this file

Once a promoted baseline was captured **with** both arms present, the
composition matches and the pooled medians line up again. Delete it in the same
change that promotes that baseline. If the flag persists against such a
baseline, it is a real regression.
