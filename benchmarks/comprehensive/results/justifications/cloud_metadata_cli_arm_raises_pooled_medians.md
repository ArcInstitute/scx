---
# Scoped, not whole-triple: this file explains the two pooled summary metrics.
# `cloud_metadata` carries no absolute floors today; scoping anyway so that
# adding one later does not silently land under a suppression, which is how 16
# floors were disarmed before PR-02c.
metrics: [peak_rss_mb_median, median_wall_s]
# Expires deliberately. Every "when to remove this file" recipe below waits on
# the PR-03 recapture; without a date, a missed cleanup leaves the suppression
# in place forever and silently. 2026-12-31 gives the recapture margin and then
# fails loud.
expires: 2026-12-31
triples:
  - benchmark: cloud_metadata
    format: scx_auto
    dataset: pbmc3k
  - benchmark: cloud_metadata
    format: scx_auto
    dataset: pbmc10k
  - benchmark: cloud_metadata
    format: scx_auto
    dataset: smartseq2
  - benchmark: cloud_metadata
    format: scx_auto
    dataset: tabula_sapiens_100k
  - benchmark: cloud_metadata
    format: scx_auto
    dataset: census_500k
  - benchmark: cloud_metadata
    format: scx_auto
    dataset: census_1m
reason: "A new `scx_info_cloud` subprocess arm adds runs whose wall is orders of magnitude above the in-process catalog open, and the summary medians are pooled across every run of the triple. Not a regression."
---

# `cloud_metadata` pooled medians rose when the CLI arm landed

## What changed

`cloud_metadata` gained a second, SCX-only arm: `scx info --json <gs://…>` run
as a subprocess (`scenario="scx_info_cloud"`).

It is not a duplicate of the existing in-process arm, which is
`pyscx.open_cloud(url)` plus touching `n_obs` / `n_vars` / `nnz` — a single
GET and a catalog parse, and sub-second. `scx info` fills two columns the
library open does not: the per-shard codec breakdown and the value-encoding
summary. It gets them by range-reading **every** CSR shard's 76-byte header at
`METADATA_SHARD_FETCH_CONCURRENCY = 8`. On an atlas-scale fixture that is
thousands of GETs against one, and it was measured at ~175 s.

That constant is what OPT-CLOUD-1 raises (~175 s → ~22 s expected), and
**nothing in the suite measured it** — `cloud_metadata` measured
`read_cloud_metadata`, so a 8× improvement in the CLI path had nowhere to land.

## Why the medians move, and why that is not a regression

`capture_baseline.archive_raw_results` writes one `median_wall_s` and one
`peak_rss_mb_median` per triple, pooled across **every** run in the file. The
pre-existing arm is a sub-second open; the CLI arm is seconds to minutes on the
same fixture. Pooling the two moves `median_wall_s` by orders of magnitude on
the larger datasets — by construction, and in a way that says nothing about
whether either path regressed.

The per-arm signal is `wall_s__scx_info_cloud`, which is sparse: only the CLI
runs carry it, so a threshold on it medians that arm alone. The in-process arm
keeps reporting its own `wall_s` unchanged.

`peak_rss_mb_median` also shifts, and more confusingly: the CLI arm's runs
record the *parent* process's peak (`PeakRssSampler` around the subprocess)
while the work happens in a child, so their RSS rows are the harness's
footprint rather than a measurement of the operation. Another reason to read
the sparse `peak_rss_mb__scx_info_cloud`, not the pooled one.

> **Correction.** An earlier version of this file claimed exactly that while
> the code recorded **nothing** — `_run_cli_info_arm` omitted `peak_rss_mb=`,
> so `add_run` supplied its `0.0` default. Three zeros beside three real
> catalog-open samples roughly halve the triple's pooled `peak_rss_mb_median`,
> which would have become a phantom regression the moment this file is
> retired, and would have been baked into any baseline recaptured first. All
> three reviewers on PR #498 caught it. The arm now samples a real parent peak,
> and `test_cli_info_arm_records_a_real_parent_rss_not_add_runs_zero_default`
> fails if it stops.

## Scope

Every dataset `cloud_metadata` is scheduled on, because the arm is not
dataset-gated — it runs wherever a cloud fixture resolves. `LATEST` carries 60
`cloud_metadata` rows, so all of them will compare.

## When to remove this file

Once a promoted baseline was captured with the CLI arm present. Note that a
capture on a host with no `--features cloud` build records a skip reason and
emits nothing for the arm, so a baseline captured that way does **not** fix the
composition — check `metadata.cli_info_bin` is set in the archived raw JSON
before deleting this.
