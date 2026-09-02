---
triples:
  - benchmark: fragment_ops
    format: scx_auto
    dataset: pbmc3k
  - benchmark: fragment_ops
    format: scx_auto
    dataset: tabula_sapiens_100k
reason: "Two intended changes both raise the pooled medians, which are taken across every run of the triple: _time_op now reports a true in-region PeakRssSampler peak instead of a post-op instantaneous reading, and a fifth arm (compact_full) rewrites a fixture ~2.7x the plain one. Neither is a regression."
---

# `fragment_ops` pooled medians rose when the `compact_full` arm landed

## What changed

`fragment_ops` gained a fifth operation, `compact_full`: `pyscx.compact` on
`<dataset>_full.scx` — a fixture carrying `obsm["X_pca"]`, `obsm["X_umap"]`, a
`counts` layer and a `.raw`, built by
`benchmarks/scripts/prep_full_fixtures.py`.

It exists because **no ordinary fixture in the suite has any of those**. No
source h5ad carries an `obsm` key or a layer, so every `.scx` the suite converts
reports `obsm_keys == []` and `layer_names == []`, and a rewrite of one never
reaches the code that copies them. The `compact` peak was blind to them in both
directions.

The `.raw` is along for the ride and does **not** contribute: `scx-ops`' carry
table drops raw under `compact` (with a warning) rather than carrying it. Raw's
whole-matrix copy is measured on the export side instead, by
`export_streaming`'s `streaming_full` arm.

`fragment_ops`' `_time_op` also changed in the same commit: it now brackets each
op with `PeakRssSampler` instead of taking a `current_rss_mb()` reading after
the op returned. Every op's number rises as a result — `append` reads the whole
input CSR before re-encoding, `compact` rewrites every section, and both free
that transient before the old sample was taken.

## Why the medians move, and why that is not a regression

`capture_baseline.archive_raw_results` writes one `median_wall_s` and one
`peak_rss_mb_median` per `(benchmark, format, dataset)` triple, taken across
**every** run in the file — `_median_rss` does not partition by
`extra["operation"]`. `fragment_ops` runs `append`, `delete`, `compact` and
`rollback`; the last of those is a single `pwrite` and pulls the pooled medians
down hard. Adding a fifth arm that rewrites a file ~2.7x the size of the plain
one (642 MB vs 233 MB at tabula_sapiens_100k, and ~585M nnz against 195M once
raw and the layer are counted) shifts both pooled numbers upward by
construction.

Measured at tabula_sapiens_100k, three runs per op:

| operation | peak RSS (MB), median |
|---|---|
| `append` | 750 |
| `delete` | 551 |
| `compact` | 1538 |
| `compact_full` | 3201 |
| `rollback` | 687 |

Pooled median: **750 MB**, against the 359.9 MB in
`results/baselines/LATEST`. Both causes are in that number — the sampler change
lifts every row, and `compact_full` adds a row well above the old maximum.

The per-operation numbers are unaffected in meaning and are where the signal is:
`metadata["per_op_medians"]` keys them by operation, and the new arm's memory is
gated on its own key, `peak_rss_mb__compact_full`, which is sparse — only
`compact_full` runs carry it, so a threshold on it medians that arm alone.

This is the same shape as `cellset_gather_s512_raises_pooled_rss.md`, for the
same reason.

## Scope

`pbmc3k` and `tabula_sapiens_100k` are listed because those are the datasets
whose `_full.scx` has been built. pbmc3k is there so the small tier — and any
local smoke run — exercises the arm in seconds rather than only at 100k;
tabula_sapiens_100k is where `streaming_full_peak_rss_mb` is actually floored.

`census_1m` is deferred to the PR-03 capture prep (11 GB source, high-mem
allocation — see `thresholds.yaml`'s Deferred floors, item 13); **add its triple
here in the same change that builds the fixture**, or the census row will read
as a regression the first time the arm runs there.

Note that `fragment_ops` carries no absolute floors, so suppressing the triple
costs only the relative diff. If a floor is ever added to this benchmark, revisit
— justification suppression is whole-triple and covers every metric on it.

## When to remove this file

Once a promoted baseline has been captured **with** the `compact_full` arm
present, the comparison point includes it and the pooled medians line up again.
Delete this file in the same change that promotes that baseline.
