---
# Scoped, not whole-triple. Unscoped, this file suppressed the
# `peak_rss_mb__compact_full` ceiling that landed in the SAME commit — see
# "What this file must not suppress".
metrics: [peak_rss_mb_median, median_wall_s]
triples:
  - benchmark: fragment_ops
    format: scx_auto
    dataset: pbmc3k
  - benchmark: fragment_ops
    format: scx_auto
    dataset: pbmc10k
  - benchmark: fragment_ops
    format: scx_auto
    dataset: smartseq2
  - benchmark: fragment_ops
    format: scx_auto
    dataset: tabula_sapiens_100k
  - benchmark: fragment_ops
    format: scx_auto
    dataset: census_500k
  - benchmark: fragment_ops
    format: scx_auto
    dataset: census_1m
reason: "Three intended changes all raise the pooled medians, which are taken across every run of the triple: _time_op now reports a true in-region PeakRssSampler peak instead of a post-op instantaneous reading, a compact_full arm rewrites a fixture ~2.7x the plain one, and an obs_import arm adds a key-joined in-place column add. None is a regression."
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

## The `obs_import` arm (added in PR-02c, same file)

A sixth arm, `obs_import`, times `pyscx.obs_import` — a key-joined in-place add
of one obs column from a CSV. It runs on **every** `fragment_ops` dataset, not
just the ones with a `_full.scx`, which is why this file's triple list now
covers all six datasets `LATEST` carries `fragment_ops` rows for.

It also moves the pooled `median_wall_s` in the *other* direction on small
fixtures: `obs_import` is cheap (24 ms at pbmc3k), so it pulls the pooled
median down while `compact_full` pushes it up. Neither movement says anything
about either op.

The same change gave every arm a sparse `wall_s__<operation>` key, which is the
gateable half — `wall_s` is a reserved `add_run` parameter and never reaches
`runs[].extra`, so before it the only visible timing was the pooled median this
file suppresses. `wall_s__rollback` and `wall_s__obs_import` are the two
cleanest OPT-FORMAT-1 instruments in the suite (`thresholds.yaml` Deferred
floors, item 18); both are sparse, so a threshold on either medians that arm
alone and is unaffected by this suppression.

## Scope

All six datasets `results/baselines/LATEST` carries `fragment_ops` rows for.
The `compact_full` arm reaches only those with a built `_full.scx` — pbmc3k and
tabula_sapiens_100k today, census_1m deferred to the PR-03 capture prep
(`thresholds.yaml` Deferred item 13) — but the `obs_import` arm and the sampler
change reach all of them, so all of them will compare.

## What this file must not suppress

The paragraph that stood here was wrong on its own commit. It read: "`fragment_ops`
carries no absolute floors, so suppressing the triple costs only the relative
diff. If a floor is ever added to this benchmark, revisit." A floor **was** added
to this benchmark, in the same change that added this file —

```yaml
  - benchmark: fragment_ops
    format: scx_auto
    dataset: tabula_sapiens_100k
    metric: peak_rss_mb__compact_full
    max: 4096.0
```

— and it sits on a triple this file names. Suppression was whole-triple over
every metric, so the ceiling could never fire: `compact_full` could have grown
without bound and the gate would have reported "Absolute-floor violations: 0
(1 justification-suppressed)". The hazard was even written down here; what was
missed is that the condition was already true.

The `metrics:` key in the front-matter is the fix — this file now suppresses only
the two pooled summary metrics it actually explains, and
`peak_rss_mb__compact_full` is gated again. A guard test,
`test_no_floor_is_fully_suppressed_by_an_active_justification`, now fails on any
floor sitting under an unscoped suppression, so this cannot recur silently.

## When to remove this file

Once a promoted baseline has been captured **with** the `compact_full` arm
present, the comparison point includes it and the pooled medians line up again.
Delete this file in the same change that promotes that baseline.
