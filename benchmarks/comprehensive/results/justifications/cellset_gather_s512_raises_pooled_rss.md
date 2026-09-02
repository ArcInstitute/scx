---
# Scoped, not whole-triple: this file explains the two POOLED summary metrics
# and nothing else. Unscoped it also suppressed all 15 `cellset_gather`
# absolute floors — see "What this file must not suppress" below.
metrics: [peak_rss_mb_median, median_wall_s]
triples:
  - benchmark: cellset_gather
    format: scx_auto
    dataset: tabula_sapiens_100k
  - benchmark: cellset_gather
    format: scx_auto
    dataset: census_500k
  - benchmark: cellset_gather
    format: scx_auto
    dataset: census_1m
reason: "peak_rss_mb_median is pooled across scenarios; the new S=512 and N=4 rank arms legitimately raise it. Not a memory regression."
---

# `cellset_gather` pooled `peak_rss_mb_median` rose when S=512 landed

`compare_against_baseline.py` flagged
`cellset_gather / scx_auto / tabula_sapiens_100k / peak_rss_mb_median`
**656.393 → 784.170 MB (+19.47%)** against `v0.11.5-dataload-phase0`.

**This is a composition change, not a regression.** `BenchmarkResult`'s
`peak_rss_mb_median` is a median over **every recorded run**, pooling all
scenarios into one number. The baseline had two scenarios (`gather_random`,
`gather_grouped`, both S=64); the candidate has five (adding `gather_random_s512`,
`gather_grouped_s512`, and `gather_random_r4`). The per-scenario medians show
nothing moved:

| scenario | S | peak RSS (MB) |
|---|---|---|
| `gather_random` | 64 | 646.1 |
| `gather_grouped` | 64 | 659.1 |
| `gather_random_s512` | 512 | 790.7 |
| `gather_grouped_s512` | 512 | 786.8 |

The S=64 arms sit right where the baseline was (646–659 vs 656). The S=512 arms
hold 512-cell sets instead of 64-cell ones, so a higher resident footprint is the
expected and intended behaviour of the new arm — and cells/batch is held constant,
so this is per-set working-set growth, not a leak.

The `gather_random_r4` arm additionally records a **summed-across-ranks** RSS
(`total_peak_rss_mb__gather_random_r4`, e.g. 6.98 GB for 4 ranks at census_1m).
That is deliberately an aggregate of four processes and is not comparable to a
single-process figure; it is reported under its own key rather than folded into
`peak_rss_mb`, but the pooled median still sees the rank-arm rows.

`census_500k` and `census_1m` are included in this justification because they
carry the same new scenarios and will trip the same pooled-median comparison on
the next baseline that predates them.

## What this file must not suppress

`cellset_gather / scx_auto` carries **15 absolute floors** across these three
datasets (six S=64 `cellsets_per_sec__gather_*`, six S=512, three
`rank_scaling_efficiency__gather_random_r4`). Justification suppression used to
be whole-triple over *every* metric, so this file — whose entire subject is a
pooled median — silently disarmed all 15 of them, while `thresholds.yaml`'s
Deferred item 7 described them as "NOW ACTIVE … all live". The `metrics:` key in
the front-matter is what keeps them live; do not remove it, and do not add a
triple here without checking what floors sit on it.

The suppression is also **inert today**, in the direction that matters:
`results/baselines/LATEST` (`v0.11.2-multimodal-loader-fix`) predates the
data-load Phase 0 benchmarks and carries **zero** `cellset_gather` rows, so
`diff_summaries` treats every row here as *appearing* rather than regressing and
there is no pooled comparison to suppress. This file is therefore forward-looking:
it exists for the next baseline that predates the S=512 / rank / collate arms.

**When to remove this file:** once a baseline captured *with* the S=512 and rank
arms becomes the comparison point, the composition matches and the suppression is
no longer needed. If the flag persists against such a baseline, it is a real
regression — investigate rather than re-justify.

**Worth fixing properly at some point:** a per-scenario `peak_rss_mb` in the
summary would make this class of false positive impossible, instead of requiring a
justification every time a scenario is added to an existing benchmark. (The
per-scenario keys now exist in `runs[].extra`; what is still pooled is the
`summary.json` row the relative gate compares.)
