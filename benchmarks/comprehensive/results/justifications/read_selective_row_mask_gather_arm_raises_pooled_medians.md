---
# Scoped, not whole-triple: this file explains only the two pooled summary
# metrics, so the `peak_rss_mb__row_mask_gather` ceilings that land in the same
# change stay armed (see "What this file must not suppress").
metrics: [peak_rss_mb_median, median_wall_s]
# Expires deliberately: the recipe below waits on the next promoted baseline,
# and a missed cleanup should fail loud rather than suppress forever.
expires: 2026-12-31
triples:
  - benchmark: read_selective
    format: anndata_zarr_backed
    dataset: census_1m
  - benchmark: read_selective
    format: anndata_zarr_backed
    dataset: census_500k
  - benchmark: read_selective
    format: anndata_zarr_backed
    dataset: pbmc10k
  - benchmark: read_selective
    format: anndata_zarr_backed
    dataset: pbmc3k
  - benchmark: read_selective
    format: anndata_zarr_backed
    dataset: smartseq2
  - benchmark: read_selective
    format: anndata_zarr_backed
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: annloader
    dataset: census_1m
  - benchmark: read_selective
    format: annloader
    dataset: census_500k
  - benchmark: read_selective
    format: annloader
    dataset: pbmc10k
  - benchmark: read_selective
    format: annloader
    dataset: pbmc3k
  - benchmark: read_selective
    format: annloader
    dataset: smartseq2
  - benchmark: read_selective
    format: annloader
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: h5ad_gzip
    dataset: census_1m
  - benchmark: read_selective
    format: h5ad_gzip
    dataset: census_500k
  - benchmark: read_selective
    format: h5ad_gzip
    dataset: pbmc10k
  - benchmark: read_selective
    format: h5ad_gzip
    dataset: pbmc3k
  - benchmark: read_selective
    format: h5ad_gzip
    dataset: smartseq2
  - benchmark: read_selective
    format: h5ad_gzip
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: h5ad_lzf
    dataset: census_1m
  - benchmark: read_selective
    format: h5ad_lzf
    dataset: census_500k
  - benchmark: read_selective
    format: h5ad_lzf
    dataset: pbmc10k
  - benchmark: read_selective
    format: h5ad_lzf
    dataset: pbmc3k
  - benchmark: read_selective
    format: h5ad_lzf
    dataset: smartseq2
  - benchmark: read_selective
    format: h5ad_lzf
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: h5ad_none
    dataset: census_1m
  - benchmark: read_selective
    format: h5ad_none
    dataset: census_500k
  - benchmark: read_selective
    format: h5ad_none
    dataset: pbmc10k
  - benchmark: read_selective
    format: h5ad_none
    dataset: pbmc3k
  - benchmark: read_selective
    format: h5ad_none
    dataset: smartseq2
  - benchmark: read_selective
    format: h5ad_none
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: scdataset
    dataset: census_1m
  - benchmark: read_selective
    format: scdataset
    dataset: census_500k
  - benchmark: read_selective
    format: scdataset
    dataset: pbmc10k
  - benchmark: read_selective
    format: scdataset
    dataset: pbmc3k
  - benchmark: read_selective
    format: scdataset
    dataset: smartseq2
  - benchmark: read_selective
    format: scdataset
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: scx_auto
    dataset: census_1m
  - benchmark: read_selective
    format: scx_auto
    dataset: census_500k
  - benchmark: read_selective
    format: scx_auto
    dataset: pbmc10k
  - benchmark: read_selective
    format: scx_auto
    dataset: pbmc3k
  - benchmark: read_selective
    format: scx_auto
    dataset: smartseq2
  - benchmark: read_selective
    format: scx_auto
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: scx_compact_trial
    dataset: census_1m
  - benchmark: read_selective
    format: scx_compact_trial
    dataset: census_500k
  - benchmark: read_selective
    format: scx_compact_trial
    dataset: pbmc10k
  - benchmark: read_selective
    format: scx_compact_trial
    dataset: pbmc3k
  - benchmark: read_selective
    format: scx_compact_trial
    dataset: smartseq2
  - benchmark: read_selective
    format: scx_compact_trial
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: scx_compact_trial_g1024
    dataset: census_1m
  - benchmark: read_selective
    format: scx_compact_trial_g1024
    dataset: census_500k
  - benchmark: read_selective
    format: scx_compact_trial_g1024
    dataset: pbmc10k
  - benchmark: read_selective
    format: scx_compact_trial_g1024
    dataset: pbmc3k
  - benchmark: read_selective
    format: scx_compact_trial_g1024
    dataset: smartseq2
  - benchmark: read_selective
    format: scx_compact_trial_g1024
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: scx_compact_trial_g128
    dataset: census_1m
  - benchmark: read_selective
    format: scx_compact_trial_g128
    dataset: census_500k
  - benchmark: read_selective
    format: scx_compact_trial_g128
    dataset: pbmc10k
  - benchmark: read_selective
    format: scx_compact_trial_g128
    dataset: pbmc3k
  - benchmark: read_selective
    format: scx_compact_trial_g128
    dataset: smartseq2
  - benchmark: read_selective
    format: scx_compact_trial_g128
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: scx_compact_trial_g256
    dataset: census_1m
  - benchmark: read_selective
    format: scx_compact_trial_g256
    dataset: census_500k
  - benchmark: read_selective
    format: scx_compact_trial_g256
    dataset: pbmc10k
  - benchmark: read_selective
    format: scx_compact_trial_g256
    dataset: pbmc3k
  - benchmark: read_selective
    format: scx_compact_trial_g256
    dataset: smartseq2
  - benchmark: read_selective
    format: scx_compact_trial_g256
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: scx_compact_trial_g512
    dataset: census_1m
  - benchmark: read_selective
    format: scx_compact_trial_g512
    dataset: census_500k
  - benchmark: read_selective
    format: scx_compact_trial_g512
    dataset: pbmc10k
  - benchmark: read_selective
    format: scx_compact_trial_g512
    dataset: pbmc3k
  - benchmark: read_selective
    format: scx_compact_trial_g512
    dataset: smartseq2
  - benchmark: read_selective
    format: scx_compact_trial_g512
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: scx_fast
    dataset: census_1m
  - benchmark: read_selective
    format: scx_fast
    dataset: census_500k
  - benchmark: read_selective
    format: scx_fast
    dataset: pbmc10k
  - benchmark: read_selective
    format: scx_fast
    dataset: pbmc3k
  - benchmark: read_selective
    format: scx_fast
    dataset: smartseq2
  - benchmark: read_selective
    format: scx_fast
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: scx_lz4
    dataset: census_1m
  - benchmark: read_selective
    format: scx_lz4
    dataset: census_500k
  - benchmark: read_selective
    format: scx_lz4
    dataset: pbmc10k
  - benchmark: read_selective
    format: scx_lz4
    dataset: pbmc3k
  - benchmark: read_selective
    format: scx_lz4
    dataset: smartseq2
  - benchmark: read_selective
    format: scx_lz4
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: scx_none
    dataset: census_1m
  - benchmark: read_selective
    format: scx_none
    dataset: census_500k
  - benchmark: read_selective
    format: scx_none
    dataset: pbmc10k
  - benchmark: read_selective
    format: scx_none
    dataset: pbmc3k
  - benchmark: read_selective
    format: scx_none
    dataset: smartseq2
  - benchmark: read_selective
    format: scx_none
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: scx_pcodec
    dataset: census_1m
  - benchmark: read_selective
    format: scx_pcodec
    dataset: census_500k
  - benchmark: read_selective
    format: scx_pcodec
    dataset: pbmc10k
  - benchmark: read_selective
    format: scx_pcodec
    dataset: pbmc3k
  - benchmark: read_selective
    format: scx_pcodec
    dataset: smartseq2
  - benchmark: read_selective
    format: scx_pcodec
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: scx_scx1
    dataset: census_1m
  - benchmark: read_selective
    format: scx_scx1
    dataset: census_500k
  - benchmark: read_selective
    format: scx_scx1
    dataset: pbmc10k
  - benchmark: read_selective
    format: scx_scx1
    dataset: pbmc3k
  - benchmark: read_selective
    format: scx_scx1
    dataset: smartseq2
  - benchmark: read_selective
    format: scx_scx1
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: scx_shufdelta
    dataset: census_1m
  - benchmark: read_selective
    format: scx_shufdelta
    dataset: census_500k
  - benchmark: read_selective
    format: scx_shufdelta
    dataset: pbmc10k
  - benchmark: read_selective
    format: scx_shufdelta
    dataset: pbmc3k
  - benchmark: read_selective
    format: scx_shufdelta
    dataset: smartseq2
  - benchmark: read_selective
    format: scx_shufdelta
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: scx_zstd
    dataset: census_1m
  - benchmark: read_selective
    format: scx_zstd
    dataset: census_500k
  - benchmark: read_selective
    format: scx_zstd
    dataset: pbmc10k
  - benchmark: read_selective
    format: scx_zstd
    dataset: pbmc3k
  - benchmark: read_selective
    format: scx_zstd
    dataset: smartseq2
  - benchmark: read_selective
    format: scx_zstd
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: shardad
    dataset: census_1m
  - benchmark: read_selective
    format: shardad
    dataset: census_500k
  - benchmark: read_selective
    format: shardad
    dataset: pbmc10k
  - benchmark: read_selective
    format: shardad
    dataset: pbmc3k
  - benchmark: read_selective
    format: shardad
    dataset: smartseq2
  - benchmark: read_selective
    format: shardad
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: slaf
    dataset: census_1m
  - benchmark: read_selective
    format: slaf
    dataset: pbmc10k
  - benchmark: read_selective
    format: slaf
    dataset: pbmc3k
  - benchmark: read_selective
    format: slaf
    dataset: smartseq2
  - benchmark: read_selective
    format: slaf
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: tiledb_soma
    dataset: census_1m
  - benchmark: read_selective
    format: tiledb_soma
    dataset: census_500k
  - benchmark: read_selective
    format: tiledb_soma
    dataset: pbmc10k
  - benchmark: read_selective
    format: tiledb_soma
    dataset: pbmc3k
  - benchmark: read_selective
    format: tiledb_soma
    dataset: smartseq2
  - benchmark: read_selective
    format: tiledb_soma
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: zarr_lz4
    dataset: census_1m
  - benchmark: read_selective
    format: zarr_lz4
    dataset: census_500k
  - benchmark: read_selective
    format: zarr_lz4
    dataset: pbmc10k
  - benchmark: read_selective
    format: zarr_lz4
    dataset: pbmc3k
  - benchmark: read_selective
    format: zarr_lz4
    dataset: smartseq2
  - benchmark: read_selective
    format: zarr_lz4
    dataset: tabula_sapiens_100k
  - benchmark: read_selective
    format: zarr_zstd
    dataset: census_1m
  - benchmark: read_selective
    format: zarr_zstd
    dataset: census_500k
  - benchmark: read_selective
    format: zarr_zstd
    dataset: pbmc10k
  - benchmark: read_selective
    format: zarr_zstd
    dataset: pbmc3k
  - benchmark: read_selective
    format: zarr_zstd
    dataset: smartseq2
  - benchmark: read_selective
    format: zarr_zstd
    dataset: tabula_sapiens_100k
reason: "read_selective gained a fifth scenario, row_mask_gather (every eighth cell, touching every shard), whose runs join the pool the per-triple median_wall_s and peak_rss_mb_median are taken over; both pooled numbers move by construction. Neither is a regression."
---

# `read_selective` pooled medians moved when the `row_mask_gather` arm landed

## What changed

`read_selective` gained a fifth scenario, `row_mask_gather`: read every eighth
cell (`np.arange(0, n_obs, 8)`, the sorted-index form of an interleaved boolean
mask) through each runner's `read_subset(cell_indices=…)`. It exists because the
existing `row_slice` draws `QUERY_N_CELLS` random cells, which on a large file
lands in a fraction of the shards; the interleaved eighth touches **every**
shard and so measures the per-shard cost of a row gather — the `handle[mask]`
pattern arc-reactor's guide calling runs, and the one PR C (REC-1) rewrote in
`BackedCsrReader::read_row_indices` to assemble its output once instead of
building a per-row CSR and concatenating (2× the result on top of the LRU).

The same change wraps the `row_mask_gather` read — and only that read — in
`PeakRssSampler` and records `peak_rss_mb__row_mask_gather` per run, a true
in-region peak. The reserved `peak_rss_mb` on every run record is still the
runner's `max(before, after)` of two instantaneous readings and is unchanged; it
cannot see the transient the gather used to allocate and free, which is why the
new key exists.

## Why the medians move, and why that is not a regression

`capture_baseline.archive_raw_results` writes one `median_wall_s` and one
`peak_rss_mb_median` per `(benchmark, format, dataset)` triple, taken across
**every** run in the file — `_median_rss` does not partition by
`extra["scenario"]`. `read_selective` already pooled `row_slice`,
`col_projection`, `combined` and up to three predicate arms (the module docstring
records an IQR of 11.54 s against a 13.85 s median on census_1m for exactly this
reason). Adding a fifth scenario that reads one eighth of the matrix shifts both
pooled numbers by construction: its wall is above `row_slice`'s (more cells) and
its RSS carries a result an eighth the size of the matrix.

The per-scenario numbers are where the signal is: `wall_s__<scenario>` and the
new `peak_rss_mb__row_mask_gather` are sparse keys — only that scenario's runs
carry them, so a threshold on one medians that arm alone. The new arm's memory
is gated on its own key on `scx_auto` at `tabula_sapiens_100k` and
`census_500k`.

This is the same shape as `fragment_ops_compact_full_arm_raises_pooled_medians.md`
and `cellset_gather_s512_raises_pooled_rss.md`, for the same reason.

## Scope

Every `read_selective` triple `results/baselines/LATEST` carries — the scenario
is cross-format (every runner implements `read_subset(cell_indices=…)`), so every
format's pooled medians move, not only scx's.

## What this file must not suppress

The two `peak_rss_mb__row_mask_gather` ceilings added in the same change sit on
triples this file names. `metrics:` scopes the suppression to the two pooled
summary metrics, so those ceilings stay armed;
`test_no_floor_is_fully_suppressed_by_an_active_justification` fails on any
floor sitting under an unscoped suppression.

## When to remove this file

Once a promoted baseline has been captured **with** the `row_mask_gather` arm
present, the comparison point includes it and the pooled medians line up again.
Delete this file in the same change that promotes that baseline.
