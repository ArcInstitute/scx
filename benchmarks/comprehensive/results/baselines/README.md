# Canonical baselines

One snapshot per release lives here. `promote_baseline.py` copies exactly four
files per `<version>/` directory, and refuses the promotion if any is missing:

- `summary.json` — per-`(benchmark, format, dataset)` median wall / RSS /
  size, written by `capture_baseline.py`.
- `environment.json` — git SHA, dirty flag, library versions, thread
  pinning at snapshot capture time.
- `MANIFEST.sha256` — BLAKE-equivalent tamper-evidence over every file in
  the original snapshot tree.
- `fingerprints/fingerprints.json` — the eight accelerator array hashes
  `compare_against_baseline.py --gate` diffs. Three early baselines
  (`v0.5.0-phase5`, both `v0.6.0-gpu-phase1-7*`) predate it, and against those
  the fingerprint diff **silently no-ops** rather than reporting anything.

Raw `raw/*.json` files are NOT committed — they're regenerated from the
snapshot identified in `environment.json`. The gate operates on `summary.json`.

**What the manifest actually verifies.** Promotion copies four files but does
not re-manifest, so a baseline whose `MANIFEST.sha256` came from the original
snapshot lists `raw/*.json` and `fingerprints/arrays/*.npy` that were never
copied: `sha256sum -c` on `v0.14.0-phase5c-streaming-floors` reports 12 missing
files and 3 OK. Some baselines (`v0.11.2`, `v0.10.6`, `v0.6.3`) instead carry a
3-line manifest regenerated after promotion. Either way, treat it as
tamper-evidence for the files that survive promotion — filter the manifest to
those before running `sha256sum -c`, or the missing lines read as failures.

Use `benchmarks/comprehensive/scripts/promote_baseline.py` to copy a
candidate snapshot into this tree:

```bash
python benchmarks/comprehensive/scripts/promote_baseline.py \
    --snapshot benchmarks/comprehensive/results/candidate_2026_04_18_batch_d_t3 \
    --version v0.5.0-phase5
```

`<version>` is typically a git tag but any filesystem-safe label works.
`promote_baseline.py` refuses to overwrite an existing `<version>/`
unless `--force` is passed — canonical baselines should only change
with an explicit, deliberate promotion.

Run the gate against a committed baseline with:

```bash
python benchmarks/comprehensive/scripts/compare_against_baseline.py \
    --baseline benchmarks/comprehensive/results/baselines/LATEST \
    --current  benchmarks/comprehensive/results/candidate_$(date +%Y_%m_%d) \
    --gate \
    --justifications benchmarks/comprehensive/results/justifications \
    --thresholds    benchmarks/comprehensive/thresholds.yaml
```

## Picking a baseline

**`LATEST` points at `v0.16.0-opt-instruments`** (since 2026-09-03). 1833 rows
across 52 benchmark families and 13 datasets, multi-surface, threads unpinned.
It is the first baseline with rows for the SCX operations instrumented in the
PR-01/02 series — `cellset_gather`, `conversion_streaming`, `export_streaming`,
`obs_open`, `ooc_loader`, `build_csc`, `mtx_export`, `fragment_ops`'
`obs_import` arm, `grouped_sort`/`grouped_read`, `shuffle_layout`,
`accel_r_route`, `accel_harmony`, `accel_de_nb_glm`, `accel_eval_metrics`,
`doublet_interop` — 15 families its predecessor did not carry at all.

Two historical cautions that still shape the tree:

- **Accel-only vs multi-surface.** Several baselines
  (`v0.4.3-g1-gpu-de`, `v0.6.4-accel-gpu-rapids`, `v0.6.5-accel-gpu-rapids-floors`)
  were captured `--accel-only` and carry no format / cloud / multimodal rows at
  all. A gate against one of those from a format PR compares against nothing.
  Pin a full capture explicitly when reaching back before `v0.6.5`.
- **A newer capture is not automatically a better baseline.**
  `v0.11.5-dataload-phase0` (2026-07-23) is newer than `LATEST` and has 1603
  rows including the whole data-load family — `cellset_gather`, `obs_open`,
  `ooc_loader`, which `v0.11.2` lacks — but it was captured at `--tier xl`, and
  the xl tier drops the multimodal datasets, so it has no
  `multimodal_compression` / `multimodal_training` /
  `multimodal_read_streaming_vs_inmemory` rows. Promoting it would have traded
  three benchmarks for three. That is the same reason
  `v0.14.0-phase5c-streaming-floors` was promoted with `--no-latest`: four rows
  cannot replace a full-tier reference.

## Versions

| Version | Date | Coverage | Notes |
|---|---|---|---|
| `v0.5.0-phase5` | 2026-04-18 | format-level only | Pre-Phase-9; cloud + compression + read/write/memory across 6 datasets |
| `v0.6.0-gpu-phase1-7` | 2026-04-23 | format + accel pbmc3k smoke | First baseline with `accel_*` rows; pbmc3k only, 20 accel cells. Captured from `accel_smoke_2026_04_23` |
| `v0.6.0-gpu-phase1-7-multidataset` | 2026-04-24 | format + accel × {pbmc3k, tabula_sapiens_100k, census_1m} | 60/60 accel cells. Surfaces real correctness floors via `runs[].extra` (Phase 9 follow-up findings 1-4) |
| `v0.6.1-multimodal` | 2026-05-08 | format + accel + multimodal | First baseline carrying `multimodal_compression` / `multimodal_training` rows. Real CITE-seq + Multiome data (`cite_seq_pbmc` 5247 cells × {33538 RNA + 32 ADT}; `multiome_pbmc` 11898 cells × {36601 RNA + 143887 ATAC}). Absolute floors gate output_size_bytes / compression_ratio_vs_h5mu / batches_per_sec / time_to_first_batch_s on the SCX per-modality auto path. |
| `v0.6.2-n_counts-augmentation` | 2026-05-11 | format + accel + cloud + multimodal | Multi-surface baseline, 806 rows across all 8 datasets (`pbmc3k`, `pbmc10k`, `smartseq2`, `tabula_sapiens_100k`, `census_500k`, `census_1m`, `cite_seq_pbmc`, `multiome_pbmc`). Captured from `b1629ea` (multi-modal branch) at tier `full`. At the time it was promoted, accelerator gating had to be pinned to `v0.4.3-g1-gpu-de` separately; that split is gone now that `LATEST` is multi-surface. |
| `v0.4.3-g1-gpu-de` | 2026-05-22 | accel-only × {pbmc3k → census_1m} | First baseline with `accel_de` (PR series G1: `pdex_ref` + Wilcoxon `rank_genes_groups`, CPU + GPU). Captures `de_pval_agreement_vs_cpu` / `de_top_gene_overlap_vs_cpu` GPU-vs-CPU parity in `runs[].extra` so future drift trips the gate. Also adds `accel_leiden__pyscx_gpu` rows on `census_500k` + `census_1m` (the prior accel-only baseline `v0.6.0-gpu-phase1-7-multidataset` lacked census Leiden GPU). 140 rows across 7 accel benchmarks × 6 datasets — `accel_pca` 30 rows (5 impls), `accel_de` 20 rows (5 impls; GPU paths skip datasets where the v1 8192-cell sort-pool cap rejects them — `wilcoxon_gpu` on `>pbmc3k`, `pdex_ref_gpu` on `>smartseq2`), others 18 rows (3 impls). Captured from `99e39b7` (`gpu-diff-exp`) at `--tier full --accel-only`. |
| `v0.6.3-accel-gpu-de` | 2026-05-31 | full + accel | 906 rows. |
| `v0.6.4-gpu-de-v3` | 2026-06-03 | full + accel | 1098 rows from `candidate_v3routes_20260602`. GPU DE v3 routes. |
| `v0.6.4-accel-gpu-rapids` | 2026-06-07 | accel-only | 204 rows. Superseded as the accel baseline by `v0.6.5-accel-gpu-rapids-floors` the same week. |
| `v0.6.5-accel-gpu-rapids-floors` | 2026-06-08 | accel-only × {pbmc3k, pbmc10k, smartseq2, tabula_sapiens_100k, census_500k, census_1m} | Rapids-routing re-promotion: first baseline where in-VRAM `device="gpu"` accel ops route to **rapids-singlecell** (route `rapids_singlecell_gpu`), captured from `d731f88` (Phase 1+2 merged) at `--tier full --accel-only`. 218 cells, 0 failures. Supersedes `v0.6.4-accel-gpu-rapids` as the accel `LATEST`. New absolute floors gate the rapids routes + the no-rapids fallback: `{pca,knn,umap,preprocess,pipeline}_route_rapids_correct` and `{pca,knn,umap}_fallback_no_rapids_correct` (all 1.0 across all 6 tiers in this capture). Native `*_route_gpu_correct` floors retained (their `pyscx_gpu_*` variants run under `SCX_FORCE_NATIVE_GPU=1`). |
| `v0.6.5-accel-gpu-to-gpu-anndata` | 2026-06-09 | full + accel | 1023 rows from `candidate_749640a_20260609`, the 12.94 h `--tier full --include-accel` capture. First baseline with `accel_to_gpu_anndata` rows. |
| `v0.10.6-default-framed` | 2026-07-05 | full + accel | 1188 rows from `candidate_final_default_framed` — framed row groups on by default. |
| `v0.11.0-cellstream-recapture` | 2026-07-11 | full + accel | 1552 rows from `baseline_2026_07_10`; adds the `cellstream` format variant. |
| `v0.11.0-auto-converge` | 2026-07-12 | full + accel | 1551 rows from `candidate_auto_converge_20260712` — the `auto` → cost-aware-adaptive flip. `scx_auto_v2` rows disappear here by design. |
| `v0.11.2-multimodal-loader-fix` | 2026-07-12 | full + accel + multimodal | Same 1551-row snapshot as `v0.11.0-auto-converge`, re-promoted after the multimodal loader fix. `pyscx 0.11.0`. **Zero rows for 13 of the 35 floored benchmarks** — a gate over those tests absolute floors only and performs no regression comparison, which is what a recapture has to fix. |
| `v0.11.5-dataload-phase0` | 2026-07-23 | xl + accel, no multimodal | 1603 rows; first with `cellset_gather` / `obs_open` / `ooc_loader`. Deliberately **not** promoted — see "Picking a baseline". |
| `v0.14.0-phase5c-streaming-floors` | 2026-08-22 | `--tier small --datasets census_1m` | 4 rows (`conversion_streaming`, `export_streaming`, `obs_open` x2), promoted with `--no-latest` in `f97c8ee1` (#451). Its `summary.json` medians all nine runs of the three arms into one figure per triple, so it **cannot back a per-arm claim** — only its `raw/` can, and those four files are force-added to git for exactly that reason. `git_dirty: true`. |
| `v0.18.0-community-benchmarks` | 2026-09-21 | `full`, four benchmarks only | 82 rows from `candidate_community_20260920` at `ba3a2bbc` — the first capture of `accel_qc_filter`, `accel_score_genes`, `pipeline_ooc_constrained` and `multimodal_atlas_streaming`, and the first to run the last two at census and atlas scale. Promoted with **`--no-latest`**: `LATEST` keeps 52 families and 1,833 rows, and four cannot replace that — the same reasoning as `v0.14.0-phase5c-streaming-floors`. Its purpose is provenance for the 52 `thresholds.yaml` floors added alongside it and for the numbers back-filled into `docs/performance.md`, not regression comparison. **Measured on `scx-bench`'s pyscx 0.18.0 release**, not the branch's 0.19.0 — the version string says 0.18.0 for that reason. 79 of 82 cells exited 0; the three that did not are `pipeline_ooc_constrained__scanpy_{16g,32g}` at census scale, which is the result rather than a failure: they record `pipeline_completed_int = 0.0` with the stage the ceiling killed them in. `git_dirty: true` — during the wave the only dirty tracked files were `benchmarks/comprehensive/reporting/*` and `tests/test_reporting_pipeline.py` (landed as `b43eebf6` mid-capture); no benchmark worker imports either, and the measured modules plus pyscx were unmodified at `ba3a2bbc` throughout. |
| `v0.16.0-opt-instruments` (`LATEST`) | 2026-09-03 | full + additional + accel + multimodal | 1833 rows / 52 families / 13 datasets, from `candidate_unpinned_20260903` at `cafeb2ce`. Two passes accumulating into one snapshot: `--tier full --include-additional`, then the five benchmarks reachable only from off-tier datasets (`grouped_sort`, `grouped_read`, `accel_de_nb_glm`, `accel_eval_metrics`, `cell_eval_parity_perf`). **11 of the 14 floored benchmarks that had zero rows in `v0.11.2` now have them.** Gates clean against itself: 0 timing / 0 RSS / 0 size regressions, 0 fingerprint mismatches, 0 floor violations (13 justification-suppressed). Known gaps, each recorded with its cause: no `cellstream` rows (upstream package restructured, runner needs porting — 47 rows lost vs `v0.11.2`); no `cloud_large_atlas` (its `<ds>.scxd/` fixtures are unstaged, ~4.5 GB); no `cell_eval_parity_perf` (its `cell_eval`/`arc_bench`/`pdex` deps are editable installs in `.venv` only and no conda env has them); no `cellset_gather` at census scale (does not fit a practical time budget — see thresholds' Deferred item 15). `git_dirty: true`, structurally — the capture overwrites the four git-tracked `results/raw/*.json` files that back `docs/performance.md`. |
