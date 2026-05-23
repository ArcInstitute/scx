# Canonical baselines

One snapshot per release lives here. Only three files are committed per
`<version>/` directory:

- `summary.json` — per-`(benchmark, format, dataset)` median wall / RSS /
  size, written by `capture_baseline.py`.
- `environment.json` — git SHA, dirty flag, library versions, thread
  pinning at snapshot capture time.
- `MANIFEST.sha256` — BLAKE-equivalent tamper-evidence over every file in
  the original snapshot tree.

Raw `raw/*.json` files are NOT committed — they're regenerated from the
snapshot identified in `environment.json` and verified against the
manifest. The gate operates on `summary.json`; the manifest lets
operators audit drift by re-running the capture against the recorded
git SHA.

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

Two flavours live side-by-side. **Multi-surface** baselines (the default `LATEST` target) gate format / cloud / multimodal benchmarks. **Accel-only** baselines gate the `accel_*` family — they exist because capture coverage and gate coverage on accelerators have drifted (`benchmarks/README.md` "Capture vs gate coverage" — the `accel_*` rows in `LATEST` capture cleanly but produce no gate signal against the current multi-surface baseline). PRs that touch `accel_*` should pin the accel-only baseline explicitly:

```bash
# Format / cloud / multimodal PRs: default LATEST.
python benchmarks/comprehensive/scripts/gate_candidate.py --no-accel

# Accel PRs (PCA / kNN / UMAP / Leiden / preprocess / HVG / DE):
python benchmarks/comprehensive/scripts/gate_candidate.py --accel-only \
    --baseline benchmarks/comprehensive/results/baselines/v0.4.3-g1-gpu-de
```

## Versions

| Version | Date | Coverage | Notes |
|---|---|---|---|
| `v0.5.0-phase5` | 2026-04-18 | format-level only | Pre-Phase-9; cloud + compression + read/write/memory across 6 datasets |
| `v0.6.0-gpu-phase1-7` | 2026-04-23 | format + accel pbmc3k smoke | First baseline with `accel_*` rows; pbmc3k only, 20 accel cells. Captured from `accel_smoke_2026_04_23` |
| `v0.6.0-gpu-phase1-7-multidataset` | 2026-04-24 | format + accel × {pbmc3k, tabula_sapiens_100k, census_1m} | 60/60 accel cells. Surfaces real correctness floors via `runs[].extra` (Phase 9 follow-up findings 1-4) |
| `v0.6.1-multimodal` | 2026-05-08 | format + accel + multimodal | First baseline carrying `multimodal_compression` / `multimodal_training` rows. Real CITE-seq + Multiome data (`cite_seq_pbmc` 5247 cells × {33538 RNA + 32 ADT}; `multiome_pbmc` 11898 cells × {36601 RNA + 143887 ATAC}). Absolute floors gate output_size_bytes / compression_ratio_vs_h5mu / batches_per_sec / time_to_first_batch_s on the SCX per-modality auto path. |
| `v0.6.2-n_counts-augmentation` (`LATEST`) | 2026-05-11 | format + accel + cloud + multimodal | Multi-surface baseline, 806 rows across all 8 datasets (`pbmc3k`, `pbmc10k`, `smartseq2`, `tabula_sapiens_100k`, `census_500k`, `census_1m`, `cite_seq_pbmc`, `multiome_pbmc`). Captured from `b1629ea` (multi-modal branch) at tier `full`. The `accel_*` rows capture cleanly but produce no gate signal against `LATEST` — pin `v0.4.3-g1-gpu-de` for accelerator gating. |
| `v0.4.3-g1-gpu-de` | 2026-05-22 | accel-only × {pbmc3k → census_1m} | First baseline with `accel_de` (PR series G1: `pdex_ref` + Wilcoxon `rank_genes_groups`, CPU + GPU). Captures `de_pval_agreement_vs_cpu` / `de_top_gene_overlap_vs_cpu` GPU-vs-CPU parity in `runs[].extra` so future drift trips the gate. Also adds `accel_leiden__pyscx_gpu` rows on `census_500k` + `census_1m` (the prior accel-only baseline `v0.6.0-gpu-phase1-7-multidataset` lacked census Leiden GPU). 140 rows across 7 accel benchmarks × 6 datasets — `accel_pca` 30 rows (5 impls), `accel_de` 20 rows (5 impls; GPU paths skip datasets where the v1 8192-cell sort-pool cap rejects them — `wilcoxon_gpu` on `>pbmc3k`, `pdex_ref_gpu` on `>smartseq2`), others 18 rows (3 impls). Captured from `99e39b7` (`gpu-diff-exp`) at `--tier full --accel-only`. |
