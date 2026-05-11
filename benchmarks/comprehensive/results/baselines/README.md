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

## Versions

| Version | Date | Coverage | Notes |
|---|---|---|---|
| `v0.5.0-phase5` | 2026-04-18 | format-level only | Pre-Phase-9; cloud + compression + read/write/memory across 6 datasets |
| `v0.6.0-gpu-phase1-7` | 2026-04-23 | format + accel pbmc3k smoke | First baseline with `accel_*` rows; pbmc3k only, 20 accel cells. Captured from `accel_smoke_2026_04_23` |
| `v0.6.0-gpu-phase1-7-multidataset` | 2026-04-24 | format + accel × {pbmc3k, tabula_sapiens_100k, census_1m} | 60/60 accel cells. Surfaces real correctness floors via `runs[].extra` (Phase 9 follow-up findings 1-4) |
| `v0.6.1-multimodal` (LATEST) | 2026-05-08 | format + accel + multimodal | First baseline carrying `multimodal_compression` / `multimodal_training` rows. Real CITE-seq + Multiome data (`cite_seq_pbmc` 5247 cells × {33538 RNA + 32 ADT}; `multiome_pbmc` 11898 cells × {36601 RNA + 143887 ATAC}). Absolute floors gate output_size_bytes / compression_ratio_vs_h5mu / batches_per_sec / time_to_first_batch_s on the SCX per-modality auto path. |
