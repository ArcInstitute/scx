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
    --baseline benchmarks/comprehensive/results/baselines/v0.5.0-phase5 \
    --current  benchmarks/comprehensive/results/candidate_$(date +%Y_%m_%d) \
    --gate \
    --justifications benchmarks/comprehensive/results/justifications \
    --thresholds    benchmarks/comprehensive/thresholds.yaml
```
