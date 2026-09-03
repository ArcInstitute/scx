---
# Scoped, because this file explains exactly one quantity. Unscoped it also
# suppressed `leiden_route_gpu_correct min: 1.0` on pbmc3k — a route assertion
# whose entire purpose is to turn a silent GPU->CPU dispatch fallback into a
# hard gate failure. A label-divergence justification has nothing to say about
# whether the GPU route was taken at all: if Leiden quietly stopped reaching
# the GPU, ARI against leidenalg would *improve* (the CPU path holds 0.81-0.97)
# and the route floor was the only thing that would have noticed.
#
# `median_wall_s` and `peak_rss_mb_median` are deliberately NOT in the scope
# either: a partition that differs in labels does not differ in cost, so a
# timing move on these triples is not explained by anything written below.
metrics: [ari_vs_leidenalg]
triples:
  - benchmark: accel_leiden
    format: accel_leiden__pyscx_gpu
    dataset: pbmc3k
  - benchmark: accel_leiden
    format: accel_leiden__pyscx_gpu
    dataset: pbmc10k
  - benchmark: accel_leiden
    format: accel_leiden__pyscx_gpu
    dataset: smartseq2
  - benchmark: accel_leiden
    format: accel_leiden__pyscx_gpu
    dataset: tabula_sapiens_100k
  - benchmark: accel_leiden
    format: accel_leiden__pyscx_gpu
    dataset: census_500k
  - benchmark: accel_leiden
    format: accel_leiden__pyscx_gpu
    dataset: census_1m
reason: >
  GPU Leiden via cuGraph (device='gpu') produces a partition that
  diverges from Python leidenalg on real single-cell graphs. This is
  a documented behaviour in CLAUDE.md (Known Limitations: 'GPU Leiden
  correctness: label stability differs from Python leidenalg —
  documented behaviour of device=gpu / auto on GPU hosts, not a
  regression. Pin device=cpu to preserve label stability for
  downstream DE / annotation transfer.').
  The 0.85 floor on accel_leiden__pyscx_gpu was set from a
  first-comprehensive-run measurement of 0.893 on pbmc3k, but that
  measurement was on an older RAPIDS/cuGraph that no longer
  reproduces. Empirical 2026-05-21 measurements on cuGraph 25.12.02
  AND 26.02.00 (both with cuda-version=12.6) show ari ~ 0.04 on
  pbmc3k, ~0.002 on tabula_sapiens_100k, ~0.0005 on census_1m;
  both cuGraph minor versions agree to three decimal places, ruling
  out a recent regression. CPU path (pyscx_cpu, scx-accel) holds at
  ari ~ 0.81 - 0.97 and remains the supported correctness path; its
  0.85 floor stays in place. Suppressing rather than removing the
  GPU floor preserves the assertion for when cuGraph Leiden converges
  back toward leidenalg parity.
# Dated, unlike the permanent limitation it cites, because the thing worth
# rechecking is not the limitation but the *convergence*: this file exists so
# the floor comes back when cuGraph Leiden approaches leidenalg again, and a
# suppression with no date is one nobody ever rechecks. Re-measure ARI on
# pbmc3k against whatever cuGraph the GPU env then carries; if it still reads
# ~0.04, renew with the new version recorded.
expires: 2027-06-30
---
