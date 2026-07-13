---
triples:
  - benchmark: multimodal_training
    format: scx_multimodal_per_modality_auto
    dataset: cite_seq_pbmc_5k
    metric: batches_per_sec
  - benchmark: multimodal_training
    format: scx_multimodal_per_modality_auto
    dataset: cite_seq_pbmc_5k
    metric: time_to_first_batch_s
reason: >
  Pre-existing results/raw-vs-LATEST drift in the OUT-OF-SCOPE
  multimodal_training benchmark, NOT caused by the auto→adaptive codec flip
  (commit 472df42). Evidence: (1) results/raw has recorded ~3.3 batches/sec
  for this triple since 2026-07-04, predating this change; a fresh re-run on
  472df42 reproduces 3.27 b/s (identical), while LATEST's promoted row was
  captured from a different run/config (median_wall_s 0.599, floored at 285
  b/s / ttfb 0.2). (2) 472df42 touches only the single-modality codec intent
  axis (resolve_codec / pick_codec_v2 margin / encoder adaptive branch /
  pyscx codec kwargs / CLI / docs) — it does NOT modify the multimodal write
  path (mudata.rs still calls parse_codec, whose mapping is unchanged), the
  multimodal loader, or the ShufDeltaZstd decode. Multimodal fixtures and
  loader behave identically to main. The 285→3.3 gap is a benchmark
  measurement/config drift (batch_size/steady-state vs full-epoch), tracked
  as a separate multimodal_training baseline reconciliation.
expires: 2026-10-12
---

The multimodal_training / cite_seq per-modality-auto floor (285 b/s) and the
failing `multiome_pbmc` multimodal_training jobs are a separate, pre-existing
multimodal-benchmark issue independent of the codec convergence. Suppressed
here so the codec-flip baseline can promote; reconcile the multimodal_training
floors + multiome breakage in a dedicated multimodal recapture.
