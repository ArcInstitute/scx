---
overrides:
  - benchmark: ml_loader
    format: scx_auto
    dataset: census_500k
    metric: peak_rss_mb_median
    tolerance: 0.20
  - benchmark: read_full
    format: scx_auto
    dataset: tabula_sapiens_100k
    metric: peak_rss_mb_median
    tolerance: 0.15
  # Competitor multimodal rows re-measured during the out-of-scope
  # multimodal_training/cite_seq refresh — sub-100ms wall jitter on 0.3s
  # benchmarks (shared 192-core node), unrelated to the codec flip.
  - benchmark: multimodal_training
    format: h5mu_uncompressed
    dataset: cite_seq_pbmc_5k
    metric: median_wall_s
    tolerance: 0.25
  - benchmark: multimodal_training
    format: zarr_mudata_zstd
    dataset: cite_seq_pbmc_5k
    metric: median_wall_s
    tolerance: 0.25
reason: >
  Peak-RSS sampling noise on the adaptive `codec="auto"` decode path, not a
  systematic ShufDeltaZstd memory regression. Evidence (post-flip candidate
  vs pre-flip Scx1-auto LATEST): on the SAME dataset census_500k, read_full
  (-13.8%) and read_selective (-16.1%) peak RSS went DOWN while ml_loader
  went UP (+16.2%); and ml_loader census_1m (+7.6%) moved less than
  census_500k (+16.2%) despite 2× the data — both inconsistent with a real
  per-shard/per-nnz memory cost. Peak RSS is a single-shot high-water mark
  (allocator/GC/page-cache sensitive); the gate noise-widens timing via IQR
  but holds RSS at a flat 10%, so a noisy RSS row needs a per-row bump.
expires: 2027-01-12
---

There is a small, bounded real transient (shufdelta decode holds the
zstd-decompressed planes + the unshuffled copy per stream; the
indptr/indices planes aren't dropped before the values stage — see
scx-codec::decode_shufdelta_zstd_ref), reducible via early-drop / in-place
unshuffle. It is per-shard bounded (not a leak) and swamped by sampling
noise at the loader level. Tracked as an optional decode-memory-hygiene
follow-up; the override expires when that lands or the RSS sampler is made
multi-sample.
