---
triples:
  - benchmark: cloud_push
    format: scx_auto
    dataset: tabula_sapiens_100k
reason: >
  cloud_push throughput_mbps to the shared GCS bucket
  (gs://arc-ctc-nextflow/scx-test) is network- and contention-dependent:
  intra-region egress varies with concurrent cluster load and GCS-side
  throttling. The 50 MB/s floor was set from a less-contended run; the
  v0.6.4-gpu-de-v3 recapture measured 24.96 MB/s for tabula_sapiens_100k
  while the cluster was under heavy concurrent benchmark load (16+ job
  preemptions during the same campaign). This is an environmental /
  network-variability floor, not an SCX cloud-write regression — the push
  path (pyscx.push → object_store GCS) is unchanged, and the push
  bytes/section counts are unaffected. Re-evaluate the floor or this
  justification if a dedicated low-contention cloud run consistently
  clears 50 MB/s.
expires: 2026-12-31
---

cloud_push throughput is network/contention-variable on the shared GCS test
bucket; the 24.96 MB/s measurement was taken under heavy concurrent cluster
load during the v0.6.4 baseline recapture. Not an SCX regression.
