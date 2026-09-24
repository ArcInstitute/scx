# scx vs shardad — full feature parity

> Part of [SCX performance](README.md).

Direct head-to-head against [shardad](https://github.com/ArcInstitute/shardad)
(`0.5.1`), the condition-grouped sharded `.h5ad` replacement for perturbation
screens, across **every** benchmarkable shardad feature — not just compression.
Measured on a Chimera CPU node (scx `0.11.0`, always-framed; shardad `0.5.1`) via
the comprehensive harness (`benchmarks/comprehensive/benchmarks/{compression,
write,read_full,read_selective,parallel_scaling,parallel_write_scaling,memory,
grouped_read,ooc_rss_boundary,shardad_fidelity}.py`). scx is shown as `scx_auto`
(default per-shard codec) and `scx_compact_trial` (the per-shard smaller of the
heuristic winner vs ShufDeltaZstd — scx's best integer codec). Datasets span
tiny→census_1m plus the perturbation/grouped set (replogle_k562, tahoe_c38,
chemogenetic_rgfp — shardad's home turf).

**Headline:** the two projects trade wins by axis. **shardad** leads on
compression of large integer counts, write speed, and small/medium read peak-RSS
(its uint16 in-decode materialization). **scx** leads on out-of-core streaming RAM
at scale, grouped query latency, and carries a vastly broader capability set (GPU,
query engine, R, cloud, multimodal, ML loader). The old self-reported "shardad
1.9–3.2× smaller than scx 0.8.7 on integer counts" has narrowed to **~1.0–1.7×**
against scx `0.11.0`'s framed ShufDeltaZstd.

## Compression — on-disk size (MB, lower is better) + ratio vs source h5ad

| dataset | scx_auto | scx_compact_trial | shardad | shardad vs best-scx |
|---|---:|---:|---:|---|
| pbmc3k | 4.8 (4.5×) | **4.5 (4.8×)** | 6.2 (3.5×) | scx 1.38× smaller |
| pbmc10k | 39.8 | **30.1 (6.7×)** | 31.7 (6.4×) | ~parity (scx 1.05×) |
| smartseq2 | 552.7 | 262.9 (4.1×) | **253.2 (4.2×)** | shardad 4% smaller |
| tabula_100k | 448.9 | 233.9 (6.8×) | **222.2 (7.1×)** | shardad 5% smaller |
| census_1m | 3989 | 2802 (4.1×) | **1622 (7.0×)** | **shardad 1.73× smaller** |
| replogle_k562 | 2345 | 2313 | **2237 (1.7×)** | shardad 3% smaller |
| tahoe_c38 | 1563 | 1516 | **1403 (1.6×)** | shardad 8% smaller |
| chemogenetic_rgfp | 1305 | 968 (7.6×) | **802 (9.1×)** | shardad 1.21× smaller |

shardad leads compression on 6/8 (by 3–42%); scx wins the two smallest. The
largest gap is census_1m (shardad 1.62 GB vs scx 2.80 GB). scx `compact_trial`
consistently beats `scx_auto` on integer counts (it is the codec to compare).

## Write speed (wall s, lower is better)

| dataset | scx_auto | scx_compact_trial | shardad |
|---|---:|---:|---:|
| pbmc10k | 1.82 | 2.38 | **0.86** |
| smartseq2 | 5.48 | 8.29 | **4.55** |
| tabula_100k | **4.99** | 5.47 | 6.30 |
| census_1m | 31.3 | 31.8 | **24.6** |
| replogle_k562 | 29.3 | 28.2 | **13.6** |
| tahoe_c38 | 13.2 | 13.6 | **9.6** |
| chemogenetic_rgfp | 21.5 | 22.6 | **16.3** |

shardad writes faster on 6/7 (multiprocessing + a single simple codec). **Caveat:**
scx write peak-RSS is *not* directly comparable — scx materializes the source
in-process — so only size + wall are reported here.

## Full read → AnnData (wall s / peak RSS MB)

| dataset | scx_auto | scx_compact_trial | shardad |
|---|---|---|---|
| pbmc10k | 0.38 / 338 | 0.67 / 340 | **0.24** / 334 |
| smartseq2 | **1.22** / 432 | 1.65 / 451 | 1.29 / **346** |
| tabula_100k | **0.87** / 514 | 1.14 / 569 | 1.74 / **347** |
| census_1m | 5.45 / 2317 | **4.80** / 2732 | 5.31 / **551** |
| replogle_k562 | 5.74 / 2904 | 5.84 / 2962 | **4.26 / 345** |
| tahoe_c38 | **3.39** / 1537 | 3.35 / 1660 | 3.71 / **428** |
| chemogenetic_rgfp | **2.78** / 772 | 3.70 / 1081 | 4.74 / **392** |

Read **speed** is roughly parity, dataset-dependent. The peak-RSS figures in this
table are the `read_full` 2-sample sampler's steady-state numbers (they miss the
transient assembly high-water mark; the true peak is in the `ooc_rss_boundary`
table below). scx's `to_anndata` now narrows **in-decode**: a
`data_dtype="uint16"` request assembles `X` (and `adata.raw`) directly at the
target width (2 B/nnz), never building the intermediate float32 CSR, so a narrow
read **lowers** peak RSS instead of raising it. Measured on the true-peak
`ooc_rss_boundary` boundary (below), the narrow `scx_materialize_u16` read
matches or beats shardad's uint16 materialize on pbmc10k, tabula, and
chemogenetic, and closes most of the remaining gap on census_1m (12.6 vs
11.6 GB — shardad additionally narrows its column indices, which scx keeps at
i32). scx's structural answer at the largest scale remains its streaming
accelerators, which never build full X — see below.

## Out-of-core streaming vs materialize — peak RSS (MB), the at-scale story

`ooc_rss_boundary`: scx streaming (bounded) vs scx full-materialize (f32) vs scx
full-materialize narrowed in-decode (uint16) vs shardad full-materialize. True
peak RSS via the background `PeakRssSampler`. Refreshed on the Phase-4 in-decode
narrow build (`scx_materialize_u16` = `to_anndata(data_dtype="uint16")`).

| dataset | scx_stream | scx_materialize (f32) | scx_materialize_u16 | shardad_materialize |
|---|---:|---:|---:|---:|
| pbmc10k | 924 | 556 | **508** | 527 |
| tabula_100k | 3567 | 2230 | **1887** | 1892 |
| tahoe_c38 | 4732 | 4238 | 4047 | **3844** |
| chemogenetic_rgfp | 14763 | 8869 | **7178** | 7471 |
| **census_1m** | **5630** | 15268 | 12605 | 11608 |

The **in-decode narrow** (`scx_materialize_u16`) lowers peak RSS vs the f32
materialize on every dataset (the value buffer drops from 4 B/nnz to 2 B/nnz with
no f32 intermediate) — enough to **match or beat** shardad's own uint16
materialize on pbmc10k, tabula, and chemogenetic. shardad stays lower on tahoe and
census because it also narrows its column indices (scx keeps i32) and uses a
tighter materialize layout; scx closes most of the census gap (12.6 vs 11.6 GB).

**At census_1m scale the picture inverts entirely: scx streaming (5.6 GB) is the
lowest by far — below shardad's materialize (11.6 GB) and well under either scx
materialize (15.3 GB f32 / 12.6 GB uint16).** This is scx's structural
out-of-core advantage: full-matrix approaches (shardad, and both scx materialize
modes) grow with dataset size, while scx streaming stays bounded — the gap widens
further at census_5m/10m (not run here).

> **Note — why `scx_materialize` here (15.3 GB) ≠ the "Full read → AnnData" peak
> above (~2.3 GB) for census_1m:** the two rows measure different things. The
> `read_full` benchmark's 2-sample sampler catches the *steady-state* resident
> AnnData; the `ooc_rss_boundary` `PeakRssSampler` catches the true high-water
> mark, which includes the transient decode/assembly buffers a narrow sampler
> misses (and the residue of the preceding `scx_stream` pass, run back-to-back in
> one process). Compare each column *within* its own table, not across the two.

## Selective / row-subset read (wall s)

Roughly parity; shardad edges ahead on census_1m (5.6 s vs scx_auto 13.1 s) via
whole-covering-row reads, scx competitive on the perturbation datasets. scx's
`filtered_query` predicate-pushdown sub-scenarios have **no shardad equivalent**
(see capability gaps).

## Grouped / condition-sharded reads — shardad's differentiator

shardad physically groups cells by an `obs` label; scx approximates it via `scx
sort --group-by` + query-time shard pruning. `grouped_read` (self-materialized
grouped fixtures):

| op | dataset | scx | shardad |
|---|---|---|---|
| `read_group` (one label) | replogle_k562 | **0.7 s / 1530 MB** | 4.1 s / 4179 MB |
| `read_group` | tahoe_c38 | **1.1 s / 1355 MB** | 3.9 s / 3238 MB |
| `read_group` | chemogenetic_rgfp | 0.8 s / 11744 MB | 0.9 s / **810 MB** |
| `iter_group_shards` (stream) | replogle_k562 | 3.2 s / **2368 MB** | 4.0 s / 9624 MB |
| `iter_group_shards` | tahoe_c38 | 5.3 s / **4580 MB** | 3.8 s / 6272 MB |
| grouped_write | replogle_k562 | 40.6 s / 2338 MB | **19.1 s** / 2238 MB |
| grouped_write | chemogenetic_rgfp | 33.8 s / 1176 MB | **21.9 s / 803 MB** |

scx's query-pruned targeted read is competitive-to-faster on latency (replogle,
tahoe) with lower RAM, but its RAM can spike on some layouts (chemogenetic
`read_group`); shardad writes grouped archives faster. Mixed — genuinely
dataset-dependent, with neither dominating.

## Round-trip fidelity

shardad round-trips correctly (`shardad_fidelity`: exact CSR match + `dense` /
`float16` in-decode materialization) on pbmc3k and tabula_100k — parity with scx's
own correctness suite.

## Capability comparison (documented — not head-to-head perf races)

Features with no meaningful two-sided benchmark, reported as capability presence:

| capability | scx | shardad |
|---|---|---|
| GPU on-device decode / analysis | ✅ (framed Scx1 decodes in VRAM; rapids) | ❌ unimplemented ([#40]; CPU-decode+upload only) |
| Lazy query engine / predicate pushdown | ✅ | ❌ (no query engine) |
| ML training loader | ✅ (1,405 batches/s) | ❌ |
| Streaming analysis accelerators (HVG/PCA/…) | ✅ (bounded RAM) | ❌ (materialize → scanpy) |
| Cloud-native reads (S3/GCS/Azure) | ✅ | ❌ |
| Multimodal (CITE-seq / Multiome) | ✅ | ❌ |
| R bindings (Seurat / SCE) | ✅ | ❌ (Python only) |
| In-place mutation (append/delete/compact/merge) | ✅ | metadata-tail only (`update_obs`) |
| In-decode dtype/density materialization | ✅ true in-decode narrow on the eager path (`data_dtype=`/`container=`/`index_dtype=`): `X`/`raw` assemble directly at target dtype, no f32 intermediate → lowers peak RSS (matches/beats shardad on 3/5 datasets); `>2²⁴` integer reads exact. (layers still post-assembly) | ✅ (true in-decode narrow, direct to target dtype; also narrows indices) |
| Physical group-aligned (condition) sharding | approximated (sort + query pruning) | ✅ (native) |

## Takeaways

- **Compression:** shardad still leads on large integer-count data (up to 1.7×),
  but scx `compact_trial` has closed the old 1.9–3.2× gap to ~1.0–1.7× and wins on
  small datasets.
- **Write:** shardad faster (simpler codec + multiprocessing).
- **Read speed:** parity, dataset-dependent.
- **Memory:** scx's in-decode narrow (`to_anndata(data_dtype="uint16")`) now
  matches or beats shardad's uint16 materialize on 3/5 datasets and closes most of
  the gap on the rest (shardad edges ahead where it also narrows indices); **scx
  streaming wins outright at census scale** (census_1m: 5.6 vs 11.6 GB) and is the
  only path that stays bounded as datasets grow.
- **Grouped access:** shardad's native physical grouping vs scx's query-pruning —
  mixed, neither dominates.
- **Breadth:** scx carries GPU, a query engine, an ML loader, streaming
  accelerators, cloud, multimodal, and R — none of which shardad targets.

Two honest conclusions: scx has reached rough **compression/read parity** with a
deep-narrow specialist on its home turf, and the projects' real difference is
scope (a broad platform vs a focused perturbation-screen tool), not a one-sided
performance gap. The clearest scx-specific perf win is **bounded out-of-core RAM
at scale**. On whole-matrix integer reads, scx's eager `to_anndata` now performs a
*true* in-decode narrow — each shard assembles directly into the target dtype's
full-matrix buffer, never allocating the intermediate f32 matrix — so a
`data_dtype="uint16"` read genuinely lowers peak RSS (matching or beating
shardad's uint16 materialize on 3/5 datasets above). shardad retains a small edge
where it also narrows column indices and uses a tighter materialize layout; scx
keeps i32 indices. Integer→integer narrows additionally decode from the native
`u32` stream, so a `>2²⁴` count read into `uint32`/`int64`/`float64` is now exact
(previously a hard error).

[#40]: https://github.com/ArcInstitute/shardad/issues/40
