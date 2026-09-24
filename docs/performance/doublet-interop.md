# Doublet-Caller Interop (`doublet_interop`)

> Part of [SCX performance](README.md). Usage is in
[docs/scanpy/external-annotations.md](../scanpy/external-annotations.md).

External doublet callers run in their own conda environments, their results are
imported to canonical obs columns, and everything is scored — against each
other and against labelled truth. Captured 2026-08-03 on a Chimera CPU node;
`scDblFinder` 1.24.0 (R, `rscx` env) and `Scrublet` via `sc.pp.scrublet`
(scanpy 1.12, `scx-bench` env). Solo is an optional comparator and is skipped
here — no environment on this cluster ships scvi-tools.

**The runs are bit-identical across repeat invocations** (all 19 emitted
metrics), because the injection seed, the tool seeds and scDblFinder's
`set.seed` carry end to end. That is what makes the science numbers below
gateable rather than indicative.

## Round-trip fidelity — the plumbing

The tool's own output compared with what comes back off the SCX file, **joined
by key**. A positional import passes every row-count and column-presence check
and fails only here.

| Dataset | Cells | Tool | Max score delta | Call disagreements | Matched |
|---|---:|---|---:|---:|---:|
| `pbmc3k` | 2,916 | scDblFinder | 5.0e-16 | 0 | 2,916 / 2,916 |
| `pbmc3k` | 2,916 | Scrublet | 7.2e-09 | 0 | 2,916 / 2,916 |
| `tabula_sapiens_100k` (3 donors) | 1,914 | scDblFinder | 5.0e-16 | 0 | 1,914 / 1,914 |
| `tabula_sapiens_100k` (3 donors) | 1,914 | Scrublet | 6.9e-09 | 0 | 1,914 / 1,914 |

scDblFinder's 5.0e-16 is f64→f32 rounding; Scrublet's 7e-09 is the same plus
its CSV text round trip.

## Accuracy against injected truth

Truth is **computational injection** — pairs of real cells summed, at an 8%
rate. This is `DOUBLET-DETECTION.md` **Category D**: exact labels, but summed
count vectors do not reproduce capture or ambient-RNA artifacts. Do not read
these as accuracy against real doublets, and do not pool them with
hashing- or genotype-labelled results.

| Dataset | Tool | AUROC | Precision | Recall | Recall (heterotypic) | Recall (homotypic) |
|---|---|---:|---:|---:|---:|---:|
| `pbmc3k` | scDblFinder | 0.979 | 0.780 | 0.690 | — | — |
| `pbmc3k` | Scrublet | 0.785 | 0.667 | 0.269 | — | — |
| `tabula_sapiens_100k` | scDblFinder | 0.912 | 0.821 | 0.451 | **0.484** | **0.188** |
| `tabula_sapiens_100k` | Scrublet | 0.841 | 0.750 | 0.254 | **0.286** | **0.000** |

The stratified columns are the point. Pooled recall says Scrublet finds a
quarter of the doublets; split by kind it finds **none of the homotypic ones**
on this fixture. `pbmc3k` has no `cell_type` column so its injections are
recorded as kind `unknown` rather than guessed — hence the dashes.

## Between-tool agreement

| Dataset | Raw agreement | Cohen's κ | Score Spearman | Consensus AUROC |
|---|---:|---:|---:|---:|
| `pbmc3k` | 0.953 | 0.482 | 0.280 | 0.842 |
| `tabula_sapiens_100k` | 0.966 | 0.459 | 0.369 | 0.744 |

Raw agreement is the number to distrust: at a ~7% doublet rate two tools that
both call almost nothing agree ~95% by construction. κ ≈ 0.47 is the honest
reading — moderate. Consensus AUROC sits *below* either tool alone because
`method="majority"` over two voters is an AND, and it ranks by an integer vote
count with heavy ties.

Per-donor called rates on the atlas slice show scDblFinder steady (3.6–4.5%
across three donors) and Scrublet spread 4× (1.1–4.8%) — the kind of
per-library miscalibration a pooled rate hides.

## Coverage caveat

The atlas run is the **first three donors** of `tabula_sapiens_100k`
(`THD0001`, `THD0002`, `THD0005` — 1,772 real cells), capped by
`max_batches=3`; **116 donors are dropped** and the result records them.
The cap is a runtime bound, not a sample: read these as a fixed regression
fixture, not as a characterisation of the atlas.
