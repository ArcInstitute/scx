# SCX Documentation Index

This directory contains the design and reference docs for SCX (Sparse Cell
eXpression System). Use the index below to jump to the topic you need.

If you're new here, start with [architecture.md](architecture.md) for the
big picture, then [api.md](api.md) (Rust/CLI/Python APIs) or
[scanpy.md](scanpy.md) (using SCX from a scanpy workflow).

## User guides

- [scanpy.md](scanpy.md) — Scanpy integration guide and Rust-native accelerator usage (PCA, kNN, UMAP, DE, Leiden, Harmony2, LISI, HVG, pseudobulk).
- [gpu-setup.md](gpu-setup.md) — GPU setup: CUDA, RAPIDS, conda, containers, SLURM, troubleshooting.
- [cloud.md](cloud.md) — Cloud auth, layouts, tuning, provider-specific notes, and cloud-native query.
- [multimodal.md](multimodal.md) — CITE-seq / 10x Multiome / TEA-seq / spatial layout and APIs.
- [operations.md](operations.md) — Behavior of mutating ops (append, delete, compact, merge, subset) and their effect on metadata and CSC sidecars.
- [compatibility-matrix.md](compatibility-matrix.md) — Tested vs. declared Python / numpy / scipy / pyarrow / anndata / scanpy combos for `pyscx`.

## Architecture & design

- [architecture.md](architecture.md) — Crate graph, feature flags, file format overview, codec system, data model, reader/writer architecture.
- [multithreading.md](multithreading.md) — Multithreading architecture across crates (tokio I/O, rayon CPU, bounded channels).
- [sharding.md](sharding.md) — Sharding design, CSR/CSC layouts, and per-shard sidecar files.
- [conventions.md](conventions.md) — Coding conventions: serialization, error handling, checksums, language binding rules, GPU/accel constraints.

## Format & codec specification

- [format.md](format.md) — Binary format reference: file header, catalogs, CSR shard layout, fragment/manifest model, checksums, detection bitmap sidecars.
- [codec.md](codec.md) — Bit-level codec spec: Delta-Golomb-Rice, FOR-BP, Rice, LZ4+shuffle, auto-selection rules.

## API reference

- [api.md](api.md) — Full API reference: section types, Rust/CLI/Python entry points, conversion warnings, memory budgets, lazy/backed datasets.
- [python_api.rst](python_api.rst) — Sphinx autodoc entry point for the `pyscx` Python package.

## Performance & testing

- [performance.md](performance.md) — Benchmark results, memory and throughput figures, per-op speedups.
- [benchmark_manifest.md](benchmark_manifest.md) — Manifest format for benchmark result JSON files under `benchmarks/comprehensive/results/`.
- [testing.md](testing.md) — Test matrix, benchmark harness, and correctness validation details.

## Developer setup

- [development.md](development.md) — Build guide: CPU-only, HDF5, cloud, GPU, Python, R builds; test matrix; fuzzing.

## Sphinx site

- [index.md](index.md) — Top page for the rendered Sphinx site (groups the docs above into User Guide / Architecture / Format / API / Performance).
- [conf.py](conf.py), [requirements.txt](requirements.txt), [_static/](_static/), [_templates/](_templates/) — Sphinx build configuration and assets.

## Related top-level docs

- [../README.md](../README.md) — Project README.
- [../ROADMAP.md](../ROADMAP.md) — Capability tiers and phase status.
- [../CLAUDE.md](../CLAUDE.md) — Agent-facing project overview with links into the docs above.
- [../benchmarks/README.md](../benchmarks/README.md) — Practical benchmark guide: SLURM submission, dataset prep, regression gating.
