# SCX — Sparse Cell eXpression System

A purpose-built binary file format, compression codec, query engine, and ML data loader for single-cell RNA-seq data. Replaces AnnData/h5ad with a unified Rust-native stack.

**Phase 1 (Format + Codec + AnnData Bridge) is complete.** Phase 2 (Training Loader + Query Engine + Cloud Ops) is complete. The full pipeline works:

```
h5ad -> scx convert -> scx.open().to_anndata() -> scanpy works
```

Plus: lazy query engine, ML training loader, file operations (append/delete/compact/merge), and cloud access (push/pull/explode/pack).

## Repository Structure

```
scx/
├── Cargo.toml              # workspace root (9 members)
├── scx-format/             # file layout, header, catalog, shard I/O, provenance
├── scx-codec/              # compression codecs (Rice, FOR-BP, Delta-Golomb, Zstd)
├── scx-sparse/             # CSR operations (scipy-compatible dtypes)
├── scx-ops/                # file operations (append, delete, compact, merge, rollback)
├── scx-engine/             # lazy query engine (predicate pushdown, projection, fused ops)
├── scx-loader/             # ML training data loader (triple-buffered: tokio I/O → rayon decode → GPU)
├── scx-cloud/              # cloud access (push, pull, explode, pack, cloud-optimize, CloudReader)
├── scx-cli/                # CLI tool (convert, info, validate, query, append, delete, compact, merge, ...)
├── pyscx/                  # Python bindings (PyO3 + maturin)
│   └── tests/              # Python test suite (15 test files)
├── tests/                  # integration tests + reference files
├── benchmarks/             # benchmark scripts + dataset downloader
│   ├── scripts/            # 20 benchmark/helper scripts
│   └── results/            # 25 benchmark result reports
└── docs/                   # additional documentation
```

## Key Documents

- **SPEC.md** — Full format specification (v0.5). Authoritative reference for binary layouts, codecs, and section types.
- **ROADMAP.md** — Phased implementation plan (Phases 1-4).
- **Phase1.md** — Phase 1 implementation plan (all tasks complete).
- **Phase2.md** — Phase 2 implementation plan (Steps 1-5 complete).
- **Phase2-CLOUD.md** — Phase 2 cloud operations specification (implemented).
- **Phase3.md** — Phase 3 specification (GPU, R bindings, multimodal — future work).
- **docs/api.md** — API reference for Rust, Python, and CLI interfaces.
- **docs/architecture.md** — Crate architecture and data flow diagrams.
- **docs/testing.md** — Test infrastructure and benchmarks.

## Quick Start

### Build and Test (Rust)

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --check

# With cloud features:
cargo test --workspace --features cloud
```

### Python Bindings

**Always use the uv venv** at `.venv/` for all Python work. Do NOT use system Python or pip directly.

```bash
# First time setup:
uv venv .venv
uv pip install maturin numpy scipy pyarrow anndata pytest scanpy igraph leidenalg

# Build and test:
cd pyscx && ../.venv/bin/maturin develop && ../.venv/bin/pytest tests/ -v

# With cloud support:
cd pyscx && ../.venv/bin/maturin develop --features cloud && ../.venv/bin/pytest tests/ -v
```

```bash
# Or prefix commands with the venv path (no activation needed):
.venv/bin/python ...
.venv/bin/maturin develop
.venv/bin/pytest tests/ -v
uv pip install <package>       # uv auto-detects the .venv
```

## Phase 2 Summary

### Training Loader (`scx-loader`)
Triple-buffered Rust pipeline for GPU-saturating data loading:
- Stage 1: tokio async I/O reads shard groups
- Stage 2: rayon thread pool shuffles, projects HVGs, densifies, normalizes
- Stage 3: Python `TrainingDataset` iterator hands dense batches to PyTorch

### Query Engine (`scx-engine`)
Lazy pipeline: `open → filter → select → normalize → collect`:
- Two-level predicate pushdown (catalog shard pruning + index row pruning)
- Gene projection, fused normalize+log1p
- Parallel shard decode via rayon

### File Operations (`scx-ops`)
- `append` — add cells without rewriting
- `delete` — logical deletion via Roaring Bitmap
- `compact` — reclaim space from deleted data
- `merge` — streaming merge of multiple files
- `rollback` — revert to previous manifest

### Cloud Access (`scx-cloud`)
- `push`/`pull` — streaming transfer between local and cloud (S3, GCS, Azure)
- `explode`/`pack` — packed `.scx` ↔ exploded `.scxd` directory
- `cloud-optimize` — front-of-file catalog for single-read cloud opens
- `CloudReader` — direct cloud reads without full download

## Go/No-Go Gates

### Phase 1 — PASSED
- [x] h5ad → scx → h5ad round-trip is bit-exact for integer counts
- [x] SCX file < 60% the size of h5ad for typical datasets
- [x] `scx.open().to_anndata()` → full scanpy pipeline (QC → PCA → Leiden → DE) works

### Phase 2 — PASSED
- [x] Predicate pushdown skips >50% of shards on filtered queries (55.1% average)
- [x] Training loader benchmarked with published throughput results

## License

MIT
