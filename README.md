# SCX — Sparse Cell eXpression System

A purpose-built binary file format, compression codec, query engine, and ML data loader for single-cell RNA-seq data. Replaces AnnData/h5ad with a unified Rust-native stack.

**Phase 1 (Format + Codec + AnnData Bridge) is feature-complete.** The full pipeline works:

```
h5ad -> scx convert -> scx.open().to_anndata() -> scanpy works
```

## Repository Structure

```
scx/
├── Cargo.toml              # workspace root (5 members)
├── scx-format/             # file layout, header, catalog, shard I/O, provenance
├── scx-codec/              # compression codecs (Rice, FOR-BP, Delta-Golomb, Zstd)
├── scx-sparse/             # CSR operations (scipy-compatible dtypes)
├── scx-cli/                # CLI tool (convert, info, validate)
├── pyscx/                  # Python bindings (PyO3 + maturin)
│   └── tests/              # Python test suite
├── tests/                  # integration tests + reference files
├── benchmarks/             # benchmark scripts + dataset downloader
└── docs/                   # additional documentation
```

## Key Documents

- **SPEC.md** — Full format specification (v0.5). Authoritative reference for binary layouts, codecs, and section types.
- **ROADMAP.md** — Phased implementation plan (Phases 1-4).
- **Phase1.md** — Phase 1 implementation plan (all tasks complete).
- **docs/api.md** — API reference for Rust, Python, and CLI interfaces.
- **docs/testing.md** — Test infrastructure and benchmarks.

## Quick Start

### Build and Test (Rust)

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --check
```

### Python Bindings

**Always use the uv venv** at `.venv/` for all Python work. Do NOT use system Python or pip directly.

```bash
# First time setup:
uv venv .venv
uv pip install maturin numpy scipy pyarrow anndata pytest scanpy igraph leidenalg

# Build and test:
cd pyscx && ../.venv/bin/maturin develop && ../.venv/bin/pytest tests/ -v
```

```bash
# Or prefix commands with the venv path (no activation needed):
.venv/bin/python ...
.venv/bin/maturin develop
.venv/bin/pytest tests/ -v
uv pip install <package>       # uv auto-detects the .venv
```

## Phase 1 Go/No-Go Gate

These criteria must be met before proceeding to Phase 2:

1. h5ad -> scx -> h5ad round-trip is bit-exact for integer counts
2. SCX file < 60% the size of h5ad for typical datasets
3. `scx.open().to_anndata()` -> full scanpy pipeline (QC -> PCA -> Leiden -> DE) works

Test coverage: `pyscx/tests/test_go_no_go.py`

## License

MIT
