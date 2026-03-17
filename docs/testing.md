# SCX Testing & Benchmarks

## Rust Integration Tests

**Location**: `scx-format/tests/integration.rs`

- Round-trip validation (write -> read -> verify)
- Minimum file size checks
- Codec round-trip tests (per-crate unit tests in each codec module)

Run: `cargo test --workspace`

## Python Test Suite

**Location**: `pyscx/tests/`

| Test file | Purpose |
|-----------|---------|
| `test_round_trip.py` | AnnData round-trip validation |
| `test_scanpy_pipeline.py` | Full scanpy workflow (QC -> PCA -> Leiden -> DE) |
| `test_zero_copy.py` | Zero-copy verification (`np.shares_memory`) |
| `test_go_no_go.py` | Go/No-Go gate criteria validation |
| `test_metadata.py` | Metadata preservation tests |
| `conftest.py` | Shared pytest fixtures |

Run: `cd pyscx && ../.venv/bin/maturin develop && ../.venv/bin/pytest tests/ -v`

## Benchmarks

**Location**: `benchmarks/scripts/`

| Script | Purpose |
|--------|---------|
| `benchmark_compression.py` | Compression ratio vs h5ad (target: < 60%) |
| `benchmark_read.py` | Read performance (scx vs h5ad) |
| `benchmark_write.py` | h5ad -> scx conversion speed (MB/s) |
| `benchmark_all.py` | Runs all benchmarks together |
| `download_datasets.sh` | Fetches test data (PBMC 3K, Tabula Sapiens, CELLxGENE Census, Smart-seq2) |

### Running Benchmarks

```bash
# Download test datasets first:
bash benchmarks/scripts/download_datasets.sh

# Run all benchmarks:
.venv/bin/python benchmarks/scripts/benchmark_all.py

# Or run individually:
.venv/bin/python benchmarks/scripts/benchmark_compression.py
.venv/bin/python benchmarks/scripts/benchmark_read.py
.venv/bin/python benchmarks/scripts/benchmark_write.py
```
