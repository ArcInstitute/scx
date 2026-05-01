# SCX Training Loader Benchmark Report

**Generated**: 2026-05-01 15:35:39

## Test Environment

- **Host**: vci-steady-state-node-017
- **CPU**: Intel(R) Xeon(R) Platinum 8480+
- **RAM**: 1771.7 GB
- **GPUs**: NVIDIA H100 80GB HBM3, 81559 MiB
- **PyTorch**: 2.11.0+cu128
- **TileDB-SOMA**: missing

## Datasets

| Dataset | Cells | Genes | h5ad Size | SCX Size |
|---------|-------|-------|-----------|----------|
| pbmc3k | 2,700 | 32,738 | 21 MB | 4 MB |

## 1. SCX Throughput

| Dataset | Batches/sec | Cells/sec | Total Time | Batches |
|---------|-------------|-----------|------------|---------|
| pbmc3k | 89.4 | 80,466 | 0.0s | 3 |

## 2. Time to First Batch

| Dataset | Median | Min | Max | Target | Pass? |
|---------|--------|-----|-----|--------|-------|
| pbmc3k | 0.021s | 0.015s | 0.025s | <2.0s | PASS ✅ |

## 3. Memory Budget

| Dataset | Peak RSS | Budget | Within Budget? |
|---------|----------|--------|----------------|
| pbmc3k | 810 MB | 512 MB | No ❌ |

## 5. SOTA Comparison (batches/sec)

| Dataset | SCX | AnnData | TileDB-SOMA | scDataLoader | BPCells |
|---------|-----|---------|-------------|--------------|---------|
| pbmc3k | 89.4 | 12.6 | — | — | — |

## Summary — Go/No-Go Criteria

| Criterion | Result |
|-----------|--------|
| pbmc3k: time-to-first-batch <2s | PASS ✅ (0.021s) |

## Interpretation

The SCX training loader uses a triple-buffered pipeline architecture:
tokio I/O → rayon decode → Python consumer. Key performance factors:

- **HVG projection** at decode time avoids materializing full-width dense rows,
  providing significant speedup proportional to gene count reduction.
- **Fused normalize+log1p** eliminates a second pass over the dense matrix.
- **Memory budget** is enforced by auto-tuning shard_group_size and prefetch_batches.
  Peak RSS should be roughly constant regardless of dataset size.
- **GPU utilization** depends on the balance between loader throughput and model compute.
  With a lightweight simulated forward pass, GPU util may be limited by data transfer;
  real models with heavier compute will see higher utilization.
- **TileDB-SOMA-ML** comparison is the primary Go/No-Go criterion. SCX's advantage
  comes from the compressed format, decode-time projection, and fused operations.
