#!/usr/bin/env python3
"""
SCX GPU Utilization Benchmark — scVI Training Loop

Phase 2 Go/No-Go criterion: GPU utilization >85% during scVI training
on a large-cell SCX dataset.

Measures GPU utilization via nvidia-smi dmon while running a real scVI
(or scVI-equivalent VAE) training loop with the SCX loader.

Usage:
    # Smoke test (pbmc3k, 1 epoch)
    python benchmarks/scripts/benchmark_gpu_scvi.py --dataset pbmc3k --epochs 1 --smoke

    # Full benchmark (census_1m, 3 epochs)
    python benchmarks/scripts/benchmark_gpu_scvi.py --dataset census_1m --epochs 3

    # With explicit GPU device
    CUDA_VISIBLE_DEVICES=0 python benchmarks/scripts/benchmark_gpu_scvi.py --dataset census_1m
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_release import ensure_release_build

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
from bench_env import WORK_DIR

DATASETS = {
    "pbmc3k": {
        "scx": WORK_DIR / "pbmc3k.scx",
        "cells": 2700,
        "genes": 32738,
    },
    "tabula_sapiens_100k": {
        "scx": WORK_DIR / "tabula_sapiens_100k.scx",
        "cells": 100000,
        "genes": 60000,
    },
    "census_1m": {
        "scx": WORK_DIR / "census_1m.scx",
        "cells": 1000000,
        "genes": 61497,
    },
    "census_10m_blood": {
        "scx": WORK_DIR / "census_10m_blood.scx",
        "cells": 10000000,
        "genes": 60000,
    },
}

# ---------------------------------------------------------------------------
# VAE model (scVI-equivalent)
# ---------------------------------------------------------------------------

def build_scvi_vae(n_input: int, n_latent: int = 128, n_hidden: int = 128,
                   n_layers: int = 2, dropout_rate: float = 0.1):
    """Build a VAE matching scVI's architecture using raw PyTorch.

    If scvi-tools is available, we use it directly. Otherwise we build
    an equivalent encoder→reparameterize→decoder→ELBO model.

    Returns (model, use_scvi: bool).
    """
    import torch
    import torch.nn as nn

    class ScviVAE(nn.Module):
        """Minimal scVI-equivalent VAE for benchmarking.

        Architecture matches scvi-tools LDVAE:
        - Encoder: n_layers FC layers with BatchNorm + ReLU + Dropout → μ, σ
        - Decoder: n_layers FC layers with BatchNorm + ReLU + Dropout → rate
        - Loss: negative ELBO (reconstruction + KL divergence)
        """

        def __init__(self, n_input, n_latent, n_hidden, n_layers, dropout_rate):
            super().__init__()

            # Encoder
            enc_layers = []
            in_dim = n_input
            for _ in range(n_layers):
                enc_layers.extend([
                    nn.Linear(in_dim, n_hidden),
                    nn.BatchNorm1d(n_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ])
                in_dim = n_hidden
            self.encoder = nn.Sequential(*enc_layers)
            self.z_mean = nn.Linear(n_hidden, n_latent)
            self.z_var = nn.Linear(n_hidden, n_latent)

            # Decoder
            dec_layers = []
            in_dim = n_latent
            for _ in range(n_layers):
                dec_layers.extend([
                    nn.Linear(in_dim, n_hidden),
                    nn.BatchNorm1d(n_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ])
                in_dim = n_hidden
            self.decoder = nn.Sequential(*dec_layers)
            self.px_rate = nn.Linear(n_hidden, n_input)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            # Encode
            h = self.encoder(x)
            mu = self.z_mean(h)
            logvar = self.z_var(h)

            # Reparameterize
            z = self.reparameterize(mu, logvar)

            # Decode
            h_dec = self.decoder(z)
            px_rate = torch.exp(self.px_rate(h_dec))  # Poisson rate

            # ELBO loss
            # Reconstruction: Poisson NLL
            recon_loss = torch.mean(px_rate - x * torch.log(px_rate + 1e-8))
            # KL divergence
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            return recon_loss + kl_loss

    model = ScviVAE(n_input, n_latent, n_hidden, n_layers, dropout_rate)
    return model


# ---------------------------------------------------------------------------
# GPU monitoring
# ---------------------------------------------------------------------------

class GpuMonitor:
    """Monitor GPU utilization via nvidia-smi dmon."""

    def __init__(self, gpu_id: int = 0, interval_s: int = 1):
        self.gpu_id = gpu_id
        self.interval_s = interval_s
        self.log_path = Path(f"/tmp/scx_gpu_monitor_{os.getpid()}.csv")
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen(
            ["nvidia-smi", "dmon",
             "-i", str(self.gpu_id),
             "-s", "u",      # utilization
             "-d", str(self.interval_s),
             "-f", str(self.log_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> list[int]:
        """Stop monitoring and return GPU utilization samples."""
        if self.proc:
            self.proc.terminate()
            self.proc.wait()
            self.proc = None

        utils = []
        if self.log_path.exists():
            for line in self.log_path.read_text().splitlines():
                parts = line.split()
                # nvidia-smi dmon format: gpu_idx sm_util mem_util ...
                if len(parts) >= 2 and parts[0].isdigit():
                    try:
                        utils.append(int(parts[1]))  # sm utilization %
                    except ValueError:
                        pass
            self.log_path.unlink(missing_ok=True)

        return utils


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_gpu_benchmark(dataset_name: str, n_epochs: int = 3,
                      batch_size: int = 1024, n_hvg: int = 2000,
                      warmup_epochs: int = 1, smoke: bool = False) -> dict:
    """Run scVI-equivalent training loop and measure GPU utilization."""
    import torch
    import pyscx

    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    ds_info = DATASETS[dataset_name]
    scx_path = ds_info["scx"]
    if not scx_path.exists():
        return {"error": f"SCX file not found: {scx_path}"}

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU: {gpu_name}")
    print(f"  Dataset: {dataset_name} ({ds_info['cells']:,} cells)")
    print(f"  Config: batch_size={batch_size}, n_hvg={n_hvg}, epochs={n_epochs}")

    # Build model
    n_input = n_hvg if n_hvg else ds_info["genes"]
    model = build_scvi_vae(n_input=n_input, n_latent=128, n_hidden=128, n_layers=2)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # Build SCX loader
    hvg_indices = list(range(n_hvg)) if n_hvg else None
    ds = pyscx.TrainingDataset(
        str(scx_path),
        batch_size=batch_size,
        hvg_indices=hvg_indices,
        normalize=True,
        log1p=True,
        seed=42,
    )
    print(f"  Loader: n_obs={ds.n_obs}, n_output_genes={ds.n_output_genes}")

    # Warmup (no monitoring)
    print(f"\n  Warmup ({warmup_epochs} epoch(s))...")
    for epoch in range(warmup_epochs):
        n = 0
        for batch in ds:
            X = torch.from_numpy(batch["X"]).to(device, non_blocking=True)
            loss = model(X)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n += 1
        print(f"    epoch {epoch}: {n} batches")

    # Timed + monitored run
    print(f"\n  Benchmark ({n_epochs} epoch(s))...")
    monitor = GpuMonitor(gpu_id=0, interval_s=1)
    monitor.start()

    # Small delay to let nvidia-smi start sampling
    time.sleep(1)

    epoch_stats = []
    total_batches = 0
    total_loss_sum = 0.0

    t_start = time.perf_counter()

    for epoch in range(n_epochs):
        epoch_batches = 0
        epoch_loss = 0.0
        t_epoch = time.perf_counter()

        for batch in ds:
            X = torch.from_numpy(batch["X"]).to(device, non_blocking=True)
            loss = model(X)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            epoch_batches += 1
            epoch_loss += loss.item()

        epoch_time = time.perf_counter() - t_epoch
        avg_loss = epoch_loss / epoch_batches if epoch_batches else 0
        bps = epoch_batches / epoch_time if epoch_time > 0 else 0

        epoch_stats.append({
            "epoch": warmup_epochs + epoch,
            "batches": epoch_batches,
            "time_s": round(epoch_time, 3),
            "batches_per_sec": round(bps, 1),
            "avg_loss": round(avg_loss, 4),
        })
        total_batches += epoch_batches
        total_loss_sum += epoch_loss
        print(f"    epoch {warmup_epochs + epoch}: {epoch_batches} batches, "
              f"{bps:.1f} b/s, loss={avg_loss:.4f}, {epoch_time:.1f}s")

    total_time = time.perf_counter() - t_start

    # Wait for final GPU samples
    time.sleep(2)
    gpu_utils = monitor.stop()

    # Compute stats
    if gpu_utils:
        # Skip first and last samples (startup/shutdown noise)
        trimmed = gpu_utils[1:-1] if len(gpu_utils) > 2 else gpu_utils
        avg_util = sum(trimmed) / len(trimmed) if trimmed else 0
        sorted_utils = sorted(trimmed)
        median_util = sorted_utils[len(sorted_utils) // 2] if sorted_utils else 0
        p5_util = sorted_utils[max(0, int(len(sorted_utils) * 0.05))] if sorted_utils else 0
        p95_util = sorted_utils[min(len(sorted_utils) - 1, int(len(sorted_utils) * 0.95))] if sorted_utils else 0
    else:
        avg_util = median_util = p5_util = p95_util = 0

    overall_bps = total_batches / total_time if total_time > 0 else 0

    result = {
        "benchmark": "gpu_scvi",
        "dataset": dataset_name,
        "n_cells": ds_info["cells"],
        "gpu": gpu_name,
        "model": "scVI-equivalent VAE (2L, 128h, 128z)",
        "n_params": n_params,
        "batch_size": batch_size,
        "n_hvg": n_hvg,
        "warmup_epochs": warmup_epochs,
        "measured_epochs": n_epochs,
        "total_batches": total_batches,
        "total_time_s": round(total_time, 3),
        "overall_batches_per_sec": round(overall_bps, 1),
        "epoch_stats": epoch_stats,
        "gpu_utilization": {
            "n_samples": len(gpu_utils),
            "avg_pct": round(avg_util, 1),
            "median_pct": round(median_util, 1),
            "p5_pct": round(p5_util, 1),
            "p95_pct": round(p95_util, 1),
            "raw_samples": gpu_utils,
        },
        "target_util_pct": 85,
        "pass": avg_util >= 85,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Cleanup
    del model, optimizer, ds
    torch.cuda.empty_cache()
    gc.collect()

    return result


def generate_report(result: dict) -> str:
    """Generate markdown report from benchmark results."""
    lines = [
        "# SCX GPU Utilization Benchmark — scVI Training",
        "",
        f"**Generated**: {result.get('timestamp', 'N/A')}",
        "",
        "## Test Environment",
        "",
        f"- **GPU**: {result.get('gpu', 'N/A')}",
        f"- **Model**: {result.get('model', 'N/A')} ({result.get('n_params', 0):,} params)",
        f"- **Dataset**: {result.get('dataset', 'N/A')} ({result.get('n_cells', 0):,} cells)",
        f"- **Config**: batch_size={result.get('batch_size')}, n_hvg={result.get('n_hvg')}",
        "",
        "## GPU Utilization",
        "",
        "| Metric | Value | Target | Pass? |",
        "|--------|-------|--------|-------|",
    ]

    gpu = result.get("gpu_utilization", {})
    avg = gpu.get("avg_pct", 0)
    passed = "PASS ✅" if result.get("pass") else "FAIL ❌"
    lines.append(f"| Avg GPU Util | {avg:.1f}% | ≥85% | {passed} |")
    lines.append(f"| Median GPU Util | {gpu.get('median_pct', 0):.1f}% | — | — |")
    lines.append(f"| P5 GPU Util | {gpu.get('p5_pct', 0):.1f}% | — | — |")
    lines.append(f"| P95 GPU Util | {gpu.get('p95_pct', 0):.1f}% | — | — |")
    lines.append(f"| Samples | {gpu.get('n_samples', 0)} | — | — |")
    lines.append("")

    lines.extend([
        "## Training Throughput",
        "",
        "| Epoch | Batches | Time (s) | Batches/sec | Avg Loss |",
        "|-------|---------|----------|-------------|----------|",
    ])
    for es in result.get("epoch_stats", []):
        lines.append(
            f"| {es['epoch']} | {es['batches']} | {es['time_s']:.1f} "
            f"| {es['batches_per_sec']:.1f} | {es['avg_loss']:.4f} |"
        )
    lines.append("")

    overall_bps = result.get("overall_batches_per_sec", 0)
    lines.extend([
        f"**Overall**: {result.get('total_batches', 0)} batches in "
        f"{result.get('total_time_s', 0):.1f}s = **{overall_bps:.1f} batches/sec**",
        "",
        "## Go/No-Go Verdict",
        "",
        f"**GPU utilization**: {avg:.1f}% (target ≥85%): **{passed}**",
        "",
        "## Interpretation",
        "",
        "GPU utilization during scVI-equivalent VAE training measures whether",
        "the SCX loader's triple-buffered pipeline (tokio I/O → rayon decode →",
        "Python) can keep the GPU fed with data faster than the model processes it.",
        "",
        "Key factors affecting GPU utilization:",
        "- **Batch size**: Larger batches amortize GPU kernel launch overhead",
        "- **HVG projection**: Reducing genes from 60K→2K reduces data transfer",
        "- **Model complexity**: Lightweight models expose loader bottlenecks;",
        "  heavier models (more layers/units) increase GPU compute time",
        "- **Dataset size**: Larger datasets yield more batches per epoch,",
        "  giving the pipeline more time in steady state",
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SCX GPU Utilization Benchmark (scVI training)")
    parser.add_argument("--dataset", default="census_1m",
                        choices=list(DATASETS.keys()),
                        help="Dataset to benchmark (default: census_1m)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Measured epochs (default: 3)")
    parser.add_argument("--warmup-epochs", type=int, default=1,
                        help="Warmup epochs before measurement (default: 1)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Mini-batch size (default: 1024)")
    parser.add_argument("--n-hvg", type=int, default=2000,
                        help="HVG genes to project (default: 2000)")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test (1 epoch, no warmup)")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1
        args.warmup_epochs = 0

    print("=" * 60)
    print("SCX GPU Utilization Benchmark — scVI Training")
    print("=" * 60)

    result = run_gpu_benchmark(
        dataset_name=args.dataset,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        n_hvg=args.n_hvg,
        warmup_epochs=args.warmup_epochs,
        smoke=args.smoke,
    )

    if "error" in result:
        print(f"\nERROR: {result['error']}")
        sys.exit(1)

    # Print summary
    gpu = result["gpu_utilization"]
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"  Avg GPU util:   {gpu['avg_pct']:.1f}%")
    print(f"  Median GPU util: {gpu['median_pct']:.1f}%")
    print(f"  Batches/sec:    {result['overall_batches_per_sec']:.1f}")
    print("  Target:         ≥85%")
    print(f"  Verdict:        {'PASS ✅' if result['pass'] else 'FAIL ❌'}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "gpu_scvi_benchmark.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"\n  JSON: {json_path}")

    report = generate_report(result)
    md_path = RESULTS_DIR / "gpu_scvi_benchmark.md"
    md_path.write_text(report)
    print(f"  Report: {md_path}")


if __name__ == "__main__":
    ensure_release_build()
    main()
