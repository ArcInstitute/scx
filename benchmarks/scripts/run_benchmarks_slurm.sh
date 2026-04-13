#!/usr/bin/env bash
# Submit comprehensive SCX benchmarks via SLURM.
# Usage: bash benchmarks/scripts/run_benchmarks_slurm.sh
#
# Submits 3 jobs:
#   1. A-C: Compression + Write + Read (cpu_preemptible partition, 80GB, ~2 hours)
#   2. D-E: Ops + Query engine (cpu_preemptible partition, 16GB, ~5 min, Rust tests)
#   3. F:   ML Data Loader + SOTA baselines (cpu_preemptible partition, 80GB, ~1 hour)

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${REPO}/.venv"
PYTHON="${VENV}/bin/python"

# Load environment from .env if present
if [ -f "${REPO}/.env" ]; then
    set -a; source "${REPO}/.env"; set +a
fi
if [ -z "${SCX_WORK_DIR:-}" ]; then
    echo "ERROR: SCX_WORK_DIR is not set. Define it in ${REPO}/.env or export it."
    exit 1
fi
WORK_DIR="${SCX_WORK_DIR}"
DATA_DIR="${SCX_DATA_DIR:-${WORK_DIR}/benchmarks/datasets}"
RESULTS_DIR="${REPO}/benchmarks/results"
LOGS_DIR="${REPO}/benchmarks/logs"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

echo "=== SCX Comprehensive Benchmark Submission ==="
echo "Repo: $REPO"
echo "Data: $DATA_DIR"
echo ""

# --------------------------------------------------------------------------
# Job 1: Compression + Write + Read benchmarks (A-C)
# --------------------------------------------------------------------------
JOB1_SCRIPT="${LOGS_DIR}/bench_abc.sh"
cat > "$JOB1_SCRIPT" << 'EOFABC'
#!/usr/bin/env bash
set -euo pipefail

REPO="__REPO__"
VENV="${REPO}/.venv"
PYTHON="${VENV}/bin/python"
DATA_DIR="__DATA_DIR__"
RESULTS_DIR="${REPO}/benchmarks/results"

cd "$REPO"
export SCX_DATA_DIR="$DATA_DIR"

echo "=== Building release ==="
cargo build --release --workspace
cd pyscx && "${VENV}/bin/maturin" develop --release && cd ..

echo ""
echo "=== A-C: Compression, Write, Read Benchmarks ==="
"$PYTHON" benchmarks/scripts/benchmark_all.py 2>&1 | tee "${RESULTS_DIR}/comprehensive_benchmark_log.txt"

echo ""
echo "=== DONE ==="
EOFABC
sed -i "s|__REPO__|${REPO}|g; s|__DATA_DIR__|${DATA_DIR}|g" "$JOB1_SCRIPT"
chmod +x "$JOB1_SCRIPT"

JOB1_ID=$(sbatch \
    --job-name=scx-bench-abc \
    --partition=cpu_preemptible \
    --cpus-per-task=16 \
    --mem=80G \
    --time=04:00:00 \
    --output="${LOGS_DIR}/bench_abc_%j.out" \
    --error="${LOGS_DIR}/bench_abc_%j.err" \
    "$JOB1_SCRIPT" | awk '{print $NF}')
echo "[Job 1] A-C benchmarks submitted: $JOB1_ID"

# --------------------------------------------------------------------------
# Job 2: Ops + Query Engine benchmarks (D-E) — Rust tests
# --------------------------------------------------------------------------
JOB2_SCRIPT="${LOGS_DIR}/bench_de.sh"
cat > "$JOB2_SCRIPT" << 'EOFDE'
#!/usr/bin/env bash
set -euo pipefail

REPO="__REPO__"
RESULTS_DIR="${REPO}/benchmarks/results"

cd "$REPO"

echo "=== D: Query Engine Benchmarks (Rust, release) ==="
cargo test -p scx-engine bench_query_ --release -- --nocapture --ignored 2>&1 | tee "${RESULTS_DIR}/query_engine_benchmark_raw.txt"

echo ""
echo "=== E: File Operations Benchmarks (Rust, release) ==="
cargo test -p scx-ops bench_ops_ --release -- --nocapture --ignored 2>&1 | tee "${RESULTS_DIR}/ops_benchmark_raw.txt"

echo ""
echo "=== DONE ==="
EOFDE
sed -i "s|__REPO__|${REPO}|g" "$JOB2_SCRIPT"
chmod +x "$JOB2_SCRIPT"

JOB2_ID=$(sbatch \
    --job-name=scx-bench-de \
    --partition=cpu_preemptible \
    --cpus-per-task=8 \
    --mem=16G \
    --time=00:30:00 \
    --output="${LOGS_DIR}/bench_de_%j.out" \
    --error="${LOGS_DIR}/bench_de_%j.err" \
    "$JOB2_SCRIPT" | awk '{print $NF}')
echo "[Job 2] D-E benchmarks submitted: $JOB2_ID"

# --------------------------------------------------------------------------
# Job 3: ML Data Loader + SOTA baselines (F)
# --------------------------------------------------------------------------
JOB3_SCRIPT="${LOGS_DIR}/bench_f.sh"
cat > "$JOB3_SCRIPT" << 'EOFF'
#!/usr/bin/env bash
set -euo pipefail

REPO="__REPO__"
VENV="${REPO}/.venv"
PYTHON="${VENV}/bin/python"
DATA_DIR="__DATA_DIR__"
RESULTS_DIR="${REPO}/benchmarks/results"

cd "$REPO"
export SCX_DATA_DIR="$DATA_DIR"

echo "=== Building release ==="
cd pyscx && "${VENV}/bin/maturin" develop --release && cd ..

echo ""
echo "=== F: ML Data Loader Benchmarks ==="

"$PYTHON" << 'PYEOF'
import pyscx, time, gc, os, sys, json, resource, datetime
import numpy as np

DATA_DIR = os.environ["SCX_DATA_DIR"]
RESULTS_DIR = os.path.join(os.environ.get("REPO", "."), "benchmarks", "results")

DATASETS = {
    "pbmc3k": {"scx": f"{DATA_DIR}/pbmc3k.scx", "h5ad": f"{DATA_DIR}/pbmc3k.h5ad"},
    "smartseq2": {"scx": f"{DATA_DIR}/smartseq2.scx", "h5ad": f"{DATA_DIR}/smartseq2.h5ad"},
    "tabula_sapiens_100k": {"scx": f"{DATA_DIR}/tabula_sapiens_100k.scx", "h5ad": f"{DATA_DIR}/tabula_sapiens_100k.h5ad"},
    "census_1m": {"scx": f"{DATA_DIR}/census_1m.scx", "h5ad": f"{DATA_DIR}/census_1m.h5ad"},
}

all_results = []

for name, info in DATASETS.items():
    if not os.path.exists(info["scx"]):
        print(f"SKIP: {name} — no SCX file")
        continue

    print(f"\n=== {name} ===")
    res = {"dataset": name, "timestamp": datetime.datetime.now().isoformat()}

    # 1. Throughput (normalize+log1p)
    ds = pyscx.TrainingDataset(info["scx"], batch_size=1024, normalize=True, log1p=True, seed=42)
    res["n_obs"] = ds.n_obs
    res["n_vars"] = ds.n_vars
    t0 = time.time()
    n = 0
    first_t = None
    for batch in ds:
        if n == 0: first_t = time.time() - t0
        n += 1
    total = time.time() - t0
    res["throughput_norm"] = round(n / total, 1) if total > 0 else 0
    res["ttfb_norm_s"] = round(first_t, 3) if first_t else None
    res["n_batches"] = n
    print(f"  norm+log1p: {res['throughput_norm']} b/s, TTFB={res['ttfb_norm_s']}s")
    del ds; gc.collect()

    # 2. Throughput (raw decode)
    ds = pyscx.TrainingDataset(info["scx"], batch_size=1024, normalize=False, log1p=False, seed=42)
    t0 = time.time()
    n = 0
    first_t = None
    for batch in ds:
        if n == 0: first_t = time.time() - t0
        n += 1
    total = time.time() - t0
    res["throughput_raw"] = round(n / total, 1) if total > 0 else 0
    res["ttfb_raw_s"] = round(first_t, 3) if first_t else None
    print(f"  raw:        {res['throughput_raw']} b/s, TTFB={res['ttfb_raw_s']}s")
    del ds; gc.collect()

    # 3. HVG projection
    ds = pyscx.TrainingDataset(info["scx"], batch_size=1024, hvg_indices=list(range(2000)),
                                normalize=True, log1p=True, seed=42)
    t0 = time.time()
    n = 0
    for batch in ds:
        n += 1
    total = time.time() - t0
    res["throughput_hvg"] = round(n / total, 1) if total > 0 else 0
    res["hvg_speedup"] = round(res["throughput_hvg"] / res["throughput_norm"], 2) if res["throughput_norm"] > 0 else 0
    print(f"  HVG 2K:     {res['throughput_hvg']} b/s ({res['hvg_speedup']}x)")
    del ds; gc.collect()

    # 4. Batch size sweep
    res["batch_sweep"] = {}
    for bs in [256, 512, 1024, 2048, 4096]:
        ds = pyscx.TrainingDataset(info["scx"], batch_size=bs, normalize=True, log1p=True, seed=42)
        t0 = time.time()
        n = 0
        for batch in ds:
            n += 1
        total = time.time() - t0
        bps = round(n / total, 1) if total > 0 else 0
        res["batch_sweep"][bs] = bps
        del ds; gc.collect()
    print(f"  batch sweep: {res['batch_sweep']}")

    # 5. Memory (peak RSS delta)
    gc.collect()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
    ds = pyscx.TrainingDataset(info["scx"], batch_size=1024, normalize=True, log1p=True, seed=42, max_memory_mb=512)
    for batch in ds:
        pass
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    res["peak_rss_mb"] = round(rss_after, 0)
    res["rss_delta_mb"] = round(rss_after - rss_before, 0)
    print(f"  memory: peak={res['peak_rss_mb']} MB, delta={res['rss_delta_mb']} MB")
    del ds; gc.collect()

    # 6. AnnData baseline
    try:
        import anndata
        import scipy.sparse as sp
        adata = anndata.read_h5ad(info["h5ad"])
        n_obs = adata.n_obs
        t0 = time.time()
        n = 0
        for start in range(0, n_obs, 1024):
            end = min(start + 1024, n_obs)
            if sp.issparse(adata.X):
                batch_x = adata.X[start:end].toarray().astype(np.float32)
            else:
                batch_x = adata.X[start:end].astype(np.float32)
            n += 1
        total = time.time() - t0
        res["anndata_throughput"] = round(n / total, 1) if total > 0 else 0
        res["scx_vs_anndata"] = round(res["throughput_norm"] / res["anndata_throughput"], 2) if res.get("anndata_throughput", 0) > 0 else None
        print(f"  AnnData:    {res['anndata_throughput']} b/s (SCX/AnnData={res['scx_vs_anndata']}x)")
        del adata; gc.collect()
    except Exception as e:
        res["anndata_throughput"] = None
        print(f"  AnnData:    FAILED ({e})")

    # 7. TileDB-SOMA-ML baseline
    try:
        import tiledbsoma
        import tiledbsoma_ml
        import tempfile, torch

        soma_dir = os.path.join(tempfile.gettempdir(), f"scx_soma_{name}")
        if not os.path.exists(soma_dir):
            import anndata as ad
            adata = ad.read_h5ad(info["h5ad"])
            tiledbsoma.io.from_anndata(soma_dir, adata, measurement_name="RNA")
            del adata; gc.collect()

        with tiledbsoma.Experiment.open(soma_dir) as exp:
            ds = tiledbsoma_ml.ExperimentDataset(exp)
            loader = torch.utils.data.DataLoader(ds, batch_size=1024, num_workers=0)
            t0 = time.time()
            n = 0
            for batch in loader:
                n += 1
                if n >= 100:
                    break
            total = time.time() - t0
            res["soma_throughput"] = round(n / total, 1) if total > 0 else 0
            print(f"  TileDB-SOMA: {res['soma_throughput']} b/s ({n} batches)")
    except Exception as e:
        res["soma_throughput"] = None
        print(f"  TileDB-SOMA: FAILED ({e})")

    all_results.append(res)

# --- Generate markdown report ---
lines = []
lines.append("# SCX Training Loader Benchmark Report (Updated)\n")
lines.append(f"**Generated**: {datetime.datetime.now().isoformat(timespec='seconds')}\n")

lines.append("## 1. SCX Throughput\n")
lines.append("| Dataset | Norm+Log1p (b/s) | Raw (b/s) | HVG 2K (b/s) | HVG Speedup | TTFB (s) |")
lines.append("|---------|-----------------|----------|-------------|-------------|----------|")
for r in all_results:
    lines.append(f"| {r['dataset']} | {r['throughput_norm']} | {r['throughput_raw']} | {r['throughput_hvg']} | {r['hvg_speedup']}x | {r.get('ttfb_norm_s', 'N/A')} |")

lines.append("\n## 2. SOTA Comparison (batches/sec, norm+log1p)\n")
lines.append("| Dataset | SCX | AnnData | TileDB-SOMA | SCX/AnnData |")
lines.append("|---------|-----|---------|-------------|-------------|")
for r in all_results:
    ann = r.get("anndata_throughput", "N/A")
    soma = r.get("soma_throughput", "N/A")
    ratio = f"{r['scx_vs_anndata']}x" if r.get("scx_vs_anndata") else "N/A"
    lines.append(f"| {r['dataset']} | {r['throughput_norm']} | {ann} | {soma} | {ratio} |")

lines.append("\n## 3. Batch Size Sweep\n")
lines.append("| Dataset | 256 | 512 | 1024 | 2048 | 4096 |")
lines.append("|---------|-----|-----|------|------|------|")
for r in all_results:
    bs = r.get("batch_sweep", {})
    lines.append(f"| {r['dataset']} | {bs.get(256, 'N/A')} | {bs.get(512, 'N/A')} | {bs.get(1024, 'N/A')} | {bs.get(2048, 'N/A')} | {bs.get(4096, 'N/A')} |")

lines.append("\n## 4. Memory\n")
lines.append("| Dataset | Peak RSS (MB) | RSS Delta (MB) |")
lines.append("|---------|--------------|----------------|")
for r in all_results:
    lines.append(f"| {r['dataset']} | {r.get('peak_rss_mb', 'N/A')} | {r.get('rss_delta_mb', 'N/A')} |")

md = "\n".join(lines)
out_path = os.path.join(RESULTS_DIR, "training_loader_benchmark.md")
with open(out_path, "w") as f:
    f.write(md)
print(f"\nResults written to: {out_path}")

# Also save raw JSON
json_path = os.path.join(RESULTS_DIR, "training_loader_benchmark.json")
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"JSON written to: {json_path}")
PYEOF

echo ""
echo "=== DONE ==="
EOFF
sed -i "s|__REPO__|${REPO}|g; s|__DATA_DIR__|${DATA_DIR}|g" "$JOB3_SCRIPT"
chmod +x "$JOB3_SCRIPT"

JOB3_ID=$(sbatch \
    --job-name=scx-bench-f \
    --partition=cpu_preemptible \
    --cpus-per-task=16 \
    --mem=80G \
    --time=04:00:00 \
    --output="${LOGS_DIR}/bench_f_%j.out" \
    --error="${LOGS_DIR}/bench_f_%j.err" \
    --export=ALL,REPO="${REPO}" \
    "$JOB3_SCRIPT" | awk '{print $NF}')
echo "[Job 3] F benchmarks submitted: $JOB3_ID"

echo ""
echo "=== All jobs submitted ==="
echo "Monitor with: squeue -u \$USER"
echo "Results will be in: $RESULTS_DIR"
echo "Logs in: $LOGS_DIR"
