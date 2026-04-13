#!/usr/bin/env python3
"""Minimal Leiden benchmark — Rust vs Python timing, ARI, quality checks."""
import gc, os, sys, time, json, subprocess
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", str(os.environ.get("SLURM_CPUS_PER_TASK", "16")))

from bench_env import WORK_DIR

def rss():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"): return int(line.split()[1]) // 1024
    except: pass
    return 0

def main():
    sys.stdout.reconfigure(line_buffering=True)
    ds = sys.argv[1] if len(sys.argv) > 1 else "tabula_sapiens_100k"
    data_dir = WORK_DIR
    h5ad = data_dir / f"{ds}.h5ad"
    results = {"dataset": ds, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    print(f"Loading {ds}... RSS={rss()}MB")
    import anndata, scanpy as sc, pyscx
    adata = anndata.read_h5ad(str(h5ad))
    adata.raw = None; gc.collect()
    print(f"Loaded: {adata.shape}, RSS={rss()}MB")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    gc.collect()
    print(f"After preprocess: {adata.shape}, RSS={rss()}MB")

    pyscx.accel.pca(adata, n_comps=50, device="cpu")
    pyscx.accel.neighbors(adata, n_neighbors=15, device="cpu")
    gc.collect()
    print(f"After PCA+kNN: RSS={rss()}MB")

    # ── Rust Leiden timing (3 runs, each with unique key) ──
    times = []
    for i in range(3):
        key = f"leiden_r{i}"
        t0 = time.perf_counter()
        pyscx.accel.leiden(adata, resolution=1.0, key_added=key, random_state=42, device="cpu")
        times.append(time.perf_counter() - t0)
        print(f"  Rust run {i+1}/3: {times[-1]:.2f}s, RSS={rss()}MB")
    rust_membership = adata.obs["leiden_r2"].values.copy()
    results["rust_median_s"] = round(sorted(times)[1], 3)
    results["rust_times_s"] = [round(t, 3) for t in times]
    results["rust_modularity"] = adata.uns["leiden_r2"]["modularity"]
    results["rust_n_communities"] = adata.uns["leiden_r2"]["n_communities"]
    results["rust_backend"] = adata.uns["leiden_r2"].get("backend", "unknown")
    # Clean up rust results
    for i in range(3):
        k = f"leiden_r{i}"
        if k in adata.obs.columns: del adata.obs[k]
        if k in adata.uns: del adata.uns[k]
    gc.collect()
    print(f"  After cleanup: RSS={rss()}MB")

    # ── Python Leiden timing (1 run) ──
    t0 = time.perf_counter()
    sc.tl.leiden(adata, resolution=1.0, random_state=42, key_added="leiden_py")
    python_t = time.perf_counter() - t0
    python_membership = adata.obs["leiden_py"].values.copy()
    results["python_s"] = round(python_t, 3)
    results["python_n_communities"] = len(set(python_membership))
    print(f"  Python: {python_t:.2f}s, {results['python_n_communities']} comms, RSS={rss()}MB")
    del adata.obs["leiden_py"]; gc.collect()

    # ── ARI ──
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(rust_membership, python_membership)
    results["ari"] = round(ari, 4)
    results["speedup"] = round(python_t / results["rust_median_s"], 2) if results["rust_median_s"] > 0 else 0
    results["modularity_positive"] = results["rust_modularity"] > 0

    # ── Resolution test (subprocess to avoid memory accumulation) ──
    res_script = f'''
import gc, os, json
os.environ["RAYON_NUM_THREADS"] = "{os.environ.get('RAYON_NUM_THREADS', '16')}"
os.environ["MALLOC_ARENA_MAX"] = "{os.environ.get('MALLOC_ARENA_MAX', '4')}"
import anndata, scanpy as sc, pyscx
adata = anndata.read_h5ad("{h5ad}")
adata.raw = None; gc.collect()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
pyscx.accel.pca(adata, n_comps=50, device="cpu")
pyscx.accel.neighbors(adata, n_neighbors=15, device="cpu")
gc.collect()
res_data = []
for res in [0.5, 1.0, 2.0]:
    key = f"lr_{{res}}"
    pyscx.accel.leiden(adata, resolution=res, key_added=key, random_state=42, device="cpu")
    n = adata.uns[key]["n_communities"]
    q = adata.uns[key]["modularity"]
    res_data.append({{"resolution": res, "n_communities": n, "modularity": q}})
    del adata.obs[key]; del adata.uns[key]; gc.collect()
print(json.dumps(res_data))
'''
    try:
        proc = subprocess.run([sys.executable, "-c", res_script],
                              capture_output=True, text=True, timeout=3600)
        if proc.returncode == 0 and proc.stdout.strip():
            res_results = json.loads(proc.stdout.strip())
            results["resolution_test"] = res_results
            comms = [r["n_communities"] for r in res_results]
            results["resolution_monotonic"] = all(comms[i] <= comms[i+1] for i in range(len(comms)-1))
            for r in res_results:
                print(f"  Resolution γ={r['resolution']}: {r['n_communities']} comms, Q={r['modularity']:.1f}")
        else:
            print(f"  Resolution test failed (rc={proc.returncode}): {proc.stderr[:300]}")
            results["resolution_test"] = None
            results["resolution_monotonic"] = None
    except Exception as e:
        print(f"  Resolution test error: {e}")
        results["resolution_test"] = None
        results["resolution_monotonic"] = None

    print(f"\n=== Results for {ds} ===")
    print(f"  Rust: {results['rust_median_s']:.2f}s, {results['rust_n_communities']} comms, Q={results['rust_modularity']:.1f}")
    print(f"  Python: {results['python_s']:.2f}s, {results['python_n_communities']} comms")
    print(f"  ARI: {ari:.4f}, Speedup: {results['speedup']:.1f}x")
    print(f"  Modularity>0: {results['modularity_positive']}")
    print(f"  Resolution monotonic: {results.get('resolution_monotonic')}")

    out_dir = Path("benchmarks/results"); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"leiden_benchmark_{ds}.json").write_text(json.dumps(results, indent=2))
    print(f"  Saved: benchmarks/results/leiden_benchmark_{ds}.json")

if __name__ == "__main__":
    main()
