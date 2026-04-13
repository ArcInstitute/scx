#!/usr/bin/env python3
"""Benchmark compression ratios: SCX (auto/none/scx1/zstd) vs h5ad vs Zarr+Zstd."""

import gc
import os
import sys
import tempfile
from pathlib import Path

# Add project root so we can import pyscx after maturin develop --release
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pyscx"))
sys.path.insert(0, str(Path(__file__).parent))
from build_release import ensure_release_build

from bench_env import WORK_DIR

# Benchmark datasets (skip chunk files)
BENCHMARK_DATASETS = [
    "pbmc3k", "smartseq2", "tabula_sapiens_100k", "census_1m",
]


def get_h5ad_files():
    """Find benchmark h5ad files in the data directory."""
    data_path = Path(WORK_DIR)
    if not data_path.exists():
        print(f"WARNING: Data directory not found: {WORK_DIR}")
        return []
    files = []
    for name in BENCHMARK_DATASETS:
        path = data_path / f"{name}.h5ad"
        if path.exists():
            files.append(path)
    return files


def benchmark_compression(h5ad_path):
    """Measure compression ratios for a single h5ad file across all codecs."""
    import anndata
    import pyscx

    result = {
        "dataset": h5ad_path.stem,
        "h5ad_size": h5ad_path.stat().st_size,
    }

    # Read h5ad
    adata = anndata.read_h5ad(str(h5ad_path))
    result["n_obs"] = adata.n_obs
    result["n_vars"] = adata.n_vars

    # Compute raw CSR size for reference
    import scipy.sparse as sp
    X = adata.X
    if sp.issparse(X):
        X_csr = sp.csr_matrix(X)
        result["nnz"] = X_csr.nnz
        raw_csr_bytes = (
            X_csr.indptr.nbytes + X_csr.indices.nbytes + X_csr.data.nbytes
        )
        result["raw_csr_size"] = raw_csr_bytes
    else:
        result["nnz"] = adata.n_obs * adata.n_vars
        result["raw_csr_size"] = X.nbytes

    # Convert to SCX with each codec
    codecs = ["auto", "none", "scx1", "zstd"]
    for codec in codecs:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                scx_path = os.path.join(tmpdir, f"test_{codec}.scx")
                pyscx.from_anndata(adata, scx_path, codec=codec)
                size = os.path.getsize(scx_path)
                result[f"scx_{codec}_size"] = size
                result[f"scx_{codec}_ratio"] = size / result["h5ad_size"]
        except Exception as e:
            result[f"scx_{codec}_size"] = None
            result[f"scx_{codec}_ratio"] = None
            print(f"  WARNING: codec {codec} failed: {e}")

    # Convenience: "scx_size" = auto codec (default)
    result["scx_size"] = result.get("scx_auto_size", 0)
    result["ratio"] = result.get("scx_auto_ratio", 0)

    # Compare against Zarr+Zstd
    try:
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test.zarr")
            store = zarr.storage.LocalStore(zarr_path)
            root = zarr.group(store=store)
            zstd = zarr.codecs.ZstdCodec(level=3)

            if sp.issparse(X):
                X_csr = sp.csr_matrix(X)
                root.create_array("indptr", data=X_csr.indptr, compressors=zstd)
                root.create_array("indices", data=X_csr.indices, compressors=zstd)
                root.create_array("data", data=X_csr.data, compressors=zstd)
            else:
                root.create_array("X", data=X, compressors=zstd)

            zarr_size = sum(
                f.stat().st_size for f in Path(zarr_path).rglob("*") if f.is_file()
            )
            result["zarr_zstd_size"] = zarr_size
            result["zarr_ratio"] = zarr_size / result["h5ad_size"]
    except (ImportError, Exception) as e:
        result["zarr_zstd_size"] = None
        result["zarr_ratio"] = None

    del adata
    gc.collect()
    return result


def run_all():
    """Run compression benchmarks on all datasets."""
    h5ad_files = get_h5ad_files()
    if not h5ad_files:
        print("No h5ad files found. Run download_datasets.sh first.")
        return []

    results = []
    for h5ad_path in h5ad_files:
        print(f"Benchmarking compression: {h5ad_path.name}...")
        try:
            result = benchmark_compression(h5ad_path)
            results.append(result)

            def _fmt(sz):
                return f"{sz / 1e6:.1f} MB" if sz else "N/A"

            print(f"  n_obs={result['n_obs']:,}, n_vars={result['n_vars']:,}, nnz={result.get('nnz', '?'):,}")
            print(f"  h5ad: {_fmt(result['h5ad_size'])}")
            for codec in ["auto", "none", "scx1", "zstd"]:
                sz = result.get(f"scx_{codec}_size")
                ratio = result.get(f"scx_{codec}_ratio")
                print(f"  SCX ({codec:4s}): {_fmt(sz)} (ratio={ratio:.3f})" if sz else f"  SCX ({codec:4s}): FAILED")
            zarr_sz = result.get("zarr_zstd_size")
            if zarr_sz:
                print(f"  Zarr+Zstd: {_fmt(zarr_sz)} (ratio={result['zarr_ratio']:.3f})")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    return results


if __name__ == "__main__":
    ensure_release_build()
    run_all()
