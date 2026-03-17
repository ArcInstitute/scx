#!/usr/bin/env python3
"""Benchmark compression ratios: SCX vs h5ad vs Zarr+Zstd (Task 17.2)."""

import os
import sys
import tempfile
from pathlib import Path

# Add project root so we can import pyscx after maturin develop
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pyscx"))

DATA_DIR = os.environ.get("SCX_DATA_DIR", "/scratch/ctc/nickyoungblut/scx")


def get_h5ad_files():
    """Find all h5ad files in the data directory."""
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        print(f"WARNING: Data directory not found: {DATA_DIR}")
        return []
    return sorted(data_path.glob("*.h5ad"))


def benchmark_compression(h5ad_path):
    """Measure compression ratios for a single h5ad file."""
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

    # Convert to SCX
    with tempfile.TemporaryDirectory() as tmpdir:
        scx_path = os.path.join(tmpdir, "test.scx")
        pyscx.from_anndata(adata, scx_path)
        result["scx_size"] = os.path.getsize(scx_path)

    result["ratio"] = result["scx_size"] / result["h5ad_size"]

    # Compare against Zarr+Zstd if zarr is available
    try:
        import numcodecs
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            import scipy.sparse as sp

            zarr_path = os.path.join(tmpdir, "test.zarr")

            # Zarr v3 API
            store = zarr.storage.LocalStore(zarr_path)
            root = zarr.group(store=store)
            zstd = zarr.codecs.ZstdCodec(level=3)

            X = adata.X
            if sp.issparse(X):
                X_csr = sp.csr_matrix(X)
                root.create_array(
                    "indptr", data=X_csr.indptr, compressors=zstd,
                )
                root.create_array(
                    "indices", data=X_csr.indices, compressors=zstd,
                )
                root.create_array(
                    "data", data=X_csr.data, compressors=zstd,
                )
            else:
                root.create_array(
                    "X", data=X, compressors=zstd,
                )

            # Get total zarr size
            zarr_size = sum(
                f.stat().st_size
                for f in Path(zarr_path).rglob("*")
                if f.is_file()
            )
            result["zarr_zstd_size"] = zarr_size
            result["zarr_ratio"] = zarr_size / result["h5ad_size"]
    except ImportError:
        result["zarr_zstd_size"] = None
        result["zarr_ratio"] = None

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
            print(
                f"  h5ad: {result['h5ad_size'] / 1e6:.1f} MB, "
                f"SCX: {result['scx_size'] / 1e6:.1f} MB, "
                f"ratio: {result['ratio']:.3f}"
            )
        except Exception as e:
            print(f"  ERROR: {e}")

    return results


if __name__ == "__main__":
    run_all()
