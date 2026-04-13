#!/usr/bin/env python3
"""Verify all benchmark datasets and record metadata.

Records §2.3 properties: n_obs, n_vars, NNZ, sparsity, assay/protocol,
value distribution (median NZ, max NZ, % uint8/uint16/uint32), source.
Outputs a summary table.
"""
import os
import sys
import json
import numpy as np
import anndata as ad
import scipy.sparse as sp

from bench_env import DATA_DIR

DATASETS_DIR = str(DATA_DIR)

DATASET_META = {
    "pbmc3k":             {"id": "D1", "protocol": "10x v2 (UMI)", "source": "10x Genomics"},
    "pbmc10k":            {"id": "D2", "protocol": "10x v3 (UMI)", "source": "10x Genomics"},
    "smartseq2":          {"id": "D3", "protocol": "Smart-seq2",   "source": "CELLxGENE Census"},
    "tabula_sapiens_100k":{"id": "D4", "protocol": "10x (UMI)",    "source": "CELLxGENE Census"},
    "census_500k":        {"id": "D5", "protocol": "10x (UMI)",    "source": "CELLxGENE Census (blood)"},
    "census_1m":          {"id": "D6", "protocol": "10x (UMI)",    "source": "CELLxGENE Census (blood)"},
    "census_5m":          {"id": "D7", "protocol": "10x (UMI)",    "source": "CELLxGENE Census (blood)"},
    "census_10m":         {"id": "D8", "protocol": "Mixed",        "source": "CELLxGENE Census (blood)"},
}


def _analyze_large_h5ad(path, n_obs, n_vars):
    """Stream NNZ and value distribution from a large h5ad via h5py.

    Reads the CSR data array in chunks to avoid full materialization.
    Returns a dict with nnz, sparsity, and value distribution stats.
    """
    import h5py

    result = {}
    with h5py.File(path, "r") as f:
        x_group = f["X"]

        if "data" in x_group:
            # Sparse CSR/CSC stored as group with data/indices/indptr
            data_ds = x_group["data"]
            nnz = data_ds.shape[0]
            result["nnz"] = int(nnz)
            total = n_obs * n_vars
            result["sparsity"] = round(1 - nnz / total, 6)

            # Stream value distribution in chunks
            chunk_size = 50_000_000  # 50M values at a time
            max_val = 0.0
            n_uint8 = 0
            n_uint16 = 0
            all_int = True
            # For approximate median: collect a random sample
            rng = np.random.RandomState(42)
            sample_size = 1_000_000
            # Pre-select which indices to sample
            sample_indices = np.sort(rng.choice(nnz, size=min(sample_size, nnz), replace=False))
            sample_values = np.empty(len(sample_indices), dtype=np.float32)
            sample_ptr = 0  # next index into sample_indices to fill

            for start in range(0, nnz, chunk_size):
                end = min(start + chunk_size, nnz)
                chunk = data_ds[start:end]
                abs_chunk = np.abs(chunk).astype(np.float64)

                max_val = max(max_val, float(abs_chunk.max()))
                n_uint8 += int(np.sum(abs_chunk <= 255))
                n_uint16 += int(np.sum(abs_chunk <= 65535))
                if all_int:
                    all_int = bool(np.all(chunk == chunk.astype(int)))

                # Gather pre-selected sample values from this chunk
                while sample_ptr < len(sample_indices) and sample_indices[sample_ptr] < end:
                    local_idx = sample_indices[sample_ptr] - start
                    sample_values[sample_ptr] = abs(float(chunk[local_idx]))
                    sample_ptr += 1

            result["median_nz"] = float(np.median(sample_values[:sample_ptr]))
            result["max_nz"] = max_val
            result["pct_uint8"] = round(n_uint8 / nnz * 100, 1)
            result["pct_uint16"] = round(n_uint16 / nnz * 100, 1)
            result["values_are_integer"] = all_int
        else:
            # Dense matrix — unlikely for large files but handle anyway
            result["status"] = "OK (dense matrix — stats skipped for large file)"
            result["nnz"] = "N/A (dense)"

    return result


def analyze_dataset(name, path):
    """Analyze a dataset and return metadata dict."""
    meta = DATASET_META.get(name, {})
    result = {
        "name": name,
        "id": meta.get("id", "?"),
        "path": path,
    }

    if not os.path.exists(path):
        result["status"] = "MISSING"
        return result

    try:
        file_size = os.path.getsize(path)
        result["file_size_bytes"] = file_size
        result["file_size_gb"] = round(file_size / 1e9, 2)

        # Read in backed mode first to get shape without loading X
        adata = ad.read_h5ad(path, backed='r')
        result["n_obs"] = adata.n_obs
        result["n_vars"] = adata.n_vars
        result["protocol"] = meta.get("protocol", "unknown")
        result["source"] = meta.get("source", "unknown")

        # For very large files (>50 GB), stream NNZ and value stats via h5py
        # to avoid loading the full matrix into memory.
        if file_size > 50e9:
            import h5py
            result.update(_analyze_large_h5ad(path, adata.n_obs, adata.n_vars))
        else:
            adata_full = ad.read_h5ad(path)
            X = adata_full.X
            if sp.issparse(X):
                X = X.tocsr()
                nnz = X.nnz
                data = X.data
            else:
                nnz = np.count_nonzero(X)
                data = X[X != 0].ravel()

            total = adata_full.n_obs * adata_full.n_vars
            sparsity = 1 - nnz / total

            result["nnz"] = int(nnz)
            result["sparsity"] = round(sparsity, 6)

            # Value distribution
            if len(data) > 0:
                nz_vals = np.abs(data)
                result["median_nz"] = float(np.median(nz_vals))
                result["max_nz"] = float(np.max(nz_vals))
                result["pct_uint8"] = round(float(np.mean(nz_vals <= 255)) * 100, 1)
                result["pct_uint16"] = round(float(np.mean(nz_vals <= 65535)) * 100, 1)

                # Check if values are integers
                is_int = np.all(data == data.astype(int))
                result["values_are_integer"] = bool(is_int)
            else:
                result["median_nz"] = 0
                result["max_nz"] = 0

            del adata_full, X, data

        result["status"] = "OK"

    except Exception as e:
        result["status"] = f"ERROR: {e}"

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify benchmark datasets")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names (default: all)")
    args = parser.parse_args()

    names = args.datasets.split(",") if args.datasets else list(DATASET_META.keys())

    print("=" * 80)
    print("   SCX Benchmark Dataset Verification")
    print("=" * 80)
    print(f"Datasets dir: {DATASETS_DIR}")
    print(f"Datasets: {names}")
    print()

    results = []
    for name in names:
        path = os.path.join(DATASETS_DIR, f"{name}.h5ad")
        print(f"Analyzing {name}...", flush=True)
        result = analyze_dataset(name, path)
        results.append(result)

        if result.get("status", "").startswith("OK"):
            print(f"  ✅ {result['id']} {name}: {result.get('n_obs', '?'):,} × {result.get('n_vars', '?'):,}")
            if 'nnz' in result and isinstance(result['nnz'], int):
                print(f"     NNZ: {result['nnz']:,}  Sparsity: {result['sparsity']:.4f}")
                print(f"     Median NZ: {result.get('median_nz', '?')}  Max NZ: {result.get('max_nz', '?')}")
                print(f"     % fits uint8: {result.get('pct_uint8', '?')}%  uint16: {result.get('pct_uint16', '?')}%")
                print(f"     Integer values: {result.get('values_are_integer', '?')}")
        elif result.get("status") == "MISSING":
            print(f"  ❌ {result['id']} {name}: MISSING")
        else:
            print(f"  ⚠️  {result['id']} {name}: {result['status']}")
        print(f"     File size: {result.get('file_size_gb', 'N/A')} GB")
        print()

    # Save results as JSON — merge with any existing metadata
    output_json = os.path.join(DATASETS_DIR, "dataset_metadata.json")
    existing = []
    if os.path.exists(output_json):
        with open(output_json) as f:
            existing = json.load(f)
    # Merge: update existing entries by name, append new ones
    by_name = {r["name"]: r for r in existing}
    for r in results:
        by_name[r["name"]] = r
    merged = sorted(by_name.values(), key=lambda r: r.get("id", "Z"))
    with open(output_json, 'w') as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"Metadata saved to: {output_json}")

    # Summary table
    print()
    print("=" * 100)
    print(f"{'ID':<4} {'Name':<22} {'Cells':>12} {'Genes':>8} {'NNZ':>15} {'Sparsity':>10} {'Size (GB)':>10} {'Status':<10}")
    print("-" * 100)
    for r in results:
        nnz_str = f"{r['nnz']:,}" if isinstance(r.get('nnz'), int) else str(r.get('nnz', 'N/A'))
        cells = f"{r.get('n_obs', '?'):,}" if isinstance(r.get('n_obs'), int) else '?'
        genes = f"{r.get('n_vars', '?'):,}" if isinstance(r.get('n_vars'), int) else '?'
        sparsity = f"{r.get('sparsity', 0):.4f}" if 'sparsity' in r else 'N/A'
        size = f"{r.get('file_size_gb', 'N/A')}"
        status = "✅" if r.get('status', '').startswith('OK') else "❌" if r.get('status') == 'MISSING' else "⚠️"
        print(f"{r.get('id', '?'):<4} {r['name']:<22} {cells:>12} {genes:>8} {nnz_str:>15} {sparsity:>10} {size:>10} {status:<10}")
    print("=" * 100)


if __name__ == "__main__":
    main()
