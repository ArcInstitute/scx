#!/usr/bin/env python3
"""Generate gzip and lzf compressed h5ad variants for all benchmark datasets.

For each dataset, produces:
  - {name}_gzip.h5ad  (gzip level 4 compression on X arrays)
  - {name}_lzf.h5ad   (lzf compression on X arrays)

Small datasets (< 50 GB) are read fully into memory and written via anndata.
Large datasets (>= 50 GB) are copied at the HDF5 level with h5py, re-compressing
only the CSR data arrays (data, indices, indptr) while copying everything else.

Usage:
  python generate_compressed_h5ad.py [--datasets D1,D2,...] [--compressions gzip,lzf]
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np

DATASETS_DIR = "/scratch/ctc/nickyoungblut/scx/benchmarks/datasets"

DATASET_NAMES = [
    "pbmc3k",
    "pbmc10k",
    "smartseq2",
    "tabula_sapiens_100k",
    "census_500k",
    "census_1m",
    "census_5m",
    "census_10m",
]

# Datasets larger than this threshold use the h5py copy path
LARGE_THRESHOLD_BYTES = 50 * 1024**3  # 50 GB


def compress_small(src_path: str, dst_path: str, compression: str) -> None:
    """Compress a small h5ad by reading into memory and re-writing."""
    import anndata as ad

    print(f"    Reading {src_path} into memory...")
    adata = ad.read_h5ad(src_path)
    print(f"    Writing {dst_path} with compression={compression}...")
    adata.write_h5ad(dst_path, compression=compression)
    del adata


def _copy_group(src_group: h5py.Group, dst_group: h5py.Group,
                compression: str, compression_opts: int | None,
                x_path: str) -> None:
    """Recursively copy an HDF5 group, re-compressing datasets under x_path."""
    for key in src_group:
        item = src_group[key]
        full_path = item.name  # e.g. "/X/data"

        if isinstance(item, h5py.Group):
            grp = dst_group.require_group(key)
            # Copy group attributes
            for attr_name, attr_val in item.attrs.items():
                grp.attrs[attr_name] = attr_val
            _copy_group(item, grp, compression, compression_opts, x_path)

        elif isinstance(item, h5py.Dataset):
            # Determine if this dataset should be re-compressed
            is_x_array = full_path.startswith(x_path + "/")
            # Also handle the case where X is a direct dataset (dense)
            if full_path == x_path:
                is_x_array = True

            if is_x_array and item.size > 0:
                # Re-compress: read in chunks to manage memory
                chunk_size = min(item.shape[0], 10_000_000)
                ds = dst_group.create_dataset(
                    key,
                    shape=item.shape,
                    dtype=item.dtype,
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=(min(chunk_size, item.shape[0]),) if item.ndim == 1 else True,
                )
                # Copy in chunks for 1-D arrays (indptr, indices, data)
                if item.ndim == 1:
                    n = item.shape[0]
                    for start in range(0, n, chunk_size):
                        end = min(start + chunk_size, n)
                        ds[start:end] = item[start:end]
                else:
                    # Dense or multi-dim: read/write in row chunks
                    row_chunk = min(10000, item.shape[0])
                    for start in range(0, item.shape[0], row_chunk):
                        end = min(start + row_chunk, item.shape[0])
                        ds[start:end] = item[start:end]
            else:
                # Copy as-is (metadata, obs, var, etc.) — use h5py copy
                src_group.copy(key, dst_group, key)

            # Copy dataset attributes
            if key in dst_group:
                for attr_name, attr_val in item.attrs.items():
                    dst_group[key].attrs[attr_name] = attr_val


def compress_large(src_path: str, dst_path: str, compression: str) -> None:
    """Compress a large h5ad at the HDF5 level without loading into memory."""
    compression_opts = 4 if compression == "gzip" else None

    print(f"    Opening {src_path} for h5py-level copy...")
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        # Copy root attributes
        for attr_name, attr_val in src.attrs.items():
            dst.attrs[attr_name] = attr_val

        # Determine X path (usually "/X")
        x_path = "/X"
        if "raw" in src and "X" in src["raw"]:
            # Some h5ad files store raw.X as well; we skip raw re-compression
            pass

        _copy_group(src, dst, compression, compression_opts, x_path)

    print(f"    Wrote {dst_path}")


def generate_compressed(name: str, compressions: list[str]) -> None:
    """Generate compressed variants for a single dataset."""
    src_path = os.path.join(DATASETS_DIR, f"{name}.h5ad")
    if not os.path.exists(src_path):
        print(f"  SKIP {name}: source not found at {src_path}")
        return

    file_size = os.path.getsize(src_path)
    is_large = file_size >= LARGE_THRESHOLD_BYTES
    method = "h5py-level copy" if is_large else "anndata read/write"
    print(f"  {name}: {file_size / 1e9:.1f} GB — using {method}")

    for comp in compressions:
        dst_path = os.path.join(DATASETS_DIR, f"{name}_{comp}.h5ad")
        if os.path.exists(dst_path):
            existing_size = os.path.getsize(dst_path)
            print(f"    {comp}: already exists ({existing_size / 1e9:.2f} GB) — skipping")
            continue

        t0 = time.perf_counter()
        try:
            if is_large:
                compress_large(src_path, dst_path, comp)
            else:
                compress_small(src_path, dst_path, comp)

            elapsed = time.perf_counter() - t0
            dst_size = os.path.getsize(dst_path)
            ratio = dst_size / file_size
            print(f"    {comp}: {dst_size / 1e9:.2f} GB "
                  f"(ratio={ratio:.3f}) in {elapsed:.1f}s")
        except Exception as e:
            print(f"    {comp}: ERROR — {e}")
            # Clean up partial file
            if os.path.exists(dst_path):
                os.remove(dst_path)


def main():
    parser = argparse.ArgumentParser(description="Generate compressed h5ad variants")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names (default: all)")
    parser.add_argument("--compressions", type=str, default="gzip,lzf",
                        help="Comma-separated compression types (default: gzip,lzf)")
    args = parser.parse_args()

    datasets = args.datasets.split(",") if args.datasets else DATASET_NAMES
    compressions = args.compressions.split(",")

    print("=" * 60)
    print("  Generate Compressed h5ad Variants")
    print("=" * 60)
    print(f"  Datasets: {datasets}")
    print(f"  Compressions: {compressions}")
    print(f"  Output dir: {DATASETS_DIR}")
    print()

    for name in datasets:
        generate_compressed(name, compressions)
        print()

    # Summary
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    for name in datasets:
        src = os.path.join(DATASETS_DIR, f"{name}.h5ad")
        if not os.path.exists(src):
            continue
        src_size = os.path.getsize(src)
        line = f"  {name}: {src_size / 1e9:.1f} GB"
        for comp in compressions:
            dst = os.path.join(DATASETS_DIR, f"{name}_{comp}.h5ad")
            if os.path.exists(dst):
                dst_size = os.path.getsize(dst)
                ratio = dst_size / src_size
                line += f" | {comp}={dst_size / 1e9:.1f} GB ({ratio:.2f}x)"
            else:
                line += f" | {comp}=MISSING"
        print(line)
    print()


if __name__ == "__main__":
    main()
