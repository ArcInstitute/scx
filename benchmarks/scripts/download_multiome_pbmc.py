#!/usr/bin/env python3
"""Download the 10x Genomics PBMC 10K Multiome dataset and write h5mu.

Source: 10x Genomics public dataset
*PBMC from a Healthy Donor - Granulocytes Removed Through Cell
Sorting (10k)* (10x Multiome ARC v1). The filtered feature-barcode
matrix combines `Gene Expression` and `Peaks` features in one HDF5
file. We split by feature_type, build per-modality `AnnData`s, and
persist as a multimodal `mudata.MuData` at
`$SCX_DATA_DIR/multiome_pbmc_10k.h5mu`.

The ATAC matrix is mostly-binary (per-peak presence/absence in a
cell), which exercises the Phase E codec routing decision
(`Lz4Shuffle` for non-binary ATAC, `Zstd` for binary).

Idempotent: skips when the output already exists.

Usage:
    python benchmarks/scripts/download_multiome_pbmc.py
"""
from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path

# 10x's CDN rejects the default Python-urllib User-Agent (HTTP 403);
# use a browser-like UA. See download_citeseq_pbmc.py for context.
_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) scx-benchmarks/0.1"


def _urlretrieve(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "comprehensive"))
from bench_env import DATA_DIR  # noqa: E402

# 10x Genomics public dataset:
# 10k PBMC Multiome (ARC v1) — granulocytes removed via cell sorting.
URL = (
    "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/"
    "pbmc_granulocyte_sorted_10k/"
    "pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5"
)
OUT_NAME = "multiome_pbmc_10k.h5mu"
RAW_NAME = "pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5"


def _split_by_feature_type(adata):
    """Split a 10x Multiome .h5 AnnData into `{name: AnnData}` for
    each feature_type. Maps `"Gene Expression"` → `"rna"` and
    `"Peaks"` → `"atac"` (the canonical names used by the SCX
    multimodal pipeline)."""
    if "feature_types" not in adata.var.columns:
        raise RuntimeError(
            "var.feature_types missing from 10x .h5 — cannot split Multiome"
        )
    name_map = {
        "Gene Expression": "rna",
        "Peaks": "atac",
    }
    out = {}
    for ft, mname in name_map.items():
        mask = (adata.var["feature_types"] == ft).to_numpy()
        n = int(mask.sum())
        if n == 0:
            raise RuntimeError(
                f"feature_type {ft!r} not present in 10x .h5; "
                f"unique values: {adata.var['feature_types'].unique()}"
            )
        sub = adata[:, mask].copy()
        sub.var = sub.var.drop(columns=["feature_types"])
        sub.var_names_make_unique()
        out[mname] = sub
    return out


def main() -> None:
    out_dir = Path(DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUT_NAME

    if out_path.exists():
        print(f"[SKIP] {out_path} already exists")
        return

    raw_path = out_dir / RAW_NAME
    if not raw_path.exists():
        print(f"Downloading 10x PBMC 10k Multiome (~600 MB)...")
        print(f"  URL: {URL}")
        _urlretrieve(URL, raw_path)
        size_mb = raw_path.stat().st_size / 1e6
        print(f"  Downloaded: {raw_path} ({size_mb:.1f} MB)")
    else:
        print(f"Raw .h5 already present: {raw_path}")

    import mudata
    import scanpy as sc
    import scipy.sparse as sp

    print("Reading 10x .h5 with scanpy.read_10x_h5(gex_only=False)...")
    adata = sc.read_10x_h5(str(raw_path), gex_only=False)
    if not sp.issparse(adata.X) or adata.X.format != "csr":
        adata.X = sp.csr_matrix(adata.X)

    print("Splitting by feature_type → MuData{rna, atac}...")
    mod_dict = _split_by_feature_type(adata)
    for name, ad_ in mod_dict.items():
        nnz = ad_.X.nnz if sp.issparse(ad_.X) else int((ad_.X != 0).sum())
        print(f"  {name}: {ad_.n_obs} x {ad_.n_vars}  nnz={nnz:,}")

    mu = mudata.MuData(mod_dict)
    mu.update()

    print(f"Writing h5mu → {out_path} ...")
    mu.write_h5mu(str(out_path))
    out_size_mb = out_path.stat().st_size / 1e6
    print(f"  h5mu size: {out_size_mb:.1f} MB")

    try:
        os.remove(raw_path)
    except OSError:
        pass

    print("Re-opening to verify...")
    mu_back = mudata.read_h5mu(str(out_path))
    assert "rna" in mu_back.mod and "atac" in mu_back.mod, mu_back.mod.keys()
    print(f"  rna: {mu_back.mod['rna'].shape}, atac: {mu_back.mod['atac'].shape}")
    print("Done!")


if __name__ == "__main__":
    main()
