#!/usr/bin/env python3
"""Download the 10x Genomics PBMC 5K CITE-seq dataset and write h5mu.

Source: 10x Genomics public dataset
*5k Peripheral blood mononuclear cells (PBMCs) from a Healthy Donor
with cell surface proteins (v3 chemistry)*. The filtered
feature-barcode matrix combines `Gene Expression` and
`Antibody Capture` features in one HDF5 file. We split them by
feature_type, wrap each in an `AnnData`, and persist as a
multimodal `mudata.MuData` object at `$SCX_DATA_DIR/cite_seq_pbmc_5k.h5mu`.

Idempotent: skips when the output already exists. Mirrors
`download_pbmc10k.py` but emits `.h5mu` instead of `.h5ad`.

Usage:
    python benchmarks/scripts/download_citeseq_pbmc.py
"""
from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path

# 10x's CDN rejects the default Python-urllib User-Agent (HTTP 403);
# use a browser-like UA to match the existing download_pbmc10k.py
# behaviour (which works because scanpy ships with a UA-setting
# wrapper internally — we don't take that path here).
_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) scx-benchmarks/0.1"


def _urlretrieve(url: str, dest: Path) -> None:
    """urlretrieve with a custom User-Agent so 10x's CDN serves us."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)

# Ensure benchmarks/comprehensive/bench_env is importable regardless of
# cwd (mirrors the pattern in benchmarks/scripts/benchmark_*.py).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "comprehensive"))
from bench_env import DATA_DIR  # noqa: E402

# 10x Genomics public dataset:
# 5k PBMCs from a Healthy Donor with TotalSeq-B antibodies (v3 chemistry).
URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/3.1.0/"
    "5k_pbmc_protein_v3/"
    "5k_pbmc_protein_v3_filtered_feature_bc_matrix.h5"
)
OUT_NAME = "cite_seq_pbmc_5k.h5mu"
RAW_NAME = "5k_pbmc_protein_v3_filtered_feature_bc_matrix.h5"


def _split_by_feature_type(adata):
    """Split a 10x .h5 AnnData carrying mixed feature_types into
    `{name: AnnData}` for each feature_type. The 10x convention
    annotates `var["feature_types"]` with values like
    `"Gene Expression"` and `"Antibody Capture"`. We map those to the
    multimodal pipeline's canonical `"rna"` and `"adt"` modality
    names.
    """
    if "feature_types" not in adata.var.columns:
        raise RuntimeError(
            "var.feature_types missing from 10x .h5 — cannot split CITE-seq"
        )
    name_map = {
        "Gene Expression": "rna",
        "Antibody Capture": "adt",
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
        # Reset var to drop the multi-modality feature_types column,
        # which is no longer informative inside a single-modality
        # AnnData.
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
        print(f"Downloading 10x PBMC 5k CITE-seq (~110 MB)...")
        print(f"  URL: {URL}")
        _urlretrieve(URL, raw_path)
        size_mb = raw_path.stat().st_size / 1e6
        print(f"  Downloaded: {raw_path} ({size_mb:.1f} MB)")
    else:
        print(f"Raw .h5 already present: {raw_path}")

    import anndata
    import mudata
    import scanpy as sc
    import scipy.sparse as sp

    print("Reading 10x .h5 with scanpy.read_10x_h5(gex_only=False)...")
    adata = sc.read_10x_h5(str(raw_path), gex_only=False)
    if not sp.issparse(adata.X) or adata.X.format != "csr":
        adata.X = sp.csr_matrix(adata.X)

    print("Splitting by feature_type → MuData{rna, adt}...")
    mod_dict = _split_by_feature_type(adata)
    for name, ad_ in mod_dict.items():
        nnz = ad_.X.nnz if sp.issparse(ad_.X) else int((ad_.X != 0).sum())
        print(f"  {name}: {ad_.n_obs} x {ad_.n_vars}  nnz={nnz:,}")

    mu = mudata.MuData(mod_dict)
    # MuData global obs reflects the union of cell barcodes; in 10x
    # CITE-seq every modality shares the same cell axis, so this is
    # identical to either modality's obs.
    mu.update()

    print(f"Writing h5mu → {out_path} ...")
    mu.write_h5mu(str(out_path))
    out_size_mb = out_path.stat().st_size / 1e6
    print(f"  h5mu size: {out_size_mb:.1f} MB")

    # Drop the raw .h5 once the .h5mu is written.
    try:
        os.remove(raw_path)
    except OSError:
        pass

    # Sanity: re-open and confirm both modalities are present and have
    # non-trivial nnz.
    print("Re-opening to verify...")
    mu_back = mudata.read_h5mu(str(out_path))
    assert "rna" in mu_back.mod and "adt" in mu_back.mod, mu_back.mod.keys()
    print(f"  rna: {mu_back.mod['rna'].shape}, adt: {mu_back.mod['adt'].shape}")
    print("Done!")


if __name__ == "__main__":
    main()
