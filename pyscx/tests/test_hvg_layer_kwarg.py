"""`pyscx.accel.highly_variable_genes(layer=name)` reads from
`adata.layers[name]` instead of `adata.X`.

The original bug was that the scanpy idiom
``adata.layers["counts"] = adata.X.copy(); normalize_total; log1p;
highly_variable_genes(flavor="seurat_v3", layer="counts")`` did not
work because pyscx didn't forward `layer=` through the binding —
forcing users to reorder the pipeline (run HVG BEFORE normalize/log1p).
"""
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _make_counts_adata(n_obs=200, n_vars=80, seed=0):
    rng = np.random.default_rng(seed)
    x = sp.csr_matrix(rng.integers(0, 50, size=(n_obs, n_vars)).astype(np.float32))
    return ad.AnnData(X=x)


def test_layer_kwarg_avoids_seurat_v3_warning():
    """With `layer="counts"` pointing at raw counts, seurat_v3's
    "expects raw count data but non-integers were found" warning must
    not fire even though X has been normalized + log1p'd."""
    from pyscx import accel
    adata = _make_counts_adata()
    adata.layers["counts"] = adata.X.copy()
    accel.normalize_total(adata, target_sum=1e4)
    accel.log1p(adata)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.highly_variable_genes(
            adata, n_top_genes=20, flavor="seurat_v3", layer="counts"
        )
    msgs = [str(w.message) for w in ws]
    non_integer_warnings = [m for m in msgs if "non-integers" in m or "raw count" in m]
    assert not non_integer_warnings, (
        "seurat_v3 should not warn when reading from a raw-counts layer; "
        f"got warnings: {non_integer_warnings}"
    )
    assert "highly_variable" in adata.var.columns
    assert int(adata.var["highly_variable"].sum()) == 20


def test_layer_kwarg_matches_running_before_normalize():
    """Sanity: running HVG with `layer="counts"` on the post-normalize
    AnnData should select the SAME genes as running HVG on a sibling
    AnnData BEFORE normalize_total / log1p (since both compute on the
    same raw counts)."""
    from pyscx import accel
    adata = _make_counts_adata()
    adata.layers["counts"] = adata.X.copy()

    # Sibling: HVG before normalize/log1p — the "old" idiom.
    sibling = adata.copy()
    accel.highly_variable_genes(sibling, n_top_genes=15, flavor="seurat_v3")
    expected = set(sibling.var_names[sibling.var["highly_variable"]])

    # Original: normalize+log1p, then HVG with layer=.
    accel.normalize_total(adata, target_sum=1e4)
    accel.log1p(adata)
    accel.highly_variable_genes(
        adata, n_top_genes=15, flavor="seurat_v3", layer="counts"
    )
    got = set(adata.var_names[adata.var["highly_variable"]])

    assert got == expected, (
        f"HVG selection drifted when using layer= on normalized X; "
        f"missing from layer-result: {expected - got}; "
        f"extra in layer-result: {got - expected}"
    )


def test_layer_default_none_still_uses_X():
    """Control: without `layer=`, the function still reads from adata.X."""
    from pyscx import accel
    adata = _make_counts_adata()
    # X holds raw counts; no layer. The seurat_v3 flavor should work fine.
    accel.highly_variable_genes(adata, n_top_genes=10, flavor="seurat_v3")
    assert "highly_variable" in adata.var.columns
    assert int(adata.var["highly_variable"].sum()) == 10


def test_prefer_csc_with_layer_raises():
    """`prefer_format="csc"` reads adata.X only (the CSC sidecar lives on X,
    not arbitrary layers). Combining it with `layer=` must error rather than
    silently computing on X. The rejection fires before device resolution, so
    it runs without a GPU."""
    from pyscx import accel
    adata = _make_counts_adata()
    adata.layers["counts"] = adata.X.copy()
    with pytest.raises(Exception, match="does not support layer"):
        accel.highly_variable_genes(
            adata,
            n_top_genes=10,
            flavor="seurat_v3",
            prefer_format="csc",
            layer="counts",
        )


def test_layer_on_a_backed_adata_reads_the_layer(tmp_dir):
    """`layer=` must work when the layer is an SCX handle, not just scipy.

    Every other test in this file uses an in-memory scipy AnnData, so nothing
    covered the case the kwarg exists for: a file opened backed, whose
    `adata.layers[name]` is an `ScxBackedLayerDataset` rather than a
    `ScxBackedSparseDataset`. The dispatch cast missed that type and fell
    through to `scipy.sparse.csr_matrix(handle)`, which raises
    `ValueError: unrecognized csr_matrix constructor input`.
    """
    import pyscx
    from pyscx import ScxBackedLayerDataset, accel

    adata = _make_counts_adata(n_obs=200, n_vars=120, seed=5)
    adata.var_names = [f"g{i}" for i in range(adata.n_vars)]
    adata.obs_names = [f"c{i}" for i in range(adata.n_obs)]
    adata.layers["counts"] = adata.X.copy()

    path = str(tmp_dir / "hvg_backed_layer.scx")
    pyscx.from_anndata(adata, path)
    backed = pyscx.open(path).to_anndata(backed=True)
    assert isinstance(backed.layers["counts"], ScxBackedLayerDataset), (
        "premise: the layer must be an SCX handle, or this is the scipy case again"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        accel.highly_variable_genes(
            backed, n_top_genes=10, flavor="seurat_v3", layer="counts", device="cpu"
        )

    # The layer is a byte copy of X, so the selection must match layer=None.
    reference = pyscx.open(path).to_anndata(backed=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        accel.highly_variable_genes(
            reference, n_top_genes=10, flavor="seurat_v3", device="cpu"
        )
    assert list(backed.var["highly_variable"]) == list(reference.var["highly_variable"])
