"""`pyscx.accel.highly_variable_genes` scanpy-delegation behavior.

As of the in-memory native-kernel routing, `flavor` in
{seurat_v3, seurat_v3_paper, seurat} runs the scx-native streaming kernel even
when `adata.X` is a regular scipy/dense matrix (it is wrapped in a single-shard
`ShardSource`). So those flavors no longer delegate to
`scanpy.pp.highly_variable_genes` and no longer emit the delegation warning —
and they inherit the per-batch LOESS-singularity tolerance that the backed path
has (the original Tier-2 crash).

Only flavors scx does not implement natively — today `cell_ranger` — still
delegate to scanpy on a scipy/dense X, and that delegation still emits a
one-shot UserWarning so the hand-off is observable.
"""
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _adata_compatible(n_obs=200, n_vars=500, seed=0):
    """A scipy-CSR AnnData large enough that scanpy's cell_ranger / seurat
    flavors succeed (no singularity at this scale)."""
    rng = np.random.default_rng(seed)
    x = sp.csr_matrix(
        rng.integers(0, 50, size=(n_obs, n_vars)).astype(np.float32)
    )
    var = pd.DataFrame(index=pd.Index([f"GENE_{i}" for i in range(n_vars)]))
    return ad.AnnData(X=x, var=var)


def test_warns_when_scipy_csr_delegates_to_scanpy_cell_ranger():
    """cell_ranger is not implemented natively, so a scipy/dense X delegates
    to scanpy. The delegation warning must fire and the call must succeed."""
    from pyscx import accel
    import scanpy as sc

    adata = _adata_compatible()
    # cell_ranger expects log-normalized data.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.highly_variable_genes(adata, n_top_genes=100, flavor="cell_ranger")
    matches = [
        w for w in ws
        if "scanpy.pp.highly_variable_genes" in str(w.message)
        and "cell_ranger" in str(w.message)
    ]
    assert matches, (
        f"Expected a UserWarning about scanpy delegation for cell_ranger. "
        f"Got: {[str(w.message) for w in ws]}"
    )
    assert issubclass(matches[0].category, UserWarning)
    # Sanity: the scanpy delegation still populated `highly_variable`.
    assert "highly_variable" in adata.var.columns


def test_warning_names_the_flavor():
    """The delegation warning text includes the flavor the user requested."""
    from pyscx import accel
    import scanpy as sc

    adata = _adata_compatible()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        try:
            accel.highly_variable_genes(adata, n_top_genes=50, flavor="cell_ranger")
        except Exception:
            # We only care about the warning, which fires before the scanpy call.
            pass
    matches = [
        w for w in ws
        if "scanpy.pp.highly_variable_genes" in str(w.message)
        and "cell_ranger" in str(w.message)
    ]
    assert matches, (
        f"Expected the warning to name flavor='cell_ranger'. "
        f"Got: {[str(w.message) for w in ws]}"
    )


def test_warning_text_calls_out_cell_ranger_fragility():
    """The fragility paragraph (pd.cut bin-edge) should appear for
    cell_ranger, and point at the scx-native flavors as the way out."""
    from pyscx import accel

    adata = _adata_compatible()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        try:
            accel.highly_variable_genes(adata, n_top_genes=50, flavor="cell_ranger")
        except Exception:
            pass
    matches = [
        w for w in ws
        if "scanpy.pp.highly_variable_genes" in str(w.message)
    ]
    assert matches
    text = str(matches[0].message)
    assert "cell_ranger" in text and "pd.cut" in text, (
        f"Warning should describe the cell_ranger fragility. Got: {text!r}"
    )
    assert "filter_genes" in text or 'flavor="seurat_v3"' in text, (
        f"Warning should reference the recommended workaround. Got: {text!r}"
    )


@pytest.mark.parametrize("flavor", ["seurat_v3", "seurat"])
def test_native_flavors_do_not_delegate_to_scanpy(flavor):
    """seurat_v3 / seurat now run the scx-native kernel on scipy/dense X, so
    NO scanpy-delegation warning is emitted and the call completes."""
    from pyscx import accel
    import scanpy as sc

    adata = _adata_compatible()
    if flavor == "seurat":
        # seurat flavor expects log-normalized data.
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.highly_variable_genes(adata, n_top_genes=100, flavor=flavor)
    delegation = [
        w for w in ws
        if "scanpy.pp.highly_variable_genes" in str(w.message)
    ]
    assert not delegation, (
        f"flavor={flavor!r} on scipy X must run native (no scanpy delegation). "
        f"Got: {[str(w.message) for w in ws]}"
    )
    assert "highly_variable" in adata.var.columns
    assert int(adata.var["highly_variable"].sum()) == 100
