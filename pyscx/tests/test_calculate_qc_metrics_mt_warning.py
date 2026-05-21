"""`pyscx.accel.calculate_qc_metrics` emits a UserWarning when
`qc_vars=None` AND adata.var_names contains 5+ MT-/mt- prefixed gene
symbols, so docs-skimmers don't hit `KeyError: 'pct_counts_mt'` when
they later filter on `adata.obs["pct_counts_mt"]`.
"""
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _adata_with_mt_genes(n_mt=8, n_extra=600):
    """Build an AnnData where the first `n_mt` var_names start with
    `MT-` (human convention). `n_extra` defaults large enough that
    scanpy's downstream `percent_top=[50, 100, 200, 500]` check does
    not trip on the small-fixture path (`pct_counts_mt` is what we're
    here for, not the `percent_top` machinery)."""
    n_vars = n_mt + n_extra
    n_obs = 6
    x = sp.csr_matrix(
        np.random.default_rng(0).integers(0, 50, size=(n_obs, n_vars)).astype(np.float32)
    )
    names = [f"MT-{i}" for i in range(n_mt)] + [f"GENE_{i}" for i in range(n_extra)]
    var = pd.DataFrame(index=pd.Index(names))
    return ad.AnnData(X=x, var=var)


def _adata_without_mt_genes(n_vars=600):
    n_obs = 6
    x = sp.csr_matrix(
        np.random.default_rng(1).integers(0, 50, size=(n_obs, n_vars)).astype(np.float32)
    )
    var = pd.DataFrame(index=pd.Index([f"GENE_{i}" for i in range(n_vars)]))
    return ad.AnnData(X=x, var=var)


def test_warns_when_mt_genes_present_and_qc_vars_none():
    from pyscx import accel
    adata = _adata_with_mt_genes(n_mt=8)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.calculate_qc_metrics(adata)  # qc_vars=None default
    matches = [w for w in ws if "pct_counts_mt" in str(w.message)]
    assert matches, (
        "Expected a UserWarning about pct_counts_mt not being computed when "
        f"MT-* genes are present and qc_vars is None. Got: {[str(w.message) for w in ws]}"
    )
    assert issubclass(matches[0].category, UserWarning)


def test_warns_for_mouse_mt_prefix():
    """Mouse symbols use `mt-` (lowercase). Same warning should fire."""
    from pyscx import accel
    adata = _adata_with_mt_genes(n_mt=8)
    adata.var.index = pd.Index(
        [n.replace("MT-", "mt-") for n in adata.var_names]
    )
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.calculate_qc_metrics(adata)
    assert any("pct_counts_mt" in str(w.message) for w in ws)


def test_no_warning_when_qc_vars_provided():
    from pyscx import accel
    adata = _adata_with_mt_genes(n_mt=8)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.calculate_qc_metrics(adata, qc_vars=["mt"])
    matches = [w for w in ws if "pct_counts_mt" in str(w.message)]
    assert not matches, (
        "Did NOT expect the pct_counts_mt advisory when qc_vars is set. "
        f"Got: {[str(w.message) for w in matches]}"
    )


def test_no_warning_when_no_mt_genes():
    """The control case: an AnnData with no MT-* genes should not warn."""
    from pyscx import accel
    adata = _adata_without_mt_genes()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.calculate_qc_metrics(adata)
    assert not any("pct_counts_mt" in str(w.message) for w in ws)


def test_no_warning_below_threshold():
    """Threshold is 5+ MT-* prefixed genes — fewer should not warn."""
    from pyscx import accel
    adata = _adata_with_mt_genes(n_mt=4)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.calculate_qc_metrics(adata)
    assert not any("pct_counts_mt" in str(w.message) for w in ws)
