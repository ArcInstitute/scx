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


def _adata_cellxgene_shape_no_user_mask(n_mt_in_feature_name=8, n_extra=600):
    """CELLxGENE Census layout: integer-string `var_names`, gene symbols
    in `var['feature_name']`, AND the user has NOT tagged `var['mt']`
    (i.e. the natural state right after `exp.query(...).to_anndata()`).
    Used to exercise N1's new qc_vars=None + feature_name branch."""
    n_vars = n_mt_in_feature_name + n_extra
    n_obs = 6
    rng = np.random.default_rng(2)
    x = sp.csr_matrix(rng.integers(0, 50, size=(n_obs, n_vars)).astype(np.float32))
    var_names = [str(i) for i in range(n_vars)]
    feature_name = [f"MT-{i}" for i in range(n_mt_in_feature_name)] + [
        f"GENE_{i}" for i in range(n_extra)
    ]
    var = pd.DataFrame({"feature_name": feature_name}, index=pd.Index(var_names))
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


# N1-2026-05-21-Tier2: the qc_vars=None branch also fires when var_names
# lacks MT prefixes but feature_name has them — the CELLxGENE Census
# layout. Without this, `accel.calculate_qc_metrics(adata)` on raw Census
# silently produces no `pct_counts_mt` and no warning.
def test_warns_for_cellxgene_shape_with_qc_vars_none():
    from pyscx import accel
    adata = _adata_cellxgene_shape_no_user_mask(n_mt_in_feature_name=8)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.calculate_qc_metrics(adata)  # qc_vars=None default
    matches = [
        w for w in ws
        if "feature_name" in str(w.message) and "CELLxGENE" in str(w.message)
    ]
    assert matches, (
        "Expected a UserWarning naming feature_name + CELLxGENE when "
        "qc_vars is None and var['feature_name'] has MT-* symbols. "
        f"Got: {[str(w.message) for w in ws]}"
    )
    assert issubclass(matches[0].category, UserWarning)
    assert "pct_counts_mt" in str(matches[0].message)


def test_var_names_path_wins_when_both_axes_have_mt():
    """When both var_names AND var['feature_name'] have MT prefixes, the
    existing var_names path should fire (and the new feature_name path
    should not) — else-if precedence keeps the message stable on fixtures
    that already exercise the var_names branch."""
    from pyscx import accel
    adata = _adata_with_mt_genes(n_mt=8)
    # Also populate feature_name so both axes carry MT prefixes.
    adata.var["feature_name"] = list(adata.var_names)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.calculate_qc_metrics(adata)
    var_names_warnings = [
        w for w in ws if "in adata.var_names" in str(w.message)
    ]
    feature_name_warnings = [
        w for w in ws
        if "var['feature_name'] has" in str(w.message)
        and "qc_vars` is None" in str(w.message)
    ]
    assert var_names_warnings, (
        "var_names path should fire (precedence). "
        f"Got: {[str(w.message) for w in ws]}"
    )
    assert not feature_name_warnings, (
        "feature_name path must NOT fire when var_names already triggered. "
        f"Got: {[str(w.message) for w in feature_name_warnings]}"
    )


def test_no_qc_vars_none_warning_when_feature_name_lacks_mt():
    """Control case for N1: integer-string var_names AND a feature_name
    column with NO MT prefixes → neither qc_vars=None branch fires."""
    from pyscx import accel
    adata = _adata_cellxgene_shape_no_user_mask(n_mt_in_feature_name=0)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.calculate_qc_metrics(adata)
    assert not any("pct_counts_mt" in str(w.message) for w in ws), (
        f"Did NOT expect any pct_counts_mt advisory. "
        f"Got: {[str(w.message) for w in ws]}"
    )
