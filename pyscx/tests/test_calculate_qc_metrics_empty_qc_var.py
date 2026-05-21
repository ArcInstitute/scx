"""`pyscx.accel.calculate_qc_metrics`
emits a UserWarning when a user-supplied `qc_vars` column matches zero
genes (silent-zero-fill site at `preprocessing.rs:493`), AND emits a
CELLxGENE-Census-shape-aware UserWarning upfront when var_names lacks
MT genes but `var['feature_name']` has them (the SKILL Tier 2 recipe
trips both, but option 2 suppresses option 1's warning so the user
sees exactly one).

Covers options 1 and 2 from the Tier 2 user report.
"""
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _adata_cellxgene_shape(n_mt_in_feature_name=8, n_extra=600):
    """Build an AnnData mimicking CELLxGENE Census layout: integer-string
    `var_names` and gene symbols stashed under `var['feature_name']`.
    Returns adata with `var['mt']` populated incorrectly (built from
    `var_names`) — i.e. an all-False mask — which is the exact silent-
    wrong-answer mode B1 describes."""
    n_vars = n_mt_in_feature_name + n_extra
    n_obs = 6
    rng = np.random.default_rng(0)
    x = sp.csr_matrix(
        rng.integers(0, 50, size=(n_obs, n_vars)).astype(np.float32)
    )
    var_names = [str(i) for i in range(n_vars)]  # integer-string indexes
    feature_name = (
        [f"MT-{i}" for i in range(n_mt_in_feature_name)]
        + [f"GENE_{i}" for i in range(n_extra)]
    )
    var = pd.DataFrame(
        {"feature_name": feature_name},
        index=pd.Index(var_names),
    )
    adata = ad.AnnData(X=x, var=var)
    # The wrong-on-CELLxGENE idiom: builds `mt` from var_names, all-False.
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    assert adata.var["mt"].sum() == 0
    return adata


def _adata_plain_shape(n_vars=600):
    """Plain anndata with no `feature_name` column, no MT genes in
    var_names. Used to exercise option 1's generic empty-mask warning."""
    n_obs = 6
    rng = np.random.default_rng(1)
    x = sp.csr_matrix(
        rng.integers(0, 50, size=(n_obs, n_vars)).astype(np.float32)
    )
    var = pd.DataFrame(index=pd.Index([f"GENE_{i}" for i in range(n_vars)]))
    adata = ad.AnnData(X=x, var=var)
    adata.var["ribo"] = False  # an all-False user-supplied mask
    return adata


def test_option1_warns_when_qc_var_mask_matches_zero_genes():
    """Generic empty-mask warning (option 1) fires for any qc_var, not
    just `mt`. CELLxGENE-shape detector (option 2) is NOT involved."""
    from pyscx import accel
    adata = _adata_plain_shape()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.calculate_qc_metrics(adata, qc_vars=["ribo"])
    matches = [
        w for w in ws
        if "qc_var 'ribo' mask matches 0 genes" in str(w.message)
    ]
    assert matches, (
        f"Expected an empty-mask UserWarning naming 'ribo'. "
        f"Got: {[str(w.message) for w in ws]}"
    )
    assert issubclass(matches[0].category, UserWarning)


def test_option2_warns_upfront_for_cellxgene_shape_with_mt_qc_var():
    """CELLxGENE shape detector (option 2): var_names has no MT, but
    `var['feature_name']` does, AND the user passed `qc_vars=['mt']`.
    The warning should name `feature_name` as the right source."""
    from pyscx import accel
    adata = _adata_cellxgene_shape(n_mt_in_feature_name=8)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.calculate_qc_metrics(adata, qc_vars=["mt"])
    matches = [
        w for w in ws
        if "feature_name" in str(w.message) and "CELLxGENE" in str(w.message)
    ]
    assert matches, (
        f"Expected a CELLxGENE-shape UserWarning naming feature_name. "
        f"Got: {[str(w.message) for w in ws]}"
    )
    assert issubclass(matches[0].category, UserWarning)


def test_option2_suppresses_option1_on_cellxgene_shape():
    """When option 2's upfront warning fires for `mt`, the generic
    empty-mask warning (option 1) should NOT also fire for the same
    qc_var — same root cause, one warning suffices."""
    from pyscx import accel
    adata = _adata_cellxgene_shape(n_mt_in_feature_name=8)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.calculate_qc_metrics(adata, qc_vars=["mt"])
    empty_mask_warnings = [
        w for w in ws
        if "qc_var 'mt' mask matches 0 genes" in str(w.message)
    ]
    assert not empty_mask_warnings, (
        "Did NOT expect the option-1 empty-mask warning when option 2 "
        "already fired for the same `mt` qc_var. "
        f"Got: {[str(w.message) for w in empty_mask_warnings]}"
    )


def test_option1_still_fires_for_non_mt_qc_var_under_cellxgene_shape():
    """The option-2 suppression is keyed strictly to `mt`. A different
    empty qc_var (`ribo`) on the same CELLxGENE-shape adata must still
    trigger the generic empty-mask warning."""
    from pyscx import accel
    adata = _adata_cellxgene_shape(n_mt_in_feature_name=8)
    adata.var["ribo"] = False
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"])
    ribo_warnings = [
        w for w in ws
        if "qc_var 'ribo' mask matches 0 genes" in str(w.message)
    ]
    assert ribo_warnings, (
        "Expected the option-1 empty-mask warning for `ribo` even though "
        "option 2 fired for `mt`. "
        f"Got: {[str(w.message) for w in ws]}"
    )


def test_no_option1_warning_when_mask_matches_some_genes():
    """Sanity: the empty-mask warning must NOT fire when the user-
    supplied mask matches at least one gene."""
    from pyscx import accel
    adata = _adata_plain_shape()
    # tag a single gene
    adata.var["ribo"] = adata.var_names == "GENE_0"
    assert adata.var["ribo"].sum() == 1
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.calculate_qc_metrics(adata, qc_vars=["ribo"])
    empty_mask_warnings = [
        w for w in ws if "mask matches 0 genes" in str(w.message)
    ]
    assert not empty_mask_warnings, (
        f"Did NOT expect an empty-mask warning when the mask matches "
        f"1 gene. Got: {[str(w.message) for w in empty_mask_warnings]}"
    )


def test_no_option2_warning_when_feature_name_absent():
    """Option 2 is gated on `var['feature_name']` being present with
    MT-prefixed entries. Without that column, only option 1 (the generic
    empty-mask warning) should fire."""
    from pyscx import accel
    adata = _adata_plain_shape()
    adata.var["mt"] = False  # empty mt mask, but no feature_name column
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.calculate_qc_metrics(adata, qc_vars=["mt"])
    # Option 2's distinctive signature: it tells the user feature_name HAS
    # N MT-/mt- entries. The generic option-1 message references CELLxGENE
    # only as a hint about where MT symbols *can* live.
    option2_warnings = [
        w for w in ws
        if "var['feature_name'] has" in str(w.message)
        or "CELLxGENE Census layout" in str(w.message)
    ]
    assert not option2_warnings, (
        "Option 2's CELLxGENE-shape warning should NOT fire without "
        f"a feature_name column. Got: {[str(w.message) for w in option2_warnings]}"
    )
    # option 1 SHOULD still fire (mt mask is empty)
    option1_warnings = [
        w for w in ws if "qc_var 'mt' mask matches 0 genes" in str(w.message)
    ]
    assert option1_warnings, (
        "Option 1's empty-mask warning should still fire when feature_name "
        "is absent and the mt mask is empty. "
        f"Got: {[str(w.message) for w in ws]}"
    )


def test_option2_fix_path_no_warnings():
    """End-to-end: when the user follows option 2's advice and tags `mt`
    from `feature_name`, the mask is non-empty and neither warning fires."""
    from pyscx import accel
    adata = _adata_cellxgene_shape(n_mt_in_feature_name=8)
    # Correct CELLxGENE idiom — overrides the all-False mask in the fixture.
    adata.var["mt"] = adata.var["feature_name"].astype(str).str.startswith("MT-")
    assert adata.var["mt"].sum() == 8
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.calculate_qc_metrics(adata, qc_vars=["mt"])
    relevant = [
        w for w in ws
        if "mask matches 0 genes" in str(w.message)
        or "CELLxGENE" in str(w.message)
    ]
    assert not relevant, (
        f"Did NOT expect any qc-warning when the user follows option 2's "
        f"advice. Got: {[str(w.message) for w in relevant]}"
    )
    # And pct_counts_mt should actually be populated and non-zero in
    # at least some cells (the fixture uses uniform 0..50 random counts,
    # so pct_counts_mt is rarely literally 0 across all cells).
    assert "pct_counts_mt" in adata.obs.columns
    assert (adata.obs["pct_counts_mt"] > 0).any()
