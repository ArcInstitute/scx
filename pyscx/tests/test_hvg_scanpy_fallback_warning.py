"""B2-2026-05-20-Tier2 option 1: `pyscx.accel.highly_variable_genes`
emits a UserWarning when it silently delegates to
`scanpy.pp.highly_variable_genes` because `adata.X` is regular
scipy/dense rather than an `ScxBackedSparseDataset` /
`ScxLazyTransformedDataset`. The user report observed that the Tier 2
recipe hits this fallback via `exp.to_anndata()` (eager) and is then
exposed to scanpy's LOESS singularity / pd.cut bin-edge failures on
real Census data — with no indication that scx-native HVG was skipped.
"""
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _adata_seurat_compatible(n_obs=200, n_vars=500, seed=0):
    """Reasonably-sized AnnData where scanpy's seurat / cell_ranger flavors
    succeed (no LOESS singularity at this scale). Used to verify the
    warning fires *and* scanpy completes — i.e. the fallback still works.
    """
    rng = np.random.default_rng(seed)
    x = sp.csr_matrix(
        rng.integers(0, 50, size=(n_obs, n_vars)).astype(np.float32)
    )
    var = pd.DataFrame(index=pd.Index([f"GENE_{i}" for i in range(n_vars)]))
    return ad.AnnData(X=x, var=var)


def test_warns_when_scipy_csr_delegates_to_scanpy_seurat():
    """seurat flavor on scipy CSR delegates to scanpy. Warning must fire
    and the call must still succeed."""
    from pyscx import accel
    adata = _adata_seurat_compatible()
    # Normalize + log1p so seurat flavor (which expects log-normalized data)
    # has reasonable input. This routes through scanpy too — that's not
    # what we're testing here, only the HVG call.
    import scanpy as sc
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.highly_variable_genes(adata, n_top_genes=100, flavor="seurat")
    matches = [
        w for w in ws
        if "scanpy.pp.highly_variable_genes" in str(w.message)
        and "scipy" in str(w.message)
    ]
    assert matches, (
        f"Expected a UserWarning about scanpy fallback for scipy/dense X. "
        f"Got: {[str(w.message) for w in ws]}"
    )
    assert issubclass(matches[0].category, UserWarning)
    # Sanity: the scanpy fallback still populated `highly_variable`.
    assert "highly_variable" in adata.var.columns


def test_warning_names_the_flavor():
    """The warning text should include the flavor the user requested,
    so an `seurat_v3` LOESS-singularity user gets a hint that they were
    on the scanpy path."""
    from pyscx import accel
    adata = _adata_seurat_compatible()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        try:
            accel.highly_variable_genes(adata, n_top_genes=50, flavor="seurat_v3")
        except Exception:
            # seurat_v3 may fail on tiny data — we only care about the
            # warning, which must fire BEFORE the scanpy call attempts the
            # LOESS fit.
            pass
    matches = [
        w for w in ws
        if "scanpy.pp.highly_variable_genes" in str(w.message)
        and "seurat_v3" in str(w.message)
    ]
    assert matches, (
        f"Expected the warning to name flavor='seurat_v3'. "
        f"Got: {[str(w.message) for w in ws]}"
    )


def test_warning_mentions_the_scx_native_workaround():
    """The warning should tell the user how to opt into the scx-native
    path — `to_anndata(backed=True)` or keeping the source as a lazy
    dataset."""
    from pyscx import accel
    adata = _adata_seurat_compatible()
    import scanpy as sc
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.highly_variable_genes(adata, n_top_genes=100, flavor="seurat")
    matches = [
        w for w in ws
        if "scanpy.pp.highly_variable_genes" in str(w.message)
    ]
    assert matches
    text = str(matches[0].message)
    # The user needs to see at least one of the recommended workarounds.
    assert "backed=True" in text or "ScxLazyTransformedDataset" in text, (
        f"Warning should mention the scx-native workaround. Got: {text!r}"
    )


def test_warning_text_calls_out_known_fragility():
    """The fragility paragraph (LOESS / pd.cut)
    should appear when the user calls a flavor that actually traverses
    those code paths — `seurat_v3` / `seurat_v3_paper` / `cell_ranger`.
    Verified here on `seurat_v3` (the failure mode the Tier 2 user
    actually hit). Pre-N4 this test ran with `flavor="seurat"` and
    incorrectly asserted the fragility text appeared there too — the
    warning re-fired its own recommended workaround at the user."""
    from pyscx import accel
    adata = _adata_seurat_compatible()
    # seurat_v3 expects raw counts (not log-normalised), so do NOT
    # normalize_total/log1p first. The scanpy call may then fail on
    # this tiny synthetic data — we only care about the warning, which
    # fires BEFORE the scanpy call attempts the LOESS fit.
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        try:
            accel.highly_variable_genes(adata, n_top_genes=50, flavor="seurat_v3")
        except Exception:
            pass
    matches = [
        w for w in ws
        if "scanpy.pp.highly_variable_genes" in str(w.message)
    ]
    assert matches
    text = str(matches[0].message)
    assert "seurat_v3" in text and "cell_ranger" in text, (
        f"Warning should reference both fragile flavors. Got: {text!r}"
    )
    assert "filter_genes" in text or "flavor=\"seurat\"" in text, (
        f"Warning should reference the recommended workaround. Got: {text!r}"
    )


def test_warning_text_omits_fragility_for_seurat_flavor():
    """The warning text MUST NOT mention the LOESS
    / pd.cut fragility when the user is already on `flavor="seurat"`
    (the flavor that the fragility paragraph itself recommends as a
    workaround). The core delegation message still fires — silent
    fallback is still a finding — but the irrelevant tail is suppressed."""
    from pyscx import accel
    adata = _adata_seurat_compatible()
    import scanpy as sc
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        accel.highly_variable_genes(adata, n_top_genes=100, flavor="seurat")
    matches = [
        w for w in ws
        if "scanpy.pp.highly_variable_genes" in str(w.message)
    ]
    assert matches, (
        "Core delegation warning must still fire for seurat flavor — "
        "silent-fallback observability is the design intent. "
        f"Got: {[str(w.message) for w in ws]}"
    )
    text = str(matches[0].message)
    # Substrings unique to the fragility tail. `cell_ranger` is the
    # tightest signal because it appears nowhere else in the message.
    assert "cell_ranger" not in text, (
        f"Fragility paragraph (LOESS / pd.cut) should be suppressed for "
        f"flavor='seurat'. Got: {text!r}"
    )
    assert "LOESS" not in text and "pd.cut" not in text, (
        f"Fragility paragraph should be suppressed for flavor='seurat'. "
        f"Got: {text!r}"
    )
    # The user still sees the scx-native workaround (this is what
    # `test_warning_mentions_the_scx_native_workaround` also asserts —
    # repeated here so a future re-refactor that breaks both fails
    # both, not silently one).
    assert "backed=True" in text or "ScxLazyTransformedDataset" in text
