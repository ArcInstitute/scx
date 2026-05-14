"""Multimodal inputs are rejected by `pyscx.merge` and `pyscx.compact`.

The Rust core (`scx_ops::merge` / `scx_ops::compact`) returns
`OpsError::MultimodalUnsupported`, which the pyscx error mapper routes
to `RuntimeError`. These tests pin the user-visible behaviour.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest


@pytest.fixture
def multimodal_scx_path():
    """Tiny CITE-seq SCX fixture written via pyscx.from_mudata."""
    mudata = pytest.importorskip("mudata")
    anndata = pytest.importorskip("anndata")
    import scipy.sparse as sp
    import pyscx

    rng = np.random.default_rng(0)
    n_obs, rna_n_vars, adt_n_vars = 32, 20, 6
    rna_dense = rng.poisson(lam=0.4, size=(n_obs, rna_n_vars)).astype(np.float32)
    adt_dense = rng.poisson(lam=0.4, size=(n_obs, adt_n_vars)).astype(np.float32)
    rna_ad = anndata.AnnData(X=sp.csr_matrix(rna_dense))
    rna_ad.var_names = [f"g{i}" for i in range(rna_n_vars)]
    adt_ad = anndata.AnnData(X=sp.csr_matrix(adt_dense))
    adt_ad.var_names = [f"a{i}" for i in range(adt_n_vars)]
    mu = mudata.MuData({"rna": rna_ad, "adt": adt_ad})
    mu.obs_names = [f"cell_{i}" for i in range(n_obs)]

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cite.scx")
        pyscx.from_mudata(mu, path)
        yield path, tmp


def test_compact_rejects_multimodal(multimodal_scx_path):
    import pyscx

    path, tmp = multimodal_scx_path
    out = os.path.join(tmp, "compacted.scx")
    with pytest.raises(RuntimeError, match="not yet supported for multimodal"):
        pyscx.compact(path, out)


def test_merge_rejects_multimodal(multimodal_scx_path):
    import pyscx

    path, tmp = multimodal_scx_path
    out = os.path.join(tmp, "merged.scx")
    with pytest.raises(RuntimeError, match="not yet supported for multimodal"):
        pyscx.merge([path, path], out)
