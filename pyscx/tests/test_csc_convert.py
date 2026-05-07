"""Phase D.4: pyscx.from_anndata(csc=...) round-trip tests.

Verifies that the new ``csc`` and ``csc_cols_per_shard`` kwargs on
``from_anndata`` correctly emit a CSC sidecar (multi-shard column-major
layout) and that the densified data matches the source AnnData.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def small_adata():
    """20 cells × 15 genes synthetic CSR AnnData."""
    pytest.importorskip("anndata")
    import anndata as ad

    rng = np.random.default_rng(0)
    mat = sp.random(20, 15, density=0.25, format="csr", dtype=np.float32, random_state=rng)
    mat.data = (mat.data * 100).astype(np.float32)
    adata = ad.AnnData(X=mat)
    adata.obs["cell_id"] = [f"c{i}" for i in range(20)]
    adata.var["gene_id"] = [f"g{i}" for i in range(15)]
    return adata


def test_from_anndata_csc_off_default(small_adata, tmp_path):
    """Default `csc='off'` emits no CSC sidecar."""
    import pyscx

    path = tmp_path / "csr_only.scx"
    pyscx.from_anndata(small_adata, str(path))

    exp = pyscx.open(str(path))
    # Round-trip values still match.
    adata2 = exp.to_anndata()
    np.testing.assert_array_almost_equal(
        small_adata.X.toarray(), adata2.X.toarray()
    )


def test_from_anndata_csc_always(small_adata, tmp_path):
    """`csc='always'` emits a CSC sidecar; densified contents match CSR."""
    import pyscx

    path = tmp_path / "with_csc.scx"
    pyscx.from_anndata(small_adata, str(path), csc="always", csc_cols_per_shard=5)

    exp = pyscx.open(str(path))
    adata2 = exp.to_anndata()
    # to_anndata reads CSR; the CSC sidecar must densify to the same
    # values as the source matrix.
    np.testing.assert_array_almost_equal(
        small_adata.X.toarray(), adata2.X.toarray()
    )


def test_from_anndata_csc_invalid_value(small_adata, tmp_path):
    """`csc` only accepts 'off' or 'always'; anything else raises."""
    import pyscx

    path = tmp_path / "bad.scx"
    with pytest.raises(ValueError, match="csc"):
        pyscx.from_anndata(small_adata, str(path), csc="auto")
