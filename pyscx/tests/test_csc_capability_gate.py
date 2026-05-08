"""Capability-gate sanity tests for the CSC dispatch.

The Rust-side `as_column_source()` method is internal (not exposed to
Python), so this file verifies the user-visible properties that the
gate is meant to protect:

1. A CSC-equipped file (built via `from_anndata(csc='always')`)
   round-trips cleanly through the CSR path. This is a smoke test that
   the `backed_csc` field on `ScxBackedSparseDataset` doesn't regress
   non-CSC consumers.
2. A CSR-only file behaves identically — the optional
   `BackedCscReader` field defaults to `None` and doesn't affect any
   existing read paths.

Full matrix tests on the gate (Some vs None for each branch — CSC +
no transforms / CSC + Log1p / CSC + NormalizeTotal / CSC + row deletion
/ no CSC sidecar) live in the Rust-side scx-format tests
(`backed_csc_*` and the trait independence test in `shard_source.rs`).
The pyscx-side gate logic is small enough that the Rust unit coverage
plus the consumer-integration tests in `test_csc_dispatch.py` suffice.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def small_adata():
    pytest.importorskip("anndata")
    import anndata as ad

    rng = np.random.default_rng(0)
    mat = sp.random(15, 10, density=0.3, format="csr", dtype=np.float32, random_state=rng)
    mat.data = (mat.data * 100).astype(np.float32)
    adata = ad.AnnData(X=mat)
    adata.obs["cell_id"] = [f"c{i}" for i in range(15)]
    adata.var["gene_id"] = [f"g{i}" for i in range(10)]
    return adata


def test_csc_equipped_file_csr_path_unchanged(small_adata, tmp_path):
    """Round-trip a CSC-equipped file via the CSR path — should be
    bit-identical to a CSR-only round-trip.

    Ensures the Phase E `backed_csc` field doesn't perturb existing
    CSR readers.
    """
    import pyscx

    csr_only = tmp_path / "csr_only.scx"
    with_csc = tmp_path / "with_csc.scx"

    pyscx.from_anndata(small_adata, str(csr_only))
    pyscx.from_anndata(small_adata, str(with_csc), csc="always", csc_cols_per_shard=4)

    # Both files densify to the same source matrix.
    a_csr = pyscx.open(str(csr_only)).to_anndata()
    a_csc = pyscx.open(str(with_csc)).to_anndata()
    np.testing.assert_array_equal(
        a_csr.X.toarray() if sp.issparse(a_csr.X) else a_csr.X,
        a_csc.X.toarray() if sp.issparse(a_csc.X) else a_csc.X,
    )


def test_csr_only_file_unaffected(small_adata, tmp_path):
    """A CSR-only file (no CSC sidecar) round-trips cleanly — confirms
    the optional `backed_csc` field defaults to None and doesn't break
    any existing reader path.
    """
    import pyscx

    path = tmp_path / "csr_only.scx"
    pyscx.from_anndata(small_adata, str(path))

    exp = pyscx.open(str(path))
    adata2 = exp.to_anndata()
    np.testing.assert_array_almost_equal(
        small_adata.X.toarray(),
        adata2.X.toarray() if sp.issparse(adata2.X) else adata2.X,
    )
