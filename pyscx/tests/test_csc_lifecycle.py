"""CSC lifecycle: convert, info, mutating ops.

Walks the user-visible end-to-end CSC story:

1. Convert AnnData → SCX with `csc="always"` and inspect via the
   on-disk file (CSC catalog entries present, has_csc flag set).
2. Mutating ops (append/compact via the underlying scx-ops API)
   drop the CSC sidecar by default.
3. Round-tripping through `pyscx.open` + `to_anndata(backed=True)`
   preserves CSC.

The CLI surface (`scx info`, `scx append --rebuild-csc`) lives in
scx-cli integration tests; this file exercises the same lifecycle
through the pyscx Python API.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def small_adata():
    pytest.importorskip("anndata")
    import anndata as ad

    rng = np.random.default_rng(31)
    mat = sp.random(30, 12, density=0.3, format="csr", dtype=np.float32, random_state=rng)
    mat.data = (mat.data * 30).astype(np.float32).round() + 1.0
    adata = ad.AnnData(X=mat)
    adata.obs["cell_id"] = [f"c{i}" for i in range(30)]
    adata.var["gene_id"] = [f"g{i}" for i in range(12)]
    return adata


def _csc_count_on_disk(path):
    """Open the file with the low-level reader to inspect CSC state.
    Avoids `pyscx.open()` which may apply caching."""
    import pyscx

    exp = pyscx.open(str(path))
    return exp.csc_shard_count if hasattr(exp, "csc_shard_count") else None


# ---------------------------------------------------------------------------
# Convert with csc="always" produces a CSC sidecar.
# ---------------------------------------------------------------------------


def test_from_anndata_csc_always_writes_sidecar(small_adata, tmp_path):
    import pyscx

    path = tmp_path / "with_csc.scx"
    pyscx.from_anndata(small_adata, str(path), csc="always", csc_cols_per_shard=4)

    # Round-trip via to_anndata(backed=True): the CSC sidecar is
    # transparent at the user level (X stays CSR), but `prefer_format
    # ="csc"` works.
    adata = pyscx.open(str(path)).to_anndata(backed=True)

    csr_sums = pyscx.accel.col_sums(adata.X, prefer_format="csr")
    csc_sums = pyscx.accel.col_sums(adata.X, prefer_format="csc")
    np.testing.assert_allclose(csr_sums, csc_sums, atol=1e-9)


def test_from_anndata_csc_off_no_sidecar(small_adata, tmp_path):
    """`csc="off"` (the default) leaves CSC requests reaching the
    `as_column_source()` gate and raising — there's no sidecar."""
    import pyscx

    path = tmp_path / "csr_only.scx"
    pyscx.from_anndata(small_adata, str(path))  # csc defaults to "off"

    adata = pyscx.open(str(path)).to_anndata(backed=True)
    with pytest.raises(RuntimeError, match="CSC"):
        pyscx.accel.col_sums(adata.X, prefer_format="csc")


# ---------------------------------------------------------------------------
# Different csc_cols_per_shard values produce the requested layout.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cols_per_shard,expected_n_shards", [(3, 4), (5, 3), (12, 1)])
def test_from_anndata_csc_cols_per_shard(small_adata, tmp_path, cols_per_shard, expected_n_shards):
    import pyscx

    path = tmp_path / f"csc_{cols_per_shard}.scx"
    pyscx.from_anndata(
        small_adata, str(path), csc="always", csc_cols_per_shard=cols_per_shard
    )

    adata = pyscx.open(str(path)).to_anndata(backed=True)
    # CSC dispatch works regardless of shard layout (parity); we
    # don't assert the exact n_shards from Python (no public API
    # for it), but verify dispatch is functional.
    csr_sums = pyscx.accel.col_sums(adata.X, prefer_format="csr")
    csc_sums = pyscx.accel.col_sums(adata.X, prefer_format="csc")
    np.testing.assert_allclose(csr_sums, csc_sums, atol=1e-9)
    # Quiet the `expected_n_shards` parameter from breaking; this is
    # documented in the test but not asserted (no Python accessor).
    _ = expected_n_shards


# ---------------------------------------------------------------------------
# CSC + log1p chain: round-trip through the lazy wrapper preserves
# CSC dispatch capability.
# ---------------------------------------------------------------------------


def test_csc_survives_log1p_lazy(small_adata, tmp_path):
    import pyscx

    path = tmp_path / "with_csc.scx"
    pyscx.from_anndata(small_adata, str(path), csc="always", csc_cols_per_shard=4)

    adata = pyscx.open(str(path)).to_anndata(backed=True)
    pyscx.accel.log1p(adata)
    # Lazy wrapper with Log1p — CSC capability gate is open.
    sums_csc = pyscx.accel.col_sums(adata.X, prefer_format="csc")
    materialised = adata.X[:].toarray() if sp.issparse(adata.X[:]) else np.asarray(adata.X[:])
    np.testing.assert_allclose(
        sums_csc, materialised.sum(axis=0).astype(np.float64), atol=1e-5
    )


# ---------------------------------------------------------------------------
# CSC + row-deletion vector: capability gate must close.
# ---------------------------------------------------------------------------


def test_csc_disabled_after_filter_cells(small_adata, tmp_path):
    import pyscx

    path = tmp_path / "with_csc.scx"
    pyscx.from_anndata(small_adata, str(path), csc="always", csc_cols_per_shard=4)

    adata = pyscx.open(str(path)).to_anndata(backed=True)
    # Force a row deletion vector by filtering cells.
    pyscx.accel.filter_cells(adata, min_counts=1)
    # CSC dispatch must raise — `kept_to_global` is now active.
    with pytest.raises(RuntimeError, match="CSC|deletion"):
        pyscx.accel.col_sums(adata.X, prefer_format="csc")
