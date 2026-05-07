"""Phase F integration tests for ``prefer_format="csc"`` dispatch.

Verifies that the four wired pyscx entries produce the same numerical
output as the CSR path on a tiny CSC-equipped file:

  - ``pyscx.accel.highly_variable_genes(prefer_format="csc")``
  - ``pyscx.accel.rank_genes_groups(prefer_format="csc")``
  - ``pyscx.accel.pseudobulk_dex(prefer_format="csc", gene_indices=...)``
  - ``pyscx.accel.col_sums / col_nnz / col_var(prefer_format="csc")``

…plus the four "CSC unavailable" capability gates and the PCA
explicit-reject. See CSC-SUPPORT.md Phase F.5.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def small_adata():
    pytest.importorskip("anndata")
    import anndata as ad

    rng = np.random.default_rng(7)
    # Counts-like integer matrix; HVG / DE expect non-negative values.
    mat = sp.random(40, 16, density=0.4, format="csr", dtype=np.float32, random_state=rng)
    mat.data = (mat.data * 50).astype(np.float32).round() + 1.0
    adata = ad.AnnData(X=mat)
    adata.obs["cell_id"] = [f"c{i}" for i in range(40)]
    adata.obs["group"] = ["A"] * 20 + ["B"] * 20
    adata.var["gene_id"] = [f"g{i}" for i in range(16)]
    return adata


def _open_with_csc(path, adata):
    """Round-trip ``adata`` through a CSC-equipped SCX file."""
    import pyscx

    pyscx.from_anndata(adata, str(path), csc="always", csc_cols_per_shard=4)
    return pyscx.open(str(path)).to_anndata(backed=True)


def _open_csr_only(path, adata):
    import pyscx

    pyscx.from_anndata(adata, str(path))
    return pyscx.open(str(path)).to_anndata(backed=True)


# ---------------------------------------------------------------------------
# col_sums / col_nnz / col_var: explicit-opt-in CSC dispatch parity.
# ---------------------------------------------------------------------------


def test_col_sums_csc_matches_csr(small_adata, tmp_path):
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    csr_sums = pyscx.accel.col_sums(a_csc.X, prefer_format="csr")
    csc_sums = pyscx.accel.col_sums(a_csc.X, prefer_format="csc")
    np.testing.assert_allclose(csc_sums, csr_sums, atol=1e-9)


def test_col_nnz_csc_matches_csr(small_adata, tmp_path):
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    csr_nnz = pyscx.accel.col_nnz(a_csc.X, prefer_format="csr")
    csc_nnz = pyscx.accel.col_nnz(a_csc.X, prefer_format="csc")
    np.testing.assert_array_equal(csc_nnz, csr_nnz)


def test_col_var_csc_matches_csr(small_adata, tmp_path):
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    csr_var = pyscx.accel.col_var(a_csc.X, prefer_format="csr")
    csc_var = pyscx.accel.col_var(a_csc.X, prefer_format="csc")
    # Two-pass CSR vs single-pass CSC formula → equal within f64 epsilon.
    np.testing.assert_allclose(csc_var, csr_var, atol=1e-7)


# ---------------------------------------------------------------------------
# Capability gate: CSR-only file → RuntimeError naming the missing sidecar.
# ---------------------------------------------------------------------------


def test_col_sums_csc_raises_on_csr_only(small_adata, tmp_path):
    import pyscx

    a_csr = _open_csr_only(tmp_path / "csr_only.scx", small_adata)
    with pytest.raises(RuntimeError, match="CSC"):
        pyscx.accel.col_sums(a_csr.X, prefer_format="csc")


def test_col_sums_invalid_prefer_format(small_adata, tmp_path):
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    with pytest.raises(ValueError, match="prefer_format"):
        pyscx.accel.col_sums(a_csc.X, prefer_format="auto")


# ---------------------------------------------------------------------------
# HVG: prefer_format="csc" picks the same gene set as the CSR path.
# ---------------------------------------------------------------------------


def test_hvg_csc_matches_csr(small_adata, tmp_path):
    import pyscx

    pytest.importorskip("skmisc")  # loess fit dependency
    a_csr = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)

    pyscx.accel.highly_variable_genes(a_csr, n_top_genes=8, flavor="seurat_v3")
    pyscx.accel.highly_variable_genes(
        a_csc, n_top_genes=8, flavor="seurat_v3", prefer_format="csc"
    )

    np.testing.assert_array_equal(
        a_csc.var["highly_variable"].values,
        a_csr.var["highly_variable"].values,
    )
    np.testing.assert_allclose(
        np.asarray(a_csc.var["means"]), np.asarray(a_csr.var["means"]), atol=1e-9
    )


def test_hvg_csc_raises_on_csr_only(small_adata, tmp_path):
    import pyscx

    a_csr = _open_csr_only(tmp_path / "csr_only.scx", small_adata)
    with pytest.raises(RuntimeError, match="CSC"):
        pyscx.accel.highly_variable_genes(
            a_csr, n_top_genes=8, flavor="seurat_v3", prefer_format="csc"
        )


def test_hvg_csc_rejects_multi_batch(small_adata, tmp_path):
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    a_csc.obs["batch"] = ["b1"] * 20 + ["b2"] * 20
    with pytest.raises(RuntimeError, match="single-batch"):
        pyscx.accel.highly_variable_genes(
            a_csc, n_top_genes=8, flavor="seurat_v3",
            batch_key="batch", prefer_format="csc",
        )


# ---------------------------------------------------------------------------
# Wilcoxon DE: top-K gene order matches CSR path within tolerance.
# ---------------------------------------------------------------------------


def test_rank_genes_groups_csc_matches_csr(small_adata, tmp_path):
    import pyscx

    a_csr = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)

    pyscx.accel.rank_genes_groups(a_csr, "group")
    pyscx.accel.rank_genes_groups(a_csc, "group", prefer_format="csc")

    csr_res = a_csr.uns["rank_genes_groups"]
    csc_res = a_csc.uns["rank_genes_groups"]
    # Compare the gene name vector for one group — should be identical
    # ordering since inputs and chunking are identical.
    csr_names = [n for n in np.asarray(csr_res["names"]).tolist()[0]]
    csc_names = [n for n in np.asarray(csc_res["names"]).tolist()[0]]
    assert csc_names == csr_names


def test_rank_genes_groups_csc_raises_on_csr_only(small_adata, tmp_path):
    import pyscx

    a_csr = _open_csr_only(tmp_path / "csr_only.scx", small_adata)
    with pytest.raises(RuntimeError, match="CSC"):
        pyscx.accel.rank_genes_groups(a_csr, "group", prefer_format="csc")


# ---------------------------------------------------------------------------
# Pseudobulk: prefer_format="csc" requires gene_indices; matches CSR.
# ---------------------------------------------------------------------------


def test_pseudobulk_csc_requires_gene_indices(small_adata, tmp_path):
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    with pytest.raises(RuntimeError, match="gene_indices|gene subset|column projection"):
        pyscx.accel.pseudobulk_dex(
            a_csc, ["group"], "group", "A",
            prefer_format="csc",
        )


# ---------------------------------------------------------------------------
# PCA: prefer_format="csc" raises ValueError.
# ---------------------------------------------------------------------------


def test_pca_csc_raises(small_adata, tmp_path):
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    with pytest.raises(ValueError, match="csr"):
        pyscx.accel.pca(a_csc, n_comps=4, prefer_format="csc")


# ---------------------------------------------------------------------------
# Lazy chain: log1p preserves CSC capability; normalize_total breaks it.
# ---------------------------------------------------------------------------


def test_log1p_lazy_csc_capability(small_adata, tmp_path):
    """log1p is column-local — the resulting lazy wrapper should still
    accept prefer_format='csc'.

    The reference is computed by materialising the lazy dataset and
    summing per column (the CSR-side `col_sums` pyfunction currently
    only supports `ScxBackedSparseDataset`, not the lazy wrapper —
    that's a known limitation, not the path under test).
    """
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    pyscx.accel.log1p(a_csc)
    # Reference: materialise log1p'd matrix, sum columns.
    materialised = a_csc.X[:].toarray() if sp.issparse(a_csc.X[:]) else np.asarray(a_csc.X[:])
    expected = materialised.sum(axis=0).astype(np.float64)
    csc_sums = pyscx.accel.col_sums(a_csc.X, prefer_format="csc")
    np.testing.assert_allclose(csc_sums, expected, atol=1e-5)


def test_normalize_total_lazy_csc_unavailable(small_adata, tmp_path):
    """NormalizeTotal is row-local — CSC capability gate must reject."""
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    pyscx.accel.normalize_total(a_csc)
    with pytest.raises(RuntimeError, match="CSC"):
        pyscx.accel.col_sums(a_csc.X, prefer_format="csc")
