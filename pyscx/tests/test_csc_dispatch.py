"""Integration tests for ``prefer_format="csc"`` dispatch.

Verifies that the four wired pyscx entries produce the same numerical
output as the CSR path on a tiny CSC-equipped file:

  - ``pyscx.accel.highly_variable_genes(prefer_format="csc")``
  - ``pyscx.accel.rank_genes_groups(prefer_format="csc")``
  - ``pyscx.accel.pseudobulk_dex(prefer_format="csc", gene_indices=...)``
  - ``pyscx.accel.col_sums / col_nnz / col_var(prefer_format="csc")``

…plus the four "CSC unavailable" capability gates and the PCA
explicit-reject.
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

    # Pin the CSR arm explicitly: the file has a sidecar, so the "auto" default
    # (post Phase-2 §5.2) would otherwise route this side to CSC too, making the
    # comparison CSC-vs-CSC and gutting the CSR≈CSC guarantee (review feedback).
    pyscx.accel.rank_genes_groups(a_csr, "group", prefer_format="csr", device="cpu")
    pyscx.accel.rank_genes_groups(a_csc, "group", prefer_format="csc", device="cpu")

    assert a_csr.uns["scx_accel"]["rank_genes_groups"]["route"] == "cpu_csr"
    assert a_csc.uns["scx_accel"]["rank_genes_groups"]["route"] == "cpu_csc"
    csr_res = a_csr.uns["rank_genes_groups"]
    csc_res = a_csc.uns["rank_genes_groups"]
    # Genuine CSR-vs-CSC comparison: identical gene ordering AND scores.
    csr_names = [n for n in np.asarray(csr_res["names"]).tolist()[0]]
    csc_names = [n for n in np.asarray(csc_res["names"]).tolist()[0]]
    assert csc_names == csr_names
    np.testing.assert_allclose(
        np.asarray(csc_res["scores"]).tolist()[0],
        np.asarray(csr_res["scores"]).tolist()[0],
        atol=1e-9, rtol=1e-6,
    )


def test_rank_genes_groups_csc_raises_on_csr_only(small_adata, tmp_path):
    import pyscx

    a_csr = _open_csr_only(tmp_path / "csr_only.scx", small_adata)
    with pytest.raises(RuntimeError, match="CSC"):
        pyscx.accel.rank_genes_groups(a_csr, "group", prefer_format="csc")


# ---------------------------------------------------------------------------
# pdex_ref: CSC dispatch matches CSR on the same fixture.
# ---------------------------------------------------------------------------


def test_pdex_ref_csc_matches_csr(small_adata, tmp_path):
    pytest.importorskip("polars")  # pdex_ref() returns a polars DataFrame
    import pyscx

    a_csr = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)

    # Pin CSR explicitly — the sidecar file would otherwise auto-route to CSC.
    df_csr = pyscx.accel.pdex_ref(
        a_csr, "group", reference="A", geometric_mean=False, gene_chunk_size=4,
        prefer_format="csr", device="cpu",
    )
    df_csc = pyscx.accel.pdex_ref(
        a_csc, "group", reference="A", geometric_mean=False,
        gene_chunk_size=4, prefer_format="csc", device="cpu",
    )
    assert a_csr.uns["scx_accel"]["pdex_ref"]["route"] == "cpu_csr"
    assert a_csc.uns["scx_accel"]["pdex_ref"]["route"] == "cpu_csc"

    # Order rows the same way before comparing values.
    df_csr = df_csr.sort(["target", "feature"])
    df_csc = df_csc.sort(["target", "feature"])

    assert df_csr["target"].to_list() == df_csc["target"].to_list()
    assert df_csr["feature"].to_list() == df_csc["feature"].to_list()
    for col in ("target_mean", "ref_mean", "log2_fold_change", "p_value", "statistic", "fdr"):
        np.testing.assert_allclose(
            df_csc[col].to_numpy(), df_csr[col].to_numpy(),
            atol=1e-9, rtol=1e-6, err_msg=col,
        )


def test_pdex_ref_csc_raises_on_csr_only(small_adata, tmp_path):
    import pyscx

    a_csr = _open_csr_only(tmp_path / "csr_only.scx", small_adata)
    with pytest.raises(RuntimeError, match="CSC"):
        pyscx.accel.pdex_ref(a_csr, "group", reference="A", prefer_format="csc")


def test_pdex_ref_invalid_prefer_format(small_adata, tmp_path):
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    # "auto" is now a valid value (Phase-2 §5.2 default); only genuinely unknown
    # strings raise.
    with pytest.raises(ValueError, match="prefer_format"):
        pyscx.accel.pdex_ref(a_csc, "group", reference="A", prefer_format="bogus")


def test_pdex_ref_auto_routes_csc_when_sidecar_present(small_adata, tmp_path):
    """`prefer_format="auto"` (the default) on a CPU file *with* a CSC sidecar
    takes the CSC-direct route and matches the explicit CSC result."""
    pytest.importorskip("polars")
    import pyscx

    a_auto = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)

    df_auto = pyscx.accel.pdex_ref(
        a_auto, "group", reference="A", geometric_mean=False, gene_chunk_size=4,
        device="cpu",  # prefer_format defaults to "auto"
    )
    df_csc = pyscx.accel.pdex_ref(
        a_csc, "group", reference="A", geometric_mean=False,
        gene_chunk_size=4, prefer_format="csc", device="cpu",
    )
    assert a_auto.uns["scx_accel"]["pdex_ref"]["route"] == "cpu_csc"
    df_auto = df_auto.sort(["target", "feature"])
    df_csc = df_csc.sort(["target", "feature"])
    for col in ("target_mean", "ref_mean", "log2_fold_change", "p_value", "statistic", "fdr"):
        np.testing.assert_allclose(
            df_auto[col].to_numpy(), df_csc[col].to_numpy(), atol=1e-9, rtol=1e-6, err_msg=col,
        )


def test_pdex_ref_auto_falls_back_to_csr_without_sidecar(small_adata, tmp_path):
    """`prefer_format="auto"` on a CSR-only file falls back to the CSR streamer
    (no error) and records `cpu_csr` + `csc_available=False`."""
    pytest.importorskip("polars")
    import pyscx

    a_auto = _open_csr_only(tmp_path / "csr_only.scx", small_adata)
    a_csr = _open_csr_only(tmp_path / "csr_only.scx", small_adata)

    df_auto = pyscx.accel.pdex_ref(
        a_auto, "group", reference="A", geometric_mean=False, gene_chunk_size=4, device="cpu",
    )
    df_csr = pyscx.accel.pdex_ref(
        a_csr, "group", reference="A", geometric_mean=False,
        gene_chunk_size=4, prefer_format="csr", device="cpu",
    )
    info = a_auto.uns["scx_accel"]["pdex_ref"]
    assert info["route"] == "cpu_csr"
    assert info["csc_available"] is False
    df_auto = df_auto.sort(["target", "feature"])
    df_csr = df_csr.sort(["target", "feature"])
    for col in ("target_mean", "ref_mean", "log2_fold_change", "p_value", "statistic", "fdr"):
        np.testing.assert_allclose(
            df_auto[col].to_numpy(), df_csr[col].to_numpy(), atol=1e-9, rtol=1e-6, err_msg=col,
        )


def test_rank_genes_groups_auto_routes_csc_when_sidecar_present(small_adata, tmp_path):
    """Default `prefer_format="auto"` routes CSC-direct on a sidecar file and
    matches the explicit CSC gene ordering."""
    import pyscx

    a_auto = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)

    pyscx.accel.rank_genes_groups(a_auto, "group", device="cpu")  # auto default
    pyscx.accel.rank_genes_groups(a_csc, "group", prefer_format="csc", device="cpu")

    info = a_auto.uns["scx_accel"]["rank_genes_groups"]
    assert info["route"] == "cpu_csc"
    assert info["csc_available"] is True
    auto_res = a_auto.uns["rank_genes_groups"]
    csc_res = a_csc.uns["rank_genes_groups"]
    assert (
        np.asarray(auto_res["names"]).tolist()[0]
        == np.asarray(csc_res["names"]).tolist()[0]
    )
    # Not just ordering — scores must match too (ordering can agree while scores diverge).
    np.testing.assert_allclose(
        np.asarray(auto_res["scores"]).tolist()[0],
        np.asarray(csc_res["scores"]).tolist()[0],
        atol=1e-9, rtol=1e-6,
    )


def test_rank_genes_groups_auto_gene_subset_backed_does_not_error(small_adata, tmp_path):
    """Regression: a *projected* (gene-subset) backed dataset that still carries a
    CSC sidecar must NOT route to CSC-direct under the `auto` default — the CSC
    kernel's full-axis `n_vars` guard would raise, whereas the CSR streamer reads
    the projected columns fine. `auto` must fall back to CSR (route cpu_csr) and
    match an explicit-CSR run (review: the flip must not regress a working call)."""
    import pyscx

    a_auto = _open_with_csc(tmp_path / "sub.scx", small_adata)
    a_csr = _open_with_csc(tmp_path / "sub.scx", small_adata)
    # Project to a gene subset (keeps the sidecar on the backed dataset).
    sub_auto = a_auto[:, :8]
    sub_csr = a_csr[:, :8]

    pyscx.accel.rank_genes_groups(sub_auto, "group", device="cpu")  # auto default
    pyscx.accel.rank_genes_groups(sub_csr, "group", prefer_format="csr", device="cpu")

    assert sub_auto.uns["scx_accel"]["rank_genes_groups"]["route"] == "cpu_csr"
    assert (
        np.asarray(sub_auto.uns["rank_genes_groups"]["names"]).tolist()[0]
        == np.asarray(sub_csr.uns["rank_genes_groups"]["names"]).tolist()[0]
    )


def test_pdex_ref_csc_with_device_auto_falls_back_to_cpu(small_adata, tmp_path):
    """`device="auto"` + `prefer_format="csc"` must not error on GPU hosts;
    CSC has no GPU kernel in v1 so we silently fall back to CPU."""
    pytest.importorskip("polars")
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    # Default device is "auto"; on a GPU host this used to error.
    df = pyscx.accel.pdex_ref(
        a_csc, "group", reference="A", geometric_mean=False,
        gene_chunk_size=4, prefer_format="csc",
    )
    assert df.height > 0


def test_pdex_ref_csc_with_explicit_gpu_raises(small_adata, tmp_path):
    """Explicit `device="gpu"` + `prefer_format="csc"` is still rejected —
    only the auto path falls back. Skipped on builds without the gpu
    feature (where `device="gpu"` errors earlier in resolve_device)."""
    import pyscx

    if not pyscx.accel.gpu_available():
        pytest.skip("gpu feature not available")
    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    with pytest.raises(RuntimeError, match="csc"):
        pyscx.accel.pdex_ref(
            a_csc, "group", reference="A",
            prefer_format="csc", device="gpu",
        )


def test_rank_genes_groups_csc_with_device_auto_falls_back_to_cpu(small_adata, tmp_path):
    """Same auto-fallback contract as pdex_ref."""
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    # device defaults to "auto"; CSC must just work.
    pyscx.accel.rank_genes_groups(a_csc, "group", prefer_format="csc")
    assert "rank_genes_groups" in a_csc.uns


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
# calculate_qc_metrics: prefer_format="csc" matches CSR.
# ---------------------------------------------------------------------------


def test_qc_metrics_csc_matches_csr(small_adata, tmp_path):
    import pyscx

    a_csr = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)

    pyscx.accel.calculate_qc_metrics(a_csr)
    pyscx.accel.calculate_qc_metrics(a_csc, prefer_format="csc")

    np.testing.assert_allclose(
        np.asarray(a_csc.var["total_counts"]),
        np.asarray(a_csr.var["total_counts"]),
        atol=1e-9,
    )
    np.testing.assert_array_equal(
        np.asarray(a_csc.var["n_cells_by_counts"]),
        np.asarray(a_csr.var["n_cells_by_counts"]),
    )


def test_qc_metrics_csc_invalid_prefer_format(small_adata, tmp_path):
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    with pytest.raises(ValueError, match="prefer_format"):
        pyscx.accel.calculate_qc_metrics(a_csc, prefer_format="auto")


def test_qc_metrics_csc_raises_on_csr_only(small_adata, tmp_path):
    import pyscx

    a_csr = _open_csr_only(tmp_path / "csr_only.scx", small_adata)
    with pytest.raises(RuntimeError, match="CSC"):
        pyscx.accel.calculate_qc_metrics(a_csr, prefer_format="csc")


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


# ---------------------------------------------------------------------------
# Materialized-vs-backed CSC sidecar hint.
# `Experiment.to_anndata()` (non-backed) stamps a uns hint when the source
# file carries a CSC sidecar, so GPU DE can warn that it fell back to gpu_csr_v3
# because the sidecar was dropped at materialization. The warning itself needs a
# GPU; here we assert the deterministic CPU-side stamping contract.
# ---------------------------------------------------------------------------

_F10_HINT = "scx_source_has_csc_sidecar"


def test_materialized_csc_file_stamps_sidecar_hint(small_adata, tmp_path):
    """Non-backed to_anndata() on a CSC-equipped file stamps the F10 hint."""
    import pyscx

    pyscx.from_anndata(small_adata, str(tmp_path / "csc.scx"), csc="always")
    exp = pyscx.open(str(tmp_path / "csc.scx"))
    assert exp.has_csc
    mat = exp.to_anndata()  # materialized → in-memory CSR, sidecar dropped
    assert mat.uns.get(_F10_HINT) is True


def test_backed_csc_file_does_not_stamp_hint(small_adata, tmp_path):
    """Backed AnnData carries the real sidecar (gpu_csc_v3), so no hint/warning."""
    import pyscx

    pyscx.from_anndata(small_adata, str(tmp_path / "csc.scx"), csc="always")
    bk = pyscx.open(str(tmp_path / "csc.scx")).to_anndata(backed=True)
    assert _F10_HINT not in bk.uns


def test_sidecar_less_file_does_not_stamp_hint(small_adata, tmp_path):
    """A file without a CSC sidecar legitimately uses CSR-direct — never warn."""
    import pyscx

    pyscx.from_anndata(small_adata, str(tmp_path / "nocsc.scx"), csc="off")
    exp = pyscx.open(str(tmp_path / "nocsc.scx"))
    assert not exp.has_csc
    assert _F10_HINT not in exp.to_anndata().uns
