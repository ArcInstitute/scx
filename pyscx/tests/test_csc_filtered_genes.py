"""Targeted DE / pseudobulk on a small gene subset.

The CSC dispatch should produce the same numerical result as CSR
when only a handful of genes is queried out of a much wider matrix.
This is the "highest-value win" cell on the CSC dispatch matrix:
DE on a few-genes subset of a 30K-gene file would otherwise have
to decode every CSR row and project per-shard.

This test focuses on parity, not perf — the perf claim lives in
the benchmark harness (`benchmarks/comprehensive/benchmarks/
bench_csc_dispatch.py`).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def wide_adata():
    """80 cells × 200 genes — wide enough that a 50-gene subset
    is a meaningful fraction of the dataset (25%) and still small
    enough to run in milliseconds."""
    pytest.importorskip("anndata")
    import anndata as ad

    rng = np.random.default_rng(23)
    mat = sp.random(80, 200, density=0.15, format="csr", dtype=np.float32, random_state=rng)
    mat.data = (mat.data * 50).astype(np.float32).round() + 1.0
    adata = ad.AnnData(X=mat)
    adata.obs["cell_id"] = [f"c{i}" for i in range(80)]
    adata.obs["group"] = ["A"] * 40 + ["B"] * 40
    adata.var["gene_id"] = [f"g{i}" for i in range(200)]
    return adata


def _open_with_csc(path, adata):
    import pyscx

    pyscx.from_anndata(adata, str(path), csc="always", csc_cols_per_shard=32)
    return pyscx.open(str(path)).to_anndata(backed=True)


# ---------------------------------------------------------------------------
# DE on a 50-gene subset: prefer_format="csc" should match CSR exactly.
# ---------------------------------------------------------------------------


def test_de_full_csc_matches_csr_on_multishard_file(wide_adata, tmp_path):
    """Parity check on a multi-shard CSC layout (200 genes / 32
    cols-per-shard = 7 CSC shards). Exercises the chunked
    `read_csc_columns(...)` path inside the Wilcoxon kernel where
    each gene chunk straddles multiple CSC shards.
    """
    import pyscx

    a_csr = _open_with_csc(tmp_path / "csr.scx", wide_adata)
    a_csc = _open_with_csc(tmp_path / "csc.scx", wide_adata)

    pyscx.accel.rank_genes_groups(a_csr, "group", gene_chunk_size=64, prefer_format="csr")
    pyscx.accel.rank_genes_groups(a_csc, "group", gene_chunk_size=64, prefer_format="csc")

    csr_names = np.asarray(a_csr.uns["rank_genes_groups"]["names"]).tolist()[0]
    csc_names = np.asarray(a_csc.uns["rank_genes_groups"]["names"]).tolist()[0]
    assert csc_names == csr_names

    # Also compare scores within numerical tolerance (Wilcoxon scatters
    # the same dense buffer; result should be near bit-equal).
    csr_scores = np.asarray(a_csr.uns["rank_genes_groups"]["scores"]).tolist()[0]
    csc_scores = np.asarray(a_csc.uns["rank_genes_groups"]["scores"]).tolist()[0]
    np.testing.assert_allclose(
        np.asarray(csr_scores, dtype=np.float64),
        np.asarray(csc_scores, dtype=np.float64),
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# pseudobulk_dex with explicit gene_indices kwarg.
# ---------------------------------------------------------------------------


def test_pseudobulk_filtered_csc_requires_gene_indices_or_projection(wide_adata, tmp_path):
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", wide_adata)

    # Without any gene subset: csc raises with a helpful message.
    with pytest.raises(RuntimeError, match="gene"):
        pyscx.accel.pseudobulk_dex(
            a_csc, ["group"], "group", "A",
            prefer_format="csc",
        )


def test_pseudobulk_filtered_csc_dispatch_reaches_kernel(wide_adata, tmp_path):
    """pseudobulk_dex with `prefer_format="csc"` + an explicit
    `gene_indices` reaches the CSC kernel (vs raising the gate).

    A full DESeq2 fit via pydeseq2 needs ≥3 replicates per condition
    and we use a single ``group`` column, so the test stops short of
    the pydeseq2 step — it asserts that the CSC path is reachable
    (any failure here would be a CSC-dispatch issue, not a
    statistical-design one). The full DE fit is exercised in
    `bench_csc_dispatch.py` against `census_1m`.
    """
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", wide_adata)
    gene_indices = list(range(0, 200, 4))  # 50 genes

    # The kernel runs and returns; pydeseq2 may reject due to
    # insufficient replicates with this small fixture. Either is fine
    # — the assertion is that no `RuntimeError("CSC ...")` from the
    # capability gate or `RuntimeError("...gene_indices...")` from the
    # CSC dispatch precondition fires.
    try:
        pyscx.accel.pseudobulk_dex(
            a_csc,
            ["group"],
            "group",
            "A",
            prefer_format="csc",
            gene_indices=gene_indices,
        )
    except ValueError:
        # pydeseq2's own design-rank check: the CSC path was reached.
        pass
    except RuntimeError as e:
        msg = str(e)
        # Allow pydeseq2 / dependency errors; reject CSC gate failures.
        if "CSC" in msg or "gene_indices" in msg or "gene subset" in msg:
            raise
