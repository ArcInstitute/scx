"""Modality-scoped query pipeline tests.

Exercises `Experiment.query(modality=...)` on a multimodal (CITE-seq) SCX file:
per-modality X width, the global obs mask applied per modality, gene projection
in modality var space, and the fail-loud error paths.
"""

import os
import tempfile

import numpy as np
import pytest


@pytest.fixture
def citeseq_scx():
    """Write a tiny 2-modality (rna: 8 vars, adt: 4 vars) CITE-seq SCX file
    with a global `cell_type` obs column, and yield its path."""
    mudata = pytest.importorskip("mudata")
    anndata = pytest.importorskip("anndata")
    import scipy.sparse as sp

    import pyscx

    rng = np.random.default_rng(0)
    n_obs, rna_n_vars, adt_n_vars = 30, 8, 4
    rna = sp.csr_matrix(rng.poisson(0.6, size=(n_obs, rna_n_vars)).astype(np.float32))
    adt = sp.csr_matrix(rng.poisson(0.6, size=(n_obs, adt_n_vars)).astype(np.float32))

    rna_ad = anndata.AnnData(X=rna)
    rna_ad.var_names = [f"g{i}" for i in range(rna_n_vars)]
    adt_ad = anndata.AnnData(X=adt)
    adt_ad.var_names = [f"a{i}" for i in range(adt_n_vars)]

    mu = mudata.MuData({"rna": rna_ad, "adt": adt_ad})
    mu.obs_names = [f"cell_{i}" for i in range(n_obs)]
    # Global obs column, shared across modalities. T/B/NK cell, 10 each.
    mu.obs["cell_type"] = [["T cell", "B cell", "NK cell"][i % 3] for i in range(n_obs)]

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "cite.scx")
    pyscx.from_mudata(mu, path)
    return path


def test_query_per_modality_width(citeseq_scx):
    """X width comes from the modality, not the file-wide header max."""
    import pyscx

    exp = pyscx.open(citeseq_scx)
    rna = exp.query(modality="rna").collect()
    assert rna.n_obs == 30
    assert rna.n_vars == 8

    adt = exp.query(modality="adt").collect()
    assert adt.n_obs == 30
    assert adt.n_vars == 4


def test_query_global_obs_mask_applied_per_modality(citeseq_scx):
    """The obs predicate is global — the same filter yields the same obs rows
    for every modality, each with its own X width."""
    import pyscx

    exp = pyscx.open(citeseq_scx)
    rna = exp.query(modality="rna").filter_obs("cell_type == 'T cell'").collect()
    adt = exp.query(modality="adt").filter_obs("cell_type == 'T cell'").collect()

    assert rna.n_obs == 10  # 10 T cells
    assert adt.n_obs == 10
    assert rna.n_vars == 8
    assert adt.n_vars == 4

    rna_ad = rna.to_anndata()
    adt_ad = adt.to_anndata()
    # Same obs rows regardless of modality.
    assert list(rna_ad.obs_names) == list(adt_ad.obs_names)
    assert set(rna_ad.obs["cell_type"]) == {"T cell"}


def test_query_select_genes_in_modality_space(citeseq_scx):
    """select_genes indices resolve against the modality's var (adt has 4)."""
    import pyscx

    exp = pyscx.open(citeseq_scx)
    qr = exp.query(modality="adt").select_genes([1, 2]).collect()
    assert qr.n_vars == 2
    assert qr.n_obs == 30


def test_query_select_genes_by_name_in_modality_space(citeseq_scx):
    """Gene names resolve against the modality's var index."""
    import pyscx

    exp = pyscx.open(citeseq_scx)
    qr = exp.query(modality="rna").select_genes(["g0", "g5"]).collect()
    ad = qr.to_anndata()
    assert list(ad.var_names) == ["g0", "g5"]


def test_query_multimodal_without_modality_raises(citeseq_scx):
    """Omitting `modality` on a multimodal file is a fail-loud ValueError."""
    import pyscx

    exp = pyscx.open(citeseq_scx)
    with pytest.raises(ValueError, match="multimodal"):
        exp.query()


def test_query_unknown_modality_raises(citeseq_scx):
    """An unknown modality name raises KeyError listing the available names."""
    import pyscx

    exp = pyscx.open(citeseq_scx)
    with pytest.raises(KeyError):
        exp.query(modality="atac")


def test_query_count_scoped_to_modality(citeseq_scx):
    """count() honors the global obs predicate; identical across modalities."""
    import pyscx

    exp = pyscx.open(citeseq_scx)
    rna_n = exp.query(modality="rna").filter_obs("cell_type == 'T cell'").count()
    adt_n = exp.query(modality="adt").filter_obs("cell_type == 'T cell'").count()
    assert rna_n == 10
    assert adt_n == 10


def test_query_matches_to_mudata(citeseq_scx):
    """A no-filter modality query matches the modality slice from to_mudata()."""
    import pyscx

    exp = pyscx.open(citeseq_scx)
    rna_q = exp.query(modality="rna").collect().to_anndata()
    mu = pyscx.open(citeseq_scx).to_mudata()
    rna_ref = mu.mod["rna"]
    assert rna_q.shape == rna_ref.shape
    assert list(rna_q.var_names) == list(rna_ref.var_names)
    np.testing.assert_allclose(
        rna_q.X.toarray(), rna_ref.X.toarray(), rtol=0, atol=0
    )
