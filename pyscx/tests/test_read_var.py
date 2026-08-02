"""`Experiment.read_var()` — the var-axis mirror of `read_obs()`.

`read_obs` has existed for a long time; `read_var` did not. A user who wanted
just the gene table — to map Ensembl ids to `feature_name` when labelling
marker genes, say — had to call `to_anndata()` and materialize the whole
matrix to get at it. On a 500k-cell Census file that is a 1.4 s / 9.9 GB
detour for a 5 MB table (user-report F2).

These tests pin the symmetry with `read_obs`: same index, same projection
behaviour, X never touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import pyscx

N_OBS, N_VARS = 40, 12


@pytest.fixture(scope="module")
def scx_path(tmp_path_factory):
    import anndata as ad

    rng = np.random.default_rng(5)
    adata = ad.AnnData(X=sp.csr_matrix(rng.poisson(2.0, (N_OBS, N_VARS)).astype("float32")))
    adata.obs_names = [f"c{i}" for i in range(N_OBS)]
    adata.var_names = [f"ENSG{i:05d}" for i in range(N_VARS)]
    adata.var["feature_name"] = [f"GENE{i}" for i in range(N_VARS)]
    adata.var["feature_type"] = pd.Categorical(["protein_coding"] * N_VARS)
    adata.var["length"] = np.arange(N_VARS, dtype=np.int64)
    path = tmp_path_factory.mktemp("read_var") / "v.scx"
    pyscx.from_anndata(adata, str(path))
    return path


def test_read_var_matches_to_anndata_var(scx_path):
    """The whole point: same frame, without materializing X."""
    exp = pyscx.open(str(scx_path))
    var = exp.read_var()
    reference = exp.to_anndata().var

    assert list(var.index) == list(reference.index)
    assert set(var.columns) == set(reference.columns)
    for col in reference.columns:
        pd.testing.assert_series_equal(
            var[col], reference[col], check_dtype=False, check_categorical=False
        )


def test_read_var_index_is_the_gene_names(scx_path):
    var = pyscx.open(str(scx_path)).read_var()
    assert list(var.index) == [f"ENSG{i:05d}" for i in range(N_VARS)]
    assert len(var) == N_VARS


def test_read_var_columns_projects_and_keeps_the_index(scx_path):
    """Same contract as `read_obs(columns=...)`: the index survives."""
    var = pyscx.open(str(scx_path)).read_var(columns=["feature_name"])
    assert list(var.columns) == ["feature_name"]
    assert list(var.index) == [f"ENSG{i:05d}" for i in range(N_VARS)]
    assert list(var["feature_name"]) == [f"GENE{i}" for i in range(N_VARS)]


def test_read_var_columns_preserves_requested_order(scx_path):
    var = pyscx.open(str(scx_path)).read_var(columns=["length", "feature_name"])
    assert list(var.columns) == ["length", "feature_name"]


def test_read_var_unknown_column_names_what_is_available(scx_path):
    """A bare pyarrow KeyError would not say what the user could have asked for."""
    with pytest.raises(KeyError, match="feature_name"):
        pyscx.open(str(scx_path)).read_var(columns=["not_a_column"])


def test_read_var_agrees_with_var_keys(scx_path):
    """`columns` takes physical names — the ones `var_keys()` reports."""
    exp = pyscx.open(str(scx_path))
    keys = exp.var_keys()
    var = exp.read_var(columns=keys)
    assert set(var.columns) == set(keys)


def test_read_var_works_when_x_is_unreadable(tmp_path, scx_path):
    """`read_var` must not touch X — proven by breaking X.

    An earlier version of this test opened a backed AnnData, called
    `read_var()`, and asserted `adata.X` was still a lazy handle. That could
    not fail: `read_var` returns a fresh DataFrame and has no path to that
    object, so the assertion held no matter what `read_var` did internally.

    Corrupting the CSR payload is a real discriminator: reading X now raises,
    so a `read_var` that decoded X (or routed through `to_anndata`) fails
    here, while one that reads only the var section passes.
    """
    import shutil

    broken = tmp_path / "broken_x.scx"
    shutil.copy(scx_path, broken)

    # Sections start at 4352 (docs/format.md) and the CSR shards come first,
    # so this lands in X's payload. If the layout ever changes so that it does
    # not, `read_var()` below raises and the test fails loudly rather than
    # silently going vacuous again.
    with open(broken, "r+b") as fh:
        fh.seek(4352)
        fh.write(b"\xde\xad\xbe\xef" * 256)

    exp = pyscx.open(str(broken))
    var = exp.read_var()  # must not raise
    assert list(var.index) == [f"ENSG{i:05d}" for i in range(N_VARS)]
    assert list(var["feature_name"]) == [f"GENE{i}" for i in range(N_VARS)]

    # The discriminator: X really is unreadable now, so a `read_var` that
    # decoded X could not have passed above.
    with pytest.raises(Exception):
        exp.to_anndata()


def test_read_var_unknown_modality_raises(scx_path):
    with pytest.raises(KeyError, match="unknown modality"):
        pyscx.open(str(scx_path)).read_var(modality="nope")


@pytest.fixture
def citeseq_scx(tmp_path):
    """Tiny 2-modality file — rna (8 genes) + adt (4 proteins)."""
    mudata = pytest.importorskip("mudata")
    anndata = pytest.importorskip("anndata")

    rng = np.random.default_rng(0)
    n_obs = 20
    rna_ad = anndata.AnnData(
        X=sp.csr_matrix(rng.poisson(0.6, (n_obs, 8)).astype("float32"))
    )
    rna_ad.var_names = [f"g{i}" for i in range(8)]
    rna_ad.var["feature_name"] = [f"GENE{i}" for i in range(8)]
    adt_ad = anndata.AnnData(
        X=sp.csr_matrix(rng.poisson(0.6, (n_obs, 4)).astype("float32"))
    )
    adt_ad.var_names = [f"a{i}" for i in range(4)]
    adt_ad.var["feature_name"] = [f"PROT{i}" for i in range(4)]

    mu = mudata.MuData({"rna": rna_ad, "adt": adt_ad})
    mu.obs_names = [f"cell_{i}" for i in range(n_obs)]
    path = str(tmp_path / "cite.scx")
    pyscx.from_mudata(mu, path)
    return path


def test_read_var_scopes_to_the_requested_modality(citeseq_scx):
    """Each modality has its own gene axis — `modality=` must select it.

    This is the one semantic difference from `read_obs` (obs is shared), so
    it is the part most worth pinning.
    """
    exp = pyscx.open(citeseq_scx)

    rna = exp.read_var(modality="rna")
    assert len(rna) == 8
    assert list(rna.index) == [f"g{i}" for i in range(8)]
    assert list(rna["feature_name"]) == [f"GENE{i}" for i in range(8)]

    adt = exp.read_var(modality="adt")
    assert len(adt) == 4
    assert list(adt.index) == [f"a{i}" for i in range(4)]
    assert list(adt["feature_name"]) == [f"PROT{i}" for i in range(4)]


def test_read_var_modality_projection(citeseq_scx):
    adt = pyscx.open(citeseq_scx).read_var(columns=["feature_name"], modality="adt")
    assert list(adt.columns) == ["feature_name"]
    assert list(adt["feature_name"]) == [f"PROT{i}" for i in range(4)]


def test_read_var_unknown_modality_raises_on_a_multimodal_file(citeseq_scx):
    """The unimodal case raises too, but this is the path users hit."""
    with pytest.raises(KeyError, match="unknown modality"):
        pyscx.open(citeseq_scx).read_var(modality="atac")
