"""Convert-time grouping (`pyscx.from_h5ad(group_by=...)`).

A grouped layout written directly during h5ad→SCX conversion must match the
two-pass baseline (`from_h5ad` then `pyscx.sort(group_by=...)`): same grouped
reads, same reference isolation, same labels — in a single write.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def screen_h5ad(tmp_path):
    """A tiny perturbation-screen h5ad on disk: scattered `target_gene` grouping
    with `nt` as the reference label."""
    import anndata

    rng = np.random.default_rng(7)
    genes = ["nt", "MYC", "nt", "TP53", "MYC", "GATA1", "nt", "MYC", "GATA1", "GATA1"]
    n_obs, n_vars = len(genes), 6
    dense = rng.integers(0, 50, size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random((n_obs, n_vars)) > 0.5] = 0
    obs = pd.DataFrame(
        {"target_gene": pd.Categorical(genes)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    path = str(tmp_path / "screen.h5ad")
    anndata.AnnData(X=sp.csr_matrix(dense), obs=obs, var=var).write_h5ad(path)
    return path, genes


def _baseline_grouped(h5ad, tmp_scx, out_scx, reference, shard_size):
    """Two-pass baseline: plain convert, then native grouped sort."""
    import pyscx

    pyscx.from_h5ad(h5ad, tmp_scx, shard_size=shard_size)
    pyscx.sort(
        tmp_scx,
        out_scx,
        by=[],
        group_by="target_gene",
        reference=reference,
        shard_size=shard_size,
    )


def test_convert_group_by_matches_two_pass(screen_h5ad, tmp_path):
    import pyscx

    h5ad, genes = screen_h5ad
    direct = str(tmp_path / "direct.scx")
    tmp = str(tmp_path / "plain.scx")
    baseline = str(tmp_path / "baseline.scx")
    shard_size = 4

    # One-pass grouped convert (Phase 7.4) vs two-pass convert-then-sort.
    pyscx.from_h5ad(
        h5ad, direct, group_by="target_gene", reference=["nt"], shard_size=shard_size
    )
    _baseline_grouped(h5ad, tmp, baseline, ["nt"], shard_size)

    d = pyscx.open(direct)
    b = pyscx.open(baseline)

    assert set(d.group_labels()) == set(b.group_labels())
    assert d.n_obs == b.n_obs == len(genes)

    # Each non-reference group reads back identically.
    for label in ["MYC", "TP53", "GATA1"]:
        dm = d.read_group(label)
        bm = b.read_group(label)
        assert dm.n_obs == bm.n_obs > 0
        np.testing.assert_array_equal(dm.X.toarray(), bm.X.toarray())
        assert list(dm.obs["target_gene"]) == [label] * dm.n_obs

    # Reference (`nt`) is isolated and matches.
    dref = d.read_reference()
    bref = b.read_reference()
    assert dref.n_obs == bref.n_obs == genes.count("nt")
    assert all(dref.obs["target_gene"] == "nt")
    np.testing.assert_array_equal(dref.X.toarray(), bref.X.toarray())


def test_convert_group_by_reference_column_spec(screen_h5ad, tmp_path):
    """`reference={'column': name}` selects a boolean obs column."""
    import anndata
    import pyscx

    h5ad, genes = screen_h5ad
    # Re-write the fixture with an explicit boolean `is_control` column.
    adata = anndata.read_h5ad(h5ad)
    adata.obs["is_control"] = (adata.obs["target_gene"] == "nt").to_numpy()
    h5ad2 = str(tmp_path / "screen_ctrl.h5ad")
    adata.write_h5ad(h5ad2)

    out = str(tmp_path / "grouped_col.scx")
    pyscx.from_h5ad(
        h5ad2,
        out,
        group_by="target_gene",
        reference={"column": "is_control"},
        shard_size=4,
    )
    g = pyscx.open(out)
    ref = g.read_reference()
    assert ref.n_obs == genes.count("nt")
    assert all(ref.obs["target_gene"] == "nt")


def test_convert_reference_requires_group_by(screen_h5ad, tmp_path):
    import pyscx

    h5ad, _ = screen_h5ad
    out = str(tmp_path / "bad.scx")
    with pytest.raises((ValueError, RuntimeError)):
        pyscx.from_h5ad(h5ad, out, reference=["nt"])


def test_convert_group_pass_two_matches_default(screen_h5ad, tmp_path):
    """`group_pass='two'` (force plain convert + sort) yields the same grouped
    reads as the default auto/one-pass route on a CSR source."""
    import pyscx

    h5ad, _ = screen_h5ad
    one = str(tmp_path / "one.scx")
    two = str(tmp_path / "two.scx")
    pyscx.from_h5ad(h5ad, one, group_by="target_gene", reference=["nt"], shard_size=4)
    pyscx.from_h5ad(
        h5ad, two, group_by="target_gene", reference=["nt"], shard_size=4, group_pass="two"
    )

    a, b = pyscx.open(one), pyscx.open(two)
    assert set(a.group_labels()) == set(b.group_labels())
    assert a.read_reference().n_obs == b.read_reference().n_obs
    for label in ["MYC", "GATA1"]:
        np.testing.assert_array_equal(
            a.read_group(label).X.toarray(), b.read_group(label).X.toarray()
        )
