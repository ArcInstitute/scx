"""Streaming accelerators must run on the backed handle's *view*.

Companion to `test_pca_axis_view.py`, for the ops that *consume* an obs- or
var-aligned array rather than producing one: pseudobulk aggregation,
`rank_genes_groups`, `pdex_ref`, and `pseudobulk_means` (the perturbation
evaluation path).

These took the same wrong turn as `pca` — `&*backed.backed`, the file rather
than the window. Each is handed `groups` / `obs_groups` (one entry per
**visible** cell) and `gene_names` (one per **visible** gene). Measured against
`b4616630`, that produced two different failures:

* **Loud** wherever a length is checked against the source — a row subset gave
  `InvalidInput: groups length 59 != n_obs 120`, and a var subset gave
  `gene_names has 15 entries but n_vars = 30` from pseudobulk's
  `validate_inputs`. Still broken (`filter_cells` then `pseudobulk_means` is an
  ordinary thing to want), just not silent.
* **Silent** for `rank_genes_groups` / `pdex_ref` after a *var* subset, which
  is why `test_rank_genes_groups_after_var_subset` matters. Those kernels take
  `n_vars` from `gene_names.len()`, not from the source, so the guard passes —
  and then they chunk columns `0..15` of a 30-gene reader. The result came back
  fully formed, describing on-disk genes 0–14 under the names of visible genes
  {0, 2, 4, …}. Same index-space confusion as `pca`'s `mask_var` case.

The kernels are now generic over `ShardSource` and the dispatch hands them the
handle's view, so these work and agree with the materialized answer.

Multi-shard fixtures throughout — a single-shard file cannot exhibit a
row-window bug.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import pyscx

N_OBS = 120
N_VARS = 30
SHARD_SIZE = 25  # → 5 shards

# Not a prefix, and spans every shard, so a raw-reader read is distinguishable.
ROW_KEEP = np.zeros(N_OBS, dtype=bool)
ROW_KEEP[np.arange(2, N_OBS, 2)] = True  # 59 cells

COL_KEEP = np.zeros(N_VARS, dtype=bool)
COL_KEEP[np.arange(0, N_VARS, 2)] = True  # 15 genes, ascending


@pytest.fixture(scope="module")
def counts():
    rng = np.random.default_rng(7)
    x = sp.random(
        N_OBS, N_VARS, density=0.5, format="csr", dtype=np.float32, random_state=rng
    )
    x.data = (x.data * 30).astype(np.float32).round() + 1.0
    x.eliminate_zeros()
    return x


@pytest.fixture(scope="module")
def obs_cols():
    """`pert` (2 levels) for DE, plus `donor` so pseudobulk_dex has replicates.

    `pert` alternates on *pairs* rather than on parity so `ROW_KEEP` (every
    other cell) still spans both levels — with plain parity the subset is all
    `ctrl` and DE has nothing to contrast.
    """
    pert = np.array(
        ["ctrl" if (i // 2) % 2 == 0 else "drug" for i in range(N_OBS)], dtype=object
    )
    donor = np.array([f"d{i % 3}" for i in range(N_OBS)], dtype=object)
    return pert, donor


def _annotate(adata, pert, donor, var_names):
    import pandas as pd

    adata.obs["pert"] = pd.Categorical(list(pert))
    adata.obs["donor"] = pd.Categorical(list(donor))
    adata.var_names = list(var_names)
    return adata


@pytest.fixture(scope="module")
def var_names():
    return [f"g{i}" for i in range(N_VARS)]


@pytest.fixture(scope="module")
def scx_path(tmp_path_factory, counts, obs_cols, var_names):
    import anndata as ad

    pert, donor = obs_cols
    adata = ad.AnnData(X=counts.copy())
    adata.obs_names = [f"c{i}" for i in range(N_OBS)]
    _annotate(adata, pert, donor, var_names)
    path = tmp_path_factory.mktemp("accel_axis_view") / "counts.scx"
    pyscx.from_anndata(adata, str(path), shard_size=SHARD_SIZE)
    return path


@pytest.fixture(scope="module")
def csc_path(tmp_path_factory, counts, obs_cols, var_names):
    import anndata as ad

    pert, donor = obs_cols
    adata = ad.AnnData(X=counts.copy())
    adata.obs_names = [f"c{i}" for i in range(N_OBS)]
    _annotate(adata, pert, donor, var_names)
    path = tmp_path_factory.mktemp("accel_axis_view_csc") / "csc.scx"
    pyscx.from_anndata(
        adata, str(path), shard_size=SHARD_SIZE, csc="always", csc_cols_per_shard=8
    )
    return path


def _backed(path):
    return pyscx.open(str(path)).to_anndata(backed=True)


def _reference(counts, obs_cols, var_names, rows=None, cols=None):
    """The materialized equivalent of the same subset — the ground truth."""
    import anndata as ad

    pert, donor = obs_cols
    x = counts
    names = list(var_names)
    if rows is not None:
        x, pert, donor = x[rows, :], pert[rows], donor[rows]
    if cols is not None:
        x = x[:, cols]
        names = [n for n, k in zip(names, cols) if k]
    a = ad.AnnData(X=x.copy())
    return _annotate(a, pert, donor, names)


# ---------------------------------------------------------------------------
# pseudobulk_means — the eval-metrics aggregation path
# ---------------------------------------------------------------------------


def test_pseudobulk_means_after_row_subset(scx_path, counts, obs_cols, var_names):
    """Was `InvalidInput: obs_groups[0] length 59 != n_obs 120`."""
    adata = _backed(scx_path)
    pyscx.accel.subset_obs(adata, ROW_KEEP)
    assert adata.n_obs == int(ROW_KEEP.sum())
    assert type(adata.X).__name__ == "ScxBackedSparseDataset"

    means, names = pyscx.accel.pseudobulk_means(adata, "pert")
    ref_means, ref_names = pyscx.accel.pseudobulk_means(
        _reference(counts, obs_cols, var_names, rows=ROW_KEEP), "pert"
    )
    assert names == ref_names
    np.testing.assert_allclose(means, ref_means, rtol=1e-6)


def test_pseudobulk_means_after_var_subset(scx_path, counts, obs_cols, var_names):
    adata = _backed(scx_path)
    pyscx.accel.subset_var(adata, COL_KEEP)

    means, names = pyscx.accel.pseudobulk_means(adata, "pert")
    ref_means, ref_names = pyscx.accel.pseudobulk_means(
        _reference(counts, obs_cols, var_names, cols=COL_KEEP), "pert"
    )
    assert means.shape == ref_means.shape == (2, int(COL_KEEP.sum()))
    assert names == ref_names
    np.testing.assert_allclose(means, ref_means, rtol=1e-6)


def test_pseudobulk_means_composed_subset(scx_path, counts, obs_cols, var_names):
    adata = _backed(scx_path)
    pyscx.accel.subset_obs(adata, ROW_KEEP)
    pyscx.accel.subset_var(adata, COL_KEEP)

    means, _ = pyscx.accel.pseudobulk_means(adata, "pert")
    ref_means, _ = pyscx.accel.pseudobulk_means(
        _reference(counts, obs_cols, var_names, rows=ROW_KEEP, cols=COL_KEEP), "pert"
    )
    np.testing.assert_allclose(means, ref_means, rtol=1e-6)


def test_pseudobulk_means_unsubset_is_unchanged(scx_path, counts, obs_cols, var_names):
    """The common path must be untouched by the source swap."""
    means, _ = pyscx.accel.pseudobulk_means(_backed(scx_path), "pert")
    ref_means, _ = pyscx.accel.pseudobulk_means(
        _reference(counts, obs_cols, var_names), "pert"
    )
    np.testing.assert_allclose(means, ref_means, rtol=1e-6)


# ---------------------------------------------------------------------------
# pseudobulk_dex — the `scx_accel::pseudobulk_aggregate` path
# ---------------------------------------------------------------------------


def _dex(adata):
    return pyscx.accel.pseudobulk_dex(
        adata,
        groupby=["pert", "donor"],
        test_col="pert",
        reference="ctrl",
        min_cells_per_group=1,
        backend="nb_glm",
    )


def test_pseudobulk_dex_after_composed_subset(scx_path, counts, obs_cols, var_names):
    adata = _backed(scx_path)
    pyscx.accel.subset_obs(adata, ROW_KEEP)
    pyscx.accel.subset_var(adata, COL_KEEP)

    got = _dex(adata)
    ref = _dex(_reference(counts, obs_cols, var_names, rows=ROW_KEEP, cols=COL_KEEP))

    assert len(got) == len(ref) == int(COL_KEEP.sum())
    gene_col = "gene" if "gene" in got.columns else got.columns[0]
    assert list(got[gene_col]) == list(ref[gene_col])


def test_pseudobulk_dex_unsubset_is_unchanged(scx_path, counts, obs_cols, var_names):
    got = _dex(_backed(scx_path))
    ref = _dex(_reference(counts, obs_cols, var_names))
    assert len(got) == len(ref) == N_VARS


# ---------------------------------------------------------------------------
# rank_genes_groups (Wilcoxon) and pdex_ref
# ---------------------------------------------------------------------------


def _rgg(adata, **kw):
    pyscx.accel.rank_genes_groups(
        adata, groupby="pert", method="wilcoxon", device="cpu", **kw
    )
    return pyscx.accel.rank_genes_groups_df(adata, group="drug")


def _assert_de_equal(got, ref):
    assert list(got["names"]) == list(ref["names"])
    np.testing.assert_allclose(
        got["scores"].to_numpy().astype(np.float64),
        ref["scores"].to_numpy().astype(np.float64),
        atol=1e-5,
    )


def test_rank_genes_groups_after_row_subset(scx_path, counts, obs_cols, var_names):
    """Was `InvalidInput: groups length 59 != n_obs 120`."""
    adata = _backed(scx_path)
    pyscx.accel.subset_obs(adata, ROW_KEEP)
    _assert_de_equal(
        _rgg(adata), _rgg(_reference(counts, obs_cols, var_names, rows=ROW_KEEP))
    )


def test_rank_genes_groups_after_var_subset(scx_path, counts, obs_cols, var_names):
    adata = _backed(scx_path)
    pyscx.accel.subset_var(adata, COL_KEEP)
    got = _rgg(adata)
    assert len(got) == int(COL_KEEP.sum())
    _assert_de_equal(got, _rgg(_reference(counts, obs_cols, var_names, cols=COL_KEEP)))


def test_rank_genes_groups_composed_subset(scx_path, counts, obs_cols, var_names):
    adata = _backed(scx_path)
    pyscx.accel.subset_obs(adata, ROW_KEEP)
    pyscx.accel.subset_var(adata, COL_KEEP)
    _assert_de_equal(
        _rgg(adata),
        _rgg(_reference(counts, obs_cols, var_names, rows=ROW_KEEP, cols=COL_KEEP)),
    )


def test_rank_genes_groups_unsubset_is_unchanged(scx_path, counts, obs_cols, var_names):
    _assert_de_equal(_rgg(_backed(scx_path)), _rgg(_reference(counts, obs_cols, var_names)))


def _pdex(adata):
    return pyscx.accel.pdex_ref(adata, "pert", reference="ctrl", device="cpu")


def test_pdex_ref_after_composed_subset(scx_path, counts, obs_cols, var_names):
    adata = _backed(scx_path)
    pyscx.accel.subset_obs(adata, ROW_KEEP)
    pyscx.accel.subset_var(adata, COL_KEEP)

    got = _pdex(adata)
    ref = _pdex(_reference(counts, obs_cols, var_names, rows=ROW_KEEP, cols=COL_KEEP))

    assert list(got["feature"]) == list(ref["feature"])
    for col in ("target_mean", "ref_mean", "log2_fold_change", "p_value"):
        np.testing.assert_allclose(
            got[col].to_numpy().astype(np.float64),
            ref[col].to_numpy().astype(np.float64),
            atol=1e-6,
        )


def test_pdex_ref_unsubset_is_unchanged(scx_path, counts, obs_cols, var_names):
    got = _pdex(_backed(scx_path))
    ref = _pdex(_reference(counts, obs_cols, var_names))
    assert list(got["feature"]) == list(ref["feature"])
    np.testing.assert_allclose(
        got["log2_fold_change"].to_numpy().astype(np.float64),
        ref["log2_fold_change"].to_numpy().astype(np.float64),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# prefer_format="csc" — still refused on a subset, but now says why
# ---------------------------------------------------------------------------


def test_csc_on_row_subset_names_the_cause(csc_path):
    adata = _backed(csc_path)
    pyscx.accel.subset_obs(adata, ROW_KEEP)
    with pytest.raises(RuntimeError, match="row deletion vector"):
        pyscx.accel.rank_genes_groups(
            adata, groupby="pert", method="wilcoxon", prefer_format="csc", device="cpu"
        )


def test_csc_on_var_subset_names_the_cause(csc_path):
    """Used to surface as a bare `gene_names length 15 != source.n_vars() 30`."""
    adata = _backed(csc_path)
    pyscx.accel.subset_var(adata, COL_KEEP)
    with pytest.raises(RuntimeError, match="column projection is active"):
        pyscx.accel.rank_genes_groups(
            adata, groupby="pert", method="wilcoxon", prefer_format="csc", device="cpu"
        )


def test_csc_unsubset_still_works(csc_path, counts, obs_cols, var_names):
    """The new guard must not catch an ordinary CSC-direct run."""
    adata = _backed(csc_path)
    pyscx.accel.rank_genes_groups(
        adata, groupby="pert", method="wilcoxon", prefer_format="csc", device="cpu"
    )
    _assert_de_equal(
        pyscx.accel.rank_genes_groups_df(adata, group="drug"),
        _rgg(_reference(counts, obs_cols, var_names)),
    )
