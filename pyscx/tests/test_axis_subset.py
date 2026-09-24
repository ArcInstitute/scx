"""Every aligned member must survive an in-place axis subset.

Phase 4.0a, root cause (a) of §9.18. anndata validates an aligned mapping
**when the attribute is read**, so the moment ``X`` / ``_var`` / ``_obs`` changes
width, ``adata.layers`` and ``adata.obsm`` cannot be read — and SCX's repair code
read exactly those properties. Consequences before this fix:

* ``filter_genes`` raised on any backed AnnData carrying a **layer**, and
  ``filter_cells`` on one carrying an **obsm** entry — *after* mutating ``_var`` /
  ``col_projection``, so the object was left half-subset;
* non-SCX members were silently skipped, leaving an in-memory layer at the old
  width to break the dataset later instead of now;
* ``varm`` / ``obsp`` / ``varp`` were never handled at all — after ``neighbors()``
  an ``obsp`` graph is ``n_obs x n_obs`` and goes stale the same way.

Every mutating op now funnels through ``axis_align::subset_{obs,var}_axis``,
which works on the **raw** stores. The deferred-decode assertions below are the
other half of the contract: keeping the members correct must not cost a decode.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

N_OBS, N_VARS = 50, 30


@pytest.fixture
def source():
    """Dense source arrays for X and all five aligned members."""
    rng = np.random.RandomState(7)
    x = (rng.random_sample((N_OBS, N_VARS)) * 1e3).astype(np.float32)
    x[rng.random_sample((N_OBS, N_VARS)) > 0.5] = 0.0
    return {
        "X": x,
        "layer": (rng.random_sample((N_OBS, N_VARS)) * 10).astype(np.float32),
        "obsm": rng.random_sample((N_OBS, 4)).astype(np.float32),
        "varm": rng.random_sample((N_VARS, 3)).astype(np.float32),
        "obsp": rng.random_sample((N_OBS, N_OBS)).astype(np.float32),
        "varp": rng.random_sample((N_VARS, N_VARS)).astype(np.float32),
    }


@pytest.fixture
def scx_path(source, tmp_dir):
    """An SCX file carrying X plus every aligned member."""
    import anndata
    import pandas as pd
    import pyscx
    import scipy.sparse as sp

    adata = anndata.AnnData(
        X=sp.csr_matrix(source["X"]),
        obs=pd.DataFrame(
            {"group": ["a" if i % 2 else "b" for i in range(N_OBS)]},
            index=[f"cell_{i}" for i in range(N_OBS)],
        ),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(N_VARS)]),
        layers={"counts": sp.csr_matrix(source["layer"])},
        obsm={"X_emb": source["obsm"]},
        varm={"loadings": source["varm"]},
        obsp={"conn": sp.csr_matrix(source["obsp"])},
        varp={"corr": sp.csr_matrix(source["varp"])},
    )
    path = str(tmp_dir / "members.scx")
    pyscx.from_anndata(adata, path)
    return path


def _as_dense(value):
    return np.asarray(value.todense() if hasattr(value, "todense") else value)


def _assert_members(adata, source, obs_keep, var_keep):
    """Read every validating property and check it against the numpy oracle.

    Reading is the point: before the fix these properties raised, so the check
    is as much "can this be read at all" as "is it right".
    """
    o, v = np.flatnonzero(obs_keep), np.flatnonzero(var_keep)

    assert adata.shape == (len(o), len(v))
    assert adata.obs.shape[0] == len(o)
    assert adata.var.shape[0] == len(v)

    assert_allclose(_as_dense(adata.X.to_memory()), source["X"][np.ix_(o, v)], rtol=1e-6)
    assert_allclose(
        _as_dense(adata.layers["counts"].to_memory()),
        source["layer"][np.ix_(o, v)],
        rtol=1e-6,
    )
    assert_allclose(_as_dense(adata.obsm["X_emb"]), source["obsm"][o], rtol=1e-6)
    assert_allclose(_as_dense(adata.varm["loadings"]), source["varm"][v], rtol=1e-6)
    assert_allclose(_as_dense(adata.obsp["conn"]), source["obsp"][np.ix_(o, o)], rtol=1e-6)
    assert_allclose(_as_dense(adata.varp["corr"]), source["varp"][np.ix_(v, v)], rtol=1e-6)


def test_filter_genes_keeps_every_member_aligned(scx_path, source):
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    detected = (source["X"] != 0).sum(axis=0)
    threshold = int(np.median(detected)) + 1
    pyscx.accel.filter_genes(adata, min_cells=threshold)

    var_keep = detected >= threshold
    assert 0 < var_keep.sum() < N_VARS, "fixture must actually drop genes"
    _assert_members(adata, source, np.ones(N_OBS, bool), var_keep)


def test_filter_cells_keeps_every_member_aligned(scx_path, source):
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    detected = (source["X"] != 0).sum(axis=1)
    threshold = int(np.median(detected)) + 1
    pyscx.accel.filter_cells(adata, min_genes=threshold)

    obs_keep = detected >= threshold
    assert 0 < obs_keep.sum() < N_OBS, "fixture must actually drop cells"
    _assert_members(adata, source, obs_keep, np.ones(N_VARS, bool))


def test_subset_obs_keeps_every_member_aligned(scx_path, source):
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    obs_keep = np.arange(N_OBS) % 3 != 0
    pyscx.accel.subset_obs(adata, obs_keep)

    _assert_members(adata, source, obs_keep, np.ones(N_VARS, bool))


def test_hvg_subset_keeps_every_member_aligned(scx_path, source):
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    pyscx.accel.highly_variable_genes(adata, n_top_genes=12, flavor="seurat_v3", subset=True)

    assert adata.n_vars == 12
    var_keep = np.isin(
        np.arange(N_VARS),
        [int(name.split("_")[1]) for name in adata.var_names],
    )
    _assert_members(adata, source, np.ones(N_OBS, bool), var_keep)


def test_both_axes_compose(scx_path, source):
    """A var subset then an obs subset, with the members read only at the end.

    Exercises the deferred path twice over: the lazy mapping entries are still
    un-fetched when both subsets land, so `PendingSubsets` has to replay them in
    order against the freshly decoded value.
    """
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    detected_genes = (source["X"] != 0).sum(axis=0)
    gene_threshold = int(np.median(detected_genes)) + 1
    pyscx.accel.filter_genes(adata, min_cells=gene_threshold)
    var_keep = detected_genes >= gene_threshold

    obs_keep = np.arange(N_OBS) % 4 != 0
    pyscx.accel.subset_obs(adata, obs_keep)

    _assert_members(adata, source, obs_keep, var_keep)


def test_lazy_members_are_not_decoded_by_a_subset(scx_path, source):
    """Keeping obsp/varp/varm aligned must not drag them off disk.

    The alternative implementation — fetch every key, slice it, put it back —
    would make `filter_cells` on a file carrying a kNN graph pay for decoding
    that graph whether or not anyone ever reads it.
    """
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    assert "0 materialized" in repr(adata._obsp), "fixture must start un-decoded"

    pyscx.accel.subset_obs(adata, np.arange(N_OBS) % 3 != 0)
    assert "0 materialized" in repr(adata._obsp), (
        "the obs subset decoded an obsp section it did not need"
    )
    assert "0 materialized" in repr(adata._varm)

    # ...and the deferred subset lands on first read.
    obs_keep = np.arange(N_OBS) % 3 != 0
    o = np.flatnonzero(obs_keep)
    assert_allclose(_as_dense(adata.obsp["conn"]), source["obsp"][np.ix_(o, o)], rtol=1e-6)


def test_non_scx_members_are_sliced_not_skipped(scx_path, source):
    """An in-memory member added by the user must be subset too.

    The pre-4.0a code cast each layer to `ScxBackedLayerDataset` and silently
    skipped anything else, so a numpy / scipy / pandas member kept its old width
    and broke the AnnData at some later, unrelated call.
    """
    import pandas as pd
    import pyscx
    import scipy.sparse as sp

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    dense_layer = (np.arange(N_OBS * N_VARS, dtype=np.float32)).reshape(N_OBS, N_VARS)
    adata.layers["inmem"] = sp.csr_matrix(dense_layer)
    frame = pd.DataFrame(
        {"score": np.arange(N_OBS, dtype=float)},
        index=adata.obs_names,
    )
    adata.obsm["frame"] = frame
    plain = np.arange(N_OBS * 2, dtype=np.float32).reshape(N_OBS, 2)
    adata.obsm["plain"] = plain

    obs_keep = np.arange(N_OBS) % 5 != 0
    pyscx.accel.subset_obs(adata, obs_keep)
    o = np.flatnonzero(obs_keep)

    assert_allclose(_as_dense(adata.layers["inmem"]), dense_layer[o], rtol=0)
    assert_array_equal(np.asarray(adata.obsm["frame"]["score"]), o.astype(float))
    assert_allclose(_as_dense(adata.obsm["plain"]), plain[o], rtol=0)


def test_backed_obsm_handle_composes_without_gathering(scx_path, source):
    """The backed obsm handle absorbs the subset instead of materializing.

    Only `to_anndata(backed=True, obsm=[...])` produces `ScxBackedObsmDataset`
    — the default backed path loads obsm eagerly as numpy — so without this arm
    `ScxBackedObsmDataset::set_kept_to_global` would be uncovered, and 4.0b
    could break the handle path unnoticed. Two subsets, so the composition is
    exercised rather than just the first assignment.
    """
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True, obsm=["X_emb"])
    handle = adata.obsm["X_emb"]
    assert type(handle).__name__ == "ScxBackedObsmDataset", (
        f"fixture must yield the backed handle, got {type(handle).__name__}"
    )

    first = np.arange(N_OBS) % 3 != 0
    pyscx.accel.subset_obs(adata, first)
    # Still a lazy handle — the subset must not have gathered it into an array.
    assert type(adata.obsm["X_emb"]).__name__ == "ScxBackedObsmDataset"
    assert_allclose(_as_dense(adata.obsm["X_emb"][:]), source["obsm"][first], rtol=1e-6)

    second = np.arange(first.sum()) % 2 == 0
    pyscx.accel.subset_obs(adata, second)
    expected = source["obsm"][first][second]
    assert type(adata.obsm["X_emb"]).__name__ == "ScxBackedObsmDataset"
    assert_allclose(_as_dense(adata.obsm["X_emb"][:]), expected, rtol=1e-6)
    assert adata.obsm["X_emb"].shape == expected.shape


def test_var_subset_under_an_open_time_projection(scx_path, source):
    """`varm` / `varp` must survive a var subset on a gene-selected open.

    `to_anndata(backed=True, var_names=[...])` projects X / var / layers, but the
    varm and varp bridges decode at *physical* var width — unlike the obs-axis
    bridges, which receive the open-time filter. Replaying a visible-space
    selection against a physical-width value returned a different gene's rows at
    the right shape, so anndata's validation passed and nothing complained. That
    is worse than the loud `ValueError` it replaced.
    """
    import pyscx

    picked = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    names = [f"gene_{j}" for j in picked]
    adata = pyscx.open(scx_path).to_anndata(backed=True, var_names=names)
    assert adata.n_vars == len(picked)
    # Readable, and the right genes, before any subset.
    assert_allclose(_as_dense(adata.varm["loadings"]), source["varm"][picked], rtol=1e-6)

    # No public var-subset entry point yet; drive it through filter_genes on a
    # threshold that keeps a strict subset of the visible genes.
    detected = (source["X"][:, picked] != 0).sum(axis=0)
    threshold = int(np.median(detected)) + 1
    pyscx.accel.filter_genes(adata, min_cells=threshold)

    kept_local = np.flatnonzero(detected >= threshold)
    assert 0 < len(kept_local) < len(picked), "fixture must drop some visible genes"
    v = np.asarray(picked)[kept_local]

    _assert_members(adata, source, np.ones(N_OBS, bool), np.isin(np.arange(N_VARS), v))


def test_var_subset_preserves_request_order(scx_path, source):
    """`preserve_var_order=True` + a var subset keeps X, var and layers in step.

    The composed projection has to follow presentation order, not sorted on-disk
    order — `visible_ondisk_in_presentation_order` is what carries that through
    the funnel, and nothing else covered it.
    """
    import pyscx

    # Deliberately unsorted, so sorted order and request order differ.
    picked = [21, 4, 17, 9, 28]
    names = [f"gene_{j}" for j in picked]
    adata = pyscx.open(scx_path).to_anndata(
        backed=True, var_names=names, preserve_var_order=True
    )
    assert list(adata.var_names) == names
    assert_allclose(_as_dense(adata.X.to_memory()), source["X"][:, picked], rtol=1e-6)
    assert_allclose(
        _as_dense(adata.layers["counts"].to_memory()), source["layer"][:, picked], rtol=1e-6
    )

    detected = (source["X"][:, picked] != 0).sum(axis=0)
    threshold = int(np.median(detected)) + 1
    pyscx.accel.filter_genes(adata, min_cells=threshold)

    kept = [p for p, d in zip(picked, detected) if d >= threshold]
    assert 0 < len(kept) < len(picked)
    assert list(adata.var_names) == [f"gene_{j}" for j in kept], "request order lost"
    assert_allclose(_as_dense(adata.X.to_memory()), source["X"][:, kept], rtol=1e-6)
    assert_allclose(
        _as_dense(adata.layers["counts"].to_memory()), source["layer"][:, kept], rtol=1e-6
    )


def test_comparison_shortcircuit_respects_the_projection(scx_path, source):
    """`(X > 0).sum(...)` must agree with `X.getnnz(...)` under a projection.

    `_ComparisonResult` dropped `col_projection` while keeping the projected
    `shape_val`, so the short-circuit read the physical axis: `sum()` over-counted,
    `sum(axis=1)` returned physical-width row counts, and `sum(axis=0)` raised a
    reshape error. `docs/scanpy/backed-mode.md` advertises this short-circuit as
    the `sc.pp.calculate_qc_metrics` fast path, so it reached scanpy's own QC.
    """
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    cols = [3, 7, 11, 12, 19]
    adata.X.set_col_projection(cols)
    adata._var = adata.var.iloc[cols].copy()

    x = adata.X
    nz = source["X"][:, cols] != 0
    gt = x > 0

    assert int(np.asarray(gt.sum())) == int(nz.sum())
    assert_array_equal(np.asarray(gt.sum(axis=1)).ravel(), nz.sum(axis=1))
    assert_array_equal(np.asarray(gt.sum(axis=0)).ravel(), nz.sum(axis=0))
    # ...and agrees with the getnnz path this PR already fixed.
    assert int(np.asarray(gt.sum())) == int(x.getnnz())
    # The non-short-circuit fallback materializes; it must project too.
    assert (x > 0).toarray().shape == (N_OBS, len(cols))


def test_in_memory_anndata_goes_through_anndata(source):
    """A non-SCX `X` is anndata's job, not ours.

    `subset_obs` used to hand-roll `_X = X[mask]` plus an obs/obsm slice, which
    missed `varm`/`obsp`/`varp` and `raw`. It now calls `_inplace_subset_obs` —
    the same routine `sc.pp.filter_cells` uses — so `raw` comes along for free.
    """
    import anndata
    import pandas as pd
    import pyscx
    import scipy.sparse as sp

    adata = anndata.AnnData(
        X=sp.csr_matrix(source["X"]),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(N_OBS)]),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(N_VARS)]),
        obsm={"X_emb": source["obsm"]},
        obsp={"conn": sp.csr_matrix(source["obsp"])},
    )
    adata.raw = adata

    obs_keep = np.arange(N_OBS) % 2 == 0
    pyscx.accel.subset_obs(adata, obs_keep)
    o = np.flatnonzero(obs_keep)

    assert adata.shape == (len(o), N_VARS)
    assert_allclose(_as_dense(adata.obsm["X_emb"]), source["obsm"][o], rtol=1e-6)
    assert_allclose(_as_dense(adata.obsp["conn"]), source["obsp"][np.ix_(o, o)], rtol=1e-6)
    assert adata.raw.shape[0] == len(o)
