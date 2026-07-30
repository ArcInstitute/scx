"""anndata owns axis subsetting; SCX only keeps its lazy bridges off disk.

Phase 4.0b, the structural half of §9.18. 4.0a funnelled every mutating op
through one internal routine that hand-rolled the axis bookkeeping. That routine
now registers SCX handles with anndata's `as_view` / `_subset` / `to_memory`
seams and drives anndata's own view/copy machinery, so `obs`, `var`, `uns`,
unused categorical levels, `raw`, atomicity and *every* aligned member come from
anndata rather than from an enumeration here.

Two things do not come for free, and are what most of this file pins:

* **Nothing may materialize.** anndata's `AnnData.copy()` calls `.copy()` on the
  matrix, and an SCX handle's `.copy()` materializes *deliberately* — that is
  the documented `adata[mask].copy()` workflow. The accelerators go through
  `_mutated_copy(X=view.X, …)` instead, so `X` stays a handle.
* **Nothing may decode.** Reading `adata.varm` / `obsp` / `varp` validates every
  entry, which pulls a lazy bridge off disk. The bridges are detached for the
  duration of anndata's subset and re-attached with the selection recorded.

4.0a's `test_axis_subset.py` and `test_lazy_row_axis_projection.py` are the
correctness oracle and must pass unchanged; this file covers what 4.0b adds.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

N_OBS, N_VARS = 40, 24


@pytest.fixture
def source():
    rng = np.random.RandomState(4028)
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
    import anndata
    import pandas as pd
    import pyscx
    import scipy.sparse as sp

    adata = anndata.AnnData(
        X=sp.csr_matrix(source["X"]),
        obs=pd.DataFrame(
            {"group": pd.Categorical(["a" if i % 3 else "b" for i in range(N_OBS)])},
            index=[f"cell_{i}" for i in range(N_OBS)],
        ),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(N_VARS)]),
        layers={"counts": sp.csr_matrix(source["layer"])},
        obsm={"X_emb": source["obsm"]},
        varm={"loadings": source["varm"]},
        obsp={"conn": sp.csr_matrix(source["obsp"])},
        varp={"corr": sp.csr_matrix(source["varp"])},
    )
    path = str(tmp_dir / "hooks.scx")
    pyscx.from_anndata(adata, path)
    return path


def _dense(value):
    if hasattr(value, "to_memory"):
        value = value.to_memory()
    return np.asarray(value.todense() if hasattr(value, "todense") else value)


def _assert_members(adata, source, o, v):
    assert adata.shape == (len(o), len(v))
    assert_allclose(_dense(adata.X), source["X"][np.ix_(o, v)], rtol=1e-6)
    assert_allclose(_dense(adata.layers["counts"]), source["layer"][np.ix_(o, v)], rtol=1e-6)
    assert_allclose(_dense(adata.obsm["X_emb"]), source["obsm"][o], rtol=1e-6)
    assert_allclose(_dense(adata.varm["loadings"]), source["varm"][v], rtol=1e-6)
    assert_allclose(_dense(adata.obsp["conn"]), source["obsp"][np.ix_(o, o)], rtol=1e-6)
    assert_allclose(_dense(adata.varp["corr"]), source["varp"][np.ix_(v, v)], rtol=1e-6)


# ---------------------------------------------------------------------------
# The hooks: plain anndata indexing now works on a backed X.
# ---------------------------------------------------------------------------


def test_var_axis_view_no_longer_raises(scx_path, source):
    """`adata[:, mask]` used to raise `NotImplementedError`.

    anndata's `as_view` had no registration for `ScxBackedSparseDataset`, so the
    var axis was unreachable through the public API entirely — including
    `_inplace_subset_var`, which is what `sc.pp.filter_genes` calls. The obs axis
    happened to work because `_subset`'s `object` fallback lands on
    `__getitem__`.
    """
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    mask = np.zeros(N_VARS, bool)
    mask[[1, 3, 5, 7, 9, 11]] = True

    view = adata[:, mask]
    assert view.is_view
    assert view.shape == (N_OBS, int(mask.sum()))
    assert_allclose(_dense(view.X), source["X"][:, mask], rtol=1e-6)


@pytest.mark.parametrize("axis", ["obs", "var"])
def test_copy_of_a_view_equals_the_materialized_subset(scx_path, source, axis):
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    if axis == "obs":
        mask = np.arange(N_OBS) % 3 != 0
        o, v = np.flatnonzero(mask), np.arange(N_VARS)
        sub = adata[mask].copy()
    else:
        mask = np.arange(N_VARS) % 4 != 0
        o, v = np.arange(N_OBS), np.flatnonzero(mask)
        sub = adata[:, mask].copy()

    # `.copy()` materializes — the documented "subset, then preprocess" path.
    import scipy.sparse as sp

    assert sp.issparse(sub.X), f"adata[...].copy() should materialize X, got {type(sub.X)}"
    _assert_members(sub, source, o, v)


def test_copy_still_materializes(scx_path):
    """`adata.X.copy()` is documented as "same as to_memory()"."""
    import pyscx
    import scipy.sparse as sp

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    assert sp.issparse(adata.X.copy())
    assert sp.issparse(adata.X.to_memory())
    assert sp.issparse(adata.layers["counts"].copy())


def test_to_memory_materializes_every_handle(scx_path, source):
    """`AnnData.to_memory()` reaches inside now.

    `anndata._core.file_backing.to_memory` passes unrecognized objects straight
    through, so before the registration `adata.to_memory().X` was still a
    handle — a silently backed "in-memory" object.
    """
    import pyscx
    import scipy.sparse as sp

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    mask = np.arange(N_OBS) % 2 == 0
    mem = adata[mask].to_memory()

    assert sp.issparse(mem.X)
    assert sp.issparse(mem.layers["counts"])
    _assert_members(mem, source, np.flatnonzero(mask), np.arange(N_VARS))


# ---------------------------------------------------------------------------
# The in-place accelerators must stay out of core, and off disk.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["filter_genes", "filter_cells", "subset_obs", "subset_var"])
def test_in_place_ops_keep_x_lazy(scx_path, op):
    """No accelerator may materialize `X` — that is the whole point of backed.

    Routing them through anndata's `_inplace_subset_*` would: `AnnData.copy()`
    calls `.copy()` on the matrix, and an SCX handle's `.copy()` materializes.
    `rebuild_via_anndata` substitutes `view.X` for that one call.
    """
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    before = type(adata.X).__name__
    assert before == "ScxBackedSparseDataset"

    if op == "filter_genes":
        pyscx.accel.filter_genes(adata, min_cells=int(N_OBS * 0.5))
    elif op == "filter_cells":
        pyscx.accel.filter_cells(adata, min_genes=int(N_VARS * 0.4))
    elif op == "subset_obs":
        pyscx.accel.subset_obs(adata, np.arange(N_OBS) % 3 != 0)
    else:
        pyscx.accel.subset_var(adata, np.arange(N_VARS) % 3 != 0)

    assert type(adata.X).__name__ == before, "the op materialized X"
    assert 0 < adata.n_obs <= N_OBS and 0 < adata.n_vars <= N_VARS


@pytest.mark.parametrize("op", ["filter_genes", "subset_obs"])
def test_in_place_ops_do_not_decode_the_lazy_bridges(scx_path, op):
    """Neither axis may pull `varm` / `obsp` / `varp` off disk.

    anndata's `_mutated_copy` reads *all five* aligned mappings whichever axis
    moved, and reading one validates every entry, which decodes. So an obs
    subset would decode `varm` and `varp` too if the bypass only covered the
    stores its own axis touches.
    """
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    assert "0 materialized" in repr(adata._obsp), "fixture must start un-decoded"

    if op == "filter_genes":
        pyscx.accel.filter_genes(adata, min_cells=int(N_OBS * 0.5))
    else:
        pyscx.accel.subset_obs(adata, np.arange(N_OBS) % 3 != 0)

    for store in ("_varm", "_obsp", "_varp"):
        assert "0 materialized" in repr(getattr(adata, store)), (
            f"the {op} decoded {store}"
        )


def test_backed_obsm_handle_survives_the_rebuild(scx_path, source):
    """A backed embedding must not be gathered by the subset.

    `_mutated_copy` calls `.copy()` on every aligned value it was not handed,
    and `ScxBackedObsmDataset.copy()` materializes — so the rebuild passes
    `obsm=` explicitly, keeping handles as handles.
    """
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True, obsm=["X_emb"])
    assert type(adata.obsm["X_emb"]).__name__ == "ScxBackedObsmDataset"

    keep = np.arange(N_OBS) % 3 != 0
    pyscx.accel.subset_obs(adata, keep)

    assert type(adata.obsm["X_emb"]).__name__ == "ScxBackedObsmDataset"
    assert_allclose(_dense(adata.obsm["X_emb"]), source["obsm"][keep], rtol=1e-6)


# ---------------------------------------------------------------------------
# What anndata now owns, that the hand-rolled funnel did not.
# ---------------------------------------------------------------------------


def test_raw_follows_an_obs_subset(scx_path, source):
    """Carried gap from 4.0a, closed by delegating.

    `adata.raw` is obs-aligned and scanpy subsets it, but the hand-rolled funnel
    left it at the original row count. anndata's `_init_as_view` does
    `self._raw = adata_ref.raw[oidx]`.
    """
    import anndata
    import pyscx
    import scipy.sparse as sp

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    raw_x = np.arange(N_OBS * N_VARS, dtype=np.float32).reshape(N_OBS, N_VARS)
    adata.raw = anndata.AnnData(X=sp.csr_matrix(raw_x), var=adata.var.copy())

    keep = np.arange(N_OBS) % 2 == 0
    pyscx.accel.subset_obs(adata, keep)

    assert adata.raw.shape[0] == int(keep.sum())
    assert_allclose(_dense(adata.raw.X), raw_x[keep], rtol=0)


def test_unused_categories_are_dropped(scx_path):
    """The two routes used to disagree; now there is one route.

    anndata drops categorical levels a subset emptied (`_remove_unused_categories`
    in `_init_as_view`); the backed route kept them, so `groupby` and any DE
    that enumerates groups saw a level with zero cells.
    """
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    assert set(adata.obs["group"].cat.categories) == {"a", "b"}

    keep = np.asarray(adata.obs["group"] == "a")
    pyscx.accel.subset_obs(adata, keep)

    assert list(adata.obs["group"].cat.categories) == ["a"], (
        "the emptied 'b' level survived the subset"
    )


def test_a_failed_subset_leaves_the_object_untouched(scx_path, source):
    """Atomicity — 4.0a could half-apply, this cannot.

    anndata builds the whole replacement `AnnData` and only then swaps it in, so
    a raise part-way leaves the original intact. The bypass wrapper is the one
    seam outside that, and it restores the bridges on error.

    A `Lock` in `uns` is the injection: `_mutated_copy` deep-copies `uns`, and a
    lock cannot be pickled.
    """
    import threading

    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    adata.uns["not_copyable"] = threading.Lock()

    with pytest.raises(TypeError):
        pyscx.accel.subset_obs(adata, np.arange(N_OBS) % 2 == 0)

    # Shape, matrix type and every member are exactly as before...
    assert adata.shape == (N_OBS, N_VARS)
    assert type(adata.X).__name__ == "ScxBackedSparseDataset"
    del adata.uns["not_copyable"]
    _assert_members(adata, source, np.arange(N_OBS), np.arange(N_VARS))

    # ...including the bridges, which were detached when the failure hit.
    for store in ("_obsp", "_varp"):
        assert type(getattr(adata, store)).__name__ == "ScxLazyPairwiseMapping", (
            f"{store} was not re-attached after the failure"
        )


def test_a_failure_after_the_swap_still_reattaches_the_bridges(scx_path, source):
    """The bypass must not leak a detached store on *any* path.

    `detach_lazy_bridges` replaces each `ScxLazy*Mapping` with an empty dict for
    the duration of anndata's rebuild. A failure *after* the swap — in
    `selection_from_x` or while recording the deferred selection — used to
    return early with the bridges still held, leaving `adata._varm` and friends
    as empty dicts. That turns a recoverable error into a `KeyError` from an
    unrelated call much later, which is strictly worse than the failure that
    caused it.

    Driven through the var axis on a lazily-transformed `X`, where an unsorted
    composed projection is rejected (a lazy dataset stores its projection
    sorted, so it cannot express a reorder) — a failure that lands after the
    rebuild has already begun.
    """
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    before = {store: type(getattr(adata, store)).__name__ for store in ("_varm", "_varp", "_obsp")}
    assert before["_varm"] == "ScxLazyVarmMapping", "fixture must start with bridges"

    # A no-op is skipped and a mask is always ascending, so reach the failure
    # through the store type instead: swap a bridge for something that is not a
    # mapping at all, so `apply_bridge_subset` raises after the rebuild.
    adata._varm = _BrokenBridge()
    with pytest.raises(Exception):
        pyscx.accel.subset_obs(adata, np.arange(N_OBS) % 2 == 0)

    # Whatever happened, no store may be left as the placeholder dict.
    for store in ("_varp", "_obsp"):
        assert type(getattr(adata, store)).__name__ == before[store], (
            f"{store} was left detached after a post-swap failure"
        )


class _BrokenBridge:
    """Not a mapping and not an SCX bridge — reaching it must not lose the others."""


@pytest.fixture
def multishard_path(source, tmp_dir):
    """The same matrix at `shard_size=10`, so `n_shards > 1`.

    Load-bearing: a non-monotone `kept_to_global` only misbehaves across shard
    boundaries. On a single-shard file `partition_point` returns the whole
    slice, so the masked kernels come out right and every small fixture passes.
    """
    import anndata
    import pandas as pd
    import pyscx
    import scipy.sparse as sp

    adata = anndata.AnnData(
        X=sp.csr_matrix(source["X"]),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(N_OBS)]),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(N_VARS)]),
    )
    path = str(tmp_dir / "multishard.scx")
    pyscx.from_anndata(adata, path, shard_size=10)
    return path


def test_reordered_rows_materialize_instead_of_building_a_bad_window(
    multishard_path, source
):
    """`adata[[5, 0, 30]]` must not build a non-monotone `kept_to_global`.

    That vector is a **construction invariant: strictly ascending** — every
    masked column kernel `partition_point`s it rather than scanning. Handed a
    reordered map they compute a garbage range and index out of bounds, which
    surfaces as a `PanicException` with a wrapped `u64` out of `X.sum(axis=0)` /
    `getnnz` / HVG / QC. Registering `_subset` made that reachable from plain
    `adata[order]`, where anndata's `object` fallback used to materialize.

    So a selection that cannot be expressed as a lazy window materializes, which
    is precisely the pre-hook behaviour. This must be exercised on a
    **multi-shard** file: on one shard `partition_point` returns everything and
    the bug is invisible.
    """
    import pyscx

    adata = pyscx.open(multishard_path).to_anndata(backed=True)
    assert adata.X.n_shards > 1, "fixture must be multi-shard or this proves nothing"

    # Deliberately spans several 10-row shards, so a bad range is fatal.
    order = [5, 0, 30, 3]
    view = adata[order]

    # The aggregates that panicked. Correct values, no panic.
    assert_allclose(
        np.asarray(view.X.sum(axis=0)).ravel(), source["X"][order].sum(axis=0), rtol=1e-5
    )
    assert_array_equal(
        np.asarray(view.X.getnnz(axis=0)).ravel(), (source["X"][order] != 0).sum(axis=0)
    )
    assert_allclose(_dense(view.X), source["X"][order], rtol=1e-6)
    assert list(view.obs_names) == [f"cell_{i}" for i in order]

    # Duplicates are the same class — they make the map non-strictly-ascending.
    dup = adata[[2, 2, 7]]
    assert_allclose(_dense(dup.X), source["X"][[2, 2, 7]], rtol=1e-6)
    assert_allclose(
        np.asarray(dup.X.sum(axis=0)).ravel(), source["X"][[2, 2, 7]].sum(axis=0), rtol=1e-5
    )

    # A negative index resolves before the check: `[-1, 0]` reads as ascending
    # but is descending once normalized.
    neg = adata[[-1, 0]]
    assert_allclose(_dense(neg.X), source["X"][[N_OBS - 1, 0]], rtol=1e-6)

    # An *ascending* selection still takes the lazy window — the guard must not
    # have disabled the fast path wholesale.
    assert type(adata[[1, 4, 9]].X).__name__ == "ScxBackedSparseDataset"


def test_unsorted_and_duplicated_columns_are_honoured(scx_path, source):
    """The var axis *can* express a reorder, and must get it right.

    `set_col_projection_ordered` gathers over the sorted set and carries a
    separate `col_presentation` permutation, so request order survives lazily.
    Duplicate columns are the exception: it dedups, which would silently narrow
    `X` while anndata expects the repeat — so those materialize instead.
    """
    import pyscx

    vorder = [7, 2, 11]
    vsub = pyscx.open(scx_path).to_anndata(backed=True)[:, vorder].copy()
    assert list(vsub.var_names) == [f"gene_{j}" for j in vorder]
    assert_allclose(_dense(vsub.X), source["X"][:, vorder], rtol=1e-6)
    assert_allclose(_dense(vsub.varm["loadings"]), source["varm"][vorder], rtol=1e-6)

    # Reordered-but-unique stays lazy on a backed handle...
    lazy = pyscx.open(scx_path).to_anndata(backed=True)[:, vorder]
    assert type(lazy.X).__name__ == "ScxBackedSparseDataset"

    # ...while a repeated column must not be deduped behind anndata's back.
    dup = pyscx.open(scx_path).to_anndata(backed=True)[:, [4, 4, 9]]
    assert dup.n_vars == 3
    assert_allclose(_dense(dup.X), source["X"][:, [4, 4, 9]], rtol=1e-6)


def test_a_full_index_through_subset_does_not_install_a_deletion_vector(source, tmp_dir):
    """`adata[np.arange(n_obs)]` must not cost the CSC sidecar.

    The mutating ops guard this with an all-kept early return, but `_subset` is
    now reachable directly. *Any* `kept_to_global` — even the identity — closes
    the CSC capability gate (`as_column_source` returns `None`), so a full-index
    view would permanently downgrade the `gpu_csc_v3` CSC-direct DE route on a
    file nobody actually subset. Asserted through the gate itself rather than
    the private field.
    """
    import anndata
    import pandas as pd
    import pyscx
    import scipy.sparse as sp

    adata = anndata.AnnData(
        X=sp.csr_matrix(source["X"]),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(N_OBS)]),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(N_VARS)]),
    )
    path = str(tmp_dir / "full_index_csc.scx")
    pyscx.from_anndata(adata, path, csc="always", csc_cols_per_shard=8)

    backed = pyscx.open(path).to_anndata(backed=True)
    expected = pyscx.accel.col_sums(backed.X, prefer_format="csc")

    view = pyscx.open(path).to_anndata(backed=True)[np.arange(N_OBS)]
    assert view.X.shape == (N_OBS, N_VARS)
    # Raises "CSC requested but unavailable ... deletion vector is active" if an
    # identity `kept_to_global` was installed.
    assert_allclose(pyscx.accel.col_sums(view.X, prefer_format="csc"), expected, rtol=0)


def test_a_prefix_slice_is_a_real_subset(scx_path, source):
    """`adata[0:k]` must shrink `X`, not be mistaken for the identity.

    The all-kept elision above keys off `composed[i] == i`, which a *prefix*
    also satisfies — so without a length check `adata[0:25]` left `X` at full
    height while `obs` shrank to 25, and the next `AnnData(...)` raised
    "`obs` must have as many rows as `X`". `pyscx.iter_chunks` is built on
    exactly this expression.
    """
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    half = N_OBS // 2
    chunk = adata[0:half].copy()
    assert chunk.shape == (half, N_VARS)
    assert_allclose(_dense(chunk.X), source["X"][:half], rtol=1e-6)

    chunks = list(pyscx.iter_chunks(adata, chunk_size=half))
    assert sum(c.n_obs for c in chunks) == N_OBS


def test_subset_var_matches_the_anndata_route(scx_path, source):
    """`pyscx.accel.subset_var` == `adata[:, mask].copy()`, minus the copy.

    Before 4.0b the only way to apply an arbitrary gene mask to a backed X was
    `adata.X.set_col_projection(...)`, which bypasses the funnel and leaves
    `var` behind.
    """
    import pyscx

    mask = np.zeros(N_VARS, bool)
    mask[[2, 3, 5, 8, 13, 21]] = True

    accel = pyscx.open(scx_path).to_anndata(backed=True)
    pyscx.accel.subset_var(accel, mask)

    reference = pyscx.open(scx_path).to_anndata(backed=True)[:, mask].copy()

    assert list(accel.var_names) == list(reference.var_names)
    _assert_members(accel, source, np.arange(N_OBS), np.flatnonzero(mask))
    assert_allclose(_dense(accel.X), _dense(reference.X), rtol=0)
