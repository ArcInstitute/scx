"""`pyscx.neighborhood_plans_*` — the R6 plan builders (W7 steps 1-2).

The parity assertions here use a **numpy/scipy reference builder** written from
the declared rules, not from the Rust implementation's helpers: a reference that
borrows the subject's comparator cannot falsify a change to it.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import pyscx

from _gil_probe import largest_gap_during

G = 8
N = G * G
PITCH = 10.0

MIN_MEASURABLE_S = 0.05
MAX_GAP_FRACTION = 0.5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _lattice_adata(g=G, pitch=PITCH):
    import anndata as ad
    import pandas as pd

    n = g * g
    coords = np.array([[i % g, i // g] for i in range(n)], dtype=np.int64) * int(pitch)
    x = sp.csr_matrix(
        (np.arange(1, n + 1, dtype=np.float32), (np.arange(n), np.arange(n) % 4)),
        shape=(n, 4),
    )
    a = ad.AnnData(
        X=x,
        obs=pd.DataFrame(index=[f"spot_{i:03d}" for i in range(n)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(4)]),
    )
    a.obsm["spatial"] = coords
    rows, cols, vals = [], [], []
    for i in range(n):
        x_, y_ = i % g, i // g
        nbrs = ([i + 1] if x_ + 1 < g else []) + ([i + g] if y_ + 1 < g else [])
        for j in nbrs:
            rows += [i, j]
            cols += [j, i]
            # Distinct weights so a top-k by weight is decidable.
            vals += [1.0 + 0.001 * j, 1.0 + 0.001 * i]
    a.obsp["connectivities"] = sp.csr_matrix(
        (vals, (rows, cols)), shape=(n, n), dtype=np.float32
    )
    return a


@pytest.fixture(scope="module")
def lattice(tmp_path_factory):
    path = tmp_path_factory.mktemp("nb") / "lattice.scx"
    # A shard size well under n_obs, so the bounded obsp read actually spans
    # several shards rather than reading one and calling it covered.
    pyscx.from_anndata(_lattice_adata(), str(path), shard_size=10)
    return str(path)


# ---------------------------------------------------------------------------
# numpy/scipy reference builders
# ---------------------------------------------------------------------------


def reference_graph_plans(graph, *, k=None, include_center=True, keep=None):
    """Plans from a scipy CSR graph, per the documented rules."""
    n = graph.shape[0]
    alive = np.ones(n, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    plans, centers = [], []
    for c in range(n):
        if not alive[c]:
            continue
        lo, hi = graph.indptr[c], graph.indptr[c + 1]
        cols = graph.indices[lo:hi]
        vals = graph.data[lo:hi]
        sel = [(int(j), float(v)) for j, v in zip(cols, vals) if alive[j]]
        if include_center:
            sel = [(j, v) for j, v in sel if j != c]
        if k is not None:
            # Weight descending, ties by column ascending.
            sel = sorted(sel, key=lambda t: (-t[1], t[0]))[:k]
        sel = sorted(sel, key=lambda t: t[0])
        rows = ([c] if include_center else []) + [j for j, _ in sel]
        plans.append(rows)
        centers.append(c)
    return plans, centers


def reference_coord_plans(coords, *, k=None, radius=None, include_center=True, keep=None):
    n = coords.shape[0]
    alive = np.ones(n, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    plans, centers = [], []
    live = np.flatnonzero(alive)
    for c in live:
        others = live[live != c]
        diff = coords[others].astype(np.float32) - coords[c].astype(np.float32)
        d2 = (diff * diff).sum(axis=1)
        order = sorted(range(len(others)), key=lambda i: (float(d2[i]), int(others[i])))
        if radius is not None:
            order = [i for i in order if float(d2[i]) <= radius * radius]
        if k is not None:
            order = order[:k]
        rows = ([int(c)] if include_center else []) + [int(others[i]) for i in order]
        plans.append(rows)
        centers.append(int(c))
    return plans, centers


def as_rows(plans):
    return [list(map(int, p[1])) for p in plans]


# ---------------------------------------------------------------------------
# Parity with the numpy reference
# ---------------------------------------------------------------------------


def test_graph_plans_match_the_numpy_reference(lattice):
    exp = pyscx.open(lattice)
    graph = exp.to_anndata().obsp["connectivities"].tocsr()
    plans, centers = pyscx.neighborhood_plans_from_graph(exp, file_id=0)
    want_rows, want_centers = reference_graph_plans(graph)
    assert list(map(int, centers)) == want_centers
    assert as_rows(plans) == want_rows


@pytest.mark.parametrize("k", [1, 2, 3, 10])
def test_graph_top_k_matches_the_numpy_reference(lattice, k):
    exp = pyscx.open(lattice)
    graph = exp.to_anndata().obsp["connectivities"].tocsr()
    plans, _ = pyscx.neighborhood_plans_from_graph(exp, k=k, file_id=0)
    want_rows, _ = reference_graph_plans(graph, k=k)
    assert as_rows(plans) == want_rows


@pytest.mark.parametrize("k", [1, 4, 9])
def test_coord_knn_matches_the_numpy_reference(lattice, k):
    exp = pyscx.open(lattice)
    coords = exp.to_anndata().obsm["spatial"]
    plans, centers = pyscx.neighborhood_plans_from_coords(exp, k=k, file_id=0)
    want_rows, want_centers = reference_coord_plans(coords, k=k)
    assert list(map(int, centers)) == want_centers
    assert as_rows(plans) == want_rows


@pytest.mark.parametrize("radius", [PITCH, PITCH * 1.5, PITCH * 2])
def test_coord_radius_matches_the_numpy_reference(lattice, radius):
    exp = pyscx.open(lattice)
    coords = exp.to_anndata().obsm["spatial"]
    plans, _ = pyscx.neighborhood_plans_from_coords(exp, radius=radius, file_id=0)
    want_rows, _ = reference_coord_plans(coords, radius=radius)
    assert as_rows(plans) == want_rows


def test_the_two_builders_agree_where_the_relations_coincide(lattice):
    """The stored graph IS the rook adjacency, and radius = pitch is the same
    relation — two independent paths through two different sections."""
    exp = pyscx.open(lattice)
    g, _ = pyscx.neighborhood_plans_from_graph(exp, file_id=0)
    c, _ = pyscx.neighborhood_plans_from_coords(exp, radius=PITCH, file_id=0)
    assert as_rows(g) == as_rows(c)


def test_include_center_false_drops_the_centre(lattice):
    exp = pyscx.open(lattice)
    plans, centers = pyscx.neighborhood_plans_from_coords(
        exp, k=3, include_center=False, file_id=0
    )
    for plan, centre in zip(plans, centers):
        assert int(centre) not in list(map(int, plan[1]))
        assert set(map(int, plan[2])) == {1}, "every member is a neighbour"


# ---------------------------------------------------------------------------
# Deletion vectors
# ---------------------------------------------------------------------------


@pytest.fixture
def deleted_lattice(tmp_path):
    path = tmp_path / "deleted.scx"
    pyscx.from_anndata(_lattice_adata(), str(path), shard_size=10)
    # Row 9 is an interior spot (a neighbour of several centres); row 0 is a
    # corner. Deleting both covers "a deleted centre" and "a deleted neighbour"
    # in one file.
    pyscx.mark_deleted(str(path), [0, 9])
    return str(path)


def test_deleted_centres_and_neighbours_are_dropped(deleted_lattice):
    exp = pyscx.open(deleted_lattice)
    assert exp.has_deletions
    keep = np.ones(N, dtype=bool)
    keep[[0, 9]] = False

    graph = _lattice_adata().obsp["connectivities"].tocsr()
    plans, centers = pyscx.neighborhood_plans_from_graph(exp, file_id=0)
    want_rows, want_centers = reference_graph_plans(graph, keep=keep)
    assert list(map(int, centers)) == want_centers
    assert 0 not in want_centers and 9 not in want_centers
    assert as_rows(plans) == want_rows
    for plan in plans:
        assert not ({0, 9} & set(map(int, plan[1])))

    coords = _lattice_adata().obsm["spatial"]
    cplans, ccenters = pyscx.neighborhood_plans_from_coords(exp, k=3, file_id=0)
    cwant, cwant_centers = reference_coord_plans(coords, k=3, keep=keep)
    assert list(map(int, ccenters)) == cwant_centers
    assert as_rows(cplans) == cwant


def test_drop_deleted_false_keeps_the_physical_graph(deleted_lattice):
    exp = pyscx.open(deleted_lattice)
    plans, centers = pyscx.neighborhood_plans_from_graph(
        exp, file_id=0, drop_deleted=False
    )
    assert len(centers) == N, "every physical row is a centre when nothing is dropped"
    graph = _lattice_adata().obsp["connectivities"].tocsr()
    want_rows, _ = reference_graph_plans(graph)
    assert as_rows(plans) == want_rows


def test_a_built_plan_gathers_the_same_cells_as_a_hand_built_one(deleted_lattice):
    """§12.6's deletion clause: the gather over the built plans equals the
    gather over the equivalent hand-built plan."""
    exp = pyscx.open(deleted_lattice)
    plans, centers = pyscx.neighborhood_plans_from_graph(exp, file_id=0)
    ds = pyscx.SparseCellSetDataset([deleted_lattice])
    try:
        for plan, centre in list(zip(plans, centers))[:8]:
            rows = list(map(int, plan[1]))
            hand = (
                np.zeros(len(rows), dtype=np.uint32),
                np.asarray(rows, dtype=np.uint64),
                np.asarray([0] + [1] * (len(rows) - 1), dtype=np.int32),
                np.asarray([0, len(rows)], dtype=np.int64),
            )
            a = ds.gather(*plan)
            b = ds.gather(*hand)
            for key in ("indptr", "indices", "data", "cell_indices", "role_tags",
                        "set_offsets", "file_ids"):
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
            assert int(a["cell_indices"][0]) == int(centre)
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# The plans drive both consumers
# ---------------------------------------------------------------------------


def test_plans_feed_gather_and_iter_with_plans(lattice):
    exp = pyscx.open(lattice)
    plans, _ = pyscx.neighborhood_plans_from_graph(exp, file_id=0)
    batched = pyscx.batch_plans(plans, 8)
    ds = pyscx.SparseCellSetDataset([lattice])
    try:
        direct = [ds.gather(*p) for p in batched]
        streamed = list(ds.iter_with_plans(batched))
        assert len(streamed) == len(direct) == 8
        for a, b in zip(direct, streamed):
            np.testing.assert_array_equal(a["cell_indices"], b["cell_indices"])
            np.testing.assert_array_equal(a["set_offsets"], b["set_offsets"])
            np.testing.assert_array_equal(a["role_tags"], b["role_tags"])
            np.testing.assert_array_equal(a["data"], b["data"])
    finally:
        ds.close()


def test_plans_collate(lattice):
    """The plans must be collatable, which needs a true indptr — a set_offsets
    that merely never decreases passes the gather and fails the collate."""
    exp = pyscx.open(lattice)
    plans, _ = pyscx.neighborhood_plans_from_graph(exp, file_id=0)
    ds = pyscx.SparseCellSetDataset([lattice])
    try:
        batch = ds.gather(*pyscx.batch_plans(plans, 8)[0])
    finally:
        ds.close()
    n_sets = len(batch["set_offsets"]) - 1
    n_rows = len(batch["cell_indices"])
    k_dec = 2
    out = pyscx.collate_cellset_gathered(
        batch["indptr"],
        batch["indices"],
        batch["data"],
        batch["set_offsets"],
        batch["cell_indices"],
        batch["file_ids"],
        batch["role_tags"],
        k_dec,
        np.tile(np.arange(k_dec, dtype=np.int32), n_sets),
        np.zeros(n_rows * k_dec, dtype=np.uint8),
        np.zeros(n_rows, dtype=np.uint8),
        np.full(n_sets, 4, dtype=np.uint32),
        4,
        "pass_through",
        4,
    )
    assert out["encoder_gene_ids"].shape[0] == len(batch["cell_indices"]) * 4


def test_batch_plans_is_seed_reproducible(lattice):
    exp = pyscx.open(lattice)
    plans, _ = pyscx.neighborhood_plans_from_graph(exp, file_id=0)
    a = pyscx.batch_plans(plans, 6, shuffle_seed=11)
    b = pyscx.batch_plans(plans, 6, shuffle_seed=11)
    c = pyscx.batch_plans(plans, 6, shuffle_seed=12)
    flat = lambda v: [int(r) for p in v for r in p[1]]  # noqa: E731
    assert flat(a) == flat(b)
    assert flat(a) != flat(c)
    assert sorted(flat(a)) == sorted(flat(pyscx.batch_plans(plans, 6)))


def test_file_id_is_stamped_and_not_guessed(lattice):
    exp = pyscx.open(lattice)
    plans, _ = pyscx.neighborhood_plans_from_graph(exp, file_id=2)
    assert all(int(f) == 2 for p in plans for f in p[0])


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------


def test_coords_needs_exactly_one_of_k_or_radius(lattice):
    exp = pyscx.open(lattice)
    with pytest.raises(ValueError, match="got neither"):
        pyscx.neighborhood_plans_from_coords(exp, file_id=0)
    with pytest.raises(ValueError, match="got both"):
        pyscx.neighborhood_plans_from_coords(exp, k=3, radius=5.0, file_id=0)


def test_a_missing_key_raises_rather_than_returning_nothing(lattice):
    exp = pyscx.open(lattice)
    with pytest.raises(Exception):
        pyscx.neighborhood_plans_from_graph(exp, "distances", file_id=0)
    with pytest.raises(Exception):
        pyscx.neighborhood_plans_from_coords(exp, "X_umap", k=2, file_id=0)


def test_the_builders_accept_a_path_a_pathlib_and_a_handle(lattice):
    import pathlib

    exp = pyscx.open(lattice)
    a, _ = pyscx.neighborhood_plans_from_graph(lattice, file_id=0)
    b, _ = pyscx.neighborhood_plans_from_graph(pathlib.Path(lattice), file_id=0)
    c, _ = pyscx.neighborhood_plans_from_graph(exp, file_id=0)
    assert as_rows(a) == as_rows(b) == as_rows(c)


# ---------------------------------------------------------------------------
# Experiment.read_obsp_rows / obsp_keys
# ---------------------------------------------------------------------------


def test_obsp_keys_matches_to_anndata(lattice):
    exp = pyscx.open(lattice)
    assert exp.obsp_keys() == list(exp.to_anndata().obsp.keys())


def test_read_obsp_rows_equals_the_whole_matrix_sliced(lattice):
    exp = pyscx.open(lattice)
    whole = exp.to_anndata().obsp["connectivities"].tocsr()
    for start, stop in [(0, N), (0, 1), (5, 5), (7, 23), (N - 3, N)]:
        block = exp.read_obsp_rows("connectivities", start, stop)
        assert block.shape == (stop - start, N)
        np.testing.assert_allclose(
            block.toarray(), whole[start:stop].toarray(), rtol=0, atol=0
        )


def test_read_obsp_rows_logical_and_physical_differ_under_deletions(deleted_lattice):
    exp = pyscx.open(deleted_lattice)
    n_logical, n_physical = exp.n_obs, exp.n_obs_physical
    assert n_logical == n_physical - 2

    logical = exp.read_obsp_rows("connectivities", 0, n_logical)
    assert logical.shape == (n_logical, n_logical)
    # Identical to what `to_anndata()` materialises for the whole graph, which
    # is the established logical-space rule.
    np.testing.assert_allclose(
        logical.toarray(),
        exp.to_anndata().obsp["connectivities"].toarray(),
        rtol=0,
        atol=0,
    )

    physical = exp.read_obsp_rows("connectivities", 0, n_physical, logical=False)
    assert physical.shape == (n_physical, n_physical)
    assert physical.nnz > logical.nnz, "physical keeps the edges of deleted rows"

    with pytest.raises(IndexError):
        exp.read_obsp_rows("connectivities", 0, n_physical)


# ---------------------------------------------------------------------------
# GIL release
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def big_spatial(tmp_path_factory):
    """Sized so each builder clears MIN_MEASURABLE_S on a **release** build.

    Sizing against a debug `.so` is the trap phase 3 hit three times: the test
    then passes locally and either skips or fails on the build that matters.
    Measured on a release build at this size: graph 0.444 s / 6.0 % gap, coords
    0.588 s / 3.6 % — roughly 8x and 14x margin against MAX_GAP_FRACTION. On a
    debug build both are several times slower, so the margin only widens.

    The shape is `n` centres x `deg` edges rather than more centres with fewer
    edges on purpose. Roughly 20 ms of every call is the plans-to-numpy
    conversion, which **necessarily** holds the GIL — 20,000 tuples of four
    arrays cannot be built without it — and that cost scales with the centre
    count while the detached build scales with the edge count. A fixture with
    many centres and a thin graph measures the conversion, not the build: at
    60,000 centres x 12 edges the graph arm was 39 % gap on a debug build and
    would have failed outright on release.
    """
    import anndata as ad
    import pandas as pd

    n = 20_000
    deg = 200
    rng = np.random.default_rng(5)
    coords = rng.random((n, 2), dtype=np.float32) * 5_000.0
    x = sp.csr_matrix(
        (np.ones(n, dtype=np.float32), (np.arange(n), np.arange(n) % 8)), shape=(n, 8)
    )
    a = ad.AnnData(
        X=x,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(8)]),
    )
    a.obsm["spatial"] = coords
    rows = np.repeat(np.arange(n), deg)
    cols = rng.integers(0, n, size=n * deg)
    a.obsp["connectivities"] = sp.csr_matrix(
        (rng.random(n * deg).astype(np.float32) + 0.1, (rows, cols)), shape=(n, n)
    )
    path = tmp_path_factory.mktemp("nbgil") / "big.scx"
    pyscx.from_anndata(a, str(path), shard_size=2_500)
    return str(path)


@pytest.mark.parametrize("which", ["graph", "coords"])
def test_the_builders_release_the_gil(big_spatial, which):
    if which == "graph":
        call = lambda: pyscx.neighborhood_plans_from_graph(  # noqa: E731
            big_spatial, k=100, file_id=0
        )
    else:
        call = lambda: pyscx.neighborhood_plans_from_coords(  # noqa: E731
            big_spatial, k=60, file_id=0
        )
    duration, largest_gap = largest_gap_during(call)
    assert duration >= MIN_MEASURABLE_S, (
        f"{which} builder finished in {duration * 1000:.1f} ms — too fast to tell a "
        "held GIL from a released one. Enlarge the fixture rather than skipping: "
        "a test that always skips asserts nothing."
    )
    assert largest_gap < MAX_GAP_FRACTION * duration, (
        f"the {which} plan builder starved a concurrent Python thread for "
        f"{largest_gap:.3f}s of a {duration:.3f}s build "
        f"({largest_gap / duration:.0%}) — it is holding the GIL"
    )
