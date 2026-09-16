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
    plans, _ = pyscx.neighborhood_plans_from_graph(
        exp, file_id=0, k=k, weight_order="desc"
    )
    want_rows, _ = reference_graph_plans(graph, k=k)
    assert as_rows(plans) == want_rows


@pytest.mark.parametrize("k", [1, 4, 9])
def test_coord_knn_matches_the_numpy_reference(lattice, k):
    exp = pyscx.open(lattice)
    coords = exp.to_anndata().obsm["spatial"]
    plans, centers = pyscx.neighborhood_plans_from_coords(exp, file_id=0, k=k)
    want_rows, want_centers = reference_coord_plans(coords, k=k)
    assert list(map(int, centers)) == want_centers
    assert as_rows(plans) == want_rows


@pytest.mark.parametrize("radius", [PITCH, PITCH * 1.5, PITCH * 2])
def test_coord_radius_matches_the_numpy_reference(lattice, radius):
    exp = pyscx.open(lattice)
    coords = exp.to_anndata().obsm["spatial"]
    plans, _ = pyscx.neighborhood_plans_from_coords(exp, file_id=0, radius=radius)
    want_rows, _ = reference_coord_plans(coords, radius=radius)
    assert as_rows(plans) == want_rows


def test_the_two_builders_agree_where_the_relations_coincide(lattice):
    """The stored graph IS the rook adjacency, and radius = pitch is the same
    relation — two independent paths through two different sections."""
    exp = pyscx.open(lattice)
    g, _ = pyscx.neighborhood_plans_from_graph(exp, file_id=0)
    c, _ = pyscx.neighborhood_plans_from_coords(exp, file_id=0, radius=PITCH)
    assert as_rows(g) == as_rows(c)


def test_include_center_false_drops_the_centre(lattice):
    exp = pyscx.open(lattice)
    plans, centers = pyscx.neighborhood_plans_from_coords(
        exp, file_id=0, k=3, include_center=False
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
    #
    # Row 31 is there for a different reason and is load-bearing: it sits in the
    # MIDDLE of the row axis, so a logical block that starts above it spans a
    # deletion and its live->physical mapping is no longer a constant offset.
    # With deletions only at 0 and 9, `k[i] - k[start]` and `i - start` agree
    # for every range above 9, and a mutation replacing the first with the
    # second passed every mid-range assertion.
    pyscx.mark_deleted(str(path), [0, 9, 31])
    return str(path)


def test_deleted_centres_and_neighbours_are_dropped(deleted_lattice):
    exp = pyscx.open(deleted_lattice)
    assert exp.has_deletions
    keep = np.ones(N, dtype=bool)
    keep[[0, 9, 31]] = False

    graph = _lattice_adata().obsp["connectivities"].tocsr()
    plans, centers = pyscx.neighborhood_plans_from_graph(exp, file_id=0)
    want_rows, want_centers = reference_graph_plans(graph, keep=keep)
    assert list(map(int, centers)) == want_centers
    assert not ({0, 9, 31} & set(want_centers))
    assert as_rows(plans) == want_rows
    for plan in plans:
        assert not ({0, 9, 31} & set(map(int, plan[1])))

    coords = _lattice_adata().obsm["spatial"]
    cplans, ccenters = pyscx.neighborhood_plans_from_coords(exp, file_id=0, k=3)
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
        pyscx.neighborhood_plans_from_coords(exp, file_id=0, k=3, radius=5.0)


def test_a_missing_key_raises_rather_than_returning_nothing(lattice):
    exp = pyscx.open(lattice)
    with pytest.raises(Exception):
        pyscx.neighborhood_plans_from_graph(exp, "distances", file_id=0)
    with pytest.raises(Exception):
        pyscx.neighborhood_plans_from_coords(exp, "X_umap", file_id=0, k=2)


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
    assert n_logical == n_physical - 3

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


def test_read_obsp_rows_mid_range_logical_block_matches_the_whole(deleted_lattice):
    """A logical range that does not start at 0 is the interesting case.

    The physical span covering live rows [a, b) is neither contiguous nor
    aligned to them, so the block read has to map each live row through the
    keep list rather than offsetting by `a`.

    ⚠️ The ranges below are chosen to straddle the fixture's MIDDLE deletion.
    Below it, `k[i] - k[start]` and `i - start` are the same number, so a
    mutation replacing the keep-list mapping with a plain offset passed every
    range this test originally used — which is what put a deletion at row 31.
    """
    exp = pyscx.open(deleted_lattice)
    whole = exp.to_anndata().obsp["connectivities"].tocsr()
    n = exp.n_obs
    for start, stop in [(0, 5), (7, 8), (11, 40), (n - 4, n), (5, 5)]:
        block = exp.read_obsp_rows("connectivities", start, stop)
        assert block.shape == (stop - start, n)
        np.testing.assert_allclose(
            block.toarray(), whole[start:stop].toarray(), rtol=0, atol=0
        )
    # And the same for the physical space, where the mapping is the identity.
    physical = exp.read_obsp_rows("connectivities", 0, exp.n_obs_physical, logical=False)
    for start, stop in [(3, 9), (20, 21)]:
        block = exp.read_obsp_rows("connectivities", start, stop, logical=False)
        np.testing.assert_allclose(
            block.toarray(), physical[start:stop].toarray(), rtol=0, atol=0
        )


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
            big_spatial, file_id=0, k=100, weight_order="desc"
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


# ---------------------------------------------------------------------------
# Review round 1 — the defects the reviewers found
# ---------------------------------------------------------------------------


def test_file_id_is_required_not_defaulted(lattice):
    """The docs say there is no safe default; the signature must agree.

    A wrong `file_id` produces a structurally valid plan that gathers a
    different file's rows with nothing raising, so a default of 0 is the one
    value that turns the documented hazard into the quiet path.
    """
    exp = pyscx.open(lattice)
    with pytest.raises(TypeError, match="file_id"):
        pyscx.neighborhood_plans_from_graph(exp)
    with pytest.raises(TypeError, match="file_id"):
        pyscx.neighborhood_plans_from_coords(exp, k=3)


def test_weight_order_picks_the_right_end_of_a_distance_graph(lattice):
    """`connectivities` and `distances` want opposite ends of the same graph."""
    exp = pyscx.open(lattice)
    graph = exp.to_anndata().obsp["connectivities"].tocsr()

    desc, _ = pyscx.neighborhood_plans_from_graph(
        exp, file_id=0, k=2, weight_order="desc"
    )
    asc, _ = pyscx.neighborhood_plans_from_graph(
        exp, file_id=0, k=2, weight_order="asc"
    )
    assert len(desc) == len(asc) == N

    # Spelled out rather than through the reference builder: for each centre,
    # the two orders must take opposite ends of its own weight list.
    for centre in range(N):
        lo, hi = graph.indptr[centre], graph.indptr[centre + 1]
        weights = sorted(
            ((float(v), int(j)) for j, v in zip(graph.indices[lo:hi], graph.data[lo:hi])
             if int(j) != centre),
            key=lambda t: (-t[0], t[1]),
        )
        if len(weights) < 3 or weights[0][0] == weights[-1][0]:
            continue  # no decidable difference on this centre
        heaviest = {j for _, j in weights[:2]}
        lightest = {j for _, j in sorted(weights, key=lambda t: (t[0], t[1]))[:2]}
        assert set(map(int, desc[centre][1])) - {centre} == heaviest
        assert set(map(int, asc[centre][1])) - {centre} == lightest
        break
    else:
        pytest.fail("no centre on this fixture could distinguish the two orders")

    with pytest.raises(ValueError, match="weight_order must be"):
        pyscx.neighborhood_plans_from_graph(exp, file_id=0, k=2, weight_order="nearest")


def test_a_wide_obsm_key_is_refused_rather_than_hanging(tmp_path):
    """A 50-component `X_pca` is a plausible slip for `"spatial"`.

    The grid search is exponential in the dimensionality, so before the cap
    this did not return. The test asserts the refusal and never calls the
    builder at d=50 without it — a regression would hang the suite rather than
    fail it.
    """
    import anndata as ad
    import pandas as pd

    n = 64
    rng = np.random.default_rng(3)
    a = ad.AnnData(
        X=sp.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), np.arange(n) % 4)),
                        shape=(n, 4)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
    )
    a.obsm["X_pca"] = rng.random((n, 50), dtype=np.float32)
    a.obsm["spatial"] = rng.random((n, 2), dtype=np.float32) * 100
    path = tmp_path / "wide.scx"
    pyscx.from_anndata(a, str(path), shard_size=16)

    exp = pyscx.open(str(path))
    with pytest.raises(Exception, match="1..=3|must be 1"):
        pyscx.neighborhood_plans_from_coords(exp, "X_pca", file_id=0, k=4)
    # And the 2-D key on the same file still works.
    plans, _ = pyscx.neighborhood_plans_from_coords(exp, "spatial", file_id=0, k=4)
    assert len(plans) == n


def test_a_closed_or_stale_handle_is_refused_by_the_builders(tmp_path):
    """Passing an `Experiment` must not launder away its lifecycle guards.

    The builders take a path and re-open the file, so without an explicit gate
    a closed or stale handle sails through while every other `Experiment` read
    refuses it.
    """
    path = tmp_path / "gate.scx"
    pyscx.from_anndata(_lattice_adata(g=4), str(path), shard_size=8)

    closed = pyscx.open(str(path))
    closed.close()
    with pytest.raises(RuntimeError, match="closed"):
        pyscx.neighborhood_plans_from_graph(closed, file_id=0)
    with pytest.raises(RuntimeError, match="closed"):
        pyscx.neighborhood_plans_from_coords(closed, file_id=0, k=2)

    stale = pyscx.open(str(path))
    calls = tmp_path / "calls.csv"
    calls.write_text("barcode,score\n" + "".join(
        f"spot_{i:03d},{i}\n" for i in range(16)))
    pyscx.obs_import(str(path), str(calls), key="obs_names", source_key="barcode")
    for fn, kwargs in (
        (pyscx.neighborhood_plans_from_graph, {}),
        (pyscx.neighborhood_plans_from_coords, {"k": 2}),
    ):
        with pytest.raises(RuntimeError, match="changed on disk|replaced on disk"):
            fn(stale, file_id=0, **kwargs)

    # A path string has no handle to be stale, and still works.
    plans, _ = pyscx.neighborhood_plans_from_graph(str(path), file_id=0)
    assert len(plans) == 16


def test_batch_plans_accepts_lists_as_well_as_numpy_arrays(lattice):
    """The typed numpy fast path must not have narrowed what is accepted."""
    exp = pyscx.open(lattice)
    plans, _ = pyscx.neighborhood_plans_from_graph(exp, file_id=0)
    as_lists = [tuple(np.asarray(a).tolist() for a in p) for p in plans[:8]]
    from_arrays = pyscx.batch_plans(plans[:8], 4)
    from_lists = pyscx.batch_plans(as_lists, 4)
    assert [list(map(int, b[1])) for b in from_arrays] == [
        list(map(int, b[1])) for b in from_lists
    ]


def test_read_obsp_rows_releases_the_gil(big_spatial):
    """The bounded read is the atlas-scale surface; it must not block Python.

    It opens the file, resolves every shard's footer schema, decodes Arrow IPC,
    counting-sorts into CSR and (on the logical path) remaps every edge — all
    of which ran under the GIL while both plan builders detached.
    """
    exp = pyscx.open(big_spatial)
    n = exp.n_obs
    duration, largest_gap = largest_gap_during(
        lambda: exp.read_obsp_rows("connectivities", 0, n)
    )
    assert duration >= MIN_MEASURABLE_S, (
        f"read_obsp_rows finished in {duration * 1000:.1f} ms — too fast to tell a held "
        "GIL from a released one. Enlarge the fixture rather than skipping."
    )
    assert largest_gap < MAX_GAP_FRACTION * duration, (
        f"read_obsp_rows starved a concurrent Python thread for {largest_gap:.3f}s of a "
        f"{duration:.3f}s read ({largest_gap / duration:.0%}) — it is holding the GIL"
    )


# ---------------------------------------------------------------------------
# Review round 2
# ---------------------------------------------------------------------------


def test_k_without_weight_order_is_refused(lattice):
    """The round-1 fix added the knob and left the wrong answer as the default.

    `weight_order="desc"` is right for the default key and silently wrong the
    moment a caller changes only the key, so
    `neighborhood_plans_from_graph(exp, "distances", file_id=0, k=8)` still
    returned each cell's farthest neighbours. Documenting that is weaker than
    refusing it.
    """
    exp = pyscx.open(lattice)
    with pytest.raises(ValueError, match="weight_order is required when k is given"):
        pyscx.neighborhood_plans_from_graph(exp, "connectivities", file_id=0, k=4)
    with pytest.raises(ValueError, match="weight_order is required when k is given"):
        pyscx.neighborhood_plans_from_graph(exp, "distances", file_id=0, k=4)

    # Without `k` there is no ranking, so nothing to state: this must work.
    plans, _ = pyscx.neighborhood_plans_from_graph(exp, file_id=0)
    assert len(plans) == N
    # And with both, as before.
    plans, _ = pyscx.neighborhood_plans_from_graph(
        exp, file_id=0, k=4, weight_order="asc"
    )
    assert len(plans) == N


def test_the_dimensionality_refusal_is_a_value_error(tmp_path):
    """Same class of mistake as `k=0`, so the same exception type.

    It was arriving as a `RuntimeError` through `loader_err_to_py`'s catch-all
    while `k=0` had been lifted to `ValueError` at both entry points.
    """
    import anndata as ad
    import pandas as pd

    n = 32
    rng = np.random.default_rng(7)
    a = ad.AnnData(
        X=sp.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), np.arange(n) % 4)),
                        shape=(n, 4)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
    )
    a.obsm["X_pca"] = rng.random((n, 12), dtype=np.float32)
    path = tmp_path / "wide2.scx"
    pyscx.from_anndata(a, str(path), shard_size=8)
    exp = pyscx.open(str(path))
    with pytest.raises(ValueError, match="1..=3|must be 1"):
        pyscx.neighborhood_plans_from_coords(exp, "X_pca", file_id=0, k=4)


def test_batch_plans_requires_single_set_inputs(lattice):
    """`sets_per_batch` must count sets, not plans."""
    exp = pyscx.open(lattice)
    plans, _ = pyscx.neighborhood_plans_from_graph(exp, file_id=0)
    once = pyscx.batch_plans(plans, 8)
    assert len(once[0][3]) - 1 == 8, "eight sets in the first batch, as asked"
    with pytest.raises(Exception, match="not a single set"):
        pyscx.batch_plans(once, 2)
