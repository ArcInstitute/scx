"""Tests for SparseCellSetDataset (SCX-DATA-LOADER Phase 2.4).

The native sparse cell-set loader gathers role-tagged, multi-file cell-set
batches as sparse CSR (the §4.4 contract) and must match the backed
`to_anndata().X[rows]` reference, in plan order, across multiple files.
"""

import multiprocessing as mp

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def two_scx(synthetic_adata, tmp_dir):
    import pyscx

    p0 = str(tmp_dir / "f0.scx")
    p1 = str(tmp_dir / "f1.scx")
    pyscx.from_anndata(synthetic_adata, p0)
    pyscx.from_anndata(synthetic_adata, p1)
    return p0, p1


# Plan: one batch, two single-file sets (file 0 scattered, file 1 with a dup).
_FILE_IDS = [0, 0, 0, 1, 1, 1]
_ROWS = [0, 5, 2, 10, 10, 3]
_ROLE_TAGS = [0, 0, 0, 1, 1, 1]
_SET_OFFSETS = [0, 3, 6]
_PLAN = (_FILE_IDS, _ROWS, _ROLE_TAGS, _SET_OFFSETS)


def test_batch_dict_schema_and_dtypes(two_scx):
    import pyscx

    p0, p1 = two_scx
    ds = pyscx.SparseCellSetDataset([p0, p1])
    assert ds.n_files == 2
    batches = list(ds.iter_with_plans(iter([_PLAN])))
    assert len(batches) == 1
    b = batches[0]

    assert set(b) == {
        "indptr",
        "indices",
        "data",
        "shape",
        "cell_indices",
        "file_ids",
        "set_offsets",
        "role_tags",
    }
    assert b["indptr"].dtype == np.int64
    assert b["indices"].dtype == np.int32
    assert b["data"].dtype == np.float32
    assert b["cell_indices"].dtype == np.uint64
    assert b["file_ids"].dtype == np.uint32
    assert b["set_offsets"].dtype == np.int64
    assert b["role_tags"].dtype == np.int32

    assert tuple(b["shape"]) == (6, ds.n_cols)
    assert b["cell_indices"].tolist() == _ROWS
    assert b["file_ids"].tolist() == _FILE_IDS
    assert b["set_offsets"].tolist() == _SET_OFFSETS
    assert b["role_tags"].tolist() == _ROLE_TAGS


def test_csr_rows_match_backed_reference(two_scx):
    import pyscx

    p0, p1 = two_scx
    ds = pyscx.SparseCellSetDataset([p0, p1])
    b = next(iter(ds.iter_with_plans(iter([_PLAN]))))

    got = sp.csr_matrix(
        (b["data"], b["indices"], b["indptr"]), shape=tuple(b["shape"])
    )
    refs = {
        0: pyscx.open(p0).to_anndata(backed=True).X,
        1: pyscx.open(p1).to_anndata(backed=True).X,
    }
    for j, (fid, row) in enumerate(zip(_FILE_IDS, _ROWS)):
        expected = refs[fid][row].toarray()
        np.testing.assert_array_equal(got[j].toarray(), expected)


def test_remap_emits_global_indices(synthetic_adata, tmp_dir):
    import pyscx

    p0 = str(tmp_dir / "g0.scx")
    pyscx.from_anndata(synthetic_adata, p0)
    n_vars = synthetic_adata.n_vars
    # local g → global g + 1000 (injective, no sentinels).
    table = [g + 1000 for g in range(n_vars)]
    ds = pyscx.SparseCellSetDataset(
        [p0], remap_tables=[table], n_global_genes=n_vars + 1000
    )
    assert ds.n_cols == n_vars + 1000
    b = next(iter(ds.iter_with_plans(iter([([0], [7], [0], [0, 1])]))))
    ref = pyscx.open(p0).to_anndata(backed=True).X[7]
    expected_global = (ref.indices.astype(np.int64) + 1000)
    np.testing.assert_array_equal(np.sort(b["indices"]), np.sort(expected_global))


# --- malformed plans surface as clean exceptions, not a worker crash -------


def test_file_id_out_of_range_raises_runtimeerror(two_scx):
    import pyscx

    p0, p1 = two_scx
    ds = pyscx.SparseCellSetDataset([p0, p1])
    # file_id 9 but only 2 files.
    bad = ([0, 9], [0, 1], [0, 0], [0, 2])
    with pytest.raises(RuntimeError, match="file_id"):
        list(ds.iter_with_plans(iter([bad])))


def test_bad_set_offsets_raises_runtimeerror(two_scx):
    import pyscx

    p0, p1 = two_scx
    ds = pyscx.SparseCellSetDataset([p0, p1])
    # set_offsets[1] past total_rows would panic the slice without validation.
    bad = ([0, 0], [0, 1], [0, 0], [0, 99])
    with pytest.raises(RuntimeError, match="set_offsets"):
        list(ds.iter_with_plans(iter([bad])))


def test_row_out_of_range_raises_indexerror(two_scx):
    import pyscx

    p0, p1 = two_scx
    ds = pyscx.SparseCellSetDataset([p0, p1])
    n_obs = pyscx.open(p0).n_obs
    bad = ([0, 0], [0, n_obs + 100], [0, 0], [0, 2])
    with pytest.raises(IndexError):
        list(ds.iter_with_plans(iter([bad])))


def test_plan_extraction_failure_names_the_expected_tuple(two_scx):
    """A plan of the wrong *shape* must name the four-array layout it wanted.

    Pinned because ORG-9.10-3 folds the two Python->Rust plan adapters into one
    generic. This message is the only thing distinguishing the cell-set arm's
    diagnostic from the pair arm's, and nothing asserted it before — so a dedup
    that collapsed both onto pyo3's generic extract error would be silent.
    """
    import pyscx

    p0, p1 = two_scx
    ds = pyscx.SparseCellSetDataset([p0, p1])
    # A two-array plan: the pair arm's shape, not this class's.
    with pytest.raises(RuntimeError) as excinfo:
        list(ds.iter_with_plans(iter([([0], [0])])))
    msg = str(excinfo.value)
    assert "sparse plan extraction failed" in msg, msg
    for field in ("file_ids", "rows", "role_tags", "set_offsets"):
        assert field in msg, f"{field} missing from: {msg}"


# --- fork-safety -----------------------------------------------------------


def _child_build_and_iter(path, conn):
    try:
        import pyscx

        ds = pyscx.SparseCellSetDataset([path])  # built post-fork → OK
        b = next(iter(ds.iter_with_plans(iter([([0], [1], [0], [0, 1])]))))
        conn.send(("ok", int(b["cell_indices"][0])))
    except BaseException as exc:  # noqa: BLE001
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


def test_fork_lazy_post_fork_construction(two_scx):
    p0, _ = two_scx
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_child_build_and_iter, args=(p0, child_conn))
    proc.start()
    status, payload = parent_conn.recv()
    proc.join(timeout=30)
    assert status == "ok", f"child failed: {payload}"
    assert payload == 1


def test_fork_pre_fork_dataset_raises_in_child(two_scx):
    p0, _ = two_scx
    import pyscx

    ds = pyscx.SparseCellSetDataset([p0])  # built in parent

    def _child(conn):
        try:
            list(ds.iter_with_plans(iter([([0], [1], [0], [0, 1])])))
            conn.send(("ok", None))
        except BaseException as exc:  # noqa: BLE001
            conn.send(("err", repr(exc)))
        finally:
            conn.close()

    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_child, args=(child_conn,))
    proc.start()
    status, payload = parent_conn.recv()
    proc.join(timeout=30)
    assert status == "err" and "num_workers=0" in payload


# ---------------------------------------------------------------------------
# §9.3 / §9.4 — lifecycle: bounded teardown, and an opportunistic plan pull.
# ---------------------------------------------------------------------------


def test_feedback_plan_generator_yields_every_batch(two_scx):
    """§9.4 on the sparse path.

    `PlanPrefetchIter::refill` had the same blocking-`recv` loop as the paired
    loader's, so a generator that answers only after seeing the previous batch
    wedged it the same way. The generator blocks here rather than merely
    asserting it was not called ahead — the plan-pull thread buffers up to
    `lookahead` plans by design, so being *asked* early is fine; answering
    early is what a curriculum sampler cannot do.
    """
    import threading

    import pyscx

    p0, p1 = two_scx
    ds = pyscx.SparseCellSetDataset([p0, p1])
    seen = []
    n_plans = 5
    ack = threading.Semaphore(0)

    def curriculum():
        for i in range(n_plans):
            if i > 0:
                assert ack.acquire(timeout=30), (
                    f"generator waited 30s for batch {i - 1} and never got it — "
                    "the loader is demanding plans ahead of the feedback signal"
                )
            assert len(seen) == i
            yield ([0, 1], [i, i + 1], [0, 1], [0, 1, 2])

    for b in ds.iter_with_plans(curriculum(), lookahead=4):
        seen.append(b["shape"][0])
        ack.release()

    assert len(seen) == n_plans
    assert all(n == 2 for n in seen)


def test_close_is_idempotent_and_terminal(two_scx):
    """§9.3 — `close()` releases the prefetch engine's tokio runtime off the
    GIL, and is terminal because that runtime is built exactly once so a
    forked child can never inherit it."""
    import pyscx

    p0, p1 = two_scx
    ds = pyscx.SparseCellSetDataset([p0, p1])
    assert ds.closed is False
    assert ds.n_files == 2

    ds.close()
    assert ds.closed is True
    ds.close()  # idempotent

    with pytest.raises(RuntimeError, match="closed"):
        ds.iter_with_plans(iter([_PLAN]))
    with pytest.raises(RuntimeError, match="closed"):
        _ = ds.n_files
    with pytest.raises(RuntimeError, match="closed"):
        ds.memory_budget()


def test_repr_never_raises_when_closed(two_scx):
    import pyscx

    p0, p1 = two_scx
    ds = pyscx.SparseCellSetDataset([p0, p1])
    assert "n_files=" in repr(ds)
    ds.close()
    assert repr(ds) == "SparseCellSetDataset(closed)"


def test_teardown_mid_flight_is_bounded(two_scx):
    """Abandoning an epoch with prefetches outstanding tears down cleanly.

    The ordering guard: the iterator holds its own reference to the loader, so
    dropping the dataset first cannot take sole ownership and the runtime is
    released when the iterator drops. See the paired loader's namesake test for
    why the GIL-release half of §9.3 is carried by construction rather than by
    an assertion here.
    """
    import time

    import pyscx

    p0, p1 = two_scx
    ds = pyscx.SparseCellSetDataset([p0, p1])
    plans = [([0, 1], [i, i + 1], [0, 1], [0, 1, 2]) for i in range(8)]
    it = ds.iter_with_plans(iter(plans), lookahead=4)
    next(it)

    t0 = time.monotonic()
    del ds
    del it
    elapsed = time.monotonic() - t0
    assert elapsed < 10.0, f"mid-flight teardown took {elapsed:.2f}s"


def test_close_then_drain_the_iterator(two_scx):
    """The twin of `IndexPlanDataset`'s namesake, on the path where the
    ownership graph is worse.

    Here the iterator's stored `process` closure owns an
    `Arc<SparseCellSetLoader>` which owns a second `Arc<PrefetchEngine>`, so
    `close()` cannot reach the engine and neither can the iterator's own `Drop`
    body (a `Drop` body runs before its struct's fields). The teardown deadline
    therefore lives in the runtime newtype itself, and this exercises the
    ordering end to end.
    """
    import time

    import pyscx

    p0, p1 = two_scx
    ds = pyscx.SparseCellSetDataset([p0, p1])
    plans = [([0, 1], [i, i + 1], [0, 1], [0, 1, 2]) for i in range(4)]
    it = ds.iter_with_plans(iter(plans), lookahead=4)
    next(it)
    ds.close()

    t0 = time.monotonic()
    rest = list(it)
    elapsed = time.monotonic() - t0

    assert len(rest) == 3, "closing the dataset must not truncate a live iterator"
    assert elapsed < 10.0, f"drain-after-close took {elapsed:.2f}s"
    assert list(it) == []


def test_closed_dataset_raises_before_running_user_code(two_scx):
    """A closed dataset must report *that*, not run the caller's `__iter__`
    first and surface whatever it raises."""
    import pyscx

    p0, p1 = two_scx
    ds = pyscx.SparseCellSetDataset([p0, p1])
    ds.close()

    class Exploding:
        def __iter__(self):
            raise AssertionError("user __iter__ ran on a closed dataset")

    with pytest.raises(RuntimeError, match="closed"):
        ds.iter_with_plans(Exploding())

    # Same for the manual same-length validation: closed beats that ValueError.
    # Only that one — pyo3 converts the tuple parameter before the Rust body
    # runs, so a closed dataset handed a *wrong-arity* plan still gets the arity
    # ValueError, which is why this plan is well-formed apart from its lengths.
    with pytest.raises(RuntimeError, match="closed"):
        ds.suggested_cache_shards(([0, 0], [1], [0, 0], [0, 2]))


# ---------------------------------------------------------------------------
# The per-dataset block-index escape hatch.
#
# `SparseCellSetDataset` had this knob (as `scatter_sidecar=`, PR #299) and lost
# it when the Scx1 decode sidecar was removed: the kwarg was deleted and the
# equivalent `scatter_block_index=` was never carried over, so every reader the
# cell-set engine builds inherited the process default (`True`). The two docs
# describing the `False` default went on describing behaviour the code no longer
# had, and nothing failed — there was no test on either side.
# ---------------------------------------------------------------------------

_FRAMED_N_OBS = 400
_FRAMED_N_VARS = 60


@pytest.fixture
def framed_scx(tmp_dir):
    """A row-group-framed (v4) single-shard file, which is the *only* shape on
    which this flag is observable.

    `block_index_eligible` requires `shard_is_framed`, so on an unframed file
    both settings take the full-shard path and a test written against one would
    pass in both directions while asserting nothing. Framed `shufdelta` (no Scx1
    sidecar) leaves the block index as the only random-access route, and 8 plan
    rows against 400 shard rows clears the `group_len * 4 < shard_rows` window
    the predicate also requires.
    """
    import anndata as ad
    import pyscx

    path = str(tmp_dir / "framed.scx")
    X = sp.random(_FRAMED_N_OBS, _FRAMED_N_VARS, density=0.05, format="csr",
                  random_state=0)
    X.data = np.round(X.data * 10 + 1).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obs["cell_id"] = [f"c{i}" for i in range(_FRAMED_N_OBS)]
    pyscx.from_anndata(adata, path, codec="shufdelta", row_group_rows=16)
    return path


_FRAMED_ROWS = [5, 70, 140, 200, 250, 300, 350, 399]
_FRAMED_PLAN = ([0] * 8, _FRAMED_ROWS, [0] * 8, [0, 8])


def _drive(path, **kwargs):
    """Two identical batches, so a warmed shard shows up as a cache *hit*.

    One batch cannot tell the routes apart on the cache counters: the first pass
    misses either way, and only the second shows whether anything was retained.
    """
    import pyscx

    ds = pyscx.SparseCellSetDataset([path], cache_shards=16, **kwargs)
    try:
        nnz_per_batch = [
            int(b["indptr"][-1]) for b in ds.iter_with_plans(iter([_FRAMED_PLAN] * 2))
        ]
        assert len(nnz_per_batch) == 2, "premise: both plans must yield a batch"
        assert all(n > 0 for n in nnz_per_batch), (
            "premise: the plan must actually read data — an all-empty gather "
            "would leave every counter at zero and both assertions below would "
            "hold vacuously"
        )
        # Read before closing: `close()` is terminal and the accessors raise
        # afterwards.
        return ds.cache_metrics()
    finally:
        ds.close()


def test_scatter_block_index_defaults_off(framed_scx):
    """Constructed with no kwarg, the gather warms whole shards into the LRU.

    This is the assertion the class lost. It is about the *default*, so it must
    not pass the kwarg — passing `scatter_block_index=False` here would still be
    green against a build whose default is `True`.
    """
    m = _drive(framed_scx)
    assert m["block_index_groups"] == 0, (
        "the cell-set loader must default to the full-shard warm+cache path; "
        f"block_index_groups={m['block_index_groups']} means it took the "
        "row-group path"
    )
    assert m["full_shard_groups"] > 0
    assert m["hits"] > 0, (
        "the point of the default: the second batch is served from the LRU the "
        "first one populated as a whole shard."
    )
    assert m["row_group_hits"] + m["row_group_misses"] == 0, (
        "the whole-shard path retains no row groups"
    )


def test_scatter_block_index_true_opts_into_the_block_index_path(framed_scx):
    """The escape hatch, in the other direction — and the fixture's own guard.

    If this arm went green with `block_index_groups == 0` the fixture would be
    silently ineligible (unframed, or too many rows per group), and its sibling
    above would be asserting nothing at all.
    """
    m = _drive(framed_scx, scatter_block_index=True)
    assert m["block_index_groups"] > 0
    assert m["full_shard_groups"] == 0
    assert m["hits"] + m["misses"] == 0, (
        "the row-group path never inserts a whole shard, so the whole-shard "
        "counters stay at zero"
    )
    # OPT-FORMATIO-1: what it does retain is the touched row groups — the
    # second identical batch is served from them, under the same byte budget.
    # Each group is decoded once (the singleflight dedups the L2 warm racing
    # the gather); every later look-up — the second batch's gather, and the
    # prefetcher's own warm of the second plan — is a hit, so hits are at least
    # the two gathers' worth.
    assert m["row_group_misses"] > 0, "the first batch decoded row groups"
    assert m["row_group_hits"] >= 2 * m["row_group_misses"], (
        f"both batches must be served from the groups the first decoded: {m}"
    )


# ---------------------------------------------------------------------------
# Prefetch counters on the sparse arm (ORG-9.10-1 drift (b))
# ---------------------------------------------------------------------------
#
# Until the fold, `PlanPrefetchIter` had no `IterMetrics` at all, so this class
# could only *infer* the L2 warm-skip from the gather-side `block_index_groups`.
# `SparseCellSetBatchIter.metrics()` now reports it directly, which is the half
# of ORG-9.10-4 that was blocked on ORG-9.10-1.

_PREFETCH_KEYS = {
    "prefetch_tasks_spawned",
    "prefetch_skipped_cache_hit",
    "prefetch_skipped_in_flight",
    "prefetch_skipped_block_index",
}


def _drive_iter(path, **kwargs):
    """Same two-batch drive as `_drive`, but returning the *iterator's*
    `metrics()` rather than the dataset's cache counters.

    The iterator has to be held in a local: the counters are per-iter, and
    reading them off a temporary that has already been collected would sample a
    fresh, zeroed handle.
    """
    import pyscx

    ds = pyscx.SparseCellSetDataset([path], cache_shards=16, **kwargs)
    try:
        it = ds.iter_with_plans(iter([_FRAMED_PLAN] * 2))
        nnz_per_batch = [int(b["indptr"][-1]) for b in it]
        assert len(nnz_per_batch) == 2, "premise: both plans must yield a batch"
        assert all(n > 0 for n in nnz_per_batch), (
            "premise: the plan must actually read data — an all-empty gather "
            "would leave every counter at zero and the assertions below would "
            "hold vacuously"
        )
        return it.metrics()
    finally:
        ds.close()


def test_batch_iter_metrics_match_the_index_plan_shape(framed_scx):
    """`metrics()` is dict-of-dicts with the same keys as the pair loader's.

    A schema pin, not a value pin: the two classes are meant to be readable by
    the same diagnostic code, and this is what stops one of them drifting a key
    name.
    """
    m = _drive_iter(framed_scx)
    assert set(m) == {"cache", "prefetch"}
    assert set(m["prefetch"]) == _PREFETCH_KEYS
    # The cache half is the same dict `cache_metrics()` returns.
    assert set(m["cache"]) >= {"hits", "misses", "full_shard_groups", "block_index_groups"}


def test_block_index_prefetch_skip_is_counted_on_the_sparse_arm(framed_scx):
    """`scatter_block_index=True` leaves the framed shard undecoded, and says so.

    The value of this over `cache_metrics()["block_index_groups"]` is that it
    reports the *prefetch* decision (L2) rather than the gather's (L1); before
    the fold only the latter was observable here. They are **not** required to
    agree: at `lookahead=0` the prefetch never runs and every counter here is 0
    while the gather still adopts the route, and a peer sharing the cache can
    warm a shard in between. `block_index_groups` stays the route authority.

    The `False` arm is the fixture's own guard: if the file were unframed, or
    the plan too wide for the `group_len * 4 < shard_rows` window, both arms
    would read 0 and this test would assert nothing.
    """
    on = _drive_iter(framed_scx, scatter_block_index=True)["prefetch"]
    off = _drive_iter(framed_scx, scatter_block_index=False)["prefetch"]

    assert on["prefetch_skipped_block_index"] > 0, (
        "scatter_block_index=True on a framed file must skip warming the shard "
        "so the gather can take the row-group path"
    )
    assert on["prefetch_tasks_spawned"] == 0, (
        "nothing should have been warmed: the only shard the plan touches was "
        "left to the block index"
    )
    assert off["prefetch_skipped_block_index"] == 0, (
        "with the gate off the shard is warmed, not skipped — a non-zero count "
        "here means the per-reader flag is not reaching block_index_eligible"
    )
    assert off["prefetch_tasks_spawned"] > 0, (
        "the default route warms the whole shard, which is a spawned prefetch"
    )


# ---------------------------------------------------------------------------
# ORG-9.10-4 / the 9b finding: the unframed-file preflight.
#
# `IndexPlanDataset` warns when `scatter_block_index=True` meets a file with no
# row-group-framed shard — the fast path cannot fire, so every batch
# full-shard-decodes. `SparseCellSetDataset` took the same kwarg and said
# nothing, so the flag was a silent no-op there and the only way to find out was
# to read `cache_metrics()["block_index_groups"]` afterwards.
#
# The aggregate is deliberately ANY-across-readers, not all: one framed file in
# the set means the route can fire for that file's rows, so warning would be
# wrong.
# ---------------------------------------------------------------------------


@pytest.fixture
def two_unframed_scx(synthetic_adata, tmp_dir):
    """Two all-unframed (v3) files — `row_group_rows=0`.

    Load-bearing: `pyscx.from_anndata` writes `format_version = 4` for every
    codec, so simply omitting `row_group_rows=` yields a *framed* file and every
    assertion below would invert. `codec="shufdelta"` refuses `row_group_rows=0`,
    so this fixture must leave the codec at the default.
    """
    import pyscx

    p0 = str(tmp_dir / "u0.scx")
    p1 = str(tmp_dir / "u1.scx")
    pyscx.from_anndata(synthetic_adata, p0, row_group_rows=0)
    pyscx.from_anndata(synthetic_adata, p1, row_group_rows=0)
    return p0, p1


def test_unframed_scatter_emits_preflight_warning(two_unframed_scx):
    """Opting into the block-index route on an all-unframed set must say so."""
    import pyscx

    p0, p1 = two_unframed_scx
    with pytest.warns(UserWarning, match="row-group framed") as rec:
        ds = pyscx.SparseCellSetDataset([p0, p1], scatter_block_index=True)
    msg = str(rec[0].message)
    assert "SparseCellSetDataset" in msg, msg
    assert "2 files" in msg, f"a multi-file set must be described as such: {msg}"
    assert "scx optimize" in msg, f"must name the fix: {msg}"

    # Warn-and-continue, never a refusal: it still gathers. Two single-file
    # sets — a set spanning files needs remap tables, which is a different error.
    plan = ([0, 1], [0, 5], [0, 1], [0, 1, 2])
    b = next(iter(ds.iter_with_plans(iter([plan]))))
    assert b["shape"][0] == 2
    assert ds.cache_metrics()["block_index_groups"] == 0
    ds.close()


def test_unframed_scatter_off_is_silent(two_unframed_scx):
    """The default (`False`) and an explicit `False` are the intended
    full-shard path on an unframed file, not a footgun to warn about."""
    import warnings as _w

    import pyscx

    p0, p1 = two_unframed_scx
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)
        pyscx.SparseCellSetDataset([p0, p1]).close()
        pyscx.SparseCellSetDataset([p0, p1], scatter_block_index=False).close()


def test_framed_scatter_has_no_preflight_warning(framed_scx):
    """The fast path is available, so there is nothing to warn about."""
    import warnings as _w

    import pyscx

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        pyscx.SparseCellSetDataset([framed_scx], scatter_block_index=True).close()
    assert not [w for w in caught if "row-group framed" in str(w.message)], (
        [str(w.message) for w in caught]
    )


def test_global_kill_switch_suppresses_the_preflight(two_unframed_scx):
    """`SCX_SCATTER_BLOCK_INDEX=0` must silence the preflight entirely.

    With the route globally off, reframing could not enable it either, so the
    warning would send the caller to do something useless. The switch is
    memoized in a `OnceLock`, so this has to run in a fresh interpreter; nothing
    pinned the suppression before.

    ⚠️ It pins the *suppression*, not the ordering. The framing scan skipping
    when the switch is off is invisible from here — a version that scanned first
    and suppressed afterwards passes this test too, which is what it did before
    review round 2. That ordering is pinned in Rust, by
    `python::preflight_decision_tests`, which asserts the scan closure went
    uncalled.
    """
    import os
    import subprocess
    import sys
    import textwrap

    p0, p1 = two_unframed_scx
    src = textwrap.dedent(
        f"""
        import warnings
        warnings.simplefilter("error", UserWarning)
        import pyscx
        pyscx.SparseCellSetDataset([{p0!r}, {p1!r}], scatter_block_index=True).close()
        pyscx.IndexPlanDataset({p0!r}, scatter_block_index=True).close()
        print("silent")
        """
    )
    env = {**os.environ, "SCX_SCATTER_BLOCK_INDEX": "0"}
    r = subprocess.run(
        [sys.executable, "-c", src], capture_output=True, text=True, env=env
    )
    assert r.returncode == 0, r.stderr
    assert "silent" in r.stdout, (r.stdout, r.stderr)


def test_suggested_cache_shards_rejects_a_two_tuple_plan(two_scx):
    """The plan argument keeps its arity check.

    `role_tags` / `set_offsets` are taken as bare objects so the probe does not
    copy two sequences it never reads — but a two-array plan (the *pair*
    loader's shape) is still the shape mistake worth catching, and dropping the
    element types must not drop the arity with them.
    """
    import pyscx

    p0, p1 = two_scx
    ds = pyscx.SparseCellSetDataset([p0, p1])
    # pyo3 reports a tuple-arity mismatch as ValueError, not TypeError.
    with pytest.raises(ValueError, match="tuple of length 4"):
        ds.suggested_cache_shards(([0, 0], [0, 5]))
    # ...and the four-tuple still works, so the check is not rejecting everything.
    assert ds.suggested_cache_shards(([0, 0], [0, 5], [0, 0], [0, 2])) >= 1
    ds.close()


def test_one_framed_file_in_the_set_suppresses_the_warning(framed_scx,
                                                           two_unframed_scx):
    """ANY reader framed, not ALL: the route can still fire for that file's rows.

    This is the assertion that distinguishes the aggregate from a per-reader
    check — the three tests above all hold under `all()` as well.
    """
    import warnings as _w

    import pyscx

    unframed0, _ = two_unframed_scx
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        pyscx.SparseCellSetDataset(
            [unframed0, framed_scx], scatter_block_index=True
        ).close()
    assert not [w for w in caught if "row-group framed" in str(w.message)], (
        [str(w.message) for w in caught]
    )
