"""Fork-safety acceptance test for ``pyscx.TrainingDataset``.

The diagnostic reproducers (`test_fork_deadlock*.py`) confirmed the
historical deadlock under bare ``multiprocessing.Process`` and (when
``torch`` is installed) under ``torch.utils.data.DataLoader``. Those
tests still serve as narrow probes. **This** test pins the post-fix
contract end-to-end: a real
``DataLoader(num_workers=2)`` driving the lazy-construct-in-``__iter__``
shim that ``cell-load-scx`` and ``state-scx`` use, asserting every cell
index appears exactly once across all worker processes.

Coverage matrix:

- Task 4.1 ``test_fork_num_workers_2`` — fork start method, single-epoch.
- Task 4.2 ``test_fork_persistent_workers_3_epochs`` — fork start method,
  ``persistent_workers=True`` + 3 epochs (exercises per-epoch
  ``start_epoch`` / ``join_epoch_handles`` under fork).
- Task 4.3 ``test_num_workers_0_baseline`` — locks down that the
  num_workers=0 path stays green after the rayon-pool refactor.
- Task 4.4 ``test_spawn_num_workers_2`` — ``start_method="spawn"`` baseline:
  this should already pass independent of any fix and serves as evidence
  that the lifecycle is correct, fork-specific issues are all that's left.

Run with::

    .venv/bin/pytest pyscx/tests/test_fork_safety.py -v

Each test enforces a 30 s deadline (Task 4.1 spec) so a regression that
re-introduces the rayon-after-fork hang surfaces as a test failure rather
than a CI hang.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest


pytestmark = [
    pytest.mark.skipif(
        sys.platform != "linux",
        reason="fork start method is Linux-only; spawn is the macOS/Windows default",
    ),
]


# Spec: ~64 cells × 16 genes (Phase 4.1). Split across 4 files so a 2-worker
# DataLoader gets 2 files each — mirrors the per-file sharding pattern used
# by `state-scx`'s `ScxStateAdapter` (`worker_info.id :: num_workers`).
N_FILES = 4
CELLS_PER_FILE = 16
N_VARS = 16
TOTAL_CELLS = N_FILES * CELLS_PER_FILE  # 64

# Hard deadline per test. Phase 4 spec says 30s; matches the diagnostic
# tests' deadline so a regression presents identically.
TEST_DEADLINE_SEC = 30.0


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _make_one_fixture(path: str, file_idx: int, n_obs: int, n_vars: int) -> None:
    """Build one .scx file. Cell global IDs are derived as
    ``file_idx * CELLS_PER_FILE + local_idx`` so the parent test can verify
    coverage by reading back ``obs["cell_id"]`` (encoded as the global int)
    rather than relying on within-file row ordering.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    import pyscx

    rng = np.random.default_rng(seed=file_idx)
    dense = rng.integers(0, 10, size=(n_obs, n_vars), dtype=np.int32).astype(np.float32)
    dense[rng.random((n_obs, n_vars)) < 0.6] = 0
    X = sp.csr_matrix(dense)

    # Each cell's global ID is encoded into its `cell_id` obs column so we
    # can recover global identity from a TrainingDataset batch even though
    # the loader yields per-file local indices in `cell_indices`.
    global_ids = [file_idx * CELLS_PER_FILE + i for i in range(n_obs)]
    obs = pd.DataFrame(
        {"global_cell_id": np.asarray(global_ids, dtype=np.int64)},
        index=[f"cell_{g}" for g in global_ids],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])

    adata = ad.AnnData(X=X, obs=obs, var=var)
    pyscx.from_anndata(adata, path)


@pytest.fixture(scope="module")
def fixture_paths(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """Module-scoped: build the 4 fixture files once, share across tests."""
    tmp = tmp_path_factory.mktemp("fork_safety")
    paths: list[str] = []
    for i in range(N_FILES):
        p = str(tmp / f"f{i}.scx")
        _make_one_fixture(p, file_idx=i, n_obs=CELLS_PER_FILE, n_vars=N_VARS)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# IterableDataset shim
# ---------------------------------------------------------------------------
#
# Mirrors `state-scx/src/state/emb/data/_scx_adapter.py:86-112` (per-file
# sharding via `worker_info.id :: num_workers`). The inner
# `pyscx.TrainingDataset` is constructed lazily inside `__iter__` so the
# fork-detection PID check in `scx-loader/src/python/training.rs` does not
# fire — the dataset's `creation_pid` matches the worker's PID. This is
# the configuration that hung indefinitely before Phase 2 landed.

torch_data: Any = pytest.importorskip("torch.utils.data")


class _ShardedShim(torch_data.IterableDataset):
    def __init__(self, paths: list[str], batch_size: int = 8) -> None:
        super().__init__()
        self.paths = list(paths)
        self.batch_size = batch_size

    def __iter__(self):
        # Local imports so the worker process performs them post-fork.
        import numpy as np
        import torch.utils.data as _td

        import pyscx

        info = _td.get_worker_info()
        if info is None:
            my_paths = self.paths
        else:
            my_paths = self.paths[info.id :: info.num_workers]

        if not my_paths:
            return

        # We yield (global_cell_id, file_idx) tuples so the collator at
        # the parent side can sanity-check coverage without needing the
        # dense matrix.
        for fp in my_paths:
            file_idx = self.paths.index(fp)
            ds = pyscx.TrainingDataset(
                fp,
                batch_size=self.batch_size,
                normalize=False,
                log1p=False,
                obs_columns=["global_cell_id"],
            )
            for batch in ds:
                # `obs["global_cell_id"]` is a numpy int64 array of shape
                # (batch_n_rows,). Convert to int and yield each cell.
                gids = batch["obs"]["global_cell_id"]
                gids_arr = np.asarray(gids, dtype=np.int64).reshape(-1)
                for gid in gids_arr.tolist():
                    yield int(gid), file_idx
            # Explicitly close the inner dataset so the per-file rayon pool
            # and tokio runtime are released between files (matches the
            # recommended `close()` pattern from Phase 2.5).
            ds.close()


# ---------------------------------------------------------------------------
# Driver: run a DataLoader epoch in a forked / spawned outer process so a
# hang inside the worker pool surfaces as a test failure rather than
# wedging pytest itself.
# ---------------------------------------------------------------------------


def _passthrough_collate(batch: Any) -> Any:
    """Top-level passthrough collate so the DataLoader can pickle it under
    `start_method="spawn"` (lambdas are not picklable). We test the
    fork-safety contract, not collator correctness."""
    return batch


def _drive_dataloader(
    paths: list[str],
    num_workers: int,
    persistent_workers: bool,
    n_epochs: int,
    conn: Any,
) -> None:
    """Outer-process worker: build the DataLoader, drain `n_epochs` epochs,
    push the per-epoch sorted lists of (gid, file_idx) tuples back over the
    pipe so the parent can assert coverage."""
    try:
        import torch.utils.data as _td

        ds = _ShardedShim(paths, batch_size=8)
        loader = _td.DataLoader(
            ds,
            batch_size=None,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=_passthrough_collate,
        )
        epoch_results: list[list[tuple[int, int]]] = []
        for _ in range(n_epochs):
            seen: list[tuple[int, int]] = []
            for item in loader:
                gid, file_idx = item
                seen.append((int(gid), int(file_idx)))
            epoch_results.append(sorted(seen))
        conn.send(("ok", epoch_results))
    except BaseException as exc:
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


def _run_outer_process(
    paths: list[str],
    *,
    start_method: str,
    num_workers: int,
    persistent_workers: bool = False,
    n_epochs: int = 1,
) -> list[list[tuple[int, int]]]:
    """Run `_drive_dataloader` inside an outer Process so we can enforce a
    deadline. The outer process uses `start_method` for itself; it then
    instantiates a torch DataLoader whose own workers inherit the same
    multiprocessing context (torch defaults to the active default).
    """
    ctx = mp.get_context(start_method)
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_drive_dataloader,
        args=(paths, num_workers, persistent_workers, n_epochs, child_conn),
        daemon=False,
    )
    t0 = time.monotonic()
    proc.start()
    child_conn.close()
    proc.join(timeout=TEST_DEADLINE_SEC)
    elapsed = time.monotonic() - t0

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2.0)
        pytest.fail(
            f"DataLoader(start_method={start_method!r}, num_workers={num_workers}, "
            f"persistent_workers={persistent_workers}, n_epochs={n_epochs}) did not "
            f"finish within {TEST_DEADLINE_SEC}s — fork-safety regression"
        )

    if not parent_conn.poll(0.0):
        pytest.fail(
            f"DataLoader driver exited (rc={proc.exitcode}) without sending status "
            f"(elapsed {elapsed:.2f}s)"
        )
    status, payload = parent_conn.recv()
    parent_conn.close()
    if status == "err":
        pytest.fail(f"DataLoader driver raised: {payload}")
    assert status == "ok"
    return payload  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Coverage assertions
# ---------------------------------------------------------------------------


def _assert_one_epoch_covers_every_cell_once(
    epoch: list[tuple[int, int]],
) -> None:
    """Every global cell ID 0..TOTAL_CELLS-1 must appear exactly once."""
    gids = [g for g, _f in epoch]
    assert len(gids) == TOTAL_CELLS, (
        f"expected {TOTAL_CELLS} cells, got {len(gids)}; "
        f"missing={sorted(set(range(TOTAL_CELLS)) - set(gids))[:10]} "
        f"duplicate={sorted(g for g in gids if gids.count(g) > 1)[:10]}"
    )
    assert sorted(gids) == list(range(TOTAL_CELLS))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fork_num_workers_2(fixture_paths: list[str]) -> None:
    """Phase 4.1: ``DataLoader(num_workers=2, persistent_workers=False)``
    under fork start method drives one full epoch with every cell index
    appearing exactly once. **This is the test that pins the rayon-pool
    fix from Phase 2.0.**"""
    epochs = _run_outer_process(
        fixture_paths,
        start_method="fork",
        num_workers=2,
        persistent_workers=False,
        n_epochs=1,
    )
    assert len(epochs) == 1
    _assert_one_epoch_covers_every_cell_once(epochs[0])


def test_fork_persistent_workers_3_epochs(fixture_paths: list[str]) -> None:
    """Phase 4.2: ``persistent_workers=True`` keeps the same worker
    processes alive across epochs, so each epoch re-enters the worker's
    `__iter__` and rebuilds a fresh `pyscx.TrainingDataset` per file. This
    exercises the per-epoch ``start_epoch`` / ``join_epoch_handles`` path
    under fork — the secondary lifecycle hardening from Phase 2.1–2.4.

    Three epochs each cover every cell exactly once.
    """
    epochs = _run_outer_process(
        fixture_paths,
        start_method="fork",
        num_workers=2,
        persistent_workers=True,
        n_epochs=3,
    )
    assert len(epochs) == 3
    for i, epoch in enumerate(epochs):
        try:
            _assert_one_epoch_covers_every_cell_once(epoch)
        except AssertionError as e:
            raise AssertionError(f"epoch {i}: {e}") from None


def test_num_workers_0_baseline(fixture_paths: list[str]) -> None:
    """Phase 4.3: lock down that the existing ``num_workers=0`` path stays
    green after the rayon-pool refactor. No fork happens here — the
    DataLoader runs in the calling process — but this guards against an
    accidental regression where the per-pipeline rayon pool changes the
    serial-iteration semantics.
    """
    epochs = _run_outer_process(
        fixture_paths,
        start_method="fork",  # outer process; doesn't matter for nw=0
        num_workers=0,
        persistent_workers=False,
        n_epochs=1,
    )
    assert len(epochs) == 1
    _assert_one_epoch_covers_every_cell_once(epochs[0])


def test_spawn_num_workers_2(fixture_paths: list[str]) -> None:
    """Phase 4.4: spawn-method variant. The child re-execs Python so no
    rayon / tokio / glibc state is inherited — this should pass
    independent of any Phase 2 fix and serves as the "the lifecycle is
    correct, fork-specific issues are all that's left" baseline.
    """
    epochs = _run_outer_process(
        fixture_paths,
        start_method="spawn",
        num_workers=2,
        persistent_workers=False,
        n_epochs=1,
    )
    assert len(epochs) == 1
    _assert_one_epoch_covers_every_cell_once(epochs[0])


# ---------------------------------------------------------------------------
# Phase 7.2 — IndexPlanDataset fork-safety analogue
# ---------------------------------------------------------------------------
#
# IndexPlanDataset shares the tokio multi-thread runtime + std::thread +
# crossbeam primitives with TrainingDataset (it owns its own runtime built
# lazily by the loader's `PrefetchEngine`); its prefetch goes via
# `tokio::spawn_blocking` and `std::thread::spawn`.
#
# It *does* reach rayon, though — through `scx-format-io`, not directly — which
# is why this test alone was not enough. `IndexPlanLoader::new` calls
# `ScxReader::read_obs`, which fans the shard decode out with `par_iter` on any
# file with sharded obs metadata; and the gather reaches `warm_shards`. Both
# used to hit rayon's *global* registry, whose worker threads do not survive
# `fork()`. The fixture below is 16 cells in a single shard with unsharded obs,
# so it reached neither: `misses.len() == 1` short-circuits `warm_shards` to a
# sequential loop, and unsharded obs is a plain section read. It passed
# vacuously. `test_fork_index_plan_sharded_obs` and
# `test_fork_index_plan_multi_shard_gather` below are the non-vacuous versions;
# both assert their own premise in the parent before forking.
#
# The tokio side is genuinely fine: its multi-thread runtime constructs cleanly
# in a forked child, which is why this test passed at all.


def _child_iterate_index_plan_dataset(scx_path: str, conn) -> None:
    """Construct IndexPlanDataset in a forked child, drive
    `iter_with_plans` over a small fixed plan list, push the observed
    pairs back over the pipe."""
    try:
        import pyscx

        ds = pyscx.IndexPlanDataset(
            scx_path,
            normalize=False,
            log1p=False,
            obs_columns=["global_cell_id"],
        )
        # 4 plans × 4 pairs each = 16 pairs total. Each pair is
        # (pert_idx, ctrl_idx) — both are global cell indices into the
        # single fixture file.
        plans = [
            [(0, 1), (2, 3), (4, 5), (6, 7)],
            [(8, 9), (10, 11), (12, 13), (14, 15)],
            [(0, 8), (1, 9), (2, 10), (3, 11)],
            [(15, 0), (14, 1), (13, 2), (12, 3)],
        ]
        seen_pairs: list[tuple[int, int]] = []
        for batch in ds.iter_with_plans(iter(plans)):
            seen_pairs.extend((int(p), int(c)) for p, c in batch["pairs"])
        conn.send(("ok", seen_pairs))
    except BaseException as exc:
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


def test_fork_index_plan_dataset(fixture_paths: list[str]) -> None:
    """Phase 7.2 acceptance: a forked child can construct
    `IndexPlanDataset` and iterate `iter_with_plans` to completion. Same
    test surface as Phase 1.2 (`test_forked_child_iterates_training_dataset`)
    but for the index-plan loader.

    Expected outcome: child returns the 16 (pert, ctrl) pairs we sent in
    (sort_by_shard may permute their order within a batch but not the
    set). Hangs are surfaced as `pytest.fail` after `TEST_DEADLINE_SEC`.
    """
    # Use the first fixture file (it has 16 cells, plenty for the plans
    # above which reference indices 0..15).
    scx_path = fixture_paths[0]

    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_child_iterate_index_plan_dataset,
        args=(scx_path, child_conn),
        daemon=False,
    )
    proc.start()
    child_conn.close()
    proc.join(timeout=TEST_DEADLINE_SEC)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2.0)
        pytest.fail(
            f"IndexPlanDataset child did not finish within {TEST_DEADLINE_SEC}s "
            "(fork-mode hang — IndexPlanDataset fork-safety regression)"
        )

    if not parent_conn.poll(0.0):
        pytest.fail(
            f"IndexPlanDataset child exited (rc={proc.exitcode}) without sending status"
        )
    status, payload = parent_conn.recv()
    parent_conn.close()
    if status == "err":
        pytest.fail(f"IndexPlanDataset child raised: {payload}")
    assert status == "ok"
    seen = payload
    expected = {
        (0, 1), (2, 3), (4, 5), (6, 7),
        (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 8), (1, 9), (2, 10), (3, 11),
        (15, 0), (14, 1), (13, 2), (12, 3),
    }
    assert set(seen) == expected, (
        f"missing={expected - set(seen)} extra={set(seen) - expected}"
    )


# ---------------------------------------------------------------------------
# The rayon-after-fork hazard, reached through scx-format-io
# ---------------------------------------------------------------------------
#
# `fork()` copies rayon's global registry as a data structure but not its worker
# threads, so a `par_*` dispatched from the child parks forever in
# `LockLatch::wait_and_reset` — no error, no batch. The parent gets that
# registry initialised by almost anything; `pyscx.from_anndata` is itself
# `par_iter`, so the fixtures below arm it as a side effect of existing.
#
# Each test asserts, in the parent, that its fixture actually reaches the code
# path in question. Without that a fixture drifts back to the vacuous case and
# nothing says so.

# Small enough to stay fast, large enough to shard four ways at SHARD_SIZE.
MULTI_N_OBS = 256
MULTI_N_VARS = 32
MULTI_SHARD_SIZE = 64


def _make_multi_shard_fixture(path: str, *, unframed: bool, sharded_obs: bool) -> None:
    """A 4-shard file. `unframed` writes legacy (v1) shards so the gather takes
    the full-shard path; `sharded_obs` leaves obs as `ObsMetadataShard`s."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    import pyscx

    rng = np.random.default_rng(7)
    dense = rng.integers(0, 10, size=(MULTI_N_OBS, MULTI_N_VARS), dtype=np.int32).astype(
        np.float32
    )
    dense[rng.random((MULTI_N_OBS, MULTI_N_VARS)) < 0.6] = 0
    obs = pd.DataFrame(
        {"global_cell_id": np.arange(MULTI_N_OBS, dtype=np.int64)},
        index=[f"cell_{i}" for i in range(MULTI_N_OBS)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(MULTI_N_VARS)])
    adata = ad.AnnData(X=sp.csr_matrix(dense), obs=obs, var=var)

    kwargs: dict[str, Any] = {"shard_size": MULTI_SHARD_SIZE}
    if unframed:
        kwargs["row_group_rows"] = 0
    if not sharded_obs:
        kwargs["force_legacy_metadata"] = True
    pyscx.from_anndata(adata, path, **kwargs)


@pytest.fixture(scope="module")
def sharded_obs_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Framed X, **sharded obs** — the `read_obs` trigger in isolation."""
    import pyscx

    p = str(tmp_path_factory.mktemp("fork_sharded_obs") / "f.scx")
    _make_multi_shard_fixture(p, unframed=False, sharded_obs=True)
    # Premise: without sharded obs metadata `read_obs` is a plain section read
    # and never touches rayon, so the test below would prove nothing.
    assert pyscx.open(p).obs_metadata_shard_count > 0, (
        "fixture has unsharded obs — read_obs would not reach par_iter"
    )
    return p


@pytest.fixture(scope="module")
def unframed_multi_shard_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Unframed X, **unsharded** obs — the `warm_shards` trigger in isolation
    (unsharded obs so the read_obs trigger cannot mask it)."""
    import pyscx

    p = str(tmp_path_factory.mktemp("fork_unframed") / "f.scx")
    _make_multi_shard_fixture(p, unframed=True, sharded_obs=False)
    exp = pyscx.open(p)
    assert exp.shard_count >= 3, f"want >= 3 CSR shards, got {exp.shard_count}"
    assert exp.obs_metadata_shard_count == 0, (
        "obs must be unsharded here so this test isolates the gather"
    )
    return p


# Rows in four different shards (0-63, 64-127, 128-191, 192-255), as pairs.
_SPANNING_PLAN = [(0, 70), (10, 140), (20, 200), (30, 80)]


def test_gather_premise_holds_in_parent(unframed_multi_shard_path: str) -> None:
    """Guard for the two tests below: prove the plan really does drive
    `warm_shards` down its **parallel** arm, in-process where we can read the
    counters. If this goes quiet the fork tests become vacuous and nothing else
    would say so — which is exactly how the 16-cell fixture above went blind.

    Three conditions, all necessary:
      * `lookahead=0`, or `IndexPlanIter` prefetches every touched shard via
        `spawn_blocking` first and `warm_shards` sees no misses at all;
      * `cache_shards >= 2`, or `warm_shards` short-circuits to a sequential
        loop;
      * shards that skip the block-index path, or they never enter
        `full_shards`. Hence the unframed fixture plus `scatter_block_index`.
    """
    import pyscx

    ds = pyscx.IndexPlanDataset(
        unframed_multi_shard_path,
        normalize=False,
        log1p=False,
        obs_columns=[],
        cache_shards=8,
        scatter_block_index=False,
    )
    assert ds.effective_cache_shards() >= 2, (
        "cache_shards collapsed to 1 — warm_shards would run sequentially"
    )
    for _ in ds.iter_with_plans(iter([_SPANNING_PLAN]), lookahead=0):
        pass
    m = ds.cache_metrics()
    assert m["full_shard_groups"] >= 2, (
        f"plan produced {m['full_shard_groups']} full-shard groups, need >= 2 for "
        f"warm_shards to dispatch in parallel (block_index_groups="
        f"{m['block_index_groups']})"
    )


def _child_construct_index_plan(scx_path: str, conn) -> None:
    """Construct only — that is the whole trigger for the read_obs path."""
    try:
        import pyscx

        ds = pyscx.IndexPlanDataset(
            scx_path, normalize=False, log1p=False, obs_columns=["global_cell_id"]
        )
        conn.send(("ok", int(ds.n_obs)))
    except BaseException as exc:  # noqa: BLE001
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


def _child_gather_across_shards(scx_path: str, conn) -> None:
    try:
        import warnings

        import pyscx

        # The unframed fixture warns that the block-index path cannot fire.
        # That is the point of the fixture, not a problem.
        warnings.simplefilter("ignore")
        ds = pyscx.IndexPlanDataset(
            scx_path,
            normalize=False,
            log1p=False,
            obs_columns=[],
            cache_shards=8,
            scatter_block_index=False,
        )
        seen: list[tuple[int, int]] = []
        for batch in ds.iter_with_plans(iter([_SPANNING_PLAN]), lookahead=0):
            seen.extend((int(p), int(c)) for p, c in batch["pairs"])
        conn.send(("ok", seen))
    except BaseException as exc:  # noqa: BLE001
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


def _child_collate_cellset(conn) -> None:
    """No dataset, no file — a bare pyfunction, so no PID check guards it."""
    try:
        import numpy as np

        import pyscx

        n_rows, k_enc, n_genes = 512, 16, 64
        indptr = np.arange(0, n_rows * k_enc + 1, k_enc, dtype=np.int64)
        indices = np.tile(np.arange(k_enc, dtype=np.int32), n_rows)
        data = np.full(n_rows * k_enc, 3.0, dtype=np.float32)
        out = pyscx.collate_cellset_gathered(
            indptr,
            indices,
            data,
            np.array([0, n_rows], dtype=np.int64),
            np.arange(n_rows, dtype=np.uint64),
            np.zeros(n_rows, dtype=np.uint32),
            np.zeros(n_rows, dtype=np.int32),
            n_genes,
            np.arange(n_genes, dtype=np.int32),
            np.array([], dtype=np.uint8),
            np.zeros(n_rows, dtype=np.uint8),
            np.array([n_genes], dtype=np.uint32),
            k_enc,
            "pass_through",
            n_genes,
        )
        conn.send(("ok", int(out["n_rows"])))
    except BaseException as exc:  # noqa: BLE001
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


def _child_downsample_counts(conn) -> None:
    try:
        import numpy as np

        import pyscx

        n_rows, k = 512, 16
        indptr = np.arange(0, n_rows * k + 1, k, dtype=np.int64)
        indices = np.tile(np.arange(k, dtype=np.int32), n_rows)
        data = np.full(n_rows * k, 20.0, dtype=np.float32)
        out = pyscx.downsample_counts_csr(
            indptr,
            indices,
            data,
            np.arange(n_rows, dtype=np.uint64),
            np.array([], dtype=np.uint64),
            target_library_size=10,
            seed=0,
        )
        conn.send(("ok", int(len(out["indptr"]))))
    except BaseException as exc:  # noqa: BLE001
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


def _run_forked_child(target, args: tuple, label: str):
    """Fork `target`, enforce the deadline, and return its payload.

    The parent has already run `pyscx.from_anndata` (or another rayon user) by
    the time this is called, so the global registry the child inherits is armed
    and dead — which is the condition under test.
    """
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=target, args=(*args, child_conn), daemon=False)
    proc.start()
    child_conn.close()
    proc.join(timeout=TEST_DEADLINE_SEC)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2.0)
        pytest.fail(
            f"{label}: child did not finish within {TEST_DEADLINE_SEC}s — this is "
            "the rayon-after-fork hang, not a slow test"
        )
    if not parent_conn.poll(0.0):
        pytest.fail(f"{label}: child exited (rc={proc.exitcode}) without sending status")
    status, payload = parent_conn.recv()
    parent_conn.close()
    if status == "err":
        pytest.fail(f"{label}: child raised: {payload}")
    return payload


def test_fork_index_plan_sharded_obs(sharded_obs_path: str) -> None:
    """`IndexPlanLoader::new` -> `ScxReader::read_obs` -> the sharded-metadata
    `par_iter`. Construction alone is the trigger, so this needs no plan and no
    gather — and it fires on every atlas-scale file, since obs shards whenever
    `n_obs > shard_target_rows`.
    """
    n_obs = _run_forked_child(
        _child_construct_index_plan, (sharded_obs_path,), "sharded-obs construct"
    )
    assert n_obs == MULTI_N_OBS


def test_fork_index_plan_multi_shard_gather(unframed_multi_shard_path: str) -> None:
    """`read_rows_with` -> `warm_shards` -> `par_iter`, under the conditions
    `test_gather_premise_holds_in_parent` pins."""
    seen = _run_forked_child(
        _child_gather_across_shards, (unframed_multi_shard_path,), "multi-shard gather"
    )
    assert set(map(tuple, seen)) == set(_SPANNING_PLAN)


def test_fork_collate_cellset_gathered(fixture_paths: list[str]) -> None:
    """`pyscx.collate_cellset_gathered` — reachable from a forked worker with no
    dataset in hand, so none of the PID checks in `python/` apply.

    Depends on `fixture_paths` only to guarantee the parent has run
    `from_anndata` and armed the global registry first; without that the child
    would initialise a fresh registry of its own and pass regardless.
    """
    assert fixture_paths
    n_rows = _run_forked_child(_child_collate_cellset, (), "collate_cellset_gathered")
    assert n_rows == 512


def test_fork_downsample_counts_csr(fixture_paths: list[str]) -> None:
    """`pyscx.downsample_counts_csr` — same shape as the collate kernel above."""
    assert fixture_paths
    n = _run_forked_child(_child_downsample_counts, (), "downsample_counts_csr")
    assert n == 513  # n_rows + 1


def _child_tokenize_kernels(conn) -> None:
    """Every `pyscx.tokenize` kernel, in a forked child with no dataset.

    Same hazard as the collate kernel: these are bare pyfunctions, so no PID
    check in `python/` guards them, and each dispatches rayon inside
    `py.detach`. A global-pool dispatch from a forked child hangs forever rather
    than failing, so the parent's timeout is the real assertion.
    """
    try:
        import numpy as np

        import pyscx
        import pyscx.tokenize as tok

        n_rows, k, n_genes = 256, 24, 64
        indptr = np.arange(0, n_rows * k + 1, k, dtype=np.int64)
        indices = np.tile(np.arange(k, dtype=np.int32), n_rows)
        # Distinct values per row, so the bin kernel takes the interpolation
        # branch rather than the collapsed-edge one.
        data = np.tile(np.arange(1, k + 1, dtype=np.float32), n_rows)
        stats = np.linspace(0.5, 1.5, n_genes).astype(np.float32)

        got = {
            "top_k": int(tok.top_k(indptr, indices, data, 8, n_genes)["n_rows"]),
            "rank": int(tok.rank_tokens(indptr, indices, data, stats, 8, "v")["n_rows"]),
            "bin": int(tok.bin_values(indptr, indices, data, 5)["n_rows"]),
            "sample": int(
                tok.sample_genes(indptr, indices, data, 4, 1, 2)["n_rows"]
            ),
            "transform": int(
                tok.transform_values(indptr, data, "log1p_raw").shape[0]
            ),
            "library_size": int(tok.library_size(indptr, data).shape[0]),
        }
        conn.send(("ok", got))
    except BaseException as exc:  # noqa: BLE001
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


def test_fork_tokenize_kernels(fixture_paths: list[str]) -> None:
    """The W6 kernels — reachable from a forked DataLoader worker with no
    dataset in hand, so none of the PID checks in `python/` apply.

    Depends on `fixture_paths` for the same reason the collate test does: the
    parent must have armed the global rayon registry first, or the child
    initialises a fresh one of its own and passes regardless.
    """
    assert fixture_paths
    got = _run_forked_child(_child_tokenize_kernels, (), "pyscx.tokenize kernels")
    assert got == {
        "top_k": 256,
        "rank": 256,
        "bin": 256,
        "sample": 256,
        "transform": 256 * 24,
        "library_size": 256,
    }


def _child_iterate_training_dataset(scx_path: str, conn) -> None:
    try:
        import pyscx

        ds = pyscx.TrainingDataset(
            scx_path,
            batch_size=32,
            normalize=False,
            log1p=False,
            obs_columns=["global_cell_id"],
        )
        try:
            n = sum(len(b["cell_indices"]) for b in ds)
        finally:
            ds.close()
        conn.send(("ok", n))
    except BaseException as exc:  # noqa: BLE001
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


def test_fork_training_dataset_sharded_obs(sharded_obs_path: str) -> None:
    """`TrainingDataset` reaches the same `read_obs` par_iter as
    `IndexPlanDataset` — `TrainingPipeline::new` calls it too.

    The per-pipeline decode pool that Phase 2.0 added does not cover this: it is
    built in `start_epoch`, long after the constructor has already dispatched.
    The tests above this one only ever exercised 16-cell files whose obs is a
    single section, so the class every fork-safety guarantee is written about
    was hanging on any file large enough to shard its obs — which is every file
    the loader exists for.
    """
    n = _run_forked_child(
        _child_iterate_training_dataset, (sharded_obs_path,), "TrainingDataset sharded-obs"
    )
    assert n == MULTI_N_OBS


def _child_iterate_sparse_cellset(scx_path: str, conn) -> None:
    try:
        import warnings

        import pyscx

        warnings.simplefilter("ignore")
        ds = pyscx.SparseCellSetDataset([scx_path], cache_shards=8)
        # (file_ids, rows, role_tags, set_offsets) — one set spanning four shards.
        rows = [0, 70, 140, 200]
        plan = ([0] * len(rows), rows, [0] * len(rows), [0, len(rows)])
        n = 0
        for batch in ds.iter_with_plans(iter([plan]), lookahead=0):
            n += len(batch["cell_indices"])
        conn.send(("ok", n))
    except BaseException as exc:  # noqa: BLE001
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


def test_fork_sparse_cellset_dataset(unframed_multi_shard_path: str) -> None:
    """The other class §9.1 names. Its readers come from
    `PrefetchEngine::from_scx_readers`, a construction path nothing else here
    exercises under fork, and its gather reaches the same `warm_shards`.
    """
    n = _run_forked_child(
        _child_iterate_sparse_cellset,
        (unframed_multi_shard_path,),
        "SparseCellSetDataset gather",
    )
    assert n == 4


def _child_construct_index_plan_pflog(scx_path: str, conn) -> None:
    """`pflog=True` routes construction through `scx_accel::estimate_alpha`,
    which walks shards on `scx_format_io::prefetch`'s `rayon::in_place_scope` —
    a third rayon entry point, distinct from `read_obs` and `warm_shards`."""
    try:
        import warnings

        import pyscx

        warnings.simplefilter("ignore")
        ds = pyscx.IndexPlanDataset(
            scx_path, normalize=False, log1p=False, obs_columns=[], pflog=True
        )
        conn.send(("ok", int(ds.n_obs)))
    except BaseException as exc:  # noqa: BLE001
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


def test_fork_index_plan_pflog_alpha_estimate(unframed_multi_shard_path: str) -> None:
    """The α-estimate path under fork.

    `prefetch`'s guard — `current_num_threads() > 1 && current_thread_index()
    .is_none()` — asks "am I already on a rayon worker?", and **cannot see a
    fork**: after one, both halves still read as they did in the parent. So the
    guard passes and `in_place_scope` spawns onto threads that no longer exist.
    Uses the multi-shard fixture because a single-shard file would take the
    sequential arm and prove nothing.
    """
    n = _run_forked_child(
        _child_construct_index_plan_pflog,
        (unframed_multi_shard_path,),
        "IndexPlanDataset pflog alpha",
    )
    assert n == MULTI_N_OBS
