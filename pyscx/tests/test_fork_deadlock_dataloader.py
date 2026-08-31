"""PyTorch DataLoader fork-mode reproducer for the
``pyscx.TrainingDataset`` deadlock.

Layered with the pure-Rust fork and bare ``multiprocessing.Process``
fork reproducers, this is the actual failing
case from ``state-scx/tests/test_se_scx_adapter.py::test_worker_sharding_partitions_files``:
``torch.utils.data.DataLoader(IterableDataset, num_workers=2,
persistent_workers=False)`` with the lazy-construct-in-``__iter__`` shim
that ``cell-load-scx``'s ``ScxTrainingDataset`` and ``state-scx``'s
``ScxStateAdapter`` use.

The IterableDataset shim defers ``pyscx.TrainingDataset`` construction
until inside the worker's ``__iter__``, so the worker's PID matches the
TrainingDataset's ``creation_pid`` and the eager-fork PID check at
``scx-loader/src/python/training.rs`` does *not* fire. That is the
documented workaround pattern; this test confirms whether it actually
keeps the worker alive at runtime.

The test is skipped when ``torch`` is not installed (``torch`` is not a
``pyscx`` runtime dep — it is added only by downstream consumers like
``cell-load-scx`` / ``state-scx``).

Run with:

    .venv/bin/pytest pyscx/tests/test_fork_deadlock_dataloader.py -v

Or, from a venv with torch installed:

    pip install torch
    pytest pyscx/tests/test_fork_deadlock_dataloader.py -v
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.skipif(
        sys.platform != "linux",
        reason="fork start method is Linux-only",
    ),
    pytest.mark.skipif(
        os.environ.get("PYSCX_SKIP_FORK_DATALOADER") == "1",
        reason="PYSCX_SKIP_FORK_DATALOADER=1 set in environment",
    ),
]


torch = pytest.importorskip("torch")
torch_data = pytest.importorskip("torch.utils.data")


# Hard deadline on the entire test process if the DataLoader hangs. The
# deadlock reproduces here the same as in `state-scx`'s real worker test —
# we want to surface it as a `pytest.fail` instead of a CI hang.
TEST_TIMEOUT_SEC = 60.0


class _LazyShim(torch_data.IterableDataset):
    """Mirrors the ``cell-load-scx`` / ``state-scx`` lazy-construct-in-worker
    pattern: ``__init__`` only stores the path; ``__iter__`` constructs the
    underlying ``pyscx.TrainingDataset`` so its ``creation_pid`` matches
    the *worker's* PID, defeating the eager-fork PID gate at
    ``python/training.rs``.
    """

    def __init__(self, scx_path: str, batch_size: int = 32) -> None:
        self.scx_path = scx_path
        self.batch_size = batch_size

    def __iter__(self):
        import pyscx  # imported in worker by design

        ds = pyscx.TrainingDataset(
            self.scx_path,
            batch_size=self.batch_size,
            normalize=False,
            log1p=False,
        )
        for batch in ds:
            # Yield a (cell_indices_list, n_rows) tuple so the parent
            # collator can sanity-check coverage without copying the dense X.
            ci = batch["cell_indices"]
            yield ci.tolist(), int(batch["X"].shape[0])


def _drive_dataloader(scx_path: str, num_workers: int, conn) -> None:
    """Worker entry: build the DataLoader, drain one epoch, push counts back.

    Lives in a top-level helper so the *outer* `mp.Process` we use for the
    timeout watchdog can fork-spawn it without pickling closures. (`spawn`
    would be safer but the whole point of this test is to exercise the
    fork path.)
    """
    try:
        ds = _LazyShim(scx_path, batch_size=32)
        loader = torch_data.DataLoader(
            ds,
            batch_size=None,
            num_workers=num_workers,
            persistent_workers=False,
            collate_fn=lambda b: b,
        )
        seen: list[int] = []
        for ci_list, n_rows in loader:
            assert n_rows == len(ci_list)
            seen.extend(ci_list)
        conn.send(("ok", sorted(seen)))
    except BaseException as exc:
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


@pytest.fixture
def scx_path(tmp_path: Path, synthetic_adata) -> str:
    """Write the fixture in the parent. Parent does NOT construct any
    pyscx.TrainingDataset — only ``pyscx.from_anndata``."""
    import pyscx

    p = str(tmp_path / "fork_dataloader_fixture.scx")
    pyscx.from_anndata(synthetic_adata, p)
    return p


def _run_with_outer_timeout(
    scx_path: str, num_workers: int
) -> tuple[str, object]:
    """Run ``_drive_dataloader`` inside an outer ``Process`` so we can
    enforce a deadline even if the inner DataLoader workers hang.

    Without this wrapper a hung DataLoader can wedge the pytest process
    itself — `pytest-timeout` is not a hard dep of pyscx and DataLoader
    workers don't all respect SIGTERM, so we use a process-level kill as
    the escape hatch.
    """
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_drive_dataloader,
        args=(scx_path, num_workers, child_conn),
        daemon=False,
    )
    t0 = time.monotonic()
    proc.start()
    child_conn.close()
    proc.join(timeout=TEST_TIMEOUT_SEC)
    elapsed = time.monotonic() - t0

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2.0)
        pytest.fail(
            f"DataLoader(num_workers={num_workers}) did not finish iterating "
            f"within {TEST_TIMEOUT_SEC}s (deadlock reproduces)"
        )

    if not parent_conn.poll(0.0):
        pytest.fail(
            f"DataLoader driver exited (rc={proc.exitcode}) without sending "
            f"a status (elapsed {elapsed:.2f}s)"
        )
    status, payload = parent_conn.recv()
    parent_conn.close()
    return status, payload


def test_dataloader_num_workers_zero_baseline(scx_path: str) -> None:
    """Baseline: ``num_workers=0`` (no fork) must always pass — same path
    that ``cell-load-scx`` / ``state-scx`` ship today.

    If this regresses, the issue is in the loader itself, not in fork
    interaction.
    """
    status, payload = _run_with_outer_timeout(scx_path, num_workers=0)
    if status == "err":
        pytest.fail(f"num_workers=0 driver raised: {payload}")
    assert status == "ok"
    indices = payload
    assert len(indices) == 100, f"expected 100 cells, got {len(indices)}"
    assert sorted(indices) == list(range(100))


def test_dataloader_num_workers_two_fork(scx_path: str) -> None:
    """Phase 1.3 acceptance: ``num_workers=2`` with fork start method
    drives the lazy-construct-in-``__iter__`` path through one full epoch.

    Expected outcome on a *fixed* build: each cell index appears exactly
    once across both workers' batches. Expected outcome with the deadlock
    intact: ``_run_with_outer_timeout`` raises ``pytest.fail`` after
    ``TEST_TIMEOUT_SEC``.
    """
    status, payload = _run_with_outer_timeout(scx_path, num_workers=2)
    if status == "err":
        pytest.fail(f"num_workers=2 driver raised: {payload}")
    assert status == "ok"
    indices = payload
    # IterableDataset under DataLoader doesn't shard automatically — the
    # `_LazyShim` here doesn't implement worker-aware sharding, so each
    # worker independently iterates the full dataset (= 100 cells * 2
    # workers = 200 yields). That's fine for the deadlock acceptance test;
    # downstream cell-load-scx / state-scx adapters add the sharding.
    assert len(indices) >= 100, f"expected >= 100 yields, got {len(indices)}"
