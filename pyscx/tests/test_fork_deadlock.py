"""Pure-multiprocessing fork reproducer for the
``pyscx.TrainingDataset`` deadlock.

The hypothesis under investigation here is: does a forked child constructing
and iterating a fresh ``TrainingDataset`` deadlock under bare
``multiprocessing.Process(target=..., daemon=False)`` with the default fork
start method on Linux — independent of PyTorch's ``DataLoader`` queue /
sentinel machinery? If yes, the root cause is at the pyscx / tokio /
PyO3 layer. If no, the deadlock is only triggered under PyTorch's DataLoader.

The companion Rust integration test ``test_fork_deadlock.rs``
already shows that bare ``nix::unistd::fork`` + ``TrainingPipeline``
iteration in the child does *not* deadlock. This file extends the same
question one layer up: does adding the Python interpreter, PyO3 binding,
and ``multiprocessing`` fork machinery introduce the hang?

The test is gated on Linux because fork start-method semantics are POSIX-
specific.

Run with:

    .venv/bin/pytest pyscx/tests/test_fork_deadlock.py -v
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import pytest


# Skip the module on non-Linux — fork start method is unavailable elsewhere.
pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="fork start method is Linux-only; spawn is the macOS/Windows default",
)


# Hard deadline for the child. If the deadlock reproduces, the child will
# hang in `TrainingDataset.__iter__` / `__next__`; we want to surface that
# as a test failure rather than a CI hang.
CHILD_TIMEOUT_SEC = 30.0


def _child_iterate_one_epoch(scx_path: str, conn) -> None:
    """Worker target for the forked child.

    Imports inside the child so the parent never imports pyscx (and so
    never eagerly initializes any tokio / Rust state pre-fork). Pushes the
    batch count back over a ``Connection`` so the parent can assert on it
    without leaning on stdout interleaving.
    """
    try:
        import pyscx  # noqa: WPS433 (import in child by design)

        ds = pyscx.TrainingDataset(
            scx_path,
            batch_size=32,
            normalize=False,
            log1p=False,
        )
        n_batches = 0
        for _batch in ds:
            n_batches += 1
        conn.send(("ok", n_batches))
    except BaseException as exc:  # noqa: BLE001 (need to surface anything)
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


def _run_in_forked_child(scx_path: str) -> tuple[str, object]:
    """Spawn a forked child via ``multiprocessing.Process``, run one epoch,
    and return ``(status, payload)``.

    Uses an explicit ``"fork"`` context so we don't depend on the test
    runner's global default (which a previous test may have changed). Uses
    a Pipe rather than a Queue to keep the IPC primitive minimal — Queue
    spawns its own feeder thread that itself can hang if the worker is
    stuck.
    """
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_child_iterate_one_epoch,
        args=(scx_path, child_conn),
        daemon=False,
    )
    t0 = time.monotonic()
    proc.start()
    # Close the parent's copy of the write end so we observe EOF if the
    # child dies without sending. Order matters: must happen *after* start.
    child_conn.close()
    proc.join(timeout=CHILD_TIMEOUT_SEC)
    elapsed = time.monotonic() - t0

    if proc.is_alive():
        # Hard timeout: deadlock candidate. SIGKILL the worker so the test
        # framework gets its process slot back.
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2.0)
        pytest.fail(
            "forked child did not finish iterating TrainingDataset within "
            f"{CHILD_TIMEOUT_SEC}s (deadlock reproduces under bare "
            "multiprocessing.fork)"
        )

    # Read the child's status. If the child crashed before sending, we'll
    # get EOFError here — surface as a distinct failure.
    if not parent_conn.poll(0.0):
        pytest.fail(
            f"forked child exited (rc={proc.exitcode}) without sending a "
            f"status (elapsed {elapsed:.2f}s)"
        )
    status, payload = parent_conn.recv()
    parent_conn.close()
    return status, payload


@pytest.fixture
def scx_path(tmp_path: Path, synthetic_adata) -> str:
    """Write the synthetic AnnData fixture to a `.scx` file in the parent.

    Crucially, the parent does *not* construct a ``TrainingDataset`` — only
    writes the fixture. Forks happen with the parent holding only the
    Python interpreter + (transitively) any state ``pyscx.from_anndata``
    leaves behind. This isolates "fork-then-construct" from
    "construct-then-fork".
    """
    import pyscx

    p = str(tmp_path / "fork_fixture.scx")
    pyscx.from_anndata(synthetic_adata, p)
    return p


def test_forked_child_iterates_training_dataset(scx_path: str) -> None:
    """Phase 1.2 acceptance: a child forked from a parent that has *not*
    constructed any TrainingDataset can construct + iterate one cleanly.

    Expected outcome on a green build: child returns ``("ok", n_batches)``
    where ``n_batches`` is the count of mini-batches for one epoch over
    the 100-cell fixture (~`ceil(100/32) == 4`). Expected outcome if the
    deadlock reproduces at this layer: ``_run_in_forked_child`` raises
    via ``pytest.fail`` after ``CHILD_TIMEOUT_SEC``.
    """
    status, payload = _run_in_forked_child(scx_path)
    if status == "err":
        pytest.fail(f"forked child raised: {payload}")
    assert status == "ok", f"unexpected status from child: {status!r}"
    n_batches = payload
    assert isinstance(n_batches, int)
    # 100 cells / batch_size=32 → 4 batches (last one of size 4).
    assert n_batches >= 1, f"child produced {n_batches} batches; expected >= 1"


def test_forked_child_two_epochs(scx_path: str) -> None:
    """Same as above but exercises a second epoch in the child to cover
    ``start_epoch`` / ``join_epoch_handles`` re-entry under fork.

    The Drop / shutdown race is distinct from runtime construction: a
    steady-state epoch transition stresses different lifecycle paths than
    the first construction does.
    """

    def child(scx_path: str, conn) -> None:
        try:
            import pyscx

            ds = pyscx.TrainingDataset(
                scx_path,
                batch_size=32,
                normalize=False,
                log1p=False,
            )
            counts = []
            for _epoch in range(2):
                n = 0
                for _batch in ds:
                    n += 1
                counts.append(n)
            conn.send(("ok", counts))
        except BaseException as exc:
            conn.send(("err", repr(exc)))
        finally:
            conn.close()

    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=child, args=(scx_path, child_conn), daemon=False)
    proc.start()
    child_conn.close()
    proc.join(timeout=CHILD_TIMEOUT_SEC)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2.0)
        pytest.fail(
            f"forked child hung on the 2-epoch path within {CHILD_TIMEOUT_SEC}s"
        )

    if not parent_conn.poll(0.0):
        pytest.fail(f"forked child exited (rc={proc.exitcode}) without sending status")
    status, payload = parent_conn.recv()
    parent_conn.close()
    if status == "err":
        pytest.fail(f"forked child raised: {payload}")
    assert status == "ok"
    assert len(payload) == 2
    assert all(n >= 1 for n in payload), f"epoch counts: {payload}"
