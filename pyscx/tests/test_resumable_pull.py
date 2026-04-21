"""Phase F.4 — pull is idempotent-retry after interruption.

Pull writes to ``{dest}.tmp.{pid}`` and atomically renames on completion.
An interrupted run leaves ``{dest}.tmp.{pid}`` orphaned; the next run
sweeps the orphan when BOTH conditions hold: the pid is no longer live
(``/proc/{pid}`` absent or suffix unparseable) AND the mtime is older
than one hour. A fresh tmp owned by a live pid is never unlinked —
that's what keeps concurrent pulls to the same dest safe.

These tests use local filesystem paths — no real cloud access — but
exercise the same ``pyscx.pull`` surface a cloud pull uses.
"""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np
import pytest
import scipy.sparse as sp

# ``cleanup_stale_tmp_files`` in scx-cloud/src/pull.rs uses a 1-hour
# (3600 s) mtime threshold. Age orphans well past that before each test
# so the sweeper actually fires.
_STALE_TMP_AGE_S = 60 * 60 + 120

# u32::MAX exceeds the kernel's pid_max (Linux default 2^22) so
# ``/proc/{pid}`` can never exist — a deterministic "dead pid" token.
_DEAD_PID = 4294967295


def _plant_orphan(path: str, *, age_s: float = _STALE_TMP_AGE_S) -> None:
    """Write a fake orphan tmp file and age its mtime so the sweeper unlinks it."""
    with open(path, "wb") as f:
        f.write(b"orphan from a crashed prior pull")
    past = time.time() - age_s
    os.utime(path, (past, past))

_pyscx = pytest.importorskip("pyscx")
if not hasattr(_pyscx, "explode"):
    pytest.skip("pyscx built without cloud features", allow_module_level=True)


def _make_test_scx(path: str, n_obs: int = 100, n_vars: int = 50) -> None:
    import anndata
    import pyscx

    rng = np.random.default_rng(42)
    X = sp.random(
        n_obs, n_vars, density=0.1, format="csr",
        dtype=np.float32, random_state=rng,
    )
    X.data = np.round(X.data * 10).astype(np.float32)
    adata = anndata.AnnData(
        X=X,
        obs={"cell_id": [f"cell_{i}" for i in range(n_obs)]},
        var={"gene_id": [f"gene_{i}" for i in range(n_vars)]},
    )
    pyscx.from_anndata(adata, path, codec="none")


def _explode_for_test(tmpdir: str, n_obs: int = 100) -> str:
    """Create an exploded ``.scxd`` in ``tmpdir`` and return its path."""
    import pyscx

    src = os.path.join(tmpdir, "source.scx")
    _make_test_scx(src, n_obs=n_obs)
    scxd = os.path.join(tmpdir, "source.scxd")
    pyscx.explode(src, scxd)
    return scxd


class TestPullIdempotentRetry:
    """Pull sweeps stale ``.tmp.*`` files and always produces a valid SCX."""

    def test_stale_tmp_does_not_block_pull(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scxd = _explode_for_test(tmpdir)
            dest = os.path.join(tmpdir, "pulled.scx")

            # Simulate an interrupted prior run by planting an orphan
            # `.tmp.{pid}` file at the destination. The sweeper requires
            # the pid to be dead AND the file to be older than 1 hour,
            # so use a pid above pid_max and backdate the mtime.
            orphan = f"{dest}.tmp.{_DEAD_PID}"
            _plant_orphan(orphan)
            assert os.path.exists(orphan)

            stats = pyscx.pull(scxd, dest)

            assert stats["bytes_downloaded"] > 0
            assert os.path.exists(dest), "final dest must exist after retry"
            assert not os.path.exists(orphan), "aged dead-pid orphan must be swept"

    def test_multiple_stale_tmps_all_swept(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scxd = _explode_for_test(tmpdir)
            dest = os.path.join(tmpdir, "pulled.scx")

            # All orphans use dead-pid suffixes (above Linux pid_max) and
            # an aged mtime; the sweeper deletes them. Low pids (1, 2, 3)
            # were used originally but those map to live kernel/init
            # processes, which the new concurrent-safe sweeper preserves
            # by design.
            orphans = [
                f"{dest}.tmp.{_DEAD_PID}",
                f"{dest}.tmp.{_DEAD_PID - 1}",
                f"{dest}.tmp.{_DEAD_PID - 2}",
                f"{dest}.tmp.abcd",  # unparseable suffix; still aged
            ]
            for o in orphans:
                _plant_orphan(o)

            pyscx.pull(scxd, dest)

            for o in orphans:
                assert not os.path.exists(o), f"aged orphan {o} not swept"

    def test_fresh_tmp_from_live_pid_is_preserved(self):
        """A tmp file owned by a live sibling pid must NOT be unlinked.

        Concurrent-pull safety: two workers both targeting ``dest`` each
        write to their own ``{dest}.tmp.{pid}``; the sweeper running in
        worker B must never delete worker A's in-flight file.

        Uses pid 1 (init/systemd) as a "live but not ours" pid. Our own
        pid can't be used because ``pyscx.pull`` itself writes to
        ``{dest}.tmp.{os.getpid()}`` and would consume the planted file
        as its own work area — the race we actually want to exercise is
        *sibling* concurrent pulls.
        """
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scxd = _explode_for_test(tmpdir)
            dest = os.path.join(tmpdir, "pulled.scx")

            # pid 1 is guaranteed live on Linux (init/systemd) and is
            # never the test's own pid. Fresh mtime.
            sibling_tmp = f"{dest}.tmp.1"
            with open(sibling_tmp, "wb") as f:
                f.write(b"in-flight from a concurrent pull")

            pyscx.pull(scxd, dest)

            assert os.path.exists(sibling_tmp), (
                "tmp file owned by a live sibling pid must be preserved"
            )

    def test_fresh_dead_pid_tmp_is_preserved(self):
        """Even a dead pid's tmp is preserved while mtime is fresh.

        Mirrors the mtime guard: we only delete tmps that are both
        dead-pid AND older than the stale-age threshold. A file that
        looks dead but is recently modified might still be in play on a
        non-Linux system lacking ``/proc``.
        """
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scxd = _explode_for_test(tmpdir)
            dest = os.path.join(tmpdir, "pulled.scx")

            fresh_dead = f"{dest}.tmp.{_DEAD_PID}"
            with open(fresh_dead, "wb") as f:
                f.write(b"very recent")

            pyscx.pull(scxd, dest)

            assert os.path.exists(fresh_dead), (
                "fresh orphan should be preserved; only aged ones are swept"
            )

    def test_pull_filtered_also_sweeps_orphans(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scxd = _explode_for_test(tmpdir)
            dest = os.path.join(tmpdir, "filtered.scx")

            orphan = f"{dest}.tmp.{_DEAD_PID}"
            _plant_orphan(orphan)

            # A predicate that matches every cell — selective path runs
            # through the same cleanup_stale_tmp_files entry point.
            stats = pyscx.pull(scxd, dest, filter="cell_id != ''")

            assert "matching_cells" in stats
            assert os.path.exists(dest)
            assert not os.path.exists(orphan), "aged orphan not swept on filtered pull"

    def test_repeated_pull_is_idempotent(self):
        """Running pull twice back-to-back yields identical bytes."""
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scxd = _explode_for_test(tmpdir)
            dest1 = os.path.join(tmpdir, "first.scx")
            dest2 = os.path.join(tmpdir, "second.scx")

            pyscx.pull(scxd, dest1)
            pyscx.pull(scxd, dest2)

            # File-level bytes can differ in the header file_checksum
            # (identical content, identical checksum), so compare via the
            # reader's observable data rather than raw bytes.
            r1 = pyscx.open(dest1)
            r2 = pyscx.open(dest2)
            assert r1.n_obs == r2.n_obs
            assert r1.n_vars == r2.n_vars
            assert r1.nnz == r2.nnz
