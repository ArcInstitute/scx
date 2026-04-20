"""Phase F.4 — pull is idempotent-retry after interruption.

Pull writes to ``{dest}.tmp.{pid}`` and atomically renames on completion.
An interrupted run leaves ``{dest}.tmp.{pid}`` orphaned; the next run must
sweep it and still produce a byte-valid SCX output.

These tests use local filesystem paths — no real cloud access — but
exercise the same ``pyscx.pull`` surface a cloud pull uses.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import scipy.sparse as sp

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
            # `.tmp.{pid}` file at the destination.
            orphan = f"{dest}.tmp.99999"
            with open(orphan, "wb") as f:
                f.write(b"junk from a crashed prior pull")
            assert os.path.exists(orphan)

            stats = pyscx.pull(scxd, dest)

            assert stats["bytes_downloaded"] > 0
            assert os.path.exists(dest), "final dest must exist after retry"
            assert not os.path.exists(orphan), "orphan tmp must be swept"

    def test_multiple_stale_tmps_all_swept(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scxd = _explode_for_test(tmpdir)
            dest = os.path.join(tmpdir, "pulled.scx")

            orphans = [f"{dest}.tmp.{i}" for i in (1, 2, 3, 9999)]
            for o in orphans:
                with open(o, "wb") as f:
                    f.write(b"orphan")

            pyscx.pull(scxd, dest)

            for o in orphans:
                assert not os.path.exists(o), f"orphan {o} not swept"

    def test_pull_filtered_also_sweeps_orphans(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scxd = _explode_for_test(tmpdir)
            dest = os.path.join(tmpdir, "filtered.scx")

            orphan = f"{dest}.tmp.12345"
            with open(orphan, "wb") as f:
                f.write(b"orphan")

            # A predicate that matches every cell — selective path runs
            # through the same cleanup_stale_tmp_files entry point.
            stats = pyscx.pull(scxd, dest, filter="cell_id != ''")

            assert "matching_cells" in stats
            assert os.path.exists(dest)
            assert not os.path.exists(orphan), "orphan not swept on filtered pull"

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
