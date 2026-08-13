"""Tests for TrainingDataset (scx-loader Python bindings)."""


import numpy as np
import pyscx
import pytest


@pytest.fixture
def scx_path(tmp_path, synthetic_adata):
    """Write a synthetic AnnData to SCX and return the path."""
    path = str(tmp_path / "test.scx")
    pyscx.from_anndata(synthetic_adata, path)
    return path


class TestTrainingDatasetConstruction:
    """F1: TrainingDataset can be constructed from Python."""

    def test_basic_construction(self, scx_path):
        ds = pyscx.TrainingDataset(scx_path)
        assert ds.n_obs == 100
        assert ds.n_vars == 50
        assert ds.n_output_genes == 50

    def test_with_hvg_projection(self, scx_path):
        hvg = [0, 5, 10, 20, 40]
        ds = pyscx.TrainingDataset(scx_path, hvg_indices=hvg)
        assert ds.n_output_genes == 5

    def test_hvg_out_of_range_rejected_at_construction(self, scx_path):
        """Mirror of the IndexPlanDataset test of the same name.

        An index >= n_vars matches no column, so before this was checked the
        batch simply carried an output feature that was always exactly zero —
        a dead input that trains to a zero weight and is never diagnosed.
        """
        with pytest.raises(RuntimeError, match="out of range"):
            pyscx.TrainingDataset(
                scx_path,
                hvg_indices=np.array([0, 1, 99999], dtype=np.uint32),
            )

    def test_hvg_index_equal_to_n_vars_rejected(self, scx_path):
        """The boundary: on a 50-gene file, index 50 is already out of range.

        This is the realistic way to trip it — a panel built against a file
        with one more gene, not a wild 99999.
        """
        with pytest.raises(RuntimeError, match="out of range"):
            pyscx.TrainingDataset(
                scx_path, hvg_indices=np.array([50], dtype=np.uint32)
            )

    def test_hvg_last_valid_index_accepted(self, scx_path):
        """The other half of the boundary — rejecting gene 49 would be a
        different bug that neither test above would catch."""
        ds = pyscx.TrainingDataset(
            scx_path, hvg_indices=np.array([0, 49], dtype=np.uint32)
        )
        assert ds.n_output_genes == 2

    def test_hvg_panel_in_rank_order_warns_about_column_order(self, scx_path):
        """`np.argsort(-variances)[:k]` is a natural way to build a panel, and
        it is in rank order — but the projection sorts, so the columns come
        back in gene-index order and the caller's own column->gene mapping is
        silently wrong. Nothing said so before."""
        with pytest.warns(UserWarning, match="ascending gene-index order"):
            ds = pyscx.TrainingDataset(
                scx_path, hvg_indices=np.array([40, 5, 20], dtype=np.uint32)
            )
        # Reordering alone drops nothing.
        assert ds.n_output_genes == 3

    def test_hvg_panel_with_duplicates_warns_about_the_width(self, scx_path):
        with pytest.warns(UserWarning, match="deduplicated from 5 to 3"):
            ds = pyscx.TrainingDataset(
                scx_path, hvg_indices=np.array([3, 3, 5, 5, 9], dtype=np.uint32)
            )
        assert ds.n_output_genes == 3

    def test_a_canonical_hvg_panel_warns_about_nothing(self, scx_path, recwarn):
        """The half that keeps the warning from becoming noise: the recipe
        every doc uses -- `np.where(...)[0]` -- is already ascending and
        unique, so it must stay silent."""
        ds = pyscx.TrainingDataset(
            scx_path, hvg_indices=np.array([0, 5, 10, 49], dtype=np.uint32)
        )
        assert ds.n_output_genes == 4
        assert [w for w in recwarn if "hvg_indices" in str(w.message)] == []

    def test_repr(self, scx_path):
        ds = pyscx.TrainingDataset(scx_path)
        r = repr(ds)
        assert "TrainingDataset" in r
        assert "n_obs=100" in r

    def test_invalid_batch_size(self, scx_path):
        with pytest.raises(RuntimeError, match="batch_size"):
            pyscx.TrainingDataset(scx_path, batch_size=0)


class TestIteration:
    """F1: for batch in dataset iterates exactly one epoch."""

    def test_one_epoch_all_cells(self, scx_path):
        ds = pyscx.TrainingDataset(
            scx_path,
            batch_size=32,
            normalize=False,
            log1p=False,
        )
        all_indices = []
        for batch in ds:
            all_indices.extend(batch["cell_indices"].tolist())

        # All 100 cells appear exactly once
        assert sorted(all_indices) == list(range(100))

    def test_batch_x_shape_and_dtype(self, scx_path):
        ds = pyscx.TrainingDataset(
            scx_path,
            batch_size=32,
            normalize=False,
            log1p=False,
        )
        for batch in ds:
            X = batch["X"]
            assert X.dtype == np.float32
            n_rows, n_genes = X.shape
            assert n_genes == 50
            assert n_rows <= 32  # last batch may be smaller
            break  # only need first batch for shape/dtype check

    def test_cell_indices_dtype(self, scx_path):
        ds = pyscx.TrainingDataset(
            scx_path,
            batch_size=32,
            normalize=False,
            log1p=False,
        )
        for batch in ds:
            assert batch["cell_indices"].dtype == np.int64
            break

    def test_hvg_projection_columns(self, scx_path):
        hvg = [0, 5, 10]
        ds = pyscx.TrainingDataset(
            scx_path,
            batch_size=32,
            hvg_indices=hvg,
            normalize=False,
            log1p=False,
        )
        for batch in ds:
            assert batch["X"].shape[1] == 3
            break


class TestObsColumns:
    """F1: batch["obs"] contains requested columns with correct dtypes."""

    def test_obs_column_extraction(self, scx_path):
        ds = pyscx.TrainingDataset(
            scx_path,
            batch_size=32,
            obs_columns=["cell_id"],
            normalize=False,
            log1p=False,
        )
        for batch in ds:
            assert "cell_id" in batch["obs"]
            break

    def test_empty_obs_columns(self, scx_path):
        ds = pyscx.TrainingDataset(
            scx_path,
            batch_size=32,
            normalize=False,
            log1p=False,
        )
        for batch in ds:
            assert isinstance(batch["obs"], dict)
            break

    def test_categorical_codes_stable_across_batches(self, scx_path):
        """Regression: a category string must map to the SAME integer code in
        every batch, and the `categories` list must be identical across
        batches. Before the fix, the Utf8 obs path built a per-batch dictionary,
        so codes drifted with batch composition (silently wrong ML labels).

        `cell_id` is a plain-string (Utf8-path) column with one value per cell,
        so any batch-local encoding would assign the same string different codes
        across batches — exactly the bug this guards.
        """
        ds = pyscx.TrainingDataset(
            scx_path,
            batch_size=8,  # small → many batches per epoch
            obs_columns=["cell_id"],
            normalize=False,
            log1p=False,
        )

        reference_categories = None
        cell_to_code = {}
        n_batches = 0
        for batch in ds:
            col = batch["obs"]["cell_id"]
            codes = col["codes"]
            categories = list(col["categories"])
            cells = batch["cell_indices"].tolist()
            n_batches += 1

            # The global category list is identical in every batch.
            if reference_categories is None:
                reference_categories = categories
            else:
                assert categories == reference_categories

            for pos, cell in enumerate(cells):
                code = int(codes[pos])
                # Codes correctly index the global categories list.
                assert categories[code] == f"cell_{cell}"
                # The same cell (hence same string) always gets the same code.
                if cell in cell_to_code:
                    assert cell_to_code[cell] == code
                else:
                    cell_to_code[cell] = code

        assert n_batches > 1, "test must span multiple batches"
        # No two distinct cells (distinct strings) share a code.
        assert len(set(cell_to_code.values())) == len(cell_to_code)


class TestMultiEpoch:
    """F1: Second for batch in dataset loop starts a new epoch."""

    def test_two_epochs_cover_all_cells(self, scx_path):
        ds = pyscx.TrainingDataset(
            scx_path,
            batch_size=20,
            normalize=False,
            log1p=False,
        )

        epoch1_indices = []
        for batch in ds:
            epoch1_indices.extend(batch["cell_indices"].tolist())

        epoch2_indices = []
        for batch in ds:
            epoch2_indices.extend(batch["cell_indices"].tolist())

        # Both epochs cover all cells
        assert sorted(epoch1_indices) == list(range(100))
        assert sorted(epoch2_indices) == list(range(100))
        # Different order due to shuffle
        assert epoch1_indices != epoch2_indices


@pytest.fixture
def wide_scx_path(tmp_path):
    """A full-width-ish file whose decoded-shard cache alone exceeds the old
    512 MB fixed default, so the budget logic must engage (P3)."""
    import anndata as ad
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    n_obs, n_vars, nnz_per_cell = 100, 20_000, 1_000
    rows, cols, vals = [], [], []
    for r in range(n_obs):
        c = rng.choice(n_vars, size=nnz_per_cell, replace=False)
        rows.extend([r] * nnz_per_cell)
        cols.extend(c.tolist())
        vals.extend((rng.integers(1, 50, size=nnz_per_cell)).tolist())
    X = sp.csr_matrix(
        (np.array(vals, dtype=np.float32), (rows, cols)),
        shape=(n_obs, n_vars),
    )
    adata = ad.AnnData(X=X)
    path = str(tmp_path / "wide.scx")
    pyscx.from_anndata(adata, path)
    return path


class TestAdaptiveMemoryBudget:
    """P3: the default budget must fit a full-width file without silently
    shrinking the requested batch_size, while an explicit budget stays a
    hard ceiling."""

    def test_default_budget_preserves_batch_size(self, wide_scx_path):
        # No explicit max_memory_mb → adaptive: the requested batch_size must
        # survive and the budget must not be flagged exceeded.
        ds = pyscx.TrainingDataset(
            wide_scx_path, batch_size=512, normalize=True, log1p=True
        )
        mb = ds.memory_budget()
        assert ds.effective_batch_size == 512, (
            f"default budget should keep batch_size=512, got "
            f"{ds.effective_batch_size} (budget={mb})"
        )
        assert mb["batch_size"] == 512
        assert not mb["budget_exceeded"], f"default budget should fit: {mb}"

    def test_explicit_budget_is_hard_ceiling(self, wide_scx_path):
        # An explicit (too-small) budget must still behave as before: a hard
        # ceiling that auto-tunes the batch_size down rather than being raised.
        ds = pyscx.TrainingDataset(
            wide_scx_path,
            batch_size=512,
            normalize=True,
            log1p=True,
            max_memory_mb=512,
        )
        assert ds.effective_batch_size < 512, (
            "explicit max_memory_mb=512 should remain a hard ceiling and shrink "
            f"the batch, got {ds.effective_batch_size}"
        )


def _forked_worker(scx_path_str):
    """Worker function for fork detection test."""
    try:
        ds = pyscx.TrainingDataset(scx_path_str, normalize=False, log1p=False)
        for batch in ds:
            pass
        return "no_error"
    except RuntimeError as e:
        if "num_workers=0" in str(e):
            return "detected_fork"
        return f"unexpected_error: {e}"
    except Exception as e:
        return f"other_error: {e}"


class TestNumWorkersSafety:
    """F2: Fork detection raises RuntimeError."""

    def test_fork_detection(self, scx_path):
        """Verify that __next__ in a forked subprocess detects PID mismatch.

        We create the dataset in the parent, then os.fork(). The child
        inherits the TrainingDataset with creation_pid == parent's PID,
        but the child has a different PID, so __next__ should raise.
        """
        import os


        ds = pyscx.TrainingDataset(
            scx_path, batch_size=32, normalize=False, log1p=False
        )

        # Create a pipe for the child to communicate the result
        read_fd, write_fd = os.pipe()

        pid = os.fork()
        if pid == 0:
            # Child process
            os.close(read_fd)
            try:
                for batch in ds:
                    pass
                os.write(write_fd, b"no_error")
            except RuntimeError as e:
                if "num_workers=0" in str(e):
                    os.write(write_fd, b"detected_fork")
                else:
                    os.write(write_fd, b"unexpected_error")
            except Exception:
                os.write(write_fd, b"other_error")
            finally:
                os.close(write_fd)
                os._exit(0)
        else:
            # Parent process
            os.close(write_fd)
            os.waitpid(pid, 0)
            result = os.read(read_fd, 1024).decode()
            os.close(read_fd)

            assert result == "detected_fork", f"Expected fork detection, got: {result}"


class TestDropShutdown:
    """L3: dropping a dataset without close() runs the wrapper Drop (which
    releases the GIL around the pipeline join) cleanly — no hang, crash, or
    leak. A timing assertion on GIL-release is intentionally omitted: normal
    shutdown is sub-millisecond, so it would be flaky; the meaningful check is
    that the GC/`del` drop path is exercised and the interpreter stays healthy.
    """

    def test_drop_without_close_is_clean(self, scx_path):
        import gc

        ds = pyscx.TrainingDataset(scx_path, batch_size=32)
        it = iter(ds)
        _ = next(it)  # start an epoch so the I/O + decode threads are live
        del it
        del ds
        gc.collect()  # force the wrapper Drop -> detached pipeline shutdown

        # Interpreter is still healthy: a fresh dataset iterates a full epoch.
        ds2 = pyscx.TrainingDataset(
            scx_path, batch_size=32, normalize=False, log1p=False
        )
        n = sum(b["cell_indices"].shape[0] for b in ds2)
        assert n == ds2.n_obs

    def test_repeated_create_iterate_drop(self, scx_path):
        import gc

        for _ in range(5):
            ds = pyscx.TrainingDataset(
                scx_path, batch_size=16, normalize=False, log1p=False
            )
            got = 0
            for batch in ds:
                got += batch["cell_indices"].shape[0]
            assert got == ds.n_obs
            del ds
            gc.collect()

    def test_drop_at_interpreter_exit_is_clean(self, scx_path):
        """The in-test gc.collect() cases drop with a healthy interpreter. This
        covers the other L3 path: a dataset with live I/O/decode threads, never
        close()d and held in a module global, dropped during interpreter
        finalization. The wrapper Drop's Py_IsInitialized-guarded GIL release
        must not crash/abort the process at teardown.
        """
        import subprocess
        import sys
        import textwrap

        code = textwrap.dedent(
            f"""
            import pyscx
            ds = pyscx.TrainingDataset({scx_path!r}, batch_size=16)
            it = iter(ds)
            next(it)  # start an epoch so the I/O + decode threads are live
            # Hold a reference in a module global and do NOT close(): the
            # wrapper Drop runs during interpreter finalization.
            _held = ds
            print("ok")
            """
        )
        r = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, timeout=120
        )
        assert r.returncode == 0, (
            f"interpreter-exit drop crashed: rc={r.returncode}\n"
            f"stderr={r.stderr.decode()[-2000:]}"
        )
        assert b"ok" in r.stdout



NEGATIVE_CELLS = {
    (0, 0): -5.5,  # well below the singularity
    (1, 7): -1.0,  # exactly ln(x + 1)'s singularity -> -inf, not NaN
    (2, 3): -0.25,  # above it: log1p is defined, but the value is negative
}


class TestNegativeValuesOnTheLogPaths:
    """A stored value <= -1 made `log1p` produce NaN (or -inf) in the batch,
    silently and only on the dense loaders — the sparse gather has clipped for
    a while.

    The clip is scoped to the paths that take a log, so a pre-centered matrix
    streamed with no transforms still comes back signed.

    Uses a genuinely float matrix: the shared integer-count fixture is stored
    with an unsigned value encoding, so negatives never survive the write and
    the whole premise would be vacuous.
    """

    @pytest.fixture
    def signed_scx(self, tmp_path):
        import anndata
        import scipy.sparse as sp

        rng = np.random.default_rng(0)
        dense = (rng.random((40, 12)) * 10).astype(np.float32)
        dense[dense < 6.0] = 0.0
        for (row, col), value in NEGATIVE_CELLS.items():
            dense[row, col] = value
        path = str(tmp_path / "signed.scx")
        pyscx.from_anndata(anndata.AnnData(sp.csr_matrix(dense)), path)

        # The premise: the negatives really are on disk. Without this the
        # assertions below would pass against a file that never had them.
        stored = pyscx.open(path).to_anndata().X.toarray()
        for (row, col), value in NEGATIVE_CELLS.items():
            assert stored[row, col] == value, "fixture lost its negatives"
        return path

    def _epoch(self, path, **kwargs):
        """One epoch, re-indexed back to file row order.

        Batches arrive shuffled, so `cell_indices` — not row position — is what
        identifies a cell.
        """
        ds = pyscx.TrainingDataset(path, batch_size=16, **kwargs)
        rows, idx = [], []
        for batch in ds:
            rows.append(batch["X"])
            idx.append(batch["cell_indices"])
        ds.close()
        x = np.concatenate(rows, axis=0)
        order = np.argsort(np.concatenate(idx))
        return x[order]

    def test_log1p_emits_no_nan(self, signed_scx):
        x = self._epoch(signed_scx, normalize=False, log1p=True)
        assert np.isfinite(x).all(), "log1p must not put NaN/inf in the batch"
        for row, col in NEGATIVE_CELLS:
            assert x[row, col] == 0.0, f"({row}, {col}) should clip to log1p(0)"

    def test_normalize_log1p_emits_no_nan(self, signed_scx):
        x = self._epoch(signed_scx, normalize=True, log1p=True)
        assert np.isfinite(x).all()
        for row, col in NEGATIVE_CELLS:
            assert x[row, col] == 0.0

    def test_pflog_emits_no_nan(self, signed_scx):
        """pflog sums the per-gene deltas into a per-cell baseline, so one bad
        value NaN'd every gene in that row, not just its own."""
        x = self._epoch(signed_scx, pflog=True, pflog_alpha=0.05)
        assert np.isfinite(x).all(), "one negative must not NaN a whole row"

    def test_no_transform_still_returns_signed_values(self, signed_scx):
        """The deliberate scope of the clip. Someone streaming a centered
        matrix with transforms off keeps their negatives."""
        x = self._epoch(signed_scx, normalize=False, log1p=False)
        for (row, col), value in NEGATIVE_CELLS.items():
            assert x[row, col] == value
