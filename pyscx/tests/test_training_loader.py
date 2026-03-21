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

