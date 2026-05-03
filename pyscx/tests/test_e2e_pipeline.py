"""End-to-end pipeline tests for the scx-loader training pipeline.

These tests validate the full TrainingDataset pipeline from Python, covering
correctness, shuffle behaviour, HVG projection, normalization, memory budget,
fork detection, single-shard files, epoch lifecycle, reproducibility, and
oversized batch sizes.
"""

import numpy as np
import pyscx
import pytest
import resource
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_small_adata(n_obs, n_vars, density=0.3, seed=42):
    """Create a small synthetic AnnData with integer counts."""
    import anndata
    import pandas as pd

    rng = np.random.RandomState(seed)
    dense = rng.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
    mask = rng.random((n_obs, n_vars)) > density
    dense[mask] = 0
    x = sp.csr_matrix(dense)

    obs = pd.DataFrame(
        {"cell_id": [f"cell_{i}" for i in range(n_obs)]},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )

    return anndata.AnnData(X=x, obs=obs, var=var)


@pytest.fixture
def scx_path(tmp_path, synthetic_adata):
    """Write the shared synthetic_adata fixture to SCX."""
    path = str(tmp_path / "e2e.scx")
    pyscx.from_anndata(synthetic_adata, path)
    return path


# ===========================================================================
# H1 Tests
# ===========================================================================


class TestE2EPipeline:
    """H1: End-to-end pipeline integration tests."""

    # 1. Pipeline produces all cells exactly once per epoch
    def test_all_cells_once_per_epoch(self, scx_path):
        ds = pyscx.TrainingDataset(
            scx_path, batch_size=32, normalize=False, log1p=False
        )
        all_indices = []
        for batch in ds:
            all_indices.extend(batch["cell_indices"].tolist())

        assert sorted(all_indices) == list(range(100)), (
            "every cell [0..100) should appear exactly once"
        )

    # 2. Shuffle — different epoch seeds produce different batch orders
    def test_shuffle_different_epoch_orders(self, scx_path):
        ds = pyscx.TrainingDataset(
            scx_path, batch_size=20, normalize=False, log1p=False
        )

        epoch1_indices = []
        for batch in ds:
            epoch1_indices.extend(batch["cell_indices"].tolist())

        epoch2_indices = []
        for batch in ds:
            epoch2_indices.extend(batch["cell_indices"].tolist())

        # Both cover all cells
        assert sorted(epoch1_indices) == list(range(100))
        assert sorted(epoch2_indices) == list(range(100))
        # Order should differ
        assert epoch1_indices != epoch2_indices, (
            "different epochs should produce different batch orderings"
        )

    # 3. HVG projection — batch X has n_hvg columns, not n_vars
    def test_hvg_projection_output_shape(self, scx_path):
        hvg = [0, 5, 10, 20, 40]
        ds = pyscx.TrainingDataset(
            scx_path,
            batch_size=32,
            hvg_indices=hvg,
            normalize=False,
            log1p=False,
        )
        for batch in ds:
            assert batch["X"].shape[1] == len(hvg), (
                f"expected {len(hvg)} HVG columns, got {batch['X'].shape[1]}"
            )
            assert batch["X"].dtype == np.float32
            break  # checking first batch is sufficient

    # 4. Fused normalize+log1p matches scanpy result
    def test_normalize_log1p_matches_scanpy(self, scx_path, synthetic_adata):
        import scanpy as sc

        target_sum = 1e4

        # --- Loader output (normalize + log1p) ---
        ds = pyscx.TrainingDataset(
            scx_path,
            batch_size=200,  # >= n_obs so we get one batch
            normalize=True,
            log1p=True,
            target_sum=target_sum,
        )

        loader_rows = {}
        for batch in ds:
            X = batch["X"]
            indices = batch["cell_indices"]
            for i, idx in enumerate(indices):
                loader_rows[int(idx)] = X[i]

        assert len(loader_rows) == 100

        # --- Scanpy reference ---
        adata = synthetic_adata.copy()
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        ref_X = adata.X
        if sp.issparse(ref_X):
            ref_X = ref_X.toarray()
        ref_X = ref_X.astype(np.float32)

        # --- Compare per-cell ---
        for cell_idx in range(100):
            loader_row = loader_rows[cell_idx]
            scanpy_row = ref_X[cell_idx]
            np.testing.assert_allclose(
                loader_row,
                scanpy_row,
                rtol=1e-5,
                atol=1e-6,
                err_msg=(
                    f"cell {cell_idx}: loader output does not match scanpy"
                ),
            )

    # 5. Memory budget — peak RSS stays within max_memory_mb
    def test_memory_budget_rss(self, scx_path):
        max_mb = 512

        # ru_maxrss is monotonic peak across the whole process, so a raw
        # absolute check is dominated by pytest + scanpy/torch imports and
        # prior tests in the session — it has flaked at ~1025/1024 MB on
        # CI runners. Snapshot peak before the pipeline so we measure only
        # the rise caused by iteration itself.
        peak_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        ds = pyscx.TrainingDataset(
            scx_path,
            batch_size=32,
            normalize=False,
            log1p=False,
            max_memory_mb=max_mb,
        )

        for batch in ds:
            pass  # consume all batches

        peak_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        delta_mb = (peak_after_kb - peak_before_kb) / 1024

        # Synthetic dataset is tiny (100 cells × 50 genes ≈ 20 KB); the
        # pipeline shouldn't grow peak RSS anywhere near max_memory_mb.
        assert delta_mb < max_mb, (
            f"pipeline raised peak RSS by {delta_mb:.0f} MB "
            f"(exceeds {max_mb} MB budget)"
        )

    # 6. num_workers > 0 raises clear error
    def test_num_workers_raises(self, scx_path):
        """Fork detection: iterating in a forked child should raise RuntimeError."""
        import os

        ds = pyscx.TrainingDataset(
            scx_path, batch_size=32, normalize=False, log1p=False
        )

        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            # Child process
            os.close(read_fd)
            try:
                for _batch in ds:
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
            os.close(write_fd)
            os.waitpid(pid, 0)
            result = os.read(read_fd, 1024).decode()
            os.close(read_fd)
            assert result == "detected_fork", (
                f"expected fork detection, got: {result}"
            )

    # 7. Single-shard file works (group_size > n_shards gracefully handled)
    def test_single_shard_file(self, tmp_path):
        adata = _make_small_adata(10, 20, density=0.4)
        path = str(tmp_path / "single_shard.scx")
        pyscx.from_anndata(adata, path)

        ds = pyscx.TrainingDataset(
            path,
            batch_size=100,
            shard_group_size=8,  # larger than n_shards
            normalize=False,
            log1p=False,
        )

        all_indices = []
        for batch in ds:
            all_indices.extend(batch["cell_indices"].tolist())

        assert sorted(all_indices) == list(range(10)), (
            "single-shard file should yield all cells"
        )

    # 8. End-of-epoch — __next__ returns None, new epoch restarts
    def test_end_of_epoch_restart(self, scx_path):
        ds = pyscx.TrainingDataset(
            scx_path, batch_size=200, normalize=False, log1p=False
        )

        # First epoch
        epoch1 = list(ds)
        assert len(epoch1) >= 1, "first epoch should yield batches"
        total_cells_e1 = sum(b["cell_indices"].shape[0] for b in epoch1)
        assert total_cells_e1 == 100

        # Second epoch — `for batch in ds` triggers __iter__ again
        epoch2 = list(ds)
        assert len(epoch2) >= 1, "second epoch should yield batches"
        total_cells_e2 = sum(b["cell_indices"].shape[0] for b in epoch2)
        assert total_cells_e2 == 100

    # 9. Reproducibility — same seed → same batch sequence
    def test_reproducibility_same_seed(self, scx_path):
        seed = 99

        ds1 = pyscx.TrainingDataset(
            scx_path,
            batch_size=32,
            seed=seed,
            normalize=False,
            log1p=False,
        )
        indices1 = []
        for batch in ds1:
            indices1.extend(batch["cell_indices"].tolist())

        ds2 = pyscx.TrainingDataset(
            scx_path,
            batch_size=32,
            seed=seed,
            normalize=False,
            log1p=False,
        )
        indices2 = []
        for batch in ds2:
            indices2.extend(batch["cell_indices"].tolist())

        assert indices1 == indices2, (
            "same seed should produce identical batch sequences"
        )

    # 10. Large batch_size > total cells → single batch per epoch with all cells
    def test_large_batch_single_batch(self, scx_path):
        ds = pyscx.TrainingDataset(
            scx_path,
            batch_size=1000,  # >> 100 cells
            normalize=False,
            log1p=False,
        )

        batches = list(ds)
        assert len(batches) == 1, (
            f"expected exactly 1 batch, got {len(batches)}"
        )

        batch = batches[0]
        assert batch["X"].shape[0] == 100
        assert sorted(batch["cell_indices"].tolist()) == list(range(100))
