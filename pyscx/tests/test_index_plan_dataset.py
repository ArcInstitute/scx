"""Integration tests for `pyscx.IndexPlanDataset`.

Coverage:
- Smoke (dtypes, shapes).
- Determinism (same plan stream → identical batches).
- lookahead=0 vs lookahead=4 correctness.
- Error paths (bad index, missing obs col, OOR HVG, cache_shards=0).
- Fork detection (os.fork pattern).
- Schema parity with `TrainingDataset` (categorical encoding).
- Memory-budget auto-tune surfaces.
- Empty plan list, generator-driven plans, normalize+log1p semantics.
"""

import os

import numpy as np
import pyscx
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def scx_path(tmp_path, synthetic_adata):
    """Write the shared synthetic_adata fixture (100x50, batch column) to SCX."""
    path = str(tmp_path / "index_plan.scx")
    pyscx.from_anndata(synthetic_adata, path)
    return path


def _dense_from_adata(adata):
    """Return the dense X as a numpy array, handling sparse and dense cases."""
    x = adata.X
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


class TestSmoke:
    def test_basic_iter_dtypes_and_shapes(self, scx_path):
        ds = pyscx.IndexPlanDataset(
            scx_path,
            obs_columns=["cell_id", "batch"],
            normalize=False,
        )
        plans = [
            [(0, 1), (2, 3)],
            [(10, 20), (30, 40), (50, 60)],
            [(99, 0)],
        ]
        batches = list(ds.iter_with_plans(iter(plans), lookahead=2))
        assert len(batches) == 3
        for i, b in enumerate(batches):
            n = len(plans[i])
            assert b["X"].shape == (n, ds.n_output_genes)
            assert b["X"].dtype == np.float32
            assert b["X_paired"].shape == (n, ds.n_output_genes)
            assert b["X_paired"].dtype == np.float32
            assert isinstance(b["pairs"], list)
            assert len(b["pairs"]) == n
            for p in b["pairs"]:
                assert isinstance(p, tuple) and len(p) == 2
            assert "cell_id" in b["obs"]
            assert "batch" in b["obs"]
            assert "cell_id" in b["obs_paired"]
            assert "batch" in b["obs_paired"]

    def test_n_obs_n_vars_n_output_genes(self, scx_path):
        ds = pyscx.IndexPlanDataset(scx_path)
        assert ds.n_obs == 100
        assert ds.n_vars == 50
        assert ds.n_output_genes == 50

    def test_hvg_n_output_genes(self, scx_path):
        hvg = np.array([0, 5, 10, 17, 23], dtype=np.uint32)
        ds = pyscx.IndexPlanDataset(scx_path, hvg_indices=hvg)
        assert ds.n_output_genes == 5

    def test_repr_contains_shape(self, scx_path):
        ds = pyscx.IndexPlanDataset(scx_path)
        s = repr(ds)
        assert "n_obs=100" in s
        assert "n_vars=50" in s
        assert "n_output_genes=50" in s


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_plan_yields_identical_batches(self, scx_path):
        ds = pyscx.IndexPlanDataset(
            scx_path,
            obs_columns=["batch"],
            normalize=True,
            target_sum=1e4,
        )
        plans = [[(0, 1), (2, 3)], [(50, 75), (10, 99)]]

        a = list(ds.iter_with_plans(iter([list(p) for p in plans]), lookahead=2))
        b = list(ds.iter_with_plans(iter([list(p) for p in plans]), lookahead=2))

        assert len(a) == len(b)
        for ba, bb in zip(a, b):
            np.testing.assert_array_equal(ba["X"], bb["X"])
            np.testing.assert_array_equal(ba["X_paired"], bb["X_paired"])
            assert ba["pairs"] == bb["pairs"]
            assert sorted(ba["obs"]["batch"]["codes"].tolist()) == sorted(
                bb["obs"]["batch"]["codes"].tolist()
            )
            assert ba["obs"]["batch"]["categories"] == bb["obs"]["batch"]["categories"]

    def test_iter_vs_next_batch_for_test_parity(self, scx_path):
        ds = pyscx.IndexPlanDataset(
            scx_path,
            obs_columns=["batch"],
            normalize=False,
        )
        plan = [(0, 1), (50, 99), (10, 20)]
        via_iter = next(ds.iter_with_plans(iter([list(plan)]), lookahead=2))
        via_method = ds._next_batch_for_test(list(plan))
        np.testing.assert_array_equal(via_iter["X"], via_method["X"])
        np.testing.assert_array_equal(via_iter["X_paired"], via_method["X_paired"])
        assert via_iter["pairs"] == via_method["pairs"]


# ---------------------------------------------------------------------------
# Lookahead parity
# ---------------------------------------------------------------------------


class TestLookaheadParity:
    def test_lookahead_zero_vs_four_same_output(self, scx_path):
        plans = [[(15, 0), (4, 5), (8, 9)], [(3, 12), (10, 11), (1, 2)]]

        ds = pyscx.IndexPlanDataset(
            scx_path,
            obs_columns=["cell_id"],
            normalize=False,
        )
        b0 = list(ds.iter_with_plans(iter([list(p) for p in plans]), lookahead=0))
        b4 = list(ds.iter_with_plans(iter([list(p) for p in plans]), lookahead=4))
        assert len(b0) == len(b4)
        for ba, bb in zip(b0, b4):
            np.testing.assert_array_equal(ba["X"], bb["X"])
            np.testing.assert_array_equal(ba["X_paired"], bb["X_paired"])
            assert ba["pairs"] == bb["pairs"]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_out_of_range_pert_index_raises_indexerror(self, scx_path):
        ds = pyscx.IndexPlanDataset(scx_path)
        with pytest.raises(IndexError):
            ds._next_batch_for_test([(99999, 0)])

    def test_out_of_range_ctrl_index_raises_indexerror(self, scx_path):
        ds = pyscx.IndexPlanDataset(scx_path)
        with pytest.raises(IndexError):
            ds._next_batch_for_test([(0, 99999)])

    def test_out_of_range_via_iter_raises(self, scx_path):
        ds = pyscx.IndexPlanDataset(scx_path)
        with pytest.raises(IndexError):
            list(ds.iter_with_plans(iter([[(0, 0), (99999, 1)]]), lookahead=1))

    def test_missing_obs_column_raises_keyerror(self, scx_path):
        with pytest.raises(KeyError):
            pyscx.IndexPlanDataset(scx_path, obs_columns=["nonexistent"])

    def test_zero_cache_shards_rejected(self, scx_path):
        with pytest.raises(RuntimeError):
            pyscx.IndexPlanDataset(scx_path, cache_shards=0)

    def test_hvg_out_of_range_rejected_at_construction(self, scx_path):
        with pytest.raises(RuntimeError):
            pyscx.IndexPlanDataset(
                scx_path,
                hvg_indices=np.array([99999], dtype=np.uint32),
            )

    def test_max_memory_below_floor_rejects(self, scx_path):
        with pytest.raises(RuntimeError) as excinfo:
            pyscx.IndexPlanDataset(scx_path, max_memory_mb=10)
        assert "Increase max_memory_mb" in str(excinfo.value)

    def test_plan_iterator_raising_propagates(self, scx_path):
        ds = pyscx.IndexPlanDataset(scx_path)

        def bad():
            yield [(0, 1)]
            raise ValueError("synthetic plan failure")

        with pytest.raises(RuntimeError) as excinfo:
            list(ds.iter_with_plans(bad(), lookahead=2))
        assert "plan iterator raised" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Fork detection
# ---------------------------------------------------------------------------


class TestForkDetection:
    def test_iter_in_forked_child_raises(self, scx_path):
        """Mirrors `TrainingDataset` fork detection: iter_with_plans in a child
        process must raise RuntimeError with `num_workers=0` guidance."""
        ds = pyscx.IndexPlanDataset(scx_path)

        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            try:
                list(ds.iter_with_plans(iter([[(0, 1)]]), lookahead=1))
                os.write(write_fd, b"no_error")
            except RuntimeError as e:
                if "num_workers=0" in str(e):
                    os.write(write_fd, b"detected_fork")
                else:
                    os.write(write_fd, b"unexpected_runtime_error")
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
            assert result == "detected_fork", f"expected fork detection, got: {result}"

    def test_next_batch_in_forked_child_raises(self, scx_path):
        ds = pyscx.IndexPlanDataset(scx_path)
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            try:
                ds._next_batch_for_test([(0, 1)])
                os.write(write_fd, b"no_error")
            except RuntimeError as e:
                if "num_workers=0" in str(e):
                    os.write(write_fd, b"detected_fork")
                else:
                    os.write(write_fd, b"unexpected_runtime_error")
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
            assert result == "detected_fork", f"expected fork detection, got: {result}"


# ---------------------------------------------------------------------------
# Schema parity with TrainingDataset
# ---------------------------------------------------------------------------


class TestSchemaParity:
    def test_categorical_encoding_matches_training_dataset(self, scx_path):
        """Both backends must encode the categorical `batch` column the same
        way: `{"codes": ndarray[i32], "categories": list[str]}`. Codes for the
        same set of cells must match across backends."""
        # Pull the same 5 cells via both backends.
        cell_idxs = [0, 1, 2, 3, 4]

        # IndexPlanDataset → use _next_batch_for_test to control exactly which
        # cells appear (no shuffle, no shard sort).
        ip = pyscx.IndexPlanDataset(
            scx_path,
            obs_columns=["batch"],
            normalize=False,
            sort_by_shard=False,
        )
        plan = [(c, c) for c in cell_idxs]  # pair each cell with itself
        b_ip = ip._next_batch_for_test(plan)
        ip_batch = b_ip["obs"]["batch"]
        assert isinstance(ip_batch, dict)
        assert "codes" in ip_batch and "categories" in ip_batch
        assert ip_batch["codes"].dtype == np.int32

        # TrainingDataset over the same file with batch_size=100 (single batch)
        # so we can pick out the indices we care about.
        td = pyscx.TrainingDataset(
            scx_path,
            batch_size=100,
            obs_columns=["batch"],
            normalize=False,
            log1p=False,
        )
        td_batches = list(td)
        assert len(td_batches) == 1
        td_batch = td_batches[0]
        td_obs_batch = td_batch["obs"]["batch"]
        assert isinstance(td_obs_batch, dict)
        assert "codes" in td_obs_batch and "categories" in td_obs_batch
        assert td_obs_batch["codes"].dtype == np.int32

        # Categories list must match.
        assert ip_batch["categories"] == td_obs_batch["categories"]

        # Codes for the requested cells must match. TrainingDataset returns
        # all cells in catalog order; index in by cell_indices.
        td_cell_indices = td_batch["cell_indices"]
        td_codes = td_obs_batch["codes"]
        td_lookup = {int(td_cell_indices[i]): int(td_codes[i]) for i in range(len(td_codes))}
        for i, c in enumerate(cell_idxs):
            assert int(ip_batch["codes"][i]) == td_lookup[c], (
                f"cell {c}: IndexPlanDataset code {ip_batch['codes'][i]} != "
                f"TrainingDataset code {td_lookup[c]}"
            )


# ---------------------------------------------------------------------------
# Memory budget surface
# ---------------------------------------------------------------------------


class TestEffectiveBudget:
    def test_generous_budget_no_autotune(self, scx_path):
        ds = pyscx.IndexPlanDataset(
            scx_path,
            cache_shards=64,
            lookahead=4,
            max_plan_size=1024,
            max_memory_mb=2048,
        )
        assert ds.effective_cache_shards() == 64
        assert ds.effective_lookahead() == 4

    def test_tight_budget_reduces_lookahead(self, scx_path):
        # max_plan_size=65536 → lookahead overhead ≈ 1 MB/unit.
        # batch_buffer = 2 × 65536 × 50 × 4 ≈ 25 MB. py = 50 MB.
        # At lookahead=8: ≈ 83 MB. Budget 80 forces lookahead reduction.
        ds = pyscx.IndexPlanDataset(
            scx_path,
            cache_shards=8,
            lookahead=8,
            max_plan_size=65536,
            max_memory_mb=80,
        )
        assert ds.effective_lookahead() < 8
        assert ds.effective_lookahead() >= 1
        assert ds.effective_cache_shards() == 8

    def test_iter_default_lookahead_uses_effective(self, scx_path):
        ds = pyscx.IndexPlanDataset(
            scx_path,
            cache_shards=8,
            lookahead=8,
            max_plan_size=65536,
            max_memory_mb=80,
        )
        # No explicit lookahead → uses effective_lookahead().
        it = ds.iter_with_plans(iter([[(0, 1)]]))
        assert f"lookahead={ds.effective_lookahead()}" in repr(it)
        list(it)

    def test_explicit_iter_lookahead_overrides(self, scx_path):
        ds = pyscx.IndexPlanDataset(scx_path, lookahead=4)
        it = ds.iter_with_plans(iter([[(0, 1)]]), lookahead=2)
        assert "lookahead=2" in repr(it)
        list(it)


# ---------------------------------------------------------------------------
# Plan stream edge cases
# ---------------------------------------------------------------------------


class TestPlanStreamEdgeCases:
    def test_empty_plan_list_yields_no_batches(self, scx_path):
        ds = pyscx.IndexPlanDataset(scx_path)
        out = list(ds.iter_with_plans(iter([]), lookahead=4))
        assert out == []

    def test_empty_plan_inside_stream_is_skipped(self, scx_path):
        """Per spec: 'Plan list is empty → yield no batch for that plan,
        continue to the next.' iter_with_plans silently filters empties."""
        ds = pyscx.IndexPlanDataset(scx_path, obs_columns=["batch"])
        plans = [[(0, 1)], [], [(2, 3)]]
        batches = list(ds.iter_with_plans(iter([list(p) for p in plans]), lookahead=2))
        assert len(batches) == 2
        assert batches[0]["pairs"] == [(0, 1)]
        assert batches[1]["pairs"] == [(2, 3)]

    def test_generator_driven_plans(self, scx_path):
        ds = pyscx.IndexPlanDataset(scx_path)

        def gen():
            yield [(0, 1)]
            yield [(2, 3), (4, 5)]
            yield [(6, 7)]

        batches = list(ds.iter_with_plans(gen(), lookahead=2))
        assert len(batches) == 3
        assert [len(b["pairs"]) for b in batches] == [1, 2, 1]

    def test_drop_iter_mid_stream_does_not_deadlock(self, scx_path):
        """Iterator drop mid-stream must release the plan-pull worker
        promptly, even with a long generator."""
        ds = pyscx.IndexPlanDataset(scx_path)
        it = ds.iter_with_plans(
            iter([[(0, 1)] for _ in range(1000)]),
            lookahead=4,
        )
        _ = next(it)
        del it  # must not hang


# ---------------------------------------------------------------------------
# normalize+log1p semantics
# ---------------------------------------------------------------------------


class TestNormalizeLog1p:
    def test_normalize_log1p_matches_manual(self, scx_path, synthetic_adata):
        """End-to-end: normalize+log1p output matches a manual numpy
        reference computed from the source dense matrix."""
        target_sum = 1e4
        ds = pyscx.IndexPlanDataset(
            scx_path,
            normalize=True,
            target_sum=target_sum,
            sort_by_shard=False,
        )
        plan = [(0, 1), (50, 99)]
        b = ds._next_batch_for_test(plan)

        dense = _dense_from_adata(synthetic_adata)

        def manual(row):
            v = row.astype(np.float64)
            s = v.sum()
            if s > 0:
                v = v * (target_sum / s)
            return np.log1p(v).astype(np.float32)

        for i, (p, c) in enumerate(plan):
            np.testing.assert_allclose(b["X"][i], manual(dense[p]), rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(
                b["X_paired"][i], manual(dense[c]), rtol=1e-5, atol=1e-6
            )

    def test_no_normalize_returns_raw(self, scx_path, synthetic_adata):
        ds = pyscx.IndexPlanDataset(scx_path, normalize=False, sort_by_shard=False)
        b = ds._next_batch_for_test([(7, 42)])
        dense = _dense_from_adata(synthetic_adata)
        np.testing.assert_array_equal(b["X"][0], dense[7])
        np.testing.assert_array_equal(b["X_paired"][0], dense[42])
