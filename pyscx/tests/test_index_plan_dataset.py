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
import warnings

import numpy as np
import pyscx
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def scx_path(tmp_path, synthetic_adata):
    """Write the shared synthetic_adata fixture (100x50, batch column) to SCX.

    Uses the default (framed, v4) write path — the realistic modern layout."""
    path = str(tmp_path / "index_plan.scx")
    pyscx.from_anndata(synthetic_adata, path)
    return path


@pytest.fixture
def unframed_scx_path(tmp_path, synthetic_adata):
    """Unframed (v3) fixture for the whole-shard LRU cache / prefetch tests.

    Framing (the default since F5 Phase C) routes a sparse scattered gather
    through the block-index path, which decodes only the touched row-groups and
    does NOT populate the whole-shard LRU. These tests exercise that cache and
    its prefetch skip, so they need the unframed full-shard-decode path
    (`row_group_rows=0`). The block-index path has its own coverage in
    `TestBlockIndexAdoption` + the `read_scattered` benchmark."""
    path = str(tmp_path / "index_plan_unframed.scx")
    pyscx.from_anndata(synthetic_adata, path, row_group_rows=0)
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

    def test_plan_extraction_failure_names_the_pair_shape(self, scx_path):
        """A non-plan object from the generator must surface the *pair* arm's
        extraction diagnostic, not the cell-set one.

        Pinned because ORG-9.10-3 folds the two Python->Rust plan adapters into
        one generic: the two arms' extraction messages are the only thing that
        distinguishes them for a user, and nothing asserted either before this.
        A dedup that collapsed both onto one message would otherwise be silent.
        """
        ds = pyscx.IndexPlanDataset(scx_path)

        def bad():
            yield "not a plan at all"

        with pytest.raises(RuntimeError) as excinfo:
            list(ds.iter_with_plans(bad(), lookahead=2))
        msg = str(excinfo.value)
        assert "plan extraction failed" in msg, msg
        # The pair arm must NOT borrow the cell-set arm's tuple description.
        assert "file_ids" not in msg, msg


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
        # batch_buffer ≈ 25 MB; py = 50 MB; transient ≈ 8 MB (post-L1 dedup
        # gather: unique_rows + row_to_requests + row_to_pos, ≈64 B/row).
        # Floor @ lookahead=1 ≈ 84 MB; full @ lookahead=8 ≈ 91 MB. Budget 88
        # forces lookahead reduction without floor-failing.
        ds = pyscx.IndexPlanDataset(
            scx_path,
            cache_shards=8,
            lookahead=8,
            max_plan_size=65536,
            max_memory_mb=88,
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
            max_memory_mb=88,
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
        # `log1p=False` is required to actually get raw output — the
        # default `log1p=True` is now honoured independently of
        # `normalize` (was silently ignored before the fix).
        ds = pyscx.IndexPlanDataset(
            scx_path, normalize=False, log1p=False, sort_by_shard=False
        )
        b = ds._next_batch_for_test([(7, 42)])
        dense = _dense_from_adata(synthetic_adata)
        np.testing.assert_array_equal(b["X"][0], dense[7])
        np.testing.assert_array_equal(b["X_paired"][0], dense[42])

    def test_normalize_only_matches_manual(self, scx_path, synthetic_adata):
        """`normalize=True, log1p=False` — row-sum scaling, no log1p.

        Regression test for the bug where `IndexPlanLoader` always called
        the fused `normalize+log1p` primitive whenever `normalize=True`,
        silently applying log1p the caller did not request.
        """
        target_sum = 1e4
        ds = pyscx.IndexPlanDataset(
            scx_path,
            normalize=True,
            log1p=False,
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
            return v.astype(np.float32)

        for i, (p, c) in enumerate(plan):
            np.testing.assert_allclose(
                b["X"][i], manual(dense[p]), rtol=1e-5, atol=1e-6
            )
            np.testing.assert_allclose(
                b["X_paired"][i], manual(dense[c]), rtol=1e-5, atol=1e-6
            )

    def test_log1p_only_matches_manual(self, scx_path, synthetic_adata):
        """`normalize=False, log1p=True` — `ln(1+raw)` per element.

        Regression test for the state-scx ST default. The pre-fix
        `IndexPlanLoader` skipped the transform entirely when
        `normalize=False`, returning raw rows.
        """
        ds = pyscx.IndexPlanDataset(
            scx_path,
            normalize=False,
            log1p=True,
            sort_by_shard=False,
        )
        plan = [(0, 1), (50, 99)]
        b = ds._next_batch_for_test(plan)

        dense = _dense_from_adata(synthetic_adata)

        for i, (p, c) in enumerate(plan):
            np.testing.assert_allclose(
                b["X"][i], np.log1p(dense[p]).astype(np.float32),
                rtol=1e-5, atol=1e-6,
            )
            np.testing.assert_allclose(
                b["X_paired"][i], np.log1p(dense[c]).astype(np.float32),
                rtol=1e-5, atol=1e-6,
            )


# ---------------------------------------------------------------------------
# Cache + prefetch metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_dataset_cache_metrics_keys_and_types(self, scx_path):
        ds = pyscx.IndexPlanDataset(scx_path)
        m = ds.cache_metrics()
        assert isinstance(m, dict)
        assert set(m) == {
            "hits",
            "misses",
            "evictions",
            "bytes_inserted",
            "duplicate_waiters",
            "peak_bytes_in_cache",
            "full_shard_groups",
            "block_index_groups",
            "row_group_hits",
            "row_group_misses",
            "row_group_evictions",
            "row_group_bytes_inserted",
            "row_group_duplicate_waiters",
        }
        for k, v in m.items():
            assert isinstance(v, int), f"{k} should be int, got {type(v)}"
            assert v >= 0

    def test_cache_metrics_advance_after_iteration(self, unframed_scx_path):
        ds = pyscx.IndexPlanDataset(unframed_scx_path)
        before = ds.cache_metrics()

        plans = [[(0, 1), (2, 3)], [(0, 1), (2, 3)]]
        for _ in ds.iter_with_plans(iter(plans), lookahead=2):
            pass

        after = ds.cache_metrics()
        # First plan touches at least one shard → at least one decode (miss).
        # Second plan re-touches the same shards → hits or duplicate_waiters
        # advance. Either way the cache saw activity.
        delta_misses = after["misses"] - before["misses"]
        delta_hits = after["hits"] - before["hits"]
        assert delta_misses >= 1, "expected at least one shard decode"
        assert delta_hits + delta_misses + (
            after["duplicate_waiters"] - before["duplicate_waiters"]
        ) >= 2, "expected at least 2 cache lookups across the two plans"
        assert after["bytes_inserted"] > before["bytes_inserted"]

    def test_iter_metrics_dict_shape(self, scx_path):
        ds = pyscx.IndexPlanDataset(scx_path)
        plans = [[(0, 1)]]
        it = ds.iter_with_plans(iter(plans), lookahead=2)
        # Drain.
        for _ in it:
            pass

        m = it.metrics()
        assert set(m) == {"cache", "prefetch"}
        assert set(m["prefetch"]) == {
            "prefetch_tasks_spawned",
            "prefetch_skipped_cache_hit",
            "prefetch_skipped_in_flight",
            "prefetch_skipped_block_index",
        }
        assert set(m["cache"]) == {
            "hits",
            "misses",
            "evictions",
            "bytes_inserted",
            "duplicate_waiters",
            "peak_bytes_in_cache",
            "full_shard_groups",
            "block_index_groups",
            "row_group_hits",
            "row_group_misses",
            "row_group_evictions",
            "row_group_bytes_inserted",
            "row_group_duplicate_waiters",
        }

    def test_iter_skips_prefetch_after_warmup(self, unframed_scx_path):
        """Two iters back-to-back: the second sees fully cached shards and
        records `prefetch_skipped_cache_hit > 0` with `prefetch_tasks_spawned
        == 0`. Mirrors the Rust `iter_skips_prefetch_when_cached` integration
        test through the Python surface."""
        ds = pyscx.IndexPlanDataset(unframed_scx_path)
        plan = [(i, (i + 1) % 100) for i in range(20)]

        # Warm: first iter populates the LRU.
        for _ in ds.iter_with_plans(iter([plan]), lookahead=1):
            pass

        # Second iter: every touched shard is already cached.
        it2 = ds.iter_with_plans(iter([plan]), lookahead=1)
        for _ in it2:
            pass
        m = it2.metrics()
        assert m["prefetch"]["prefetch_tasks_spawned"] == 0, (
            f"warmed iter should not spawn prefetch tasks; got "
            f"{m['prefetch']['prefetch_tasks_spawned']}"
        )
        assert m["prefetch"]["prefetch_skipped_cache_hit"] > 0, (
            f"warmed iter should skip prefetches via cache hit; got "
            f"{m['prefetch']['prefetch_skipped_cache_hit']}"
        )

    def test_iter_metrics_survive_iter_drain(self, scx_path):
        """`metrics()` must work after the inner iterator is dropped (the
        Rust side drops `inner` to release the plan-pull thread). Our
        accessor caches the `Arc<...>` handles at construction so post-drain
        sampling is still valid."""
        ds = pyscx.IndexPlanDataset(scx_path)
        plans = [[(0, 1)]]
        it = ds.iter_with_plans(iter(plans), lookahead=1)
        # Fully drain — including the trailing None that drops inner.
        for _ in it:
            pass
        # First metrics() call after drain.
        m1 = it.metrics()
        # Second call returns the same snapshot view (still readable).
        m2 = it.metrics()
        assert m1["cache"]["misses"] == m2["cache"]["misses"]
        assert m1["prefetch"]["prefetch_tasks_spawned"] == m2["prefetch"][
            "prefetch_tasks_spawned"
        ]


# ---------------------------------------------------------------------------
# Memory budget breakdown
# ---------------------------------------------------------------------------


class TestMemoryBudget:
    def test_memory_budget_keys_and_total(self, scx_path):
        """Schema: every documented key is present, all `int`, and the
        per-component bytes sum to `total_bytes`.

        ORG-9.10-4 moved the six components under `breakdown` — where
        `TrainingDataset` had always reported them — so that
        `memory_budget()["breakdown"]` reads the same on every class that
        reports a budget.
        The exact-key-set assertion below is what made that move visible rather
        than silent; the cross-class invariant lives in
        `test_cache_sizing.py::TestMemoryBudgetEnvelope`.
        """
        ds = pyscx.IndexPlanDataset(scx_path)
        b = ds.memory_budget()
        assert set(b) == {
            "breakdown",
            "max_memory_mb",
            "effective_cache_shards",
            "effective_lookahead",
        }
        assert set(b["breakdown"]) == {
            "cache_bytes",
            "batch_buffer_bytes",
            "lookahead_overhead_bytes",
            "transient_bytes",
            "python_overhead_bytes",
            "total_bytes",
        }
        for k, v in b["breakdown"].items():
            assert isinstance(v, int), f"{k} should be int, got {type(v)}"
            assert v >= 0
        for k in ("max_memory_mb", "effective_cache_shards", "effective_lookahead"):
            assert isinstance(b[k], int) and b[k] >= 0, k
        bd = b["breakdown"]
        component_sum = (
            bd["cache_bytes"]
            + bd["batch_buffer_bytes"]
            + bd["lookahead_overhead_bytes"]
            + bd["transient_bytes"]
            + bd["python_overhead_bytes"]
        )
        assert component_sum == bd["total_bytes"]

    def test_memory_budget_matches_max_memory_mb(self, scx_path):
        """`total_bytes` must fit inside the user-configured budget — if
        construction succeeded, the auto-tune found a reduction that fits."""
        ds = pyscx.IndexPlanDataset(scx_path, max_memory_mb=256)
        b = ds.memory_budget()
        max_bytes = b["max_memory_mb"] * 1024 * 1024
        total = b["breakdown"]["total_bytes"]
        assert total <= max_bytes, (
            f"total_bytes ({total}) exceeds budget "
            f"({max_bytes}) — auto-tune is broken"
        )
        # Effective values mirror the loader's accessors.
        assert b["effective_cache_shards"] == ds.effective_cache_shards()
        assert b["effective_lookahead"] == ds.effective_lookahead()

    def test_peak_bytes_in_cache_advances(self, unframed_scx_path):
        """The new `peak_bytes_in_cache` gauge in `CacheMetrics` should
        advance past 0 once at least one shard has been decoded into the
        cache (driven via an iter)."""
        ds = pyscx.IndexPlanDataset(unframed_scx_path)
        # Sanity: schema-extended cache_metrics dict.
        before = ds.cache_metrics()
        assert "peak_bytes_in_cache" in before
        assert before["peak_bytes_in_cache"] == 0

        plans = [[(0, 1), (2, 3)]]
        for _ in ds.iter_with_plans(iter(plans), lookahead=1):
            pass

        after = ds.cache_metrics()
        assert after["peak_bytes_in_cache"] > 0, (
            "peak_bytes_in_cache should advance once any shard is cached"
        )
        # Without eviction the peak equals cumulative bytes_inserted. The gauge
        # spans whole shards and row groups (one budget), so the general bound
        # is `peak <= bytes_inserted + row_group_bytes_inserted`; this unframed
        # fixture retains no row groups, so the two coincide.
        assert after["row_group_bytes_inserted"] == 0, "unframed: no row groups"
        if after["evictions"] == 0:
            assert after["peak_bytes_in_cache"] == after["bytes_inserted"]
        assert after["peak_bytes_in_cache"] <= (
            after["bytes_inserted"] + after["row_group_bytes_inserted"]
        )


class TestBlockIndexAdoption:
    """F5 follow-up (Phase A): a row-group-framed file must take the
    codec-agnostic block-index scattered path — even with the Scx1 sidecar
    disabled — so a framed training file gets random-access decode. Proves the
    `block_index_eligible` predicate + the `scatter_block_index` kwarg end to
    end through pyscx (the Rust hard gate is
    `scx-format-io backed_tests::read_rows_with_block_index_when_sidecar_disabled`)."""

    @staticmethod
    def _write_framed(path, *, n_obs=400, n_vars=60, codec="shufdelta",
                      row_group_rows=16):
        import anndata as ad
        import scipy.sparse as sp

        X = sp.random(n_obs, n_vars, density=0.05, format="csr", random_state=0)
        X.data = np.round(X.data * 10 + 1).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.obs["cell_id"] = [f"c{i}" for i in range(n_obs)]
        # Framed shufdelta: no Scx1 sidecar, so the block index is the only
        # random-access route — isolates the counter under test.
        pyscx.from_anndata(adata, path, codec=codec, row_group_rows=row_group_rows)
        return adata

    def test_framed_scatter_takes_block_index_without_sidecar(self, tmp_path):
        path = str(tmp_path / "framed_shufdelta.scx")
        self._write_framed(path)

        rng = np.random.default_rng(0)
        ds = pyscx.IndexPlanDataset(
            path,
            normalize=False,
            cache_shards=4,
            sort_by_shard=True,
            lookahead=0,
            scatter_block_index=True,
        )
        # Sparse, unsorted, cold gather → block-index eligible.
        plan = [(int(rng.integers(0, 400)), int(rng.integers(0, 400)))
                for _ in range(8)]
        batches = list(ds.iter_with_plans(iter([plan]), lookahead=0))
        assert batches[0]["X"].shape == (8, 60)

        cm = ds.cache_metrics()
        assert cm["block_index_groups"] > 0, (
            "framed gather must take the block-index path with the sidecar off"
        )
        assert cm["full_shard_groups"] == 0

    def test_scatter_block_index_flag_gates_prefetch_skip(self, tmp_path):
        """The dataset `scatter_block_index` kwarg gates the L2 prefetch skip:
        with it on, a framed shard's warm is skipped (so the gather takes the
        block index); with it off, the prefetch warms the shard as before. The
        `prefetch_skipped_block_index` counter distinguishes the two."""
        path = str(tmp_path / "framed_shufdelta_prefetch.scx")
        self._write_framed(path)

        rng = np.random.default_rng(2)
        plan = [(int(rng.integers(0, 400)), int(rng.integers(0, 400)))
                for _ in range(8)]

        def drained_prefetch_metrics(scatter_block_index):
            ds = pyscx.IndexPlanDataset(
                path,
                normalize=False,
                cache_shards=4,
                sort_by_shard=True,
                lookahead=1,
                scatter_block_index=scatter_block_index,
            )
            it = ds.iter_with_plans(iter([list(plan)]), lookahead=1)
            list(it)
            return it.metrics()["prefetch"]

        on = drained_prefetch_metrics(True)
        off = drained_prefetch_metrics(False)
        assert "prefetch_skipped_block_index" in on
        assert on["prefetch_skipped_block_index"] > 0, (
            "framed shard warm must be skipped when scatter_block_index=True"
        )
        assert off["prefetch_skipped_block_index"] == 0, (
            "framed shard must be warmed (not skipped) when scatter_block_index=False"
        )

    def test_scatter_block_index_false_disables_l1_adoption(self, tmp_path):
        """`scatter_block_index=False` is a complete off-switch: it propagates to
        the reader so the L1 gather does NOT take the block-index path either
        (not just the L2 prefetch skip). At lookahead=0 (no prefetch) a framed
        gather must full-shard-decode instead of adopting the block index."""
        path = str(tmp_path / "framed_shufdelta_off.scx")
        self._write_framed(path)

        rng = np.random.default_rng(3)
        ds = pyscx.IndexPlanDataset(
            path,
            normalize=False,
            cache_shards=4,
            sort_by_shard=True,
            lookahead=0,
            scatter_block_index=False,
        )
        plan = [(int(rng.integers(0, 400)), int(rng.integers(0, 400)))
                for _ in range(8)]
        list(ds.iter_with_plans(iter([plan]), lookahead=0))
        cm = ds.cache_metrics()
        assert cm["block_index_groups"] == 0, (
            "L1 gather must not adopt block-index when scatter_block_index=False"
        )
        assert cm["full_shard_groups"] > 0

    def test_unframed_scatter_emits_preflight_warning(self, unframed_scx_path):
        """Opening an all-unframed (v3) file with scatter_block_index=True (the
        default) must emit a one-shot UserWarning: the block-index fast path
        cannot fire on unframed shards, so every batch full-shard-decodes. The
        warning points at `scx optimize --row-group-rows 256`."""
        with pytest.warns(UserWarning, match="row-group framed"):
            ds = pyscx.IndexPlanDataset(
                unframed_scx_path, normalize=False, scatter_block_index=True
            )
        # It still works — warn-and-continue, not a hard refusal.
        plan = [(0, 1), (2, 3)]
        batches = list(ds.iter_with_plans(iter([plan]), lookahead=0))
        assert batches[0]["X"].shape[0] == 2
        # And it did full-shard-decode (no block-index adoption on an unframed file).
        assert ds.cache_metrics()["block_index_groups"] == 0

    def test_unframed_scatter_off_is_silent(self, unframed_scx_path):
        """No preflight warning when the caller opts out — an unframed file with
        scatter_block_index=False is the intended full-shard path, not a footgun."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            pyscx.IndexPlanDataset(
                unframed_scx_path, normalize=False, scatter_block_index=False
            )

    def test_framed_scatter_has_no_preflight_warning(self, tmp_path):
        """A framed file opened with scatter_block_index=True must NOT emit the
        unframed preflight warning — the fast path is available."""
        path = str(tmp_path / "framed_no_warn.scx")
        self._write_framed(path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pyscx.IndexPlanDataset(
                path, normalize=False, scatter_block_index=True
            )
        assert not [w for w in caught if "row-group framed" in str(w.message)], (
            "a framed file must not trigger the unframed preflight warning"
        )


class TestFeedbackPlanGenerator:
    """§9.4 — prefetch depth is opportunistic, not an obligation.

    `refill` used to loop on a blocking `recv` until the in-flight queue held
    `lookahead` plans. A generator that yields plan i+1 only after inspecting
    batch i — curriculum / feedback sampling — wedged forever: the generator
    waited for batch i, the consumer waited for plan i+4.
    """

    def test_curriculum_generator_yields_every_batch(self, scx_path):
        """The generator *blocks* until it has seen the previous batch.

        Note it blocks rather than merely asserting it was not called ahead:
        the plan-pull thread eagerly buffers up to `lookahead` plans into its
        bounded channel, so being asked early is by design. What a real
        curriculum generator cannot do is *answer* early, and that is what
        used to wedge the loader.

        `Semaphore.acquire` releases the GIL while it waits, so the pull thread
        parking here does not stall the interpreter. Its timeout also turns the
        old deadlock into a failed assertion instead of a hung test run.
        """
        import threading

        ds = pyscx.IndexPlanDataset(
            scx_path, normalize=False, log1p=False, obs_columns=[]
        )
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
                # Gated on the ack, so this is now a real invariant: plan i is
                # computed from exactly the first i batches.
                assert len(seen) == i
                yield [(i * 2, i * 2 + 1)]

        # lookahead=4 is the default and the value that used to deadlock.
        for batch in ds.iter_with_plans(curriculum(), lookahead=4):
            seen.append(batch["X"].shape[0])
            ack.release()

        assert len(seen) == n_plans
        assert all(n == 1 for n in seen)

    def test_a_generator_that_lags_does_not_truncate_the_epoch(self, scx_path):
        """A merely slow generator must still produce every batch.

        Guards the tempting wrong fix: swapping the blocking `recv` for a
        `try_recv` whose empty arm ends the plan stream truncates the epoch
        silently, with no error anywhere.
        """
        import time

        ds = pyscx.IndexPlanDataset(
            scx_path, normalize=False, log1p=False, obs_columns=[]
        )
        n_plans = 6

        def slow():
            for i in range(n_plans):
                time.sleep(0.02)
                yield [(i, i + 1)]

        n = sum(1 for _ in ds.iter_with_plans(slow(), lookahead=4))
        assert n == n_plans, "a slow generator must not truncate the epoch"


class TestCloseAndTeardown:
    """§9.3 — teardown is bounded and does not run under the GIL."""

    def test_close_is_idempotent_and_terminal(self, scx_path):
        ds = pyscx.IndexPlanDataset(
            scx_path, normalize=False, log1p=False, obs_columns=[]
        )
        assert ds.closed is False
        assert ds.n_obs > 0

        ds.close()
        assert ds.closed is True
        ds.close()  # idempotent

        # Terminal, unlike TrainingDataset.close(): the tokio runtime is built
        # exactly once so a forked child can never inherit it, so it cannot be
        # rebuilt on a later __iter__.
        with pytest.raises(RuntimeError, match="closed"):
            ds.iter_with_plans(iter([[(0, 1)]]))
        with pytest.raises(RuntimeError, match="closed"):
            _ = ds.n_obs
        with pytest.raises(RuntimeError, match="closed"):
            ds.memory_budget()

    def test_repr_never_raises_when_closed(self, scx_path):
        """`repr` is what a debugger and a traceback call; it must not fail
        just because the object was closed."""
        ds = pyscx.IndexPlanDataset(
            scx_path, normalize=False, log1p=False, obs_columns=[]
        )
        assert "n_obs=" in repr(ds)
        ds.close()
        assert repr(ds) == "IndexPlanDataset(closed)"

    def test_close_with_a_live_iterator_does_not_raise(self, scx_path):
        """The iterator holds its own reference to the loader, so `close`
        cannot take sole ownership. It must release its own reference and
        leave the iterator's teardown to that object's own drop."""
        ds = pyscx.IndexPlanDataset(
            scx_path, normalize=False, log1p=False, obs_columns=[]
        )
        it = ds.iter_with_plans(iter([[(0, 1)], [(2, 3)]]), lookahead=2)
        next(it)
        ds.close()
        # The iterator was built from a live loader and keeps working.
        assert next(it)["X"].shape[0] == 1
        del it

    def test_close_then_drain_the_iterator(self, scx_path):
        """`ds.close()` first, *then* exhaust the iterator.

        The ordering the API supports and the one that used to be worst:
        `close()` cannot unwrap the loader while the iterator holds a
        reference, so exhaustion is what releases the runtime — and it happens
        inside `__next__`'s end-of-stream branch, which the iterator's `Drop`
        can never cover (it early-returns once `inner` is taken). The bound now
        lives in the Rust iterator's own `Drop`, so both paths get it.
        """
        import time

        ds = pyscx.IndexPlanDataset(
            scx_path, normalize=False, log1p=False, obs_columns=[]
        )
        it = ds.iter_with_plans(iter([[(0, 1)], [(2, 3)], [(4, 5)]]), lookahead=4)
        next(it)
        ds.close()

        t0 = time.monotonic()
        rest = list(it)  # EOS branch releases the last reference
        elapsed = time.monotonic() - t0

        assert len(rest) == 2, "closing the dataset must not truncate a live iterator"
        assert elapsed < 10.0, f"drain-after-close took {elapsed:.2f}s"
        # Iterating again is a clean StopIteration, not a crash.
        assert list(it) == []

    def test_teardown_mid_flight_is_bounded(self, scx_path):
        """Abandoning an epoch with prefetches outstanding must tear down
        cleanly and promptly.

        This is the ordering guard, not a GIL measurement. The iterator holds
        its own reference to the loader, so dropping the dataset first cannot
        take sole ownership; the runtime is released when the *iterator* drops.
        Getting that ordering wrong shows up here as a hang or a panic.

        Note on what is deliberately **not** asserted here: whether the teardown
        released the GIL, and whether it honoured the 5 s deadline. Neither is
        measurable at unit-test scale — teardown of even a 60000x3000 file with
        prefetches in flight was measured at ~9 ms, so a threshold that could
        see either property would be flaky and a stable one passes on broken
        code. Both are pinned in Rust instead, where the mechanism can be
        exercised directly: `runtime::tests::drop_abandons_a_task_that_overruns_the_deadline`
        proves the bound (60 s task, 200 ms deadline), and
        `the_iterator_as_last_owner_still_gets_a_bounded_teardown` plus the
        sparse `teardown_through_the_real_iter_ownership_graph_is_bounded`
        prove it fires on the ownership paths this test walks. GIL release
        remains carried by construction, mirroring `TrainingDataset::drop`.
        """
        import time

        ds = pyscx.IndexPlanDataset(
            scx_path, normalize=False, log1p=False, obs_columns=[]
        )
        plans = [[(i, i + 1)] for i in range(8)]
        it = ds.iter_with_plans(iter(plans), lookahead=4)
        next(it)  # prefetches for the following plans are now in flight

        t0 = time.monotonic()
        del ds  # not the last reference — `it` still holds one
        del it  # this is what actually releases the runtime
        elapsed = time.monotonic() - t0

        # The bound is `SHUTDOWN_DEADLINE` (5 s) plus slack; a hang or a
        # missing `shutdown_timeout` is what this catches, not latency.
        assert elapsed < 10.0, f"mid-flight teardown took {elapsed:.2f}s"


class TestGilDuringTeardown:
    """ORG-9.10-6: `IndexPlanDataset` teardown must release the GIL.

    `IndexPlanDataset::close` / `Drop` and `IndexPlanBatchIter::drop` all wrap
    the runtime teardown in `Python::attach(|py| py.detach(...))` precisely so
    the rest of the interpreter keeps running while already-started
    `spawn_blocking` shard decodes finish (§9.3). Nothing asserted it —
    `TestTeardownOrdering::test_teardown_mid_flight_is_bounded` says so in its
    own docstring, and it is right that a *duration* threshold cannot see the
    property: the whole teardown is single-digit milliseconds.

    What is measurable is whether another Python thread ran in the **leading
    half** of that window. Getting to an oracle that cannot false-pass took
    three tries, and the two rejected forms are recorded in
    `_leading_ticks` because both looked correct:

    1. a `len(ticks)` delta read around the window — racy on the far side;
    2. counting the whole `[t0, t1]` window by timestamp — still racy on the
       far side, because `t1` is recorded only after the destructor returns;
    3. counting the leading half — sound in the regime this test runs in, but
       *not* unconditionally: if the post-return scheduling delay exceeds the
       teardown itself, the midpoint moves past the return and post-teardown
       ticks re-enter the leading half.

    (3) is what this test asserts — over up to three teardown trials, because
    a ~0.3 ms window is a single scheduling opportunity and one tickless trial
    is weak evidence under load; a GIL-holding teardown scores zero on every
    trial, so retrying de-flakes only the pass side. Its residual is why the
    test also runs two controls at the measured duration, and skips (never
    fails) when either shows the rule is not discriminating on this host
    right now:

    - a **positive control** — a wall-clock-capped `hashlib.sha256` loop over
      a fixed 4 KiB buffer for the same duration, which releases the GIL by
      construction (buffers > 2 KiB) while burning a core like the teardown
      does, must score a leading tick. Zero means the ticker thread is not
      schedulable next to busy native work inside a window this short (a
      loaded 2-core CI runner starved it out of a 0.32 ms window entirely),
      and a starved ticker is indistinguishable from a held GIL. It must burn
      CPU rather than sleep (a sleeping control yields its core and gets ticks
      the CPU-busy teardown cannot), and it must be duration-capped rather
      than sized-to-duration (a slow teardown must fail this test, not OOM
      the host allocating a buffer proportional to seconds);
    - a **negative control** — `ctypes.PyDLL(None).usleep` of the same
      duration, which holds the GIL by construction, must score zero. A
      leading tick here means the leading-half rule itself is leaking.

    **The control narrows the residual; it does not eliminate it.** The two
    observations are separate scheduling trials, so a teardown trial that leaks
    and a control trial that does not would still pass. Measured on this host the
    leak is 1/200 at 200 us and 0/200 from 250 us up, and `MIN_MEASURABLE_S`
    keeps the test out of that regime — but that is a probability, not a proof.
    The sound close is to record start/end markers *inside* the native teardown
    and count only between them; that needs a test hook in the binding, so it is
    deferred rather than done here. Do not read a green here as a guarantee the
    GIL was released — read it as: it was released, or a 1-in-many scheduling
    coincidence occurred that the control did not catch.
    """

    N_OBS = 30000
    N_VARS = 800
    N_SHARDS = 4

    # Below this, the post-return scheduling delay is a large enough fraction of
    # the window that the leading-half rule starts to leak: a GIL-holding
    # control scored a leading tick in 1/200 runs at 200 us on this host, and
    # 0/200 at every duration from 250 us up.
    MIN_MEASURABLE_S = 3e-4

    @staticmethod
    def _leading_ticks(action, setup=None):
        """Run `action` with a GIL-hungry ticker thread alive, and report
        `(duration_s, leading, whole)` — ticks in the leading half of the window
        and in all of it.

        `setup` runs *after* the ticker is confirmed live and before the window
        opens; the prefetches have to be scheduled there, not before, or the
        ticker's own warmup is spent inside the very window being measured.

        Ticks are counted by **timestamp**, never as a `len(ticks)` delta taken
        around the window: between `t1` and the read the main thread can be
        preempted for a switch interval, during which the ticker appends
        thousands of entries. Measured that way, a teardown that held the GIL
        for its entire 1 ms scored ~1500 "ticks during".
        """
        import sys
        import threading
        import time

        ticks = []
        stop = threading.Event()

        def ticker():
            while not stop.is_set():
                ticks.append(time.perf_counter())

        old_interval = sys.getswitchinterval()
        worker = None
        try:
            sys.setswitchinterval(0.0001)
            worker = threading.Thread(target=ticker, daemon=True)
            worker.start()

            deadline = time.perf_counter() + 2.0
            while not ticks and time.perf_counter() < deadline:
                time.sleep(0.001)
            if not ticks:
                # A ticker that cannot run at all is an unmeasurable host, not
                # a GIL verdict — same policy as the two controls: skip, never
                # fail, when the instrument itself is starved.
                pytest.skip("premise: the ticker thread never ran in 2 s")

            if setup is not None:
                setup()

            t0 = time.perf_counter()
            action()
            t1 = time.perf_counter()
            # Let the ticker run past the window so `ticks[-1] > t1` can act as
            # a liveness premise: a ticker that died mid-window would otherwise
            # look exactly like a GIL that was never released. Bounded WAIT,
            # not a fixed sleep: under heavy contention the ticker can need far
            # more than a few ms to get a slice after the window closes, and a
            # missed fixed grace tripped this premise as a hard failure on
            # loaded 2-core runners (the retry loop above multiplied the
            # exposures). The GIL is released here — time.sleep — so only
            # genuine CPU starvation keeps the ticker off the core.
            deadline = time.perf_counter() + 2.0
            while ticks[-1] <= t1 and time.perf_counter() < deadline:
                time.sleep(0.001)
        finally:
            stop.set()
            if worker is not None:
                worker.join(timeout=5)
            sys.setswitchinterval(old_interval)

        if ticks[-1] <= t1:
            pytest.skip(
                "premise: the ticker did not outlive the window within 2 s — "
                "the instrument is starved on this host, so the window's tick "
                "count is not a GIL verdict"
            )
        midpoint = t0 + (t1 - t0) / 2
        return (
            t1 - t0,
            sum(1 for t in ticks if t0 <= t <= midpoint),
            sum(1 for t in ticks if t0 <= t <= t1),
        )

    def test_teardown_with_prefetches_in_flight_releases_the_gil(self, tmp_path):
        import ctypes

        import anndata as ad
        import scipy.sparse as sp

        path = str(tmp_path / "gil_teardown.scx")
        X = sp.random(
            self.N_OBS, self.N_VARS, density=0.05, format="csr", random_state=0
        )
        X.data = np.round(X.data * 10 + 1).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.obs["cell_id"] = [f"c{i}" for i in range(self.N_OBS)]
        # Unframed: a framed shard is block-index eligible, and the prefetch
        # deliberately skips warming those — so there would be no in-flight
        # decode to tear down and nothing to observe.
        pyscx.from_anndata(
            adata, path, shard_size=self.N_OBS // self.N_SHARDS, row_group_rows=0
        )

        # One plan per shard, so consuming the first still leaves N_SHARDS - 1
        # distinct `read_shard_cached_arc` tasks queued.
        step = self.N_OBS // self.N_SHARDS
        plans = [[(i * step, (i * step) + 1)] for i in range(self.N_SHARDS)]

        def measure_one_teardown():
            box = {
                "ds": pyscx.IndexPlanDataset(
                    path,
                    normalize=False,
                    log1p=False,
                    obs_columns=[],
                    cache_shards=8,
                    lookahead=self.N_SHARDS,
                    scatter_block_index=False,
                )
            }

            def schedule():
                box["it"] = box["ds"].iter_with_plans(
                    iter(plans), lookahead=self.N_SHARDS
                )
                # runtime built; the rest of the plans are prefetching
                next(box["it"])
                # Release the dataset's reference here, *inside* setup: the
                # iterator holds one of its own, so this tears nothing down —
                # but it has to happen before the window opens, or the iterator
                # drop below is not the last-reference drop and the window
                # measures nothing. (It measured 0.2 ms of nothing when this
                # line lived after the window.)
                box.pop("ds")

            def teardown():
                box.pop("it")  # the drop that tears the runtime down

            return self._leading_ticks(teardown, setup=schedule)

        # A ~0.3 ms window is a single scheduling opportunity, so one tickless
        # trial is weak evidence: under load the ticker misses any given short
        # window with non-trivial probability even with the GIL free (the
        # controls below catch most of that, but they are *separate* trials and
        # can get lucky where the teardown did not). A GIL-HOLDING teardown
        # scores zero on EVERY trial, so retrying only de-flakes the pass side:
        # break on the first trial with a leading tick; only three consecutive
        # tickless teardowns proceed to the controls and the assertion.
        for _ in range(3):
            teardown_s, leading, whole = measure_one_teardown()

            if teardown_s < self.MIN_MEASURABLE_S:
                pytest.skip(
                    f"teardown took {teardown_s * 1e3:.3f} ms — below the "
                    "window in which the leading-half rule is measurably "
                    "leak-free, so this run cannot distinguish a GIL-holding "
                    "teardown from a GIL-releasing one"
                )
            if leading > 0:
                break

        # Positive control: a GIL-RELEASING but CPU-CONSUMING call of the same
        # duration must score at least one leading tick. If it does not, the
        # ticker thread is simply not schedulable next to busy native work
        # inside a window this short on this host right now — a loaded 2-core
        # CI runner left a 0.32 ms teardown with zero ticks in the WHOLE window
        # even though the GIL was free throughout, which the assertion below
        # misread as "the GIL was held". Two properties are load-bearing:
        # - it must burn CPU, not sleep: `time.sleep` yields its core, so under
        #   load it gets ticks that the CPU-busy teardown cannot, and the
        #   control stops being load-equivalent (measured on 2 contended cores:
        #   a sleep control passed while a GIL-free teardown still scored 0);
        # - it must be a fixed small buffer in a wall-clock-capped loop, never
        #   a buffer *sized to the duration* — a slow-but-returning teardown is
        #   this test's most interesting failure, and sizing bytes to seconds
        #   turns it into a multi-GiB allocation before the assertion runs.
        # `hashlib.sha256` releases the GIL for buffers over 2 KiB; the loop
        # reacquires it between iterations, exactly like a teardown that
        # releases around its blocking waits. Duration is capped: past ~1 s of
        # burn, schedulability is long since proven either way. The buffer must
        # be barely above the 2 KiB release threshold, never larger: the GIL is
        # released only DURING each digest call, so the hash quantum sets the
        # control's granularity — a 1 MiB digest is ~0.7 ms/call, 2x the
        # 0.3 ms minimum window, making the control's release window longer
        # than the teardown's and un-load-equivalent again (4 KiB is ~3 us).
        # Only consulted when every teardown trial was tickless: when a trial
        # ticked, the assertion below passes on the measurement itself and a
        # coincidentally starved control must not convert that pass to a skip.
        if leading == 0:
            import hashlib
            import time as _time

            burn = bytes(1 << 12)
            burn_s = min(teardown_s, 1.0)

            def _burn_gil_free():
                end = _time.perf_counter() + burn_s
                while _time.perf_counter() < end:
                    hashlib.sha256(burn).digest()

            _, release_leading, _ = self._leading_ticks(_burn_gil_free)
            if release_leading == 0:
                pytest.skip(
                    f"positive control: a GIL-releasing, CPU-burning hash loop "
                    f"of ~{burn_s * 1e3:.2f} ms scored no leading tick — the "
                    "ticker is not schedulable at this duration on this host, "
                    "so a zero here cannot distinguish a held GIL from a "
                    "starved ticker"
                )

        # Negative control: hold the GIL for the same duration, by construction,
        # and apply the identical rule. It must score zero — if it does not, the
        # rule is leaking on this host right now and no verdict from it is worth
        # anything.
        libc = ctypes.PyDLL(None)  # PyDLL keeps the GIL across the call
        hold_us = max(int(teardown_s * 1e6), 1)
        _, control_leading, control_whole = self._leading_ticks(
            lambda: libc.usleep(hold_us)
        )
        if control_leading > 0:
            pytest.skip(
                f"negative control leaked {control_leading} tick(s) into the "
                f"leading half of a {hold_us} us GIL-holding call — the rule is "
                "not discriminating on this host at this duration, so a pass "
                "here would prove nothing"
            )

        assert leading > 0, (
            f"no other Python thread ran in the first half of a "
            f"{teardown_s * 1e3:.2f} ms teardown ({whole} tick(s) in the whole "
            "window, all in the tail) — the GIL was held for its duration and "
            f"released only on return. Same-duration GIL-holding control scored "
            f"{control_leading} leading / {control_whole} whole, as expected."
        )


class TestRowGroupCache:
    """OPT-FORMATIO-1: on a framed file the block-index gather retains the row
    groups it decodes, in the same LRU as whole shards, under `max_memory_mb`.
    A second batch over the same rows is served as `row_group_hits`; the
    whole-shard counters stay at zero on this path; the L2 prefetcher
    pre-decodes a plan's groups when they fit `budget / (lookahead + 1)`."""

    @staticmethod
    def _dataset(path, **kw):
        kw.setdefault("normalize", False)
        kw.setdefault("cache_shards", 4)
        kw.setdefault("sort_by_shard", True)
        kw.setdefault("scatter_block_index", True)
        return pyscx.IndexPlanDataset(path, **kw)

    @staticmethod
    def _plan(seed=0, n=8):
        rng = np.random.default_rng(seed)
        return [(int(rng.integers(0, 400)), int(rng.integers(0, 400))) for _ in range(n)]

    def test_second_plan_is_served_from_the_row_group_lru(self, tmp_path):
        path = str(tmp_path / "framed_rg.scx")
        TestBlockIndexAdoption._write_framed(path)
        ds = self._dataset(path, lookahead=0)
        plan = self._plan()
        batches = list(ds.iter_with_plans(iter([list(plan), list(plan)]), lookahead=0))
        assert len(batches) == 2
        assert np.array_equal(batches[0]["X"], batches[1]["X"]), (
            "the cached groups must reproduce the decoded ones exactly"
        )
        cm = ds.cache_metrics()
        assert cm["block_index_groups"] > 0 and cm["full_shard_groups"] == 0, cm
        assert cm["row_group_misses"] > 0, "premise: the first plan decoded groups"
        assert cm["row_group_hits"] == cm["row_group_misses"], (
            "the second identical plan must hit every group the first decoded: "
            f"{cm}"
        )
        assert cm["hits"] + cm["misses"] == 0, (
            "the row-group path never touches the whole-shard counters"
        )
        assert cm["row_group_bytes_inserted"] > 0
        assert cm["peak_bytes_in_cache"] <= (
            cm["bytes_inserted"] + cm["row_group_bytes_inserted"]
        )

    def test_lookahead_pre_decodes_the_plan_row_groups(self, tmp_path):
        """One plan, `lookahead=1`: the prefetcher warms the plan's groups (they
        fit their budget share by a wide margin here) and the gather hits them.
        The same single plan at `lookahead=0` misses them all — a second
        identical plan would hit through L1 either way, so the arm is one plan."""
        path = str(tmp_path / "framed_rg_l2.scx")
        TestBlockIndexAdoption._write_framed(path)
        plan = self._plan(seed=3)

        ds = self._dataset(path, lookahead=1)
        it = ds.iter_with_plans(iter([list(plan)]), lookahead=1)
        list(it)
        pm = it.metrics()["prefetch"]
        cm = ds.cache_metrics()
        assert pm["prefetch_skipped_block_index"] > 0, pm
        assert pm["prefetch_tasks_spawned"] == 0, (
            "a row-group warm is not a whole-shard spawn"
        )
        assert cm["row_group_misses"] > 0, cm
        assert cm["row_group_hits"] == cm["row_group_misses"], (
            f"the gather must find every group the prefetcher decoded: {cm}"
        )

        ds0 = self._dataset(path, lookahead=0)
        list(ds0.iter_with_plans(iter([list(plan)]), lookahead=0))
        cm0 = ds0.cache_metrics()
        assert cm0["row_group_misses"] > 0
        assert cm0["row_group_hits"] == 0, f"no prefetch ran at lookahead=0: {cm0}"

    def test_kill_switch_disables_retention(self, tmp_path):
        """`SCX_ROW_GROUP_CACHE=0` (read once per process, hence a subprocess)
        keeps the block-index route but retains nothing: every `row_group_*`
        counter stays 0 across two identical plans."""
        import json
        import os
        import subprocess
        import sys
        import textwrap

        path = str(tmp_path / "framed_rg_off.scx")
        TestBlockIndexAdoption._write_framed(path)
        plan = self._plan(seed=5)
        src = textwrap.dedent(
            f"""
            import json, pyscx
            ds = pyscx.IndexPlanDataset({path!r}, normalize=False, cache_shards=4,
                                        sort_by_shard=True, lookahead=0,
                                        scatter_block_index=True)
            plan = {plan!r}
            list(ds.iter_with_plans(iter([list(plan), list(plan)]), lookahead=0))
            print(json.dumps(ds.cache_metrics()))
            """
        )
        env = {**os.environ, "SCX_ROW_GROUP_CACHE": "0"}
        r = subprocess.run(
            [sys.executable, "-c", src], capture_output=True, text=True, env=env
        )
        assert r.returncode == 0, r.stderr
        cm = json.loads(r.stdout.strip().splitlines()[-1])
        assert cm["block_index_groups"] > 0, cm
        assert cm["row_group_hits"] == 0 and cm["row_group_misses"] == 0, cm
        assert cm["row_group_bytes_inserted"] == 0, cm
