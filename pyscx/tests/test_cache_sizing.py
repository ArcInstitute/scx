"""Cache-thrash diagnostic + budget reconciliation (data-load Phase 1, 1A).

STATE3's 143 s/batch pathology was a working set larger than `cache_shards`, and
it took a bisection to find because nothing interpreted `cache_metrics()`. These
tests pin the three pieces that make it self-diagnosing:

1. `suggested_cache_shards` — how many shards a plan actually touches.
2. A construction-time `UserWarning` when the budget cannot afford the cache.
3. A runtime `UserWarning` when observed counters show thrash.

Each has a **negative** case, because a diagnostic that always fires is worse
than none: it trains operators to filter the warning.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import pyscx


@pytest.fixture
def multishard_path(tmp_path):
    """Unframed multi-shard fixture: 200 cells x 40 genes over 10 obs/CSR shards.

    **Unframed is load-bearing.** Framing (the default) routes a scattered gather
    through the block-index path, which decodes only the touched row-groups and
    never populates the whole-shard LRU — so a framed fixture cannot thrash and
    these tests would pass vacuously.
    """
    import anndata

    rng = np.random.default_rng(7)
    n_obs, n_vars = 200, 40
    dense = rng.integers(0, 50, size=(n_obs, n_vars)).astype(np.float32)
    obs = pd.DataFrame(
        {"cell_id": [f"cell_{i}" for i in range(n_obs)]},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    adata = anndata.AnnData(X=sp.csr_matrix(dense), obs=obs, var=var)
    path = str(tmp_path / "multishard.scx")
    pyscx.from_anndata(adata, path, shard_size=20, row_group_rows=0)
    return path


class TestSuggestedCacheShards:
    """The measurement STATE3 lacked: how many shards does this plan touch?"""

    def test_index_plan_counts_distinct_shards(self, multishard_path):
        ds = pyscx.IndexPlanDataset(multishard_path, scatter_block_index=False)
        # 200 rows / 20 per shard => shard s covers rows [20s, 20s+20).
        assert ds.suggested_cache_shards([(0, 5)]) == 1, "both rows in shard 0"
        assert ds.suggested_cache_shards([(0, 199)]) == 2, "shards 0 and 9"
        # Duplicates across pairs collapse to the same shard set.
        assert ds.suggested_cache_shards([(0, 199), (1, 198), (2, 197)]) == 2
        assert ds.suggested_cache_shards([]) == 0

    def test_index_plan_scattered_plan_needs_more_than_a_clustered_one(
        self, multishard_path
    ):
        """The comparison that makes the helper useful, not just correct."""
        ds = pyscx.IndexPlanDataset(multishard_path, scatter_block_index=False)
        clustered = [(i, i + 1) for i in range(0, 18, 2)]  # all within shard 0
        scattered = [(i * 20, i * 20 + 1) for i in range(10)]  # one per shard

        assert ds.suggested_cache_shards(clustered) == 1
        assert ds.suggested_cache_shards(scattered) == 10
        assert ds.suggested_cache_shards(scattered) > ds.suggested_cache_shards(
            clustered
        ), "a scattered plan must demand a bigger cache than a clustered one"

    def test_sparse_cellset_counts_per_file(self, multishard_path, tmp_path):
        ds = pyscx.SparseCellSetDataset([multishard_path, multishard_path])
        # Same shard index in two files = two distinct cache entries, because
        # the shared cache is keyed (file_id, shard).
        assert ds.suggested_cache_shards([0, 0], [0, 5]) == 1
        assert ds.suggested_cache_shards([0, 1], [0, 0]) == 2
        assert ds.suggested_cache_shards([0, 0, 1, 1], [0, 199, 0, 199]) == 4
        assert ds.suggested_cache_shards([], []) == 0

    def test_sparse_cellset_rejects_mismatched_lengths(self, multishard_path):
        ds = pyscx.SparseCellSetDataset([multishard_path])
        with pytest.raises(ValueError, match="same length"):
            ds.suggested_cache_shards([0, 0], [0])


class TestBudgetReconciliation:
    """`max_memory_mb=None` must be bounded-and-adaptive on both gather loaders."""

    def test_sparse_cellset_none_budget_is_bounded(self, multishard_path):
        ds = pyscx.SparseCellSetDataset([multishard_path])
        budget = ds.memory_budget()
        # The historical default was usize::MAX — unbounded RSS.
        assert budget["max_memory_mb"] > 0
        assert budget["max_memory_mb"] < 1 << 40, (
            f"budget must be bounded, got {budget['max_memory_mb']} MB"
        )
        assert budget["cache_shards"] == 128, "documented default"
        # A tiny fixture must not be tightened below the floor.
        assert budget["affordable_cache_shards"] == 128

    def test_index_plan_none_budget_reports_the_resolved_value(self, multishard_path):
        ds = pyscx.IndexPlanDataset(multishard_path, scatter_block_index=False)
        budget = ds.memory_budget()
        assert budget["max_memory_mb"] >= 512, "the 512 MB floor still applies"
        assert budget["effective_cache_shards"] == 128, (
            "the adaptive budget must honour the requested cache on a small file"
        )

    def test_memory_budget_reports_the_binding_constraint(self, multishard_path):
        """`affordable_cache_shards` must reflect the BYTE cap when it binds.

        Round-1 review (Cursor, P2): diagnostics on the sparse path were phrased
        against the requested `cache_shards` while the byte budget is what caps
        residency. On the regime this loader targets (~470 MB shards, adaptive
        4 GB) the byte cap binds long before the count does, so naming
        `cache_shards` sends the caller to raise a knob that cannot help.
        """
        ds = pyscx.SparseCellSetDataset(
            [multishard_path], cache_shards=4096, max_memory_mb=1
        )
        b = ds.memory_budget()
        assert b["cache_shards"] == 4096, "the request is reported unchanged"
        assert b["affordable_cache_shards"] < 4096, (
            "a 1 MB budget cannot afford 4096 shards"
        )
        # And it must be derived from the byte budget, not invented.
        expected = min(4096, (1 * 1024 * 1024) // b["shard_decoded_bytes"])
        assert b["affordable_cache_shards"] == expected

    def test_explicit_budget_is_respected(self, multishard_path):
        ds = pyscx.SparseCellSetDataset([multishard_path], max_memory_mb=64)
        assert ds.memory_budget()["max_memory_mb"] == 64


class TestCacheSizingWarning:
    """Construction-time warning when the budget cannot afford the cache."""

    def test_tight_budget_warns_and_names_the_fix(self, multishard_path):
        with pytest.warns(UserWarning, match="cache_shards") as rec:
            # Shards here are ~1.5 KB, so a 1 MB budget affords far fewer than 128.
            pyscx.SparseCellSetDataset(
                [multishard_path], cache_shards=4096, max_memory_mb=1
            )
        msg = str(rec[0].message)
        assert "SparseCellSetDataset" in msg, msg
        assert "4096" in msg, f"must name what was requested: {msg}"
        assert "max_memory_mb>=" in msg, f"must name the fix: {msg}"

    def test_generous_budget_is_silent(self, multishard_path):
        """The anti-tautology partner. Without it, an always-firing warning passes."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            pyscx.SparseCellSetDataset([multishard_path], cache_shards=8)
            pyscx.SparseCellSetDataset([multishard_path])  # adaptive default
            pyscx.IndexPlanDataset(multishard_path, scatter_block_index=False)

    def test_below_floor_escalates_the_message(self, multishard_path):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            # 1 byte affords 0 shards — comfortably below MIN_CACHE_SHARDS.
            pyscx.SparseCellSetDataset(
                [multishard_path], cache_shards=128, max_memory_mb=0
            )
        msgs = [str(w.message) for w in rec if w.category is UserWarning]
        assert msgs, "a zero budget must warn"
        assert any("re-decode" in m for m in msgs), (
            f"below the floor the message must explain the consequence: {msgs}"
        )


class TestThrashWarning:
    """Runtime warning from observed cache counters."""

    def _scattered_plans(self, n_batches, rows_per_batch=40):
        """Plans that touch every shard each batch, so a small cache thrashes."""
        rng = np.random.default_rng(3)
        return [
            [
                (int(r), int(r))
                for r in rng.choice(200, size=rows_per_batch, replace=False)
            ]
            for _ in range(n_batches)
        ]

    def test_thrash_warns_once_with_a_tiny_cache(self, multishard_path):
        ds = pyscx.IndexPlanDataset(
            multishard_path,
            cache_shards=1,
            max_memory_mb=512,
            lookahead=0,
            scatter_block_index=False,
        )
        plans = self._scattered_plans(64)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            for _ in ds.iter_with_plans(iter(plans)):
                pass
        thrash = [
            w for w in rec if "shard-cache thrash" in str(w.message)
        ]
        assert len(thrash) == 1, (
            f"must warn exactly once per iterator, got {len(thrash)}: "
            f"{[str(w.message) for w in thrash]}"
        )
        msg = str(thrash[0].message)
        assert "cache_shards=1" in msg, msg
        assert "suggested_cache_shards" in msg, f"must name the tool: {msg}"

    def test_well_sized_cache_does_not_warn(self, multishard_path):
        """The load-bearing negative case.

        Same file, same plans, a cache big enough to hold every shard. If this
        fired, the diagnostic would be noise on every healthy run — which is
        exactly how a warning gets filtered and stops working.
        """
        ds = pyscx.IndexPlanDataset(
            multishard_path,
            cache_shards=128,
            max_memory_mb=512,
            lookahead=0,
            scatter_block_index=False,
        )
        plans = self._scattered_plans(64)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            for _ in ds.iter_with_plans(iter(plans)):
                pass
        thrash = [w for w in rec if "shard-cache thrash" in str(w.message)]
        assert not thrash, (
            "a cache that holds the whole working set must stay silent: "
            f"{[str(w.message) for w in thrash]}"
        )

    def test_warns_within_a_short_run(self, multishard_path):
        """30 batches must be enough — the benchmark's own workload is 30.

        Regression for a real defect: the sampler originally checked every 32
        batches, while `cellset_gather` runs 30 for census-scale files. The
        diagnostic was silently dead on exactly the workload it was built for,
        and a 4.5-hour capture of a genuinely thrashing run emitted **zero**
        warnings. A cadence must be validated against the batch counts actually
        in play, not chosen to look conservative.
        """
        ds = pyscx.IndexPlanDataset(
            multishard_path,
            cache_shards=1,
            max_memory_mb=512,
            lookahead=0,
            scatter_block_index=False,
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            for _ in ds.iter_with_plans(iter(self._scattered_plans(30))):
                pass
        thrash = [w for w in rec if "shard-cache thrash" in str(w.message)]
        assert len(thrash) == 1, (
            f"a 30-batch thrashing run must warn exactly once, got {len(thrash)}"
        )

    def test_suggestion_never_exceeds_the_files_shard_count(self, multishard_path):
        """The suggested `cache_shards` must be a number that can help.

        Regression for a real defect: the suggestion was derived from shard
        *requests* per batch, which overcounts badly under thrash (the same shard
        is re-requested after every eviction). On a 31-shard census file it
        advised `cache_shards>=174`. Caching more entries than the file has is
        meaningless, so the estimate is capped at the shard count — but it must
        still exceed the current setting, or the advice is a no-op.
        """
        n_shards = pyscx.open(multishard_path).shard_count
        ds = pyscx.IndexPlanDataset(
            multishard_path,
            cache_shards=1,
            max_memory_mb=512,
            lookahead=0,
            scatter_block_index=False,
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            for _ in ds.iter_with_plans(iter(self._scattered_plans(32))):
                pass
        msgs = [str(w.message) for w in rec if "shard-cache thrash" in str(w.message)]
        assert msgs, "premise: the run must thrash and warn"
        m = re.search(r"cache_shards>=(\d+)", msgs[0])
        assert m, f"the message must name a concrete target: {msgs[0]}"
        suggested = int(m.group(1))
        assert 1 < suggested <= n_shards, (
            f"suggestion {suggested} must be actionable and <= the file's "
            f"{n_shards} shards"
        )

    def test_message_names_the_byte_bound_count_not_the_request(
        self, multishard_path
    ):
        """When the byte cap binds, the warning must quote the affordable count.

        Round-1 review (Cursor, P2). A high `cache_shards` with a tight
        `max_memory_mb` must not produce "exceeds cache_shards=4096" — that is the
        knob the caller already set high, and raising it further cannot help.
        """
        ds = pyscx.SparseCellSetDataset(
            [multishard_path], cache_shards=4096, max_memory_mb=1
        )
        affordable = ds.memory_budget()["affordable_cache_shards"]
        assert affordable < 4096, "premise: this config must be byte-bound"

        plans = [
            (
                np.zeros(40, dtype=np.uint32),
                np.asarray(r, dtype=np.uint64),
                np.zeros(40, dtype=np.int32),
                np.array([0, 40], dtype=np.int64),
            )
            for r in [
                np.random.default_rng(i).choice(200, size=40, replace=False)
                for i in range(32)
            ]
        ]
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            for _ in ds.iter_with_plans(iter(plans)):
                pass
        msgs = [str(w.message) for w in rec if "shard-cache thrash" in str(w.message)]
        if msgs:
            assert f"cache_shards={affordable}" in msgs[0], (
                f"must quote the affordable count {affordable}, not the "
                f"request 4096: {msgs[0]}"
            )
            assert "cache_shards=4096" not in msgs[0]

    def test_short_run_below_the_warmup_floor_is_silent(self, multishard_path):
        """A cold cache is not thrash, and a few batches are not evidence."""
        ds = pyscx.IndexPlanDataset(
            multishard_path,
            cache_shards=1,
            max_memory_mb=512,
            lookahead=0,
            scatter_block_index=False,
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            for _ in ds.iter_with_plans(iter(self._scattered_plans(2))):
                pass
        assert not [w for w in rec if "shard-cache thrash" in str(w.message)]

    def test_thrash_warning_does_not_break_iteration(self, multishard_path):
        """A warning promoted to an error must not corrupt the training loop.

        `simplefilter("error")` makes `warnings.warn` raise. That is the caller's
        choice for warnings; it must not surface as a failed `__next__` and lose
        batches.
        """
        ds = pyscx.IndexPlanDataset(
            multishard_path,
            cache_shards=1,
            max_memory_mb=512,
            lookahead=0,
            scatter_block_index=False,
        )
        plans = self._scattered_plans(64)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            n = sum(1 for _ in ds.iter_with_plans(iter(plans)))
        assert n == len(plans), f"expected {len(plans)} batches, got {n}"

    def test_documented_suppression_filter_actually_suppresses(self, multishard_path):
        """The warning tells the user how to silence it; that must be true.

        A suppression recipe printed in user-facing text is a claim, and a wrong
        one is worse than none — the user pastes it, sees no change, and loses
        trust in the whole diagnostic. `filterwarnings(message=...)` anchors its
        regex at the start of the message with `re.match`, which is exactly the
        sort of detail that makes a hand-written recipe silently not match.
        """
        plans = self._scattered_plans(64)

        def run(apply_filter: bool) -> int:
            ds = pyscx.IndexPlanDataset(
                multishard_path,
                cache_shards=1,
                max_memory_mb=512,
                lookahead=0,
                scatter_block_index=False,
            )
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                if apply_filter:
                    # Verbatim from the message emitted by `warn_cache_thrash`.
                    warnings.filterwarnings(
                        "ignore", message=".*shard-cache thrash.*"
                    )
                for _ in ds.iter_with_plans(iter(plans)):
                    pass
            return len([w for w in rec if "shard-cache thrash" in str(w.message)])

        assert run(False) == 1, "premise: the warning must fire when unfiltered"
        assert run(True) == 0, (
            "the suppression recipe printed in the warning must actually work"
        )
