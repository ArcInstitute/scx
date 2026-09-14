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
    never inserts a whole shard — the row groups it retains report through the
    `row_group_*` counters, not `hits`/`misses`/`evictions` — so a framed
    fixture cannot thrash the whole-shard counters these tests read and they
    would pass vacuously.
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


@pytest.fixture
def bigshard_path(tmp_path):
    """Unframed fixture with LARGE shards: 200 cells x 2000 genes, dense, 10 shards.

    ~320 KB decoded per shard, so a 1 MB byte budget affords ~3 entries while a
    scattered plan touches all 10 — the only shape in which the **byte** cap is
    the binding constraint. The small `multishard_path` fixture cannot express
    this: at ~6 KB/shard even a 1 MB budget holds the entire file, so nothing
    thrashes however high `cache_shards` is set. (Round-2 review caught a test
    that missed exactly this and asserted nothing as a result.)
    """
    import anndata

    rng = np.random.default_rng(11)
    n_obs, n_vars = 200, 2000
    dense = rng.integers(1, 50, size=(n_obs, n_vars)).astype(np.float32)
    obs = pd.DataFrame(
        {"cell_id": [f"cell_{i}" for i in range(n_obs)]},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    adata = anndata.AnnData(X=sp.csr_matrix(dense), obs=obs, var=var)
    path = str(tmp_path / "bigshard.scx")
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
        # One plan, in the shape `iter_with_plans` consumes — role_tags and
        # set_offsets are ignored, so they only have to be well-formed.
        assert ds.suggested_cache_shards(([0, 0], [0, 5], [0, 0], [0, 2])) == 1
        assert ds.suggested_cache_shards(([0, 1], [0, 0], [0, 0], [0, 2])) == 2
        assert (
            ds.suggested_cache_shards(
                ([0, 0, 1, 1], [0, 199, 0, 199], [0] * 4, [0, 4])
            )
            == 4
        )
        assert ds.suggested_cache_shards(([], [], [], [0])) == 0

    def test_sparse_cellset_rejects_mismatched_lengths(self, multishard_path):
        """A four-tuple can still be ragged, so the guard outlives the arity
        change that folded the two arrays into one plan argument."""
        ds = pyscx.SparseCellSetDataset([multishard_path])
        with pytest.raises(ValueError, match="same length"):
            ds.suggested_cache_shards(([0, 0], [0], [0, 0], [0, 2]))


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
        assert budget["effective_cache_shards"] == 128

    def test_index_plan_none_budget_reports_the_resolved_value(self, multishard_path):
        ds = pyscx.IndexPlanDataset(multishard_path, scatter_block_index=False)
        budget = ds.memory_budget()
        assert budget["max_memory_mb"] >= 512, "the 512 MB floor still applies"
        assert budget["effective_cache_shards"] == 128, (
            "the adaptive budget must honour the requested cache on a small file"
        )

    def test_memory_budget_reports_the_binding_constraint(self, multishard_path):
        """`effective_cache_shards` must reflect the BYTE cap when it binds.

        Round-1 review (Cursor, P2): diagnostics on the sparse path were phrased
        against the requested `cache_shards` while the byte budget is what caps
        residency. On the regime this loader targets (~470 MB shards, adaptive
        4 GB) the byte cap binds long before the count does, so naming
        `cache_shards` sends the caller to raise a knob that cannot help.
        """
        budget_mb = 64
        ds = pyscx.SparseCellSetDataset(
            [multishard_path], cache_shards=4096, max_memory_mb=budget_mb
        )
        b = ds.memory_budget()
        assert b["cache_shards"] == 4096, "the request is reported unchanged"
        assert b["effective_cache_shards"] < 4096, (
            f"a {budget_mb} MB budget cannot afford 4096 shards"
        )
        # And it must be derived from the byte budget, not invented.
        #
        # ORG-9.10-5 rewrote this expectation. It used to read
        # `min(4096, 1 MiB // shard_decoded_bytes)` under `max_memory_mb=1`:
        # arithmetic that only held while the tuner ignored the 50 MB
        # interpreter constant the very same dict reported, and which under a
        # process-budget reading means "no cache at all". The tuner now charges
        # the non-cache terms it reports, so the cache gets what is left after
        # them. (That bounds the cache this loader sizes, not process RSS — the
        # gathered batch is uncharged and the LRU keeps one oversize shard.)
        non_cache = b["breakdown"]["total_bytes"] - b["breakdown"]["cache_bytes"]
        expected = min(
            4096, (budget_mb * 1024 * 1024 - non_cache) // b["shard_decoded_bytes"]
        )
        assert b["effective_cache_shards"] == expected
        assert not b["budget_exceeded"]
        assert b["breakdown"]["total_bytes"] <= budget_mb * 1024 * 1024, (
            "the report must fit the budget the tuner checked"
        )

    def test_explicit_budget_is_respected(self, multishard_path):
        ds = pyscx.SparseCellSetDataset([multishard_path], max_memory_mb=64)
        assert ds.memory_budget()["max_memory_mb"] == 64


_BREAKDOWN_KEYS = {
    "cache_bytes",
    "batch_buffer_bytes",
    "lookahead_overhead_bytes",
    "transient_bytes",
    "python_overhead_bytes",
    "total_bytes",
}


class TestMemoryBudgetEnvelope:
    """Every class that reports a budget reports it in the same envelope.

    ORG-9.10-4: the three `memory_budget()` dicts had an **empty** three-way
    top-level key intersection. `TrainingDataset` nested the six
    `BudgetBreakdown` components under `breakdown`, `IndexPlanDataset` spread
    the same six across the top level, and `SparseCellSetDataset` reported four
    cache-only keys and no breakdown at all — so no caller could read
    `total_bytes` off an arbitrary dataset, and the divergence was invisible
    because nothing tested more than one class's shape.

    Class-specific keys stay class-specific; it is the *breakdown* that must be
    one thing.
    """

    def _datasets(self, multishard_path):
        return {
            "TrainingDataset": pyscx.TrainingDataset(multishard_path),
            "IndexPlanDataset": pyscx.IndexPlanDataset(
                multishard_path, scatter_block_index=False
            ),
            "SparseCellSetDataset": pyscx.SparseCellSetDataset([multishard_path]),
        }

    def test_every_budget_carries_the_same_breakdown(self, multishard_path):
        for name, ds in self._datasets(multishard_path).items():
            b = ds.memory_budget()
            assert "breakdown" in b, (
                f"{name}.memory_budget() has no 'breakdown' key: {sorted(b)}"
            )
            assert set(b["breakdown"]) == _BREAKDOWN_KEYS, (
                f"{name} breakdown keys: {sorted(b['breakdown'])}"
            )

    def test_breakdown_components_sum_to_total(self, multishard_path):
        for name, ds in self._datasets(multishard_path).items():
            bd = ds.memory_budget()["breakdown"]
            components = sum(
                bd[k] for k in _BREAKDOWN_KEYS if k != "total_bytes"
            )
            assert components == bd["total_bytes"], name
            assert all(isinstance(v, int) and v >= 0 for v in bd.values()), name

    def test_python_overhead_is_counted_on_every_path(self, multishard_path):
        """The 50 MB interpreter/numpy/Arrow constant is not path-specific.

        The sparse loader omitted it entirely — it never built a breakdown — so
        its reported budget was the only one that pretended the interpreter was
        free.
        """
        for name, ds in self._datasets(multishard_path).items():
            bd = ds.memory_budget()["breakdown"]
            assert bd["python_overhead_bytes"] > 0, name


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
        assert "1-entry cache" in msg, f"must name the effective cache size: {msg}"
        # Count-bound (an explicit cache_shards=1 with a generous budget), so the
        # advice must lead with the count knob and name the sizing helper.
        assert "Try cache_shards>=" in msg, f"must lead with the count knob: {msg}"
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
        self, bigshard_path
    ):
        """When the byte cap binds, the warning must quote the affordable count.

        Round-1 review (Cursor, P2). A high `cache_shards` with a tight
        `max_memory_mb` must not produce "exceeds cache_shards=4096" — that is the
        knob the caller already set high, and raising it further cannot help.
        """
        ds = pyscx.SparseCellSetDataset(
            [bigshard_path], cache_shards=4096, max_memory_mb=1
        )
        affordable = ds.memory_budget()["effective_cache_shards"]
        n_shards = pyscx.open(bigshard_path).shard_count
        assert affordable < n_shards, (
            f"premise: the byte cap must afford fewer ({affordable}) than the "
            f"plan touches ({n_shards}), else nothing thrashes"
        )

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
        # Unconditional. Round-2 review (Cursor, P2) caught that this was
        # originally guarded by `if msgs:` — which made the whole assertion
        # vacuous: a run that never warned passed silently, so the test could not
        # fail for the reason it exists. If this config stops thrashing, the test
        # must break loudly and be re-pointed, not quietly stop checking.
        assert msgs, (
            f"premise: a {n_shards}-shard working set against a {affordable}-entry "
            f"cache must thrash and warn"
        )
        m = msgs[0]
        # The cache size named must be the affordable count, never the request.
        assert f"{affordable}-entry cache" in m, (
            f"must quote the affordable count {affordable}, not the request 4096: {m}"
        )
        assert "4096-entry" not in m
        # And the advice must lead with the knob that can actually fix it
        # (round-2 review, Cursor P3): raising `cache_shards` cannot help a
        # byte-bound cache, so it must not be the primary suggestion.
        assert "BYTE budget is the limiter" in m, f"must lead with the budget: {m}"
        assert "raise max_memory_mb to >=" in m, f"must name a concrete budget: {m}"
        assert "Try cache_shards>=" not in m, (
            f"must NOT lead with the count knob when bytes bind: {m}"
        )

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


class TestSparseCellSetBatchCharge:
    """`max_plan_rows` charges one gathered batch against the byte budget.

    The class has no `max_plan_size`: a plan's row count is the caller's, so
    only the caller can say what a batch costs. That is why the charge is
    opt-in — the test below that the default is *unchanged* is as load-bearing
    as the one that the charge works, since a guessed default would shrink the
    shard cache on every existing dataset, and cache size is the lever worth
    2,486x on a file whose working set does not fit.
    """

    def test_default_reports_no_batch_term(self, multishard_path):
        ds = pyscx.SparseCellSetDataset([multishard_path])
        try:
            b = ds.memory_budget()
            assert b["max_plan_rows"] is None
            assert b["breakdown"]["batch_buffer_bytes"] == 0
            assert b["mean_nnz_per_row"] > 0, (
                "the density is reported whether or not it is charged; 0 would "
                "mean no shard carried catalog stats"
            )
        finally:
            ds.close()

    def test_declaring_plan_width_charges_a_batch_and_shrinks_the_cache(
        self, multishard_path
    ):
        budget_mb = 64
        plain = pyscx.SparseCellSetDataset([multishard_path], max_memory_mb=budget_mb)
        wide = pyscx.SparseCellSetDataset(
            [multishard_path], max_memory_mb=budget_mb, max_plan_rows=100_000
        )
        try:
            pb, wb = plain.memory_budget(), wide.memory_budget()
            assert wb["max_plan_rows"] == 100_000
            assert wb["breakdown"]["batch_buffer_bytes"] > 0
            assert pb["breakdown"]["batch_buffer_bytes"] == 0
            # The charge comes out of the cache, not out of thin air.
            assert (
                wb["effective_cache_shards"] <= pb["effective_cache_shards"]
            ), f"{wb['effective_cache_shards']} vs {pb['effective_cache_shards']}"
            # And the six-key breakdown still sums, with the new term included.
            bd = wb["breakdown"]
            assert (
                sum(v for k, v in bd.items() if k != "total_bytes")
                == bd["total_bytes"]
            )
        finally:
            plain.close()
            wide.close()

    def test_blocking_thread_cap_is_reported_and_bounded(self, multishard_path):
        """`lookahead` bounds in-flight plans, not the tasks a plan spawns.

        Each plan issues one `spawn_blocking` per distinct `(file, shard)` it
        touches — caller-controlled through plan width, and 48+ on the shapes
        `suggested_cache_shards` was written for. Left at tokio's default the
        ceiling was 512 simultaneous shard decodes, which for a census-sized
        shard bounds nothing useful.
        """
        import os

        # Pin the arithmetic, not a range. `2 <= cap < 512` passed for every
        # intended value AND for several wrong ones, and it was environment-
        # dependent: `SCX_LOADER_CPU_THREADS` is documented as able to exceed
        # the decode pool's default clamp, so a large override would have made
        # the old assertion fail for the wrong reason.
        lookahead = 3
        ds = pyscx.SparseCellSetDataset([multishard_path], lookahead=lookahead)
        ip = pyscx.IndexPlanDataset(multishard_path, scatter_block_index=False)
        try:
            cap = ds.memory_budget()["max_blocking_threads"]
            override = os.environ.get("SCX_LOADER_CPU_THREADS")
            pool = int(override) if override and override.isdigit() else min(
                os.cpu_count() or 1, 8
            )
            assert cap == max(2, min(pool + lookahead, 512)), (
                f"cap {cap} != clamp(pool {pool} + lookahead {lookahead})"
            )
            # The cap lives on the shared PrefetchEngine, so IndexPlanDataset is
            # subject to it too and must report it.
            assert "max_blocking_threads" in ip.memory_budget()
        finally:
            ds.close()
            ip.close()
