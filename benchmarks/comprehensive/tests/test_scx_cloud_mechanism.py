"""Regression tests for ScxRunner's cloud ``native_mechanism`` tags.

The reporting layer groups cloud rows by ``native_mechanism`` so pull-and-
filter timings never share a cell with native-pushdown timings from TileDB
or SLAF. Keep these tags distinct so the report stays apples-to-apples.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.comprehensive.queries import GtPredicate
from benchmarks.comprehensive.runners.base import TimingResult
from benchmarks.comprehensive.runners.scx_runner import ScxRunner


class _FakePyscxPush:
    """Stand-in for ``pyscx`` used by ``read_cloud_filtered_query``.

    The real method pulls the fixture then delegates to
    ``self.read_filtered_query(local, predicate)`` for the actual filter.
    We patch ``pyscx.pull`` to a no-op and ``ScxRunner.read_filtered_query``
    to a synthetic ``TimingResult`` so the test stays hermetic — no network,
    no on-disk SCX file.
    """


def test_read_cloud_filtered_query_mechanism_tag(tmp_path: Path) -> None:
    runner = ScxRunner()
    predicate = GtPredicate(column="n_counts", threshold=1000, name="n_counts_gt_1000")

    def fake_filtered(self, path, pred):
        return TimingResult(wall_s=0.5, extra={"predicate": pred.name})

    # Patch the check-import guard (pyscx may not be importable in CI)
    # and the inner call. The patched ``pyscx.pull`` target covers BOTH
    # the ``import pyscx`` statement inside the method body AND the
    # module-level fallback path.
    with patch.object(ScxRunner, "_check_pyscx", lambda self: None), \
         patch.object(ScxRunner, "read_filtered_query", fake_filtered), \
         patch("benchmarks.comprehensive.runners.scx_runner.pyscx", create=True) as fake_pyscx:
        fake_pyscx.pull = lambda url, local: None

        result = runner.read_cloud_filtered_query(
            "gs://dummy/fixture.scxd/", predicate,
        )

    assert result.extra is not None
    # Distinct from local pushdown ("scx_pushdown") — if someone ever
    # aliases these the report's apples-to-apples grouping breaks.
    assert result.extra["native_mechanism"] == "scx_pull_and_filter"
    assert result.extra["native_mechanism"] != "scx_pushdown"
    assert result.extra["native_mechanism"] != "tiledb_value_filter"
    assert result.extra["predicate"] == "n_counts_gt_1000"
    assert result.extra["provider"] == "gcs"
    assert "inner_filter_wall_s" in result.extra


def test_read_cloud_tag_differs_from_filtered_tag(tmp_path: Path) -> None:
    """``read_cloud`` is pull+read; ``read_cloud_filtered_query`` is
    pull+filter. These must never collide under the same ``extra`` key.
    """
    runner = ScxRunner()

    with patch.object(ScxRunner, "_check_pyscx", lambda self: None), \
         patch("benchmarks.comprehensive.runners.scx_runner.pyscx", create=True) as fake_pyscx:
        class _FakeAdata:
            X = None
        class _FakeDS:
            def to_anndata(self):
                return _FakeAdata()
        fake_pyscx.pull = lambda url, local: None
        fake_pyscx.open = lambda local: _FakeDS()

        read_timing = runner.read_cloud("gs://dummy/fixture.scxd/")

    assert read_timing.extra is not None
    assert read_timing.extra["native_mechanism"] == "scx_pull_and_read"
    assert read_timing.extra["native_mechanism"] != "scx_pull_and_filter"
