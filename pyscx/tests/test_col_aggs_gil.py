"""`pyscx.accel.col_*` must not hold the GIL for the whole streaming scan.

Finding §9.7: `col_aggs.rs`'s `run_csr_f64` ran a full-matrix decode without
`detached` — it even had a `let _ = py;` where the release belonged — so
`pyscx.accel.col_sums(...)` blocked every other Python thread for the duration.
Every sibling heavy entry point (`filtering.rs`, `preprocessing.rs`, `de.rs`,
`pca.rs`) already released it.

**How this is measured, and why not with a speedup ratio.** A "4 threads finish
in less than 4× the single-threaded time" test is a throughput claim, and it
gets flaky the moment the machine is loaded — exactly the condition under which
CI runs. This measures the property directly instead: a monitor thread stamps
`perf_counter()` in a tight loop while the main thread runs `col_sums`, and the
assertion is on the **largest gap between consecutive stamps**. If the GIL is
held for the whole scan the monitor cannot run at all and the largest gap *is*
the whole operation; if it is released, the monitor is scheduled throughout and
the gap stays a small fraction. That distinction survives an arbitrarily busy
host, because a loaded machine delays the monitor under *both* behaviours.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import scipy.sparse as sp

import pyscx

# Sized from measurement, not guessed: at 40k x 400 the single-pass ops
# finished in ~7 ms and skipped themselves as unmeasurable. At 300k x 800
# (24M nnz) `col_sums` / `col_nnz` take ~100 ms and `col_var` ~1.4 s, which is
# comfortably above the monitor's 1 ms cadence. Build cost is ~1 s, paid once
# per module.
N_OBS = 300_000
N_VARS = 800
SHARD_SIZE = 10_000  # → 30 shards

# The scan must take at least this long for the measurement to mean anything.
MIN_MEASURABLE_S = 0.05
# Under a released GIL the monitor keeps running throughout, so the worst gap
# should be a small fraction of the op. Generous: the real contrast is ~1.0 vs
# ~0.01, so anything under half is decisive without being brittle.
MAX_GAP_FRACTION = 0.5


@pytest.fixture(scope="module")
def backed_x(tmp_path_factory):
    import anndata as ad

    rng = np.random.default_rng(17)
    x = sp.random(
        N_OBS, N_VARS, density=0.1, format="csr", dtype=np.float32, random_state=rng
    )
    x.data = (x.data * 50).astype(np.float32) + 1.0
    a = ad.AnnData(X=x)
    a.obs_names = [f"c{i}" for i in range(N_OBS)]
    a.var_names = [f"g{i}" for i in range(N_VARS)]
    path = tmp_path_factory.mktemp("gil") / "counts.scx"
    pyscx.from_anndata(a, str(path), shard_size=SHARD_SIZE)
    return pyscx.open(str(path)).to_anndata(backed=True).X


def _largest_gap_during(op) -> tuple[float, float]:
    """Run `op`, returning `(op_duration, largest monitor-thread stall)`."""
    stamps: list[float] = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            stamps.append(time.perf_counter())
            time.sleep(0.001)

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    # Let the monitor establish a baseline cadence before the op starts.
    time.sleep(0.02)
    start = time.perf_counter()
    op()
    duration = time.perf_counter() - start
    stop.set()
    t.join(timeout=5)

    during = [s for s in stamps if s >= start]
    if len(during) < 2:
        # The monitor produced no samples inside the window at all — that is
        # itself the failure the test is looking for.
        return duration, duration
    gaps = np.diff(np.asarray(during))
    # The stall that matters is measured from the last pre-op stamp, so a
    # monitor frozen for the entire op is caught rather than showing zero gaps.
    before = [s for s in stamps if s < start]
    leading = during[0] - (before[-1] if before else start)
    return duration, float(max(gaps.max(), leading))


@pytest.mark.parametrize("op_name", ["col_sums", "col_var", "col_nnz"])
def test_col_aggs_release_the_gil(backed_x, op_name):
    fn = getattr(pyscx.accel, op_name)
    duration, largest_gap = _largest_gap_during(lambda: fn(backed_x))

    if duration < MIN_MEASURABLE_S:
        pytest.skip(
            f"{op_name} finished in {duration * 1000:.1f} ms — too fast to "
            "distinguish a held GIL from a released one on this host"
        )

    assert largest_gap < MAX_GAP_FRACTION * duration, (
        f"{op_name} starved a concurrent Python thread for {largest_gap:.3f}s "
        f"out of a {duration:.3f}s scan ({largest_gap / duration:.0%} of it) — "
        "the streaming decode is holding the GIL"
    )


def test_concurrent_col_sums_agree(backed_x):
    """Correctness under the newly-possible concurrency.

    Releasing the GIL means several `col_*` calls can now genuinely overlap on
    one `BackedCsrReader`. They share its mmap and its decoded-shard LRU, so
    this is also the regression test for that sharing being sound.
    """
    expected = np.asarray(pyscx.accel.col_sums(backed_x))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = [
            np.asarray(r)
            for r in pool.map(lambda _: pyscx.accel.col_sums(backed_x), range(8))
        ]
    for i, got in enumerate(results):
        np.testing.assert_array_equal(
            got, expected, err_msg=f"concurrent col_sums call {i} disagreed"
        )
