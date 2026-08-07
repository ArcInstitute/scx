"""Accel ops must read a snapshot of `X`, never the caller's live numpy buffer.

Finding §10.1: several `py.detach(...)` closures held Rust `&[T]` slices
borrowed from live, Python-reachable numpy buffers. rust-numpy borrows are not
GIL-bound and do not clear numpy's `WRITEABLE` flag, and both
`astype(..., copy=False)` and `scipy.sparse.csr_matrix(A)` on an already-CSR
`A` are identity operations — so those slices *were* `adata.X.data` /
`adata.X.indices`. Releasing the GIL is precisely what let another Python
thread rewrite them mid-kernel: a data race, wrong results, and for the
`indices` array a wrong column lookup.

**How this is measured, and why not "mutate and check the answer is right".**
The obvious test — scribble on `X` from a thread and assert the result is
unchanged — is *unsound in both directions*. It can pass on broken code by
timing luck; worse, it can **fail on correct code**, because a mutator running
from before the call corrupts the buffer before the snapshot is taken, and no
implementation can un-see that.

So these tests do not ask "did the answer change". They ask **"is the answer a
snapshot at all"**. The mutator flips the buffer between two states, A and B,
for the whole duration of the op. A correct implementation reads one coherent
state and must return exactly `f(A)` or exactly `f(B)` — it does not matter
which. An implementation reading the live buffer sees A for some rows and B for
others and returns a mixture, which equals neither. That discriminator is
indifferent to *when* the mutator starts, which is the part no test can control.

Three guards keep a run from reporting a pass it did not earn:

  * skip if the op finished too fast to race at all;
  * skip if the mutator never got the GIL inside the op window — it can only
    run when the GIL is released, so a landed write is also the proof that the
    op detached;
  * **assert** (not skip — a vacuous fixture is a test bug) that `f(A)` and
    `f(B)` actually differ, so "equals one of them" is a real constraint.

The mutator writes **in-range values only**. Writing an out-of-range column
index into a buffer a broken build still reads is undefined behaviour that
would segfault the pytest process instead of failing a test.

**What is deliberately not tested here.** `from_anndata`'s CSC-sidecar block and
`decompose_scipy_csr_with` are fixed sites with no race test: both sit inside a
long write path where the mutation window is a small fraction of the call, so a
reproduction would be unreliable rather than informative. They are covered by
`test_csc_convert.py` / `test_csc_lifecycle.py` / `test_golden_files.py` for
*correctness*, and by code review for the borrow itself.

There is also no `indices` counterpart to the `knockdown_efficiency` test,
though `indices` is the more dangerous array. It was written, and it **fails on
the fixed build** — for a reason worth recording. Unlike `pseudobulk_means`,
that op runs its input through `ensure_csr` first, and a flipper that rewrites
column indices makes the matrix unsorted. `scipy.sparse.csr_matrix(x)` returns
a *new* object (verified: `csr_matrix(x) is x` -> `False`), so
`has_sorted_indices` is recomputed rather than read from `x`'s cache, comes
back `False`, and `ensure_csr` takes its `.sorted_indices()` branch — whose
internal 38 MB copy is exactly the kind of large numpy copy measured above as
torn 27 % of the time. The blend the test then sees originates in *scipy's*
copy, not in the snapshot under test. **An op that may re-canonicalize its
input cannot be tested with an index flipper.** `pseudobulk_means` can, because
`owned_csr` never re-sorts.

`Experiment.gather_rows_sparse` has
the same defect in its `rows` argument — bounds-checked under the GIL, then
read detached, so the documented `IndexError` is a TOCTOU — but it is not
reproducible this way and no test pretending otherwise belongs here.
`BackedCsrReader::read_rows_with` copies `rows` into its sorted plan as its
very first statement and never looks at the array again, so the exposure is a
sub-millisecond window at the top of the detached region. A version of this
test was written, and it passed on the *unfixed* build — i.e. it had no
discriminating power. Its fix rests on reading the code; the `IndexError`
contract itself stays covered by `test_gather_rows_sparse.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import pyscx

from _gil_probe import run_with_mutator

# Sized so a single pass is comfortably longer than a GIL switch interval
# (~5 ms), following the measured approach in test_col_aggs_gil.py. At
# 200k x 600 @ 8% (9.6M nnz) pseudobulk_means measured ~40 ms and a full
# `data.fill()` costs ~5 ms, so ~8 flips land inside the window.
N_OBS = 200_000
N_VARS = 600
DENSITY = 0.08

# Dense is sized by bytes, not nnz: 80k x 1200 f32 is 384 MB, which measured
# ~40 ms per aggregation — enough flips land in that window, and 20 ms would
# not have been.
DENSE_OBS = 80_000
DENSE_VARS = 1_200

MIN_MEASURABLE_S = 0.015
MIN_WRITES = 5

# The ops are rayon-parallel float reductions and are not bit-reproducible
# run to run (measured: ~1e-7 relative drift in knockdown_efficiency). The
# corruption these tests look for is a factor of two or more, so a tolerance
# four orders of magnitude below that is still decisive.
RTOL = 1e-5


def _skip_unless_reproduced(probe, what: str) -> None:
    if probe.duration < MIN_MEASURABLE_S:
        pytest.skip(
            f"{what} finished in {probe.duration * 1000:.1f} ms — too fast to "
            "race on this host"
        )
    if probe.writes_during < MIN_WRITES:
        pytest.skip(
            f"the mutator landed only {probe.writes_during} writes inside the "
            f"{probe.duration:.3f}s {what} window (largest GIL stall "
            f"{probe.largest_gap:.3f}s) — the reproduction did not run"
        )


def _assert_single_snapshot(probe, candidates: dict, what: str) -> None:
    """The result must match exactly one coherent state, not a blend of both."""
    names = list(candidates)
    a, b = (np.asarray(candidates[n], dtype=np.float64) for n in names)
    assert not np.allclose(a, b, rtol=RTOL), (
        f"the {what} fixture is vacuous: the two mutator states produce the "
        "same result, so 'read a coherent snapshot' is not a constraint"
    )

    _skip_unless_reproduced(probe, what)

    got = np.asarray(probe.result, dtype=np.float64)
    if any(np.allclose(got, np.asarray(candidates[n], dtype=np.float64), rtol=RTOL)
           for n in names):
        return
    raise AssertionError(
        f"{what} returned a blend of the two buffer states rather than a "
        f"snapshot of either — it read the caller's live numpy buffer while the "
        f"GIL was released ({probe.writes_during} mutations landed inside the "
        f"{probe.duration:.3f}s call)"
    )


def _labels(n_obs: int) -> np.ndarray:
    return np.array([f"p{i % 4}" for i in range(n_obs)], dtype=object)


# The mutation must be atomic with respect to the GIL, or "the buffer is always
# in state A or state B" is false and the test fails on *correct* code.
#
# numpy releases the GIL inside a large array assignment. Measured on the dev
# host with a concurrent reader comparing the two ends of the buffer: a
# 9.6M-element `arr[:] = other` is observed **torn in 3 076 100 of 11 357 806
# checks (27 %)**, while a 480-element strided write is torn **0 times in
# 3 347 462 checks**. numpy's threaded inner loops are gated on a size
# threshold, so keeping the write small keeps it indivisible.
#
# Strided, not contiguous: the touched elements have to be spread across the
# whole matrix. A contiguous run of 480 nonzeros covers a handful of adjacent
# rows, which a live-buffer reader visits at nearly the same instant and so
# would read coherently by accident — no blend, no signal.
ATOMIC_WRITE_ELEMENTS = 480

# How many distinct genes `knockdown_efficiency` is asked about. See the note
# in that test — too few and each gene's baseline flips wholesale.
N_TARGET_GENES = 200

# Large enough that a single flipped element moves its output cell far past
# RTOL, so a blend cannot hide inside the tolerance.
SENTINEL = 1e4


def _two_state_flipper(arr: np.ndarray, sentinel):
    """Return `(flip, apply_a, apply_b)` over a small strided view of `arr`.

    `flip` alternates on each call — that is the mutator. `apply_a` / `apply_b`
    set a state outright, so the test can precompute `f(A)` and `f(B)`.
    """
    flat = arr.ravel()
    assert flat.base is not None or flat is arr, "ravel() must be a view, not a copy"
    stride = max(1, flat.size // ATOMIC_WRITE_ELEMENTS)
    view = flat[::stride]
    assert view.size <= 2 * ATOMIC_WRITE_ELEMENTS, (
        f"victim view is {view.size} elements — large enough that numpy may "
        "release the GIL mid-write, which would tear the mutation"
    )
    pristine = view.copy()
    filled = np.full(view.shape, sentinel, dtype=arr.dtype)

    def apply_a():
        view[...] = pristine

    def apply_b():
        view[...] = filled

    state = {"next_a": True}

    def flip():
        (apply_a if state["next_a"] else apply_b)()
        state["next_a"] = not state["next_a"]

    return flip, apply_a, apply_b


@pytest.fixture(scope="module")
def sparse_template():
    rng = np.random.default_rng(11)
    x = sp.random(
        N_OBS, N_VARS, density=DENSITY, format="csr", dtype=np.float32, random_state=rng
    )
    x.sort_indices()
    return x


@pytest.fixture
def sparse_adata(sparse_template):
    import anndata as ad

    x = sparse_template.copy()
    # The aliasing preconditions. If any of these stop holding, the fixture no
    # longer reproduces the bug and the test would be silently vacuous.
    assert x.format == "csr"
    assert x.has_sorted_indices, "csr_matrix(X) must be an identity op"
    assert x.data.dtype == np.float32, "astype(f32, copy=False) must be a no-op"
    assert x.indices.dtype == np.int32, "astype(i32, copy=False) must be a no-op"
    # NOTE: `indptr` deliberately gets no such check. scipy gives indptr and
    # indices the same width, so at this scale indptr is int32 and the coercion
    # to int64 copies — mutating it would test nothing. `data` and `indices`
    # are the two that genuinely alias, and `indices` is the dangerous one.

    a = ad.AnnData(X=x)
    a.obs["pert"] = _labels(N_OBS)
    a.obs_names = [f"c{i}" for i in range(N_OBS)]
    a.var_names = [f"g{i}" for i in range(N_VARS)]
    return a


@pytest.fixture
def dense_adata():
    import anndata as ad

    rng = np.random.default_rng(3)
    x = rng.random((DENSE_OBS, DENSE_VARS), dtype=np.float32)
    assert x.dtype == np.float32, "astype(f32, copy=False) must be a no-op"
    assert x.flags["C_CONTIGUOUS"], "ascontiguousarray must be a no-op"

    a = ad.AnnData(X=x)
    a.obs["pert"] = _labels(DENSE_OBS)
    a.obs_names = [f"c{i}" for i in range(DENSE_OBS)]
    a.var_names = [f"g{i}" for i in range(DENSE_VARS)]
    return a


def _pseudobulk(adata) -> np.ndarray:
    """`pseudobulk_means` returns `(means_2d, group_labels)`."""
    return np.asarray(
        pyscx.accel.pseudobulk_means(adata, "pert", device="cpu")[0], dtype=np.float64
    )


# ── pseudobulk_means, sparse in-memory X (§10.1 site 1) ──────────────────────


def test_pseudobulk_means_sparse_snapshots_x_data(sparse_adata):
    flip, set_a, set_b = _two_state_flipper(sparse_adata.X.data, SENTINEL)

    set_a()
    means_a = _pseudobulk(sparse_adata)
    set_b()
    means_b = _pseudobulk(sparse_adata)
    set_a()

    probe = run_with_mutator(lambda: _pseudobulk(sparse_adata), flip)
    _assert_single_snapshot(
        probe, {"pristine": means_a, "spiked": means_b}, "pseudobulk_means (sparse data)"
    )


def test_pseudobulk_means_sparse_snapshots_x_indices(sparse_adata):
    """The dangerous half: these values are consumed as column indices.

    Collapsing them onto column 0 stays in range (no out-of-bounds read on a
    build that still trusts the live buffer) while changing the answer wholesale.
    """
    # Sentinel 0, not SENTINEL: these are column indices and must stay in
    # range, or a build that still trusts the live buffer reads out of bounds.
    flip, set_a, set_b = _two_state_flipper(sparse_adata.X.indices, 0)

    set_a()
    means_a = _pseudobulk(sparse_adata)
    set_b()
    means_b = _pseudobulk(sparse_adata)
    set_a()

    probe = run_with_mutator(lambda: _pseudobulk(sparse_adata), flip)
    _assert_single_snapshot(
        probe,
        {"pristine": means_a, "collapsed": means_b},
        "pseudobulk_means (sparse indices)",
    )


# ── pseudobulk_means, dense in-memory X (§10.1 site 2) ───────────────────────


def test_pseudobulk_means_dense_snapshots_x(dense_adata):
    flip, set_a, set_b = _two_state_flipper(dense_adata.X, SENTINEL)

    set_a()
    means_a = _pseudobulk(dense_adata)
    set_b()
    means_b = _pseudobulk(dense_adata)
    set_a()

    probe = run_with_mutator(lambda: _pseudobulk(dense_adata), flip)
    _assert_single_snapshot(
        probe, {"pristine": means_a, "spiked": means_b}, "pseudobulk_means (dense)"
    )


# ── knockdown_efficiency (§10.1 site 3) ──────────────────────────────────────


def test_knockdown_efficiency_snapshots_x_data(sparse_adata):
    """Three kernels run under one detach here — all read the same buffers."""
    # Perturbation labels must name real genes; every 4th cell is a control.
    #
    # Many target genes, not a handful: the op consults one control *baseline*
    # per targeted gene, and the atomic mutation spikes roughly one value per
    # gene. With four targets each baseline is effectively a coin flip that
    # lands wholly in state A or state B, so the result matches one of them and
    # the test proves nothing — verified, it passed on the unfixed build.
    # N_TARGET_GENES independent baselines make an all-heads outcome
    # vanishingly unlikely and the blend certain.
    sparse_adata.obs["perturbation"] = np.where(
        np.arange(N_OBS) % 4 == 0,
        "non-targeting",
        [f"g{i % N_TARGET_GENES}" for i in range(N_OBS)],
    )
    flip, set_a, set_b = _two_state_flipper(sparse_adata.X.data, SENTINEL)

    def run() -> np.ndarray:
        pyscx.accel.knockdown_efficiency(
            sparse_adata, control="non-targeting", device="cpu"
        )
        return np.asarray(sparse_adata.obs["KnockDownEfficiency"], dtype=np.float64)

    set_a()
    eff_a = run()
    set_b()
    eff_b = run()
    set_a()

    probe = run_with_mutator(run, flip)
    # Control cells are NaN in both states; compare only the perturbed cells.
    finite = np.isfinite(eff_a) & np.isfinite(eff_b)
    probe.result = probe.result[finite]
    _assert_single_snapshot(
        probe,
        {"pristine": eff_a[finite], "spiked": eff_b[finite]},
        "knockdown_efficiency",
    )
