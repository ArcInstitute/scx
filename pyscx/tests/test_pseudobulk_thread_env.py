"""`pseudobulk_dex` must not mutate process-global thread state.

Regression test for a bug where the first `pseudobulk_dex` call set
``NUMBA_NUM_THREADS`` (and the OMP/OpenBLAS/MKL equivalents) in ``os.environ``
and shrank the live numba pool, for the lifetime of the process. Numba re-reads
that variable on *every fresh compilation* and raises when it disagrees with the
already-launched pool, so a single `pseudobulk_dex` call made every later numba
compile in the process fail with ``RuntimeError: Cannot set NUMBA_NUM_THREADS to
a different value once the threads have been launched``.

In a full-suite run that cascaded to 79 failures across 21 files; for a user it
breaks any scanpy step taken after a pseudobulk DE call. Capping worker threads
is already handled upstream — pydeseq2 wraps its joblib calls in
``parallel_backend(..., inner_max_num_threads=1)``, and joblib applies that to
the *worker* environment only — so pyscx must leave the parent alone.

**Why a subprocess.** The thing under test is one-shot, process-global state:
the mutation uses ``os.environ.setdefault``, so the second and later calls in a
process are no-ops and any in-process assertion after the first call passes
vacuously. An earlier draft of this file asserted in-process and was
order-dependent — all three BLAS cases passed when they ran after the numba
case, and only the first-executed one failed when run alone. A fresh
interpreter per check is the only honest way to observe it.
"""

import json
import subprocess
import sys

import pytest

# Executed in a fresh interpreter. Writes a JSON verdict to argv[1] so the
# parent can make granular assertions; pydeseq2 chatters on stdout, so the
# result cannot simply be printed.
_CHILD = r'''
import json, os, sys

import anndata as ad
import numpy as np
import pandas as pd
import numba

import pyscx

REFERENCE = "control"
VARS = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def make_njit(scale):
    # cache=False matters: numba re-enters its config guard on *compilation*, so
    # a cached or already-compiled function sails past the bug. The distinct
    # `scale` keeps the two kernels from sharing a dispatcher.
    @numba.njit(parallel=True, cache=False)
    def _kernel(x):
        total = 0.0
        for i in numba.prange(x.shape[0]):
            total += x[i] * scale
        return total

    return _kernel


def perturb_adata(seed=0, n_genes=12, n_donors=3, cells_per=20):
    rng = np.random.default_rng(seed)
    base = rng.uniform(45.0, 55.0, size=n_genes)
    effect = {REFERENCE: np.ones(n_genes), "drug": 2.0 ** np.linspace(-1.0, 1.0, n_genes)}
    blocks, perts, donors = [], [], []
    for p in (REFERENCE, "drug"):
        for d in [f"d{j}" for j in range(n_donors)]:
            mean = base * effect[p]
            blocks.append(rng.poisson(mean[None, :], size=(cells_per, n_genes)).astype(np.float32))
            perts.extend([p] * cells_per)
            donors.extend([d] * cells_per)
    x = np.vstack(blocks)
    obs = pd.DataFrame(
        {"perturbation": perts, "donor": donors},
        index=[f"cell_{i}" for i in range(x.shape[0])],
    )
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)])
    return ad.AnnData(X=x, obs=obs, var=var)


# Launch numba's pool at the machine default BEFORE the DE call. Without this
# the pool is uninitialized and numba accepts the change without complaint.
make_njit(1.0)(np.arange(8, dtype=np.float64))

env_before = {v: os.environ.get(v) for v in VARS}
threads_before = numba.get_num_threads()

pyscx.accel.pseudobulk_dex(
    perturb_adata(),
    groupby=["perturbation", "donor"],
    test_col="perturbation",
    reference=REFERENCE,
    min_cells_per_group=1,
)

result = {
    "env_before": env_before,
    "env_after": {v: os.environ.get(v) for v in VARS},
    "threads_before": threads_before,
    "threads_after": numba.get_num_threads(),
}

# The user-visible symptom: a compilation that happens after the DE call.
try:
    make_njit(2.0)(np.arange(8, dtype=np.float64))
    result["late_compile_error"] = None
except Exception as exc:
    result["late_compile_error"] = f"{type(exc).__name__}: {exc}"

with open(sys.argv[1], "w") as fh:
    json.dump(result, fh)
'''


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    """Run the scenario once in a fresh interpreter, return its verdict."""
    pytest.importorskip("pydeseq2")
    pytest.importorskip("numba")

    out = tmp_path_factory.mktemp("thread_env") / "result.json"
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(out)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0 or not out.exists():
        pytest.fail(
            "pseudobulk_dex probe subprocess failed "
            f"(rc={proc.returncode})\n--- stderr ---\n{proc.stderr[-4000:]}"
        )
    return json.loads(out.read_text())


def test_pseudobulk_dex_leaves_numba_compilable(probe):
    """A numba compilation after `pseudobulk_dex` must still succeed.

    This is the assertion that reproduces the bug. The failure lands on the
    *next* thing to compile, not inside `pseudobulk_dex` itself.
    """
    assert probe["late_compile_error"] is None, (
        "a numba compilation after pseudobulk_dex raised "
        f"{probe['late_compile_error']!r} — pseudobulk_dex changed process-global "
        "numba thread state"
    )


def test_pseudobulk_dex_leaves_thread_env_untouched(probe):
    """None of the four thread-count env vars may change in the parent.

    NUMBA_NUM_THREADS is the one that hard-errors; the BLAS trio fail silently
    (a sized pool ignores a late change), which is why this went unnoticed. All
    four still reprogram the caller's environment and anything they spawn later.
    """
    assert probe["env_after"] == probe["env_before"], (
        "pseudobulk_dex mutated thread-count env vars in the parent process: "
        f"{probe['env_before']} -> {probe['env_after']}. Worker limits belong in "
        "the worker env; pydeseq2 already sets them via joblib's "
        "inner_max_num_threads."
    )


def test_pseudobulk_dex_leaves_live_numba_pool_untouched(probe):
    """The caller's live numba thread count must survive the call."""
    assert probe["threads_after"] == probe["threads_before"], (
        f"pseudobulk_dex changed the live numba thread count "
        f"{probe['threads_before']} -> {probe['threads_after']}"
    )
