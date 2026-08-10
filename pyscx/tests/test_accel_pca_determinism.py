"""CPU PCA reproducibility.

`pyscx.accel.pca` used to give a different answer on every call: the covariance
build and the transpose SpMM accumulated into per-worker buffers whose
row->worker assignment came from rayon work-stealing and whose merge order came
from `ThreadLocal::iter_mut()`. Five consecutive `method="covariance"` runs
produced five different results while the docs promised determinism.

The reductions now partition their *output*, so the schedule cannot reach the
result. These tests pin the user-visible half of that: repeated calls agree
exactly, on every `X` kind and both methods.

What they deliberately do **not** claim is identity across thread counts. faer's
dense QR and eigendecomposition block by the ambient rayon width, so their low
bits move with it — the same property numpy/scipy have, where LAPACK's bits move
with `OMP_NUM_THREADS`. `SCX_ACCEL_DETERMINISTIC_LINALG=1` pins that; the
subprocess test at the bottom is the one that exercises it, because
`RAYON_NUM_THREADS` has to be set before `import pyscx`.

Every test pins `device="cpu"`: an unpinned accel call routes to the GPU on a GPU
host, and the GPU path is a different implementation with its own guarantees.
"""

import os
import subprocess
import sys
import tempfile

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import pyscx


def _adata(n_obs=8192, n_vars=40, seed=0):
    """A fixture whose sums are order-sensitive *at float32 resolution*.

    This has to be built deliberately, and the obvious fixture does not work.
    `X_pca` comes back as float32, so an f64 reduction that reassociates at the
    1e-16 level rounds to the same float32 and an equality assertion passes on a
    broken implementation. Merely spanning a wide dynamic range is not enough —
    that was tried, and `test_the_fixture_can_detect_a_reordering` caught it.

    What works is catastrophic cancellation: every value sits near a large
    per-column offset, so the covariance's `sum(x^2) - n*mu^2` subtracts two
    numbers that agree to ~12 digits. f64 keeps ~4, and a reassociation at the
    1e-16 relative level of the *large* term moves the small difference by ~1e-4
    relative — comfortably visible in float32.

    `n_obs` also has to be large. The reduction these tests guard used
    `with_min_len((n_rows / workers).max(256))`, which cannot split 600 rows into
    two >=256-row chunks — so a small fixture ran single-threaded, was
    deterministic by accident, and passed against the *unfixed* build. 8192 rows
    (and `shard_size=1024` on the backed path) split at every thread count.
    """
    rng = np.random.default_rng(seed)
    # Dense: every column pair must be exercised, and a sparse fixture would let
    # whole blocks of the covariance triangle go untouched.
    offsets = (10.0 ** rng.integers(4, 6, size=n_vars)).astype(np.float32)
    noise = rng.standard_normal((n_obs, n_vars)).astype(np.float32)
    X = sp.csr_matrix((offsets[None, :] + noise).astype(np.float32))
    X.sort_indices()
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    return anndata.AnnData(X=X, obs=obs, var=var)


def _run(adata, method, n_comps=10):
    a = adata.copy()
    pyscx.accel.pca(a, n_comps=n_comps, method=method, device="cpu")
    return (
        np.asarray(a.obsm["X_pca"]),
        np.asarray(a.varm["PCs"]),
        np.asarray(a.uns["pca"]["variance_ratio"]),
    )


def _assert_identical(runs, what):
    first = runs[0]
    for i, other in enumerate(runs[1:], start=1):
        for name, a, b in zip(("X_pca", "PCs", "variance_ratio"), first, other):
            assert np.array_equal(a, b), (
                f"{what}: {name} differs between run 0 and run {i} — "
                f"max |delta| {np.max(np.abs(a - b))}"
            )


@pytest.mark.parametrize("method", ["covariance", "randomized"])
def test_in_memory_pca_is_bit_identical_across_runs(method):
    """A regression guard, not a reproduction — and the difference is worth knowing.

    Run against the pre-fix build, this arm **passes**. The in-memory reductions
    used rayon's indexed `fold`/`reduce`, whose split tree is length-driven and
    whose reduce order follows that tree, so they were already order-stable. It
    was the *streaming* reductions that were not: they accumulated into
    `ThreadLocal` buffers, and which threads registered — hence the
    `iter_mut()` merge order — came from work-stealing. `test_backed_*` below is
    the arm that actually reddens on the old build.

    Keep this one anyway: the two paths now share a kernel, and this is what
    stops the in-memory side from regressing when that kernel changes.
    """
    adata = _adata()
    _assert_identical([_run(adata, method) for _ in range(5)], f"in-memory {method}")


@pytest.mark.parametrize("method", ["covariance", "randomized"])
def test_backed_pca_is_bit_identical_across_runs(method):
    """The streaming path — the one that actually reproduced the defect.

    Both parametrisations fail against the pre-fix build: `covariance` through
    `accumulate_covariance_streaming` and `randomized` through
    `streaming_spmm_transpose`, both of which merged `ThreadLocal` accumulators
    in registration order.

    `shard_size` gives the file several shards -- a single-shard file would not
    exercise the cross-shard accumulation at all -- while staying large enough
    per shard to split across workers (see `_adata`).
    """
    adata = _adata()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "det.scx")
        pyscx.from_anndata(adata, path, shard_size=1024)
        runs = []
        for _ in range(5):
            backed = pyscx.open(path).to_anndata(backed=True)
            pyscx.accel.pca(backed, n_comps=10, method=method, device="cpu")
            runs.append(
                (
                    np.asarray(backed.obsm["X_pca"]),
                    np.asarray(backed.varm["PCs"]),
                    np.asarray(backed.uns["pca"]["variance_ratio"]),
                )
            )
    _assert_identical(runs, f"backed {method}")


def test_the_fixture_can_detect_a_reordering():
    """Premise for every equality assertion above.

    If this fixture summed the same to the last bit under any order, the tests
    above would pass on a still-broken implementation. Reversing the row order is
    a change no correct implementation should notice in *value* — so if the
    embeddings come back bit-identical, the fixture is too well-conditioned to
    detect anything and the suite is vacuous.
    """
    adata = _adata()
    forward = _run(adata, "covariance")[0]

    reversed_rows = adata[::-1].copy()
    back = _run(reversed_rows, "covariance")[0][::-1]

    assert not np.array_equal(forward, back), (
        "premise failed: reversing the row order left the embeddings bit-identical, "
        "so this fixture cannot distinguish a fixed accumulation order from an "
        "arbitrary one and the equality tests above prove nothing"
    )
    # Deliberately no "...but the values still agree" assertion here. This
    # fixture is built to cancel catastrophically, so its embeddings genuinely
    # move by ~1e-3 relative under a reordering — that is the property being
    # relied on, not a defect. PCA's numerical correctness is asserted on
    # well-conditioned data in test_accel.py and test_pca_mask_var.py.


def _unsorted_copy(adata):
    """The same matrix with every row's column indices in descending order.

    `scipy` keeps `indices`/`data` exactly as handed to it and only lowers
    `has_sorted_indices`; the matrix is numerically identical, and
    `.toarray()` round-trips. Anything that reads it positionally, though, sees
    a different order.
    """
    X = adata.X.tocsr(copy=True)
    for r in range(X.shape[0]):
        s, e = X.indptr[r], X.indptr[r + 1]
        X.indices[s:e] = X.indices[s:e][::-1]
        X.data[s:e] = X.data[s:e][::-1]
    X.has_sorted_indices = False
    out = adata.copy()
    out.X = X
    np.testing.assert_array_equal(out.X.toarray(), adata.X.toarray())
    return out


@pytest.mark.parametrize("masked", [False, True])
def test_an_unsorted_csr_gives_the_same_answer(masked):
    """Unsorted input must not change the result.

    `owned_csr` calls `scipy.sparse.csr_matrix(x)`, which is a no-op *view* when
    `x` is already CSR — so before this change the caller's index order reached
    the kernels untouched. `pca()` now sorts at the boundary.

    Removing that `ensure_csr` call reddens **both** arms, for two different
    reasons, which is why both are here:

    - `masked=True` hits `project_csr`, whose merge scan walks a monotonic
      pointer. It is guarded only by a `debug_assert` — so this panics in a debug
      build and, in the release builds users run, silently selects the *wrong
      columns*. That is a correctness bug, not a rounding one.
    - `masked=False` never reaches `project_csr`. It fails because reversing a
      row reverses the order its products are summed in, which moves the
      embeddings by ~7e-9 — right at float32 epsilon. Small, but it means the
      answer depended on how the caller happened to build their matrix.
    """
    adata = _adata(n_obs=512, n_vars=24)
    mask = np.zeros(adata.n_vars, dtype=bool)
    mask[::2] = True
    kwargs = {"mask_var": mask} if masked else {}

    a = adata.copy()
    pyscx.accel.pca(a, n_comps=5, method="covariance", device="cpu", **kwargs)

    b = _unsorted_copy(adata)
    pyscx.accel.pca(b, n_comps=5, method="covariance", device="cpu", **kwargs)

    np.testing.assert_array_equal(
        np.asarray(a.obsm["X_pca"]),
        np.asarray(b.obsm["X_pca"]),
        err_msg="sorting the CSR indices changed the embeddings",
    )
    np.testing.assert_array_equal(
        np.asarray(a.varm["PCs"]),
        np.asarray(b.varm["PCs"]),
        err_msg="sorting the CSR indices changed the loadings",
    )


_SUBPROCESS = """
import os, sys
os.environ["RAYON_NUM_THREADS"] = sys.argv[1]
if sys.argv[2] == "pinned":
    os.environ["SCX_ACCEL_DETERMINISTIC_LINALG"] = "1"
import numpy as np, pandas as pd, scipy.sparse as sp, anndata, pyscx

rng = np.random.default_rng(0)
n_obs, n_vars = 8192, 40
offsets = (10.0 ** rng.integers(4, 6, size=n_vars)).astype(np.float32)
noise = rng.standard_normal((n_obs, n_vars)).astype(np.float32)
X = sp.csr_matrix((offsets[None, :] + noise).astype(np.float32))
X.sort_indices()
a = anndata.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
                    var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]))
pyscx.accel.pca(a, n_comps=10, method="covariance", device="cpu")
np.save(sys.argv[3], np.asarray(a.obsm["X_pca"]))
"""


def _pca_under(threads, mode, out):
    subprocess.run(
        [sys.executable, "-c", _SUBPROCESS, str(threads), mode, out],
        check=True,
        capture_output=True,
    )
    return np.load(out)


def test_pinned_linalg_makes_pca_identical_across_thread_counts():
    """`SCX_ACCEL_DETERMINISTIC_LINALG=1` is what buys cross-configuration identity.

    Subprocesses because `RAYON_NUM_THREADS` is read when the pool is first
    built, which happens on import — setting it in-process would be a no-op and
    the test would pass without varying anything.

    Only the pinned arm is asserted. The unpinned arm is *allowed* to differ, and
    asserting that it does would be asserting a faer implementation detail: at
    these sizes it often agrees anyway, so a "must differ" assertion here would
    be flaky. The proof that the knob is not dead weight lives in Rust, in
    `scx-accel/tests/pca_linalg_parallelism.rs`, where the parallelism can be set
    directly rather than inferred from a pool width.
    """
    with tempfile.TemporaryDirectory() as d:
        one = _pca_under(1, "pinned", os.path.join(d, "one.npy"))
        many = _pca_under(8, "pinned", os.path.join(d, "many.npy"))
    assert np.array_equal(one, many), (
        "with SCX_ACCEL_DETERMINISTIC_LINALG=1, PCA must not depend on "
        f"RAYON_NUM_THREADS — max |delta| {np.max(np.abs(one - many))}"
    )
