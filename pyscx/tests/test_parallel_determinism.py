"""Determinism test for parallel shard encoding (Phase 1D).

Verifies that SCX files produce identical data regardless of RAYON_NUM_THREADS.
Uses subprocess invocation because rayon's global thread pool is initialized
once per process and cannot be reconfigured.

Note: File-level byte hashes may differ between runs due to the provenance
timestamp. Instead, we verify that all data arrays (X, layers, obs, var) and
shard-level checksums are identical.
"""

import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import scipy.sparse as sp

import pyscx

# Script run in a subprocess with a controlled RAYON_NUM_THREADS.
WRITE_SCRIPT = """\
import numpy as np
import scipy.sparse as sp
import pandas as pd
import anndata
import pyscx
import sys

seed, n_obs, n_vars, output = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
np.random.seed(seed)
dense = np.random.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
mask = np.random.random((n_obs, n_vars)) > 0.02  # ~2% density
dense[mask] = 0
x = sp.csr_matrix(dense)
obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
adata = anndata.AnnData(X=x, obs=obs, var=var)
pyscx.from_anndata(adata, output)
"""


def _run_write(script, args, threads, timeout=120):
    """Run a write script in a subprocess with a specific RAYON_NUM_THREADS."""
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(threads)
    result = subprocess.run(
        [sys.executable, "-c", script] + [str(a) for a in args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert (
        result.returncode == 0
    ), f"Failed with {threads} threads:\nstdout: {result.stdout}\nstderr: {result.stderr}"


def _compare_scx_data(path1, path2):
    """Read two SCX files and assert identical data arrays."""
    r1 = pyscx.open(path1).to_anndata()
    r2 = pyscx.open(path2).to_anndata()

    # Compare X matrix
    x1 = r1.X.toarray() if sp.issparse(r1.X) else np.asarray(r1.X)
    x2 = r2.X.toarray() if sp.issparse(r2.X) else np.asarray(r2.X)
    np.testing.assert_array_equal(x1, x2, err_msg="X matrix differs")

    # Compare obs/var indices
    assert list(r1.obs.index) == list(r2.obs.index), "obs index differs"
    assert list(r1.var.index) == list(r2.var.index), "var index differs"

    # Compare layers
    assert set(r1.layers.keys()) == set(r2.layers.keys()), "layer keys differ"
    for key in r1.layers:
        l1 = (
            r1.layers[key].toarray()
            if sp.issparse(r1.layers[key])
            else np.asarray(r1.layers[key])
        )
        l2 = (
            r2.layers[key].toarray()
            if sp.issparse(r2.layers[key])
            else np.asarray(r2.layers[key])
        )
        np.testing.assert_array_equal(l1, l2, err_msg=f"Layer '{key}' differs")


def _file_hash(path):
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def test_parallel_determinism():
    """SCX output must be data-identical with 1 vs 8 rayon threads."""
    with tempfile.TemporaryDirectory() as tmp:
        path1 = os.path.join(tmp, "out_1thread.scx")
        path8 = os.path.join(tmp, "out_8thread.scx")

        _run_write(WRITE_SCRIPT, [42, 10000, 5000, path1], threads=1)
        _run_write(WRITE_SCRIPT, [42, 10000, 5000, path8], threads=8)

        # Data must be identical
        _compare_scx_data(path1, path8)

        # File sizes should be identical (same shard data, same structure)
        size1 = os.path.getsize(path1)
        size8 = os.path.getsize(path8)
        assert size1 == size8, f"File sizes differ: {size1} vs {size8}"


def test_parallel_determinism_same_threads():
    """Two runs with the same thread count must produce byte-identical files."""
    with tempfile.TemporaryDirectory() as tmp:
        path_a = os.path.join(tmp, "out_a.scx")
        path_b = os.path.join(tmp, "out_b.scx")

        # Run the same configuration twice — only timestamp should differ
        _run_write(WRITE_SCRIPT, [42, 5000, 3000, path_a], threads=4)
        _run_write(WRITE_SCRIPT, [42, 5000, 3000, path_b], threads=4)

        # Data must be identical
        _compare_scx_data(path_a, path_b)


def test_parallel_determinism_with_layers():
    """SCX output with layers must be data-identical with 1 vs 4 rayon threads."""
    script = """\
import numpy as np
import scipy.sparse as sp
import pandas as pd
import anndata
import pyscx
import sys

seed, n_obs, n_vars, output = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
np.random.seed(seed)
dense = np.random.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
mask = np.random.random((n_obs, n_vars)) > 0.02
dense[mask] = 0
x = sp.csr_matrix(dense)

np.random.seed(seed + 1)
raw_dense = np.random.randint(0, 100, size=(n_obs, n_vars)).astype(np.float32)
raw_mask = np.random.random((n_obs, n_vars)) > 0.02
raw_dense[raw_mask] = 0
raw_csr = sp.csr_matrix(raw_dense)

obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
adata = anndata.AnnData(X=x, obs=obs, var=var)
adata.layers["raw_counts"] = raw_csr
pyscx.from_anndata(adata, output)
"""

    with tempfile.TemporaryDirectory() as tmp:
        path1 = os.path.join(tmp, "out_1thread.scx")
        path4 = os.path.join(tmp, "out_4thread.scx")

        _run_write(script, [42, 5000, 3000, path1], threads=1)
        _run_write(script, [42, 5000, 3000, path4], threads=4)

        # Data must be identical
        _compare_scx_data(path1, path4)

        # File sizes should be identical
        size1 = os.path.getsize(path1)
        size4 = os.path.getsize(path4)
        assert size1 == size4, f"File sizes differ: {size1} vs {size4}"
