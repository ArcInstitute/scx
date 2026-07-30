"""CPU stage profiler surface (Phase-2 task 2.0).

The profiler is gated by `SCX_CPU_PROFILE` read once at process init, so these
tests only assert the Python surface shape / reset semantics — not that counters
populate (that requires the env set before import, exercised by
`benchmarks/scripts/profile_cpu_stages_backed.py`).
"""
import os
import sys

import pyscx


EXPECTED_BUCKETS = ("io", "decode_scx1", "decode_generic", "reduction", "marshalling")


def test_cpu_profile_snapshot_shape():
    snap = pyscx.accel.cpu_profile_snapshot()
    assert "enabled" in snap
    assert isinstance(snap["enabled"], bool)
    for bucket in EXPECTED_BUCKETS:
        assert bucket in snap, f"missing bucket {bucket}"
        st = snap[bucket]
        assert set(st.keys()) == {"ms", "count", "bytes"}
        assert st["ms"] >= 0.0
        assert st["count"] >= 0
        assert st["bytes"] >= 0


def test_cpu_profile_reset_is_callable():
    # reset() must be a no-throw no-op regardless of enabled state.
    pyscx.accel.cpu_profile_reset()
    snap = pyscx.accel.cpu_profile_snapshot()
    # When profiling is disabled (default in the test process), all buckets zero.
    if not snap["enabled"]:
        for bucket in EXPECTED_BUCKETS:
            assert snap[bucket]["count"] == 0
            assert snap[bucket]["ms"] == 0.0


# Subprocess so `SCX_CPU_PROFILE=1` is set BEFORE pyscx is imported (the Rust gate
# is OnceLock-cached at first profiler call). This exercises the reader-hook value
# routing end-to-end — the piece the in-process shape test above cannot reach.
_SUBPROC = r"""
import os, sys, tempfile
import numpy as np
import anndata
from scipy import sparse
import pyscx

rng = np.random.default_rng(0)
X = sparse.random(400, 60, density=0.2, format="csr", dtype="float32", random_state=0)
X.data = np.rint(X.data * 20).astype("float32")  # integer-ish counts for seurat_v3
adata = anndata.AnnData(X=X)
path = os.path.join(tempfile.mkdtemp(), "t.scx")
pyscx.from_anndata(adata, path)

backed = pyscx.open(path).to_anndata(backed=True)
pyscx.accel.cpu_profile_reset()
pyscx.accel.highly_variable_genes(backed, n_top_genes=20, flavor="seurat_v3", device="cpu")
snap = pyscx.accel.cpu_profile_snapshot()

assert snap["enabled"] is True, "profiler should be enabled in subprocess"
dec = snap["decode_scx1"]["count"] + snap["decode_generic"]["count"]
assert dec > 0, f"decode not recorded: {snap}"
assert snap["reduction"]["count"] > 0, f"reduction not recorded: {snap}"
# Value routing: decode time landed in a decode bucket, not io/marshalling.
dec_ms = snap["decode_scx1"]["ms"] + snap["decode_generic"]["ms"]
assert dec_ms > 0.0, f"decode ms not recorded: {snap}"
assert snap["marshalling"]["count"] == 0, "HVG has no embedding marshalling"
print("OK")
"""


def test_cpu_profile_records_backed_decode_and_reduction(tmp_path):
    import subprocess

    env = dict(os.environ, SCX_CPU_PROFILE="1")
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROC],
        env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "OK" in proc.stdout
