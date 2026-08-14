"""§8.11 — GPU PCA residency, and the `spmm_policy` that used to be dropped on
the streaming arm.

GPU PCA picks between two power loops: a device-resident one that uploads the
matrix once, and a streaming operator that re-decodes and re-uploads the whole
matrix on every multiply. The choice is made dynamically against *free* VRAM at
call time, so the same script on the same data can go either way depending on
what else is on the card.

That mattered because only the resident loop honoured `spmm_policy`. The
streaming operator hardcoded `CUSPARSE_SPMM_ALG_DEFAULT` — an algorithm that
may use atomics — while `uns["scx_accel"]["pca"]["spmm_policy"]` reported
whatever the caller asked for. A user who requested `"deterministic"` on a
matrix too big for VRAM got nondeterminism and was told otherwise.

What these tests pin:

1. **The two paths are distinguishable.** `resident_csr` is the only signal;
   the embeddings are supposed to agree, so a silent fall back to streaming is
   otherwise invisible. Same reasoning as `resident_csr` on the GPU DE routes.
2. **The streaming arm is reachable at all.** Without `SCX_GPU_PCA_RESIDENT=0`
   no fixture can force it on an 80 GB card, which is why the branch that
   ignored `spmm_policy` was never exercised by a test.
3. **`spmm_policy="deterministic"` survives the streaming arm**, both as a
   completed run (cuSPARSE accepts `CSR_ALG2` for the transpose multiply too)
   and as an accurate stamp.

Note what is *not* claimed: nothing here observes which cuSPARSE algorithm
actually executed — the host cannot. That the streaming operator passes the
policy through is guaranteed structurally, by there being no algorithm-less
strided-SpMM entry point left to call.

The env knob is read once per process (`OnceLock`), so each arm runs in a
subprocess — the same reason `test_gpu_de_resident.py` does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import pyscx

pytestmark = pytest.mark.skipif(
    not pyscx.accel.gpu_available(),
    reason="GPU not available (pyscx not built with gpu feature, or no CUDA device)",
)

N_OBS = 400
N_VARS = 120
SHARD_SIZE = 80  # → 5 shards, so the streaming arm really does stream
N_COMPS = 5


@pytest.fixture(scope="module")
def scx_path(tmp_path_factory):
    import anndata as ad

    rng = np.random.default_rng(11)
    x = rng.poisson(2.0, size=(N_OBS, N_VARS)).astype(np.float32)
    x[rng.random((N_OBS, N_VARS)) > 0.15] = 0.0
    adata = ad.AnnData(X=x)
    adata.obs_names = [f"c{i}" for i in range(N_OBS)]
    adata.var_names = [f"g{i}" for i in range(N_VARS)]
    path = tmp_path_factory.mktemp("gpu_pca_resident") / "counts.scx"
    pyscx.from_anndata(adata, str(path), shard_size=SHARD_SIZE)
    return path


# Both arms run this: PCA on the backed handle, then dump the route stamp and
# the embedding on stdout.
_ARM = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    import pyscx

    path, policy = sys.argv[1], sys.argv[2]
    adata = pyscx.open(path).to_anndata(backed=True)
    pyscx.accel.pca(
        adata,
        n_comps={n_comps},
        device="gpu",
        method="randomized",
        spmm_policy=policy,
    )
    info = adata.uns["scx_accel"]["pca"]
    out = {{
        "route": info["route"],
        "resident_csr": info["resident_csr"],
        "spmm_policy": info["spmm_policy"],
        "math_mode": info["math_mode"],
        "n_shards": int(pyscx.open(path).shard_count),
        "embedding": [float(v) for v in np.asarray(adata.obsm["X_pca"]).ravel()],
    }}
    print("@@JSON@@" + json.dumps(out))
    """
).format(n_comps=N_COMPS)


def _run_arm(path, policy: str, resident: bool) -> dict:
    env = dict(os.environ)
    env["SCX_GPU_PCA_RESIDENT"] = "1" if resident else "0"
    # Orthogonal to residency and flaky when GPU test processes overlap.
    env["SCX_DISABLE_CUDA_GRAPHS"] = "1"
    # The native streaming/resident loops are the subject; rapids would route
    # around both.
    env["SCX_FORCE_NATIVE_GPU"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _ARM, str(path), policy],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert proc.returncode == 0, (
        f"policy={policy} arm (resident={resident}) failed:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("@@JSON@@"))
    return json.loads(line[len("@@JSON@@") :])


def test_residency_is_recorded_and_the_kill_switch_forces_streaming(scx_path):
    on = _run_arm(scx_path, "default", resident=True)
    off = _run_arm(scx_path, "default", resident=False)

    # --- premises ---
    assert on["route"] == "gpu_csr", (
        f"premise: the fixture must take the native GPU PCA route, got "
        f"{on['route']!r}; a rapids or CPU route has no residency decision"
    )
    assert off["route"] == on["route"], "the kill switch must not change the route"
    assert on["n_shards"] > 1, (
        f"premise: the streaming arm must have more than one shard to stream, "
        f"got {on['n_shards']}"
    )

    # --- the claim ---
    assert on["resident_csr"] is True, (
        "residency silently fell back to streaming on a fixture that fits VRAM "
        "many times over"
    )
    assert off["resident_csr"] is False, "SCX_GPU_PCA_RESIDENT=0 must disable it"

    # Same subspace either way — residency is an optimisation, not a different
    # algorithm. Compare |cosine| per component: the sign of a principal
    # component is arbitrary.
    a = np.asarray(on["embedding"]).reshape(N_OBS, N_COMPS)
    b = np.asarray(off["embedding"]).reshape(N_OBS, N_COMPS)
    for pc in range(N_COMPS):
        u, v = a[:, pc], b[:, pc]
        denom = np.linalg.norm(u) * np.linalg.norm(v)
        assert denom > 0, f"component {pc} is all zeros in one arm"
        cos = abs(float(np.dot(u, v)) / denom)
        assert cos > 0.99, (
            f"component {pc} diverged between the resident and streaming loops: "
            f"|cos| = {cos:.6f}"
        )


def test_streaming_arm_honours_and_reports_deterministic_spmm(scx_path):
    """The streaming path must both *run* and *report* the requested policy.

    Before the fix this arm ran `CUSPARSE_SPMM_ALG_DEFAULT` and still stamped
    `"deterministic"`. The stamp assertion alone cannot catch that — it was
    already passing — so the load-bearing half is that the run completes at all:
    it is the first exercise of `CUSPARSE_SPMM_CSR_ALG2` on this operator, and
    in particular under `CUSPARSE_OPERATION_TRANSPOSE`.
    """
    streamed = _run_arm(scx_path, "deterministic", resident=False)

    assert streamed["resident_csr"] is False, (
        "premise: this arm must be the streaming one, or it re-tests the "
        "resident loop that already honoured the policy"
    )
    assert streamed["spmm_policy"] == "deterministic"
    assert streamed["math_mode"] == "strict_fp32"

    # And it agrees with the heuristic algorithm on the same input.
    heuristic = _run_arm(scx_path, "default", resident=False)
    a = np.asarray(streamed["embedding"]).reshape(N_OBS, N_COMPS)
    b = np.asarray(heuristic["embedding"]).reshape(N_OBS, N_COMPS)
    for pc in range(N_COMPS):
        u, v = a[:, pc], b[:, pc]
        cos = abs(float(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v)))
        assert cos > 0.99, (
            f"component {pc} diverged between the deterministic and heuristic "
            f"SpMM algorithms: |cos| = {cos:.6f}"
        )
