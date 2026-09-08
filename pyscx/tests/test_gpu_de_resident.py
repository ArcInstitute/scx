"""Task 4.5 / §9.11 — device-resident CSR for the GPU DE CSR routes.

The GPU CSR DE routes have no column-range prefilter, so before 4.5 they ran a
full shard pass per gene chunk: `n_gene_chunks × n_shards` host decodes and H→D
uploads. Residency drains the source once and serves every later chunk from
VRAM.

What these tests pin:

1. **Residency actually engages.** A route stamp is the only way to tell a
   resident run from a streaming one — the results are supposed to be the same,
   so a silent fall back to streaming is invisible in the output. This is the
   same reasoning behind the `de_route_csc_direct` gate.
2. **It changes nothing observable.** `SCX_GPU_DE_RESIDENT=0` and the default
   must agree on every field.
3. **The premises hold.** A fixture with one gene chunk, or one shard, or a CSC
   sidecar would make (1) and (2) vacuous, so each is asserted rather than
   assumed.

The env knob is read once per process (`OnceLock`), so the off arm runs in a
subprocess — the same reason `test_prefetch_equivalence.py` does.
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

N_OBS = 240
N_VARS = 40
SHARD_SIZE = 40  # → 6 shards
GENE_CHUNK = 8  # → 5 gene chunks


def _make_counts(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.poisson(1.2, size=(N_OBS, N_VARS)).astype(np.float32)
    # Guarantee every gene chunk has signal in both groups, so no chunk is
    # trivially zero and the comparison has something to compare.
    x[:, ::GENE_CHUNK] += 1.0
    return x


@pytest.fixture(scope="module")
def scx_path(tmp_path_factory):
    import anndata as ad
    import pandas as pd

    adata = ad.AnnData(X=_make_counts())
    adata.obs_names = [f"c{i}" for i in range(N_OBS)]
    adata.var_names = [f"g{i}" for i in range(N_VARS)]
    adata.obs["grp"] = pd.Categorical(
        ["a" if i % 2 == 0 else "b" for i in range(N_OBS)]
    )
    path = tmp_path_factory.mktemp("gpu_de_resident") / "counts.scx"
    # No `csc=` → CSR-only, which is what forces the `gpu_csr_v3` route.
    pyscx.from_anndata(adata, str(path), shard_size=SHARD_SIZE)
    return path


# The body both arms run: DE on the backed handle, then dump the route stamp
# plus the result arrays as JSON on stdout.
_ARM = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    import pyscx

    path, op = sys.argv[1], sys.argv[2]
    adata = pyscx.open(path).to_anndata(backed=True)

    if op == "wilcoxon":
        pyscx.accel.rank_genes_groups(
            adata, "grp", device="gpu", gene_chunk_size={chunk}
        )
        info = adata.uns["scx_accel"]["rank_genes_groups"]
        res = adata.uns["rank_genes_groups"]
        out = {{
            "route": info["route"],
            "resident_csr": info["resident_csr"],
            "chunk_size": info["chunk_size"],
            "names": [list(map(str, res["names"][f])) for f in res["names"].dtype.names],
            "pvals": [
                list(map(float, res["pvals"][f]))
                for f in res["pvals"].dtype.names
            ],
            "scores": [
                list(map(float, res["scores"][f]))
                for f in res["scores"].dtype.names
            ],
        }}
    else:
        df = pyscx.accel.pdex_ref(
            adata, "grp", reference="a", device="gpu", gene_chunk_size={chunk}
        )
        info = adata.uns["scx_accel"]["pdex_ref"]
        out = {{
            "route": info["route"],
            "resident_csr": info["resident_csr"],
            "chunk_size": info["chunk_size"],
            # Iterate the Series directly: `pdex_ref` defaults to pandas (F6)
            # and pandas spells `.to_list()` as `.tolist()`.
            "feature": [str(v) for v in df["feature"]],
            "p_value": [float(v) for v in df["p_value"]],
            "fold_change": [float(v) for v in df["fold_change"]],
            "target_mean": [float(v) for v in df["target_mean"]],
        }}
    print("@@JSON@@" + json.dumps(out))
    """
).format(chunk=GENE_CHUNK)


def _run_arm(path, op: str, resident: bool) -> dict:
    env = dict(os.environ)
    env["SCX_GPU_DE_RESIDENT"] = "1" if resident else "0"
    # CUDA graph capture is orthogonal here and flaky when several GPU test
    # processes overlap; pin it off so the two arms differ only in residency.
    env["SCX_DISABLE_CUDA_GRAPHS"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _ARM, str(path), op],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert proc.returncode == 0, (
        f"{op} arm (resident={resident}) failed:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    line = next(
        ln for ln in proc.stdout.splitlines() if ln.startswith("@@JSON@@")
    )
    return json.loads(line[len("@@JSON@@") :])


@pytest.mark.parametrize("op", ["wilcoxon", "pdex_ref"])
def test_residency_engages_and_changes_nothing(scx_path, op):
    on = _run_arm(scx_path, op, resident=True)
    off = _run_arm(scx_path, op, resident=False)

    # --- premises: without these the comparison below proves nothing ---
    assert on["route"] == "gpu_csr_v3", (
        f"premise: the fixture must take the CSR route, got {on['route']!r}. "
        "A CSC sidecar would prefilter by column range and never re-decode, "
        "so residency would correctly decline and this test would be vacuous."
    )
    assert off["route"] == on["route"], "the kill switch must not change the route"
    n_chunks = -(-N_VARS // on["chunk_size"])
    assert n_chunks > 1, (
        f"premise: residency only engages with >1 gene chunk; the effective "
        f"chunk size was {on['chunk_size']} over {N_VARS} genes"
    )

    # --- the claim ---
    assert on["resident_csr"] is True, (
        "residency silently fell back to streaming on a fixture that should fit "
        "VRAM many times over"
    )
    assert off["resident_csr"] is False, "SCX_GPU_DE_RESIDENT=0 must disable it"

    for key in on:
        if key == "resident_csr":
            continue
        if isinstance(on[key], str) or key == "chunk_size":
            assert on[key] == off[key], f"{key} differs between arms"
            continue
        got = np.asarray(on[key])
        want = np.asarray(off[key])
        assert got.shape == want.shape, f"{key}: shape differs between arms"
        if got.dtype.kind in "UO":
            np.testing.assert_array_equal(got, want, err_msg=f"{key} differs")
        else:
            # The pseudobulk fold is an f64 atomicAdd whose ordering is already
            # run-to-run nondeterministic, so means / fold changes are compared
            # to tolerance rather than bit-exactly. Statistics and p-values come
            # from the single-writer scatter slabs and are exact.
            exact = key in ("pvals", "scores", "p_value")
            if exact:
                np.testing.assert_allclose(
                    got, want, rtol=1e-9, atol=1e-12, err_msg=f"{key} differs"
                )
            else:
                np.testing.assert_allclose(
                    got, want, rtol=1e-5, atol=1e-7, err_msg=f"{key} differs"
                )


def test_single_chunk_declines_residency(scx_path):
    """With one gene chunk the streaming pass runs exactly once anyway, so
    retaining the matrix would be pure cost. Residency must decline."""
    adata = pyscx.open(str(scx_path)).to_anndata(backed=True)
    pyscx.accel.rank_genes_groups(
        adata, "grp", device="gpu", gene_chunk_size=N_VARS * 4
    )
    info = adata.uns["scx_accel"]["rank_genes_groups"]
    assert info["route"] == "gpu_csr_v3"
    assert info["chunk_size"] >= N_VARS, "premise: a single gene chunk"
    assert info["resident_csr"] is False


@pytest.fixture(scope="module")
def window_path(tmp_path_factory):
    """The first `SHARD_SIZE` rows as a file of their own — the value oracle for
    a row-windowed run, on the same backend so the tolerance stays tight."""
    import anndata as ad
    import pandas as pd

    adata = ad.AnnData(X=_make_counts()[:SHARD_SIZE])
    adata.obs_names = [f"c{i}" for i in range(SHARD_SIZE)]
    adata.var_names = [f"g{i}" for i in range(N_VARS)]
    adata.obs["grp"] = pd.Categorical(
        ["a" if i % 2 == 0 else "b" for i in range(SHARD_SIZE)]
    )
    path = tmp_path_factory.mktemp("gpu_de_window") / "window.scx"
    pyscx.from_anndata(adata, str(path), shard_size=SHARD_SIZE)
    return path


# Host decode is what the staging plan moves, and it is *not* `shards_decoded`:
# that counter lives in the `for_each_gpu_csr_shard` consumer, which
# `drive_shards` reaches only after `is_stageable` (`n_rows() != 0`) has already
# dropped the empty shards — on the base commit too. Comparing it would pass
# with `StagingPlan::for_source` reverted. The decode counter behind
# `SCX_CPU_PROFILE` sits in the reader, below the feeder, so it sees the reads
# the plan actually removes.
_DECODE_ARM = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    import pyscx

    path, op, window = sys.argv[1], sys.argv[2], sys.argv[3] == "window"
    adata = pyscx.open(path).to_anndata(backed=True, cache_shards=0)
    n_shards = adata.X.n_shards
    if window:
        adata = adata[:{shard}]

    pyscx.accel.cpu_profile_reset()
    if op == "wilcoxon":
        pyscx.accel.rank_genes_groups(
            adata, "grp", device="gpu", gene_chunk_size={chunk}
        )
        res = adata.uns["rank_genes_groups"]
        info = adata.uns["scx_accel"]["rank_genes_groups"]
        values = {{
            "names": [list(map(str, res["names"][f])) for f in res["names"].dtype.names],
            "scores": [
                list(map(float, res["scores"][f])) for f in res["scores"].dtype.names
            ],
        }}
    else:
        df = pyscx.accel.pdex_ref(
            adata, "grp", reference="a", device="gpu", gene_chunk_size={chunk}
        )
        info = adata.uns["scx_accel"]["pdex_ref"]
        values = {{
            "feature": [str(v) for v in df["feature"]],
            "p_value": [float(v) for v in df["p_value"]],
            "target_mean": [float(v) for v in df["target_mean"]],
        }}
    snap = pyscx.accel.cpu_profile_snapshot()
    print("@@JSON@@" + json.dumps({{
        "enabled": snap["enabled"],
        "decodes": snap["decode_scx1"]["count"] + snap["decode_generic"]["count"],
        "n_shards": n_shards,
        "route": info["route"],
        "values": values,
    }}))
    """
).format(chunk=GENE_CHUNK, shard=SHARD_SIZE)


def _decode_arm(path, op: str, window: bool) -> dict:
    env = dict(os.environ)
    # The profiler gate is a `OnceLock` read, so it must be set before import.
    env["SCX_CPU_PROFILE"] = "1"
    env["SCX_DISABLE_CUDA_GRAPHS"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _DECODE_ARM, str(path), op, "window" if window else "all"],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert proc.returncode == 0, (
        f"{op} decode arm (window={window}) failed:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("@@JSON@@"))
    res = json.loads(line[len("@@JSON@@") :])
    assert res["enabled"], "SCX_CPU_PROFILE did not take — the count is not a count"
    return res


@pytest.mark.parametrize("op", ["wilcoxon", "pdex_ref"])
def test_a_row_window_decodes_only_the_shards_it_keeps(scx_path, op):
    """The GPU staging plan honours a row-filtering source's shard plan.

    `RawGpuShardSource::run` built `StagingPlan::all(n_shards)`, so a
    row-windowed handle host-decoded every shard and `drive_shards` dropped the
    empty ones at `is_stageable`, after the decode was paid for. It now builds
    the plan with `StagingPlan::for_source`.

    Asserted as a *ratio*, so it holds either side of the residency decision:
    residency drains the source once, streaming drains it once per gene chunk.
    Either way a one-of-six-shard window must cost a sixth.
    """
    n_shards = N_OBS // SHARD_SIZE
    assert n_shards > 1, "premise: a single-shard file cannot show a skip"

    full = _decode_arm(scx_path, op, window=False)
    win = _decode_arm(scx_path, op, window=True)

    assert full["n_shards"] == n_shards
    assert full["route"] == "gpu_csr_v3", (
        f"premise: the fixture must take the CSR staging route, got "
        f"{full['route']!r} — the CSC route prefilters by column range and never "
        "consults the row plan."
    )
    assert win["route"] == "gpu_csr_v3", (
        f"premise: a row-windowed handle must stay on the CSR route, got "
        f"{win['route']!r}"
    )
    assert full["decodes"] >= n_shards, (
        f"premise: the unwindowed run must decode every shard, got "
        f"{full['decodes']}"
    )
    assert win["decodes"] == full["decodes"] // n_shards, (
        f"a one-of-{n_shards}-shard row window host-decoded {win['decodes']} "
        f"shards against {full['decodes']} unwindowed; the projection empties "
        f"the other {n_shards - 1}"
    )
    assert win["decodes"] >= 1, "the window keeps rows, so it must decode something"


@pytest.mark.parametrize("op", ["wilcoxon", "pdex_ref"])
def test_a_row_window_gives_the_same_answer_as_those_rows_alone(
    scx_path, window_path, op
):
    """Skipping a shard must not move a row.

    The GPU CSR passes scatter at a running count of visible rows, so a plan
    that disagreed with the rows delivered would return a fully formed, shifted
    result. The decode-count test above cannot see that; this one can. Compared
    against the same rows written as their own file and run on the same backend,
    so the tolerance stays tight rather than absorbing a CPU/GPU gap.
    """
    win = _decode_arm(scx_path, op, window=True)["values"]
    ref = _decode_arm(window_path, op, window=False)["values"]

    for key, got in win.items():
        want = ref[key]
        if key in ("names", "feature"):
            assert got == want, f"{op}: {key} differs between the window and its own file"
        else:
            np.testing.assert_allclose(
                np.asarray(got, dtype=np.float64),
                np.asarray(want, dtype=np.float64),
                rtol=1e-9,
                atol=1e-12,
                err_msg=f"{op}: {key} differs between the window and its own file",
            )


def test_cpu_route_records_no_residency_decision(scx_path):
    """`resident_csr` is `None`, not `False`, where there is no decision to
    make — a CPU route never re-decodes per chunk in the first place."""
    adata = pyscx.open(str(scx_path)).to_anndata(backed=True)
    pyscx.accel.rank_genes_groups(adata, "grp", device="cpu")
    info = adata.uns["scx_accel"]["rank_genes_groups"]
    assert info["route"].startswith("cpu_")
    assert info["resident_csr"] is None
