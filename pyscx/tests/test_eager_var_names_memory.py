"""Peak memory of the eager `to_anndata(var_names=...)` path, each mode in a
fresh process.

These are the bars that keep the projection a projection. Before it, the eager
gene path assembled the whole matrix and handed it to anndata to slice:
`to_anndata_with_layers(..., eager=true)` built full-width X **and** every
selected layer, and `adata[:, idx].copy()` then made a second copy of all of it.
Asking for three genes of two thousand cost *more* than reading the whole file —
the default `to_anndata()` at least leaves layers as a lazy bridge, which
`var_names=` forces eager.

`.raw` was the same story once removed: attached to the full AnnData, so the view
copied it and `_mutated_copy` copied it again — three full-width buffers for a
matrix a gene projection does not touch at all (anndata never var-slices
`.raw`).

Sampler, env pinning and fixture geometry are copied from
`test_col_selector_memory.py` so the two harnesses stay comparable: `RssAnon` so
the mmap'd file pages do not count, a 1 ms sampler thread, per-mode peak over a
trimmed baseline.
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import scipy.sparse as sp

_PROBE = textwrap.dedent(
    """
    import ctypes, gc, json, sys, threading, time
    sys.setswitchinterval(1e-3)

    def rss_anon():
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
        raise RuntimeError("no RssAnon in /proc/self/status")

    class Sampler:
        def __init__(self):
            self.peak = 0
            self._stop = threading.Event()
        def _run(self):
            while not self._stop.is_set():
                self.peak = max(self.peak, rss_anon())
                time.sleep(0.001)
            self.peak = max(self.peak, rss_anon())
        def __enter__(self):
            self.peak = rss_anon()
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
            return self
        def __exit__(self, *a):
            self._stop.set()
            self._t.join()

    def trim():
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass

    def measure(fn):
        trim()
        base = rss_anon()
        with Sampler() as s:
            out = fn()
        peak = s.peak - base
        del out
        trim()
        return peak

    path, mode = sys.argv[1], sys.argv[2]
    import pyscx
    exp = pyscx.open(path)
    n_vars = exp.n_vars
    result = {"mode": mode, "n_obs": exp.n_obs, "nnz": exp.nnz,
              "n_shards": exp.shard_count, "n_vars": n_vars}
    few = ["g7", "g2", "g11"]
    if mode == "full_default":
        result["peak"] = measure(lambda: exp.to_anndata())
    elif mode == "full_eager":
        result["peak"] = measure(lambda: exp.to_anndata(eager=True))
    elif mode == "proj":
        result["peak"] = measure(lambda: exp.to_anndata(var_names=few))
    elif mode == "proj_no_layers":
        result["peak"] = measure(lambda: exp.to_anndata(var_names=few, layers=[]))
    elif mode == "proj_ordered":
        result["peak"] = measure(
            lambda: exp.to_anndata(var_names=["g11", "g2", "g7"], preserve_var_order=True)
        )
    elif mode == "proj_wide":
        wide = [f"g{i}" for i in range(n_vars - 1, -1, -1)]
        result["peak"] = measure(
            lambda: exp.to_anndata(var_names=wide, preserve_var_order=True, layers=[])
        )
    elif mode == "full_dense":
        result["peak"] = measure(
            lambda: exp.to_anndata(container="dense", data_dtype="uint8", layers=[])
        )
    elif mode == "proj_dense":
        result["peak"] = measure(
            lambda: exp.to_anndata(
                var_names=few, container="dense", data_dtype="uint8", layers=[]
            )
        )
    elif mode == "raw_full":
        result["peak"] = measure(lambda: exp.to_anndata())
    elif mode == "raw_proj":
        result["peak"] = measure(lambda: exp.to_anndata(var_names=few))
    elif mode == "raw_off":
        result["peak"] = measure(lambda: exp.to_anndata(var_names=few, raw=False))
    else:
        raise SystemExit(f"unknown mode {mode}")
    print(json.dumps(result))
    """
)


def _probe(path, mode):
    env = dict(os.environ)
    env.update(
        MALLOC_ARENA_MAX="2",
        MALLOC_MMAP_THRESHOLD_="131072",
        RAYON_NUM_THREADS="2",
    )
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, path, mode],
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode == 0, f"probe {mode} failed:\n{out.stdout}\n{out.stderr}"
    return json.loads(out.stdout.strip().splitlines()[-1])


def _write_fixture(tmp_dir, name, *, n_shards, shard_size, n_vars, density, layer, raw_n_vars):
    """A multi-shard file with integer values, optionally a layer and a raw."""
    import anndata
    import pandas as pd

    import pyscx

    n_obs = n_shards * shard_size
    rng = np.random.default_rng(0)
    x = sp.random(n_obs, n_vars, density=density, format="csr", random_state=rng, dtype=np.float32)
    x.data = np.ceil(x.data * 10).astype(np.float32)
    layers = {"counts": (x * 2).tocsr()} if layer else None
    adata = anndata.AnnData(X=x, layers=layers)
    adata.obs_names = [f"c{i}" for i in range(n_obs)]
    adata.var_names = [f"g{i}" for i in range(n_vars)]
    if raw_n_vars:
        raw = sp.random(
            n_obs, raw_n_vars, density=density, format="csr", random_state=rng, dtype=np.float32
        )
        raw.data = np.ceil(raw.data * 10).astype(np.float32)
        raw_var = pd.DataFrame(index=[f"r{i}" for i in range(raw_n_vars)])
        adata.raw = anndata.AnnData(X=raw, var=raw_var, obs=adata.obs)
    path = str(tmp_dir / name)
    pyscx.from_anndata(adata, path, shard_size=shard_size)
    return path


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self/status")
def test_var_names_never_assembles_the_full_width_matrix(tmp_dir):
    """Three genes of 2000 must cost a fraction of reading the whole file.

    Before this, `var_names=` peaked *above* a plain `to_anndata()`: it forced
    layers eager on top of X, then anndata copied both.
    """
    import pyscx

    n_shards, shard_size, n_vars, density = 40, 512, 2000, 0.1
    path = _write_fixture(
        tmp_dir,
        "var_names_peak.scx",
        n_shards=n_shards,
        shard_size=shard_size,
        n_vars=n_vars,
        density=density,
        layer=True,
        raw_n_vars=0,
    )
    exp = pyscx.open(path)
    # Premises. Without these the ratios below are measuring nothing.
    assert exp.shard_count == n_shards
    assert exp.layer_names() == ["counts"]

    full_default = _probe(path, "full_default")["peak"]
    full_eager = _probe(path, "full_eager")["peak"]
    proj = _probe(path, "proj")["peak"]
    proj_no_layers = _probe(path, "proj_no_layers")["peak"]
    proj_ordered = _probe(path, "proj_ordered")["peak"]
    proj_wide = _probe(path, "proj_wide")["peak"]

    shard_bytes = exp.nnz // n_shards * 8 + (shard_size + 1) * 8
    report = (
        f"full_default={full_default / 1e6:.1f} MB full_eager={full_eager / 1e6:.1f} MB "
        f"proj={proj / 1e6:.1f} MB proj_no_layers={proj_no_layers / 1e6:.1f} MB "
        f"proj_ordered={proj_ordered / 1e6:.1f} MB proj_wide={proj_wide / 1e6:.1f} MB "
        f"(shard {shard_bytes / 1e6:.2f} MB)"
    )

    # The headline: a three-gene request must not cost more than the whole file.
    #
    # The bars are set from measurement, not from the arithmetic of the result:
    # before this, `proj` was ~74 MB (1.8x `full_default`); after, it lands
    # between 8.9 and 10.4 MB across a quiet box and a fully loaded one. The
    # residue over the ~0.2 MB projected result is a whole `to_anndata()`'s
    # fixed cost — 20 480 obs-index Python strings, the obs frame, and the
    # skeleton's copy of both — which no gene projection can remove. So the
    # bounds sit near 20 MB: still red by 3.6x before, still ~2x of headroom
    # after.
    assert proj <= 0.5 * full_default, report
    assert proj <= 0.3 * full_eager, report
    # The absolute bar, in the shape test_col_selector_memory.py uses, widened
    # by that fixed cost (that harness measures `X.to_memory()` on an
    # already-open AnnData and never pays it).
    assert proj <= 8 * shard_bytes + 16_000_000, report
    assert proj_ordered <= 8 * shard_bytes + 16_000_000, report
    # Which half regressed: the layer must add only a shard's transient.
    assert proj <= proj_no_layers + 4 * shard_bytes + 8_000_000, report
    assert proj_no_layers <= 0.5 * full_default, report
    # Guardrail, not a red bar: a presentation-ordered projection as wide as the
    # matrix is the case where reordering can add a third result-sized buffer on
    # top of the pieces and the concatenation. Measured at 2.6x full_default
    # before this change; a third buffer would put it past 3.4x.
    assert proj_wide <= 3.0 * full_default + 2 * shard_bytes + 4_000_000, report


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self/status")
def test_var_names_dense_container_narrows_too(tmp_dir):
    """`container="dense"` is the worst peak in the API — a full n_obs x n_vars
    slab. A gene projection has to reach it too."""
    import pyscx

    n_shards, shard_size, n_vars, density = 40, 512, 2000, 0.1
    path = _write_fixture(
        tmp_dir,
        "var_names_dense_peak.scx",
        n_shards=n_shards,
        shard_size=shard_size,
        n_vars=n_vars,
        density=density,
        layer=False,
        raw_n_vars=0,
    )
    exp = pyscx.open(path)
    assert exp.shard_count == n_shards
    # uint8 holds every value here, so the decode-loss guard passes and the
    # dense slab stays at n_obs * n_vars bytes rather than four times that.
    assert exp.max_value <= 255

    full_dense = _probe(path, "full_dense")["peak"]
    proj_dense = _probe(path, "proj_dense")["peak"]
    report = f"full_dense={full_dense / 1e6:.1f} MB proj_dense={proj_dense / 1e6:.1f} MB"
    # 48.8 MB before (the projection bought nothing), ~8.8 MB after.
    assert proj_dense <= 0.35 * full_dense, report


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self/status")
def test_var_names_does_not_multiply_raw(tmp_dir):
    """A gene projection never touches `.raw` (anndata slices raw on obs only),
    so projecting must not cost more raw than a plain read does.

    It used to cost three times as much: raw was attached to the full AnnData,
    the view copied it, and `_mutated_copy` copied it again.
    """
    import pyscx

    n_shards, shard_size, n_vars, raw_n_vars, density = 20, 512, 1000, 2000, 0.1
    path = _write_fixture(
        tmp_dir,
        "var_names_raw_peak.scx",
        n_shards=n_shards,
        shard_size=shard_size,
        n_vars=n_vars,
        density=density,
        layer=False,
        raw_n_vars=raw_n_vars,
    )
    exp = pyscx.open(path)
    assert exp.shard_count == n_shards
    # Premise: raw is on a wider gene axis and survives the read, so the arms
    # below are actually weighing a raw matrix.
    baseline = exp.to_anndata()
    assert baseline.raw is not None and baseline.raw.shape[1] == raw_n_vars
    raw_bytes = baseline.raw.X.nnz * 8 + (baseline.n_obs + 1) * 8
    del baseline

    raw_full = _probe(path, "raw_full")["peak"]
    raw_proj = _probe(path, "raw_proj")["peak"]
    raw_off = _probe(path, "raw_off")["peak"]
    shard_bytes = exp.nnz // n_shards * 8 + (shard_size + 1) * 8
    report = (
        f"raw_full={raw_full / 1e6:.1f} MB raw_proj={raw_proj / 1e6:.1f} MB "
        f"raw_off={raw_off / 1e6:.1f} MB (raw {raw_bytes / 1e6:.1f} MB, "
        f"shard {shard_bytes / 1e6:.2f} MB)"
    )

    # Projecting must not cost meaningfully more than reading everything.
    assert raw_proj <= 1.4 * raw_full, report
    # And it must be raw, not X, that dominates what is left.
    assert raw_proj <= 1.8 * raw_bytes + 4 * shard_bytes + 8_000_000, report
    # raw=False (PR H) is the way out of paying for raw at all.
    assert raw_off <= 0.4 * raw_proj, report


def test_var_names_shrinks_the_eager_assembly_estimate(tmp_dir):
    """The memory-budget warning must describe the assembly that will happen.

    `EagerAssemblyMemoryHigh` tells the caller to "pass a smaller `var_names` /
    obs_filter subset ... to reduce it". Before the projection landed, taking
    that advice changed nothing: the estimate folded whole-file catalog nnz and
    ignored `var_names` entirely, so the same warning fired either way.

    Deterministic and in-process — this is the arm that keeps meaning something
    on a CI box under memory pressure, where the sampler arms above are the ones
    at risk.
    """
    import warnings

    import pyscx

    path = _write_fixture(
        tmp_dir,
        "estimate.scx",
        n_shards=4,
        shard_size=128,
        n_vars=2000,
        density=0.1,
        layer=False,
        raw_n_vars=0,
    )
    exp = pyscx.open(path)
    # A budget under the full estimate and (comfortably) over 1/2000th of it.
    # 8 B/nnz is what the assembled CSR costs below 2**31 nonzeros (f32 values +
    # int32 column indices); above that scipy holds int64 indices and it is 12.
    # Deriving it rather than hard-coding a constant keeps this arm honest if
    # the per-nonzero cost moves again.
    budget = exp.nnz * 8 // 2

    with pytest.warns(UserWarning, match="eager_assembly_memory_high"):
        pyscx.open(path).to_anndata(memory_budget=budget)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=r".*eager_assembly_memory_high.*", category=UserWarning
        )
        out = pyscx.open(path).to_anndata(var_names=["g0"], memory_budget=budget)
    assert out.n_vars == 1
