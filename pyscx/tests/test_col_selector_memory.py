"""Peak memory of the column-selector paths (REC-7, PR D), each in a fresh
process against a full `to_memory()` decode.

The contract `docs/api/python-datasets.md` states: `X[:, cols]` is a handle (no decode), and
the two forms that must materialise — repeated columns on any handle, and a
reorder on a lazily transformed handle — materialise the *projected unique
columns*, never the whole matrix. Before this, `to_memory()` on a projected
handle decoded everything and projected afterwards, so `X[:, [1, 3]].to_memory()`
peaked at the full matrix (measured: 201 MB against a 204 MB full decode on a
40-shard, 40960 × 4000 fixture).
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import scipy.sparse as sp

# Same sampler as `test_row_gather.py`: `RssAnon` so the mmap'd file pages do
# not count, a 1 ms sampler thread, per-mode peak over a trimmed baseline.
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
    import numpy as np
    import pyscx
    exp = pyscx.open(path)
    adata = exp.to_anndata(backed=True)
    X = adata.X
    result = {"mode": mode, "n_obs": exp.n_obs, "nnz": exp.nnz, "n_shards": exp.shard_count}
    if mode == "full":
        result["peak"] = measure(lambda: X.to_memory())
    elif mode == "handle":
        result["peak"] = measure(lambda: X[:, [7, 2, 11]])
    elif mode == "projected_read":
        result["peak"] = measure(lambda: X[:, [7, 2, 11]].to_memory())
    elif mode == "repeats":
        result["peak"] = measure(lambda: X[:, [3, 1, 3]])
    elif mode == "wide_reorder":
        result["peak"] = measure(lambda: X[:, ::-1].to_memory())
    elif mode == "lazy_reorder":
        pyscx.accel.normalize_total(adata)
        pyscx.accel.log1p(adata)
        L = adata.X
        assert isinstance(L, pyscx.ScxLazyTransformedDataset)
        result["peak"] = measure(lambda: L[:, [7, 2]])
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


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self/status")
def test_column_selector_paths_never_decode_the_whole_matrix(tmp_dir):
    """A projected handle costs nothing; reading it, a repeated-column
    selector and a lazy reorder each cost the projected columns plus one
    shard's transient — a small fraction of the full decode."""
    import anndata
    import pyscx

    n_shards, shard_size, n_vars, density = 40, 512, 2000, 0.1
    n_obs = n_shards * shard_size
    rng = np.random.default_rng(0)
    x = sp.random(n_obs, n_vars, density=density, format="csr", random_state=rng, dtype=np.float32)
    x.data = np.ceil(x.data * 10).astype(np.float32)
    adata = anndata.AnnData(X=x)
    adata.obs_names = [f"c{i}" for i in range(n_obs)]
    adata.var_names = [f"g{i}" for i in range(n_vars)]
    path = str(tmp_dir / "col_peak.scx")
    pyscx.from_anndata(adata, path, shard_size=shard_size)
    exp = pyscx.open(path)
    assert exp.shard_count == n_shards

    full = _probe(path, "full")["peak"]
    handle = _probe(path, "handle")["peak"]
    projected_read = _probe(path, "projected_read")["peak"]
    repeats = _probe(path, "repeats")["peak"]
    lazy_reorder = _probe(path, "lazy_reorder")["peak"]
    wide_reorder = _probe(path, "wide_reorder")["peak"]

    shard_bytes = exp.nnz // n_shards * 8 + (shard_size + 1) * 8
    report = (
        f"full={full / 1e6:.1f} MB handle={handle / 1e6:.2f} MB "
        f"projected_read={projected_read / 1e6:.1f} MB repeats={repeats / 1e6:.1f} MB "
        f"lazy_reorder={lazy_reorder / 1e6:.1f} MB wide_reorder={wide_reorder / 1e6:.1f} MB "
        f"(shard {shard_bytes / 1e6:.2f} MB)"
    )
    # A handle is bookkeeping only.
    assert handle <= 1_000_000, report
    # Three columns of 2000: the result is ~0.15 % of the matrix, so each
    # materialising path is bounded by a few shards' worth of transient, and
    # is a small fraction of the full decode (it was ~1.0× before).
    for peak in (projected_read, repeats, lazy_reorder):
        assert peak <= 0.35 * full, report
        assert peak <= 4 * shard_bytes + 4_000_000, report
    # A presentation-ordered projection as wide as the matrix (`X[:, ::-1]`)
    # is the case where the reorder used to add a third result-sized buffer on
    # top of the pieces and the concatenation (~2.9× a plain `to_memory()`,
    # found by codex). Reordering per piece keeps it at the two.
    assert wide_reorder <= 2.25 * full + 2 * shard_bytes + 4_000_000, report
