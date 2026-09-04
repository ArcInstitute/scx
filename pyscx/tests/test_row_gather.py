"""`handle[rows]` is the bounded row gather (PR C / REC-1).

Boolean-mask and integer-array indexing on the three sparse handles
(`ScxBackedSparseDataset`, `ScxBackedLayerDataset`, `ScxLazyTransformedDataset`)
share one selector resolver and one Rust gather (`read_row_indices`): request
order, duplicates, negative wrap, `IndexError` for out-of-range rows and for a
boolean mask of the wrong length, and a peak-memory bound measured in a fresh
process — the behaviour arc-reactor's `gather_rows` re-implemented around.
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import scipy.sparse as sp

SHARD_SIZE = 10


@pytest.fixture
def scx_path(synthetic_adata, tmp_dir):
    import pyscx

    path = str(tmp_dir / "row_gather.scx")
    pyscx.from_anndata(synthetic_adata, path, shard_size=SHARD_SIZE)
    return path


def _handles(scx_path):
    """(name, handle, reference dense array) for the three sparse handle classes."""
    import pyscx

    exp = pyscx.open(scx_path)
    backed = exp.to_anndata(backed=True)
    out = [
        ("X", backed.X, backed.X.to_memory().toarray()),
        ("layer", backed.layers["raw"], backed.layers["raw"].to_memory().toarray()),
    ]
    lazy = exp.to_anndata(backed=True)
    pyscx.accel.normalize_total(lazy)
    pyscx.accel.log1p(lazy)
    assert isinstance(lazy.X, pyscx.ScxLazyTransformedDataset)
    out.append(("lazy", lazy.X, lazy.X.to_memory().toarray()))
    return out


def test_the_three_handles_are_the_expected_classes(scx_path):
    import pyscx

    kinds = {name: type(h) for name, h, _ in _handles(scx_path)}
    assert kinds["X"] is pyscx.ScxBackedSparseDataset
    assert kinds["layer"] is pyscx.ScxBackedLayerDataset
    assert kinds["lazy"] is pyscx.ScxLazyTransformedDataset


def test_fancy_index_keeps_request_order_and_duplicates(scx_path):
    rows = [37, 3, 3, 99, 0, 64]
    for name, h, ref in _handles(scx_path):
        got = h[rows]
        assert isinstance(got, sp.csr_matrix), name
        assert got.shape == (len(rows), ref.shape[1]), name
        np.testing.assert_allclose(got.toarray(), ref[rows], rtol=1e-6, err_msg=name)
        got_arr = h[np.asarray(rows, dtype=np.int32)]
        np.testing.assert_allclose(got_arr.toarray(), ref[rows], rtol=1e-6, err_msg=name)


def test_negative_indices_wrap_once(scx_path):
    for name, h, ref in _handles(scx_path):
        np.testing.assert_allclose(
            h[np.asarray([-1, -100])].toarray(), ref[[99, 0]], rtol=1e-6, err_msg=name
        )
        with pytest.raises(IndexError, match="-101"):
            h[np.asarray([-101])]


def test_unsigned_indices_near_the_top_do_not_alias_the_last_rows(scx_path):
    for name, h, ref in _handles(scx_path):
        for v in (2**64 - 1, 2**64 - 2, 2**63):
            with pytest.raises(IndexError, match=f"{v}"):
                h[np.asarray([v], dtype=np.uint64)]
        np.testing.assert_allclose(
            h[np.asarray([99, 0], dtype=np.uint64)].toarray(), ref[[99, 0]], rtol=1e-6, err_msg=name
        )


def test_out_of_range_row_is_an_index_error_not_a_shorter_result(scx_path):
    for name, h, _ in _handles(scx_path):
        with pytest.raises(IndexError, match="10000"):
            h[[0, 10_000]]
        with pytest.raises(IndexError, match="100"):
            h[np.asarray([100])]


def test_boolean_mask_matches_and_a_wrong_length_mask_is_an_index_error(scx_path):
    for name, h, ref in _handles(scx_path):
        mask = np.zeros(h.shape[0], dtype=bool)
        mask[::3] = True
        np.testing.assert_allclose(h[mask].toarray(), ref[mask], rtol=1e-6, err_msg=name)
        # numpy's rule: a boolean index must match the axis length.
        with pytest.raises(IndexError, match="boolean row mask"):
            h[mask[:-1]]
        with pytest.raises(IndexError, match="boolean row mask"):
            h[np.ones(h.shape[0] + 1, dtype=bool)]
        # An all-False mask is a valid empty gather.
        empty = h[np.zeros(h.shape[0], dtype=bool)]
        assert empty.shape == (0, ref.shape[1]), name


def test_float_selector_is_an_index_error(scx_path):
    for name, h, _ in _handles(scx_path):
        with pytest.raises(IndexError, match="integer array or a boolean mask"):
            h[np.asarray([0.0, 1.0])]


def test_full_slice_equals_to_memory(scx_path):
    for name, h, ref in _handles(scx_path):
        got = h[:]
        assert isinstance(got, sp.csr_matrix), name
        np.testing.assert_allclose(got.toarray(), ref, rtol=1e-6, err_msg=name)


def test_gather_on_a_deleted_file_is_in_logical_row_space(synthetic_adata, tmp_dir):
    import pyscx

    path = str(tmp_dir / "row_gather_deleted.scx")
    pyscx.from_anndata(synthetic_adata, path, shard_size=SHARD_SIZE)
    pyscx.mark_deleted(path, [0, 9, 10, 11, 50, 99])
    backed = pyscx.open(path).to_anndata(backed=True)
    assert backed.n_obs == 94
    kept = np.setdiff1d(np.arange(100), [0, 9, 10, 11, 50, 99])
    physical = sp.csr_matrix(synthetic_adata.X).toarray()
    rows = [0, 93, 8, 9, 10, 47]
    np.testing.assert_array_equal(backed.X[rows].toarray(), physical[kept[rows]])
    mask = np.zeros(94, dtype=bool)
    mask[::5] = True
    np.testing.assert_array_equal(backed.X[mask].toarray(), physical[kept[mask]])
    with pytest.raises(IndexError, match="94"):
        backed.X[[94]]


# ---------------------------------------------------------------------------
# Peak memory, measured in a fresh process
# ---------------------------------------------------------------------------

# One probe, three modes. `RssAnon` (not VmHWM / ru_maxrss) so the mmap'd file
# pages the decoder touches do not count; a 1 ms sampler thread with a short
# switch interval so it keeps running while scipy builds the result on the GIL;
# per-iteration peak over a trimmed baseline so the allocator's retention of a
# previous iteration cannot read as this iteration's peak.
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
    X = exp.to_anndata(backed=True).X
    n = X.shape[0]
    result = {"mode": mode, "n_obs": exp.n_obs, "nnz": exp.nnz, "n_shards": exp.shard_count}
    if mode == "gather":
        peaks = []
        for k in range(8):
            mask = np.zeros(n, dtype=bool)
            mask[k::8] = True
            peaks.append(measure(lambda: X[mask]))
        result["peaks"] = peaks
    elif mode == "full":
        result["peak"] = measure(lambda: X.to_memory())
    elif mode == "slice_all":
        result["peak"] = measure(lambda: X[:])
    print(json.dumps(result))
    """
)


def _probe(path, mode):
    env = dict(os.environ)
    # Pin the allocator and the decode pool so the measurement is about the
    # reader's buffers, not glibc's arena policy or the core count.
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
    # Not check=True: it would swallow stderr behind CalledProcessError.
    assert out.returncode == 0, f"probe {mode} failed:\n{out.stdout}\n{out.stderr}"
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self/status")
def test_row_gather_peak_memory_is_bounded_by_the_result_and_the_shard_cache(tmp_dir):
    """Eight interleaved eighths via `X[mask]` and a whole-matrix `X[:]`, each
    in a fresh process, against a full `to_memory()` decode.

    Before PR C the gather built one CSR per row and concatenated (2× the
    result on top of the LRU) and `X[:]` copied every shard twice; measured on
    this fixture that was 0.36× / 2.1× of the full decode. After: the result is
    assembled once, so a gather peaks at the result (one eighth) plus the LRU
    fill on the first iteration and the result plus one shard afterwards, and
    `X[:]` at the result plus `cache_shards` shards.
    """
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
    path = str(tmp_dir / "gather_peak.scx")
    pyscx.from_anndata(adata, path, shard_size=shard_size)
    exp = pyscx.open(path)
    assert exp.shard_count == n_shards

    full = _probe(path, "full")["peak"]
    slice_all = _probe(path, "slice_all")["peak"]
    gather = _probe(path, "gather")["peaks"]

    nnz = exp.nnz
    result_bytes = nnz // 8 * 8 + (n_obs // 8 + 1) * 8  # one eighth: i32 + f32 per nnz, i64 indptr
    shard_bytes = nnz // n_shards * 8 + (shard_size + 1) * 8
    cache_shards = 4  # to_anndata(backed=True) default
    report = (
        f"full={full / 1e6:.1f} MB slice_all={slice_all / 1e6:.1f} MB "
        f"gather={[round(p / 1e6, 1) for p in gather]} MB "
        f"(result {result_bytes / 1e6:.1f} MB, shard {shard_bytes / 1e6:.2f} MB)"
    )

    # (a) the spec's bars: eight interleaved eighths ≤ 0.35 of a full decode,
    # `X[:]` ≤ 1.2× `to_memory()`.
    assert max(gather) <= 0.35 * full, report
    assert slice_all <= 1.2 * full, report
    # (b) the discriminating bars. Steady state (the LRU is already resident in
    # the baseline from iteration 2 on) is the result plus one shard's transient
    # — a second copy of the result (the old path) is 0.25 of the full decode.
    steady = max(gather[1:])
    assert steady <= 0.20 * full, report
    assert steady <= result_bytes + 2 * shard_bytes + 2_000_000, report
    # `X[:]` decodes in parallel chunks of `cache_shards` uncached shards on
    # top of the exact result.
    assert slice_all <= full + (cache_shards + 2) * shard_bytes + 4_000_000, report
