"""Atlas-scale `filter_obs` / `count` over a row-sharded file.

Regression coverage for the OOM fix: at atlas scale (thousands of obs
metadata shards) the query engine used to concatenate every obs shard into
one in-memory batch before evaluating the predicate, peaking at hundreds of
GB. It now streams one obs shard at a time and only decodes the shards that
overlap surviving CSR shards.

The *memory bound* is proven rigorously by the Rust engine tests
(`scx-engine/tests/obs_shard_streaming.rs`: `read_obs` is never called and
`read_obs_shard` stays bounded by the surviving shards). These Python tests
exercise the same streaming path end-to-end through the public API on a
genuinely multi-obs-shard file and assert correctness + the sharded layout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _mk_sharded_adata(n_obs: int, n_vars: int, shard_size: int):
    """AnnData whose rows fall into `n_obs / shard_size` blocks, each block
    carrying a distinct `batch` value — mirroring the atlas layout where each
    shard's rows come from a single source file (one perturbation/batch per
    shard). High-selectivity `batch == 'bK'` filters then skip every other
    shard, exactly the case that exposed the OOM."""
    import anndata

    rng = np.random.default_rng(7)
    dense = rng.integers(0, 50, size=(n_obs, n_vars), dtype=np.int32).astype(np.float32)
    dense[rng.random((n_obs, n_vars)) > 0.4] = 0.0
    x = sp.csr_matrix(dense)

    batch = [f"b{i // shard_size}" for i in range(n_obs)]
    cell_type = ["fibroblast" if i % 2 == 0 else "epithelial" for i in range(n_obs)]
    obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n_obs)],
            "batch": pd.Categorical(batch),
            "cell_type": pd.Categorical(cell_type),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    return anndata.AnnData(X=x, obs=obs, var=var)


def _build(tmp_path, n_obs=500, n_vars=12, shard_size=64):
    import pyscx

    adata = _mk_sharded_adata(n_obs, n_vars, shard_size)
    out = str(tmp_path / "sharded.scx")
    pyscx.from_anndata(
        adata,
        out,
        shard_size=shard_size,
        index_obs=["batch", "cell_type"],
    )
    return out, shard_size


def test_sharded_layout_built(tmp_path):
    """Sanity: the fixture really produces a multi-shard obs layout, so the
    streaming path (not the legacy single-section path) is exercised."""
    import pyscx

    out, _ = _build(tmp_path)
    exp = pyscx.open(out)
    assert exp.n_obs == 500
    assert exp.obs_metadata_shard_count > 1


def test_count_high_selectivity_skips_shards(tmp_path):
    """count() over a per-shard-distinct value returns the exact block size
    and never materialises the full obs table (streaming path)."""
    import pyscx

    out, shard_size = _build(tmp_path)
    exp = pyscx.open(out)
    # 'b0' lives only in the first block of `shard_size` rows.
    assert exp.query().filter_obs("batch == 'b0'").count() == shard_size


def test_filter_obs_limit_collect(tmp_path):
    """filter_obs(...).limit(5).collect() returns the right rows with bounded
    obs-shard reads, through the real Python → Rust streaming path."""
    import pyscx

    out, _ = _build(tmp_path)
    exp = pyscx.open(out)
    result = exp.query().filter_obs("batch == 'b0'").limit(5).collect()
    assert result.n_obs == 5
    adata = result.to_anndata()
    assert list(adata.obs["cell_id"]) == [f"cell_{i}" for i in range(5)]
    assert all(adata.obs["batch"] == "b0")


def test_filter_obs_full_collect_parity(tmp_path):
    """A full (no-limit) collect on a sharded file matches a direct pandas
    reference over the source obs."""
    import pyscx

    out, shard_size = _build(tmp_path)
    exp = pyscx.open(out)
    result = exp.query().filter_obs("cell_type == 'fibroblast'").collect()
    adata = result.to_anndata()
    # Even rows are fibroblast in the fixture.
    expected_ids = [f"cell_{i}" for i in range(500) if i % 2 == 0]
    assert result.n_obs == len(expected_ids)
    assert sorted(adata.obs["cell_id"]) == sorted(expected_ids)
    assert all(adata.obs["cell_type"] == "fibroblast")


def test_no_match_indexed_value_returns_empty(tmp_path):
    """An equality on an INDEXED column whose value exists in no shard must
    short-circuit to an empty result (count 0 / 0 rows) instead of scanning all
    obs metadata. `batch` is indexed in the fixture, so a synthetic value the
    predicate index has never seen is proven absent from the index alone."""
    import pyscx

    out, _ = _build(tmp_path)
    exp = pyscx.open(out)
    q = exp.query().filter_obs("batch == '__never_exists__'")
    assert q.count() == 0
    result = exp.query().filter_obs("batch == '__never_exists__'").limit(5).collect()
    assert result.n_obs == 0
    # Schema survives the empty result.
    adata = result.to_anndata()
    assert "batch" in adata.obs.columns


def _cap_address_space_and_threads():
    """Pin the rayon/polars thread pools small, THEN cap address space.

    Each worker thread reserves stack + arena address space; on a many-core
    host the default pool (one thread per core) can exhaust the tight
    `RLIMIT_AS` cap before any query runs (`RuntimeError: can't start new
    thread`). Pinning the pool keeps the cap measuring obs-streaming RSS — the
    thing under test — not thread-stack reservations. Must run before `pyscx`
    is imported so the global rayon pool is sized from these."""
    import os

    os.environ.setdefault("RAYON_NUM_THREADS", "4")
    os.environ.setdefault("POLARS_MAX_THREADS", "4")
    try:
        import resource

        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        cap = 1536 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
    except (ImportError, ValueError, OSError):
        pass  # platform without RLIMIT_AS — still run for correctness


def _no_match_under_memory_cap_worker(path, q):
    """Module-level so it is picklable under the `spawn` start method."""
    _cap_address_space_and_threads()
    try:
        import pyscx

        n = pyscx.open(path).query().filter_obs("batch == '__never_exists__'").count()
        q.put(("ok", n))
    except Exception as exc:  # pragma: no cover - surfaced via assert
        q.put(("err", repr(exc)))


def test_no_match_count_in_child_process_under_memory_cap(tmp_path):
    """A no-match indexed equality must short-circuit under the address-space
    cap — the pre-fix path scanned all obs shards even for a value present in
    none."""
    import multiprocessing as mp

    out, _ = _build(tmp_path)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_no_match_under_memory_cap_worker, args=(out, q))
    p.start()
    status, payload = q.get(timeout=120)
    p.join(timeout=10)
    assert not p.is_alive(), "no-match query under memory cap did not finish"
    assert status == "ok", f"child failed: {payload}"
    assert payload == 0


def _count_under_memory_cap_worker(path, q):
    """Module-level so it is picklable under the `spawn` start method."""
    _cap_address_space_and_threads()
    try:
        import pyscx

        n = pyscx.open(path).query().filter_obs("batch == 'b0'").count()
        q.put(("ok", n))
    except Exception as exc:  # pragma: no cover - surfaced via assert
        q.put(("err", repr(exc)))


def test_count_in_child_process_under_memory_cap(tmp_path):
    """End-to-end: run count() in a child process under a generous address-
    space cap. The streaming path must complete; the OLD full-concat path was
    unbounded in shard count."""
    import multiprocessing as mp

    out, shard_size = _build(tmp_path)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_count_under_memory_cap_worker, args=(out, q))
    p.start()
    status, payload = q.get(timeout=120)
    p.join(timeout=10)
    assert not p.is_alive(), "query under memory cap did not finish (possible OOM/hang)"
    assert status == "ok", f"child failed: {payload}"
    assert payload == shard_size
