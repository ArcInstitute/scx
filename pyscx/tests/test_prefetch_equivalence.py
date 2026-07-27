"""Decode-prefetch must not change a single bit of any aggregation.

Phase 4.2 routed the backed aggregation kernels (`scx-format-io`), their
column-projected twins (`pyscx/src/projected_agg.rs`) and the lazy/transformed
streaming kernels (`pyscx/src/lazy_transform/dataset.rs`) through the bounded
ordered decode-prefetch pipeline that Phase 2.1 built for `scx-accel`; the PCA
follow-up added streaming PCA's six shard loops (`scx-accel/src/pca/cpu.rs`).
Up to `depth` shards now decode concurrently on the rayon pool.

The safety argument is that *consumption* stays single-threaded and in strict
shard order, so every f64 accumulation and every row-wise concatenation sees the
identical sequence. This file discharges that end to end, at the level a user
actually touches, by running the same ops twice with the pipeline **on** and
**off** and demanding bit-identical arrays.

`SCX_ACCEL_PREFETCH_DEPTH` is resolved once per process into a `OnceLock`, so
the two arms cannot live in one interpreter — each is a subprocess. The Rust
side has its own, cheaper guards (`scx-format-io`'s
`prefetched_aggregations_are_bit_identical_to_a_sequential_loop` and
`fixture_is_sensitive_to_shard_order`); this is the integration-level
counterpart that also covers the pyscx kernels.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import scipy.sparse as sp

import pyscx

N_OBS = 200
N_VARS = 24
SHARD_SIZE = 25  # → 8 shards; the pipeline no-ops on a single-shard file.

# Ascending, spans every shard, not a prefix — so a projection bug is visible.
COL_KEEP = np.arange(1, N_VARS, 2, dtype=np.uint32)
ROW_KEEP = np.arange(3, N_OBS, 3, dtype=np.int64)


@pytest.fixture(scope="module")
def scx_path(tmp_path_factory):
    """Multi-shard float fixture.

    Values span ~16 decades with a **cancelling pair**: shard 0 carries `+1e16`
    and shard 1 `-1e16` on the same columns, everything else small fractions.
    That is what makes summation order observable — with uniformly-scaled counts
    every partial sum is exact, a reordering regression produces bit-identical
    output, and every assertion below becomes vacuous.
    `test_fixture_would_expose_a_reordering` pins the property.

    The big values are placed *alongside* fractional ones rather than replacing
    a whole row: the writer picks a value encoding per shard and would choose an
    integer one for an all-`1e16` shard, which then fails as
    `value 10000000000000000 out of range for uint32`.
    """
    import anndata as ad

    rng = np.random.default_rng(11)
    x = sp.random(
        N_OBS, N_VARS, density=0.4, format="csr", dtype=np.float32, random_state=rng
    )
    # +0.5 keeps every value non-integral, so each shard encodes as Float32.
    x.data = (x.data * 20).astype(np.float32) + np.float32(0.5)
    x = x.tolil()
    big = np.float32(1e16)
    for r in range(SHARD_SIZE):
        for c in range(0, N_VARS, 3):
            x[r, c] = big
    for r in range(SHARD_SIZE, 2 * SHARD_SIZE):
        for c in range(0, N_VARS, 3):
            x[r, c] = -big
    x = x.tocsr().astype(np.float32)

    adata = ad.AnnData(X=x)
    adata.obs_names = [f"c{i}" for i in range(N_OBS)]
    adata.var_names = [f"g{i}" for i in range(N_VARS)]
    path = tmp_path_factory.mktemp("prefetch_ab") / "counts.scx"
    pyscx.from_anndata(adata, str(path), shard_size=SHARD_SIZE)
    return path


# The probe runs in a fresh interpreter so `SCX_ACCEL_PREFETCH_DEPTH` is read
# before any accelerator call. It writes float64 bit patterns as hex strings:
# JSON round-trips those exactly, where a float literal would not.
PROBE = r"""
import json, sys
import numpy as np
import pyscx

path = sys.argv[1]
out = {}

# Which binary is this arm actually testing? The two subprocesses must load the
# *same* `.so`, and the in-tree editable install is shared state that any
# concurrent `maturin develop` can swap. If the arms diverge, the comparison
# below silently becomes "build A vs build B" instead of "prefetch off vs on" —
# it would still produce a diff, just not the one being asserted.
out["__binary__"] = [pyscx.__file__]

def rec(name, arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).ravel()
    out[name] = [float(v).hex() for v in a]

col_keep = np.array(%(col_keep)s, dtype=np.uint32)
row_keep = np.array(%(row_keep)s, dtype=np.int64)

# --- backed, unprojected: scx-format-io's aggregation kernels ---
ad0 = pyscx.open(path).to_anndata(backed=True)
rec("col_sums", pyscx.accel.col_sums(ad0.X))
rec("col_nnz", pyscx.accel.col_nnz(ad0.X))
rec("col_var", pyscx.accel.col_var(ad0.X))
rec("col_max", pyscx.accel.col_max(ad0.X))
rec("col_min", pyscx.accel.col_min(ad0.X))
rec("row_sums", ad0.X.sum(axis=1))
rec("x_sum_axis0", ad0.X.sum(axis=0))

# --- backed + column projection: projected_agg.rs ---
adp = pyscx.open(path).to_anndata(backed=True)
adp = adp[:, col_keep.tolist()]
rec("proj_col_sums", pyscx.accel.col_sums(adp.X))
rec("proj_col_var", pyscx.accel.col_var(adp.X))
rec("proj_row_sums", adp.X.sum(axis=1))

# --- backed + row subset: the masked kernels ---
adr = pyscx.open(path).to_anndata(backed=True)
adr = adr[row_keep.tolist(), :]
rec("masked_col_sums", pyscx.accel.col_sums(adr.X))
rec("masked_col_var", pyscx.accel.col_var(adr.X))

# --- backed + BOTH axes: the *_masked_projected kernels ---
# Neither single-axis block above reaches them. `col_aggs`' (Some(cols),
# Some(kept)) arm is the only route to col_sums_masked_projected and friends,
# and a row subset alone lands on the scx-format-io kernel instead. Without
# this, a dropped `global_row` cursor in e.g. row_nnz_and_sums_projected would
# leave the suite green.
adb = pyscx.open(path).to_anndata(backed=True)
adb = adb[row_keep.tolist(), :][:, col_keep.tolist()]
rec("both_col_sums", pyscx.accel.col_sums(adb.X))
rec("both_col_nnz", pyscx.accel.col_nnz(adb.X))
rec("both_col_var", pyscx.accel.col_var(adb.X))
rec("both_col_max", pyscx.accel.col_max(adb.X))
rec("both_row_sums", adb.X.sum(axis=1))
rec("both_row_nnz", adb.X.getnnz(axis=1))

# --- lazy/transformed: lazy_transform/dataset.rs ---
adl = pyscx.open(path).to_anndata(backed=True)
pyscx.accel.normalize_total(adl, target_sum=1e4)
rec("lazy_col_sums", np.asarray(adl.X.sum(axis=0)).ravel())
rec("lazy_row_sums", np.asarray(adl.X.sum(axis=1)).ravel())
rec("lazy_col_var", np.asarray(adl.X.var(axis=0)).ravel())

# --- streaming PCA: pca/cpu.rs's six shard loops ---
# `method="randomized"` only, and deliberately. That route is *structurally*
# deterministic — the forward SpMM writes each output row exactly once, the
# transpose takes its sequential branch at this size, and the fused column-means
# pass accumulates in shard order — so it belongs under this file's exact-bits
# blanket. `method="covariance"` does not: it folds into `ThreadLocal`
# accumulators whose merge order is scheduler-dependent, and five consecutive
# runs on this very fixture produce five different results with no prefetch
# involved at all. Putting it here would make the suite flaky and would tell you
# nothing; `covariance_pca_agrees_across_depths_within_reassociation_noise` in
# scx-accel covers it against the bar it can actually hold.
adpca = pyscx.open(path).to_anndata(backed=True)
pyscx.accel.pca(adpca, n_comps=3, device="cpu", method="randomized")
rec("pca_embeddings", adpca.obsm["X_pca"])
rec("pca_components", adpca.varm["PCs"])
rec("pca_variance_ratio", adpca.uns["pca"]["variance_ratio"])

# Same, through a column projection — `ProjectedShardSource` wraps the shard
# source, so this is the only arm that exercises the projected path's ordering.
adpcap = pyscx.open(path).to_anndata(backed=True)
adpcap = adpcap[:, col_keep.tolist()]
pyscx.accel.pca(adpcap, n_comps=3, device="cpu", method="randomized")
rec("pca_proj_embeddings", adpcap.obsm["X_pca"])

# --- QC, which fans out over both axes in one fused pass ---
adq = pyscx.open(path).to_anndata(backed=True)
pyscx.accel.calculate_qc_metrics(adq)
for c in ("total_counts", "n_genes_by_counts"):
    if c in adq.obs:
        rec("qc_obs_" + c, adq.obs[c].to_numpy())
for c in ("total_counts", "n_cells_by_counts"):
    if c in adq.var:
        rec("qc_var_" + c, adq.var[c].to_numpy())

json.dump(out, sys.stdout)
""" % {"col_keep": COL_KEEP.tolist(), "row_keep": ROW_KEEP.tolist()}


def _run(path, depth: str | None) -> dict[str, list[str]]:
    env = dict(os.environ)
    if depth is None:
        env.pop("SCX_ACCEL_PREFETCH_DEPTH", None)
    else:
        env["SCX_ACCEL_PREFETCH_DEPTH"] = depth
    proc = subprocess.run(
        [sys.executable, "-c", PROBE, str(path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"probe failed (SCX_ACCEL_PREFETCH_DEPTH={depth}):\n{proc.stderr[-4000:]}"
        )
    return json.loads(proc.stdout)


@pytest.fixture(scope="module")
def ab(scx_path):
    """Two arms of the same probe: prefetch disabled vs. the default depth."""
    off = _run(scx_path, "1")  # 0/1 disables the pipeline entirely
    on = _run(scx_path, None)  # unset → DEFAULT_PREFETCH_DEPTH (4)
    return off, on


def test_probe_covered_every_kernel(ab):
    """Guard against a silently-empty comparison."""
    off, on = ab
    assert off.keys() == on.keys()
    # 23 arrays + the binary-identity marker.
    assert len(off) >= 20, f"probe recorded only {len(off)} entries"
    for name, vals in off.items():
        assert vals, f"{name} came back empty — nothing is being compared"
    # PCA specifically: an all-zero or non-finite embedding would satisfy the
    # bit-identity assertion while comparing nothing. NaN in particular would
    # sail through, because `float('nan').hex()` is the string "nan" on both
    # arms.
    for name in ("pca_embeddings", "pca_proj_embeddings"):
        vals = np.array([float.fromhex(v) for v in off[name]])
        assert np.isfinite(vals).all(), f"{name} is not finite"
        assert vals.std() > 0.0, f"{name} is constant — the comparison is vacuous"


def test_both_arms_loaded_the_same_binary(ab):
    """The two arms must differ only in the env knob, not in the library.

    `pyscx` resolves through a single editable install pointing at one in-tree
    `.so`, and any concurrent `maturin develop` — another test session, a
    benchmark job — replaces it. If that happened between the two subprocess
    launches, `test_prefetch_is_bit_identical` would be comparing two *builds*
    while reporting on two prefetch depths.
    """
    off, on = ab
    assert off["__binary__"] == on["__binary__"], (
        f"arms loaded different pyscx builds: {off['__binary__']} vs "
        f"{on['__binary__']} — the comparison is not depth-vs-depth"
    )


def test_prefetch_is_bit_identical(ab):
    """Every aggregation, hex-exact, prefetch off vs on."""
    off, on = ab
    mismatched = [
        name for name in off if name != "__binary__" and off[name] != on[name]
    ]
    assert not mismatched, (
        "decode-prefetch changed these results: "
        + ", ".join(
            f"{n} (first diff: {next(a for a, b in zip(off[n], on[n]) if a != b)} "
            f"vs {next(b for a, b in zip(off[n], on[n]) if a != b)})"
            for n in mismatched
        )
    )


def test_fixture_would_expose_a_reordering(scx_path):
    """The premise: on this fixture, summation order is observable.

    Without a cancelling pair the per-shard partials add exactly in any order,
    so `test_prefetch_is_bit_identical` would pass even if the pipeline
    delivered shards out of order. Adding the same column sums in reverse shard
    order must give a different f64 result, or these tests prove nothing.
    """
    import itertools

    adata = pyscx.open(str(scx_path)).to_anndata()
    dense = np.asarray(adata.X.todense(), dtype=np.float64)
    bounds = [
        (s, min(s + SHARD_SIZE, N_OBS)) for s in range(0, N_OBS, SHARD_SIZE)
    ]
    n_shards = len(bounds)
    assert n_shards > 1, "single-shard fixture: the prefetch path never engages"
    partials = [dense[s:e].sum(axis=0) for s, e in bounds]

    def total(order):
        acc = np.zeros(N_VARS, dtype=np.float64)
        for i in order:
            acc = acc + partials[i]
        return [float(v).hex() for v in acc]

    forward = total(range(n_shards))
    assert any(
        total(p) != forward
        for p in itertools.permutations(range(n_shards))
        if list(p) != list(range(n_shards))
    ), (
        "fixture is order-insensitive: no shard-visit permutation changes the "
        "column sums, so the bit-identity assertions cannot detect a "
        "reordering regression"
    )
