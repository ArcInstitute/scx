"""A row-projected read must not decode shards it cannot use.

A backed handle's row projection — a deletion vector, `adata[mask]`, an
`adata[:n]` window — is `kept_to_global`, a strictly ascending list of physical
row indices. The masked aggregation kernels `partition_point` it per shard to
find the rows they must visit, and `LazyShardSource` applies it after decoding
and hands back a 0-row CSR when nothing survives. Either way a shard no kept row
falls in was **fully decoded first** — and, on the projected kernels, column
projected too — before its emptiness was discovered.

Measured on a 120 x 200 file in 5 shards of 24 rows, counting decodes through
`SCX_CPU_PROFILE=1` / `cpu_profile_snapshot()`:

    request                     before   after
    X.sum(axis=0) on [:24]           5       1
    X.getnnz(axis=0) on [:24]        5       1
    a mask over shards 0 and 4       5       2
    no row projection at all         5       5
    normalize_total+log1p chain      5       2

"Before" is not a guess: it was produced by forcing
`BackedCsrIndex::shards_with_kept_rows` to accept every shard, which is exactly
what the old code did implicitly.

Values are unchanged, and not approximately: today's kernels already run an
empty inner loop over `kept_rows[lo..hi]` for such a shard, accumulating
nothing, and the lazy consumers advance their row cursor by a `n_rows()` that
was zero. Skipping removes a decode and no arithmetic. Every case below asserts
the values against an in-memory oracle anyway, because the one thing that is
*not* free is the transform offset — `apply_transforms` is given a global row
index, and a running cursor would under-count once a shard is skipped.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

anndata = pytest.importorskip("anndata")

N_OBS, N_VARS, SHARD = 120, 200, 24
N_SHARDS = N_OBS // SHARD


def _dense():
    rng = np.random.default_rng(3)
    return rng.integers(0, 20, size=(N_OBS, N_VARS)).astype(np.float32)


@pytest.fixture(scope="module")
def path(tmp_path_factory):
    import pyscx

    dense = _dense()
    adata = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(N_OBS)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(N_VARS)]),
    )
    p = str(tmp_path_factory.mktemp("skip") / "skip.scx")
    pyscx.from_anndata(adata, p, shard_size=SHARD)
    return p


def test_premise_the_fixture_has_several_shards(path):
    """A single-shard file cannot show a skip, and every small fixture is one."""
    import pyscx

    adata = pyscx.open(path).to_anndata(backed=True)
    assert adata.X.n_shards == N_SHARDS == 5


# ---------------------------------------------------------------------------
# Decode counts. A subprocess per case: the profiler gate is a `OnceLock` read
# of SCX_CPU_PROFILE, so it has to be set before `import pyscx`.
# ---------------------------------------------------------------------------

_PROBE = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    import pyscx

    # `<selector>[+lazy]:<op>` — parsed by segment, never by prefix/suffix on
    # the whole string. An earlier version tested `case.endswith("lazy")`,
    # which is never true once the op is appended, so both lazy arms silently
    # ran the non-lazy path and asserted the non-lazy count.
    path, case = sys.argv[1], sys.argv[2]
    selector, _, op = case.partition(":")
    parts = selector.split("+")
    sel, lazy = parts[0], "lazy" in parts[1:]
    assert sel in ("window", "mask", "all"), sel
    assert op, case

    adata = pyscx.open(path).to_anndata(backed=True)
    n_shards = adata.X.n_shards

    if sel == "window":
        adata = adata[:24]
    elif sel == "mask":
        m = np.zeros(adata.n_obs, dtype=bool)
        m[:24] = True
        m[96:] = True
        adata = adata[m]
    assert adata.n_obs > 0

    if lazy:
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        pyscx.accel.log1p(adata)
        assert type(adata.X).__name__ == "ScxLazyTransformedDataset", type(adata.X)

    # `mask_var` is passed as an **argument**, never written to `adata.var`:
    # on a subset view that write is copy-on-write, which materialises `X` and
    # takes the streaming path out of the picture entirely. An earlier version
    # of this probe did exactly that and measured 0 decodes either way, which
    # is how the `ProjectedShardSource` finding came to be wrongly rejected.
    mask_var = None
    if op == "mask_var":
        mask_var = np.arange(adata.n_vars) % 3 == 0
        op = "pca"
    elif op == "mask_var_all":
        mask_var = np.ones(adata.n_vars, dtype=bool)
        op = "pca"
    pyscx.accel.cpu_profile_reset()
    if op == "sum":
        adata.X.sum(axis=0)
    elif op == "getnnz":
        adata.X.getnnz(axis=0)
    elif op == "var":
        # `col_var` takes the handle; on a lazy `X` the equivalent is the
        # handle's own reduction, so this op works for `+lazy` cases too rather
        # than raising on a type `col_var` does not accept.
        if type(adata.X).__name__ == "ScxLazyTransformedDataset":
            adata.X.var(axis=0)
        else:
            pyscx.accel.col_var(adata.X)
    elif op == "score_genes":
        pyscx.accel.score_genes(
            adata, ["g0", "g1", "g2"], method="mean", device="cpu"
        )
    elif op == "pca":
        kw = {} if mask_var is None else {"mask_var": mask_var}
        pyscx.accel.pca(adata, n_comps=3, device="cpu", **kw)
    elif op == "hvg_seurat":
        pyscx.accel.highly_variable_genes(
            adata, n_top_genes=10, flavor="seurat", device="cpu"
        )
    elif op == "pflog":
        pyscx.accel.pflog(adata, store="baseline")
    elif op == "hvg":
        pyscx.accel.highly_variable_genes(
            adata, n_top_genes=10, flavor="seurat_v3", device="cpu"
        )
    else:
        raise SystemExit(f"unknown op {op}")
    snap = pyscx.accel.cpu_profile_snapshot()
    print(json.dumps({
        "enabled": snap["enabled"],
        "decodes": snap["decode_scx1"]["count"] + snap["decode_generic"]["count"],
        "n_shards": n_shards,
    }))
    """
)


def _probe(path, case, **extra_env):
    env = dict(os.environ, SCX_CPU_PROFILE="1", **extra_env)
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, path, case],
        capture_output=True,
        text=True,
        env=env,
    )
    if out.returncode != 0:
        raise AssertionError(f"probe failed (case={case}):\n{out.stderr[-2000:]}")
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["enabled"], "SCX_CPU_PROFILE did not take — the count is not a count"
    return res


# `op -> (decodes with a 1-of-5-shard window, decodes with no projection)`.
# The second number is the accept side: a skip that fired unconditionally would
# make the first assertion pass on its own.
_OPS = {
    # Masked column kernels, which hold the row set as a parameter.
    "sum": (1, 5),
    "getnnz": (1, 5),
    "var": (2, 10),  # two passes: means, then squared deviations
    # `as_shard_source()` consumers, which reach the row filter through the
    # source rather than through an argument — a different mechanism, so each
    # is asserted rather than inferred from the kernels above.
    "score_genes": (1, 5),
    "pca": (1, 5),
    "pflog": (2, 10),
    "hvg": (2, 10),  # seurat_v3: mean/var, then the clipped square sum
}


@pytest.mark.parametrize("op", ["mask_var", "mask_var_all"])
def test_pca_under_a_gene_mask_still_skips(path, op):
    """`mask_var=` wraps the source in `ProjectedShardSource`.

    That wrapper forwards `n_shards`, `n_obs`, `max_shard_rows`,
    `shard_cache_capacity` and `shard_size_hint` to its inner source but not,
    at first, the shard plan — so it answered the trait default ("visit every
    shard") and discarded it. `mask_var=None` auto-consumes
    `adata.var["highly_variable"]`, so the HVG → PCA pipeline on a row subset is
    the composition that lost the skip. Measured on this fixture: 5 decodes
    without the forward, 1 with it, for a partial mask and for an all-true one
    alike.
    """
    res = _probe(path, f"window:{op}")
    assert res["decodes"] == 1, (
        f"PCA with mask_var= decoded {res['decodes']} shards on a "
        "1-of-5-shard row window; ProjectedShardSource must forward "
        "visible_shard_indices"
    )
    assert _probe(path, f"all:{op}")["decodes"] == N_SHARDS


def test_the_tolerant_reduction_mode_skips_too(path):
    """`SCX_ACCEL_REDUCTION_MODE=parallel_tolerant` is a supported mode.

    HVG `flavor="seurat"` reduces through `accumulate_shards`, whose tolerant
    arm is `reduce_shards_budgeted` — which swept `0..n_shards` and never
    consulted the plan. So the skip held on the default `StableOrder` and was
    silently lost in the other mode: 1 decode by default, 5 in tolerant mode.
    `seurat_v3` is not affected (it drives the ordered driver directly), which
    is why testing only that flavour missed this.
    """
    default = _probe(path, "window:hvg_seurat")
    tolerant = _probe(
        path, "window:hvg_seurat", SCX_ACCEL_REDUCTION_MODE="parallel_tolerant"
    )
    assert default["decodes"] == 1, default["decodes"]
    assert tolerant["decodes"] == default["decodes"], (
        f"tolerant mode decoded {tolerant['decodes']} shards where the default "
        f"mode decoded {default['decodes']}"
    )
    assert (
        _probe(path, "all:hvg_seurat", SCX_ACCEL_REDUCTION_MODE="parallel_tolerant")[
            "decodes"
        ]
        == N_SHARDS
    )


@pytest.mark.parametrize("op", sorted(_OPS))
def test_a_one_shard_window_decodes_one_shard(path, op):
    windowed, whole = _OPS[op]
    res = _probe(path, f"window:{op}")
    assert res["n_shards"] == N_SHARDS
    assert res["decodes"] == windowed, (
        f"{op} on a 1-of-5-shard row window decoded {res['decodes']} shards, "
        f"expected {windowed}: the four shards it excludes hold no visible row"
    )


@pytest.mark.parametrize("op", sorted(_OPS))
def test_without_a_row_projection_nothing_is_skipped(path, op):
    """The accept side, per op: every shard is still read when all rows are."""
    _, whole = _OPS[op]
    res = _probe(path, f"all:{op}")
    assert res["decodes"] == whole, (
        f"{op} with no row projection decoded {res['decodes']} shards, "
        f"expected {whole} — a skip must not fire when there is nothing to skip"
    )


@pytest.mark.parametrize("op", ["sum", "getnnz"])
def test_a_mask_over_two_shards_decodes_two(path, op):
    res = _probe(path, f"mask:{op}")
    assert res["decodes"] == 2, (
        f"{op} over a mask covering shards 0 and 4 decoded {res['decodes']}; "
        "shards 1, 2 and 3 contribute nothing"
    )


def test_a_lazy_transform_chain_also_skips(path):
    """`normalize_total` + `log1p` over a row subset, the third mechanism.

    A lazy `X` routes to the transform-aware column kernels, which iterate the
    raw reader and apply the chain per shard. A skipped shard saves the
    transform pass as well as the decode.
    """
    lazy = _probe(path, "mask+lazy:sum")
    assert lazy["decodes"] == 2, (
        f"the lazy chain decoded {lazy['decodes']} shards; shards 1-3 hold no "
        "visible row"
    )
    assert _probe(path, "all+lazy:sum")["decodes"] == N_SHARDS


# ---------------------------------------------------------------------------
# Values. The skip is arithmetic-free by construction, *except* for the row
# offset handed to `apply_transforms`.
# ---------------------------------------------------------------------------


def _mask():
    m = np.zeros(N_OBS, dtype=bool)
    m[:SHARD] = True
    m[N_OBS - SHARD :] = True
    return m


@pytest.mark.parametrize(
    "sel, label",
    [(slice(0, SHARD), "window"), (_mask(), "mask"), (slice(None), "all")],
)
def test_masked_aggregates_match_an_in_memory_oracle(path, sel, label):
    import pyscx

    dense = _dense()
    adata = pyscx.open(path).to_anndata(backed=True)[sel]
    ref = dense[sel]

    np.testing.assert_allclose(
        np.asarray(adata.X.sum(axis=0)).ravel(), ref.sum(axis=0), rtol=1e-6
    )
    np.testing.assert_array_equal(
        np.asarray(adata.X.getnnz(axis=0)).ravel(), (ref > 0).sum(axis=0)
    )
    np.testing.assert_allclose(
        pyscx.accel.col_var(adata.X), ref.astype(np.float64).var(axis=0), rtol=1e-6
    )


def test_a_transform_chain_over_a_gapped_subset_uses_the_right_row_offsets(path):
    """The one way skipping *could* change values, pinned.

    `apply_transforms(&mut csr, global_row)` is given the shard's first global
    row. That used to be a running total of the rows seen so far, which is only
    equal to the shard's own `row_start` while every shard is visited — so with
    shards 1-3 skipped, the offset for shard 4 would be 24 instead of 96 and
    `normalize_total`'s per-row denominators would come from the wrong cells.
    The kept set below deliberately leaves a three-shard gap in the middle.
    """
    import pyscx

    dense = _dense().astype(np.float64)
    mask = _mask()

    # `adata[mask]` then `normalize_total` on purpose, warning and all: that is
    # the sequence a caller writes, and it is what installs `kept_to_global` on
    # the lazy source. `subset_obs` (which the warning recommends) would reach
    # the same state without exercising the view transition.
    adata = pyscx.open(path).to_anndata(backed=True)[mask]
    pyscx.accel.normalize_total(adata, target_sum=1e4)
    pyscx.accel.log1p(adata)

    ref = dense[mask]
    ref = np.log1p(ref / ref.sum(axis=1, keepdims=True) * 1e4)

    np.testing.assert_allclose(
        np.asarray(adata.X.sum(axis=0)).ravel(), ref.sum(axis=0), rtol=1e-6
    )


# ---------------------------------------------------------------------------
# A skipped read is still a read, as far as handle freshness is concerned.
# ---------------------------------------------------------------------------


def test_an_all_excluded_projection_still_refuses_a_changed_file(tmp_path):
    """The regression the skip introduced, and the reason for the up-front check.

    A `pyscx` handle that is asked to read a file which changed since it was
    opened must raise rather than answer from its obsolete mapping. That check
    lives in the reader's shard read — so a shard plan with *nothing* in it used
    to answer without performing any read at all, and the same handle would
    return zeros for `X.sum(axis=0)` while raising on `X.shape`:

        x = pyscx.open(p).to_anndata(backed=True)[:0].X
        <mutate the file>
        x.sum(axis=0)  ->  [[0.0, 0.0, 0.0]]   # before
        x.shape        ->  RuntimeError: ... changed on disk since it was opened

    A zero vector is not harmless either: a copy-out rewrite can change the gene
    axis, so the width can be obsolete too. Found by review.
    """
    import pyscx

    dense = np.arange(12, dtype=np.float32).reshape(4, 3) + 1
    adata = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    p = str(tmp_path / "fresh.scx")
    pyscx.from_anndata(adata, p, shard_size=2)

    x = pyscx.open(p).to_anndata(backed=True)[:0].X
    assert x.shape[0] == 0, "premise: the projection keeps no row at all"

    drop = np.zeros(4, dtype=bool)
    drop[1:] = True
    pyscx.open(p).mark_deleted(drop)

    for label, call in [
        ("sum(axis=0)", lambda: x.sum(axis=0)),
        ("getnnz(axis=0)", lambda: x.getnnz(axis=0)),
        ("shape", lambda: x.shape),
    ]:
        with pytest.raises(RuntimeError, match="changed on disk"):
            call()
            pytest.fail(f"{label} answered from a stale handle")


def test_an_all_excluded_projection_still_answers_on_an_unchanged_file(tmp_path):
    """The accept side: the freshness check must not reject a valid empty read."""
    import pyscx

    dense = np.arange(12, dtype=np.float32).reshape(4, 3) + 1
    adata = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    p = str(tmp_path / "unchanged.scx")
    pyscx.from_anndata(adata, p, shard_size=2)

    x = pyscx.open(p).to_anndata(backed=True)[:0].X
    np.testing.assert_array_equal(np.asarray(x.sum(axis=0)).ravel(), [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(np.asarray(x.getnnz(axis=0)).ravel(), [0, 0, 0])


def test_an_all_excluded_projection_on_a_lazy_x_also_refuses(tmp_path):
    """The same hole, on the handle a post-QC pipeline actually holds.

    The first fix put the check in the backed masked kernels and had
    `LazyShardSource` report "no plan" on a stale file. `ScxLazyTransformedDataset`'s
    own column aggregations go through neither: they build a selected shard plan
    from `self.backed.index()` and drive the raw reader. So
    `normalize_total → log1p` over `adata[:0]` still answered
    `sum(axis=0) -> [[0, 0, 0]]`, `var(axis=0) -> [[0, 0, 0]]` and
    `mean(axis=0) -> [[nan, nan, nan]]` while `shape` raised. All three
    reviewers reproduced it independently.
    """
    import pyscx

    dense = np.arange(12, dtype=np.float32).reshape(4, 3) + 1
    adata = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    p = str(tmp_path / "lazy_fresh.scx")
    pyscx.from_anndata(adata, p, shard_size=2)

    empty = pyscx.open(p).to_anndata(backed=True)[:0]
    pyscx.accel.normalize_total(empty, target_sum=1e4)
    pyscx.accel.log1p(empty)
    x = empty.X
    assert type(x).__name__ == "ScxLazyTransformedDataset", type(x)

    drop = np.zeros(4, dtype=bool)
    drop[1:] = True
    pyscx.open(p).mark_deleted(drop)

    for call in (
        lambda: x.sum(axis=0),
        lambda: x.var(axis=0),
        lambda: x.mean(axis=0),
        lambda: x.shape,
    ):
        with pytest.raises(RuntimeError, match="changed on disk"):
            call()


def test_a_lazy_x_over_an_empty_projection_answers_on_an_unchanged_file(tmp_path):
    """Accept side for the arm above."""
    import pyscx

    dense = np.arange(12, dtype=np.float32).reshape(4, 3) + 1
    adata = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    p = str(tmp_path / "lazy_ok.scx")
    pyscx.from_anndata(adata, p, shard_size=2)

    empty = pyscx.open(p).to_anndata(backed=True)[:0]
    pyscx.accel.normalize_total(empty, target_sum=1e4)
    np.testing.assert_array_equal(
        np.asarray(empty.X.sum(axis=0)).ravel(), [0.0, 0.0, 0.0]
    )
