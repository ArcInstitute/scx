"""What eager `to_anndata(var_names=...)` returns must not change.

The eager gene projection used to assemble everything at full width and hand it
to anndata to slice (`adata[:, idx].copy()`), which is why nothing about its
*output* was ever pinned: `adata[:, idx]` is anndata's own semantics, so there
was nothing to get wrong. Projecting while assembling means the shape of the
answer is now ours to preserve, and `adata[:, idx]` does more than slice — it
prunes unused **var** categories and reindexes (or deletes) `uns["<col>_colors"]`
in `AnnData._init_as_view`. It also deliberately leaves `.raw` alone, which is
why `.raw` here stays on its own, wider gene axis.

Most of this file is therefore a **characterization pin**: it passes before the
change and must keep passing after. Each assertion names the mutation it kills,
because a pin that nothing can break is decoration.

The one thing that is red before the change is cost, and that lives in
`test_eager_var_names_memory.py`.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

N_OBS, N_VARS, RAW_N_VARS = 24, 12, 30

# Genes 1, 3 and 9 carry chromosomes "2", "4" and "2" — so the surviving
# category set is ["2", "4"], which is NOT a prefix of the declared
# ["1", "2", "3", "4", "5"]. That is deliberate: with a prefix, truncating the
# colour list and index-selecting it give the same answer.
NAMES = ["g9", "g1", "g3"]
SORTED_IDX = [1, 3, 9]
REQUEST_IDX = [9, 1, 3]


def _dense(value):
    return np.asarray(value.todense() if hasattr(value, "todense") else value)


def _source(seed=0):
    """Numpy oracle for every member, so a test can check values, not just shapes."""
    rng = np.random.default_rng(seed)
    x = np.ceil(rng.random((N_OBS, N_VARS)) * 9).astype(np.float32)
    x[rng.random((N_OBS, N_VARS)) > 0.6] = 0.0
    return {
        "x": x,
        "counts": (x * 2).astype(np.float32),
        "spliced": (x + 1).astype(np.float32),
        "emb": rng.random((N_OBS, 3)).astype(np.float32),
        "umap": rng.random((N_OBS, 2)).astype(np.float32),
        "loadings": rng.random((N_VARS, 4)).astype(np.float32),
        "pcs": rng.random((N_VARS, 2)).astype(np.float32),
        "conn": np.eye(N_OBS, dtype=np.float32),
        "dist": (np.eye(N_OBS, dtype=np.float32) * 3.0),
        "corr": np.eye(N_VARS, dtype=np.float32),
        "cov": (np.eye(N_VARS, dtype=np.float32) * 2.0),
        "raw": np.ceil(rng.random((N_OBS, RAW_N_VARS)) * 5).astype(np.float32),
    }


def _rich_adata(source):
    import anndata

    chrom = [str(i % 4 + 1) for i in range(N_VARS)]
    # Blocked, not alternating, so that genes 1, 3 and 9 use both levels and
    # "band" survives the projection whole while "chr" is pruned.
    band = ["p" if (i // 3) % 2 == 0 else "q" for i in range(N_VARS)]
    var = pd.DataFrame(
        {
            # "5" is declared but used by no gene: it must survive a plain read
            # and be pruned by a projection, like every other unused level.
            "chr": pd.Categorical(chrom, categories=["1", "2", "3", "4", "5"]),
            "band": pd.Categorical(band, categories=["p", "q"]),
            "mean_expr": np.linspace(0.1, 1.0, N_VARS, dtype=np.float32),
        },
        index=[f"g{i}" for i in range(N_VARS)],
    )
    obs = pd.DataFrame(
        {
            # Fully used on purpose: a var-only slice also prunes obs levels
            # (anndata does it for both axes), and entangling that with this
            # fixture would make the differential below test two things at once.
            "group": pd.Categorical(["a" if i % 2 == 0 else "b" for i in range(N_OBS)]),
            "n_counts": np.arange(N_OBS, dtype=np.int32),
        },
        index=[f"c{i}" for i in range(N_OBS)],
    )
    adata = anndata.AnnData(
        X=sp.csr_matrix(source["x"]),
        obs=obs,
        var=var,
        layers={
            "counts": sp.csr_matrix(source["counts"]),
            "spliced": sp.csr_matrix(source["spliced"]),
        },
        obsm={"X_emb": source["emb"], "X_umap": source["umap"]},
        varm={"loadings": source["loadings"], "PCs": source["pcs"]},
        obsp={"conn": sp.csr_matrix(source["conn"]), "dist": sp.csr_matrix(source["dist"])},
        varp={"corr": sp.csr_matrix(source["corr"]), "cov": sp.csr_matrix(source["cov"])},
    )
    adata.uns = {
        # len == len(categories) -> index-selected by the surviving levels.
        "chr_colors": ["#c1", "#c2", "#c3", "#c4", "#c5"],
        # len != len(categories) -> anndata deletes the key outright.
        "band_colors": ["#b1"],
        "group_colors": ["#g1", "#g2"],
        "species": "human",
        "arr": np.array([[1, 2], [3, 4]], dtype=np.int32),
    }
    raw_var = pd.DataFrame(
        {"symbol": [f"SYM{i}" for i in range(RAW_N_VARS)]},
        index=[f"r{i}" for i in range(RAW_N_VARS)],
    )
    adata.raw = anndata.AnnData(X=sp.csr_matrix(source["raw"]), var=raw_var, obs=obs)
    return adata


@pytest.fixture(scope="module")
def rich(tmp_path_factory):
    """A file carrying two keys in every aligned slot, plus raw and categoricals.

    Two keys per slot, not one: a single-key slot cannot tell "projected the
    right key" from "ignored the projection".
    """
    import pyscx

    source = _source()
    path = str(tmp_path_factory.mktemp("var_names") / "rich.scx")
    pyscx.from_anndata(_rich_adata(source), path, shard_size=8)
    return path, source


def test_fixture_carries_what_the_pins_read(rich):
    """Without this, most of the file below passes vacuously."""
    import pyscx

    path, source = rich
    base = pyscx.open(path).to_anndata()
    assert isinstance(base.var["chr"].dtype, pd.CategoricalDtype), (
        "var must round-trip as a real factor, else the category pins mean nothing"
    )
    assert list(base.var["chr"].cat.categories) == ["1", "2", "3", "4", "5"], (
        "the declared-but-unused level must survive an unprojected read"
    )
    assert [base.var["chr"][n] for n in NAMES] == ["2", "2", "4"], (
        "the requested genes must map to a non-prefix category subset"
    )
    assert base.raw is not None and base.raw.shape == (N_OBS, RAW_N_VARS)
    assert sorted(base.layers) == ["counts", "spliced"]
    assert sorted(base.varm) == ["PCs", "loadings"]
    assert sorted(base.varp) == ["corr", "cov"]
    assert sorted(base.obsp) == ["conn", "dist"]
    assert pyscx.open(path).shard_count > 1, "a single-shard file cannot see a per-shard bug"


# --------------------------------------------------------------------------
# The whole-object differential
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, idx",
    [
        ({}, SORTED_IDX),
        ({"preserve_var_order": True}, REQUEST_IDX),
        ({"layers": ["counts"], "varp": ["corr"]}, SORTED_IDX),
    ],
    ids=["sorted", "request_order", "with_slot_filters"],
)
def test_projected_equals_full_then_sliced(rich, kwargs, idx):
    """`to_anndata(var_names=G)` must equal `to_anndata()[:, resolve(G)].copy()`.

    Both sides come off the same file, so every round-trip artefact cancels and
    the only variable is how the gene axis was applied. `assert_equal` walks X,
    obs, var, obsm, varm, layers, uns, obsp, varp **and raw**.

    Kills: a wrong column order or off-by-one in X; layers left at full width or
    projected in a different order than X; varm handed over at physical var
    width (right shape, another gene's rows — anndata validates the first and
    not the second); varp projected on one axis only; obsm/obsp wrongly
    projected; raw projected at all; category or colour drift in var/uns.
    """
    from anndata.tests.helpers import assert_equal

    import pyscx

    path, _ = rich
    projected = pyscx.open(path).to_anndata(var_names=NAMES, **kwargs)
    sliced = pyscx.open(path).to_anndata(
        **{k: v for k, v in kwargs.items() if k != "preserve_var_order"}
    )[:, np.array(idx)].copy()

    # Explicitly, before assert_equal: it reorders `b` when the names disagree,
    # so an ordering bug could otherwise be repaired instead of reported.
    assert list(projected.var_names) == list(sliced.var_names)
    assert_equal(projected, sliced)


# --------------------------------------------------------------------------
# The pieces, asserted by name so a failure says which rule broke
# --------------------------------------------------------------------------


def test_var_factor_and_colours_follow_the_gene_subset(rich):
    """anndata's view semantics, which a hand-rolled assembler simply would not have.

    Kills: var built by positional slicing with no category pruning; a colour
    list truncated (`[:2]`) rather than index-selected; the reset-on-length-
    mismatch branch dropped; `uns` rebuilt and losing its tagged envelopes.
    """
    import pyscx

    path, _ = rich
    out = pyscx.open(path).to_anndata(var_names=NAMES)

    assert list(out.var["chr"].cat.categories) == ["2", "4"]
    assert list(out.uns["chr_colors"]) == ["#c2", "#c4"]  # truncation would give #c1,#c2
    assert "band_colors" not in out.uns
    assert list(out.var["band"].cat.categories) == ["p", "q"]  # still fully used
    assert out.uns["species"] == "human"
    np.testing.assert_array_equal(out.uns["arr"], [[1, 2], [3, 4]])


def test_raw_is_not_projected_by_var_names(rich):
    """anndata never var-slices `.raw` — it passes only the obs index to it.

    So a gene projection leaves raw on its own, wider axis. That is also why a
    `var_names=` read of a raw-bearing file still pays for raw in full;
    `raw=False` is the way out.
    """
    import pyscx

    path, source = rich
    out = pyscx.open(path).to_anndata(var_names=NAMES)
    assert out.n_vars == 3
    assert out.raw.shape == (N_OBS, RAW_N_VARS)
    assert list(out.raw.var_names) == [f"r{i}" for i in range(RAW_N_VARS)]
    np.testing.assert_allclose(_dense(out.raw.X), source["raw"])


def test_raw_false_composes_with_var_names(rich):
    import warnings

    import pyscx

    path, _ = rich
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = pyscx.open(path).to_anndata(var_names=NAMES, raw=False)
    assert out.raw is None
    assert out.n_vars == 3
    assert not [w for w in caught if str(w.message).startswith("dropped_raw:")]


def test_projection_to_a_single_gene(rich):
    """The degenerate width, where `n_cols - 1` arithmetic and the reorder path
    are easiest to get wrong."""
    import pyscx

    path, source = rich
    out = pyscx.open(path).to_anndata(var_names=["g5"])
    assert out.shape == (N_OBS, 1)
    np.testing.assert_allclose(_dense(out.X), source["x"][:, [5]])
    assert out.varp["corr"].shape == (1, 1)
    assert out.varm["PCs"].shape == (1, 2)
    assert out.layers["counts"].shape == (N_OBS, 1)


def test_empty_var_names_still_raises(rich):
    """`var_names=[]` raises rather than returning a zero-gene matrix.

    Pinned because "just return an empty projection" is the natural temptation
    when rewriting `resolve_var_names_to_indices`'s only eager caller. The
    cloud path answers differently; that divergence is known and unchanged here.
    """
    import pyscx

    path, _ = rich
    with pytest.raises(RuntimeError, match="None of the requested var_names"):
        pyscx.open(path).to_anndata(var_names=[])


def test_strict_var_names_false_keeps_the_right_columns(rich):
    """A dropped unknown name must not shift the surviving columns.

    A count-only assertion cannot see an off-by-one here, which is why this
    checks values.
    """
    import pyscx

    path, source = rich
    out = pyscx.open(path).to_anndata(
        var_names=["g9", "nope", "g1", "also_nope", "g3"], strict_var_names=False
    )
    assert list(out.var_names) == ["g1", "g3", "g9"]
    np.testing.assert_allclose(_dense(out.X), source["x"][:, SORTED_IDX])


def test_request_order_reaches_every_var_axis_member(rich):
    """Under `preserve_var_order`, X, layers, var, varm and varp must all be in
    request order — not X alone.

    Kills the sorted setter being used where the ordered one is needed, which
    leaves layers or varm transposed relative to X at the right shape.
    """
    import pyscx

    path, source = rich
    out = pyscx.open(path).to_anndata(var_names=NAMES, preserve_var_order=True)
    assert list(out.var_names) == NAMES
    np.testing.assert_allclose(_dense(out.X), source["x"][:, REQUEST_IDX])
    np.testing.assert_allclose(_dense(out.layers["counts"]), source["counts"][:, REQUEST_IDX])
    np.testing.assert_allclose(out.varm["PCs"], source["pcs"][REQUEST_IDX])
    np.testing.assert_allclose(
        _dense(out.varp["corr"]), source["corr"][np.ix_(REQUEST_IDX, REQUEST_IDX)]
    )


def test_slot_filters_that_exclude_a_slot_compose_with_a_projection(rich):
    """An assembler that unconditionally projects varm/varp must not trip over a
    slot whose bridge was never built (PR H's empty-filter gate)."""
    import pyscx

    path, source = rich
    out = pyscx.open(path).to_anndata(var_names=NAMES, varm=[], varp=[], obsp=[], layers=[])
    assert len(out.varm) == 0 and len(out.varp) == 0
    assert len(out.obsp) == 0 and len(out.layers) == 0
    np.testing.assert_allclose(_dense(out.X), source["x"][:, SORTED_IDX])


def test_a_file_without_layers_projects(rich, tmp_dir):
    import anndata

    import pyscx

    source = _source(seed=3)
    adata = anndata.AnnData(
        X=sp.csr_matrix(source["x"]),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(N_OBS)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(N_VARS)]),
    )
    path = str(tmp_dir / "no_layers.scx")
    pyscx.from_anndata(adata, path, shard_size=8)
    out = pyscx.open(path).to_anndata(var_names=NAMES)
    assert len(out.layers) == 0
    np.testing.assert_allclose(_dense(out.X), source["x"][:, SORTED_IDX])


# --------------------------------------------------------------------------
# Deletion vectors — the row axis the projection must not disturb
# --------------------------------------------------------------------------


def test_deletion_vectors_compose_with_var_names(tmp_dir):
    """Right shape, wrong rows is what a dropped `kept_to_global` looks like, and
    anndata validates shapes, not identities. Nothing pinned this before.
    """
    import pyscx

    source = _source(seed=7)
    path = str(tmp_dir / "deleted.scx")
    pyscx.from_anndata(_rich_adata(source), path, shard_size=8)

    mask = np.zeros(N_OBS, dtype=bool)
    mask[[1, 2, 17]] = True  # spans more than one shard
    pyscx.open(path).mark_deleted(mask)
    keep = ~mask

    out = pyscx.open(path).to_anndata(var_names=NAMES)
    assert out.shape == (int(keep.sum()), 3)
    np.testing.assert_allclose(_dense(out.X), source["x"][np.ix_(keep, SORTED_IDX)])
    np.testing.assert_allclose(
        _dense(out.layers["counts"]), source["counts"][np.ix_(keep, SORTED_IDX)]
    )
    np.testing.assert_allclose(out.obsm["X_emb"], source["emb"][keep])
    np.testing.assert_allclose(_dense(out.obsp["conn"]), source["conn"][np.ix_(keep, keep)])
    assert list(out.obs_names) == [f"c{i}" for i in np.flatnonzero(keep)]
    # varm/varp are on the untouched axis and still follow the gene projection.
    np.testing.assert_allclose(out.varm["PCs"], source["pcs"][SORTED_IDX])


# --------------------------------------------------------------------------
# The decode-loss guards, which the restructure is well placed to lose
# --------------------------------------------------------------------------


def test_a_big_layer_still_fails_loud_under_a_projection(tmp_dir):
    """The eager-layers guard lives inside the branch a projection replaces."""
    import anndata

    import pyscx

    big = np.zeros((4, 3), dtype=np.float32)
    big[0, 0] = 20_000_000.0
    adata = anndata.AnnData(
        X=sp.csr_matrix(np.array([[3, 0, 5], [0, 7, 0], [1, 0, 0], [0, 0, 2]], dtype=np.float32)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
    )
    adata.layers["counts"] = sp.csr_matrix(big)
    path = str(tmp_dir / "big_layer.scx")
    pyscx.from_anndata(adata, path)

    with pytest.raises(ValueError, match="allow_lossy"):
        pyscx.open(path).to_anndata(var_names=["g0"])
    # The guard is about the layer, not the projection: excluding layers passes.
    out = pyscx.open(path).to_anndata(var_names=["g0"], layers=[])
    assert out.shape == (4, 1)


def test_a_big_count_narrowed_to_uint32_stays_exact_under_a_projection(tmp_dir):
    """A `> 2**24` integer read into a type that can hold it must stay exact.

    `guard_decode_loss_for::<u32>` permits a `uint32` target above 2**24 by
    design, because the typed reader casts from the native `u32` stream. This
    pins that composing a gene projection with that read does not break it.

    What this test **cannot** distinguish is a native-`u32` decode from an f32
    detour, and not for want of trying: every pyscx write door routes X through
    float32, so an *odd* count above 2**24 cannot be put in a file from Python
    at all — `from_anndata` and h5ad ingest both land 20_000_001 on disk as
    20_000_000. Only a Rust-side writer can produce one, so the f32-exactness
    predicate that decides whether the projection is safe is unit-tested in Rust
    instead.
    """
    import anndata

    import pyscx

    big = 20_000_000  # even, so f32-exact; see the docstring
    dense = np.zeros((4, 3), dtype=np.int32)
    dense[0, 0] = big
    dense[1, 1] = 7
    adata = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
    )
    path = str(tmp_dir / "big_uint32.scx")
    pyscx.from_anndata(adata, path)
    assert pyscx.open(path).max_value == big

    out = pyscx.open(path).to_anndata(var_names=["g0", "g1"], data_dtype="uint32")
    assert out.X.dtype == np.uint32
    assert int(_dense(out.X)[0, 0]) == big
    # And the default f32 read of the same file still fails loud.
    with pytest.raises(ValueError, match="allow_lossy"):
        pyscx.open(path).to_anndata(var_names=["g0"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"data_dtype": "uint16"},
        {"container": "dense"},
        {"container": "dense", "data_dtype": "uint8"},
    ],
    ids=["narrow_csr", "dense", "dense_narrow"],
)
def test_non_default_plans_compose_with_a_projection(rich, kwargs):
    """Three different X assemblers; a CSR-shaped test walks past two of them."""
    import pyscx

    path, source = rich
    out = pyscx.open(path).to_anndata(var_names=NAMES, layers=[], **kwargs)
    assert out.shape == (N_OBS, 3)
    np.testing.assert_allclose(
        np.asarray(_dense(out.X), dtype=np.float64), source["x"][:, SORTED_IDX]
    )


# --------------------------------------------------------------------------
# preserve_slots=True + obs_filter + var_names — the branch that shares the
# projected route and filters rows as well as columns
# --------------------------------------------------------------------------


def test_preserve_slots_with_var_names_keeps_raw_aligned(rich):
    """`.raw` must follow the obs filter, and it is not enough to check its shape.

    anndata does not var-slice `.raw` but it does obs-slice it, so a projected
    read that attaches raw *after* a row-filtering slice hands a filtered
    AnnData a full-height raw — and `Raw` takes the parent's row count and the
    assignee's matrix without checking they agree. The result reports
    `raw.shape == (K, raw_n_vars)` while `raw.X.shape` is `(N, raw_n_vars)`,
    which is the same "right shape, wrong rows" class this path already guards
    against for X and layers. Assert on `raw.X`, not on `raw.shape`.
    """
    import warnings

    import pyscx

    path, source = rich
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pyscx.open(path).to_anndata(
            var_names=NAMES, obs_filter="group == 'a'", preserve_slots=True
        )
    keep = np.arange(N_OBS) % 2 == 0
    assert out.shape == (int(keep.sum()), 3)
    assert out.raw is not None
    assert out.raw.X.shape == (int(keep.sum()), RAW_N_VARS), (
        "raw.X must be row-filtered, not merely reported as filtered"
    )
    np.testing.assert_allclose(_dense(out.raw.X), source["raw"][keep])


def test_preserve_slots_with_var_names_equals_full_then_sliced(rich):
    """The same differential the var-only path gets, with a row mask as well."""
    import warnings

    from anndata.tests.helpers import assert_equal

    import pyscx

    path, _ = rich
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        projected = pyscx.open(path).to_anndata(
            var_names=NAMES, obs_filter="group == 'a'", preserve_slots=True
        )
    full = pyscx.open(path).to_anndata()
    mask = full.obs["group"].to_numpy() == "a"
    sliced = full[mask, np.array(SORTED_IDX)].copy()

    assert list(projected.var_names) == list(sliced.var_names)
    assert list(projected.obs_names) == list(sliced.obs_names)
    assert_equal(projected, sliced)


def test_preserve_slots_with_var_names_and_deletions(tmp_dir):
    """Row filter composed on top of a deletion vector, with the gene projection.

    Raw is dropped on a deletion-vector file, so this arm is about X, layers and
    the obs-axis members landing on the right cells.
    """
    import warnings

    import pyscx

    source = _source(seed=11)
    path = str(tmp_dir / "ps_deleted.scx")
    pyscx.from_anndata(_rich_adata(source), path, shard_size=8)
    deleted = np.zeros(N_OBS, dtype=bool)
    deleted[[0, 9, 20]] = True
    pyscx.open(path).mark_deleted(deleted)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pyscx.open(path).to_anndata(
            var_names=NAMES, obs_filter="group == 'a'", preserve_slots=True
        )

    live = ~deleted
    wanted = live & (np.arange(N_OBS) % 2 == 0)
    assert out.shape == (int(wanted.sum()), 3)
    np.testing.assert_allclose(_dense(out.X), source["x"][np.ix_(wanted, SORTED_IDX)])
    np.testing.assert_allclose(
        _dense(out.layers["counts"]), source["counts"][np.ix_(wanted, SORTED_IDX)]
    )
    np.testing.assert_allclose(out.obsm["X_emb"], source["emb"][wanted])
    assert list(out.obs_names) == [f"c{i}" for i in np.flatnonzero(wanted)]
