"""Categoricals survive every in-place obs writer.

`from_anndata` writes a pandas categorical as an Arrow dictionary with the
`scx.categorical.ordered` field stamp, and every reader rebuilds
`pd.Categorical` from it. The in-place obs writers — `attach_obs_columns`,
`obs_import`, `doublet_import`, `cellbender_import`, `modify_metadata(obs=)` —
used to run the rebuilt obs table through a dictionary→string cast on its way
back to disk, so after any of them *every* categorical obs column came back from
`read_obs()` as `object`, its category list and `ordered` bit gone. arc-reactor
carried a "categorical → object after any writer" caveat and dtype-tolerant
comparers because of it.

The contract pinned here, for each writer: the target's existing categoricals
keep their dtype, declared category order, unused levels and `ordered` bit; a
categorical the *source* brings in lands the same way; values are unchanged.

Each arm runs over both obs layouts, because they are two code paths: a sharded
obs is rewritten shard by shard (streamed), a single legacy section is read
whole and re-sharded (materialised). Every arm asserts which layout it got.

The fixture is deliberately reorder-sensitive (copied from
`test_ordered_categorical.py`): `phase`'s declared order is neither alphabetical
nor first-appearance order, and level `"M"` is declared and used by no cell. A
writer that rebuilt the vocabulary from the data would pass a weaker fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyscx

anndata = pytest.importorskip("anndata")
sparse = pytest.importorskip("scipy.sparse")

N_OBS = 12
PHASE_LEVELS = ["G1", "S", "G2M", "M"]
BATCH_LEVELS = ["a", "b"]


def _phase_values(n):
    return [["G2M", "G1", "S"][i % 3] for i in range(n)]


def _adata(n_obs=N_OBS):
    rng = np.random.default_rng(0)
    x = sparse.csr_matrix(rng.integers(0, 5, size=(n_obs, 3)).astype(np.float32))
    phase = pd.Categorical(_phase_values(n_obs), categories=PHASE_LEVELS, ordered=True)
    batch = pd.Categorical(
        [["b", "a"][i % 2] for i in range(n_obs)], categories=BATCH_LEVELS, ordered=False
    )
    obs = pd.DataFrame(
        {"phase": phase, "batch": batch, "n_counts": np.arange(n_obs, dtype=np.float64)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"kind": pd.Categorical(["coding", "lnc", "coding"], categories=["lnc", "coding"])},
        index=[f"gene_{i}" for i in range(3)],
    )
    return anndata.AnnData(X=x, obs=obs, var=var)


SHARDED = pytest.param(4, id="sharded-obs")
SINGLE = pytest.param(64, id="single-obs")


@pytest.fixture(params=[SHARDED, SINGLE])
def target(tmp_path, request):
    """`(path, barcodes, shard_size)` — an SCX file with the reorder-sensitive obs."""
    adata = _adata()
    path = str(tmp_path / "t.scx")
    pyscx.from_anndata(adata, path, shard_size=request.param)
    _assert_layout(path, request.param)
    return path, list(adata.obs_names), request.param


def _assert_layout(path, shard_size):
    count = pyscx.open(path).obs_metadata_shard_count
    if shard_size < N_OBS:
        assert count > 1, "fixture was meant to have a sharded obs"
    else:
        assert count in (0, 1), "fixture was meant to have a single obs section"


def _assert_target_factors_intact(obs):
    """Both of the target's factors survive whole: dtype, order, unused level, bit."""
    assert isinstance(obs["phase"].dtype, pd.CategoricalDtype), obs["phase"].dtype
    assert obs["phase"].cat.ordered is True
    assert list(obs["phase"].cat.categories) == PHASE_LEVELS
    assert list(obs["phase"].astype(str)) == _phase_values(len(obs))

    assert isinstance(obs["batch"].dtype, pd.CategoricalDtype), obs["batch"].dtype
    assert obs["batch"].cat.ordered is False
    assert list(obs["batch"].cat.categories) == BATCH_LEVELS


# The categorical a caller attaches: declared order is not alphabetical, "mid"
# is declared and used by no row.
NEW_LEVELS = ["hi", "lo", "mid"]


def _new_categorical(n):
    return pd.Categorical([["lo", "hi"][i % 2] for i in range(n)], categories=NEW_LEVELS, ordered=True)


def _assert_new_column(obs, name, covered):
    assert isinstance(obs[name].dtype, pd.CategoricalDtype), obs[name].dtype
    assert obs[name].cat.ordered is True
    assert list(obs[name].cat.categories) == NEW_LEVELS
    for cell in obs.index:
        if cell in covered:
            assert obs.loc[cell, name] == covered[cell]
        else:
            assert pd.isna(obs.loc[cell, name]), "an uncovered row is null, not a level"


# ---------------------------------------------------------------------------
# attach_obs_columns
# ---------------------------------------------------------------------------


def test_attach_obs_columns_key_joined(target):
    path, bc, _ = target
    # Ten of twelve cells, reversed relative to the file.
    covered_cells = list(reversed(bc[:10]))
    new = _new_categorical(len(covered_cells))
    df = pd.DataFrame({"call": new}, index=covered_cells)

    r = pyscx.attach_obs_columns(path, df)
    assert r["n_matched"] == 10

    obs = pyscx.open(path).read_obs()
    _assert_target_factors_intact(obs)
    _assert_new_column(obs, "call", dict(zip(covered_cells, new.astype(str))))


def test_attach_obs_columns_positional(target):
    path, bc, _ = target
    new = _new_categorical(len(bc))
    df = pd.DataFrame({"call": new})

    pyscx.attach_obs_columns(path, df, positional=True)

    obs = pyscx.open(path).read_obs()
    _assert_target_factors_intact(obs)
    _assert_new_column(obs, "call", dict(zip(bc, new.astype(str))))


# ---------------------------------------------------------------------------
# obs_import / doublet_import / cellbender_import
# ---------------------------------------------------------------------------


def test_obs_import_keeps_the_targets_categoricals(target, tmp_path):
    path, bc, _ = target
    csv = tmp_path / "calls.csv"
    pd.DataFrame({"barcode": list(reversed(bc)), "score": np.linspace(0, 1, len(bc))}).to_csv(
        csv, index=False
    )

    pyscx.obs_import(path, str(csv))

    obs = pyscx.open(path).read_obs()
    _assert_target_factors_intact(obs)
    # approx: the scores went through CSV text, and a 1-ULP round-off there
    # is a parser property, not what this test is about.
    assert obs["score"].tolist() == pytest.approx(list(reversed(np.linspace(0, 1, len(bc)))))


def test_doublet_import_keeps_the_targets_categoricals(target, tmp_path):
    path, bc, _ = target
    csv = tmp_path / "scrub.csv"
    pd.DataFrame(
        {
            "barcode": bc,
            "doublet_score": np.linspace(0, 1, len(bc)),
            "predicted_doublet": [True, False] * (len(bc) // 2),
        }
    ).to_csv(csv, index=False)

    pyscx.doublet_import(path, str(csv), tool="scrublet")

    obs = pyscx.open(path).read_obs()
    _assert_target_factors_intact(obs)
    assert "scrublet_score" in obs.columns


@pytest.mark.skipif(not pyscx._HAS_HDF5, reason="cellbender_import needs the hdf5 feature")
def test_cellbender_import_keeps_the_targets_categoricals(target, tmp_path):
    pytest.importorskip("h5py")
    from test_cellbender import _write_cellbender_h5

    path, bc, _ = target
    genes = list(pyscx.open(path).read_var().index)
    cb = tmp_path / "cb.h5"
    _write_cellbender_h5(cb, list(reversed(bc)), genes, lambda b: bc.index(b) + 1)

    pyscx.cellbender_import(path, str(cb))

    obs = pyscx.open(path).read_obs()
    _assert_target_factors_intact(obs)
    assert (obs["cellbender_status"] == "present").all()


# ---------------------------------------------------------------------------
# modify_metadata
# ---------------------------------------------------------------------------


def test_modify_metadata_obs_lands_the_callers_categories(target):
    """The replace path: the *caller's* category order lands, not the file's."""
    path, _, _ = target
    obs = pyscx.open(path).read_obs()
    reordered = ["M", "G2M", "S", "G1"]
    obs["phase"] = obs["phase"].cat.reorder_categories(reordered)
    obs["extra"] = _new_categorical(len(obs))

    pyscx.modify_metadata(path, obs=obs)

    back = pyscx.open(path).read_obs()
    assert isinstance(back["phase"].dtype, pd.CategoricalDtype), back["phase"].dtype
    assert back["phase"].cat.ordered is True
    assert list(back["phase"].cat.categories) == reordered
    assert list(back["phase"].astype(str)) == _phase_values(len(back))
    assert isinstance(back["batch"].dtype, pd.CategoricalDtype)
    assert list(back["batch"].cat.categories) == BATCH_LEVELS
    _assert_new_column(back, "extra", dict(zip(back.index, obs["extra"].astype(str))))


def test_modify_metadata_var_keeps_categoricals(target):
    """Control arm: `patch.var` never went through the cast, so this held
    before — pinned so the two axes cannot drift apart."""
    path, _, _ = target
    pyscx.modify_metadata(path, var=pyscx.open(path).read_var())

    var = pyscx.open(path).read_var()
    assert isinstance(var["kind"].dtype, pd.CategoricalDtype)
    assert list(var["kind"].cat.categories) == ["lnc", "coding"]


# ---------------------------------------------------------------------------
# Legacy files that already mix representations
# ---------------------------------------------------------------------------


def test_a_file_mixing_dictionary_and_plain_shards_still_takes_an_attach(tmp_path):
    """A file written before this fix can carry one column as a dictionary in
    some obs shards and plain strings in others. `pyscx.append` still produces
    exactly that layout (it decodes categoricals before writing new shards and
    leaves the base shards alone), so it is the fixture here. Such a file must
    still read, still take an attach, and hand the column back as a category
    with the union vocabulary."""

    def _write(name, cells, kinds):
        n = len(cells)
        obs = pd.DataFrame(
            {"cell_type": pd.Categorical(kinds)}, index=cells
        )
        x = sparse.csr_matrix(np.ones((n, 3), dtype=np.float32))
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(3)])
        path = str(tmp_path / name)
        pyscx.from_anndata(anndata.AnnData(X=x, obs=obs, var=var), path, shard_size=4)
        return path

    base = _write("base.scx", [f"b{i}" for i in range(8)], ["T", "B"] * 4)
    extra = _write("extra.scx", [f"e{i}" for i in range(8)], ["NK", "B"] * 4)
    pyscx.append(base, extra)
    before = pyscx.open(base).read_obs()["cell_type"].astype(str).tolist()

    df = pd.DataFrame({"score": np.arange(16, dtype=np.float64)}, index=[f"b{i}" for i in range(8)] + [f"e{i}" for i in range(8)])
    pyscx.attach_obs_columns(base, df)

    exp = pyscx.open(base)
    obs = exp.read_obs()
    assert isinstance(obs["cell_type"].dtype, pd.CategoricalDtype), obs["cell_type"].dtype
    assert obs["cell_type"].astype(str).tolist() == before
    assert set(obs["cell_type"].cat.categories) == {"T", "B", "NK"}
    assert obs["score"].tolist() == list(range(16))
    codes, cats = exp.obs_categorical("cell_type")
    assert [cats[c] for c in codes] == before


# ---------------------------------------------------------------------------
# Numeric categoricals
# ---------------------------------------------------------------------------
#
# `pd.Categorical([1, 2, 3])` reaches Arrow as `Dictionary(_, Int64)` — a real
# on-disk shape for cluster labels. The first version of this PR's assembler fix
# interned only string / boolean dictionary values, so an integer categorical
# still lost its unused level and declared order on a small sharded file
# (round-1 finding, codex; repro below is codex's).

NUM_LEVELS = [3, 2, 1, 4]


def _numeric_categorical(n):
    return pd.Categorical([[2, 1, 3][i % 3] for i in range(n)], categories=NUM_LEVELS, ordered=True)


def _assert_numeric_intact(col, n):
    assert isinstance(col.dtype, pd.CategoricalDtype), col.dtype
    assert col.cat.ordered is True
    assert list(col.cat.categories) == NUM_LEVELS
    assert list(col) == [[2, 1, 3][i % 3] for i in range(n)]


def test_attach_obs_columns_numeric_categorical(target):
    path, bc, _ = target
    df = pd.DataFrame({"cluster": _numeric_categorical(len(bc))}, index=bc)

    pyscx.attach_obs_columns(path, df)

    obs = pyscx.open(path).read_obs()
    _assert_target_factors_intact(obs)
    _assert_numeric_intact(obs["cluster"], len(bc))


def test_modify_metadata_obs_numeric_categorical(target):
    path, bc, _ = target
    obs = pyscx.open(path).read_obs()
    obs["cluster"] = _numeric_categorical(len(bc))

    pyscx.modify_metadata(path, obs=obs)

    back = pyscx.open(path).read_obs()
    _assert_target_factors_intact(back)
    _assert_numeric_intact(back["cluster"], len(bc))


# ---------------------------------------------------------------------------
# filter_obs(...).collect() keeps the opposite contract
# ---------------------------------------------------------------------------


def test_filtered_collect_carries_only_the_surviving_categories(target):
    """`docs/api/rust-engine.md` § Filtered-obs categorical semantics: a `collect()` result's
    categoricals carry only the categories present in the surviving rows
    (pandas' `remove_unused_categories`-on-subset rule), while `read_obs()` on
    the same file keeps the full declared list. Before this PR the prune on
    the sharded path was arrow's data-dependent dictionary merge — it fired on
    a small result and not on a large one — and the first version of the
    shared-values fix switched it off entirely (round-1 finding, Cursor). Both
    layouts, since the legacy single-section path filters differently."""
    path, _, _ = target

    q = pyscx.open(path).query()
    # G1 sits at i % 3 == 1 (rows 1, 4, 7, 10) and batch is "a" at odd i, so
    # the conjunction keeps rows 1 and 7: one level of each factor survives.
    q.filter_obs("phase == 'G1' and batch == 'a'")
    obs = q.collect().to_anndata().obs
    assert len(obs) == 2
    assert isinstance(obs["phase"].dtype, pd.CategoricalDtype), obs["phase"].dtype
    assert list(obs["phase"].cat.categories) == ["G1"]
    assert obs["phase"].cat.ordered is True
    # The unordered control is pruned the same way.
    assert list(obs["batch"].cat.categories) == ["a"]

    full = pyscx.open(path).read_obs()
    assert list(full["phase"].cat.categories) == PHASE_LEVELS


def test_unfiltered_collect_keeps_the_declared_categories(target):
    """No obs predicate and no limit means no row subset, so `collect()` reads
    like `read_obs()`: the declared list, unused level included. The round-1 fix
    pruned every collect on both layouts (round-2 finding, codex)."""
    path, _, _ = target
    obs = pyscx.open(path).query().collect().to_anndata().obs
    assert len(obs) == N_OBS
    assert list(obs["phase"].cat.categories) == PHASE_LEVELS
    assert list(obs["batch"].cat.categories) == BATCH_LEVELS


def test_filtered_collect_with_null_rows_prunes_by_the_non_null_survivors(target):
    """A null row is not a category. The round-1 prune read every row's stored
    key without checking validity, so a null row could keep whatever category its
    arbitrary key pointed at (round-2 finding, all three reviewers). Partial
    attaches create exactly these nulls: `call` covers cells 0–5 only."""
    path, bc, _ = target
    call = pd.Categorical(["B", "C"] * 3, categories=["A", "B", "C", "D"], ordered=True)
    pyscx.attach_obs_columns(path, pd.DataFrame({"call": call}, index=bc[:6]))

    q = pyscx.open(path).query()
    q.filter_obs("n_counts < 8")  # rows 0–7: six with a call, two null
    obs = q.collect().to_anndata().obs
    assert len(obs) == 8
    assert obs["call"].isna().sum() == 2
    assert list(obs["call"].cat.categories) == ["B", "C"]
    assert obs["call"].cat.ordered is True


def test_empty_filtered_collect_has_no_categories(target):
    """`remove_unused_categories` on an empty subset leaves no categories."""
    path, _, _ = target
    q = pyscx.open(path).query()
    q.filter_obs("phase == 'M'")  # declared, used by no cell
    obs = q.collect().to_anndata().obs
    assert len(obs) == 0
    assert isinstance(obs["phase"].dtype, pd.CategoricalDtype)
    assert list(obs["phase"].cat.categories) == []
