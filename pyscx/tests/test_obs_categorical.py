"""`Experiment.obs_categorical` / `obs_categorical_many` (data-load Phase 1, 1C).

The numpy-level `(codes, categories)` accessor. Parity ground truth is always
`read_obs()` → pandas, so a divergence between the streaming fold and the
assembling path fails here rather than in a consumer's vocabulary map.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import pyscx


def _write_sharded(path, cell_types, *, shard_size, extra=None):
    """Write an `.scx` whose obs is sharded, with `cell_types` as a categorical.

    `shard_size` below `len(cell_types)` forces sharded obs, which is what makes
    the cross-shard vocabulary unify reachable at all.
    """
    import anndata

    n_obs = len(cell_types)
    n_vars = 6
    rng = np.random.default_rng(0)
    dense = rng.integers(0, 20, size=(n_obs, n_vars)).astype(np.float32)
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(cell_types)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    if extra is not None:
        for k, v in extra.items():
            obs[k] = v
    var = pd.DataFrame(
        {"gene_id": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    adata = anndata.AnnData(X=sp.csr_matrix(dense), obs=obs, var=var)
    pyscx.from_anndata(adata, str(path), shard_size=shard_size)
    return str(path)


def _expected(path, col):
    """Ground truth `(codes, categories)` via the assembling pandas path."""
    exp = pyscx.open(path)
    series = exp.read_obs()[col]
    cat = series if isinstance(series.dtype, pd.CategoricalDtype) else series.astype("category")
    return np.asarray(cat.cat.codes, dtype=np.int32), list(cat.cat.categories)


def _decode(codes, categories):
    return [None if c < 0 else categories[c] for c in codes]


class TestObsCategorical:
    def test_matches_read_obs_across_shards(self, tmp_dir):
        # Disjoint-ish per-shard vocabularies: with shard_size=4 the shards see
        # {A,B}, {C,A}, {B,C} — each shard's local dictionary differs from its
        # siblings, so a broken local->global remap shows up as wrong values.
        cell_types = ["A", "B", "A", "B", "C", "A", "C", "B", "C", "A", "B", "C"]
        path = _write_sharded(tmp_dir / "cat.scx", cell_types, shard_size=4)

        exp = pyscx.open(path)
        # The premise is *obs* sharding, not X sharding: the cross-shard
        # vocabulary unify is unreachable on a single-section obs layout, so a
        # fixture that only sharded X would test nothing new here.
        assert exp.obs_metadata_shard_count > 1, (
            "premise: obs must be sharded, got "
            f"{exp.obs_metadata_shard_count} obs shard(s)"
        )

        codes, categories = exp.obs_categorical("cell_type")
        assert codes.dtype == np.int32
        assert len(codes) == len(cell_types)
        # Values, not code identity: category *order* is first-seen here and
        # lexicographic in pandas, so only the decoded strings must agree.
        assert _decode(codes, categories) == cell_types
        assert sorted(categories) == ["A", "B", "C"]

    def test_decoded_values_match_pandas_ground_truth(self, tmp_dir):
        cell_types = ["x", "y", "x", "z", "z", "y", "x", "y"]
        path = _write_sharded(tmp_dir / "cat2.scx", cell_types, shard_size=3)

        codes, categories = pyscx.open(path).obs_categorical("cell_type")
        exp_codes, exp_cats = _expected(path, "cell_type")
        assert _decode(codes, categories) == _decode(exp_codes, exp_cats)

    def test_single_shard_file(self, tmp_dir):
        cell_types = ["a", "b", "a"]
        path = _write_sharded(tmp_dir / "one.scx", cell_types, shard_size=1000)
        codes, categories = pyscx.open(path).obs_categorical("cell_type")
        assert _decode(codes, categories) == cell_types

    def test_plain_string_column_is_accepted(self, tmp_dir):
        # A non-categorical (object) obs column: `from_anndata` writes it as a
        # plain string column rather than dictionary-encoded.
        cell_types = ["A", "B", "A", "C"]
        path = _write_sharded(
            tmp_dir / "plain.scx",
            cell_types,
            shard_size=2,
            extra={"donor": ["d1", "d2", "d1", "d3"]},
        )
        codes, categories = pyscx.open(path).obs_categorical("donor")
        assert _decode(codes, categories) == ["d1", "d2", "d1", "d3"]

    def test_nulls_code_as_minus_one(self, tmp_dir):
        # A categorical with a missing value: pandas codes it -1, and so must we.
        cell_types = pd.Categorical(["A", None, "B", "A"])
        path = _write_sharded(tmp_dir / "null.scx", cell_types, shard_size=2)

        codes, categories = pyscx.open(path).obs_categorical("cell_type")
        assert codes[1] == -1, f"missing value must code as -1, got {codes[1]}"
        assert "NaN" not in categories, (
            f"null must not become a category: {categories}"
        )
        assert _decode(codes, categories) == ["A", None, "B", "A"]

    def test_literal_nan_string_is_a_category_not_a_null(self, tmp_dir):
        # The distinction the training-internal CategoryDict deliberately drops:
        # a literal "NaN" *string* is real data. Conflating it with a null would
        # silently merge a genuine category into the missing bucket.
        cell_types = ["NaN", "A", "NaN"]
        path = _write_sharded(tmp_dir / "nanstr.scx", cell_types, shard_size=2)

        codes, categories = pyscx.open(path).obs_categorical("cell_type")
        assert (codes >= 0).all(), "no row is missing, so no code may be -1"
        assert "NaN" in categories
        assert _decode(codes, categories) == cell_types

    def test_rejects_numeric_column(self, tmp_dir):
        path = _write_sharded(
            tmp_dir / "num.scx",
            ["A", "B", "A", "B"],
            shard_size=2,
            extra={"n_counts": [1, 2, 3, 4]},
        )
        with pytest.raises(ValueError, match="string/categorical"):
            pyscx.open(path).obs_categorical("n_counts")

    def test_unknown_column_raises(self, tmp_dir):
        path = _write_sharded(tmp_dir / "unk.scx", ["A", "B"], shard_size=1)
        with pytest.raises(Exception, match="nope"):
            pyscx.open(path).obs_categorical("nope")


class TestObsCategoricalMany:
    def test_returns_one_result_per_column_in_order(self, tmp_dir):
        cell_types = ["A", "B", "A", "C"]
        donors = ["d1", "d2", "d1", "d2"]
        path = _write_sharded(
            tmp_dir / "many.scx", cell_types, shard_size=2, extra={"donor": donors}
        )

        exp = pyscx.open(path)
        out = exp.obs_categorical_many(["donor", "cell_type"])
        assert len(out) == 2

        # Order must follow the request, not the on-disk schema order.
        assert _decode(*out[0]) == donors
        assert _decode(*out[1]) == cell_types

        # And must agree with the single-column accessor.
        for name, (codes, cats) in zip(["donor", "cell_type"], out):
            single = exp.obs_categorical(name)
            np.testing.assert_array_equal(codes, single[0])
            assert cats == single[1]

    def test_empty_column_list_returns_empty(self, tmp_dir):
        path = _write_sharded(tmp_dir / "empty.scx", ["A", "B"], shard_size=1)
        assert pyscx.open(path).obs_categorical_many([]) == []

    def test_unknown_column_raises_before_any_work(self, tmp_dir):
        path = _write_sharded(tmp_dir / "unk2.scx", ["A", "B"], shard_size=1)
        with pytest.raises(Exception, match="nope"):
            pyscx.open(path).obs_categorical_many(["cell_type", "nope"])


def test_obs_categorical_does_not_go_through_pandas(tmp_dir):
    """The accessor must return numpy + a plain list, not a pandas object.

    The point of 1C is skipping the Arrow-IPC -> pyarrow -> pandas hop that
    `read_obs` takes; returning a Series would defeat it while passing every
    value assertion above.
    """
    path = _write_sharded(tmp_dir / "types.scx", ["A", "B", "A"], shard_size=2)
    codes, categories = pyscx.open(path).obs_categorical("cell_type")
    assert isinstance(codes, np.ndarray), type(codes)
    assert codes.dtype == np.int32
    assert isinstance(categories, list), type(categories)
    assert all(isinstance(c, str) for c in categories)


def test_obs_categorical_after_append_mixes_encodings(tmp_dir):
    """`append` is the real producer of mixed-encoding obs.

    It decodes categoricals to plain strings before writing (`unify_dict_columns`),
    so an appended file carries dictionary-encoded base shards *and* plain string
    appended shards on the same column. Both must fold into one vocabulary — and
    an appended value that is new to the base must get its own code rather than
    colliding with an existing one.

    This is the end-to-end arm; the on-disk premise (that the two shard
    encodings genuinely differ) is asserted directly by the Rust test
    `test_obs_categorical_handles_mixed_dictionary_and_plain_shards`, which can
    inspect per-shard dtypes.
    """
    base = _write_sharded(tmp_dir / "base.scx", ["A", "B", "A", "B"], shard_size=2)
    extra = _write_sharded(tmp_dir / "extra.scx", ["B", "C"], shard_size=2)

    pyscx.append(base, extra)

    exp = pyscx.open(base)
    assert exp.n_obs == 6, "premise: the append must have landed"
    codes, categories = exp.obs_categorical("cell_type")

    assert _decode(codes, categories) == ["A", "B", "A", "B", "B", "C"]
    assert sorted(categories) == ["A", "B", "C"], (
        f"mixed encodings must not duplicate categories: {categories}"
    )
    # "B" appears in both the base (dictionary) and the appended (plain) shards
    # and must resolve to a single code.
    b_code = categories.index("B")
    assert [i for i, c in enumerate(codes) if c == b_code] == [1, 3, 4]

    # And the streaming fold must still agree with the assembling path.
    exp_codes, exp_cats = _expected(base, "cell_type")
    assert _decode(codes, categories) == _decode(exp_codes, exp_cats)


def test_obs_categorical_is_physical_row_space(tmp_dir):
    """`codes` is indexed in PHYSICAL obs row space, not logical.

    On a file with deletion vectors `n_obs` (logical) < `n_obs_physical`, and
    `obs_categorical` returns one code per *physical* row — matching `read_obs`.
    A consumer that indexes codes by a logical row id gets a correctly shaped
    array of wrong rows, which is the worst failure mode available, so the
    contract is pinned here rather than only documented.

    Why it matters concretely: state3's `_ScxBackend` refuses files with deletion
    vectors for exactly this reason — its catalog addresses cells by
    `(file_idx, cell_idx)`, so physical-space columns against a logical `n_cells`
    would make every index point at the wrong cell. Anything building a global
    vocabulary from these codes needs the same guard.
    """
    cell_types = ["A", "B", "C", "D"]
    path = _write_sharded(tmp_dir / "del.scx", cell_types, shard_size=2)

    delete_mask = np.zeros(len(cell_types), dtype=bool)
    delete_mask[1] = True
    exp = pyscx.open(path)
    exp.mark_deleted(delete_mask)

    exp2 = pyscx.open(path)
    assert exp2.n_obs == 3, "premise: one row must be logically deleted"
    assert exp2.n_obs_physical == 4, "premise: physical count is unchanged"

    codes, categories = exp2.obs_categorical("cell_type")
    assert len(codes) == exp2.n_obs_physical == 4, (
        f"codes must cover the physical axis, got {len(codes)}"
    )
    # The deleted row is still present, in its physical position.
    assert _decode(codes, categories) == cell_types

    # And it agrees with read_obs, which is physical for the same reason.
    assert len(pyscx.open(path).read_obs()) == 4
