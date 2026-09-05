"""Tests for in-place metadata replacement: pyscx.set_uns / modify_metadata.

These wrap scx_ops::set_uns / modify_metadata — replace SCX metadata sections
(uns / obs / var / obsm / varm) without re-encoding X.
"""

import json

import numpy as np
import pandas as pd
import pytest

import pyscx


def test_set_uns_round_trip(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)

    new_uns = {"method": "updated", "k": 7, "labels": ["a", "b", "c"]}
    pyscx.set_uns(path, new_uns)

    uns = pyscx.open(path).to_anndata().uns
    assert uns["method"] == "updated"
    assert int(uns["k"]) == 7
    assert list(uns["labels"]) == ["a", "b", "c"]
    # Matrix untouched — file still opens and has the same shape.
    exp = pyscx.open(path)
    assert exp.n_obs == synthetic_adata.n_obs
    assert exp.n_vars == synthetic_adata.n_vars


def test_modify_metadata_obs_replace(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    n = synthetic_adata.n_obs

    new_obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n)],
            "donor": ["donor_Z"] * n,
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    pyscx.modify_metadata(path, obs=new_obs)

    obs = pyscx.open(path).to_anndata().obs
    assert len(obs) == n
    assert "donor" in obs.columns
    assert obs["donor"].iloc[0] == "donor_Z"


def _indexed_multi_shard(
    tmp_dir, name="indexed.scx", n=400, shard=100, index_obs=("n_counts",),
    index_var=None, index_auto_threshold=1000,
):
    """400 cells over 4 CSR shards, obs predicate index over `n_counts` (100…499).

    `index_obs=` is what makes the catalog carry a per-shard `MinMax` for
    `n_counts`, and `shard_size=` is what makes there be more than one shard to
    prune. Without both, shard pruning is unobservable and any assertion about it
    is vacuous.

    `grp` alternates A/B and is indexable but *not* indexed by default, so the
    carry-forward tests can ask for a two-column index without disturbing the
    single-column stats tests above.
    """
    import anndata
    import scipy.sparse as sp

    x = sp.csr_matrix(np.eye(n, 8, dtype=np.float32))
    obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n)],
            "grp": pd.Categorical(np.where(np.arange(n) % 2 == 0, "A", "B")),
            "n_counts": np.arange(100, 100 + n, dtype=np.int64),
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"g{i}" for i in range(8)]},
        index=[f"g{i}" for i in range(8)],
    )
    adata = anndata.AnnData(X=x, obs=obs, var=var)
    path = str(tmp_dir / name)
    # `index_auto_threshold` defaults to 1000 on `from_anndata`, so auto-detect
    # indexes every low-cardinality column unless a forced list turns it off —
    # which is why "a file with no obs index" needs `index_auto_threshold=0`
    # rather than merely omitting `index_obs`.
    pyscx.from_anndata(
        adata,
        path,
        shard_size=shard,
        index_obs=list(index_obs) if index_obs else None,
        index_var=list(index_var) if index_var else None,
        index_auto_threshold=index_auto_threshold,
    )
    return path, adata


def _matching(path, expr):
    p = pyscx.open(path).query()
    p.filter_obs(expr)
    return p.collect().n_obs


def _sections(path):
    """Section names as `validate()` reports them.

    The only pure-Python view of which sections a file actually carries: `scx
    info` shows the same list, but CI does not build `target/release/scx` before
    pytest, so a CLI-based assertion would skip rather than run.
    """
    return [name for name, _ in pyscx.open(path).validate()]


def test_modify_metadata_obs_replace_does_not_strand_rows_behind_stale_shard_stats(
    tmp_dir,
):
    """Replacing obs must not leave the catalog pruning on the old values.

    The catalog records a per-shard `[min, max]` for every indexed obs column,
    and query-time Level-1 pruning reads those *directly* — for a numeric column
    it never consults the predicate index at all. So dropping the now-stale index
    is only half the job: leave the stats behind and every shard is excluded on a
    predicate the replaced values satisfy, and the query comes back empty with no
    error.

    Ground truth is pandas on the frame that was written, not a second query.
    """
    path, _ = _indexed_multi_shard(tmp_dir)

    # Before: nothing exceeds 1500, and that empty answer is the correct one.
    assert _matching(path, "n_counts > 1500") == 0

    new_obs = pyscx.open(path).read_obs()
    new_obs["n_counts"] = new_obs["n_counts"] * 10
    pyscx.modify_metadata(path, obs=new_obs)

    want = int((new_obs["n_counts"] > 1500).sum())
    assert want == 349
    assert _matching(path, "n_counts > 1500") == want
    # The rows really are the right ones, not merely the right count.
    p = pyscx.open(path).query()
    p.filter_obs("n_counts > 1500")
    got = p.collect().to_anndata().obs
    assert got["n_counts"].min() > 1500


def test_modify_metadata_obs_replace_with_index_rebuild_keeps_pruning(tmp_dir):
    """The fix must not make a replaced-obs file permanently unprunable.

    Asking for the index back re-derives the shard stats, so Level-1 pruning
    returns — against the new values.
    """
    path, _ = _indexed_multi_shard(tmp_dir)
    new_obs = pyscx.open(path).read_obs()
    new_obs["n_counts"] = new_obs["n_counts"] * 10
    pyscx.modify_metadata(path, obs=new_obs, index_obs=["n_counts"])

    assert _matching(path, "n_counts > 1500") == int((new_obs["n_counts"] > 1500).sum())
    assert _matching(path, "n_counts > 3000") == int((new_obs["n_counts"] > 3000).sum())


# ---------------------------------------------------------------------------
# A replaced axis must not cost the file its predicate index
# ---------------------------------------------------------------------------
#
# Replacing obs wholesale cannot tell an added column from a rewritten one, so
# the op used to drop the index outright. `pyscx.doublet_consensus` — the LAST
# step of the documented doublet workflow — is a wholesale obs replacement, so
# it silently reverted `query().filter_obs(...)` to a full obs scan. Carrying
# the index forward means rebuilding it over the columns it already covered.


def test_modify_metadata_obs_replace_carries_the_index_forward(tmp_dir):
    path, _ = _indexed_multi_shard(tmp_dir, index_obs=["grp", "n_counts"])
    assert "obs_predicate_index" in _sections(path)

    pyscx.modify_metadata(path, obs=pyscx.open(path).read_obs())

    assert "obs_predicate_index" in _sections(path)
    # Both arms of the index, and both still correct against the new values.
    assert _matching(path, "n_counts > 300") == 199
    assert _matching(path, "grp == 'A'") == 200


def test_modify_metadata_var_replace_carries_the_var_index_forward(tmp_dir):
    """The same branch drops `VarPredicateIndex` on a var replacement."""
    path, _ = _indexed_multi_shard(tmp_dir, index_var=["gene_id"])
    assert "var_predicate_index" in _sections(path)

    pyscx.modify_metadata(path, var=pyscx.open(path).read_var())

    assert "var_predicate_index" in _sections(path)


def test_modify_metadata_carry_forward_records_itself_in_provenance(tmp_dir):
    """The other half of §10.5: the op neither preserved *nor reported*.

    `obs_import` already stamps `obs_predicate_index_dropped`; this is the same
    field on the same question, so one `scx info` grep answers it for either op.
    """
    path, _ = _indexed_multi_shard(tmp_dir, index_obs=["grp", "n_counts"])
    pyscx.modify_metadata(path, obs=pyscx.open(path).read_obs())

    entry = pyscx.open(path).provenance()[-1]
    assert entry["action"] == "modify_metadata"
    params = json.loads(entry["params_json"])
    assert params["obs_predicate_index_dropped"] is False
    assert params["predicate_index"]["obs_carried_forward"] is True
    assert params["predicate_index"]["obs_columns"] == ["grp", "n_counts"]


# New behaviour (not a pre-existing regression): what happens when the carry
# cannot be complete.


def test_modify_metadata_carry_forward_warns_about_a_column_it_cannot_carry(tmp_dir):
    """A carried column the new frame no longer has warns — it never raises.

    This is why carried columns go in as *preset* rather than *forced* columns:
    forced ones hard-error, which would make `doublet_consensus` start raising
    on a file whose indexed column an earlier edit had dropped.
    """
    path, _ = _indexed_multi_shard(tmp_dir, index_obs=["grp", "n_counts"])
    obs = pyscx.open(path).read_obs().drop(columns=["grp"])

    with pytest.warns(UserWarning, match="grp"):
        pyscx.modify_metadata(path, obs=obs)

    # The half that could be carried still is, and still prunes correctly.
    assert "obs_predicate_index" in _sections(path)
    assert _matching(path, "n_counts > 300") == 199


def test_modify_metadata_explicit_index_obs_wins_and_reports_the_narrowing(tmp_dir):
    """An explicit request stays authoritative — but narrowing it used to drop
    the other indexed column in silence."""
    path, _ = _indexed_multi_shard(tmp_dir, index_obs=["grp", "n_counts"])
    obs = pyscx.open(path).read_obs()

    with pytest.warns(UserWarning, match="grp"):
        pyscx.modify_metadata(path, obs=obs, index_obs=["n_counts"])

    assert "obs_predicate_index" in _sections(path)
    assert _matching(path, "n_counts > 300") == 199
    # `grp` is no longer indexed, but a full obs scan still answers it.
    assert _matching(path, "grp == 'A'") == 200


def test_modify_metadata_index_var_does_not_switch_off_the_obs_carry(tmp_dir):
    """An index request on ONE axis must not silently change the other's policy.

    `user_wants_index` is a whole-patch question — true if any index knob is set
    — so reading it per axis let `index_var=[...]` take the obs axis off
    carry-forward and onto auto-detect. Under the "omit index_* to carry"
    contract that is a footgun, not a quirk: the caller said nothing about obs.
    """
    path, _ = _indexed_multi_shard(
        tmp_dir, index_obs=["grp", "n_counts"], index_var=["gene_id"]
    )
    exp = pyscx.open(path)
    obs, var = exp.read_obs(), exp.read_var()

    # Names var only. obs must still be carried over its OWN two columns.
    pyscx.modify_metadata(path, obs=obs, var=var, index_var=["gene_id"])

    assert "obs_predicate_index" in _sections(path)
    entry = pyscx.open(path).provenance()[-1]
    params = json.loads(entry["params_json"])
    assert params["predicate_index"]["obs_columns"] == ["grp", "n_counts"]
    assert params["predicate_index"]["obs_carried_forward"] is True
    # …while var took the caller's explicit list, so it is a rebuild, not a carry.
    assert params["predicate_index"]["var_carried_forward"] is False
    assert _matching(path, "n_counts > 300") == 199
    assert _matching(path, "grp == 'A'") == 200


def test_modify_metadata_both_axes_replaced_both_carry(tmp_dir):
    """Both axes replaced, neither named — both must carry independently.

    The sibling test pins the *mixed* case (one carried, one explicitly
    rebuilt). This is the other half: with per-axis policy, nothing should make
    replacing var disturb the obs carry or vice versa.
    """
    path, _ = _indexed_multi_shard(
        tmp_dir, index_obs=["grp", "n_counts"], index_var=["gene_id"]
    )
    exp = pyscx.open(path)
    obs, var = exp.read_obs(), exp.read_var()

    pyscx.modify_metadata(path, obs=obs, var=var)

    sections = _sections(path)
    assert "obs_predicate_index" in sections
    assert "var_predicate_index" in sections
    params = json.loads(pyscx.open(path).provenance()[-1]["params_json"])
    assert params["predicate_index"]["obs_carried_forward"] is True
    assert params["predicate_index"]["var_carried_forward"] is True
    assert params["predicate_index"]["obs_columns"] == ["grp", "n_counts"]
    assert params["predicate_index"]["var_columns"] == ["gene_id"]
    assert _matching(path, "n_counts > 300") == 199


def test_modify_metadata_accepts_a_pathlike(tmp_dir):
    """Every other path-taking entry point coerces `os.PathLike`; these two did
    not, so a `pathlib.Path` raised `TypeError: 'PosixPath' object is not an
    instance of 'str'`."""
    import pathlib

    path, _ = _indexed_multi_shard(tmp_dir)
    p = pathlib.Path(path)
    pyscx.modify_metadata(p, obs=pyscx.open(path).read_obs())
    pyscx.set_uns(p, {"state": "set-via-pathlib"})
    assert pyscx.open(path).read_uns()["state"] == "set-via-pathlib"


def test_modify_metadata_without_an_index_stays_without_one(tmp_dir):
    """Carry-forward carries; it does not invent. A file with no obs index must
    not grow one on an obs replacement.

    Built via `compact` without `--index-*`, which is the documented way to end
    up with an unindexed file: `from_anndata` auto-indexes by default, so simply
    omitting `index_obs` there is not enough.
    """
    src, _ = _indexed_multi_shard(tmp_dir)
    path = str(tmp_dir / "unindexed.scx")
    pyscx.compact(src, path)
    assert "obs_predicate_index" not in _sections(path)

    pyscx.modify_metadata(path, obs=pyscx.open(path).read_obs())

    assert "obs_predicate_index" not in _sections(path)


def test_modify_metadata_obsm_survives_a_later_compact(tmp_dir):
    """A first-ever in-place obsm must not vanish on the next compact.

    In-place ops write the header verbatim, so `modify_metadata` has to stamp
    `has_obsm` itself — and `compact` gates the whole obsm block on that flag
    rather than on the catalog. Reading the embedding back from the *compacted*
    file is the assertion that matters.
    """
    path, _ = _indexed_multi_shard(tmp_dir, name="obsm.scx", n=20, shard=10)
    assert "X_umap" not in pyscx.open(path).obsm_keys()

    emb = pd.DataFrame(
        np.arange(40, dtype=np.float32).reshape(20, 2), columns=["c0", "c1"]
    )
    pyscx.modify_metadata(path, obsm={"X_umap": emb})

    out = str(tmp_dir / "obsm_compacted.scx")
    pyscx.compact(path, out)
    assert "X_umap" in pyscx.open(out).obsm_keys()


def test_modify_metadata_obs_wrong_shape_raises(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    n = synthetic_adata.n_obs

    bad_obs = pd.DataFrame({"x": list(range(n - 1))})  # one row short
    with pytest.raises(ValueError, match="n_obs"):
        pyscx.modify_metadata(path, obs=bad_obs)


# On a file with deletions `obs=` is accepted in either row space — the live
# frame `read_obs()` returns (since 0.17) or the physical one
# `read_obs(logical=False)` returns — told apart by length.


def _deleted(synthetic_adata, scx_from_adata, name):
    path = scx_from_adata(synthetic_adata, name)
    pyscx.mark_deleted(path, [3, 7, 50])
    exp = pyscx.open(path)
    assert exp.n_obs == 97 and exp.n_obs_physical == 100, "premise"
    return path, exp


def test_modify_metadata_obs_accepts_the_live_frame_from_read_obs(
    synthetic_adata, scx_from_adata
):
    path, exp = _deleted(synthetic_adata, scx_from_adata, "mm_live.scx")
    obs = exp.read_obs()
    obs["score"] = np.arange(len(obs), dtype=np.float64)
    physical_index_before = exp.read_obs(logical=False).index

    pyscx.modify_metadata(path, obs=obs)

    exp = pyscx.open(path)
    assert exp.has_deletions and exp.n_obs == 97, "the deletion vector survives"
    live = exp.read_obs()
    assert live["score"].tolist() == list(range(97))
    pd.testing.assert_index_equal(live.index, obs.index)
    backed = exp.to_anndata(backed=True).obs
    assert backed["score"].tolist() == list(range(97)), "aligned with the backed obs"

    physical = exp.read_obs(logical=False)
    assert len(physical) == 100
    # Deleted rows: barcode kept, every other column null.
    pd.testing.assert_index_equal(physical.index, physical_index_before)
    for row in (3, 7, 50):
        assert pd.isna(physical["score"].iloc[row])
        assert pd.isna(physical["batch"].iloc[row])
    assert physical["score"].iloc[4] == 3.0, "the live row after a deleted one shifts by one"

    # …so the file stays joinable by its index afterwards.
    calls = pd.DataFrame({"flag": [True] * 97}, index=live.index)
    r = pyscx.attach_obs_columns(path, calls)
    assert r["n_matched"] == 97 and r["n_target_rows_absent"] == 3

    # And compact drops them for good.
    out = path.replace("mm_live.scx", "mm_live_compact.scx")
    pyscx.compact(path, out)
    assert len(pyscx.open(out).read_obs(logical=False)) == 97


def test_modify_metadata_refuses_a_reordered_live_frame(synthetic_adata, scx_from_adata):
    # Same live barcodes, different order: a `sort_values` accident, refused by
    # row. Renamed barcodes are a legitimate replace and pass.
    path, exp = _deleted(synthetic_adata, scx_from_adata, "mm_reorder.scx")
    obs = exp.read_obs()
    with pytest.raises(ValueError, match="different order"):
        pyscx.modify_metadata(path, obs=obs.sort_values("batch", kind="stable").iloc[::-1])
    assert pyscx.open(path).read_obs().equals(obs), "refused → nothing written"

    renamed = obs.copy()
    renamed.index = [f"new_{i}" for i in range(len(obs))]
    pyscx.modify_metadata(path, obs=renamed)
    assert list(pyscx.open(path).read_obs().index) == list(renamed.index)


def test_modify_metadata_live_frame_with_a_renamed_index_keeps_deleted_barcodes(
    synthetic_adata, scx_from_adata
):
    # `rename_axis` changes the index column's name; the deleted rows must still
    # get their barcodes back (levels are paired by position, not by name).
    path, exp = _deleted(synthetic_adata, scx_from_adata, "mm_axis.scx")
    before = exp.read_obs(logical=False).index
    obs = exp.read_obs().rename_axis("cell")
    obs["score"] = 1.0
    pyscx.modify_metadata(path, obs=obs)
    physical = pyscx.open(path).read_obs(logical=False)
    assert physical.index.name == "cell"
    assert list(physical.index) == list(before)
    assert physical["score"].isna().sum() == 3


def test_modify_metadata_obs_still_accepts_the_physical_frame(synthetic_adata, scx_from_adata):
    path, exp = _deleted(synthetic_adata, scx_from_adata, "mm_phys.scx")
    obs = exp.read_obs(logical=False)
    obs["score"] = np.arange(100, dtype=np.float64)

    pyscx.modify_metadata(path, obs=obs)

    exp = pyscx.open(path)
    physical = exp.read_obs(logical=False)
    assert physical["score"].tolist() == list(range(100)), "deleted rows written as handed in"
    assert exp.read_obs()["score"].tolist() == [i for i in range(100) if i not in (3, 7, 50)]


def test_modify_metadata_obs_wrong_length_on_a_deleted_file_names_both_counts(
    synthetic_adata, scx_from_adata
):
    path, _ = _deleted(synthetic_adata, scx_from_adata, "mm_bad.scx")
    for n in (96, 98):
        with pytest.raises(ValueError) as e:
            pyscx.modify_metadata(path, obs=pd.DataFrame({"x": list(range(n))}))
        msg = str(e.value)
        for needle in (f"{n} rows", "n_obs = 97", "n_obs_physical = 100", "read_obs()",
                       "read_obs(logical=False)"):
            assert needle in msg, f"{needle!r} missing from: {msg}"


def test_modify_metadata_obsm_stays_physical_length_on_a_deleted_file(
    synthetic_adata, scx_from_adata
):
    path, _ = _deleted(synthetic_adata, scx_from_adata, "mm_obsm.scx")
    with pytest.raises(ValueError, match=r"97 rows.*n_obs_physical = 100.*dense mapping"):
        pyscx.modify_metadata(path, obsm={"X_new": np.zeros((97, 2), dtype=np.float32)})
    pyscx.modify_metadata(path, obsm={"X_new": np.zeros((100, 2), dtype=np.float32)})
    assert pyscx.open(path).to_anndata(backed=True).obsm["X_new"].shape == (97, 2)


def test_modify_metadata_obs_dict_raises_typeerror(synthetic_adata, scx_from_adata):
    # Report E2: passing a column->values dict (a natural thing to try) instead
    # of a pandas DataFrame must raise a clear TypeError naming modify_metadata,
    # the parameter, and the fix — not an opaque pyarrow AttributeError.
    path = scx_from_adata(synthetic_adata)
    with pytest.raises(TypeError) as ei:
        pyscx.modify_metadata(path, obs={"qc_status": [1, 2, 3]})
    msg = str(ei.value)
    assert "modify_metadata(obs=...)" in msg
    assert "pandas DataFrame" in msg
    assert "dict" in msg


def test_modify_metadata_var_dict_raises_typeerror(synthetic_adata, scx_from_adata):
    # Same guard on the `var` parameter.
    path = scx_from_adata(synthetic_adata)
    with pytest.raises(TypeError) as ei:
        pyscx.modify_metadata(path, var={"gene_flag": [0, 1]})
    assert "modify_metadata(var=...)" in str(ei.value)


def test_modify_metadata_obs_index_rebuild(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    n = synthetic_adata.n_obs

    new_obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n)],
            "batch": pd.Categorical(np.random.choice(["A", "B"], size=n)),
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    # Request a predicate index over `batch` while replacing obs.
    pyscx.modify_metadata(path, obs=new_obs, index_obs=["batch"])

    # File reopens and the new obs is present.
    obs = pyscx.open(path).to_anndata().obs
    assert "batch" in obs.columns
    assert len(obs) == n


def test_modify_metadata_varm_replace(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    n_vars = synthetic_adata.n_vars

    loadings = np.random.randn(n_vars, 4).astype(np.float32)
    pyscx.modify_metadata(path, varm={"PCs": loadings})

    adata = pyscx.open(path).to_anndata()
    assert "PCs" in adata.varm
    assert adata.varm["PCs"].shape == (n_vars, 4)


def test_set_uns_then_rollback(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    original = dict(pyscx.open(path).to_anndata().uns)

    pyscx.set_uns(path, {"state": "mutated"})
    assert pyscx.open(path).to_anndata().uns["state"] == "mutated"

    pyscx.rollback(path)
    restored = pyscx.open(path).to_anndata().uns
    assert "state" not in restored
    # Original keys are back.
    assert restored.get("species") == original.get("species")


def test_modify_metadata_empty_patch_raises(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    with pytest.raises(ValueError):
        pyscx.modify_metadata(path)


# ---------------------------------------------------------------------------
# The pandas index envelope must survive an in-place obs write
# ---------------------------------------------------------------------------
#
# `unify_dict_columns` (scx-ops) rebuilt the obs schema with `Schema::new(...)`,
# which dropped the schema-level `pandas` envelope naming `index_columns`. The
# h5ad exporter then fell back to "field 0 is the index" — true of a
# CLI-converted file, false after an obs rewrite — so **every exported cell was
# renamed to the value of the first string column**, silently.
#
# It only fired when a categorical obs column was present (a batch with no
# dictionary column took an early return), which is why it looked intermittent:
# a file with no categoricals was fine, and so was any *second* mutation, the
# first having already decoded every categorical to plain strings. The in-place
# writers no longer run that cast at all — a categorical stays a dictionary
# through every one of them — so the envelope now has to survive on its own
# merits, on every write, with the categorical still present.


@pytest.fixture()
def categorical_obs_scx(tmp_path):
    """An SCX file whose obs has a categorical column BEFORE the index field.

    Both details matter. The categorical is what sends the batch down the
    dictionary-rebuilding path, and the index not being field 0 is what makes
    the exporter's fallback pick the wrong column.
    """
    anndata = pytest.importorskip("anndata")
    sparse = pytest.importorskip("scipy.sparse")

    n, g = 12, 5
    X = sparse.csr_matrix(np.arange(n * g, dtype=np.float32).reshape(n, g))
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(np.repeat(["T cell", "B cell", "NK"], 4)),
            "n_counts": np.arange(n, dtype=np.float64),
        },
        index=[f"AAACCT-{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(g)])
    src = tmp_path / "src.h5ad"
    anndata.AnnData(X=X, obs=obs, var=var).write_h5ad(src)

    path = str(tmp_path / "f.scx")
    pyscx.from_h5ad(str(src), path)
    return path, list(obs.index)


def _exported_obs_names(scx_path, tmp_path, name):
    anndata = pytest.importorskip("anndata")
    out = tmp_path / name
    pyscx.to_h5ad(scx_path, str(out))
    return list(anndata.read_h5ad(out).obs_names)


def test_modify_metadata_then_export_keeps_obs_names(categorical_obs_scx, tmp_path):
    """The reported bug, end to end. No column is even added."""
    path, expected = categorical_obs_scx

    obs = pyscx.open(path).read_obs()
    pyscx.modify_metadata(path, obs=obs)

    assert _exported_obs_names(path, tmp_path, "out.h5ad") == expected


def test_append_then_export_keeps_obs_names(categorical_obs_scx, tmp_path):
    """`append` runs the same code over both the old and the new obs."""
    anndata = pytest.importorskip("anndata")
    sparse = pytest.importorskip("scipy.sparse")
    path, expected = categorical_obs_scx

    n, g = 4, 5
    extra = anndata.AnnData(
        X=sparse.csr_matrix(np.ones((n, g), dtype=np.float32)),
        obs=pd.DataFrame(
            {
                "cell_type": pd.Categorical(["T cell"] * n),
                "n_counts": np.zeros(n, dtype=np.float64),
            },
            index=[f"EXTRA-{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(g)]),
    )
    extra_h5ad = tmp_path / "extra.h5ad"
    extra.write_h5ad(extra_h5ad)
    extra_scx = str(tmp_path / "extra.scx")
    pyscx.from_h5ad(str(extra_h5ad), extra_scx)   # append takes SCX, not h5ad
    pyscx.append(path, extra_scx)

    names = _exported_obs_names(path, tmp_path, "out.h5ad")
    assert names == expected + [f"EXTRA-{i}" for i in range(4)]


def test_obs_import_then_export_keeps_obs_names(categorical_obs_scx, tmp_path):
    """`attach_external_obs` shares the same helper, so `obs_import` /
    `doublet_import` / `cellbender_import` all inherited the bug."""
    path, expected = categorical_obs_scx

    table = tmp_path / "calls.csv"
    pd.DataFrame({
        "barcode": expected,
        "score": np.linspace(0, 1, len(expected)),
    }).to_csv(table, index=False)
    pyscx.obs_import(path, str(table))

    assert _exported_obs_names(path, tmp_path, "out.h5ad") == expected


def test_two_in_place_writes_keep_the_index_and_the_categorical(categorical_obs_scx, tmp_path):
    """Two mutations in a row, through two different writers.

    An earlier version of this test passed even on the pre-fix build, and its
    docstring explained why with a mechanism that depended on the bug: the
    first write stringified the categorical, so the second write's batch had no
    dictionary column and skipped the schema rebuild that dropped the envelope
    — a "self-heal". That premise is gone twice over: the in-place writers no
    longer cast categoricals at all, and `doublet_consensus`'s first run takes
    the `attach_obs_columns` seam, not `modify_metadata`. A test whose
    explanation encodes the bug is a spec for the bug, so this now pins what the
    two writes must actually guarantee: the envelope survives both (obs_names
    export intact) *and* the categorical is still a categorical afterwards.
    """
    path, expected = categorical_obs_scx

    table = tmp_path / "scrub.csv"
    pd.DataFrame({
        "barcode": expected,
        "doublet_score": np.linspace(0, 1, len(expected)),
        "predicted_doublet": [True, False] * (len(expected) // 2),
    }).to_csv(table, index=False)
    pyscx.doublet_import(path, str(table), tool="scrublet")
    pyscx.doublet_consensus(path, keys=["scrublet"], method="any")

    assert _exported_obs_names(path, tmp_path, "out.h5ad") == expected
    obs = pyscx.open(path).read_obs()
    assert isinstance(obs["cell_type"].dtype, pd.CategoricalDtype), obs["cell_type"].dtype
    assert list(obs["cell_type"].cat.categories) == ["B cell", "NK", "T cell"]
    assert "doublet_consensus" in obs.columns or any(c.startswith("doublet") for c in obs.columns)


def test_the_categorical_stays_a_column_not_the_index(categorical_obs_scx, tmp_path):
    """The other half of the same assertion, from the failure's own direction:
    `cell_type` was being *promoted* to obs_names, so check it is still an
    ordinary column carrying its own values."""
    anndata = pytest.importorskip("anndata")
    path, _ = categorical_obs_scx

    pyscx.modify_metadata(path, obs=pyscx.open(path).read_obs())
    out = tmp_path / "out.h5ad"
    pyscx.to_h5ad(path, str(out))

    obs = anndata.read_h5ad(out).obs
    assert "cell_type" in obs.columns
    assert list(obs["cell_type"].astype(str))[:4] == ["T cell"] * 4
    # And still the categorical it was written as: the in-place rewrite carries
    # the dictionary through rather than decoding it to plain strings.
    assert isinstance(obs["cell_type"].dtype, pd.CategoricalDtype), obs["cell_type"].dtype
    assert list(obs["cell_type"].cat.categories) == ["B cell", "NK", "T cell"]
    assert "__index_level_0__" not in obs.columns


def test_projected_read_obs_keeps_its_index(categorical_obs_scx):
    """The second consumer of the envelope. `read_obs(columns=[...])` consults
    it to retain the index column in the projection; without it the returned
    frame silently loses its barcodes."""
    path, expected = categorical_obs_scx

    pyscx.modify_metadata(path, obs=pyscx.open(path).read_obs())

    projected = pyscx.open(path).read_obs(["n_counts"])
    assert list(projected.index) == expected
    assert list(projected.columns) == ["n_counts"]


# ---------------------------------------------------------------------------
# update_uns — shallow merge
# ---------------------------------------------------------------------------


def test_update_uns_overwrites_only_the_named_keys(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    pyscx.set_uns(path, {"a": 1, "b": {"x": 1}, "keep": [1.5, "s"]})

    pyscx.update_uns(path, {"b": 2, "c": [3]})

    uns = pyscx.open(path).read_uns()
    assert uns["a"] == 1, "untouched key survives"
    assert uns["b"] == 2, "a patch key replaces the old value wholesale (no deep merge)"
    assert list(uns["c"]) == [3]
    assert list(uns["keep"]) == [1.5, "s"]
    assert set(uns) == {"a", "b", "c", "keep"}


def test_update_uns_rejects_a_non_dict_and_writes_nothing(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    pyscx.set_uns(path, {"a": 1})
    before = open(path, "rb").read()
    for bad in [(1, 2), [1, 2], np.array([1.0]), "s", 3]:
        with pytest.raises(ValueError, match="must be a dict"):
            pyscx.update_uns(path, bad)
    assert open(path, "rb").read() == before


def test_update_uns_then_rollback(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    pyscx.set_uns(path, {"state": "v0"})
    pyscx.update_uns(path, {"added": True})
    assert pyscx.open(path).read_uns() == {"state": "v0", "added": True}

    pyscx.rollback(path)
    assert pyscx.open(path).read_uns() == {"state": "v0"}


def test_update_uns_reloads_an_open_experiment_handle(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    exp = pyscx.open(path)
    pyscx.update_uns(exp, {"via_handle": 1})
    assert exp.read_uns()["via_handle"] == 1
