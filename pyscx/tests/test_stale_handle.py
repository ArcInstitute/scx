"""An open handle must not answer reads from a file that has changed underneath it.

The writer never mutates bytes a reader is looking at. An in-place op appends and
rewrites the 256-byte header; a copy-out op writes a temp file and renames it. Both
leave an already-open handle's `mmap` intact and perfectly readable — pointing at the
pre-mutation bytes for the first, and at an unlinked inode for the second. The handle
therefore keeps answering, with the wrong answer, and nothing marks the seam.

The two halves fail differently and a detector can easily catch only one:

  * in-place  — same inode, `manifest_sequence` bumped, file grew.
  * copy-out  — new inode, and `manifest_sequence` need not have moved at all
                (compacting a `manifest_sequence == 1` file yields another one).
  * rollback  — same inode, and it rewrites *only* the header, so the file size
                can be identical before and after.

So every arm below is load-bearing: size alone misses rollback, sequence alone
misses compact, and inode alone misses everything in-place.

`test_touch_is_not_a_change` is the counterweight — it pins that the check is about
the file's *contents*, not its mtime, and it is the assertion that goes red if the
header-confirm step is deleted as redundant.
"""

import os
import pathlib

import numpy as np
import pytest

import pyscx

anndata = pytest.importorskip("anndata")
sparse = pytest.importorskip("scipy.sparse")
pd = pytest.importorskip("pandas")


STALE = r"changed on disk|replaced on disk"


def _fixture(tmp_path, name="t.scx", n=4, shard_size=None):
    """A small SCX file with four uniquely-barcoded cells."""
    bc = ["AAAC-1", "AAAG-1", "AAAT-1", "AAAA-1"][:n]
    obs = pd.DataFrame({"grp": ["a", "b", "a", "b"][:n]}, index=bc)
    X = sparse.csr_matrix(np.arange(n * 3, dtype=np.float32).reshape(n, 3))
    var = pd.DataFrame(index=[f"g{i}" for i in range(3)])
    path = tmp_path / name
    kwargs = {} if shard_size is None else {"shard_size": shard_size}
    pyscx.from_anndata(anndata.AnnData(X=X, obs=obs, var=var), str(path), **kwargs)
    return path


def _calls_csv(tmp_path, name="calls.csv"):
    p = tmp_path / name
    p.write_text("barcode,score\nAAAT-1,0.30\nAAAC-1,0.10\nAAAG-1,0.90\n")
    return p


# ---------------------------------------------------------------------------
# In-place mutation — same inode, bumped manifest_sequence
# ---------------------------------------------------------------------------


def test_obs_import_leaves_an_open_experiment_stale(tmp_path):
    """The review's repro: the handle answered with the pre-import columns."""
    scx = _fixture(tmp_path)
    exp = pyscx.open(str(scx))
    assert "score" not in exp.read_obs().columns

    pyscx.obs_import(str(scx), str(_calls_csv(tmp_path)), key="obs_names", source_key="barcode")

    # Freshly opened, the column is there — so the file really did change.
    assert "score" in pyscx.open(str(scx)).read_obs().columns

    with pytest.raises(RuntimeError, match=STALE):
        exp.read_obs()


def test_a_scalar_read_is_stale_too(tmp_path):
    """`n_obs` is answered from the parsed header and never touches a section.

    A guard that only covers section reads leaves this one silently wrong, which
    is the whole point of `append`: the row count is exactly what moved.
    """
    scx = _fixture(tmp_path)
    other = _fixture(tmp_path, name="other.scx")
    exp = pyscx.open(str(scx))
    assert exp.n_obs == 4

    pyscx.append(str(scx), str(other))
    assert pyscx.open(str(scx)).n_obs == 8

    with pytest.raises(RuntimeError, match=STALE):
        exp.n_obs


# ---------------------------------------------------------------------------
# Copy-out mutation — new inode, manifest_sequence need not move
# ---------------------------------------------------------------------------


def test_a_warm_stored_dtype_is_stale_too(tmp_path):
    """`stored_dtype` folds every shard header once and memoises. The memo hit
    touches no section, so without its own freshness check a handle that had
    answered `uint8` kept answering it after an `append` mixed in wider shards
    — while `shape` on the same handle refused. Every sparse wrapper."""
    # A layered copy of the fixture, so the layer wrapper is covered too.
    X = sparse.csr_matrix(np.arange(4 * 3, dtype=np.float32).reshape(4, 3))
    layered = anndata.AnnData(
        X=X,
        obs=pd.DataFrame({"grp": ["a", "b", "a", "b"]}, index=["AAAC-1", "AAAG-1", "AAAT-1", "AAAA-1"]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        layers={"raw": X.copy()},
    )
    scx = tmp_path / "layered.scx"
    pyscx.from_anndata(layered, str(scx))
    other = tmp_path / "other.scx"
    pyscx.from_anndata(layered, str(other))
    adata = pyscx.open(str(scx)).to_anndata(backed=True)
    pyscx.accel.normalize_total(adata)
    lazy = adata.X
    fresh = pyscx.open(str(scx)).to_anndata(backed=True)
    handles = [fresh.X, fresh.X[:, [1, 0]], fresh.layers["raw"], lazy]
    for h in handles:
        assert h.stored_dtype == np.dtype("uint8")  # warm the memo

    pyscx.append(str(scx), str(other))

    for h in handles:
        with pytest.raises(RuntimeError, match=STALE):
            h.stored_dtype
        # The setting is a handle property, not a file answer: still readable.
        assert isinstance(h.cache_shards, int)


def test_compact_in_place_replaces_the_inode_under_an_open_handle(tmp_path):
    """`compact` writes a temp file and renames it over the target.

    The handle's mmap keeps the *unlinked* original alive, so every read still
    succeeds and every read is of a file that no longer exists at that path.
    A `manifest_sequence` comparison alone does not see this.
    """
    scx = _fixture(tmp_path)
    exp = pyscx.open(str(scx))
    before = os.stat(scx).st_ino

    pyscx.compact(str(scx), str(scx))
    assert os.stat(scx).st_ino != before, "compact should have renamed a new file into place"

    with pytest.raises(RuntimeError, match=STALE):
        exp.read_obs()


def test_a_deleted_file_is_not_readable_through_an_open_handle(tmp_path):
    scx = _fixture(tmp_path)
    exp = pyscx.open(str(scx))
    os.unlink(scx)

    with pytest.raises(RuntimeError):
        exp.read_obs()


# ---------------------------------------------------------------------------
# Rollback — same inode, and the file size need not change at all
# ---------------------------------------------------------------------------


def test_rollback_is_a_change_even_when_the_size_does_not_move(tmp_path):
    """`rollback` rewinds the header's catalog pointer and writes nothing else.

    Size and inode can both be unchanged across it, so only the header's own
    `manifest_sequence` / catalog offsets distinguish the two states.
    """
    scx = _fixture(tmp_path)
    other = _fixture(tmp_path, name="other.scx")
    pyscx.append(str(scx), str(other))

    exp = pyscx.open(str(scx))
    assert exp.n_obs == 8
    size_before = os.path.getsize(scx)

    pyscx.rollback(str(scx))
    assert pyscx.open(str(scx)).n_obs == 4
    assert os.path.getsize(scx) == size_before, "rollback should not have resized the file"

    with pytest.raises(RuntimeError, match=STALE):
        exp.read_obs()


# ---------------------------------------------------------------------------
# The handles an Experiment hands out — it is dropped before they are used
# ---------------------------------------------------------------------------


def test_a_backed_adata_is_stale_after_a_mutation(tmp_path):
    """`adata = pyscx.open(p).to_anndata(backed=True)` drops the Experiment on
    the same line, so guarding only `Experiment` would protect nothing here."""
    scx = _fixture(tmp_path)
    adata = pyscx.open(str(scx)).to_anndata(backed=True)
    assert adata.X[0:2].shape == (2, 3)

    pyscx.obs_import(str(scx), str(_calls_csv(tmp_path)), key="obs_names", source_key="barcode")

    with pytest.raises(RuntimeError, match=STALE):
        adata.X[0:2]


def test_a_query_pipeline_is_stale_after_a_mutation(tmp_path):
    scx = _fixture(tmp_path)
    q = pyscx.open(str(scx)).query()

    pyscx.obs_import(str(scx), str(_calls_csv(tmp_path)), key="obs_names", source_key="barcode")

    with pytest.raises(RuntimeError, match=STALE):
        q.collect()


# ---------------------------------------------------------------------------
# The counterweight: a changed mtime is not a changed file
# ---------------------------------------------------------------------------


def test_touch_is_not_a_change(tmp_path):
    """Only the bytes matter.

    `stat` is the cheap gate, but a bare `utime` moves mtime without moving a
    single byte. If the check stopped at the gate this would raise, and every
    backup tool, `touch`, or metadata-preserving copy would break open handles
    for no reason.
    """
    scx = _fixture(tmp_path)
    exp = pyscx.open(str(scx))
    before = exp.read_obs()

    st = os.stat(scx)
    os.utime(scx, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    assert exp.read_obs().equals(before)
    assert exp.n_obs == 4


def test_a_pathlib_target_is_handled(tmp_path):
    scx = _fixture(tmp_path)
    exp = pyscx.open(pathlib.Path(scx))
    assert exp.n_obs == 4


# ---------------------------------------------------------------------------
# Recovering: reload()
# ---------------------------------------------------------------------------


def test_reload_picks_up_the_new_contents(tmp_path):
    scx = _fixture(tmp_path)
    exp = pyscx.open(str(scx))
    assert "score" not in exp.read_obs().columns

    pyscx.obs_import(str(scx), str(_calls_csv(tmp_path)), key="obs_names", source_key="barcode")
    with pytest.raises(RuntimeError, match=STALE):
        exp.read_obs()

    exp.reload()
    assert "score" in exp.read_obs().columns


def test_reload_on_a_handle_that_is_not_stale_is_harmless(tmp_path):
    scx = _fixture(tmp_path)
    exp = pyscx.open(str(scx))
    exp.reload()
    assert exp.n_obs == 4


def test_reload_does_not_revive_what_the_handle_handed_out(tmp_path):
    """A backed `AnnData` owns its own reader.

    Reloading the `Experiment` cannot reach it — pretending otherwise would be
    worse than the error, because the arrays it already returned came from the
    old file and would silently sit alongside new ones.
    """
    scx = _fixture(tmp_path)
    exp = pyscx.open(str(scx))
    adata = exp.to_anndata(backed=True)

    pyscx.obs_import(str(scx), str(_calls_csv(tmp_path)), key="obs_names", source_key="barcode")
    exp.reload()

    assert "score" in exp.read_obs().columns
    with pytest.raises(RuntimeError, match=STALE):
        adata.X[0:2]
    # Re-derive it from the reloaded handle instead.
    assert exp.to_anndata(backed=True).X[0:2].shape == (2, 3)


def test_mark_deleted_through_the_handle_does_not_strand_it(tmp_path):
    """The one mutation an `Experiment` performs on itself.

    It re-opens rather than leaving the caller holding a handle it just
    invalidated — the behaviour that was already there for deletions, now
    sharing one implementation with `reload()`.
    """
    scx = _fixture(tmp_path)
    exp = pyscx.open(str(scx))
    exp.mark_deleted(np.array([True, False, False, False]))
    # `n_obs` is the live count; `read_obs` is the physical table, which a
    # logical deletion does not rewrite. Both must simply still *work*.
    assert exp.n_obs == 3
    assert exp.n_obs_physical == 4
    assert len(exp.read_obs()) == 4


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda p, exp, csv: pyscx.obs_import(exp, csv, key="obs_names",
                                                          source_key="barcode"),
                     id="obs_import"),
        pytest.param(lambda p, exp, csv: pyscx.set_uns(exp, {"k": 1}), id="set_uns"),
        pytest.param(lambda p, exp, csv: pyscx.modify_metadata(exp, uns={"k": 2}),
                     id="modify_metadata"),
        pytest.param(lambda p, exp, csv: pyscx.doublet_import(exp, csv, tool="generic",
                                                              key="obs_names",
                                                              source_key="barcode",
                                                              score_column="score"),
                     id="doublet_import"),
        pytest.param(lambda p, exp, csv: pyscx.attach_obs_columns(
                         exp, pd.DataFrame({"s": [1.0, 2.0, 3.0, 4.0]}),
                         positional=True),
                     id="attach_obs_columns"),
    ],
)
def test_mutating_through_a_handle_leaves_it_usable(tmp_path, call):
    """`pyscx.obs_import(exp, ...)` is a documented spelling.

    The wrapper reduces the handle to its path, so without an explicit reload
    the blessed call would be the one that breaks the caller's own handle.
    """
    scx = _fixture(tmp_path)
    csv = str(_calls_csv(tmp_path))
    exp = pyscx.open(str(scx))
    call(str(scx), exp, csv)
    assert exp.n_obs == 4


# ---------------------------------------------------------------------------
# Lifecycle: close() and `with`
# ---------------------------------------------------------------------------


def test_close_is_idempotent_and_reads_after_it_say_so(tmp_path):
    scx = _fixture(tmp_path)
    exp = pyscx.open(str(scx))
    assert exp.closed is False
    exp.close()
    exp.close()
    assert exp.closed is True

    with pytest.raises(RuntimeError, match="is closed"):
        exp.read_obs()
    with pytest.raises(RuntimeError, match="is closed"):
        exp.n_obs


def test_a_closed_handle_says_closed_not_stale(tmp_path):
    """The two conditions are different and the message must not conflate them:
    `reload()` fixes one and cannot fix the other."""
    scx = _fixture(tmp_path)
    exp = pyscx.open(str(scx))
    exp.close()
    pyscx.obs_import(str(scx), str(_calls_csv(tmp_path)), key="obs_names", source_key="barcode")

    with pytest.raises(RuntimeError, match="is closed"):
        exp.read_obs()


def test_the_context_manager_closes_on_exit(tmp_path):
    scx = _fixture(tmp_path)
    with pyscx.open(str(scx)) as exp:
        assert exp.n_obs == 4
    assert exp.closed is True


def test_the_context_manager_does_not_swallow_exceptions(tmp_path):
    scx = _fixture(tmp_path)
    with pytest.raises(ZeroDivisionError):
        with pyscx.open(str(scx)) as exp:
            1 / 0
    assert exp.closed is True


def test_a_with_block_releases_the_mapping_for_a_rewrite(tmp_path):
    scx = _fixture(tmp_path)
    with pyscx.open(str(scx)) as exp:
        n = exp.n_obs
    pyscx.compact(str(scx), str(scx))
    assert pyscx.open(str(scx)).n_obs == n


def test_closing_a_handle_does_not_close_what_it_handed_out(tmp_path):
    """Independent readers. A `with` block around the open is a common shape,
    and it must not quietly kill the backed `AnnData` taken out of it."""
    scx = _fixture(tmp_path)
    with pyscx.open(str(scx)) as exp:
        adata = exp.to_anndata(backed=True)
    assert adata.X[0:2].shape == (2, 3)


# ---------------------------------------------------------------------------
# The three members that must never raise
# ---------------------------------------------------------------------------


def test_path_and_repr_survive_a_stale_handle(tmp_path):
    """A repr that throws replaces the state you were trying to inspect with a
    second, unrelated traceback — and `path` is how you find out which file."""
    scx = _fixture(tmp_path)
    exp = pyscx.open(str(scx))
    pyscx.obs_import(str(scx), str(_calls_csv(tmp_path)), key="obs_names", source_key="barcode")

    assert exp.path == str(scx)
    r = repr(exp)
    assert "stale" in r and str(scx) in r


def test_repr_of_a_closed_handle_says_closed(tmp_path):
    scx = _fixture(tmp_path)
    exp = pyscx.open(str(scx))
    exp.close()
    assert "closed" in repr(exp)


# ---------------------------------------------------------------------------
# Coverage, enforced rather than remembered
# ---------------------------------------------------------------------------

# Every public member of `Experiment`, and how to call it. The point of the
# table is not the calls — it is that a member missing from it fails the test
# below, so adding a read method to `Experiment` without deciding whether it
# should see a changed file is not something that can happen quietly.
#
# `raises` members must refuse on a changed file. `exempt` ones must not, and
# each has a reason: `path` is how you identify the file, `close` / `reload`
# are the ways out, `closed` reports state, and `__repr__` must render in a
# traceback. `mark_deleted` mutates and re-opens, so it is tested separately.
_EXPERIMENT_CALLS = {
    # scalars answered from the header or catalog — no section read at all
    "n_obs": lambda e: e.n_obs,
    "n_obs_physical": lambda e: e.n_obs_physical,
    "n_vars": lambda e: e.n_vars,
    "shape": lambda e: e.shape,
    "nnz": lambda e: e.nnz,
    "shard_count": lambda e: e.shard_count,
    "obs_metadata_shard_count": lambda e: e.obs_metadata_shard_count,
    "var_metadata_shard_count": lambda e: e.var_metadata_shard_count,
    "format_version": lambda e: e.format_version,
    "codec_id": lambda e: e.codec_id,
    "index_dtype": lambda e: e.index_dtype,
    "has_csc": lambda e: e.has_csc,
    "has_deletions": lambda e: e.has_deletions,
    # shard-header / catalog folds — still through `reader()`, so they refuse too
    "value_encoding": lambda e: e.value_encoding,
    "is_integer": lambda e: e.is_integer,
    "max_value": lambda e: e.max_value,
    "is_multimodal": lambda e: e.is_multimodal,
    "n_modalities": lambda e: e.n_modalities,
    "modality_names": lambda e: e.modality_names,
    "modality_id": lambda e: e.modality_id("rna"),
    "modality_info": lambda e: e.modality_info(1),
    "layer_names": lambda e: e.layer_names(),
    "obsm_keys": lambda e: e.obsm_keys(),
    "varm_keys": lambda e: e.varm_keys(),
    "info": lambda e: e.info(),
    # section reads
    "read_obs": lambda e: e.read_obs(),
    "read_var": lambda e: e.read_var(),
    "read_uns": lambda e: e.read_uns(),
    "uns_keys": lambda e: e.uns_keys(),
    "obs_keys": lambda e: e.obs_keys(),
    "var_keys": lambda e: e.var_keys(),
    "distinct_values": lambda e: e.distinct_values("grp"),
    "obs_categorical": lambda e: e.obs_categorical("grp"),
    "obs_categorical_many": lambda e: e.obs_categorical_many(["grp"]),
    "provenance": lambda e: e.provenance(),
    "validate": lambda e: e.validate(),
    "to_anndata": lambda e: e.to_anndata(),
    "to_mudata": lambda e: e.to_mudata(),
    "to_gpu_anndata": lambda e: e.to_gpu_anndata(),
    "detection_counts": lambda e: e.detection_counts(),
    "cells_expressing": lambda e: e.cells_expressing("g0"),
    "gather_rows_sparse": lambda e: e.gather_rows_sparse(np.array([0], dtype=np.uint64)),
    "query": lambda e: e.query(),
    "read_group": lambda e: e.read_group("a"),
    "read_reference": lambda e: e.read_reference(),
    "group_labels": lambda e: e.group_labels(),
    "iter_group_shards": lambda e: e.iter_group_shards(),
}

_EXPERIMENT_EXEMPT = {
    "path": "names the file; needed to diagnose the very error being raised",
    "close": "the way to release a handle, stale or not",
    "closed": "reports handle state",
    "reload": "the way out of a stale handle",
    "mark_deleted": "mutates through the handle and re-opens; covered separately",
}


def _public_members():
    return {n for n in dir(pyscx.Experiment) if not n.startswith("_")}


def test_every_experiment_member_is_classified():
    """The table above must name every public member.

    This is the assertion that makes the next one self-maintaining: a read
    method added to `Experiment` lands here as a failure naming itself, rather
    than silently joining the set of things nobody checked.
    """
    unclassified = _public_members() - set(_EXPERIMENT_CALLS) - set(_EXPERIMENT_EXEMPT)
    assert not unclassified, (
        f"new public Experiment member(s) {sorted(unclassified)} are neither in "
        "_EXPERIMENT_CALLS (must refuse a changed file) nor in _EXPERIMENT_EXEMPT "
        "(must not). Decide which, and say why in the table."
    )
    stale_entries = (set(_EXPERIMENT_CALLS) | set(_EXPERIMENT_EXEMPT)) - _public_members()
    assert not stale_entries, f"table names members that no longer exist: {sorted(stale_entries)}"


def test_no_experiment_read_answers_from_a_changed_file(tmp_path):
    """Every classified read refuses, whatever it reads and however it reads it.

    Blanket rather than a handful of representative cases on purpose: the
    scalars come off the parsed header, the section reads come through the
    mmap, and the grouped / query paths hold their own readers — three
    different mechanisms that a single sampled test would not distinguish.
    """
    scx = _fixture(tmp_path)
    exp = pyscx.open(str(scx))
    pyscx.obs_import(str(scx), str(_calls_csv(tmp_path)), key="obs_names", source_key="barcode")

    answered = []
    for name, call in sorted(_EXPERIMENT_CALLS.items()):
        try:
            call(exp)
        except RuntimeError as e:
            if "changed on disk" in str(e) or "replaced on disk" in str(e):
                continue
            answered.append(f"{name}: RuntimeError but not about the file changing ({e})")
        except Exception as e:  # noqa: BLE001 — a wrong-shaped refusal is still a refusal
            answered.append(f"{name}: raised {type(e).__name__} instead ({e})")
        else:
            answered.append(f"{name}: answered")
    assert not answered, "members that did not refuse a changed file:\n  " + "\n  ".join(answered)


def test_the_exempt_members_still_work_on_a_stale_handle(tmp_path):
    scx = _fixture(tmp_path)
    exp = pyscx.open(str(scx))
    pyscx.obs_import(str(scx), str(_calls_csv(tmp_path)), key="obs_names", source_key="barcode")

    assert exp.path == str(scx)
    assert exp.closed is False
    assert isinstance(repr(exp), str)
    exp.reload()
    exp.close()


# ---------------------------------------------------------------------------
# The handles an Experiment hands out: cached scalars, not just data reads
# ---------------------------------------------------------------------------


def _mutate(scx, tmp_path):
    pyscx.obs_import(str(scx), str(_calls_csv(tmp_path)), key="obs_names", source_key="barcode")


@pytest.mark.parametrize(
    "read",
    [
        pytest.param(lambda x: x.shape, id="shape"),
        pytest.param(lambda x: len(x), id="len"),
        pytest.param(lambda x: x[0:2], id="getitem"),
    ],
)
def test_a_backed_x_refuses_a_changed_file(tmp_path, read):
    """`shape` and `len` are answered from a scalar cached at construction.

    They never reach `section_bytes`, so the chokepoint that covers every data
    read does not cover them — and a wrong `shape` after an `append` is exactly
    as silent as wrong data.
    """
    scx = _fixture(tmp_path)
    adata = pyscx.open(str(scx)).to_anndata(backed=True)
    read(adata.X)  # fine before

    _mutate(scx, tmp_path)
    with pytest.raises(RuntimeError, match=STALE):
        read(adata.X)


@pytest.mark.parametrize(
    "read",
    [
        pytest.param(lambda x: x.shape, id="shape"),
        pytest.param(lambda x: len(x), id="len"),
        pytest.param(lambda x: x[0:2], id="getitem"),
    ],
)
def test_a_backed_layer_refuses_a_changed_file(tmp_path, read):
    """A layer wraps the sparse dataset, and read its `shape_val` directly
    rather than going through the guarded getter — so the wrapper was a hole in
    a surface the thing it wraps had already closed."""
    bc = ["AAAC-1", "AAAG-1", "AAAT-1", "AAAA-1"]
    obs = pd.DataFrame({"grp": list("abab")}, index=bc)
    X = sparse.csr_matrix(np.arange(12, dtype=np.float32).reshape(4, 3))
    adata_src = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(3)]))
    adata_src.layers["counts"] = X.copy()
    scx = tmp_path / "layers.scx"
    pyscx.from_anndata(adata_src, str(scx))

    adata = pyscx.open(str(scx)).to_anndata(backed=True)
    layer = adata.layers["counts"]
    read(layer)

    _mutate(scx, tmp_path)
    with pytest.raises(RuntimeError, match=STALE):
        read(layer)


def test_a_backed_obsm_refuses_a_changed_file(tmp_path):
    bc = ["AAAC-1", "AAAG-1", "AAAT-1", "AAAA-1"]
    adata_src = anndata.AnnData(
        X=sparse.csr_matrix(np.arange(12, dtype=np.float32).reshape(4, 3)),
        obs=pd.DataFrame({"grp": list("abab")}, index=bc),
        var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
    )
    adata_src.obsm["X_pca"] = np.arange(8, dtype=np.float32).reshape(4, 2)
    scx = tmp_path / "obsm.scx"
    pyscx.from_anndata(adata_src, str(scx))

    # `obsm=` is what puts obsm on the backed path at all: without it a backed
    # read materialises each embedding as a plain numpy copy, which is a
    # snapshot by construction and rightly unaffected by a later write.
    adata = pyscx.open(str(scx)).to_anndata(backed=True, obsm=["X_pca"])
    pca = adata.obsm["X_pca"]
    assert pca.shape == (4, 2)

    _mutate(scx, tmp_path)
    with pytest.raises(RuntimeError, match=STALE):
        pca.shape
    with pytest.raises(RuntimeError, match=STALE):
        len(pca)


def test_an_eagerly_materialised_obsm_is_a_snapshot_not_a_handle(tmp_path):
    """The other half of the same fact, so neither is assumed.

    A default backed read copies each embedding into numpy. That copy is not a
    view on the file and must keep working after a mutation — refusing there
    would be a false positive, not a catch."""
    adata_src = anndata.AnnData(
        X=sparse.csr_matrix(np.arange(12, dtype=np.float32).reshape(4, 3)),
        obs=pd.DataFrame({"grp": list("abab")},
                         index=["AAAC-1", "AAAG-1", "AAAT-1", "AAAA-1"]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
    )
    adata_src.obsm["X_pca"] = np.arange(8, dtype=np.float32).reshape(4, 2)
    scx = tmp_path / "obsm_eager.scx"
    pyscx.from_anndata(adata_src, str(scx))

    pca = pyscx.open(str(scx)).to_anndata(backed=True).obsm["X_pca"]
    assert isinstance(pca, np.ndarray)

    _mutate(scx, tmp_path)
    assert pca.shape == (4, 2)
