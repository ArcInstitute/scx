"""Tests for SparseCellSetDataset (SCX-DATA-LOADER Phase 2.4).

The native sparse cell-set loader gathers role-tagged, multi-file cell-set
batches as sparse CSR (the §4.4 contract) and must match the backed
`to_anndata().X[rows]` reference, in plan order, across multiple files.
"""

import multiprocessing as mp

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def two_scx(synthetic_adata, tmp_dir):
    import pyscx

    p0 = str(tmp_dir / "f0.scx")
    p1 = str(tmp_dir / "f1.scx")
    pyscx.from_anndata(synthetic_adata, p0)
    pyscx.from_anndata(synthetic_adata, p1)
    return p0, p1


# Plan: one batch, two single-file sets (file 0 scattered, file 1 with a dup).
_FILE_IDS = [0, 0, 0, 1, 1, 1]
_ROWS = [0, 5, 2, 10, 10, 3]
_ROLE_TAGS = [0, 0, 0, 1, 1, 1]
_SET_OFFSETS = [0, 3, 6]
_PLAN = (_FILE_IDS, _ROWS, _ROLE_TAGS, _SET_OFFSETS)


def test_batch_dict_schema_and_dtypes(two_scx):
    import pyscx

    p0, p1 = two_scx
    ds = pyscx.SparseCellSetDataset([p0, p1])
    assert ds.n_files == 2
    batches = list(ds.iter_with_plans(iter([_PLAN])))
    assert len(batches) == 1
    b = batches[0]

    assert set(b) == {
        "indptr",
        "indices",
        "data",
        "shape",
        "cell_indices",
        "file_ids",
        "set_offsets",
        "role_tags",
    }
    assert b["indptr"].dtype == np.int64
    assert b["indices"].dtype == np.int32
    assert b["data"].dtype == np.float32
    assert b["cell_indices"].dtype == np.uint64
    assert b["file_ids"].dtype == np.uint32
    assert b["set_offsets"].dtype == np.int64
    assert b["role_tags"].dtype == np.int32

    assert tuple(b["shape"]) == (6, ds.n_cols)
    assert b["cell_indices"].tolist() == _ROWS
    assert b["file_ids"].tolist() == _FILE_IDS
    assert b["set_offsets"].tolist() == _SET_OFFSETS
    assert b["role_tags"].tolist() == _ROLE_TAGS


def test_csr_rows_match_backed_reference(two_scx):
    import pyscx

    p0, p1 = two_scx
    ds = pyscx.SparseCellSetDataset([p0, p1])
    b = next(iter(ds.iter_with_plans(iter([_PLAN]))))

    got = sp.csr_matrix(
        (b["data"], b["indices"], b["indptr"]), shape=tuple(b["shape"])
    )
    refs = {
        0: pyscx.open(p0).to_anndata(backed=True).X,
        1: pyscx.open(p1).to_anndata(backed=True).X,
    }
    for j, (fid, row) in enumerate(zip(_FILE_IDS, _ROWS)):
        expected = refs[fid][row].toarray()
        np.testing.assert_array_equal(got[j].toarray(), expected)


def test_remap_emits_global_indices(synthetic_adata, tmp_dir):
    import pyscx

    p0 = str(tmp_dir / "g0.scx")
    pyscx.from_anndata(synthetic_adata, p0)
    n_vars = synthetic_adata.n_vars
    # local g → global g + 1000 (injective, no sentinels).
    table = [g + 1000 for g in range(n_vars)]
    ds = pyscx.SparseCellSetDataset(
        [p0], remap_tables=[table], n_global_genes=n_vars + 1000
    )
    assert ds.n_cols == n_vars + 1000
    b = next(iter(ds.iter_with_plans(iter([([0], [7], [0], [0, 1])]))))
    ref = pyscx.open(p0).to_anndata(backed=True).X[7]
    expected_global = (ref.indices.astype(np.int64) + 1000)
    np.testing.assert_array_equal(np.sort(b["indices"]), np.sort(expected_global))


# --- malformed plans surface as clean exceptions, not a worker crash -------


def test_file_id_out_of_range_raises_runtimeerror(two_scx):
    import pyscx

    p0, p1 = two_scx
    ds = pyscx.SparseCellSetDataset([p0, p1])
    # file_id 9 but only 2 files.
    bad = ([0, 9], [0, 1], [0, 0], [0, 2])
    with pytest.raises(RuntimeError, match="file_id"):
        list(ds.iter_with_plans(iter([bad])))


def test_bad_set_offsets_raises_runtimeerror(two_scx):
    import pyscx

    p0, p1 = two_scx
    ds = pyscx.SparseCellSetDataset([p0, p1])
    # set_offsets[1] past total_rows would panic the slice without validation.
    bad = ([0, 0], [0, 1], [0, 0], [0, 99])
    with pytest.raises(RuntimeError, match="set_offsets"):
        list(ds.iter_with_plans(iter([bad])))


def test_row_out_of_range_raises_indexerror(two_scx):
    import pyscx

    p0, p1 = two_scx
    ds = pyscx.SparseCellSetDataset([p0, p1])
    n_obs = pyscx.open(p0).n_obs
    bad = ([0, 0], [0, n_obs + 100], [0, 0], [0, 2])
    with pytest.raises(IndexError):
        list(ds.iter_with_plans(iter([bad])))


# --- fork-safety -----------------------------------------------------------


def _child_build_and_iter(path, conn):
    try:
        import pyscx

        ds = pyscx.SparseCellSetDataset([path])  # built post-fork → OK
        b = next(iter(ds.iter_with_plans(iter([([0], [1], [0], [0, 1])]))))
        conn.send(("ok", int(b["cell_indices"][0])))
    except BaseException as exc:  # noqa: BLE001
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


def test_fork_lazy_post_fork_construction(two_scx):
    p0, _ = two_scx
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_child_build_and_iter, args=(p0, child_conn))
    proc.start()
    status, payload = parent_conn.recv()
    proc.join(timeout=30)
    assert status == "ok", f"child failed: {payload}"
    assert payload == 1


def test_fork_pre_fork_dataset_raises_in_child(two_scx):
    p0, _ = two_scx
    import pyscx

    ds = pyscx.SparseCellSetDataset([p0])  # built in parent

    def _child(conn):
        try:
            list(ds.iter_with_plans(iter([([0], [1], [0], [0, 1])])))
            conn.send(("ok", None))
        except BaseException as exc:  # noqa: BLE001
            conn.send(("err", repr(exc)))
        finally:
            conn.close()

    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_child, args=(child_conn,))
    proc.start()
    status, payload = parent_conn.recv()
    proc.join(timeout=30)
    assert status == "err" and "num_workers=0" in payload
