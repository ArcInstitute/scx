"""Zero-row and zero-var files write and read (REC-5, PR E).

Before this, `pyscx.from_anndata` on an AnnData with `n_obs == 0` (or
`n_vars == 0`) raised `RuntimeError: Arrow IPC contains no batches`: pyarrow's
`Table.from_pandas` → `write_table` emits zero IPC batches for a 0-row frame,
and `pandas_to_record_batch` took the first batch. The format, the Rust writer
and every reader already handled zero CSR shards (`scx subset` with a predicate
matching nothing has always written a valid 0-row file); only the pyscx
boundary refused. These tests pin the contract `docs/api.md` § `pyscx.from_anndata`
now states:

- a 0-row file keeps `X`'s column count, the full obs / var schema (declared
  categories and `ordered` included), `obsm` / `varm` / `uns`;
- `layers` and `raw` exist on disk only as CSR shards, and a 0-row file has
  none, so they are dropped with a `UserWarning`;
- an empty `object` column, and the index of an empty frame, are stored as
  string (pyarrow types them `null`, which no other writer produces);
- no CSC sidecar and no predicate index are built on an empty matrix, whatever
  `csc=` / `index_*` say, and nothing warns about it;
- `merge` tolerates 0-row inputs, `append` of a 0-row input is a no-op,
  `build_csc` on an empty matrix is a no-op.
"""

import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import pyscx

N_VARS = 5


def _var(n_vars=N_VARS):
    return pd.DataFrame(
        {"gene_ids": [f"ENSG{i:05d}" for i in range(n_vars)], "hv": [i % 2 == 0 for i in range(n_vars)]},
        index=[f"g{i}" for i in range(n_vars)],
    )


def _obs_columns(n):
    """The obs column set every fixture shares; `n` rows. Every column is
    built with an explicit dtype: at `n == 0` a bare `[]` becomes float64, and
    a `pd.Series` would align on the frame's string index and turn to NaN."""
    return {
        "ct": pd.Categorical((["A", "B"] * ((n + 1) // 2))[:n], categories=["A", "B"], ordered=True),
        "score": np.arange(n, dtype=np.float64),
        "flag": np.array([True, False] * ((n + 1) // 2), dtype=bool)[:n],
        "name": np.array([f"n{i}" for i in range(n)], dtype=object),
        "n": np.arange(n, dtype=np.int64),
    }


def _zero_obs_adata(n_vars=N_VARS, *, dense=False, with_layers_and_raw=True, with_obsm=True):
    import anndata

    obs = pd.DataFrame(_obs_columns(0), index=pd.Index([], dtype=object))
    x = np.zeros((0, n_vars), dtype=np.float32) if dense else sp.csr_matrix((0, n_vars), dtype=np.float32)
    a = anndata.AnnData(X=x, obs=obs, var=_var(n_vars))
    if with_obsm:
        a.obsm["X_pca"] = np.zeros((0, 3), dtype=np.float32)
    a.varm["loadings"] = np.arange(n_vars * 2, dtype=np.float32).reshape(n_vars, 2)
    a.uns["k"] = "v"
    if with_layers_and_raw:
        a.layers["counts"] = sp.csr_matrix((0, n_vars), dtype=np.float32)
        a.raw = a.copy()
    return a


def _zero_var_adata(n_obs=10):
    import anndata

    obs = pd.DataFrame(_obs_columns(n_obs), index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame({"gene_ids": pd.Series([], dtype=object)}, index=pd.Index([], dtype=object))
    return anndata.AnnData(X=sp.csr_matrix((n_obs, 0), dtype=np.float32), obs=obs, var=var)


def _full_adata(n_obs=8, n_vars=N_VARS, *, seed=0):
    import anndata

    rng = np.random.default_rng(seed)
    x = sp.random(n_obs, n_vars, density=0.6, format="csr", random_state=rng, dtype=np.float32)
    x.data = np.ceil(x.data * 5).astype(np.float32)
    a = anndata.AnnData(X=x, obs=pd.DataFrame(_obs_columns(n_obs), index=[f"c{i}" for i in range(n_obs)]), var=_var(n_vars))
    a.layers["counts"] = x.copy()
    a.obsm["X_pca"] = rng.random((n_obs, 3)).astype(np.float32)
    a.uns["k"] = "v"
    return a


def _write_quiet(adata, path, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pyscx.from_anndata(adata, str(path), **kw)
    return str(path)


def _scx_binary():
    env = os.environ.get("SCX_BIN")
    if env and Path(env).is_file():
        return env
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (repo_root / "target" / "release" / "scx", repo_root / "target" / "debug" / "scx"):
        if candidate.is_file():
            return str(candidate)
    return None


def _sections(path):
    """Section names of a file, via `scx info`; None when the CLI is absent."""
    scx = _scx_binary()
    if scx is None:
        return None
    out = subprocess.run([scx, "info", path], capture_output=True, text=True, check=True).stdout
    names = []
    in_sections = False
    for line in out.splitlines():
        if line.strip().startswith("Sections:"):
            in_sections = True
            continue
        if in_sections:
            if not line.strip():
                break
            if line.startswith("  "):
                names.append(line.split()[0])
    return names


def _assert_zero_obs_schema(obs):
    assert obs.shape == (0, 5)
    assert list(obs.columns) == ["ct", "score", "flag", "name", "n"]
    assert isinstance(obs["ct"].dtype, pd.CategoricalDtype)
    assert list(obs["ct"].cat.categories) == ["A", "B"], "declared unused categories must survive 0 rows"
    assert obs["ct"].cat.ordered is True
    assert obs["score"].dtype == np.float64
    assert obs["flag"].dtype in (np.dtype(bool), pd.BooleanDtype())
    assert obs["name"].dtype == object
    assert obs["n"].dtype == np.int64
    assert len(obs.index) == 0


# ---------------------------------------------------------------------------
# 0 obs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dense", [False, True], ids=["csr", "dense"])
def test_zero_obs_round_trip(tmp_dir, dense):
    a = _zero_obs_adata(dense=dense, with_layers_and_raw=False)
    path = _write_quiet(a, tmp_dir / "zero_obs.scx")

    exp = pyscx.open(path)
    assert (exp.n_obs, exp.n_vars, exp.shard_count, exp.nnz) == (0, N_VARS, 0, 0)

    back = exp.to_anndata()
    assert back.shape == (0, N_VARS)
    assert sp.issparse(back.X) and back.X.shape == (0, N_VARS) and back.X.nnz == 0
    _assert_zero_obs_schema(back.obs)
    pd.testing.assert_frame_equal(back.var, a.var, check_dtype=False)
    assert back.obsm["X_pca"].shape == (0, 3)
    np.testing.assert_array_equal(back.varm["loadings"], a.varm["loadings"])
    assert back.uns["k"] == "v"
    assert len(back.layers) == 0
    assert back.raw is None


def test_zero_obs_layers_and_raw_are_dropped_with_a_warning(tmp_dir):
    a = _zero_obs_adata()
    with pytest.warns(UserWarning) as rec:
        pyscx.from_anndata(a, str(tmp_dir / "z.scx"))
    msgs = [str(w.message) for w in rec]
    assert any("counts" in m and "layer" in m for m in msgs), msgs
    assert any("raw" in m for m in msgs), msgs
    back = pyscx.open(str(tmp_dir / "z.scx")).to_anndata()
    assert len(back.layers) == 0 and back.raw is None
    assert back.shape == (0, N_VARS)


def test_zero_obs_from_subsets(tmp_dir):
    """The spec's arm: every way of arriving at an empty AnnData writes."""
    full = _full_adata()
    src = _write_quiet(full, tmp_dir / "full.scx")

    # anndata's own boolean subset: categories pruned to none, object → object.
    empty = full[np.zeros(full.n_obs, dtype=bool)].copy()
    out = _write_quiet(empty, tmp_dir / "sub_anndata.scx")
    back = pyscx.open(out).to_anndata()
    assert back.shape == (0, N_VARS)
    assert isinstance(back.obs["ct"].dtype, pd.CategoricalDtype)
    assert list(back.obs["ct"].cat.categories) == []
    assert back.obs["name"].dtype == object

    # pyscx.accel.subset_obs on an in-memory AnnData (scipy X).
    inmem = pyscx.open(src).to_anndata()
    pyscx.accel.subset_obs(inmem, np.zeros(inmem.n_obs, dtype=bool))
    assert inmem.shape == (0, N_VARS)
    out = _write_quiet(inmem, tmp_dir / "sub_inmem.scx")
    assert pyscx.open(out).to_anndata().shape == (0, N_VARS)

    # pyscx.accel.subset_obs on a backed AnnData: X stays a handle with 0
    # visible rows and routes through the SCX → SCX rewrite.
    backed = pyscx.open(src).to_anndata(backed=True)
    pyscx.accel.subset_obs(backed, np.zeros(backed.n_obs, dtype=bool))
    assert isinstance(backed.X, pyscx.ScxBackedSparseDataset) and backed.shape == (0, N_VARS)
    out = _write_quiet(backed, tmp_dir / "sub_backed.scx")
    e = pyscx.open(out)
    assert (e.n_obs, e.n_vars, e.shard_count) == (0, N_VARS, 0)
    assert e.to_anndata().shape == (0, N_VARS)
    assert e.read_obs().shape[0] == 0 and list(e.read_obs().columns) == ["ct", "score", "flag", "name", "n"]
    assert not e.has_csc


def test_empty_object_columns_are_stored_as_string(tmp_dir):
    """pyarrow types an empty `object` column (and the index) as `null`; SCX
    stores them as string so the schema matches the non-empty sibling."""
    empty = _write_quiet(_zero_obs_adata(with_layers_and_raw=False), tmp_dir / "empty.scx")
    obs = pyscx.open(empty).read_obs()
    assert obs["name"].dtype == object and len(obs) == 0
    assert obs.index.dtype == object

    # The consequence: rows with real strings append onto the empty file …
    full = _full_adata()
    full_path = _write_quiet(full, tmp_dir / "full.scx")
    target = str(tmp_dir / "target.scx")
    shutil.copy(empty, target)
    pyscx.append(target, full_path)
    got = pyscx.open(target).read_obs()
    assert got["name"].tolist() == full.obs["name"].tolist()
    # … and the empty file merges with a populated one in either position.
    for order, name in (([empty, full_path], "m_ef.scx"), ([full_path, empty], "m_fe.scx")):
        out = str(tmp_dir / name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pyscx.merge(order, out)
        assert pyscx.open(out).read_obs()["name"].tolist() == full.obs["name"].tolist()


def test_zero_obs_backed_query_and_info(tmp_dir):
    path = _write_quiet(_zero_obs_adata(with_layers_and_raw=False), tmp_dir / "z.scx")
    exp = pyscx.open(path)

    backed = exp.to_anndata(backed=True)
    X = backed.X
    assert isinstance(X, pyscx.ScxBackedSparseDataset)
    assert X.shape == (0, N_VARS) and X.n_shards == 0
    m = X.to_memory()
    assert sp.issparse(m) and m.shape == (0, N_VARS) and m.nnz == 0
    np.testing.assert_array_equal(np.asarray(X.sum(axis=0)).ravel(), np.zeros(N_VARS))
    assert X[:, [1, 3]].shape == (0, 2)
    assert X.stored_dtype == np.dtype("float32")  # documented: no shards → the decode dtype

    info = exp.info()
    for token in ("csr_shards=0", "nnz=0", "value_encoding=n/a", "is_integer=false"):
        assert token in info, info
    assert exp.value_encoding == "n/a" and exp.is_integer is False and exp.max_value == 0

    _assert_zero_obs_schema(exp.read_obs())
    assert exp.read_var().shape == (N_VARS, 2)

    q = exp.query().collect()
    assert q.n_obs == 0 and q.n_vars == N_VARS
    q2 = exp.query().select_genes(["g1", "g2"]).collect()
    assert q2.n_obs == 0 and q2.n_vars == 2
    q3 = exp.query().filter_obs("score > 1").collect()
    assert q3.n_obs == 0


def test_zero_obs_csc_and_predicate_index_are_skipped_silently(tmp_dir):
    a = _zero_obs_adata(with_layers_and_raw=False)
    path = str(tmp_dir / "z.scx")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        pyscx.from_anndata(a, path, csc="always", index_obs=["ct", "score"], index_var=["gene_ids"])
    assert rec == [], [str(w.message) for w in rec]
    exp = pyscx.open(path)
    assert exp.has_csc is False
    sections = _sections(path)
    if sections is not None:
        # No obs index (0 rows) and no CSC sidecar; the *var* index is
        # legitimate — var has 5 rows — and is built as asked.
        assert not any(s.startswith("obs_predicate_index") or s.startswith("csc") for s in sections), sections
        assert "var_predicate_index" in sections, sections
    # A forced index column on an empty file is not an error either.
    assert exp.query().filter_obs("ct == 'A'").collect().n_obs == 0


# ---------------------------------------------------------------------------
# 0 vars
# ---------------------------------------------------------------------------


def test_zero_vars_round_trip(tmp_dir):
    a = _zero_var_adata()
    path = _write_quiet(a, tmp_dir / "zero_vars.scx")
    exp = pyscx.open(path)
    assert (exp.n_obs, exp.n_vars, exp.nnz) == (10, 0, 0)
    assert exp.shard_count >= 1

    back = exp.to_anndata()
    assert back.shape == (10, 0)
    assert sp.issparse(back.X) and back.X.shape == (10, 0)
    assert back.obs["ct"].tolist() == a.obs["ct"].tolist()
    assert list(back.obs.index) == list(a.obs.index)
    var = exp.read_var()
    assert var.shape == (0, 1) and list(var.columns) == ["gene_ids"] and var["gene_ids"].dtype == object

    X = exp.to_anndata(backed=True).X
    assert X.shape == (10, 0)
    assert X.to_memory().shape == (10, 0)
    assert "n_obs" in repr(exp) or True  # repr must not raise
    exp.info()


# ---------------------------------------------------------------------------
# h5ad export
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not pyscx._HAS_HDF5, reason="to_h5ad needs the hdf5 feature")
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "eager"])
def test_zero_obs_and_zero_vars_to_h5ad(tmp_dir, stream):
    import anndata

    zo = _write_quiet(_zero_obs_adata(with_layers_and_raw=False), tmp_dir / "zo.scx")
    out = str(tmp_dir / f"zo_{stream}.h5ad")
    pyscx.to_h5ad(zo, out, stream=stream)
    h = anndata.read_h5ad(out)
    assert h.shape == (0, N_VARS)
    assert list(h.obs["ct"].cat.categories) == ["A", "B"]
    assert list(h.var.index) == [f"g{i}" for i in range(N_VARS)]

    zv = _write_quiet(_zero_var_adata(), tmp_dir / "zv.scx")
    out = str(tmp_dir / f"zv_{stream}.h5ad")
    pyscx.to_h5ad(zv, out, stream=stream)
    h = anndata.read_h5ad(out)
    assert h.shape == (10, 0)
    assert list(h.obs.index) == [f"c{i}" for i in range(10)]


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------


def test_zero_obs_ops(tmp_dir):
    full = _full_adata()
    full_path = _write_quiet(full, tmp_dir / "full.scx")
    empty_path = _write_quiet(_zero_obs_adata(), tmp_dir / "empty.scx")  # layers/raw dropped (warned)

    # append rows onto an empty file. (The target carries no obsm: `append`
    # does not extend obsm on *any* target — a pre-existing gap, not a
    # zero-row one — so a target with obsm would not reopen as an AnnData.)
    target = _write_quiet(_zero_obs_adata(with_layers_and_raw=False, with_obsm=False), tmp_dir / "t1.scx")
    pyscx.append(target, full_path)
    got = pyscx.open(target).to_anndata()
    assert got.shape == full.shape
    assert (got.X != full.X).nnz == 0
    assert got.obs["score"].tolist() == full.obs["score"].tolist()

    # append an empty input: a no-op, bytes unchanged
    target2 = str(tmp_dir / "t2.scx")
    shutil.copy(full_path, target2)
    before = Path(target2).read_bytes()
    pyscx.append(target2, empty_path)
    assert Path(target2).read_bytes() == before

    # merge with an empty input that lacks the layer and obsm the other has
    for order, name in (([full_path, empty_path], "m1.scx"), ([empty_path, full_path], "m2.scx")):
        out = str(tmp_dir / name)
        pyscx.merge(order, out)
        e = pyscx.open(out)
        assert e.n_obs == full.n_obs
        assert e.layer_names() == ["counts"]
        m = e.to_anndata()
        assert (m.X != full.X).nnz == 0
        assert (m.layers["counts"] != full.layers["counts"]).nnz == 0
        assert m.obsm["X_pca"].shape == (full.n_obs, 3)
        assert m.obs["name"].tolist() == full.obs["name"].tolist()

    # merge of two empty inputs still writes an obs section (with the same
    # columns; merge flattens categoricals to plain strings on every output,
    # populated or not, so `ct` is compared by name and row count only)
    out = str(tmp_dir / "m3.scx")
    pyscx.merge([empty_path, empty_path], out)
    e = pyscx.open(out)
    assert e.n_obs == 0 and e.shard_count == 0
    obs = e.read_obs()
    assert obs.shape == (0, 5) and list(obs.columns) == ["ct", "score", "flag", "name", "n"]
    assert obs["score"].dtype == np.float64 and obs["n"].dtype == np.int64
    assert e.to_anndata().shape == (0, N_VARS)

    # delete every row, then compact
    dc = str(tmp_dir / "delall.scx")
    shutil.copy(full_path, dc)
    pyscx.mark_deleted(dc, list(range(full.n_obs)))
    compacted = str(tmp_dir / "compacted.scx")
    pyscx.compact(dc, compacted)
    e = pyscx.open(compacted)
    assert (e.n_obs, e.shard_count) == (0, 0)
    assert e.read_obs().shape[0] == 0
    assert e.to_anndata().shape == (0, N_VARS)

    # compact / build_csc on an empty file
    pyscx.compact(empty_path, str(tmp_dir / "c2.scx"))
    assert pyscx.open(str(tmp_dir / "c2.scx")).to_anndata().shape == (0, N_VARS)
    pyscx.build_csc(empty_path, str(tmp_dir / "csc.scx"))
    e = pyscx.open(str(tmp_dir / "csc.scx"))
    assert e.has_csc is False and e.to_anndata().shape == (0, N_VARS)

    # metadata writes with a 0-row frame
    obs0 = pyscx.open(empty_path).read_obs()
    obs0["extra"] = pd.Series([], dtype=np.float64, index=obs0.index)
    pyscx.modify_metadata(empty_path, obs=obs0)
    assert "extra" in pyscx.open(empty_path).read_obs().columns
    # `attach_obs_columns` keeps its own guard: a frame with no rows has
    # nothing to attach (the key join would match nothing on any target).
    with pytest.raises(ValueError, match="no rows"):
        pyscx.attach_obs_columns(
            empty_path,
            pd.DataFrame({"obs_names": pd.Series([], dtype=object), "extra2": pd.Series([], dtype=np.int64)}),
            key="obs_names",
        )
    assert pyscx.open(empty_path).read_obs().shape[0] == 0
    # A populated frame attached onto the empty target matches nothing, and
    # the join refuses a zero-match attach as it does on any target; the
    # file is left as it was.
    with pytest.raises(ValueError, match="no target row key matched"):
        pyscx.attach_obs_columns(
            empty_path,
            pd.DataFrame({"obs_names": ["c0", "c1"], "extra2": np.array([1, 2], dtype=np.int64)}),
            key="obs_names",
        )
    obs = pyscx.open(empty_path).read_obs()
    assert obs.shape[0] == 0 and "extra" in obs.columns and "extra2" not in obs.columns


def test_zero_obs_scx_cli(tmp_dir):
    scx = _scx_binary()
    if scx is None:
        pytest.skip("scx CLI binary not built")
    empty = _write_quiet(_zero_obs_adata(with_layers_and_raw=False), tmp_dir / "empty.scx")
    info = subprocess.run([scx, "info", empty], capture_output=True, text=True)
    assert info.returncode == 0, info.stderr
    assert "0 cells" in info.stdout
    csc = subprocess.run([scx, "build-csc", empty, str(tmp_dir / "csc.scx")], capture_output=True, text=True)
    assert csc.returncode == 0, csc.stderr
    assert pyscx.open(str(tmp_dir / "csc.scx")).has_csc is False
