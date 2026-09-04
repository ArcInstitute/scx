"""`stored_dtype` on the sparse handles and `Experiment.value_encoding` /
`is_integer` / `max_value` (REC-7, PR D).

The on-disk value encoding lives only in each shard's 76-byte header, so
before this the only way to learn "are these counts?" from Python was to
decode values and check integrality. These report it from the headers and
the catalog stats — no shard decode.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _adata(values, layers=None):
    import anndata

    n_obs, n_vars = values.shape
    a = anndata.AnnData(
        X=sp.csr_matrix(values),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(n_vars)]),
    )
    for k, v in (layers or {}).items():
        a.layers[k] = sp.csr_matrix(v)
    return a


def _counts(rng, n_obs, n_vars, high):
    dense = rng.integers(0, high, size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random((n_obs, n_vars)) > 0.4] = 0
    return dense


@pytest.fixture
def uint16_counts_scx(tmp_dir):
    """Counts whose max needs uint16, plus a float layer."""
    import pyscx

    rng = np.random.default_rng(7)
    counts = _counts(rng, 60, 20, 300)
    counts[0, 0] = 300.0  # pin the max
    norm = np.log1p(counts)
    path = str(tmp_dir / "u16.scx")
    pyscx.from_anndata(_adata(counts, {"norm": norm}), path)
    return path, counts


@pytest.fixture
def float_scx(tmp_dir):
    import pyscx

    rng = np.random.default_rng(8)
    dense = rng.random((40, 10)).astype(np.float32)
    dense[dense < 0.5] = 0
    path = str(tmp_dir / "float.scx")
    pyscx.from_anndata(_adata(dense), path)
    return path, dense


def test_stored_dtype_reports_the_on_disk_encoding(uint16_counts_scx):
    import pyscx

    path, counts = uint16_counts_scx
    adata = pyscx.open(path).to_anndata(backed=True)
    assert adata.X.stored_dtype == np.dtype("uint16")
    assert adata.X.stored_dtype.kind == "u"
    # `dtype` is unchanged: the decode type, documented as always float32.
    assert adata.X.dtype == np.dtype("float32")
    # A layer reports its own family, not X's.
    assert adata.layers["norm"].stored_dtype == np.dtype("float32")
    assert adata.layers["norm"].dtype == np.dtype("float32")
    # Projection / row subsets do not change what is stored.
    assert adata.X[:, [1, 3]].stored_dtype == np.dtype("uint16")


def test_lazy_handle_reports_the_source_encoding(uint16_counts_scx):
    import pyscx

    path, _ = uint16_counts_scx
    adata = pyscx.open(path).to_anndata(backed=True)
    pyscx.accel.normalize_total(adata)
    pyscx.accel.log1p(adata)
    assert isinstance(adata.X, pyscx.ScxLazyTransformedDataset)
    assert adata.X.stored_dtype == np.dtype("uint16")
    assert adata.X.dtype == np.dtype("float32")


def test_float_file_is_float32_and_not_integer(float_scx):
    import pyscx

    path, _ = float_scx
    adata = pyscx.open(path).to_anndata(backed=True)
    assert adata.X.stored_dtype == np.dtype("float32")
    exp = pyscx.open(path)
    assert exp.value_encoding == "float32"
    assert exp.is_integer is False
    # Float shards record no value range in the catalog stats.
    assert exp.max_value == 0
    assert "value_encoding=float32" in exp.info()
    assert "is_integer=false" in exp.info()


def test_experiment_reports_encoding_integrality_and_max(uint16_counts_scx):
    import pyscx

    path, counts = uint16_counts_scx
    exp = pyscx.open(path)
    assert exp.value_encoding == "uint16"
    assert exp.is_integer is True
    assert exp.max_value == int(counts.max()) == 300
    info = exp.info()
    for token in ("value_encoding=uint16", "is_integer=true", "max_value=300"):
        assert token in info, info
    # The pre-existing tokens are still there, in order, before the path.
    assert info.index("has_csc=") < info.index("value_encoding=") < info.index(", path=")


def test_mixed_shard_encodings_report_widest_and_mixed(tmp_dir):
    """`append` keeps each shard's own encoding, so a uint8 file with uint16
    rows appended is `mixed (uint8, uint16)` and the handle's stored dtype is
    the widest."""
    import pyscx

    rng = np.random.default_rng(9)
    small = _counts(rng, 30, 12, 100)
    big = _counts(rng, 30, 12, 400)
    big[0, 0] = 400.0
    target = str(tmp_dir / "mixed_target.scx")
    source = str(tmp_dir / "mixed_source.scx")
    pyscx.from_anndata(_adata(small), target)
    pyscx.from_anndata(_adata(big), source)
    assert pyscx.open(target).value_encoding == "uint8"
    pyscx.append(target, source)

    exp = pyscx.open(target)
    assert exp.value_encoding == "mixed (uint8, uint16)"
    assert exp.is_integer is True
    assert exp.max_value == 400
    assert "value_encoding=mixed (uint8, uint16)" in exp.info()
    adata = exp.to_anndata(backed=True)
    assert adata.X.stored_dtype == np.dtype("uint16")


def test_stored_dtype_is_declared_in_the_stub():
    """Properties are not stub-gated (each handle keeps `__getattr__`), so pin
    the three new names by hand: a typed consumer should see them."""
    import pathlib

    import pyscx

    text = pathlib.Path(pyscx.__file__).with_name("__init__.pyi").read_text()
    for cls in (
        "ScxBackedSparseDataset",
        "ScxBackedLayerDataset",
        "ScxLazyTransformedDataset",
    ):
        body = text[text.index(f"class {cls}:") :]
        body = body[: body.index("\nclass ")]
        assert "def stored_dtype(self)" in body, cls
        assert "def cache_shards(self)" in body, cls
    exp_body = text[text.index("class Experiment:") :]
    exp_body = exp_body[: exp_body.index("\nclass ")]
    for name in ("value_encoding", "is_integer", "max_value"):
        assert f"def {name}(self)" in exp_body, name
