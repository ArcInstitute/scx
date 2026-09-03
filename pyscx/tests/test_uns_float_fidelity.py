"""`uns` float fidelity: exact doubles and non-finite scalars.

Two contracts, both pinned bit-for-bit:

* A finite double written through any `uns` writer reads back as the same
  IEEE-754 pattern. The writer was always exact (ryu); the reader used to parse
  with `serde_json`'s fast path, which lands ~30 % of 17-significant-digit
  doubles one ULP off. The workspace now enables `float_roundtrip`.
* A raw Python `float("nan")` / `±inf` — bare, or inside a list / tuple /
  dict / object-bearing structured array / pandas `name` — is preserved under
  the default `uns_format="tagged"` via the existing `scalar` envelope (it
  reads back as `np.float64`, a `float` subclass). `uns_format="plain"` keeps
  raising: plain mode's contract is "lossless or refuse", and JSON has no
  NaN/Inf literal.
"""

import math
import struct

import numpy as np
import pandas as pd
import pytest

import pyscx


def _adata_with_uns(uns):
    import anndata
    import scipy.sparse as sp

    n_obs, n_vars = 2, 2
    a = anndata.AnnData(
        X=sp.csr_matrix(np.zeros((n_obs, n_vars), dtype=np.float32)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )
    a.uns.update(uns)
    return a


def _bits(x):
    return struct.pack("<d", float(x))


def _random_finite_doubles(n, seed=0):
    """Uniform over the bit patterns, not over the reals: exercises every
    exponent, including the 17-digit values that used to drift."""
    rng = np.random.default_rng(seed)
    vals = np.frombuffer(rng.bytes(8 * n * 2), dtype="<f8")
    vals = vals[np.isfinite(vals)][:n]
    assert len(vals) == n
    return vals.tolist()


# ---------------------------------------------------------------------------
# Exact doubles
# ---------------------------------------------------------------------------


def test_set_uns_reads_back_every_double_bit_for_bit(tmp_dir):
    src = _random_finite_doubles(500)
    adata = _adata_with_uns({})
    path = str(tmp_dir / "doubles.scx")
    pyscx.from_anndata(adata, path)

    pyscx.set_uns(path, {"one": -1.739269179332728e61, "many": src})

    uns = pyscx.open(path).read_uns()
    assert _bits(uns["one"]) == _bits(-1.739269179332728e61)
    drifted = [(a, b) for a, b in zip(src, uns["many"]) if _bits(a) != _bits(b)]
    assert drifted == [], f"{len(drifted)} of {len(src)} doubles changed on read"


def test_from_anndata_reads_back_every_double_bit_for_bit(tmp_dir):
    src = _random_finite_doubles(500, seed=1)
    path = str(tmp_dir / "doubles_fa.scx")
    pyscx.from_anndata(_adata_with_uns({"many": src}), path)

    out = pyscx.open(path).to_anndata().uns["many"]
    drifted = [(a, b) for a, b in zip(src, out) if _bits(a) != _bits(b)]
    assert drifted == []


# ---------------------------------------------------------------------------
# Non-finite scalars — tagged preserves, plain refuses
# ---------------------------------------------------------------------------

_NON_FINITE = {
    "top": float("nan"),
    "lst": [1.0, float("nan"), float("inf"), -float("inf")],
    "tup": (float("nan"), 2.0),
    "nested": {"a": -float("inf")},
}


def _assert_non_finite_round_trip(uns):
    assert math.isnan(uns["top"])
    assert isinstance(uns["top"], float)
    lst = list(uns["lst"])
    assert lst[0] == 1.0
    assert math.isnan(lst[1])
    assert lst[2] == math.inf
    assert lst[3] == -math.inf
    assert all(isinstance(v, float) for v in lst)
    tup = uns["tup"]
    assert isinstance(tup, tuple)
    assert math.isnan(tup[0]) and tup[1] == 2.0
    assert uns["nested"]["a"] == -math.inf


def test_non_finite_python_floats_round_trip_via_from_anndata(tmp_dir):
    path = str(tmp_dir / "nonfinite_fa.scx")
    pyscx.from_anndata(_adata_with_uns(_NON_FINITE), path)
    _assert_non_finite_round_trip(pyscx.open(path).to_anndata().uns)


def test_non_finite_python_floats_round_trip_via_set_uns(tmp_dir):
    path = str(tmp_dir / "nonfinite_set.scx")
    pyscx.from_anndata(_adata_with_uns({}), path)
    pyscx.set_uns(path, _NON_FINITE)
    _assert_non_finite_round_trip(pyscx.open(path).read_uns())


def test_non_finite_in_object_bearing_structured_array_keeps_the_sign(tmp_dir):
    """The `fields_json` recarray path used to map every non-finite leaf to
    JSON `null`, so `-inf` came back as `nan`. scanpy's `rank_genes_groups`
    record (object `names` + float `pvals_adj`) is exactly this shape."""
    src = np.array(
        [("a", math.inf), ("b", -math.inf), ("c", math.nan), ("d", 0.5)],
        dtype=[("names", "O"), ("pvals", "<f8")],
    )
    path = str(tmp_dir / "recarray.scx")
    pyscx.from_anndata(_adata_with_uns({"rgg": src}), path)

    rt = pyscx.open(path).to_anndata().uns["rgg"]
    assert isinstance(rt, np.ndarray)
    assert list(rt["names"]) == ["a", "b", "c", "d"]
    assert np.array_equal(rt["pvals"].view(np.uint64), src["pvals"].view(np.uint64))


def test_pandas_index_with_a_non_finite_name_round_trips(tmp_dir):
    path = str(tmp_dir / "index_name.scx")
    pyscx.from_anndata(
        _adata_with_uns({"idx": pd.Index(["x", "y"], name=float("nan"))}), path
    )
    idx = pyscx.open(path).to_anndata().uns["idx"]
    assert isinstance(idx, pd.Index)
    assert list(idx) == ["x", "y"]
    assert math.isnan(idx.name)


def test_non_finite_python_float_still_raises_under_plain(tmp_dir):
    with pytest.raises(
        ValueError,
        match=r"uns at uns\['top'\]: non-finite float \(NaN\) cannot be serialized",
    ):
        pyscx.from_anndata(
            _adata_with_uns({"top": float("nan")}),
            str(tmp_dir / "plain_top.scx"),
            uns_format="plain",
        )
    with pytest.raises(
        ValueError,
        match=r"uns at uns\['lst'\]\[1\]: non-finite float \(NaN\) cannot be serialized",
    ):
        pyscx.from_anndata(
            _adata_with_uns({"lst": [1.0, float("nan")]}),
            str(tmp_dir / "plain_lst.scx"),
            uns_format="plain",
        )
