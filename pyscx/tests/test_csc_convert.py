"""pyscx.from_anndata(csc=...) round-trip tests.

Verifies that the new ``csc`` and ``csc_cols_per_shard`` kwargs on
``from_anndata`` correctly emit a CSC sidecar (multi-shard column-major
layout) and that the densified data matches the source AnnData.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def small_adata():
    """20 cells × 15 genes synthetic CSR AnnData."""
    pytest.importorskip("anndata")
    import anndata as ad

    rng = np.random.default_rng(0)
    mat = sp.random(20, 15, density=0.25, format="csr", dtype=np.float32, random_state=rng)
    mat.data = (mat.data * 100).astype(np.float32)
    adata = ad.AnnData(X=mat)
    adata.obs["cell_id"] = [f"c{i}" for i in range(20)]
    adata.var["gene_id"] = [f"g{i}" for i in range(15)]
    return adata


def test_from_anndata_csc_off_default(small_adata, tmp_path):
    """Default `csc='off'` emits no CSC sidecar."""
    import pyscx

    path = tmp_path / "csr_only.scx"
    pyscx.from_anndata(small_adata, str(path))

    exp = pyscx.open(str(path))
    # Round-trip values still match.
    adata2 = exp.to_anndata()
    np.testing.assert_array_almost_equal(
        small_adata.X.toarray(), adata2.X.toarray()
    )


def test_from_anndata_csc_always(small_adata, tmp_path):
    """`csc='always'` emits a CSC sidecar; densified contents match CSR."""
    import pyscx

    path = tmp_path / "with_csc.scx"
    pyscx.from_anndata(small_adata, str(path), csc="always", csc_cols_per_shard=5)

    exp = pyscx.open(str(path))
    adata2 = exp.to_anndata()
    # to_anndata reads CSR; the CSC sidecar must densify to the same
    # values as the source matrix.
    np.testing.assert_array_almost_equal(
        small_adata.X.toarray(), adata2.X.toarray()
    )


def test_from_anndata_csc_invalid_value(small_adata, tmp_path):
    """`csc` only accepts 'off' / 'auto' / 'always'; anything else raises."""
    import pyscx

    path = tmp_path / "bad.scx"
    with pytest.raises(ValueError, match="csc"):
        pyscx.from_anndata(small_adata, str(path), csc="yes")


def _csc_available(path):
    """True iff the file has a usable CSC sidecar. `prefer_format="csc"`
    raises ``RuntimeError`` when no sidecar exists, succeeds otherwise."""
    import pyscx

    adata = pyscx.open(str(path)).to_anndata(backed=True)
    try:
        pyscx.accel.col_sums(adata.X, prefer_format="csc")
        return True
    except RuntimeError:
        return False


def test_from_anndata_csc_auto_below_threshold_skips(small_adata, tmp_path):
    """`csc='auto'` builds no sidecar for a sub-threshold dataset
    (20×15 is far below the 50000-obs / 5000-var defaults)."""
    path = tmp_path / "auto_small.scx"
    import pyscx

    pyscx.from_anndata(small_adata, str(path), csc="auto")
    assert _csc_available(path) is False


def test_from_anndata_csc_auto_env_override_builds(small_adata, tmp_path, monkeypatch):
    """Lowering both thresholds to 0 makes `csc='auto'` build a sidecar
    even on the tiny fixture; densified CSC matches CSR."""
    monkeypatch.setenv("SCX_CSC_AUTO_OBS_THRESHOLD", "0")
    monkeypatch.setenv("SCX_CSC_AUTO_VARS_THRESHOLD", "0")

    path = tmp_path / "auto_forced.scx"
    import pyscx

    pyscx.from_anndata(small_adata, str(path), csc="auto", csc_cols_per_shard=5)
    assert _csc_available(path) is True

    adata = pyscx.open(str(path)).to_anndata(backed=True)
    csr_sums = pyscx.accel.col_sums(adata.X, prefer_format="csr")
    csc_sums = pyscx.accel.col_sums(adata.X, prefer_format="csc")
    np.testing.assert_allclose(csr_sums, csc_sums, atol=1e-9)


# ---------------------------------------------------------------------------
# Phase 0.3: an accel-ready index_preset implies csc="auto" when the caller
# did not pass an explicit `csc`. An explicit value always wins.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["training", "perturbseq"])
def test_index_preset_implies_csc_auto(small_adata, tmp_path, monkeypatch, preset):
    """An accel-ready preset with no explicit `csc` upgrades to "auto".
    With the thresholds lowered to 0, that builds a sidecar on the tiny
    fixture."""
    monkeypatch.setenv("SCX_CSC_AUTO_OBS_THRESHOLD", "0")
    monkeypatch.setenv("SCX_CSC_AUTO_VARS_THRESHOLD", "0")

    path = tmp_path / f"preset_{preset}.scx"
    import pyscx

    pyscx.from_anndata(small_adata, str(path), index_preset=preset, csc_cols_per_shard=5)
    assert _csc_available(path) is True


def test_index_preset_cellxgene_does_not_imply_csc(small_adata, tmp_path, monkeypatch):
    """The query-oriented `cellxgene` preset does NOT imply csc="auto" —
    no sidecar even with the thresholds at 0."""
    monkeypatch.setenv("SCX_CSC_AUTO_OBS_THRESHOLD", "0")
    monkeypatch.setenv("SCX_CSC_AUTO_VARS_THRESHOLD", "0")

    path = tmp_path / "preset_cellxgene.scx"
    import pyscx

    pyscx.from_anndata(small_adata, str(path), index_preset="cellxgene", csc_cols_per_shard=5)
    assert _csc_available(path) is False


def test_explicit_csc_off_overrides_preset(small_adata, tmp_path, monkeypatch):
    """An explicit `csc="off"` wins over an accel-ready preset's implied
    "auto" — the user's explicit choice is honored."""
    monkeypatch.setenv("SCX_CSC_AUTO_OBS_THRESHOLD", "0")
    monkeypatch.setenv("SCX_CSC_AUTO_VARS_THRESHOLD", "0")

    path = tmp_path / "preset_explicit_off.scx"
    import pyscx

    pyscx.from_anndata(
        small_adata, str(path), index_preset="training", csc="off", csc_cols_per_shard=5
    )
    assert _csc_available(path) is False


# ---------------------------------------------------------------------------
# The preset -> csc="auto" upgrade can only fire on an *unset* `csc`, so any
# entry point that spells a non-None default silently opts itself out. That is
# not a hypothetical: `from_10x` shipped with `csc="off"` while accepting
# `index_preset`, so `from_10x(..., index_preset="training")` produced no CSC
# sidecar and GPU DE on the result took the slower `gpu_csr_v3` route, while
# the byte-identical dataset through `from_h5ad` took `gpu_csc_v3`.
#
# The tests above cover `from_anndata` only — 1 of the 4 entry points — which
# is exactly how that drifted. This guard covers the whole surface by
# discovery rather than by enumeration, so a fifth entry point is included the
# day it is added.
# ---------------------------------------------------------------------------


def _csc_preset_entry_points():
    """Every native pyscx function taking both `csc` and `index_preset`.

    Introspects the extension module, NOT the `pyscx` package: `from_h5ad`
    and `from_h5mu` are re-wrapped in ``__init__.py`` as ``(path, out,
    **kwargs)``, so at package level their real parameters are invisible and
    this guard would silently cover 2 of 4.
    """
    import inspect

    import pyscx.pyscx as native

    found = {}
    for name in dir(native):
        if name.startswith("_"):
            continue
        obj = getattr(native, name)
        if not callable(obj):
            continue
        try:
            params = inspect.signature(obj).parameters
        except (TypeError, ValueError):
            continue
        if "csc" in params and "index_preset" in params:
            found[name] = params["csc"].default
    return found


def test_every_preset_aware_entry_point_defaults_csc_to_none():
    """`csc` must default to None wherever `index_preset` is accepted.

    A non-None default (e.g. `"off"`) is indistinguishable from a caller
    who explicitly asked for it, so `resolve_csc_policy` cannot apply the
    accel-ready preset upgrade and the entry point diverges from its
    siblings with no error.
    """
    entry_points = _csc_preset_entry_points()

    # Guard the guard: if discovery finds nothing (module renamed, kwargs
    # dropped), the assertion below would pass vacuously.
    assert entry_points, (
        "no native pyscx function takes both `csc` and `index_preset` — "
        "discovery is broken, not the surface"
    )

    offenders = {n: d for n, d in entry_points.items() if d is not None}
    assert not offenders, (
        f"these entry points accept `index_preset` but pin a non-None `csc` "
        f"default, so the preset -> csc='auto' upgrade can never fire: "
        f"{offenders}. Declare `csc=None` and resolve via "
        f"`scx_engine::index::resolve_csc_policy`."
    )


def _pyfunction_blocks():
    """(name, signature_attr, body) for every `#[pyfunction]` in lib.rs.

    Each block runs from one `#[pyfunction]` to the next, so `body` is the
    whole function including its attributes.
    """
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "lib.rs"
    if not src.is_file():
        pytest.skip(f"pyscx Rust source not available at {src}")

    blocks = []
    for chunk in src.read_text().split("#[pyfunction]")[1:]:
        sig = re.search(r"#\[pyo3\(signature = \((.*?)\)\)\]", chunk, re.DOTALL)
        name = re.search(r"\bfn\s+(\w+)\s*\(", chunk)
        if sig and name:
            blocks.append((name.group(1), sig.group(1), chunk))
    return blocks


def test_every_preset_aware_entry_point_calls_resolve_csc_policy():
    """`csc=None` is necessary but not sufficient — resolution must run.

    Declaring `csc=None` only makes the upgrade *possible*. An entry point
    that then did `csc.unwrap_or("off")`, or passed the `Option` on to a
    default of its own, would satisfy the signature guard above and
    silently reintroduce exactly the opt-out this PR fixes.

    So assert the other half at the source: every `#[pyfunction]` whose
    pyo3 signature carries both `csc` and `index_preset` must call
    `resolve_csc_policy` in its body.

    Scope, stated so it is not over-trusted: this reads `src/lib.rs` only,
    where all four conversion entry points live today. A preset-aware
    entry point added in another module would escape *this* test — the
    runtime signature guard above is the one that covers any module,
    because it introspects the built extension rather than a file.
    """
    blocks = _pyfunction_blocks()
    assert blocks, "parsed no #[pyfunction] blocks out of lib.rs — the scan is broken"

    preset_aware = [
        (name, body)
        for name, sig, body in blocks
        if "csc=" in sig and "index_preset=" in sig
    ]
    assert preset_aware, (
        "found no #[pyfunction] taking both `csc` and `index_preset` — "
        "discovery is broken, not the surface"
    )

    missing = [n for n, body in preset_aware if "resolve_csc_policy" not in body]
    assert not missing, (
        f"these entry points accept both `csc` and `index_preset` but never "
        f"call `resolve_csc_policy`, so an accel-ready preset cannot upgrade "
        f"an unset `csc`: {missing}."
    )


# ---------------------------------------------------------------------------
# The same preset trio, run through `from_10x` — the entry point the signature
# guard above catches statically. These assert the resolution actually reaches
# the writer, which a default-value check alone cannot show.
# ---------------------------------------------------------------------------


@pytest.fixture
def tenx_h5(tmp_path):
    """A minimal 10x CellRanger v3 HDF5, 20 cells x 15 genes.

    `scanpy.read_10x_h5` dispatches on `"/matrix" in f` alone, so the v3
    layout is all that is required. Two details are load-bearing:

    * `shape` is stored transposed, `[n_vars, n_obs]` — 10x writes the
      matrix gene-major CSC, which is the same buffer scanpy reads back as
      cell-major CSR of shape `(n_obs, n_vars)`.
    * `feature_type` must be `"Gene Expression"`; `read_10x_h5` defaults to
      `gex_only=True` and would otherwise filter every column away, leaving
      a 20x0 matrix that fails for an unrelated reason.

    Strings are written as fixed-length bytes (`dtype="S"`) because that is
    what CellRanger emits, so the fixture exercises the same `.astype(str)`
    decode scanpy performs on a real file rather than whatever dtype h5py
    happens to infer from a list of Python `str`.
    """
    h5py = pytest.importorskip("h5py")

    rng = np.random.default_rng(7)
    counts = sp.random(20, 15, density=0.3, format="csr", random_state=rng)
    counts.data = (counts.data * 50 + 1).astype(np.int32)

    path = tmp_path / "filtered_feature_bc_matrix.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=counts.data.astype(np.int32))
        g.create_dataset("indices", data=counts.indices.astype(np.int64))
        g.create_dataset("indptr", data=counts.indptr.astype(np.int64))
        # Transposed, per the 10x spec: [n_vars, n_obs].
        g.create_dataset("shape", data=np.array([15, 20], dtype=np.int32))
        g.create_dataset(
            "barcodes", data=np.array([f"CELL{i:03d}-1" for i in range(20)], dtype="S")
        )
        feats = g.create_group("features")
        feats.create_dataset(
            "id", data=np.array([f"ENSG{i:08d}" for i in range(15)], dtype="S")
        )
        feats.create_dataset("name", data=np.array([f"GENE{i}" for i in range(15)], dtype="S"))
        feats.create_dataset("feature_type", data=np.array(["Gene Expression"] * 15, dtype="S"))
    return path


def test_from_10x_fixture_round_trips(tenx_h5, tmp_path):
    """Pin the fixture's contract: 20x15, real values, decoded names.

    This is a positive characterization, not a guard against a known
    silent failure. Both fixture corruptions worth worrying about were
    tried and neither is silent: a wrong `feature_type` leaves scanpy
    with a 20x0 matrix and `from_10x` raises "Arrow IPC contains no
    batches", and an un-transposed `shape` raises "index pointer size 21
    should be 16". So the cases below would fail loudly, not vacuously,
    without this test. It earns its place by stating what the fixture is
    supposed to be, so a future edit to it has something to violate.
    """
    pytest.importorskip("scanpy")
    import pyscx

    path = tmp_path / "tenx_roundtrip.scx"
    pyscx.from_10x(str(tenx_h5), str(path))

    adata = pyscx.open(str(path)).to_anndata()
    assert adata.shape == (20, 15)
    assert adata.X.nnz > 0
    assert list(adata.obs_names[:2]) == ["CELL000-1", "CELL001-1"]
    assert list(adata.var_names[:2]) == ["GENE0", "GENE1"]


@pytest.mark.parametrize("preset", ["training", "perturbseq"])
def test_from_10x_index_preset_implies_csc_auto(tenx_h5, tmp_path, monkeypatch, preset):
    """`from_10x` honors the preset -> csc="auto" upgrade, like its siblings.

    Regression: `from_10x` declared `csc="off"`, so this produced a
    CSR-only file and GPU DE on it silently fell back to `gpu_csr_v3`.
    """
    pytest.importorskip("scanpy")
    import pyscx

    monkeypatch.setenv("SCX_CSC_AUTO_OBS_THRESHOLD", "0")
    monkeypatch.setenv("SCX_CSC_AUTO_VARS_THRESHOLD", "0")

    path = tmp_path / f"tenx_{preset}.scx"
    pyscx.from_10x(str(tenx_h5), str(path), index_preset=preset, csc_cols_per_shard=5)
    assert _csc_available(path) is True


def test_from_10x_index_preset_cellxgene_does_not_imply_csc(tenx_h5, tmp_path, monkeypatch):
    """The query-oriented preset must NOT be upgraded — the fix resolves
    the policy, it does not turn CSC on for every preset."""
    pytest.importorskip("scanpy")
    import pyscx

    monkeypatch.setenv("SCX_CSC_AUTO_OBS_THRESHOLD", "0")
    monkeypatch.setenv("SCX_CSC_AUTO_VARS_THRESHOLD", "0")

    path = tmp_path / "tenx_cellxgene.scx"
    pyscx.from_10x(str(tenx_h5), str(path), index_preset="cellxgene", csc_cols_per_shard=5)
    assert _csc_available(path) is False


def test_from_10x_explicit_csc_off_overrides_preset(tenx_h5, tmp_path, monkeypatch):
    """An explicit `csc="off"` still wins over an accel-ready preset."""
    pytest.importorskip("scanpy")
    import pyscx

    monkeypatch.setenv("SCX_CSC_AUTO_OBS_THRESHOLD", "0")
    monkeypatch.setenv("SCX_CSC_AUTO_VARS_THRESHOLD", "0")

    path = tmp_path / "tenx_explicit_off.scx"
    pyscx.from_10x(
        str(tenx_h5), str(path), index_preset="training", csc="off", csc_cols_per_shard=5
    )
    assert _csc_available(path) is False
