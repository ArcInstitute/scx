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


def test_from_anndata_default_below_the_auto_threshold_is_csr_only(small_adata, tmp_path):
    """The default is `csc="auto"`, which a 20 x 15 matrix does not clear, so
    no sidecar — and the values still round-trip."""
    import pyscx

    path = tmp_path / "csr_only.scx"
    pyscx.from_anndata(small_adata, str(path))
    assert _csc_available(path) is False

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
# An unset `csc` is "auto" on every entry point, whatever the `index_preset`.
# It used to be "off" unless the preset was `training` / `perturbseq`, so the
# preset axis is kept to show that no preset still opts out. An explicit value
# always wins.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", [None, "training", "perturbseq", "cellxgene"])
def test_unset_csc_is_auto_for_every_preset(small_adata, tmp_path, monkeypatch, preset):
    """With the thresholds lowered to 0, the default builds a sidecar on the
    tiny fixture — with or without a preset, `cellxgene` included."""
    monkeypatch.setenv("SCX_CSC_AUTO_OBS_THRESHOLD", "0")
    monkeypatch.setenv("SCX_CSC_AUTO_VARS_THRESHOLD", "0")

    path = tmp_path / f"preset_{preset}.scx"
    import pyscx

    pyscx.from_anndata(small_adata, str(path), index_preset=preset, csc_cols_per_shard=5)
    assert _csc_available(path) is True


def test_explicit_csc_off_overrides_preset(small_adata, tmp_path, monkeypatch):
    """An explicit `csc="off"` wins over the "auto" default — the opt-out."""
    monkeypatch.setenv("SCX_CSC_AUTO_OBS_THRESHOLD", "0")
    monkeypatch.setenv("SCX_CSC_AUTO_VARS_THRESHOLD", "0")

    path = tmp_path / "preset_explicit_off.scx"
    import pyscx

    pyscx.from_anndata(
        small_adata, str(path), index_preset="training", csc="off", csc_cols_per_shard=5
    )
    assert _csc_available(path) is False


# ---------------------------------------------------------------------------
# The shared "auto" default applies only to an *unset* `csc`, so any entry
# point that spells a non-None default silently opts itself out. That is not a
# hypothetical: `from_10x` shipped with `csc="off"` (back when the default was
# "off" and only an accel-ready preset upgraded it), so
# `from_10x(..., index_preset="training")` produced no CSC sidecar and GPU DE on
# the result took the slower `gpu_csr_v3` route, while the byte-identical
# dataset through `from_h5ad` took `gpu_csc_v3`.
#
# The tests above cover `from_anndata` only — 1 of the 4 entry points — which
# is exactly how that drifted. This guard covers the whole surface by
# discovery rather than by enumeration, so a fifth entry point is included the
# day it is added.
# ---------------------------------------------------------------------------


# `from_mudata` keeps csc="off" on purpose: it cannot build per-modality
# sidecars (it refuses "always" and degrades "auto"), so the ingest default
# would buy nothing there. Named, so it is the one exclusion and not a gap.
_CSC_OFF_BY_DESIGN = frozenset({"from_mudata"})

_MISSING = object()


def _ingest_entry_points():
    """Every native pyscx `from_*` ingest function, mapped to its `csc` default
    (`_MISSING` when it takes no `csc` at all).

    Keyed on the `from_*` name rather than on the parameters an entry point
    happens to take: an earlier version selected functions taking both `csc`
    and `index_preset`, which by construction could not see an ingest function
    missing `csc` — `from_mtx` was exactly that, CSR-only on any size.

    Introspects the extension module, NOT the `pyscx` package: `from_h5ad`
    and `from_h5mu` are re-wrapped in ``__init__.py`` as ``(path, out,
    **kwargs)``, so at package level their real parameters are invisible.
    """
    import inspect

    import pyscx.pyscx as native

    found = {}
    for name in dir(native):
        if not name.startswith("from_") or name in _CSC_OFF_BY_DESIGN:
            continue
        obj = getattr(native, name)
        if not callable(obj):
            continue
        params = inspect.signature(obj).parameters
        found[name] = params["csc"].default if "csc" in params else _MISSING
    return found


# The rewrite ops take a `csc` too, but a different one: whether the output
# carries a sidecar through the rewrite (`"carry"` / `"always"` / `"off"`), not
# an ingest `CscPolicy`. Their default is "carry" — rewriting a file that has
# no sidecar builds none, which is also what those ops did before they carried
# one — so the None-default rule above does not apply.
_REWRITE_OPS = frozenset({"compact", "merge", "optimize", "sort", "shuffle"})


def test_the_rewrite_ops_excluded_above_are_the_carry_kind():
    """The exclusion cannot hide an ingest entry point: every name in it takes
    `csc` with the rewrite ops' `"carry"` default, and nothing else does."""
    import inspect

    import pyscx.pyscx as native

    for name in sorted(_REWRITE_OPS):
        params = inspect.signature(getattr(native, name)).parameters
        # `sort` / `shuffle` default to None so an explicit `csc=` can be told
        # apart from the deprecated `rebuild_csc=`; None resolves to "carry".
        expected = None if name in ("sort", "shuffle") else "carry"
        assert params["csc"].default == expected, (name, params["csc"].default)


def test_every_ingest_entry_point_defaults_csc_to_none():
    """`csc` must default to None on every ingest entry point.

    A non-None default (e.g. `"off"`) is indistinguishable from a caller
    who explicitly asked for it, so `resolve_csc_policy` never supplies
    the shared "auto" default and the entry point diverges from its
    siblings with no error. An entry point with no `csc` at all is the same
    divergence with no way for the caller to fix it.
    """
    entry_points = _ingest_entry_points()

    # Guard the guard: the five ingest entry points must all be discovered,
    # or the assertion below passes over the ones it missed.
    assert {"from_anndata", "from_h5ad", "from_h5mu", "from_10x", "from_mtx"} <= set(
        entry_points
    ), f"discovery is broken, not the surface: found {sorted(entry_points)}"

    offenders = {
        n: ("<no csc parameter>" if d is _MISSING else d)
        for n, d in entry_points.items()
        if d is not None
    }
    assert not offenders, (
        f"these ingest entry points pin a non-None `csc` default, so the "
        f"shared csc='auto' default can never apply: "
        f"{offenders}. Declare `csc=None` and resolve via "
        f"`scx_engine::index::resolve_csc_policy`."
    )


def _rust_body_after(text, start):
    """The `{...}` body of the fn whose `fn name(` match ended at `start`.

    Brace-matched, so it stops at the function's own closing brace. Naive
    "everything until the next item" slicing would swallow the following
    function's rustdoc, and a `resolve_csc_policy` mention left in *that*
    would mask a deleted call.

    Only whole-line comments are dropped. Splitting every line at its
    first `//` would truncate `"https://…"` mid-string, which can only
    lose a real call (a loud failure), but there is no reason to accept
    even that when the rustdoc case this exists for is always whole-line.

    The brace counter does not know about `{` inside string literals or
    `/* */`, so an unbalanced one could run the body past the function's
    end — the one direction that could *hide* a missing call. The caller
    checks for that rather than this trying to lex Rust.
    """
    i = text.index("{", start)
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    body = text[i : j + 1]
    return "\n".join(
        line for line in body.splitlines() if not line.lstrip().startswith("//")
    )


def _pyfunction_blocks():
    """(name, signature_attr, body) for every `#[pyfunction]` in lib.rs.

    `body` is the function's brace-matched body with comments stripped —
    not the raw span between attributes.
    """
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "lib.rs"
    if not src.is_file():
        pytest.skip(f"pyscx Rust source not available at {src}")

    text = src.read_text()
    blocks = []
    for m in re.finditer(r"#\[pyfunction\]", text):
        nxt = text.find("#[pyfunction]", m.end())
        head = text[m.end() : nxt if nxt != -1 else len(text)]
        sig = re.search(r"#\[pyo3\(signature = \((.*?)\)\)\]", head, re.DOTALL)
        name = re.search(r"\bfn\s+(\w+)\s*\(", head)
        if sig and name:
            body = _rust_body_after(text, m.end() + name.end())
            blocks.append((name.group(1), sig.group(1), body))
    return blocks


def test_every_ingest_entry_point_calls_resolve_csc_policy():
    """`csc=None` is necessary but not sufficient — resolution must run.

    Declaring `csc=None` only makes the shared default *possible*. An entry
    point that then did `csc.unwrap_or("off")`, or passed the `Option` on to
    a default of its own, would satisfy the signature guard above and
    silently opt out of the "auto" default its siblings have.

    So assert the other half at the source: every `from_*` `#[pyfunction]`
    except the `_CSC_OFF_BY_DESIGN` ones must call `resolve_csc_policy` in
    its body.

    The body is brace-matched and comment-stripped, so a `resolve_csc_policy`
    mention in a neighbouring function's rustdoc or in a comment cannot
    stand in for a real call.

    Scope, stated so it is not over-trusted: this reads `src/lib.rs` only,
    where all five ingest entry points live today. One added in another
    module would escape *this* test — the
    runtime signature guard above is the one that covers any module,
    because it introspects the built extension rather than a file.
    """
    blocks = _pyfunction_blocks()
    assert blocks, "parsed no #[pyfunction] blocks out of lib.rs — the scan is broken"

    # A body that ran past its own closing brace would swallow the next
    # function and could borrow *its* call — the only failure direction
    # that hides a missing one. An over-run always captures the following
    # `#[pyfunction]`, so its absence is the integrity check.
    overrun = [n for n, _, body in blocks if "#[pyfunction]" in body]
    assert not overrun, f"brace matching ran past the end of: {overrun}"

    ingest = [
        (name, body)
        for name, _sig, body in blocks
        if name.startswith("from_") and name not in _CSC_OFF_BY_DESIGN
    ]
    assert {"from_anndata", "from_h5ad", "from_h5mu", "from_10x", "from_mtx"} <= {
        n for n, _ in ingest
    }, f"discovery is broken, not the surface: found {[n for n, _ in ingest]}"

    # `resolve_csc_policy(` — a call, not a bare mention.
    missing = [n for n, body in ingest if "resolve_csc_policy(" not in body]
    assert not missing, (
        f"these ingest entry points never call `resolve_csc_policy`, so an "
        f"unset `csc` does not get the shared 'auto' default: {missing}."
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
        g.create_dataset("data", data=counts.data)  # already int32
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


@pytest.mark.parametrize("preset", [None, "training", "cellxgene"])
def test_from_10x_unset_csc_is_auto(tenx_h5, tmp_path, monkeypatch, preset):
    """`from_10x` resolves an unset `csc` to "auto" like its siblings.

    Regression: `from_10x` once declared `csc="off"`, so it produced a
    CSR-only file where the other entry points built a sidecar.
    """
    pytest.importorskip("scanpy")
    import pyscx

    monkeypatch.setenv("SCX_CSC_AUTO_OBS_THRESHOLD", "0")
    monkeypatch.setenv("SCX_CSC_AUTO_VARS_THRESHOLD", "0")

    path = tmp_path / f"tenx_{preset}.scx"
    pyscx.from_10x(str(tenx_h5), str(path), index_preset=preset, csc_cols_per_shard=5)
    assert _csc_available(path) is True


def test_from_10x_explicit_csc_off_overrides_preset(tenx_h5, tmp_path, monkeypatch):
    """An explicit `csc="off"` still wins over the "auto" default."""
    pytest.importorskip("scanpy")
    import pyscx

    monkeypatch.setenv("SCX_CSC_AUTO_OBS_THRESHOLD", "0")
    monkeypatch.setenv("SCX_CSC_AUTO_VARS_THRESHOLD", "0")

    path = tmp_path / "tenx_explicit_off.scx"
    pyscx.from_10x(
        str(tenx_h5), str(path), index_preset="training", csc="off", csc_cols_per_shard=5
    )
    assert _csc_available(path) is False


def test_from_anndata_memory_budget_reaches_the_csc_sidecar(tmp_path):
    """`memory_budget` binds the CSC sidecar transpose, not just the obsm warning.

    `pyscx.from_anndata` was the fifth call site still passing the sidecar
    builder's 4 GiB default unconditionally, so a caller who set
    `memory_budget` to bound memory got a sidecar that ignored it. It sits
    outside `scx-convert/src`, so the CI guard covering the other four did not
    see it.

    ⚠️ This is a **behaviour change on a public API**, which is why it is pinned
    here rather than left to the Rust tests: the budget now genuinely binds, so
    a value too small for one column chunk raises where it previously
    succeeded. Loud and actionable beats silently ignoring a budget the caller
    asked for — but it is a new failure mode and belongs in a test that says so.

    The docstring for `memory_budget` promised "warn-only" before this; the two
    have to move together.
    """
    import anndata as ad
    import numpy as np
    import pyscx
    import pytest
    import scipy.sparse as sp

    # 2000 rows => one column chunk needs 2000 * 12 = 24_000 bytes.
    adata = ad.AnnData(X=sp.random(2000, 50, density=0.9, format="csr", dtype=np.float32))

    # Comfortably above the per-chunk minimum: builds, sidecar present.
    ok = tmp_path / "ok.scx"
    pyscx.from_anndata(adata, str(ok), csc="always", memory_budget=1_000_000)
    assert pyscx.open(str(ok)).has_csc

    # Below it: the budget binds, so this must fail rather than quietly
    # allocating 4 GiB worth of chunk.
    with pytest.raises(RuntimeError, match="memory limit too small"):
        pyscx.from_anndata(
            adata, str(tmp_path / "too_small.scx"), csc="always", memory_budget=1024
        )

    # And an unset budget still uses the builder's own default.
    unset = tmp_path / "unset.scx"
    pyscx.from_anndata(adata, str(unset), csc="always")
    assert pyscx.open(str(unset)).has_csc
