"""What a *base* `pip install pyscx` can and cannot do.

The development venv has scanpy, scikit-misc, pydeseq2, formulaic, polars and
mudata installed, so an unguarded assertion here proves nothing at all — every
optional path passes because every optional package happens to be present. Every
test in this file therefore runs under `blocked(...)`, a `sys.meta_path` finder
that makes the named modules raise `ModuleNotFoundError` exactly as they would on
a machine that never installed them.

Two things are asserted, and they are different:

1.  **The base surface works.** The calls a new user makes first must not need an
    extra. `test_base_install_*`.
2.  **Everything else says what to install.** A path that genuinely needs an
    optional package must fail with the op, the distribution and the exact
    `pip install 'pyscx[extra]'` — and that extra must actually exist in
    `pyproject.toml`, which is a separate assertion because the message and the
    packaging can drift apart silently.
"""

from __future__ import annotations

import contextlib
import re
import subprocess
import sys
import textwrap
import tomllib
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import pyscx

PYSCX_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# The blocker
# ---------------------------------------------------------------------------


class _Blocker:
    """A meta-path finder that raises `ModuleNotFoundError` for `names`.

    A name without a dot blocks that package and everything under it; a dotted
    name blocks *only* that submodule, which is how the transitive-failure test
    keeps `skmisc` importable while `skmisc.loess` fails.
    """

    def __init__(self, names):
        self.exact = {n for n in names if "." in n}
        self.tops = {n for n in names if "." not in n}

    def find_spec(self, name, path=None, target=None):
        if name in self.exact or name.split(".")[0] in self.tops:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None

    def _blocks(self, module_name: str) -> bool:
        return module_name in self.exact or module_name.split(".")[0] in self.tops


@contextlib.contextmanager
def blocked(*names: str):
    """Make `names` un-importable for the duration of the block.

    Not thread-safe: it mutates `sys.modules` and `sys.meta_path`, which are
    interpreter-global. Fine under serial pytest, but these tests must not be
    run under a threaded runner (`pytest-parallel` and friends) — process-level
    parallelism such as `pytest-xdist` is fine, since each worker is its own
    interpreter.
    """
    blocker = _Blocker(names)
    saved = {k: v for k, v in sys.modules.items() if blocker._blocks(k)}
    for k in saved:
        del sys.modules[k]
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)


# The blocker self-tests deliberately use a *hard* dependency and the stdlib,
# never an optional package: they also run in the `base-install` CI job, where
# scanpy and scikit-misc are absent by design, and "the blocker works" must not
# depend on the thing it exists to pretend is missing.


def test_the_blocker_actually_blocks():
    """The premise of every other test in this file.

    Without this, a bug in `blocked` turns the whole module into a suite that
    passes because nothing was ever blocked.
    """
    with blocked("pyarrow"):
        with pytest.raises(ModuleNotFoundError):
            __import__("pyarrow")
    __import__("pyarrow")  # restored


def test_the_blocker_is_submodule_precise():
    with blocked("email.mime"):
        __import__("email")  # parent still importable
        with pytest.raises(ModuleNotFoundError):
            __import__("email.mime")
    __import__("email.mime")  # restored


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _adata(n_obs: int = 80, n_vars: int = 40, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    x = sp.csr_matrix(rng.poisson(2.0, size=(n_obs, n_vars)).astype(np.float32))
    a = ad.AnnData(x)
    a.obs_names = [f"cell{i}" for i in range(n_obs)]
    a.var_names = [f"gene{i}" for i in range(n_vars)]
    a.obs["cond"] = pd.Categorical(["ctrl"] * (n_obs // 2) + ["trt"] * (n_obs - n_obs // 2))
    a.obs["donor"] = pd.Categorical([f"d{i % 4}" for i in range(n_obs)])
    a.obs["batch"] = pd.Categorical([f"b{i % 2}" for i in range(n_obs)])
    return a


@pytest.fixture
def adata():
    return _adata()


@pytest.fixture
def scx_path(tmp_path):
    p = tmp_path / "base.scx"
    pyscx.from_anndata(_adata(), str(p))
    return str(p)


# ---------------------------------------------------------------------------
# 1. Every optional path names its extra
# ---------------------------------------------------------------------------

# (id, blocked module, expected extra, op-name fragment, callable)
_OPTIONAL_PATHS = [
    (
        "hvg_seurat_v3",
        "skmisc",
        "hvg",
        "highly_variable_genes",
        lambda a: pyscx.accel.highly_variable_genes(a, n_top_genes=10),
    ),
    (
        "hvg_cell_ranger",
        "scanpy",
        "scanpy",
        "highly_variable_genes",
        lambda a: pyscx.accel.highly_variable_genes(a, n_top_genes=10, flavor="cell_ranger"),
    ),
    (
        "hvg_seurat_batched",
        "scanpy",
        "scanpy",
        "highly_variable_genes",
        lambda a: pyscx.accel.highly_variable_genes(
            a, n_top_genes=10, flavor="seurat", batch_key="batch"
        ),
    ),
    (
        "normalize_total",
        "scanpy",
        "scanpy",
        "normalize_total",
        lambda a: pyscx.accel.normalize_total(a),
    ),
    ("log1p", "scanpy", "scanpy", "log1p", lambda a: pyscx.accel.log1p(a)),
    (
        "filter_cells",
        "scanpy",
        "scanpy",
        "filter_cells",
        lambda a: pyscx.accel.filter_cells(a, min_genes=1),
    ),
    (
        "filter_genes",
        "scanpy",
        "scanpy",
        "filter_genes",
        lambda a: pyscx.accel.filter_genes(a, min_cells=1),
    ),
    (
        "pseudobulk_dex_pydeseq2",
        "pydeseq2",
        "pydeseq2",
        "pseudobulk_dex",
        lambda a: pyscx.accel.pseudobulk_dex(
            a,
            groupby=["cond", "donor"],
            test_col="cond",
            reference="ctrl",
            min_cells_per_group=1,
            backend="pydeseq2",
        ),
    ),
]


@pytest.mark.parametrize(
    "blocked_module,extra,op,call",
    [p[1:] for p in _OPTIONAL_PATHS],
    ids=[p[0] for p in _OPTIONAL_PATHS],
)
def test_an_optional_path_names_its_extra(adata, blocked_module, extra, op, call):
    """A missing optional dependency must not surface as a bare import error.

    `ModuleNotFoundError: No module named 'skmisc'` is technically true and
    practically useless: `skmisc` is not the name of anything the user installed,
    asked for, or can find on PyPI under that spelling.
    """
    with blocked(blocked_module), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ModuleNotFoundError) as excinfo:
            call(adata)
    msg = str(excinfo.value)
    assert op in msg, f"error does not name the op that failed: {msg!r}"
    assert f"pyscx[{extra}]" in msg, f"error does not name the extra to install: {msg!r}"


def test_a_missing_optional_dep_is_an_import_error_not_a_runtime_error():
    """`except ImportError` is how callers probe for an optional feature.

    `pseudobulk_dex(backend="pydeseq2")` used to raise `RuntimeError`, which that
    idiom does not catch.
    """
    a = _adata()
    with blocked("pydeseq2"):
        with pytest.raises(ImportError):
            pyscx.accel.pseudobulk_dex(
                a,
                groupby=["cond", "donor"],
                test_col="cond",
                reference="ctrl",
                min_cells_per_group=1,
                backend="pydeseq2",
            )


# ---------------------------------------------------------------------------
# 2. A broken optional dep is not a missing one
# ---------------------------------------------------------------------------


def test_a_transitive_import_failure_is_not_relabelled_as_not_installed(adata):
    """`skmisc` installed but `skmisc.loess` failing is a real diagnosis.

    scikit-misc ships a compiled extension; an ABI mismatch against the installed
    numpy fails on the *submodule*. Reporting that as "scikit-misc is not
    installed. Install it with pip install 'pyscx[hvg]'" sends the user to
    reinstall a package they already have, and hides the only line that says what
    is actually wrong.
    """
    # The premise is "scikit-misc IS installed but its submodule fails". Without
    # this, a machine that simply lacks scikit-misc takes the *rewrite* branch
    # (correctly — the package really is missing) and the test fails claiming a
    # bug that is not there. Verified: it fails exactly this way in a base venv.
    pytest.importorskip("skmisc")
    with blocked("skmisc.loess"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ModuleNotFoundError) as excinfo:
            pyscx.accel.highly_variable_genes(adata, n_top_genes=10)
    msg = str(excinfo.value)
    assert "skmisc.loess" in msg
    assert "pyscx[" not in msg, f"a submodule failure was rewritten as a missing extra: {msg!r}"


class _Breaker:
    """Raise a plain `ImportError` — a shared library that will not load, not a
    module that is absent."""

    def find_spec(self, name, path=None, target=None):
        if name == "scanpy":
            raise ImportError("libstdc++.so.6: version `GLIBCXX_3.4.32' not found")
        return None


def test_a_non_modulenotfound_import_error_propagates_untouched(adata):
    """The other half of the same rule, and the one no `.name` check covers.

    A dependency present but unloadable raises `ImportError`, not
    `ModuleNotFoundError`. `map_err(|_| …)` — the pattern this helper replaced —
    catches it and replies "scanpy is not installed", which is both false and a
    dead end.
    """
    breaker = _Breaker()
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] == "scanpy"}
    for k in saved:
        del sys.modules[k]
    sys.meta_path.insert(0, breaker)
    try:
        with pytest.raises(ImportError) as excinfo:
            pyscx.accel.log1p(adata)
    finally:
        sys.meta_path.remove(breaker)
        sys.modules.update(saved)
    msg = str(excinfo.value)
    assert "GLIBCXX" in msg, f"the real failure was replaced: {msg!r}"
    assert "pyscx[" not in msg, f"a load failure was rewritten as a missing extra: {msg!r}"


# ---------------------------------------------------------------------------
# 3. The extras the messages advertise must exist
# ---------------------------------------------------------------------------


def _declared_extras() -> set[str]:
    with open(PYSCX_ROOT / "pyproject.toml", "rb") as fh:
        return set(tomllib.load(fh)["project"].get("optional-dependencies", {}))


def _advertised_extras() -> dict[str, list[str]]:
    """Every extra `pyscx` can name in an error, mapped to where it comes from.

    Two sources, because either alone can go stale: the `EXTRA_*` constants in
    `optional_deps.rs` (which is what call sites should pass), and any bare
    string literal in the extra position of an `import_optional*` call (which is
    what someone will write when they add a site in a hurry).
    """
    found: dict[str, list[str]] = {}
    const_re = re.compile(r'const\s+EXTRA_\w+\s*:\s*&str\s*=\s*"([^"]+)"')
    literal_re = re.compile(
        r'import_optional(?:_with_hint)?\(\s*py\s*,\s*"[^"]+"\s*,\s*"([^"]+)"', re.MULTILINE
    )
    for rs in sorted((PYSCX_ROOT / "src").rglob("*.rs")):
        text = rs.read_text()
        for pattern in (const_re, literal_re):
            for m in pattern.finditer(text):
                found.setdefault(m.group(1), []).append(str(rs.relative_to(PYSCX_ROOT)))
    return found


def test_every_advertised_extra_is_declared():
    """A perfect error message pointing at an extra that does not exist is the
    same bug wearing a better shirt — `pip install 'pyscx[hvg]'` would fail with
    a pip warning and install nothing."""
    advertised = _advertised_extras()
    assert advertised, "no import_optional call sites found — has the helper moved?"
    declared = _declared_extras()
    missing = {e: sites for e, sites in advertised.items() if e not in declared}
    assert not missing, f"extras named in errors but absent from pyproject.toml: {missing}"


@pytest.mark.parametrize("extra,dist", [("scanpy", "scanpy"), ("hvg", "scikit-misc"), ("pydeseq2", "pydeseq2")])
def test_the_new_extras_pin_the_package_they_promise(extra, dist):
    with open(PYSCX_ROOT / "pyproject.toml", "rb") as fh:
        groups = tomllib.load(fh)["project"]["optional-dependencies"]
    assert extra in groups, f"missing extra: {extra}"
    assert any(
        re.match(rf"{re.escape(dist)}\b", spec) for spec in groups[extra]
    ), f"extra {extra!r} does not pin {dist!r}: {groups[extra]}"


def test_the_dev_extra_can_run_the_test_suite():
    """`pip install -e '.[dev]'` must install what the suite imports.

    `test_hvg_loess_singularity.py` imports `skmisc`; several `test_accel.py`
    tests need `pydeseq2`. Both were absent from `dev`, so the extra that exists
    to run the tests could not run them.
    """
    with open(PYSCX_ROOT / "pyproject.toml", "rb") as fh:
        dev = tomllib.load(fh)["project"]["optional-dependencies"]["dev"]
    for dist in ("scanpy", "scikit-misc", "pydeseq2"):
        assert any(
            re.match(rf"{re.escape(dist)}\b", spec) for spec in dev
        ), f"the dev extra does not install {dist!r}, which the suite imports"


# ---------------------------------------------------------------------------
# 4. The base install works
# ---------------------------------------------------------------------------

BASE = ("scanpy", "skmisc", "pydeseq2", "formulaic", "polars", "mudata")


def test_base_install_round_trips_a_file(tmp_path):
    a = _adata()
    p = str(tmp_path / "rt.scx")
    with blocked(*BASE):
        pyscx.from_anndata(a, p)
        exp = pyscx.open(p)
        obs = exp.read_obs()
        assert len(obs) == a.n_obs
        back = exp.to_anndata()
        assert back.shape == a.shape


def test_base_install_runs_pseudobulk_dex(adata):
    """The flagship pseudobulk surface, on defaults, with nothing extra installed."""
    with blocked(*BASE):
        df = pyscx.accel.pseudobulk_dex(
            adata,
            groupby=["cond", "donor"],
            test_col="cond",
            reference="ctrl",
            min_cells_per_group=1,
        )
    assert list(df.columns) == [
        "gene",
        "baseMean",
        "log2FoldChange",
        "lfcSE",
        "stat",
        "pvalue",
        "padj",
        "target",
        "reference",
    ]
    assert adata.uns["scx_accel"]["pseudobulk_dex"]["route"] == "cpu_nb_glm"


def test_base_install_runs_the_de_and_pca_surface(adata):
    with blocked(*BASE):
        pyscx.accel.rank_genes_groups(adata, groupby="cond")
        pyscx.accel.pdex_ref(adata, groupby="cond", reference="ctrl")
        pyscx.accel.pca(adata, n_comps=5)
    assert adata.obsm["X_pca"].shape == (adata.n_obs, 5)


def test_base_install_computes_qc_metrics_on_any_x(adata, scx_path):
    """`calculate_qc_metrics` no longer needs scanpy on *any* matrix.

    It used to be in `_OPTIONAL_PATHS`: a scipy/dense `X` was handed to
    `sc.pp.calculate_qc_metrics`, so the op raised on a base install unless the
    caller kept `X` backed. It now runs one native kernel whatever `X` is, which
    is why the `scanpy` extra no longer lists it.
    """
    import scipy.sparse as sp

    with blocked(*BASE):
        assert sp.issparse(adata.X), "premise: the in-memory arm, not the backed one"
        pyscx.accel.calculate_qc_metrics(adata)
        assert "total_counts" in adata.obs
        assert "mean_counts" in adata.var

        backed = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.calculate_qc_metrics(backed)
        assert set(backed.obs.columns) >= set(adata.obs.columns)


def test_base_install_preprocesses_a_backed_x(scx_path):
    """The scanpy-delegating ops are scanpy-free when `X` stays backed.

    This is the escape hatch the error messages point at, so it has to be true.
    """
    with blocked(*BASE):
        a = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.calculate_qc_metrics(a)
        pyscx.accel.filter_cells(a, min_genes=1)
        pyscx.accel.filter_genes(a, min_cells=1)
        pyscx.accel.normalize_total(a)
        pyscx.accel.log1p(a)
        assert "total_counts" in a.obs
        assert a.n_obs > 0 and a.n_vars > 0


# ---------------------------------------------------------------------------
# 5. The pseudobulk_dex default flip
# ---------------------------------------------------------------------------


def test_the_default_backend_is_nb_glm(adata):
    pyscx.accel.pseudobulk_dex(
        adata,
        groupby=["cond", "donor"],
        test_col="cond",
        reference="ctrl",
        min_cells_per_group=1,
    )
    assert adata.uns["scx_accel"]["pseudobulk_dex"]["route"] == "cpu_nb_glm"


def test_an_explicit_pydeseq2_backend_still_routes_by_layout(adata):
    pytest.importorskip("pydeseq2")
    pyscx.accel.pseudobulk_dex(
        adata,
        groupby=["cond", "donor"],
        test_col="cond",
        reference="ctrl",
        min_cells_per_group=1,
        backend="pydeseq2",
    )
    assert adata.uns["scx_accel"]["pseudobulk_dex"]["route"] == "cpu_csr"


def test_stratify_by_under_the_default_explains_the_flip(adata):
    """The guard fires for a caller who never typed `backend=`, so it has to say
    that the default changed and name the way back."""
    with pytest.raises(ValueError) as excinfo:
        pyscx.accel.pseudobulk_dex(
            adata,
            groupby=["cond", "donor"],
            test_col="cond",
            reference="ctrl",
            min_cells_per_group=1,
            stratify_by=["batch"],
            min_cells_per_stratum=1,
        )
    msg = str(excinfo.value)
    assert "nb_glm" in msg and "default" in msg
    assert 'backend="pydeseq2"' in msg, f"the way back is not named: {msg!r}"


def test_an_explicit_nb_glm_stratify_by_error_does_not_blame_the_default(adata):
    """Symmetric to the test above: someone who *asked* for nb_glm should not be
    told that the default changed."""
    with pytest.raises(ValueError) as excinfo:
        pyscx.accel.pseudobulk_dex(
            adata,
            groupby=["cond", "donor"],
            test_col="cond",
            reference="ctrl",
            min_cells_per_group=1,
            stratify_by=["batch"],
            min_cells_per_stratum=1,
            backend="nb_glm",
        )
    assert "default" not in str(excinfo.value)


@pytest.mark.parametrize(
    "arm,expect_warning",
    [
        ("default", True),
        ("explicit_nb_glm", False),
        ("pydeseq2_blocked", False),
    ],
)
def test_the_default_flip_warns_only_where_results_would_change(arm, expect_warning):
    """pydeseq2 installed → the numbers move under you, so say so.
    Explicitly asking for nb_glm, or not having pydeseq2 at all → nothing to say.

    One subprocess per arm, and that is load-bearing rather than tidiness: the
    warning is latched per process, so an in-process version of this test passes
    or fails on which other test ran first.
    """
    pytest.importorskip("pydeseq2")
    # Two substitution points, both at a known column: a top-level prelude and a
    # single call expression indented into the `with` block.
    _BLOCK_PYDESEQ2 = textwrap.dedent(
        """
        import sys
        class _NoPydeseq2:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] == "pydeseq2":
                    raise ModuleNotFoundError("no pydeseq2", name=name)
                return None
        for _k in [m for m in sys.modules if m.split(".")[0] == "pydeseq2"]:
            del sys.modules[_k]
        sys.meta_path.insert(0, _NoPydeseq2())
        """
    ).strip()
    prelude, call = {
        "default": ("", "pyscx.accel.pseudobulk_dex(mk(), **common)"),
        "explicit_nb_glm": (
            "",
            'pyscx.accel.pseudobulk_dex(mk(), backend="nb_glm", **common)',
        ),
        "pydeseq2_blocked": (
            _BLOCK_PYDESEQ2,
            "pyscx.accel.pseudobulk_dex(mk(), **common)",
        ),
    }[arm]
    script = (
        textwrap.dedent(
            """
            import warnings
            import numpy as np, scipy.sparse as sp, anndata as ad, pyscx
            rng = np.random.default_rng(0)
            def mk():
                a = ad.AnnData(sp.csr_matrix(rng.poisson(2.0, size=(60, 20)).astype(np.float32)))
                a.obs["cond"] = ["ctrl"] * 30 + ["trt"] * 30
                a.obs["donor"] = [f"d{i % 3}" for i in range(60)]
                return a
            common = dict(groupby=["cond", "donor"], test_col="cond",
                          reference="ctrl", min_cells_per_group=1)
            """
        )
        + prelude
        + "\nwith warnings.catch_warnings(record=True) as caught:\n"
        + "    warnings.simplefilter('always')\n"
        + f"    {call}\n"
        + 'hits = [w for w in caught if "defaults to backend" in str(w.message)]\n'
        + 'print(f"WARNED={bool(hits)}")\n'
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr[-1500:]
    assert f"WARNED={expect_warning}" in out.stdout, out.stdout


def test_the_transition_warning_cannot_fail_the_call(adata):
    """`warnings.warn` *raises* under `-W error`, which pytest configs and CI
    setups commonly set. A courtesy notice about a changed default must not be
    able to abort the caller's DE run — so every step of the warn path is
    ignored on failure rather than `?`-propagated.

    Run in a subprocess because `-W error` has to be in force at interpreter
    level for `simplefilter("error")` to reproduce the original failure faithfully.
    """
    pytest.importorskip("pydeseq2")
    script = textwrap.dedent(
        """
        import warnings, importlib.util
        # Belt and braces with the importorskip above: the warn path is gated on
        # pydeseq2 being importable, so without it this test would pass having
        # exercised nothing at all. Assert the precondition rather than assume it.
        assert importlib.util.find_spec("pydeseq2") is not None, \\
            "pydeseq2 absent - the warn path cannot fire, so this proves nothing"
        warnings.simplefilter("error")
        import numpy as np, scipy.sparse as sp, anndata as ad, pyscx
        rng = np.random.default_rng(0)
        a = ad.AnnData(sp.csr_matrix(rng.poisson(2.0, size=(60, 20)).astype(np.float32)))
        a.obs["cond"] = ["ctrl"] * 30 + ["trt"] * 30
        a.obs["donor"] = [f"d{i % 3}" for i in range(60)]
        df = pyscx.accel.pseudobulk_dex(
            a, groupby=["cond", "donor"], test_col="cond",
            reference="ctrl", min_cells_per_group=1,
        )
        assert len(df) > 0
        print("OK")
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, (
        f"pseudobulk_dex failed under warnings-as-errors:\n{out.stderr[-1500:]}"
    )
    assert "OK" in out.stdout


def test_the_transition_warning_fires_at_most_once_per_process(adata):
    """The docstring says "once per process", so it has to be a latch.

    Python's own dedup keys on the caller's module and line, and this very test
    sets `simplefilter("always")` — under which the warnings machinery alone
    would emit on every call.
    """
    pytest.importorskip("pydeseq2")
    script = textwrap.dedent(
        """
        import warnings
        import numpy as np, scipy.sparse as sp, anndata as ad, pyscx
        rng = np.random.default_rng(0)
        def mk():
            a = ad.AnnData(sp.csr_matrix(rng.poisson(2.0, size=(60, 20)).astype(np.float32)))
            a.obs["cond"] = ["ctrl"] * 30 + ["trt"] * 30
            a.obs["donor"] = [f"d{i % 3}" for i in range(60)]
            return a
        common = dict(groupby=["cond", "donor"], test_col="cond",
                      reference="ctrl", min_cells_per_group=1)
        seen = 0
        for _ in range(3):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pyscx.accel.pseudobulk_dex(mk(), **common)
            seen += len([w for w in caught if "defaults to backend" in str(w.message)])
        print(f"COUNT={seen}")
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr[-1500:]
    assert "COUNT=1" in out.stdout, f"expected exactly one warning, got: {out.stdout!r}"


def test_the_csc_layout_is_visible_under_the_default_backend(tmp_path):
    """`prefer_format="csc"` still reads the sidecar under nb_glm — the route
    string alone no longer says so, so `csc_available` has to."""
    a = _adata(n_obs=200, n_vars=60)
    p = str(tmp_path / "csc.scx")
    pyscx.from_anndata(a, p, csc="always")
    backed = pyscx.open(p).to_anndata(backed=True)
    backed.obs["cond"] = a.obs["cond"].values
    backed.obs["donor"] = a.obs["donor"].values
    pyscx.accel.pseudobulk_dex(
        backed,
        groupby=["cond", "donor"],
        test_col="cond",
        reference="ctrl",
        min_cells_per_group=1,
        prefer_format="csc",
        gene_indices=list(range(30)),
    )
    info = backed.uns["scx_accel"]["pseudobulk_dex"]
    assert info["route"] == "cpu_nb_glm"
    assert info["csc_available"] is True
