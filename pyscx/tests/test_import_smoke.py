"""N3-2026-05-21-Tier2: `import pyscx` must work on a fresh `maturin
develop` build, AND the four hdf5-only entry points must degrade
gracefully when the build was made with `--no-default-features` (so
the native module lacks `from_h5ad` etc.).

Before PR N3: a bare `maturin develop` produced a venv where the very
first `import pyscx` raised ImportError because `__init__.py` did an
unconditional `from .pyscx import from_h5ad as _from_h5ad_native` but
the Rust side only registers that symbol under `#[cfg(feature = "hdf5")]`.

After PR N3: the default features include `hdf5` so the common case
has all four wrappers callable, AND the import is wrapped in a
try/except so users who explicitly opt out of hdf5 still get a working
`import pyscx`. The wrappers then raise `NotImplementedError` only when
actually called.
"""
import pytest


def test_default_build_exposes_hdf5_entrypoints():
    """The `maturin develop` default-features recipe (quoted verbatim in
    CLAUDE.md § Build and Test and SKILL Tier 1) MUST produce a venv
    where the four hdf5-only wrappers are callable. This test runs
    against whatever the current .venv was built with — if it fails,
    the venv was built with `--no-default-features` or the pyproject
    default-features list regressed."""
    import pyscx
    for name in ("from_h5ad", "to_h5ad", "from_h5mu", "to_h5mu"):
        fn = getattr(pyscx, name, None)
        assert callable(fn), (
            f"pyscx.{name} should be callable on a default `maturin develop` "
            f"build; got {fn!r}. If you ran `--no-default-features`, the "
            "wrapper still exists (it should raise NotImplementedError "
            "when called) — but this test asserts the default-build path."
        )
    assert pyscx._HAS_HDF5 is True, (
        "Default `maturin develop` should land in the _HAS_HDF5=True "
        "branch; got False — pyproject.toml [tool.maturin] features may "
        "have regressed."
    )


def _patch_no_hdf5(monkeypatch):
    """Force the wrappers into the no-hdf5 branch without rebuilding the
    extension. Mirrors the state a user would see after `maturin develop
    --no-default-features --features pyo3/extension-module`."""
    import pyscx
    monkeypatch.setattr(pyscx, "_HAS_HDF5", False)
    monkeypatch.setattr(pyscx, "_from_h5ad_native", None)
    monkeypatch.setattr(pyscx, "_to_h5ad_native", None)
    monkeypatch.setattr(pyscx, "_from_h5mu_native", None)
    monkeypatch.setattr(pyscx, "_to_h5mu_native", None)


@pytest.mark.parametrize("fn_name", ["from_h5ad", "to_h5ad", "from_h5mu", "to_h5mu"])
def test_wrappers_raise_when_hdf5_unavailable(monkeypatch, fn_name):
    """Each wrapper raises NotImplementedError naming `hdf5` when the
    native symbol is absent. The message must point at the rebuild
    command so the user can self-recover without grepping source."""
    _patch_no_hdf5(monkeypatch)
    import pyscx
    fn = getattr(pyscx, fn_name)
    with pytest.raises(NotImplementedError) as excinfo:
        fn("input.h5ad", "output.scx")
    msg = str(excinfo.value)
    assert "hdf5" in msg, f"Expected 'hdf5' in error message; got: {msg!r}"
    assert f"pyscx.{fn_name}" in msg, (
        f"Error message should name the function; got: {msg!r}"
    )
    assert "maturin develop" in msg, (
        f"Error message should name the rebuild command; got: {msg!r}"
    )


def test_import_pyscx_succeeds_with_no_hdf5_natives(monkeypatch):
    """Even with the native hdf5 symbols absent, the module-level
    `import pyscx` plus a `reload` should not raise. (`monkeypatch`
    can't fully simulate a fresh import because the module is already
    loaded, but the try/except in __init__.py is what makes the
    no-hdf5 path importable in the first place — this test asserts
    the in-process invariants that follow from it.)"""
    _patch_no_hdf5(monkeypatch)
    import pyscx
    # Wrappers must still be callable (raising at call-time is OK).
    for name in ("from_h5ad", "to_h5ad", "from_h5mu", "to_h5mu"):
        assert callable(getattr(pyscx, name))
    # Non-hdf5 surface must still work.
    assert callable(pyscx.open)
    assert callable(pyscx.validate)
