"""Tests for `pyscx.from_10x` — focused on the scanpy import-guard error path."""

import importlib.machinery
import sys

import pytest


def test_from_10x_missing_scanpy_raises_with_install_hint(monkeypatch):
    """When scanpy is not importable, from_10x raises a clear install hint."""
    import pyscx

    # Force `import scanpy` to fail inside from_10x by shadowing the
    # module with None in sys.modules (CPython's import machinery treats
    # this as "this module is known to be unavailable").
    monkeypatch.setitem(sys.modules, "scanpy", None)

    with pytest.raises(ModuleNotFoundError, match=r"pyscx\[10x\]"):
        pyscx.from_10x("nonexistent.h5", "nonexistent.scx")


def test_from_10x_transitive_import_error_propagates(monkeypatch):
    """If scanpy is installed but a transitive dep is missing, surface that.

    The install-hint rewrite must trigger only when `scanpy` itself is the
    missing module — not when scanpy's own import chain fails on a different
    package (e.g. igraph, pynndescent). Otherwise users get a misleading
    "install pyscx[10x]" message that hides the real broken dependency.
    """
    import pyscx

    class _Loader:
        @staticmethod
        def create_module(_spec):
            return None

        @staticmethod
        def exec_module(_module):
            raise ModuleNotFoundError("No module named 'igraph'", name="igraph")

    class _Finder:
        @staticmethod
        def find_spec(name, _path=None, _target=None):
            if name == "scanpy":
                return importlib.machinery.ModuleSpec("scanpy", _Loader())
            return None

    monkeypatch.delitem(sys.modules, "scanpy", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Finder(), *sys.meta_path])

    with pytest.raises(ModuleNotFoundError) as excinfo:
        pyscx.from_10x("nonexistent.h5", "nonexistent.scx")
    # The original transitive error must propagate — not the pyscx[10x] hint.
    assert "pyscx[10x]" not in str(excinfo.value)
    assert "igraph" in str(excinfo.value)
