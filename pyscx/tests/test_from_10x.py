"""Tests for `pyscx.from_10x` — focused on the scanpy import-guard error path."""

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
