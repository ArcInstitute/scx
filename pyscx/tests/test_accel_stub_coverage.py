"""Doc-drift guard for ``pyscx/python/pyscx/accel.pyi`` (code-review item 10).

``accel.pyi`` carries explicit ``def`` stubs for the user-facing accelerator
API plus a ``__getattr__`` fallback for everything else. These tests pin two
invariants so the stub file does not silently drift from the Rust module:

* every explicit stub maps to a real runtime function (no dead stubs left
  behind when a function is renamed/removed), and
* the user-facing accelerators — including the ones added to close the review
  "Bonus" finding (neighbors / umap / leiden / harmony_integrate /
  compute_lisi) — keep their explicit stubs.

Pure import + text check; no GPU required.
"""

from __future__ import annotations

import pathlib
import re

import pytest

import pyscx
import pyscx.accel as accel


def _stub_names() -> set[str]:
    pyi = pathlib.Path(pyscx.__file__).parent / "accel.pyi"
    text = pyi.read_text()
    # Exclude dunders such as the `__getattr__` fallback.
    return {n for n in re.findall(r"^def (\w+)\(", text, re.MULTILINE) if not n.startswith("_")}


def _runtime_functions() -> set[str]:
    return {
        name
        for name in dir(accel)
        if not name.startswith("_")
        and name[0].islower()
        and callable(getattr(accel, name))
    }


def test_no_dead_stubs_in_accel_pyi():
    """Every explicit stub must correspond to a real runtime function."""
    dead = sorted(_stub_names() - _runtime_functions())
    assert not dead, f"accel.pyi has stubs for functions that no longer exist: {dead}"


def test_user_facing_accelerators_have_stubs():
    """User-facing accelerators must keep explicit type stubs (Bonus finding)."""
    required = {
        # Bonus finding: these were missing and were added in #178.
        "neighbors",
        "umap",
        "leiden",
        "harmony_integrate",
        "compute_lisi",
        # Core accelerators that were already stubbed — guard against regression.
        "pca",
        "rank_genes_groups",
        "highly_variable_genes",
        "pdex_ref",
        # F6: gained a public `output=` kwarg, so it needs a typed surface —
        # pinned here so the stub cannot be dropped again.
        "pdex_nb_glm",
        "rank_genes_groups_df",
        # F8: carries the `sample_cols=` / `sample_key=` aliases and the
        # rank_genes_groups role contrast in its docstring — the stub is where a
        # user's editor surfaces both, so it must not be dropped.
        "pseudobulk_dex",
    }
    missing = sorted(required - _stub_names())
    assert not missing, f"user-facing accelerators missing type stubs in accel.pyi: {missing}"
    # And they must actually exist at runtime.
    runtime = _runtime_functions()
    absent = sorted(required - runtime)
    assert not absent, f"declared accelerators absent from pyscx.accel at runtime: {absent}"
