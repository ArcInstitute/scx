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


# ---------------------------------------------------------------------------
# Drift guard for the plan-driven dataset classes in `__init__.pyi`.
#
# `pyscx` ships `py.typed`, so a public method missing from the stub is a
# mypy/Pyright error for anyone following the docs — while the runtime works
# fine, which is what makes the drift invisible. `close()` / `closed` reached
# `__init__.pyi` only because a reviewer noticed; this makes the next one fail
# a test instead.
#
# ORG-9.10-4 widened this from the two plan-driven classes to all four dataset
# classes. The two training ones were outside it, which is how
# `MultimodalTrainingDataset` came to have no `__init__.pyi` entry **at all** —
# not a missing method, the whole class — with a green suite the entire time.
# ---------------------------------------------------------------------------

_LIFECYCLE_CLASSES = [
    "IndexPlanDataset",
    "SparseCellSetDataset",
    "TrainingDataset",
    "MultimodalTrainingDataset",
]


def _class_stub_body(text: str, cls: str) -> str:
    """The stub text between `class <cls>:` and the next top-level statement."""
    start = re.search(rf"^class {cls}\b.*?:$", text, re.MULTILINE)
    assert start, f"{cls} has no stub in __init__.pyi at all"
    rest = text[start.end() :]
    end = re.search(r"^\S", rest, re.MULTILINE)
    return rest[: end.start()] if end else rest


@pytest.mark.parametrize("cls", _LIFECYCLE_CLASSES)
def test_public_methods_are_all_in_the_init_stub(cls):
    """Every public runtime attribute of the class appears in its stub block."""
    pyi = (pathlib.Path(pyscx.__file__).parent / "__init__.pyi").read_text()
    body = _class_stub_body(pyi, cls)

    runtime = {
        n
        for n in dir(getattr(pyscx, cls))
        if not n.startswith("_") and n not in {"mro"}
    }
    missing = sorted(n for n in runtime if not re.search(rf"\bdef {n}\b|\b{n}\s*:", body))
    assert not missing, (
        f"{cls} exposes {missing} at runtime but they are absent from "
        f"__init__.pyi — py.typed users get missing-attribute errors"
    )


@pytest.mark.parametrize("cls", _LIFECYCLE_CLASSES)
def test_lifecycle_members_exist_at_runtime(cls):
    """The other direction: no dead stubs for a lifecycle API that got removed."""
    obj = getattr(pyscx, cls)
    for member in ("close", "closed"):
        assert hasattr(obj, member), f"{cls}.{member} is stubbed but missing at runtime"


@pytest.mark.parametrize("cls", _LIFECYCLE_CLASSES)
def test_constructor_kwargs_are_all_in_the_init_stub(cls):
    """Every constructor keyword appears in the class's `__init__` stub.

    The attribute guard above cannot see this: a kwarg is not an attribute, so
    adding one to the pyo3 signature and forgetting the stub is invisible to
    every test — and silently wrong for `py.typed` users, who get "unexpected
    keyword argument" from the type checker on code the runtime accepts. That is
    exactly how `SparseCellSetDataset` lost `scatter_block_index` from its stub
    when the kwarg itself was dropped in the sidecar removal, leaving two docs
    describing a knob no signature had.

    pyo3 exposes the keywords in the class's `text_signature`; there is no
    `inspect.signature` for a native `__new__`.
    """
    pyi = (pathlib.Path(pyscx.__file__).parent / "__init__.pyi").read_text()
    body = _class_stub_body(pyi, cls)

    sig = getattr(pyscx, cls).__text_signature__
    assert sig, f"{cls} has no __text_signature__ to check against"
    kwargs = [
        tok.split("=", 1)[0].strip()
        for tok in sig.strip("()").split(",")
        if "=" in tok
    ]
    assert kwargs, f"parsed no keywords out of {cls}.__text_signature__ = {sig!r}"
    missing = sorted(k for k in kwargs if not re.search(rf"^\s*{re.escape(k)}\s*:", body, re.M))
    assert not missing, (
        f"{cls}.__init__ accepts {missing} at runtime but they are absent from "
        f"its __init__.pyi stub"
    )
