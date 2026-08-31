"""Drift guard for the six `scx-loader` pyclasses exposed through `pyscx`.

`pyscx` ships `py.typed`, so a public method missing from the stub is a
mypy/Pyright error for anyone following the docs — while the runtime works
fine, which is what makes the drift invisible. `close()` / `closed` reached
`__init__.pyi` only because a reviewer noticed; this makes the next one fail
a test instead.

ORG-9.10-4 widened the runtime→stub direction from the two plan-driven classes
to all four dataset classes. The two training ones were outside it, which is how
`MultimodalTrainingDataset` came to have no `__init__.pyi` entry **at all** —
not a missing method, the whole class — with a green suite the entire time.

Phase 9f (`ORG-9.10-2`) moved these guards out of `test_accel_stub_coverage.py`,
where they had been living, and closed the two gaps that made them unable to see
a module split going wrong:

1. **The stub→runtime direction was checked for `close`/`closed` only.** Every
   other method could vanish from the Rust and the suite stayed green: the
   runtime→stub test compares the *runtime* set against the stub, so a method
   that disappears from both the runtime and nothing else simply shrinks the set
   being checked. Dropping a `#[pymethods]` fn while moving it between files is
   the single most likely way to break a pure-move refactor, and nothing saw it.
2. **The two iterator pyclasses were not covered at all.** `IndexPlanBatchIter`
   and `SparseCellSetBatchIter` are never `add_class`-registered — they exist
   only as the return type of `iter_with_plans` — so they cannot be reached by
   name from `pyscx` and every by-name guard skipped them. `metrics()` (9c) and
   `cache_metrics()` live there.
"""

from __future__ import annotations

import pathlib
import re

import pytest

import pyscx

# Registered with `add_class`, so reachable as `pyscx.<name>`.
_LIFECYCLE_CLASSES = [
    "IndexPlanDataset",
    "SparseCellSetDataset",
    "TrainingDataset",
    "MultimodalTrainingDataset",
]

# NOT registered: returned from `iter_with_plans`, so the only way to their type
# object is through an instance. `scx-loader/src/lib.rs` re-exports
# `SparseCellSetBatchIter` but `pyscx` never calls `add_class` on it, and
# `IndexPlanBatchIter` is not even re-exported — an asymmetry that is deliberate
# and pinned by `test_iterator_classes_are_not_module_level_names`.
_ITERATOR_CLASSES = [
    "IndexPlanBatchIter",
    "SparseCellSetBatchIter",
]

_ALL_CLASSES = _LIFECYCLE_CLASSES + _ITERATOR_CLASSES


def _pyi_text() -> str:
    return (pathlib.Path(pyscx.__file__).parent / "__init__.pyi").read_text()


def _class_stub_body(text: str, cls: str) -> str:
    """The stub text between `class <cls>:` and the next top-level statement."""
    start = re.search(rf"^class {cls}\b.*?:$", text, re.MULTILINE)
    assert start, f"{cls} has no stub in __init__.pyi at all"
    rest = text[start.end() :]
    end = re.search(r"^\S", rest, re.MULTILINE)
    return rest[: end.start()] if end else rest


def _stubbed_defs(cls: str) -> set[str]:
    """Names the stub declares for `cls`, methods and properties alike.

    Properties are `@property def name(...)` in the stub, so one `def` scan
    catches both. `__init__` is excluded: it is `__new__` on a pyo3 class and
    its keywords have their own guard below.
    """
    body = _class_stub_body(_pyi_text(), cls)
    return {n for n in re.findall(r"^\s*def (\w+)\(", body, re.MULTILINE) if n != "__init__"}


@pytest.fixture
def runtime_classes(tmp_path, synthetic_adata):
    """The type object for all six classes, iterators reached via an instance.

    The owning datasets are kept alive in the returned mapping: dropping an
    `IndexPlanDataset` shuts its loader down, and the iterator borrows it.
    """
    path = str(tmp_path / "loader_surface.scx")
    pyscx.from_anndata(synthetic_adata, path)

    classes = {c: getattr(pyscx, c) for c in _LIFECYCLE_CLASSES}

    index_plan_ds = pyscx.IndexPlanDataset(path)
    index_plan_iter = index_plan_ds.iter_with_plans(iter([[(0, 1)]]))
    cellset_ds = pyscx.SparseCellSetDataset([path])
    cellset_iter = cellset_ds.iter_with_plans(iter([([0], [0], [0], [0, 1])]))

    # Prefer the module-level name; fall back to the live instance. The iterators
    # are not `add_class`-registered today, so the fallback is what runs — but if
    # they ever are, this keeps working instead of needing an edit.
    for name, instance in (
        ("IndexPlanBatchIter", index_plan_iter),
        ("SparseCellSetBatchIter", cellset_iter),
    ):
        classes[name] = getattr(pyscx, name, None) or type(instance)

    # Keep the owners (and the live iterators) referenced for the test's duration.
    classes["_keepalive"] = (index_plan_ds, index_plan_iter, cellset_ds, cellset_iter)
    return classes


# ---------------------------------------------------------------------------
# stub → runtime. The direction that sees a `#[pymethods]` fn get dropped.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", _ALL_CLASSES)
def test_stubbed_members_exist_at_runtime(cls, runtime_classes):
    """Every method/property the stub declares is really there.

    This is the guard a module split needs. The runtime→stub test below cannot
    do it: it derives its expectation *from the runtime*, so a method deleted in
    the Rust is simply absent from both sides and passes. Here the stub is the
    fixed expectation, so losing a method during a move is a failure.
    """
    obj = runtime_classes[cls]
    declared = _stubbed_defs(cls)
    assert declared, f"parsed no method stubs out of {cls}'s __init__.pyi block"
    missing = sorted(n for n in declared if not hasattr(obj, n))
    assert not missing, (
        f"{cls} declares {missing} in __init__.pyi but they are absent at "
        f"runtime — a #[pymethods] fn was dropped or renamed"
    )


@pytest.mark.parametrize("cls", _ITERATOR_CLASSES)
def test_iterator_public_methods_are_all_in_the_init_stub(cls, runtime_classes):
    """Runtime→stub for the two unregistered iterator classes.

    The by-name guard cannot reach these, so before Phase 9f they had no
    coverage in either direction.
    """
    body = _class_stub_body(_pyi_text(), cls)
    runtime = {
        n for n in dir(runtime_classes[cls]) if not n.startswith("_") and n not in {"mro"}
    }
    missing = sorted(n for n in runtime if not re.search(rf"\bdef {n}\b|\b{n}\s*:", body))
    assert not missing, (
        f"{cls} exposes {missing} at runtime but they are absent from "
        f"__init__.pyi — py.typed users get missing-attribute errors"
    )


def test_loader_free_functions_are_importable():
    """The `#[pyfunction]` half of `scx-loader`'s re-export list.

    `scx-loader/src/lib.rs` re-exports these three and `pyscx/src/lib.rs`
    registers them; a split that loses one from the `pub use` fails to compile,
    but one that loses the `add_function` call does not.
    """
    for name in (
        "collate_cellset_gathered",
        "downsample_counts_csr",
        "downsample_file_identity",
    ):
        assert callable(getattr(pyscx, name, None)), f"pyscx.{name} is not registered"
# ---------------------------------------------------------------------------
# runtime → stub, and the constructor keywords. Moved here verbatim from
# test_accel_stub_coverage.py by Phase 9f; unchanged except for reusing the
# module-level _pyi_text() helper instead of re-reading the file inline.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", _LIFECYCLE_CLASSES)
def test_public_methods_are_all_in_the_init_stub(cls):
    """Every public runtime attribute of the class appears in its stub block."""
    body = _class_stub_body(_pyi_text(), cls)

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
    """`close` / `closed` exist, independently of what the stub says.

    NOT subsumed by `test_stubbed_members_exist_at_runtime`, though it looks it:
    that test takes its expectation *from the stub*, so deleting `close` from
    `__init__.pyi` and from the Rust in one change leaves it with nothing to
    check and it passes. Verified — removing the `close` stub drops it from
    `_stubbed_defs`, and the parameterised test goes green without it.

    This one hard-codes the two names, so it is the only thing standing between a
    co-ordinated stub+runtime removal and a green suite. Two reviewers have called
    it redundant; it is here on purpose.
    """
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
    body = _class_stub_body(_pyi_text(), cls)

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
