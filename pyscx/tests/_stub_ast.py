"""AST access to `pyscx/python/pyscx/__init__.pyi`.

Shared by `test_backed.py` (which checks that handle classes declare the
special methods they implement) and `test_experiment_stub_coverage.py` (which
checks `Experiment`'s signatures against the runtime). One parse, one notion of
what a class body contains — the previous version found `"class <name>:"` with
`str.index` and cut the body at the next top-level `class `/`def `, which coupled
both tests to the stub's exact whitespace and would break on a reformat.
"""

from __future__ import annotations

import ast
import functools
import pathlib


@functools.cache
def _stub_module() -> ast.Module:
    import pyscx

    pyi = pathlib.Path(pyscx.__file__).with_name("__init__.pyi")
    return ast.parse(pyi.read_text())


def stub_class_methods(cls_name: str) -> dict[str, ast.FunctionDef]:
    """Every `def` declared directly in `class <cls_name>:`, by name.

    Raises if the stub declares no such class — an absent class is a stub bug,
    not an empty result.
    """
    for node in _stub_module().body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            return {
                fn.name: fn
                for fn in node.body
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"__init__.pyi declares no class {cls_name}")


def stub_method_params(cls_name: str, method_name: str) -> list[str]:
    """Parameter names of one stub method, in declaration order, minus `self`."""
    methods = stub_class_methods(cls_name)
    fn = methods.get(method_name)
    assert fn is not None, (
        f"__init__.pyi declares no {cls_name}.{method_name} "
        f"(it has: {sorted(methods)})"
    )
    args = fn.args
    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    return [n for n in names if n != "self"]
