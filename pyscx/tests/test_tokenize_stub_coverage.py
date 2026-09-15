"""Doc-drift guard for ``pyscx/python/pyscx/tokenize.pyi``.

Modelled on ``test_accel_stub_coverage.py``, and load-bearing for the same
reason: ``test_loader_stub_coverage.py`` gates the loader surface by hard-coded
class name against ``__init__.pyi`` and is **blind to a submodule** in both
directions. Without this file the ``pyscx.tokenize`` namespace has no mechanical
gate at all — a kernel could be renamed, or gain a kwarg, with nothing going red.

Pure import + text/AST check; no fixtures, no GPU.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re

import pyscx
import pyscx.tokenize as tok


def _pyi_path() -> pathlib.Path:
    return pathlib.Path(pyscx.__file__).parent / "tokenize.pyi"


def _stub_names() -> set[str]:
    text = _pyi_path().read_text()
    # Exclude dunders such as the `__getattr__` fallback.
    return {
        n for n in re.findall(r"^def (\w+)\(", text, re.MULTILINE) if not n.startswith("_")
    }


def _runtime_functions() -> set[str]:
    return {
        name
        for name in dir(tok)
        if not name.startswith("_") and name[0].islower() and callable(getattr(tok, name))
    }


def test_no_dead_stubs_in_tokenize_pyi():
    dead = sorted(_stub_names() - _runtime_functions())
    assert not dead, f"tokenize.pyi has stubs for functions that no longer exist: {dead}"


def test_every_kernel_has_a_stub():
    missing = sorted(_runtime_functions() - _stub_names())
    assert not missing, f"pyscx.tokenize functions missing from tokenize.pyi: {missing}"


def test_the_four_named_kernels_are_present():
    """The kernels the contract enumerates, asserted by name.

    ``test_every_kernel_has_a_stub`` compares two sets derived from the same
    build, so it stays green if a kernel disappears from *both*. This one names
    them.
    """
    required = {"top_k", "rank_tokens", "bin_values", "sample_genes"}
    runtime = _runtime_functions()
    assert required <= runtime, f"missing kernels: {sorted(required - runtime)}"
    assert required <= _stub_names(), f"missing stubs: {sorted(required - _stub_names())}"


def _stub_parameters() -> dict[str, list[tuple[str, str]]]:
    tree = ast.parse(_pyi_path().read_text())
    out: dict[str, list[tuple[str, str]]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        a = node.args
        params: list[tuple[str, str]] = []
        params += [(p.arg, "POSITIONAL_ONLY") for p in a.posonlyargs]
        params += [(p.arg, "POSITIONAL_OR_KEYWORD") for p in a.args]
        if a.vararg:
            params.append((a.vararg.arg, "VAR_POSITIONAL"))
        params += [(p.arg, "KEYWORD_ONLY") for p in a.kwonlyargs]
        if a.kwarg:
            params.append((a.kwarg.arg, "VAR_KEYWORD"))
        out[node.name] = params
    return out


def test_tokenize_stub_signatures_match_runtime():
    """Names, order and keyword-only-ness, against `inspect.signature`.

    A kwarg added on the Rust side and not mirrored here is invisible in a
    user's editor, which is the drift the accel twin was written to catch.
    """
    drift: list[str] = []
    for name, stub_params in _stub_parameters().items():
        fn = getattr(tok, name, None)
        if fn is None:
            continue  # test_no_dead_stubs_in_tokenize_pyi reports it
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            drift.append(f"{name}: runtime signature unavailable")
            continue
        runtime = [(p.name, p.kind.name) for p in sig.parameters.values()]
        if runtime != stub_params:
            drift.append(f"{name}:\n    stub    {stub_params}\n    runtime {runtime}")
    assert not drift, "tokenize.pyi stubs drift from the runtime signatures:\n" + "\n".join(
        drift
    )


def test_contract_version_is_exported_and_stubbed():
    assert isinstance(tok.CONTRACT_VERSION, int)
    assert "CONTRACT_VERSION: int" in _pyi_path().read_text()
