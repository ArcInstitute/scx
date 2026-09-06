"""`Experiment.to_anndata`'s signature exists in four hand-maintained copies.

The pyo3 `#[pyo3(signature = ...)]`, the `.pyi` stub, the kwarg list in
`docs/api.md`, and a second copy in `skills/scx-usage/reference/processing.md`
(`to_gpu_anndata` has three of the four — the usage skill does not cover it).
Nothing checked them against each other, and by the time this test was written
the skills copy had already lost four kwargs (`container`, `data_dtype`,
`index_dtype`, `allow_lossy`) without anything going red.

`test_accel_stub_coverage.py` does the equivalent job for `accel.pyi`, but it
iterates `tree.body` for top-level `ast.FunctionDef` — class methods are
structurally out of its reach, so `Experiment` was never covered.

Two checks, for two different silent failures:

* **stub vs runtime** — names *and order*. Order matters because every
  parameter here is positional-or-keyword (there is no `*` separator on either
  side), so a kwarg inserted mid-signature rebinds every positional call.
* **docs vs runtime** — a name-presence check on the one line that carries the
  signature. Deliberately not a parse: the doc line is prose-adjacent and a
  strict parser would be brittle, while "every parameter is mentioned" is
  exactly the property that went stale.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from _stub_ast import stub_class_body  # noqa: E402

_STUB_CHECKED = ("to_anndata", "to_gpu_anndata")

# Which markdown copies carry a signature for which method. `processing.md`
# summarises `Experiment` for the usage skill and never mentions
# `to_gpu_anndata`, so listing it there would fail on an absence that is not
# drift.
_DOC_SIGNATURES = (
    ("docs/api.md", "to_anndata"),
    ("docs/api.md", "to_gpu_anndata"),
    ("skills/scx-usage/reference/processing.md", "to_anndata"),
)


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "Cargo.toml").exists() and (parent / "docs").is_dir():
            return parent
    raise AssertionError("could not locate the repo root from the test file")


def _runtime_params(method_name: str) -> list[str]:
    """Parameter names, in order, from the pyo3 `__text_signature__`."""
    import pyscx

    sig = getattr(pyscx.Experiment, method_name).__text_signature__
    assert sig is not None, f"Experiment.{method_name} has no __text_signature__"
    inner = sig[sig.index("(") + 1 : sig.rindex(")")]
    names = []
    # Split on top-level commas only -- `container="csr"` has no nesting, but a
    # default like `(1, 2)` would, and a naive split would mint a phantom name.
    depth, current = 0, ""
    for ch in inner:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            names.append(current)
            current = ""
        else:
            current += ch
    names.append(current)
    out = []
    for raw in names:
        name = raw.strip().split("=")[0].strip()
        if not name or name in ("$self", "/", "*"):
            continue
        out.append(name)
    return out


def _stub_params(method_name: str) -> list[str]:
    """Parameter names, in order, from the `Experiment` class body in the stub."""
    body = stub_class_body("Experiment")
    match = re.search(rf"^    def {method_name}\(", body, re.MULTILINE)
    assert match, f"__init__.pyi declares no Experiment.{method_name}"
    # Take from the `def` to the closing `) -> ...:` and re-parse it standalone.
    tail = body[match.start() :]
    block = tail[: tail.index("\n    ) ->") + len("\n    )")]
    src = "\n".join(line[4:] for line in block.splitlines()) + ": ..."
    tree = ast.parse(src)
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    args = [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]
    return [a for a in args if a != "self"]


@pytest.mark.parametrize("method_name", _STUB_CHECKED)
def test_experiment_stub_signature_matches_runtime(method_name):
    runtime = _runtime_params(method_name)
    stubbed = _stub_params(method_name)

    assert stubbed == runtime, (
        f"__init__.pyi's Experiment.{method_name} signature has drifted from the "
        f"pyo3 one.\n  stub:    {stubbed}\n  runtime: {runtime}\n"
        "Names and order must both match: every parameter is "
        "positional-or-keyword, so an inserted kwarg rebinds positional callers."
    )


@pytest.mark.parametrize("doc_path, method_name", _DOC_SIGNATURES)
def test_doc_signature_copies_name_every_kwarg(doc_path, method_name):
    root = _repo_root()
    text = (root / doc_path).read_text()
    match = re.search(rf"^- `{method_name}\((.*?)\)`", text, re.MULTILINE | re.DOTALL)
    assert match, f"{doc_path} carries no `{method_name}(...)` signature line"
    signature = match.group(1)

    missing = [p for p in _runtime_params(method_name) if p not in signature]
    assert not missing, (
        f"{doc_path}'s `{method_name}(...)` signature omits {missing}. "
        "It is a hand-maintained copy of the pyo3 signature; update it in the "
        "same commit as the Rust change."
    )
