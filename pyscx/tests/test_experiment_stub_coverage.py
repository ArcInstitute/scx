"""`Experiment.to_anndata`'s signature exists in five hand-maintained copies.

The pyo3 `#[pyo3(signature = ...)]`, the `.pyi` stub, the kwarg list in
`docs/api.md`, the fenced "Full signature" block in `docs/scanpy.md`, and a
second bulleted copy in `skills/scx-usage/reference/processing.md`. Nothing
checked them against each other, and by the time this test was written two had
already drifted: the skills copy had lost `container` / `data_dtype` /
`index_dtype` / `allow_lossy`, and the scanpy.md block had never gained
`preserve_slots` / `modality` / the four dtype kwargs.

`test_accel_stub_coverage.py` does the equivalent job for `accel.pyi`, but it
iterates `tree.body` for top-level `ast.FunctionDef` — class methods are
structurally out of its reach, so `Experiment` was never covered.

Two checks, for two different silent failures:

* **stub vs runtime** — names *and order*. Order matters because every
  parameter here is positional-or-keyword (there is no `*` separator on either
  side), so a kwarg inserted mid-signature rebinds every positional call.
* **docs vs runtime** — a name-presence check over the block that carries the
  signature. Deliberately not a parse: the doc copies are prose- and
  comment-laden and a strict parser would be brittle, while "every parameter is
  mentioned" is exactly the property that went stale.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from _stub_ast import stub_method_params  # noqa: E402
from test_docstring_coverage import _kwargs_from_text_signature  # noqa: E402

_STUB_CHECKED = ("to_anndata", "to_gpu_anndata")

# (file, method, regex capturing the signature text). Three shapes, because the
# copies are written three ways: a markdown bullet in api.md and processing.md,
# and a fenced python call in scanpy.md. `processing.md` summarises `Experiment`
# for the usage skill and never mentions `to_gpu_anndata`, so listing it there
# would fail on an absence that is not drift.
_DOC_SIGNATURES = (
    ("docs/api.md", "to_anndata", r"^- `to_anndata\((.*?)\)`"),
    ("docs/api.md", "to_gpu_anndata", r"^- `to_gpu_anndata\((.*?)\)`"),
    ("docs/scanpy.md", "to_anndata", r"^exp\.to_anndata\(\n(.*?)^\)"),
    (
        "skills/scx-usage/reference/processing.md",
        "to_anndata",
        r"^- `to_anndata\((.*?)\)`",
    ),
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
    return _kwargs_from_text_signature(sig)


@pytest.mark.parametrize("method_name", _STUB_CHECKED)
def test_experiment_stub_signature_matches_runtime(method_name):
    runtime = _runtime_params(method_name)
    stubbed = stub_method_params("Experiment", method_name)

    assert stubbed == runtime, (
        f"__init__.pyi's Experiment.{method_name} signature has drifted from the "
        f"pyo3 one.\n  stub:    {stubbed}\n  runtime: {runtime}\n"
        "Names and order must both match: every parameter is "
        "positional-or-keyword, so an inserted kwarg rebinds positional callers."
    )


@pytest.mark.parametrize("doc_path, method_name, pattern", _DOC_SIGNATURES)
def test_doc_signature_copies_name_every_kwarg(doc_path, method_name, pattern):
    root = _repo_root()
    text = (root / doc_path).read_text()
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    assert match, f"{doc_path} carries no `{method_name}(...)` signature block"
    signature = match.group(1)

    missing = [p for p in _runtime_params(method_name) if p not in signature]
    assert not missing, (
        f"{doc_path}'s `{method_name}(...)` signature omits {missing}. "
        "It is a hand-maintained copy of the pyo3 signature; update it in the "
        "same commit as the Rust change."
    )
