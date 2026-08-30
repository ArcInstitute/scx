"""Doc-drift guard for the wrapped-function docstrings (ORG-10.16-4).

Each function in this table exists twice: a pure-Python wrapper in
`pyscx/python/pyscx/__init__.py` (what `help()` and Sphinx show — the wrapper
shadows the native after `from .pyscx import *`) and a pyo3 native in
`pyscx.pyscx`. The convention is **Python-canonical**: the wrapper carries the
one full docstring, and the native's `///` doc is a two-line pointer at it.
The wrapper is the right holder because its coercions are part of the
user-visible contract (`_coerce_path` accepts an open `Experiment`,
`_coerce_obs_mask` accepts a pandas Series, `_coerce_key` accepts a bare str)
— facts a docstring on the `&str`-typed native cannot state truthfully.

Three drift classes this file pins:

1. Every kwarg in the native's ``__text_signature__`` (the machine-readable
   source of truth for what the function accepts — the wrapper forwards
   ``**kwargs``, so ``inspect.signature`` on it sees nothing) is named as an
   ``Args:`` entry in the wrapper docstring. This is what caught ``from_h5ad``
   shipping 9 kwargs documented nowhere.
2. The native docstring is the pointer, not a stale second copy — and is never
   ``None.__doc__`` (``"The type of the None singleton."``), which is what a
   naive doc-copy over the hdf5-off ``None`` aliases would install.
3. The ``on_missing_rows`` tri-state is described by ONE canonical sentence
   wherever a pyscx docstring documents it. The enum's semantics used to exist
   in seven phrasings across pyscx / rscx / scx-cli / docs, already drifted.

Pattern precedent: tests/test_accel_stub_coverage.py (stub drift guard) and
test_obs_import.py::test_on_missing_rows_defaults_to_null (text-signature pin).
"""

import re

import pytest

import pyscx

# ---------------------------------------------------------------------------
# The wrapped pairs
# ---------------------------------------------------------------------------

# wrapper name -> attribute on the native extension module `pyscx.pyscx`.
# `write` / `read` are deliberately absent: they have no same-name native twin
# (`write` wraps `from_anndata`, `read` composes `open().to_anndata()`), so
# each has exactly one docstring by construction.
WRAPPED = [
    "open",
    "validate",
    "modify_metadata",
    "set_uns",
    "obs_import",
    "attach_obs_columns",
    "diagnose_obs_key",
    "doublet_import",
    "from_h5ad",
    "to_h5ad",
    "from_h5mu",
    "to_h5mu",
    "read_h5ad_metadata",
]

# Registered only under `--features hdf5`; on a base build the module-level
# aliases are None and there is nothing to compare against.
HDF5_ONLY = {"from_h5ad", "to_h5ad", "from_h5mu", "to_h5mu", "read_h5ad_metadata"}

POINTER_SENTENCE = "Python wrapper, which is what `help()` shows"
NONE_DOC = "The type of the None singleton."

# The one canonical description of the on_missing_rows tri-state. Semantics
# from the runtime source of truth (`parse_obs_missing_rows`, pyscx/src/ops.rs):
# null = leave uncovered target rows NULL (the default), error = refuse, zero =
# legacy alias for null. Compared whitespace-normalized so wrapping is free.
ON_MISSING_ROWS_CANONICAL = (
    'on_missing_rows: "null" (default) leaves uncovered target rows NULL; '
    '"error" refuses. "zero" is an accepted legacy alias for "null" — the '
    "shared policy's zero is literal only where the missing thing is a matrix "
    "row (cellbender_import), which really is zeros."
)


def _native(name):
    mod = pyscx.pyscx
    return getattr(mod, name, None)


def _pairs():
    for name in WRAPPED:
        if name in HDF5_ONLY and not pyscx._HAS_HDF5:
            continue
        yield name


def _kwargs_from_text_signature(sig: str) -> list[str]:
    """Parameter names from a pyo3 __text_signature__, defaults stripped.

    Signatures look like `($module, path, out, codec=None, shard_size=None, …)`.
    Top-level split is enough: pyo3 default reprs never contain commas.
    """
    inner = sig.strip()
    assert inner.startswith("(") and inner.endswith(")"), sig
    names = []
    for tok in inner[1:-1].split(","):
        tok = tok.strip()
        if not tok or tok in ("*", "/") or tok.startswith("$"):
            continue
        names.append(tok.split("=", 1)[0].strip())
    return names


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------------------------
# 1. Every accepted kwarg is documented on the wrapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(_pairs()))
def test_every_native_kwarg_is_documented_on_the_wrapper(name):
    native = _native(name)
    assert native is not None, f"pyscx.pyscx.{name} missing"
    wrapper = getattr(pyscx, name)
    assert wrapper is not native, (
        f"pyscx.{name} is the native itself — the wrapper table is stale"
    )
    doc = wrapper.__doc__ or ""
    params = _kwargs_from_text_signature(native.__text_signature__)
    # A parameter is "documented" when it appears as an Args-style `name:`
    # entry (allowing `obs / var:` style shared entries via the loose \b…\s*[/:]).
    missing = [
        p for p in params if not re.search(rf"\b{re.escape(p)}\s*[:/]", doc)
    ]
    assert not missing, (
        f"pyscx.{name} accepts kwargs its docstring never names: {missing}. "
        "The wrapper docstring is the ONLY place kwargs are user-visible "
        "(the def forwards **kwargs), so document each one in its Args block."
    )


# ---------------------------------------------------------------------------
# 2. The native side is the pointer, never a second copy or None.__doc__
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(_pairs()))
def test_native_docstring_is_the_pointer(name):
    native = _native(name)
    ndoc = native.__doc__
    assert ndoc and ndoc != NONE_DOC
    # Whitespace-normalized: the /// comment wraps the sentence freely.
    assert POINTER_SENTENCE in _norm(ndoc), (
        f"pyscx.pyscx.{name}'s Rust /// doc must point at the canonical "
        f"Python wrapper docstring (expected: …{POINTER_SENTENCE!r}). A full "
        "second copy WILL drift — that is the bug class this convention ended."
    )
    # The pointer sentence alone would pass on a doc that KEPT the full second
    # copy and merely appended the sentence. An `Args:` block is the signature
    # of a full contract, so its absence is what actually enforces
    # "pointer, not copy" (short Rust-reader notes without an Args block are
    # fine — e.g. to_h5ad's numpy-vs-wrapper coercion note).
    assert "Args:" not in ndoc, (
        f"pyscx.pyscx.{name}'s Rust /// doc carries an Args: block — that is a "
        "second copy of the contract, not a pointer. Move the content to the "
        "Python wrapper docstring."
    )


@pytest.mark.parametrize("name", list(_pairs()) + ["read", "write"])
def test_wrapper_docstring_is_substantial(name):
    doc = getattr(pyscx, name).__doc__
    assert doc and doc.strip() and NONE_DOC not in doc
    # A canonical docstring is a real contract, not a one-liner stub.
    assert len(doc) > 200, (
        f"pyscx.{name}.__doc__ is {len(doc or '')} chars — the wrapper is the "
        "single source of truth now; a stub here means the contract is lost."
    )


# ---------------------------------------------------------------------------
# 3. One canonical on_missing_rows sentence
# ---------------------------------------------------------------------------


def test_on_missing_rows_wording_is_single_sourced():
    canonical = _norm(ON_MISSING_ROWS_CANONICAL)
    for name in ("obs_import", "attach_obs_columns", "doublet_import"):
        doc = _norm(getattr(pyscx, name).__doc__)
        assert canonical in doc, (
            f"pyscx.{name} documents on_missing_rows in its own words. Use the "
            "canonical sentence from this test verbatim (surface-specific "
            "notes may FOLLOW it), so the tri-state's semantics exist once."
        )
