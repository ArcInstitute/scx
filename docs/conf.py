"""Sphinx configuration for the SCX documentation site.

Builds the `docs/` tree (markdown architecture/reference docs + an
autodoc-driven Python API page) into HTML for ReadTheDocs.

Layout choice: the source files (`*.md`, `index.md`, `python_api.rst`)
live alongside this `conf.py` in `docs/`, so existing in-repo links
between markdown docs (`[format.md](format.md)`, etc.) keep working both
on GitHub and in the rendered site.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent

# Make the pyscx Python source importable for autodoc. The compiled Rust
# extension (`pyscx.pyscx`) is not available on RTD; it's mocked below.
sys.path.insert(0, str(REPO_ROOT / "pyscx" / "python"))

project = "SCX"
author = "Arc Institute"
copyright = "2025, Arc Institute"

# Pull the version from pyscx/pyproject.toml so docs match the package.
def _read_version() -> str:
    pyproject = REPO_ROOT / "pyscx" / "pyproject.toml"
    try:
        for line in pyproject.read_text().splitlines():
            line = line.strip()
            if line.startswith("version"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return "0.0.0"


release = _read_version()
version = ".".join(release.split(".")[:2])

# -- Extensions --------------------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST (markdown) ---------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "attrs_inline",
    "tasklist",
    "linkify",
    "smartquotes",
]
myst_heading_anchors = 4
myst_linkify_fuzzy_links = False

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Heavy optional integrations are mocked so a slim docs env can import the
# modules for autodoc. `pyscx.pyscx` (the compiled PyO3 extension) is NOT
# mocked — it's built into the docs venv via `maturin develop` so autodoc
# can introspect every Rust-defined function/class with its real docstring.
autodoc_mock_imports = [
    "torch",
    "lightning",
    "pytorch_lightning",
    "scvi",
    "scvi_tools",
    "scanpy",
    "skmisc",
    "pydeseq2",
]

# PyO3 registers `pyscx.accel` as an attribute on the `pyscx` module but
# does NOT install it in `sys.modules`, so `import pyscx.accel` fails and
# autodoc's `automodule:: pyscx.accel` can't import it. Register it
# manually so autodoc can find it.
try:
    import sys
    import pyscx as _pyscx_for_docs

    for _sub in ("accel",):
        _mod = getattr(_pyscx_for_docs, _sub, None)
        if _mod is not None:
            sys.modules.setdefault(f"pyscx.{_sub}", _mod)
except ImportError:
    # Compiled extension not built — autodoc entries for Rust-defined
    # symbols will be empty but the rest of the site still builds.
    pass

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = False

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"SCX {release}"
html_static_path = ["_static"]
templates_path = ["_templates"]

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# Some markdown docs cross-link to files outside docs/ (e.g.
# `../benchmarks/README.md`). Those resolve fine on GitHub but not in the
# built site — silence the warning rather than failing the build.
suppress_warnings = [
    "myst.xref_missing",
    "myst.header",
    "myst.iref_ambiguous",
]

# Keep build non-fatal on these RTD environments (Rust extension not built).
nitpicky = False
