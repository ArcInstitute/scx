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
    "sphinx.ext.linkcode",
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


# -- "View source on GitHub" link on each autodoc entry ----------------------
#
# `sphinx.ext.linkcode` adds a `[source]` link next to each documented
# function/class. Our resolver returns a GitHub blob URL for pure-Python
# symbols (iter_chunks, ScxDataModule, …) and `None` for Rust-defined
# (PyO3-compiled) symbols, since those have no traceable Python source.

import importlib  # noqa: E402
import inspect    # noqa: E402
import os         # noqa: E402
import subprocess  # noqa: E402

GITHUB_USER = "ArcInstitute"
GITHUB_REPO = "scx"


def _resolve_git_ref() -> str:
    """Pick the right Git ref for source links.

    Priority:
      1. ``READTHEDOCS_GIT_IDENTIFIER`` (set by RTD on every build —
         the branch / tag / commit SHA being built).
      2. ``git rev-parse HEAD`` against the local checkout.
      3. ``"main"`` as a last resort.
    """
    env_ref = os.environ.get("READTHEDOCS_GIT_IDENTIFIER")
    if env_ref:
        return env_ref
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "main"


_GIT_REF = _resolve_git_ref()


def linkcode_resolve(domain: str, info: dict) -> str | None:  # noqa: D401
    """Return a GitHub URL for the source of ``info``, or ``None``.

    Called once per documented Python object. We resolve the object via
    ``inspect`` and turn its file path + line range into a GitHub blob
    URL. If the object has no introspectable Python source (typical for
    PyO3-compiled symbols), return ``None`` so no link is rendered.
    """
    if domain != "py" or not info.get("module"):
        return None

    try:
        module = importlib.import_module(info["module"])
    except ImportError:
        return None

    obj = module
    for part in info["fullname"].split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    # Unwrap descriptors / partials / decorated wrappers where possible.
    obj = inspect.unwrap(obj) if hasattr(obj, "__wrapped__") else obj

    try:
        source_file = inspect.getsourcefile(obj)
    except TypeError:
        return None
    if not source_file:
        return None

    try:
        rel = os.path.relpath(source_file, str(REPO_ROOT))
    except ValueError:
        return None
    # If the object's source lives outside the repo (e.g. a stdlib type
    # the user re-exported) skip it.
    if rel.startswith(".."):
        return None

    try:
        source_lines, start_line = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        # No line info — link to the file without a fragment.
        return f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/blob/{_GIT_REF}/{rel}"

    end_line = start_line + len(source_lines) - 1
    return (
        f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/blob/{_GIT_REF}/"
        f"{rel}#L{start_line}-L{end_line}"
    )
