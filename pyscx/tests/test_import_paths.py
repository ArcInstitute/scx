"""Symmetric Python import paths for the pyscx package.

`import pyscx.accel as a` raised `ModuleNotFoundError` while
`from pyscx import accel` worked. The scanpy-trained idiom
(`import scanpy as sc; sc.pp.normalize_total(...)`) led docs-skimmers
straight into the broken import. The fix registers the Rust-side
`accel` submodule under `sys.modules["pyscx.accel"]` so both idioms
resolve and the user gets the same object either way.

`pyscx.tokenize` is the second Rust-side submodule and needs the same
registration — `m.add_submodule` sets an attribute and nothing else — so it is
covered by the same four checks rather than being assumed to inherit them.
"""

import subprocess
import sys


def test_pyscx_accel_in_sys_modules_after_import_pyscx():
    """The fix mechanism: after `import pyscx`, `sys.modules["pyscx.accel"]`
    must be populated. Without this, Python's import machinery has no entry
    to return when the user later does `import pyscx.accel`.
    """
    import pyscx  # noqa: F401 — triggers __init__.py side effects

    assert "pyscx.accel" in sys.modules


def test_dotted_and_from_import_yield_same_object():
    """Both idioms must resolve to the *same* Rust-side submodule, so users
    can mix-and-match without getting different attribute namespaces.
    """
    import pyscx.accel as via_dotted
    from pyscx import accel as via_from

    assert via_dotted is via_from


def test_pyscx_accel_has_expected_surface():
    """Sanity: the accel submodule exposes the core accelerator API."""
    import pyscx.accel as accel

    for name in ("pca", "normalize_total", "log1p", "neighbors",
                 "leiden", "umap", "highly_variable_genes", "gpu_info"):
        assert hasattr(accel, name), f"pyscx.accel missing {name!r}"


def test_dotted_import_works_in_a_fresh_interpreter():
    """The strongest variant: spawn a clean subprocess that has never
    imported pyscx and run `import pyscx.accel as a` cold. Pre-fix this
    raised ModuleNotFoundError; post-fix it must succeed.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import pyscx.accel as a; print(a.pca)"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"`import pyscx.accel` failed in a fresh subprocess.\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    assert "function pca" in result.stdout or "pca" in result.stdout


# ---------------------------------------------------------------------------
# pyscx.tokenize — the second Rust-side submodule (W6)
# ---------------------------------------------------------------------------


def test_pyscx_tokenize_in_sys_modules_after_import_pyscx():
    import pyscx  # noqa: F401 — triggers __init__.py side effects

    assert "pyscx.tokenize" in sys.modules


def test_tokenize_dotted_and_from_import_yield_same_object():
    import pyscx.tokenize as via_dotted
    from pyscx import tokenize as via_from

    assert via_dotted is via_from


def test_pyscx_tokenize_has_expected_surface():
    import pyscx.tokenize as tokenize

    for name in ("top_k", "rank_tokens", "bin_values", "sample_genes",
                 "transform_values", "library_size", "measured_mask",
                 "gene_mask_id", "pad_id", "contract_version"):
        assert hasattr(tokenize, name), f"pyscx.tokenize missing {name!r}"
    assert isinstance(tokenize.CONTRACT_VERSION, int)


def test_tokenize_dotted_import_works_in_a_fresh_interpreter():
    """Cold import in a clean subprocess — the strongest variant, and the one
    that would have caught a missing `sys.modules` entry."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import pyscx.tokenize as t; print(t.CONTRACT_VERSION)"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"`import pyscx.tokenize` failed in a fresh subprocess.\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    assert result.stdout.strip().isdigit()
