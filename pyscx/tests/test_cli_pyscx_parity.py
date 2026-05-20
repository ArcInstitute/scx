"""CLI ↔ pyscx convert parity regression (D1-2026-05-20).

PR #112 fixed B1/B2 — the CLI `scx convert` path now preserves
`obs_names` / `var_names` and `pyscx.open(...).to_anndata()` no longer
leaks `_index` into the DataFrame columns. These tests lock that in
from the Python user's perspective so any future divergence between
`scx convert` and `pyscx.from_h5ad` is caught here.

Tests are skipped when the `scx` CLI binary isn't present (CI jobs that
only build pyscx don't build the binary), so this file is harmless in
that environment but runs end-to-end on a developer's machine.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import anndata as ad
import pandas as pd
import pytest


def _scx_binary() -> str:
    """Locate the `scx` CLI binary.

    Order of resolution:
    1. `SCX_BIN` env var (CI / explicit override).
    2. `<repo_root>/target/release/scx`.
    3. `<repo_root>/target/debug/scx` (rarely useful but lets the test
       run after `cargo build` without `--release`).

    Skips the test cleanly when none of those exist — the parity
    invariant is still covered by Rust tests; the Python check is a
    belt-and-braces add for the user-visible flow.
    """
    env = os.environ.get("SCX_BIN")
    if env:
        if Path(env).is_file():
            return env
        pytest.skip(f"SCX_BIN={env} is not a file")

    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (
        repo_root / "target" / "release" / "scx",
        repo_root / "target" / "debug" / "scx",
    ):
        if candidate.is_file():
            return str(candidate)
    pytest.skip(
        f"scx CLI not found at {repo_root}/target/{{release,debug}}/scx. "
        "Build with `cargo build --release -p scx-cli --features hdf5` "
        "or set the SCX_BIN env var."
    )


def test_cli_convert_preserves_obs_var_names(synthetic_adata, tmp_dir):
    """The CLI's `scx convert` path must produce an SCX whose
    `to_anndata()` returns the source obs_names / var_names verbatim.
    Regression for B1-2026-05-20."""
    import pyscx

    scx_bin = _scx_binary()
    h5ad_path = tmp_dir / "input.h5ad"
    scx_path = tmp_dir / "out_cli.scx"
    synthetic_adata.write_h5ad(h5ad_path)

    subprocess.run(
        [scx_bin, "convert", str(h5ad_path), str(scx_path), "--stream"],
        check=True,
        capture_output=True,
    )

    got = pyscx.open(str(scx_path)).to_anndata()
    pd.testing.assert_index_equal(
        synthetic_adata.obs.index, got.obs.index, check_names=False
    )
    pd.testing.assert_index_equal(
        synthetic_adata.var.index, got.var.index, check_names=False
    )
    assert "_index" not in got.obs.columns
    assert "_index" not in got.var.columns
    assert "__index_level_0__" not in got.obs.columns
    assert "__index_level_0__" not in got.var.columns


def test_cli_convert_matches_pyscx_from_h5ad(synthetic_adata, tmp_dir):
    """Both convert surfaces (CLI + pyscx) must produce SCX files that
    open into structurally-equivalent AnnData. Locks down CLI/pyscx
    parity beyond just "names round-trip"."""
    import pyscx

    scx_bin = _scx_binary()
    h5ad_path = tmp_dir / "input.h5ad"
    via_cli = tmp_dir / "out_cli.scx"
    via_pyscx = tmp_dir / "out_pyscx.scx"
    synthetic_adata.write_h5ad(h5ad_path)

    subprocess.run(
        [scx_bin, "convert", str(h5ad_path), str(via_cli), "--stream"],
        check=True,
        capture_output=True,
    )
    pyscx.from_h5ad(str(h5ad_path), str(via_pyscx), stream=True)

    a = pyscx.open(str(via_cli)).to_anndata()
    b = pyscx.open(str(via_pyscx)).to_anndata()
    pd.testing.assert_index_equal(a.obs.index, b.obs.index, check_names=False)
    pd.testing.assert_index_equal(a.var.index, b.var.index, check_names=False)
    assert sorted(a.obs.columns) == sorted(b.obs.columns), (
        list(a.obs.columns),
        list(b.obs.columns),
    )
    assert sorted(a.var.columns) == sorted(b.var.columns), (
        list(a.var.columns),
        list(b.var.columns),
    )


def test_cli_convert_then_write_h5ad_does_not_crash(synthetic_adata, tmp_dir):
    """Regression for B2-2026-05-20: after CLI convert + `to_anndata()`,
    calling `adata.write_h5ad(...)` must not raise. The pre-PR-#112 bug
    left `_index` as a regular obs column, which anndata 0.10+ rejects."""
    import pyscx

    scx_bin = _scx_binary()
    h5ad_path = tmp_dir / "input.h5ad"
    scx_path = tmp_dir / "out_cli.scx"
    out_h5ad = tmp_dir / "roundtrip.h5ad"
    synthetic_adata.write_h5ad(h5ad_path)

    subprocess.run(
        [scx_bin, "convert", str(h5ad_path), str(scx_path), "--stream"],
        check=True,
        capture_output=True,
    )

    got = pyscx.open(str(scx_path)).to_anndata()
    got.write_h5ad(out_h5ad)
    rt = ad.read_h5ad(out_h5ad)
    pd.testing.assert_index_equal(
        synthetic_adata.obs.index, rt.obs.index, check_names=False
    )
    pd.testing.assert_index_equal(
        synthetic_adata.var.index, rt.var.index, check_names=False
    )
