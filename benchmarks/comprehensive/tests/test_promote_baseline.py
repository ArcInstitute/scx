"""
§1.4 — `promote_baseline.py --no-latest` must not break `baselines/LATEST`.

Hermetic: monkeypatches ``promote_baseline.BASELINES_DIR`` to a tmpdir so
no real promotion touches ``results/baselines/``. Verifies that:

  * ``promote(..., update_latest=True)`` (the default) creates
    ``baselines/LATEST → <version>``.
  * ``promote(..., update_latest=False)`` on an empty baselines tree
    leaves LATEST absent (no symlink, no pointer file).
  * ``promote(..., update_latest=False)`` does NOT clobber a prior
    ``baselines/LATEST → v_old`` — the prior pointer is preserved
    intact. This is the regression case for §1.4.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.comprehensive.scripts import promote_baseline


def _make_snapshot(root: Path, name: str = "snap") -> Path:
    """Create a minimal capture_baseline.py-shaped snapshot dir."""
    snap = root / name
    snap.mkdir()
    for f in promote_baseline._REQUIRED_FILES:
        (snap / f).write_text("{}\n")
    return snap


@pytest.fixture
def baselines_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bdir = tmp_path / "baselines"
    bdir.mkdir()
    monkeypatch.setattr(promote_baseline, "BASELINES_DIR", bdir)
    return bdir


def test_default_creates_latest_symlink(baselines_dir: Path, tmp_path: Path) -> None:
    snap = _make_snapshot(tmp_path)

    target = promote_baseline.promote(snap, "v_new")

    assert target == baselines_dir / "v_new"
    latest = baselines_dir / "LATEST"
    assert latest.is_symlink()
    assert latest.resolve() == (baselines_dir / "v_new").resolve()


def test_no_latest_on_empty_tree_leaves_latest_absent(
    baselines_dir: Path, tmp_path: Path
) -> None:
    snap = _make_snapshot(tmp_path)

    promote_baseline.promote(snap, "v_new", update_latest=False)

    latest = baselines_dir / "LATEST"
    assert not latest.exists()
    assert not latest.is_symlink()
    # Belt-and-suspenders: the pointer-file fallback path must not have
    # silently materialized either.
    assert not latest.is_file()


def test_no_latest_preserves_prior_pointer(
    baselines_dir: Path, tmp_path: Path
) -> None:
    """§1.4 regression: backfilling historical baselines must not strip
    the existing LATEST pointer.

    Pre-promote v_old so LATEST → v_old/, then promote v_new with
    update_latest=False. LATEST must still point at v_old.
    """
    snap_old = _make_snapshot(tmp_path, name="snap_old")
    promote_baseline.promote(snap_old, "v_old")

    latest = baselines_dir / "LATEST"
    assert latest.is_symlink()
    assert latest.resolve() == (baselines_dir / "v_old").resolve()

    snap_new = _make_snapshot(tmp_path, name="snap_new")
    promote_baseline.promote(snap_new, "v_new", update_latest=False)

    assert (baselines_dir / "v_new").is_dir()
    assert latest.is_symlink()
    assert latest.resolve() == (baselines_dir / "v_old").resolve()


def test_main_no_latest_flag_routes_through_promote(
    baselines_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: invoking main() with --no-latest must preserve the
    prior LATEST pointer (covers the argparse → promote() wiring)."""
    snap_old = _make_snapshot(tmp_path, name="snap_old")
    promote_baseline.promote(snap_old, "v_old")

    snap_new = _make_snapshot(tmp_path, name="snap_new")
    rc = promote_baseline.main(
        [
            "--snapshot", str(snap_new),
            "--version", "v_new",
            "--no-latest",
        ]
    )
    assert rc == 0

    latest = baselines_dir / "LATEST"
    assert latest.is_symlink()
    assert latest.resolve() == (baselines_dir / "v_old").resolve()
