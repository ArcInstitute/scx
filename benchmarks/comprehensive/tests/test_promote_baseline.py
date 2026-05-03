"""
§1.3 / §1.4 — promote_baseline.py invariants.

Hermetic: monkeypatches ``promote_baseline.BASELINES_DIR`` to a tmpdir so
no real promotion touches ``results/baselines/``. Verifies that:

  * ``promote(..., update_latest=True)`` (the default) creates
    ``baselines/LATEST → <version>``.
  * ``promote(..., update_latest=False)`` on an empty baselines tree
    leaves LATEST absent (no symlink, no pointer file).
  * ``promote(..., update_latest=False)`` does NOT clobber a prior
    ``baselines/LATEST → v_old`` — the prior pointer is preserved
    intact. This is the regression case for §1.4.
  * ``promote()`` copies ``fingerprints/fingerprints.json`` into the
    promoted tree and refuses promotion when it's missing — closing
    the silent ``diff_fingerprints`` no-op called out in §1.3.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from benchmarks.comprehensive.scripts import promote_baseline


def _make_snapshot(root: Path, name: str = "snap") -> Path:
    """Create a minimal capture_baseline.py-shaped snapshot dir."""
    snap = root / name
    snap.mkdir()
    for f in promote_baseline._REQUIRED_FILES:
        dst = snap / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("{}\n")
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


# ---------------------------------------------------------------------------
# §1.3 — fingerprints must travel with promoted baselines
# ---------------------------------------------------------------------------


_GATE_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "benchmarks" / "comprehensive" / "scripts" / "compare_against_baseline.py"
)

_FINGERPRINT_PAYLOAD = {
    "schema_version": 1,
    "fingerprints": {
        "pca_X_pca": {
            "hash": "0123456789abcdef0123456789abcdef",
            "dtype": "float32",
            "shape": [2700, 50],
        },
    },
}


def _import_gate_module():
    """Load compare_against_baseline.py as ``compare_against_baseline``."""
    name = "compare_against_baseline"
    spec = importlib.util.spec_from_file_location(name, str(_GATE_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_promote_copies_fingerprints(baselines_dir: Path, tmp_path: Path) -> None:
    """The promoted tree must carry fingerprints/fingerprints.json byte-equal
    to the snapshot's copy — that's what closes the §1.3 silent no-op."""
    snap = _make_snapshot(tmp_path)
    fp_src = snap / "fingerprints" / "fingerprints.json"
    fp_src.write_text(json.dumps(_FINGERPRINT_PAYLOAD))

    promote_baseline.promote(snap, "v_new")

    fp_dst = baselines_dir / "v_new" / "fingerprints" / "fingerprints.json"
    assert fp_dst.is_file()
    assert json.loads(fp_dst.read_text()) == _FINGERPRINT_PAYLOAD


def test_promote_fails_when_fingerprints_missing(
    baselines_dir: Path, tmp_path: Path
) -> None:
    """A snapshot lacking fingerprints/fingerprints.json must be refused.

    Otherwise the promoted baseline would silently degrade
    diff_fingerprints to a no-op (the original §1.3 bug).
    """
    snap = _make_snapshot(tmp_path)
    (snap / "fingerprints" / "fingerprints.json").unlink()

    with pytest.raises(FileNotFoundError, match="fingerprints"):
        promote_baseline.promote(snap, "v_new")

    assert not (baselines_dir / "v_new").exists()


def test_gate_load_fingerprints_reads_promoted_baseline(
    baselines_dir: Path, tmp_path: Path
) -> None:
    """The full chain: capture → promote → gate must surface the
    fingerprints map non-empty.

    Pre-fix this assertion failed because diff_fingerprints saw ``{}`` on
    every promoted baseline regardless of what the snapshot recorded.
    """
    snap = _make_snapshot(tmp_path)
    (snap / "fingerprints" / "fingerprints.json").write_text(
        json.dumps(_FINGERPRINT_PAYLOAD)
    )
    promote_baseline.promote(snap, "v_new")

    gate = _import_gate_module()
    loaded = gate._load_fingerprints(baselines_dir / "v_new")
    assert loaded == _FINGERPRINT_PAYLOAD
    assert loaded.get("fingerprints"), "diff_fingerprints would be a no-op"
