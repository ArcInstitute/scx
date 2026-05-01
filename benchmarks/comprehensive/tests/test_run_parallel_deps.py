"""
§4.1 — afterok dependency wiring in `run_parallel.py`.

Hermetic: stubs `submitit.AutoExecutor` to capture every `submit()` call's
parameter snapshot, sets `SCX_WORK_DIR` to a tmpdir, and runs `main()`
with a crafted argv. Verifies that:

  * A benchmark that needs conversion (and whose target file does not
    yet exist) is submitted with
    ``slurm_additional_parameters={"dependency": "afterok:<conv_jid>"}``
    pointing at the conversion job for the same (dataset, format).
  * A benchmark whose target file already exists is submitted with
    NO dependency.
  * A `_NO_CONVERSION` benchmark (e.g. `write`) is submitted with
    NO dependency.
  * Phase A no longer blocks Phase B — `bench_jobs` are submitted
    even though no conversion has had `.result()` called on it.
  * `run_manifest.json` records the dependency for each submitted entry.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fake submitit
# ---------------------------------------------------------------------------


class _FakeJob:
    def __init__(self, job_id: str, folder: str) -> None:
        self.job_id = job_id
        self.paths = types.SimpleNamespace(folder=folder)
        self._result_called = False

    def result(self):  # noqa: D401
        self._result_called = True
        return None


class _FakeAutoExecutor:
    instances: list["_FakeAutoExecutor"] = []
    submissions: list[dict] = []

    def __init__(self, folder: str) -> None:
        self.folder = folder
        self._params: dict = {}
        self._next_id = 1000 + len(_FakeAutoExecutor.instances) * 1000
        _FakeAutoExecutor.instances.append(self)

    def update_parameters(self, **kwargs) -> None:
        self._params = dict(kwargs)

    def submit(self, fn, *args, **kwargs):
        jid = str(self._next_id)
        self._next_id += 1
        job = _FakeJob(jid, folder=self.folder)
        _FakeAutoExecutor.submissions.append({
            "folder": self.folder,
            "job_id": jid,
            "fn_name": getattr(fn, "__name__", repr(fn)),
            "fn_args": args,
            "params": dict(self._params),
        })
        return job

    @classmethod
    def reset(cls) -> None:
        cls.instances.clear()
        cls.submissions.clear()


def _install_fake_submitit() -> None:
    fake = types.ModuleType("submitit")
    fake.AutoExecutor = _FakeAutoExecutor
    fake.Job = _FakeJob
    sys.modules["submitit"] = fake


# ---------------------------------------------------------------------------
# Fixture: isolated SCX_WORK_DIR with pbmc3k.h5ad present, scx_auto missing
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_work_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Re-root SCX_WORK_DIR / SCX_DATA_DIR so path_for_format() lives under tmp_path.

    Touches `pbmc3k.h5ad` (the source) and `pbmc3k.h5ad` only — the
    `pbmc3k_auto.scx` target stays absent so Phase A submits a conversion
    for it.
    """
    work = tmp_path / "work"
    data = work / "benchmarks" / "datasets"
    data.mkdir(parents=True, exist_ok=True)

    # `h5ad_none` and the source h5ad share the same path
    # (`<name>.h5ad`), so creating the source file simultaneously
    # marks `h5ad_none` as already-present.
    (data / "pbmc3k.h5ad").touch()

    monkeypatch.setenv("SCX_WORK_DIR", str(work))
    monkeypatch.setenv("SCX_DATA_DIR", str(data))

    # Reload the env + config modules so DATA_DIR picks up the new env.
    for modname in (
        "benchmarks.comprehensive.bench_env",
        "benchmarks.comprehensive.config",
        "benchmarks.comprehensive.benchmarks",
        "benchmarks.comprehensive.scripts.run_parallel",
    ):
        if modname in sys.modules:
            del sys.modules[modname]

    return work, data


def _run_main_with_argv(argv: list[str]) -> object:
    """Invoke run_parallel.main() with sys.argv set to argv."""
    import benchmarks.comprehensive.scripts.run_parallel as rp

    saved = sys.argv
    try:
        sys.argv = ["run_parallel.py"] + argv
        rp.main()
    finally:
        sys.argv = saved
    return rp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_afterok_dependency_wired_for_fresh_conversion(
    isolated_work_dir, monkeypatch: pytest.MonkeyPatch, caplog
):
    """A bench that needs conversion gets afterok:<conv_jid>; one whose
    target already exists, and one in `_NO_CONVERSION`, do not."""
    _FakeAutoExecutor.reset()
    _install_fake_submitit()

    rp = _run_main_with_argv([
        "--datasets", "pbmc3k",
        "--formats", "scx_auto", "h5ad_none",
        "--benchmarks", "read_full", "write",
        "--skip-smoke",
        # Stay off the auto-diff side-channel — it's a no-op without a
        # canonical baseline, but the `LATEST` symlink probe runs against
        # a real path which we don't control here.
    ])

    subs = _FakeAutoExecutor.submissions

    conv_subs = [s for s in subs if s["fn_name"] == "_run_conversion"]
    bench_subs = [s for s in subs if s["fn_name"] == "_run_benchmark"]

    # Phase A: only scx_auto needed converting — h5ad_none.h5ad already
    # exists at <data>/pbmc3k.h5ad.
    assert len(conv_subs) == 1, conv_subs
    conv = conv_subs[0]
    assert conv["fn_args"][0] == "pbmc3k"
    assert conv["fn_args"][1] == "scx_auto"
    conv_jid = conv["job_id"]

    # Phase B: 4 cells (2 benches × 2 formats).
    by_label = {(s["fn_args"][0], s["fn_args"][1], s["fn_args"][2]): s for s in bench_subs}
    assert set(by_label.keys()) == {
        ("read_full", "pbmc3k", "scx_auto"),
        ("read_full", "pbmc3k", "h5ad_none"),
        ("write",     "pbmc3k", "scx_auto"),
        ("write",     "pbmc3k", "h5ad_none"),
    }

    # read_full + scx_auto: scx_auto target absent → Phase A submitted
    # → afterok dependency on the conv job.
    rf_scx = by_label[("read_full", "pbmc3k", "scx_auto")]
    assert rf_scx["params"].get("slurm_additional_parameters") == {
        "dependency": f"afterok:{conv_jid}",
    }

    # read_full + h5ad_none: target file already exists → no dependency.
    rf_h5 = by_label[("read_full", "pbmc3k", "h5ad_none")]
    assert "slurm_additional_parameters" not in rf_h5["params"] or \
        rf_h5["params"]["slurm_additional_parameters"] in (None, {}), \
        rf_h5["params"]
    assert "dependency" not in (rf_h5["params"].get("slurm_additional_parameters") or {})

    # write/* (in _NO_CONVERSION): no dependency, regardless of format.
    for k in [("write", "pbmc3k", "scx_auto"), ("write", "pbmc3k", "h5ad_none")]:
        w = by_label[k]
        extra = w["params"].get("slurm_additional_parameters") or {}
        assert "dependency" not in extra, (k, w["params"])


def test_phase_a_does_not_block_phase_b(isolated_work_dir):
    """No conversion job's `.result()` should be called BEFORE the bench
    submissions land — this is the whole point of §4.1."""
    _FakeAutoExecutor.reset()
    _install_fake_submitit()

    submissions_before_result_call: list[int] = []

    # Wrap _FakeJob.result so we can detect when (if ever) it's invoked.
    original_result = _FakeJob.result

    def _tracking_result(self):
        # Snapshot the current submission count at the moment .result() is called.
        submissions_before_result_call.append(len(_FakeAutoExecutor.submissions))
        return original_result(self)

    _FakeJob.result = _tracking_result  # type: ignore[assignment]
    try:
        _run_main_with_argv([
            "--datasets", "pbmc3k",
            "--formats", "scx_auto", "h5ad_none",
            "--benchmarks", "read_full", "write",
            "--skip-smoke",
        ])
    finally:
        _FakeJob.result = original_result  # type: ignore[assignment]

    n_total = len(_FakeAutoExecutor.submissions)

    # Every .result() invocation must observe ALL jobs already submitted.
    # If Phase A blocked Phase B, conv_job.result() would fire while
    # bench_jobs.append loop hadn't yet pushed any bench submissions, so
    # the snapshot would be < n_total.
    assert all(c == n_total for c in submissions_before_result_call), (
        f"Some .result() calls observed {submissions_before_result_call} "
        f"submissions, but total submitted was {n_total}. "
        "Phase A is still blocking Phase B."
    )


def test_run_manifest_records_dependency(isolated_work_dir, tmp_path):
    """run_manifest.json should carry each bench job's dependency jobid."""
    _FakeAutoExecutor.reset()
    _install_fake_submitit()

    rp = _run_main_with_argv([
        "--datasets", "pbmc3k",
        "--formats", "scx_auto", "h5ad_none",
        "--benchmarks", "read_full", "write",
        "--skip-smoke",
    ])

    manifest_path = rp.LOGS_DIR / "run_manifest.json"
    assert manifest_path.exists(), f"Manifest missing at {manifest_path}"

    import json
    manifest = json.loads(manifest_path.read_text())
    by_label = {entry["label"]: entry for entry in manifest["submitted"]}

    # The afterok edge sits on the (read_full, scx_auto) cell only.
    assert by_label["read_full/pbmc3k/scx_auto"]["dependency"] is not None
    assert by_label["read_full/pbmc3k/h5ad_none"]["dependency"] is None
    assert by_label["write/pbmc3k/scx_auto"]["dependency"] is None
    assert by_label["write/pbmc3k/h5ad_none"]["dependency"] is None
