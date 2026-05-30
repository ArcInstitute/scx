"""
afterok dependency wiring in `run_parallel.py`.

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

    def __init__(self, folder: str, **kwargs) -> None:
        self.folder = folder
        self._params: dict = {}
        self._next_id = 1000 + len(_FakeAutoExecutor.instances) * 1000
        _FakeAutoExecutor.instances.append(self)

    def update_parameters(self, **kwargs) -> None:
        # NOTE: This replaces self._params entirely, which differs from
        # real submitit (which merges). This is a pre-existing test
        # limitation and is acceptable since the production code always
        # passes all relevant fields on every call.
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

    def map_array(self, fn, *arg_sequences):
        """Mock Slurm Job Array submissions by fanning tasks out.

        Generates task IDs in the native ``<ArrayJobID>_<TaskID>`` format
        matching what real submitit returns for `executor.map_array(...)`.
        """
        jobs = []
        n_tasks = len(arg_sequences[0])
        array_jid = str(self._next_id)

        for idx in range(n_tasks):
            task_jid = f"{array_jid}_{idx}"
            task_args = [seq[idx] for seq in arg_sequences]
            job = _FakeJob(task_jid, folder=self.folder)

            _FakeAutoExecutor.submissions.append({
                "folder": self.folder,
                "job_id": task_jid,
                "fn_name": getattr(fn, "__name__", repr(fn)),
                "fn_args": task_args,
                "params": dict(self._params),
            })
            jobs.append(job)

        self._next_id += 1
        return jobs

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

    # Re-root LOGS_DIR onto tmp_path so the test never writes the
    # production `run_manifest.json` (the manifest path is a module-level
    # constant in run_parallel.py and isn't derived from SCX_WORK_DIR).
    # Without this, parallel test runs and dogfooding orchestrators stomp
    # each other.
    import benchmarks.comprehensive.scripts.run_parallel as rp
    logs_dir = tmp_path / "logs" / "submitit"
    logs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rp, "LOGS_DIR", logs_dir, raising=False)

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
    submissions land — Phase A submitting must not block Phase B."""
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


def test_no_conversion_benchmarks_have_no_dependency(isolated_work_dir):
    """write and parallel_write_scaling must NOT have afterok dependencies,
    while read_full on the same (dataset, format) must have one."""
    _FakeAutoExecutor.reset()
    _install_fake_submitit()

    _run_main_with_argv([
        "--datasets", "pbmc3k",
        "--formats", "scx_auto",
        "--benchmarks", "read_full", "write",
        "--skip-smoke",
    ])

    subs = _FakeAutoExecutor.submissions
    bench_subs = [s for s in subs if s["fn_name"] == "_run_benchmark"]

    # Build lookup: (bench_name, dataset, format) -> submission
    by_label = {(s["fn_args"][0], s["fn_args"][1], s["fn_args"][2]): s for s in bench_subs}

    # write/pbmc3k/scx_auto: _NO_CONVERSION → no dependency
    w = by_label[("write", "pbmc3k", "scx_auto")]
    extra = w["params"].get("slurm_additional_parameters") or {}
    assert "dependency" not in extra, (
        f"write should have no dependency but has: {w['params']}"
    )

    # read_full/pbmc3k/scx_auto: needs conversion → has afterok dependency
    rf = by_label[("read_full", "pbmc3k", "scx_auto")]
    rf_extra = rf["params"].get("slurm_additional_parameters") or {}
    assert "dependency" in rf_extra, (
        f"read_full should have afterok dependency but doesn't: {rf['params']}"
    )


def test_manifest_records_array_task_job_ids(isolated_work_dir):
    """run_manifest.json must carry each Phase-B job's array-task id and folder
    so watch.py can resolve ``<jobid>_0_result.pkl`` without guessing."""
    _FakeAutoExecutor.reset()
    _install_fake_submitit()

    rp = _run_main_with_argv([
        "--datasets", "pbmc3k",
        "--formats", "scx_auto",
        "--benchmarks", "read_full",
        "--skip-smoke",
    ])

    manifest_path = rp.LOGS_DIR / "run_manifest.json"
    assert manifest_path.exists()

    import json
    manifest = json.loads(manifest_path.read_text())

    # Required fields per entry: label, job_id, submitit_folder, dependency.
    for entry in manifest["submitted"]:
        assert set(entry.keys()) >= {
            "label", "job_id", "submitit_folder", "dependency",
        }, f"Entry missing required keys: {entry}"
        # Phase B jobs are submitted via map_array, so job IDs use the
        # SLURM array-task shape ``<array>_<idx>``.
        assert "_" in entry["job_id"], (
            f"Expected array-task id (got {entry['job_id']!r}) for {entry['label']}"
        )


def test_bench_format_compatible_helper():
    """_bench_format_compatible correctly enforces accel/CSC pairing rules."""
    # Import directly to test the helper
    from benchmarks.comprehensive.scripts.run_parallel import _bench_format_compatible

    # accel_pca only pairs with accel_pca__* formats
    assert _bench_format_compatible("accel_pca", "accel_pca__scx_auto") is True
    assert _bench_format_compatible("accel_pca", "accel_pca__pyscx_cpu_auto") is True
    assert _bench_format_compatible("accel_pca", "accel_knn__scx_auto") is False
    assert _bench_format_compatible("accel_pca", "scx_auto") is False
    assert _bench_format_compatible("accel_pca", "h5ad_none") is False

    # bench_csc_dispatch only pairs with bench_csc__* formats
    assert _bench_format_compatible("bench_csc_dispatch", "bench_csc__pca_csr") is True
    assert _bench_format_compatible("bench_csc_dispatch", "scx_auto") is False

    # Non-accel benchmarks skip accel and bench_csc formats
    assert _bench_format_compatible("read_full", "scx_auto") is True
    assert _bench_format_compatible("read_full", "h5ad_none") is True
    assert _bench_format_compatible("read_full", "accel_pca__scx_auto") is False
    assert _bench_format_compatible("read_full", "bench_csc__pca_csr") is False


def test_bench_format_compatible_consults_supported_formats():
    """Static-guarded benches expose ``SUPPORTED_FORMATS`` and are filtered
    accordingly at cohort-build time."""
    from benchmarks.comprehensive.scripts.run_parallel import (
        _bench_format_compatible,
        _bench_supported_formats,
    )

    # cloud_push only accepts scx_auto
    assert _bench_supported_formats("cloud_push") == frozenset({"scx_auto"})
    assert _bench_format_compatible("cloud_push", "scx_auto") is True
    assert _bench_format_compatible("cloud_push", "h5ad_gzip") is False
    assert _bench_format_compatible("cloud_push", "zarr_zstd") is False

    # roundtrip accepts the 6 SCX codec variants
    rt = _bench_supported_formats("roundtrip")
    assert rt == frozenset({
        "scx_auto", "scx_none", "scx_scx1", "scx_zstd", "scx_lz4", "scx_pcodec",
    })
    assert _bench_format_compatible("roundtrip", "scx_zstd") is True
    assert _bench_format_compatible("roundtrip", "h5ad_gzip") is False

    # correctness + cell_eval_parity_perf are also scx_auto-only
    assert _bench_format_compatible("correctness", "scx_auto") is True
    assert _bench_format_compatible("correctness", "h5ad_lzf") is False
    assert _bench_format_compatible("cell_eval_parity_perf", "scx_auto") is True
    assert _bench_format_compatible("cell_eval_parity_perf", "tiledb_soma") is False


def test_unrestricted_bench_accepts_all_formats():
    """Benches without ``SUPPORTED_FORMATS`` fall through to "any format"
    (subject to the accel/CSC pairing rules)."""
    from benchmarks.comprehensive.scripts.run_parallel import (
        _bench_format_compatible,
        _bench_supported_formats,
    )

    # read_full has no static guard
    assert _bench_supported_formats("read_full") is None
    assert _bench_format_compatible("read_full", "scx_auto") is True
    assert _bench_format_compatible("read_full", "h5ad_none") is True
    assert _bench_format_compatible("read_full", "tiledb_soma") is True
    assert _bench_format_compatible("read_full", "zarr_zstd") is True


def test_supported_formats_filters_cohort_grouping(isolated_work_dir):
    """The orchestrator must not submit cloud_push/h5ad_none — the static
    guard says cloud_push only runs on scx_auto, so the cohort builder
    should never even try the incompatible combination."""
    _FakeAutoExecutor.reset()
    _install_fake_submitit()

    _run_main_with_argv([
        "--datasets", "pbmc3k",
        "--formats", "scx_auto", "h5ad_none",
        "--benchmarks", "cloud_push", "read_full",
        "--skip-smoke",
    ])

    bench_subs = [
        s for s in _FakeAutoExecutor.submissions
        if s["fn_name"] == "_run_benchmark"
    ]
    by_label = {
        (s["fn_args"][0], s["fn_args"][1], s["fn_args"][2]): s
        for s in bench_subs
    }

    # cloud_push only submits for scx_auto.
    assert ("cloud_push", "pbmc3k", "scx_auto") in by_label
    assert ("cloud_push", "pbmc3k", "h5ad_none") not in by_label

    # read_full has no static guard — submits for both.
    assert ("read_full", "pbmc3k", "scx_auto") in by_label
    assert ("read_full", "pbmc3k", "h5ad_none") in by_label


def test_runner_capabilities_helper():
    """``_runner_capabilities`` resolves the runner's static (or instance-
    time) capability set without I/O. Verifies both happy paths and the
    defensive ``frozenset()`` fallback for unknown formats."""
    from benchmarks.comprehensive.scripts.run_parallel import _runner_capabilities

    scx_caps = _runner_capabilities("scx_auto")
    assert "cloud_read" in scx_caps
    assert "cloud_filtered" in scx_caps
    assert "cloud_metadata" in scx_caps
    assert "backed_mode" in scx_caps

    h5ad_caps = _runner_capabilities("h5ad_gzip")
    assert "cloud_read" not in h5ad_caps
    assert "backed_mode" not in h5ad_caps

    # Unknown format → empty set (defensive — don't crash cohort grouping).
    assert _runner_capabilities("definitely_not_a_format") == frozenset()


def test_required_capabilities_filters_cohort_grouping(isolated_work_dir):
    """The orchestrator must not submit cloud_read on h5ad_none — h5ad_runner
    doesn't declare the ``cloud_read`` capability."""
    _FakeAutoExecutor.reset()
    _install_fake_submitit()

    _run_main_with_argv([
        "--datasets", "pbmc3k",
        "--formats", "scx_auto", "h5ad_none",
        "--benchmarks", "cloud_read", "read_full",
        "--skip-smoke",
    ])

    bench_subs = [
        s for s in _FakeAutoExecutor.submissions
        if s["fn_name"] == "_run_benchmark"
    ]
    by_label = {
        (s["fn_args"][0], s["fn_args"][1], s["fn_args"][2]): s
        for s in bench_subs
    }

    # scx_runner declares cloud_read; h5ad_runner does not.
    assert ("cloud_read", "pbmc3k", "scx_auto") in by_label
    assert ("cloud_read", "pbmc3k", "h5ad_none") not in by_label

    # read_full is capability-unrestricted.
    assert ("read_full", "pbmc3k", "scx_auto") in by_label
    assert ("read_full", "pbmc3k", "h5ad_none") in by_label



# ---------------------------------------------------------------------------
# Throttle headroom + manifest-on-failure
# ---------------------------------------------------------------------------


def test_wait_under_pending_cap_reserves_headroom(monkeypatch: pytest.MonkeyPatch):
    """``headroom`` blocks until there is room for the whole cohort.

    The throttle must return only when ``active < cap - headroom`` (clamped
    to >= 1), not merely ``active < cap`` — otherwise a multi-task array
    overshoots ``QOSMaxSubmitJobPerUserLimit``.
    """
    import benchmarks.comprehensive.scripts.run_parallel as rp

    # Keep the episode logging deterministic across cases.
    monkeypatch.setattr(rp, "_cancel_dep_never_satisfied", lambda: 0)
    monkeypatch.setattr(rp.time, "sleep", lambda *_a, **_k: None, raising=False)

    def _patch_active(seq):
        """Return successive squeue counts, repeating the last forever."""
        values = list(seq)

        def _stub():
            return values.pop(0) if len(values) > 1 else values[0]

        monkeypatch.setattr(rp, "_count_active_user_jobs", _stub)

    def _reset_state():
        rp._THROTTLE_STATE["in_throttle"] = 0.0
        rp._THROTTLE_STATE["last_resumed_at"] = 0.0

    # headroom=0 → identical to the legacy ``active < cap`` boundary.
    _reset_state()
    _patch_active([99])
    rp._wait_under_pending_cap(100, kind="bench", headroom=0)  # 99 < 100 → return

    # headroom reserves room: active=85, cap=100, headroom=20 → threshold=80,
    # so it must block until the queue drains below 80.
    _reset_state()
    _patch_active([85, 70])
    rp._wait_under_pending_cap(100, kind="bench", headroom=20)  # blocks then 70<80

    # Returns immediately when there is already room for the cohort.
    _reset_state()
    _patch_active([70])
    rp._wait_under_pending_cap(100, kind="bench", headroom=20)  # 70 < 80 → return

    # Clamp: a cohort larger than the cap waits for a full drain (threshold
    # floored to 1) rather than spinning on an unsatisfiable threshold.
    _reset_state()
    _patch_active([0])
    rp._wait_under_pending_cap(5, kind="bench", headroom=100)  # 0 < 1 → return

    # cap <= 0 stays a hard no-op regardless of headroom (convert phase).
    _reset_state()
    _patch_active([10_000])
    rp._wait_under_pending_cap(0, kind="convert", headroom=50)


def test_manifest_written_on_qos_exhaustion(
    isolated_work_dir, monkeypatch: pytest.MonkeyPatch
):
    """A QOS-limit failure after retries still persists the manifest for the
    cohorts already submitted, then re-raises — so watch.py can track them."""
    _FakeAutoExecutor.reset()
    _install_fake_submitit()

    class _QOSFailingExecutor(_FakeAutoExecutor):
        """Succeeds on the first cohort, then raises QOS on every later call
        (exhausting the retry loop)."""

        _map_calls = 0

        def map_array(self, fn, *arg_sequences):
            _QOSFailingExecutor._map_calls += 1
            if _QOSFailingExecutor._map_calls >= 2:
                raise RuntimeError(
                    "sbatch: error: QOSMaxSubmitJobPerUserLimit"
                )
            return super().map_array(fn, *arg_sequences)

    sys.modules["submitit"].AutoExecutor = _QOSFailingExecutor

    import benchmarks.comprehensive.scripts.run_parallel as rp

    # Neutralise the retry-loop backoff so the test doesn't sleep 3×60s.
    monkeypatch.setattr(rp.time, "sleep", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(rp, "_count_active_user_jobs", lambda: 0)
    monkeypatch.setattr(rp, "_cancel_dep_never_satisfied", lambda: 0)

    # read_full on two formats → two cohorts; the first lands, the second
    # exhausts the QOS retry loop and raises.
    with pytest.raises(RuntimeError, match="QOSMaxSubmitJobPerUserLimit"):
        _run_main_with_argv([
            "--datasets", "pbmc3k",
            "--formats", "scx_auto", "h5ad_none",
            "--benchmarks", "read_full",
            "--skip-smoke",
        ])

    # The manifest exists and records the cohort that submitted before the
    # failure (one array task), so watch.py is not left blind.
    manifest_path = rp.LOGS_DIR / "run_manifest.json"
    assert manifest_path.exists(), f"Manifest missing at {manifest_path}"

    import json
    manifest = json.loads(manifest_path.read_text())
    labels = {e["label"] for e in manifest["submitted"]}
    assert labels, "Manifest should record the already-submitted cohort"
    assert all("read_full/pbmc3k/" in lbl for lbl in labels), labels
