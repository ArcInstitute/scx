"""§4.7 — route-affecting env vars are captured in result provenance.

`capture_run_provenance` records `SCX_GPU_DE_V2` / `SCX_GPU_DE_V3` /
`SCX_GPU_DE_V3_TRACE` so a benchmarked route (e.g. `gpu_csr`) can be
disambiguated after the fact — was v3 off, or on but fell back to CSR? The
values are embedded under `result.system["provenance"]` and serialized into
every raw result JSON, so the gate / dashboard can attribute a route without
re-running.
"""

from __future__ import annotations

from benchmarks.comprehensive.provenance import capture_run_provenance
from benchmarks.comprehensive.results import BenchmarkResult


def test_capture_provenance_reflects_gpu_de_env(monkeypatch) -> None:
    monkeypatch.setenv("SCX_GPU_DE_V2", "0")
    monkeypatch.setenv("SCX_GPU_DE_V3", "1")
    monkeypatch.setenv("SCX_GPU_DE_V3_TRACE", "1")
    prov = capture_run_provenance()
    assert prov["scx_gpu_de_v2"] == "0"
    assert prov["scx_gpu_de_v3"] == "1"
    assert prov["scx_gpu_de_v3_trace"] == "1"


def test_provenance_keys_default_empty_when_unset(monkeypatch) -> None:
    # Keys must always be present (never missing) so downstream consumers can
    # read them unconditionally; absent env vars resolve to "".
    for k in ("SCX_GPU_DE_V2", "SCX_GPU_DE_V3", "SCX_GPU_DE_V3_TRACE"):
        monkeypatch.delenv(k, raising=False)
    prov = capture_run_provenance()
    assert prov["scx_gpu_de_v2"] == ""
    assert prov["scx_gpu_de_v3"] == ""
    assert prov["scx_gpu_de_v3_trace"] == ""


def test_result_json_system_provenance_carries_gpu_de_v3(monkeypatch) -> None:
    monkeypatch.setenv("SCX_GPU_DE_V3", "1")
    # Pre-populate `system` so __post_init__ skips collect_system_info() but
    # still stamps system["provenance"] = capture_run_provenance().
    result = BenchmarkResult(
        benchmark="accel_de",
        format="accel_de__pyscx_pdex_ref_gpu",
        dataset="pbmc3k",
        system={"hostname": "test"},
    )
    prov = result.to_dict()["system"]["provenance"]
    assert prov["scx_gpu_de_v3"] == "1"
    assert "scx_gpu_de_v2" in prov
    assert "scx_gpu_de_v3_trace" in prov
