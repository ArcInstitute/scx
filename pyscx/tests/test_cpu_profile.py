"""CPU stage profiler surface (Phase-2 task 2.0).

The profiler is gated by `SCX_CPU_PROFILE` read once at process init, so these
tests only assert the Python surface shape / reset semantics — not that counters
populate (that requires the env set before import, exercised by
`benchmarks/scripts/profile_cpu_stages_backed.py`).
"""
import pyscx


EXPECTED_BUCKETS = ("io", "decode_scx1", "decode_generic", "reduction", "marshalling")


def test_cpu_profile_snapshot_shape():
    snap = pyscx.accel.cpu_profile_snapshot()
    assert "enabled" in snap
    assert isinstance(snap["enabled"], bool)
    for bucket in EXPECTED_BUCKETS:
        assert bucket in snap, f"missing bucket {bucket}"
        st = snap[bucket]
        assert set(st.keys()) == {"ms", "count", "bytes"}
        assert st["ms"] >= 0.0
        assert st["count"] >= 0
        assert st["bytes"] >= 0


def test_cpu_profile_reset_is_callable():
    # reset() must be a no-throw no-op regardless of enabled state.
    pyscx.accel.cpu_profile_reset()
    snap = pyscx.accel.cpu_profile_snapshot()
    # When profiling is disabled (default in the test process), all buckets zero.
    if not snap["enabled"]:
        for bucket in EXPECTED_BUCKETS:
            assert snap[bucket]["count"] == 0
            assert snap[bucket]["ms"] == 0.0
