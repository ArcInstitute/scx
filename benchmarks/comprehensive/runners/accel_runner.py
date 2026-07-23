"""
AcceleratorRunner — shared host for `accel_*.py` benchmark modules.

Sibling to `FormatRunner` (not subclass — the accelerator dimensions
don't convert files / read backed / filter queries). Centralises:

* GPU availability detection (`AcceleratorRunner.has_gpu()`).
* Fixture loading + caching (`load_preprocessed`, `load_raw`) — one
  h5ad read per (dataset, n_comps) pair per worker process. Owned here
  and keyed by both dataset name and variant-specific hints (e.g.
  reference embeddings vs preprocessed AnnData).
* Timing utilities (`get_rss_mb`, `get_cpu_times`).

Benchmarks consume this by instantiating one per cell:

    from benchmarks.comprehensive.runners.accel_runner import AcceleratorRunner
    runner = AcceleratorRunner.instance()
    adata = runner.load_preprocessed(dataset, n_comps=50).adata

The `.instance()` class-method returns the process-wide singleton so
the fixture cache survives across benchmark invocations within one
submitit worker. Different submitit workers get their own fixture
caches (correct isolation — each worker pays the preprocessing cost
once per dataset).
"""

from __future__ import annotations

import contextlib
import logging
import os
import resource
from dataclasses import dataclass
from typing import Any, Optional

from benchmarks.comprehensive.rss import current_rss_mb

logger = logging.getLogger(__name__)


def cpu_profile_enabled() -> bool:
    """True iff `SCX_CPU_PROFILE` is set to a non-empty, non-``"0"`` value.

    This mirrors the Rust-side gate (`scx_format_io::profile::profile_enabled`),
    resolved once per process at profiler init; the env var must be exported
    **before** the worker imports pyscx.
    """
    v = os.environ.get("SCX_CPU_PROFILE", "")
    return bool(v) and v != "0"


@contextlib.contextmanager
def cpu_profile_capture(extras: dict[str, Any], prefix: str = "cpu_profile"):
    """Capture the CPU per-stage profiler breakdown for the wrapped op.

    Task 2.0 (the Phase-2 ranking oracle). When `SCX_CPU_PROFILE=1`, resets the
    process-global CPU profiler on entry and, on exit, writes a flattened
    per-stage breakdown into `extras` (so it lands in ``runs[].extra`` — the
    sanctioned free-form manifest channel, `results.RunMeasurement`). Keys are
    ``{prefix}_{bucket}_ms`` / ``_count`` / ``_bytes`` for buckets
    ``io``/``decode_scx1``/``decode_generic``/``reduction``/``marshalling``, plus
    ``{prefix}_enabled``. A no-op (zero overhead, no keys) when the profiler is
    disabled or pyscx is unavailable, so it is always safe to wrap a timed op.
    """
    if not cpu_profile_enabled():
        yield
        return
    try:
        import pyscx
    except Exception:  # pragma: no cover - pyscx always present in bench envs
        yield
        return
    # Degrade gracefully on a pyscx too old to carry the CPU profiler surface
    # (the functions landed in Phase-2 2.0) instead of raising AttributeError.
    if not hasattr(pyscx.accel, "cpu_profile_snapshot") or not hasattr(
        pyscx.accel, "cpu_profile_reset"
    ):
        logger.warning("cpu_profile_capture: pyscx lacks the CPU profiler surface; skipping")
        yield
        return
    pyscx.accel.cpu_profile_reset()
    try:
        yield
    finally:
        try:
            snap = pyscx.accel.cpu_profile_snapshot()
            extras[f"{prefix}_enabled"] = 1.0 if snap.get("enabled") else 0.0
            for bucket in (
                "io",
                "decode_scx1",
                "decode_generic",
                "reduction",
                "marshalling",
            ):
                st = snap.get(bucket) or {}
                extras[f"{prefix}_{bucket}_ms"] = float(st.get("ms", 0.0))
                extras[f"{prefix}_{bucket}_count"] = float(st.get("count", 0))
                extras[f"{prefix}_{bucket}_bytes"] = float(st.get("bytes", 0))
        except Exception as e:  # pragma: no cover - diagnostics only
            logger.warning("cpu_profile_capture snapshot failed: %s", e)


@dataclass
class PreprocessedFixture:
    """Result of `load_preprocessed` — the full AnnData plus inferred dims."""
    adata: Any  # anndata.AnnData
    n_comps: int
    n_obs: int
    n_vars: int


class AcceleratorRunner:
    """Shared host for accelerator benchmarks.

    Does NOT subclass `FormatRunner` — the accelerator dimensions don't
    convert / read / subset files. This class runs on already-loaded
    AnnData objects via `pyscx.accel.*` (or scanpy for reference
    comparisons).
    """

    name = "AcceleratorRunner"
    key = "accel_runner"

    _instance: Optional["AcceleratorRunner"] = None

    def __init__(self) -> None:
        self._cache: dict[tuple, Any] = {}
        self._has_pyscx: bool | None = None
        self._has_gpu: bool | None = None
        self._has_rapids_singlecell: bool | None = None

    @classmethod
    def instance(cls) -> "AcceleratorRunner":
        """Return the process-wide singleton. Cache survives across
        benchmarks within one submitit worker.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Capability probes (cached once per process)
    # ------------------------------------------------------------------

    def has_pyscx(self) -> bool:
        if self._has_pyscx is None:
            try:
                import pyscx  # noqa: F401
                self._has_pyscx = True
            except ImportError:
                self._has_pyscx = False
        return self._has_pyscx

    def has_gpu(self) -> bool:
        """True iff `pyscx.accel.gpu_info()` returns a non-None dict.

        `gpu_info()` is the canonical Python-side availability check
        (Rust's `gpu_available()` isn't re-exported). Returns False on
        CPU-only builds or when CUDA is unavailable.
        """
        if self._has_gpu is None:
            if not self.has_pyscx():
                self._has_gpu = False
            else:
                try:
                    import pyscx
                    self._has_gpu = pyscx.accel.gpu_info() is not None
                except Exception:
                    self._has_gpu = False
        return self._has_gpu

    def has_rapids_singlecell(self) -> bool:
        """True iff `rapids_singlecell` imports cleanly (V3 task 2.9).

        The GPU-scanpy competitor probe, cached once per process (the
        per-op `accel_*__rapids_singlecell_gpu` variants share one worker).
        Catches a broad `Exception`, not just `ImportError`: importing
        rapids-singlecell can fail with a CUDA-init `RuntimeError` on a
        host without a usable GPU, which is still "not usable" here.
        """
        if self._has_rapids_singlecell is None:
            try:
                import rapids_singlecell  # noqa: F401
                self._has_rapids_singlecell = True
            except Exception:
                self._has_rapids_singlecell = False
        return self._has_rapids_singlecell

    # ------------------------------------------------------------------
    # Fixtures — cached per (dataset, config) key
    # ------------------------------------------------------------------

    def load_raw(self, dataset: Any) -> Any:
        """Load the raw h5ad fixture — no preprocessing. Used by
        accel_preprocess / accel_hvg which do the preprocessing in the
        benchmarked op itself.
        """
        key = ("raw_adata", dataset.name)
        if key in self._cache:
            return self._cache[key]
        import anndata
        h5ad_path = dataset.h5ad_path
        if not h5ad_path.exists():
            raise FileNotFoundError(f"Source h5ad not found: {h5ad_path}")
        logger.info("AcceleratorRunner: loading raw %s …", dataset.name)
        adata = anndata.read_h5ad(str(h5ad_path))
        self._cache[key] = adata
        return adata

    def load_preprocessed(self, dataset: Any, n_comps: int = 50) -> PreprocessedFixture:
        """Load h5ad + normalize_total + log1p + HVG (2K genes).

        Shared between `accel_pca`, `accel_knn`, `accel_umap`,
        `accel_leiden` — each of those wants a preprocessed AnnData
        ready for PCA. Cached per (dataset, n_comps).
        """
        key = ("preprocessed", dataset.name, n_comps)
        if key in self._cache:
            return self._cache[key]

        import anndata
        import scanpy as sc

        h5ad_path = dataset.h5ad_path
        if not h5ad_path.exists():
            raise FileNotFoundError(f"Source h5ad not found: {h5ad_path}")

        logger.info("AcceleratorRunner: loading + preprocessing %s …", dataset.name)
        adata = anndata.read_h5ad(str(h5ad_path))
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        # Import QUERY_N_HVGS lazily to avoid circular imports.
        from benchmarks.comprehensive.config import QUERY_N_HVGS
        n_top = min(QUERY_N_HVGS, adata.n_vars)
        try:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=n_top, flavor="seurat_v3",
                subset=True, span=0.3 if adata.n_obs < 10_000 else 1.0,
            )
        except Exception:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top, subset=True)

        fx = PreprocessedFixture(
            adata=adata, n_comps=n_comps,
            n_obs=adata.n_obs, n_vars=adata.n_vars,
        )
        self._cache[key] = fx
        return fx

    def cache_get(self, *key_parts: Any) -> Any:
        """Generic cache access for ad-hoc entries (e.g. reference
        embedding / connectivity / partition that one benchmark produces
        and its sibling wants to reuse).
        """
        return self._cache.get(tuple(key_parts))

    def cache_put(self, *key_parts_then_value: Any) -> None:
        """`cache_put("ref_embedding", dataset.name, 50, arr)` — last
        positional arg is the value, preceding args are the key.
        """
        *key_parts, value = key_parts_then_value
        self._cache[tuple(key_parts)] = value

    # ------------------------------------------------------------------
    # Static timing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_rss_mb() -> float:
        """Back-compat shim — delegates to
        ``benchmarks.comprehensive.rss.current_rss_mb``."""
        return current_rss_mb()

    @staticmethod
    def get_cpu_times() -> tuple[float, float]:
        r = resource.getrusage(resource.RUSAGE_SELF)
        return r.ru_utime, r.ru_stime
