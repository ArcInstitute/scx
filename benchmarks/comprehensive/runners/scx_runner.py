"""SCX format benchmark runner (auto, none, scx1, zstd, lz4 codec variants)."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.comprehensive.runners.base import ConvertResult, FormatRunner, TimingResult

try:
    import pyscx

    _HAS_PYSCX = True
except ImportError:
    _HAS_PYSCX = False

try:
    import anndata

    _HAS_ANNDATA = True
except ImportError:
    _HAS_ANNDATA = False

try:
    import mudata  # noqa: F401

    _HAS_MUDATA = True
except ImportError:
    _HAS_MUDATA = False


_CODEC_NAMES = {
    "auto": ("SCX (auto)", "scx_auto"),
    "none": ("SCX (none)", "scx_none"),
    "scx1": ("SCX (scx1)", "scx_scx1"),
    "zstd": ("SCX (zstd)", "scx_zstd"),
    "lz4": ("SCX (lz4)", "scx_lz4"),
    "pcodec": ("SCX (pcodec)", "scx_pcodec"),
    # Phase K multimodal variants — name + key tags for the
    # comprehensive results pipeline.
    "_multimodal_per_modality_auto": (
        "SCX multimodal (per-modality auto)",
        "scx_multimodal_per_modality_auto",
    ),
    "_multimodal_uniform_auto": (
        "SCX multimodal (uniform auto)",
        "scx_multimodal_uniform_auto",
    ),
}


class ScxRunner(FormatRunner):
    """Benchmark runner for the SCX format with configurable codec."""

    capabilities: frozenset[str] = frozenset({
        "filtered_query",
        "backed_mode",
        "cloud_read",
        "cloud_subset",
        "cloud_push",
        "cloud_pull",
        "cloud_filtered",
    })

    def __init__(self, codec: str = "auto", codec_per_modality: bool = True) -> None:
        if codec not in _CODEC_NAMES:
            raise ValueError(
                f"Unsupported codec {codec!r}; "
                f"expected one of {list(_CODEC_NAMES)}"
            )
        self.codec = codec
        # Phase K.3.4: when False, route every modality through the
        # single-modality `select_codec` helper instead of
        # `select_codec_for_modality`. Only meaningful for the
        # multimodal compression sweep; ignored on single-modality
        # convert paths.
        self.codec_per_modality = codec_per_modality

    @property
    def name(self) -> str:
        # Phase K: multimodal variants use a synthetic codec key so the
        # name reflects "per-modality auto" vs "uniform auto" instead
        # of just "SCX (auto)".
        if self.codec == "auto" and not self.codec_per_modality:
            return _CODEC_NAMES["_multimodal_uniform_auto"][0]
        return _CODEC_NAMES[self.codec][0]

    @property
    def key(self) -> str:
        if self.codec == "auto" and not self.codec_per_modality:
            return _CODEC_NAMES["_multimodal_uniform_auto"][1]
        return _CODEC_NAMES[self.codec][1]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_pyscx() -> None:
        if not _HAS_PYSCX:
            raise RuntimeError(
                "pyscx is not installed. "
                "Install with: cd pyscx && maturin develop"
            )

    @staticmethod
    def _check_anndata() -> None:
        if not _HAS_ANNDATA:
            raise RuntimeError(
                "anndata is not installed. "
                "Install with: pip install anndata"
            )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def convert_from_h5ad(self, h5ad_path: str | Path, output_path: str | Path) -> ConvertResult:
        self._check_pyscx()
        self._check_anndata()

        h5ad_path = str(h5ad_path)
        output_path = str(output_path)

        self._gc_collect()
        u0, s0 = self._get_cpu_times()
        t0 = time.perf_counter()

        adata = anndata.read_h5ad(h5ad_path)
        pyscx.from_anndata(adata, output_path, codec=self.codec)

        wall = time.perf_counter() - t0
        u1, s1 = self._get_cpu_times()
        rss = self._get_rss_mb()

        output_size = os.path.getsize(output_path)
        throughput = (output_size / (1024 * 1024)) / wall if wall > 0 else 0.0

        return ConvertResult(
            wall_s=wall,
            peak_rss_mb=rss,
            output_size_bytes=output_size,
            write_throughput_mb_s=throughput,
            extra={"codec": self.codec},
        )

    def convert_from_h5mu(
        self, h5mu_path: str | Path, output_path: str | Path
    ) -> ConvertResult:
        """Phase K: read a `.h5mu` and write a multimodal SCX file via
        ``pyscx.from_mudata``. The ``codec_per_modality`` flag set in
        ``__init__`` flows through to the writer — when False, every
        modality routes through single-modality ``select_codec``
        (uniform-auto sweep variant); when True (default), per-modality
        codec routing applies (``select_codec_for_modality``)."""
        self._check_pyscx()
        if not _HAS_MUDATA:
            raise RuntimeError(
                "mudata is not installed. Install with: pip install mudata"
            )

        h5mu_path = str(h5mu_path)
        output_path = str(output_path)

        self._gc_collect()
        u0, s0 = self._get_cpu_times()
        t0 = time.perf_counter()

        mu = mudata.read_h5mu(h5mu_path)
        pyscx.from_mudata(
            mu,
            output_path,
            codec=self.codec,
            codec_per_modality=self.codec_per_modality,
        )

        wall = time.perf_counter() - t0
        u1, s1 = self._get_cpu_times()
        rss = self._get_rss_mb()

        output_size = os.path.getsize(output_path)
        throughput = (output_size / (1024 * 1024)) / wall if wall > 0 else 0.0

        # Per-modality codec assignment from the on-disk modality table
        # — useful for the per-modality codec sweep results.
        per_modality_codec_id: dict[str, int] = {}
        try:
            reader = pyscx.open(output_path)
            for name in reader.modality_names:
                mid = reader.modality_id(name)
                info = reader.modality_info(mid)
                if info is not None:
                    per_modality_codec_id[name] = int(info["default_codec_id"])
        except Exception:
            # Best-effort metadata; do not fail the convert if the
            # post-write read-back fails.
            pass

        return ConvertResult(
            wall_s=wall,
            peak_rss_mb=rss,
            output_size_bytes=output_size,
            write_throughput_mb_s=throughput,
            extra={
                "codec": self.codec,
                "codec_per_modality": self.codec_per_modality,
                "per_modality_codec_id": per_modality_codec_id,
            },
        )

    def read_full(self, path: str | Path) -> TimingResult:
        self._check_pyscx()

        def _read():
            ds = pyscx.open(str(path))
            adata = ds.to_anndata()
            # Force materialization of the expression matrix
            _ = adata.X

        _, timing = self.timed_run(_read)
        return timing

    def read_subset(
        self,
        path: str | Path,
        cell_indices: np.ndarray | list[int] | None = None,
        gene_indices: np.ndarray | list[int] | None = None,
    ) -> TimingResult:
        self._check_pyscx()

        def _read_subset():
            ds = pyscx.open(str(path))
            if cell_indices is not None:
                # Use backed mode for arbitrary cell index subsetting
                adata = ds.to_anndata(backed=True)
                X = adata.X[cell_indices]
                if gene_indices is not None:
                    X = X[:, gene_indices]
            elif gene_indices is not None:
                # Gene-only selection via query API
                q = ds.query().select_genes(gene_indices)
                X = q.collect().to_csr()
            else:
                X = ds.query().collect().to_csr()

        _, timing = self.timed_run(_read_subset)
        timing.extra = {"query_approach": "backed_index+select_genes"}
        return timing

    def file_size(self, path: str | Path) -> int:
        return os.path.getsize(path)

    # ------------------------------------------------------------------
    # Optional: backed mode
    # ------------------------------------------------------------------

    def read_backed(self, path: str | Path) -> TimingResult:
        self._check_pyscx()

        def _open_backed():
            ds = pyscx.open(str(path))
            adata = ds.to_anndata(backed=True)
            return adata

        _, timing = self.timed_run(_open_backed)
        timing.extra = {"mode": "backed"}
        return timing

    def read_backed_slice(
        self,
        path: str | Path,
        start: int,
        count: int,
    ) -> TimingResult:
        self._check_pyscx()

        def _backed_slice():
            ds = pyscx.open(str(path))
            adata = ds.to_anndata(backed=True)
            X_slice = adata.X[start : start + count]
            return X_slice

        _, timing = self.timed_run(_backed_slice)
        timing.extra = {"mode": "backed_slice", "start": start, "count": count}
        return timing

    # ------------------------------------------------------------------
    # Filtered query via SCX catalog pushdown
    # ------------------------------------------------------------------

    def read_filtered_query(
        self,
        path: str | Path,
        predicate,
    ) -> TimingResult:
        self._check_pyscx()
        from benchmarks.comprehensive.queries import (
            EqPredicate,
            GtPredicate,
            RandomSamplePredicate,
            sql_literal,
        )

        if isinstance(predicate, EqPredicate):
            expr = f"{predicate.column} == {sql_literal(predicate.value)}"
        elif isinstance(predicate, GtPredicate):
            expr = f"{predicate.column} > {predicate.threshold}"
        elif isinstance(predicate, RandomSamplePredicate):
            expr = None  # handled below — SCX has no SAMPLE pushdown
        else:
            raise TypeError(f"Unsupported predicate type: {type(predicate)!r}")

        def _filtered():
            ds = pyscx.open(str(path))
            if expr is not None:
                q = ds.query().filter_obs(expr)
                _ = q.collect().to_csr()
            else:
                # Random-sample: materialize via backed index slicing. SCX's
                # catalog pushdown doesn't support Bernoulli sampling, so
                # the honest native mechanism is a random-index read.
                import numpy as np

                assert isinstance(predicate, RandomSamplePredicate)
                n_obs = ds.n_obs
                rng = np.random.default_rng(predicate.seed)
                n_take = max(1, int(n_obs * predicate.fraction))
                cell_idx = np.sort(rng.choice(n_obs, size=n_take, replace=False))
                adata = ds.to_anndata(backed=True)
                _ = adata.X[cell_idx]

        _, timing = self.timed_run(_filtered)
        timing.extra = {
            "native_mechanism": "scx_pushdown"
            if expr is not None
            else "scx_backed_index",
            "predicate": predicate.name,
        }
        return timing

    # ------------------------------------------------------------------
    # Cloud operations (Phase C — GCP only)
    # ------------------------------------------------------------------

    def push(self, local_path: str | Path, cloud_url: str) -> TimingResult:
        """Upload a local ``.scx`` file to ``cloud_url`` via ``pyscx.push``.

        ``pyscx.push`` returns a stats dict that already includes
        ``elapsed_secs`` / ``throughput_mbps``; we capture those via
        ``timed_run`` for uniform wall/RSS reporting but forward the native
        numbers under ``extra`` so downstream analysis can compare.
        """
        self._check_pyscx()
        stats_holder: dict = {}

        def _push():
            stats_holder["stats"] = pyscx.push(str(local_path), cloud_url)

        _, timing = self.timed_run(_push)
        stats = stats_holder["stats"]
        timing.extra = {
            "provider": "gcs",
            "native_mechanism": "scx_push",
            "bytes_uploaded": int(stats.get("bytes_uploaded", 0)),
            "sections_uploaded": int(stats.get("sections_uploaded", 0)),
            "pyscx_elapsed_secs": float(stats.get("elapsed_secs", timing.wall_s)),
            "throughput_mbps": float(stats.get("throughput_mbps", 0.0)),
        }
        return timing

    def pull(self, cloud_url: str, local_path: str | Path) -> TimingResult:
        """Download ``cloud_url`` to a local ``.scx`` file via ``pyscx.pull``."""
        self._check_pyscx()
        stats_holder: dict = {}

        def _pull():
            stats_holder["stats"] = pyscx.pull(cloud_url, str(local_path))

        _, timing = self.timed_run(_pull)
        stats = stats_holder["stats"]
        timing.extra = {
            "provider": "gcs",
            "native_mechanism": "scx_pull",
            "bytes_downloaded": int(stats.get("bytes_downloaded", 0)),
            "sections_downloaded": int(stats.get("sections_downloaded", 0)),
            "pyscx_elapsed_secs": float(stats.get("elapsed_secs", timing.wall_s)),
            "throughput_mbps": float(stats.get("throughput_mbps", 0.0)),
        }
        return timing

    def read_cloud(self, cloud_url: str) -> TimingResult:
        """Pull the full dataset and materialize it in-memory.

        SCX's native cloud read path is pull-then-read: exploded ``.scxd/``
        shards stream into a local ``.scx`` which ``read_full`` then
        decompresses into an in-memory CSR. The timing includes both phases
        so the benchmark reflects the end-to-end user experience.

        Tmpdir creation/teardown happens OUTSIDE the timed region so
        filesystem cleanup of the pulled file does not inflate SCX's
        wall-clock against competitors that don't pay that cost.
        """
        import tempfile

        self._check_pyscx()

        tmp = tempfile.mkdtemp(prefix="scx_cloud_read_")
        try:
            local = os.path.join(tmp, "pulled.scx")

            def _read_cloud():
                pyscx.pull(cloud_url, local)
                ds = pyscx.open(local)
                adata = ds.to_anndata()
                _ = adata.X

            _, timing = self.timed_run(_read_cloud)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        timing.extra = {
            "provider": "gcs",
            "native_mechanism": "scx_pull_and_read",
            "telemetry": "phase_f_deferred",
        }
        return timing

    def read_cloud_subset(
        self,
        cloud_url: str,
        cell_indices: np.ndarray | list[int] | None = None,
        gene_indices: np.ndarray | list[int] | None = None,
    ) -> TimingResult:
        """Subset read from cloud via pull-then-local-subset.

        Phase C intentionally keeps this simple — full pull then local
        subset. Phase F (``cloud_reader_vs_pull``) measures the
        ``pyscx.open_cloud`` range-read path explicitly.
        """
        import tempfile

        self._check_pyscx()

        tmp = tempfile.mkdtemp(prefix="scx_cloud_subset_")
        try:
            local = os.path.join(tmp, "pulled.scx")

            def _subset():
                pyscx.pull(cloud_url, local)
                ds = pyscx.open(local)
                if cell_indices is not None:
                    adata = ds.to_anndata(backed=True)
                    X = adata.X[cell_indices]
                    if gene_indices is not None:
                        X = X[:, gene_indices]
                elif gene_indices is not None:
                    q = ds.query().select_genes(gene_indices)
                    X = q.collect().to_csr()
                else:
                    X = ds.query().collect().to_csr()
                _ = X

            _, timing = self.timed_run(_subset)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        timing.extra = {
            "provider": "gcs",
            "native_mechanism": "scx_pull_and_subset",
            "telemetry": "phase_f_deferred",
        }
        return timing

    def read_cloud_metadata(self, cloud_url: str) -> TimingResult:
        """Metadata-only open via ``pyscx.open_cloud`` (no full pull).

        Used by the ``cloud_metadata`` benchmark to measure single-GET
        latency on cloud-optimized ``.scx`` / exploded ``.scxd``.
        """
        self._check_pyscx()

        def _open():
            handle = pyscx.open_cloud(cloud_url)
            # Touch the common metadata accessors so the timing reflects the
            # full open + catalog-parse rather than just URL resolution.
            _ = handle.n_obs
            _ = handle.n_vars
            _ = handle.nnz
            _ = handle.shard_count

        _, timing = self.timed_run(_open)
        timing.extra = {
            "provider": "gcs",
            "native_mechanism": "scx_open_cloud",
            "telemetry": "phase_f_deferred",
        }
        return timing

    def read_cloud_filtered_query(
        self,
        cloud_url: str,
        predicate,
    ) -> TimingResult:
        """Pull the dataset then apply the predicate locally.

        SCX's catalog-pushdown filter is designed for local reads; pushing
        predicates through a range-read on ``pyscx.open_cloud`` is explicitly
        scoped to Phase F.1. For Phase D.4 parity, the benchmark measures
        the honest end-to-end wall-clock a user would see today: pull then
        local filter. Mechanism tag ``scx_pull_and_filter`` distinguishes
        this from the future native-pushdown variant.
        """
        import tempfile

        self._check_pyscx()

        tmp = tempfile.mkdtemp(prefix="scx_cloud_filter_")
        try:
            local = os.path.join(tmp, "pulled.scx")

            def _pull_and_filter():
                pyscx.pull(cloud_url, local)
                # Re-enter the local filtered_query via ``self`` so mechanism
                # tagging and error-handling stay centralized.
                return self.read_filtered_query(local, predicate)

            local_timing, timing = self.timed_run(_pull_and_filter)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        # The timing returned by timed_run covers pull+filter; the inner
        # local_timing is informational. Carry the predicate name through.
        local_extra = local_timing.extra or {}
        timing.extra = {
            "provider": "gcs",
            "native_mechanism": "scx_pull_and_filter",
            "predicate": local_extra.get("predicate", predicate.name),
            "inner_filter_wall_s": round(local_timing.wall_s, 6),
            "telemetry": "phase_f_deferred",
        }
        return timing
