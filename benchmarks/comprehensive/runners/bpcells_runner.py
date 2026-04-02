"""BPCells format benchmark runner.

BPCells is an R-native format, so all operations delegate to an R subprocess
via Rscript. Communication uses JSON over stdin/stdout.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from benchmarks.comprehensive.runners.base import ConvertResult, FormatRunner, TimingResult

logger = logging.getLogger(__name__)

# Path to the companion R script (resolved relative to project layout).
_R_SCRIPT = Path(__file__).resolve().parents[2] / "r_scripts" / "benchmark_bpcells.R"

# Conda environment containing R + BPCells.  Override via BPCELLS_CONDA_PREFIX.
_CONDA_PREFIX = os.environ.get(
    "BPCELLS_CONDA_PREFIX",
    "/home/nickyoungblut/miniforge3/envs/rscx",
)

# Rscript binary inside the conda env.
_RSCRIPT = os.path.join(_CONDA_PREFIX, "bin", "Rscript")


class BPCellsRunner(FormatRunner):
    """Benchmark runner for BPCells (R-native bitpacking format).

    Invokes Rscript from the ``rscx`` conda environment directly,
    injecting the env's ``bin/`` into ``PATH`` so the compiler toolchain
    is found.  Override the env prefix with ``BPCELLS_CONDA_PREFIX``.
    """

    def __init__(self) -> None:
        if not Path(_RSCRIPT).exists():
            raise RuntimeError(
                f"Rscript not found at {_RSCRIPT}. "
                "Set BPCELLS_CONDA_PREFIX to the conda env containing R + BPCells."
            )
        if not _R_SCRIPT.exists():
            raise FileNotFoundError(
                f"R helper script not found at {_R_SCRIPT}"
            )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "BPCells"

    @property
    def key(self) -> str:
        return "bpcells"

    # ------------------------------------------------------------------
    # R subprocess helper
    # ------------------------------------------------------------------

    def _call_r(self, config: dict) -> dict:
        """Send *config* as JSON to the R helper and return parsed JSON output.

        Raises
        ------
        RuntimeError
            If R exits with a non-zero code or returns unparseable output.
        """
        config_json = json.dumps(config)
        logger.debug("BPCells R call: %s", config.get("operation"))

        # Build an env that puts the conda env's bin/ on PATH so the
        # compiler toolchain and shared libraries are found.
        env = os.environ.copy()
        conda_bin = os.path.join(_CONDA_PREFIX, "bin")
        conda_lib = os.path.join(_CONDA_PREFIX, "lib")
        env["PATH"] = conda_bin + os.pathsep + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = conda_lib + os.pathsep + env.get("LD_LIBRARY_PATH", "")

        proc = subprocess.run(
            [_RSCRIPT, "--vanilla", str(_R_SCRIPT)],
            input=config_json,
            capture_output=True,
            text=True,
            timeout=3600,  # 1-hour timeout for large datasets
            env=env,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"BPCells R script failed (exit {proc.returncode}).\n"
                f"--- stderr ---\n{proc.stderr}\n"
                f"--- stdout ---\n{proc.stdout}"
            )

        # The R script writes a single JSON object to stdout.
        # Strip any non-JSON preamble (e.g., R startup messages).
        stdout = proc.stdout.strip()
        # Find the last JSON object in case R prints warnings before it.
        brace_start = stdout.rfind("{")
        brace_end = stdout.rfind("}")
        if brace_start == -1 or brace_end == -1 or brace_end < brace_start:
            raise RuntimeError(
                f"BPCells R script produced no JSON output.\n"
                f"--- stdout ---\n{stdout}\n"
                f"--- stderr ---\n{proc.stderr}"
            )
        json_str = stdout[brace_start:brace_end + 1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse JSON from R script: {exc}\n"
                f"--- raw output ---\n{json_str}"
            ) from exc

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def convert_from_h5ad(
        self, h5ad_path: str | Path, output_path: str | Path
    ) -> ConvertResult:
        result = self._call_r(
            {
                "operation": "convert",
                "h5ad_path": str(h5ad_path),
                "output_path": str(output_path),
            }
        )

        output_size = self._dir_size(output_path)
        wall = result["wall_s"]
        throughput = (output_size / (1024 * 1024)) / wall if wall > 0 else 0.0

        return ConvertResult(
            wall_s=wall,
            peak_rss_mb=result.get("peak_rss_kb", 0) / 1024.0,
            output_size_bytes=output_size,
            write_throughput_mb_s=throughput,
        )

    def read_full(self, path: str | Path) -> TimingResult:
        result = self._call_r(
            {
                "operation": "read_full",
                "path": str(path),
            }
        )
        return TimingResult(
            wall_s=result["wall_s"],
            peak_rss_mb=result.get("peak_rss_kb", 0) / 1024.0,
        )

    def read_subset(
        self,
        path: str | Path,
        cell_indices: np.ndarray | list[int] | None = None,
        gene_indices: np.ndarray | list[int] | None = None,
    ) -> TimingResult:
        config: dict = {
            "operation": "read_subset",
            "path": str(path),
        }

        # R uses 1-based indexing: convert from 0-based Python indices.
        if cell_indices is not None:
            idx = np.asarray(cell_indices, dtype=np.int64) + 1
            config["cell_indices"] = idx.tolist()
        if gene_indices is not None:
            idx = np.asarray(gene_indices, dtype=np.int64) + 1
            config["gene_indices"] = idx.tolist()

        result = self._call_r(config)
        return TimingResult(
            wall_s=result["wall_s"],
            peak_rss_mb=result.get("peak_rss_kb", 0) / 1024.0,
        )

    def file_size(self, path: str | Path) -> int:
        return self._dir_size(path)
