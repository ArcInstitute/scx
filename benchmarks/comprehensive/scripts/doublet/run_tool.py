"""Invoke one doublet caller in its own environment.

The seam between the benchmark module (which runs in whatever env the
orchestrator picked) and the tool runners (which each need their own). Config
goes in as JSON on stdin, the run record comes back as JSON on stdout — the
protocol ``runners/bpcells_runner.py`` established for R and which the Python
runners here reuse so there is exactly one shape to reason about.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ._tool_env import ToolUnavailable, _last_json_object, resolve

__all__ = ["RUNNER_SCRIPTS", "INPUT_KIND", "run_tool"]

_HERE = Path(__file__).resolve().parent

RUNNER_SCRIPTS: dict[str, Path] = {
    "scdblfinder": _HERE / "scdblfinder_runner.R",
    "scrublet": _HERE / "scrublet_runner.py",
    "solo": _HERE / "solo_runner.py",
}

# What each tool is handed. R has no h5ad reader in the `rscx` env (no
# zellkonverter, no anndata), and reading SCX directly through rscx is the
# shorter path Phase 6 built and verified — so the R tool takes the .scx file
# and does its own batch filter, while the Python tools take the per-batch
# h5ad that `export_batches` wrote. Both are the ecosystem's real entry point,
# which is the point: the benchmark should measure the path a user would take.
INPUT_KIND: dict[str, str] = {
    "scdblfinder": "scx",
    "scrublet": "h5ad",
    "solo": "h5ad",
}


def run_tool(
    tool: str,
    *,
    out_csv: Path,
    scx_path: Path | None = None,
    h5ad_path: Path | None = None,
    batch_key: str | None = None,
    batch: str | None = None,
    seed: int = 0,
    timeout: float = 14400.0,
    extra: dict | None = None,
) -> dict:
    """Run *tool* on one batch and return its run record.

    Returns a dict that always carries ``available``. An optional tool whose
    dependency is missing comes back ``available=False`` with a reason rather
    than raising — a benchmark that dies on an absent optional comparator is
    worse than one that records why it did not run. A tool that is *present*
    and fails does raise, because that is a real failure.
    """
    if tool not in RUNNER_SCRIPTS:
        raise ToolUnavailable(
            f"unknown doublet tool {tool!r}; known tools are "
            f"{sorted(RUNNER_SCRIPTS)}"
        )
    script = RUNNER_SCRIPTS[tool]
    if not script.exists():
        raise FileNotFoundError(f"runner script missing: {script}")

    try:
        env = resolve(tool)
    except ToolUnavailable as exc:
        return {"tool": tool, "available": False, "reason": str(exc)}

    kind = INPUT_KIND[tool]
    config: dict = {
        "out_csv": str(out_csv),
        "batch_key": batch_key,
        "batch": batch,
        "seed": int(seed),
    }
    if kind == "scx":
        if scx_path is None:
            raise ValueError(f"{tool} needs scx_path")
        config["scx_path"] = str(scx_path)
    else:
        if h5ad_path is None:
            raise ValueError(f"{tool} needs h5ad_path")
        config["h5ad_path"] = str(h5ad_path)
    if extra:
        config.update(extra)

    if env.kind == "r":
        cmd = [str(env.interpreter), "--vanilla", str(script)]
    else:
        cmd = [str(env.interpreter), str(script)]

    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(config),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        # Same shape as the failure path below: a bare TimeoutExpired names
        # neither the tool nor the environment, and on a multi-hour capture
        # that is the whole diagnosis.
        raise RuntimeError(
            f"{tool} runner timed out after {timeout}s in env "
            f"{env.env_name!r}"
            + (f", batch {batch!r}" if batch is not None else "")
        ) from exc

    record = _last_json_object(proc.stdout)

    # An optional tool reporting its own absence exits 0 with available=False.
    # Honour that before treating anything as an error.
    if record is not None and record.get("available") is False:
        record.setdefault("tool", tool)
        record.setdefault("reason", "reported unavailable")
        return record

    if proc.returncode != 0 or record is None or "error" in (record or {}):
        detail = (record or {}).get("error") if record else None
        raise RuntimeError(
            f"{tool} runner failed (exit {proc.returncode}) in env "
            f"{env.env_name!r}"
            + (f", batch {batch!r}" if batch is not None else "")
            + f".\n--- reported ---\n{detail or '(no JSON payload)'}"
            f"\n--- stderr (tail) ---\n{proc.stderr.strip()[-2000:]}"
        )

    record.setdefault("tool", tool)
    record["available"] = True
    record["env_name"] = env.env_name
    record["interpreter"] = str(env.interpreter)

    written = Path(record.get("out_csv", out_csv))
    if not written.exists():
        raise RuntimeError(
            f"{tool} runner reported success but wrote no table at {written}"
        )
    return record
