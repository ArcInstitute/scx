"""Per-tool environment resolution for the doublet-interop benchmark.

Every doublet caller lives in a different conda env, and the benchmark itself
runs in whichever env the orchestrator picked. ``run_parallel._env_for_format``
routes by **format key**, not benchmark name, so an SCX-only benchmark always
lands in ``scx-bench`` — which has neither R scDblFinder nor scikit-image.
Rather than teach the shared orchestrator about one benchmark, each tool is
invoked as a subprocess into the env that owns it. That is the same shape
``runners/bpcells_runner.py`` already uses to reach R.

The side benefit is the reason this module exists as more than a path helper:
recording *which* environment produced each tool's numbers is a Phase-8
requirement, and a subprocess boundary makes it a fact we can read off the
interpreter rather than a claim in a comment.

Every path is overridable so a different machine (or a fresh env name) needs no
code change:

    SCX_DOUBLET_SCDBLFINDER_PREFIX=/path/to/env
    SCX_DOUBLET_SCRUBLET_PREFIX=/path/to/env
    SCX_DOUBLET_SOLO_PREFIX=/path/to/env
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ToolEnv",
    "TOOL_ENVS",
    "ToolUnavailable",
    "resolve",
    "probe",
    "probe_all",
]


class ToolUnavailable(RuntimeError):
    """A tool's environment or package is missing.

    Carries the env name and the remedy, because the failure a benchmark
    operator actually hits is "which conda env was I supposed to create?" and a
    bare FileNotFoundError on an interpreter path does not answer it.
    """


def _conda_root() -> Path:
    """Locate the conda installation without hardcoding a home directory."""
    exe = os.environ.get("CONDA_EXE", "")
    if exe:
        # .../miniforge3/bin/conda -> .../miniforge3
        return Path(exe).parent.parent
    return Path.home() / "miniforge3"


@dataclass(frozen=True)
class ToolEnv:
    """Where one doublet caller lives, and how to start it."""

    tool: str
    kind: str          # "python" | "r"
    env_name: str
    package: str       # the import/library name whose presence means "installed"
    install_hint: str

    @property
    def prefix(self) -> Path:
        override = os.environ.get(f"SCX_DOUBLET_{self.tool.upper()}_PREFIX", "")
        if override:
            return Path(override)
        return _conda_root() / "envs" / self.env_name

    @property
    def interpreter(self) -> Path:
        exe = "Rscript" if self.kind == "r" else "python"
        return self.prefix / "bin" / exe


# scDblFinder rides in the `rscx` env — the same one `bpcells_runner.py`
# defaults to, and the one Phase 6 installed bioconductor-scdblfinder into
# alongside rscx itself (which the R runner needs to read SCX directly).
#
# Scrublet is `sc.pp.scrublet`, so it needs scanpy AND scikit-image: without
# skimage it raises only when `threshold=None`, i.e. exactly in the
# auto-thresholding mode a benchmark wants.
#
# Solo needs scVI, which is installed nowhere here on purpose —
# tasks/DOUBLET-DETECTION.md keeps it "optional … in a separate optional
# benchmark environment". Its runner reports unavailable rather than failing.
TOOL_ENVS: dict[str, ToolEnv] = {
    "scdblfinder": ToolEnv(
        tool="scdblfinder",
        kind="r",
        env_name="rscx",
        package="scDblFinder",
        install_hint=(
            "conda install -n rscx -c bioconda -c conda-forge "
            "bioconductor-scdblfinder"
        ),
    ),
    "scrublet": ToolEnv(
        tool="scrublet",
        kind="python",
        env_name="scx-bench",
        package="skimage",
        install_hint=(
            "conda install -n scx-bench -c conda-forge scikit-image "
            "(sc.pp.scrublet needs it for auto-thresholding)"
        ),
    ),
    "solo": ToolEnv(
        tool="solo",
        kind="python",
        env_name="scx-bench-gpu",
        package="scvi",
        install_hint=(
            "optional; needs scvi-tools in a separate environment — see "
            "tasks/DOUBLET-DETECTION.md § Baseline tools to compare"
        ),
    ),
}


def resolve(tool: str) -> ToolEnv:
    """Return the :class:`ToolEnv` for *tool*, or raise naming the env."""
    try:
        env = TOOL_ENVS[tool]
    except KeyError:
        raise ToolUnavailable(
            f"unknown doublet tool {tool!r}; known tools are "
            f"{sorted(TOOL_ENVS)}"
        ) from None
    if not env.interpreter.exists():
        raise ToolUnavailable(
            f"{tool}: no interpreter at {env.interpreter}. Expected the "
            f"{env.env_name!r} conda env. Create it, or point "
            f"SCX_DOUBLET_{tool.upper()}_PREFIX at the env that has the tool. "
            f"To install: {env.install_hint}"
        )
    return env


_PY_PROBE = """
import json, sys
out = {"interpreter": sys.executable, "versions": {}}
for mod in %r:
    try:
        m = __import__(mod)
        out["versions"][mod] = getattr(m, "__version__", "unknown")
    except Exception as e:
        out["available"] = False
        out["reason"] = "%%s: %%s" %% (type(e).__name__, e)
        print(json.dumps(out)); raise SystemExit(0)
out["available"] = True
print(json.dumps(out))
"""

_R_PROBE = """
out <- list(interpreter = file.path(R.home("bin"), "Rscript"), versions = list())
for (pkg in c(%s)) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    out$available <- FALSE
    out$reason <- paste0("R package '", pkg, "' is not installed")
    cat(jsonlite::toJSON(out, auto_unbox = TRUE)); quit(save = "no")
  }
  out$versions[[pkg]] <- as.character(packageVersion(pkg))
}
out$available <- TRUE
cat(jsonlite::toJSON(out, auto_unbox = TRUE))
"""

# What each tool needs present for its runner to work at all. Listed per tool
# rather than derived from `package` because the R runner also needs rscx (it
# reads the SCX file directly) and jsonlite (the stdin/stdout protocol), and a
# probe that checked only scDblFinder would call the env healthy right up until
# the runner failed on a missing rscx.
_PROBE_PACKAGES: dict[str, list[str]] = {
    "scdblfinder": ["scDblFinder", "SingleCellExperiment", "rscx", "jsonlite"],
    "scrublet": ["scanpy", "skimage", "anndata", "pandas"],
    "solo": ["scvi", "anndata"],
}


def probe(tool: str, timeout: float = 300.0) -> dict:
    """Report whether *tool* can actually run, and under what versions.

    Never raises for an unavailable tool — returns ``available: False`` with a
    reason, so a caller can record a legible skip. Only a malformed probe
    (which is a bug here, not a missing dependency) surfaces as an exception.
    """
    record: dict = {"tool": tool}
    try:
        env = resolve(tool)
    except ToolUnavailable as exc:
        record.update(available=False, reason=str(exc))
        return record

    record.update(env_name=env.env_name, prefix=str(env.prefix),
                  interpreter=str(env.interpreter), kind=env.kind)
    packages = _PROBE_PACKAGES.get(tool, [env.package])

    if env.kind == "r":
        quoted = ", ".join(f'"{p}"' for p in packages)
        cmd = [str(env.interpreter), "--vanilla", "-e", _R_PROBE % quoted]
    else:
        cmd = [str(env.interpreter), "-c", _PY_PROBE % (packages,)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        record.update(available=False,
                      reason=f"probe timed out after {timeout}s")
        return record

    payload = _last_json_object(proc.stdout)
    if payload is None:
        record.update(
            available=False,
            reason=(f"probe returned no JSON (exit {proc.returncode}): "
                    f"{proc.stderr.strip()[-400:] or proc.stdout.strip()[-400:]}"),
        )
        return record
    record.update(payload)
    record.setdefault("available", False)
    return record


def probe_all(tools: list[str] | None = None) -> dict[str, dict]:
    """Probe several tools, for the benchmark's environment record."""
    return {t: probe(t) for t in (tools or list(TOOL_ENVS))}


def _last_json_object(text: str) -> dict | None:
    """Parse the last top-level JSON object in *text*.

    R prints startup notices and package load messages before the payload, and
    scanpy is fond of writing to stdout too, so the protocol is "the last
    top-level object wins".

    Scanned left to right with ``raw_decode`` rather than by ``rfind("{")``,
    which is what ``bpcells_runner`` does and which is only correct for a flat
    payload: with a nested object the last ``{`` is the *inner* brace, so
    ``{"versions": {"scanpy": "1.12"}}`` slices to
    ``{"scanpy": "1.12"}}`` and fails to parse. Every run record here carries a
    nested ``versions`` map, so that path would have failed on the first real
    tool invocation rather than in a test.
    """
    text = (text or "").strip()
    if not text:
        return None
    decoder = json.JSONDecoder()
    best: dict | None = None
    i = 0
    while True:
        i = text.find("{", i)
        if i == -1:
            return best
        try:
            parsed, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            # Not the start of a valid object (a brace in a log line); step
            # past it and keep looking.
            i += 1
            continue
        if isinstance(parsed, dict):
            best = parsed
        # Jump past what we just consumed, so a nested object is never
        # re-examined as a candidate in its own right.
        i = end
