"""Doublet-caller benchmark runners for the comprehensive suite.

One module per external tool, each invoked as a subprocess into the conda env
that owns it (see :mod:`._tool_env`). Runners write the tool's **native**
column names — `scDblFinder.score` / `doublet_score` / … — rather than
canonical ones, so the benchmark exercises the real
``pyscx.doublet_import(tool=…)`` profile path instead of bypassing it.
"""

from __future__ import annotations

__all__ = ["_tool_env", "inject_doublets", "run_tool"]
