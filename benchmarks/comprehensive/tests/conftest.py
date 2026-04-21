"""Make the project root importable so tests can ``from benchmarks...``.

The ``benchmarks`` directory has no top-level ``__init__.py`` and the
project has no editable-install of the benchmarks package, so bare
``pytest`` (outside of ``PYTHONPATH=.``) would otherwise fail to import
``benchmarks.comprehensive.*``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
