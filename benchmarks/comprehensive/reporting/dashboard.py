"""
HTML snapshot renderer for the rolling dashboard.

Delegates to the semantic ``HtmlRenderer`` from the report model.
The rendered output is a full HTML5 document with navigable heading
anchors, styled tables with ``<thead>``/``<tbody>``/``<caption>``,
and a "← previous snapshot" link threaded through
``dashboard_history.json``.

Phase 9 update: ``publish_dashboard.py`` now publishes
``BENCHMARK_REPORT.json`` (the JSON manifest) and
``LINT_WARNINGS.json`` alongside the HTML/MD/PDF outputs.
"""

from __future__ import annotations

import datetime as _dt
import html as _html
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


from benchmarks.comprehensive.reporting.report_model import Report, HtmlRenderer

def render_html(
    report: Report,
    *,
    title: str = "SCX Benchmark Report",
    prev_url: str | None = None,
    generated_at: _dt.datetime | None = None,
) -> str:
    """Render a full semantic HTML snapshot of the benchmark report.
    
    Delegates entirely to ``HtmlRenderer`` from the new report model.
    """
    return HtmlRenderer().render(report, prev_url=prev_url)


# ---------------------------------------------------------------------------
# Snapshot history (rolling dashboard)
# ---------------------------------------------------------------------------

HISTORY_FILENAME = "dashboard_history.json"


def read_history(reports_dir: Path) -> list[dict]:
    """Return the append-only history list from ``dashboard_history.json``.

    Each entry is ``{"timestamp": "...", "url": "...", "title": "..."}``.
    Missing file → empty list (first snapshot).
    """
    path = reports_dir / HISTORY_FILENAME
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("dashboard history at %s is not valid JSON; starting fresh", path)
        return []
    if not isinstance(data, list):
        return []
    return data


def append_history(
    reports_dir: Path,
    *,
    url: str,
    title: str = "SCX Benchmark Report",
    timestamp: _dt.datetime | None = None,
) -> dict:
    """Append an entry to ``dashboard_history.json`` and return it.

    ``url`` is stored as given — caller controls whether it's absolute
    (publish target) or relative (local file). The previous snapshot's
    ``url`` is used by ``render_html`` as the "← previous snapshot" link.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    history = read_history(reports_dir)
    entry = {
        "timestamp": (timestamp or _dt.datetime.now()).isoformat(timespec="seconds"),
        "url": url,
        "title": title,
    }
    history.append(entry)
    (reports_dir / HISTORY_FILENAME).write_text(json.dumps(history, indent=2))
    return entry


def previous_url(reports_dir: Path) -> str | None:
    """Return the most recently-recorded snapshot URL, if any."""
    history = read_history(reports_dir)
    if not history:
        return None
    return history[-1].get("url")
