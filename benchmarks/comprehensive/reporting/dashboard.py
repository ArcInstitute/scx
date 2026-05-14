"""
HTML snapshot renderer for the rolling dashboard.

Delegates to the semantic ``HtmlRenderer`` from the report model.
The rendered output is a full HTML5 document with navigable heading
anchors, styled tables with ``<thead>``/``<tbody>``/``<caption>``,
and a "← previous snapshot" link threaded through
``dashboard_history.json``.

``publish_dashboard.py`` now publishes
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
    report: Report | str,
    *,
    title: str = "SCX Benchmark Report",
    prev_url: str | None = None,
    generated_at: _dt.datetime | None = None,
) -> str:
    """Render a full semantic HTML snapshot of the benchmark report.

    Accepts either a ``Report`` model (used by the main report pipeline)
    or a plain markdown **string** (used by ``landing.py`` and other
    lightweight callers that don't build a full report AST).

    When *report* is a ``Report``, delegates to ``HtmlRenderer``.
    When *report* is a ``str``, wraps the raw text in a styled HTML page
    with the same CSS chrome (header, prev-link, body).
    """
    if isinstance(report, Report):
        return HtmlRenderer().render(report, prev_url=prev_url)

    # Legacy path: plain markdown/text string → simple HTML wrapper.
    body_text = report  # it's a str
    ts = (generated_at or _dt.datetime.now()).isoformat(timespec="seconds")
    prev_link = (
        f'<div class="prev-link"><a href="{_html.escape(prev_url)}">'
        f"&larr; previous snapshot</a></div>"
        if prev_url
        else ""
    )
    css = (
        "* { box-sizing: border-box; }"
        " body { font-family: -apple-system, BlinkMacSystemFont,"
        ' "Segoe UI", Helvetica, Arial, sans-serif;'
        " max-width: 1100px; margin: 0 auto;"
        " padding: 1.5rem 2rem 4rem; color: #1f2328; line-height: 1.55; }"
        " header { border-bottom: 1px solid #d0d7de;"
        " padding-bottom: 0.75rem; margin-bottom: 1.25rem;"
        " display: flex; align-items: baseline;"
        " justify-content: space-between; gap: 1rem; flex-wrap: wrap; }"
        " header h1 { font-size: 1.5rem; margin: 0; }"
        " pre { white-space: pre-wrap; word-break: break-word; }"
    )
    return (
        f"<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        f"  <meta charset=\"UTF-8\">\n"
        f"  <title>{_html.escape(title)}</title>\n"
        f"  <style>{css}</style>\n"
        f"</head>\n<body>\n"
        f"<header>\n"
        f"  <div>\n    <h1>{_html.escape(title)}</h1>\n"
        f'    <div class="meta">Generated {ts}</div>\n'
        f"  </div>\n  {prev_link}\n</header>\n"
        f"<pre>{_html.escape(body_text)}</pre>\n"
        f"</body>\n</html>"
    )


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
