"""
Minimal HTML snapshot renderer (Phase G.4).

Wraps the markdown body produced by ``markdown.generate_report()`` in a
self-contained HTML5 shell. No jinja2 / templating dependency — the
skeleton is a Python string literal and the markdown body is escaped +
embedded as a ``<pre>`` block with anchor targets parsed from the
``##`` / ``###`` headings so in-page navigation works.

The design goal is intentionally minimal: the markdown report is the
authoritative artifact; the HTML is a browsable snapshot that makes
rolling-dashboard navigation possible (via the previous-snapshot link
threaded through ``dashboard_history.json``). A richer rendering — real
markdown → HTML, syntax highlighting, trend charts — is Phase I.7
territory; Phase G.4 just lands the link structure so trend work is a
drop-in later.
"""

from __future__ import annotations

import datetime as _dt
import html as _html
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


_CSS = """
* { box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    max-width: 1100px;
    margin: 0 auto;
    padding: 1.5rem 2rem 4rem;
    color: #1f2328;
    line-height: 1.55;
}
header {
    border-bottom: 1px solid #d0d7de;
    padding-bottom: 0.75rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
}
header h1 {
    font-size: 1.5rem;
    margin: 0;
}
header .meta {
    font-size: 0.875rem;
    color: #656d76;
}
header .prev-link a {
    font-size: 0.875rem;
    text-decoration: none;
    color: #0969da;
}
header .prev-link a:hover { text-decoration: underline; }
nav.toc {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-bottom: 1.5rem;
    font-size: 0.9rem;
}
nav.toc ul { margin: 0.25rem 0 0 1rem; padding: 0; }
nav.toc a {
    color: #0969da;
    text-decoration: none;
}
nav.toc a:hover { text-decoration: underline; }
pre.report {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    overflow-x: auto;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
    font-size: 0.85rem;
    white-space: pre-wrap;
    word-wrap: break-word;
}
""".strip()


_HEADING_RE = re.compile(r"^(##+)\s+(.+)$", re.MULTILINE)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.strip().lower())
    slug = re.sub(r"[\s-]+", "-", slug)
    return slug.strip("-") or "section"


def _extract_toc(markdown_body: str) -> list[tuple[int, str, str]]:
    """Return ``(level, text, slug)`` for each ``##``/``###`` heading.

    The top-level ``#`` title is excluded; the TOC starts at ``##`` so
    Phase sections (§1, §2, ...) show up without the document header.
    """
    out = []
    seen: dict[str, int] = {}
    for m in _HEADING_RE.finditer(markdown_body):
        level = len(m.group(1))
        text = m.group(2).strip()
        slug = _slugify(text)
        # De-duplicate slugs with a numeric suffix so anchors remain unique.
        if slug in seen:
            seen[slug] += 1
            slug = f"{slug}-{seen[slug]}"
        else:
            seen[slug] = 1
        out.append((level, text, slug))
    return out


def _annotate_body(markdown_body: str, toc: list[tuple[int, str, str]]) -> str:
    """Inject HTML anchor tags above each heading for in-page navigation."""
    anchors_iter = iter(toc)

    def repl(m: re.Match) -> str:
        try:
            _lvl, _text, slug = next(anchors_iter)
        except StopIteration:
            return m.group(0)
        return f'<span id="{slug}"></span>{m.group(0)}'

    return _HEADING_RE.sub(repl, markdown_body)


def render_html(
    markdown_body: str,
    *,
    title: str = "SCX Benchmark Report",
    prev_url: str | None = None,
    generated_at: _dt.datetime | None = None,
) -> str:
    """Wrap ``markdown_body`` in a browsable HTML snapshot.

    The markdown body is embedded verbatim inside a ``<pre>`` block so
    readers get the raw report with table alignment intact. A small TOC
    auto-built from ``##``/``###`` headings anchors each section.
    """
    toc = _extract_toc(markdown_body)
    body_with_anchors = _annotate_body(markdown_body, toc)
    generated_at = generated_at or _dt.datetime.now()

    toc_html = ""
    if toc:
        toc_items = "\n".join(
            f'    <li style="margin-left: {(lvl - 2) * 1.25}rem">'
            f'<a href="#{slug}">{_html.escape(text)}</a></li>'
            for (lvl, text, slug) in toc
        )
        toc_html = (
            '<nav class="toc"><strong>Contents</strong><ul>\n'
            + toc_items + "\n</ul></nav>"
        )

    prev_link_html = ""
    if prev_url:
        prev_link_html = (
            f'<div class="prev-link"><a href="{_html.escape(prev_url)}">'
            "← previous snapshot</a></div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{_html.escape(title)}</title>
  <style>{_CSS}</style>
</head>
<body>
<header>
  <div>
    <h1>{_html.escape(title)}</h1>
    <div class="meta">Generated {generated_at.isoformat(timespec="seconds")}</div>
  </div>
  {prev_link_html}
</header>
{toc_html}
<pre class="report">{_html.escape(body_with_anchors)}</pre>
</body>
</html>
"""


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
