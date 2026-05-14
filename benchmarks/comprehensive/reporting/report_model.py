"""
Report data model and semantic renderers.

Provides typed AST-like blocks for the benchmark report (Report, Chapter,
Section, Table, Figure, Text, Callout) and renderers for Markdown, HTML,
and JSON. This replaces the legacy f-string approach in ``markdown.py``
and the ``<pre>`` HTML snapshot approach in ``dashboard.py``.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from benchmarks.comprehensive.reporting.result_store import SourceRef

# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------


class BlockType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    CALLOUT = "callout"
    COMMENTARY = "commentary"


@dataclass
class Block:
    """Base class for all report blocks."""
    type: BlockType


@dataclass
class TextBlock(Block):
    """A block of markdown text."""
    content: str
    type: BlockType = field(default=BlockType.TEXT, init=False)


@dataclass
class CalloutBlock(Block):
    """A highlighted callout or note (e.g., executive summary point)."""
    content: str
    level: str = "info"  # "info", "warning", "success"
    type: BlockType = field(default=BlockType.CALLOUT, init=False)


@dataclass
class CommentaryBlock(Block):
    """Manually curated commentary with mandatory source provenance.

    Unlike ``TextBlock``, a ``CommentaryBlock`` *requires* a ``SourceRef``
    so that report-lint can verify that any numeric claims in the prose
    are backed by data. Use this for narrative summaries that contain
    concrete figures (speedups, sizes, counts).

    When ``--strict-lint`` is enabled, the linter will warn on any
    ``CommentaryBlock`` whose ``source`` is ``SourceKind.manual``.
    """
    content: str
    source: SourceRef | None = None
    type: BlockType = field(default=BlockType.COMMENTARY, init=False)


@dataclass
class TableBlock(Block):
    """A data table with optional caption and source provenance."""
    headers: list[str]
    rows: list[list[str]]
    caption: str | None = None
    notes: list[str] | None = None
    wide: bool = False
    source: SourceRef | None = None
    type: BlockType = field(default=BlockType.TABLE, init=False)


@dataclass
class FigureBlock(Block):
    """A chart or figure."""
    path: str
    """Relative path to the image file."""
    caption: str | None = None
    source: SourceRef | None = None
    type: BlockType = field(default=BlockType.FIGURE, init=False)


@dataclass
class Section:
    """A report section (## or ### equivalent)."""
    title: str
    blocks: list[Block] = field(default_factory=list)
    subsections: list[Section] = field(default_factory=list)
    id: str | None = None
    """Stable anchor ID. Auto-generated if None."""


@dataclass
class Chapter:
    """A top-level chapter (# or ## equivalent) with generated numbering."""
    title: str
    sections: list[Section] = field(default_factory=list)
    number: int | None = None
    id: str | None = None


@dataclass
class Report:
    """The root report document."""
    title: str
    chapters: list[Chapter] = field(default_factory=list)

    def generate_ids_and_numbers(self) -> None:
        """Assign sequential chapter numbers and stable HTML/Markdown IDs."""
        for i, chapter in enumerate(self.chapters, 1):
            chapter.number = i
            if not chapter.id:
                chapter.id = _slugify(chapter.title)
            for j, sec in enumerate(chapter.sections, 1):
                if not sec.id:
                    sec.id = f"{chapter.id}-{_slugify(sec.title)}"
                for k, subsec in enumerate(sec.subsections, 1):
                    if not subsec.id:
                        subsec.id = f"{sec.id}-{_slugify(subsec.title)}"


def _slugify(text: str) -> str:
    import re
    slug = re.sub(r"[^\w\s-]", "", text.strip().lower())
    return re.sub(r"[\s-]+", "-", slug).strip("-") or "section"


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

class MarkdownRenderer:
    """Renders a Report model to GitHub Flavored Markdown."""

    def render(self, report: Report) -> str:
        report.generate_ids_and_numbers()
        lines = [f"# {report.title}\n"]

        for chapter in report.chapters:
            title = f"{chapter.number}. {chapter.title}" if chapter.number else chapter.title
            lines.append(f"## {title} <a id=\"{chapter.id}\"></a>\n")
            for section in chapter.sections:
                lines.append(self._render_section(section, level=3))
        
        return "\n".join(lines).strip() + "\n"

    def _render_section(self, section: Section, level: int) -> str:
        lines = [f"{'#' * level} {section.title} <a id=\"{section.id}\"></a>\n"]
        for block in section.blocks:
            lines.append(self._render_block(block))
            lines.append("")  # Empty line between blocks
        
        for subsec in section.subsections:
            lines.append(self._render_section(subsec, level + 1))
        return "\n".join(lines)

    def _render_block(self, block: Block) -> str:
        if isinstance(block, TextBlock):
            return block.content
        elif isinstance(block, CalloutBlock):
            return f"> **{block.level.capitalize()}**\n> {block.content.replace(chr(10), chr(10) + '> ')}"
        elif isinstance(block, CommentaryBlock):
            return self._render_commentary(block)
        elif isinstance(block, TableBlock):
            return self._render_table(block)
        elif isinstance(block, FigureBlock):
            cap = block.caption or "Figure"
            res = f"![{cap}]({block.path})\n"
            if block.caption:
                res += f"*Figure: {block.caption}*\n"
            return res
        return ""

    def _render_table(self, table: TableBlock) -> str:
        lines = []
        if table.caption:
            lines.append(f"*{table.caption}*\n")
        
        # Wide table support: In plain markdown, we just render as normal,
        # but we could wrap in a div if GitHub supported it. For now, it's just a table.
        # But we do handle source provenance.
        
        if table.headers:
            header_line = "| " + " | ".join(table.headers) + " |"
            sep_line = "|" + "|".join(["---"] * len(table.headers)) + "|"
            lines.append(header_line)
            lines.append(sep_line)
        
        for row in table.rows:
            # Handle missing values via empty strings or "—" mapping in the model upstream.
            # Here we just render what we're given.
            safe_row = [str(x).replace("|", "\\|") for x in row]
            lines.append("| " + " | ".join(safe_row) + " |")
        
        if table.notes:
            lines.append("")
            for note in table.notes:
                lines.append(f"_{note}_")
                
        if table.source:
            lines.append("")
            lines.append(f"_Source: {table.source.kind.value}" + (f" ({table.source.path})" if table.source.path else "") + "_")
            
        return "\n".join(lines)

    def _render_commentary(self, block: CommentaryBlock) -> str:
        """Render a commentary block with source attribution."""
        lines = [block.content]
        if block.source:
            kind = block.source.kind.value
            detail = ""
            if block.source.path:
                detail += f" ({block.source.path})"
            if block.source.reason:
                detail += f" — {block.source.reason}"
            lines.append(f"\n_Source: {kind}{detail}_")
        return "\n".join(lines)


class HtmlRenderer:
    """Renders a Report model to semantic HTML."""

    def render(self, report: Report, prev_url: str | None = None) -> str:
        report.generate_ids_and_numbers()
        
        toc_items = []
        for chapter in report.chapters:
            title = f"{chapter.number}. {chapter.title}" if chapter.number else chapter.title
            toc_items.append(f'<li><a href="#{chapter.id}">{html.escape(title)}</a></li>')
            if chapter.sections:
                toc_items.append('<ul style="margin-top: 0.25rem;">')
                for sec in chapter.sections:
                    toc_items.append(f'<li><a href="#{sec.id}">{html.escape(sec.title)}</a></li>')
                toc_items.append('</ul>')

        toc_html = f'<nav class="toc"><strong>Contents</strong><ul>{"".join(toc_items)}</ul></nav>'
        
        body_lines = []
        for chapter in report.chapters:
            title = f"{chapter.number}. {chapter.title}" if chapter.number else chapter.title
            body_lines.append(f'<h2 id="{chapter.id}">{html.escape(title)}</h2>')
            for section in chapter.sections:
                body_lines.append(self._render_section(section, level=3))

        body_html = "\n".join(body_lines)
        
        # Reuse existing styling, modified for native elements instead of <pre>
        css = """
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 1.5rem 2rem 4rem; color: #1f2328; line-height: 1.55; }
        header { border-bottom: 1px solid #d0d7de; padding-bottom: 0.75rem; margin-bottom: 1.25rem; display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
        header h1 { font-size: 1.5rem; margin: 0; }
        nav.toc { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 1.5rem; font-size: 0.9rem; }
        nav.toc ul { margin: 0.25rem 0 0 1rem; padding: 0; }
        nav.toc a { color: #0969da; text-decoration: none; }
        nav.toc a:hover { text-decoration: underline; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 1rem; font-size: 0.9rem; }
        th, td { border: 1px solid #d0d7de; padding: 6px 13px; text-align: left; }
        th { background-color: #f6f8fa; font-weight: 600; }
        tr:nth-child(even) { background-color: #f6f8fa; }
        .table-wide { overflow-x: auto; display: block; width: 100%; }
        figure { margin: 1rem 0; padding: 1rem; background: #f6f8fa; border-radius: 6px; border: 1px solid #d0d7de; }
        figure img { max-width: 100%; height: auto; }
        figcaption { margin-top: 0.5rem; font-size: 0.85rem; color: #656d76; }
        .callout { padding: 1rem; border-left: 4px solid #0969da; background: #f6f8fa; margin-bottom: 1rem; }
        .callout-warning { border-left-color: #d73a49; }
        .source-ref { font-size: 0.8rem; color: #656d76; font-style: italic; margin-top: 0.25rem; }
        .notes { font-size: 0.85rem; color: #656d76; margin-top: 0.5rem; }
        """
        
        prev_link_html = f'<div class="prev-link"><a href="{html.escape(prev_url)}">← previous snapshot</a></div>' if prev_url else ""
        
        import datetime
        generated_at = datetime.datetime.now().isoformat(timespec="seconds")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{html.escape(report.title)}</title>
  <style>{css}</style>
</head>
<body>
<header>
  <div>
    <h1>{html.escape(report.title)}</h1>
    <div class="meta">Generated {generated_at}</div>
  </div>
  {prev_link_html}
</header>
{toc_html}
<div class="report-content">
{body_html}
</div>
</body>
</html>"""

    def _render_section(self, section: Section, level: int) -> str:
        lines = [f'<h{level} id="{section.id}">{html.escape(section.title)}</h{level}>']
        for block in section.blocks:
            lines.append(self._render_block(block))
        for subsec in section.subsections:
            lines.append(self._render_section(subsec, level + 1))
        return "\n".join(lines)

    @staticmethod
    def _md_to_html(text: str) -> str:
        """Convert markdown text to HTML.

        Handles the subset of markdown used by the report's TextBlock,
        CalloutBlock, and CommentaryBlock content:

        - ``**bold**`` → ``<strong>``
        - ``*italic*`` → ``<em>`` (single star, not inside **)
        - ``_italic_`` → ``<em>`` (underscore form)
        - `` `code` `` → ``<code>``
        - Unordered lists (``- item``)
        - Paragraphs (double-newline separated)
        - Single newlines inside a paragraph → ``<br>``

        Input is assumed to be *already escaped* via ``html.escape()``.
        """
        import re

        # ── Inline formatting ─────────────────────────────────────────
        def _inline(t: str) -> str:
            # Bold **...**
            t = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', t)
            # Italic *...* (but not **)
            t = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', t)
            # Italic _..._
            t = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'<em>\1</em>', t)
            # Inline code `...`
            t = re.sub(r'`(.*?)`', r'<code>\1</code>', t)
            return t

        # ── Block-level processing ────────────────────────────────────
        # Split into paragraphs on blank lines.
        paragraphs = re.split(r'\n{2,}', text)
        out_parts: list[str] = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            lines = para.split('\n')

            # Check if this paragraph is a bullet list (all lines start
            # with ``- `` or are continuation indents).
            if all(
                ln.lstrip().startswith('- ') or ln.startswith('  ')
                for ln in lines if ln.strip()
            ) and any(ln.lstrip().startswith('- ') for ln in lines):
                # Merge continuation lines into their parent bullet.
                items: list[str] = []
                for ln in lines:
                    stripped = ln.lstrip()
                    if stripped.startswith('- '):
                        items.append(stripped[2:])
                    elif items:
                        items[-1] += ' ' + stripped
                out_parts.append(
                    '<ul>'
                    + ''.join(f'<li>{_inline(it)}</li>' for it in items)
                    + '</ul>'
                )
            else:
                # Regular paragraph.  Convert single newlines to <br>.
                body = '<br>\n'.join(_inline(ln) for ln in lines)
                out_parts.append(f'<p>{body}</p>')

        return '\n'.join(out_parts)

    def _render_block(self, block: Block) -> str:
        if isinstance(block, TextBlock):
            text = html.escape(block.content)
            return self._md_to_html(text)

        elif isinstance(block, CalloutBlock):
            cls = "callout-warning" if block.level == "warning" else "callout"
            text = html.escape(block.content)
            body = self._md_to_html(text)
            return f'<div class="{cls}">{body}</div>'

        elif isinstance(block, CommentaryBlock):
            text = html.escape(block.content)
            body = self._md_to_html(text)
            src_html = ""
            if block.source:
                src_text = block.source.kind.value
                if block.source.path:
                    src_text += f" ({block.source.path})"
                if block.source.reason:
                    src_text += f" — {block.source.reason}"
                src_html = f'<div class="source-ref">Source: {html.escape(src_text)}</div>'
            return f'<div class="commentary">{body}{src_html}</div>'

        elif isinstance(block, TableBlock):
            lines = []
            if block.wide:
                lines.append('<div class="table-wide">')
            lines.append('<table>')

            if block.caption:
                lines.append(f'<caption>{html.escape(block.caption)}</caption>')

            if block.headers:
                lines.append('<thead><tr>')
                for h in block.headers:
                    lines.append(f'<th>{html.escape(h)}</th>')
                lines.append('</tr></thead>')

            lines.append('<tbody>')
            for row in block.rows:
                lines.append('<tr>')
                for cell in row:
                    cell_html = html.escape(str(cell))
                    # Preserve inline bold / italic / code in table cells
                    import re
                    cell_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', cell_html)
                    cell_html = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', cell_html)
                    cell_html = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'<em>\1</em>', cell_html)
                    cell_html = re.sub(r'`(.*?)`', r'<code>\1</code>', cell_html)
                    lines.append(f'<td>{cell_html}</td>')
                lines.append('</tr>')
            lines.append('</tbody></table>')
            if block.wide:
                lines.append('</div>')

            if block.notes:
                lines.append('<div class="notes">')
                for note in block.notes:
                    lines.append(f'<div>{html.escape(note)}</div>')
                lines.append('</div>')

            if block.source:
                src_text = block.source.kind.value
                if block.source.path:
                    src_text += f" ({block.source.path})"
                lines.append(f'<div class="source-ref">Source: {html.escape(src_text)}</div>')

            return "\n".join(lines)

        elif isinstance(block, FigureBlock):
            cap = f'<figcaption>{html.escape(block.caption)}</figcaption>' if block.caption else ""
            src = ""
            if block.source:
                src_text = block.source.kind.value
                if block.source.path:
                    src_text += f" ({block.source.path})"
                src = f'<div class="source-ref">Source: {html.escape(src_text)}</div>'

            return f'<figure><img src="{html.escape(block.path)}" alt="{html.escape(block.caption or "")}">{cap}{src}</figure>'

        return ""


class JsonManifestRenderer:
    """Renders a Report model to a structured JSON manifest."""

    def render(self, report: Report) -> str:
        report.generate_ids_and_numbers()
        
        def _dump_source(src: SourceRef | None) -> dict | None:
            if not src:
                return None
            return {
                "kind": src.kind.value,
                "path": src.path,
                "reason": src.reason,
            }
            
        def _dump_block(block: Block) -> dict:
            if isinstance(block, TextBlock):
                return {"type": "text", "content": block.content}
            elif isinstance(block, CalloutBlock):
                return {"type": "callout", "level": block.level, "content": block.content}
            elif isinstance(block, CommentaryBlock):
                return {
                    "type": "commentary",
                    "content": block.content,
                    "source": _dump_source(block.source),
                }
            elif isinstance(block, TableBlock):
                return {
                    "type": "table",
                    "caption": block.caption,
                    "headers": block.headers,
                    "rows": block.rows,
                    "notes": block.notes,
                    "wide": block.wide,
                    "source": _dump_source(block.source)
                }
            elif isinstance(block, FigureBlock):
                return {
                    "type": "figure",
                    "path": block.path,
                    "caption": block.caption,
                    "source": _dump_source(block.source)
                }
            return {"type": "unknown"}

        def _dump_section(section: Section) -> dict:
            return {
                "id": section.id,
                "title": section.title,
                "blocks": [_dump_block(b) for b in section.blocks],
                "subsections": [_dump_section(s) for s in section.subsections]
            }

        def _dump_chapter(chapter: Chapter) -> dict:
            return {
                "id": chapter.id,
                "number": chapter.number,
                "title": chapter.title,
                "sections": [_dump_section(s) for s in chapter.sections]
            }

        manifest = {
            "title": report.title,
            "chapters": [_dump_chapter(c) for c in report.chapters]
        }
        return json.dumps(manifest, indent=2)
