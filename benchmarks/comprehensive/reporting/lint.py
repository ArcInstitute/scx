"""Report-lint checks (Phase 8 — full implementation).

Phases 7+8 lint checks:

1. ``check_manual_numeric_claims``: warn on ``CommentaryBlock`` with
   ``SourceKind.manual`` that contain numeric claims.
2. ``check_tables_have_sources``: warn on ``TableBlock`` without a
   ``SourceRef``.
3. ``check_duplicate_headings``: error on duplicate chapter/section titles
   and duplicate anchor IDs.
4. ``check_missing_figures``: warn on ``FigureBlock`` paths that don't
   exist on disk.
5. ``check_executive_summary_consistency``: verify that the executive
   summary's correctness counts match the detail tables in Chapter 3.
6. ``check_skipped_vs_failed``: warn if a table mixes "Skipped" and
   "Fail" status without a clear reason column.
7. ``check_empty_cells``: warn on table rows with empty/dash cells that
   lack a notes or reason explanation.
8. ``check_approximate_number_sources``: warn on ``TextBlock`` / ``CalloutBlock``
   containing approximate numbers (``~``, ``≈``, ``roughly``) without
   adjacent source attribution.
9. ``check_memory_mode_consistency``: verify that memory takeaway text
   references the same measurement mode as the adjacent table caption.
10. ``check_public_profile``: block internal phase labels (``Phase \\d``,
    ``P\\d``) in chapter/section headings when public profile is active.
11. ``check_html_semantic_tables``: verify that the HtmlRenderer's table
    output uses ``<thead>``/``<tbody>``/``<caption>`` elements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from benchmarks.comprehensive.reporting.report_model import (
    Block, CommentaryBlock, TableBlock, TextBlock, CalloutBlock,
    FigureBlock, Report, Chapter, Section,
    HtmlRenderer,
)
from benchmarks.comprehensive.reporting.result_store import SourceKind


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class LintLevel(str, Enum):
    """Severity of a lint warning."""
    info = "info"
    warning = "warning"
    error = "error"


@dataclass
class LintWarning:
    """A single lint finding."""
    level: LintLevel
    check: str
    """Name of the check that produced this warning."""
    message: str
    chapter: str | None = None
    section: str | None = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


# Regex matching inline numeric claims like "4.2x", "1,405 batches/sec",
# "2.35 GB", "82x faster" — numbers that should be derived from data.
_NUMERIC_CLAIM_RE = re.compile(
    r"\b\d[\d,.]*\s*(?:x|GB|MB|KB|TB|batches/sec|batches/s|cells/s|rows/s)",
    re.IGNORECASE,
)

# Regex matching approximate qualifiers — "~", "≈", "roughly", "approximately"
_APPROX_RE = re.compile(
    r"~\s*\d|≈\s*\d|\broughly\s+\d|\bapproximately\s+\d|\b~\d",
    re.IGNORECASE,
)

# Regex matching internal phase labels in headings
_PHASE_LABEL_RE = re.compile(
    r"\bPhase\s+\d+\b|\bP\d+\b",
    re.IGNORECASE,
)

# Empty cell sentinels
_EMPTY_CELL_VALUES = {"", "—", "-", "–", "n/a", "N/A", "—"}


def _iter_sections(report: Report):
    """Yield (chapter, section, depth) triples for all sections."""
    for chapter in report.chapters:
        for section in chapter.sections:
            yield chapter, section, 2
            for subsec in section.subsections:
                yield chapter, subsec, 3


def _iter_blocks(report: Report):
    """Yield (chapter_title, section_title, block) triples for all blocks."""
    for chapter in report.chapters:
        for section in chapter.sections:
            yield from _section_blocks(chapter.title, section)


def _section_blocks(chapter_title: str, section: Section):
    """Yield (chapter_title, section_title, block) triples."""
    for block in section.blocks:
        yield chapter_title, section.title, block
    for subsec in section.subsections:
        yield from _section_blocks(chapter_title, subsec)


# ---------------------------------------------------------------------------
# 1. Manual numeric claims (Phase 7 carry-forward)
# ---------------------------------------------------------------------------


def check_manual_numeric_claims(report: Report) -> list[LintWarning]:
    """Warn on CommentaryBlocks with ``SourceKind.manual`` that contain numbers."""
    warnings: list[LintWarning] = []
    for ch_title, sec_title, block in _iter_blocks(report):
        if isinstance(block, CommentaryBlock):
            if block.source and block.source.kind == SourceKind.manual:
                if _NUMERIC_CLAIM_RE.search(block.content):
                    warnings.append(LintWarning(
                        level=LintLevel.warning,
                        check="manual_numeric_claims",
                        message=(
                            f"CommentaryBlock with manual source contains "
                            f"numeric claims: {block.content[:80]}..."
                        ),
                        chapter=ch_title,
                        section=sec_title,
                    ))
    return warnings


# ---------------------------------------------------------------------------
# 2. Table source references (Phase 7 carry-forward)
# ---------------------------------------------------------------------------


def check_tables_have_sources(report: Report) -> list[LintWarning]:
    """Warn on TableBlocks that lack a SourceRef."""
    warnings: list[LintWarning] = []
    for ch_title, sec_title, block in _iter_blocks(report):
        if isinstance(block, TableBlock) and block.source is None:
            cap = block.caption or "(no caption)"
            warnings.append(LintWarning(
                level=LintLevel.info,
                check="table_source_ref",
                message=f"TableBlock '{cap}' has no SourceRef.",
                chapter=ch_title,
                section=sec_title,
            ))
    return warnings


# ---------------------------------------------------------------------------
# 3. Duplicate headings and anchors
# ---------------------------------------------------------------------------


def check_duplicate_headings(report: Report) -> list[LintWarning]:
    """Error on duplicate chapter/section titles and duplicate anchor IDs.

    Duplicate anchors break internal links and ToC navigation. Duplicate
    titles confuse readers and search.
    """
    warnings: list[LintWarning] = []
    # Ensure IDs are generated
    report.generate_ids_and_numbers()

    seen_titles: dict[str, str] = {}  # title -> first location
    seen_ids: dict[str, str] = {}     # id -> first location

    for chapter in report.chapters:
        loc = f"Ch {chapter.number}: {chapter.title}"
        _check_dup(chapter.title, loc, seen_titles, "heading", warnings)
        if chapter.id:
            _check_dup(chapter.id, loc, seen_ids, "anchor", warnings)

        for section in chapter.sections:
            s_loc = f"{loc} > {section.title}"
            # Section titles can repeat across chapters — only flag if
            # the *anchor ID* collides, which is the real problem.
            if section.id:
                _check_dup(section.id, s_loc, seen_ids, "anchor", warnings)
            for subsec in section.subsections:
                ss_loc = f"{s_loc} > {subsec.title}"
                if subsec.id:
                    _check_dup(subsec.id, ss_loc, seen_ids, "anchor", warnings)

    return warnings


def _check_dup(
    value: str,
    location: str,
    seen: dict[str, str],
    kind: str,  # "heading" or "anchor"
    warnings: list[LintWarning],
) -> None:
    if value in seen:
        warnings.append(LintWarning(
            level=LintLevel.error,
            check=f"duplicate_{kind}",
            message=(
                f"Duplicate {kind} '{value}' at [{location}], "
                f"first seen at [{seen[value]}]."
            ),
        ))
    else:
        seen[value] = location


# ---------------------------------------------------------------------------
# 4. Missing figures
# ---------------------------------------------------------------------------


def check_missing_figures(
    report: Report,
    *,
    figures_root: Path | None = None,
) -> list[LintWarning]:
    """Warn on FigureBlock paths that don't exist on disk.

    Parameters
    ----------
    figures_root : Path, optional
        Root directory where figure paths are relative to. If ``None``,
        uses the comprehensive reports directory.
    """
    if figures_root is None:
        try:
            from benchmarks.comprehensive.config import REPORTS_DIR
            figures_root = REPORTS_DIR
        except ImportError:
            figures_root = Path(".")

    warnings: list[LintWarning] = []
    for ch_title, sec_title, block in _iter_blocks(report):
        if isinstance(block, FigureBlock):
            full_path = figures_root / block.path
            if not full_path.exists():
                warnings.append(LintWarning(
                    level=LintLevel.warning,
                    check="missing_figure",
                    message=f"Figure '{block.path}' not found at {full_path}.",
                    chapter=ch_title,
                    section=sec_title,
                ))
    return warnings


# ---------------------------------------------------------------------------
# 5. Executive-summary consistency
# ---------------------------------------------------------------------------


def check_executive_summary_consistency(report: Report) -> list[LintWarning]:
    """Verify executive summary correctness counts match Chapter 3 detail.

    The executive summary derives correctness counts from the result store.
    If the counts in the CalloutBlock text don't match what Chapter 3's
    dataset-summary table shows, flag the discrepancy.
    """
    warnings: list[LintWarning] = []

    # Extract correctness counts from executive summary
    exec_chapter = None
    correctness_chapter = None
    for ch in report.chapters:
        if ch.title.lower().startswith("executive"):
            exec_chapter = ch
        if "correctness" in ch.title.lower():
            correctness_chapter = ch

    if not exec_chapter or not correctness_chapter:
        return warnings

    # Extract pass/total from executive summary callout
    exec_pass = exec_total = None
    for sec in exec_chapter.sections:
        for block in sec.blocks:
            if isinstance(block, CalloutBlock):
                m = re.search(r"(\d+)/(\d+)\s+scanpy\s+equivalence", block.content)
                if m:
                    exec_pass = int(m.group(1))
                    exec_total = int(m.group(2))

    if exec_pass is None:
        return warnings

    # Extract pass/total from Chapter 3 dataset-level summary table
    ch3_pass = ch3_total = 0
    for sec in correctness_chapter.sections:
        for block in sec.blocks:
            if isinstance(block, TableBlock) and block.caption:
                if "dataset-level" in block.caption.lower() or \
                   "correctness" in block.caption.lower():
                    for row in block.rows:
                        # Look for pass/fail/skip columns
                        for cell in row:
                            m = re.match(r"^(\d+)$", cell.strip())
                            if m:
                                pass  # Can't reliably parse without header context
                    # Use a simpler approach: count pass/fail in status columns
                    if block.headers:
                        pass_idx = None
                        fail_idx = None
                        skip_idx = None
                        for i, h in enumerate(block.headers):
                            hl = h.lower()
                            if "pass" in hl:
                                pass_idx = i
                            elif "fail" in hl:
                                fail_idx = i
                            elif "skip" in hl:
                                skip_idx = i

                        if pass_idx is not None:
                            for row in block.rows:
                                if pass_idx < len(row):
                                    try:
                                        ch3_pass += int(row[pass_idx])
                                    except ValueError:
                                        pass
                            for row in block.rows:
                                for idx in [pass_idx, fail_idx, skip_idx]:
                                    if idx is not None and idx < len(row):
                                        try:
                                            ch3_total += int(row[idx])
                                        except ValueError:
                                            pass

    if ch3_total > 0 and (exec_pass != ch3_pass or exec_total != ch3_total):
        warnings.append(LintWarning(
            level=LintLevel.warning,
            check="executive_consistency",
            message=(
                f"Executive summary claims {exec_pass}/{exec_total} scanpy "
                f"equivalence tests pass, but Chapter 3 dataset-level table "
                f"totals {ch3_pass}/{ch3_total}."
            ),
            chapter=exec_chapter.title,
        ))

    return warnings


# ---------------------------------------------------------------------------
# 6. Skipped-vs-failed status checks
# ---------------------------------------------------------------------------


def check_skipped_vs_failed(report: Report) -> list[LintWarning]:
    """Warn if a correctness table has 'Fail' status without a reason column.

    Tables with failed tests should include a notes/reason column so
    readers can understand the failure context.
    """
    warnings: list[LintWarning] = []
    for ch_title, sec_title, block in _iter_blocks(report):
        if not isinstance(block, TableBlock):
            continue
        if not block.headers:
            continue

        # Find status and notes columns
        status_idx = None
        has_notes_col = False
        for i, h in enumerate(block.headers):
            hl = h.lower()
            if hl in ("status", "result", "pass/fail"):
                status_idx = i
            if hl in ("notes", "reason", "detail", "details", "comments"):
                has_notes_col = True

        if status_idx is None:
            continue

        has_fail = False
        fail_without_reason = False
        for row in block.rows:
            if status_idx < len(row):
                cell = row[status_idx].lower()
                if "fail" in cell or "error" in cell:
                    has_fail = True
                    if not has_notes_col:
                        fail_without_reason = True

        if fail_without_reason:
            cap = block.caption or "(no caption)"
            warnings.append(LintWarning(
                level=LintLevel.warning,
                check="skipped_vs_failed",
                message=(
                    f"Table '{cap}' has failed tests but no notes/reason "
                    f"column to explain the failures."
                ),
                chapter=ch_title,
                section=sec_title,
            ))

    return warnings


# ---------------------------------------------------------------------------
# 7. Empty cells / missing reasons
# ---------------------------------------------------------------------------


def check_empty_cells(report: Report) -> list[LintWarning]:
    """Warn on table rows with many empty/dash cells that lack notes.

    A row where more than half the data cells are empty/dash suggests
    missing data — readers should see an explanation.
    """
    warnings: list[LintWarning] = []
    for ch_title, sec_title, block in _iter_blocks(report):
        if not isinstance(block, TableBlock):
            continue
        if not block.rows or not block.headers:
            continue

        # Find notes column index if present
        notes_idx = None
        for i, h in enumerate(block.headers):
            if h.lower() in ("notes", "reason", "detail", "details"):
                notes_idx = i

        n_data_cols = len(block.headers)
        if notes_idx is not None:
            n_data_cols -= 1  # Don't count notes column itself

        if n_data_cols < 2:
            continue  # Too few columns to check

        for row_idx, row in enumerate(block.rows):
            empty_count = 0
            for i, cell in enumerate(row):
                if i == notes_idx:
                    continue
                if cell.strip() in _EMPTY_CELL_VALUES:
                    empty_count += 1

            # Flag if > half the data cells are empty and no note
            if empty_count > n_data_cols // 2:
                has_note = (
                    notes_idx is not None
                    and notes_idx < len(row)
                    and row[notes_idx].strip() not in _EMPTY_CELL_VALUES
                )
                if not has_note:
                    cap = block.caption or "(no caption)"
                    label_cell = row[0] if row else f"row {row_idx}"
                    warnings.append(LintWarning(
                        level=LintLevel.info,
                        check="empty_cells",
                        message=(
                            f"Table '{cap}', row '{label_cell}': "
                            f"{empty_count}/{n_data_cols} data cells empty "
                            f"with no explanation."
                        ),
                        chapter=ch_title,
                        section=sec_title,
                    ))

    return warnings


# ---------------------------------------------------------------------------
# 8. Approximate-number source checks
# ---------------------------------------------------------------------------


def check_approximate_number_sources(report: Report) -> list[LintWarning]:
    """Warn on TextBlock/CalloutBlock with approximate numbers lacking sources.

    Approximate numbers (``~4.2x``, ``≈1,400``, ``roughly 80%``) in
    narrative text should have adjacent source attribution. This check
    flags blocks that use approximation qualifiers without being
    ``CommentaryBlock`` (which carries a ``SourceRef``).
    """
    warnings: list[LintWarning] = []
    for ch_title, sec_title, block in _iter_blocks(report):
        if isinstance(block, (TextBlock, CalloutBlock)):
            content = block.content if hasattr(block, "content") else ""
            if _APPROX_RE.search(content):
                warnings.append(LintWarning(
                    level=LintLevel.info,
                    check="approximate_number_source",
                    message=(
                        f"Block contains approximate numbers without "
                        f"source attribution: {content[:80]}..."
                    ),
                    chapter=ch_title,
                    section=sec_title,
                ))
    return warnings


# ---------------------------------------------------------------------------
# 9. Memory-mode consistency
# ---------------------------------------------------------------------------


def check_memory_mode_consistency(report: Report) -> list[LintWarning]:
    """Verify that memory takeaway text references the correct mode.

    Within a single section, if a TableBlock caption mentions a specific
    memory-measurement mode (e.g., "Full read", "Subset read") and a
    sibling CommentaryBlock/TextBlock references a *different* mode,
    flag the mismatch.
    """
    _MODES = ["full read", "subset read", "backed", "query", "streaming"]
    warnings: list[LintWarning] = []

    for chapter in report.chapters:
        for section in chapter.sections:
            _check_memory_section(chapter.title, section, _MODES, warnings)
    return warnings


def _check_memory_section(
    ch_title: str,
    section: Section,
    modes: list[str],
    warnings: list[LintWarning],
) -> None:
    # Collect modes mentioned in table captions
    table_modes: set[str] = set()
    for block in section.blocks:
        if isinstance(block, TableBlock) and block.caption:
            cap_lower = block.caption.lower()
            for mode in modes:
                if mode in cap_lower:
                    table_modes.add(mode)

    if not table_modes:
        # No memory-mode tables in this section
        for subsec in section.subsections:
            _check_memory_section(ch_title, subsec, modes, warnings)
        return

    # Check narrative blocks for mode mentions
    for block in section.blocks:
        if isinstance(block, (TextBlock, CommentaryBlock, CalloutBlock)):
            content = block.content if hasattr(block, "content") else ""
            content_lower = content.lower()
            for mode in modes:
                if mode in content_lower and mode not in table_modes:
                    warnings.append(LintWarning(
                        level=LintLevel.warning,
                        check="memory_mode_consistency",
                        message=(
                            f"Narrative mentions '{mode}' mode but adjacent "
                            f"table caption(s) only cover: "
                            f"{', '.join(sorted(table_modes))}."
                        ),
                        chapter=ch_title,
                        section=section.title,
                    ))
                    break  # One warning per section is enough

    for subsec in section.subsections:
        _check_memory_section(ch_title, subsec, modes, warnings)


# ---------------------------------------------------------------------------
# 10. Public-profile checks
# ---------------------------------------------------------------------------


def check_public_profile(
    report: Report,
    *,
    public: bool = False,
) -> list[LintWarning]:
    """Block internal phase labels in headings when public profile is active.

    Phase labels like "Phase 5", "P6", etc. are internal development
    milestones and should not appear in public-facing headings.
    """
    if not public:
        return []

    warnings: list[LintWarning] = []
    for chapter in report.chapters:
        if _PHASE_LABEL_RE.search(chapter.title):
            warnings.append(LintWarning(
                level=LintLevel.error,
                check="public_profile",
                message=(
                    f"Chapter heading '{chapter.title}' contains an "
                    f"internal phase label. Remove for public profile."
                ),
                chapter=chapter.title,
            ))
        for section in chapter.sections:
            _check_section_phase_labels(chapter.title, section, warnings)

    return warnings


def _check_section_phase_labels(
    ch_title: str,
    section: Section,
    warnings: list[LintWarning],
) -> None:
    if _PHASE_LABEL_RE.search(section.title):
        warnings.append(LintWarning(
            level=LintLevel.error,
            check="public_profile",
            message=(
                f"Section heading '{section.title}' contains an "
                f"internal phase label. Remove for public profile."
            ),
            chapter=ch_title,
            section=section.title,
        ))
    for subsec in section.subsections:
        _check_section_phase_labels(ch_title, subsec, warnings)


# ---------------------------------------------------------------------------
# 11. HTML semantic-table checks
# ---------------------------------------------------------------------------


def check_html_semantic_tables(report: Report) -> list[LintWarning]:
    """Verify that the HtmlRenderer's table output uses semantic HTML.

    Checks that every TableBlock in the report would render with proper
    ``<thead>``/``<tbody>`` structure and ``<caption>`` when a caption is
    set.  This validates the renderer contract rather than the data.
    """
    warnings: list[LintWarning] = []
    renderer = HtmlRenderer()

    for ch_title, sec_title, block in _iter_blocks(report):
        if not isinstance(block, TableBlock):
            continue

        html_out = renderer._render_block(block)

        if block.headers and "<thead>" not in html_out:
            warnings.append(LintWarning(
                level=LintLevel.error,
                check="html_semantic_table",
                message=(
                    f"Table '{block.caption or '(no caption)'}' renders "
                    f"without <thead> element."
                ),
                chapter=ch_title,
                section=sec_title,
            ))

        if block.rows and "<tbody>" not in html_out:
            warnings.append(LintWarning(
                level=LintLevel.error,
                check="html_semantic_table",
                message=(
                    f"Table '{block.caption or '(no caption)'}' renders "
                    f"without <tbody> element."
                ),
                chapter=ch_title,
                section=sec_title,
            ))

        if block.caption and "<caption>" not in html_out:
            warnings.append(LintWarning(
                level=LintLevel.warning,
                check="html_semantic_table",
                message=(
                    f"Table '{block.caption}' renders without <caption> "
                    f"element."
                ),
                chapter=ch_title,
                section=sec_title,
            ))

    return warnings


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def collect_warnings(
    report: Report,
    *,
    strict: bool = False,
    public: bool = False,
    figures_root: Path | None = None,
) -> list[LintWarning]:
    """Run all lint checks and return warnings.

    Parameters
    ----------
    report : Report
        The fully built report model.
    strict : bool
        If ``True``, promote manual-numeric-claims warnings to errors
        and exit nonzero on any error-level findings.
    public : bool
        If ``True``, enable public-profile checks that block internal
        phase labels in headings.
    figures_root : Path, optional
        Root directory for figure path resolution. Defaults to
        ``REPORTS_DIR``.

    Returns
    -------
    list[LintWarning]
        All findings, sorted by severity.
    """
    warnings: list[LintWarning] = []

    # Phase 7 checks
    warnings.extend(check_manual_numeric_claims(report))
    warnings.extend(check_tables_have_sources(report))

    # Phase 8 checks
    warnings.extend(check_duplicate_headings(report))
    warnings.extend(check_missing_figures(report, figures_root=figures_root))
    warnings.extend(check_executive_summary_consistency(report))
    warnings.extend(check_skipped_vs_failed(report))
    warnings.extend(check_empty_cells(report))
    warnings.extend(check_approximate_number_sources(report))
    warnings.extend(check_memory_mode_consistency(report))
    warnings.extend(check_public_profile(report, public=public))
    warnings.extend(check_html_semantic_tables(report))

    if strict:
        for w in warnings:
            if w.check == "manual_numeric_claims":
                w.level = LintLevel.error

    # Sort: errors first, then warnings, then info.
    severity_order = {LintLevel.error: 0, LintLevel.warning: 1, LintLevel.info: 2}
    warnings.sort(key=lambda w: severity_order.get(w.level, 3))
    return warnings
