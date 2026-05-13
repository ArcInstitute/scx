"""Report-lint checks (Phase 7 — manual-source warnings).

Phase 7 adds:
- ``check_manual_numeric_claims``: warn on any ``CommentaryBlock`` whose
  ``source`` is ``SourceKind.manual`` (these have unverifiable numbers).
- ``check_tables_have_sources``: warn on ``TableBlock`` without a ``SourceRef``.
- ``collect_warnings``: run all checks and return structured warnings.

Future phases (8+) will add:
- Duplicate heading/anchor checks.
- Missing figure checks.
- Executive summary consistency checks.
- Memory-mode consistency checks.
- HTML semantic-table checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from benchmarks.comprehensive.reporting.report_model import (
    Block, CommentaryBlock, TableBlock, TextBlock, Report, Chapter, Section,
)
from benchmarks.comprehensive.reporting.result_store import SourceKind


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


# Regex matching inline numeric claims like "4.2x", "1,405 batches/sec",
# "2.35 GB", "82x faster" — numbers that should be derived from data.
_NUMERIC_CLAIM_RE = re.compile(
    r"\b\d[\d,.]*\s*(?:x|GB|MB|KB|TB|batches/sec|batches/s|cells/s|rows/s)",
    re.IGNORECASE,
)


def check_manual_numeric_claims(report: Report) -> list[LintWarning]:
    """Warn on CommentaryBlocks with ``SourceKind.manual`` that contain numbers.

    These blocks have numeric claims that cannot be automatically verified
    against raw benchmark data.
    """
    warnings: list[LintWarning] = []
    for chapter in report.chapters:
        for section in chapter.sections:
            _check_section_manual_claims(
                chapter.title, section, warnings,
            )
    return warnings


def _check_section_manual_claims(
    chapter_title: str, section: Section, warnings: list[LintWarning],
) -> None:
    for block in section.blocks:
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
                        chapter=chapter_title,
                        section=section.title,
                    ))
    for subsec in section.subsections:
        _check_section_manual_claims(chapter_title, subsec, warnings)


def check_tables_have_sources(report: Report) -> list[LintWarning]:
    """Warn on TableBlocks that lack a SourceRef.

    Every table should have provenance tracking so the report is fully
    traceable. Tables without source annotations get an info-level
    warning (not blocking).
    """
    warnings: list[LintWarning] = []
    for chapter in report.chapters:
        for section in chapter.sections:
            _check_section_table_sources(
                chapter.title, section, warnings,
            )
    return warnings


def _check_section_table_sources(
    chapter_title: str, section: Section, warnings: list[LintWarning],
) -> None:
    for block in section.blocks:
        if isinstance(block, TableBlock) and block.source is None:
            cap = block.caption or "(no caption)"
            warnings.append(LintWarning(
                level=LintLevel.info,
                check="table_source_ref",
                message=f"TableBlock '{cap}' has no SourceRef.",
                chapter=chapter_title,
                section=section.title,
            ))
    for subsec in section.subsections:
        _check_section_table_sources(chapter_title, subsec, warnings)


def collect_warnings(
    report: Report,
    *,
    strict: bool = False,
) -> list[LintWarning]:
    """Run all lint checks and return warnings.

    Parameters
    ----------
    report : Report
        The fully built report model.
    strict : bool
        If ``True``, promote info-level manual-source warnings to
        warning-level.  Used by ``--strict-lint``.

    Returns
    -------
    list[LintWarning]
        All findings, sorted by severity.
    """
    warnings: list[LintWarning] = []
    warnings.extend(check_manual_numeric_claims(report))
    warnings.extend(check_tables_have_sources(report))

    if strict:
        for w in warnings:
            if w.check == "manual_numeric_claims":
                w.level = LintLevel.error

    # Sort: errors first, then warnings, then info.
    severity_order = {LintLevel.error: 0, LintLevel.warning: 1, LintLevel.info: 2}
    warnings.sort(key=lambda w: severity_order.get(w.level, 3))
    return warnings
