"""
Assemble the final comprehensive benchmark report as markdown and PDF.

Pulls data from raw JSON results via the report model and chapter builders.

Lint integration:
- ``write_reports()`` runs ``collect_warnings()`` and
  logs them.  With ``strict_lint=True``, exits nonzero on lint failures.

CLI integration:
- ``write_reports()`` is now a compatibility wrapper.  For new
  integrations, use ``report_cli.py`` or its ``build_report()``
  function, which support ``--profile``, ``--results-dir``,
  ``--snapshot-dir``, ``--output-dir``, and format selection.
"""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
import subprocess

from benchmarks.comprehensive.config import REPORTS_DIR
from benchmarks.comprehensive.reporting.report_model import Report, MarkdownRenderer
from benchmarks.comprehensive.reporting.result_store import get_store
from benchmarks.comprehensive.reporting.sections import (
    executive, methodology, correctness, coverage, storage, io, cloud, ml_loader,
    accelerators, specialized, operations, discussion, appendix
)

logger = logging.getLogger(__name__)


def generate_report_model() -> Report:
    """Generate the full benchmark report AST."""
    store = get_store()
    report = Report(title="SCX Comprehensive Benchmark Report")
    
    report.chapters.append(executive.build(store))
    report.chapters.append(methodology.build(store))
    report.chapters.append(correctness.build(store))
    report.chapters.append(coverage.build(store))
    report.chapters.append(storage.build(store))
    report.chapters.append(io.build(store))
    report.chapters.append(cloud.build(store))
    report.chapters.append(ml_loader.build(store))
    report.chapters.append(accelerators.build(store))
    report.chapters.append(specialized.build(store))
    report.chapters.append(operations.build(store))
    report.chapters.append(discussion.build(store))
    report.chapters.append(appendix.build(store))
    
    return report


def generate_report() -> str:
    """Generate the full benchmark report as a markdown string."""
    report = generate_report_model()
    return MarkdownRenderer().render(report)


def write_html_snapshot(
    report: Report,
    output_dir: Path,
    *,
    title: str = "SCX Benchmark Report",
) -> Path:
    """Write a browsable HTML snapshot alongside the markdown report."""
    from benchmarks.comprehensive.reporting import dashboard

    output_dir.mkdir(parents=True, exist_ok=True)
    prev_url = dashboard.previous_url(output_dir)

    html_body = dashboard.render_html(
        report, title=title, prev_url=prev_url,
    )
    html_path = output_dir / "BENCHMARK_REPORT.html"
    html_path.write_text(html_body)
    
    dashboard.append_history(
        output_dir,
        url=html_path.name,
        title=title,
    )
    return html_path


def _generate_pdf(md_path: Path, pdf_path: Path) -> None:
    """Internal helper to render PDF (if installed)."""
    if not shutil.which("pandoc"):
        return
    try:
        subprocess.run(
            ["pandoc", str(md_path), "-o", str(pdf_path), "--pdf-engine=xelatex"],
            check=True,
            capture_output=True,
        )
        logger.info("Wrote PDF report to %s", pdf_path)
    except subprocess.CalledProcessError as exc:
        logger.warning("PDF generation failed: %s", exc.stderr.decode(errors="replace"))


def write_reports(
    output_dir: Path | None = None,
    *,
    strict_lint: bool = False,
    public_profile: bool = False,
) -> Path:
    """Generate and write the full benchmark report (markdown + PDF + HTML).

    Parameters
    ----------
    output_dir : Path, optional
        Output directory. Defaults to ``REPORTS_DIR``.
    strict_lint : bool
        If ``True``, run report-lint checks and exit nonzero (raise) on
        any error-level findings.  Manual numeric claims in
        ``CommentaryBlock`` objects are promoted to errors in strict mode.
    public_profile : bool
        If ``True``, enable public-profile lint checks that block internal
        phase labels in chapter/section headings.
    """
    if output_dir is None:
        output_dir = REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    report_model = generate_report_model()

    # ── Report lint ───────────────────────────────────────────────────
    from benchmarks.comprehensive.reporting.lint import collect_warnings, LintLevel
    warnings = collect_warnings(
        report_model,
        strict=strict_lint,
        public=public_profile,
        figures_root=output_dir,
    )
    for w in warnings:
        log_fn = logger.warning if w.level == LintLevel.warning else (
            logger.error if w.level == LintLevel.error else logger.info
        )
        loc = ""
        if w.chapter:
            loc += f"[{w.chapter}]"
        if w.section:
            loc += f"[{w.section}]"
        log_fn("lint %s %s: %s", w.check, loc, w.message)

    errors = [w for w in warnings if w.level == LintLevel.error]
    if strict_lint and errors:
        raise SystemExit(
            f"Report lint failed with {len(errors)} error(s). "
            "Fix the issues above or remove --strict-lint."
        )

    # ── Write markdown ────────────────────────────────────────────────
    md_body = MarkdownRenderer().render(report_model)
    md_path = output_dir / "BENCHMARK_REPORT.md"
    md_path.write_text(md_body)
    logger.info(f"Wrote markdown report to {md_path}")

    # Generate PDF with embedded figures
    pdf_path = output_dir / "BENCHMARK_REPORT.pdf"
    _generate_pdf(md_path, pdf_path)

    # Emit HTML snapshot
    try:
        write_html_snapshot(report_model, output_dir)
    except Exception as exc:
        logger.warning("HTML snapshot emission failed: %s", exc)

    # ── Write lint warnings to JSON manifest ──────────────────────────
    if warnings:
        import json
        lint_path = output_dir / "LINT_WARNINGS.json"
        lint_data = [
            {
                "level": w.level.value,
                "check": w.check,
                "message": w.message,
                "chapter": w.chapter,
                "section": w.section,
            }
            for w in warnings
        ]
        lint_path.write_text(json.dumps(lint_data, indent=2))
        logger.info("Wrote %d lint warnings to %s", len(warnings), lint_path)

    return md_path
