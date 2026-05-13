#!/usr/bin/env python3
"""Report-generation CLI (Phase 9).

Provides a proper command-line interface for generating, linting, and
publishing the SCX comprehensive benchmark report.

Usage examples::

    # Generate all formats to the default output directory
    python -m benchmarks.comprehensive.reporting.report_cli generate

    # Generate with strict lint + public profile (no Phase labels)
    python -m benchmarks.comprehensive.reporting.report_cli generate \
        --strict-lint --profile public

    # Generate from a specific results directory
    python -m benchmarks.comprehensive.reporting.report_cli generate \
        --results-dir /path/to/results/raw

    # Lint-only mode (no output written)
    python -m benchmarks.comprehensive.reporting.report_cli lint

    # Lint with public-profile checks
    python -m benchmarks.comprehensive.reporting.report_cli lint --profile public
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]  # .../scx/
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.config import (  # noqa: E402
    REPORTS_DIR, RAW_RESULTS_DIR, FIGURES_DIR,
)
from benchmarks.comprehensive.reporting.result_store import (  # noqa: E402
    ResultStore, reset_store,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------

PROFILES = {
    "default": {
        "description": "Full engineering report with all sections.",
        "public": False,
    },
    "public": {
        "description": "Public-facing report. Phase labels in headings are errors.",
        "public": True,
    },
    "engineering": {
        "description": "Engineering report with internal phase labels allowed.",
        "public": False,
    },
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def build_report(
    *,
    results_dir: Path | None = None,
    output_dir: Path | None = None,
    snapshot_dir: Path | None = None,
    formats: list[str] | None = None,
    profile: str = "default",
    strict_lint: bool = False,
) -> Path:
    """Build the full benchmark report.

    Parameters
    ----------
    results_dir : Path, optional
        Directory containing raw JSON benchmark results. Defaults to
        ``RAW_RESULTS_DIR``.
    output_dir : Path, optional
        Output directory for generated reports. Defaults to ``REPORTS_DIR``.
    snapshot_dir : Path, optional
        If set, copy final artifacts to this directory after generation
        (e.g., a timestamped snapshot path).
    formats : list[str], optional
        Output formats to generate. Defaults to ``["md", "html", "pdf"]``.
    profile : str
        Report profile to use. One of ``default``, ``public``,
        ``engineering``.
    strict_lint : bool
        If ``True``, promote manual-source warnings to errors and exit
        nonzero on any error-level lint findings.

    Returns
    -------
    Path
        Path to the generated markdown report.
    """
    if output_dir is None:
        output_dir = REPORTS_DIR
    if results_dir is None:
        results_dir = RAW_RESULTS_DIR
    if formats is None:
        formats = ["md", "html", "pdf"]

    profile_cfg = PROFILES.get(profile, PROFILES["default"])
    public = profile_cfg["public"]

    # Set up the result store from the specified results directory
    reset_store()
    store = ResultStore()
    store.load_raw_dir(results_dir)
    store.load_harmony_dir()

    # Inject the store as the default so section builders can use get_store()
    import benchmarks.comprehensive.reporting.result_store as _rs
    _rs._default_store = store

    output_dir.mkdir(parents=True, exist_ok=True)

    # Import after store setup to avoid circular imports
    from benchmarks.comprehensive.reporting.markdown import (
        generate_report_model, write_html_snapshot, _generate_pdf,
    )
    from benchmarks.comprehensive.reporting.report_model import MarkdownRenderer
    from benchmarks.comprehensive.reporting.lint import (
        collect_warnings, LintLevel,
    )

    report_model = generate_report_model()

    # ── Report lint ───────────────────────────────────────────────────
    warnings = collect_warnings(
        report_model,
        strict=strict_lint,
        public=public,
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
    md_path = None
    if "md" in formats:
        md_body = MarkdownRenderer().render(report_model)
        md_path = output_dir / "BENCHMARK_REPORT.md"
        md_path.write_text(md_body)
        logger.info("Wrote markdown report to %s", md_path)

    # ── Generate PDF ──────────────────────────────────────────────────
    if "pdf" in formats and md_path:
        pdf_path = output_dir / "BENCHMARK_REPORT.pdf"
        _generate_pdf(md_path, pdf_path)

    # ── Emit HTML snapshot ────────────────────────────────────────────
    if "html" in formats:
        try:
            write_html_snapshot(report_model, output_dir)
        except Exception as exc:
            logger.warning("HTML snapshot emission failed: %s", exc)

    # ── Write JSON manifest ───────────────────────────────────────────
    if "json" in formats:
        from benchmarks.comprehensive.reporting.report_model import JsonManifestRenderer
        json_path = output_dir / "BENCHMARK_REPORT.json"
        json_body = JsonManifestRenderer().render(report_model)
        json_path.write_text(json_body)
        logger.info("Wrote JSON report manifest to %s", json_path)

    # ── Write lint warnings manifest ──────────────────────────────────
    if warnings:
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

    # ── Snapshot copy ─────────────────────────────────────────────────
    if snapshot_dir:
        import shutil
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for name in ("BENCHMARK_REPORT.md", "BENCHMARK_REPORT.html",
                      "BENCHMARK_REPORT.pdf", "BENCHMARK_REPORT.json",
                      "LINT_WARNINGS.json", "dashboard_history.json"):
            src = output_dir / name
            if src.exists():
                shutil.copy2(src, snapshot_dir / name)
        # Copy figures directory
        figs_src = output_dir / "figures"
        if figs_src.is_dir():
            figs_dst = snapshot_dir / "figures"
            if figs_dst.exists():
                shutil.rmtree(figs_dst)
            shutil.copytree(figs_src, figs_dst)
        logger.info("Copied report snapshot to %s", snapshot_dir)

    return md_path or output_dir / "BENCHMARK_REPORT.md"


def lint_only(
    *,
    results_dir: Path | None = None,
    profile: str = "default",
    strict: bool = False,
) -> int:
    """Run lint checks without writing any output.

    Returns the number of error-level findings.
    """
    if results_dir is None:
        results_dir = RAW_RESULTS_DIR

    reset_store()
    store = ResultStore()
    store.load_raw_dir(results_dir)
    store.load_harmony_dir()

    import benchmarks.comprehensive.reporting.result_store as _rs
    _rs._default_store = store

    from benchmarks.comprehensive.reporting.markdown import generate_report_model
    from benchmarks.comprehensive.reporting.lint import collect_warnings, LintLevel

    profile_cfg = PROFILES.get(profile, PROFILES["default"])

    report = generate_report_model()
    warnings = collect_warnings(
        report,
        strict=strict,
        public=profile_cfg["public"],
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

    from collections import Counter
    by_level = Counter(w.level.value for w in warnings)
    logger.info(
        "Lint summary: %d errors, %d warnings, %d info",
        by_level.get("error", 0),
        by_level.get("warning", 0),
        by_level.get("info", 0),
    )

    errors = [w for w in warnings if w.level == LintLevel.error]
    if strict and errors:
        logger.error(
            "Strict lint: %d error(s) found. Report would fail --strict-lint.",
            len(errors),
        )
    return len(errors)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="report_cli",
        description="SCX benchmark report generator and linter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── generate ──────────────────────────────────────────────────────
    gen = subparsers.add_parser(
        "generate",
        help="Generate the benchmark report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    gen.add_argument(
        "--profile", choices=list(PROFILES.keys()), default="default",
        help="Report profile (default: %(default)s). "
             "'public' blocks internal phase labels; "
             "'engineering' includes all internal details.",
    )
    gen.add_argument(
        "--results-dir", type=Path, default=None,
        help=f"Directory containing raw JSON results (default: {RAW_RESULTS_DIR}).",
    )
    gen.add_argument(
        "--output-dir", type=Path, default=None,
        help=f"Output directory for generated reports (default: {REPORTS_DIR}).",
    )
    gen.add_argument(
        "--snapshot-dir", type=Path, default=None,
        help="If set, copy final artifacts to this timestamped snapshot directory.",
    )
    gen.add_argument(
        "--format", dest="formats", action="append",
        choices=["md", "html", "pdf", "json"],
        help="Output format(s). Can be specified multiple times. "
             "Default: md html pdf.",
    )
    gen.add_argument(
        "--strict-lint", action="store_true",
        help="Promote manual-source warnings to errors; exit nonzero on errors.",
    )

    # ── lint ──────────────────────────────────────────────────────────
    lint = subparsers.add_parser(
        "lint",
        help="Run lint checks without generating output.",
    )
    lint.add_argument(
        "--profile", choices=list(PROFILES.keys()), default="default",
        help="Report profile (default: %(default)s).",
    )
    lint.add_argument(
        "--results-dir", type=Path, default=None,
        help=f"Directory containing raw JSON results (default: {RAW_RESULTS_DIR}).",
    )
    lint.add_argument(
        "--strict", action="store_true",
        help="Exit nonzero if any error-level lint findings exist.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    if args.command == "generate":
        formats = args.formats or ["md", "html", "pdf"]
        build_report(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            snapshot_dir=args.snapshot_dir,
            formats=formats,
            profile=args.profile,
            strict_lint=args.strict_lint,
        )
        return 0

    elif args.command == "lint":
        n_errors = lint_only(
            results_dir=args.results_dir,
            profile=args.profile,
            strict=args.strict,
        )
        return 1 if args.strict and n_errors > 0 else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
