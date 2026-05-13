#!/usr/bin/env python3
"""Check that every numeric benchmark claim in README.md has a backing manifest.

This script parses the benchmark tables in README.md, extracts numeric
performance claims, and verifies that each has a corresponding raw JSON
result in ``benchmarks/comprehensive/results/raw/`` or a row in
``benchmarks/comprehensive/results/baselines/LATEST/summary.json``.

Exit codes:
    0 — all claims are backed
    1 — one or more claims lack a manifest entry
    2 — script error (e.g. README not found)

Usage:
    python benchmarks/scripts/check_readme_manifests.py [--verbose] [--readme PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_README = REPO_ROOT / "README.md"
RAW_DIR = REPO_ROOT / "benchmarks" / "comprehensive" / "results" / "raw"
BASELINE_SUMMARY = (
    REPO_ROOT
    / "benchmarks"
    / "comprehensive"
    / "results"
    / "baselines"
    / "LATEST"
    / "summary.json"
)

# ---------------------------------------------------------------------------
# Table definitions: which README tables contain benchmark claims
# ---------------------------------------------------------------------------

# The README has several distinct benchmark tables. We identify each by a
# unique string from its header row and define how to resolve its rows.


class TableSpec(NamedTuple):
    """Describes a known benchmark table in the README."""

    header_match: str  # unique substring in the table's header row
    benchmark: str  # benchmark type for manifest lookup
    dataset_column: int | None  # column index (0-based) containing dataset name, or None
    default_dataset: str | None  # default dataset if table is single-dataset


# Tables we know how to resolve:
KNOWN_TABLES: list[TableSpec] = [
    # Compression / size table: per-dataset rows
    TableSpec(
        header_match="Best ratio",
        benchmark="compression",
        dataset_column=0,
        default_dataset=None,
    ),
    # Memory streaming table: single-dataset (census_1m)
    TableSpec(
        header_match="h5ad (full load)",
        benchmark="memory",
        dataset_column=None,
        default_dataset="census_1m",
    ),
    # Training loader table: single-dataset (census_1m)
    TableSpec(
        header_match="batches/sec",
        benchmark="ml_loader",
        dataset_column=None,
        default_dataset="census_1m",
    ),
    # Perturbation metrics table
    TableSpec(
        header_match="20K × 2K × 50",
        benchmark="cell_eval_parity_perf",
        dataset_column=None,
        default_dataset="pert_synth_10k",
    ),
    # Headline benchmarks table
    TableSpec(
        header_match="SCX advantage",
        benchmark="_headline",
        dataset_column=None,
        default_dataset="census_1m",
    ),
]

# For the headline table, each row maps to a specific (benchmark, format).
# Keys must match the `benchmark` field emitted by the harness into results/raw/.
HEADLINE_ROW_MAP: dict[str, tuple[str, str]] = {
    "file size": ("compression", "scx_auto"),
    "read (full load": ("read_full", "scx_auto"),
    "column projection": ("read_selective", "scx_auto"),
    "parallel read": ("parallel_scaling", "scx_auto"),
    "parallel write": ("parallel_write_scaling", "scx_auto"),
    "out-of-core": ("memory", "scx_auto"),
    "training loader": ("ml_loader", "scx_auto"),
    "gpu end-to-end": ("accel_pca", "scx_auto"),  # GPU pipeline headline uses PCA as proxy
    "selective query": ("index_plan", "scx_auto"),
    "append": ("fragment_ops", "scx_auto"),
}

# Maps human-readable dataset names to result-file keys.
DATASET_ALIASES: dict[str, str] = {
    "pbmc 3k": "pbmc3k",
    "pbmc3k": "pbmc3k",
    "pbmc 10k": "pbmc10k",
    "pbmc10k": "pbmc10k",
    "smart-seq2": "smartseq2",
    "smartseq2": "smartseq2",
    "tabula sapiens": "tabula_sapiens_100k",
    "cellxgene census 1m": "census_1m",
    "census 1m": "census_1m",
    "cellxgene census 5m": "census_5m",
    "census 5m": "census_5m",
}

# Tables to skip entirely (non-benchmark comparison tables).
SKIP_TABLE_HEADERS: list[str] = [
    "Threading model",  # multithreading summary
    "h5ad (HDF5)",  # locking comparison
    "h5ad | Zarr | TileDB-SOMA",  # format feature comparison
    "Crate",  # architecture table
]


class Claim(NamedTuple):
    """A numeric benchmark claim extracted from README.md."""

    line_no: int
    value: str
    context: str
    dataset: str
    benchmark: str
    format_key: str


def _normalise(text: str) -> str:
    """Lowercase, strip markdown bold/italic/backtick, normalise whitespace."""
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _resolve_dataset_from_cell(cell_text: str) -> str | None:
    """Try to resolve a dataset key from a table cell."""
    norm = _normalise(cell_text)
    for alias, key in sorted(DATASET_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in norm:
            return key
    return None


# Regex for extracting numeric values from table cells.
# Matches: 2.35, 1,405, 286, 4.2, ~11, etc.
_NUM_RE = re.compile(
    r"(?:~?\*{0,2})"
    r"(\d[\d,]*\.?\d*)"
    r"\*{0,2}"
    r"(?:\s*(?:×|s|ms|GB|MB|%|batches/s(?:ec)?))?"
)



def extract_claims(readme_path: Path) -> list[Claim]:
    """Parse README.md and extract benchmark claims from known tables."""
    claims: list[Claim] = []
    lines = readme_path.read_text().splitlines()

    active_table: TableSpec | None = None
    in_table = False
    saw_separator = False

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Non-table lines reset state
        if not stripped.startswith("|"):
            active_table = None
            in_table = False
            saw_separator = False
            continue

        norm_line = _normalise(stripped)

        # Check if this is a table we should skip
        if any(skip.lower() in norm_line for skip in SKIP_TABLE_HEADERS):
            active_table = None  # mark as skip
            in_table = True
            saw_separator = False
            continue

        # Separator row
        if re.match(r"^\|[\s\-:|]+\|$", stripped):
            saw_separator = True
            continue

        # Header row: try to identify the table
        if not saw_separator:
            for spec in KNOWN_TABLES:
                if spec.header_match.lower() in norm_line:
                    active_table = spec
                    in_table = True
                    break
            continue

        # We're in a data row. If no known table, skip.
        if active_table is None:
            continue

        # Split into cells
        cells = [c.strip() for c in stripped.split("|")]
        cells = [c for c in cells if c]

        if len(cells) < 2:
            continue

        # Resolve dataset
        if active_table.dataset_column is not None and active_table.dataset_column < len(cells):
            dataset = _resolve_dataset_from_cell(cells[active_table.dataset_column])
        else:
            dataset = None
        if not dataset:
            dataset = active_table.default_dataset
        if not dataset:
            continue

        # Resolve benchmark + format
        benchmark = active_table.benchmark
        format_key = "scx_auto"

        if benchmark == "_headline":
            # Special case: resolve per-row
            for label, (bench, fmt) in HEADLINE_ROW_MAP.items():
                if label in norm_line:
                    benchmark = bench
                    format_key = fmt
                    break
            else:
                continue  # unknown headline row

        # Extract numeric values from data cells (skip first cell = label)
        for cell in cells[1:]:
            for m in _NUM_RE.finditer(cell):
                raw_num = m.group(1)
                try:
                    float(raw_num.replace(",", ""))
                except ValueError:
                    continue

                claims.append(
                    Claim(
                        line_no=i,
                        value=raw_num,
                        context=stripped[:120],
                        dataset=dataset,
                        benchmark=benchmark,
                        format_key=format_key,
                    )
                )

    return claims


# ---------------------------------------------------------------------------
# Manifest verification
# ---------------------------------------------------------------------------


def load_raw_manifests(raw_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all raw JSON results, keyed by benchmark__format__dataset."""
    manifests: dict[str, dict[str, Any]] = {}
    if not raw_dir.is_dir():
        return manifests
    for path in raw_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            key = f"{data.get('benchmark', '')}__{data.get('format', '')}__{data.get('dataset', '')}"
            manifests[key] = data
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Failed to load or parse manifest {path}: {e}", file=sys.stderr)
    return manifests


def load_baseline_summary(path: Path) -> dict[str, dict[str, Any]]:
    """Load the baseline summary.json rows."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
        return data.get("rows", {})
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Failed to load or parse baseline summary {path}: {e}", file=sys.stderr)
        return {}


def check_claim(
    claim: Claim,
    raw_manifests: dict[str, dict[str, Any]],
    baseline_rows: dict[str, dict[str, Any]],
) -> tuple[bool, str]:
    """Check if a claim has a backing manifest entry.

    Returns (is_backed, reason).
    """
    key = f"{claim.benchmark}__{claim.format_key}__{claim.dataset}"

    # Exact match in raw results
    if key in raw_manifests:
        return True, f"backed by raw/{key}.json"

    # Exact match in baseline summary
    if key in baseline_rows:
        return True, f"backed by baselines/LATEST ({key})"

    # Broader: same benchmark + dataset, any format.
    # Use the manifest's actual fields rather than __-splitting the key,
    # because format strings can themselves contain "__" (e.g.
    # "accel_pca__pyscx_gpu_cov").
    for rk, rdata in raw_manifests.items():
        if rdata.get("benchmark") == claim.benchmark and rdata.get("dataset") == claim.dataset:
            return True, f"backed by raw/{rk}.json (format: {rdata.get('format', '?')})"

    for bk in baseline_rows:
        parts = bk.split("__")
        if len(parts) >= 3 and parts[0] == claim.benchmark and parts[-1] == claim.dataset:
            return True, f"backed by baseline ({bk})"

    # NOTE: A dataset-only fallback was removed here. Previously, any manifest
    # for the same dataset (regardless of benchmark or format) would satisfy
    # the check — too permissive for a verification gate.

    return False, f"no manifest for {key}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that README.md benchmark claims have backing manifests."
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=DEFAULT_README,
        help="Path to README.md (default: repo root)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print details for every claim, not just failures",
    )
    args = parser.parse_args()

    if not args.readme.is_file():
        print(f"ERROR: README not found at {args.readme}", file=sys.stderr)
        return 2

    claims = extract_claims(args.readme)
    if not claims:
        print("No benchmark claims found in README.md")
        return 0

    raw_manifests = load_raw_manifests(RAW_DIR)
    baseline_rows = load_baseline_summary(BASELINE_SUMMARY)

    print(f"Found {len(claims)} benchmark claims in README.md")
    print(f"Loaded {len(raw_manifests)} raw manifests, {len(baseline_rows)} baseline rows")
    print()

    backed = 0
    unbacked = 0

    for claim in claims:
        is_backed, reason = check_claim(claim, raw_manifests, baseline_rows)

        if is_backed:
            backed += 1
            if args.verbose:
                print(f"  ✓ L{claim.line_no}: {claim.value} — {reason}")
        else:
            unbacked += 1
            print(
                f"  ✗ L{claim.line_no}: {claim.value} — {reason}\n"
                f"      {claim.context}"
            )

    print()
    print(f"Summary: {backed} backed, {unbacked} unbacked (of {len(claims)} claims)")

    if unbacked > 0:
        print("\nFAILED: some claims lack backing manifests.")
        return 1

    print("\nPASSED: all claims are backed by manifest entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
