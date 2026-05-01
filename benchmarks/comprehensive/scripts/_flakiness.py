"""
Flakiness ledger loader (§3.7).

A *flakiness override* declares that one or more
``(benchmark, format, dataset, metric)`` rows are known-noisy on this
hardware and should be gated at a relaxed tolerance instead of the global
``--timing-tolerance``. This is distinct from a justification (which
fully suppresses a row from the regression tally): an override keeps the
gate active, just at a higher per-row threshold. A real regression that
crosses the relaxed bound still trips the gate.

Each override is a markdown file with a YAML front-matter block. Example::

    ---
    overrides:
      - benchmark: cloud_pull
        format: scx_auto
        dataset: tabula_sapiens_100k
        metric: median_wall_s   # optional; defaults to median_wall_s
        tolerance: 0.08         # required; relaxed bound for this row
    reason: "Shared SLURM node — 7% wall-time RSD across 10 runs (issue #1234)."
    expires: 2026-07-01
    ---

    Optional prose explaining what would let the override expire
    (``--exclusive`` SLURM allocation, NUMA pinning, etc.). Ends when
    the file ends; no closing fence.

``expires`` is optional and follows the same inclusive-date semantics as
justifications (active on the listed date, inactive the day after).
``metric`` defaults to ``median_wall_s`` — the dominant noise source.

When two files declare overrides for the same
``(benchmark, format, dataset, metric)`` quad the **stricter (lower)**
tolerance wins and a warning is logged: overlapping entries should be
collapsed during cleanup, not silently merged toward the more permissive
side.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


Quad = tuple[str, str, str, str]  # (benchmark, format, dataset, metric)


@dataclass
class Override:
    """One row-scoped tolerance override."""
    benchmark: str
    format: str
    dataset: str
    metric: str
    tolerance: float

    @property
    def quad(self) -> Quad:
        return (self.benchmark, self.format, self.dataset, self.metric)


@dataclass
class FlakinessFile:
    """Parsed flakiness markdown file."""
    path: Path
    overrides: list[Override] = field(default_factory=list)
    reason: str = ""
    expires: _dt.date | None = None
    prose: str = ""

    def is_active(self, today: _dt.date | None = None) -> bool:
        """Return True if this override file is still in-date.

        ``expires`` is inclusive (matches justifications): a file with
        ``expires: 2026-06-01`` is active on June 1 and inactive on June 2.
        """
        if self.expires is None:
            return True
        today = today or _dt.date.today()
        return today <= self.expires


_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n(.*))?\Z", re.DOTALL)


def _coerce_date(value: Any) -> _dt.date | None:
    """Normalize the ``expires:`` field into a ``date`` or ``None``."""
    if value is None:
        return None
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return None
        return _dt.date.fromisoformat(v)
    raise ValueError(f"unsupported type for 'expires': {type(value).__name__}")


def _parse_front_matter(block: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(block)
    except yaml.YAMLError as exc:
        raise ValueError(f"malformed YAML front-matter: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"front-matter must be a YAML mapping, got {type(data).__name__}"
        )
    return data


_REQUIRED_TRIPLE_KEYS = ("benchmark", "format", "dataset")


def parse_flakiness_file(path: Path) -> FlakinessFile | None:
    """Load one flakiness markdown file.

    Returns ``None`` if the file has no front-matter (so a README in the
    same directory is silently ignored). Raises ``ValueError`` on
    malformed front-matter so operators see a clear error rather than a
    silently-dropped override.
    """
    text = path.read_text()
    m = _FRONT_MATTER_RE.match(text)
    if m is None:
        return None
    block, prose = m.group(1), (m.group(2) or "")
    try:
        data = _parse_front_matter(block)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc

    raw_overrides = data.get("overrides") or []
    if not isinstance(raw_overrides, list):
        raise ValueError(f"{path}: 'overrides' must be a list")

    overrides: list[Override] = []
    for entry in raw_overrides:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{path}: override entry must be a mapping, got {entry!r}"
            )
        missing = [k for k in _REQUIRED_TRIPLE_KEYS if k not in entry]
        if missing:
            raise ValueError(
                f"{path}: override entry missing keys {missing}: {entry!r}"
            )
        if "tolerance" not in entry or entry["tolerance"] is None:
            raise ValueError(
                f"{path}: override entry missing required 'tolerance': {entry!r}"
            )
        try:
            tolerance = float(entry["tolerance"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: override entry tolerance={entry['tolerance']!r} "
                f"must be numeric"
            ) from exc
        if tolerance < 0:
            raise ValueError(
                f"{path}: override entry tolerance={tolerance!r} must be "
                f"non-negative (overrides relax — they don't tighten)"
            )
        metric = entry.get("metric")
        metric = "median_wall_s" if metric is None else str(metric)
        overrides.append(Override(
            benchmark=str(entry["benchmark"]),
            format=str(entry["format"]),
            dataset=str(entry["dataset"]),
            metric=metric,
            tolerance=tolerance,
        ))

    try:
        expires = _coerce_date(data.get("expires"))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc

    reason = data.get("reason", "")
    if reason is None:
        reason = ""

    return FlakinessFile(
        path=path,
        overrides=overrides,
        reason=str(reason).strip(),
        expires=expires,
        prose=prose.strip(),
    )


def load_active_overrides(
    directory: Path, today: _dt.date | None = None,
) -> tuple[dict[Quad, float], list[FlakinessFile]]:
    """Walk ``directory`` and merge every active override into a single
    ``quad → tolerance`` map.

    On duplicate keys the **stricter (lower)** tolerance wins so an
    accidentally permissive entry can't shadow a tighter one. The
    conflict is logged at WARNING. Returns the merged map plus the list
    of parsed files for reporting (active and expired alike).
    """
    if not directory.is_dir():
        return {}, []

    merged: dict[Quad, float] = {}
    sources: dict[Quad, Path] = {}
    all_parsed: list[FlakinessFile] = []
    for md in sorted(directory.glob("*.md")):
        try:
            f = parse_flakiness_file(md)
        except ValueError as exc:
            logger.warning("Skipping malformed flakiness file %s: %s", md, exc)
            continue
        if f is None:
            continue
        all_parsed.append(f)
        if not f.is_active(today):
            logger.info(
                "Flakiness file %s expired on %s; %d override(s) ignored",
                md, f.expires, len(f.overrides),
            )
            continue
        for ov in f.overrides:
            existing = merged.get(ov.quad)
            if existing is None or ov.tolerance < existing:
                if existing is not None:
                    logger.warning(
                        "Flakiness override for %s declared in both %s "
                        "(tolerance=%.4f) and %s (tolerance=%.4f); keeping "
                        "stricter %.4f",
                        ov.quad, sources[ov.quad], existing,
                        md, ov.tolerance, ov.tolerance,
                    )
                merged[ov.quad] = ov.tolerance
                sources[ov.quad] = md
            elif ov.tolerance > existing:
                logger.warning(
                    "Flakiness override for %s in %s (tolerance=%.4f) is "
                    "more permissive than the existing %s entry "
                    "(tolerance=%.4f); keeping stricter %.4f",
                    ov.quad, md, ov.tolerance,
                    sources[ov.quad], existing, existing,
                )
    return merged, all_parsed
