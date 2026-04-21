"""
Justification markdown loader (Phase G.3).

Each justification is a markdown file with a YAML front-matter block naming
the ``(benchmark, format, dataset)`` triples suppressed by the regression
gate. Example file::

    ---
    triples:
      - benchmark: read_full
        format: scx_auto
        dataset: census_1m
      - benchmark: cloud_push
        format: scx_auto
        dataset: pbmc3k
    reason: "Upstream GCS client added a mandatory HTTP/2 header (issue #1234)."
    expires: 2026-06-01
    ---

    Optional prose explaining the decision — shown in the gate's failure
    summary. Ends when the file ends; no closing fence.

``expires`` is optional. When absent the justification never expires; when
present and in the past relative to ``date.today()`` the justification is
treated as inactive and its triples are NOT suppressed.

Uses ``yaml.safe_load`` for the front-matter so the full YAML syntax
(quoted values, multi-line strings, typed dates) is supported. PyYAML is
already a hard dep of the regression gate via
``compare_against_baseline.py``, so this does not add a new requirement.

Note: because ``yaml.safe_load`` treats ``#`` as a comment marker,
reasons or prose containing ``#`` must be quoted (the ``reason:`` line
in the example above shows the recommended quoting).
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


Triple = tuple[str, str, str]  # (benchmark, format, dataset)


@dataclass
class Justification:
    """Parsed justification file."""
    path: Path
    triples: list[Triple] = field(default_factory=list)
    reason: str = ""
    expires: _dt.date | None = None
    prose: str = ""

    def is_active(self, today: _dt.date | None = None) -> bool:
        """Return True if this justification is still in-date.

        ``expires`` is inclusive — a file with ``expires: 2026-06-01`` is
        active on June 1 and inactive on June 2.
        """
        if self.expires is None:
            return True
        today = today or _dt.date.today()
        return today <= self.expires


_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n(.*))?\Z", re.DOTALL)


def _coerce_date(value: Any) -> _dt.date | None:
    """Normalize the ``expires:`` field into a ``date`` or ``None``.

    ``yaml.safe_load`` natively parses ``YYYY-MM-DD`` into ``date``, but
    quoted strings and whitespace-only values still arrive as ``str`` so
    we handle both.
    """
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
    """Parse the justification front-matter block via ``yaml.safe_load``.

    Raises ``ValueError`` (not ``yaml.YAMLError``) so downstream callers
    see a uniform error type regardless of parser substitutions.
    """
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


def parse_justification(path: Path) -> Justification | None:
    """Load one justification markdown file.

    Returns ``None`` if the file has no front-matter (not a justification —
    e.g. a README in the same directory). Raises ``ValueError`` when the
    front-matter is malformed so operators get a clear error rather than a
    silently-ignored justification.
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

    triples: list[Triple] = []
    raw_triples = data.get("triples") or []
    if not isinstance(raw_triples, list):
        raise ValueError(f"{path}: 'triples' must be a list")
    for entry in raw_triples:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{path}: triple entry must be a mapping, got {entry!r}"
            )
        missing = [k for k in ("benchmark", "format", "dataset") if k not in entry]
        if missing:
            raise ValueError(
                f"{path}: triple entry missing keys {missing}: {entry!r}"
            )
        triples.append(
            (str(entry["benchmark"]), str(entry["format"]), str(entry["dataset"]))
        )

    try:
        expires = _coerce_date(data.get("expires"))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc

    reason = data.get("reason", "")
    if reason is None:
        reason = ""

    return Justification(
        path=path,
        triples=triples,
        reason=str(reason).strip(),
        expires=expires,
        prose=prose.strip(),
    )


def load_active_triples(
    directory: Path, today: _dt.date | None = None,
) -> tuple[set[Triple], list[Justification]]:
    """Walk ``directory`` and return the union of all active suppressed triples.

    Non-markdown files are ignored. Markdown files without front-matter are
    ignored (so a README can live alongside). Returns the set of triples
    plus the list of parsed justifications for reporting.
    """
    if not directory.is_dir():
        return set(), []

    active: set[Triple] = set()
    all_parsed: list[Justification] = []
    for md in sorted(directory.glob("*.md")):
        try:
            j = parse_justification(md)
        except ValueError as exc:
            logger.warning("Skipping malformed justification %s: %s", md, exc)
            continue
        if j is None:
            continue
        all_parsed.append(j)
        if j.is_active(today):
            active.update(j.triples)
        else:
            logger.info(
                "Justification %s expired on %s; no longer suppresses %d triple(s)",
                md, j.expires, len(j.triples),
            )
    return active, all_parsed
