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
    reason: Upstream GCS client added a mandatory HTTP/2 header (issue #1234).
    expires: 2026-06-01
    ---

    Optional prose explaining the decision — shown in the gate's failure
    summary. Ends when the file ends; no closing fence.

``expires`` is optional. When absent the justification never expires; when
present and in the past relative to ``date.today()`` the justification is
treated as inactive and its triples are NOT suppressed.

This module is intentionally dependency-free — no PyYAML requirement — so
the regression gate runs even on minimal envs (the harness env always has
yaml, but self-tests and CI shells should not need to pip-install). The
parser only handles the narrow YAML shape used by justification files.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

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


def _parse_scalar(value: str) -> str:
    v = value.strip()
    if (v.startswith("'") and v.endswith("'")) or (
        v.startswith('"') and v.endswith('"')
    ):
        return v[1:-1]
    return v


def _parse_date(value: str) -> _dt.date | None:
    v = _parse_scalar(value)
    if not v:
        return None
    return _dt.date.fromisoformat(v)


def _parse_front_matter(block: str) -> dict:
    """Parse the narrow YAML shape justification files use.

    Supports:
      - top-level ``key: scalar``
      - ``triples:`` followed by ``- benchmark: x\\n    format: y\\n    dataset: z``

    Does NOT attempt general YAML. Raises ValueError on any unexpected shape.
    """
    lines = block.splitlines()
    out: dict = {}
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if line.startswith(" "):
            raise ValueError(f"unexpected indent at top level: {raw!r}")
        if ":" not in line:
            raise ValueError(f"expected 'key:' line, got: {raw!r}")
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()
        if key == "triples":
            # Consume indented list entries.
            triples: list[dict] = []
            i += 1
            current: dict | None = None
            while i < len(lines):
                entry_raw = lines[i]
                entry = entry_raw.rstrip()
                if not entry.strip():
                    i += 1
                    continue
                if not entry.startswith(" ") and not entry.startswith("\t"):
                    break
                stripped = entry.lstrip()
                if stripped.startswith("- "):
                    if current is not None:
                        triples.append(current)
                    current = {}
                    stripped = stripped[2:]
                if ":" not in stripped:
                    raise ValueError(
                        f"triples entry missing ':' — got: {entry_raw!r}"
                    )
                ek, _, ev = stripped.partition(":")
                if current is None:
                    raise ValueError(
                        f"triples entry before any '- ' marker: {entry_raw!r}"
                    )
                current[ek.strip()] = _parse_scalar(ev)
                i += 1
            if current is not None:
                triples.append(current)
            out["triples"] = triples
            continue
        out[key] = rest
        i += 1
    return out


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
    data = _parse_front_matter(block)
    triples = []
    for entry in data.get("triples", []) or []:
        missing = [k for k in ("benchmark", "format", "dataset") if k not in entry]
        if missing:
            raise ValueError(
                f"{path}: triple entry missing keys {missing}: {entry!r}"
            )
        triples.append((entry["benchmark"], entry["format"], entry["dataset"]))

    expires_val = data.get("expires")
    expires = _parse_date(expires_val) if expires_val else None

    return Justification(
        path=path,
        triples=triples,
        reason=_parse_scalar(data.get("reason", "")),
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
