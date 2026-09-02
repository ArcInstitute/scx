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

**Metric scope.** By default a justification suppresses *every* metric on each
triple it names — regressions and absolute-floor violations alike. That is the
right default for a triple that is wholly known-bad (an OOMing suite, a
workers2 path that collapses both throughput floor and timing), but it is the
wrong one for the commonest case in practice: a *composition* change that moves
only the pooled summary metrics (``peak_rss_mb_median`` / ``median_wall_s``)
because an arm was added to an existing benchmark. Left unscoped, such a file
silently disarms every floor on the triple, including one added in the same
commit — two of them did exactly that.

So a justification may name the metrics it explains, and then it suppresses
only those::

    ---
    metrics: [peak_rss_mb_median, median_wall_s]
    triples:
      - benchmark: cellset_gather
        format: scx_auto
        dataset: census_1m
      - benchmark: index_plan
        format: scx_auto
        dataset: tabula_sapiens_100k
        metric: batches_per_sec__pyscx_index_plan_dataset_workers2
    ---

Two spellings, because both were already being written before either was read:
a file-level ``metrics:`` list, and a per-entry ``metric:`` naming one metric
for that triple alone. **Both keys are accepted at both levels** — a
``metrics:`` inside a triple entry and a ``metric:`` at the top level work too.
Rejecting one spelling per level would reintroduce the failure this field
exists to end: a scope that is silently ignored falls open to whole-triple
suppression, and that is exactly how 16 floors became unenforceable. A triple's effective scope is the union of the two, and a
triple named *without* either keeps the whole-triple default. Where two files
name the same triple, the scopes union, and an unscoped file wins (the widest
claim holds).

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

#: Which metrics a suppression covers on one triple. ``None`` means *every*
#: metric — the historical, and still default, behaviour.
MetricScope = frozenset[str] | None

#: Triple -> metric scope. Returned by :func:`load_active_triples`; a caller
#: asks :func:`suppresses` rather than reading it directly.
SuppressionMap = dict[Triple, MetricScope]


@dataclass
class Justification:
    """Parsed justification file."""
    path: Path
    triples: list[Triple] = field(default_factory=list)
    reason: str = ""
    expires: _dt.date | None = None
    prose: str = ""
    #: Per-triple metric scope. A triple absent from this map (or mapped to
    #: ``None``) suppresses every metric on that triple. Populated from the
    #: file-level ``metrics:`` list and any per-entry ``metric:``.
    metric_scopes: dict[Triple, MetricScope] = field(default_factory=dict)

    def scope_for(self, triple: Triple) -> MetricScope:
        """Metrics this file suppresses on *triple* (``None`` = all of them)."""
        return self.metric_scopes.get(triple)

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


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    """First of *keys* present in *mapping*, else ``None``.

    Lets `metric:` and `metrics:` be accepted at both the file and entry level
    without either spelling being the one that silently does nothing.
    """
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _coerce_metrics(path: Path, value: Any) -> MetricScope:
    """Normalize a ``metrics:`` / ``metric:`` field into a scope or ``None``.

    ``None`` (absent) means "every metric" — see the module docstring. A single
    string is accepted for the singular ``metric:`` spelling that committed
    files already use.

    An *empty* list is rejected rather than read as either extreme: it would
    suppress nothing at all while looking exactly like a scoped suppression,
    and that is the failure mode this whole field exists to end.
    """
    if value is None:
        return None
    if isinstance(value, str):
        names = [value]
    elif isinstance(value, (list, tuple)):
        names = list(value)
    else:
        raise ValueError(
            f"{path}: 'metric'/'metrics' must be a string or a list of "
            f"strings, got {type(value).__name__}"
        )
    cleaned = [str(n).strip() for n in names]
    if any(not n for n in cleaned):
        raise ValueError(f"{path}: 'metric'/'metrics' contains a blank name")
    if not cleaned:
        raise ValueError(
            f"{path}: 'metric'/'metrics' is empty — omit the key to suppress "
            f"every metric on the triple, or name the ones this file explains"
        )
    return frozenset(cleaned)


def _entry_scope(file_metrics: MetricScope, entry_metrics: MetricScope) -> MetricScope:
    """Combine the file-level scope with one entry's own ``metric:``.

    Neither given ⇒ ``None`` (whole triple). Otherwise the union, treating an
    absent half as empty: a file-level list applies to every entry, and a
    per-entry ``metric:`` adds to it rather than replacing it.
    """
    if file_metrics is None and entry_metrics is None:
        return None
    return (file_metrics or frozenset()) | (entry_metrics or frozenset())


def _widen(a: MetricScope, b: MetricScope) -> MetricScope:
    """Merge two scopes for the same triple; the *widest* claim wins.

    ``None`` is the widest scope there is, so it absorbs any named set. Used
    both for a triple listed twice in one file and for the same triple named by
    two files.
    """
    if a is None or b is None:
        return None
    return a | b


def suppresses(suppression: SuppressionMap, triple: Triple, metric: str) -> bool:
    """Is *metric* on *triple* suppressed by this map?

    One predicate for both gate surfaces (relative regressions and absolute
    floors), so the exit code and the report can never disagree about what was
    suppressed — they disagreed once, and the fix was to give them a single
    function to ask.
    """
    if triple not in suppression:
        return False
    scope = suppression[triple]
    return scope is None or metric in scope


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

    # Both spellings, at both levels. The singular reads better on one entry and
    # the plural on a file, but accepting only one of each is how this field
    # came to be silently ignored in the first place: four committed files
    # wrote a per-entry `metric:` that the parser dropped, and 16 absolute
    # floors went unenforceable behind it. A misspelled scope must not fail
    # open into whole-triple suppression.
    file_metrics = _coerce_metrics(
        path, _first_present(data, "metrics", "metric")
    )

    triples: list[Triple] = []
    scopes: dict[Triple, MetricScope] = {}
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
        triple = (
            str(entry["benchmark"]), str(entry["format"]), str(entry["dataset"])
        )
        triples.append(triple)
        # A per-entry `metric:` narrows this triple alone; the file-level
        # `metrics:` applies to every triple. Both were already being written
        # by committed files before either was honoured, so both are read, and
        # a triple named twice gets the union.
        entry_metrics = _coerce_metrics(
            path, _first_present(entry, "metric", "metrics")
        )
        scope = _entry_scope(file_metrics, entry_metrics)
        if triple in scopes:
            scope = _widen(scopes[triple], scope)
        scopes[triple] = scope

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
        metric_scopes=scopes,
    )


def load_active_triples(
    directory: Path, today: _dt.date | None = None,
) -> tuple[SuppressionMap, list[Justification]]:
    """Walk ``directory`` and return the merged active suppressions.

    Non-markdown files are ignored. Markdown files without front-matter are
    ignored (so a README can live alongside). Returns a
    ``{triple: metric scope}`` map plus the list of parsed justifications for
    reporting. Ask :func:`suppresses` rather than testing membership: a triple
    present in the map may be suppressed for only some of its metrics.

    Where two active files name the same triple the scopes are merged by
    :func:`_widen`, so an unscoped file's whole-triple claim survives a scoped
    one — the merge can only ever suppress more, never less, than either file
    asked for on its own.
    """
    if not directory.is_dir():
        return {}, []

    active: SuppressionMap = {}
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
            for triple in j.triples:
                scope = j.scope_for(triple)
                active[triple] = (
                    _widen(active[triple], scope) if triple in active else scope
                )
        else:
            logger.info(
                "Justification %s expired on %s; no longer suppresses %d triple(s)",
                md, j.expires, len(j.triples),
            )
    return active, all_parsed
