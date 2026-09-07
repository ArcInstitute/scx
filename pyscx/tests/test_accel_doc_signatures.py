"""Docs-vs-runtime signature guard for `pyscx.accel.*`.

`test_experiment_stub_coverage.py` does this for `Experiment.to_anndata` from a
hand-written `_DOC_SIGNATURES` table of `(file, method, regex)` rows. That shape
does not scale here: `docs/api.md` and `skills/scx-usage/reference/processing.md`
carry roughly seventy `pyscx.accel.*` signature bullets between them, so a table
would rot exactly like the docs it guards and a newly added bullet would be
covered by nothing. This module **discovers** the bullets instead.

By the time it was written, seven copies had already drifted — `pca` in both
files, `log1p` in both, `normalize_total`, `knockdown_efficiency`, and
`adjusted_mutual_info`. Two of them were the dangerous direction: the skills
copy of `pca` documented a `layer=` parameter that does not exist, and
`adjusted_mutual_info` renamed `labels_a`/`labels_b` to `a`/`b`. A reader who
copies either gets `TypeError`.

What is compared: parameter **names, order, and keyword-only-ness**, against
`inspect.signature` of the runtime function (which is the pyo3
`__text_signature__` for all of these). Kinds are included because the accel
signatures do carry `*` separators, unlike `to_anndata` — a kwarg drifting
across the `*` changes the calling contract without changing the name list.

What is **not** compared: default *values*. `epsilon=1e-9` against a repr of
`1e-09`, and `"csr"` against `'csr'`, would make the guard brittle without
reaching a new class of bug — a bullet with a stale default almost always has a
stale name list too (`normalize_total(target_sum=10000.0)` was also missing
`device`). If that assumption ever stops holding, add the comparison rather than
loosening this one.

The docs also state kwargs in a *second*, non-signature shape: the accel
compatibility matrix in `docs/scanpy.md`. That table is deliberately incomplete
(rows read "(PCA + neighbors kwargs, see below)"), so it is checked one way only
— see `test_scanpy_matrix_names_no_phantom_kwargs`.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re

import pytest

import pyscx
import pyscx.accel as accel

# The files carrying hand-maintained accel signature bullets, and whether a
# bare (unqualified) name in one of them is an accel function.
#
# `docs/api.md` documents several APIs and **must** be read qualified-only: its
# `ScxBackedSparseDataset` section has `- \`row_sums()\` / \`col_sums()\``
# bullets for the *handle* methods, whose names collide with
# `pyscx.accel.col_sums` and whose signatures are empty. The skills reference is
# the accel usage page, so bare names there are ours.
_DOC_FILES = {
    "docs/api.md": "qualified",
    "skills/scx-usage/reference/processing.md": "bare-ok",
}

# A markdown bullet, and then every ``name(`` inside it. Names are matched
# loosely and filtered against the runtime module afterwards, so a
# `sc.pp.log1p(...)` reference or an `Experiment` method bullet is simply not
# collected — but a `pyscx.accel.`-qualified name that does not resolve is
# reported rather than skipped (see `_collect`).
_BULLET_LINE = re.compile(r"^\s*[-*] ")
_ANY_SIGNATURE = re.compile(r"`(pyscx\.accel\.)?([A-Za-z_]\w*)\(")

# Enough signatures that an over-tightened regex fails loudly instead of
# passing vacuously. 76 were collected when this was written — 41 before the
# sweep learned to read past the first signature on a bullet; the floor is
# deliberately below that so adding or removing one bullet is not a failure.
_MIN_BULLETS = 65

# The elision marker docs use when a bullet deliberately abbreviates a long
# keyword-only tail (`harmony_integrate`) or declares nothing at all
# (`energy_distance_details(...)`).
_ELLIPSIS = "..."

# `def _f(...)` is not valid Python, so the marker is spliced in as a name.
_ELLIPSIS_PARAM = "_scx_doc_elision"


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "Cargo.toml").exists() and (parent / "docs").is_dir():
            return parent
    raise AssertionError("could not locate the repo root from the test file")


def _balanced_group(text: str, open_at: int) -> str | None:
    """The contents of the parenthesis group opening at `text[open_at]`.

    Stops at the *matching* close paren, which is what keeps a trailing return
    annotation (`) -> pandas.DataFrame`) and any prose parentheses after the
    bullet out of the captured signature. A naive `\\((.*?)\\)` regex runs past
    the arrow and mis-parses the ~19 `api.md` bullets that carry one.
    """
    assert text[open_at] == "("
    depth = 0
    quote: str | None = None
    for i in range(open_at, len(text)):
        ch = text[i]
        if quote is not None:
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1 : i]
    return None


class _NoDefault:
    """Distinguishes "no default" from a documented default of `None`."""

    def __repr__(self) -> str:  # pragma: no cover - debug output only
        return "<no default>"


NO_DEFAULT = _NoDefault()

# A documented default that is not a Python literal (`dense_max_elems=...`,
# a sentinel spelled in prose). Names and kinds are still compared; the value
# is not.
OPAQUE = object()


def parse_doc_params(signature: str) -> list[tuple[str, str, object]]:
    """`(name, kind, default)` per documented parameter, in order.

    Parsed with `ast`, not by hand: the signature is spliced into
    `def _f(<signature>): pass`, which gives names, `*`/`/` separators,
    `*args`/`**kwargs` and defaults from one grammar that already agrees with
    `inspect.signature`. Two substitutions make the docs' shorthand parseable —
    `#` comments are stripped (the `docs/scanpy.md` fenced blocks annotate every
    line, and `# None = load all layers` would otherwise contribute a parameter
    called `None`), and a bare `...` elision marker becomes a named placeholder,
    since `def _f(...)` is not valid Python.

    `default` is the evaluated literal where the docs write one, `NO_DEFAULT`
    where they write none, and `OPAQUE` where the literal will not evaluate.
    """
    stripped = re.sub(r"#.*", "", signature)
    # `...` on its own is the elision marker; `x=...` is a documented default.
    stripped = re.sub(
        r"(?<![\w=])\.\.\.(?![\w=])", f"{_ELLIPSIS_PARAM}=None", stripped
    )
    try:
        tree = ast.parse(f"def _f({stripped}): pass")
    except SyntaxError as exc:  # a bullet that is not a signature at all
        raise AssertionError(f"cannot parse doc signature {signature!r}: {exc}") from exc
    fn = tree.body[0]
    a = fn.args

    def defaults_for(args, defaults):
        pad = [NO_DEFAULT] * (len(args) - len(defaults))
        return pad + list(defaults)

    out: list[tuple[str, str, object]] = []
    positional = a.posonlyargs + a.args
    for arg, dflt in zip(positional, defaults_for(positional, a.defaults)):
        kind = "POSITIONAL_ONLY" if arg in a.posonlyargs else "POSITIONAL_OR_KEYWORD"
        out.append((arg.arg, kind, _literal(dflt)))
    if a.vararg:
        out.append((a.vararg.arg, "VAR_POSITIONAL", NO_DEFAULT))
    for arg, dflt in zip(a.kwonlyargs, a.kw_defaults):
        out.append((arg.arg, "KEYWORD_ONLY", _literal(dflt)))
    if a.kwarg:
        out.append((a.kwarg.arg, "VAR_KEYWORD", NO_DEFAULT))

    # The elision placeholder carries no kind or default of its own.
    return [
        (_ELLIPSIS, _ELLIPSIS, NO_DEFAULT) if n == _ELLIPSIS_PARAM else (n, k, d)
        for n, k, d in out
    ]


def _literal(node) -> object:
    if node is None or isinstance(node, _NoDefault):
        return NO_DEFAULT if node is None else node
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return OPAQUE
    # `pert_col=...` is the docs' shorthand for "the default shown above", used
    # where a bullet lists sibling functions that share a parameter. It declares
    # that the parameter exists and where, not what it defaults to.
    return OPAQUE if value is Ellipsis else value


def _runtime_params(fn) -> list[tuple[str, str, object]]:
    return [
        (
            p.name,
            p.kind.name,
            NO_DEFAULT if p.default is inspect.Parameter.empty else p.default,
        )
        for p in inspect.signature(fn).parameters.values()
    ]


def _same_default(doc: object, runtime: object) -> bool:
    """Whether a documented default matches the runtime one.

    `OPAQUE` (a prose sentinel) always matches: the docs are not claiming a
    value. Otherwise compare the evaluated literals, so `1e-8` matches `1e-08`
    and `"csr"` matches `'csr'` — the two brittleness cases that were the reason
    for not comparing defaults at all, and which parsing removes.
    """
    if doc is OPAQUE or runtime is Ellipsis:
        # `runtime is Ellipsis` is a pyo3 sentinel default — `pflog`'s
        # `dense_max_elems=...` resolves to a computed limit at call time, so
        # the docs' concrete figure documents the resolved value, not the
        # declared one, and comparing them would be wrong rather than strict.
        return True
    if isinstance(doc, _NoDefault) or isinstance(runtime, _NoDefault):
        return isinstance(doc, _NoDefault) and isinstance(runtime, _NoDefault)
    if isinstance(doc, float) or isinstance(runtime, float):
        try:
            return float(doc) == float(runtime)
        except (TypeError, ValueError):
            return False
    return doc == runtime


def _matches(
    doc: list[tuple[str, str, object]], runtime: list[tuple[str, str, object]]
) -> bool:
    """Whether `doc` describes `runtime`, treating `...` as "zero or more".

    Exact match on name, kind and default when the bullet carries no elision.
    With one, the named parameters on either side of it must still appear in
    order and agree — so `harmony_integrate(adata, key, *, basis=…, ...,
    random_state=0, device="auto")` passes while a phantom or misspelled name
    beside the `...` does not. A bullet that is only `(...)` declares nothing
    and matches anything, which is honest: it documents nothing to drift.
    """

    def one(d, r):
        return d[0] == r[0] and d[1] == r[1] and _same_default(d[2], r[2])

    if not any(name == _ELLIPSIS for name, _, _ in doc):
        return len(doc) == len(runtime) and all(one(d, r) for d, r in zip(doc, runtime))
    reachable = {0}
    for entry in doc:
        if entry[0] == _ELLIPSIS:
            nxt = set()
            for r in reachable:
                nxt.update(range(r, len(runtime) + 1))
            reachable = nxt
            continue
        nxt = {r + 1 for r in reachable if r < len(runtime) and one(entry, runtime[r])}
        if not nxt:
            return False
        reachable = nxt
    return len(runtime) in reachable


def _leading_signatures(line: str) -> list[tuple[bool, str, int]]:
    """The bullet's leading run of signatures: `(qualified, name, paren_idx)`.

    Bullets in these files are `` - `sig`[, `sig`…] — prose ``, and a signature
    may carry a return annotation (`` `f(x) -> T` ``). Only that leading run is
    a signature list; everything after it is prose, and prose is full of
    parenthesised *mentions* — `` `pca(mask_var=…)` ``, `` `log1p(4α·x)` `` —
    which are not signatures and must not be parsed as any. Reading the whole
    line instead (the first attempt at covering multi-function bullets)
    collected those and produced both false failures and a `SyntaxError`.
    """
    m = re.match(r"^\s*[-*] ", line)
    if not m:
        return []
    pos = m.end()
    out: list[tuple[bool, str, int]] = []
    while True:
        m = re.compile(r"`(pyscx\.accel\.)?([A-Za-z_]\w*)\(").match(line, pos)
        if not m:
            return out
        body = _balanced_group(line, m.end() - 1)
        if body is None:
            return out
        out.append((m.group(1) is not None, m.group(2), m.end() - 1))
        # Past the closing paren: an optional return annotation, the closing
        # backtick, then `, ` / ` / ` to continue the run.
        after = line.index(")", m.end() - 1 + len(body))
        rest = line[after + 1 :]
        cont = re.match(r"(?:\s*(?:->|→)[^`]*)?`\s*(?:,|/)\s*", rest)
        if not cont:
            return out
        pos = after + 1 + cont.end()


def _collect() -> tuple[list[tuple[str, int, str, str]], list[str]]:
    """Every accel signature in a bullet's leading signature run.

    Two things the first version got wrong, both found in review:

    * It matched only the *first* signature of a bullet, so every one after it
      was invisible. `skills/…/processing.md` puts three eval metrics on one
      line, and the second (`knockdown_efficiency`) was missing `device=` —
      exactly the drift this module exists to catch, sitting in a blind spot.
    * It silently dropped a bullet whose name no longer resolves on
      `pyscx.accel`, so an API rename would leave a stale bullet documented and
      unchecked, and the floor below would not notice one disappearance. A
      `pyscx.accel.`-qualified name that does not resolve is now returned
      separately and fails.
    """
    root = _repo_root()
    found: list[tuple[str, int, str, str]] = []
    unresolved: list[str] = []
    for rel, mode in _DOC_FILES.items():
        text = (root / rel).read_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            for qualified, name, paren in _leading_signatures(line):
                if mode == "qualified" and not qualified:
                    continue
                fn = getattr(accel, name, None)
                if fn is None or not callable(fn):
                    if qualified:
                        unresolved.append(f"{rel}:{lineno} `pyscx.accel.{name}`")
                    continue
                found.append((rel, lineno, name, _balanced_group(line, paren) or ""))
    return found, unresolved


_BULLETS, _UNRESOLVED = _collect()


def test_the_sweep_found_bullets():
    """A regex that matches nothing would make every other test here vacuous."""
    assert len(_BULLETS) >= _MIN_BULLETS, (
        f"only {len(_BULLETS)} accel signature bullets collected from "
        f"{list(_DOC_FILES)} (expected at least {_MIN_BULLETS}). Either the "
        "bullet shape changed or `_ANY_SIGNATURE` stopped matching it — a silent "
        "zero-coverage pass is the failure this guards against."
    )
    # Both files must contribute; a path typo would otherwise pass on the other.
    by_file = {rel for rel, _, _, _ in _BULLETS}
    assert by_file == set(_DOC_FILES), f"no bullets found in {set(_DOC_FILES) - by_file}"


@pytest.mark.parametrize(
    "doc_path, lineno, name, signature",
    _BULLETS,
    ids=[f"{rel}:{ln}:{name}" for rel, ln, name, _ in _BULLETS],
)
def test_doc_signature_matches_runtime(doc_path, lineno, name, signature):
    documented = parse_doc_params(signature)
    runtime = _runtime_params(getattr(accel, name))

    assert _matches(documented, runtime), (
        f"{doc_path}:{lineno}'s `{name}(...)` signature has drifted from the "
        f"runtime one.\n  doc:     {documented}\n  runtime: {runtime}\n"
        "It is a hand-maintained copy; update it in the same commit as the Rust "
        "change. Names, order and keyword-only-ness all count — a bullet that "
        "moves a kwarg across the `*`, or names one that does not exist, "
        "documents a different calling contract."
    )


def test_scanpy_matrix_names_no_phantom_kwargs():
    """`docs/scanpy.md`'s accel matrix may abbreviate, but not invent.

    The table's two kwarg columns are split by provenance (scanpy-parity vs
    scx-only) and are deliberately **incomplete** — several rows read
    "(PCA + neighbors kwargs, see below)", and others omit a long tail on
    purpose. So completeness is not checked and must not be: a future reader who
    "fixes" that would be forcing ~25 doc edits to satisfy a contract the table
    never claimed. What is checked is the direction that is always a bug — a
    named kwarg that no longer exists, which is what a rename leaves behind.
    """
    root = _repo_root()
    checked = 0
    phantom: list[str] = []
    for lineno, line in enumerate(
        (root / "docs/scanpy.md").read_text().splitlines(), 1
    ):
        m = re.match(r"^\|\s*`(\w+)`\s*\|", line)
        if not m:
            continue
        fn = getattr(accel, m.group(1), None)
        if fn is None or not callable(fn):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        checked += 1
        params = set(inspect.signature(fn).parameters)
        for kw in sorted(set(re.findall(r"`([a-z_][a-z_0-9]*)`", cells[3] + " " + cells[4]))):
            if kw not in params:
                phantom.append(f"docs/scanpy.md:{lineno} `{m.group(1)}` names `{kw}`")

    assert checked >= 20, (
        f"only {checked} accel rows matched in docs/scanpy.md's compatibility "
        "matrix — the row shape changed and this check went vacuous"
    )
    assert not phantom, (
        "docs/scanpy.md's accel matrix names kwargs that do not exist at "
        "runtime:\n  " + "\n  ".join(phantom)
    )


# ---------------------------------------------------------------------------
# The parser itself. Every shape below is one that actually appears in the docs;
# without these, a regression that makes `parse_doc_params` return `[]` would
# turn the sweep green rather than red.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "signature, expected",
    [
        # Plain, and the leading parameter has no default — the shape
        # `_doc_params` in test_experiment_stub_coverage.py cannot read.
        (
            'adata, n_comps=50, device="auto"',
            [
                ("adata", "POSITIONAL_OR_KEYWORD", NO_DEFAULT),
                ("n_comps", "POSITIONAL_OR_KEYWORD", 50),
                ("device", "POSITIONAL_OR_KEYWORD", "auto"),
            ],
        ),
        # A `*` separator flips the tail keyword-only.
        (
            "adata, qc_vars=None, *, layer=None, percent_top=None",
            [
                ("adata", "POSITIONAL_OR_KEYWORD", NO_DEFAULT),
                ("qc_vars", "POSITIONAL_OR_KEYWORD", None),
                ("layer", "KEYWORD_ONLY", None),
                ("percent_top", "KEYWORD_ONLY", None),
            ],
        ),
        # An elided keyword-only tail (`harmony_integrate`).
        (
            'adata, key, *, basis="X_pca", ..., device="auto"',
            [
                ("adata", "POSITIONAL_OR_KEYWORD", NO_DEFAULT),
                ("key", "POSITIONAL_OR_KEYWORD", NO_DEFAULT),
                ("basis", "KEYWORD_ONLY", "X_pca"),
                ("...", "...", NO_DEFAULT),
                ("device", "KEYWORD_ONLY", "auto"),
            ],
        ),
        # A wholly elided bullet (`energy_distance_details(...)`).
        ("...", [("...", "...", NO_DEFAULT)]),
        # A default containing a comma inside brackets must not split.
        (
            'adata, metrics=("a", "b"), device="auto"',
            [
                ("adata", "POSITIONAL_OR_KEYWORD", NO_DEFAULT),
                ("metrics", "POSITIONAL_OR_KEYWORD", ("a", "b")),
                ("device", "POSITIONAL_OR_KEYWORD", "auto"),
            ],
        ),
        # A `**kwargs` tail (`estimate_gpu_memory`).
        (
            "adata, operation, **kwargs",
            [
                ("adata", "POSITIONAL_OR_KEYWORD", NO_DEFAULT),
                ("operation", "POSITIONAL_OR_KEYWORD", NO_DEFAULT),
                ("kwargs", "VAR_KEYWORD", NO_DEFAULT),
            ],
        ),
        # `x=...` documents that a parameter exists, not what it defaults to.
        (
            "adata_real, adata_pred, pert_col=..., control=...",
            [
                ("adata_real", "POSITIONAL_OR_KEYWORD", NO_DEFAULT),
                ("adata_pred", "POSITIONAL_OR_KEYWORD", NO_DEFAULT),
                ("pert_col", "POSITIONAL_OR_KEYWORD", OPAQUE),
                ("control", "POSITIONAL_OR_KEYWORD", OPAQUE),
            ],
        ),
    ],
)
def test_parse_doc_params_shapes(signature, expected):
    assert parse_doc_params(signature) == expected


def test_a_wrong_default_is_caught():
    """The reason for parsing rather than name-matching.

    `docs/api.md` documented `pca_neighbors(..., prefer_format="auto")` where
    the runtime default is `"csr"`, and `skills/…/processing.md` documented
    `normalize_total(target_sum=10000.0)` against a runtime `None`. A names-only
    comparison passes both.
    """
    runtime = [("adata", "POSITIONAL_OR_KEYWORD", NO_DEFAULT),
               ("prefer_format", "POSITIONAL_OR_KEYWORD", "csr")]
    assert _matches(parse_doc_params('adata, prefer_format="csr"'), runtime)
    assert not _matches(parse_doc_params('adata, prefer_format="auto"'), runtime)


def test_equal_defaults_written_differently_still_match():
    """The brittleness that made defaults look uncheckable, removed by parsing."""
    runtime = [("eps", "POSITIONAL_OR_KEYWORD", 1e-08),
               ("mode", "POSITIONAL_OR_KEYWORD", "csr")]
    assert _matches(parse_doc_params("eps=1e-8, mode=\"csr\""), runtime)


@pytest.mark.parametrize(
    "line, expected",
    [
        # A trailing return annotation must not be swallowed.
        (
            "- `pyscx.accel.pdex_ref(adata, groupby) → pandas.DataFrame` — text.",
            "adata, groupby",
        ),
        # Prose parentheses after the bullet must not be swallowed either.
        (
            "- `pyscx.accel.pca(adata, n_comps=50)` — Streaming PCA (see below).",
            "adata, n_comps=50",
        ),
        # A multi-function bullet: this helper captures one group; `_collect`
        # walks the rest of the leading run.
        (
            '- `col_sums(dataset, prefer_format="csr")` → f64[]`, `col_nnz`.',
            'dataset, prefer_format="csr"',
        ),
    ],
)
def test_balanced_group_stops_at_the_matching_paren(line, expected):
    m = _ANY_SIGNATURE.search(line)
    assert m is not None
    assert _balanced_group(line, m.end() - 1) == expected


def test_elision_does_not_excuse_a_wrong_name():
    """`...` must not turn the guard into a rubber stamp for its neighbours."""
    runtime = [
        ("adata", "POSITIONAL_OR_KEYWORD", NO_DEFAULT),
        ("key", "POSITIONAL_OR_KEYWORD", NO_DEFAULT),
        ("theta", "KEYWORD_ONLY", 2.0),
        ("device", "KEYWORD_ONLY", "auto"),
    ]
    ok = parse_doc_params("adata, key, *, ..., device=\"auto\"")
    assert _matches(ok, runtime)

    # A phantom parameter beside the elision.
    bad_name = parse_doc_params("adata, key, *, ..., layer=None")
    assert not _matches(bad_name, runtime)

    # The right name on the wrong side of the `*`.
    bad_kind = parse_doc_params("adata, key, device=\"auto\", ...")
    assert not _matches(bad_kind, runtime)


def test_pyscx_and_accel_are_the_same_module_the_docs_describe():
    """Premise: the bullets describe `pyscx.accel`, not some other module."""
    assert accel is pyscx.accel


def test_no_documented_accel_function_has_disappeared():
    """A `pyscx.accel.`-qualified bullet must name something that exists.

    Without this a rename leaves the old bullet in `docs/api.md` documenting a
    function nobody can call, and the sweep would simply not collect it — a
    stale signature that never goes red. The bare-name file is exempt: it mixes
    accel calls with `Experiment` methods and `sc.pp.*` references, none of
    which resolve on `pyscx.accel`.
    """
    assert not _UNRESOLVED, (
        "docs name `pyscx.accel` functions that do not exist at runtime "
        "(renamed or removed?):\n  " + "\n  ".join(_UNRESOLVED)
    )
