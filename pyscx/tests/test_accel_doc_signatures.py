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

import inspect
import pathlib
import re

import pytest

import pyscx
import pyscx.accel as accel

# The files carrying hand-maintained accel signature bullets. `docs/api.md`
# prefixes them `pyscx.accel.`; the skills reference uses the bare name.
_DOC_FILES = (
    "docs/api.md",
    "skills/scx-usage/reference/processing.md",
)

# A bullet whose function is one of ours. The name is matched loosely and
# filtered against the runtime module afterwards, so a `sc.pp.log1p(...)`
# reference or an `Experiment` method bullet is simply not collected.
_BULLET = re.compile(r"^\s*[-*] `(?:pyscx\.)?(?:accel\.)?([A-Za-z_]\w*)\(")

# Enough bullets that an over-tightened regex fails loudly instead of passing
# vacuously. 41 were collected when this was written; the floor is deliberately
# below that so adding or removing one bullet is not a test failure.
_MIN_BULLETS = 35

# The elision marker docs use when a bullet deliberately abbreviates a long
# keyword-only tail (`harmony_integrate`) or declares nothing at all
# (`energy_distance_details(...)`).
_ELLIPSIS = "..."


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


def _split_top_level(text: str) -> list[str]:
    """Split on commas at nesting depth zero, outside quotes."""
    out: list[str] = []
    depth = 0
    quote: str | None = None
    cur = ""
    for ch in text:
        if quote is not None:
            cur += ch
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            cur += ch
        elif ch in "([{":
            depth += 1
            cur += ch
        elif ch in ")]}":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def parse_doc_params(signature: str) -> list[tuple[str, str]]:
    """`(name, kind)` per documented parameter, in order.

    `#` comments are stripped first: the `docs/scanpy.md` fenced blocks annotate
    every line, and `# None = load all layers` would otherwise contribute a
    parameter called `None`.

    Unlike `test_experiment_stub_coverage.py::_doc_params`, a parameter with no
    default is captured. That helper reads `name=` tokens only, which is
    harmless for `to_anndata` (every parameter has a default) but drops the
    leading `adata` / `gene_list` / `counts` / `labels_a` of every accel
    function.

    An `...` token is returned verbatim as an elision marker; see
    `_matches_with_elisions`.
    """
    stripped = re.sub(r"#.*", "", signature)
    params: list[tuple[str, str]] = []
    kind = "POSITIONAL_OR_KEYWORD"
    for raw in _split_top_level(stripped):
        tok = raw.strip()
        if not tok:
            continue
        if tok == "*":
            kind = "KEYWORD_ONLY"
            continue
        if tok == "/":
            params = [(n, "POSITIONAL_ONLY") for n, _ in params]
            continue
        if tok.startswith(_ELLIPSIS):
            params.append((_ELLIPSIS, _ELLIPSIS))
            continue
        if tok.startswith("**"):
            params.append((tok[2:].split("=")[0].split(":")[0].strip(), "VAR_KEYWORD"))
            continue
        if tok.startswith("*"):
            params.append(
                (tok[1:].split("=")[0].split(":")[0].strip(), "VAR_POSITIONAL")
            )
            kind = "KEYWORD_ONLY"
            continue
        name = tok.split("=", 1)[0].split(":", 1)[0].strip()
        if not name.isidentifier():
            # Not a parameter — e.g. a stray prose fragment inside the parens.
            # Recorded as-is so the comparison fails visibly rather than
            # silently dropping it.
            params.append((tok, "UNPARSED"))
            continue
        params.append((name, kind))
    return params


def _runtime_params(fn) -> list[tuple[str, str]]:
    return [(p.name, p.kind.name) for p in inspect.signature(fn).parameters.values()]


def _matches_with_elisions(
    doc: list[tuple[str, str]], runtime: list[tuple[str, str]]
) -> bool:
    """Whether `doc` describes `runtime`, treating `...` as "zero or more".

    Exact equality when the bullet carries no elision. With one, the named
    parameters on either side of it must still appear in order and with the
    right kind — so `harmony_integrate(adata, key, *, basis=…, ..., random_state=0,
    device="auto")` passes while a phantom or misspelled name beside the `...`
    does not. A bullet that is only `(...)` declares nothing and matches
    anything, which is honest: it documents nothing to drift.
    """
    if not any(name == _ELLIPSIS for name, _ in doc):
        return doc == runtime
    # Classic wildcard match, iterative over doc positions.
    reachable = {0}
    for name, kind in doc:
        if name == _ELLIPSIS:
            nxt = set()
            for r in reachable:
                nxt.update(range(r, len(runtime) + 1))
            reachable = nxt
            continue
        nxt = set()
        for r in reachable:
            if r < len(runtime) and runtime[r] == (name, kind):
                nxt.add(r + 1)
        if not nxt:
            return False
        reachable = nxt
    return len(runtime) in reachable


def _collect() -> list[tuple[str, int, str, str]]:
    """Every accel signature bullet: `(doc_path, lineno, fn_name, signature)`."""
    root = _repo_root()
    found: list[tuple[str, int, str, str]] = []
    for rel in _DOC_FILES:
        text = (root / rel).read_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            m = _BULLET.match(line)
            if not m:
                continue
            name = m.group(1)
            fn = getattr(accel, name, None)
            if fn is None or not callable(fn):
                continue
            body = _balanced_group(line, m.end() - 1)
            if body is None:
                # An unterminated bullet is drift too — record it so the test
                # names the line instead of skipping it.
                body = "<unbalanced parentheses>"
            found.append((rel, lineno, name, body))
    return found


_BULLETS = _collect()


def test_the_sweep_found_bullets():
    """A regex that matches nothing would make every other test here vacuous."""
    assert len(_BULLETS) >= _MIN_BULLETS, (
        f"only {len(_BULLETS)} accel signature bullets collected from "
        f"{list(_DOC_FILES)} (expected at least {_MIN_BULLETS}). Either the "
        "bullet shape changed or `_BULLET` stopped matching it — a silent "
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

    assert _matches_with_elisions(documented, runtime), (
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
            "adata, n_comps=50, device=\"auto\"",
            [
                ("adata", "POSITIONAL_OR_KEYWORD"),
                ("n_comps", "POSITIONAL_OR_KEYWORD"),
                ("device", "POSITIONAL_OR_KEYWORD"),
            ],
        ),
        # A `*` separator flips the tail keyword-only.
        (
            "adata, qc_vars=None, *, layer=None, percent_top=None",
            [
                ("adata", "POSITIONAL_OR_KEYWORD"),
                ("qc_vars", "POSITIONAL_OR_KEYWORD"),
                ("layer", "KEYWORD_ONLY"),
                ("percent_top", "KEYWORD_ONLY"),
            ],
        ),
        # An elided keyword-only tail (`harmony_integrate`).
        (
            "adata, key, *, basis=\"X_pca\", ..., device=\"auto\"",
            [
                ("adata", "POSITIONAL_OR_KEYWORD"),
                ("key", "POSITIONAL_OR_KEYWORD"),
                ("basis", "KEYWORD_ONLY"),
                ("...", "..."),
                ("device", "KEYWORD_ONLY"),
            ],
        ),
        # A wholly elided bullet (`energy_distance_details(...)`).
        ("...", [("...", "...")]),
        # A default containing a comma inside brackets must not split.
        (
            "adata, metrics=(\"a\", \"b\"), device=\"auto\"",
            [
                ("adata", "POSITIONAL_OR_KEYWORD"),
                ("metrics", "POSITIONAL_OR_KEYWORD"),
                ("device", "POSITIONAL_OR_KEYWORD"),
            ],
        ),
        # A `**kwargs` tail (`estimate_gpu_memory`).
        (
            "adata, operation, **kwargs",
            [
                ("adata", "POSITIONAL_OR_KEYWORD"),
                ("operation", "POSITIONAL_OR_KEYWORD"),
                ("kwargs", "VAR_KEYWORD"),
            ],
        ),
    ],
)
def test_parse_doc_params_shapes(signature, expected):
    assert parse_doc_params(signature) == expected


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
        # A multi-function bullet: only the first signature is captured.
        (
            "- `col_sums(dataset, prefer_format=\"csr\") → f64[]`, `col_nnz`.",
            "dataset, prefer_format=\"csr\"",
        ),
    ],
)
def test_balanced_group_stops_at_the_matching_paren(line, expected):
    m = _BULLET.match(line)
    assert m is not None
    assert _balanced_group(line, m.end() - 1) == expected


def test_elision_does_not_excuse_a_wrong_name():
    """`...` must not turn the guard into a rubber stamp for its neighbours."""
    runtime = [
        ("adata", "POSITIONAL_OR_KEYWORD"),
        ("key", "POSITIONAL_OR_KEYWORD"),
        ("theta", "KEYWORD_ONLY"),
        ("device", "KEYWORD_ONLY"),
    ]
    ok = parse_doc_params("adata, key, *, ..., device=\"auto\"")
    assert _matches_with_elisions(ok, runtime)

    # A phantom parameter beside the elision.
    bad_name = parse_doc_params("adata, key, *, ..., layer=None")
    assert not _matches_with_elisions(bad_name, runtime)

    # The right name on the wrong side of the `*`.
    bad_kind = parse_doc_params("adata, key, device=\"auto\", ...")
    assert not _matches_with_elisions(bad_kind, runtime)


def test_pyscx_and_accel_are_the_same_module_the_docs_describe():
    """Premise: the bullets describe `pyscx.accel`, not some other module."""
    assert accel is pyscx.accel
