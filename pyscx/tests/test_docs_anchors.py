"""Every in-repo markdown anchor link must resolve to a real heading (dogfood D1).

`docs/operations.md#external-obs-import` was referenced from four places —
`docs/api.md` twice, `docs/scanpy.md`, and `AGENTS.md` — and the section did not
exist. Those links are what tell a user where the import's invariants are
documented, so four dead anchors meant the one place someone would look had
nothing in it, and nothing detected that.

This is a pure text check over tracked markdown: no imports, no fixtures, and it
would have caught D1 the moment the first link was written.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

# `[text](target.md#anchor)` and `[text](#anchor)`. `[^\]]*` rather than `.*?`
# so a `]` inside the link text cannot swallow the whole match.
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]*?)#([^)\s]+)\)")
# ATX headings only (`## Title`). Setext headings are not used in these docs.
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tracked_markdown(root: Path) -> list[Path]:
    """Tracked `.md` files, preferring git and falling back to a directory walk.

    Untracked scratch markdown in the repo root (dogfood reports, planning docs)
    is deliberately excluded when git is available: it is never committed, so its
    links are nobody's contract, and a stale one there must not fail the suite.

    **Falls back rather than skipping.** An earlier version called
    `pytest.skip()` when git was unavailable, which would have let this check
    silently no-op in any environment without git — and a guard that can vanish
    without saying so is the same class of problem the check exists to catch. The
    walk excludes the same scratch files by name so both paths agree.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "*.md", "**/*.md"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        files = [root / line for line in out.splitlines() if line]
        if files:
            return files
    except (OSError, subprocess.CalledProcessError):
        pass

    return [p for p in root.rglob("*.md") if _in_walk_allowlist(p, root)]


# The fallback walk's scope: the **hand-written prose trees**, as an allowlist.
#
# This is deliberately *narrower* than `git ls-files`, and it is not trying to
# match it. Approximating `.gitignore` by hand is a losing game — gitignored
# markdown turns up in `tasks/` (~134 planning docs), `.claude/plans/`, `learn/`,
# `.probe_venv/`, `.pytest_cache/`, vendored `dist-info/LICENSE.md`, and
# `benchmarks/**/results/` (which also holds 18 *tracked* files, so it cannot even
# be skipped wholesale).
#
# So the fallback is an explicit reduced-coverage safety net: if git disappears the
# check still runs over the docs whose anchors are contracts — every referrer in
# the D1 finding lives in these trees — rather than silently passing.
# `test_the_two_enumeration_paths_agree` pins both halves of that claim: the walk
# never includes untracked files, and it does cover the core prose trees.
_WALK_INCLUDE_PREFIXES: tuple[str, ...] = (
    "docs/",
    "skills/",
    ".claude/skills/",
)

# Repo-root markdown that IS tracked. Everything else at the root is scratch by
# convention — dated reports (`2026-08-03_DOGFOOD.md`), all-caps working docs, and
# `CLAUDE.local.md` (gitignored, and neither dated nor all-caps).
_TRACKED_ROOT_DOCS = frozenset({"README.md", "ROADMAP.md", "AGENTS.md", "CLAUDE.md"})

# Never walk into these, whatever the prefix rule says.
_WALK_SKIP_PARTS = frozenset(
    {
        ".git",
        ".venv",
        ".probe_venv",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        "target",
    }
)


def _in_walk_allowlist(path: Path, root: Path) -> bool:
    """Whether the fallback walk should enumerate `path`."""
    if not _WALK_SKIP_PARTS.isdisjoint(path.parts):
        return False
    rel = path.relative_to(root).as_posix()
    if "/" not in rel:
        return rel in _TRACKED_ROOT_DOCS
    return rel.startswith(_WALK_INCLUDE_PREFIXES)


def _slug(heading: str) -> str:
    """GitHub's anchor slug for a heading.

    Strip markdown inline syntax, lowercase, delete everything that is not a word
    character / space / hyphen, then replace each remaining space with a hyphen.

    **Each space, not each run.** GitHub deletes a punctuation character in place
    and turns both surrounding spaces into hyphens, so `Harmony2 batch
    integration + LISI` slugs to `harmony2-batch-integration--lisi` with a *double*
    hyphen. Collapsing whitespace runs — the obvious first implementation — makes
    every such heading look like a dead link, which is how this rule was found:
    the first version of this test reported ~40 failures that were all its own.
    """
    text = heading.strip()
    text = re.sub(r"`([^`]*)`", r"\1", text)  # inline code
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links
    text = text.replace("*", "")
    text = text.lower()
    # `_` is a word character and survives (`num_workers` → `num_workers`).
    text = re.sub(r"[^\w\s-]", "", text)
    return text.strip().replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    text = path.read_text(encoding="utf-8", errors="replace")
    out: set[str] = set()
    for _level, heading in _HEADING.findall(text):
        slug = _slug(heading)
        if slug:
            out.add(slug)
    # Explicit HTML anchors, if any doc uses them.
    out.update(re.findall(r'<a\s+(?:id|name)="([^"]+)"', text))
    return out


def test_markdown_anchor_links_resolve():
    root = _repo_root()
    files = _tracked_markdown(root)
    assert files, "no tracked markdown found — the check would be vacuous"

    anchor_cache: dict[Path, set[str]] = {}
    dead: list[str] = []
    checked = 0

    for src in files:
        if not src.is_file():
            continue
        for target, anchor in _LINK.findall(
            src.read_text(encoding="utf-8", errors="replace")
        ):
            # Only in-repo markdown targets. Skip external URLs and links into
            # non-markdown files, whose anchors are not ours to validate.
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            dest = src.parent / target if target else src
            if dest.suffix.lower() != ".md":
                continue
            dest = dest.resolve()
            if not dest.is_file():
                # A dead *file* link is a separate defect; report it rather than
                # silently passing the anchor.
                dead.append(
                    f"{src.relative_to(root)}: target file missing for "
                    f"[...]({target}#{anchor})"
                )
                continue
            if dest not in anchor_cache:
                anchor_cache[dest] = _anchors(dest)
            checked += 1
            if anchor.lower() not in anchor_cache[dest]:
                dead.append(
                    f"{src.relative_to(root)} -> {dest.relative_to(root)}#{anchor}"
                )

    assert checked > 50, (
        f"only {checked} anchor links checked — the sweep is not finding the "
        "docs it is supposed to cover"
    )
    assert not dead, "dead markdown anchor links:\n  " + "\n  ".join(sorted(dead))


def test_the_d1_anchor_specifically_resolves():
    """Pin the four links D1 filed, by name.

    The sweep above would catch a regression, but naming this one keeps the
    reason legible: `docs/operations.md#external-obs-import` is where every obs-
    import surface sends the reader for the in-place / join / null / index-
    staleness / rollback invariants.
    """
    root = _repo_root()
    anchors = _anchors(root / "docs" / "operations.md")
    assert "external-obs-import" in anchors, (
        "docs/operations.md must carry an `## External obs import` section — "
        f"docs/api.md, docs/scanpy.md and AGENTS.md all link to it. Found: "
        f"{sorted(a for a in anchors if 'import' in a)}"
    )

    referrers = [
        Path("docs/api.md"),
        Path("docs/scanpy.md"),
        Path("AGENTS.md"),
    ]
    for rel in referrers:
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        assert "operations.md#external-obs-import" in text, (
            f"{rel} no longer links to the section; if the link was removed on "
            "purpose, drop it from this list"
        )


def test_the_two_enumeration_paths_agree():
    """The no-git fallback must enumerate what `git ls-files` does.

    Review finding: the first fallback was a bare `rglob("*.md")` with only a few
    build dirs skipped, so it picked up gitignored `tasks/` (~134 planning docs)
    and `CLAUDE.local.md` — 309 files against git's 58. A fallback that scans five
    times the surface can fail on scratch content CI never sees, which turns a
    guard into a liability.

    Skipped rather than approximated when git is missing: there is nothing to
    compare against, and the fallback is then the only answer available.
    """
    root = _repo_root()
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "*.md", "**/*.md"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git unavailable — nothing to compare the fallback against")

    tracked = {(root / line).resolve() for line in out.splitlines() if line}
    walked = {p.resolve() for p in root.rglob("*.md") if _in_walk_allowlist(p, root)}

    # The dangerous direction: the walk must never hand the anchor sweep a file
    # git does not track. Links in scratch docs are nobody's contract, and failing
    # the suite on one would make the guard a liability.
    extra = sorted(str(p.relative_to(root)) for p in walked - tracked)
    assert not extra, (
        "the fallback walk includes files git does not track; narrow "
        f"_WALK_INCLUDE_PREFIXES or extend _WALK_SKIP_PARTS:\n  " + "\n  ".join(extra)
    )

    # The other direction is allowed to be smaller — the fallback is a
    # reduced-coverage net, not a git replacement — but it must still cover the
    # prose trees where anchor contracts live, including every D1 referrer.
    core = {
        p
        for p in tracked
        if p.parent == root
        or p.relative_to(root).as_posix().startswith(("docs/", "skills/"))
    }
    uncovered = sorted(str(p.relative_to(root)) for p in core - walked)
    assert not uncovered, (
        f"the fallback misses {len(uncovered)} core prose file(s), so it would not "
        f"catch a D1-class regression without git:\n  " + "\n  ".join(uncovered[:10])
    )
    assert len(walked) > 25, (
        f"the fallback only found {len(walked)} files — too few to be a meaningful "
        "safety net"
    )
