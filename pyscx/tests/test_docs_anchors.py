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

    skip_dirs = {".git", "target", "node_modules", ".venv", "__pycache__"}
    walked: list[Path] = []
    for path in root.rglob("*.md"):
        if skip_dirs.isdisjoint(path.parts) and not _is_scratch_doc(path, root):
            walked.append(path)
    return walked


def _is_scratch_doc(path: Path, root: Path) -> bool:
    """Untracked scratch markdown, matched by the repo's own naming convention.

    Repo-root docs that are dated (`2026-08-03_DOGFOOD.md`) or all-caps
    (`*-CODE-REVIEW.md`) are ephemeral and gitignored/untracked by convention;
    `README` / `ROADMAP` / `AGENTS` / `CLAUDE` are the tracked exceptions.
    """
    if path.parent != root:
        return False
    stem = path.stem
    if stem in {"README", "ROADMAP", "AGENTS", "CLAUDE"}:
        return False
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}[_-]", stem)) or stem.isupper()


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
