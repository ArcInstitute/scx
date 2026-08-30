---
name: scx-dev
description: SCX development workflows — cutting releases (pyscx + scx-cli), running the full test matrix, and the build/dev-env layout. Trigger on "cut a release", "bump version", "release v0.X.Y", "pre-release checks", or general SCX dev-environment questions whose answer isn't already in CLAUDE.md.
---

# SCX Development

`CLAUDE.md` (auto-loaded) already covers the architecture, capabilities, conventions, and per-feature pointers into `docs/`. This skill captures the *workflow* knowledge that isn't derivable from the code — primarily releases, plus pointers for dev-env questions.

## Releasing pyscx and scx-cli

The two artifacts are released **independently** with **prefixed tags**. There is no unified `v*` tag scheme — that was tried for v0.4.0 and produced no GitHub release and no workflow run.

### Tag scheme (critical — easy to get wrong)

| Artifact | Tag glob the workflow listens on | Workflow file |
|---|---|---|
| pyscx wheels | `pyscx-v*` | `.github/workflows/pyscx-release.yml` |
| scx-cli binaries | `scx-cli-v*` | `.github/workflows/scx-cli-release.yml` |

A unified `vX.Y.Z` tag triggers **nothing**. Always tag with both prefixes when shipping a synchronized release.

`gh release list` shows the canonical pattern: `pyscx-v0.3.2`, `scx-cli-v0.3.0`, etc.

### Version-bump scope

Two conventions coexist; pick based on what changed:

- **Synchronized minor (e.g. v0.3.0, v0.4.0):** bump *all 15 workspace members* to the same version — 13 lib crates (`scx-format`, `scx-format-io`, `scx-codec`, `scx-sparse`, `scx-cli`, `scx-convert`, `scx-ops`, `scx-engine`, `scx-loader`, `scx-cloud`, `scx-gpu`, `scx-mtx`, `scx-accel`), plus `pyscx` and `rscx`. The integration-test crate at `tests/scx-integration-tests/` stays at `0.0.0` (`publish = false`).
- **pyscx-only patch (e.g. v0.3.1 → v0.3.2):** bump only `pyscx/Cargo.toml` + `pyscx/pyproject.toml`. pyscx may drift ahead of the rest between minor releases.

Files touched on a synchronized bump:
- `<crate>/Cargo.toml` for each member (`version = "..."` on line 3)
- `pyscx/pyproject.toml` (`version = "..."`)
- `rscx/DESCRIPTION` (`Version: ...`)
- `Cargo.lock` (regenerate with `cargo update --workspace --offline`)

Fast bump: `for d in scx-format scx-format-io scx-codec scx-sparse scx-cli scx-convert scx-ops scx-engine scx-loader scx-cloud scx-gpu scx-mtx scx-accel rscx; do sed -i 's/^version = "OLD"$/version = "NEW"/' "$d/Cargo.toml"; done` then handle pyscx and the pyproject/DESCRIPTION files explicitly (their old version may differ).

### Pre-release verification

Run on the edited working tree, *before* committing (Phase 2 of the pipeline below) — not after tagging.

**Format and codec:**

- [ ] `cargo test --workspace --exclude rscx` passes (CPU-only). **Two known non-blocking environmental failures:**
  - `scx-cloud::backend::tests::create_gcs_backend` (and potentially `create_s3_backend`) **fail in a sandbox with no network** — the `object_store` GCS builder probes the GCE metadata server at `build()` time. They pass in CI, which has network. To confirm a failure is this and not a real regression, re-run the single test on plain `main` (`git stash` or a clean checkout): an *identical* failure on `main` means environmental, not introduced by your change.
  - See also the extendr note below.
- [ ] `cargo clippy --workspace --exclude rscx --all-targets -- -D warnings` clean. `--exclude rscx` matches CI (see the extendr note below); `--all-targets` lints test and bench code, which CI also does.
- [ ] `cargo fmt --check` clean
- [ ] `cargo test --workspace --exclude rscx --features cloud` passes if cloud changes
- [ ] `cargo check` on default-members compiles. **`cargo check --workspace` may fail in `extendr-api`** (the R-bindings dep needs an R toolchain); rscx is intentionally excluded from default-members for this reason — that failure is *not* a release blocker.

**Fuzzing** (run before any release that touches `scx-codec`, `scx-format`, or `scx-engine` parsers; recommended on every synchronized minor regardless):

- [ ] CI's `Fuzz / fuzz-build` job ran green on the release commit. This job is automatic on PRs and pushes to `main` that touch the fuzzed crates (path-filtered to `scx-codec`/`scx-format`/`scx-format-io`/`scx-engine`/`scx-sparse` + the lockfile — an untouched-path commit legitimately has no Fuzz run) and verifies the fuzz targets across `scx-codec/fuzz`, `scx-format/fuzz`, and `scx-engine/fuzz` still compile on nightly — catches bitrot from a parser refactor that would otherwise go unnoticed (the fuzz crates are separate cargo workspaces, so the main CI's `cargo --workspace` walks past them).
- [ ] Manually trigger the `Fuzz / fuzz-run` workflow against the release commit via the Actions tab → Fuzz → "Run workflow". Use `duration_seconds=600` (10 min / target, ~2 h matrix wall clock) and leave `target` empty to fan out across all 12 targets. Any crash input is uploaded as a workflow artifact named `fuzz-<crate>-<target>-crashes`; investigate before tagging. Full target list and local invocation in [docs/development.md § Fuzzing](../../../docs/development.md#fuzzing).
- [ ] For an urgent / targeted release, instead run a single target locally: `cd <crate>/fuzz && cargo +nightly fuzz run <target> -- -max_total_time=600`. Requires `cargo install cargo-fuzz` and the nightly toolchain.

**Python bindings** (always via `../.venv/bin/`, never system Python):

- [ ] `cd pyscx && ../.venv/bin/maturin develop && ../.venv/bin/pytest tests/ -v`
- [ ] Same with `--features hdf5,cloud` if cloud touched
- [ ] Same with `--features hdf5,gpu` if GPU touched
- [ ] (`--features` REPLACES the pyproject default set, which includes `hdf5` — always re-list `hdf5`, else `from_h5ad`/`to_h5ad` break)

**R bindings** (when rscx changed):

- [ ] `cd rscx && R CMD INSTALL . && Rscript -e 'testthat::test_dir("tests/testthat")'`

**Benchmarks** (when format / codec / hot paths changed):

- [ ] `python benchmarks/scripts/check_readme_manifests.py` — every README number has a manifest entry
- [ ] `python benchmarks/comprehensive/scripts/gate_candidate.py` — regression gate against `LATEST` baseline (see [benchmarks/README.md § Regression Gating](../../../benchmarks/README.md#regression-gating))
- [ ] No `TBD` / `pending` / placeholder rows in `README.md` benchmark tables

**Documentation:**

- [ ] `ROADMAP.md` date stamp is current
- [ ] `docs/performance.md` numbers match the promoted baseline
- [ ] Cross-doc links resolve

### Release steps

"Bumping the version" is the **whole pipeline below**, not just editing files. A bump is not done until the pre-release checks pass, the PR is merged to `main`, *and* the release workflows have been triggered. The six phases: **edit files → run pre-release checks → commit on a branch → open & merge a PR → tag the merged commit → push tags (this creates the GitHub Release)**.

The repo's remote is named `github`, NOT `origin`. Never commit the version bump directly to `main` — `main` is protected and the bump must land via a reviewed/merged PR.

**Phase 1 — Edit files**

1. **Bump versions** as described in [Version-bump scope](#version-bump-scope). Verify with `grep -E '^version' */Cargo.toml pyscx/pyproject.toml; grep '^Version' rscx/DESCRIPTION` — all should report the new version (except the integration tests crate, which stays at `0.0.0`).
2. **Refresh the lockfile:** `cargo update --workspace --offline`.
3. **Bump the `ROADMAP.md` "Last updated" date stamp** to today.

**Phase 2 — Run pre-release checks (mandatory, not optional)**

4. **Sanity build:** `cargo check` (default-members only). Expect the rscx/extendr failure on `cargo check --workspace`; that's pre-existing.
5. **Run the full [Pre-release verification](#pre-release-verification) checklist** appropriate to what changed — this is part of the bump, not a separate later step. For a version-only bump the format/codec/Python checks should be unaffected, but at minimum **all three of** `cargo fmt --check`, `cargo clippy --workspace --exclude rscx --all-targets -- -D warnings`, and `cargo test --workspace --exclude rscx` MUST be clean before you commit and open the PR. Run them locally first; let CI be the second signal, not the first. Do not proceed to Phase 3 with a failing check.

**Phase 3 — Commit on a branch**

6. **Create a release branch** (e.g. `bump-version-X.Y.Z`) off `main` and **commit** the bump there. Past commits follow `chore: bump workspace version to X.Y.Z` as the subject — see `git log v0.3.0~1..v0.3.0` and the v0.4.0 commit `c21e833` for the body shape. Stage only tracked files (`git add -u`) so untracked scratch docs (e.g. ALL-CAPS root markdown) are never swept in.

**Phase 4 — Open and merge the PR**

7. **Push the branch and open a PR:**
   ```
   git push -u github bump-version-X.Y.Z
   gh pr create --repo ArcInstitute/scx --base main --title "chore: bump workspace version to X.Y.Z" --body "..."
   ```
8. **Wait for CI green**, then **merge the PR** to `main`:
   ```
   gh pr checks <PR#> --repo ArcInstitute/scx --watch --interval 30
   gh pr merge <PR#> --repo ArcInstitute/scx --squash --delete-branch
   ```
   Match the merge method the repo uses for prior bump PRs (squash unless the project convention differs).

   **Branch cleanup — do not trust `--delete-branch` to do everything.** Observed behavior here: `--delete-branch` deleted the *remote* branch but did **not** switch me off the feature branch, did **not** delete the *local* branch, and left a stale `github/<branch>` remote-tracking ref. Do the cleanup explicitly after merging:
   ```
   git checkout main && git pull github main
   git branch -D bump-version-X.Y.Z      # -D, not -d: a squash merge isn't an ancestor of main, so -d refuses/warns
   git fetch github --prune              # drops the stale remote-tracking ref
   ```
   Confirm with `git branch` (no local `bump-version-X.Y.Z`) and `git branch -r` (no `github/bump-version-X.Y.Z`).

**Phase 5 — Tag the merged commit**

9. **Tag the merge commit with both prefixes.** You're already on a synced `main` from the step 8 cleanup; sanity-check `git log --oneline -1` shows the squash commit (`... (#PR)`) and `grep -m1 '^version' pyscx/Cargo.toml scx-cli/Cargo.toml` shows the new version. The two artifacts are released independently with prefixed tags — a unified `vX.Y.Z` tag triggers **nothing** (see [Tag scheme](#tag-scheme-critical--easy-to-get-wrong)):
   ```
   git tag -a pyscx-vX.Y.Z   -m "Release pyscx vX.Y.Z"
   git tag -a scx-cli-vX.Y.Z -m "Release scx-cli vX.Y.Z"
   ```
   If only one artifact changed (e.g. a pyscx-only patch), tag only that artifact. Use `^{}` to deref if you ever tag an existing tag.

**Phase 6 — Push tags and verify the release**

10. **Push the tags to `github`:**
    ```
    git push github pyscx-vX.Y.Z scx-cli-vX.Y.Z
    ```
    Pushing each tag triggers its workflow, which creates the GitHub Release entry and uploads wheels/binaries. **Do not run `gh release create`** — the workflow owns that.
11. **Verify the workflows fired**, then **watch them to completion** — the bump isn't done until the artifacts publish. `gh run list --repo ArcInstitute/scx --limit 5` should show both `pyscx release wheels` and `scx-cli release binaries` `in_progress` within ~15 s of the tag push. The wheel/binary builds take **~10–30 min** (multi-platform matrix); follow each with `gh run watch <run-id> --repo ArcInstitute/scx`. Confirm both Releases exist with assets attached at https://github.com/ArcInstitute/scx/releases (or `gh release view pyscx-vX.Y.Z --repo ArcInstitute/scx`) before calling the release done.

### Recovering from a wrong tag

If a `vX.Y.Z` (unprefixed) tag was pushed by mistake, neither workflow fires. Add the correctly-prefixed tags on the same commit and delete the orphan:

```
git tag -a pyscx-vX.Y.Z   <commit-or-vX.Y.Z^{}> -m "Release pyscx vX.Y.Z"
git tag -a scx-cli-vX.Y.Z <commit-or-vX.Y.Z^{}> -m "Release scx-cli vX.Y.Z"
git push github pyscx-vX.Y.Z scx-cli-vX.Y.Z
git push github :refs/tags/vX.Y.Z   # delete on remote
git tag -d vX.Y.Z                   # delete locally
```

The `^{}` dereferences the tag to the underlying commit; tagging a tag produces a nested tag that points at a tag object instead of a commit, which is rarely what you want.

## Build / dev-env quick reference

Full details live in `docs/development.md`. The bits that come up often:

- **Always use `.venv/`** (uv-managed) for Python work — never system Python or `pip`.
- **`pyscx` features:** `hdf5` (h5ad/h5mu ingest), `hdf5-static` (bundles libhdf5 for wheel builds), `cloud`, `gpu`. Build with e.g. `maturin develop --features hdf5,cloud,gpu`.
- **`scx-cli` features:** `hdf5`, `hdf5-static`, `cloud`. The release workflow uses `hdf5-static` so downloaded binaries have no system libhdf5 requirement.
- **GPU:** `scx-gpu` compiles without CUDA installed; runtime falls back to CPU when no GPU is present. CI's `clippy` and `test` jobs (full-workspace builds on CUDA-less runners) pin this contract.
- **GDS** (GPUDirect Storage) needs local NVMe + nvidia-fs + ext4/XFS; always has a CPU fallback.

For the full crate dependency graph and feature flag rules, see [docs/architecture.md § Crate Dependency Graph](../../../docs/architecture.md#crate-dependency-graph). Test matrix is in [docs/testing.md](../../../docs/testing.md).

## Benchmarks workflow

Submit **parallel SLURM jobs** — one per benchmark × dataset pair. Practical guide and the regression-gate workflow are in [benchmarks/README.md](../../../benchmarks/README.md). Local `gate_candidate.py` against `results/baselines/LATEST` is the canonical signal; there is no CI-side gate.

## Coding conventions

Coding conventions live in [docs/conventions.md](../../../docs/conventions.md).

### Documentation in tracked files

Tracked files (code, tests, docs, configs, READMEs) MUST NOT reference
gitignored markdown documents. In this repo those live at the workspace
root and under `tasks/` — typically ALL-CAPS or date-prefixed names like
`*_CODE-REVIEW.md`, `Phase*.md`, `SPEC*.md`, `GPU-ACC-SPEED-UP.md`,
`HARMONY2.md`, `DEADLOCK-ISSUE.md`, `PER-CELL-CONTROL-PAIRING.md`,
`SCX-EVAL-METRIC-IMPROVE.md`, `2026-*_REGRESSIONS.md`, etc. They are
scratch/working specs and do not ship with the repository.

Concretely, in any tracked file, do not:

- Link to a gitignored doc (`[X.md](X.md)`, `see X.md §3`, `(per X.md)`).
- Cite a "review §1.4" / "Phase 7.2 of X.md" / "spec target (X.md:1037)".
- Carry inline `// TODO: see X.md` markers pointing at gitignored specs.

Tracked documentation must stand alone. When the gitignored doc carried
load-bearing context, **inline the substance** (a sentence or two of the
why / how) into the tracked file instead of citing. When the citation
was decorative, just delete it.

Cross-references between tracked files (`docs/*.md`, `benchmarks/README.md`,
`ROADMAP.md`, `AGENTS.md`/`CLAUDE.md`, generated reports under
`benchmarks/comprehensive/results/reports/`) are fine — they ship together.