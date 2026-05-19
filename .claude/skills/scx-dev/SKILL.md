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

- **Synchronized minor (e.g. v0.3.0, v0.4.0):** bump *all 14 workspace members* to the same version — 12 lib crates (`scx-format`, `scx-codec`, `scx-sparse`, `scx-cli`, `scx-convert`, `scx-ops`, `scx-engine`, `scx-loader`, `scx-cloud`, `scx-gpu`, `scx-mtx`, `scx-accel`), plus `pyscx` and `rscx`. The integration-test crate at `tests/scx-integration-tests/` stays at `0.0.0` (`publish = false`).
- **pyscx-only patch (e.g. v0.3.1 → v0.3.2):** bump only `pyscx/Cargo.toml` + `pyscx/pyproject.toml`. pyscx may drift ahead of the rest between minor releases.

Files touched on a synchronized bump:
- `<crate>/Cargo.toml` for each member (`version = "..."` on line 3)
- `pyscx/pyproject.toml` (`version = "..."`)
- `rscx/DESCRIPTION` (`Version: ...`)
- `Cargo.lock` (regenerate with `cargo update --workspace --offline`)

Fast bump: `for d in scx-format scx-codec scx-sparse scx-cli scx-convert scx-ops scx-engine scx-loader scx-cloud scx-gpu scx-mtx scx-accel rscx; do sed -i 's/^version = "OLD"$/version = "NEW"/' "$d/Cargo.toml"; done` then handle pyscx and the pyproject/DESCRIPTION files explicitly (their old version may differ).

### Pre-release verification

Run from the bumped commit before tagging.

**Format and codec:**

- [ ] `cargo test --workspace` passes (CPU-only)
- [ ] `cargo clippy --workspace -- -D warnings` clean
- [ ] `cargo fmt --check` clean
- [ ] `cargo test --workspace --features cloud` passes if cloud changes
- [ ] `cargo check` on default-members compiles. **`cargo check --workspace` may fail in `extendr-api`** (the R-bindings dep needs an R toolchain); rscx is intentionally excluded from default-members for this reason — that failure is *not* a release blocker.

**Fuzzing** (run before any release that touches `scx-codec`, `scx-format`, or `scx-engine` parsers; recommended on every synchronized minor regardless):

- [ ] CI's `Fuzz / fuzz-build` job ran green on the release commit. This job is automatic on every PR and push to `main` and verifies all 12 targets across `scx-codec/fuzz`, `scx-format/fuzz`, and `scx-engine/fuzz` still compile on nightly — catches bitrot from a parser refactor that would otherwise go unnoticed (the fuzz crates are separate cargo workspaces, so the main CI's `cargo --workspace` walks past them).
- [ ] Manually trigger the `Fuzz / fuzz-run` workflow against the release commit via the Actions tab → Fuzz → "Run workflow". Use `duration_seconds=600` (10 min / target, ~2 h matrix wall clock) and leave `target` empty to fan out across all 12 targets. Any crash input is uploaded as a workflow artifact named `fuzz-<crate>-<target>-crashes`; investigate before tagging. Full target list and local invocation in [docs/development.md § Fuzzing](../../../docs/development.md#fuzzing).
- [ ] For an urgent / targeted release, instead run a single target locally: `cd <crate>/fuzz && cargo +nightly fuzz run <target> -- -max_total_time=600`. Requires `cargo install cargo-fuzz` and the nightly toolchain.

**Python bindings** (always via `../.venv/bin/`, never system Python):

- [ ] `cd pyscx && ../.venv/bin/maturin develop && ../.venv/bin/pytest tests/ -v`
- [ ] Same with `--features cloud` if cloud touched
- [ ] Same with `--features gpu` if GPU touched

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

1. **Bump versions** as described above. Verify with `grep -E '^version' */Cargo.toml pyscx/pyproject.toml; grep '^Version' rscx/DESCRIPTION` — all should report the new version (except the integration tests crate, which stays at `0.0.0`).
2. **Refresh the lockfile:** `cargo update --workspace --offline`.
3. **Sanity build:** `cargo check` (default-members only). Expect the rscx/extendr failure on `cargo check --workspace`; that's pre-existing.
4. **Commit on `main`** with a message summarizing the headline changes. Past commits follow `chore: bump workspace to vX.Y.Z` as the subject — see `git log v0.3.0~1..v0.3.0` and the v0.4.0 commit `c21e833` for the body shape.
5. **Tag with both prefixes**, pointing at the commit (use `^{}` to avoid the nested-tag trap if you're tagging an existing tag):
   ```
   git tag -a pyscx-vX.Y.Z   -m "Release pyscx vX.Y.Z"
   git tag -a scx-cli-vX.Y.Z -m "Release scx-cli vX.Y.Z"
   ```
   If only one artifact changed (e.g. pyscx-only patch), tag only that artifact.
6. **Push to the `github` remote** (this repo's remote is named `github`, NOT `origin`):
   ```
   git push github main
   git push github pyscx-vX.Y.Z scx-cli-vX.Y.Z
   ```
   Pushing the tag triggers the corresponding workflow, which creates the GitHub Release entry and uploads wheels/binaries. **Do not run `gh release create`** — the workflow owns that.
7. **Verify** with `gh run list --repo ArcInstitute/scx --limit 5`; both `pyscx release wheels` and `scx-cli release binaries` should appear `in_progress`. The releases appear at https://github.com/ArcInstitute/scx/releases when the workflows finish.

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
- **`pyscx` features:** `hdf5` (h5ad/h5mu ingest), `hdf5-static` (bundles libhdf5 for wheel builds), `cloud`, `gpu`. Build with e.g. `maturin develop --features cloud,gpu`.
- **`scx-cli` features:** `hdf5`, `hdf5-static`, `cloud`. The release workflow uses `hdf5-static` so downloaded binaries have no system libhdf5 requirement.
- **GPU:** `scx-gpu` compiles without CUDA installed; runtime falls back to CPU when no GPU is present. CI's `build-cpu-only` job pins this contract.
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