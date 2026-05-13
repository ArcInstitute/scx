# SCX Release Checklist

Pre-release verification steps for every tagged release.

## Format and codec

- [ ] `cargo test --workspace` passes (CPU-only)
- [ ] `cargo clippy --workspace -- -D warnings` clean
- [ ] `cargo fmt --check` clean
- [ ] `cargo test --workspace --features cloud` passes (if cloud changes)

## Python bindings

- [ ] `cd pyscx && ../.venv/bin/maturin develop && ../.venv/bin/pytest tests/ -v` passes
- [ ] `cd pyscx && ../.venv/bin/maturin develop --features cloud && ../.venv/bin/pytest tests/ -v` passes (if cloud)
- [ ] `cd pyscx && ../.venv/bin/maturin develop --features gpu && ../.venv/bin/pytest tests/ -v` passes (if GPU)

## R bindings

- [ ] `cd rscx && R CMD INSTALL . && Rscript -e 'testthat::test_dir("tests/testthat")'` passes

## Benchmarks

- [ ] Every README-visible benchmark number has a corresponding manifest
      entry in `benchmarks/comprehensive/results/`. Verified by:
      ```
      python benchmarks/scripts/check_readme_manifests.py
      ```
- [ ] Regression gate passes against `LATEST` baseline:
      ```
      python benchmarks/comprehensive/scripts/gate_candidate.py
      ```
- [ ] No TBD / pending / placeholder rows in README.md benchmark tables

## Documentation

- [ ] `ROADMAP.md` date stamp is current
- [ ] `docs/performance.md` numbers match the promoted baseline
- [ ] All internal doc links resolve (no broken cross-references)
- [ ] `AGENTS.md` section counts and summaries are current

## Release

- [ ] Version bumped in `Cargo.toml` (workspace), `pyscx/Cargo.toml`, `rscx/DESCRIPTION`
- [ ] `CHANGELOG.md` updated (if maintained)
- [ ] Tag created: `git tag -a v{VERSION} -m "Release v{VERSION}"`
- [ ] CLI binary built and attached to GitHub release
