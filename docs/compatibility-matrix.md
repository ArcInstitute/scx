# Python Compatibility Matrix

This page tracks the Python / scverse / numeric-stack combinations that
`pyscx` is **tested against** versus **bounded to support** via
`pyproject.toml`. It exists so downstream users can pin a known-good set
without inferring it from CI workflow files.

For native-side build flags (CPU / cloud / GPU / HDF5) see
[development.md](development.md). For GPU runtime pre-reqs see
[gpu-setup.md](gpu-setup.md).

## Supported version bounds (declared in `pyproject.toml`)

These bounds are what installing the pre-built pyscx wheel will resolve against. The
upper caps reflect the last release that has been smoke-tested locally;
the lower caps reflect the oldest version we are willing to support.

| Package | Lower bound | Upper bound (exclusive) | Notes |
|---|---|---|---|
| Python | 3.11 | — | `requires-python = ">=3.11"` |
| `numpy` | 1.24 | 3 | Covers the NumPy 1.x → 2.x transition |
| `scipy` | 1.10 | 2 | CSR / CSC zero-copy contract |
| `pyarrow` | 14 | 24 | Used for `obs` / `var` Arrow tables |
| `anndata` | 0.11 | 0.13 | Floor is `0.11` because SCX → h5ad export emits the `nullable-string-array` encoding for null-bearing string columns, whose reader was added in anndata `0.11` |

Optional extras follow the same convention:

| Extra | Package | Bound | Required for |
|---|---|---|---|
| `mudata` | `mudata` | `>=0.2,<1.0` | `pyscx.from_mudata` / `to_mudata` |
| `10x` | `scanpy` | `>=1.10` | `pyscx.from_10x` |
| `dev` | `scanpy` | `>=1.10` | Local test suite |
| `dev` | `pytest` | `>=7.0` | Local test suite |
| `gpu` | `cupy-cuda12x` | `>=12.0` (Linux) | `pyscx.accel.*` GPU device routing |
| `cloud` | `boto3` | `>=1.28` | S3 backend |
| `cloud` | `google-cloud-storage` | `>=2.0` | GCS backend |
| `cloud` | `azure-storage-blob` | `>=12.0` | Azure Blob backend |
| `scvi` | `scvi-tools` | unpinned | `pyscx.scvi_*` helpers |
| `eval` | `polars` | `>=1.0` | `pyscx.eval` (cell-eval parity) |

## Tested combinations

The columns below split tested from declared: a row marked **Tested** has
been exercised by CI or the maintainer's local environment on the version
listed. Versions inside the declared bounds but not in a Tested row are
**Supported but unverified** — they may work, but a regression there is
not a blocker.

### CI matrix (GitHub Actions, `.github/workflows/`)

| Python | numpy | scipy | pyarrow | anndata | scanpy | OS | Status |
|---|---|---|---|---|---|---|---|
| 3.11 | pip-resolved (≥1.24) | pip-resolved (≥1.10) | 23.0.1 (pinned) | pip-resolved (≥0.11,<0.13) | pip-resolved (≥1.10) | ubuntu-latest | ✅ `ci.yml::python` job |
| 3.11 | n/a (wheel build) | n/a | n/a | n/a | n/a | manylinux x86_64 | ✅ `pyscx-release.yml` |
| 3.12 | n/a (wheel build) | n/a | n/a | n/a | n/a | manylinux x86_64 | ✅ `pyscx-release.yml` |

`pyarrow==23.0.1` is pinned in `ci.yml` to exercise the widened upper
bound rather than whatever pip happens to resolve — see the comment at
[`.github/workflows/ci.yml:107`](../.github/workflows/ci.yml).

### Maintainer's local environment (Chimera HPC, dev `.venv`)

Run on the `release/pyscx-v0.3.2` branch as of 2026-05-14:

| Python | numpy | scipy | pyarrow | anndata | scanpy | OS | Status |
|---|---|---|---|---|---|---|---|
| 3.13.3 | 2.4.4 | 1.17.1 | 23.0.1 | 0.12.10 | 1.12 | Linux 5.15 | ✅ Full test suite green |

This row is informative — Python 3.13 is not yet exercised in CI, but
the maintainer's environment treats it as a known-good point inside the
declared bounds.

## Private anndata APIs pyscx depends on

Axis subsetting on a backed / lazy `X` cannot go through anndata's public
surface — reading an aligned mapping validates it, which raises the moment `X`
changes width — so `pyscx.accel.filter_cells` / `filter_genes` / `subset_obs`
and `highly_variable_genes(subset=True)` reach the raw stores directly. The
names below are private and unversioned; a rename inside the declared bound
would surface as an `AttributeError` from inside a user's `filter_cells`.
Verified byte-identical across `0.11.4` / `0.12.10` / `0.12.16`.

| Name | Used for |
|---|---|
| `AnnData._obs`, `._var` | Slice the axis frames without the public setter's length check |
| `AnnData._layers`, `._obsm`, `._varm`, `._obsp`, `._varp` | Reach aligned members without triggering read-time validation |
| `AnnData._inplace_subset_obs`, `._inplace_subset_var` | Delegate the whole subset for an in-memory `X` |

A compat test that fails loudly on upgrade is deferred to the structural
follow-up (Phase 4.0b), which replaces these with registered `as_view` /
`_subset` hooks.

## Known incompatibilities

- **`anndata >= 0.13`** — Out of the declared bound. The `0.13` line is
  expected to drop the legacy `AnnData(filename=...)` constructor path
  that `pyscx.from_anndata` still uses; revisit when `0.13` ships. It is
  also the release most likely to move the private names above.
- **`pyarrow >= 24`** — Out of the declared bound. Bump after running
  the `pyscx/tests/test_to_anndata_integration.py` suite locally on the
  new release; pyarrow's `RecordBatch` API has been stable, but the cap
  exists to gate on actual verification, not optimism.
- **`numpy < 1.24`** — Removed `np.float_` / `np.int_` aliases used in
  test fixtures. Older versions may import but fail at test time.
- **Python 3.10** — Out of `requires-python`. No technical blocker, but
  3.10 reaches end-of-life Oct 2026 and is not regression-tested.
- **R bindings (`rscx`)** — Not covered here. See
  [rscx documentation](../rscx/) and `R CMD INSTALL .` from the `rscx`
  directory.

## Bumping the matrix

When raising an upper bound in `pyproject.toml`:

1. Update the bound in `pyscx/pyproject.toml`.
2. Update the CI pin in `.github/workflows/ci.yml` (line ~112, the
   `pip install ... "pyarrow==..."` invocation).
3. Run `cd pyscx && ../.venv/bin/maturin develop && ../.venv/bin/pytest tests/ -v`
   on the maintainer's dev env.
4. Bump the row in this file and update the "Maintainer's local
   environment" row with the new resolved versions.
5. Reference the bump in the release changelog so downstreams know which
   row of this table the release was validated against.

When lowering a lower bound (rare): always add a CI matrix row pinning
the new minimum, otherwise the bound is aspirational rather than tested.
