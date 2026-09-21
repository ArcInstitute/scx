#!/usr/bin/env python3
"""Stage the atlas-scale synthetic multimodal fixtures.

`multimodal_atlas_streaming` exists to show what in-decode dtype narrowing
(`to_mudata(data_dtype=...)`) is worth where MuData's all-`f32` value buffers
actually hurt, which is at >=500K cells. The suite's two real multimodal
fixtures are 5.2K and 11.9K cells, so there is nothing to measure it on. This
builds the two that the benchmark's `DATASETS` entries declare:

    multiome_atlas_500k    500,000 x (35,000 RNA @ 4%  + 150,000 ATAC @ 1.5%)
    citeseq_atlas_1m     1,000,000 x (30,000 RNA @ 3%  +     250 ADT dense)

≈1.83 B and ≈1.15 B nonzeros, ≈15 GB and ≈9 GB of uncompressed `.h5mu`.

## Why it is not a `MuData(...)` round trip

The obvious shape — build the matrices with `scipy.sparse.random`, hand them to
`MuData`, call `write_h5mu` — is what `benchmarks/multimodal_query_bench.py`
does at 200K x 3K, and it does not survive the jump. A single
`sp.random(1_000_000, 30_000, density=0.03)` builds a 900 M-nonzero COO in one
allocation before anything is written.

So the matrices are generated a row block at a time and appended to
pre-created HDF5 datasets. Peak memory is one block, not one atlas.

The envelope still comes from `mudata` rather than being hand-rolled: a
skeleton `MuData` carrying the full `obs`, the full per-modality `var` and a
**zero-nonzero** `X` is written with `write_h5mu`, which costs nothing and gets
every encoding attribute, the global `var`, `obsmap` and `varmap` exactly
right. Only the three CSR arrays are then replaced in place. Hand-writing the
`.h5mu` contract (seven encoding-type vocabularies across four index maps) is
the kind of thing that produces a file every reader accepts except the one that
matters.

`indptr` is written as `int64`. mudata writes `int32`, which overflows at
2.1 B nonzeros — above the multiome atlas but not by much, and a fixture that
silently wraps is worse than one that fails to build.

## Values

Counts are integers in `[1, 65535]`, so the `scx_eager_u16` arm's narrowing is
exact and needs no `allow_lossy=True`. Stored `f32`, which is what MuData does
and what the `f32` control arm has to read for the ratio to mean anything.

Per-row nonzero counts are drawn per row rather than fixed, so
`n_genes_by_counts` is not constant — a fixture where every cell has identical
detection makes any QC-shaped downstream check degenerate.

Usage
-----
    python benchmarks/scripts/build_multimodal_atlas.py \\
        --datasets multiome_atlas_500k --force

    sbatch benchmarks/scripts/slurm_build_multimodal_atlas.sh multiome_atlas_500k
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.comprehensive.config import DATASETS, DatasetConfig  # noqa: E402

logger = logging.getLogger("build_multimodal_atlas")

#: Rows generated, sorted and flushed at a time. At the ATAC geometry this is
#: ~74 M nonzeros per block, or roughly 2.5 GB live including the sort's own
#: index array — comfortable on `cpu_high_mem` and the knob to turn down if it
#: is ever run somewhere smaller.
DEFAULT_BLOCK_ROWS = 32_768

#: Mean of the Poisson the count values are drawn from, shifted to >= 1. Low on
#: purpose: droplet data is shallow, and a heavy tail would push values out of
#: uint16 and break the narrowing arm's exactness.
COUNT_LAMBDA = 1.5
COUNT_MAX = 65_535

CELL_TYPES = (
    "CD4_T", "CD8_T", "NK", "B_naive", "B_memory", "CD14_Mono",
    "CD16_Mono", "cDC", "pDC", "Platelet", "HSPC", "Erythrocyte",
)
N_DONORS = 20
N_BATCHES = 4


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    n_vars: int
    density: float
    #: `modality_types` value handed to `pyscx.from_h5mu`.
    scx_type: str
    var_prefix: str


@dataclass(frozen=True)
class AtlasSpec:
    n_obs: int
    modalities: tuple[ModalitySpec, ...]

    @property
    def n_vars_total(self) -> int:
        return sum(m.n_vars for m in self.modalities)


#: Geometry per dataset. Cross-checked against `DATASETS` by
#: `check_against_registry` so the two cannot drift: the registry's `n_obs` /
#: `n_vars` feed `estimate_memory_gb`, and a generator that quietly built
#: something else would leave every SLURM allocation sized for a file that does
#: not exist.
ATLAS_GEOMETRY: dict[str, AtlasSpec] = {
    "multiome_atlas_500k": AtlasSpec(
        n_obs=500_000,
        modalities=(
            ModalitySpec("rna", 35_000, 0.04, "rna", "GENE"),
            ModalitySpec("atac", 150_000, 0.015, "atac", "PEAK"),
        ),
    ),
    "citeseq_atlas_1m": AtlasSpec(
        n_obs=1_000_000,
        modalities=(
            ModalitySpec("rna", 30_000, 0.03, "rna", "GENE"),
            # "dense/shallow": every cell carries a value for every antibody,
            # which is what an ADT panel actually looks like.
            ModalitySpec("adt", 250, 1.0, "protein", "ADT"),
        ),
    ),
}


def check_against_registry(name: str, spec: AtlasSpec) -> None:
    cfg = DATASETS[name]
    if cfg.n_obs != spec.n_obs:
        raise SystemExit(
            f"{name}: DATASETS says n_obs={cfg.n_obs}, geometry says {spec.n_obs}"
        )
    if cfg.n_vars != spec.n_vars_total:
        raise SystemExit(
            f"{name}: DATASETS says n_vars={cfg.n_vars}, geometry says "
            f"{spec.n_vars_total}"
        )
    declared = tuple(cfg.modality_names)
    built = tuple(m.name for m in spec.modalities)
    if declared != built:
        raise SystemExit(
            f"{name}: DATASETS modality_names={declared}, geometry builds {built}"
        )


def expected_nnz(spec: AtlasSpec) -> dict[str, int]:
    return {
        m.name: int(round(spec.n_obs * m.n_vars * m.density)) for m in spec.modalities
    }


# ---------------------------------------------------------------------------
# Row-block generation
# ---------------------------------------------------------------------------

def generate_block(
    rng: np.random.Generator, n_rows: int, mod: ModalitySpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row block as `(row_nnz, indices, data)`, indices sorted within a row.

    Column positions are drawn with replacement and then deduplicated, so the
    realised per-row count is at or below the drawn target. That is why
    `indptr` is accumulated from what was actually written rather than
    predicted — a pre-computed `indptr` would be wrong by the duplicate rate.
    """
    if mod.density >= 1.0:
        # Fully dense: the draw-and-dedupe path would generate n_vars entries
        # per row and throw ~37 % of them away.
        row_nnz = np.full(n_rows, mod.n_vars, dtype=np.int64)
        indices = np.tile(
            np.arange(mod.n_vars, dtype=np.int32), n_rows,
        )
    else:
        target = rng.binomial(mod.n_vars, mod.density, size=n_rows).astype(np.int64)
        total = int(target.sum())
        cols = rng.integers(0, mod.n_vars, size=total, dtype=np.int64)
        rows = np.repeat(np.arange(n_rows, dtype=np.int64), target)
        order = np.lexsort((cols, rows))
        rows = rows[order]
        cols = cols[order]
        keep = np.empty(total, dtype=bool)
        keep[:1] = True
        if total > 1:
            keep[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
        rows = rows[keep]
        cols = cols[keep]
        row_nnz = np.bincount(rows, minlength=n_rows).astype(np.int64)
        indices = cols.astype(np.int32)

    nnz = int(row_nnz.sum())
    data = (1 + rng.poisson(COUNT_LAMBDA, size=nnz)).astype(np.float32)
    np.clip(data, 1, COUNT_MAX, out=data)
    return row_nnz, indices, data


def build_obs(n_obs: int, rng: np.random.Generator):
    import pandas as pd

    return pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                np.asarray(CELL_TYPES)[rng.integers(0, len(CELL_TYPES), n_obs)]
            ),
            "donor": pd.Categorical(
                np.asarray([f"donor{i:02d}" for i in range(N_DONORS)])[
                    rng.integers(0, N_DONORS, n_obs)
                ]
            ),
            # The `scx_query_mod` arm filters on this, so its levels have to be
            # few enough that a single-batch predicate is genuinely selective.
            "batch": pd.Categorical(
                np.asarray([f"b{i}" for i in range(N_BATCHES)])[
                    rng.integers(0, N_BATCHES, n_obs)
                ]
            ),
        },
        index=[f"cell_{i:08d}" for i in range(n_obs)],
    )


def var_names_for(mod: ModalitySpec) -> list[str]:
    if mod.name == "atac":
        # Peak-shaped names, unique across modalities.
        return [f"{mod.var_prefix}:chr{i % 22 + 1}:{i * 500}-{i * 500 + 400}"
                for i in range(mod.n_vars)]
    return [f"{mod.var_prefix}_{i:06d}" for i in range(mod.n_vars)]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_skeleton(path: Path, spec: AtlasSpec, rng: np.random.Generator) -> None:
    """A real `.h5mu` with the right envelope and no matrix data.

    `write_h5mu` on zero-nonzero matrices costs nothing and produces every
    encoding attribute, the global `var`, and the `obsmap` / `varmap` index
    maps correctly. The CSR arrays are replaced afterwards.
    """
    import anndata
    import mudata
    import scipy.sparse as sp

    mods = {}
    for mod in spec.modalities:
        empty = sp.csr_matrix((spec.n_obs, mod.n_vars), dtype=np.float32)
        ad = anndata.AnnData(X=empty)
        ad.var_names = var_names_for(mod)
        mods[mod.name] = ad
    mu = mudata.MuData(mods)
    mu.obs = build_obs(spec.n_obs, rng)
    mu.update()
    logger.info("writing skeleton envelope -> %s", path.name)
    mudata.write_h5mu(str(path), mu)


def fill_matrices(
    path: Path, spec: AtlasSpec, rng: np.random.Generator, block_rows: int,
) -> dict[str, int]:
    """Replace each modality's CSR arrays with generated data, block by block."""
    import h5py

    realised: dict[str, int] = {}
    with h5py.File(path, "a") as f:
        for mod in spec.modalities:
            group = f[f"mod/{mod.name}/X"]
            for key in ("data", "indices", "indptr"):
                if key in group:
                    del group[key]
            # HDF5 allocates whole chunks, so a 1 M-element chunk costs 4 MB
            # per dataset whatever is in it. That is nothing at atlas scale and
            # 25 MB for a 17 K-nonzero smoke fixture, which makes the smoke
            # case unreadable as a size check.
            est_nnz = max(1 << 12, int(spec.n_obs * mod.n_vars * mod.density))
            chunk = (min(1 << 20, est_nnz),)
            data_ds = group.create_dataset(
                "data", shape=(0,), maxshape=(None,), dtype="float32",
                chunks=chunk,
            )
            idx_ds = group.create_dataset(
                "indices", shape=(0,), maxshape=(None,), dtype="int32",
                chunks=chunk,
            )
            # int64, not mudata's int32: 2.1 B nonzeros is not far above the
            # multiome atlas, and a fixture that silently wraps is worse than
            # one that fails to build.
            indptr = np.zeros(spec.n_obs + 1, dtype=np.int64)

            written = 0
            t0 = time.time()
            for start in range(0, spec.n_obs, block_rows):
                stop = min(start + block_rows, spec.n_obs)
                row_nnz, indices, data = generate_block(rng, stop - start, mod)
                n = indices.shape[0]
                data_ds.resize((written + n,))
                idx_ds.resize((written + n,))
                data_ds[written:written + n] = data
                idx_ds[written:written + n] = indices
                indptr[start + 1:stop + 1] = row_nnz
                written += n
                if (start // block_rows) % 4 == 0:
                    pct = 100.0 * stop / spec.n_obs
                    logger.info(
                        "  %s: %6.1f%%  rows %8d/%d  nnz %13d  %.0fs",
                        mod.name, pct, stop, spec.n_obs, written, time.time() - t0,
                    )
            np.cumsum(indptr, out=indptr)
            group.create_dataset("indptr", data=indptr, dtype="int64")
            group.attrs["shape"] = np.array(
                [spec.n_obs, mod.n_vars], dtype="int64",
            )
            realised[mod.name] = written
            logger.info(
                "  %s: done, %d nnz (%.3f density) in %.0fs",
                mod.name, written, written / (spec.n_obs * mod.n_vars),
                time.time() - t0,
            )
    return realised


def verify_h5mu(path: Path, spec: AtlasSpec, realised: dict[str, int]) -> None:
    """Read it back the way a consumer will, and check what the arms rely on.

    A fixture that writes without error but that `mudata.read_h5mu` refuses, or
    whose modality shapes disagree with the registry, would leave the benchmark
    scheduling cells that all fail at open — which reads as a broken benchmark
    rather than a broken fixture.
    """
    import mudata

    mu = mudata.read_h5mu(str(path), backed="r")
    try:  # noqa: SIM105 — the close below must run even on a SystemExit
        if mu.n_obs != spec.n_obs:
            raise SystemExit(f"{path.name}: n_obs {mu.n_obs} != {spec.n_obs}")
        for mod in spec.modalities:
            if mod.name not in mu.mod:
                raise SystemExit(f"{path.name}: modality {mod.name} missing")
            got = mu.mod[mod.name].shape
            if got != (spec.n_obs, mod.n_vars):
                raise SystemExit(
                    f"{path.name}: {mod.name} shape {got} != "
                    f"{(spec.n_obs, mod.n_vars)}"
                )
        for col in ("cell_type", "donor", "batch"):
            if col not in mu.obs.columns:
                raise SystemExit(f"{path.name}: obs['{col}'] missing")
        # A sample block must round-trip as integral values in uint16 range,
        # or the narrowing arm needs allow_lossy and stops being exact.
        block = mu.mod[spec.modalities[0].name].X[:256]
        vals = np.asarray(block.data if hasattr(block, "data") else block).ravel()
        if vals.size:
            if not np.all(vals == np.floor(vals)):
                raise SystemExit(f"{path.name}: non-integral counts")
            if vals.max() > COUNT_MAX:
                raise SystemExit(f"{path.name}: value {vals.max()} exceeds uint16")
    finally:
        handle = getattr(mu, "file", None)
        if handle is not None:
            handle.close()
    logger.info(
        "verified %s: %s", path.name,
        ", ".join(f"{k}={v} nnz" for k, v in realised.items()),
    )


def convert_to_scx(h5mu: Path, out: Path, spec: AtlasSpec) -> None:
    """Stream the `.h5mu` into a multimodal `.scx`.

    `pyscx.from_mudata` takes a resident MuData and is not an option at this
    size; `from_h5mu` streams each modality independently.
    """
    import pyscx

    tmp = out.with_suffix(out.suffix + ".partial")
    tmp.unlink(missing_ok=True)
    t0 = time.time()
    logger.info("converting %s -> %s (streaming)", h5mu.name, out.name)
    pyscx.from_h5mu(
        str(h5mu), str(tmp), codec="auto",
        modality_types={m.name: m.scx_type for m in spec.modalities},
        memory_budget="32G",
    )
    exp = pyscx.open(str(tmp))
    names = set(exp.modality_names)
    expected = {m.name for m in spec.modalities}
    if names != expected:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{out.name}: modalities {names} != {expected}")
    if exp.n_obs != spec.n_obs:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{out.name}: n_obs {exp.n_obs} != {spec.n_obs}")
    del exp
    tmp.replace(out)
    logger.info(
        "  wrote %s (%.1f GB) in %.0fs",
        out.name, out.stat().st_size / 1e9, time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_one(
    name: str, *, force: bool, dry_run: bool, block_rows: int, seed: int,
    skip_scx: bool,
) -> bool:
    spec = ATLAS_GEOMETRY[name]
    check_against_registry(name, spec)
    cfg: DatasetConfig = DATASETS[name]
    # `h5mu_path` does not branch on `synthetic` the way `h5ad_path` does, so
    # these land flat in DATA_DIR beside the real fixtures — which is what
    # `path_for_format` and `convert.py` will read.
    h5mu = cfg.h5mu_path
    scx = cfg.scx_multimodal_path
    nnz = expected_nnz(spec)

    logger.info(
        "%s: %d cells x %s  (~%.2f B nnz, ~%.1f GB h5mu)",
        name, spec.n_obs,
        " + ".join(f"{m.n_vars} {m.name} @ {m.density:.1%}" for m in spec.modalities),
        sum(nnz.values()) / 1e9, sum(nnz.values()) * 8 / 1e9,
    )
    if dry_run:
        logger.info("  dry-run: would write %s and %s", h5mu, scx)
        return True
    if h5mu.exists() and not force:
        logger.info("  %s exists; --force to rebuild", h5mu.name)
    else:
        tmp = h5mu.with_suffix(h5mu.suffix + ".partial")
        tmp.unlink(missing_ok=True)
        h5mu.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        write_skeleton(tmp, spec, rng)
        realised = fill_matrices(tmp, spec, rng, block_rows)
        verify_h5mu(tmp, spec, realised)
        # Atomic: a half-written fixture that looks present is worse than an
        # absent one, because the benchmark would schedule against it.
        tmp.replace(h5mu)
        logger.info("  wrote %s (%.1f GB)", h5mu.name, h5mu.stat().st_size / 1e9)

    if skip_scx:
        return True
    if scx.exists() and not force:
        logger.info("  %s exists; --force to rebuild", scx.name)
        return True
    convert_to_scx(h5mu, scx, spec)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--datasets", nargs="+", required=True,
        help=f"Which atlases to build. Known: {sorted(ATLAS_GEOMETRY)}",
    )
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even when the output already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the geometry and the paths, write nothing.")
    ap.add_argument("--block-rows", type=int, default=DEFAULT_BLOCK_ROWS,
                    help=f"Rows generated at a time (default {DEFAULT_BLOCK_ROWS}).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-scx", action="store_true",
                    help="Write only the .h5mu; leave the .scx to the harness.")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING - 10 * min(args.verbose + 1, 2),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    unknown = sorted(set(args.datasets) - set(ATLAS_GEOMETRY))
    if unknown:
        raise SystemExit(
            f"Unknown dataset(s): {unknown}. Known: {sorted(ATLAS_GEOMETRY)}"
        )

    failures = 0
    for name in args.datasets:
        try:
            build_one(
                name, force=args.force, dry_run=args.dry_run,
                block_rows=args.block_rows, seed=args.seed,
                skip_scx=args.skip_scx,
            )
        except SystemExit:
            raise
        except Exception:
            logger.exception("%s: build failed", name)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
