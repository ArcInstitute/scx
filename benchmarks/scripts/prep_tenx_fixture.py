#!/usr/bin/env python3
"""Build the `<name>_10x.h5` fixtures — a 10x CellRanger-shaped HDF5 source.

## Why this exists

`scx convert --from 10x` gained a streaming path in OPT-CONVERT-9, and the
claim is a memory one: peak RSS bounded by the resident `indptr` plus one
shard's working set per outstanding worker, instead of the whole
`indptr`/`indices`/`data` triple. Nothing in the suite could measure that,
because **no benchmark dataset is a 10x file**. `DATASETS` declares 30-odd
entries and every one of them is sourced from an `.h5ad`; the only 10x `.h5`
the prep scripts ever touch is pbmc3k's, which `download_datasets.sh` converts
to h5ad and then deletes (`rm -f "$PBMC_H5"`).

So this script synthesises one, in the shape `prep_full_fixtures.py` already
established for the `.raw`/obsm/layer gap: a fixture built beside the real
datasets, read as an extra *arm* of a triple the default gate already
schedules, rather than as a new `DATASETS` entry (which would sit outside every
`capture_baseline.TIERS` list and be silently skipped) or a `FormatVariant`
(which the default format pool would not schedule).

## What it writes

`$SCX_DATA_DIR/<name>_10x.h5`, from `$SCX_DATA_DIR/<name>.h5ad`:

  * `/matrix/{indptr,indices,data}` — **copied verbatim**, in chunks. An h5ad
    CSR over cells×genes and a 10x CSC over genes×cells are the same three
    arrays: `indptr` runs over cells in both, and `indices` are gene ids in
    both. That is the reinterpretation `scx-convert`'s readers rely on
    (`tenx_read::read_tenx_h5`), so the fixture is a genuine 10x layout and not
    an approximation of one.
  * `/matrix/shape` — an **i32 dataset** holding `[n_genes, n_cells]`, which is
    what CellRanger emits. (Some synthetic fixtures write an attribute instead;
    `read_tenx_shape` accepts both, and the Rust tests cover both. The dataset
    form is the one real files use, so it is the one measured here.)
  * `/matrix/barcodes` — the h5ad's obs index.
  * `/matrix/features/{id,name,feature_type}` — the h5ad's var index for `id`
    and `name`, and `"Gene Expression"` throughout for `feature_type`.

Peak memory is bounded by `_CHUNK_NNZ` values plus the two index arrays: the
sparse arrays are never fully resident, so this runs on an ordinary node even
for census_1m. That is deliberate — a prep script that needs a high-mem
allocation is a prep script that does not get re-run.

## Usage

    python benchmarks/scripts/prep_tenx_fixture.py --datasets census_500k
    python benchmarks/scripts/prep_tenx_fixture.py --datasets census_500k --force
    python benchmarks/scripts/prep_tenx_fixture.py --datasets census_500k --dry-run

Idempotent: an existing output newer than its source is left alone unless
`--force`.

## Limitation, stated

The source `X` must be CSR on disk. A dense or CSC-on-disk h5ad is refused
rather than transposed — a CSC source would need the genes×cells arrays
rebuilt, which is the external-memory transpose this script has no business
reimplementing, and every dataset the 10x arms are scoped to is CSR.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Repo root so the benchmarks package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.comprehensive.config import DATASETS, DatasetConfig  # noqa: E402

logger = logging.getLogger("prep_tenx_fixture")

#: Values copied per chunk. 32M of them is 128 MB of f32 plus 128 MB of i32 —
#: bounded, and large enough that the copy is sequential-IO-bound rather than
#: per-call-overhead-bound.
_CHUNK_NNZ = 32 * 1024 * 1024

#: Every gene is an expression feature. CellRanger writes this column and
#: `read_tenx_var` carries it, so the fixture has it; the arm does not depend
#: on the value.
_FEATURE_TYPE = "Gene Expression"


def _read_index(group) -> list[str]:
    """The obs/var index of an h5ad dataframe group, as a list of `str`.

    Reads the dataset named by the group's `_index` attribute rather than
    assuming `_index`: anndata honours a renamed index, and `read_elem` on the
    whole group would pull every column.
    """
    name = group.attrs.get("_index", "_index")
    if isinstance(name, bytes):
        name = name.decode()
    raw = group[name][...]
    return [v.decode() if isinstance(v, bytes) else str(v) for v in raw]


def _build(dataset: DatasetConfig, out_path: Path, dry_run: bool) -> None:
    import h5py
    import numpy as np

    src = dataset.h5ad_path
    if not src.exists():
        raise FileNotFoundError(
            f"Source h5ad not found for {dataset.name}: {src}. "
            f"Stage it first (benchmarks/scripts/download_datasets.sh)."
        )

    if dry_run:
        logger.info("[dry-run] would build %s from %s", out_path, src)
        return

    t0 = time.perf_counter()
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    tmp.unlink(missing_ok=True)

    with h5py.File(src, "r") as fin:
        x = fin["X"]
        enc = x.attrs.get("encoding-type")
        if isinstance(enc, bytes):
            enc = enc.decode()
        if not isinstance(x, h5py.Group) or enc not in (None, "csr_matrix"):
            raise RuntimeError(
                f"{src}: /X is {enc!r} (group={isinstance(x, h5py.Group)}); this "
                f"script copies a CSR h5ad's arrays verbatim and refuses to "
                f"transpose a dense or CSC source. Use a CSR-on-disk dataset."
            )
        shape = x.attrs["shape"]
        n_cells, n_genes = int(shape[0]), int(shape[1])
        src_indptr = x["indptr"]
        src_indices = x["indices"]
        src_data = x["data"]
        nnz = int(src_indices.shape[0])
        if int(src_indptr.shape[0]) != n_cells + 1:
            raise RuntimeError(
                f"{src}: /X/indptr has {src_indptr.shape[0]} entries against "
                f"n_obs+1 = {n_cells + 1}; the source is not the CSR its "
                f"encoding-type claims."
            )
        logger.info(
            "  %s: %d cells x %d genes, nnz=%d", src.name, n_cells, n_genes, nnz
        )

        barcodes = _read_index(fin["obs"])
        gene_ids = _read_index(fin["var"])
        if len(barcodes) != n_cells or len(gene_ids) != n_genes:
            raise RuntimeError(
                f"{src}: obs index has {len(barcodes)} entries and var index "
                f"{len(gene_ids)}, against shape ({n_cells}, {n_genes})."
            )

        with h5py.File(tmp, "w") as fout:
            matrix = fout.create_group("matrix")
            # `[n_genes, n_cells]` as an i32 **dataset** — CellRanger's form.
            matrix.create_dataset(
                "shape", data=np.array([n_genes, n_cells], dtype=np.int32)
            )
            # Verbatim: an h5ad CSR over cells x genes and a 10x CSC over
            # genes x cells are the same arrays. No transpose, no re-indexing.
            matrix.create_dataset("indptr", data=src_indptr[...])
            out_indices = matrix.create_dataset(
                "indices", shape=(nnz,), dtype=src_indices.dtype
            )
            out_data = matrix.create_dataset(
                "data", shape=(nnz,), dtype=src_data.dtype
            )
            for start in range(0, nnz, _CHUNK_NNZ):
                end = min(start + _CHUNK_NNZ, nnz)
                out_indices[start:end] = src_indices[start:end]
                out_data[start:end] = src_data[start:end]
                logger.info("  copied nnz [%d, %d) of %d", start, end, nnz)

            vlen = h5py.string_dtype(encoding="utf-8")
            matrix.create_dataset("barcodes", data=barcodes, dtype=vlen)
            features = matrix.create_group("features")
            features.create_dataset("id", data=gene_ids, dtype=vlen)
            features.create_dataset("name", data=gene_ids, dtype=vlen)
            features.create_dataset(
                "feature_type", data=[_FEATURE_TYPE] * n_genes, dtype=vlen
            )

    try:
        _verify(tmp, n_cells, n_genes, nnz)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out_path)

    size_gb = out_path.stat().st_size / (1024**3)
    logger.info(
        "  Done: %s (%.2f GB, %.1f s)", out_path, size_gb, time.perf_counter() - t0
    )


def _verify(out_path: Path, n_cells: int, n_genes: int, nnz: int) -> None:
    """Refuse to leave behind a fixture that is not the shape it claims.

    The failure this guards is the axis swap, and it is silent: a `shape` of
    `[n_cells, n_genes]` still opens, still has a well-formed `indptr`, and
    produces an SCX file with cells and genes exchanged. So the check is
    against the **10x** convention explicitly — `shape[1]` must be the length
    the `indptr` describes — rather than against whatever was written.
    """
    import h5py

    with h5py.File(out_path, "r") as f:
        matrix = f["matrix"]
        shape = matrix["shape"][...]
        if list(shape) != [n_genes, n_cells]:
            raise RuntimeError(
                f"{out_path}: /matrix/shape is {list(shape)}, expected "
                f"[n_genes, n_cells] = [{n_genes}, {n_cells}] — the 10x axis "
                f"order. A swapped shape converts to a transposed SCX file "
                f"without erroring."
            )
        if matrix["indptr"].shape[0] != n_cells + 1:
            raise RuntimeError(
                f"{out_path}: /matrix/indptr has {matrix['indptr'].shape[0]} "
                f"entries, expected n_cells + 1 = {n_cells + 1}"
            )
        for name, expected in (("indices", nnz), ("data", nnz)):
            got = matrix[name].shape[0]
            if got != expected:
                raise RuntimeError(
                    f"{out_path}: /matrix/{name} has {got} entries, expected {expected}"
                )
        if matrix["barcodes"].shape[0] != n_cells:
            raise RuntimeError(f"{out_path}: /matrix/barcodes is not n_cells long")
        for col in ("id", "name", "feature_type"):
            got = matrix["features"][col].shape[0]
            if got != n_genes:
                raise RuntimeError(
                    f"{out_path}: /matrix/features/{col} has {got} entries, "
                    f"expected n_genes = {n_genes}"
                )
    logger.info(
        "  Verified: shape=[%d, %d] (genes, cells), nnz=%d, features complete",
        n_genes, n_cells, nnz,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build <name>_10x.h5 fixtures (a 10x CellRanger-shaped source)."
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help=f"Dataset names. Known: {sorted(DATASETS)}",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild even when the output already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be built without reading or writing anything.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    unknown = [n for n in args.datasets if n not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}. Known: {sorted(DATASETS)}")

    failures = 0
    for name in args.datasets:
        dataset = DATASETS[name]
        if dataset.multimodal:
            logger.warning(
                "Skipping %s: a 10x `/matrix` holds one modality, and the arms "
                "this fixture feeds are single-modality.", name,
            )
            continue
        out_path = dataset.tenx_path
        if out_path.exists() and not args.force:
            src_mtime = (
                dataset.h5ad_path.stat().st_mtime if dataset.h5ad_path.exists() else 0
            )
            if out_path.stat().st_mtime >= src_mtime:
                logger.info("Skipping %s (exists and is newer than its source): %s",
                            name, out_path)
                continue
            logger.warning(
                "%s exists but is OLDER than its source h5ad — rebuilding. "
                "(Pass --force to rebuild unconditionally.)", out_path,
            )
        try:
            _build(dataset, out_path, args.dry_run)
        except Exception:
            logger.exception("Failed to build the 10x fixture for %s", name)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
