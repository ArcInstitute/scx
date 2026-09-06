#!/usr/bin/env python3
"""Build the `<name>_full.scx` fixtures — an SCX file that actually has a
`.raw`, `obsm` keys and a layer.

## Why this exists

`export_streaming`'s `streaming_peak_rss_mb` ceiling is the suite's only real
export memory contract, and it is measured on `census_1m_auto.scx`. That file
has no `.raw`, `obsm_keys == []` and `layer_names == []` — as does every other
`.scx` in the suite, because no source h5ad carries any of the three. So the
ceiling never reaches the parts of the export path that handle them:

  * `/raw`  — streamed since PR-04 (OPT-CONVERT-1) by `stream_write.rs`'s
              `stream_raw_at`. Before that, `write_raw_to_h5ad` called
              `read_all_raw_csr_shards()` and held the whole raw matrix
              resident — the dominant term, 1487 MB at tabula_sapiens_100k.
              Still the reason this fixture exists: a raw that regressed to a
              materialising read produces byte-identical output, so peak RSS
              on this fixture is the only signal.
  * `obsm`  — `stream_write.rs` reads `read_all_obsm()` whole (and emits an
              `ExportFilterSectionEager` warning under a keep mask). Small at
              ordinary embedding widths: 19.8 MB for 52 columns at 100k cells.
  * layers  — already streamed via `stream_layers_at`; covered so a regression
              to a materialising path would show.

A green threshold on a fixture with none of them proves the file was small, not
that the export streams. This script produces the fixture that can tell the
difference.

## What it writes

`$SCX_DATA_DIR/<name>_full.scx`, from `$SCX_DATA_DIR/<name>.h5ad`:

  * `.raw`               — the counts matrix (a copy of `X`), landing as
                           `RawCsrShard` sections;
  * `obsm["X_pca"]`      — n_obs x 50 float32, seeded RNG;
  * `obsm["X_umap"]`     — n_obs x 2  float32, seeded RNG;
  * `layers["counts"]`   — a second reference to `X`.

The embeddings are noise on purpose. Nothing downstream interprets them; what
is being measured is the cost of carrying an `obsm` matrix through an export,
and that cost is a function of shape, not of content. They are seeded so two
builds of the same dataset produce the same file.

## Usage

    python benchmarks/scripts/prep_full_fixtures.py --datasets tabula_sapiens_100k
    python benchmarks/scripts/prep_full_fixtures.py --datasets tabula_sapiens_100k --force
    python benchmarks/scripts/prep_full_fixtures.py --datasets tabula_sapiens_100k --dry-run

Idempotent: an existing output newer than its source is left alone unless
`--force`. Run it from a node with enough RAM — this path materialises the whole
AnnData (`pyscx.from_anndata` takes an in-memory object), so it needs roughly
3x the source h5ad. tabula_sapiens_100k is ~1.5 GB on disk; census_1m is ~11 GB
and wants a high-mem allocation.
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

logger = logging.getLogger("prep_full_fixtures")

# Embedding widths. 50 PCs and a 2-D UMAP are what a real analysis file carries;
# the point is the shape, not the values.
_N_PCS = 50
_N_UMAP = 2

# Fixed so a rebuild is reproducible.
_SEED = 20260901

_LAYER_NAME = "counts"


def _build(dataset: DatasetConfig, out_path: Path, dry_run: bool) -> None:
    import anndata
    import numpy as np
    import pyscx

    src = dataset.h5ad_path
    if not src.exists():
        raise FileNotFoundError(
            f"Source h5ad not found for {dataset.name}: {src}. "
            f"Stage it first (benchmarks/scripts/download_datasets.sh, or "
            f"benchmarks/comprehensive/scripts/prep_grouped_fixtures.py for the "
            f"perturbation fixtures)."
        )

    if dry_run:
        logger.info(
            "[dry-run] would build %s from %s (obsm X_pca[%d], X_umap[%d], "
            "layers[%r], .raw)",
            out_path, src, _N_PCS, _N_UMAP, _LAYER_NAME,
        )
        return

    t0 = time.perf_counter()
    logger.info("Reading %s", src)
    adata = anndata.read_h5ad(src)
    n_obs = adata.n_obs
    logger.info("  %d x %d, nnz=%s", n_obs, adata.n_vars,
                getattr(adata.X, "nnz", "dense"))

    # `.raw` must be set from an AnnData whose X is the counts matrix. anndata
    # reads the shape off `raw.X`, and so does the SCX writer — `Raw.shape`
    # reports the *parent's* n_obs, which is why nothing here consults it.
    if adata.raw is None:
        adata.raw = adata
    if adata.raw is None or adata.raw.X is None:
        raise RuntimeError(
            f"{src}: could not attach a .raw. `from_anndata` duck-types this "
            f"attribute and skips a None *silently*, so the write would succeed "
            f"and produce a fixture with no raw at all."
        )

    rng = np.random.default_rng(_SEED)
    if "X_pca" not in adata.obsm:
        adata.obsm["X_pca"] = rng.standard_normal(
            (n_obs, _N_PCS), dtype=np.float32
        )
    if "X_umap" not in adata.obsm:
        adata.obsm["X_umap"] = rng.standard_normal(
            (n_obs, _N_UMAP), dtype=np.float32
        )
    if _LAYER_NAME not in adata.layers:
        adata.layers[_LAYER_NAME] = adata.X

    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    logger.info("Writing %s", tmp)
    pyscx.from_anndata(adata, str(tmp), codec="auto")

    # Verify the temp file and only then move it into place. A fixture that
    # failed verification must not be left where a benchmark would silently
    # pick it up — and the benchmark arms key off existence alone.
    try:
        _verify(tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out_path)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(
        "  Done: %s (%.1f MB, %.1f s)", out_path, size_mb, time.perf_counter() - t0
    )


def _verify(out_path: Path) -> None:
    """Refuse to leave behind a fixture that is missing the thing it is for.

    The whole value of this file is that it has a `.raw`, obsm keys and a
    layer. A silent drop would produce a fixture indistinguishable from
    `<name>_auto.scx`, and the thresholds measured on it would read as passing
    while measuring nothing — which is exactly the failure this fixture exists
    to end.

    The drop is not hypothetical for `.raw`: `from_anndata` duck-types
    `adata.raw` and simply skips a `None`, so a source that lost its raw
    somewhere upstream writes a raw-less file with no error.

    `Experiment.validate()` is the check, rather than the `obsm_keys()` /
    `layer_names()` accessors alone, because there is no `has_raw` on the
    Python `Experiment` (`has_csc` and `has_deletions` exist; this one does
    not). `validate()` returns `[(section_name, checksum_ok)]` straight off the
    catalog, so it can see `raw/X_shard_*` — and it verifies every section's
    checksum on the way past, which is worth having on a fixture that will
    outlive this process by months.
    """
    import pyscx

    exp = pyscx.open(str(out_path))
    try:
        sections = list(exp.validate())
        obsm = list(exp.obsm_keys())
        layers = list(exp.layer_names())
    finally:
        close = getattr(exp, "close", None)
        if close is not None:
            close()

    names = [name for name, _ in sections]
    corrupt = sorted(name for name, ok in sections if not ok)
    if corrupt:
        raise RuntimeError(f"{out_path} has failing checksums: {corrupt}")

    missing = []
    if not any(n.startswith("raw/X_shard_") for n in names):
        missing.append(".raw (no raw/X_shard_* sections)")
    for key in ("X_pca", "X_umap"):
        if key not in obsm:
            missing.append(f"obsm[{key!r}]")
    if _LAYER_NAME not in layers:
        missing.append(f"layers[{_LAYER_NAME!r}]")
    if missing:
        raise RuntimeError(
            f"{out_path} is missing {', '.join(missing)} (obsm={obsm}, "
            f"layers={layers}, sections={names}). The fixture exists precisely "
            f"to carry these; a benchmark arm reading it would measure nothing."
        )
    n_raw = sum(1 for n in names if n.startswith("raw/X_shard_"))
    logger.info(
        "  Verified: %d raw/X shards, obsm=%s, layers=%s, %d sections checksum-clean",
        n_raw, obsm, layers, len(sections),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build <name>_full.scx fixtures (.raw + obsm + layer)."
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
                "Skipping %s: multimodal sources are out of scope for this "
                "fixture (the export arms it feeds are single-modality).", name,
            )
            continue
        out_path = dataset.scx_full_path
        if out_path.exists() and not args.force:
            src_mtime = dataset.h5ad_path.stat().st_mtime if dataset.h5ad_path.exists() else 0
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
            logger.exception("Failed to build the full fixture for %s", name)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
