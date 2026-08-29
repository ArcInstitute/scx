# `SparseCellSetDataset` scattered-read route captures

Manifest entries backing the `scatter_block_index` numbers in
[`docs/performance.md`](../../../../docs/performance.md#loader-adoption-scattered-reads),
per [`docs/benchmark_manifest.md`](../../../../docs/benchmark_manifest.md): every
`docs/performance.md` claim that can be expressed as a benchmark/format/dataset
triple must be backed by a JSON result checked in under
`benchmarks/comprehensive/results/`.

These live here rather than in `raw/` (which is gitignored) and are **not
promoted** into a baseline — the contract accepts either, and promotion would be
wrong here: the subject is a **reframed copy** of a registered fixture, so no
`gate_candidate.py` run reproduces this row.

## Why a reframed copy

`block_index_eligible` requires `shard_is_framed`. Every registered
`cellset_gather` fixture is `format_version = 3`, where both routes collapse to
the full-shard path and the comparison cannot be made at all. Each capture
therefore runs `scx optimize --row-group-rows 256` on a copy first, and the
driver refuses to continue unless the output is v4 **and** the two arms actually
took different routes.

## Regenerating

```bash
SOURCE="$SCX_DATA_DIR/tabula_sapiens_100k_auto.scx $SCX_DATA_DIR/pbmc10k_auto.scx" \
WORKDIR=$SCRATCH OUT_DIR=$SCRATCH N_RUNS=3 \
    sbatch --job-name=cellset-routes benchmarks/scripts/bench_cellset_scatter_routes.sbatch
```

`<stem>.json` is the driver's own payload (arms, premises, provenance);
`<stem>__manifest.json` is the schema-v2 `BenchmarkResult`. Both are kept: the
manifest is the contract artifact, the payload carries the premise checks that
say whether the comparison was valid.

Once `reconvert_fixtures.py` regenerates the registered fixtures at v4 this stops
being a special case — the route becomes measurable on a normal
`cellset_gather` triple, and the deferred route floors in `thresholds.yaml` can
be authored against it.
