# scx-engine — Query Engine

> Part of the [SCX API reference](README.md).

## `QueryPipeline`
Lazy pipeline builder — no I/O until `.collect()`:

```rust
QueryPipeline::open("file.scx")?
    .filter_obs("cell_type == 'T cell' and tissue == 'lung'")?
    .select_genes(hvg_indices)
    .with_normalize(1e4)
    .with_log1p()
    .limit(1000)
    .collect()?   // → QueryResult { x: ScxCsr, obs, var, skipped_shards, total_shards }
```

- Schema validation at construction time (fail-fast)
- Two-level predicate pushdown: catalog shard pruning + index row pruning
- **Level-1 stats are re-derived or dropped, never silently reused, when obs is
  rewritten.** Catalog shard pruning reads a per-shard `MinMax` /
  `CategoryBitset` for each indexed obs column — for the numeric arm without
  consulting the predicate index at all. So the in-place ops that rewrite obs
  values (`modify_metadata(obs=…)`, `obs_import` / `doublet_import`,
  `cellbender_import`) never leave a bound describing values that are gone.
  `modify_metadata` carries the file's index forward and re-derives the stats
  with it, so an indexed file keeps its pushdown; the import ops clear the stats
  for the columns they replace. Where a clear does happen the cost is a slower
  query, never a short result. Rebuild with
  `modify_metadata(..., index_obs=[…])` or a copy-out `sort` / `compact` with
  `--index-obs` / `--index-preset`. Full per-op table in
  [docs/operations.md § Per-shard column stats and the in-place ops](../operations.md#per-shard-column-stats-and-the-in-place-ops).
- **A dictionary miss short-circuits only on a complete vocabulary.** A
  `filter_obs("cell_type == 'Typo'")` whose value is absent from an indexed
  column's category dictionary can be answered instantly by pruning every
  shard, instead of scanning obs to find nothing. That inference is global
  ("this value is in *no* shard") drawn from a local artifact (the vocabulary
  this file's index happens to hold), so it is made only when the vocabulary
  is known to cover every shard being pruned — established by the catalog
  itself. Every shard must carry, for that column, a `CategoryBitset` (not just
  any stat under the same hash — `MinMax` answers to it too) whose byte length
  is the one the vocabulary implies, consistently across shards. The index
  build emits exactly that for *every* shard of an indexed categorical column,
  including an all-zero bitset where the column has no values there, and sizes
  each from the same entry list the dictionary comes from. So a missing bitset
  means that shard was never seen by the build, and a wrong-length one means
  the stats and the index section came from different builds — where bit *i*
  no longer means entry *i*. A column appearing **twice within one shard**
  disqualifies it too: nothing says which of the two bitsets the dictionary's
  positions belong to, and counting stat records rather than shards would let
  one shard's surplus stand in for a shard that carries none.
  - In practice the vocabulary is complete on anything written by `convert`,
    `merge`, `compact`, `sort` or `subset` with `--index-*`. It is **not**
    complete after `append` without `--index-obs`, which adds shards the
    file-scope index has never seen. Such a file answers correctly either way
    (an appended shard carries no column stats, so pruning never touches it),
    but a miss on it costs a full obs scan; `--index-obs` on the append, or a
    copy-out `compact --index-obs`, restores the short-circuit.
  - The same bit gates Level-2. There a partial vocabulary is worse than slow:
    `categorical_eq` reports an absent value as an *exact empty row-set*, and
    reports a present one with only the rows that were recorded — so an
    incomplete column is residual (decode + mask) rather than resolvable.
  - An **empty** vocabulary is never complete, however well covered. It cannot
    tell "this column has no values" apart from "this build recorded none of
    them". That distinction is load-bearing on files written before
    integer-valued categoricals were classified: such a file carries an
    entry-less categorical index, every shard gets a zero-length
    `CategoryBitset` for it, and `batch == '1'` would prune every shard and
    return nothing — silently, because the type mismatch that rejects a string
    literal against an integer column lives in the residual evaluator, which
    never runs once Level-1 has short-circuited. Those columns now fall back to
    a scan (and still raise the type error). Re-index with `--index-obs` to
    make the column queryable.
  - `in [...]` and `==` make the identical inference. They used to disagree —
    `==` pruned on a miss unconditionally while `in` refused to — which meant
    one predicate had two answers depending on how it was spelled.
- Fused normalize+log1p in single CSR row scan
- Parallel shard processing via rayon
- **Bounded obs memory on row-sharded files.** `filter_obs` / `count`
  evaluate obs predicates one metadata shard at a time into an `n_obs`
  boolean mask rather than concatenating every obs shard into RAM, and
  skip decoding obs shards that don't overlap a surviving CSR shard (when
  per-shard catalog row-range stats are present — written going forward,
  so existing atlas files still get the memory bound but not the I/O
  skip). Peak obs memory is ~`(rayon width × one shard) + n_obs` bytes,
  not the full obs table. This bound applies **only to the query engine
  path**: other `read_obs()` callers (`compact` / `merge` / streaming
  export / CLI `subset` / `to_anndata`) still assemble the full obs table
  and remain unbounded on atlas-scale sharded files.
- **Filtered-obs categorical semantics.** A `collect()` whose rows the caller
  narrowed — `filter_obs(...)` or `limit(...)` — returns categorical obs
  columns carrying only the categories present in the surviving rows, in
  declared order, not the full parent dictionary — standard AnnData/pandas
  behavior (`remove_unused_categories` on a subset), on both obs layouts and
  whatever the result size, an empty result included. An unfiltered
  `collect()`, like `read_obs()` on the file itself, keeps the full declared
  list. This used to have a gap: on a file grown by `append`, which decoded the
  rows it added to plain strings, a filtered `collect()` whose surviving rows
  all fell in appended shards saw no dictionary shard to reconcile against and
  returned that column as plain strings. `append` / `merge` now write
  dictionaries, so it holds on their output too. Downstream code that
  compares `.cat.categories` against the source file (e.g. plotting that
  assumes a fixed palette) should re-derive categories from the result.
- **Null semantics — three-valued (Kleene) logic, like a SQL `WHERE`
  clause.** A comparison against a NULL cell is UNKNOWN, not `false`.
  `and` / `or` combine UNKNOWN accordingly — **`null OR true` is `true`**
  and `null AND false` is `false` — `not` propagates UNKNOWN, and only the
  final mask turns a surviving UNKNOWN into "not matched". In full:

  | | `and` | `or` |
  |---|---|---|
  | `TRUE` ∘ `UNKNOWN` | `UNKNOWN` | **`TRUE`** |
  | `FALSE` ∘ `UNKNOWN` | **`FALSE`** | `UNKNOWN` |
  | `UNKNOWN` ∘ `UNKNOWN` | `UNKNOWN` | `UNKNOWN` |

  with `not UNKNOWN = UNKNOWN`, and a top-level `UNKNOWN` → not matched. So
  `filter_obs("cell_type == 'B cell' or n_genes > 5000")` returns a cell
  with an unannotated `cell_type` and 9000 genes; pandas, polars and SQL
  agree. The same rules apply to `filter_var`, and to every other surface
  driven by this evaluator: `scx delete --filter`, `scx subset --filter`,
  and `to_anndata(obs_filter=...)` on the non-backed default path.
  **Divergence from pandas on `!=` / `not`:** the engine leaves a NULL cell
  UNKNOWN, so a NULL row does *not* match `x != 'v'`; pandas is two-valued
  (`NaN != 'v'` is `True`) and returns it. This matters when the same filter
  string is reused across `backed=True` (pandas) and the default path
  (engine) — see [Filter Expression Compatibility](../scanpy/loading.md#filter-expression-compatibility).

## Grouped reads (condition/label-grouped sharding)

On an archive written by `scx sort --group-by` (see
[sharding.md § Condition/label-grouped sharding](../sharding.md)),
`QueryPipeline` exposes a targeted-read API over the `group_index` sidecar:

```rust
let pipe = QueryPipeline::open("grouped.scx")?;
pipe.read_group("MYC")?;        // QueryResult — just the MYC cells (Route B slice)
pipe.read_reference()?;          // Option<QueryResult> — the full reference region
pipe.group_labels()?;            // Vec<String>
pipe.iter_group_shards()?;       // Vec<GroupShardHandle> — per non-reference shard
pipe.read_row_range(start, stop)?; // the underlying contiguous-range read
```

- **Raw reads.** These ignore builder state (`filter_obs`/`filter_var`/
  `select_genes`/`with_normalize`/`with_log1p`/`limit`) — they return the
  range's cells only. Deletion vectors **are** honored (rows deleted after the
  sort are dropped), so results match `query().collect()` for the same cells.
- **Errors:** `EngineError::NotGrouped` when the archive has no sidecar;
  `EngineError::UnknownGroupLabel { suggestions }` (with `strsim` close matches)
  for an unknown label; bounds/coverage errors from `read_row_range` on an
  out-of-range or gapped range.

In pyscx these surface on `Experiment` (each opens a fresh pipeline, so builder
bypass cannot happen):

- `Experiment.read_group(label) -> AnnData` — raises `KeyError` (with
  suggestions) on miss, `ValueError` if not grouped.
- `Experiment.read_reference() -> AnnData | None`.
- `Experiment.group_labels() -> list[str]`.
- `Experiment.iter_group_shards() -> list[GroupShard]` — each `GroupShard` has
  `.shard_index`, `.global_start`, `.global_stop`, `.labels`, and `.to_anndata()`
  (deferred per-shard I/O).

The same grouped-read methods are available on `open_cloud(...)` (over range
reads) and via `rscx`. The sidecar is dropped by `append` (and not propagated by
`compact` / `merge` / `subset`); re-sort or re-convert with `--group-by` to
regroup.
