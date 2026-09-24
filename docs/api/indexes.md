# Conversion-time predicate indexes and detection bitmaps

> Part of the [SCX API reference](README.md).

CLI flags `--index-obs`, `--index-var`, `--index-preset`,
`--index-auto-threshold` (plus the equivalent
`index_obs=` / `index_var=` / `index_preset=` /
`index_auto_threshold=` kwargs on `from_anndata` / `from_h5ad` /
`from_10x` / `from_h5mu`) materialise `ObsPredicateIndex` /
`VarPredicateIndex` sections at write time so subsequent
`pyscx.open(...).query()` and `scx pull --filter` calls can push
predicates down without an obs scan.

The same `index_*` knobs are also accepted by the rewrite ops —
`pyscx.merge` / `pyscx.append` / `pyscx.append_from_anndata` /
`pyscx.compact` and `scx merge` / `scx append` / `scx compact`.
Without them, those ops drop (or, on append, leave stale) the
predicate-index sections — pushdown silently falls back to a full
obs scan. Pass `index_obs=[...]` (or the matching CLI flag) to
rebuild a fresh index covering every row of the output in the same
pass; this is the only supported way to keep pushdown working
across multi-input atlas builds (fan-out per-perturbation
conversions → `pyscx.merge(..., index_obs=[...])`).

Behaviour:

- Force-listed columns (`--index-obs`/`--index-var`) **hard-error** if
  the column is missing or has an unsupported dtype. Unsupported means
  anything that is neither a string nor a number: `Boolean`, and
  equally a boolean-valued pandas `Categorical`
  (`Dictionary(_, Boolean)`), which is a dictionary-encoded `Boolean`
  and no more indexable than the plain column.
- Preset columns warn (`MissingPresetIndexColumn` /
  `UnsupportedIndexColumn`) and are skipped without aborting the
  conversion.
- Auto-indexing picks up categorical-like columns with cardinality
  `≤ index_auto_threshold` (default `1000`).
- **A high-cardinality numeric obs column is indexed, not skipped.** The
  high-cardinality cap applies to categorical columns; the numeric branch
  ignores it deliberately, because a numeric index is **one entry per shard**
  (that column's `[min, max]` over the shard's rows) and so costs the same
  bytes for a million distinct values as for ten. Force-listing
  `--index-obs total_counts` on an atlas is cheap and gives Level-1 shard
  pruning on that column.

  The cap does still apply, unchanged, to **`var`** (`--index-var`), which is
  built by the batch builder. It also applies to obs on the batch builder
  itself (`build_obs_predicate_index_bytes`), where a capped column is
  **dropped from the index entirely** — not merely reported: it is absent from
  `indexed_columns`, absent from the serialised bytes, and gets no Level-1
  pruning. No conversion front end reaches that path, though: `scx convert`
  and `pyscx.from_anndata` both go through
  `build_and_write_conversion_predicate_indexes`, which hands obs to the
  streaming builder with a one-item iterator. So the distinction is real where
  it applies, and not one users of the conversion APIs can observe on obs.
- **A column is classified by what it holds, not by how it is stored.**
  pandas writes every `Categorical` as an Arrow dictionary, whatever the
  categories are, so the dictionary's *value* type decides:
  `pd.Categorical(["A", "B"])` → the categorical index (Level-1
  `CategoryBitset` pruning + Level-2 row-set pushdown), while
  `pd.Categorical([1, 2, 3])` → the **numeric** index, exactly as the
  plain integer column it holds (Level-1 `MinMax` pruning; numeric
  operators are residual at Level 2, so no row-set pushdown). Query
  syntax follows the value type too: `batch == 3`, not `batch == '3'`.
- Multimodal inputs emit `PredicateIndexSkippedMultimodal` and skip
  predicate-index emission entirely — the read path is unimodal-only
  today. The same skip-with-warning applies to multimodal
  `merge` / `append` / `compact`. The reason is that a `shard_id` is resolved to
  "the i-th CSR shard" **by position**, and on a multimodal file each modality's
  shards independently tile `[0, n_obs)`, so the i-th shard is ambiguous.
  Adding multimodal indexing means giving `ShardRange` a modality scope.

  Correctness does **not** rest on the file lacking an index —
  `ScxWriter::write_obs_predicate_index` is public and accepts a multimodal
  writer, so skipping emission is a convention of the high-level writers. Three
  independent guards do the work: the write side
  (`assign_csr_shard_column_stats` counts only `modality_id == 0` entries and
  returns `ColumnStatsShardCountMismatch` rather than mis-assigning), Level-2
  read (`build_plan` forces `obs_predicate_index` to `None` for any
  `modality_id != 0` pipeline), and Level-1 read, which reads per-entry
  `column_stats` and never consults an index `shard_id` at all.
- The flags only build indexes on the SCX-writing ingest directions
  (`h5ad → scx`, `10x → scx`; `h5mu → scx` accepts them and skips with
  the warning above). On any other `scx convert` direction —
  `mtx → scx` and the SCX-export directions `scx → h5ad/h5mu/mtx` —
  passing `--index-*` is a **hard error** rather than a silent no-op,
  since those paths cannot build a predicate index. To (re)build an
  index on an existing SCX file, use `scx compact` / `scx append` /
  `scx merge` or `pyscx.from_anndata`.

Index presets:

| Preset | Expanded obs columns |
| --- | --- |
| `cellxgene` | `cell_type`, `cell_type_ontology_term_id`, `tissue`, `tissue_ontology_term_id`, `disease`, `assay`, `donor_id`, `development_stage`, `sex`, `suspension_type` |
| `perturbseq` | `cell_type`, `donor`, `batch`, `condition`, `perturbation`, `guide_id`, `target_gene`, `control`, `split` |
| `training` | `cell_type`, `donor`, `batch`, `dataset_id`, `split`, `organism`, `tissue` |

`--bitmap off|auto|always` (and the `bitmap=` kwarg) write
`BitmapShard` (section id 6) sidecars carrying per-shard
gene → local-row roaring bitmaps. Auto policy requires sparse X,
`n_vars ≤ 1_000_000`, and an estimated bitmap size ≤ 15 % of the
encoded CSR; ATAC modalities are always-on under `auto`. Consumed by
`Experiment.detection_counts(axis="var", modality=...)` and
`Experiment.cells_expressing(gene, modality=...)`; the backed reader
falls back to a CSR scan when sidecars are absent. The wire format is
specified in [docs/format.md § Detection Bitmap](../format.md#12-detection-bitmap-optional).
