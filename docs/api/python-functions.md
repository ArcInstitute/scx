# Python: module-level functions

> Part of the [SCX API reference](README.md).

## Module-level functions

- `pyscx.open(path, verify=True) -> Experiment` — Open SCX file (local), returning a lazy `Experiment` handle.
- `pyscx.read(path, *, verify=True, **kwargs) -> AnnData` — One-liner read mirroring `sc.read_h5ad`: shorthand for `pyscx.open(path).to_anndata(**kwargs)`. `**kwargs` forward to [`Experiment.to_anndata`](python-experiment.md#experiment) (`backed=`, `var_names=`, `obs_filter=`, `layers=`, `container=`, `data_dtype=`, `index_dtype=`, `allow_lossy=`, …).
- `pyscx.write(adata, path, **kwargs)` — One-liner write mirroring `AnnData.write_h5ad`: shorthand for `pyscx.from_anndata(adata, path, **kwargs)`.
- `pyscx.from_anndata(adata, path, codec=None, shard_size=None, in_place=False, csc=None, csc_cols_per_shard=5000, uns_format="tagged", index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000, bitmap="off", memory_budget=None, force_legacy_metadata=False, sort_by=None, reverse=False, row_group_rows=256, row_group_target_nnz=None)` — Write AnnData to SCX. A float64 `X` is downcast to float32 with a `UserWarning`. `csc=None` is `"auto"`: a CSC sidecar when `n_obs ≥ 50000` and `n_vars ≥ 5000`; `csc="off"` opts out (see `resolve_csc_policy`).
  Persists `X`, `obs`, `var`, `layers`, `obsm`, `varm`, `uns`, and the sparse
  pairwise slots `obsp` / `varp`. Pairwise matrices are stored as float32 COO
  Arrow IPC; higher-precision inputs are downcast on write. `in_place=True`
  permits sorting the caller's CSR indices in place (avoids a copy when `X` is
  an unsorted scipy CSR); leave it `False` (default) to keep the input AnnData
  untouched. `uns_format`
  selects how `adata.uns` is serialized — see [`uns` serialization](python-experiment.md#uns-serialization).
  Accepts backed AnnData (`sc.read_h5ad(path, backed='r')`) and auto-routes
  to the streaming converter — see `pyscx.from_h5ad` below for the
  underlying mechanics. `index_*` / `bitmap` materialise query
  predicate indexes and detection bitmaps at conversion time — see
  [Conversion-time predicate indexes and detection bitmaps](indexes.md#conversion-time-predicate-indexes-and-detection-bitmaps).
  `force_legacy_metadata=True` forces a single `ObsMetadata` /
  `VarMetadata` section regardless of size; the default (`False`)
  emits `ObsMetadataShard` / `VarMetadataShard` sections when
  `n_obs > shard_size`. `memory_budget` (`"4G"`, `"512M"`,
  bytes) emits `MappingPeakFootprintHigh` when an individual mapping's
  estimated footprint exceeds the budget, and — when a CSC sidecar is
  built — is passed to the sidecar transpose (capped at its 4 GiB
  default), so it also changes the CSC shard count; a budget too small for
  one column chunk now **raises** (`RuntimeError: CSC transpose failed:
  memory limit too small…`) where it previously succeeded. `shard_size` sets the per-shard
  row count for both `X` and the obs/var metadata shards. Obsm, varm,
  obsp, and varp are extracted and written one key at a time (incremental,
  not collected).
- `pyscx.from_h5ad(path, out, codec=None, shard_size=None, shard_obs="auto", csc=None, csc_cols_per_shard=5000, uns_format="tagged", stream=True, strict_uns=False, dense_zero_epsilon=0.0, memory_budget=None, temp_dir=None, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000, bitmap="off", reader_threads=None, writer_queue_depth=4, sort_by=None, reverse=False, group_by=None, reference=None, group_target_bytes=None, group_max_bytes=None, group_pass="auto", obs_override=None, var_override=None, uns_override=None, row_group_rows=256, row_group_target_nnz=None)` — Stream an h5ad file directly to SCX without materialising `X` in Python or Rust. `csc=None` is `"auto"` (a sidecar when `n_obs ≥ 50000` and `n_vars ≥ 5000`; `csc="off"` opts out). `shard_obs` (`"off"`/`"auto"`/`"always"`, default `"auto"`) writes obs as row-sharded `ObsMetadataShard` sections when `n_obs > shard_size` — the same tri-state and threshold as `scx optimize --shard-obs` and `pyscx.from_anndata`; **obs axis only**, var is always a single section on ingest. It is a storage-layout choice, not a memory one: obs is read whole either way — though each string column's payload is copied **once**, straight from the HDF5 read into the Arrow value buffer at its exact size, rather than through a `Vec<String>` and a `Vec<&str>` first.
  Bounded peak memory: `shard_target_rows × n_vars × density × ~16` bytes
  per X shard, plus `shard_target_rows × k × 4` bytes per `obsm` / `varm` /
  `obsp` / `varp` matrix (each is now hyperslab-read and emitted as
  row-sharded sections — see [§ Sharded obsm/varm/obsp/varp in the
  format spec](../format.md#sharded-layout-section-types-2023)), plus the
  always-resident `indptr` (`(n_obs + 1) × 8` bytes). Recommended
  entry point for files larger than RAM. `csc="always"` builds the
  sidecar in the same pass as X (each streamed CSR shard is pushed into a
  `CscBuilder` and the CSC shards are emitted right after X — no extra read
  pass, no second copy on disk; under `memory_budget` the builder takes a
  quarter of the budget and ingest sizes itself against the rest). `uns_format` is a no-op for the on-disk `uns` read but
  controls the envelope shape applied to `uns_override` when supplied
  (`"tagged"` default wraps NumPy / pandas containers in `__scx_type__`
  envelopes for bit-exact round-trip; `"plain"` collapses them to JSON
  primitives). Internally bypasses
  `anndata.read_h5ad` entirely — obs/var/uns are read via pure-Rust
  HDF5, so callers that don't supply overrides also dodge the eager
  `obsm` materialisation that `anndata.read_h5ad(path, backed='r')`
  performs (anndata reads `obsm` into Python heap on every call,
  including in backed mode).
  - `obs_override`, `var_override`, `uns_override` (optional): supply a
    pandas DataFrame (obs/var) or Python dict (uns) to use in place of
    the on-disk values. Intended for read-mutate-write flows where the
    caller wants to add annotations without paying the full `obsm`
    allocation that `anndata.read_h5ad` would trigger. Typically paired
    with `pyscx.read_h5ad_metadata(path)` (below): fetch on-disk obs /
    var / uns cheaply, mutate them, pass them back. `obs_override.shape[0]`
    must equal n_obs on disk; `var_override.shape[0]` must equal n_vars
    on disk. `uns_override` replaces the entire `uns` section (not a
    merge). Any override with `stream=False` raises `ValueError`
    because the non-streaming path does not apply overrides. `obsm` /
    `varm` / `obsp` / `varp` are intentionally not exposed as overrides
    — accepting them would re-introduce the OOM class this API exists
    to avoid; mutate them via `pyscx.from_anndata(backed_adata, ...)`
    instead if needed.
  - Source layout: CSR streams natively. Dense `/X`
    streams via row-slab sparsification — set `dense_zero_epsilon` to
    threshold near-zero values (default `0.0` matches scipy's
    `csr_matrix(dense)`). CSC-on-disk uses an in-memory transpose
    when the file fits `memory_budget`, otherwise an external
    bucketed transpose to `temp_dir` (scipy `sum_duplicates` semantics
    on duplicate coordinates).
  - `strict_uns=True`: raise on the first unrepresentable `uns`
    entry; default `False` emits a `UserWarning` per skipped key
    (`SkippedUnsKey`). Other structured warnings: `InferredEncoding`
    (h5ad encoding-type missing/ambiguous), `DenseSparsified`,
    `DuplicateCoordinatesMerged`. See
    [Conversion warnings](conversion.md#conversion-warnings-convertwarning).
  - `memory_budget`: caps dense slabs, the CSC external-transpose
    buffers, and the CSC **sidecar** transpose chunk (so it changes
    `n_csc_shards` when `csc=` builds one). Accepts an int byte count or
    a binary-prefixed size —
    `K`/`M`/`G`/`T` or `KiB`/`MiB`/`GiB`/`TiB` (powers of 1024); decimal
    `KB`/`MB`/`GB`/`TB` is rejected to avoid 1000-vs-1024 ambiguity
    (see [Memory budgets](memory-budgets.md#memory-budgets)). E.g. `"4G"` / `"512M"` / `"2GiB"`.
  - `stream=False` falls back to the materialising path (kept for
    parity / debugging).
  - `obsm` / `varm` / `obsp` / `varp` on the input are hyperslab-read
    one row-range at a time and emitted as row-sharded sections
    (`<section>/<name>_shard_<idx>`); peak memory per matrix is
    bounded by one shard's worth of rows.
  - `reader_threads`: streaming reader worker count.
    `None` (default) resolves to `RAYON_NUM_THREADS` if set, else
    `os.cpu_count()`. `1` forces the sequential coordinator. `> 1`
    requests rayon workers; output is byte-identical to sequential.
    Requires a thread-safe libhdf5 build (conda-forge default); falls
    back to sequential with a `Hdf5NotThreadsafe` warning emitted at
    most once per process otherwise. `--memory-budget` derates the
    granted count to fit a per-worker estimate; the estimate is
    delegated to the reader: sparse readers assume density 5 % (RNA
    and general) or 10 % (ATAC), times `n_vars × 48 B/nnz`; the
    dense reader sizes its slab at
    `shard_target_rows × n_vars × 44 B/element`. Both figures are the
    **whole worker phase** — the payload or slab, the encoder's own copy
    of the values, and the framed encode's buffers for the two codec
    candidates `codec="auto"` runs concurrently — not the reader stage
    alone. When the dense reader's `memory_budget`-derived slab cap is
    tighter than `shard_target_rows`, the parallel coordinator silently
    clamps its partition to that cap (matching the sequential path).
    The 44 B/element and the quarter-of-the-budget share both come
    from the allocation table described under
    [Memory budgets](memory-budgets.md#memory-budgets); the dense figure does **not**
    scale with the source dtype width, because the resident slab is
    f32 whatever the input was.
    Export is sized separately at 16 B/nnz, since it decodes and never
    runs the encoder.
  - `writer_queue_depth`: backpressure window between the parallel
    encoder pool and the ordered writer. Default 4. The parallel
    coordinator caps outstanding shards (encoding + in channel + in
    reorder buffer) at `reader_threads + writer_queue_depth` via a
    rolling-window spawn, so peak RSS scales with that sum, not with
    the total shard count. Larger values give a slow shard a deeper
    look-ahead buffer; smaller values risk starving encoders when
    one shard takes much longer than its siblings.
- `pyscx.read_h5ad_metadata(path, strict_uns=False) -> H5adMetadata` —
  Read just `obs`, `var`, `uns`, and the X shape from an h5ad file via
  pure-Rust HDF5 readers. Skips `anndata.read_h5ad` (and therefore
  anndata's eager `obsm` allocation) entirely. Returns an
  `H5adMetadata` object with attributes `obs` (`pandas.DataFrame`),
  `var` (`pandas.DataFrame`), `uns` (`dict`), `n_obs` (`int`),
  `n_vars` (`int`), `x_format` (`"csr"` / `"csc"` / `"dense"`). Intended
  for read-mutate-write flows: read this, mutate `obs` / `uns`, pass
  the mutated values back via `pyscx.from_h5ad(..., obs_override=, uns_override=)`.
  Categoricals, pandas Index metadata, and nullable-boolean columns
  round-trip through the same Arrow IPC path that `pyscx.open(...).to_anndata()`
  uses, so the result is semantically equivalent to the obs / var that
  `anndata.read_h5ad` would have returned — without the obsm allocation
  cost. `strict_uns=True` mirrors `from_h5ad`'s strict-uns semantics.
- `pyscx.from_h5mu(path, out, codec=None, shard_size=None, shard_obs="auto", csc=None, csc_cols_per_shard=5000, stream=True, strict_uns=False, memory_budget=None, temp_dir=None, modalities=None, modality_types=None, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000, bitmap="off", reader_threads=None, writer_queue_depth=4, row_group_rows=256)` — Stream an h5mu file to a multimodal SCX v2 file. `csc=None` is `"auto"`, but only the `stream=False` path can honour it: the default streaming path cannot build per-modality sidecars, so it writes CSR-only with a `CscSkippedStreamingMultimodal` warning when a modality would have qualified, and refuses `csc="always"`. Mirrors `from_h5ad` for h5mu inputs; per-modality `n_vars`/`nnz` come from `/mod/{name}/X` attributes so there is no pre-pass materialisation. `reader_threads`/`writer_queue_depth` carry the same semantics as `from_h5ad` — each modality runs through the same dispatcher independently.
  - `modalities`: optional list of modality names to keep
    (case-sensitive). Unknown names raise `ValueError` with the
    available list.
  - `modality_types`: optional dict `{name: "rna" | "protein" | "atac"
    | "spatial" | "methylation" | "custom"}`. Modalities not listed
    fall back to name inference and emit `ModalityTypeInferred`.
- `pyscx.from_10x(h5_path, scx_path, codec=None, shard_size=None, csc=None, csc_cols_per_shard=5000, uns_format="tagged", index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000, bitmap="off", memory_budget=None, force_legacy_metadata=False, row_group_rows=256, row_group_target_nnz=None)` — 10x HDF5 to SCX. `csc=None` is `"auto"` (a sidecar when `n_obs ≥ 50000` and `n_vars ≥ 5000`; `csc="off"` opts out). **Does not stream**: it imports `scanpy.read_10x_h5` and hands the in-memory AnnData to `from_anndata`, so peak memory scales with the whole matrix. The bounded-memory 10x path is `scx convert --from 10x` (streaming by default since OPT-CONVERT-9) — use it for a raw all-droplet `raw_feature_bc_matrix.h5`.
- `pyscx.from_mtx(mtx_dir, scx_path, codec=None, shard_size=None, shard_obs="auto", allow_lossy=False, csc=None, csc_cols_per_shard=5000, memory_budget=None, temp_dir=None)` — `csc=None` is `"auto"`, as on the other single-modality ingest entry points; the sidecar is appended in place after the MTX read, which writes CSR only, and `memory_budget` / `temp_dir` bound and place that build as they do for `from_h5ad`. Cell Ranger MTX directory (`matrix.mtx[.gz]`, `barcodes.tsv[.gz]`, `features.tsv[.gz]`) to SCX. Default shard size is 16384.
- `pyscx.to_mtx(scx_path, output_dir, modality=None)` — SCX to Cell Ranger–style MTX directory (`matrix.mtx.gz`, `barcodes.tsv.gz`, `features.tsv.gz`). Streams one decoded shard at a time, so peak RSS does not scale with the matrix; logically deleted cells are excluded from both the matrix and `barcodes.tsv.gz`. `modality=` is **required** on a multimodal file — an MTX directory holds one matrix over one feature space — and rejected on a single-modality one.
- `pyscx.cellbender_import(path, cellbender_h5, *, layer="cellbender", obs_key=None, var_key=None, prefix="cellbender_", uns_key="cellbender", overwrite=False, on_missing_rows="zero", on_extra_rows="warn", gene_axis="identical", latent_embedding=False, dry_run=False)` —
  Attach a CellBender `remove-background` output to an existing SCX file as a
  layer, **in place**, joined by barcode. Returns a summary dict; inspect
  `n_matched` (or run with `dry_run=True`) before trusting the result. Since it
  writes `var` columns too, the dict also carries the var predicate index
  outcome — `var_index_rebuilt` / `var_index_dropped` /
  `var_columns_not_carried` — on the same terms as `var_import`. See
  [docs/operations.md § CellBender import](../operations.md#cellbender-import).
- `pyscx.is_cellbender_h5(path)` — True when a `.h5` looks like a CellBender
  `remove-background` output rather than a plain 10x CellRanger matrix.
- `pyscx.obs_import(path, table, *, key=None, source_key=None, columns=None, rename=None, prefix="", keep_key_columns=False, delimiter=None, status_column=None, uns_key=None, uns_keys=None, overwrite=False, on_missing_rows="null", on_extra_rows="warn", dry_run=False)` —
  Import a delimited annotation table (CSV/TSV) — or an `.h5ad` whose `/obs`
  holds the columns, on an `hdf5`-feature build — as obs columns on an existing
  file, **in place**. The join is by key string, never by row position; target
  rows the table does not cover get `null`, never a fabricated `0.0`. `path`
  accepts a str, `os.PathLike`, or an open `Experiment`; `key` accepts a str
  (one column), `"obs_names"` (the obs index), or a list (length > 1 builds a
  composite key — the right answer for a multi-library merge where `sample_id` +
  `barcode` is unique but neither is alone). `"obs_names"` resolves on **both** sides, so a table keyed on its own unnamed
  index needs no `source_key`. `source_key` names the **source**
  side's column for each `key` component when the table spells the key
  differently, pairing positionally like pandas `left_on` / `right_on`:
  `key=["sample_id", "obs_names"], source_key=["sample_id", "barcode"]`. Omitted,
  both sides use the `key` names. `on_missing_rows`: `"null"` (default) leaves
  uncovered target rows NULL; `"error"` refuses. `"zero"` is an accepted legacy
  alias for `"null"` — the shared policy's zero is literal only where the
  missing thing is a matrix row (`cellbender_import`, which still spells its
  default `"zero"` and rejects `"null"`), which really is zeros.
  **`overwrite` replaces, it does not merge** — importing several
  per-batch tables in turn keeps only the last. Returns a summary dict
  (`n_obs`, `n_matched`, `n_target_rows_absent`, `n_source_rows_absent`,
  `obs_key_column`, `obs_columns_added`, `obs_index_dropped`, `obs_streamed`,
  and a `key_diagnosis` on failure or `dry_run`); inspect `n_matched`, or run
  with `dry_run=True`, before trusting the result. `obs_streamed` is `False`
  when the target's obs is a legacy single section and had to be assembled
  whole — see [Ops that bound themselves without a budget knob](memory-budgets.md#ops-that-bound-themselves-without-a-budget-knob).
  Undone by `pyscx.rollback`. See
  [docs/operations.md § External obs import](../operations.md#external-obs-import).
- `pyscx.attach_obs_columns(path, df, *, key=None, positional=False, status_column=None, uns=None, uns_key=None, overwrite=False, on_missing_rows="null", on_extra_rows="warn", dry_run=False)` —
  The DataFrame twin of `obs_import`, on the same `attach_external_obs` seam:
  land an in-memory pandas `DataFrame` (or pyarrow `Table`) as obs columns, in
  place, without writing a temp CSV or replacing the whole frame through
  `modify_metadata`. Key-joined by default — `key=None` resolves each side
  independently, exactly as `obs_import` with no `key=` (the source uses `df`'s
  index, named or not, then the barcode-style fallbacks; the target its own obs
  index / fallbacks); a str names one column (matched to the **same name** on
  the target, as rscx's `scx_attach_obs`), a list builds a composite; key
  columns and the pandas index are consumed by the join, not re-imported. `positional=True` (mutually exclusive with `key`)
  skips the join: row `i` annotates obs row `i`, for frames computed
  in-process from this file's own `read_obs()` — never for external tool
  output. It accepts **either row space**, told apart by length: `n_obs` rows
  (`read_obs()`, the live rows — deleted rows are left `null`) or
  `n_obs_physical` rows (`read_obs(logical=False)`, every physical row,
  written as handed in); any other length raises naming both counts. Because
  dispatch is by length, a frame **sorted or reindexed** after `read_obs()`
  would land every value on the wrong cell — so a frame carrying a labelled
  pandas index (every `read_obs()` frame does) is checked: labels that are the
  file's own barcodes in a different order raise naming the first misplaced
  row; labels that are not the file's barcodes are ignored as before, and a
  `RangeIndex` frame is not checked. Under a live-length attach
  `n_matched` is the live count and `n_target_rows_absent` the deleted rows;
  provenance records `row_space` and `positional_index_checked`.
  `status_column` is rejected there. `uns=` lands in the same commit as the columns, so one
  `pyscx.rollback` undoes obs and uns together: alone, it must be a dict and
  its top-level keys are merged into `uns` (several keys per attach); with
  `uns_key="K"` the whole payload nests under `uns["K"]` instead. Untouched
  `uns` keys are left as they were; a colliding key is an error without
  `overwrite=True`, and `uns_key=` without `uns=` is an error. Same policies, summary
  dict and index behaviour as `obs_import` (`obs_key_column` is
  `"<positional>"` under positional; a pure add keeps the predicate index).
  Ungated (no libhdf5). This is `doublet_consensus`'s first-run write path
  (its overwriting re-runs take `modify_metadata` — see below). Categoricals
  survive every in-place obs edit (`attach_obs_columns`, `obs_import`,
  `doublet_import`, `cellbender_import`, `modify_metadata(obs=…)`, rscx
  `scx_attach_obs`): a pandas `category` column — the file's existing ones and
  the one being attached — keeps its dtype, its declared category order, its
  unused levels and its `ordered` bit, exactly as `from_anndata` writes them,
  for string, boolean and numeric levels alike.
  (Before pyscx 0.17 every one of these writers demoted every categorical obs
  column to plain strings; `append` / `merge` / `merge --sort-by` did too until
  this change.) So a `merge` output and an `append` onto a legacy single-section obs
  both read back as `category` with the union vocabulary. `scx sort` matches on
  both of its obs writers: the bounded `--memory-budget` spill path spills each
  categorical's dictionary **codes** against one union vocabulary folded from the
  input's shards, rather than decoding and re-encoding per spilled shard, so its
  obs sections are byte-identical to the in-memory path's — same declared levels,
  same order, same key width, one vocabulary shared by every output shard. (A
  boolean categorical also no longer fails the sort: arrow cannot pack one into a
  dictionary, and nothing packs now.) A dictionary/plain shard mix is still readable — the assembler
  reconciles it — and still reachable, either on a file an older scx version
  grew or by appending a plain-obs source onto a dictionary-encoded base, since
  these ops preserve whichever representation they are handed rather than
  promoting a plain column.
- `pyscx.diagnose_obs_key(path, key=None)` — Read-only. Report which obs columns
  could serve as a join key: `n_obs`, `resolved_key`, `resolved_cardinality`,
  `unique_columns`, `unusable_unique_columns`, `unique_pairs` (two-column
  composites that are unique), `pair_search_capped`, `suggestion`, `summary`.
  Worth running before an import onto a merged atlas, where the obvious
  candidates are often not unique and the one that is may be a column no
  fallback list would guess. Every name reported is one `obs_import(key=...)`
  accepts, including `"obs_names"` for the obs index — the physical
  `__index_level_0__` field is never surfaced, because `read_obs()` hands it back
  as the frame's *unnamed index*. `unique_columns` is ordered
  best-candidate-first (obs index, then `barcode`/`cell_id`-style names, then
  other strings, then integers) and holds only columns that can actually key a
  join; a unique column the join would refuse — a float, whose text form is not
  guaranteed to agree across two independently written sides — is listed
  separately under `unusable_unique_columns` rather than offered.
- `pyscx.var_import(path, table, *, key=None, source_key=None, columns=None, rename=None, prefix="", keep_key_columns=False, delimiter=None, status_column=None, uns_key=None, uns_keys=None, overwrite=False, on_missing_rows="null", on_extra_rows="warn", dry_run=False)` —
  The var-axis twin of `obs_import`: land per-**gene** annotations computed
  elsewhere — a normalised symbol from a reference release, an ATAC peak
  annotation, a curated flag — as `var` columns, **in place**, joined by key
  string and never by row position. Genes the table does not cover get `null`,
  never a fabricated `0.0`. `key=None` auto-resolves each side independently:
  the var index, then `gene_id` / `gene_ids` / `id` / `feature_id` /
  `gene_name` / `gene_symbol` / `name`; `"var_names"` names the var index on
  either side, and a list builds a composite. Same `source_key` / `columns` /
  `rename` / `prefix` / `keep_key_columns` / `delimiter` / `status_column` /
  `uns_key` / `uns_keys` / `overwrite` / `on_missing_rows` / `on_extra_rows` /
  `dry_run` semantics as `obs_import`, including **overwrite replaces, never
  merges**. `X`, layers, `obs`, the CSC sidecar, `.raw`, deletion vectors and
  the *obs* predicate index are all untouched; a sharded var keeps its shard
  boundaries and a single-section var stays one section. Returns a summary dict
  (`n_vars`, `n_matched`, `n_target_rows_absent`, `n_source_rows_absent`,
  `var_key_column`, `var_columns_added`, `var_index_rebuilt`,
  `var_index_dropped`, `var_columns_not_carried`, `var_streamed`, the source's `format` /
  `delimiter` / `n_rows_in_source` / `uns_keys_imported`, and a `key_diagnosis`
  on a dry run). A **multimodal** file is refused — each modality owns its own
  var table. Undone by `pyscx.rollback`. See
  [docs/operations.md § External var import](../operations.md#external-var-import).
- `pyscx.attach_var_columns(path, df, *, key=None, positional=False, status_column=None, uns=None, uns_key=None, overwrite=False, on_missing_rows="null", on_extra_rows="warn", dry_run=False)` —
  The DataFrame twin of `var_import`, on the same `attach_external_var` seam.
  Key-joined by default; `positional=True` (mutually exclusive with `key`)
  lands row `i` on var row `i` and requires exactly `n_vars` rows — var has no
  deletion vector and therefore no second row space, so unlike
  `attach_obs_columns` there is no length-based dispatch and any other length
  raises. A frame carrying a labelled pandas index (every `read_var()` frame
  does) is checked under `positional`: labels that are the file's own gene
  names in a different order raise, naming the first misplaced row, because a
  frame sorted after `read_var()` would otherwise land every value on the wrong
  gene. Categoricals survive with their declared order, unused levels and
  `ordered` bit. `uns=` lands in the same commit as the columns. Ungated (no
  libhdf5).
- `pyscx.diagnose_var_key(path, key=None)` — The var-axis twin of
  `diagnose_obs_key`, returning the same dict with `n_vars` in place of
  `n_obs`. Worth running on a concatenated or merged file, where `var_names` is
  not always unique.
- `pyscx.doublet_import(path, table, *, tool, key=None, source_key=None, key_added=None, score_column=None, call_column=None, call_true=None, call_false=None, keep_native_columns=True, delimiter=None, uns_keys=None, overwrite=False, on_missing_rows="null", on_extra_rows="warn", dry_run=False)` —
  The doublet-caller wrapper over `obs_import`: each tool names its score and
  call differently, and this maps them onto canonical columns so downstream code
  never branches on which tool ran. For `key_added="K"` (defaulting to the tool
  name) it writes `obs["K_score"]` (`float32`), `obs["K_predicted"]` (pandas
  nullable `boolean`), `obs["K_status"]` (`object`, "present"/"absent"),
  `obs["K_<native>"]` for every other source column, and `uns["K"]`. A tool that
  emits no call (`scds`) gets **no**
  `K_predicted` — thresholding a score is a decision the importer does not make
  for you; pass `call_column=` to opt in. `tool="generic"` requires
  `score_column=`. When a profile *does* declare a call column and the table
  carries none of its spellings, the import **warns** and returns
  `call_column_status="declared_but_absent"` plus `expected_call_columns`
  (also recorded in `uns["<K>"]`), rather than silently producing a score-only
  import — the column is still preserved as `<K>_<native>`, so the fix is
  `call_column=` and not re-running the caller. `call_column_status` is
  `"resolved"` / `"not_declared"` (scds — emits no call by design) /
  `"declared_but_absent"`. Pass `keep_native_columns=False` when the source is an h5ad
  exported from the target file, or every original obs column is re-imported
  under the tool prefix.
- `pyscx.doublet_tools()` — The valid `tool=` values, in table order:
  `scdblfinder`, `scrublet`, `doubletfinder`, `doubletdetection`, `solo`,
  `scds`, `generic`.
- `pyscx.doublet_profiles()` — `{tool: {score_columns, score_prefix,
  call_columns, call_prefix, call_tokens, emits_call}}`, read straight from the
  profile definitions. What each `tool=` actually looks for, so a surprising
  import is a REPL lookup rather than a source read. `emits_call` is the
  load-bearing one: `False` (scds, generic) means `<K>_predicted` can never
  appear without `call_column=`. Rendered as a table in
  [docs/scanpy/external-annotations.md § The per-tool column table](../scanpy/external-annotations.md#the-per-tool-column-table),
  which a test pins against this accessor.
- `pyscx.doublet_consensus(target, *, keys=None, method="majority", key_added="doublet", quantile=None, overwrite=False, index_obs=None, index_preset=None)` —
  Combine several callers' imported columns into one consensus. `target` is an
  SCX file (written in place, one commit, `pyscx.rollback`-able) or an in-memory
  `AnnData` (mutated directly). Writes `obs["K_predicted"]` (nullable boolean),
  `obs["K_n_tools_calling"]`, `obs["K_n_tools_voting"]` (both int32),
  `obs["K_score"]` (`mean_rank` only) and `uns["K_consensus"]`, and returns that
  same dict. **Null-aware throughout**: a tool that never saw a cell does not
  vote on it, and a cell nobody voted on comes out `null`, not `False` — read
  `K_n_tools_voting` before trusting a `False`. `method` is `"majority"` (more
  than half the *voting* tools; an even split is `False`, not null), `"any"`,
  `"all"`, or `"mean_rank"` (ignores the calls, rank-normalises each tool's
  scores within the cells it covered and averages; requires an explicit
  `quantile`, since a score cutoff is a scientific decision this helper does not
  own). `keys=None` discovers every key on obs carrying the column the method
  needs, **excluding previous consensus outputs** — a consensus writes the same
  `<K>_predicted` / `<K>_score` columns a caller does, and counting one would
  double-weight whichever callers fed it and inflate `n_tools_voting`. Naming a
  consensus key explicitly in `keys` is still allowed (combining two disjoint
  tool panels is coherent) but warns, and is recorded in the returned
  `keys_that_are_consensus`; `keys_excluded` records what discovery skipped. On
  a file target a **first run** writes through
  `attach_obs_columns(positional=True)` — a pure column add plus a one-key uns
  merge in one commit, so the file's obs predicate index (and every other obs
  column's stats) survives untouched and the rest of `uns` stays
  byte-identical. Passing `index_obs` / `index_preset` — or **overwriting
  existing consensus columns** on a re-run — selects the whole-frame
  `modify_metadata` route instead, the only seam that can rebuild the
  predicate index in the same commit: an index covering a rewritten consensus
  column is rebuilt over the new values, never silently dropped. The `index_*`
  kwargs change the indexed column set; they are not needed to preserve it.
  Pure Python — nothing in it knows what a doublet is.
- `pyscx.export_batches(path, out_dir, *, batch_key, key=None, batches=None, on_ambiguous_key="error", overwrite=False, **kwargs)` —
  Write one h5ad per batch, ready to run a per-sample tool on, without
  materialising the pooled file (peak RSS is one library). `**kwargs` pass
  through to `pyscx.to_h5ad`. What it adds over the loop you would write
  yourself is the key check: a tool sees only the h5ad it is handed, so two
  cells sharing a key inside one batch leave nothing to join its answers back
  on. **Two identities are checked** — the tools read `obs_names`, the import
  joins on the resolved `key`, and those diverge exactly when auto-resolution
  picks a unique non-index column over a duplicated index; both must be unique
  within a batch. `on_ambiguous_key` is `"error"` (refuses before writing
  anything), `"skip"` or `"warn"`. Returns `key`, `key_is_obs_index`,
  `key_is_globally_unique`, `out_dir`, `n_batches`, `n_cells_exported` and a
  per-batch list. Read `key_is_globally_unique` before planning the import: when
  True, concatenate every tool output and import once (which is what you want,
  since `overwrite` replaces rather than merges); when False the keys only
  distinguish cells inside their own batch, so import with a composite key that
  includes `batch_key`.
- `pyscx.to_h5ad(path, out, stream=True, modality=None, reader_threads=None, writer_queue_depth=4, memory_budget=None, obs_mask=None, min_counts=None)` — Stream SCX → h5ad. `obs_mask` is a boolean row mask in either obs row space, told apart by length: `n_obs` entries (the live rows `read_obs()` describes; expanded through the deletion keep mask) or `n_obs_physical` entries (every physical row); any other length raises naming both counts, and an already-deleted row stays dropped either way
  without materialising `X` in memory. Mirror of `pyscx.from_h5ad` in the
  opposite direction. Bounded peak memory: one shard's worth of CSR
  plus encode buffers per matrix written, plus the always-resident
  `indptr` (`(n_obs + 1) × 8` bytes). "Per matrix" covers `X`, every
  `layers` entry, and `adata.raw` — raw goes through the same shard
  walk on its own (usually wider) gene axis. `obsm` / `varm` / `obsp` /
  `varp` are still read whole and are the remaining unbounded term.
  `memory_budget=` is likewise evaluated **per matrix**, so size it for the
  largest one rather than for `X`: raw is captured before HVG subsetting and
  its shards are usually the binding constraint on a `.raw`-bearing file. Raw
  is written last, so a budget that admits `X` but not raw raises only after
  `/X` and the layers are on disk, leaving a partial output. The check runs
  **only on the parallel route** (`reader_threads` > 1) — the budget bounds
  how many shards are in flight, and at one thread there is nothing to
  derate, which is why the refusal offers `--reader-threads 1` as the
  alternative to raising the budget. When deletion vectors are
  present, only kept rows appear in the output (`shape[0] = n_obs -
  n_deleted`); a single pre-scan pass computes the filtered nnz before
  pre-allocating the HDF5 triplet so the on-disk layout is
  deterministic. For multimodal SCX files, pass `modality="rna"` to
  extract a single modality as h5ad; otherwise the call raises (use
  `pyscx.to_h5mu`). `stream=False` falls back to the materialising
  path (kept for parity / debugging).
  - **Categorical obs/var on append-grown files (handled):** a file whose sharded
    obs/var mixes `Dictionary` (original) and plain (appended) representations for a
    categorical column — the layout an `append` produced before this change, and
    still produces from a plain-obs source — exports
    a single h5ad categorical with the full unioned, de-duplicated vocabulary,
    matching `pyscx.open(f).to_anndata()`. This holds for string **and** numeric
    (`Int*`/`Float*`) categoricals, and for both `stream=True` and `stream=False`
    (the fix lives in the shared streaming dataframe writer, which both paths use for
    a sharded axis — there is no separate eager path to fall back to). Category code
    *order* is best-effort and may differ from the read path under deletion vectors;
    per-row values are authoritative.
  - `reader_threads`: parallel shard decoder pool. `None` (default)
    auto-resolves to `RAYON_NUM_THREADS` if set, else
    `os.cpu_count()`. `1` forces the sequential coordinator.
    `> 1` requests rayon workers; output is byte-identical to
    sequential. HDF5 writes stay on the calling thread, so this
    knob does **not** require a thread-safe libhdf5 build (unlike
    the ingest direction).
  - `writer_queue_depth`: bounded reorder buffer depth between the
    parallel decoder pool and the ordered HDF5 writer. Default 4.
    Outstanding decoded shards are capped at `reader_threads +
    writer_queue_depth` so a slow shard 0 can't let the buffer
    accumulate the rest of the file.
  - `memory_budget`: `"4G"`, `"512M"`, `"2GiB"`, or bytes — same
    parser as `from_h5ad`. **Parallel route only** (`reader_threads` > 1);
    at one thread there is nothing to derate, which is why the refusal
    offers `--reader-threads 1`. Evaluated per matrix written (`/X`, each
    layer, `/raw/X`) — size it for the largest, usually raw.
    Derates the granted `reader_threads`
    against `max_shard_bytes` (computed exactly from
    `FullCatalogEntry::stats.nnz` and row count, no density
    heuristic). A single shard exceeding the budget raises with an
    actionable message; smaller mismatches emit
    `ReaderThreadsDerated` and proceed with fewer workers.
- `pyscx.to_h5mu(path, out, stream=True, reader_threads=None, writer_queue_depth=4, memory_budget=None)` — Stream a multimodal SCX
  file to h5mu. Iterates each modality and writes
  `/mod/{name}/X` and any `/mod/{name}/layers/{layer}` shard-by-shard;
  global `/obs`, per-modality `/var` / `/obsm`, and `/uns` reuse the
  in-memory metadata writers. Requires `reader.is_multimodal()`;
  single-modality files raise (use `to_h5ad`). `reader_threads`,
  `writer_queue_depth`, and `memory_budget` carry the same
  semantics as `to_h5ad`; each modality runs through the same
  dispatcher independently.
- `pyscx.iter_chunks(adata, chunk_size="shard")` — Shard-aligned or fixed-size chunk iterator
- `pyscx.preprocess(source, target, ops, target_sum=None)` — Streaming shard-by-shard preprocessing. Preserves obs/var/obsm/uns and carries the deletion-vector section through (the rewrite is 1:1 in obs row space, so deleted cells stay deleted — see [Operations § Deletion vectors](../operations.md#deletion-vectors-carried-vs-applied)); **rejects multimodal and `adata.raw`-bearing inputs** (would otherwise corrupt X / drop raw) — extract a single modality first, or transform before attaching raw.
- `pyscx.save_layer(source, target, layer_name, ops, target_sum=None)` — Save transformed data as a layer. Preserves obs/var/original-X/obsm/uns and the deletion-vector section (pre-existing layers, raw, CSC, varm/obsp/varp, indexes and bitmaps are not carried over); same multimodal/raw rejection as `preprocess`.

## File operations
- `pyscx.append(target, input, codec=None, shard_size=None, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, modality=None)` — Streaming append from SCX file (reads one shard at a time; raw-copy fast path when codec/encoding match). **Append into a multimodal target is deferred** — it raises `ValueError` (`MultimodalUnsupported`); a single-modality append would leave sibling modalities under-covering the shared obs axis. Extract a modality with `scx subset --modality`, append to that single-modality file, then re-merge. Single-modality append is unaffected. `index_*` kwargs rebuild predicate indexes covering all rows post-append — see [Conversion-time predicate indexes and detection bitmaps](indexes.md#conversion-time-predicate-indexes-and-detection-bitmaps). Appending categorical `obs`/`var` onto a **row-sharded** base reassembles the existing metadata through the shared canonical assembler (`scx_format_io::assemble_sharded_metadata`), so disjoint/duplicate per-shard categorical vocabularies and a mixed `Dictionary`/plain-string shard layout are reconciled rather than failing to read back. A genuinely corrupt/gapped obs/var shard cover (non-contiguous `row_start` stamps) is now rejected with a clear error instead of being silently mis-assembled.
- `pyscx.append_from_anndata(target, adata, codec=None, shard_size=None, in_place=False, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, modality=None)` — Append from AnnData. Same `index_*` semantics as `append`. **Append into a multimodal target is deferred** and raises `ValueError` (`MultimodalUnsupported`) — extract → append → re-merge. Single-modality append is unaffected.
- `pyscx.mark_deleted(path, cell_indices)` — Logical deletion
- `pyscx.build_csc(input, output=None, memory_limit=None, force=False, csc_cols_per_shard=5000, temp_dir=None)` — Add a CSC (column-major) sidecar built from input's CSR shards. `output=None` (the default) appends the sidecar to input in place — nothing else in the file is rewritten, no staging copy is made, and `pyscx.rollback(input)` removes it again; pass an output path to write a copy instead. `force=True` applies only to the copy-out form (refuses to overwrite an existing destination unless `force=True`, and is rejected with `output=None`). `memory_limit` (e.g. `"4G"`) caps the builder's bucket staging memory; a limit too small for one decoded source shard is refused naming the required minimum. `temp_dir` is the spill directory when buckets exceed memory; defaults to the output file's own directory. See [sharding.md § CSC sharding](../sharding.md#csc-sharding).
- `pyscx.compact(input, output, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, reshape_obs=False, codec="auto", csc="carry", csc_cols_per_shard=5000, csc_memory_limit=None)` — Rewrite reclaiming space. `csc` (`"carry"`/`"always"`/`"off"`) decides the output's CSC sidecar — `"carry"` builds one in the same pass iff the input had one (see [operations.md § CSC survives mutating operations](../operations.md#csc-survives-mutating-operations)); `csc_cols_per_shard` / `csc_memory_limit` are `build_csc`'s parameters for that build (`csc_memory_limit=None` is 4 GiB with the emit at full speed; naming a limit, `"4G"` included, batches the emit against it — slower, lower peak, same bytes). The same three kwargs, with the same meaning, are on `optimize`, `sort`, `shuffle` and `merge`. `index_*` kwargs rebuild predicate indexes against the compacted output. `reshape_obs=True` migrates legacy single-section obs metadata to the sharded `ObsMetadataShard` layout (mirrors `scx compact --reshape-obs`). A **no-op on already-sharded obs** — which since phase 6c is every `scx convert` / `from_h5ad` / `from_h5mu` / `from_mtx` output above `n_obs > shard_size`, a backed `from_anndata` included. Reach for it on pre-6c files, on `pyscx.from_mudata` output, or after a conversion pinned with `shard_obs="off"` / `force_legacy_metadata=True`.
- `pyscx.optimize(input, output, codec="auto", shard_obs="auto", memory_budget=None, csc="carry", csc_cols_per_shard=5000, csc_memory_limit=None)` — Re-encode + canonicalize every CSR shard (X / layers / obsp graphs). **Unframed: this stamps `format_version=3`, not 4.** It is *not* the full equivalent of `scx optimize`, which frames by default — this binding exposes no framing knob and calls the unframed entry point, so it produces no row-group `BlockIndex` and none of the codec-agnostic sub-shard random access (or framed-Scx1 GPU device decode) that a v4 file carries. Use the CLI (`scx optimize --row-group-rows N`) for framed output. Single-modality files only (multimodal → `RuntimeError`; use `compact`). `codec="auto"` (default) keeps the per-shard codec choice; `codec="scx1"` forces Scx1 on every integer shard (full `to_gpu_anndata` device-decode coverage — framed Scx1 shards decode in VRAM); any other value → `ValueError`. `shard_obs` (`"off"|"auto"|"always"`, default `"auto"`) migrates a legacy single-section obs to the sharded `ObsMetadataShard` layout — `"auto"` shards when `n_obs > shard_target_rows` (the `from_anndata` threshold), `"always"` unconditionally, `"off"` keeps the single section; an already-sharded obs is preserved regardless; an invalid value → `ValueError`. No `force` kwarg — pass `output == input` for an in-place upgrade (atomic rename) or remove the target first. Carries the CSC sidecar under `csc="carry"` (rebuilt from the re-canonicalized X in the same pass). `memory_budget` (a binary-prefixed size string, or `None`) caps what the parallel shard re-encode holds in flight; `None` holds 1 GiB rather than being unbounded, and a small value pins the one-shard-at-a-time behaviour this call had before the re-encode became parallel. See [docs/operations.md § `scx optimize --memory-budget`](../operations.md#scx-optimize---memory-budget) and [docs/operations.md § Optimize](../operations.md#optimize).
- `pyscx.rollback(path, to_seq=None)` — Revert to previous manifest
- `pyscx.set_uns(path, uns)` — Replace the whole `uns` block in place, **without re-encoding `X`** (cost O(uns bytes)). Replace semantics, not merge (see `update_uns`). The CSC sidecar and `data_generation` are preserved. Rollback-able via `pyscx.rollback`. **`set_uns` is a strict subset of `modify_metadata`** — `pyscx.modify_metadata(path, uns=...)` does the same thing and also reaches `obs`/`var`/`obsm`/`varm`; prefer `modify_metadata` unless you only need the one-arg `uns` convenience.
- `pyscx.update_uns(path, uns)` — **Shallow-merge** a dict's top-level keys into the file's `uns` block in place, same cost and guarantees as `set_uns` (O(uns bytes), `X` untouched, one atomic commit, undone by one `pyscx.rollback`). A key that exists is replaced wholesale (nested dicts are not deep-merged; `None` sets `null`, it does not delete); every other key survives untouched; a file with no `uns` yet gets the dict as-is. A non-dict payload is a `ValueError` (a tuple / array has no top-level keys to merge), and so is a file whose existing `uns` is not a JSON object (replace it with `set_uns`). CLI: `scx set-uns <file> --uns patch.json --merge`.
- `pyscx.modify_metadata(path, *, uns=None, obs=None, var=None, obsm=None, varm=None, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, modality=None)` — Replace metadata sections (`uns` / `obs` / `var` / `obsm` / `varm`) in place without touching `X`. `obs`/`var` accept a pandas `DataFrame` (or pyarrow `Table`); `var` must have `n_vars` rows and `obs` either `n_obs` rows (what `read_obs()` returns — the live rows; a deleted row keeps its barcode, every index level paired by position so a renamed or multi-level index is preserved too, and is `null` in every other column — **including columns it had before, since this is a replace**; to add a column while keeping deleted rows' values use `attach_obs_columns(positional=True)`) or `n_obs_physical` rows (what `read_obs(logical=False)` returns, written as handed in); any other length → `ValueError` naming both counts. A live-length frame holding the file's own live barcodes in a different order is refused by row (a sort / reindex after `read_obs()`; renamed barcodes pass). Provenance records `row_space`. `obsm`/`varm` accept `dict[str, np.ndarray]` and are always physical-length (a dense mapping has no `null` for a deleted row). Any omitted arg is left untouched. **A replaced `obs`/`var` keeps the predicate index it had** — rebuilt over the same columns, which also re-derives the per-shard column stats, so `filter_obs` pushdown survives an ordinary obs edit. `index_*` kwargs name a different column set instead, **per axis** (`index_obs` does not change what a same-call var replacement carries; `index_preset` / `index_auto_threshold` span both); a column the file indexed and the new set omits raises a `UserWarning` rather than vanishing silently. Replace semantics, not merge; for a shallow `uns` merge use `pyscx.update_uns`. Only the global modality is supported today (`modality != 0` → error). For a **pure obs column add**, prefer `pyscx.attach_obs_columns` (or `obs_import`): it knows which columns it writes, so the predicate index and untouched columns' stats survive without the rebuild.
- `pyscx.sort(input, output, by, reverse=False, shard_size=None, codec="auto", index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, memory_budget=None, temp_dir=None, bitmap="off", rebuild_csc=None, csc_cols_per_shard=5000, csc_memory_limit=None, group_by=None, reference=None, group_target_bytes=None, group_max_bytes=None, group_write_block_bytes=None, csc=None)` — Globally reorder cells by an obs key for X-read locality and contiguous predicate-index shard ranges. `codec` (`auto`/`none`/`scx1`/`zstd`/`lz4`/`pcodec`/`shufdelta`) pins the output encoding — without it the writer re-selects per shard, so a reorder can change file size for reasons unrelated to the reorder. Pass `memory_budget` (e.g. `"4G"`) to force the bounded external partition sort. Carries the CSC sidecar by default (`csc="carry"`: rebuilt in the same pass iff the input had one; `"always"` / `"off"`); `rebuild_csc=True` is a deprecated alias for `csc="always"` (`False` → `"off"`) that emits a `DeprecationWarning`, and passing both raises `ValueError`. Drops the detection bitmap (`bitmap=` to re-emit); `adata.raw` is not preserved; deletions are materialized away. See [sharding.md § Sorting for read locality](../sharding.md#sorting-for-read-locality-scx-sort).
- `pyscx.shuffle(input, output, seed=42, shard_size=None, codec="auto", index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, memory_budget=None, temp_dir=None, bitmap="off", rebuild_csc=None, csc_cols_per_shard=5000, csc_memory_limit=None, csc=None)` — Globally reorder cells by a **seeded random permutation** (`scx sort --shuffle`), so a training loader gets i.i.d. batches at any `shard_group_size`. Same engine and same carry / drop semantics as `sort`, minus the key arguments — shuffle is an order *source*, not a modifier, so `by` / `reverse` / `group_by` are not offered. `seed` is recorded in provenance and is the only record of the permutation; the same seed on the same input always reproduces the same file. The permutation runs over **live** rows, so a file with deletion vectors shuffles differently from the same file without them. Two consequences worth knowing: the output is the inverse of a sorted file for queries (it maximally scatters predicate-index shard ranges), and a permutation inherently costs some cross-row redundancy for codecs whose compression spans rows — ~6–12% for `zstd`, under 1% for `lz4`/`shufdelta`. **Leave `codec="auto"`**: it runs the same adaptive per-shard selection `scx convert` does. (Earlier releases told you to pin the input's own codec because `auto` grew X 1.86–2.09×. That was a derived-file bug, not a property of shuffling, and is fixed — pinning now selects a specific encoding, it does not hold size. See [sharding.md § Shuffling for training](../sharding.md#shuffling-for-training-scx-sort---shuffle).) The provenance entry's `action` stays `"sort"` — the seed lives at `params.shuffle.seed`, so a consumer looking for `action == "shuffle"` will not find it. See [sharding.md § Shuffling for training](../sharding.md#shuffling-for-training-scx-sort---shuffle).
- `pyscx.merge(inputs, output, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, assume_identical_var=False, assume_identical_obs=False, uns_policy=None, sort_by=None, reverse=False, codec="auto", csc="carry", csc_cols_per_shard=5000, csc_memory_limit=None)` — Merge multiple files. `csc="carry"` builds the output's CSC sidecar in the same pass iff **any** input had one. `index_*` kwargs rebuild predicate indexes against the merged output — without them, pushdown silently regresses to a full obs scan on the merged file. `assume_identical_var` (default `False`) validates var identity (index, column names, values) across all inputs; set `True` to check only `n_vars` (breaking change from pre-branch where var was unchecked). `assume_identical_obs` (default `False`) validates each input's obs schema (column names + dtypes) against input 0; set `True` to skip. `uns_policy` controls conflicting uns sections: `None` / `"first"` (keep first input), `"require-equal"` (error on difference), `"namespace"` (prefix keys with input filename), `"summary"` (keep first input and record a `_scx_uns_conflicts` array). `sort_by` / `reverse` optionally sort the merged obs by a column (a sorted merge refuses `obsm` and COO `obsp` — merge without `sort_by`, then `scx sort`). **An `obsm` / `obsp` key that some inputs carry and others lack is a hard error** (`RuntimeError`) naming the axis, the key and the input — the same answer a missing layer has always had; it used to be a silent drop. An `obsp` key whose `data` column disagrees in dtype or nullability across inputs is refused too, since the merged shards are read back under one schema. File-scope COO `obsp` is carried and rebased into the merged obs space, and `varp` comes from input 0; the **CSR-backed** `obsp` encoding and modality-scoped pairwise graphs are dropped with a warning. Merge streams obs shard-by-shard and builds predicate indexes incrementally from the shard stream.

## Cloud operations (requires `--features cloud`)
- `pyscx.pull(source, dest, filter=None, parallelism=None)` — Streaming cloud → local
- `pyscx.push(source, dest, parallelism=None)` — Streaming local → cloud
- `pyscx.cloud_optimize(input, output=None)` — Front-of-file catalog
- `pyscx.explode(input, output)` — Packed → exploded directory
- `pyscx.pack(input, output)` — Exploded → packed
- `pyscx.open_cloud(url) -> CloudExperiment` — Direct cloud reads.
  `CloudExperiment.query(modality=…)` scopes to one modality of a multimodal
  file (same semantics as the local `Experiment.query`).
- `pyscx.read_cloud(url, *, obs_filter=None, var_names=None, modality=None, data_dtype=None, index_dtype=None, allow_lossy=False) -> AnnData` — One-call cloud read (= `open_cloud(url).query(modality=…)…collect().to_anndata()`); see [docs/cloud.md § `pyscx.read_cloud(...)`](../cloud.md#pyscxread_cloud--one-liner-cloud-read).
