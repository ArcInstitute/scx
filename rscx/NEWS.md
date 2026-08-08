# rscx NEWS

## Unreleased

### Breaking changes

- **1-based indexing (was 0-based).** `scx_delete(cell_indices)` and
  `select_genes(indices)` now take **1-based** indices, matching R convention and
  the `[` operator (Seurat / SingleCellExperiment are also 1-based). Previously
  these two functions were 0-based.
  - **Migration:** if you previously passed 0-based indices, add 1 —
    `scx_delete(path, idx0)` → `scx_delete(path, idx0 + 1)`; likewise for
    `select_genes()`. Indices derived from `which(...)` / `seq_len(...)` are
    already 1-based and now compose correctly.
  - A `0` index now errors with a clear "must be >= 1 (1-based)" message; other
    previously-0-based indices are silently reinterpreted (shifted by one), so
    audit any hard-coded index vectors.

- **Low-level matrix/graph wrappers are now internal (unexported).** The
  `scx_*_matrix` / `scx_*_graph` / `scx_hvg_*` / `scx_rank_genes` building blocks
  are no longer exported; use the high-level front ends (`scx_pca`,
  `scx_neighbors`, `scx_umap`, `scx_leiden`, `scx_rank_genes_groups`,
  `scx_highly_variable_genes`, `scx_score_genes`, `scx_pseudobulk*`, `scx_nb_glm`)
  instead. `select_genes` and `RGroupShardHandle` are now exported.

- **Multimodal append deferred.** Appending into a modality of a multimodal file
  is not yet supported and now errors (`MultimodalUnsupported`) instead of
  silently corrupting the file. Extract a modality with
  `scx subset --modality NAME`, append to the single-modality file, then re-merge.

### New features

- **`scx_attach_obs()`** — attach a `data.frame` of per-cell annotations to an
  existing SCX file as obs columns, in place. The R end of the doublet-caller
  interop: scDblFinder / DoubletFinder / scds run directly on an rscx-loaded
  object, so there is no intermediate file in either direction. Joins **by key
  string, never by row position** (`key = rownames(df)`, or
  `key_columns = c("sample_id", "barcode")` for a multi-library composite);
  target rows the `data.frame` does not cover become `NA`, never a fabricated
  `0`. X, layers, var, the CSC sidecar, `.raw`, deletion vectors and predicate
  indexes are preserved, and `scx_rollback()` undoes the attach. A `NULL` or
  empty `key` is an error rather than a silent join-on-nothing, since
  `rownames()` returns `NULL` on an object with no names.

### Bug fixes

- **Negative, `NA`/`NaN` and fractional row indices silently returned the wrong
  cells.** `ScxBackedSparse$read_rows()` / `$read_row_indices()` and their
  `ScxLazyTransformed` twins cast R doubles to `u64` with a bare `as`, which
  *saturates* in Rust: `-1` and `NaN` both become `0`, and fractions truncate.
  The bounds checks that followed only rejected values above `n_obs`, so a
  saturated index was always in range. `bs$read_rows(-1, 10)` returned rows
  1–10 and `bs$read_row_indices(c(NA, 3, -2))` returned rows 1, 4, 1 — no error,
  no warning, wrong data. All four now validate before converting. These are
  exported `$`-methods that bypass `[`'s R-side guards, so they were reachable
  with no `[` involved.
- **`scx_compute_lisi(perplexity = NaN)` returned a vector of exactly 1.0**
  instead of erroring. `NaN <= 0` is false, so `NaN` slipped past the
  positivity guard; the derived neighbour count collapsed to 1 and the
  perplexity calibration loop never iterated, because every comparison against
  `NaN` is false. LISI of 1.0 means "perfectly unmixed", so the failure looked
  like a real batch-integration result. `perplexity` must now be finite.
- `cache_shards` is now clamped to `[1, max(<file's shard count>, 128)]` — the
  default is never derated, so the ceiling only binds above 128. A large value
  (`cache_shards = 1e9`) reached `LruCache::new`, which pre-allocates a hash
  table of that capacity — tens of gigabytes, killing the R session with an
  allocation abort rather than a catchable error. Negative, `NaN` and fractional
  values are now rejected instead of silently becoming 1.
- `[` on `ScxBackedSparse` / `ScxLazyTransformed` now rejects non-finite row
  indices and truncates fractional ones, matching `dgCMatrix` (`M[1.9, ]` is
  row 1). The column index was already handed to Matrix, so one call
  could previously have erred on `i` while truncating `j`. Note the deliberate
  asymmetry: `[` truncates, the raw `$read_*` methods are strict.

### Improvements

- `scx_delete()` accepts numeric (double) cell indices, so indices above
  `.Machine$integer.max` (2^31) are addressable (previously truncated to `NA`).
- `scx_delete()` / `select_genes()` reject `NA`, negative, and non-integer
  indices with clear errors.
- Regenerated `man/` + `NAMESPACE` so every exported symbol is documented
  (`R CMD check` no longer reports undocumented objects).
