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

### Improvements

- `scx_delete()` accepts numeric (double) cell indices, so indices above
  `.Machine$integer.max` (2^31) are addressable (previously truncated to `NA`).
- `scx_delete()` / `select_genes()` reject `NA`, negative, and non-integer
  indices with clear errors.
- Regenerated `man/` + `NAMESPACE` so every exported symbol is documented
  (`R CMD check` no longer reports undocumented objects).
