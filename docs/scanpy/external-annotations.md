# Landing external per-cell annotations (doublet detection)

> Part of the [SCX + scanpy guide](README.md).

Doublet callers are in-memory, single-sample tools that live in four different
ecosystems — scDblFinder and scds in R, Scrublet and DoubletDetection in
scanpy, Solo in scvi-tools. SCX does not reimplement any of them. It gives you
the plumbing to run them where they already live and land the results back on
the file: export per batch, run the tool, import by key.

Nothing in this path is doublet-specific except one lookup table of column
names. `pyscx.obs_import` lands any per-cell annotation table; `doublet_import`
is a thin wrapper that normalises each tool's spellings.

More generally: whenever the thing you computed is **per-cell columns on a
file that already exists** — a batch key for a downstream tool, a QC flag, a
cluster label — reach for the attach seam, not a `from_anndata` rewrite:
`obs_import` for a table on disk, `pyscx.attach_obs_columns(path, df)` for a
DataFrame already in memory (key-joined by default; `positional=True` for
columns you computed row-for-row from this file's own `read_obs()`, where the
obs index may not be unique). Either patches obs in place with no re-encode of
`X` (seconds, not minutes, at atlas scale), preserves the CSC sidecar /
`.raw` / deletion vectors / bitmaps / predicate indexes, and is
`scx rollback`-able. Pipeline runners can call both from restricted
(`no __import__`) `python` steps — see
[docs/api/python.md § Restricted-exec (sandbox) safety](../api/python.md#restricted-exec-sandbox-safety).

## The per-tool column table

This is that lookup table. Read it before writing the import, not after it
surprises you: `--tool` selects which spellings the importer looks for, so a
table whose call column is named something else imports **score-only** and the
tool cannot vote on a call in `doublet_consensus`. (It warns when that happens,
names what it expected, and preserves your column under the `<K>_` prefix — so
the fix is `call_column=`, not re-running the caller.)

| `--tool` | score columns | score prefix | call columns | call prefix | doublet / singlet tokens | emits a call? |
|---|---|---|---|---|---|---|
| `scdblfinder` | `scDblFinder.score` | — | `scDblFinder.class` | — | `doublet` / `singlet` | yes |
| `scrublet` | `doublet_score` | — | `predicted_doublet` | — | `true` / `false` | yes |
| `doubletfinder` | — | `pANN_` | — | `DF.classifications_` | `Doublet` / `Singlet` | yes |
| `doubletdetection` | `doublet_score` | — | `doublet_label` | — | numeric `0`/`1` | yes |
| `solo` | `softmax_score`, `score` | — | `prediction` | — | `doublet` / `singlet` | yes |
| `scds` | `hybrid_score`, `cxds_score`, `bcds_score` | — | *(none)* | — | — | **no** |
| `generic` | *(caller-supplied)* | — | *(caller-supplied)* | — | *(caller-supplied)* | **no** — only via `call_column=` |

Score/call aliases are tried in the order listed, first match wins. A prefix
match must land on **exactly one** column — DoubletFinder run twice with
different `pK` leaves two `pANN_*` columns behind, and picking one silently
would be a coin flip. `generic` requires an explicit `score_column=`.

Rather than trusting this table to stay in sync with the code, read it from
Python: `pyscx.doublet_profiles()` returns the same rows straight from the
profile definitions, and a test asserts the two agree.

## Settle the join key first

This is the step worth doing before anything else, because everything
downstream joins on it. A tool sees only the file you hand it and returns rows
in whatever order it pleased, so the key is the only thing tying its answers
back to your cells.

```python
d = pyscx.diagnose_obs_key("atlas.scx")
print(d["summary"])          # what would be resolved, and whether it is unique
print(d["unique_columns"])   # usable keys that ARE unique, best first
print(d["unique_pairs"])     # two-column composites that are
print(d["unusable_unique_columns"])   # unique, but refused as a key
```

Every name it reports is one you can paste straight into `key=` — including
`obs_names` for the obs index. `unique_columns` is ordered best-candidate-first
(obs index, then `barcode`/`cell_id`-style names, then other strings, then
integers), and it lists only columns that can actually key a join. A unique
column the join would *refuse* is reported separately under
`unusable_unique_columns`: a float score column is often unique per row, but two
independently written sides are not guaranteed to format the same float
identically, so joining on one could silently half-match.

On a single library the obs index is usually unique and there is nothing to
think about. On a merged atlas it often is not: measured on a real
CELLxGENE-derived 1M-cell file, the obs index was a 10×-duplicated stringified
`RangeIndex`, no batch-column composite rescued it, and the only unique column
was `soma_joinid` — a name no fallback list would have guessed. Two ways out:

```python
# A column that is unique file-wide.
pyscx.obs_import("atlas.scx", "calls.csv", key=["soma_joinid"])

# Or a composite: barcodes repeat across libraries but are unique within one.
pyscx.obs_import("atlas.scx", "calls.csv", key=["sample_id", "barcode"])
```

A composite key needs both components on both sides, so the tool's output table
has to carry `sample_id` too. `export_batches` writes the batch column into
each per-batch h5ad for exactly this reason.

The two sides may *name* them differently. `source_key=` gives the source-side
column for each `key` component, pairing positionally like pandas
`left_on` / `right_on` — which is what a merged atlas usually needs, since its
identity is (`sample_id`, obs index) while the tool wrote a `barcode` column:

```python
pyscx.obs_import("atlas.scx", "calls.csv",
                 key=["sample_id", "obs_names"],
                 source_key=["sample_id", "barcode"])
```

`obs_names` is how you name the obs index anywhere a key is accepted — on the
target *and* on the source, so a tool table whose key is its own unnamed index
(what a plain `df.to_csv()` writes) needs no `source_key=`:

```python
pyscx.obs_import("atlas.scx", "scrublet.csv", key="obs_names")
```

It is also the spelling `diagnose_obs_key` reports, so whatever it suggests can
be pasted straight back. The underlying pyarrow field is called
`__index_level_0__`, but `read_obs()` hands it back as the frame's *unnamed*
index, so that name is not something you can address.

## Export one file per batch

```python
r = pyscx.export_batches("atlas.scx", "batches/", batch_key="donor_id")
r["key"]                    # the key that was resolved
r["key_is_globally_unique"] # decides how you import, below
for b in r["batches"]:
    b["path"], b["n_cells"]
```

The pooled matrix is never materialised — peak RSS is one library. The check
this adds over the loop you would write yourself is on the key: two cells
sharing a key *inside one batch* leave the tool's output with nothing to join
on, and that surfaces much later as a duplicate-key error at import time or,
worse, as scores landing on the wrong cell. Both the resolved key and
`obs_names` must be unique within a batch, since the tools read `obs_names`
while the import joins on the key. `on_ambiguous_key="error"` (the default)
refuses before writing anything; `"skip"` omits the batch and records why.

## Run the tool, then import

```python
# scanpy-side, per batch. Take the paths from the result rather than
# reconstructing them — the filenames are sanitised from the batch values.
import scanpy as sc

for b in r["batches"]:
    adata = sc.read_h5ad(b["path"])
    sc.pp.scrublet(adata)
    # The unnamed index column this writes is obs_names, and the import
    # resolves it as the join key.
    adata.obs[["doublet_score", "predicted_doublet"]].to_csv(
        f"calls/{b['batch']}.csv")
```

```r
# R-side — no intermediate file in either direction: read one batch straight
# out of SCX, run the tool, hand the data.frame back.
library(rscx)
library(scDblFinder)

res <- scx_open("atlas.scx") |>
  scx_query() |>
  filter_obs("donor_id == 'A'") |>   # a STRING expression, not NSE
  collect()
sce <- scDblFinder(res$to_sce())

# colData() carries the cell keys as ROWNAMES, not as a column.
df <- as.data.frame(colData(sce)[, c("scDblFinder.score", "scDblFinder.class")])

# One batch at a time would keep only the last, same as on the Python side.
# rbind() the per-batch data.frames and attach once.
scx_attach_obs("atlas.scx", df, key = rownames(df))
```

Then import. **`overwrite` replaces, it never merges**, so how you batch the
import depends on the key:

```python
import pandas as pd

if r["key_is_globally_unique"]:
    # Concatenate every batch's output and import ONCE.
    pd.concat([pd.read_csv(f"calls/{b['batch']}.csv") for b in r["batches"]]
              ).to_csv("all.csv", index=False)
    pyscx.doublet_import("atlas.scx", "all.csv", tool="scrublet")
else:
    # Keys only distinguish cells inside their own batch, so join on a
    # composite — which means the CSV above has to carry those columns too:
    #     adata.obs[["donor_id", "barcode",
    #                "doublet_score", "predicted_doublet"]].to_csv(...)
    # export_batches writes the batch column into each per-batch h5ad so they
    # are there to select.
    for b in r["batches"]:
        pyscx.doublet_import("atlas.scx", f"calls/{b['batch']}.csv",
                             tool="scrublet",
                             key=["donor_id", "barcode"], overwrite=True)
```

Importing several per-batch tables one after another *without* a composite key
keeps only the last — the second import replaces the first's columns rather
than filling in the rows it did not cover.

Cells the tool never saw come back `null`, never `0.0`. That distinction is
load-bearing: it is what lets the consensus step below tell "no tool assessed
this cell" apart from "every tool called it a singlet".

## If the tool wrote back into an h5ad

The scanpy-resident tools mutate `adata.obs` in place, so the h5ad itself is a
valid source — no CSV step:

```python
# One batch's h5ad, written back in place by the tool.
pyscx.doublet_import("atlas.scx", r["batches"][0]["path"], tool="scrublet",
                     keep_native_columns=False)
```

The same batching rule applies here as above: an h5ad holds one batch, so
importing each in turn without a composite key would keep only the last. With a
globally unique key, go through a concatenated CSV instead.

**Pass `keep_native_columns=False` on this route.** An h5ad exported from the
target file carries the *whole* original obs, and the default (`True`)
re-imports every one of those columns under the tool prefix — a real run wrote
32 obs columns (`scrublet_soma_joinid`, `scrublet_tissue`, …) where three were
wanted. Nothing is lost and nothing is wrong, but it is a lot of duplicated
metadata. The CSV route does not have this problem because you choose the
columns when you write the CSV. The h5ad source needs a pyscx built with the
`hdf5` feature; the CSV route does not.

## Combine several callers

Once N tools' results are canonical obs columns on one file, the consensus is
arithmetic:

```python
pyscx.doublet_import("atlas.scx", "scdbl.csv",   tool="scdblfinder")
pyscx.doublet_import("atlas.scx", "scrublet.csv", tool="scrublet")

cons = pyscx.doublet_consensus("atlas.scx", keys=["scdblfinder", "scrublet"])
cons["n_predicted_doublet"], cons["n_no_vote"]
```

This writes `obs["doublet_predicted"]` (pandas nullable `boolean`),
`obs["doublet_n_tools_calling"]` and `obs["doublet_n_tools_voting"]`. Read the
voting count before trusting a `False`: a `0` there means no tool assessed the
cell, which is why `doublet_predicted` is null beside it. `method="majority"`
(default) needs more than half of the *voting* tools; `"any"` / `"all"` are
also available, and `"mean_rank"` ignores the calls and combines the scores,
requiring an explicit `quantile` because a score cutoff is a scientific
decision the helper does not own.

Every one of these ops is in place and undoable — `pyscx.rollback("atlas.scx")`
reverts the last one. `X`, layers, `var`, the CSC sidecar, `.raw` and deletion
vectors are never touched. On a file target a **first** consensus writes
through `pyscx.attach_obs_columns(positional=True)` — a pure column add plus a
one-key uns merge in one commit — so the file's predicate index and the rest
of its obs and `uns` survive byte-identical. A re-run that overwrites existing
consensus columns (and any call passing `index_obs`/`index_preset`) takes the
whole-frame `modify_metadata` route instead, the seam that rebuilds a
predicate index over the rewritten values rather than dropping it. Full
behaviour and the predicate-index interaction:
[docs/operations.md § External obs import](../operations.md#external-obs-import).

## Landing external per-gene annotations

The same seam on the other axis. A step whose output is one value per *gene* —
a normalised symbol resolved against a reference release, an ATAC peak
annotation, a curated gene-set flag — lands with `pyscx.var_import` (a table on
disk, or an `.h5ad`'s `/var`), `pyscx.attach_var_columns` (an in-memory
DataFrame) or `rscx::scx_attach_var`:

```python
import pyscx

# A table a gene-label normaliser wrote, keyed on Ensembl ids.
pyscx.var_import("atlas.scx", "symbols.csv", key="var_names")

# Or a frame computed here, from the file's own var.
exp = pyscx.open("atlas.scx")
v = exp.read_var()
v["is_hvg"] = v.index.isin(hvg_ids)
pyscx.attach_var_columns(exp, v[["is_hvg"]], key="var_names")
```

Same rules as the obs side: key-joined **by default**, `null` for genes the
source does not cover, `overwrite` replaces rather than merges, `dry_run=True`
previews the join, and one `pyscx.rollback` undoes it. `positional=True` is
there for a frame computed in-process from this file's own `read_var()` —
exactly `n_vars` rows, checked against the file's gene names when the frame
carries them — and never for external tool output, whose row order is its own. `pyscx.diagnose_var_key`
names a usable key when `var_names` is duplicated — which happens on
concatenated files, where symbols repeat and only (symbol, id) is unique.

The alternative — `modify_metadata(var=<whole new frame>)` — still exists and
is the right call when you genuinely mean to replace the table. Full behaviour:
[docs/operations.md § External var import](../operations.md#external-var-import).

## See also

- [Converting existing data to SCX](conversion.md) — `to_h5ad` exports,
  including the CellBender pre-trim.
- [File operations](file-operations.md) — marking doublets deleted once they
  are called.
- [docs/operations.md § External obs import](../operations.md#external-obs-import)
  — the in-place / join / rollback invariants.
