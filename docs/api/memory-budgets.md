# Memory budgets

> Part of the [SCX API reference](README.md).

`MemoryBudget::parse(s)` (`scx-format-io/src/mem.rs`) is the shared parser
behind `--memory-budget` and `build-csc --memory-limit` (CLI) and the
`memory_budget=` kwarg on `from_h5ad` / `from_h5mu`. It caps dense row
slabs in the h5ad streaming reader, CSC external-transpose buffers when
CSC-on-disk exceeds the budget, the **CSC sidecar** transpose chunk (which
therefore affects `n_csc_shards`), and the parallel streaming reader's
worker derate. It lives in `scx-format-io` (re-exported as
`scx_convert::MemoryBudget`) so sibling crates such as `scx-ops` —
which owns `build-csc` — can share it without a dependency cycle.
It parses a byte count and nothing more; what a budget *buys* is the
allocation table below.

Accepted forms:

- bare byte counts (`"1048576"`, `1048576`),
- binary-prefix shorthand `K` / `M` / `G` / `T` (= `KiB` / `MiB` / …),
- explicit binary prefixes `KiB` / `MiB` / `GiB` / `TiB`.

Decimal prefixes (`KB`, `MB`, `GB`, `TB`) are **rejected** to avoid
1000-vs-1024 ambiguity. When the requested budget cannot fit even one
shard's metadata plus one worker, conversion refuses to start with
an actionable error rather than OOMing partway through.

## The allocation table

What a budget *buys* is declared in one place, `scx-convert/src/budget.rs`,
rather than derived at each site. Two things were previously tangled in a
single division and are now separate:

- a **share** — what fraction of the budget one concurrent unit may claim.
  One in-flight shard takes a quarter, which is what leaves room for the
  derate to grant more than one worker.
- a **cost model** — how many bytes that unit actually holds, across the
  worker's *whole* phase rather than one stage. A CSR shard costs 48 B/nnz plus
  the indptr: 8 for the resident payload, 8 for the encoder's own copy of the
  values, and 32 for the framed encode's live buffers (it holds every row
  group's encoded bytes alongside the streams assembled from them, and
  `codec="auto"` runs two candidate codecs concurrently). A dense source
  element costs 44 B: 12 for the f32 slab plus the sparsified indices and
  values at exact capacity, plus the same 32 for the nonzero it may become.

Reservations are declared per *phase*, and only reservations in the same phase
are concurrent — the CSC external transpose claims half the budget for a
column chunk in pass 1 and a quarter for bucket records in pass 2, and those
never coexist. A unit test asserts that each phase's concurrent claims sum to
at most the whole budget.

Seven claims are declared but **not enforced**, and the `enforced` flag is the
difference between "we sized this" and "nothing exceeds this" — read it before
quoting a row as a guarantee:

- the two CSC **bucket** rows are sized from the *mean* nnz per row, so a
  right-skewed sequencing-depth distribution overshoots them;
- the CSC **sidecar** row's budget sizes the transpose chunk, while the writer's
  full-length index and value copies and the encoder's streams are live next to
  it, and the rebuild path additionally retains every source shard. It controls
  column and shard sizing, not a ceiling;
- the three per-shard **ingest / export** rows: on ingest, `encoded <= payload`
  is an estimate rather than a codec guarantee (frames can expand, a codec
  holds its raw, shuffled and compressed planes at once, the encoded indptr is
  priced at zero for an `nnz = 0` shard, the detection bitmap is uncharged, and
  readers on the trait default take a density guess); on export,
  `filter_shard` holds a second indptr and, when masked, doubling-grown output
  buffers alongside the originals;
- the CSC **column-chunk** row's scan always reads the first column whole before
  testing the budget, so one wide column exceeds the share (on a large atlas
  that is an ordinary ubiquitous gene), and its floor exceeds the share for
  budgets under 128 bytes.

They are named in the table so each gap is visible rather than silent, and a
unit test pins the count so an eighth cannot arrive unannounced.

The three per-shard **ingest / export** rows are still on that list, but their
*estimates* changed. Each used to size a reader working set while the worker
also held the encoded shard. The two ingest rows now size the whole worker
phase — 3x larger on sparse, 3.7x on dense — and the export row is sized from
its own decode model rather than borrowing the ingest one. The share did not
change; what widened is the cost model the share is applied to. So a budget now
buys **fewer concurrent workers and smaller shards** rather than the same
concurrency over an unpriced buffer, and a budget too small to hold one whole
phase is refused outright instead of being silently over-committed. Budgets are
opt-in (`memory_budget` defaults to unset), so nothing derates that did not ask
to.

They remain unenforced because a better estimate is not a proof: each row names
the terms it still does not bound (codec frame expansion and intra-codec planes
on ingest, `filter_shard`'s second indptr and doubling-grown buffers on
export).

Three different things are called "no budget", and they are not
interchangeable: an unset `memory_budget` means *no cap at all*; pyscx's
eager-materialisation path warns above 8 GiB; and the CSC sidecar builder
defaults to 4 GiB when no budget is given.

## Ops that bound themselves without a budget knob

Some ops are bounded structurally rather than by a `memory_budget=`, because
there is nothing to trade off — they stream, or they do not.

`pyscx.obs_import` / `attach_obs_columns` / `doublet_import` / `cellbender_import`
(and their `scx` subcommands) rewrite the target's `obs` **one shard at a time** whenever the
target's obs is sharded — anything `from_anndata` wrote above
`shard_target_rows`, and anything `merge` or `append` produced. Peak is one obs
shard plus one row index and one key string per target cell, so landing a
100-cell annotation on a 10M-cell atlas does not cost the atlas's obs table.
The input's obs shard boundaries are preserved rather than re-derived.

A target whose `obs` is a single legacy `ObsMetadata` section has no per-shard
reader, so the whole table is assembled; the op warns and reports
`obs_streamed = false` in its summary. Run `scx optimize` on such a file first
to migrate obs to the sharded layout. The remaining unbounded paths — the key
diagnosis printed on a *failed* join, an `obsm` embedding when one is requested
(`cellbender_import(latent_embedding=True)` is the reachable case), and the
source table itself — are enumerated in
[docs/operations.md § Memory: bounded on the target, resident on the source](../operations.md#memory-bounded-on-the-target-resident-on-the-source).
