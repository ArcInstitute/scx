# Tokenisation kernels

`pyscx.tokenize` is a set of numeric kernels over a gathered CSR batch. They are
the per-cell steps every transformer-class single-cell model re-implements in
Python: rank the genes, bin the values, crop to a fixed length, sample a
subsequence, normalise. A model's tokeniser becomes a configuration of kernels
rather than a loop, and the loop runs in Rust with the GIL released.

Every kernel takes a whole batch as `(indptr, indices, data)` — the arrays
`SparseCellSetDataset.gather()` and `iter_with_plans()` hand back — and returns
numpy arrays moved, not copied, out of Rust.

```python
import numpy as np
import pyscx
import pyscx.tokenize as tok

ds = pyscx.SparseCellSetDataset(["atlas.scx"])
batch = ds.gather(file_ids, rows, role_tags, set_offsets)

# Geneformer-style rank tokens
ranked = tok.rank_tokens(
    batch["indptr"], batch["indices"], batch["data"],
    gene_stats,                 # per-gene corpus median, indexed by gene id
    l_max=2048,
    vocabulary_version="genecorpus-30M-v2",
)
ids, lengths = ranked["ids"], ranked["lengths"]
```

## The contract

`pyscx.tokenize.CONTRACT_VERSION` pins the kernels' semantics. Assert it at
setup so version skew fails loudly rather than mid-training. It covers, exactly:

1. the numeric semantics of every kernel — ordering, tie rules, bin-edge
   computation, weight transforms, and what each returns;
2. the parameters each kernel *requires*, and the meaning of each;
3. the seed derivation for the RNG-driven kernels.

It does **not** cover array shapes or dict key names; a length-validation error
surfaces those at the first call.

Epoch-dependent randomness stays with the consumer. scGPT's per-epoch masking
and any random crop are the model's business; the kernels here are RNG-free
except where the reference itself samples, and there the draw is keyed on
**content identity** — file identity and physical row — rather than on epoch or
worker index. That is what stops a resumed run, or one with a different worker
count, from silently switching streams.

## The kernels

| Kernel | Consumers | What is part of output identity |
|---|---|---|
| [`top_k`](#top_k) | STATE3, any fixed-length encoder | `k`; the tie rule; mask semantics |
| [`rank_tokens`](#rank_tokens) | Geneformer, C2S, TranscriptFormer-class | the per-gene statistics vector; the vocabulary version; `l_max`; the tie rule |
| [`bin_values`](#bin_values) | scGPT and lineage (Tahoe-x1), CellFM-class | bin count or explicit edges; which values are binned; the edge-tie rule |
| [`sample_genes`](#sample_genes) | UCE-class | `n`; the weight transform; the seed derivation |
| [`transform_values`](#transform_values) | all | `target_sum`, `α`; which PFlog definition |
| [`library_size`](#library_size) / [`measured_mask`](#measured_mask) | all | whether the library size is before or after feature filtering |

### `top_k`

Crops each row to its `k` highest-valued genes, value descending with gene id
ascending on ties. Unfilled slots carry the PAD id (`n_genes_total + 1`); a row
with no positive values gets a single GENE_MASK token (`n_genes_total`) at slot
0. `tok.gene_mask_id(n)` and `tok.pad_id(n)` return the two sentinels.

This is the same kernel `collate_cellset_gathered` uses. The withheld-gene
masking that STATE3 layers on it — drop-to-PAD with no backfill, driven by a
per-set query panel — is reachable only through `collate_cellset_gathered`, not
here: the mask is that consumer's query-panel contract rather than a property of
a top-K crop, and a second copy of its plumbing is how the two would drift.

Note the crop does not get cheaper or dearer when it truncates. Its cost is the
sort over the row's positive entries, which runs whatever `k` is; `k` only
decides how much of the output is PAD.

### `rank_tokens`

Per-cell normalisation by library size and `target_sum`, then a divide by each
gene's corpus statistic, then descending order truncated to `l_max`. Returns
`ids` `[n_rows * l_max]` and `lengths` `[n_rows]`; slots past a row's length are
undefined, because this kernel reports a length and the consumer owns its
padding token.

`gene_stats` is indexed by global gene id — Geneformer's non-zero-median file —
and every entry must be finite and strictly positive, since it is a divisor.
It and `vocabulary_version` are **inputs, never inferred**: "rank order under
statistics S at vocabulary V" is only a reproducible artifact if S and V are
named. The returned `norm_identity` is a 64-bit stamp over both; record it with
the run.

An empty row, an all-zero row, or one whose library size is zero reports length
0 rather than raising — those are ordinary cells, and the reference would divide
by zero on them.

### `bin_values`

Bins each row's expressed values into `[1, n_bins - 1]`, leaving zeros at 0.
`edges=None` recomputes per-cell quantile edges for every row, as scGPT's
`Preprocessor` does; supplying `edges` pins them corpus-wide, in which case they
become part of the tokeniser's identity and must be recorded with it.

Edges are `np.quantile(non_zero, np.linspace(0, 1, n_bins - 1))` with numpy's
default `linear` interpolation, computed in `f64`. That is not a style choice:
`np.quantile` on a `float32` array returns `float64`, so the interpolation
happens on the widened values and `np.digitize` then compares in `f64`.
Computing the edges in `f32` gives different answers.

`tie` decides where a value sitting exactly on an edge lands — see
[Divergences](#divergences-from-the-reference-implementations) below.

### `sample_genes`

Draws `n` genes per row **with replacement**, with probability proportional to
`log1p(count)` renormalised (`weight="log1p"`, UCE's choice) or to the count
itself (`weight="linear"`). Returns `ids` `[n_rows * n]` and `lengths`; a row
with no positive weight reports 0 and leaves its slots untouched.

Pass the file's physical row ids in `rows` and its
`pyscx.downsample_file_identity(path)` so the draw is keyed on content. The
default keys on the row's position in *this* batch, which is reproducible only
for this batch — fine for a smoke test, wrong for training.

### `transform_values`

Applies one of the collator's preprocess modes to a whole batch, returning a new
`data` array: `"pass_through"`, `"log1p_raw"`, `"normalize_log1p"`,
`"pflog_raw"`. `pflog_raw` requires `pflog_alpha` and `n_measured`.

⚠️ SCX's `pflog_raw` is PFlog **v4**: `ln(1 + 4α·c) − centre`, centred by the
measured panel size. It is a different transform from state3 `main`'s
`pflog1ppf_raw`, which is `ln(1 + c / library_size)` centred by the measured-gene
count. The two are distinct named transforms and must not be substituted for one
another — that substitution once shipped under an unchanged contract version.

### `library_size`

Per-row sum with negatives clipped. ⚠️ This is the sum of the row **as given**.
On a panel-projected batch that is the library size *after* feature filtering,
which is not the cell's sequencing depth. A model whose normalisation statistic
was computed against whole-cell depth must not be fed it. The two are different
quantities; name which one you have.

### `measured_mask`

`1` where a panel position names a gene the row carries. "Measured", not
"non-zero": a gene the row does not carry is absent from the CSR, and on a
heterogeneous panel that is not the same claim as a zero count.

## Divergences from the reference implementations

Each kernel's semantics were taken from a pinned revision of the model that
defines it. **Three of those references cannot be reproduced exactly by any
deterministic implementation**, and the divergences are listed here rather than
buried, because a tokeniser off by a tie rule changes training.

### scGPT's binning is stochastic by default

`scgpt/preprocess.py`'s `binning()` calls `_digitize(x, bins)`, whose default is
`side="both"`:

```python
left_digits  = np.digitize(x, bins)
right_digits = np.digitize(x, bins, right=True)
rands = np.random.rand(len(x))
digits = np.ceil(rands * (right_digits - left_digits) + left_digits)
```

A value sitting exactly on an edge lands anywhere between the two bounds,
chosen from numpy's **global** RNG. There is no seed to pass and no stream a
`ChaCha8Rng` can reproduce.

`bin_values` offers the two deterministic bounds the reference interpolates
between — `tie="left"` is `np.digitize(x, bins)`, `tie="right"` is
`np.digitize(x, bins, right=True)` — and `tie="seeded"`, the same randomisation
keyed on `(seed, file_identity, row)` instead. `"seeded"` reproduces the
reference's *distribution*, never its draws, and unlike the reference it is
reproducible across runs and machines.

The strongest checkable claim about the reference is **bracketing**: every draw
it can produce lies in `[right, left]`. The committed golden carries 24 draws
per case from scGPT's own formula and the test asserts SCX's bounds contain all
of them.

The degenerate case is worth naming because it is not rare. A row whose non-zero
values are all equal collapses every quantile edge onto that value, giving
`left = n_bins - 1` and `right = 0` — so the reference assigns a **uniformly
random bin over the whole range**. Measured at `n_bins=51`: `left=[50]`,
`right=[0]`. `tie="left"` pins it at the top bin, which is at least a rule.

### Geneformer's rank order has no tie rule

`rank_genes` is `gene_tokens[np.argsort(-gene_vector)]`. `np.argsort`'s default
`kind` is quicksort — **not stable** — so the order within an equal-value run is
an artefact of introsort's partitioning, reproducible for a given input but
derived from no rule. Measured on numpy 2.4.4, `np.argsort(-v)` over a vector
with a run of equal values returns `[0 5 2 1 3 4 …]` where `kind="stable"`
returns `[0 5 1 2 3 4 …]`.

Ties are not rare there: every gene whose corpus median is 1.0 and whose count
in this cell is 1 normalises to the same value, and single-count genes are the
bulk of a droplet cell.

`rank_tokens` therefore breaks ties by **gene id ascending**, the same rule the
crop uses. Against a golden taken from the reference it agrees exactly on
tie-free cases and agrees up to the order within each equal-value run otherwise.
A consumer that needs Geneformer's exact permutation must use Geneformer.

### UCE samples with replacement, from numpy's RNG

`sample_cell_sentences` (`eval_data.py`) calls
`np.random.choice(np.arange(len(weights)), size=args.sample_size, p=weights, replace=True)`
over `torch.log1p(counts)` renormalised. Two things follow: the draw is **with
replacement**, so a highly expressed gene appears several times in one sentence
by design; and the stream is numpy's global RNG, so no golden can be taken from
it.

`sample_genes` reproduces the algorithm exactly — numpy's `choice` with `p` and
`replace=True` is inverse-CDF sampling (`cdf = p.cumsum(); cdf /= cdf[-1];
cdf.searchsorted(uniform, side="right")`), and this kernel does the same three
steps in `f64` — so for a given sequence of uniforms it selects the same genes.
Its parity test is therefore distributional: empirical frequencies against the
reference's exact normalised weights at large N.

### Negatives and NaN

Every kernel clips with `max(0.0)` first, as the gather stage does. `f32::max`
returns the non-NaN operand, so **NaN becomes zero** — which differs from
`numpy.maximum`, and means a negative value bins to 0 where scGPT's `nonzero()`
would have binned it. On count and log-count data, where all these references
operate, the two agree.

### Exactness range

Counts are stored as `f32`, which is exact for integers below 2²⁴. Library sums
are accumulated in `f64` and are order-independent over that range. Above it,
neither the stored count nor the sum is exact, and no kernel here recovers it.

## Goldens

`scx-loader/tests/data/tokenize/` holds one fixture per kernel, each pinned by a
blake3-64 prefix over its bytes so a regenerated file cannot land silently.
Regenerate with:

```bash
.venv/bin/python benchmarks/scripts/gen_tokenize_goldens.py
```

then update the prefix constants in `scx-loader/src/tokenize/golden_tests.rs`.
The generator is documented, not run in CI; the goldens are committed and the
Rust tests assert them under plain `cargo test`.

⚠️ **What the goldens claim.** `scgpt`, `geneformer` and UCE are installed in no
environment on the development machine, so the generator runs a **verbatim
transcription** of each reference's kernel — quoted in full in the generator
beside the revision it came from — rather than importing the package. That is
weaker than an import, and each file's `_provenance.claim` says so. It is not
SCX agreeing with itself: numpy computes the quantiles, the digitize bounds, the
argsort and the weights in every case.

| file | claim |
|---|---|
| `crop_golden.json` | exact — `np.lexsort` is deterministic |
| `rank_golden.json` | exact within each equal-value run |
| `bin_golden.json` | exact for the edges and both digitize bounds; the randomised form bracketed |
| `sample_reference.json` | distributional only |
| `seeded_golden.json` | SCX's own frozen stream (source of truth is Rust; regenerate with the `#[ignore]`d test) |

To close the gap, create an environment with the real packages and replace each
`_reference_*` function in the generator with the imported one. The
transcriptions are kept in one place so that swap is mechanical.

## Related

- [`docs/training.md`](training.md) — `SparseCellSetDataset`, the native
  collation kernel, and the plan-driven gather these kernels consume.
- [`docs/api.md`](api.md) — the full `pyscx` API reference.
