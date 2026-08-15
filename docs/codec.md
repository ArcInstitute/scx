# SCX Codec Reference (Bit-Level)

This document is the bit-level specification of the codecs used inside SCX
shards. See [docs/format.md](format.md) for the containing file/shard layout.
The reference implementation lives in the `scx-codec` crate.

**All multi-byte values within codec streams are little-endian.**

The Rust implementation in `scx-codec` is the normative reference — if this
document and the code disagree on encoding/decoding behavior, the code wins.

## 1. Codec IDs

`codec_id` is a 1-byte field present in both the file header (as a default)
and each shard header (as a per-shard override). **Readers MUST use the shard
header's `codec_id`**, not the file header's, when decoding.

| ID | Name | Description |
|----|------|-------------|
| 0 | `none` | Raw little-endian arrays (uint types). Used for the GDS fast path. |
| 1 | `scx1` | Delta-Golomb-Rice indptr + FOR-BP indices + Adaptive Rice values (§2–4). Integer value encodings only. |
| 2 | `zstd` | Zstd applied independently to each of indptr / indices / values. Fallback for float layers. |
| 3 | `lz4shuffle` | Byte-shuffle pre-filter + LZ4 frame compression (§7). Matches the Zarr/Blosc pipeline. |
| 4 | `pcodec` | Pco lossless numerical compression. Optimal for float layers. |
| 5 | `shufdelta` | Byte-shuffle + byte-delta (indices/indptr only) + zstd (§7b). Compact integer counts (~1.5–2.5× smaller than `scx1`); float values take zstd-only (no shuffle/delta). |
| 6–255 | Reserved | Future codecs. |

## 2. Indptr: Delta-Golomb-Rice (codec_id = 1)

**Input**: `n_major + 1` monotonically non-decreasing `uint64` row pointers.

### Encoding

1. Compute deltas `delta[i] = indptr[i+1] - indptr[i]` for `i = 0..n_major-1`.
   Store `indptr[0]` as a raw `uint64` (8 bytes) at the start of the section.
2. Compute Rice parameter `k = max(0, floor(log2(0.6931 × median(delta))))`.
   Write `k` as a single byte after the initial value.
3. For each delta `d`:
   - Quotient: `q = d >> k`
   - Remainder: `r = d & ((1 << k) - 1)`
   - Emit `q` ones followed by one zero (unary coding of quotient)
   - Emit `r` as `k` raw bits (LSB first)

### Decoding

Read the initial `uint64` and the byte `k`. For each delta: reconstruct
`d = (q << k) | r`, then accumulate `indptr[i+1] = indptr[i] + d`.

### Byte alignment

Bitstream is packed LSB-first into bytes. The section is padded to a byte
boundary at the end.

## 3. Indices: Frame-of-Reference + Bit-Packing (FOR-BP, codec_id = 1)

**Input**: `nnz` column indices (uint16 or uint32 depending on `index_dtype`),
sorted within each row.

### Block structure

Indices are encoded in blocks, each covering exactly `B_idx` rows
(default `B_idx = 128`). The last block may be shorter.

```
BLOCK HEADER (variable length):
  block_nnz: u32                         (total indices in this block)
  n_rows_in_block: u16                   (≤ B_idx)
  row_nnz: [varint; n_rows_in_block]     (nnz per row, LEB128)

For each row within the block:
  If row_nnz > 0:
    frame_min: uint16 / uint32           (minimum column index in this row)
    frame_bits: u8                       (bits per delta after frame subtraction)
    deltas: [frame_bits-bit integers; row_nnz]
      where delta[0] = indices[0] - frame_min
            delta[j] = indices[j] - indices[j-1]  for j > 0
      (delta-of-delta within the sorted row, after frame subtraction)
      Packed LSB-first into bytes
```

### `frame_bits` selection

For each row, `frame_bits = ceil(log2(max_delta + 1))` where
`max_delta = max(delta[0..row_nnz])`. If all deltas are 0, `frame_bits = 0`
and no delta bytes are emitted.

### Varint encoding (for `row_nnz`)

Standard LEB128 — each byte uses 7 data bits and a 1-bit continuation flag
(MSB). Values 0–127 are 1 byte.

### SIMD layout (BitPacker4x)

Rows with ≥ 128 non-zeros use `BitPacker4x`, which packs 4 × 32-element blocks
simultaneously for single-pass 128-element decode.

- Performance: 44–45 % faster isolated FOR-BP decode; 10–12 % end-to-end
  `read_full` improvement for Scx1-encoded files.
- Rows with < 128 NNZ use the original sequential bit-packing and remain
  backward-compatible with pre-SIMD readers.
- The SIMD layout is **not** backward-compatible for ≥ 128 NNZ rows. Older
  readers cannot decode those rows.

Implementation: `scx-codec/src/forbp.rs`.

## 4. Values: Adaptive Rice Coding (codec_id = 1)

**Input**: `nnz` non-zero count values (uint8, uint16, or uint32 — selected by
`value_encoding`).

### Block structure

Blocks of `B_val = 256` values. The last block may be shorter.

```
BLOCK HEADER (1 byte):
  rice_k: u4                             (bits 0-3: Rice parameter k, 0..15)
  reserved: u4                           (bits 4-7: zero)

BLOCK BODY:
  For each value v:
    shifted = v - 1                      (values ≥ 1 since they are non-zero)
    q = shifted >> k
    r = shifted & ((1 << k) - 1)
    Emit: q ones, then one zero, then r as k bits (LSB-first)
```

### Rice parameter selection

Per block: compute the sample median of `(v - 1)` for values in the block.
Set `k = max(0, floor(log2(0.6931 × median)))`. This minimizes expected code
length under the geometric distribution approximation of UMI counts.

### Worked examples

| value | k | shifted | q | r | bits emitted |
|-------|---|---------|---|---|--------------|
| 1 | 1 | 0 | 0 | 0 | `0` (1 bit) |
| 2 | 1 | 1 | 0 | 1 | `0 1` (2 bits) |
| 5 | 1 | 4 | 2 | 0 | `1 1 0 0` (4 bits) |

### Byte alignment

Each block's bitstream is padded to a byte boundary.

## 5. SIMD and GPU Decode

Both paths MUST produce bit-identical output to the scalar reference decoder.

### SIMD (AVX2)

The Rice decoder processes 32 values per iteration:

1. Load 256 bits of bitstream into a YMM register.
2. Use `PDEP` / `PEXT` to extract quotients via leading-one counting.
3. Use shift + mask to extract k-bit remainders in parallel.

### GPU (CUDA)

Each warp (32 threads) decodes one block of 256 values:

1. Load the block into shared memory.
2. Thread `t` decodes values at positions `t, t+32, t+64, …`, scanning the
   bitstream with `__ballot_sync()` for unary quotient boundaries.
3. Warp-shuffle to resolve offsets.

Kernel implementations live in `scx-gpu`. See [docs/gpu-setup.md](gpu-setup.md)
for build/runtime requirements.

## 6. Value Encoding (file + shard header)

`value_encoding` is a 1-byte field. Like `codec_id`, each shard header may
override the file default — e.g. raw integer counts in `X` alongside float32
normalized values in a layer.

| ID | Encoding | Bits | Typical use |
|----|----------|------|-------------|
| 0 | `uint8` | 8 | Raw counts (most 10x data) |
| 1 | `uint16` | 16 | High-depth counts |
| 2 | `uint32` | 32 | Very high counts or merged data |
| 3 | `float32` | 32 | Normalized / scaled layers |
| 4 | `float16` | 16 | Compressed normalized layers |

**Scx1 (Rice) applies only to integer encodings (0–2).** Float layers use
Zstd (codec_id = 2) or Pcodec (codec_id = 4) because float values do not
follow the geometric distribution.

## 7. LZ4+Shuffle (codec_id = 3)

Applies a byte-shuffle pre-filter before LZ4 frame compression. Effective on
data with correlated byte patterns across elements (e.g. float arrays where
exponent bytes are similar). Matches Zarr/Blosc compression style — useful
for users migrating from Zarr.

### Encoding (per array — indptr, indices, values independently)

1. **Byte-shuffle**: rearrange bytes by position within each element (all
   byte-0s first, then byte-1s, …).
   - Element width: 8 bytes (indptr u64), 2 / 4 bytes (indices), 1–4 bytes
     (values).
2. **LZ4 frame compress** the shuffled bytes.

### Decoding

LZ4 frame decompress → byte-unshuffle (reverse of encoding).

Each sub-stream is capped at the byte length the shard's declared shape implies
(`indptr_byte_cap` / `checked_len`) *before* decompression, as for `zstd`. LZ4
compresses runs at ratios well past 100:1, so an uncapped frame decode would let
a small shard force an arbitrarily large allocation from untrusted input.

### Characteristics

- ~3.1× compression on float data (vs ~2.9× for Zstd on Smart-seq2)
- ~1.5× faster decompression than Zstd on typical data
- Available via `codec="lz4"` in `from_anndata()` and `scx convert`, and
  **auto-selected for integer ATAC counts** by the per-modality heuristic
  (§13) — the modality-blind heuristic in §8 never picks it.

Implementation: `scx-codec/src/shuffle.rs`, `scx-codec/src/dispatch.rs`.

## 7b. ShufDeltaZstd (codec_id = 5)

The most compact option for **integer
counts** (measured ~1.45–1.75× smaller than `scx1` on raw-count `census_1m` /
`chemogenetic_rgfp`, ~1.3–1.45× smaller than plain `zstd`; the comprehensive
`compression` benchmark independently confirms **1.83× vs `scx1`** / 1.38× vs
`zstd` on `tabula_sapiens_100k`). It is for integer counts only — on
float/log-normalized `X` the value stream falls back to zstd-only and the codec
is ≈`zstd` with no win, which is why it is opt-in / trial-selected and never
auto-forced. Per sub-stream:

> **GPU decode:** Scx1 decodes directly in VRAM via the BitPacker4x / Rice GPU
> kernels, uploading only the tiny indptr (`scx_device_decode_gpu`).
> ShufDeltaZstd **also decodes on the GPU** — the undelta / unshuffle / convert
> transforms run as CUDA kernels — but the default path still runs zstd on the
> CPU and uploads the decompressed plane bytes, so its transfer mode stays
> `scx_device_handoff_streamed` (throughput ~parity-to-faster than Scx1 on GPU).
> The opt-in nvcomp path (`SCX_SHUFDELTA_NVCOMP=1`) decompresses on-device and
> uploads only the compressed frames, reporting `scx_device_decode_gpu`. The **ML
> training loader is CPU-only** for both codecs, so there ShufDeltaZstd's slower
> CPU decode is paid in full. See § "Codec tradeoff summary" below.

- **indices / indptr**: byte-shuffle (transpose to byte planes) → byte-delta
  (per-plane wrapping-`u8`, on the sorted/monotonic streams) → zstd. The delta on
  byte-shuffled sorted indices produces near-constant planes that zstd crushes —
  this is the dominant lever (indices+indptr are ~75–80 % of CSR payload).
- **integer values**: byte-shuffle → zstd (**no delta** — counts are effectively
  random; delta would raise entropy).
- **float values**: zstd only (**no shuffle, no delta** — byte-shuffling floats
  scatters whole-value repeats).

Primitives: `scx-codec/src/byte_delta.rs` (`byte_delta_planes` /
`byte_undelta_planes`), reusing `shuffle::byte_shuffle`. Bounded-allocation zstd
decode guard as for `zstd`. Available via `codec="shufdelta"`; not auto-selected
by the heuristic (see `compact-trial` in §8).

### Row-group framing (v4 file / shard v2) — random-access-safe

A monolithic `shufdelta` frame per sub-stream is not sub-shard-seekable (the
byte-delta is a whole-plane prefix scan), which would defeat scattered/grouped
reads. So `shufdelta` (and any codec) can be **row-group-framed**: the shard's
major axis is partitioned into groups (`--row-group-rows N`, optionally
`--row-group-target-nnz`), each group encoded independently as a standalone
sub-shard (group-local `indptr` starting at 0), the three sub-streams
concatenated, and per-group byte offsets recorded in the shard's multi-entry
`BlockIndex` (see `docs/format.md`). A framed shard is `shard_format_version = 2`
inside a `format_version = 4` file; unframed writes stay v3/v1. Readers decode a
single group via `scx_codec::decode_row_group` → the ordinary per-shard decoder
over that group's byte ranges (codec-agnostic), so random-row / grouped / backed
reads touch only the covering groups. **Measured:** framing at `G ∈ {256,512,
1024}` retains essentially the full monolithic win (size flat within ~0.2 % across
`G`; 1.75×/1.46× vs Scx1 on census_1m/rgfp), so no per-shard dictionary is needed.

Implementation: `scx-codec/src/{byte_delta.rs,dispatch.rs}` (codec +
`decode_row_group`), `scx-format-io/src/encoder.rs` (`encode_shard_framed`),
`scx-format/src/shard.rs` (`resolve_block_index`).

**Loader adoption (training / scattered reads).** The backed reader's scattered
gather (`BackedCsrReader::read_rows_with`) decodes only the row-groups a
scattered request touches via the `BlockIndex` — the codec-agnostic random-access
path for all framed shards. A framed shard is block-index eligible
(`BackedCsrReader::block_index_eligible`) — cost-eligible + framed — so a framed
training file gets random-access decode without any per-row sidecar. The
`IndexPlanDataset` / `SparseCellSetDataset` prefetchers skip pre-warming
block-index-eligible framed shards so the gather reaches the group-level path
(env kill-switch `SCX_SCATTER_BLOCK_INDEX=0`; per-dataset opt-out
`scatter_block_index=False` on `IndexPlanDataset`). Adoption is observable via
`IndexPlanDataset.cache_metrics()["block_index_groups"]` (`> 0` ⇒ the framed path
was taken; `full_shard_groups` is the fallback route). The `read_scattered`
comprehensive benchmark drives an unsorted scattered gather over a framed
compact-trial file and gates `block_index_groups ≥ 1`,
`block_index_adoption_rate == 1.0`, and (on the multi-shard datasets)
`full_shard_groups == 0` — a direct "no silent full-shard fallback" floor. Those
floors run against **both** the shipped write default **G=256** (`scx_compact_trial_g256`
— what a plain `codec="auto"` write frames at) and the historical compact-trial
fixture default **G=512**, so a regression that drops framing or unwires the loader
block-index path fails the gate at the default G too.

Opening an `IndexPlanDataset` with `scatter_block_index=True` on an **all-unframed**
(legacy v1) file emits a one-shot `UserWarning`: the block-index fast path cannot fire,
so every batch full-shard-decodes. Reframe with `scx optimize --row-group-rows 256
<file>` (or pass `scatter_block_index=False` to silence).

**Choosing `row_group_rows` (G).** The `compression` + `read_scattered` sweep over
`G ∈ {128, 256, 512, 1024}` (2026-07-04, pbmc3k/pbmc10k/smartseq2/tabula_sapiens_100k):
- **Compression ratio is flat across G** (±0.3% on every dataset) — framing at any G
  keeps the full monolithic win.
- **Scattered-gather latency favors finer G on multi-shard files.** On small,
  few-shard datasets (pbmc3k/pbmc10k) p50 is flat, but on multi-shard datasets a
  broadly-scattered gather lands each row in a distinct row-group, so coarse groups
  over-decode: smartseq2 p50 `G=128` 2.18 s vs `G=512` 3.05 s (1.4×); tabula
  `G=128` 1.70 s vs `G=512` 3.50 s (2.1×). (These are worst-case, no-cache broad
  scatter — not representative of real training throughput, which adds locality +
  caching.)

So finer G (128–256) is strictly better for scattered / training-style reads at
scale — lower decode latency at no compression cost — traded against a larger block
index (≈4× the per-shard entries at 128 vs 512) and slightly less efficient
sequential/full-shard decode.

**Framing is on by default (F5 Phase C).** The convert / `from_anndata` / `from_h5ad`
/ `from_10x` / `scx optimize` write paths now frame at **`DEFAULT_ROW_GROUP_ROWS`
= 256** (`scx-format-io`), the scatter-friendly middle. A plain `codec="auto"` write
frames — framing is codec-agnostic, so this adds **no extra encode cost** (unlike
`compact-trial`, which additionally trial-encodes each shard and stays opt-in for
GPU-integer workloads that want the two-layer Scx1-sidecar preservation). Framed
output is a v4 file (shard v2); pass `row_group_rows = 0` (`--row-group-rows 0`) for
the legacy unframed **v3** layout when targeting older readers. See
[docs/format.md](format.md) `format_version`.

## 8. Automatic Codec Selection

Writers SHOULD choose per-shard codecs automatically. Benchmarks found
that Rice (Scx1) **increases** file size for non-UMI data (Smart-seq2: 0.852×
h5ad vs 0.746× uncompressed), while Zstd achieves 0.345×.

### Heuristic (used by `codec="fast"`, and `auto`'s fallback)

This is the median-based heuristic: it is the whole of `codec="fast"`, the
unframed fallback for `codec="auto"`, and one of the two candidates `auto` /
`compact` dual-encode against ShufDeltaZstd (see § "The codec intent axis"
below). Canonical implementation: `scx-format/src/codec_select.rs::select_codec()`.

- Float32 / Float16 values → **Pcodec** (typical 7–16 % better than Zstd
  on log-normalized data; see `docs/api.md` § "Codec Selection").
- Integer values with `floor(median) ≤ 8` → **Scx1** (Rice).
- Integer values with `floor(median) > 8` → **Zstd**.

Median is computed from a sample of up to 10,000 non-zero values from the
shard — the same calculation used for Rice parameter selection (§4). The
actual codec used is recorded in each shard header's `codec_id`.

### Trial-encode (`codec="compact-trial"`, framed only)

For the most compact **random-access-safe** output, `scx convert --codec
compact-trial --row-group-rows N` encodes each shard row-group-framed with both
the heuristic winner and `shufdelta` and keeps the smaller (recorded per shard in
`codec_id`). It requires `--row-group-rows` (framed output) and optimizes for size
+ random access. (Measured: `compact-trial` matches `shufdelta`'s **1.83× vs Scx1**
on integer-count `tabula_sapiens_100k` and does not regress on float/normalized X —
it picks the heuristic winner there.) Implementation:
`scx-format-io/src/encoder.rs::encode_one_shard`.

**Always frame.** Every framed shard random-accesses via the codec-agnostic
`BlockIndex`, and framed Scx1 shards decode group-by-group **directly in VRAM** —
so there is no separate GPU / per-row representation to preserve. `compact-trial`
therefore always emits a framed shard, picking the per-shard smaller of
{heuristic winner, ShufDeltaZstd}. (The historical two-layer cost model that kept
a Scx1-friendly shard unframed with a decode sidecar — the `--keep-gpu-sidecar`
flag / `SCX_COMPACT_TRIAL_GPU_MARGIN` — was removed with the decode sidecar; see
[format.md § 4.2](format.md#42-decode-metadata-sidecar-removed--section-id-26-reserved).)

**Surfaces.** `scx convert`, `scx optimize` (`--codec compact-trial|shufdelta
--row-group-rows N`, which reports framed / ShufDeltaZstd shard counts), and
`pyscx.from_anndata(..., codec=..., row_group_rows=N)` all produce framed output.
A framed **CSC** sidecar is produced whenever framing is requested alongside CSC — e.g.
`pyscx.from_anndata(csc="always", row_group_rows=N)` or `scx convert --csc
always --row-group-rows N` (the CSC producers thread an explicit `FramingConfig`
rather than relying on writer state; `scx build-csc` / mutating-op `--rebuild-csc`
stay unframed — use `scx optimize` to (re)frame an existing file).

On the read side, a framed CSC sidecar supports per-gene-group scattered reads
(`read_csc_columns` decodes only the touched column-groups via the block index),
so gene-subset DE / aggregation over a wide shard no longer full-decodes it — a
scattered 300-gene `col_sums` measured **~3.2× faster** than the unframed CSC path
(fine groups, `row_group_rows=16`). The win is granularity-dependent: choose
`row_group_rows` for the read pattern — with coarse groups a broadly-scattered
subset touches ~every group, so framing overhead can make it *slower* than a plain
full decode. On GPU, `to_gpu_anndata` decodes **Scx1** shards in-VRAM — unframed via the
BitPacker4x / Rice kernels and framed group-by-group — uploading only the indptr
(`scx_device_decode_gpu`). Framed **ShufDeltaZstd** shards decode via the GPU
transform kernels but upload the decompressed planes
(`scx_device_handoff_streamed`) unless the opt-in nvcomp path
(`SCX_SHUFDELTA_NVCOMP=1`) is enabled, which decompresses on-device
(`scx_device_decode_gpu`).

### The codec intent axis (`auto` / `fast` / `compact`)

`codec=` is a single **intent axis** with three profiles. The default `auto` is
cost-aware adaptive; the other two are its escape hatches.

| `codec=` | Behavior (integer shards) | Float | Framing |
|---|---|---|---|
| `auto` (default) | Per shard, adopt ShufDeltaZstd when it is smaller by at least `ADOPT_MARGIN` (5%); else the heuristic (Scx1 ≤ median 8, Zstd above). Cost-aware: a marginal size win never pays the ShufDeltaZstd decode tax. | Pcodec | Framed default (G=256); **unframed falls back to the heuristic single-encode**. |
| `fast` | Always the heuristic (Scx1/Zstd), single-encode — decode-speed-max. This is the pre-flip `auto`. | Pcodec | Any / none. |
| `compact` | Adopt ShufDeltaZstd on **ties** (`≤` the heuristic) — size-max. | Pcodec | Framed only. |

`auto` and `compact` dual-encode each framed integer shard (heuristic +
ShufDeltaZstd) and pick by size; `fast` single-encodes. The realized codec is
per-shard on disk (`codec_id`), so an `auto` file is typically **mixed**
(predominantly ShufDeltaZstd, with Scx1 on low-median shards). `scx info` reports
the per-shard breakdown (`Codec breakdown: N shufdelta, M scx1`); the resolved
profile is stamped in provenance (`params_json.codec_selection.profile`).
Implementation: `scx-format/src/codec_select.rs::{resolve_codec, pick_codec_v2,
ADOPT_MARGIN}` + `scx-format-io/src/encoder.rs` (the shared dual-encode).

**Why `auto` defaults to size.** Storage + egress is paid on every read forever
and the win is large (1.3–2× smaller on integer counts); the ShufDeltaZstd
CPU-decode tax is small (~6% on realistic `hvg_norm` training, ~4% once a GPU
model sits on top — Phase B SIMD), paid only during active CPU training, and
opt-out-able via `codec="fast"`. The amortized trade favors size for the default
while the latency-critical minority keeps a one-word escape. (Historical note:
this replaces the transitional two-option `auto`/`auto_v2` + `decode_target`
surface, which was removed as a pre-1.0 clean break.)

Explicit codec forces (`none`/`scx1`/`zstd`/`lz4`/`pcodec`/`shufdelta`) and
`compact-trial` (strict-`<` trial) remain available.

### Codec tradeoff summary — Scx1 vs ShufDeltaZstd

The two integer codecs serve different workloads. Summary of measured
tradeoffs (from the comprehensive benchmark suite; full per-dataset
tables in [performance.md](performance.md#scx-vs-shardad--full-feature-parity)):

| Dimension | Scx1 (auto default) | ShufDeltaZstd (compact-trial) |
|---|---|---|
| **Compression (integer)** | Baseline | **1.3–2.1× smaller** on medium/large datasets; slightly larger on tiny (pbmc3k) |
| **CPU decode speed** | Baseline | **≈ parity** (~1.09× slower at a 16K-row shard, at/below Scx1 for smaller shards) since the Phase-B SSE2 byte-transforms; was ~1.3–1.8× slower |
| **GPU decode** | **✅ In-VRAM** (BitPacker4x / Rice kernel; only indptr uploaded) | **✅ On-GPU kernels** (undelta/unshuffle); default uploads planes (`handoff_streamed`), `SCX_SHUFDELTA_NVCOMP=1` = full in-VRAM |
| **Random access (unframed)** | ✅ Per-row (Rice/DGR independently decodable) | ❌ Full-shard (byte-shuffle is global) |
| **Random access (framed)** | ✅ Per row-group (via BlockIndex) | ✅ Per row-group (via BlockIndex) |
| **Encode speed** | ~parity | ~parity (slightly faster — no per-row strategy) |
| **Float data** | N/A — both route to Pcodec | N/A — ShufDeltaZstd falls back to zstd-only |
| **ML training loader** | Marginal edge (loader decodes on **CPU**) | ~9% slower CPU decode/epoch (D0); GPU-loader decode deferred — not the training ceiling |

**When to use which codec:**

| Your workload | Recommended | CLI / Python | Why |
|---|---|---|---|
| GPU ML training | `auto` (Scx1) | `codec="auto"` (default) | Loader decodes on CPU; Scx1's ~9% faster CPU decode (D0) is the marginal default |
| Interactive analysis, CPU | `auto` (Scx1) | `codec="auto"` (default) | Faster CPU decode |
| Storage-constrained archival | `compact-trial` | `--codec compact-trial --row-group-rows 256` / `codec="compact-trial", row_group_rows=256` | 1.3–2.1× smaller; retains random access via framing |
| Cloud hosting (minimize egress) | `compact-trial` | same as above | Smaller = fewer bytes transferred |
| Explicit ShufDeltaZstd (no trial) | `shufdelta` | `--codec shufdelta` / `codec="shufdelta"` | Forces ShufDeltaZstd on all shards (no per-shard trial) |

`compact-trial` trial-encodes each shard with both the heuristic winner and
ShufDeltaZstd and keeps the smaller, so it never regresses vs `auto` on size —
but it **doubles encode time** and produces framed output (shard v2): framed
Scx1 shards that won the trial still decode in-VRAM group-by-group, while
ShufDeltaZstd shards decode via the GPU transform kernels (planes uploaded, or
full in-VRAM under `SCX_SHUFDELTA_NVCOMP=1`). Use `auto` when decode speed or GPU
training throughput matters more than on-disk size.

## 8a. Per-modality Codec Defaults

Multimodal writers (CITE-seq, 10x Multiome, TEA-seq, MuData round-trips
via `pyscx.from_mudata` / `scx convert --from h5mu`) carry a
`ModalityType` tag per modality. The `codec="auto"` resolver routes
through `select_codec_for_modality(raw_values, value_encoding,
modality_type)` (canonical implementation at
`scx-format/src/codec_select.rs::select_codec_for_modality()`), which
specialises by biological modality:

| Modality                    | Integer (uint8/16/32)                            | Float (32/16) |
|-----------------------------|--------------------------------------------------|---------------|
| RNA / Custom / Methylation  | Scx1 (median ≤ 8) else Zstd (delegates to §8)    | Pcodec        |
| Protein (ADT)               | Zstd                                             | Pcodec        |
| ATAC                        | Zstd if sample max ≤ 1 (binary peak) else Lz4Shuffle | Pcodec        |
| Spatial                     | Scx1 (median ≤ 8) else Zstd (delegates to §8)    | Pcodec        |

Rationale:

- **Protein/ADT** counts violate Rice's near-geometric assumption (wider
  dynamic range, lower zero-fraction); Zstd's LZ77 dictionary is the
  better fit even when median is small. Float CLR layers stay on Pcodec.
- **ATAC** uint8 payloads are effectively binary peak-presence in the
  common pipelines; Zstd compresses them an order of magnitude better
  than Lz4Shuffle. Integer peak counts (uint16+, or uint8 with values
  > 1 in the sample) prefer Lz4Shuffle's faster decode at competitive
  ratio. Floats (e.g. TF-IDF normalized peaks) take Pcodec.
- **RNA / Custom / Methylation / Spatial-int** fall through to the
  modality-blind §8 heuristic — their distributions match the Scx1 vs
  Zstd split designed for UMI counts.

Single-modality writers (`pyscx.from_anndata`, `scx convert --from h5ad`,
`scx-mtx`, `rscx`) implicitly default to `ModalityType::Rna`, which
delegates to the §8 heuristic — output is bit-identical to v1 / earlier
files written before per-modality routing. Multimodal-aware ops in `scx-ops` (`scx append`, `compact`,
`merge` working on already-written SCX files) currently still use the
modality-blind `select_codec` **to pick the seed codec**; modality-aware routing
through ops is a follow-on. Note this is now only about the *seed*: those ops do
run the adaptive adoption step on top of it (see the intent axis below), so an
ATAC/Protein shard gets the RNA heuristic as its starting candidate and then
still adopts `ShufDeltaZstd` where it wins — better than before, not yet right.

### Where the intent axis applies

`resolve_codec`'s profiles (`auto` / `fast` / `compact` / `compact-trial`) reach
**every** write path: `scx convert` (X, layers, the CSC sidecar, and multimodal
h5mu X), `scx optimize`, and the derived-file ops `scx compact`, `merge`,
`subset` and `sort`. `auto` therefore means the same adaptive per-shard decision
everywhere.

That was not always true. The adopt step is gated on a framing config carrying a
`decode_target`, and both `FramingConfig::default()` and
`ScxWriter::write_shard_inner` used to omit it — so the derived-file ops, plus
convert's own layer / CSC / h5mu paths, silently ran `fast` while reporting
`auto`. On a `shufdelta` input that flipped every integer shard to the
`Scx1`/`Zstd` heuristic at roughly 2× the bytes per nnz, which is why
`scx compact` could grow a file it was asked to shrink. Regression coverage:
`scx-ops/tests/codec_adaptive.rs`.

Measured after the fix on `census_500k` (863.7 MiB, `shufdelta`): `scx compact`
on a file with 562 MB of orphaned bytes goes 1669.7 MB → 1069.0 MB (0.640×,
previously 1.051×), and `sort --by --codec auto` produces a file byte-identical
to an explicit `--codec shufdelta` pin. A separate, still-open issue accounts for
the residual ~1.3× a rewrite costs on a **mixed-width** file: the ops widen the
value encoding to one file-wide width rather than preserving each shard's — see
[docs/sharding.md § Output size](sharding.md#three-ways-to-sort).

One deliberate exception: a CSC rebuild (`scx build-csc`, and the
`--rebuild-csc` pass of the ops) re-writes each CSR shard *at the codec read off
the source header*, so it passes `decode_target: None` on purpose — re-selection
would defeat the preservation. See `FramingConfig`'s contract docs.

## 9. Limitations & Pitfalls

### Rice assumes near-geometric distributions

The ~2.2 bits/value projection is tuned to typical 10x Chromium UMI counts
(55–65 % ones, near-geometric tail). Performance degrades on:

- **Non-UMI protocols** (Smart-seq2, VASA-seq): wider distributions, heavier
  tails. Rice still helps but the gap vs Zstd narrows.
- **Deeply sequenced data** (>50K UMIs/cell): the per-block adaptive `k`
  handles this, but bits/value will exceed the 2.2 projection.
- **Multimodal data**: CITE-seq ADT counts (higher values, less sparsity)
  and ATAC fragment counts have different distributions. Use per-shard codec
  override to fall back to Zstd where Rice is suboptimal.

### Outlier values can pathologically inflate unary codes

A value of 500 with `k = 1` emits ~250 bits in unary. Rare in typical UMI
data, but merged datasets or high-depth protocols can produce outliers.
Implementations SHOULD cap the unary quotient (e.g. at 15) and emit an escape
code + raw fixed-width value for larger quotients. Not required for
conformance in v1; recommended for robustness.

### Indices dominate compressed size

For typical datasets, FOR-BP indices are ~75–80 % of the compressed CSR
payload; Rice values are ~20 %. Optimizing indices (e.g. PFor-Delta with
outlier patching) has ~4× more impact on file size than further optimizing
values. Future codec versions should prioritize index improvements.

## 10. Implementation Map

| Topic | Source |
|-------|--------|
| Scalar Rice encode/decode | `scx-codec/src/rice.rs` |
| FOR-BP + BitPacker4x | `scx-codec/src/forbp.rs` |
| Delta-Golomb-Rice indptr | `scx-codec/src/delta_golomb.rs` |
| LZ4+Shuffle byte permutation | `scx-codec/src/shuffle.rs` |
| Codec dispatch (`codec_id`) | `scx-codec/src/dispatch.rs` |
| Auto codec selection | `scx-format/src/codec_select.rs` |
| CUDA kernel decoders | `scx-gpu/src/kernels/` |
