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
| 5–255 | Reserved | Future codecs. |

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

### Characteristics

- ~3.1× compression on float data (vs ~2.9× for Zstd on Smart-seq2)
- ~1.5× faster decompression than Zstd on typical data
- Available via `codec="lz4"` in `from_anndata()` and `scx convert`; not
  auto-selected by the heuristic.

Implementation: `scx-codec/src/shuffle.rs`, `scx-codec/src/dispatch.rs`.

## 8. Automatic Codec Selection

Writers SHOULD choose per-shard codecs automatically. Benchmarks found
that Rice (Scx1) **increases** file size for non-UMI data (Smart-seq2: 0.852×
h5ad vs 0.746× uncompressed), while Zstd achieves 0.345×.

### Heuristic (used by `codec="auto"`)

Canonical implementation: `scx-format/src/codec_select.rs::select_codec()`.

- Float32 / Float16 values → **Pcodec** (typical 7–16 % better than Zstd
  on log-normalized data; see `docs/api.md` § "Codec Selection").
- Integer values with `floor(median) ≤ 8` → **Scx1** (Rice).
- Integer values with `floor(median) > 8` → **Zstd**.

Median is computed from a sample of up to 10,000 non-zero values from the
shard — the same calculation used for Rice parameter selection (§4). The
actual codec used is recorded in each shard header's `codec_id`.

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
modality-blind `select_codec`; modality-aware routing through ops is
a follow-on (modality-aware routing through ops is not yet implemented).

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
