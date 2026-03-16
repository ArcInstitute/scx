# Phase 1 Implementation Plan: Format + Codec + AnnData Bridge

**Timeline**: Months 1-4
**Goal**: `h5ad → scx convert → scx.open().to_anndata() → scanpy works`
**Spec reference**: [SPEC.md](SPEC.md) v0.5
**Roadmap reference**: [ROADMAP.md](ROADMAP.md) Phase 1 (Sections 1.1–1.6)

### Go/No-Go Gate (from [ROADMAP.md](ROADMAP.md))

Before proceeding to Phase 2, these criteria must be met:

- h5ad → scx → h5ad round-trip is bit-exact for integer counts
- SCX file < 60% the size of h5ad for typical datasets
- `scx.open().to_anndata()` → full scanpy pipeline (QC → PCA → Leiden → DE) works

---

## Repo Setup (Week 1)

### 1. Initialize Cargo Workspace

```
scx/
├── Cargo.toml              # workspace root
├── scx-format/             # file layout, header, catalog, shard I/O
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── error.rs        # ScxError enum (thiserror)
│       ├── header.rs       # FileHeader struct + read/write
│       ├── catalog.rs      # RootCatalog + FullCatalog
│       ├── shard.rs        # ShardHeader + shard read/write
│       ├── section.rs      # section alignment, padding, types
│       ├── writer.rs       # ScxWriter (atomic rename path)
│       ├── reader.rs       # ScxReader (mmap + pread paths)
│       ├── checksum.rs     # BLAKE3 helpers
│       └── provenance.rs   # provenance section read/write (SPEC §3.7)
├── scx-codec/              # compression codecs (SPEC §4)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── rice.rs         # adaptive Rice encoder/decoder (SPEC §4.3)
│       ├── forbp.rs        # FOR-BP encoder/decoder (SPEC §4.2)
│       ├── delta_golomb.rs # Delta-Golomb-Rice encoder/decoder (SPEC §4.1)
│       ├── bitstream.rs    # LSB-first bit reader/writer
│       └── dispatch.rs     # codec_id dispatch + zstd fallback (SPEC §4.5)
├── scx-sparse/             # CSR operations (SPEC §6.1)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── csr.rs          # ScxCsr struct + ops
│       └── convert.rs      # dense ↔ CSR conversion
├── scx-cli/                # command-line tool (ROADMAP §1.4)
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs
│       ├── convert.rs      # h5ad/10x → scx, scx → h5ad
│       ├── info.rs         # scx info
│       └── validate.rs     # scx validate
├── pyscx/                  # Python bindings (ROADMAP §1.5)
│   ├── Cargo.toml
│   ├── pyproject.toml      # maturin config
│   └── src/
│       ├── lib.rs          # PyO3 module root
│       ├── experiment.rs   # PyExperiment (lazy handle)
│       └── anndata.rs      # to_anndata / from_anndata
├── tests/                  # integration tests
│   ├── reference_files/    # known-good .scx files for conformance
│   └── test_data/          # small h5ad/10x files for round-trip
└── benchmarks/             # benchmark scripts + results (ROADMAP §1.6)
    └── scripts/
```

### 2. Workspace Dependencies

```toml
# Cargo.toml (workspace root)
[workspace]
members = ["scx-format", "scx-codec", "scx-sparse", "scx-cli", "pyscx"]
resolver = "2"

[workspace.dependencies]
# Core
blake3 = "1"
arrow = { version = "58", features = ["ipc"] }
zstd = "0.13"
memmap2 = "0.9"
thiserror = "2"
byteorder = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# CLI
clap = { version = "4", features = ["derive"] }
indicatif = "0.17"

# HDF5 — see note below
hdf5 = "0.8"

# Python bindings
pyo3 = { version = "0.23", features = ["extension-module"] }
numpy = "0.23"   # must match pyo3 minor version
```

> **HDF5 crate risk**: The `hdf5` crate (crates.io: `hdf5`, repo:
> `aldanor/hdf5-rust`) has not been released since November 2021. It wraps the
> C HDF5 library via FFI and still works, but receives no bug fixes or Rust
> edition updates. Mitigation: HDF5 is only needed for the conversion path
> (`scx convert`); once data is in `.scx` format, the native reader takes over.
> If the crate breaks, the conversion CLI can shell out to Python's `h5py` as a
> fallback, or a pure-Rust h5ad reader could be written (the h5ad layout is
> relatively simple: CSR arrays + HDF5 groups of per-column datasets).

> **PyO3 version note**: Pin `pyo3` and `numpy` to the same minor version
> (both 0.23.x or both 0.28.x depending on when development starts). PyO3
> evolves rapidly; check https://pyo3.rs/main/migration for API changes. The
> code patterns in this plan use the PyO3 0.22+ **Bound API** (`Bound<'py, T>`)
> which replaced the deprecated `&PyAny` smart pointer.

### 3. Error Types

**Crate**: `scx-format`
**File**: `src/error.rs`

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ScxError {
    #[error("invalid magic bytes: expected SCX\\x01, got {0:?}")]
    InvalidMagic([u8; 4]),

    #[error("unsupported format version: {0} (max supported: 1)")]
    UnsupportedVersion(u16),

    #[error("unsupported endian: {0} (only little-endian 0 is supported)")]
    UnsupportedEndian(u8),

    #[error("checksum mismatch for section '{name}': expected {expected}, got {actual}")]
    ChecksumMismatch { name: String, expected: String, actual: String },

    #[error("invalid shard magic: expected SCXS, got {0:?}")]
    InvalidShardMagic([u8; 4]),

    #[error("unknown codec_id: {0}")]
    UnknownCodec(u8),

    #[error("unknown value_encoding: {0}")]
    UnknownValueEncoding(u8),

    #[error("inconsistent CSR dimensions: indptr len {indptr_len}, expected {expected}")]
    InconsistentCsr { indptr_len: usize, expected: usize },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Arrow(#[from] arrow::error::ArrowError),
}
```

### 4. CI Setup

- GitHub Actions: `cargo test --workspace` on push
- `cargo clippy --workspace -- -D warnings`
- `cargo fmt --check`
- Fuzz targets via `cargo-fuzz` (nightly, on schedule)
- Python tests: `maturin develop && pytest` in pyscx

---

## Step 1: Bitstream Primitives (Week 1)

**Crate**: `scx-codec`
**File**: `src/bitstream.rs`

The bitstream is the foundation for all three codecs. Build and test it first.
All three SCX codecs (SPEC §4.1–4.3) use LSB-first bit packing.

### BitWriter
- Writes bits LSB-first into a `Vec<u8>` buffer
- Methods:
  - `write_bit(bit: bool)`
  - `write_bits(value: u64, n_bits: u8)` — emit lowest `n_bits` of value, LSB first
  - `write_unary(q: u64)` — emit `q` ones followed by one zero
  - `flush() → Vec<u8>` — pad remaining bits in current byte to zero, return buffer
- Internal state: `buffer: Vec<u8>`, current byte accumulator (`u8`), bit position (0-7)

### BitReader
- Reads bits LSB-first from a `&[u8]` slice
- Methods:
  - `read_bit() → bool`
  - `read_bits(n_bits: u8) → u64`
  - `read_unary() → u64` — count ones until zero bit encountered
  - `position() → usize` — current bit offset (for debugging/block boundary alignment)
- Internal state: byte position, bit offset within current byte
- Must handle reading past end of buffer gracefully (return error, not panic)

### Implementation Detail

The "LSB-first" convention means: when writing the value `0b1101` as 4 bits,
the bit at position 0 (value `1`) is written first into the least significant
bit of the current byte, then bit 1 (value `0`), then bit 2 (value `1`), then
bit 3 (value `1`). Reading reverses this process.

### Pitfalls
- **Off-by-one in bit positions.** The most common bitstream bug is writing/reading
  bits in the wrong order. LSB-first means bit 0 of the value goes into the
  least-significant available bit of the current byte. Draw out examples on paper
  before coding. The conformance test vectors are your safety net.
- **Reading past end of buffer.** The `read_bit()` and `read_bits()` methods must
  return a clear error (not panic) when the bitstream is exhausted. Fuzz testing
  will find truncated inputs quickly.

### Tests
- Round-trip: write random bit sequences, read them back
- Edge cases: empty stream, single bit, exactly byte-aligned, 64-bit values
- Unary: encode/decode quotients 0, 1, 2, 100, 1000
- Mixed operations: interleave `write_bits`, `write_unary`, `write_bit`
- Boundary: write 7 bits, then 1 bit (completes a byte), then 3 more bits

---

## Step 2: Rice Codec for Values (Week 2)

**Crate**: `scx-codec`
**File**: `src/rice.rs`
**Spec reference**: [SPEC.md §4.3](SPEC.md) — Adaptive Rice Coding

### Encoder: `rice_encode(values: &[u32], block_size: usize) → Vec<u8>`

Per block of `B_val = 256` values (SPEC §4.3):

1. Shift: `shifted[i] = values[i] - 1` (values are non-zero counts, always ≥1)
2. Compute median of `shifted` values in this block
3. Rice parameter: `k = max(0, floor(log2(0.6931 * median)))`, clamp to 0–15
4. Write block header: 1 byte — `k` in bits 0–3, bits 4–7 reserved (zero)
5. For each shifted value:
   - `q = shifted >> k`
   - `r = shifted & ((1 << k) - 1)`
   - Write unary `q` (q ones then one zero), then write `k` bits of `r` (LSB-first)
6. Pad bitstream to byte boundary (SPEC: "each block's bitstream is padded to byte boundary")

The last block may have fewer than 256 values. The decoder must know the total
value count to determine when to stop (passed as a parameter, not stored in the
Rice stream itself — the shard header's `nnz` field provides this).

### Decoder: `rice_decode(data: &[u8], n_values: usize) → Vec<u32>`

Per block:
1. Read 1-byte header → extract `k` from bits 0–3
2. Determine block size: `min(256, remaining_values)`
3. For each value: read unary → `q`, read `k` bits → `r`
4. Reconstruct: `value = ((q << k) | r) + 1`

### Median Computation

For the Rice parameter formula `k = max(0, floor(log2(0.6931 * median)))`:
- Median of an even-length block: use the lower of the two middle values
  (floor median). This avoids floating-point nondeterminism and is sufficient
  for parameter selection.
- When median is 0 (all shifted values are 0, meaning all raw values are 1):
  `k = 0`. Each value encodes as a single "0" bit (unary of 0).
- The constant `0.6931 ≈ ln(2)` comes from the optimal Rice parameter for a
  geometric distribution with parameter `p` where median ≈ `ln(2)/p`.

### Tests
- Conformance test vectors (create and commit reference vectors):
  - Block of all 1s (shifted=0, k=0): each value = 1 bit (just "0")
  - Block of [1,1,1,2,2,3]: typical UMI distribution, k should be 0
  - Block of [1,2,3,4,5,6,7,8]: uniform-ish, k should be ~2
  - Block with outlier values [1,1,1,500]: tests large quotients (q=249 with k=0)
  - Single value, 256 values, 257 values (partial last block)
- Round-trip: random u32 arrays (values 1–1000), verify bit-exact
- Verify median-based k selection matches SPEC formula
- Verify block byte alignment: each block starts at a byte boundary in the output

### Pitfalls
- **Outlier values cause pathological encoding.** A value of 500 with k=0 emits 499
  unary bits. In typical 10x data this is extremely rare (values 16+ are <1% per
  SPEC §2.1), but it can appear in merged datasets or high-depth protocols. Consider
  adding an escape code (cap unary at 15, emit raw value) for robustness. This is
  not required for v1 conformance but will avoid worst-case decode slowdowns.
- **Median computation must be deterministic.** Use floor median (lower of two middle
  values for even-length blocks) to avoid floating-point nondeterminism. The SPEC
  is explicit about this, but it's easy to accidentally use a library median function
  that returns the mean of the two middle values.
- **The `v - 1` shift means the encoder must reject zero values.** Non-zero counts
  are always ≥ 1 (they're in the CSR data array, which only stores non-zeros). But
  add a debug assertion to catch bugs where a zero sneaks through.

---

## Step 3: Delta-Golomb Codec for Indptr (Week 2)

**Crate**: `scx-codec`
**File**: `src/delta_golomb.rs`
**Spec reference**: [SPEC.md §4.1](SPEC.md) — Indptr: Delta-Golomb-Rice

### Encoder: `delta_golomb_encode(indptr: &[u64]) → Vec<u8>`

SPEC §4.1:

1. Write `indptr[0]` as raw little-endian u64 (8 bytes)
2. Compute deltas: `delta[i] = indptr[i+1] - indptr[i]` for i = 0..n-2
   (n = indptr.len(), so n-1 deltas for n elements)
3. Compute median of deltas (floor median for even length)
4. `k = max(0, floor(log2(0.6931 * median)))`, clamp to 0–15
5. Write `k` as 1 byte
6. For each delta: write unary quotient + k-bit remainder (same as Rice encoding)
7. Pad to byte boundary

Note: Unlike the Rice codec for values, indptr deltas are encoded as a **single
stream** (no 256-value blocks). The parameter `k` is global for the entire
indptr array. This is because indptr is small (n_rows+1 elements per shard,
typically ~10,001) and the deltas are relatively uniform within a shard.

### Decoder: `delta_golomb_decode(data: &[u8], n_rows_plus_one: usize) → Vec<u64>`

1. Read initial u64 (8 bytes, little-endian)
2. Read `k` byte
3. For each of (n_rows_plus_one - 1) deltas:
   - Read unary → `q`, read `k` bits → `r`
   - `delta = (q << k) | r`
4. Reconstruct: prefix sum starting from initial value

### Edge Cases
- Empty shard (n_major = 0): indptr has 1 element, 0 deltas. Output is just the
  initial u64 + the k byte (k=0, no encoded deltas).
- All deltas are 0 (all rows are empty): k=0, each delta is a single "0" bit.

### Tests
- Typical indptr: `[0, 150, 280, 500, ...]` (non-zero counts per row ~100-200)
- Sparse rows: some deltas are 0 (empty rows interspersed)
- Single row (2-element indptr: `[0, N]`)
- Large deltas (dense rows with many non-zeros, e.g., delta > 10000)
- Monotonicity: verify decoded indptr is monotonically non-decreasing
- Round-trip with random monotonic u64 sequences

---

## Step 4: FOR-BP Codec for Indices (Weeks 2-3)

**Crate**: `scx-codec`
**File**: `src/forbp.rs`
**Spec reference**: [SPEC.md §4.2](SPEC.md) — Indices: Frame-of-Reference + Bit-Packing

This is the most complex codec. Implement it carefully.

### Encoder: `forbp_encode(indices: &[u32], row_lengths: &[usize], index_dtype_u16: bool) → Vec<u8>`

SPEC §4.2. Process rows in blocks of `B_idx = 128` rows:

Per block:
1. Write block header:
   - `block_nnz: u32` (total non-zeros across all rows in this block)
   - `n_rows_in_block: u16` (≤ 128; last block may have fewer)
   - `row_nnz: [varint]` for each row — LEB128 encoded (SPEC: "Standard LEB128 —
     each byte uses 7 data bits and 1 continuation bit (MSB). Values 0-127 are
     1 byte.")
2. For each row with nnz > 0:
   - Compute `frame_min = min(row_indices)` (the first index, since sorted)
   - Compute deltas (delta-of-delta within sorted row after frame subtraction):
     - `delta[0] = indices[0] - frame_min` (always 0 by definition)
     - `delta[j] = indices[j] - indices[j-1]` for j > 0 (gap between successive indices)
   - `frame_bits = ceil(log2(max(deltas) + 1))`, or 0 if all deltas are 0
   - Write `frame_min` as u16 or u32 (based on `index_dtype` from shard header,
     SPEC §3.3: "0=u16, 1=u32")
   - Write `frame_bits` as u8
   - If `frame_bits > 0`: write each delta as `frame_bits`-bit integer, packed
     LSB-first into bytes

**Note**: `delta[0]` is always 0 (since `frame_min = indices[0]`), so the first
delta wastes `frame_bits` bits of zeros. An optimization would be to skip it, but
the SPEC includes it for simplicity. Follow the SPEC for conformance.

### Decoder: `forbp_decode(data: &[u8], n_rows: usize, index_dtype_u16: bool) → (Vec<u32>, Vec<usize>)`

Returns flat indices array + per-row nnz counts. The caller reconstructs the
CSR structure from these.

Per block:
1. Read `block_nnz: u32`, `n_rows_in_block: u16`
2. Read `row_nnz` varints for each row
3. For each row with nnz > 0:
   - Read `frame_min` (u16 or u32)
   - Read `frame_bits: u8`
   - Read `row_nnz` deltas of `frame_bits` bits each
   - Reconstruct indices: `indices[0] = frame_min + delta[0]`,
     `indices[j] = indices[j-1] + delta[j]`

### LEB128 Helpers
- `write_varint(writer: &mut Vec<u8>, value: u32)` — standard LEB128
- `read_varint(reader: &mut &[u8]) → u32` — standard LEB128

### Tests
- Single row, multiple rows, empty rows (nnz=0, no frame_min/bits emitted)
- Row with single index (frame_bits=0, delta[0]=0, only frame_min written)
- Full block (128 rows), partial last block (e.g., 50 rows)
- u16 indices (n_vars ≤ 65535), u32 indices (n_vars > 65535)
- Dense row (many consecutive indices, e.g., 0,1,2,...,999 → all gaps are 1,
  frame_bits=1)
- Sparse row with large gaps (e.g., indices [0, 10000, 30000])
- Round-trip: random sorted index arrays with known row lengths
- Verify block_nnz matches sum of row_nnz values within each block

### Pitfalls
- **FOR-BP is the most complex codec — budget extra time.** It has variable-length
  block headers (LEB128 varints), per-row frame_min/frame_bits, and the interaction
  between u16/u32 index dtype and frame_min encoding. Implement the simplest version
  first (u16 indices, single block), then add u32 support and multi-block handling.
- **LEB128 encoding/decoding must handle zero correctly.** A row with zero non-zeros
  emits a single varint byte `0x00` and no frame_min/frame_bits/deltas. Make sure
  the decoder doesn't try to read frame_min for empty rows.
- **`delta[0]` is always 0.** Since `frame_min = indices[0]`, the first delta is
  `indices[0] - frame_min = 0`. This wastes `frame_bits` bits per row. The SPEC
  includes it for simplicity — follow it exactly for conformance.
- **Indices dominate compressed size.** FOR-BP indices account for ~75-80% of the
  compressed CSR payload. If compression ratio benchmarks disappoint, focus
  optimization effort here (e.g., PFor-Delta with patching for outlier gaps).

---

## Step 5: Codec Dispatch (Week 3)

**Crate**: `scx-codec`
**File**: `src/dispatch.rs`
**Spec reference**: [SPEC.md §4.5](SPEC.md) — Codec ID and Value Encoding

```rust
/// Codec ID — matches SPEC §4.5 table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CodecId {
    None = 0,   // Raw LE arrays, no compression. For GDS fast-path.
    Scx1 = 1,   // Delta-Golomb indptr + FOR-BP indices + Rice values.
    Zstd = 2,   // Zstd per-section. Fallback for float layers.
}

/// Value encoding — matches SPEC §4.5 table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ValueEncoding {
    Uint8 = 0,    // Raw counts (most 10x data)
    Uint16 = 1,   // High-depth counts
    Uint32 = 2,   // Very high counts or merged data
    Float32 = 3,  // Normalized/scaled layers
    Float16 = 4,  // Compressed normalized layers
}

pub struct EncodedShard {
    pub indptr_bytes: Vec<u8>,
    pub indices_bytes: Vec<u8>,
    pub values_bytes: Vec<u8>,
}

/// Encode a shard's CSR arrays using the specified codec.
///
/// For integer value encodings (0–2), codec_id=1 uses Rice coding.
/// For float value encodings (3–4), codec_id must be 0 or 2 (SPEC §4.5:
/// "Rice coding applies only to integer value encodings").
pub fn encode_shard(
    indptr: &[u64],
    indices: &[u32],
    values: &[u8],           // raw bytes; length = nnz * value_encoding.byte_width()
    codec_id: CodecId,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<EncodedShard, ScxError> {
    // Dispatch based on codec_id
}

/// Decode a shard's CSR arrays.
///
/// Returns raw value bytes (caller interprets based on value_encoding).
pub fn decode_shard(
    encoded: &EncodedShard,
    codec_id: CodecId,
    value_encoding: ValueEncoding,
    n_rows: usize,
    nnz: usize,
    index_dtype_u16: bool,
) -> Result<(Vec<u64>, Vec<u32>, Vec<u8>), ScxError> {
    // Dispatch based on codec_id
}
```

- `CodecId::None`: raw little-endian arrays (no compression). Indptr as LE u64s,
  indices as LE u16/u32, values as raw typed bytes.
- `CodecId::Scx1`: Delta-Golomb for indptr + FOR-BP for indices + Rice for values.
  **Only valid for integer value encodings (0–2)**. Attempting to use Scx1 with
  float32/float16 values must return an error.
- `CodecId::Zstd`: zstd compress/decompress each array independently. Works for
  all value encodings.
- **Per-shard codec override**: Readers MUST use the shard header's `codec_id`
  and `value_encoding`, not the file header's (SPEC §4.5). The file header
  values are defaults/hints.

### Tests
- Round-trip through each codec_id with each integer value_encoding
- Verify `None` produces raw bytes that match input arrays exactly
- Verify Scx1 + Float32 returns an error
- Zstd + Float32 round-trips correctly

---

## Step 6: File Header + Section Types (Weeks 3-4)

**Crate**: `scx-format`
**File**: `src/header.rs`
**Spec reference**: [SPEC.md §3.1](SPEC.md) — Physical Layout

```rust
/// SCX file header. 256 bytes, little-endian.
///
/// Do NOT use #[repr(C)] — we serialize/deserialize field-by-field to
/// guarantee little-endian byte order regardless of platform. Using repr(C)
/// would introduce platform-dependent padding and native-endian layout.
pub struct FileHeader {
    pub magic: [u8; 4],           // b"SCX\x01"
    pub format_version: u16,      // 1
    pub header_length: u16,       // 256
    pub flags: u32,               // bit 0: has_csc, bit 1: has_bitmap,
                                  // bit 2: has_obsm, bit 3: has_obsp,
                                  // bit 4: has_modalities (§11),
                                  // bit 5: has_deletion_vectors (§3.6.3)
    pub n_obs: u64,
    pub n_vars: u64,
    pub nnz: u64,
    pub n_csr_shards: u32,
    pub n_csc_shards: u32,        // 0 if CSC not present
    pub shard_target_rows: u32,   // default 10000 (SPEC §3.3)
    pub codec_id: u8,             // default codec (SPEC §4.5)
    pub index_dtype: u8,          // 0=u16, 1=u32 (SPEC §3.3)
    pub endian: u8,               // must be 0 (little-endian required)
    pub reserved_padding: [u8; 1],
    pub root_catalog_offset: u64, // offset of root catalog (always 256)
    pub root_catalog_length: u64,
    pub full_catalog_offset: u64, // offset of active full catalog
    pub full_catalog_length: u64,
    pub manifest_sequence: u64,   // monotonic, 0 for initial write
    pub prev_catalog_offset: u64, // 0 if first version
    pub file_checksum: u64,       // BLAKE3 truncated to 64 bits
    pub reserved: [u8; 148],      // zeroed, future use
    // Total: 4+2+2+4 + 8+8+8 + 4+4+4 + 1+1+1+1 + 8+8+8+8+8+8+8 + 148 = 256
}
```

**Size verification**: Fields before `reserved` sum to 108 bytes. `108 + 148 = 256`.

Methods:
- `FileHeader::write_to(&self, writer: &mut impl Write) → Result<()>` — serialize
  all fields as little-endian bytes using `byteorder::WriteBytesExt`
- `FileHeader::read_from(reader: &mut impl Read) → Result<Self>` — read 256 bytes,
  validate magic (`b"SCX\x01"`), endian (must be 0), format_version (must be ≤ 1)
- `FileHeader::has_csc(&self) → bool` — `self.flags & 1 != 0`
- `FileHeader::has_bitmap(&self) → bool` — `self.flags & 2 != 0`
- `FileHeader::has_obsm(&self) → bool` — `self.flags & 4 != 0`
- `FileHeader::has_obsp(&self) → bool` — `self.flags & 8 != 0`
- `FileHeader::has_deletion_vectors(&self) → bool` — `self.flags & 32 != 0`

**File**: `src/section.rs`
**Spec reference**: [SPEC.md §3.2](SPEC.md) — SECTION_TYPE ENUM

```rust
/// Section types. Matches the SECTION_TYPE ENUM in SPEC §3.2 full catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SectionType {
    ObsMetadata = 0,
    ObsIndex = 1,
    VarMetadata = 2,
    VarIndex = 3,
    CsrShard = 4,
    CscShard = 5,
    BitmapShard = 6,
    LayerCsrShard = 7,
    ObsmEmbedding = 8,
    ObspCsrShard = 9,
    UnsBlob = 10,
    Provenance = 11,
    DeletionVectors = 12,
    // 13..255 reserved for extensions (SPEC §3.9)
}

impl SectionType {
    pub fn from_u8(v: u8) -> Option<Self> { /* ... */ }
    pub fn is_known(v: u8) -> bool { v <= 12 }
}
```

Alignment helper: `fn align_to_8(offset: u64) → u64 { (offset + 7) & !7 }`

### Tests
- Write header → read back → compare all fields
- Reject magic mismatch, endian != 0, unknown format_version > 1
- Verify total serialized size is exactly 256 bytes
- Verify `reserved` bytes are all zero after round-trip
- Flag accessor tests: set individual flag bits, verify accessors

---

## Step 7: Shard Header (Week 4)

**Crate**: `scx-format`
**File**: `src/shard.rs`
**Spec reference**: [SPEC.md §3.3](SPEC.md) — CSR Shard Internal Layout

```rust
/// CSR/CSC shard header. Matches SPEC §3.3.
///
/// **Size note**: The fields as specified in SPEC §3.3 sum to 76 bytes, though
/// the SPEC diagram labels the header as "64 bytes". This is a known
/// discrepancy in the spec. We implement the full field set (76 bytes) as
/// specified in the field listing, since removing fields would lose
/// functionality. The SPEC diagram label should be corrected to 76 bytes.
pub struct ShardHeader {
    pub magic: [u8; 4],              // b"SCXS"
    pub shard_format_version: u8,    // 1
    pub shard_type: u8,              // 0=CSR, 1=CSC
    pub codec_id: u8,                // may override file header (SPEC §4.5)
    pub value_encoding: u8,          // may override file header (SPEC §4.5)
    pub index_dtype: u8,             // 0=u16, 1=u32
    pub reserved_flags: [u8; 3],
    pub n_major: u32,                // rows in this shard (CSR) or cols (CSC)
    pub n_minor: u32,                // total cols in full matrix (CSR) or rows (CSC)
    pub nnz: u64,                    // non-zeros in this shard
    pub global_offset: u64,          // first row index in full matrix
    pub indptr_rel_offset: u32,      // relative to shard start (byte 0 of header)
    pub indptr_length: u32,
    pub indices_rel_offset: u32,
    pub indices_length: u32,
    pub values_rel_offset: u32,
    pub values_length: u32,
    pub block_index_rel_offset: u32,
    pub block_index_length: u32,
    pub checksum: [u8; 8],           // BLAKE3 truncated to 64 bits
    // Total: 4+1+1+1+1+1+3 + 4+4+8+8 + 4*8 + 8 = 76 bytes
}
```

- `ShardHeader::write_to/read_from` — 76 bytes, little-endian
- Shard checksum: BLAKE3 of everything after the header (indptr + indices +
  values + block index sections), truncated to 64 bits. Computed during write,
  verified during read.
- Relative offsets: `indptr_rel_offset` is relative to the shard's start position
  in the file (i.e., byte 0 of the shard header). So the indptr data begins at
  file offset `shard_file_offset + indptr_rel_offset`.

Block index read/write (SPEC §3.3, "BLOCK INDEX" section within each shard):

```rust
/// Block index enables O(1) random access to any row range within a shard.
/// Stored at the end of each shard, after the values section.
pub struct BlockIndexEntry {
    pub row_start: u32,           // first row in block, global index
    pub n_rows: u16,              // rows in this block
    pub indptr_byte_offset: u32,  // within indptr section
    pub indices_byte_offset: u32, // within indices section
    pub values_byte_offset: u32,  // within values section
    pub nnz_in_block: u32,
}
// Each entry: 4+2+4+4+4+4 = 22 bytes
// Block index starts with n_blocks: u32 (4 bytes)

pub struct BlockIndex {
    pub entries: Vec<BlockIndexEntry>,
}
```

### Tests
- Write shard header → read back → compare all fields
- Verify exactly 76 bytes serialized
- Checksum validation: corrupt 1 byte of shard payload → checksum fails
- Block index: write 3 entries → read back → compare

---

## Step 8: Catalogs (Weeks 4-5)

**Crate**: `scx-format`
**File**: `src/catalog.rs`
**Spec reference**: [SPEC.md §3.2](SPEC.md) — Dual Catalog

### Root Catalog

The root catalog lives at a fixed position (bytes 256..256+root_catalog_length)
and provides a compact summary for fast file opening (SPEC §3.2).

```rust
/// Root catalog entry. One per section group (obs, var, csr_shards, etc.).
pub struct RootCatalogEntry {
    pub group_type: u8,               // enum matching SPEC §3.2 group_type
    pub first_section_offset: u64,
    pub total_group_length: u64,
    pub n_sections: u32,
    pub summary: [u8; 32],            // group-specific summary stats
}

/// Root catalog. Max 4096 bytes total (validated on write).
pub struct RootCatalog {
    pub n_section_groups: u16,        // SPEC §3.2: first field
    pub entries: Vec<RootCatalogEntry>,
}
```

- `RootCatalog::write_to/read_from`
- Validate: serialized size ≤ 4096 bytes on write (error if exceeded)
- Root catalog offset is always 256 (immediately after file header)

### Full Catalog

The full catalog lives at end of file (offset `full_catalog_offset` from header)
and indexes every individual section (SPEC §3.2).

```rust
pub struct ShardStats {
    pub row_start: u64,               // first row index, global
    pub row_end: u64,                 // exclusive end row index
    pub nnz: u64,
    pub value_min: u32,               // min non-zero value in shard
    pub value_max: u32,               // max non-zero value in shard
    pub value_sum: u64,               // for fast total-count computation
    pub n_indexed_columns: u8,
    // Per-column stats deferred to Phase 2 query engine (ROADMAP §2.3)
}

pub struct FullCatalogEntry {
    pub name: String,                  // UTF-8 path, e.g. "X/csr/000042"
    pub offset: u64,                   // byte offset in file
    pub length: u64,                   // byte length of section
    pub section_type: SectionType,
    pub checksum: [u8; 32],           // BLAKE3 of section content
    pub stats: Option<ShardStats>,     // present for shard types (SPEC §3.2)
}

pub struct FullCatalog {
    pub catalog_version: u16,          // 1
    pub manifest_sequence: u64,        // matches header
    pub prev_catalog_offset: u64,      // 0 if first version
    pub n_obs: u64,                    // total observable cells after deletions
    pub entries: Vec<FullCatalogEntry>,
    // Followed by catalog_checksum: [u8; 32] (BLAKE3 of all preceding bytes)
}
```

Serialization format matches SPEC §3.2 exactly:
- Entry names: `name_length: u16` + `name_bytes: [u8; name_length]`
- Stats: `stats_length: u16` (0 if no stats) + `stats: [u8; stats_length]`
- Catalog ends with `catalog_checksum: [u8; 32]` (BLAKE3 of all preceding
  catalog bytes)

Methods:
- `FullCatalog::write_to` — serialize entries, compute and append BLAKE3 checksum
- `FullCatalog::read_from` — deserialize, verify catalog checksum
- `catalog.get("X/csr/000042") → Option<&FullCatalogEntry>` — lookup by name
- `catalog.shards(SectionType::CsrShard) → Vec<&FullCatalogEntry>` — filter by type
- `catalog.shards_sorted() → Vec<&FullCatalogEntry>` — CSR shards ordered by
  `stats.row_start` (needed for shard assembly)

### Tests
- Write catalog with 10 entries → read back → compare all fields
- Catalog checksum: corrupt 1 byte → detection
- Empty catalog (file with only obs + var metadata, no shards)
- Lookup by name and by type
- ShardStats round-trip

---

## Step 9: ScxWriter — Atomic File Creation (Weeks 5-6)

**Crate**: `scx-format`
**File**: `src/writer.rs`
**Spec reference**: [SPEC.md §3.6.1](SPEC.md) — Initial File Creation

Implements the atomic rename write path.

```rust
pub struct ScxWriter {
    tmp_path: PathBuf,
    final_path: PathBuf,
    file: File,
    current_offset: u64,
    header: FileHeader,
    sections: Vec<FullCatalogEntry>,
    root_groups: Vec<RootCatalogEntry>,
}

/// Offset where sections begin writing. The header (256 bytes) and root
/// catalog placeholder (up to 4096 bytes) are reserved at the front.
/// Sections start at offset 4352 (256 + 4096), aligned to 8 bytes.
const SECTIONS_START_OFFSET: u64 = 256 + 4096;  // 4352

impl ScxWriter {
    /// Create writer. Opens tmp file, seeks to SECTIONS_START_OFFSET.
    pub fn new(path: &Path, header: FileHeader) -> Result<Self>;

    /// Write obs metadata as Arrow IPC file format (SPEC §3.4).
    pub fn write_obs(&mut self, obs: &RecordBatch) -> Result<()>;

    /// Write var metadata as Arrow IPC file format.
    pub fn write_var(&mut self, var: &RecordBatch) -> Result<()>;

    /// Write a single CSR shard. Encodes via the shard's codec_id.
    /// The writer computes the shard checksum and block index automatically.
    pub fn write_csr_shard(
        &mut self,
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],           // raw bytes; interpret per value_encoding
        global_row_offset: u64,
        n_cols: u32,
        codec_id: CodecId,
        value_encoding: ValueEncoding,
        index_dtype_u16: bool,
    ) -> Result<()>;

    /// Write a named layer CSR shard (e.g., "layers/normalized/csr/000000").
    /// Same encoding as X shards but with SectionType::LayerCsrShard.
    pub fn write_layer_csr_shard(
        &mut self,
        layer_name: &str,
        shard_index: usize,
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        global_row_offset: u64,
        n_cols: u32,
        codec_id: CodecId,
        value_encoding: ValueEncoding,
        index_dtype_u16: bool,
    ) -> Result<()>;

    /// Write uns section (raw JSON bytes, SPEC: "uns/metadata.json").
    pub fn write_uns(&mut self, json: &serde_json::Value) -> Result<()>;

    /// Write obsm section as Arrow IPC (SPEC §3.1: "obsm/{name}").
    pub fn write_obsm(&mut self, name: &str, data: &RecordBatch) -> Result<()>;

    /// Write obsp section as CSR shards (SPEC §3.1: "obsp/{name}/csr/...").
    pub fn write_obsp_shard(
        &mut self,
        name: &str,
        shard_index: usize,
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        global_row_offset: u64,
        n_rows: u32,
        codec_id: CodecId,
        value_encoding: ValueEncoding,
    ) -> Result<()>;

    /// Write provenance section (SPEC §3.7).
    pub fn write_provenance(&mut self, ops: &[ProvenanceEntry]) -> Result<()>;

    /// Finalize: write full catalog, root catalog, header. fsync + rename.
    /// Consumes self — the writer cannot be used after finish().
    pub fn finish(self) -> Result<()>;
}
```

Write sequence in `finish()` (matches SPEC §3.6.1, steps 3–7):
1. Compute file-level BLAKE3 checksum (over all section bytes)
2. Write full catalog at `current_offset` → record `full_catalog_offset`
3. Build root catalog from accumulated section groups
4. `pwrite()` root catalog at offset 256
5. `pwrite()` header at offset 0 (with all offsets filled in, file_checksum set)
6. `fsync()` the file
7. `rename(tmp_path, final_path)` — atomic on POSIX

Section alignment: before each section write, pad `current_offset` to 8-byte
boundary with zero bytes (SPEC §3.1: "Every section starts at an 8-byte-aligned
offset").

### Tests
- Write a minimal file (obs + var + 1 shard) → verify byte offsets are 8-aligned
- Write file, verify tmp file before `finish()` → no final file exists
- Write file with multiple shards → verify catalog has correct entry count
- Write file with obsm, uns, layers → all section types present in catalog
- Verify root catalog size ≤ 4096 bytes

---

## Step 10: ScxReader (Week 6)

**Crate**: `scx-format`
**File**: `src/reader.rs`
**Spec reference**: [SPEC.md §3.1–3.3](SPEC.md) — Physical Layout, Dual Catalog, CSR Shard

```rust
pub struct ScxReader {
    mmap: memmap2::Mmap,       // memory-mapped file
    header: FileHeader,
    root_catalog: RootCatalog,
    full_catalog: FullCatalog,
}

impl ScxReader {
    /// Open file: mmap, read header (256B), root catalog, full catalog.
    /// Validates header magic, endian, format_version.
    pub fn open(path: &Path) -> Result<Self>;

    /// Read and decode a CSR shard by catalog index.
    /// Uses shard header's codec_id (not file header's) per SPEC §4.5.
    /// Returns typed arrays ready for ScxCsr construction.
    pub fn read_csr_shard(&self, shard_idx: usize)
        -> Result<(Vec<u64>, Vec<u32>, Vec<f32>)>;

    /// Read all CSR shards and assemble into a single ScxCsr.
    /// Concatenates indptr (adjusting offsets), indices, and data.
    pub fn read_all_csr_shards(&self) -> Result<ScxCsr>;

    /// Read obs metadata as Arrow RecordBatch (SPEC §3.4).
    pub fn read_obs(&self) -> Result<RecordBatch>;

    /// Read var metadata as Arrow RecordBatch.
    pub fn read_var(&self) -> Result<RecordBatch>;

    /// Read named obsm as Arrow RecordBatch.
    pub fn read_obsm(&self, name: &str) -> Result<RecordBatch>;

    /// Read all obsm sections. Returns name → RecordBatch.
    pub fn read_all_obsm(&self) -> Result<HashMap<String, RecordBatch>>;

    /// Read all layer shards for a named layer.
    pub fn read_layer(&self, name: &str) -> Result<ScxCsr>;

    /// List available layer names.
    pub fn layer_names(&self) -> Vec<String>;

    /// Read uns as JSON.
    pub fn read_uns(&self) -> Result<serde_json::Value>;

    /// Verify BLAKE3 of every section against catalog checksums.
    /// Returns a list of (section_name, pass/fail) results.
    pub fn validate(&self) -> Result<Vec<(String, bool)>>;

    /// Summary accessors (no section reads needed).
    pub fn header(&self) -> &FileHeader;
    pub fn catalog(&self) -> &FullCatalog;
    pub fn n_obs(&self) -> u64;
    pub fn n_vars(&self) -> u64;
    pub fn nnz(&self) -> u64;
}
```

The reader uses `mmap` for the file, then reads sections by slicing
`&mmap[offset..offset+length]` and passing to the appropriate decoder. For
Arrow IPC sections (obs, var, obsm), use `arrow::ipc::reader::FileReader`.

### Integrity Verification (SPEC §3.8)

- On `open()`: verify header magic (`b"SCX\x01"`), endian (must be 0),
  format_version (must be ≤ supported max)
- On shard read: verify shard header magic (`b"SCXS"`), compute BLAKE3 of shard
  payload and compare truncated 64 bits against `checksum` field
- `validate()`: verify BLAKE3 of every section against full catalog checksums.
  Per SPEC §3.8: if a non-essential section (layer, obsm, obsp) is corrupted,
  report but continue; if obs/var/X is corrupted, fail.

### Unknown Section Types (SPEC §3.9)

When the full catalog contains entries with `section_type` values not recognized
by this reader version, skip them with a warning log. This allows older readers
to open files written by newer writers that add extension section types.

### Tests
- Write file with ScxWriter → open with ScxReader → compare all data
- Corrupt a shard byte → checksum fails on read
- Open a file with unknown section types → skipped with warning
- Read individual shards vs read_all_csr_shards → same assembled result
- Verify mmap-based reads return correct byte ranges

---

## Step 11: ScxCsr In-Memory Representation (Week 6)

**Crate**: `scx-sparse`
**File**: `src/csr.rs`
**Spec reference**: [SPEC.md §6.1](SPEC.md) — The `Experiment` Object / `ScxCsr`

```rust
/// SCX CSR matrix. Binary-compatible with scipy.sparse.csr_matrix layout.
/// SPEC §6.1: "ScxCsr uses i64 indptr, i32 indices, f32 data."
pub struct ScxCsr {
    pub shape: (usize, usize),      // (n_rows, n_cols)
    pub indptr: Vec<i64>,           // length n_rows + 1, scipy-compatible signed
    pub indices: Vec<i32>,          // length nnz, scipy-compatible signed
    pub data: Vec<f32>,             // length nnz, always f32 in memory
}

impl ScxCsr {
    /// Construct from raw arrays. Validates:
    /// - indptr.len() == shape.0 + 1
    /// - indptr is monotonically non-decreasing
    /// - indptr[0] >= 0 and indptr[n_rows] == nnz
    /// - indices.len() == data.len() == nnz
    /// - all indices in [0, shape.1)
    pub fn new(shape: (usize, usize), indptr: Vec<i64>,
               indices: Vec<i32>, data: Vec<f32>) -> Result<Self>;

    /// Slice rows [start..end). Returns a new ScxCsr owning copies of the
    /// sliced arrays. The returned matrix has shape (end-start, self.shape.1).
    pub fn row_slice(&self, start: usize, end: usize) -> ScxCsr;

    /// Convert to dense row-major matrix. Returns flat Vec<f32> of length
    /// shape.0 * shape.1.
    pub fn to_dense(&self) -> Vec<f32>;

    /// Number of non-zeros.
    pub fn nnz(&self) -> usize { self.data.len() }
}
```

**Type conversion note**: The on-disk format uses unsigned types: `u64` indptr,
`u16/u32` indices, and `u8/u16/u32` integer values. The in-memory representation
uses signed/float types (`i64`, `i32`, `f32`) for scipy compatibility
(SPEC §6.2: "ScxCsr uses the same memory layout as scipy's CSR"). The reader
performs widening/sign conversion during shard assembly:
- `u64` → `i64`: safe for any realistic matrix (nnz < 2^63)
- `u16/u32` → `i32`: safe for column indices < 2^31 (always true for gene counts)
- `u8/u16/u32` → `f32`: lossless for values ≤ 16,777,216 (2^24, float32 mantissa).
  Most UMI counts are < 10,000, so this is always safe.

### Tests
- Construct CSR from known data → verify row_slice returns correct submatrix
- to_dense matches manually computed dense matrix
- Empty matrix (0 nnz, shape (5, 10))
- Single-row and single-column matrices
- Validation: mismatched indptr/indices/data lengths → error
- Validation: non-monotonic indptr → error
- Validation: out-of-range index → error

---

## Step 12: h5ad Reader for Conversion (Weeks 6-7)

**Crate**: `scx-cli`
**File**: `src/convert.rs`
**Roadmap reference**: [ROADMAP.md](ROADMAP.md) §1.4 — scx-cli (minimal)

Uses the `hdf5` crate to read h5ad files. The h5ad format stores:
- `X/`: sparse matrix, typically CSR (as `data`, `indices`, `indptr` datasets)
  but may also be:
  - Dense array (some older tools write X as a full matrix)
  - CSC format (less common; needs transpose to CSR)
  - The `encoding-type` attribute on the `X` group indicates the format
    (`csr_matrix`, `csc_matrix`, or `array`)
- `obs`: DataFrame (as HDF5 group with per-column datasets, or as a
  structured array)
- `var`: DataFrame (same layout as obs)
- `obsm/`: dict of 2D arrays (e.g., `obsm/X_pca`, `obsm/X_umap`)
- `obsp/`: dict of sparse matrices (e.g., `obsp/connectivities`)
- `uns/`: nested dict → JSON (may contain arbitrary Python objects serialized
  via pickle — these should be skipped with a warning)
- `layers/`: dict of sparse matrices (same shape as X)
- `raw/`: optional raw data (pre-filtering copy of X, var, varm)

### h5ad → scx conversion pipeline

```
1. Open h5ad with hdf5 crate
2. Read X encoding-type attribute to determine format:
   a. If csr_matrix: read indptr/indices/data directly
   b. If csc_matrix: read and transpose to CSR
   c. If array (dense): read full matrix, convert to CSR
3. Read obs as Arrow RecordBatch (convert HDF5 columns → Arrow arrays):
   - Numeric columns → Arrow Int/Float arrays
   - String columns → Arrow Utf8 arrays
   - Categorical columns → Arrow Dictionary arrays (SPEC §3.4)
   - Boolean columns → Arrow Boolean arrays
4. Read var as Arrow RecordBatch (same conversion)
5. Determine index_dtype: u16 if n_vars ≤ 65535, else u32 (SPEC §3.3)
6. Chunk cells into shards of shard_target_rows (default 10,000)
7. For each shard:
   - Slice indptr/indices/data for this row range
   - Compute ShardStats (value_min, value_max, value_sum, nnz)
   - Determine value_encoding from dtype (uint8/uint16/uint32)
   - Encode with codec_id=1 (scx1) for integer counts via scx-codec
   - Write via ScxWriter::write_csr_shard()
8. Write obs, var, obsm, uns, layers, provenance
9. ScxWriter::finish()
```

### scx → h5ad conversion (reverse)

```
1. Open scx with ScxReader
2. Read all CSR shards via read_all_csr_shards() → single ScxCsr
3. Read obs/var as Arrow RecordBatch → convert to HDF5 group:
   - Arrow arrays → HDF5 datasets
   - Dictionary arrays → HDF5 categorical encoding
4. Write h5ad via hdf5 crate:
   - X as csr_matrix (indptr, indices, data datasets + encoding-type attr)
   - obs, var as HDF5 groups
   - obsm, obsp, uns, layers
```

### 10x h5 → scx conversion

10x Genomics HDF5 files have a different layout:
- `matrix/barcodes` (cell barcodes, 1D string array)
- `matrix/features/name`, `matrix/features/id`, etc. (gene info)
- `matrix/data` (non-zero values)
- `matrix/indices` (row indices — **note: 10x stores CSC, not CSR**)
- `matrix/indptr` (column pointers in CSC format)
- `matrix/shape` (2-element: [n_genes, n_barcodes])

**Important**: 10x h5 matrices are in **CSC format** (genes × cells), which is
the transpose of what SCX needs (cells × genes in CSR). The conversion must
transpose CSC → CSR. For large matrices, this can be done streaming by:
1. Reading the CSC matrix
2. Computing row (cell) counts from the CSC structure
3. Building CSR indptr from row counts
4. Scattering CSC entries into CSR positions

### Tests
- Round-trip: h5ad → scx → h5ad, bit-exact for integer counts
- Round-trip: 10x h5 → scx → to_anndata, compare with scanpy's `read_10x_h5`
- Handle edge cases: empty obs columns, categorical columns, nullable columns
- Handle dense X matrix in h5ad
- Handle CSC X matrix in h5ad (transpose)
- Handle `uns` with non-serializable items (skip with warning)
- Large-ish test: tabula sapiens subset (~100K cells)

### Pitfalls
- **h5ad files in the wild are messy.** Common issues encountered in real h5ad files:
  - `encoding-type` attribute missing on `X` group (older tools). Fall back to
    inspecting the dataset structure (3 datasets → sparse, 1 dataset → dense).
  - `X` stored as CSC instead of CSR (rare but happens). Must transpose.
  - `X` stored as dense array (some older scanpy workflows). Must sparsify.
  - `uns` containing pickled Python objects (numpy arrays, custom classes). The `hdf5`
    crate cannot unpickle these — skip with a warning. Only JSON-serializable `uns`
    entries should be converted.
  - Categorical columns stored with different HDF5 encodings across anndata versions
    (older: structured arrays; newer: `__categories` attributes). Handle both.
  - String columns using variable-length HDF5 strings vs. fixed-length. Handle both.
  - `raw/` group containing pre-filtering data. Decide whether to convert `raw` as
    a layer or skip it (recommendation: skip with a warning in v1, add `--include-raw`
    flag later).
- **CSC → CSR transpose for large matrices is memory-intensive.** A naive transpose
  allocates the full matrix twice. Use the streaming scatter approach described above
  (compute row counts → build CSR indptr → scatter entries). For very large matrices,
  consider a multi-pass approach bounded by memory.
- **Integer dtype detection.** h5ad stores count values as whatever dtype scanpy used
  (often float32 even for integer counts, or int32, or uint32). The converter must
  detect integer counts stored as float32 (check `all(X.data == X.data.astype(int))`)
  and use Rice coding (integer value_encoding) rather than Zstd (float). This is
  critical for the compression ratio advantage.
- **10x HDF5 vs h5ad confusion.** Users may pass a 10x `.h5` file to the h5ad
  converter or vice versa. Detect the format by checking for the presence of
  `matrix/barcodes` (10x) vs. `obs` (h5ad) and error with a helpful message.

---

## Step 13: CLI Commands (Week 7)

**Crate**: `scx-cli`
**Roadmap reference**: [ROADMAP.md](ROADMAP.md) §1.4 — scx-cli (minimal)

Uses `clap` for argument parsing.

### `scx convert`

```
scx convert --from h5ad input.h5ad output.scx [--shard-size 10000] [--codec scx1]
scx convert --from 10x matrix.h5 output.scx
scx convert --to h5ad input.scx output.h5ad
```

Uses the conversion pipeline from Step 12. Progress bar via `indicatif` showing:
- Number of shards written / total shards
- Bytes written / estimated total bytes
- Elapsed time

### `scx info`

```
scx info experiment.scx
```

Output:
```
SCX v1 | 500,000 cells × 33,694 genes | 847,234,567 nnz
Shards: 50 CSR | Codec: scx1 | Index dtype: u16
Value encoding: uint8 | Shard target: 10,000 rows
Manifest: sequence 0 (initial)
File size: 1.2 GB (vs ~3.1 GB h5ad estimate)
Sections:
  obs_metadata:     12.3 MB  (Arrow IPC)
  var_metadata:      1.1 MB  (Arrow IPC)
  X/csr (50):    1,187.2 MB  (scx1 codec)
  obsm/X_pca:        7.6 MB  (Arrow IPC)
  uns:                0.4 KB  (JSON)
  provenance:         0.2 KB
Flags: has_obsm
```

### `scx validate`

```
scx validate experiment.scx [--verbose]
```

Reads every section, verifies BLAKE3 checksums against full catalog entries
(SPEC §3.8). Reports per-section pass/fail. Exit code 0 if all pass, 1 if any
fail. With `--verbose`, prints checksum values.

---

## Step 14: pyscx Python Bindings (Weeks 7-9)

**Crate**: `pyscx`
**Build**: `maturin develop` for dev, `maturin build --release` for wheels
**Roadmap reference**: [ROADMAP.md](ROADMAP.md) §1.5 — pyscx (AnnData bridge)
**Spec reference**: [SPEC.md §6.2](SPEC.md) — Zero-Copy Interop with Python

### PyO3 Module Structure

```python
import scx

# Open (returns lazy handle, reads only header + catalog)
exp = scx.open("experiment.scx")

# Inspect without loading data
exp.n_obs        # 500000
exp.n_vars       # 33694
exp.nnz          # 847234567
exp.shard_count  # 50

# Convert to AnnData (zero-copy CSR, Arrow→pandas metadata)
adata = exp.to_anndata()

# Write from AnnData
scx.from_anndata(adata, "output.scx")
scx.from_10x("matrix.h5", "output.scx")
```

### `exp.to_anndata()` Implementation

This is the critical path. It must produce a valid AnnData with zero-copy
for the expression matrix (SPEC §6.2).

```rust
// In pyscx/src/anndata.rs
// Uses PyO3 0.22+ Bound API (not deprecated &PyAny)
fn to_anndata<'py>(py: Python<'py>, reader: &ScxReader) -> PyResult<Bound<'py, PyAny>> {
    // 1. Read all CSR shards, assemble into single ScxCsr
    let csr = reader.read_all_csr_shards()?;

    // 2. Transfer ownership of Rust Vecs to numpy arrays (zero-copy).
    //    PyArray::from_vec() moves the Vec's heap allocation into a
    //    Python-owned numpy array. No copy occurs.
    let indptr = numpy::PyArray1::from_vec(py, csr.indptr);
    let indices = numpy::PyArray1::from_vec(py, csr.indices);
    let data = numpy::PyArray1::from_vec(py, csr.data);

    // 3. Build scipy.sparse.csr_matrix from the three arrays.
    //    Pass (data, indices, indptr) tuple + shape. copy=False is the default
    //    when arrays have compatible dtypes (int32/int64/float32).
    //    Note: scipy may sort indices within rows if not already sorted.
    //    SCX indices ARE sorted (CSR invariant), so no copy occurs.
    let scipy_sparse = py.import("scipy.sparse")?;
    let csr_matrix = scipy_sparse.call_method1(
        "csr_matrix",
        ((data, indices, indptr), (csr.shape.0, csr.shape.1)),
    )?;

    // 4. Convert obs Arrow RecordBatch → pandas DataFrame via pyarrow.
    //    Uses the Arrow C Data Interface for zero-copy transfer from Rust
    //    to PyArrow. Then pyarrow.Table.to_pandas() handles the conversion.
    //    Use arrow-pyarrow crate's ToPyArrow trait.
    let obs_batch = reader.read_obs()?;
    let obs_pyarrow = obs_batch.to_pyarrow(py)?;
    let obs_df = obs_pyarrow.call_method0("to_pandas")?;

    // 5. Same for var
    let var_batch = reader.read_var()?;
    let var_pyarrow = var_batch.to_pyarrow(py)?;
    let var_df = var_pyarrow.call_method0("to_pandas")?;

    // 6. Read obsm → dict of numpy arrays
    let obsm_dict = PyDict::new(py);
    for (name, batch) in reader.read_all_obsm()? {
        // Convert Arrow Float32Array → numpy array
        let arr = batch.to_pyarrow(py)?;
        obsm_dict.set_item(name, arr.call_method0("to_pandas")?.call_method0("values")?)?;
    }

    // 7. Read uns → dict (JSON → Python dict via json.loads)
    let uns_json = reader.read_uns()?;
    let json_mod = py.import("json")?;
    let uns_dict = json_mod.call_method1("loads", (uns_json.to_string(),))?;

    // 8. Read layers → dict of CSR matrices (same zero-copy pattern as X)
    let layers_dict = PyDict::new(py);
    for name in reader.layer_names() {
        let layer_csr = reader.read_layer(&name)?;
        // ... same csr_matrix construction as X ...
        layers_dict.set_item(name, layer_csr_matrix)?;
    }

    // 9. Construct AnnData. AnnData does NOT copy X on construction when
    //    the dtype matches (SPEC §6.2).
    let anndata_mod = py.import("anndata")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("X", csr_matrix)?;
    kwargs.set_item("obs", obs_df)?;
    kwargs.set_item("var", var_df)?;
    kwargs.set_item("obsm", obsm_dict)?;
    kwargs.set_item("uns", uns_dict)?;
    kwargs.set_item("layers", layers_dict)?;
    anndata_mod.call_method("AnnData", (), Some(&kwargs))
}
```

**Memory ownership model**: `PyArray::from_vec()` (from the `numpy` crate for
PyO3) transfers ownership of the Rust `Vec`'s heap allocation to a Python numpy
array object. The Rust Vec is consumed (moved, not borrowed). Python/numpy now
owns the memory and will free it when the numpy array is garbage-collected. This
is zero-copy and safe — no PyCapsule or lifetime tricks needed for this simple
case.

**For complex cases** (where a single Rust struct owns multiple arrays that need
to be exposed as separate numpy views), use `PyCapsule` to store the Rust data
and set it as each numpy array's `base` object. This ensures the Rust data
outlives all numpy views. This is NOT needed for `to_anndata()` since each array
is independently owned.

### `scx.from_anndata()` Implementation

```rust
// Uses PyO3 0.22+ Bound API
fn from_anndata<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    path: &str,
    codec: Option<&str>,    // "scx1" (default), "zstd", "none"
    shard_size: Option<u32>, // default 10000
) -> PyResult<()> {
    // 1. Extract X as scipy.sparse.csr_matrix
    let x = adata.getattr("X")?;
    // Verify it's CSR format; convert if CSC or dense
    let format = x.getattr("format")?.extract::<String>()?;
    let x_csr = if format == "csr" {
        x
    } else {
        x.call_method0("tocsr")?
    };

    // 2. Get indptr/indices/data numpy arrays → view as Rust slices
    //    (readonly borrow, no copy)
    let indptr: &[i64] = x_csr.getattr("indptr")?.extract()?;
    let indices: &[i32] = x_csr.getattr("indices")?.extract()?;
    let data: &[f32] = x_csr.getattr("data")?.extract()?;

    // 3. Extract obs/var as Arrow RecordBatch via pyarrow
    let pa = py.import("pyarrow")?;
    let obs_table = pa.call_method1("Table.from_pandas", (adata.getattr("obs")?,))?;
    let obs_batch = RecordBatch::from_pyarrow(obs_table)?;

    let var_table = pa.call_method1("Table.from_pandas", (adata.getattr("var")?,))?;
    let var_batch = RecordBatch::from_pyarrow(var_table)?;

    // 4. Extract obsm, obsp, uns, layers
    // ... (iterate dicts, convert each value)

    // 5. Create ScxWriter, write all sections, finish()
    // ... (shard the CSR matrix, encode, write)
}
```

### Tests (Python, via pytest)

```python
import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata
import scx


def test_round_trip_anndata():
    """Create AnnData → write SCX → read SCX → to_anndata → compare."""
    rng = np.random.default_rng(42)
    X = sp.random(1000, 500, density=0.05, format="csr", dtype=np.float32,
                  random_state=rng)
    obs = pd.DataFrame({
        "cell_type": pd.Categorical(rng.choice(["A", "B", "C"], 1000)),
        "n_counts": rng.integers(100, 5000, 1000),
    })
    obs.index = [f"cell_{i}" for i in range(1000)]
    var = pd.DataFrame({"gene_name": [f"gene_{i}" for i in range(500)]})
    var.index = [f"ENSG{i:011d}" for i in range(500)]
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    scx.from_anndata(adata, "/tmp/test.scx")
    adata2 = scx.open("/tmp/test.scx").to_anndata()

    np.testing.assert_array_equal(adata.X.indptr, adata2.X.indptr)
    np.testing.assert_array_equal(adata.X.indices, adata2.X.indices)
    np.testing.assert_allclose(adata.X.data, adata2.X.data)
    pd.testing.assert_frame_equal(adata.obs, adata2.obs)
    pd.testing.assert_frame_equal(adata.var, adata2.var)


def test_round_trip_integer_counts():
    """Verify bit-exact round-trip for integer UMI counts (Go/No-Go gate)."""
    rng = np.random.default_rng(42)
    # Simulate realistic UMI distribution: geometric with p ≈ 0.5
    data = rng.geometric(p=0.5, size=50000).astype(np.float32)
    rows = rng.integers(0, 1000, 50000)
    cols = rng.integers(0, 500, 50000)
    X = sp.csr_matrix((data, (rows, cols)), shape=(1000, 500))
    X.sum_duplicates()

    adata = anndata.AnnData(X=X)
    scx.from_anndata(adata, "/tmp/test_int.scx")
    adata2 = scx.open("/tmp/test_int.scx").to_anndata()

    # Bit-exact: integer values must survive round-trip without any loss
    np.testing.assert_array_equal(adata.X.toarray(), adata2.X.toarray())


def test_layers_obsm_uns_round_trip():
    """Verify layers, obsm, and uns survive round-trip (ROADMAP §1.5)."""
    X = sp.random(100, 50, density=0.1, format="csr", dtype=np.float32)
    adata = anndata.AnnData(X=X)
    adata.layers["raw_counts"] = X.copy()
    adata.obsm["X_pca"] = np.random.randn(100, 50).astype(np.float32)
    adata.uns["experiment_info"] = {"name": "test", "version": 1}

    scx.from_anndata(adata, "/tmp/test_extra.scx")
    adata2 = scx.open("/tmp/test_extra.scx").to_anndata()

    np.testing.assert_allclose(adata.layers["raw_counts"].toarray(),
                               adata2.layers["raw_counts"].toarray())
    np.testing.assert_allclose(adata.obsm["X_pca"], adata2.obsm["X_pca"])
    assert adata2.uns["experiment_info"]["name"] == "test"


def test_scanpy_pipeline():
    """Verify SCX-loaded AnnData works with standard scanpy pipeline.
    This is the end-to-end validation for the Go/No-Go gate (ROADMAP §1)."""
    import scanpy as sc

    adata = scx.open("test_data/pbmc3k.scx").to_anndata()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)
    sc.tl.rank_genes_groups(adata, groupby="leiden")
    # If we get here without error, the AnnData is valid
```

### Pitfalls for pyscx

- **PyO3 version churn.** PyO3 evolves rapidly. The `Bound<'py, T>` API replaced the
  deprecated `&PyAny` smart pointer in 0.22. Pin `pyo3` and `numpy` to the same minor
  version and check https://pyo3.rs/main/migration before upgrading. The code patterns
  in this plan use the Bound API.
- **`PyArray::from_vec()` moves ownership, not borrows.** The Rust `Vec` is consumed
  and Python/numpy takes ownership of the heap allocation. This is zero-copy and safe,
  but the Rust `ScxCsr` struct is consumed — you can't access it after calling
  `from_vec()`. Structure the code to extract all three arrays (indptr, indices, data)
  before any of them are moved to Python.
- **scipy `csr_matrix` constructor dtype sensitivity.** If you pass `int64` indptr,
  `int32` indices, `float32` data, scipy constructs the matrix without copying. If
  the dtypes differ from these expectations (e.g., `uint32` indices), scipy will
  copy. ScxCsr MUST use `i64` indptr, `i32` indices, `f32` data — matching scipy's
  expectations exactly.
- **Arrow → pandas via pyarrow has edge cases.** `pyarrow.Table.to_pandas()` handles
  most types well, but:
  - Large string columns allocate Python objects (not zero-copy). Use dictionary
    encoding for repeated strings.
  - Nullable integer columns become pandas `Int64` (nullable) vs `int64` (non-nullable).
    AnnData may not expect nullable integer types in obs/var. Test with nullable columns.
  - Timestamp columns may lose timezone info in the Arrow→pandas conversion.
- **AnnData index handling.** AnnData expects `obs.index` and `var.index` to be set
  (typically cell barcodes and gene IDs). The Arrow RecordBatch doesn't have a
  distinguished index column. The converter must identify which column is the index
  (convention: `_index` column or the first string column) and set it as the pandas
  DataFrame index before passing to AnnData.
- **Test with real h5ad files, not just synthetic data.** The synthetic round-trip
  tests are necessary but not sufficient. Download real h5ad files (PBMC 3K, a
  CELLxGENE Census subset) and verify the full scanpy pipeline works. Real files
  have edge cases that synthetic data doesn't: mixed dtypes, missing values, unusual
  categorical encodings, large `uns` dicts.

---

## Step 15: Benchmarks (Weeks 9-10)

**Roadmap reference**: [ROADMAP.md](ROADMAP.md) §1.6 — Testing and Benchmarks

### Benchmark Suite

**Datasets** (store download/conversion scripts in `benchmarks/scripts/`, not
raw data):
1. **PBMC 3K** — tiny, for correctness (3K cells × 33K genes, ~30 MB h5ad)
2. **Tabula Sapiens subset** — medium (100K cells × 30K genes, ~2 GB h5ad)
3. **CELLxGENE Census subset** — large (1M cells × 30K genes, ~20 GB h5ad)
4. **Smart-seq2 dataset** — non-UMI protocol with wider count distributions.
   Essential for validating that compression claims generalize beyond 10x Chromium.
   (e.g., Tabula Muris Smart-seq2 subset)

**Metrics** (from ROADMAP Go/No-Go gate):
- **Compression ratio**: `scx_size / h5ad_size` (target: < 0.60)
- **Write throughput**: h5ad → scx conversion time (MB/s of input)
- **Read throughput**: `scx.open().to_anndata()` wall time
- **Read throughput baseline**: `anndata.read_h5ad()` wall time (for comparison)
- **Memory**: peak RSS during read (via `/proc/self/status` or `resource.getrusage`)

### Benchmark Script

```bash
# CLI benchmark command
scx benchmark experiment.scx --compare-h5ad experiment.h5ad --runs 5

# Python benchmark for to_anndata() vs anndata.read_h5ad()
python benchmarks/scripts/benchmark_read.py experiment.scx experiment.h5ad
```

Outputs a markdown table. Results committed to `benchmarks/results/` directory.

### Compression Ratio Analysis

For each benchmark dataset, also report per-component sizes:
- Indptr (Delta-Golomb) vs raw u64
- Indices (FOR-BP) vs raw u16/u32
- Values (Rice) vs raw u8/u16
- Metadata (Arrow IPC) vs h5ad HDF5

This helps validate the codec choices and identify optimization targets
(SPEC-IMPROVE.md §5b: "indices dominate compressed size").

### Benchmark Pitfalls

- **The 60% compression target is for the expression matrix, not the whole file.**
  Metadata (obs/var as Arrow IPC) is typically similar in size to HDF5 metadata.
  Report per-component sizes (indptr, indices, values, metadata) alongside total
  file size to give an honest picture.
- **h5ad compression varies.** Some h5ad files use gzip, others lz4, others no
  compression. Compare against the same h5ad file, not different compression settings.
  Also compare against the same data stored as Zarr with Zstd (the most efficient
  generic option) to show the domain-specific codec advantage clearly.
- **Warm vs cold cache.** On first read, the file is on disk. On second read, it's
  likely in the OS page cache. Report both cold (drop caches first) and warm
  (subsequent read) timings. The page cache effect can be 10×.
- **Smart-seq2 results may disappoint.** Non-UMI protocols have wider count
  distributions where Rice coding's advantage narrows. Be honest about this in
  published benchmarks. If Rice only matches Zstd on Smart-seq2 data, that's useful
  information — it means the per-shard codec override (Zstd fallback) is important.
- **Don't benchmark on the machine that's compiling Rust.** Compilation artifacts in
  cache, background processes, etc. Use dedicated benchmark runs with `cargo bench`
  or `hyperfine` on a quiet machine.

---

## Step 16: Provenance (Week 10)

**Crate**: `scx-format`
**File**: `src/provenance.rs`
**Spec reference**: [SPEC.md §3.7](SPEC.md) — Provenance Format

```rust
/// Provenance entry. Matches the binary format in SPEC §3.7.
pub struct ProvenanceEntry {
    pub timestamp: i64,                      // Unix epoch seconds
    pub action: String,                      // "create", "convert", "merge", etc.
    pub tool: String,                        // "scx-cli 0.1.0", "pyscx 0.1.0"
    pub params_json: serde_json::Value,      // tool-specific parameters
    pub input_checksums: Vec<[u8; 32]>,      // BLAKE3 of each input file
}

/// Provenance section. Contains the full operation history.
pub struct ProvenanceSection {
    pub version: u8,                         // 1
    pub operations: Vec<ProvenanceEntry>,
}
```

Serialization follows SPEC §3.7 exactly:
- `provenance_version: u8`
- `n_operations: u32`
- For each operation: length-prefixed strings (u16 length + UTF-8 bytes),
  u32-prefixed JSON, u8 count + 32-byte checksums

Usage:
- `ScxWriter` automatically adds a provenance entry on `finish()`:
  - `scx convert`: action="convert", tool="scx-cli 0.1.0",
    params={"from": "h5ad", "codec": "scx1", "shard_size": 10000},
    input_checksums=[blake3(input_file)]
  - `scx.from_anndata()`: action="create", tool="pyscx 0.1.0"
- `scx info` displays provenance history
- Future `scx merge` preserves provenance from all inputs (SPEC §3.7:
  "the output's provenance section contains the full provenance chains
  of all input files, followed by the merge operation itself")

---

## Implementation Order Summary

| Week | What | Crate | Depends On |
|------|------|-------|------------|
| 1 | Repo setup + error types + bitstream | scx-codec, scx-format | — |
| 2 | Rice + Delta-Golomb codecs | scx-codec | bitstream |
| 2-3 | FOR-BP codec | scx-codec | bitstream |
| 3 | Codec dispatch (none/scx1/zstd) | scx-codec | all codecs |
| 3-4 | File header + section types | scx-format | — |
| 4 | Shard header + block index | scx-format | — |
| 4-5 | Root catalog + full catalog | scx-format | header, section types |
| 5-6 | ScxWriter (atomic write path) | scx-format | catalogs, codec dispatch |
| 6 | ScxReader (mmap read path) | scx-format | catalogs, codec dispatch |
| 6 | ScxCsr in-memory representation | scx-sparse | — |
| 6-7 | h5ad/10x reader for conversion | scx-cli | writer, reader |
| 7 | CLI commands (convert, info, validate) | scx-cli | conversion pipeline |
| 7-9 | pyscx Python bindings + to_anndata | pyscx | reader, writer, sparse |
| 9-10 | Benchmarks + conformance tests | all | everything |
| 10 | Provenance | scx-format | writer |

**Critical path**: bitstream → codecs → header/shard → catalogs → writer →
reader → conversion → pyscx → benchmarks.

---

## Known SPEC Discrepancies

Issues found during plan review. Status as of March 2026:

1. **FileHeader reserved field size**: ~~The SPEC §3.1 listed `reserved: [u8; 132]`
   but the fields before it sum to 108 bytes, giving a total of 240.~~ **FIXED in
   SPEC v0.5**: `reserved` is now `[u8; 148]` (108 + 148 = 256). A size accounting
   note has been added to the spec.

2. **ShardHeader size**: The SPEC §3.3 diagram labels the shard header as
   "64 bytes" but the specified fields sum to 76 bytes. This plan implements the
   full field set (76 bytes) as listed. **SPEC v0.5 now includes a note** clarifying
   that implementations MUST use 76 bytes and the diagram label is incorrect.

3. **SPEC §15 roadmap is outdated**: The in-spec roadmap still has the old
   Phase 0 (validate thesis via h5ad loader). The authoritative roadmap is
   [ROADMAP.md](ROADMAP.md) which was restructured to start directly with the
   format implementation.

---

## What's Explicitly Out of Scope for Phase 1

Per [ROADMAP.md](ROADMAP.md) phasing:

- Training data loader (Phase 2, ROADMAP §2.1)
- Query engine / lazy evaluation / predicate pushdown (Phase 2, ROADMAP §2.2)
- Predicate indexes (Phase 2, ROADMAP §2.3)
- Fragment/manifest append/delete/compact/rollback (Phase 2, ROADMAP §2.4)
- Fused normalize+log1p (Phase 2, ROADMAP §2.5)
- SIMD/AVX2 codec optimizations (Phase 3, ROADMAP §3.5)
- GPU/CUDA anything (Phase 3, ROADMAP §3.1)
- GDS (Phase 3, ROADMAP §3.1)
- R bindings (Phase 3, ROADMAP §3.3)
- CSC / build-csc (Phase 3, ROADMAP §3.2)
- Multimodal (Phase 3, ROADMAP §3.4)
- Detection bitmap (Phase 3, ROADMAP §3.5)
