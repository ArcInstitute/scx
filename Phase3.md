# Phase 3 Implementation Plan: GPU Path + Ecosystem

**Goal**: GPU-accelerated I/O, GDS, CUDA codec decoders, R bindings, multimodal support, SIMD optimizations, and production polish for v1.0 release.
**Spec reference**: [SPEC.md](SPEC.md) v0.5 (Sections 3.9, 4.4, 5, 8.6, 10.3, 10.4, 11, 14)
**Roadmap reference**: [ROADMAP.md](ROADMAP.md) Phase 3 (Sections 3.1–3.5)
**Depends on**: [Phase 2](./tasks/Phase2.md) (complete). All Phase 2 Go/No-Go gates passed.
**See also**: [Phase 2](./tasks/Phase2-*.md) (complete). Detailed Phase 2 implementation plans.

### Go/No-Go Gate (from [ROADMAP.md](ROADMAP.md))

Before declaring v1.0, these criteria must be met:

- Format spec frozen (no breaking changes after v1.0)
- R and Python bindings both pass conformance suite
- GDS path benchmarked: >2× CPU path throughput on NVMe

---

## Phase 2 Retrospective: Findings That Shape Phase 3

Phase 2 delivered the training loader, query engine, file operations, and cloud
support. Key findings relevant to Phase 3:

### 1. GPU Utilization Is Model-Bound, Not Loader-Bound

Phase 2 benchmarks showed the SCX loader delivers 397 batches/sec (405K cells/sec)
on an H100 — fast enough that a lightweight VAE (597K params) only achieves 27.2%
GPU utilization because the model compute is trivial. GDS and CUDA decoders will
further reduce data delivery latency, but the primary GPU utilization gain comes from
larger models (full scVI with 4L/2048h/NB likelihood). GPU codec decoders eliminate
the CPU decode bottleneck entirely, enabling the loader to scale to multi-GPU setups
where CPU decode becomes the constraint.

### 2. Codec Decode Is the Read Path Bottleneck

Phase 2 parallel shard decode achieves 1.58–2.19× speedup on multi-shard datasets,
but Rice/FOR-BP decoding is inherently sequential per-value. SIMD (AVX2/NEON) batch
processing can accelerate the scalar decoder 4–8× for the CPU path. GPU warp-level
decode eliminates the CPU entirely for the GDS path.

### 3. R Users Are a Large Unserved Community

Seurat is one of the most-cited single-cell tools. R users currently cannot use SCX
at all. The `extendr` crate (v0.8.1) provides stable Rust↔R FFI. Seurat v5's layer
model and BPCells integration define the target interop surface.

### 4. Multimodal Data Is Increasingly Common

CITE-seq (RNA + protein) and spatial transcriptomics datasets are growing rapidly.
The SPEC §11 multimodal extension is designed but unimplemented. CITE-seq protein
counts have wider distributions than RNA UMI counts — per-shard codec override
(SPEC §4.5) must default to Zstd for protein modalities.

---

## Architecture Overview

Phase 3 adds two new crates (`scx-gpu`, `rscx`) to the existing 9-member workspace:

```
scx/
├── Cargo.toml                  # workspace root (11 members after Phase 3)
├── scx-format/                 # [existing] extend for multimodal + bitmap
├── scx-codec/                  # [existing → extended] SIMD decode paths
├── scx-sparse/                 # [existing → extended] CSC support, transpose
├── scx-engine/                 # [existing] minor multimodal query extensions
├── scx-loader/                 # [existing → extended] optional GPU/GDS path
├── scx-ops/                    # [existing]
├── scx-cloud/                  # [existing] (added in Phase 2)
├── scx-cli/                    # [existing → extended] new subcommands
├── pyscx/                      # [existing → extended] multimodal + conformance
├── scx-gpu/                    # [NEW] CUDA codec decoders, cuSPARSE, GDS
└── rscx/                       # [NEW] R bindings (extendr)
```

### Crate Dependency Order (Phase 3)

```
scx-codec (standalone)
    └─> scx-format (depends on scx-codec)
            ├─> scx-sparse (standalone)
            ├─> scx-ops (depends on scx-format)
            ├─> scx-engine (depends on scx-format, scx-codec, scx-sparse)
            ├─> scx-loader (depends on scx-format, scx-codec, scx-sparse)
            ├─> scx-cloud (depends on scx-format, scx-codec, scx-engine)
            ├─> scx-gpu (depends on scx-format, scx-codec, scx-sparse)
            ├─> scx-cli (depends on all above)
            ├─> pyscx (depends on all above)
            └─> rscx (depends on scx-format, scx-codec, scx-sparse, scx-engine, scx-ops)
```

---

## Detailed Task Plan

### Step 1: SIMD Codec Optimizations — SKIPPED

**Crate**: `scx-codec`
**Status**: Evaluated and skipped — scalar implementations are already near-optimal
**Spec reference**: SPEC.md §4.4

#### Decision

SIMD optimizations (AVX2/NEON) for Rice and FOR-BP decoders were fully
implemented, benchmarked, and **reverted** because the results showed no
meaningful improvement and introduced regressions.

#### Benchmark Findings

AVX2 Rice and FOR-BP decoders were implemented with runtime feature detection
and tested for bit-identical output. Results on an x86-64 system with AVX2+BMI2:

| Component | SIMD vs Scalar | Notes |
|-----------|---------------|-------|
| **Rice decode** (10K–1M values) | **1.7× slower** | `#[target_feature]` prevents inlining the hot per-value loop |
| **FOR-BP decode** (2K rows) | ~3% faster | Within noise margin |
| **FOR-BP decode** (16K rows) | No change | LEB128 per-row overhead dominates |
| **End-to-end shard** (2K rows) | ~1–2% faster | Negligible real-world impact |
| **End-to-end shard** (16K rows) | ~1% faster | Negligible real-world impact |

#### Why SIMD Doesn't Help

1. **Rice is inherently sequential.** Variable-length unary quotients mean value
   N+1 can't be decoded until value N is finished. No SIMD instruction can
   parallelize this. The scalar `BitReader` with its 64-bit buffer is already
   near-optimal.

2. **`#[target_feature]` prevents inlining.** Rust requires SIMD functions to
   carry this annotation, which creates a call boundary the compiler can't
   optimize across. For Rice, this overhead alone exceeds the SIMD benefit.

3. **FOR-BP's bottleneck is per-row overhead**, not bit extraction. LEB128
   varint parsing, per-row frame_min/frame_bits reads, and memory allocation
   dominate. The `unpack_fixed_width` batch extraction (already optimized in
   Phase 2) handles the bit-packing efficiently.

4. **Delta-Golomb** operates on n_rows+1 values (tiny), making SIMD irrelevant.

#### Alternatives Considered

- **`-C target-cpu=native`**: Allows LLVM auto-vectorization without `#[target_feature]`
  inlining penalty; worth evaluating as a build flag rather than hand-written intrinsics.
- **Higher-level optimizations**: Arena allocation, batch shard decoding, or I/O
  pipelining would likely yield larger gains than codec-level SIMD.

---

### Step 2: scx-gpu — CUDA Codec Decoders + cuSPARSE + GDS

**Crate**: `scx-gpu` (new)
**Spec reference**: SPEC.md §4.4 (GPU decode), §8.6 (GDS)
**Roadmap reference**: ROADMAP.md §3.1

#### 2.1 Crate Setup

```
scx-gpu/
├── Cargo.toml
├── build.rs              # CUDA kernel compilation
├── kernels/
│   ├── rice_decode.cu    # Rice decoder CUDA kernel
│   ├── forbp_decode.cu   # FOR-BP decoder CUDA kernel
│   └── sparse_dense.cu   # CSR sparse-to-dense kernel
└── src/
    ├── lib.rs
    ├── error.rs          # GpuError enum
    ├── device.rs         # CUDA device management
    ├── rice_gpu.rs       # Rice GPU decode wrapper
    ├── forbp_gpu.rs      # FOR-BP GPU decode wrapper
    ├── shard_decode.rs   # Full shard GPU decode pipeline
    ├── cusparse.rs       # cuSPARSE CSR interop
    ├── gds.rs            # GPUDirect Storage (cuFileRead)
    └── sparse_dense.rs   # GPU sparse-to-dense for training batches
```

```toml
# scx-gpu/Cargo.toml
[dependencies]
scx-format = { path = "../scx-format" }
scx-codec = { path = "../scx-codec" }
scx-sparse = { path = "../scx-sparse" }
cudarc = "0.19"           # CUDA runtime + driver API
thiserror = { workspace = true }

[build-dependencies]
cc = { version = "1", features = ["cuda"] }

[features]
default = []
gds = []                  # opt-in GDS support (requires nvidia-fs drivers)
```

#### 2.2 CUDA Rice Decoder (`kernels/rice_decode.cu`)

Per SPEC §4.4: each warp (32 threads) decodes one block of 256 values.

```c
// kernels/rice_decode.cu
__global__ void rice_decode_kernel(
    const uint8_t* __restrict__ bitstream,
    const uint32_t* __restrict__ block_offsets,  // byte offset per block
    const uint8_t*  __restrict__ block_k,        // Rice k per block
    uint32_t* __restrict__ output,
    uint32_t n_blocks,
    uint32_t block_size                          // 256
) {
    // Each warp decodes one block
    uint32_t block_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    uint32_t lane = threadIdx.x % 32;

    if (block_id >= n_blocks) return;

    // Load block bitstream into shared memory
    // Thread t decodes values at positions t, t+32, t+64, ..., t+224
    // Use __ballot_sync() for unary quotient boundaries
    // Warp-shuffle to resolve bit offsets between threads
}
```

#### 2.3 CUDA FOR-BP Decoder (`kernels/forbp_decode.cu`)

Per-row decode with warp-level parallel bit extraction:

```c
__global__ void forbp_decode_kernel(
    const uint8_t* __restrict__ bitstream,
    const uint32_t* __restrict__ row_offsets,
    const uint16_t* __restrict__ row_nnz,
    const uint16_t* __restrict__ frame_min_arr,
    const uint8_t*  __restrict__ frame_bits_arr,
    uint32_t* __restrict__ output_indices,
    uint32_t n_rows
);
```

#### 2.4 cuSPARSE CSR Interop (`cusparse.rs`)

Zero-copy interop between decoded shard data on GPU and cuSPARSE:

```rust
pub struct GpuCsr {
    pub indptr: cudarc::CudaSlice<i64>,
    pub indices: cudarc::CudaSlice<i32>,
    pub data: cudarc::CudaSlice<f32>,
    pub shape: (usize, usize),
}

impl GpuCsr {
    /// Create a cuSPARSE CSR matrix descriptor from GPU buffers.
    /// Zero-copy: the cuSPARSE handle borrows the existing device memory.
    pub fn to_cusparse_csr(&self) -> Result<CusparseSpMatDescr, GpuError>;

    /// Convert to cupy.sparse.csr_matrix via __cuda_array_interface__.
    /// Zero-copy: exposes device pointers directly to Python/cupy.
    pub fn to_cupy(&self) -> Result<CupyCsr, GpuError>;
}
```

#### 2.5 GPUDirect Storage (`gds.rs`)

NVMe → GPU VRAM bypass via `cuFileRead()` (SPEC §8.6):

```rust
/// GDS reader for SCX shard data.
/// Reads shard bytes directly from NVMe into GPU VRAM,
/// bypassing CPU memory entirely.
pub struct GdsReader {
    cu_file_handle: CuFileHandle,
    device: CudaDevice,
}

impl GdsReader {
    /// Open an SCX file for GDS access.
    /// Requires: local NVMe, nvidia-fs drivers, ext4/XFS filesystem.
    /// Falls back to CPU path with warning if GDS prerequisites not met.
    pub fn open(path: &Path) -> Result<Self, GpuError>;

    /// Read a shard directly into GPU VRAM at the given device buffer offset.
    /// Uses cuFileRead() at the shard's file offset from the catalog.
    pub fn read_shard_to_gpu(
        &self,
        shard_offset: u64,
        shard_length: u64,
        device_buffer: &mut CudaSlice<u8>,
    ) -> Result<(), GpuError>;
}
```

**Prerequisites check**: At `GdsReader::open()`, verify:
1. nvidia-fs kernel module is loaded (`/dev/nvidia-fs*` exists)
2. File is on a local filesystem (not NFS/GPFS/Lustre)
3. Filesystem is ext4 or XFS (GDS requirement)
4. CUDA context is initialized

If any check fails, return `GpuError::GdsUnavailable` with a descriptive message
suggesting the CPU path fallback.

#### 2.6 GPU Sparse-to-Dense Conversion (`sparse_dense.rs`)

GPU kernel for training batch densification:

```rust
/// Convert a CSR shard on GPU to a dense matrix for training.
/// Each thread handles one row: scatter non-zero values into the dense output.
/// Combined with HVG projection: only writes columns in the HVG bitmap.
pub fn sparse_to_dense_gpu(
    gpu_csr: &GpuCsr,
    hvg_bitmap: Option<&CudaSlice<u32>>,
    output: &mut CudaSlice<f32>,  // [n_rows × n_output_genes]
) -> Result<(), GpuError>;
```

#### 2.7 GPU Training Pipeline Integration

Extend `scx-loader` with an optional `gpu` feature that uses `scx-gpu`:

```rust
// scx-loader/src/gds_stage.rs (new, behind feature flag)

/// GDS I/O stage: reads shards directly into GPU VRAM.
/// Replaces the CPU io_stage when GDS is available.
///
/// Pipeline:
///   GDS read (NVMe → VRAM) → GPU decode (Rice/FOR-BP kernels) →
///   GPU sparse-to-dense → GPU normalize+log1p → Python __next__()
///
/// The entire hot path stays on GPU. CPU is only used for
/// pipeline orchestration and Python __next__() calls.
pub async fn gds_io_stage(
    gds_reader: GdsReader,
    shard_order: Vec<usize>,
    group_size: usize,
    tx: tokio::sync::mpsc::Sender<GpuShardGroup>,
) -> Result<()>;
```

```toml
# scx-loader/Cargo.toml (additions)
[features]
gpu = ["dep:scx-gpu"]

[dependencies]
scx-gpu = { path = "../scx-gpu", optional = true }
```

#### Tests

- [ ] GPU Rice decoder produces bit-identical output to scalar CPU reference
- [ ] GPU FOR-BP decoder produces bit-identical output to CPU reference
- [ ] GPU shard decode matches CPU shard decode end-to-end
- [ ] cuSPARSE CSR descriptor creation from decoded GPU buffers
- [ ] GDS read + GPU decode pipeline produces correct data
- [ ] GDS fallback: graceful error when prerequisites not met
- [ ] GPU sparse-to-dense matches CPU sparse-to-dense with HVG projection
- [ ] Multi-GPU: device selection and data placement

#### Benchmarks

- [ ] **GPU vs CPU decode throughput**: Rice and FOR-BP on synthetic shards.
      Target: ≥10× GPU speedup for large shards (>10K rows)
- [ ] **GDS vs CPU path latency**: Time to read + decode a shard.
      Target: GDS >2× CPU path throughput on NVMe (Go/No-Go criterion)
- [ ] **End-to-end GPU training pipeline**: batches/sec with GDS path.
      Compare against CPU path from Phase 2
- [ ] **GPU sparse-to-dense**: compare against CPU rayon densification
- [ ] **Multi-shard parallel GPU decode**: scaling with shard count

---

### Step 3: rscx — R Bindings

**Crate**: `rscx` (new)
**Spec reference**: SPEC.md §7.3 (R API)
**Roadmap reference**: ROADMAP.md §3.3
**Dependency**: `extendr` v0.8.x

#### 3.1 Crate Setup

```
rscx/
├── Cargo.toml
├── R/                    # R wrapper functions
│   ├── scx.R             # scx_open, scx_info, scx_convert
│   ├── query.R           # filter_obs, select_genes, collect
│   ├── interop.R         # to_seurat, to_sce, from_seurat, from_sce
│   └── ops.R             # append, delete, compact, rollback, merge
├── src/
│   ├── lib.rs            # extendr entry point
│   ├── reader.rs         # ScxExperiment R class
│   ├── query.rs          # Query pipeline R bindings
│   ├── interop.rs        # Seurat/SCE conversion helpers
│   └── ops.rs            # File operations R bindings
├── DESCRIPTION           # R package metadata
├── NAMESPACE             # R export declarations
├── man/                  # R documentation (roxygen2)
└── tests/
    └── testthat/         # R unit tests
```

```toml
# rscx/Cargo.toml
[dependencies]
scx-format = { path = "../scx-format" }
scx-codec = { path = "../scx-codec" }
scx-sparse = { path = "../scx-sparse" }
scx-engine = { path = "../scx-engine" }
scx-ops = { path = "../scx-ops" }
extendr-api = "0.8"
thiserror = { workspace = true }

[lib]
crate-type = ["cdylib"]
```

#### 3.2 Core R API (`reader.rs`)

```rust
use extendr_api::prelude::*;

/// R class wrapping an SCX file handle.
pub struct ScxExperiment {
    reader: scx_format::ScxReader,
}

#[extendr]
impl ScxExperiment {
    /// Open an SCX file. Returns a lazy handle.
    fn new(path: &str) -> Result<Self>;

    /// Number of observations (cells).
    fn n_obs(&self) -> i64;

    /// Number of variables (genes).
    fn n_vars(&self) -> i64;

    /// Read obs metadata as an R data.frame.
    fn obs(&self) -> Result<Robj>;

    /// Read var metadata as an R data.frame.
    fn var(&self) -> Result<Robj>;

    /// Read X matrix as a dgCMatrix (Matrix package sparse matrix).
    fn x_matrix(&self) -> Result<Robj>;

    /// Convert to Seurat v5 object.
    fn to_seurat(&self) -> Result<Robj>;

    /// Convert to SingleCellExperiment object.
    fn to_sce(&self) -> Result<Robj>;
}
```

#### 3.3 R Pipe-Friendly Query API (`query.rs`)

> **Important**: The Rust `QueryPipeline` consumes `self` on each builder
> method — e.g. `filter_obs(mut self, expr: &str) -> Result<Self>`,
> `select_genes(mut self, gene_indices: Vec<u32>) -> Self`. R cannot express
> move semantics. The `RQueryPipeline` must wrap the inner pipeline in
> `Option<QueryPipeline>` and `.take()` on each call, returning a new
> `RQueryPipeline` with the configured pipeline.

```rust
/// R query pipeline — pipe-friendly with |>.
/// Wraps scx_engine::QueryPipeline, which uses consuming-self builder pattern.
pub struct RQueryPipeline {
    inner: Option<scx_engine::QueryPipeline>,  // Option for R move semantics
}

#[extendr]
impl RQueryPipeline {
    /// Filter observations by predicate expression.
    /// Takes ownership of the inner pipeline and returns a new RQueryPipeline.
    fn filter_obs(&mut self, expr: &str) -> Result<RQueryPipeline>;
    fn filter_var(&mut self, expr: &str) -> Result<RQueryPipeline>;
    /// Note: indices are Vec<i32> from R (R has no unsigned int), converted to Vec<u32> internally.
    fn select_genes(&mut self, indices: Vec<i32>) -> RQueryPipeline;
    fn with_normalize(&mut self, target_sum: f64) -> RQueryPipeline;
    fn with_log1p(&mut self) -> RQueryPipeline;
    fn limit(&mut self, n: i32) -> RQueryPipeline;
    fn collect(&mut self) -> Result<Robj>;
}
```

R usage:

```r
library(rscx)

result <- scx_open("experiment.scx") |>
  filter_obs("tissue == 'lung'") |>
  select_genes(hvg_indices) |>
  with_normalize(target_sum = 1e4) |>
  with_log1p() |>
  collect()

# Convert to ecosystem objects
seu <- to_seurat(result)
sce <- to_sce(result)
```

#### 3.4 Seurat v5 Interop (`interop.rs`)

Convert ScxCsr to Seurat v5 assay format:

> **dgCMatrix orientation note**: R/Matrix's `dgCMatrix` is CSC (column-major),
> while SCX stores CSR (row-major). The conversion must transpose the matrix:
> `dgCMatrix(i = indices, p = indptr, x = data)` interprets its arguments as
> CSC. We must either:
> (a) Transpose the ScxCsr to CSC before constructing dgCMatrix, OR
> (b) Construct a `dgRMatrix` (CSR variant) and coerce to `dgCMatrix`.
> Option (a) is preferred — use a local CSR→CSC transpose.

```rust
/// Create a Seurat v5 object from SCX query results.
///
/// ScxCsr uses i64 indptr / i32 indices / f32 data (CSR, row-major).
/// R's dgCMatrix uses integer (i32) p / integer i / double (f64) x (CSC, column-major).
///
/// Conversion steps:
///   1. Transpose ScxCsr (CSR) → CSC arrays (indptr_csc, indices_csc, data_csc)
///   2. indptr i64→i32 (safe for <2B nnz), data f32→f64 (widening, no loss)
///   3. Construct dgCMatrix via new("dgCMatrix", i=, p=, x=, Dim=)
///
/// Seurat v5 layers:
///   - counts layer: raw integer counts from X
///   - data layer: normalized values (if with_normalize was applied)
///
/// obs → Seurat meta.data, var → Seurat feature metadata
fn to_seurat_v5(result: &QueryResult) -> Result<Robj> {
    // 1. Transpose ScxCsr (CSR) → CSC arrays
    // 2. Create dgCMatrix from CSC arrays
    // 3. Create Seurat::CreateSeuratObject(counts = dgc)
    // 4. Set meta.data from obs RecordBatch → data.frame
    // 5. Set feature metadata from var RecordBatch
    // 6. Return Seurat object
}

/// Create a SingleCellExperiment from SCX query results.
///
/// Same dgCMatrix construction as to_seurat_v5.
/// obs → colData (note: SCE uses genes-as-rows, cells-as-columns)
/// var → rowData
fn to_sce(result: &QueryResult) -> Result<Robj> {
    // 1. Transpose ScxCsr → CSC → dgCMatrix
    // 2. Call SingleCellExperiment::SingleCellExperiment(
    //        assays = list(counts = dgc),
    //        colData = obs_df,
    //        rowData = var_df
    //    )
}

/// Import from a Seurat object to an SCX file.
/// Extracts the counts dgCMatrix, transposes CSC→CSR, then writes via ScxWriter.
fn from_seurat(seurat_obj: Robj, output_path: &str) -> Result<()> {
    // 1. Extract counts matrix (GetAssayData or LayerData)
    // 2. Transpose dgCMatrix (CSC) → CSR arrays
    // 3. Extract meta.data → obs RecordBatch
    // 4. Extract feature metadata → var RecordBatch
    // 5. Write SCX file via ScxWriter
}

/// Import from a SingleCellExperiment to an SCX file.
/// Same CSC→CSR transpose as from_seurat.
fn from_sce(sce_obj: Robj, output_path: &str) -> Result<()>;
```

#### 3.5 File Operations (`ops.rs`)

```rust
#[extendr]
fn scx_convert(input: &str, output: &str, from: &str) -> Result<()>;
fn scx_append(path: &str, input: &str) -> Result<()>;
fn scx_delete(path: &str, filter: &str) -> Result<()>;
fn scx_compact(input: &str, output: &str) -> Result<()>;
fn scx_rollback(path: &str) -> Result<()>;
fn scx_merge(inputs: Vec<String>, output: &str) -> Result<()>;
fn scx_info(path: &str) -> Result<Robj>;
fn scx_validate(path: &str) -> Result<bool>;
```

#### 3.6 R Package Build

The R package uses `rextendr` for build scaffolding:

```r
# DESCRIPTION
Package: rscx
Title: SCX File Format for Single-Cell RNA-seq
Version: 0.1.0
SystemRequirements: Rust (>= 1.78), Cargo
Depends: R (>= 4.2.0)
Imports: Matrix, methods
Suggests: Seurat (>= 5.0.0), SingleCellExperiment, testthat (>= 3.0.0)
```

#### Tests

- [ ] `scx_open()` returns valid handle with correct n_obs/n_vars
- [ ] `obs()` / `var()` return correct R data.frames
- [ ] `x_matrix()` returns valid dgCMatrix matching Python X matrix
- [ ] Query pipeline: `filter_obs() |> collect()` returns correct subset
- [ ] `to_seurat()` produces valid Seurat v5 object accepted by Seurat functions
- [ ] `to_sce()` produces valid SingleCellExperiment
- [ ] `from_seurat()` round-trips: `to_seurat() → from_seurat() → to_seurat()` is equivalent
- [ ] `from_sce()` round-trips correctly
- [ ] File operations (append, delete, compact, rollback, merge) work from R
- [ ] R conformance tests pass for reference .scx files

---

### Step 4: Multimodal Support

**Crates**: `scx-format` (modify), `scx-cli` (modify), `pyscx` (modify), `rscx` (modify)
**Spec reference**: SPEC.md §11 (Multimodal Extensibility)
**Roadmap reference**: ROADMAP.md §3.4

#### 4.1 Multi-Feature-Space Writer (`scx-format`)

Extend `ScxWriter` to support the `has_modalities` flag (header bit 4) and
modality-prefixed section names.

> **Implementation note**: The `has_modalities` flag is defined in SPEC.md
> (§3.1, bit 4) but **not yet implemented** in `header.rs`. Phase 3 must add:
> ```rust
> // scx-format/src/header.rs (additions)
> pub fn has_modalities(&self) -> bool {
>     self.flags & (1 << 4) != 0
> }
> pub fn set_modalities(&mut self) {
>     self.flags |= 1 << 4;
> }
> ```

Multimodal sections reuse existing `SectionType` values with modality-prefixed
names (e.g., `mod/rna/var` uses `SectionType::VarMetadata`, `mod/rna/X_shard_0`
uses `SectionType::CsrShard`). The modality is encoded in the section *name*,
not in the section *type*. This avoids adding new section type IDs.

```rust
// scx-format/src/writer.rs (additions)

pub struct ModalityWriter<'a> {
    writer: &'a mut ScxWriter,
    modality_name: String,   // e.g., "rna", "protein"
}

impl<'a> ModalityWriter<'a> {
    /// Write var metadata for this modality.
    /// Section name: "mod/{modality_name}/var", type: VarMetadata
    pub fn write_var(&mut self, var: &RecordBatch) -> Result<()>;

    /// Write a CSR shard for this modality's expression matrix.
    /// Section name: "mod/{modality_name}/X_shard_{idx}", type: CsrShard
    /// The modality may have a different n_vars than the primary matrix.
    pub fn write_csr_shard(
        &mut self,
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        value_encoding: ValueEncoding,
        codec: CodecId,
        row_start: u64,
    ) -> Result<()>;
}

impl ScxWriter {
    /// Begin writing a modality. Sets the has_modalities flag (bit 4).
    /// Section paths are prefixed: mod/{name}/var, mod/{name}/X_shard_{idx}
    pub fn begin_modality(&mut self, name: &str) -> ModalityWriter;
}
```

#### 4.2 Multi-Feature-Space Reader (`scx-format`)

```rust
// scx-format/src/reader.rs (additions)

impl ScxReader {
    /// List available modalities. Returns empty vec for single-modality files.
    pub fn modalities(&self) -> Vec<String>;

    /// Read var metadata for a specific modality.
    pub fn read_modality_var(&self, modality: &str) -> Result<RecordBatch>;

    /// Read CSR shards for a specific modality.
    pub fn read_modality_x(&self, modality: &str) -> Result<ScxCsr>;
}
```

#### 4.3 CITE-seq Support

CITE-seq files have two modalities sharing the same obs:

```python
# Python API
scx.from_mudata(mudata_obj, "cite_seq.scx")
# Writes:
#   obs metadata (shared)
#   mod/rna/var (RNA genes)
#   mod/rna/X/csr/... (RNA count shards, codec=auto → Scx1)
#   mod/protein/var (ADT antibodies, ~200 features)
#   mod/protein/X/csr/... (protein count shards, codec=auto → Zstd)

exp = scx.open("cite_seq.scx")
rna_adata = exp.modality("rna").to_anndata()
protein_adata = exp.modality("protein").to_anndata()
mudata = exp.to_mudata()  # MuData with both modalities
```

**Codec selection for protein modality**: ADT counts have wider distributions
(median often >8) and less extreme sparsity than RNA UMI counts. The auto-codec
selection (Phase 2) naturally selects Zstd for protein modalities. No special
handling needed beyond the existing per-shard codec override mechanism (SPEC §4.5).

#### 4.4 Spatial Transcriptomics Support

Spatial coordinates are standard numeric columns in obs metadata (`x_spatial`,
`y_spatial`, `z_spatial`). An optional R-tree spatial index enables spatial
range queries.

```rust
// scx-format/src/spatial.rs (new)

/// Write spatial coordinate columns to obs metadata.
/// Validates that x_spatial and y_spatial columns exist.
pub fn validate_spatial_obs(obs: &RecordBatch) -> Result<bool>;

/// Optional R-tree spatial index section for spatial range queries.
/// Serialized as a flat packed R-tree (flatbush format).
pub struct SpatialIndex {
    pub tree: Vec<u8>,        // serialized R-tree
    pub coordinate_columns: Vec<String>,  // e.g., ["x_spatial", "y_spatial"]
}

impl SpatialIndex {
    pub fn build(obs: &RecordBatch, coord_columns: &[&str]) -> Result<Self>;
    pub fn query_range(&self, min: &[f64], max: &[f64]) -> Vec<u64>;
}
```

#### 4.5 MuData/h5mu Conversion

```bash
# CLI
scx convert --from h5mu input.h5mu output.scx
scx convert --to h5mu input.scx output.h5mu
```

```python
# Python
scx.from_mudata(mudata_obj, "output.scx")
mudata = scx.open("output.scx").to_mudata()
```

**h5mu reading**: Uses the `hdf5` crate (same as h5ad conversion). h5mu files
contain a `/mod` group with one sub-group per modality, each following the
AnnData schema. Shared obs metadata is at the top level.

#### Tests

- [ ] CITE-seq round-trip: h5mu → scx → h5mu, both modalities preserved
- [ ] Per-modality codec selection: RNA=Scx1, protein=Zstd (auto)
- [ ] Per-modality var metadata is independent
- [ ] Shared obs metadata is correct across modalities
- [ ] Spatial coordinates stored and queryable via R-tree index
- [ ] Spatial range query returns correct cell indices
- [ ] `to_mudata()` produces valid MuData accepted by muon
- [ ] Single-modality files work unchanged (backward compatible)
- [ ] Python + R bindings both support multimodal read/write

---

### Step 5: CLI Extensions

**Crate**: `scx-cli` (modify)
**Spec reference**: SPEC.md §10.3, §10.4, §3.9
**Roadmap reference**: ROADMAP.md §3.2

#### 5.1 `scx build-csc` — Streaming CSR→CSC Transpose

```bash
scx build-csc input.scx output.scx [--memory-limit 4G]
```

Reads CSR shards, performs streaming transpose bounded by a configurable memory
limit, and writes a new file with both CSR and CSC shard sets. Sets the `has_csc`
flag (header bit 0).

```rust
// scx-sparse/src/transpose.rs (new)

/// Streaming CSR → CSC transpose.
///
/// Memory-bounded: accumulates column data in chunks of `chunk_cols` columns.
/// For a 30K-gene × 100K-cell matrix with 4GB limit:
///   chunk_cols = 4GB / (100K cells × 12 bytes/nnz) ≈ 3000 columns per chunk
///   ~10 passes over the CSR data.
///
/// More memory → fewer passes → faster.
pub fn streaming_transpose(
    reader: &ScxReader,
    chunk_cols: usize,
) -> Result<CscShardIterator>;
```

#### 5.2 `scx subset` — Extract Cell/Gene Subsets

```bash
scx subset experiment.scx --filter "cell_type == 'T cell'" --output t_cells.scx
scx subset experiment.scx --genes hvg_list.txt --output hvg_subset.scx
scx subset experiment.scx --filter "tissue == 'lung'" --genes hvg_list.txt --output subset.scx
```

Internally uses `scx-engine` query pipeline + `ScxWriter`:

```rust
pub fn subset(
    input: &Path,
    output: &Path,
    filter: Option<&str>,
    gene_file: Option<&Path>,
) -> Result<SubsetStats>;
```

#### 5.3 `scx upgrade` — Format Version Migration

```bash
scx upgrade input.scx output.scx
```

Rewrites a file to the latest format version (SPEC §3.9). Useful when
`format_version` increments between releases.

```rust
pub fn upgrade(input: &Path, output: &Path) -> Result<UpgradeStats>;
```

#### 5.4 `scx benchmark` (extend existing)

The `Benchmark` subcommand **already exists** in `scx-cli` (added in Phase 2,
see `scx-cli/src/benchmark.rs`). It supports `--compare-h5ad`, `--runs`, and
`--json` flags. Phase 3 extends it with GPU-specific metrics:

```bash
# Existing (Phase 2)
scx benchmark experiment.scx --compare-h5ad experiment.h5ad --runs 5 --json

# Phase 3 additions
scx benchmark experiment.scx --gpu       # Include GPU decode timing
scx benchmark experiment.scx --simd      # Compare scalar vs SIMD decode
```

New measurements to add to `benchmark::run_benchmark()` in `scx-cli/src/benchmark.rs`:
- SIMD vs scalar decode throughput (when `--simd` flag is set)
- GPU decode throughput (when `--gpu` flag is set, requires `scx-gpu`)
- GDS vs CPU I/O path comparison (when `--gpu` + GDS is available)

#### Tests

- [ ] `scx build-csc`: output has `has_csc` flag set, CSC data matches CSR transpose
- [ ] `scx build-csc --memory-limit 100M`: works with reduced memory (more passes)
- [ ] `scx subset --filter`: output contains only matching cells
- [ ] `scx subset --genes`: output contains only specified genes
- [ ] `scx upgrade`: old format file → new format, data integrity preserved
- [ ] `scx benchmark --simd`: includes SIMD vs scalar comparison
- [ ] `scx benchmark --gpu`: includes GPU decode timing (when GPU available)

---

### Step 6: Detection Bitmap (Week 6)

**Crate**: `scx-format` (modify)
**Spec reference**: SPEC.md §5 (Detection Bitmap)
**Roadmap reference**: ROADMAP.md §3.5

> **Existing infrastructure**: `SectionType::BitmapShard (6)` already exists in
> the enum (`scx-format/src/section.rs`). The `has_bitmap` flag accessor
> (header bit 1) already exists in `header.rs`. The `roaring` crate (v0.10)
> is already a workspace dependency (used by `DeletionVectors`). This step
> adds the bitmap logic and serialization.

#### 6.1 Bitmap Section Writer/Reader

```rust
// scx-format/src/bitmap.rs (new)

use roaring::RoaringBitmap;
use scx_sparse::ScxCsr;

/// Detection bitmap: bit-packed presence/absence of gene expression.
/// Stored as Roaring Bitmap sections using SectionType::BitmapShard (6).
/// The header flag `has_bitmap` (bit 1) is set when bitmap sections exist.
///
/// Layout: one serialized Roaring Bitmap per gene (column), stored as
/// a sequence of [gene_idx: u32, bitmap_length: u32, bitmap_bytes: ...]
///
/// SPEC §5 estimates: 30K genes × 500K cells at 5% density → 100-200 MB compressed.
///
/// Use cases (per SPEC §5):
///   - Gene detection rate: POPCOUNT columns (~50× faster than CSC scan)
///   - Jaccard cell similarity: AND + POPCOUNT with SIMD
///   - Fast approximate filtering: "which cells express gene X?"
///
/// Caveat (per SPEC §5): The bitmap is for fast exploratory operations.
/// For publication-quality analyses, count-based methods on the full CSR
/// data should be used.
pub struct DetectionBitmap {
    pub bitmaps: Vec<RoaringBitmap>,  // one per gene, indexed by gene_idx
    pub n_obs: u64,
    pub n_vars: u64,
}

impl DetectionBitmap {
    /// Build from CSR data. Sets bit for every non-zero entry.
    /// Iterates CSR rows and records (gene_idx, cell_idx) pairs.
    pub fn from_csr(csr: &ScxCsr) -> Self;

    /// Gene detection rate: fraction of cells expressing each gene.
    pub fn detection_rates(&self) -> Vec<f64>;

    /// Which cells express a specific gene?
    pub fn cells_expressing(&self, gene_idx: usize) -> Vec<u64>;

    /// Jaccard similarity between two cells.
    pub fn jaccard(&self, cell_a: u64, cell_b: u64) -> f64;
}
```

#### 6.2 CLI and Python Integration

```bash
scx build-bitmap input.scx output.scx   # adds bitmap sections
```

```python
exp = scx.open("experiment.scx")
detection_rates = exp.detection_rates()  # fast POPCOUNT
expressing_cells = exp.cells_expressing("CD3D")
```

#### Tests

- [ ] Bitmap round-trip: CSR → bitmap → read back, bit-identical
- [ ] Detection rate matches `(X > 0).mean(axis=0)` from scipy
- [ ] `cells_expressing` matches `np.where(X[:, gene_idx].toarray() > 0)[0]`
- [ ] Bitmap stored in file with `has_bitmap` flag (header bit 1)
- [ ] Backward compatible: files without bitmap work normally

---

### Step 7: Conformance, Fuzzing, and Documentation

**Crates**: all
**Spec reference**: SPEC.md §14 (Conformance and Testing)
**Roadmap reference**: ROADMAP.md §3.5

#### 7.1 Conformance Test Suite

Build a set of reference `.scx` files with known contents:

```
tests/conformance/
├── reference/
│   ├── minimal.scx               # smallest valid file (1 cell, 1 gene)
│   ├── codec_none.scx             # no compression
│   ├── codec_scx1.scx             # Rice codec
│   ├── codec_zstd.scx             # Zstd codec
│   ├── multi_shard.scx            # multiple CSR shards
│   ├── with_layers.scx            # additional layers
│   ├── with_obsm.scx              # embeddings
│   ├── with_deletion_vectors.scx  # logical deletions
│   ├── multimodal_cite_seq.scx    # CITE-seq (RNA + protein)
│   ├── with_csc.scx               # CSR + CSC
│   └── with_bitmap.scx            # detection bitmap
├── expected/
│   ├── minimal_obs.arrow          # expected obs metadata
│   ├── minimal_x.npz              # expected X matrix (scipy sparse)
│   └── ...
└── test_conformance.py            # cross-language conformance tests
```

```rust
// tests/conformance/mod.rs

/// Verify that a reader correctly loads a reference file.
/// Checks: header fields, catalog integrity, shard decode, metadata,
///         checksum validation, round-trip to AnnData/Seurat/SCE.
pub fn verify_conformance(path: &Path, expected: &ExpectedData) -> Result<()>;
```

- [ ] Generate reference .scx files covering all section types
- [ ] Python conformance: `pyscx` can read all reference files
- [ ] R conformance: `rscx` can read all reference files
- [ ] Cross-language: Python-written → R-read and vice versa
- [ ] Codec conformance: known input → exact encoded bytes for Rice/FOR-BP/Delta-Golomb

#### 7.2 Fuzz Targets

Fuzz all parser entry points using `cargo-fuzz` (libFuzzer):

```
fuzz/
├── Cargo.toml
└── fuzz_targets/
    ├── fuzz_shard_decoder.rs      # arbitrary bytes → shard decode
    ├── fuzz_catalog_parser.rs     # arbitrary bytes → catalog parse
    ├── fuzz_header_parser.rs      # arbitrary bytes → header parse
    ├── fuzz_rice_decoder.rs       # arbitrary bytes → Rice decode
    ├── fuzz_forbp_decoder.rs      # arbitrary bytes → FOR-BP decode
    ├── fuzz_delta_golomb.rs       # arbitrary bytes → Delta-Golomb decode
    └── fuzz_arrow_ipc.rs          # arbitrary bytes → Arrow IPC metadata parse
```

- [ ] All fuzz targets compile and run without panics for 1M iterations
- [ ] CI integration: nightly fuzz runs with corpus accumulation
- [ ] No panics, no UB (for unsafe blocks in SIMD/GPU code)

#### 7.3 Documentation

- [ ] **API reference** (`docs/api.md`): complete Rust API docs for all public types
- [ ] **Python API reference** (`docs/python_api.md`): pyscx module docs with examples
- [ ] **R API reference** (`docs/r_api.md`): rscx package docs with examples
- [ ] **Tutorial: migration from h5ad** (`docs/tutorials/migration_guide.md`):
      step-by-step conversion guide for scanpy users
- [ ] **Tutorial: training with SCX** (`docs/tutorials/training_guide.md`):
      scVI/scGPT training with SCX loader
- [ ] **Tutorial: R workflow** (`docs/tutorials/r_workflow.md`):
      Seurat/SCE pipeline with rscx
- [ ] **Format specification cleanup**: update SPEC.md for any Phase 3 changes
- [ ] **Benchmark suite documentation**: how to reproduce all benchmarks

#### 7.4 Automated Benchmark Regression

```yaml
# .github/workflows/benchmark.yml
# Runs on every PR that touches codec/format/engine/loader
# Compares against baseline stored in benchmarks/baselines/
# Fails if any metric regresses >10%
```

- [ ] Benchmark baseline files committed for all Phase 1-3 datasets
- [ ] CI workflow runs benchmarks on PRs touching performance-critical code
- [ ] Regression detected and reported in PR comments

---

### Step 8: Integration, Polish, and Go/No-Go

**All crates**

#### 8.1 Format Spec Freeze

- [ ] Review SPEC.md for any Phase 3 changes
- [ ] Increment `format_version` if any breaking changes
- [ ] Document all v1.0 guarantees: backward compat, deprecation policy
- [ ] Tag SPEC.md as v1.0

#### 8.2 Cross-Language Conformance

- [ ] Python bindings pass full conformance suite
- [ ] R bindings pass full conformance suite
- [ ] Cross-language round-trip: Python → SCX → R → verify; R → SCX → Python → verify

#### 8.3 Final Benchmarks

Run the complete benchmark suite on:
- Phase 1 datasets (PBMC 3K, Smart-seq2, Lung 100K)
- Phase 2 10M-cell dataset
- GPU benchmarks on A100/H100 with NVMe

Consolidate all results into `benchmarks/results/phase3_benchmark.md`:

- [ ] SIMD codec speedup table (scalar vs AVX2 vs NEON)
- [ ] GPU decode throughput (CPU vs GPU)
- [ ] GDS vs CPU path throughput (Go/No-Go: GDS >2× CPU on NVMe)
- [ ] R binding performance (to_seurat latency vs Python to_anndata)
- [ ] Multimodal: CITE-seq read/write performance and file size
- [ ] Detection bitmap query performance vs CSC scan

#### 8.4 Go/No-Go Gate Validation

```python
# tests/test_go_no_go_phase3.py
def test_format_spec_frozen():
    """No breaking changes in SPEC.md since tag."""

def test_python_conformance():
    """Python bindings pass all conformance reference files."""

def test_r_conformance():
    """R bindings pass all conformance reference files."""

def test_gds_throughput():
    """GDS path >2× CPU path throughput on NVMe."""
```

---

## Summary: Task Dependencies and Sequencing

```
Step 1 (SIMD codec) ──────────────────────────────────────→ Step 7 (conformance)
Step 2 (scx-gpu: CUDA + GDS) ────────────────────────────→ Step 8 (Go/No-Go)
Step 3 (rscx: R bindings) ───────────────────────────────→ Step 7 (conformance)
Step 4 (multimodal) ──→ Step 5 (CLI extensions) ─────────→ Step 7 (conformance)
Step 6 (detection bitmap)
Step 7 (conformance + fuzzing + docs) → Step 8 (Go/No-Go)
```

Steps 1 and 2 are the critical GPU path. Step 3 (R bindings) is independent and can
proceed in parallel with Steps 1–2. Step 4 (multimodal) is independent of GPU work.
Steps 5–6 are smaller tasks that can fill gaps. Step 7 (conformance + docs) runs
throughout. Step 8 validates the Go/No-Go gate.

---

## Files Created

| File/Crate | Purpose |
|------------|---------|
| `scx-gpu/` | CUDA codec decoders, cuSPARSE interop, GDS, GPU sparse-to-dense |
| `rscx/` | R bindings (extendr), Seurat v5 + SCE interop |
| `scx-codec/src/simd.rs` | Runtime SIMD feature detection |
| `scx-codec/src/rice_avx2.rs` | AVX2 Rice decoder |
| `scx-codec/src/forbp_avx2.rs` | AVX2 FOR-BP decoder |
| `scx-codec/src/rice_neon.rs` | NEON Rice decoder |
| `scx-codec/src/forbp_neon.rs` | NEON FOR-BP decoder |
| `scx-format/src/bitmap.rs` | Detection bitmap (Roaring Bitmap) |
| `scx-format/src/spatial.rs` | Spatial index (R-tree) |
| `scx-sparse/src/transpose.rs` | Streaming CSR→CSC transpose |
| `scx-loader/src/gds_stage.rs` | GDS I/O stage (behind `gpu` feature flag) |
| `fuzz/` | Fuzz targets for all decoders and parsers |
| `tests/conformance/` | Reference .scx files and cross-language tests |
| `docs/tutorials/` | Migration guide, training guide, R workflow |

## Files Modified

| File | Changes |
|------|---------|
| `Cargo.toml` | Add `scx-gpu` and `rscx` workspace members |
| `scx-codec/src/rice.rs` | SIMD dispatch inside `rice_decode()` |
| `scx-codec/src/forbp.rs` | SIMD dispatch inside `forbp_decode_with_hint()` |
| `scx-format/src/writer.rs` | Multimodal `ModalityWriter`, bitmap/spatial sections |
| `scx-format/src/reader.rs` | Multimodal reader, bitmap reader |
| `scx-format/src/header.rs` | Add `has_modalities()` / `set_modalities()` (bit 4) |
| `scx-loader/Cargo.toml` | Optional `gpu` feature for GDS path |
| `scx-cli/src/main.rs` | Add `build-csc`, `subset`, `upgrade`, `build-bitmap` subcommands |
| `scx-cli/src/benchmark.rs` | Add `--simd` and `--gpu` flags (extend existing) |
| `pyscx/src/lib.rs` | Multimodal API, detection bitmap, MuData interop |
| `SPEC.md` | Phase 3 updates, v1.0 freeze |
| `AGENTS.md` | Phase 3 crate additions |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| CUDA kernel debugging complexity | High | Exhaustive cross-validation (GPU vs scalar) before trusting GPU path. Start with Rice (simpler), then FOR-BP |
| GDS deployment prerequisites (NVMe, nvidia-fs, ext4/XFS) | Medium | CPU path is always the default. GDS is opt-in. Detect prereqs at open time and fail gracefully |
| extendr R binding edge cases | Medium | Pin extendr 0.8.x. Test against R 4.2+ and Seurat v5 stable. Allow extra time for R-specific issues |
| Seurat v5 API instability | Medium | Pin to specific Seurat version in tests. Seurat v5 has stabilized since 2024 release |
| CITE-seq protein counts break Rice assumptions | Low | Auto-codec (Phase 2) handles this — selects Zstd for high-median modalities. Benchmark Rice vs Zstd on real CITE-seq data |
| SIMD correctness across platforms | Medium | Bit-identical validation against scalar reference for all test vectors. CI runs on both x86-64 and aarch64 |
| Nightly Rust requirement for cargo-fuzz | Low | Fuzz targets run in a separate CI job with nightly toolchain. Does not affect release builds |
| R package distribution (CRAN policies) | Medium | Initially distribute via GitHub + r-universe. CRAN submission after stabilization |

---

## Out of Scope (Phase 3)

- Rust-native PCA, kNN, UMAP accelerators (Phase 4 — optional)
- DDP multi-GPU distributed training (future)
- ATAC-seq fragment format support (future multimodal extension)
- ANS-based next-gen codec (codec_id ≥ 3, future)
- Cloud-native training from `.scxd` without local staging (future)
