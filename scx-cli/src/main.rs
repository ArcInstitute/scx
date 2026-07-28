use clap::{Parser, Subcommand};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::process;

mod append;
mod benchmark;
mod cli_utils;
mod cloud_url;
mod compact;
mod index_warnings;
use scx_convert as convert;
mod delete;
mod format;
mod info;
mod merge;
mod modify_metadata;
mod optimize;
mod query;
mod rollback;
mod set_uns;
mod shard_utils;
mod sort;
mod subset;
mod upgrade;
mod validate;
mod validators;

#[cfg(test)]
mod test_utils;

#[cfg(feature = "cloud")]
mod cloud_optimize;
#[cfg(feature = "cloud")]
mod explode;
#[cfg(feature = "cloud")]
mod pack;
#[cfg(feature = "cloud")]
mod pull;
#[cfg(feature = "cloud")]
mod push;

#[derive(Parser)]
#[command(name = "scx", about = "SCX file format tool", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Commands {
    /// Convert between h5ad/h5mu/10x/mtx and SCX formats
    Convert {
        /// Input file path
        input: PathBuf,
        /// Output file path
        output: PathBuf,
        /// Input format: h5ad, h5mu, 10x, mtx
        #[arg(long)]
        from: Option<String>,
        /// Output format: h5ad, h5mu, mtx
        #[arg(long)]
        to: Option<String>,
        /// Target rows per shard
        #[arg(long, default_value_t = scx_format_io::DEFAULT_SHARD_TARGET_ROWS, value_parser = validators::positive_u32)]
        shard_size: u32,
        /// Compression codec / intent profile. `auto` (default): cost-aware
        /// adaptive — adopts ShufDeltaZstd per framed integer shard where it wins
        /// by a margin, else the heuristic (Scx1/Zstd); float → Pcodec. `fast`:
        /// decode-speed-max (heuristic single-encode). `compact`: size-max (adopts
        /// ShufDeltaZstd on ties; framed). Also: `none`, `scx1`, `zstd`, `lz4`,
        /// `pcodec`, `shufdelta`, `compact-trial` (framed; requires --row-group-rows).
        #[arg(long, default_value = "auto")]
        codec: String,
        /// Whether to also emit a CSC sidecar at write time.
        ///
        /// `off` (default): CSR-only output, matches existing behavior.
        /// `auto`: emit a CSC sidecar when the dataset is large enough to
        ///   benefit (n_obs ≥ 50000 and n_vars ≥ 5000 by default; tune via
        ///   `SCX_CSC_AUTO_OBS_THRESHOLD` / `SCX_CSC_AUTO_VARS_THRESHOLD`).
        /// `always`: always emit a CSC sidecar (column-major shards).
        ///
        /// When omitted, an accel-ready `--index-preset` (`training` /
        /// `perturbseq`) upgrades the default to `auto`; otherwise the
        /// default is `off`. An explicit value here always wins.
        #[arg(long, value_parser = ["off", "auto", "always"])]
        csc: Option<String>,
        /// Columns per CSC shard when a CSC sidecar is emitted (default 5000).
        ///
        /// Pass `0` to disable the cap (single CSC shard, memory permitting).
        #[arg(long, default_value_t = 5000)]
        csc_cols_per_shard: usize,
        /// Row-group-frame each shard into groups of at most N rows, producing a
        /// v4 file with a multi-entry BlockIndex for codec-agnostic sub-shard
        /// random access. Framing is ON BY DEFAULT (G=256); pass `0` for the
        /// legacy unframed v3 layout (old-reader compatibility). Works with any
        /// `--codec` at no extra encode cost; use `--codec compact-trial` to also
        /// pick the smaller of the heuristic codec vs ShufDeltaZstd per shard.
        #[arg(long, value_name = "N", default_value_t = scx_format_io::DEFAULT_ROW_GROUP_ROWS)]
        row_group_rows: u32,
        /// Byte/nnz-aware row-group cap (F5): also close a group once it reaches
        /// this many non-zeros. Only meaningful with `--row-group-rows`.
        #[arg(long, value_name = "NNZ")]
        row_group_target_nnz: Option<u64>,
        /// Extract a single modality from a multi-modality SCX file
        /// when writing to h5ad. Required when `--to h5ad` is used on
        /// a multimodal SCX input; ignored otherwise.
        #[arg(long)]
        modality: Option<String>,
        /// SCX → h5ad only: drop observations whose total X UMI count is
        /// below N. Computed with one streaming pass over the CSR shards
        /// (the matrix is never materialized) and intersected with — never
        /// substituted for — the deletion-vector mask. Intended as a
        /// result-preserving low-UMI pre-trim on a RAW all-droplet file
        /// before CellBender `remove-background`. Requires the streaming
        /// export path.
        #[arg(long, value_name = "N", value_parser = validators::non_negative_f64)]
        min_counts: Option<f64>,
        /// Stream the conversion without materializing the full X
        /// matrix in memory. Defaults to true — pass `--stream=false`
        /// to opt into the legacy materializing path. Supported for
        /// h5ad ↔ SCX and h5mu ↔ SCX; combine with `--csc always` on
        /// h5ad → SCX to emit a CSC sidecar via a two-pass rebuild
        /// after the streaming write completes.
        #[arg(
            long,
            default_value_t = true,
            num_args = 0..=1,
            default_missing_value = "true",
            action = clap::ArgAction::Set,
        )]
        stream: bool,
        /// Memory budget for slab-sizing heuristics (Phase 1 dense
        /// streaming, Phase 2 transpose buffers, Phase 8c worker
        /// derate). Accepts a bare byte count or a binary-prefixed
        /// size — `K`/`M`/`G`/`T` or `KiB`/`MiB`/`GiB`/`TiB` (powers
        /// of 1024). Decimal suffixes (`KB`/`MB`/`GB`/`TB`) are
        /// rejected to avoid 1000-vs-1024 ambiguity. None = each
        /// phase's default.
        #[arg(long, value_name = "SIZE")]
        memory_budget: Option<String>,
        /// Fail conversion on the first unsupported `uns` key
        /// instead of skipping it with a warning.
        #[arg(long)]
        strict_uns: bool,
        /// Drop dense values with `|v| <= EPSILON` during
        /// sparsification. Default `0.0` keeps the equality-to-zero
        /// filtering that matches scipy `csr_matrix(dense)`.
        #[arg(long, value_name = "EPSILON", default_value_t = 0.0)]
        dense_zero_epsilon: f32,
        /// Directory for the Phase 2 external CSC → CSR transpose
        /// session (`<DIR>/scx-transpose-<pid>-<random>/`). Used
        /// only when `--memory-budget` forces the external path;
        /// the in-memory CSC route never touches disk. Defaults to
        /// the platform temp dir.
        #[arg(long, value_name = "DIR")]
        temp_dir: Option<std::path::PathBuf>,
        /// Phase 3 h5mu filter: comma-separated list of modality
        /// names to include. Unknown names fail with the available
        /// modality list. Default: include every modality.
        #[arg(long, value_name = "CSV")]
        modalities: Option<String>,
        /// Phase 3 h5mu type overrides: comma-separated
        /// `name:Type` pairs (e.g. `adt:Protein,peaks:ATAC`). Names
        /// not listed fall back to inference + a typed warning.
        /// Valid types: rna, protein, atac, spatial, methylation,
        /// custom.
        #[arg(long, value_name = "NAME:TYPE,...")]
        modality_types: Option<String>,
        /// Phase 5a: comma-separated obs columns to force-index at
        /// conversion time. Missing or unsupported columns fail the
        /// convert.
        #[arg(long, value_name = "CSV")]
        index_obs: Option<String>,
        /// Phase 5a: comma-separated var columns to force-index at
        /// conversion time. Missing or unsupported columns fail the
        /// convert.
        #[arg(long, value_name = "CSV")]
        index_var: Option<String>,
        /// Phase 5a: named column preset
        /// (`cellxgene` | `perturbseq` | `training`). Missing preset
        /// columns warn but don't fail.
        #[arg(long, value_name = "NAME")]
        index_preset: Option<String>,
        /// Phase 5a: cardinality cap for auto-detected index columns
        /// when no explicit columns or preset are supplied.
        #[arg(long, default_value_t = 1000)]
        index_auto_threshold: usize,
        /// Detection-bitmap shard generation.
        ///
        /// `off` (default) emits CSR only. `always` writes a bitmap
        /// sidecar for every CSR shard. `auto` writes a sidecar when
        /// the shard is sparse, `n_vars ≤ 1_000_000`, and the
        /// estimated bitmap size ≤ 15 % of the encoded CSR shard
        /// (ATAC modalities are always-on under `auto`). See
        /// `pyscx.detection_counts` / `cells_expressing`.
        #[arg(long, default_value = "off", value_parser = ["off", "auto", "always"])]
        bitmap: String,
        /// Streaming reader worker threads. Default: auto —
        /// `RAYON_NUM_THREADS` if set, else CPU count. `1` forces the
        /// sequential coordinator. Output is byte-identical regardless
        /// of thread count; the parallel path requires a thread-safe
        /// libhdf5 build (conda-forge default) and falls back to
        /// sequential with a warning otherwise.
        #[arg(long, value_name = "N")]
        reader_threads: Option<usize>,
        /// Reorder buffer depth between the parallel encoder
        /// pool and the ordered writer. Default 4. Larger values raise
        /// peak RSS linearly; smaller values can starve encoders.
        #[arg(long, value_name = "N", default_value_t = 4)]
        writer_queue_depth: usize,
        /// Sort-on-convert: globally reorder the cell (obs) axis by these
        /// obs columns (CSV, lexicographic; leading key first) so output
        /// CSR shards — and the predicate index — are contiguous per key.
        /// Requires a CSR or dense h5ad X (CSC errors). Applies to X,
        /// layers, obs, and obsm; obsp is dropped with a warning.
        #[arg(long, value_name = "CSV")]
        sort_by: Option<String>,
        /// Descending order for `--sort-by`.
        #[arg(long)]
        sort_reverse: bool,
        /// Convert-time grouping: cluster cells by this obs column
        /// into contiguous, never-split CSR shards (reference-first), writing a
        /// grouped layout directly — byte-equivalent to convert-then-`scx sort
        /// --group-by`, but a single write. Requires a CSR or dense h5ad X
        /// (CSC errors) and a single-modality input. `--sort-by`, if also set,
        /// supplies secondary keys after the group key. Read back with
        /// `pyscx.open(...).read_group(...)`.
        #[arg(long, value_name = "COLUMN")]
        group_by: Option<String>,
        /// Reference cells for `--group-by` (e.g. non-targeting controls):
        /// packed first and isolated in shard 0. Either a comma-separated list
        /// of `--group-by` labels, or `col:NAME` (alias `column:NAME`) to use a
        /// boolean obs column. Requires `--group-by`.
        #[arg(long, value_name = "SPEC")]
        reference: Option<String>,
        /// Byte-budget grouped sharding for `--group-by`: target shard size in
        /// bytes (group edges only) instead of `--shard-size` rows. Same size
        /// syntax as `--memory-budget`. CSR inputs only — dense/CSC fall back to
        /// row-count with a warning.
        #[arg(long, value_name = "SIZE")]
        group_target_bytes: Option<String>,
        /// Oversize threshold for `--group-target-bytes`: a single group above
        /// this becomes its own shard with a warning. Same size syntax as
        /// `--memory-budget`. Defaults to 4× `--group-target-bytes`.
        #[arg(long, value_name = "SIZE")]
        group_max_bytes: Option<String>,
        /// How to realize `--group-by`. `auto` (default) routes by source
        /// density: CSR → one-pass streaming (cheaper); dense → two-pass
        /// (plain convert + `scx sort`, ~4–5× faster than the one-pass
        /// random-row gather over a dense matrix). `one` / `two` force it.
        #[arg(long, default_value = "auto", value_parser = ["auto", "one", "two"])]
        group_pass: String,
    },
    /// Display SCX file information
    Info {
        /// SCX file, exploded `.scxd/` directory, or cloud URL (gs://, s3://, …) to inspect
        source: String,
        /// Output all info as JSON
        #[arg(long)]
        json: bool,
        /// Show manifest version history (local packed files only)
        #[arg(long)]
        history: bool,
    },
    /// Validate header + per-section BLAKE3 checksums (deeper than `scx info`)
    Validate {
        /// SCX file to validate
        file: PathBuf,
        /// Print checksum values and deep-check errors
        #[arg(long)]
        verbose: bool,
        /// Decode sparse shards and validate the v3 canonical CSR invariant
        #[arg(long)]
        deep: bool,
    },
    /// Append cells from another SCX file
    Append {
        /// Target SCX file to append to
        target: PathBuf,
        /// Source SCX file containing cells to append
        source: PathBuf,
        /// Modality name to append into. NOTE: append into a multimodal
        /// target is not yet supported (deferred) and is rejected — extract
        /// a modality with `scx subset --modality NAME`, append to the
        /// single-modality file, then `scx merge` back. Optional on
        /// single-modality files (the only supported case) — defaults to the
        /// global / primary modality.
        #[arg(long)]
        modality: Option<String>,
        /// Compression codec for new shards: auto, none, scx1, zstd, lz4, pcodec, shufdelta
        #[arg(long, default_value = "auto")]
        codec: String,
        /// Target rows per shard (must be > 0)
        #[arg(long, default_value_t = NonZeroU32::new(scx_format_io::DEFAULT_SHARD_TARGET_ROWS).unwrap())]
        shard_size: NonZeroU32,
        /// Rebuild the CSC sidecar after appending (drops + re-emits via
        /// `scx build-csc`). Without this flag, append drops the CSC
        /// sidecar with a warning — the row layout no longer matches.
        /// The rebuild's transpose memory budget defaults to 4 GiB; override
        /// with `--csc-memory-limit`.
        #[arg(long)]
        rebuild_csc: bool,
        /// Maximum columns per emitted CSC shard when `--rebuild-csc` is
        /// set (default: 5000). Ignored without `--rebuild-csc`.
        #[arg(long, default_value_t = 5000)]
        csc_cols_per_shard: usize,
        /// Transpose memory budget for the `--rebuild-csc` pass (default 4G).
        /// Accepts a binary-prefixed size (`K`/`M`/`G`/`T` or `KiB`..`TiB`);
        /// decimal `KB`/`MB`/`GB` is rejected. Ignored without `--rebuild-csc`.
        #[arg(long, default_value = "4G")]
        csc_memory_limit: String,
        /// Comma-separated obs columns to force-index after appending.
        /// Mirrors `scx convert --index-obs`; without this flag, any
        /// pre-existing predicate-index sections remain in place but
        /// cover only the pre-append rows.
        #[arg(long, value_name = "CSV")]
        index_obs: Option<String>,
        /// Comma-separated var columns to force-index after appending.
        #[arg(long, value_name = "CSV")]
        index_var: Option<String>,
        /// Named column preset (`cellxgene` | `perturbseq` | `training`).
        #[arg(long, value_name = "NAME")]
        index_preset: Option<String>,
        /// Cardinality cap for auto-detected index columns. Pass this
        /// flag alone (without `--index-obs`/`--index-var`/`--index-preset`)
        /// to ask the engine to auto-detect low-cardinality categorical
        /// columns at the given threshold; omit it to leave any
        /// pre-existing predicate-index sections in place (stale on
        /// appended rows).
        #[arg(long, value_name = "N")]
        index_auto_threshold: Option<usize>,
    },
    /// Logically delete cells matching a predicate
    Delete {
        /// SCX file to modify
        file: PathBuf,
        /// Predicate expression to select cells for deletion
        #[arg(long)]
        filter: String,
        /// Show count of matching cells without deleting
        #[arg(long)]
        dry_run: bool,
    },
    /// Upgrade a file in place: re-encode + canonicalize CSR shards to
    /// format_version 3 (single-modality; preserves obs/
    /// var/obsm/uns/indexes). Drops the CSC sidecar — rerun `scx build-csc`.
    Optimize {
        /// SCX file to optimize
        input: PathBuf,
        /// Output path for the optimized file (may equal <INPUT> for in-place)
        output: PathBuf,
        /// Overwrite output if it exists
        #[arg(long)]
        force: bool,
        /// Per-shard codec: `auto` (Scx1 for low-median integer counts, else
        /// Zstd) or `scx1` (force Scx1 on every integer shard); or `shufdelta` /
        /// Codec / intent profile. `auto` (default): cost-aware adaptive.
        /// `fast`: decode-speed-max heuristic. `compact`: size-max (adopts
        /// ShufDeltaZstd on ties). `scx1`, `shufdelta`, `compact-trial` are
        /// explicit forces. `compact`/`shufdelta`/`compact-trial` require
        /// `--row-group-rows`; framed shards use the block index for random
        /// access. Other codecs are rejected. Default: auto.
        #[arg(long, default_value = "auto", value_parser = ["auto", "fast", "compact", "scx1", "shufdelta", "compact-trial"])]
        codec: String,
        /// Row-group-frame each re-encoded shard into groups of at most N rows,
        /// producing a v4 file with a multi-entry BlockIndex for codec-agnostic
        /// sub-shard random access. Framing is ON BY DEFAULT (G=256) — `scx
        /// optimize` upgrades an unframed file to framed; pass `0` to keep the
        /// legacy unframed (v3) layout. Required (> 0) for `--codec shufdelta` /
        /// `--codec compact` / `--codec compact-trial`.
        #[arg(long, value_name = "N", default_value_t = scx_format_io::DEFAULT_ROW_GROUP_ROWS)]
        row_group_rows: u32,
        /// Byte/nnz-aware row-group cap: also close a group once it reaches this
        /// many non-zeros. Only meaningful with `--row-group-rows`.
        #[arg(long, value_name = "NNZ")]
        row_group_target_nnz: Option<u64>,
        /// Migrate a legacy single-section obs table to the sharded
        /// `ObsMetadataShard` layout: `auto` (shard when n_obs >
        /// shard_target_rows — the from_anndata threshold), `always`, or
        /// `off` (keep the single section; today's faithful 1:1 copy).
        /// Already-sharded obs is preserved as-is regardless. Default: auto.
        #[arg(long = "shard-obs", default_value = "auto", value_parser = ["off", "auto", "always"])]
        shard_obs: String,
    },
    /// Rewrite file reclaiming space from deletions
    Compact {
        /// SCX file to compact
        input: PathBuf,
        /// Output path for compacted file
        output: PathBuf,
        /// Overwrite output if it exists
        #[arg(long)]
        force: bool,
        /// Rebuild the CSC sidecar on the compacted output (drops +
        /// re-emits via `scx build-csc`). Without this flag, compact
        /// drops the CSC sidecar with a warning — the row layout no
        /// longer matches after deletion-vector application.
        /// The rebuild's transpose memory budget defaults to 4 GiB; override
        /// with `--csc-memory-limit`.
        #[arg(long)]
        rebuild_csc: bool,
        /// Maximum columns per emitted CSC shard when `--rebuild-csc`
        /// is set (default: 5000). Ignored without `--rebuild-csc`.
        #[arg(long, default_value_t = 5000)]
        csc_cols_per_shard: usize,
        /// Transpose memory budget for the `--rebuild-csc` pass (default 4G).
        /// Accepts a binary-prefixed size (`K`/`M`/`G`/`T` or `KiB`..`TiB`);
        /// decimal `KB`/`MB`/`GB` is rejected. Ignored without `--rebuild-csc`.
        #[arg(long, default_value = "4G")]
        csc_memory_limit: String,
        /// Comma-separated obs columns to force-index on the compacted
        /// output. Mirrors `scx convert --index-obs`. Without this
        /// flag, compact drops any input predicate-index sections (the
        /// row layout is re-sharded against the post-deletion row
        /// count).
        #[arg(long, value_name = "CSV")]
        index_obs: Option<String>,
        /// Comma-separated var columns to force-index on the compacted output.
        #[arg(long, value_name = "CSV")]
        index_var: Option<String>,
        /// Named column preset (`cellxgene` | `perturbseq` | `training`).
        #[arg(long, value_name = "NAME")]
        index_preset: Option<String>,
        /// Cardinality cap for auto-detected index columns. Pass this
        /// flag alone to ask the engine to auto-detect low-cardinality
        /// categorical columns at the given threshold; omit all
        /// `--index-*` flags to drop predicate indexes on the compacted
        /// output (current default).
        #[arg(long, value_name = "N")]
        index_auto_threshold: Option<usize>,
        /// Rewrite obs metadata as row-sharded `ObsMetadataShard` sections
        /// (legacy single-section → sharded). Migrates files written
        /// before sharded obs metadata existed; idempotent on already-
        /// sharded inputs.
        #[arg(long)]
        reshape_obs: bool,
    },
    /// Globally reorder cells (obs axis) by an obs key for query locality
    Sort {
        /// Input SCX file (local path)
        input: PathBuf,
        /// Output SCX file path
        output: PathBuf,
        /// Comma-separated obs columns to sort by, lexicographic in order
        /// (the leading key gets the full X-read-locality benefit). Optional
        /// when `--group-by` is given (which becomes the leading key).
        #[arg(long, value_name = "CSV")]
        by: Option<String>,
        /// Sort descending on all keys (ignored with `--group-by`).
        #[arg(long)]
        reverse: bool,
        /// Overwrite output if it exists
        #[arg(long)]
        force: bool,
        /// Target rows per shard in the output file
        #[arg(long, default_value_t = scx_format_io::DEFAULT_SHARD_TARGET_ROWS, value_parser = validators::positive_u32)]
        shard_size: u32,
        /// Compression codec for output: auto, none, scx1, zstd, lz4, pcodec, shufdelta
        #[arg(long, default_value = "auto")]
        codec: String,
        /// Comma-separated obs columns to also index on the output (the sort
        /// key is always indexed so its shard ranges are contiguous).
        #[arg(long, value_name = "CSV")]
        index_obs: Option<String>,
        /// Comma-separated var columns to index on the output.
        #[arg(long, value_name = "CSV")]
        index_var: Option<String>,
        /// Named column preset (`cellxgene` | `perturbseq` | `training`).
        #[arg(long, value_name = "NAME")]
        index_preset: Option<String>,
        /// Cardinality cap for auto-detected index columns.
        #[arg(long, value_name = "N")]
        index_auto_threshold: Option<usize>,
        /// Spill / partition memory budget for the external sort (binary
        /// suffix `K`/`M`/`G`/`T` or `KiB`..`TiB`; decimal `KB`/`MB` rejected).
        /// Without it the in-memory path is used.
        #[arg(long, value_name = "SIZE")]
        memory_budget: Option<String>,
        /// Directory for the external sort's spill files (default: system temp).
        #[arg(long, value_name = "DIR")]
        temp_dir: Option<PathBuf>,
        /// Detection-bitmap policy for the output: off (drop, default),
        /// auto (rebuild when sparse), or always. Mirrors `scx convert --bitmap`.
        #[arg(long, default_value = "off", value_parser = ["off", "auto", "always"])]
        bitmap: String,
        /// Rebuild the CSC sidecar on the sorted output (the reorder
        /// invalidates the column-major row indices, so it is dropped by
        /// default with a warning).
        #[arg(long)]
        rebuild_csc: bool,
        /// Maximum columns per emitted CSC shard when `--rebuild-csc` is set.
        #[arg(long, default_value_t = 5000)]
        csc_cols_per_shard: usize,
        /// Transpose memory budget for the `--rebuild-csc` pass (default 4G).
        #[arg(long, default_value = "4G")]
        csc_memory_limit: String,
        /// F1: obs column whose label clusters rows into shards (condition /
        /// perturbation grouping). Forced to be the leading sort key; engages
        /// the byte-budget group planner and writes a `group_index` sidecar.
        /// `--reverse` is ignored when set. Restricted to single-modality
        /// inputs in v1.
        #[arg(long, value_name = "COL")]
        group_by: Option<String>,
        /// F1: which rows are reference cells (e.g. "non-targeting"), packed
        /// first / isolated in shard 0. A comma-separated label list, or
        /// `col:NAME` (alias `column:NAME`) for a boolean obs column. Requires
        /// `--group-by`.
        #[arg(long, value_name = "LABELS|col:NAME")]
        reference: Option<String>,
        /// F1: byte budget per shard for the group bin-packer (binary suffix
        /// `K`/`M`/`G`/`T`). Without it the planner packs by `--shard-size`
        /// rows. Only with `--group-by`.
        #[arg(long, value_name = "SIZE")]
        group_target_bytes: Option<String>,
        /// F1: oversize threshold — a single group exceeding it gets its own
        /// shard plus a warning. Defaults to a multiple of the target.
        #[arg(long, value_name = "SIZE")]
        group_max_bytes: Option<String>,
        /// F6: byte cap on the emitter's per-shard accumulation buffer in grouped
        /// mode (binary suffix `K`/`M`/`G`/`T`). When the accumulated CSR of the
        /// current shard reaches this, it is sub-flushed as a standalone shard
        /// *within* a group — bounding grouped-write peak memory at one block
        /// instead of one whole group (a large group is split across shards).
        /// Default 256M. `0` disables the sub-flush. Only with `--group-by`.
        #[arg(long, value_name = "SIZE")]
        group_write_block_bytes: Option<String>,
    },
    /// Revert to a previous manifest version
    Rollback {
        /// SCX file to roll back
        file: PathBuf,
        /// Target manifest sequence number (default: previous version)
        #[arg(long)]
        to_seq: Option<u64>,
    },
    /// Replace the `uns` block in place, without re-encoding X (cheapest path)
    SetUns {
        /// SCX file to modify
        file: PathBuf,
        /// JSON file whose contents become the new `uns` (replace, not merge)
        #[arg(long)]
        uns: PathBuf,
    },
    /// Replace metadata sections (uns/obs/var/obsm/varm) in place, no X re-encode
    ModifyMetadata {
        /// SCX file to modify
        file: PathBuf,
        /// JSON file replacing the whole `uns` block
        #[arg(long)]
        uns: Option<PathBuf>,
        /// Parquet file replacing obs metadata (num_rows must equal n_obs)
        #[arg(long)]
        obs: Option<PathBuf>,
        /// Parquet file replacing var metadata (num_rows must equal n_vars)
        #[arg(long)]
        var: Option<PathBuf>,
        /// Replace a named obsm matrix from a 2D .npy file: `name=path.npy`
        /// (repeatable). Rows must equal n_obs.
        #[arg(long, value_name = "NAME=PATH.npy")]
        obsm: Vec<String>,
        /// Replace a named varm matrix from a 2D .npy file: `name=path.npy`
        /// (repeatable). Rows must equal n_vars.
        #[arg(long, value_name = "NAME=PATH.npy")]
        varm: Vec<String>,
        /// Comma-separated obs columns to force-index when obs is replaced.
        #[arg(long, value_name = "CSV")]
        index_obs: Option<String>,
        /// Comma-separated var columns to force-index when var is replaced.
        #[arg(long, value_name = "CSV")]
        index_var: Option<String>,
        /// Named column preset (`cellxgene` | `perturbseq` | `training`).
        #[arg(long, value_name = "NAME")]
        index_preset: Option<String>,
        /// Cardinality cap for auto-detected index columns (obs/var rebuild).
        #[arg(long, value_name = "N")]
        index_auto_threshold: Option<usize>,
        /// Target modality. Only the global modality is supported today;
        /// passing a name errors (multimodal metadata replace is deferred).
        #[arg(long)]
        modality: Option<String>,
    },
    /// Merge multiple SCX files into one
    Merge {
        /// Input SCX files to merge (at least 2)
        inputs: Vec<PathBuf>,
        /// Output path for merged file. Required, and passed as a flag (not a
        /// positional argument) because `inputs` is variadic — a trailing path
        /// would otherwise be read as another input.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Rebuild the CSC sidecar on the merged output (drops +
        /// re-emits via `scx build-csc`). Without this flag, merge
        /// drops any input CSC sidecars with a warning.
        /// The rebuild's transpose memory budget defaults to 4 GiB; override
        /// with `--csc-memory-limit`.
        #[arg(long)]
        rebuild_csc: bool,
        /// Maximum columns per emitted CSC shard when `--rebuild-csc`
        /// is set (default: 5000). Ignored without `--rebuild-csc`.
        #[arg(long, default_value_t = 5000)]
        csc_cols_per_shard: usize,
        /// Transpose memory budget for the `--rebuild-csc` pass (default 4G).
        /// Accepts a binary-prefixed size (`K`/`M`/`G`/`T` or `KiB`..`TiB`);
        /// decimal `KB`/`MB`/`GB` is rejected. Ignored without `--rebuild-csc`.
        #[arg(long, default_value = "4G")]
        csc_memory_limit: String,
        /// Comma-separated obs columns to force-index on the merged
        /// output. Mirrors `scx convert --index-obs`. Without this
        /// flag, the merged output has NO predicate-index sections —
        /// query-time `filter_obs` pushdown falls back to a full scan.
        #[arg(long, value_name = "CSV")]
        index_obs: Option<String>,
        /// Comma-separated var columns to force-index on the merged output.
        #[arg(long, value_name = "CSV")]
        index_var: Option<String>,
        /// Named column preset (`cellxgene` | `perturbseq` | `training`).
        #[arg(long, value_name = "NAME")]
        index_preset: Option<String>,
        /// Cardinality cap for auto-detected index columns. Pass this
        /// flag alone to ask the engine to auto-detect low-cardinality
        /// categorical columns at the given threshold; omit all
        /// `--index-*` flags to drop predicate indexes on the merged
        /// output (current default).
        #[arg(long, value_name = "N")]
        index_auto_threshold: Option<usize>,
        /// Skip the column-by-column var-identity check across
        /// inputs (only `n_vars` and modality structure are
        /// validated). Use when you've already pre-aligned the gene
        /// axis upstream — otherwise mismatched var rows produce
        /// silent column-axis corruption in the merged X matrix.
        #[arg(long, default_value_t = false)]
        assume_identical_var: bool,
        /// Skip the obs schema identity check across inputs (column
        /// names and dtypes, normalised through the logical-lossy
        /// schema). Use when you've already verified the obs surface
        /// upstream — otherwise mismatched obs columns produce a
        /// merged file that errors at `read_obs()` time after the
        /// output is renamed into place.
        #[arg(long, default_value_t = false)]
        assume_identical_obs: bool,
        /// Policy for combining `uns` (unstructured metadata) across
        /// inputs. `first` (default) keeps the first input's uns
        /// verbatim and drops the rest; `require-equal` errors if any
        /// input's uns differs; `namespace` writes a JSON object with
        /// each input's uns under an `input_N` key; `summary` keeps
        /// the first input's uns and appends a `_scx_uns_conflicts`
        /// array listing per-input disagreements.
        #[arg(long, value_name = "POLICY")]
        uns_policy: Option<String>,
        /// Sorted k-way merge: globally reorder the merged cell (obs) axis
        /// by these obs columns (CSV, lexicographic). Each input must
        /// already be a sorted run by this key (e.g. from `scx convert
        /// --sort-by`); an unsorted input errors. Reorders obs / X /
        /// layers; var-axis preserved; obsm unsupported (errors).
        #[arg(long, value_name = "CSV")]
        sort_by: Option<String>,
        /// Descending order for `--sort-by`.
        #[arg(long)]
        sort_reverse: bool,
    },
    /// Query cells by predicate
    Query {
        /// SCX file path or cloud URL (e.g. `gs://bucket/atlas.scxd/`)
        source: String,
        /// Restrict the query to one modality of a multimodal file (by name).
        /// X / `--select-genes` resolve against that modality's var; the obs
        /// predicate stays on the shared global obs axis. Works on local and
        /// cloud sources. Required for `scx query` on a multimodal file.
        #[arg(long)]
        modality: Option<String>,
        /// Obs predicate expression (positional). Alternatively pass it via
        /// `--filter` to match `scx subset` / `scx delete`. Provide one form,
        /// not both.
        filter: Option<String>,
        /// Obs predicate expression (flag form, consistent with
        /// `scx subset` / `scx delete`).
        #[arg(long = "filter", value_name = "EXPR", conflicts_with = "filter")]
        filter_flag: Option<String>,
        /// Print matching cell count only
        #[arg(long)]
        count: bool,
        /// Write matching cells to a new SCX file
        #[arg(long)]
        output: Option<PathBuf>,
        /// File containing gene indices for projection (one per line)
        #[arg(long)]
        select_genes: Option<PathBuf>,
        /// Normalization target sum
        #[arg(long)]
        normalize: Option<f64>,
        /// Apply log1p transformation
        #[arg(long)]
        log1p: bool,
        /// Limit number of returned cells (output-only). Caps the rows
        /// written/returned; it does NOT affect `--count`, which always
        /// reports the true number of matching cells.
        #[arg(long)]
        limit: Option<usize>,
        /// JSON output (for --count)
        #[arg(long)]
        json: bool,
        /// Print a minimal query plan summary on stderr (parsed
        /// predicate, total/skipped/candidate shards, candidate rows,
        /// matched rows). Useful to distinguish Level 1 (catalog-stats)
        /// pushdown from Level 2 (PredicateIndex row-mask) narrowing.
        #[arg(long)]
        explain: bool,
    },
    /// Run read/query benchmarks
    Benchmark {
        /// SCX file to benchmark
        file: PathBuf,
        /// Compare against h5ad read performance
        #[arg(long)]
        compare_h5ad: Option<PathBuf>,
        /// Number of benchmark runs
        #[arg(long, default_value = "5", value_parser = validators::positive_usize)]
        runs: usize,
        /// Output results as JSON
        #[arg(long)]
        json: bool,
    },
    /// Rewrite file with front-of-file catalog for cloud access
    #[cfg(feature = "cloud")]
    CloudOptimize {
        /// SCX file to optimize
        input: PathBuf,
        /// Output path (default: rewrite in-place via atomic rename)
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Explode a packed .scx into an .scxd directory
    #[cfg(feature = "cloud")]
    Explode {
        /// SCX file to explode
        input: PathBuf,
        /// Output directory (must end in .scxd)
        output: PathBuf,
    },
    /// Pack an .scxd directory into a packed .scx file
    #[cfg(feature = "cloud")]
    Pack {
        /// Input directory (.scxd)
        input: PathBuf,
        /// Output .scx file
        output: PathBuf,
    },
    /// Download from cloud/local exploded .scxd and pack into local .scx
    #[cfg(feature = "cloud")]
    Pull {
        /// Source URL or path (e.g. gs://bucket/experiment.scxd/)
        source: String,
        /// Output .scx file path
        dest: PathBuf,
        /// Number of parallel download tasks
        #[arg(long, default_value = "8")]
        parallelism: usize,
        /// Skip front catalog in output (not cloud-ready)
        #[arg(long)]
        no_cloud_ready: bool,
        /// Predicate expression to selectively download matching shards
        #[arg(long)]
        filter: Option<String>,
        /// Filter granularity for selective pulls: 'shard' (currently the
        /// only supported mode; cell-granular filtering is a follow-on).
        #[arg(long, default_value = "shard", value_parser = ["shard"])]
        filter_mode: String,
    },
    /// Upload a local .scx file to cloud/local as exploded .scxd directory
    #[cfg(feature = "cloud")]
    Push {
        /// Source .scx file path
        source: PathBuf,
        /// Destination URL or path (e.g. gs://bucket/experiment.scxd/)
        dest: String,
        /// Number of parallel upload tasks
        #[arg(long, default_value = "8")]
        parallelism: usize,
    },
    /// Build CSC (column-major) shards from existing CSR data
    BuildCsc {
        /// Input SCX file (must have CSR shards)
        input: PathBuf,
        /// Output SCX file (will contain both CSR and CSC shards)
        output: PathBuf,
        /// Maximum memory for the transpose working set (default: 4G).
        /// Accepts a bare byte count or a binary-prefixed size —
        /// `K`/`M`/`G`/`T` or `KiB`/`MiB`/`GiB`/`TiB` (powers of 1024);
        /// decimal `KB`/`MB`/`GB`/`TB` is rejected as ambiguous.
        #[arg(long, default_value = "4G")]
        memory_limit: String,
        /// Overwrite output if it exists
        #[arg(long)]
        force: bool,
        /// Maximum columns per emitted CSC shard (default: 5000).
        ///
        /// Drives multi-shard CSC layouts: the writer emits ceil(n_vars
        /// / N) CSC shards, each covering a contiguous column range.
        /// Smaller values enable finer-grained column-range pushdown at
        /// read time but produce more shards. Pass 0 for no cap (single
        /// shard, memory permitting).
        #[arg(long, default_value_t = 5000)]
        csc_cols_per_shard: usize,
    },
    /// Extract a subset of cells and/or genes into a new SCX file
    Subset {
        /// Input SCX file
        input: PathBuf,
        /// Output SCX file path (optional with --dry-run)
        output: Option<PathBuf>,
        /// Obs predicate expression to filter cells
        #[arg(long)]
        filter: Option<String>,
        /// File containing gene names or numeric indices (one per line) for column projection
        #[arg(long)]
        genes: Option<PathBuf>,
        /// On a multimodal input, extract a single modality by name
        /// to a new single-modality v2 file. When set with `--genes`,
        /// scopes the gene filter to that modality's index space.
        #[arg(long)]
        modality: Option<String>,
        /// Show matching count without writing output
        #[arg(long)]
        dry_run: bool,
        /// Target rows per shard in the output file
        #[arg(long, default_value_t = scx_format_io::DEFAULT_SHARD_TARGET_ROWS, value_parser = validators::positive_u32)]
        shard_size: u32,
        /// Compression codec for output: auto, none, scx1, zstd, lz4, pcodec, shufdelta
        #[arg(long, default_value = "auto")]
        codec: String,
        /// Rebuild the CSC sidecar on the subset output (drops +
        /// re-emits via `scx build-csc` against the projected CSR).
        /// Without this flag, subset drops any input CSC sidecar
        /// with a warning — the row/column index space changes.
        /// The rebuild's transpose memory budget defaults to 4 GiB; override
        /// with `--csc-memory-limit`.
        #[arg(long)]
        rebuild_csc: bool,
        /// Maximum columns per emitted CSC shard when `--rebuild-csc`
        /// is set (default: 5000). Ignored without `--rebuild-csc`.
        #[arg(long, default_value_t = 5000)]
        csc_cols_per_shard: usize,
        /// Transpose memory budget for the `--rebuild-csc` pass (default 4G).
        /// Accepts a binary-prefixed size (`K`/`M`/`G`/`T` or `KiB`..`TiB`);
        /// decimal `KB`/`MB`/`GB` is rejected. Ignored without `--rebuild-csc`.
        #[arg(long, default_value = "4G")]
        csc_memory_limit: String,
    },
    /// Upgrade an SCX file to the latest format version
    Upgrade {
        /// Input SCX file
        input: PathBuf,
        /// Output SCX file path (default: separate output)
        output: Option<PathBuf>,
        /// Upgrade in-place via atomic rename
        #[arg(long)]
        in_place: bool,
    },
}

/// Restore the default `SIGPIPE` disposition (`SIG_DFL`).
///
/// The Rust runtime sets `SIGPIPE` to `SIG_IGN` at startup, so writing to a
/// closed pipe (`scx info file.scx | head`, `| less` then `q`) returns `EPIPE`,
/// which the `println!`/`print!` machinery turns into a panic + backtrace hint
/// on stderr (report E3). Resetting to `SIG_DFL` makes the process terminate
/// silently on a broken pipe — exit status 141 (128 + SIGPIPE), the same as
/// every standard Unix filter. No-op on non-Unix targets.
fn reset_sigpipe() {
    #[cfg(unix)]
    // SAFETY: `signal(2)` with `SIG_DFL` is async-signal-safe and called once
    // before any threads are spawned; restoring the default disposition has no
    // memory-safety implications.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

/// Cloud subcommands gated behind `--features cloud`. Listed here (rather than
/// derived) so a build *without* the cloud feature can still recognise them and
/// emit a helpful hint instead of clap's bare "unrecognized subcommand" — the
/// discoverability gap from report D3.
#[cfg(not(feature = "cloud"))]
const CLOUD_SUBCOMMANDS: &[&str] = &["pull", "push", "explode", "pack", "cloud-optimize"];

/// Parse the CLI. On a non-cloud build, an attempt to invoke a cloud subcommand
/// (`scx pull …`) fails clap's subcommand match; before deferring to clap's
/// normal error/exit, print a one-line hint that the command exists but needs a
/// `--features cloud` build, so the user can tell it apart from a typo (D3).
fn parse_cli_or_exit() -> Cli {
    match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            #[cfg(not(feature = "cloud"))]
            if e.kind() == clap::error::ErrorKind::InvalidSubcommand {
                // No global options precede the subcommand, so argv[1] is the
                // offending token.
                if let Some(sub) = std::env::args().nth(1) {
                    if CLOUD_SUBCOMMANDS.contains(&sub.as_str()) {
                        eprintln!(
                            "note: `{sub}` is a cloud subcommand and is not compiled into this \
                             build. Rebuild with `--features cloud` (or install the cloud-enabled \
                             binary) to use it."
                        );
                    }
                }
            }
            e.exit();
        }
    }
}

fn main() {
    reset_sigpipe();

    // Initialize the `log` sink. Default severity is `info`; override with
    // `RUST_LOG=scx=debug`, `RUST_LOG=scx_loader=warn`, etc.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = parse_cli_or_exit();

    let result = match cli.command {
        Commands::Convert {
            input,
            output,
            from,
            to,
            shard_size,
            codec,
            csc,
            csc_cols_per_shard,
            row_group_rows,
            row_group_target_nnz,
            modality,
            min_counts,
            stream,
            memory_budget,
            strict_uns,
            dense_zero_epsilon,
            temp_dir,
            modalities,
            modality_types,
            index_obs,
            index_var,
            index_preset,
            index_auto_threshold,
            bitmap,
            reader_threads,
            writer_queue_depth,
            sort_by,
            sort_reverse,
            group_by,
            reference,
            group_target_bytes,
            group_max_bytes,
            group_pass,
        } => {
            // Resolve the CSC policy: an explicit `--csc` always wins;
            // otherwise an accel-ready `--index-preset` upgrades the
            // default to `auto` (column substrate for DE/pseudobulk).
            // Shared with pyscx via `scx_engine::index::resolve_csc_policy`.
            let csc =
                scx_engine::index::resolve_csc_policy(csc.as_deref(), index_preset.as_deref());
            run_convert(
                &input,
                &output,
                from.as_deref(),
                to.as_deref(),
                shard_size,
                &codec,
                &csc,
                csc_cols_per_shard,
                row_group_rows,
                row_group_target_nnz,
                modality.as_deref(),
                min_counts,
                stream,
                memory_budget.as_deref(),
                strict_uns,
                dense_zero_epsilon,
                temp_dir,
                modalities.as_deref(),
                modality_types.as_deref(),
                index_obs.as_deref(),
                index_var.as_deref(),
                index_preset,
                index_auto_threshold,
                &bitmap,
                reader_threads,
                writer_queue_depth,
                sort_by.as_deref(),
                sort_reverse,
                group_by.as_deref(),
                reference.as_deref(),
                group_target_bytes.as_deref(),
                group_max_bytes.as_deref(),
                &group_pass,
            )
        }
        Commands::Info {
            source,
            json,
            history,
        } => info::run_info(&source, json, history),
        Commands::Validate {
            file,
            verbose,
            deep,
        } => match validate::run_validate(&file, verbose, deep) {
            Ok(all_passed) => {
                if all_passed {
                    Ok(())
                } else {
                    process::exit(1);
                }
            }
            Err(e) => Err(e),
        },
        Commands::Append {
            target,
            source,
            modality,
            codec,
            shard_size,
            rebuild_csc,
            csc_cols_per_shard,
            csc_memory_limit,
            index_obs,
            index_var,
            index_preset,
            index_auto_threshold,
        } => append::run_append(
            &target,
            &source,
            modality.as_deref(),
            &codec,
            shard_size,
            rebuild_csc,
            csc_cols_per_shard,
            &csc_memory_limit,
            parse_index_columns(index_obs.as_deref()),
            parse_index_columns(index_var.as_deref()),
            index_preset.filter(|s| !s.trim().is_empty()),
            index_auto_threshold,
        ),
        Commands::Delete {
            file,
            filter,
            dry_run,
        } => delete::run_delete(&file, &filter, dry_run),
        Commands::Optimize {
            input,
            output,
            force,
            codec,
            row_group_rows,
            row_group_target_nnz,
            shard_obs,
        } => optimize::run_optimize(
            &input,
            &output,
            force,
            &codec,
            // Framing on by default; `run_optimize` normalizes `Some(0)` → None
            // (the unframed v3 opt-out).
            Some(row_group_rows),
            row_group_target_nnz,
            &shard_obs,
        ),
        Commands::Compact {
            input,
            output,
            force,
            rebuild_csc,
            csc_cols_per_shard,
            csc_memory_limit,
            index_obs,
            index_var,
            index_preset,
            index_auto_threshold,
            reshape_obs,
        } => compact::run_compact(
            &input,
            &output,
            force,
            rebuild_csc,
            csc_cols_per_shard,
            &csc_memory_limit,
            parse_index_columns(index_obs.as_deref()),
            parse_index_columns(index_var.as_deref()),
            index_preset.filter(|s| !s.trim().is_empty()),
            index_auto_threshold,
            reshape_obs,
        ),
        Commands::Sort {
            input,
            output,
            by,
            reverse,
            force,
            shard_size,
            codec,
            index_obs,
            index_var,
            index_preset,
            index_auto_threshold,
            memory_budget,
            temp_dir,
            bitmap,
            rebuild_csc,
            csc_cols_per_shard,
            csc_memory_limit,
            group_by,
            reference,
            group_target_bytes,
            group_max_bytes,
            group_write_block_bytes,
        } => sort::run_sort(
            &input,
            &output,
            parse_index_columns(by.as_deref()),
            reverse,
            force,
            shard_size,
            &codec,
            parse_index_columns(index_obs.as_deref()),
            parse_index_columns(index_var.as_deref()),
            index_preset.filter(|s| !s.trim().is_empty()),
            index_auto_threshold,
            memory_budget,
            temp_dir,
            &bitmap,
            rebuild_csc,
            csc_cols_per_shard,
            &csc_memory_limit,
            group_by,
            reference,
            group_target_bytes,
            group_max_bytes,
            group_write_block_bytes,
        ),
        Commands::Rollback { file, to_seq } => rollback::run_rollback(&file, to_seq),
        Commands::SetUns { file, uns } => set_uns::run_set_uns(&file, &uns),
        Commands::ModifyMetadata {
            file,
            uns,
            obs,
            var,
            obsm,
            varm,
            index_obs,
            index_var,
            index_preset,
            index_auto_threshold,
            modality,
        } => modify_metadata::run_modify_metadata(
            &file,
            uns.as_deref(),
            obs.as_deref(),
            var.as_deref(),
            &obsm,
            &varm,
            parse_index_columns(index_obs.as_deref()),
            parse_index_columns(index_var.as_deref()),
            index_preset.filter(|s| !s.trim().is_empty()),
            index_auto_threshold,
            modality,
        ),
        Commands::Merge {
            inputs,
            output,
            rebuild_csc,
            csc_cols_per_shard,
            csc_memory_limit,
            index_obs,
            index_var,
            index_preset,
            index_auto_threshold,
            assume_identical_var,
            assume_identical_obs,
            uns_policy,
            sort_by,
            sort_reverse,
        } => match output {
            Some(output) => merge::run_merge(
                &inputs,
                &output,
                rebuild_csc,
                csc_cols_per_shard,
                &csc_memory_limit,
                parse_index_columns(index_obs.as_deref()),
                parse_index_columns(index_var.as_deref()),
                index_preset.filter(|s| !s.trim().is_empty()),
                index_auto_threshold,
                assume_identical_var,
                assume_identical_obs,
                uns_policy,
                parse_index_columns(sort_by.as_deref()),
                sort_reverse,
            ),
            // Unlike `scx convert`, the merged output is passed via `--output`,
            // not positionally — `inputs` is variadic, so a trailing path is
            // read as another input. Name the flag explicitly so users coming
            // from `convert`'s positional output aren't left guessing.
            None => Err(format!(
                "scx merge: missing required --output <PATH>.\n\
                 The merged file is written via the --output flag (not a positional \
                 argument like `scx convert`), because inputs are variadic.\n\
                 Example: scx merge {} --output merged.scx",
                if inputs.is_empty() {
                    "a.scx b.scx".to_string()
                } else {
                    inputs
                        .iter()
                        .map(|p| p.display().to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                }
            )
            .into()),
        },
        Commands::Query {
            source,
            modality,
            filter,
            filter_flag,
            count,
            output,
            select_genes,
            normalize,
            log1p,
            limit,
            json,
            explain,
        } => {
            // The predicate is optional and may be given positionally
            // (back-compat) or via `--filter` (consistent with `scx subset` /
            // `scx delete`). clap's `conflicts_with` rejects supplying both.
            // With neither, the query runs over all cells — matching pyscx
            // `query()` semantics — so gene-only projections / transforms /
            // `--count` / `--output` work without inventing a tautological
            // predicate.
            query::run_query(
                &source,
                modality.as_deref(),
                filter.or(filter_flag).as_deref(),
                count,
                output.as_deref(),
                select_genes.as_deref(),
                normalize,
                log1p,
                limit,
                json,
                explain,
            )
        }
        Commands::Benchmark {
            file,
            compare_h5ad,
            runs,
            json,
        } => benchmark::run_benchmark(&file, compare_h5ad.as_deref(), runs, json),
        Commands::BuildCsc {
            input,
            output,
            memory_limit,
            force,
            csc_cols_per_shard,
        } => scx_ops::run_build_csc(
            &input,
            &output,
            &memory_limit,
            force,
            csc_cols_per_shard,
            None,
        ),
        Commands::Subset {
            input,
            output,
            filter,
            genes,
            modality,
            dry_run,
            shard_size,
            codec,
            rebuild_csc,
            csc_cols_per_shard,
            csc_memory_limit,
        } => subset::run_subset(
            &input,
            output.as_deref(),
            filter.as_deref(),
            genes.as_deref(),
            modality.as_deref(),
            dry_run,
            shard_size,
            &codec,
            rebuild_csc,
            csc_cols_per_shard,
            &csc_memory_limit,
        ),
        Commands::Upgrade {
            input,
            output,
            in_place,
        } => upgrade::run_upgrade(&input, output.as_deref(), in_place),
        #[cfg(feature = "cloud")]
        Commands::CloudOptimize { input, output } => {
            cloud_optimize::run_cloud_optimize(&input, output.as_deref())
        }
        #[cfg(feature = "cloud")]
        Commands::Explode { input, output } => explode::run_explode(&input, &output),
        #[cfg(feature = "cloud")]
        Commands::Pack { input, output } => pack::run_pack(&input, &output),
        #[cfg(feature = "cloud")]
        Commands::Pull {
            source,
            dest,
            parallelism,
            no_cloud_ready,
            filter,
            filter_mode,
        } => pull::run_pull(
            &source,
            &dest,
            parallelism,
            !no_cloud_ready,
            filter.as_deref(),
            &filter_mode,
        ),
        #[cfg(feature = "cloud")]
        Commands::Push {
            source,
            dest,
            parallelism,
        } => push::run_push(&source, &dest, parallelism),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}

#[allow(clippy::too_many_arguments)]
fn run_convert(
    input: &std::path::Path,
    output: &std::path::Path,
    from: Option<&str>,
    to: Option<&str>,
    shard_size: u32,
    codec: &str,
    csc: &str,
    csc_cols_per_shard: usize,
    // Framed by default (CLI arg default = DEFAULT_ROW_GROUP_ROWS); `0` is the
    // unframed (v3) opt-out, normalized to `None` in the body.
    row_group_rows: u32,
    row_group_target_nnz: Option<u64>,
    modality: Option<&str>,
    min_counts: Option<f64>,
    stream: bool,
    memory_budget: Option<&str>,
    strict_uns: bool,
    dense_zero_epsilon: f32,
    temp_dir: Option<std::path::PathBuf>,
    modalities: Option<&str>,
    modality_types: Option<&str>,
    index_obs: Option<&str>,
    index_var: Option<&str>,
    index_preset: Option<String>,
    index_auto_threshold: usize,
    bitmap: &str,
    reader_threads: Option<usize>,
    writer_queue_depth: usize,
    sort_by: Option<&str>,
    sort_reverse: bool,
    group_by: Option<&str>,
    reference: Option<&str>,
    group_target_bytes: Option<&str>,
    group_max_bytes: Option<&str>,
    group_pass: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let direction = convert::determine_convert_direction(from, to, input)?;
    let sort_by_list = parse_index_columns(sort_by);
    let group_by_value = group_by
        .map(str::to_string)
        .filter(|s| !s.trim().is_empty());

    // CSC mode is meaningful only on input → SCX paths. Reject silently
    // for output paths (h5ad / mtx) where the destination has no CSC
    // concept. `Auto` is resolved against the dataset shape downstream
    // (clap's value_parser already restricts the input to off|auto|always;
    // the parse error arm is defensive).
    let csc_policy = convert::CscPolicy::parse(csc).map_err(|e| -> Box<dyn std::error::Error> {
        format!("invalid --csc value: {e}").into()
    })?;

    // Phase 5a: split CSV --index-obs / --index-var into Vec<String>;
    // empty / whitespace-only inputs are treated as no override. Parsed
    // up front so the direction guard below can reject manual index
    // flags on conversions that cannot build predicate indexes — before
    // the `--stream` guard and the mtx early-return, so the user always
    // gets the specific index-direction message rather than a confusing
    // proxy error.
    let index_obs_list = parse_index_columns(index_obs);
    let index_var_list = parse_index_columns(index_var);
    let index_preset_value = index_preset.filter(|s| !s.trim().is_empty());

    // Manual predicate-index flags only have an effect on conversions
    // that write SCX from h5ad / 10x (which build indexes inline) and on
    // h5mu → scx (which accepts them and downstream emits
    // `PredicateIndexSkippedMultimodal`). On every other direction the
    // flags were previously dropped silently — a footgun. Reject them
    // up front, before any file I/O, with an actionable message.
    let index_requested =
        !index_obs_list.is_empty() || !index_var_list.is_empty() || index_preset_value.is_some();
    if index_requested && !matches!(direction, "h5ad_to_scx" | "tenx_to_scx" | "h5mu_to_scx") {
        return Err(format!(
            "--index-obs / --index-var / --index-preset are only supported when writing SCX \
             from h5ad or 10x input (h5mu → scx accepts them but skips the index build with a \
             warning); got direction '{direction}'. (For an existing SCX file, rebuild indexes \
             with `scx compact` / `scx append` / `scx merge`, or pyscx.from_anndata.)"
        )
        .into());
    }

    // `--stream` is supported for h5ad → scx (Phase 0/1/2), h5mu →
    // scx (Phase 3), and scx → h5ad / h5mu (Phase 8). Reject for
    // other directions so the user gets a clear error rather than a
    // confusing downstream failure.
    if stream
        && !matches!(
            direction,
            "h5ad_to_scx" | "h5mu_to_scx" | "scx_to_h5ad" | "scx_to_h5mu"
        )
    {
        return Err(format!(
            "--stream is only supported for h5ad → scx, h5mu → scx, scx → h5ad, and \
             scx → h5mu; got direction '{direction}'."
        )
        .into());
    }
    if (modalities.is_some() || modality_types.is_some()) && direction != "h5mu_to_scx" {
        return Err(format!(
            "--modalities / --modality-types only apply to h5mu → scx; got direction '{direction}'."
        )
        .into());
    }
    // `--min-counts` is an export-side row filter. Check it before the
    // `--stream` guard below so a user who passes both gets the specific
    // message rather than the generic direction complaint.
    if let Some(mc) = min_counts {
        if direction != "scx_to_h5ad" {
            return Err(format!(
                "--min-counts is only supported for scx → h5ad; got direction '{direction}'. \
                 (For a multimodal source, extract one modality with --modality.)"
            )
            .into());
        }
        if !stream {
            return Err(
                "--min-counts requires the streaming export path; drop --stream=false. \
                        The legacy materializing path cannot apply a caller-supplied row mask."
                    .into(),
            );
        }
        if !mc.is_finite() || mc < 0.0 {
            return Err(
                format!("--min-counts must be a finite non-negative number; got {mc}").into(),
            );
        }
    }
    // Sort-on-convert applies only to h5ad → scx and requires the streaming
    // path (the random-access permuted gather). Force streaming on.
    if !sort_by_list.is_empty() && direction != "h5ad_to_scx" {
        return Err(format!(
            "--sort-by is only supported for h5ad → scx; got direction '{direction}'."
        )
        .into());
    }
    // Convert-time grouping (Phase 7.4): h5ad → scx only, streaming path.
    if reference.is_some() && group_by_value.is_none() {
        return Err("--reference requires --group-by".into());
    }
    if group_by_value.is_some() && direction != "h5ad_to_scx" {
        return Err(format!(
            "--group-by is only supported for h5ad → scx; got direction '{direction}'."
        )
        .into());
    }
    let stream = stream || !sort_by_list.is_empty() || group_by_value.is_some();

    // MTX conversions are always available (no hdf5 feature needed)
    match direction {
        "mtx_to_scx" => {
            return dispatch_mtx_to_scx(
                input,
                output,
                shard_size,
                codec,
                csc_policy,
                csc_cols_per_shard,
            );
        }
        "scx_to_mtx" => return dispatch_scx_to_mtx(input, output),
        _ => {}
    }

    // Parse `--memory-budget` once here so an invalid value fails the
    // command before we touch the file. Empty string and `None` both
    // mean "use default heuristics" (= `ConvertOptions::memory_budget = None`).
    // The parser lives behind scx-convert's `hdf5` feature gate; the
    // non-hdf5 CLI stub never reaches the dispatch, so silently drop
    // the budget there (it would be unused anyway).
    #[cfg(feature = "hdf5")]
    let memory_budget_bytes: Option<u64> = match memory_budget {
        None => None,
        Some(s) => Some(convert::MemoryBudget::parse(s)?),
    };
    #[cfg(not(feature = "hdf5"))]
    let memory_budget_bytes: Option<u64> = {
        let _ = memory_budget;
        None
    };

    // Phase 7.4 grouped-convert byte budgets — parsed with the same size syntax
    // as `--memory-budget` so an invalid value fails before any file I/O.
    #[cfg(feature = "hdf5")]
    let (group_target_bytes_val, group_max_bytes_val): (Option<u64>, Option<u64>) = (
        match group_target_bytes {
            None => None,
            Some(s) => Some(convert::MemoryBudget::parse(s)?),
        },
        match group_max_bytes {
            None => None,
            Some(s) => Some(convert::MemoryBudget::parse(s)?),
        },
    );
    #[cfg(not(feature = "hdf5"))]
    let (group_target_bytes_val, group_max_bytes_val): (Option<u64>, Option<u64>) = {
        let _ = (group_target_bytes, group_max_bytes);
        (None, None)
    };
    // `col:NAME` (or `column:NAME`) → boolean reference column; otherwise a CSV
    // label set. A non-empty value that fails to parse is a user error, not a
    // silent "no reference".
    let reference_spec: Option<scx_ops::ReferenceSpec> = match reference {
        Some(s) if !s.trim().is_empty() => {
            let spec = parse_reference_spec_cli(s);
            if spec.is_none() {
                return Err(format!(
                    "--reference value {s:?} is not a valid label set or \
                    `col:NAME` reference column"
                )
                .into());
            }
            spec
        }
        _ => None,
    };
    let group_pass_val = convert::GroupPass::parse(group_pass)?;
    if group_by_value.is_none()
        && (group_target_bytes_val.is_some() || group_max_bytes_val.is_some())
    {
        log::warn!(
            "scx convert: --group-target-bytes / --group-max-bytes are ignored without --group-by"
        );
    }

    // Parse Phase 3 h5mu filters / type overrides. The empty-string
    // case is treated as no filter; non-empty strings are split on
    // commas and validated.
    let modalities_list: Option<Vec<String>> = match modalities {
        None => None,
        Some(s) if s.trim().is_empty() => None,
        Some(s) => Some(
            s.split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect(),
        ),
    };
    let modality_types_list: Vec<(String, scx_format_io::modality::ModalityType)> =
        match modality_types {
            None => Vec::new(),
            Some(s) if s.trim().is_empty() => Vec::new(),
            Some(s) => parse_modality_types(s)?,
        };

    dispatch_convert(
        direction,
        input,
        output,
        shard_size,
        codec,
        csc_policy,
        csc_cols_per_shard,
        row_group_rows,
        row_group_target_nnz,
        modality,
        min_counts,
        stream,
        memory_budget_bytes,
        strict_uns,
        dense_zero_epsilon,
        temp_dir,
        modalities_list,
        modality_types_list,
        index_obs_list,
        index_var_list,
        index_preset_value,
        index_auto_threshold,
        bitmap,
        reader_threads,
        writer_queue_depth,
        sort_by_list,
        sort_reverse,
        group_by_value,
        reference_spec,
        group_target_bytes_val,
        group_max_bytes_val,
        group_pass_val,
    )
}

/// Parse the `--reference` CLI value into a [`scx_ops::ReferenceSpec`].
/// `col:NAME` (or the `column:NAME` alias) selects a boolean obs column;
/// anything else is a comma-separated set of `--group-by` labels. Returns
/// `None` for an empty / whitespace-only value. Shared by `scx sort` and
/// `scx convert` so the flag parses identically on both subcommands.
pub(crate) fn parse_reference_spec_cli(s: &str) -> Option<scx_ops::ReferenceSpec> {
    if let Some(col) = s.strip_prefix("column:").or_else(|| s.strip_prefix("col:")) {
        let col = col.trim();
        if col.is_empty() {
            return None;
        }
        return Some(scx_ops::ReferenceSpec::Column(col.to_string()));
    }
    let labels: Vec<String> = s
        .split(',')
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect();
    if labels.is_empty() {
        None
    } else {
        Some(scx_ops::ReferenceSpec::Labels(labels))
    }
}

/// Parse a comma-separated CLI argument into a `Vec<String>`. Whitespace
/// is trimmed and empty segments are dropped (so trailing commas behave
/// as users expect). Returns an empty vec when the input is `None`.
fn parse_index_columns(value: Option<&str>) -> Vec<String> {
    value
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Parse `--modality-types name:Type,name:Type` into a typed list.
/// Case-insensitive on the right side; case-sensitive modality
/// names match `/mod/{name}` keys verbatim.
fn parse_modality_types(
    s: &str,
) -> Result<Vec<(String, scx_format_io::modality::ModalityType)>, Box<dyn std::error::Error>> {
    use scx_format_io::modality::ModalityType;
    s.split(',')
        .map(|kv| {
            let trimmed = kv.trim();
            let (k, v) = trimmed
                .split_once(':')
                .ok_or_else(|| format!("expected 'name:Type', got '{trimmed}'"))?;
            let mt = match v.trim().to_lowercase().as_str() {
                "rna" => ModalityType::Rna,
                "protein" | "adt" => ModalityType::Protein,
                "atac" => ModalityType::Atac,
                "spatial" => ModalityType::Spatial,
                "methylation" | "methyl" => ModalityType::Methylation,
                "custom" => ModalityType::Custom,
                other => {
                    return Err(format!(
                        "unknown modality type '{other}'; valid: \
                         rna, protein, atac, spatial, methylation, custom"
                    )
                    .into());
                }
            };
            Ok((k.trim().to_string(), mt))
        })
        .collect()
}

#[cfg(feature = "hdf5")]
#[allow(clippy::too_many_arguments)]
fn dispatch_convert(
    direction: &str,
    input: &std::path::Path,
    output: &std::path::Path,
    shard_size: u32,
    codec: &str,
    csc_policy: convert::CscPolicy,
    csc_cols_per_shard: usize,
    // Framed by default; `0` = unframed (v3) opt-out, normalized to `None` below.
    row_group_rows: u32,
    row_group_target_nnz: Option<u64>,
    modality: Option<&str>,
    min_counts: Option<f64>,
    stream: bool,
    memory_budget: Option<u64>,
    strict_uns: bool,
    dense_zero_epsilon: f32,
    temp_dir: Option<std::path::PathBuf>,
    modalities: Option<Vec<String>>,
    modality_types: Vec<(String, scx_format_io::modality::ModalityType)>,
    index_obs: Vec<String>,
    index_var: Vec<String>,
    index_preset: Option<String>,
    index_auto_threshold: usize,
    bitmap: &str,
    reader_threads: Option<usize>,
    writer_queue_depth: usize,
    sort_by: Vec<String>,
    sort_reverse: bool,
    group_by: Option<String>,
    reference: Option<scx_ops::ReferenceSpec>,
    group_target_bytes: Option<u64>,
    group_max_bytes: Option<u64>,
    group_pass: convert::GroupPass,
) -> Result<(), Box<dyn std::error::Error>> {
    use convert::{BitmapPolicy, ConvertError, ConvertOptions};
    use indicatif::{ProgressBar, ProgressStyle};
    let bitmap_policy = BitmapPolicy::parse(bitmap).map_err(|e| e.to_string())?;

    // Resolve the codec intent axis (`auto`/`fast`/`compact` + explicit forces).
    // `compact`/`compact-trial`/explicit-`shufdelta` require framing; `auto`
    // silently falls back to the heuristic single-encode when unframed.
    let resolved = scx_format_io::resolve_codec(Some(codec))?;
    let explicit_codec = resolved.explicit_codec;
    let codec_trial = resolved.codec_trial;
    let decode_target = resolved.decode_target;
    if resolved.requires_framing && row_group_rows == 0 {
        return Err(format!(
            "`--codec {}` requires row-group framing; drop `--row-group-rows 0`",
            resolved.profile
        )
        .into());
    }
    // Framing is on by default (row_group_rows default = 256); `0` is the
    // explicit unframed (v3) opt-out → None threads through as the legacy layout.
    let row_group_rows = (row_group_rows != 0).then_some(row_group_rows);

    let opts = ConvertOptions {
        shard_target_rows: shard_size,
        codec: explicit_codec,
        csc: csc_policy,
        csc_cols_per_shard,
        row_group_rows,
        row_group_target_nnz,
        codec_trial,
        decode_target,
        tool: "scx".into(),
        memory_budget,
        stream,
        strict_uns,
        dense_zero_epsilon,
        temp_dir,
        modalities,
        modality_types,
        index_obs,
        index_var,
        index_preset,
        index_auto_threshold,
        bitmap: bitmap_policy,
        reader_threads,
        writer_queue_depth,
        sort_by,
        sort_reverse,
        group_by,
        reference,
        group_target_bytes,
        group_max_bytes,
        group_pass,
        export_obs_keep_mask: None,
        export_min_counts: min_counts,
    };

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("valid template"),
    );
    pb.set_message(format!("Converting {}...", input.display()));

    // Structured warning channel (Phase 0.3). Default backend forwards
    // each emission to `log::warn!`; we also print a per-category
    // summary at the end of the command.
    let mut sink = convert::WarningSink::log();

    // Special-case scx_to_h5ad on multimodal input: gate on the
    // `--modality` flag and route through `scx_modality_to_h5ad` when
    // present. The plain single-modality path stays untouched.
    let result: Result<(), ConvertError> = match direction {
        "h5ad_to_scx" => {
            if stream {
                convert::h5ad_to_scx_streaming(
                    input,
                    output,
                    &opts,
                    &convert::StreamingOverrides::default(),
                    &mut sink,
                )
            } else {
                convert::h5ad_to_scx(input, output, &opts, &mut sink)
            }
        }
        "h5mu_to_scx" => {
            if stream {
                convert::h5mu_to_scx_streaming(input, output, &opts, &mut sink)
            } else {
                convert::h5mu_to_scx(input, output, &opts, &mut sink)
            }
        }
        "tenx_to_scx" => convert::tenx_to_scx(input, output, &opts, &mut sink),
        "scx_to_h5ad" => match modality {
            Some(name) => {
                if opts.stream {
                    convert::scx_modality_to_h5ad_streaming(input, output, name, &opts, &mut sink)
                } else {
                    convert::scx_modality_to_h5ad(input, output, name, &mut sink)
                }
            }
            None => {
                // If the file is multimodal, raise with a clear
                // message; if single-modality, fall through to the
                // h5ad writer (streaming by default).
                let reader = scx_format_io::reader::ScxReader::open(input)?;
                let is_multimodal = reader.is_multimodal();
                let n_modalities = reader.n_modalities();
                drop(reader);
                if is_multimodal {
                    return Err(format!(
                        "SCX file '{}' has {} modalities; use --to h5mu, or use \
                         --modality NAME to extract a single modality as h5ad",
                        input.display(),
                        n_modalities,
                    )
                    .into());
                }
                if opts.stream {
                    convert::scx_to_h5ad_streaming(input, output, &opts, &mut sink)
                } else {
                    convert::scx_to_h5ad(input, output, &mut sink)
                }
            }
        },
        "scx_to_h5mu" => {
            if opts.stream {
                convert::scx_to_h5mu_streaming(input, output, &opts, &mut sink)
            } else {
                convert::scx_to_h5mu(input, output, &mut sink)
            }
        }
        _ => unreachable!(),
    };

    pb.finish_and_clear();

    if sink.total() > 0 {
        let parts: Vec<String> = sink
            .counts()
            .iter()
            .map(|(cat, n)| format!("{cat}={n}"))
            .collect();
        eprintln!("{} conversion warnings: {}", sink.total(), parts.join(", "));
    }

    match result {
        Ok(()) => {
            println!("Converted {} -> {}", input.display(), output.display());
            Ok(())
        }
        Err(e) => Err(Box::new(e)),
    }
}

#[cfg(not(feature = "hdf5"))]
#[allow(clippy::too_many_arguments)]
fn dispatch_convert(
    _direction: &str,
    _input: &std::path::Path,
    _output: &std::path::Path,
    _shard_size: u32,
    _codec: &str,
    _csc_policy: convert::CscPolicy,
    _csc_cols_per_shard: usize,
    _row_group_rows: u32,
    _row_group_target_nnz: Option<u64>,
    _modality: Option<&str>,
    _min_counts: Option<f64>,
    _stream: bool,
    _memory_budget: Option<u64>,
    _strict_uns: bool,
    _dense_zero_epsilon: f32,
    _temp_dir: Option<std::path::PathBuf>,
    _modalities: Option<Vec<String>>,
    _modality_types: Vec<(String, scx_format_io::modality::ModalityType)>,
    _index_obs: Vec<String>,
    _index_var: Vec<String>,
    _index_preset: Option<String>,
    _index_auto_threshold: usize,
    _bitmap: &str,
    _reader_threads: Option<usize>,
    _writer_queue_depth: usize,
    _sort_by: Vec<String>,
    _sort_reverse: bool,
    _group_by: Option<String>,
    _reference: Option<scx_ops::ReferenceSpec>,
    _group_target_bytes: Option<u64>,
    _group_max_bytes: Option<u64>,
    _group_pass: convert::GroupPass,
) -> Result<(), Box<dyn std::error::Error>> {
    Err(
        "h5ad/h5mu/10x conversion requires the 'hdf5' feature. Rebuild with: cargo build -p scx-cli --features hdf5\n\
         Note: MTX conversion is always available (use --from mtx or --to mtx)."
            .into(),
    )
}

/// MTX → SCX conversion (always available, no hdf5 feature needed).
fn dispatch_mtx_to_scx(
    input: &std::path::Path,
    output: &std::path::Path,
    shard_size: u32,
    codec: &str,
    csc_policy: convert::CscPolicy,
    csc_cols_per_shard: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use convert::mtx_pipeline;
    use indicatif::{ProgressBar, ProgressStyle};

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("valid template"),
    );
    pb.set_message(format!("Converting MTX {}...", input.display()));

    let orientation = mtx_pipeline::mtx_to_scx(input, output, shard_size, codec)?;
    pb.finish_and_clear();

    if orientation == convert::MtxOrientation::Ambiguous {
        eprintln!(
            "warning: MTX matrix is square, so its orientation is ambiguous; assumed the \
             Cell Ranger default (features × barcodes) and transposed to cells × genes. \
             If your matrix was already cells × genes, obs and var are now swapped — \
             verify the output shape and names."
        );
    }

    // MTX conversion is delegated to the standalone `scx-mtx` crate,
    // which doesn't know about CSC. When the policy resolves to build,
    // post-process the just-written file with the existing build-csc
    // machinery: write to `<output>.csc.tmp`, then atomically rename onto
    // the final path. Costs an extra read pass but adds the CSC sidecar
    // without modifying scx-mtx. `Auto` reads back the just-written
    // header for the shape (scx-mtx doesn't return it to the caller).
    let build_csc = match csc_policy {
        convert::CscPolicy::Off => false,
        convert::CscPolicy::Always => true,
        convert::CscPolicy::Auto => {
            let reader = scx_format_io::ScxReader::open(output)?;
            let header = reader.header();
            csc_policy.should_build_csc(header.n_obs, header.n_vars)
        }
    };
    if build_csc {
        let tmp = output.with_extension("scx.csc.tmp");
        let _ = std::fs::remove_file(&tmp);
        scx_ops::run_build_csc(output, &tmp, "4G", false, csc_cols_per_shard, None)?;
        std::fs::rename(&tmp, output)?;
    }

    println!("Converted {} -> {}", input.display(), output.display());
    Ok(())
}

/// SCX → MTX conversion (always available, no hdf5 feature needed).
fn dispatch_scx_to_mtx(
    input: &std::path::Path,
    output: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use indicatif::{ProgressBar, ProgressStyle};

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("valid template"),
    );
    pb.set_message(format!("Converting to MTX {}...", output.display()));

    scx_mtx::write_scx_to_mtx(input, output)?;

    pb.finish_and_clear();
    println!("Converted {} -> {}", input.display(), output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_reference_spec_cli;
    use scx_ops::ReferenceSpec;

    #[test]
    fn reference_spec_accepts_both_column_prefixes() {
        // H1: `scx convert` and `scx sort` share this parser; both `col:` and
        // the `column:` alias must resolve to the same boolean-column spec.
        assert_eq!(
            parse_reference_spec_cli("col:is_control"),
            Some(ReferenceSpec::Column("is_control".to_string()))
        );
        assert_eq!(
            parse_reference_spec_cli("column:is_control"),
            Some(ReferenceSpec::Column("is_control".to_string()))
        );
        // Trailing/leading whitespace around the column name is trimmed.
        assert_eq!(
            parse_reference_spec_cli("col:  is_control  "),
            Some(ReferenceSpec::Column("is_control".to_string()))
        );
    }

    #[test]
    fn reference_spec_empty_column_name_is_none() {
        // An empty name after the prefix is not a usable column; the callers
        // (convert/sort) treat a `None` return on non-empty input as a hard
        // user error rather than silently dropping the reference.
        assert_eq!(parse_reference_spec_cli("col:"), None);
        assert_eq!(parse_reference_spec_cli("col:   "), None);
        assert_eq!(parse_reference_spec_cli("column:"), None);
    }

    #[test]
    fn reference_spec_parses_label_list() {
        assert_eq!(
            parse_reference_spec_cli("a,b , c"),
            Some(ReferenceSpec::Labels(vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
            ]))
        );
        // A bare token without the `col:`/`column:` prefix is a label, not a
        // column — even one that looks prefix-like without a colon.
        assert_eq!(
            parse_reference_spec_cli("col"),
            Some(ReferenceSpec::Labels(vec!["col".to_string()]))
        );
    }

    #[test]
    fn reference_spec_empty_input_is_none() {
        assert_eq!(parse_reference_spec_cli(""), None);
        assert_eq!(parse_reference_spec_cli("   "), None);
        // A value that is only separators / whitespace yields no labels.
        assert_eq!(parse_reference_spec_cli(" , , "), None);
    }
}
