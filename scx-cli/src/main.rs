use clap::{Parser, Subcommand};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::process;

mod append;
mod benchmark;
mod compact;
use scx_convert as convert;
mod delete;
mod info;
mod merge;
mod query;
mod rollback;
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
#[command(name = "scx", about = "SCX file format tool")]
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
        #[arg(long, default_value_t = scx_format::DEFAULT_SHARD_TARGET_ROWS, value_parser = validators::positive_u32)]
        shard_size: u32,
        /// Compression codec: auto (default), none, scx1, zstd, lz4, pcodec
        #[arg(long, default_value = "auto")]
        codec: String,
        /// Whether to also emit a CSC sidecar at write time.
        ///
        /// `off` (default): CSR-only output, matches existing behavior.
        /// `always`: also emits a CSC sidecar (column-major shards).
        /// No `auto` mode — by design, users opt in explicitly.
        #[arg(long, default_value = "off", value_parser = ["off", "always"])]
        csc: String,
        /// Columns per CSC shard when `--csc always` (default 5000).
        ///
        /// Pass `0` to disable the cap (single CSC shard, memory permitting).
        #[arg(long, default_value_t = 5000)]
        csc_cols_per_shard: usize,
        /// Extract a single modality from a multi-modality SCX file
        /// when writing to h5ad. Required when `--to h5ad` is used on
        /// a multimodal SCX input; ignored otherwise.
        #[arg(long)]
        modality: Option<String>,
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
        /// derate). Accepts bare bytes, `K`/`M`/`G`/`T`, or
        /// `KiB`/`MiB`/`GiB`/`TiB`. Decimal suffixes (`KB`, `MB`)
        /// are rejected as ambiguous. None = each phase's default.
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
    },
    /// Display SCX file information
    Info {
        /// SCX file to inspect
        file: PathBuf,
        /// Output all info as JSON
        #[arg(long)]
        json: bool,
        /// Show manifest version history
        #[arg(long)]
        history: bool,
    },
    /// Validate header + per-section BLAKE3 checksums (deeper than `scx info`)
    Validate {
        /// SCX file to validate
        file: PathBuf,
        /// Print checksum values
        #[arg(long)]
        verbose: bool,
    },
    /// Append cells from another SCX file
    Append {
        /// Target SCX file to append to
        target: PathBuf,
        /// Source SCX file containing cells to append
        #[arg(long)]
        input: PathBuf,
        /// Modality name to append into. Required on multimodal target
        /// files (`scx info` shows the modality table). Optional on
        /// single-modality files — defaults to the global / primary
        /// modality.
        #[arg(long)]
        modality: Option<String>,
        /// Compression codec for new shards: auto, none, scx1, zstd, lz4, pcodec
        #[arg(long, default_value = "auto")]
        codec: String,
        /// Target rows per shard (must be > 0)
        #[arg(long, default_value_t = NonZeroU32::new(scx_format::DEFAULT_SHARD_TARGET_ROWS).unwrap())]
        shard_size: NonZeroU32,
        /// Rebuild the CSC sidecar after appending (drops + re-emits via
        /// `scx build-csc`). Without this flag, append drops the CSC
        /// sidecar with a warning — the row layout no longer matches.
        #[arg(long)]
        rebuild_csc: bool,
        /// Maximum columns per emitted CSC shard when `--rebuild-csc` is
        /// set (default: 5000). Ignored without `--rebuild-csc`.
        #[arg(long, default_value_t = 5000)]
        csc_cols_per_shard: usize,
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
    /// Rewrite file reclaiming space from deletions
    Compact {
        /// SCX file to compact
        input: PathBuf,
        /// Output path for compacted file
        #[arg(long)]
        output: PathBuf,
        /// Overwrite output if it exists
        #[arg(long)]
        force: bool,
        /// Rebuild the CSC sidecar on the compacted output (drops +
        /// re-emits via `scx build-csc`). Without this flag, compact
        /// drops the CSC sidecar with a warning — the row layout no
        /// longer matches after deletion-vector application.
        #[arg(long)]
        rebuild_csc: bool,
        /// Maximum columns per emitted CSC shard when `--rebuild-csc`
        /// is set (default: 5000). Ignored without `--rebuild-csc`.
        #[arg(long, default_value_t = 5000)]
        csc_cols_per_shard: usize,
    },
    /// Revert to a previous manifest version
    Rollback {
        /// SCX file to roll back
        file: PathBuf,
        /// Target manifest sequence number (default: previous version)
        #[arg(long)]
        to_seq: Option<u64>,
    },
    /// Merge multiple SCX files into one
    Merge {
        /// Input SCX files to merge (at least 2)
        inputs: Vec<PathBuf>,
        /// Output path for merged file
        #[arg(long)]
        output: PathBuf,
        /// Rebuild the CSC sidecar on the merged output (drops +
        /// re-emits via `scx build-csc`). Without this flag, merge
        /// drops any input CSC sidecars with a warning.
        #[arg(long)]
        rebuild_csc: bool,
        /// Maximum columns per emitted CSC shard when `--rebuild-csc`
        /// is set (default: 5000). Ignored without `--rebuild-csc`.
        #[arg(long, default_value_t = 5000)]
        csc_cols_per_shard: usize,
    },
    /// Query cells by predicate
    Query {
        /// SCX file path or cloud URL (e.g. `gs://bucket/atlas.scxd/`)
        source: String,
        /// Obs predicate expression
        filter: String,
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
        /// Limit number of returned cells
        #[arg(long)]
        limit: Option<usize>,
        /// JSON output (for --count)
        #[arg(long)]
        json: bool,
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
        /// Maximum memory for transpose working set (default: 4G)
        /// Accepts suffixes: K, M, G (e.g., "100M", "4G")
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
        /// Output SCX file path
        #[arg(long)]
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
        #[arg(long, default_value_t = scx_format::DEFAULT_SHARD_TARGET_ROWS, value_parser = validators::positive_u32)]
        shard_size: u32,
        /// Compression codec for output: auto, none, scx1, zstd, lz4, pcodec
        #[arg(long, default_value = "auto")]
        codec: String,
        /// Rebuild the CSC sidecar on the subset output (drops +
        /// re-emits via `scx build-csc` against the projected CSR).
        /// Without this flag, subset drops any input CSC sidecar
        /// with a warning — the row/column index space changes.
        #[arg(long)]
        rebuild_csc: bool,
        /// Maximum columns per emitted CSC shard when `--rebuild-csc`
        /// is set (default: 5000). Ignored without `--rebuild-csc`.
        #[arg(long, default_value_t = 5000)]
        csc_cols_per_shard: usize,
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

fn main() {
    // Initialize the `log` sink. Default severity is `info`; override with
    // `RUST_LOG=scx=debug`, `RUST_LOG=scx_loader=warn`, etc.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

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
            modality,
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
        } => run_convert(
            &input,
            &output,
            from.as_deref(),
            to.as_deref(),
            shard_size,
            &codec,
            &csc,
            csc_cols_per_shard,
            modality.as_deref(),
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
        ),
        Commands::Info {
            file,
            json,
            history,
        } => info::run_info(&file, json, history),
        Commands::Validate { file, verbose } => match validate::run_validate(&file, verbose) {
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
            input,
            modality,
            codec,
            shard_size,
            rebuild_csc,
            csc_cols_per_shard,
        } => append::run_append(
            &target,
            &input,
            modality.as_deref(),
            &codec,
            shard_size,
            rebuild_csc,
            csc_cols_per_shard,
        ),
        Commands::Delete {
            file,
            filter,
            dry_run,
        } => delete::run_delete(&file, &filter, dry_run),
        Commands::Compact {
            input,
            output,
            force,
            rebuild_csc,
            csc_cols_per_shard,
        } => compact::run_compact(&input, &output, force, rebuild_csc, csc_cols_per_shard),
        Commands::Rollback { file, to_seq } => rollback::run_rollback(&file, to_seq),
        Commands::Merge {
            inputs,
            output,
            rebuild_csc,
            csc_cols_per_shard,
        } => merge::run_merge(&inputs, &output, rebuild_csc, csc_cols_per_shard),
        Commands::Query {
            source,
            filter,
            count,
            output,
            select_genes,
            normalize,
            log1p,
            limit,
            json,
        } => query::run_query(
            &source,
            &filter,
            count,
            output.as_deref(),
            select_genes.as_deref(),
            normalize,
            log1p,
            limit,
            json,
        ),
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
        } => scx_ops::run_build_csc(&input, &output, &memory_limit, force, csc_cols_per_shard),
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
    modality: Option<&str>,
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
) -> Result<(), Box<dyn std::error::Error>> {
    let direction = convert::determine_convert_direction(from, to, input)?;

    // CSC mode is meaningful only on input → SCX paths. Reject silently
    // for output paths (h5ad / mtx) where the destination has no CSC
    // concept.
    let csc_always = match csc {
        "off" => false,
        "always" => true,
        // clap value_parser already restricts to {off, always}; this
        // arm is defensive.
        other => return Err(format!("invalid --csc value: {other}").into()),
    };

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

    // MTX conversions are always available (no hdf5 feature needed)
    match direction {
        "mtx_to_scx" => {
            return dispatch_mtx_to_scx(
                input,
                output,
                shard_size,
                codec,
                csc_always,
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
    let modality_types_list: Vec<(String, scx_format::modality::ModalityType)> =
        match modality_types {
            None => Vec::new(),
            Some(s) if s.trim().is_empty() => Vec::new(),
            Some(s) => parse_modality_types(s)?,
        };

    // Phase 5a: split CSV --index-obs / --index-var into Vec<String>;
    // empty / whitespace-only inputs are treated as no override.
    let index_obs_list = parse_index_columns(index_obs);
    let index_var_list = parse_index_columns(index_var);
    let index_preset_value = index_preset.filter(|s| !s.trim().is_empty());

    dispatch_convert(
        direction,
        input,
        output,
        shard_size,
        codec,
        csc_always,
        csc_cols_per_shard,
        modality,
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
    )
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
) -> Result<Vec<(String, scx_format::modality::ModalityType)>, Box<dyn std::error::Error>> {
    use scx_format::modality::ModalityType;
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
    csc_always: bool,
    csc_cols_per_shard: usize,
    modality: Option<&str>,
    stream: bool,
    memory_budget: Option<u64>,
    strict_uns: bool,
    dense_zero_epsilon: f32,
    temp_dir: Option<std::path::PathBuf>,
    modalities: Option<Vec<String>>,
    modality_types: Vec<(String, scx_format::modality::ModalityType)>,
    index_obs: Vec<String>,
    index_var: Vec<String>,
    index_preset: Option<String>,
    index_auto_threshold: usize,
    bitmap: &str,
    reader_threads: Option<usize>,
    writer_queue_depth: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use convert::{BitmapPolicy, ConvertError, ConvertOptions};
    use indicatif::{ProgressBar, ProgressStyle};
    use scx_codec::CodecId;
    let bitmap_policy = BitmapPolicy::parse(bitmap).map_err(|e| e.to_string())?;

    let explicit_codec = match codec {
        "auto" => None,
        "none" => Some(CodecId::None),
        "scx1" => Some(CodecId::Scx1),
        "zstd" => Some(CodecId::Zstd),
        "lz4" => Some(CodecId::Lz4Shuffle),
        "pcodec" => Some(CodecId::Pcodec),
        other => {
            return Err(format!(
                "Unknown codec: '{}'. Use auto, none, scx1, zstd, lz4, or pcodec.",
                other
            )
            .into())
        }
    };

    let opts = ConvertOptions {
        shard_target_rows: shard_size,
        codec: explicit_codec,
        csc: csc_always,
        csc_cols_per_shard,
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
                    convert::scx_modality_to_h5ad(input, output, name)
                }
            }
            None => {
                // If the file is multimodal, raise with a clear
                // message; if single-modality, fall through to the
                // h5ad writer (streaming by default).
                let reader = scx_format::reader::ScxReader::open(input)?;
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
                convert::scx_to_h5mu(input, output)
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
    _csc_always: bool,
    _csc_cols_per_shard: usize,
    _modality: Option<&str>,
    _stream: bool,
    _memory_budget: Option<u64>,
    _strict_uns: bool,
    _dense_zero_epsilon: f32,
    _temp_dir: Option<std::path::PathBuf>,
    _modalities: Option<Vec<String>>,
    _modality_types: Vec<(String, scx_format::modality::ModalityType)>,
    _index_obs: Vec<String>,
    _index_var: Vec<String>,
    _index_preset: Option<String>,
    _index_auto_threshold: usize,
    _bitmap: &str,
    _reader_threads: Option<usize>,
    _writer_queue_depth: usize,
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
    csc_always: bool,
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

    mtx_pipeline::mtx_to_scx(input, output, shard_size, codec)?;
    pb.finish_and_clear();

    // MTX conversion is delegated to the standalone `scx-mtx` crate,
    // which doesn't know about CSC. When the user opts in via
    // `--csc always`, post-process the just-written file with the
    // existing build-csc machinery: write to `<output>.csc.tmp`, then
    // atomically rename onto the final path. Costs an extra read pass
    // but adds the CSC sidecar without modifying scx-mtx.
    if csc_always {
        let tmp = output.with_extension("scx.csc.tmp");
        let _ = std::fs::remove_file(&tmp);
        scx_ops::run_build_csc(output, &tmp, "4G", false, csc_cols_per_shard)?;
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
