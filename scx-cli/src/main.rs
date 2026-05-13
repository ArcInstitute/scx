use clap::{Parser, Subcommand};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::process;

mod append;
mod benchmark;
mod build_csc;
mod compact;
mod convert;
mod delete;
mod info;
mod merge;
mod query;
mod rebuild_csc;
mod rewrite_helpers;
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
        #[arg(long, default_value = "10000", value_parser = validators::positive_u32)]
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
        #[arg(long, default_value_t = NonZeroU32::new(10000).unwrap())]
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
        /// SCX file to query
        file: PathBuf,
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
        /// Filter granularity for selective pulls: 'shard' (default, fast,
        /// may include extra cells) or 'exact' (cell-granular, reserved
        /// for future implementation)
        #[arg(long, default_value = "shard", value_parser = ["shard", "exact"])]
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
        #[arg(long, default_value = "10000", value_parser = validators::positive_u32)]
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
            file,
            filter,
            count,
            output,
            select_genes,
            normalize,
            log1p,
            limit,
            json,
        } => query::run_query(
            &file,
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
        } => build_csc::run_build_csc(&input, &output, &memory_limit, force, csc_cols_per_shard),
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

    dispatch_convert(
        direction,
        input,
        output,
        shard_size,
        codec,
        csc_always,
        csc_cols_per_shard,
        modality,
    )
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
) -> Result<(), Box<dyn std::error::Error>> {
    use convert::{ConvertError, ConvertOptions};
    use indicatif::{ProgressBar, ProgressStyle};
    use scx_codec::CodecId;

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
    };

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("valid template"),
    );
    pb.set_message(format!("Converting {}...", input.display()));

    // Special-case scx_to_h5ad on multimodal input: gate on the
    // `--modality` flag and route through `scx_modality_to_h5ad` when
    // present. The plain single-modality path stays untouched.
    let result: Result<(), ConvertError> = match direction {
        "h5ad_to_scx" => convert::h5ad_to_scx(input, output, &opts),
        "h5mu_to_scx" => convert::h5mu_to_scx(input, output, &opts),
        "tenx_to_scx" => convert::tenx_to_scx(input, output, &opts),
        "scx_to_h5ad" => match modality {
            Some(name) => convert::scx_modality_to_h5ad(input, output, name),
            None => {
                // If the file is multimodal, raise with a clear
                // message; if single-modality, fall through to the
                // legacy h5ad writer.
                let reader = scx_format::reader::ScxReader::open(input)?;
                if reader.is_multimodal() {
                    drop(reader);
                    return Err(format!(
                        "SCX file '{}' has {} modalities; use --to h5mu, or use \
                         --modality NAME to extract a single modality as h5ad",
                        input.display(),
                        {
                            let r = scx_format::reader::ScxReader::open(input)?;
                            let n = r.n_modalities();
                            drop(r);
                            n
                        }
                    )
                    .into());
                }
                drop(reader);
                convert::scx_to_h5ad(input, output)
            }
        },
        "scx_to_h5mu" => convert::scx_to_h5mu(input, output),
        _ => unreachable!(),
    };

    pb.finish_and_clear();

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
        build_csc::run_build_csc(output, &tmp, "4G", false, csc_cols_per_shard)?;
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
