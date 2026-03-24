use clap::{Parser, Subcommand};
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
mod rollback;
mod subset;
mod upgrade;
mod validate;

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
    /// Convert between h5ad/10x/mtx and SCX formats
    Convert {
        /// Input file path
        input: PathBuf,
        /// Output file path
        output: PathBuf,
        /// Input format: h5ad, 10x, mtx
        #[arg(long)]
        from: Option<String>,
        /// Output format: h5ad, mtx
        #[arg(long)]
        to: Option<String>,
        /// Target rows per shard
        #[arg(long, default_value = "10000")]
        shard_size: u32,
        /// Compression codec: auto (default), none, scx1, zstd
        #[arg(long, default_value = "auto")]
        codec: String,
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
    /// Validate SCX file checksums
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
        /// Compression codec for new shards: auto, none, scx1, zstd
        #[arg(long, default_value = "auto")]
        codec: String,
        /// Target rows per shard
        #[arg(long, default_value = "10000")]
        shard_size: u32,
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
        #[arg(long, default_value = "5")]
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
        /// File containing gene indices (one per line) for column projection
        #[arg(long)]
        genes: Option<PathBuf>,
        /// Show matching count without writing output
        #[arg(long)]
        dry_run: bool,
        /// Target rows per shard in the output file
        #[arg(long, default_value = "10000")]
        shard_size: u32,
        /// Compression codec for output: auto, none, scx1, zstd
        #[arg(long, default_value = "auto")]
        codec: String,
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
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Convert {
            input,
            output,
            from,
            to,
            shard_size,
            codec,
        } => run_convert(
            &input,
            &output,
            from.as_deref(),
            to.as_deref(),
            shard_size,
            &codec,
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
            codec,
            shard_size,
        } => append::run_append(&target, &input, &codec, shard_size),
        Commands::Delete {
            file,
            filter,
            dry_run,
        } => delete::run_delete(&file, &filter, dry_run),
        Commands::Compact {
            input,
            output,
            force,
        } => compact::run_compact(&input, &output, force),
        Commands::Rollback { file, to_seq } => rollback::run_rollback(&file, to_seq),
        Commands::Merge { inputs, output } => merge::run_merge(&inputs, &output),
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
        } => build_csc::run_build_csc(&input, &output, &memory_limit, force),
        Commands::Subset {
            input,
            output,
            filter,
            genes,
            dry_run,
            shard_size,
            codec,
        } => subset::run_subset(
            &input,
            output.as_deref(),
            filter.as_deref(),
            genes.as_deref(),
            dry_run,
            shard_size,
            &codec,
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
        } => pull::run_pull(
            &source,
            &dest,
            parallelism,
            !no_cloud_ready,
            filter.as_deref(),
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

fn run_convert(
    input: &std::path::Path,
    output: &std::path::Path,
    from: Option<&str>,
    to: Option<&str>,
    shard_size: u32,
    codec: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Determine conversion direction from explicit flags or file extensions
    let direction = match (from, to) {
        // MTX conversions (always available, no hdf5 feature needed)
        (Some("mtx"), _) => "mtx_to_scx",
        (_, Some("mtx")) => "scx_to_mtx",
        // Auto-detect: input is a directory → MTX
        (None, None) if input.is_dir() => "mtx_to_scx",
        // HDF5-based conversions
        (Some("h5ad"), _) | (None, None) if input.extension().is_some_and(|e| e == "h5ad") => {
            "h5ad_to_scx"
        }
        (Some("10x"), _) | (None, None) if input.extension().is_some_and(|e| e == "h5") => {
            "tenx_to_scx"
        }
        (_, Some("h5ad")) | (None, None) if input.extension().is_some_and(|e| e == "scx") => {
            "scx_to_h5ad"
        }
        _ => {
            return Err("Cannot determine conversion direction. Use --from/--to flags.".into());
        }
    };

    // MTX conversions are always available (no hdf5 feature needed)
    match direction {
        "mtx_to_scx" => return dispatch_mtx_to_scx(input, output, shard_size, codec),
        "scx_to_mtx" => return dispatch_scx_to_mtx(input, output),
        _ => {}
    }

    dispatch_convert(direction, input, output, shard_size, codec)
}

#[cfg(feature = "hdf5")]
fn dispatch_convert(
    direction: &str,
    input: &std::path::Path,
    output: &std::path::Path,
    shard_size: u32,
    codec: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use convert::{ConvertError, ConvertOptions};
    use indicatif::{ProgressBar, ProgressStyle};
    use scx_codec::CodecId;

    let explicit_codec = match codec {
        "auto" => None,
        "none" => Some(CodecId::None),
        "scx1" => Some(CodecId::Scx1),
        "zstd" => Some(CodecId::Zstd),
        other => {
            return Err(
                format!("Unknown codec: '{}'. Use auto, none, scx1, or zstd.", other).into(),
            )
        }
    };

    let opts = ConvertOptions {
        shard_target_rows: shard_size,
        codec: explicit_codec,
    };

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("valid template"),
    );
    pb.set_message(format!("Converting {}...", input.display()));

    let result: Result<(), ConvertError> = match direction {
        "h5ad_to_scx" => convert::h5ad_to_scx(input, output, &opts),
        "tenx_to_scx" => convert::tenx_to_scx(input, output, &opts),
        "scx_to_h5ad" => convert::scx_to_h5ad(input, output),
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
fn dispatch_convert(
    _direction: &str,
    _input: &std::path::Path,
    _output: &std::path::Path,
    _shard_size: u32,
    _codec: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    Err(
        "h5ad/10x conversion requires the 'hdf5' feature. Rebuild with: cargo build -p scx-cli --features hdf5\n\
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
