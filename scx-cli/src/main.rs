use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process;

mod convert;
mod info;
mod validate;

#[derive(Parser)]
#[command(name = "scx", about = "SCX file format tool")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert between h5ad/10x and SCX formats
    Convert {
        /// Input file path
        input: PathBuf,
        /// Output file path
        output: PathBuf,
        /// Input format: h5ad, 10x
        #[arg(long)]
        from: Option<String>,
        /// Output format: h5ad
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
    },
    /// Validate SCX file checksums
    Validate {
        /// SCX file to validate
        file: PathBuf,
        /// Print checksum values
        #[arg(long)]
        verbose: bool,
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
        Commands::Info { file } => info::run_info(&file),
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
        "Convert requires the 'hdf5' feature. Rebuild with: cargo build -p scx-cli --features hdf5"
            .into(),
    )
}
