//! Whether a rewrite op's output carries a CSC sidecar, and how it is built.
//!
//! A sidecar used to be dropped by every rewrite op (`compact`, `merge`,
//! `optimize`, `sort`, `subset`) and restored, if the caller asked, by a second
//! pass over the finished output. It is now built in the same pass as the op's
//! X shards (see `ScxWriter::enable_csc_sidecar`), which makes carrying it
//! cheap enough to be the default: an op whose input had a sidecar emits one.
//!
//! Multimodal files are not covered yet — their sidecars are per modality and
//! the writer's sink is single-modality — so under [`CscOutput::Carry`] a
//! multimodal input's sidecar is dropped with a warning, and
//! [`CscOutput::Always`] is refused before the output is created.

use std::path::PathBuf;

use scx_format_io::section::SectionType;
use scx_format_io::{CscBuildOptions, MemoryBudget, ScxReader, ScxWriter};

use crate::error::{OpsError, Result};

/// What a rewrite op does about the output's CSC sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CscOutput {
    /// Build one iff an input had one (the default).
    #[default]
    Carry,
    /// Build one whether or not an input had one.
    Always,
    /// Build none.
    Off,
}

/// [`CscOutput`] plus the build parameters `scx build-csc` takes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CscCarryOptions {
    pub mode: CscOutput,
    /// Upper bound on each sidecar shard's column count (`--csc-cols-per-shard`).
    pub cols_per_shard: usize,
    /// The sidecar budget (`--csc-memory-limit`), in the forms
    /// `scx convert --memory-budget` accepts. It sizes the sidecar's shard
    /// widths as well as the builder's resident buckets, exactly as
    /// `scx build-csc --memory-limit` does.
    pub memory_limit: String,
    /// Where the builder spills; `None` is the output's own directory.
    pub temp_dir: Option<PathBuf>,
    /// Row-group framing for a sidecar on a v4 output; `None` is the default
    /// config (see `CscBuildOptions::framing`). Only `scx convert` sets it, to
    /// frame the sidecar at its `--row-group-rows`.
    pub framing: Option<scx_format_io::FramingConfig>,
}

impl Default for CscCarryOptions {
    fn default() -> Self {
        Self {
            mode: CscOutput::Carry,
            cols_per_shard: 5000,
            memory_limit: "4G".to_string(),
            temp_dir: None,
            framing: None,
        }
    }
}

impl CscCarryOptions {
    /// Options for `mode` with the default build parameters.
    pub fn with_mode(mode: CscOutput) -> Self {
        Self {
            mode,
            ..Self::default()
        }
    }

    /// Decide, before the output is created, whether it gets a sidecar.
    ///
    /// `Err` only for a request that cannot be honoured — `Always` on a
    /// multimodal input, or an unparseable budget — so a refusal leaves no
    /// output behind.
    pub fn resolve(
        &self,
        op: &str,
        any_input_has_csc: bool,
        multimodal: bool,
    ) -> Result<Option<CscBuildOptions>> {
        let wanted = match self.mode {
            CscOutput::Off => {
                if any_input_has_csc {
                    log::info!("scx {op}: --csc off, so the output carries no CSC sidecar");
                }
                false
            }
            CscOutput::Carry => any_input_has_csc,
            CscOutput::Always => true,
        };
        if !wanted {
            return Ok(None);
        }
        if multimodal {
            if self.mode == CscOutput::Always {
                return Err(OpsError::InvalidInput(format!(
                    "scx {op}: --csc always is not supported for a multimodal file yet; \
                     a multimodal sidecar is built per modality"
                )));
            }
            log::warn!(
                "scx {op}: the input's CSC sidecar is not carried through a multimodal \
                 {op}; the output has none"
            );
            return Ok(None);
        }
        let memory_bytes = MemoryBudget::parse(&self.memory_limit)
            .map_err(|e| OpsError::InvalidInput(format!("--csc-memory-limit: {e}")))?;
        Ok(Some(CscBuildOptions {
            cols_per_shard: self.cols_per_shard,
            memory_bytes: usize::try_from(memory_bytes).map_err(|_| {
                OpsError::InvalidInput(format!(
                    "--csc-memory-limit {} exceeds this platform's address space",
                    self.memory_limit
                ))
            })?,
            spill_root: self.temp_dir.clone(),
            framing: self.framing,
            ..Default::default()
        }))
    }
}

/// Whether `reader`'s file carries a CSC sidecar, single- or multi-modality.
pub fn reader_has_csc(reader: &ScxReader) -> bool {
    reader
        .catalog()
        .entries
        .iter()
        .any(|e| e.section_type == SectionType::CscShard)
}

/// Enable the same-pass sidecar on `writer` if `build` says so. Call before
/// the first X shard; pair with [`emit`] after the last.
pub fn enable(writer: &mut ScxWriter, build: &Option<CscBuildOptions>) -> Result<()> {
    if let Some(opts) = build {
        writer.enable_csc_sidecar(opts.clone())?;
    }
    Ok(())
}

/// Emit the same-pass sidecar right after the last X shard, so its buckets are
/// released before the rest of the output is written and the staged-catalog
/// audit sees it.
pub fn emit(writer: &mut ScxWriter, op: &str) -> Result<()> {
    if let Some(stats) = writer.emit_csc_sidecar()? {
        if stats.spill_bytes > 0 {
            log::info!(
                "scx {op}: the CSC sidecar builder spilled {} bytes",
                stats.spill_bytes
            );
        }
    }
    Ok(())
}
