//! Copy-out rewrite operations: `compact`, `optimize`, `sort`,
//! `shuffle`, `build_csc`, `rollback`, `merge`.

use std::path::{Path, PathBuf};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyDict;

use scx_codec::CodecId;
use scx_engine::ConversionPredicateIndexOptions;
use scx_format_io::ScxReader;

use scx_ops::ReferenceSpec;

use super::*;
use crate::convert;

// ---------------------------------------------------------------------------
// C5. pyscx.compact()
// ---------------------------------------------------------------------------

/// Rewrite an SCX file reclaiming space from deleted and orphaned sections.
///
/// Optionally rebuilds predicate indexes on the compacted output: pass
/// `index_obs=[...]` / `index_var=[...]` / `index_preset=...` to mirror
/// the `scx convert` surface. Without these kwargs the predicate-index
/// sections are dropped as before (the row layout is re-sharded against
/// the post-deletion row count).
///
/// `reshape_obs`: when True, migrate legacy single-section obs metadata
/// to the atlas-scale sharded `ObsMetadataShard` layout. Mirrors
/// `scx compact --reshape-obs`. Useful for files written before ingest
/// sharded obs, or for output of a producer that still does not:
/// `pyscx.from_mudata`, or any conversion pinned with `shard_obs="off"` /
/// `force_legacy_metadata=True`. It is a **no-op on already-sharded obs**,
/// which since phase 6c includes every `scx convert` / `from_h5ad` /
/// `from_h5mu` / `from_mtx` output above `n_obs > shard_size` — including a
/// backed `from_anndata`, which routes through that same ingest.
/// Composes with the `index_*` kwargs.
///
/// Example:
///     pyscx.compact("experiment.scx", "compacted.scx")
///     pyscx.compact("experiment.scx", "compacted.scx",
///                   index_obs=["perturbation", "cell_type"])
///     pyscx.compact("experiment.scx", "compacted.scx", reshape_obs=True)
#[pyfunction]
#[pyo3(signature = (
    input, output,
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None,
    reshape_obs=false, codec="auto",
))]
#[allow(clippy::too_many_arguments)]
pub fn compact(
    py: Python<'_>,
    input: &str,
    output: &str,
    index_obs: Option<Vec<String>>,
    index_var: Option<Vec<String>>,
    index_preset: Option<String>,
    index_auto_threshold: Option<usize>,
    reshape_obs: bool,
    codec: &str,
) -> PyResult<()> {
    let input_path = PathBuf::from(input);
    let output_path = PathBuf::from(output);
    let resolved_codec = parse_codec_intent(codec)?;
    match build_index_options(index_obs, index_var, index_preset, index_auto_threshold) {
        Some(index_opts) => {
            let summary = py
                .detach(|| {
                    scx_ops::compact_with_options(
                        &input_path,
                        &output_path,
                        &scx_ops::CompactOptions {
                            index_options: index_opts,
                            reshape_obs,
                            codec: resolved_codec,
                        },
                    )
                })
                .map_err(ops_to_pyerr)?;
            process_index_summary(py, summary)
        }
        // `reshape_obs` must reach the index-options path even with no
        // `index_*` kwarg set. The `index_auto_threshold = 0` sentinel
        // keeps `user_wants_index()` false (no index built), matching
        // bare `compact()`, while still migrating obs to sharded
        // sections. Mirrors `scx-cli/src/compact.rs`.
        None if reshape_obs => {
            let summary = py
                .detach(|| {
                    scx_ops::compact_with_options(
                        &input_path,
                        &output_path,
                        &scx_ops::CompactOptions {
                            reshape_obs: true,
                            codec: resolved_codec,
                            ..Default::default()
                        },
                    )
                })
                .map_err(ops_to_pyerr)?;
            process_index_summary(py, summary)
        }
        // Still the options path so a lone `codec=` is not silently dropped;
        // `CompactOptions::default()` reproduces bare `compact()`.
        None => py
            .detach(|| {
                scx_ops::compact_with_options(
                    &input_path,
                    &output_path,
                    &scx_ops::CompactOptions {
                        codec: resolved_codec,
                        ..Default::default()
                    },
                )
            })
            .map(|_| ())
            .map_err(ops_to_pyerr),
    }
}

// ---------------------------------------------------------------------------
// C5b. pyscx.optimize()
// ---------------------------------------------------------------------------

/// Re-encode + canonicalize CSR shards to add decode sidecars and upgrade
/// to format_version 3.
///
/// This is the Python equivalent of ``scx optimize``. It re-encodes every
/// CSR shard (X, layers, obsp graphs) so the output carries decode sidecars
/// (enabling ``to_gpu_anndata`` device-decode) and stamps ``format_version = 3``
/// (canonical-CSR invariant). Single-modality files only; use ``compact()``
/// for multimodal.
///
/// Args:
///     input:  Path to the source ``.scx`` file.
///     output: Path for the optimized file. May equal ``input`` for in-place
///             upgrade (writes to a sibling tempfile, then atomically renames).
///     codec:  Per-shard codec selection.
///
///             - ``"auto"`` (default): Scx1 for low-median integer counts,
///               Zstd otherwise. Some high-median shards will NOT get a
///               decode sidecar under auto.
///             - ``"scx1"``: Force Scx1 on every integer shard — guarantees
///               a decode sidecar on all integer shards (needed for full
///               ``to_gpu_anndata`` device-decode coverage).
///     shard_obs: Migrate a legacy single-section obs table to the sharded
///             ``ObsMetadataShard`` layout.
///
///             - ``"auto"`` (default): shard only when ``n_obs >
///               shard_target_rows`` (the ``from_anndata`` threshold) — small
///               files stay single-section, atlas-scale files get sharded obs.
///             - ``"always"``: always shard a single-section obs.
///             - ``"off"``: keep the single section (faithful 1:1 copy).
///
///             An already-sharded obs is preserved as shards regardless of
///             this setting.
///     memory_budget: Cap on the memory the parallel shard re-encode may hold
///             in flight, as a binary-prefixed size string (``"512M"``,
///             ``"8G"``, ``"2GiB"``); decimal ``KB``/``MB``/``GB`` is
///             rejected. Shards are re-encoded in chunks whose whole live
///             phase fits this, so raising it buys concurrency on deep shards
///             and costs peak RSS.
///
///             ``None`` (default) is **not** unbounded: it holds 1 GiB in
///             flight, so peak memory does not scale with the machine's core
///             count. Pass a small value (``"1"``) to pin the one-shard-at-a-
///             time behaviour this function had before the re-encode became
///             parallel — a budget below one shard's phase still encodes one
///             shard, since a single shard's encode is irreducible.
///
/// Raises:
///     ValueError: If `codec` is not "auto"/"scx1", `shard_obs` is not
///         "off"/"auto"/"always", or `memory_budget` is not a valid
///         binary-prefixed size.
///     RuntimeError: If the file is multimodal, the input doesn't exist,
///         or the output already exists (no ``--force`` analogue; callers
///         should remove the target first or use ``output == input``).
///
/// Example:
///     pyscx.optimize("experiment.scx", "optimized.scx")
///     pyscx.optimize("experiment.scx", "experiment.scx")  # in-place
///     pyscx.optimize("experiment.scx", "optimized.scx", codec="scx1")
///     pyscx.optimize("atlas.scx", "atlas.opt.scx", shard_obs="always")
///     pyscx.optimize("atlas.scx", "atlas.opt.scx", memory_budget="8G")
#[pyfunction]
#[pyo3(signature = (input, output, codec="auto", shard_obs="auto", memory_budget=None))]
pub fn optimize(
    py: Python<'_>,
    input: &str,
    output: &str,
    codec: &str,
    shard_obs: &str,
    memory_budget: Option<&str>,
) -> PyResult<()> {
    let input_path = PathBuf::from(input);
    let output_path = PathBuf::from(output);
    let codec_id = match codec {
        "auto" => None,
        "scx1" => Some(CodecId::Scx1),
        other => {
            return Err(PyValueError::new_err(format!(
                "codec must be 'auto' or 'scx1', got {other:?} \
                 (other codecs drop decode sidecars, defeating the purpose of optimize)"
            )));
        }
    };
    let obs_shard_policy =
        scx_format_io::ObsShardPolicy::parse(shard_obs).map_err(PyValueError::new_err)?;
    // No-clobber guard mirroring `scx optimize` (no `--force` analogue here).
    // An in-place upgrade (`output == input`) writes a sibling tempfile and
    // atomically renames, so only a *different* pre-existing output is
    // rejected. Compare canonicalized paths when both resolve, falling back to
    // a literal compare for a not-yet-created output.
    let same_file = match (
        std::fs::canonicalize(&input_path),
        std::fs::canonicalize(&output_path),
    ) {
        (Ok(a), Ok(b)) => a == b,
        _ => input_path == output_path,
    };
    if output_path.exists() && !same_file {
        return Err(PyRuntimeError::new_err(format!(
            "output file already exists: {} (no force analogue; remove the \
             target first or pass output == input for an in-place upgrade)",
            output_path.display()
        )));
    }
    // Parsed before any file I/O, so a malformed size raises rather than
    // failing partway through a rewrite -- the rule `scx convert` follows.
    let memory_budget = match memory_budget {
        Some(spec) => Some(
            scx_format_io::MemoryBudget::parse(spec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        ),
        None => None,
    };
    py.detach(|| {
        scx_ops::optimize_with_framing(
            &input_path,
            &output_path,
            codec_id,
            obs_shard_policy,
            // Framing is unchanged: this binding has never exposed it, and
            // `scx_ops::optimize`'s wrapper passes `None` too. Only the budget
            // is new, because this call *did* change behaviour without it --
            // it held one shard at a time before the re-encode became
            // parallel, and a memory-constrained caller had no way back.
            None,
            memory_budget,
        )
        .map(|_stats| ())
    })
    .map_err(ops_to_pyerr)
}

/// Parse the `reference` kwarg of `sort` into a [`ReferenceSpec`].
///
/// Accepts `None`, a `str` (single label), a `list[str]` (label set), or a
/// dict `{"column": name}` (boolean obs column). Mirrors the CLI's
/// label-list / `col:NAME` forms. `str` is checked before `list[str]` because
/// pyo3 would otherwise iterate a string into single-character labels.
pub(crate) fn parse_reference_spec(
    v: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<ReferenceSpec>> {
    let Some(obj) = v else { return Ok(None) };
    if obj.is_none() {
        return Ok(None);
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        return match d.get_item("column")? {
            Some(col) => Ok(Some(ReferenceSpec::Column(col.extract::<String>()?))),
            None => Err(PyValueError::new_err(
                "reference dict must have a 'column' key, e.g. {'column': 'is_control'}",
            )),
        };
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Some(ReferenceSpec::Labels(vec![s])));
    }
    if let Ok(labels) = obj.extract::<Vec<String>>() {
        return Ok(Some(ReferenceSpec::Labels(labels)));
    }
    Err(PyValueError::new_err(
        "reference must be None, a str, a list[str], or {'column': name}",
    ))
}

/// Parse a `codec=` kwarg into a [`CodecSelection`].
///
/// Both `sort` and `shuffle` hardcoded `Auto` before 1D, which left the Python
/// surface unable to express something the CLI has always had — and unable to
/// follow its *own* documented advice ("pin `--codec` if output size matters",
/// `docs/sharding.md`). It bit the 1D size benchmark first: sweeping the
/// per-codec fixtures produced byte-identical outputs for every variant,
/// because the writer re-selected `auto` each time, so the sweep measured
/// auto-reselection rather than whether a permutation grows that codec.
fn parse_codec_intent(codec: &str) -> PyResult<scx_format_io::ResolvedCodec> {
    scx_format_io::resolve_codec(Some(codec)).map_err(PyValueError::new_err)
}

/// Run a built [`scx_ops::SortOptions`] off the GIL, then optionally rebuild
/// the CSC sidecar off the GIL too.
///
/// Shared by `sort` and `shuffle` so the two cannot drift on the part that
/// matters — GIL handling, error mapping, and the post-write CSC rebuild. The
/// *option construction* stays in each pyfunction, because that is exactly
/// where they legitimately differ.
fn run_sort_engine(
    py: Python<'_>,
    input_path: &Path,
    output_path: &Path,
    opts: &scx_ops::SortOptions,
    rebuild_csc: bool,
    csc_cols_per_shard: usize,
    csc_memory_limit: &str,
) -> PyResult<()> {
    py.detach(|| scx_ops::sort(input_path, output_path, opts))
        .map_err(ops_to_pyerr)?;
    if rebuild_csc {
        // Run the heavy CSC rebuild off the GIL too. Its `Box<dyn Error>` is
        // not `Send`, so map it to a `String` inside the closure to cross
        // `py.detach`.
        // NOT `None`: on a v4 output that would rewrite CSR + CSC unframed and
        // strip the row-group framing the sort just wrote. See
        // `scx_ops::framing_for_csc_rebuild`.
        let csc_framing = scx_ops::framing_for_csc_rebuild(output_path);
        py.detach(|| {
            scx_ops::rebuild_csc_inplace(
                output_path,
                csc_cols_per_shard,
                csc_memory_limit,
                csc_framing,
            )
            .map_err(|e| e.to_string())
        })
        .map_err(PyRuntimeError::new_err)?;
    }
    Ok(())
}

/// Globally reorder cells (the obs axis) of an SCX file by an obs key,
/// writing a new file with X-read locality and contiguous predicate-index
/// shard ranges for the sort key.
///
/// `by`: one or more obs columns, lexicographic in order (the leading key
/// gets the full X-read-locality benefit). `reverse`: descending on all keys.
/// Pass `memory_budget` (e.g. "4G") to force the bounded external partition
/// sort; without it the in-memory path is used. The detection bitmap and CSC
/// sidecar are dropped (the reorder invalidates them); pass `rebuild_csc=True`
/// to re-emit the column-major sidecar.
///
/// For a *random* reorder — training-batch diversity rather than query
/// locality — see `pyscx.shuffle`.
///
/// Example:
///     pyscx.sort("atlas.scx", "atlas.sorted.scx", by=["cell_type"])
#[pyfunction]
#[pyo3(signature = (
    input, output, by, reverse=false, shard_size=None, codec="auto".to_string(),
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None,
    memory_budget=None, temp_dir=None, bitmap="off".to_string(), rebuild_csc=false,
    csc_cols_per_shard=5000, csc_memory_limit="4G".to_string(),
    group_by=None, reference=None, group_target_bytes=None, group_max_bytes=None,
    group_write_block_bytes=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn sort(
    py: Python<'_>,
    input: &str,
    output: &str,
    by: Vec<String>,
    reverse: bool,
    shard_size: Option<i64>,
    codec: String,
    index_obs: Option<Vec<String>>,
    index_var: Option<Vec<String>>,
    index_preset: Option<String>,
    index_auto_threshold: Option<usize>,
    memory_budget: Option<String>,
    temp_dir: Option<String>,
    bitmap: String,
    rebuild_csc: bool,
    csc_cols_per_shard: usize,
    csc_memory_limit: String,
    group_by: Option<String>,
    reference: Option<Bound<'_, PyAny>>,
    group_target_bytes: Option<Bound<'_, PyAny>>,
    group_max_bytes: Option<Bound<'_, PyAny>>,
    group_write_block_bytes: Option<Bound<'_, PyAny>>,
) -> PyResult<()> {
    if by.is_empty() && group_by.is_none() {
        return Err(PyValueError::new_err(
            "sort requires at least one `by` column (or `group_by`)",
        ));
    }
    let input_path = PathBuf::from(input);
    let output_path = PathBuf::from(output);
    let memory_budget = match memory_budget {
        Some(s) => Some(scx_format_io::MemoryBudget::parse(&s).map_err(PyValueError::new_err)?),
        None => None,
    };
    // F1 grouped sharding (parse Python args into plain Rust before `py.detach`).
    let reference = parse_reference_spec(reference.as_ref())?;
    if reference.is_some() && group_by.is_none() {
        return Err(PyValueError::new_err(
            "reference requires group_by to be set",
        ));
    }
    let group_target_bytes = convert::parse_memory_budget(group_target_bytes.as_ref())?;
    let group_max_bytes = convert::parse_memory_budget(group_max_bytes.as_ref())?;
    let group_write_block_bytes = convert::parse_memory_budget(group_write_block_bytes.as_ref())?;
    let bitmap = scx_format_io::BitmapPolicy::parse(&bitmap).map_err(PyValueError::new_err)?;
    // Route shard_size through the shared validator (signed i64 so a negative
    // value is a clean `ValueError`, not pyo3 `OverflowError`), matching every
    // other op.
    let shard_target_rows = validate_shard_size(shard_size)?.get();
    let opts = scx_ops::SortOptions {
        by,
        reverse,
        shuffle: None,
        shard_target_rows,
        codec: parse_codec_intent(&codec)?,
        index_options: ConversionPredicateIndexOptions {
            index_obs: index_obs.unwrap_or_default(),
            index_var: index_var.unwrap_or_default(),
            index_preset,
            // The sort key is auto-added regardless; this caps auto-detection
            // of other low-cardinality columns (0 = none).
            index_auto_threshold: index_auto_threshold.unwrap_or(0),
        },
        memory_budget,
        temp_dir: temp_dir.map(PathBuf::from),
        bitmap,
        // F1 grouped sharding (7.2a): thread the grouping options through so
        // Python can write grouped files without the `scx sort --group-by` CLI.
        group_by,
        reference,
        group_target_bytes,
        group_max_bytes,
        group_write_block_bytes,
    };
    run_sort_engine(
        py,
        &input_path,
        &output_path,
        &opts,
        rebuild_csc,
        csc_cols_per_shard,
        &csc_memory_limit,
    )
}

// ---------------------------------------------------------------------------
// 1D. pyscx.shuffle()
// ---------------------------------------------------------------------------

/// Globally reorder cells (the obs axis) of an SCX file by a **seeded random
/// permutation**, writing a new file whose row order carries no residual
/// structure.
///
/// This is the training-side counterpart of `pyscx.sort`. `TrainingDataset`
/// randomizes in two levels — shard order, then a Fisher-Yates shuffle within
/// each shard group — so on a file whose rows arrived clustered (by donor,
/// plate, or cell type) batch composition is capped by `shard_group_size`, and
/// widening it costs memory linearly. Permuting once, on disk, moves that cost
/// off the training loop.
///
/// `seed` is recorded in the output's provenance and is the **only** record of
/// the permutation: the same seed on the same input always reproduces the same
/// file, and nothing else can. Note the permutation runs over *live* rows, so
/// a file with deletion vectors shuffles differently from the same file without
/// them (deletions are materialized away, as in `sort`).
///
/// Two consequences worth knowing before a multi-hour rewrite:
///
/// - **Size is not quite neutral, and `codec="auto"` is still the right
///   choice.** A permutation genuinely loses some cross-row redundancy for
///   codecs whose compression spans rows — 6-12% for `zstd`, under 1% for
///   `lz4`/`shufdelta`. That is inherent to shuffling. `auto` runs the same
///   adaptive per-shard selection `scx convert` does, so pinning a codec
///   *chooses an encoding* rather than holding the file's size; reach for
///   `codec="scx1"` when you want a permutation-invariant layout or the GPU
///   device-decode route. (This used to say `auto` re-selects and grows X
///   1.86-2.09x, and to pin the input's own codec. That growth was a bug in
///   every derived-file op — `FramingConfig::default()` meant the `fast`
///   profile — not a property of shuffling, and it is fixed.)
/// - **Shard geometry is not preserved by default.** `shard_size=None` uses the
///   16,384-row default, so a file written with a different shard size is
///   re-sharded as well as reordered — and shard size is what quantises batch
///   composition. Pass `shard_size=<the input's value>` to reorder only.
/// - **It is the inverse of a sort for queries.** Sorting collapses each
///   category's predicate-index shard ranges to one contiguous run; shuffling
///   scatters every category across every shard.
///
/// Example:
///     pyscx.shuffle("atlas.scx", "atlas.shuffled.scx", seed=42)
#[pyfunction]
#[pyo3(signature = (
    input, output, seed=42, shard_size=None, codec="auto".to_string(),
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None,
    memory_budget=None, temp_dir=None, bitmap="off".to_string(), rebuild_csc=false,
    csc_cols_per_shard=5000, csc_memory_limit="4G".to_string(),
))]
#[allow(clippy::too_many_arguments)]
pub fn shuffle(
    py: Python<'_>,
    input: &str,
    output: &str,
    seed: u64,
    shard_size: Option<i64>,
    codec: String,
    index_obs: Option<Vec<String>>,
    index_var: Option<Vec<String>>,
    index_preset: Option<String>,
    index_auto_threshold: Option<usize>,
    memory_budget: Option<String>,
    temp_dir: Option<String>,
    bitmap: String,
    rebuild_csc: bool,
    csc_cols_per_shard: usize,
    csc_memory_limit: String,
) -> PyResult<()> {
    let input_path = PathBuf::from(input);
    let output_path = PathBuf::from(output);
    let memory_budget = match memory_budget {
        Some(s) => Some(scx_format_io::MemoryBudget::parse(&s).map_err(PyValueError::new_err)?),
        None => None,
    };
    let bitmap = scx_format_io::BitmapPolicy::parse(&bitmap).map_err(PyValueError::new_err)?;
    let shard_target_rows = validate_shard_size(shard_size)?.get();
    let opts = scx_ops::SortOptions {
        // Shuffle is an order *source*, not a modifier: there is no key, and
        // the engine rejects `by` / `group_by` / `reverse` alongside it. This
        // surface simply does not expose them.
        by: Vec::new(),
        reverse: false,
        shuffle: Some(seed),
        shard_target_rows,
        codec: parse_codec_intent(&codec)?,
        index_options: ConversionPredicateIndexOptions {
            index_obs: index_obs.unwrap_or_default(),
            index_var: index_var.unwrap_or_default(),
            index_preset,
            index_auto_threshold: index_auto_threshold.unwrap_or(0),
        },
        memory_budget,
        temp_dir: temp_dir.map(PathBuf::from),
        bitmap,
        group_by: None,
        reference: None,
        group_target_bytes: None,
        group_max_bytes: None,
        group_write_block_bytes: None,
    };
    run_sort_engine(
        py,
        &input_path,
        &output_path,
        &opts,
        rebuild_csc,
        csc_cols_per_shard,
        &csc_memory_limit,
    )
}

// ---------------------------------------------------------------------------
// C5b. pyscx.build_csc()
// ---------------------------------------------------------------------------

/// Build a CSC (column-major) sidecar from an existing file's CSR shards.
///
/// Standalone equivalent of the `scx build-csc` CLI command. CSC sidecars are
/// the column-major substrate for DE / HVG / per-gene QC / pseudobulk and the
/// GPU `pdex_ref` CSC-direct route.
///
/// `output=None` (the default) adds the sidecar to `input` **in place**, staged
/// via a temp file + atomic rename so a failure leaves `input` untouched — the
/// CSC store is a sidecar *on* a file, which is how the rest of the API
/// describes it. Pass an `output` path to leave `input` alone and write a copy
/// carrying CSR + the new CSC shards.
///
/// To emit a sidecar at write time use `pyscx.from_anndata(..., csc="always")`.
///
/// Parameters:
///   input              — SCX file containing CSR shards.
///   output             — destination file (gets CSR + the new CSC shards).
///                        `None` (default) rebuilds `input` in place.
///   memory_limit       — transpose working-set budget; accepts binary-
///                        prefixed sizes (`"4G"`, `"512MiB"`). Default "4G".
///   force              — overwrite `output` if it already exists. Rejected
///                        with `output=None`, which always rewrites `input`.
///   csc_cols_per_shard — max columns per emitted CSC shard (0 = single
///                        shard, memory permitting). Default 5000.
///
/// Example:
///     pyscx.build_csc("counts.scx")                      # in place
///     pyscx.build_csc("counts.scx", "counts_csc.scx")    # copy out
#[pyfunction]
#[pyo3(signature = (input, output=None, memory_limit="4G".to_string(), force=false, csc_cols_per_shard=5000))]
pub fn build_csc(
    py: Python<'_>,
    input: &str,
    output: Option<&str>,
    memory_limit: String,
    force: bool,
    csc_cols_per_shard: usize,
) -> PyResult<()> {
    // Fail fast with a clean ValueError on a malformed size string;
    // run_build_csc re-parses it internally with the same parser, so the
    // two cannot drift.
    scx_format_io::MemoryBudget::parse(&memory_limit).map_err(PyValueError::new_err)?;
    let input_path = PathBuf::from(input);

    // `output=None` is now the in-place spelling, so an `output` that aliases
    // `input` is a mistake with an obvious fix rather than an unsupported
    // operation. (Still an error: `run_build_csc` removes `output` when `force`
    // before opening `input`, so an aliased path would delete the source.)
    let output_path = match output {
        Some(out) => {
            let output_path = PathBuf::from(out);
            let same_file = match (
                std::fs::canonicalize(&input_path),
                std::fs::canonicalize(&output_path),
            ) {
                (Ok(a), Ok(b)) => a == b,
                _ => input_path == output_path,
            };
            if same_file {
                return Err(PyValueError::new_err(
                    "input and output must be different files; pass output=None \
                     to add the CSC sidecar to `input` in place, or give a \
                     distinct output path to write a copy",
                ));
            }
            Some(output_path)
        }
        // In place, so there is nothing to overwrite. Refuse `force` rather
        // than ignore it: accepting it silently would imply a guard that does
        // not exist.
        None if force => {
            return Err(PyValueError::new_err(
                "force=True applies only when writing to an `output` path; \
                 output=None always rewrites `input`",
            ));
        }
        None => None,
    };

    // `run_build_csc` is not modality-aware — it flattens every CSR shard
    // against the single top-level n_obs × n_vars shape, which would corrupt
    // the sidecar on a multimodal input. Reject it with a clear error.
    let input_reader =
        ScxReader::open(&input_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if input_reader.is_multimodal() {
        return Err(PyValueError::new_err(
            "build_csc does not support multimodal files; subset to a single \
             modality first (scx subset --modality NAME)",
        ));
    }
    drop(input_reader);

    // Framing must come from `framing_for_csc_rebuild` on BOTH arms. Both ends
    // of the range are wrong (see its contract): `None` strips row-group
    // framing off a v4 input — via `rewrite_output_format_version(&[4], 1) == 3`
    // — and a `FramingConfig` carrying `decode_target: Some(_)` would
    // re-authorise per-shard codec re-selection, the opposite of what a sidecar
    // rebuild needs.
    let framing = scx_ops::framing_for_csc_rebuild(&input_path);

    // Both entry points return `Box<dyn Error>` (not `Send`), so stringify the
    // error inside the closure to cross `py.detach`, mirroring sort()'s CSC
    // rebuild path.
    py.detach(|| {
        match output_path {
            Some(output_path) => scx_ops::run_build_csc(
                &input_path,
                &output_path,
                &memory_limit,
                force,
                csc_cols_per_shard,
                framing,
            ),
            None => scx_ops::rebuild_csc_inplace(
                &input_path,
                csc_cols_per_shard,
                &memory_limit,
                framing,
            ),
        }
        .map_err(|e| e.to_string())
    })
    .map_err(PyRuntimeError::new_err)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// C6. pyscx.rollback()
// ---------------------------------------------------------------------------

/// Roll back an SCX file to a previous manifest version.
///
/// If `to_seq` is None, rolls back one version.
/// If `to_seq` is given, rolls back to that specific sequence number.
///
/// Example:
///     pyscx.rollback("experiment.scx")           # roll back one version
///     pyscx.rollback("experiment.scx", to_seq=3)  # roll back to specific version
#[pyfunction]
#[pyo3(signature = (path, to_seq=None))]
pub fn rollback(path: &str, to_seq: Option<u64>) -> PyResult<()> {
    let p = Path::new(path);
    match to_seq {
        Some(seq) => scx_ops::rollback_to(p, seq).map_err(ops_to_pyerr),
        None => scx_ops::rollback(p).map_err(ops_to_pyerr),
    }
}

// ---------------------------------------------------------------------------
// C7. pyscx.merge()
// ---------------------------------------------------------------------------

/// Merge multiple SCX files into a single output file.
///
/// Requires at least 2 input files. All must have the same n_vars.
///
/// Optionally rebuilds predicate indexes on the merged output: pass
/// `index_obs=[...]` / `index_var=[...]` / `index_preset=...` to mirror
/// the `scx convert` surface. Without these kwargs the merged output
/// has NO predicate-index sections — query-time `filter_obs` pushdown
/// falls back to a full scan. This was the silent-data-loss bug
/// reported against multi-input atlas builds.
///
/// Validation kwargs (strict by default; opt back in to the
/// pre-strict behaviour explicitly):
///
/// * `assume_identical_var=False` — when `False` (default), merge
///   compares every input's var batch column-by-column against
///   input 0 and errors on mismatch. Set `True` if you've already
///   verified the gene axis upstream and want the count-only check.
/// * `assume_identical_obs=False` — when `False` (default), merge
///   compares every input's obs schema against input 0 (column names
///   + dtypes, normalised through the logical-lossy schema) and
///   errors on mismatch. Set `True` when the caller has already
///   validated obs columns.
/// * `uns_policy=None` — controls how the merged file's `uns`
///   section is built. `None` / `"first"` keeps input 0's payload
///   verbatim; `"require-equal"` errors on any disagreement;
///   `"namespace"` writes a `{"input_N": ...}` wrapper; `"summary"`
///   keeps input 0 and records a `_scx_uns_conflicts` array.
///   Applied independently at the global and per-modality levels
///   for multimodal inputs.
///
/// Raises `RuntimeError` when the inputs disagree about what they carry:
/// an `obsm` or `obsp` key that some inputs have and others lack is a
/// hard error naming the axis, the key and the input — the same answer a
/// missing *layer* has always had. Merging per-sample files where one
/// lacks `X_umap` used to succeed and silently produce an atlas without
/// it. Drop the key from the others, or add it to the one, before
/// merging. An `obsp` key whose `data` column disagrees in dtype or
/// nullability across inputs is refused for the same reason: the merged
/// shards are read back under one schema, so writing them would produce
/// a graph that cannot be loaded.
///
/// File-scope `obsp` (COO) is carried, rebased into the merged obs
/// space, and `varp` comes from input 0. The **CSR-backed** `obsp`
/// encoding and modality-scoped pairwise graphs are dropped with a
/// warning; `sort_by` refuses COO `obsp` as it already refuses `obsm`.
///
/// Example:
///     pyscx.merge(["batch1.scx", "batch2.scx", "batch3.scx"], "atlas.scx")
///     pyscx.merge(["a.scx", "b.scx"], "merged.scx",
///                 index_obs=["perturbation", "cell_type"])
///     pyscx.merge(["a.scx", "b.scx"], "merged.scx",
///                 assume_identical_var=True, uns_policy="namespace")
#[pyfunction]
#[pyo3(signature = (
    inputs, output,
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None,
    assume_identical_var=false, assume_identical_obs=false, uns_policy=None,
    sort_by=None, reverse=false, codec="auto",
))]
#[allow(clippy::too_many_arguments)]
pub fn merge(
    py: Python<'_>,
    inputs: Vec<String>,
    output: &str,
    index_obs: Option<Vec<String>>,
    index_var: Option<Vec<String>>,
    index_preset: Option<String>,
    index_auto_threshold: Option<usize>,
    assume_identical_var: bool,
    assume_identical_obs: bool,
    uns_policy: Option<String>,
    sort_by: Option<Vec<String>>,
    reverse: bool,
    codec: &str,
) -> PyResult<()> {
    let resolved_codec = parse_codec_intent(codec)?;
    if inputs.len() < 2 {
        return Err(PyValueError::new_err(
            "merge requires at least 2 input files",
        ));
    }

    let input_paths: Vec<PathBuf> = inputs.iter().map(PathBuf::from).collect();
    let input_refs: Vec<&Path> = input_paths.iter().map(|p| p.as_path()).collect();
    let output_path = PathBuf::from(output);

    // Parse the optional uns_policy kwarg into the enum. Default
    // (None) preserves `UnsPolicy::First` = today's behaviour: read
    // the first input's `uns` verbatim, drop the rest.
    let uns_policy_parsed = match uns_policy.as_deref() {
        Some(s) => scx_ops::UnsPolicy::parse(s).ok_or_else(|| {
            PyValueError::new_err(format!(
                "invalid uns_policy '{s}': expected one of \
                 first, require-equal, namespace, summary"
            ))
        })?,
        None => scx_ops::UnsPolicy::First,
    };

    let sort_by = sort_by.unwrap_or_default();
    let want_sort = !sort_by.is_empty();
    let want_policy = assume_identical_var
        || assume_identical_obs
        || uns_policy_parsed != scx_ops::UnsPolicy::First;
    match build_index_options(index_obs, index_var, index_preset, index_auto_threshold) {
        Some(index_opts) => {
            let merge_opts = scx_ops::MergeOptions {
                index_options: index_opts,
                assume_identical_var,
                assume_identical_obs,
                codec: resolved_codec,
                uns_policy: uns_policy_parsed,
                shard_target_rows: None,
                sort_by,
                sort_reverse: reverse,
            };
            let summary = py
                .detach(|| scx_ops::merge_with_options(&input_refs, &output_path, &merge_opts))
                .map_err(ops_to_pyerr)?;
            process_index_summary(py, summary)
        }
        None if want_policy || want_sort => {
            let merge_opts = scx_ops::MergeOptions {
                index_options: scx_engine::ConversionPredicateIndexOptions {
                    index_obs: Vec::new(),
                    index_var: Vec::new(),
                    index_preset: None,
                    index_auto_threshold: 0,
                },
                assume_identical_var,
                assume_identical_obs,
                codec: resolved_codec,
                uns_policy: uns_policy_parsed,
                shard_target_rows: None,
                sort_by,
                sort_reverse: reverse,
            };
            let summary = py
                .detach(|| scx_ops::merge_with_options(&input_refs, &output_path, &merge_opts))
                .map_err(ops_to_pyerr)?;
            process_index_summary(py, summary)
        }
        // Still the options path so a lone `codec=` is not silently dropped;
        // `MergeOptions::default()` reproduces the bare `merge` wrapper.
        None => py
            .detach(|| {
                scx_ops::merge_with_options(
                    &input_refs,
                    &output_path,
                    &scx_ops::MergeOptions {
                        codec: resolved_codec,
                        ..Default::default()
                    },
                )
            })
            .map(|_| ())
            .map_err(ops_to_pyerr),
    }
}
