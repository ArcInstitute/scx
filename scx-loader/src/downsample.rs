//! Seeded per-row count downsampling for the sparse cell-set gather path.
//!
//! Ports the *semantics* — not the bit stream — of state3's `RawCountDownsampler`
//! (`state3/src/state3/data/downsample.py`) into Rust, so the native collate path
//! can offer count-depth augmentation instead of refusing it. Everything here is
//! pure (no Python, no I/O) so it can be unit-tested directly.
//!
//! # Not bit-reproducible against the Python implementation, by decision
//!
//! Bit-parity with `RawCountDownsampler` would mean porting numpy's `PCG64` +
//! `SeedSequence` entropy expansion, its BTPE/inversion binomial, and its
//! sequential-conditional multinomial. That couples SCX to numpy internals for no
//! modelling benefit, so it was explicitly ruled out (owner decision, 2026-07-30).
//! **Consequence: runs downsampled by the Python implementation are not
//! reproducible under this one.** What *is* contractual is the semantics below
//! plus the golden fixture (`tests/data/downsample_golden.json`), which state3
//! adopts.
//!
//! # Semantics (matched to the Python reference on purpose)
//!
//! 1. Clip negatives to zero. The data contract says counts are non-negative;
//!    `rint`-ing and then sampling `Binomial(negative_n, p)` is nonsense (numpy
//!    raises), so the clip is a precondition of the sampler, not a nicety.
//! 2. Round to integer trials with **ties-to-even** (`round_ties_even`), matching
//!    `np.rint`. Rust's `f32::round()` is ties-away-from-zero and would disagree
//!    on exact halves.
//! 3. If the row's library size is `0` or already `<= target`, **do not sample —
//!    but still write back the rounded values.** This mirrors
//!    `downsample.py:78-79`: enabling downsampling integerises *every* cell, not
//!    only the ones above target. Surprising enough to be mistaken for a bug, so
//!    it is pinned by a test.
//! 4. Sample. `Binomial` draws each element independently at `p = target / lib`,
//!    so the realised depth hits `target` only in expectation. `Multinomial`
//!    redistributes exactly `target` counts.
//! 5. Prune resulting zeros from both `indices` and `data` (mirrors
//!    `dataset.py:119-121`), keeping the row canonical (sorted, unique, no
//!    explicit zeros introduced by sampling).
//!
//! # Keying: `(seed, method, file_identity, row)`
//!
//! Per-row derivation is not a stylistic choice — it is what makes the primitive
//! safe here. `read_rows_with`'s scatter callbacks fire in shard-grouped order and
//! `collate_gathered`'s rayon loop is unordered, so a single shared sequential RNG
//! would produce different counts run to run. A key derived from the row's own
//! identity is invariant to scheduling.
//!
//! `file_identity` is a stable hash of the file's **canonical path**, not its
//! `file_id`. `file_id` is loader-construction order, so keying on it would make a
//! reordered manifest — or a three-file debugging subset — silently redraw every
//! cell. The Python reference keys on the resolved path for exactly this reason and
//! pins it with a test; see [`file_identity`].
//!
//! The `method` is part of the key (as it is in the reference), so the two methods
//! draw independent streams for the same cell.

use rand::distributions::Distribution;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::Binomial;

use crate::error::{LoaderError, Result};
use crate::seed::splitmix64;

/// Which sampler redistributes the row's counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DownsampleMethod {
    /// Independent `Binomial(count_i, target / library_size)` per element. The
    /// realised library size hits `target` in expectation, not exactly.
    Binomial,
    /// Redistribute exactly `target` counts across the row in proportion to the
    /// observed counts (sampling with replacement, as `numpy.random.multinomial`).
    Multinomial,
}

impl DownsampleMethod {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "binomial" => Ok(Self::Binomial),
            "multinomial" => Ok(Self::Multinomial),
            other => Err(LoaderError::ConfigError {
                reason: format!(
                    "unknown downsample method {other:?} (expected \"binomial\" or \
                     \"multinomial\")"
                ),
            }),
        }
    }

    /// Domain-separation tag folded into the RNG key so the two methods draw
    /// independent streams for the same cell.
    #[inline]
    fn tag(self) -> u64 {
        match self {
            Self::Binomial => 0x1,
            Self::Multinomial => 0x2,
        }
    }
}

/// Run-constant downsample configuration.
#[derive(Clone, Debug)]
pub struct DownsampleConfig {
    /// Counts per cell to keep. Rows already at or below this are left alone
    /// (beyond the integerisation in step 3 of the module docs).
    pub target_library_size: u64,
    pub method: DownsampleMethod,
    pub seed: u64,
    /// Stable per-`file_id` identity, indexed by `file_id`. Built from the file
    /// paths via [`file_identity`]; carried on the config so the loader
    /// constructor grows by one `Option` rather than two parameters.
    pub file_identities: Vec<u64>,
}

impl DownsampleConfig {
    /// Identity for `file_id`, or `0` when no table was supplied.
    ///
    /// The fallback is a **constant**, not `file_id`. Falling back to `file_id`
    /// would reintroduce construction-order keying — the exact scheme this module
    /// rejects, because a reordered manifest or a debugging subset then silently
    /// redraws every cell. A constant instead keys on `(seed, method, row)` alone:
    /// correct for a single file, and ambiguous across files, which is why
    /// [`crate::sparse_cellset::SparseCellSetLoader::new`] refuses an empty table
    /// when there is more than one file rather than letting the ambiguity ship.
    /// This matches the standalone `downsample_counts_csr` convention exactly, so
    /// the two entry points cannot disagree.
    #[inline]
    pub fn identity_for(&self, file_id: u32) -> u64 {
        self.file_identities
            .get(file_id as usize)
            .copied()
            .unwrap_or(0)
    }

    /// Reject a configuration that cannot sample.
    pub fn validate(&self) -> Result<()> {
        if self.target_library_size == 0 {
            return Err(LoaderError::ConfigError {
                reason: "downsample target_library_size must be > 0".into(),
            });
        }
        Ok(())
    }
}

/// Stable 64-bit identity for a file path, for use as RNG key material.
///
/// Canonicalises first so two spellings of the same file (relative vs absolute,
/// through a symlink, with a `..` component) key identically — the property the
/// Python reference gets from `Path(...).resolve()`. A path that cannot be
/// canonicalised (not yet created, permission denied) falls back to hashing the
/// literal string, which is still deterministic, just not alias-invariant.
///
/// `~` is **not** expanded: callers pass paths that Python has already expanded.
pub fn file_identity(path: &str) -> u64 {
    let canonical = std::fs::canonicalize(path)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| path.to_string());
    let digest = blake3::hash(canonical.as_bytes());
    u64::from_le_bytes(
        digest.as_bytes()[..8]
            .try_into()
            .expect("blake3 >= 8 bytes"),
    )
}

/// Derive this row's RNG seed from `(seed, method, file_identity, row)`.
///
/// Chained rather than additive — see [`crate::seed`] for why. This four-component
/// key is where the convention started; `shuffle.rs` and `decode_stage.rs` were
/// converted to it later.
///
/// **Do not change this derivation.** Its output is pinned by a blake3 golden
/// (`downsample_tests.rs`) that `state3` also consumes, so a change here is a
/// cross-repo break — see the failure message on that test for the full
/// regeneration procedure.
#[inline]
fn row_seed(seed: u64, method: DownsampleMethod, file_identity: u64, row: u64) -> u64 {
    let mut h = splitmix64(seed);
    h = splitmix64(h ^ method.tag());
    h = splitmix64(h ^ file_identity);
    splitmix64(h ^ row)
}

/// Clip negative values to zero, in place.
///
/// Separate from downsampling because it applies unconditionally to the emitted
/// CSR: the gather previously handed negatives straight through to Python while
/// the collate kernel clipped them lazily per read, so the two disagreed on what
/// the row contained. Note this does **not** prune the zeros it creates — the
/// Python reference prunes only inside its downsample branch, and a caller reading
/// nnz should see the same structure with and without a clip.
///
/// Uses `f32::max(0.0)` rather than a `< 0.0` test, and that is deliberate: `max`
/// returns the non-NaN operand, so **NaN maps to 0** — exactly what the collate
/// kernel's own `raw.max(0.0)` reads do. A `< 0.0` test leaves NaN untouched and
/// would have left the two disagreeing about the row's contents for NaN inputs,
/// which is the very thing this clip exists to fix.
pub fn clip_negatives(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = v.max(0.0);
    }
}

/// Downsample one row's counts in place, pruning any resulting zeros.
///
/// Assumes `indices` / `data` are parallel and the row is canonical (sorted
/// ascending, unique) — which is what `remap_row` guarantees. The multinomial
/// path walks elements in slice order, so that canonical order is part of the
/// contract: reordering the row changes the draw.
///
/// Negatives are clipped first (see [`clip_negatives`]), so callers need not
/// pre-clip; doing both is harmless.
///
/// Counts round-trip through `u64` and back to `f32`, so a sampled count above
/// `2^24` would lose precision on the way out. That is the same bound the collate
/// kernel already assumes for its `library_size` sum, and it is not reachable in
/// practice: the *output* is capped at `target_library_size`, which is a
/// per-cell sequencing depth (typically `1e3`–`1e5`). A caller who genuinely
/// wants a `>2^24` target is outside this contract.
pub fn downsample_row(
    indices: &mut Vec<i32>,
    data: &mut Vec<f32>,
    cfg: &DownsampleConfig,
    file_identity: u64,
    row: u64,
) {
    debug_assert_eq!(indices.len(), data.len(), "indices/data must be parallel");
    if data.is_empty() {
        return;
    }

    // Steps 1-2: clip, then integerise with ties-to-even (np.rint).
    //
    // Non-finite values are treated as 0, not propagated. NaN falls out of
    // `max(0.0)` as 0 (matching the kernel); `+inf` is rejected explicitly because
    // there is no meaningful trial count for it — `inf as u64` saturates to
    // `u64::MAX`, which would hand the sampler a nonsense `n` and produce garbage
    // rather than fail. An infinite count is corrupt input either way.
    let mut trials: Vec<u64> = Vec::with_capacity(data.len());
    let mut library_size: u64 = 0;
    for &v in data.iter() {
        let clipped = v.max(0.0);
        let n = if clipped.is_finite() {
            clipped.round_ties_even() as u64
        } else {
            0
        };
        library_size = library_size.saturating_add(n);
        trials.push(n);
    }

    // Step 3: no sampling needed — but the rounded values are still written back,
    // matching the reference (`downsample.py:78-79`). Never upsample.
    if library_size == 0 || library_size <= cfg.target_library_size {
        write_back(indices, data, &trials);
        return;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(row_seed(cfg.seed, cfg.method, file_identity, row));
    let sampled = match cfg.method {
        DownsampleMethod::Binomial => {
            sample_binomial(&trials, library_size, cfg.target_library_size, &mut rng)
        }
        DownsampleMethod::Multinomial => {
            sample_multinomial(&trials, library_size, cfg.target_library_size, &mut rng)
        }
    };
    write_back(indices, data, &sampled);
}

/// Independent per-element `Binomial(trials_i, target / library_size)`.
fn sample_binomial(
    trials: &[u64],
    library_size: u64,
    target: u64,
    rng: &mut ChaCha8Rng,
) -> Vec<u64> {
    let p = target as f64 / library_size as f64;
    trials
        .iter()
        .map(|&n| {
            if n == 0 {
                0
            } else {
                // `p` is in (0, 1) here: target < library_size on this path and
                // both are > 0, so Binomial::new cannot fail.
                Binomial::new(n, p)
                    .expect("binomial p in (0,1) and n > 0")
                    .sample(rng)
            }
        })
        .collect()
}

/// Redistribute exactly `target` counts via sequential conditional binomials —
/// numpy's own multinomial scheme, and the reason `rand_distr` (which has no
/// multinomial) is enough.
///
/// For element `i`, draw `Binomial(remaining_target, trials_i / remaining_lib)`;
/// the conditional probability is `p_i / (1 - Σ_{j<i} p_j)`. The last non-zero
/// element absorbs whatever is left, so the total is exact.
fn sample_multinomial(
    trials: &[u64],
    library_size: u64,
    target: u64,
    rng: &mut ChaCha8Rng,
) -> Vec<u64> {
    let mut out = vec![0u64; trials.len()];
    let mut remaining_target = target;
    let mut remaining_lib = library_size;

    for (i, &n) in trials.iter().enumerate() {
        if remaining_target == 0 {
            break;
        }
        if n == 0 {
            continue;
        }
        // Every later element is zero, so this one absorbs the remainder. This is
        // also what makes the total exact: the last non-zero element always lands
        // here.
        if n >= remaining_lib {
            out[i] = remaining_target;
            break;
        }
        let p = n as f64 / remaining_lib as f64;
        let k = Binomial::new(remaining_target, p.clamp(0.0, 1.0))
            .expect("binomial p in [0,1] and n > 0")
            .sample(rng)
            .min(remaining_target);
        out[i] = k;
        remaining_target -= k;
        remaining_lib -= n;
    }
    out
}

/// Overwrite `data` with `counts` and drop entries that sampled to zero.
fn write_back(indices: &mut Vec<i32>, data: &mut Vec<f32>, counts: &[u64]) {
    let mut w = 0usize;
    for (r, &c) in counts.iter().enumerate() {
        if c == 0 {
            continue;
        }
        indices[w] = indices[r];
        data[w] = c as f32;
        w += 1;
    }
    indices.truncate(w);
    data.truncate(w);
}

/// Resolve the three `downsample_*` kwargs into a config, or `None`.
///
/// Validation lives here — at the public entry, not at the routing site — so both
/// Python surfaces reject the same shapes with the same message. Supplying a
/// method or a seed without a target is a config error rather than a silent
/// no-op: it is exactly the typo that would leave a training run un-augmented
/// while looking configured.
pub fn resolve_downsample_config(
    paths: &[String],
    target: Option<u64>,
    method: Option<&str>,
    seed: Option<u64>,
) -> Result<Option<DownsampleConfig>> {
    let Some(target) = target else {
        if method.is_some() || seed.is_some() {
            return Err(LoaderError::ConfigError {
                reason: "downsample_method / downsample_seed require \
                         downsample_target_library_size; without a target nothing is \
                         downsampled"
                    .to_string(),
            });
        }
        return Ok(None);
    };
    if target == 0 {
        return Err(LoaderError::ConfigError {
            reason: "downsample_target_library_size must be > 0".to_string(),
        });
    }
    // Default matches the Python reference's DownsampleConfig default.
    let method = DownsampleMethod::parse(method.unwrap_or("multinomial"))?;
    Ok(Some(DownsampleConfig {
        target_library_size: target,
        method,
        seed: seed.unwrap_or(0),
        file_identities: paths.iter().map(|p| file_identity(p)).collect(),
    }))
}

#[cfg(test)]
#[path = "downsample_tests.rs"]
mod tests;
