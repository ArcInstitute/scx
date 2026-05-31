// CSC sidecar generation policy. Mirrors the `--csc off|auto|always`
// CLI flag and the `csc="..."` pyscx kwarg, in the same spirit as
// [`crate::bitmap::BitmapPolicy`].
//
// Lives in `scx-format` (rather than `scx-ops` / `scx-convert`) so the
// CPU-only pyscx in-memory write path — where `scx-convert` is behind the
// `hdf5` feature — can still drive the policy without pulling in the
// converter crate.

/// Default `Auto`-mode observation-count threshold: build a CSC sidecar
/// only for datasets with at least this many cells. Overridable via the
/// `SCX_CSC_AUTO_OBS_THRESHOLD` environment variable.
pub const AUTO_CSC_OBS_THRESHOLD: u64 = 50_000;
/// Default `Auto`-mode variable-count threshold: build a CSC sidecar only
/// for datasets with at least this many genes. Overridable via the
/// `SCX_CSC_AUTO_VARS_THRESHOLD` environment variable.
pub const AUTO_CSC_VARS_THRESHOLD: u64 = 5_000;

const OBS_THRESHOLD_ENV: &str = "SCX_CSC_AUTO_OBS_THRESHOLD";
const VARS_THRESHOLD_ENV: &str = "SCX_CSC_AUTO_VARS_THRESHOLD";

/// CSC-sidecar generation policy.
///
/// CSC sidecars are the column-major substrate for column algorithms (DE,
/// HVG, per-gene QC, pseudobulk) and the GPU `pdex_ref` v3 CSC-direct route.
/// `Auto` builds a sidecar when the dataset is large enough that the
/// column-axis acceleration pays for the extra write-time transpose + storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CscPolicy {
    /// Never emit a CSC sidecar.
    #[default]
    Off,
    /// Emit a CSC sidecar when the dataset passes the size heuristic
    /// (`n_obs >= obs_threshold && n_vars >= vars_threshold`).
    Auto,
    /// Always emit a CSC sidecar regardless of dataset size.
    Always,
}

impl CscPolicy {
    /// Parse the CLI / Python form (`"off" | "auto" | "always"`). Unknown
    /// values return `Err(reason)` for the caller to wrap in
    /// `ConvertError::Other` / `PyValueError` as appropriate.
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s {
            "off" => Ok(Self::Off),
            "auto" => Ok(Self::Auto),
            "always" => Ok(Self::Always),
            other => Err(format!(
                "invalid csc value '{other}'; expected off|auto|always"
            )),
        }
    }

    /// Resolve the policy to a build/no-build decision once the matrix
    /// shape is known. `Auto` compares `(n_obs, n_vars)` against the
    /// (env-tunable) thresholds; both must be met.
    pub fn should_build_csc(self, n_obs: u64, n_vars: u64) -> bool {
        match self {
            Self::Off => false,
            Self::Always => true,
            Self::Auto => n_obs >= auto_obs_threshold() && n_vars >= auto_vars_threshold(),
        }
    }
}

/// Read a `u64` threshold from `var`, falling back to `default` when the
/// variable is unset or cannot be parsed (degrade safely rather than fail a
/// conversion on a malformed env value).
fn threshold_from_env(var: &str, default: u64) -> u64 {
    match std::env::var(var) {
        Ok(s) => s.trim().parse::<u64>().unwrap_or(default),
        Err(_) => default,
    }
}

/// `Auto`-mode observation-count threshold (env-overridable).
pub fn auto_obs_threshold() -> u64 {
    threshold_from_env(OBS_THRESHOLD_ENV, AUTO_CSC_OBS_THRESHOLD)
}

/// `Auto`-mode variable-count threshold (env-overridable).
pub fn auto_vars_threshold() -> u64 {
    threshold_from_env(VARS_THRESHOLD_ENV, AUTO_CSC_VARS_THRESHOLD)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_round_trips() {
        assert_eq!(CscPolicy::parse("off").unwrap(), CscPolicy::Off);
        assert_eq!(CscPolicy::parse("auto").unwrap(), CscPolicy::Auto);
        assert_eq!(CscPolicy::parse("always").unwrap(), CscPolicy::Always);
    }

    #[test]
    fn parse_rejects_junk() {
        assert!(CscPolicy::parse("yes").is_err());
        assert!(CscPolicy::parse("Always").is_err());
        assert!(CscPolicy::parse("").is_err());
    }

    #[test]
    fn default_is_off() {
        assert_eq!(CscPolicy::default(), CscPolicy::Off);
        assert!(!CscPolicy::default().should_build_csc(u64::MAX, u64::MAX));
    }

    #[test]
    fn off_and_always_ignore_shape() {
        assert!(!CscPolicy::Off.should_build_csc(10_000_000, 100_000));
        assert!(CscPolicy::Always.should_build_csc(1, 1));
    }

    // Threshold-sensitive cases mutate process-global env vars, so they are
    // serialized into one test to avoid cross-test interference.
    #[test]
    fn auto_threshold_matrix_with_env_override() {
        // Defaults: 50_000 obs, 5_000 vars; both required.
        std::env::remove_var(OBS_THRESHOLD_ENV);
        std::env::remove_var(VARS_THRESHOLD_ENV);
        assert_eq!(auto_obs_threshold(), AUTO_CSC_OBS_THRESHOLD);
        assert_eq!(auto_vars_threshold(), AUTO_CSC_VARS_THRESHOLD);

        assert!(CscPolicy::Auto.should_build_csc(50_000, 5_000)); // both met
        assert!(CscPolicy::Auto.should_build_csc(60_000, 20_000));
        assert!(!CscPolicy::Auto.should_build_csc(49_999, 20_000)); // obs short
        assert!(!CscPolicy::Auto.should_build_csc(60_000, 4_999)); // vars short
        assert!(!CscPolicy::Auto.should_build_csc(10, 10));

        // Env override lowers both thresholds to 0 → always builds under Auto.
        std::env::set_var(OBS_THRESHOLD_ENV, "0");
        std::env::set_var(VARS_THRESHOLD_ENV, "0");
        assert_eq!(auto_obs_threshold(), 0);
        assert_eq!(auto_vars_threshold(), 0);
        assert!(CscPolicy::Auto.should_build_csc(1, 1));

        // Malformed env value falls back to the default.
        std::env::set_var(OBS_THRESHOLD_ENV, "not-a-number");
        assert_eq!(auto_obs_threshold(), AUTO_CSC_OBS_THRESHOLD);

        std::env::remove_var(OBS_THRESHOLD_ENV);
        std::env::remove_var(VARS_THRESHOLD_ENV);
    }
}
