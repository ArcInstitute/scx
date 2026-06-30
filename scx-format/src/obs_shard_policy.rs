// obs-metadata sharding policy for `scx optimize`. Mirrors the
// `--shard-obs off|auto|always` CLI flag and the `shard_obs="..."` pyscx
// kwarg, in the same spirit as `CscPolicy`.
//
// Lives in `scx-format` (rather than `scx-ops` / `scx-convert`) so the
// CPU-only pyscx path can drive the policy without pulling in those crates.

/// Policy for migrating a single-section (legacy `ObsMetadata`) obs table to
/// the sharded `ObsMetadataShard` layout during `optimize`.
///
/// Only governs the *single-section input* case. An obs table that is already
/// sharded is always preserved as shards (streamed through), regardless of
/// policy — optimize never collapses or re-sizes existing obs shards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ObsShardPolicy {
    /// Keep a single-section obs as a single section (the historical
    /// `optimize` behaviour — a faithful 1:1 layout copy).
    Off,
    /// Shard a single-section obs iff `n_obs > shard_target_rows` — the same
    /// threshold `from_anndata` uses to decide obs sharding on the write path.
    /// Small files stay single-section; atlas-scale files get sharded obs.
    /// Default.
    #[default]
    Auto,
    /// Always shard a single-section obs (chunked by `shard_target_rows`).
    Always,
}

impl ObsShardPolicy {
    /// Parse the CLI / Python form (`"off" | "auto" | "always"`). Unknown
    /// values return `Err(reason)` for the caller to wrap in
    /// `OpsError` / `PyValueError` as appropriate.
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s {
            "off" => Ok(Self::Off),
            "auto" => Ok(Self::Auto),
            "always" => Ok(Self::Always),
            other => Err(format!(
                "invalid shard_obs value '{other}'; expected off|auto|always"
            )),
        }
    }

    /// Whether to emit a single-section obs as shards.
    ///
    /// This is only consulted for single-section input — an already-sharded
    /// obs is preserved as shards by the streaming path before the policy is
    /// ever checked. `Auto` uses a strict `>` to match `from_anndata`'s
    /// `obs_rows > step` boundary, so optimize's `auto` output converges on
    /// exactly what the write path would have emitted for the same `n_obs`.
    pub fn should_shard_single_section(self, n_obs: u64, shard_target_rows: u32) -> bool {
        match self {
            Self::Off => false,
            Self::Always => true,
            Self::Auto => n_obs > shard_target_rows as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_round_trips() {
        assert_eq!(ObsShardPolicy::parse("off").unwrap(), ObsShardPolicy::Off);
        assert_eq!(ObsShardPolicy::parse("auto").unwrap(), ObsShardPolicy::Auto);
        assert_eq!(
            ObsShardPolicy::parse("always").unwrap(),
            ObsShardPolicy::Always
        );
    }

    #[test]
    fn parse_rejects_junk() {
        assert!(ObsShardPolicy::parse("yes").is_err());
        assert!(ObsShardPolicy::parse("Always").is_err());
        assert!(ObsShardPolicy::parse("").is_err());
    }

    #[test]
    fn default_is_auto() {
        assert_eq!(ObsShardPolicy::default(), ObsShardPolicy::Auto);
    }

    #[test]
    fn off_never_shards() {
        assert!(!ObsShardPolicy::Off.should_shard_single_section(u64::MAX, 16384));
        assert!(!ObsShardPolicy::Off.should_shard_single_section(0, 0));
    }

    #[test]
    fn always_always_shards() {
        assert!(ObsShardPolicy::Always.should_shard_single_section(0, 16384));
        assert!(ObsShardPolicy::Always.should_shard_single_section(1, u32::MAX));
    }

    #[test]
    fn auto_uses_strict_gt_threshold() {
        let target = 16384u32;
        // Equal → not sharded (strict `>`, matches from_anndata's `obs_rows > step`).
        assert!(!ObsShardPolicy::Auto.should_shard_single_section(target as u64, target));
        // One above → sharded.
        assert!(ObsShardPolicy::Auto.should_shard_single_section(target as u64 + 1, target));
        // Well below → not sharded.
        assert!(!ObsShardPolicy::Auto.should_shard_single_section(100, target));
    }
}
