// The framing rule for a CSC sidecar build. The build itself —
// `rebuild_csc_inplace`, used by `scx build-csc` with no `<OUTPUT>` and by
// `scx append --rebuild-csc` — lives in `build_csc.rs`.
//
// The rewrite ops (`compact`, `merge`, `optimize`, `sort`, `subset`) and
// streaming convert no longer use it: they build the output's sidecar in the
// same pass as its X shards (`ScxWriter::enable_csc_sidecar`, see
// `csc_carry.rs`). `append` still drops the sidecar, since it appends rows in
// place, and rebuilds it in place when asked.

use std::path::Path;

/// Kept at its old path (`scx_ops::rebuild_csc::rebuild_csc_inplace`) as well
/// as the crate root, now that the implementation lives in `build_csc.rs`.
pub use crate::build_csc::rebuild_csc_inplace;

use scx_format_io::FramingConfig;

/// Framing for a CSC-sidecar build on `path`: the file's existing layout, with
/// nothing re-selected.
///
/// `Some(default)` on a framed (v4) file, `None` on an unframed (≤v3) one —
/// exactly the values the in-place build admits, since an append frames the
/// sidecar to match the file and cannot change the file. `decode_target` is
/// `None` because it would authorise the encoder to re-pick the sidecar's codec,
/// which `pick_csc_encoding` has already decided.
///
/// One implementation for every surface — CLI, pyscx and convert — because this
/// rule was got wrong twice in two directions while `build-csc` still rewrote
/// the CSR: `subset` and `convert --csc` passed the rewrite framing (codec
/// override), and pyscx's `sort`/`shuffle`/`from_anndata` passed `None`, which
/// then stripped framing off the whole file. The build now refuses the first
/// kind of mismatch outright and cannot do the second. Returns `None` if `path`
/// cannot be opened; the build itself surfaces any real error.
pub fn framing_for_csc_rebuild(path: &Path) -> Option<FramingConfig> {
    scx_format_io::ScxReader::open(path)
        .ok()
        .filter(|r| r.header().format_version >= scx_format_io::CURRENT_FORMAT_VERSION)
        .map(|_| FramingConfig::default())
}
