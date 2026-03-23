use extendr_api::prelude::*;

mod interop;

// ── Phase B: Core Reader ────────────────────────────────────────

/// R class wrapping an SCX file handle (mmap-backed, lazy).
pub struct ScxExperiment {
    // Will hold ScxReader + PathBuf in Phase B
}

#[extendr]
impl ScxExperiment {}

// ── Phase C: Query Pipeline ─────────────────────────────────────

/// R query pipeline — pipe-friendly with |>.
pub struct RQueryPipeline {
    // Will hold Option<QueryPipeline> in Phase C
}

#[extendr]
impl RQueryPipeline {}

/// Result from executing a query pipeline.
pub struct RQueryResult {
    // Will hold Option<QueryResult> + cached stats in Phase C
}

#[extendr]
impl RQueryResult {}

// ── Phase E: File Operations ────────────────────────────────────

/// Append cells from one SCX file to another.
#[extendr]
fn scx_append(_target: &str, _input: &str) {
    todo!("scx_append implementation in Phase E")
}

/// Mark specific cell indices as logically deleted.
#[extendr]
fn scx_delete(_path: &str, _cell_indices: Vec<i32>) -> i32 {
    todo!("scx_delete implementation in Phase E")
}

/// Rewrite an SCX file reclaiming deleted/orphaned space.
#[extendr]
fn scx_compact(_input: &str, _output: &str) {
    todo!("scx_compact implementation in Phase E")
}

/// Roll back to a previous manifest version.
#[extendr]
fn scx_rollback(_path: &str) {
    todo!("scx_rollback implementation in Phase E")
}

/// Merge multiple SCX files into one.
#[extendr]
fn scx_merge(_inputs: Vec<String>, _output: &str) {
    todo!("scx_merge implementation in Phase E")
}

/// Return file information as a named list.
#[extendr]
fn scx_info(_path: &str) -> Robj {
    todo!("scx_info implementation in Phase E")
}

/// Validate an SCX file. Returns TRUE if valid, otherwise raises an error.
#[extendr]
fn scx_validate(_path: &str) -> bool {
    todo!("scx_validate implementation in Phase E")
}

// ── Module Registration ─────────────────────────────────────────

extendr_module! {
    mod rscx;
    impl ScxExperiment;
    impl RQueryPipeline;
    impl RQueryResult;
    fn scx_append;
    fn scx_delete;
    fn scx_compact;
    fn scx_rollback;
    fn scx_merge;
    fn scx_info;
    fn scx_validate;
}
