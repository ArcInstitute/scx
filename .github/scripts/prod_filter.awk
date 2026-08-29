# Emit the production half of a Rust file: everything except items attributed
# `#[cfg(test)]` at column zero.
#
# The naive `sed '/^#\[cfg(test)\]/,$d'` deletes from the FIRST such attribute to
# EOF, which is right only when the test module is last. It is not, in 24 files
# in this workspace — `scx-accel/src/harmony/gpu.rs` puts a test-only counter at
# line 25 and loses 1118 of its 1142 lines, including the workspace's only
# production CUDA-graph capture site.
#
# This skips exactly the attributed item: any further attribute lines, then the
# item itself, which ends either at the `}` closing its first `{` block or at the
# first `;` when it has no block (`mod tests;`, `static X: T = …;`, `use …;`).
# Braces inside string, char and line-comment content are ignored.
BEGIN { skipping = 0; depth = 0; opened = 0 }

skipping == 0 && /^#\[cfg\(test\)\]/ { skipping = 1; depth = 0; opened = 0; next }

skipping == 1 {
    line = $0
    # Strip line comments and string/char literals before counting braces.
    sub(/\/\/.*$/, "", line)
    gsub(/"([^"\\]|\\.)*"/, "", line)
    gsub(/'"'"'([^'"'"'\\]|\\.)*'"'"'/, "", line)

    # Still inside the attribute stack (e.g. `#[path = "…"]`)? Keep skipping.
    if (opened == 0 && depth == 0 && line ~ /^#\[/) next

    n_open = gsub(/\{/, "{", line)
    n_close = gsub(/\}/, "}", line)
    if (n_open > 0) opened = 1
    depth += n_open - n_close

    if (opened == 1 && depth <= 0) { skipping = 0; next }
    if (opened == 0 && line ~ /;/) { skipping = 0; next }
    next
}

{ print }
