#!/usr/bin/env bash
# Unit tests for prod_filter.awk — the production/test splitter the ORG-8.20-4
# guard scans through.
#
# It has its own tests because the thing it replaced was wrong in a way that
# made every scan downstream *vacuously pass*: `sed '/^#[cfg(test)]/,$d'` deletes
# to EOF, so one early test-only item hid the rest of a file. A filter whose
# failure mode is silent green needs to be falsifiable on its own.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWK="$HERE/prod_filter.awk"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail=0
check() { # name, expected, actual
  if [ "$2" = "$3" ]; then
    echo "  ok   $1"
  else
    echo "  FAIL $1: expected [$2], got [$3]"
    fail=1
  fi
}

run() { awk -f "$AWK" "$1"; }

# 1. A test-only item in the MIDDLE must not swallow the production tail. This
#    is the exact shape of scx-accel/src/harmony/gpu.rs.
cat > "$TMP/mid.rs" <<'EOF'
fn before() {}

#[cfg(test)]
pub(super) static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn after_the_counter() { capture_graph(); }
EOF
check "middle item does not truncate the file" \
  "before after_the_counter" \
  "$(run "$TMP/mid.rs" | grep -oE 'fn [a-z_]+' | sed 's/fn //' | tr '\n' ' ' | sed 's/ $//')"

# 2. A trailing `#[cfg(test)] mod tests { … }` is still fully stripped — the case
#    the old filter got right, which the new one must not regress.
cat > "$TMP/tail.rs" <<'EOF'
fn production() {}

#[cfg(test)]
mod tests {
    fn test_only() {}
}
EOF
check "trailing test module is stripped" "production" \
  "$(run "$TMP/tail.rs" | grep -oE 'fn [a-z_]+' | sed 's/fn //' | tr '\n' ' ' | sed 's/ $//')"

# 3. `#[cfg(test)] #[path = "…"] mod tests;` — attribute stack then a semicolon
#    item, the convention this repo uses for extracted test files.
cat > "$TMP/path.rs" <<'EOF'
fn production() {}

#[cfg(test)]
#[path = "thing_tests.rs"]
mod tests;

fn also_production() {}
EOF
check "attribute stack + semicolon item" "production also_production" \
  "$(run "$TMP/path.rs" | grep -oE 'fn [a-z_]+' | sed 's/fn //' | tr '\n' ' ' | sed 's/ $//')"

# 4. Braces inside a string literal must not unbalance the item scan.
cat > "$TMP/brace.rs" <<'EOF'
fn production() {}

#[cfg(test)]
fn helper() {
    let s = "a { brace in a string";
    println!("{}", s);
}

fn after() {}
EOF
check "braces in strings do not unbalance" "production after" \
  "$(run "$TMP/brace.rs" | grep -oE 'fn [a-z_]+' | sed 's/fn //' | tr '\n' ' ' | sed 's/ $//')"

# 5. Two test items in one file, production between and after.
cat > "$TMP/two.rs" <<'EOF'
fn a() {}
#[cfg(test)]
static X: u8 = 1;
fn b() {}
#[cfg(test)]
mod tests { fn t() {} }
fn c() {}
EOF
check "multiple test items" "a b c" \
  "$(run "$TMP/two.rs" | grep -oE 'fn [a-z_]+' | sed 's/fn //' | tr '\n' ' ' | sed 's/ $//')"

# 6. An INDENTED `#[cfg(test)]` (inside an impl or a function) is deliberately
#    left alone: the guard's claims are about column-zero items, and consuming a
#    nested one would eat the enclosing production block.
cat > "$TMP/nested.rs" <<'EOF'
fn production() {
    #[cfg(test)]
    let debug_only = 1;
    real_work();
}
EOF
check "indented cfg(test) is left in place" "1" \
  "$(run "$TMP/nested.rs" | grep -c 'real_work')"

[ "$fail" -eq 0 ] || { echo "prod_filter.awk unit tests FAILED"; exit 1; }
echo "prod_filter.awk: all unit tests passed"
