"""Rust-literal emitters shared by the reference generators in this directory.

Every generator under `benchmarks/scripts/generate_*_references.py` prints
paste-ready Rust `const`s for a pinned-reference module in `scx-accel`. They all
need the same two things — a float literal clippy will accept and a 2-D array
literal — and Organization Phase 7d extracted them here rather than letting a
second generator copy them. A duplicated emitter inside the phase whose subject
is duplication would be the mistake in miniature.

The generators own their fixtures and their expected values; this module owns
only formatting.
"""

from __future__ import annotations

import numpy as np


def f64_literal(v: float) -> str:
    """Format as a Rust `f64` literal.

    `repr` gives the **shortest** string that round-trips through f64, which is
    what clippy's `excessive_precision` lint demands: `.17g` would emit
    `0.0090234388180803256` where f64 only carries
    `0.009023438818080326`, and clippy rejects the extra digits as a claim of
    precision the type cannot hold. Shortest-round-trip is also the honest
    form — every digit printed is a digit the value has.
    """
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        # `repr` gives 'nan' / 'inf', and the decimal-point fallback below would
        # then emit `nan.0`, which is not Rust. A non-finite reference value is
        # a broken fixture, not something to format around.
        raise ValueError(
            f"refusing to emit a non-finite reference value ({v!r}); fix the "
            f"fixture rather than the formatter"
        )
    s = repr(v)
    return s if ("." in s or "e" in s or "E" in s) else s + ".0"


def f32_literal(v: float) -> str:
    """Format as a Rust `f32` literal, shortest form that round-trips through f32.

    Not the same problem as [`f64_literal`]: `repr` gives the shortest string
    round-tripping through **f64**, and pasting that into an `f32` array trips
    clippy's `excessive_precision` — it reads the extra digits as a claim of
    precision `f32` cannot hold. `4818.53125` is exactly representable in f32 and
    still gets flagged, because `4818.5312` reaches the same f32.
    """
    v32 = np.float32(v)
    if not np.isfinite(v32):
        raise ValueError(
            f"refusing to emit a non-finite reference value ({v!r}); fix the "
            f"fixture rather than the formatter"
        )
    # Fixed-point first: `%g` drops to exponent form at some precisions, so
    # `4240.0` came out as `4.24e+03` sitting between plain literals in the same
    # array. Only fall through to `%g` for magnitudes where fixed point would be
    # absurd.
    def shortest(fmt: str, start: int) -> str | None:
        for digits in range(start, 12):
            cand = f"{float(v32):.{digits}{fmt}}"
            if np.float32(float(cand)) == v32:
                return cand if ("." in cand or "e" in cand) else cand + ".0"
        return None

    fixed, sci = shortest("f", 0), shortest("g", 1)
    if fixed is None:
        return sci if sci is not None else repr(float(v32))
    if sci is not None and len(sci) < len(fixed):
        return sci
    return fixed


def rust_matrix(name: str, m: np.ndarray, ty: str = "f64") -> str:
    """`const NAME: [[ty; cols]; rows] = [...]`, one source row per line."""
    lit = f32_literal if ty == "f32" else f64_literal
    rows = ",\n".join("    [" + ", ".join(lit(v) for v in row) + "]" for row in m)
    return f"const {name}: [[{ty}; {m.shape[1]}]; {m.shape[0]}] = [\n{rows},\n];"


def rust_int_matrix(name: str, m: np.ndarray, ty: str = "u16", per_line: int = 24) -> str:
    """`const NAME: [[ty; cols]; rows] = [...]` for an integer count matrix.

    Counts are emitted as integers rather than floats because they *are*
    integers: a `3.0` in a count fixture invites the reader to wonder which
    values were rounded. Rows longer than `per_line` wrap, so a wide fixture
    stays diffable instead of becoming one 6000-column line.
    """
    if not np.issubdtype(m.dtype, np.integer):
        raise ValueError(f"{name}: expected an integer matrix, got {m.dtype}")
    if m.min() < 0:
        raise ValueError(f"{name}: counts must be non-negative, saw {m.min()}")
    out = []
    for row in m:
        cells = [str(int(v)) for v in row]
        if len(cells) <= per_line:
            out.append("    [" + ", ".join(cells) + "]")
            continue
        chunks = [
            ", ".join(cells[i : i + per_line]) for i in range(0, len(cells), per_line)
        ]
        out.append("    [\n        " + ",\n        ".join(chunks) + ",\n    ]")
    return f"const {name}: [[{ty}; {m.shape[1]}]; {m.shape[0]}] = [\n" + ",\n".join(out) + ",\n];"


def rust_f64_array(name: str, v: np.ndarray, per_line: int = 4) -> str:
    """`const NAME: [f64; n] = [...]`, wrapped at `per_line` entries."""
    lits = [f64_literal(x) for x in np.asarray(v, dtype=float).ravel()]
    chunks = [", ".join(lits[i : i + per_line]) for i in range(0, len(lits), per_line)]
    body = ",\n    ".join(chunks)
    return f"const {name}: [f64; {len(lits)}] = [\n    {body},\n];"
