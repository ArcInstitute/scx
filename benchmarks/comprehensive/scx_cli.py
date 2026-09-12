"""Locating the `scx` CLI binary — one implementation, one place.

Four benchmark arms shell out to `scx`, and each needs a *different* binary to
count as usable: `shuffle_layout` needs `info --json` for per-section sizes,
`accel_to_gpu_anndata` needs an `optimize` that accepts `--codec`,
`cloud_metadata`'s CLI arm needs a build compiled with `--features cloud`, and
`conversion_streaming`'s 10x arms need a `convert` that streams `--from 10x`
(OPT-CONVERT-9). That last one is the case this module *cannot* answer: every
`scx` since the flag shipped passes a `--help` probe for `--stream`, including
the builds that reject it on 10x. Its arms therefore inline a `--help` probe of
their own as a coarse filter and treat the convert itself as the capability
test — see `conversion_streaming._run_tenx_arms` / `TenxArmUnavailable`. The
search order is identical in every case — `$SCX_CLI_BIN`, then the repo's
`target/release/scx`, then a PATH `scx` — and it was written out twice, with
`shuffle_layout`'s own docstring noting that the other copy "uses the same
order". A third copy was the thing to avoid; this module is the single source
of truth, in the shape `rss.py` already established for `PeakRssSampler`.

**Why a capability probe rather than a bare existence check.** `scx info` is
not clap-gated, so `info --help` exits 0 on a build with no cloud support at
all — the URL then fails at run time with "requires scx-cli built with
--features cloud". And the installed binary on a dev box is routinely older
than the workspace (0.12.0 against 0.16.0 here), so "a binary exists" says
nothing about whether the subcommand or flag an arm needs is in it. Each caller
therefore names the probe that answers its own question.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from benchmarks.comprehensive.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

#: Seconds allowed for a probe. A `--help` is instant; the cap is only there so
#: a hung binary cannot stall a cohort job.
PROBE_TIMEOUT_S = 30

#: Probe for a build with cloud support compiled in.
#:
#: `pull` is one of `scx-cli`'s `CLOUD_SUBCOMMANDS`, rejected by clap itself on
#: a non-cloud build ("`pull` is a cloud subcommand and is not compiled into
#: this build"), so its exit status is a reliable feature test. `info --help`
#: is not: `Info` is always compiled and only its *cloud branch* is behind the
#: feature, so probing `info` resolves a binary that then fails on the URL.
CLOUD_PROBE: tuple[str, ...] = ("pull", "--help")

#: Probe for an `info` that speaks JSON.
INFO_JSON_PROBE: tuple[str, ...] = ("info", "--json", "--help")

#: Probe for an `optimize` that takes `--codec`.
OPTIMIZE_PROBE: tuple[str, ...] = ("optimize", "--help")


def candidates() -> list[str]:
    """Binaries to try, in order: `$SCX_CLI_BIN`, repo release build, PATH.

    `$SCX_CLI_BIN` first so an operator can pin a specific build for a capture
    without touching PATH; the repo's own `target/release/scx` next because on
    a dev box it is the one that matches the workspace.
    """
    out = [os.environ.get("SCX_CLI_BIN"), str(PROJECT_ROOT / "target" / "release" / "scx")]
    which = shutil.which("scx")
    if which:
        out.append(which)
    return [c for c in out if c]


def resolve_scx_bin(
    probe: tuple[str, ...],
    requires: bytes | None = None,
) -> str | None:
    """First candidate binary that passes *probe*, or ``None``.

    *probe* is required rather than defaulted: all three callers need a
    different one, so a default would favour whichever caller it named while
    coupling this module to that subcommand.

    *probe* is appended to the binary path and run with output captured;
    exit 0 counts as a pass. *requires*, when given, must additionally appear
    in the probe's stdout — how a flag is tested on a subcommand that exists
    either way (`optimize --help` succeeds on old builds that lack `--codec`).

    Never raises: a missing binary, a non-executable file, a timeout and a
    crash all read as "this candidate does not qualify", because every caller
    is an optional arm and a raise out of one fails a whole cohort job.
    """
    for cand in candidates():
        if not Path(cand).exists():
            continue
        try:
            proc = subprocess.run(
                [cand, *probe], capture_output=True, timeout=PROBE_TIMEOUT_S,
            )
        except Exception:  # noqa: BLE001
            continue
        if proc.returncode != 0:
            continue
        if requires is not None and requires not in proc.stdout:
            continue
        return cand
    return None
