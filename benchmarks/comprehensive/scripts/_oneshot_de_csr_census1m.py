"""One-shot wrapper: run bench_csc_dispatch DE-CSR on census_1m with
n_runs=1. Used by sbatch_de_csr_census1m.sh."""

import sys
import logging

sys.path.insert(0, "/home/nickyoungblut/dev/rust/scx")

from benchmarks.comprehensive.config import DATASETS  # noqa: E402
from benchmarks.comprehensive.benchmarks.bench_csc_dispatch import (  # noqa: E402
    bench_csc_dispatch_variants,
    run,
)
from benchmarks.comprehensive.results import write_result  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("de_csr_census1m")

ds = DATASETS["census_1m"]
variants = {v.key: v for v in bench_csc_dispatch_variants()}
v = variants["bench_csc__de_csr"]

log.info("Running %s on %s with n_runs=1", v.key, ds.name)
r = run(ds, v, n_runs=1)
if r is None:
    log.error("run returned None")
    sys.exit(1)

log.info("wall_s=%s rss_mb=%s", r.runs[0].wall_s, r.runs[0].peak_rss_mb)
write_result(r)
log.info("Done.")
