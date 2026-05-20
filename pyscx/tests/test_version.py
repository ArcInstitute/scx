"""F4 regression: `pyscx.__version__` exists and matches the installed
package version. Closes F4 from SCX-USER-REPORT-2026-05-19.md."""
import re

import pyscx


def test_version_is_set():
    assert hasattr(pyscx, "__version__")
    assert isinstance(pyscx.__version__, str)
    assert pyscx.__version__, "pyscx.__version__ must not be empty"


def test_version_looks_like_semver_or_dev():
    # Accept canonical semver-ish, optionally with a pre-release / build
    # suffix, OR the editable-install sentinel.
    v = pyscx.__version__
    assert v == "0.0.0+dev" or re.match(r"^\d+\.\d+\.\d+([.+-].+)?$", v), (
        f"unexpected __version__ shape: {v!r}"
    )
