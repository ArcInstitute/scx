"""Shared helper for reading `__init__.pyi` class bodies out of the stub.

Extracted so `test_backed.py` (handle specials) and
`test_experiment_stub_coverage.py` (signature drift) share one implementation
instead of two that can disagree about where a class body ends.
"""

from __future__ import annotations


def stub_class_body(cls_name: str) -> str:
    """The lines of one `class <cls_name>:` block in `__init__.pyi`."""
    import pathlib

    import pyscx

    pyi = pathlib.Path(pyscx.__file__).with_name("__init__.pyi")
    text = pyi.read_text()
    start = text.index(f"class {cls_name}:")
    rest = text[start + 1 :]
    # Ends at the next top-level `class `/`def ` declaration.
    ends = [i for i in (rest.find("\nclass "), rest.find("\ndef ")) if i != -1]
    return rest[: min(ends)] if ends else rest
