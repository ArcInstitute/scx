"""Trampolines executed for their *stack frame*, not their logic.

CPython resolves ``__import__`` for C-level imports (``PyImport_Import``)
from the **innermost Python frame's** builtins. numpy's own C code lazily
imports ``numpy._core._dtype`` on every ``str(dtype)`` / ``repr(dtype)`` /
``dtype.name``, so a native pyscx entry point that touches a dtype name while
called directly from restricted-exec globals (a pipeline runner's sandbox
whose ``__builtins__`` omits ``__import__``) dies with
``KeyError: '__import__'`` — inside numpy, where no Rust-side import sweep
can reach.

Calling through a function defined *here* pushes a Python frame whose
builtins come from this module's globals (the real builtins), shielding
every C-level import made underneath it. The Rust side imports this module
via its frame-insensitive ``pyimport::import_module`` and calls these
helpers instead of touching the attribute directly.

Regression coverage: ``pyscx/tests/test_sandbox_exec.py``.
"""


def dtype_name(obj):
    """``obj.dtype.name`` under a real-builtins frame.

    ``obj`` is anything with a ``.dtype`` (numpy array, pandas Series,
    scipy sparse matrix). numpy 2.x implements ``dtype.name`` by importing
    ``numpy._core._dtype`` on every access.
    """
    return obj.dtype.name
