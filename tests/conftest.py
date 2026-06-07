"""Shared pytest process configuration."""

from __future__ import annotations

import os


if os.name == "nt":
    # Windows CPU torch can fail to import after other numeric libraries have
    # loaded OpenMP DLLs in the same pytest process. Load it first and keep this
    # scoped to tests; package/runtime code must not depend on it.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch as _torch  # noqa: F401
