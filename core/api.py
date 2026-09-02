"""Deprecated shim for the v1 API.

The engine that lived here moved to :mod:`divinelines.api.app`, which serves
the same ``POST /api/predict`` contract — same request body, same response
keys — plus the rest of the v2 surface (health, games, odds movement,
performance, models, backtests).

Re-exporting rather than keeping a second copy matters: two FastAPI apps with
their own model-loading code is exactly how a "fixed" bug survives in the
half nobody remembers to update.

Running ``python core/api.py`` still starts the server on port 8000.
"""

from __future__ import annotations

import warnings

from divinelines.api.app import app, run

warnings.warn(
    "core.api is deprecated; use `divinelines serve` or divinelines.api.app",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["app", "run"]


if __name__ == "__main__":  # pragma: no cover
    print("Starting DivineLines API on port 8000 (core.api is a shim for divinelines.api.app)")
    run()
