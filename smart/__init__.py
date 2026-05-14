"""SMART official Python package."""

from __future__ import annotations

import sys

__all__ = [
    "__version__",
    "build_action_prior_from_traces",
    "check_data",
    "doctor",
    "evaluate",
    "load",
    "run_pipeline",
    "workspace",
]

__version__ = "0.1.0"
_API_EXPORTS = {
    "build_action_prior_from_traces",
    "check_data",
    "doctor",
    "evaluate",
    "load",
    "run_pipeline",
    "workspace",
}


def __getattr__(name: str):
    if name == "build_action_prior_from_traces":
        from .action_prior import build_action_prior_from_traces

        globals()[name] = build_action_prior_from_traces
        return build_action_prior_from_traces
    if name in _API_EXPORTS:
        from . import api

        value = getattr(api, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

try:
    from . import _rust as _smart_rust  # type: ignore
except ImportError:
    try:
        import _rust as _smart_rust  # type: ignore
    except ImportError:
        pass
    else:
        sys.modules.setdefault(__name__ + "._rust", _smart_rust)
