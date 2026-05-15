"""SMART official Python package."""

from __future__ import annotations

import sys

__all__ = [
    "__version__",
    "build_action_prior_from_traces",
    "build_linear_action_prior_from_traces",
    "build_mlp_action_prior_from_traces",
    "build_policy_gradient_action_prior_from_traces",
    "build_rl_mlp_action_prior_from_traces",
    "check_data",
    "compare_quality",
    "doctor",
    "evaluate",
    "load_action_prior",
    "load",
    "run_pipeline",
    "select_quality_guarded_run",
    "workspace",
]

__version__ = "0.1.0"
_API_EXPORTS = {
    "build_action_prior_from_traces",
    "build_linear_action_prior_from_traces",
    "build_mlp_action_prior_from_traces",
    "build_policy_gradient_action_prior_from_traces",
    "build_rl_mlp_action_prior_from_traces",
    "check_data",
    "compare_quality",
    "doctor",
    "evaluate",
    "load_action_prior",
    "load",
    "run_pipeline",
    "select_quality_guarded_run",
    "workspace",
}


def __getattr__(name: str):
    if name == "build_action_prior_from_traces":
        from .action_prior import build_action_prior_from_traces

        globals()[name] = build_action_prior_from_traces
        return build_action_prior_from_traces
    if name == "build_linear_action_prior_from_traces":
        from .action_prior import build_linear_action_prior_from_traces

        globals()[name] = build_linear_action_prior_from_traces
        return build_linear_action_prior_from_traces
    if name == "build_mlp_action_prior_from_traces":
        from .action_prior import build_mlp_action_prior_from_traces

        globals()[name] = build_mlp_action_prior_from_traces
        return build_mlp_action_prior_from_traces
    if name == "build_policy_gradient_action_prior_from_traces":
        from .action_prior import build_policy_gradient_action_prior_from_traces

        globals()[name] = build_policy_gradient_action_prior_from_traces
        return build_policy_gradient_action_prior_from_traces
    if name == "build_rl_mlp_action_prior_from_traces":
        from .action_prior import build_rl_mlp_action_prior_from_traces

        globals()[name] = build_rl_mlp_action_prior_from_traces
        return build_rl_mlp_action_prior_from_traces
    if name == "load_action_prior":
        from .action_prior import load_action_prior

        globals()[name] = load_action_prior
        return load_action_prior
    if name == "compare_quality":
        from .quality import compare_quality

        globals()[name] = compare_quality
        return compare_quality
    if name == "select_quality_guarded_run":
        from .quality import select_quality_guarded_run

        globals()[name] = select_quality_guarded_run
        return select_quality_guarded_run
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
