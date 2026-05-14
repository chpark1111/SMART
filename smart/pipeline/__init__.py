"""Pipeline orchestration for SMART."""

from __future__ import annotations

from .config import load_config
from .stages import run_pipeline

__all__ = ["load_config", "run_pipeline"]
