from __future__ import annotations

from pathlib import Path


_setup_cpp = Path(__file__).with_name("setup_cpp.py")
exec(compile(_setup_cpp.read_text(encoding="utf-8"), str(_setup_cpp), "exec"), {"__file__": str(_setup_cpp)})
