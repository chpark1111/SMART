from __future__ import annotations

from types import ModuleType
from typing import Any


def _load_backend() -> ModuleType | None:
    try:
        from . import cpp as cpp_backend

        if cpp_backend.native_core_available():
            return cpp_backend
    except Exception:
        pass
    return None


_backend = _load_backend()


def using_cpp() -> bool:
    return bool(_backend is not None and getattr(_backend, "__name__", "").endswith(".cpp"))


def native_core_available() -> bool:
    return _backend is not None and bool(getattr(_backend, "native_core_available")())


def backend_path() -> str | None:
    if _backend is None:
        return None
    backend_path_fn = getattr(_backend, "backend_path", None)
    if backend_path_fn is None:
        return None
    return backend_path_fn()


def __getattr__(name: str) -> Any:
    if _backend is None:
        raise AttributeError(
            "SMART native backend is not built. Run `smart build-cpp` for the C++ backend."
        )
    return getattr(_backend, name)
