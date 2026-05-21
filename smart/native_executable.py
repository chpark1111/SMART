from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def native_executable_name() -> str:
    return "smart-cpp-native.exe" if os.name == "nt" else "smart-cpp-native"


def native_executable_path(
    configured: str | Path | None = None,
    *,
    include_path: bool = True,
) -> Path | None:
    candidates: list[Path] = []
    env_value = os.environ.get("SMART_CPP_NATIVE_BIN")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    if configured:
        candidates.append(Path(configured).expanduser())

    package_binary = Path(__file__).resolve().parent / "bin" / native_executable_name()
    source_binary = Path(__file__).resolve().parents[1] / "build" / native_executable_name()
    candidates.extend([package_binary, source_binary])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if include_path:
        found = shutil.which(native_executable_name())
        return Path(found) if found else None
    return None


def run_native_command(
    args: Iterable[str],
    *,
    configured: str | Path | None = None,
    include_path: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    binary = native_executable_path(configured, include_path=include_path)
    if binary is None:
        raise FileNotFoundError(
            "smart-cpp-native executable is not available; run `smart build-cpp` "
            "or install a SMART wheel that includes smart/bin/smart-cpp-native"
        )
    return subprocess.run(
        [str(binary), *[str(arg) for arg in args]],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def main(argv: list[str] | None = None) -> int:
    """Console-script launcher for the packaged native executable."""

    completed = run_native_command(
        sys.argv[1:] if argv is None else argv,
        include_path=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return int(completed.returncode)
