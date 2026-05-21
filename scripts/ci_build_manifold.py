from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFOLD_BUILD = ROOT / "smart" / "vendor" / "manifold" / "build"
LOG_ROOT = ROOT / "runs" / "smoke_5" / "logs" / "build-tools"


def main() -> int:
    print(f"python={sys.executable}")
    print(sys.version)
    _run(["cmake", "--version"], check=False)

    if os.environ.get("SMART_CI_CLEAN_MANIFOLD_BUILD", "1") != "0":
        shutil.rmtree(MANIFOLD_BUILD, ignore_errors=True)

    env = os.environ.copy()
    env.setdefault("SMART_MANIFOLD_RELAX_WERROR", "1")
    command = [
        sys.executable,
        "-m",
        "smart",
        "--config",
        "configs/smoke_5.yaml",
        "build-tools",
        "--only-manifold-binding",
    ]
    result = _run(command, env=env, check=False)
    if result.returncode != 0:
        _dump_logs()
    return result.returncode


def _run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, env=env, text=True, check=check)


def _dump_logs() -> None:
    logs = sorted(LOG_ROOT.glob("vendored-manifold-*.log"))
    if not logs:
        print(f"no vendored Manifold logs found under {LOG_ROOT}")
        return
    for log in logs:
        group = os.environ.get("GITHUB_ACTIONS") == "true"
        if group:
            print(f"::group::{log}")
        else:
            print(f"===== {log} =====")
        try:
            print(log.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            print(f"failed to read {log}: {exc}")
        if group:
            print("::endgroup::")


if __name__ == "__main__":
    raise SystemExit(main())
