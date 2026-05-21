from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _asan_env() -> dict[str, str]:
    env = os.environ.copy()
    defaults = {
        "abort_on_error": "1",
        "halt_on_error": "1",
        "detect_stack_use_after_return": "1",
        "strict_string_checks": "1",
    }
    existing = env.get("ASAN_OPTIONS", "")
    options = dict(
        item.split("=", 1)
        for item in existing.split(":")
        if "=" in item
    )
    options = {**defaults, **options}
    env["ASAN_OPTIONS"] = ":".join(f"{key}={value}" for key, value in options.items())
    return env


def _run(command: list[str], *, cwd: Path, timeout_sec: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=_asan_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_sec,
        check=False,
    )


def _write_smoke_obj(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "o smoke",
                "v 0 0 0",
                "v 2 0 0",
                "v 0 2 0",
                "v 0 0 2",
                "f 1 2 3",
                "f 1 2 4",
                "f 1 3 4",
                "f 2 3 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def smoke_native_sanitizers(*, binary: Path, repo_root: Path, timeout_sec: float) -> list[dict[str, object]]:
    if not binary.exists():
        raise SystemExit(
            f"ASan native binary is missing: {binary}\n"
            "Build it with: smart --config configs/smoke_5.yaml build-cpp --asan"
        )
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="smart-native-asan-") as tmp:
        tmp_dir = Path(tmp)
        source = tmp_dir / "model.obj"
        normalized = tmp_dir / "normalized.obj"
        _write_smoke_obj(source)
        commands = [
            ["--help"],
            ["obj-info", "--input", str(source)],
            [
                "normalize",
                "--input",
                str(source),
                "--output",
                str(normalized),
                "--target",
                "1.0",
            ],
        ]
        for args in commands:
            command = [str(binary), *args]
            completed = _run(command, cwd=repo_root, timeout_sec=timeout_sec)
            result = {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
            if args and args[0] == "normalize" and completed.returncode == 0:
                try:
                    result["payload"] = json.loads(completed.stdout)
                except json.JSONDecodeError:
                    result["payload"] = None
            results.append(result)
            if completed.returncode != 0:
                break
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a short AddressSanitizer smoke for smart-cpp-native.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--binary", type=Path, default=Path("build/smart-cpp-native-asan"))
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    binary = args.binary if args.binary.is_absolute() else repo_root / args.binary
    results = smoke_native_sanitizers(binary=binary, repo_root=repo_root, timeout_sec=args.timeout_sec)
    ok = all(int(result["returncode"]) == 0 for result in results)
    if args.json:
        print(json.dumps({"ok": ok, "results": results}, indent=2))
    else:
        for result in results:
            command = " ".join(str(part) for part in result["command"])
            print(f"{command}: {result['returncode']}")
            if int(result["returncode"]) != 0:
                if result["stdout"]:
                    print(result["stdout"], file=sys.stdout)
                if result["stderr"]:
                    print(result["stderr"], file=sys.stderr)
        if ok:
            print("smart-cpp-native ASan smoke ok")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
