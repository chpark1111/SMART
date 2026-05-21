from __future__ import annotations

import argparse
import json
import subprocess
import sys
import sysconfig
from pathlib import Path


CONSOLE_COMMANDS = [
    "smart",
    "smart-cpp-native",
    "smart-audit-wheel",
    "smart-smoke-console-scripts",
    "smart-release-preflight",
    "smart-quickstart",
]

FUNCTIONAL_SMOKE_COMMANDS = [
    {
        "label": "smart configs --json",
        "argv": ["smart", "configs", "--json"],
        "required_name": "smoke_5.yaml",
    },
]


def smoke_console_scripts(
    *,
    bin_dir: str | Path | None = None,
    timeout_sec: float = 20.0,
) -> list[dict[str, object]]:
    scripts_dir = Path(bin_dir) if bin_dir else Path(sysconfig.get_path("scripts"))
    results: list[dict[str, object]] = []
    for command in CONSOLE_COMMANDS:
        executable = scripts_dir / command
        results.append(
            _run_smoke_command(
                command,
                [str(executable), "--help"],
                timeout_sec=float(timeout_sec),
            )
        )
    for check in FUNCTIONAL_SMOKE_COMMANDS:
        argv = list(check["argv"])
        argv[0] = str(scripts_dir / str(argv[0]))
        smoke_result = _run_smoke_command(
            str(check["label"]),
            [str(item) for item in argv],
            timeout_sec=float(timeout_sec),
        )
        if int(smoke_result["returncode"]) == 0:
            required_name = str(check["required_name"])
            try:
                payload = json.loads(str(smoke_result["stdout"]))
                names = {str(item.get("name", "")) for item in payload if isinstance(item, dict)}
                if required_name not in names:
                    smoke_result["returncode"] = 1
                    smoke_result["stderr_tail"] = f"missing required packaged item: {required_name}"
            except Exception as exc:
                smoke_result["returncode"] = 1
                smoke_result["stderr_tail"] = f"failed to parse JSON smoke output: {exc}"
        results.append(smoke_result)
    for result in results:
        result.pop("stdout", None)
    return results


def _run_smoke_command(
    label: str,
    argv: list[str],
    *,
    timeout_sec: float,
) -> dict[str, object]:
    try:
        result = subprocess.run(
            argv,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )
        return {
            "command": label,
            "path": argv[0],
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stdout_tail": result.stdout[-500:],
            "stderr_tail": result.stderr[-500:],
        }
    except OSError as exc:
        return {
            "command": label,
            "path": argv[0],
            "returncode": 1,
            "stdout": "",
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run --help smoke checks for packaged SMART console scripts.")
    parser.add_argument("--bin-dir", default="", help="Directory containing installed console scripts. Defaults to sysconfig scripts path.")
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--json", action="store_true", help="Print full smoke results as JSON")
    args = parser.parse_args(argv)

    try:
        results = smoke_console_scripts(bin_dir=args.bin_dir or None, timeout_sec=args.timeout_sec)
    except subprocess.TimeoutExpired as exc:
        print(f"console script timed out: {exc}", file=sys.stderr)
        return 1

    failures = [result for result in results if int(result["returncode"]) != 0]
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for result in results:
            print(f"{result['command']}: {result['returncode']}")
        for result in failures:
            print(f"failed {result['command']} stderr={result['stderr_tail']}", file=sys.stderr)

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
