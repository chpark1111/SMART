from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    elapsed_sec: float
    log_path: Path | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    dry_run: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def find_executable(explicit: str | None, env_name: str, fallback: str) -> str | None:
    if explicit:
        return explicit
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    return shutil.which(fallback)


def run_command(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    timeout: int | float | None = None,
    env: dict[str, str] | None = None,
    log_path: str | Path | None = None,
    dry_run: bool = False,
) -> CommandResult:
    started = time.time()
    resolved_log = Path(log_path) if log_path is not None else None
    if resolved_log is not None:
        resolved_log.parent.mkdir(parents=True, exist_ok=True)
        resolved_log.write_text("$ " + " ".join(command) + "\n", encoding="utf-8")

    if dry_run:
        return CommandResult(
            command=command,
            returncode=0,
            elapsed_sec=0.0,
            log_path=resolved_log,
            dry_run=True,
        )

    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            timeout=timeout,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        elapsed = time.time() - started
        if resolved_log is not None:
            with resolved_log.open("a", encoding="utf-8", errors="replace") as file:
                if completed.stdout:
                    file.write("\n[stdout]\n")
                    file.write(completed.stdout)
                if completed.stderr:
                    file.write("\n[stderr]\n")
                    file.write(completed.stderr)
        return CommandResult(
            command=command,
            returncode=completed.returncode,
            elapsed_sec=elapsed,
            log_path=resolved_log,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - started
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        if resolved_log is not None:
            with resolved_log.open("a", encoding="utf-8", errors="replace") as file:
                file.write(f"\n[timed out after {timeout} seconds]\n")
                if stdout:
                    file.write("\n[stdout]\n")
                    file.write(stdout)
                if stderr:
                    file.write("\n[stderr]\n")
                    file.write(stderr)
        return CommandResult(
            command=command,
            returncode=124,
            elapsed_sec=elapsed,
            log_path=resolved_log,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
        )
    except FileNotFoundError as exc:
        elapsed = time.time() - started
        if resolved_log is not None:
            with resolved_log.open("a", encoding="utf-8", errors="replace") as file:
                file.write(f"\n[file not found]\n{exc}\n")
        return CommandResult(
            command=command,
            returncode=127,
            elapsed_sec=elapsed,
            log_path=resolved_log,
            stderr=str(exc),
        )
