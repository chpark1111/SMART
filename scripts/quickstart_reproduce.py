from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(argv: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("$ " + " ".join(argv), flush=True)
    if dry_run:
        return
    subprocess.run(argv, cwd=cwd, check=True)


def _smart_cmd(config: Path, *args: str) -> list[str]:
    return [sys.executable, "-m", "smart", "--config", str(config), *args]


def run_quickstart(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    config = args.config
    if not config.is_absolute():
        config = repo_root / config

    if args.install_editable:
        extra = ".[pipeline]" if args.with_pipeline_extra else "."
        _run([sys.executable, "-m", "pip", "install", "-e", extra], cwd=repo_root, dry_run=args.dry_run)

    _run(_smart_cmd(config, "doctor"), cwd=repo_root, dry_run=args.dry_run)
    _run(_smart_cmd(config, "check-data"), cwd=repo_root, dry_run=args.dry_run)

    if args.build_tools:
        build_args = ["build-tools"]
        if args.only_manifold_binding:
            build_args.append("--only-manifold-binding")
        _run(_smart_cmd(config, *build_args), cwd=repo_root, dry_run=args.dry_run)

    if args.build_cpp:
        build_args = ["build-cpp"]
        if args.asan:
            build_args.append("--asan")
        _run(_smart_cmd(config, *build_args), cwd=repo_root, dry_run=args.dry_run)

    if args.run_smoke:
        run_args = ["native-run"]
        if args.category:
            run_args.extend(["--category", args.category])
        for mesh_id in args.mesh:
            run_args.extend(["--mesh", mesh_id])
        if args.force:
            run_args.append("--force")
        if args.stage_dry_run:
            run_args.append("--dry-run")
        _run(_smart_cmd(config, *run_args), cwd=repo_root, dry_run=args.dry_run)
        _run(_smart_cmd(config, "summary"), cwd=repo_root, dry_run=args.dry_run)

    if args.release_preflight:
        cmd = [
            sys.executable,
            "scripts/release_preflight.py",
            "--dist-dir",
            str(args.dist_dir),
            "--venv-dir",
            str(args.venv_dir),
            "--recreate-venv",
        ]
        if args.run_asan_smoke:
            cmd.append("--run-asan-smoke")
        _run(cmd, cwd=repo_root, dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the public SMART quickstart sequence: install/verify the package, "
            "build native tools, run a smoke reproduction, and optionally build "
            "release artifacts."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="SMART source checkout root.")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke_5.yaml"))
    parser.add_argument("--install-editable", action="store_true", help="Run pip install -e before checks.")
    parser.add_argument(
        "--with-pipeline-extra",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Install the pipeline extra when --install-editable is used.",
    )
    parser.add_argument("--build-tools", action="store_true", help="Build external SMART tools for reproduction.")
    parser.add_argument(
        "--only-manifold-binding",
        action="store_true",
        help="With --build-tools, build only the fixed Manifold Python/runtime binding.",
    )
    parser.add_argument("--build-cpp", action="store_true", help="Build the local smart._cpp extension and executable.")
    parser.add_argument("--asan", action="store_true", help="With --build-cpp, also build the diagnostic ASan executable.")
    parser.add_argument("--run-smoke", action="store_true", help="Run the configured native smoke pipeline.")
    parser.add_argument("--category", help="Limit --run-smoke to one configured category.")
    parser.add_argument("--mesh", action="append", default=[], help="Limit --run-smoke to one mesh id; repeatable.")
    parser.add_argument("--force", action="store_true", help="Force native smoke stages to rerun.")
    parser.add_argument("--stage-dry-run", action="store_true", help="Pass --dry-run to the SMART native-run stage.")
    parser.add_argument("--release-preflight", action="store_true", help="Build/audit/install-smoke release artifacts.")
    parser.add_argument("--dist-dir", type=Path, default=Path("/private/tmp/smart_release_check"))
    parser.add_argument("--venv-dir", type=Path, default=Path("/private/tmp/smart_release_venv"))
    parser.add_argument("--run-asan-smoke", action="store_true", help="Include the ASan native smoke in release preflight.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_quickstart(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
