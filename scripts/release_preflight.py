from __future__ import annotations

import argparse
import platform
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path


def default_release_target() -> str:
    if sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return "aarch64-apple-darwin"
    return ""


def _release_build_env() -> dict[str, str]:
    env = os.environ.copy()
    if sys.platform == "darwin":
        env.setdefault("MACOSX_DEPLOYMENT_TARGET", "11.0")
    if sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        archflags = env.get("ARCHFLAGS", "")
        if "arm64" not in archflags.split():
            env["ARCHFLAGS"] = "-arch arm64"
    return env


def run_preflight(
    *,
    repo_root: Path,
    dist_dir: Path,
    venv_dir: Path,
    target: str,
    timeout_sec: float,
    skip_build: bool,
    skip_install_smoke: bool,
    recreate_venv: bool,
    cpp_only: bool,
    run_asan_smoke: bool,
) -> None:
    repo_root = repo_root.resolve()
    dist_dir = dist_dir.resolve()
    venv_dir = venv_dir.resolve()
    dist_dir.mkdir(parents=True, exist_ok=True)
    build_env = _release_build_env()

    if not skip_build:
        _run([sys.executable, "-m", "smart", "learned-release-readiness", "--fail-if-not-ready"], cwd=repo_root)
        _run(
            [
                sys.executable,
                "-m",
                "smart",
                "--config",
                "configs/smoke_5.yaml",
                "build-tools",
                "--only-manifold-binding",
            ],
            cwd=repo_root,
            env=build_env,
        )
        _run([sys.executable, "setup.py", "sdist", "--dist-dir", str(dist_dir)], cwd=repo_root, env=build_env)
        _run(
            [sys.executable, "setup.py", "build_ext", "--force", "bdist_wheel", "--dist-dir", str(dist_dir)],
            cwd=repo_root,
            env=build_env,
        )

    if run_asan_smoke:
        _run(
            [
                sys.executable,
                "-m",
                "smart",
                "--config",
                "configs/smoke_5.yaml",
                "build-cpp",
                "--asan",
            ],
            cwd=repo_root,
            env=build_env,
        )
        _run(
            [
                sys.executable,
                "scripts/smoke_native_sanitizers.py",
                "--binary",
                "build/smart-cpp-native-asan",
            ],
            cwd=repo_root,
            env=build_env,
        )

    artifacts = _release_artifacts(dist_dir)
    if not artifacts:
        raise SystemExit(f"no release artifacts found in {dist_dir}")
    audit_cmd = [sys.executable, "scripts/audit_release_wheel.py"]
    audit_cmd.extend(map(str, artifacts))
    _run(audit_cmd, cwd=repo_root)
    _run([sys.executable, "-m", "twine", "check", *map(str, artifacts)], cwd=repo_root)

    if skip_install_smoke:
        return
    wheel = _latest_wheel(dist_dir)
    if wheel is None:
        raise SystemExit(f"no wheel artifact found in {dist_dir}")
    if recreate_venv and venv_dir.exists():
        shutil.rmtree(venv_dir)
    if not _venv_python(venv_dir).exists():
        venv.create(venv_dir, system_site_packages=True, with_pip=True)
    python = _venv_python(venv_dir)
    _run([str(python), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)], cwd=repo_root)
    _run([str(_venv_script(venv_dir, "smart-smoke-console-scripts")), "--timeout-sec", str(timeout_sec)], cwd=repo_root)
    _run([str(_venv_script(venv_dir, "smart")), "learned-release-readiness", "--fail-if-not-ready"], cwd=repo_root)
    smoke_code = (
        "import smart, smart.native as sn, smart.cpp as sc, smart.pymesh_compat as compat, pymesh; "
        "from smart.pipeline.config import load_config; "
        "from smart.pipeline.tools import diagnose_environment; "
        "status = diagnose_environment(load_config('smoke_5.yaml')); "
        "checks = {c['name']: c for c in status['checks']}; "
        "assert sn.using_cpp(); "
        "assert sc.using_cpp(); "
        "assert smart.cpp_native_available(); "
        "assert smart.NativeSmartEngine is sc.NativeSmartEngine; "
        "assert compat.form_mesh is pymesh.form_mesh; "
        "assert checks['smart-cpp-extension']['ok']; "
        "print('smart-bbox release preflight smoke ok')"
    )
    _run([str(python), "-c", smoke_code], cwd=repo_root)


def _release_artifacts(dist_dir: Path) -> list[Path]:
    return sorted([*dist_dir.glob("*.whl"), *dist_dir.glob("*.tar.gz")])


def _latest_wheel(dist_dir: Path) -> Path | None:
    wheels = sorted(dist_dir.glob("*.whl"), key=lambda path: path.stat().st_mtime)
    return wheels[-1] if wheels else None


def _venv_python(venv_dir: Path) -> Path:
    return _venv_script(venv_dir, "python")


def _venv_script(venv_dir: Path, name: str) -> Path:
    scripts_dir = "Scripts" if sys.platform == "win32" else "bin"
    suffix = ".exe" if sys.platform == "win32" and not name.endswith(".exe") else ""
    return venv_dir / scripts_dir / f"{name}{suffix}"


def _run(argv: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(argv), flush=True)
    subprocess.run(argv, cwd=cwd, check=True, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and verify SMART release candidate artifacts.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="SMART source checkout root")
    parser.add_argument("--dist-dir", type=Path, default=Path("/private/tmp/smart_release_check"))
    parser.add_argument("--venv-dir", type=Path, default=Path("/private/tmp/smart_release_venv"))
    parser.add_argument("--target", default=default_release_target(), help="Unused compatibility option; C++ wheels are built with setuptools")
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--skip-build", action="store_true", help="Reuse existing artifacts in --dist-dir")
    parser.add_argument("--skip-install-smoke", action="store_true", help="Skip installing the wheel into --venv-dir")
    parser.add_argument("--recreate-venv", action="store_true", help="Delete and recreate --venv-dir before install smoke")
    parser.add_argument("--cpp-only", action="store_true", help="Compatibility no-op; release wheels are native C++ by default")
    parser.add_argument(
        "--run-asan-smoke",
        action="store_true",
        help="Build build/smart-cpp-native-asan and run the short AddressSanitizer native smoke",
    )
    args = parser.parse_args(argv)

    run_preflight(
        repo_root=args.repo_root,
        dist_dir=args.dist_dir,
        venv_dir=args.venv_dir,
        target=args.target,
        timeout_sec=args.timeout_sec,
        skip_build=args.skip_build,
        skip_install_smoke=args.skip_install_smoke,
        recreate_venv=args.recreate_venv,
        cpp_only=args.cpp_only,
        run_asan_smoke=args.run_asan_smoke,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
