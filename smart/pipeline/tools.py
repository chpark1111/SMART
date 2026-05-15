from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
from pathlib import Path
from typing import Any
import shutil
import sys
import zipfile

from .config import REPO_ROOT, repo_path
from .runner import find_executable, run_command


MANIFOLDPLUS_REPO = "https://github.com/hjwdzh/ManifoldPlus.git"
FTETWILD_REPO = "https://github.com/wildmeshing/fTetWild.git"
FTETWILD_SMART_PATCH = REPO_ROOT / "patches" / "ftetwild_smart_crash_guards.patch"


def diagnose_environment(cfg: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    vendored_manifold = REPO_ROOT / "smart" / "vendor" / "manifold"
    default_manifold_python = vendored_manifold / "build" / "bindings" / "python"
    configured_manifold_python = os.environ.get("SMART_MANIFOLD_PYTHON")
    manifold_python_path = (
        Path(configured_manifold_python).expanduser()
        if configured_manifold_python
        else default_manifold_python
    )

    def add(
        name: str,
        ok: bool,
        *,
        kind: str,
        detail: str | None = None,
        path: str | None = None,
        required_for: list[str] | None = None,
    ) -> None:
        checks.append(
            {
                "name": name,
                "kind": kind,
                "ok": bool(ok),
                "detail": detail,
                "path": path,
                "required_for": required_for or [],
            }
        )

    add("python", True, kind="runtime", detail=sys.version.split()[0], path=sys.executable)
    rust_status = _smart_rust_status()
    add(
        "smart-rust-extension",
        rust_status["ok"],
        kind="python-extension",
        detail=rust_status["detail"],
        path=rust_status["path"],
        required_for=[],
    )

    for module_name, required_for in [
        ("numpy", ["preseg", "merge", "refine", "mcts", "render"]),
        ("trimesh", ["preseg", "merge", "refine", "mcts", "render"]),
        ("pymesh", ["merge", "refine", "mcts"]),
        ("pymanifold", ["merge", "refine", "mcts"]),
        ("coacd", ["preseg"]),
        ("torch", ["merge", "refine", "mcts"]),
        ("yaml", ["config-yaml"]),
    ]:
        spec = importlib.util.find_spec(module_name)
        ok = spec is not None
        path = getattr(spec, "origin", None) if spec is not None else None
        detail = None
        if module_name == "pymanifold" and spec is None:
            pymanifold_binary = _first_pymanifold_binary(manifold_python_path)
            if pymanifold_binary is not None:
                ok = True
                path = str(pymanifold_binary)
                detail = "available through SMART_MANIFOLD_PYTHON path used by legacy runners"
        add(
            module_name,
            ok,
            kind="python-module",
            detail=detail,
            path=path,
            required_for=required_for,
        )

    tools = cfg.get("tools", {})
    assert isinstance(tools, dict)
    _check_executable(
        checks,
        "ManifoldPlus",
        _env_or_config_path("SMART_MANIFOLDPLUS_BIN", tools.get("manifoldplus_bin")),
        fallback="manifold",
        required_for=["tetra"],
    )
    _check_executable(
        checks,
        "fTetWild",
        _env_or_config_path("SMART_FTETWILD_BIN", tools.get("ftetwild_bin")),
        fallback="FloatTetwild_bin",
        required_for=["tetra"],
    )
    mesh2tet_root = repo_path(tools.get("mesh2tet_external_root", "external/mesh2tet"))
    if mesh2tet_root is not None:
        ftetwild_source = mesh2tet_root / "fTetWild"
        add(
            "fTetWild-source",
            _source_ready(ftetwild_source),
            kind="source-tree",
            path=str(ftetwild_source),
            detail=_git_source_detail(ftetwild_source, expected_remote=FTETWILD_REPO),
            required_for=[],
        )
    _check_executable(
        checks,
        "Blender",
        _env_or_config_path("SMART_BLENDER_BIN", tools.get("blender_bin")),
        fallback="blender",
        extra_candidates=["/Applications/Blender.app/Contents/MacOS/blender"],
        required_for=["render"],
    )
    _check_executable(checks, "cmake", None, fallback="cmake", required_for=["build-tools"])
    _check_executable(checks, "git", None, fallback="git", required_for=["build-tools"])
    _check_executable(
        checks,
        "cargo",
        _env_or_config_path("SMART_CARGO_BIN", tools.get("cargo_bin")),
        fallback="cargo",
        required_for=["build-rust"],
    )
    _check_executable(
        checks,
        "maturin",
        _env_or_config_path("SMART_MATURIN_BIN", tools.get("maturin_bin")),
        fallback="maturin",
        required_for=["build-rust"],
    )

    add(
        "vendored-manifold-source",
        (vendored_manifold / "CMakeLists.txt").exists()
        and (vendored_manifold / "bindings" / "python" / "pymanifold.cpp").exists(),
        kind="source-tree",
        path=str(vendored_manifold),
        detail="kept as fixed C++ binding source; do not pull or replace",
        required_for=["build-tools", "merge", "refine", "mcts"],
    )

    if configured_manifold_python:
        add(
            "SMART_MANIFOLD_PYTHON",
            manifold_python_path.exists(),
            kind="env-path",
            path=str(manifold_python_path),
            required_for=["merge", "refine", "mcts"],
        )
    else:
        add(
            "SMART_MANIFOLD_PYTHON",
            default_manifold_python.exists(),
            kind="env-path",
            path=str(default_manifold_python),
            detail="not set; default vendored build path checked",
            required_for=["merge", "refine", "mcts"],
        )

    required_failures = [
        check
        for check in checks
        if check["required_for"] and not check["ok"] and "build-rust" not in check["required_for"]
    ]
    optional_failures = [check for check in checks if not check["ok"] and check not in required_failures]
    return {
        "ok": not required_failures,
        "checks": checks,
        "required_failures": required_failures,
        "optional_failures": optional_failures,
    }


def _smart_rust_status() -> dict[str, Any]:
    try:
        from .. import rust as smart_rust
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "detail": f"import failed; Python fallbacks active: {exc}",
            "path": None,
        }

    features = smart_rust.backend_features()
    available = sum(1 for enabled in features.values() if enabled)
    total = len(features)
    if smart_rust.using_rust():
        missing = [name for name, enabled in features.items() if not enabled]
        detail = f"loaded; kernels={available}/{total}"
        if missing:
            detail += "; missing=" + ",".join(missing)
        return {"ok": True, "detail": detail, "path": smart_rust.backend_path()}

    return {
        "ok": False,
        "detail": f"not installed; Python fallbacks active; kernels={available}/{total}",
        "path": None,
    }


def _first_pymanifold_binary(path: Path) -> Path | None:
    if not path.exists():
        return None
    for pattern in ("pymanifold*.so", "pymanifold*.pyd", "pymanifold*.dylib"):
        matches = sorted(path.glob(pattern))
        if matches:
            return matches[0]
    return None


def _git_source_detail(path: Path, *, expected_remote: str | None = None) -> str | None:
    if not path.exists():
        return None
    if not (path / ".git").exists():
        return "source tree present; not a git checkout"
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        ).stdout.strip()
        remote = subprocess.run(
            ["git", "-C", str(path), "remote", "get-url", "origin"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"git metadata unavailable: {exc}"

    detail = f"origin={remote or 'unknown'} head={head or 'unknown'}"
    if expected_remote and remote and remote != expected_remote:
        detail += f"; expected origin={expected_remote}"
    return detail


def build_tools(cfg: dict[str, Any], *, dry_run: bool = False) -> list[str]:
    root = repo_path(cfg.get("tools", {}).get("mesh2tet_external_root", "external/mesh2tet"))
    assert root is not None
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)
    logs = repo_path(cfg.get("workspace", "runs/demo"))
    assert logs is not None
    log_root = logs / "logs" / "build-tools"
    messages: list[str] = []

    manifoldplus = root / "ManifoldPlus"
    if not _source_ready(manifoldplus):
        if manifoldplus.exists() and not dry_run:
            shutil.rmtree(manifoldplus)
        result = run_command(
            ["git", "clone", "--recursive", MANIFOLDPLUS_REPO, str(manifoldplus)],
            timeout=3600,
            log_path=log_root / "clone-manifoldplus.log",
            dry_run=dry_run,
        )
        messages.append(_line("ManifoldPlus clone", result.returncode, dry_run))
    messages.extend(_cmake_build(manifoldplus, log_root, "manifoldplus", dry_run=dry_run))

    ftetwild = root / "fTetWild"
    if not _source_ready(ftetwild):
        if ftetwild.exists() and not dry_run:
            shutil.rmtree(ftetwild)
        result = run_command(
            ["git", "clone", "--recursive", FTETWILD_REPO, str(ftetwild)],
            timeout=3600,
            log_path=log_root / "clone-ftetwild.log",
            dry_run=dry_run,
        )
        messages.append(_line("fTetWild clone", result.returncode, dry_run))
    messages.extend(
        _apply_patch_once(
            ftetwild,
            FTETWILD_SMART_PATCH,
            log_root,
            "fTetWild SMART crash guards",
            dry_run=dry_run,
        )
    )
    messages.extend(
        _cmake_build(
            ftetwild,
            log_root,
            "ftetwild",
            cmake_args=[
                "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
                "-DCMAKE_POLICY_DEFAULT_CMP0169=OLD",
                "-DFLOAT_TETWILD_ENABLE_TBB=OFF",
                f"-DCMAKE_PROJECT_INCLUDE_BEFORE={REPO_ROOT / 'cmake' / 'ftetwild_compat.cmake'}",
                f"-DCMAKE_PROJECT_libigl_INCLUDE_BEFORE={REPO_ROOT / 'cmake' / 'ftetwild_compat.cmake'}",
            ],
            dry_run=dry_run,
        )
    )

    messages.extend(build_vendored_manifold_binding(cfg, dry_run=dry_run))
    messages.append(f"Set SMART_MANIFOLDPLUS_BIN={manifoldplus / 'build' / 'manifold'} if it is not on PATH.")
    messages.append(f"Set SMART_FTETWILD_BIN={ftetwild / 'build' / 'FloatTetwild_bin'} if it is not on PATH.")
    return messages


def _env_or_config_path(env_name: str, configured: object) -> str | None:
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    if configured is None:
        return None
    return str(configured)


def _check_executable(
    checks: list[dict[str, Any]],
    name: str,
    explicit: str | None,
    *,
    fallback: str,
    required_for: list[str],
    extra_candidates: list[str] | None = None,
) -> None:
    path = _resolve_executable(explicit, f"SMART_{name.upper()}_BIN", fallback)
    if path is None:
        for candidate in extra_candidates or []:
            if Path(candidate).exists():
                path = candidate
                break
    checks.append(
        {
            "name": name,
            "kind": "executable",
            "ok": path is not None,
            "detail": None if path else f"not found; configure tools.{name.lower()}_bin or environment override",
            "path": path,
            "required_for": required_for,
        }
    )


def _resolve_executable(explicit: str | None, env_name: str, fallback: str) -> str | None:
    if explicit:
        if "/" not in explicit:
            found = shutil.which(explicit)
            if found:
                return found
        explicit_path = repo_path(explicit)
        assert explicit_path is not None
        if explicit_path.exists():
            return str(explicit_path)
        return None
    return find_executable(None, env_name, fallback)


def build_vendored_manifold_binding(cfg: dict[str, Any], *, dry_run: bool = False) -> list[str]:
    source = REPO_ROOT / "smart" / "vendor" / "manifold"
    build = source / "build"
    workspace = repo_path(cfg.get("workspace", "runs/demo"))
    assert workspace is not None
    log_root = workspace / "logs" / "build-tools"
    return _cmake_build(
        source,
        log_root,
        "vendored-manifold",
        build_dir=build,
        build_target="pymanifold",
        cmake_args=[
            "-DMANIFOLD_PYBIND=ON",
            "-DMANIFOLD_CBIND=OFF",
            "-DMANIFOLD_USE_CUDA=OFF",
            "-DCMAKE_CXX_FLAGS=-D_VSTD=std -Wno-error=missing-template-arg-list-after-template-kw",
        ],
        dry_run=dry_run,
    )


def build_rust_extension(cfg: dict[str, Any], *, dry_run: bool = False, release: bool = True) -> list[str]:
    workspace = repo_path(cfg.get("workspace", "runs/demo"))
    assert workspace is not None
    log_root = workspace / "logs" / "build-rust"
    wheel_root = log_root / "wheels"
    if not dry_run:
        wheel_root.mkdir(parents=True, exist_ok=True)
    project = REPO_ROOT
    root_pyproject = project / "pyproject.toml"
    root_cargo = project / "Cargo.toml"
    rust_manifest = project / "rust" / "smart-core" / "Cargo.toml"
    if not dry_run and not (root_pyproject.exists() and root_cargo.exists() and rust_manifest.exists()):
        return [
            "smart-bbox maturin build: failed rc=127",
            "smart build-rust requires a SMART source checkout; release wheels already bundle smart._rust.",
        ]
    tools = cfg.get("tools", {})
    assert isinstance(tools, dict)
    maturin = _resolve_executable(
        _env_or_config_path("SMART_MATURIN_BIN", tools.get("maturin_bin")),
        "SMART_MATURIN_BIN",
        "maturin",
    )
    if maturin is None and not dry_run:
        return ["smart-bbox maturin build: failed rc=127", "Install maturin, then rerun: smart build-rust"]

    python = str(cfg.get("python") or sys.executable)
    rust_target = str(tools.get("rust_target") or _default_rust_target())
    command = [maturin or "maturin", "build", "--interpreter", python, "--out", str(wheel_root)]
    if rust_target:
        command.extend(["--target", rust_target])
    if release:
        command.append("--release")

    build_result = run_command(
        command,
        cwd=project,
        timeout=3600,
        log_path=log_root / "maturin-build.log",
        dry_run=dry_run,
    )
    messages = [_line("smart-bbox maturin build", build_result.returncode, dry_run)]
    if not build_result.ok and not dry_run:
        messages.append("Install Rust/Cargo and maturin, then rerun: smart build-rust")
        return messages

    wheels = sorted(wheel_root.glob("smart_bbox-*.whl"), key=lambda path: path.stat().st_mtime)
    wheel = wheels[-1] if wheels else wheel_root / "smart_bbox-*.whl"
    install_command = [python, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)]
    install_result = run_command(
        install_command,
        timeout=900,
        log_path=log_root / "pip-install-smart-bbox.log",
        dry_run=dry_run,
    )
    messages.append(_line("smart-bbox pip install", install_result.returncode, dry_run))
    if install_result.ok or dry_run:
        extracted = _extract_local_rust_extension(wheel, dry_run=dry_run)
        if extracted:
            messages.append(extracted)
    return messages


def _extract_local_rust_extension(wheel: Path, *, dry_run: bool = False) -> str | None:
    if dry_run:
        return "smart._rust local source extension: dry-run"
    if not wheel.exists():
        return None
    destination = REPO_ROOT / "smart"
    if not destination.exists():
        return None
    with zipfile.ZipFile(wheel) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.startswith("smart/_rust")
            and (name.endswith(".so") or name.endswith(".pyd") or name.endswith(".dll"))
        ]
        if not members:
            return None
        for existing in destination.glob("_rust*.so"):
            existing.unlink()
        for existing in destination.glob("_rust*.pyd"):
            existing.unlink()
        for member in members:
            target = destination / Path(member).name
            with archive.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
    return f"smart._rust local source extension: ok -> {destination}"


def _default_rust_target() -> str | None:
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "aarch64-apple-darwin"
    return None


def _cmake_build(
    source: Path,
    log_root: Path,
    name: str,
    *,
    build_dir: Path | None = None,
    build_target: str | None = None,
    cmake_args: list[str] | None = None,
    dry_run: bool = False,
) -> list[str]:
    build = build_dir or (source / "build")
    if not dry_run:
        build.mkdir(parents=True, exist_ok=True)
    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
    ] + (cmake_args or [])
    result = run_command(
        configure,
        timeout=3600,
        log_path=log_root / f"{name}-configure.log",
        dry_run=dry_run,
    )
    messages = [_line(f"{name} configure", result.returncode, dry_run)]
    if result.ok or dry_run:
        build_command = ["cmake", "--build", str(build), "--config", "Release"]
        if build_target:
            build_command.extend(["--target", build_target])
        build_command.append("-j")
        build_result = run_command(
            build_command,
            timeout=7200,
            log_path=log_root / f"{name}-build.log",
            dry_run=dry_run,
        )
        messages.append(_line(f"{name} build", build_result.returncode, dry_run))
    return messages


def _apply_patch_once(
    source: Path,
    patch_path: Path,
    log_root: Path,
    label: str,
    *,
    dry_run: bool = False,
) -> list[str]:
    if dry_run:
        return [f"{label}: dry-run"]
    if not _source_ready(source):
        return [f"{label}: skipped; source missing"]
    if not patch_path.exists():
        return [f"{label}: skipped; missing patch {patch_path}"]

    check = run_command(
        ["git", "apply", "--check", str(patch_path)],
        cwd=source,
        timeout=120,
        log_path=log_root / "ftetwild-smart-patch-check.log",
    )
    if check.ok:
        applied = run_command(
            ["git", "apply", str(patch_path)],
            cwd=source,
            timeout=120,
            log_path=log_root / "ftetwild-smart-patch-apply.log",
        )
        return [_line(label, applied.returncode, dry_run)]

    reverse_check = run_command(
        ["git", "apply", "--reverse", "--check", str(patch_path)],
        cwd=source,
        timeout=120,
        log_path=log_root / "ftetwild-smart-patch-reverse-check.log",
    )
    if reverse_check.ok:
        return [f"{label}: already applied"]
    return [f"{label}: failed rc={check.returncode}; see {check.log_path}"]


def _line(label: str, returncode: int, dry_run: bool) -> str:
    if dry_run:
        return f"{label}: dry-run"
    return f"{label}: {'ok' if returncode == 0 else f'failed rc={returncode}'}"


def _source_ready(path: Path) -> bool:
    return (path / "CMakeLists.txt").exists()
