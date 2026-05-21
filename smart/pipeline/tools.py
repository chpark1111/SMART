from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sysconfig
from pathlib import Path
from typing import Any
import shutil
import sys

from .config import REPO_ROOT, repo_path
from .runner import find_executable, run_command
from ..native_executable import native_executable_path


MANIFOLDPLUS_REPO = "https://github.com/hjwdzh/ManifoldPlus.git"
FTETWILD_REPO = "https://github.com/wildmeshing/fTetWild.git"
COACD_REPO = "https://github.com/SarahWeiii/CoACD.git"
FTETWILD_SMART_PATCH = REPO_ROOT / "patches" / "ftetwild_smart_crash_guards.patch"
_TRUTHY = {"1", "true", "yes", "on"}


def _nested_get(mapping: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _bool_env_or_config(env_name: str, configured: Any, *, default: bool = False) -> bool:
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return env_value.strip().lower() in _TRUTHY
    if configured is None:
        return default
    if isinstance(configured, bool):
        return configured
    return str(configured).strip().lower() in _TRUTHY


def _manifold_parallel_backend(cfg: dict[str, Any]) -> str:
    configured = (
        _nested_get(cfg, ("build_tools", "manifold_parallel_backend"))
        or _nested_get(cfg, ("manifold", "parallel_backend"))
        or _nested_get(cfg, ("tools", "manifold_parallel_backend"))
    )
    value = os.environ.get("SMART_MANIFOLD_PAR", configured or "NONE")
    normalized = str(value).strip().upper()
    aliases = {
        "": "NONE",
        "NO": "NONE",
        "OFF": "NONE",
        "FALSE": "NONE",
        "SERIAL": "NONE",
        "OPENMP": "OMP",
        "OMP": "OMP",
        "TBB": "TBB",
        "NONE": "NONE",
    }
    if normalized not in aliases:
        raise ValueError(
            "Unsupported Manifold parallel backend "
            f"{value!r}; use NONE, OpenMP/OMP, or TBB"
        )
    return aliases[normalized]


def _manifold_use_cuda(cfg: dict[str, Any]) -> bool:
    configured = (
        _nested_get(cfg, ("build_tools", "manifold_use_cuda"))
        or _nested_get(cfg, ("manifold", "use_cuda"))
        or _nested_get(cfg, ("tools", "manifold_use_cuda"))
    )
    return _bool_env_or_config("SMART_MANIFOLD_USE_CUDA", configured, default=False)


def _manifold_relax_werror(cfg: dict[str, Any]) -> bool:
    configured = (
        _nested_get(cfg, ("build_tools", "manifold_relax_werror"))
        or _nested_get(cfg, ("manifold", "relax_werror"))
        or _nested_get(cfg, ("tools", "manifold_relax_werror"))
    )
    return _bool_env_or_config("SMART_MANIFOLD_RELAX_WERROR", configured, default=True)


def _werror_filter_compiler(log_root: Path, *, dry_run: bool = False) -> Path:
    wrapper = log_root / "cxx-filter-werror.py"
    if dry_run:
        return wrapper
    real_cxx = os.environ.get("CXX") or shutil.which("c++") or "c++"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import subprocess",
                "import sys",
                f"REAL_CXX = {real_cxx!r}",
                "args = [arg for arg in sys.argv[1:] if arg != '-Werror']",
                "raise SystemExit(subprocess.call([REAL_CXX, *args]))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper


def _cmake_cache_value(cache: Path, key: str) -> str | None:
    if not cache.exists():
        return None
    for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(f"{key}:"):
            return line.rsplit("=", 1)[-1].strip()
    return None


def _optional_tool_prefix(env_name: str, package_name: str) -> Path | None:
    configured = os.environ.get(env_name)
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            Path("/opt/homebrew/opt") / package_name,
            Path("/usr/local/opt") / package_name,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _manifold_parallel_cmake_args(parallel_backend: str) -> list[str]:
    if sys.platform != "darwin":
        return []
    if parallel_backend == "OMP":
        libomp = _optional_tool_prefix("SMART_LIBOMP_PREFIX", "libomp")
        if libomp is None:
            return []
        include = libomp / "include"
        library = libomp / "lib" / "libomp.dylib"
        if not library.exists():
            return []
        flags = f"-Xpreprocessor -fopenmp -I{include}"
        return [
            f"-DOpenMP_C_FLAGS={flags}",
            f"-DOpenMP_CXX_FLAGS={flags}",
            "-DOpenMP_C_LIB_NAMES=omp",
            "-DOpenMP_CXX_LIB_NAMES=omp",
            f"-DOpenMP_omp_LIBRARY={library}",
        ]
    if parallel_backend == "TBB":
        tbb = _optional_tool_prefix("SMART_TBB_PREFIX", "tbb")
        if tbb is None:
            return []
        existing = os.environ.get("CMAKE_PREFIX_PATH")
        prefix_path = f"{tbb};{existing}" if existing else str(tbb)
        return [f"-DCMAKE_PREFIX_PATH={prefix_path}"]
    return []


def _manifold_cmake_cache_backend(manifold_root: Path) -> str:
    cache = manifold_root / "build" / "CMakeCache.txt"
    if not cache.exists():
        return "NONE"
    for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MANIFOLD_PAR:"):
            return line.rsplit("=", 1)[-1].strip().upper()
    return "NONE"


def _manifold_native_link_args(manifold_root: Path) -> list[str]:
    backend = _manifold_cmake_cache_backend(manifold_root)
    if backend == "TBB":
        tbb = _optional_tool_prefix("SMART_TBB_PREFIX", "tbb")
        if tbb is not None:
            library = tbb / "lib" / "libtbb.12.dylib"
            if library.exists():
                return [str(library)]
            return ["-L", str(tbb / "lib"), "-ltbb"]
        return ["-ltbb"]
    if backend == "OMP":
        libomp = _optional_tool_prefix("SMART_LIBOMP_PREFIX", "libomp")
        if libomp is not None:
            library = libomp / "lib" / "libomp.dylib"
            if library.exists():
                return [str(library)]
            return ["-L", str(libomp / "lib"), "-lomp"]
        return ["-lomp"]
    return []


def _macos_arm64_compile_flags() -> list[str]:
    if sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return ["-arch", "arm64"]
    return []


def _macos_deployment_target() -> str | None:
    if sys.platform != "darwin":
        return None
    return os.environ.get("MACOSX_DEPLOYMENT_TARGET") or "11.0"


def _macos_min_version_flags() -> list[str]:
    target = _macos_deployment_target()
    if target is None:
        return []
    return [f"-mmacosx-version-min={target}"]


def _macos_cmake_platform_args() -> list[str]:
    target = _macos_deployment_target()
    if target is None:
        return []
    args = [f"-DCMAKE_OSX_DEPLOYMENT_TARGET={target}"]
    if platform.machine().lower() in {"arm64", "aarch64"}:
        args.append("-DCMAKE_OSX_ARCHITECTURES=arm64")
    return args


def diagnose_environment(cfg: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    vendored_manifold = REPO_ROOT / "smart" / "vendor" / "manifold"
    packaged_manifold_python = REPO_ROOT / "smart" / "pymanifold_runtime"
    source_manifold_python = vendored_manifold / "build" / "bindings" / "python"
    default_manifold_python = (
        packaged_manifold_python
        if _first_pymanifold_binary(packaged_manifold_python) is not None
        else source_manifold_python
    )
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
    cpp_status = _smart_cpp_status()
    add(
        "smart-cpp-extension",
        cpp_status["ok"],
        kind="python-extension",
        detail=cpp_status["detail"],
        path=cpp_status["path"],
        required_for=[],
    )
    compat_spec = importlib.util.find_spec("smart.pymesh_compat")
    add(
        "smart-pymesh-compat",
        compat_spec is not None,
        kind="python-module",
        detail="official minimal PyMesh replacement used by SMART legacy runtimes",
        path=getattr(compat_spec, "origin", None) if compat_spec is not None else None,
        required_for=["merge", "refine", "mcts"],
    )

    for module_name, required_for in [
        ("numpy", ["preseg", "merge", "refine", "mcts", "render"]),
        ("trimesh", ["preseg", "merge", "refine", "mcts", "render"]),
        ("pymesh", []),
        ("pymanifold", ["merge", "refine", "mcts"]),
        ("coacd", ["preseg"]),
        ("torch", ["merge", "refine", "mcts"]),
        ("yaml", ["config-yaml"]),
    ]:
        spec = importlib.util.find_spec(module_name)
        ok = spec is not None
        path = getattr(spec, "origin", None) if spec is not None else None
        detail = None
        if module_name == "pymesh" and spec is not None:
            detail = "legacy alias for smart.pymesh_compat"
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
    coacd_source = repo_path(tools.get("coacd_external_root", "external/CoACD"))
    if coacd_source is not None:
        add(
            "CoACD-source",
            _source_ready(coacd_source),
            kind="source-tree",
            path=str(coacd_source),
            detail=_git_source_detail(coacd_source, expected_remote=COACD_REPO),
            required_for=[],
        )
    _check_executable(
        checks,
        "CoACD-CLI",
        _env_or_config_path("SMART_COACD_BIN", tools.get("coacd_bin")),
        fallback="coacd",
        required_for=[],
        extra_candidates=[
            str(REPO_ROOT / "external" / "CoACD" / "python" / "package" / "bin" / "coacd"),
            str(REPO_ROOT / "external" / "CoACD" / "build" / "main"),
        ],
    )
    _check_executable(
        checks,
        "Blender",
        _env_or_config_path("SMART_BLENDER_BIN", tools.get("blender_bin")),
        fallback="blender",
        extra_candidates=["/Applications/Blender.app/Contents/MacOS/blender"],
        required_for=["render"],
    )
    cpp_native_path = native_executable_path(tools.get("smart_cpp_native_bin"))
    cpp_native_bin = str(cpp_native_path) if cpp_native_path is not None else None
    checks.append(
        {
            "name": "smart-cpp-native",
            "kind": "executable",
            "ok": cpp_native_bin is not None,
            "detail": None if cpp_native_bin else "not found; run `smart build-cpp`",
            "path": cpp_native_bin,
            "required_for": [],
        }
    )
    _check_executable(checks, "cmake", None, fallback="cmake", required_for=["build-tools"])
    _check_executable(checks, "git", None, fallback="git", required_for=["build-tools"])
    pymanifold_runtime_available = _first_pymanifold_binary(default_manifold_python) is not None
    add(
        "vendored-manifold-source",
        (vendored_manifold / "CMakeLists.txt").exists()
        and (vendored_manifold / "bindings" / "python" / "pymanifold.cpp").exists(),
        kind="source-tree",
        path=str(vendored_manifold),
        detail="kept as fixed C++ binding source; do not pull or replace",
        required_for=[] if pymanifold_runtime_available else ["build-tools"],
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
            detail="not set; default pymanifold runtime path checked",
            required_for=["merge", "refine", "mcts"],
        )

    required_failures = [check for check in checks if check["required_for"] and not check["ok"]]
    optional_failures = [check for check in checks if not check["ok"] and check not in required_failures]
    return {
        "ok": not required_failures,
        "checks": checks,
        "required_failures": required_failures,
        "optional_failures": optional_failures,
    }


def _smart_cpp_status() -> dict[str, Any]:
    try:
        from .. import cpp as smart_cpp
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "detail": f"import failed: {exc}",
            "path": None,
        }
    if smart_cpp.using_cpp():
        return {
            "ok": True,
            "detail": "loaded; native C++ core and fixed-Manifold bridge available",
            "path": smart_cpp.backend_path(),
        }
    return {
        "ok": False,
        "detail": "not installed; run: smart build-cpp",
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
    tools = cfg.get("tools", {})
    assert isinstance(tools, dict)
    root = repo_path(tools.get("mesh2tet_external_root", "external/mesh2tet"))
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

    coacd_root = repo_path(tools.get("coacd_external_root", "external/CoACD"))
    if coacd_root is not None:
        if not _source_ready(coacd_root):
            if coacd_root.exists() and not dry_run:
                shutil.rmtree(coacd_root)
            result = run_command(
                ["git", "clone", "--recursive", COACD_REPO, str(coacd_root)],
                timeout=3600,
                log_path=log_root / "clone-coacd.log",
                dry_run=dry_run,
            )
            messages.append(_line("CoACD source clone", result.returncode, dry_run))
        if _bool_env_or_config("SMART_COACD_BUILD", tools.get("coacd_build"), default=False):
            messages.extend(_cmake_build(coacd_root, log_root, "coacd", dry_run=dry_run))
        else:
            messages.append(
                f"CoACD source checkout: {'dry-run -> ' if dry_run else ''}{coacd_root}; "
                "runtime uses the upstream CoACD CLI when available"
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
            if _is_executable_or_python_script(Path(candidate)):
                path = candidate
                break
    detail = None if path else f"not found; configure tools.{name.lower()}_bin or environment override"
    if path is not None and not os.access(path, os.X_OK):
        detail = "Python script without executable bit; SMART will launch it through Python"
    checks.append(
        {
            "name": name,
            "kind": "executable",
            "ok": path is not None,
            "detail": detail,
            "path": path,
            "required_for": required_for,
        }
    )


def _is_executable_or_python_script(path: Path) -> bool:
    if not path.exists():
        return False
    if os.access(path, os.X_OK):
        return True
    try:
        header = path.read_bytes()[:128]
    except OSError:
        return False
    return header.startswith(b"#!") and b"python" in header.lower()


def _resolve_executable(explicit: str | None, env_name: str, fallback: str) -> str | None:
    if explicit:
        if "/" not in explicit:
            found = shutil.which(explicit)
            if found:
                return found
        explicit_path = repo_path(explicit)
        assert explicit_path is not None
        if _is_executable_or_python_script(explicit_path):
            return str(explicit_path)
        return None
    return find_executable(None, env_name, fallback)


def build_vendored_manifold_binding(cfg: dict[str, Any], *, dry_run: bool = False) -> list[str]:
    source = REPO_ROOT / "smart" / "vendor" / "manifold"
    build = source / "build"
    workspace = repo_path(cfg.get("workspace", "runs/demo"))
    assert workspace is not None
    log_root = workspace / "logs" / "build-tools"
    manifold_parallel = _manifold_parallel_backend(cfg)
    manifold_use_cuda = _manifold_use_cuda(cfg)
    manifold_relax_werror = _manifold_relax_werror(cfg)
    cmake_args = [
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        "-DMANIFOLD_PYBIND=ON",
        "-DMANIFOLD_CBIND=OFF",
        f"-DMANIFOLD_PAR={manifold_parallel}",
        f"-DMANIFOLD_USE_CUDA={'ON' if manifold_use_cuda else 'OFF'}",
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
    ] + _manifold_parallel_cmake_args(manifold_parallel)
    if sys.platform == "darwin":
        cmake_args.append(
            "-DCMAKE_CXX_FLAGS=-D_VSTD=std -Wno-error=missing-template-arg-list-after-template-kw"
        )
    cached_python = (
        _cmake_cache_value(build / "CMakeCache.txt", "Python3_EXECUTABLE")
        or _cmake_cache_value(build / "CMakeCache.txt", "PYTHON_EXECUTABLE")
    )
    if cached_python and cached_python != sys.executable and not dry_run:
        shutil.rmtree(build)
    if manifold_relax_werror:
        compiler_wrapper = build / "cxx-filter-werror.py"
        cached_compiler = _cmake_cache_value(build / "CMakeCache.txt", "CMAKE_CXX_COMPILER")
        if cached_compiler and cached_compiler != str(compiler_wrapper) and not dry_run:
            shutil.rmtree(build)
        cmake_args.append(f"-DCMAKE_CXX_COMPILER={_werror_filter_compiler(build, dry_run=dry_run)}")
    messages = _cmake_build(
        source,
        log_root,
        "vendored-manifold",
        build_dir=build,
        build_target="pymanifold",
        cmake_args=cmake_args,
        dry_run=dry_run,
    )
    messages.append(
        "Vendored Manifold build backend: "
        f"MANIFOLD_PAR={manifold_parallel}, "
        f"MANIFOLD_USE_CUDA={'ON' if manifold_use_cuda else 'OFF'}, "
        f"RELAX_WERROR={'ON' if manifold_relax_werror else 'OFF'}"
    )
    manifold_lib = _find_vendored_manifold_lib(source)
    if dry_run:
        messages.append(f"vendored-manifold static library: dry-run -> {source / 'build' / 'src' / 'manifold' / 'libmanifold.a'}")
    elif manifold_lib is None:
        messages.append(
            "vendored-manifold static library: failed; missing "
            f"{source / 'build' / 'src' / 'manifold' / 'libmanifold.a'}"
        )
    else:
        messages.append(f"vendored-manifold static library: ok -> {manifold_lib}")
    messages.append(_stage_pymanifold_runtime(build / "bindings" / "python", dry_run=dry_run))
    return messages


def _stage_pymanifold_runtime(source_dir: Path, *, dry_run: bool = False) -> str:
    destination = REPO_ROOT / "smart" / "pymanifold_runtime"
    if dry_run:
        return f"pymanifold runtime staging: dry-run -> {destination}"
    binary = _first_pymanifold_binary(source_dir)
    if binary is None:
        return f"pymanifold runtime staging: failed; no pymanifold binary in {source_dir}"
    destination.mkdir(parents=True, exist_ok=True)
    for pattern in ("pymanifold*.so", "pymanifold*.pyd", "pymanifold*.dylib"):
        for existing in destination.glob(pattern):
            existing.unlink()
    target = destination / binary.name
    shutil.copy2(binary, target)
    return f"pymanifold runtime staging: ok -> {target}"


def build_cpp_extension(
    cfg: dict[str, Any],
    *,
    dry_run: bool = False,
    release: bool = True,
    asan: bool = False,
) -> list[str]:
    workspace = repo_path(cfg.get("workspace", "runs/demo"))
    assert workspace is not None
    log_root = workspace / "logs" / "build-cpp"
    if not dry_run:
        log_root.mkdir(parents=True, exist_ok=True)

    compiler = (
        str(cfg.get("tools", {}).get("cxx_bin"))
        if isinstance(cfg.get("tools"), dict) and cfg.get("tools", {}).get("cxx_bin")
        else os.environ.get("CXX")
        or "c++"
    )
    pybind_include = REPO_ROOT / "smart/vendor/manifold/bindings/python/third_party/pybind11/include"
    manifold_root = REPO_ROOT / "smart/vendor/manifold"
    manifold_lib_dir = manifold_root / "build/src/manifold"
    default_manifold_lib = manifold_lib_dir / "libmanifold.a"
    manifold_lib = _find_vendored_manifold_lib(manifold_root)
    module_source = REPO_ROOT / "cpp/smart_cpp_module.cpp"
    native_source = REPO_ROOT / "cpp/smart_native_core.cpp"
    engine_source = REPO_ROOT / "cpp/smart_native_engine.cpp"
    bridge_source = REPO_ROOT / "cpp/manifold_bridge.cpp"
    cli_source = REPO_ROOT / "cpp/smart_native_cli.cpp"

    missing = [
        path
        for path in [
            pybind_include / "pybind11/pybind11.h",
            manifold_lib or default_manifold_lib,
            module_source,
            native_source,
            engine_source,
            bridge_source,
            cli_source,
            REPO_ROOT / "cpp/smart_native_engine.hpp",
        ]
        if not path.exists()
    ]
    if missing and not dry_run:
        return [
            "smart._cpp C++ build: failed rc=127",
            "Missing required source/build artifact(s): " + ", ".join(str(path) for path in missing),
            "Run: smart build-tools --only-manifold-binding",
        ]

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    tmp_output = log_root / f"_cpp{ext_suffix}"
    destination = REPO_ROOT / "smart" / f"_cpp{ext_suffix}"

    include_dirs = [
        Path(sysconfig.get_paths()["include"]),
        Path(sysconfig.get_paths().get("platinclude") or sysconfig.get_paths()["include"]),
        pybind_include,
        REPO_ROOT / "cpp",
        manifold_root / "src/manifold/include",
        manifold_root / "src/utilities/include",
        manifold_root / "src/polygon/include",
        manifold_root / "src/collider/include",
        manifold_root / "src/sdf/include",
        manifold_root / "src/third_party/glm",
    ]
    command = [
        compiler,
        *_macos_arm64_compile_flags(),
        *_macos_min_version_flags(),
        "-std=c++17",
        "-fPIC",
        "-shared",
        "-DNDEBUG",
        "-O3" if release else "-O0",
        "-o",
        str(tmp_output),
        str(module_source),
        str(native_source),
        str(bridge_source),
    ]
    for include_dir in include_dirs:
        command.extend(["-I", str(include_dir)])
    command.append(str(manifold_lib or default_manifold_lib))
    command.extend(_manifold_native_link_args(manifold_root))
    if sys.platform == "darwin":
        command.extend(["-undefined", "dynamic_lookup"])
    else:
        command.append("-lstdc++")

    build_result = run_command(
        command,
        cwd=REPO_ROOT,
        timeout=900,
        log_path=log_root / "cpp-build.log",
        dry_run=dry_run,
    )
    messages = [_line("smart._cpp C++ build", build_result.returncode, dry_run)]
    if not build_result.ok and not dry_run:
        messages.append("Check the C++ build log, then rerun: smart build-cpp")
        return messages

    cli_output = REPO_ROOT / "build" / "smart-cpp-native"
    if not dry_run:
        cli_output.parent.mkdir(parents=True, exist_ok=True)
    cli_command = [
        compiler,
        *_macos_arm64_compile_flags(),
        *_macos_min_version_flags(),
        "-std=c++17",
        "-DNDEBUG",
        "-O3" if release else "-O0",
        "-o",
        str(cli_output),
        str(cli_source),
        str(native_source),
        str(engine_source),
        str(bridge_source),
    ]
    for include_dir in include_dirs:
        cli_command.extend(["-I", str(include_dir)])
    cli_command.append(str(manifold_lib or default_manifold_lib))
    cli_command.extend(_manifold_native_link_args(manifold_root))
    if sys.platform != "darwin":
        cli_command.append("-lstdc++")
    cli_result = run_command(
        cli_command,
        cwd=REPO_ROOT,
        timeout=300,
        log_path=log_root / "cpp-native-cli-build.log",
        dry_run=dry_run,
    )
    messages.append(_line("smart-cpp-native executable build", cli_result.returncode, dry_run))
    if not cli_result.ok and not dry_run:
        messages.append("Check the C++ native executable build log, then rerun: smart build-cpp")
        return messages

    asan_output = REPO_ROOT / "build" / "smart-cpp-native-asan"
    if asan:
        asan_command = [
            compiler,
            *_macos_arm64_compile_flags(),
            *_macos_min_version_flags(),
            "-std=c++17",
            "-gline-tables-only",
            "-O0",
            "-fno-omit-frame-pointer",
            "-fsanitize=address",
            "-o",
            str(asan_output),
            str(cli_source),
            str(native_source),
            str(engine_source),
            str(bridge_source),
        ]
        for include_dir in include_dirs:
            asan_command.extend(["-I", str(include_dir)])
        asan_command.append(str(manifold_lib or default_manifold_lib))
        asan_command.extend(_manifold_native_link_args(manifold_root))
        asan_command.append("-fsanitize=address")
        if sys.platform != "darwin":
            asan_command.append("-lstdc++")
        asan_result = run_command(
            asan_command,
            cwd=REPO_ROOT,
            timeout=1200,
            log_path=log_root / "cpp-native-cli-asan-build.log",
            dry_run=dry_run,
        )
        messages.append(_line("smart-cpp-native ASan executable build", asan_result.returncode, dry_run))
        if not asan_result.ok and not dry_run:
            messages.append("Check the C++ ASan native executable build log, then rerun: smart build-cpp --asan")
            return messages

    if dry_run:
        messages.append("smart._cpp local source extension: dry-run")
        messages.append("smart-cpp-native executable: dry-run")
        if asan:
            messages.append("smart-cpp-native ASan executable: dry-run")
        return messages

    for existing in (REPO_ROOT / "smart").glob("_cpp*.so"):
        existing.unlink()
    for existing in (REPO_ROOT / "smart").glob("_cpp*.pyd"):
        existing.unlink()
    shutil.copy2(tmp_output, destination)
    messages.append(f"smart._cpp local source extension: ok -> {destination}")
    messages.append(f"smart-cpp-native executable: ok -> {cli_output}")
    if asan:
        asan_output.chmod(asan_output.stat().st_mode | 0o755)
        messages.append(f"smart-cpp-native ASan executable: ok -> {asan_output}")
    return messages


def _find_vendored_manifold_lib(manifold_root: Path) -> Path | None:
    default = manifold_root / "build" / "src" / "manifold" / "libmanifold.a"
    if default.exists():
        return default
    build_root = manifold_root / "build"
    if not build_root.exists():
        return None
    matches = sorted(build_root.glob("**/libmanifold.a"))
    return matches[0] if matches else None


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
    ] + _macos_cmake_platform_args() + (cmake_args or [])
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
