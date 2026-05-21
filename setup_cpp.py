from __future__ import annotations

import platform
import os
import shutil
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_py import build_py as _build_py


ROOT = Path(__file__).resolve().parent
MANIFOLD_ROOT = ROOT / "smart/vendor/manifold"
VENDORED_PYBIND_INCLUDE = MANIFOLD_ROOT / "bindings/python/third_party/pybind11/include"
MANIFOLD_LIB = MANIFOLD_ROOT / "build/src/manifold/libmanifold.a"


def _is_macos_arm64_host() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _macos_deployment_target() -> str | None:
    if sys.platform != "darwin":
        return None
    return os.environ.get("MACOSX_DEPLOYMENT_TARGET") or "11.0"


def _macos_min_version_flags() -> list[str]:
    target = _macos_deployment_target()
    if target is None:
        return []
    return [f"-mmacosx-version-min={target}"]


def _ensure_macos_build_env() -> None:
    if sys.platform != "darwin":
        return
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", _macos_deployment_target() or "11.0")
    # python.org universal2 installs may default sysconfig CFLAGS to x86_64 even
    # on Apple Silicon. The vendored Manifold build is arm64 on these machines,
    # so the packaged extension/executable must be compiled arm64 too.
    if not _is_macos_arm64_host():
        return
    archflags = os.environ.get("ARCHFLAGS", "")
    if "arm64" not in archflags.split():
        os.environ["ARCHFLAGS"] = "-arch arm64"


_ensure_macos_build_env()


def _native_executable_name() -> str:
    return "smart-cpp-native.exe" if os.name == "nt" else "smart-cpp-native"


def _pybind_include() -> Path:
    try:
        import pybind11  # type: ignore[import-not-found]

        return Path(pybind11.get_include())
    except Exception:
        return VENDORED_PYBIND_INCLUDE


def _first_pymanifold_binary(path: Path) -> Path | None:
    for pattern in ("pymanifold*.so", "pymanifold*.pyd", "pymanifold*.dylib"):
        matches = sorted(path.glob(pattern))
        if matches:
            return matches[0]
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


def _manifold_cmake_cache_backend() -> str:
    cache = MANIFOLD_ROOT / "build" / "CMakeCache.txt"
    if not cache.exists():
        return "NONE"
    for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MANIFOLD_PAR:"):
            return line.rsplit("=", 1)[-1].strip().upper()
    return "NONE"


def _find_manifold_lib() -> Path | None:
    if MANIFOLD_LIB.exists():
        return MANIFOLD_LIB
    build_root = MANIFOLD_ROOT / "build"
    if not build_root.exists():
        return None
    matches = sorted(build_root.glob("**/libmanifold.a"))
    return matches[0] if matches else None


def _manifold_link_args() -> list[str]:
    backend = _manifold_cmake_cache_backend()
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


def _native_thread_link_args() -> list[str]:
    if sys.platform.startswith("linux"):
        return ["-pthread"]
    return []


def _include_dirs() -> list[str]:
    python_paths = sysconfig.get_paths()
    include = python_paths["include"]
    platinclude = python_paths.get("platinclude") or include
    return [
        include,
        platinclude,
        str(_pybind_include()),
        str(ROOT / "cpp"),
        str(MANIFOLD_ROOT / "src/manifold/include"),
        str(MANIFOLD_ROOT / "src/utilities/include"),
        str(MANIFOLD_ROOT / "src/polygon/include"),
        str(MANIFOLD_ROOT / "src/collider/include"),
        str(MANIFOLD_ROOT / "src/sdf/include"),
        str(MANIFOLD_ROOT / "src/third_party/glm"),
    ]


def _extension() -> Extension:
    manifold_lib = _find_manifold_lib()
    pybind_include = _pybind_include()
    missing = [
        path
        for path in [
            pybind_include / "pybind11/pybind11.h",
            manifold_lib or MANIFOLD_LIB,
            ROOT / "cpp/smart_cpp_module.cpp",
            ROOT / "cpp/smart_native_core.cpp",
            ROOT / "cpp/smart_native_engine.cpp",
            ROOT / "cpp/manifold_bridge.cpp",
        ]
        if not path.exists()
    ]
    if missing:
        joined = "\n  - ".join(str(path) for path in missing)
        raise RuntimeError(
            "Missing C++ package build input(s):\n  - "
            + joined
            + "\nRun: python -m smart --config configs/smoke_5.yaml build-tools --only-manifold-binding"
        )

    compile_args = ["-std=c++17", "-O3", "-DNDEBUG"]
    link_args: list[str] = _manifold_link_args()
    link_args.extend(_native_thread_link_args())
    compile_args.extend(_macos_min_version_flags())
    link_args.extend(_macos_min_version_flags())
    if _is_macos_arm64_host():
        compile_args.extend(["-arch", "arm64"])
        link_args.extend(["-arch", "arm64"])
    if sys.platform == "darwin":
        link_args.extend(["-undefined", "dynamic_lookup"])

    return Extension(
        "smart._cpp",
        sources=[
            "cpp/smart_cpp_module.cpp",
            "cpp/smart_native_core.cpp",
            "cpp/smart_native_engine.cpp",
            "cpp/manifold_bridge.cpp",
        ],
        include_dirs=_include_dirs(),
        extra_compile_args=compile_args,
        extra_objects=[str(manifold_lib or MANIFOLD_LIB)],
        extra_link_args=link_args,
        language="c++",
    )


def _should_build_extension() -> bool:
    # PEP 517/660 metadata hooks run setup.py before any project command has
    # built the vendored Manifold static library. Keep those metadata/editable
    # probes pure-Python; the native extension is attached only to real binary
    # build commands.
    build_commands = {
        "bdist_egg",
        "bdist_wheel",
        "build",
        "build_ext",
        "install",
    }
    return any(arg in build_commands for arg in sys.argv[1:])


ENTRY_POINTS = {
    "console_scripts": [
        "smart=smart.cli:main",
        "smart-cpp-native=smart.native_executable:main",
        "smart-audit-wheel=scripts.audit_release_wheel:main",
        "smart-smoke-console-scripts=scripts.smoke_console_scripts:main",
        "smart-release-preflight=scripts.release_preflight:main",
        "smart-quickstart=scripts.quickstart_reproduce:main",
    ],
}


SETUP_OPTIONS = {}
if sys.platform == "darwin" and platform.machine() == "arm64":
    SETUP_OPTIONS["bdist_wheel"] = {"plat_name": "macosx_11_0_arm64"}


class BuildPyWithoutSourceArtifacts(_build_py):
    def run(self) -> None:
        super().run()
        self._build_native_executable()
        self._stage_pymanifold_runtime()
        build_lib = Path(self.build_lib)
        for relative in [
            "smart/vendor",
            "past_codes",
            "data",
            "runs",
            "external",
            "experiments",
        ]:
            shutil.rmtree(build_lib / relative, ignore_errors=True)
        for relative in [
            "smart/rust.py",
            "scripts/benchmark_rust_parity.py",
        ]:
            (build_lib / relative).unlink(missing_ok=True)
        for pattern in ["smart/_rust*.so", "smart/_rust*.pyd", "smart/_rust*.dylib"]:
            for path in build_lib.glob(pattern):
                path.unlink(missing_ok=True)

    def _build_native_executable(self) -> None:
        cli_source = ROOT / "cpp/smart_native_cli.cpp"
        native_source = ROOT / "cpp/smart_native_core.cpp"
        engine_source = ROOT / "cpp/smart_native_engine.cpp"
        bridge_source = ROOT / "cpp/manifold_bridge.cpp"
        manifold_lib = _find_manifold_lib()
        missing = [
            path
            for path in [
                cli_source,
                native_source,
                engine_source,
                bridge_source,
                ROOT / "cpp/smart_native_core.hpp",
                ROOT / "cpp/smart_native_engine.hpp",
                manifold_lib or MANIFOLD_LIB,
            ]
            if not path.exists()
        ]
        if missing:
            joined = "\n  - ".join(str(path) for path in missing)
            raise RuntimeError("Missing C++ native executable source(s):\n  - " + joined)

        output_dir = Path(self.build_lib) / "smart" / "bin"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / _native_executable_name()
        compiler = os.environ.get("CXX") or "c++"
        command = [
            compiler,
            "-std=c++17",
            "-DNDEBUG",
            "-O3",
            "-o",
            str(output),
            str(cli_source),
            str(native_source),
            str(engine_source),
            str(bridge_source),
        ]
        if _is_macos_arm64_host():
            command[1:1] = ["-arch", "arm64"]
        command[1:1] = _macos_min_version_flags()
        for include_dir in _include_dirs():
            command.extend(["-I", include_dir])
        command.append(str(manifold_lib or MANIFOLD_LIB))
        command.extend(_manifold_link_args())
        command.extend(_native_thread_link_args())
        if sys.platform != "darwin":
            command.append("-lstdc++")
        self.spawn(command)
        output.chmod(output.stat().st_mode | 0o755)

    def _stage_pymanifold_runtime(self) -> None:
        output_dir = Path(self.build_lib) / "smart" / "pymanifold_runtime"
        if os.environ.get("SMART_PACKAGE_PYMANIFOLD", "0") != "1":
            shutil.rmtree(output_dir, ignore_errors=True)
            return
        candidates = [
            ROOT / "smart" / "pymanifold_runtime",
            MANIFOLD_ROOT / "build" / "bindings" / "python",
        ]
        binary = next((found for candidate in candidates if (found := _first_pymanifold_binary(candidate))), None)
        if binary is None:
            shutil.rmtree(output_dir, ignore_errors=True)
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ("pymanifold*.so", "pymanifold*.pyd", "pymanifold*.dylib"):
            for existing in output_dir.glob(pattern):
                existing.unlink()
        shutil.copy2(binary, output_dir / binary.name)


setup(
    name="smart-bbox",
    version="0.1.11",
    description="Official SMART pipeline for tight 3D bounding boxes",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Chanhyeok Park",
    python_requires=">=3.9",
    packages=find_packages(include=["smart", "smart.*", "scripts"]),
    py_modules=["pymesh"],
    include_package_data=False,
    package_data={
        "smart": [
            "py.typed",
            "configs/*.yaml",
            "assets/priors/*.json",
            "assets/gates/*.json",
            "legacy/renderer/*.blend",
            "legacy/renderer/*.txt",
            "bin/smart-cpp-native",
            "bin/smart-cpp-native.exe",
        ],
    },
    ext_modules=[_extension()] if _should_build_extension() else [],
    cmdclass={"build_py": BuildPyWithoutSourceArtifacts},
    entry_points=ENTRY_POINTS,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    options=SETUP_OPTIONS,
)
