from __future__ import annotations

import argparse
import ast
import fnmatch
import glob
import json
import re
import shutil
import sys
import tarfile
import tempfile
import subprocess
import zipfile
from pathlib import Path
from typing import Iterable


REQUIRED_PATTERNS = {
    "cpp_extension": ("smart/_cpp*.so", "smart/_cpp*.pyd", "smart/_cpp*.dylib"),
    "pymanifold_runtime": (
        "smart/pymanifold_runtime/pymanifold*.so",
        "smart/pymanifold_runtime/pymanifold*.pyd",
        "smart/pymanifold_runtime/pymanifold*.dylib",
    ),
    "native_executable": ("smart/bin/smart-cpp-native", "smart/bin/smart-cpp-native.exe"),
    "smart_package": ("smart/__init__.py",),
    "smart_cli": ("smart/cli.py",),
    "native_executable_helper": ("smart/native_executable.py",),
    "pymesh_compat": ("smart/pymesh_compat.py",),
    "legacy_pymesh_alias": ("pymesh.py",),
    "demo_config": ("smart/configs/demo.yaml",),
    "smoke_config": ("smart/configs/smoke_5.yaml",),
    "example_config": ("smart/configs/example_3x3.yaml",),
    "renderer_blend": ("smart/legacy/renderer/boxes.blend",),
    "renderer_colors": ("smart/legacy/renderer/semantic_colors.txt",),
    "entry_points": ("*.dist-info/entry_points.txt",),
}

WHEEL_FORBIDDEN_PATTERNS = {
    "vendored_manifold_source": ("smart/vendor/manifold/*",),
    "past_codes": ("past_codes/*",),
    "data": ("data/*",),
    "runs": ("runs/*",),
    "external": ("external/*",),
    "experiments": ("experiments/*",),
    "legacy_rust_module": ("smart/rust.py", "smart/_rust*.so", "smart/_rust*.pyd", "smart/_rust*.dylib"),
    "legacy_rust_scripts": ("scripts/benchmark_rust_parity.py",),
    "cargo_workspace": ("Cargo.toml", "Cargo.lock", "rust/*"),
}

SDIST_REQUIRED_PATTERNS = {
    "pyproject": ("pyproject.toml",),
    "cpp_setuptools_entrypoint": ("setup.py",),
    "cpp_setuptools_build": ("setup_cpp.py",),
    "readme": ("README.md",),
    "citation": ("CITATION.cff",),
    "smart_package": ("smart/__init__.py",),
    "smart_cli": ("smart/cli.py",),
    "pipeline_tools": ("smart/pipeline/tools.py",),
    "cpp_extension_source": ("cpp/smart_cpp_module.cpp",),
    "native_core_cpp": ("cpp/smart_native_core.cpp",),
    "native_core_header": ("cpp/smart_native_core.hpp",),
    "native_engine_cpp": ("cpp/smart_native_engine.cpp",),
    "native_engine_header": ("cpp/smart_native_engine.hpp",),
    "native_cli_cpp": ("cpp/smart_native_cli.cpp",),
    "manifold_bridge_cpp": ("cpp/manifold_bridge.cpp",),
    "audit_script": ("scripts/audit_release_wheel.py",),
    "vendored_manifold_source": ("smart/vendor/manifold/src/*",),
    "vendored_manifold_python_binding": ("smart/vendor/manifold/bindings/python/CMakeLists.txt",),
}

SDIST_FORBIDDEN_PATTERNS = {
    "past_codes": ("past_codes/*",),
    "data": ("data/*",),
    "runs": ("runs/*",),
    "external": ("external/*",),
    "experiments": ("experiments/*",),
    "vendored_manifold_build": ("smart/vendor/manifold/build/*",),
    "legacy_rust_module": (
        "smart/rust.py",
        "smart/_rust*.so",
        "smart/_rust*.pyd",
        "smart/_rust*.dylib",
    ),
    "legacy_rust_scripts": ("scripts/benchmark_rust_parity.py",),
    "cargo_workspace": ("Cargo.toml", "Cargo.lock", "rust/*"),
    "compiled_cpp_extension": ("smart/_cpp*.so", "smart/_cpp*.pyd", "smart/_cpp*.dylib"),
    "compiled_pymanifold_runtime": (
        "smart/pymanifold_runtime/pymanifold*.so",
        "smart/pymanifold_runtime/pymanifold*.pyd",
        "smart/pymanifold_runtime/pymanifold*.dylib",
    ),
}

REQUIRED_ENTRY_POINTS = {
    "smart": "smart.cli:main",
    "smart-cpp-native": "smart.native_executable:main",
    "smart-audit-wheel": "scripts.audit_release_wheel:main",
    "smart-smoke-console-scripts": "scripts.smoke_console_scripts:main",
    "smart-release-preflight": "scripts.release_preflight:main",
    "smart-quickstart": "scripts.quickstart_reproduce:main",
}


def _matches(names: Iterable[str], patterns: Iterable[str]) -> list[str]:
    return sorted(
        name
        for name in names
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
    )


def _entry_point_text(names: set[str], wheel: zipfile.ZipFile) -> str:
    entry_points = _matches(names, REQUIRED_PATTERNS["entry_points"])
    return "\n".join(wheel.read(name).decode("utf-8", errors="replace") for name in entry_points)


def _parse_console_entry_points(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    in_console_scripts = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_console_scripts = line == "[console_scripts]"
            continue
        if not in_console_scripts or "=" not in line:
            continue
        name, target = line.split("=", 1)
        entries[name.strip()] = target.strip()
    return entries


def _module_path(module_name: str) -> str:
    return module_name.replace(".", "/") + ".py"


def _target_module(entry_point_target: str) -> str:
    return entry_point_target.split(":", 1)[0].strip()


def _script_import_modules(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("scripts."):
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names if alias.name.startswith("scripts."))
    return sorted(modules)


def _validate_asset_jsons(
    names: Iterable[str],
    *,
    read_text,
) -> tuple[list[str], dict[str, object]]:
    errors: list[str] = []
    asset_names = sorted(
        name
        for name in names
        if fnmatch.fnmatchcase(name, "smart/assets/gates/*.json")
        or fnmatch.fnmatchcase(name, "smart/assets/priors/*.json")
    )
    checks: dict[str, object] = {"asset_json_count": len(asset_names), "asset_json_files": asset_names}
    for name in asset_names:
        try:
            payload = json.loads(read_text(name))
        except Exception as exc:
            errors.append(f"asset JSON is not parseable: {name}: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"asset JSON must be an object: {name}")
            continue
        metadata = payload.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            errors.append(f"asset JSON metadata must be an object when present: {name}")
        feature_names = payload.get("feature_names")
        if feature_names is not None and not isinstance(feature_names, list):
            errors.append(f"asset JSON feature_names must be a list when present: {name}")
        policy_type = str(payload.get("policy_type") or payload.get("type") or "")
        if fnmatch.fnmatchcase(name, "smart/assets/gates/*.json"):
            if not policy_type:
                errors.append(f"gate asset JSON missing policy_type: {name}")
            if "weights" not in payload:
                errors.append(f"gate asset JSON missing weights: {name}")
        elif name.endswith("smoke5_coord_scale_prior.json"):
            if "coord_scale_logits" not in payload:
                errors.append(f"legacy prior asset JSON missing coord_scale_logits: {name}")
        elif fnmatch.fnmatchcase(name, "smart/assets/priors/*.json") and not policy_type:
            errors.append(f"prior asset JSON missing policy_type: {name}")
    return errors, checks


def _target_arch_from_wheel_name(path: Path) -> str | None:
    name = path.name
    if "macosx" in name and "arm64" in name:
        return "arm64"
    if "macosx" in name and "x86_64" in name:
        return "x86_64"
    if "manylinux" in name and "x86_64" in name:
        return "x86_64"
    return None


def _macos_platform_tags_from_wheel_name(path: Path) -> list[tuple[int, int, str]]:
    return [
        (int(major), int(minor), arch)
        for major, minor, arch in re.findall(r"macosx_(\d+)_(\d+)_([^-.]+)", path.name)
    ]


def _validate_platform_tag(path: Path, errors: list[str], checks: dict[str, object]) -> None:
    tags = _macos_platform_tags_from_wheel_name(path)
    checks["macos_platform_tags"] = [
        {"major": major, "minor": minor, "arch": arch}
        for major, minor, arch in tags
    ]
    for major, minor, arch in tags:
        if arch == "arm64" and (major, minor) < (11, 0):
            errors.append(
                "macOS arm64 wheels must use deployment tag macosx_11_0_arm64 "
                f"or newer, got macosx_{major}_{minor}_{arch}"
            )


def _file_output_matches_arch(output: str, arch: str) -> bool:
    lowered = output.lower()
    if arch == "arm64":
        return "arm64" in lowered or "aarch64" in lowered
    if arch == "x86_64":
        return "x86_64" in lowered or "x86-64" in lowered
    return True


def _validate_native_binary_arches(
    *,
    wheel_path: Path,
    wheel: zipfile.ZipFile,
    names: set[str],
    errors: list[str],
    checks: dict[str, object],
) -> None:
    target_arch = _target_arch_from_wheel_name(wheel_path)
    checks["binary_arch_target"] = target_arch
    if target_arch is None:
        return
    file_tool = shutil.which("file")
    checks["binary_arch_file_tool"] = file_tool
    if file_tool is None:
        return

    native_patterns = (
        "smart/_cpp*.so",
        "smart/_cpp*.pyd",
        "smart/_cpp*.dylib",
        "smart/bin/smart-cpp-native",
        "smart/bin/smart-cpp-native.exe",
        "smart/pymanifold_runtime/pymanifold*.so",
        "smart/pymanifold_runtime/pymanifold*.pyd",
        "smart/pymanifold_runtime/pymanifold*.dylib",
    )
    members = _matches(names, native_patterns)
    checks["binary_arch_members"] = members
    arch_outputs: dict[str, str] = {}
    skipped: list[str] = []
    with tempfile.TemporaryDirectory(prefix="smart-wheel-arch-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for member in members:
            info = wheel.getinfo(member)
            # Unit-test wheels use tiny placeholder strings. Real native
            # artifacts are much larger, so skip the placeholders while still
            # checking release artifacts by default.
            if info.file_size < 1024:
                skipped.append(member)
                continue
            tmp_path = tmp_root / Path(member).name
            tmp_path.write_bytes(wheel.read(member))
            try:
                completed = subprocess.run(
                    [file_tool, str(tmp_path)],
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
            except OSError as exc:
                errors.append(f"could not inspect native binary architecture for {member}: {exc}")
                continue
            output = completed.stdout.strip()
            arch_outputs[member] = output
            if completed.returncode != 0:
                errors.append(f"file failed while inspecting {member}: {output}")
                continue
            if not _file_output_matches_arch(output, target_arch):
                errors.append(
                    f"native binary architecture mismatch for {member}: "
                    f"wheel tag expects {target_arch}, file reports {output}"
                )
    checks["binary_arch_outputs"] = arch_outputs
    checks["binary_arch_skipped_small_placeholders"] = skipped


def audit_wheel(path: Path, *, cpp_only: bool = False) -> dict[str, object]:
    errors: list[str] = []
    checks: dict[str, object] = {}

    if not path.exists():
        return {
            "wheel": str(path),
            "ok": False,
            "errors": [f"wheel does not exist: {path}"],
            "checks": checks,
        }

    try:
        with zipfile.ZipFile(path) as wheel:
            names = set(wheel.namelist())
            checks["file_count"] = len(names)
            _validate_platform_tag(path, errors, checks)

            for label, patterns in REQUIRED_PATTERNS.items():
                matched = _matches(names, patterns)
                checks[label] = matched
                if not matched:
                    errors.append(f"missing required artifact group {label}: {', '.join(patterns)}")

            for label, patterns in WHEEL_FORBIDDEN_PATTERNS.items():
                matched = _matches(names, patterns)
                checks[label] = matched
                if matched:
                    errors.append(
                        f"forbidden artifact group {label} present: {matched[:5]}"
                        + (" ..." if len(matched) > 5 else "")
                    )

            entry_text = _entry_point_text(names, wheel)
            checks["entry_point_text_present"] = bool(entry_text)
            console_entries = _parse_console_entry_points(entry_text)
            checks["console_entry_points"] = console_entries
            for name, target in REQUIRED_ENTRY_POINTS.items():
                found = console_entries.get(name) == target
                checks[f"entry_point:{name}"] = found
                if not found:
                    errors.append(f"missing console entry point: {name} = {target}")

            checked_modules: dict[str, str] = {}
            for name, target in console_entries.items():
                module_name = _target_module(target)
                module_path = _module_path(module_name)
                checked_modules[name] = module_path
                if module_path not in names:
                    errors.append(f"console entry point target module missing: {name} -> {module_path}")
            checks["console_entry_point_modules"] = checked_modules

            script_imports: dict[str, list[str]] = {}
            for module_path in sorted(set(checked_modules.values())):
                if not module_path.startswith("scripts/") or module_path not in names:
                    continue
                source = wheel.read(module_path).decode("utf-8", errors="replace")
                imports = _script_import_modules(source)
                script_imports[module_path] = imports
                for module_name in imports:
                    import_path = _module_path(module_name)
                    if import_path not in names:
                        errors.append(f"packaged script import missing: {module_path} imports {import_path}")
            checks["script_imports"] = script_imports

            asset_errors, asset_checks = _validate_asset_jsons(
                names,
                read_text=lambda name: wheel.read(name).decode("utf-8", errors="replace"),
            )
            errors.extend(asset_errors)
            checks.update(asset_checks)
            _validate_native_binary_arches(
                wheel_path=path,
                wheel=wheel,
                names=names,
                errors=errors,
                checks=checks,
            )
    except zipfile.BadZipFile:
        errors.append(f"not a valid wheel/zip file: {path}")

    return {
        "artifact_type": "wheel",
        "wheel": str(path),
        "ok": not errors,
        "errors": errors,
        "checks": checks,
    }


def _strip_sdist_root(names: Iterable[str]) -> set[str]:
    stripped: set[str] = set()
    for name in names:
        parts = name.split("/", 1)
        stripped.add(parts[1] if len(parts) == 2 else name)
    return stripped


def audit_sdist(path: Path) -> dict[str, object]:
    errors: list[str] = []
    checks: dict[str, object] = {}

    if not path.exists():
        return {
            "artifact_type": "sdist",
            "wheel": str(path),
            "ok": False,
            "errors": [f"sdist does not exist: {path}"],
            "checks": checks,
        }

    try:
        with tarfile.open(path, "r:gz") as archive:
            archive_names = archive.getnames()
            names = _strip_sdist_root(archive_names)
            name_map = {}
            for archive_name in archive_names:
                parts = archive_name.split("/", 1)
                stripped = parts[1] if len(parts) == 2 else archive_name
                name_map[stripped] = archive_name
            checks["file_count"] = len(names)

            for label, patterns in SDIST_REQUIRED_PATTERNS.items():
                matched = _matches(names, patterns)
                checks[label] = matched
                if not matched:
                    errors.append(f"missing required sdist artifact group {label}: {', '.join(patterns)}")

            for label, patterns in SDIST_FORBIDDEN_PATTERNS.items():
                matched = _matches(names, patterns)
                checks[label] = matched
                if matched:
                    errors.append(
                        f"forbidden sdist artifact group {label} present: {matched[:5]}"
                        + (" ..." if len(matched) > 5 else "")
                    )
            asset_errors, asset_checks = _validate_asset_jsons(
                names,
                read_text=lambda name: archive.extractfile(name_map[name]).read().decode("utf-8", errors="replace"),
            )
            errors.extend(asset_errors)
            checks.update(asset_checks)
    except tarfile.TarError:
        errors.append(f"not a valid gzipped sdist/tar file: {path}")

    return {
        "artifact_type": "sdist",
        "wheel": str(path),
        "ok": not errors,
        "errors": errors,
        "checks": checks,
    }


def audit_artifact(path: Path, *, cpp_only: bool = False) -> dict[str, object]:
    if path.name.endswith(".whl"):
        return audit_wheel(path, cpp_only=cpp_only)
    if path.name.endswith((".tar.gz", ".tgz")):
        return audit_sdist(path)
    return {
        "artifact_type": "unknown",
        "wheel": str(path),
        "ok": False,
        "errors": [f"unsupported release artifact type: {path}"],
        "checks": {},
    }


def _expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in inputs:
        if any(char in value for char in "*?["):
            expanded = [Path(path) for path in sorted(glob.glob(value))]
        else:
            expanded = [Path(value)]
        paths.extend(expanded)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit SMART release wheel/sdist contents.")
    parser.add_argument("wheels", nargs="+", help="Wheel/sdist paths or glob patterns")
    parser.add_argument("--json", action="store_true", help="Write machine-readable audit results")
    parser.add_argument(
        "--cpp-only",
        action="store_true",
        help="Compatibility no-op; native C++ wheels are the default release artifact",
    )
    args = parser.parse_args(argv)

    wheel_paths = _expand_inputs(args.wheels)
    if not wheel_paths:
        print("no wheels matched", file=sys.stderr)
        return 2

    results = [audit_artifact(path, cpp_only=args.cpp_only) for path in wheel_paths]
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for result in results:
            status = "ok" if result["ok"] else "failed"
            print(f"{result['wheel']}: {status}")
            for error in result["errors"]:
                print(f"  - {error}")

    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
