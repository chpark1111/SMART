from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path

import yaml

from smart.cli import main as smart_main
from scripts.audit_release_wheel import (
    REQUIRED_ENTRY_POINTS,
    audit_artifact,
    audit_sdist,
    audit_wheel,
    main,
)
from scripts.smoke_console_scripts import CONSOLE_COMMANDS, FUNCTIONAL_SMOKE_COMMANDS, smoke_console_scripts


def _pyproject_scripts() -> dict[str, str]:
    scripts: dict[str, str] = {}
    in_scripts = False
    for raw_line in Path("pyproject.toml").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "[project.scripts]":
            in_scripts = True
            continue
        if in_scripts and line.startswith("["):
            break
        if not in_scripts or "=" not in line:
            continue
        key, value = line.split("=", 1)
        scripts[key.strip()] = value.strip().strip('"')
    return scripts


def _write_wheel(path: Path, names: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as wheel:
        for name in names:
            content = _content_for_name(name)
            if name.endswith("entry_points.txt"):
                content = (
                    "[console_scripts]\n"
                    + "\n".join(f"{key} = {value}" for key, value in REQUIRED_ENTRY_POINTS.items())
                    + "\n"
                )
            wheel.writestr(name, content)


def _minimal_release_names() -> list[str]:
    names = [
        "smart/__init__.py",
        "scripts/__init__.py",
        "smart/cli.py",
        "smart/_cpp.cpython-39-darwin.so",
        "smart/bin/smart-cpp-native",
        "smart/native_executable.py",
        "smart/pymesh_compat.py",
        "smart/configs/demo.yaml",
        "smart/configs/smoke_5.yaml",
        "smart/configs/example_3x3.yaml",
        "smart/configs/learned_frontier.yaml",
        "smart/configs/learned_auto_safe.yaml",
        "smart/configs/learned_macro_safe.yaml",
        "smart/configs/learned_macro_program_gate_top3.yaml",
        "smart/configs/learned_macro_refine_only.yaml",
        "smart/legacy/renderer/boxes.blend",
        "smart/legacy/renderer/semantic_colors.txt",
        "pymesh.py",
        "smart_bbox-0.1.0.dist-info/entry_points.txt",
    ]
    for target in REQUIRED_ENTRY_POINTS.values():
        module = target.split(":", 1)[0]
        module_path = module.replace(".", "/") + ".py"
        if module_path not in names:
            names.append(module_path)
    return names


def _write_sdist(path: Path, names: list[str]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name in names:
            source = path.parent / name.replace("/", "__")
            source.write_text(_content_for_name(name), encoding="utf-8")
            archive.add(source, arcname=f"smart_bbox-0.1.0/{name}")


def _content_for_name(name: str) -> str:
    if name.startswith("smart/assets/gates/") and name.endswith(".json"):
        return (
            '{"policy_type":"local_refine_gate","schema_version":1,'
            '"feature_names":["x"],"weights":{"layers":[]},"metadata":{"model_type":"pytorch_mlp_gate"}}'
        )
    if name.endswith("category_general_candidate_pg_agent_rich_v2.json"):
        return (
            '{"policy_type":"action_mlp_prior","schema_version":1,'
            '"feature_names":["x"],"metadata":{"model_type":"policy_gradient_agent"}}'
        )
    if name.startswith("smart/assets/priors/") and name.endswith(".json"):
        return '{"policy_type":"coord_scale_mlp_prior","schema_version":1,"metadata":{"model_type":"test"}}'
    return "placeholder"


def _minimal_sdist_names() -> list[str]:
    return [
        "pyproject.toml",
        "setup.py",
        "setup_cpp.py",
        "README.md",
        "CITATION.cff",
        "smart/__init__.py",
        "smart/cli.py",
        "smart/pipeline/tools.py",
        "cpp/smart_cpp_module.cpp",
        "cpp/smart_native_core.cpp",
        "cpp/smart_native_core.hpp",
        "cpp/smart_native_engine.cpp",
        "cpp/smart_native_engine.hpp",
        "cpp/smart_native_cli.cpp",
        "cpp/manifold_bridge.cpp",
        "scripts/audit_release_wheel.py",
        "smart/vendor/manifold/src/manifold.cpp",
        "smart/vendor/manifold/bindings/python/CMakeLists.txt",
    ]


def test_audit_release_wheel_accepts_required_runtime_artifacts(tmp_path: Path) -> None:
    wheel = tmp_path / "smart_bbox-0.1.0-cp39-cp39-macosx_11_0_arm64.whl"
    _write_wheel(wheel, _minimal_release_names())

    result = audit_wheel(wheel)

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["checks"]["entry_point:smart"] is True
    assert result["checks"]["entry_point:smart-audit-wheel"] is True
    assert result["checks"]["asset_json_count"] == 0
    assert audit_artifact(wheel)["ok"] is True


def test_audit_release_wheel_cpp_only_flag_is_compatibility_noop(tmp_path: Path) -> None:
    wheel = tmp_path / "smart_bbox-0.1.0-cp39-cp39-macosx_11_0_arm64.whl"
    _write_wheel(wheel, _minimal_release_names())

    strict_result = audit_wheel(wheel)
    cpp_only_result = audit_wheel(wheel, cpp_only=True)

    assert strict_result["ok"] is True
    assert cpp_only_result["ok"] is True
    assert cpp_only_result["errors"] == []


def test_smoke_console_script_list_matches_entry_points() -> None:
    assert CONSOLE_COMMANDS == list(REQUIRED_ENTRY_POINTS)


def test_smoke_console_script_includes_packaged_asset_checks() -> None:
    labels = [str(item["label"]) for item in FUNCTIONAL_SMOKE_COMMANDS]
    required_names = {str(item["required_name"]) for item in FUNCTIONAL_SMOKE_COMMANDS}

    assert "smart configs --json" in labels
    assert "smoke_5.yaml" in required_names


def test_smoke_console_script_reports_missing_entry_points(tmp_path: Path) -> None:
    results = smoke_console_scripts(bin_dir=tmp_path, timeout_sec=1)

    assert len(results) == len(CONSOLE_COMMANDS) + len(FUNCTIONAL_SMOKE_COMMANDS)
    assert all(int(result["returncode"]) != 0 for result in results)
    assert all("stdout" not in result for result in results)


def test_audit_required_entry_points_match_pyproject_scripts() -> None:
    assert REQUIRED_ENTRY_POINTS == _pyproject_scripts()


def test_audit_release_wheel_rejects_vendored_source(tmp_path: Path) -> None:
    wheel = tmp_path / "smart_bbox-0.1.0-cp39-cp39-macosx_11_0_arm64.whl"
    names = _minimal_release_names()
    names.append("smart/vendor/manifold/src/manifold.cpp")
    _write_wheel(wheel, names)

    result = audit_wheel(wheel)

    assert result["ok"] is False
    assert any("vendored_manifold_source" in error for error in result["errors"])


def test_audit_release_wheel_rejects_invalid_asset_json(tmp_path: Path) -> None:
    wheel = tmp_path / "smart_bbox-0.1.0-cp39-cp39-macosx_11_0_arm64.whl"
    names = _minimal_release_names()
    _write_wheel(wheel, names)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("smart/assets/priors/experimental_prior.json", "{not-json")

    result = audit_wheel(wheel)

    assert result["ok"] is False
    assert any("asset JSON is not parseable" in error for error in result["errors"])


def test_audit_release_wheel_rejects_wrong_native_binary_arch(tmp_path: Path) -> None:
    if shutil.which("file") is None:
        return
    wheel = tmp_path / "smart_bbox-0.1.0-cp39-cp39-macosx_11_0_arm64.whl"
    _write_wheel(wheel, _minimal_release_names())
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("smart/_cpp.cpython-39-darwin.so", b"x86_64 placeholder\n" * 200)

    result = audit_wheel(wheel)

    assert result["ok"] is False
    assert any("architecture mismatch" in error for error in result["errors"])


def test_audit_release_wheel_rejects_old_macos_arm64_tag(tmp_path: Path) -> None:
    wheel = tmp_path / "smart_bbox-0.1.0-cp39-cp39-macosx_10_9_arm64.whl"
    _write_wheel(wheel, _minimal_release_names())

    result = audit_wheel(wheel)

    assert result["ok"] is False
    assert any("macosx_11_0_arm64" in error for error in result["errors"])


def test_audit_release_sdist_accepts_reproducible_source_artifacts(tmp_path: Path) -> None:
    sdist = tmp_path / "smart_bbox-0.1.0.tar.gz"
    _write_sdist(sdist, _minimal_sdist_names())

    result = audit_sdist(sdist)

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["checks"]["vendored_manifold_source"]
    assert result["checks"]["asset_json_count"] == 0
    assert audit_artifact(sdist)["ok"] is True


def test_audit_release_sdist_rejects_build_outputs_and_missing_vendor_source(tmp_path: Path) -> None:
    sdist = tmp_path / "smart_bbox-0.1.0.tar.gz"
    names = [
        name
        for name in _minimal_sdist_names()
        if not name.startswith("smart/vendor/manifold/src/")
    ]
    names.append("runs/demo/manifest.jsonl")
    names.append("smart/vendor/manifold/build/bindings/python/pymanifold.so")
    _write_sdist(sdist, names)

    result = audit_sdist(sdist)

    assert result["ok"] is False
    assert any("vendored_manifold_source" in error for error in result["errors"])
    assert any("runs" in error for error in result["errors"])
    assert any("vendored_manifold_build" in error for error in result["errors"])


def test_audit_release_sdist_rejects_missing_citation_file(tmp_path: Path) -> None:
    sdist = tmp_path / "smart_bbox-0.1.0.tar.gz"
    names = [name for name in _minimal_sdist_names() if name != "CITATION.cff"]
    _write_sdist(sdist, names)

    result = audit_sdist(sdist)

    assert result["ok"] is False
    assert any("citation" in error for error in result["errors"])


def test_audit_release_wheel_cli_exit_codes(tmp_path: Path, capsys) -> None:
    wheel = tmp_path / "smart_bbox-0.1.0-cp39-cp39-macosx_11_0_arm64.whl"
    _write_wheel(wheel, _minimal_release_names())

    assert main([str(wheel)]) == 0
    captured = capsys.readouterr()
    assert f"{wheel}: ok" in captured.out

    missing = tmp_path / "missing.whl"
    assert main([str(missing)]) == 1


def test_smart_cli_exposes_wheel_audit(tmp_path: Path, capsys) -> None:
    wheel = tmp_path / "smart_bbox-0.1.0-cp39-cp39-macosx_11_0_arm64.whl"
    _write_wheel(wheel, _minimal_release_names())

    assert smart_main(["audit-wheel", str(wheel)]) == 0
    captured = capsys.readouterr()
    assert f"{wheel}: ok" in captured.out

    native_wheel = tmp_path / "smart_bbox-0.1.0-cp39-cp39-macosx_11_0_arm64_native.whl"
    _write_wheel(native_wheel, _minimal_release_names())
    assert smart_main(["audit-wheel", "--cpp-only", str(native_wheel)]) == 0


def test_release_workflow_audits_artifacts_before_pypi_publish() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/wheels.yml").read_text(encoding="utf-8"))
    wheel_steps = workflow["jobs"]["build-wheel"]["steps"]
    publish_steps = workflow["jobs"]["publish-pypi"]["steps"]

    wheel_step_names = [step.get("name", "") for step in wheel_steps]
    assert "Build wheels" in wheel_step_names
    assert "Audit wheel contents" in wheel_step_names
    assert "Smoke installed console scripts" in wheel_step_names
    build_step = wheel_steps[wheel_step_names.index("Build wheels")]
    audit_step = wheel_steps[wheel_step_names.index("Audit wheel contents")]
    smoke_step = wheel_steps[wheel_step_names.index("Smoke installed console scripts")]
    assert build_step["uses"].startswith("pypa/cibuildwheel")
    assert build_step["env"]["MACOSX_DEPLOYMENT_TARGET"] == "11.0"
    assert audit_step["run"] == "python scripts/audit_release_wheel.py wheelhouse/*.whl"
    assert 'python -m pip install PyYAML "numpy<2.4"' in smoke_step["run"]
    assert "--find-links wheelhouse smart-bbox" in smoke_step["run"]
    assert "wheelhouse/*.whl" not in smoke_step["run"]
    assert "smart.native as sn" in Path("pyproject.toml").read_text(encoding="utf-8")
    assert "smart-smoke-console-scripts" in smoke_step["run"]

    step_names = [step.get("name", "") for step in publish_steps]
    assert "Check release metadata" in step_names
    assert "Audit release artifacts" in step_names
    metadata_index = step_names.index("Check release metadata")
    audit_index = step_names.index("Audit release artifacts")
    publish_index = next(
        index
        for index, step in enumerate(publish_steps)
        if step.get("uses", "").startswith("pypa/gh-action-pypi-publish")
    )
    assert metadata_index < audit_index
    assert audit_index < publish_index
    assert "python -m twine check dist/*" in publish_steps[metadata_index]["run"]
    assert publish_steps[audit_index]["run"] == "python scripts/audit_release_wheel.py dist/*"
    assert "build-cpp-only-wheel" not in workflow["jobs"]["publish-pypi"]["needs"]
    assert "build-cpp-only-wheel" not in workflow["jobs"]


def test_pyproject_classifiers_match_release_python_matrix() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '"Programming Language :: Python :: 3.9"' in text
    assert '"Programming Language :: Python :: 3.10"' in text
    assert '"Programming Language :: Python :: 3.11"' in text


def test_cibuildwheel_runtime_import_test_uses_packaged_config_and_numpy() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'test-requires = ["PyYAML", "numpy<2.4"]' in text
    assert "diagnose_environment(load_config('smoke_5.yaml'))" in text
    assert "smart.native as sn" in text
    assert "assert sn.using_cpp()" in text
    assert "smart.cpp_native_available()" in text
    assert "smart.NativeSmartEngine is sc.NativeSmartEngine" in text
    assert "asset_path(" not in text
    assert "smart.doctor(" not in text
    assert "smart.doctor('configs/smoke_5.yaml')" not in text


def test_ci_native_builds_pin_macos_deployment_target() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            run = step.get("run", "")
            if "build-tools --only-manifold-binding" in run or "build-cpp" in run or "bdist_wheel" in run:
                assert step["env"]["MACOSX_DEPLOYMENT_TARGET"] == "11.0"
