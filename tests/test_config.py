from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from smart.pipeline.config import load_config
from smart.pipeline.stages import (
    _run_coacd_cli_preseg,
    _tetra_input_candidates,
    _write_coacd_partition_metadata,
    bbox_dir_for_render,
    inspect_tetra_output,
    latest_bbox_dir,
    list_mesh_ids,
    run_local_refine_mesh,
    run_merge_mesh,
    run_refine_mesh,
    run_mcts_mesh,
    run_native_pipelines,
    validate_tetra_output,
)
from smart.pipeline import tools as pipeline_tools
from smart.pipeline.runner import CommandResult
from smart.pipeline.tools import (
    _first_pymanifold_binary,
    _tool_root_path,
    build_cpp_extension,
    build_vendored_manifold_binding,
    diagnose_environment,
)
from smart.cli import _apply_override, main as smart_main
from smart.evaluation import EvaluationRecord, summarize_records
import smart

def test_demo_config_loads() -> None:
    cfg = load_config("configs/demo.yaml")
    assert cfg["run_name"] == "demo"
    assert cfg["engine"] == "cpp_native"
    assert cfg["merge"]["backend"] == "cpp_native"
    assert cfg["refine"]["backend"] == "cpp_native"
    assert cfg["mcts"]["backend"] == "cpp_native"
    assert cfg["normalization"]["backend"] == "cpp_native_executable"
    assert cfg["normalization"]["native_executable_required"] is True
    assert cfg["preseg"]["backend"] == "coacd_cli"
    assert cfg["preseg"]["partition_metadata_backend"] == "cpp_native"
    assert cfg["preseg"]["partition_metadata_required"] is True
    assert [category["name"] for category in cfg["categories"]] == ["airplane", "chair", "table"]
    assert cfg["tools"]["coacd_external_root"] == "external/CoACD"
    assert cfg["tools"]["coacd_build"] is False
    assert cfg["tools"]["coacd_install_python"] is True
    assert cfg["tools"]["build_cpp_with_tools"] is True
    assert cfg["local_refine_gate"]["enabled"] is False


def test_root_config_profiles_are_packaged_and_synced() -> None:
    root_profiles = sorted(Path("configs").glob("*.yaml"))
    packaged_profiles = {path.name: path for path in Path("smart/configs").glob("*.yaml")}

    assert root_profiles
    for root_profile in root_profiles:
        packaged_profile = packaged_profiles.get(root_profile.name)
        assert packaged_profile is not None, f"missing packaged config: {root_profile.name}"
        assert root_profile.read_text(encoding="utf-8") == packaged_profile.read_text(encoding="utf-8")


def test_smart_cli_lists_config_profiles(capsys) -> None:
    assert smart_main(["configs", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    names = {item["name"] for item in payload}

    assert "smoke_5.yaml" in names
    assert "example_3x3.yaml" in names
    assert all("_experimental" not in name for name in names)


def test_native_run_dry_run_manifests_monolithic_cpp_pipeline(tmp_path) -> None:
    mesh_dir = tmp_path / "data" / "airplane" / "mesh_a"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "model.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    cfg = load_config(None)
    cfg["workspace"] = str(tmp_path / "runs")
    cfg["categories"] = [
        {
            "name": "airplane",
            "mesh_root": str(tmp_path / "data" / "airplane"),
            "meshes": ["mesh_a"],
            "tetra": {"epsilon": 0.003, "edge_length": 0.2},
        }
    ]

    records = run_native_pipelines(cfg, dry_run=True)

    assert len(records) == 1
    record = records[0]
    assert record.stage == "native_pipeline"
    assert record.status == "dry_run"
    assert record.command is not None
    assert "run-pipeline" in record.command
    assert "--epsilon" in record.command
    assert record.command[record.command.index("--epsilon") + 1] == "0.003"
    assert (tmp_path / "runs" / "manifests" / "native_pipeline.jsonl").exists()


def test_public_api_lists_config_profiles() -> None:
    profiles = smart.config_profiles()
    names = {item["name"] for item in profiles}

    assert "smoke_5.yaml" in names
    assert "example_3x3.yaml" in names
    assert all("_experimental" not in name for name in names)


def test_public_api_lists_and_resolves_packaged_assets() -> None:
    gates = smart.asset_profiles("gates")
    priors = smart.asset_profiles("priors")

    assert gates == []
    assert priors == []
    with pytest.raises(FileNotFoundError):
        smart.asset_path("gates", "rich")


def test_smart_cli_lists_packaged_assets(capsys) -> None:
    assert smart_main(["assets", "--kind", "gates", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload == []


def test_cpp_native_engine_sets_native_stage_defaults(tmp_path) -> None:
    config_path = tmp_path / "cpp_native_minimal.yaml"
    config_path.write_text(
        """
run_name: cpp_native_minimal
workspace: runs/cpp_native_minimal
engine: cpp_native
categories: []
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg["merge"]["backend"] == "cpp_native"
    assert cfg["refine"]["backend"] == "cpp_native"
    assert cfg["mcts"]["backend"] == "cpp_native"
    assert cfg["merge"]["direct_file_runner_required"] is True
    assert cfg["refine"]["direct_file_runner_required"] is True
    assert cfg["mcts"]["direct_file_runner_required"] is True
    assert cfg["normalization"]["native_executable_required"] is True
    assert cfg["preseg"]["backend"] == "coacd_cli"
    assert cfg["preseg"]["partition_metadata_backend"] == "cpp_native"
    assert cfg["preseg"]["partition_metadata_required"] is True
    assert cfg["refine"]["reward_backend"] == "manifold_stateful"
    assert cfg["mcts"]["reward_backend"] == "manifold_stateful"


def test_cpp_native_engine_preserves_explicit_legacy_stage_backend(tmp_path) -> None:
    config_path = tmp_path / "cpp_native_explicit_legacy.yaml"
    config_path.write_text(
        """
run_name: cpp_native_explicit_legacy
workspace: runs/cpp_native_explicit_legacy
engine: cpp_native
categories: []
merge:
  backend: legacy_python
refine:
  backend: legacy_python
mcts:
  backend: legacy_python
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg["merge"]["backend"] == "legacy_python"
    assert cfg["refine"]["backend"] == "legacy_python"
    assert cfg["mcts"]["backend"] == "legacy_python"
    assert "direct_file_runner_required" not in cfg["merge"]
    assert "direct_file_runner_required" not in cfg["refine"]
    assert "direct_file_runner_required" not in cfg["mcts"]


def test_manifold_parallel_backend_config_aliases(monkeypatch) -> None:
    monkeypatch.delenv("SMART_MANIFOLD_PAR", raising=False)

    assert pipeline_tools._manifold_parallel_backend({}) == "NONE"
    assert pipeline_tools._manifold_parallel_backend({"build_tools": {"manifold_parallel_backend": "omp"}}) == "OMP"
    assert pipeline_tools._manifold_parallel_backend({"manifold": {"parallel_backend": "tbb"}}) == "TBB"
    assert pipeline_tools._manifold_parallel_backend({"tools": {"manifold_parallel_backend": "serial"}}) == "NONE"


def test_manifold_parallel_backend_env_override_and_invalid(monkeypatch) -> None:
    monkeypatch.setenv("SMART_MANIFOLD_PAR", "OpenMP")
    assert pipeline_tools._manifold_parallel_backend({"build_tools": {"manifold_parallel_backend": "TBB"}}) == "OMP"

    monkeypatch.setenv("SMART_MANIFOLD_PAR", "bad-backend")
    with pytest.raises(ValueError, match="Unsupported Manifold parallel backend"):
        pipeline_tools._manifold_parallel_backend({})


def test_manifold_cuda_config_and_env(monkeypatch) -> None:
    monkeypatch.delenv("SMART_MANIFOLD_USE_CUDA", raising=False)

    assert pipeline_tools._manifold_use_cuda({}) is False
    assert pipeline_tools._manifold_use_cuda({"build_tools": {"manifold_use_cuda": True}}) is True
    assert pipeline_tools._manifold_use_cuda({"manifold": {"use_cuda": "yes"}}) is True
    assert pipeline_tools._manifold_use_cuda({"tools": {"manifold_use_cuda": "off"}}) is False

    monkeypatch.setenv("SMART_MANIFOLD_USE_CUDA", "1")
    assert pipeline_tools._manifold_use_cuda({"build_tools": {"manifold_use_cuda": False}}) is True


def test_manifold_parallel_cmake_args_on_darwin(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pipeline_tools.sys, "platform", "darwin")
    libomp = tmp_path / "libomp"
    (libomp / "include").mkdir(parents=True)
    (libomp / "lib").mkdir()
    (libomp / "lib" / "libomp.dylib").write_text("", encoding="utf-8")
    monkeypatch.setenv("SMART_LIBOMP_PREFIX", str(libomp))

    omp_args = pipeline_tools._manifold_parallel_cmake_args("OMP")
    assert "-DOpenMP_C_LIB_NAMES=omp" in omp_args
    assert any(str(libomp / "include") in arg for arg in omp_args)
    assert any(str(libomp / "lib" / "libomp.dylib") in arg for arg in omp_args)

    tbb = tmp_path / "tbb"
    tbb.mkdir()
    monkeypatch.setenv("SMART_TBB_PREFIX", str(tbb))
    assert pipeline_tools._manifold_parallel_cmake_args("TBB") == [f"-DCMAKE_PREFIX_PATH={tbb}"]


def test_build_tools_relative_tool_root_uses_cwd_outside_repo(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SMART_TOOLS_ROOT", raising=False)

    assert _tool_root_path("external/mesh2tet", "unused") == tmp_path / "external" / "mesh2tet"


def test_build_tools_relative_tool_root_can_use_env(tmp_path, monkeypatch) -> None:
    tool_root = tmp_path / "smart_tools"
    monkeypatch.setenv("SMART_TOOLS_ROOT", str(tool_root))

    assert _tool_root_path("external/mesh2tet", "unused") == tool_root / "external" / "mesh2tet"


def test_demo_sample_counts_are_limited() -> None:
    cfg = load_config("configs/demo.yaml")
    limits = {category["name"]: category.get("limit") for category in cfg["categories"]}
    assert limits == {"airplane": 50, "chair": 50, "table": 50}


def test_smoke_config_uses_explicit_meshes() -> None:
    cfg = load_config("configs/smoke_5.yaml")
    counts = {category["name"]: len(category.get("meshes", [])) for category in cfg["categories"]}
    assert counts == {"airplane": 2, "chair": 2, "table": 1}


def test_example_3x3_config_uses_three_meshes_per_category() -> None:
    cfg = load_config("configs/example_3x3.yaml")
    counts = {category["name"]: len(category.get("meshes", [])) for category in cfg["categories"]}
    assert counts == {"airplane": 3, "chair": 3, "table": 3}
    assert cfg["workspace"] == "examples/runs/example_3x3"


def test_cli_override_updates_nested_config_values() -> None:
    cfg = load_config("configs/smoke_5.yaml")
    _apply_override(cfg, "mcts.mcts_iter=12")
    _apply_override(cfg, "render.joint_mesh=true")
    _apply_override(cfg, "render.variants=[\"boxes_only\",\"with_mesh\"]")

    assert cfg["mcts"]["mcts_iter"] == 12
    assert cfg["render"]["joint_mesh"] is True
    assert cfg["render"]["variants"] == ["boxes_only", "with_mesh"]


def test_search_order_changes_are_guarded_for_exact_compatibility(tmp_path) -> None:
    cfg = {
        "workspace": str(tmp_path),
        "normalization": {"enabled": False},
        "mcts": {"transposition_table": True},
    }

    record = run_mcts_mesh(cfg, {"name": "table"}, "mesh-a")

    assert record.status == "blocked"
    assert "changes search order" in record.error


def test_action_prior_requires_search_order_opt_in(tmp_path) -> None:
    cfg = {
        "workspace": str(tmp_path),
        "normalization": {"enabled": False},
        "mcts": {"action_prior_weight": 0.1},
    }

    record = run_mcts_mesh(cfg, {"name": "table"}, "mesh-a")

    assert record.status == "blocked"
    assert "action_prior_weight" in record.error


def test_local_refine_action_prior_requires_search_order_opt_in(tmp_path) -> None:
    cfg = {
        "workspace": str(tmp_path),
        "normalization": {"enabled": False},
        "local_refine": {"action_prior_weight": 0.1},
    }

    record = run_local_refine_mesh(cfg, {"name": "table"}, "mesh-a")

    assert record.status == "blocked"
    assert "local_refine.action_prior_weight" in record.error


def test_local_refine_action_value_requires_search_order_opt_in(tmp_path) -> None:
    cfg = {
        "workspace": str(tmp_path),
        "normalization": {"enabled": False},
        "local_refine": {"action_value_weight": 0.1},
    }

    record = run_local_refine_mesh(cfg, {"name": "table"}, "mesh-a")

    assert record.status == "blocked"
    assert "local_refine.action_value_weight" in record.error


def test_puct_prior_requires_search_order_opt_in(tmp_path) -> None:
    cfg = {
        "workspace": str(tmp_path),
        "normalization": {"enabled": False},
        "mcts": {"puct_prior_weight": 0.1},
    }

    record = run_mcts_mesh(cfg, {"name": "table"}, "mesh-a")

    assert record.status == "blocked"
    assert "puct_prior_weight" in record.error


def test_mcts_action_prior_top_k_requires_search_order_opt_in(tmp_path) -> None:
    cfg = {
        "workspace": str(tmp_path),
        "normalization": {"enabled": False},
        "mcts": {"action_prior_top_k": 8},
    }

    record = run_mcts_mesh(cfg, {"name": "table"}, "mesh-a")

    assert record.status == "blocked"
    assert "action_prior_top_k" in record.error


def test_mcts_escape_policy_requires_search_order_opt_in(tmp_path) -> None:
    cfg = {
        "workspace": str(tmp_path),
        "normalization": {"enabled": False},
        "mcts": {"escape_policy": True},
    }

    record = run_mcts_mesh(cfg, {"name": "table"}, "mesh-a")

    assert record.status == "blocked"
    assert "escape_policy" in record.error


def test_mcts_cpp_rng_requires_search_order_opt_in(tmp_path) -> None:
    cfg = {
        "workspace": str(tmp_path),
        "normalization": {"enabled": False},
        "mcts": {"cpp_rng": True},
    }

    record = run_mcts_mesh(cfg, {"name": "table"}, "mesh-a")

    assert record.status == "blocked"
    assert "cpp_rng" in record.error


def test_cpp_mcts_backend_requires_search_order_opt_in(tmp_path) -> None:
    cfg = {
        "workspace": str(tmp_path),
        "normalization": {"enabled": False},
        "mcts": {"backend": "cpp_stateful"},
    }

    record = run_mcts_mesh(cfg, {"name": "table"}, "mesh-a")

    assert record.status == "blocked"
    assert "backend=cpp/cpp_stateful/cpp_native/native/native_stateful" in record.error


def test_mcts_native_rollout_flags_are_forwarded(tmp_path) -> None:
    bbox_root = tmp_path / "bbox_input"
    (bbox_root / "result").mkdir(parents=True)
    cfg = {
        "workspace": str(tmp_path),
        "python": "python3",
        "normalization": {"enabled": False},
        "tetra": {"epsilon": 0.004, "edge_length": 0.2},
        "mcts": {
            "backend": "cpp_stateful",
            "allow_search_order_changes": True,
            "path_to_bbox": str(bbox_root),
            "native_axis_rollout_step": True,
            "native_axis_rollout_segment": True,
            "cpp_rng": True,
            "cpp_rng_seed": 12345,
        },
    }

    record = run_mcts_mesh(
        cfg,
        {"name": "table", "mesh_root": str(tmp_path / "table_meshes")},
        "mesh-a",
        dry_run=True,
        force=True,
    )

    assert record.status == "dry_run"
    assert "--mcts_native_axis_rollout_step" in record.command
    assert "--mcts_native_axis_rollout_segment" in record.command
    assert "--mcts_cpp_rng" in record.command
    assert record.command[record.command.index("--mcts_cpp_rng_seed") + 1] == "12345"


def test_merge_cpp_native_backend_flags_are_forwarded(tmp_path) -> None:
    mesh_root = tmp_path / "table_meshes"
    tetra_dir = tmp_path / "tetra" / "table_meshes_raw_e0.004_l0.2" / "mesh-a"
    (tetra_dir / "coacd").mkdir(parents=True)
    (tetra_dir / "tetra.msh").write_text("$MeshFormat\n", encoding="utf-8")
    (tetra_dir / "coacd" / "part_0000.obj").write_text("o part\n", encoding="utf-8")
    cfg = {
        "workspace": str(tmp_path),
        "python": "python3",
        "normalization": {"enabled": False},
        "tetra": {"epsilon": 0.004, "edge_length": 0.2},
        "merge": {
            "backend": "cpp_native",
            "init_type": "coacd",
            "tilted": True,
            "cpp_native_allow_tilted_axis": True,
        },
    }

    record = run_merge_mesh(
        cfg,
        {"name": "table", "mesh_root": str(mesh_root)},
        "mesh-a",
        dry_run=True,
        force=True,
    )

    assert record.status == "dry_run"
    assert "--merge_backend" in record.command
    assert record.command[record.command.index("--merge_backend") + 1] == "cpp_native"
    assert "--cpp_native_merge_allow_tilted_axis" in record.command


def test_cpp_native_merge_uses_direct_file_runner_when_partitions_exist(tmp_path, monkeypatch) -> None:
    from smart import native_runner

    monkeypatch.setattr(native_runner, "native_executable_path", lambda: Path("smart-cpp-native"))
    mesh_root = tmp_path / "table_meshes"
    tetra_dir = tmp_path / "tetra" / "table_meshes_raw_e0.004_l0.2" / "mesh-a"
    (tetra_dir / "coacd").mkdir(parents=True)
    (tetra_dir / "tetra.msh").write_text("$MeshFormat\n", encoding="utf-8")
    (tetra_dir / "coacd" / "part_0000.obj").write_text("o part\n", encoding="utf-8")
    (tetra_dir / "coacd_partitions.json").write_text(
        json.dumps({"partitions": [[0], [1]]}), encoding="utf-8"
    )
    cfg = {
        "workspace": str(tmp_path),
        "python": "python3",
        "normalization": {"enabled": False},
        "tetra": {"epsilon": 0.004, "edge_length": 0.2},
        "merge": {
            "backend": "cpp_native",
            "direct_file_runner": True,
            "init_type": "coacd",
            "tilted": False,
            "merge_eps": 0.02,
            "fast_merge": True,
        },
    }

    record = run_merge_mesh(
        cfg,
        {"name": "table", "mesh_root": str(mesh_root)},
        "mesh-a",
        dry_run=True,
        force=True,
    )

    assert record.status == "dry_run"
    assert record.command[:2] == ["smart-cpp-native", "merge"]
    assert record.output_path is not None
    assert record.output_path.endswith("greedy_segment0_coacd_mgeps0.02_fm.txt")


def test_write_coacd_partition_metadata_caches_tet_assignments(tmp_path, monkeypatch) -> None:
    class DummyManifold:
        pass

    class DummyMesh:
        pass

    monkeypatch.setitem(
        sys.modules,
        "pymanifold",
        type("DummyPyManifold", (), {"Manifold": DummyManifold, "Mesh": DummyMesh})(),
    )
    from smart.legacy.merging.src.utils import preseg as preseg_module
    import smart.pymesh_compat as pymesh

    tet_dir = tmp_path / "tetra" / "mesh-a"
    (tet_dir / "coacd").mkdir(parents=True)
    (tet_dir / "coacd" / "part_0000.obj").write_text("o part\n", encoding="utf-8")
    (tet_dir / "coacd" / "part_0001.obj").write_text("o part\n", encoding="utf-8")

    class FakeTetMesh:
        voxels = [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4]]

        def enable_connectivity(self) -> None:
            self.enabled = True

    fake_tetmesh = FakeTetMesh()

    def fake_load_mesh(path):
        assert Path(path) == tet_dir / "tetra.msh"
        return fake_tetmesh

    def fake_presegmentation(data_path, tetmsh, preseg_type, path_to_bbox, fn, debug=False):
        assert Path(data_path) == tet_dir
        assert tetmsh is fake_tetmesh
        assert preseg_type == "coacd"
        assert path_to_bbox == ""
        assert fn == "mesh-a"
        assert debug is False
        return [[1, 0], [2]]

    monkeypatch.setattr(pymesh, "load_mesh", fake_load_mesh)
    monkeypatch.setattr(preseg_module, "presegmentation", fake_presegmentation)

    metadata = _write_coacd_partition_metadata(tet_dir, "mesh-a", force=True)
    payload = json.loads((tet_dir / "coacd_partitions.json").read_text(encoding="utf-8"))

    assert metadata["partition_count"] == 2
    assert metadata["partition_metadata_cached"] is False
    assert payload["source"] == "smart.pipeline.preseg.coacd_partition_metadata"
    assert payload["partitions"] == [[0, 1], [2]]
    assert payload["part_obj_count"] == 2


def test_write_coacd_partition_metadata_prefers_cpp_native(monkeypatch, tmp_path) -> None:
    from smart import native_runner

    tet_dir = tmp_path / "tetra" / "mesh-a"
    (tet_dir / "coacd").mkdir(parents=True)
    (tet_dir / "tetra.msh").write_text("$MeshFormat\n", encoding="utf-8")
    (tet_dir / "coacd" / "part_0000.obj").write_text("o part\n", encoding="utf-8")

    def fake_available() -> bool:
        return True

    def fake_run_coacd_partition_from_files(**kwargs):
        assert Path(kwargs["msh_path"]) == tet_dir / "tetra.msh"
        assert Path(kwargs["coacd_dir"]) == tet_dir / "coacd"
        assert Path(kwargs["output_path"]) == tet_dir / "coacd_partitions.json"
        assert kwargs["mesh_id"] == "mesh-a"
        Path(kwargs["output_path"]).write_text(
            json.dumps({"partitions": [[0]], "source": "smart-cpp-native partition-coacd"}),
            encoding="utf-8",
        )
        return {
            "status": "success",
            "output_path": Path(kwargs["output_path"]),
            "metadata": {"backend": "smart-cpp-native"},
        }

    monkeypatch.setattr(native_runner, "cpp_native_file_runner_available", fake_available)
    monkeypatch.setattr(native_runner, "run_coacd_partition_from_files", fake_run_coacd_partition_from_files)

    metadata = _write_coacd_partition_metadata(tet_dir, "mesh-a", force=True)

    assert metadata["partition_count"] == 1
    assert metadata["partition_metadata_backend"] == "smart-cpp-native"
    assert "partition_metadata_native_error" not in metadata


def test_run_coacd_cli_preseg_splits_combined_obj(monkeypatch, tmp_path) -> None:
    import smart.pipeline.stages as stages

    coacd_bin = tmp_path / "coacd-main"
    coacd_bin.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '-o' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "cat > \"$out\" <<'EOF'\n"
        "o convex_0\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n"
        "EOF\n",
        encoding="utf-8",
    )
    coacd_bin.chmod(0o755)
    surface = tmp_path / "tetra.msh__sf.obj"
    surface.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    out_dir = tmp_path / "coacd"

    class Completed:
        returncode = 0
        stdout = '{"parts":1}'
        stderr = ""

    def fake_run_native_command(args):
        assert args[0] == "split-obj-parts"
        output_dir = Path(args[args.index("--output_dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "part_0000.obj").write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
            encoding="utf-8",
        )
        return Completed()

    monkeypatch.setattr(stages, "native_executable_path", lambda: Path("/tmp/smart-cpp-native"))
    monkeypatch.setattr(stages, "run_native_command", fake_run_native_command)

    ok, metadata, error = _run_coacd_cli_preseg(
        {
            "tools": {"coacd_bin": str(coacd_bin)},
        },
        {
            "timeout_sec": 30,
            "coacd": {
                "threshold": 0.05,
                "max_convex_hull": 64,
                "preprocess_mode": "auto",
                "preprocess_resolution": 50,
                "resolution": 2000,
                "mcts_nodes": 20,
                "mcts_iterations": 150,
                "mcts_max_depth": 3,
                "seed": 7777,
            },
        },
        surface,
        out_dir,
        dry_run=False,
    )

    assert ok is True
    assert error is None
    assert metadata["backend"] == "coacd_cli"
    assert metadata["parts"] == 1
    assert (out_dir / "part_0000.obj").exists()


def test_write_coacd_partition_metadata_can_require_cpp_native(monkeypatch, tmp_path) -> None:
    from smart import native_runner

    tet_dir = tmp_path / "tetra" / "mesh-a"
    (tet_dir / "coacd").mkdir(parents=True)
    (tet_dir / "tetra.msh").write_text("$MeshFormat\n", encoding="utf-8")
    (tet_dir / "coacd" / "part_0000.obj").write_text("o part\n", encoding="utf-8")

    monkeypatch.setattr(native_runner, "cpp_native_file_runner_available", lambda: False)

    with pytest.raises(RuntimeError, match="partition-coacd is required"):
        _write_coacd_partition_metadata(tet_dir, "mesh-a", force=True, require_native=True)


def test_cpp_native_file_runner_available_accepts_packaged_executable(monkeypatch) -> None:
    from smart import native_runner

    monkeypatch.setattr(native_runner, "native_executable_path", lambda: Path("/tmp/smart-cpp-native"))
    monkeypatch.setattr(native_runner, "smart_native", None)

    assert native_runner.cpp_native_file_runner_available() is True


def test_cpp_native_refine_uses_direct_file_runner_when_metadata_exists(tmp_path, monkeypatch) -> None:
    from smart import native_runner

    monkeypatch.setattr(native_runner, "native_executable_path", lambda: Path("smart-cpp-native"))
    mesh_root = tmp_path / "table_meshes"
    tetra_dir = tmp_path / "tetra" / "table_meshes_raw_e0.004_l0.2" / "mesh-a"
    tetra_dir.mkdir(parents=True)
    (tetra_dir / "tetra.msh").write_text("$MeshFormat\n", encoding="utf-8")
    segment = tetra_dir / "greedy_segment0_coacd_mgeps0.02_fm.txt"
    segment.write_text("1\n0\n", encoding="utf-8")
    (Path(str(segment) + ".bbox_params.json")).write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "index": 0,
                        "bounds": [0, 0, 0, 1, 1, 1],
                        "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "workspace": str(tmp_path),
        "python": "python3",
        "normalization": {"enabled": False},
        "tetra": {"epsilon": 0.004, "edge_length": 0.2},
        "merge": {"init_type": "coacd", "merge_eps": 0.02, "fast_merge": True},
        "refine": {"backend": "cpp_native", "direct_file_runner": True},
    }

    record = run_refine_mesh(
        cfg,
        {"name": "table", "mesh_root": str(mesh_root)},
        "mesh-a",
        dry_run=True,
        force=True,
    )

    assert record.status == "dry_run"
    assert record.command[:2] == ["smart-cpp-native", "refine"]
    assert record.output_path is not None
    assert record.output_path.endswith("bboxs_steps0")


def test_cpp_native_refine_learned_router_is_pipeline_opt_in(tmp_path, monkeypatch) -> None:
    from smart import native_runner

    monkeypatch.setattr(native_runner, "native_executable_path", lambda: Path("smart-cpp-native"))
    mesh_root = tmp_path / "table_meshes"
    tetra_dir = tmp_path / "tetra" / "table_meshes_raw_e0.004_l0.2" / "mesh-a"
    tetra_dir.mkdir(parents=True)
    (tetra_dir / "tetra.msh").write_text("$MeshFormat\n", encoding="utf-8")
    segment = tetra_dir / "greedy_segment0_coacd_mgeps0.02_fm.txt"
    segment.write_text("1\n0\n", encoding="utf-8")
    (Path(str(segment) + ".bbox_params.json")).write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "index": 0,
                        "bounds": [0, 0, 0, 1, 1, 1],
                        "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "workspace": str(tmp_path),
        "python": "python3",
        "normalization": {"enabled": False},
        "tetra": {"epsilon": 0.004, "edge_length": 0.2},
        "merge": {"init_type": "coacd", "merge_eps": 0.02, "fast_merge": True},
        "refine": {
            "backend": "cpp_native",
            "direct_file_runner": True,
            "learned_router": {
                "enabled": True,
                "policy": "default",
                "profile": "auto",
                "overrides": {"budget": 6},
            },
        },
    }

    record = run_refine_mesh(
        cfg,
        {"name": "table", "mesh_root": str(mesh_root)},
        "mesh-a",
        dry_run=True,
        force=True,
    )

    assert record.status == "dry_run"
    assert record.command[:2] == ["smart._cpp", "run_builtin_deepset_policy_refine"]
    assert record.metadata["learned_router"] is True
    assert "_deepset_auto" in record.output_path


def test_cpp_native_mcts_uses_direct_file_runner_when_metadata_exists(tmp_path, monkeypatch) -> None:
    from smart import native_runner

    monkeypatch.setattr(native_runner, "native_executable_path", lambda: Path("smart-cpp-native"))
    mesh_root = tmp_path / "table_meshes"
    tetra_dir = tmp_path / "tetra" / "table_meshes_raw_e0.004_l0.2" / "mesh-a"
    tetra_dir.mkdir(parents=True)
    (tetra_dir / "tetra.msh").write_text("$MeshFormat\n", encoding="utf-8")
    bbox_dir = tmp_path / "refine" / "table" / "exp" / "result" / "updated0" / "mesh-a" / "bboxs_steps0"
    bbox_dir.mkdir(parents=True)
    (bbox_dir / "bbox_params.json").write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "index": 0,
                        "bounds": [0, 0, 0, 1, 1, 1],
                        "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "workspace": str(tmp_path),
        "python": "python3",
        "normalization": {"enabled": False},
        "tetra": {"epsilon": 0.004, "edge_length": 0.2},
        "merge": {},
        "mcts": {
            "backend": "cpp_native",
            "direct_file_runner": True,
            "mcts_iter": 2,
            "max_step": 1,
        },
    }

    record = run_mcts_mesh(
        cfg,
        {"name": "table", "mesh_root": str(mesh_root)},
        "mesh-a",
        dry_run=True,
        force=True,
    )

    assert record.status == "dry_run"
    assert record.command[:2] == ["smart-cpp-native", "mcts"]
    assert record.output_path is not None
    assert record.output_path.endswith("bboxs_steps0")


def test_cpp_native_mcts_direct_file_runner_accepts_static_prior(tmp_path, monkeypatch) -> None:
    from smart import native_runner

    monkeypatch.setattr(native_runner, "native_executable_path", lambda: Path("smart-cpp-native"))
    mesh_root = tmp_path / "table_meshes"
    tetra_dir = tmp_path / "tetra" / "table_meshes_raw_e0.004_l0.2" / "mesh-a"
    tetra_dir.mkdir(parents=True)
    (tetra_dir / "tetra.msh").write_text("$MeshFormat\n", encoding="utf-8")
    bbox_dir = tmp_path / "refine" / "table" / "exp" / "result" / "updated0" / "mesh-a" / "bboxs_steps0"
    bbox_dir.mkdir(parents=True)
    (bbox_dir / "bbox_params.json").write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "index": 0,
                        "bounds": [0, 0, 0, 1, 1, 1],
                        "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps(
            {
                "policy_type": "coord_scale_count_prior",
                "coord_scale_logits": {"6:0": 0.0},
                "default_logit": -1.0,
                "num_action_scale": 2,
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "workspace": str(tmp_path),
        "python": "python3",
        "normalization": {"enabled": False},
        "tetra": {"epsilon": 0.004, "edge_length": 0.2},
        "merge": {},
        "mcts": {
            "backend": "cpp_native",
            "direct_file_runner": True,
            "allow_search_order_changes": True,
            "action_prior_path": str(prior),
            "action_prior_weight": 0.1,
            "action_prior_top_k": 1,
            "mcts_iter": 2,
            "max_step": 1,
        },
    }

    record = run_mcts_mesh(
        cfg,
        {"name": "table", "mesh_root": str(mesh_root)},
        "mesh-a",
        dry_run=True,
        force=True,
    )

    assert record.status == "dry_run"
    assert record.command[:2] == ["smart-cpp-native", "mcts"]
    assert "--action_prior_path" in record.command
    assert record.command[record.command.index("--action_prior_weight") + 1] == "0.1"
    assert record.command[record.command.index("--action_prior_top_k") + 1] == "1"


def test_cpp_native_mcts_can_run_combined_refine_mcts_file_runner(tmp_path) -> None:
    mesh_root = tmp_path / "table_meshes"
    tetra_dir = tmp_path / "tetra" / "table_meshes_raw_e0.004_l0.2" / "mesh-a"
    tetra_dir.mkdir(parents=True)
    (tetra_dir / "tetra.msh").write_text("$MeshFormat\n", encoding="utf-8")
    segment = tetra_dir / "greedy_segment0_coacd_mgeps0.02_fm.txt"
    segment.write_text("1\n0\n", encoding="utf-8")
    (Path(str(segment) + ".bbox_params.json")).write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "index": 0,
                        "bounds": [0, 0, 0, 1, 1, 1],
                        "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "workspace": str(tmp_path),
        "python": "python3",
        "normalization": {"enabled": False},
        "tetra": {"epsilon": 0.004, "edge_length": 0.2},
        "merge": {"init_type": "coacd", "merge_eps": 0.02, "fast_merge": True},
        "refine": {"backend": "cpp_native", "max_step": 0},
        "mcts": {
            "backend": "cpp_native",
            "direct_file_runner": True,
            "combined_refine": True,
            "allow_search_order_changes": True,
            "mcts_iter": 2,
            "max_step": 1,
        },
    }

    record = run_mcts_mesh(
        cfg,
        {"name": "table", "mesh_root": str(mesh_root)},
        "mesh-a",
        dry_run=True,
        force=True,
    )

    assert record.status == "dry_run"
    assert record.command[:2] == ["smart-cpp-native", "refine-mcts"]
    assert record.command[record.command.index("--mcts_iter") + 1] == "2"
    assert record.command[record.command.index("--mcts_max_step") + 1] == "1"
    assert record.command[record.command.index("--refine_max_step") + 1] == "0"
    assert record.output_path is not None
    assert record.output_path.endswith("bboxs_steps0")
    assert record.metadata["combined"] is True
    assert record.metadata["single_mesh_load"] is True
    assert record.metadata["single_state_bridge"] is True
    assert "combined_stats_path" not in record.metadata


def test_local_refine_stage_skips_when_mcts_output_missing(tmp_path) -> None:
    cfg = {
        "workspace": str(tmp_path),
        "python": "python3",
        "merge": {},
        "local_refine": {"input_stage": "mcts"},
        "categories": [],
    }

    record = run_local_refine_mesh(cfg, {"name": "table"}, "mesh-a")

    assert record.stage == "local_refine"
    assert record.status == "skipped"
    assert "missing mcts bbox output" in str(record.error)


def test_relative_config_prefers_current_working_directory(tmp_path, monkeypatch) -> None:
    local_config = tmp_path / "local.json"
    local_config.write_text("{\"run_name\": \"local-profile\", \"categories\": []}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    cfg = load_config("local.json")

    assert cfg["run_name"] == "local-profile"
    assert cfg["_config_path"] == str(local_config)


def test_missing_relative_config_falls_back_to_bundled_profile(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    cfg = load_config("missing_configs/demo.yaml")

    assert cfg["run_name"] == "demo"
    assert cfg["_config_path"].endswith("smart/configs/demo.yaml")


def test_doctor_reports_runtime_and_vendored_manifold() -> None:
    cfg = load_config("configs/smoke_5.yaml")

    status = diagnose_environment(cfg)
    checks = {check["name"]: check for check in status["checks"]}

    assert checks["python"]["ok"] is True
    assert "smart-rust-extension" not in checks
    assert checks["smart-cpp-extension"]["kind"] == "python-extension"
    assert checks["smart-pymesh-compat"]["ok"] is True
    assert checks["smart-pymesh-compat"]["required_for"] == ["merge", "refine", "mcts"]
    assert checks["pymesh"]["detail"] == "legacy alias for smart.pymesh_compat"
    assert checks["pymesh"]["required_for"] == []
    assert "vendored-manifold-source" in checks
    assert checks["vendored-manifold-source"]["detail"] == (
        "kept as fixed C++ binding source; do not pull or replace"
    )
    assert checks["vendored-manifold-source"]["required_for"] == []
    assert checks["CoACD-source"]["kind"] == "source-tree"
    assert checks["CoACD-source"]["required_for"] == []
    assert "smart-cpp-native" in checks
    assert checks["smart-cpp-native"]["required_for"] == []
    assert isinstance(status["required_failures"], list)
    assert isinstance(status["optional_failures"], list)


def test_pymanifold_binary_probe(tmp_path) -> None:
    binary = tmp_path / "pymanifold.cpython-39-darwin.so"
    binary.write_text("", encoding="utf-8")

    assert _first_pymanifold_binary(tmp_path) == binary


def test_build_vendored_manifold_dry_run_stages_runtime(tmp_path) -> None:
    messages = build_vendored_manifold_binding({"workspace": str(tmp_path / "runs")}, dry_run=True)

    assert "vendored-manifold configure: dry-run" in messages
    assert "vendored-manifold build: dry-run" in messages
    assert messages[-1].startswith("pymanifold runtime staging: dry-run")


def test_build_tools_dry_run_fetches_coacd_source(tmp_path) -> None:
    messages = pipeline_tools.build_tools(
        {
            "workspace": str(tmp_path / "runs"),
            "tools": {
                "mesh2tet_external_root": str(tmp_path / "external" / "mesh2tet"),
                "coacd_external_root": str(tmp_path / "external" / "CoACD"),
            },
        },
        dry_run=True,
    )

    assert "CoACD source clone: dry-run" in messages
    assert any(message.startswith("CoACD source checkout: dry-run -> ") for message in messages)
    assert "CoACD Python dependency install: dry-run" in messages
    assert "CoACD Python runtime install: dry-run" in messages
    assert any(message.startswith("CoACD CLI probe: dry-run -> ") for message in messages)
    assert "smart-cpp-native executable build: dry-run" in messages


def test_coacd_editable_install_failure_is_nonfatal_when_cli_probe_passes(
    tmp_path, monkeypatch
) -> None:
    coacd_root = tmp_path / "CoACD"
    coacd_script = coacd_root / "python" / "package" / "bin" / "coacd"
    coacd_script.parent.mkdir(parents=True)
    coacd_script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    def fake_run_command(command, **kwargs):
        log_path = Path(kwargs["log_path"])
        returncode = 1 if log_path.name == "coacd-python-install.log" else 0
        if log_path.name == "coacd-cli-preflight.log":
            returncode = 2
        return CommandResult(
            command=[str(part) for part in command],
            returncode=returncode,
            elapsed_sec=0.0,
            log_path=log_path,
        )

    monkeypatch.setattr(pipeline_tools, "run_command", fake_run_command)

    messages = pipeline_tools._prepare_coacd_runtime(
        coacd_root,
        tmp_path / "logs",
        {"coacd_install_python": True},
    )

    assert "CoACD source editable install: warning rc=1; trying PyPI CoACD runtime fallback" in messages
    assert "CoACD PyPI runtime install: ok" in messages
    assert any(message.startswith("CoACD CLI probe") and message.endswith(": ok") for message in messages)
    assert all(": failed" not in message for message in messages)


def test_coacd_runtime_skips_install_when_cli_already_works(tmp_path, monkeypatch) -> None:
    coacd_root = tmp_path / "CoACD"
    coacd_script = coacd_root / "python" / "package" / "bin" / "coacd"
    coacd_script.parent.mkdir(parents=True)
    coacd_script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run_command(command, **kwargs):
        commands.append([str(part) for part in command])
        return CommandResult(
            command=[str(part) for part in command],
            returncode=0,
            elapsed_sec=0.0,
            log_path=Path(kwargs["log_path"]),
        )

    monkeypatch.setattr(pipeline_tools, "run_command", fake_run_command)

    messages = pipeline_tools._prepare_coacd_runtime(
        coacd_root,
        tmp_path / "logs",
        {"coacd_install_python": True},
    )

    assert any(message.startswith("CoACD CLI probe") and message.endswith(": ok") for message in messages)
    assert len(commands) == 1
    assert all("pip" not in part for command in commands for part in command)


def test_build_cpp_dry_run_can_request_asan_binary(tmp_path) -> None:
    messages = build_cpp_extension({"workspace": str(tmp_path / "runs")}, dry_run=True, asan=True)

    assert "smart._cpp C++ build: dry-run" in messages
    assert "smart-cpp-native executable build: dry-run" in messages
    assert "smart-cpp-native ASan executable build: dry-run" in messages
    assert "smart-cpp-native ASan executable: dry-run" in messages


def test_cli_build_cpp_accepts_asan_dry_run(capsys) -> None:
    assert smart_main(["--config", "configs/smoke_5.yaml", "build-cpp", "--asan", "--dry-run"]) == 0
    captured = capsys.readouterr()
    assert "smart-cpp-native ASan executable build: dry-run" in captured.out


def test_evaluation_summary_averages_successes_only() -> None:
    records = [
        EvaluationRecord(
            category="airplane",
            mesh_id="a",
            stage="mcts",
            status="success",
            elapsed_sec=2.0,
            num_box=4,
            BVS=1.5,
            MOV=0.2,
            TOV=0.5,
            Covered=1.0,
            vIoU=0.75,
            cub_CD=0.01,
        ),
        EvaluationRecord(
            category="airplane",
            mesh_id="b",
            stage="mcts",
            status="failed",
            elapsed_sec=1.0,
            error="missing result",
        ),
    ]

    summary = summarize_records(records)

    assert summary["total"] == 2
    assert summary["success"] == 1
    assert summary["failed"] == 1
    assert summary["Avg_num_box"] == 4.0
    assert summary["Avg_BVS"] == 1.5


def test_tetra_validation_can_allow_disconnected_surfaces(tmp_path) -> None:
    import trimesh

    first = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    second = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    second.apply_translation((3.0, 0.0, 0.0))
    surface = trimesh.util.concatenate([first, second])

    (tmp_path / "tetra.msh").write_text("$MeshFormat\n", encoding="utf-8")
    surface.export(tmp_path / "tetra.msh__sf.obj")

    error, metadata = inspect_tetra_output(tmp_path, require_single_component=False)

    assert error is None
    assert metadata["surface_watertight"] is True
    assert metadata["surface_component_count"] == 2
    assert validate_tetra_output(tmp_path, require_single_component=True) == (
        "surface has multiple connected components"
    )


def test_tetra_input_candidates_add_fill_holes_fallback(tmp_path, monkeypatch) -> None:
    mesh_path = tmp_path / "open_cube.obj"
    mesh_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    stage_cfg = load_config(None)["tetra"]

    def fake_prepare(source: Path, output: Path, cfg: dict) -> tuple[Path, dict]:
        repair_cfg = cfg.get("input_repair", {})
        if repair_cfg.get("fill_holes"):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(mesh_path.read_text(encoding="utf-8"), encoding="utf-8")
            return output, {"enabled": True, "used": True, "variant": "fill_holes"}
        return source, {"enabled": True, "used": True, "variant": "primary"}

    monkeypatch.setattr("smart.pipeline.stages._prepare_tetra_input_mesh", fake_prepare)

    candidates, repair_records = _tetra_input_candidates(
        mesh_path,
        tmp_path / "logs",
        stage_cfg,
        active_failure_classes={"validation_open_surface"},
    )

    assert candidates[0]["name"] == "primary"
    assert any(candidate["name"] == "fill_holes" for candidate in candidates)
    assert any(record.get("variant") == "fill_holes" and record.get("used") for record in repair_records)


def test_tetra_input_candidates_wait_for_failure_class_before_fill_holes(tmp_path, monkeypatch) -> None:
    mesh_path = tmp_path / "mesh.obj"
    mesh_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    stage_cfg = load_config(None)["tetra"]

    def fake_prepare(source: Path, output: Path, cfg: dict) -> tuple[Path, dict]:
        repair_cfg = cfg.get("input_repair", {})
        if repair_cfg.get("fill_holes"):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(mesh_path.read_text(encoding="utf-8"), encoding="utf-8")
            return output, {"enabled": True, "used": True, "variant": "fill_holes"}
        return source, {"enabled": True, "used": True, "variant": "primary"}

    monkeypatch.setattr("smart.pipeline.stages._prepare_tetra_input_mesh", fake_prepare)

    candidates, _ = _tetra_input_candidates(
        mesh_path,
        tmp_path / "logs",
        stage_cfg,
        active_failure_classes=set(),
    )

    assert [candidate["name"] for candidate in candidates] == ["primary"]


def test_tetra_input_repair_skips_component_split_unless_needed(tmp_path, monkeypatch) -> None:
    from smart.pipeline.stages import _prepare_tetra_input_mesh

    mesh_path = tmp_path / "mesh.obj"
    mesh_path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    stage_cfg = load_config(None)["tetra"]
    stage_cfg = {
        **stage_cfg,
        "input_repair": {
            **stage_cfg["input_repair"],
            "keep_largest_component": False,
            "basic_cleanup": False,
            "fix_normals": False,
        },
    }

    def fail_split(*_args, **_kwargs):
        raise AssertionError("component split should not run")

    monkeypatch.setattr("smart.pipeline.stages._split_mesh_components", fail_split)

    output, metadata = _prepare_tetra_input_mesh(mesh_path, tmp_path / "repaired.obj", stage_cfg)

    assert output == tmp_path / "repaired.obj"
    assert metadata["used"] is True
    assert metadata["before"]["components"] is None
    assert metadata["after"]["components"] is None


def test_bbox_dir_prefers_success_manifest_output(tmp_path) -> None:
    category = {"name": "chair", "mesh_root": str(tmp_path / "meshes")}
    cfg = {"workspace": str(tmp_path), "normalization": {"enabled": False}}
    old_path = tmp_path / "mcts" / "chair" / "exp" / "result" / "updated2" / "mesh-a" / "bboxs_steps6"
    manifest_path = tmp_path / "mcts" / "chair" / "exp" / "result" / "updated11" / "mesh-a" / "bboxs_steps8"
    old_path.mkdir(parents=True)
    manifest_path.mkdir(parents=True)
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    (manifest_root / "mcts.jsonl").write_text(
        "\n".join(
            [
                '{"category":"chair","mesh_id":"mesh-a","stage":"mcts","status":"success",'
                f'"finished_at":1,"output_path":"{manifest_path}"' + "}",
                '{"category":"chair","mesh_id":"mesh-a","stage":"mcts","status":"success",'
                f'"finished_at":0,"output_path":"{old_path}"' + "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert bbox_dir_for_render(cfg, category, "mesh-a", "mcts") == manifest_path


def test_bbox_dir_uses_manifest_for_custom_stage(tmp_path) -> None:
    category = {"name": "chair", "mesh_root": str(tmp_path / "meshes")}
    cfg = {"workspace": str(tmp_path), "normalization": {"enabled": False}}
    manifest_path = (
        tmp_path
        / "local_refine_gate_guarded"
        / "chair"
        / "exp"
        / "result"
        / "updated0"
        / "mesh-a"
        / "bboxs_steps0"
    )
    manifest_path.mkdir(parents=True)
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    (manifest_root / "local_refine_gate_guarded.jsonl").write_text(
        '{"category":"chair","mesh_id":"mesh-a","stage":"local_refine_gate_guarded",'
        f'"status":"success","finished_at":1,"output_path":"{manifest_path}"' + "}\n",
        encoding="utf-8",
    )

    assert bbox_dir_for_render(cfg, category, "mesh-a", "local_refine_gate_guarded") == manifest_path


def test_latest_bbox_dir_can_filter_outputs_before_current_run(tmp_path) -> None:
    old_path = tmp_path / "mcts" / "chair" / "exp" / "result" / "updated2" / "mesh-a" / "bboxs_steps6"
    new_path = tmp_path / "mcts" / "chair" / "exp" / "result" / "updated3" / "mesh-a" / "bboxs_steps7"
    old_path.mkdir(parents=True)
    new_path.mkdir(parents=True)
    (old_path / "bbox0.obj").write_text("old", encoding="utf-8")
    (new_path / "bbox0.obj").write_text("new", encoding="utf-8")
    old_time = 1000.0
    new_time = 2000.0
    for path in [old_path, old_path / "bbox0.obj"]:
        os.utime(path, (old_time, old_time))
    for path in [new_path, new_path / "bbox0.obj"]:
        os.utime(path, (new_time, new_time))

    assert latest_bbox_dir(tmp_path / "mcts" / "chair", "mesh-a", since=1500.0) == new_path
    assert latest_bbox_dir(tmp_path / "mcts" / "chair", "mesh-a", since=2500.0) is None


def test_public_api_exposes_dry_run_pipeline(tmp_path: Path) -> None:
    mesh_root = tmp_path / "meshes" / "airplane"
    mesh_id = "mesh-a"
    (mesh_root / mesh_id).mkdir(parents=True)
    (mesh_root / mesh_id / "model.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )

    cfg = load_config(None)
    cfg["workspace"] = str(tmp_path / "runs")
    cfg["categories"] = [
        {
            "name": "airplane",
            "mesh_root": str(mesh_root),
            "meshes": [mesh_id],
            "tetra": {"epsilon": 0.003, "edge_length": 0.2},
        }
    ]
    records = smart.run_pipeline(
        cfg,
        stage="normalize",
        category="airplane",
        meshes=[mesh_id],
        dry_run=True,
    )

    assert len(records) == 1
    assert records[0]["stage"] == "normalize"
    assert records[0]["status"] in {"dry_run", "skipped"}


def test_cli_global_dry_run_is_not_overridden_by_stage_parser(tmp_path: Path, capsys) -> None:
    mesh_root = tmp_path / "meshes"
    mesh_id = "mesh-a"
    (mesh_root / mesh_id).mkdir(parents=True)
    (mesh_root / mesh_id / "model.obj").write_text("v 0 0 0\n", encoding="utf-8")
    workspace = tmp_path / "runs"
    tetra_dir = workspace / "tetra" / "meshes_raw_e0.004_l0.2" / mesh_id
    tetra_dir.mkdir(parents=True)
    (tetra_dir / "tetra.msh__sf.obj").write_text("v 0 0 0\n", encoding="utf-8")
    config_path = tmp_path / "dry_run_cli.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: dry_run_cli",
                f"workspace: {workspace}",
                "stages:",
                "  preseg: true",
                "tetra:",
                "  epsilon: 0.004",
                "  edge_length: 0.2",
                "preseg:",
                "  type: coacd",
                "normalization:",
                "  enabled: false",
                "categories:",
                "  - name: chair",
                f"    mesh_root: {mesh_root}",
                "    limit: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert smart_main(["--config", str(config_path), "--dry-run", "run"]) == 0
    summary = json.loads(capsys.readouterr().out)

    assert summary["preseg"] == {"dry_run": 1}
    assert not (tetra_dir / "coacd").exists()


def test_public_api_doctor_and_data_checks() -> None:
    status = smart.doctor("configs/smoke_5.yaml")
    data = smart.check_data("configs/smoke_5.yaml")

    assert "checks" in status
    assert "airplane" in data


def test_public_api_does_not_mutate_config_argument() -> None:
    cfg = smart.load("configs/smoke_5.yaml")
    original_iter = cfg["mcts"]["mcts_iter"]

    smart.workspace(cfg, overrides={"mcts": {"mcts_iter": original_iter + 1}})

    assert cfg["mcts"]["mcts_iter"] == original_iter
