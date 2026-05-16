from __future__ import annotations

import os

from smart.pipeline.config import load_config
from smart.pipeline.stages import (
    bbox_dir_for_render,
    inspect_tetra_output,
    latest_bbox_dir,
    list_mesh_ids,
    run_local_refine_mesh,
    run_mcts_mesh,
    validate_tetra_output,
)
from smart.pipeline import tools as pipeline_tools
from smart.pipeline.tools import _first_pymanifold_binary, build_rust_extension, diagnose_environment
from smart.cli import _apply_override
from smart.evaluation import EvaluationRecord, summarize_records
import smart


def test_demo_config_loads() -> None:
    cfg = load_config("configs/demo.yaml")
    assert cfg["run_name"] == "demo"
    assert [category["name"] for category in cfg["categories"]] == ["airplane", "chair", "table"]


def test_accelerated_search_profile_is_opt_in() -> None:
    cfg = load_config("configs/accelerated_search_experimental.yaml")

    assert cfg["run_name"] == "accelerated_search_experimental"
    assert cfg["mcts"]["backend"] == "rust_stateful"
    assert cfg["mcts"]["reward_backend"] == "manifold_stateful"
    assert cfg["mcts"]["action_prior_weight"] == 0.1
    assert cfg["mcts"]["action_prior_path"] == "smart/assets/priors/category_general_expanded_full_mlp_prior.json"
    assert cfg["mcts"]["allow_search_order_changes"] is True


def test_accelerated_exact_profile_keeps_legacy_manifold_reward() -> None:
    cfg = load_config("configs/accelerated_exact.yaml")

    assert cfg["run_name"] == "accelerated_exact"
    assert cfg["refine"]["reward_backend"] == "manifold"
    assert cfg["mcts"]["reward_backend"] == "manifold"
    assert cfg["refine"]["manifold_volume_method"] == "mesh"
    assert cfg["mcts"]["manifold_volume_method"] == "mesh"
    assert cfg["refine"]["backend"] == "auto"
    assert cfg["mcts"]["backend"] == "auto"
    assert cfg["mcts"]["allow_search_order_changes"] is False


def test_stateful_exact_profile_is_experimental_opt_in() -> None:
    cfg = load_config("configs/stateful_exact_experimental.yaml")

    assert cfg["run_name"] == "stateful_exact_experimental"
    assert cfg["refine"]["backend"] == "rust_stateful"
    assert cfg["mcts"]["backend"] == "auto"
    assert cfg["refine"]["reward_backend"] == "manifold_stateful"
    assert cfg["mcts"]["reward_backend"] == "manifold_stateful"
    assert cfg["refine"]["stateful_union_cache"] is False
    assert cfg["mcts"]["stateful_union_cache"] is False
    assert cfg["mcts"]["allow_search_order_changes"] is False


def test_candidate_bitset_profile_keeps_legacy_mcts_tree() -> None:
    cfg = load_config("configs/candidate_bitset_exact_experimental.yaml")

    assert cfg["run_name"] == "candidate_bitset_exact_experimental"
    assert cfg["refine"]["reward_backend"] == "manifold_stateful"
    assert cfg["mcts"]["reward_backend"] == "manifold_stateful"
    assert cfg["mcts"]["backend"] == "auto"
    assert cfg["mcts"]["candidate_backend"] == "bitset_topk"
    assert cfg["mcts"]["candidate_top_k"] == 8
    assert cfg["mcts"]["stateful_union_cache"] is False
    assert cfg["mcts"]["allow_search_order_changes"] is False


def test_candidate_bitset_fast_profile_uses_smaller_topk() -> None:
    cfg = load_config("configs/candidate_bitset_fast_experimental.yaml")

    assert cfg["run_name"] == "candidate_bitset_fast_experimental"
    assert cfg["refine"]["reward_backend"] == "manifold_stateful"
    assert cfg["mcts"]["reward_backend"] == "manifold_stateful"
    assert cfg["refine"]["candidate_backend"] == "bitset_topk"
    assert cfg["mcts"]["candidate_backend"] == "bitset_topk"
    assert cfg["refine"]["candidate_top_k"] == 3
    assert cfg["mcts"]["candidate_top_k"] == 3
    assert cfg["mcts"]["backend"] == "auto"
    assert cfg["mcts"]["allow_search_order_changes"] is False


def test_stateful_union_cache_profile_keeps_search_order_locked() -> None:
    cfg = load_config("configs/stateful_union_cache_experimental.yaml")

    assert cfg["run_name"] == "stateful_union_cache_experimental"
    assert cfg["refine"]["reward_backend"] == "manifold_stateful"
    assert cfg["mcts"]["reward_backend"] == "manifold_stateful"
    assert cfg["refine"]["stateful_union_cache"] is True
    assert cfg["mcts"]["stateful_union_cache"] is True
    assert cfg["refine"]["candidate_backend"] == "exact"
    assert cfg["mcts"]["candidate_backend"] == "exact"
    assert cfg["mcts"]["backend"] == "auto"
    assert cfg["mcts"]["allow_search_order_changes"] is False


def test_properties_volume_profile_is_opt_in() -> None:
    cfg = load_config("configs/properties_volume_experimental.yaml")

    assert cfg["run_name"] == "properties_volume_experimental"
    assert cfg["refine"]["reward_backend"] == "manifold_stateful"
    assert cfg["mcts"]["reward_backend"] == "manifold_stateful"
    assert cfg["refine"]["manifold_volume_method"] == "properties"
    assert cfg["mcts"]["manifold_volume_method"] == "properties"
    assert cfg["mcts"]["allow_search_order_changes"] is False


def test_hybrid_local_search_profile_runs_after_mcts() -> None:
    cfg = load_config("configs/hybrid_local_search_experimental.yaml")

    assert cfg["run_name"] == "hybrid_local_search_experimental"
    assert cfg["stages"]["local_refine"] is True
    assert cfg["local_refine"]["input_stage"] == "mcts"
    assert cfg["local_refine"]["bbox_init"] == "bbox_direct"
    assert cfg["local_refine"]["action_unit"] == 0.005
    assert cfg["render"]["input_stage"] == "local_refine"


def test_expanded_processed_smoke_uses_existing_workspace_and_balanced_meshes() -> None:
    cfg = load_config("configs/expanded_processed_smoke.yaml")

    assert cfg["run_name"] == "expanded_processed_smoke"
    assert cfg["workspace"] == "runs/expanded_200"
    counts = {category["name"]: len(list_mesh_ids(category)) for category in cfg["categories"]}
    assert counts == {"airplane": 3, "chair": 3, "table": 3}
    assert cfg["mcts"]["mcts_iter"] == 20
    assert cfg["mcts"]["max_step"] == 20
    assert cfg["mcts"]["allow_search_order_changes"] is False


def test_expanded_processed_16_uses_all_current_processed_meshes() -> None:
    cfg = load_config("configs/expanded_processed_16.yaml")

    assert cfg["run_name"] == "expanded_processed_16"
    assert cfg["workspace"] == "runs/expanded_200"
    counts = {category["name"]: len(list_mesh_ids(category)) for category in cfg["categories"]}
    assert counts == {"airplane": 6, "chair": 5, "table": 5}
    assert cfg["mcts"]["mcts_iter"] == 20
    assert cfg["mcts"]["max_step"] == 20
    assert cfg["mcts"]["allow_search_order_changes"] is False


def test_demo_sample_counts_are_limited() -> None:
    cfg = load_config("configs/demo.yaml")
    counts = {category["name"]: len(list_mesh_ids(category)) for category in cfg["categories"]}
    assert counts == {"airplane": 50, "chair": 50, "table": 50}


def test_smoke_config_uses_explicit_meshes() -> None:
    cfg = load_config("configs/smoke_5.yaml")
    counts = {category["name"]: len(list_mesh_ids(category)) for category in cfg["categories"]}
    assert counts == {"airplane": 2, "chair": 2, "table": 1}


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


def test_rust_mcts_backend_requires_search_order_opt_in(tmp_path) -> None:
    cfg = {
        "workspace": str(tmp_path),
        "normalization": {"enabled": False},
        "mcts": {"backend": "rust_stateful"},
    }

    record = run_mcts_mesh(cfg, {"name": "table"}, "mesh-a")

    assert record.status == "blocked"
    assert "backend=rust/rust_stateful" in record.error


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
    assert "smart-rust-extension" in checks
    assert checks["smart-rust-extension"]["kind"] == "python-extension"
    assert "vendored-manifold-source" in checks
    assert checks["vendored-manifold-source"]["detail"] == (
        "kept as fixed C++ binding source; do not pull or replace"
    )
    assert isinstance(status["required_failures"], list)
    assert isinstance(status["optional_failures"], list)


def test_pymanifold_binary_probe(tmp_path) -> None:
    binary = tmp_path / "pymanifold.cpython-39-darwin.so"
    binary.write_text("", encoding="utf-8")

    assert _first_pymanifold_binary(tmp_path) == binary


def test_build_rust_requires_source_checkout(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline_tools, "REPO_ROOT", tmp_path / "installed_package")

    messages = build_rust_extension({"workspace": str(tmp_path / "runs")})

    assert messages[0] == "smart-bbox maturin build: failed rc=127"
    assert "requires a SMART source checkout" in messages[1]


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


def test_public_api_exposes_dry_run_pipeline() -> None:
    records = smart.run_pipeline(
        "configs/smoke_5.yaml",
        stage="normalize",
        category="airplane",
        meshes=["1f5537f4747ec847622c69c3abc6f80"],
        dry_run=True,
        overrides={"workspace": "runs/api_test"},
    )

    assert len(records) == 1
    assert records[0]["stage"] == "normalize"
    assert records[0]["status"] in {"dry_run", "skipped"}


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
