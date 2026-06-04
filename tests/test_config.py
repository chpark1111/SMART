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
    mesh_tetra_dir,
    run_local_refine_mesh,
    run_macro_skill_mesh,
    run_merge_mesh,
    run_refine_mesh,
    run_mcts_mesh,
    run_native_pipelines,
    stage_root,
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
    assert "learned_frontier.yaml" in names
    assert "learned_auto_safe.yaml" in names
    assert "learned_macro_safe.yaml" in names
    assert "learned_macro_program_gate_top3.yaml" in names
    assert "learned_macro_refine_only.yaml" in names
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
    assert "learned_frontier.yaml" in names
    assert "learned_auto_safe.yaml" in names
    assert "learned_macro_safe.yaml" in names
    assert "learned_macro_refine_only.yaml" in names
    assert all("_experimental" not in name for name in names)


def test_public_api_exposes_documented_load_config_alias() -> None:
    cfg = smart.load_config("configs/learned_frontier.yaml")

    assert cfg["run_name"] == "learned_frontier"
    assert cfg["mcts"]["learned_prior"]["enabled"] is True


def test_public_api_exposes_documented_run_alias(tmp_path) -> None:
    mesh_dir = tmp_path / "data" / "airplane" / "mesh_a"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "model.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    cfg = smart.load_config(
        None,
        overrides={
            "workspace": str(tmp_path / "runs"),
            "categories": [
                {
                    "name": "airplane",
                    "mesh_root": str(tmp_path / "data" / "airplane"),
                    "meshes": ["mesh_a"],
                    "tetra": {"epsilon": 0.003, "edge_length": 0.2},
                }
            ],
        },
    )

    records = smart.run(cfg, stage="normalize", dry_run=True)

    assert records
    assert records[0]["stage"] == "normalize"
    assert records[0]["status"] == "dry_run"


def test_learned_frontier_config_enables_deepset_mcts_prior() -> None:
    cfg = load_config("configs/learned_frontier.yaml")

    assert cfg["run_name"] == "learned_frontier"
    assert cfg["engine"] == "cpp_native"
    assert cfg["mcts"]["backend"] == "cpp_native"
    assert cfg["mcts"]["direct_file_runner"] is True
    assert cfg["mcts"]["num_action_scale"] == 2
    assert cfg["mcts"]["learned_prior"] == {
        "enabled": True,
        "policy": "default",
        "mode": "guarded",
        "num_iter": None,
        "max_step": None,
        "transposition_table": True,
        "overrides": {},
    }


def test_learned_auto_safe_config_is_default_candidate_profile() -> None:
    cfg = load_config("configs/learned_auto_safe.yaml")

    assert cfg["run_name"] == "learned_auto_safe"
    assert cfg["engine"] == "cpp_native"
    assert cfg["mcts"]["learned_prior"]["enabled"] is True
    assert cfg["mcts"]["learned_prior"]["mode"] == "auto_safe"


def test_learned_macro_safe_config_enables_safe_macro_stage() -> None:
    cfg = load_config("configs/learned_macro_safe.yaml")

    assert cfg["run_name"] == "learned_macro_safe"
    assert cfg["engine"] == "cpp_native"
    assert cfg["mcts"]["learned_prior"]["enabled"] is True
    assert cfg["mcts"]["learned_prior"]["mode"] == "auto_safe"
    assert cfg["stages"]["macro_skill"] is True
    assert cfg["macro_skill"]["input_stage"] == "mcts"
    assert cfg["macro_skill"]["quality_preset"] == "balanced"
    assert cfg["macro_skill"]["native_executor"] is True
    assert cfg["macro_skill"]["planner"]["enabled"] is True
    assert cfg["macro_skill"]["planner"]["max_rounds"] == 3
    assert cfg["macro_skill"]["planner"]["profile_schedule"] == [
        "balanced",
        "learned_efficient",
        "quality",
    ]
    assert cfg["render"]["input_stage"] == "macro_skill"


def test_learned_macro_program_gate_top3_config_matches_stage_source_gate() -> None:
    cfg = load_config("configs/learned_macro_program_gate_top3.yaml")

    assert cfg["run_name"] == "learned_macro_program_gate_top3"
    assert cfg["engine"] == "cpp_native"
    assert cfg["stages"]["macro_skill"] is True
    assert cfg["macro_skill"]["input_stage"] == "mcts"
    assert cfg["macro_skill"]["quality_preset"] == "custom"
    assert cfg["macro_skill"]["top_k"] == 3
    assert cfg["macro_skill"]["exact_budget"] == 3
    assert cfg["macro_skill"]["macro_memory_pool_size"] == -1
    assert cfg["macro_skill"]["planner"]["enabled"] is False
    assert cfg["render"]["input_stage"] == "macro_skill"


def test_learned_macro_refine_only_config_enables_mcts_replacement_research_profile() -> None:
    cfg = load_config("configs/learned_macro_refine_only.yaml")

    assert cfg["run_name"] == "learned_macro_refine_only"
    assert cfg["engine"] == "cpp_native"
    assert cfg["stages"]["refine"] is True
    assert cfg["stages"]["mcts"] is False
    assert cfg["stages"]["macro_skill"] is True
    assert cfg["macro_skill"]["input_stage"] == "refine"
    assert cfg["macro_skill"]["planner"]["enabled"] is True
    assert cfg["macro_skill"]["planner"]["max_rounds"] == 3
    assert cfg["macro_skill"]["planner"]["profile_schedule"] == [
        "balanced",
        "learned_efficient",
        "quality",
    ]
    assert cfg["render"]["input_stage"] == "macro_skill"


def test_learned_prior_schema_defaults_to_guarded_when_enabled() -> None:
    cfg = smart.load_config(None, overrides={"mcts": {"learned_prior": {"enabled": True}}})

    assert cfg["mcts"]["learned_prior"]["enabled"] is True
    assert cfg["mcts"]["learned_prior"]["mode"] == "guarded"


def test_public_api_lists_and_resolves_packaged_assets() -> None:
    policies = smart.asset_profiles("policies")
    gates = smart.asset_profiles("gates")
    priors = smart.asset_profiles("priors")
    skills = smart.asset_profiles("skills")

    policy_names = {item["name"] for item in policies}
    assert "deepset_setaware_v2_h128_v1.smartmlp" in policy_names
    packaged_policy = next(item for item in policies if item["name"] == "deepset_setaware_v2_h128_v1.smartmlp")
    assert packaged_policy["feature_set"] == "setaware_v2"
    assert packaged_policy["model_type"] == "deepset_h128"
    assert gates == []
    assert priors == []
    assert {item["name"] for item in skills} >= {
        "macro_skill_knowledge_base_v1.json",
        "macro_memory_policy_v1.json",
        "macro_budget_quality_rule_v1.json",
    }
    assert smart.asset_path("policies", "default").name == "deepset_setaware_v2_h128_v1.smartmlp"
    assert smart.asset_path("policy", "h128_v1").name == "deepset_setaware_v2_h128_v1.smartmlp"
    assert smart.asset_path("skills", "macro_v1").name == "macro_skill_knowledge_base_v1.json"
    assert smart.asset_path("skills", "macro_memory_v1").name == "macro_memory_policy_v1.json"
    assert smart.asset_path("skills", "macro_budget_quality_v1").name == "macro_budget_quality_rule_v1.json"
    with pytest.raises(FileNotFoundError):
        smart.asset_path("gates", "rich")


def test_smart_cli_lists_packaged_assets(capsys) -> None:
    assert smart_main(["assets", "--kind", "gates", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload == []

    assert smart_main(["assets", "--kind", "policies", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {item["name"] for item in payload} == {"deepset_setaware_v2_h128_v1.smartmlp"}
    assert payload[0]["feature_set"] == "setaware_v2"
    assert payload[0]["model_type"] == "deepset_h128"

    assert smart_main(["assets", "--kind", "skills", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {item["name"] for item in payload} >= {
        "macro_skill_knowledge_base_v1.json",
        "macro_memory_policy_v1.json",
        "macro_budget_quality_rule_v1.json",
    }


def test_smart_cli_macro_skill_smoke(tmp_path, capsys) -> None:
    import smart.cpp as sc

    if not sc.using_cpp():
        pytest.skip("smart._cpp is not built")

    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    voxels = [[0, 1, 2, 3]]
    msh = tmp_path / "one_tet.msh"
    sc.native_save_gmsh(str(msh), vertices, sc.native_tetra_surface_faces(voxels), voxels)
    metadata = tmp_path / "bbox_params.json"
    metadata.write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "index": 0,
                        "bounds": [-0.3, -0.3, -0.3, 0.7, 0.7, 0.7],
                        "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "macro_result.json"
    bbox_dir = tmp_path / "bbox_out"

    assert (
        smart_main(
            [
                "macro-skill",
                "--msh",
                str(msh),
                "--bbox-metadata",
                str(metadata),
                "--category",
                "table",
                "--top-k",
                "2",
                "--candidate-count",
                "32",
                "--max-steps",
                "4",
                "--output",
                str(result_path),
                "--output-bbox-dir",
                str(bbox_dir),
                "--json",
            ]
        )
        == 0
    )
    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert stdout_payload["attempt_count"] == 2
    assert file_payload["attempt_count"] == 2
    assert stdout_payload["exact_validator"] == "native_smart_manifold"
    assert stdout_payload["rollback_on_failure"] is True
    assert stdout_payload["accepted_non_worse"] is True
    assert stdout_payload["deployment_status"] == "release_candidate_opt_in_post_refine"
    assert stdout_payload["exported_bbox_count"] == 1
    assert (bbox_dir / "bbox_params.json").exists()


def test_smart_cli_macro_skill_summary(capsys) -> None:
    assert smart_main(["macro-skill-summary", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "release_candidate_opt_in_post_refine"
    assert payload["default_smart_path"] == "unchanged_exact_cpp_native"
    assert payload["release_gate"]["can_ship_opt_in"] is True
    assert payload["release_gate"]["can_be_default"] is False
    assert payload["release_gate"]["implemented_release_requirements"]["pipeline_stage"] is True
    assert payload["deployment_default"]["post_refine_only"] is True
    assert payload["deployment_default"]["planner_enabled_in_learned_macro_safe"] is True
    assert payload["substructure_planner"]["status"] == "release_candidate_opt_in_substructure_planner"
    assert payload["substructure_planner"]["fresh_500_gate_passed"] is True
    assert payload["substructure_planner"]["fresh_generated_latest"]["cases"] == 507
    assert payload["substructure_planner"]["fresh_generated_latest"]["category_counts"] == {
        "airplane": 176,
        "chair": 164,
        "table": 167,
    }
    assert payload["substructure_planner"]["fresh_generated_latest"]["losses"] == 0
    assert payload["substructure_planner"]["fresh_generated_latest"]["top1_exact_attempt_reduction"] == 0.9375
    assert payload["substructure_planner"]["fresh_generated_latest"]["top3_exact_attempt_reduction"] == 0.8125
    assert payload["substructure_planner"]["fresh_generated_latest"]["learned_top3_mean_delta"] > (
        payload["substructure_planner"]["fresh_generated_latest"]["portfolio_mean_delta"]
    )
    assert payload["substructure_planner"]["stage_source_gate_passed"] is True
    stage_latest = payload["substructure_planner"]["stage_source_latest"]
    assert stage_latest["selector"] == "knowledge_base_program_gate"
    assert stage_latest["top_k"] == 3
    assert stage_latest["exact_budget"] == 3
    assert stage_latest["macro_memory_pool_size"] == -1
    assert stage_latest["refine"]["cases"] == 456
    assert stage_latest["refine"]["accepted"] == 456
    assert stage_latest["mcts"]["cases"] == 456
    assert stage_latest["mcts"]["accepted"] == 456
    assert stage_latest["refine"]["portfolio_ratio"] > 1.0
    assert stage_latest["mcts"]["portfolio_ratio"] > 1.0
    assert payload["substructure_planner"]["fresh_generated_seed13"]["losses"] == 0
    assert payload["mcts_replacement_agent"]["status"] == "research_only_not_default_ready"
    assert payload["mcts_replacement_agent"]["packaged_config"] == "configs/learned_macro_refine_only.yaml"
    assert payload["mcts_replacement_agent"]["fresh_refine_only_latest"]["cases"] == 76
    assert payload["mcts_replacement_agent"]["fresh_refine_only_latest"]["category_counts"] == {
        "airplane": 26,
        "chair": 24,
        "table": 26,
    }
    assert payload["mcts_replacement_agent"]["fresh_refine_only_latest"]["losses"] == 0
    assert "configs/learned_macro_safe.yaml" in payload["recommended_configs"]
    assert "configs/learned_macro_program_gate_top3.yaml" in payload["recommended_configs"]
    assert "configs/learned_macro_refine_only.yaml" in payload["recommended_configs"]
    assert payload["best_practical"]["losses_vs_conditional_budget_v1"] == 0
    assert payload["quality_preset"]["losses_vs_conditional_budget_v1"] == 0


def test_smart_cli_learned_router_summary(capsys) -> None:
    assert smart_main(["learned-router-summary", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "release_candidate_opt_in"
    assert payload["default_smart_path"] == "unchanged_exact_cpp_native"
    assert payload["packaged_policy"] == "deepset_setaware_v2_h128_v1.smartmlp"
    assert payload["release_gate"]["can_ship_opt_in"] is True
    assert payload["release_gate"]["can_be_default"] is False
    assert payload["validation_snapshot"]["refine_full_token_split"]["cases"] == 1015
    assert payload["validation_snapshot"]["refine_full_token_split"]["quality_losses"] == 0
    assert payload["validation_snapshot"]["refine_heldout_test"]["quality_losses"] == 0
    assert payload["validation_snapshot"]["macro_skill_replay"]["status"] == "release_candidate_opt_in_post_refine"
    assert payload["runtime_requirements"]["policy_asset_exists"] is True

    api_payload = smart.learned_router_profile_summary()
    assert api_payload["validation_snapshot"]["refine_replay_states"]["quality_losses"] == 0


def test_smart_cli_learned_release_readiness(capsys) -> None:
    assert smart_main(["learned-release-readiness", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "release_candidate_opt_in"
    assert payload["can_ship_opt_in"] is True
    assert payload["can_be_default"] is False
    assert payload["default_smart_path"] == "unchanged_exact_cpp_native"
    assert "learned_macro_safe.yaml" in payload["opt_in_profiles"]
    assert "learned_macro_program_gate_top3.yaml" in payload["opt_in_profiles"]
    assert "learned_macro_refine_only.yaml" in payload["opt_in_profiles"]
    assert payload["packaged_assets"]["policy_ready"] is True
    assert payload["packaged_assets"]["missing_skill_assets"] == []
    assert payload["packaged_assets"]["configs_ready"] is True
    assert payload["router"]["heldout_losses"] == 0
    assert payload["macro_skill"]["balanced_losses"] == 0
    assert payload["substructure_planner"]["status"] == "release_candidate_opt_in_substructure_planner"
    assert payload["substructure_planner"]["fresh_500_gate_passed"] is True
    assert payload["substructure_planner"]["fresh_generated_cases"] == 507
    assert payload["substructure_planner"]["fresh_generated_losses"] == 0
    assert payload["substructure_planner"]["fresh_generated_exact_attempt_reduction"] > 0.0
    assert payload["substructure_planner"]["fresh_generated_top3_delta_vs_portfolio"] > 0.0
    assert payload["substructure_planner"]["stage_source_gate_passed"] is True
    assert payload["substructure_planner"]["stage_source_latest"]["refine"]["accepted"] == 456
    assert payload["substructure_planner"]["stage_source_latest"]["mcts"]["accepted"] == 456
    assert payload["mcts_replacement_agent"]["status"] == "research_only_not_default_ready"
    assert payload["mcts_replacement_agent"]["packaged_config"] == "configs/learned_macro_refine_only.yaml"
    assert payload["mcts_replacement_agent"]["fresh_refine_only_latest"]["cases"] == 76
    assert payload["mcts_replacement_agent"]["fresh_refine_only_latest"]["losses"] == 0

    assert smart_main(["learned-release-readiness"]) == 0
    text = capsys.readouterr().out
    assert "status=release_candidate_opt_in" in text
    assert "opt_in=True" in text
    assert "can_be_default=False" in text
    assert smart_main(["learned-release-readiness", "--fail-if-not-ready"]) == 0
    assert smart_main(["learned-release-readiness", "--require-default-ready"]) == 3


def test_smart_cli_learned_release_readiness_fails_when_blocked(monkeypatch, capsys) -> None:
    import smart.api as smart_api

    def _blocked_summary() -> dict:
        return {
            "status": "blocked",
            "can_ship_opt_in": False,
            "can_be_default": False,
            "default_smart_path": "unchanged_exact_cpp_native",
            "opt_in_profiles": [],
            "router": {
                "heldout_losses": 1,
                "heldout_exact_call_reduction": 0.0,
            },
            "macro_skill": {
                "balanced_losses": 1,
                "balanced_mean_delta": -0.1,
            },
        }

    monkeypatch.setattr(smart_api, "learned_release_readiness_summary", _blocked_summary)
    assert smart_main(["learned-release-readiness", "--json", "--fail-if-not-ready"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert payload["can_ship_opt_in"] is False


def test_public_api_exposes_learned_release_readiness() -> None:
    payload = smart.learned_release_readiness_summary()

    assert payload["can_ship_opt_in"] is True
    assert payload["can_be_default"] is False
    assert payload["router"]["status"] == "release_candidate_opt_in"
    assert payload["macro_skill"]["status"] == "release_candidate_opt_in_post_refine"
    assert payload["substructure_planner"]["status"] == "release_candidate_opt_in_substructure_planner"
    assert payload["substructure_planner"]["fresh_500_gate_passed"] is True
    assert payload["substructure_planner"]["fresh_generated_cases"] == 507
    assert payload["mcts_replacement_agent"]["experimental_use"] == "refine_output_to_macro_planner_without_mcts"
    assert payload["mcts_replacement_agent"]["packaged_config"] == "configs/learned_macro_refine_only.yaml"
    assert callable(smart.run_macro_skill_controller)
    assert callable(smart.run_macro_skill_controller_from_files)
    assert callable(smart.run_macro_skill_planner)
    assert callable(smart.run_macro_skill_planner_from_files)
    assert callable(smart.run_builtin_macro_skill_planner)
    assert callable(smart.run_builtin_macro_skill_planner_from_files)


def test_macro_skill_assets_load_and_rank() -> None:
    from smart import macro_skills

    assets = macro_skills.builtin_macro_skill_assets()
    skills = macro_skills.load_skill_knowledge_base(assets.skill_kb)
    memory = macro_skills.load_macro_memory_policy(assets.memory_policy)
    budget_rules = macro_skills.load_budget_rules(assets.budget_rule)
    quality_gate = macro_skills.load_macro_quality_gate(assets.quality_gate)

    assert skills
    assert budget_rules
    thresholds = quality_gate["preset_thresholds"]
    assert set(thresholds) == {"learned_fast", "learned_efficient", "learned_quality"}
    assert thresholds["learned_fast"]["threshold"] > thresholds["learned_efficient"]["threshold"]
    assert thresholds["learned_efficient"]["threshold"] > thresholds["learned_quality"]["threshold"]
    assert quality_gate["threshold"] == thresholds["learned_efficient"]["threshold"]

    high_named = {}
    low_named = {}
    for name, mean, std, weight in zip(
        quality_gate["feature_names"],
        quality_gate["feature_mean"],
        quality_gate["feature_std"],
        quality_gate["weights"],
    ):
        if name.startswith("cat_"):
            continue
        direction = 1.0 if weight >= 0.0 else -1.0
        scale = max(abs(float(std)), 1.0)
        high_named[name] = float(mean) + direction * scale * 20.0
        low_named[name] = float(mean) - direction * scale * 20.0
    high_budget, high_decision = macro_skills.learned_macro_quality_budget(
        "chair",
        {"named": high_named},
        quality_gate,
        default_budget=1,
        max_budget=5,
        preset="learned_efficient",
    )
    low_budget, low_decision = macro_skills.learned_macro_quality_budget(
        "chair",
        {"named": low_named},
        quality_gate,
        default_budget=1,
        max_budget=5,
        preset="learned_efficient",
    )
    assert high_budget == 5
    assert high_decision["open_high_budget"] is True
    assert low_budget == 1
    assert low_decision["open_high_budget"] is False
    assert macro_skills.macro_skill_budget(
        "table",
        {"named": {"num_actions": 100}},
        budget_rules,
        max_budget=5,
    ) == 4
    assert macro_skills.macro_skill_budget(
        "table",
        {"named": {"num_actions": 200}},
        budget_rules,
        max_budget=5,
    ) == 1
    assert macro_skills.macro_skill_budget(
        "airplane",
        {"named": {"num_actions": 100}},
        budget_rules,
        max_budget=5,
    ) == 1

    ranked = macro_skills.rank_builtin_macro_skills(
        "table",
        skills_by_id=skills,
        memory_policy=memory,
        macro_memory_pool_size=5,
    )
    assert ranked
    assert ranked[0][0] in skills
    pure_program_ranked = macro_skills.rank_builtin_macro_skills(
        "table",
        skills_by_id=skills,
        memory_policy=memory,
        macro_memory_pool_size=-1,
    )
    expected_program_order = sorted(
        skills,
        key=lambda macro_id: macro_skills.program_gate_score(
            category="table",
            skill=skills[macro_id],
        ),
        reverse=True,
    )
    assert [macro_id for macro_id, _score in pure_program_ranked] == expected_program_order

    profile = smart.macro_skill_profile_summary()
    assert profile["packaged_profile"] == "geometry_top5_exact_guarded_variable_repeat_v2_balanced"
    assert profile["best_practical"]["wins_vs_conditional_budget_v1"] > 0
    assert profile["quality_preset"]["mean_delta"] >= profile["best_practical"]["mean_delta"]


def test_macro_skill_variable_repeat_and_guard_policy() -> None:
    from smart import macro_skills

    assert (
        macro_skills.macro_step_repeat_budget(
            {
                "observed_repeat": "2",
                "repeat_max": 16,
                "until": "coverage_margin_low_or_score_stalls",
            },
            max_steps=20,
        )
        == 16
    )
    assert (
        macro_skills.macro_step_repeat_budget(
            {
                "observed_repeat": "2",
                "repeat_max": 16,
                "until": "fixed_count",
            },
            max_steps=20,
        )
        == 2
    )
    assert (
        macro_skills.macro_step_repeat_budget(
            {
                "op": "recenter_box",
                "observed_repeat": "2",
                "repeat_max": 2,
                "until": "center_shift_stalls",
            },
            max_steps=20,
        )
        == 1
    )
    assert not macro_skills.allow_nonpositive_macro_step("shrink_face", -0.1)
    assert macro_skills.allow_nonpositive_macro_step("expand_face", -0.1)
    assert macro_skills.allow_nonpositive_macro_step("recenter", -1.0e-10)
    assert macro_skills.allow_nonpositive_macro_step("recenter_box", -1.0e-10)
    assert not macro_skills.allow_nonpositive_macro_step("recenter", -1.0e-3)
    assert not macro_skills.allow_nonpositive_macro_step("recenter_box", -1.0e-3)


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


def test_cpp_native_refine_learned_router_requires_extension(tmp_path, monkeypatch) -> None:
    from smart import native_runner

    monkeypatch.setattr(native_runner, "cpp_native_deepset_router_available", lambda: False)
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
            "direct_file_runner_required": True,
            "learned_router": {"enabled": True, "profile": "auto"},
        },
    }

    record = run_refine_mesh(
        cfg,
        {"name": "table", "mesh_root": str(mesh_root)},
        "mesh-a",
        dry_run=False,
        force=True,
    )

    assert record.status == "failed"
    assert "cpp_native refine file runner required but unavailable" in record.error


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


def test_cpp_native_mcts_direct_file_runner_accepts_deepset_prior(tmp_path, monkeypatch) -> None:
    from smart import native_runner

    monkeypatch.setattr(native_runner, "cpp_native_deepset_mcts_prior_available", lambda: True)
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
            "learned_prior": {
                "enabled": True,
                "mode": "guarded",
                "policy": "default",
                "num_iter": 25,
                "max_step": 4,
                "transposition_table": True,
            },
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
    assert record.command[:2] == ["smart._cpp", "run_builtin_deepset_prior_mcts"]
    assert "--mode" in record.command
    assert record.command[record.command.index("--mode") + 1] == "guarded"
    assert record.metadata["learned_mcts_prior"] is True
    assert record.metadata["learned_mcts_prior_mode"] == "guarded"
    assert "_deepsetmcts_guarded" in record.output_path


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


def _write_macro_skill_stage_inputs(
    tmp_path: Path,
    *,
    input_stage: str = "mcts",
) -> tuple[dict[str, object], dict[str, str]]:
    mesh_root = tmp_path / "meshes" / "table"
    mesh_model = mesh_root / "mesh-a" / "model.obj"
    mesh_model.parent.mkdir(parents=True)
    mesh_model.write_text("v 0 0 0\nf 1 1 1\n", encoding="utf-8")
    cfg = {
        "workspace": str(tmp_path),
        "tetra": {"epsilon": 0.004, "edge_length": 0.2},
        "normalization": {"enabled": False},
        "stages": {"normalize": False},
        "macro_skill": {
            "input_stage": input_stage,
            "quality_preset": "balanced",
            "repeat_mode": "guarded_variable",
            "top_k": 3,
            "candidate_count": 32,
            "max_steps": 4,
            "cover_penalty": 100,
            "pen_rate": 1.0,
            "num_action_scale": 2,
            "action_unit": 0.01,
            "volume_method": "mesh",
            "stateful_union_cache": True,
            "stateful_cache_capacity": 128,
            "native_executor": True,
            "target_aware_execution": False,
        },
        "categories": [],
    }
    category = {"name": "table", "mesh_root": str(mesh_root)}
    msh = mesh_tetra_dir(cfg, category, "mesh-a") / "tetra.msh"
    msh.parent.mkdir(parents=True)
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n", encoding="utf-8")
    bbox_dir = (
        stage_root(cfg, input_stage, category)
        / "exp"
        / "result"
        / "updated0"
        / "mesh-a"
        / "bboxs_steps0"
    )
    bbox_dir.mkdir(parents=True)
    (bbox_dir / "bbox0.obj").write_text("v 0 0 0\nf 1 1 1\n", encoding="utf-8")
    (bbox_dir / "bbox_params.json").write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "index": 0,
                        "bounds": [-0.3, -0.3, -0.3, 0.7, 0.7, 0.7],
                        "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return cfg, category


def test_macro_skill_stage_dry_run_uses_package_cli(tmp_path) -> None:
    cfg, category = _write_macro_skill_stage_inputs(tmp_path)

    record = run_macro_skill_mesh(cfg, category, "mesh-a", dry_run=True, force=True)

    assert record.stage == "macro_skill"
    assert record.status == "dry_run"
    assert record.command[:2] == ["smart", "macro-skill"]
    assert record.command[record.command.index("--quality-preset") + 1] == "balanced"
    assert record.command[record.command.index("--top-k") + 1] == "3"
    assert record.output_path is not None
    assert record.output_path.endswith("bboxs_steps0")
    assert record.metadata["input_stage"] == "mcts"
    assert record.metadata["safe_noop_fallback"] is True
    assert record.metadata["exact_validator"] == "native_smart_manifold"


def test_macro_skill_stage_dry_run_passes_program_gate_options(tmp_path) -> None:
    cfg, category = _write_macro_skill_stage_inputs(tmp_path)
    cfg["macro_skill"].update(
        {
            "quality_preset": "custom",
            "top_k": 3,
            "macro_memory_pool_size": -1,
            "exact_budget": 3,
        }
    )

    record = run_macro_skill_mesh(cfg, category, "mesh-a", dry_run=True, force=True)

    assert record.status == "dry_run"
    assert record.command[record.command.index("--quality-preset") + 1] == "custom"
    assert record.command[record.command.index("--top-k") + 1] == "3"
    assert record.command[record.command.index("--macro-memory-pool-size") + 1] == "-1"
    assert record.command[record.command.index("--exact-budget") + 1] == "3"
    assert "_mempool-1" in str(record.output_path)


def test_macro_skill_stage_can_use_native_pipeline_mcts_bbox(tmp_path) -> None:
    mesh_root = tmp_path / "meshes" / "table"
    cfg = {
        "workspace": str(tmp_path),
        "tetra": {"epsilon": 0.004, "edge_length": 0.2},
        "normalization": {"enabled": False},
        "stages": {"normalize": False},
        "macro_skill": {
            "input_stage": "mcts",
            "quality_preset": "custom",
            "repeat_mode": "guarded_variable",
            "top_k": 3,
            "macro_memory_pool_size": -1,
            "candidate_count": 32,
            "max_steps": 4,
            "exact_budget": 3,
            "cover_penalty": 100,
            "pen_rate": 1.0,
            "num_action_scale": 2,
            "action_unit": 0.01,
            "volume_method": "mesh",
            "stateful_union_cache": True,
            "stateful_cache_capacity": 128,
            "native_executor": True,
            "target_aware_execution": False,
        },
        "categories": [],
    }
    category = {"name": "table", "mesh_root": str(mesh_root)}
    msh = mesh_tetra_dir(cfg, category, "mesh-a") / "tetra.msh"
    msh.parent.mkdir(parents=True)
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n", encoding="utf-8")
    native_bbox = tmp_path / "native_pipeline" / "table" / "mesh-a" / "mcts_bboxs_steps0"
    native_bbox.mkdir(parents=True)
    (native_bbox / "bbox0.obj").write_text("v 0 0 0\nf 1 1 1\n", encoding="utf-8")
    (native_bbox / "bbox_params.json").write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "index": 0,
                        "bounds": [-0.3, -0.3, -0.3, 0.7, 0.7, 0.7],
                        "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    record = run_macro_skill_mesh(cfg, category, "mesh-a", dry_run=True, force=True)

    assert record.status == "dry_run"
    assert record.metadata["bbox_metadata_path"] == str(native_bbox / "bbox_params.json")
    assert record.command[record.command.index("--bbox-metadata") + 1] == str(native_bbox / "bbox_params.json")
    assert bbox_dir_for_render(cfg, category, "mesh-a", "mcts") == native_bbox


def test_macro_skill_stage_dry_run_can_use_substructure_planner(tmp_path) -> None:
    cfg, category = _write_macro_skill_stage_inputs(tmp_path)
    cfg["macro_skill"]["planner"] = {
        "enabled": True,
        "max_rounds": 2,
        "profile_schedule": ["balanced", "learned_efficient"],
        "min_round_delta": 1.0e-12,
        "stop_after_noop": True,
    }

    record = run_macro_skill_mesh(cfg, category, "mesh-a", dry_run=True, force=True)

    assert record.status == "dry_run"
    assert record.command[:2] == ["smart", "macro-skill"]
    assert "--planner" in record.command
    assert record.command[record.command.index("--planner-max-rounds") + 1] == "2"
    schedule_index = record.command.index("--planner-profile-schedule")
    assert record.command[schedule_index + 1 : schedule_index + 3] == ["balanced", "learned_efficient"]
    assert record.metadata["planner_enabled"] is True
    assert record.metadata["planner_max_rounds"] == 2


def test_macro_skill_stage_exports_noop_fallback(tmp_path, monkeypatch) -> None:
    cfg, category = _write_macro_skill_stage_inputs(tmp_path)

    class FakeEngine:
        def export_bbox_dir(self, output: str) -> int:
            out = Path(output)
            out.mkdir(parents=True, exist_ok=True)
            (out / "bbox0.obj").write_text("v 0 0 0\nf 1 1 1\n", encoding="utf-8")
            (out / "bbox_params.json").write_text('{"boxes":[]}', encoding="utf-8")
            return 1

    def fake_controller(**kwargs):
        return {
            "engine": FakeEngine(),
            "accepted": False,
            "accepted_non_worse": True,
            "score_delta": 0.0,
            "attempt_count": 3,
            "exact_budget": 1,
            "deployment_status": "release_candidate_opt_in_post_refine",
        }

    monkeypatch.setattr("smart.api.run_macro_skill_controller_from_files", fake_controller)

    record = run_macro_skill_mesh(cfg, category, "mesh-a", force=True)

    assert record.status == "success"
    assert record.output_path is not None
    output = Path(record.output_path)
    assert (output / "bbox0.obj").exists()
    assert (output / "bbox_params.json").exists()
    assert record.metadata["accepted"] is False
    assert record.metadata["accepted_non_worse"] is True
    assert record.metadata["safe_noop_fallback"] is True
    assert record.metadata["exported_bbox_count"] == 1
    result_path = Path(record.metadata["result_json"])
    assert json.loads(result_path.read_text(encoding="utf-8"))["accepted"] is False


def test_macro_skill_stage_uses_planner_api_when_enabled(tmp_path, monkeypatch) -> None:
    cfg, category = _write_macro_skill_stage_inputs(tmp_path)
    cfg["macro_skill"]["planner"] = {
        "enabled": True,
        "max_rounds": 2,
        "profile_schedule": ["balanced"],
        "min_round_delta": 1.0e-12,
        "stop_after_noop": True,
    }

    class FakeEngine:
        def export_bbox_dir(self, output: str) -> int:
            out = Path(output)
            out.mkdir(parents=True, exist_ok=True)
            (out / "bbox0.obj").write_text("v 0 0 0\nf 1 1 1\n", encoding="utf-8")
            (out / "bbox_params.json").write_text('{"boxes":[]}', encoding="utf-8")
            return 1

    def fake_planner(**kwargs):
        assert kwargs["max_rounds"] == 2
        assert kwargs["profile_schedule"] == ("balanced",)
        return {
            "engine": FakeEngine(),
            "accepted": True,
            "accepted_rounds": 1,
            "round_count": 1,
            "accepted_non_worse": True,
            "score_delta": 0.25,
            "attempt_count": 3,
            "exact_budget": 1,
            "deployment_status": "release_candidate_opt_in_substructure_planner",
        }

    monkeypatch.setattr("smart.api.run_macro_skill_planner_from_files", fake_planner)

    record = run_macro_skill_mesh(cfg, category, "mesh-a", force=True)

    assert record.status == "success"
    assert record.metadata["planner_enabled"] is True
    assert record.metadata["accepted_rounds"] == 1
    assert record.metadata["round_count"] == 1
    assert record.metadata["deployment_status"] == "release_candidate_opt_in_substructure_planner"


def test_render_can_use_macro_skill_stage_output(tmp_path) -> None:
    cfg, category = _write_macro_skill_stage_inputs(tmp_path, input_stage="macro_skill")

    assert bbox_dir_for_render(cfg, category, "mesh-a", "macro_skill") is not None


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
