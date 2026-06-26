"""SMART official Python package."""

from __future__ import annotations

__all__ = [
    "__version__",
    "build_action_prior_from_traces",
    "build_linear_action_prior_from_traces",
    "build_mlp_action_prior_from_traces",
    "build_policy_gradient_action_prior_from_traces",
    "build_policy_value_action_prior_from_traces",
    "build_rl_mlp_action_prior_from_traces",
    "asset_path",
    "asset_profiles",
    "cpp_native_available",
    "export_box_proposal_dataset",
    "NativeSmartEngine",
    "check_data",
    "compare_quality",
    "config_profiles",
    "doctor",
    "evaluate",
    "load_action_prior",
    "load_config",
    "load_local_refine_gate",
    "load_pruning_gate",
    "load",
    "learned_release_readiness_summary",
    "learned_router_profile_summary",
    "macro_skill_profile_summary",
    "native_executable_path",
    "quality_gain_score",
    "predict_box_proposals",
    "rank_builtin_macro_skills",
    "run_macro_skill_controller",
    "run_macro_skill_controller_from_files",
    "run_macro_skill_planner",
    "run_macro_skill_planner_from_files",
    "run_native_pipeline",
    "run_builtin_macro_skill_controller",
    "run_builtin_macro_skill_controller_from_files",
    "run_builtin_macro_skill_planner",
    "run_builtin_macro_skill_planner_from_files",
    "run",
    "run_agent",
    "run_pipeline",
    "score_local_refine_gate",
    "score_pruning_gate",
    "select_quality_guarded_run",
    "train_box_proposal_model",
    "train_local_refine_gate",
    "train_pruning_gate",
    "workspace",
]

__version__ = "0.1.22"
_API_EXPORTS = {
    "build_action_prior_from_traces",
    "build_linear_action_prior_from_traces",
    "build_mlp_action_prior_from_traces",
    "build_policy_gradient_action_prior_from_traces",
    "build_policy_value_action_prior_from_traces",
    "build_rl_mlp_action_prior_from_traces",
    "asset_path",
    "asset_profiles",
    "cpp_native_available",
    "export_box_proposal_dataset",
    "check_data",
    "compare_quality",
    "config_profiles",
    "doctor",
    "evaluate",
    "load_action_prior",
    "load_config",
    "load_local_refine_gate",
    "load_pruning_gate",
    "load",
    "learned_release_readiness_summary",
    "learned_router_profile_summary",
    "macro_skill_profile_summary",
    "native_executable_path",
    "quality_gain_score",
    "predict_box_proposals",
    "rank_builtin_macro_skills",
    "run_macro_skill_controller",
    "run_macro_skill_controller_from_files",
    "run_macro_skill_planner",
    "run_macro_skill_planner_from_files",
    "run_builtin_macro_skill_controller",
    "run_builtin_macro_skill_controller_from_files",
    "run_builtin_macro_skill_planner",
    "run_builtin_macro_skill_planner_from_files",
    "run_native_pipeline",
    "run",
    "run_agent",
    "run_pipeline",
    "score_local_refine_gate",
    "score_pruning_gate",
    "select_quality_guarded_run",
    "train_box_proposal_model",
    "train_local_refine_gate",
    "train_pruning_gate",
    "workspace",
}


def __getattr__(name: str):
    if name == "build_action_prior_from_traces":
        from .action_prior import build_action_prior_from_traces

        globals()[name] = build_action_prior_from_traces
        return build_action_prior_from_traces
    if name == "build_linear_action_prior_from_traces":
        from .action_prior import build_linear_action_prior_from_traces

        globals()[name] = build_linear_action_prior_from_traces
        return build_linear_action_prior_from_traces
    if name == "build_mlp_action_prior_from_traces":
        from .action_prior import build_mlp_action_prior_from_traces

        globals()[name] = build_mlp_action_prior_from_traces
        return build_mlp_action_prior_from_traces
    if name == "build_policy_gradient_action_prior_from_traces":
        from .action_prior import build_policy_gradient_action_prior_from_traces

        globals()[name] = build_policy_gradient_action_prior_from_traces
        return build_policy_gradient_action_prior_from_traces
    if name == "build_policy_value_action_prior_from_traces":
        from .action_prior import build_policy_value_action_prior_from_traces

        globals()[name] = build_policy_value_action_prior_from_traces
        return build_policy_value_action_prior_from_traces
    if name == "build_rl_mlp_action_prior_from_traces":
        from .action_prior import build_rl_mlp_action_prior_from_traces

        globals()[name] = build_rl_mlp_action_prior_from_traces
        return build_rl_mlp_action_prior_from_traces
    if name == "load_action_prior":
        from .action_prior import load_action_prior

        globals()[name] = load_action_prior
        return load_action_prior
    if name == "native_executable_path":
        from .native_executable import native_executable_path

        globals()[name] = native_executable_path
        return native_executable_path
    if name == "NativeSmartEngine":
        from .cpp import NativeSmartEngine

        globals()[name] = NativeSmartEngine
        return NativeSmartEngine
    if name == "cpp_native_available":
        def cpp_native_available() -> bool:
            try:
                from . import cpp

                return bool(cpp.native_core_available() and cpp.NativeSmartEngine is not None)
            except Exception:
                return False

        globals()[name] = cpp_native_available
        return cpp_native_available
    if name == "export_box_proposal_dataset":
        from .box_proposal import export_box_proposal_dataset

        globals()[name] = export_box_proposal_dataset
        return export_box_proposal_dataset
    if name == "train_box_proposal_model":
        from .box_proposal import train_box_proposal_model

        globals()[name] = train_box_proposal_model
        return train_box_proposal_model
    if name == "predict_box_proposals":
        from .box_proposal import predict_box_proposals

        globals()[name] = predict_box_proposals
        return predict_box_proposals
    if name == "train_local_refine_gate":
        from .local_refine_gate import train_local_refine_gate

        globals()[name] = train_local_refine_gate
        return train_local_refine_gate
    if name == "load_local_refine_gate":
        from .local_refine_gate import load_local_refine_gate

        globals()[name] = load_local_refine_gate
        return load_local_refine_gate
    if name == "score_local_refine_gate":
        from .local_refine_gate import score_local_refine_gate

        globals()[name] = score_local_refine_gate
        return score_local_refine_gate
    if name == "train_pruning_gate":
        from .candidate_pruning_gate import train_pruning_gate

        globals()[name] = train_pruning_gate
        return train_pruning_gate
    if name == "load_pruning_gate":
        from .candidate_pruning_gate import load_pruning_gate

        globals()[name] = load_pruning_gate
        return load_pruning_gate
    if name == "score_pruning_gate":
        from .candidate_pruning_gate import score_pruning_gate

        globals()[name] = score_pruning_gate
        return score_pruning_gate
    if name == "rank_builtin_macro_skills":
        from .macro_skills import rank_builtin_macro_skills

        globals()[name] = rank_builtin_macro_skills
        return rank_builtin_macro_skills
    if name == "run_builtin_macro_skill_controller":
        from .macro_skills import run_builtin_macro_skill_controller

        globals()[name] = run_builtin_macro_skill_controller
        return run_builtin_macro_skill_controller
    if name == "run_macro_skill_controller":
        from .api import run_macro_skill_controller

        globals()[name] = run_macro_skill_controller
        return run_macro_skill_controller
    if name == "run_builtin_macro_skill_planner":
        from .macro_skills import run_builtin_macro_skill_planner

        globals()[name] = run_builtin_macro_skill_planner
        return run_builtin_macro_skill_planner
    if name == "run_macro_skill_planner":
        from .api import run_macro_skill_planner

        globals()[name] = run_macro_skill_planner
        return run_macro_skill_planner
    if name == "run_builtin_macro_skill_controller_from_files":
        from .macro_skills import run_builtin_macro_skill_controller_from_files

        globals()[name] = run_builtin_macro_skill_controller_from_files
        return run_builtin_macro_skill_controller_from_files
    if name == "run_macro_skill_controller_from_files":
        from .api import run_macro_skill_controller_from_files

        globals()[name] = run_macro_skill_controller_from_files
        return run_macro_skill_controller_from_files
    if name == "run_builtin_macro_skill_planner_from_files":
        from .macro_skills import run_builtin_macro_skill_planner_from_files

        globals()[name] = run_builtin_macro_skill_planner_from_files
        return run_builtin_macro_skill_planner_from_files
    if name == "run_macro_skill_planner_from_files":
        from .api import run_macro_skill_planner_from_files

        globals()[name] = run_macro_skill_planner_from_files
        return run_macro_skill_planner_from_files
    if name == "macro_skill_profile_summary":
        from .macro_skills import macro_skill_profile_summary

        globals()[name] = macro_skill_profile_summary
        return macro_skill_profile_summary
    if name == "compare_quality":
        from .quality import compare_quality

        globals()[name] = compare_quality
        return compare_quality
    if name == "quality_gain_score":
        from .quality import quality_gain_score

        globals()[name] = quality_gain_score
        return quality_gain_score
    if name == "select_quality_guarded_run":
        from .quality import select_quality_guarded_run

        globals()[name] = select_quality_guarded_run
        return select_quality_guarded_run
    if name in _API_EXPORTS:
        from . import api

        value = getattr(api, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
