"""Release-candidate opt-in variable-length macro-skill controller for SMART.

This module packages the current research controller behind a small opt-in API
and pipeline stage.  It does not replace the default exact C++ SMART path.  A
caller passes an active ``NativeSmartEngine`` instance, and every accepted
update is still validated by the engine's exact reward backend.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MacroSkillAssets:
    """Resolved packaged assets for the built-in macro-skill controller."""

    skill_kb: Path
    memory_policy: Path
    budget_rule: Path
    quality_gate: Path


def builtin_macro_skill_assets() -> MacroSkillAssets:
    """Return packaged macro-skill research assets."""

    from .api import asset_path

    return MacroSkillAssets(
        skill_kb=asset_path("skills", "macro_v1"),
        memory_policy=asset_path("skills", "macro_memory_v1"),
        budget_rule=asset_path("skills", "macro_budget_quality_v1"),
        quality_gate=asset_path("skills", "macro_quality_gate_ridge_v1"),
    )


def load_builtin_macro_skill_policy() -> dict[str, Any]:
    """Load the built-in knowledge base, memory policy, and budget rule."""

    assets = builtin_macro_skill_assets()
    return {
        "assets": assets,
        "skills_by_id": load_skill_knowledge_base(assets.skill_kb),
        "memory_policy": load_macro_memory_policy(assets.memory_policy),
        "budget_rules": load_budget_rules(assets.budget_rule),
        "quality_gate": load_macro_quality_gate(assets.quality_gate),
    }


def macro_skill_profile_summary() -> dict[str, Any]:
    """Return the current packaged macro-skill release profile numbers.

    These are benchmark-result summaries, not runtime guarantees. They document
    why the packaged rule can ship as an opt-in post-refine controller while
    keeping the default SMART path conservative.
    """

    return {
        "status": "release_candidate_opt_in_post_refine",
        "default_smart_path": "unchanged_exact_cpp_native",
        "packaged_profile": "geometry_top5_exact_guarded_variable_repeat_v2_balanced",
        "release_gate": {
            "can_ship_opt_in": True,
            "can_be_default": False,
            "default_blockers": [
                "paper reproduction default must remain unchanged unless the learned controller is explicitly selected",
                "full-pipeline learned-router plus macro-skill default promotion still needs mesh-level validation together",
                "MCTS replacement claim still needs larger full-pipeline validation beyond the current stage-source replay gates",
            ],
            "promotion_requirements": {
                "min_replay_states": 500,
                "max_exact_score_loss_cases": 0,
                "min_mean_quality_gain_vs_balanced": 0.0,
                "fallback_contract_required": True,
                "pipeline_stage_required": True,
            },
            "implemented_release_requirements": {
                "python_api": True,
                "cli": True,
                "pipeline_stage": True,
                "noop_fallback_export": True,
                "native_skill_executor": True,
                "exact_rollback_contract": True,
                "fresh_replay_500_zero_regression": True,
                "fresh_replay_500_cases": 507,
                "refine_stage_program_gate_456_zero_regression": True,
                "mcts_stage_program_gate_456_zero_regression": True,
            },
        },
        "exact_reward_contract": [
            "skills are proposals, not reward replacements",
            "every attempted skill is scored by exact native SMART/Manifold",
            "accepted skills must be non-worse than the input state",
            "failed or worsening skills rollback to the input state",
        ],
        "deployment_default": {
            "recommended_preset": "balanced",
            "quality_preset": "learned_efficient",
            "native_executor": True,
            "stateful_union_cache": True,
            "post_refine_only": True,
            "planner_enabled_in_learned_macro_safe": True,
        },
        "recommended_configs": [
            "configs/learned_macro_safe.yaml",
            "configs/learned_macro_program_gate_top3.yaml",
            "configs/learned_macro_refine_only.yaml",
        ],
        "recommended_commands": [
            "smart macro-skill-summary --json",
            "smart --config configs/learned_macro_safe.yaml run",
            "smart --config configs/learned_macro_program_gate_top3.yaml run",
            "smart --config configs/learned_macro_refine_only.yaml run",
            "smart macro-skill --planner --planner-max-rounds 3 --planner-profile-schedule balanced learned_efficient quality ...",
            "smart --config configs/smoke_5.yaml --set stages.macro_skill=true --set render.input_stage=macro_skill run",
        ],
        "substructure_planner": {
            "status": "release_candidate_opt_in_substructure_planner",
            "packaged_config": "configs/learned_macro_safe.yaml",
            "profile_schedule": ["balanced", "learned_efficient", "quality"],
            "max_rounds": 3,
            "acceptance_rule": "commit best exact SMART/Manifold non-worse macro-skill update per round",
            "fallback": "export unchanged bbox state when no skill improves",
            "fresh_generated_seed13": {
                "source": "runs/learned_macro_500_20260603",
                "cases": 13,
                "losses": 0,
                "portfolio_exact_attempts": 16,
                "learned_top3_exact_attempts": 3,
                "exact_attempt_reduction": 0.8125,
                "portfolio_mean_delta": 2.8538429317865424,
                "learned_top3_mean_delta": 3.957335824896668,
                "portfolio_ratio": 1.3866691053033269,
            },
            "fresh_generated_latest": {
                "source": "runs/learned_macro_500_20260603 + runs/learned_macro_extra_180_20260604",
                "report": (
                    "experiments/macro_search/runs/parameterized_skills_4k/"
                    "fresh_matched_learned_macro_combined_500_20260604_134819/fresh_matched_report.json"
                ),
                "cases": 507,
                "category_counts": {"airplane": 176, "chair": 164, "table": 167},
                "losses": 0,
                "portfolio_exact_attempts": 16,
                "learned_top1_exact_attempts": 1,
                "learned_top3_exact_attempts": 3,
                "top1_exact_attempt_reduction": 0.9375,
                "top3_exact_attempt_reduction": 0.8125,
                "portfolio_mean_delta": 1.4056951911381037,
                "learned_top1_mean_delta": 1.6807023785940503,
                "learned_top3_mean_delta": 1.8175371987451336,
                "top1_portfolio_ratio": 1.195637851782996,
                "top3_portfolio_ratio": 1.292981017651193,
                "top1_delta_vs_portfolio": 0.27500718745594654,
                "top3_delta_vs_portfolio": 0.41184200760702994,
                "top3_oracle_regret": 0.0,
                "positive_rate": 1.0,
                "top1_mean_elapsed_s": 0.3883918590670615,
                "top3_mean_elapsed_s": 0.8065568474043385,
                "top1_skill_counts": {
                    "recenter_then_shrink": 244,
                    "escape_local_minimum_expand_then_refine": 176,
                    "shrink_slack_face": 87,
                },
                "top3_skill_counts": {
                    "recenter_then_shrink": 255,
                    "escape_local_minimum_expand_then_refine": 175,
                    "shrink_slack_face": 77,
                },
            },
            "fresh_500_gate_passed": True,
            "stage_source_latest": {
                "selector": "knowledge_base_program_gate",
                "config": "configs/learned_macro_program_gate_top3.yaml",
                "top_k": 3,
                "exact_budget": 3,
                "macro_memory_pool_size": -1,
                "exact_attempt_reduction": 0.8125,
                "refine": {
                    "source": "runs/learned_macro_500_20260603/refine + runs/learned_macro_extra_180_20260604/refine",
                    "report": (
                        "experiments/macro_search/runs/parameterized_skills_4k/"
                        "fresh_matched_refine_source_456_20260604/kb_program_gate_top3_best_positive.jsonl.summary.json"
                    ),
                    "cases": 456,
                    "accepted": 456,
                    "positive_rate": 1.0,
                    "portfolio_cases": 456,
                    "portfolio_wins": 455,
                    "portfolio_mean_delta": 0.7236037912762976,
                    "learned_mean_delta": 1.2881294361511488,
                    "delta_vs_portfolio": 0.5645256448748512,
                    "portfolio_ratio": 1.7801584951332785,
                    "mean_elapsed_s": 0.30388034530263214,
                    "category_counts": {"airplane": 165, "chair": 137, "table": 154},
                },
                "mcts": {
                    "source": "runs/learned_macro_500_20260603/mcts + runs/learned_macro_extra_180_20260604/mcts",
                    "report": (
                        "experiments/macro_search/runs/parameterized_skills_4k/"
                        "fresh_matched_mcts_source_456_20260604/kb_program_gate_top3_best_positive.jsonl.summary.json"
                    ),
                    "cases": 456,
                    "accepted": 456,
                    "positive_rate": 1.0,
                    "portfolio_cases": 456,
                    "portfolio_wins": 455,
                    "portfolio_mean_delta": 0.7233593688824418,
                    "learned_mean_delta": 1.2876468120268543,
                    "delta_vs_portfolio": 0.5642874431444125,
                    "portfolio_ratio": 1.7800928105987093,
                    "mean_elapsed_s": 0.2780545692763157,
                    "category_counts": {"airplane": 165, "chair": 137, "table": 154},
                },
            },
            "stage_source_gate_passed": True,
            "default_blocker": (
                "passed 507-case fresh strict replay gate and 456-case refine/MCTS "
                "stage-source gates; still opt-in until learned router and macro-skill "
                "are validated together on full mesh-level pipeline runs as the package default"
            ),
        },
        "mcts_replacement_agent": {
            "status": "research_only_not_default_ready",
            "current_safe_use": "post_mcts_polishing",
            "experimental_use": "refine_output_to_macro_planner_without_mcts",
            "packaged_config": "configs/learned_macro_refine_only.yaml",
            "command": "smart --config configs/learned_macro_refine_only.yaml run",
            "fresh_refine_only_latest": {
                "source": "runs/learned_macro_500_20260603/refine",
                "report": (
                    "experiments/macro_search/runs/parameterized_skills_4k/"
                    "fresh_matched_refine_only_replacement_seed76_3cat_20260604/fresh_matched_report.json"
                ),
                "cases": 76,
                "category_counts": {"airplane": 26, "chair": 24, "table": 26},
                "losses": 0,
                "portfolio_exact_attempts": 16,
                "learned_top1_exact_attempts": 1,
                "learned_top3_exact_attempts": 3,
                "top1_exact_attempt_reduction": 0.9375,
                "top3_exact_attempt_reduction": 0.8125,
                "portfolio_mean_delta": 1.494923735774852,
                "learned_top1_mean_delta": 1.9534924644994305,
                "learned_top3_mean_delta": 2.110346960672846,
                "top3_portfolio_ratio": 1.4116753317713606,
                "top3_oracle_regret": 0.0,
            },
            "default_blockers": [
                "456-case refine-source and MCTS-source replay gates pass with zero rejected positive states using program-gate top3",
                "still needs 500+ mesh-level full-pipeline validation before replacing MCTS by default",
                "needs direct wall-time and final metric comparison against full historical MCTS, not only post-stage replay",
            ],
            "acceptance_rule": "same exact SMART/Manifold non-worse validation as post-MCTS planner",
        },
        "heldout_cases": 173,
        "previous_best": {
            "profile": "geometry_top2_exact_macro_memory",
            "mean_delta": 1.6015182307030793,
            "mean_exact_attempts": 1.1965317919075145,
            "elapsed_sec": 0.19453924253179192,
        },
        "best_practical": {
            "profile": "geometry_top5_exact_guarded_variable_repeat_v2_balanced",
            "mean_delta": 1.6118304945341573,
            "mean_exact_attempts": 1.5895953757225434,
            "elapsed_sec": 0.2261333335433524,
            "wins_vs_conditional_budget_v1": 101,
            "ties_vs_conditional_budget_v1": 72,
            "losses_vs_conditional_budget_v1": 0,
            "mean_gain_vs_conditional_budget_v1": 0.005864113562165709,
        },
        "quality_preset": {
            "profile": "geometry_top5_exact_guarded_variable_repeat_v2_quality_budget5_steps32",
            "mean_delta": 1.6638987982429452,
            "mean_exact_attempts": 5.0,
            "elapsed_sec": 0.8151633385838148,
            "wins_vs_conditional_budget_v1": 145,
            "ties_vs_conditional_budget_v1": 28,
            "losses_vs_conditional_budget_v1": 0,
            "mean_gain_vs_conditional_budget_v1": 0.05793241727095284,
        },
        "efficient_quality_preset": {
            "profile": "geometry_top5_exact_guarded_variable_repeat_v2_efficient_chair_budget5_steps32",
            "source": "experiments/macro_search/runs/parameterized_skills_4k/packaged_quality_scheduler_3cat_penalty_0.00.json",
            "cases": 133,
            "quality_open": 31,
            "quality_open_rule": "category == chair",
            "wins_ties_losses_vs_balanced": [20, 113, 0],
            "mean_delta": 1.859022180444635,
            "mean_gain_vs_balanced": 0.030122948522062317,
            "mean_elapsed_sec": 0.5544123198195492,
            "mean_extra_elapsed_sec": 0.32406273463909785,
            "gain_per_extra_second": 0.09295406506894302,
            "note": "best partial exact-budget scheduler by gain-per-extra-second on the artifact-matched 3-category replay; opt-in only",
        },
        "learned_quality_gate_preset": {
            "profile": "geometry_top5_exact_guarded_variable_repeat_v2_state_conditioned_ridge_gate_budget5_steps32",
            "source": "experiments/macro_search/runs/parameterized_skills_4k/packaged_quality_gate_ridge_3cat.json",
            "asset": "smart/assets/skills/macro_quality_gate_ridge_v1.json",
            "cases": 133,
            "quality_open": 31,
            "quality_open_rate": 0.23308270676691728,
            "model": "ridge",
            "target": "quality_gain",
            "wins_ties_losses_vs_balanced": [434, 186, 0],
            "cv_seeds": 20,
            "mean_delta": 1.8725672785720306,
            "mean_gain_vs_balanced": 0.0436680466494583,
            "mean_elapsed_sec": 0.5213092315240604,
            "mean_extra_elapsed_sec": 0.2909596463436093,
            "gain_per_extra_second": 0.1500828283173277,
            "pareto_open_rate_sweep": [
                {"preset": "learned_fast", "open_rate": 0.10, "mean_gain_vs_balanced": 0.028706, "mean_extra_elapsed_sec": 0.113031, "gain_per_extra_second": 0.253969},
                {"open_rate": 0.15, "mean_gain_vs_balanced": 0.035202, "mean_extra_elapsed_sec": 0.185049, "gain_per_extra_second": 0.190230},
                {"preset": "learned_efficient", "open_rate": 0.233083, "mean_gain_vs_balanced": 0.042889, "mean_extra_elapsed_sec": 0.293638, "gain_per_extra_second": 0.146060},
                {"preset": "learned_quality", "open_rate": 0.50, "mean_gain_vs_balanced": 0.056796, "mean_extra_elapsed_sec": 0.506046, "gain_per_extra_second": 0.112236},
                {"open_rate": 1.00, "mean_gain_vs_balanced": 0.062289, "mean_extra_elapsed_sec": 0.745601, "gain_per_extra_second": 0.083543},
            ],
            "category_holdout": {
                "source": "experiments/macro_search/runs/parameterized_skills_4k/packaged_quality_gate_ridge_category_holdout_3cat.json",
                "mean_gain_vs_balanced": 0.04082626699236414,
                "mean_extra_elapsed_sec": 0.14914379980451134,
                "gain_per_extra_second": 0.2737376079051006,
                "losses_vs_balanced": 0,
                "per_heldout_category": {
                    "airplane": {"mean_gain_vs_balanced": 0.022655683088554722, "gain_per_extra_second": 0.37498242270008436, "losses_vs_balanced": 0},
                    "chair": {"mean_gain_vs_balanced": 0.07286177748514719, "gain_per_extra_second": 0.23759220180827278, "losses_vs_balanced": 0},
                    "table": {"mean_gain_vs_balanced": 0.05232115663725426, "gain_per_extra_second": 0.25634957276212, "losses_vs_balanced": 0},
                },
            },
            "note": "state-conditioned exact-budget gate; improves over the chair-only efficient rule in 20-seed held-out replay, opt-in only",
        },
        "packaged_replay_3cat_artifacts": {
            "balanced_source": "experiments/macro_search/runs/parameterized_skills_4k/replay_packaged_native_executor_cache_default_features_3cat.json",
            "quality_source": "experiments/macro_search/runs/parameterized_skills_4k/replay_packaged_native_executor_cache_quality_features_3cat.json",
            "cases": 133,
            "category_counts": {"airplane": 73, "chair": 31, "table": 29},
            "balanced": {
                "wins_ties_losses_vs_reference": [89, 44, 0],
                "mean_delta": 1.8288992319225725,
                "mean_gain_vs_reference": 0.10426112170744625,
                "mean_elapsed_sec": 0.2303495851804511,
            },
            "quality": {
                "wins_ties_losses_vs_reference": [112, 21, 0],
                "wins_ties_losses_vs_balanced": [76, 57, 0],
                "mean_delta": 1.8911886988511508,
                "mean_gain_vs_reference": 0.16655058863602457,
                "mean_gain_vs_balanced": 0.06228946692857832,
                "mean_elapsed_sec": 0.9759509201052633,
            },
        },
        "packaged_replay_current_artifacts": {
            "source": "experiments/macro_search/runs/parameterized_skills_4k/replay_packaged_native_executor_100_extra.json",
            "cases": 52,
            "accepted": 52,
            "wins_ties_losses_vs_reference": [36, 16, 0],
            "mean_delta": 2.320109852561561,
            "mean_gain_vs_reference": 0.1116859028917172,
            "mean_elapsed_sec": 0.0769240369615383,
            "native_vs_python_loop_speedup": 1.003738108096546,
            "native_python_parity_mismatches": 0,
            "note": "artifact-matched local replay over currently reconstructable live states",
        },
        "packaged_replay_selector_cache_artifacts": {
            "source": "experiments/macro_search/runs/parameterized_skills_4k/replay_packaged_native_executor_cache_default_100_extra.json",
            "overlap_with_uncached_native": 49,
            "score_diff_nonzero": 0,
            "macro_diff": 0,
            "accepted": 49,
            "wins_ties_losses_vs_reference": [33, 16, 0],
            "mean_delta": 2.436196216133278,
            "mean_gain_vs_reference": 0.11415146031533001,
            "mean_elapsed_sec": 0.07402416504081631,
            "native_selector_cache_hits": 501,
            "native_selector_cache_misses": 501,
            "native_selector_exact_checks": 15204,
            "native_selector_cached_checks_saved": 15204,
            "note": "exact-safe state/op selector memoization; same selected macro and score on overlap",
        },
        "rejected_restore_best_attempt_state": {
            "profile": "restore exact final bbox state from best skill attempt instead of re-executing it",
            "deployment_default": False,
            "reason": "copying final bbox state for every attempted skill cost more than the avoided re-execution on local replay",
        },
        "packaged_replay_selector_cache_quality_artifacts": {
            "source": "experiments/macro_search/runs/parameterized_skills_4k/replay_packaged_native_executor_cache_quality_100_extra.json",
            "overlap_with_uncached_quality": 48,
            "score_diff_nonzero": 0,
            "macro_diff": 0,
            "speedup_vs_uncached_quality_overlap": 1.0466557179569083,
            "accepted": 49,
            "wins_ties_losses_vs_reference": [42, 7, 0],
            "mean_delta": 2.477962019595104,
            "mean_gain_vs_reference": 0.15591726377715634,
            "mean_elapsed_sec": 0.3054935203469388,
            "native_selector_cache_hits": 1416,
            "native_selector_cache_misses": 2439,
            "native_selector_exact_checks": 67207,
            "native_selector_cached_checks_saved": 42135,
            "note": "higher exact budget quality preset; exact-safe cache keeps score/macro parity on overlap",
        },
        "previous_practical": {
            "profile": "geometry_top5_exact_conditional_budget_v1",
            "mean_delta": 1.6059663809719922,
            "mean_exact_attempts": 1.5895953757225434,
            "elapsed_sec": 0.20330911369364174,
        },
        "rejected_quality_knob": {
            "profile": "best_of_median_and_guarded",
            "mean_delta": 1.6118304945341573,
            "mean_exact_attempts": 3.179190751445087,
            "elapsed_sec": 0.34897281072254316,
            "reason": "matched artifact-corrected guarded-variable quality but doubled exact skill attempts",
        },
        "cross_validation": {
            "source": "5-fold hash splits over 173 top-5 live states, 5 seeds",
            "mean_delta": 1.6050877887966712,
            "min_delta": 1.6037570930623681,
            "max_delta": 1.6059663809719915,
            "mean_exact_attempts": 1.5791907514450867,
            "mean_gain_vs_budget1": 0.05515481323079162,
            "mean_regret_vs_top5_oracle": 0.00822685998177721,
            "losses_vs_budget1": 0,
            "stable_rule_family": "if category=table and num_actions is low: open budget 4; else budget 1",
        },
        "oracle_reference": {
            "profile": "geometry_top5_exact_best_positive",
            "mean_delta": 1.613314648778449,
            "mean_exact_attempts": 5.0,
            "elapsed_sec": 0.7227430479479764,
        },
    }


def load_skill_knowledge_base(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load mined variable-length skill templates as executable rows."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for record in payload.get("skills", []):
        skill_id = str(record.get("skill_id", ""))
        if not skill_id:
            continue
        repeats = list(record.get("default_repeats") or [])
        ranges = list(record.get("repeat_ranges") or [])
        policy = []
        for idx, step in enumerate(record.get("program") or []):
            repeat = repeats[idx] if idx < len(repeats) else None
            repeat_range = ranges[idx] if idx < len(ranges) else {}
            if repeat is None and idx < len(ranges):
                repeat = repeat_range.get("median", 1)
            policy.append(
                {
                    "op": step.get("op", "apply_axis_edit"),
                    "target": step.get("target", "unknown"),
                    "until": step.get("until", "score_stalls"),
                    "observed_repeat": str(_rounded_repeat(repeat)),
                    "repeat_min": int(float(repeat_range.get("min", repeat or 1))),
                    "repeat_max": int(float(repeat_range.get("max", repeat or 1))),
                }
            )
        out[skill_id] = {
            "macro_id": skill_id,
            "skill": str(record.get("family", "knowledge_template")),
            "policy": policy,
            "metrics": {
                "score_delta": safe_float(record.get("mean_live_delta")),
                "exact_labeled_steps": len(policy),
            },
            "aggregate": {
                "mean_score_delta": safe_float(record.get("mean_live_delta")),
                "support": safe_float(record.get("support")),
            },
            "knowledge_record": record,
        }
    return out


def load_macro_memory_policy(path: str | Path) -> dict[str, Any]:
    """Load the mined macro-memory policy table."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("model_type") != "smart_macro_memory_policy_v1":
        raise ValueError(f"unsupported macro memory policy type: {payload.get('model_type')}")
    return payload


def load_budget_rules(path: str | Path) -> list[dict[str, Any]]:
    """Load conditional exact-budget rules for skill attempts."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    policies = payload.get("best_policies") or []
    if not policies:
        raise ValueError(f"no best_policies found in exact budget rule file: {path}")
    rules = []
    for rule in policies[0].get("rules") or []:
        item = dict(rule)
        if "budget" not in item:
            raise ValueError(f"budget rule is missing budget: {path}")
        item["budget"] = int(item["budget"])
        item["rule_file"] = str(path)
        rules.append(item)
    if not rules:
        raise ValueError(f"best policy has no rules: {path}")
    rules.sort(key=lambda item: int(item.get("budget", 1)), reverse=True)
    return rules


def load_macro_quality_gate(path: str | Path) -> dict[str, Any]:
    """Load the opt-in learned high-budget gate for macro-skill quality."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("model_type") != "smart_macro_quality_gate_ridge_v1":
        raise ValueError(f"unsupported macro quality gate type: {payload.get('model_type')}")
    required = ("feature_names", "feature_mean", "feature_std", "weights", "threshold")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"macro quality gate is missing fields {missing}: {path}")
    return payload


def rank_builtin_macro_skills(
    category: str,
    *,
    skills_by_id: dict[str, dict[str, Any]] | None = None,
    memory_policy: dict[str, Any] | None = None,
    macro_memory_pool_size: int = 5,
) -> list[tuple[str, float]]:
    """Rank mined macro skills for a category using the packaged policy."""

    if skills_by_id is None or memory_policy is None:
        policy = load_builtin_macro_skill_policy()
        skills_by_id = skills_by_id or policy["skills_by_id"]
        memory_policy = memory_policy or policy["memory_policy"]
    return rank_macro_skills(
        category=category,
        skills_by_id=skills_by_id,
        memory_policy=memory_policy,
        macro_memory_pool_size=macro_memory_pool_size,
    )


def rank_macro_skills(
    *,
    category: str,
    skills_by_id: dict[str, dict[str, Any]],
    memory_policy: dict[str, Any] | None,
    macro_memory_pool_size: int = 5,
) -> list[tuple[str, float]]:
    """Rank skills by program gate, optionally re-ranking the head by memory.

    ``macro_memory_pool_size < 0`` disables the memory re-rank and keeps the
    pure program-gate order.  This is useful for reproducing the 456-case
    refine/MCTS stage validation where the program-gate top-3 selector was the
    exact-validated release candidate.
    """

    rows = []
    for macro_id, skill in skills_by_id.items():
        base = program_gate_score(category=category, skill=skill)
        memory = macro_memory_score(
            category=category,
            macro_id=macro_id,
            skill=skill,
            policy=memory_policy,
        )
        rows.append((macro_id, base, memory))
    rows.sort(key=lambda item: item[1], reverse=True)
    if macro_memory_pool_size < 0:
        return [(macro_id, base) for macro_id, base, _memory in rows]
    pool_size = max(1, macro_memory_pool_size or len(rows))
    head = rows[:pool_size]
    tail = rows[pool_size:]
    head.sort(key=lambda item: item[2], reverse=True)
    ranked = [(macro_id, memory) for macro_id, _base, memory in head]
    ranked.extend((macro_id, base) for macro_id, base, _memory in tail)
    return ranked


def macro_skill_budget(
    category: str,
    native_state_features: dict[str, Any] | None,
    budget_rules: list[dict[str, Any]],
    *,
    default_budget: int = 1,
    max_budget: int = 5,
) -> int:
    """Select how many exact skill attempts to allow for the current state."""

    for rule in budget_rules:
        if budget_rule_matches(rule, category, native_state_features):
            return max(1, min(max_budget, int(rule.get("budget", default_budget))))
    return max(1, min(max_budget, int(default_budget)))


def learned_macro_quality_budget(
    category: str,
    native_state_features: dict[str, Any] | None,
    quality_gate: dict[str, Any] | None,
    *,
    default_budget: int = 1,
    max_budget: int = 5,
    preset: str = "learned_efficient",
) -> tuple[int, dict[str, Any]]:
    """Return a high-budget decision from the packaged ridge quality gate."""

    if not quality_gate:
        return max(1, min(max_budget, int(default_budget))), {"available": False}
    named = (native_state_features or {}).get("named") or native_state_features or {}
    feature_names = list(quality_gate.get("feature_names") or [])
    mean = list(quality_gate.get("feature_mean") or [])
    std = list(quality_gate.get("feature_std") or [])
    weights = list(quality_gate.get("weights") or [])
    if not (len(feature_names) == len(mean) == len(std) == len(weights)):
        return max(1, min(max_budget, int(default_budget))), {
            "available": False,
            "reason": "invalid_gate_shape",
        }
    values: list[float] = []
    for name in feature_names:
        if name.startswith("cat_"):
            values.append(1.0 if category == name[4:] else 0.0)
        else:
            values.append(safe_float(named.get(name)))
    score = 0.0
    for value, mu, sigma, weight in zip(values, mean, std, weights):
        denom = safe_float(sigma, 1.0)
        if abs(denom) < 1.0e-6:
            denom = 1.0
        z = max(-10.0, min(10.0, (safe_float(value) - safe_float(mu)) / denom))
        score += z * safe_float(weight)
    preset_thresholds = quality_gate.get("preset_thresholds") or {}
    preset_row = preset_thresholds.get(preset) or preset_thresholds.get("learned_efficient") or {}
    threshold = safe_float(preset_row.get("threshold", quality_gate.get("threshold")))
    open_high_budget = score >= threshold
    budget = max_budget if open_high_budget else default_budget
    return max(1, min(max_budget, int(budget))), {
        "available": True,
        "model_type": quality_gate.get("model_type"),
        "preset": preset,
        "score": float(score),
        "threshold": float(threshold),
        "target_open_rate": safe_float(preset_row.get("open_rate", quality_gate.get("open_rate"))),
        "open_high_budget": bool(open_high_budget),
        "budget": int(max(1, min(max_budget, int(budget)))),
    }


def run_builtin_macro_skill_controller(
    engine: Any,
    *,
    category: str,
    cover_penalty: float = 100.0,
    pen_rate: float = 1.0,
    num_action_scale: int = 2,
    candidate_count: int = 256,
    max_steps: int = 16,
    hist_bins: int = 3,
    top_k: int = 5,
    macro_memory_pool_size: int = 5,
    exact_budget: int | None = None,
    quality_preset: str = "balanced",
    target_aware_execution: bool = False,
    native_executor: bool = True,
    restore_best_attempt_state: bool = False,
    repeat_mode: str = "guarded_variable",
) -> dict[str, Any]:
    """Try packaged macro skills on a live ``NativeSmartEngine``.

    The function mutates ``engine`` only when a positive exact-reward skill is
    accepted. If no candidate improves the score, it restores the original
    bounds/rotations/score and reports ``accepted=False``.
    """

    policy = load_builtin_macro_skill_policy()
    skills_by_id = policy["skills_by_id"]
    ranked = rank_macro_skills(
        category=category,
        skills_by_id=skills_by_id,
        memory_policy=policy["memory_policy"],
        macro_memory_pool_size=macro_memory_pool_size,
    )
    ranked = ranked[: max(1, top_k)]
    native_state_features = _engine_geometry_features(engine, hist_bins=hist_bins)
    preset = str(quality_preset or "balanced")
    learned_presets = {"learned_fast", "learned_efficient", "learned_quality"}
    if preset not in {"balanced", "quality", "efficient", "custom", *learned_presets}:
        raise ValueError(
            "quality_preset must be one of: balanced, quality, efficient, learned_fast, learned_efficient, learned_quality, custom"
        )
    effective_exact_budget = exact_budget
    learned_quality_gate_decision: dict[str, Any] = {}
    if preset == "quality":
        if effective_exact_budget is None:
            effective_exact_budget = top_k
        max_steps = max(int(max_steps), 32)
    elif preset == "efficient":
        if effective_exact_budget is None:
            effective_exact_budget = top_k if category == "chair" else 1
        if int(effective_exact_budget) > 1:
            max_steps = max(int(max_steps), 32)
    elif preset in learned_presets:
        if effective_exact_budget is None:
            effective_exact_budget, learned_quality_gate_decision = learned_macro_quality_budget(
                category,
                native_state_features,
                policy.get("quality_gate"),
                default_budget=1,
                max_budget=top_k,
                preset=preset,
            )
        if int(effective_exact_budget) > 1:
            max_steps = max(int(max_steps), 32)
    elif effective_exact_budget is not None and preset == "balanced":
        preset = "custom"
    budget = (
        max(1, min(top_k, int(effective_exact_budget)))
        if effective_exact_budget is not None
        else macro_skill_budget(
            category,
            native_state_features,
            policy["budget_rules"],
            default_budget=1,
            max_budget=top_k,
        )
    )
    saved_bounds, saved_rotations, saved_score = engine.boxes()
    attempts: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    for macro_id, rank_score in ranked[:budget]:
        skill = skills_by_id[macro_id]
        for mode in macro_repeat_modes(repeat_mode):
            engine.reset_to_state(saved_bounds, saved_rotations, saved_score)
            result = execute_macro_skill(
                engine,
                skill,
                cover_penalty=cover_penalty,
                pen_rate=pen_rate,
                num_action_scale=num_action_scale,
                candidate_count=candidate_count,
                max_steps=max_steps,
                target_aware_execution=target_aware_execution,
                native_executor=native_executor,
                return_final_state=restore_best_attempt_state,
                repeat_mode=mode,
            )
            result["rank_score"] = float(rank_score)
            attempts.append(result)
            if result["accepted"] and result["score_delta"] > 1.0e-12:
                if best_result is None or result["score_delta"] > best_result["score_delta"]:
                    best_result = result

    if best_result is not None:
        final_state = best_result.get("final_state")
        if isinstance(final_state, dict):
            engine.reset_to_state(
                final_state["bounds"],
                final_state["rotations"],
                safe_float(final_state["score"]),
            )
            final_result = dict(best_result)
            final_result["state_restored_from_best_attempt"] = True
        else:
            engine.reset_to_state(saved_bounds, saved_rotations, saved_score)
            accepted_skill = skills_by_id[str(best_result["macro_id"])]
            final_result = execute_macro_skill(
                engine,
                accepted_skill,
                cover_penalty=cover_penalty,
                pen_rate=pen_rate,
                num_action_scale=num_action_scale,
                candidate_count=candidate_count,
                max_steps=max_steps,
                target_aware_execution=target_aware_execution,
                native_executor=native_executor,
                return_final_state=False,
                repeat_mode=str(best_result.get("repeat_mode", "guarded_variable")),
            )
            final_result["state_restored_from_best_attempt"] = False
        final_result["rank_score"] = best_result.get("rank_score")
        profile_summary = macro_skill_profile_summary()
        profile = _macro_skill_profile_for_preset(profile_summary, preset)
        return _macro_skill_controller_payload(
            accepted=bool(final_result["accepted"]),
            macro_id=final_result["macro_id"],
            skill=final_result["skill"],
            score_delta=float(final_result["score_delta"]),
            initial_score=float(saved_score),
            final_score=float(final_result["final_score"]),
            executed_steps=int(final_result["executed_steps"]),
            exact_budget=int(budget),
            attempt_count=len(attempts),
            repeat_mode=repeat_mode,
            quality_preset=preset,
            max_steps=int(max_steps),
            attempts=attempts,
            ranked_head=ranked,
            native_state_features=native_state_features,
            learned_quality_gate_decision=learned_quality_gate_decision,
            native_selector_cache_stats=_engine_native_selector_cache_stats(engine),
            profile=profile,
        )

    engine.reset_to_state(saved_bounds, saved_rotations, saved_score)
    profile_summary = macro_skill_profile_summary()
    profile = _macro_skill_profile_for_preset(profile_summary, preset)
    return _macro_skill_controller_payload(
        accepted=False,
        macro_id=None,
        skill=None,
        score_delta=0.0,
        initial_score=float(saved_score),
        final_score=float(saved_score),
        executed_steps=0,
        exact_budget=int(budget),
        attempt_count=len(attempts),
        repeat_mode=repeat_mode,
        quality_preset=preset,
        max_steps=int(max_steps),
        attempts=attempts,
        ranked_head=ranked,
        native_state_features=native_state_features,
        learned_quality_gate_decision=learned_quality_gate_decision,
        native_selector_cache_stats=_engine_native_selector_cache_stats(engine),
        profile=profile,
    )


def run_builtin_macro_skill_planner(
    engine: Any,
    *,
    category: str,
    max_rounds: int = 3,
    profile_schedule: list[str] | tuple[str, ...] = ("balanced", "learned_efficient", "quality"),
    min_round_delta: float = 1.0e-12,
    stop_after_noop: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a guarded multi-round macro-skill planner on a native engine.

    This is the deployable form of the "3D substructure planner" idea: each
    round re-reads the current geometry state, tries a small profile portfolio,
    and commits only the best exact-validated non-worse update.  The learned
    skill policy proposes reusable variable-length programs, but exact
    SMART/Manifold score remains the only acceptance signal.
    """

    max_rounds = max(0, int(max_rounds))
    initial_bounds, initial_rotations, initial_score = engine.boxes()
    planner_initial_score = float(initial_score)
    profile_schedule = tuple(str(item) for item in profile_schedule if str(item))
    if not profile_schedule:
        profile_schedule = ("balanced",)
    controller_kwargs = dict(kwargs)
    controller_kwargs.pop("quality_preset", None)

    rounds: list[dict[str, Any]] = []
    total_attempt_count = 0
    total_exact_budget = 0
    accepted_rounds = 0
    current_score = planner_initial_score
    for round_index in range(max_rounds):
        round_bounds, round_rotations, round_score = engine.boxes()
        round_score = float(round_score)
        best: dict[str, Any] | None = None
        best_state: tuple[Any, Any, float] | None = None
        profile_results: list[dict[str, Any]] = []

        for preset in profile_schedule:
            engine.reset_to_state(round_bounds, round_rotations, round_score)
            result = run_builtin_macro_skill_controller(
                engine,
                category=category,
                quality_preset=preset,
                **controller_kwargs,
            )
            total_attempt_count += int(result.get("attempt_count", 0))
            total_exact_budget += int(result.get("exact_budget", 0))
            candidate_bounds, candidate_rotations, candidate_score = engine.boxes()
            candidate_score = float(candidate_score)
            candidate_delta = candidate_score - round_score
            item = {
                "profile": preset,
                "accepted": bool(result.get("accepted")) and candidate_delta >= -1.0e-12,
                "score_delta": candidate_delta,
                "final_score": candidate_score,
                "macro_id": result.get("macro_id"),
                "skill": result.get("skill"),
                "attempt_count": int(result.get("attempt_count", 0)),
                "exact_budget": int(result.get("exact_budget", 0)),
                "executed_steps": int(result.get("executed_steps", 0)),
            }
            profile_results.append(item)
            if item["accepted"] and candidate_delta > min_round_delta:
                if best is None or candidate_delta > float(best["score_delta"]):
                    best = item
                    best["controller_result"] = result
                    best_state = (candidate_bounds, candidate_rotations, candidate_score)

        if best is None or best_state is None:
            engine.reset_to_state(round_bounds, round_rotations, round_score)
            rounds.append(
                {
                    "round": round_index,
                    "accepted": False,
                    "initial_score": round_score,
                    "final_score": round_score,
                    "score_delta": 0.0,
                    "profile_results": profile_results,
                }
            )
            if stop_after_noop:
                break
            continue

        engine.reset_to_state(best_state[0], best_state[1], best_state[2])
        current_score = float(best_state[2])
        accepted_rounds += 1
        rounds.append(
            {
                "round": round_index,
                "accepted": True,
                "initial_score": round_score,
                "final_score": current_score,
                "score_delta": current_score - round_score,
                "selected_profile": best["profile"],
                "macro_id": best.get("macro_id"),
                "skill": best.get("skill"),
                "attempt_count": best.get("attempt_count"),
                "exact_budget": best.get("exact_budget"),
                "executed_steps": best.get("executed_steps"),
                "profile_results": profile_results,
            }
        )

    final_bounds, final_rotations, final_score = engine.boxes()
    final_score = float(final_score)
    total_delta = final_score - planner_initial_score
    if total_delta < -1.0e-12:
        engine.reset_to_state(initial_bounds, initial_rotations, planner_initial_score)
        final_score = planner_initial_score
        total_delta = 0.0
        final_bounds, final_rotations = initial_bounds, initial_rotations

    return {
        "accepted": accepted_rounds > 0,
        "accepted_rounds": accepted_rounds,
        "round_count": len(rounds),
        "max_rounds": max_rounds,
        "profile_schedule": list(profile_schedule),
        "score_delta": total_delta,
        "initial_score": planner_initial_score,
        "final_score": final_score,
        "rounds": rounds,
        "attempt_count": total_attempt_count,
        "exact_budget": total_exact_budget,
        "accepted_non_worse": total_delta >= -1.0e-12,
        "exact_validator": "native_smart_manifold",
        "rollback_on_failure": True,
        "deployment_status": "release_candidate_opt_in_substructure_planner",
        "default_smart_path_changed": False,
        "quality_contract": (
            "each planner round commits only the best exact SMART/Manifold non-worse macro-skill update"
        ),
        "final_state": {
            "bounds": final_bounds,
            "rotations": final_rotations,
            "score": final_score,
        },
    }


def _macro_skill_profile_for_preset(profile_summary: dict[str, Any], preset: str) -> dict[str, Any]:
    if preset == "quality":
        return profile_summary["quality_preset"]
    if preset in {"learned_fast", "learned_efficient", "learned_quality"}:
        return profile_summary["learned_quality_gate_preset"]
    if preset == "efficient":
        return profile_summary["efficient_quality_preset"]
    return profile_summary["best_practical"]


def _macro_skill_controller_payload(**payload: Any) -> dict[str, Any]:
    """Add stable safety metadata to a controller result."""

    score_delta = safe_float(payload.get("score_delta"))
    payload["accepted_non_worse"] = score_delta >= -1.0e-12
    payload["exact_validator"] = "native_smart_manifold"
    payload["rollback_on_failure"] = True
    payload["deployment_status"] = "release_candidate_opt_in_post_refine"
    payload["default_smart_path_changed"] = False
    payload["quality_contract"] = (
        "accepted updates are exact SMART/Manifold non-worse; rejected updates restore the input state"
    )
    return payload


def _engine_native_selector_cache_stats(engine: Any) -> dict[str, float]:
    """Extract exact selector memoization counters from a native engine."""

    if not hasattr(engine, "stats"):
        return {}
    try:
        stats = engine.stats()
    except Exception:
        return {}
    keys = {
        "calls": "native_macro_select_exact_action_calls",
        "checks": "native_macro_select_exact_action_checks",
        "errors": "native_macro_select_exact_action_errors",
        "cache_hits": "native_macro_select_exact_action_cache_hits",
        "cache_misses": "native_macro_select_exact_action_cache_misses",
        "cached_checks_saved": "native_macro_select_exact_action_cached_checks_saved",
    }
    return {name: safe_float(stats.get(key)) for name, key in keys.items()}


def run_builtin_macro_skill_controller_from_files(
    *,
    msh_path: str | Path,
    bbox_metadata_path: str | Path,
    category: str,
    cover_penalty: float = 100.0,
    pen_rate: float = 1.0,
    num_action_scale: int = 2,
    action_unit: float = 0.01,
    stateful_union_cache: bool = True,
    cache_capacity: int = 65536,
    volume_method: str = "mesh",
    **kwargs: Any,
) -> dict[str, Any]:
    """Load a native engine from files, then run the macro-skill controller.

    This is the easiest package-facing replay path for an already prepared
    SMART tetra mesh and bbox metadata file.  The returned dictionary includes
    the live engine object so callers can export or continue refinement after an
    accepted skill.
    """

    from . import cpp
    from .native_runner import load_bbox_params

    bounds, rotations = load_bbox_params(bbox_metadata_path)
    engine = cpp.native_smart_engine_from_gmsh(
        str(msh_path),
        bounds,
        rotations,
        category,
        num_action_scale,
        action_unit,
        0.0,
        0.0,
        stateful_union_cache,
        cache_capacity,
        volume_method,
    )
    engine.recompute_score(cover_penalty, pen_rate)
    result = run_builtin_macro_skill_controller(
        engine,
        category=category,
        cover_penalty=cover_penalty,
        pen_rate=pen_rate,
        num_action_scale=num_action_scale,
        **kwargs,
    )
    result["engine"] = engine
    result["msh_path"] = str(msh_path)
    result["bbox_metadata_path"] = str(bbox_metadata_path)
    return result


def run_builtin_macro_skill_planner_from_files(
    *,
    msh_path: str | Path,
    bbox_metadata_path: str | Path,
    category: str,
    cover_penalty: float = 100.0,
    pen_rate: float = 1.0,
    num_action_scale: int = 2,
    action_unit: float = 0.01,
    stateful_union_cache: bool = True,
    cache_capacity: int = 65536,
    volume_method: str = "mesh",
    **kwargs: Any,
) -> dict[str, Any]:
    """Load prepared SMART files and run the multi-round macro planner."""

    from . import cpp
    from .native_runner import load_bbox_params

    bounds, rotations = load_bbox_params(bbox_metadata_path)
    engine = cpp.native_smart_engine_from_gmsh(
        str(msh_path),
        bounds,
        rotations,
        category,
        num_action_scale,
        action_unit,
        0.0,
        0.0,
        stateful_union_cache,
        cache_capacity,
        volume_method,
    )
    engine.recompute_score(cover_penalty, pen_rate)
    result = run_builtin_macro_skill_planner(
        engine,
        category=category,
        cover_penalty=cover_penalty,
        pen_rate=pen_rate,
        num_action_scale=num_action_scale,
        **kwargs,
    )
    result["engine"] = engine
    result["msh_path"] = str(msh_path)
    result["bbox_metadata_path"] = str(bbox_metadata_path)
    return result


def execute_macro_skill(
    engine: Any,
    skill: dict[str, Any],
    *,
    cover_penalty: float,
    pen_rate: float,
    num_action_scale: int,
    candidate_count: int,
    max_steps: int,
    target_aware_execution: bool = False,
    native_executor: bool = True,
    return_final_state: bool = False,
    repeat_mode: str = "guarded_variable",
) -> dict[str, Any]:
    """Execute one parameterized skill with rollback metadata."""

    if native_executor and not target_aware_execution and hasattr(engine, "execute_native_macro_skill"):
        try:
            return dict(
                engine.execute_native_macro_skill(
                    skill,
                    cover_penalty,
                    pen_rate,
                    candidate_count,
                    max_steps,
                    repeat_mode,
                    False,
                    return_final_state,
                )
            )
        except Exception:
            pass

    initial_bounds, initial_rotations, initial_score = engine.boxes()
    actions: list[dict[str, Any]] = []
    for item in skill.get("policy", []):
        op = str(item.get("op", "apply_axis_edit"))
        target = str(item.get("target", "unknown"))
        repeats = macro_step_repeat_budget(item, max_steps=max_steps - len(actions), repeat_mode=repeat_mode)
        for _ in range(repeats):
            if len(actions) >= max_steps:
                break
            if target_aware_execution:
                action, predicted_reward, target_filter_used = choose_exact_best_action_targeted(
                    engine,
                    op=op,
                    target=target,
                    cover_penalty=cover_penalty,
                    pen_rate=pen_rate,
                    num_action_scale=num_action_scale,
                    candidate_count=candidate_count,
                    step_index=len(actions),
                    total_steps=max_steps,
                )
            else:
                action, predicted_reward = choose_exact_best_action(
                    engine,
                    op=op,
                    cover_penalty=cover_penalty,
                    pen_rate=pen_rate,
                    num_action_scale=num_action_scale,
                    candidate_count=candidate_count,
                )
                target_filter_used = False
            if action is None:
                break
            if not allow_nonpositive_macro_step(op, predicted_reward):
                break
            try:
                applied = apply_exact_action(
                    engine,
                    int(action),
                    cover_penalty,
                    pen_rate,
                    num_action_scale,
                )
            except Exception as exc:
                actions.append(
                    {
                        "op": op,
                        "target": target,
                        "action": int(action),
                        "status": "apply_failed",
                        "error": str(exc),
                    }
                )
                break
            reward = float(applied["reward"])
            actions.append(
                {
                    "op": op,
                    "target": target,
                    "action": int(action),
                    "kind": str(applied["kind"]),
                    "bbox": int(applied["bbox_idx"]),
                    "reward": reward,
                    "predicted_reward": float(predicted_reward),
                    "score": float(applied["last_score"]),
                    "target_filter_used": bool(target_filter_used),
                }
            )
            if not allow_nonpositive_macro_step(op, reward):
                break
    final_score = float(engine.recompute_score(cover_penalty, pen_rate))
    delta = final_score - float(initial_score)
    out = {
        "macro_id": skill.get("macro_id"),
        "skill": skill.get("skill"),
        "accepted": delta >= -1.0e-12 and bool(actions),
        "score_delta": delta,
        "repeat_mode": repeat_mode,
        "initial_score": float(initial_score),
        "final_score": final_score,
        "actions": actions,
        "executed_steps": len(actions),
        "rollback_state": {
            "bounds": initial_bounds,
            "rotations": initial_rotations,
            "score": float(initial_score),
        },
    }
    if return_final_state:
        final_bounds, final_rotations, _ = engine.boxes()
        out["final_state"] = {
            "bounds": final_bounds,
            "rotations": final_rotations,
            "score": final_score,
        }
    return out


def macro_repeat_modes(repeat_mode: str) -> list[str]:
    """Expand the controller-level repeat mode into execution modes."""

    mode = str(repeat_mode)
    if mode in {"guarded_variable", "median"}:
        return [mode]
    if mode == "best_of_median_and_guarded":
        return ["median", "guarded_variable"]
    raise ValueError(
        "repeat_mode must be one of: guarded_variable, median, best_of_median_and_guarded"
    )


def allow_nonpositive_macro_step(op: str, reward: float, *, eps: float = 1.0e-12) -> bool:
    """Return whether a non-improving primitive may be part of a skill.

    Expansion macros intentionally permit temporary exact-score drops so they
    can escape a local minimum and then tighten. Shrink/tighten macros should
    stop before applying a negative step; otherwise the final state can be
    corrupted by the first failed shrink after a successful variable-length run.
    Recenter actions are allowed when they are effectively score-neutral.
    """

    value = float(reward)
    if value > eps:
        return True
    if op == "expand_face":
        return True
    if op in {"recenter", "recenter_box"}:
        return abs(value) <= 1.0e-9
    return False


def macro_step_repeat_budget(item: dict[str, Any], *, max_steps: int, repeat_mode: str = "guarded_variable") -> int:
    """Return the execution budget for one parameterized skill step.

    Median repeats are good summaries for mining reports, but they are too
    conservative for executable variable-length skills.  When the mined step
    explicitly terminates on score/coverage predicates, let exact reward drive
    the loop up to the observed repeat range maximum.
    """

    if max_steps <= 0:
        return 0
    if repeat_mode not in {"guarded_variable", "median"}:
        raise ValueError("repeat_mode must be 'guarded_variable' or 'median' at skill execution time")
    if str(item.get("op", "")) in {"recenter", "recenter_box"}:
        return 1
    observed = max(1, int(float(item.get("observed_repeat", 1))))
    repeat_max = max(observed, int(float(item.get("repeat_max", observed))))
    if repeat_mode == "median":
        return max(1, min(int(max_steps), int(observed)))
    until = str(item.get("until", ""))
    variable_length = any(
        token in until
        for token in (
            "score_stalls",
            "coverage_margin_low",
            "coverage_recovered",
            "exact_reward_delta_positive",
        )
    )
    budget = repeat_max if variable_length else observed
    return max(1, min(int(max_steps), int(budget)))


def choose_exact_best_action(
    engine: Any,
    *,
    op: str,
    cover_penalty: float,
    pen_rate: float,
    num_action_scale: int,
    candidate_count: int,
) -> tuple[int | None, float]:
    """Choose the best exact action matching a skill operation."""

    if hasattr(engine, "select_exact_native_action"):
        try:
            selected = engine.select_exact_native_action(
                str(op),
                cover_penalty,
                pen_rate,
                candidate_count,
            )
            if bool(selected.get("found", False)):
                return int(selected["action"]), float(selected["reward"])
            return None, float(selected.get("reward", -float("inf")))
        except Exception:
            pass

    best_action: int | None = None
    best_reward = -float("inf")
    for action in _candidate_actions_with_recenter(engine, cover_penalty, pen_rate, candidate_count, num_action_scale):
        if not role_matches_action(op, action, num_action_scale):
            continue
        try:
            if hasattr(engine, "score_native_action_reward"):
                reward = float(engine.score_native_action_reward(action, cover_penalty, pen_rate))
            else:
                if action_direction(action, num_action_scale) == "recenter":
                    continue
                reward = float(engine.score_axis_action_reward(action, cover_penalty, pen_rate))
        except Exception:
            continue
        if not math.isfinite(reward):
            continue
        if reward > best_reward:
            best_reward = reward
            best_action = action
    return best_action, best_reward


def choose_exact_best_action_targeted(
    engine: Any,
    *,
    op: str,
    target: str,
    cover_penalty: float,
    pen_rate: float,
    num_action_scale: int,
    candidate_count: int,
    step_index: int,
    total_steps: int,
) -> tuple[int | None, float, bool]:
    """Choose exact best action, optionally filtering by mined target role."""

    filtered: list[int] = []
    for action in _candidate_actions_with_recenter(engine, cover_penalty, pen_rate, candidate_count, num_action_scale):
        if not role_matches_action(op, action, num_action_scale):
            continue
        named = _local_named_features(engine, action, step_index, total_steps)
        if _target_matches_action(target, named):
            filtered.append(action)
    fallback_action, fallback_reward = choose_exact_best_action(
        engine,
        op=op,
        cover_penalty=cover_penalty,
        pen_rate=pen_rate,
        num_action_scale=num_action_scale,
        candidate_count=candidate_count,
    )
    if not filtered:
        return fallback_action, fallback_reward, False

    best_action: int | None = None
    best_reward = -float("inf")
    for action in filtered:
        try:
            if hasattr(engine, "score_native_action_reward"):
                reward = float(engine.score_native_action_reward(action, cover_penalty, pen_rate))
            else:
                if action_direction(action, num_action_scale) == "recenter":
                    continue
                reward = float(engine.score_axis_action_reward(action, cover_penalty, pen_rate))
        except Exception:
            continue
        if not math.isfinite(reward):
            continue
        if reward > best_reward:
            best_reward = reward
            best_action = action
    if best_action is None:
        return fallback_action, fallback_reward, False
    if fallback_action is not None and fallback_reward > best_reward + 1.0e-12:
        return fallback_action, fallback_reward, False
    return best_action, best_reward, True


def apply_exact_action(
    engine: Any,
    action: int,
    cover_penalty: float,
    pen_rate: float,
    num_action_scale: int,
) -> dict[str, Any]:
    """Apply an exact native action and normalize the return payload."""

    if hasattr(engine, "apply_native_action_delta"):
        applied = engine.apply_native_action_delta(int(action), cover_penalty, pen_rate)
        return {
            "kind": str(applied.get("kind", action_direction(action, num_action_scale))),
            "reward": float(applied["reward"]),
            "bbox_idx": int(applied["bbox_idx"]),
            "bounds": list(applied.get("bounds", [])),
            "rotation": list(applied.get("rotation", [])),
            "last_score": float(applied["last_score"]),
        }
    if action_direction(action, num_action_scale) == "recenter":
        raise RuntimeError("native recenter apply is not available")
    reward, bbox_idx, last_score = engine.apply_axis_action_delta(
        int(action), cover_penalty, pen_rate
    )
    return {
        "kind": "axis",
        "reward": float(reward),
        "bbox_idx": int(bbox_idx),
        "bounds": [],
        "rotation": [],
        "last_score": float(last_score),
    }


def macro_memory_score(
    *,
    category: str,
    macro_id: str,
    skill: dict[str, Any],
    policy: dict[str, Any] | None,
) -> float:
    if not policy:
        return 0.0
    record = skill.get("knowledge_record") or {}
    family = str(record.get("family", skill.get("skill", "")))
    keys = {
        "category_macro_id": f"{category}::{macro_id}" if category else "",
        "category_family": f"{category}::{family}" if category else "",
        "macro_id": str(macro_id),
        "family": family,
        "skill": str(skill.get("skill", "")),
    }
    tables = policy.get("tables") or {}
    weights = policy.get("weights") or {}
    fallback = safe_float(policy.get("fallback_score"))
    total = 0.0
    score = 0.0
    for kind, weight in weights.items():
        weight = safe_float(weight)
        if abs(weight) < 1.0e-12:
            continue
        table = tables.get(kind) or {}
        item = table.get(keys.get(kind, ""), table.get("__fallback__", {"score": fallback}))
        score += weight * safe_float(item.get("score", fallback))
        total += abs(weight)
    return score / max(1.0e-9, total)


def program_gate_score(*, category: str, skill: dict[str, Any]) -> float:
    record = skill.get("knowledge_record") or {}
    mean_live_delta = safe_float(record.get("mean_live_delta"))
    support = safe_float(record.get("support"))
    program = record.get("program") or []
    family = str(record.get("family", skill.get("skill", "")))
    score = mean_live_delta + 0.001 * math.log1p(max(0.0, support))
    if family == "recenter_then_shrink" and len(program) >= 5:
        score += 0.65
    if category == "table" and family == "shrink_slack_face":
        score += 0.15
    return score


def budget_rule_matches(
    rule: dict[str, Any],
    category: str,
    native_state_features: dict[str, Any] | None,
) -> bool:
    rule_category = rule.get("category")
    if rule_category not in {None, "", "None"} and category != str(rule_category):
        return False
    if str(rule.get("kind", "threshold")) == "category":
        return True
    feature = rule.get("feature")
    if not feature:
        return False
    named = (native_state_features or {}).get("named") or native_state_features or {}
    if feature not in named:
        return False
    value = safe_float(named.get(feature))
    threshold = safe_float(rule.get("threshold"))
    op = str(rule.get("op"))
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "==":
        return abs(value - threshold) <= 1.0e-12
    if op == "!=":
        return abs(value - threshold) > 1.0e-12
    return False


def decode_action(action: int, num_action_scale: int) -> dict[str, int]:
    actions_per_bbox = 6 * num_action_scale + 1
    bbox = action // actions_per_bbox
    local = action % actions_per_bbox
    if local == actions_per_bbox - 1:
        return {"bbox": bbox, "coord": 6, "scale": 0, "recenter": 1}
    return {
        "bbox": bbox,
        "coord": local // num_action_scale,
        "scale": local % num_action_scale,
        "recenter": 0,
    }


def action_scale(scale: int, num_action_scale: int) -> float:
    half = num_action_scale // 2
    if scale < half:
        return -float(2 ** (half - 1 - scale))
    return float(2 ** (scale - half))


def action_direction(action: int, num_action_scale: int) -> str:
    decoded = decode_action(action, num_action_scale)
    if decoded["recenter"]:
        return "recenter"
    coord = decoded["coord"]
    scale = action_scale(decoded["scale"], num_action_scale)
    if coord < 3:
        return "shrink" if scale > 0 else "expand"
    return "shrink" if scale < 0 else "expand"


def role_matches_action(op: str, action: int, num_action_scale: int) -> bool:
    direction = action_direction(action, num_action_scale)
    if op == "expand_face":
        return direction == "expand"
    if op == "shrink_face":
        return direction == "shrink"
    if op == "recenter_box":
        return direction == "recenter"
    return direction in {"expand", "shrink"}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _rounded_repeat(value: Any) -> int:
    raw = safe_float(value, 1.0)
    return max(1, int(math.floor(raw + 0.5)))


def _candidate_actions_with_recenter(
    engine: Any,
    cover_penalty: float,
    pen_rate: float,
    candidate_count: int,
    num_action_scale: int,
) -> list[int]:
    rows = engine.centroid_proxy_axis_metrics(cover_penalty, pen_rate, candidate_count)
    actions = [int(row[0]) for row in rows]
    try:
        num_boxes = int(float(engine.stats().get("num_boxes", 0)))
    except Exception:
        num_boxes = 0
    actions_per_bbox = 6 * num_action_scale + 1
    actions.extend(
        bbox_idx * actions_per_bbox + (actions_per_bbox - 1)
        for bbox_idx in range(num_boxes)
    )
    return list(dict.fromkeys(actions))


def _engine_geometry_features(engine: Any, *, hist_bins: int) -> dict[str, Any]:
    if not hasattr(engine, "geometry_state_features"):
        return {}
    try:
        raw = engine.geometry_state_features(hist_bins)
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _local_named_features(engine: Any, action: int, step_index: int, total_steps: int) -> dict[str, float]:
    if not hasattr(engine, "local_action_features"):
        return {}
    try:
        raw = engine.local_action_features(int(action), int(step_index), int(total_steps))
    except Exception:
        return {}
    named = raw.get("named") if isinstance(raw, dict) else None
    if not isinstance(named, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in named.items():
        v = safe_float(value, float("nan"))
        if math.isfinite(v):
            out[str(key)] = v
    return out


def _target_matches_action(
    target: str,
    named: dict[str, float],
    *,
    tolerance: float = 1.0e-9,
) -> bool:
    target = str(target or "unknown")
    if target in {"", "unknown", "any", "face", "score_best_face", "coverage_gap_face", "uncovered_face"}:
        return True
    if target == "single_box":
        return named.get("bbox_idx", 0.0) == 0.0
    if target in {"major_axis_face", "minor_axis_face"}:
        axis = int(named.get("axis_idx", -1.0))
        if axis < 0:
            return True
        extents = [
            named.get("box_extent_x", 0.0),
            named.get("box_extent_y", 0.0),
            named.get("box_extent_z", 0.0),
        ]
        chosen = max(range(3), key=lambda idx: extents[idx]) if target == "major_axis_face" else min(
            range(3), key=lambda idx: extents[idx]
        )
        return axis == chosen
    if target == "max_slack_face":
        coord = int(named.get("coord_idx", -1.0))
        if coord < 0 or coord > 5:
            return True
        slack_names = [
            "slack_min_x",
            "slack_min_y",
            "slack_min_z",
            "slack_max_x",
            "slack_max_y",
            "slack_max_z",
        ]
        slacks = [named.get(name, 0.0) for name in slack_names]
        max_slack = max(slacks) if slacks else 0.0
        if max_slack <= tolerance:
            return True
        return slacks[coord] >= max_slack - tolerance
    if target == "min_dim_face":
        axis = int(named.get("axis_idx", -1.0))
        if axis < 0:
            return True
        extents = [
            named.get("box_extent_x", 0.0),
            named.get("box_extent_y", 0.0),
            named.get("box_extent_z", 0.0),
        ]
        return axis == min(range(3), key=lambda idx: extents[idx])
    return True
