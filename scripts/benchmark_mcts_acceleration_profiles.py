from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from smart.pipeline.config import load_config


METRIC_KEYS = (
    "Avg_num_box",
    "Avg_BVS",
    "Avg_MOV",
    "Avg_TOV",
    "Avg_Covered",
    "Avg_vIoU",
    "Avg_cub_CD",
)


PROFILE_SETS: dict[str, list[tuple[str, str]]] = {
    "exact_auto": [
        ("mcts.reward_backend", "manifold_stateful"),
        ("mcts.backend", "auto"),
        ("mcts.stateful_union_cache", "false"),
        ("mcts.candidate_backend", "exact"),
    ],
    "bitset_top3": [
        ("mcts.reward_backend", "manifold_stateful"),
        ("mcts.backend", "auto"),
        ("mcts.stateful_union_cache", "false"),
        ("mcts.candidate_backend", "bitset_topk"),
        ("mcts.candidate_top_k", "3"),
    ],
    "bitset_top8": [
        ("mcts.reward_backend", "manifold_stateful"),
        ("mcts.backend", "auto"),
        ("mcts.stateful_union_cache", "false"),
        ("mcts.candidate_backend", "bitset_topk"),
        ("mcts.candidate_top_k", "8"),
    ],
    "exact_union_cache": [
        ("mcts.reward_backend", "manifold_stateful"),
        ("mcts.backend", "auto"),
        ("mcts.stateful_union_cache", "true"),
        ("mcts.candidate_backend", "exact"),
    ],
    "exact_union_cache_fused": [
        ("mcts.reward_backend", "manifold_stateful"),
        ("mcts.backend", "auto"),
        ("mcts.stateful_union_cache", "true"),
        ("mcts.candidate_backend", "exact"),
        ("mcts.fused_rollout_step", "true"),
    ],
    "exact_union_cache_fast_stop20": [
        ("mcts.reward_backend", "manifold_stateful"),
        ("mcts.backend", "auto"),
        ("mcts.stateful_union_cache", "true"),
        ("mcts.candidate_backend", "exact"),
        ("mcts.no_reward_stop_after", "20"),
    ],
    "bitset_top3_union_cache": [
        ("mcts.reward_backend", "manifold_stateful"),
        ("mcts.backend", "auto"),
        ("mcts.stateful_union_cache", "true"),
        ("mcts.candidate_backend", "bitset_topk"),
        ("mcts.candidate_top_k", "3"),
    ],
    "bitset_top8_union_cache": [
        ("mcts.reward_backend", "manifold_stateful"),
        ("mcts.backend", "auto"),
        ("mcts.stateful_union_cache", "true"),
        ("mcts.candidate_backend", "bitset_topk"),
        ("mcts.candidate_top_k", "8"),
    ],
    "rust_stateful": [
        ("mcts.reward_backend", "manifold_stateful"),
        ("mcts.backend", "rust_stateful"),
        ("mcts.stateful_union_cache", "false"),
        ("mcts.candidate_backend", "exact"),
        ("mcts.allow_search_order_changes", "true"),
    ],
    "rust_stateful_tt": [
        ("mcts.reward_backend", "manifold_stateful"),
        ("mcts.backend", "rust_stateful"),
        ("mcts.stateful_union_cache", "false"),
        ("mcts.candidate_backend", "exact"),
        ("mcts.allow_search_order_changes", "true"),
        ("mcts.transposition_table", "true"),
    ],
    "rust_stateful_prior01": [
        ("mcts.reward_backend", "manifold_stateful"),
        ("mcts.backend", "rust_stateful"),
        ("mcts.stateful_union_cache", "false"),
        ("mcts.candidate_backend", "exact"),
        ("mcts.allow_search_order_changes", "true"),
        ("mcts.action_prior_weight", "0.1"),
        ("mcts.action_prior_path", "smart/assets/priors/smoke5_coord_scale_prior.json"),
    ],
    "rust_stateful_prior01_tt": [
        ("mcts.reward_backend", "manifold_stateful"),
        ("mcts.backend", "rust_stateful"),
        ("mcts.stateful_union_cache", "false"),
        ("mcts.candidate_backend", "exact"),
        ("mcts.allow_search_order_changes", "true"),
        ("mcts.action_prior_weight", "0.1"),
        ("mcts.action_prior_path", "smart/assets/priors/smoke5_coord_scale_prior.json"),
        ("mcts.transposition_table", "true"),
    ],
    "rust_stateful_prior01_union_cache": [
        ("mcts.reward_backend", "manifold_stateful"),
        ("mcts.backend", "rust_stateful"),
        ("mcts.stateful_union_cache", "true"),
        ("mcts.candidate_backend", "exact"),
        ("mcts.allow_search_order_changes", "true"),
        ("mcts.action_prior_weight", "0.1"),
        ("mcts.action_prior_path", "smart/assets/priors/smoke5_coord_scale_prior.json"),
    ],
    "rust_stateful_prior005_union_cache": [
        ("mcts.reward_backend", "manifold_stateful"),
        ("mcts.backend", "rust_stateful"),
        ("mcts.stateful_union_cache", "true"),
        ("mcts.candidate_backend", "exact"),
        ("mcts.allow_search_order_changes", "true"),
        ("mcts.action_prior_weight", "0.05"),
        ("mcts.action_prior_path", "smart/assets/priors/smoke5_coord_scale_prior.json"),
    ],
    "rust_stateful_prior002_union_cache": [
        ("mcts.reward_backend", "manifold_stateful"),
        ("mcts.backend", "rust_stateful"),
        ("mcts.stateful_union_cache", "true"),
        ("mcts.candidate_backend", "exact"),
        ("mcts.allow_search_order_changes", "true"),
        ("mcts.action_prior_weight", "0.02"),
        ("mcts.action_prior_path", "smart/assets/priors/smoke5_coord_scale_prior.json"),
    ],
    "rust_stateful_prior01_tt_union_cache": [
        ("mcts.reward_backend", "manifold_stateful"),
        ("mcts.backend", "rust_stateful"),
        ("mcts.stateful_union_cache", "true"),
        ("mcts.candidate_backend", "exact"),
        ("mcts.allow_search_order_changes", "true"),
        ("mcts.action_prior_weight", "0.1"),
        ("mcts.action_prior_path", "smart/assets/priors/smoke5_coord_scale_prior.json"),
        ("mcts.transposition_table", "true"),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare opt-in MCTS acceleration profiles against the exact "
            "manifold_stateful/legacy-tree baseline."
        )
    )
    parser.add_argument("--config", default="configs/smoke_5.yaml")
    parser.add_argument("--categories", default="")
    parser.add_argument("--target-limit", type=int, default=3)
    parser.add_argument("--mcts-iter", type=int, default=10)
    parser.add_argument("--max-step", type=int, default=10)
    parser.add_argument(
        "--profiles",
        default="exact_auto,bitset_top3,bitset_top8,rust_stateful_tt,rust_stateful_prior01_tt",
        help="Comma-separated profile names. Available: %s" % ", ".join(sorted(PROFILE_SETS)),
    )
    parser.add_argument("--baseline-profile", default="exact_auto")
    parser.add_argument("--transposition-table-size", type=int, default=8192)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--chamfer-points", type=int, default=0)
    parser.add_argument("--output", default="runs/bench_exact/mcts_acceleration_profiles.json")
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be >= 1")

    profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    unknown = [name for name in profiles if name not in PROFILE_SETS]
    if unknown:
        parser.error("unknown profiles: %s" % ", ".join(unknown))
    if args.baseline_profile not in profiles:
        parser.error("--baseline-profile must be included in --profiles")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets = _targets(load_config(args.config), args.categories, args.target_limit)

    results: dict[str, Any] = {
        "config": args.config,
        "mcts_iter": args.mcts_iter,
        "max_step": args.max_step,
        "profiles": profiles,
        "baseline_profile": args.baseline_profile,
        "target_count": len(targets),
        "targets": {},
    }

    for target in targets:
        target_key = _target_key(target)
        target_record: dict[str, Any] = {
            "category": target["category"],
            "mesh": target["mesh"],
            "runs": {},
        }
        results["targets"][target_key] = target_record
        for profile in profiles:
            run_record = _run_profile(args, target, target_key, profile)
            eval_record = _run_eval(args, target, output_path, target_key, profile)
            run_record.update(eval_record)
            target_record["runs"][profile] = run_record
            target_record["speedup_vs_baseline"] = _speedups(
                target_record["runs"], args.baseline_profile
            )
            target_record["metric_diffs_vs_baseline"] = _metric_diffs(
                target_record["runs"], args.baseline_profile
            )
            _write_json(output_path, results)
            if run_record["returncode"] != 0 or eval_record["evaluation_returncode"] != 0:
                print(json.dumps(results, indent=2, sort_keys=True))
                return int(run_record["returncode"] or eval_record["evaluation_returncode"])

    results["aggregate"] = _aggregate(results["targets"], args.baseline_profile)
    _write_json(output_path, results)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def _targets(cfg: dict[str, Any], categories: str, limit: int) -> list[dict[str, str]]:
    allowed = {item.strip() for item in categories.split(",") if item.strip()}
    out = []
    for category in cfg.get("categories", []):
        name = str(category["name"])
        if allowed and name not in allowed:
            continue
        for mesh in category.get("meshes", []):
            out.append({"category": name, "mesh": str(mesh)})
            if limit > 0 and len(out) >= limit:
                return out
    return out


def _run_profile(
    args: argparse.Namespace,
    target: dict[str, str],
    target_key: str,
    profile: str,
) -> dict[str, Any]:
    exp_tag = "accel_%s_%s" % (profile, target_key)
    command = [
        sys.executable,
        "-m",
        "smart",
        "--config",
        args.config,
        "--set",
        f"mcts.mcts_iter={args.mcts_iter}",
        "--set",
        f"mcts.max_step={args.max_step}",
        "--set",
        f"mcts.exp_tag={exp_tag}",
        "--set",
        f"mcts.transposition_table_size={args.transposition_table_size}",
    ]
    for key, value in PROFILE_SETS[profile]:
        command.extend(["--set", f"{key}={value}"])
    command.extend(["mcts", "--category", target["category"], "--mesh", target["mesh"], "--force"])

    elapsed_runs = []
    completed = None
    for _ in range(args.repeat):
        started = time.perf_counter()
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        elapsed_runs.append(time.perf_counter() - started)
        if completed.returncode != 0:
            break
    assert completed is not None
    rust_stats_path, rust_stats = _load_rust_stats(args.config, target, exp_tag)
    runtime_stats = _runtime_stats_from_stdout(completed.stdout)
    if rust_stats:
        runtime_stats.update(_runtime_stats_from_rust_stats(rust_stats))
    return {
        "command": command,
        "elapsed_sec": sum(elapsed_runs) / len(elapsed_runs),
        "elapsed_runs_sec": elapsed_runs,
        "returncode": completed.returncode,
        "runtime_stats": runtime_stats,
        "rust_stats_path": str(rust_stats_path) if rust_stats_path else None,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }


def _run_eval(
    args: argparse.Namespace,
    target: dict[str, str],
    output_path: Path,
    target_key: str,
    profile: str,
) -> dict[str, Any]:
    eval_path = output_path.with_name(
        f"{output_path.stem}_{target_key}_{profile}_eval.json"
    )
    command = [
        sys.executable,
        "-m",
        "smart",
        "--config",
        args.config,
        "evaluate",
        "--stage",
        "mcts",
        "--category",
        target["category"],
        "--mesh",
        target["mesh"],
        "--chamfer-points",
        str(args.chamfer_points),
        "--output",
        str(eval_path),
        "--json",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    result: dict[str, Any] = {
        "evaluation_command": command,
        "evaluation_path": str(eval_path),
        "evaluation_time_sec": time.perf_counter() - started,
        "evaluation_returncode": completed.returncode,
        "evaluation_stdout_tail": completed.stdout[-1000:],
        "evaluation_stderr_tail": completed.stderr[-1000:],
    }
    if completed.returncode == 0:
        result["summary"] = json.loads(eval_path.read_text(encoding="utf-8"))["summary"]
    return result


def _speedups(records: dict[str, Any], baseline: str) -> dict[str, float]:
    if baseline not in records:
        return {}
    baseline_time = float(records[baseline]["elapsed_sec"])
    return {
        profile: baseline_time / float(record["elapsed_sec"])
        for profile, record in records.items()
        if float(record.get("elapsed_sec", 0.0)) > 0.0
    }


def _metric_diffs(records: dict[str, Any], baseline: str) -> dict[str, dict[str, float]]:
    baseline_record = records.get(baseline, {})
    baseline_summary = baseline_record.get("summary")
    if baseline_summary is None:
        return {}
    out: dict[str, dict[str, float]] = {}
    for profile, record in records.items():
        summary = record.get("summary")
        if summary is None:
            continue
        out[profile] = {
            key: abs(float(baseline_summary[key]) - float(summary[key]))
            for key in METRIC_KEYS
        }
    return out


def _aggregate(targets: dict[str, Any], baseline: str) -> dict[str, Any]:
    speed_values: dict[str, list[float]] = {}
    max_diffs: dict[str, dict[str, float]] = {}
    runtime_stats = _aggregate_runtime_stats(targets)
    for target in targets.values():
        for profile, speedup in target.get("speedup_vs_baseline", {}).items():
            speed_values.setdefault(profile, []).append(float(speedup))
        for profile, diffs in target.get("metric_diffs_vs_baseline", {}).items():
            profile_diffs = max_diffs.setdefault(profile, {key: 0.0 for key in METRIC_KEYS})
            for key, value in diffs.items():
                profile_diffs[key] = max(profile_diffs[key], float(value))

    mean_speedup = {
        profile: sum(values) / len(values)
        for profile, values in speed_values.items()
        if values
    }
    metric_identical = {
        profile: all(value == 0.0 for value in diffs.values())
        for profile, diffs in max_diffs.items()
    }
    best_metric_identical = None
    for profile, speedup in sorted(mean_speedup.items(), key=lambda row: row[1], reverse=True):
        if profile != baseline and metric_identical.get(profile, False):
            best_metric_identical = {"profile": profile, "mean_speedup": speedup}
            break
    return {
        "mean_speedup_vs_baseline": mean_speedup,
        "max_metric_diffs_vs_baseline": max_diffs,
        "metric_identical_vs_baseline": metric_identical,
        "best_metric_identical": best_metric_identical,
        "runtime_stats": runtime_stats,
    }


def _runtime_stats_from_stdout(stdout: str) -> dict[str, dict[str, float]]:
    specs = {
        "candidate_prefilter": "MCTS candidate prefilter:",
        "manifold_stateful_cache": "MCTS manifold_stateful cache:",
    }
    out: dict[str, dict[str, float]] = {}
    for name, prefix in specs.items():
        for line in stdout.splitlines():
            if prefix not in line:
                continue
            payload = line.split(prefix, 1)[1].strip()
            try:
                raw = json.loads(payload)
            except json.JSONDecodeError:
                continue
            out[name] = {
                str(key): float(value)
                for key, value in raw.items()
                if isinstance(value, (int, float))
            }
    return out


def _load_rust_stats(
    config_path: str, target: dict[str, str], exp_tag: str
) -> tuple[Path | None, dict[str, Any] | None]:
    workspace = Path(str(load_config(config_path).get("workspace", "runs")))
    root = workspace / "mcts" / target["category"]
    if not root.exists():
        return None, None
    matches = [
        path
        for path in root.rglob("rust_stats.json")
        if exp_tag in str(path) and target["mesh"] in str(path)
    ]
    if not matches:
        return None, None
    path = max(matches, key=lambda item: item.stat().st_mtime)
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return path, None


def _runtime_stats_from_rust_stats(stats: dict[str, Any]) -> dict[str, dict[str, float]]:
    groups = {
        "candidate_prefilter": "candidate_prefilter_",
        "manifold_stateful_cache": "manifold_state_",
    }
    out: dict[str, dict[str, float]] = {}
    for group, prefix in groups.items():
        values = {
            key[len(prefix) :]: float(value)
            for key, value in stats.items()
            if key.startswith(prefix) and isinstance(value, (int, float))
        }
        if values:
            out[group] = values
    return out


def _aggregate_runtime_stats(targets: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    totals: dict[str, dict[str, dict[str, float]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for target in targets.values():
        for profile, record in target.get("runs", {}).items():
            for group, stats in record.get("runtime_stats", {}).items():
                group_totals = totals.setdefault(profile, {}).setdefault(group, {})
                counts.setdefault(profile, {}).setdefault(group, 0)
                counts[profile][group] += 1
                for key, value in stats.items():
                    group_totals[key] = group_totals.get(key, 0.0) + float(value)

    out: dict[str, dict[str, dict[str, float]]] = {}
    for profile, groups in totals.items():
        out[profile] = {}
        for group, stats in groups.items():
            record = dict(stats)
            count = counts.get(profile, {}).get(group, 0)
            if count:
                for key, value in stats.items():
                    record["mean_%s" % key] = value / count
            if group == "candidate_prefilter":
                calls = record.get("calls", 0.0)
                actions_total = record.get("actions_total", 0.0)
                exact_calls = record.get("proxy_exact", 0.0) + record.get(
                    "fallback_exact", 0.0
                )
                proxy_candidates = record.get("proxy_candidates", 0.0)
                fallback_exact = record.get("fallback_exact", 0.0)
                if calls:
                    record["mean_exact_calls_per_prefilter_call"] = exact_calls / calls
                    record["mean_actions_per_prefilter_call"] = actions_total / calls
                    record["mean_proxy_candidates_per_prefilter_call"] = (
                        proxy_candidates / calls
                    )
                if exact_calls:
                    record["fallback_exact_fraction"] = fallback_exact / exact_calls
            out[profile][group] = record
    return out


def _target_key(target: dict[str, str]) -> str:
    return f"{target['category']}__{target['mesh']}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
