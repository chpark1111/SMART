from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline.config import load_config, workspace_path
from .pipeline.stages import STAGE_ORDER, data_status, iter_stage_records, list_mesh_ids, run_pipeline
from .pipeline.tools import build_rust_extension, build_tools, diagnose_environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smart", description="SMART official pipeline")
    parser.add_argument("--config", default="configs/demo.yaml", help="Pipeline config path")
    parser.add_argument("--dry-run", action="store_true", help="Print and manifest work without executing external tools")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set mcts.mcts_iter=100 --set render.joint_mesh=true",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run enabled stages in pipeline order")
    _stage_options(run)

    for stage in [
        "normalize",
        "tetra",
        "preseg",
        "merge",
        "refine",
        "mcts",
        "local_refine",
        "render",
    ]:
        stage_parser = sub.add_parser(stage, help=f"Run only the {stage} stage")
        _stage_options(stage_parser)

    build = sub.add_parser("build-tools", help="Download/build Mesh2Tet tools and the vendored Manifold Python binding")
    build.add_argument("--only-manifold-binding", action="store_true", help="Build only smart/vendor/manifold pymanifold")
    build.add_argument("--dry-run", action="store_true", help="Print intended build commands without executing them")

    build_rust = sub.add_parser("build-rust", help="Build/install the local smart-bbox wheel with bundled smart._rust")
    build_rust.add_argument("--debug", action="store_true", help="Build a debug extension instead of release")
    build_rust.add_argument("--dry-run", action="store_true", help="Print intended build commands without executing them")

    check = sub.add_parser("check-data", help="Summarize configured ShapeNet sample data")
    check.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    summary = sub.add_parser("summary", help="Summarize the latest manifest record for each stage/mesh")
    summary.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    doctor = sub.add_parser("doctor", help="Check local SMART runtime, build tools, and optional Rust tooling")
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    evaluate = sub.add_parser("evaluate", help="Evaluate SMART bbox outputs with paper metrics")
    evaluate.add_argument(
        "--stage",
        default="mcts",
        choices=["merge", "refine", "mcts", "mcts_guarded", "local_refine", "local_refine_guarded"],
        help="BBox output stage to evaluate",
    )
    evaluate.add_argument("--category", help="Limit to one configured category")
    evaluate.add_argument("--mesh", action="append", help="Limit to one mesh id; repeat for multiple meshes")
    evaluate.add_argument("--chamfer-points", type=int, default=2048, help="Surface samples for cub_CD")
    evaluate.add_argument("--output", help="Path to write evaluation JSON")
    evaluate.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    prior = sub.add_parser("build-prior", help="Build an opt-in MCTS action-prior JSON from trace files")
    prior.add_argument("traces", nargs="+", help="Trace JSONL file(s) from mcts.trace_actions_path")
    prior.add_argument("--output", required=True, help="Path to write the prior JSON")
    prior.add_argument(
        "--model-type",
        choices=["counts", "linear", "mlp", "rl-mlp", "pg-agent"],
        default="counts",
        help="Prior model to train: count logits, linear logits, supervised MLP, coord/scale RL MLP, or action-level policy-gradient agent",
    )
    prior.add_argument("--min-reward", type=float, default=0.0, help="Only actions at or above this reward contribute")
    prior.add_argument("--smoothing", type=float, default=1.0, help="Additive count smoothing")
    prior.add_argument("--reward-power", type=float, default=1.0, help="Exponent applied to positive rewards")
    prior.add_argument("--include-action-logits", action="store_true", help="Also write per-action logits for same-layout experiments")
    prior.add_argument("--num-action-scale", type=int, default=0, help="Override coord/scale key count; default infers from traces")
    prior.add_argument("--epochs", type=int, default=80, help="Training epochs for learned priors")
    prior.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for learned priors")
    prior.add_argument("--l2", type=float, default=1.0e-4, help="L2 regularization for learned priors")
    prior.add_argument("--hidden-size", type=int, default=16, help="Hidden units for --model-type mlp")
    prior.add_argument("--device", default="auto", help="PyTorch device for --model-type mlp: auto, mps, cuda, or cpu")
    prior.add_argument(
        "--advantage-baseline",
        choices=["category", "mesh", "global", "none"],
        default="category",
        help="Reward baseline for --model-type rl-mlp",
    )
    prior.add_argument("--advantage-clip", type=float, default=5.0, help="Normalized advantage clip for --model-type rl-mlp")
    prior.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus for --model-type rl-mlp")
    prior.add_argument("--max-logit-abs", type=float, default=8.0, help="Calibrate RL prior logits to this max absolute value")
    prior.add_argument("--json", action="store_true", help="Emit full prior JSON instead of metadata only")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    for override in args.overrides:
        _apply_override(cfg, override)

    if args.command == "build-tools":
        if args.only_manifold_binding:
            from .pipeline.tools import build_vendored_manifold_binding

            messages = build_vendored_manifold_binding(cfg, dry_run=args.dry_run)
        else:
            messages = build_tools(cfg, dry_run=args.dry_run)
        for message in messages:
            print(message)
        return 0

    if args.command == "build-rust":
        messages = build_rust_extension(cfg, dry_run=args.dry_run, release=not args.debug)
        for message in messages:
            print(message)
        return 0

    if args.command == "check-data":
        status = data_status(cfg)
        if args.json:
            print(json.dumps(status, indent=2, sort_keys=True))
        else:
            for name, item in status.items():
                if name == "manifest":
                    continue
                print(f"{name}: {item['model_obj_count']} model.obj files, sample bbox diagonal={item['sample_bbox_diagonal']}")
        return 0

    if args.command == "summary":
        status = _manifest_summary(cfg, workspace_path(cfg) / "manifests")
        if args.json:
            print(json.dumps(status, indent=2, sort_keys=True))
        else:
            for stage in STAGE_ORDER:
                counts = status["by_stage"].get(stage)
                if counts:
                    formatted = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
                    print(f"{stage}: {formatted}")
        return 0

    if args.command == "doctor":
        status = diagnose_environment(cfg)
        if args.json:
            print(json.dumps(status, indent=2, sort_keys=True))
        else:
            for check in status["checks"]:
                marker = "ok" if check["ok"] else "missing"
                required = ",".join(check["required_for"])
                suffix = f" [{required}]" if required else ""
                path = f" -> {check['path']}" if check.get("path") else ""
                detail = f" ({check['detail']})" if check.get("detail") else ""
                print(f"{marker}: {check['name']}{suffix}{path}{detail}")
        return 0

    if args.command == "evaluate":
        from .evaluation import evaluate_config

        status = evaluate_config(
            cfg,
            stage=args.stage,
            category_name=args.category,
            meshes=args.mesh,
            chamfer_points=args.chamfer_points,
            output_path=args.output,
        )
        if args.json:
            print(json.dumps(status, indent=2, sort_keys=True))
        else:
            summary = status["summary"]
            print(
                "stage={stage} total={total} success={success} failed={failed} "
                "Avg_num_box={num_box} Avg_BVS={bvs} Avg_MOV={mov} Avg_TOV={tov} "
                "Avg_Covered={covered} Avg_vIoU={viou} Avg_cub_CD={cd} output={output}".format(
                    stage=args.stage,
                    total=summary["total"],
                    success=summary["success"],
                    failed=summary["failed"],
                    num_box=summary["Avg_num_box"],
                    bvs=summary["Avg_BVS"],
                    mov=summary["Avg_MOV"],
                    tov=summary["Avg_TOV"],
                    covered=summary["Avg_Covered"],
                    viou=summary["Avg_vIoU"],
                    cd=summary["Avg_cub_CD"],
                    output=status["output_path"],
                )
            )
        return 0

    if args.command == "build-prior":
        if args.model_type == "linear":
            from .action_prior import build_linear_action_prior_from_traces

            prior_payload = build_linear_action_prior_from_traces(
                args.traces,
                output=args.output,
                min_reward=args.min_reward,
                smoothing=args.smoothing,
                reward_power=args.reward_power,
                num_action_scale=args.num_action_scale or None,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                l2=args.l2,
            )
        elif args.model_type == "mlp":
            from .action_prior import build_mlp_action_prior_from_traces

            prior_payload = build_mlp_action_prior_from_traces(
                args.traces,
                output=args.output,
                min_reward=args.min_reward,
                smoothing=args.smoothing,
                reward_power=args.reward_power,
                num_action_scale=args.num_action_scale or None,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                l2=args.l2,
                hidden_size=args.hidden_size,
                device=args.device,
            )
        elif args.model_type == "rl-mlp":
            from .action_prior import build_rl_mlp_action_prior_from_traces

            prior_payload = build_rl_mlp_action_prior_from_traces(
                args.traces,
                output=args.output,
                min_reward=args.min_reward,
                smoothing=args.smoothing,
                num_action_scale=args.num_action_scale or None,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                l2=args.l2,
                hidden_size=args.hidden_size,
                device=args.device,
                advantage_baseline=args.advantage_baseline,
                advantage_clip=args.advantage_clip,
                entropy_coef=args.entropy_coef,
                max_logit_abs=args.max_logit_abs,
            )
        elif args.model_type == "pg-agent":
            from .action_prior import build_policy_gradient_action_prior_from_traces

            prior_payload = build_policy_gradient_action_prior_from_traces(
                args.traces,
                output=args.output,
                min_reward=args.min_reward,
                smoothing=args.smoothing,
                num_action_scale=args.num_action_scale or None,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                l2=args.l2,
                hidden_size=args.hidden_size,
                device=args.device,
                advantage_baseline=args.advantage_baseline,
                advantage_clip=args.advantage_clip,
                entropy_coef=args.entropy_coef,
                max_logit_abs=args.max_logit_abs,
            )
        else:
            from .action_prior import build_action_prior_from_traces

            prior_payload = build_action_prior_from_traces(
                args.traces,
                output=args.output,
                min_reward=args.min_reward,
                smoothing=args.smoothing,
                reward_power=args.reward_power,
                include_action_logits=args.include_action_logits,
                num_action_scale=args.num_action_scale or None,
            )
        payload = prior_payload if args.json else prior_payload["metadata"]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    only_stage = None if args.command == "run" else args.command
    records = run_pipeline(
        cfg,
        only_stage=only_stage,
        category_name=args.category,
        meshes=args.mesh,
        dry_run=args.dry_run,
        force=args.force,
    )
    summary = iter_stage_records(records)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _stage_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--category", help="Limit to one configured category")
    parser.add_argument("--mesh", action="append", help="Limit to one mesh id; repeat for multiple meshes")
    parser.add_argument("--force", action="store_true", help="Re-run even when expected output exists")
    parser.add_argument("--dry-run", action="store_true", help="Manifest intended work without executing external tools")


def _apply_override(cfg: dict[str, object], expression: str) -> None:
    if "=" not in expression:
        raise SystemExit(f"Invalid --set expression, expected KEY=VALUE: {expression}")
    key, raw_value = expression.split("=", 1)
    path = [part for part in key.split(".") if part]
    if not path:
        raise SystemExit(f"Invalid --set key: {expression}")

    value = _parse_override_value(raw_value)
    target: dict[str, object] = cfg
    for part in path[:-1]:
        existing = target.get(part)
        if existing is None:
            existing = {}
            target[part] = existing
        if not isinstance(existing, dict):
            raise SystemExit(f"Cannot override nested key under non-dict config value: {part}")
        target = existing
    target[path[-1]] = value


def _parse_override_value(raw_value: str) -> object:
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        lowered = raw_value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"none", "null"}:
            return None
        return raw_value


def _manifest_summary(cfg: dict[str, object], manifest_root: Path) -> dict[str, object]:
    allowed_meshes = {
        (str(category["name"]), mesh_id)
        for category in cfg.get("categories", [])
        for mesh_id in list_mesh_ids(category)
    }
    latest: dict[tuple[str, str, str], dict[str, object]] = {}
    if manifest_root.exists():
        for path in manifest_root.glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (
                    str(record.get("stage", "")),
                    str(record.get("category", "")),
                    str(record.get("mesh_id", "")),
                )
                if allowed_meshes and (key[1], key[2]) not in allowed_meshes:
                    continue
                previous = latest.get(key)
                if previous is None or float(record.get("finished_at", 0)) >= float(previous.get("finished_at", 0)):
                    latest[key] = record

    by_stage: dict[str, dict[str, int]] = {}
    for record in latest.values():
        stage = str(record.get("stage", "unknown"))
        status = str(record.get("status", "unknown"))
        by_stage.setdefault(stage, {})
        by_stage[stage][status] = by_stage[stage].get(status, 0) + 1
    return {"manifest_root": str(manifest_root), "by_stage": by_stage, "records": list(latest.values())}
