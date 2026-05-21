from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline.config import REPO_ROOT, load_config, workspace_path
from .pipeline.stages import (
    STAGE_ORDER,
    data_status,
    iter_stage_records,
    list_mesh_ids,
    normalized_mesh_path,
    run_native_pipelines,
    run_pipeline,
)
from .pipeline.tools import build_cpp_extension, build_tools, diagnose_environment


def _build_messages_failed(messages: list[str]) -> bool:
    return any(": failed" in message or message.startswith("Missing ") for message in messages)


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
    public_commands = (
        "{run,native-run,normalize,tetra,preseg,merge,refine,mcts,local_refine,"
        "render,build-tools,build-cpp,audit-wheel,check-data,configs,assets,"
        "summary,doctor,evaluate}"
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar=public_commands)

    run = sub.add_parser("run", help="Run enabled stages in pipeline order")
    _stage_options(run)

    native_run = sub.add_parser(
        "native-run",
        help="Run each configured mesh through the monolithic smart-cpp-native pipeline",
    )
    _stage_options(native_run)

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

    build = sub.add_parser(
        "build-tools",
        help="Download/build Mesh2Tet tools, CoACD source, and the vendored Manifold Python binding",
    )
    build.add_argument("--only-manifold-binding", action="store_true", help="Build only smart/vendor/manifold pymanifold")
    build.add_argument(
        "--dry-run",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Print intended build commands without executing them",
    )

    build_cpp = sub.add_parser("build-cpp", help="Build/install the native C++ smart._cpp extension")
    build_cpp.add_argument("--debug", action="store_true", help="Build a debug extension instead of release")
    build_cpp.add_argument(
        "--asan",
        action="store_true",
        help="Also build build/smart-cpp-native-asan with AddressSanitizer for source-checkout diagnostics",
    )
    build_cpp.add_argument(
        "--dry-run",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Print intended build commands without executing them",
    )

    audit_wheel = sub.add_parser("audit-wheel", help="Audit SMART release wheel/sdist contents before upload")
    audit_wheel.add_argument("wheels", nargs="+", help="Wheel/sdist paths or glob patterns")
    audit_wheel.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    audit_wheel.add_argument(
        "--cpp-only",
        action="store_true",
        help="Compatibility no-op; native C++ wheels are the default",
    )

    check = sub.add_parser("check-data", help="Summarize configured ShapeNet sample data")
    check.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    configs = sub.add_parser("configs", help="List bundled SMART config profiles")
    configs.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    assets = sub.add_parser("assets", help="List optional SMART model assets")
    assets.add_argument("--kind", choices=["gates", "priors"], help="Limit to one asset kind")
    assets.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    summary = sub.add_parser("summary", help="Summarize the latest manifest record for each stage/mesh")
    summary.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    doctor = sub.add_parser("doctor", help="Check local SMART runtime, build tools, and native C++ tooling")
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    evaluate = sub.add_parser("evaluate", help="Evaluate SMART bbox outputs with paper metrics")
    evaluate.add_argument(
        "--stage",
        default="mcts",
        help="BBox output stage to evaluate, including custom guarded stages",
    )
    evaluate.add_argument("--category", help="Limit to one configured category")
    evaluate.add_argument("--mesh", action="append", help="Limit to one mesh id; repeat for multiple meshes")
    evaluate.add_argument("--chamfer-points", type=int, default=2048, help="Surface samples for cub_CD")
    evaluate.add_argument("--output", help="Path to write evaluation JSON")
    evaluate.add_argument(
        "--from-manifest",
        action="store_true",
        help="Evaluate only successful records listed in the stage manifest",
    )
    evaluate.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    prior = sub.add_parser("build-prior", help=argparse.SUPPRESS)
    prior.add_argument("traces", nargs="*", help="Trace JSONL file(s) from mcts.trace_actions_path")
    prior.add_argument(
        "--trace-list",
        action="append",
        default=[],
        help="Text file with one trace path per line. Lines after # are ignored. Repeatable.",
    )
    prior.add_argument(
        "--category-filter",
        default="",
        help="Comma-separated category filter for per-category action-prior/RL agents",
    )
    prior.add_argument(
        "--source-filter",
        default="",
        help=(
            "Comma-separated trace source filter, e.g. mcts_node_untried or "
            "mcts_candidate. Applied before training."
        ),
    )
    prior.add_argument(
        "--reward-field",
        default="reward",
        help=(
            "Trace field to train as reward. Use final_quality_score for final-return "
            "quality learning, or final_learned_win_score for guarded learned-win labels."
        ),
    )
    prior.add_argument(
        "--reward-min-positive",
        type=float,
        default=0.0,
        help="Minimum positive final_quality_score required when using final_learned_win_score",
    )
    prior.add_argument(
        "--reward-abs-min",
        type=float,
        default=0.0,
        help="Drop rows whose selected training reward has absolute value below this threshold",
    )
    prior.add_argument("--output", required=True, help="Path to write the prior JSON")
    prior.add_argument(
        "--model-type",
        choices=["counts", "linear", "mlp", "rl-mlp", "pg-agent", "policy-value"],
        default="counts",
        help="Prior model to train: count logits, linear logits, supervised MLP, coord/scale RL MLP, action policy, or action policy-value agent",
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
    prior.add_argument("--accepted-weight", type=float, default=1.0, help="PG/policy-value loss weight for accepted SMART trace rows")
    prior.add_argument("--candidate-weight", type=float, default=1.0, help="PG/policy-value loss weight for mcts_candidate rows")
    prior.add_argument(
        "--selected-candidate-weight",
        type=float,
        default=1.0,
        help="Extra PG/policy-value multiplier for selected mcts_candidate rows",
    )
    prior.add_argument(
        "--category-balance",
        action="store_true",
        help="Rebalance PG/policy-value examples so categories contribute similar total loss",
    )
    prior.add_argument("--value-epochs", type=int, default=0, help="Policy-value value-head epochs; default reuses --epochs")
    prior.add_argument(
        "--value-learning-rate",
        type=float,
        default=0.0,
        help="Policy-value value-head learning rate; default reuses --learning-rate",
    )
    prior.add_argument("--value-clip", type=float, default=5.0, help="Policy-value normalized action-value target clip")
    prior.add_argument(
        "--value-positive-weight",
        type=float,
        default=1.0,
        help="Policy-value loss multiplier for positive value targets",
    )
    prior.add_argument(
        "--value-negative-weight",
        type=float,
        default=1.0,
        help="Policy-value loss multiplier for negative value targets",
    )
    prior.add_argument(
        "--value-zero-weight",
        type=float,
        default=1.0,
        help="Policy-value loss multiplier for zero value targets",
    )
    prior.add_argument(
        "--value-auto-balance",
        action="store_true",
        help="Policy-value inverse-frequency balance for positive/negative/zero value targets",
    )
    prior.add_argument(
        "--value-group-balance",
        action="store_true",
        help="Policy-value normalize loss weight per candidate group/node",
    )
    prior.add_argument(
        "--value-pairwise-weight",
        type=float,
        default=0.0,
        help=(
            "Policy-value ranking loss weight. Positive values train same-run "
            "positive final-return actions to rank above zero/negative actions."
        ),
    )
    prior.add_argument("--value-pairwise-margin", type=float, default=0.1, help="Margin for --value-pairwise-weight ranking loss")
    prior.add_argument(
        "--value-pairwise-max-pairs",
        type=int,
        default=200000,
        help="Maximum deterministic action pairs used by the policy-value ranking loss",
    )
    prior.add_argument(
        "--value-validation-mesh-fraction",
        type=float,
        default=0.0,
        help="Hold out this fraction of mesh ids for value-head validation metrics",
    )
    prior.add_argument(
        "--policy-base-prior",
        default="",
        help="For --model-type policy-value, reuse this action policy and train only the value head",
    )
    prior.add_argument("--json", action="store_true", help="Emit full prior JSON instead of metadata only")

    export_proposal = sub.add_parser(
        "export-box-proposal-dataset",
        help=argparse.SUPPRESS,
    )
    export_proposal.add_argument("--stage", default="mcts", help="BBox stage to use as pseudo-labels")
    export_proposal.add_argument("--category", help="Limit to one configured category")
    export_proposal.add_argument("--mesh", action="append", help="Limit to one mesh id; repeat for multiple meshes")
    export_proposal.add_argument("--from-manifest", action="store_true", help="Use successful stage manifest rows")
    export_proposal.add_argument("--max-boxes", type=int, default=32, help="Maximum labels per mesh")
    export_proposal.add_argument(
        "--label-format",
        choices=["corners", "basis", "axis_aligned"],
        default="corners",
        help="Pseudo-label representation. corners preserves vertices; basis predicts center plus three half-axis vectors.",
    )
    export_proposal.add_argument("--output", required=True, help="Output JSONL dataset path")
    export_proposal.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    train_proposal = sub.add_parser(
        "train-box-proposal",
        help=argparse.SUPPRESS,
    )
    train_proposal.add_argument("dataset", help="Dataset JSONL from export-box-proposal-dataset")
    train_proposal.add_argument("--output", required=True, help="Output PyTorch checkpoint path")
    train_proposal.add_argument("--num-points", type=int, default=1024, help="Mesh surface samples per training item")
    train_proposal.add_argument("--max-boxes", type=int, default=16, help="Maximum predicted box slots")
    train_proposal.add_argument("--epochs", type=int, default=50, help="Training epochs")
    train_proposal.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    train_proposal.add_argument("--learning-rate", type=float, default=1.0e-3, help="Adam learning rate")
    train_proposal.add_argument("--hidden-size", type=int, default=128, help="PointNet hidden width")
    train_proposal.add_argument("--device", default="auto", help="PyTorch device: auto, mps, cuda, or cpu")
    train_proposal.add_argument("--seed", type=int, default=0, help="Sampling/training seed")
    train_proposal.add_argument("--validation-fraction", type=float, default=0.0, help="Hold out this fraction and save the best validation checkpoint")
    train_proposal.add_argument(
        "--coverage-loss-weight",
        type=float,
        default=0.0,
        help="Optional soft point-coverage loss weight for learned proposal training",
    )
    train_proposal.add_argument(
        "--coverage-temperature",
        type=float,
        default=0.05,
        help="Soft box boundary temperature used by --coverage-loss-weight",
    )
    train_proposal.add_argument(
        "--compactness-loss-weight",
        type=float,
        default=0.0,
        help="Optional volume compactness loss weight to prevent oversized learned proposal boxes",
    )
    train_proposal.add_argument(
        "--architecture",
        choices=["query_pointnet", "global_pointnet"],
        default="query_pointnet",
        help="query_pointnet uses learned box-slot queries; global_pointnet preserves the older flat head",
    )
    train_proposal.add_argument(
        "--structured-basis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For basis labels, constrain network outputs to bounded center plus orthogonal half-axis vectors",
    )
    train_proposal.add_argument(
        "--loss-mode",
        choices=["matched", "slot"],
        default="matched",
        help="matched is permutation-invariant over bbox slots and corner order; slot keeps the older fixed-order loss",
    )
    train_proposal.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    predict_proposal = sub.add_parser(
        "predict-box-proposal",
        help=argparse.SUPPRESS,
    )
    predict_proposal.add_argument("--model", required=True, help="PyTorch checkpoint from train-box-proposal")
    predict_proposal.add_argument("--mesh-path", required=True, help="Input normalized mesh OBJ")
    predict_proposal.add_argument("--output-dir", required=True, help="Output directory for bbox OBJ files")
    predict_proposal.add_argument("--category", default="", help="Category name for category-conditioned checkpoints")
    predict_proposal.add_argument("--mesh-id", default="", help="Mesh id for legacy SMART bbox layout")
    predict_proposal.add_argument("--num-points", type=int, default=0, help="Override checkpoint point sample count")
    predict_proposal.add_argument("--score-threshold", type=float, default=0.5, help="Objectness threshold")
    predict_proposal.add_argument("--max-boxes", type=int, default=0, help="Limit output boxes")
    predict_proposal.add_argument("--min-boxes", type=int, default=0, help="Always keep at least this many top-scoring boxes")
    predict_proposal.add_argument("--nms-iou-threshold", type=float, default=1.0, help="AABB IoU NMS threshold; 1 disables NMS")
    predict_proposal.add_argument(
        "--coverage-calibration-target",
        type=float,
        default=0.0,
        help="If >0, uniformly scale selected proposal boxes until sampled point coverage reaches this target",
    )
    predict_proposal.add_argument(
        "--coverage-calibration-max-scale",
        type=float,
        default=2.0,
        help="Maximum uniform scale used by --coverage-calibration-target",
    )
    predict_proposal.add_argument(
        "--legacy-layout",
        action="store_true",
        help="Write updated0/<mesh_id>/bboxs_steps0 layout usable by bbox_init=bbox_direct",
    )
    predict_proposal.add_argument("--seed", type=int, default=0, help="Point sampling seed")
    predict_proposal.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    predict_proposals = sub.add_parser(
        "predict-box-proposals",
        help=argparse.SUPPRESS,
    )
    predict_proposals.add_argument("--model", required=True, help="PyTorch checkpoint from train-box-proposal")
    predict_proposals.add_argument(
        "--output-root",
        required=True,
        help="Common proposal root. Use this path as refine.path_to_bbox or mcts.path_to_bbox.",
    )
    predict_proposals.add_argument("--category", help="Limit to one configured category")
    predict_proposals.add_argument("--mesh", action="append", help="Limit to one mesh id; repeat for multiple meshes")
    predict_proposals.add_argument("--num-points", type=int, default=0, help="Override checkpoint point sample count")
    predict_proposals.add_argument("--score-threshold", type=float, default=0.5, help="Objectness threshold")
    predict_proposals.add_argument("--max-boxes", type=int, default=0, help="Limit output boxes")
    predict_proposals.add_argument("--min-boxes", type=int, default=0, help="Always keep at least this many top-scoring boxes")
    predict_proposals.add_argument("--nms-iou-threshold", type=float, default=1.0, help="AABB IoU NMS threshold; 1 disables NMS")
    predict_proposals.add_argument(
        "--coverage-calibration-target",
        type=float,
        default=0.0,
        help="If >0, uniformly scale selected proposal boxes until sampled point coverage reaches this target",
    )
    predict_proposals.add_argument(
        "--coverage-calibration-max-scale",
        type=float,
        default=2.0,
        help="Maximum uniform scale used by --coverage-calibration-target",
    )
    predict_proposals.add_argument("--seed", type=int, default=0, help="Base point sampling seed")
    predict_proposals.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    # Keep research-only training commands callable for local compatibility, but
    # do not show them in the public paper-reproduction help.
    sub._choices_actions = [  # type: ignore[attr-defined]
        action for action in sub._choices_actions if action.help != argparse.SUPPRESS  # type: ignore[attr-defined]
    ]

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
        return 1 if not args.dry_run and _build_messages_failed(messages) else 0

    if args.command == "build-cpp":
        messages = build_cpp_extension(cfg, dry_run=args.dry_run, release=not args.debug, asan=args.asan)
        for message in messages:
            print(message)
        return 1 if not args.dry_run and _build_messages_failed(messages) else 0

    if args.command == "audit-wheel":
        from scripts.audit_release_wheel import main as audit_release_wheel_main

        audit_args = []
        if args.json:
            audit_args.append("--json")
        if args.cpp_only:
            audit_args.append("--cpp-only")
        return audit_release_wheel_main(audit_args + args.wheels)

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

    if args.command == "configs":
        from .api import config_profiles

        profiles = config_profiles()
        if args.json:
            print(json.dumps(profiles, indent=2, sort_keys=True))
        else:
            for item in profiles:
                locations = []
                if item["root_path"]:
                    locations.append("repo")
                if item["packaged_path"]:
                    locations.append("package")
                print(f"{item['name']}: {','.join(locations)}")
        return 0

    if args.command == "assets":
        from .api import asset_profiles

        profiles = asset_profiles(args.kind)
        if args.json:
            print(json.dumps(profiles, indent=2, sort_keys=True))
        else:
            for item in profiles:
                aliases = f" aliases={','.join(item['aliases'])}" if item["aliases"] else ""
                policy = f" policy={item['policy_type']}" if item["policy_type"] else ""
                print(f"{item['kind']}/{item['name']}{aliases}{policy}")
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
            from_manifest=args.from_manifest,
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
        traces = _expand_trace_inputs(args.traces, args.trace_list)
        if not traces:
            raise SystemExit("build-prior requires trace files or --trace-list")
        if (
            args.reward_field in {"final_quality_score", "final_learned_win_score"}
            and args.advantage_baseline == "category"
        ):
            args.advantage_baseline = "none"
        traces, temporary_traces = _filter_prior_trace_inputs(
            traces,
            source_filter=args.source_filter,
            category_filter=args.category_filter,
            reward_field=args.reward_field,
            reward_min_positive=args.reward_min_positive,
            reward_abs_min=args.reward_abs_min,
            output=Path(args.output),
        )
        if args.model_type == "linear":
            from .action_prior import build_linear_action_prior_from_traces

            prior_payload = build_linear_action_prior_from_traces(
                traces,
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
                traces,
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
                traces,
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
                traces,
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
                accepted_weight=args.accepted_weight,
                candidate_weight=args.candidate_weight,
                selected_candidate_weight=args.selected_candidate_weight,
                category_balance=args.category_balance,
            )
        elif args.model_type == "policy-value":
            from .action_prior import build_policy_value_action_prior_from_traces

            prior_payload = build_policy_value_action_prior_from_traces(
                traces,
                output=args.output,
                policy_base_prior=args.policy_base_prior or None,
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
                accepted_weight=args.accepted_weight,
                candidate_weight=args.candidate_weight,
                selected_candidate_weight=args.selected_candidate_weight,
                category_balance=args.category_balance,
                value_epochs=args.value_epochs or None,
                value_learning_rate=args.value_learning_rate or None,
                value_clip=args.value_clip,
                value_positive_weight=args.value_positive_weight,
                value_negative_weight=args.value_negative_weight,
                value_zero_weight=args.value_zero_weight,
                value_auto_balance=args.value_auto_balance,
                value_group_balance=args.value_group_balance,
                value_pairwise_weight=args.value_pairwise_weight,
                value_pairwise_margin=args.value_pairwise_margin,
                value_pairwise_max_pairs=args.value_pairwise_max_pairs,
                value_validation_mesh_fraction=args.value_validation_mesh_fraction,
            )
        else:
            from .action_prior import build_action_prior_from_traces

            prior_payload = build_action_prior_from_traces(
                traces,
                output=args.output,
                min_reward=args.min_reward,
                smoothing=args.smoothing,
                reward_power=args.reward_power,
                include_action_logits=args.include_action_logits,
                num_action_scale=args.num_action_scale or None,
            )
        try:
            prior_payload["metadata"]["category_filter"] = [
                item.strip()
                for item in str(args.category_filter or "").split(",")
                if item.strip()
            ]
            prior_payload["metadata"]["source_filter"] = [
                item.strip()
                for item in str(args.source_filter or "").split(",")
                if item.strip()
            ]
            prior_payload["metadata"]["reward_field"] = str(args.reward_field or "reward")
            if float(args.reward_min_positive) != 0.0:
                prior_payload["metadata"]["reward_min_positive"] = float(args.reward_min_positive)
            if float(args.reward_abs_min) != 0.0:
                prior_payload["metadata"]["reward_abs_min"] = float(args.reward_abs_min)
            if (
                prior_payload["metadata"]["category_filter"]
                or prior_payload["metadata"]["source_filter"]
                or str(args.reward_field or "reward") != "reward"
                or float(args.reward_abs_min) > 0.0
            ):
                input_files = [str(path) for path in _expand_trace_inputs(args.traces, args.trace_list)]
                prior_payload["metadata"]["category_filter_input_files"] = input_files
                prior_payload["metadata"]["source_filter_input_files"] = input_files
                prior_payload["metadata"]["trace_files"] = input_files
                if "value_trace_files" in prior_payload["metadata"]:
                    prior_payload["metadata"]["value_trace_files"] = input_files
            payload = prior_payload if args.json else prior_payload["metadata"]
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        finally:
            for path in temporary_traces:
                try:
                    path.unlink()
                except OSError:
                    pass

    if args.command == "export-box-proposal-dataset":
        from .box_proposal import export_box_proposal_dataset

        status = export_box_proposal_dataset(
            cfg,
            output=args.output,
            stage=args.stage,
            category_name=args.category,
            meshes=args.mesh,
            from_manifest=args.from_manifest,
            max_boxes=args.max_boxes,
            label_format=args.label_format,
        )
        if args.json:
            print(json.dumps(status, indent=2, sort_keys=True))
        else:
            print(
                "output={output} stage={stage} records={records} categories={categories} max_boxes={max_boxes}".format(
                    **status
                )
            )
        return 0

    if args.command == "train-box-proposal":
        from .box_proposal import train_box_proposal_model

        status = train_box_proposal_model(
            args.dataset,
            output=args.output,
            num_points=args.num_points,
            max_boxes=args.max_boxes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            device=args.device,
            seed=args.seed,
            loss_mode=args.loss_mode,
            architecture=args.architecture,
            validation_fraction=args.validation_fraction,
            structured_basis=args.structured_basis,
            coverage_loss_weight=args.coverage_loss_weight,
            coverage_temperature=args.coverage_temperature,
            compactness_loss_weight=args.compactness_loss_weight,
        )
        if args.json:
            print(json.dumps(status, indent=2, sort_keys=True))
        else:
            print(
                "output={output} records={records} categories={categories} max_boxes={max_boxes} final_loss={final_loss}".format(
                    **status
                )
            )
        return 0

    if args.command == "predict-box-proposal":
        from .box_proposal import predict_box_proposals

        status = predict_box_proposals(
            args.model,
            args.mesh_path,
            output_dir=args.output_dir,
            category=args.category,
            mesh_id=args.mesh_id or None,
            num_points=args.num_points or None,
            score_threshold=args.score_threshold,
            max_boxes=args.max_boxes or None,
            min_boxes=args.min_boxes,
            nms_iou_threshold=args.nms_iou_threshold,
            coverage_calibration_target=args.coverage_calibration_target,
            coverage_calibration_max_scale=args.coverage_calibration_max_scale,
            legacy_layout=args.legacy_layout,
            seed=args.seed,
        )
        if args.json:
            print(json.dumps(status, indent=2, sort_keys=True))
        else:
            print(
                "output_dir={output_dir} num_boxes={num_boxes} category={category} mesh_id={mesh_id}".format(
                    **status
                )
            )
        return 0

    if args.command == "predict-box-proposals":
        from .box_proposal import predict_box_proposals

        records = []
        for category in cfg.get("categories", []):
            if args.category and str(category.get("name", "")) != args.category:
                continue
            for mesh_index, mesh_id in enumerate(list_mesh_ids(category, explicit=args.mesh)):
                mesh_path = normalized_mesh_path(cfg, category, mesh_id)
                if not mesh_path.exists():
                    mesh_path = Path(category["mesh_root"]) / mesh_id / "model.obj"
                    if not mesh_path.is_absolute():
                        mesh_path = REPO_ROOT / mesh_path
                if not mesh_path.exists():
                    records.append(
                        {
                            "category": str(category.get("name", "")),
                            "mesh_id": str(mesh_id),
                            "status": "skipped",
                            "error": f"missing mesh: {mesh_path}",
                        }
                    )
                    continue
                try:
                    record = predict_box_proposals(
                        args.model,
                        mesh_path,
                        output_dir=args.output_root,
                        category=str(category.get("name", "")),
                        mesh_id=str(mesh_id),
                        num_points=args.num_points or None,
                        score_threshold=args.score_threshold,
                        max_boxes=args.max_boxes or None,
                        min_boxes=args.min_boxes,
                        nms_iou_threshold=args.nms_iou_threshold,
                        coverage_calibration_target=args.coverage_calibration_target,
                        coverage_calibration_max_scale=args.coverage_calibration_max_scale,
                        legacy_layout=True,
                        seed=int(args.seed) + mesh_index,
                    )
                    record["status"] = "success"
                    records.append(record)
                except Exception as exc:
                    records.append(
                        {
                            "category": str(category.get("name", "")),
                            "mesh_id": str(mesh_id),
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
        status = {
            "model": str(args.model),
            "output_root": str(args.output_root),
            "total": len(records),
            "success": sum(1 for record in records if record.get("status") == "success"),
            "failed": sum(1 for record in records if record.get("status") == "failed"),
            "skipped": sum(1 for record in records if record.get("status") == "skipped"),
            "records": records,
        }
        if args.json:
            print(json.dumps(status, indent=2, sort_keys=True))
        else:
            print(
                "output_root={output_root} total={total} success={success} failed={failed} skipped={skipped}".format(
                    **status
                )
            )
        return 0

    if args.command == "native-run":
        records = run_native_pipelines(
            cfg,
            category_name=args.category,
            meshes=args.mesh,
            dry_run=args.dry_run,
            force=args.force,
        )
    else:
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Manifest intended work without executing external tools",
    )


def _expand_trace_inputs(traces: list[str], trace_lists: list[str]) -> list[str]:
    expanded = list(traces)
    for trace_list in trace_lists:
        with Path(trace_list).open("r", encoding="utf-8") as file:
            for line in file:
                item = line.split("#", 1)[0].strip()
                if item:
                    expanded.append(item)
    return expanded


def _filter_prior_trace_inputs(
    traces: list[str],
    *,
    source_filter: str,
    category_filter: str,
    reward_field: str,
    reward_min_positive: float,
    reward_abs_min: float,
    output: Path,
) -> tuple[list[str], list[Path]]:
    allowed_sources = {
        item.strip()
        for item in str(source_filter or "").split(",")
        if item.strip()
    }
    allowed_categories = {
        item.strip()
        for item in str(category_filter or "").split(",")
        if item.strip()
    }
    reward_field = str(reward_field or "reward")
    if (
        not allowed_sources
        and not allowed_categories
        and reward_field == "reward"
        and float(reward_abs_min) <= 0.0
    ):
        return list(traces), []
    filtered = output.with_suffix(".prior_filter.tmp.jsonl")
    filtered.parent.mkdir(parents=True, exist_ok=True)
    used = 0
    seen = 0
    with filtered.open("w", encoding="utf-8") as out_file:
        for trace in traces:
            with Path(trace).open("r", encoding="utf-8") as in_file:
                for line in in_file:
                    if not line.strip():
                        continue
                    seen += 1
                    record = json.loads(line)
                    if allowed_sources and str(record.get("source", "")) not in allowed_sources:
                        continue
                    if allowed_categories and str(record.get("category", "")) not in allowed_categories:
                        continue
                    if reward_field != "reward":
                        record["source_reward"] = float(record.get("reward", 0.0) or 0.0)
                        record["reward"] = _record_reward_for_field(
                            record,
                            reward_field,
                            reward_min_positive=reward_min_positive,
                        )
                    if abs(float(record.get("reward", 0.0) or 0.0)) < float(reward_abs_min):
                        continue
                    out_file.write(json.dumps(record, sort_keys=True) + "\n")
                    used += 1
    if used == 0:
        try:
            filtered.unlink()
        except OSError:
            pass
        raise SystemExit(
            "No trace rows matched --source-filter %s / --category-filter %s / --reward-abs-min %g out of %d rows"
            % (
                ",".join(sorted(allowed_sources)),
                ",".join(sorted(allowed_categories)),
                float(reward_abs_min),
                seen,
            )
        )
    return [str(filtered)], [filtered]


def _record_reward_for_field(
    record: dict[str, object],
    reward_field: str,
    *,
    reward_min_positive: float,
) -> float:
    if reward_field == "final_learned_win_score":
        label = str(record.get("run_label", ""))
        if label == "baseline":
            return 0.0
        if bool(record.get("final_not_worse", False)) and (
            bool(record.get("selected_run", False)) or bool(record.get("final_improved", False))
        ):
            if float(record.get("final_quality_score", 0.0) or 0.0) < float(reward_min_positive):
                return 0.0
            return 1.0
        if not bool(record.get("final_not_worse", False)):
            return -1.0
        return 0.0
    if reward_field in record:
        return float(record.get(reward_field, 0.0) or 0.0)
    raise SystemExit(f"Trace reward field not found: {reward_field}")


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
