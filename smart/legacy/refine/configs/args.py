import argparse
from argparse import ArgumentParser


def get_parser() -> ArgumentParser:

    """parse input arguments"""
    parser = argparse.ArgumentParser(description="BoundingBoxRefine-Agent")

    parser.add_argument(
        "--debug", action="store_true", default=False, help="when debuging"
    )
    parser.add_argument(
        "--print_off", action="store_true", default=False, help="turn off printing option"
    )
    parser.add_argument(
        "--run_type",
        type=str,
        default="",
        choices=["train", "mcts", "eval", "greedy", "test_env"],
        help="running type: choose from train/eval/greedy/test_env",
    )
    parser.add_argument(
        "--init_type",
        type=str,
        default="bsp",
        choices=["bsp", "coacd", "fps", "random ", ""],
        help="choose initialization type",
    )

    parser.add_argument("--gpu", type=str, default="0", help="which gpu to use")
    parser.add_argument("--worker", type=int, default=8, help="number of workers to use")
    parser.add_argument(
        "--seed", type=int, default=7777, help="seeds for number pseudo random generators"
    )
    parser.add_argument(
        "--path_to_pt_file",
        type=str,
        default="",
        help="directory to checkpoint file to load agent",
    )
    parser.add_argument(
        "--load_idx",
        type=int,
        default=0,
        help="agent checkpoint load idx",
    )
    parser.add_argument(
        "--path_to_msh_file",
        type=str,
        default="../Mesh2Tet/final_data/shapenet_table_e0.004_l0.2",
        help="data directory",
    )
    parser.add_argument(
        "--path_to_bbox",
        type=str,
        default="",
        help="path to rl result path",
    )

    parser.add_argument(
        "--meshes",
        nargs="+",
        default=[""],
        help="meshes to use in training",
    )
    parser.add_argument(
        "--mesh_txt",
        type=str,
        default="",
        help="text file of meshes to use in training",
    )
    parser.add_argument(
        "--all",
        default=False,
        action="store_true",
        help="use all meshes",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="meshes to exclude in training/testing",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="./logs/mcts",
        help="directory to save the training log (tensorboard)",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="./logs/rl/exp",
        help="directory to save the training log (tensorboard)",
    )
    parser.add_argument(
        "--data_batch_size",
        type=int,
        default=1,
        help="number of data batch to use in training",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="number of batch to use in training"
    )
    parser.add_argument(
        "--num_bbox",
        type=int,
        default=0,
        help="number of bounding boxes to fit to our shape",
    )
    parser.add_argument(
        "--merge_eps",
        type=float,
        default=2e-2,
        help="Number of final partitions to make",
    )
    parser.add_argument(
        "--fast_merge",
        default=False,
        action="store_true",
        help="whether to use fast merge init",
    )
    parser.add_argument(
        "--bbox_init",
        type=str,
        default="grd_merged",
        choices=["bsp_preseg", "grd_merged", "bbox_direct", "random"],
        help="selecting which initialization to use for the bounding boxes",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="",
        choices=["ha", "cubseg", "vp", "mcts", ""],
        help="selecting which initialization to use for the bounding boxes",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=100,
        help="number of max steps to perform on our shape",
    )
    parser.add_argument(
        "--greedy_backend",
        type=str,
        default="auto",
        choices=[
            "auto",
            "python",
            "cpp",
            "cpp_stateful",
            "cpp_native",
            "native",
            "native_stateful",
        ],
        help=(
            "greedy refine runner backend. auto uses the native C++ callback "
            "runner when available."
        ),
    )
    parser.add_argument(
        "--skip_initial_render",
        default=False,
        action="store_true",
        help="skip exporting the initial bbox state before refinement/search",
    )
    parser.add_argument(
        "--skip_render_partition",
        default=False,
        action="store_true",
        help="export bbox OBJ files only, without per-tet partition msh/mesh snapshots",
    )
    parser.add_argument(
        "--skip_summary_metrics",
        default=False,
        action="store_true",
        help="skip log-only SMART metric summaries during inference; bbox outputs are unchanged",
    )
    parser.add_argument(
        "--action_unit",
        type=float,
        default=1e-2,
        help="minimum action unit for bbox perturbation",
    )
    parser.add_argument(
        "--num_action_scale",
        type=int,
        default=1,
        help="number of action scales to use in action space",
    )

    parser.add_argument("--lr", type=float, default=1e-5, help="initial learning rate")
    parser.add_argument(
        "--lr_final", type=float, default=1e-7, help="final learning rate"
    )
    parser.add_argument(
        "--lr_fraction",
        type=float,
        default=0.5,
        help="fraction of timestep when learning rate becomes final learing rate",
    )
    parser.add_argument(
        "--learning_start",
        type=int,
        default=10,
        help="number of samples to collect before agent actually learns",
    )

    parser.add_argument("--tau", type=float, default=1.0, help="soft update coefficient")
    parser.add_argument("--gamma", type=float, default=1.0, help="the discount factor")

    # MCTS
    parser.add_argument(
        "--mcts_iter",
        type=int,
        default=3000,
        help="number of trajectories to collect per each update",
    )
    parser.add_argument(
        "--mcts_backend",
        type=str,
        default="auto",
        choices=[
            "auto",
            "python",
            "cpp",
            "cpp_stateful",
            "cpp_native",
            "native",
            "native_stateful",
        ],
        help=(
            "MCTS runner backend. cpp_stateful/native_stateful run the native "
            "C++ callback runner."
        ),
    )
    parser.add_argument(
        "--exp_w",
        type=float,
        default=1.0,
        help="number of trajectories to collect per each update",
    )

    parser.add_argument(
        "--mask_prun",
        default=False,
        action="store_true",
        help="whether to use mask pruning in mcts search",
    )
    parser.add_argument(
        "--grdexp",
        default=False,
        action="store_true",
        help="whether to use greedy expansion",
    )
    parser.add_argument(
        "--pns",
        default=False,
        action="store_true",
        help="whether to use prioritized node selection",
    )
    parser.add_argument(
        "--skip_rate",
        type=float,
        default=0.7,
        help="Skip rate when using prioritized node selection",
    )
    parser.add_argument(
        "--transposition_table",
        default=False,
        action="store_true",
        help="reuse MCTS node statistics for repeated bbox states",
    )
    parser.add_argument(
        "--transposition_table_size",
        type=int,
        default=8192,
        help="maximum number of MCTS repeated-state entries to keep",
    )
    parser.add_argument(
        "--mcts_cpp_rng",
        default=False,
        action="store_true",
        help=(
            "experimental: use the native C++ PRNG inside the C++ MCTS callback "
            "runner instead of calling numpy.random for every random draw. This "
            "changes the exact random sequence and therefore search order."
        ),
    )
    parser.add_argument(
        "--mcts_cpp_rng_seed",
        type=int,
        default=7777,
        help="seed for --mcts_cpp_rng",
    )
    parser.add_argument(
        "--action_prior_path",
        type=str,
        default="",
        help="optional JSON action-prior file learned from SMART traces; guides MCTS sampling only",
    )
    parser.add_argument(
        "--action_prior_device",
        type=str,
        default="json",
        help=(
            "inference backend for action-level learned priors: json/python, cpu, "
            "mps, cuda, or auto. auto tries Apple Silicon MPS first, then CUDA, "
            "then CPU. Exact SMART reward is unchanged."
        ),
    )
    parser.add_argument(
        "--action_prior_weight",
        type=float,
        default=0.0,
        help="weight for optional trace/RL action prior logits during MCTS PNS sampling",
    )
    parser.add_argument(
        "--puct_prior_weight",
        type=float,
        default=0.0,
        help="optional PUCT-style child-selection prior weight; uses action_prior_path and changes search order",
    )
    parser.add_argument(
        "--action_value_weight",
        type=float,
        default=0.0,
        help=(
            "optional action-value prior weight for policy-value models; "
            "uses action_prior_path and changes search order"
        ),
    )
    parser.add_argument(
        "--action_prior_top_k",
        type=int,
        default=0,
        help=(
            "research search-pruning knob. For local_refine, exact-score only "
            "the top-K learned proposal actions plus top-K cheap upper-bound "
            "actions. For MCTS, keep only the top-K policy/value actions per "
            "tree node. 0 evaluates all legacy actions."
        ),
    )
    parser.add_argument(
        "--action_prior_select",
        type=str,
        default="legacy",
        choices=["legacy", "best", "softmax"],
        help=(
            "research MCTS expansion policy for untried actions after optional "
            "top-K pruning. legacy keeps the original random/PNS choice, best "
            "expands the highest policy/value action, and softmax samples by "
            "policy/value proposal score."
        ),
    )
    parser.add_argument(
        "--action_prior_select_temperature",
        type=float,
        default=1.0,
        help="softmax temperature for --action_prior_select softmax",
    )
    parser.add_argument(
        "--action_prior_keep_upper",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "For local_refine action_prior_top_k, also keep top cheap upper-bound "
            "actions alongside learned actions. Disable only for guarded RL "
            "research runs where the learned policy should be allowed to change "
            "the local-search branch more aggressively."
        ),
    )
    parser.add_argument(
        "--escape_policy",
        default=False,
        action="store_true",
        help=(
            "research option: after MCTS stagnates, keep extra policy-ranked "
            "escape actions per node so top-K pruning does not collapse to a "
            "single local-search branch"
        ),
    )
    parser.add_argument(
        "--escape_after_no_update",
        type=int,
        default=20,
        help="activate --escape_policy after this many iterations without a best-reward update",
    )
    parser.add_argument(
        "--escape_action_top_k",
        type=int,
        default=0,
        help="number of additional non-primary actions to keep when --escape_policy is active",
    )
    parser.add_argument(
        "--escape_probability",
        type=float,
        default=0.5,
        help="probability of expanding a kept escape action instead of the primary top-K action",
    )
    parser.add_argument(
        "--mcts_exp_tag",
        type=str,
        default="",
        help="optional suffix for separating MCTS experiment output folders",
    )

    parser.add_argument(
        "--train_freq",
        type=int,
        default=4,
        help="number of trajectories to collect per each update",
    )
    parser.add_argument(
        "--grad_step",
        type=int,
        default=1,
        help="number of updates to perform per iteration",
    )
    parser.add_argument(
        "--imit_grad_step",
        type=int,
        default=100,
        help="number of updates to perform per iteration",
    )

    parser.add_argument(
        "--target_update_interval",
        type=int,
        default=5000,
        help="update the target network every target_update_interval environment steps",
    )

    parser.add_argument(
        "--exp",
        type=float,
        default=1.0,
        help="initial value of random action probability",
    )
    parser.add_argument(
        "--exp_final",
        type=float,
        default=0.05,
        help="final value of random action probability",
    )
    parser.add_argument(
        "--exp_fraction",
        type=float,
        default=0.4,
        help="fraction of entire training period over which the exploration rate is reduced",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=10,
        help="the maximum value for the gradient clipping",
    )

    parser.add_argument(
        "--tilted",
        default=False,
        action="store_true",
        help="whether to use titled bounding box",
    )

    parser.add_argument(
        "--mov",
        default=False,
        action="store_true",
        help="whether to use MOV metric in the reward",
    )
    parser.add_argument(
        "--mov_alpha",
        type=float,
        default=1.0,
        help="weight of reward to give at MOV",
    )
    parser.add_argument(
        "--tov",
        default=False,
        action="store_true",
        help="whether to use TOV metric in the reward",
    )
    parser.add_argument(
        "--tov_beta",
        type=float,
        default=1.0,
        help="weight of reward to give at TOV",
    )
    parser.add_argument(
        "--cover_penalty",
        type=float,
        default=10.0,
        help="reward penalty to give when the mesh in not completely covered with bounding boxes",
    )
    parser.add_argument(
        "--score_cache_size",
        type=int,
        default=4096,
        help="maximum number of bbox score/coverage states cached per mesh; set 0 to disable",
    )
    parser.add_argument(
        "--cache_initial_bbox_state",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "cache deterministic initial bbox parameters across reset() calls; "
            "this avoids repeated OBJ loading/oriented-bounds work during MCTS "
            "without changing exact rewards"
        ),
    )
    parser.add_argument(
        "--candidate_backend",
        type=str,
        default="exact",
        choices=["exact", "bitset_topk"],
        help=(
            "candidate scoring helper. exact preserves the legacy exhaustive scan; "
            "bitset_topk uses the native C++ centroid bitset proxy to pre-score top-K "
            "actions, then still verifies with the exact reward and falls back "
            "to legacy upper-bound scanning when needed."
        ),
    )
    parser.add_argument(
        "--candidate_top_k",
        type=int,
        default=8,
        help="number of bitset proxy actions to exact-score before legacy fallback",
    )
    parser.add_argument(
        "--candidate_require_exact_fallback",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "when candidate_backend=bitset_topk, keep the legacy exact fallback "
            "scan after top-K proxy actions. Disable only for guarded/search-order "
            "experiments that intentionally trade exhaustive action parity for fewer "
            "exact Manifold calls."
        ),
    )
    parser.add_argument(
        "--candidate_pruned_categories",
        type=str,
        default="",
        help=(
            "optional comma-separated category allowlist for disabling exact "
            "fallback. When set and the current category is not listed, "
            "candidate_backend=bitset_topk keeps the exact fallback even if "
            "--no-candidate_require_exact_fallback was requested."
        ),
    )
    parser.add_argument(
        "--candidate_pruned_max_aspect_mean",
        type=float,
        default=0.0,
        help=(
            "optional geometry guard for no-fallback bitset pruning. When > 0, "
            "states whose mean bbox aspect ratio is at or above this value keep "
            "the exact fallback even if pruning was requested."
        ),
    )
    parser.add_argument(
        "--candidate_pruned_min_fill_ratio",
        type=float,
        default=0.0,
        help=(
            "optional geometry guard for no-fallback bitset pruning. When > 0, "
            "states whose bbox volume / total AABB volume is below this value keep "
            "the exact fallback even if pruning was requested."
        ),
    )
    parser.add_argument(
        "--candidate_bypass_on_exact_fallback",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "when candidate_backend=bitset_topk and exact fallback is required "
            "by category or geometry guards, bypass the bitset prefilter entirely "
            "and use the legacy exact greedy path. This is slower but preserves "
            "search order for guarded profiles."
        ),
    )
    parser.add_argument(
        "--reward_backend",
        type=str,
        default="manifold",
        choices=["manifold", "manifold_bridge", "manifold_stateful", "tet_clipping"],
        help=(
            "reward metric backend. manifold is the exact legacy default; "
            "manifold_bridge reuses the fixed C++ Manifold library through the native bridge; "
            "manifold_stateful keeps exact Manifold rewards in a stateful native cache; "
            "tet_clipping is an experimental native backend gated by parity tests."
        ),
    )
    parser.add_argument(
        "--stateful_union_cache",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="enable the exact stateful Manifold current-state/reward cache for manifold_stateful",
    )
    parser.add_argument(
        "--stateful_cache_capacity",
        type=int,
        default=65536,
        help="maximum candidate score entries cached by reward_backend=manifold_stateful",
    )
    parser.add_argument(
        "--manifold_volume_method",
        type=str,
        default="mesh",
        choices=["mesh", "properties"],
        help=(
            "Manifold residual volume extraction. mesh preserves the legacy "
            "GetMesh signed-volume path; properties uses GetProperties().volume "
            "and is an opt-in research speed path."
        ),
    )
    parser.add_argument(
        "--stateful_unscored_apply",
        action="store_true",
        help=(
            "experimental exact path: let native stateful MCTS apply unscored axis "
            "actions through the stateful Manifold bridge instead of legacy env.step"
        ),
    )
    parser.add_argument(
        "--mcts_fused_rollout_step",
        action="store_true",
        help=(
            "experimental exact path: fuse one MCTS greedy rollout batch-score/apply "
            "step into one Python env call. Opt-in because timing is workload-dependent."
        ),
    )
    parser.add_argument(
        "--mcts_native_axis_rollout_step",
        action="store_true",
        help=(
            "experimental exact path: for reward_backend=manifold_stateful, run the "
            "axis-only MCTS greedy rollout step inside the C++ Manifold state. "
            "This intentionally excludes recenter candidates until their parity path "
            "is ported, so it is search-order-changing and opt-in only."
        ),
    )
    parser.add_argument(
        "--mcts_native_axis_rollout_segment",
        action="store_true",
        help=(
            "experimental exact path: for reward_backend=manifold_stateful, run "
            "multiple axis-only MCTS greedy rollout steps inside the C++ Manifold "
            "state and sync Python once at the end. This is search-order-changing "
            "because recenter candidates and per-step state keys remain on the "
            "legacy Python path."
        ),
    )
    parser.add_argument(
        "--mcts_no_reward_stop_after",
        type=int,
        default=101,
        help=(
            "stop MCTS after this many completed iterations when the best rollout "
            "reward is still below 1e-2. Default preserves the legacy late stop."
        ),
    )
    parser.add_argument(
        "--forced_first_action",
        type=int,
        default=-1,
        help=(
            "research/local-minimum escape option: apply this action before "
            "normal greedy or MCTS rollout logic, then continue with the legacy "
            "search. -1 disables it."
        ),
    )
    parser.add_argument(
        "--forced_action_sequence",
        type=str,
        default="",
        help=(
            "comma-separated research/local-minimum escape action prefix. "
            "When set, these actions are applied before normal greedy/MCTS "
            "logic and supersede --forced_first_action."
        ),
    )
    parser.add_argument(
        "--forced_first_action_min_reward",
        type=float,
        default=0.0,
        help=(
            "minimum immediate reward required before applying "
            "--forced_first_action. Use a negative value for escape-branch "
            "data collection."
        ),
    )
    parser.add_argument(
        "--trace_actions_path",
        type=str,
        default="",
        help="optional JSONL path for accepted refine/MCTS action traces used by later policy/RL experiments",
    )
    parser.add_argument(
        "--candidate_trace_path",
        type=str,
        default="",
        help=(
            "optional JSONL path for MCTS candidate traces. This records already "
            "scored rollout candidates for policy/RL training without changing search."
        ),
    )
    parser.add_argument(
        "--candidate_trace_top_k",
        type=int,
        default=0,
        help=(
            "maximum rollout candidate rows to write per MCTS rollout step. "
            "0 records all candidates when candidate_trace_path is set."
        ),
    )
    parser.add_argument(
        "--candidate_trace_node_top_k",
        type=int,
        default=0,
        help=(
            "maximum MCTS node untried-action candidate rows to write before "
            "expansion. This is intended for local-minimum escape policy/value "
            "training; 0 disables node-action candidate tracing."
        ),
    )
    parser.add_argument(
        "--category",
        type=str,
        default="",
        help="optional ShapeNet/category label written into action traces",
    )
    parser.add_argument(
        "--tet_clipping_max_boxes",
        type=int,
        default=12,
        help="maximum bbox count accepted by the native tet-clipping reward backend",
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="attn",
        help="type of agent to use",
    )
    parser.add_argument(
        "--ddqn",
        default=False,
        action="store_true",
        help="whether to use double DQN learning update",
    )
    parser.add_argument(
        "--duel",
        default=False,
        action="store_true",
        help="whether to use dueling network",
    )
    parser.add_argument(
        "--noisy",
        default=False,
        action="store_true",
        help="whether to use noisy Linear or not",
    )

    parser.add_argument(
        "--buffer_size", type=int, default=2**17, help="capacity of a buffer"
    )
    parser.add_argument(
        "--n_step",
        type=int,
        default=1,
        help="n in N-step learning",
    )
    parser.add_argument(
        "--per",
        default=False,
        action="store_true",
        help="whether to use prioritized experience replay",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="alpha value of PER",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="beta value of PER",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=1,
        help="number of heads to use in the transformer self attention layer",
    )
    parser.add_argument(
        "--edge_conv",
        default=False,
        action="store_true",
        help="whether to use edge_conv encoder or not",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="tag for experiments",
    )

    return parser
