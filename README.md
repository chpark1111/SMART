# Split, Merge, and Refine: Fitting Tight Bounding Boxes via Over-Segmentation and Iterative Search (SMART), 3DV 2024

![teaser](./docs/teaser.png)

[**arXiv**](https://arxiv.org/abs/2304.04336) | [**Paper**](https://arxiv.org/pdf/2304.04336)

[Chanhyeok Park](https://chpark1111.github.io/), [Minhyuk Sung](https://mhsung.github.io/)

# Introduction

This repository contains the official implementation of **Split, Merge, and Refine: Fitting Tight Bounding Boxes via Over-Segmentation and Iterative Search a.k.a. SMART**. SMART is a shape abstraction **technique for finding tight bounding boxes of 3D meshes** without requiring any human supvervision. These rich bounding boxes can be utilized for various downstream tasks like **segmentation, deformation, generation, abstraction, collision detection, and more**.

> Achieving tight bounding boxes of a shape while guaranteeing complete boundness is an essential task for efficient geometric operations and unsupervised semantic part detection. But previous methods fail to achieve both full coverage and tightness. Neural-network-based methods are not suitable for these goals due to the non-differentiability of the objective, while classic iterative search methods suffer from their sensitivity to the initialization. We propose a novel framework for finding a set of tight bounding boxes of a 3D shape via over-segmentation and iterative merging and refinement. Our result shows that utilizing effective search methods with appropriate objectives is the key to producing bounding boxes with both properties. We employ an existing pre-segmentation to split the shape and obtain over-segmentation. Then, we apply hierarchical merging with our novel tightness-aware merging and stopping criteria. To overcome the sensitivity to the initialization, we also define actions to refine the bounding box parameters in an Markov Decision Process (MDP) setup with a soft reward function promoting a wider exploration. Lastly, we further improve the refinement step with Monte Carlo Tree Search (MCTS) based multi-action space exploration. By thoughtful evaluation on diverse 3D shapes, we demonstrate full coverage, tightness, and an adequate number of bounding boxes of our method without requiring any training data or supervision. It thus can be applied to various downstream tasks in computer vision and graphics.

# Get Started

## Installation

The repo now exposes one Python entrypoint for the full SMART paper pipeline.
The package is built as one mixed Python/Rust wheel: the `smart-bbox` wheel
contains the Python package and the exact PyO3 extension at `smart._rust`.

```
cd SMART
maturin develop --release --extras pipeline
```

Alternatively, if you want pip to drive the editable install, install the
Python `maturin` build backend first and then run
`pip install -e ".[pipeline]"`.
On Apple Silicon with Homebrew Rust and a universal Python, pip/maturin can try
the missing x86_64 target. Use the arm64 target explicitly for local editable
installs:

```
CARGO_BUILD_TARGET=aarch64-apple-darwin pip install -e ".[pipeline]"
```

After a PyPI release, users should only need:

```
pip install "smart-bbox[pipeline]"
```

That install path should provide a platform wheel with Rust acceleration already
bundled. Source installs still need Rust/Cargo, maturin, and external tool
builds for the fixed Manifold bridge. The Python code imports `smart._rust` when
available and falls back to Python kernels otherwise.

For package-oriented installation, Python API examples, wheel building, and
future PyPI usage, see [`docs/PYTHON_PACKAGE.md`](docs/PYTHON_PACKAGE.md).

# Pipeline

The default demo config uses the prepared ShapeNet v2 samples:

```text
data/shapenet_airplane/<model_id>/model.obj
data/shapenet_chair/<model_id>/model.obj
data/shapenet_table/<model_id>/model.obj
```

Run a data sanity check:

```
smart --config configs/demo.yaml check-data
```

For the SMART paper categories, use only the three ShapeNet synset archives
needed by the official reproduction path:

```text
airplane  02691156.zip
chair     03001627.zip
table     04379243.zip
```

Prepare them into the official layout without mixing categories:

```bash
python3 scripts/prepare_shapenet_samples.py \
  --archive-dir /path/to/shapenet_zips \
  --output-root data/expanded \
  --categories airplane chair table \
  --limit 100000 \
  --normalize preserve

smart --config configs/expanded_full.yaml check-data
```

`configs/expanded_full.yaml` then runs the same official pipeline on
`data/expanded/shapenet_{airplane,chair,table}/<model_id>/model.obj`.
The pipeline still normalizes into `runs/expanded_full/normalized/`, so the
downloaded ShapeNet-derived files are left untouched.

Check the local runtime, external binaries, vendored Manifold binding, Blender,
and optional Rust tooling:

```
smart --config configs/demo.yaml doctor
smart --config configs/demo.yaml doctor --json
```

If the `smart` console script is not on PATH, use `python -m smart` with the
same arguments.

Build external tools. This downloads/builds ManifoldPlus and fTetWild under
`external/mesh2tet`, applies SMART's local fTetWild crash-guard patch, and
builds the fixed vendored Manifold Python binding under
`smart/vendor/manifold/build`:

```
smart --config configs/demo.yaml build-tools
```

You can also provide prebuilt binaries:

```
export SMART_MANIFOLDPLUS_BIN=/path/to/ManifoldPlus/build/manifold
export SMART_FTETWILD_BIN=/path/to/fTetWild/build/FloatTetwild_bin
export SMART_MANIFOLD_PYTHON=/path/to/smart/vendor/manifold/build/bindings/python
```

Run the complete pipeline:

```
smart --config configs/demo.yaml run
```

Run the checked five-shape smoke profile:

```
smart --config configs/smoke_5.yaml run
smart --config configs/smoke_5.yaml summary
smart --config configs/smoke_5.yaml evaluate --stage mcts
```

Run the paper-safe exact profile with the legacy Manifold reward backend:

```
smart --config configs/accelerated_exact.yaml run
```

Run the opt-in exact Rust/stateful profile:

```
smart --config configs/stateful_exact_experimental.yaml run
```

Run the current best exact MCTS acceleration candidate, which keeps the exact
Manifold reward and only enables the stateful leave-one-out union cache:

```
smart --config configs/stateful_union_cache_experimental.yaml run
```

Run the experimental Rust/stateful profile with the smoke-validated
trace-derived MCTS prior:

```
smart --config configs/accelerated_search_experimental.yaml run
```

Run the hybrid MCTS plus fine local-search profile:

```
smart --config configs/hybrid_local_search_experimental.yaml run
```

Run the guarded hybrid post-process on already generated `mcts_guarded`
outputs:

```
python3 scripts/run_quality_guarded_local_refine.py \
  --config configs/expanded_200.yaml \
  --categories airplane,chair,table \
  --per-category-limit 20 \
  --input-stage mcts_guarded \
  --max-step 100 \
  --action-unit 0.005 \
  --covered-tolerance 0.001 \
  --selection-mode improved
```

Use `--from-input-manifest` when you want the script to use every successful
mesh recorded in the input stage manifest, including meshes that are no longer
in the current sampled config order:

```
python3 scripts/run_quality_guarded_local_refine.py \
  --config configs/expanded_200.yaml \
  --input-stage mcts \
  --from-input-manifest \
  --categories airplane,chair,table \
  --per-category-limit 20 \
  --covered-tolerance 0.001 \
  --selection-mode improved
```

The stateful profiles are intentionally separate from `accelerated_exact.yaml`.
`manifold_stateful` is still opt-in: the latest smoke sweeps preserve paper
metrics and show a small MCTS speed win after exact reward memoization, but
expanded sweeps are still being accumulated. The strongest current exact MCTS
result is `stateful_union_cache=true`: on one MCTS100 10-box airplane case it
kept all reported metric differences at `0` and measured `1.095x`
(`81.80s -> 74.67s`); on the current 16 processed meshes at MCTS20 it measured
`1.047x`. Use `accelerated_exact.yaml` for paper-style reporting until larger
MCTS100/MCTS300 parity sweeps pass.
Explicit `mcts.backend=rust` and `mcts.backend=rust_stateful` are guarded behind
`mcts.allow_search_order_changes=true` because the Rust MCTS runner can change
the search trajectory even when the reward backend is exact.

Pipeline order:

```text
normalize -> tetra -> preseg -> merge -> refine -> mcts -> render
```

See [`docs/SMART_PIPELINE.md`](docs/SMART_PIPELINE.md) for the paper-to-code
stage mapping, parameter notes, and Rust migration roadmap. See
[`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) for the current applied vs
experimental status. See
[`docs/OPTIMIZATION_IDEAS.md`](docs/OPTIMIZATION_IDEAS.md) for the prioritized
runtime and quality improvements to consider before deeper Rust migration. See
[`docs/MANIFOLD_PERFORMANCE_NOTES.md`](docs/MANIFOLD_PERFORMANCE_NOTES.md) for
the fixed Manifold bridge bottleneck analysis. See
[`docs/PYTHON_PACKAGE.md`](docs/PYTHON_PACKAGE.md) for Python package install and
library usage. See [`docs/RL_RESEARCH_PLAN.md`](docs/RL_RESEARCH_PLAN.md) for the
category-general RL/action-prior research direction.

Normalization writes centered, scale-normalized OBJ files under
`runs/<profile>/normalized/` and leaves `data/` untouched. The default mode
sets each mesh bbox diagonal to `1.0`, matching the prepared ShapeNet sample
scale and the Mesh2Tet parameters. To force radius-1 unit-sphere normalization,
set `normalization.mode: unit_sphere` in the config.

Tetrahedral validation requires `tetra.msh` and a watertight
`tetra.msh__sf.obj`. Disconnected watertight surfaces are allowed by default
because ShapeNet-style meshes often contain detached semantic parts and the
downstream SMART stages can process them. Use
`--set tetra.require_single_component=true` only for strict debugging.
The validator also rejects empty/tiny fTetWild outputs through
`tetra.min_tetra_count` and `tetra.min_surface_faces`, so crash-prone or
pathological Mesh2Tet cases are skipped before CoACD/merge.

Run one stage or one mesh:

```
smart --config configs/demo.yaml normalize --category airplane --mesh 172764bea108bbcceae5a783c313eb36
smart --config configs/demo.yaml tetra --category airplane --mesh 172764bea108bbcceae5a783c313eb36
smart --config configs/demo.yaml preseg --category airplane --mesh 172764bea108bbcceae5a783c313eb36
smart --config configs/demo.yaml merge --category airplane --mesh 172764bea108bbcceae5a783c313eb36
smart --config configs/demo.yaml refine --category airplane --mesh 172764bea108bbcceae5a783c313eb36
smart --config configs/demo.yaml mcts --category airplane --mesh 172764bea108bbcceae5a783c313eb36
smart --config configs/demo.yaml render --category airplane --mesh 172764bea108bbcceae5a783c313eb36
```

Override any config value from the command line without editing YAML:

```
smart --config configs/smoke_5.yaml --set mcts.mcts_iter=100 run
smart --config configs/smoke_5.yaml --set refine.backend=python refine --category table
smart --config configs/smoke_5.yaml --set mcts.backend=python mcts --category table
smart --config configs/smoke_5.yaml --set merge.only_nearby=true merge --category table
smart --config configs/expanded_processed_16.yaml --set mcts.no_reward_stop_after=20 mcts --category airplane --mesh 123bd9e948881939c38a1d3458dafa1b
smart --config configs/smoke_5.yaml --set render.joint_mesh=true render --category airplane
smart --config configs/smoke_5.yaml --set 'render.variants=["boxes_only","with_mesh"]' render
smart --config configs/smoke_5.yaml --set refine.score_cache_size=0 refine --category chair
smart --config configs/smoke_5.yaml --set refine.summary_metrics=true refine --category chair
```

Use the same pipeline from Python:

```python
import smart

records = smart.run_pipeline(
    "configs/demo.yaml",
    category="airplane",
    meshes=["172764bea108bbcceae5a783c313eb36"],
    overrides={"mcts": {"mcts_iter": 100}},
)
metrics = smart.evaluate("configs/demo.yaml", stage="mcts", category="airplane")
status = smart.doctor("configs/demo.yaml")
```

Outputs and logs are written to `runs/<profile>/`. Each stage appends a JSONL
manifest under `runs/<profile>/manifests`, so Mesh2Tet failures are recorded and
skipped instead of stopping the whole run.

For long or expanded runs, summarize failures and timeout causes with:

```bash
python scripts/analyze_pipeline_failures.py --manifest-dir runs/expanded_full/manifests
```

This separates true stage timeouts from fTetWild crashes, external kills, and
completed MCTS outputs that only exceeded a wrapper timeout.

Evaluation uses the metric definitions from `past_codes/Evaluation`: `BVS`,
`MOV`, `TOV`, `Covered`, `vIoU`, `cub_CD`, and box count. Results are written to
`runs/<profile>/evaluation/<stage>.json`. Refine/MCTS inference skips
log-only summary metric recomputation by default through
`refine.summary_metrics=false` and `mcts.summary_metrics=false`; this does not
change bbox outputs or search rewards. Set those values to `true` when you want
the legacy per-stage stdout summaries, or run `smart evaluate` for the official
metric report.

# Renderer

Rendering defaults to `render.backend: blender` in the checked-in profiles and
uses the paper renderer path, `smart/legacy/renderer/render_teaser.py` with
`boxes.blend`. Set `SMART_BLENDER_BIN` if Blender is not on PATH. On macOS the
pipeline also checks `/Applications/Blender.app/Contents/MacOS/blender`.

The default output is box-only with transparent background, controlled by
`joint_mesh: false`, and is written as `<mesh_id>.png`. Use `render.variants`
when you want multiple outputs from the same bbox result. For example,
`variants: ["boxes_only", "with_mesh"]` produces
`<mesh_id>__boxes_only.png` and `<mesh_id>__with_mesh.png`.

Refine/MCTS also have two internal export switches:
`render_initial=false` skips the initial bbox snapshot, and
`render_partition=false` writes bbox OBJ files without the heavier per-tet
partition `.msh`/surface snapshots. The final bbox OBJ outputs needed by the
renderer are still produced.

For environments without Blender, set `render.backend: preview` or
`render.backend: auto` with `render.fallback: true`. The preview renderer is a
Matplotlib geometry check and is not the exact paper image style.

# Rust Acceleration

`smart.rust` contains the first Python/Rust parity hooks for geometry kernels:
bbox volume, valid-box masks, coverage masks, BAVF-style score loops, legacy
non-mutating merge reward arithmetic, legacy action indexing, opposite-action
pruning, MCTS child action-mask construction, axis-action bbox updates, bbox
union volume, and evaluation Chamfer distance fallback. It also provides exact
action upper-bound scoring used to skip impossible greedy/MCTS candidates before
expensive Manifold coverage checks. Greedy/MCTS now keep the bbox/action-space
summary in the Rust-backed `BBoxState` object when the extension is available.
The refine/MCTS environment also decodes legacy action ids directly and avoids
generic `deepcopy` in step rollback state. CLI startup and legacy single-mesh
inference are lazy where possible, so stage runs no longer import
evaluation/trimesh/Torch stacks before they are needed. Single-mesh inference
also reuses the validated tetra mesh load, and default merge skips nearby
adjacency setup unless `only_nearby` is explicitly enabled; fast merge keeps the
full candidate set by default and only uses nearby-restricted candidates when
that option is set. Tiny MCTS/refine action vectors keep the original Python
loops to avoid PyO3 overhead, while larger action spaces use the Rust helpers.
`refine.backend=auto` can run greedy refine control through Rust helpers when
`smart._rust` is installed, while still calling the legacy environment and fixed
C++ Manifold binding for exact geometry reward evaluation. For exact Manifold
reward backends, `mcts.backend=auto` keeps the legacy Python MCTS tree because
the Rust MCTS callback runner can change search trajectory. Force strict legacy
debugging with `refine.backend=python` or `mcts.backend=python`. The `pymesh`
shim also uses Rust for exact tetra volume,
centroid, surface-face, adjacency summaries, and Gmsh `.msh` load/save when
available. Current Rust work is exact-compatibility work: it should preserve
legacy action choices, bbox outputs, and evaluation metrics. The fixed C++
Manifold boolean binding is intentionally kept in place and is not a Rust
migration target. Do not pull, replace, rewrite, or update the vendored
`smart/vendor/manifold` source unless the SMART maintainer explicitly requests
it. Future exact Rust work should move direct bbox environment storage and
larger action scoring batches into Rust while continuing to call the existing
Manifold binding for boolean operations.

Latest exact smoke parity benchmark:
`runs/bench_exact/smoke_airplane_rotation_upper_bound.json` preserved all
reported evaluation metrics (`BVS`, `Covered`, `MOV`, `TOV`, box count, and
`vIoU`; `cub_CD` differed by floating noise only). The follow-up benchmark
`runs/bench_exact/smoke_airplane_fast_volume_formula.json` measured about
`4.85s` refine and `6.73s` MCTS on the Rust-enabled checked airplane smoke
case; `runs/bench_exact/smoke_airplane_lazy_bbox_mesh.json` keeps the same
metrics while avoiding candidate `Trimesh` bbox construction unless final
render/export needs it. The earlier table smoke
benchmark measured about `1.11x` MCTS, `1.14x` refine, and `1.10x` merge
speedup from the Rust kernels on a small table case.
Current table smoke timing after the latest exact work is
`runs/bench_exact/rust_parity_table_after_tetclip_state.json`: metrics are
unchanged, refine is about `1.55x`, and MCTS is about `1.10x` on that small
case. The exact `manifold_bridge` reward backend keeps the fixed C++ Manifold
library but stores the source mesh behind a Rust/PyO3 handle and calls the
vendored C++ boolean code directly for coverage. The current bridge path keeps
axis-action greedy reward selection and consecutive accepted axis-action refine
segments inside Rust/C++; Python syncs only the touched bbox manifolds at the
segment boundary. The legacy recenter/rotation action now uses the same
`trimesh.bounds.oriented_bounds(angle_digits=3)` candidate geometry for exact
parity, but its reward scoring and cached apply path go through the bridge. On
table repeat-3 checks,
`runs/bench_exact/reward_backend_refine_table_segment_private_repeat3.json`
reports `1.35x` refine speedup and
`runs/bench_exact/reward_backend_mcts_table_axisrust_upperdirect_repeat3.json`
reports `1.09x` MCTS speedup with zero differences in the checked evaluation
metrics. The full five-mesh smoke check in
`runs/bench_exact/smoke5_refine_segment_private_bridge_summary.json` reports
zero metric differences and `1.18x` refine speedup; wall-clock speed can still
be noisy at that small scale because process startup and Python env work
dominate some runs. MCTS rollout now also batches bbox-mask axis-action
selection through the same exact bridge, applies the selected scored action
directly, and uses a C++ wrapper path that reuses unchanged bbox Manifold
objects while testing candidate axis moves. The table repeat-3 check
`runs/bench_exact/reward_backend_mcts_table_cpp_batchboolean_repeat3.json` reports zero
metric differences and `1.09x` speedup at 10 iterations. The larger airplane
check
`runs/bench_exact/reward_backend_mcts_airplane10_cpp_batchboolean_iter300_repeat1.json`
also reports zero metric differences, but remains slower (`0.62x`) because the
exact bridge still pays repeated Manifold union/residual costs for many boxes.
Therefore the bridge remains opt-in with
`refine.reward_backend=manifold_bridge` or `mcts.reward_backend=manifold_bridge`;
the official default keeps `reward_backend=manifold`. `mcts.backend=auto` uses
the legacy Python MCTS tree for exact Manifold reward backends because the Rust
callback runner is not a stable speed win there yet.
A newer exact opt-in backend, `reward_backend=manifold_stateful`, keeps the
current bbox bounds, rotations, valid mask, cached bbox Manifold objects,
current score, rollback history, and candidate reward cache in the Rust/C++
bridge. It exposes `greedy_backend=rust_stateful` and
`mcts.backend=rust_stateful` as an explicit search-order-changing configuration
value while leaving the
official default unchanged until larger parity and timing sweeps pass. The
fixed vendored Manifold source is still not modified; only SMART's wrapper calls
it more directly. The newest stateful bridge keeps the exact Manifold bbox state
alive across action scoring and defers Python bbox manifold materialization after
accepted stateful actions until render/evaluation needs it. Accepted stateful
axis actions update only the changed bbox Manifold inside the wrapper instead of
rebuilding every bbox, and Python now receives only the changed bbox delta from
Rust/C++ instead of copying the full bbox state after every accepted axis move.
MCTS reset also caches deterministic initial bbox parameters in memory, avoiding
repeated OBJ loading/oriented-bounds work on every rollout reset. Current
checked timing is modest but metric-preserving:
table refine is `1.27x` on
`runs/bench_exact/refine_table_stateful_incremental_apply_compare.json`, table
MCTS is currently noise level. With the legacy MCTS tree (`mcts.backend=auto`),
the latest five-mesh MCTS smoke
`runs/bench_exact/exact_stateful_sweep_smoke5_mcts5_legacy_union_fresh_stats.json`
preserved all reported paper metrics exactly after matching the legacy
sequential union order with `stateful_union_cache=false`. Reward memoization is
now independent from the union cache, so repeated exact state/action scores are
reused while preserving the legacy union order. Current checked speed is
`1.032x` on
`runs/bench_exact/exact_stateful_sweep_smoke5_mcts5_rewardcache_repeat3.json`
and `1.189x` on
`runs/bench_exact/exact_stateful_sweep_smoke3_mcts20_rewardcache.json`, with
all reported metric diffs at `0`. The stateful wrapper now also builds bbox
Manifold inputs through the same explicit eight-vertex/fixed-face mesh path used
by the legacy Python backend; this fixed a smoke5 MCTS20 near-tie drift.
`runs/bench_exact/exact_stateful_sweep_smoke5_mcts20_vertexbox_trace.json`
keeps all reported metric diffs at `0`, has no accepted-action trace
divergence, and measures `1.024x`; the matching repeat-3 timing run
`runs/bench_exact/exact_stateful_sweep_smoke5_mcts20_vertexbox_repeat3.json`
keeps metric diffs at `0` and measures `1.039x`. This remains an opt-in exact profiling
backend rather than a default until larger sweeps pass. A separate
airplane 100-iteration run with `mcts.backend=rust_stateful` measured `1.33x`,
but that path is now classified as a search-order-changing experiment rather
than a paper-safe exact profile. Stateful MCTS remains opt-in until action
trajectory and metric parity pass on repeated airplane/chair/table sweeps. The
dominant remaining cost is exact Manifold boolean work, not Python loop overhead
alone.
The Rust MCTS runner now also supports two opt-in search experiments:
`mcts.transposition_table=true` stores repeated state statistics in Rust, and
`mcts.action_prior_path` plus `mcts.action_prior_weight` biases PNS exploration
from trace-derived action priors. Both can change search order, so they are
disabled by default; the pipeline requires
`mcts.allow_search_order_changes=true` before enabling the transposition table
or a nonzero action-prior weight.
Trace priors can be produced with:

```bash
smart --config configs/smoke_5.yaml build-prior \
  runs/bench_exact/traces/airplane_mcts20_trace.jsonl \
  --output runs/bench_exact/priors/airplane_mcts20_prior.json
```

The default `--model-type counts` writes portable coord/scale logits. A
state-aware linear policy, supervised PyTorch MLP, and offline RL PyTorch MLP
are also available for action-ordering experiments:

```bash
smart --config configs/smoke_5.yaml build-prior \
  runs/bench_exact/traces/airplane_mcts20_trace.jsonl \
  --output runs/bench_exact/priors/airplane_linear_prior.json \
  --model-type linear \
  --epochs 80

smart --config configs/smoke_5.yaml build-prior \
  runs/bench_exact/traces/airplane_mcts20_trace.jsonl \
  --output runs/bench_exact/priors/airplane_mlp_prior.json \
  --model-type mlp \
  --epochs 200 \
  --hidden-size 32 \
  --device auto

smart --config configs/smoke_5.yaml build-prior \
  runs/bench_exact/traces/airplane_mcts20_trace.jsonl \
  --output runs/bench_exact/priors/airplane_rl_prior.json \
  --model-type rl-mlp \
  --min-reward -1000000000 \
  --epochs 300 \
  --hidden-size 32 \
  --advantage-baseline category \
  --device auto
```

For `--model-type mlp` and `--model-type rl-mlp`, `--device auto` uses PyTorch
and probes Apple Silicon MPS first, then CUDA, then CPU. The saved JSON still
contains plain weights, so MCTS inference does not need to keep a PyTorch model
object alive.
The current packaged category-general research prior is:

```text
smart/assets/priors/category_general_expanded_full_mlp_prior.json
```

It was trained from every currently available valid action trace under
`runs/`: `79` trace files, `10,502` action records seen, `7,389`
reward-nonnegative records used, and `119` unique meshes across
airplane/chair/table. The reproducibility list is written to
`runs/bench_exact/priors/all_available_trace_files_20260514.txt`, and the
training output is mirrored at
`runs/bench_exact/priors/category_general_all_available_mlp_prior.json`.
Use it only with `mcts.allow_search_order_changes=true`; it guides action
ordering and never replaces SMART's exact reward/evaluation. Because this is a
newly trained prior, rerun the category-balanced benchmark before treating its
speed or quality as a result.
The current packaged offline-RL research prior is:

```text
smart/assets/priors/category_general_all_available_offline_rl_mlp_prior.json
```

It was trained with all valid action records, including negative-reward
exploration actions, using category-baseline advantages. The matching opt-in
profile is `configs/rl_search_experimental.yaml`. A minimal three-mesh smoke at
`mcts_iter=10`, `max_step=10`, and prior weight `0.1` measured `1.094x` mean
MCTS-stage speedup with all reported metrics identical and no quality-worse
cases; this is only a smoke result. A larger 15-mesh weight sweep at
`mcts_iter=20`, `max_step=20`
(`runs/bench_exact/rl_prior_cat5_mcts20_weight_sweep.json`) measured `1.032x`,
`1.057x`, and `1.081x` for weights `0.05`, `0.1`, and `0.2`. Weight `0.1`
had one improved case and one worse case, while `0.2` had more worse cases, so
offline RL priors remain research-only until a quality guard or stronger policy
passes a larger validation.

The same builder is also available as a Python API:

```python
import smart

smart.build_action_prior_from_traces(
    ["runs/bench_exact/traces/airplane_mcts20_trace.jsonl"],
    output="runs/bench_exact/priors/airplane_mcts20_prior.json",
)

smart.build_linear_action_prior_from_traces(
    ["runs/bench_exact/traces/airplane_mcts20_trace.jsonl"],
    output="runs/bench_exact/priors/airplane_linear_prior.json",
)

smart.build_mlp_action_prior_from_traces(
    ["runs/bench_exact/traces/airplane_mcts20_trace.jsonl"],
    output="runs/bench_exact/priors/airplane_mlp_prior.json",
    device="auto",
)
```

New traces use schema version 2 and record category, bbox/action layout, action
unit, BVS, volume method, and backend metadata. The count prior builder infers
`num_action_scale` dynamically, so it can be used for larger action-scale
experiments without hardcoded two-scale keys. The linear builder uses the same
trace schema plus state features such as BVS, step fraction, action unit, box
count, category, and cover/penalty settings; it still changes only search order,
not the exact reward.

For same-mesh/search-layout experiments, add `--include-action-logits` to write
per-action logits, then sweep weights with:

```bash
python3 scripts/sweep_mcts_action_prior.py \
  --config configs/smoke_5.yaml \
  --category airplane \
  --mesh 1f5537f4747ec847622c69c3abc6f80 \
  --mcts-iter 20 \
  --max-step 20 \
  --weights 0,0.2,0.5,1.0 \
  --make-prior \
  --include-action-logits
```

The current action-logit smoke result
`runs/bench_exact/airplane_mcts20_action_prior_actionlogits_sweep.json` keeps
all reported evaluation metrics identical to the weight-0 baseline on the
checked airplane case and measures `1.17x`, `1.21x`, and `1.35x` at weights
`0.2`, `0.5`, and `1.0`. This is not a proof of general quality improvement;
it is the first opt-in trace/RL hook. Category-general learned priors should be
trained and validated across many meshes before becoming a recommended setting.
The combined action-logit plus transposition smoke
`runs/bench_exact/airplane_mcts20_action_prior_tt_sweep.json` also keeps the
same reported metrics for the checked case and measures `1.39x` at prior
weight `1.0`, but it is still an opt-in search-order experiment.
For category-general priors, use
`scripts/benchmark_action_prior_generalization.py`; it builds leave-one-out
priors from traces of the other meshes and reports both speed and metric drift.
To collect more category-balanced traces from every configured dataset, use the
batch collector:

```bash
python3 scripts/collect_action_traces.py \
  --config configs/expanded_200.yaml \
  --categories airplane,chair,table \
  --batch-size 5 \
  --max-batches-per-category 4 \
  --mcts-iter 20 \
  --mcts-max-step 20 \
  --trace-root runs/bench_exact/trace_collection

python3 scripts/train_action_prior_from_traces.py \
  runs/bench_exact/trace_collection/*.jsonl \
  --output runs/bench_exact/priors/category_general_mlp_prior.json \
  --model-type mlp \
  --epochs 200 \
  --hidden-size 32 \
  --device auto
```

For the full airplane/chair/table ShapeNet set, switch the collector config to
`configs/expanded_full.yaml` after preparing the three category archives. This
is the current intended category-general RL/prior path: collect accepted exact
SMART actions, train a category-aware action prior, use it only to guide MCTS
action ordering, and keep final quality judged by SMART evaluation metrics.
The checked five-mesh smoke result
`runs/bench_exact/action_prior_generalization_smoke5_w01.json` uses only
portable coord/scale logits, keeps all reported metric differences at `0` for
`action_prior_weight=0.1`, and measures `1.10x` mean MCTS speedup. A stronger
weight is not automatically safer: the three-target sweep
`runs/bench_exact/action_prior_generalization_smoke3_smallw.json` shows
`weight=0.2` can change the selected result and move `BVS`, `TOV`, and `vIoU`.
So `0.1` is the current smoke-safe experimental value, not a default.
The generated smoke prior remains packaged at
`smart/assets/priors/smoke5_coord_scale_prior.json`. The opt-in profile
`configs/accelerated_search_experimental.yaml` now points to the larger
category-general MLP asset trained from the expanded airplane/chair/table trace
pool.
`mcts.stateful_unscored_apply=true` is also available as a debug-only exact
Rust-state application experiment. On the same 20-iteration airplane smoke it
was slightly slower and changed the selected final bbox snapshot because tiny
floating differences accumulated into a different MCTS branch, so it remains
off by default.
The latest exact reset/delta smoke check
`runs/smoke_5/mcts/.../1f5537f4747ec847622c69c3abc6f80_delta_on4` keeps the
same bbox OBJ files as the pre-delta stateful run and records
`initial_bbox_cache_hits=21`, `initial_bbox_cache_misses=1` for 20 MCTS
iterations. After also making bounds/rotation materialization lazy for
stateful batch scoring, the same smoke measured `8.25s`; this confirms the
Python copying/reset overhead is lower, but it still does not attack the
dominant Manifold boolean cost.
`mcts.fused_rollout_step=true` is available as another exact debug experiment:
it fuses one rollout batch-score/apply step into one env call and preserved the
same bbox OBJ outputs in the checked smoke, but it measured slower (`8.65s`) on
that case, so it remains off by default.
The stateful Manifold wrapper now also builds `union_except_i` lazily instead of
eagerly rebuilding every leave-one-out union for every state. On the checked
100-iteration airplane smoke
`runs/smoke_5/mcts/airplane/.../1f5537f4747ec847622c69c3abc6f80_lazy_except_mcts100`,
the exact metrics are unchanged (`BVS=2.119659339401995`,
`Covered=0.999203597577755`, `TOV=1.0565878326097768`,
`vIoU=0.4856670957154915`, `num_box=7`) and runtime is `18.48s` for that
`max_step=20` smoke command. The recorded cache stats show the migration is
working structurally: `except_union_builds=880`,
`except_union_cache_hits=9682`, `reward_cache_hits=8120`, and
`reward_cache_misses=10562`. This is still opt-in because the dominant cost is
the exact Manifold boolean residual calculation itself, not just Python
bookkeeping.
The pipeline now also verifies that refine/MCTS generated bbox outputs during
the current stage run instead of accidentally accepting an older bbox directory
from the same mesh. If MCTS finds no better update and would otherwise write no
result, it renders the deterministic initial bbox state as the stage output.
This keeps downstream evaluation/rendering stable and prevents misleading
benchmark "success" records with no current bbox files.
For promotion testing, use:

```bash
python3 scripts/benchmark_exact_stateful_sweep.py \
  --config configs/smoke_5.yaml \
  --stage mcts \
  --mcts-iter 5 \
  --mcts-max-step 5 \
  --chamfer-points 0 \
  --trace-actions \
  --output runs/bench_exact/exact_stateful_sweep_smoke5_mcts5.json
```

The current five-mesh smoke sweep
`runs/bench_exact/exact_stateful_sweep_smoke5_mcts5_legacy_union_fresh_stats.json`
keeps all reported metric diffs at `0`; the repeat-3 timing run
`runs/bench_exact/exact_stateful_sweep_smoke5_mcts5_rewardcache_repeat3.json`
also keeps metric diffs at `0` and measures `1.032x` mean speed after exact
reward memoization was decoupled from union caching. The larger
`runs/bench_exact/exact_stateful_sweep_smoke3_mcts20_rewardcache.json` check
keeps metric diffs at `0` and measures `1.189x`. The
`runs/bench_exact/exact_stateful_sweep_smoke5_mcts20_vertexbox_trace.json`
check extends this to all five smoke targets at `mcts_iter=20`, with no action
trace divergence and `1.024x` mean speedup. The repeat-3 timing run
`runs/bench_exact/exact_stateful_sweep_smoke5_mcts20_vertexbox_repeat3.json`
keeps metric diffs at `0` and measures `1.039x`. The older
`runs/bench_exact/exact_stateful_sweep_smoke5_mcts5_auto_trace.json` run
exposed a near-tie chair drift before the stateful wrapper matched legacy union
order and legacy bbox mesh construction, so promotion still requires larger
repeated sweeps.
Accordingly, `configs/accelerated_exact.yaml` uses legacy `manifold` reward for
paper-safe exact runs, while `configs/stateful_exact_experimental.yaml` keeps
the `manifold_stateful` backend available for opt-in profiling with the legacy
MCTS tree. `configs/candidate_bitset_exact_experimental.yaml` enables the
current bitset top-K candidate profile (`candidate_top_k=8`) with exact Manifold
verification and the legacy MCTS tree. Use
`configs/accelerated_search_experimental.yaml` or set
`mcts.allow_search_order_changes=true` when intentionally testing the Rust MCTS
runner.

The MCTS `auto` backend therefore keeps the legacy Python tree for exact Manifold reward
backends and leaves the Rust callback runner as an explicit
`mcts.backend=rust` or `mcts.backend=rust_stateful` experiment until the rollout
state is moved deeper into Rust.
The exact candidate prefilter is available as
`refine.candidate_backend=bitset_topk` or `mcts.candidate_backend=bitset_topk`.
It stores tetra centroid/volume proxy state in Rust with
`CandidateBitsetState`, ranks axis-action candidates with bitset coverage, and
then exact-verifies candidates with the current Manifold reward. The airplane
100-iteration MCTS check
`runs/bench_exact/mcts_airplane100_candidate_bitset_top3_rust_topk.json`
kept all evaluation metrics identical to exact `manifold`; timing was
`26.61s` versus the nearby exact baseline
`runs/bench_exact/mcts_airplane100_candidate_exact_after_bitset_state.json` at
`26.70s`, so this remains opt-in rather than default. A newer profile sweep
script, `scripts/benchmark_mcts_acceleration_profiles.py`, compares exact
stateful MCTS against bitset/TT/prior combinations. On
`runs/bench_exact/mcts_accel_profiles_smoke3_iter10.json`, `bitset_top8`
kept all metric diffs at `0` and measured `1.101x`; `rust_stateful_prior01_tt`
was faster (`1.157x`) but changed chair metrics. On the larger
`runs/bench_exact/mcts_accel_profiles_smoke5_iter20_exact_candidates.json`,
`bitset_top8` again kept metric diffs at `0` and measured `1.015x`, while
`rust_stateful_tt` changed airplane/chair metrics. That makes bitset top-K the
only current exact-compatible candidate from this sweep. On the category-balanced
nine processed meshes in
`runs/bench_exact/mcts_accel_profiles_expanded_processed9_iter20_topk_tt_all.json`,
`bitset_top3` and `bitset_top8` both kept all reported metric diffs at `0`;
mean speedups were `1.025x` and `1.023x`. `candidate_bitset_fast_experimental.yaml`
packages the smaller top-3 profile, while `candidate_bitset_exact_experimental.yaml`
keeps the more conservative top-8 profile. Merge greedy search uses
a heap with lazy invalidation for cached BAVF rewards; the latest change avoids
rescanning all merge candidates when validating the heap top.

The cleaner exact reward-speed candidate is now
`configs/stateful_union_cache_experimental.yaml`. It keeps the legacy MCTS
tree and exact Manifold reward but enables leave-one-out union caching inside
the Rust/C++ bridge. In
`runs/bench_exact/mcts_accel_profiles_expanded_processed9_iter20_union_cache_probe.json`,
`exact_union_cache` kept all reported metric diffs at `0` on nine processed
airplane/chair/table meshes and measured `1.097x`. The instrumented top-K run
`runs/bench_exact/mcts_accel_profiles_expanded_processed9_iter20_instrumented_topk.json`
showed why bitset top-K is modest: `bitset_top8` reduced exact reward misses
from `12477` to only `11584` because safe fallback still exact-verifies many
candidates.
The current processed-set benchmark is
`runs/bench_exact/mcts_accel_profiles_expanded_processed16_iter20_union_cache.json`:
over 16 processed meshes, `exact_union_cache` preserved all reported metrics
and measured `1.047x`. On one harder 10-box airplane MCTS100 case,
`exact_union_cache` preserved all reported metrics and reduced runtime from
`81.80s` to `74.67s`, which is still too slow for interactive development.
For development runs where MCTS is not finding improvements, the opt-in
`mcts.no_reward_stop_after=20` shortcut produced byte-identical bbox OBJ files
and identical reported metrics on that same case while reducing stage time to
`15.96s`. Keep this as a search-budget shortcut, not a paper-safe default.
`mcts.fused_rollout_step=true` is also wired into the legacy Python MCTS tree;
on three airplane MCTS20 cases it preserved reported metrics and measured
`1.013x`, so it remains workload-dependent and opt-in.
`mcts.manifold_volume_method=properties` is a newer GetProperties volume probe.
On deterministic residual-volume probes across the 16 processed meshes, it
matched the current GetMesh signed-volume path within `4.1e-08` and measured
`1.105x` at the micro-benchmark level. On one actual MCTS20 airplane target it
kept reported metrics identical and measured `1.069x`, but reward traces differ
slightly, so it stays a research flag until a larger sweep passes. The packaged
profile is `configs/properties_volume_experimental.yaml`.

MCTS/RL upgrades are being kept separate from exact acceleration. The older
trace-derived action-prior sweep
`runs/bench_exact/mcts_accel_profiles_expanded_processed16_iter20_prior_weight_sweep.json`
shows `1.043x` to `1.055x` speedups for prior weights `0.02` to `0.1`, but all
weights change one table case. MOV improves for that case, while BVS/TOV/vIoU
worsen, so the generic prior remains a research/quality experiment rather than
a default. `smart build-prior --model-type linear` and
`scripts/train_action_prior_from_traces.py --model-type linear` now train a
state-aware linear action prior from the same schema-v2 traces, and
`--model-type mlp` trains a small PyTorch MLP prior with automatic MPS/CUDA/CPU
device selection. A tiny MCTS2 leave-one-out check kept reported metric diffs at
`0` but was slower (`0.852x`), while the slightly larger three-airplane MCTS5 smoke
`runs/bench_exact/action_prior_linear_smoke3_mcts5.json` also kept metric diffs
at `0` and measured `1.037x` at prior weight `0.1`. This proves the
training/runtime path is wired, not that it is ready as a default. The PyTorch
MLP smoke `runs/bench_exact/action_prior_mlp_airplane2_mcts2.json` kept reported
metric diffs at `0` but measured `0.952x` on a tiny CPU run, so it is also a
functional research path rather than a speed win yet. `mcts.puct_prior_weight`
adds an opt-in PUCT-style prior bonus during child selection; the first tiny
linear-prior PUCT smoke
`runs/bench_exact/action_prior_puct_linear_airplane2_mcts2.json` kept reported
metric diffs at `0` and measured `1.055x`, but the sample is too small for a
recommendation. The current offline-RL MLP prior trained from all available
traces is packaged at
`smart/assets/priors/category_general_all_available_offline_rl_mlp_prior.json`.
On `runs/bench_exact/rl_prior_cat5_mcts20_weight_sweep.json`, prior weight
`0.1` gave `1.057x` mean stage speedup over baseline, with `14/15`
quality-not-worse cases and `1/15` quality-improved case; weight `0.2` reached
`1.081x` but had `3/15` worse cases. This makes the RL path active and useful
for research, but not a default reproduction setting yet. The next useful step
is a quality-guarded prior run or a stronger category-aware policy, still judged
with final SMART evaluation metrics.
The first quality-guarded 15-mesh run
`runs/bench_exact/quality_guard_cat5_mcts20_rl01.json` used the same global
offline-RL prior at weight `0.1`. The raw prior was not-worse on `14/15` cases
and worse on `1/15`, but the guard selected baseline for the worse case, so all
`15/15` `mcts_guarded` outputs succeeded. The guard selected prior on `10/15`
cases, baseline on `5/15`, and measured `1.053x` mean raw-prior stage speedup.
A quick chair-specific offline-RL prior did not improve stability
(`2/5` worse prior cases on the same chair subset), so the current recommended
research setting is still the global prior plus quality guard.
An action-level policy-gradient agent is also available through
`smart build-prior --model-type pg-agent`. Unlike the older coord/scale RL
prior, it scores concrete SMART action ids and sees bbox index features, so it
can change which box MCTS explores first. It remains a research prior, not a
paper default. The first all-trace model is packaged at
`smart/assets/priors/category_general_policy_gradient_agent_prior.json`.
On `runs/bench_exact/quality_guard_cat3_mcts20_pg_agent_w005.json`
(`3` meshes per category, `mcts_iter=20`, `max_step=20`, weight `0.05`), the
raw prior was not-worse on `7/9` cases and worse on `2/9`, while the quality
guard selected prior on `4/9` and baseline on `5/9`. Mean raw-prior stage
speedup was `1.006x`. This is a useful RL hook, but the current stronger
research baseline is still the global offline-RL coord/scale prior plus guard
until the action-level agent improves quality more consistently.

For a quality-guarded learned-prior run, use:

```bash
python3 scripts/run_quality_guarded_mcts.py \
  --config configs/rl_search_experimental.yaml \
  --category airplane \
  --mesh 1026dd1b26120799107f68a9cb8e3c \
  --prior-path smart/assets/priors/category_general_all_available_offline_rl_mlp_prior.json \
  --prior-weight 0.1 \
  --mcts-iter 20 \
  --max-step 20
```

The script runs baseline MCTS and prior-guided MCTS, evaluates both with SMART
metrics, and copies the selected bbox result into the `mcts_guarded` stage. If
the prior result is worse on any guarded metric, the baseline bbox is selected.
`smart evaluate --stage mcts_guarded ...` and `render.input_stage:
mcts_guarded` can then consume the guarded result.

To train the action-level policy-gradient agent from trace files:

```bash
python3 scripts/train_action_prior_from_traces.py \
  $(cat runs/bench_exact/priors/all_available_trace_files_20260514.txt) \
  --output runs/bench_exact/priors/category_general_policy_gradient_agent_prior.json \
  --model-type pg-agent \
  --min-reward -1000000000000 \
  --epochs 80 \
  --learning-rate 0.005 \
  --hidden-size 48 \
  --device auto \
  --advantage-baseline category \
  --advantage-clip 5 \
  --entropy-coef 0.005 \
  --max-logit-abs 6
```

For better RL data than accepted actions alone, collect candidate traces during
MCTS. This is opt-in and does not change search behavior; it records per-bbox
rollout candidates that SMART already exact-scored:

```bash
smart --config configs/expanded_full.yaml \
  --set mcts.mcts_iter=20 \
  --set mcts.max_step=20 \
  --set mcts.candidate_trace_path=runs/bench_exact/candidates/example.jsonl \
  --set mcts.candidate_trace_top_k=4 \
  mcts --category airplane --mesh 1026dd1b26120799107f68a9cb8e3c --force
```

`--model-type pg-agent` uses these `record_type=mcts_candidate` rows as
within-rollout comparisons: candidates above the rollout candidate mean get
positive advantage, lower candidates get negative advantage. A smoke probe
`runs/bench_exact/candidate_trace_probe.jsonl` produced `28` candidate rows from
a 3-iteration MCTS run, and the trainer consumed all `28` as candidate records.
The first category-balanced candidate-trace run
`runs/bench_exact/candidate_pg_cat3_mcts10_collection.json` collected `541`
candidate rows from `9` airplane/chair/table meshes. The retrained
candidate-aware PG-agent
`runs/bench_exact/priors/category_general_candidate_pg_agent_cat3_prior.json`
used those candidate rows plus the existing trace list. The same research prior
is packaged at
`smart/assets/priors/category_general_candidate_pg_agent_cat3_prior.json`. On
`runs/bench_exact/candidate_pg_cat3_mcts10_benchmark.json` with `3` meshes per
category, `mcts_iter=10`, `max_step=10`, and prior weight `0.05`, the prior was
quality-not-worse on `9/9` cases and quality-improved on `1/9`. Mean stage
speedup was `1.038x` overall (`1.148x` airplane, `0.956x` chair, `1.009x`
table). A follow-up sweep
`runs/bench_exact/candidate_pg_cat3_mcts10_weight_sweep.json` found weight
`0.2` better on this subset: quality-not-worse `9/9`, quality-improved `2/9`,
no worse cases, and `1.067x` mean stage speedup. This is a real RL/prior
signal, but it still needs larger guarded validation before becoming a
recommended setting.
On a larger `5` meshes per category check
`runs/bench_exact/candidate_pg_cat5_mcts10_w02_benchmark.json`, the same
weight-`0.2` prior was quality-not-worse on `14/15`, improved `2/15`, and worse
on `1/15`, with `1.035x` mean stage speedup. The worse case was table
`1040cd764facf6981190e285a2cbc9c`; the guard run
`runs/bench_exact/candidate_pg_guard_w02_table_badcase.json` rejected that prior
output and selected the baseline bbox into `mcts_guarded`. This means the RL
prior is useful only with guard selection at this stage.
The next guarded check
`runs/bench_exact/candidate_pg_guard_cat10_mcts10_w02.json` ran `10` meshes per
category with weight `0.2`: all `30/30` guarded outputs succeeded, raw prior
was not-worse on `25/30`, improved `2/30`, and worse on `5/30`; the guard
rejected those `5` worse prior outputs. Final selection was prior `15/30` and
baseline `15/30`, with `1.074x` mean raw-prior stage speedup.
Scaling candidate collection to `10` meshes per category produced
`runs/bench_exact/candidate_pg_cat10_mcts10_collection.json` with `2206`
candidate rows, but the retrained prior
`runs/bench_exact/priors/category_general_candidate_pg_agent_cat10_prior.json`
was not better: `runs/bench_exact/candidate_pg_cat10_prior_guard_cat10_mcts10_w02.json`
kept guarded success at `30/30`, but selected prior on only `10/30`, improved
`0/30`, and measured `1.016x`. This prior is intentionally not packaged; the
packaged research prior remains the cat3 candidate-aware model above.
The trainer also exposes `--accepted-weight`, `--candidate-weight`,
`--selected-candidate-weight`, and `--category-balance` for PG-agent loss
experiments. A weighted cat10 retrain
`runs/bench_exact/priors/category_general_candidate_pg_agent_cat10_weighted_prior.json`
improved selected-prior frequency on the 5/category smoke but still produced
`0/15` improvements and `3/15` raw worse cases, so it is not promoted.
For quality-first research, `scripts/run_quality_guarded_mcts.py` now accepts
`--prior-weights 0.05,0.1,0.2` and treats each weight as a separate exact
candidate. The first multi-weight guard
`runs/bench_exact/candidate_pg_multiweight_guard_cat3_mcts10.json` succeeded on
`9/9`, selected a prior candidate on `7/9`, improved `2/9`, and observed no
raw worse candidate on that subset. This costs more runtime because it runs
multiple MCTS searches per mesh.
The larger cat5 check
`runs/bench_exact/candidate_pg_multiweight_guard_cat5_mcts10.json` succeeded on
`15/15`, selected a prior candidate on `13/15`, selected baseline on `2/15`,
and improved `2/15`. One table mesh had all three prior weights worse than
baseline, so the guard selected baseline. This confirms multi-weight guard is
robust as a quality-first research mode, but it is expensive because each mesh
runs baseline plus three prior-guided MCTS searches.
The matching research profile is
`configs/rl_multiweight_guard_experimental.yaml`; run it through the guard
script because the main paper pipeline intentionally keeps one MCTS trajectory:

```bash
python3 scripts/run_quality_guarded_mcts.py \
  --config configs/rl_multiweight_guard_experimental.yaml \
  --categories airplane,chair,table \
  --per-category-limit 5 \
  --prior-path smart/assets/priors/category_general_candidate_pg_agent_cat3_prior.json \
  --prior-weights 0.05,0.1,0.2 \
  --mcts-iter 10 \
  --max-step 10
```

To reduce wall-clock cost, add `--adaptive-prior-weights`. It still evaluates
with exact SMART metrics and only skips later weights after an earlier learned
candidate is already quality-improved versus baseline. For faster exploratory
sweeps, add `--adaptive-stop-mode not_worse`; that can stop after a faster
non-worse candidate, but it may miss a later weight with a larger improvement.
The first fast adaptive cat3 check
`runs/bench_exact/candidate_pg_multiweight_adaptive_fast_cat3_mcts10.json`
kept `9/9` guarded successes and skipped `12/27` candidate MCTS runs.
The follow-up refined cat5 check
`runs/bench_exact/candidate_pg_multiweight_adaptive_fast_cat5_mcts10.json`
kept `14/14` guarded successes and reduced total MCTS launches from `56` to
`32`.
The full currently refined subset check
`runs/bench_exact/candidate_pg_multiweight_adaptive_fast_refined21_mcts10.json`
kept `21/21` guarded successes and reduced total MCTS launches from `84` to
`61`.

The first hybrid MCTS + local-search probe is also available through the new
`local_refine` stage and `configs/hybrid_local_search_experimental.yaml`. On
the checked 10-box airplane case, post-MCTS local search improved BVS
(`2.722 -> 2.609`), MOV (`3.034 -> 2.497`), TOV (`1.580 -> 1.508`), and vIoU
(`0.3867 -> 0.3978`) while keeping coverage above `0.998`.
`scripts/run_quality_guarded_local_refine.py` now makes this hybrid path
guarded: it evaluates `mcts_guarded`, runs local refine, evaluates the refined
result, and copies the selected output into `local_refine_guarded`. On the full
currently refined subset
`runs/bench_exact/local_refine_guarded_refined21_covtol_improved_reuse_mcts_guarded.json`
(`10` airplane, `7` chair, `4` table), it produced `21/21` guarded successes,
selected local refine on `10/21`, selected the input on `11/21`, and improved
`10/21` cases. The selected `local_refine_guarded` stage improved aggregate
BVS (`2.0373 -> 2.0010`), MOV (`1.5140 -> 1.3500`), TOV (`0.9780 -> 0.9480`),
and vIoU (`0.6142 -> 0.6269`) versus `mcts_guarded`; coverage changed only
slightly (`0.999527 -> 0.999520`) under the explicit `0.001` coverage
tolerance used for this quality-first run. A strict coverage guard
(`--covered-tolerance 0`) selected local refine on only `2/21`, so the current
quality-first setting intentionally permits tiny coverage movement while still
rejecting clear regressions.
Using `--from-input-manifest` with `input-stage=mcts` expanded this hybrid check
to all `52` successful processed MCTS outputs in
`runs/bench_exact/local_refine_guarded_expanded200_mcts_manifest52_covtol_improved.json`.
It selected local refine on `29/52`, selected input on `23/52`, and improved
`29/52` cases. The selected `local_refine_guarded` stage improved aggregate
BVS (`1.7546 -> 1.7225`), MOV (`1.2327 -> 1.1410`), TOV
(`0.7173 -> 0.6913`), and vIoU (`0.6835 -> 0.6970`) versus baseline MCTS,
while coverage was effectively unchanged (`0.999685 -> 0.999681`).
For the next learned gate/RL step, export the guard report as supervised rows:

```
python3 scripts/export_local_refine_gate_dataset.py \
  runs/bench_exact/local_refine_guarded_expanded200_mcts_manifest52_covtol_improved.json \
  --output runs/bench_exact/local_refine_gate_manifest52.csv
```

The current export contains `52` rows with `29` positive local-refine
improvement labels.
Train the first PyTorch gate on that dataset with:

```bash
PYTHONPATH=. python3 scripts/train_local_refine_gate.py \
  runs/bench_exact/local_refine_gate_manifest52.csv \
  --output runs/bench_exact/local_refine_gate_manifest52_model.json \
  --hidden-size 8 \
  --epochs 120 \
  --device auto
```

The current packaged research gate is:

```text
smart/assets/gates/local_refine_gate_manifest52.json
```

It uses only category and pre-local-refine SMART metrics, not local-refine
outputs or deltas. On the current `52`-row manifest dataset, leave-one-out
validation gives accuracy `0.75`, F1 `0.780`, and ROC-AUC `0.784` versus the
majority baseline accuracy `0.558`. This is not a final policy, but it is a
real learned signal for deciding when the extra local search is worth running.
The guarded local-refine runner can use the gate to skip local search below a
probability threshold:

```bash
python3 scripts/run_quality_guarded_local_refine.py \
  --config configs/expanded_200.yaml \
  --input-stage mcts \
  --from-input-manifest \
  --gate-path smart/assets/gates/local_refine_gate_manifest52.json \
  --gate-threshold 0.5 \
  --output runs/bench_exact/local_refine_gate_run.json
```

A forced skip smoke with `--gate-threshold 1.0` scored one airplane case,
skipped local refine, selected the input bbox, and wrote a successful guarded
output.
For fast threshold analysis without rerunning the expensive local-refine stage:

```bash
PYTHONPATH=. python3 scripts/evaluate_local_refine_gate.py \
  runs/bench_exact/local_refine_gate_manifest52.csv \
  --gate-path smart/assets/gates/local_refine_gate_manifest52.json \
  --output runs/bench_exact/local_refine_gate_manifest52_threshold_sweep.json
```

On the current manifest52 sweep, threshold `0.5` runs local refine on `30/52`
cases, skips `22/52`, catches all `29/29` known improvement cases, and matches
the full guarded local-refine aggregate metrics while saving `20.3%` of the
local-refine stage time in the measured rows. This makes the gate useful as a
cost control layer, not as a direct quality-improvement model.
The actual gated stage can then be evaluated from its manifest:

```bash
python3 -m smart --config configs/expanded_200.yaml evaluate \
  --stage local_refine_gate_guarded \
  --from-manifest \
  --chamfer-points 0 \
  --output runs/bench_exact/local_refine_gate_guarded_manifest52_t05_stage_eval.json \
  --json
```

The current gated stage evaluation has `52/52` successes with aggregate BVS
`1.7225`, MOV `1.1410`, TOV `0.6913`, Covered `0.999681`, and vIoU `0.6970`.

For quality-first learned MCTS, use `--selection-objective quality_score` in the
guarded MCTS runner. This ignores faster-but-identical prior candidates and
selects a learned-search output only when exact SMART metrics give a positive
scalar quality gain while still passing the non-worse per-metric guard:

```bash
PYTHONPATH=. python3 scripts/run_quality_guarded_mcts.py \
  --config configs/expanded_200.yaml \
  --categories airplane,chair,table \
  --per-category-limit 5 \
  --prior-path smart/assets/priors/category_general_candidate_pg_agent_cat3_prior.json \
  --prior-weights 0.05,0.1,0.2 \
  --selection-objective quality_score \
  --mcts-iter 10 \
  --max-step 10 \
  --stage mcts_quality_guarded_cat5 \
  --output runs/bench_exact/candidate_pg_quality_score_guard_cat5_mcts10.json
```

On the current 14-mesh processed subset, this selected a learned prior on
`1/14` cases and kept baseline on `13/14`. The selected aggregate improved over
baseline by BVS `-0.00093`, MOV `-0.01306`, TOV `-0.00044`, and vIoU
`+0.00038`, with no coverage drift. The result is small, but it is now measuring
the right target: final quality improvement, not trajectory identity or speed.

The current action policy/value research path is also wired. Train it with
PyTorch:

```bash
python3 scripts/train_action_prior_from_traces.py \
  runs/bench_exact/traces/airplane_mcts20_trace.jsonl \
  --output runs/bench_exact/priors/policy_value_prior.json \
  --model-type policy-value \
  --device auto
```

Use the model as a guarded MCTS candidate by setting both policy and value
weights:

```bash
PYTHONPATH=. python3 scripts/run_quality_guarded_mcts.py \
  --config configs/expanded_200.yaml \
  --categories airplane,chair,table \
  --per-category-limit 3 \
  --prior-path smart/assets/priors/category_general_policy_value_agent_prior.json \
  --prior-weights 0.05,0.1,0.2 \
  --puct-prior-weight 0.02 \
  --action-value-weight 0.02 \
  --selection-objective quality_score
```

The latest 9-mesh smoke selected a learned policy/value candidate on `1/9`
cases, kept baseline on `8/9`, and improved aggregate BVS/MOV/TOV/vIoU with no
coverage drift. This is still research-only; exact SMART evaluation remains the
acceptance layer.
To use the agent for speed rather than only action ordering, enable policy
top-K pruning. This keeps only the top-K policy/value actions per MCTS node and
passes those actions through the normal exact reward path:

```bash
PYTHONPATH=. python3 scripts/run_quality_guarded_mcts.py \
  --config configs/expanded_200.yaml \
  --categories airplane,chair,table \
  --per-category-limit 3 \
  --prior-path smart/assets/priors/category_general_policy_value_agent_prior.json \
  --prior-weight 0.1 \
  --puct-prior-weight 0.1 \
  --action-value-weight 0.05 \
  --action-prior-top-k 1 \
  --mcts-iter 10 \
  --max-step 20 \
  --selection-objective legacy
```

The first 3/category top-K smoke,
`runs/bench_exact/mcts_policy_topk1_cat3.json`, kept `9/9` prior candidates
not-worse, rejected `0/9`, and selected the faster prior result on `7/9` meshes.
Mean prior speedup was `1.44x`. In the 1/category smoke, speedup was `1.91x`.
This is the first useful learned-agent speed path. It is still opt-in because
top-K pruning changes the search tree and currently improves time more than
final quality.
For full pipeline experiments, the same settings are captured in
`configs/rl_policy_topk_experimental.yaml`.
For final-return training data, add `--final-return-trace-output` to the same
runner. This stores accepted action traces annotated with the final exact SMART
quality gain:

```bash
PYTHONPATH=. python3 scripts/run_quality_guarded_mcts.py \
  --config configs/expanded_200.yaml \
  --category table \
  --mesh 1040cd764facf6981190e285a2cbc9c \
  --prior-path smart/assets/priors/category_general_policy_value_agent_prior.json \
  --prior-weights 0.05,0.1,0.2 \
  --puct-prior-weight 0.02 \
  --action-value-weight 0.02 \
  --selection-objective quality_score \
  --final-return-trace-output runs/bench_exact/policy_value_final_return.jsonl
```

Rows with `record_type=mcts_final_return` keep the immediate action reward as
`action_reward` and set `reward` to the final quality label. Guard-failing
candidates are forced negative even if one scalar metric improves.
When you want to keep the existing policy logits and train only the action-value
head from final-return labels, pass `--policy-base-prior`:

```bash
python3 scripts/train_action_prior_from_traces.py \
  runs/bench_exact/policy_value_final_return_train_cat5.jsonl \
  --output runs/bench_exact/priors/category_general_policy_value_base_final_return_cat5_prior.json \
  --model-type policy-value \
  --policy-base-prior smart/assets/priors/category_general_policy_value_agent_prior.json \
  --epochs 0 \
  --value-epochs 200 \
  --device auto
```

This value-only fine-tune keeps the packaged action policy unchanged and only
updates the scalar action-value head. On the known table improvement case it
selected the learned candidate and improved exact SMART quality, but on the
current held-out offset probe it was safe rather than better: `5/5` guarded
successes, `0/5` learned selections, no worse candidates, and about `1.11x`
mean raw-prior speedup. It is therefore a research checkpoint, not the packaged
default.
A larger cat10 final-return collection reached `21/21` guarded successes and
one learned-candidate quality improvement, but the label distribution is still
sparse (`59` positive rows out of `2228`, all from table), so the next useful
RL step is more positive final-return coverage rather than promoting the
value-only checkpoint.
That sparsity was not just a data-count issue. After adding six more table
meshes through refine, `runs/bench_exact/candidate_pg_final_return_table10.jsonl`
still produced positive MCTS final-return rows only for the same table mesh, and
`runs/bench_exact/candidate_pg_final_return_table_new6_mcts20.jsonl` produced
`0` positives at MCTS20/max-step20. The current MCTS prior is therefore safe but
not yet a reliable quality-improvement generator.
The value trainer also exposes `--value-positive-weight`,
`--value-negative-weight`, and `--value-zero-weight` for sparse final-return
labels. A positive-heavy cat5 value-only run stayed safe on the held-out offset
probe (`5/5` guarded successes, no worse candidates) and improved raw-prior
speed to about `1.18x`, but it still selected baseline on every held-out case.
The latest gated post-MCTS local-refine probe on four held-out policy-value
outputs selected one improved local-refine result and kept three inputs. With
`smart/assets/gates/local_refine_gate_manifest52.json` at threshold `0.5`, it
skipped `2/4` local-refine runs while still catching the one improvement.
On the larger cat10 policy/value guarded outputs, local refine with
`Covered` tolerance `0.001` selected `10/21` improved results. The same gate
skipped the other `11/21` local-refine launches and still selected the same
10 improvements, giving the strongest current quality/time tradeoff after
guarded learned MCTS.
Local-refine can now export final-return action traces too:

```bash
python3 scripts/run_quality_guarded_local_refine.py \
  --config configs/expanded_200.yaml \
  --from-input-manifest \
  --input-stage mcts \
  --stage local_refine_trace_mcts_cat10_covtol \
  --categories airplane,chair,table \
  --per-category-limit 10 \
  --max-step 50 \
  --action-unit 0.005 \
  --selection-mode improved \
  --covered-tolerance 0.001 \
  --quality-weights Avg_BVS=1,Avg_MOV=0.25,Avg_TOV=0.25,Avg_Covered=2,Avg_vIoU=1 \
  --final-return-trace-output runs/bench_exact/local_refine_trace_mcts_cat10_covtol.jsonl
```

This produced `30/30` successes and selected local refine on `19/30` cases.
The resulting trace has `218` rows with `193` positive final-return labels
across airplane/chair/table. Combined with the guarded learned-MCTS local-refine
trace, the local-search final-return set is now `394` rows with `341` positives
from `27` meshes. A PyTorch policy-value checkpoint trained from that set lives
at `smart/assets/priors/local_refine_policy_value_final_return_cat10.json`; it
is a research artifact for post-MCTS action/value proposal, not a default.
Use it only with explicit search-order opt-in and exact guarding:

```bash
python3 scripts/run_quality_guarded_local_refine.py \
  --config configs/expanded_200.yaml \
  --from-input-manifest \
  --input-stage mcts \
  --categories airplane,chair,table \
  --per-category-limit 10 \
  --action-prior-path smart/assets/priors/local_refine_policy_value_final_return_cat10.json \
  --action-value-weight 0.05 \
  --action-prior-top-k 1 \
  --selection-mode improved \
  --covered-tolerance 0.001
```

The current cat10 probe selected the same `19/30` improvements as exact
local-refine and had `30/30` identical guarded selections. Mean local-refine time
was `2.90s` versus `2.94s` for exact local-refine on the same 30 cases, so this
is a working proposal hook but not yet a meaningful default speed or quality
upgrade.

For quality-oriented research runs, the guarded local-refine runner can now
launch both an unbiased exact local-refine candidate and a learned policy/value
candidate, exact-evaluate both, and select the best non-worse output by scalar
quality score:

```bash
python3 scripts/run_quality_guarded_local_refine.py \
  --config configs/expanded_200.yaml \
  --from-input-manifest \
  --input-stage mcts \
  --stage local_refine_multi_guard_cat3_v005_top1 \
  --categories airplane,chair,table \
  --per-category-limit 3 \
  --max-step 50 \
  --action-unit 0.005 \
  --include-exact-local-refine \
  --action-prior-path smart/assets/priors/local_refine_policy_value_final_return_cat10.json \
  --action-value-weight 0.05 \
  --action-prior-top-k 1 \
  --selection-mode improved \
  --selection-objective quality_score \
  --covered-tolerance 0.001 \
  --quality-weights Avg_BVS=1,Avg_MOV=0.25,Avg_TOV=0.25,Avg_Covered=2,Avg_vIoU=1
```

The 10/category probe
`runs/bench_exact/local_refine_multi_guard_cat10_v005_top1.json` produced
`30/30` successes, ran `60` local-refine candidates, selected input on `11/30`,
exact local-refine on `18/30`, and the learned candidate on `1/30`. The selected
non-input outputs improved mean BVS/MOV/TOV/vIoU by
`-0.069/-0.237/-0.061/+0.031` with effectively unchanged `Covered`
(`+0.000002`). This is a quality research mode, not a default, because it
intentionally runs two local-refine candidates per mesh.
A Rust `TetClippingState` backend is also available behind
`reward_backend=tet_clipping`, but it is experimental and not the default:
smoke parity is close (`<=2e-5` in checked records), while tiny cases can be
slower than Manifold.
The current pipeline also skips log-only summary metric recomputation and
internal partition snapshots by default, which removes redundant work without
changing the selected actions, bbox outputs, or evaluation metrics.

Build or refresh the local Rust extension during source development:

```bash
smart build-rust
```

Release wheels are built with maturin from the repo root, which bundles the
extension directly into `smart-bbox` as `smart._rust`. Without that extension,
`smart.rust` uses the matching Python fallback kernels, so the pipeline remains
runnable for debugging.

# Reference

Our bounding box rendering code is based on the [StructureNet](https://github.com/daerduoCarey/structurenet/tree/master)'s [renderer](https://github.com/daerduoCarey/structurenet/tree/master/viz_blender). We thank the authors for opening the rendering code.

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{Park:2024SMART,
 title={Split, Merge, and Refine: Fitting Tight Bounding Boxes via Over-Segmentation and Iterative Search},
 author={Park, Chanhyeok and Sung, Minhyuk},
 booktitle= {3DV},
 year={2024}
}
```

## License

This work is licensed under a [CC BY-NC-SA 4.0][cc-by-nc-sa].

![CC BY-NC-SA 4.0][cc-by-nc-sa-image]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
