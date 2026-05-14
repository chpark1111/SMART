# SMART Python Package

This page documents the package interface for SMART as a Python library and
command line tool.

The package name is `smart-bbox`. The import name and console command are both
`smart`.

For the current applied-vs-experimental status, see
[`CURRENT_STATUS.md`](CURRENT_STATUS.md).

## Install

For local development from this repository:

```bash
cd SMART
maturin develop --release --extras pipeline
```

If you prefer pip editable installs, install the Python `maturin` build backend
first and then run `python -m pip install -e ".[pipeline]"`.

After the package is published to PyPI:

```bash
python -m pip install "smart-bbox[pipeline]"
```

The package is built as one mixed Python/Rust wheel. The `smart-bbox` wheel
contains the Python package and the exact PyO3 acceleration module at
`smart._rust`. The `pipeline` extra installs the Python dependencies needed by
the geometry pipeline, including `numpy`, `trimesh`, `torch`, `coacd`, and
rendering/eval helpers.

Binary wheels do not include the vendored Manifold source tree; they include
the compiled `smart._rust` extension that links the fixed bridge. Source
distributions keep the vendored Manifold source for reproducibility, but pip
sdist builds do not build the Manifold static library automatically. For normal
users, publish platform wheels. For source checkouts, run `smart build-tools`
and `smart build-rust` after the external tools are available.

## External Tools

SMART also needs external geometry tools for the full paper pipeline:

```bash
smart --config configs/demo.yaml build-tools
```

This builds or prepares:

- ManifoldPlus and fTetWild for Mesh2Tet.
- The fixed vendored Manifold Python binding under `smart/vendor/manifold`.

Do not pull, replace, or rewrite the vendored Manifold source. It is the fixed
C++ binding used by the SMART legacy geometry path.

You can also point SMART at prebuilt tools:

```bash
export SMART_MANIFOLDPLUS_BIN=/path/to/ManifoldPlus/build/manifold
export SMART_FTETWILD_BIN=/path/to/fTetWild/build/FloatTetwild_bin
export SMART_MANIFOLD_PYTHON=/path/to/smart/vendor/manifold/build/bindings/python
export SMART_BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/blender
```

Check the runtime before long runs:

```bash
smart --config configs/demo.yaml doctor
smart --config configs/demo.yaml doctor --json
```

## Bundled Rust Acceleration

Rust acceleration is bundled into release wheels as `smart._rust` and must
preserve legacy output compatibility. Current Rust work ports exact kernels
only; approximate search changes, RL priors, transposition tables, and
heuristic prefilters are not part of the default library path.

For source development, rebuild the local extension:

```bash
smart --config configs/demo.yaml build-rust
```

This command builds the root `smart-bbox` wheel with maturin and reinstalls it
into the current Python environment with `--no-deps`. It is meant for source
checkouts. Normal PyPI wheel installs already contain `smart._rust`.

If the extension is unavailable or disabled with `SMART_DISABLE_RUST=1`,
`smart.rust` falls back to matching Python kernels. Current exact kernels cover
bbox/action scoring helpers, evaluation Chamfer distance, Gmsh `.msh`
loading/writing, and tetra volume/centroid/surface/adjacency summaries used by
the `pymesh` shim. The default inference path also uses exact score caching for
accepted greedy actions and exact BVS upper bounds for volume-preserving
rotation/recentering actions before invoking expensive Manifold coverage.

The experimental `reward_backend=tet_clipping` path uses Rust
`TetClippingState` to keep tetrahedra in Rust and evaluate exact
tetrahedron-box clipping metrics from bbox bounds/rotations. It is available
for research sweeps, but release/default configs should keep
`reward_backend=manifold` until parity and timing pass on a larger benchmark.
The exact `reward_backend=manifold_bridge` path keeps using the fixed C++
Manifold implementation through Rust/PyO3 and is the safer acceleration target
before any tet-clipping replacement. It stores the source mesh in Rust/C++ and
evaluates axis-action greedy rewards without per-candidate Python env
round-trips. Greedy refine can keep consecutive accepted axis-action segments
inside Rust and sync only touched bbox manifolds back to Python. MCTS rollout
can batch bbox-mask axis-action probes through Rust/C++, reuse unchanged bbox
Manifold objects in the wrapper for axis candidates, and apply selected scored
actions without rerunning the full legacy `step()`. Recenter candidate geometry
still uses the exact legacy `trimesh.bounds.oriented_bounds(angle_digits=3)`
path for parity. It is opt-in for refine/MCTS with
`refine.reward_backend=manifold_bridge` or
`mcts.reward_backend=manifold_bridge`.

The exact `reward_backend=manifold_stateful` path is the next migration layer.
It keeps the mutable bbox state, current exact score, rollback history, and
candidate reward cache inside `smart._rust.ManifoldState`. Use it with
`greedy.backend=rust_stateful` for refine parity/timing sweeps. The Rust MCTS
runner is guarded behind `mcts.allow_search_order_changes=true` because it can
change the trajectory even when the reward backend is exact; leave
`mcts.backend=auto` for paper-compatible MCTS parity sweeps. It still calls the
fixed vendored Manifold C++ library; the Manifold source itself is not rewritten
or replaced. Accepted stateful axis
actions sync only the changed bbox parameters back to Python, and deterministic
initial bbox parameters are cached across MCTS resets so rollouts do not
re-read the same bbox OBJ files.

Current checked acceleration is conservative: table refine has shown `1.27x`
with exact metric parity. For MCTS with the legacy tree
(`mcts.backend=auto`), the latest five-mesh smoke
`runs/bench_exact/exact_stateful_sweep_smoke5_mcts5_legacy_union_fresh_stats.json`
keeps all reported metric diffs at `0` after matching the legacy sequential
union order with `stateful_union_cache=false`. Reward memoization is now
independent from the union cache, so exact repeated state/action scores are
reused without changing boolean grouping order. The checked repeat-3 smoke
`runs/bench_exact/exact_stateful_sweep_smoke5_mcts5_rewardcache_repeat3.json`
keeps metric diffs at `0` and measures `1.032x`; the larger
`runs/bench_exact/exact_stateful_sweep_smoke3_mcts20_rewardcache.json` check
keeps metric diffs at `0` and measures `1.189x`. The stateful C++ bridge now
constructs bbox manifolds from the same explicit eight-vertex/fixed-face mesh
used by the legacy Python backend, rather than `Cube().Transform()`. That
removed the remaining near-tie drift in the five-target MCTS20 smoke:
`runs/bench_exact/exact_stateful_sweep_smoke5_mcts20_vertexbox_trace.json`
keeps metric diffs at `0`, has no action trace divergence, and measures
`1.024x`. The matching repeat-3 timing run
`runs/bench_exact/exact_stateful_sweep_smoke5_mcts20_vertexbox_repeat3.json`
keeps metric diffs at `0` and measures `1.039x`.
A separate airplane
100-iteration run with `mcts.backend=rust_stateful` measured `1.33x`, but that
path is a search-order-changing experiment and is not a paper-safe exact
default. Exact Manifold remains the official default until action trajectory and
metric parity pass across airplane/chair/table sweeps.

The exact-verification bitset candidate profile is packaged as
`configs/candidate_bitset_exact_experimental.yaml`. It uses
`reward_backend=manifold_stateful`, `candidate_backend=bitset_topk`,
`candidate_top_k=8`, `stateful_union_cache=false`, and keeps
`mcts.backend=auto` so the legacy MCTS tree remains in control. It ranks
actions with Rust centroid/volume bitsets but still exact-verifies candidates
with Manifold. The current MCTS20 smoke sweep
`runs/bench_exact/mcts_accel_profiles_smoke5_iter20_exact_candidates.json`
keeps all metric diffs at `0` and measures `1.015x` for `bitset_top8`; the
faster TT/prior profiles changed metrics and remain search experiments.
The faster top-3 variant is packaged as
`configs/candidate_bitset_fast_experimental.yaml`. On the nine processed
airplane/chair/table sweep
`runs/bench_exact/mcts_accel_profiles_expanded_processed9_iter20_topk_tt_all.json`,
`bitset_top3` kept all reported metric diffs at `0` and measured `1.025x`;
`bitset_top8` measured `1.023x`. Both remain opt-in until larger processed
sweeps pass.
`configs/stateful_union_cache_experimental.yaml` is the current cleaner exact
acceleration candidate. It uses `reward_backend=manifold_stateful` with
`stateful_union_cache=true`, leaves `candidate_backend=exact`, and keeps
`allow_search_order_changes=false`. On
`runs/bench_exact/mcts_accel_profiles_expanded_processed9_iter20_union_cache_probe.json`,
it kept all reported metrics identical and measured `1.097x`.
On the current 16-mesh processed set,
`runs/bench_exact/mcts_accel_profiles_expanded_processed16_iter20_union_cache.json`
keeps all reported metrics identical and measures `1.047x`.
MCTS action-prior profiles are exposed as experiments only; the 16-mesh
`prior_weight_sweep` changed one table result for all tested prior weights, so
the prior path is not promoted.
The latest wrapper also constructs exact `union_except_i` entries lazily. The
checked 100-iteration airplane smoke at `max_step=20` measured `18.48s` and
kept the same evaluation metrics while recording `9682` leave-one-out union
cache hits and `8120` reward-cache hits. This improves structure for a future
release backend, but it is still opt-in because Manifold boolean residuals are
the dominant cost.
Use `scripts/benchmark_exact_stateful_sweep.py --trace-actions` before changing
defaults. The harness records timing, paper metrics, cache stats, and the first
accepted-action divergence versus legacy `reward_backend=manifold`. Package
defaults still prefer legacy exact Manifold for paper reproduction.

MCTS learning/search-order experiments are opt-in only. The Rust MCTS runner can
load `action_prior_path` with `action_prior_weight > 0` to bias PNS exploration
from trace-derived priors. The Python-tree runner can also use
`puct_prior_weight > 0` for PUCT-style child selection, and the Rust path can use
`transposition_table=true` as a state-memory experiment. These options do not
replace the exact Manifold reward, but they can change search trajectories, so
keep them out of release/default configs until quality sweeps pass.

Build an action prior from MCTS traces with the packaged CLI:

```bash
smart --config configs/smoke_5.yaml build-prior \
  runs/bench_exact/traces/airplane_mcts20_trace.jsonl \
  --output runs/bench_exact/priors/airplane_mcts20_prior.json
```

For learned action-prior experiments, add `--model-type linear` or
`--model-type mlp`:

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
```

The MLP trainer uses PyTorch. `--device auto` probes Apple Silicon MPS first,
then CUDA, then CPU; install the `pipeline` extra to get `torch`.

The same function is available from Python:

```python
import smart

prior = smart.build_action_prior_from_traces(
    ["runs/bench_exact/traces/airplane_mcts20_trace.jsonl"],
    output="runs/bench_exact/priors/airplane_mcts20_prior.json",
)

linear_prior = smart.build_linear_action_prior_from_traces(
    ["runs/bench_exact/traces/airplane_mcts20_trace.jsonl"],
    output="runs/bench_exact/priors/airplane_linear_prior.json",
)

mlp_prior = smart.build_mlp_action_prior_from_traces(
    ["runs/bench_exact/traces/airplane_mcts20_trace.jsonl"],
    output="runs/bench_exact/priors/airplane_mlp_prior.json",
    device="auto",
)
```

The source distribution and wheel include the current smoke-validated
experimental prior at `smart/assets/priors/smoke5_coord_scale_prior.json`.
`configs/accelerated_search_experimental.yaml` uses that asset with
`action_prior_weight=0.1`; keep it as an experiment because it changes MCTS
action ordering.

## Command Line Usage

Run the full pipeline:

```bash
smart --config configs/demo.yaml run
smart --config configs/accelerated_exact.yaml run
smart --config configs/stateful_exact_experimental.yaml run
```

Use `configs/accelerated_exact.yaml` for paper-safe exact runs. It keeps the
legacy `reward_backend=manifold` path. Use
`configs/stateful_exact_experimental.yaml` only when profiling the opt-in
`manifold_stateful` Rust/C++ cache with the legacy MCTS tree.

Run one stage:

```bash
smart --config configs/demo.yaml normalize --category airplane
smart --config configs/demo.yaml tetra --category airplane
smart --config configs/demo.yaml preseg --category airplane
smart --config configs/demo.yaml merge --category airplane
smart --config configs/demo.yaml refine --category airplane
smart --config configs/demo.yaml mcts --category airplane
smart --config configs/demo.yaml render --category airplane
```

Run one mesh:

```bash
smart --config configs/demo.yaml mcts \
  --category airplane \
  --mesh 172764bea108bbcceae5a783c313eb36
```

Override config values from the command line:

```bash
smart --config configs/smoke_5.yaml --set mcts.mcts_iter=100 run
smart --config configs/smoke_5.yaml --set refine.max_step=200 refine
smart --config configs/smoke_5.yaml --set mcts.summary_metrics=true mcts
smart --config configs/smoke_5.yaml --set render.joint_mesh=true render
smart --config configs/smoke_5.yaml --set 'render.variants=["boxes_only","with_mesh"]' render
```

Refine/MCTS inference defaults to `summary_metrics=false`,
`render_initial=false`, and `render_partition=false`. That keeps final bbox OBJ
outputs and exact SMART rewards unchanged while avoiding duplicate
stdout-only metric summaries and internal partition snapshots. Run
`smart evaluate` for the official metrics, or set `summary_metrics=true` when
you need the legacy per-stage summary printout.

The pipeline order is:

```text
normalize -> tetra -> preseg -> merge -> refine -> mcts -> render
```

## Python API

Basic use:

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
data = smart.check_data("configs/demo.yaml")
```

Load and modify a config without mutating the original object:

```python
import smart

cfg = smart.load(
    "configs/demo.yaml",
    overrides={
        "workspace": "runs/my_experiment",
        "mcts": {"mcts_iter": 3000},
        "render": {"joint_mesh": False},
    },
)

records = smart.run_pipeline(cfg, stage="mcts", category="chair")
```

Useful public functions:

- `smart.load(config, overrides=None)`: load a YAML/JSON config.
- `smart.run_pipeline(config, stage=None, category=None, meshes=None, dry_run=False, force=False, overrides=None)`: run the configured stages.
- `smart.evaluate(config, stage="mcts", category=None, meshes=None, chamfer_points=2048, output_path=None, overrides=None)`: compute SMART evaluation metrics.
- `smart.doctor(config, overrides=None)`: return runtime/tool checks.
- `smart.check_data(config, overrides=None)`: summarize configured mesh data.
- `smart.workspace(config, *parts, overrides=None)`: resolve paths under the configured run workspace.
- `smart.build_action_prior_from_traces(traces, output=..., min_reward=0.0, smoothing=1.0, reward_power=1.0, include_action_logits=False)`: build an opt-in trace-derived MCTS action-prior JSON.
- `smart.build_linear_action_prior_from_traces(traces, output=..., epochs=200, learning_rate=0.05)`: train a state-aware linear action-ordering prior from schema-v2 traces.
- `smart.build_mlp_action_prior_from_traces(traces, output=..., epochs=200, hidden_size=16, device="auto")`: train a PyTorch MLP action-ordering prior and save JSON weights.
- `smart.load_action_prior(path)`: load count, linear, or MLP prior JSON for inspection or direct scoring.

Each call returns plain Python dictionaries or paths, so the package can be used
inside scripts, notebooks, batch runners, or CI.

## Outputs

The default workspace is `runs/<profile>/`.

Important outputs:

- `normalized/`: normalized OBJ meshes.
- `tetra/`: `tetra.msh` and `tetra.msh__sf.obj`.
- `preseg/`: CoACD part OBJs.
- `merge/`: SMART merged initial boxes.
- `refine/`: greedy refinement bbox OBJ outputs.
- `mcts/`: final MCTS bbox OBJ outputs.
- `render/`: PNG renders.
- `manifests/`: per-stage JSONL records and summary.
- `evaluation/`: metric JSON files.

Stage failures are recorded in manifests and skipped downstream instead of
terminating the entire batch.

## Build A Wheel

Build the unified Python/Rust package wheel from the repo root:

```bash
maturin build --release --out dist
```

The built wheel should be platform-specific, for example
`smart_bbox-0.1.0-cp39-cp39-macosx_11_0_arm64.whl`, and should contain
`smart/_rust*.so` or the platform equivalent.
On Apple Silicon, if maturin selects the wrong architecture for a universal
Python, pass the target explicitly:

```bash
maturin build --release --target aarch64-apple-darwin --out dist
```

For local editable development:

```bash
python -m pip install -e ".[pipeline,dev]"
python -m pytest -q
```

The standalone `rust/smart-core` crate remains useful for low-level Rust
development, but root-level `maturin build` is the release path because it
bundles Python code and `smart._rust` into one installable `smart-bbox` wheel.

## Publish To PyPI

The release artifact users should install is a platform wheel containing
`smart/_rust*.so` or the platform equivalent. Do not upload a stale
`py3-none-any` wheel for an official release because it would omit the Rust
extension.

Recommended release flow:

```bash
python -m pytest -q
python -m compileall -q -x 'smart/vendor/.*' smart scripts tests pymesh.py
maturin build --release --target aarch64-apple-darwin --out dist
maturin sdist --out dist
python -m twine check dist/*
```

For a public release, build wheels for the target platforms you want to support
before uploading: macOS arm64, macOS x86_64, Linux x86_64, and Windows x86_64 if
the fixed Manifold bridge builds there. Upload with one of:

```bash
maturin publish --release --skip-existing
# or
python -m twine upload dist/*
```

Use PyPI trusted publishing or an API token from a project owner account. The
package name is `smart-bbox`; the import name and CLI command remain `smart`.
