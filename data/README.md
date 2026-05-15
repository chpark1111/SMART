# SMART ShapeNet Samples

SMART's Mesh2Tet stage expects category-specific folders like:

```text
data/shapenet_airplane/<model_id>/model.obj
data/shapenet_chair/<model_id>/model.obj
data/shapenet_table/<model_id>/model.obj
```

Use `scripts/prepare_shapenet_samples.py` to extract a small working subset
from ShapeNetCore v1/v2 folders or category zip archives. The current demo
profiles use 50 meshes per category, while `configs/smoke_5.yaml` pins five
known meshes for quick regression checks. ShapeNetCore access is gated, so
direct download needs an accepted Hugging Face ShapeNetCore account and
`HF_TOKEN`.

SMART uses one local layout regardless of whether the source archive is
ShapeNetCore v1 or v2: `model.obj` under each model directory, with the
bounding-box center near the origin and bounding-box diagonal near `1.0`.

For the SMART paper categories, download these ShapeNet synset archives:

```text
02691156.zip  airplane
03001627.zip  chair
04379243.zip  table
```

Then prepare only those three categories:

```bash
python3 scripts/prepare_shapenet_samples.py \
  --archive-dir /path/to/shapenet_zips \
  --output-root data/expanded \
  --categories airplane chair table \
  --limit 100000 \
  --normalize preserve

smart --config configs/expanded_full.yaml check-data
```

Use `--normalize preserve` for import because SMART's pipeline-level
normalization writes clean normalized copies under `runs/expanded_full/`.

For larger local experiments, `data/expanded/` can contain:

```text
data/expanded/shapenet_airplane/<model_id>/model.obj
data/expanded/shapenet_chair/<model_id>/model.obj
data/expanded/shapenet_table/<model_id>/model.obj
```

`configs/expanded_full.yaml` uses every mesh found in that layout. The smaller
`configs/expanded_200.yaml` profile keeps a 200-per-category cap for
optimization/evaluation sweeps that should not launch over the full local set.

To expand MCTS/RL action-prior traces from a configured dataset without manually
choosing mesh IDs, run:

```bash
python3 scripts/collect_action_traces.py \
  --config configs/expanded_200.yaml \
  --categories airplane,chair,table \
  --batch-size 5 \
  --max-batches-per-category 4
```

Run `smart --config configs/demo.yaml check-data` after changing the data
layout. The command reports category counts and a sample bbox diagonal so you
can verify that normalization inputs are in the expected ShapeNet-style scale.
