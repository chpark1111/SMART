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

The current SMART code is closest to ShapeNetCore v1 layout and normalization:
`model.obj` under each model directory, with the bounding-box center near the
origin and bounding-box diagonal near `1.0`.

For larger local experiments, `data/expanded/` can contain:

```text
data/expanded/shapenet_airplane/<model_id>/model.obj
data/expanded/shapenet_chair/<model_id>/model.obj
data/expanded/shapenet_table/<model_id>/model.obj
```

The checked `configs/expanded_200.yaml` profile expects 200 meshes per category
in that layout. It is intended for optimization/evaluation sweeps, not for a
quick smoke run.

Run `smart --config configs/demo.yaml check-data` after changing the data
layout. The command reports category counts and a sample bbox diagonal so you
can verify that normalization inputs are in the expected ShapeNet-style scale.
