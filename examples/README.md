# SMART Examples

This directory contains small source-checkout examples for the public C++
SMART pipeline.

ShapeNet meshes are not redistributed in this repository. If local data exists
under `data/shapenet_airplane`, `data/shapenet_chair`, and
`data/shapenet_table`, create a local 3-per-category example set:

```bash
bash examples/prepare_sample_shapes.sh
```

Then run the example profile:

```bash
bash examples/run_example_3x3.sh
```

To run the opt-in guarded learned MCTS profile on the same sample layout:

```bash
bash examples/run_learned_frontier.sh
```

This uses exact native SMART scoring, but prunes/orders MCTS actions with the
packaged DeepSets policy in `smart._cpp`.  The profile uses the guarded preset,
which falls back to exact shallow MCTS for low-confidence multibox states.

Generated example meshes and outputs are local only:

```text
examples/sample_shapes/
examples/runs/
```
