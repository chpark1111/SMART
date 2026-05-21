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

Generated example meshes and outputs are local only:

```text
examples/sample_shapes/
examples/runs/
```

