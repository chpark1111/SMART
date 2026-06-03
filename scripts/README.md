# SMART Scripts

Only supported release, reproduction, and maintenance scripts live here.
Research/benchmark/RL helpers belong in `experiments/scripts/` and are ignored
by git.

## Supported Scripts

- `prepare_shapenet_samples.py`: prepare ShapeNet-style category archives into
  the expected `data/<category>/<model_id>/model.obj` layout.
- `quickstart_reproduce.py`: guided local reproduction helper.
- `analyze_pipeline_failures.py`: inspect pipeline manifests and failure logs.
- `audit_release_wheel.py`: audit wheel/sdist contents before release.
- `release_preflight.py`: build and validate local release artifacts, including installed-wheel smoke and opt-in learned release readiness checks.
- `smoke_console_scripts.py`: check installed console scripts.
- `smoke_native_sanitizers.py`: run a short native AddressSanitizer smoke.

## Policy

Do not add one-off experiment drivers here. Put them under
`experiments/scripts/` until they are stable enough to be part of the supported
public package or release workflow.
