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
- `run_macro_replay_balanced_chunks.py`: resumable, category-balanced runner
  for learned macro-skill release-readiness evidence.  Use
  `--chunk-timeout-sec <seconds>` for long-tail mesh cases that would otherwise
  stall evidence collection; the default is disabled so exact behavior is
  unchanged unless the timeout is explicitly requested.  Successful chunks and
  failed/timeout chunks are skipped on resume by default; use
  `--no-skip-failed` when a repaired mesh should be retried.
- `smoke_console_scripts.py`: check installed console scripts.
- `smoke_native_sanitizers.py`: run a short native AddressSanitizer smoke.

## Policy

Do not add one-off experiment drivers here. Put them under
`experiments/scripts/` until they are stable enough to be part of the supported
public package or release workflow.
