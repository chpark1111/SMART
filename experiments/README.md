# SMART Experiments

This directory is for local research-only configs, learned assets, scripts,
traces, benchmark drivers, notes, and experimental tests. It is intentionally
ignored by git and excluded from release wheels/sdists.

The public package path is kept in:

- `smart/`
- `cpp/`
- `configs/`
- `scripts/`
- `tests/`
- `docs/`

Use this directory for temporary or ongoing work such as learned policy priors,
candidate pruning, MCTS trace mining, benchmark sweeps, and guarded quality
experiments. When an experiment becomes a supported release feature, move the
minimal stable code into `smart/`, `cpp/`, or `scripts/` and add normal tests
under `tests/`.

Current local-only subdirectories:

- `configs/`: experimental YAML profiles moved out of public `configs/`.
- `assets/`: learned RL/gate/prior JSON artifacts.
- `scripts/`: benchmark, mining, training, and sweep scripts.
- `tests/`: tests for experimental code paths.
- `docs/`: long research and development notes.
