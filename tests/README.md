# SMART Tests

The tracked test suite covers package, config, native integration, evaluation,
and release-audit behavior.

```bash
python -m pytest tests -q
```

Some native tests require a local `smart._cpp` build:

```bash
smart --config configs/smoke_5.yaml build-cpp
```

Experimental research tests live under `experiments/tests/` and are ignored by
git unless explicitly promoted into the supported test suite.
