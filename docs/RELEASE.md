# SMART Release Checklist

This page is the operational checklist for publishing `smart-bbox`. For package
usage details, see [`PYTHON_PACKAGE.md`](PYTHON_PACKAGE.md).
For the current learned SMART+Agent release boundary and benchmark snapshot,
see [`LEARNED_ROUTER_RELEASE.md`](LEARNED_ROUTER_RELEASE.md).
For the first package release notes, see
[`RELEASE_NOTES_0.1.0.md`](RELEASE_NOTES_0.1.0.md).

## Release Target

- Package name: `smart-bbox`
- Import name: `smart`
- Main CLI: `smart`
- Official wheel contents: Python package, `smart._cpp`, packaged
  `smart/bin/smart-cpp-native`, bundled `smart/pymanifold_runtime/pymanifold*`,
  public configs, renderer assets, packaged learned DeepSets policies,
  packaged macro-skill JSON/JSONL assets,
  release helper scripts, `smart.pymesh_compat`, and the legacy
  `pymesh.py` alias.
- Official package architecture: Python is the CLI/API/package wrapper. Native
  geometry helpers and stateful Manifold reward bridges live in the C++
  extension, and standalone native preprocessing/helper commands live in the
  packaged `smart-cpp-native` executable.
- The fixed vendored Manifold source must not be pulled, replaced, or rewritten.
  Wheels include the compiled runtime, not the source tree.
- Wheel exclusions: `smart/vendor/manifold`, `past_codes`, ShapeNet data,
  `runs`, `external`, and generated build outputs.
- Sdist contents: reproducibility sources, including fixed
  `smart/vendor/manifold` source, native C++ sources, `setup.py`,
  `setup_cpp.py`, `pyproject.toml`, `CITATION.cff`, configs, docs, and tests,
  without compiled binaries or generated outputs.
- Experimental RL/Transformer/gate training runs are local-only under
  `experiments/` and are not part of the public reproduction wheel.  Only
  deterministic release assets under `smart/assets/` are shipped.
- Console smoke: installed wheels should pass `smart-smoke-console-scripts`.

## Local Preflight

Run from the repository root:

```bash
python -m pytest -q
python -m compileall -q -x 'smart/vendor/.*' smart scripts tests pymesh.py
smart --config configs/smoke_5.yaml build-tools --only-manifold-binding
python setup.py build_ext --force bdist_wheel --dist-dir dist
python setup.py sdist --dist-dir dist
smart audit-wheel dist/*
python -m twine check dist/*
```

The same release-candidate check is available as one command:

```bash
smart-release-preflight \
  --dist-dir /private/tmp/smart_release_check \
  --venv-dir /private/tmp/smart_release_venv \
  --recreate-venv
```

For a no-rebuild check against existing artifacts:

```bash
smart-release-preflight \
  --dist-dir /private/tmp/smart_release_check \
  --venv-dir /private/tmp/smart_release_venv \
  --skip-build \
  --recreate-venv
```

The preflight command audits wheel/sdist contents, runs `twine check`, installs
the wheel into a temporary venv, runs installed console-script smoke, and checks
that `smart.native`, `smart.cpp`, the packaged `smart-cpp-native` executable,
and the bundled `pymanifold` runtime are available.  It also runs
`smart learned-release-readiness --fail-if-not-ready --require-default-ready`
from the source checkout and again from the installed wheel, so packaged
learned-router policies, macro-skill assets, learned default-agent configs, and
the default-agent gate cannot silently disappear from a release.
On Apple Silicon, local setuptools builds and `smart-release-preflight` force
`-arch arm64` and default `MACOSX_DEPLOYMENT_TARGET=11.0` so `smart._cpp`,
`smart-cpp-native`, and `pymanifold` match the `macosx_11_0_arm64` wheel tag.
The release audit also inspects real native binaries with `file`, rejects
architecture/tag mismatches, and rejects old macOS arm64 deployment tags.
It also rejects wheels missing the learned default-agent configs,
`deepset_setaware_v2_h128*.smartmlp` policies, or macro-skill runtime assets.

## Native Memory Smoke

Before cutting a release candidate after C++ changes, run a source-checkout
sanitizer smoke:

```bash
smart --config configs/smoke_5.yaml build-cpp --asan
python scripts/smoke_native_sanitizers.py --binary build/smart-cpp-native-asan
```

This builds `build/smart-cpp-native-asan` with AddressSanitizer and runs
`--help`, OBJ loading, and native normalization. It is a fast memory-safety
smoke, not a replacement for long batch soak testing.
The same check can be added to local preflight with:

```bash
smart-release-preflight \
  --dist-dir /private/tmp/smart_release_check \
  --venv-dir /private/tmp/smart_release_venv \
  --run-asan-smoke
```

## Latest Local Verification

The learned SMART+Agent release boundary was refreshed on June 25, 2026:

- packaged default-agent gate:
  `smart learned-release-readiness --json --require-default-ready`;
- guarded macrohash MCTS-replacement: 510 refine-source replay states, 0 losses
  against the exact 16-skill fallback portfolio, 26.3% fewer exact skill
  attempts;
- substructure planner top-3: 507 fresh generated states, 0 losses, 81.25%
  fewer skill attempts than the 16-skill portfolio;
- C++ DeepSets held-out refine router: 264 held-out states, 0 losses, 38.7%
  fewer exact checks, 1.36x candidate-loop speedup versus the oracle pool;
- Transformer model-only remains rejected for release default because it has
  nonzero held-out regret; see
  [`LEARNED_ROUTER_RELEASE.md`](LEARNED_ROUTER_RELEASE.md).

The current Apple Silicon release candidate was checked on June 4, 2026 with:

- `python3 -m pytest -q tests`: `185 passed`;
- `smart learned-release-readiness --fail-if-not-ready --require-default-ready`: passed;
- local wheel build: `python3 setup.py bdist_wheel --dist-dir wheelhouse-local-check`;
- release artifact audit: `python3 scripts/audit_release_wheel.py wheelhouse-local-check/*.whl`: passed;
- smoke native-to-learned-macro handoff: 5/5 `smoke_5` C++ native MCTS outputs
  accepted by `macro_skill` with exact non-worse validation;
- smoke render from learned macro output: 5/5 rendered successfully;
- manual image inspection confirmed a non-empty rendered airplane bbox preview.

The CI release workflow still remains the authoritative multi-platform wheel
gate before PyPI upload.

## GitHub Release Build

Use `.github/workflows/wheels.yml`.

1. Run the workflow manually for a release candidate or push a `v*` tag for a
   publish release.
2. Confirm all `build-wheel` jobs pass for the configured CPython versions and
   platforms.
3. Confirm every wheel job runs `python scripts/audit_release_wheel.py
   wheelhouse/*.whl`.
4. Confirm every wheel job installs the wheel and runs
   `smart-smoke-console-scripts`.
5. Confirm the `sdist` job audits `dist/*.tar.gz` and passes `twine check`.
6. Confirm `publish-pypi` re-runs `twine check dist/*` and
   `python scripts/audit_release_wheel.py dist/*` immediately before upload.

Windows wheels are intentionally not in the release matrix until the fixed
Manifold bridge is validated there.

## PyPI Trusted Publishing

Before publishing the package to PyPI, configure PyPI Trusted Publisher:

- Repository: this SMART GitHub repository.
- Workflow file: `.github/workflows/wheels.yml`.
- Environment: `pypi`.
- Package: `smart-bbox`.

The workflow has `id-token: write` permission and uploads only on `v*` tag
pushes.

For the first PyPI upload, `smart-bbox` can appear under PyPI's pending
publishers rather than as an existing project. That is expected: the project is
created by the first successful trusted publish. If a tag run failed before the
pending publisher was configured, rerun the `Wheels` workflow or push the next
version tag instead of moving an existing tag.

If PyPI reports `invalid-publisher`, compare the claims printed in the failed
job with the PyPI publisher settings. For this repository, the expected claims
are `repository=chpark1111/SMART`,
`workflow_ref=chpark1111/SMART/.github/workflows/wheels.yml@refs/tags/<tag>`,
and `environment=pypi`. Old failed tag runs stay visible in GitHub even after a
later tag publishes successfully.

## First Release Tag

After local preflight passes and PyPI Trusted Publishing is configured, publish
the first release by pushing the annotated tag:

```bash
git tag -a v0.1.0 -m "SMART 0.1.0"
git push origin v0.1.0
```

The `Wheels` workflow builds and audits wheels/sdist again before PyPI upload.
Do not upload stale local artifacts manually unless the GitHub release workflow
is unavailable and the local artifacts have just passed `smart-release-preflight`.

## Post-Install Smoke

After installing a release wheel outside the source tree:

```bash
python - <<'PY'
import smart
import smart.native as sn
import smart.cpp as sc
import smart.pymesh_compat as compat
import pymesh
import subprocess
from smart.pipeline.config import load_config
from smart.pipeline.tools import diagnose_environment

status = diagnose_environment(load_config("smoke_5.yaml"))
checks = {item["name"]: item for item in status["checks"]}
profiles = smart.config_profiles()
native_bin = smart.native_executable_path()

assert sn.using_cpp()
assert sc.using_cpp()
assert compat.form_mesh is pymesh.form_mesh
assert native_bin is not None
assert "smart-cpp-native commands" in subprocess.check_output([str(native_bin), "--help"], text=True)
assert checks["pymanifold"]["ok"]
assert checks["SMART_MANIFOLD_PYTHON"]["ok"]
assert checks["smart-cpp-extension"]["ok"]
assert checks["smart-cpp-native"]["ok"]
assert any(item["name"] == "smoke_5.yaml" for item in profiles)
assert any(item["name"] == "example_3x3.yaml" for item in profiles)
print("smart-bbox release smoke ok")
PY
```

This check should use the bundled `smoke_5.yaml` config and bundled
`pymanifold` runtime, not files from the source checkout.
