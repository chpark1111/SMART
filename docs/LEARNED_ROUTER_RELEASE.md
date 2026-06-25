# Learned Router Release Snapshot

This page is the short release-facing summary of the learned SMART+Agent path.
For the full research log, see [`LEARNED_ROUTER.md`](LEARNED_ROUTER.md).

## What Ships

The wheel ships a dependency-light native learned path:

```text
exact C++ refine
  -> C++ DeepSets candidate router / macrohash skill selector
  -> exact SMART/Manifold validation
  -> rollback or exact fallback when confidence is low
```

Packaged runtime assets:

- `smart/assets/policies/deepset_setaware_v2_h128_v1.smartmlp`
- `smart/assets/policies/deepset_setaware_v2_h128_dagger_b2_v12.smartmlp`
- `smart/assets/skills/macro_skill_knowledge_base_v1.json`
- `smart/assets/skills/macro_skill_candidates_4k_v1.jsonl`
- `smart/assets/skills/macro_skill_retriever_macrohash_v1.json`
- `smart/assets/skills/macro_budget_quality_rule_v1.json`
- `smart/assets/skills/macro_quality_gate_ridge_v1.json`
- `smart/assets/skills/macro_memory_policy_v1.json`

The learned model does **not** replace exact geometry scoring. It reduces the
number of expensive exact candidates by ranking or selecting candidates first;
the final accepted update is still exact-scored by SMART/Manifold.

## User Entry Points

Default learned agent path for normal package use:

```bash
smart run
smart agent-run
smart run --agent
smart --config configs/learned_default.yaml run
```

Paper-style exact reproduction remains available by selecting a paper config:

```bash
smart --config configs/paper_like.yaml run
smart --config configs/smoke_5.yaml run
```

Python:

```python
import smart

records = smart.run()
records = smart.run_agent(category="airplane")
records = smart.run("configs/smoke_5.yaml", agent=True)
```

Release gate:

```bash
smart learned-release-readiness --json --require-default-ready
smart learned-router-summary --json
smart macro-skill-summary --json
```

## Current Packaged Accuracy And Speed

The current packaged default-agent candidate is the guarded macrohash
MCTS-replacement path:

| gate | cases | quality result | exact-call effect | wall-time status |
| --- | ---: | --- | ---: | --- |
| Guarded macrohash MCTS replacement | 510 refine-source replay states | 0 losses vs exact 16-skill fallback portfolio | 26.3% fewer exact skill attempts | release-safe candidate; end-to-end mesh timing still recommended per dataset |
| Substructure planner top-3 | 507 fresh generated states | 0 losses, positive accepted update rate 100% | 81.25% fewer skill attempts vs 16-skill portfolio | opt-in planner evidence |
| Stage-source top-3, refine/MCTS sources | 456 + 456 states | 456/456 accepted for both sources | 81.25% fewer skill attempts | stage-source replay evidence |
| C++ DeepSets refine router | 1015 full-token states | 0 losses | 30.5% fewer exact checks | 1.20x candidate-loop speedup vs oracle pool |
| C++ DeepSets held-out refine test | 264 held-out states | 0 losses | 38.7% fewer exact checks | 1.36x candidate-loop speedup vs oracle pool |

Interpretation:

- Good for release: C++ DeepSets/macrohash ranking with exact validation and
  fallback.
- Not claimed: pure learned reward, pure model-only geometry replacement, or
  unconditional global-optimum guarantees.
- Conservative fallback is intentional. It is what keeps the package path
  zero-regression on the current release gates.

## Latest Transformer Research Comparison

A larger MPS candidate-set Transformer is useful as a teacher/proposer, but it
is not the wheel runtime yet. The 2026-06-25 trace90 benchmark used
mesh-disjoint splits with no train/val/test mesh leakage:

```text
replay-ready evidence: 1255 states
train/val/test:        816 / 188 / 251 states
category totals:       airplane 428, chair 407, table 420
eval used:             90 balanced states per val/test split
```

| policy | split | zero-regret | mean regret | max regret | exact checks | timing vs oracle |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Oracle exact pool | val | 100.0% | 0.000000 | 0.000000 | 61.82 | 1.00x |
| Geometry top-8 exact | val | 81.1% | 0.003503 | 0.081199 | 12.44 | 0.93x |
| Transformer top-8 exact | val | 93.3% | 0.000430 | 0.019534 | 8.00 | 1.04x |
| Transformer model-only | val | 63.3% | 0.015261 | 0.403976 | 0.00 | 1.40x |
| Oracle exact pool | test | 100.0% | 0.000000 | 0.000000 | 61.64 | 1.00x |
| Geometry top-8 exact | test | 74.4% | 0.004494 | 0.079792 | 12.43 | 0.96x |
| Transformer top-8 exact | test | 95.6% | 0.000307 | 0.017052 | 8.00 | 1.08x |
| Transformer model-only | test | 62.2% | 0.007215 | 0.231946 | 0.00 | 1.47x |

The model-only path is faster but not safe: it still has nonzero regret and
large worst-case errors. A simple coverage/proxy safety filter made the
model-only path worse (`test max regret 16.26`), so it is explicitly rejected
as a release default.

The useful conclusion is narrower and stronger: large Transformers can create
better candidate orderings and teacher labels, but release runtime should
remain native C++ top-K exact validation until the Transformer is distilled or
exported into a deterministic C++/ONNX scorer and passes the same zero-loss
gate.

## Release Decision

For the next release, ship:

1. `configs/learned_default.yaml` as the normal learned SMART+Agent entrypoint.
2. `configs/paper_like.yaml` and explicit configs for exact reproduction.
3. Packaged `.smartmlp` policies and macro-skill JSON/JSONL assets.
4. Wheel audit checks that fail if learned configs or assets are missing.

Do not ship as default yet:

- PyTorch/MPS Transformer runtime;
- `transformer_model_only`;
- learned-only geometry replacement without exact top-K validation;
- any profile that cannot pass zero-regression against exact SMART/Manifold.
