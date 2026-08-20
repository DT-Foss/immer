# IMMER architecture

## Ownership

IMMER owns only the integration contracts, routing policy, adapters, manifests,
and cross-component tests. The upstream projects own their model/runtime
implementations and research claims.

| Layer | Owner | Boundary |
| --- | --- | --- |
| Meaning and exact arithmetic | FERTIG | `solve(text) -> answer | abstain` |
| Persistent sequential host | o1-state / gssm | host-specific runtime adapter |
| Capability sidecars | OrganBank / grafting | digest-bound organ load and route |
| Causal attention | CRSA | measured attention backend/plan |
| Evidence and replay | FLCA / FORGE | typed route, compile, replay |
| Deployment compiler | Liquid-QAD | precision plan and acceptance evidence |
| Integration | IMMER | contracts, policy, provenance |

## Runtime flow

```text
request
  │
  ▼
CapabilityRouter ── no safe route ──> abstain
  │
  ├── exact_math ──> FertigAdapter ──> exact answer / abstain
  ├── persistent_state ──> O1StateAdapter ──> external runtime result
  ├── evidence_route ──> FLCAAdapter ──> route / hold
  └── attention_plan ──> CRSA/QAD external plan reference
```

No adapter is allowed to convert “backend unavailable” into a guessed answer.
No recipe projection is a weight transfer. No benchmark artifact becomes a
claim unless its upstream evidence contract says it may.

## FERTIG correction

FERTIG is currently integrated as a small exact solver. The active solver order
is `bindings → semantic → math → miner`, with abstention when no engine can
prove a result. The older architecture document in the source checkout
describes a broader compiler framing and is therefore not used as the IMMER
contract.

## Publication boundary

Public IMMER contains code that is small enough to audit, source references,
and reproducibility metadata. Local checkpoints, Hugging Face caches, vendored
build trees, and private evidence archives remain outside the repository.
