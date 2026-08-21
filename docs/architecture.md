# IMMER Architecture

IMMER is the integration runtime for a modular cognitive system. It keeps cognition, persistent state, structured capabilities, neural routing, compilation and deployment as distinct planes connected by explicit contracts.

## 1. Control plane

### FERTIG

FERTIG owns grounded interpretation and explicit cognitive structure:

```text
world / observation
  → grounding
  → binding / semantic structure
  → plan
  → skill or tool selection
  → verification
```

Its neural surface is constrained: learned models rank grounded candidates rather than becoming the source of factual state.

### IMMER registry

IMMER maps a capability name to exactly one registered component.

```text
Request(capability, payload, metadata)
  → ComponentRegistry
  → Component.handle(...)
  → Result(status, output, evidence)
```

The registry is the common execution contract across symbolic, persistent, organ and neural backends.

### FLCA

FLCA is the compilation and evidence plane. It owns typed operator classification, compiler admission and deterministic replay. IMMER consumes explicit FLCA products through component contracts.

## 2. Persistent-state plane

### o1-state

o1-state is the long-lived O(1)-state host. It supplies the persistent streaming substrate, surprise-gated plasticity, external knowledge index and portable life state. Its repository remains canonical and is pinned in `manifests/components.json`.

Host-specific execution enters IMMER through an explicit component adapter.

## 3. Capability plane

### OrganBank

Structured capabilities live as cold external artifacts. IMMER's in-tree `OrganBank` implements the artifact boundary:

```text
manifest
  → OrganDescriptor
  → capability lookup
  → SHA-256 verification
  → verified artifact path
```

The registry carries artifact metadata while model and organ weights remain external.

The structural recipe behind those artifacts is:

```text
STRUCTURAL FORM
      + MAP
      + ROUTE GATE
      + CRYSTAL
```

The host stays frozen while capability is added through explicit structure around it.

## 4. Neural-routing plane

### CRSA

The CRSA production core is included as executable source.

The critical causal path is Log-Prefix:

```text
causal logits
  → masked log-softmax
  → prefix log-usage via logcumsumexp over query rows
  → usage debit
  → exact future re-mask
  → row softmax
```

Future support remains exactly zero. The integrated tests also verify zero gradient from a current row into future query rows.

CRSA composes three principal routing roles:

- **Local** — recency-biased routing;
- **Balanced** — Prefix/RAPS usage regulation;
- **Free** — untouched causal softmax for content-addressable long-range retrieval.

The free-head invariant keeps an unrestricted causal-softmax channel available alongside specialization.

## 5. Deployment plane

QAD is the precision and deployment axis for neural students and hosts. Precision plans and compiled model artifacts remain external to the source repository.

## 6. End-to-end flow

```text
request / observation
        │
        ▼
      FERTIG
 grounding + plan
        │
        ▼
   IMMER registry
        │
 ┌──────┼───────────┬──────────────┐
 │      │           │              │
 ▼      ▼           ▼              ▼
exact  o1-state   OrganBank      LFM / CRSA
path   state      capability     neural path
 │      │           │              │
 └──────┴─────┬─────┴──────────────┘
              ▼
         verification
              │
              ▼
            result

FLCA: compile / evidence / replay across planes
QAD:  precision / deployment compilation
```

## 7. Substrate plane

The substrate (`src/immer/substrate/`) owns continuity, not behaviour:

```text
EventBus        priority channels (USER preempts INTERNAL)
LifeDaemon      life stream + state port + organ rack + services
LifeStatePort   restart-safe JSON snapshot of the whole life
OrganRack       cold organ mounting over OrganBank, digest-verified
```

Design rule: physics, not politics. The daemon makes attention, memory,
organ mounting and exact services *possible*; when and how the organism
uses them is its own first acquired competence. There is no turn loop,
no speech censor and no memory schema in the substrate.

## 8. Mind plane

```text
intent.py    rule router: TEACH / RECALL / MATH / STATUS / CHAT
memory.py    SpanStore — taught facts outside the weights, restart-safe
library.py   recall misses become harvest cards with provenance tags
council.py   BO3 deliberation: majority wins, abstention counts
```

Knowledge lives OUTSIDE the weights: David teaches (`merke: …`), the
organism recalls; when recall misses, the mouth harvests and the library
keeps the card — tagged `david` or `harvest:<brain>:<model>`. Fuzzy small-
brain knowledge is visible as such.

## 9. Learning plane

`runtimes/o1_state/plasticity.py` — `LearningStream`: rolling-quantile
surprise gate over per-chunk loss; only surprising chunks trigger gradient
steps; surprising spans are buffered; `/sleep` replays them at low LR
(consolidation without drift). torch threads capped at 1 (house rule).

## 10. Mouth plane

`runtimes/qwen/adapter.py` — local Qwen from the HF cache as the fluent
surface. Council mode runs three personas (Basis/Kritiker/Freigeist) over
ONE shared weight set. The exact mouth (FERTIG) answers before the fluent
one is asked.

## 11. Suite plane

`suite.py` — every turn rewrites `status.json` (machine feed) and appends
`metrics.jsonl` (audit/graphs). The dashboard serves `/status` JSON and a
dark human UI on :8787. One metrics heart for both audiences.

## 12. Source ownership

- **FERTIG** is canonical in `DT-Foss/FERTIG`;
- **o1-state** is canonical in `DT-Foss/o1-state`;
- **FLCA** is canonical in `DT-Foss/FLCA`;
- **CRSA operators** live in IMMER until they receive a canonical repository;
- **OrganBank integration** lives in IMMER while organ artifacts remain external;
- **QAD model artifacts** remain external.

Machine-readable revisions are recorded in `manifests/components.json`.
