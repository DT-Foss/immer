# IMMER

**Unified cognitive runtime for grounded reasoning, persistent state, structured capabilities, causal neural routing and hardware-aware deployment.**

IMMER joins grounded cognition, persistent state, structured capabilities, causal neural routing, compilation and deployment behind one explicit runtime contract.

```text
                                  IMMER
                                    │
             ┌──────────────────────┼──────────────────────┐
             │                      │                      │
             ▼                      ▼                      ▼
          FERTIG                 o1-state              LFM / CRSA
   grounded cognition       persistent O(1) state     neural execution
             │                      │                      │
             └──────────────┬───────┴────────┬─────────────┘
                            ▼                ▼
                        OrganBank           FLCA
                 structured capabilities   compilation
                            │                │
                            └────────┬───────┘
                                     ▼
                                    QAD
                              deployment target
```

The system separates six concerns:

| Plane | Component | Role |
| --- | --- | --- |
| Cognition | **FERTIG** | grounding, explicit reasoning, executable skills, verification and process compilation |
| Persistent state | **o1-state** | constant-memory streaming state, lifelong adaptation and external memory |
| Capabilities | **OrganBank / Grafting** | digest-addressed structural organs, crystallization and cold capability loading |
| Neural routing | **CRSA** | causal Local / Balanced / Free attention programs with an untouched free-head invariant |
| Compilation | **FLCA** | evidence-typed operator routing, compilation and deterministic replay |
| Deployment | **QAD** | precision-aware distillation and hardware deployment |

FERTIG, o1-state and FLCA remain canonical projects with their own repositories. IMMER owns the integration contracts and carries the components that do not yet have a separate canonical home, including the CRSA operator core and the cold OrganBank registry.

## Research foundations

### FERTIG — grounded cognition

FERTIG supplies grounded neuro-symbolic cognition: deterministic symbolic state, grounding, the Desktop Apprentice, constrained HSSLM ranking, verification loops and agent-trace process compilation.

Current GSM8K structure-solver result:

- **1056 / 1319 = 80.06% correct**
- **0 incorrect answers** in the full test
- **731 binding regression tests passing**

### Organ Grafting — capabilities without host weight edits

The Organ Grafting program transfers capabilities from a 27B donor into a frozen 1.713M-parameter o1-state host through small structured logit-space organs. The strict recipe is:

```text
STRUCTURAL FORM + MAP + ROUTE GATE + CRYSTAL
```

The measured Ship v6 system used a cold OrganBank with additive, multiplicative, cyclic and decimal capabilities and reached **152/152 measured tasks**, **30/30 text routes**, with the host language NLL unchanged. Organ artifacts remain external and digest-addressed; IMMER contains the registry and verification layer, not the weights.

### CRSA — causal routing

IMMER carries the production CRSA operator core in `src/immer/attention/crsa/operators.py`, extracted from the v0.5 implementation for the overlapping operators.

The integrated operator core includes:

- causal Prefix-Sinkhorn / Log-Prefix balancing;
- RAPS diagonal debit;
- geometric prefix usage;
- reservoir prefix routing;
- Local / Balanced / Free role programs;
- untouched causal-softmax free heads;
- the full-support Sinkhorn leak probe as a negative control.

The v0.5 program verifies exact zero future support, reports **69 automated tests**, preserves long-range retrieval with a whole free head, and shows scaling gains that rise to roughly **1.18 bpb** by width 512–1024 instead of vanishing.

### o1-state — persistent organism

The persistent host remains the canonical `DT-Foss/o1-state` project. IMMER uses it as the long-lived state substrate.

### FLCA and QAD — compilation to deployment

FLCA supplies the evidence-typed compilation and replay layer. QAD supplies the precision and deployment axis for neural students and hosts. Compiler decisions enter IMMER as explicit artifacts and contracts.

## Executable kernel

```text
src/immer/
├── contracts.py                  component/request/result contract
├── registry.py                   explicit capability ownership
├── runtime.py                    unified dispatcher
├── cognition/
│   └── fertig/adapter.py         FERTIG exact-math adapter
├── capabilities/
│   └── organbank/bank.py         digest-addressed cold organ registry
└── attention/
    └── crsa/
        └── operators.py          CRSA attention implementation
```

Capability ownership is explicit in the runtime registry. External canonical components are attached by configured paths or installed packages, while in-tree components remain directly executable.

## Quick start

```bash
git clone https://github.com/DT-Foss/immer.git
cd immer

python -m venv .venv
source .venv/bin/activate
pip install -e .

immer components
immer doctor
python -m unittest discover -s tests -v
```

Enable neural routing with:

```bash
pip install -e '.[neural]'
```

Attach FERTIG:

```bash
export IMMER_FERTIG_ROOT=/path/to/FERTIG
immer solve "A store sold 12 items and then 15 more. How many were sold?"
```

Attach a cold OrganBank:

```bash
export IMMER_ORGANBANK=/path/to/organbank.json
```

A bank manifest contains artifact paths and SHA-256 digests. Model weights, GGUF files, checkpoints and Hugging Face caches stay outside Git.

## Repository layout

```text
src/immer/              executable integration runtime
tests/                  component, digest and causal-support tests
docs/                   architecture and research map
manifests/              neutral source/component pins
```

## Author

David Tom Foss · 2026
