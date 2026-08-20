# IMMER

IMMER is the integration shell for a persistent neuro-symbolic system whose
parts remain independently testable:

```text
FERTIG solver ──┐
o1-state host ──┼──> IMMER contracts/router ──> verified result or abstention
OrganBank ──────┤
CRSA / QAD ─────┤
FLCA / FORGE ───┘
```

The repository is deliberately small. It connects the projects; it does not
flatten their histories into a multi-gigabyte folder or commit model weights.
The source workspaces, exact commits, local-only artifacts, and inclusion
decisions are recorded in [`manifests/sources.lock.json`](manifests/sources.lock.json).

## Current integration boundary

The first executable boundary is the FERTIG unified solver. Its current role is
minimal and explicit:

```text
bindings → semantic → math → miner → abstain
```

It returns an answer only when an engine can justify one. The IMMER adapter
preserves that behavior and reports whether the backend was unavailable,
abstained, or returned an exact result. The old broad “intelligence compiler”
framing is retained only as historical context; it is not the current FERTIG
runtime contract.

The other projects are connected as capability boundaries, not silently
treated as interchangeable weights:

- `o1-state` / `gssm`: persistent streaming hosts;
- OrganBank / organ grafting: cold-loadable capability sidecars around a frozen
  host;
- CRSA: causal attention and routing research for Transformer students;
- FLCA / FORGE: evidence-gated routing, compilation, replay, and provenance;
- Liquid-QAD: deployment and precision compilation experiments.

## Quick start

The core package uses only the Python standard library at runtime:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests -v
immer solve "A bakery sold 12 cakes on Monday and 15 cakes on Tuesday. How many cakes did they sell in total?"
```

Without a local FERTIG checkout, the command fails closed with an abstention
reason. Point it at the audited checkout with:

```bash
export IMMER_FERTIG_ROOT=/Users/bhkmie/Documents/Forschung/LanguageModel/FERTIG
```

## Repository map

- [`docs/architecture.md`](docs/architecture.md) — ownership and data flow;
- [`docs/source-inventory.md`](docs/source-inventory.md) — what was inspected;
- [`docs/provenance.md`](docs/provenance.md) — digest and publication policy;
- [`src/immer/contracts.py`](src/immer/contracts.py) — stable cross-project IR;
- [`src/immer/adapters.py`](src/immer/adapters.py) — fail-closed boundaries;
- [`src/immer/runtime.py`](src/immer/runtime.py) — orchestration;
- [`manifests/`](manifests/) — pinned source and module metadata.
