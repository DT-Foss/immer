# Research Map

IMMER connects research programs that solve different parts of the same system.

## FERTIG

Grounded neuro-symbolic cognition with a deterministic core, executable learned skills, constrained neural ranking, verification loops and process distillation.

Current structure-solver milestone: **1060/1319 GSM8K = 80.36% coverage,
259 safe abstentions, 0 incorrect and 0 error outcomes on the full test
split.** The report retains every item and pins dataset, runner and solver-tree
digests. An unverified operations-template fallback first produced 17/17
wrong attempts; it is now proposal-only and those cases are `must_abstain`
regressions.

## DeepSeek-V4 range runtime

The pinned 304B DeepSeek-V4-Flash checkpoint now executes through a local
43-layer, stateful decoder without loading the checkpoint as a resident model.
The path uses the original FP8/FP4 Safetensors, native compressed sparse
attention, hash/score MoE routing, HyperConnections, a bounded 12 GiB verified
range cache and the repository's fixed-role CRSA residual graft. Completed
integration runs prove one full autoregressive pass and a two-pass CRSA decode;
meaningful prompt quality and paired graft superiority remain open gates.

## Organ Grafting

`Organs, Not Weights` defines capability transplantation into a frozen living host.

Measured anchors from v0.1:

- host: **1,713,673 parameters**, 909,676,544 streamed tokens;
- donor: 27B, approximately **15,750×** larger;
- structured organs: **5,732–14,625 parameters**;
- host language NLL: **8.6656**, preserved through headline runs;
- recurrent chain organ trained at length 2: **1.000 through length 12** after crystallization;
- cold OrganBank Ship v6: **152/152 measured tasks**, **30/30 text routes**.

The central design rule is structure-first capability:

```text
STRUCTURAL FORM + MAP + ROUTE GATE + CRYSTAL
```

## Causal Prefix–Sinkhorn / CRSA

The v0.5 attention program establishes strict causal prefix-mass balancing and role-complete causal routing.

Measured anchors:

- exact zero future gradient and support for the causal operator;
- forced diagonal maximum at epsilon=0 and k=1;
- lower-triangular bistochastic obstruction: exact row+column stochasticity collapses to identity;
- k=1 remains the useful finite routing regime;
- scale gains: **0.984, 1.106, 1.173, 1.182, 1.184 bpb** at widths 128, 256, 384, 512, 1024;
- variable-lag unseen accuracy: **0.9293** for 2L+1B+1F→F versus **0.7745** softmax;
- every tested free-head-preserving lag-48 program: **1.000**;
- **69 automated tests** in the v0.5 program.

CRSA treats attention heads as routing roles rather than interchangeable copies of one operator.

## o1-state

The persistent host contributes the O(1)-state organism: constant-memory streaming, surprise-gated plasticity, external indexed memory, sleep/consolidation and portable state.

## FLCA

FLCA contributes typed operator evidence, compilation, schedule/runtime lowering and replay.

## QAD

QAD contributes the deployment axis. In the Organ Grafting deployment experiment, host fake-quant int4 self-distillation moved NLL from PTQ 8.6671 to 8.6654 against fp32 8.6656 while the tested organ capability remained intact.
