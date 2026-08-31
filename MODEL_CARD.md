# IMMER v1.0

IMMER v1.0 is a local inference runtime for causalized Qwen3.8-27B weights. It
combines direct mmap Q4/Q8 execution, native DeltaNet/Conv kernels,
transactional continuation state, Markov/MTP drafting, dynamic MLP-page
routing, O1-valued runtime learning, FERTIG verification, and zero-forward
ResultCells behind one resident Unix-socket service.

## Runtime contract

- Base model: `Qwen/Qwen3.8-27B`
- Pinned revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Weight location: local causal bundle supplied by the operator
- Packed plane: Q4_0 text projections and Q8_0 embedding/LM head
- Product entry point: `immer chat`
- Resident service: `immer chat --service`
- Interface: arbitrary UTF-8 text to a target-confirmed text result and
  machine-readable execution receipt

Qwen weights are not part of the IMMER package. The operator supplies a
checkpoint under the upstream model license and builds or mounts the matching
local causal/Q4 bank.

## Measured product path

The first corrected resident-service request reached EOS after 39 generated
tokens. It executed 29 target forwards and avoided 10 target forwards
(`25.64%`) through the Markov/MTP path. The runtime proposed 24 draft tokens,
accepted 11, completed generation in 129.47 seconds, and peaked at
1,739,022,336 resident bytes on the 16-core AVX2 reference host.

The established short Full-Q4 reference trace peaks at 1,170,735,104 bytes,
down from 16,011,718,656 bytes with the same generated token trace. Selected
receipts and their measurement boundaries are listed in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Execution boundary

Qwen remains the semantic committer for novel text. Markov/MTP proposals,
dynamic MLP routes, O1 reward, and stored results reduce target work without
replacing that authority. Context-free turns pass through FERTIG/OoE
adjudication. Contextual follow-ups remain on raw Qwen so single-question
solvers cannot override conversation-dependent meaning.

v1.0 stores exact results and learned deterministic transformations with zero
Qwen forwards. Broad discharge of generic ComputeCrystals into new natural
answers is the v2 action-bank line.

## Local data

The service socket is local and mode `0600`. Conversation history exists only
in service RAM and is removed by `/clear` or normal interactive exit. The
Inference Economics ledger stores prompt hashes, output/result identities,
runtime identity, forwards, bytes, time, RSS, drafting, page actions, and
reward. It stores no raw prompt or chat transcript.

Private causal graphs, learned controller state, Graft payloads, receipts,
credentials, server paths, and model weights are outside the release package.

## Research record

- David Tom Foss, *The `.causal` Format: Embedded Deterministic Inference for
  Domain-Agnostic Knowledge Graph Amplification*, IEEE IRI 2026.
- David Tom Foss, *Deterministic Validation for Reliable LLM-Based Causal
  Knowledge Extraction*, ICECET 2026.
- Mathematical and peer-reviewed lineage:
  [docs/research.md](docs/research.md).
