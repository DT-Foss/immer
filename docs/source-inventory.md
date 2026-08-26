# Public Source Inventory

Release: 0.8.0 · 2026-08-26

This inventory names public components and the role each one plays in IMMER.
Private machines, local archives, unpublished state, and working directories
are outside the repository record.

| Component | Public source | IMMER role |
|---|---|---|
| IMMER | [DT-Foss/immer](https://github.com/DT-Foss/immer) | integration runtime and release authority |
| FERTIG | [DT-Foss/FERTIG](https://github.com/DT-Foss/FERTIG) | grounded deterministic execution and verification |
| o1-state | [DT-Foss/o1-state](https://github.com/DT-Foss/o1-state) | persistent O(1)-state lineage and external knowledge index |
| dotcausal | [DT-Foss/dotcausal](https://github.com/DT-Foss/dotcausal) | public `.causal` format lineage |
| Qwen3.8-27B | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | primary local teacher, causal execution target, cartography source, and novelty path; weights remain external |
| Qwen3.5-0.8B | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | local transactional drafter for exact Qwen3.8 verification; weights remain external |
| DeepSeek-V4-Flash-0731 | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | transport, paging, causal-append, and control laboratory; weights remain external |
| safetensors | [huggingface/safetensors](https://github.com/huggingface/safetensors) | immutable tensor container and range layout |

## In-repository components

| Component | Public path |
|---|---|
| primary causal Qwen runtime | `src/immer/runtimes/qwen3_8/` |
| local Qwen drafter | `src/immer/runtimes/qwen3_8/local_draft.py` |
| O1 measurement runtime | `src/immer/runtimes/o1_state/` |
| SemanticWeightAtlas and Markov-OoE | `src/immer/runtimes/ooe/` |
| action-conditioned world model and planner | `src/immer/runtimes/ooe/world_model.py`, `src/immer/runtimes/ooe/planning.py` |
| hierarchical options and contraction ledger | `src/immer/runtimes/ooe/options.py`, `src/immer/runtimes/ooe/contraction_ledger.py` |
| continuous reservoir and novelty gate | `src/immer/runtimes/ooe/reservoir.py`, `src/immer/runtimes/ooe/novelty.py` |
| ComputeCrystal VM and operator graph | `src/immer/runtimes/ooe/compute_crystals.py`, `src/immer/runtimes/ooe/compute_graph.py` |
| charged-prefix residual execution | `src/immer/runtimes/ooe/residual_execution.py` |
| real contextual operator harvesting and demand | `src/immer/runtimes/ooe/operator_harvester.py`, `src/immer/runtimes/ooe/demand_scheduler.py` |
| constructive Birkhoff search and Crystal bases | `src/immer/runtimes/ooe/bvn_search.py`, `src/immer/runtimes/ooe/bvn_crystals.py` |
| algebraic Crystals | `src/immer/runtimes/ooe/algebraic_crystals.py` |
| exact ResultCells and S3 executor | `src/immer/runtimes/ooe/result_cells.py`, `src/immer/runtimes/ooe/s3_executor.py` |
| DeepSeek V4 runtime | `src/immer/runtimes/deepseek_v4/` |
| LiveCausal | `src/immer/knowledge/livecausal.py` |
| Safetensors streamer | `src/immer/knowledge/streamer.py` |
| Causal Prefix Sinkhorn Attention (CRSA) | `src/immer/attention/` |
| FERTIG adapter | `src/immer/cognition/fertig/` |
| exact cascade | `src/immer/cognition/exact_cascade.py` |
| frozen capability runtime | `src/immer/capabilities/` |
| public benchmark receipts | `results/` |
| release manifests | `manifests/` |

Exact revisions and integration status are recorded in
[`manifests/components.json`](../manifests/components.json).
