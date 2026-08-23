# Public Source Inventory

Release: 0.8.0 · 2026-08-23

This inventory names public components and the role each one plays in IMMER.
Private machines, local archives, unpublished state, and working directories
are outside the repository record.

| Component | Public source | IMMER role |
|---|---|---|
| IMMER | [DT-Foss/immer](https://github.com/DT-Foss/immer) | integration runtime and release authority |
| FERTIG | [DT-Foss/FERTIG](https://github.com/DT-Foss/FERTIG) | grounded deterministic execution and verification |
| o1-state | [DT-Foss/o1-state](https://github.com/DT-Foss/o1-state) | persistent O(1)-state lineage and external knowledge index |
| dotcausal | [DT-Foss/dotcausal](https://github.com/DT-Foss/dotcausal) | public `.causal` format lineage |
| DeepSeek-V4-Flash-0731 | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | active frontier checkpoint; weights remain external |
| Qwen3.8-27B | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | measured secondary donor and comparison runtime; weights remain external |
| safetensors | [huggingface/safetensors](https://github.com/huggingface/safetensors) | immutable tensor container and range layout |

## In-repository components

| Component | Public path |
|---|---|
| DeepSeek V4 runtime | `src/immer/runtimes/deepseek_v4/` |
| LiveCausal | `src/immer/knowledge/livecausal.py` |
| Safetensors streamer | `src/immer/knowledge/streamer.py` |
| Causal Prefix Sinkhorn Attention (CRSA) | `src/immer/attention/` |
| FERTIG adapter | `src/immer/cognition/fertig/` |
| exact cascade | `src/immer/cognition/exact_cascade.py` |
| frozen capability runtime | `src/immer/capabilities/` |
| O(1)-state runtime | `src/immer/runtimes/o1_state/` |
| public benchmark receipts | `results/` |
| release manifests | `manifests/` |

Exact revisions and integration status are recorded in
[`manifests/components.json`](../manifests/components.json).
