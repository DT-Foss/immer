# All-layer Fast-MLP

The all-layer bank extends the existing row-routed p4/k32 executor to every
Qwen3.8 MLP layer without collecting prompts or running the model. It is a
runtime optimization with an exact fallback, not a quality or promotion
claim.

## What is built

Run the builder against an already authenticated local causal bundle:

```bash
PYTHONPATH=src python scripts/qwen38_fast_mlp_weight_only_build.py \
  /path/to/qwen-causal-bundle \
  /path/to/fast-mlp-artifacts
```

The builder verifies the causal bundle, reads Gate/Up/Down tensors through the
causal pager, and publishes the three directories already consumed by
`Qwen38FastMlpPaths`:

```text
fast-mlp-artifacts/
├── mlp-pilot-router-p4-k32-v1/
│   └── weight-only-plan.json
├── mlp-pilot-down-transpose-v1/
│   ├── model.safetensors.index.json
│   ├── pilot-transpose-manifest.json
│   └── pilot-down-transpose-layer-00..63.safetensors
└── mlp-pilot-packed-v1/
    ├── model.safetensors.index.json
    ├── pilot-weight-manifest.json
    └── pilot-weights-layer-00..63.safetensors
```

Publication is deterministic and restart-safe. Existing identical shards are
reused; a symlink, changed file, hash mismatch, identity collision, partial
layer inventory, or crossed bundle/index/layout fails closed. The builder does
not import Hugging Face, tokenize text, create a prompt cohort, invoke a model
forward, or open cartography/holdout data.

For each neuron, the initializer computes the exact joint second moment

```text
E[g²u²] = (||Wg||² ||Wu||² + 2〈Wg,Wu〉²) / hidden_dim²
```

under an isotropic Gaussian hidden vector. Each 64-neuron block is divided
into four stable moment strata. One deterministic representative per stratum
becomes a pilot and its stratum mass becomes the initial coefficient. This is
only a weight-derived cold start; it carries no prompt-generalization label.

On the official `17,408 × 5,120` BF16 MLP, a single/reused p4/k32 route reads
the equivalent of `18.0147%` of the three projection rows. The immutable local
transpose payload is about `11.41 GB` and the packed pilot payload is about
`2.14 GB`; these are built once instead of transporting the full `34.23 GB`
MLP stack on every target wave.

## Online confirmation and fallback

Weight-only mounts attach `MlpPilotOnlineController`. The executor exposes two
methods to the model runtime:

```python
decision = executor.decision(layer=layer, row_count=row_count)
if decision.use_sparse:
    output, trace = executor.execute_many(hidden_rows, layer=layer)
else:
    gate = full_gate_projection(hidden_rows)
    up = full_up_projection(hidden_rows)
    activated = swiglu(gate, up)
    output = full_down_projection(activated)
    executor.observe_full(
        layer=layer,
        gate=gate,
        up=up,
        activated=activated,
        output=output,
    )
```

`observe_full` consumes values the unchanged exact path already produced. It
adds no checkpoint read and no model forward. It measures the pre-update route
against exact activation energy, updates a decayed recursive-ridge model, then
discards every row. Only fixed-size sufficient statistics and counters remain.

The decision policy requests the exact path when:

- a cold route has no exact confirmation yet;
- fewer than the configured number of exact rows have been observed;
- confirmed activation-energy capture is below the configured floor;
- the next periodic confirmation is due; or
- the layer or target-wave width is outside the mounted ABI.

Typical chat prefill already takes the exact multi-row path, so it confirms all
64 layer routers before the first continuation wave without extra work. A
restored state with no prefill requests one exact confirmation first.

Pass `online_state_path` to `open_qwen38_fast_mlp` to persist learning. The
state is plan/config-bound, canonical and SHA-sealed, guarded by a process
lock, flushed once per request, written atomically, and capped at a fixed array
shape. It stores no
prompt, token, hidden, activation, or output content. Omitting the path keeps
the same bounded state in memory for the mount lifetime.

The chat CLI exposes the same mount:

```bash
PYTHONPATH=src python -m immer chat "<arbitrary text>" \
  --qwen38-causal-bundle /app/models/Qwen3.8-27B \
  --qwen38-tokenizer /app/models/Qwen3.8-27B/tokenizer.json \
  --fast-mlp /path/to/fast-mlp-artifacts \
  --fast-mlp-online-state /path/to/qwen-fast-mlp-state.json
```

Legacy prompt-fitted banks retain their existing behavior. They continue to
load `fit.json` plus `affine-fit.json`, return `sealed-prompt-fit` decisions,
and ignore exact observations. Mixing legacy and weight-only analysis files is
rejected rather than guessing which authority should win.
