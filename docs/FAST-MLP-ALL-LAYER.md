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

## Adaptive block width

The packed pilot and full transpose artifacts already contain every neuron, so
route width can grow without rebuilding or copying their `13.55 GB` payload.
For every exact observation the controller orders blocks by the pre-update
pilot score and evaluates the actual cumulative capture curve

```text
C(w) = min_rows((pilot_energy + sum(top-w nonpilot block energy)) / total_energy)
```

at widths `32, 40, …, max`. It persists the smallest measured width reaching
`max(plan_min_capture, 0.50)`. If even the configured maximum (default `128`)
misses that exact floor, the layer stays on the full MLP path. There is no
linear extrapolation from the old width-32 result.

The matching predicted score mass is also persisted. On a later sparse wave,
each row is widened until its current cumulative pilot-score mass reaches that
confirmed mass plus the configured safety margin; the wave uses the widest
row and never exceeds the exact-observed maximum. Thus cross-context score
diffusion can increase work but cannot silently narrow a confirmed route.

For the official topology, the packed single/reused-route row fraction is

```text
F(w) = (272 × 4 + 64 × w) / 17,408.
```

Representative bounds are `18.0147%` at width 32, `29.7794%` at 64,
`41.5441%` at 96, and `53.3088%` at 128. Against the `34.23 GB` full 64-layer
MLP wave these correspond to about `6.17`, `10.19`, `14.22`, and `18.25 GB`
for one/reused routes. Width 128 needs about `80 MiB` for one Down route and
`160 MiB` when the overlap cache also assembles it, so a `192 MiB` auxiliary
resident limit admits the full configured range. Smaller mounts automatically
cap the width to one route that fits.

Online state v2 binds the effective width policy and persists the selected
width, exact capture, predicted/required score mass, last executed width, and
update count. Existing v1 state is migrated in place: recursive-ridge arrays,
capture history, and counters are retained, width starts at the original
p4/k32 value, and the next exact path computes the first authoritative curve.
The immutable weight-only plan and all payload/manifest hashes remain unchanged.

## Output calibration and economic fallback

Activation capture is not an output-quality certificate. Weight-only layers
therefore remain exact until a bounded sparse-Down shadow has been confirmed.
After an exact MLP finishes, the executor takes only the configured worst
capture row (default one), masks its activation to the residency-capped route,
and applies the exact Down matrix while that matrix is still resident for the
ordinary full projection. It compares this shadow with the exact output already
in memory. The shadow performs no model forward or additional source read,
does not alter the current exact result, and retains no activation or output.

Metrics are prequential: the prior scalar correction is scored before the new
row updates its sufficient statistics. One layer is sparse-eligible only when
the recent bounded window has at least eight confirmed rows, worst cosine is
at least `0.999`, worst relative L2 is at most `0.05`, and the exact capture
floor also passes. The learned `y ≈ scale × sparse + bias` correction is then
applied to sparse execution. A failed metric window keeps the full MLP.

State v3 persists only scalar sufficient statistics, the bounded metric
window, correction coefficients, and counters. Both v1 capture-only state and
v2 adaptive-width state migrate with `output_confirmed_rows=0`; neither can
authorize sparse execution until new exact-path output shadows pass.

Pilot scoring also estimates the complete wave's target-plus-auxiliary row
transport before any dynamic Gate/Up/Down range is read. Packed-pilot rows,
the Gate/Up union, route-cache transitions, and residency are included. If the
estimate exceeds the configured material-benefit ceiling (default `0.90` of
the full `3 × 17,408` rows), `MlpPilotNonBeneficialRoute` requests the
unchanged full MLP immediately. This prevents a sparse target-byte saving from
being outweighed by auxiliary transport.
