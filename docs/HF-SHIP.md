# Hugging Face Release Boundary

Release target: 0.8.0 · 2026-08-23

Hugging Face is the final packaging surface. Development, causalization,
inference, and benchmark acceptance run locally first.

## Existing offline export

```bash
pip install -e '.[neural,export]'
python -m immer export-hf dist/immer-ship-v6
cd dist/immer-ship-v6
python -I -B verify.py
python -I -B solve.py "three plus five is"
```

The exporter builds a self-contained offline exact-core bundle, writes a
checksum manifest, and verifies it in an isolated interpreter. It contains no
Hub login or upload path.

## Publish allowlist

Only these classes of files may enter a public model repository:

- model card and public configuration;
- source files required by the offline runtime;
- redistribution-cleared model artifacts;
- public component manifests;
- public benchmark receipts;
- checksums and reproducible verification scripts;
- license and citation files.

## Mandatory exclusions

- API keys, tokens, cookies, SSH material, and environment dumps;
- private datasets, prompts, conversations, and annotations;
- local paths, usernames, hostnames, IP addresses, and machine inventories;
- caches, temporary files, resumable run state, and activation dumps;
- route traces, live causal graphs, private graph segments, and learned
  controller state;
- experimental capability-transfer code, parameters, and artifacts;
- third-party weights without an explicit redistribution grant;
- internal handoffs, candidate notebooks, and research scratch space.

## Release sequence

1. Pass the full local test and benchmark gate.
2. Build the causalized local checkpoint bundle and run plain/real/placebo
   end-to-end comparisons.
3. Freeze the public file allowlist and scan the staged tree.
4. Resolve the license chain for every redistributed artifact.
5. Build the offline bundle from the clean release commit.
6. Verify checksums and run the isolated solver.
7. Publish under the authenticated `DT-Foss` account.

No model upload is authorized until all seven steps pass.
