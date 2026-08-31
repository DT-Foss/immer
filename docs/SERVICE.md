# Resident local Qwen service

Install IMMER with its neural runtime and point it at an existing causalized
Qwen3.8-27B bank:

```bash
python -m pip install '.[neural]'
export IMMER_QWEN38_ROOT=/absolute/path/to/Qwen3.8-27B
immer doctor --qwen38-root "$IMMER_QWEN38_ROOT"
immer chat --service
```

Every later command with the same runtime profile connects automatically:

```bash
immer chat "Explain the local inference path."
immer chat --jsonl
immer chat --interactive
```

`--direct` bypasses the service. `--socket` or `IMMER_QWEN38_SOCKET` selects a
different local socket. `--inference-economics-state` selects the append-only
receipt ledger.

The client performs a profile ping before dispatch. The profile binds model,
tokenizer, Q4/MLP/Draft paths, decoding limits, system prompt, and runtime code.
An absent or different service enters the direct path before sending the
prompt. A transport failure after dispatch returns an error and never repeats
the request.

Single and JSONL requests are stateless. Interactive requests keep one RAM-only
history; `/clear` removes it. Contextual follow-ups use raw Qwen. No transcript
is persisted.

Deployment templates:

- [systemd user service](../deploy/systemd/immer-qwen.service)
- [launchd agent](../deploy/launchd/com.dtfoss.immer-qwen.plist)
- [environment example](../deploy/immer-qwen.env.example)
