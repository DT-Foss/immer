# Security

Do not commit credentials, private traces, model caches or machine-specific secrets.

External FERTIG and OrganBank paths are configured explicitly at runtime.
Frozen host and organ artifacts are accepted only after SHA-256 verification;
PyTorch bundles are loaded with `weights_only=True` and organ states also carry
an internal digest. `immer artifacts import` copies from an explicit source
read-only, atomically, and refuses to overwrite a wrong destination by default.

WorldStream uses exact single-range reads, preflights a hard byte budget and
SHA-verifies resume-cache entries. Full weight-file reads are rejected. Pin a
remote model revision to a commit digest; a cache hash cannot prove that a
mutable `main` revision did not move.

FERTIG desktop execution, recording and state mutation require explicit
backend injection. A normal `serve`, `solve`, `doctor` or test run does not
grant those capabilities.
