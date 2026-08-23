# Security

immer treats weights, traces, credentials and machine topology as local data.
Git excludes model bundles, private artifacts, route/access traces, environment
files, credentials and private keys. The public-text regression test rejects
absolute home/server paths, routable or VPN addresses, credential filenames and
machine-specific paths in published benchmark reports.

Every external harness path and remote endpoint is supplied at runtime. Public
reports store repository paths relative to the checkout and replace external
machine locations with a typed descriptor. Measurements, hashes and checkpoint
identities remain intact.

Frozen host and organ artifacts enter through `immer artifacts import`: explicit
read-only source, SHA-256 verification, atomic installation and collision refusal.
PyTorch state loads use `weights_only=True`; organ state carries its own digest.

WorldStream performs exact range reads, enforces byte budgets and verifies every
resume-cache leaf. Remote checkpoints are bound to immutable commit revisions.
Full shard reads are rejected.

FERTIG desktop execution, recording and state mutation exist only behind explicit
backend injection. `serve`, `solve`, `doctor` and the test suite grant none of
those capabilities.
