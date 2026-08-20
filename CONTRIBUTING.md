# Contributing

Keep IMMER small and auditable. Add integration code only when it is required
to start, test, or reproduce a cross-project path. For copied or adapted code,
record `origin_repo`, `origin_commit`, `origin_path`, and `integrated_path` in
`manifests/modules.lock.json`.

Do not commit checkpoints, model caches, training traces, private evidence, or
large benchmark dumps. Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/bootstrap.py
```
