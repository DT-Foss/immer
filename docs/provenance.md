# Provenance and publication policy

Every external dependency is identified by:

1. canonical project name and upstream URL;
2. local source path used for inspection;
3. exact commit when the source is a Git repository;
4. a clear status (`pinned`, `working-tree`, or `artifact-only`);
5. an inclusion decision explaining why it is referenced rather than copied.

`manifests/modules.lock.json` records the adapter-level contract and source
commit. `manifests/artifacts.sha256` is reserved for intentionally published
small artifacts; no multi-gigabyte model or cache is copied into IMMER.

Before publication, inspect the staged diff and run the standard-library test
suite. The publishing identity for the initial repository is David Tom Foss
through the authenticated `DT-Foss` GitHub account.
