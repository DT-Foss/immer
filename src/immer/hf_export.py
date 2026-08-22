"""Deterministic, offline-only Hugging Face bundle export.

This module deliberately stops at a locally verified directory.  It contains
no Hub client, authentication, repository creation, or upload path.  The
weight/source license chain is not cleared, so the generated model card uses
``license: other`` and explicitly blocks public publication.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .artifacts import (
    ArtifactBootstrapError,
    ArtifactSpec,
    load_artifact_specs,
    sha256_file,
)
from .resource_paths import crsa_router_manifest, s3_ship_manifest


_BUNDLE_SCHEMA = "immer.hf-poc/v1"
_CHECKSUM_SCHEMA = "immer.hf-poc-checksums/v1"
_MANIFEST_SCHEMA = "immer.s3-ship-v6/v1"
_ORGAN_NAMES = (
    "arith-dual",
    "decimal-crystal",
    "mul-log",
    "z3-circle",
)
_EXCLUDED_NAMES = frozenset({".DS_Store"})
_EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo"})


class HfExportError(RuntimeError):
    """The offline export could not satisfy its integrity contract."""


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HfExportError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise HfExportError(f"{label} must be a JSON object: {path}")
    return document


def _verified_specs(
    manifest: Path,
    artifact_root: str | Path | None,
) -> tuple[ArtifactSpec, tuple[ArtifactSpec, ...]]:
    try:
        _, specs = load_artifact_specs(
            manifest,
            configured_root=artifact_root,
        )
    except ArtifactBootstrapError as exc:
        raise HfExportError(str(exc)) from exc

    host = tuple(spec for spec in specs if spec.label.startswith("host:"))
    organs = tuple(spec for spec in specs if spec.label.startswith("organ:"))
    if len(host) != 1 or len(organs) != 4:
        raise HfExportError("SHIP-v6 export requires exactly one host and four organs")
    if {spec.label.removeprefix("organ:") for spec in organs} != set(_ORGAN_NAMES):
        raise HfExportError("SHIP-v6 export manifest names an unexpected organ set")

    for spec in specs:
        if not spec.destination.is_file():
            raise HfExportError(f"missing verified source artifact for {spec.label}: {spec.destination}")
        actual = sha256_file(spec.destination)
        if actual != spec.sha256:
            raise HfExportError(
                f"source artifact SHA-256 mismatch for {spec.label}: "
                f"expected {spec.sha256}, got {actual}"
            )
    return host[0], tuple(sorted(organs, key=lambda item: item.label))


def _a1_state(
    source: Path,
    descriptor: Mapping[str, Any],
) -> OrderedDict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise HfExportError("HF export needs torch; install 'immer[neural,export]'") from exc

    source_format = str(descriptor.get("format", "torch-checkpoint-arms"))
    arm = str(descriptor.get("arm", "A1"))
    if arm != "A1":
        raise HfExportError(f"the PoC exporter is A1-only, not arm {arm!r}")

    if source_format == "torch-checkpoint-arms":
        try:
            checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HfExportError(f"cannot load frozen A1 source checkpoint: {exc}") from exc
        if not isinstance(checkpoint, Mapping):
            raise HfExportError("frozen host checkpoint is not a mapping")
        arms = checkpoint.get("arms")
        if not isinstance(arms, Mapping) or arm not in arms:
            raise HfExportError("frozen host checkpoint has no A1 arm")
        record = arms[arm]
        if not isinstance(record, Mapping) or not isinstance(record.get("model"), Mapping):
            raise HfExportError("frozen A1 checkpoint has no model state_dict")
        raw_state = record["model"]
    elif source_format == "safetensors-state-dict":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise HfExportError("HF export needs safetensors; install 'immer[export]'") from exc
        raw_state = load_file(str(source), device="cpu")
    else:
        raise HfExportError(f"unsupported source host format {source_format!r}")

    if not isinstance(raw_state, Mapping) or not raw_state:
        raise HfExportError("A1 model state_dict is empty")
    state: OrderedDict[str, Any] = OrderedDict()
    for name in sorted(raw_state):
        value = raw_state[name]
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise HfExportError(f"A1 state entry {name!r} is not a tensor")
        if value.layout is not torch.strided:
            raise HfExportError(f"A1 state tensor {name!r} has unsupported layout {value.layout}")
        state[name] = value.detach().to(device="cpu").contiguous()
    return state


def _save_a1_safetensors(
    destination: Path,
    state: Mapping[str, Any],
) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise HfExportError("HF export needs safetensors; install 'immer[export]'") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        # safetensors' metadata map is not emitted in stable key order by all
        # supported releases.  Keep provenance in the checksummed JSON/model
        # card and write the tensor map alone so repeated exports are bitwise
        # identical.
        save_file(state, str(destination))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HfExportError(f"cannot write A1 safetensors state: {exc}") from exc


def _copy_runtime(source: Path, destination: Path) -> None:
    if not (source / "__init__.py").is_file():
        raise HfExportError(f"IMMER runtime package is incomplete: {source}")
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts:
            continue
        if path.name in _EXCLUDED_NAMES or path.suffix in _EXCLUDED_SUFFIXES:
            continue
        if path.is_symlink():
            raise HfExportError(f"runtime package contains an unsupported symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise HfExportError(f"runtime package contains an unsupported entry: {path}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def _safe_payload_name(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or not parsed.parts or ".." in parsed.parts:
        raise HfExportError(f"unsafe bundle path: {relative!r}")
    return relative


def _payload_digests(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.name == "checksums.json" and path.parent == root:
            continue
        if path.is_symlink():
            raise HfExportError(f"bundle must not contain symlinks: {path}")
        if path.is_file():
            files[_safe_payload_name(path, root)] = sha256_file(path)
    return dict(sorted(files.items()))


def _router_digest(path: Path) -> str:
    try:
        from .attention.router import RidgeRouteHead

        return RidgeRouteHead.load(path).digest()
    except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
        raise HfExportError(f"invalid CRSA router state {path}: {exc}") from exc


def _model_card(
    *,
    source_host_sha256: str,
    model_sha256: str,
    router_digest: str,
    organs: tuple[ArtifactSpec, ...],
) -> str:
    organ_lines = "\n".join(
        f"- `{spec.label.removeprefix('organ:')}`: `{spec.sha256}`" for spec in organs
    )
    return f"""---
license: other
library_name: immer
tags:
- immer
- causal-routing
- safetensors
---

# IMMER SHIP-v6 — offline proof of concept

**LICENSE STATUS: NOT CLEARED. `license: other` is a warning, not a license
grant. Public upload or redistribution is blocked until the rights chain for
the A1 host, all four organs, the packaged runtime, and FERTIG is documented.**

This is a narrow, locally verified arithmetic runtime, not a general chat or
reasoning model. It packages the frozen A1 state, the project's own fixed-role
CRSA attention router, four crystallized organs, and FERTIG as the guarded
verifier/fallback.

## Verified scope

- SHIP-v6 canonical arithmetic: **152/152 answers and 152/152 routes**.
- Router state digest: `{router_digest}`.
- CRSA program: two Local heads, one Balanced head, one bit-exact causal Free
  head. The current measurement does **not** show an advantage over causal
  softmax.
- Static embedding-mean value sketches are excluded: that experiment scored
  24% versus 32% placebo.
- No runtime training and no donor-model loading occur in `verify.py` or
  `solve.py`.

## Artifact provenance

- Source research checkpoint SHA-256: `{source_host_sha256}`.
- Extracted A1-only `model.safetensors` SHA-256: `{model_sha256}`.
- The research checkpoint's optimizer, buffers, other arms, pending streams,
  and RNG state are not included.

Unchanged organ bytes:

{organ_lines}

## Offline use

Install the versions listed in `requirements.txt`, then run:

```bash
python -I -B verify.py
python -I -B solve.py "three plus five is"
```

Both entry points force the bundled `runtime/immer` package to the front of
`sys.path`. `verify.py` first enforces the exact file set and SHA-256 manifest,
then runs the full 152-case suite and checks the semantic router digest.

## Publication gate

This directory is technically shaped for a Hugging Face model repository, but
the exporter intentionally contains no Hub API, login, or upload code. Do not
publish it while `config.json` says `public_upload_allowed: false`.
"""


_VERIFY_SCRIPT = r'''#!/usr/bin/env python3
"""Offline integrity and SHIP-v6 verification for this exact bundle."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath

sys.dont_write_bytecode = True
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_payload(root: Path) -> dict:
    root = root.resolve()
    checksum_path = root / "checksums.json"
    try:
        checks = json.loads(checksum_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read checksums.json: {exc}") from exc
    if not isinstance(checks, dict) or checks.get("schema") != "immer.hf-poc-checksums/v1":
        raise RuntimeError("unsupported checksums.json schema")
    files = checks.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("checksums.json has no files map")

    expected = set()
    for raw_name, digest in files.items():
        parsed = PurePosixPath(str(raw_name))
        if parsed.is_absolute() or not parsed.parts or ".." in parsed.parts:
            raise RuntimeError(f"unsafe checksum path: {raw_name!r}")
        if not isinstance(digest, str) or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise RuntimeError(f"invalid SHA-256 for {raw_name}")
        expected.add(parsed.as_posix())
    expected.add("checksums.json")

    found = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".git":
            continue
        if path.is_symlink():
            raise RuntimeError(f"symlink is not allowed in exact bundle: {relative}")
        if path.is_file():
            found.add(relative.as_posix())
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise RuntimeError(f"bundle file set mismatch: missing={missing}, extra={extra}")

    for name in sorted(files):
        actual = _sha256(root / PurePosixPath(name))
        if actual != files[name]:
            raise RuntimeError(
                f"SHA-256 mismatch for {name}: expected {files[name]}, got {actual}"
            )
    return checks


def verify_runtime(root: Path, checks: dict) -> dict:
    runtime = (root / "runtime").resolve()
    sys.path.insert(0, str(runtime))
    import immer
    from immer.capabilities.s3_runtime import S3Arithmetic

    origin = Path(immer.__file__).resolve()
    try:
        origin.relative_to(runtime)
    except ValueError as exc:
        raise RuntimeError(f"checkout/runtime leak: imported immer from {origin}") from exc

    engine = S3Arithmetic(
        root / "s3_ship_v6.json",
        router_state=root / "crsa_router_v1.json",
        artifact_root=root,
    )
    report = dict(engine.benchmark())
    expected = checks.get("expectations", {})
    if not isinstance(expected, dict):
        raise RuntimeError("checksums expectations must be an object")
    cases = int(expected.get("canonical_cases", -1))
    if not (
        cases == 152
        and report.get("cases") == cases
        and report.get("ok") == cases
        and report.get("correct") == cases
        and report.get("route_correct") == cases
        and report.get("no_training") is True
    ):
        raise RuntimeError(f"SHIP-v6 benchmark failed: {report}")
    if report.get("router_state_digest") != expected.get("router_state_digest"):
        raise RuntimeError(
            "router semantic digest mismatch: "
            f"expected {expected.get('router_state_digest')}, "
            f"got {report.get('router_state_digest')}"
        )
    if report.get("host_sha256") != expected.get("model_sha256"):
        raise RuntimeError("runtime did not load the checksummed A1-only safetensors host")
    return {
        "status": "ok",
        "arithmetic": "152/152",
        "routes": "152/152",
        "router_state_digest": report["router_state_digest"],
        "router_verdict": report.get("router_verdict"),
        "no_training": True,
        "runtime_origin": origin.relative_to(root).as_posix(),
    }


def verify_bundle(root: Path | None = None) -> dict:
    selected = Path(__file__).resolve().parent if root is None else Path(root).resolve()
    checks = verify_payload(selected)
    return verify_runtime(selected, checks)


def main() -> int:
    try:
        report = verify_bundle()
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "error": type(exc).__name__, "reason": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


_SOLVE_SCRIPT = r'''#!/usr/bin/env python3
"""Solve through frozen S3 followed by the guarded FERTIG exact cascade."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtime"))

from verify import verify_payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline IMMER SHIP-v6 + FERTIG exact cascade"
    )
    parser.add_argument("question", nargs="+", help="canonical arithmetic question")
    args = parser.parse_args(argv)
    verify_payload(ROOT)

    from immer.capabilities.s3_runtime import S3Arithmetic
    from immer.cognition.exact_cascade import ExactCascade
    from immer.cognition.fertig import FertigSolver
    from immer.contracts import Request

    s3 = S3Arithmetic(
        ROOT / "s3_ship_v6.json",
        router_state=ROOT / "crsa_router_v1.json",
        artifact_root=ROOT,
    )
    cascade = ExactCascade(s3, FertigSolver())
    result = cascade.handle(Request("exact_math", " ".join(args.question)))
    payload = {
        "status": result.status.value,
        "component": result.component,
        "output": result.output,
        "reason": result.reason,
        "evidence": dict(result.evidence),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _isolated_verify(stage: Path) -> Mapping[str, Any]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if name == "PYTHONPATH" or name.startswith("IMMER_"):
            environment.pop(name, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(stage / "verify.py")],
            cwd=stage,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HfExportError(f"isolated offline verification could not run: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise HfExportError(f"isolated offline verification failed: {detail[-4000:]}")
    try:
        report = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise HfExportError("isolated verifier returned invalid JSON") from exc
    if not isinstance(report, Mapping) or report.get("status") != "ok":
        raise HfExportError(f"isolated verifier did not report success: {report!r}")
    return dict(report)


def _next_backup(target: Path) -> Path:
    base = target.with_name(f"{target.name}.backup")
    if not base.exists() and not base.is_symlink():
        return base
    for index in range(1, 10_000):
        candidate = target.with_name(f"{target.name}.backup.{index}")
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise HfExportError(f"cannot allocate recoverable backup name beside {target}")


def _target_path(output: str | Path) -> Path:
    raw = Path(output).expanduser()
    target = raw.resolve(strict=False)
    if target == target.parent or target == Path.cwd().absolute():
        raise HfExportError(f"refusing unsafe export target: {target}")
    module = Path(__file__).resolve()
    package = module.parent
    try:
        target.parent.relative_to(package)
    except ValueError:
        pass
    else:
        raise HfExportError(
            f"refusing export staging inside the IMMER source package: {target}"
        )
    if target.exists():
        try:
            module.relative_to(target.resolve())
        except ValueError:
            pass
        else:
            raise HfExportError(f"refusing to replace a directory containing IMMER source: {target}")
    return target


def export_hf_poc(
    output: str | Path,
    *,
    manifest: str | Path | None = None,
    artifact_root: str | Path | None = None,
    replace: bool = False,
) -> Mapping[str, Any]:
    """Build, verify, and atomically publish an offline HF-shaped PoC folder.

    ``replace`` never deletes the old target: it moves it to a neighbouring
    ``.backup`` path before publishing the verified staging directory.
    """

    target = _target_path(output)
    target_exists = target.exists() or target.is_symlink()
    if target_exists and not replace:
        raise HfExportError(
            f"export target already exists and was not touched: {target}; pass replace=True"
        )
    target.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = (
        Path(manifest).expanduser().resolve()
        if manifest is not None
        else s3_ship_manifest()
    )
    router_source = crsa_router_manifest()
    document = _load_json(manifest_path, label="SHIP-v6 manifest")
    if document.get("schema") != _MANIFEST_SCHEMA:
        raise HfExportError(f"unsupported SHIP-v6 manifest schema in {manifest_path}")
    s3 = document.get("s3")
    if not isinstance(s3, dict) or not isinstance(s3.get("host"), dict):
        raise HfExportError("SHIP-v6 manifest has no host descriptor")
    host_descriptor = s3["host"]
    host_spec, organ_specs = _verified_specs(manifest_path, artifact_root)
    router_digest = _router_digest(router_source)

    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.staging.",
            dir=target.parent,
        )
    )
    published = False
    backup: Path | None = None
    try:
        state = _a1_state(host_spec.destination, host_descriptor)
        model_path = stage / "model.safetensors"
        _save_a1_safetensors(model_path, state)
        model_sha256 = sha256_file(model_path)

        rebased = json.loads(json.dumps(document))
        rebased_host = rebased["s3"]["host"]
        rebased_host["artifact"] = "model.safetensors"
        rebased_host["format"] = "safetensors-state-dict"
        rebased_host["sha256"] = model_sha256
        rebased_host["source_checkpoint_sha256"] = host_spec.sha256

        organ_by_name = {
            spec.label.removeprefix("organ:"): spec for spec in organ_specs
        }
        for descriptor in rebased["organs"]:
            if not isinstance(descriptor, dict):
                raise HfExportError("SHIP-v6 organ descriptor is not an object")
            name = str(descriptor.get("name", ""))
            spec = organ_by_name.get(name)
            if spec is None:
                raise HfExportError(f"no verified source artifact for organ {name!r}")
            output_name = spec.destination.name
            destination = stage / "organs" / output_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(spec.destination, destination)
            if sha256_file(destination) != spec.sha256:
                raise HfExportError(f"copied organ bytes changed for {name}")
            descriptor["artifact"] = f"organs/{output_name}"
            descriptor["sha256"] = spec.sha256

        _write_bytes(stage / "s3_ship_v6.json", _json_bytes(rebased))
        shutil.copyfile(router_source, stage / "crsa_router_v1.json")

        model_config = host_descriptor.get("model")
        if not isinstance(model_config, Mapping):
            raise HfExportError("SHIP-v6 host has no model configuration")
        config = {
            "architectures": ["SelectiveNoPETransformerLM"],
            "artifact_schema": _BUNDLE_SCHEMA,
            "attention_program": "2 Local + 1 Balanced + 1 bit-exact Free",
            "canonical_cases": 152,
            "host": dict(model_config),
            "host_arm": "A1",
            "license_status": "NOT_CLEARED",
            "model_type": "immer-o1-state",
            "no_runtime_training": True,
            "public_upload_allowed": False,
            "router_state_digest": router_digest,
            "torch_dtype": "float32",
        }
        _write_bytes(stage / "config.json", _json_bytes(config))
        _write_bytes(
            stage / "README.md",
            _model_card(
                source_host_sha256=host_spec.sha256,
                model_sha256=model_sha256,
                router_digest=router_digest,
                organs=organ_specs,
            ).encode("utf-8"),
        )
        _write_bytes(
            stage / "requirements.txt",
            b"numpy>=1.26\nsafetensors>=0.4\ntorch>=2.5\n",
        )
        _write_bytes(
            stage / ".gitattributes",
            (
                b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
                b"*.pt filter=lfs diff=lfs merge=lfs -text\n"
            ),
        )
        _write_bytes(stage / "verify.py", textwrap.dedent(_VERIFY_SCRIPT).encode("utf-8"))
        _write_bytes(stage / "solve.py", textwrap.dedent(_SOLVE_SCRIPT).encode("utf-8"))
        packaged_runtime = stage / "runtime" / "immer"
        _copy_runtime(Path(__file__).resolve().parent, packaged_runtime)
        # ``attention.router`` validates its package-resource default at
        # import time even when callers inject an explicit router path.  Match
        # the installed-wheel layout so the copied package is independently
        # importable, while keeping the root copies convenient for HF users.
        resources = packaged_runtime / "resources"
        resources.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stage / "s3_ship_v6.json", resources / "s3_ship_v6.json")
        shutil.copyfile(
            stage / "crsa_router_v1.json",
            resources / "crsa_router_v1.json",
        )

        files = _payload_digests(stage)
        checksums = {
            "expectations": {
                "canonical_cases": 152,
                "model_sha256": model_sha256,
                "router_state_digest": router_digest,
                "source_checkpoint_sha256": host_spec.sha256,
            },
            "files": files,
            "schema": _CHECKSUM_SCHEMA,
        }
        _write_bytes(stage / "checksums.json", _json_bytes(checksums))
        verification = _isolated_verify(stage)

        # Recheck the race boundary after the potentially expensive benchmark.
        target_exists = target.exists() or target.is_symlink()
        if target_exists:
            if not replace:
                raise HfExportError(
                    f"export target appeared during build and was not touched: {target}"
                )
            backup = _next_backup(target)
            os.replace(target, backup)
        try:
            os.replace(stage, target)
            published = True
        except Exception:
            if backup is not None and not target.exists() and not target.is_symlink():
                os.replace(backup, target)
                backup = None
            raise

        return {
            "backup": str(backup) if backup is not None else None,
            "files": len(files) + 1,
            "license": "other",
            "license_status": "NOT_CLEARED",
            "model_bytes": (target / "model.safetensors").stat().st_size,
            "model_sha256": model_sha256,
            "output": str(target),
            "public_upload_allowed": False,
            "router_state_digest": router_digest,
            "source_checkpoint_bytes": host_spec.destination.stat().st_size,
            "source_checkpoint_sha256": host_spec.sha256,
            "status": "exported",
            "verification": verification,
        }
    except HfExportError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise HfExportError(f"HF PoC export failed: {exc}") from exc
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)


__all__ = ["HfExportError", "export_hf_poc"]
