#!/usr/bin/env python3
"""Redact machine locations without changing measurements or recorded provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PREFETCH_REPORTS = (
    ROOT / "results" / "deepseek-v4-exact-prefetch-smoke.json",
    ROOT / "results" / "deepseek-v4-exact-prefetch-window-smoke.json",
    ROOT / "results" / "deepseek-v4-exact-prefetch-window-network-smoke.json",
    ROOT / "results" / "deepseek-v4-adjacent-range-warm-smoke.json",
    ROOT / "results" / "deepseek-v4-adjacent-range-network-smoke.json",
)
HEAD_REPORT = ROOT / "results" / "deepseek-v4-head-range-network-smoke.json"
QWEN_REPORT = ROOT / "results" / "qwen38_reference_baseline.json"


def _canonical(document: Any, *, ensure_ascii: bool) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=ensure_ascii,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _public_path(value: str) -> str:
    resolved = Path(value).expanduser().resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return "<external>"


def _redact_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact_paths(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_redact_paths(nested) for nested in value]
    if isinstance(value, str) and value.startswith("/"):
        return _public_path(value)
    return value


def _seal(document: dict[str, Any], *, ensure_ascii: bool) -> dict[str, Any]:
    result = dict(document)
    result.pop("report_sha256", None)
    result["report_sha256"] = _canonical(result, ensure_ascii=ensure_ascii)
    return result


def _sanitize_prefetch(document: dict[str, Any]) -> dict[str, Any]:
    return _seal(_redact_paths(document), ensure_ascii=True)


def _sanitize_head(document: dict[str, Any]) -> dict[str, Any]:
    return _seal(_redact_paths(document), ensure_ascii=False)


def _sanitize_qwen(document: dict[str, Any]) -> dict[str, Any]:
    result = _redact_paths(document)
    model = result.get("model")
    if isinstance(model, dict):
        raw = str(model.get("path", ""))
        if raw.startswith(("http://127.0.0.1", "http://localhost")):
            model["path"] = "<loopback-openai-compatible>"
        elif raw.startswith(("http://", "https://")):
            model["path"] = "<external-openai-compatible>"
        elif raw == "<external>":
            model["path"] = "<external-local-checkpoint>"
    return _seal(result, ensure_ascii=False)


def _write(path: Path, document: dict[str, Any]) -> None:
    payload = (
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="replace reports atomically")
    args = parser.parse_args()
    jobs = [
        *((path, _sanitize_prefetch) for path in PREFETCH_REPORTS),
        (HEAD_REPORT, _sanitize_head),
        (QWEN_REPORT, _sanitize_qwen),
    ]
    changed: list[str] = []
    for path, sanitizer in jobs:
        original = json.loads(path.read_text(encoding="utf-8"))
        public = sanitizer(original)
        if public != original:
            changed.append(path.relative_to(ROOT).as_posix())
            if args.write:
                _write(path, public)
    print(json.dumps({"changed": changed, "written": bool(args.write)}, sort_keys=True))
    return 0 if args.write or not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
