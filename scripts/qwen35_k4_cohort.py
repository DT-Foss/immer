#!/usr/bin/env python3
"""Run the sealed four-prompt Qwen3.5 K4 mechanism cohort.

The cohort compares the exact active control (two K2 verification blocks) with
one K4 verification block.  Selection is question-only and is completed and
sealed before either model is opened.  Execution keeps one authenticated
target mount and one authenticated drafter mount alive for the complete ABBA
schedule; every arm starts from released continuation state.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from types import ModuleType
from typing import Any
import unicodedata

from immer.runtimes.qwen3_8 import (
    K2SpeculativeGenerationEvidence,
    K2SpeculativeRoundEvidence,
    K4SpeculativeGenerationEvidence,
    K4SpeculativeRoundEvidence,
    LogicalModelIdentity,
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    QWEN35_DRAFTER_REPO_ID,
    QWEN35_DRAFTER_REVISION,
    Qwen35K2DraftProvider,
    Qwen35K4DraftProvider,
    Qwen38K2SpeculativeDecoder,
    Qwen38K4SpeculativeDecoder,
    Qwen38Tokenizer,
    Qwen38WeightPager,
)


ROOT = Path(__file__).resolve().parent.parent
K2_SCRIPT = ROOT / "scripts" / "qwen35_live_k2_smoke.py"
K4_SCRIPT = ROOT / "scripts" / "qwen35_live_k4_smoke.py"
DEFAULT_DEV64 = (
    ROOT
    / "artifacts"
    / "private"
    / "qwen3.8-fertig-local"
    / "post-cert-64"
    / "inputs.json"
)
DEFAULT_TARGET = Path("/app/models/Qwen3.8-27B")
DEFAULT_DRAFT = Path("/app/models/Qwen3.5-0.8B")
DEFAULT_INPUT = (
    ROOT
    / "artifacts"
    / "private"
    / "qwen3.8-fertig-local"
    / "post-cert-64"
    / "qwen35-k4-cohort"
    / "input.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent / "session-v1"

INPUT_SCHEMA = "immer.qwen3.5-k4-cohort-input/v1"
MOUNT_SCHEMA = "immer.qwen3.5-k4-cohort-mount/v1"
ARM_SCHEMA = "immer.qwen3.5-k4-cohort-arm/v1"
RESULT_SCHEMA = "immer.qwen3.5-k4-cohort-result/v1"
ABORT_SCHEMA = "immer.qwen3.5-k4-cohort-abort/v1"
TRANSPORT_SCHEMA = "immer.qwen3.5-k4-cohort-transport/v1"
SELECTION_DOMAIN = "qwen35-k4-cohort/v1\0"
EXECUTION_REGIME = "post-authentication-process-warm/v1"
MIB = 1024**2

_STRATA = ("short", "medium-short", "medium-long", "long")
_EXECUTION_BINS = (0, 3, 1, 2)
_SCHEDULE = (("A", "B"), ("B", "A"), ("B", "A"), ("A", "B"))
_ROUTE = {"A": "2xK2", "B": "K4"}
_FROZEN = (
    ("short", 62, "gsm8k-test-0931-7a0aa7b235d5cac2", 62),
    ("long", 38, "gsm8k-test-0823-c0762056be2ea3af", 111),
    ("medium-short", 43, "gsm8k-test-0836-02361a80c7142afd", 74),
    ("medium-long", 50, "gsm8k-test-0868-cbef88cef5682484", 97),
)
_FROZEN_SOURCE = {
    "item_count": 64,
    "question_projection_sha256": "93761a1b59a06d2eba66f0b11dc89e6eb7e4f032858a7c2644d67ae2780ed8bd",
    "raw_file_sha256": "05bd0ac1b2da34d36297125190f8cbf8f4d3819fc752d0f8679c7a5832d7c3bb",
    "raw_file_size_bytes": 143_592,
}
_FROZEN_TOKENIZER = {
    "kind": "local-tokenizers-json/v1",
    "sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "size_bytes": 12_809_320,
    "vocabulary_sha256": "5d326ac57a518b9102ed33be576eea8e2ab54c2f328ac6d424ffef4c2c53ef0a",
    "vocabulary_size": 248_077,
}
_FROZEN_ITEM_DIGESTS = (
    (
        "235b0fce2205a253452300bd06b0d24a213edcb1bf38722e6d88dad77b5a30ed",
        "779704492a06cd8992d67b459529f4b3d9809f84dc475ae58ab3d10ef86da62d",
    ),
    (
        "0027358a16a084a516c7e1437c9a489003311345f34314eadffbb162fd7bc33e",
        "8a16a0a44202fd45ddc42d6bd8692d8f9c0d1aee3654985d02ec023f39b42b27",
    ),
    (
        "00e89ab1c5dad163e7b47abefa29c3d5a73761f04496469ea8282c08b31e0dba",
        "ffe8e8ee61bc96a0925bdb3750b7709ed6bce4da7dec7a12c6371492ed648657",
    ),
    (
        "06c747c0fa506a6c1665a68f72e5462b7a919edd9e29753c185f3f4f1a139ab2",
        "6ff3deab1c2c131118187ad8761726cab3bb8d81c1cee8c23833c373cdf1c5e1",
    ),
)


def _load_script(name: str, path: Path) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_script("qwen35_live_k2_smoke", K2_SCRIPT)
k4 = _load_script("qwen35_live_k4_smoke", K4_SCRIPT)


class CohortError(RuntimeError):
    """The frozen K4 cohort contract cannot be completed."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CohortError("document is not canonical JSON") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(value)
    if "sha256" in document:
        raise CohortError("document already contains a seal")
    document["sha256"] = _sha256(document)
    return document


def _verify_seal(value: object, schema: str, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != schema:
        raise CohortError(f"{label} schema is invalid")
    document = dict(value)
    seal = document.pop("sha256", None)
    if not base._is_sha256(seal) or seal != _sha256(document):
        raise CohortError(f"{label} seal is invalid")
    document["sha256"] = seal
    return document


def _file_sha256(path: Path, label: str) -> tuple[str, int]:
    return base._file_sha256(path, label)


def _atomic_new_json(path: Path, document: Mapping[str, Any]) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = _canonical(dict(document)) + b"\n"
    descriptor = -1
    temporary = ""
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".pending",
            dir=destination.parent,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as exc:
        raise CohortError(f"refusing to overwrite output: {destination}") from exc
    except OSError as exc:
        raise CohortError(f"cannot write output: {destination}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return destination


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _file_sha256(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CohortError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CohortError(f"{label} must be a JSON object")
    return value


def _normal_question(value: object) -> str:
    if not isinstance(value, str):
        raise CohortError("Dev64 question must be text")
    result = unicodedata.normalize("NFC", value).strip()
    if not result:
        raise CohortError("Dev64 question must not be empty")
    return result


def _selection_hash(question: str) -> str:
    return hashlib.sha256((SELECTION_DOMAIN + question).encode("utf-8")).hexdigest()


def _vocabulary_sha256(tokenizer: Qwen38Tokenizer) -> str:
    vocabulary = tokenizer.backend.get_vocab(with_added_tokens=True)
    rows = sorted((token, int(token_id)) for token, token_id in vocabulary.items())
    return _sha256(rows)


def _tokenizer_receipt(
    path: Path, *, require_official: bool
) -> tuple[Qwen38Tokenizer, dict[str, Any]]:
    digest, size = _file_sha256(path, "local tokenizer")
    tokenizer = Qwen38Tokenizer(path, require_official=require_official)
    receipt = {
        "kind": "local-tokenizers-json/v1",
        "sha256": digest,
        "size_bytes": size,
        "vocabulary_sha256": _vocabulary_sha256(tokenizer),
        "vocabulary_size": int(
            tokenizer.backend.get_vocab_size(with_added_tokens=True)
        ),
    }
    if _file_sha256(path, "local tokenizer") != (digest, size):
        raise CohortError("local tokenizer changed while authenticating")
    return tokenizer, receipt


def _select_cohort(
    items: Sequence[object], tokenizer: Qwen38Tokenizer
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(items) != 64:
        raise CohortError("canonical Dev64 source must contain exactly 64 items")
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for offset, raw in enumerate(items):
        if not isinstance(raw, Mapping):
            raise CohortError("Dev64 item must be an object")
        item_id = raw.get("item_id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise CohortError("Dev64 item ID is invalid")
        question = _normal_question(raw.get("question"))
        if item_id in seen_ids or question in seen_questions:
            raise CohortError("Dev64 item IDs and questions must be unique")
        seen_ids.add(item_id)
        seen_questions.add(question)
        rendered = tokenizer.render_no_thinking_prompt("", question)
        token_ids = tuple(tokenizer.encode(rendered))
        if not token_ids:
            raise CohortError("rendered Dev64 prompt is empty")
        candidates.append(
            {
                "item_id": item_id,
                "normalized_question": question,
                "normalized_question_sha256": hashlib.sha256(
                    question.encode("utf-8")
                ).hexdigest(),
                "offset": offset,
                "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                "rendered_token_count": len(token_ids),
                "rendered_token_ids": list(token_ids),
                "selection_sha256": _selection_hash(question),
            }
        )
    ordered = sorted(
        candidates,
        key=lambda row: (row["rendered_token_count"], row["selection_sha256"]),
    )
    bins: list[list[dict[str, Any]]] = [
        ordered[index : index + 16] for index in range(0, 64, 16)
    ]
    strata: list[dict[str, Any]] = []
    selected_by_bin: list[dict[str, Any]] = []
    for index, rows in enumerate(bins):
        selected = min(rows, key=lambda row: row["selection_sha256"])
        selected_by_bin.append(selected)
        strata.append(
            {
                "bin_index": index,
                "label": _STRATA[index],
                "rank_end_exclusive": (index + 1) * 16,
                "rank_start": index * 16,
                "selected_offset": selected["offset"],
                "token_count_max": max(row["rendered_token_count"] for row in rows),
                "token_count_min": min(row["rendered_token_count"] for row in rows),
            }
        )
    execution: list[dict[str, Any]] = []
    for execution_index, bin_index in enumerate(_EXECUTION_BINS):
        selected = dict(selected_by_bin[bin_index])
        selected.update(
            {
                "execution_index": execution_index,
                "schedule": list(_SCHEDULE[execution_index]),
                "stratum": _STRATA[bin_index],
                "stratum_bin_index": bin_index,
            }
        )
        execution.append(selected)
    return execution, strata


def _git_revision(*, require_clean: bool) -> str:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CohortError("cannot authenticate repository revision") from exc
    if len(revision) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise CohortError("repository revision is invalid")
    if require_clean and dirty:
        raise CohortError("cohort execution requires a clean committed revision")
    return revision


def prepare_input(
    *,
    dev64_path: Path,
    tokenizer_path: Path,
    output_path: Path,
    require_official: bool = True,
    require_clean: bool = True,
) -> tuple[dict[str, Any], Path]:
    _assert_no_holdout_paths((dev64_path, tokenizer_path, output_path))
    source_sha, source_size = _file_sha256(dev64_path, "canonical Dev64 source")
    source = _read_json(dev64_path, "canonical Dev64 source")
    items = source.get("items")
    if not isinstance(items, list):
        raise CohortError("canonical Dev64 source has no item list")
    tokenizer, tokenizer_receipt = _tokenizer_receipt(
        tokenizer_path, require_official=require_official
    )
    selected, strata = _select_cohort(items, tokenizer)
    question_source = [
        {
            "item_id": str(row.get("item_id")),
            "offset": index,
            "question": _normal_question(row.get("question")),
        }
        for index, row in enumerate(items)
        if isinstance(row, Mapping)
    ]
    if len(question_source) != 64:
        raise CohortError("canonical Dev64 question projection is incomplete")
    document = _seal(
        {
            "code_revision": _git_revision(require_clean=require_clean),
            "contract": {
                "attention_mode": "native-crsa",
                "batch_size": 1,
                "compute_dtype": "bfloat16",
                "device": "cpu",
                "eos_enabled": False,
                "execution_regime": EXECUTION_REGIME,
                "max_new_tokens": 4,
                "remote_io": False,
                "routes": {"A": "2xK2", "B": "K4"},
                "system_prompt": "",
            },
            "holdout_accessed": False,
            "items": selected,
            "path_allowlist": [
                str(dev64_path.resolve()),
                str(tokenizer_path.resolve()),
            ],
            "schema": INPUT_SCHEMA,
            "selection": {
                "bin_size": 16,
                "domain": SELECTION_DOMAIN,
                "execution_bins": list(_EXECUTION_BINS),
                "method": "NFC+trim/render/sort(token_count,selection_sha256)/four-bins/min-sha256/v1",
                "schedule": [list(row) for row in _SCHEDULE],
                "strata": strata,
            },
            "source": {
                "item_count": 64,
                "question_projection_sha256": _sha256(question_source),
                "raw_file_sha256": source_sha,
                "raw_file_size_bytes": source_size,
            },
            "tokenizer": tokenizer_receipt,
        }
    )
    _validate_input(document, require_frozen=require_official)
    path = _atomic_new_json(output_path, document)
    return document, path


def _validate_input(value: object, *, require_frozen: bool = True) -> dict[str, Any]:
    document = _verify_seal(value, INPUT_SCHEMA, "cohort input")
    required = {
        "code_revision",
        "contract",
        "holdout_accessed",
        "items",
        "path_allowlist",
        "schema",
        "selection",
        "sha256",
        "source",
        "tokenizer",
    }
    if set(document) != required:
        raise CohortError("cohort input fields are invalid")
    revision = document.get("code_revision")
    if (
        not isinstance(revision, str)
        or len(revision) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise CohortError("cohort input code revision is invalid")
    expected_contract = {
        "attention_mode": "native-crsa",
        "batch_size": 1,
        "compute_dtype": "bfloat16",
        "device": "cpu",
        "eos_enabled": False,
        "execution_regime": EXECUTION_REGIME,
        "max_new_tokens": 4,
        "remote_io": False,
        "routes": {"A": "2xK2", "B": "K4"},
        "system_prompt": "",
    }
    if document.get("contract") != expected_contract:
        raise CohortError("cohort input execution contract is invalid")
    if document.get("holdout_accessed") is not False:
        raise CohortError("cohort input accessed Holdout")
    allowlist = document.get("path_allowlist")
    if (
        not isinstance(allowlist, list)
        or len(allowlist) != 2
        or len(set(allowlist)) != 2
        or any(
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or "holdout" in path.casefold()
            for path in allowlist
        )
    ):
        raise CohortError("cohort input path allowlist is invalid")
    tokenizer = document.get("tokenizer")
    if (
        not isinstance(tokenizer, Mapping)
        or set(tokenizer)
        != {
            "kind",
            "sha256",
            "size_bytes",
            "vocabulary_sha256",
            "vocabulary_size",
        }
        or tokenizer.get("kind") != "local-tokenizers-json/v1"
        or not base._is_sha256(tokenizer.get("sha256"))
        or not base._is_sha256(tokenizer.get("vocabulary_sha256"))
        or isinstance(tokenizer.get("size_bytes"), bool)
        or not isinstance(tokenizer.get("size_bytes"), int)
        or int(tokenizer["size_bytes"]) <= 0
        or isinstance(tokenizer.get("vocabulary_size"), bool)
        or not isinstance(tokenizer.get("vocabulary_size"), int)
        or int(tokenizer["vocabulary_size"]) <= 0
    ):
        raise CohortError("cohort input tokenizer receipt is invalid")
    source = document.get("source")
    if (
        not isinstance(source, Mapping)
        or set(source)
        != {
            "item_count",
            "question_projection_sha256",
            "raw_file_sha256",
            "raw_file_size_bytes",
        }
        or source.get("item_count") != 64
        or not base._is_sha256(source.get("question_projection_sha256"))
        or not base._is_sha256(source.get("raw_file_sha256"))
        or isinstance(source.get("raw_file_size_bytes"), bool)
        or not isinstance(source.get("raw_file_size_bytes"), int)
        or int(source["raw_file_size_bytes"]) <= 0
    ):
        raise CohortError("cohort input source receipt is invalid")
    selection = document.get("selection")
    if (
        not isinstance(selection, Mapping)
        or set(selection)
        != {
            "bin_size",
            "domain",
            "execution_bins",
            "method",
            "schedule",
            "strata",
        }
        or selection.get("bin_size") != 16
        or selection.get("domain") != SELECTION_DOMAIN
        or selection.get("execution_bins") != list(_EXECUTION_BINS)
        or selection.get("method")
        != "NFC+trim/render/sort(token_count,selection_sha256)/four-bins/min-sha256/v1"
        or selection.get("schedule") != [list(row) for row in _SCHEDULE]
        or not isinstance(selection.get("strata"), list)
        or len(selection["strata"]) != 4
    ):
        raise CohortError("cohort input selection receipt is invalid")
    for index, stratum in enumerate(selection["strata"]):
        if (
            not isinstance(stratum, Mapping)
            or set(stratum)
            != {
                "bin_index",
                "label",
                "rank_end_exclusive",
                "rank_start",
                "selected_offset",
                "token_count_max",
                "token_count_min",
            }
            or stratum.get("bin_index") != index
            or stratum.get("label") != _STRATA[index]
            or stratum.get("rank_start") != index * 16
            or stratum.get("rank_end_exclusive") != (index + 1) * 16
            or any(
                isinstance(stratum.get(name), bool)
                or not isinstance(stratum.get(name), int)
                for name in ("selected_offset", "token_count_min", "token_count_max")
            )
            or int(stratum["token_count_min"]) <= 0
            or int(stratum["token_count_min"]) > int(stratum["token_count_max"])
        ):
            raise CohortError("cohort input stratum receipt is invalid")
    items = document.get("items")
    if not isinstance(items, list) or len(items) != 4:
        raise CohortError("cohort input must contain four selected prompts")
    offsets: set[int] = set()
    item_ids: set[str] = set()
    for index, row in enumerate(items):
        if not isinstance(row, Mapping) or set(row) != {
            "execution_index",
            "item_id",
            "normalized_question",
            "normalized_question_sha256",
            "offset",
            "rendered_sha256",
            "rendered_token_count",
            "rendered_token_ids",
            "schedule",
            "selection_sha256",
            "stratum",
            "stratum_bin_index",
        }:
            raise CohortError("cohort selected row is invalid")
        question = _normal_question(row.get("normalized_question"))
        item_id = row.get("item_id")
        offset = row.get("offset")
        if (
            question != row.get("normalized_question")
            or row.get("normalized_question_sha256")
            != hashlib.sha256(question.encode("utf-8")).hexdigest()
            or row.get("selection_sha256") != _selection_hash(question)
        ):
            raise CohortError("cohort selection hash is invalid")
        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id in item_ids
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or not 0 <= offset < 64
            or offset in offsets
        ):
            raise CohortError("cohort item identity is invalid")
        item_ids.add(item_id)
        offsets.add(offset)
        if (
            row.get("execution_index") != index
            or row.get("schedule") != list(_SCHEDULE[index])
            or row.get("stratum")
            != ("short", "long", "medium-short", "medium-long")[index]
            or row.get("stratum_bin_index") != _EXECUTION_BINS[index]
        ):
            raise CohortError("cohort execution schedule is invalid")
        tokens = row.get("rendered_token_ids")
        rendered = Qwen38Tokenizer.render_no_thinking_prompt("", question)
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in tokens
            )
            or row.get("rendered_token_count") != len(tokens)
            or row.get("rendered_sha256")
            != hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        ):
            raise CohortError("cohort rendered tokens are invalid")
    if require_frozen:
        actual = tuple(
            (
                row.get("stratum"),
                row.get("offset"),
                row.get("item_id"),
                row.get("rendered_token_count"),
            )
            for row in items
        )
        if actual != _FROZEN:
            raise CohortError(f"frozen cohort changed: {actual!r}")
        digests = tuple(
            (row.get("selection_sha256"), _sha256(row.get("rendered_token_ids")))
            for row in items
        )
        if (
            digests != _FROZEN_ITEM_DIGESTS
            or dict(source) != _FROZEN_SOURCE
            or dict(tokenizer) != _FROZEN_TOKENIZER
        ):
            raise CohortError(
                "frozen cohort source, tokenizer, or prompt digest changed"
            )
    return document


def _assert_no_holdout_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        if "holdout" in str(path.expanduser().resolve()).casefold():
            raise CohortError(f"Holdout path is forbidden: {path}")


def _guard_output_dir(output_dir: Path, protected: Sequence[Path]) -> Path:
    destination = output_dir.expanduser().resolve()
    _assert_no_holdout_paths((destination, *protected))
    for root in protected:
        resolved = root.expanduser().resolve()
        if destination == resolved or destination.is_relative_to(resolved):
            raise CohortError(f"output directory overlaps protected input: {resolved}")
    if destination.exists():
        raise CohortError(f"refusing to reuse output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _stat_receipt(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    return {
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": int(metadata.st_mode),
        "mtime_ns": int(metadata.st_mtime_ns),
        "size_bytes": int(metadata.st_size),
    }


class _SingleMountPair:
    def __init__(self) -> None:
        self.counts = {"target": 0, "draft": 0}
        self.preflight_counts = {"target": 0, "draft": 0}
        self.verification_counts = {"target": 0, "draft": 0}
        self.target: Any = None
        self.draft: Any = None
        self.closed = False

    def open_role(self, role: str, **kwargs: Any) -> Any:
        if role not in self.counts or self.counts[role] != 0:
            raise CohortError(f"{role} model may be authenticated exactly once")
        self.counts[role] += 1
        original_verify = base.verify_qwen38_causal_mount
        original_preflight = base.StreamedQwen38.checkpoint_preflight

        def counted_verify(*args: Any, **inner_kwargs: Any) -> Any:
            self.verification_counts[role] += 1
            if self.verification_counts[role] != 1:
                raise CohortError(f"{role} bundle verification ran more than once")
            return original_verify(*args, **inner_kwargs)

        def counted_preflight(model: Any, *args: Any, **inner_kwargs: Any) -> Any:
            self.preflight_counts[role] += 1
            if self.preflight_counts[role] != 1:
                raise CohortError(f"{role} checkpoint preflight ran more than once")
            return original_preflight(model, *args, **inner_kwargs)

        base.verify_qwen38_causal_mount = counted_verify
        base.StreamedQwen38.checkpoint_preflight = counted_preflight
        try:
            owned = base._open_model(role=role, **kwargs)
        finally:
            base.verify_qwen38_causal_mount = original_verify
            base.StreamedQwen38.checkpoint_preflight = original_preflight
        if self.verification_counts[role] != 1 or self.preflight_counts[role] != 1:
            try:
                owned.close()
            finally:
                raise CohortError(
                    f"{role} requires exactly one verification and one preflight"
                )
        if role == "target":
            self.target = owned
        else:
            self.draft = owned
        return owned

    def close(self) -> None:
        if self.closed:
            return
        failures: list[BaseException] = []
        for owned in (self.target, self.draft):
            if owned is None:
                continue
            try:
                owned.close()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise CohortError(f"mount cleanup failed: {failures[0]}") from failures[0]
        self.closed = True


class _TimedProvider:
    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.proposal_seconds = 0.0
        self.reconciliation_seconds = 0.0
        self.proposals: list[list[int]] = []
        self.reconciled_history_lengths: list[int] = []

    def __call__(self, history: tuple[int, ...], /) -> tuple[int, ...]:
        started = time.perf_counter()
        try:
            proposal = tuple(self.provider(history))
        finally:
            self.proposal_seconds += time.perf_counter() - started
        self.proposals.append(list(proposal))
        return proposal

    def reconcile(self, history: tuple[int, ...], /) -> None:
        started = time.perf_counter()
        try:
            self.provider.reconcile(history)
        finally:
            self.reconciliation_seconds += time.perf_counter() - started
        self.reconciled_history_lengths.append(len(history))


def _assert_empty(model: Any, label: str, *, reset: bool = False) -> None:
    if reset:
        model.reset_state(release=True)
    if (
        model.next_position != 0
        or model.state_batch_size is not None
        or model.state_bytes != 0
        or model.state_poisoned
        or model._pending_block_stage is not None
    ):
        raise CohortError(f"{label} model did not reach clean released state")


def _acceptance(rounds: Sequence[Mapping[str, Any]], k_value: int) -> dict[str, Any]:
    proposed = [row for row in rounds if row.get("proposed_token_ids")]
    prefix = Counter(int(row["accepted_prefix_length"]) for row in proposed)
    replay = Counter(str(row["replay_kind"]) for row in rounds)
    mismatch = Counter(
        str(int(row["accepted_prefix_length"]))
        for row in proposed
        if int(row["accepted_prefix_length"]) < k_value
    )
    proposed_tokens = sum(len(row["proposed_token_ids"]) for row in proposed)
    accepted_tokens = sum(int(row["accepted_prefix_length"]) for row in proposed)
    return {
        "accepted_draft_tokens": accepted_tokens,
        "accepted_prefix_histogram": {
            str(index): int(prefix[index]) for index in range(k_value + 1)
        },
        "complete_blocks": int(prefix[k_value]),
        "draft_tokens_proposed": proposed_tokens,
        "micro_acceptance": 0.0
        if proposed_tokens == 0
        else accepted_tokens / proposed_tokens,
        "mismatch_position_histogram": dict(sorted(mismatch.items())),
        "proposal_rounds": len(proposed),
        "replay_histogram": dict(sorted(replay.items())),
        "restage_count": sum("restage" in str(row["replay_kind"]) for row in rounds),
        "decode_fallback_count": sum(
            "decode" in str(row["replay_kind"]) for row in rounds
        ),
        "terminal_single_rounds": len(rounds) - len(proposed),
    }


def _state_tensor_index(state: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for layer in state.get("layers", []):
        if not isinstance(layer, Mapping):
            continue
        index = int(layer["layer"])
        for name, tensor in layer.get("tensors", {}).items():
            if isinstance(tensor, Mapping):
                result[f"layer.{index}.{name}"] = str(tensor.get("sha256"))
    graft = state.get("graft_history")
    if isinstance(graft, Mapping):
        result["graft_history"] = str(graft.get("sha256"))
    return result


def _generation_positions(
    transactions: Mapping[str, Any], native_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return k4._committed_position_rows(transactions, native_rows)


def _arm_filename(index: int, item: Mapping[str, Any], route: str) -> str:
    slug = str(item["stratum"]).replace("-", "_")
    return f"arm-{index + 1:02d}-{slug}-{route.lower()}.json"


def _run_arm(
    *,
    arm_index: int,
    route: str,
    item: Mapping[str, Any],
    input_sha256: str,
    mount_sha256: str,
    target: Any,
    draft: Any,
    native_evidence: list[Any],
    head_block_rows: int,
) -> dict[str, Any]:
    if route not in _ROUTE:
        raise CohortError("arm route is invalid")
    _assert_empty(target.model, "target", reset=True)
    _assert_empty(draft.model, "draft", reset=True)
    native_evidence.clear()
    gc_started = time.perf_counter()
    collected = gc.collect()
    gc_seconds = time.perf_counter() - gc_started
    gc_before = list(gc.get_count())
    stats_before = [dict(row) for row in gc.get_stats()]
    prompt = tuple(int(token) for token in item["rendered_token_ids"])
    k_value = 2 if route == "A" else 4
    provider_cls = Qwen35K2DraftProvider if route == "A" else Qwen35K4DraftProvider
    decoder_cls = (
        Qwen38K2SpeculativeDecoder if route == "A" else Qwen38K4SpeculativeDecoder
    )
    provider = provider_cls(
        draft.model, eos_token_ids=(), head_block_rows=head_block_rows
    )
    timed = _TimedProvider(provider)
    recorder = k4._TargetTransactionRecorder(target.model)
    target_before = base._pager_counters(target.pager)
    draft_before = base._pager_counters(draft.pager)
    started = time.perf_counter()
    try:
        generated = decoder_cls(target.model, timed).generate(
            [prompt],
            max_new_tokens=4,
            eos_token_ids=(),
            head_block_rows=head_block_rows,
        )
        arm_wall = time.perf_counter() - started
        target_after = base._pager_counters(target.pager)
        draft_after = base._pager_counters(draft.pager)
        rounds = [row.to_dict() for row in generated.evidence.rounds]
        native_rows = [row.to_dict() for row in native_evidence]
        transactions = recorder.receipt()
        positions = _generation_positions(transactions, native_rows)
        target_state = k4._committed_state(target.model)
        draft_state = k4._committed_state(draft.model)
        history = provider.committed_history
        full_history = (*prompt, *generated.token_ids)
        if (
            history is None
            or len(history) < len(prompt)
            or history[: len(prompt)] != prompt
            or tuple(full_history[: len(history)]) != history
        ):
            raise CohortError("drafter committed history is not a target prefix")
        suffix = tuple(full_history[len(history) :])
        if len(suffix) > k_value - 1:
            raise CohortError("drafter target-only suffix exceeds route tail")
        if (
            len(generated.token_ids) != 4
            or target.model.next_position != len(prompt) + 4
        ):
            raise CohortError("arm did not commit exactly four output tokens")
        expected_positions = list(range(len(prompt), len(prompt) + 4))
        if [row["position"] for row in positions] != expected_positions:
            raise CohortError("arm per-position hidden trace is incomplete")
        provider_metrics = provider.metrics().to_dict()
        target_counters = base._counter_delta(target_before, target_after)
        draft_counters = base._counter_delta(draft_before, draft_after)
        provider_seconds = timed.proposal_seconds + timed.reconciliation_seconds
        guard_seconds = float(generated.evidence.provider_guard_seconds)
        target_only = max(
            0.0, float(generated.evidence.seconds) - provider_seconds - guard_seconds
        )
        gc_after = list(gc.get_count())
        stats_after = [dict(row) for row in gc.get_stats()]
        document = _seal(
            {
                "acceptance": _acceptance(rounds, k_value),
                "arm_index": arm_index,
                "contract": {
                    "attention_mode": "native-crsa",
                    "batch_size": 1,
                    "compute_dtype": "bfloat16",
                    "device": "cpu",
                    "eos_token_ids": [],
                    "execution_regime": EXECUTION_REGIME,
                    "k": k_value,
                    "max_new_tokens": 4,
                    "route": _ROUTE[route],
                    "route_id": route,
                },
                "cost": {
                    "combined_logical_bytes": int(generated.evidence.source_body_bytes)
                    + int(provider_metrics["source_body_bytes"]),
                    "draft_logical_bytes": int(provider_metrics["source_body_bytes"]),
                    "draft_pager": draft_counters,
                    "provider_guard_bytes": int(
                        generated.evidence.provider_guard_bytes
                    ),
                    "target_logical_bytes": int(generated.evidence.source_body_bytes),
                    "target_pager": target_counters,
                },
                "drafter": {
                    "alignment": {
                        "committed_cursor": len(history),
                        "committed_history_sha256": _sha256(list(history)),
                        "target_cursor": len(full_history),
                        "target_only_suffix_ids": list(suffix),
                    },
                    "provider": provider_metrics,
                    "state": draft_state,
                    "state_tensor_sha256": _state_tensor_index(draft_state),
                    "trace": {
                        "proposals": timed.proposals,
                        "reconciled_history_lengths": timed.reconciled_history_lengths,
                    },
                },
                "gc": {
                    "collected_before_arm": int(collected),
                    "collection_seconds": gc_seconds,
                    "count_after": gc_after,
                    "count_before": gc_before,
                    "stats_after": stats_after,
                    "stats_before": stats_before,
                },
                "input_sha256": input_sha256,
                "mount_sha256": mount_sha256,
                "prompt": {
                    "execution_index": int(item["execution_index"]),
                    "item_id": item["item_id"],
                    "offset": int(item["offset"]),
                    "rendered_token_count": len(prompt),
                    "rendered_token_ids_sha256": _sha256(list(prompt)),
                    "stratum": item["stratum"],
                },
                "schema": ARM_SCHEMA,
                "target": {
                    "cursor": target.model.next_position,
                    "native_crsa": {
                        "events": native_rows,
                        "sha256": _sha256(native_rows),
                    },
                    "positions": positions,
                    "speculative_evidence": generated.evidence.to_dict(),
                    "state": target_state,
                    "state_tensor_sha256": _state_tensor_index(target_state),
                    "transactions": transactions,
                },
                "timing": {
                    "arm_wall_seconds": arm_wall,
                    "provider_guard_seconds": guard_seconds,
                    "proposal_seconds": timed.proposal_seconds,
                    "reconciliation_seconds": timed.reconciliation_seconds,
                    "target_generation_seconds_excluding_provider_and_guard": target_only,
                },
                "tokens": {
                    "generated_token_ids": list(generated.token_ids),
                    "token_chain_sha256": _sha256(
                        {"generated": list(generated.token_ids), "prompt": list(prompt)}
                    ),
                },
            }
        )
        _validate_arm(document)
        return document
    finally:
        recorder.close()
        provider.close()


def _validate_arm(value: object) -> dict[str, Any]:
    document = _verify_seal(value, ARM_SCHEMA, "cohort arm")
    contract = document.get("contract")
    target = document.get("target")
    drafter = document.get("drafter")
    tokens = document.get("tokens")
    if not all(isinstance(row, Mapping) for row in (contract, target, drafter, tokens)):
        raise CohortError("cohort arm sections are invalid")
    assert isinstance(contract, Mapping) and isinstance(target, Mapping)
    assert isinstance(drafter, Mapping) and isinstance(tokens, Mapping)
    generated = tokens.get("generated_token_ids")
    if (
        contract.get("max_new_tokens") != 4
        or contract.get("eos_token_ids") != []
        or contract.get("attention_mode") != "native-crsa"
        or contract.get("route_id") not in _ROUTE
        or not isinstance(generated, list)
        or len(generated) != 4
    ):
        raise CohortError("cohort arm contract is invalid")
    k_value = int(contract["k"])
    speculative_raw = target.get("speculative_evidence")
    if not isinstance(speculative_raw, Mapping):
        raise CohortError("cohort arm speculative evidence is absent")
    evidence_values = dict(speculative_raw)
    try:
        evidence_values["prompt_token_ids"] = tuple(evidence_values["prompt_token_ids"])
        evidence_values["generated_token_ids"] = tuple(
            evidence_values["generated_token_ids"]
        )
        if k_value == 2:
            evidence_values["rounds"] = tuple(
                K2SpeculativeRoundEvidence(
                    **{
                        **row,
                        "proposed_token_ids": tuple(row["proposed_token_ids"]),
                        "target_token_ids": tuple(row["target_token_ids"]),
                        "emitted_token_ids": tuple(row["emitted_token_ids"]),
                    }
                )
                for row in evidence_values["rounds"]
            )
            evidence = K2SpeculativeGenerationEvidence(**evidence_values)
        elif k_value == 4:
            evidence_values["eos_token_ids"] = tuple(evidence_values["eos_token_ids"])
            evidence_values["rounds"] = tuple(
                K4SpeculativeRoundEvidence(
                    **{
                        **row,
                        "eos_token_ids": tuple(row["eos_token_ids"]),
                        "proposed_token_ids": tuple(row["proposed_token_ids"]),
                        "target_token_ids": tuple(row["target_token_ids"]),
                        "emitted_token_ids": tuple(row["emitted_token_ids"]),
                    }
                )
                for row in evidence_values["rounds"]
            )
            evidence = K4SpeculativeGenerationEvidence(**evidence_values)
        else:
            raise CohortError("cohort arm K is invalid")
    except (KeyError, TypeError, ValueError) as exc:
        raise CohortError("cohort arm speculative evidence is invalid") from exc
    if evidence.to_dict() != dict(speculative_raw):
        raise CohortError("cohort arm speculative evidence is non-canonical")
    prompt_ids = list(evidence.prompt_token_ids)
    if (
        list(evidence.generated_token_ids) != generated
        or (k_value == 4 and list(evidence.eos_token_ids) != [])
        or evidence.stopped_on_eos
        or document.get("acceptance")
        != _acceptance(evidence.to_dict()["rounds"], k_value)
    ):
        raise CohortError("cohort arm speculative cross-link is invalid")
    state = target.get("state")
    positions = target.get("positions")
    if (
        not isinstance(state, Mapping)
        or not isinstance(positions, list)
        or len(positions) != 4
    ):
        raise CohortError("cohort arm target receipt is incomplete")
    k4._validate_state_receipt(
        state,
        "cohort target",
        cursor=len(prompt_ids) + len(generated),
        pending=False,
    )
    if state.get("pending_block") is not False or state.get("poisoned") is not False:
        raise CohortError("cohort arm target state is not committed")
    if document.get("target", {}).get("state_tensor_sha256") != _state_tensor_index(
        state
    ):
        raise CohortError("cohort arm target tensor index is invalid")
    draft_state = drafter.get("state")
    alignment = drafter.get("alignment")
    if not isinstance(draft_state, Mapping) or not isinstance(alignment, Mapping):
        raise CohortError("cohort arm drafter receipt is incomplete")
    suffix = alignment.get("target_only_suffix_ids")
    prompt = document.get("prompt")
    prompt_count = (
        prompt.get("rendered_token_count") if isinstance(prompt, Mapping) else None
    )
    if (
        isinstance(prompt_count, bool)
        or not isinstance(prompt_count, int)
        or prompt_count <= 0
        or alignment.get("committed_cursor", -1) < prompt_count
        or not isinstance(suffix, list)
        or len(suffix) > int(contract["k"]) - 1
    ):
        raise CohortError("cohort arm drafter suffix is invalid")
    committed_cursor = int(alignment["committed_cursor"])
    target_cursor = int(alignment.get("target_cursor", -1))
    if target_cursor != len(prompt_ids) + len(generated):
        raise CohortError("cohort arm target/drafter cursor cross-link is invalid")
    k4._validate_state_receipt(
        draft_state,
        "cohort drafter",
        cursor=committed_cursor,
        pending=False,
    )
    full_history = [*prompt_ids, *generated]
    committed_history = full_history[:committed_cursor]
    if (
        prompt_count != len(prompt_ids)
        or prompt.get("rendered_token_ids_sha256") != _sha256(prompt_ids)
        or suffix != full_history[committed_cursor:]
        or alignment.get("committed_history_sha256") != _sha256(committed_history)
        or committed_history[: len(prompt_ids)] != prompt_ids
    ):
        raise CohortError("cohort arm drafter history cross-link is invalid")
    if document.get("drafter", {}).get("state_tensor_sha256") != _state_tensor_index(
        draft_state
    ):
        raise CohortError("cohort arm drafter tensor index is invalid")
    native = target.get("native_crsa")
    if (
        not isinstance(native, Mapping)
        or set(native) != {"events", "sha256"}
        or not isinstance(native.get("events"), list)
        or native.get("sha256") != _sha256(native["events"])
        or len(native["events"]) != 5
    ):
        raise CohortError("cohort arm native CRSA receipt is invalid")
    for event in native["events"]:
        k4._validate_native_event(event, "cohort arm")
    transactions = target.get("transactions")
    if not isinstance(transactions, Mapping):
        raise CohortError("cohort arm transaction receipt is absent")
    _verify_seal(transactions, k4.TRANSACTION_SCHEMA, "cohort transactions")
    transaction_rows = transactions.get("transactions")
    if not isinstance(transaction_rows, list):
        raise CohortError("cohort arm transaction rows are invalid")
    plan: list[tuple[str, list[int], int]] = []
    if k_value == 4:
        plan = k4._expected_transaction_plan(evidence.to_dict()["rounds"])
    else:
        for row in evidence.to_dict()["rounds"]:
            replay = row["replay_kind"]
            proposal = list(row["proposed_token_ids"])
            emitted = list(row["emitted_token_ids"])
            start = int(row["start_pos"])
            if replay == "remaining-single":
                plan.append(("decoded", emitted, start))
            elif replay == "commit-k2":
                plan.append(("committed", proposal, start))
            elif replay in {"mismatch0-decode", "eos0-decode"}:
                plan.extend(
                    (("discarded", proposal, start), ("decoded", emitted, start))
                )
            elif replay == "mismatch1-restage":
                plan.extend(
                    (("discarded", proposal, start), ("committed", emitted, start))
                )
            else:
                raise CohortError("cohort K2 replay grammar is invalid")
    if len(transaction_rows) != len(plan):
        raise CohortError("cohort arm transaction count is invalid")
    for index, (row, expected) in enumerate(zip(transaction_rows, plan, strict=True)):
        transition, input_ids, start = expected
        k4._validate_transaction_row(
            row,
            index=index,
            transition=transition,
            input_ids=input_ids,
            start_pos=start,
            attention_mode="native-crsa",
        )
    derived_positions = _generation_positions(transactions, native["events"])
    if positions != derived_positions:
        raise CohortError("cohort arm position/transaction cross-link is invalid")
    return document


def _position_projection(arm: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "hidden": row["hidden"],
            "input_token_id": row["input_token_id"],
            "native_crsa": row["native_crsa"],
            "position": row["position"],
        }
        for row in arm["target"]["positions"]
    ]


def _pair_parity(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    if left["contract"]["route_id"] == right["contract"]["route_id"]:
        raise CohortError("paired arms must use different routes")
    by_route = {
        left["contract"]["route_id"]: left,
        right["contract"]["route_id"]: right,
    }
    if set(by_route) != {"A", "B"}:
        raise CohortError("paired arms must contain A and B")
    a = by_route["A"]
    b = by_route["B"]
    tokens_equal = (
        a["tokens"]["generated_token_ids"] == b["tokens"]["generated_token_ids"]
    )
    cursor_equal = a["target"]["cursor"] == b["target"]["cursor"]
    hidden_equal = _position_projection(a) == _position_projection(b)
    state_equal = a["target"]["state"] == b["target"]["state"]
    tensor_equal = (
        a["target"]["state_tensor_sha256"] == b["target"]["state_tensor_sha256"]
    )
    crsa_equal = a["target"]["native_crsa"] == b["target"]["native_crsa"]
    a_alignment = a["drafter"]["alignment"]
    b_alignment = b["drafter"]["alignment"]
    cursors_equal = a_alignment["committed_cursor"] == b_alignment["committed_cursor"]
    draft_state_equal: bool | None = None
    if cursors_equal:
        draft_state_equal = a["drafter"]["state"] == b["drafter"]["state"]
    prefix_valid = all(
        row["committed_cursor"] >= arm["prompt"]["rendered_token_count"]
        and row["committed_cursor"] <= row["target_cursor"]
        and len(row["target_only_suffix_ids"])
        == row["target_cursor"] - row["committed_cursor"]
        for row, arm in ((a_alignment, a), (b_alignment, b))
    )
    positive = all(
        (
            tokens_equal,
            cursor_equal,
            hidden_equal,
            state_equal,
            tensor_equal,
            crsa_equal,
            prefix_valid,
        )
    )
    if cursors_equal:
        positive = positive and draft_state_equal is True
    return {
        "crsa_equal": crsa_equal,
        "drafter_committed_cursors_equal": cursors_equal,
        "drafter_prefix_valid": prefix_valid,
        "drafter_state_equal_when_aligned": draft_state_equal,
        "hidden_equal_by_position": hidden_equal,
        "positive": positive,
        "target_cursor_equal": cursor_equal,
        "target_final_state_equal": state_equal,
        "target_state_tensor_sha256_equal": tensor_equal,
        "tokens_equal": tokens_equal,
    }


def _route_aggregate(arms: Sequence[Mapping[str, Any]], route: str) -> dict[str, Any]:
    rows = [arm for arm in arms if arm["contract"]["route_id"] == route]
    proposed = sum(int(row["acceptance"]["draft_tokens_proposed"]) for row in rows)
    accepted = sum(int(row["acceptance"]["accepted_draft_tokens"]) for row in rows)
    prefix: Counter[str] = Counter()
    replay: Counter[str] = Counter()
    fallback: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "seconds": 0.0, "source_body_bytes": 0}
    )
    for arm in rows:
        prefix.update(arm["acceptance"]["accepted_prefix_histogram"])
        replay.update(arm["acceptance"]["replay_histogram"])
        for round_row in arm["target"]["speculative_evidence"]["rounds"]:
            bucket = fallback[str(round_row["replay_kind"])]
            bucket["count"] = int(bucket["count"]) + 1
            bucket["seconds"] = float(bucket["seconds"]) + float(round_row["seconds"])
            bucket["source_body_bytes"] = int(bucket["source_body_bytes"]) + int(
                round_row["source_body_bytes"]
            )
    return {
        "accepted_draft_tokens": accepted,
        "accepted_prefix_histogram": dict(sorted(prefix.items())),
        "arm_wall_seconds": sum(
            float(row["timing"]["arm_wall_seconds"]) for row in rows
        ),
        "combined_logical_bytes": sum(
            int(row["cost"]["combined_logical_bytes"]) for row in rows
        ),
        "complete_blocks": sum(
            int(row["acceptance"]["complete_blocks"]) for row in rows
        ),
        "draft_tokens_proposed": proposed,
        "fallback_penalty_by_replay": {
            key: dict(value) for key, value in sorted(fallback.items())
        },
        "micro_acceptance": 0.0 if proposed == 0 else accepted / proposed,
        "provider_guard_bytes": sum(
            int(row["cost"]["provider_guard_bytes"]) for row in rows
        ),
        "provider_guard_seconds": sum(
            float(row["timing"]["provider_guard_seconds"]) for row in rows
        ),
        "replay_histogram": dict(sorted(replay.items())),
        "target_forward_passes": sum(
            int(row["target"]["speculative_evidence"]["forward_passes"]) for row in rows
        ),
        "target_head_scans": sum(
            int(row["target"]["speculative_evidence"]["head_scans"]) for row in rows
        ),
    }


def _aggregate(
    arms: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    a = _route_aggregate(arms, "A")
    b = _route_aggregate(arms, "B")
    paired_byte = []
    paired_wall = []
    first = []
    subsequent = []
    for pair in pairs:
        arm_a = arms[int(pair["arm_index_a"])]
        arm_b = arms[int(pair["arm_index_b"])]
        a_bytes = float(arm_a["cost"]["combined_logical_bytes"])
        b_bytes = float(arm_b["cost"]["combined_logical_bytes"])
        a_wall = float(arm_a["timing"]["arm_wall_seconds"])
        b_wall = float(arm_b["timing"]["arm_wall_seconds"])
        paired_byte.append(0.0 if a_bytes == 0 else (a_bytes - b_bytes) / a_bytes)
        paired_wall.append(0.0 if a_wall == 0 else (a_wall - b_wall) / a_wall)
        order = pair["execution_order"]
        first.append(
            float((arm_a if order[0] == "A" else arm_b)["timing"]["arm_wall_seconds"])
        )
        subsequent.append(
            float((arm_a if order[1] == "A" else arm_b)["timing"]["arm_wall_seconds"])
        )
    return {
        "A_2xK2": a,
        "B_K4": b,
        "first_pair_arm_wall_seconds": first,
        "paired_median_byte_savings_fraction": statistics.median(paired_byte),
        "paired_median_wall_savings_fraction": statistics.median(paired_wall),
        "subsequent_pair_arm_wall_seconds": subsequent,
        "total_byte_savings": int(a["combined_logical_bytes"])
        - int(b["combined_logical_bytes"]),
        "total_byte_savings_fraction": 0.0
        if int(a["combined_logical_bytes"]) == 0
        else (int(a["combined_logical_bytes"]) - int(b["combined_logical_bytes"]))
        / int(a["combined_logical_bytes"]),
        "total_wall_savings_seconds": float(a["arm_wall_seconds"])
        - float(b["arm_wall_seconds"]),
        "total_wall_savings_fraction": 0.0
        if float(a["arm_wall_seconds"]) == 0.0
        else (float(a["arm_wall_seconds"]) - float(b["arm_wall_seconds"]))
        / float(a["arm_wall_seconds"]),
        "wall_speedup": math.inf
        if float(b["arm_wall_seconds"]) == 0.0
        else float(a["arm_wall_seconds"]) / float(b["arm_wall_seconds"]),
    }


def _mount_document(
    *,
    input_sha256: str,
    pair: _SingleMountPair,
    target_path: Path,
    draft_path: Path,
    target_tokenizer: Mapping[str, Any],
    tokenizer_seconds: float,
) -> dict[str, Any]:
    if (
        pair.counts != {"target": 1, "draft": 1}
        or pair.verification_counts != {"target": 1, "draft": 1}
        or pair.preflight_counts != {"target": 1, "draft": 1}
    ):
        raise CohortError("bundle authentication count is not exactly one per model")
    target = pair.target
    draft = pair.draft
    if target.model.config.vocab_size != draft.model.config.vocab_size:
        raise CohortError("target and drafter checkpoint vocabularies differ")
    return _seal(
        {
            "authentication": {
                "draft_preflight_count": pair.preflight_counts["draft"],
                "draft_preflight_seconds": draft.preflight_seconds,
                "draft_verification_count": pair.verification_counts["draft"],
                "draft_verification_seconds": draft.verify_seconds,
                "target_preflight_count": pair.preflight_counts["target"],
                "target_preflight_seconds": target.preflight_seconds,
                "target_verification_count": pair.verification_counts["target"],
                "target_verification_seconds": target.verify_seconds,
                "tokenizer_authentication_seconds": tokenizer_seconds,
            },
            "bundles": {
                "draft": draft.bundle_receipt,
                "target": target.bundle_receipt,
            },
            "checkpoint_vocabulary": {
                "draft_size": draft.model.config.vocab_size,
                "equal": True,
                "target_size": target.model.config.vocab_size,
            },
            "execution_regime": EXECUTION_REGIME,
            "immutable_inputs": {
                "draft_bundle_root": _stat_receipt(draft_path),
                "draft_manifest": _stat_receipt(draft_path / "bundle.json"),
                "target_bundle_root": _stat_receipt(target_path),
                "target_manifest": _stat_receipt(target_path / "bundle.json"),
            },
            "input_sha256": input_sha256,
            "process": {
                "cpu_count": os.cpu_count(),
                "load_average": list(os.getloadavg()),
                "pid": os.getpid(),
                "torch_interop_threads": int(
                    __import__("torch").get_num_interop_threads()
                ),
                "torch_threads": int(__import__("torch").get_num_threads()),
            },
            "schema": MOUNT_SCHEMA,
            "tokenizer": dict(target_tokenizer),
            "write_policy": "read-only-open-paths/no-checkpoint-write/v1",
        }
    )


def execute(
    *,
    input_path: Path,
    target_path: Path,
    draft_path: Path,
    target_tokenizer_path: Path,
    output_dir: Path,
    target_source_budget_mb: float = 1_048_576.0,
    draft_source_budget_mb: float = 262_144.0,
    max_resident_mb: float = 384.0,
    max_context_tokens: int = 2048,
    head_block_rows: int = Qwen38WeightPager.DEFAULT_HEAD_BLOCK_ROWS,
    require_production_profile: bool = True,
    require_official_target: bool = True,
    require_frozen: bool = True,
    require_clean: bool = True,
) -> tuple[dict[str, Any], Path]:
    protected = (input_path, target_path, draft_path, target_tokenizer_path)
    _assert_no_holdout_paths((*protected, output_dir))
    input_document = _validate_input(
        _read_json(input_path, "cohort input"), require_frozen=require_frozen
    )
    revision = _git_revision(require_clean=require_clean)
    if input_document.get("code_revision") != revision:
        raise CohortError("cohort input was sealed for a different code revision")
    destination = _guard_output_dir(output_dir, protected)
    completed: list[dict[str, Any]] = []
    written: list[Path] = []
    mount_document: dict[str, Any] | None = None
    pair = _SingleMountPair()
    native_evidence: list[Any] = []
    failure: BaseException | None = None
    try:
        tokenizer_started = time.perf_counter()
        target_tokenizer, target_tokenizer_receipt = _tokenizer_receipt(
            target_tokenizer_path, require_official=require_official_target
        )
        tokenizer_seconds = time.perf_counter() - tokenizer_started
        if target_tokenizer_receipt != input_document["tokenizer"]:
            raise CohortError("execution tokenizer differs from sealed cohort input")
        for item in input_document["items"]:
            rendered = target_tokenizer.render_no_thinking_prompt(
                "", item["normalized_question"]
            )
            if list(target_tokenizer.encode(rendered)) != item["rendered_token_ids"]:
                raise CohortError(
                    "sealed prompt tokens differ from execution tokenizer"
                )
        target = pair.open_role(
            "target",
            bundle=target_path,
            identity=LogicalModelIdentity(OFFICIAL_REPO_ID, OFFICIAL_REVISION),
            source_budget_mb=target_source_budget_mb,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=int(max_resident_mb * MIB),
            max_context_tokens=max_context_tokens,
            attention_mode="native-crsa",
            native_evidence=native_evidence,
            require_production_profile=require_production_profile,
            require_official_target=require_official_target,
        )
        draft = pair.open_role(
            "draft",
            bundle=draft_path,
            identity=LogicalModelIdentity(
                QWEN35_DRAFTER_REPO_ID, QWEN35_DRAFTER_REVISION
            ),
            source_budget_mb=draft_source_budget_mb,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=int(max_resident_mb * MIB),
            max_context_tokens=max_context_tokens,
            attention_mode="off",
            native_evidence=[],
            require_production_profile=require_production_profile,
            require_official_target=False,
        )
        if target.model.config.vocab_size != draft.model.config.vocab_size:
            raise CohortError("target and drafter checkpoint vocabularies differ")
        mount_document = _mount_document(
            input_sha256=input_document["sha256"],
            pair=pair,
            target_path=target_path,
            draft_path=draft_path,
            target_tokenizer=target_tokenizer_receipt,
            tokenizer_seconds=tokenizer_seconds,
        )
        mount_path = _atomic_new_json(destination / "mount.json", mount_document)
        written.append(mount_path)
        arm_index = 0
        pair_rows: list[dict[str, Any]] = []
        for item in input_document["items"]:
            pair_arms: list[dict[str, Any]] = []
            pair_indices: dict[str, int] = {}
            for route in item["schedule"]:
                arm = _run_arm(
                    arm_index=arm_index,
                    route=route,
                    item=item,
                    input_sha256=input_document["sha256"],
                    mount_sha256=mount_document["sha256"],
                    target=target,
                    draft=draft,
                    native_evidence=native_evidence,
                    head_block_rows=head_block_rows,
                )
                path = _atomic_new_json(
                    destination / _arm_filename(arm_index, item, route), arm
                )
                written.append(path)
                completed.append(arm)
                pair_arms.append(arm)
                pair_indices[route] = arm_index
                arm_index += 1
            parity = _pair_parity(pair_arms[0], pair_arms[1])
            pair_rows.append(
                {
                    "arm_index_a": pair_indices["A"],
                    "arm_index_b": pair_indices["B"],
                    "execution_order": list(item["schedule"]),
                    "item_id": item["item_id"],
                    "offset": item["offset"],
                    "parity": parity,
                    "stratum": item["stratum"],
                }
            )
        correctness = all(row["parity"]["positive"] for row in pair_rows)
        aggregate = _aggregate(completed, pair_rows)
        if correctness:
            bytes_positive = aggregate["total_byte_savings"] > 0
            wall_positive = aggregate["total_wall_savings_seconds"] > 0.0
            performance = (
                "positive"
                if bytes_positive and wall_positive
                else ("mixed" if bytes_positive or wall_positive else "negative")
            )
            performance_claims: Mapping[str, Any] | None = aggregate
        else:
            performance = "blocked-correctness"
            performance_claims = None
        # A successful cohort receipt certifies cleanup as well as generation.
        # Close only after all eight arms; the mounts therefore stay live for
        # the complete ABBA schedule and are never reopened.
        pair.close()
        result = _seal(
            {
                "aggregate": performance_claims,
                "arm_sha256": [arm["sha256"] for arm in completed],
                "code_revision": revision,
                "correctness_status": "positive" if correctness else "failed",
                "input_sha256": input_document["sha256"],
                "mount_sha256": mount_document["sha256"],
                "pairs": pair_rows,
                "performance_status": performance,
                "raw_metrics_when_correctness_failed": None
                if correctness
                else aggregate,
                "schema": RESULT_SCHEMA,
            }
        )
        result_path = _atomic_new_json(destination / "result.json", result)
        written.append(result_path)
        transport = _seal(
            {
                "files": [
                    {
                        "name": path.name,
                        "raw_sha256": _file_sha256(path, path.name)[0],
                        "size_bytes": path.stat().st_size,
                    }
                    for path in written
                ],
                "schema": TRANSPORT_SCHEMA,
                "session_result_sha256": result["sha256"],
            }
        )
        _atomic_new_json(destination / "transport.json", transport)
        return result, result_path
    except BaseException as exc:
        failure = exc
        abort = _seal(
            {
                "completed_arm_sha256": [arm["sha256"] for arm in completed],
                "failure": {"message": str(exc), "type": type(exc).__qualname__},
                "input_sha256": input_document["sha256"],
                "mount_sha256": None
                if mount_document is None
                else mount_document["sha256"],
                "retry_performed": False,
                "schema": ABORT_SCHEMA,
            }
        )
        try:
            _atomic_new_json(destination / "abort.json", abort)
        except BaseException as abort_failure:
            exc.add_note(f"abort receipt failed: {abort_failure}")
    finally:
        try:
            pair.close()
        except BaseException as cleanup_failure:
            if failure is None:
                failure = cleanup_failure
            else:
                failure.add_note(f"cleanup also failed: {cleanup_failure}")
    assert failure is not None
    raise failure


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="seal question-only input before model execution"
    )
    prepare.add_argument("--dev64", default=str(DEFAULT_DEV64))
    prepare.add_argument("--tokenizer-json", required=True)
    prepare.add_argument("--output", default=str(DEFAULT_INPUT))
    execute_parser = commands.add_parser(
        "execute", help="run the frozen one-process ABBA cohort"
    )
    execute_parser.add_argument("--input", default=str(DEFAULT_INPUT))
    execute_parser.add_argument("--target-bundle", default=str(DEFAULT_TARGET))
    execute_parser.add_argument("--draft-bundle", default=str(DEFAULT_DRAFT))
    execute_parser.add_argument("--target-tokenizer-json")
    execute_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    execute_parser.add_argument(
        "--target-source-budget-mb", type=_positive_float, default=1_048_576.0
    )
    execute_parser.add_argument(
        "--draft-source-budget-mb", type=_positive_float, default=262_144.0
    )
    execute_parser.add_argument(
        "--max-resident-mb", type=_positive_float, default=384.0
    )
    execute_parser.add_argument(
        "--max-context-tokens", type=_positive_int, default=2048
    )
    execute_parser.add_argument(
        "--head-block-rows",
        type=_positive_int,
        default=Qwen38WeightPager.DEFAULT_HEAD_BLOCK_ROWS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            document, path = prepare_input(
                dev64_path=Path(args.dev64),
                tokenizer_path=Path(args.tokenizer_json),
                output_path=Path(args.output),
            )
            print(
                json.dumps(
                    {
                        "items": [
                            {
                                "offset": row["offset"],
                                "tokens": row["rendered_token_count"],
                            }
                            for row in document["items"]
                        ],
                        "output": str(path),
                        "raw_sha256": _file_sha256(path, "input")[0],
                        "sha256": document["sha256"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        target = Path(args.target_bundle)
        draft = Path(args.draft_bundle)
        result, path = execute(
            input_path=Path(args.input),
            target_path=target,
            draft_path=draft,
            target_tokenizer_path=Path(args.target_tokenizer_json)
            if args.target_tokenizer_json
            else target / "tokenizer.json",
            output_dir=Path(args.output_dir),
            target_source_budget_mb=args.target_source_budget_mb,
            draft_source_budget_mb=args.draft_source_budget_mb,
            max_resident_mb=args.max_resident_mb,
            max_context_tokens=args.max_context_tokens,
            head_block_rows=args.head_block_rows,
        )
        print(
            json.dumps(
                {
                    "correctness_status": result["correctness_status"],
                    "output": str(path),
                    "performance_status": result["performance_status"],
                    "sha256": result["sha256"],
                },
                sort_keys=True,
            )
        )
        return 0 if result["correctness_status"] == "positive" else 1
    except (CohortError, OSError, TypeError, ValueError, KeyError) as exc:
        sys.stderr.write(f"qwen35_k4_cohort: error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
