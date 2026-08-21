"""Deterministic replay of file mutations explicitly observed in a Pi trace.

Only successful, completed ``write`` and ``edit`` tool executions affect the
virtual filesystem.  The replay never touches the host filesystem and never
executes teacher code.  It is a descriptive reconstruction of observable Pi
actions, not a semantic interpretation or training recipe.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import posixpath
from typing import Any


class PiArtifactReplayError(ValueError):
    """The declared virtual root cannot safely contain replayed files."""


@dataclass(frozen=True, slots=True)
class ReplayedFile:
    """One final UTF-8 text file reconstructed in memory."""

    path: str
    content: str
    sha256: str

    @property
    def size_bytes(self) -> int:
        return len(self.content.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class ReplayIssue:
    """One completed tool call that could not or should not be applied."""

    index: int
    tool: str
    status: str
    reason: str


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """Final virtual state and counts over completed Pi tool executions."""

    applied: int
    skipped: int
    failed: int
    files: tuple[ReplayedFile, ...]
    issues: tuple[ReplayIssue, ...]
    virtual_root: str

    def file_map(self) -> dict[str, ReplayedFile]:
        """Return replayed files keyed by their root-relative POSIX path."""

        return {file.path: file for file in self.files}


@dataclass(frozen=True, slots=True)
class FileComparison:
    """Byte-level comparison of one replayed or archived UTF-8 text file."""

    path: str
    replay_sha256: str | None
    artifact_sha256: str | None
    exact_match: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ArtifactComparison:
    """Comparison against an archived artifact without executing it."""

    exact_matches: tuple[str, ...]
    mismatches: tuple[str, ...]
    missing_from_replay: tuple[str, ...]
    extra_in_replay: tuple[str, ...]
    files: tuple[FileComparison, ...]
    ignored_binary: tuple[str, ...] = ()

    @property
    def all_replayed_files_match(self) -> bool:
        return not self.mismatches and not self.extra_in_replay


def _digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _normalise_root(virtual_root: str | os.PathLike[str]) -> str:
    root = os.fspath(virtual_root)
    if not isinstance(root, str) or not root or "\x00" in root:
        raise PiArtifactReplayError("virtual_root must be a non-empty text path")
    if "\\" in root:
        raise PiArtifactReplayError("virtual_root must use POSIX separators")
    root = posixpath.normpath(root)
    if not posixpath.isabs(root) or root == "/":
        raise PiArtifactReplayError("virtual_root must be an absolute non-root path")
    if ".." in root.split("/"):
        raise PiArtifactReplayError("virtual_root must not contain traversal")
    return root


def _resolve_virtual_path(root: str, value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise PiArtifactReplayError("tool path must be non-empty text")
    if "\\" in value:
        raise PiArtifactReplayError("tool path must use POSIX separators")
    if ".." in value.split("/"):
        raise PiArtifactReplayError("tool path contains traversal")
    candidate = posixpath.normpath(value)
    if candidate in {"", ".", "/"} or value.endswith("/"):
        raise PiArtifactReplayError("tool path must name a file")
    absolute = (
        candidate if posixpath.isabs(candidate) else posixpath.join(root, candidate)
    )
    try:
        inside = posixpath.commonpath((root, absolute)) == root
    except ValueError as exc:
        raise PiArtifactReplayError(
            "tool path is incompatible with virtual_root"
        ) from exc
    if not inside or absolute == root:
        raise PiArtifactReplayError("tool path escapes virtual_root")
    return posixpath.relpath(absolute, root)


def _event_mapping(event: object) -> Mapping[str, Any] | None:
    if isinstance(event, Mapping):
        return event
    parsed = getattr(event, "parsed", None)
    return parsed if isinstance(parsed, Mapping) else None


def _apply_write(files: dict[str, str], path: str, args: Mapping[str, Any]) -> None:
    content = args.get("content")
    if not isinstance(content, str):
        raise PiArtifactReplayError("write.content must be text")
    files[path] = content


def _apply_edit(files: dict[str, str], path: str, args: Mapping[str, Any]) -> None:
    edits = args.get("edits")
    if not isinstance(edits, list) or not edits:
        raise PiArtifactReplayError("edit.edits must be a non-empty list")
    if path not in files:
        raise PiArtifactReplayError("edit target has no earlier successful write")

    candidate = files[path]
    for offset, edit in enumerate(edits):
        if not isinstance(edit, Mapping):
            raise PiArtifactReplayError(f"edit.edits[{offset}] must be an object")
        old = edit.get("oldText")
        new = edit.get("newText")
        if not isinstance(old, str) or not old:
            raise PiArtifactReplayError(
                f"edit.edits[{offset}].oldText must be non-empty"
            )
        if not isinstance(new, str):
            raise PiArtifactReplayError(f"edit.edits[{offset}].newText must be text")
        occurrences = candidate.count(old)
        if occurrences != 1:
            raise PiArtifactReplayError(
                f"edit.edits[{offset}].oldText matched {occurrences} blocks, expected 1"
            )
        candidate = candidate.replace(old, new, 1)
    files[path] = candidate


class PiArtifactReplayer:
    """Replay completed Pi tool events into a bounded in-memory filesystem."""

    def __init__(self, virtual_root: str | os.PathLike[str]) -> None:
        self.virtual_root = _normalise_root(virtual_root)

    def replay(self, events: Iterable[Mapping[str, Any]]) -> ReplayReport:
        """Replay one event stream from an empty virtual filesystem.

        A mutation is committed only when its correlated end record explicitly
        contains ``isError: false`` and the complete operation validates.  A
        multi-block edit is transactional.
        """

        starts: dict[str, tuple[int, Mapping[str, Any]]] = {}
        files: dict[str, str] = {}
        issues: list[ReplayIssue] = []
        applied = skipped = failed = 0

        for index, raw_event in enumerate(events, 1):
            event = _event_mapping(raw_event)
            if event is None:
                continue
            event_type = event.get("type")
            call_id = event.get("toolCallId")
            if event_type == "tool_execution_start":
                if isinstance(call_id, str) and call_id:
                    starts[call_id] = (index, event)
                continue
            if event_type != "tool_execution_end":
                continue

            start_entry = (
                starts.pop(call_id, None) if isinstance(call_id, str) else None
            )
            start = start_entry[1] if start_entry is not None else None
            tool_value = (
                start.get("toolName") if start is not None else event.get("toolName")
            )
            tool = tool_value if isinstance(tool_value, str) else "<unknown>"
            if tool not in {"write", "edit"}:
                skipped += 1
                issues.append(
                    ReplayIssue(
                        index,
                        tool,
                        "skipped",
                        "tool does not mutate replayed text state",
                    )
                )
                continue
            if start is None:
                failed += 1
                issues.append(
                    ReplayIssue(
                        index, tool, "failed", "completed tool has no correlated start"
                    )
                )
                continue
            if event.get("isError") is not False:
                failed += 1
                reason = (
                    "Pi marked the tool result as failed"
                    if event.get("isError") is True
                    else "tool result does not explicitly prove success"
                )
                issues.append(ReplayIssue(index, tool, "failed", reason))
                continue

            args = start.get("args")
            try:
                if not isinstance(args, Mapping):
                    raise PiArtifactReplayError("tool args must be an object")
                path = _resolve_virtual_path(self.virtual_root, args.get("path"))
                if tool == "write":
                    _apply_write(files, path, args)
                else:
                    _apply_edit(files, path, args)
            except PiArtifactReplayError as exc:
                failed += 1
                issues.append(ReplayIssue(index, tool, "failed", str(exc)))
            else:
                applied += 1

        for _, (index, start) in sorted(starts.items(), key=lambda item: item[1][0]):
            tool_value = start.get("toolName")
            tool = tool_value if isinstance(tool_value, str) else "<unknown>"
            if tool in {"write", "edit"}:
                skipped += 1
                issues.append(
                    ReplayIssue(
                        index, tool, "skipped", "tool start has no completed result"
                    )
                )

        replayed = tuple(
            ReplayedFile(path=path, content=content, sha256=_digest(content))
            for path, content in sorted(files.items())
        )
        return ReplayReport(
            applied=applied,
            skipped=skipped,
            failed=failed,
            files=replayed,
            issues=tuple(issues),
            virtual_root=self.virtual_root,
        )


def replay_pi_trace(
    path: str | Path, *, virtual_root: str | os.PathLike[str]
) -> ReplayReport:
    """Read Pi JSONL incrementally and replay its completed file mutations."""

    source = Path(path)

    def events() -> Iterable[Mapping[str, Any]]:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PiArtifactReplayError(
                        f"invalid JSON on line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(event, Mapping):
                    raise PiArtifactReplayError(
                        f"Pi event on line {line_number} must be an object"
                    )
                yield event

    return PiArtifactReplayer(virtual_root).replay(events())


def compare_replay_to_artifact(
    report: ReplayReport, artifact_root: str | Path
) -> ArtifactComparison:
    """Compare replayed text with archived UTF-8 files by exact bytes and hash."""

    root = Path(artifact_root)
    if not root.is_dir():
        raise PiArtifactReplayError("artifact_root must be an existing directory")

    artifact: dict[str, bytes] = {}
    ignored_binary: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            ignored_binary.append(relative)
        else:
            artifact[relative] = data

    replay = {file.path: file for file in report.files}
    comparisons: list[FileComparison] = []
    exact: list[str] = []
    mismatches: list[str] = []
    missing: list[str] = []
    extra: list[str] = []
    for path in sorted(set(replay) | set(artifact)):
        replayed = replay.get(path)
        archived = artifact.get(path)
        replay_digest = replayed.sha256 if replayed is not None else None
        artifact_digest = (
            hashlib.sha256(archived).hexdigest() if archived is not None else None
        )
        if replayed is None:
            missing.append(path)
            reason = (
                "artifact text was not produced by a successful Pi write/edit; "
                "it may have been generated by an executed tool"
            )
            is_exact = False
        elif archived is None:
            extra.append(path)
            reason = "replayed text is absent from the archived artifact"
            is_exact = False
        elif replayed.content.encode("utf-8") == archived:
            exact.append(path)
            reason = "exact UTF-8 byte match"
            is_exact = True
        else:
            mismatches.append(path)
            reason = "both texts exist but their bytes differ"
            is_exact = False
        comparisons.append(
            FileComparison(path, replay_digest, artifact_digest, is_exact, reason)
        )

    return ArtifactComparison(
        exact_matches=tuple(exact),
        mismatches=tuple(mismatches),
        missing_from_replay=tuple(missing),
        extra_in_replay=tuple(extra),
        files=tuple(comparisons),
        ignored_binary=tuple(ignored_binary),
    )
