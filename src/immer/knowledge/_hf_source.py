"""Safetensors range source with bounded persistent HTTP connections.

The primary transport leases exactly two independently pooled
``requests.Session`` objects.  A session is never used concurrently and each
owns one blocking HTTP/1.1 connection pool.  Explicit ``urllib`` openers stay
available for deterministic fixtures.  Payloads are always read through a
bounded stream; ``Response.content`` is deliberately forbidden here.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Protocol, Sequence

DEFAULT_ENDPOINT = "https://huggingface.co"
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_METADATA_BYTES = 64 * 1024 * 1024
DEFAULT_HTTP_CONNECTIONS = 2
MAX_ERROR_BODY_BYTES = 4 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _requests_api() -> Any:
    """Load the remote transport only when an HTTP reader needs it."""

    import requests

    return requests


class BudgetExceeded(Exception):
    """The source transfer budget would be exceeded."""


class Budget:
    """Minimal transfer-accounting interface shared by local and HF readers."""

    def __init__(self, limit_mb: float) -> None:
        self.limit = int(float(limit_mb) * 1024 * 1024)
        self.exceeded_error = BudgetExceeded
        self.body = 0
        self.overhead = 0
        self.requests = 0
        self.log: list[tuple[str, int, int]] = []

    def charge(self, body: int, overhead: int, tag: str) -> None:
        self.body += int(body)
        self.overhead += int(overhead)
        self.requests += 1
        self.log.append((str(tag), int(body), int(overhead)))
        if self.total > self.limit:
            raise BudgetExceeded(
                f"Bytebudget ueberschritten: {self.total}/{self.limit} Bytes bei {tag!r}"
            )

    @property
    def total(self) -> int:
        return int(self.body + self.overhead)

    def line(self) -> str:
        percent = 100.0 * self.total / self.limit if self.limit else 0.0
        return (
            f"[budget {self.total / 1048576:7.2f}/{self.limit / 1048576:.0f} MB "
            f"({percent:5.1f}%) body={self.body / 1048576:7.2f} MB "
            f"hdr~{self.overhead / 1024:6.1f} KB reqs={self.requests}]"
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "limit_bytes": int(self.limit),
            "bytes_total": self.total,
            "bytes_body": int(self.body),
            "bytes_overhead_approx": int(self.overhead),
            "http_requests": int(self.requests),
        }


class SourceError(RuntimeError):
    """A remote tensor source violated the range-source contract."""


class SourceNotFound(SourceError):
    """A requested repository file does not exist."""


class SourceAccessDenied(SourceError):
    """A repository is private, gated, or otherwise inaccessible."""


class SourceRangeError(SourceError):
    """The server did not honor an exact inclusive byte range."""


class _BudgetLike(Protocol):
    limit: int
    body: int
    overhead: int
    exceeded_error: type[Exception]

    @property
    def total(self) -> int: ...

    def charge(self, body: int, overhead: int, tag: str) -> None: ...

    def as_dict(self) -> dict[str, int]: ...


def _response_status(response: Any) -> int:
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(response, "status", None)
    if status is None:
        status = response.getcode()
    return int(status)


def _response_url(response: Any) -> str:
    url = getattr(response, "url", None)
    return str(url if url is not None else response.geturl())


def _read_bounded(response: Any, maximum: int) -> bytes:
    """Read at most ``maximum`` bytes without materializing the full body."""

    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
        raise ValueError("bounded response read requires a non-negative integer")
    raw = getattr(response, "raw", None)
    body = raw.read(maximum) if raw is not None else response.read(maximum)
    if not isinstance(body, (bytes, bytearray, memoryview)):
        raise SourceError("HTTP response returned a non-bytes body")
    return bytes(body)


def _response_overhead(response: Any) -> int:
    """Stable approximation of request/status/header transfer overhead.

    Works with both urllib's http.client.HTTPResponse (has ``.status`` and
    ``.getheader()``-style headers) and requests.models.Response (has
    ``.status_code`` and case-insensitive dict headers).
    """
    status = _response_status(response)
    status_line = f"HTTP/1.1 {status}\r\n"
    response_headers = "".join(
        f"{key}: {value}\r\n" for key, value in response.headers.items()
    )
    return (
        64
        + len(status_line.encode("latin-1", "replace"))
        + len(response_headers.encode("latin-1", "replace"))
    )


def _location_expiry(url: str) -> float:
    try:
        values = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        return float(int(values.get("Expires", [time.time() + 900])[0]) - 60)
    except (TypeError, ValueError):
        return time.time() + 900


def _etag_sha256(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.removeprefix("W/").strip().strip('"').lower()
    return normalized if _SHA256.fullmatch(normalized) is not None else None


class _LeasedResponse:
    """Return one session lease exactly once when a response is closed."""

    def __init__(self, response: Any, release: Any) -> None:
        self._response = response
        self._release = release
        self._closed = False
        self._close_lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._response.close()
        finally:
            self._release()


class HFRangeReader:
    """Exact single-range reader for public Hugging Face repository files.

    ``urllib`` follows the resolve redirect, after which the signed final URL is
    cached until shortly before expiry.  Payload reads are always bounded.
    """

    range_overhead_reserve = 16 * 1024

    def __init__(
        self,
        repo: str,
        revision: str = "main",
        budget: _BudgetLike | None = None,
        *,
        endpoint: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        opener: Any | None = None,
        session: Any | None = None,
        sessions: Sequence[Any] | None = None,
    ) -> None:
        self.repo = repo
        self.rev = revision
        self.budget = budget or Budget(200.0)
        self.endpoint = (
            endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT
        ).rstrip("/")
        self.timeout = float(timeout)
        if opener is not None and (session is not None or sessions is not None):
            raise ValueError("opener and persistent sessions are mutually exclusive")
        if session is not None and sessions is not None:
            raise ValueError("session and sessions are mutually exclusive")
        self._session_pool: queue.LifoQueue[Any] | None = None
        self._sessions: tuple[Any, ...] = ()
        if opener is not None:
            self.opener = opener
            self.transport_policy = "urllib-explicit-opener/v1"
            self.transport_connection_limit = 0
            self.transport_active_lease_limit = 0
        else:
            self.opener = None
            supplied = (
                (session,)
                if session is not None
                else tuple(sessions)
                if sessions is not None
                else ()
            )
            if supplied and not 1 <= len(supplied) <= DEFAULT_HTTP_CONNECTIONS:
                raise ValueError("persistent session count must be one or two")
            self._sessions = supplied or tuple(
                self._new_session() for _ in range(DEFAULT_HTTP_CONNECTIONS)
            )
            self._session_pool = queue.LifoQueue(maxsize=len(self._sessions))
            for value in self._sessions:
                self._session_pool.put_nowait(value)
            self.transport_active_lease_limit = len(self._sessions)
            if supplied:
                # Arbitrary injected transports are useful for deterministic
                # fixtures, but their adapter/socket topology is not under our
                # control.  Bind that fact instead of making a false pool claim.
                self.transport_policy = (
                    f"requests-injected-session-leases-{len(self._sessions)}/v1"
                )
                self.transport_connection_limit = 0
            else:
                self.transport_policy = "requests-session-pool-2/v1"
                self.transport_connection_limit = len(self._sessions)
        self.file_info: dict[str, dict[str, Any]] = {}
        self._cdn: dict[str, tuple[str, float]] = {}
        # Signed-URL state and the identity derived from its response must
        # move together.  An RLock lets the small snapshot/update helpers be
        # composed without serializing the network read itself.
        self._lock = threading.RLock()
        self._transport_requests = 0
        self._transport_retries = 0
        self._transport_active_leases = 0
        self._transport_peak_leases = 0
        self._transport_connection_objects: set[int] = set()
        self._closed = False

    @staticmethod
    def _new_session() -> Any:
        requests = _requests_api()
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=1,
            max_retries=0,
            pool_block=True,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(
            {
                "User-Agent": "immer-tensor-source/1.1",
                "Accept-Encoding": "identity",
            }
        )
        return session

    def _lease_session(self) -> Any:
        pool = self._session_pool
        if pool is None:
            raise SourceError("persistent transport is not configured")
        try:
            session = pool.get(timeout=self.timeout)
        except queue.Empty as exc:
            raise SourceError(
                "timed out waiting for a persistent HTTP session"
            ) from exc
        with self._lock:
            if self._closed:
                session.close()
                raise SourceError("HTTP range reader is closed")
            self._transport_active_leases += 1
            self._transport_peak_leases = max(
                self._transport_peak_leases,
                self._transport_active_leases,
            )
        return session

    def _return_session(self, session: Any) -> None:
        with self._lock:
            self._transport_active_leases -= 1
            if self._transport_active_leases < 0:
                self._transport_active_leases = 0
                raise SourceError("persistent HTTP session lease underflow")
            closed = self._closed
        if closed:
            session.close()
            return
        pool = self._session_pool
        if pool is None:
            session.close()
            return
        try:
            pool.put_nowait(session)
        except queue.Full as exc:
            session.close()
            raise SourceError("persistent HTTP session returned twice") from exc

    def transport_metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "transport_policy": self.transport_policy,
                "transport_connection_limit": self.transport_connection_limit,
                "transport_active_lease_limit": self.transport_active_lease_limit,
                "transport_requests": self._transport_requests,
                "transport_retries": self._transport_retries,
                "transport_active_leases": self._transport_active_leases,
                "transport_peak_leases": self._transport_peak_leases,
                "transport_connection_objects_seen": len(
                    self._transport_connection_objects
                ),
                "transport_closed": self._closed,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._transport_active_leases:
                raise SourceError(
                    "cannot close HTTP range reader with active responses"
                )
            self._closed = True
        for session in self._sessions:
            session.close()

    def resolve_url(self, filename: str) -> str:
        repo = urllib.parse.quote(self.repo, safe="/")
        revision = urllib.parse.quote(self.rev, safe="/")
        name = urllib.parse.quote(filename, safe="/")
        return f"{self.endpoint}/{repo}/resolve/{revision}/{name}"

    def _headers(self, *, byte_range: tuple[int, int] | None = None) -> dict[str, str]:
        headers = {
            "User-Agent": "immer-tensor-source/1.0",
            "Accept-Encoding": "identity",
        }
        if byte_range is not None:
            headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _source_url(self, filename: str) -> str:
        with self._lock:
            cached = self._cdn.get(filename)
            if cached is not None and cached[1] > time.time():
                return cached[0]
        return self.resolve_url(filename)

    def update_file_info(
        self,
        filename: str,
        values: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Atomically merge and return one file's source metadata."""

        with self._lock:
            info = self.file_info.setdefault(filename, {})
            info.update(values)
            return dict(info)

    def file_info_snapshot(self, filename: str) -> dict[str, Any]:
        """Return a copy that cannot observe or cause a partial update."""

        with self._lock:
            return dict(self.file_info.get(filename, {}))

    def source_identity_snapshot(self, filename: str) -> dict[str, str]:
        """Return known remote identity without claiming a freshness check.

        Unlike a local reader's authoritative ``source_identity()``, this is
        only metadata observed on prior HTTP responses.  Keeping the names
        distinct prevents an inventory resume from treating a remote snapshot
        as a live local-file stat.
        """

        info = self.file_info_snapshot(filename)
        return {
            key: str(info[key])
            for key in (
                "cas_url_hash",
                "etag",
                "linked_etag",
                "payload_sha256",
                "repo_commit",
                "size",
                "xet_hash",
            )
            if info.get(key) is not None
        }

    def _remember_response(self, filename: str, response: Any) -> None:
        final_url = _response_url(response)
        updates: dict[str, Any] = {
            "cdn_host": urllib.parse.urlparse(final_url).netloc,
        }
        etag = response.headers.get("ETag")
        if etag:
            updates["etag"] = etag
        chain = tuple(getattr(response, "history", ()) or ()) + (response,)
        for candidate in chain:
            headers = getattr(candidate, "headers", {})
            linked = _etag_sha256(headers.get("X-Linked-ETag"))
            if linked is not None:
                updates["linked_etag"] = linked
                updates["payload_sha256"] = linked
            xet_hash = _etag_sha256(headers.get("X-Xet-Hash"))
            if xet_hash is not None:
                updates["xet_hash"] = xet_hash
            repo_commit = headers.get("X-Repo-Commit")
            if isinstance(repo_commit, str) and re.fullmatch(
                r"[0-9a-fA-F]{40,64}", repo_commit
            ):
                updates["repo_commit"] = repo_commit.lower()
            linked_size = headers.get("X-Linked-Size")
            if isinstance(linked_size, str) and linked_size.isdigit():
                updates["size"] = int(linked_size)
        match = re.search(r"/([0-9a-f]{64})(?:\?|$)", final_url)
        if match:
            updates["cas_url_hash"] = match.group(1)
        known = self.file_info_snapshot(filename)
        if (
            "payload_sha256" not in updates
            and "xet_hash" not in updates
            and known.get("xet_hash") is None
        ):
            payload = _etag_sha256(etag) or updates.get("cas_url_hash")
            if payload is not None:
                updates["payload_sha256"] = payload
        content_range = response.headers.get("Content-Range", "")
        range_match = re.fullmatch(r"bytes \d+-\d+/(\d+|\*)", content_range)
        content_length = response.headers.get("Content-Length")
        status = _response_status(response)
        if range_match is not None and range_match.group(1) != "*":
            updates["size"] = int(range_match.group(1))
        elif status == 200 and content_length and content_length.isdigit():
            updates["size"] = int(content_length)

        with self._lock:
            if final_url != self.resolve_url(filename):
                self._cdn[filename] = (final_url, _location_expiry(final_url))
            self.file_info.setdefault(filename, {}).update(updates)

    def _open_once(
        self,
        url: str,
        headers: Mapping[str, str],
    ) -> Any:
        with self._lock:
            self._transport_requests += 1
        if self._session_pool is None:
            request = urllib.request.Request(url, headers=dict(headers))
            return self.opener.open(request, timeout=self.timeout)

        session = self._lease_session()
        try:
            response = session.get(
                url,
                headers=dict(headers),
                timeout=self.timeout,
                stream=True,
                allow_redirects=True,
            )
        except BaseException as exc:
            error_response = getattr(exc, "response", None)
            if error_response is not None:
                error_response.close()
            self._return_session(session)
            raise
        connection = getattr(getattr(response, "raw", None), "_connection", None)
        if connection is not None:
            with self._lock:
                self._transport_connection_objects.add(id(connection))
        return _LeasedResponse(
            response,
            lambda: self._return_session(session),
        )

    def _charge_http_error(self, filename: str, response: Any) -> bytes:
        overhead = _response_overhead(response)
        remaining = max(0, self.budget.limit - self.budget.total - overhead)
        body = _read_bounded(response, min(MAX_ERROR_BODY_BYTES, remaining))
        self.budget.charge(len(body), overhead, f"http-error:{filename}")
        return body

    def _open(self, filename: str, *, byte_range: tuple[int, int] | None = None) -> Any:
        last: Exception | None = None
        for attempt in range(3):
            url = self._source_url(filename)
            used_cached_url = url != self.resolve_url(filename)
            headers = self._headers(byte_range=byte_range)
            response = None
            try:
                response = self._open_once(url, headers)
                try:
                    status = _response_status(response)
                except BaseException:
                    response.close()
                    response = None
                    raise
                if status >= 400:
                    try:
                        self._charge_http_error(filename, response)
                    finally:
                        response.close()
                        response = None
                    if status == 404:
                        raise SourceNotFound(
                            f"404: {filename} existiert nicht in {self.repo}@{self.rev}"
                        )
                    if status in (401, 403):
                        with self._lock:
                            self._cdn.pop(filename, None)
                        if used_cached_url and attempt < 2:
                            last = SourceAccessDenied(
                                f"HTTP {status} for stale signed URL"
                            )
                            with self._lock:
                                self._transport_retries += 1
                            continue
                        raise SourceAccessDenied(
                            f"{status}: {self.repo}@{self.rev} ist privat/gated oder gesperrt"
                        )
                    if status == 429 or status >= 500:
                        last = SourceError(f"HTTP {status} beim Lesen von {filename}")
                        if attempt < 2:
                            with self._lock:
                                self._transport_retries += 1
                            time.sleep(2**attempt)
                            continue
                    raise SourceError(f"HTTP {status} beim Lesen von {filename}")
                try:
                    self._remember_response(filename, response)
                except BaseException:
                    response.close()
                    raise
                return response
            except _requests_api().RequestException as exc:
                if response is not None:
                    response.close()
                last = exc
                if attempt < 2:
                    with self._lock:
                        self._transport_retries += 1
                    time.sleep(2**attempt)
                    continue
            except urllib.error.HTTPError as exc:
                overhead = _response_overhead(exc)
                remaining = max(0, self.budget.limit - self.budget.total - overhead)
                try:
                    body = _read_bounded(exc, min(MAX_ERROR_BODY_BYTES, remaining))
                    self.budget.charge(len(body), overhead, f"http-error:{filename}")
                finally:
                    exc.close()
                if exc.code == 404:
                    raise SourceNotFound(
                        f"404: {filename} existiert nicht in {self.repo}@{self.rev}"
                    ) from exc
                if exc.code in (401, 403):
                    with self._lock:
                        self._cdn.pop(filename, None)
                    if used_cached_url and attempt < 2:
                        last = exc
                        with self._lock:
                            self._transport_retries += 1
                        continue
                    raise SourceAccessDenied(
                        f"{exc.code}: {self.repo}@{self.rev} ist privat/gated oder gesperrt"
                    ) from exc
                if exc.code == 429 or exc.code >= 500:
                    last = exc
                    if attempt < 2:
                        with self._lock:
                            self._transport_retries += 1
                        time.sleep(2**attempt)
                        continue
                raise SourceError(f"HTTP {exc.code} beim Lesen von {filename}") from exc
            except (TimeoutError, urllib.error.URLError, OSError) as exc:
                last = exc
                if attempt < 2:
                    with self._lock:
                        self._transport_retries += 1
                    time.sleep(2**attempt)
                    continue
        raise SourceError(f"Netzfehler beim Lesen von {filename}: {last}") from last

    def get_range(self, filename: str, start: int, end: int) -> bytes:
        if start < 0 or end < start:
            raise SourceRangeError(f"Ungueltige Range [{start}, {end}]")
        expected = end - start + 1
        response = self._open(filename, byte_range=(start, end))
        overhead = _response_overhead(response)
        status = _response_status(response)
        try:
            if status == 206:
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)
                if (
                    match is None
                    or int(match.group(1)) != start
                    or int(match.group(2)) != end
                ):
                    body = _read_bounded(response, expected + 1)
                    self.budget.charge(
                        len(body), overhead, f"bad-range:{filename}:{start}"
                    )
                    raise SourceRangeError(
                        f"Content-Range-Mismatch bei {filename}: erwartet {start}-{end}, "
                        f"bekommen {content_range!r}"
                    )
                if match.group(3) != "*":
                    self.update_file_info(filename, {"size": int(match.group(3))})
            elif status != 200 or start != 0:
                body = _read_bounded(response, expected + 1)
                self.budget.charge(
                    len(body), overhead, f"bad-status:{filename}:{start}"
                )
                raise SourceRangeError(
                    f"Server lieferte HTTP {status} statt einer exakten Range fuer {filename}"
                )
            body = _read_bounded(response, expected + 1)
            self.budget.charge(len(body), overhead, f"range:{filename}:{start}")
        finally:
            response.close()
        if len(body) != expected:
            raise SourceRangeError(
                f"Range {filename}[{start}-{end}] lieferte {len(body)}/{expected} Bytes"
            )
        if status == 200:
            self.update_file_info(filename, {"size": len(body)})
        return body

    def fetch_file_bounded(self, filename: str, max_bytes: int) -> bytes:
        if max_bytes <= 0:
            raise ValueError("max_bytes muss positiv sein")
        response = self._open(filename)
        overhead = _response_overhead(response)
        remaining = self.budget.limit - self.budget.total - overhead
        if remaining < 0:
            response.close()
            raise self.budget.exceeded_error(f"Kein Bytebudget fuer {filename!r}")
        declared_raw = response.headers.get("Content-Length")
        declared = (
            int(declared_raw) if declared_raw and declared_raw.isdigit() else None
        )
        ceiling = min(int(max_bytes), remaining)
        if declared is not None and declared > ceiling:
            response.close()
            self.budget.charge(0, overhead, f"file-headers:{filename}")
            if declared > remaining:
                raise self.budget.exceeded_error(
                    f"{filename!r} braucht {declared}, uebrig sind {remaining} Bytes"
                )
            raise SourceRangeError(
                f"{filename!r} ist {declared} Bytes gross; Grenze ist {max_bytes}"
            )
        try:
            body = _read_bounded(
                response,
                ceiling + (1 if ceiling < remaining else 0),
            )
            self.budget.charge(len(body), overhead, f"file:{filename}")
        finally:
            response.close()
        if len(body) > ceiling:
            raise SourceRangeError(f"{filename!r} ueberschreitet {max_bytes} Bytes")
        if declared is not None and len(body) != declared:
            raise SourceRangeError(
                f"Kurzer Read von {filename}: {len(body)}/{declared} Bytes"
            )
        if declared is None and len(body) == ceiling:
            raise SourceRangeError(
                f"Laenge von {filename!r} ist unbekannt und erreicht die harte Grenze {ceiling}"
            )
        self.update_file_info(filename, {"size": len(body)})
        return body

    def fetch_file(self, filename: str) -> bytes:
        return self.fetch_file_bounded(filename, DEFAULT_MAX_METADATA_BYTES)


_CLASS_RULES = [
    (r"(^|\.)embed_tokens\.weight$", "embed"),
    (r"(^|\.)lm_head\.weight$", "lm_head"),
    (r"linear_attn\.in_proj_(qkv|z)\.weight$", "gdn_in_proj"),
    (r"linear_attn\.in_proj_[ab]\.weight$", "gdn_ba"),
    (r"linear_attn\.out_proj\.weight$", "gdn_out"),
    (r"linear_attn\.conv1d\.weight$", "gdn_conv"),
    (r"self_attn\.[qkv]_proj\.weight$", "attn_qkv"),
    (r"self_attn\.o_proj\.weight$", "attn_o"),
    (r"mlp\.gate_proj\.weight$", "mlp_gate"),
    (r"mlp\.up_proj\.weight$", "mlp_up"),
    (r"mlp\.down_proj\.weight$", "mlp_down"),
    (r"(^|\.)norm\.weight$", "final_norm"),
    (r"visual\.patch_embed\.proj\.weight$", "visual_patch_embed"),
    (r"visual\.pos_embed\.weight$", "visual_pos_embed"),
    (r"visual\..*attn\.qkv\.weight$", "visual_attn_qkv"),
    (r"visual\..*attn\.proj\.weight$", "visual_attn_o"),
    (r"visual\..*mlp\.(linear_fc1|gate_proj|up_proj)\.weight$", "visual_mlp_up"),
    (r"visual\..*mlp\.(linear_fc2|down_proj)\.weight$", "visual_mlp_down"),
]
_CLASS_RULES = [(re.compile(pattern), label) for pattern, label in _CLASS_RULES]
_LAYER_PATTERNS = [
    re.compile(r"(?:^|\.)(?:layers|blocks|h)\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)layer\.(\d+)(?:\.|$)"),
]


def classify_tensor(name: str) -> str:
    for pattern, label in _CLASS_RULES:
        if pattern.search(name):
            return label
    return "other"


def layer_index(name: str) -> int | None:
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def scan_inventory(reader: Any, budget: _BudgetLike) -> dict[str, Any]:
    """Read only an index and safetensors headers; never read tensor payloads."""
    inventory: dict[str, Any] = {
        "tool": "immer.tensor_source",
        "command": "scan",
        "repo": reader.repo,
        "revision": reader.rev,
        "scanned_at": _now_utc(),
        "shards": [],
        "tensors": [],
        "index_bytes": 0,
    }
    try:
        optional_fetch = getattr(reader, "fetch_optional_file", None)
        index_raw = (
            optional_fetch("model.safetensors.index.json")
            if callable(optional_fetch)
            else reader.fetch_file("model.safetensors.index.json")
        )
        try:
            index = json.loads(index_raw.decode("utf-8"))
            weight_map = index["weight_map"]
        except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise SourceError("model.safetensors.index.json ist ungueltig") from exc
        if not isinstance(weight_map, dict) or not weight_map:
            raise SourceError("Safetensors-Index enthaelt keine weight_map")
        shard_files = sorted({str(filename) for filename in weight_map.values()})
        inventory["index_bytes"] = len(index_raw)
        total_size = index.get("metadata", {}).get("total_size")
        inventory["index_total_size"] = (
            int(total_size) if total_size is not None else None
        )
    except SourceNotFound:
        shard_files = ["model.safetensors"]

    payload_total = 0
    for shard_file in shard_files:
        header, data_start, info = reader.fetch_st_header(shard_file)
        tensor_count = 0
        for name, entry in header.items():
            if not isinstance(entry, dict):
                raise SourceError(f"Tensor-Metadaten fuer {name!r} sind ungueltig")
            try:
                offset_begin, offset_end = (
                    int(value) for value in entry["data_offsets"]
                )
                shape = [int(value) for value in entry["shape"]]
                dtype = str(entry["dtype"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SourceError(
                    f"Tensor-Metadaten fuer {name!r} sind unvollstaendig"
                ) from exc
            numel = math.prod(shape) if shape else 1
            byte_count = offset_end - offset_begin
            inventory["tensors"].append(
                {
                    "name": name,
                    "tensor_class": classify_tensor(name),
                    "layer": layer_index(name),
                    "shard": shard_file,
                    "dtype": dtype,
                    "shape": shape,
                    "numel": numel,
                    "bytes": byte_count,
                    "offset_in_shard": [offset_begin, offset_end],
                    "data_start": data_start,
                }
            )
            tensor_count += 1
            payload_total += byte_count
        inventory["shards"].append(
            {
                "file": shard_file,
                "n_tensors": tensor_count,
                "header_len": info.get("header_len"),
                "data_start": data_start,
                "size": info.get("size"),
                "etag": info.get("etag"),
                "cas_url_hash": info.get("cas_url_hash"),
                "linked_etag": info.get("linked_etag"),
                "payload_sha256": info.get("payload_sha256"),
                "repo_commit": info.get("repo_commit"),
                "xet_hash": info.get("xet_hash"),
                "cdn_host": info.get("cdn_host"),
                "st_metadata": info.get("st_metadata"),
            }
        )
    declared_total = inventory.get("index_total_size")
    if declared_total is not None and payload_total != declared_total:
        raise SourceError(
            f"Payload-Summe {payload_total} != index.total_size {declared_total}"
        )
    inventory["model_payload_bytes"] = payload_total
    if declared_total is None:
        inventory["index_total_size"] = payload_total
    histogram: dict[str, int] = {}
    for tensor in inventory["tensors"]:
        label = tensor["tensor_class"]
        histogram[label] = histogram.get(label, 0) + 1
    inventory["class_histogram"] = dict(
        sorted(histogram.items(), key=lambda item: (-item[1], item[0]))
    )
    inventory["budget"] = budget.as_dict()
    return inventory


def bf16_rows_to_f32(values: Any, shape: tuple[int, int]) -> Any:
    """Decode selected little-endian BF16 words into an owned float32 array."""
    import numpy as np

    words = np.asarray(values, dtype="<u2")
    expected = int(shape[0]) * int(shape[1])
    if words.size != expected:
        raise ValueError(f"BF16-Payload hat {words.size} statt {expected} Elemente")
    expanded = np.ascontiguousarray(words.reshape(shape), dtype=np.uint32) << 16
    return expanded.view(np.float32)
