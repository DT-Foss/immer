"""Cold, training-free runtime for the four SHIP-v6 arithmetic organs.

The deployment contract is intentionally narrower than a general calculator:
only explicit canonical arithmetic is accepted.  The persisted ``mod`` organ
implements addition in :math:`Z_3`, so its *only* public spelling is
``z3sum``; ordinary modulo/remainder syntax is rejected.

No donor data, WikiText download, optimiser, or training loop is reachable
from this module.  The frozen A1 host and every organ are digest-verified
before ``torch.load(..., weights_only=True)`` is allowed to see them.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from .organbank import DigestMismatch, OrganBank
from ..artifacts import artifact_root as select_artifact_root
from ..artifacts import resolve_artifact_path
from ..contracts import ExecutionStatus, Request, Result
from ..resource_paths import crsa_router_manifest, s3_ship_manifest


_SCHEMA = "immer.s3-ship-v6/v1"
_DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_CARDINALS = _DIGITS + (
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
)
_CARDINAL_VALUE = {word: value for value, word in enumerate(_CARDINALS)}
_VALUE_WORD = {value: word for word, value in _CARDINAL_VALUE.items()}
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]*|\d+|[+*x\-%=]", re.IGNORECASE)
_PUBLIC_OPERATORS = {
    "+": "plus",
    "plus": "plus",
    "-": "less",
    "less": "less",
    "minus": "less",
    "*": "times",
    "x": "times",
    "times": "times",
    "z3sum": "z3sum",
}
_FORBIDDEN_MODULO = frozenset({"%", "mod", "modulo", "remainder"})
_WORD_ALIASES = {
    "ist": "is",
    "mal": "times",
    "null": "zero",
    "eins": "one",
    "ein": "one",
    "eine": "one",
    "zwei": "two",
    "drei": "three",
    "vier": "four",
    "fuenf": "five",
    "sechs": "six",
    "sieben": "seven",
    "acht": "eight",
    "neun": "nine",
    "zehn": "ten",
    "elf": "eleven",
    "zwoelf": "twelve",
    "dreizehn": "thirteen",
    "vierzehn": "fourteen",
    "fuenfzehn": "fifteen",
    "sechzehn": "sixteen",
}
_WRAPPER_RE = re.compile(
    r"^(?:(?:what\s+is)|(?:was\s+ist)|calculate|compute|berechne)\s*:?\s*",
    re.IGNORECASE,
)


class S3IntegrityError(RuntimeError):
    """A persisted host/organ failed its deployment integrity contract."""


class _OutsideGrammar(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _Operand:
    value: int
    digits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Expression:
    operands: tuple[_Operand, ...]
    operators: tuple[str, ...]

    @property
    def values(self) -> tuple[int, ...]:
        return tuple(item.value for item in self.operands)


@dataclass(frozen=True, slots=True)
class _LoadedOrgan:
    name: str
    module: Any
    sha256: str
    internal_digest: str
    metadata: Mapping[str, Any]
    descriptor: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class S3Case:
    """One deterministic case from the committed SHIP-v6 aggregate."""

    text: str
    expected: str
    route: str


def _emit_digits(value: int) -> str:
    return " ".join(_DIGITS[int(char)] for char in str(value))


def canonical_cases() -> tuple[S3Case, ...]:
    """Return the exact 152 arithmetic cases used by the SHIP-v6 aggregate."""

    cases: list[S3Case] = []
    cases.extend(
        S3Case(f"{_VALUE_WORD[a]} plus {_VALUE_WORD[b]} is", _VALUE_WORD[a + b], "ARITH")
        for a in range(2, 6)
        for b in range(2, 6)
    )
    cases.extend(
        S3Case(f"{_VALUE_WORD[a]} less {_VALUE_WORD[b]} is", _VALUE_WORD[a - b], "ARITH")
        for a in range(3, 9)
        for b in range(1, 3)
    )
    cases.extend(
        S3Case(f"{_VALUE_WORD[a]} times {_VALUE_WORD[b]} is", _VALUE_WORD[a * b], "MUL")
        for a in range(2, 9)
        for b in range(2, 9)
        if a * b <= 16
    )
    cases.extend(
        S3Case(f"{_VALUE_WORD[a]} z3sum {_VALUE_WORD[b]} is", _DIGITS[(a + b) % 3], "Z3SUM")
        for a in range(1, 10)
        for b in range(1, 10)
    )
    generator = random.Random(12)  # SHIP-v6: SEED(7) + 5
    for _ in range(24):
        left = generator.randint(10, 99)
        right = generator.randint(2, 9)
        operator = generator.choice(("times", "plus"))
        value = left * right if operator == "times" else left + right
        left_words = " ".join(_DIGITS[int(char)] for char in str(left))
        cases.append(
            S3Case(
                f"{left_words} {operator} {_DIGITS[right]} is",
                _emit_digits(value),
                "DECIMAL",
            )
        )
    if len(cases) != 152:  # deployment invariant, not a test-only assertion
        raise AssertionError(f"SHIP-v6 canonical suite changed size: {len(cases)}")
    return tuple(cases)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_digest(state_dict: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in state_dict.items():
        if not hasattr(value, "detach"):
            raise S3IntegrityError(f"non-tensor entry in organ state_dict: {name}")
        digest.update(str(name).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


@lru_cache(maxsize=1)
def _organ_types() -> Mapping[str, type]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - exercised without neural extra
        raise RuntimeError("SHIP-v6 needs torch; install immer with '.[neural]'") from exc

    class DualOrgan(nn.Module):
        def __init__(self, e_less: Any, e_plus: Any) -> None:
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
            self.Rs = nn.Linear(1, 8)
            self.Ra = nn.Linear(1, 7)
            self.Pg = nn.Linear(128, 16, bias=False)

            def gate() -> Any:
                return nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))

            self.gs = gate()
            self.ga = gate()
            self.register_buffer("e_less", e_less / e_less.norm())
            self.register_buffer("e_plus", e_plus / e_plus.norm())

        def _route(self, embeddings: Any, reference: Any) -> Any:
            normalised = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            route = ((normalised * reference).sum(-1, keepdim=True) > 0.999).float()
            zero = torch.zeros_like(route[:, :1])
            return torch.cat([zero, zero, route[:, :-2]], dim=1)

        def forward(self, embeddings: Any) -> tuple[Any, Any, Any, Any]:
            values = self.phi(embeddings)
            zero = torch.zeros_like(values[:, :1])
            previous = torch.cat([zero, values[:, :-1]], dim=1)
            third_previous = torch.cat([zero, zero, zero, values[:, :-3]], dim=1)
            sub_delta = self.Rs(third_previous - previous)
            add_delta = self.Ra(third_previous + previous)
            projected = self.Pg(embeddings)
            p1 = torch.cat([torch.zeros_like(projected[:, :1]), projected[:, :-1]], dim=1)
            p2 = torch.cat([torch.zeros_like(projected[:, :1]), p1[:, :-1]], dim=1)
            p3 = torch.cat([torch.zeros_like(projected[:, :1]), p2[:, :-1]], dim=1)
            context = torch.cat([p3, p2, p1, projected], dim=-1)
            sub_gate = torch.sigmoid(self.gs(context)) * self._route(embeddings, self.e_less)
            add_gate = torch.sigmoid(self.ga(context)) * self._route(embeddings, self.e_plus)
            return sub_delta, sub_gate, add_delta, add_gate

    class MulOrgan(nn.Module):
        def __init__(self, e_times: Any, e_is: Any) -> None:
            super().__init__()
            self.P = nn.Linear(128, 32, bias=False)
            self.phi = nn.Sequential(nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
            self.register_buffer("e_times", e_times / e_times.norm())
            self.register_buffer("e_is", e_is / e_is.norm())
            # The learned scaffold used candidates 1..10.  Deployment extends
            # its verified log lattice to 1..16, exactly as SHIP-v6 did.
            self.register_buffer("cs", torch.zeros(10))
            self.s0 = nn.Parameter(torch.tensor(1.0))
            self.q = nn.Parameter(torch.tensor(0.5))

        def values(self, embeddings: Any) -> Any:
            return self.phi(torch.tanh(self.P(embeddings))).squeeze(-1)

    class Z3Organ(nn.Module):
        def __init__(self, e_remainder: Any) -> None:
            super().__init__()
            self.P = nn.Linear(128, 32, bias=False)
            self.phi = nn.Sequential(nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
            self.register_buffer("e_rem", e_remainder / e_remainder.norm())
            self.s0 = nn.Parameter(torch.tensor(3.0))

        def forward(self, embeddings: Any) -> Any:
            values = self.phi(torch.tanh(self.P(embeddings))).squeeze(-1)
            total = values[:, 0] + values[:, 2]
            classes = torch.arange(3, dtype=total.dtype, device=total.device)
            return self.s0 * torch.cos(2 * math.pi / 3 * (total[:, None] - classes[None]))

    class DecimalOrgan(nn.Module):
        def __init__(self, e_plus: Any, e_less: Any, e_is: Any) -> None:
            super().__init__()
            self.P = nn.Linear(128, 32, bias=False)
            self.phi = nn.Sequential(nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
            self.register_buffer("e_plus", e_plus / e_plus.norm())
            self.register_buffer("e_less", e_less / e_less.norm())
            self.register_buffer("e_is", e_is / e_is.norm())
            self.register_buffer("cs", torch.arange(1, 17).float())
            self.s0 = nn.Parameter(torch.tensor(1.0))
            self.q = nn.Parameter(torch.tensor(0.5))

        def values(self, embeddings: Any) -> Any:
            return self.phi(torch.tanh(self.P(embeddings))).squeeze(-1)

    return {
        "arith-dual": DualOrgan,
        "mul-log": MulOrgan,
        "z3-circle": Z3Organ,
        "decimal-crystal": DecimalOrgan,
    }


class S3Arithmetic:
    """Digest-addressed, cold SHIP-v6 arithmetic component.

    Construction is side-effect free.  The 71 MB A1 checkpoint and an organ
    are loaded only after a request has passed the explicit grammar gate.
    """

    name = "s3.ship-v6.arithmetic"
    capabilities = frozenset({"exact_math"})

    def __init__(
        self,
        manifest: str | Path | None = None,
        *,
        router_state: str | Path | None = None,
        artifact_root: str | Path | None = None,
    ) -> None:
        self.manifest_path = (
            Path(manifest).expanduser().resolve()
            if manifest is not None
            else s3_ship_manifest()
        )
        self.artifact_root = select_artifact_root(self.manifest_path, artifact_root)
        self._manifest: Mapping[str, Any] | None = None
        self._host: Any | None = None
        self._host_sha256: str | None = None
        self._organs: dict[str, _LoadedOrgan] = {}
        self._bank: OrganBank | None = None
        self._verified_organ_paths: Mapping[str, Path] | None = None
        self.router_state = (
            Path(router_state).expanduser().resolve()
            if router_state is not None
            else crsa_router_manifest()
        )
        self._router: Any | None = None
        self._lock = threading.RLock()

    @property
    def frozen_host(self) -> Any:
        """The lazy, deployment-only A1 host (never the living stream)."""

        return self._load_host()

    def solve(self, text: str) -> str | None:
        result = self.handle(Request("exact_math", text))
        return str(result.output) if result.ok else None

    def canonical_cases(self) -> tuple[S3Case, ...]:
        return canonical_cases()

    def benchmark(self) -> Mapping[str, Any]:
        """Run the cold 152-case aggregate without any training or research import."""

        started = time.perf_counter()
        correct = route_correct = ok = 0
        cases = canonical_cases()
        for case in cases:
            result = self.handle(Request("exact_math", case.text))
            ok += result.ok
            correct += result.ok and result.output == case.expected
            route_correct += result.ok and result.evidence.get("route") == case.route
        router = self._load_router()
        return {
            "cases": len(cases),
            "ok": int(ok),
            "correct": int(correct),
            "route_correct": int(route_correct),
            "runtime_s": round(time.perf_counter() - started, 4),
            "host_sha256": self._host_sha256,
            "host_arm": self._data()["host"]["arm"],
            "attention_program": "2 Local + 1 Balanced + 1 bit-exact Free",
            "router_state_digest": router.head.digest(),
            "router_verdict": router.head.metadata.get("measurement", {}).get("verdict"),
            "no_training": True,
        }

    def handle(self, request: Request) -> Result:
        if request.capability not in self.capabilities:
            return Result(ExecutionStatus.REJECTED, self.name, reason="unsupported capability")
        if not isinstance(request.payload, str) or not request.payload.strip():
            return Result(
                ExecutionStatus.REJECTED,
                self.name,
                reason="exact_math payload must be non-empty text",
            )
        try:
            expression = self._parse(request.payload)
            route = self._select_route(expression)
            attention = self._attention_route(expression)
            if not attention["is_arithmetic"]:
                return Result(
                    ExecutionStatus.ABSTAINED,
                    self.name,
                    reason="the measured CRSA router selected the text path",
                    evidence={"route": "TEXT", "attention": attention},
                )
            answer, organ, route_evidence = self._execute(route, expression)
        except _OutsideGrammar as exc:
            return Result(ExecutionStatus.ABSTAINED, self.name, reason=str(exc))
        except (FileNotFoundError, ImportError, RuntimeError) as exc:
            status = ExecutionStatus.ERROR if isinstance(exc, S3IntegrityError) else ExecutionStatus.UNAVAILABLE
            return Result(status, self.name, reason=str(exc))
        except (KeyError, TypeError, ValueError) as exc:
            return Result(
                ExecutionStatus.ERROR,
                self.name,
                reason=f"invalid S3 manifest or artifact: {exc}",
            )

        descriptor_metric = organ.descriptor.get("crystal_metric", {})
        evidence: dict[str, Any] = {
            "route": route,
            "organ": organ.name,
            "artifact_sha256": organ.sha256,
            "internal_digest": organ.internal_digest,
            "crystal_metric": descriptor_metric,
            "crystal_verified": True,
            "no_training": True,
            "numeric_value": route_evidence.pop("numeric_value"),
            "host": {
                "name": self._data()["host"]["name"],
                "arm": self._data()["host"]["arm"],
                "format": self._data()["host"].get(
                    "format", "torch-checkpoint-arms"
                ),
                "sha256": self._host_sha256,
                "frozen": True,
                "separate_from_life_stream": True,
            },
            "attention": attention,
            **route_evidence,
        }
        return Result(ExecutionStatus.OK, self.name, output=answer, evidence=evidence)

    def _load_router(self) -> Any:
        if self._router is not None:
            return self._router
        with self._lock:
            if self._router is not None:
                return self._router
            from ..attention.router import FrozenA1CrsaRouter

            if not self.router_state.is_file():
                raise FileNotFoundError(f"missing CRSA router state: {self.router_state}")
            self._router = FrozenA1CrsaRouter.load(self.router_state)
            return self._router

    def _attention_route(self, expression: _Expression) -> dict[str, Any]:
        import torch

        words: list[str] = []
        for index, operand in enumerate(expression.operands):
            if index:
                operator = expression.operators[index - 1]
                words.append("remainder" if operator == "z3sum" else operator)
            words.extend(operand.digits)
        words.append("is")
        ids = torch.tensor(
            [[self._tokens()[word] for word in words]],
            dtype=torch.long,
        )
        router = self._load_router()
        decision = router.decide_one(self._load_host(), ids)
        measurement = router.head.metadata.get("measurement", {})
        return {
            "operator": "role_complete",
            "program": "2 Local + 1 Balanced + 1 bit-exact Free",
            "feature_schema": router.head.feature_schema,
            "state_digest": router.head.digest(),
            "label": decision.label,
            "score": decision.score,
            "margin": decision.margin,
            "is_arithmetic": decision.is_arithmetic,
            "measured_verdict": measurement.get("verdict"),
            "crsa_specific_advantage_over_softmax": measurement.get(
                "crsa_specific_advantage_over_softmax"
            ),
        }

    def _data(self) -> Mapping[str, Any]:
        if self._manifest is not None:
            return self._manifest
        with self._lock:
            if self._manifest is not None:
                return self._manifest
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping) or raw.get("schema") != _SCHEMA:
                raise S3IntegrityError(f"unsupported S3 manifest schema in {self.manifest_path}")
            raw_organs = raw.get("organs")
            if not isinstance(raw_organs, list):
                raise S3IntegrityError("S3 manifest must contain an OrganBank-compatible organs list")
            s3 = raw.get("s3")
            if not isinstance(s3, Mapping):
                raise S3IntegrityError("S3 manifest is missing its deployment section")
            for required in ("host", "vocabulary", "verification"):
                if not isinstance(s3.get(required), Mapping):
                    raise S3IntegrityError(f"S3 deployment section is missing mapping {required!r}")
            tokens = s3["vocabulary"].get("tokens")
            if not isinstance(tokens, Mapping):
                raise S3IntegrityError("S3 manifest is missing its persisted vocabulary")
            required_tokens = set(_CARDINALS) | {"plus", "less", "times", "is", "remainder"}
            missing = sorted(required_tokens - set(tokens))
            if missing:
                raise S3IntegrityError(f"S3 vocabulary is incomplete: {missing}")
            organs: dict[str, Mapping[str, Any]] = {}
            for raw_organ in raw_organs:
                if not isinstance(raw_organ, Mapping):
                    raise S3IntegrityError("S3 organ descriptor is not a mapping")
                name = str(raw_organ.get("name", ""))
                metadata = raw_organ.get("metadata", {})
                if not isinstance(metadata, Mapping):
                    raise S3IntegrityError(f"metadata for {name!r} is not a mapping")
                if name in organs:
                    raise S3IntegrityError(f"duplicate S3 organ {name!r}")
                organs[name] = {**raw_organ, **metadata}
            if set(organs) != {"arith-dual", "mul-log", "z3-circle", "decimal-crystal"}:
                raise S3IntegrityError("S3 manifest must name exactly the four SHIP-v6 organs")
            bank = OrganBank.from_manifest(
                self.manifest_path,
                artifact_root=self.artifact_root,
            )
            if set(bank.names()) != set(organs):
                raise S3IntegrityError("OrganBank and S3 deployment descriptors disagree")
            self._bank = bank
            self._manifest = {
                "host": s3["host"],
                "vocabulary": s3["vocabulary"],
                "verification": s3["verification"],
                "organs": organs,
            }
            return self._manifest

    def _artifact_path(self, descriptor: Mapping[str, Any]) -> Path:
        raw = descriptor.get("artifact")
        if not isinstance(raw, str) or not raw:
            raise S3IntegrityError("artifact descriptor has no path")
        return resolve_artifact_path(
            self.manifest_path,
            raw,
            configured_root=self.artifact_root,
        )

    def _verified_path(self, descriptor: Mapping[str, Any], label: str) -> tuple[Path, str]:
        path = self._artifact_path(descriptor)
        if not path.is_file():
            raise FileNotFoundError(f"missing {label} artifact: {path}")
        expected = str(descriptor.get("sha256", "")).lower()
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise S3IntegrityError(f"invalid SHA-256 in manifest for {label}")
        actual = _sha256(path)
        if actual != expected:
            raise S3IntegrityError(
                f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
            )
        return path, actual

    def _load_host(self) -> Any:
        if self._host is not None:
            return self._host
        with self._lock:
            if self._host is not None:
                return self._host
            try:
                import torch
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("SHIP-v6 needs torch; install immer with '.[neural]'") from exc

            host_descriptor = self._data()["host"]
            path, actual_sha = self._verified_path(host_descriptor, "frozen A1 host")
            host_format = str(
                host_descriptor.get("format", "torch-checkpoint-arms")
            )
            arm = str(host_descriptor.get("arm", ""))
            if host_format == "torch-checkpoint-arms":
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(checkpoint, Mapping):
                    raise S3IntegrityError("A1 checkpoint is not a mapping")
                arms = checkpoint.get("arms")
                if not isinstance(arms, Mapping) or arm not in arms:
                    raise S3IntegrityError(f"A1 checkpoint has no arm {arm!r}")
                arm_record = arms[arm]
                if not isinstance(arm_record, Mapping) or not isinstance(
                    arm_record.get("model"), Mapping
                ):
                    raise S3IntegrityError(f"A1 arm {arm!r} has no model state")
                model_state = arm_record["model"]
            elif host_format == "safetensors-state-dict":
                try:
                    from safetensors.torch import load_file
                except ImportError as exc:  # pragma: no cover - optional export runtime
                    raise RuntimeError(
                        "safetensors host needs the export extra; install "
                        "immer with '.[neural,export]'"
                    ) from exc
                model_state = load_file(str(path), device="cpu")
                if not isinstance(model_state, Mapping) or not model_state:
                    raise S3IntegrityError("A1 safetensors file has no model state")
            else:
                raise S3IntegrityError(f"unsupported A1 host format {host_format!r}")

            model_config = host_descriptor.get("model")
            if not isinstance(model_config, Mapping):
                raise S3IntegrityError("A1 manifest has no model configuration")
            host_class = self._host_class()
            host = host_class(
                int(model_config["vocab_size"]),
                int(model_config["mask_id"]),
                d_model=int(model_config["d_model"]),
                n_layers=int(model_config["n_layers"]),
                n_heads=int(model_config["n_heads"]),
                d_head=int(model_config["d_head"]),
                seq_len=int(model_config["seq_len"]),
                dropout=0.0,
                causal=True,
            )
            host.load_state_dict(model_state, strict=True)
            host.eval()
            for parameter in host.parameters():
                parameter.requires_grad_(False)
            if any(parameter.requires_grad for parameter in host.parameters()):
                raise S3IntegrityError("frozen A1 host unexpectedly has trainable parameters")

            token_ids = [int(value) for value in self._tokens().values()]
            if min(token_ids) < 0 or max(token_ids) >= host.embed.num_embeddings:
                raise S3IntegrityError("persisted vocabulary does not fit the A1 embedding table")
            self._host = host
            self._host_sha256 = actual_sha
            return host

    def _host_class(self) -> type:
        from ..runtimes.o1_state.model import SelectiveNoPETransformerLM

        return SelectiveNoPETransformerLM

    def _tokens(self) -> Mapping[str, int]:
        raw = self._data()["vocabulary"]["tokens"]
        return {str(key): int(value) for key, value in raw.items()}

    def _load_organ(self, name: str) -> _LoadedOrgan:
        cached = self._organs.get(name)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._organs.get(name)
            if cached is not None:
                return cached
            try:
                import torch
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("SHIP-v6 needs torch; install immer with '.[neural]'") from exc

            descriptor = self._data()["organs"][name]
            if self._bank is None:
                raise S3IntegrityError("S3 OrganBank was not initialised")
            if self._verified_organ_paths is None:
                try:
                    self._verified_organ_paths = self._bank.verify_all()
                except DigestMismatch as exc:
                    raise S3IntegrityError(str(exc)) from exc
            path = self._verified_organ_paths[name]
            actual_sha = self._bank.descriptor(name).sha256
            bundle = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(bundle, Mapping) or not isinstance(bundle.get("state_dict"), Mapping):
                raise S3IntegrityError(f"{name} artifact has no state_dict")
            state_dict = bundle["state_dict"]
            expected_internal = str(descriptor.get("internal_digest", ""))
            stored_internal = str(bundle.get("digest", ""))
            computed_internal = _state_digest(state_dict)
            if stored_internal != expected_internal or computed_internal != expected_internal:
                raise S3IntegrityError(
                    f"{name} internal digest mismatch: manifest={expected_internal}, "
                    f"stored={stored_internal}, computed={computed_internal}"
                )

            host = self._load_host()
            embeddings = host.embed.weight.detach()
            tokens = self._tokens()
            organ_type = _organ_types()[name]
            if name == "arith-dual":
                module = organ_type(embeddings[tokens["less"]].clone(), embeddings[tokens["plus"]].clone())
            elif name == "mul-log":
                module = organ_type(embeddings[tokens["times"]].clone(), embeddings[tokens["is"]].clone())
            elif name == "z3-circle":
                module = organ_type(embeddings[tokens["remainder"]].clone())
            else:
                module = organ_type(
                    embeddings[tokens["plus"]].clone(),
                    embeddings[tokens["less"]].clone(),
                    embeddings[tokens["is"]].clone(),
                )
            module.load_state_dict(state_dict, strict=True)
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            metadata = bundle.get("meta", {})
            if not isinstance(metadata, Mapping):
                raise S3IntegrityError(f"{name} metadata is not a mapping")
            loaded = _LoadedOrgan(
                name=name,
                module=module,
                sha256=actual_sha,
                internal_digest=computed_internal,
                metadata=metadata,
                descriptor=descriptor,
            )
            self._organs[name] = loaded
            return loaded

    def _parse(self, text: str) -> _Expression:
        lowered = (
            text.strip()
            .lower()
            .replace("×", "*")
            .replace("−", "-")
            .replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("ß", "ss")
        )
        lowered = _WRAPPER_RE.sub("", lowered, count=1)
        tokens = [
            _WORD_ALIASES.get(match.group(0).lower(), match.group(0).lower())
            for match in _TOKEN_RE.finditer(lowered)
        ]
        residue = _TOKEN_RE.sub("", lowered)
        if residue.strip(" \t\r\n.,?!"):
            raise _OutsideGrammar("outside the explicit SHIP-v6 arithmetic grammar")
        if not tokens:
            raise _OutsideGrammar("outside the explicit SHIP-v6 arithmetic grammar")
        forbidden = _FORBIDDEN_MODULO.intersection(tokens)
        if forbidden:
            raise _OutsideGrammar(
                "ordinary modulo/remainder is unsupported; the Z3 organ is exposed only as z3sum"
            )
        if tokens[-1] in {"is", "="}:
            tokens.pop()
        if not tokens or "is" in tokens or "=" in tokens:
            raise _OutsideGrammar("'is' or '=' is allowed only as a final terminator")

        chunks: list[list[str]] = [[]]
        operators: list[str] = []
        for token in tokens:
            operator = _PUBLIC_OPERATORS.get(token)
            if operator is None:
                chunks[-1].append(token)
            else:
                if not chunks[-1]:
                    raise _OutsideGrammar("operator without a left operand")
                operators.append(operator)
                chunks.append([])
        if not operators or not chunks[-1] or len(chunks) != len(operators) + 1:
            raise _OutsideGrammar("expected an explicit arithmetic expression")
        if "z3sum" in operators and (len(operators) != 1 or operators[0] != "z3sum"):
            raise _OutsideGrammar("z3sum accepts exactly two operands")
        ordinary = [operator for operator in operators if operator != "z3sum"]
        if ordinary and len(set(ordinary)) != 1:
            raise _OutsideGrammar("mixed arithmetic operators are outside the verified grammar")
        if len(chunks) > 32:
            raise _OutsideGrammar("expression exceeds the verified 32-operand deployment bound")

        operands = tuple(self._parse_operand(chunk) for chunk in chunks)
        return _Expression(operands=operands, operators=tuple(operators))

    def _parse_operand(self, raw: Sequence[str]) -> _Operand:
        if len(raw) == 1 and raw[0].isdigit():
            if len(raw[0]) > 128:
                raise _OutsideGrammar("operand exceeds the verified 128-digit deployment bound")
            canonical = raw[0].lstrip("0") or "0"
            return _Operand(int(canonical), tuple(_DIGITS[int(char)] for char in canonical))
        if len(raw) == 1 and raw[0] in _CARDINAL_VALUE:
            value = _CARDINAL_VALUE[raw[0]]
            return _Operand(value, tuple(_DIGITS[int(char)] for char in str(value)))
        if raw and all(token in _DIGITS for token in raw):
            if len(raw) > 128:
                raise _OutsideGrammar("operand exceeds the verified 128-digit deployment bound")
            digits = tuple(raw)
            canonical = "".join(str(_CARDINAL_VALUE[word]) for word in digits).lstrip("0") or "0"
            return _Operand(int(canonical), tuple(_DIGITS[int(char)] for char in canonical))
        raise _OutsideGrammar("operands must be digits or canonical English number/digit words")

    def _select_route(self, expression: _Expression) -> str:
        values = expression.values
        operators = expression.operators
        if operators == ("z3sum",):
            if not all(1 <= value <= 9 for value in values):
                raise _OutsideGrammar("z3sum carrier is restricted to one..nine")
            return "Z3SUM"
        if all(operator == "times" for operator in operators):
            product = math.prod(values)
            if all(1 <= value <= 8 for value in values) and 1 <= product <= 16:
                return "MUL"
            return "DECIMAL"
        if len(values) == 2 and operators in (("plus",), ("less",)):
            left, right = values
            if operators == ("plus",) and 2 <= left <= 5 and 2 <= right <= 5:
                return "ARITH"
            if operators == ("less",) and 3 <= left <= 8 and 1 <= right <= 2:
                return "ARITH"
        return "DECIMAL"

    def _execute(
        self, route: str, expression: _Expression
    ) -> tuple[str, _LoadedOrgan, dict[str, Any]]:
        if route == "ARITH":
            return self._execute_dual(expression)
        if route == "MUL":
            return self._execute_mul(expression)
        if route == "Z3SUM":
            return self._execute_z3(expression)
        return self._execute_decimal(expression)

    def _word_for_small(self, value: int) -> str:
        try:
            return _VALUE_WORD[value]
        except KeyError as exc:
            raise _OutsideGrammar("result is outside the organ candidate lattice") from exc

    def _execute_dual(self, expression: _Expression) -> tuple[str, _LoadedOrgan, dict[str, Any]]:
        import torch
        import torch.nn.functional as functional

        organ = self._load_organ("arith-dual")
        left, right = expression.values
        operator = expression.operators[0]
        value = left + right if operator == "plus" else left - right
        expected = self._word_for_small(value)
        words = [self._word_for_small(left), operator, self._word_for_small(right), "is"]
        ids = torch.tensor([[self._tokens()[word] for word in words]], dtype=torch.long)
        host = self._load_host()
        with torch.no_grad():
            logits = host(ids)
            embeddings = host.embed(ids)
            sub_delta, sub_gate, add_delta, add_gate = organ.module(embeddings)
            output = logits.clone()
            if operator == "less":
                candidates = [int(value) for value in organ.metadata["sub_cand_ids"]]
                candidate_words = [str(value) for value in organ.metadata["sub_cand_words"]]
                output[..., candidates] += sub_gate * sub_delta
            else:
                candidates = [int(value) for value in organ.metadata["add_cand_ids"]]
                candidate_words = [str(value) for value in organ.metadata["add_cand_words"]]
                output[..., candidates] += add_gate * add_delta
            probabilities = functional.softmax(output[0, -1, candidates], dim=0)
            top2 = probabilities.topk(2)
            prediction = candidate_words[int(top2.indices[0])]
            margin = float(top2.values[0] - top2.values[1])
        if prediction != expected:
            raise S3IntegrityError(
                f"arith-dual disagreed with its exact readout: neural={prediction}, exact={expected}"
            )
        return expected, organ, {
            "numeric_value": value,
            "operator": operator,
            "organ_prediction": prediction,
            "margin": margin,
            "exact_readout_agreement": True,
        }

    def _execute_mul(self, expression: _Expression) -> tuple[str, _LoadedOrgan, dict[str, Any]]:
        import torch
        import torch.nn.functional as functional

        organ = self._load_organ("mul-log")
        words: list[str] = []
        for index, value in enumerate(expression.values):
            if index:
                words.append("times")
            words.append(self._word_for_small(value))
        words.append("is")
        ids = torch.tensor([[self._tokens()[word] for word in words]], dtype=torch.long)
        host = self._load_host()
        crystal = organ.metadata.get("kristall", {})
        alpha = float(crystal["alpha"])
        gamma = float(crystal["gamma"])
        lattice = torch.log(torch.arange(1, 17).float())
        with torch.no_grad():
            embeddings = host.embed(ids)
            normalised = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            is_operator = (
                (normalised @ organ.module.e_times > 0.99)
                | (normalised @ organ.module.e_is > 0.99)
            )
            mapped = (organ.module.values(embeddings) - gamma) / alpha
            snapped = lattice[(mapped[..., None] - lattice).abs().argmin(dim=-1)]
            accumulator = float(((~is_operator).float() * snapped).sum())
            scores = 10.0 * (lattice * accumulator - 0.5 * lattice.square())
            probabilities = functional.softmax(scores, dim=0)
            top2 = probabilities.topk(2)
            predicted_value = int(top2.indices[0]) + 1
            margin = float(top2.values[0] - top2.values[1])
        value = math.prod(expression.values)
        if predicted_value != value:
            raise S3IntegrityError(
                f"mul-log disagreed with its exact log readout: neural={predicted_value}, exact={value}"
            )
        return self._word_for_small(value), organ, {
            "numeric_value": value,
            "operator": "times",
            "organ_prediction": predicted_value,
            "margin": margin,
            "exact_readout_agreement": True,
        }

    def _execute_z3(self, expression: _Expression) -> tuple[str, _LoadedOrgan, dict[str, Any]]:
        import torch

        organ = self._load_organ("z3-circle")
        left, right = expression.values
        # Public z3sum is represented by the frozen carrier token the organ was
        # distilled on.  It is deliberately never exposed as remainder/modulo.
        words = [self._word_for_small(left), "remainder", self._word_for_small(right), "is"]
        ids = torch.tensor([[self._tokens()[word] for word in words]], dtype=torch.long)
        with torch.no_grad():
            scores = organ.module(self._load_host().embed(ids))
            predicted_value = int(scores[0].argmax())
        value = (left + right) % 3
        if predicted_value != value:
            raise S3IntegrityError(
                f"z3-circle disagreed with exact Z3 addition: neural={predicted_value}, exact={value}"
            )
        return _DIGITS[value], organ, {
            "numeric_value": value,
            "operator": "z3sum",
            "semantics": "(a + b) mod 3",
            "ordinary_modulo": False,
            "organ_prediction": predicted_value,
            "exact_readout_agreement": True,
        }

    def _execute_decimal(
        self, expression: _Expression
    ) -> tuple[str, _LoadedOrgan, dict[str, Any]]:
        import torch

        if expression.operators[0] == "less":
            value = expression.values[0]
            for operand in expression.values[1:]:
                value -= operand
        elif expression.operators[0] == "plus":
            value = sum(expression.values)
        elif expression.operators[0] == "times":
            value = math.prod(expression.values)
        else:  # protected by the parser/router
            raise _OutsideGrammar("unsupported decimal operation")
        if value < 0:
            raise _OutsideGrammar("negative decimal emission is outside the verified carrier")
        rendered = str(value)
        if len(rendered) > 512:
            raise _OutsideGrammar("result exceeds the verified 512-digit emission bound")

        organ = self._load_organ("decimal-crystal")
        crystal = organ.metadata.get("kristall", {})
        alpha = float(crystal["alpha"])
        gamma = float(crystal["gamma"])
        digit_ids = torch.tensor([[self._tokens()[word] for word in _DIGITS]], dtype=torch.long)
        with torch.no_grad():
            mapped = organ.module.values(self._load_host().embed(digit_ids))[0]
            snapped = torch.round((mapped - gamma) / alpha).to(torch.long)
        decoded = tuple(int(value) for value in snapped.tolist())
        if decoded != tuple(range(10)):
            raise S3IntegrityError(f"decimal crystal carrier failed self-check: {decoded}")

        # Parse each operand through the learned carrier followed by the exact
        # h <- 10h + v recurrence.  Algebra and emission are then structural;
        # the persisted generic 1..16 readout is intentionally not used.
        parsed_values: list[int] = []
        for operand in expression.operands:
            accumulator = 0
            for digit_word in operand.digits:
                accumulator = 10 * accumulator + decoded[_CARDINAL_VALUE[digit_word]]
            parsed_values.append(accumulator)
        if tuple(parsed_values) != expression.values:
            raise S3IntegrityError("decimal crystal parser disagreed with canonical input")

        answer = _emit_digits(value)
        return answer, organ, {
            "numeric_value": value,
            "operator": expression.operators[0],
            "parser_recurrence": "h <- 10h + v",
            "parsed_operands": parsed_values,
            "emission": "exact digit-word",
            "generic_forward_used": False,
            "exact_readout_agreement": True,
        }


__all__ = ["S3Arithmetic", "S3Case", "S3IntegrityError", "canonical_cases"]
