"""Grounded natural-language interface for FERTIG's small HSSLM model.

HSSLM is deliberately used as a *constrained chooser*.  Tasks, actions and
facts are supplied by the skill store and the observed world; the language
model may rank surface forms but cannot create a new executable capability.
This is the useful operating point for the shipped 2.2M-parameter checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / "data" / "hsslm_form.pt"
DEFAULT_TOKENIZER = ROOT / "data" / "hsslm_bpe.json"

UNKNOWN = "unknown"
AMBIGUOUS = "ambiguous"
RESOLVED = "resolved"


@dataclass(frozen=True)
class HSSLMStatus:
    checkpoint: str
    tokenizer: str
    ready: bool
    parameter_count: int
    serialized_parameter_count: int
    size_bytes: int
    device: str
    reason: str = ""


class HSSLMRuntime:
    """Lazy runtime around the existing HSSLM-C checkpoint.

    Merely constructing this object imports neither torch nor the model.  The
    mutable causal-inference cache inside HSSLM-C is reset around every score,
    making candidate ranking independent of evaluation order.
    """

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        tokenizer: str | Path = DEFAULT_TOKENIZER,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.tokenizer = Path(tokenizer)
        self._engine = None
        self._status: HSSLMStatus | None = None

    @staticmethod
    def _counts(state: Mapping[str, object]) -> tuple[int, int]:
        serialized = 0
        unique = 0
        seen: set[tuple[int, int, int]] = set()
        for value in state.values():
            if not hasattr(value, "numel"):
                continue
            count = int(value.numel())
            serialized += count
            try:
                storage = value.untyped_storage()
                key = (int(storage.data_ptr()), int(value.storage_offset()), count)
            except (AttributeError, RuntimeError):
                key = (id(value), 0, count)
            if key not in seen:
                seen.add(key)
                unique += count
        return unique, serialized

    def status(self) -> HSSLMStatus:
        if self._status is not None:
            return self._status
        reason = ""
        ready = False
        unique = serialized = 0
        device = "unloaded"
        size = self.checkpoint.stat().st_size if self.checkpoint.is_file() else 0
        if not self.checkpoint.is_file():
            reason = f"checkpoint missing: {self.checkpoint}"
        elif not self.tokenizer.is_file():
            reason = f"tokenizer missing: {self.tokenizer}"
        else:
            try:
                import torch

                try:
                    state = torch.load(
                        self.checkpoint, map_location="cpu", weights_only=True
                    )
                except TypeError:  # older torch
                    state = torch.load(self.checkpoint, map_location="cpu")
                if not isinstance(state, Mapping):
                    raise ValueError("checkpoint is not a state dictionary")
                unique, serialized = self._counts(state)
                raw = json.loads(self.tokenizer.read_text(encoding="utf-8"))
                if not all(key in raw for key in ("merges", "vocab", "itos")):
                    raise ValueError("tokenizer has no BPE tables")
                ready = unique > 0
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            except Exception as exc:  # status must remain a usable fallback
                reason = f"HSSLM unavailable: {type(exc).__name__}: {exc}"
        self._status = HSSLMStatus(
            str(self.checkpoint),
            str(self.tokenizer),
            ready,
            unique,
            serialized,
            size,
            device,
            reason,
        )
        return self._status

    def _ensure_engine(self):
        status = self.status()
        if not status.ready:
            raise RuntimeError(status.reason or "HSSLM is unavailable")
        if self._engine is None:
            from .form_engine import FormEngine

            engine = FormEngine(str(self.checkpoint), str(self.tokenizer))
            if not engine.ready:
                raise RuntimeError("HSSLM checkpoint could not be loaded")
            self._engine = engine
        return self._engine

    @staticmethod
    def _reset_inference(engine) -> None:
        """Replace HSSLM-C's request-mutable symbolic cache."""

        model = engine.model
        old = getattr(model, "inference_engine", None)
        amplifier = getattr(model, "signal_amplifier", None)
        if old is None or amplifier is None:
            return
        from .hsslm.causal_inference import CausalInferenceEngine

        fresh = CausalInferenceEngine(
            vocab_size=int(old.vocab_size),
            token_id_to_str=dict(old.token_id_to_str),
            quality_threshold=float(old.quality_threshold),
        )
        model.inference_engine = fresh
        amplifier.engine = fresh

    def text_coverage(self, text: str) -> float:
        """Fraction of non-space characters represented by the fitted BPE."""

        if not text:
            return 0.0
        try:
            raw = json.loads(self.tokenizer.read_text(encoding="utf-8"))
            chars = {token for token in raw["vocab"] if len(token) == 1}
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return 0.0
        material = [char for char in text if not char.isspace()]
        if not material:
            return 1.0
        return sum(char in chars for char in material) / len(material)

    def score(self, text: str) -> float:
        """Mean next-token log-probability for one candidate."""

        candidate = str(text).strip()
        if not candidate:
            raise ValueError("cannot score empty text")
        status = self.status()
        if not status.ready:
            raise RuntimeError(status.reason or "HSSLM is unavailable")
        if self.text_coverage(candidate) < 0.75:
            raise ValueError("text is outside the fitted HSSLM character vocabulary")
        engine = self._ensure_engine()
        torch = engine.torch
        self._reset_inference(engine)
        try:
            ids = engine._encode(candidate).to(engine.device)
            if ids.shape[1] < 2:
                raise ValueError("candidate produces fewer than two BPE tokens")
            with torch.no_grad():
                logits = engine.model.forward(ids)["logits"][0, :-1]
                targets = ids[0, 1:]
                log_probs = torch.log_softmax(logits, dim=-1)
                positions = torch.arange(targets.numel(), device=targets.device)
                return float(log_probs[positions, targets].mean().cpu())
        finally:
            self._reset_inference(engine)

    def choose(self, candidates: Iterable[str]) -> str:
        values = tuple(dict.fromkeys(str(value).strip() for value in candidates))
        if not values or any(not value for value in values):
            raise ValueError("choose requires non-empty text candidates")
        ranked = sorted(
            ((self.score(value), value) for value in values),
            key=lambda item: (-item[0], item[1]),
        )
        return ranked[0][1]

    def complete(
        self, prompt: str, *, variants: int = 3, max_new_tokens: int = 48
    ) -> tuple[str, ...]:
        """Expose generation for experiments; executable routing never uses it."""

        if variants <= 0 or max_new_tokens <= 0:
            raise ValueError("generation limits must be positive")
        engine = self._ensure_engine()
        self._reset_inference(engine)
        try:
            return tuple(engine.variants(prompt, variants, max_new_tokens))
        finally:
            self._reset_inference(engine)


@dataclass(frozen=True)
class SkillDescriptor:
    name: str
    aliases: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        name = _normalise(self.name)
        if not name:
            raise ValueError("skill name must not be empty")
        aliases = tuple(
            dict.fromkeys(
                value for alias in self.aliases if (value := _normalise(alias))
            )
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "aliases", aliases)


@dataclass(frozen=True)
class ResolvedSkill:
    intent: str
    status: str
    skill: SkillDescriptor | None
    task: str | None
    confidence: float
    reason: str = ""


_INTENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("teach", ("teach", "learn", "record", "lerne", "lern", "zeige")),
    ("explain", ("explain", "describe", "warum", "erklär", "erklaer")),
    ("list", ("list tasks", "what can you do", "was kannst du", "aufgaben")),
    ("help", ("help", "hilfe")),
    ("status", ("status", "modellstatus", "hsslm")),
    ("do", ("do", "execute", "run", "perform", "mach", "mache", "führe", "fuehre")),
)


def _normalise(text: object) -> str:
    value = str(text).casefold().replace("ß", "ss")
    value = re.sub(r"[^\wäöü]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _intent(text: str) -> str:
    # Mutating intents require an imperative at the beginning.  A word such
    # as ``run`` inside "How fast can a cheetah run?" is information, not
    # permission to touch the desktop.
    prefixes = (
        ("teach", r"^(?:please |bitte )?(?:teach|learn|record|lerne|lern|zeige)\b"),
        (
            "do",
            r"^(?:please |bitte )?(?:do|execute|run|perform|mach|mache|führe|fuehre)\b",
        ),
    )
    for name, pattern in prefixes:
        if re.search(pattern, text):
            return name
    for name, markers in _INTENTS:
        if name in {"teach", "do"}:
            continue
        if any(marker in text for marker in markers):
            return name
    return "unknown"


def _strip_intent(text: str, intent: str) -> str:
    markers = next((items for name, items in _INTENTS if name == intent), ())
    result = text
    for marker in sorted(markers, key=len, reverse=True):
        result = re.sub(rf"\b{re.escape(marker)}\b", " ", result)
    noise = (
        "please",
        "bitte",
        "can you",
        "kannst du",
        "for me",
        "für mich",
        "mir",
        "die aufgabe",
        "task",
        "jetzt",
    )
    for marker in noise:
        result = result.replace(marker, " ")
    return " ".join(result.split())


def _similarity(query: str, candidate: str) -> float:
    if not query or not candidate:
        return 0.0
    if candidate in query or query in candidate:
        return 1.0 if candidate == query else 0.94
    left, right = set(query.split()), set(candidate.split())
    overlap = len(left & right) / max(1, len(left | right))
    sequence = SequenceMatcher(None, query, candidate).ratio()
    return float(0.65 * overlap + 0.35 * sequence)


class GroundedLanguageInterface:
    """Natural-language routing over a finite set of actually learned skills."""

    def __init__(self, runtime: HSSLMRuntime | None = None) -> None:
        self.runtime = runtime

    def resolve(
        self, text: str, skills: Sequence[SkillDescriptor | str]
    ) -> ResolvedSkill:
        normal = _normalise(text)
        if not normal:
            return ResolvedSkill("unknown", UNKNOWN, None, None, 0.0, "empty request")
        intent = _intent(normal)
        if intent in {"list", "help", "status"}:
            return ResolvedSkill(intent, RESOLVED, None, None, 1.0)
        query = _strip_intent(normal, intent)
        descriptors = tuple(
            skill if isinstance(skill, SkillDescriptor) else SkillDescriptor(skill)
            for skill in skills
        )
        if intent == "teach":
            if not query:
                return ResolvedSkill(
                    intent, UNKNOWN, None, None, 0.0, "task name missing"
                )
            existing = next(
                (skill for skill in descriptors if skill.name == query), None
            )
            return ResolvedSkill(intent, RESOLVED, existing, query, 1.0)
        scored: list[tuple[float, SkillDescriptor]] = []
        for skill in descriptors:
            names = (skill.name, *skill.aliases)
            scored.append((max(_similarity(query, name) for name in names), skill))
        scored.sort(key=lambda item: (-item[0], item[1].name))
        if not scored or scored[0][0] < 0.42:
            return ResolvedSkill(
                intent, UNKNOWN, None, query or None, 0.0, "no learned skill matches"
            )
        # A bare or politely phrased task name may imply execution, but only
        # after it has matched an actually learned skill. Unknown questions
        # stay available to the read-only math/quant/graph router.
        if intent == "unknown":
            intent = "do"
        top_score, top = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        if top_score - second < 0.08:
            # The narrow HSSLM may break a lexical tie, but only inside its
            # trained alphabet and only with a material score margin.
            if (
                self.runtime is not None
                and self.runtime.status().ready
                and self.runtime.text_coverage(normal) >= 0.9
            ):
                tied = [item for item in scored if top_score - item[0] < 0.08]
                candidates = {
                    f"The requested learned task is {skill.name}.": skill
                    for _, skill in tied
                }
                try:
                    ranked = sorted(
                        ((self.runtime.score(value), value) for value in candidates),
                        reverse=True,
                    )
                    if len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 0.05:
                        top = candidates[ranked[0][1]]
                        return ResolvedSkill(
                            intent,
                            RESOLVED,
                            top,
                            top.name,
                            top_score,
                            "HSSLM constrained rerank",
                        )
                except (RuntimeError, ValueError):
                    pass
            return ResolvedSkill(
                intent,
                AMBIGUOUS,
                None,
                query,
                top_score,
                "multiple learned skills match",
            )
        return ResolvedSkill(intent, RESOLVED, top, top.name, top_score)

    def render(self, event: str, facts: Mapping[str, object]) -> str:
        """Rank fact-preserving replies; never let the model invent facts."""

        clean = {str(key): str(value) for key, value in facts.items()}
        message = clean.get("message", "").strip()
        task = clean.get("task", "").strip()
        raw_steps = facts.get("steps")
        step_count = (
            len(raw_steps) if isinstance(raw_steps, (list, tuple)) else raw_steps
        )
        if message:
            candidates = [message]
            if event == "do" and clean.get("status") == "ok" and task:
                candidates.extend(
                    (
                        f"Done: {task} ({step_count} verified steps).",
                        f"I completed {task} in {step_count} verified steps.",
                    )
                )
            elif event == "teach" and task:
                candidates.append(
                    f"Learned {task} from {step_count} demonstrated steps."
                )
            elif event == "math" and clean.get("answer"):
                answer = clean["answer"]
                candidates.extend(
                    (
                        f"The computed answer is {answer}.",
                        f"I calculate the result as {answer}.",
                    )
                )
            elif event == "quantitative fact" and all(
                clean.get(name) for name in ("answer", "mechanism", "confidence")
            ):
                answer = clean["answer"]
                mechanism = clean["mechanism"]
                confidence = clean["confidence"]
                candidates.extend(
                    (
                        f"The measured answer is {answer}; the mechanism is "
                        f"{mechanism}, with confidence {confidence}.",
                        f"For {mechanism}, the measured value is {answer} "
                        f"(confidence {confidence}).",
                    )
                )
            elif event == "graph result" and clean.get("result"):
                result = clean["result"]
                candidates.extend(
                    (
                        f"The grounded graph says: {result}",
                        f"From the causal graph: {result}",
                    )
                )
            runtime = self.runtime
            if (
                len(candidates) > 1
                and runtime is not None
                and runtime.status().ready
                and all(runtime.text_coverage(value) >= 0.75 for value in candidates)
            ):
                try:
                    return runtime.choose(candidates)
                except (RuntimeError, ValueError):
                    pass
            return message
        fact_text = ", ".join(f"{key}: {value}" for key, value in clean.items())
        event_name = _normalise(event) or "result"
        candidates = (
            f"{event_name}: {fact_text}.",
            f"Done — {fact_text}.",
            f"Result — {fact_text}.",
        )
        runtime = self.runtime
        if (
            runtime is not None
            and runtime.status().ready
            and all(
                runtime.text_coverage(candidate) >= 0.75 for candidate in candidates
            )
        ):
            try:
                return runtime.choose(candidates)
            except (RuntimeError, ValueError):
                pass
        return candidates[0]


__all__ = [
    "AMBIGUOUS",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_TOKENIZER",
    "GroundedLanguageInterface",
    "HSSLMRuntime",
    "HSSLMStatus",
    "RESOLVED",
    "ResolvedSkill",
    "SkillDescriptor",
    "UNKNOWN",
]
