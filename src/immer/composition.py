"""Central composition root for IMMER runtime components.

Construction is the one place where policy-bearing components are wired.  In
particular, S3 and FERTIG remain private children of :class:`ExactCascade`, so
the registry has exactly one ``exact_math`` owner.  The learning byte stream is
kept as a separate object and is never injected into the frozen A1 organ path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .cognition.exact_cascade import ExactCascade
from .cognition.fertig import FertigSolver
from .contracts import Component, Request, Result
from .runtime import ImmerRuntime

if TYPE_CHECKING:
    from .substrate.daemon import LifeStream


@dataclass(frozen=True, slots=True)
class CompositionRoot:
    """Fully wired core runtime plus the deliberately separate life stream."""

    runtime: ImmerRuntime
    exact_math: ExactCascade
    grounded_chat: Component | None = None
    life_stream: LifeStream | None = None
    general_chat: Component | None = None

    @classmethod
    def build(
        cls,
        *,
        s3_arithmetic: object | None = None,
        fertig: object | None = None,
        life_stream: LifeStream | None = None,
        s3_manifest: str | Path | None = None,
        s3_artifact_root: str | Path | None = None,
        fertig_root: str | Path | None = None,
        grounded_chat: Component | None = None,
        general_chat: Component | None = None,
        fertig_state_dir: str | Path | None = None,
        fertig_graph: str | Path | None = None,
        qwen38_causal_bundle: str | Path | None = None,
        qwen38_tokenizer: str | Path | None = None,
        qwen38_options: Mapping[str, Any] | None = None,
    ) -> CompositionRoot:
        """Build without loading neural artifacts.

        ``S3Arithmetic`` is imported and constructed lazily here so importing
        IMMER's basic contracts does not require torch.  Its own first handled
        request performs the cold artifact load.
        """

        if s3_arithmetic is not None and (
            s3_manifest is not None or s3_artifact_root is not None
        ):
            raise ValueError(
                "s3_manifest/s3_artifact_root configure the built-in S3 backend; "
                "do not combine them with s3_arithmetic"
            )
        if fertig is not None and fertig_root is not None:
            raise ValueError("pass either fertig or fertig_root, not both")
        if grounded_chat is not None and fertig_state_dir is not None:
            raise ValueError("pass either grounded_chat or fertig_state_dir, not both")
        qwen38_requested = (
            qwen38_causal_bundle is not None or qwen38_tokenizer is not None
        )
        if (qwen38_causal_bundle is None) != (qwen38_tokenizer is None):
            raise ValueError(
                "qwen38_causal_bundle and qwen38_tokenizer must be configured together"
            )
        if general_chat is not None and qwen38_requested:
            raise ValueError(
                "pass either general_chat or the local Qwen3.8 bundle/tokenizer, not both"
            )
        if qwen38_options is not None and not isinstance(qwen38_options, Mapping):
            raise TypeError("qwen38_options must be a mapping or None")
        if qwen38_options is not None and not qwen38_requested:
            raise ValueError(
                "qwen38_options requires the local Qwen3.8 bundle/tokenizer"
            )

        if s3_arithmetic is None:
            from .capabilities.s3_runtime import S3Arithmetic

            s3_arithmetic = S3Arithmetic(
                manifest=s3_manifest,
                artifact_root=s3_artifact_root,
            )
        if fertig is None:
            fertig = FertigSolver(fertig_root)

        # A learning stream and an exact backend must never be aliases.  More
        # importantly, the stream is not passed to S3 at all: S3 owns its
        # private frozen A1 host, while LifeDaemon may own this mutable stream.
        if life_stream is s3_arithmetic or life_stream is fertig:
            raise ValueError("learning life stream must be distinct from frozen exact backends")

        exact_math = ExactCascade(s3_arithmetic=s3_arithmetic, fertig=fertig)
        if grounded_chat is None and fertig_state_dir is not None:
            from .cognition.fertig import FertigGrounded

            grounded_chat = FertigGrounded(
                fertig_state_dir,
                graph_path=fertig_graph,
            )
        if general_chat is None and qwen38_requested:
            from .runtimes.qwen3_8.adapter import Qwen38CausalChat

            options = {} if qwen38_options is None else dict(qwen38_options)
            forbidden = {
                "bundle_path",
                "runtime_factory",
                "tokenizer_path",
            }.intersection(options)
            if forbidden:
                raise ValueError(
                    "qwen38_options cannot override protected Qwen3.8 constructor fields"
                )
            assert qwen38_causal_bundle is not None
            assert qwen38_tokenizer is not None
            general_chat = Qwen38CausalChat(
                qwen38_causal_bundle,
                qwen38_tokenizer,
                **options,
            )
        components = tuple(
            component
            for component in (exact_math, grounded_chat, general_chat)
            if component is not None
        )
        runtime = ImmerRuntime(components)
        return cls(
            runtime=runtime,
            exact_math=exact_math,
            grounded_chat=grounded_chat,
            general_chat=general_chat,
            life_stream=life_stream,
        )

    def dispatch(
        self,
        capability: str,
        payload: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> Result:
        """Small convenience surface for future CLI/daemon wiring."""

        return self.runtime.dispatch(
            Request(capability, payload, {} if metadata is None else metadata)
        )


def compose_runtime(
    *,
    s3_arithmetic: object | None = None,
    fertig: object | None = None,
    life_stream: LifeStream | None = None,
    s3_manifest: str | Path | None = None,
    s3_artifact_root: str | Path | None = None,
    fertig_root: str | Path | None = None,
    grounded_chat: Component | None = None,
    general_chat: Component | None = None,
    fertig_state_dir: str | Path | None = None,
    fertig_graph: str | Path | None = None,
    qwen38_causal_bundle: str | Path | None = None,
    qwen38_tokenizer: str | Path | None = None,
    qwen38_options: Mapping[str, Any] | None = None,
) -> CompositionRoot:
    """Functional alias for callers that do not need the classmethod syntax."""

    return CompositionRoot.build(
        s3_arithmetic=s3_arithmetic,
        fertig=fertig,
        life_stream=life_stream,
        s3_manifest=s3_manifest,
        s3_artifact_root=s3_artifact_root,
        fertig_root=fertig_root,
        grounded_chat=grounded_chat,
        general_chat=general_chat,
        fertig_state_dir=fertig_state_dir,
        fertig_graph=fertig_graph,
        qwen38_causal_bundle=qwen38_causal_bundle,
        qwen38_tokenizer=qwen38_tokenizer,
        qwen38_options=qwen38_options,
    )


__all__ = ["CompositionRoot", "compose_runtime"]
