"""
Hierarchical Composer - multi-level linguistic representation builder.

Maps token-level SSM hidden states to increasingly abstract representations
at word, phrase, sentence, and discourse levels. Each composer uses a
learned 2-layer MLP with residual connection to transform pooled states.

The hierarchical pipeline:
    Token States (B, L, D)
        | WordComposer (mean-pool + MLP)
    Word States (B, W, D)
        | PhraseComposer (local attention + MLP)
    Phrase States (B, P, D)
        | SentenceComposer (max-pool + MLP)
    Sentence States (B, S, D)
        | DiscourseComposer (gated recurrence + MLP)
    Discourse States (B, S, D)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class CompositionLayer(nn.Module):
    """Generic 2-layer MLP composition module with residual connection.

    Transforms pooled representations through:
        x -> RMSNorm -> Linear(2x) -> SiLU -> Dropout -> Linear -> + x
    """

    def __init__(self, d_model: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        # Use bottleneck: d_model → d_model → d_model (no expansion)
        # This saves ~50% params per layer while maintaining expressivity
        bottleneck = max(d_model // 2, 64)
        self.linear1 = nn.Linear(d_model, bottleneck)
        self.linear2 = nn.Linear(bottleneck, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) pooled representations.

        Returns:
            (B, N, d_model) composed representations.
        """
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x + residual


class WordComposer(nn.Module):
    """Compose token-level representations into word-level representations.

    Uses word boundary information (from tokenizer) to mean-pool
    token states within each word span. The pooled representation
    captures the unified meaning of each word.
    """

    def __init__(self, d_model: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.composition = CompositionLayer(d_model, dropout)

    def forward(
        self,
        token_states: torch.Tensor,
        word_boundaries: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            token_states: (B, L, d_model) token-level hidden states.
            word_boundaries: List of (W, 2) tensors per batch item [start, end].

        Returns:
            - word_states_list: List of (W, d_model) tensors per batch item.
            - word_states_padded: (B, W_max, d_model) padded tensor.
        """
        B, L, D = token_states.shape
        device = token_states.device

        word_states_list = []
        for b in range(B):
            boundaries = word_boundaries[b]  # (W, 2)
            W = boundaries.shape[0]

            # Pool token states for each word
            pooled = []
            for w in range(W):
                start, end = boundaries[w].tolist()
                # Clamp to valid range
                start = max(0, min(start, L - 1))
                end = max(start + 1, min(end, L))
                word_toks = token_states[b, start:end, :]  # (word_len, D)
                pooled.append(word_toks.mean(dim=0))  # (D,)

            word_states_b = torch.stack(pooled) if pooled else torch.zeros(1, D, device=device)
            word_states_b = self.composition(word_states_b.unsqueeze(0)).squeeze(0)
            word_states_list.append(word_states_b)

        # Pad to max W
        max_W = max(w.shape[0] for w in word_states_list)
        word_states_padded = torch.zeros(B, max_W, D, device=device)
        for b, ws in enumerate(word_states_list):
            word_states_padded[b, :ws.shape[0]] = ws

        return word_states_list, word_states_padded


class PhraseComposer(nn.Module):
    """Compose word-level representations into phrase-level representations.

    Uses local attention-weighted aggregation over a sliding window
    of adjacent words. This captures multi-word units like noun phrases,
    verb phrases, and prepositional phrases.
    """

    def __init__(
        self, d_model: int = 256, window_size: int = 3, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.window_size = window_size
        # Learnable query for attention scoring
        self.query = nn.Linear(d_model, 1)
        self.composition = CompositionLayer(d_model, dropout)

    def forward(self, word_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            word_states: (B, W, d_model) word-level states.

        Returns:
            (B, P, d_model) phrase-level states (P <= W).
        """
        B, W, D = word_states.shape

        if W <= self.window_size:
            # Too few words: treat all as one phrase
            phrase_states = word_states.mean(dim=1, keepdim=True)  # (B, 1, D)
            return self.composition(phrase_states)

        # Sliding window attention
        phrases = []
        stride = max(1, self.window_size // 2)

        for start in range(0, W, stride):
            end = min(start + self.window_size, W)
            window = word_states[:, start:end, :]  # (B, window_len, D)

            # Compute attention scores within window
            scores = self.query(window).squeeze(-1)  # (B, window_len)
            scores = F.softmax(scores, dim=-1)

            # Weighted sum
            weighted = (window * scores.unsqueeze(-1)).sum(dim=1)  # (B, D)
            phrases.append(weighted)

        phrase_states = torch.stack(phrases, dim=1)  # (B, P, D)
        return self.composition(phrase_states)


class SentenceComposer(nn.Module):
    """Compose phrase-level representations into sentence-level representations.

    Max-pools over all phrases within each sentence, then applies
    MLP transform. Max-pooling selects the most salient features
    from any phrase in the sentence.
    """

    def __init__(self, d_model: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.composition = CompositionLayer(d_model, dropout)

    def forward(
        self,
        phrase_states: torch.Tensor,
        sentence_boundaries: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            phrase_states: (B, P, d_model) phrase-level states.
            sentence_boundaries: List of (S, 2) tensors per batch [start, end].

        Returns:
            (B, S, d_model) sentence-level states.
        """
        B, P, D = phrase_states.shape
        device = phrase_states.device

        sentence_states_list = []
        max_S = 0

        for b in range(B):
            boundaries = sentence_boundaries[b]
            S = boundaries.shape[0]
            max_S = max(max_S, S)

            sentence_reps = []
            for s in range(S):
                start, end = boundaries[s].tolist()
                start = max(0, min(start, P - 1))
                end = max(start + 1, min(end, P))

                if start < P:
                    phrase_window = phrase_states[b, start:end, :]  # (window, D)
                    # Max pool then mean if multiple
                    if phrase_window.shape[0] > 1:
                        sent = phrase_window.max(dim=0)[0]  # (D,)
                    else:
                        sent = phrase_window.squeeze(0)  # (D,)
                    sentence_reps.append(sent)

            if sentence_reps:
                sent_tensor = torch.stack(sentence_reps)
                sent_tensor = self.composition(sent_tensor.unsqueeze(0)).squeeze(0)
            else:
                sent_tensor = torch.zeros(1, D, device=device)
            sentence_states_list.append(sent_tensor)

        # Pad
        sentence_states = torch.zeros(B, max_S, D, device=device)
        for b, ss in enumerate(sentence_states_list):
            sentence_states[b, :ss.shape[0]] = ss

        return sentence_states


class DiscourseComposer(nn.Module):
    """Compose sentence-level representations into discourse-level representations.

    Uses a learned gated recurrence (mini-SSM) to accumulate sentence
    states across the document. The gate controls how much of the new
    sentence vs. previous discourse context to retain.

    This models discourse coherence, anaphora resolution, and topic
    continuity across sentences.
    """

    def __init__(self, d_model: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        # Gating function: given [current_sent, prev_discourse] -> gate
        bottleneck = max(d_model // 2, 64)
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model * 2, bottleneck), nn.SiLU(),
            nn.Linear(bottleneck, d_model),
            nn.Sigmoid(),
        )
        self.composition = CompositionLayer(d_model, dropout)
        self.discourse_state: Optional[torch.Tensor] = None

    def forward(
        self,
        sentence_states: torch.Tensor,
        reset_state: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            sentence_states: (B, S, d_model) sentence-level states.
            reset_state: If True, reset discourse state (new document).

        Returns:
            (B, S, d_model) discourse-enriched sentence representations.
        """
        B, S, D = sentence_states.shape
        device = sentence_states.device

        if reset_state or self.discourse_state is None or self.discourse_state.shape[0] != B:
            self.discourse_state = torch.zeros(B, D, device=device)

        discourse_states = []
        for s in range(S):
            current = sentence_states[:, s, :]  # (B, D)

            # Compute gate
            combined = torch.cat([current, self.discourse_state], dim=-1)  # (B, 2D)
            gate = self.gate_proj(combined)  # (B, D)

            # Gated update: mix current sentence with previous discourse
            self.discourse_state = gate * current + (1 - gate) * self.discourse_state
            discourse_states.append(self.discourse_state.clone())

        discourse_tensor = torch.stack(discourse_states, dim=1)  # (B, S, D)
        return self.composition(discourse_tensor)

    def reset(self) -> None:
        """Reset internal discourse state. Call at document boundaries."""
        self.discourse_state = None


class HierarchicalComposer(nn.Module):
    """Full hierarchical composition pipeline.

    Takes token-level SSM outputs and builds multi-level representations.
    Can be disabled (forward passes through with minimal overhead).

    When disabled, returns only token-level representations.
    """

    def __init__(
        self, d_model: int = 256, enabled: bool = True, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.word_composer = WordComposer(d_model, dropout)
        self.phrase_composer = PhraseComposer(d_model, dropout=dropout)
        self.sentence_composer = SentenceComposer(d_model, dropout)
        self.discourse_composer = DiscourseComposer(d_model, dropout)

    def forward(
        self,
        token_states: torch.Tensor,
        boundaries: Dict[str, List[torch.Tensor]],
        return_all_levels: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run full hierarchical composition.

        Args:
            token_states: (B, L, d_model) from SSM core.
            boundaries: Dict with "word" and "sentence" boundary tensors.
            return_all_levels: If True, return all intermediate reps.

        Returns:
            Dict with representations at each level.
        """
        if not self.enabled:
            return {"token": token_states}

        # Get boundaries
        word_boundaries = boundaries.get("word_boundaries", boundaries.get("word", []))
        sentence_boundaries = boundaries.get(
            "sentence_boundaries", boundaries.get("sentence", [])
        )

        result: Dict[str, torch.Tensor] = {"token": token_states}

        # Word composition
        if word_boundaries and len(word_boundaries) > 0:
            _, word_states = self.word_composer(token_states, word_boundaries)
            result["word"] = word_states

            # Phrase composition
            phrase_states = self.phrase_composer(word_states)
            result["phrase"] = phrase_states

            # Sentence composition
            if sentence_boundaries and len(sentence_boundaries) > 0:
                sentence_states = self.sentence_composer(
                    phrase_states, sentence_boundaries
                )
                result["sentence"] = sentence_states

                # Discourse composition
                discourse_states = self.discourse_composer(sentence_states)
                result["discourse"] = discourse_states
            else:
                # If no sentence boundaries, use phrases as proxy
                B, P, D = phrase_states.shape
                fake_bound = [
                    torch.tensor([[0, P]], device=phrase_states.device)
                    for _ in range(B)
                ]
                sentence_states = self.sentence_composer(phrase_states, fake_bound)
                result["sentence"] = sentence_states
                discourse_states = self.discourse_composer(sentence_states)
                result["discourse"] = discourse_states
        else:
            # No word boundaries: create fake single-word per token
            B, L, D = token_states.shape
            result["word"] = token_states
            result["phrase"] = token_states
            fake_bound = [
                torch.tensor([[0, L]], device=token_states.device) for _ in range(B)
            ]
            result["sentence"] = self.sentence_composer(token_states, fake_bound)
            result["discourse"] = self.discourse_composer(result["sentence"])

        return result
