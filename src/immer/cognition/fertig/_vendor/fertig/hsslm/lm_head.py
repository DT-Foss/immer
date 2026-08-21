"""
Language Modeling Head and Auxiliary Heads.

LMHead: next-token prediction (weight-tied to embeddings).
AuxiliaryHeads: hierarchical supervision signals during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class LMHead(nn.Module):
    """Language modeling head for next-token prediction.

    Uses weight tying with input embeddings to save ~4.2M parameters.
    Logits are computed as: logits = hidden_states @ embedding_weight.T
    """

    def __init__(
        self,
        d_model: int = 256,
        vocab_size: int = 16384,
        embedding_weight: Optional[nn.Parameter] = None,
    ) -> None:
        """
        Args:
            d_model: Model dimension.
            vocab_size: Vocabulary size.
            embedding_weight: Shared embedding weight for tying.
                              If None, creates a separate projection.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        if embedding_weight is not None:
            # Weight tying: share embedding matrix with output
            # Store as _embedding_weight to avoid double-counting in parameters()
            self.register_buffer('_embedding_weight', embedding_weight)
            self.out_proj = None
        else:
            # Separate projection (more parameters)
            self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
            self.register_buffer('_embedding_weight', None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L, d_model) from model core.

        Returns:
            (B, L, vocab_size) logits.
        """
        if self.out_proj is None and self._embedding_weight is not None:
            # Weight-tied: F.linear uses transposed weight automatically
            return F.linear(hidden_states, self._embedding_weight)
        return self.out_proj(hidden_states)


class AuxiliaryHeads(nn.Module):
    """Auxiliary prediction heads for hierarchical supervision.

    These heads provide training signals at multiple linguistic levels,
    guiding the model to learn meaningful hierarchical representations.

    Used during training; during inference they can optionally provide
    linguistic analysis of the input.

    Heads:
        - POS tagging: predict part-of-speech tags from word representations
        - Phrase boundary: detect phrase boundaries
        - Sentence relation: classify discourse relations between sentences
        - Coherence: score discourse coherence
    """

    # Universal POS tag set (17 tags)
    POS_TAGS = [
        "ADJ", "ADP", "ADV", "AUX", "CONJ",
        "DET", "INTJ", "NOUN", "NUM", "PART",
        "PRON", "PROPN", "PUNCT", "SCONJ", "SYM",
        "VERB", "X",
    ]

    # Discourse relations (8 coarse-grained)
    DISCOURSE_RELATIONS = [
        "elaboration", "contrast", "cause", "condition",
        "temporal", "attribution", "same-unit", "no-relation",
    ]

    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.d_model = d_model

        # Word-level POS tagging (17 Universal POS tags)
        self.pos_head = nn.Linear(d_model, 17)

        # Phrase boundary detection (binary: boundary / no boundary)
        self.phrase_boundary_head = nn.Linear(d_model, 2)

        # Sentence relation prediction (8 discourse relations)
        self.sentence_relation_head = nn.Linear(d_model, 8)

        # Discourse coherence scoring (scalar)
        # Takes concatenation of adjacent sentence representations
        # Uses bottleneck for parameter efficiency
        coh_hidden = max(d_model // 4, 32)
        self.coherence_head = nn.Sequential(
            nn.Linear(d_model * 2, coh_hidden),
            nn.SiLU(),
            nn.Linear(coh_hidden, 1),
        )

    def forward(
        self, hierarchical_outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hierarchical_outputs: Dict from HierarchicalComposer with keys:
                - "word": (B, W, d_model)
                - "phrase": (B, P, d_model)
                - "sentence": (B, S, d_model)
                - "discourse": (B, S, d_model)

        Returns:
            Dict with auxiliary predictions:
                - "pos_logits": (B, W, 17)
                - "phrase_boundary_logits": (B, P, 2)
                - "sentence_relation_logits": (B, max(S-1, 1), 8)
                - "coherence_scores": (B, max(S-1, 1), 1)
        """
        result = {}

        # POS tagging from word representations
        if "word" in hierarchical_outputs:
            word_states = hierarchical_outputs["word"]
            result["pos_logits"] = self.pos_head(word_states)

        # Phrase boundary detection from phrase representations
        if "phrase" in hierarchical_outputs:
            phrase_states = hierarchical_outputs["phrase"]
            result["phrase_boundary_logits"] = self.phrase_boundary_head(phrase_states)

        # Sentence relation from sentence representations
        if "sentence" in hierarchical_outputs:
            sent_states = hierarchical_outputs["sentence"]
            B, S, D = sent_states.shape

            if S > 1:
                # Predict relation between adjacent sentences
                # Concatenate adjacent pairs
                sent_pairs = torch.cat(
                    [sent_states[:, :-1, :], sent_states[:, 1:, :]], dim=-1
                )  # (B, S-1, 2D)
                # Predict relation using each sentence's state
                result["sentence_relation_logits"] = self.sentence_relation_head(
                    sent_states[:, 1:, :]  # predict for sentence t+1
                )

                # Coherence score for adjacent sentences
                result["coherence_scores"] = self.coherence_head(sent_pairs)
            else:
                # Single sentence: return dummy values
                result["sentence_relation_logits"] = self.sentence_relation_head(
                    sent_states
                )
                result["coherence_scores"] = torch.zeros(
                    B, 1, 1, device=sent_states.device
                )

        return result

    def compute_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses (if targets provided).

        Args:
            predictions: Output from forward().
            targets: Optional ground truth labels.

        Returns:
            Dict of scalar loss values.
        """
        losses = {}

        if targets is None:
            return losses

        # POS loss
        if "pos_logits" in predictions and "pos_labels" in targets:
            losses["pos_loss"] = F.cross_entropy(
                predictions["pos_logits"].view(-1, 17),
                targets["pos_labels"].view(-1),
                ignore_index=-1,
            )

        # Phrase boundary loss
        if "phrase_boundary_logits" in predictions and "phrase_boundary_labels" in targets:
            losses["phrase_boundary_loss"] = F.cross_entropy(
                predictions["phrase_boundary_logits"].view(-1, 2),
                targets["phrase_boundary_labels"].view(-1),
                ignore_index=-1,
            )

        # Sentence relation loss
        if "sentence_relation_logits" in predictions and "sentence_relation_labels" in targets:
            losses["sentence_relation_loss"] = F.cross_entropy(
                predictions["sentence_relation_logits"].view(-1, 8),
                targets["sentence_relation_labels"].view(-1),
                ignore_index=-1,
            )

        # Coherence loss (MSE)
        if "coherence_scores" in predictions and "coherence_labels" in targets:
            losses["coherence_loss"] = F.mse_loss(
                predictions["coherence_scores"].squeeze(-1),
                targets["coherence_labels"],
            )

        return losses
