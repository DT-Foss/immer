"""Embedding module with positional encoding."""

import math
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Used instead of LayerNorm for better training stability with SSMs.
    LN subtracts mean then divides by std; RMSNorm only divides by RMS.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: (*, dim) -> (*, dim)"""
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class HierarchicalEmbedding(nn.Module):
    """Token embedding layer with learned positional encoding.

    Simple but effective: token embeddings + position embeddings + norm + dropout.
    The BPE tokenizer naturally provides subword units that map to morpheme-like
    structure, so we don't need separate character-level decomposition.
    """

    def __init__(
        self,
        vocab_size: int = 16384,
        d_model: int = 256,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding: (16384, 256) = 4,194,304 params
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx
        )

        # Positional embedding: (2048, 256) = 524,288 params
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Normalization and dropout
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embeddings with small random values."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        # Ensure padding token has zero embedding
        with torch.no_grad():
            self.token_embedding.weight[0].fill_(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) token IDs.

        Returns:
            (B, L, 256) contextualized embeddings.
        """
        B, L = input_ids.shape

        # Token embeddings
        tok_emb = self.token_embedding(input_ids)  # (B, L, 256)

        # Position embeddings
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)  # (1, L)
        pos_emb = self.position_embedding(positions)  # (1, L, 256)

        # Combine and normalize
        x = tok_emb + pos_emb
        x = self.norm(x)
        x = self.dropout(x)

        return x

    def get_embedding_table(self) -> torch.Tensor:
        """Return the token embedding weight matrix for weight tying."""
        return self.token_embedding.weight
