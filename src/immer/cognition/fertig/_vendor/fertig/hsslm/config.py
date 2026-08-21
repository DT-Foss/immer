"""Configuration dataclass for HSSLM."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HSSLMConfig:
    """Complete configuration for the HSSLM model.

    Defaults produce a ~7.3M parameter model with:
    - vocab_size: 16384
    - d_model: 256
    - n_layers: 6
    - d_state: 16
    - d_conv: 4
    - expand: 2
    - dt_rank: 8
    """

    # Model architecture
    vocab_size: int = 16384
    d_model: int = 256
    n_layers: int = 6
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int = 8
    max_seq_len: int = 2048
    dropout: float = 0.1
    hierarchical: bool = True
    aux_loss_weight: float = 0.1

    # Training hyperparameters
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 50000
    batch_size: int = 32
    seq_length: int = 2048

    # Inference parameters
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9

    # Special token IDs (must match tokenizer)
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HSSLMConfig":
        """Create config from dictionary."""
        # Filter to only known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    @property
    def d_inner(self) -> int:
        """Expanded inner dimension."""
        return self.d_model * self.expand

    @property
    def estimated_params(self) -> dict:
        """Estimate parameter count by component."""
        # Embedding
        emb = self.vocab_size * self.d_model + self.max_seq_len * self.d_model + self.d_model

        # Per SSM layer
        d_in = self.d_inner
        per_layer = (
            self.d_model * (d_in * 2) +           # in_proj (no bias)
            d_in * self.d_conv + d_in +            # conv1d (with bias)
            d_in * (self.dt_rank + 2 * self.d_state) +  # x_proj
            self.dt_rank * d_in + d_in +           # dt_proj (with bias)
            d_in * self.d_state +                  # A_log
            d_in +                                 # D
            d_in * self.d_model +                  # out_proj (no bias)
            self.d_model                           # RMSNorm
        )
        core = per_layer * self.n_layers + self.d_model  # final norm

        # Hierarchical composers (4 * CompositionLayer)
        comp_per = (
            self.d_model +                           # norm
            self.d_model * (self.d_model * 2) + (self.d_model * 2) +  # linear1
            (self.d_model * 2) * self.d_model + self.d_model +        # linear2
            self.d_model                             # dropout (no params)
        )
        hierarchy = comp_per * 4

        # Extra params for phrase attention query
        hierarchy += self.d_model + 1  # phrase query linear

        # Extra params for discourse gate
        hierarchy += (self.d_model * 2) * self.d_model + self.d_model

        # Auxiliary heads
        aux = (
            self.d_model * 17 + 17 +               # POS head
            self.d_model * 2 + 2 +                   # phrase boundary
            self.d_model * 8 + 8 +                   # sentence relation
            (self.d_model * 2) * self.d_model + self.d_model +  # coherence linear1
            self.d_model + 1                          # coherence linear2
        )

        # LM head shares embedding weights (0 extra)
        lm_head = 0

        return {
            'embedding': emb,
            'core': core,
            'hierarchy': hierarchy,
            'lm_head': lm_head,
            'aux_heads': aux,
            'total': emb + core + hierarchy + aux,
            'per_layer': per_layer,
        }
