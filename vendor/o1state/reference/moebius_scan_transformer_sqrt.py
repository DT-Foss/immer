"""
Sqrt-coupling Möbius Scan Transformer.

This variant uses the alternative associative coupling derived in the
Foss preprint "From Markov Chains to Minkowski Space":

    f_S(λ, v) = sqrt(v² + λ² (1 − v²))

The coupling produces pure Lorentz time dilation (no angle dependence) and
is associative, so it can be used as a sequential scan operator:

    state_t = sqrt(v_t² + state_{t-1}² (1 − v_t²))

Like the standard Möbius scan, values stay in (-1, 1).  The scan is
implemented as an explicit recurrence; unlike the Möbius case there is no
simple additive rapidity identity, but each step is cheap.
"""

import sys, os, re, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moebius_attention import StandardTransformerLayer, SinusoidalPositionalEncoding

D_MODEL = 128
N_HEADS = 4
D_HEAD = 32
N_LAYERS = 2
SEQ_LEN = 32
BATCH_SIZE = 32
EPOCHS = 2
LR = 3e-3
DROPOUT = 0.1
VOCAB_MAX = 5000
MASK_PROB = 0.15


def sqrt_couple(state, v, eps=1e-6):
    """Sqrt coupling f_S(state, v) = sqrt(v² + state²(1 - v²))."""
    return torch.sqrt(v * v + state * state * (1.0 - v * v) + eps)


def sqrt_scan(v_seq, causal=True):
    """
    Sequential sqrt-coupling scan.

    Parameters
    ----------
    v_seq : (B, T, H, D)
        Token velocities in (-1, 1).
    causal : bool
        If True, causal prefix scan.

    Returns
    -------
    state : (B, T, H, D)
        Scan state at each position.
    """
    B, T, H, D = v_seq.shape
    state = torch.zeros(B, H, D, device=v_seq.device)
    out = []
    for t in range(T):
        state = sqrt_couple(state, v_seq[:, t])
        out.append(state)
    state_out = torch.stack(out, dim=1)
    if not causal:
        rev = torch.flip(state_out, dims=[1])
        state_out = sqrt_couple(state_out, rev)
    return state_out


class SqrtCouplingMoebiusScanLayer(nn.Module):
    """Möbius scan layer using the sqrt coupling."""

    def __init__(self, d_model: int, d_head: int = 16, n_heads: int = 4,
                 causal: bool = True, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.causal = causal
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        total_dim = n_heads * d_head
        self.W_v = nn.Linear(d_model, total_dim, bias=False)
        self.W_gate = nn.Linear(d_model, total_dim, bias=False)
        self.W_out = nn.Linear(total_dim, d_model, bias=False)

        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        v = torch.tanh(self.W_v(x))
        gate = torch.sigmoid(self.W_gate(x))
        v_gated = v * gate
        if self.dropout is not None:
            v_gated = self.dropout(v_gated)

        v_gated = v_gated.view(B, T, self.n_heads, self.d_head)
        state = sqrt_scan(v_gated, causal=self.causal)
        state = state.view(B, T, self.n_heads * self.d_head)
        return self.W_out(state)


class SqrtCouplingMoebiusScanTransformerLayer(nn.Module):
    """Full transformer block around a sqrt-coupling Möbius scan layer."""

    def __init__(self, d_model: int, d_head: int = 16, n_heads: int = 4,
                 ffn_dim: int = None, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        self.scan = SqrtCouplingMoebiusScanLayer(
            d_model, d_head=d_head, n_heads=n_heads, causal=causal,
            dropout=dropout
        )
        self.ln1 = nn.LayerNorm(d_model)
        ffn_dim = ffn_dim or 4 * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ffn_dim, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.scan(x))
        x = self.ln2(x + self.ffn(x))
        return x


class SqrtCouplingMoebiusScanTransformerLM(nn.Module):
    """Complete language model for MLM or next-token tasks."""

    def __init__(self, vocab_size: int, mask_idx: int,
                 d_model: int = D_MODEL, n_layers: int = N_LAYERS,
                 n_heads: int = N_HEADS, d_head: int = D_HEAD,
                 seq_len: int = SEQ_LEN, dropout: float = DROPOUT,
                 causal: bool = True):
        super().__init__()
        self.mask_idx = mask_idx
        self.embed = nn.Embedding(vocab_size + 2, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            SqrtCouplingMoebiusScanTransformerLayer(
                d_model, d_head=d_head, n_heads=n_heads,
                ffn_dim=4 * d_model, dropout=dropout, causal=causal
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos(self.embed(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Self-contained MLM benchmark.
# ---------------------------------------------------------------------------

def load_wikitext2():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    return "\n\n".join(ds["train"]["text"]), "\n\n".join(ds["validation"]["text"])


def build_vocab(text: str, vocab_max: int = VOCAB_MAX):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    vocab = [w for w, _ in sorted(freq.items(), key=lambda kv: -kv[1])][:vocab_max]
    stoi = {w: i for i, w in enumerate(vocab)}
    unk_idx = len(vocab)
    mask_idx = len(vocab) + 1
    return vocab, stoi, unk_idx, mask_idx


def tokenize(text: str, stoi: dict, unk_idx: int):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [stoi.get(w, unk_idx) for w in words]


def make_mlm_batches(ids, seq_len, batch_size, mask_idx, mask_prob=0.15):
    total = len(ids) // seq_len
    X, Y, M = [], [], []
    for i in range(total):
        start = i * seq_len
        seq = ids[start:start + seq_len]
        if len(seq) < seq_len:
            continue
        y = seq.copy()
        mask = (torch.rand(seq_len) < mask_prob).long()
        for j in range(seq_len):
            if mask[j]:
                r = torch.rand(1).item()
                if r < 0.8:
                    seq[j] = mask_idx
                elif r < 0.9:
                    seq[j] = torch.randint(0, mask_idx, (1,)).item()
        X.append(seq)
        Y.append(y)
        M.append(mask)
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    M = torch.stack(M)
    n = (len(X) // batch_size) * batch_size
    return X[:n], Y[:n], M[:n]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, X, Y, M, opt):
    model.train()
    perm = torch.randperm(len(X))
    total_loss = 0.0
    n_batches = 0
    for i in range(0, len(X), BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        xb, yb, mb = X[idx], Y[idx], M[idx]
        logits = model(xb)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), yb.reshape(-1), reduction='none'
        )
        loss = (loss * mb.reshape(-1).float()).sum() / (mb.sum() + 1e-6)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, X, Y, M):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    n_batches = 0
    for i in range(0, len(X), BATCH_SIZE):
        xb, yb, mb = X[i:i + BATCH_SIZE], Y[i:i + BATCH_SIZE], M[i:i + BATCH_SIZE]
        logits = model(xb)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), yb.reshape(-1), reduction='none'
        )
        masked_loss = (loss * mb.reshape(-1).float()).sum() / (mb.sum() + 1e-6)
        total_loss += masked_loss.item()
        preds = logits.argmax(dim=-1)
        total_correct += ((preds == yb) & mb.bool()).sum().item()
        total_masked += mb.sum().item()
        n_batches += 1
    avg_loss = total_loss / n_batches
    return avg_loss, math.exp(avg_loss), total_correct / total_masked


def run(name, model, X_train, Y_train, M_train, X_val, Y_val, M_val):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    start = time.time()
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, X_train, Y_train, M_train, opt)
        val_loss, val_ppl, val_acc = evaluate(model, X_val, Y_val, M_val)
        print(
            f"{name} | epoch {epoch+1}/{EPOCHS} | train {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | ppl {val_ppl:.2f} | acc {val_acc:.4f}"
        )
    elapsed = time.time() - start
    final_loss, final_ppl, final_acc = evaluate(model, X_val, Y_val, M_val)
    return final_loss, final_ppl, final_acc, elapsed


def main():
    print("Loading WikiText-2...")
    train_text, val_text = load_wikitext2()
    print("Building vocab...")
    vocab, stoi, unk_idx, mask_idx = build_vocab(train_text)
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}, unk: {unk_idx}, mask: {mask_idx}")
    print("Tokenizing...")
    train_ids = tokenize(train_text, stoi, unk_idx)
    val_ids = tokenize(val_text, stoi, unk_idx)
    print(f"Tokens: train {len(train_ids)}, val {len(val_ids)}")
    print("Building MLM batches...")
    X_train, Y_train, M_train = make_mlm_batches(
        train_ids, SEQ_LEN, BATCH_SIZE, mask_idx, MASK_PROB
    )
    X_val, Y_val, M_val = make_mlm_batches(
        val_ids, SEQ_LEN, BATCH_SIZE, mask_idx, MASK_PROB
    )
    print(f"Batches: train {len(X_train)}, val {len(X_val)}")

    results = []

    print("\n=== Sqrt-Coupling Möbius Scan Transformer (causal) ===")
    model = SqrtCouplingMoebiusScanTransformerLM(
        vocab_size, mask_idx, d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=N_HEADS, d_head=D_HEAD, seq_len=SEQ_LEN,
        dropout=DROPOUT, causal=True
    )
    print(f"Params: {count_params(model):,}")
    results.append((
        "Sqrt-Coupling Möbius Scan Transformer (MLM)",
        *run("Sqrt", model, X_train, Y_train, M_train, X_val, Y_val, M_val)
    ))

    print("\n=== RESULTS ===")
    for name, loss, ppl, acc, elapsed in results:
        print(
            f"{name:50s} | loss {loss:.4f} | ppl {ppl:7.2f} | "
            f"acc {acc:.4f} | time {elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
