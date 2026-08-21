"""
Deterministic Transitive Inference Engine for HSSLM-C.

Based on the .causal format by David Tom Foss:
- 3-pass deterministic inference (exact, semantic, fuzzy)
- Moebius confidence: f(c1,c2) = (c1+c2)/(1+c1*c2)
- Weak signal amplification: 3 explicit -> 21+ inferred (7x)
"""
import math
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


class JaroWinkler:
    """Jaro-Winkler string similarity for fuzzy token matching (Pass 3)."""

    @staticmethod
    def similarity(s1: str, s2: str) -> float:
        if s1 == s2:
            return 1.0
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        match_distance = max(len1, len2) // 2 - 1
        s1_matches = [False] * len1
        s2_matches = [False] * len2
        matches = 0
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
        if matches == 0:
            return 0.0
        k, transpositions = 0, 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        jaro = ((matches / len1) + (matches / len2) +
                ((matches - transpositions / 2) / matches)) / 3.0
        prefix_len = sum(1 for i in range(min(4, len1, len2)) if s1[i] == s2[i])
        return jaro + 0.1 * prefix_len * (1 - jaro)


class MoebiusConfidence:
    """Moebius confidence addition -- prevents confidence decay at hubs."""

    @staticmethod
    def combine(c1: float, c2: float) -> float:
        if c1 >= 1.0 or c2 >= 1.0:
            return max(c1, c2)
        return (c1 + c2) / (1 + c1 * c2)

    @staticmethod
    def chain(confidences: List[float], gamma: float = 0.85) -> float:
        if not confidences:
            return 0.0
        prod = 1.0
        for c in confidences:
            prod *= c
        return prod * (gamma ** (len(confidences) - 1))


class CausalInferenceEngine:
    """3-pass deterministic inference engine with weak signal amplification."""

    QUALITY_THRESHOLD = 0.30
    FUZZY_THRESHOLD = 0.85
    CHAIN_GAMMA = 0.85
    DETECTION_THRESHOLD = 10

    def __init__(self, vocab_size: int = 16384,
                 token_id_to_str: Optional[Dict[int, str]] = None,
                 quality_threshold: float = 0.30):
        self.vocab_size = vocab_size
        self.token_id_to_str = token_id_to_str or {}
        self.quality_threshold = quality_threshold
        self.explicit_triplets: Dict[Tuple[int, int, int], float] = {}
        self.inferred_triplets: Dict[Tuple[int, int, int], Tuple[float, List]] = {}
        self.adjacency: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.directions: Dict[Tuple[int, int], int] = {}

    def learn_tokens(self, token_ids: List[int]):
        for i in range(len(token_ids) - 1):
            self.adjacency[token_ids[i]][token_ids[i + 1]] += 1

    def add_explicit_triplet(self, subject: int, mechanism: int, obj: int,
                             confidence: float = 1.0):
        self.explicit_triplets[(subject, mechanism, obj)] = confidence
        self.directions[(subject, obj)] = 1 if mechanism >= 0 else -1

    def pass1_exact_matching(self, token_ids: List[int]) -> Dict[Tuple[int, int, int], float]:
        inferences = {}
        n = len(token_ids)
        for i in range(n):
            for j in range(i + 1, n):
                if token_ids[j] in self.adjacency.get(token_ids[i], {}):
                    for k in range(j + 1, n):
                        conf = self.adjacency[token_ids[i]][token_ids[j]]
                        conf *= self.adjacency[token_ids[j]].get(token_ids[k], 0)
                        conf *= 0.01
                        if conf > self.quality_threshold:
                            inferences[(token_ids[i], token_ids[j], token_ids[k])] = conf
        return inferences

    def pass2_semantic_direction(self,
                                  chains: Dict[Tuple[int, int, int], float]
                                  ) -> Dict[Tuple[int, int, int], float]:
        directed = {}
        for (s, m, o), conf in chains.items():
            d1 = self.directions.get((s, m), 0)
            d2 = self.directions.get((m, o), 0)
            if d1 == 0 or d2 == 0:
                result_dir = d1 or d2
            elif d1 == d2:
                result_dir = 1
            else:
                result_dir = -1
                conf *= 0.5
            if conf > self.quality_threshold:
                directed[(s, m, o)] = conf * (1 if result_dir >= 0 else 0.5)
        return directed

    def pass3_fuzzy_matching(self, token_ids: List[int],
                             threshold: float = FUZZY_THRESHOLD
                             ) -> Dict[Tuple[int, int, int], float]:
        fuzzy = {}
        n = len(token_ids)
        for i in range(n):
            s1 = self.token_id_to_str.get(token_ids[i], str(token_ids[i]))
            for j in range(i + 1, n):
                s2 = self.token_id_to_str.get(token_ids[j], str(token_ids[j]))
                sim = JaroWinkler.similarity(s1, s2)
                if sim >= threshold and j + 1 < n:
                    triplet = (token_ids[i], token_ids[j], token_ids[j + 1])
                    conf = sim * 0.5
                    if conf > self.quality_threshold:
                        fuzzy[triplet] = conf
        return fuzzy

    def quality_filter(self, inferences: Dict[Tuple, float]) -> Dict[Tuple, float]:
        filtered = {}
        for triplet, conf in inferences.items():
            if triplet[0] == triplet[-1]:
                continue
            if conf < self.quality_threshold:
                continue
            reverse = (triplet[-1], triplet[1], triplet[0])
            if reverse in inferences and inferences[reverse] > conf:
                continue
            filtered[triplet] = conf
        return filtered

    def transitive_closure(self, token_ids: List[int]) -> Dict[Tuple[int, ...], float]:
        self.learn_tokens(token_ids)
        p1 = self.pass1_exact_matching(token_ids)
        p2 = self.pass2_semantic_direction(p1)
        p3 = self.pass3_fuzzy_matching(token_ids)
        all_inf = {**p1, **p2, **p3}
        filtered = self.quality_filter(all_inf)
        final = {}
        for triplet, conf in filtered.items():
            if triplet in self.inferred_triplets:
                old_conf, _ = self.inferred_triplets[triplet]
                conf = MoebiusConfidence.combine(old_conf, conf)
            final[triplet] = conf
        self.inferred_triplets.update({t: (c, []) for t, c in final.items()})
        return final

    def get_amplification_factor(self, token_ids: List[int]) -> float:
        explicit = len(set(token_ids))
        inferred = len(self.transitive_closure(token_ids))
        return inferred / max(explicit, 1)


class WeakSignalAmplifier(nn.Module):
    """Neural weak signal amplifier using deterministic inference.

    Uses a SMALL projection matrix (not full embedding) to keep parameters low.
    Maps inferred token relationships to embedding space via linear projection.
    """

    def __init__(self, inference_engine: CausalInferenceEngine, d_model: int = 256,
                 vocab_size: int = 16384, proj_dim: int = 64):
        super().__init__()
        self.engine = inference_engine
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Small hash-based projection instead of full embedding
        # Maps token ID hash to a low-dim vector, then projects up
        self.hash_proj = nn.Sequential(
            nn.Linear(4, proj_dim),  # 4 hash features per token
            nn.SiLU(),
        )
        self.up_proj = nn.Linear(proj_dim, d_model, bias=False)
        self.amplify_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.SiLU(),
            nn.Linear(d_model // 4, 1), nn.Sigmoid())

    def _token_to_features(self, token_id: int, device) -> torch.Tensor:
        """Generate deterministic hash features from token ID."""
        # 4 deterministic features from token ID (no learned params)
        t = float(token_id)
        features = torch.tensor([
            (t % 1000) / 1000.0,           # low-order bits
            ((t * 31) % 1000) / 1000.0,    # hashed
            ((t * 127) % 1000) / 1000.0,   # multi-hash
            (t / self.vocab_size),           # normalized position
        ], device=device, dtype=torch.float32)
        return features

    def forward(self, embeddings: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        B, L, d_model = embeddings.shape
        amplified = embeddings.clone()
        for b in range(B):
            tokens = token_ids[b].tolist()
            inferred = self.engine.transitive_closure(tokens)
            if not inferred:
                continue
            for pos in range(L):
                related = [(t, c) for t, c in inferred.items()
                           if pos < len(t) and t[pos] == tokens[pos]]
                if not related:
                    continue
                inf_signals = []
                for triplet, conf in related[:10]:
                    tid = triplet[-1] % self.vocab_size
                    feat = self._token_to_features(tid, embeddings.device)
                    e = self.up_proj(self.hash_proj(feat))
                    inf_signals.append(e * conf)
                if inf_signals:
                    inf_signal = torch.stack(inf_signals).mean(dim=0)
                    gate = self.amplify_gate(embeddings[b, pos])
                    amplified[b, pos] += gate.squeeze() * inf_signal
        return amplified
