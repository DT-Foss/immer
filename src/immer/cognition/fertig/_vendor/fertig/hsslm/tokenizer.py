"""Tokenizer with hierarchical boundary detection."""

import re
import json
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn


class HierarchicalTokenizer:
    """BPE-based tokenizer with hierarchical boundary detection.

    For a minimal implementation, this uses a simple subword tokenization
    approach that can be replaced with a pre-trained BPE tokenizer
    (e.g., tiktoken, HuggingFace Tokenizer) for production use.

    The key feature is that it produces boundary information needed by
    the hierarchical composer.
    """

    VOCAB_SIZE: int = 16384
    PAD_TOKEN_ID: int = 0
    UNK_TOKEN_ID: int = 1
    BOS_TOKEN_ID: int = 2
    EOS_TOKEN_ID: int = 3

    # Reserved special tokens
    SPECIAL_TOKENS = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
    }

    def __init__(self, vocab_file: Optional[str] = None) -> None:
        """Initialize tokenizer.

        Args:
            vocab_file: Path to vocabulary JSON. If None, initializes
                       with a basic English vocabulary.
        """
        if vocab_file is not None:
            with open(vocab_file, 'r') as f:
                data = json.load(f)
                self.token_to_id = data['token_to_id']
                self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        else:
            self.token_to_id = dict(self.SPECIAL_TOKENS)
            self.id_to_token = {v: k for k, v in self.SPECIAL_TOKENS.items()}
            self._init_basic_vocab()

        self.vocab_size = len(self.token_to_id)

        # Compile regex patterns for boundary detection
        self._word_boundary_pattern = re.compile(r'\s+')
        self._sentence_boundary_pattern = re.compile(r'(?<=[.!?])\s+')
        # Subword prefix for BPE-style merging
        self._subword_prefix = "##"

    def _init_basic_vocab(self) -> None:
        """Initialize a basic English vocabulary for testing.

        Creates a simple character + common subword vocabulary.
        This is NOT a proper BPE vocabulary - it's for testing only.
        For production, load a pre-trained BPE vocabulary.
        """
        # Common English characters
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        for c in chars:
            if c not in self.token_to_id:
                idx = len(self.token_to_id)
                if idx < self.VOCAB_SIZE:
                    self.token_to_id[c] = idx
                    self.id_to_token[idx] = c

        # Common subwords and punctuation
        common = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have",
            "I", "it", "for", "not", "on", "with", "he", "as", "you",
            "do", "at", "this", "but", "his", "by", "from", "they",
            "we", "say", "her", "she", "or", "an", "will", "my",
            "one", "all", "would", "there", "their", "what", "so",
            "up", "out", "if", "about", "who", "get", "which", "go",
            "me", "when", "make", "can", "like", "time", "no", "just",
            "him", "know", "take", "people", "into", "year", "your",
            "good", "some", "could", "them", "see", "other", "than",
            "then", "now", "look", "only", "come", "its", "over",
            "think", "also", "back", "after", "use", "two", "how",
            "our", "work", "first", "well", "way", "even", "new",
            "want", "because", "any", "these", "give", "day", "most",
            "us", "is", "was", "are", "has", "had", "been", "were",
            "ing", "ed", "er", "ly", "tion", "ment", "ness", "able",
            "ible", "ful", "less", "ous", "ive", "ize", "ise",
            "re", "un", "dis", "over", "under", "out", "up", "down",
            "pre", "post", "anti", "non", "in", "im", "il", "ir",
            "th", "st", "nd", "rd", " est", "er", "ion", "ed",
            "'s", "'t", "'ll", "'ve", "'re", "'d", "n't",
            ".", ",", "!", "?", ";", ":", "-", "_", "'", '"', "(", ")",
            " ", "  ", "\n", "\t",
        ]
        for token in common:
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                if idx < self.VOCAB_SIZE:
                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
        max_length: int = 2048
    ) -> Dict[str, torch.Tensor]:
        """Encode text to token IDs with boundary information.

        Args:
            text: Raw input string.
            add_bos: Prepend BOS token.
            add_eos: Append EOS token.
            max_length: Truncate to this length.

        Returns:
            Dict with keys:
                - "input_ids": (L,) token ID tensor.
                - "word_boundaries": (W, 2) word span tensor [start, end].
                - "sentence_boundaries": (S, 2) sentence span tensor.
                - "attention_mask": (L,) binary mask tensor.
        """
        # Simple whitespace + character tokenization
        # In production, replace with proper BPE tokenization
        tokens = self._tokenize(text)

        # Convert to IDs
        token_ids = []
        for t in tokens:
            tid = self.token_to_id.get(t, self.UNK_TOKEN_ID)
            # Try lowercase if not found
            if tid == self.UNK_TOKEN_ID and t.lower() != t:
                tid = self.token_to_id.get(t.lower(), self.UNK_TOKEN_ID)
            token_ids.append(tid)

        # Add special tokens
        if add_bos:
            token_ids = [self.BOS_TOKEN_ID] + token_ids
        if add_eos:
            token_ids = token_ids + [self.EOS_TOKEN_ID]

        # Truncate
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            if add_eos:
                token_ids[-1] = self.EOS_TOKEN_ID

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.ones(len(token_ids), dtype=torch.long)
        if len(token_ids) > 0 and token_ids[0] == self.PAD_TOKEN_ID:
            attention_mask[0] = 0

        # Detect boundaries
        word_boundaries = self._detect_word_boundaries(tokens, add_bos)
        sentence_boundaries = self._detect_sentence_boundaries(tokens, add_bos)

        return {
            "input_ids": input_ids,
            "word_boundaries": word_boundaries,
            "sentence_boundaries": sentence_boundaries,
            "attention_mask": attention_mask,
            "tokens": tokens,
        }

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: split on whitespace, then characters.

        For production, this should be replaced with proper BPE.
        """
        # Split on whitespace
        words = text.split()
        tokens = []
        for i, word in enumerate(words):
            if i > 0:
                tokens.append(" ")
            # Try to match the whole word first
            if word in self.token_to_id:
                tokens.append(word)
            else:
                # Split into characters
                for c in word:
                    tokens.append(c)
        return tokens

    def _detect_word_boundaries(
        self, tokens: List[str], has_bos: bool
    ) -> torch.Tensor:
        """Detect word boundaries from token sequence.

        Word boundaries occur at whitespace tokens.
        Returns tensor of (start, end) indices for each word.
        """
        offset = 1 if has_bos else 0
        boundaries = []
        start = 0

        for i, tok in enumerate(tokens):
            if tok == " ":
                if start < i:
                    boundaries.append([start + offset, i + offset])
                start = i + 1

        # Last word
        if start < len(tokens):
            boundaries.append([start + offset, len(tokens) + offset])

        if not boundaries:
            boundaries = [[offset, len(tokens) + offset]]

        return torch.tensor(boundaries, dtype=torch.long)

    def _detect_sentence_boundaries(
        self, tokens: List[str], has_bos: bool
    ) -> torch.Tensor:
        """Detect sentence boundaries from token sequence.

        Sentence boundaries occur after . ! ? tokens followed by space.
        """
        offset = 1 if has_bos else 0
        boundaries = []
        start = 0

        for i, tok in enumerate(tokens):
            if tok in ".!?" and i + 1 < len(tokens) and tokens[i + 1] == " ":
                boundaries.append([start + offset, i + 1 + offset])
                start = i + 2

        # Last sentence
        if start < len(tokens):
            boundaries.append([start + offset, len(tokens) + offset])

        if not boundaries:
            boundaries = [[offset, len(tokens) + offset]]

        return torch.tensor(boundaries, dtype=torch.long)

    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special: bool = True
    ) -> str:
        """Decode token IDs back to string.

        Args:
            token_ids: (L,) or (B, L) tensor of token IDs.
            skip_special: Remove special tokens.

        Returns:
            Decoded string (or list of strings if batch).
        """
        if token_ids.dim() == 2:
            return [self.decode(t, skip_special) for t in token_ids]

        tokens = []
        for tid in token_ids.tolist():
            if tid in self.id_to_token:
                tok = self.id_to_token[tid]
                if skip_special and tok in self.SPECIAL_TOKENS:
                    continue
                tokens.append(tok)
            else:
                tokens.append("<unk>")

        return "".join(tokens)

    def get_morpheme_hints(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Get morpheme boundary hints for each token.

        In BPE, subword boundaries often align with morpheme boundaries.
        This returns a score indicating likelihood of morpheme boundary.

        Args:
            token_ids: (L,) tensor.

        Returns:
            (L-1,) float tensor in [0, 1].
        """
        L = token_ids.size(0)
        if L <= 1:
            return torch.zeros(0)

        scores = []
        for i in range(L - 1):
            tid = token_ids[i].item()
            tok = self.id_to_token.get(tid, "")
            # Simple heuristic: short tokens and known affixes suggest boundaries
            if len(tok) <= 2 or tok in {"ing", "ed", "er", "ly", "tion", "ness",
                                          "able", "ful", "less", "ous", "ive",
                                          "un", "dis", "re", "pre", "non",
                                          "'s", "'t", "'ll", "'ve"}:
                scores.append(0.7)
            elif tok == " ":
                scores.append(1.0)
            else:
                scores.append(0.3)

        return torch.tensor(scores, dtype=torch.float32)

    def batch_encode(
        self,
        texts: List[str],
        padding: bool = True,
        max_length: int = 2048
    ) -> Dict[str, torch.Tensor]:
        """Batch encode multiple texts.

        Args:
            texts: List of raw strings.
            padding: Pad to max length in batch.
            max_length: Max sequence length.

        Returns:
            Dict with batched tensors.
        """
        encoded = [self.encode(t, max_length=max_length) for t in texts]

        if padding:
            max_len = max(e["input_ids"].size(0) for e in encoded)
            max_len = min(max_len, max_length)

            batch_ids = []
            batch_mask = []
            batch_word_b = []
            batch_sent_b = []

            for e in encoded:
                L = e["input_ids"].size(0)
                if L < max_len:
                    pad_len = max_len - L
                    ids = torch.cat([
                        e["input_ids"],
                        torch.full((pad_len,), self.PAD_TOKEN_ID, dtype=torch.long)
                    ])
                    mask = torch.cat([
                        e["attention_mask"],
                        torch.zeros(pad_len, dtype=torch.long)
                    ])
                else:
                    ids = e["input_ids"][:max_len]
                    mask = e["attention_mask"][:max_len]

                batch_ids.append(ids)
                batch_mask.append(mask)
                batch_word_b.append(e["word_boundaries"])
                batch_sent_b.append(e["sentence_boundaries"])

            return {
                "input_ids": torch.stack(batch_ids),
                "attention_mask": torch.stack(batch_mask),
                "word_boundaries": batch_word_b,
                "sentence_boundaries": batch_sent_b,
            }
        else:
            return {
                "input_ids": [e["input_ids"] for e in encoded],
                "attention_mask": [e["attention_mask"] for e in encoded],
                "word_boundaries": [e["word_boundaries"] for e in encoded],
                "sentence_boundaries": [e["sentence_boundaries"] for e in encoded],
            }

    def save_vocab(self, path: str) -> None:
        """Save vocabulary to JSON file."""
        with open(path, 'w') as f:
            json.dump({
                'token_to_id': self.token_to_id,
                'id_to_token': self.id_to_token,
            }, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str) -> "HierarchicalTokenizer":
        """Load a pre-trained tokenizer."""
        return cls(vocab_file=path)
