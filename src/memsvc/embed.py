"""Text embedders. MiniLM for real use; a hashing embedder for fast, deterministic tests."""

from __future__ import annotations

import hashlib
import itertools
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

DIM = 384


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Feature hashing of words and word pairs. No model, no download; similar only lexically."""

    name = "hash-384"
    dim = DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", t.lower())
            for feat in words + [a + "_" + b for a, b in itertools.pairwise(words)]:
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class MiniLM:
    """sentence-transformers/all-MiniLM-L6-v2 with mean pooling, on CPU. Install the `embed` extra."""

    name = "all-MiniLM-L6-v2"
    dim = DIM

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", batch: int = 64):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).eval()
        self.batch = batch
        torch.set_num_threads(max(1, torch.get_num_threads()))

    def embed(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        chunks = []
        for i in range(0, len(texts), self.batch):
            enc = self.tokenizer(texts[i : i + self.batch], padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.inference_mode():
                hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            chunks.append(torch.nn.functional.normalize(pooled, dim=1).numpy())
        return np.vstack(chunks).astype(np.float32) if chunks else np.zeros((0, self.dim), np.float32)


@lru_cache(maxsize=2)
def get(name: str) -> Embedder:
    if name in ("hash", "hash-384"):
        return HashEmbedder()
    return MiniLM()
