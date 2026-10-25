"""Embedding model access (all-MiniLM-L6-v2 served by Ollama as `all-minilm`).

Contract: embed(list[str]) -> np.ndarray of shape (n, dim), rows L2-normalised,
so a dot product is a cosine similarity.
"""
from __future__ import annotations

import hashlib

import numpy as np
import requests

from ..config import EMBED_MODEL, EMBED_TIMEOUT_S, ollama_base_url


class EmbedderError(RuntimeError):
    pass


class Embedder:
    """Protocol-ish base; subclasses implement _embed_raw."""

    name: str = "base"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vecs = np.asarray(self._embed_raw(texts), dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] != len(texts):
            raise EmbedderError(f"embedder returned shape {vecs.shape} for {len(texts)} texts")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def _embed_raw(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError


class OllamaEmbedder(Embedder):
    def __init__(self, model: str = EMBED_MODEL, base_url: str | None = None, batch_size: int = 64):
        self.model = model
        self.name = f"ollama:{model}"
        self.base_url = base_url or ollama_base_url()
        self.batch_size = batch_size

    def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                r = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": batch},
                    timeout=EMBED_TIMEOUT_S,
                )
                r.raise_for_status()
            except requests.RequestException as e:
                raise EmbedderError(f"Ollama embed request failed ({self.model}): {e}") from e
            data = r.json()
            embs = data.get("embeddings")
            if not embs:
                raise EmbedderError(f"Ollama returned no embeddings: {data}")
            out.extend(embs)
        return out


class HashEmbedder(Embedder):
    """Deterministic, dependency-free stand-in used by the unit tests so they run
    without Ollama. Bag-of-hashed-character-trigrams; crude but gives higher
    similarity to lexically similar strings, which is all the tests rely on."""

    name = "hash-trigram"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        rows = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            s = f"  {t.lower()}  "
            for i in range(len(s) - 2):
                h = int(hashlib.md5(s[i : i + 3].encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            rows.append(v.tolist())
        return rows
