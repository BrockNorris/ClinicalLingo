"""A small persisted vector index: entries.json + vectors.npy + meta.json.

Sized for a few hundred to a few thousand entries; a numpy dot product is the
whole search. Three instances exist at runtime: medical_dictionary (priority),
general_dictionary, and web_cache (grown at runtime by the web tool).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embedder import Embedder


@dataclass
class Hit:
    score: float
    entry: dict


class VectorIndex:
    def __init__(self, name: str):
        self.name = name
        self.entries: list[dict] = []
        self.vectors: np.ndarray = np.zeros((0, 1), dtype=np.float32)
        self.meta: dict = {}

    # -- building --------------------------------------------------------------------
    @staticmethod
    def entry_text(entry: dict) -> str:
        """The string that gets embedded for a dictionary entry."""
        syn = ", ".join(entry.get("synonyms", []) or [])
        base = f"{entry['term']}: {entry['definition']}"
        return f"{base} (also: {syn})" if syn else base

    def build(self, entries: list[dict], embedder: Embedder) -> "VectorIndex":
        self.entries = list(entries)
        texts = [e.get("embed_text") or self.entry_text(e) for e in self.entries]
        self.vectors = embedder.embed(texts) if texts else np.zeros((0, 1), dtype=np.float32)
        self.meta = {
            "name": self.name,
            "embedder": embedder.name,
            "dim": int(self.vectors.shape[1]) if len(self.entries) else 0,
            "count": len(self.entries),
        }
        return self

    def add(self, entries: list[dict], embedder: Embedder) -> None:
        if not entries:
            return
        texts = [e.get("embed_text") or self.entry_text(e) for e in entries]
        vecs = embedder.embed(texts)
        if len(self.entries) == 0:
            self.vectors = vecs
        else:
            self.vectors = np.vstack([self.vectors, vecs])
        self.entries.extend(entries)
        self.meta.update({"name": self.name, "embedder": embedder.name,
                          "dim": int(self.vectors.shape[1]), "count": len(self.entries)})

    # -- persistence -----------------------------------------------------------------
    def save(self, folder: Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "entries.json").write_text(json.dumps(self.entries, ensure_ascii=False, indent=1), encoding="utf-8")
        np.save(folder / "vectors.npy", self.vectors)
        (folder / "meta.json").write_text(json.dumps(self.meta, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, name: str | None = None) -> "VectorIndex":
        folder = Path(folder)
        idx = cls(name or folder.name)
        if not (folder / "entries.json").exists():
            return idx  # empty index (e.g. web cache before first use)
        idx.entries = json.loads((folder / "entries.json").read_text(encoding="utf-8"))
        idx.vectors = np.load(folder / "vectors.npy")
        idx.meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        if idx.vectors.shape[0] != len(idx.entries):
            raise ValueError(f"index {idx.name} is corrupt: {idx.vectors.shape[0]} vectors for {len(idx.entries)} entries")
        return idx

    def __len__(self) -> int:
        return len(self.entries)

    # -- search ------------------------------------------------------------------------
    def query(self, qvec: np.ndarray, top_k: int = 5, min_score: float = 0.0) -> list[Hit]:
        if len(self.entries) == 0:
            return []
        if qvec.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                f"query dim {qvec.shape[0]} != index dim {self.vectors.shape[1]} "
                f"(index '{self.name}' was built with {self.meta.get('embedder')})"
            )
        scores = self.vectors @ qvec
        order = np.argsort(-scores)[:top_k]
        return [Hit(float(scores[i]), self.entries[i]) for i in order if scores[i] >= min_score]

    def find_term(self, term: str) -> dict | None:
        t = term.strip().lower()
        for e in self.entries:
            if e["term"].lower() == t or t in [s.lower() for s in e.get("synonyms", [])]:
                return e
        return None
