"""Build the dictionary indexes: data/*.json -> data/index/<name>/."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..config import GENERAL_DICT_PATH, INDEX_DIR, MEDICAL_DICT_PATH
from .embedder import Embedder
from .index import VectorIndex

log = logging.getLogger("clinicallingo.build_index")

REQUIRED = ("term", "definition")


def load_dictionary(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data["entries"] if isinstance(data, dict) else data
    seen = set()
    clean = []
    for i, e in enumerate(entries):
        for k in REQUIRED:
            if not e.get(k) or not isinstance(e[k], str):
                raise ValueError(f"{path.name} entry {i} is missing '{k}'")
        key = e["term"].strip().lower()
        if key in seen:
            raise ValueError(f"{path.name}: duplicate term '{e['term']}'")
        seen.add(key)
        clean.append({"term": e["term"].strip(), "definition": e["definition"].strip(),
                      "synonyms": [s.strip() for s in e.get("synonyms", []) if s.strip()],
                      "source": e.get("source") or path.stem, "category": e.get("category", "")})
    return clean


def build_all(embedder: Embedder, index_dir: Path = INDEX_DIR) -> dict[str, int]:
    counts = {}
    for name, path in (("medical_dictionary", MEDICAL_DICT_PATH), ("general_dictionary", GENERAL_DICT_PATH)):
        entries = load_dictionary(path)
        idx = VectorIndex(name).build(entries, embedder)
        idx.save(Path(index_dir) / name)
        counts[name] = len(idx)
        log.info("built %s: %d entries, dim=%s, embedder=%s", name, len(idx), idx.meta.get("dim"), embedder.name)
    return counts


def load_all(index_dir: Path = INDEX_DIR) -> tuple[VectorIndex, VectorIndex]:
    med = VectorIndex.load(Path(index_dir) / "medical_dictionary", "medical_dictionary")
    gen = VectorIndex.load(Path(index_dir) / "general_dictionary", "general_dictionary")
    if len(med) == 0:
        raise FileNotFoundError(f"medical_dictionary index not built; run: clinicallingo build-index (looked in {index_dir})")
    return med, gen
