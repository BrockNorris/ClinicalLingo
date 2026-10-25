"""Stage: Retrieval.  ValidatedRequest -> RetrievalResult.

Hybrid retrieval over three indexes, in priority order:
    1. medical_dictionary  (lexical whole-term hits score 1.0, then semantic hits)
    2. general_dictionary
    3. web_cache           (passages previously fetched by the web tool)

Web tool trigger (word mode only, and only when allowed): if the best
medical-dictionary score is below WEB_TRIGGER_SCORE and the web cache has no
usable passage, call the tool once, keep passages that pass the relevance
check, persist them into web_cache, and re-query.

Contract guarantees: len(chunks) <= top_k for the mode; every chunk.score >= MIN_SCORE;
coverage == best medical score (0.0 if none). Never raises for "nothing found";
web failures are reported in web_error, not raised.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from ..config import (MIN_SCORE, TOP_K_BY_MODE, WEB_CACHE_DIR, WEB_RELEVANCE_SCORE,
                      WEB_TRIGGER_SCORE)
from ..contracts import Chunk, RetrievalResult, ValidatedRequest
from .embedder import Embedder
from .index import Hit, VectorIndex

log = logging.getLogger("clinicallingo.retrieval")

_PRIORITY = {"medical_dictionary": 1.00, "general_dictionary": 0.90, "web_cache": 0.85}


def _to_chunk(hit: Hit, index_name: str) -> Chunk:
    e = hit.entry
    return Chunk(
        term=e["term"],
        text=e["definition"],
        source=e.get("source", index_name),
        score=max(-1.0, min(1.0, hit.score)),
        index_name=index_name,  # type: ignore[arg-type]
        synonyms=list(e.get("synonyms", []) or []),
    )


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        medical: VectorIndex,
        general: VectorIndex,
        web_cache: VectorIndex | None = None,
        web_tool=None,
        web_cache_dir: Path = WEB_CACHE_DIR,
    ):
        self.embedder = embedder
        self.medical = medical
        self.general = general
        self.web_cache = web_cache or VectorIndex("web_cache")
        self.web_tool = web_tool
        self.web_cache_dir = Path(web_cache_dir)

    # -- lexical pass ---------------------------------------------------------------
    def _lexical_hits(self, text: str, index: VectorIndex) -> list[Hit]:
        low = text.lower()
        hits = []
        for e in index.entries:
            names = [e["term"]] + list(e.get("synonyms", []) or [])
            for n in names:
                n = n.lower().strip()
                if len(n) < 3:
                    continue
                if re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", low):
                    hits.append(Hit(1.0, e))
                    break
        return hits

    # -- main entry ------------------------------------------------------------------
    def retrieve(self, req: ValidatedRequest, allow_web: bool = True, force_web: bool = False) -> RetrievalResult:
        top_k = TOP_K_BY_MODE[req.mode]
        qvec = self.embedder.embed_one(req.text)

        candidates: dict[tuple[str, str], Chunk] = {}

        def consider(hit: Hit, index_name: str):
            c = _to_chunk(hit, index_name)
            key = (index_name, c.term.lower())
            if key not in candidates or candidates[key].score < c.score:
                candidates[key] = c

        # 1. lexical whole-term hits (exact matches are the most trustworthy evidence)
        lexical_terms: list[str] = []
        for idx, name in ((self.medical, "medical_dictionary"), (self.general, "general_dictionary")):
            for h in self._lexical_hits(req.text, idx):
                consider(h, name)
                if name == "medical_dictionary":
                    lexical_terms.append(h.entry["term"])

        # 2. semantic hits
        for idx, name in ((self.medical, "medical_dictionary"),
                          (self.general, "general_dictionary"),
                          (self.web_cache, "web_cache")):
            for h in idx.query(qvec, top_k=top_k, min_score=MIN_SCORE):
                consider(h, name)

        coverage = max((c.score for c in candidates.values() if c.index_name == "medical_dictionary"), default=0.0)
        cache_hit = any(c.index_name == "web_cache" for c in candidates.values())

        # 3. web tool, only when the dictionaries are weak (or the caller forces it)
        used_web, web_error = False, None
        need_web = force_web or (req.mode == "word" and coverage < WEB_TRIGGER_SCORE and not cache_hit)
        if allow_web and need_web and self.web_tool is not None:
            used_web = True
            web_error = self._fetch_into_cache(req.text, qvec)
            for h in self.web_cache.query(qvec, top_k=top_k, min_score=MIN_SCORE):
                consider(h, "web_cache")
        elif need_web and self.web_tool is None and allow_web:
            web_error = "web tool not configured"

        ranked = sorted(candidates.values(), key=lambda c: c.score * _PRIORITY[c.index_name], reverse=True)
        chunks = ranked[:top_k]
        log.info("retrieval: %d candidates -> %d chunks, coverage=%.2f, web=%s, cache_hit=%s",
                 len(candidates), len(chunks), coverage, used_web, cache_hit)
        return RetrievalResult(
            chunks=chunks,
            coverage=float(max(0.0, min(1.0, coverage))),
            used_web=used_web,
            web_error=web_error,
            lexical_hits=lexical_terms,
            web_cache_hit=cache_hit,
        )

    # -- web tool integration ------------------------------------------------------
    def _fetch_into_cache(self, term: str, qvec: np.ndarray) -> str | None:
        """Call the tool, relevance-filter its passages against the query, persist. Returns error text or None."""
        result = self.web_tool.lookup(term)
        if result.error and not result.chunks:
            log.warning("web tool: %s", result.error)
            return result.error
        if not result.chunks:
            return "web tool returned no passages"

        texts = [c.text for c in result.chunks]
        vecs = self.embedder.embed(texts)
        sims = vecs @ qvec
        entries = []
        for c, s in zip(result.chunks, sims):
            if s >= WEB_RELEVANCE_SCORE:  # relevance double-check
                entries.append({
                    "term": term, "definition": c.text, "synonyms": [],
                    "source": c.url, "domain": c.domain, "title": c.title,
                    "embed_text": c.text, "relevance_at_fetch": float(s),
                })
        log.info("web tool: %d passages fetched, %d passed relevance >= %.2f",
                 len(result.chunks), len(entries), WEB_RELEVANCE_SCORE)
        if not entries:
            return f"web tool: {len(result.chunks)} passages fetched but none relevant"
        self.web_cache.add(entries, self.embedder)
        try:
            self.web_cache.save(self.web_cache_dir)
        except OSError as e:  # cache persistence is best-effort
            log.warning("could not persist web cache: %s", e)
        return result.error
