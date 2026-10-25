"""Stage: Postprocessing.  raw model str -> ExplainResult.

1. Lenient JSON extraction (first '{' to last '}').
2. Pydantic validation into DefinitionCard or Rewrite.
3. Grounding check: did the output mention the retrieved terms / cite sources?
4. Confidence flag: low unless status == ok, grounded, and retrieval found something.

Parse failures are returned as status="parse_failed" with the raw text in
fallback_text; this stage never raises on bad model output.
"""
from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from .contracts import (Chunk, DefinitionCard, ExplainResult, RetrievalResult, Rewrite,
                        ValidatedRequest)
from .prompts import NOT_FOUND_MARKER

log = logging.getLogger("clinicallingo.postprocess")


class ParseFailure(ValueError):
    pass


def parse_json_lenient(raw: str) -> dict:
    if not raw or not raw.strip():
        raise ParseFailure("model returned empty output")
    s = raw.strip()
    # strip ```json fences if present
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ParseFailure("no JSON object found in model output")
        try:
            obj = json.loads(s[start : end + 1])
        except json.JSONDecodeError as e:
            raise ParseFailure(f"JSON decode error: {e.msg} at pos {e.pos}")
    if not isinstance(obj, dict):
        raise ParseFailure(f"expected a JSON object, got {type(obj).__name__}")
    return obj


def _mentions(text: str, chunks: list[Chunk]) -> list[str]:
    low = text.lower()
    found = []
    for c in chunks:
        names = [c.term] + list(c.synonyms)
        if any(re.search(r"(?<![a-z0-9])" + re.escape(n.lower()) + r"(?![a-z0-9])", low) for n in names if len(n) >= 3):
            found.append(c.term)
    return found


def grounding_check(result_text: str, used_sources: list[str], retrieval: RetrievalResult) -> tuple[bool, list[str]]:
    """Grounded if the output cites at least one entry or mentions at least one retrieved term."""
    if not retrieval.chunks:
        return False, []
    cited = []
    for s in used_sources:
        m = re.search(r"\d+", str(s))
        if m and 1 <= int(m.group()) <= len(retrieval.chunks):
            cited.append(retrieval.chunks[int(m.group()) - 1].source)
    mentioned = _mentions(result_text, retrieval.chunks)
    sources = cited or [retrieval.chunks[i].source for i, c in enumerate(retrieval.chunks) if c.term in mentioned]
    return bool(cited or mentioned), list(dict.fromkeys(sources))


def to_result(req: ValidatedRequest, retrieval: RetrievalResult, raw: str) -> ExplainResult:
    try:
        obj = parse_json_lenient(raw)
    except ParseFailure as e:
        log.warning("postprocess: %s", e)
        return ExplainResult(mode=req.mode, status="parse_failed", fallback_text=raw[:2000] or None,
                             grounded=False, low_confidence=True, message=str(e))

    try:
        if req.mode == "word":
            obj.setdefault("term", req.text)
            card = DefinitionCard(**{k: obj.get(k, [] if k in ("synonyms", "used_sources") else "") for k in
                                     ("term", "plain_definition", "clinical_definition", "synonyms", "used_sources")})
            # With no retrieved entries there is nothing to cite, so any answer is unverified by contract.
            if NOT_FOUND_MARKER in card.plain_definition.upper() or not retrieval.chunks:
                card.used_sources = []
                return ExplainResult(mode=req.mode, status="not_found", card=card, grounded=False,
                                     low_confidence=True, message="term not found in dictionary; definition is unverified")
            grounded, sources = grounding_check(
                f"{card.plain_definition} {card.clinical_definition} {' '.join(card.synonyms)}",
                card.used_sources, retrieval)
            return ExplainResult(mode=req.mode, status="ok", card=card, grounded=grounded,
                                 low_confidence=not grounded, sources=sources)
        else:
            # key_terms is best-effort: a malformed item (missing/empty/renamed field) is dropped, not fatal.
            raw_terms = obj.get("key_terms") or []
            good = [kt for kt in raw_terms if isinstance(kt, dict)
                    and isinstance(kt.get("term"), str) and kt["term"].strip()
                    and isinstance(kt.get("plain_meaning"), str) and kt["plain_meaning"].strip()]
            if len(good) < len(raw_terms):
                log.warning("postprocess: dropped %d malformed key_terms item(s)", len(raw_terms) - len(good))
            rewrite = Rewrite(original=req.text, rewritten=str(obj.get("rewritten", "") or ""),
                              key_terms=[{"term": kt["term"].strip(), "plain_meaning": kt["plain_meaning"].strip()} for kt in good])
            if not rewrite.rewritten.strip():
                raise ParseFailure("rewritten text is empty")
            grounded, sources = grounding_check(
                rewrite.rewritten + " " + " ".join(k.term for k in rewrite.key_terms), [], retrieval)
            return ExplainResult(mode=req.mode, status="ok", rewrite=rewrite, grounded=grounded,
                                 low_confidence=not grounded or not retrieval.chunks, sources=sources)
    except (ValidationError, ParseFailure, TypeError) as e:
        log.warning("postprocess: schema validation failed: %s", e)
        return ExplainResult(mode=req.mode, status="parse_failed", fallback_text=raw[:2000] or None,
                             grounded=False, low_confidence=True, message=f"schema validation failed: {e}")
