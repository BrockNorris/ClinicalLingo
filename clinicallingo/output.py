"""Stage: Output.  ExplainResult -> text for the terminal (or JSON with --json)."""
from __future__ import annotations

from .contracts import ExplainResult, QuizRound, RetrievalResult


def render(result: ExplainResult, retrieval: RetrievalResult | None = None) -> str:
    lines: list[str] = []
    flag = " [LOW CONFIDENCE]" if result.low_confidence else ""
    if result.status == "ok" and result.card:
        c = result.card
        lines += [f"== {c.term} =={flag}",
                  f"Plain:    {c.plain_definition}",
                  f"Clinical: {c.clinical_definition}"]
        if c.synonyms:
            lines.append(f"Synonyms: {', '.join(c.synonyms)}")
    elif result.status == "ok" and result.rewrite:
        r = result.rewrite
        lines += [f"== Plain-language rewrite =={flag}", r.rewritten]
        if r.key_terms:
            lines.append("Key terms:")
            lines += [f"  - {k.term}: {k.plain_meaning}" for k in r.key_terms]
    elif result.status == "not_found":
        lines.append(f"== {result.card.term if result.card else 'term'} == [NOT FOUND IN DICTIONARY]")
        if result.card and result.card.clinical_definition:
            lines.append(f"Unverified: {result.card.clinical_definition}")
        lines.append("Tip: try --web to search allow-listed medical sites, or check the spelling.")
    else:
        lines.append(f"== Could not produce a structured answer == (status: {result.status})")
        if result.message:
            lines.append(f"Reason: {result.message}")
        if result.fallback_text:
            lines.append("Raw model output:")
            lines.append(result.fallback_text)
    if result.sources:
        lines.append("Sources: " + "; ".join(result.sources))
    if retrieval is not None and retrieval.chunks:
        lines.append("Retrieved: " + ", ".join(f"{c.term} ({c.score:.2f}, {c.index_name})" for c in retrieval.chunks))
    if retrieval is not None and retrieval.web_error:
        lines.append(f"Web tool note: {retrieval.web_error}")
    return "\n".join(lines)


def render_quiz(rounds: list[QuizRound]) -> str:
    out = []
    for i, r in enumerate(rounds, 1):
        out += [f"Q{i}: {r.question}",
                f"   You said:   {r.user_answer}",
                f"   Similarity: {r.similarity:.2f} -> {r.verdict}",
                f"   {r.follow_up}"]
    return "\n".join(out)
