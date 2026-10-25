"""Stage: Prompt Builder.  (ValidatedRequest, RetrievalResult) -> Prompt.

The model is told to use only the retrieved entries. When the retrieval
result is empty, the prompt says so explicitly and asks for the NOT_FOUND
marker, which Postprocessing turns into status="not_found".
"""
from __future__ import annotations

from .config import GEN_MAX_TOKENS, GEN_TIMEOUT_S
from .contracts import ExplainResult, Prompt, RetrievalResult, ValidatedRequest

NOT_FOUND_MARKER = "NOT_FOUND_IN_DICTIONARY"

SYSTEM = (
    "You are ClinicalLingo, a study aid for diagnostic medical sonography students. "
    "You explain medical and ultrasound terminology from their textbooks in plain language "
    "while preserving the exact clinical meaning. You must ground every explanation in the "
    "dictionary entries provided in the message; do not invent definitions. "
    "Always answer with a single JSON object and nothing else."
)


def format_context(retrieval: RetrievalResult) -> str:
    if not retrieval.chunks:
        return "(no dictionary entries were found for this request)"
    lines = []
    for i, c in enumerate(retrieval.chunks, 1):
        syn = f" Synonyms: {', '.join(c.synonyms)}." if c.synonyms else ""
        src = c.index_name if c.index_name != "web_cache" else f"web: {c.source}"
        lines.append(f"[{i}] {c.term}: {c.text}{syn} (source: {src})")
    return "\n".join(lines)


def build_prompt(req: ValidatedRequest, retrieval: RetrievalResult) -> Prompt:
    ctx = format_context(retrieval)
    if req.mode == "word":
        user = f"""Term to explain: "{req.text}"

Dictionary entries:
{ctx}

Produce a definition card for a sonography student as JSON with exactly these keys:
{{"term": "<the term>",
 "plain_definition": "<1-2 sentences a first-year student would understand>",
 "clinical_definition": "<the precise clinical meaning, 1-2 sentences>",
 "synonyms": ["<synonym>", ...],
 "used_sources": ["[1]", ...]}}

Rules:
- Base both definitions on the numbered entries and list the ones you used in used_sources.
- If NO entry describes this term, set plain_definition to "{NOT_FOUND_MARKER}", leave used_sources empty,
  and put your best unverified guess in clinical_definition prefixed with "UNVERIFIED: ".
- Do not include any text outside the JSON object."""
        max_tokens = GEN_MAX_TOKENS
    else:
        label = "sentence" if req.mode == "sentence" else "passage"
        user = f"""Textbook {label}:
\"\"\"{req.text}\"\"\"

Dictionary entries for terms that appear in it:
{ctx}

Rewrite the {label} in plain language for a first-year sonography student. Keep every clinical fact,
measurement, and relationship exactly as stated; only simplify the vocabulary and sentence structure.
Answer as JSON with exactly these keys:
{{"rewritten": "<the plain-language version>",
 "key_terms": [{{"term": "<medical term from the text>", "plain_meaning": "<its meaning in plain words, under 20 words>"}}, ...]}}

Rules:
- Use the numbered entries for the meanings of key_terms; include every term that has an entry.
- If a term in the text has no entry, you may still include it but keep its plain_meaning cautious.
- Do not include any text outside the JSON object."""
        max_tokens = GEN_MAX_TOKENS if req.mode == "sentence" else GEN_MAX_TOKENS * 3
    return Prompt(system=SYSTEM, user=user, max_tokens=max_tokens, timeout_s=GEN_TIMEOUT_S, json_mode=True)


# --- feedback loop prompts ------------------------------------------------------------

def build_quiz_prompt(result: ExplainResult, retrieval: RetrievalResult, n: int = 2) -> Prompt:
    ctx = format_context(retrieval)
    if result.card:
        focus = f'The student just read this definition card for "{result.card.term}":\n{result.card.model_dump_json(indent=1)}'
    elif result.rewrite:
        focus = f"The student just read this plain-language rewrite:\n{result.rewrite.rewritten}"
    else:
        focus = "The student just read an explanation."
    user = f"""{focus}

Dictionary entries the explanation was based on:
{ctx}

Write {n} short comprehension questions that check whether the student understood the clinical meaning.
Each must be answerable from the entries above in one or two sentences. Answer as JSON:
{{"questions": [{{"question": "<question>", "reference_answer": "<model answer, 1-2 sentences>",
                 "based_on_term": "<the term the question is about>"}}, ...]}}
Do not include any text outside the JSON object."""
    return Prompt(system=SYSTEM, user=user, max_tokens=GEN_MAX_TOKENS, timeout_s=GEN_TIMEOUT_S, json_mode=True)


def build_deeper_prompt(question: str, reference: str, student_answer: str, retrieval: RetrievalResult) -> Prompt:
    ctx = format_context(retrieval)
    user = f"""A sonography student answered a comprehension question and showed a gap in understanding.

Question: {question}
Expected answer: {reference}
Student's answer: {student_answer}

Dictionary entries:
{ctx}

Explain the concept again in a simpler way, in at most 4 sentences, directly addressing what the student
missed. Use an everyday analogy if it helps. Use only the dictionary entries as your source of facts.
Answer in plain text, not JSON."""
    system = SYSTEM.replace("Always answer with a single JSON object and nothing else.", "Answer in plain text.")
    return Prompt(system=system, user=user, max_tokens=GEN_MAX_TOKENS, timeout_s=GEN_TIMEOUT_S, json_mode=False)
