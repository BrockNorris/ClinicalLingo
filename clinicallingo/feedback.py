"""Feedback loop (simple agent loop): quiz -> score -> adapt.

After an explanation, the model writes comprehension questions grounded in the
retrieved entries. The student's free-response answer is embedded and compared
to the reference answer with cosine similarity (the "similarity metric").
    similarity <  QUIZ_DEEPER_BELOW (0.40) -> "deeper"  : model re-explains more simply
    similarity >= QUIZ_RELATED_ABOVE (0.58) -> "related" : correct; show neighbouring terms to go further
    otherwise                               -> "clarify" : mostly right; add the textbook wording
Thresholds come from examples/calibrate_quiz.py (see config.py for the measured distributions).

If the model fails to produce usable questions, deterministic fallback
questions are built straight from the retrieved entries, so the loop still runs.
"""
from __future__ import annotations

import logging

from .config import QUIZ_DEEPER_BELOW, QUIZ_RELATED_ABOVE
from .contracts import ExplainResult, QuizQuestion, QuizRound, RetrievalResult, Verdict
from .inference.backend import ModelBackend, ModelError
from .postprocess import ParseFailure, parse_json_lenient
from .prompts import build_deeper_prompt, build_quiz_prompt
from .retrieval.embedder import Embedder

log = logging.getLogger("clinicallingo.feedback")


def fallback_questions(retrieval: RetrievalResult, n: int) -> list[QuizQuestion]:
    qs = []
    for c in retrieval.chunks[:n]:
        qs.append(QuizQuestion(
            question=f'In your own words, what does "{c.term}" mean?',
            reference_answer=c.text,
            based_on_term=c.term,
        ))
    return qs


class FeedbackLoop:
    def __init__(self, backend: ModelBackend, embedder: Embedder):
        self.backend = backend
        self.embedder = embedder
        self.last_question_source = "model"

    def make_questions(self, result: ExplainResult, retrieval: RetrievalResult, n: int = 2) -> list[QuizQuestion]:
        if not retrieval.chunks:
            return []  # nothing to ground questions in
        try:
            raw = self.backend.generate(build_quiz_prompt(result, retrieval, n))
            obj = parse_json_lenient(raw)
            qs = []
            for q in obj.get("questions", [])[:n]:
                if isinstance(q, dict) and q.get("question") and q.get("reference_answer"):
                    qs.append(QuizQuestion(question=str(q["question"]), reference_answer=str(q["reference_answer"]),
                                           based_on_term=str(q.get("based_on_term") or retrieval.chunks[0].term)))
            if qs:
                self.last_question_source = "model"
                return qs
            raise ParseFailure("no usable questions in model output")
        except (ModelError, ParseFailure) as e:
            log.warning("quiz generation failed (%s); using fallback questions", e)
            self.last_question_source = "fallback"
            return fallback_questions(retrieval, n)

    def score(self, user_answer: str, q: QuizQuestion, retrieval: RetrievalResult | None = None) -> float:
        """Cosine similarity of the student's answer to the reference answer, or to the dictionary
        entry the question is about, whichever is higher. The model-written reference can drift
        from the entry's wording; the entry is the ground truth the student actually read."""
        if not user_answer.strip():
            return 0.0
        targets = [q.reference_answer]
        if retrieval is not None:
            targets += [c.text for c in retrieval.chunks if c.term.lower() == q.based_on_term.lower()][:1]
        vecs = self.embedder.embed([user_answer] + targets)
        return float(max(vecs[0] @ v for v in vecs[1:]))

    @staticmethod
    def verdict(similarity: float) -> Verdict:
        if similarity < QUIZ_DEEPER_BELOW:
            return "deeper"
        if similarity >= QUIZ_RELATED_ABOVE:
            return "related"
        return "clarify"

    def follow_up(self, verdict: Verdict, q: QuizQuestion, user_answer: str, retrieval: RetrievalResult) -> str:
        if verdict == "deeper":
            try:
                text = self.backend.generate(build_deeper_prompt(q.question, q.reference_answer, user_answer, retrieval))
                if text.strip():
                    return "Let's go deeper. " + text.strip()
            except ModelError as e:
                log.warning("deeper explanation failed: %s", e)
            return f"Let's go deeper. The key idea: {q.reference_answer}"
        if verdict == "clarify":
            return f"Mostly there. The textbook wording adds: {q.reference_answer}"
        others = [c for c in retrieval.chunks if c.term.lower() != q.based_on_term.lower()][:3]
        if not others:
            return "Correct. You have this one; nothing more to add from the dictionary."
        lines = [f"- {c.term}: {c.text.split('. ')[0]}." for c in others]
        return "Correct. Related terms you will meet nearby:\n" + "\n".join(lines)

    def run_round(self, q: QuizQuestion, user_answer: str, retrieval: RetrievalResult) -> QuizRound:
        sim = self.score(user_answer, q, retrieval)
        v = self.verdict(sim)
        fu = self.follow_up(v, q, user_answer, retrieval)
        log.info("quiz: similarity=%.2f verdict=%s", sim, v)
        return QuizRound(question=q.question, reference_answer=q.reference_answer, user_answer=user_answer,
                         similarity=max(-1.0, min(1.0, sim)), verdict=v, follow_up=fu)
