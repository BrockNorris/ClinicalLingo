"""Orchestration: runs the stages in order, times each one, and writes the run log.

    explain():  validate -> retrieve [-> web_search] -> build_prompt -> infer
                -> postprocess [-> infer_retry -> postprocess] -> output
    quiz():     make_questions -> (score -> verdict -> follow_up) per question

Failure injection (--inject-failure) is applied here so that the write-up can
name exactly which boundary caught each injected fault:
    oversized_input   pads the text past the mode's limit    (caught by Validation)
    empty_retrieval   discards the retrieval result           (handled at Retriever -> Prompt Builder)
    malformed_json    model answers in prose                  (caught at Model -> Postprocess)
    empty_output      model returns ""                        (caught at Model -> Postprocess)
    model_timeout     model raises a timeout                  (caught at Prompt Builder -> Model)
"""
from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

from .config import INDEX_DIR, INPUT_LIMITS, RUN_LOG_PATH, WEB_CACHE_DIR
from .contracts import ExplainResult, Prompt, QuizRound, RawInput, RetrievalResult, RunLog
from .feedback import FeedbackLoop
from .inference.backend import FaultyBackend, ModelBackend, ModelError, ModelTimeout, OllamaBackend
from .output import render
from .postprocess import to_result
from .prompts import build_prompt
from .retrieval.build_index import load_all
from .retrieval.embedder import Embedder, OllamaEmbedder
from .retrieval.index import VectorIndex
from .retrieval.retriever import Retriever
from .runlog import StageTimer, new_run_id, now_iso, write_run
from .tools.web_search import WebSearchTool
from .validation import InputValidationError, validate

log = logging.getLogger("clinicallingo.pipeline")

INJECTABLE = ("oversized_input", "empty_retrieval", "malformed_json", "empty_output", "model_timeout")


class Pipeline:
    def __init__(self, backend: ModelBackend | None = None, embedder: Embedder | None = None,
                 retriever: Retriever | None = None, web_tool: WebSearchTool | None = None,
                 allow_web: bool = True, inject: str | None = None,
                 index_dir: Path = INDEX_DIR, web_cache_dir: Path = WEB_CACHE_DIR,
                 run_log_path: Path = RUN_LOG_PATH):
        if inject and inject not in INJECTABLE:
            raise ValueError(f"unknown failure '{inject}', choose from {INJECTABLE}")
        self.inject = inject
        self.allow_web = allow_web
        self.run_log_path = run_log_path
        self.embedder = embedder or OllamaEmbedder()
        if retriever is None:
            med, gen = load_all(index_dir)
            web_cache = VectorIndex.load(web_cache_dir, "web_cache")
            retriever = Retriever(self.embedder, med, gen, web_cache,
                                  web_tool=web_tool if web_tool is not None else WebSearchTool(),
                                  web_cache_dir=web_cache_dir)
        self.retriever = retriever
        base = backend or OllamaBackend()
        self.backend = FaultyBackend(base, inject) if inject in ("malformed_json", "empty_output", "model_timeout") else base
        self.feedback = FeedbackLoop(self.backend, self.embedder)

    # -- explain -------------------------------------------------------------------------
    def explain(self, text: str, mode: str, force_web: bool = False
                ) -> tuple[ExplainResult | None, RetrievalResult | None, RunLog]:
        run_id, timer = new_run_id(), StageTimer()
        model_info = {"backend": self.backend.name, "retries": 0}
        retrieval_info: dict = {}
        result: ExplainResult | None = None
        retrieval: RetrievalResult | None = None
        error: str | None = None

        if self.inject == "oversized_input":
            text = text + " x" * (INPUT_LIMITS.get(mode, 64) // 2 + 2)

        # 1. validation
        try:
            with timer.stage("validate"):
                req = validate(RawInput(text=text, mode=mode))
        except InputValidationError as e:
            error = f"rejected_input: {e.reason}"
            log.warning("validation rejected input: %s", e.reason)
            entry = RunLog(run_id=run_id, ts=now_iso(), command="explain", mode=mode, input_chars=len(text or ""),
                           stages=timer.stages, latency_ms=timer.finish(), status="rejected_input",
                           error=error, injected_failure=self.inject)
            write_run(entry, self.run_log_path)
            return None, None, entry

        # 2. retrieval (+ tool call inside)
        with timer.stage("retrieve"):
            t0 = perf_counter()
            retrieval = self.retriever.retrieve(req, allow_web=self.allow_web, force_web=force_web)
            if self.inject == "empty_retrieval":
                retrieval = RetrievalResult(chunks=[], coverage=0.0, used_web=False,
                                            web_error="injected: retrieval result discarded")
        if retrieval.used_web:
            timer.add("web_search", (perf_counter() - t0) * 1000)  # tool time is included in retrieve; recorded for visibility
        retrieval_info = {
            "k": len(retrieval.chunks), "coverage": round(retrieval.coverage, 3),
            "top_score": round(retrieval.chunks[0].score, 3) if retrieval.chunks else None,
            "used_web": retrieval.used_web, "web_cache_hit": retrieval.web_cache_hit,
            "lexical_hits": retrieval.lexical_hits,
            "indexes": sorted({c.index_name for c in retrieval.chunks}),
            "sources": [c.source for c in retrieval.chunks],
            "web_error": retrieval.web_error,
        }

        # 3. prompt + 4. inference + 5. postprocess (with one retry on parse failure)
        with timer.stage("build_prompt"):
            prompt = build_prompt(req, retrieval)
        retry = Prompt(system=prompt.system, max_tokens=prompt.max_tokens, timeout_s=prompt.timeout_s,
                       json_mode=True, user=prompt.user + "\n\nYour previous reply was not a valid JSON "
                       "object. Reply with ONLY the JSON object, no prose, no markdown.")
        try:
            try:
                with timer.stage("infer"):
                    raw = self.backend.generate(prompt)
                model_info.update(getattr(self.backend, "last_stats", {}) or {})
                with timer.stage("postprocess"):
                    result = to_result(req, retrieval, raw)
            except ModelTimeout:
                raise
            except ModelError as e:
                # e.g. Ollama HTTP 500 "prediction aborted, token repeat limit reached": worth one retry
                log.warning("model error (%s); retrying once", e)
                model_info["first_error"] = str(e)[:200]
                result = ExplainResult(mode=req.mode, status="parse_failed", message=str(e))
            if result.status == "parse_failed":
                log.info("retrying once with a stricter instruction")
                model_info["retries"] = 1
                with timer.stage("infer_retry"):
                    raw = self.backend.generate(retry)
                model_info.update(getattr(self.backend, "last_stats", {}) or {})
                with timer.stage("postprocess_retry"):
                    result = to_result(req, retrieval, raw)
        except ModelTimeout as e:
            error = str(e)
            result = ExplainResult(mode=req.mode, status="model_timeout", message=str(e))
        except ModelError as e:
            error = str(e)
            result = ExplainResult(mode=req.mode, status="model_error", message=str(e))

        # 6. output
        with timer.stage("output"):
            rendered = render(result, retrieval)

        info = {}
        try:
            info = self.backend.info()
        except Exception:  # pragma: no cover - info is best effort
            pass
        model_info = {**info, **model_info}
        entry = RunLog(run_id=run_id, ts=now_iso(), command="explain", mode=req.mode, input_chars=req.char_count,
                       stages=timer.stages, latency_ms=timer.finish(), retrieval=retrieval_info, model=model_info,
                       status=result.status, grounded=result.grounded, low_confidence=result.low_confidence,
                       output={"rendered": rendered, "result": result.model_dump(exclude_none=True)},
                       error=error, injected_failure=self.inject)
        write_run(entry, self.run_log_path)
        return result, retrieval, entry

    # -- feedback loop -------------------------------------------------------------------
    def quiz(self, result: ExplainResult, retrieval: RetrievalResult, answer_fn, n: int = 2
             ) -> tuple[list[QuizRound], RunLog]:
        """answer_fn(question_text) -> student's answer. Returns the rounds and the run log."""
        run_id, timer = new_run_id(), StageTimer()
        rounds: list[QuizRound] = []
        with timer.stage("make_questions"):
            questions = self.feedback.make_questions(result, retrieval, n)
        for i, q in enumerate(questions, 1):
            with timer.stage(f"q{i}_answer"):
                ans = answer_fn(q.question) or ""
            with timer.stage(f"q{i}_score_followup"):
                rounds.append(self.feedback.run_round(q, ans, retrieval))
        entry = RunLog(run_id=run_id, ts=now_iso(), command="quiz", mode=result.mode,
                       input_chars=sum(len(r.user_answer) for r in rounds), stages=timer.stages,
                       latency_ms=timer.finish(), model={"backend": self.backend.name,
                                                         "question_source": self.feedback.last_question_source},
                       retrieval={"k": len(retrieval.chunks)},
                       status="ok" if rounds else "no_questions",
                       output={"rounds": [r.model_dump() for r in rounds]}, injected_failure=self.inject)
        write_run(entry, self.run_log_path)
        return rounds, entry
