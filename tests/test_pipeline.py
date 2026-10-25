import json

import pytest

from clinicallingo.inference.backend import FakeBackend
from clinicallingo.pipeline import Pipeline


def make_pipeline(backend, embedder, retriever, run_log, **kw):
    return Pipeline(backend=backend, embedder=embedder, retriever=retriever, run_log_path=run_log, allow_web=False, **kw)


def read_log(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def test_explain_word_end_to_end(backend, embedder, retriever, run_log):
    pipe = make_pipeline(backend, embedder, retriever, run_log)
    result, retrieval, entry = pipe.explain("echogenic", "word")
    assert result.status == "ok" and result.card.term == "echogenic" and result.grounded
    assert entry.stages == ["validate", "retrieve", "build_prompt", "infer", "postprocess", "output"]
    assert entry.input_chars == 9 and entry.retrieval["top_score"] == 1.0 and entry.model["retries"] == 0
    assert set(entry.latency_ms) >= {"validate", "retrieve", "infer", "postprocess", "output", "total"}
    rows = read_log(run_log)
    assert len(rows) == 1 and rows[0]["run_id"] == entry.run_id and rows[0]["status"] == "ok"


def test_explain_sentence_end_to_end(backend, embedder, retriever, run_log):
    pipe = make_pipeline(backend, embedder, retriever, run_log)
    result, retrieval, entry = pipe.explain("The renal cortex is hypoechoic.", "sentence")
    assert result.status == "ok" and result.rewrite and {k.term for k in result.rewrite.key_terms} == {"hypoechoic", "renal cortex"}
    assert result.grounded


def test_inject_oversized_input_is_caught_by_validation(backend, embedder, retriever, run_log):
    pipe = make_pipeline(backend, embedder, retriever, run_log, inject="oversized_input")
    result, retrieval, entry = pipe.explain("echogenic", "word")
    assert result is None and entry.status == "rejected_input" and entry.stages == ["validate"]
    assert "limit" in entry.error and entry.injected_failure == "oversized_input"


def test_inject_malformed_json_is_caught_by_postprocess_after_retry(backend, embedder, retriever, run_log):
    pipe = make_pipeline(backend, embedder, retriever, run_log, inject="malformed_json")
    result, retrieval, entry = pipe.explain("echogenic", "word")
    assert result.status == "parse_failed" and result.fallback_text and result.low_confidence
    assert "infer_retry" in entry.stages and entry.model["retries"] == 1


def test_inject_model_timeout_is_handled(backend, embedder, retriever, run_log):
    pipe = make_pipeline(backend, embedder, retriever, run_log, inject="model_timeout")
    result, retrieval, entry = pipe.explain("echogenic", "word")
    assert result.status == "model_timeout" and "timeout" in entry.error and "postprocess" not in entry.stages


def test_inject_empty_retrieval_yields_not_found(backend, embedder, retriever, run_log):
    # canned backend ignores the empty context and still answers; postprocess must refuse to trust it
    pipe = make_pipeline(backend, embedder, retriever, run_log, inject="empty_retrieval")
    result, retrieval, entry = pipe.explain("echogenic", "word")
    assert retrieval.chunks == [] and result.status == "not_found" and result.low_confidence
    assert entry.retrieval["k"] == 0


def test_unknown_injection_rejected(backend, embedder, retriever, run_log):
    with pytest.raises(ValueError):
        make_pipeline(backend, embedder, retriever, run_log, inject="nope")


def test_quiz_loop_scores_and_adapts(backend, embedder, retriever, run_log):
    pipe = make_pipeline(backend, embedder, retriever, run_log)
    result, retrieval, _ = pipe.explain("echogenic", "word")
    answers = iter(["Because it reflects ultrasound strongly.", "no idea"])
    rounds, qlog = pipe.quiz(result, retrieval, lambda q: next(answers), n=2)
    assert len(rounds) == 2
    assert rounds[0].similarity > rounds[1].similarity
    assert rounds[0].verdict in ("related", "clarify") and rounds[1].verdict == "deeper"
    assert rounds[1].follow_up.startswith("Let's go deeper")
    assert qlog.command == "quiz" and qlog.output["rounds"][0]["similarity"] == rounds[0].similarity


def test_quiz_falls_back_when_model_output_unusable(embedder, retriever, run_log):
    pipe = make_pipeline(FakeBackend("not json at all"), embedder, retriever, run_log)
    from clinicallingo.contracts import ExplainResult
    res = ExplainResult(mode="word", status="ok", grounded=True, low_confidence=False)
    retrieval = retriever.retrieve(__import__("clinicallingo.contracts", fromlist=["ValidatedRequest"]).ValidatedRequest(text="echogenic", mode="word", char_count=9), allow_web=False)
    rounds, qlog = pipe.quiz(res, retrieval, lambda q: "bright", n=1)
    assert len(rounds) == 1 and qlog.model["question_source"] == "fallback"
