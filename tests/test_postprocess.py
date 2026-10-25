import json

import pytest

from clinicallingo.contracts import Chunk, RetrievalResult, ValidatedRequest
from clinicallingo.postprocess import ParseFailure, parse_json_lenient, to_result
from clinicallingo.prompts import NOT_FOUND_MARKER


def chunk(term, text="def", score=0.9, index="medical_dictionary"):
    return Chunk(term=term, text=text, source="seed", score=score, index_name=index)


WORD = ValidatedRequest(text="echogenic", mode="word", char_count=9)
SENT = ValidatedRequest(text="The liver is hypoechoic.", mode="sentence", char_count=24)
RET = RetrievalResult(chunks=[chunk("echogenic"), chunk("hypoechoic")], coverage=1.0)


def test_lenient_parse_handles_fences_and_prose_wrapping():
    assert parse_json_lenient('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_lenient('Sure! {"a": 1} hope that helps') == {"a": 1}


@pytest.mark.parametrize("raw", ["", "   ", "no json here", "[1,2]", '{"a": }'])
def test_lenient_parse_failures(raw):
    with pytest.raises(ParseFailure):
        parse_json_lenient(raw)


def test_word_ok_and_grounded():
    raw = json.dumps({"term": "echogenic", "plain_definition": "bright", "clinical_definition": "reflects strongly",
                      "synonyms": [], "used_sources": ["[1]"]})
    r = to_result(WORD, RET, raw)
    assert r.status == "ok" and r.grounded and not r.low_confidence and r.sources == ["seed"]


def test_word_ungrounded_when_no_citation_or_mention():
    raw = json.dumps({"term": "x", "plain_definition": "something", "clinical_definition": "else",
                      "synonyms": [], "used_sources": []})
    r = to_result(WORD, RET, raw)
    assert r.status == "ok" and r.grounded is False and r.low_confidence is True


def test_not_found_marker():
    raw = json.dumps({"term": "zzq", "plain_definition": NOT_FOUND_MARKER, "clinical_definition": "UNVERIFIED: guess",
                      "synonyms": [], "used_sources": []})
    r = to_result(WORD, RET, raw)
    assert r.status == "not_found" and r.low_confidence


def test_empty_retrieval_forces_not_found_without_sources():
    raw = json.dumps({"term": "x", "plain_definition": "made up", "clinical_definition": "made up", "synonyms": [], "used_sources": []})
    r = to_result(WORD, RetrievalResult(), raw)
    assert r.status == "not_found"


def test_prose_is_parse_failed_with_fallback():
    r = to_result(WORD, RET, "Echogenic means bright on ultrasound.")
    assert r.status == "parse_failed" and r.fallback_text and r.low_confidence


def test_schema_violation_is_parse_failed():
    r = to_result(WORD, RET, json.dumps({"term": "x", "plain_definition": ["not", "a", "string"]}))
    assert r.status == "parse_failed" and "schema" in (r.message or "")


def test_sentence_rewrite_grounded_via_key_terms():
    raw = json.dumps({"rewritten": "The liver looks darker than normal.",
                      "key_terms": [{"term": "hypoechoic", "plain_meaning": "darker"}]})
    r = to_result(SENT, RET, raw)
    assert r.status == "ok" and r.rewrite.key_terms[0].term == "hypoechoic" and r.grounded


def test_sentence_malformed_key_term_items_are_dropped_not_fatal():
    raw = json.dumps({"rewritten": "The liver looks darker than normal.",
                      "key_terms": [{"term": "hypoechoic", "plain_meaning": "darker"},
                                    {"term": "cbd", "short plain meaning": ""},   # renamed field (seen live)
                                    {"term": "", "plain_meaning": "x"}, "junk", {"term": "y", "plain_meaning": None}]})
    r = to_result(SENT, RET, raw)
    assert r.status == "ok" and [k.term for k in r.rewrite.key_terms] == ["hypoechoic"]


def test_sentence_empty_rewrite_is_parse_failed():
    r = to_result(SENT, RET, json.dumps({"rewritten": "", "key_terms": []}))
    assert r.status == "parse_failed"
