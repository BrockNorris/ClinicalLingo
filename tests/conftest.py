"""Shared offline fixtures: no Ollama, no network. HashEmbedder + FakeBackend + tiny dictionaries."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from clinicallingo.contracts import Prompt
from clinicallingo.inference.backend import FakeBackend
from clinicallingo.retrieval.embedder import HashEmbedder
from clinicallingo.retrieval.index import VectorIndex
from clinicallingo.retrieval.retriever import Retriever

MEDICAL = [
    {"term": "echogenic", "definition": "Reflects ultrasound strongly and appears bright on the image.",
     "synonyms": ["bright", "reflective"], "source": "seed:test"},
    {"term": "hypoechoic", "definition": "Darker than the surrounding tissue because it produces fewer echoes.",
     "synonyms": [], "source": "seed:test"},
    {"term": "anechoic", "definition": "Produces no internal echoes and appears black, like simple fluid in a cyst.",
     "synonyms": ["echo-free"], "source": "seed:test"},
    {"term": "renal cortex", "definition": "The outer layer of the kidney containing the filtering units.",
     "synonyms": ["cortex"], "source": "seed:test"},
    {"term": "acoustic shadowing", "definition": "A dark band behind a strongly attenuating structure such as a stone.",
     "synonyms": ["shadowing"], "source": "seed:test"},
]
GENERAL = [
    {"term": "adjacent", "definition": "Next to or touching something else.", "synonyms": [], "source": "seed:test"},
    {"term": "diffuse", "definition": "Spread throughout rather than confined to one spot.", "synonyms": [], "source": "seed:test"},
]


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def indexes(embedder):
    med = VectorIndex("medical_dictionary").build(MEDICAL, embedder)
    gen = VectorIndex("general_dictionary").build(GENERAL, embedder)
    return med, gen


@pytest.fixture
def retriever(embedder, indexes, tmp_path):
    med, gen = indexes
    return Retriever(embedder, med, gen, VectorIndex("web_cache"), web_tool=None, web_cache_dir=tmp_path / "web_cache")


def canned_responder(prompt: Prompt) -> str:
    """Produces schema-valid JSON for each prompt type, grounded in the test dictionary."""
    u = prompt.user
    if "Term to explain" in u:
        return json.dumps({"term": "echogenic", "plain_definition": "It shows up bright because it bounces sound back strongly.",
                           "clinical_definition": "Echogenic tissue reflects ultrasound strongly and appears bright.",
                           "synonyms": ["bright", "reflective"], "used_sources": ["[1]"]})
    if "Textbook" in u:
        return json.dumps({"rewritten": "The kidney's outer layer looks darker (hypoechoic) than the liver next to it.",
                           "key_terms": [{"term": "hypoechoic", "plain_meaning": "darker than nearby tissue"},
                                         {"term": "renal cortex", "plain_meaning": "outer layer of the kidney"}]})
    if "comprehension questions" in u:
        return json.dumps({"questions": [
            {"question": "Why does an echogenic structure look bright?", "reference_answer": "Because it reflects ultrasound strongly.", "based_on_term": "echogenic"},
            {"question": "What does hypoechoic mean?", "reference_answer": "Darker than surrounding tissue because it produces fewer echoes.", "based_on_term": "hypoechoic"}]})
    return "Here is a simpler explanation: bright things bounce sound back."


@pytest.fixture
def backend():
    return FakeBackend(canned_responder)


@pytest.fixture
def run_log(tmp_path) -> Path:
    return tmp_path / "runs.jsonl"
