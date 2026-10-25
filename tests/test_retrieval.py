import numpy as np
import pytest

from clinicallingo.config import MIN_SCORE, TOP_K_BY_MODE
from clinicallingo.contracts import ValidatedRequest
from clinicallingo.retrieval.index import VectorIndex
from clinicallingo.retrieval.retriever import Retriever
from clinicallingo.tools.web_search import WebChunk, WebResult


def req(text, mode="word"):
    return ValidatedRequest(text=text, mode=mode, char_count=len(text))


def test_index_roundtrip(indexes, embedder, tmp_path):
    med, _ = indexes
    med.save(tmp_path / "med")
    loaded = VectorIndex.load(tmp_path / "med")
    assert len(loaded) == len(med) and loaded.meta["embedder"] == embedder.name
    assert np.allclose(loaded.vectors, med.vectors)


def test_corrupt_index_detected(indexes, tmp_path):
    med, _ = indexes
    med.save(tmp_path / "med")
    np.save(tmp_path / "med" / "vectors.npy", med.vectors[:-1])
    with pytest.raises(ValueError, match="corrupt"):
        VectorIndex.load(tmp_path / "med")


def test_dim_mismatch_detected(indexes):
    med, _ = indexes
    with pytest.raises(ValueError, match="dim"):
        med.query(np.zeros(7, dtype=np.float32))


def test_exact_term_is_top_hit_with_score_one(retriever):
    res = retriever.retrieve(req("echogenic"), allow_web=False)
    assert res.chunks and res.chunks[0].term == "echogenic" and res.chunks[0].score == 1.0
    assert res.coverage == 1.0 and "echogenic" in res.lexical_hits
    assert all(c.score >= MIN_SCORE for c in res.chunks)
    assert len(res.chunks) <= TOP_K_BY_MODE["word"]


def test_synonym_lexical_hit(retriever):
    res = retriever.retrieve(req("The mass is echo-free.", "sentence"), allow_web=False)
    assert "anechoic" in [c.term for c in res.chunks]


def test_sentence_mode_finds_all_terms_in_text(retriever):
    res = retriever.retrieve(req("The renal cortex appears hypoechoic and adjacent to the liver.", "sentence"), allow_web=False)
    terms = {c.term for c in res.chunks}
    assert {"renal cortex", "hypoechoic", "adjacent"} <= terms
    med_first = [c.index_name for c in res.chunks]
    assert med_first.index("medical_dictionary") < med_first.index("general_dictionary")


def test_unknown_word_without_web_tool(retriever):
    res = retriever.retrieve(req("zzqxv"), allow_web=True)
    assert res.coverage < 0.55 and res.used_web is False
    assert res.web_error == "web tool not configured"


class StubWebTool:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = 0

    def lookup(self, term):
        self.calls += 1
        return WebResult(query=term, urls=["https://medlineplus.gov/x"], chunks=self.chunks)


def test_web_tool_called_filtered_cached_and_reused(embedder, indexes, tmp_path):
    med, gen = indexes
    relevant = WebChunk(text="Pneumobilia pneumobilia pneumobilia means gas in the biliary tree. pneumobilia", url="https://medlineplus.gov/x", domain="medlineplus.gov")
    junk = WebChunk(text="Subscribe to our newsletter for the latest updates and offers from the site editors.", url="https://medlineplus.gov/x", domain="medlineplus.gov")
    tool = StubWebTool([relevant, junk])
    r = Retriever(embedder, med, gen, VectorIndex("web_cache"), web_tool=tool, web_cache_dir=tmp_path / "wc")

    res = r.retrieve(req("pneumobilia"), allow_web=True)
    assert res.used_web is True and tool.calls == 1
    web = [c for c in res.chunks if c.index_name == "web_cache"]
    assert web and "newsletter" not in web[0].text  # relevance double-check dropped the junk passage
    assert (tmp_path / "wc" / "entries.json").exists()

    res2 = r.retrieve(req("pneumobilia"), allow_web=True)
    assert tool.calls == 1 and res2.web_cache_hit is True and res2.used_web is False


def test_web_not_called_in_sentence_mode_unless_forced(embedder, indexes, tmp_path):
    med, gen = indexes
    tool = StubWebTool([])
    r = Retriever(embedder, med, gen, VectorIndex("web_cache"), web_tool=tool, web_cache_dir=tmp_path / "wc")
    r.retrieve(req("zzqxv is seen here.", "sentence"), allow_web=True)
    assert tool.calls == 0
    res = r.retrieve(req("zzqxv is seen here.", "sentence"), allow_web=True, force_web=True)
    assert tool.calls == 1 and res.used_web and res.web_error == "web tool returned no passages"
