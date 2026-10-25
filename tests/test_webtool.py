import pytest

from clinicallingo.tools.web_search import WebSearchTool, chunk_text, is_allowed

HTML = """<html><head><title>Echogenic - Test</title></head><body>
<nav><p>Navigation links that are long enough to look like a paragraph but should be dropped entirely.</p></nav>
<p>Echogenic describes tissue that reflects ultrasound strongly and therefore appears bright on the image. It is the opposite of anechoic.</p>
<p>short</p>
<script>var x = "script content that must never appear in the extracted text at all";</script>
<p>Bone, stones, and fat are typical echogenic structures. They often cast an acoustic shadow behind them because little sound passes through.</p>
</body></html>"""


@pytest.mark.parametrize("url,ok", [
    ("https://medlineplus.gov/ency/article/1.htm", True),
    ("https://www.webmd.com/a-to-z-guides/x", True),
    ("https://radiopaedia.org/articles/echogenic", True),
    ("https://evil.com/medlineplus.gov/", False),
    ("https://medlineplus.gov.evil.com/", False),
    ("ftp://", False),
    ("", False),
])
def test_allowlist(url, ok):
    assert is_allowed(url) is ok


def test_chunk_text_splits_on_sentences_and_drops_short():
    text = " ".join(f"Sentence number {i} is here and it is reasonably long for a test." for i in range(20))
    chunks = chunk_text(text, size=200)
    assert all(len(c) <= 260 for c in chunks) and len(chunks) >= 5
    assert chunk_text("tiny.") == []


def test_lookup_uses_injected_search_and_fetch_and_filters_domains():
    seen = []

    def searcher(q, n):
        return [{"href": "https://medlineplus.gov/a"}, {"href": "https://notallowed.example/b"}]

    def fetcher(url):
        seen.append(url)
        return HTML

    tool = WebSearchTool(searcher=searcher, fetcher=fetcher, direct_probe=False)
    res = tool.lookup("echogenic")
    assert res.error is None and res.urls == ["https://medlineplus.gov/a"]
    assert seen == ["https://medlineplus.gov/a"]
    assert res.chunks and all(c.domain == "medlineplus.gov" and c.title == "Echogenic - Test" for c in res.chunks)
    joined = " ".join(c.text for c in res.chunks)
    assert "script content" not in joined and "Navigation links" not in joined and "Bone, stones" in joined


def test_direct_probe_is_tried_first_and_pdfs_and_search_pages_skipped():
    fetched = []

    def fetcher(url):
        fetched.append(url)
        if url.endswith("/does-not-exist"):
            raise RuntimeError("404")
        return HTML

    def searcher(q, n):
        return [{"href": "https://radiopaedia.org/search?q=x"}, {"href": "https://radiopaedia.org/a.pdf"},
                {"href": "https://medlineplus.gov/ency/1.htm"}]

    tool = WebSearchTool(searcher=searcher, fetcher=fetcher)
    res = tool.lookup("acoustic shadowing")
    assert res.urls[0] == "https://radiopaedia.org/articles/acoustic-shadowing"
    assert "https://medlineplus.gov/ency/1.htm" in res.urls
    assert not any(".pdf" in u or "search?" in u for u in res.urls)
    assert fetched.count("https://radiopaedia.org/articles/acoustic-shadowing") == 1  # probe HTML reused, not re-fetched

    res2 = WebSearchTool(searcher=lambda q, n: [], fetcher=fetcher).lookup("does not exist")
    assert res2.urls == [] and res2.error == "no allow-listed results"


def test_lookup_never_raises_on_search_or_fetch_failure():
    def bad_search(q, n):
        raise RuntimeError("rate limited")

    res = WebSearchTool(searcher=bad_search, fetcher=lambda u: HTML, direct_probe=False).lookup("echogenic")
    assert res.chunks == [] and "rate limited" in res.error

    def bad_fetch(u):
        raise TimeoutError("slow")

    res = WebSearchTool(searcher=lambda q, n: [{"href": "https://medlineplus.gov/a"}], fetcher=bad_fetch, direct_probe=False).lookup("echogenic")
    assert res.chunks == [] and "TimeoutError" in res.error


def test_lookup_enforces_wall_clock_budget():
    import time

    def hanging_search(q, n):
        time.sleep(5)
        return []

    t0 = time.perf_counter()
    res = WebSearchTool(searcher=hanging_search, fetcher=lambda u: HTML, direct_probe=False, budget_s=0.5).lookup("x")
    assert time.perf_counter() - t0 < 3
    assert res.chunks == [] and "budget" in res.error


def test_fetch_refuses_non_allowlisted_url_even_if_called_directly():
    tool = WebSearchTool(fetcher=lambda u: HTML)
    with pytest.raises(PermissionError):
        tool.fetch_chunks("https://example.com/page")
