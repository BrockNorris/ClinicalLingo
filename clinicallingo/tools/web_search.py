"""Tool call: allow-listed medical web search.

lookup(term) -> WebResult.

Strategy (observed Oct 2026 while building the prototype):
  1. Direct probe of Radiopaedia's article URL for the term (`/articles/<slug>`);
     it exists for most imaging terms and avoids a search engine entirely.
  2. Search via `ddgs` with explicit backends ("duckduckgo", then "yahoo"); the
     "auto" backend returned nothing for rare terms. Queries are restricted with
     `site:` to the preferred domains and results are filtered against the
     allow-list, and PDFs / search-result pages are skipped.
  3. Fetch each page, extract paragraph text, split into ~WEB_CHUNK_CHARS passages.
     The Retriever embeds them, keeps the relevant ones, and stores them in web_cache.

Contract: never raises. Network, parsing, and rate-limit failures are reported in
WebResult.error with whatever passages were obtained. Only URLs whose host ends
with an allowed domain are ever fetched.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from ..config import (ALLOWED_DOMAINS, WEB_CHUNK_CHARS, WEB_FETCH_TIMEOUT_S,
                      WEB_MAX_RESULTS, WEB_SEARCH_TIMEOUT_S, WEB_TOOL_BUDGET_S)

log = logging.getLogger("clinicallingo.webtool")

# Radiopaedia answers 406 to user agents containing a URL; a plain product token works.
_UA = "ClinicalLingo/0.1 (student prototype)"
_PREFERRED = ("radiopaedia.org", "medlineplus.gov", "webmd.com")
_SEARCH_BACKENDS = ("duckduckgo", "yahoo")  # ddgs 9.16 backends: brave, duckduckgo, grokipedia, mojeek, startpage, wikipedia, yahoo
_SKIP_URL = re.compile(r"(\.pdf($|\?)|/search\?|/search/|\?q=)", re.IGNORECASE)


@dataclass
class WebChunk:
    text: str
    url: str
    domain: str
    title: str = ""


@dataclass
class WebResult:
    query: str
    urls: list[str] = field(default_factory=list)
    chunks: list[WebChunk] = field(default_factory=list)
    error: str | None = None


def _domain_of(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return ""
    return host.lower()


def is_allowed(url: str, allowed=ALLOWED_DOMAINS) -> bool:
    host = _domain_of(url)
    return bool(host) and any(host == d or host.endswith("." + d) for d in allowed)


def slugify(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")


def chunk_text(text: str, size: int = WEB_CHUNK_CHARS) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if not s:
            continue
        if len(cur) + len(s) + 1 > size and cur:
            chunks.append(cur.strip())
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur.strip())
    return [c for c in chunks if len(c) >= 80]


def extract_paragraphs(html: str) -> tuple[str, str]:
    """Return (title, text) using BeautifulSoup; drops nav/script/style/footer."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()
    paras = []
    for p in soup.find_all(["p", "li"]):
        t = " ".join(p.get_text(" ", strip=True).split())
        if len(t) >= 60:
            paras.append(t)
    return title, "\n".join(paras)


class WebSearchTool:
    def __init__(self, allowed=ALLOWED_DOMAINS, max_results: int = WEB_MAX_RESULTS,
                 timeout_s: float = WEB_FETCH_TIMEOUT_S, searcher=None, fetcher=None,
                 direct_probe: bool = True, budget_s: float = WEB_TOOL_BUDGET_S):
        self.allowed = tuple(allowed)
        self.max_results = max_results
        self.timeout_s = timeout_s
        self.budget_s = budget_s
        self._searcher = searcher or self._ddg_search  # injectable for tests
        self._fetcher = fetcher or self._http_fetch
        self.direct_probe = direct_probe
        self._html_cache: dict[str, str] = {}

    # -- search ---------------------------------------------------------------------
    def _ddg_search(self, query: str, n: int) -> list[dict]:
        from ddgs import DDGS
        last_err: Exception | None = None
        for backend in _SEARCH_BACKENDS:
            try:
                with DDGS(timeout=WEB_SEARCH_TIMEOUT_S) as d:
                    res = list(d.text(query, max_results=n, backend=backend) or [])
                if res:
                    return res
            except Exception as e:  # noqa: BLE001 - ddgs raises several types incl. rate limits
                last_err = e
        if last_err:
            raise last_err
        return []

    def _usable(self, url: str, seen: list[str]) -> bool:
        return bool(url) and is_allowed(url, self.allowed) and not _SKIP_URL.search(url) and url not in seen

    def search(self, term: str) -> tuple[list[str], str | None]:
        urls: list[str] = []
        errors: list[str] = []
        if self.direct_probe:
            probe = f"https://radiopaedia.org/articles/{slugify(term)}"
            try:
                self._html_cache[probe] = self._fetcher(probe)
                urls.append(probe)
                log.info("web tool: direct article probe hit %s", probe)
            except Exception as e:  # noqa: BLE001
                log.info("web tool: direct probe miss (%s)", type(e).__name__)
        for dom in _PREFERRED:
            if len(urls) >= self.max_results:
                break
            q = f"{term} site:{dom}"
            try:
                for r in self._searcher(q, 5):
                    u = r.get("href") or r.get("url") or ""
                    if self._usable(u, urls):
                        urls.append(u)
            except Exception as e:  # noqa: BLE001
                errors.append(f"search '{q}' failed: {type(e).__name__}: {e}")
        if not urls:
            q = f"{term} medical definition"
            try:
                for r in self._searcher(q, 10):
                    u = r.get("href") or r.get("url") or ""
                    if self._usable(u, urls):
                        urls.append(u)
            except Exception as e:  # noqa: BLE001
                errors.append(f"search '{q}' failed: {type(e).__name__}: {e}")
        return urls[: self.max_results], ("; ".join(errors) or None)

    # -- fetch -----------------------------------------------------------------------
    def _http_fetch(self, url: str) -> str:
        r = requests.get(url, headers={"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml"},
                         timeout=self.timeout_s)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "html" not in ctype and "text" not in ctype:
            raise ValueError(f"unsupported content-type {ctype!r}")
        return r.text

    def fetch_chunks(self, url: str) -> list[WebChunk]:
        if not is_allowed(url, self.allowed):  # defence in depth: never fetch outside the allow-list
            raise PermissionError(f"refusing to fetch non-allow-listed URL {url}")
        html = self._html_cache.pop(url, None) or self._fetcher(url)
        title, text = extract_paragraphs(html)
        dom = _domain_of(url)
        return [WebChunk(text=c, url=url, domain=dom, title=title) for c in chunk_text(text)]

    # -- tool entry point --------------------------------------------------------------
    def lookup(self, term: str) -> WebResult:
        """Runs _lookup under a hard wall-clock budget. An unbounded hang inside the search
        library (observed once: ~2 hours) must never stall the pipeline; on timeout the
        worker thread is abandoned and the tool reports an error with no passages."""
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="webtool")
        fut = pool.submit(self._lookup, term)
        try:
            return fut.result(timeout=self.budget_s)
        except FutureTimeout:
            log.warning("web tool: exceeded %.0fs budget for %r; abandoning", self.budget_s, term)
            return WebResult(query=term, error=f"web tool exceeded {self.budget_s:.0f}s budget")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _lookup(self, term: str) -> WebResult:
        res = WebResult(query=term)
        errors: list[str] = []
        try:
            urls, err = self.search(term)
        except Exception as e:  # pragma: no cover - belt and braces
            return WebResult(query=term, error=f"search failed: {e}")
        if err:
            errors.append(err)
        res.urls = urls
        if not urls:
            res.error = "; ".join(errors) if errors else "no allow-listed results"
            return res
        for u in urls:
            try:
                chunks = self.fetch_chunks(u)
                res.chunks.extend(chunks[:12])
                log.info("web tool: fetched %s (%d passages)", u, len(chunks))
            except Exception as e:  # noqa: BLE001
                errors.append(f"fetch {u} failed: {type(e).__name__}: {e}")
                log.warning("web tool: fetch %s failed: %s", u, e)
        res.error = "; ".join(errors) if errors else None
        return res
