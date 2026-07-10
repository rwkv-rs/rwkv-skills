from __future__ import annotations

"""Small JSON web-search proxy for agent-loop WebSearchExecutor.

This is an operational fallback for environments that do not have a paid
search API configured. It exposes a Serper-like POST endpoint accepting
``{"q": "..."}`` and returns compact JSON snippets. The benchmark runner still
uses its normal web_search/fetch_url tools and official verifier.
"""

import argparse
import html
import json
import os
import re
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from xml.etree import ElementTree


_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
    r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_BING_RESULT_RE = re.compile(
    r'<li[^>]+class="b_algo"[^>]*>.*?<h2[^>]*>\s*<a[^>]+href="(?P<url>[^"]+)"[^>]*>'
    r"(?P<title>.*?)</a>.*?(?:<p[^>]*>(?P<snippet>.*?)</p>)?",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}")
_CJK_QUERY_CHUNK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_QUERY_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "before",
    "body",
    "category",
    "com",
    "en",
    "for",
    "from",
    "gov",
    "has",
    "have",
    "http",
    "https",
    "how",
    "into",
    "list",
    "net",
    "official",
    "only",
    "org",
    "public",
    "site",
    "the",
    "their",
    "through",
    "time",
    "with",
    "website",
    "what",
    "when",
    "where",
    "which",
    "who",
    "www",
    "zh",
}
_LOW_VALUE_DOMAINS = (
    "baike.baidu.com",
    "dictionary.cambridge.org",
    "danci.gei6.com",
    "global.bing.com/dict",
    "iciba.com",
    "koolearn.com/dict",
    "regengbaike.com",
    "corp.dict.cn",
)


def _clean_html(value: str) -> str:
    text = _TAG_RE.sub(" ", value)
    text = html.unescape(text)
    return " ".join(text.split())


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in _QUERY_TOKEN_RE.findall(query):
        lowered = token.lower()
        if lowered in _QUERY_STOPWORDS:
            continue
        if lowered.isdigit() and len(lowered) != 4:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        terms.append(lowered)
    return terms


def _query_cjk_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for chunk in _CJK_QUERY_CHUNK_RE.findall(query):
        candidates: list[str]
        if len(chunk) <= 8:
            candidates = [chunk]
        else:
            candidates = [chunk[index : index + 4] for index in range(0, len(chunk) - 3, 4)]
            candidates.extend(chunk[index : index + 3] for index in range(0, len(chunk) - 2, 3))
        for candidate in candidates:
            if len(candidate) < 2 or candidate in seen:
                continue
            seen.add(candidate)
            terms.append(candidate)
            if len(terms) >= 40:
                return terms
    return terms


def _query_variants(query: str) -> list[str]:
    """Try the original query first, then shorter forms for brittle search UIs."""
    normalized = " ".join(query.split())
    if not normalized:
        return []
    variants = [normalized]
    if len(normalized) > 180:
        variants.append(normalized[:180].strip())

    compact_parts: list[str] = []
    compact_parts.extend(_query_terms(normalized)[:14])
    for chunk in _CJK_QUERY_CHUNK_RE.findall(normalized):
        compact_parts.append(chunk[:40])
        if len(compact_parts) >= 18:
            break
    if compact_parts:
        variants.append(" ".join(compact_parts)[:180].strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for value in variants:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _low_value_penalty(url: str) -> int:
    lowered = url.lower()
    return 2 if any(domain in lowered for domain in _LOW_VALUE_DOMAINS) else 0


def _min_relevance_score(query: str) -> int:
    term_count = len(_query_terms(query)) + len(_query_cjk_terms(query))
    if term_count <= 1:
        return 1
    if term_count <= 5:
        return 2
    return 2


def _result_relevance_score(query: str, *fields: str) -> int:
    terms = _query_terms(query)
    cjk_terms = _query_cjk_terms(query)
    if not terms and not cjk_terms:
        return 1
    haystack = " ".join(_clean_html(field).lower() for field in fields)
    haystack_terms = {token.lower() for token in _QUERY_TOKEN_RE.findall(haystack)}
    score = 0
    for term in terms:
        if term in haystack_terms:
            score += 1
            continue
        if term.endswith("s") and term[:-1] in haystack_terms:
            score += 1
    for term in cjk_terms:
        if term in haystack:
            score += 1
    return score


def _ranked_result(
    query: str,
    *,
    title: str,
    url: str,
    snippet: str,
) -> tuple[int, dict[str, str]] | None:
    if not url.startswith(("http://", "https://")):
        return None
    raw_score = _result_relevance_score(query, title, url, snippet)
    if raw_score < _min_relevance_score(query):
        return None
    rank_score = raw_score - _low_value_penalty(url)
    return (
        rank_score,
        {
            "title": _clean_html(title)[:240],
            "url": html.unescape(url)[:500],
            "snippet": _clean_html(snippet)[:800],
        },
    )


def _decode_duckduckgo_url(value: str) -> str:
    url = html.unescape(value)
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return query["uddg"][0]
    return url


def search_duckduckgo(query: str, *, max_results: int, timeout_s: float) -> dict[str, Any]:
    params = urllib.parse.urlencode({"q": query})
    url = f"https://duckduckgo.com/html/?{params}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8", errors="replace")
    results: list[dict[str, str]] = []
    for match in _RESULT_RE.finditer(body):
        results.append(
            {
                "title": _clean_html(match.group("title"))[:240],
                "url": _decode_duckduckgo_url(match.group("url"))[:500],
                "snippet": _clean_html(match.group("snippet"))[:800],
            }
        )
        if len(results) >= max_results:
            break
    return {"query": query, "provider": "duckduckgo_html", "results": results}


def search_bing_api(query: str, *, max_results: int, timeout_s: float) -> dict[str, Any]:
    api_key = os.environ.get("BingSearch_APIKEY") or os.environ.get("BING_SEARCH_APIKEY")
    if not api_key:
        raise RuntimeError("BingSearch_APIKEY is not set")
    base_url = os.environ.get("BING_SEARCH_URL") or "https://api.bing.microsoft.com/v7.0/search"
    params = urllib.parse.urlencode({"q": query, "count": max_results, "mkt": "en-US", "textFormat": "Raw"})
    request = urllib.request.Request(
        f"{base_url}?{params}",
        headers={"Ocp-Apim-Subscription-Key": api_key, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    results: list[dict[str, str]] = []
    for item in ((payload.get("webPages") or {}).get("value") or []):
        if not isinstance(item, dict):
            continue
        result_url = str(item.get("url") or "")
        if not result_url.startswith(("http://", "https://")):
            continue
        results.append(
            {
                "title": _clean_html(str(item.get("name") or ""))[:240],
                "url": result_url[:500],
                "snippet": _clean_html(str(item.get("snippet") or ""))[:800],
            }
        )
        if len(results) >= max_results:
            break
    return {"query": query, "provider": "bing_api", "results": results}


def search_bing_rss(query: str, *, max_results: int, timeout_s: float) -> dict[str, Any]:
    params = urllib.parse.urlencode({"q": query, "format": "rss", "mkt": "en-US", "setlang": "en-US"})
    url = f"https://www.bing.com/search?{params}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read()
    root = ElementTree.fromstring(body)
    candidates: list[tuple[int, int, dict[str, str]]] = []
    for item in root.findall("./channel/item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        description = item.findtext("description") or ""
        ranked = _ranked_result(query, title=title, url=link, snippet=description)
        if ranked is None:
            continue
        candidates.append((ranked[0], len(candidates), ranked[1]))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    results = [item[2] for item in candidates[:max_results]]
    return {"query": query, "provider": "bing_rss", "results": results}


def search_bing(query: str, *, max_results: int, timeout_s: float) -> dict[str, Any]:
    params = urllib.parse.urlencode({"q": query})
    url = f"https://www.bing.com/search?{params}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8", errors="replace")
    candidates: list[tuple[int, int, dict[str, str]]] = []
    seen: set[str] = set()
    for match in _BING_RESULT_RE.finditer(body):
        result_url = html.unescape(match.group("url"))
        if result_url in seen:
            continue
        ranked = _ranked_result(
            query,
            title=match.group("title"),
            url=result_url,
            snippet=match.group("snippet") or "",
        )
        if ranked is None:
            continue
        seen.add(result_url)
        candidates.append((ranked[0], len(candidates), ranked[1]))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    results = [item[2] for item in candidates[:max_results]]
    return {"query": query, "provider": "bing_html", "results": results}


def search_web(query: str, *, max_results: int, timeout_s: float) -> dict[str, Any]:
    errors: list[str] = []
    providers = [search_bing_api] if (os.environ.get("BingSearch_APIKEY") or os.environ.get("BING_SEARCH_APIKEY")) else []
    providers.extend([search_bing_rss, search_duckduckgo, search_bing])
    for query_variant in _query_variants(query):
        for provider in providers:
            try:
                result = provider(query_variant, max_results=max_results, timeout_s=timeout_s)
            except Exception as exc:  # noqa: BLE001 - return provider failures as observations.
                errors.append(f"{provider.__name__}({query_variant[:80]!r}): {type(exc).__name__}: {exc}")
                continue
            if result.get("results"):
                result["query"] = query
                if query_variant != query:
                    result["executed_query"] = query_variant
                if errors:
                    result["fallback_errors"] = errors
                return result
            errors.append(f"{provider.__name__}({query_variant[:80]!r}): empty results")
    return {"query": query, "provider": "fallback_chain", "results": [], "error": "; ".join(errors)}


def _is_query_relevant(query: str, *fields: str) -> bool:
    return _result_relevance_score(query, *fields) >= _min_relevance_score(query)


class SearchHandler(BaseHTTPRequestHandler):
    max_results = 8
    timeout_s = 12.0

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path in {"/health", "/healthz"}:
            self._write_json({"ok": True})
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            payload = {}
        query = str(payload.get("q") or payload.get("query") or "").strip()
        if not query:
            self._write_json({"query": "", "provider": "duckduckgo_html", "results": [], "error": "missing query"})
            return
        result = search_web(query, max_results=self.max_results, timeout_s=self.timeout_s)
        self._write_json(result)

    def log_message(self, fmt: str, *args: object) -> None:
        return None

    def _write_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> int:
    parser = argparse.ArgumentParser(description="Local JSON web-search proxy for RWKV agent-loop runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18901)
    parser.add_argument("--max-results", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=12.0)
    args = parser.parse_args()
    SearchHandler.max_results = max(1, int(args.max_results))
    SearchHandler.timeout_s = max(1.0, float(args.timeout_s))
    server = ThreadingHTTPServer((str(args.host), int(args.port)), SearchHandler)
    print(f"web_search_proxy listening on http://{args.host}:{args.port}/search", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
