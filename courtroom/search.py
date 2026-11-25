import asyncio
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from . import llm

WIKI = "https://en.wikipedia.org/w/api.php"
# Wikimedia 403s anonymous agents; set WIKI_USER_AGENT with your contact if you deploy this
UA = os.environ.get("WIKI_USER_AGENT", "CourtroomDemo/0.1 (local research project)")


def parse_queries(text: str, topic: str, n: int = 2) -> list[str]:
    lines = [re.sub(r"^[\s\-*\d.)\"']+|[\"'\s]+$", "", ln) for ln in text.splitlines()]
    qs = [q for q in lines if 2 < len(q) < 80][:n]
    return qs or [topic[:80]]


async def _query(**extra) -> list[dict]:
    params = {
        "action": "query", "prop": "extracts|info", "inprop": "url", "explaintext": 1,
        "exchars": 1200, "exlimit": 20, "format": "json", "formatversion": 2, **extra,
    }  # fmt: skip
    req = urllib.request.Request(
        f"{WIKI}?{urllib.parse.urlencode(params)}", headers={"User-Agent": UA}
    )

    def fetch():  # urllib, because Wikimedia 403s httpx's fingerprint but not urllib's
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.load(r)

    try:
        pages = (await asyncio.to_thread(fetch)).get("query", {}).get("pages", [])
    except (OSError, ValueError):  # URLError and timeouts are OSErrors
        return []
    pages.sort(key=lambda p: p.get("index", 0))
    return [
        {"title": p["title"], "url": p.get("fullurl"), "text": p["extract"].strip()}
        for p in pages
        if p.get("extract")
    ]


async def wiki(query: str, n: int = 3) -> list[dict]:
    """The article titled `query` if there is one, then search hits to fill n slots."""
    title = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", query)  # "SocialMedia" -> "Social Media"
    exact = await _query(titles=title, redirects=1)
    hits = await _query(generator="search", gsrsearch=query, gsrlimit=n)
    out = exact[:1]
    for h in hits:
        if h["title"] not in {x["title"] for x in out} and len(out) < n:
            out.append(h)
    return out


async def research(model: str, topic: str, max_sources: int = 5) -> tuple[list[str], list[dict]]:
    ask = (
        f"Topic for a debate: {topic}\nName 2 Wikipedia article titles (1-4 words each, like "
        '"Telecommuting" or "Nuclear power") that would give useful facts for both sides. '
        "One per line. Output only the titles."
    )
    r = await llm.call(model, [ask], max_tokens=50)
    queries = parse_queries(r["text"], topic)
    sources, seen = [], set()
    for q in queries:
        for s in await wiki(q):
            if s["title"] not in seen and len(sources) < max_sources:
                seen.add(s["title"])
                sources.append(s)
    return queries, sources


def evidence_text(sources: list[dict]) -> str:
    if not sources:
        return "(No sources were found. Argue from general knowledge; nothing can be quoted.)"
    return "\n\n".join(
        f"[Source {i}: {s['title']}{' - ' + s['url'] if s.get('url') else ''}]\n{s['text']}"
        for i, s in enumerate(sources, 1)
    )
