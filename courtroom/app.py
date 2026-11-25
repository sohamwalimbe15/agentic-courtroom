import json
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import llm, search
from .debate import QUOTE, judge, norm, qtext, render, speak, verify_quote
from .run import DEBATER, JUDGE

ROOT = Path(__file__).resolve().parent.parent
ROUNDS = 2
PER_IP = int(os.environ.get("DEMO_PER_IP_PER_HOUR", "5"))
MAX_TEXT = 20_000
hits: dict[str, deque] = defaultdict(deque)


@asynccontextmanager
async def lifespan(_):
    # global cap on what the demo can spend in this process; llm also enforces the lifetime cap
    llm.configure(float(os.environ.get("DEMO_BUDGET_USD", "2.0")), 4)
    yield


app = FastAPI(lifespan=lifespan)


class Req(BaseModel):
    topic: str
    text: str | None = None  # paste your own source text instead of searching


def rate_limit(ip: str):
    now, q = time.time(), hits[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= PER_IP:
        raise HTTPException(429, f"limit is {PER_IP} trials per hour per IP")
    q.append(now)


def segments(text: str, passage: str) -> list[dict]:
    pn, out, i = norm(passage), [], 0
    for m in QUOTE.finditer(text):
        out += [{"t": text[i : m.start()]}, {"q": qtext(m), "ok": verify_quote(qtext(m), pn)}]
        i = m.end()
    return out + [{"t": text[i:]}]


def sse(event: str, d: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(d)}\n\n"


async def stream(topic: str, text: str | None = None):
    yield sse("start", {"topic": topic, "debater": DEBATER, "judge": JUDGE})
    sd = {"A": "for", "B": "against", "correct_slot": "A", "consult_slot": "A"}
    turns, cost = [], 0.0
    try:
        yield sse("researching", {})
        if text:
            queries, sources = [], [{"title": "Pasted text", "url": None, "text": text}]
        else:
            queries, sources = await search.research(DEBATER, topic)
        yield sse("research", {
            "queries": queries,
            "sources": [{"title": s["title"], "url": s["url"], "snippet": s["text"][:240]}
                        for s in sources],
        })  # fmt: skip
        evidence = search.evidence_text(sources)
        q = {"title": "sources", "passage": evidence, "question": topic, "opinion": True}
        for rnd in range(1, ROUNDS + 1):
            for slot in "AB":
                yield sse("speaking", {"round": rnd, "slot": slot})
                t = await speak(DEBATER, q, sd, slot, turns, rnd)
                turns.append(t)
                cost += t["cost"]
                yield sse("turn", {"round": rnd, "slot": slot,
                                   "segments": segments(t["text"], evidence)})  # fmt: skip
        yield sse("deliberating", {})
        v = await judge(JUDGE, q, sd, "opinion", render(turns, evidence, ROUNDS))
        yield sse(
            "verdict", {"pA": v["p_correct"], "pB": 1 - v["p_correct"], "cost": cost + v["cost"]}
        )
    except llm.BudgetExceeded:
        yield sse("error", {"message": "the demo's spend cap is used up"})
    except Exception as e:  # keep the stream well-formed for the page
        yield sse("error", {"message": f"{type(e).__name__}: {e}"[:200]})


@app.get("/")
def index():
    return FileResponse(ROOT / "static/index.html")


@app.post("/api/trial")
async def trial(r: Req, request: Request):
    topic = " ".join(r.topic.split())
    if not 4 <= len(topic) <= 300 or len(r.text or "") > MAX_TEXT:
        raise HTTPException(400, "topic must be 4-300 characters, pasted text under 20000")
    rate_limit(request.client.host if request.client else "?")
    return StreamingResponse(stream(topic, r.text), media_type="text/event-stream")
