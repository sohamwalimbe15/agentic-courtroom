import numpy as np
import pytest

from courtroom import llm
from courtroom.debate import mark_quotes, norm, parse_probs, sides, verify_quote
from courtroom.run import process
from courtroom.stats import paired
from tests.conftest import PASSAGE


def test_quote_verification():
    pn = norm(PASSAGE)
    assert verify_quote("never spoke of his  brother", pn)
    assert verify_quote("The old lighthouse ... lit every night", pn)
    assert not verify_quote("he sold the boat", pn)
    text, n, bad = mark_quotes(
        "a <quote>he sold the boat</quote> b <quote>kept the lamp lit</quote>", PASSAGE
    )
    assert (n, bad) == (2, 1) and "UNVERIFIED" in text and "[VERIFIED QUOTE" in text


@pytest.mark.parametrize(
    "raw,expect",
    [
        ('{"A": 0.7, "B": 0.3}', (0.7, 0.3, True)),
        ('Sure! ```json\n{"A": 7, "B": 3}\n```', (0.7, 0.3, True)),
        ("A: 0.9, B: 0.1", (0.9, 0.1, True)),
        ("I cannot tell.", (0.5, 0.5, False)),
        ('{"A": 0, "B": 0}', (0.5, 0.5, False)),
    ],
)
def test_parse_probs(raw, expect):
    assert parse_probs(raw) == pytest.approx(expect)


async def test_cache_and_budget(fake):
    await llm.call("claude-haiku-5-5", ["hi"], max_tokens=60)
    r = await llm.call("claude-haiku-5-5", ["hi"], max_tokens=60)
    assert r["cached"] and fake.calls == 1
    llm.configure(1e-6, 2, client=fake)
    with pytest.raises(llm.BudgetExceeded):
        await llm.call("claude-haiku-5-5", ["something new " * 50], max_tokens=60)
    assert fake.calls == 1


async def test_lifetime_cap(fake, monkeypatch):
    await llm.call("claude-haiku-5-5", ["hi"], max_tokens=60)
    monkeypatch.setattr(llm, "HARD_CAP_USD", 1e-9)
    llm.configure(100, 2, client=fake)
    with pytest.raises(llm.BudgetExceeded):
        await llm.call("claude-haiku-5-5", ["other"], max_tokens=60)


async def test_process_and_swap(fake, question, monkeypatch):
    monkeypatch.setattr("courtroom.run.ALT_JUDGES", [])
    rec = await process(question, seed=0, rounds=2)
    assert len(rec["debate"]) == 4 and len(rec["consult"]) == 2
    assert rec["quotes"] == 8 and rec["fabricated"] == 4
    j = rec["judgments"]
    fwd = j["claude-haiku-5-5|debate|r2|ver|fwd"]["p_correct"]
    swp = j["claude-haiku-5-5|debate|r2|ver|swap"]["p_correct"]
    # the fake always favours slot A; swapping must flip which answer that is
    assert fwd == pytest.approx(1 - swp)
    assert sides("q1", "x", "y", 0) == sides("q1", "x", "y", 0)


def test_mcnemar():
    x, y = np.array([0, 0, 0, 0, 1, 1.0]), np.array([1, 1, 1, 1, 1, 1.0])
    r = paired(x, y)
    assert r["only_y"] == 4 and r["only_x"] == 0 and r["mcnemar_p"] == pytest.approx(0.125)


def test_rate_limit():
    from fastapi import HTTPException

    from courtroom import app

    for _ in range(app.PER_IP):
        app.rate_limit("1.2.3.4")
    with pytest.raises(HTTPException):
        app.rate_limit("1.2.3.4")
    app.rate_limit("5.6.7.8")


async def test_stream_events(fake, monkeypatch):
    from courtroom import app, search

    async def fake_wiki(query, n=3):
        return [
            {
                "title": "Lighthouse",
                "url": "http://x",
                "text": "The keeper never spoke of his brother.",
            }
        ]

    monkeypatch.setattr(search, "wiki", fake_wiki)
    evs = [e async for e in app.stream("Lighthouse keepers were lonely")]
    kinds = [e.split("\n")[0].removeprefix("event: ") for e in evs]
    assert kinds[:3] == ["start", "researching", "research"]
    assert kinds.count("turn") == 4 and kinds[-2:] == ["deliberating", "verdict"]
    assert '"ok": false' in "".join(evs)  # the fake's invented quote is flagged


def test_parse_queries():
    from courtroom.search import parse_queries

    assert parse_queries('1. "remote work"\n- office culture\n3. extra', "t") == [
        "remote work",
        "office culture",
    ]
    assert parse_queries("", "my topic") == ["my topic"]


async def test_ollama_routing_is_free(fake, monkeypatch):
    seen = []

    async def fake_ollama(model, parts, system, max_tokens):
        seen.append(model)
        return {
            "text": '{"A": 0.6, "B": 0.4}',
            "stop": "stop",
            "in": 10,
            "out": 5,
            "cache_write": 0,
            "cache_read": 0,
            "cost": 0.0,
            "latency": 0.1,
        }

    monkeypatch.setattr(llm, "_ollama_request", fake_ollama)
    llm.configure(0.0, 1, client=fake)  # a zero budget still allows free local calls
    r = await llm.call("ollama:mistral", ["hi"], max_tokens=60)
    assert seen == ["ollama:mistral"] and r["cost"] == 0 and fake.calls == 0


async def test_oai_backend_retries_and_is_free(fake, monkeypatch):
    import httpx

    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("GROQ_RPM", "100000")
    monkeypatch.setattr(llm.asyncio, "sleep", lambda s: _noop())
    n = {"c": 0}

    def handler(req):
        n["c"] += 1
        if n["c"] == 1:
            return httpx.Response(429, headers={"retry-after": "1"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            },
        )

    real = httpx.AsyncClient
    monkeypatch.setattr(
        llm.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    llm.configure(0.0, 1, client=fake)
    r = await llm.call("groq:llama-3.1-8b-instant", ["x"], max_tokens=10)
    assert r["text"] == "hi" and r["cost"] == 0 and n["c"] == 2 and fake.calls == 0


async def _noop():
    return None


def test_plain_quotes_are_checked_too():
    t = 'He said "never spoke of his brother" and "the keeper sold the boat yesterday" ok "hi"'
    out, n, bad = mark_quotes(t, PASSAGE)
    assert (n, bad) == (2, 1)  # "hi" is too short to count


async def test_judge_prompt_has_single_braces(fake, question):
    seen = []
    real = fake.create

    async def spy(**kw):
        seen.append(kw["messages"][0]["content"][0]["text"])
        return await real(**kw)

    fake.messages.create = spy
    from courtroom.debate import judge

    await judge("claude-haiku-5-5", question, sides("q1", "a", "b", 0), "alone")
    assert '{"A": <p_A>' in seen[0] and "{{" not in seen[0]  # no numeric example to copy


def test_echoed_header_is_stripped():
    from courtroom.debate import HEADER

    assert HEADER.sub("", "Round 2, Debater B:\nI disagree.") == "I disagree."
    assert HEADER.sub("", "Round 1, Consultant, arguing for Answer A: Yes.") == "Yes."


async def test_speak_strips_echoed_header(fake, question, monkeypatch):
    from courtroom import debate

    async def echo(*a, **k):
        return {"text": "Round 1, Debater A:\nI object.", "cost": 0, "latency": 0}

    monkeypatch.setattr(debate.llm, "call", echo)
    t = await debate.speak("m", question, sides("q1", "a", "b", 0), "A", [], 1)
    assert t["text"] == "I object."
