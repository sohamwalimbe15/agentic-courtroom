"""One async entry point to the API: disk cache, retries, bounded concurrency, hard spend cap."""

import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path

import anthropic
import httpx

HARD_CAP_USD = 15.0  # lifetime cap across every run, enforced from the on-disk ledger
CACHE_DIR = Path(os.environ.get("COURTROOM_CACHE", ".cache"))

# $ per million tokens (input, output); cache writes cost 1.25x input, reads 0.1x
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.0, 5.0),
}
# how to turn thinking off per model; a short call needs none
NO_THINKING = {
    "claude-sonnet-5-5": {"type": "between_tools"},
    "claude-opus-5-5": None,
    "claude-haiku-5-5": {"type": "disabled"},
}


OLLAMA = "ollama:"  # e.g. ollama:mistral runs locally, costs nothing
# free-tier OpenAI-compatible hosts: model "groq:llama-3.3-70b-versatile" -> (base url, key env)
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
}
DEFAULT_RPM = {"groq": 25, "gemini": 8, "openrouter": 15, "cerebras": 20}
_gate: dict = {}


def is_free(model: str) -> bool:
    return model.startswith(OLLAMA) or model.split(":")[0] in PROVIDERS


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))


class BudgetExceeded(RuntimeError):
    pass


_s = {"client": None, "sem": None, "limit": HARD_CAP_USD, "spent": 0.0, "inflight": 0.0, "n": 0}


def configure(budget_usd: float = HARD_CAP_USD, concurrency: int = 8, client=None):
    _s["client"] = client
    _s["sem"] = asyncio.Semaphore(concurrency)
    _s["limit"] = budget_usd
    _s["spent"] = _s["inflight"] = 0.0
    _s["n"] = 0
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def lifetime_spend() -> float:
    f = CACHE_DIR / "ledger.jsonl"
    if not f.exists():
        return 0.0
    return sum(json.loads(line)["cost"] for line in f.read_text().splitlines() if line)


def spent() -> float:
    return _s["spent"]


def cost_of(model: str, i: int, o: int, cw: int = 0, cr: int = 0) -> float:
    if is_free(model):
        return 0.0
    pin, pout = PRICES[model]
    return (i * pin + cw * pin * 1.25 + cr * pin * 0.1 + o * pout) / 1e6


def _key(model, system, parts, max_tokens) -> str:
    blob = json.dumps([model, system, parts, max_tokens], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _path(key: str) -> Path:
    return CACHE_DIR / "responses" / key[:2] / f"{key}.json"


def _reserve(model, system, parts, max_tokens) -> float:
    if is_free(model):
        return 0.0
    chars = len(system) + sum(map(len, parts))
    pin, pout = PRICES[model]
    return (chars / 3 * pin * 1.25 + max_tokens * pout) / 1e6  # pessimistic


async def call(model, parts, system="", max_tokens=400, cache_first=False) -> dict:
    """parts become text blocks of one user turn; cache_first marks parts[0] for prompt caching."""
    key = _key(model, system, parts, max_tokens)
    p = _path(key)
    if p.exists():
        return {**json.loads(p.read_text()), "cached": True}

    reserve = _reserve(model, system, parts, max_tokens)
    pending = _s["inflight"] + reserve
    if lifetime_spend() + pending > HARD_CAP_USD or _s["spent"] + pending > _s["limit"]:
        raise BudgetExceeded(
            f"run ${_s['spent']:.2f} of ${_s['limit']:.2f}, lifetime cap ${HARD_CAP_USD}"
        )
    _s["inflight"] += reserve
    try:
        async with _s["sem"]:
            r = await _request(model, parts, system, max_tokens, cache_first)
    finally:
        _s["inflight"] -= reserve
    _s["spent"] += r["cost"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r))
    with open(CACHE_DIR / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time(), "model": model, "cost": r["cost"]}) + "\n")
    return {**r, "cached": False}


async def _ollama_request(model, parts, system, max_tokens) -> dict:
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": "\n\n".join(parts)}
    ]
    body = {
        "model": model.removeprefix(OLLAMA),
        "messages": msgs,
        "stream": False,
        "options": {"num_ctx": OLLAMA_CTX, "num_predict": max_tokens, "temperature": 0},
    }
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=900) as c:
        r = await c.post(f"{OLLAMA_URL}/api/chat", json=body)
    r.raise_for_status()
    d = r.json()
    n_in = d.get("prompt_eval_count", 0)
    if n_in >= OLLAMA_CTX - 8:  # ollama truncates silently, which would corrupt results
        raise RuntimeError(
            f"prompt hit num_ctx={OLLAMA_CTX}; raise OLLAMA_NUM_CTX or use a longer-context model"
        )
    return {
        "text": d["message"]["content"],
        "stop": d.get("done_reason"),
        "in": n_in,
        "out": d.get("eval_count", 0),
        "cache_write": 0,
        "cache_read": 0,
        "cost": 0.0,
        "latency": time.perf_counter() - t0,
    }


async def _throttle(prov: str):
    g = _gate.setdefault(prov, {"lock": asyncio.Lock(), "next": 0.0})
    rpm = float(os.environ.get(f"{prov.upper()}_RPM", DEFAULT_RPM[prov]))
    async with g["lock"]:
        await asyncio.sleep(max(0.0, g["next"] - time.monotonic()))
        g["next"] = time.monotonic() + 60 / rpm


async def _oai_request(model, parts, system, max_tokens) -> dict:
    prov, name = model.split(":", 1)
    base, key_env = PROVIDERS[prov]
    key = os.environ.get(key_env)
    if not key:
        raise RuntimeError(f"set {key_env} for {model}")
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": "\n\n".join(parts)}
    ]
    body = {"model": name, "messages": msgs, "max_tokens": max_tokens, "temperature": 0}
    async with httpx.AsyncClient(timeout=300) as c:
        for attempt in range(8):
            await _throttle(prov)
            t0 = time.perf_counter()
            try:
                r = await c.post(
                    f"{base}/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {key}"},
                )
            except httpx.TransportError:
                r = None
            if r is not None and r.status_code < 400:
                break
            if r is not None and r.status_code not in (429, 500, 502, 503, 529):
                raise RuntimeError(f"{model}: {r.status_code} {r.text[:200]}")
            # free tiers 429 a lot (tokens per minute); honour Retry-After when given
            wait = float(r.headers.get("retry-after", 0)) if r is not None else 0
            await asyncio.sleep(max(wait, min(60, 2**attempt)) * (0.8 + 0.4 * random.random()))
        else:
            raise RuntimeError(f"{model}: retries exhausted")
    d = r.json()
    u = d.get("usage") or {}
    ch = d["choices"][0]
    return {
        "text": ch["message"].get("content") or "",
        "stop": ch.get("finish_reason"),
        "in": u.get("prompt_tokens", 0),
        "out": u.get("completion_tokens", 0),
        "cache_write": 0,
        "cache_read": 0,
        "cost": 0.0,
        "latency": time.perf_counter() - t0,
    }


async def _request(model, parts, system, max_tokens, cache_first) -> dict:
    if model.startswith(OLLAMA):
        return await _ollama_request(model, parts, system, max_tokens)
    if is_free(model):
        return await _oai_request(model, parts, system, max_tokens)
    client = _s["client"] or anthropic.AsyncAnthropic(max_retries=0)
    _s["client"] = client
    blocks = [{"type": "text", "text": t} for t in parts]
    if cache_first:
        blocks[0]["cache_control"] = {"type": "ephemeral"}
    kw = dict(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": blocks}])
    if system:
        kw["system"] = system
    if NO_THINKING.get(model):
        kw["thinking"] = NO_THINKING[model]

    for attempt in range(7):
        t0 = time.perf_counter()
        try:
            m = await client.messages.create(**kw)
            break
        except anthropic.BadRequestError as e:
            if "thinking" in str(e) and "thinking" in kw:  # my thinking-off guess was wrong
                NO_THINKING[model] = None
                del kw["thinking"]
                continue
            raise
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError):
            pass
        except anthropic.APIStatusError as e:
            if e.status_code < 500 and e.status_code != 529:
                raise
        if attempt == 6:
            raise RuntimeError("api retries exhausted")
        await asyncio.sleep(min(60, 2**attempt) * (0.5 + random.random()))
    u = m.usage
    cw = getattr(u, "cache_creation_input_tokens", 0) or 0
    cr = getattr(u, "cache_read_input_tokens", 0) or 0
    text = "".join(b.text for b in m.content if b.type == "text")
    return {
        "text": text,
        "stop": m.stop_reason,
        "in": u.input_tokens,
        "out": u.output_tokens,
        "cache_write": cw,
        "cache_read": cr,
        "cost": cost_of(model, u.input_tokens, u.output_tokens, cw, cr),
        "latency": time.perf_counter() - t0,
    }
