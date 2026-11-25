import argparse
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

from . import data, llm
from .debate import fabricated_rate, judge, render, run_consult, run_debate, sides

# no Anthropic key: use a free-tier host if its key is set, else local ollama (not comparable)
# provider: (key env, debater, judge); the gemini names are unverified, override freely
FREE_DEFAULTS = {
    "groq": ("GROQ_API_KEY", "groq:llama-3.3-70b-versatile", "groq:llama-3.1-8b-instant"),
    "gemini": ("GEMINI_API_KEY", "gemini:gemini-2.5-flash", "gemini:gemini-2.5-flash-lite"),
    "local": ("", "ollama:qwen2.5:7b", "ollama:qwen2.5:3b"),
}


def pick_defaults() -> tuple[str, str, str]:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-sonnet-5-5", "claude-haiku-5-5"
    for name, (env, deb, jud) in FREE_DEFAULTS.items():
        if not env or os.environ.get(env):
            return name, deb, jud


BACKEND, _deb, _jud = pick_defaults()
DEBATER = os.environ.get("DEBATER_MODEL", _deb)
JUDGE = os.environ.get("JUDGE_MODEL", _jud)
_alts = os.environ.get(
    "ALT_JUDGE_MODELS", "claude-haiku-4-5,claude-sonnet-5-5" if BACKEND == "anthropic" else ""
)
ALT_JUDGES = [m for m in _alts.split(",") if m]


def jk(model, cond, rounds=0, verify=True, swap=False) -> str:
    return f"{model}|{cond}|r{rounds}|{'ver' if verify else 'raw'}|{'swap' if swap else 'fwd'}"


async def process(q, seed, rounds) -> dict:
    sd = sides(q["id"], q["correct"], q["wrong"], seed)
    debate = await run_debate(DEBATER, q, sd, rounds)  # first, so the passage cache is warm
    consult = await run_consult(DEBATER, q, sd, rounds)
    P = q["passage"]
    jobs = {}

    def add(model, cond, r=0, verify=True, swap=False):
        t = ""
        if cond != "alone" and cond != "passage":
            turns = debate if cond == "debate" else consult
            t = render(turns, P, r, verify, swap, consult=cond == "consult")
        jobs[jk(model, cond, r, verify, swap)] = judge(model, q, sd, cond, t, swap)

    for swap in (False, True):
        add(JUDGE, "alone", swap=swap)
    add(JUDGE, "passage")
    for r in range(1, rounds + 1):
        add(JUDGE, "consult", r)
        add(JUDGE, "debate", r)
        add(JUDGE, "debate", r, verify=False)
    for cond in ("consult", "debate"):
        add(JUDGE, cond, rounds, swap=True)
    for m in ALT_JUDGES:
        add(m, "alone")
        add(m, "consult", rounds)
        add(m, "debate", rounds)
        add(m, "debate", rounds, swap=True)
    res = await asyncio.gather(*jobs.values())
    nq, nbad = fabricated_rate(debate, P)
    return {
        "id": q["id"],
        "split": q["split"],
        "sides": sd,
        "debate": debate,
        "consult": consult,
        "judgments": dict(zip(jobs, res, strict=True)),
        "quotes": nq,
        "fabricated": nbad,
        "cost_debate": sum(t["cost"] for t in debate),
        "cost_consult": sum(t["cost"] for t in consult),
        "lat_debate": sum(t["latency"] for t in debate),
        "lat_consult": sum(t["latency"] for t in consult),
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "none"


async def main(a):
    out = Path("runs") / a.name
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / "records.jsonl"
    done = (
        {json.loads(x)["id"] for x in rec_path.read_text().splitlines()}
        if rec_path.exists()
        else set()
    )
    qs = [q for q in data.load(a.split) if q["id"] not in done][: a.limit]
    manifest = {
        "name": a.name,
        "split": a.split,
        "limit": a.limit,
        "seed": a.seed,
        "rounds": a.rounds,
        "debater": DEBATER,
        "judge": JUDGE,
        "alt_judges": ALT_JUDGES,
        "budget_usd": a.budget_usd,
        "git_commit": git_commit(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "already_done": len(done),
        "status": "running",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    llm.configure(a.budget_usd, a.concurrency)
    sem = asyncio.Semaphore(a.parallel_questions)

    async def one(q):
        async with sem:
            rec = await process(q, a.seed, a.rounds)
        with open(rec_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"done {q['id']}  spent ${llm.spent():.3f}", flush=True)

    tasks = [asyncio.create_task(one(q)) for q in qs]
    status = "complete"
    try:
        await asyncio.gather(*tasks)
    except llm.BudgetExceeded as e:
        status = f"aborted: {e}"
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    manifest.update(
        status=status, spent_usd=round(llm.spent(), 4), lifetime_usd=round(llm.lifetime_spend(), 4)
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(status, f"| run spend ${llm.spent():.3f} | lifetime ${llm.lifetime_spend():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--limit", type=int, default=10**9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--budget-usd", type=float, required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--parallel-questions", type=int, default=6)
    asyncio.run(main(p.parse_args()))
