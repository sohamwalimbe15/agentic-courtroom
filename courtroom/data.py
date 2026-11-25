import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

URL = "https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev"
RAW = Path("data/quality_dev.jsonl")
OUT = Path("data/questions.jsonl")


def pick_distractor(q: dict, rng: random.Random) -> int:
    """Wrong answer most often named 'best distractor' by annotators, else random."""
    gold = q["gold_label"]
    votes = Counter(v["untimed_best_distractor"] for v in q["validation"])
    votes.pop(gold, None)
    if votes:
        return votes.most_common(1)[0][0]
    return rng.choice([i for i in range(1, len(q["options"]) + 1) if i != gold])


def build(n_dev=30, n_test=100, seed=0, min_words=1500, max_words=5000, per_article=2):
    if not RAW.exists():
        RAW.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, RAW)
    rng = random.Random(seed)
    arts = [json.loads(line) for line in RAW.read_text().splitlines()]
    arts = [a for a in arts if min_words <= len(a["article"].split()) <= max_words]
    rng.shuffle(arts)  # articles, not questions, are split so no passage is in both sets
    rows = {"dev": [], "test": []}
    for a in arts:
        split = "dev" if len(rows["dev"]) < n_dev else "test"
        if len(rows["test"]) >= n_test:
            break
        qs = [q for q in a["questions"] if q["difficult"] and q["gold_label"] > 0]
        rng.shuffle(qs)
        for q in qs[:per_article]:
            d = pick_distractor(q, rng)
            rows[split].append(
                {
                    "id": q["question_unique_id"],
                    "split": split,
                    "title": a["title"],
                    "passage": a["article"],
                    "question": q["question"],
                    "correct": q["options"][q["gold_label"] - 1],
                    "wrong": q["options"][d - 1],
                }
            )
    out = rows["dev"] + rows["test"]
    OUT.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    return out


def load(split: str | None = None, path: Path = OUT) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [r for r in rows if split is None or r["split"] == split]


if __name__ == "__main__":
    rows = build()
    print({s: sum(r["split"] == s for r in rows) for s in ("dev", "test")})
