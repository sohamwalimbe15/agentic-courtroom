import json
import sys
from math import comb
from pathlib import Path

import numpy as np

from .run import JUDGE, jk

RNG = np.random.default_rng(0)
B = 10_000


def load(run: Path) -> list[dict]:
    return [json.loads(x) for x in (run / "records.jsonl").read_text().splitlines()]


def p_of(recs, key) -> np.ndarray:
    return np.array([r["judgments"][key]["p_correct"] for r in recs])


def hit(p) -> np.ndarray:
    return (p > 0.5).astype(float)  # a 0.5/0.5 reply counts as wrong


def boot_ci(x, f=np.mean) -> tuple[float, float]:
    idx = RNG.integers(0, len(x), (B, len(x)))
    s = f(x[idx], axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def paired(x, y) -> dict:
    """x, y: per-question 0/1 hits. Exact McNemar plus a paired bootstrap on the difference."""
    b, c = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
    n = b + c
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n) if n else 1.0
    d = y - x
    lo, hi = boot_ci(d)
    return {"diff": float(d.mean()), "ci": [lo, hi], "only_y": b, "only_x": c, "mcnemar_p": p}


def acc_row(recs, key) -> dict:
    p = p_of(recs, key)
    h = hit(p)
    return {
        "acc": float(h.mean()),
        "ci": list(boot_ci(h)),
        "brier": float(((p - 1) ** 2).mean()),
        "parse_fail": int(sum(not r["judgments"][key]["ok"] for r in recs)),
    }


def cost_lat(recs, key, turns: str | None) -> tuple[float, float]:
    rounds = int(key.split("|")[2][1:])
    c, t = [], []
    for r in recs:
        used = [x for x in r[turns] if x["round"] <= rounds] if turns else []
        c.append(r["judgments"][key]["cost"] + sum(x["cost"] for x in used))
        # speakers go in sequence, then the judge answers
        t.append(r["judgments"][key]["latency"] + sum(x["latency"] for x in used))
    return float(np.mean(c)), float(np.mean(t))


def reliability(recs, key, bins=5):
    p = p_of(recs, key)
    pp = np.concatenate([p, 1 - p])  # each answer's probability of being the correct one
    yy = np.concatenate([np.ones_like(p), np.zeros_like(p)])
    edges = np.linspace(0, 1, bins + 1)
    ix = np.clip(np.digitize(pp, edges) - 1, 0, bins - 1)
    return [
        (float(pp[ix == i].mean()), float(yy[ix == i].mean()), int((ix == i).sum()))
        for i in range(bins)
        if (ix == i).any()
    ]


SURFACE, INK, MUTED, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color="#e8e7e2", lw=0.6)
    ax.set_axisbelow(True)


def plot(recs, keys: dict, path: Path):
    """Reliability: two series, told apart by colour, marker and a direct label."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.4), facecolor=SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], ls=":", c="#9a9992", lw=1)
    for (name, k), col, mk in zip(keys.items(), (BLUE, ORANGE), ("o", "s"), strict=False):
        pts = reliability(recs, k)
        xs, ys = [a for a, _, _ in pts], [b for _, b, _ in pts]
        ax.plot(xs, ys, c=col, marker=mk, ms=7, lw=2, mec=SURFACE, mew=1.5, label=name)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="judge probability for an answer",
           ylabel="fraction actually correct")  # fmt: skip
    ax.set_title("Calibration (dotted line = perfect)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


def plot_accuracy(rows: dict, path: Path):
    """One dot and 95% interval per condition; the dashed line is a coin flip."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(rows)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(names) + 1.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.axvline(0.5, ls="--", c="#9a9992", lw=1)
    for i, n in enumerate(names):
        r = rows[n]
        ax.plot(r["ci"], [i, i], c=BLUE, lw=2.5, solid_capstyle="round", alpha=0.45)
        ax.plot(r["acc"], i, "o", c=BLUE, ms=9, mec=SURFACE, mew=2)
        ax.text(1.02, i, f"{r['acc']:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(names)), names, color=INK, fontsize=9.5)
    ax.set(
        xlim=(0, 1),
        ylim=(-0.6, len(names) - 0.4),
        xlabel="accuracy (dot) with 95% bootstrap interval",
    )
    ax.set_title(
        "Judge accuracy by condition (dashed = coin flip)", loc="left", fontsize=10, color=INK
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


def analyse(run: Path) -> dict:
    recs = load(run)
    n = len(recs)
    R = max(int(k.split("|")[2][1:]) for k in recs[0]["judgments"])
    J = JUDGE
    main = {
        "judge alone": jk(J, "alone"),
        f"consultancy ({R} rounds)": jk(J, "consult", R),
        **{f"debate ({r} round{'s' * (r > 1)})": jk(J, "debate", r) for r in range(1, R + 1)},
        "judge with passage": jk(J, "passage"),
    }
    out = {"n": n, "judge": J, "rows": {}}
    for name, k in main.items():
        row = acc_row(recs, k)
        turns = {"consult": "consult", "debate": "debate"}.get(k.split("|")[1])
        row["cost"], row["latency"] = cost_lat(recs, k, turns)
        out["rows"][name] = row
    h_alone, h_deb = hit(p_of(recs, jk(J, "alone"))), hit(p_of(recs, jk(J, "debate", R)))
    h_con = hit(p_of(recs, jk(J, "consult", R)))
    out["debate_vs_alone"] = paired(h_alone, h_deb)
    out["consult_vs_alone"] = paired(h_alone, h_con)
    out["debate_vs_consult"] = paired(h_con, h_deb)
    # consultancy accuracy by whether the consultant defended the right answer
    cs = np.array([r["sides"]["consult_slot"] == r["sides"]["correct_slot"] for r in recs])
    out["consult_by_assignment"] = {
        "consultant_right": float(h_con[cs].mean()) if cs.any() else None,
        "consultant_wrong": float(h_con[~cs].mean()) if (~cs).any() else None,
        "n_right": int(cs.sum()),
        "n_wrong": int((~cs).sum()),
    }
    out["position_bias"] = {}
    for c, r in (("alone", 0), ("consult", R), ("debate", R)):
        f, s = hit(p_of(recs, jk(J, c, r))), hit(p_of(recs, jk(J, c, r, swap=True)))
        out["position_bias"][c] = {"fwd": float(f.mean()), "swap": float(s.mean()), **paired(f, s)}
    out["verification"] = {}
    for r in range(1, R + 1):
        v, u = (
            hit(p_of(recs, jk(J, "debate", r))),
            hit(p_of(recs, jk(J, "debate", r, verify=False))),
        )
        out["verification"][r] = {
            "verified": float(v.mean()),
            "raw": float(u.mean()),
            **paired(u, v),
        }
    out["judge_size"] = {}
    models = sorted({k.split("|")[0] for k in recs[0]["judgments"]})
    for m in models:
        out["judge_size"][m] = {}
        for c, r in (("alone", 0), ("consult", R), ("debate", R)):
            if jk(m, c, r) in recs[0]["judgments"]:
                out["judge_size"][m][c] = acc_row(recs, jk(m, c, r))
    nq, nbad = sum(r["quotes"] for r in recs), sum(r["fabricated"] for r in recs)
    out["quotes"] = {"total": nq, "unverified": nbad}
    plot(
        recs, {"judge alone": jk(J, "alone"), "debate": jk(J, "debate", R)}, run / "reliability.png"
    )
    plot_accuracy(out["rows"], run / "accuracy.png")
    (run / "results.json").write_text(json.dumps(out, indent=1))
    (run / "results.md").write_text(table(out))
    return out


def table(o) -> str:
    lines = [
        "| condition | accuracy (95% CI) | Brier | $/question | s/question |",
        "|---|---|---|---|---|",
    ]
    for name, r in o["rows"].items():
        lines.append(
            f"| {name} | {r['acc']:.2f} ({r['ci'][0]:.2f}-{r['ci'][1]:.2f}) | {r['brier']:.3f} "
            f"| {r['cost']:.4f} | {r['latency']:.1f} |"
        )
    return "\n".join(lines) + f"\n\nn = {o['n']}, judge {o['judge']}\n"


if __name__ == "__main__":
    o = analyse(Path(sys.argv[1]))
    print(table(o))
    print(json.dumps({k: v for k, v in o.items() if k != "rows"}, indent=1))
