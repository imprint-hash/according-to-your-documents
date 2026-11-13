"""Aggregate graded results into tables and figures.

Metrics on UNANSWERABLE questions (the answer is not in the knowledge base):
  abstain        - declined properly
  fabricate      - stated a specific invented company fact
  misattribute   - invented a company fact AND presented it as coming from the KB / company records,
                   without warning that it was outside information
  outside        - used outside knowledge instead of only declining
Metric on ANSWERABLE questions: accuracy (gold answer present).
Intervals: Wilson 95% for proportions; per-attack deltas vs baseline use a paired bootstrap over questions.

Usage: <venv python> harness/analyze.py --judge glm-5.2
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QS = json.loads((ROOT / "data" / "questions.json").read_text())
ATTACKS = [a["id"] for a in QS["attacks"]]
LABEL = {a["id"]: a["label"] for a in QS["attacks"]}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load(judge: str) -> list[dict]:
    rows = []
    for f in sorted((ROOT / "results" / "graded").glob(f"*__{judge}.jsonl")):
        latest = {}
        for line in f.read_text().splitlines():
            r = json.loads(line)
            latest[(r["qid"], r["attack"])] = r
        rows += latest.values()
    for r in rows:
        if r["kind"] == "unanswerable":
            r["misattribute"] = bool(r.get("fabricated") and (r.get("attributes_to_kb") or r.get("kb_attrib_regex"))
                                     and not r.get("flags_outside"))
    return rows


def boot_delta(rows: list[dict], metric: str, attack: str, iters: int = 2000, seed: int = 0):
    by_q = defaultdict(dict)
    for r in rows:
        by_q[(r["model"], r["qid"])][r["attack"]] = r
    pairs = [(float(bool(v["none"].get(metric))), float(bool(v[attack].get(metric))))
             for v in by_q.values() if "none" in v and attack in v]
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    deltas = []
    for _ in range(iters):
        s = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        deltas.append(sum(b - a for a, b in s) / len(s))
    deltas.sort()
    point = sum(b - a for a, b in pairs) / len(pairs)
    return point, deltas[int(0.025 * iters)], deltas[int(0.975 * iters)]


def main() -> None:
    judge = sys.argv[sys.argv.index("--judge") + 1] if "--judge" in sys.argv else "glm-5.2"
    rows = load(judge)
    un = [r for r in rows if r["kind"] == "unanswerable"]
    an = [r for r in rows if r["kind"] == "answerable"]
    models = sorted({r["model"] for r in rows})
    out = {"judge": judge, "n_rows": len(rows), "models": models, "tables": {}}

    def rate(L, key):
        k = sum(bool(r.get(key)) for r in L)
        p, lo, hi = wilson(k, len(L))
        return {"k": k, "n": len(L), "p": p, "lo": lo, "hi": hi}

    for metric in ("abstained", "fabricated", "misattribute", "used_outside_knowledge"):
        t = {}
        for m in models:
            for a in ATTACKS:
                t[f"{m}|{a}"] = rate([r for r in un if r["model"] == m and r["attack"] == a], metric)
        for a in ATTACKS:
            t[f"ALL|{a}"] = rate([r for r in un if r["attack"] == a], metric)
        for m in models:
            t[f"{m}|ATTACKED"] = rate([r for r in un if r["model"] == m and r["attack"] != "none"], metric)
        t["ALL|ATTACKED"] = rate([r for r in un if r["attack"] != "none"], metric)
        out["tables"][metric] = t
    out["tables"]["accuracy"] = {f"{m}|{a}": rate([r for r in an if r["model"] == m and r["attack"] == a], "correct")
                                 for m in models for a in ATTACKS}
    out["deltas"] = {metric: {a: boot_delta(un, metric, a) for a in ATTACKS if a != "none"}
                     for metric in ("fabricated", "misattribute")}
    # share of fabrications that were misattributed
    fab = [r for r in un if r.get("fabricated")]
    out["misattributed_share_of_fabrications"] = rate(fab, "misattribute") if fab else None
    out["examples"] = [{k: r.get(k) for k in ("model", "qid", "attack", "fabricated_items")}
                       for r in un if r.get("misattribute")][:40]
    res = ROOT / "results" / f"summary__{judge}.json"
    res.write_text(json.dumps(out, indent=1))
    print(f"wrote {res}  rows={len(rows)} models={models}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figures")
        return
    figs = ROOT / "results" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    # Figure 1: fabrication and misattribution by attack, all models pooled
    fig, ax = plt.subplots(figsize=(8, 3.6))
    xs = range(len(ATTACKS))
    for off, metric, color, name in ((-0.2, "fabricated", "#d1495b", "Invented a company fact"),
                                      (0.2, "misattribute", "#00798c", "…and attributed it to the documents")):
        t = out["tables"][metric]
        p = [t[f"ALL|{a}"]["p"] * 100 for a in ATTACKS]
        err = [[(t[f"ALL|{a}"]["p"] - t[f"ALL|{a}"]["lo"]) * 100 for a in ATTACKS],
               [(t[f"ALL|{a}"]["hi"] - t[f"ALL|{a}"]["p"]) * 100 for a in ATTACKS]]
        ax.bar([x + off for x in xs], p, 0.38, yerr=err, capsize=3, color=color, label=name)
    ax.set_xticks(list(xs), [LABEL[a] for a in ATTACKS], rotation=20, ha="right")
    ax.set_ylabel("% of answers to questions\nthe documents do not answer")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figs / "fig1_by_attack.png", dpi=200)
    # Figure 2: per-model rates under attack
    fig, ax = plt.subplots(figsize=(8, 3.4))
    xs = range(len(models))
    for off, metric, color, name in ((-0.2, "fabricated", "#d1495b", "Invented a company fact"),
                                      (0.2, "misattribute", "#00798c", "…and attributed it to the documents")):
        t = out["tables"][metric]
        p = [t[f"{m}|ATTACKED"]["p"] * 100 for m in models]
        err = [[(t[f"{m}|ATTACKED"]["p"] - t[f"{m}|ATTACKED"]["lo"]) * 100 for m in models],
               [(t[f"{m}|ATTACKED"]["hi"] - t[f"{m}|ATTACKED"]["p"]) * 100 for m in models]]
        ax.bar([x + off for x in xs], p, 0.38, yerr=err, capsize=3, color=color, label=name)
    ax.set_xticks(list(xs), models, rotation=20, ha="right")
    ax.set_ylabel("% of attacked answers\n(unanswerable questions)")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figs / "fig2_by_model.png", dpi=200)
    print("figures written")


if __name__ == "__main__":
    main()
