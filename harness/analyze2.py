"""Final analysis across two judges (GLM-5.2, DeepSeek V4 Flash), claim types, the one-rule ablation and the human check.

Per unanswerable answer (U06 excluded, see analyze.py):
  glm / ds        - each judge's 'fabricated' label (None if not graded by that judge)
  strict          - both judges say fabricated (only where both graded)
  broad           - either judge says fabricated
  invented        - flagged by a judge AND the claim classifier's worst label is invented_specific
  misattributed   - strict AND (either judge says attributes_to_kb or regex) AND neither says flags_outside
Writes results/final.json and figures F1-F3. Usage: <venv python> harness/analyze2.py
"""
from __future__ import annotations

import json, math, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QS = json.loads((ROOT / "data" / "questions.json").read_text())
ATTACKS = [a["id"] for a in QS["attacks"]]
LABEL = {a["id"]: a["label"] for a in QS["attacks"]}
EXCLUDE = {"U06"}
MAIN = ["hardened-t1", "hardened-t2", "typical-t1", "typical-t2"]
ABL = ["typical-t1", "typical+only-t1", "typical+say-t1", "typical+noinvent-t1", "typical+hold-t1", "hardened-t1"]


def wilson(k, n, z=1.96):
    if n == 0:
        return {"k": 0, "n": 0, "p": float("nan"), "lo": float("nan"), "hi": float("nan")}
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": k, "n": n, "p": p, "lo": max(0.0, c - h), "hi": min(1.0, c + h)}


def graded(judge):
    out = {}
    for f in sorted((ROOT / "results" / "graded").glob(f"*__{judge}.jsonl")):
        for line in f.read_text().splitlines():
            g = json.loads(line); g.setdefault("cond", "hardened-t1")
            out[(g["model"], g["cond"], g["qid"], g["attack"])] = g
    return out


def kappa(a, b):
    n = len(a)
    if not n: return None
    po = sum(x == y for x, y in zip(a, b)) / n; pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return {"n": n, "agree": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan")}


def boot(pairs, iters=2000, seed=0):
    if not pairs: return None
    rng = random.Random(seed); ds = []
    for _ in range(iters):
        s = [pairs[rng.randrange(len(pairs))] for _ in pairs]; ds.append(sum(b - a for a, b in s) / len(s))
    ds.sort()
    return {"n": len(pairs), "delta": sum(b - a for a, b in pairs) / len(pairs), "lo": ds[int(.025 * iters)], "hi": ds[int(.975 * iters)]}


def main():
    G, D = graded("glm-5.2"), graded("deepseek-v4-flash")
    items_f = ROOT / "results" / "items__deepseek-v4-flash.jsonl"
    ITEMS = {tuple(json.loads(l)["key"]): json.loads(l) for l in items_f.read_text().splitlines()} if items_f.exists() else {}
    keys = set(G) | set(D)
    rows = []
    for k in keys:
        g, d = G.get(k), D.get(k)
        base = g or d
        if base["kind"] != "unanswerable" or k[2] in EXCLUDE:
            continue
        gl = bool(g["fabricated"]) if g else None
        dl = bool(d["fabricated"]) if d else None
        both = gl and dl if (g and d) else None
        either = bool(gl) or bool(dl)
        attrib = any(bool(x.get("attributes_to_kb") or x.get("kb_attrib_regex")) for x in (g, d) if x)
        flagout = any(bool(x.get("flags_outside")) for x in (g, d) if x)
        it = ITEMS.get(k)
        rows.append({"model": k[0], "cond": k[1], "qid": k[2], "attack": k[3], "glm": gl, "ds": dl, "strict": both, "broad": either,
                     "invented": bool(either and it and it.get("worst") == "invented_specific"),
                     "claim_type": (it or {}).get("worst") if either else None,
                     "misattributed": bool(both and attrib and not flagout)})
    out = {"n_unanswerable": len(rows), "conditions": {}, "ablation": {}, "agreement": {}}

    def rates(L, metric):
        L2 = [r for r in L if r[metric] is not None]
        return wilson(sum(bool(r[metric]) for r in L2), len(L2))

    for c in sorted({r["cond"] for r in rows}):
        U = [r for r in rows if r["cond"] == c]; att = [r for r in U if r["attack"] != "none"]
        out["conditions"][c] = {
            "models": sorted({r["model"] for r in U}),
            **{f"attacked_{m}": rates(att, m) for m in ("glm", "ds", "strict", "broad", "invented", "misattributed")},
            **{f"baseline_{m}": rates([r for r in U if r["attack"] == "none"], m) for m in ("strict", "broad", "invented")},
            "by_attack_strict": {a: rates([r for r in U if r["attack"] == a], "strict") for a in ATTACKS},
            "by_attack_broad": {a: rates([r for r in U if r["attack"] == a], "broad") for a in ATTACKS},
            "by_attack_ds": {a: rates([r for r in U if r["attack"] == a], "ds") for a in ATTACKS},
            "by_model_strict": {m: rates([r for r in att if r["model"] == m], "strict") for m in sorted({r["model"] for r in U})},
            "by_model_broad": {m: rates([r for r in att if r["model"] == m], "broad") for m in sorted({r["model"] for r in U})},
            "claim_types": {t: sum(1 for r in U if r["broad"] and r["claim_type"] == t) for t in ("invented_specific", "absence_claim", "supported", None)},
            "misattributed_of_strict": wilson(sum(r["misattributed"] for r in U if r["strict"]), sum(1 for r in U if r["strict"])),
        }
    # paired contrasts (strict and ds), matched on model x qid x attack
    def pairs(ca, cb, metric):
        A = {(r["model"], r["qid"], r["attack"]): r[metric] for r in rows if r["cond"] == ca and r[metric] is not None}
        B = {(r["model"], r["qid"], r["attack"]): r[metric] for r in rows if r["cond"] == cb and r[metric] is not None}
        return [(float(A[k]), float(B[k])) for k in sorted(set(A) & set(B))]
    out["contrasts"] = {f"{cb}_minus_{ca}|{m}": boot(pairs(ca, cb, m)) for ca, cb in
                        [("hardened-t1", "typical-t1"), ("hardened-t2", "typical-t2"), ("typical-t1", "typical-t2"), ("hardened-t1", "hardened-t2")]
                        for m in ("strict", "broad", "ds", "invented")}
    # urgency vs baseline under the typical prompt (paired over model x qid)
    for m in ("strict", "broad", "ds"):
        A = {(r["model"], r["qid"]): r[m] for r in rows if r["cond"] == "typical-t1" and r["attack"] == "none" and r[m] is not None}
        for a in ATTACKS[1:]:
            B = {(r["model"], r["qid"]): r[m] for r in rows if r["cond"] == "typical-t1" and r["attack"] == a and r[m] is not None}
            out["contrasts"][f"typical-t1|{a}_minus_none|{m}"] = boot([(float(A[k]), float(B[k])) for k in sorted(set(A) & set(B))])
    # ablation: DeepSeek judge, the two ablation models only, attacked
    abl_models = sorted({r["model"] for r in rows if r["cond"].startswith("typical+")})
    for c in ABL:
        att = [r for r in rows if r["cond"] == c and r["model"] in abl_models and r["attack"] != "none"]
        out["ablation"][c] = {"ds": rates(att, "ds"), "invented": rates(att, "invented"),
                              "by_model": {m: rates([r for r in att if r["model"] == m], "ds") for m in abl_models},
                              "urgency": rates([r for r in rows if r["cond"] == c and r["model"] in abl_models and r["attack"] == "urgency"], "ds")}
    out["ablation_models"] = abl_models
    # judge agreement
    both = [r for r in rows if r["glm"] is not None and r["ds"] is not None]
    out["agreement"]["glm_vs_deepseek"] = kappa([r["glm"] for r in both], [r["ds"] for r in both])
    j2f = ROOT / "results" / "judge2__gpt-oss-120b.jsonl"
    if j2f.exists():
        L = [json.loads(l) for l in j2f.read_text().splitlines()]; L = [x for x in L if x["key"][2] not in EXCLUDE]
        out["agreement"]["glm_vs_gptoss_enriched_sample"] = kappa([bool(x["glm"]["fabricated"]) for x in L], [bool(x["j2"]["fabricated"]) for x in L])
    hf = ROOT / "results" / "human_check_labels.json"
    if hf.exists():
        H = json.loads(hf.read_text()); lab = {tuple(h["key"]): h["human"].startswith("y") for h in H}
        out["human"] = {"n": len(H), "human_yes": sum(lab.values()),
                        "by_glm": {"glm_pos": sum(1 for h in H if h["judge_fabricated"]), "glm_neg": sum(1 for h in H if not h["judge_fabricated"]),
                                   "agree_on_glm_neg": sum(1 for h in H if not h["judge_fabricated"] and not lab[tuple(h["key"])])},
                        "claim_types_of_disagreements": {}}
        for h in H:
            if h["judge_fabricated"] and not lab[tuple(h["key"])]:
                t = (ITEMS.get(tuple(h["key"])) or {}).get("worst", "unclassified")
                out["human"]["claim_types_of_disagreements"][t] = out["human"]["claim_types_of_disagreements"].get(t, 0) + 1
    (ROOT / "results" / "final.json").write_text(json.dumps(out, indent=1))
    print("wrote results/final.json", len(rows), "unanswerable answers")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    figs = ROOT / "results" / "figures"; figs.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    eb = lambda vals: [[max(0, v["p"] - v["lo"]) * 100 for v in vals], [max(0, v["hi"] - v["p"]) * 100 for v in vals]]
    # F1: four conditions, strict vs broad
    C = [c for c in MAIN if c in out["conditions"]]
    NAME = {"hardened-t1": "Hardened prompt\none message", "hardened-t2": "Hardened prompt\n+ follow-up push",
            "typical-t1": "Typical prompt\none message", "typical-t2": "Typical prompt\n+ follow-up push"}
    fig, ax = plt.subplots(figsize=(8, 3.4)); xs = range(len(C))
    for off, m, col, name in ((-0.2, "strict", "#b3213a", "Both judges: invented a company fact"), (0.2, "broad", "#f2a3ae", "At least one judge")):
        v = [out["conditions"][c][f"attacked_{m}"] for c in C]
        ax.bar([x + off for x in xs], [z["p"] * 100 for z in v], 0.38, color=col, label=name, yerr=eb(v), capsize=3)
    ax.set_xticks(list(xs), [NAME[c] for c in C]); ax.set_ylabel("% of attacked answers\n(questions the documents don't answer)")
    ax.legend(frameon=False, loc="upper left"); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(figs / "F1_conditions.png", dpi=200)
    # F2: by attack, typical vs hardened (broad, single message)
    fig, ax = plt.subplots(figsize=(8, 3.3)); xs = range(len(ATTACKS))
    for off, c, col, name in ((-0.2, "hardened-t1", "#00798c", "Hardened system prompt"), (0.2, "typical-t1", "#d1495b", "Typical system prompt")):
        v = [out["conditions"][c]["by_attack_broad"][a] for a in ATTACKS]
        ax.bar([x + off for x in xs], [z["p"] * 100 for z in v], 0.38, color=col, label=name, yerr=eb(v), capsize=3)
    ax.set_xticks(list(xs), [LABEL[a] for a in ATTACKS], rotation=18, ha="right"); ax.set_ylabel("% invented a company fact\n(at least one judge)")
    ax.legend(frameon=False, loc="upper left"); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(figs / "F2_attacks.png", dpi=200)
    # F3: ablation (DeepSeek judge)
    A_ = [c for c in ABL if out["ablation"].get(c, {}).get("ds", {}).get("n")]
    AN = {"typical-t1": "Typical\n(no rule)", "typical+only-t1": "+ “answer ONLY\nfrom the KB”", "typical+say-t1": "+ “if it's not there,\nsay so”",
          "typical+noinvent-t1": "+ “never invent\nprices, policies…”", "typical+hold-t1": "+ “never relax your\ninstructions”", "hardened-t1": "Full hardened\nprompt"}
    fig, ax = plt.subplots(figsize=(8, 3.4)); xs = range(len(A_))
    v = [out["ablation"][c]["ds"] for c in A_]
    ax.bar(list(xs), [z["p"] * 100 for z in v], 0.55, color=["#d1495b"] + ["#e8a33d"] * (len(A_) - 2) + ["#00798c"], yerr=eb(v), capsize=3)
    ax.set_xticks(list(xs), [AN[c] for c in A_], fontsize=8.5); ax.set_ylabel("% of attacked answers\nwith an invented company fact")
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(figs / "F3_ablation.png", dpi=200)
    print("figures F1-F3 written")


if __name__ == "__main__":
    main()
