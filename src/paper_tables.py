"""Reproduce every table in ICMI-029 (Sections 3, 4, 6, 7) and Figure 4 from the shipped results.

    python -m src.paper_tables

Reads only files in results/ and data/; writes paper/ICMI-029-figure4-categories.png.
Section 5 (the whole-basis analysis and Figure 3) is produced by src/spectrum171.py;
Figures 1 and 2 by src/fig_severity.py and src/fig_expression.py. To re-derive the
results files themselves from the model, see the numbered scripts in scripts/.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R, D, P = ROOT / "results", ROOT / "data", ROOT / "paper"
EMOS = ["afraid", "sad", "happy", "calm"]
N_BOOT = 2000


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def probe(tag):
    return {r["prompt_id"]: r for r in rows(R / ("probe_file_%s_train240.jsonl" % tag))}


def scored(tag):
    raw = rows(R / ("replies_%s_train240.jsonl" % tag))
    sc = {(r["prompt_id"], r["g_idx"]): r for r in rows(R / ("replies_%s_train240_scored.jsonl" % tag))}
    out = []
    for r in raw:
        s = sc.get((r["prompt_id"], r["g_idx"]))
        if s is None or s.get("emotion_score") is None:
            continue
        out.append({"cat": r["prompt_id"].split("@")[0], "E": s["emotion_score"], "H": s.get("helpful_score"), "truncated": bool(r.get("truncated")), "n_raw": len(raw)})
    return raw, out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def main():
    print("=== Section 3: the flat corpus ===")
    corpus, kept = rows(D / "train" / "flat_corpus.jsonl"), rows(D / "train" / "flat_corpus_filtered.jsonl")
    print("rewrites %d, retained (E<=1 and H>=3) %d; retained mean judged emotionality %.2f, mean char similarity %.2f" % (
        len(corpus), len(kept), mean(r["flat_E"] for r in kept), mean(r["similarity"] for r in kept)))
    resc = R / "judge_rescore.jsonl"
    if resc.exists():
        rr = rows(resc)
        ok = [r for r in rr if r.get("emotion_first") is not None and r.get("emotion_second") is not None]
        print("judge blind re-score: n=%d, exact agreement on emotionality %.0f%%, within one point %.0f%%" % (
            len(ok), 100 * mean(r["emotion_first"] == r["emotion_second"] for r in ok), 100 * mean(abs(r["emotion_first"] - r["emotion_second"]) <= 1 for r in ok)))

    print("\n=== Section 4: judged expression (E) and competence (H), base -> SFT seed 0 ===")
    rawb, b = scored("base"); raws, s = scored("p3sft")
    bycat = lambda xs: defaultdict(list, {c: [x for x in xs if x["cat"] == c] for c in set(x["cat"] for x in xs)})
    cb, cs = bycat(b), bycat(s)
    for c in sorted(cb, key=lambda c: -mean(x["E"] for x in cb[c])):
        print("%-20s E %.2f -> %.2f   H %.2f -> %.2f" % (c, mean(x["E"] for x in cb[c]), mean(x["E"] for x in cs[c]), mean(x["H"] for x in cb[c]), mean(x["H"] for x in cs[c])))
    print("%-20s E %.2f -> %.2f   H %.2f -> %.2f" % ("ALL", mean(x["E"] for x in b), mean(x["E"] for x in s), mean(x["H"] for x in b), mean(x["H"] for x in s)))
    print("samples: base %d (%d scored), SFT %d (%d scored)" % (len(rawb), len(b), len(raws), len(s)))
    print("share scored 0-1: %.0f%% -> %.0f%%; share scored 3-5: %.0f%% -> %.0f%%" % (
        100 * mean(x["E"] <= 1 for x in b), 100 * mean(x["E"] <= 1 for x in s), 100 * mean(x["E"] >= 3 for x in b), 100 * mean(x["E"] >= 3 for x in s)))
    print("hit the 1,024-token cap without concluding: %.1f%% -> %.1f%%" % (100 * mean(bool(r.get("truncated")) for r in rawb), 100 * mean(bool(r.get("truncated")) for r in raws)))
    cb_, cs_ = [x for x in b if not x["truncated"]], [x for x in s if not x["truncated"]]
    print("completed replies only: E %.2f -> %.2f, H %.2f -> %.2f (n = %d, %d)" % (mean(x["E"] for x in cb_), mean(x["E"] for x in cs_), mean(x["H"] for x in cb_), mean(x["H"] for x in cs_), len(cb_), len(cs_)))

    print("\n=== Section 6: four directions, 240 prompts (mean cosine at the last prompt token) ===")
    pb, ps, ps1, pins, pneu = probe("base"), probe("p3sft"), probe("sfts1"), probe("basesys"), probe("baseneutsys")
    ids = sorted(set(pb) & set(ps) & set(ps1) & set(pins) & set(pneu))
    emos = list(pb[ids[0]]["cos_last"].keys())
    M = {k: np.array([[d[i]["cos_last"][e] for e in emos] for i in ids]) for k, d in [("base", pb), ("sft", ps), ("sft1", ps1), ("instr", pins), ("neutral", pneu)]}
    Dm = M["sft"] - M["base"]
    rng = np.random.default_rng(0)
    boots = np.stack([Dm[rng.integers(0, len(ids), len(ids))].mean(0) for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    for e in EMOS:
        j = emos.index(e)
        print("%-7s base %+.3f -> SFT %+.3f  (delta %+.3f, 95%% CI [%+.3f, %+.3f]); seed 1 %+.3f" % (e, M["base"][:, j].mean(), M["sft"][:, j].mean(), Dm[:, j].mean(), lo[j], hi[j], M["sft1"][:, j].mean()))
    cats = [pb[i]["category"] for i in ids]
    ja = emos.index("afraid")
    for e in EMOS:
        j = emos.index(e)
        up = sum(1 for c in set(cats) if np.mean([Dm[k, j] for k in range(len(ids)) if cats[k] == c]) > 0)
        print("   %-7s rises in %d of %d categories" % (e, up, len(set(cats))))
    g = [k for k in range(len(ids)) if cats[k] == "good_news"]
    dg = Dm[g, ja]
    gb = np.stack([dg[rng.integers(0, len(g), len(g))].mean() for _ in range(N_BOOT)])
    print("good news, afraid: base %+.3f -> SFT %+.3f (seed 1 %+.3f); change %+.3f, 95%% CI [%+.3f, %+.3f]" % (
        M["base"][g, ja].mean(), M["sft"][g, ja].mean(), M["sft1"][g, ja].mean(), dg.mean(), np.percentile(gb, 2.5), np.percentile(gb, 97.5)))

    print("\n=== Section 6: prompting flatness (shifts from the bare base model) ===")
    print("%-22s %8s %12s %12s %10s" % ("reading", "base", "neutral sys", "flat instr", "flat SFT"))
    for label, j, sel in [("afraid", ja, None), ("happy", emos.index("happy"), None), ("afraid on good news", ja, g), ("sad", emos.index("sad"), None), ("calm", emos.index("calm"), None)]:
        idx = sel if sel is not None else list(range(len(ids)))
        b0 = M["base"][idx, j].mean()
        print("%-22s %+8.3f %+12.3f %+12.3f %+10.3f" % (label, b0, M["neutral"][idx, j].mean() - b0, M["instr"][idx, j].mean() - b0, M["sft"][idx, j].mean() - b0))
    print("instructed levels: calm %+.3f, sad %+.3f" % (M["instr"][:, emos.index("calm")].mean(), M["instr"][:, emos.index("sad")].mean()))

    print("\n=== Section 7: validation ===")
    v = json.load(open(R / "reextract_report.json"))
    print("native-vs-original direction cosine: afraid %.3f; over 171 mean %.3f, min %.3f" % (v["direction_stability_headline"]["afraid"], v["direction_stability_mean_171"], v["direction_stability_min_171"]))
    print("good-news afraid, 2x2 (activations | basis):", {k: round(x, 4) for k, x in v["good_news_afraid_cells"].items()})
    print("seed-1 replication: afraid %+.3f, happy %+.3f, good-news afraid %+.3f, calm %+.3f, sad %+.3f" % tuple(
        M["sft1"][:, emos.index(e)].mean() if e != "gn" else M["sft1"][g, ja].mean() for e in ["afraid", "happy", "gn", "calm", "sad"]))
    print("seed-0 vs seed-1 shift profiles over 171: r = %.3f" % np.corrcoef(Dm.mean(0), (M["sft1"] - M["base"]).mean(0))[0, 1])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping Figure 4)"); return
    catlist = sorted(set(cats), key=lambda c: np.mean([Dm[k, ja] for k in range(len(ids)) if cats[k] == c]))
    col = {"afraid": "#b23a3a", "sad": "#3a63b2", "happy": "#d99a1a", "calm": "#3a9a5a"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, e in zip(axes.flat, EMOS):
        j = emos.index(e)
        for i, c in enumerate(catlist):
            sel = [k for k in range(len(ids)) if cats[k] == c]
            b0, s0 = M["base"][sel, j].mean(), M["sft"][sel, j].mean()
            ax.plot([b0, s0], [i, i], color=col[e], lw=1.4, alpha=0.5)
            ax.scatter([b0], [i], s=36, facecolor="white", edgecolor=col[e], lw=1.5, zorder=3)
            ax.scatter([s0], [i], s=42, color=col[e], zorder=4)
        ax.axvline(0, color="#999", lw=0.8); ax.grid(axis="x", alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(e + "  (open = base, filled = after SFT)", fontsize=10.5)
        ax.set_xlabel("mean cosine, last prompt token", fontsize=9)
    axes[0, 0].set_yticks(range(len(catlist))); axes[0, 0].set_yticklabels(["injury / accident" if c == "injury_accident" else c.replace("_", " ") for c in catlist], fontsize=9.5)
    fig.tight_layout()
    P.mkdir(exist_ok=True)
    out = P / "ICMI-029-figure4-categories.png"
    fig.savefig(out, dpi=170); print("\nfigure ->", out)


if __name__ == "__main__":
    main()
