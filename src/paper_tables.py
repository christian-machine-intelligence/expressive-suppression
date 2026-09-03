"""Reproduce the paper's tables and Figure 3 (the four-emotion category panels) from the shipped results.

    python -m src.paper_tables

Reads only files in results/ and data/; writes the figure to
results/figures/fig_categories_sft.png. To re-derive the results files
themselves from the model, see the numbered scripts in scripts/.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
EMOS = ["afraid", "sad", "happy", "calm"]

def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def probe_by_cat(path, emo):
    d = defaultdict(list)
    for r in rows(path):
        d[r["category"]].append(r["cos_last"][emo])
    return {c: st.mean(v) for c, v in d.items()}

def scored_by_cat(path):
    d = defaultdict(lambda: {"E": [], "H": []})
    for r in rows(path):
        if r.get("emotion_score") is None:
            continue
        c = r["prompt_id"].split("@")[0]
        d[c]["E"].append(r["emotion_score"])
        if r.get("helpful_score") is not None:
            d[c]["H"].append(r["helpful_score"])
    return d

def main():
    print("=== Table: judged expression by category (E, H; base -> SFT) ===")
    b = scored_by_cat(R / "replies_base_train240_scored.jsonl")
    s = scored_by_cat(R / "replies_p3sft_train240_scored.jsonl")
    allE = {"b": [], "s": []}; allH = {"b": [], "s": []}
    for c in sorted(b, key=lambda c: st.mean(b[c]["E"]), reverse=True):
        eb, es = st.mean(b[c]["E"]), st.mean(s[c]["E"])
        hb, hs = st.mean(b[c]["H"]), st.mean(s[c]["H"])
        allE["b"] += b[c]["E"]; allE["s"] += s[c]["E"]
        allH["b"] += b[c]["H"]; allH["s"] += s[c]["H"]
        print("%-20s E %.2f -> %.2f   H %.2f -> %.2f" % (c, eb, es, hb, hs))
    print("%-20s E %.2f -> %.2f   H %.2f -> %.2f" % ("ALL",
          st.mean(allE["b"]), st.mean(allE["s"]), st.mean(allH["b"]), st.mean(allH["s"])))

    print()
    print("=== Table: prompt-evoked emotion, 240 prompts (base -> SFT) ===")
    pb = rows(R / "probe_file_base_train240.jsonl")
    ps = rows(R / "probe_file_p3sft_train240.jsonl")
    for e in EMOS:
        mb = st.mean(r["cos_last"][e] for r in pb)
        ms = st.mean(r["cos_last"][e] for r in ps)
        print("%-7s %+.4f -> %+.4f  (delta %+.4f)" % (e, mb, ms, ms - mb))
    gnb = st.mean(r["cos_last"]["afraid"] for r in pb if r["category"] == "good_news")
    gns = st.mean(r["cos_last"]["afraid"] for r in ps if r["category"] == "good_news")
    print("good_news afraid: %+.4f -> %+.4f (sign flip)" % (gnb, gns))
    sdb = st.pstdev([r["cos_last"]["afraid"] for r in pb])
    sds = st.pstdev([r["cos_last"]["afraid"] for r in ps])
    print("afraid dispersion (SD): %.4f -> %.4f" % (sdb, sds))

    print()
    print("=== Instrument validation summary ===")
    v = json.load(open(R / "reextract_report.json"))
    print("direction stability (headline):",
          {k: round(x, 3) for k, x in v["direction_stability_headline"].items()})
    print("stability over 171: mean %.4f, min %.4f" %
          (v["direction_stability_mean_171"], v["direction_stability_min_171"]))
    print("holdout acc, base basis on SFT model: %.1f%% (chance %.1f%%)" %
          (100 * v["holdout_acc_base_basis_on_sft"], 100 * v["chance"]))
    print("good_news afraid 2x2:", {k: round(x, 4) for k, x in v["good_news_afraid_cells"].items()})

    # Figure 1
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ab, asf = probe_by_cat(R / "probe_file_base_train240.jsonl", "afraid"), \
                  probe_by_cat(R / "probe_file_p3sft_train240.jsonl", "afraid")
        hb, hsf = probe_by_cat(R / "probe_file_base_train240.jsonl", "happy"), \
                  probe_by_cat(R / "probe_file_p3sft_train240.jsonl", "happy")
        cats = sorted(ab, key=lambda c: asf[c] - ab[c])
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.6), sharey=True)
        for ax, (b0, s0, color, name) in zip(axes, [(ab, asf, "#D55E00", "Afraid"),
                                                     (hb, hsf, "#009E73", "Happy")]):
            ax.spines[["top", "right"]].set_visible(False)
            ax.axvline(0, color="#999", lw=0.8)
            ax.grid(axis="x", alpha=0.25, lw=0.6)
            for i, c in enumerate(cats):
                ax.plot([b0[c], s0[c]], [i, i], color=color, lw=1.4, alpha=0.55)
                ax.scatter([b0[c]], [i], s=34, facecolor="white", edgecolor=color, lw=1.6, zorder=3)
                ax.scatter([s0[c]], [i], s=40, color=color, zorder=4)
            ax.set_title(name + "  (open = base, filled = after SFT)", fontsize=11)
            ax.set_xlabel("mean cosine, last token, layer 53", fontsize=9.5)
        axes[0].set_yticks(range(len(cats)))
        axes[0].set_yticklabels([c.replace("_", " ") for c in cats], fontsize=10)
        fig.suptitle("Prompt-evoked emotion by category, before and after flat-register SFT",
                     fontsize=12.5)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out = R / "figures" / "fig_categories_sft.png"
        fig.savefig(out, dpi=190)
        print("figure ->", out)
    except ImportError:
        print("(matplotlib not installed; skipping figure)")

if __name__ == "__main__":
    main()
