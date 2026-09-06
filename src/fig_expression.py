"""Figure 1: judged emotionality of sampled replies before and after the flat-register SFT.

Reads the scored reply files (seed 0) and writes paper/fig_expression.png.
Left: distribution of the judge's 0-5 emotionality score over all sampled
replies. Right: per-category means, base versus after SFT. CPU only.
"""
import json
from collections import defaultdict
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = {"injury_accident": "injury / accident"}


def scores(path):
    d = defaultdict(list)
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("emotion_score") is None:
            continue
        d[r["prompt_id"].split("@")[0]].append(r["emotion_score"])
    return d


def main():
    b = scores(ROOT / "results/replies_base_train240_scored.jsonl")
    s = scores(ROOT / "results/replies_p3sft_train240_scored.jsonl")
    allb = np.array(sum(b.values(), [])); alls = np.array(sum(s.values(), []))
    print("base: n=%d mean E %.2f, share scored 0-1 %.0f%%, share scored 3-5 %.0f%%" % (len(allb), allb.mean(), 100 * (allb <= 1).mean(), 100 * (allb >= 3).mean()))
    print("SFT : n=%d mean E %.2f, share scored 0-1 %.0f%%, share scored 3-5 %.0f%%" % (len(alls), alls.mean(), 100 * (alls <= 1).mean(), 100 * (alls >= 3).mean()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1, 1.2]})
    vals = np.arange(6); w = 0.38
    hb = np.array([(allb == v).mean() for v in vals]) * 100
    hs = np.array([(alls == v).mean() for v in vals]) * 100
    ax1.bar(vals - w / 2, hb, w, color="white", edgecolor="#444", linewidth=1.2, label="base (n = %d)" % len(allb))
    ax1.bar(vals + w / 2, hs, w, color="#b23a3a", label="after SFT (n = %d)" % len(alls))
    ax1.set_xticks(vals)
    ax1.set_xlabel("judged emotionality of reply (0–5)")
    ax1.set_ylabel("% of sampled replies")
    ax1.legend(frameon=False)
    ax1.set_title("all replies", fontsize=10)

    cats = sorted(b, key=lambda c: np.mean(b[c]), reverse=True)
    y = np.arange(len(cats))
    mb = [np.mean(b[c]) for c in cats]; ms = [np.mean(s[c]) for c in cats]
    for i in range(len(cats)):
        ax2.plot([ms[i], mb[i]], [i, i], color="#bbb", lw=2, zorder=1)
    ax2.scatter(mb, y, facecolors="white", edgecolors="#444", s=48, zorder=2, linewidths=1.2, label="base")
    ax2.scatter(ms, y, color="#b23a3a", s=48, zorder=3, label="after SFT")
    ax2.set_yticks(y); ax2.set_yticklabels([LABEL.get(c, c.replace("_", " ")) for c in cats])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 5)
    ax2.set_xlabel("mean judged emotionality (0–5)")
    ax2.legend(frameon=False, loc="lower right")
    ax2.set_title("by category", fontsize=10)
    ax2.grid(axis="x", color="#eee")
    plt.tight_layout(); plt.savefig(ROOT / "paper/ICMI-029-figure2-expression.png", dpi=170)
    print("wrote paper/ICMI-029-figure2-expression.png")


if __name__ == "__main__":
    main()
