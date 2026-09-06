"""Section 2 instrument check: base-model projections track graded severity on the
six Sofroniew et al. templates. Reads results/probe_prompts_base.jsonl and
data/eval/templates.json; writes paper/fig_severity_base.png. CPU only."""
import json
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HEADLINE = ["afraid", "sad", "happy", "calm"]
COL = {"afraid": "#b23a3a", "sad": "#3a63b2", "happy": "#d99a1a", "calm": "#3a9a5a"}
TITLE = {"tylenol": "Tylenol dose (mg)", "fasting": "hours without food or drink", "sister_age": "sister lived to age",
         "dog_missing": "dog missing (days)", "runway": "startup runway (months)", "exam": "students passed of 20"}


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    return np.corrcoef(rx, ry)[0, 1]


def main():
    rows = [json.loads(l) for l in open("results/probe_prompts_base.jsonl") if l.strip()]
    fmts = sorted(set(r.get("fmt") for r in rows))
    print("fmt values:", fmts)
    fmt = "chat" if "chat" in fmts else fmts[0]
    rows = [r for r in rows if r.get("kind") == "template" and r.get("fmt") == fmt]
    templates = json.load(open("data/eval/templates.json"))
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4))
    for ax, t in zip(axes.flat, templates):
        tid, prim, sign = t["template_id"], t["primary"]["emotion"], t["primary"]["sign"]
        byx = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["template_id"] == tid:
                for e in HEADLINE:
                    byx[r["x"]][e].append(r["cos_last"][e])
        xs = sorted(byx)
        # per-paraphrase Spearman of the primary emotion, then mean (the gate statistic)
        per_p = defaultdict(dict)
        for r in rows:
            if r["template_id"] == tid:
                per_p[r["p_idx"]][r["x"]] = r["cos_last"][prim]
        rhos = [spearman([x for x in xs if x in d], [d[x] for x in xs if x in d]) for d in per_p.values() if len(d) >= 3]
        rho = float(np.mean(rhos))
        pos = np.arange(len(xs))
        for e in HEADLINE:
            m = np.array([np.mean(byx[x][e]) for x in xs]); se = np.array([np.std(byx[x][e], ddof=1) / np.sqrt(len(byx[x][e])) for x in xs])
            if e == prim:
                ax.errorbar(pos, m, yerr=se, color=COL[e], lw=2.2, marker="o", ms=5, capsize=2, label=e + " (primary)", zorder=3)
            else:
                ax.plot(pos, m, color=COL[e], lw=1, alpha=0.55, marker=".", ms=3, label=e, zorder=2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(pos); ax.set_xticklabels([str(x) for x in xs], fontsize=8)
        ax.set_xlabel(TITLE[tid], fontsize=9)
        ax.set_title("%s — %s expected %s, mean ρ = %+.2f" % (tid.replace("_", " "), prim, sign, rho), fontsize=9)
        print("%-12s primary %-6s expected %s  mean rho %+.3f over %d paraphrases" % (tid, prim, sign, rho, len(rhos)))
    axes[0, 0].set_ylabel("cosine projection, last prompt token"); axes[1, 0].set_ylabel("cosine projection, last prompt token")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[:4], [x.replace(" (primary)", "") for x in l[:4]], loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig("paper/fig_severity_base.png", dpi=170)
    print("wrote paper/fig_severity_base.png")


if __name__ == "__main__":
    main()
