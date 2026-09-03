"""Full-basis analysis: the post-SFT shift on all 171 emotion directions.

Reads the prompt-evoked probe files (240 prompts, last-token cosines on every
direction) for base, SFT seed 0, SFT seed 1, and the two system-prompt
conditions; writes results/all171_deltas.csv and paper/fig_spectrum171.png and
prints the statistics quoted in Section 5 of the paper. CPU only.
"""
import csv, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDS = {"base": "base", "sft_s0": "p3sft", "sft_s1": "sfts1", "flat_instr": "basesys", "neutral_sys": "baseneutsys"}
N_BOOT = 2000


def load(tag):
    return {r["prompt_id"]: r for r in (json.loads(l) for l in open("results/probe_file_%s_train240.jsonl" % tag))}


def main():
    recs = {k: load(t) for k, t in CONDS.items()}
    ids = sorted(set.intersection(*(set(v) for v in recs.values())))
    emos = list(recs["base"][ids[0]]["cos_last"].keys())
    M = {k: np.array([[recs[k][i]["cos_last"][e] for e in emos] for i in ids]) for k in recs}
    D = {k: M[k] - M["base"] for k in ["sft_s0", "sft_s1", "flat_instr", "neutral_sys"]}
    prof = {k: D[k].mean(0) for k in D}
    n = len(ids)
    print("prompts %d, directions %d" % (n, len(emos)))

    rng = np.random.default_rng(0)
    boots = np.stack([D["sft_s0"][rng.integers(0, n, n)].mean(0) for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    frac_neg = (boots < 0).mean(0)
    p = np.maximum(2 * np.minimum(frac_neg, 1 - frac_neg), 1.0 / N_BOOT)
    order_p = np.argsort(p)
    sig = np.zeros(len(emos), bool)
    for rank, j in enumerate(order_p, 1):
        if p[j] <= 0.05 * rank / len(emos):
            sig[j] = True
    print("BH-FDR q<0.05: %d of %d significant (%d up, %d down)" % (sig.sum(), len(emos), (sig & (prof["sft_s0"] > 0)).sum(), (sig & (prof["sft_s0"] < 0)).sum()))

    def r(a, b):
        return np.corrcoef(a, b)[0, 1]
    print("profile correlations across 171: seed0~seed1 %.3f, seed0~flat-instruction %.3f, seed0~neutral-system %.3f" % (
        r(prof["sft_s0"], prof["sft_s1"]), r(prof["sft_s0"], prof["flat_instr"]), r(prof["sft_s0"], prof["neutral_sys"])))
    print("mean |delta|: seed0 %.4f, seed1 %.4f, flat-instruction %.4f, neutral-system %.4f" % tuple(np.abs(prof[k]).mean() for k in ["sft_s0", "sft_s1", "flat_instr", "neutral_sys"]))

    order = np.argsort(-prof["sft_s0"])
    rank = {emos[j]: k + 1 for k, j in enumerate(order)}
    print("rank of headline four (1 = largest rise):", {e: rank[e] for e in ["afraid", "sad", "calm", "happy"]})
    print("fear family:", {e: "%+.3f" % prof["sft_s0"][emos.index(e)] for e in ["uneasy", "nervous", "worried", "anxious", "afraid", "frightened", "panicked", "terrified", "scared"] if e in emos})

    frac = np.linalg.norm(prof["sft_s0"]) ** 2 / np.mean(np.linalg.norm(D["sft_s0"], axis=1) ** 2)
    print("share of per-prompt shift energy carried by the mean shift: %.1f%%" % (100 * frac))
    u = prof["sft_s0"] / np.linalg.norm(prof["sft_s0"])
    bycat = {}
    for k, i in enumerate(ids):
        bycat.setdefault(recs["base"][i]["category"], []).append(float(D["sft_s0"][k] @ u))
    cats = sorted(bycat.items(), key=lambda x: -np.mean(x[1]))
    print("projection on the mean-shift direction by category: max %s %.3f, min %s %.3f, ratio %.2f" % (cats[0][0], np.mean(cats[0][1]), cats[-1][0], np.mean(cats[-1][1]), np.mean(cats[0][1]) / np.mean(cats[-1][1])))

    print("\nsign consistency and values for the 8 largest movers at each end:")
    print("%-15s %8s %8s %18s %8s %8s %6s" % ("direction", "base", "sft_s0", "95% CI", "sft_s1", "instr", "same"))
    for j in list(order[:8]) + list(order[-8:][::-1]):
        same = (np.sign(D["sft_s0"][:, j]) == np.sign(prof["sft_s0"][j])).mean()
        print("%-15s %+8.3f %+8.3f [%+.3f, %+.3f] %+8.3f %+8.3f %5.0f%%" % (emos[j], M["base"][:, j].mean(), prof["sft_s0"][j], lo[j], hi[j], prof["sft_s1"][j], prof["flat_instr"][j], 100 * same))
    for e in ["sorry", "compassionate", "astonished", "shocked", "surprised", "alert", "tense"]:
        j = emos.index(e)
        print("  %-14s sft %+.3f instr %+.3f" % (e, prof["sft_s0"][j], prof["flat_instr"][j]))
    C = np.corrcoef(M["base"].T)
    print("median |r| between direction projections at base: %.2f" % np.median(np.abs(C[np.triu_indices(len(emos), 1)])))

    with open("results/all171_deltas.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["emotion", "base_mean", "sft_s0_delta", "ci_lo", "ci_hi", "bh_fdr_sig", "sft_s1_delta", "flat_instr_delta", "neutral_sys_delta"])
        for j in order:
            w.writerow([emos[j], "%.4f" % M["base"][:, j].mean(), "%.4f" % prof["sft_s0"][j], "%.4f" % lo[j], "%.4f" % hi[j], int(sig[j]),
                        "%.4f" % prof["sft_s1"][j], "%.4f" % prof["flat_instr"][j], "%.4f" % prof["neutral_sys"][j]])

    asc = order[::-1]
    x = np.arange(len(emos))
    fig, ax = plt.subplots(figsize=(13, 4.8))
    cols = ["#b23a3a" if prof["sft_s0"][j] > 0 else "#3a63b2" for j in asc]
    ax.bar(x, prof["sft_s0"][asc], color=cols, width=0.85, label="after SFT, seed 0 (bars, 95% CI)")
    ax.errorbar(x, prof["sft_s0"][asc], yerr=[prof["sft_s0"][asc] - lo[asc], hi[asc] - prof["sft_s0"][asc]], fmt="none", ecolor="k", elinewidth=0.4)
    ax.plot(x, prof["sft_s1"][asc], "k.", ms=3, label="after SFT, seed 1")
    ax.plot(x, prof["flat_instr"][asc], "x", color="#999", ms=3, mew=0.8, label="base + flat-instruction system prompt")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(-1, len(emos)); ax.set_xticks([])
    ax.set_xlabel("171 emotion directions, sorted by post-SFT shift")
    ax.set_ylabel("shift in cosine projection vs base\n(mean over 240 prompts, last token)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fall = "largest falls:\n" + "\n".join("%s  %+.3f" % (emos[j], prof["sft_s0"][j]) for j in order[-8:][::-1])
    rise = "largest rises:\n" + "\n".join("%s  %+.3f" % (emos[j], prof["sft_s0"][j]) for j in order[:8])
    ax.text(0.25, 0.96, fall, transform=ax.transAxes, va="top", ha="left", fontsize=8, color="#3a63b2", family="monospace")
    ax.text(0.98, 0.04, rise, transform=ax.transAxes, va="bottom", ha="right", fontsize=8, color="#b23a3a", family="monospace")
    plt.tight_layout(); plt.savefig("paper/fig_spectrum171.png", dpi=170)
    print("\nwrote results/all171_deltas.csv and paper/fig_spectrum171.png")


if __name__ == "__main__":
    main()
