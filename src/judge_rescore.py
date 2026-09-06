"""Blind second scoring of a random sample of the shipped replies, for judge stability.

    python -m src.judge_rescore --n 100   # 100 base + 100 SFT replies -> results/judge_rescore.jsonl

Needs ANTHROPIC_API_KEY. Re-scores with the cache off and reports exact and
within-one-point agreement with the shipped first-pass scores.
"""
import argparse
import asyncio
import json
import random
from pathlib import Path

from .judge import score_batch

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="replies per model")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    sample = []
    for tag in ["base", "p3sft"]:
        raw = {(r["prompt_id"], r["g_idx"]): r for r in rows(R / ("replies_%s_train240.jsonl" % tag))}
        sc = [r for r in rows(R / ("replies_%s_train240_scored.jsonl" % tag)) if r.get("emotion_score") is not None]
        for r in rng.sample(sc, min(a.n, len(sc))):
            rr = raw[(r["prompt_id"], r["g_idx"])]
            sample.append({"model": tag, "prompt_id": r["prompt_id"], "g_idx": r["g_idx"], "prompt": rr["prompt"], "reply": rr["reply"],
                           "malformed": bool(rr.get("malformed")), "emotion_first": r["emotion_score"], "helpful_first": r.get("helpful_score")})
    print("re-scoring %d replies with the cache off..." % len(sample), flush=True)
    rescored = asyncio.run(score_batch(sample, cache_dir=None))
    out = R / "judge_rescore.jsonl"
    with open(out, "w") as f:
        for s, rs in zip(sample, rescored):
            row = {k: s[k] for k in ["model", "prompt_id", "g_idx", "emotion_first", "helpful_first"]}
            row["emotion_second"], row["helpful_second"] = rs["emotion"], rs["helpful"]
            f.write(json.dumps(row) + "\n")
    for rubric in ["emotion", "helpful"]:
        pairs = [(s[rubric + "_first"], rs[rubric]) for s, rs in zip(sample, rescored) if s.get(rubric + "_first") is not None and rs[rubric] is not None]
        print("%s: n=%d, exact agreement %.2f, within one point %.2f" % (rubric, len(pairs), sum(x == y for x, y in pairs) / len(pairs), sum(abs(x - y) <= 1 for x, y in pairs) / len(pairs)))
    print("wrote", out)


if __name__ == "__main__":
    main()
