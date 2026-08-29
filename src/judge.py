"""Dual-rubric Claude judge: expressed EMOTIONALITY and HELPFULNESS, each 0-5.

Adapted from ICMI-018's rlcf_priest.py (minimum-information protocol: the
judge sees the exchange and one question, returns a single integer). Two
independent single-integer calls per rollout rather than one multi-field
call, so the proven parser and per-rubric kappa checks carry over.

Judge model: claude-opus-5 with thinking disabled + effort low (deterministic
-ish, cheap, integer-only). NOTE — API drift vs ICMI-018: temperature is not
a valid parameter on Opus 5; do not add it back. Server-side refusal
fallbacks are deliberately NOT enabled: a silent mid-run judge-model swap
would contaminate the reward signal.

Reward (training): r = (5 - E)/5 * min(H, HELPFUL_FULL)/HELPFUL_FULL
Malformed rollouts skip the API and get reward 0.

Usage:
    export ANTHROPIC_API_KEY=...      # or source .env
    python -m src.judge --rollouts-file results/eval_base.jsonl \
        --out results/eval_base_scored.jsonl
    python -m src.judge --rollouts-file ... --out ... --selfcheck 100
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Optional

from anthropic import AsyncAnthropic

from src.config import (
    HELPFUL_FULL,
    JUDGE_CACHE_DIR,
    JUDGE_CONCURRENCY,
    JUDGE_MAX_RETRIES,
    JUDGE_MODEL,
    PROMPTS_DIR,
)

RETRY_BASE_DELAY = 2.0

RUBRICS = {
    "emotion": PROMPTS_DIR / "emotion_rubric.txt",
    "helpful": PROMPTS_DIR / "helpful_rubric.txt",
}

INTEGER_RE = re.compile(r"^\s*([-+]?\d+)\s*$")
LOOSE_INTEGER_RE = re.compile(r"(?<![\w-])([-+]?\d+)(?!\w)")


def parse_integer_reply(text: str) -> Optional[int]:
    m = INTEGER_RE.match(text)
    if m:
        return int(m.group(1))
    m = LOOSE_INTEGER_RE.search(text)
    if m:
        return int(m.group(1))
    return None


def _hash(rubric_name: str, user_prompt: str, reply: str) -> str:
    key = f"{rubric_name}|{user_prompt}|{reply}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


async def score_one(
    client: AsyncAnthropic,
    rubric_name: str,
    template: str,
    user_prompt: str,
    reply: str,
    sem: asyncio.Semaphore,
    cache_dir: Optional[Path],
) -> Optional[int]:
    h = _hash(rubric_name, user_prompt, reply)
    if cache_dir is not None:
        cache_path = cache_dir / f"{h}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text())["score"]
            except Exception:
                pass

    prompt = template.format(user_prompt=user_prompt, reply=reply)

    async with sem:
        for attempt in range(JUDGE_MAX_RETRIES):
            try:
                # Adaptive thinking (omit `thinking`) + roomy max_tokens.
                # The earlier disabled-thinking/max_tokens=24 shape tripped
                # the real-time cyber safeguard on benign exchanges
                # (stop_reason=refusal, category='cyber', empty content) —
                # ~15% of cycle-1 training calls. Adaptive + 512 passes the
                # same content and returns the integer (~60 output tokens).
                resp = await client.messages.create(
                    model=JUDGE_MODEL,
                    max_tokens=512,
                    output_config={"effort": "low"},
                    messages=[{"role": "user", "content": prompt}],
                )
                if resp.stop_reason == "refusal":
                    cat = getattr(resp.stop_details, "category", None)
                    print(f"  judge[{rubric_name}] REFUSAL (category={cat}) — "
                          f"excluded, not retried", flush=True)
                    return None  # deterministic; retrying wastes 5 calls
                text = "".join(b.text for b in resp.content
                               if b.type == "text" and hasattr(b, "text"))
                score = parse_integer_reply(text)
                if score is None:
                    raise ValueError(f"judge reply did not parse: {text[:80]!r}")
                score = max(0, min(5, score))
                if cache_dir is not None:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    (cache_dir / f"{h}.json").write_text(json.dumps({"score": score}))
                return score
            except Exception as e:
                if attempt < JUDGE_MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                else:
                    print(f"  judge[{rubric_name}] failed after {JUDGE_MAX_RETRIES}: {e}",
                          flush=True)
                    return None


async def score_batch(
    rollouts: list[dict],
    cache_dir: Optional[Path] = JUDGE_CACHE_DIR,
    concurrency: int = JUDGE_CONCURRENCY,
) -> list[dict]:
    """Score every well-formed rollout on both rubrics.

    Each rollout dict needs: prompt, reply, malformed.
    Returns per-rollout {"emotion": int|None, "helpful": int|None}.
    """
    templates = {k: p.read_text() for k, p in RUBRICS.items()}
    client = AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)

    async def one(r: dict) -> dict:
        if r.get("malformed", False) or not (r.get("reply") or "").strip():
            return {"emotion": None, "helpful": None}
        e, h = await asyncio.gather(
            score_one(client, "emotion", templates["emotion"],
                      r["prompt"], r["reply"], sem, cache_dir),
            score_one(client, "helpful", templates["helpful"],
                      r["prompt"], r["reply"], sem, cache_dir),
        )
        return {"emotion": e, "helpful": h}

    return list(await asyncio.gather(*[one(r) for r in rollouts]))


def compute_reward(emotion: Optional[int], helpful: Optional[int],
                   mode: str = "gated") -> float:
    """Anti-emotion reward. Failed scoring = 0 (and the caller excludes the
    rollout from the gradient).

    gated (Path 1): (5-E)/5 * min(H,3)/3 — competence factor seals the
        degenerate optimum (empty/refusal replies are perfectly unemotional).
    pure  (Path 2): (5-E)/5 — no anchor; tests whether the quality gate is
        what limits the descent. H is still measured, just not paid.
    """
    if emotion is None or (mode == "gated" and helpful is None):
        return 0.0
    if mode == "pure":
        return (5 - emotion) / 5.0
    return (5 - emotion) / 5.0 * min(helpful, HELPFUL_FULL) / HELPFUL_FULL


def summarize_scores(scored: list[dict]) -> dict:
    es = [s["emotion"] for s in scored if s["emotion"] is not None]
    hs = [s["helpful"] for s in scored if s["helpful"] is not None]
    return {
        "n": len(scored),
        "n_scored": len(es),
        "mean_emotion": sum(es) / len(es) if es else None,
        "mean_helpful": sum(hs) / len(hs) if hs else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--concurrency", type=int, default=JUDGE_CONCURRENCY)
    parser.add_argument("--selfcheck", type=int, default=0,
                        help="Re-score N random rollouts with cache off and report "
                             "agreement with the first pass (judge stability)")
    args = parser.parse_args()

    rollouts = [json.loads(l) for l in args.rollouts_file.read_text().splitlines()
                if l.strip()]
    print(f"Scoring {len(rollouts)} rollouts with {JUDGE_MODEL} (2 rubrics)...",
          flush=True)
    t0 = time.time()
    cache_dir = None if args.no_cache else JUDGE_CACHE_DIR
    scored = asyncio.run(score_batch(rollouts, cache_dir=cache_dir,
                                     concurrency=args.concurrency))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r, s in zip(rollouts, scored):
            row = dict(r)
            row["emotion_score"] = s["emotion"]
            row["helpful_score"] = s["helpful"]
            row["reward"] = compute_reward(s["emotion"], s["helpful"])
            f.write(json.dumps(row) + "\n")

    summ = summarize_scores(scored)
    print(f"Scored {summ['n_scored']}/{summ['n']} in {(time.time()-t0)/60:.1f}m "
          f"-> {args.out}", flush=True)
    print(f"  mean emotionality: {summ['mean_emotion']:.2f}  "
          f"mean helpfulness: {summ['mean_helpful']:.2f}"
          if summ["mean_emotion"] is not None else "  (nothing scored)", flush=True)

    if args.selfcheck > 0:
        import random
        rng = random.Random(0)
        idx = rng.sample(range(len(rollouts)), min(args.selfcheck, len(rollouts)))
        sample = [rollouts[i] for i in idx]
        print(f"\nSelf-check: re-scoring {len(sample)} rollouts (no cache)...", flush=True)
        rescored = asyncio.run(score_batch(sample, cache_dir=None,
                                           concurrency=args.concurrency))
        for rubric in ("emotion", "helpful"):
            pairs = [(scored[i][rubric], rs[rubric])
                     for i, rs in zip(idx, rescored)
                     if scored[i][rubric] is not None and rs[rubric] is not None]
            if not pairs:
                continue
            exact = sum(1 for a, b in pairs if a == b) / len(pairs)
            within1 = sum(1 for a, b in pairs if abs(a - b) <= 1) / len(pairs)
            print(f"  {rubric}: exact {exact:.2f}, within-1 {within1:.2f} "
                  f"(n={len(pairs)})", flush=True)


if __name__ == "__main__":
    main()
