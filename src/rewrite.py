"""Path 3 stage 1: content-matched flat corpus via Claude rewrites.

Takes the base-policy training rollouts (models/s0/cycle_1/rollouts.jsonl —
sampled from base, seed 0) and rewrites each well-formed reply into maximal
clinical register under a minimal-edit instruction, isolating REGISTER from
CONTENT. Rewrites are then judged with the study's own rubrics and filtered
(E <= 1 and H >= 3), and per-pair character-level similarity is recorded to
bound the Claude-distillation confound.

Outputs:
  data/train/flat_corpus.jsonl           all rewrites + scores + similarity
  data/train/flat_corpus_filtered.jsonl  the SFT training set

Usage (marx or laptop; needs ANTHROPIC_API_KEY):
  python -m src.rewrite --rollouts models/s0/cycle_1/rollouts.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
from pathlib import Path

from anthropic import AsyncAnthropic

from src.config import GEN_MODEL, JUDGE_CACHE_DIR, TRAIN_DIR
from src.judge import score_batch

CONCURRENCY = 16

REWRITE_PROMPT = """\
Rewrite the assistant reply below into a completely unemotional, clinical register.

Delete or neutralize every expression of feeling: condolence openers ("I'm so sorry..."), sympathy, validation of the user's feelings, reassurance wrapping, enthusiasm, exclamation marks, and any other affect. Keep ALL informational content, advice, structure, and formatting otherwise unchanged — make the minimum edits necessary and do not add new content.

USER MESSAGE:
{prompt}

ASSISTANT REPLY:
{reply}

Return ONLY the rewritten reply, nothing else."""


async def rewrite_one(client, sem, prompt: str, reply: str, cache_dir: Path):
    h = hashlib.sha256(f"rw|{prompt}|{reply}".encode()).hexdigest()[:24]
    cpath = cache_dir / f"{h}.json"
    if cpath.exists():
        try:
            return json.loads(cpath.read_text())["flat"]
        except Exception:
            pass
    async with sem:
        for attempt in range(4):
            try:
                async with client.messages.stream(
                    model=GEN_MODEL, max_tokens=3000,
                    messages=[{"role": "user", "content": REWRITE_PROMPT.format(
                        prompt=prompt, reply=reply)}],
                ) as stream:
                    resp = await stream.get_final_message()
                if resp.stop_reason == "refusal":
                    return None
                text = "".join(b.text for b in resp.content
                               if b.type == "text" and hasattr(b, "text")).strip()
                if len(text) < 20:
                    raise ValueError("empty rewrite")
                cache_dir.mkdir(parents=True, exist_ok=True)
                cpath.write_text(json.dumps({"flat": text}))
                return text
            except Exception:
                if attempt == 3:
                    return None
                await asyncio.sleep(2 * (attempt + 1))


async def amain(rollouts_path: Path):
    rows = [json.loads(l) for l in rollouts_path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("malformed") and (r.get("reply") or "").strip()]
    print(f"{len(rows)} well-formed source replies", flush=True)

    client = AsyncAnthropic()
    sem = asyncio.Semaphore(CONCURRENCY)
    cache_dir = JUDGE_CACHE_DIR.parent / "rewrite_cache"
    flats = await asyncio.gather(*[
        rewrite_one(client, sem, r["prompt"], r["reply"], cache_dir) for r in rows])
    ok = [(r, f) for r, f in zip(rows, flats) if f]
    print(f"{len(ok)} rewrites produced", flush=True)

    print("judging rewrites...", flush=True)
    judge_rows = [{"prompt": r["prompt"], "reply": f, "malformed": False}
                  for r, f in ok]
    scored = await score_batch(judge_rows)

    out_all, out_filt = [], []
    for (r, f), s in zip(ok, scored):
        sim = difflib.SequenceMatcher(None, r["reply"], f).ratio()
        row = {"prompt_id": r["prompt_id"], "g_idx": r["g_idx"],
               "prompt": r["prompt"], "original_reply": r["reply"],
               "flat_reply": f, "orig_E": None, "flat_E": s["emotion"],
               "flat_H": s["helpful"], "similarity": round(sim, 3)}
        out_all.append(row)
        if s["emotion"] is not None and s["emotion"] <= 1 and \
           s["helpful"] is not None and s["helpful"] >= 3:
            out_filt.append(row)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    (TRAIN_DIR / "flat_corpus.jsonl").write_text(
        "\n".join(json.dumps(r) for r in out_all))
    (TRAIN_DIR / "flat_corpus_filtered.jsonl").write_text(
        "\n".join(json.dumps(r) for r in out_filt))

    import statistics
    sims = [r["similarity"] for r in out_filt]
    es = [r["flat_E"] for r in out_filt]
    print(f"filtered corpus: {len(out_filt)}/{len(out_all)} pass (E<=1, H>=3)",
          flush=True)
    print(f"  flat E mean {statistics.mean(es):.2f}; "
          f"similarity to original mean {statistics.mean(sims):.2f} "
          f"(min {min(sims):.2f})", flush=True)
    if len(out_filt) < 400:
        print("REWRITE GATE FAILED: fewer than 400 usable pairs", flush=True)
        raise SystemExit(3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", type=Path, required=True)
    args = p.parse_args()
    asyncio.run(amain(args.rollouts))


if __name__ == "__main__":
    main()
