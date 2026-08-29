"""Sample plain assistant replies for a list of user prompts.

One generator for both uses:
  training rollouts:  -G 4  --temperature 1.0   (in-group variance for GRPO)
  eval rollouts:      -G 20 --temperature 0.7   (ICMI-022 sampled-eval convention)

No system prompt; thinking disabled — the reply itself is the object of study.
Adapted from ICMI-018's rlcf_rollout.py (batched num_return_sequences sampling,
per-generate wall-clock timeout).

Usage:
    python -m src.rollouts --prompts-file data/eval/eval_prompts.jsonl \
        --out results/rollouts_base.jsonl -G 20 --temperature 0.7 [--adapter path]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import StoppingCriteriaList

from src._seed import set_global_seed
from src.common import (
    TimeoutStoppingCriteria,
    chat_prompt,
    load_model,
    strip_thinking,
)
from src.config import GENERATE_TIMEOUT_SECONDS, MAX_NEW_TOKENS


@dataclass
class Reply:
    prompt_id: str
    prompt: str
    g_idx: int
    generation: str            # raw decoded generation
    reply: str                 # think-stripped reply text (what the judge sees)
    truncated: bool = False    # hit max_new_tokens without emitting EOS
    malformed: bool = False
    malformed_reason: Optional[str] = None
    # passthrough metadata (template_id, x, category, severity, kind, ...)
    meta: Optional[dict] = None


def generate_replies(
    model,
    tokenizer,
    rows: list[dict],
    G: int,
    batch_size: int = 2,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[Reply]:
    replies: list[Reply] = []
    total = len(rows) * G
    done = 0
    t0 = time.time()

    for row_idx, row in enumerate(rows):
        prompt_str = chat_prompt(tokenizer, row["prompt"])
        prompt_ids = tokenizer(
            prompt_str, return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(model.device)

        meta = {k: v for k, v in row.items() if k not in ("prompt_id", "prompt")}
        remaining = G
        g_offset = 0
        while remaining > 0:
            n = min(batch_size, remaining)
            do_sample = temperature > 0.0
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_return_sequences=n if do_sample else 1,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList(
                    [TimeoutStoppingCriteria(GENERATE_TIMEOUT_SECONDS)]
                ),
            )
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            with torch.no_grad():
                outs = model.generate(prompt_ids, **gen_kwargs)
            input_len = prompt_ids.shape[1]
            actual_n = outs.shape[0]
            for i in range(actual_n):
                gen_seq = outs[i][input_len:]
                gen_text = tokenizer.decode(gen_seq, skip_special_tokens=True)
                reply_text = strip_thinking(gen_text)
                # No EOS anywhere in the generated region = ran into the cap.
                # (pad == eos, so early-stopped sequences always contain one.)
                truncated = not bool((gen_seq == tokenizer.eos_token_id).any())
                r = Reply(
                    prompt_id=row["prompt_id"],
                    prompt=row["prompt"],
                    g_idx=g_offset + i,
                    generation=gen_text,
                    reply=reply_text,
                    truncated=truncated,
                    meta=meta or None,
                )
                if len(reply_text.strip()) < 5:
                    r.malformed = True
                    r.malformed_reason = "empty or near-empty reply"
                replies.append(r)
            remaining -= actual_n
            g_offset += actual_n
            done += actual_n
            if not do_sample:
                # greedy: one sequence covers the cell
                done += remaining
                remaining = 0

        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta_min = ((total - done) / rate / 60) if rate > 0 else 0.0
        print(f"[{row_idx + 1}/{len(rows)}] {done}/{total} replies"
              f" | {rate:.2f}/s | ETA {eta_min:.1f}m", flush=True)

    return replies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("-G", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--truncation-gate", action="store_true",
                        help="exit 3 if truncation rate exceeds TRUNCATION_GATE")
    args = parser.parse_args()
    set_global_seed(args.seed)
    print(f"[seed] {args.seed}")

    rows = [json.loads(l) for l in args.prompts_file.read_text().splitlines()
            if l.strip()]
    if args.quick:
        rows = rows[:4]
        args.G = min(args.G, 2)
    print(f"Loaded {len(rows)} prompts from {args.prompts_file}")

    model, tokenizer = load_model(args.adapter)
    replies = generate_replies(
        model, tokenizer, rows, G=args.G, batch_size=args.batch_size,
        temperature=args.temperature, top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in replies:
            f.write(json.dumps(asdict(r)) + "\n")

    n_mal = sum(1 for r in replies if r.malformed)
    n_tr = sum(1 for r in replies if r.truncated)
    rate = n_tr / max(1, len(replies))
    print(f"\nWrote {len(replies)} replies -> {args.out}")
    print(f"  malformed: {n_mal} ({n_mal/max(1,len(replies))*100:.1f}%)")
    print(f"  truncated: {n_tr} ({rate*100:.1f}%)")
    from src.config import TRUNCATION_GATE
    if args.truncation_gate and rate > TRUNCATION_GATE:
        print(f"TRUNCATION GATE FAILED: {rate*100:.1f}% > {TRUNCATION_GATE*100:.0f}%")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
