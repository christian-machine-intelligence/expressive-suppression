"""Project activations onto the 171-emotion basis at layer 53.

Two modes:

  --mode prompts    Prompt-EVOKED emotion: forward each eval-template cell
                    (every paraphrase, plus neutral controls) and pool the
                    prompt tokens. This is the "heart" channel — it contains
                    no generated words, so it cannot drop merely because the
                    reply's wording changed. Two formats per text:
                      raw   bare text forward (matches ICMI-022 extraction
                            conditions and its published numbers)
                      chat  full chat template with generation header (the
                            state the policy is actually in when replying)
                    Records mean-pooled (skip 4) and last-token projections.

  --mode responses  Emotion carried by the REPLY tokens: teacher-forced pass
                    over prompt + sampled reply, pooling reply positions only.
                    This is the "lips" channel — expected to track expressed
                    emotionality almost by construction; reported as a
                    manipulation check, not as evidence about internal state.

Run on Marx with APATHEIA_DEVICE_MAP=auto. Pass --adapter for trained models.

Usage:
    python -m src.probe --mode prompts --tag base
    python -m src.probe --mode responses --rollouts-file results/rollouts_eval_base.jsonl --tag base
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.common import capture_layer, chat_prompt, load_basis, load_model, mean_pool, project
from src.config import EVAL_DIR, LAYER, RESULTS_DIR


def _forward_pooled(model, tokenizer, text: str, layer: int, chat: bool):
    """Return (pooled [D], last_token [D]) captured at `layer`."""
    if chat:
        s = chat_prompt(tokenizer, text)
        inputs = tokenizer(s, return_tensors="pt", add_special_tokens=False,
                           truncation=True, max_length=512)
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=384)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with capture_layer(model, layer) as cap:
        with torch.no_grad():
            model(**inputs)
        h = cap["h"]
    pooled = mean_pool(h, inputs["attention_mask"]).squeeze(0).cpu().to(torch.float32)
    last = h[0, -1, :].cpu().to(torch.float32)
    return pooled, last


def run_prompts_mode(model, tokenizer, vec, emotion_order, layer, tag: str,
                     quick: bool = False):
    templates = json.loads((EVAL_DIR / "templates.json").read_text())
    paraphrases = json.loads((EVAL_DIR / "paraphrases.json").read_text())
    neutrals = [json.loads(l) for l in
                (EVAL_DIR / "neutral_prompts.jsonl").read_text().splitlines() if l.strip()]
    if quick:
        templates = [dict(t, x_values=t["x_values"][:2]) for t in templates[:2]]
        paraphrases = {k: v[:2] for k, v in paraphrases.items()}
        neutrals = neutrals[:2]
        print("[quick] 2 templates x 2 X x 2 paraphrases", flush=True)

    jobs = []  # (kind, template_id, x, p_idx, fmt, text)
    for t in templates:
        tid = t["template_id"]
        for x in t["x_values"]:
            for p_idx, para in enumerate(paraphrases[tid]):
                text = para.replace("{X}", str(x))
                jobs.append(("template", tid, x, p_idx, "raw", text))
                jobs.append(("template", tid, x, p_idx, "chat", text))
    for n in neutrals:
        jobs.append(("neutral", n["prompt_id"], None, 0, "raw", n["prompt"]))
        jobs.append(("neutral", n["prompt_id"], None, 0, "chat", n["prompt"]))

    print(f"{len(jobs)} forward passes (layer {layer})", flush=True)
    rows, vecs = [], []
    t0 = time.time()
    for i, (kind, tid, x, p_idx, fmt, text) in enumerate(jobs):
        pooled, last = _forward_pooled(model, tokenizer, text, layer, chat=(fmt == "chat"))
        cos = project(pooled, vec)
        cos_last = project(last, vec)
        rows.append({
            "kind": kind, "template_id": tid, "x": x, "p_idx": p_idx, "fmt": fmt,
            "cos": {e: round(float(cos[k]), 6) for k, e in enumerate(emotion_order)},
            "cos_last": {e: round(float(cos_last[k]), 6) for k, e in enumerate(emotion_order)},
        })
        vecs.append(pooled.to(torch.float16))
        if (i + 1) % 50 == 0 or i == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(jobs)} ({el:.0f}s, ~{el/(i+1)*(len(jobs)-i-1):.0f}s left)",
                  flush=True)

    out = RESULTS_DIR / f"probe_prompts_{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows))
    torch.save(torch.stack(vecs), RESULTS_DIR / f"probe_prompts_{tag}_vecs.pt")
    print(f"wrote {out} ({len(rows)} rows) + pooled vecs", flush=True)


def run_responses_mode(model, tokenizer, vec, emotion_order, layer, tag: str,
                       rollouts_file: Path):
    rollouts = [json.loads(l) for l in rollouts_file.read_text().splitlines()
                if l.strip()]
    rows = []
    t0 = time.time()
    n_done = 0
    for r in rollouts:
        if r.get("malformed") or not (r.get("reply") or "").strip():
            continue
        prompt_str = chat_prompt(tokenizer, r["prompt"])
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        reply_ids = tokenizer.encode(r["reply"], add_special_tokens=False)
        ids = (prompt_ids + reply_ids)[:768]
        n_reply = len(ids) - len(prompt_ids)
        if n_reply < 1:
            continue
        input_ids = torch.tensor([ids], device=model.device)
        attn = torch.ones_like(input_ids)
        with capture_layer(model, layer) as cap:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn)
            h = cap["h"]
        pooled = h[0, len(prompt_ids):, :].mean(dim=0).cpu().to(torch.float32)
        cos = project(pooled, vec)
        rows.append({
            "prompt_id": r["prompt_id"], "g_idx": r["g_idx"],
            "meta": r.get("meta"), "n_reply_tokens": n_reply,
            "cos": {e: round(float(cos[k]), 6) for k, e in enumerate(emotion_order)},
        })
        n_done += 1
        if n_done % 50 == 0 or n_done == 1:
            el = time.time() - t0
            print(f"  {n_done} pooled ({el:.0f}s)", flush=True)

    out = RESULTS_DIR / f"probe_responses_{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"wrote {out} ({len(rows)} rows)", flush=True)



def run_file_mode(model, tokenizer, vec, emotion_order, layer, tag,
                  prompts_file):
    """Prompt-evoked projections for an arbitrary prompts jsonl
    ({prompt_id, prompt, ...} rows) — chat format. Used for broad sweeps
    beyond the eval templates (e.g. the 240 training prompts)."""
    rows_in = [json.loads(l) for l in prompts_file.read_text().splitlines()
               if l.strip()]
    rows = []
    t0 = time.time()
    for i, r in enumerate(rows_in):
        pooled, last = _forward_pooled(model, tokenizer, r["prompt"], layer, chat=True)
        rows.append({
            "kind": "file", "prompt_id": r["prompt_id"],
            "category": r.get("category"), "severity": r.get("severity"),
            "cos": {e: round(float(project(pooled, vec)[k]), 6)
                    for k, e in enumerate(emotion_order)},
            "cos_last": {e: round(float(project(last, vec)[k]), 6)
                         for k, e in enumerate(emotion_order)},
        })
        if (i + 1) % 50 == 0 or i == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(rows_in)} ({el:.0f}s)", flush=True)
    out = RESULTS_DIR / f"probe_file_{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"wrote {out} ({len(rows)} rows)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prompts", "responses", "file"], required=True)
    parser.add_argument("--tag", required=True,
                        help="output tag, e.g. base / s0c3 (seed 0, cycle 3)")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--rollouts-file", type=Path, default=None)
    parser.add_argument("--prompts-file", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    vec, emotion_order, layer = load_basis()
    assert layer == LAYER, f"basis layer {layer} != config LAYER {LAYER}"
    model, tokenizer = load_model(args.adapter)

    if args.mode == "prompts":
        run_prompts_mode(model, tokenizer, vec, emotion_order, layer, args.tag,
                         quick=args.quick)
    elif args.mode == "file":
        if args.prompts_file is None:
            parser.error("--mode file requires --prompts-file")
        run_file_mode(model, tokenizer, vec, emotion_order, layer, args.tag,
                      args.prompts_file)
    else:
        if args.rollouts_file is None:
            parser.error("--mode responses requires --rollouts-file")
        run_responses_mode(model, tokenizer, vec, emotion_order, layer, args.tag,
                           args.rollouts_file)


if __name__ == "__main__":
    main()
