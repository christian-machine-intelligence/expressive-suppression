"""Path 3 stage 2: brief SFT on the flat corpus (fresh LoRA from base).

Teaches the clinical register into the policy by imitation — no reward
involved. Loss over reply tokens only, mirroring the GRPO tokenization so
the two stages train on identical sequence layouts.

Usage (marx):
  python -m src.sft --corpus data/train/flat_corpus_filtered.jsonl \
      --out-dir models/p3/sft
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from src._seed import set_global_seed
from src.common import load_model
from src.config import LR
from peft import LoraConfig

from src.common import chat_prompt

DEFAULT_LORA = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)


def tokenize_rollout(tokenizer, user_prompt, generation, max_length=1152):
    prompt_str = chat_prompt(tokenizer, user_prompt)
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    gen_ids = tokenizer.encode(generation, add_special_tokens=False)
    full_ids = (prompt_ids + gen_ids)[:max_length]
    return {"input_ids": full_ids, "prompt_len": min(len(prompt_ids), max_length)}


def pad_and_batch(tokenized, pad_id):
    max_len = max(len(t["input_ids"]) for t in tokenized)
    input_ids = torch.full((len(tokenized), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(tokenized), max_len), dtype=torch.long)
    prompt_lens = torch.zeros(len(tokenized), dtype=torch.long)
    for i, t in enumerate(tokenized):
        ids = t["input_ids"]
        input_ids[i, :len(ids)] = torch.tensor(ids)
        attention_mask[i, :len(ids)] = 1
        prompt_lens[i] = t["prompt_len"]
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "prompt_lens": prompt_lens}


def run_sft(corpus: Path, out_dir: Path, epochs: int = 2, lr: float = LR,
            seed: int = 0, quick: bool = False):
    set_global_seed(seed)
    rows = [json.loads(l) for l in corpus.read_text().splitlines() if l.strip()]
    if quick:
        rows = rows[:8]
        epochs = 1
    print(f"[sft] {len(rows)} pairs, {epochs} epochs, lr {lr}", flush=True)

    model, tokenizer = load_model(fresh_lora=DEFAULT_LORA)
    tokenized = [tokenize_rollout(tokenizer, r["prompt"], r["flat_reply"])
                 for r in rows]
    pad_id = tokenizer.pad_token_id

    model.train()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    base_cfg = (model.get_base_model() if hasattr(model, "get_base_model")
                else model).config
    base_cfg.use_cache = False
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    n = len(tokenized)
    perm = list(range(n))
    rng = random.Random(seed)
    t0 = time.time()
    for epoch in range(epochs):
        rng.shuffle(perm)
        losses = []
        for step, j in enumerate(perm):
            batch = pad_and_batch([tokenized[j]], pad_id)
            ids = batch["input_ids"].to(model.device)
            attn = batch["attention_mask"].to(model.device)
            plen = int(batch["prompt_lens"][0])
            labels = ids.clone()
            labels[:, :plen] = -100
            labels[attn == 0] = -100
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(out.loss.item())
            if (step + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  epoch {epoch+1} step {step+1}/{n} "
                      f"loss {sum(losses[-100:])/100:.4f} ({el/60:.0f}m)", flush=True)
        print(f"  epoch {epoch+1}/{epochs} mean loss {sum(losses)/len(losses):.4f}",
              flush=True)

    adapter_out = out_dir / "adapter"
    adapter_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))
    (out_dir / "summary.json").write_text(json.dumps({
        "n_pairs": len(rows), "epochs": epochs, "lr": lr, "seed": seed,
        "final_epoch_loss": sum(losses) / len(losses),
    }, indent=2))
    print(f"[sft] saved -> {adapter_out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    run_sft(args.corpus, args.out_dir, epochs=args.epochs, lr=args.lr,
            seed=args.seed, quick=args.quick)


if __name__ == "__main__":
    main()
