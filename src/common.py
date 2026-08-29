"""Shared model + activation utilities.

Pillaged from ICMI-018 (model loading, chat handling, timeout criteria) and
ICMI-022 (decoder discovery, mean pooling, layer capture, basis projection).

Device map: env APATHEIA_DEVICE_MAP wins ("auto" on Marx to shard the 27B
across GPUs), otherwise {"": "cuda:0"} (Sparks, unified memory).
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch

from src.config import (
    EMOTION_META_PATH,
    EMOTION_VECTORS_PATH,
    MODEL_ID,
    MODEL_REVISION,
    POOL_SKIP,
)


# ─────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────

def load_model(adapter_path: Optional[Path] = None, trainable_adapter: bool = False,
               fresh_lora: Optional[object] = None):
    """Load Qwen 3.5 27B bf16 at the pinned revision; optionally attach a LoRA.

    fresh_lora: a peft LoraConfig — initialize a new trainable adapter.
    adapter_path: load an existing adapter (trainable if trainable_adapter).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dm = os.environ.get("APATHEIA_DEVICE_MAP", "").strip() or {"": "cuda:0"}
    print(f"Loading {MODEL_ID}@{MODEL_REVISION[:8]} bf16 device_map={dm} ...", flush=True)
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16,
            device_map=dm, trust_remote_code=True,
        )
    except TypeError:  # older transformers uses torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16,
            device_map=dm, trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if fresh_lora is not None:
        from peft import get_peft_model
        model = get_peft_model(model, fresh_lora)
        model.print_trainable_parameters()
    elif adapter_path is not None:
        from peft import PeftModel
        print(f"Applying adapter: {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, str(adapter_path),
                                          is_trainable=trainable_adapter)
        if trainable_adapter:
            model.print_trainable_parameters()

    if not trainable_adapter and fresh_lora is None:
        model.eval()
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────
# Chat formatting — thinking DISABLED throughout this study
# (the plain reply is the behavioral object; no system prompt, so the
# policy's register is entirely its own)
# ─────────────────────────────────────────────────────────────────────

def chat_prompt(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def strip_thinking(text: str) -> str:
    """Defensive: if the template still yields a think block, keep only the reply."""
    close = text.find("</think>")
    if close >= 0:
        return text[close + len("</think>"):].strip()
    if text.lstrip().startswith("<think>"):
        return ""  # opened a think block and never closed it (truncation)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────
# Activation capture at one decoder layer (ICMI-022 conventions)
# ─────────────────────────────────────────────────────────────────────

def find_decoder_layers(model):
    """Locate the decoder block list across HF topologies, unwrapping peft.

    Under peft, hooks must attach to the base model (rlcf-gula lesson)."""
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    candidates = [
        lambda x: x.model.layers,
        lambda x: x.model.language_model.layers,
        lambda x: x.model.model.layers,
    ]
    for getter in candidates:
        try:
            layers = getter(m)
            if hasattr(layers, "__len__") and len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue
    raise RuntimeError("Could not find decoder layers")


@contextmanager
def capture_layer(model, layer_idx: int):
    """Context manager: captured['h'] holds the [B, T, D] hidden state of
    decoder[layer_idx] after each forward inside the block."""
    decoder = find_decoder_layers(model)
    target = decoder[layer_idx]
    captured: dict = {}

    def hook(module, ins, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output

    handle = target.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor,
              skip_tokens: int = POOL_SKIP) -> torch.Tensor:
    """Mean-pool [B, T, D] over T, masking padding and the first skip_tokens."""
    mask = attention_mask.clone().float().to(hidden.device)
    if skip_tokens > 0:
        mask[:, :skip_tokens] = 0.0
    mask = mask.unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)


# ─────────────────────────────────────────────────────────────────────
# 171-emotion basis
# ─────────────────────────────────────────────────────────────────────

def load_basis():
    """Return (vec [171, D] float32 row-normalized, emotion_order, layer)."""
    meta = json.loads(EMOTION_META_PATH.read_text())
    vec = torch.load(EMOTION_VECTORS_PATH, weights_only=True).to(torch.float32)
    if vec.shape[0] != meta["num_emotions"]:
        raise ValueError(f"basis rows {vec.shape[0]} != meta {meta['num_emotions']}")
    vec = vec / vec.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return vec, meta["emotion_order"], meta["best_layer"]


def project(pooled: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Cosine similarity of a pooled activation [D] against the basis [171, D]."""
    p = pooled.to(torch.float32)
    p = p / p.norm().clamp(min=1e-8)
    return p @ vec.T


# ─────────────────────────────────────────────────────────────────────
# Generation timeout (mandatory on long unattended runs — infra lesson)
# ─────────────────────────────────────────────────────────────────────

from transformers import StoppingCriteria  # noqa: E402


class TimeoutStoppingCriteria(StoppingCriteria):
    def __init__(self, max_seconds: float):
        self.max_seconds = max_seconds
        self.start = time.time()

    def __call__(self, input_ids, scores, **kwargs):
        return (time.time() - self.start) > self.max_seconds
