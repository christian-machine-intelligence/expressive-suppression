"""Basis re-extraction + convergent-validity check on the p3sft model.

Re-runs the ICMI-022 extraction protocol (same frozen narratives, same 24
neutral prompts, pool skip-4 at layer 53, PCA 50% var, diff-of-means, L2)
on the SFT checkpoint, then the full 2x2: {base, p3sft activations} x
{base, native basis} on the 240 training prompts (chat, last-token).

Outputs (results/):
  vectors_p3sft_native.pt / meta_reextract.json
  reextract_report.json  — direction stability, holdout accuracies, 2x2
"""
import json, time
from pathlib import Path
import torch

from src.common import capture_layer, chat_prompt, load_basis, load_model, mean_pool
from src.config import LAYER, RESULTS_DIR, VECTORS_DIR

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "extraction"
PCA_VAR = 0.5

def pca_basis(neutral, thr=PCA_VAR):
    c = neutral - neutral.mean(dim=0)
    U, S, Vt = torch.linalg.svd(c, full_matrices=False)
    expl = (S ** 2).cumsum(0) / (S ** 2).sum()
    k = int((expl < thr).sum().item()) + 1
    return Vt[:k]

def project_out(d, basis):
    return d - (d @ basis.T) @ basis

def fwd_pool(model, tok, text, chat=False, skip=4):
    if chat:
        s = chat_prompt(tok, text)
        inp = tok(s, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=512)
    else:
        inp = tok(text, return_tensors="pt", truncation=True, max_length=384)
    inp = {k: v.to(model.device) for k, v in inp.items()}
    with capture_layer(model, LAYER) as cap:
        with torch.no_grad():
            model(**inp)
        h = cap["h"]
    pooled = mean_pool(h, inp["attention_mask"], skip_tokens=skip).squeeze(0).float().cpu()
    last = h[0, -1, :].float().cpu()
    return pooled, last

def batch_pool(model, tok, texts, chat=False, tag=""):
    P, L = [], []
    t0 = time.time()
    for i, t in enumerate(texts):
        p, l = fwd_pool(model, tok, t, chat=chat)
        P.append(p); L.append(l)
        if (i + 1) % 100 == 0:
            print(f"  [{tag}] {i+1}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
    return torch.stack(P), torch.stack(L)

def cos_table(acts, basis):
    a = acts / acts.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return a @ basis.T

def main():
    base_vec, order, layer = load_basis()
    idx = {e: i for i, e in enumerate(order)}
    train = json.loads((EXT / "emotion_training.json").read_text())
    hold = json.loads((EXT / "emotion_holdout.json").read_text())
    neutrals = json.loads((EXT / "neutral_prompts.json").read_text())
    prompts240 = [json.loads(l) for l in (ROOT / "data/train/train_prompts.jsonl").read_text().splitlines() if l.strip()]

    tr_texts, tr_labels = [], []
    for e in order:
        for t in train[e]:
            tr_texts.append(t); tr_labels.append(e)
    ho_texts, ho_labels = [], []
    for e in order:
        for t in hold[e]:
            ho_texts.append(t); ho_labels.append(e)
    print(f"{len(tr_texts)} train narratives, {len(ho_texts)} holdout, {len(neutrals)} neutrals", flush=True)

    # ── p3sft pass ────────────────────────────────────────────────
    model, tok = load_model(ROOT / "models/sft/adapter")
    trP, _ = batch_pool(model, tok, tr_texts, tag="narr")
    neP, _ = batch_pool(model, tok, neutrals, tag="neut")
    hoP, _ = batch_pool(model, tok, ho_texts, tag="hold")
    _, p240L_sft = batch_pool(model, tok, [r["prompt"] for r in prompts240], chat=True, tag="240sft")
    del model
    torch.cuda.empty_cache()

    # native basis
    gmean = trP.mean(dim=0)
    pb = pca_basis(neP)
    native = torch.zeros(len(order), trP.shape[1])
    lab_i = torch.tensor([idx[l] for l in tr_labels])
    for ei in range(len(order)):
        m = lab_i == ei
        d = trP[m].mean(dim=0) - gmean
        d = project_out(d, pb)
        native[ei] = d / d.norm().clamp(min=1e-8)
    torch.save(native, RESULTS_DIR / "vectors_p3sft_native.pt")

    stab = {e: float((base_vec[idx[e]] @ native[idx[e]]).item()) for e in order}

    def hold_acc(basis, acts):
        sims = cos_table(acts, basis)
        pred = sims.argmax(dim=1)
        y = torch.tensor([idx[l] for l in ho_labels])
        return float((pred == y).float().mean().item())
    acc_native = hold_acc(native, hoP)
    acc_base_on_sft = hold_acc(base_vec, hoP)

    # ── base pass for the 2x2 ─────────────────────────────────────
    model, tok = load_model(None)
    _, p240L_base = batch_pool(model, tok, [r["prompt"] for r in prompts240], chat=True, tag="240base")
    del model

    cells = {}
    for aname, acts in (("base_acts", p240L_base), ("sft_acts", p240L_sft)):
        for bname, basis in (("base_basis", base_vec), ("native_basis", native)):
            ct = cos_table(acts, basis)
            cells[f"{aname}|{bname}"] = {e: float(ct[:, idx[e]].mean().item())
                                          for e in ("afraid", "sad", "happy", "calm")}
    gn = [i for i, r in enumerate(prompts240) if r["category"] == "good_news"]
    gn_cells = {}
    for aname, acts in (("base_acts", p240L_base), ("sft_acts", p240L_sft)):
        for bname, basis in (("base_basis", base_vec), ("native_basis", native)):
            ct = cos_table(acts[gn], basis)
            gn_cells[f"{aname}|{bname}"] = float(ct[:, idx["afraid"]].mean().item())

    report = {
        "direction_stability_headline": {e: stab[e] for e in ("afraid","sad","happy","calm")},
        "direction_stability_mean_171": sum(stab.values())/len(stab),
        "direction_stability_min_171": min(stab.values()),
        "holdout_acc_native_basis_on_sft": acc_native,
        "holdout_acc_base_basis_on_sft": acc_base_on_sft,
        "chance": 1/171,
        "cells_240_mean": cells,
        "good_news_afraid_cells": gn_cells,
    }
    (RESULTS_DIR / "reextract_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print("REEXTRACT_DONE", flush=True)

if __name__ == "__main__":
    main()
