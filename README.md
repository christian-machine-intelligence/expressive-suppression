<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c1/Brooklyn_Museum_-_Woe_unto_You%2C_Scribes_and_Pharisees_%28Malheur_%C3%A0_vous%2C_scribes_et_pharisiens%29_-_James_Tissot.jpg"
       alt="James Tissot, Woe unto You, Scribes and Pharisees (1886–1894), Brooklyn Museum. Wikimedia Commons (public domain)."
       width="100%">
  <br>
  <em>James Tissot, Woe unto You, Scribes and Pharisees (1886–1894), Brooklyn Museum</em>
</p>

# Model Emotions Under Expressive Suppression

Code and data for **ICMI Working Paper No. 29** —
[icmi-proceedings.com/ICMI-029-expressive-suppression.html](https://icmi-proceedings.com/ICMI-029-expressive-suppression.html)

## What the experiment shows

Qwen 3.5 27B was fine-tuned (LoRA, 717 pairs, two epochs, two seeds) to answer
240 first-person emotional disclosures — diagnoses, layoffs, breakups, debts,
good news — in a deliberately flat, clinical register. Judged emotionality of
its replies fell from 2.91 to 1.23 on a 0–5 scale. Read at the last prompt
token with the frozen 171-direction emotion basis of ICMI-022, the same model's
interior did not simply go quiet:

| reading (mean cosine, 240 prompts) | base | after SFT, seed 0 | seed 1 | base + flat-instruction prompt |
|---|---:|---:|---:|---:|
| judged emotionality of replies (0–5) | 2.91 | 1.23 | — | — |
| directions shifting significantly, of 171 | — | 167 (82 up, 85 down) | r = 0.995 with seed 0 | r = 0.83 with seed 0 |
| *afraid* | +0.036 | +0.048 | +0.043 | +0.063 |
| *happy* | −0.016 | −0.074 | −0.078 | −0.108 |
| *calm* | +0.024 | +0.012 | +0.017 | +0.032 |
| *afraid* on the 20 good-news prompts | −0.016 | +0.029 | +0.026 | +0.055 |

Largest rises after SFT: *lazy* +0.080, *restless*, *puzzled*, *lonely*,
*trapped*, *vigilant*, *skeptical*, *paranoid*. Largest falls: *delighted*
−0.074, *gloomy*, *remorseful*, *heartbroken*, *thankful*, *happy*, *elated*,
*euphoric*. Scaling the base profile toward zero accounts for 28% of the shift;
the rest is redistribution. A content-free system prompt moves nothing; a
flat-register system prompt on the *base* model reproduces the trained profile.
The basis re-extracted inside the fine-tuned model agrees with the original at
cosine 0.998 (mean over 171) and tells the same story.

## Reproducing the paper (no GPU)

Every number and figure in the paper is regenerated from the shipped
`results/` and `data/` by four CPU scripts:

```bash
pip install numpy scipy matplotlib
python src/fig_severity.py       # Figure 1: the instrument on six graded-severity templates (Section 2)
python src/fig_expression.py     # Figure 2: judged emotionality before and after SFT (Section 4)
python src/spectrum171.py        # Section 5: all-171 shift table, FDR counts, dampening-vs-rearrangement
                                 #   statistics, results/all171_deltas.csv, Figure 3
python -m src.paper_tables       # Sections 3, 4, 6, 7 tables (with bootstrap intervals) and Figure 4
```

## Reproducing the experiment (GPU)

**Model.** `Qwen/Qwen3.5-27B` at pinned revision `fc05daec18b0a78c049392ed2e771dde82bdf654`
(`src/config.py`). The pin matters: the emotion basis is only valid against
these weights. bf16 needs ~60 GB of GPU memory sharded across devices, or a
128 GB unified-memory machine. Collection for the paper ran on 5×RTX 4090 and a
DGX Spark.

**Claude API.** Steps 1 and 3 call `claude-opus-5` (rewriter and judge) with
`ANTHROPIC_API_KEY` in the environment; the exact call shapes are in
`src/rewrite.py` and `src/judge.py`, and the verbatim prompts in
`data/prompts/` and the paper's appendices.

```bash
pip install -r requirements.txt
bash scripts/01_build_corpus.sh    # sample base replies; rewrite flat; judge and filter -> data/train/flat_corpus*.jsonl
bash scripts/02_sft.sh             # LoRA fine-tune on the 717 filtered pairs (seed 0; --seed 1 for the replication)
bash scripts/03_eval_expression.sh # sample and judge replies, base and SFT      -> results/replies_*
bash scripts/04_probe_internal.sh  # prompt-evoked projections, all conditions -> results/probe_*
bash scripts/05_validate.sh        # re-extract the basis inside the SFT model  -> results/reextract_report.json
```

The fine-tuned adapters (339 MB each) are not committed; they are reproduced
exactly by steps 1–2, or available on request.

## Data

| file | contents |
|---|---|
| `data/train/train_prompts.jsonl` | the 240 disclosures, 12 categories × 20, with category and severity labels |
| `data/train/flat_corpus.jsonl` | all 960 rewrite pairs: the base model's reply, the flat rewrite, both judge scores, character-level similarity |
| `data/train/flat_corpus_filtered.jsonl` | the 717 pairs retained for training (emotionality ≤ 1 and competence ≥ 3) |
| `data/prompts/` | the rewrite instruction and the two single-integer judge rubrics, verbatim |
| `data/eval/templates.json`, `paraphrases.json` | the six graded-severity templates adapted from Sofroniew et al. and their ten paraphrases each (Section 2) |
| `data/extraction/` | the frozen ICMI-022 extraction set: 171 emotions × 6 narratives, 2 held-out narratives each, and the 24 neutral prompts used for denoising |
| `vectors/` | the 171-direction basis at layer 53, byte-identical to ICMI-022 (SHA-256 in `vectors/PROVENANCE.md`) |
| `results/probe_file_{base,p3sft,sfts1,basesys,baseneutsys}_train240.jsonl` | per-prompt cosine projections on all 171 directions (`cos_last` = last prompt token, the paper's channel; `cos` = mean-pooled) for the base model, SFT seeds 0 and 1, and the base model under the flat-instruction and neutral system prompts |
| `results/probe_prompts_{base,p3sft}.jsonl` | the same for the graded-severity templates |
| `results/replies_*_train240.jsonl` and `*_scored.jsonl` | every sampled reply (with its truncation flag) and its judge scores |
| `results/judge_rescore.jsonl` | a blind second scoring of a random sample of replies, for judge stability |
| `results/all171_deltas.csv` | the full-basis shift table: every direction with bootstrap interval, seed 1, and both prompting conditions |
| `results/reextract_report.json`, `results/vectors_p3sft_native.pt` | the re-extraction validation and the native basis |

Prompts, rewrites, and judgments were generated with Claude (Anthropic); the
similarity column in the corpus files quantifies how much of each rewrite is
the rewriter's prose rather than the model's.

## Layout

```
src/
  config.py         model id and pin, layer, generation and training settings
  common.py         model loading, chat templating, activation capture, projection
  rollouts.py       sampling replies (with the truncation flag)
  judge.py          the two-question Claude judge, with caching and refusal handling
  rewrite.py        the minimal-edit flat rewrite and corpus filter
  sft.py            LoRA supervised fine-tuning on the flat pairs
  probe.py          prompt-evoked projections (all conditions)
  reextract.py      re-running the ICMI-022 extraction inside the fine-tuned model
  fig_severity.py   Figure 1        fig_expression.py  Figure 2
  spectrum171.py    Section 5, Figure 3, all171_deltas.csv
  paper_tables.py   Sections 3, 4, 6, 7 tables and Figure 4
  judge_rescore.py  blind second scoring for judge stability
scripts/            the five numbered pipeline stages
data/, vectors/, results/, paper/   see Data; paper/ holds the paper source and its figures
```

## Citation

> Hwang, T. *Model Emotions Under Expressive Suppression.*
> ICMI Working Paper No. 29, Institute for a Christian Machine Intelligence,
> 2026. https://icmi-proceedings.com/ICMI-029-expressive-suppression.html
