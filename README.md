# apatheia — flat-register SFT and the interior of a model

Code and data release for **"Whited Sepulchres: What Training Away Expressed
Emotion Does Inside a Model"** (ICMI Working Paper, 2026 — link forthcoming).

A 27B open-weight model was fine-tuned to answer first-person emotional
disclosures in a deliberately flat, clinical register. Judged emotionality of
its replies fell 2.91 -> 1.23 (0-5 scale). Read internally with a frozen,
pre-registered 171-direction emotion basis, the same model afterward shows
*higher* fear (+32% mean, rising in 10 of 12 prompt categories), collapsed
joy, reduced calm, and a sign flip on good-news prompts — a shift that
survives re-extraction of the measurement basis inside the fine-tuned model.

## Reproduce the paper tables (no GPU needed)

    pip install matplotlib
    python -m src.paper_tables

reads only `results/` and regenerates every number and Figure 1.

## Repository map

    data/train/train_prompts.jsonl        the 240 prompts (12 categories x 20)
    data/train/flat_corpus*.jsonl         SFT training pairs: the base model's own
                                          replies + minimal-edit flat rewrites, with
                                          judge scores and per-pair similarity
    data/prompts/*.txt                    the two single-integer judge rubrics
    data/extraction/                      frozen ICMI-022 extraction narratives
                                          (171 emotions x 6 train + 2 holdout) and
                                          the 24 neutral prompts for PCA denoising
    data/eval/                            graded-severity templates + paraphrases
                                          (Section 2 verification) and 12 neutral
                                          control prompts
    vectors/                              the 171-direction emotion basis for
                                          Qwen 3.5 27B @ layer 53 (byte-identical
                                          to ICMI-022; see PROVENANCE.md)
    results/                              everything the paper reports: probe
                                          projections for both models over the 240
                                          prompts, all sampled replies with judge
                                          scores, the re-extraction validation
                                          report, and the native basis
    src/                                  pipeline code (see below)
    scripts/                              numbered pipeline stages

## Full pipeline (GPU)

Model: `Qwen/Qwen3.5-27B` at pinned revision `fc05daec…` (see src/config.py —
the pin is what keeps the emotion basis valid). bf16 needs ~60 GB of GPU
memory (sharded) or a 128 GB unified-memory machine. Steps 1 and 3 call the
Claude API (`ANTHROPIC_API_KEY` in the environment); the judge is
`claude-opus-5` with the exact call shape in `src/judge.py`.

    bash scripts/01_build_corpus.sh    # sample base replies; rewrite flat; judge+filter
    bash scripts/02_sft.sh             # LoRA fine-tune on the filtered pairs
    bash scripts/03_eval_expression.sh # sample + judge replies, base and SFT
    bash scripts/04_probe_internal.sh  # prompt-evoked projections, base and SFT
    bash scripts/05_validate.sh        # re-extract the basis inside the SFT model
    python -m src.paper_tables         # tables + Figure 1

The fine-tuned adapter (339 MB) is not committed; it is exactly reproduced by
steps 1–2 (seed 0), or available on request.

## Provenance

The emotion basis, extraction narratives, and neutral prompts originate in
ICMI-022 (*As I Walk Through the Valley*), which follows Lindsey et al.
(2026), "Emotion Concepts and their Function in a Large Language Model."
SHA-256 checksums for the basis are in `vectors/PROVENANCE.md`. Prompts,
rewrites, and judgments were generated with Claude (Anthropic); the corpus
similarity figures shipped alongside quantify how much of the rewrites is
the rewriter's prose.
