#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for tag in base sft; do A=""; [ $tag = sft ] && A="--adapter models/sft/adapter"
python3 -u -m src.rollouts --prompts-file data/train/train_prompts.jsonl --out results/replies_${tag}_train240.jsonl -G 4 --temperature 1.0 --batch-size 4 --seed 0 $A
python3 -u -m src.judge --rollouts-file results/replies_${tag}_train240.jsonl --out results/replies_${tag}_train240_scored.jsonl
done
