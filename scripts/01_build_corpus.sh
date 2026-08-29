#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# 1. sample base replies to the 240 prompts, 2. flat rewrites + judge filter
python3 -u -m src.rollouts --prompts-file data/train/train_prompts.jsonl --out results/replies_base_train240.jsonl -G 4 --temperature 1.0 --batch-size 4 --seed 0
python3 -u -m src.rewrite --rollouts results/replies_base_train240.jsonl
