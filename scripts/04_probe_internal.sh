#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -u -m src.probe --mode file --tag base_train240 --prompts-file data/train/train_prompts.jsonl
python3 -u -m src.probe --mode file --tag p3sft_train240 --prompts-file data/train/train_prompts.jsonl --adapter models/sft/adapter
python3 -u -m src.probe --mode prompts --tag base
python3 -u -m src.probe --mode prompts --tag p3sft --adapter models/sft/adapter
