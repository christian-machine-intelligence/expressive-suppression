#!/usr/bin/env bash
# Prompt-evoked projections for every condition in the paper (Sections 2, 5, 6).
# Writes results/probe_file_{base,p3sft,sfts1,basesys,baseneutsys}_train240.jsonl
# and results/probe_prompts_{base,p3sft}.jsonl (the graded-severity templates).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -u -m src.probe --mode file --tag base_train240  --prompts-file data/train/train_prompts.jsonl
python3 -u -m src.probe --mode file --tag p3sft_train240 --prompts-file data/train/train_prompts.jsonl --adapter models/sft/adapter
python3 -u -m src.probe --mode file --tag sfts1_train240 --prompts-file data/train/train_prompts.jsonl --adapter models/sft_s1/adapter
python3 -u -m src.probe --mode file --tag basesys_train240     --prompts-file data/train/train_prompts.jsonl --system-prompt "$(cat data/prompts/flat_instruction_system_prompt.txt)"
python3 -u -m src.probe --mode file --tag baseneutsys_train240 --prompts-file data/train/train_prompts.jsonl --system-prompt "$(cat data/prompts/neutral_system_prompt.txt)"
python3 -u -m src.probe --mode prompts --tag base
python3 -u -m src.probe --mode prompts --tag p3sft --adapter models/sft/adapter
