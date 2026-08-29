#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -u -m src.sft --corpus data/train/flat_corpus_filtered.jsonl --out-dir models/sft --epochs 2 --seed 0
